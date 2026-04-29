from __future__ import annotations

import inspect
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from data.s5_cot.opd_hf import (
    cached_teacher_token_probs_hf,
    evaluate_clean_ce_loss_hf,
    evaluate_saved_clean_s5_metrics_hf,
    rollout_student_hf,
)
from data.s5_cot.prompt_bank import load_prompt_bank, select_train_subset
from data.s5_cot.semantic_key_noise import (
    SEMANTIC_KEY_NOISE_LAW,
    semantic_key_noise_config_from_obj,
)
from data.synthetic.random_suffix_noise import (
    RANDOM_SUFFIX_AFTER_ERROR_LAW,
    random_suffix_noise_config_from_obj,
    validate_random_suffix_applies_to_task,
)
from data.synthetic.target_spans import (
    canonical_target_len,
    print_prompt_bank_target_span_diagnostic,
)
from hf_checkpoint import (
    apply_nanogpt_bias_policy,
    build_hf_model_from_nanogpt_args,
    load_nanogpt_checkpoint,
    load_nanogpt_checkpoint_as_hf,
    set_hf_causal_lm_loss,
)
from nanogpt.methods.student_prefix import (
    FixedPromptCycle,
    extract_answer_logits,
    gather_action_log_probs,
    sample_teacher_actions,
    teacher_forward_kl,
)
from nanogpt.trainers.configs import OpdHfConfig
from nanogpt.trainers.runtime import (
    build_autocast_context,
    build_grad_scaler,
    capture_rng_state,
    get_linear_warmup_lr,
    resolve_device,
    resolve_dtype,
    restore_rng_state,
)
from nanogpt.trainers.wandb import maybe_init_wandb
from nanogpt.utils.repo import write_launch_metadata


HF_MODEL_DIRNAME = "hf_model"
TRAINING_STATE_NAME = "training_state.pt"


def resolve_student_rollout_temperature(cfg_or_meta) -> float:
    rollout_temperature = getattr(cfg_or_meta, "student_rollout_temperature", None)
    if rollout_temperature is None and isinstance(cfg_or_meta, dict):
        rollout_temperature = cfg_or_meta.get("student_rollout_temperature")
    if rollout_temperature is None:
        rollout_temperature = getattr(cfg_or_meta, "student_temperature", None)
    if rollout_temperature is None and isinstance(cfg_or_meta, dict):
        rollout_temperature = cfg_or_meta.get("student_temperature")
    if rollout_temperature is None:
        return 0.0
    return float(rollout_temperature)


def format_student_rollout_tag(*, rollout_temperature: float, student_temperature: float) -> str:
    rollout_tag = "greedy" if rollout_temperature == 0 else f"t{rollout_temperature}".replace(".", "p")
    student_tag = "greedy" if student_temperature == 0 else f"t{student_temperature}".replace(".", "p")
    if rollout_temperature == student_temperature:
        return student_tag
    return f"roll{rollout_tag}-stud{student_tag}"


def validate_config(cfg: OpdHfConfig) -> None:
    student_rollout_temperature = resolve_student_rollout_temperature(cfg)
    teacher_law = getattr(cfg, "teacher_law", "distributional_noise")
    if teacher_law == SEMANTIC_KEY_NOISE_LAW:
        semantic_key_noise_config_from_obj(getattr(cfg, "semantic_key_noise", None))
    if teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        random_suffix_config = random_suffix_noise_config_from_obj(
            getattr(cfg, "random_suffix_noise", None)
        )
        validate_random_suffix_applies_to_task(random_suffix_config, task_name="s5")
    if cfg.objective != "reverse_kl_tm" and cfg.student_temperature <= 0:
        raise ValueError(
            "student_temperature must be > 0 for forward-KL objectives because "
            "the loss is defined against the temperature-adjusted student policy."
        )
    if student_rollout_temperature < 0:
        raise ValueError("student_rollout_temperature must be non-negative.")
    if getattr(cfg, "compile", False) and not hasattr(torch, "compile"):
        raise ValueError("--compile requires a PyTorch build with torch.compile support.")


def save_run_metadata(
    out_dir: Path,
    *,
    subset_indices: torch.Tensor,
    metadata: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(subset_indices.to(device="cpu", dtype=torch.long), out_dir / "subset_indices.pt")
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def validate_resume_metadata(out_dir: Path, metadata: dict[str, object]) -> None:
    meta_path = out_dir / "run_meta.json"
    if not meta_path.exists():
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    keys = [
        "backend",
        "teacher_checkpoint",
        "prompt_bank_dir",
        "subset_size",
        "eta",
        "teacher_law",
    ]
    if metadata.get("teacher_law") == SEMANTIC_KEY_NOISE_LAW:
        keys.append("semantic_key_noise")
    if metadata.get("teacher_law") == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        keys.append("random_suffix_noise")
    keys.extend([
        "objective",
        "student_temperature",
        "student_rollout_temperature",
        "shuffle_prompts",
        "seed",
    ])
    for key in keys:
        saved_value = saved.get(key)
        current_value = metadata.get(key)
        if key == "student_rollout_temperature":
            if saved_value is None:
                saved_value = resolve_student_rollout_temperature(saved)
            if current_value is None:
                current_value = resolve_student_rollout_temperature(metadata)
        if saved_value != current_value:
            raise ValueError(
                f"Resume mismatch for {key}: saved={saved_value!r} "
                f"current={current_value!r}"
            )


def mark_complete(out_dir: Path, iter_num: int) -> None:
    with open(out_dir / "completed.txt", "w", encoding="utf-8") as f:
        f.write(f"iter_num={iter_num}\n")


def configure_hf_optimizer(
    model: GPT2LMHeadModel,
    *,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
) -> torch.optim.Optimizer:
    param_dict = {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    decay_params = [param for _, param in param_dict.items() if param.dim() >= 2]
    nodecay_params = [param for _, param in param_dict.items() if param.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(param.numel() for param in decay_params)
    num_nodecay_params = sum(param.numel() for param in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = {"fused": True} if use_fused else {}
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        **extra_args,
    )
    print(f"using fused AdamW: {use_fused}")
    return optimizer


def run_eval(
    *,
    model,
    prompt_bank,
    prompt_bank_dir: str,
    device: str,
    autocast_context,
    eval_n: int,
    eval_batch_size: int,
) -> dict[str, float]:
    model.eval()
    val_loss = evaluate_clean_ce_loss_hf(
        model,
        prompt_bank,
        batch_size=eval_batch_size,
        device=device,
        autocast_context=autocast_context,
    )
    metrics = evaluate_saved_clean_s5_metrics_hf(
        model,
        device=device,
        data_dir=prompt_bank_dir,
        n_eval=eval_n,
        batch_size=eval_batch_size,
        autocast_context=autocast_context,
    )
    model.train()
    return {
        "val/loss": val_loss,
        "val/cot_exact": metrics["cot_exact"],
        "val/clean_full_exact": metrics["clean_full_exact"],
        "val/clean_final_exact": metrics["clean_final_exact"],
    }


def save_hf_checkpoint(
    checkpoint_dir: Path,
    *,
    model: GPT2LMHeadModel,
    optimizer: torch.optim.Optimizer,
    model_args: dict[str, object],
    iter_num: int,
    best_val_loss: float,
    config: dict[str, object],
    prompt_cycle: FixedPromptCycle,
    device: str,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir / HF_MODEL_DIRNAME, safe_serialization=False)
    training_state = {
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
        "prompt_cycle_state": prompt_cycle.state_dict(),
        "rng_state": capture_rng_state(device),
    }
    torch.save(training_state, checkpoint_dir / TRAINING_STATE_NAME)


def maybe_save_snapshot(
    *,
    out_dir: Path,
    model: GPT2LMHeadModel,
    optimizer: torch.optim.Optimizer,
    model_args: dict[str, object],
    iter_num: int,
    best_val_loss: float,
    config: dict[str, object],
    prompt_cycle: FixedPromptCycle,
    device: str,
) -> None:
    snapshot_dir = out_dir / f"checkpoint_{iter_num:07d}"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    save_hf_checkpoint(
        snapshot_dir,
        model=model,
        optimizer=optimizer,
        model_args=model_args,
        iter_num=iter_num,
        best_val_loss=best_val_loss,
        config=config,
        prompt_cycle=prompt_cycle,
        device=device,
    )


def load_student(
    args: OpdHfConfig,
    *,
    device: str,
    torch_dtype: torch.dtype,
) -> tuple[GPT2LMHeadModel, dict[str, object], int, float, dict[str, object] | None, dict[str, object] | None]:
    del torch_dtype  # student weights stay in fp32; autocast controls activation dtype.
    if args.init_from == "resume":
        training_state = torch.load(
            Path(args.out_dir) / TRAINING_STATE_NAME,
            map_location="cpu",
            weights_only=False,
        )
        model = GPT2LMHeadModel.from_pretrained(
            Path(args.out_dir) / HF_MODEL_DIRNAME,
        )
        set_hf_causal_lm_loss(model)
        apply_nanogpt_bias_policy(
            model,
            has_bias=bool(training_state["model_args"].get("bias", True)),
        )
        model.to(device)
        model.train()
        return (
            model,
            dict(training_state["model_args"]),
            int(training_state["iter_num"]),
            float(training_state.get("best_val_loss", float("inf"))),
            training_state.get("prompt_cycle_state"),
            training_state.get("rng_state"),
        )

    teacher_checkpoint = load_nanogpt_checkpoint(args.teacher_checkpoint, map_location="cpu")
    model_args = dict(teacher_checkpoint["model_args"])
    del teacher_checkpoint
    model = build_hf_model_from_nanogpt_args(
        model_args,
        device=device,
        eval_mode=False,
    )
    model.train()
    return model, model_args, 0, float("inf"), None, None


def set_mode(model: torch.nn.Module, compiled_model: torch.nn.Module, train: bool) -> None:
    model.train(train)
    if compiled_model is not model:
        compiled_model.train(train)


def run_opd_hf(cfg: OpdHfConfig, *, launcher_command: list[str]) -> None:
    validate_config(cfg)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("The HF OPD trainer is single-GPU only in v1.")

    device = resolve_device(cfg.device)
    torch_dtype = resolve_dtype(cfg.dtype, device)
    autocast_context = build_autocast_context(device, torch_dtype)
    scaler = build_grad_scaler(device=device, torch_dtype=torch_dtype)

    if cfg.wandb_run_name is None:
        eta_tag = str(cfg.eta).replace(".", "p")
        temp_tag = format_student_rollout_tag(
            rollout_temperature=resolve_student_rollout_temperature(cfg),
            student_temperature=cfg.student_temperature,
        )
        cfg.wandb_run_name = (
            f"s5-opd-hf-{cfg.objective}-n{cfg.subset_size}-eta{eta_tag}-"
            f"{cfg.teacher_law}-{temp_tag}"
        )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_launch_metadata(out_dir, cfg=cfg, command=launcher_command)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    prompt_bank = load_prompt_bank(cfg.prompt_bank_dir)
    target_len = canonical_target_len(prompt_bank)
    training_seq_len = prompt_bank.prompt_len + target_len - 1
    print_prompt_bank_target_span_diagnostic(
        method_name=f"opd_hf/{cfg.objective}",
        prompt_bank=prompt_bank,
        actual_target_len=target_len,
        total_sequence_len=training_seq_len,
        target_description=(
            "clean reference continuation; HF online teacher supervision is applied "
            "over the same target positions"
        ),
    )
    subset_indices = select_train_subset(prompt_bank, cfg.subset_size)

    run_metadata = {
        "backend": "hf",
        "prompt_len": prompt_bank.prompt_len,
        "cot_len": prompt_bank.cot_len,
        "target_len": prompt_bank.target_len,
        "final_answer_len": prompt_bank.final_answer_len,
        "answer_len": prompt_bank.answer_len,
        "target_span": prompt_bank.meta.get("target_span", "cot_with_final_answer_suffix"),
        "teacher_checkpoint": cfg.teacher_checkpoint,
        "prompt_bank_dir": cfg.prompt_bank_dir,
        "subset_size": cfg.subset_size,
        "eta": cfg.eta,
        "teacher_law": cfg.teacher_law,
        "objective": cfg.objective,
        "student_temperature": cfg.student_temperature,
        "student_rollout_temperature": resolve_student_rollout_temperature(cfg),
        "shuffle_prompts": cfg.shuffle_prompts,
        "seed": cfg.seed,
        "device": device,
        "dtype": str(torch_dtype).replace("torch.", ""),
        "compile": bool(cfg.compile),
    }
    if cfg.teacher_law == SEMANTIC_KEY_NOISE_LAW:
        run_metadata["semantic_key_noise"] = semantic_key_noise_config_from_obj(
            cfg.semantic_key_noise
        ).to_dict()
    if cfg.teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        run_metadata["random_suffix_noise"] = random_suffix_noise_config_from_obj(
            cfg.random_suffix_noise
        ).to_dict()
    if cfg.init_from == "resume":
        validate_resume_metadata(out_dir, run_metadata)
    save_run_metadata(out_dir, subset_indices=subset_indices, metadata=run_metadata)

    student, model_args, iter_num, best_val_loss, prompt_cycle_state, rng_state = load_student(
        cfg,
        device=device,
        torch_dtype=torch_dtype,
    )
    teacher = load_nanogpt_checkpoint_as_hf(
        cfg.teacher_checkpoint,
        map_location="cpu",
        device=device,
        torch_dtype=torch_dtype,
        eval_mode=True,
    )
    for param in teacher.parameters():
        param.requires_grad = False

    prompt_cycle = FixedPromptCycle(
        prompt_bank.clean_train_prompt_ids,
        order=subset_indices,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_prompts,
        seed=cfg.seed,
    )
    if prompt_cycle_state is not None:
        prompt_cycle.load_state_dict(prompt_cycle_state)
    if rng_state is not None:
        restore_rng_state(rng_state, device)

    optimizer = configure_hf_optimizer(
        student,
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        device_type="cuda" if "cuda" in device else "cpu",
    )
    if cfg.init_from == "resume":
        training_state = torch.load(
            out_dir / TRAINING_STATE_NAME,
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(training_state["optimizer"])
        del training_state

    train_student = student
    if cfg.compile:
        print("compiling the student train path... (takes a ~minute)")
        train_student = torch.compile(student)

    config = cfg.config_dict()
    config.update(
        {
            "resolved_device": device,
            "resolved_dtype": str(torch_dtype).replace("torch.", ""),
            "prompt_len": prompt_bank.prompt_len,
            "cot_len": prompt_bank.cot_len,
            "target_len": prompt_bank.target_len,
            "final_answer_len": prompt_bank.final_answer_len,
            "answer_len": prompt_bank.answer_len,
            "target_span": prompt_bank.meta.get("target_span", "cot_with_final_answer_suffix"),
            "backend": "hf",
        }
    )
    wandb = maybe_init_wandb(
        enabled=cfg.wandb_log,
        project=cfg.wandb_project,
        run_name=cfg.wandb_run_name,
        run_id=cfg.wandb_run_id,
        out_dir=cfg.out_dir,
        init_from=cfg.init_from,
        init_timeout=cfg.wandb_init_timeout,
        config=config,
    )

    policy_temperature = cfg.student_temperature if cfg.student_temperature > 0 else None
    rollout_temperature = resolve_student_rollout_temperature(cfg)
    running_metrics: dict[str, float] = {}
    running_steps = 0
    t0 = time.time()

    while iter_num < cfg.max_iters:
        lr = get_linear_warmup_lr(iter_num, learning_rate=cfg.learning_rate, warmup_iters=cfg.warmup_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % cfg.eval_interval == 0:
            eval_stats = run_eval(
                model=student,
                prompt_bank=prompt_bank,
                prompt_bank_dir=cfg.prompt_bank_dir,
                device=device,
                autocast_context=autocast_context,
                eval_n=cfg.eval_n,
                eval_batch_size=cfg.eval_batch_size,
            )
            best_val_loss = min(best_val_loss, eval_stats["val/loss"])
            msg = (
                f"eval step {iter_num}: val loss {eval_stats['val/loss']:.4f}, "
                f"val cot_exact {eval_stats['val/cot_exact']:.4f}, "
                f"val clean_full_exact {eval_stats['val/clean_full_exact']:.4f}, "
                f"val clean_final_exact {eval_stats['val/clean_final_exact']:.4f}"
            )
            print(msg)
            save_hf_checkpoint(
                out_dir,
                model=student,
                optimizer=optimizer,
                model_args=model_args,
                iter_num=iter_num,
                best_val_loss=best_val_loss,
                config=config,
                prompt_cycle=prompt_cycle,
                device=device,
            )
            if cfg.save_interval > 0 and iter_num > 0 and iter_num % cfg.save_interval == 0:
                maybe_save_snapshot(
                    out_dir=out_dir,
                    model=student,
                    optimizer=optimizer,
                    model_args=model_args,
                    iter_num=iter_num,
                    best_val_loss=best_val_loss,
                    config=config,
                    prompt_cycle=prompt_cycle,
                    device=device,
                )
            if wandb is not None:
                wandb.log({"iter": iter_num, **eval_stats})

        prompt_indices = prompt_cycle.next_batch_indices()
        prompt_ids = prompt_bank.clean_train_prompt_ids.index_select(0, prompt_indices)
        clean_target_ids = prompt_bank.clean_train_cot_ids.index_select(
            0,
            prompt_indices,
        )[:, :target_len]

        set_mode(student, train_student, train=False)
        with torch.no_grad():
            full_seq, actions, log_q = rollout_student_hf(
                student,
                prompt_ids,
                target_len=target_len,
                temperature=rollout_temperature,
                device=device,
                autocast_context=autocast_context,
            )
            rollout_inputs = full_seq[:, :-1]
            teacher_prob_kwargs = {}
            if cfg.teacher_law == SEMANTIC_KEY_NOISE_LAW:
                teacher_prob_kwargs["semantic_key_noise_config"] = cfg.semantic_key_noise
            if cfg.teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
                teacher_prob_kwargs["clean_target_ids"] = clean_target_ids
                teacher_prob_kwargs["random_suffix_noise_config"] = cfg.random_suffix_noise
            teacher_probs = cached_teacher_token_probs_hf(
                teacher,
                prompt_ids,
                actions,
                eta=cfg.eta,
                teacher_law=cfg.teacher_law,
                device=device,
                autocast_context=autocast_context,
                **teacher_prob_kwargs,
            )
            if cfg.objective == "reverse_kl_tm":
                teacher_action_probs = teacher_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
                log_teacher = torch.log(teacher_action_probs.clamp_min(cfg.eps))
                advantage = log_teacher - log_q
            elif cfg.objective == "forward_kl_simple":
                teacher_targets = sample_teacher_actions(teacher_probs)
                teacher_target_probs = teacher_probs.gather(2, teacher_targets.unsqueeze(-1)).squeeze(-1)
                log_teacher_target = torch.log(teacher_target_probs.clamp_min(cfg.eps))
        set_mode(student, train_student, train=True)

        with autocast_context:
            outputs = train_student(
                input_ids=rollout_inputs,
                use_cache=False,
            )
            p_answer_logits = extract_answer_logits(
                outputs.logits,
                prompt_len=prompt_bank.prompt_len,
                target_len=target_len,
            )
            if (
                int(actions.size(1)) != target_len
                or int(teacher_probs.size(1)) != target_len
                or int(p_answer_logits.size(1)) != target_len
            ):
                raise ValueError("HF OPD target tensors do not match target_len")
            if cfg.objective == "reverse_kl_tm":
                log_p = gather_action_log_probs(p_answer_logits, actions)
                importance_weight = torch.exp(log_p - log_q.detach())
                loss = -(importance_weight * advantage.detach()).mean()
                step_metrics = {
                    "train/loss": float(loss.item()),
                    "train/advantage": float(advantage.mean().item()),
                    "train/log_q": float(log_q.mean().item()),
                    "train/log_teacher": float(log_teacher.mean().item()),
                }
            elif cfg.objective == "forward_kl_simple":
                log_student_target = gather_action_log_probs(
                    p_answer_logits,
                    teacher_targets,
                    temperature=policy_temperature,
                )
                loss = -log_student_target.mean()
                step_metrics = {
                    "train/loss": float(loss.item()),
                    "train/log_student_target": float(log_student_target.mean().item()),
                    "train/log_teacher_target": float(log_teacher_target.mean().item()),
                }
            else:
                token_kl, teacher_ce, teacher_entropy = teacher_forward_kl(
                    teacher_probs,
                    p_answer_logits,
                    temperature=policy_temperature,
                    eps=cfg.eps,
                )
                loss = token_kl.mean()
                step_metrics = {
                    "train/loss": float(loss.item()),
                    "train/forward_kl": float(token_kl.mean().item()),
                    "train/teacher_ce": float(teacher_ce.mean().item()),
                    "train/teacher_entropy": float(teacher_entropy.mean().item()),
                }

        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        for key, value in step_metrics.items():
            running_metrics[key] = running_metrics.get(key, 0.0) + value
        running_steps += 1

        if iter_num % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            denom = max(1, running_steps)
            train_stats = {key: value / denom for key, value in running_metrics.items()}
            train_stats["lr"] = lr
            train_stats["iter"] = iter_num
            metric_str = ", ".join(
                f"{key.split('/')[-1]} {train_stats[key]:.4f}"
                for key in train_stats
                if key.startswith("train/")
            )
            print(f"iter {iter_num}: {metric_str}, time {dt*1000:.2f}ms")
            if wandb is not None:
                wandb.log(train_stats)
            running_metrics = {}
            running_steps = 0

        iter_num += 1

    final_stats = run_eval(
        model=student,
        prompt_bank=prompt_bank,
        prompt_bank_dir=cfg.prompt_bank_dir,
        device=device,
        autocast_context=autocast_context,
        eval_n=cfg.eval_n,
        eval_batch_size=cfg.eval_batch_size,
    )
    best_val_loss = min(best_val_loss, final_stats["val/loss"])
    print(
        f"final step {iter_num}: val loss {final_stats['val/loss']:.4f}, "
        f"val cot_exact {final_stats['val/cot_exact']:.4f}, "
        f"val clean_full_exact {final_stats['val/clean_full_exact']:.4f}, "
        f"val clean_final_exact {final_stats['val/clean_final_exact']:.4f}"
    )
    save_hf_checkpoint(
        out_dir,
        model=student,
        optimizer=optimizer,
        model_args=model_args,
        iter_num=iter_num,
        best_val_loss=best_val_loss,
        config=config,
        prompt_cycle=prompt_cycle,
        device=device,
    )
    mark_complete(out_dir, iter_num)
    if wandb is not None:
        wandb.log({"iter": iter_num, **final_stats})
        wandb.finish()
validate_args = validate_config
