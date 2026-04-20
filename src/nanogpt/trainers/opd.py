from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from data.modular_addition.task import (
    corruptible_token_ids as modadd_corruptible_token_ids,
    evaluate_saved_clean_modadd_metrics,
)
from data.s5_cot.opd import (
    FixedPromptCycle,
    cached_teacher_token_probs,
    evaluate_clean_ce_loss,
    extract_answer_logits,
    forward_kl_full_loss,
    forward_kl_simple_loss,
    gather_action_log_probs,
    reverse_kl_full_loss,
    reverse_kl_tm_loss,
    rollout_student,
    sample_teacher_actions,
)
from data.s5_cot.task import CORRUPTIBLE_IDS as S5_CORRUPTIBLE_IDS
from data.s5_cot.task import evaluate_saved_clean_s5_metrics
from data.synthetic.prompt_bank import load_prompt_bank, select_train_subset
from nanogpt_checkpoint import (
    build_nanogpt_model,
    load_nanogpt_checkpoint,
    load_nanogpt_model,
)
from nanogpt.trainers.configs import OpdConfig
from nanogpt.trainers.runtime import (
    build_autocast_context,
    build_grad_scaler,
    capture_rng_state,
    get_nanogpt_lr,
    resolve_device,
    resolve_dtype,
    restore_rng_state,
)
from nanogpt.trainers.wandb import maybe_init_wandb
from nanogpt.utils.repo import write_launch_metadata


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


def validate_config(cfg: OpdConfig) -> None:
    init_from = getattr(cfg, "init_from", "scratch")
    init_from_ckpt = getattr(cfg, "init_from_ckpt", None)
    continue_from_subset_size = getattr(cfg, "continue_from_subset_size", 0)
    single_epoch = getattr(cfg, "single_epoch", False)
    shuffle_prompts = getattr(cfg, "shuffle_prompts", False)
    subset_size = getattr(cfg, "subset_size", 0)
    student_rollout_temperature = resolve_student_rollout_temperature(cfg)

    if cfg.objective in {"forward_kl_simple", "forward_kl_full"} and cfg.student_temperature <= 0:
        raise ValueError(
            "student_temperature must be > 0 for forward-KL objectives because "
            "the loss is defined against the temperature-adjusted student policy."
        )
    if student_rollout_temperature < 0:
        raise ValueError("student_rollout_temperature must be non-negative.")
    if getattr(cfg, "compile", False) and not hasattr(torch, "compile"):
        raise ValueError("--compile requires a PyTorch build with torch.compile support.")
    if init_from == "warm_start" and not init_from_ckpt:
        raise ValueError("--init_from=warm_start requires --init_from_ckpt.")
    if init_from != "warm_start" and init_from_ckpt is not None:
        raise ValueError("--init_from_ckpt is only supported with --init_from=warm_start.")
    if continue_from_subset_size < 0:
        raise ValueError("--continue_from_subset_size must be non-negative.")
    if continue_from_subset_size > 0:
        if init_from != "warm_start":
            raise ValueError(
                "--continue_from_subset_size is only supported with --init_from=warm_start."
            )
        if not single_epoch:
            raise ValueError("--continue_from_subset_size requires --single_epoch.")
        if shuffle_prompts:
            raise ValueError("--continue_from_subset_size requires prompts to remain unshuffled.")
        if continue_from_subset_size > subset_size:
            raise ValueError(
                "--continue_from_subset_size cannot exceed the current --subset_size."
            )


def normalize_state_dict_for_save(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state_dict.items()}


def task_reports_cot_exact(task_name: str) -> bool:
    return False


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
    for key in (
        "task",
        "p",
        "m",
        "teacher_checkpoint",
        "prompt_bank_dir",
        "subset_size",
        "eta",
        "teacher_law",
        "objective",
        "student_temperature",
        "student_rollout_temperature",
        "shuffle_prompts",
        "single_epoch",
        "init_from_ckpt",
        "continue_from_subset_size",
        "seed",
    ):
        current_value = metadata.get(key)
        current_task = metadata.get("task", "s5")
        if key == "task" and current_value is None:
            current_value = "s5"
        if key == "p" and current_value is None and current_task == "s5":
            current_value = 5
        if key == "m" and current_value is None and current_task == "s5":
            current_value = saved.get("m")
        if key == "student_rollout_temperature" and current_value is None:
            current_value = resolve_student_rollout_temperature(metadata)
        saved_value = saved.get(key)
        if key == "task" and saved_value is None:
            saved_value = "s5"
        if key == "p" and saved_value is None and current_task == "s5":
            saved_value = 5
        if key == "m" and saved_value is None and current_task == "s5":
            saved_value = current_value
        if key == "objective" and saved_value is None:
            saved_value = "reverse_kl_tm"
        if key == "student_rollout_temperature" and saved_value is None:
            saved_value = resolve_student_rollout_temperature(saved)
        if saved_value != current_value:
            raise ValueError(
                f"Resume mismatch for {key}: saved={saved_value!r} "
                f"current={current_value!r}"
            )


def mark_complete(out_dir: Path, iter_num: int) -> None:
    with open(out_dir / "completed.txt", "w", encoding="utf-8") as f:
        f.write(f"iter_num={iter_num}\n")


def load_student(
    args: OpdConfig,
    *,
    device: str,
) -> tuple[
    torch.nn.Module,
    dict[str, object],
    int,
    float,
    dict[str, object] | None,
    dict[str, object] | None,
    dict[str, object] | None,
]:
    if args.init_from == "resume":
        model, checkpoint = load_nanogpt_model(
            args.out_dir,
            map_location="cpu",
            device=device,
            eval_mode=False,
            return_checkpoint=True,
        )
        iter_num = int(checkpoint["iter_num"])
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        return (
            model,
            dict(checkpoint["model_args"]),
            iter_num,
            best_val_loss,
            checkpoint.get("prompt_cycle_state"),
            checkpoint.get("rng_state"),
            checkpoint,
        )
    if args.init_from == "warm_start":
        model, checkpoint = load_nanogpt_model(
            args.init_from_ckpt,
            map_location="cpu",
            device=device,
            eval_mode=False,
            return_checkpoint=True,
        )
        iter_num = int(checkpoint["iter_num"])
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        return (
            model,
            dict(checkpoint["model_args"]),
            iter_num,
            best_val_loss,
            checkpoint.get("prompt_cycle_state"),
            checkpoint.get("rng_state"),
            checkpoint,
        )

    teacher_checkpoint = load_nanogpt_checkpoint(args.teacher_checkpoint, map_location="cpu")
    model_args = dict(teacher_checkpoint["model_args"])
    model = build_nanogpt_model(model_args)
    model.to(device)
    model.train()
    return model, model_args, 0, float("inf"), None, None, None


def validate_warm_start_checkpoint(
    args: OpdConfig,
    *,
    subset_indices: torch.Tensor,
    checkpoint: dict[str, object],
) -> None:
    source_config = checkpoint.get("config", {})
    for key in (
        "task",
        "eta",
        "teacher_law",
        "objective",
        "student_temperature",
        "student_rollout_temperature",
        "shuffle_prompts",
        "single_epoch",
    ):
        source_value = source_config.get(key)
        current_value = getattr(args, key, None)
        if key == "student_rollout_temperature":
            if source_value is None:
                source_value = resolve_student_rollout_temperature(source_config)
            current_value = resolve_student_rollout_temperature(args)
        if source_value is not None and source_value != current_value:
            raise ValueError(
                f"Warm-start mismatch for {key}: checkpoint has {source_value!r}, "
                f"current config requests {current_value!r}"
            )

    if args.continue_from_subset_size <= 0:
        return

    prompt_cycle_state = checkpoint.get("prompt_cycle_state")
    if prompt_cycle_state is None:
        raise ValueError(
            "Warm-start continuation requires prompt_cycle_state in the source checkpoint."
        )

    expected_seen = int(args.continue_from_subset_size)
    source_n = int(prompt_cycle_state["n"])
    source_pos = int(prompt_cycle_state["pos"])
    source_epoch = int(prompt_cycle_state["epoch"])
    if source_n != expected_seen:
        raise ValueError(
            f"Warm-start source checkpoint covered n={source_n}, expected "
            f"continue_from_subset_size={expected_seen}"
        )
    if source_pos != expected_seen or source_epoch != 0:
        raise ValueError(
            "Warm-start source checkpoint must represent a completed single pass "
            f"through the first {expected_seen} prompts; got pos={source_pos}, epoch={source_epoch}"
        )

    source_subset_size = source_config.get("subset_size")
    if source_subset_size is not None and int(source_subset_size) != expected_seen:
        raise ValueError(
            f"Warm-start source checkpoint subset_size={source_subset_size} does not match "
            f"continue_from_subset_size={expected_seen}"
        )

    source_base_order = prompt_cycle_state["base_order"].to(device="cpu", dtype=torch.long)
    expected_base_order = subset_indices[:expected_seen].to(device="cpu", dtype=torch.long)
    if source_base_order.numel() != expected_seen or not torch.equal(source_base_order, expected_base_order):
        raise ValueError(
            "Warm-start source checkpoint does not match the prefix of the requested larger subset."
        )

def resolve_task_helpers(task_name: str, *, p: int):
    if task_name == "s5":
        return tuple(S5_CORRUPTIBLE_IDS), evaluate_saved_clean_s5_metrics
    if task_name == "modadd":
        return tuple(modadd_corruptible_token_ids(p)), evaluate_saved_clean_modadd_metrics
    raise ValueError(f"unknown task {task_name!r}")


def save_eval_summary(out_dir: Path, *, iter_num: int, reason: str, metrics: dict[str, float]) -> None:
    payload = {
        "iter": int(iter_num),
        "reason": reason,
        **{key: float(value) for key, value in metrics.items()},
    }
    with open(out_dir / "last_eval.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(out_dir / "eval_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def run_eval(
    *,
    model,
    prompt_bank,
    prompt_bank_dir: str,
    task_name: str,
    device: str,
    autocast_context,
    eval_n: int,
    eval_batch_size: int,
) -> dict[str, float]:
    model.eval()
    val_loss = evaluate_clean_ce_loss(
        model,
        prompt_bank,
        batch_size=eval_batch_size,
        device=device,
        autocast_context=autocast_context,
    )
    _, evaluate_saved_metrics = resolve_task_helpers(task_name, p=prompt_bank.p)
    metrics = evaluate_saved_metrics(
        model,
        device=device,
        data_dir=prompt_bank_dir,
        n_eval=eval_n,
        batch_size=eval_batch_size,
    )
    model.train()
    eval_stats = {
        "val/loss": val_loss,
        "val/clean_full_exact": metrics["clean_full_exact"],
        "val/clean_final_exact": metrics["clean_final_exact"],
    }
    if task_reports_cot_exact(task_name):
        eval_stats["val/cot_exact"] = metrics["cot_exact"]
    return eval_stats


def save_checkpoint(
    *,
    out_dir: Path,
    model,
    optimizer,
    model_args: dict[str, object],
    iter_num: int,
    best_val_loss: float,
    config: dict[str, object],
    prompt_cycle: FixedPromptCycle,
    device: str,
) -> None:
    checkpoint = {
        "model": normalize_state_dict_for_save(model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
        "prompt_cycle_state": prompt_cycle.state_dict(),
        "rng_state": capture_rng_state(device),
    }
    torch.save(checkpoint, out_dir / "ckpt.pt")


def run_opd(cfg: OpdConfig, *, launcher_command: list[str]) -> None:
    validate_config(cfg)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("train_opd.py is single-GPU only in v1.")

    device = resolve_device(cfg.device)
    torch_dtype = resolve_dtype(cfg.dtype, device)
    autocast_context = build_autocast_context(device, torch_dtype)
    scaler = build_grad_scaler(device=device, torch_dtype=torch_dtype)

    prompt_bank = load_prompt_bank(cfg.prompt_bank_dir)
    if prompt_bank.task != cfg.task:
        raise ValueError(
            f"Prompt bank task mismatch: prompt bank has task={prompt_bank.task!r} "
            f"but task={cfg.task!r}"
        )
    corruptible_ids, _ = resolve_task_helpers(cfg.task, p=prompt_bank.p)

    if cfg.wandb_run_name is None:
        eta_tag = str(cfg.eta).replace(".", "p")
        temp_tag = format_student_rollout_tag(
            rollout_temperature=resolve_student_rollout_temperature(cfg),
            student_temperature=cfg.student_temperature,
        )
        cfg.wandb_run_name = (
            f"{cfg.task}-opd-{cfg.objective}-p{prompt_bank.p}-m{prompt_bank.m}-n{cfg.subset_size}-eta{eta_tag}-"
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

    subset_indices = select_train_subset(prompt_bank, cfg.subset_size)

    run_metadata = {
        "task": cfg.task,
        "p": prompt_bank.p,
        "m": prompt_bank.m,
        "prompt_len": prompt_bank.prompt_len,
        "cot_len": prompt_bank.cot_len,
        "final_answer_len": prompt_bank.final_answer_len,
        "teacher_checkpoint": cfg.teacher_checkpoint,
        "prompt_bank_dir": cfg.prompt_bank_dir,
        "subset_size": cfg.subset_size,
        "eta": cfg.eta,
        "teacher_law": cfg.teacher_law,
        "objective": cfg.objective,
        "student_temperature": cfg.student_temperature,
        "student_rollout_temperature": resolve_student_rollout_temperature(cfg),
        "shuffle_prompts": cfg.shuffle_prompts,
        "single_epoch": cfg.single_epoch,
        "init_from_ckpt": cfg.init_from_ckpt,
        "continue_from_subset_size": cfg.continue_from_subset_size,
        "seed": cfg.seed,
        "device": device,
        "dtype": str(torch_dtype).replace("torch.", ""),
        "compile": bool(cfg.compile),
    }
    if cfg.init_from == "resume":
        validate_resume_metadata(out_dir, run_metadata)
    save_run_metadata(out_dir, subset_indices=subset_indices, metadata=run_metadata)

    student, model_args, iter_num, best_val_loss, prompt_cycle_state, rng_state, source_checkpoint = load_student(
        cfg,
        device=device,
    )
    if cfg.init_from == "warm_start":
        assert source_checkpoint is not None
        validate_warm_start_checkpoint(
            cfg,
            subset_indices=subset_indices,
            checkpoint=source_checkpoint,
        )
    teacher = load_nanogpt_model(
        cfg.teacher_checkpoint,
        map_location="cpu",
        device=device,
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
    if cfg.init_from == "resume" and prompt_cycle_state is not None:
        prompt_cycle.load_state_dict(prompt_cycle_state)
    elif cfg.init_from == "warm_start" and cfg.continue_from_subset_size > 0:
        prompt_cycle.pos = int(cfg.continue_from_subset_size)
    if rng_state is not None:
        restore_rng_state(rng_state, device)

    optimizer = student.configure_optimizers(
        cfg.weight_decay,
        cfg.learning_rate,
        (cfg.beta1, cfg.beta2),
        "cuda" if "cuda" in device else "cpu",
    )
    if cfg.init_from in {"resume", "warm_start"}:
        assert source_checkpoint is not None
        optimizer.load_state_dict(source_checkpoint["optimizer"])

    if cfg.compile:
        print("compiling the student model... (takes a ~minute)")
        student = torch.compile(student)

    config = cfg.config_dict()
    config.update(
        {
            "task": cfg.task,
            "p": prompt_bank.p,
            "m": prompt_bank.m,
            "resolved_device": device,
            "resolved_dtype": str(torch_dtype).replace("torch.", ""),
            "prompt_len": prompt_bank.prompt_len,
            "cot_len": prompt_bank.cot_len,
            "final_answer_len": prompt_bank.final_answer_len,
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
        if cfg.single_epoch and not prompt_cycle.has_remaining_in_epoch():
            break

        lr = (
            get_nanogpt_lr(
                iter_num,
                learning_rate=cfg.learning_rate,
                warmup_iters=cfg.warmup_iters,
                lr_decay_iters=cfg.lr_decay_iters,
                min_lr=cfg.min_lr,
            )
            if cfg.decay_lr
            else cfg.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % cfg.eval_interval == 0:
            eval_stats = run_eval(
                model=student,
                prompt_bank=prompt_bank,
                prompt_bank_dir=cfg.prompt_bank_dir,
                task_name=cfg.task,
                device=device,
                autocast_context=autocast_context,
                eval_n=cfg.eval_n,
                eval_batch_size=cfg.eval_batch_size,
            )
            best_val_loss = min(best_val_loss, eval_stats["val/loss"])
            save_eval_summary(out_dir, iter_num=iter_num, reason="periodic", metrics=eval_stats)
            msg_parts = [
                f"eval step {iter_num}: val loss {eval_stats['val/loss']:.4f}",
            ]
            if "val/cot_exact" in eval_stats:
                msg_parts.append(f"val cot_exact {eval_stats['val/cot_exact']:.4f}")
            msg_parts.append(f"val clean_full_exact {eval_stats['val/clean_full_exact']:.4f}")
            msg_parts.append(f"val clean_final_exact {eval_stats['val/clean_final_exact']:.4f}")
            msg = ", ".join(msg_parts)
            print(msg)
            save_checkpoint(
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
            if cfg.save_interval > 0 and iter_num > 0 and iter_num % cfg.save_interval == 0:
                torch.save(
                    torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False),
                    out_dir / f"ckpt_{iter_num:07d}.pt",
                )
            if wandb is not None:
                wandb.log({"iter": iter_num, **eval_stats})

        if cfg.single_epoch:
            prompt_ids = prompt_cycle.next_batch_no_wrap()
        else:
            prompt_ids = prompt_cycle.next_batch()

        student.eval()
        with torch.no_grad():
            full_seq, actions, log_q = rollout_student(
                student,
                prompt_ids,
                target_len=prompt_bank.cot_len,
                temperature=rollout_temperature,
                device=device,
                autocast_context=autocast_context,
            )
            rollout_inputs = full_seq[:, :-1]
            teacher_probs = cached_teacher_token_probs(
                teacher,
                prompt_ids,
                actions,
                eta=cfg.eta,
                teacher_law=cfg.teacher_law,
                corruptible_token_ids=corruptible_ids,
                device=device,
                autocast_context=autocast_context,
            )
            if cfg.objective == "reverse_kl_tm":
                teacher_action_probs = teacher_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
                log_teacher = torch.log(teacher_action_probs.clamp_min(cfg.eps))
                advantage = log_teacher - log_q
            elif cfg.objective == "forward_kl_simple":
                teacher_targets = sample_teacher_actions(teacher_probs)
                teacher_target_probs = teacher_probs.gather(2, teacher_targets.unsqueeze(-1)).squeeze(-1)
                log_teacher_target = torch.log(teacher_target_probs.clamp_min(cfg.eps))
        student.train()

        with autocast_context:
            p_logits, _ = student(rollout_inputs, return_full_logits=True)
            p_answer_logits = extract_answer_logits(
                p_logits,
                prompt_len=prompt_bank.prompt_len,
                target_len=prompt_bank.cot_len,
            )
            if cfg.objective == "reverse_kl_tm":
                loss, objective_stats = reverse_kl_tm_loss(
                    p_answer_logits,
                    actions,
                    log_q=log_q,
                    teacher_probs=teacher_probs,
                    eps=cfg.eps,
                )
                step_metrics = {
                    "train/loss": float(loss.item()),
                    "train/advantage": float(objective_stats["advantage"].mean().item()),
                    "train/log_q": float(log_q.mean().item()),
                    "train/log_teacher": float(objective_stats["log_teacher"].mean().item()),
                }
            elif cfg.objective == "reverse_kl_full":
                loss, objective_stats = reverse_kl_full_loss(
                    p_answer_logits,
                    teacher_probs=teacher_probs,
                    eps=cfg.eps,
                )
                step_metrics = {
                    "train/loss": float(loss.item()),
                    "train/reverse_kl": float(objective_stats["reverse_kl"].mean().item()),
                    "train/student_teacher_ce": float(objective_stats["student_teacher_ce"].mean().item()),
                    "train/student_entropy": float(objective_stats["student_entropy"].mean().item()),
                }
            elif cfg.objective == "forward_kl_simple":
                loss, objective_stats = forward_kl_simple_loss(
                    p_answer_logits,
                    teacher_targets,
                    teacher_probs=teacher_probs,
                    temperature=policy_temperature,
                    eps=cfg.eps,
                )
                step_metrics = {
                    "train/loss": float(loss.item()),
                    "train/log_student_target": float(objective_stats["log_student_target"].mean().item()),
                    "train/log_teacher_target": float(objective_stats["log_teacher_target"].mean().item()),
                }
            else:
                loss, objective_stats = forward_kl_full_loss(
                    p_answer_logits,
                    teacher_probs=teacher_probs,
                    temperature=policy_temperature,
                    eps=cfg.eps,
                )
                step_metrics = {
                    "train/loss": float(loss.item()),
                    "train/forward_kl": float(objective_stats["forward_kl"].mean().item()),
                    "train/teacher_ce": float(objective_stats["teacher_ce"].mean().item()),
                    "train/teacher_entropy": float(objective_stats["teacher_entropy"].mean().item()),
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
        task_name=cfg.task,
        device=device,
        autocast_context=autocast_context,
        eval_n=cfg.eval_n,
        eval_batch_size=cfg.eval_batch_size,
    )
    best_val_loss = min(best_val_loss, final_stats["val/loss"])
    save_eval_summary(out_dir, iter_num=iter_num, reason="final", metrics=final_stats)
    final_msg_parts = [
        f"final step {iter_num}: val loss {final_stats['val/loss']:.4f}",
    ]
    if "val/cot_exact" in final_stats:
        final_msg_parts.append(f"val cot_exact {final_stats['val/cot_exact']:.4f}")
    final_msg_parts.append(f"val clean_full_exact {final_stats['val/clean_full_exact']:.4f}")
    final_msg_parts.append(f"val clean_final_exact {final_stats['val/clean_final_exact']:.4f}")
    print(", ".join(final_msg_parts))
    save_checkpoint(
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
    mark_complete(out_dir, iter_num)
    if wandb is not None:
        wandb.log({"iter": iter_num, **final_stats})
        wandb.finish()
validate_args = validate_config
