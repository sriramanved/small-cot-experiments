from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from contextlib import nullcontext
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
    reverse_kl_tm_loss,
    rollout_student,
    sample_teacher_actions,
)
from data.s5_cot.task import CORRUPTIBLE_IDS as S5_CORRUPTIBLE_IDS
from data.s5_cot.task import evaluate_saved_clean_s5_metrics
from data.synthetic.prompt_bank import load_prompt_bank, select_train_subset
from hf_checkpoint import DTYPE_LOOKUP
from nanogpt_checkpoint import (
    build_nanogpt_model,
    load_nanogpt_checkpoint,
    load_nanogpt_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a synthetic-task student from scratch with on-policy "
            "distillation against a noisy teacher derived from a clean expert."
        )
    )
    parser.add_argument("--task", choices=("s5", "modadd"), default="s5")
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--prompt_bank_dir", type=str, required=True)
    parser.add_argument("--subset_size", type=int, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument(
        "--teacher_law",
        choices=("distributional_noise", "corrupted_greedy"),
        default="distributional_noise",
    )
    parser.add_argument(
        "--objective",
        choices=("reverse_kl_tm", "forward_kl_simple", "forward_kl_full"),
        default="reverse_kl_tm",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--init_from", choices=("scratch", "resume"), default="scratch")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_iters", type=int, default=110000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--student_temperature", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--eval_n", type=int, default=5000)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--shuffle_prompts", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=sorted(DTYPE_LOOKUP), default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-10)

    parser.add_argument("--wandb_log", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="small-cot-experiments")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.objective != "reverse_kl_tm" and args.student_temperature <= 0:
        raise ValueError(
            "student_temperature must be > 0 for forward-KL objectives because "
            "the loss is defined against the temperature-adjusted student policy."
        )
    if getattr(args, "compile", False) and not hasattr(torch, "compile"):
        raise ValueError("--compile requires a PyTorch build with torch.compile support.")


def resolve_device(device_arg: str | None) -> str:
    if device_arg is not None:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype_name: str | None, device: str) -> torch.dtype:
    if dtype_name is not None:
        return DTYPE_LOOKUP[dtype_name]
    if "cuda" in device:
        return torch.float16
    return torch.float32


def build_autocast_context(device: str, torch_dtype: torch.dtype):
    if "cuda" not in device:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch_dtype)


def get_lr(step: int, *, learning_rate: float, warmup_iters: int) -> float:
    if warmup_iters <= 0 or step >= warmup_iters:
        return learning_rate
    lr_start = 1e-6
    return lr_start + (learning_rate - lr_start) * (step + 1) / (warmup_iters + 1)


def capture_rng_state(device: str) -> dict[str, object]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if "cuda" in device and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object], device: str) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "cuda" in device and torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def normalize_state_dict_for_save(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state_dict.items()}


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
        "shuffle_prompts",
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
        saved_value = saved.get(key)
        if key == "task" and saved_value is None:
            saved_value = "s5"
        if key == "p" and saved_value is None and current_task == "s5":
            saved_value = 5
        if key == "m" and saved_value is None and current_task == "s5":
            saved_value = current_value
        if key == "objective" and saved_value is None:
            saved_value = "reverse_kl_tm"
        if saved_value != current_value:
            raise ValueError(
            f"Resume mismatch for {key}: saved={saved_value!r} "
                f"current={current_value!r}"
            )


def mark_complete(out_dir: Path, iter_num: int) -> None:
    with open(out_dir / "completed.txt", "w", encoding="utf-8") as f:
        f.write(f"iter_num={iter_num}\n")


def load_student(
    args: argparse.Namespace,
    *,
    device: str,
) -> tuple[torch.nn.Module, dict[str, object], int, float, dict[str, object] | None, dict[str, object] | None]:
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
        )

    teacher_checkpoint = load_nanogpt_checkpoint(args.teacher_checkpoint, map_location="cpu")
    model_args = dict(teacher_checkpoint["model_args"])
    model = build_nanogpt_model(model_args)
    model.to(device)
    model.train()
    return model, model_args, 0, float("inf"), None, None


def maybe_init_wandb(args: argparse.Namespace, config: dict[str, object]):
    if not args.wandb_log:
        return None
    import wandb

    out_dir = Path(args.out_dir)
    state_path = out_dir / "wandb_state.json"
    has_saved_run_id = False
    run_id = args.wandb_run_id
    if run_id is None and state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            saved_state = json.load(f)
        run_id = saved_state.get("run_id")
        has_saved_run_id = run_id is not None
    if run_id is None:
        digest = hashlib.sha1(
            f"{args.wandb_project}:{out_dir.resolve()}".encode("utf-8")
        ).hexdigest()
        run_id = digest[:16]
        if args.init_from == "resume":
            print(
                "warning: no saved W&B run id found; using a deterministic fallback id. "
                "This may create a new W&B run instead of resuming the original graph."
            )
    has_explicit_resume_id = args.wandb_run_id is not None or has_saved_run_id
    resume_mode = "must" if args.init_from == "resume" and has_explicit_resume_id else "allow"
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=run_id,
        resume=resume_mode,
        config=config,
    )
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": wandb.run.id,
                "project": args.wandb_project,
                "name": args.wandb_run_name,
            },
            f,
            indent=2,
        )
    return wandb


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
    return {
        "val/loss": val_loss,
        "val/cot_exact": metrics["cot_exact"],
        "val/clean_full_exact": metrics["clean_full_exact"],
        "val/clean_final_exact": metrics["clean_final_exact"],
    }


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


def main() -> None:
    args = parse_args()
    validate_args(args)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("train_opd.py is single-GPU only in v1.")

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype, device)
    autocast_context = build_autocast_context(device, torch_dtype)
    scaler_device = "cuda" if "cuda" in device else "cpu"
    scaler = torch.amp.GradScaler(
        scaler_device,
        enabled=("cuda" in device and torch_dtype == torch.float16),
    )

    prompt_bank = load_prompt_bank(args.prompt_bank_dir)
    if prompt_bank.task != args.task:
        raise ValueError(
            f"Prompt bank task mismatch: prompt bank has task={prompt_bank.task!r} "
            f"but --task={args.task!r}"
        )
    corruptible_ids, _ = resolve_task_helpers(args.task, p=prompt_bank.p)

    if args.wandb_run_name is None:
        eta_tag = str(args.eta).replace(".", "p")
        temp_tag = "greedy" if args.student_temperature == 0 else f"t{args.student_temperature}".replace(".", "p")
        args.wandb_run_name = (
            f"{args.task}-opd-{args.objective}-p{prompt_bank.p}-m{prompt_bank.m}-n{args.subset_size}-eta{eta_tag}-"
            f"{args.teacher_law}-{temp_tag}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    subset_indices = select_train_subset(prompt_bank, args.subset_size)

    run_metadata = {
        "task": args.task,
        "p": prompt_bank.p,
        "m": prompt_bank.m,
        "prompt_len": prompt_bank.prompt_len,
        "cot_len": prompt_bank.cot_len,
        "final_answer_len": prompt_bank.final_answer_len,
        "teacher_checkpoint": args.teacher_checkpoint,
        "prompt_bank_dir": args.prompt_bank_dir,
        "subset_size": args.subset_size,
        "eta": args.eta,
        "teacher_law": args.teacher_law,
        "objective": args.objective,
        "student_temperature": args.student_temperature,
        "shuffle_prompts": args.shuffle_prompts,
        "seed": args.seed,
        "device": device,
        "dtype": str(torch_dtype).replace("torch.", ""),
        "compile": bool(args.compile),
    }
    if args.init_from == "resume":
        validate_resume_metadata(out_dir, run_metadata)
    save_run_metadata(out_dir, subset_indices=subset_indices, metadata=run_metadata)

    student, model_args, iter_num, best_val_loss, prompt_cycle_state, rng_state = load_student(
        args,
        device=device,
    )
    teacher = load_nanogpt_model(
        args.teacher_checkpoint,
        map_location="cpu",
        device=device,
        eval_mode=True,
    )
    for param in teacher.parameters():
        param.requires_grad = False

    prompt_cycle = FixedPromptCycle(
        prompt_bank.clean_train_prompt_ids,
        order=subset_indices,
        batch_size=args.batch_size,
        shuffle=args.shuffle_prompts,
        seed=args.seed,
    )
    if prompt_cycle_state is not None:
        prompt_cycle.load_state_dict(prompt_cycle_state)
    if rng_state is not None:
        restore_rng_state(rng_state, device)

    optimizer = student.configure_optimizers(
        args.weight_decay,
        args.learning_rate,
        (args.beta1, args.beta2),
        "cuda" if "cuda" in device else "cpu",
    )
    if args.init_from == "resume":
        checkpoint = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        del checkpoint

    if args.compile:
        print("compiling the student model... (takes a ~minute)")
        student = torch.compile(student)

    config = vars(args).copy()
    config.update(
        {
            "task": args.task,
            "p": prompt_bank.p,
            "m": prompt_bank.m,
            "resolved_device": device,
            "resolved_dtype": str(torch_dtype).replace("torch.", ""),
            "prompt_len": prompt_bank.prompt_len,
            "cot_len": prompt_bank.cot_len,
            "final_answer_len": prompt_bank.final_answer_len,
        }
    )
    wandb = maybe_init_wandb(args, config)

    policy_temperature = args.student_temperature if args.student_temperature > 0 else None
    running_metrics: dict[str, float] = {}
    running_steps = 0
    t0 = time.time()

    while iter_num < args.max_iters:
        lr = get_lr(iter_num, learning_rate=args.learning_rate, warmup_iters=args.warmup_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % args.eval_interval == 0:
            eval_stats = run_eval(
                model=student,
                prompt_bank=prompt_bank,
                prompt_bank_dir=args.prompt_bank_dir,
                task_name=args.task,
                device=device,
                autocast_context=autocast_context,
                eval_n=args.eval_n,
                eval_batch_size=args.eval_batch_size,
            )
            best_val_loss = min(best_val_loss, eval_stats["val/loss"])
            save_eval_summary(out_dir, iter_num=iter_num, reason="periodic", metrics=eval_stats)
            msg = (
                f"eval step {iter_num}: val loss {eval_stats['val/loss']:.4f}, "
                f"val cot_exact {eval_stats['val/cot_exact']:.4f}, "
                f"val clean_full_exact {eval_stats['val/clean_full_exact']:.4f}, "
                f"val clean_final_exact {eval_stats['val/clean_final_exact']:.4f}"
            )
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
            if args.save_interval > 0 and iter_num > 0 and iter_num % args.save_interval == 0:
                torch.save(
                    torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False),
                    out_dir / f"ckpt_{iter_num:07d}.pt",
                )
            if wandb is not None:
                wandb.log({"iter": iter_num, **eval_stats})

        prompt_ids = prompt_cycle.next_batch()

        student.eval()
        with torch.no_grad():
            full_seq, actions, log_q = rollout_student(
                student,
                prompt_ids,
                target_len=prompt_bank.cot_len,
                temperature=args.student_temperature,
                device=device,
                autocast_context=autocast_context,
            )
            rollout_inputs = full_seq[:, :-1]
            teacher_probs = cached_teacher_token_probs(
                teacher,
                prompt_ids,
                actions,
                eta=args.eta,
                teacher_law=args.teacher_law,
                corruptible_token_ids=corruptible_ids,
                device=device,
                autocast_context=autocast_context,
            )
            if args.objective == "reverse_kl_tm":
                teacher_action_probs = teacher_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
                log_teacher = torch.log(teacher_action_probs.clamp_min(args.eps))
                advantage = log_teacher - log_q
            elif args.objective == "forward_kl_simple":
                teacher_targets = sample_teacher_actions(teacher_probs)
                teacher_target_probs = teacher_probs.gather(2, teacher_targets.unsqueeze(-1)).squeeze(-1)
                log_teacher_target = torch.log(teacher_target_probs.clamp_min(args.eps))
        student.train()

        with autocast_context:
            p_logits, _ = student(rollout_inputs, return_full_logits=True)
            p_answer_logits = extract_answer_logits(
                p_logits,
                prompt_len=prompt_bank.prompt_len,
                target_len=prompt_bank.cot_len,
            )
            if args.objective == "reverse_kl_tm":
                loss, objective_stats = reverse_kl_tm_loss(
                    p_answer_logits,
                    actions,
                    log_q=log_q,
                    teacher_probs=teacher_probs,
                    eps=args.eps,
                )
                step_metrics = {
                    "train/loss": float(loss.item()),
                    "train/advantage": float(objective_stats["advantage"].mean().item()),
                    "train/log_q": float(log_q.mean().item()),
                    "train/log_teacher": float(objective_stats["log_teacher"].mean().item()),
                }
            elif args.objective == "forward_kl_simple":
                loss, objective_stats = forward_kl_simple_loss(
                    p_answer_logits,
                    teacher_targets,
                    teacher_probs=teacher_probs,
                    temperature=policy_temperature,
                    eps=args.eps,
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
                    eps=args.eps,
                )
                step_metrics = {
                    "train/loss": float(loss.item()),
                    "train/forward_kl": float(objective_stats["forward_kl"].mean().item()),
                    "train/teacher_ce": float(objective_stats["teacher_ce"].mean().item()),
                    "train/teacher_entropy": float(objective_stats["teacher_entropy"].mean().item()),
                }

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        for key, value in step_metrics.items():
            running_metrics[key] = running_metrics.get(key, 0.0) + value
        running_steps += 1

        if iter_num % args.log_interval == 0:
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
        prompt_bank_dir=args.prompt_bank_dir,
        task_name=args.task,
        device=device,
        autocast_context=autocast_context,
        eval_n=args.eval_n,
        eval_batch_size=args.eval_batch_size,
    )
    best_val_loss = min(best_val_loss, final_stats["val/loss"])
    save_eval_summary(out_dir, iter_num=iter_num, reason="final", metrics=final_stats)
    print(
        f"final step {iter_num}: val loss {final_stats['val/loss']:.4f}, "
        f"val cot_exact {final_stats['val/cot_exact']:.4f}, "
        f"val clean_full_exact {final_stats['val/clean_full_exact']:.4f}, "
        f"val clean_final_exact {final_stats['val/clean_final_exact']:.4f}"
    )
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


if __name__ == "__main__":
    main()
