from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from data.s5_cot.opd import (
    FixedPromptCycle,
    compute_teacher_log_probs,
    evaluate_clean_ce_loss,
    extract_answer_logits,
    gather_action_log_probs,
    rollout_student,
)
from data.s5_cot.prompt_bank import load_prompt_bank, select_train_subset
from data.s5_cot.task import evaluate_saved_clean_s5_metrics
from hf_checkpoint import DTYPE_LOOKUP
from nanogpt_checkpoint import (
    build_nanogpt_model,
    load_nanogpt_checkpoint,
    load_nanogpt_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an S5 student from scratch with on-policy reverse-KL "
            "distillation against a noisy teacher derived from a clean expert."
        )
    )
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--prompt_bank_dir", type=str, required=True)
    parser.add_argument("--subset_size", type=int, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument(
        "--teacher_law",
        choices=("distributional_noise", "corrupted_greedy"),
        default="distributional_noise",
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
    parser.add_argument("--eps", type=float, default=1e-10)

    parser.add_argument("--wandb_log", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="small-cot-experiments")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


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
        "teacher_checkpoint",
        "prompt_bank_dir",
        "subset_size",
        "eta",
        "teacher_law",
        "student_temperature",
        "shuffle_prompts",
        "seed",
    ):
        if saved.get(key) != metadata.get(key):
            raise ValueError(
                f"Resume mismatch for {key}: saved={saved.get(key)!r} "
                f"current={metadata.get(key)!r}"
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

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config,
    )
    return wandb


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
    val_loss = evaluate_clean_ce_loss(
        model,
        prompt_bank,
        batch_size=eval_batch_size,
        device=device,
        autocast_context=autocast_context,
    )
    metrics = evaluate_saved_clean_s5_metrics(
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
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("train_TM_opd.py is single-GPU only in v1.")

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype, device)
    autocast_context = build_autocast_context(device, torch_dtype)
    scaler_device = "cuda" if "cuda" in device else "cpu"
    scaler = torch.amp.GradScaler(
        scaler_device,
        enabled=("cuda" in device and torch_dtype == torch.float16),
    )

    if args.wandb_run_name is None:
        eta_tag = str(args.eta).replace(".", "p")
        temp_tag = "greedy" if args.student_temperature == 0 else f"t{args.student_temperature}".replace(".", "p")
        args.wandb_run_name = (
            f"s5-tm-opd-n{args.subset_size}-eta{eta_tag}-"
            f"{args.teacher_law}-{temp_tag}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompt_bank = load_prompt_bank(args.prompt_bank_dir)
    subset_indices = select_train_subset(prompt_bank, args.subset_size)

    run_metadata = {
        "teacher_checkpoint": args.teacher_checkpoint,
        "prompt_bank_dir": args.prompt_bank_dir,
        "subset_size": args.subset_size,
        "eta": args.eta,
        "teacher_law": args.teacher_law,
        "student_temperature": args.student_temperature,
        "shuffle_prompts": args.shuffle_prompts,
        "seed": args.seed,
        "device": device,
        "dtype": str(torch_dtype).replace("torch.", ""),
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

    config = vars(args).copy()
    config.update(
        {
            "resolved_device": device,
            "resolved_dtype": str(torch_dtype).replace("torch.", ""),
            "prompt_len": prompt_bank.prompt_len,
            "cot_len": prompt_bank.cot_len,
        }
    )
    wandb = maybe_init_wandb(args, config)

    running_loss = 0.0
    running_advantage = 0.0
    running_log_q = 0.0
    running_log_teacher = 0.0
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
                device=device,
                autocast_context=autocast_context,
                eval_n=args.eval_n,
                eval_batch_size=args.eval_batch_size,
            )
            best_val_loss = min(best_val_loss, eval_stats["val/loss"])
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
            full_seq, actions = rollout_student(
                student,
                prompt_ids,
                target_len=prompt_bank.cot_len,
                temperature=args.student_temperature,
                device=device,
                autocast_context=autocast_context,
            )
            rollout_inputs = full_seq[:, :-1]
            with autocast_context:
                q_logits, _ = student(rollout_inputs, return_full_logits=True)
                teacher_logits, _ = teacher(rollout_inputs, return_full_logits=True)
            q_answer_logits = extract_answer_logits(
                q_logits,
                prompt_len=prompt_bank.prompt_len,
                target_len=prompt_bank.cot_len,
            )
            teacher_answer_logits = extract_answer_logits(
                teacher_logits,
                prompt_len=prompt_bank.prompt_len,
                target_len=prompt_bank.cot_len,
            )
            q_temperature = args.student_temperature if args.student_temperature > 0 else None
            log_q = gather_action_log_probs(
                q_answer_logits,
                actions,
                temperature=q_temperature,
            )
            log_teacher = compute_teacher_log_probs(
                teacher_answer_logits,
                actions,
                eta=args.eta,
                teacher_law=args.teacher_law,
                eps=args.eps,
            )
            advantage = log_teacher - log_q
        student.train()

        with autocast_context:
            p_logits, _ = student(rollout_inputs, return_full_logits=True)
            p_answer_logits = extract_answer_logits(
                p_logits,
                prompt_len=prompt_bank.prompt_len,
                target_len=prompt_bank.cot_len,
            )
            log_p = gather_action_log_probs(p_answer_logits, actions)
            importance_weight = torch.exp(log_p - log_q.detach())
            loss = -(importance_weight * advantage.detach()).mean()

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.item())
        running_advantage += float(advantage.mean().item())
        running_log_q += float(log_q.mean().item())
        running_log_teacher += float(log_teacher.mean().item())
        running_steps += 1

        if iter_num % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            denom = max(1, running_steps)
            train_stats = {
                "train/loss": running_loss / denom,
                "train/advantage": running_advantage / denom,
                "train/log_q": running_log_q / denom,
                "train/log_teacher": running_log_teacher / denom,
                "lr": lr,
                "iter": iter_num,
            }
            print(
                f"iter {iter_num}: loss {train_stats['train/loss']:.4f}, "
                f"advantage {train_stats['train/advantage']:.4f}, "
                f"log_q {train_stats['train/log_q']:.4f}, "
                f"log_teacher {train_stats['train/log_teacher']:.4f}, "
                f"time {dt*1000:.2f}ms"
            )
            if wandb is not None:
                wandb.log(train_stats)
            running_loss = 0.0
            running_advantage = 0.0
            running_log_q = 0.0
            running_log_teacher = 0.0
            running_steps = 0

        iter_num += 1

    final_stats = run_eval(
        model=student,
        prompt_bank=prompt_bank,
        prompt_bank_dir=args.prompt_bank_dir,
        device=device,
        autocast_context=autocast_context,
        eval_n=args.eval_n,
        eval_batch_size=args.eval_batch_size,
    )
    best_val_loss = min(best_val_loss, final_stats["val/loss"])
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
