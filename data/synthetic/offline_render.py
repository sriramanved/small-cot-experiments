from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.synthetic.prompt_bank import PromptBank, build_xy_from_prompt_and_target
from hf_checkpoint import DTYPE_LOOKUP, load_nanogpt_checkpoint_as_hf


def resolve_torch_dtype(dtype_name: str | None, device: str | torch.device) -> torch.dtype:
    if dtype_name is not None:
        return DTYPE_LOOKUP[dtype_name]
    if "cuda" in str(device):
        return torch.float16
    return torch.float32


def load_hf_teacher(
    teacher_checkpoint: str | Path,
    *,
    device: str | torch.device,
    dtype_name: str | None,
):
    torch_dtype = resolve_torch_dtype(dtype_name, device)
    model = load_nanogpt_checkpoint_as_hf(
        teacher_checkpoint,
        map_location="cpu",
        device=device,
        torch_dtype=torch_dtype,
        eval_mode=True,
    )
    model.config.use_cache = True
    return model


ROLLOUT_MODE_CHOICES = ("greedy_then_corrupt", "sample_then_corrupt")


@torch.inference_mode()
def generate_teacher_targets(
    model,
    prompt_ids: torch.Tensor,
    *,
    target_len: int,
    eta: float,
    rollout_mode: str,
    device: str | torch.device,
    corrupt_ids_fn: Callable[[torch.Tensor, float], torch.Tensor],
) -> torch.Tensor:
    if rollout_mode not in ROLLOUT_MODE_CHOICES:
        raise ValueError(f"unknown rollout_mode={rollout_mode!r}")

    input_ids = prompt_ids.to(device=device, dtype=torch.long, non_blocking=True)
    generated = torch.empty(
        (prompt_ids.size(0), target_len),
        dtype=prompt_ids.dtype,
        device=device,
    )
    past_key_values = None

    for step in range(target_len):
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if rollout_mode == "greedy_then_corrupt":
            next_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        else:
            probs = torch.softmax(outputs.logits[:, -1, :].float(), dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
        next_ids = corrupt_ids_fn(next_ids, eta).to(dtype=prompt_ids.dtype)
        generated[:, step] = next_ids
        input_ids = next_ids.unsqueeze(1).to(dtype=torch.long)
        past_key_values = outputs.past_key_values

    return generated.to(device="cpu", dtype=prompt_ids.dtype)


def render_train_split(
    model,
    prompt_bank: PromptBank,
    subset_idx: torch.Tensor,
    *,
    eta: float,
    rollout_mode: str,
    gen_batch_size: int,
    device: str | torch.device,
    corrupt_ids_fn: Callable[[torch.Tensor, float], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    subset_size = int(subset_idx.numel())
    train_x = torch.empty((subset_size, prompt_bank.xy_len), dtype=prompt_bank.token_dtype)
    train_y = torch.empty((subset_size, prompt_bank.xy_len), dtype=prompt_bank.label_dtype)

    for start in range(0, subset_size, gen_batch_size):
        end = min(start + gen_batch_size, subset_size)
        batch_idx = subset_idx[start:end]
        batch_prompt_ids = prompt_bank.clean_train_prompt_ids.index_select(0, batch_idx)
        batch_target_ids = generate_teacher_targets(
            model,
            batch_prompt_ids,
            target_len=prompt_bank.cot_len,
            eta=eta,
            rollout_mode=rollout_mode,
            device=device,
            corrupt_ids_fn=corrupt_ids_fn,
        )
        batch_x, batch_y = build_xy_from_prompt_and_target(batch_prompt_ids, batch_target_ids)
        train_x[start:end] = batch_x
        train_y[start:end] = batch_y
        print(f"train: rendered {end}/{subset_size}")

    return train_x, train_y


def build_oracle_val_split(prompt_bank: PromptBank) -> tuple[torch.Tensor, torch.Tensor]:
    return build_xy_from_prompt_and_target(
        prompt_bank.clean_val_prompt_ids,
        prompt_bank.clean_val_cot_ids,
    )


def save_rendered_dataset(
    *,
    prompt_bank: PromptBank,
    subset_idx: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    save_dir: str | Path,
    meta: dict[str, Any],
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(train_x, save_dir / "train_x.pt")
    torch.save(train_y, save_dir / "train_y.pt")
    torch.save(val_x, save_dir / "val_x.pt")
    torch.save(val_y, save_dir / "val_y.pt")
    del train_x, train_y, val_x, val_y

    clean_train_prompt_ids = prompt_bank.clean_train_prompt_ids.index_select(0, subset_idx)
    clean_train_cot_ids = prompt_bank.clean_train_cot_ids.index_select(0, subset_idx)

    torch.save(subset_idx.to(device="cpu", dtype=torch.long), save_dir / "subset_indices.pt")
    torch.save(clean_train_prompt_ids, save_dir / "clean_train_prompt_ids.pt")
    torch.save(clean_train_cot_ids, save_dir / "clean_train_cot_ids.pt")
    torch.save(prompt_bank.clean_val_prompt_ids, save_dir / "clean_val_prompt_ids.pt")
    torch.save(prompt_bank.clean_val_cot_ids, save_dir / "clean_val_cot_ids.pt")

    with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def build_dataset_meta(
    *,
    prompt_bank: PromptBank,
    prompt_bank_dir: str | Path,
    teacher_checkpoint: str | Path,
    subset_size: int,
    eta: float,
    rollout_mode: str,
    gen_batch_size: int,
    device: str | torch.device,
    dtype_name: str | None,
    seed: int,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        "task": prompt_bank.task,
        "p": prompt_bank.p,
        "m": prompt_bank.m,
        "prompt_len": prompt_bank.prompt_len,
        "cot_len": prompt_bank.cot_len,
        "final_answer_len": prompt_bank.final_answer_len,
        "subset_size": subset_size,
        "eta": eta,
        "gen_batch_size": gen_batch_size,
        "device": str(device),
        "dtype": dtype_name,
        "seed": seed,
        "prompt_bank_dir": str(prompt_bank_dir),
        "teacher_checkpoint": str(teacher_checkpoint),
        "train_targets_source": "teacher_rollout_with_optional_eta_corruption",
        "train_decode_mode": rollout_mode,
        "val_targets_source": "fixed_clean_oracle",
        "nested_subset_order_saved": True,
    }
    if extra_meta:
        meta.update(extra_meta)
    return meta


def render_offline_dataset(
    *,
    teacher_checkpoint: str | Path,
    prompt_bank_dir: str | Path,
    save_dir: str | Path,
    subset_size: int,
    eta: float,
    rollout_mode: str,
    gen_batch_size: int,
    device: str | torch.device,
    dtype_name: str | None,
    seed: int,
    prompt_bank: PromptBank,
    subset_idx: torch.Tensor,
    corrupt_ids_fn: Callable[[torch.Tensor, float], torch.Tensor],
    extra_meta: dict[str, Any] | None = None,
) -> None:
    model = load_hf_teacher(
        teacher_checkpoint,
        device=device,
        dtype_name=dtype_name,
    )

    train_x, train_y = render_train_split(
        model,
        prompt_bank,
        subset_idx,
        eta=eta,
        rollout_mode=rollout_mode,
        gen_batch_size=gen_batch_size,
        device=device,
        corrupt_ids_fn=corrupt_ids_fn,
    )
    val_x, val_y = build_oracle_val_split(prompt_bank)

    meta = build_dataset_meta(
        prompt_bank=prompt_bank,
        prompt_bank_dir=prompt_bank_dir,
        teacher_checkpoint=teacher_checkpoint,
        subset_size=subset_size,
        eta=eta,
        rollout_mode=rollout_mode,
        gen_batch_size=gen_batch_size,
        device=device,
        dtype_name=dtype_name,
        seed=seed,
        extra_meta=extra_meta,
    )
    save_rendered_dataset(
        prompt_bank=prompt_bank,
        subset_idx=subset_idx,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        save_dir=save_dir,
        meta=meta,
    )
