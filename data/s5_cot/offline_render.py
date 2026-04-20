from __future__ import annotations

from pathlib import Path

import torch

from data.s5_cot.prompt_bank import PromptBank, load_prompt_bank, select_train_subset
from data.s5_cot.task import VOCAB_SIZE, CORRUPTIBLE_IDS
from data.s5_cot.task import corrupt_ids
from data.synthetic.offline_render import (
    DTYPE_LOOKUP,
    ROLLOUT_MODE_CHOICES,
    TARGET_MODE_CHOICES,
    build_dataset_meta as shared_build_dataset_meta,
    build_oracle_val_split,
    generate_teacher_targets as shared_generate_teacher_targets,
    load_hf_teacher,
    render_offline_dataset as shared_render_offline_dataset,
    render_train_split as shared_render_train_split,
    resolve_torch_dtype,
    save_rendered_dataset,
)
from nanogpt.methods.student_prefix import compute_teacher_token_probs


def generate_teacher_targets(
    model,
    prompt_ids: torch.Tensor,
    *,
    target_len: int,
    eta: float,
    rollout_mode: str,
    target_mode: str,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return shared_generate_teacher_targets(
        model,
        prompt_ids,
        target_len=target_len,
        eta=eta,
        rollout_mode=rollout_mode,
        target_mode=target_mode,
        device=device,
        corrupt_ids_fn=corrupt_ids,
        teacher_probs_fn=lambda clean_logits: compute_teacher_token_probs(
            clean_logits,
            eta=eta,
            teacher_law="distributional_noise",
            corruptible_token_ids=CORRUPTIBLE_IDS,
        ),
    )


def render_train_split(
    model,
    prompt_bank: PromptBank,
    subset_idx: torch.Tensor,
    *,
    eta: float,
    rollout_mode: str,
    target_mode: str,
    gen_batch_size: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return shared_render_train_split(
        model,
        prompt_bank,
        subset_idx,
        eta=eta,
        rollout_mode=rollout_mode,
        target_mode=target_mode,
        gen_batch_size=gen_batch_size,
        device=device,
        corrupt_ids_fn=corrupt_ids,
        teacher_probs_fn=lambda clean_logits: compute_teacher_token_probs(
            clean_logits,
            eta=eta,
            teacher_law="distributional_noise",
            corruptible_token_ids=CORRUPTIBLE_IDS,
        ),
    )


def build_dataset_meta(
    *,
    prompt_bank: PromptBank,
    prompt_bank_dir: str | Path,
    teacher_checkpoint: str | Path,
    subset_size: int,
    eta: float,
    rollout_mode: str,
    target_mode: str,
    gen_batch_size: int,
    device: str | torch.device,
    dtype_name: str | None,
    seed: int,
) -> dict[str, object]:
    extra_meta: dict[str, object] = {
        "vocab_size": VOCAB_SIZE,
        "train_target_type": target_mode,
    }
    if target_mode == "teacher_probs":
        extra_meta["teacher_law"] = "distributional_noise"
    return shared_build_dataset_meta(
        prompt_bank=prompt_bank,
        prompt_bank_dir=prompt_bank_dir,
        teacher_checkpoint=teacher_checkpoint,
        subset_size=subset_size,
        eta=eta,
        rollout_mode=rollout_mode,
        target_mode=target_mode,
        gen_batch_size=gen_batch_size,
        device=device,
        dtype_name=dtype_name,
        seed=seed,
        extra_meta=extra_meta,
    )


def render_offline_dataset(
    *,
    teacher_checkpoint: str | Path,
    prompt_bank_dir: str | Path,
    save_dir: str | Path,
    subset_size: int,
    eta: float,
    rollout_mode: str,
    target_mode: str,
    gen_batch_size: int,
    device: str | torch.device,
    dtype_name: str | None,
    seed: int,
) -> None:
    prompt_bank = load_prompt_bank(prompt_bank_dir)
    subset_idx = select_train_subset(prompt_bank, subset_size)
    extra_meta: dict[str, object] = {
        "vocab_size": VOCAB_SIZE,
        "train_target_type": target_mode,
    }
    if target_mode == "teacher_probs":
        extra_meta["teacher_law"] = "distributional_noise"
    shared_render_offline_dataset(
        teacher_checkpoint=teacher_checkpoint,
        prompt_bank_dir=prompt_bank_dir,
        save_dir=save_dir,
        subset_size=subset_size,
        eta=eta,
        rollout_mode=rollout_mode,
        target_mode=target_mode,
        gen_batch_size=gen_batch_size,
        device=device,
        dtype_name=dtype_name,
        seed=seed,
        prompt_bank=prompt_bank,
        subset_idx=subset_idx,
        corrupt_ids_fn=corrupt_ids,
        teacher_probs_fn=lambda clean_logits: compute_teacher_token_probs(
            clean_logits,
            eta=eta,
            teacher_law="distributional_noise",
            corruptible_token_ids=CORRUPTIBLE_IDS,
        ),
        extra_meta=extra_meta,
    )
