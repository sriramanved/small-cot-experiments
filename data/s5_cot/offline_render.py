from __future__ import annotations

from pathlib import Path

import torch

from data.s5_cot.prompt_bank import PromptBank, load_prompt_bank, select_train_subset
from data.s5_cot.semantic_key_noise import (
    SEMANTIC_KEY_NOISE_LAW,
    eligible_token_ids_from_values,
    semantic_key_mask_for_step,
    semantic_key_noise_config_from_obj,
)
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


TEACHER_LAW_CHOICES = ("distributional_noise", "corrupted_greedy", SEMANTIC_KEY_NOISE_LAW)


def _build_step_teacher_probs_fn(
    *,
    eta: float,
    teacher_law: str,
    semantic_key_noise_config=None,
):
    semantic_config = None
    eligible_token_ids = None
    if teacher_law == SEMANTIC_KEY_NOISE_LAW:
        semantic_config = semantic_key_noise_config_from_obj(semantic_key_noise_config)
        eligible_token_ids = eligible_token_ids_from_values(semantic_config.eligible_values)

    def step_teacher_probs(clean_logits: torch.Tensor, step: int, prompt_ids: torch.Tensor) -> torch.Tensor:
        key_mask = None
        if semantic_config is not None:
            key_mask = semantic_key_mask_for_step(
                prompt_ids.to(device=clean_logits.device),
                step,
                semantic_config,
            )
        return compute_teacher_token_probs(
            clean_logits,
            eta=eta,
            teacher_law=teacher_law,
            corruptible_token_ids=CORRUPTIBLE_IDS,
            key_mask=key_mask,
            eligible_token_ids=eligible_token_ids,
        )

    return step_teacher_probs


def generate_teacher_targets(
    model,
    prompt_ids: torch.Tensor,
    *,
    target_len: int,
    eta: float,
    rollout_mode: str,
    target_mode: str,
    device: str | torch.device,
    teacher_law: str = "distributional_noise",
    semantic_key_noise_config=None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if teacher_law not in TEACHER_LAW_CHOICES:
        raise ValueError(f"unknown teacher_law={teacher_law!r}")
    step_teacher_probs_fn = None
    rollout_probs_step_fn = None
    if teacher_law == SEMANTIC_KEY_NOISE_LAW:
        step_teacher_probs_fn = _build_step_teacher_probs_fn(
            eta=eta,
            teacher_law=teacher_law,
            semantic_key_noise_config=semantic_key_noise_config,
        )
        rollout_probs_step_fn = step_teacher_probs_fn
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
            teacher_law=teacher_law,
            corruptible_token_ids=CORRUPTIBLE_IDS,
        ),
        teacher_probs_step_fn=step_teacher_probs_fn,
        rollout_probs_step_fn=rollout_probs_step_fn,
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
    teacher_law: str = "distributional_noise",
    semantic_key_noise_config=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if teacher_law not in TEACHER_LAW_CHOICES:
        raise ValueError(f"unknown teacher_law={teacher_law!r}")
    step_teacher_probs_fn = None
    rollout_probs_step_fn = None
    if teacher_law == SEMANTIC_KEY_NOISE_LAW:
        step_teacher_probs_fn = _build_step_teacher_probs_fn(
            eta=eta,
            teacher_law=teacher_law,
            semantic_key_noise_config=semantic_key_noise_config,
        )
        rollout_probs_step_fn = step_teacher_probs_fn
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
            teacher_law=teacher_law,
            corruptible_token_ids=CORRUPTIBLE_IDS,
        ),
        teacher_probs_step_fn=step_teacher_probs_fn,
        rollout_probs_step_fn=rollout_probs_step_fn,
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
    teacher_law: str = "distributional_noise",
    semantic_key_noise_config=None,
) -> dict[str, object]:
    extra_meta: dict[str, object] = {
        "vocab_size": VOCAB_SIZE,
        "train_target_type": target_mode,
        "teacher_law": teacher_law,
    }
    if teacher_law == SEMANTIC_KEY_NOISE_LAW:
        semantic_config = semantic_key_noise_config_from_obj(semantic_key_noise_config)
        extra_meta["semantic_key_noise"] = semantic_config.to_dict()
        extra_meta["requested_rollout_mode"] = rollout_mode
        extra_meta["train_decode_mode"] = "semantic_key_noise_sample"
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
    teacher_law: str = "distributional_noise",
    semantic_key_noise_config=None,
) -> None:
    if teacher_law not in TEACHER_LAW_CHOICES:
        raise ValueError(f"unknown teacher_law={teacher_law!r}")
    prompt_bank = load_prompt_bank(prompt_bank_dir)
    subset_idx = select_train_subset(prompt_bank, subset_size)
    extra_meta: dict[str, object] = {
        "vocab_size": VOCAB_SIZE,
        "train_target_type": target_mode,
        "teacher_law": teacher_law,
    }
    step_teacher_probs_fn = None
    rollout_probs_step_fn = None
    if teacher_law == SEMANTIC_KEY_NOISE_LAW:
        semantic_config = semantic_key_noise_config_from_obj(semantic_key_noise_config)
        step_teacher_probs_fn = _build_step_teacher_probs_fn(
            eta=eta,
            teacher_law=teacher_law,
            semantic_key_noise_config=semantic_config,
        )
        rollout_probs_step_fn = step_teacher_probs_fn
        extra_meta["semantic_key_noise"] = semantic_config.to_dict()
        extra_meta["requested_rollout_mode"] = rollout_mode
        extra_meta["train_decode_mode"] = "semantic_key_noise_sample"
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
            teacher_law=teacher_law,
            corruptible_token_ids=CORRUPTIBLE_IDS,
        ),
        teacher_probs_step_fn=step_teacher_probs_fn,
        rollout_probs_step_fn=rollout_probs_step_fn,
        extra_meta=extra_meta,
    )
