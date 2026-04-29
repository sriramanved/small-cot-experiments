from __future__ import annotations

from pathlib import Path

import torch

from data.s5_cot.prompt_bank import (
    PromptBank,
    build_xy_from_prompt_and_target,
    load_prompt_bank,
    select_train_subset,
)
from data.s5_cot.semantic_key_noise import (
    SEMANTIC_KEY_NOISE_LAW,
    SemanticKeyNoiseConfig,
    eligible_token_ids_from_values,
    semantic_key_mask_for_step,
    semantic_key_noise_config_from_obj,
)
from data.s5_cot.task import VOCAB_SIZE, CORRUPTIBLE_IDS, LPAREN_ID, RPAREN_ID
from data.s5_cot.task import corrupt_ids
from data.synthetic.random_suffix_noise import (
    RANDOM_SUFFIX_AFTER_ERROR_LAW,
    RandomSuffixNoiseConfig,
    RandomSuffixStepSpec,
    generate_random_suffix_after_error_targets,
    make_random_suffix_generator,
    random_suffix_noise_config_from_obj,
    random_suffix_noise_meta,
    validate_random_suffix_applies_to_task,
)
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


TEACHER_LAW_CHOICES = (
    "distributional_noise",
    "corrupted_greedy",
    SEMANTIC_KEY_NOISE_LAW,
    RANDOM_SUFFIX_AFTER_ERROR_LAW,
)


def _semantic_config_from_random_suffix_config(
    config: RandomSuffixNoiseConfig,
) -> SemanticKeyNoiseConfig:
    return semantic_key_noise_config_from_obj(
        {
            "enabled": True,
            "coord_strategy": config.coord_strategy,
            "fixed_coord": config.fixed_coord,
            "seed": config.seed,
            "include_clean_value": True,
            "eligible_values": config.eligible_values,
            "apply_to": "partial_perm_image",
            "one_key_per_block": config.one_key_per_block,
        }
    )


def _build_random_suffix_step_spec_fn(config: RandomSuffixNoiseConfig):
    semantic_config = _semantic_config_from_random_suffix_config(config)

    def step_spec(step: int, prompt_ids: torch.Tensor, device: torch.device) -> RandomSuffixStepSpec:
        prompt_on_device = prompt_ids.to(device=device, dtype=torch.long)
        key_mask = semantic_key_mask_for_step(prompt_on_device, step, semantic_config)
        offset = int(step) % 7
        semantic = 1 <= offset <= 5
        scaffold_token = LPAREN_ID if offset == 0 else RPAREN_ID
        return RandomSuffixStepSpec(
            key_mask=key_mask,
            semantic_mask=torch.full(
                (prompt_ids.size(0),),
                semantic,
                dtype=torch.bool,
                device=device,
            ),
            scaffold_token_ids=torch.full(
                (prompt_ids.size(0),),
                scaffold_token,
                dtype=torch.long,
                device=device,
            ),
        )

    return step_spec


def _generate_random_suffix_targets(
    model,
    prompt_ids: torch.Tensor,
    *,
    target_len: int,
    eta: float,
    rollout_mode: str,
    target_mode: str,
    device: str | torch.device,
    random_suffix_noise_config=None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    config = random_suffix_noise_config_from_obj(random_suffix_noise_config)
    validate_random_suffix_applies_to_task(config, task_name="s5")
    eligible_token_ids = eligible_token_ids_from_values(config.eligible_values)
    return generate_random_suffix_after_error_targets(
        model,
        prompt_ids,
        target_len=target_len,
        eta=eta,
        rollout_mode=rollout_mode,
        target_mode=target_mode,
        device=device,
        config=config,
        eligible_token_ids=eligible_token_ids,
        step_spec_fn=_build_random_suffix_step_spec_fn(config),
        generator=generator,
    )


def _random_suffix_extra_meta(
    *,
    eta: float,
    rollout_mode: str,
    random_suffix_noise_config=None,
) -> dict[str, object]:
    config = random_suffix_noise_config_from_obj(random_suffix_noise_config)
    validate_random_suffix_applies_to_task(config, task_name="s5")
    eligible_token_ids = eligible_token_ids_from_values(config.eligible_values)
    return {
        "random_suffix_noise": random_suffix_noise_meta(
            config,
            eta=eta,
            task_name="s5",
            eligible_token_ids=eligible_token_ids,
        ),
        "requested_rollout_mode": rollout_mode,
        "train_decode_mode": f"{RANDOM_SUFFIX_AFTER_ERROR_LAW}_sample",
    }


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
    random_suffix_noise_config=None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if teacher_law not in TEACHER_LAW_CHOICES:
        raise ValueError(f"unknown teacher_law={teacher_law!r}")
    if teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        targets, teacher_probs, _ = _generate_random_suffix_targets(
            model,
            prompt_ids,
            target_len=target_len,
            eta=eta,
            rollout_mode=rollout_mode,
            target_mode=target_mode,
            device=device,
            random_suffix_noise_config=random_suffix_noise_config,
        )
        return targets, teacher_probs
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
    random_suffix_noise_config=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if teacher_law not in TEACHER_LAW_CHOICES:
        raise ValueError(f"unknown teacher_law={teacher_law!r}")
    if teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        return _render_random_suffix_train_split(
            model,
            prompt_bank,
            subset_idx,
            eta=eta,
            rollout_mode=rollout_mode,
            target_mode=target_mode,
            gen_batch_size=gen_batch_size,
            device=device,
            random_suffix_noise_config=random_suffix_noise_config,
        )
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
    random_suffix_noise_config=None,
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
    if teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        extra_meta.update(
            _random_suffix_extra_meta(
                eta=eta,
                rollout_mode=rollout_mode,
                random_suffix_noise_config=random_suffix_noise_config,
            )
        )
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
    random_suffix_noise_config=None,
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
    if teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        extra_meta.update(
            _random_suffix_extra_meta(
                eta=eta,
                rollout_mode=rollout_mode,
                random_suffix_noise_config=random_suffix_noise_config,
            )
        )
        model = load_hf_teacher(
            teacher_checkpoint,
            device=device,
            dtype_name=dtype_name,
        )
        train_x, train_y, train_teacher_probs = render_train_split(
            model,
            prompt_bank,
            subset_idx,
            eta=eta,
            rollout_mode=rollout_mode,
            target_mode=target_mode,
            gen_batch_size=gen_batch_size,
            device=device,
            teacher_law=teacher_law,
            random_suffix_noise_config=random_suffix_noise_config,
        )
        val_x, val_y = build_oracle_val_split(prompt_bank)
        meta = shared_build_dataset_meta(
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
        save_rendered_dataset(
            prompt_bank=prompt_bank,
            subset_idx=subset_idx,
            train_x=train_x,
            train_y=train_y,
            train_teacher_probs=train_teacher_probs,
            val_x=val_x,
            val_y=val_y,
            save_dir=save_dir,
            meta=meta,
        )
        return
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


def _render_random_suffix_train_split(
    model,
    prompt_bank: PromptBank,
    subset_idx: torch.Tensor,
    *,
    eta: float,
    rollout_mode: str,
    target_mode: str,
    gen_batch_size: int,
    device: str | torch.device,
    random_suffix_noise_config=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    config = random_suffix_noise_config_from_obj(random_suffix_noise_config)
    validate_random_suffix_applies_to_task(config, task_name="s5")
    subset_size = int(subset_idx.numel())
    target_len = prompt_bank.target_len
    train_x = torch.empty((subset_size, prompt_bank.xy_len), dtype=prompt_bank.token_dtype)
    train_y = torch.empty((subset_size, prompt_bank.xy_len), dtype=prompt_bank.label_dtype)
    train_teacher_probs = None
    if target_mode == "teacher_probs":
        train_teacher_probs = torch.empty(
            (subset_size, target_len, model.config.vocab_size),
            dtype=torch.float16,
        )
    generator = make_random_suffix_generator(device=device, seed=config.seed)

    for start in range(0, subset_size, gen_batch_size):
        end = min(start + gen_batch_size, subset_size)
        batch_idx = subset_idx[start:end]
        batch_prompt_ids = prompt_bank.clean_train_prompt_ids.index_select(0, batch_idx)
        batch_target_ids, batch_teacher_probs, _ = _generate_random_suffix_targets(
            model,
            batch_prompt_ids,
            target_len=target_len,
            eta=eta,
            rollout_mode=rollout_mode,
            target_mode=target_mode,
            device=device,
            random_suffix_noise_config=config,
            generator=generator,
        )
        batch_x, batch_y = build_xy_from_prompt_and_target(batch_prompt_ids, batch_target_ids)
        train_x[start:end] = batch_x
        train_y[start:end] = batch_y
        if train_teacher_probs is not None:
            if batch_teacher_probs is None:
                raise ValueError("batch_teacher_probs unexpectedly missing")
            train_teacher_probs[start:end] = batch_teacher_probs
        print(f"train: rendered {end}/{subset_size}")

    return train_x, train_y, train_teacher_probs
