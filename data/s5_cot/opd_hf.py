from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from data.s5_cot.prompt_bank import PromptBank, build_xy_from_prompt_and_target
from data.s5_cot.semantic_key_noise import (
    SEMANTIC_KEY_NOISE_LAW,
    S5_BLOCK_LEN,
    S5_NUM_COORDS,
    S5_VALUE_OFFSET,
    eligible_token_ids_from_values,
    semantic_key_mask,
    semantic_key_mask_for_step,
    semantic_key_noise_config_from_obj,
)
from data.s5_cot.task import LPAREN_ID, RPAREN_ID
from data.synthetic.random_suffix_noise import (
    RANDOM_SUFFIX_AFTER_ERROR_LAW,
    compute_poisoned_before,
    effective_trigger_eta,
    random_suffix_after_error_probs,
    random_suffix_noise_config_from_obj,
    validate_random_suffix_applies_to_task,
)
from nanogpt.methods.student_prefix import compute_teacher_token_probs, gather_action_log_probs

try:
    from transformers import StaticCache
except ImportError:  # pragma: no cover - transformers is expected, but keep a safe fallback.
    StaticCache = None


HF_IGNORE_INDEX = -100


def to_hf_labels(labels: torch.Tensor) -> torch.Tensor:
    hf_labels = labels.to(dtype=torch.long).clone()
    hf_labels[hf_labels < 0] = HF_IGNORE_INDEX
    return hf_labels


def _maybe_build_cache(
    model,
    *,
    max_cache_len: int,
):
    if StaticCache is None:
        return None
    return StaticCache(config=model.config, max_cache_len=max_cache_len)


@torch.no_grad()
def rollout_student_hf(
    model,
    prompt_ids: torch.Tensor,
    *,
    target_len: int,
    temperature: float,
    device: str | torch.device,
    autocast_context=nullcontext(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt = prompt_ids.to(device=device, dtype=torch.long, non_blocking=True)
    batch_size, prompt_len = prompt.shape
    full_seq = torch.empty((batch_size, prompt_len + target_len), dtype=torch.long, device=device)
    full_seq[:, :prompt_len] = prompt
    actions = full_seq[:, prompt_len:]
    log_q = torch.empty((batch_size, target_len), dtype=torch.float32, device=device)
    q_temperature = temperature if temperature > 0 else None
    input_ids = prompt
    cache = _maybe_build_cache(
        model,
        max_cache_len=prompt_len + target_len,
    )

    for step in range(target_len):
        with autocast_context:
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        next_logits = outputs.logits[:, -1, :]
        cache = outputs.past_key_values
        if temperature > 0:
            probs = F.softmax(next_logits.float() / temperature, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_ids = torch.argmax(next_logits, dim=-1)
        actions[:, step] = next_ids
        log_q[:, step] = gather_action_log_probs(
            next_logits.unsqueeze(1),
            next_ids.unsqueeze(1),
            temperature=q_temperature,
        ).squeeze(1)
        input_ids = next_ids.unsqueeze(1)

    return full_seq, actions, log_q


@torch.no_grad()
def cached_teacher_token_probs_hf(
    model,
    prompt_ids: torch.Tensor,
    actions: torch.Tensor,
    *,
    eta: float,
    teacher_law: str,
    semantic_key_noise_config: dict[str, object] | object | None = None,
    clean_target_ids: torch.Tensor | None = None,
    random_suffix_noise_config: dict[str, object] | object | None = None,
    device: str | torch.device,
    autocast_context=nullcontext(),
) -> torch.Tensor:
    prompt = prompt_ids.to(device=device, dtype=torch.long, non_blocking=True)
    actions = actions.to(device=device, dtype=torch.long, non_blocking=True)
    teacher_probs = torch.empty(
        (*actions.shape, model.config.vocab_size),
        dtype=torch.float32,
        device=device,
    )
    input_ids = prompt
    cache = _maybe_build_cache(
        model,
        max_cache_len=prompt.size(1) + actions.size(1),
    )
    semantic_config = None
    eligible_token_ids = None
    if teacher_law == SEMANTIC_KEY_NOISE_LAW:
        semantic_config = semantic_key_noise_config_from_obj(semantic_key_noise_config)
        eligible_token_ids = eligible_token_ids_from_values(semantic_config.eligible_values)
    random_suffix_config = None
    random_suffix_key_mask = None
    random_suffix_semantic_mask = None
    random_suffix_scaffold_token_ids = None
    random_suffix_eligible_token_ids = None
    poisoned_before = None
    random_suffix_eta = float(eta)
    if teacher_law == RANDOM_SUFFIX_AFTER_ERROR_LAW:
        random_suffix_config = random_suffix_noise_config_from_obj(random_suffix_noise_config)
        validate_random_suffix_applies_to_task(random_suffix_config, task_name="s5")
        if clean_target_ids is None:
            raise ValueError(
                f"{RANDOM_SUFFIX_AFTER_ERROR_LAW} HF teacher queries require "
                "clean_target_ids so poisoned prefixes can be inferred."
            )
        clean_targets = clean_target_ids.to(device=device, dtype=torch.long, non_blocking=True)
        if clean_targets.ndim != 2 or clean_targets.size(0) != actions.size(0):
            raise ValueError(
                "clean_target_ids must have shape [B, T] with the same batch size "
                f"as actions; got {tuple(clean_targets.shape)} for actions "
                f"{tuple(actions.shape)}"
            )
        if clean_targets.size(1) < actions.size(1):
            raise ValueError(
                f"clean_target_ids length {clean_targets.size(1)} is shorter than "
                f"actions length {actions.size(1)}"
            )
        clean_targets = clean_targets[:, :actions.size(1)]
        random_suffix_key_mask = semantic_key_mask(
            prompt,
            int(actions.size(1)),
            {
                "enabled": True,
                "coord_strategy": random_suffix_config.coord_strategy,
                "fixed_coord": random_suffix_config.fixed_coord,
                "seed": random_suffix_config.seed,
                "include_clean_value": True,
                "eligible_values": random_suffix_config.eligible_values,
                "apply_to": "partial_perm_image",
                "one_key_per_block": random_suffix_config.one_key_per_block,
            },
        ).to(device=device, dtype=torch.bool)
        offsets = torch.arange(
            int(actions.size(1)),
            dtype=torch.long,
            device=device,
        ) % S5_BLOCK_LEN
        semantic_row = (
            (offsets >= S5_VALUE_OFFSET)
            & (offsets < S5_VALUE_OFFSET + S5_NUM_COORDS)
        )
        scaffold_row = torch.where(
            offsets.eq(0),
            torch.full_like(offsets, LPAREN_ID),
            torch.full_like(offsets, RPAREN_ID),
        )
        random_suffix_semantic_mask = semantic_row.view(1, -1).expand_as(actions)
        random_suffix_scaffold_token_ids = scaffold_row.view(1, -1).expand_as(actions)
        random_suffix_eligible_token_ids = eligible_token_ids_from_values(
            random_suffix_config.eligible_values
        )
        poisoned_before = compute_poisoned_before(
            actions,
            clean_targets,
            random_suffix_key_mask,
        )
        random_suffix_eta = effective_trigger_eta(float(eta), random_suffix_config)

    for step in range(actions.size(1)):
        with autocast_context:
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        cache = outputs.past_key_values
        if random_suffix_config is not None:
            assert poisoned_before is not None
            assert random_suffix_key_mask is not None
            assert random_suffix_semantic_mask is not None
            assert random_suffix_scaffold_token_ids is not None
            assert random_suffix_eligible_token_ids is not None
            teacher_probs[:, step, :] = random_suffix_after_error_probs(
                F.softmax(outputs.logits[:, -1:, :].float(), dim=-1),
                eta=random_suffix_eta,
                poisoned=poisoned_before[:, step],
                key_mask=random_suffix_key_mask[:, step],
                semantic_mask=random_suffix_semantic_mask[:, step],
                eligible_token_ids=random_suffix_eligible_token_ids,
                scaffold_token_ids=random_suffix_scaffold_token_ids[:, step],
                keep_format_tokens=random_suffix_config.keep_format_tokens,
            ).squeeze(1)
        else:
            key_mask = None
            if semantic_config is not None:
                key_mask = semantic_key_mask_for_step(prompt, step, semantic_config)
            teacher_probs[:, step, :] = compute_teacher_token_probs(
                outputs.logits[:, -1:, :],
                eta=eta,
                teacher_law=teacher_law,
                key_mask=key_mask,
                eligible_token_ids=eligible_token_ids,
            ).squeeze(1)
        input_ids = actions[:, step:step + 1]

    return teacher_probs


@torch.no_grad()
def greedy_generate_target_ids_batched_hf(
    model,
    prompt_ids_batch: torch.Tensor,
    max_new_tokens: int,
    device: str | torch.device,
    autocast_context=nullcontext(),
) -> torch.Tensor:
    prompt = prompt_ids_batch.to(device=device, dtype=torch.long, non_blocking=True)
    generated = torch.empty((prompt.size(0), max_new_tokens), dtype=torch.long, device=device)
    input_ids = prompt
    cache = _maybe_build_cache(
        model,
        max_cache_len=prompt.size(1) + max_new_tokens,
    )

    for step in range(max_new_tokens):
        with autocast_context:
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        cache = outputs.past_key_values
        next_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated[:, step] = next_id
        input_ids = next_id.unsqueeze(1)

    return generated.to(device="cpu", dtype=torch.long)


@torch.no_grad()
def teacher_forced_exact_batch_hf(
    model,
    prompt_ids_batch: torch.Tensor,
    cot_ids_batch: torch.Tensor,
    *,
    device: str | torch.device,
    autocast_context=nullcontext(),
) -> torch.Tensor:
    seq = torch.cat((prompt_ids_batch, cot_ids_batch), dim=1).to(device=device, dtype=torch.long, non_blocking=True)
    x = seq[:, :-1].contiguous()
    y = seq[:, 1:].clone()
    y[:, :prompt_ids_batch.size(1) - 1] = -1
    hf_y = to_hf_labels(y)
    with autocast_context:
        outputs = model(
            input_ids=x,
            labels=hf_y,
            use_cache=False,
        )
    pred = outputs.logits.argmax(dim=-1)
    mask = hf_y != HF_IGNORE_INDEX
    return torch.logical_or(pred.eq(hf_y), ~mask).all(dim=1).to(device="cpu")


@torch.no_grad()
def evaluate_saved_clean_s5_metrics_hf(
    model,
    *,
    device: str | torch.device,
    data_dir: str,
    n_eval: int | None = None,
    batch_size: int = 256,
    autocast_context=nullcontext(),
) -> dict[str, float]:
    prompt_ids_all = torch.load(f"{data_dir}/clean_val_prompt_ids.pt", map_location="cpu").long()
    cot_ids_all = torch.load(f"{data_dir}/clean_val_cot_ids.pt", map_location="cpu").long()

    if n_eval is not None:
        prompt_ids_all = prompt_ids_all[:n_eval]
        cot_ids_all = cot_ids_all[:n_eval]

    tf_full_ok = 0
    ar_full_ok = 0
    ar_final_ok = 0
    n = prompt_ids_all.size(0)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        prompt_ids_batch = prompt_ids_all[start:end]
        cot_ids_batch = cot_ids_all[start:end]

        tf_ok = teacher_forced_exact_batch_hf(
            model,
            prompt_ids_batch,
            cot_ids_batch,
            device=device,
            autocast_context=autocast_context,
        )
        tf_full_ok += int(tf_ok.sum().item())

        pred_ids_batch = greedy_generate_target_ids_batched_hf(
            model,
            prompt_ids_batch,
            cot_ids_batch.size(1),
            device=device,
            autocast_context=autocast_context,
        )
        ar_full_ok += int(pred_ids_batch.eq(cot_ids_batch).all(dim=1).sum().item())
        ar_final_ok += int(pred_ids_batch[:, -7:].eq(cot_ids_batch[:, -7:]).all(dim=1).sum().item())

    return {
        "cot_exact": tf_full_ok / n,
        "clean_full_exact": ar_full_ok / n,
        "clean_final_exact": ar_final_ok / n,
    }


@torch.no_grad()
def evaluate_clean_ce_loss_hf(
    model,
    prompt_bank: PromptBank,
    *,
    batch_size: int,
    device: str | torch.device,
    autocast_context=nullcontext(),
) -> float:
    total_loss = 0.0
    total_examples = 0

    for start in range(0, prompt_bank.clean_val_prompt_ids.size(0), batch_size):
        end = min(start + batch_size, prompt_bank.clean_val_prompt_ids.size(0))
        prompt_ids = prompt_bank.clean_val_prompt_ids[start:end]
        cot_ids = prompt_bank.clean_val_cot_ids[start:end]
        x, y = build_xy_from_prompt_and_target(prompt_ids, cot_ids)
        x = x.to(device=device, dtype=torch.long, non_blocking=True)
        hf_y = to_hf_labels(y).to(device=device, non_blocking=True)
        with autocast_context:
            outputs = model(
                input_ids=x,
                labels=hf_y,
                use_cache=False,
            )
        batch_n = end - start
        total_loss += float(outputs.loss.item()) * batch_n
        total_examples += batch_n

    return total_loss / max(total_examples, 1)
