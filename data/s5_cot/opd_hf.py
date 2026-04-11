from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from data.s5_cot.opd import compute_teacher_token_probs, gather_action_log_probs
from data.s5_cot.prompt_bank import PromptBank, build_xy_from_prompt_and_target

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

    for step in range(actions.size(1)):
        with autocast_context:
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        cache = outputs.past_key_values
        teacher_probs[:, step, :] = compute_teacher_token_probs(
            outputs.logits[:, -1:, :],
            eta=eta,
            teacher_law=teacher_law,
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
