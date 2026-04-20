from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

from data.s5_cot.prompt_bank import PromptBank, build_xy_from_prompt_and_target
from data.s5_cot.task import CORRUPTIBLE_IDS


DEFAULT_ROLLOUT_TEMPERATURE = {
    "opd": 1.0,
    "nail": 0.0,
}

LEGACY_OBJECTIVE_MAP = {
    "forward_kl_simple": ("mc", "forward"),
    "forward_kl_full": ("full", "forward"),
    "reverse_kl_simple": ("mc", "reverse"),
    "reverse_kl_tm": ("mc", "reverse"),
    "reverse_kl_full": ("full", "reverse"),
}


def _lookup(value: Mapping[str, Any] | object, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def default_rollout_temperature(method_family: str) -> float:
    try:
        return DEFAULT_ROLLOUT_TEMPERATURE[method_family]
    except KeyError as exc:
        raise ValueError(f"unknown method_family {method_family!r}") from exc


def legacy_objective_to_teacher_signal_loss(objective: str) -> tuple[str, str]:
    try:
        return LEGACY_OBJECTIVE_MAP[objective]
    except KeyError as exc:
        raise ValueError(f"unknown legacy objective {objective!r}") from exc


def resolve_method_family_for_reverse_loss(
    *,
    loss: str,
    rollout_temperature: float | None,
    default_method_family: str | None,
) -> str:
    if loss == "forward":
        return "nail"
    if default_method_family is not None:
        return default_method_family
    if rollout_temperature is not None and float(rollout_temperature) == 0.0:
        return "nail"
    return "opd"


def normalize_student_prefix_method(
    value: Mapping[str, Any] | object,
    *,
    default_method_family: str | None = None,
) -> dict[str, Any]:
    method_family = _lookup(value, "method_family") or default_method_family
    teacher_signal = _lookup(value, "teacher_signal")
    loss = _lookup(value, "loss")
    rollout_override = _optional_float(_lookup(value, "rollout_temperature_override"))
    loss_override = _optional_float(_lookup(value, "loss_temperature_override"))

    objective = _lookup(value, "objective")
    legacy_loss_temperature = _optional_float(_lookup(value, "student_temperature"))
    legacy_rollout_temperature = _optional_float(_lookup(value, "student_rollout_temperature"))

    if teacher_signal is None or loss is None:
        if objective in (None, ""):
            teacher_signal, loss = ("mc", "reverse")
        else:
            teacher_signal, loss = legacy_objective_to_teacher_signal_loss(str(objective))

    resolved_rollout = rollout_override
    if resolved_rollout is None:
        if legacy_rollout_temperature is not None:
            resolved_rollout = legacy_rollout_temperature
        elif objective not in (None, "") and legacy_loss_temperature is not None:
            # Older configs used one temperature knob for both rollout and loss.
            resolved_rollout = legacy_loss_temperature

    method_family = method_family or resolve_method_family_for_reverse_loss(
        loss=str(loss),
        rollout_temperature=resolved_rollout,
        default_method_family=default_method_family,
    )

    if resolved_rollout is None:
        resolved_rollout = default_rollout_temperature(str(method_family))

    resolved_loss = loss_override
    if resolved_loss is None and objective not in (None, "") and str(loss) == "forward":
        resolved_loss = legacy_loss_temperature

    return {
        "method_family": str(method_family),
        "teacher_signal": str(teacher_signal),
        "loss": str(loss),
        "rollout_temperature_override": rollout_override,
        "loss_temperature_override": loss_override,
        "resolved_rollout_temperature": float(resolved_rollout),
        "resolved_loss_temperature": None if resolved_loss is None else float(resolved_loss),
    }


def format_temperature_tag(temperature: float | None) -> str:
    if temperature is None:
        return "default"
    if float(temperature) == 0.0:
        return "greedy"
    return f"t{str(float(temperature)).replace('.', 'p')}"


class FixedPromptCycle:
    def __init__(
        self,
        prompt_ids: torch.Tensor,
        *,
        order: torch.Tensor | None = None,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 1337,
    ):
        self.prompt_ids = prompt_ids.to(device="cpu", dtype=torch.uint8).contiguous()
        if order is None:
            order = torch.arange(self.prompt_ids.size(0), dtype=torch.long)
        self.base_order = order.to(device="cpu", dtype=torch.long).contiguous()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.n = int(self.base_order.numel())
        self.epoch = 0
        self.pos = 0
        self.order = self._make_order()

    def _make_order(self) -> torch.Tensor:
        if not self.shuffle:
            return self.base_order.clone()
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=generator)
        return self.base_order.index_select(0, perm)

    def state_dict(self) -> dict[str, torch.Tensor | int | bool]:
        return {
            "base_order": self.base_order.clone(),
            "pos": int(self.pos),
            "n": int(self.n),
            "epoch": int(self.epoch),
            "batch_size": int(self.batch_size),
            "shuffle": bool(self.shuffle),
            "seed": int(self.seed),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | int | bool]) -> None:
        self.base_order = state["base_order"].to(device="cpu", dtype=torch.long).clone()
        self.pos = int(state["pos"])
        self.n = int(state["n"])
        self.epoch = int(state["epoch"])
        self.batch_size = int(state["batch_size"])
        self.shuffle = bool(state["shuffle"])
        self.seed = int(state["seed"])
        self.order = self._make_order()

    def _advance_epoch(self) -> None:
        self.epoch += 1
        self.pos = 0
        self.order = self._make_order()

    def has_remaining_in_epoch(self) -> bool:
        return self.pos < self.n

    def next_batch_no_wrap(self) -> torch.Tensor:
        if self.pos >= self.n:
            return self.prompt_ids[:0]
        take = min(self.batch_size, self.n - self.pos)
        idx = self.order[self.pos:self.pos + take]
        self.pos += take
        return self.prompt_ids.index_select(0, idx)

    def next_batch(self) -> torch.Tensor:
        batches = []
        remaining = self.batch_size

        while remaining > 0:
            if self.pos >= self.n:
                self._advance_epoch()

            take = min(remaining, self.n - self.pos)
            idx = self.order[self.pos:self.pos + take]
            batches.append(self.prompt_ids.index_select(0, idx))
            self.pos += take
            remaining -= take

        if len(batches) == 1:
            return batches[0]
        return torch.cat(batches, dim=0)


@torch.no_grad()
def rollout_student(
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
    past_key_values = None

    for step in range(target_len):
        with autocast_context:
            logits, _, past_key_values = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        next_logits = logits[:, -1, :]
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


def gather_action_log_probs(
    logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    temperature: float | None = None,
) -> torch.Tensor:
    log_probs = log_probs_from_logits(logits, temperature=temperature)
    return log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)


def log_probs_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float | None = None,
) -> torch.Tensor:
    work_logits = logits.float()
    if temperature is not None and temperature > 0:
        work_logits = work_logits / temperature
    return F.log_softmax(work_logits, dim=-1)


def extract_answer_logits(
    full_logits: torch.Tensor,
    *,
    prompt_len: int,
    target_len: int,
) -> torch.Tensor:
    return full_logits[:, prompt_len - 1:prompt_len + target_len - 1, :]


def distributional_noisy_teacher_probs(
    clean_logits: torch.Tensor,
    *,
    eta: float,
    corruptible_token_ids: tuple[int, ...] | list[int] = CORRUPTIBLE_IDS,
) -> torch.Tensor:
    clean_probs = F.softmax(clean_logits.float(), dim=-1)
    if eta <= 0:
        return clean_probs

    corruptible_ids = torch.as_tensor(corruptible_token_ids, dtype=torch.long, device=clean_probs.device)
    num_digits = int(corruptible_ids.numel())
    if num_digits == 0:
        return clean_probs

    selected_probs = clean_probs.index_select(dim=-1, index=corruptible_ids)
    noisy_probs = clean_probs.clone()
    noisy_selected_probs = (
        (1.0 - eta) * selected_probs
        + (eta / num_digits) * selected_probs.sum(dim=-1, keepdim=True)
    )
    noisy_probs.index_copy_(dim=-1, index=corruptible_ids, source=noisy_selected_probs)
    return noisy_probs


def corrupted_greedy_teacher_probs(
    clean_logits: torch.Tensor,
    *,
    eta: float,
    corruptible_token_ids: tuple[int, ...] | list[int] = CORRUPTIBLE_IDS,
) -> torch.Tensor:
    greedy_actions = torch.argmax(clean_logits, dim=-1)
    corruptible_ids = torch.as_tensor(corruptible_token_ids, dtype=torch.long, device=clean_logits.device)
    is_greedy_digit = greedy_actions.unsqueeze(-1).eq(
        corruptible_ids.view(*([1] * greedy_actions.ndim), -1)
    ).any(dim=-1)
    probs = torch.zeros_like(clean_logits, dtype=torch.float32)
    num_digits = int(corruptible_ids.numel())
    if num_digits > 0:
        uniform_noise = is_greedy_digit.unsqueeze(-1).to(dtype=torch.float32).expand(*greedy_actions.shape, num_digits)
        probs.index_copy_(dim=-1, index=corruptible_ids, source=uniform_noise * (eta / num_digits))
    base_mass = torch.where(
        is_greedy_digit,
        torch.full_like(greedy_actions, 1.0 - eta, dtype=torch.float32),
        torch.ones_like(greedy_actions, dtype=torch.float32),
    )
    probs.scatter_add_(2, greedy_actions.unsqueeze(-1), base_mass.unsqueeze(-1))
    return probs


def compute_teacher_token_probs(
    clean_logits: torch.Tensor,
    *,
    eta: float,
    teacher_law: str,
    corruptible_token_ids: tuple[int, ...] | list[int] = CORRUPTIBLE_IDS,
) -> torch.Tensor:
    if teacher_law == "distributional_noise":
        return distributional_noisy_teacher_probs(
            clean_logits,
            eta=eta,
            corruptible_token_ids=corruptible_token_ids,
        )
    if teacher_law == "corrupted_greedy":
        return corrupted_greedy_teacher_probs(
            clean_logits,
            eta=eta,
            corruptible_token_ids=corruptible_token_ids,
        )
    raise ValueError(f"Unknown teacher_law: {teacher_law}")


def compute_teacher_log_probs(
    clean_logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    eta: float,
    teacher_law: str,
    corruptible_token_ids: tuple[int, ...] | list[int] = CORRUPTIBLE_IDS,
    eps: float = 1e-10,
) -> torch.Tensor:
    teacher_probs = compute_teacher_token_probs(
        clean_logits,
        eta=eta,
        teacher_law=teacher_law,
        corruptible_token_ids=corruptible_token_ids,
    )
    teacher_action_probs = teacher_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    return torch.log(teacher_action_probs.clamp_min(eps))


def sample_teacher_actions(teacher_probs: torch.Tensor) -> torch.Tensor:
    batch_size, target_len, vocab_size = teacher_probs.shape
    flat_samples = torch.multinomial(
        teacher_probs.reshape(batch_size * target_len, vocab_size),
        num_samples=1,
    )
    return flat_samples.reshape(batch_size, target_len)


def teacher_cross_entropy(
    teacher_probs: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float | None = None,
) -> torch.Tensor:
    student_log_probs = log_probs_from_logits(student_logits, temperature=temperature)
    return -(teacher_probs * student_log_probs).sum(dim=-1)


def teacher_forward_kl(
    teacher_probs: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_log_probs = torch.log(teacher_probs.clamp_min(eps))
    ce = teacher_cross_entropy(
        teacher_probs,
        student_logits,
        temperature=temperature,
    )
    entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
    return ce - entropy, ce, entropy


def reverse_kl_tm_loss(
    student_logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    log_q: torch.Tensor,
    teacher_probs: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    teacher_action_probs = teacher_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    log_teacher = torch.log(teacher_action_probs.clamp_min(eps))
    advantage = log_teacher - log_q
    log_p = gather_action_log_probs(student_logits, actions)
    importance_weight = torch.exp(log_p - log_q.detach())
    loss = -(importance_weight * advantage.detach()).mean()
    return loss, {
        "log_p": log_p,
        "log_teacher": log_teacher,
        "advantage": advantage,
        "importance_weight": importance_weight,
    }


def reverse_kl_full_loss(
    student_logits: torch.Tensor,
    *,
    teacher_probs: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    student_probs = student_log_probs.exp()
    teacher_log_probs = torch.log(teacher_probs.clamp_min(eps))
    reverse_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    student_teacher_ce = -(student_probs * teacher_log_probs).sum(dim=-1)
    student_entropy = -(student_probs * student_log_probs).sum(dim=-1)
    loss = reverse_kl.mean()
    return loss, {
        "reverse_kl": reverse_kl,
        "student_teacher_ce": student_teacher_ce,
        "student_entropy": student_entropy,
    }


def forward_kl_simple_loss(
    student_logits: torch.Tensor,
    teacher_targets: torch.Tensor,
    *,
    teacher_probs: torch.Tensor,
    temperature: float | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    teacher_target_probs = teacher_probs.gather(2, teacher_targets.unsqueeze(-1)).squeeze(-1)
    log_teacher_target = torch.log(teacher_target_probs.clamp_min(eps))
    log_student_target = gather_action_log_probs(
        student_logits,
        teacher_targets,
        temperature=temperature,
    )
    loss = -log_student_target.mean()
    return loss, {
        "log_student_target": log_student_target,
        "log_teacher_target": log_teacher_target,
    }


def forward_kl_full_loss(
    student_logits: torch.Tensor,
    *,
    teacher_probs: torch.Tensor,
    temperature: float | None = None,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    token_kl, teacher_ce, teacher_entropy = teacher_forward_kl(
        teacher_probs,
        student_logits,
        temperature=temperature,
        eps=eps,
    )
    loss = token_kl.mean()
    return loss, {
        "forward_kl": token_kl,
        "teacher_ce": teacher_ce,
        "teacher_entropy": teacher_entropy,
    }


@torch.no_grad()
def cached_teacher_token_probs(
    model,
    prompt_ids: torch.Tensor,
    actions: torch.Tensor,
    *,
    eta: float,
    teacher_law: str,
    corruptible_token_ids: tuple[int, ...] | list[int] = CORRUPTIBLE_IDS,
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
    past_key_values = None

    for step in range(actions.size(1)):
        with autocast_context:
            logits, _, past_key_values = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        teacher_probs[:, step, :] = compute_teacher_token_probs(
            logits[:, -1:, :],
            eta=eta,
            teacher_law=teacher_law,
            corruptible_token_ids=corruptible_token_ids,
        ).squeeze(1)
        input_ids = actions[:, step:step + 1]

    return teacher_probs


@torch.no_grad()
def cached_teacher_log_probs(
    model,
    prompt_ids: torch.Tensor,
    actions: torch.Tensor,
    *,
    eta: float,
    teacher_law: str,
    corruptible_token_ids: tuple[int, ...] | list[int] = CORRUPTIBLE_IDS,
    eps: float,
    device: str | torch.device,
    autocast_context=nullcontext(),
) -> torch.Tensor:
    teacher_probs = cached_teacher_token_probs(
        model,
        prompt_ids,
        actions,
        eta=eta,
        teacher_law=teacher_law,
        corruptible_token_ids=corruptible_token_ids,
        device=device,
        autocast_context=autocast_context,
    )
    teacher_action_probs = teacher_probs.gather(2, actions.to(device=device).unsqueeze(-1)).squeeze(-1)
    return torch.log(teacher_action_probs.clamp_min(eps))


@torch.no_grad()
def evaluate_clean_ce_loss(
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
        y = y.to(device=device, dtype=torch.long, non_blocking=True)
        with autocast_context:
            _, loss = model(x, y)
        batch_n = end - start
        total_loss += float(loss.item()) * batch_n
        total_examples += batch_n

    return total_loss / max(total_examples, 1)
