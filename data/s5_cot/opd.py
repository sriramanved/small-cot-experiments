from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from data.s5_cot.prompt_bank import PromptBank, build_xy_from_prompt_and_target
from data.s5_cot.task import CORRUPTIBLE_IDS, DIGIT_END_ID, DIGIT_START_ID


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
) -> tuple[torch.Tensor, torch.Tensor]:
    seq = prompt_ids.to(device=device, dtype=torch.long, non_blocking=True)
    actions = torch.empty((seq.size(0), target_len), dtype=torch.long, device=device)

    for step in range(target_len):
        idx_cond = seq[:, -model.config.block_size:]
        with autocast_context:
            logits, _ = model(idx_cond)
        next_logits = logits[:, -1, :]
        if temperature > 0:
            probs = F.softmax(next_logits.float() / temperature, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_ids = torch.argmax(next_logits, dim=-1)
        actions[:, step] = next_ids
        seq = torch.cat((seq, next_ids.unsqueeze(1)), dim=1)

    return seq, actions


def gather_action_log_probs(
    logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    temperature: float | None = None,
) -> torch.Tensor:
    work_logits = logits.float()
    if temperature is not None and temperature > 0:
        work_logits = work_logits / temperature
    log_probs = F.log_softmax(work_logits, dim=-1)
    return log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)


def extract_answer_logits(
    full_logits: torch.Tensor,
    *,
    prompt_len: int,
    target_len: int,
) -> torch.Tensor:
    return full_logits[:, prompt_len - 1:prompt_len + target_len - 1, :]


def distributional_noisy_teacher_log_probs(
    clean_logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    eta: float,
    eps: float = 1e-10,
) -> torch.Tensor:
    clean_probs = F.softmax(clean_logits.float(), dim=-1)
    p_clean_action = clean_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    digit_ids = torch.tensor(CORRUPTIBLE_IDS, device=clean_logits.device, dtype=torch.long)
    digit_mass = clean_probs.index_select(2, digit_ids).sum(dim=-1)
    is_digit = (actions >= DIGIT_START_ID) & (actions <= DIGIT_END_ID)

    num_digits = len(CORRUPTIBLE_IDS)
    noisy_probs = torch.where(
        is_digit,
        (1.0 - eta) * p_clean_action + (eta / num_digits) * digit_mass,
        p_clean_action,
    )
    return torch.log(noisy_probs.clamp_min(eps))


def corrupted_greedy_teacher_log_probs(
    clean_logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    eta: float,
    eps: float = 1e-10,
) -> torch.Tensor:
    greedy_actions = torch.argmax(clean_logits, dim=-1)
    is_greedy_digit = (greedy_actions >= DIGIT_START_ID) & (greedy_actions <= DIGIT_END_ID)
    is_action_digit = (actions >= DIGIT_START_ID) & (actions <= DIGIT_END_ID)
    same_as_greedy = actions.eq(greedy_actions)

    probs = torch.zeros(actions.shape, device=clean_logits.device, dtype=torch.float32)
    probs = torch.where(
        (~is_greedy_digit) & same_as_greedy,
        torch.ones_like(probs),
        probs,
    )

    num_digits = len(CORRUPTIBLE_IDS)
    greedy_digit_prob = 1.0 - eta + (eta / num_digits)
    other_digit_prob = eta / num_digits

    probs = torch.where(
        is_greedy_digit & is_action_digit & same_as_greedy,
        torch.full_like(probs, greedy_digit_prob),
        probs,
    )
    probs = torch.where(
        is_greedy_digit & is_action_digit & (~same_as_greedy),
        torch.full_like(probs, other_digit_prob),
        probs,
    )
    return torch.log(probs.clamp_min(eps))


def compute_teacher_log_probs(
    clean_logits: torch.Tensor,
    actions: torch.Tensor,
    *,
    eta: float,
    teacher_law: str,
    eps: float = 1e-10,
) -> torch.Tensor:
    if teacher_law == "distributional_noise":
        return distributional_noisy_teacher_log_probs(
            clean_logits,
            actions,
            eta=eta,
            eps=eps,
        )
    if teacher_law == "corrupted_greedy":
        return corrupted_greedy_teacher_log_probs(
            clean_logits,
            actions,
            eta=eta,
            eps=eps,
        )
    raise ValueError(f"Unknown teacher_law: {teacher_law}")


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
