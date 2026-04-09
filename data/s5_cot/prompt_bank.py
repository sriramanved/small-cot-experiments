from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class PromptBank:
    clean_train_prompt_ids: torch.Tensor
    clean_train_cot_ids: torch.Tensor
    clean_val_prompt_ids: torch.Tensor
    clean_val_cot_ids: torch.Tensor
    train_order: torch.Tensor
    meta: dict[str, Any]

    @property
    def prompt_len(self) -> int:
        return int(self.clean_train_prompt_ids.size(1))

    @property
    def cot_len(self) -> int:
        return int(self.clean_train_cot_ids.size(1))

    @property
    def xy_len(self) -> int:
        return self.prompt_len + self.cot_len - 1

    @property
    def m(self) -> int:
        if "m" in self.meta:
            return int(self.meta["m"])
        return (self.prompt_len - 1) // 7


def load_prompt_bank(prompt_bank_dir: str | Path) -> PromptBank:
    prompt_bank_dir = Path(prompt_bank_dir)
    meta_path = prompt_bank_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return PromptBank(
        clean_train_prompt_ids=torch.load(prompt_bank_dir / "clean_train_prompt_ids.pt", map_location="cpu"),
        clean_train_cot_ids=torch.load(prompt_bank_dir / "clean_train_cot_ids.pt", map_location="cpu"),
        clean_val_prompt_ids=torch.load(prompt_bank_dir / "clean_val_prompt_ids.pt", map_location="cpu"),
        clean_val_cot_ids=torch.load(prompt_bank_dir / "clean_val_cot_ids.pt", map_location="cpu"),
        train_order=torch.load(prompt_bank_dir / "train_order.pt", map_location="cpu"),
        meta=meta,
    )


def select_train_subset(prompt_bank: PromptBank, subset_size: int) -> torch.Tensor:
    if subset_size > prompt_bank.clean_train_prompt_ids.size(0):
        raise ValueError(
            f"subset_size={subset_size} exceeds prompt bank size "
            f"{prompt_bank.clean_train_prompt_ids.size(0)}"
        )
    return prompt_bank.train_order[:subset_size]


def build_xy_from_prompt_and_target(
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = prompt_ids.to(dtype=torch.uint8)
    target_ids = target_ids.to(dtype=torch.uint8)
    seq = torch.cat((prompt_ids, target_ids), dim=1)
    x = seq[:, :-1].contiguous()
    y = seq[:, 1:].to(dtype=torch.int16).contiguous()
    y[:, :prompt_ids.size(1) - 1] = -1
    return x, y
