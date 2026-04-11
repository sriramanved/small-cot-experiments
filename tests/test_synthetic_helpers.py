from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import torch

from data.s5_cot.opd import compute_teacher_token_probs
from data.synthetic.offline_render import build_dataset_meta, generate_teacher_targets
from data.synthetic.prompt_bank import (
    PromptBank,
    build_xy_from_prompt_and_target,
    load_prompt_bank,
    select_train_subset,
)


class ConstantTeacherModel:
    def __init__(self, *, vocab_size: int, preferred_token: int):
        self.config = types.SimpleNamespace(vocab_size=vocab_size, use_cache=True)
        self.preferred_token = preferred_token

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, self.config.vocab_size), -20.0)
        logits[..., self.preferred_token] = 20.0
        return types.SimpleNamespace(logits=logits, past_key_values=())


class SyntheticHelperTests(unittest.TestCase):
    def test_build_xy_from_prompt_and_target_masks_prompt_positions_for_int32(self):
        prompt_ids = torch.tensor([[0, 1, 7]], dtype=torch.int32)
        target_ids = torch.tensor([[2, 3]], dtype=torch.int32)
        x, y = build_xy_from_prompt_and_target(prompt_ids, target_ids)

        self.assertEqual(x.dtype, torch.int32)
        self.assertEqual(y.dtype, torch.int32)
        torch.testing.assert_close(x, torch.tensor([[0, 1, 7, 2]], dtype=torch.int32))
        torch.testing.assert_close(y, torch.tensor([[-1, -1, 2, 3]], dtype=torch.int32))

    def test_select_train_subset_rejects_negative_size(self):
        prompt_bank = PromptBank(
            clean_train_prompt_ids=torch.zeros((3, 2), dtype=torch.uint8),
            clean_train_cot_ids=torch.zeros((3, 1), dtype=torch.uint8),
            clean_val_prompt_ids=torch.zeros((1, 2), dtype=torch.uint8),
            clean_val_cot_ids=torch.zeros((1, 1), dtype=torch.uint8),
            train_order=torch.arange(3, dtype=torch.long),
            meta={"task": "s5", "p": 5, "m": 1, "prompt_len": 2, "cot_len": 1, "final_answer_len": 1},
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            select_train_subset(prompt_bank, -1)

    def test_load_prompt_bank_defaults_s5_metadata_when_meta_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            torch.save(torch.tensor([[0, 1, 2]], dtype=torch.uint8), root / "clean_train_prompt_ids.pt")
            torch.save(torch.tensor([[3, 4]], dtype=torch.uint8), root / "clean_train_cot_ids.pt")
            torch.save(torch.tensor([[0, 1, 2]], dtype=torch.uint8), root / "clean_val_prompt_ids.pt")
            torch.save(torch.tensor([[3, 4]], dtype=torch.uint8), root / "clean_val_cot_ids.pt")
            torch.save(torch.tensor([0], dtype=torch.long), root / "train_order.pt")

            prompt_bank = load_prompt_bank(root)
            self.assertEqual(prompt_bank.task, "s5")
            self.assertEqual(prompt_bank.prompt_len, 3)
            self.assertEqual(prompt_bank.cot_len, 2)
            self.assertEqual(prompt_bank.p, 5)
            self.assertEqual(prompt_bank.final_answer_len, 7)

    def test_generate_teacher_targets_applies_corruption_fn_and_preserves_dtype(self):
        model = ConstantTeacherModel(vocab_size=5, preferred_token=1)
        prompt_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

        def bump_token(ids: torch.Tensor, eta: float) -> torch.Tensor:
            self.assertEqual(eta, 0.5)
            return ids + 1

        targets = generate_teacher_targets(
            model,
            prompt_ids,
            target_len=3,
            eta=0.5,
            rollout_mode="greedy_then_corrupt",
            device="cpu",
            corrupt_ids_fn=bump_token,
        )

        self.assertEqual(targets.dtype, torch.int32)
        torch.testing.assert_close(targets, torch.full((2, 3), 2, dtype=torch.int32))

    def test_build_dataset_meta_includes_task_shape_fields(self):
        prompt_bank = PromptBank(
            clean_train_prompt_ids=torch.zeros((2, 5), dtype=torch.int32),
            clean_train_cot_ids=torch.zeros((2, 4), dtype=torch.int32),
            clean_val_prompt_ids=torch.zeros((1, 5), dtype=torch.int32),
            clean_val_cot_ids=torch.zeros((1, 4), dtype=torch.int32),
            train_order=torch.arange(2, dtype=torch.long),
            meta={"task": "modadd", "p": 7, "m": 4, "prompt_len": 5, "cot_len": 4, "final_answer_len": 1},
        )
        meta = build_dataset_meta(
            prompt_bank=prompt_bank,
            prompt_bank_dir="prompt_bank",
            teacher_checkpoint="teacher",
            subset_size=2,
            eta=0.1,
            rollout_mode="greedy_then_corrupt",
            gen_batch_size=8,
            device="cpu",
            dtype_name="float32",
            seed=7,
        )
        self.assertEqual(meta["task"], "modadd")
        self.assertEqual(meta["p"], 7)
        self.assertEqual(meta["m"], 4)
        self.assertEqual(meta["prompt_len"], 5)
        self.assertEqual(meta["cot_len"], 4)
        self.assertEqual(meta["final_answer_len"], 1)

    def test_compute_teacher_token_probs_handles_custom_noncontiguous_corruptible_ids(self):
        clean_logits = torch.full((1, 1, 6), -10.0)
        clean_logits[..., 4] = 10.0
        teacher_probs = compute_teacher_token_probs(
            clean_logits,
            eta=0.3,
            teacher_law="corrupted_greedy",
            corruptible_token_ids=(1, 4),
        )
        expected = torch.zeros_like(teacher_probs)
        expected[..., 1] = 0.15
        expected[..., 4] = 0.85
        torch.testing.assert_close(teacher_probs, expected)


if __name__ == "__main__":
    unittest.main()
