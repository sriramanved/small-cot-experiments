from __future__ import annotations

import json
import random
import tempfile
import types
import unittest
from pathlib import Path

import torch

from nanogpt.methods.student_prefix import compute_teacher_token_probs
from data.s5_cot.task import (
    LPAREN_ID,
    RPAREN_ID,
    compose_perm,
    sample_cot_example_ids_from_rng as sample_s5_cot_example_ids_from_rng,
)
from data.synthetic.offline_render import (
    build_dataset_meta,
    generate_teacher_targets,
    render_train_split,
    resolve_offline_target_len,
)
from data.synthetic.prompt_bank import (
    PromptBank,
    build_xy_from_prompt_and_target,
    load_prompt_bank,
    select_train_subset,
)
from data.synthetic.target_spans import target_ids_from_y_row
from nanogpt.trainers.native_student_prefix import resolve_student_prefix_target_len


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
        torch.testing.assert_close(target_ids_from_y_row(y[0]), target_ids[0].long())

    def test_prompt_bank_target_len_aliases_cot_len_and_keeps_answer_suffix(self):
        prompt_bank = PromptBank(
            clean_train_prompt_ids=torch.tensor([[1, 2, 7]], dtype=torch.int32),
            clean_train_cot_ids=torch.tensor([[1, 3]], dtype=torch.int32),
            clean_val_prompt_ids=torch.tensor([[2, 1, 7]], dtype=torch.int32),
            clean_val_cot_ids=torch.tensor([[2, 3]], dtype=torch.int32),
            train_order=torch.arange(1, dtype=torch.long),
            meta={"task": "modadd", "p": 7, "m": 2, "prompt_len": 3, "cot_len": 2, "final_answer_len": 1},
        )

        self.assertEqual(prompt_bank.target_len, prompt_bank.cot_len)
        self.assertEqual(prompt_bank.answer_len, prompt_bank.final_answer_len)
        self.assertEqual(resolve_offline_target_len(prompt_bank), 2)
        self.assertEqual(resolve_student_prefix_target_len(prompt_bank), 2)

        _, y = build_xy_from_prompt_and_target(
            prompt_bank.clean_train_prompt_ids,
            prompt_bank.clean_train_cot_ids,
        )
        decoded_target = target_ids_from_y_row(y[0])
        torch.testing.assert_close(decoded_target, prompt_bank.clean_train_cot_ids[0].long())
        torch.testing.assert_close(
            decoded_target[-prompt_bank.final_answer_len:],
            torch.tensor([3], dtype=torch.long),
        )

    def test_s5_cot_len_includes_final_answer_suffix(self):
        prompt_ids, cot_ids = sample_s5_cot_example_ids_from_rng(random.Random(11), m=3)
        running = None
        for start in range(0, len(prompt_ids) - 1, 7):
            self.assertEqual(prompt_ids[start], LPAREN_ID)
            self.assertEqual(prompt_ids[start + 6], RPAREN_ID)
            sigma = tuple(int(token_id) - 2 for token_id in prompt_ids[start + 1:start + 6])
            running = sigma if running is None else compose_perm(running, sigma)
        expected_final = [LPAREN_ID, *(digit + 2 for digit in running), RPAREN_ID]
        self.assertEqual(cot_ids[-7:], expected_final)

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
            self.assertEqual(prompt_bank.target_len, 2)
            self.assertEqual(prompt_bank.p, 5)
            self.assertEqual(prompt_bank.final_answer_len, 7)

    def test_generate_teacher_targets_applies_corruption_fn_and_preserves_dtype(self):
        # Offline LogLossBC data is rendered once from the noisy teacher; this
        # checks that the renderer actually uses the configured corruption law.
        model = ConstantTeacherModel(vocab_size=5, preferred_token=1)
        prompt_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

        def bump_token(ids: torch.Tensor, eta: float) -> torch.Tensor:
            self.assertEqual(eta, 0.5)
            return ids + 1

        targets, teacher_probs = generate_teacher_targets(
            model,
            prompt_ids,
            target_len=3,
            eta=0.5,
            rollout_mode="greedy_then_corrupt",
            target_mode="tokens",
            device="cpu",
            corrupt_ids_fn=bump_token,
        )

        self.assertIsNone(teacher_probs)
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
            target_mode="tokens",
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
        self.assertEqual(meta["target_len"], 4)
        self.assertEqual(meta["final_answer_len"], 1)
        self.assertEqual(meta["answer_len"], 1)

    def test_offline_render_uses_canonical_target_len(self):
        # The rendered target span must match the prompt-bank target definition
        # used by both LogLossBC and student-prefix objectives.
        prompt_bank = PromptBank(
            clean_train_prompt_ids=torch.tensor([[0, 1, 7], [2, 3, 7]], dtype=torch.int32),
            clean_train_cot_ids=torch.tensor([[0, 1, 1], [2, 5, 5]], dtype=torch.int32),
            clean_val_prompt_ids=torch.tensor([[1, 1, 7]], dtype=torch.int32),
            clean_val_cot_ids=torch.tensor([[1, 2, 2]], dtype=torch.int32),
            train_order=torch.arange(2, dtype=torch.long),
            meta={"task": "modadd", "p": 7, "m": 3, "prompt_len": 3, "cot_len": 3, "final_answer_len": 1},
        )
        model = ConstantTeacherModel(vocab_size=8, preferred_token=4)
        train_x, train_y, teacher_probs = render_train_split(
            model,
            prompt_bank,
            torch.arange(2, dtype=torch.long),
            eta=0.0,
            rollout_mode="greedy_then_corrupt",
            target_mode="tokens",
            gen_batch_size=2,
            device="cpu",
            corrupt_ids_fn=lambda ids, eta: ids,
        )

        self.assertIsNone(teacher_probs)
        self.assertEqual(train_x.shape, (2, prompt_bank.prompt_len + prompt_bank.target_len - 1))
        self.assertEqual(train_y.shape, train_x.shape)
        self.assertEqual(int(train_y[0].ne(-1).sum().item()), prompt_bank.target_len)
        torch.testing.assert_close(
            target_ids_from_y_row(train_y[0]),
            torch.full((prompt_bank.target_len,), 4, dtype=torch.long),
        )

    def test_compute_teacher_token_probs_handles_custom_noncontiguous_corruptible_ids(self):
        # Teacher laws should respect task-provided corruptible token sets
        # instead of assuming contiguous S5-style value IDs.
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
