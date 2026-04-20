from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from data.modular_addition.prompt_bank import load_prompt_bank
from data.modular_addition.task import (
    corrupt_ids,
    corruptible_token_ids,
    get_batch,
    sample_cot_example_ids_from_rng,
)
from nanogpt.methods.student_prefix import compute_teacher_token_probs
from nanogpt.trainers.opd import validate_resume_metadata


class ModularAdditionTests(unittest.TestCase):
    def test_invalid_task_params_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "p must be >= 2"):
            sample_cot_example_ids_from_rng(__import__("random").Random(0), p=1, m=4)
        with self.assertRaisesRegex(ValueError, "m must be >= 1"):
            get_batch(batch_size=2, device="cpu", p=7, m=0)

    def test_sample_cot_example_ids_have_expected_running_sums(self):
        prompt_ids, cot_ids = sample_cot_example_ids_from_rng(__import__("random").Random(0), p=7, m=4)
        self.assertEqual(len(prompt_ids), 5)
        self.assertEqual(len(cot_ids), 4)
        self.assertEqual(prompt_ids[-1], 7)

        running = 0
        expected = []
        for token in prompt_ids[:-1]:
            running = (running + token) % 7
            expected.append(running)
        self.assertEqual(cot_ids, expected)

    def test_corrupt_ids_only_touches_residue_tokens(self):
        torch.manual_seed(0)
        ids = torch.tensor([[0, 1, 7, 2, 7]], dtype=torch.int32)
        corrupted = corrupt_ids(ids, 1.0, p=7)
        self.assertTrue(torch.equal(corrupted[:, 2], ids[:, 2]))
        self.assertTrue(torch.equal(corrupted[:, 4], ids[:, 4]))
        self.assertTrue(torch.all((corrupted[:, [0, 1, 3]] >= 0) & (corrupted[:, [0, 1, 3]] < 7)))

    def test_modadd_prompt_bank_meta_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            torch.save(torch.tensor([[1, 2, 7]], dtype=torch.int32), root / "clean_train_prompt_ids.pt")
            torch.save(torch.tensor([[1, 3]], dtype=torch.int32), root / "clean_train_cot_ids.pt")
            torch.save(torch.tensor([[2, 1, 7]], dtype=torch.int32), root / "clean_val_prompt_ids.pt")
            torch.save(torch.tensor([[2, 3]], dtype=torch.int32), root / "clean_val_cot_ids.pt")
            torch.save(torch.tensor([0], dtype=torch.long), root / "train_order.pt")
            with open(root / "meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "task": "modadd",
                        "p": 7,
                        "m": 2,
                        "prompt_len": 3,
                        "cot_len": 2,
                        "final_answer_len": 1,
                    },
                    f,
                )

            prompt_bank = load_prompt_bank(root)
            self.assertEqual(prompt_bank.task, "modadd")
            self.assertEqual(prompt_bank.p, 7)
            self.assertEqual(prompt_bank.m, 2)
            self.assertEqual(prompt_bank.prompt_len, 3)
            self.assertEqual(prompt_bank.cot_len, 2)
            self.assertEqual(prompt_bank.final_answer_len, 1)
            self.assertEqual(prompt_bank.token_dtype, torch.int32)
            self.assertEqual(prompt_bank.label_dtype, torch.int32)

    def test_modadd_distributional_teacher_probs_normalize(self):
        torch.manual_seed(0)
        clean_logits = torch.randn(2, 3, 8)
        teacher_probs = compute_teacher_token_probs(
            clean_logits,
            eta=0.2,
            teacher_law="distributional_noise",
            corruptible_token_ids=corruptible_token_ids(7),
        )
        torch.testing.assert_close(teacher_probs.sum(dim=-1), torch.ones(2, 3))

    def test_resume_metadata_rejects_task_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "task": "modadd",
                        "p": 7,
                        "m": 21,
                        "teacher_checkpoint": "teacher",
                        "prompt_bank_dir": "prompt_bank",
                        "subset_size": 4,
                        "eta": 0.1,
                        "teacher_law": "distributional_noise",
                        "objective": "forward_kl_full",
                        "student_temperature": 1.0,
                        "shuffle_prompts": False,
                        "seed": 123,
                    },
                    f,
                )
            with self.assertRaisesRegex(ValueError, "Resume mismatch for task"):
                validate_resume_metadata(
                    out_dir,
                    {
                        "task": "s5",
                        "p": 5,
                        "m": 21,
                        "teacher_checkpoint": "teacher",
                        "prompt_bank_dir": "prompt_bank",
                        "subset_size": 4,
                        "eta": 0.1,
                        "teacher_law": "distributional_noise",
                        "method_family": "nail",
                        "teacher_signal": "full",
                        "loss": "forward",
                        "shuffle_prompts": False,
                        "seed": 123,
                    },
                )


if __name__ == "__main__":
    unittest.main()
