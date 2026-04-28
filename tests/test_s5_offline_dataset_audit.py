from __future__ import annotations

import json
import random
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from data.s5_cot.task import LPAREN_ID, RPAREN_ID, sample_cot_example_ids_from_rng
from data.synthetic.prompt_bank import build_xy_from_prompt_and_target
from scripts.audit_s5_offline_dataset_accuracy import (
    audit_dataset,
    compute_s5_offline_audit_metrics,
    final_answer_exact,
    resolved_config,
    s5_dataset_name,
    s5_final_answer_ids,
)
from data.synthetic.offline_render import generate_teacher_targets


def _target_row() -> torch.Tensor:
    return torch.tensor(
        [
            LPAREN_ID,
            3,
            4,
            5,
            6,
            7,
            RPAREN_ID,
            LPAREN_ID,
            7,
            6,
            5,
            4,
            3,
            RPAREN_ID,
        ],
        dtype=torch.uint8,
    )


class FeedbackTeacherModel:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(vocab_size=8)
        self.calls = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        self.calls += 1
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, self.config.vocab_size), -20.0)
        if self.calls == 1:
            next_token = torch.ones(batch_size, dtype=torch.long)
        else:
            last_token = input_ids[:, -1].to(dtype=torch.long)
            next_token = torch.where(last_token.eq(4), torch.full_like(last_token, 2), torch.full_like(last_token, 3))
        logits[torch.arange(batch_size), -1, next_token] = 20.0
        return types.SimpleNamespace(logits=logits, past_key_values=())


class S5OfflineDatasetAuditTests(unittest.TestCase):
    def test_final_answer_extraction_uses_s5_suffix(self):
        target = _target_row()
        torch.testing.assert_close(
            s5_final_answer_ids(target, final_answer_len=7),
            torch.tensor([LPAREN_ID, 7, 6, 5, 4, 3, RPAREN_ID], dtype=torch.uint8),
        )
        self.assertTrue(final_answer_exact(target.unsqueeze(0), target.unsqueeze(0), final_answer_len=7)[0].item())

    def test_clean_targets_self_compare_exactly(self):
        clean = torch.stack([_target_row(), _target_row()])
        metrics = compute_s5_offline_audit_metrics(
            clean,
            clean,
            eta=0.0,
            final_answer_len=7,
            chunk_size=1,
        )
        self.assertEqual(metrics["noisy_final_exact"], 1.0)
        self.assertEqual(metrics["noisy_full_exact"], 1.0)
        self.assertEqual(metrics["token_match_rate"], 1.0)
        self.assertEqual(metrics["empirical_corruption_rate_corruptible"], 0.0)

    def test_corruptible_digit_differences_are_counted_without_punctuation_mismatch(self):
        clean = torch.stack([_target_row(), _target_row()])
        noisy = clean.clone()
        noisy[0, 2] = 7
        noisy[1, 9] = 3
        metrics = compute_s5_offline_audit_metrics(
            noisy,
            clean,
            eta=0.5,
            final_answer_len=7,
            chunk_size=1,
        )
        self.assertLess(metrics["corruptible_token_match_rate"], 1.0)
        self.assertEqual(metrics["noncorruptible_token_match_rate"], 1.0)
        self.assertGreater(metrics["empirical_corruption_rate_corruptible"], 0.0)

    def test_renderer_feeds_corrupted_token_back_into_next_teacher_query(self):
        model = FeedbackTeacherModel()
        calls = []

        def corrupt_first_step(ids: torch.Tensor, eta: float) -> torch.Tensor:
            calls.append(ids.detach().cpu().clone())
            if len(calls) == 1:
                return torch.full_like(ids, 4)
            return ids

        targets, _ = generate_teacher_targets(
            model,
            torch.tensor([[0]], dtype=torch.long),
            target_len=2,
            eta=1.0,
            rollout_mode="greedy_then_corrupt",
            target_mode="tokens",
            device="cpu",
            corrupt_ids_fn=corrupt_first_step,
        )

        torch.testing.assert_close(targets, torch.tensor([[4, 2]], dtype=torch.long))

    def test_hydra_style_dataset_name_matches_s5_resolver_shape(self):
        self.assertEqual(
            s5_dataset_name(
                m=21,
                subset_size=1_000_000,
                eta=0.5,
                rollout_mode="greedy_then_corrupt",
                target_mode="tokens",
                render_seed=20260417,
            ),
            "s5_noisy_offline_seed20260417_m21_n1000000_eta_0p5",
        )

    def test_resolved_config_accepts_existing_dataset_name_without_eta(self):
        cfg = resolved_config(
            Namespace(
                data_dir=None,
                prompt_bank_dir=None,
                eta=None,
                num_examples=3,
                examples_start=0,
                max_rows=None,
                chunk_size=8,
                output_dir=None,
                json_out=None,
                csv_out=None,
                no_json=False,
                no_examples=False,
                overrides=["task.dataset=s5_noisy_offline_sample_then_corrupt_n8000000_eta_0p1"],
            )
        )
        self.assertEqual(
            cfg["data_dir"],
            "data/s5_noisy_offline_sample_then_corrupt_n8000000_eta_0p1",
        )
        self.assertEqual(cfg["num_examples"], 3)

    def test_audit_dataset_loads_rendered_layout_and_prompt_bank_from_meta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_bank_dir = root / "prompt_bank"
            dataset_dir = root / "dataset"
            prompt_bank_dir.mkdir()
            dataset_dir.mkdir()

            rng = random.Random(19)
            train_prompts = []
            train_cots = []
            val_prompts = []
            val_cots = []
            for _ in range(2):
                prompt_ids, cot_ids = sample_cot_example_ids_from_rng(rng, m=1)
                train_prompts.append(prompt_ids)
                train_cots.append(cot_ids)
            prompt_ids, cot_ids = sample_cot_example_ids_from_rng(rng, m=1)
            val_prompts.append(prompt_ids)
            val_cots.append(cot_ids)

            clean_train_prompt = torch.tensor(train_prompts, dtype=torch.uint8)
            clean_train_cot = torch.tensor(train_cots, dtype=torch.uint8)
            clean_val_prompt = torch.tensor(val_prompts, dtype=torch.uint8)
            clean_val_cot = torch.tensor(val_cots, dtype=torch.uint8)
            train_order = torch.tensor([1, 0], dtype=torch.long)

            torch.save(clean_train_prompt, prompt_bank_dir / "clean_train_prompt_ids.pt")
            torch.save(clean_train_cot, prompt_bank_dir / "clean_train_cot_ids.pt")
            torch.save(clean_val_prompt, prompt_bank_dir / "clean_val_prompt_ids.pt")
            torch.save(clean_val_cot, prompt_bank_dir / "clean_val_cot_ids.pt")
            torch.save(train_order, prompt_bank_dir / "train_order.pt")
            prompt_meta = {
                "task": "s5",
                "m": 1,
                "n_train": 2,
                "n_val": 1,
                "prompt_len": 8,
                "cot_len": 7,
                "target_len": 7,
                "final_answer_len": 7,
                "answer_len": 7,
            }
            (prompt_bank_dir / "meta.json").write_text(json.dumps(prompt_meta), encoding="utf-8")

            subset_prompt = clean_train_prompt.index_select(0, train_order)
            subset_clean_cot = clean_train_cot.index_select(0, train_order)
            noisy_cot = subset_clean_cot.clone()
            noisy_cot[0, 2] = 3 if int(noisy_cot[0, 2].item()) != 3 else 4
            train_x, train_y = build_xy_from_prompt_and_target(subset_prompt, noisy_cot)
            val_x, val_y = build_xy_from_prompt_and_target(clean_val_prompt, clean_val_cot)

            torch.save(train_x, dataset_dir / "train_x.pt")
            torch.save(train_y, dataset_dir / "train_y.pt")
            torch.save(val_x, dataset_dir / "val_x.pt")
            torch.save(val_y, dataset_dir / "val_y.pt")
            torch.save(train_order, dataset_dir / "subset_indices.pt")
            torch.save(subset_prompt, dataset_dir / "clean_train_prompt_ids.pt")
            torch.save(subset_clean_cot, dataset_dir / "clean_train_cot_ids.pt")
            torch.save(clean_val_prompt, dataset_dir / "clean_val_prompt_ids.pt")
            torch.save(clean_val_cot, dataset_dir / "clean_val_cot_ids.pt")
            dataset_meta = {
                "eta": 0.5,
                "seed": 7,
                "prompt_bank_dir": str(prompt_bank_dir),
                "train_targets_source": "teacher_rollout_with_optional_eta_corruption",
                "val_targets_source": "fixed_clean_oracle",
            }
            (dataset_dir / "meta.json").write_text(json.dumps(dataset_meta), encoding="utf-8")

            report = audit_dataset(
                {
                    "data_dir": str(dataset_dir),
                    "prompt_bank_dir": str(root / "wrong_default_prompt_bank"),
                    "prompt_bank_dir_explicit": False,
                    "eta": 0.5,
                    "bank_seed": 19,
                    "teacher_seed": 7,
                    "render_seed": 7,
                    "chunk_size": 1,
                    "no_examples": True,
                    "num_examples": 0,
                    "examples_start": 0,
                    "max_rows": None,
                }
            )

            self.assertEqual(report["prompt_bank_dir"], str(prompt_bank_dir))
            self.assertEqual(report["target_lengths"]["prompt_len"], 8)
            self.assertEqual(report["target_lengths"]["target_len"], 7)
            self.assertLess(report["metrics"]["noisy_full_exact"], 1.0)
            self.assertTrue(report["validation_consistency"]["same_clean_prompt_bank"])
            for check in report["validation_consistency"]["checks"]:
                self.assertTrue(check["ok"], check["name"])


if __name__ == "__main__":
    unittest.main()
