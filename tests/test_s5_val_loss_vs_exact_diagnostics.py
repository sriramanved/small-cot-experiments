from __future__ import annotations

import math
import unittest

import torch

from data.s5_cot.task import CORRUPTIBLE_IDS, EQUALS_ID, LPAREN_ID, RPAREN_ID
from data.s5_cot.validation_diagnostics import (
    _token_stats_from_logits,
    deterministic_uniform_corruption_entropy,
    extract_supervised_targets,
    noisy_teacher_entropy_baseline,
)
from data.synthetic.prompt_bank import build_xy_from_prompt_and_target
from model import causal_lm_loss


class S5ValLossVsExactDiagnosticsTests(unittest.TestCase):
    def test_offline_bc_prompt_tokens_are_masked_in_validation_loss(self):
        prompt_ids = torch.tensor([[LPAREN_ID, CORRUPTIBLE_IDS[0], EQUALS_ID]], dtype=torch.long)
        target_ids = torch.tensor([[CORRUPTIBLE_IDS[1], RPAREN_ID]], dtype=torch.long)
        _, y = build_xy_from_prompt_and_target(prompt_ids, target_ids)

        targets, target_start = extract_supervised_targets(y)
        self.assertEqual(target_start, prompt_ids.size(1) - 1)
        torch.testing.assert_close(targets, target_ids)
        self.assertTrue(y[:, :target_start].eq(-1).all().item())

        vocab_size = 8
        logits = torch.zeros((1, y.size(1), vocab_size), dtype=torch.float32)
        logits[:, :target_start, 0] = 100.0
        logits[:, target_start:, :] = -5.0
        logits[0, target_start, int(target_ids[0, 0])] = 5.0
        logits[0, target_start + 1, int(target_ids[0, 1])] = 5.0

        masked_loss = causal_lm_loss(logits, y, ignore_index=-1)
        suffix_loss = torch.nn.functional.cross_entropy(
            logits[:, target_start:, :].reshape(-1, vocab_size),
            target_ids.reshape(-1),
        )
        torch.testing.assert_close(masked_loss, suffix_loss)

    def test_val_loss_helper_is_clean_one_hot_ce_on_supervised_suffix(self):
        prompt_ids = torch.tensor([[LPAREN_ID, CORRUPTIBLE_IDS[0], EQUALS_ID]], dtype=torch.long)
        target_ids = torch.tensor([[CORRUPTIBLE_IDS[2], RPAREN_ID]], dtype=torch.long)
        _, y = build_xy_from_prompt_and_target(prompt_ids, target_ids)
        vocab_size = 8
        logits = torch.randn((1, y.size(1), vocab_size), generator=torch.Generator().manual_seed(3))

        metrics, _ = _token_stats_from_logits(
            logits,
            y,
            final_answer_len=2,
            loss_threshold=1.0,
        )
        expected = causal_lm_loss(logits, y, ignore_index=-1)
        self.assertAlmostEqual(metrics["clean_ce_loss"], float(expected.item()), places=6)

    def test_clean_full_exact_span_matches_validation_loss_span(self):
        prompt_ids = torch.tensor([[LPAREN_ID, CORRUPTIBLE_IDS[0], EQUALS_ID]], dtype=torch.long)
        cot_ids = torch.tensor([[CORRUPTIBLE_IDS[3], RPAREN_ID, LPAREN_ID]], dtype=torch.long)
        _, y = build_xy_from_prompt_and_target(prompt_ids, cot_ids)
        supervised_targets, _ = extract_supervised_targets(y)

        # Existing clean_full_exact compares greedy generated targets to clean_val_cot_ids.
        # The validation CE target suffix extracted from Y is that same clean CoT tensor.
        torch.testing.assert_close(supervised_targets, cot_ids)

    def test_final_answer_suffix_mask_is_last_tokens(self):
        y = torch.tensor([[-1, -1, 3, 4, 5, 6]], dtype=torch.long)
        logits = torch.zeros((1, y.size(1), 8), dtype=torch.float32)
        metrics, tensors = _token_stats_from_logits(
            logits,
            y,
            final_answer_len=2,
            loss_threshold=1.0,
        )

        torch.testing.assert_close(
            tensors["final_mask"],
            torch.tensor([[False, False, True, True]]),
        )
        expected_uniform_loss = math.log(8)
        self.assertAlmostEqual(metrics["final_answer_loss"], expected_uniform_loss, places=6)
        self.assertAlmostEqual(metrics["cot_body_loss"], expected_uniform_loss, places=6)

    def test_teacher_forced_token_acc_can_be_one_while_ce_is_nonzero(self):
        prompt_ids = torch.tensor([[LPAREN_ID, CORRUPTIBLE_IDS[0], EQUALS_ID]], dtype=torch.long)
        target_ids = torch.tensor([[CORRUPTIBLE_IDS[1], CORRUPTIBLE_IDS[2]]], dtype=torch.long)
        _, y = build_xy_from_prompt_and_target(prompt_ids, target_ids)
        vocab_size = 8
        logits = torch.zeros((1, y.size(1), vocab_size), dtype=torch.float32)
        target_start = prompt_ids.size(1) - 1
        for pos, target in enumerate(target_ids[0].tolist(), start=target_start):
            logits[0, pos, :] = -2.0
            logits[0, pos, int(target)] = 2.0
            logits[0, pos, int(CORRUPTIBLE_IDS[4])] = 1.5

        metrics, _ = _token_stats_from_logits(
            logits,
            y,
            final_answer_len=2,
            loss_threshold=0.1,
            greedy_targets=target_ids,
        )
        self.assertEqual(metrics["teacher_forced_token_acc"], 1.0)
        self.assertLess(metrics["clean_token_prob_mean"], 1.0)
        self.assertGreater(metrics["clean_ce_loss"], 0.0)
        self.assertEqual(metrics["sequence_nll_when_full_exact"], metrics["sequence_nll_mean"])

    def test_noisy_entropy_baseline_matches_uniform_digit_corruption_scale(self):
        clean_targets = torch.tensor(
            [[LPAREN_ID, *CORRUPTIBLE_IDS, RPAREN_ID]],
            dtype=torch.long,
        )
        prompt_ids = torch.tensor([[LPAREN_ID, CORRUPTIBLE_IDS[0], EQUALS_ID]], dtype=torch.long)
        eta = 0.5
        h = deterministic_uniform_corruption_entropy(eta, len(CORRUPTIBLE_IDS))
        metrics = noisy_teacher_entropy_baseline(
            clean_targets=clean_targets,
            prompt_ids=prompt_ids,
            final_answer_len=7,
            eta=eta,
            teacher_law="distributional_noise",
            meta={},
        )

        self.assertAlmostEqual(metrics["noisy_teacher_entropy_corruptible_mean"], h)
        self.assertAlmostEqual(metrics["noisy_teacher_entropy_mean"], h * 5.0 / 7.0)
        self.assertAlmostEqual(metrics["noisy_teacher_entropy_final_answer_mean"], h * 5.0 / 7.0)


if __name__ == "__main__":
    unittest.main()
