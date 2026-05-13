from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import torch

from nanogpt.methods.student_prefix import (
    cached_teacher_token_probs,
    compute_teacher_log_probs,
    compute_teacher_token_probs,
    extract_answer_logits,
    teacher_forward_kl,
)
from nanogpt.trainers.nail import validate_args as validate_nail_args
from nanogpt.trainers.nail import validate_resume_metadata as validate_nail_resume_metadata
from nanogpt.trainers.opd import validate_args as validate_opd_args
from nanogpt.trainers.opd import validate_resume_metadata as validate_opd_resume_metadata


VOCAB_SIZE = 8


class ToyCachedModel:
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.config = types.SimpleNamespace(vocab_size=vocab_size)

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        logits = torch.nn.functional.one_hot(
            input_ids.to(dtype=torch.long),
            num_classes=self.config.vocab_size,
        ).to(dtype=torch.float32)
        logits = 3.0 * logits + 0.5
        if use_cache:
            return logits, None, ()
        return logits, None


class OpdObjectiveTests(unittest.TestCase):
    @staticmethod
    def _cfg(**overrides):
        base = dict(
            method_family="opd",
            teacher_signal="mc",
            loss="reverse",
            kl_beta=None,
            init_from="scratch",
            init_from_ckpt=None,
            continue_from_subset_size=0,
            single_epoch=False,
            shuffle_prompts=False,
            subset_size=4,
            rollout_temperature_override=None,
            loss_temperature_override=None,
            compile=False,
        )
        base.update(overrides)
        return types.SimpleNamespace(**base)

    def test_distributional_teacher_probs_sum_to_one_and_match_log_probs(self):
        torch.manual_seed(0)
        clean_logits = torch.randn(2, 3, VOCAB_SIZE)
        teacher_probs = compute_teacher_token_probs(
            clean_logits,
            eta=0.2,
            teacher_law="distributional_noise",
        )
        actions = torch.randint(0, VOCAB_SIZE, (2, 3))
        expected = torch.log(
            teacher_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1).clamp_min(1e-10)
        )
        actual = compute_teacher_log_probs(
            clean_logits,
            actions,
            eta=0.2,
            teacher_law="distributional_noise",
            eps=1e-10,
        )
        torch.testing.assert_close(
            teacher_probs.sum(dim=-1),
            torch.ones(2, 3),
        )
        torch.testing.assert_close(actual, expected)

    def test_corrupted_greedy_teacher_probs_match_digit_case(self):
        clean_logits = torch.full((1, 1, VOCAB_SIZE), -10.0)
        clean_logits[..., 4] = 10.0
        teacher_probs = compute_teacher_token_probs(
            clean_logits,
            eta=0.2,
            teacher_law="corrupted_greedy",
        )
        expected = torch.zeros_like(teacher_probs)
        expected[..., 3:8] = 0.04
        expected[..., 4] = 0.84
        torch.testing.assert_close(teacher_probs, expected)

    def test_corrupted_greedy_teacher_probs_match_non_digit_case(self):
        clean_logits = torch.full((1, 1, VOCAB_SIZE), -10.0)
        clean_logits[..., 2] = 10.0
        teacher_probs = compute_teacher_token_probs(
            clean_logits,
            eta=0.2,
            teacher_law="corrupted_greedy",
        )
        expected = torch.zeros_like(teacher_probs)
        expected[..., 2] = 1.0
        torch.testing.assert_close(teacher_probs, expected)

    def test_cached_teacher_probs_have_expected_shape_and_normalize(self):
        model = ToyCachedModel()
        prompt_ids = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.uint8)
        actions = torch.tensor([[3, 4, 5], [5, 4, 3]], dtype=torch.long)
        teacher_probs = cached_teacher_token_probs(
            model,
            prompt_ids,
            actions,
            eta=0.15,
            teacher_law="distributional_noise",
            device="cpu",
        )
        self.assertEqual(teacher_probs.shape, (2, 3, VOCAB_SIZE))
        torch.testing.assert_close(
            teacher_probs.sum(dim=-1),
            torch.ones(2, 3),
        )

    def test_cached_teacher_probs_match_explicit_stepwise_teacher_distribution(self):
        # Online methods query the noisy teacher along learner-induced prefixes;
        # this guards the stepwise cached path against teacher-forcing drift.
        model = ToyCachedModel()
        prompt_ids = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.uint8)
        actions = torch.tensor([[3, 4, 5], [5, 4, 3]], dtype=torch.long)

        teacher_probs = cached_teacher_token_probs(
            model,
            prompt_ids,
            actions,
            eta=0.15,
            teacher_law="distributional_noise",
            device="cpu",
        )

        full_seq = torch.cat((prompt_ids.long(), actions), dim=1)
        full_logits, _ = model(full_seq[:, :-1], use_cache=False)
        answer_logits = extract_answer_logits(
            full_logits,
            prompt_len=prompt_ids.size(1),
            target_len=actions.size(1),
        )
        expected = compute_teacher_token_probs(
            answer_logits,
            eta=0.15,
            teacher_law="distributional_noise",
        )

        torch.testing.assert_close(teacher_probs, expected)

    def test_forward_kl_is_zero_when_student_matches_teacher(self):
        torch.manual_seed(1)
        clean_logits = torch.randn(2, 4, VOCAB_SIZE)
        teacher_probs = compute_teacher_token_probs(
            clean_logits,
            eta=0.1,
            teacher_law="distributional_noise",
        )
        student_logits = teacher_probs.log()
        token_kl, teacher_ce, teacher_entropy = teacher_forward_kl(
            teacher_probs,
            student_logits,
            eps=1e-10,
        )
        torch.testing.assert_close(token_kl, torch.zeros_like(token_kl), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(teacher_ce, teacher_entropy, atol=1e-6, rtol=1e-6)

    def test_opd_rejects_forward_loss(self):
        with self.assertRaisesRegex(ValueError, "OPD only supports reverse loss"):
            validate_opd_args(self._cfg(loss="forward"))

    def test_nail_accepts_forward_and_reverse_losses(self):
        validate_nail_args(self._cfg(method_family="nail", loss="forward"))
        validate_nail_args(self._cfg(method_family="nail", loss="reverse"))

    def test_nail_accepts_mixed_mc_loss_with_beta(self):
        validate_nail_args(
            self._cfg(
                method_family="nail",
                teacher_signal="mc",
                loss="mixed",
                kl_beta=0.5,
            )
        )

    def test_nail_accepts_jsd_mc_loss_with_beta(self):
        validate_nail_args(
            self._cfg(
                method_family="nail",
                teacher_signal="mc",
                loss="jsd",
                kl_beta=0.5,
            )
        )

    def test_beta_losses_require_nail_mc_and_valid_beta(self):
        # Mixed/JSD beta ablations are defined for the student-prefix MC path;
        # config validation should keep unsupported combinations out.
        invalid_cases = [
            (self._cfg(method_family="nail", loss="mixed"), "requires task.kl_beta"),
            (self._cfg(method_family="nail", loss="jsd"), "requires task.kl_beta"),
            (
                self._cfg(method_family="nail", loss="mixed", kl_beta=-0.1),
                r"task\.kl_beta must be in \[0, 1\]",
            ),
            (
                self._cfg(method_family="nail", loss="jsd", kl_beta=-0.1),
                r"task\.kl_beta must be in \[0, 1\]",
            ),
            (
                self._cfg(method_family="nail", loss="mixed", kl_beta=1.1),
                r"task\.kl_beta must be in \[0, 1\]",
            ),
            (
                self._cfg(method_family="nail", loss="jsd", kl_beta=1.1),
                r"task\.kl_beta must be in \[0, 1\]",
            ),
            (
                self._cfg(method_family="nail", teacher_signal="full", loss="mixed", kl_beta=0.5),
                "requires teacher_signal='mc'",
            ),
            (
                self._cfg(method_family="nail", teacher_signal="full", loss="jsd", kl_beta=0.5),
                "requires teacher_signal='mc'",
            ),
            (
                self._cfg(method_family="opd", loss="mixed", kl_beta=0.5),
                "mixed loss is only supported for NAIL",
            ),
            (
                self._cfg(method_family="opd", loss="jsd", kl_beta=0.5),
                "jsd loss is only supported for NAIL",
            ),
        ]
        for cfg, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_nail_args(cfg)

    def test_kl_beta_rejected_for_non_mixed_losses(self):
        with self.assertRaisesRegex(ValueError, "only supported when task.loss is 'mixed' or 'jsd'"):
            validate_nail_args(self._cfg(method_family="nail", loss="forward", kl_beta=0.5))

    def test_reverse_loss_rejects_loss_temperature_override(self):
        with self.assertRaisesRegex(ValueError, "only supported for forward, mixed, or jsd loss"):
            validate_nail_args(
                self._cfg(
                    method_family="nail",
                    loss="reverse",
                    loss_temperature_override=1.0,
                )
            )

    def test_forward_loss_accepts_positive_temperature_override(self):
        validate_nail_args(
            self._cfg(
                method_family="nail",
                loss="forward",
                loss_temperature_override=0.7,
            )
        )

    def test_mixed_loss_accepts_positive_loss_temperature_override(self):
        validate_nail_args(
            self._cfg(
                method_family="nail",
                loss="mixed",
                kl_beta=0.5,
                loss_temperature_override=0.7,
            )
        )

    def test_jsd_loss_accepts_positive_loss_temperature_override(self):
        validate_nail_args(
            self._cfg(
                method_family="nail",
                loss="jsd",
                kl_beta=0.5,
                loss_temperature_override=0.7,
            )
        )

    def test_teacher_laws_diverge_on_digit_and_nondigit_cases(self):
        digit_case_logits = torch.zeros((1, 1, VOCAB_SIZE), dtype=torch.float32)
        digit_case_logits[..., 4] = 2.0
        digit_case_logits[..., 5] = 1.0
        digit_case_logits[..., 2] = -0.5
        nondigit_case_logits = torch.zeros((1, 1, VOCAB_SIZE), dtype=torch.float32)
        nondigit_case_logits[..., 2] = 2.0
        nondigit_case_logits[..., 4] = 1.5
        nondigit_case_logits[..., 5] = 0.5

        dist_digit = compute_teacher_token_probs(
            digit_case_logits,
            eta=0.2,
            teacher_law="distributional_noise",
        )
        greedy_digit = compute_teacher_token_probs(
            digit_case_logits,
            eta=0.2,
            teacher_law="corrupted_greedy",
        )
        self.assertGreater((dist_digit - greedy_digit).abs().max().item(), 1e-2)

        dist_nondigit = compute_teacher_token_probs(
            nondigit_case_logits,
            eta=0.2,
            teacher_law="distributional_noise",
        )
        greedy_nondigit = compute_teacher_token_probs(
            nondigit_case_logits,
            eta=0.2,
            teacher_law="corrupted_greedy",
        )
        self.assertGreater((dist_nondigit - greedy_nondigit).abs().max().item(), 1e-2)

    def test_resume_metadata_translates_legacy_tm_opd_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "teacher_checkpoint": "teacher",
                        "prompt_bank_dir": "prompt_bank",
                        "subset_size": 4,
                        "eta": 0.1,
                        "teacher_law": "distributional_noise",
                        "student_temperature": 1.0,
                        "seed": 123,
                    },
                    f,
                )
            validate_opd_resume_metadata(
                out_dir,
                {
                    "task": "s5",
                    "p": 5,
                    "teacher_checkpoint": "teacher",
                    "prompt_bank_dir": "prompt_bank",
                    "subset_size": 4,
                    "eta": 0.1,
                    "teacher_law": "distributional_noise",
                    "method_family": "opd",
                    "teacher_signal": "mc",
                    "loss": "reverse",
                    "shuffle_prompts": False,
                    "seed": 123,
                },
                )

    def test_resume_metadata_defaults_missing_rollout_temperature_to_student_temperature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "teacher_checkpoint": "teacher",
                        "prompt_bank_dir": "prompt_bank",
                        "subset_size": 4,
                        "eta": 0.1,
                        "teacher_law": "distributional_noise",
                        "objective": "reverse_kl_tm",
                        "student_temperature": 1.0,
                        "shuffle_prompts": False,
                        "seed": 123,
                    },
                    f,
                )
            validate_opd_resume_metadata(
                out_dir,
                {
                    "task": "s5",
                    "p": 5,
                    "teacher_checkpoint": "teacher",
                    "prompt_bank_dir": "prompt_bank",
                    "subset_size": 4,
                    "eta": 0.1,
                    "teacher_law": "distributional_noise",
                    "method_family": "opd",
                    "teacher_signal": "mc",
                    "loss": "reverse",
                    "shuffle_prompts": False,
                    "seed": 123,
                },
            )

    def test_resume_metadata_maps_legacy_greedy_reverse_run_to_nail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "task": "s5",
                        "teacher_checkpoint": "teacher",
                        "prompt_bank_dir": "prompt_bank",
                        "subset_size": 4,
                        "eta": 0.1,
                        "teacher_law": "distributional_noise",
                        "objective": "reverse_kl_full",
                        "student_temperature": 0.0,
                        "student_rollout_temperature": 0.0,
                        "shuffle_prompts": False,
                        "seed": 123,
                    },
                    f,
                )
            validate_nail_resume_metadata(
                out_dir,
                {
                    "task": "s5",
                    "p": 5,
                    "teacher_checkpoint": "teacher",
                    "prompt_bank_dir": "prompt_bank",
                    "subset_size": 4,
                    "eta": 0.1,
                    "teacher_law": "distributional_noise",
                    "method_family": "nail",
                    "teacher_signal": "full",
                    "loss": "reverse",
                    "shuffle_prompts": False,
                    "seed": 123,
                },
            )


if __name__ == "__main__":
    unittest.main()
