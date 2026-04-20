from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

import torch

from data.s5_cot.opd import (
    compute_teacher_token_probs,
    extract_answer_logits,
    gather_action_log_probs,
    teacher_forward_kl,
)

HF_AVAILABLE = importlib.util.find_spec("transformers") is not None

if HF_AVAILABLE:
    from data.s5_cot.opd_hf import cached_teacher_token_probs_hf, rollout_student_hf
    from hf_checkpoint import build_hf_model_from_nanogpt_args
    from nanogpt.trainers.opd_hf import validate_args, validate_resume_metadata
else:
    cached_teacher_token_probs_hf = None
    rollout_student_hf = None
    build_hf_model_from_nanogpt_args = None
    validate_args = None
    validate_resume_metadata = None


VOCAB_SIZE = 8


def tiny_model_args(*, bias: bool = False) -> dict[str, int | float | bool]:
    return {
        "block_size": 16,
        "vocab_size": VOCAB_SIZE,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
        "dropout": 0.0,
        "bias": bias,
    }


@unittest.skipUnless(HF_AVAILABLE, "transformers is not installed")
class HFOpdHelperTests(unittest.TestCase):
    def test_rollout_student_hf_matches_full_forward_log_probs(self):
        torch.manual_seed(0)
        model = build_hf_model_from_nanogpt_args(
            tiny_model_args(),
            device="cpu",
            torch_dtype=torch.float32,
            eval_mode=True,
        )
        prompt_ids = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.uint8)
        full_seq, actions, log_q = rollout_student_hf(
            model,
            prompt_ids,
            target_len=4,
            temperature=1.0,
            device="cpu",
        )
        outputs = model(
            input_ids=full_seq[:, :-1],
            use_cache=False,
        )
        answer_logits = extract_answer_logits(
            outputs.logits,
            prompt_len=prompt_ids.size(1),
            target_len=actions.size(1),
        )
        expected_log_q = gather_action_log_probs(
            answer_logits,
            actions,
            temperature=1.0,
        )
        self.assertEqual(full_seq.shape, (2, 7))
        self.assertEqual(actions.shape, (2, 4))
        self.assertEqual(log_q.shape, (2, 4))
        torch.testing.assert_close(log_q, expected_log_q, atol=1e-5, rtol=1e-5)

    def test_cached_teacher_probs_hf_match_direct_teacher_distribution(self):
        torch.manual_seed(1)
        model = build_hf_model_from_nanogpt_args(
            tiny_model_args(),
            device="cpu",
            torch_dtype=torch.float32,
            eval_mode=True,
        )
        prompt_ids = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.uint8)
        actions = torch.tensor([[3, 4, 5], [5, 4, 3]], dtype=torch.long)
        teacher_probs = cached_teacher_token_probs_hf(
            model,
            prompt_ids,
            actions,
            eta=0.2,
            teacher_law="distributional_noise",
            device="cpu",
        )
        outputs = model(
            input_ids=torch.cat((prompt_ids.long(), actions), dim=1)[:, :-1],
            use_cache=False,
        )
        answer_logits = extract_answer_logits(
            outputs.logits,
            prompt_len=prompt_ids.size(1),
            target_len=actions.size(1),
        )
        expected = compute_teacher_token_probs(
            answer_logits,
            eta=0.2,
            teacher_law="distributional_noise",
        )
        self.assertEqual(teacher_probs.shape, (2, 3, VOCAB_SIZE))
        torch.testing.assert_close(teacher_probs.sum(dim=-1), torch.ones(2, 3))
        torch.testing.assert_close(teacher_probs, expected, atol=1e-5, rtol=1e-5)

    def test_build_hf_model_freezes_biases_when_nanogpt_bias_is_false(self):
        model = build_hf_model_from_nanogpt_args(
            tiny_model_args(bias=False),
            device="cpu",
            torch_dtype=torch.float32,
            eval_mode=False,
        )
        bias_params = [
            param
            for name, param in model.named_parameters()
            if name.endswith(".bias")
        ]
        self.assertTrue(bias_params)
        self.assertTrue(all(not param.requires_grad for param in bias_params))
        self.assertTrue(all(torch.count_nonzero(param).item() == 0 for param in bias_params))

    def test_forward_kl_is_zero_when_student_matches_teacher(self):
        torch.manual_seed(2)
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

    def test_forward_kl_rejects_zero_temperature(self):
        with self.assertRaisesRegex(ValueError, "student_temperature must be > 0"):
            validate_args(
                types.SimpleNamespace(
                    objective="forward_kl_simple",
                    student_temperature=0.0,
                    student_rollout_temperature=0.0,
                    compile=False,
                )
            )
        validate_args(
            types.SimpleNamespace(
                objective="forward_kl_simple",
                student_temperature=1.0,
                student_rollout_temperature=0.0,
                compile=False,
            )
        )
        validate_args(
            types.SimpleNamespace(
                objective="reverse_kl_tm",
                student_temperature=0.0,
                student_rollout_temperature=0.0,
                compile=False,
            )
        )

    def test_resume_metadata_rejects_backend_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "backend": "hf",
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
            with self.assertRaisesRegex(ValueError, "Resume mismatch for backend"):
                validate_resume_metadata(
                    out_dir,
                    {
                        "backend": "nanogpt",
                        "teacher_checkpoint": "teacher",
                        "prompt_bank_dir": "prompt_bank",
                        "subset_size": 4,
                        "eta": 0.1,
                        "teacher_law": "distributional_noise",
                        "objective": "forward_kl_full",
                        "student_temperature": 1.0,
                        "student_rollout_temperature": 1.0,
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
                        "backend": "hf",
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
            validate_resume_metadata(
                out_dir,
                {
                    "backend": "hf",
                    "teacher_checkpoint": "teacher",
                    "prompt_bank_dir": "prompt_bank",
                    "subset_size": 4,
                    "eta": 0.1,
                    "teacher_law": "distributional_noise",
                    "objective": "forward_kl_full",
                    "student_temperature": 1.0,
                    "student_rollout_temperature": 1.0,
                    "shuffle_prompts": False,
                    "seed": 123,
                },
            )


if __name__ == "__main__":
    unittest.main()
