from __future__ import annotations

import types
import unittest

import torch

from data.s5_cot.offline_render import generate_teacher_targets
from data.s5_cot.semantic_key_noise import (
    SemanticKeyNoiseConfig,
    eligible_token_ids_from_values,
    semantic_key_mask,
)
from data.s5_cot.task import LPAREN_ID
from nanogpt.methods.student_prefix import (
    cached_teacher_token_probs,
    compute_teacher_token_probs,
    semantic_key_noise_probs,
)


VOCAB_SIZE = 8


class PrefixSensitiveTeacher:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(vocab_size=VOCAB_SIZE)

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        del past_key_values, use_cache
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, VOCAB_SIZE), -30.0)
        if seq_len > 1:
            next_token = torch.full((batch_size,), LPAREN_ID, dtype=torch.long)
        else:
            last_token = input_ids[:, -1].to(dtype=torch.long)
            next_token = torch.where(
                last_token.eq(LPAREN_ID),
                torch.full_like(last_token, 3),
                torch.where(last_token.eq(3), torch.full_like(last_token, 6), torch.full_like(last_token, 5)),
            )
        logits[torch.arange(batch_size), -1, next_token] = 30.0
        return logits, None, ()


class UniformTeacher:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(vocab_size=VOCAB_SIZE)

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        del past_key_values
        logits = torch.zeros((*input_ids.shape, VOCAB_SIZE), dtype=torch.float32)
        if use_cache:
            return logits, None, ()
        return logits, None


class S5SemanticKeyNoiseTests(unittest.TestCase):
    def test_semantic_key_noise_probs_only_mixes_key_positions(self):
        # Semantic-key noise corrupts one chosen value coordinate per block; all
        # other positions should keep the clean expert distribution.
        teacher_probs = torch.tensor(
            [
                [0.05, 0.10, 0.10, 0.20, 0.25, 0.10, 0.10, 0.10],
                [0.02, 0.03, 0.05, 0.10, 0.50, 0.10, 0.10, 0.10],
            ],
            dtype=torch.float32,
        )
        key_mask = torch.tensor([False, True])
        eligible = eligible_token_ids_from_values([1, 2, 3, 4, 5])

        actual = semantic_key_noise_probs(
            teacher_probs,
            eta=0.3,
            key_mask=key_mask,
            eligible_token_ids=eligible,
        )

        expected_key = 0.7 * teacher_probs[1].clone()
        expected_key[list(eligible)] += 0.3 / len(eligible)
        torch.testing.assert_close(actual[0], teacher_probs[0])
        torch.testing.assert_close(actual[1], expected_key)
        torch.testing.assert_close(actual.sum(dim=-1), torch.ones(2))
        self.assertGreater(actual[1, 4].item(), 0.7 * teacher_probs[1, 4].item())

    def test_compute_teacher_token_probs_semantic_key_includes_clean_argmax_support(self):
        clean_logits = torch.full((1, 1, VOCAB_SIZE), -20.0)
        clean_logits[..., 4] = 20.0
        eligible = eligible_token_ids_from_values([1, 2, 3, 4, 5])

        probs = compute_teacher_token_probs(
            clean_logits,
            eta=0.5,
            teacher_law="semantic_key_noise",
            key_mask=torch.tensor([True]),
            eligible_token_ids=eligible,
        )

        self.assertIn(4, eligible)
        self.assertGreater(probs[0, 0, 4].item(), 0.5)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones((1, 1)))

    def test_cyclic_and_fixed_coordinate_selection_have_one_key_per_block(self):
        prompt_ids = torch.zeros((2, 22), dtype=torch.long)
        cyclic = SemanticKeyNoiseConfig(coord_strategy="cyclic")
        cyclic_mask = semantic_key_mask(prompt_ids, 21, cyclic)
        self.assertEqual(cyclic_mask.sum(dim=1).tolist(), [3, 3])
        self.assertEqual(torch.nonzero(cyclic_mask[0], as_tuple=False).flatten().tolist(), [1, 9, 17])

        fixed = SemanticKeyNoiseConfig(coord_strategy="fixed", fixed_coord=4)
        fixed_mask = semantic_key_mask(prompt_ids, 21, fixed)
        self.assertEqual(fixed_mask.sum(dim=1).tolist(), [3, 3])
        self.assertEqual(torch.nonzero(fixed_mask[0], as_tuple=False).flatten().tolist(), [5, 12, 19])

    def test_hash_coordinate_selection_is_deterministic_given_seed(self):
        prompt_ids = torch.tensor([[0, 3, 4, 5, 6, 7], [0, 7, 6, 5, 4, 3]], dtype=torch.long)
        config = SemanticKeyNoiseConfig(coord_strategy="hash", seed=123)

        first = semantic_key_mask(prompt_ids, 28, config)
        second = semantic_key_mask(prompt_ids, 28, config)

        torch.testing.assert_close(first, second)
        self.assertEqual(first.sum(dim=1).tolist(), [4, 4])

    def test_offline_semantic_rollout_feeds_sampled_key_token_back(self):
        # Offline rendering feeds the realized noisy token into the next teacher
        # query, so later labels can depend on earlier corruption.
        torch.manual_seed(1)
        config = {
            "enabled": True,
            "coord_strategy": "fixed",
            "fixed_coord": 0,
            "seed": 0,
            "include_clean_value": True,
            "eligible_values": [1, 2, 3, 4, 5],
            "apply_to": "partial_perm_image",
            "one_key_per_block": True,
        }

        targets, _ = generate_teacher_targets(
            PrefixSensitiveTeacher(),
            torch.tensor([[0, 1]], dtype=torch.long),
            target_len=3,
            eta=1.0,
            rollout_mode="greedy_then_corrupt",
            target_mode="tokens",
            teacher_law="semantic_key_noise",
            semantic_key_noise_config=config,
            device="cpu",
        )

        torch.testing.assert_close(targets, torch.tensor([[LPAREN_ID, 6, 5]], dtype=torch.long))

    def test_online_cached_teacher_uses_same_semantic_key_mask(self):
        # Online teacher queries must use the same key-position rule as offline
        # rendering so S5 method comparisons are apples-to-apples.
        prompt_ids = torch.zeros((1, 8), dtype=torch.long)
        actions = torch.zeros((1, 14), dtype=torch.long)
        config = SemanticKeyNoiseConfig(coord_strategy="cyclic")
        eligible = eligible_token_ids_from_values([1, 2, 3, 4, 5])
        key_mask = semantic_key_mask(prompt_ids, actions.size(1), config)

        teacher_probs = cached_teacher_token_probs(
            UniformTeacher(),
            prompt_ids,
            actions,
            eta=0.5,
            teacher_law="semantic_key_noise",
            semantic_key_noise_config=config,
            device="cpu",
        )

        clean = torch.full((VOCAB_SIZE,), 1.0 / VOCAB_SIZE)
        expected_key = 0.5 * clean.clone()
        expected_key[list(eligible)] += 0.5 / len(eligible)
        for pos in range(actions.size(1)):
            expected = expected_key if bool(key_mask[0, pos].item()) else clean
            torch.testing.assert_close(teacher_probs[0, pos], expected)


if __name__ == "__main__":
    unittest.main()
