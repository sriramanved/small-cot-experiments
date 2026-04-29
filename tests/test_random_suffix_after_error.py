from __future__ import annotations

import types
import unittest

import torch

from data.modular_addition.offline_render import (
    generate_teacher_targets as generate_modadd_teacher_targets,
)
from data.s5_cot.offline_render import generate_teacher_targets as generate_s5_teacher_targets
from data.s5_cot.offline_render import render_train_split as render_s5_train_split
from data.s5_cot.prompt_bank import PromptBank
from data.s5_cot.semantic_key_noise import semantic_key_mask
from data.s5_cot.task import CORRUPTIBLE_IDS, LPAREN_ID, RPAREN_ID
from data.synthetic.random_suffix_noise import (
    RANDOM_SUFFIX_AFTER_ERROR_LAW,
    compute_poisoned_before,
)
from nanogpt.methods.student_prefix import (
    FixedPromptCycle,
    cached_teacher_token_probs,
    compute_teacher_token_probs,
)


S5_VOCAB_SIZE = 8


class ScriptedTeacher:
    def __init__(self, clean_target: list[int], *, vocab_size: int) -> None:
        self.clean_target = [int(token_id) for token_id in clean_target]
        self.config = types.SimpleNamespace(vocab_size=vocab_size)
        self.inputs: list[torch.Tensor] = []

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        del use_cache
        step = 0 if past_key_values is None else int(past_key_values)
        self.inputs.append(input_ids.detach().cpu().clone())
        logits = torch.full(
            (*input_ids.shape, self.config.vocab_size),
            -30.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, -1, self.clean_target[step]] = 30.0
        return logits, None, step + 1


def s5_random_suffix_config(seed: int = 1) -> dict[str, object]:
    return {
        "enabled": True,
        "key_positions": "semantic_key",
        "trigger_eta": None,
        "random_suffix_mode": "valid_tokens",
        "keep_format_tokens": True,
        "seed": seed,
        "apply_to": "both",
        "coord_strategy": "fixed",
        "fixed_coord": 0,
        "eligible_values": [1, 2, 3, 4, 5],
        "one_key_per_block": True,
    }


class RandomSuffixAfterErrorTests(unittest.TestCase):
    def test_compute_poisoned_before_uses_previous_key_mismatches(self):
        actions = torch.tensor([[1, 9, 3, 4], [1, 2, 8, 4]], dtype=torch.long)
        clean = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.long)
        key_mask = torch.tensor([False, True, True, False])

        poisoned = compute_poisoned_before(actions, clean, key_mask)

        expected = torch.tensor(
            [[False, False, True, True], [False, False, False, True]]
        )
        torch.testing.assert_close(poisoned, expected)

    def test_fixed_prompt_cycle_can_return_matching_indices(self):
        prompts = torch.tensor([[10], [11], [12]], dtype=torch.long)
        cycle = FixedPromptCycle(
            prompts,
            order=torch.tensor([2, 0, 1]),
            batch_size=2,
            shuffle=False,
        )

        first_idx = cycle.next_batch_indices()
        second_idx = cycle.next_batch_indices()

        torch.testing.assert_close(first_idx, torch.tensor([2, 0]))
        torch.testing.assert_close(second_idx, torch.tensor([1, 2]))
        torch.testing.assert_close(
            prompts.index_select(0, first_idx).to(dtype=torch.uint8),
            torch.tensor([[12], [10]], dtype=torch.uint8),
        )

    def test_s5_eta_zero_matches_clean_greedy_rendering(self):
        clean = [LPAREN_ID, 3, 4, 5, 6, 7, RPAREN_ID]
        targets, teacher_probs = generate_s5_teacher_targets(
            ScriptedTeacher(clean, vocab_size=S5_VOCAB_SIZE),
            torch.tensor([[0, 1]], dtype=torch.long),
            target_len=len(clean),
            eta=0.0,
            rollout_mode="greedy_then_corrupt",
            target_mode="tokens",
            teacher_law=RANDOM_SUFFIX_AFTER_ERROR_LAW,
            random_suffix_noise_config=s5_random_suffix_config(seed=0),
            device="cpu",
        )

        torch.testing.assert_close(targets, torch.tensor([clean], dtype=torch.long))
        self.assertIsNone(teacher_probs)

    def test_s5_render_train_split_packs_random_suffix_targets(self):
        clean = [LPAREN_ID, 3, 4, 5, 6, 7, RPAREN_ID]
        prompt_ids = torch.tensor([[0, 1]], dtype=torch.uint8)
        clean_targets = torch.tensor([clean], dtype=torch.uint8)
        prompt_bank = PromptBank(
            clean_train_prompt_ids=prompt_ids,
            clean_train_cot_ids=clean_targets,
            clean_val_prompt_ids=prompt_ids,
            clean_val_cot_ids=clean_targets,
            train_order=torch.tensor([0], dtype=torch.long),
            meta={
                "task": "s5",
                "prompt_len": 2,
                "cot_len": len(clean),
                "target_len": len(clean),
                "final_answer_len": len(clean),
            },
        )

        train_x, train_y, teacher_probs = render_s5_train_split(
            ScriptedTeacher(clean, vocab_size=S5_VOCAB_SIZE),
            prompt_bank,
            torch.tensor([0], dtype=torch.long),
            eta=0.0,
            rollout_mode="greedy_then_corrupt",
            target_mode="tokens",
            gen_batch_size=1,
            device="cpu",
            teacher_law=RANDOM_SUFFIX_AFTER_ERROR_LAW,
            random_suffix_noise_config=s5_random_suffix_config(seed=0),
        )

        self.assertEqual(tuple(train_x.shape), (1, prompt_bank.xy_len))
        self.assertEqual(tuple(train_y.shape), (1, prompt_bank.xy_len))
        self.assertIsNone(teacher_probs)
        torch.testing.assert_close(train_y[:, prompt_bank.prompt_len - 1:], clean_targets.to(dtype=train_y.dtype))

    def test_s5_poison_triggers_at_key_and_random_suffix_keeps_format_valid(self):
        clean = [
            LPAREN_ID, 3, 4, 5, 6, 7, RPAREN_ID,
            LPAREN_ID, 3, 4, 5, 6, 7, RPAREN_ID,
        ]
        prompt = torch.tensor([[0, 1]], dtype=torch.long)
        config = s5_random_suffix_config(seed=1)
        teacher = ScriptedTeacher(clean, vocab_size=S5_VOCAB_SIZE)

        targets, _ = generate_s5_teacher_targets(
            teacher,
            prompt,
            target_len=len(clean),
            eta=1.0,
            rollout_mode="greedy_then_corrupt",
            target_mode="tokens",
            teacher_law=RANDOM_SUFFIX_AFTER_ERROR_LAW,
            random_suffix_noise_config=config,
            device="cpu",
        )

        key_mask = semantic_key_mask(prompt, len(clean), {
            "enabled": True,
            "coord_strategy": "fixed",
            "fixed_coord": 0,
            "seed": 1,
            "include_clean_value": True,
            "eligible_values": [1, 2, 3, 4, 5],
            "apply_to": "partial_perm_image",
            "one_key_per_block": True,
        })
        key_diffs = targets.ne(torch.tensor([clean])) & key_mask
        first_poison = int(torch.nonzero(key_diffs[0], as_tuple=False).flatten()[0].item())
        self.assertEqual(first_poison, 1)

        for pos in (6, 7, 13):
            self.assertEqual(int(targets[0, pos].item()), clean[pos])

        semantic_after = [pos for pos in range(first_poison + 1, len(clean)) if pos % 7 in {1, 2, 3, 4, 5}]
        self.assertTrue(all(int(targets[0, pos].item()) in CORRUPTIBLE_IDS for pos in semantic_after))
        self.assertTrue(any(int(targets[0, pos].item()) != clean[pos] for pos in semantic_after))

    def test_s5_rolls_sampled_corruption_into_next_teacher_query(self):
        clean = [LPAREN_ID, 3, 4, 5]
        teacher = ScriptedTeacher(clean, vocab_size=S5_VOCAB_SIZE)
        targets, _ = generate_s5_teacher_targets(
            teacher,
            torch.tensor([[0, 1]], dtype=torch.long),
            target_len=len(clean),
            eta=1.0,
            rollout_mode="greedy_then_corrupt",
            target_mode="tokens",
            teacher_law=RANDOM_SUFFIX_AFTER_ERROR_LAW,
            random_suffix_noise_config=s5_random_suffix_config(seed=1),
            device="cpu",
        )

        self.assertEqual(int(teacher.inputs[2][0, -1].item()), int(targets[0, 1].item()))

    def test_s5_no_poison_fraction_decreases_with_eta(self):
        clean = [
            LPAREN_ID, 3, 4, 5, 6, 7, RPAREN_ID,
            LPAREN_ID, 3, 4, 5, 6, 7, RPAREN_ID,
        ]
        prompt = torch.tensor([[0, 1]] * 64, dtype=torch.long)
        clean_batch = torch.tensor([clean] * prompt.size(0), dtype=torch.long)
        config = s5_random_suffix_config(seed=7)
        key_mask = semantic_key_mask(prompt, len(clean), {
            "enabled": True,
            "coord_strategy": "fixed",
            "fixed_coord": 0,
            "seed": 7,
            "include_clean_value": True,
            "eligible_values": [1, 2, 3, 4, 5],
            "apply_to": "partial_perm_image",
            "one_key_per_block": True,
        })

        clean_targets, _ = generate_s5_teacher_targets(
            ScriptedTeacher(clean, vocab_size=S5_VOCAB_SIZE),
            prompt,
            target_len=len(clean),
            eta=0.0,
            rollout_mode="greedy_then_corrupt",
            target_mode="tokens",
            teacher_law=RANDOM_SUFFIX_AFTER_ERROR_LAW,
            random_suffix_noise_config=config,
            device="cpu",
        )
        noisy_targets, _ = generate_s5_teacher_targets(
            ScriptedTeacher(clean, vocab_size=S5_VOCAB_SIZE),
            prompt,
            target_len=len(clean),
            eta=1.0,
            rollout_mode="greedy_then_corrupt",
            target_mode="tokens",
            teacher_law=RANDOM_SUFFIX_AFTER_ERROR_LAW,
            random_suffix_noise_config=config,
            device="cpu",
        )

        clean_no_poison = (~(clean_targets.ne(clean_batch) & key_mask).any(dim=1)).float().mean()
        noisy_no_poison = (~(noisy_targets.ne(clean_batch) & key_mask).any(dim=1)).float().mean()
        self.assertEqual(float(clean_no_poison.item()), 1.0)
        self.assertLess(float(noisy_no_poison.item()), float(clean_no_poison.item()))

    def test_modadd_random_suffix_after_error_poison_suffix_is_valid(self):
        p = 5
        clean = [1, 2, 3, 4]
        targets = generate_modadd_teacher_targets(
            ScriptedTeacher(clean, vocab_size=p + 1),
            torch.tensor([[1, 2, p]], dtype=torch.long),
            target_len=len(clean),
            eta=1.0,
            rollout_mode="greedy_then_corrupt",
            teacher_law=RANDOM_SUFFIX_AFTER_ERROR_LAW,
            random_suffix_noise_config={
                "enabled": True,
                "key_positions": "semantic_key",
                "trigger_eta": None,
                "random_suffix_mode": "valid_tokens",
                "keep_format_tokens": True,
                "seed": 0,
                "apply_to": "both",
            },
            device="cpu",
            p=p,
        )

        self.assertTrue(torch.all((targets >= 0) & (targets < p)))
        self.assertFalse(torch.equal(targets, torch.tensor([clean], dtype=torch.long)))

    def test_s5_online_random_suffix_uses_student_prefix_to_poison(self):
        clean = [
            LPAREN_ID, 3, 4, 5, 6, 7, RPAREN_ID,
            LPAREN_ID, 3, 4, 5, 6, 7, RPAREN_ID,
        ]
        actions = torch.tensor([clean], dtype=torch.long)
        actions[0, 1] = 4

        teacher_probs = cached_teacher_token_probs(
            ScriptedTeacher(clean, vocab_size=S5_VOCAB_SIZE),
            torch.tensor([[0, 1]], dtype=torch.long),
            actions,
            clean_target_ids=torch.tensor([clean], dtype=torch.long),
            eta=0.5,
            teacher_law=RANDOM_SUFFIX_AFTER_ERROR_LAW,
            random_suffix_noise_config=s5_random_suffix_config(seed=0),
            task_name="s5",
            device="cpu",
        )

        eligible = torch.tensor(CORRUPTIBLE_IDS, dtype=torch.long)
        expected_key = torch.zeros(S5_VOCAB_SIZE)
        expected_key[3] = 0.5
        expected_key[eligible] += 0.5 / float(len(CORRUPTIBLE_IDS))
        torch.testing.assert_close(teacher_probs[0, 1], expected_key)

        expected_uniform = torch.zeros(S5_VOCAB_SIZE)
        expected_uniform[eligible] = 1.0 / float(len(CORRUPTIBLE_IDS))
        torch.testing.assert_close(teacher_probs[0, 2], expected_uniform)

        expected_rparen = torch.zeros(S5_VOCAB_SIZE)
        expected_rparen[RPAREN_ID] = 1.0
        torch.testing.assert_close(teacher_probs[0, 6], expected_rparen)

    def test_modadd_online_random_suffix_uses_student_prefix_to_poison(self):
        p = 5
        clean = [1, 2, 3]
        actions = torch.tensor([[4, 2, 3]], dtype=torch.long)

        teacher_probs = cached_teacher_token_probs(
            ScriptedTeacher(clean, vocab_size=p + 1),
            torch.tensor([[1, 2, p]], dtype=torch.long),
            actions,
            clean_target_ids=torch.tensor([clean], dtype=torch.long),
            eta=0.0,
            teacher_law=RANDOM_SUFFIX_AFTER_ERROR_LAW,
            corruptible_token_ids=tuple(range(p)),
            random_suffix_noise_config={
                "enabled": True,
                "key_positions": "semantic_key",
                "trigger_eta": None,
                "random_suffix_mode": "valid_tokens",
                "keep_format_tokens": True,
                "seed": 0,
                "apply_to": "both",
            },
            task_name="modadd",
            device="cpu",
        )

        expected_uniform = torch.zeros(p + 1)
        expected_uniform[:p] = 1.0 / float(p)
        torch.testing.assert_close(teacher_probs[0, 1], expected_uniform)

    def test_stateless_random_suffix_teacher_probs_raise_clear_error(self):
        clean_logits = torch.zeros((1, 1, S5_VOCAB_SIZE), dtype=torch.float32)
        with self.assertRaisesRegex(NotImplementedError, "stateful"):
            compute_teacher_token_probs(
                clean_logits,
                eta=0.1,
                teacher_law=RANDOM_SUFFIX_AFTER_ERROR_LAW,
                corruptible_token_ids=CORRUPTIBLE_IDS,
            )

    def test_distributional_law_still_normalizes(self):
        logits = torch.randn(2, 3, S5_VOCAB_SIZE)
        probs = compute_teacher_token_probs(
            logits,
            eta=0.2,
            teacher_law="distributional_noise",
            corruptible_token_ids=CORRUPTIBLE_IDS,
        )
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2, 3))


if __name__ == "__main__":
    unittest.main()
