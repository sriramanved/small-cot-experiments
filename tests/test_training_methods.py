from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from data.s5_cot.opd import (
    forward_kl_full_loss,
    forward_kl_simple_loss,
    gather_action_log_probs,
    reverse_kl_full_loss,
    reverse_kl_tm_loss,
    teacher_forward_kl,
)
from data.synthetic.offline_losses import offline_teacher_prob_loss_from_logits
from model import causal_lm_loss


class TrainingMethodTests(unittest.TestCase):
    def test_reverse_kl_tm_loss_matches_manual_formula_and_detaches_log_q(self):
        student_logits = torch.tensor(
            [[[1.2, -0.3, 0.4], [0.1, 0.5, -0.7]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        teacher_probs = torch.tensor(
            [[[0.7, 0.2, 0.1], [0.15, 0.25, 0.6]]],
            dtype=torch.float32,
        )
        actions = torch.tensor([[0, 2]], dtype=torch.long)
        log_q = torch.tensor([[-0.8, -1.4]], dtype=torch.float32, requires_grad=True)

        loss, stats = reverse_kl_tm_loss(
            student_logits,
            actions,
            log_q=log_q,
            teacher_probs=teacher_probs,
            eps=1e-10,
        )

        teacher_action_probs = teacher_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        expected_log_teacher = torch.log(teacher_action_probs.clamp_min(1e-10))
        expected_advantage = expected_log_teacher - log_q
        expected_log_p = F.log_softmax(student_logits, dim=-1).gather(2, actions.unsqueeze(-1)).squeeze(-1)
        expected_importance = torch.exp(expected_log_p - log_q.detach())
        expected_loss = -(expected_importance * expected_advantage.detach()).mean()

        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(stats["log_teacher"], expected_log_teacher)
        torch.testing.assert_close(stats["advantage"], expected_advantage)
        torch.testing.assert_close(stats["log_p"], expected_log_p)
        torch.testing.assert_close(stats["importance_weight"], expected_importance)

        loss.backward()
        self.assertIsNone(log_q.grad)
        self.assertIsNotNone(student_logits.grad)
        self.assertGreater(student_logits.grad.abs().sum().item(), 0.0)

    def test_forward_kl_simple_loss_matches_manual_formula_and_ignores_teacher_grad(self):
        student_logits = torch.tensor(
            [[[0.3, -0.1, 0.7], [0.5, -0.4, 0.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        teacher_probs = torch.tensor(
            [[[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        teacher_targets = torch.tensor([[2, 0]], dtype=torch.long)

        loss, stats = forward_kl_simple_loss(
            student_logits,
            teacher_targets,
            teacher_probs=teacher_probs,
            temperature=0.7,
            eps=1e-10,
        )

        expected_log_student = gather_action_log_probs(
            student_logits,
            teacher_targets,
            temperature=0.7,
        )
        expected_log_teacher = torch.log(
            teacher_probs.gather(2, teacher_targets.unsqueeze(-1)).squeeze(-1).clamp_min(1e-10)
        )
        expected_loss = -expected_log_student.mean()

        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(stats["log_student_target"], expected_log_student)
        torch.testing.assert_close(stats["log_teacher_target"], expected_log_teacher)

        loss.backward()
        self.assertIsNone(teacher_probs.grad)
        self.assertIsNotNone(student_logits.grad)
        self.assertGreater(student_logits.grad.abs().sum().item(), 0.0)

    def test_forward_kl_full_loss_matches_teacher_forward_kl(self):
        torch.manual_seed(0)
        student_logits = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
        teacher_probs = torch.softmax(torch.randn(2, 3, 4, dtype=torch.float32), dim=-1)

        loss, stats = forward_kl_full_loss(
            student_logits,
            teacher_probs=teacher_probs,
            temperature=1.3,
            eps=1e-10,
        )
        expected_token_kl, expected_ce, expected_entropy = teacher_forward_kl(
            teacher_probs,
            student_logits,
            temperature=1.3,
            eps=1e-10,
        )
        expected_loss = expected_token_kl.mean()

        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(stats["forward_kl"], expected_token_kl)
        torch.testing.assert_close(stats["teacher_ce"], expected_ce)
        torch.testing.assert_close(stats["teacher_entropy"], expected_entropy)

        loss.backward()
        self.assertIsNotNone(student_logits.grad)
        self.assertGreater(student_logits.grad.abs().sum().item(), 0.0)

    def test_forward_kl_full_matches_teacher_forward_kl_across_randomized_cases(self):
        for seed, shape, temperature in (
            (3, (2, 4, 5), None),
            (5, (3, 2, 4), 0.7),
            (7, (2, 3, 6), 1.5),
        ):
            with self.subTest(seed=seed, shape=shape, temperature=temperature):
                torch.manual_seed(seed)
                student_logits = torch.randn(*shape, dtype=torch.float32, requires_grad=True)
                teacher_logits = 4.0 * torch.randn(*shape, dtype=torch.float32)
                teacher_probs = torch.softmax(teacher_logits, dim=-1)

                loss, stats = forward_kl_full_loss(
                    student_logits,
                    teacher_probs=teacher_probs,
                    temperature=temperature,
                    eps=1e-10,
                )
                expected_token_kl, expected_ce, expected_entropy = teacher_forward_kl(
                    teacher_probs,
                    student_logits,
                    temperature=temperature,
                    eps=1e-10,
                )

                torch.testing.assert_close(loss, expected_token_kl.mean())
                torch.testing.assert_close(stats["forward_kl"], expected_token_kl)
                torch.testing.assert_close(stats["teacher_ce"], expected_ce)
                torch.testing.assert_close(stats["teacher_entropy"], expected_entropy)

                loss.backward()
                self.assertTrue(torch.isfinite(student_logits.grad).all().item())
                self.assertGreater(student_logits.grad.abs().sum().item(), 0.0)

    def test_reverse_kl_full_loss_matches_manual_formula(self):
        student_logits = torch.tensor(
            [[[0.8, -0.1, 0.3], [0.2, 0.6, -0.7]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        teacher_probs = torch.tensor(
            [[[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]]],
            dtype=torch.float32,
        )

        loss, stats = reverse_kl_full_loss(
            student_logits,
            teacher_probs=teacher_probs,
            eps=1e-10,
        )

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_probs = student_log_probs.exp()
        teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-10))
        expected_reverse_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        expected_student_teacher_ce = -(student_probs * teacher_log_probs).sum(dim=-1)
        expected_student_entropy = -(student_probs * student_log_probs).sum(dim=-1)
        expected_loss = expected_reverse_kl.mean()

        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(stats["reverse_kl"], expected_reverse_kl)
        torch.testing.assert_close(stats["student_teacher_ce"], expected_student_teacher_ce)
        torch.testing.assert_close(stats["student_entropy"], expected_student_entropy)

        loss.backward()
        self.assertIsNotNone(student_logits.grad)
        self.assertGreater(student_logits.grad.abs().sum().item(), 0.0)

    def test_reverse_kl_full_loss_is_zero_when_student_matches_teacher(self):
        teacher_probs = torch.tensor(
            [[[0.4, 0.5, 0.1], [0.2, 0.3, 0.5]]],
            dtype=torch.float32,
        )
        student_logits = teacher_probs.log().clone().detach().requires_grad_(True)

        loss, stats = reverse_kl_full_loss(
            student_logits,
            teacher_probs=teacher_probs,
            eps=1e-10,
        )

        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            stats["reverse_kl"],
            torch.zeros_like(stats["reverse_kl"]),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            stats["student_teacher_ce"],
            stats["student_entropy"],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_reverse_kl_full_increases_when_student_moves_to_teacher_low_mass_token(self):
        teacher_probs = torch.tensor(
            [[[0.97, 0.02, 0.01], [0.90, 0.09, 0.01]]],
            dtype=torch.float32,
        )
        student_close_logits = torch.tensor(
            [[[4.0, 1.5, -1.0], [3.0, 1.0, -1.5]]],
            dtype=torch.float32,
        )
        student_wrong_logits = torch.tensor(
            [[[-1.0, 0.5, 4.5], [-1.0, 0.5, 4.0]]],
            dtype=torch.float32,
        )

        close_loss, _ = reverse_kl_full_loss(student_close_logits, teacher_probs=teacher_probs)
        wrong_loss, _ = reverse_kl_full_loss(student_wrong_logits, teacher_probs=teacher_probs)

        self.assertGreater(wrong_loss.item(), close_loss.item() + 1.0)

    def test_reverse_kl_tm_loss_handles_sharp_teacher_probs_and_has_finite_gradients(self):
        student_logits = torch.tensor(
            [
                [[4.5, -1.0, -2.5, -3.0], [-2.0, 4.0, -1.5, -3.5], [-3.5, -1.0, 4.2, -0.5]],
                [[-2.0, 3.8, -0.5, -3.0], [4.2, -2.0, -0.8, -4.0], [-3.0, -0.3, 3.5, -1.5]],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        teacher_probs = torch.tensor(
            [
                [[1e-8, 0.999999, 5e-7, 5e-7], [0.999, 5e-4, 4e-4, 1e-4], [5e-6, 5e-6, 0.99998, 1e-5]],
                [[0.9995, 2e-4, 2e-4, 1e-4], [1e-8, 0.9999, 5e-5, 5e-5], [5e-4, 3e-4, 0.9989, 3e-4]],
            ],
            dtype=torch.float32,
        )
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)
        actions = torch.tensor([[1, 0, 2], [0, 1, 2]], dtype=torch.long)
        log_q = torch.tensor(
            [[-0.2, -0.7, -0.3], [-0.6, -0.4, -0.5]],
            dtype=torch.float32,
            requires_grad=True,
        )

        loss, stats = reverse_kl_tm_loss(
            student_logits,
            actions,
            log_q=log_q,
            teacher_probs=teacher_probs,
            eps=1e-10,
        )

        teacher_action_probs = teacher_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        expected_log_teacher = torch.log(teacher_action_probs.clamp_min(1e-10))
        expected_advantage = expected_log_teacher - log_q
        expected_log_p = F.log_softmax(student_logits, dim=-1).gather(2, actions.unsqueeze(-1)).squeeze(-1)
        expected_importance = torch.exp(expected_log_p - log_q.detach())
        expected_loss = -(expected_importance * expected_advantage.detach()).mean()

        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(stats["log_teacher"], expected_log_teacher)
        torch.testing.assert_close(stats["advantage"], expected_advantage)
        torch.testing.assert_close(stats["log_p"], expected_log_p)
        torch.testing.assert_close(stats["importance_weight"], expected_importance)

        loss.backward()
        self.assertTrue(torch.isfinite(student_logits.grad).all().item())
        self.assertGreater(student_logits.grad.abs().sum().item(), 0.0)
        self.assertIsNone(log_q.grad)

    def test_mc_and_full_forward_objectives_diverge_on_the_same_wrong_batch(self):
        student_logits = torch.tensor(
            [[[0.2, 2.3, -2.0], [2.1, -0.4, 0.1]]],
            dtype=torch.float32,
        )
        teacher_probs = torch.tensor(
            [[[0.51, 0.48, 0.01], [0.05, 0.05, 0.90]]],
            dtype=torch.float32,
        )
        teacher_targets = torch.tensor([[0, 2]], dtype=torch.long)

        mc_loss, _ = forward_kl_simple_loss(
            student_logits,
            teacher_targets,
            teacher_probs=teacher_probs,
            temperature=1.0,
            eps=1e-10,
        )
        full_loss, _ = forward_kl_full_loss(
            student_logits,
            teacher_probs=teacher_probs,
            temperature=1.0,
            eps=1e-10,
        )

        self.assertGreater(abs(mc_loss.item() - full_loss.item()), 0.1)

    def test_offline_bc_loss_matches_manual_masked_cross_entropy(self):
        logits = torch.tensor(
            [[[2.0, 0.0, -1.0], [0.0, 1.0, -0.5], [0.3, -0.2, 0.1], [-0.1, 0.2, 0.4]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.tensor([[-1, 1, -1, 2]], dtype=torch.long)

        loss = causal_lm_loss(logits, targets, ignore_index=-1)

        valid_logits = torch.stack((logits[0, 1], logits[0, 3]), dim=0)
        valid_targets = torch.tensor([1, 2], dtype=torch.long)
        expected_loss = F.cross_entropy(valid_logits, valid_targets)
        torch.testing.assert_close(loss, expected_loss)

        loss.backward()
        self.assertTrue(torch.equal(logits.grad[0, 0], torch.zeros_like(logits.grad[0, 0])))
        self.assertTrue(torch.equal(logits.grad[0, 2], torch.zeros_like(logits.grad[0, 2])))
        self.assertGreater(logits.grad[0, 1].abs().sum().item(), 0.0)
        self.assertGreater(logits.grad[0, 3].abs().sum().item(), 0.0)

    def test_offline_teacher_prob_loss_extracts_suffix_and_rejects_mismatches(self):
        logits = torch.tensor(
            [[[2.0, 0.0, -1.0], [0.3, -0.2, 0.1], [-0.1, 0.2, 0.4], [0.0, 0.5, -0.4]]],
            dtype=torch.float32,
        )
        y = torch.tensor([[-1, -1, 2, 1]], dtype=torch.long)
        teacher_probs = torch.tensor(
            [[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]],
            dtype=torch.float32,
        )

        student_suffix_logits, loss, _ = offline_teacher_prob_loss_from_logits(logits, y, teacher_probs)
        self.assertEqual(tuple(student_suffix_logits.shape), (1, 2, 3))
        self.assertGreater(loss.item(), 0.0)

        with self.assertRaisesRegex(ValueError, "final contiguous suffix"):
            offline_teacher_prob_loss_from_logits(
                logits,
                torch.tensor([[-1, 2, -1, 1]], dtype=torch.long),
                teacher_probs,
            )
        with self.assertRaisesRegex(ValueError, "shorter than teacher_probs"):
            offline_teacher_prob_loss_from_logits(
                logits[:, :1, :],
                y[:, :1],
                teacher_probs,
            )
        with self.assertRaisesRegex(ValueError, "vocab size"):
            offline_teacher_prob_loss_from_logits(
                logits,
                y,
                teacher_probs[..., :2],
            )


if __name__ == "__main__":
    unittest.main()
