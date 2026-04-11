from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from data.s5_cot.opd import (
    forward_kl_full_loss,
    forward_kl_simple_loss,
    gather_action_log_probs,
    reverse_kl_tm_loss,
    teacher_forward_kl,
)
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


if __name__ == "__main__":
    unittest.main()
