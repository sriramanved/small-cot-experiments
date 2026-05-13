from __future__ import annotations

import json
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from hydra import compose, initialize_config_dir

from data.modular_addition.task import sample_cot_example_ids_from_rng as sample_modadd_cot_example_ids_from_rng
from data.s5_cot.task import VOCAB_SIZE, sample_cot_example_ids_from_rng
from model import GPT, GPTConfig
from nanogpt.config_schema import materialize_config
from nanogpt.methods.student_prefix import jsd_mc_loss as method_jsd_mc_loss
from nanogpt.methods.student_prefix import reverse_kl_tm_loss as method_reverse_kl_tm_loss
from nanogpt.trainers.nail import run_nail
from nanogpt.trainers.configs import (
    project_nail_config,
    project_opd_config,
    project_pretrain_config,
)
from nanogpt.trainers.opd import run_opd
from nanogpt.trainers.pretrain import run_pretrain
from nanogpt.utils.resolvers import register_resolvers


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def _run_hydra(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON), "-m", "nanogpt.run", *args],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _compose_app(*overrides: str):
    register_resolvers()
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "hydra_configs")):
        raw_cfg = compose(config_name="config", overrides=list(overrides))
    return materialize_config(raw_cfg)


def _can_bind_local_socket() -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def _write_s5_prompt_bank(root: Path, *, m: int, n_train: int, n_val: int, seed: int) -> None:
    rng = random.Random(seed)
    prompt_len = 7 * m + 1
    cot_len = 7 * m

    train_prompt = torch.empty((n_train, prompt_len), dtype=torch.uint8)
    train_cot = torch.empty((n_train, cot_len), dtype=torch.uint8)
    val_prompt = torch.empty((n_val, prompt_len), dtype=torch.uint8)
    val_cot = torch.empty((n_val, cot_len), dtype=torch.uint8)

    for row in range(n_train):
        prompt_ids, cot_ids = sample_cot_example_ids_from_rng(rng, m=m)
        train_prompt[row] = torch.tensor(prompt_ids, dtype=torch.uint8)
        train_cot[row] = torch.tensor(cot_ids, dtype=torch.uint8)
    for row in range(n_val):
        prompt_ids, cot_ids = sample_cot_example_ids_from_rng(rng, m=m)
        val_prompt[row] = torch.tensor(prompt_ids, dtype=torch.uint8)
        val_cot[row] = torch.tensor(cot_ids, dtype=torch.uint8)

    root.mkdir(parents=True, exist_ok=True)
    torch.save(train_prompt, root / "clean_train_prompt_ids.pt")
    torch.save(train_cot, root / "clean_train_cot_ids.pt")
    torch.save(val_prompt, root / "clean_val_prompt_ids.pt")
    torch.save(val_cot, root / "clean_val_cot_ids.pt")
    torch.save(torch.arange(n_train, dtype=torch.long), root / "train_order.pt")
    with open(root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "s5",
                "m": m,
                "n_train": n_train,
                "n_val": n_val,
                "prompt_len": prompt_len,
                "cot_len": cot_len,
                "final_answer_len": 7,
                "seed": seed,
            },
            f,
            indent=2,
        )


def _write_modadd_prompt_bank(root: Path, *, p: int, m: int, n_train: int, n_val: int, seed: int) -> None:
    rng = random.Random(seed)
    prompt_len = m + 1
    cot_len = m

    train_prompt = torch.empty((n_train, prompt_len), dtype=torch.int32)
    train_cot = torch.empty((n_train, cot_len), dtype=torch.int32)
    val_prompt = torch.empty((n_val, prompt_len), dtype=torch.int32)
    val_cot = torch.empty((n_val, cot_len), dtype=torch.int32)

    for row in range(n_train):
        prompt_ids, cot_ids = sample_modadd_cot_example_ids_from_rng(rng, p=p, m=m)
        train_prompt[row] = torch.tensor(prompt_ids, dtype=torch.int32)
        train_cot[row] = torch.tensor(cot_ids, dtype=torch.int32)
    for row in range(n_val):
        prompt_ids, cot_ids = sample_modadd_cot_example_ids_from_rng(rng, p=p, m=m)
        val_prompt[row] = torch.tensor(prompt_ids, dtype=torch.int32)
        val_cot[row] = torch.tensor(cot_ids, dtype=torch.int32)

    root.mkdir(parents=True, exist_ok=True)
    torch.save(train_prompt, root / "clean_train_prompt_ids.pt")
    torch.save(train_cot, root / "clean_train_cot_ids.pt")
    torch.save(val_prompt, root / "clean_val_prompt_ids.pt")
    torch.save(val_cot, root / "clean_val_cot_ids.pt")
    torch.save(torch.arange(n_train, dtype=torch.long), root / "train_order.pt")
    with open(root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "modadd",
                "p": p,
                "m": m,
                "n_train": n_train,
                "n_val": n_val,
                "prompt_len": prompt_len,
                "cot_len": cot_len,
                "final_answer_len": 1,
                "seed": seed,
            },
            f,
            indent=2,
        )


def _write_teacher_checkpoint(root: Path, *, block_size: int) -> None:
    model_args = {
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 16,
        "block_size": block_size,
        "bias": False,
        "vocab_size": VOCAB_SIZE,
        "dropout": 0.0,
    }
    model = GPT(GPTConfig(**model_args))
    root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_args": model_args,
            "model": model.state_dict(),
            "iter_num": 0,
            "best_val_loss": 0.0,
            "config": {},
        },
        root / "ckpt.pt",
    )


def _write_modadd_teacher_checkpoint(root: Path, *, p: int, block_size: int) -> None:
    model_args = {
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 16,
        "block_size": block_size,
        "bias": False,
        "vocab_size": p + 1,
        "dropout": 0.0,
    }
    model = GPT(GPTConfig(**model_args))
    root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_args": model_args,
            "model": model.state_dict(),
            "iter_num": 0,
            "best_val_loss": 0.0,
            "config": {},
        },
        root / "ckpt.pt",
    )


class HydraConfigTests(unittest.TestCase):
    def test_pretrain_projection_rejects_cpu_runtime_with_nccl_backend(self):
        cfg = _compose_app(
            "experiment=s5_noisy_bc",
            "runtime=cpu",
            "runtime.backend=nccl",
        )
        with self.assertRaisesRegex(ValueError, "runtime=cpu requires runtime.backend=gloo"):
            project_pretrain_config(cfg)

    def test_pretrain_projection_rejects_invalid_offline_warm_start_continuation(self):
        cfg = _compose_app(
            "experiment=s5_clean_offline_bc",
            "optim.init_from=warm_start",
            "optim.init_from_ckpt=/tmp/source.ckpt",
            "optim.continue_from_subset_size=2",
            "optim.offline_single_epoch=false",
        )
        with self.assertRaisesRegex(ValueError, "offline_single_epoch=True"):
            project_pretrain_config(cfg)

    def test_opd_projection_rejects_parallel_torchrun(self):
        cfg = _compose_app(
            "experiment=s5_opd",
            "runtime.torchrun.nproc_per_node=2",
        )
        with self.assertRaisesRegex(ValueError, "single-process only"):
            project_opd_config(cfg)

    def test_nail_projection_rejects_parallel_torchrun(self):
        cfg = _compose_app(
            "experiment=s5_nail",
            "runtime.torchrun.nproc_per_node=2",
        )
        with self.assertRaisesRegex(ValueError, "single-process only"):
            project_nail_config(cfg)

    def test_native_student_prefix_projection_rejects_legacy_objective_override(self):
        cfg = _compose_app(
            "experiment=modadd_opd",
            "+task.objective=forward_kl_simple",
        )
        with self.assertRaisesRegex(ValueError, "task.loss and task.teacher_signal"):
            project_opd_config(cfg)

    def test_s5_nail_reverse_mc_fixed_projection_wires_reverse_controls(self):
        cfg = _compose_app(
            "experiment=s5_nail_reverse_mc_fixed",
            "task.rollout_temperature_override=0.2",
            "optim.shuffle_prompts=true",
        )
        projected = project_nail_config(cfg)
        self.assertEqual(projected.loss, "reverse")
        self.assertEqual(projected.teacher_signal, "mc")
        self.assertEqual(projected.rollout_temperature_override, 0.2)
        self.assertTrue(projected.shuffle_prompts)

    def test_s5_nail_reverse_full_projection_wires_full_reverse_controls(self):
        cfg = _compose_app(
            "experiment=s5_nail_reverse_full",
            "optim.shuffle_prompts=true",
        )
        projected = project_nail_config(cfg)
        self.assertEqual(projected.loss, "reverse")
        self.assertEqual(projected.teacher_signal, "full")
        self.assertEqual(projected.rollout_temperature_override, 0.0)
        self.assertTrue(projected.shuffle_prompts)

    def test_modadd_nail_reverse_mc_fixed_projection_wires_reverse_controls(self):
        cfg = _compose_app(
            "experiment=modadd_nail_reverse_mc_fixed",
            "task.rollout_temperature_override=0.2",
            "optim.shuffle_prompts=true",
        )
        projected = project_nail_config(cfg)
        self.assertEqual(projected.task, "modadd")
        self.assertEqual(projected.loss, "reverse")
        self.assertEqual(projected.teacher_signal, "mc")
        self.assertEqual(projected.rollout_temperature_override, 0.2)
        self.assertTrue(projected.shuffle_prompts)

    def test_modadd_nail_mixed_projection_wires_beta_and_run_name(self):
        cfg = _compose_app(
            "experiment=modadd_nail",
            "task.loss=mixed",
            "task.teacher_signal=mc",
            "task.kl_beta=0.5",
        )
        projected = project_nail_config(cfg)
        self.assertEqual(projected.task, "modadd")
        self.assertEqual(projected.loss, "mixed")
        self.assertEqual(projected.teacher_signal, "mc")
        self.assertEqual(projected.kl_beta, 0.5)
        self.assertIn("beta0p5", cfg.run.name)

    def test_modadd_nail_jsd_projection_wires_beta_and_run_name(self):
        cfg = _compose_app(
            "experiment=modadd_nail",
            "task.loss=jsd",
            "task.teacher_signal=mc",
            "task.kl_beta=0.5",
        )
        projected = project_nail_config(cfg)
        self.assertEqual(projected.task, "modadd")
        self.assertEqual(projected.loss, "jsd")
        self.assertEqual(projected.teacher_signal, "mc")
        self.assertEqual(projected.kl_beta, 0.5)
        self.assertIn("jsd_beta0p5", cfg.run.name)

    def test_modadd_nail_forward_run_name_unchanged_without_beta(self):
        cfg = _compose_app("experiment=modadd_nail")
        self.assertNotIn("beta", cfg.run.name)
        self.assertEqual(project_nail_config(cfg).kl_beta, None)

    def test_all_supported_experiments_compose(self):
        experiments = [
            "s5_cot",
            "s5_cot_len21",
            "s5_base",
            "s5_base_len21",
            "s5_prompt_bank",
            "s5_render",
            "s5_clean_offline_bc",
            "s5_noisy_bc",
            "s5_noisy_bc_full_dist",
            "modadd_cot_p7_m21",
            "modadd_base_p7_m30",
            "modadd_noisy_bc",
            "s5_opd",
            "s5_nail",
            "s5_nail_reverse_full",
            "s5_nail_reverse_mc_fixed",
            "s5_nail_reverse_debug",
            "modadd_opd",
            "modadd_nail",
            "modadd_nail_reverse_full",
            "modadd_nail_reverse_mc_fixed",
        ]

        for experiment in experiments:
            with self.subTest(experiment=experiment):
                result = _run_hydra(f"experiment={experiment}", "--cfg", "job", "--resolve")
                self.assertIn("pipeline:", result.stdout)
                self.assertIn("run:", result.stdout)
                self.assertNotIn("experiment:", result.stdout)

    def test_all_supported_sweeps_compose(self):
        sweep_matrix = {
            "s5_clean_offline_subset": "s5_clean_offline_bc",
            "s5_noisy_bc_eta": "s5_noisy_bc",
            "s5_noisy_bc_eta_full_dist": "s5_noisy_bc_full_dist",
            "modadd_noisy_bc_eta": "modadd_noisy_bc",
            "s5_opd_eta": "s5_opd",
            "s5_nail_eta": "s5_nail",
            "modadd_opd_eta": "modadd_opd",
            "modadd_nail_eta": "modadd_nail",
        }

        for sweep, experiment in sweep_matrix.items():
            with self.subTest(sweep=sweep):
                result = _run_hydra(
                    f"experiment={experiment}",
                    f"sweep={sweep}",
                    "--cfg",
                    "hydra",
                    "--resolve",
                )
                self.assertIn("hydra:", result.stdout)
                self.assertIn("params:", result.stdout)

    def test_s5_nail_reverse_mc_fixed_logs_auxiliary_reverse_diagnostics_to_wandb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_bank_dir = root / "prompt_bank"
            teacher_dir = root / "teacher"
            out_dir = root / "nail_reverse_mc_fixed_out"

            _write_s5_prompt_bank(prompt_bank_dir, m=2, n_train=4, n_val=2, seed=17)
            _write_teacher_checkpoint(teacher_dir, block_size=28)

            cfg = project_nail_config(
                _compose_app(
                    "experiment=s5_nail_reverse_mc_fixed",
                    "runtime=cpu",
                    "logging.wandb_log=true",
                    "task.s5_m=2",
                    f"task.teacher_checkpoint={teacher_dir}",
                    f"task.prompt_bank_dir={prompt_bank_dir}",
                    "task.subset_size=4",
                    f"run.out_dir={out_dir}",
                    "optim.batch_size=2",
                    "optim.max_iters=1",
                    "optim.learning_rate=0.001",
                    "optim.warmup_iters=0",
                    "optim.eval_interval=1",
                    "optim.eval_n=2",
                    "optim.eval_batch_size=2",
                    "optim.log_interval=1",
                    "optim.seed=7",
                )
            )

            logged_payloads: list[dict[str, object]] = []

            class FakeWandb:
                def log(self, payload):
                    logged_payloads.append(dict(payload))

                def finish(self):
                    return None

            with mock.patch(
                "nanogpt.trainers.native_student_prefix.maybe_init_wandb",
                return_value=FakeWandb(),
            ):
                run_nail(
                    cfg,
                    launcher_command=[str(PYTHON), "-m", "nanogpt.run", "experiment=s5_nail_reverse_mc_fixed"],
                )

            train_payloads = [payload for payload in logged_payloads if "train/loss" in payload]
            self.assertTrue(train_payloads)
            train_payload = train_payloads[-1]
            for key in (
                "train/log_p",
                "train/importance_weight_mean",
                "train/importance_weight_std",
                "train/importance_weight_max",
                "train/importance_weight_min",
                "train/rollout_log_q_mean",
                "train/aux_log_q_mean",
                "train/aux_equals_rollout_rate",
                "train/pre_clip_grad_norm",
                "train/post_clip_grad_norm",
                "train/grad_clipped",
                "train/clipping_fraction_ema",
                "train/lr",
                "train/param_norm",
            ):
                self.assertIn(key, train_payload)

            checkpoint = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
            self.assertIn("diagnostics_state", checkpoint)
            self.assertIn("clipping_fraction_ema", checkpoint["diagnostics_state"])

    def test_s5_nail_reverse_mc_fixed_uses_rollout_prefixes_and_auxiliary_actions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_bank_dir = root / "prompt_bank"
            teacher_dir = root / "teacher"
            out_dir = root / "nail_reverse_mc_fixed_out"

            _write_s5_prompt_bank(prompt_bank_dir, m=1, n_train=2, n_val=1, seed=19)
            _write_teacher_checkpoint(teacher_dir, block_size=14)

            cfg = project_nail_config(
                _compose_app(
                    "experiment=s5_nail_reverse_mc_fixed",
                    "runtime=cpu",
                    "logging=disabled",
                    "task.s5_m=1",
                    f"task.teacher_checkpoint={teacher_dir}",
                    f"task.prompt_bank_dir={prompt_bank_dir}",
                    "task.subset_size=2",
                    f"run.out_dir={out_dir}",
                    "optim.batch_size=2",
                    "optim.max_iters=1",
                    "optim.learning_rate=0.001",
                    "optim.warmup_iters=0",
                    "optim.eval_interval=1",
                    "optim.eval_n=1",
                    "optim.eval_batch_size=1",
                    "optim.log_interval=1",
                    "optim.seed=7",
                )
            )

            rollout_actions = torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 0]],
                dtype=torch.long,
            )
            rollout_log_q = torch.full((2, 7), -0.7, dtype=torch.float32)
            aux_actions = torch.tensor(
                [[6, 5, 4, 3, 2, 1, 0], [0, 6, 5, 4, 3, 2, 1]],
                dtype=torch.long,
            )
            aux_log_q = torch.full((2, 7), -1.1, dtype=torch.float32)
            teacher_probs = torch.full((2, 7, VOCAB_SIZE), 1.0 / VOCAB_SIZE, dtype=torch.float32)
            captured: dict[str, torch.Tensor] = {}

            def fake_rollout_student(model, prompt_ids, *, target_len, temperature, device, autocast_context):
                del model, temperature, autocast_context
                self.assertEqual(target_len, rollout_actions.size(1))
                prompt = prompt_ids.to(device=device, dtype=torch.long)
                actions = rollout_actions.to(device=device)
                full_seq = torch.cat((prompt, actions), dim=1)
                return full_seq, actions, rollout_log_q.to(device=device)

            def fake_cached_teacher_token_probs(
                model,
                prompt_ids,
                actions,
                *,
                eta,
                teacher_law,
                corruptible_token_ids,
                device,
                autocast_context,
            ):
                del model, prompt_ids, eta, teacher_law, corruptible_token_ids, autocast_context
                captured["teacher_actions"] = actions.detach().cpu().clone()
                return teacher_probs.to(device=device)

            def wrapped_reverse_kl_tm_loss(student_logits, actions, *, log_q, teacher_probs, eps):
                captured["reverse_actions"] = actions.detach().cpu().clone()
                captured["reverse_log_q"] = log_q.detach().cpu().clone()
                return method_reverse_kl_tm_loss(
                    student_logits,
                    actions,
                    log_q=log_q,
                    teacher_probs=teacher_probs,
                    eps=eps,
                )

            with mock.patch(
                "nanogpt.trainers.native_student_prefix.run_eval",
                return_value={
                    "val/loss": 0.0,
                    "val/clean_full_exact": 0.0,
                    "val/clean_final_exact": 0.0,
                },
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.rollout_student",
                side_effect=fake_rollout_student,
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.cached_teacher_token_probs",
                side_effect=fake_cached_teacher_token_probs,
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.sample_student_aux_actions",
                return_value=(aux_actions, aux_log_q),
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.reverse_kl_tm_loss",
                side_effect=wrapped_reverse_kl_tm_loss,
            ):
                run_nail(
                    cfg,
                    launcher_command=[str(PYTHON), "-m", "nanogpt.run", "experiment=s5_nail_reverse_mc_fixed"],
                )

            torch.testing.assert_close(captured["teacher_actions"], rollout_actions)
            torch.testing.assert_close(captured["reverse_actions"], aux_actions)
            torch.testing.assert_close(captured["reverse_log_q"], aux_log_q)

    def test_s5_opd_reverse_mc_keeps_rollout_actions_and_log_q(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_bank_dir = root / "prompt_bank"
            teacher_dir = root / "teacher"
            out_dir = root / "opd_reverse_out"

            _write_s5_prompt_bank(prompt_bank_dir, m=1, n_train=2, n_val=1, seed=19)
            _write_teacher_checkpoint(teacher_dir, block_size=14)

            cfg = project_opd_config(
                _compose_app(
                    "experiment=s5_opd",
                    "runtime=cpu",
                    "logging=disabled",
                    "task.s5_m=1",
                    f"task.teacher_checkpoint={teacher_dir}",
                    f"task.prompt_bank_dir={prompt_bank_dir}",
                    "task.subset_size=2",
                    f"run.out_dir={out_dir}",
                    "optim.batch_size=2",
                    "optim.max_iters=1",
                    "optim.learning_rate=0.001",
                    "optim.warmup_iters=0",
                    "optim.eval_interval=1",
                    "optim.eval_n=1",
                    "optim.eval_batch_size=1",
                    "optim.log_interval=1",
                    "optim.seed=7",
                )
            )

            rollout_actions = torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 0]],
                dtype=torch.long,
            )
            rollout_log_q = torch.full((2, 7), -0.7, dtype=torch.float32)
            teacher_probs = torch.full((2, 7, VOCAB_SIZE), 1.0 / VOCAB_SIZE, dtype=torch.float32)
            captured: dict[str, torch.Tensor] = {}

            def fake_rollout_student(model, prompt_ids, *, target_len, temperature, device, autocast_context):
                del model, temperature, autocast_context
                self.assertEqual(target_len, rollout_actions.size(1))
                prompt = prompt_ids.to(device=device, dtype=torch.long)
                actions = rollout_actions.to(device=device)
                full_seq = torch.cat((prompt, actions), dim=1)
                return full_seq, actions, rollout_log_q.to(device=device)

            def fake_cached_teacher_token_probs(
                model,
                prompt_ids,
                actions,
                *,
                eta,
                teacher_law,
                corruptible_token_ids,
                device,
                autocast_context,
            ):
                del model, prompt_ids, eta, teacher_law, corruptible_token_ids, autocast_context
                captured["teacher_actions"] = actions.detach().cpu().clone()
                return teacher_probs.to(device=device)

            def wrapped_reverse_kl_tm_loss(student_logits, actions, *, log_q, teacher_probs, eps):
                captured["reverse_actions"] = actions.detach().cpu().clone()
                captured["reverse_log_q"] = log_q.detach().cpu().clone()
                return method_reverse_kl_tm_loss(
                    student_logits,
                    actions,
                    log_q=log_q,
                    teacher_probs=teacher_probs,
                    eps=eps,
                )

            with mock.patch(
                "nanogpt.trainers.native_student_prefix.run_eval",
                return_value={
                    "val/loss": 0.0,
                    "val/clean_full_exact": 0.0,
                    "val/clean_final_exact": 0.0,
                },
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.rollout_student",
                side_effect=fake_rollout_student,
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.cached_teacher_token_probs",
                side_effect=fake_cached_teacher_token_probs,
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.sample_student_aux_actions",
                side_effect=AssertionError("OPD reverse-MC should keep rollout actions"),
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.reverse_kl_tm_loss",
                side_effect=wrapped_reverse_kl_tm_loss,
            ):
                run_opd(
                    cfg,
                    launcher_command=[str(PYTHON), "-m", "nanogpt.run", "experiment=s5_opd"],
                )

            torch.testing.assert_close(captured["teacher_actions"], rollout_actions)
            torch.testing.assert_close(captured["reverse_actions"], rollout_actions)
            torch.testing.assert_close(captured["reverse_log_q"], rollout_log_q)

    def test_modadd_nail_reverse_mc_fixed_uses_rollout_prefixes_and_auxiliary_actions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_bank_dir = root / "prompt_bank"
            teacher_dir = root / "teacher"
            out_dir = root / "modadd_nail_reverse_mc_fixed_out"

            _write_modadd_prompt_bank(prompt_bank_dir, p=3, m=2, n_train=2, n_val=1, seed=19)
            _write_modadd_teacher_checkpoint(teacher_dir, p=3, block_size=4)

            cfg = project_nail_config(
                _compose_app(
                    "experiment=modadd_nail_reverse_mc_fixed",
                    "runtime=cpu",
                    "logging=disabled",
                    "task.modadd_p=3",
                    "task.modadd_m=2",
                    f"task.teacher_checkpoint={teacher_dir}",
                    f"task.prompt_bank_dir={prompt_bank_dir}",
                    "task.subset_size=2",
                    f"run.out_dir={out_dir}",
                    "optim.batch_size=2",
                    "optim.max_iters=1",
                    "optim.learning_rate=0.001",
                    "optim.warmup_iters=0",
                    "optim.eval_interval=1",
                    "optim.eval_n=1",
                    "optim.eval_batch_size=1",
                    "optim.log_interval=1",
                    "optim.seed=7",
                )
            )

            rollout_actions = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            rollout_log_q = torch.full((2, 2), -0.7, dtype=torch.float32)
            aux_actions = torch.tensor([[2, 1], [0, 2]], dtype=torch.long)
            aux_log_q = torch.full((2, 2), -1.1, dtype=torch.float32)
            teacher_probs = torch.full((2, 2, 4), 0.25, dtype=torch.float32)
            captured: dict[str, torch.Tensor] = {}

            def fake_rollout_student(model, prompt_ids, *, target_len, temperature, device, autocast_context):
                del model, temperature, autocast_context
                self.assertEqual(target_len, rollout_actions.size(1))
                prompt = prompt_ids.to(device=device, dtype=torch.long)
                actions = rollout_actions.to(device=device)
                full_seq = torch.cat((prompt, actions), dim=1)
                return full_seq, actions, rollout_log_q.to(device=device)

            def fake_cached_teacher_token_probs(
                model,
                prompt_ids,
                actions,
                *,
                eta,
                teacher_law,
                corruptible_token_ids,
                device,
                autocast_context,
            ):
                del model, prompt_ids, eta, teacher_law, corruptible_token_ids, autocast_context
                captured["teacher_actions"] = actions.detach().cpu().clone()
                return teacher_probs.to(device=device)

            def wrapped_reverse_kl_tm_loss(student_logits, actions, *, log_q, teacher_probs, eps):
                captured["reverse_actions"] = actions.detach().cpu().clone()
                captured["reverse_log_q"] = log_q.detach().cpu().clone()
                return method_reverse_kl_tm_loss(
                    student_logits,
                    actions,
                    log_q=log_q,
                    teacher_probs=teacher_probs,
                    eps=eps,
                )

            with mock.patch(
                "nanogpt.trainers.native_student_prefix.run_eval",
                return_value={
                    "val/loss": 0.0,
                    "val/clean_full_exact": 0.0,
                    "val/clean_final_exact": 0.0,
                },
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.rollout_student",
                side_effect=fake_rollout_student,
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.cached_teacher_token_probs",
                side_effect=fake_cached_teacher_token_probs,
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.sample_student_aux_actions",
                return_value=(aux_actions, aux_log_q),
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.reverse_kl_tm_loss",
                side_effect=wrapped_reverse_kl_tm_loss,
            ):
                run_nail(
                    cfg,
                    launcher_command=[str(PYTHON), "-m", "nanogpt.run", "experiment=modadd_nail_reverse_mc_fixed"],
                )

            torch.testing.assert_close(captured["teacher_actions"], rollout_actions)
            torch.testing.assert_close(captured["reverse_actions"], aux_actions)
            torch.testing.assert_close(captured["reverse_log_q"], aux_log_q)

    def test_modadd_nail_beta_endpoint_losses_skip_unused_sampling_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_bank_dir = root / "prompt_bank"
            teacher_dir = root / "teacher"
            _write_modadd_prompt_bank(prompt_bank_dir, p=3, m=2, n_train=2, n_val=1, seed=19)
            _write_modadd_teacher_checkpoint(teacher_dir, p=3, block_size=4)

            rollout_actions = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            rollout_log_q = torch.full((2, 2), -0.7, dtype=torch.float32)
            aux_actions = torch.tensor([[2, 1], [0, 2]], dtype=torch.long)
            aux_log_q = torch.full((2, 2), -1.1, dtype=torch.float32)
            teacher_probs = torch.full((2, 2, 4), 0.25, dtype=torch.float32)

            def fake_rollout_student(model, prompt_ids, *, target_len, temperature, device, autocast_context):
                del model, temperature, autocast_context
                self.assertEqual(target_len, rollout_actions.size(1))
                prompt = prompt_ids.to(device=device, dtype=torch.long)
                actions = rollout_actions.to(device=device)
                full_seq = torch.cat((prompt, actions), dim=1)
                return full_seq, actions, rollout_log_q.to(device=device)

            def fake_cached_teacher_token_probs(
                model,
                prompt_ids,
                actions,
                *,
                eta,
                teacher_law,
                corruptible_token_ids,
                device,
                autocast_context,
            ):
                del model, prompt_ids, actions, eta, teacher_law, corruptible_token_ids, autocast_context
                return teacher_probs.to(device=device)

            for loss_name in ("mixed", "jsd"):
                for beta in (0.0, 1.0):
                    with self.subTest(loss=loss_name, beta=beta):
                        out_dir = root / f"{loss_name}_beta_{str(beta).replace('.', 'p')}"
                        cfg = project_nail_config(
                            _compose_app(
                                "experiment=modadd_nail",
                                "runtime=cpu",
                                "logging=disabled",
                                "task.modadd_p=3",
                                "task.modadd_m=2",
                                f"task.loss={loss_name}",
                                "task.teacher_signal=mc",
                                f"task.kl_beta={beta}",
                                f"task.teacher_checkpoint={teacher_dir}",
                                f"task.prompt_bank_dir={prompt_bank_dir}",
                                "task.subset_size=2",
                                f"run.out_dir={out_dir}",
                                "optim.batch_size=2",
                                "optim.max_iters=1",
                                "optim.learning_rate=0.001",
                                "optim.warmup_iters=0",
                                "optim.eval_interval=1",
                                "optim.eval_n=1",
                                "optim.eval_batch_size=1",
                                "optim.log_interval=1",
                                "optim.seed=7",
                            )
                        )

                        teacher_patch = (
                            mock.patch(
                                "nanogpt.trainers.native_student_prefix.sample_teacher_actions",
                                return_value=rollout_actions,
                            )
                            if beta == 0.0
                            else mock.patch(
                                "nanogpt.trainers.native_student_prefix.sample_teacher_actions",
                                side_effect=AssertionError("beta=1 should skip teacher-target sampling"),
                            )
                        )
                        aux_patch = (
                            mock.patch(
                                "nanogpt.trainers.native_student_prefix.sample_student_aux_actions",
                                side_effect=AssertionError("beta=0 should skip reverse auxiliary sampling"),
                            )
                            if beta == 0.0
                            else mock.patch(
                                "nanogpt.trainers.native_student_prefix.sample_student_aux_actions",
                                return_value=(aux_actions, aux_log_q),
                            )
                        )

                        with mock.patch(
                            "nanogpt.trainers.native_student_prefix.run_eval",
                            return_value={
                                "val/loss": 0.0,
                                "val/clean_full_exact": 0.0,
                                "val/clean_final_exact": 0.0,
                            },
                        ), mock.patch(
                            "nanogpt.trainers.native_student_prefix.rollout_student",
                            side_effect=fake_rollout_student,
                        ), mock.patch(
                            "nanogpt.trainers.native_student_prefix.cached_teacher_token_probs",
                            side_effect=fake_cached_teacher_token_probs,
                        ), teacher_patch, aux_patch:
                            run_nail(
                                cfg,
                                launcher_command=[
                                    str(PYTHON),
                                    "-m",
                                    "nanogpt.run",
                                    "experiment=modadd_nail",
                                    f"task.loss={loss_name}",
                                ],
                            )

                        with open(out_dir / "run_meta.json", "r", encoding="utf-8") as f:
                            run_meta = json.load(f)
                        self.assertEqual(run_meta["loss"], loss_name)
                        self.assertEqual(run_meta["kl_beta"], beta)

            out_dir = root / "jsd_beta_0p5"
            cfg = project_nail_config(
                _compose_app(
                    "experiment=modadd_nail",
                    "runtime=cpu",
                    "logging=disabled",
                    "task.modadd_p=3",
                    "task.modadd_m=2",
                    "task.loss=jsd",
                    "task.teacher_signal=mc",
                    "task.kl_beta=0.5",
                    f"task.teacher_checkpoint={teacher_dir}",
                    f"task.prompt_bank_dir={prompt_bank_dir}",
                    "task.subset_size=2",
                    f"run.out_dir={out_dir}",
                    "optim.batch_size=2",
                    "optim.max_iters=1",
                    "optim.learning_rate=0.001",
                    "optim.warmup_iters=0",
                    "optim.eval_interval=1",
                    "optim.eval_n=1",
                    "optim.eval_batch_size=1",
                    "optim.log_interval=1",
                    "optim.seed=7",
                )
            )
            captured: dict[str, bool] = {}

            def wrapped_jsd_mc_loss(
                student_logits,
                teacher_targets,
                student_actions,
                *,
                teacher_probs,
                beta,
                temperature,
                eps,
            ):
                captured["jsd_called"] = True
                return method_jsd_mc_loss(
                    student_logits,
                    teacher_targets,
                    student_actions,
                    teacher_probs=teacher_probs,
                    beta=beta,
                    temperature=temperature,
                    eps=eps,
                )

            with mock.patch(
                "nanogpt.trainers.native_student_prefix.run_eval",
                return_value={
                    "val/loss": 0.0,
                    "val/clean_full_exact": 0.0,
                    "val/clean_final_exact": 0.0,
                },
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.rollout_student",
                side_effect=fake_rollout_student,
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.cached_teacher_token_probs",
                side_effect=fake_cached_teacher_token_probs,
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.sample_teacher_actions",
                return_value=rollout_actions,
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.sample_student_aux_actions",
                return_value=(aux_actions, aux_log_q),
            ), mock.patch(
                "nanogpt.trainers.native_student_prefix.jsd_mc_loss",
                side_effect=wrapped_jsd_mc_loss,
            ):
                run_nail(
                    cfg,
                    launcher_command=[
                        str(PYTHON),
                        "-m",
                        "nanogpt.run",
                        "experiment=modadd_nail",
                        "task.loss=jsd",
                    ],
                )

            self.assertTrue(captured["jsd_called"])
            with open(out_dir / "run_meta.json", "r", encoding="utf-8") as f:
                run_meta = json.load(f)
            self.assertEqual(run_meta["loss"], "jsd")
            self.assertEqual(run_meta["kl_beta"], 0.5)

    def test_submitit_aics_config_composes(self):
        result = _run_hydra(
            "hydra/launcher=submitit_slurm",
            "cluster=aics",
            "experiment=s5_opd",
            "--cfg",
            "hydra",
            "--resolve",
        )
        self.assertIn("hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher", result.stdout)
        self.assertIn("submitit_folder:", result.stdout)
        self.assertIn("timeout_min: 720", result.stdout)

    def test_local_multirun_smoke_executes_two_jobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_bank_dir = root / "prompt_bank"
            teacher_dir = root / "teacher"
            output_root = root / "outputs"

            _write_s5_prompt_bank(prompt_bank_dir, m=2, n_train=4, n_val=2, seed=17)
            _write_teacher_checkpoint(teacher_dir, block_size=28)

            _run_hydra(
                "--multirun",
                "experiment=s5_opd",
                "runtime=cpu",
                "logging=disabled",
                "task.s5_m=2",
                f"task.teacher_checkpoint={teacher_dir}",
                f"task.prompt_bank_dir={prompt_bank_dir}",
                "task.subset_size=4",
                "task.eta=0.1,0.2",
                "optim.batch_size=2",
                "optim.max_iters=1",
                "optim.learning_rate=0.001",
                "optim.warmup_iters=0",
                "optim.eval_interval=1",
                "optim.eval_n=2",
                "optim.eval_batch_size=2",
                "optim.log_interval=1",
                f"run.output_root={output_root}",
            )

            eta_a = output_root / "out-s5-opd-reverse-mc-m2-n4-eta0p1-distributional_noise-seed1337"
            eta_b = output_root / "out-s5-opd-reverse-mc-m2-n4-eta0p2-distributional_noise-seed1337"
            self.assertTrue((eta_a / "completed.txt").exists())
            self.assertTrue((eta_b / "completed.txt").exists())
            self.assertTrue((eta_a / "launcher_config.json").exists())
            self.assertTrue((eta_b / "launcher_command.txt").exists())

    def test_pretrain_ddp_bootstrap_writes_worker_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "bootstrap-out"
            cfg = project_pretrain_config(
                _compose_app(
                    "experiment=s5_cot_len21",
                    "runtime=cpu",
                    "runtime.torchrun.nproc_per_node=2",
                    "logging=disabled",
                    "model=tiny_debug",
                    "model.block_size=28",
                    "task.s5_m=2",
                    f"run.out_dir={out_dir}",
                    "optim.batch_size=1",
                    "optim.gradient_accumulation_steps=2",
                    "optim.learning_rate=0.001",
                    "optim.max_iters=1",
                    "optim.eval_interval=1",
                    "optim.eval_iters=1",
                    "optim.always_save_checkpoint=true",
                    "optim.final_eval_on_exit=true",
                    "optim.s5_eval_metrics=true",
                    "optim.s5_eval_n=2",
                    "optim.s5_eval_batch_size=2",
                )
            )

            with mock.patch("nanogpt.trainers.pretrain.subprocess.run") as run_mock:
                run_pretrain(
                    cfg,
                    launcher_command=[str(PYTHON), "-m", "nanogpt.run", "experiment=s5_cot_len21"],
                )

            run_mock.assert_called_once()
            command = run_mock.call_args.args[0]
            self.assertIn("-m", command)
            self.assertIn("nanogpt.workers.pretrain", command)
            self.assertTrue((out_dir / "worker_config.json").exists())
            self.assertTrue((out_dir / "launcher_command.txt").exists())
            with open(out_dir / "worker_config.json", "r", encoding="utf-8") as f:
                worker_cfg = json.load(f)
            self.assertEqual(worker_cfg["backend"], "gloo")
            self.assertEqual(worker_cfg["device"], "cpu")

    def test_pretrain_cpu_ddp_smoke_writes_worker_config(self):
        if not _can_bind_local_socket():
            self.skipTest("local socket binding is unavailable in this sandbox")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "ddp-out"

            _run_hydra(
                "experiment=s5_cot_len21",
                "runtime=cpu",
                "runtime.torchrun.nproc_per_node=2",
                "runtime.torchrun.standalone=false",
                "runtime.torchrun.master_addr=127.0.0.1",
                "runtime.torchrun.master_port=29671",
                "logging=disabled",
                "model=tiny_debug",
                "model.block_size=28",
                "task.s5_m=2",
                f"run.out_dir={out_dir}",
                "optim.batch_size=1",
                "optim.gradient_accumulation_steps=2",
                "optim.learning_rate=0.001",
                "optim.max_iters=1",
                "optim.eval_interval=1",
                "optim.eval_iters=1",
                "optim.always_save_checkpoint=true",
                "optim.final_eval_on_exit=true",
                "optim.s5_eval_metrics=true",
                "optim.s5_eval_n=2",
                "optim.s5_eval_batch_size=2",
            )

            self.assertTrue((out_dir / "worker_config.json").exists())
            self.assertTrue((out_dir / "launcher_command.txt").exists())
            self.assertTrue((out_dir / "completed.txt").exists())
            with open(out_dir / "worker_config.json", "r", encoding="utf-8") as f:
                worker_cfg = json.load(f)
            self.assertEqual(worker_cfg["backend"], "gloo")
            self.assertEqual(worker_cfg["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
