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

from data.s5_cot.task import VOCAB_SIZE, sample_cot_example_ids_from_rng
from model import GPT, GPTConfig
from nanogpt.config_schema import materialize_config
from nanogpt.trainers.configs import (
    project_opd_config,
    project_opd_hf_config,
    project_pretrain_config,
)
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

    def test_opd_hf_projection_rejects_parallel_torchrun(self):
        cfg = _compose_app(
            "experiment=s5_opd_hf",
            "runtime.torchrun.nproc_per_node=2",
        )
        with self.assertRaisesRegex(ValueError, "single-process only"):
            project_opd_hf_config(cfg)

    def test_all_supported_experiments_compose(self):
        experiments = [
            "shakespeare_char",
            "gpt2",
            "finetune_shakespeare",
            "eval_gpt2",
            "eval_gpt2_medium",
            "eval_gpt2_large",
            "eval_gpt2_xl",
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
            "modadd_opd",
            "s5_opd_hf",
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
            "modadd_opd_eta": "modadd_opd",
            "s5_opd_hf_eta": "s5_opd_hf",
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

            eta_a = output_root / "out-s5-opd-reverse_kl_tm-m2-n4-eta0p1-distributional_noise-rollgreedy-studt1p0-seed1337"
            eta_b = output_root / "out-s5-opd-reverse_kl_tm-m2-n4-eta0p2-distributional_noise-rollgreedy-studt1p0-seed1337"
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
