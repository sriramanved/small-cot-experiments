from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from data.s5_cot.task import VOCAB_SIZE, sample_cot_example_ids_from_rng
from model import GPT, GPTConfig


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
    def test_all_supported_experiments_compose(self):
        experiments = [
            "shakespeare_char",
            "gpt2",
            "finetune_shakespeare",
            "eval_gpt2",
            "eval_gpt2_medium",
            "eval_gpt2_large",
            "eval_gpt2_xl",
            "s5_cot_len21",
            "s5_base_len21",
            "s5_clean_offline_bc",
            "s5_noisy_bc",
            "s5_noisy_bc_full_dist",
            "modadd_cot_p7_m21",
            "modadd_base_p7_m30",
            "modadd_clean_offline_bc",
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
            "modadd_clean_offline_subset": "modadd_clean_offline_bc",
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

            eta_a = output_root / "out-s5-opd-reverse_kl_tm-n4-eta0p1-distributional_noise-t1p0"
            eta_b = output_root / "out-s5-opd-reverse_kl_tm-n4-eta0p2-distributional_noise-t1p0"
            self.assertTrue((eta_a / "completed.txt").exists())
            self.assertTrue((eta_b / "completed.txt").exists())
            self.assertTrue((eta_a / "launcher_config.json").exists())
            self.assertTrue((eta_b / "launcher_command.txt").exists())


if __name__ == "__main__":
    unittest.main()
