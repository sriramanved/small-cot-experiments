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

from data.s5_cot.offline_render import render_offline_dataset
from data.s5_cot.task import VOCAB_SIZE, sample_cot_example_ids_from_rng
from model import GPT, GPTConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"


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

    g = torch.Generator()
    g.manual_seed(seed)
    train_order = torch.randperm(n_train, generator=g)

    root.mkdir(parents=True, exist_ok=True)
    torch.save(train_prompt, root / "clean_train_prompt_ids.pt")
    torch.save(train_cot, root / "clean_train_cot_ids.pt")
    torch.save(val_prompt, root / "clean_val_prompt_ids.pt")
    torch.save(val_cot, root / "clean_val_cot_ids.pt")
    torch.save(train_order, root / "train_order.pt")
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


def _write_teacher_checkpoint(root: Path, *, vocab_size: int, block_size: int) -> None:
    model_args = {
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 16,
        "block_size": block_size,
        "bias": False,
        "vocab_size": vocab_size,
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


class S5FullDistributionOfflineBCTests(unittest.TestCase):
    def test_render_teacher_probs_rejects_unsupported_rollout_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_bank_dir = root / "prompt_bank"
            teacher_dir = root / "teacher"
            dataset_dir = root / "dataset"

            _write_s5_prompt_bank(prompt_bank_dir, m=2, n_train=4, n_val=2, seed=3)
            _write_teacher_checkpoint(teacher_dir, vocab_size=VOCAB_SIZE, block_size=28)

            with self.assertRaisesRegex(ValueError, "only supports rollout_mode='sample_then_corrupt'"):
                render_offline_dataset(
                    teacher_checkpoint=str(teacher_dir),
                    prompt_bank_dir=str(prompt_bank_dir),
                    save_dir=str(dataset_dir),
                    subset_size=4,
                    eta=0.2,
                    rollout_mode="greedy_then_corrupt",
                    target_mode="teacher_probs",
                    gen_batch_size=2,
                    device="cpu",
                    dtype_name="float32",
                    seed=7,
                )

    def test_render_teacher_prob_dataset_writes_probs_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_bank_dir = root / "prompt_bank"
            teacher_dir = root / "teacher"
            dataset_dir = root / "dataset"

            _write_s5_prompt_bank(prompt_bank_dir, m=2, n_train=4, n_val=2, seed=11)
            _write_teacher_checkpoint(teacher_dir, vocab_size=VOCAB_SIZE, block_size=28)

            render_offline_dataset(
                teacher_checkpoint=str(teacher_dir),
                prompt_bank_dir=str(prompt_bank_dir),
                save_dir=str(dataset_dir),
                subset_size=4,
                eta=0.2,
                rollout_mode="sample_then_corrupt",
                target_mode="teacher_probs",
                gen_batch_size=2,
                device="cpu",
                dtype_name="float32",
                seed=7,
            )

            teacher_probs = torch.load(dataset_dir / "train_teacher_probs.pt", map_location="cpu")
            self.assertEqual(tuple(teacher_probs.shape), (4, 14, VOCAB_SIZE))
            self.assertEqual(teacher_probs.dtype, torch.float16)
            torch.testing.assert_close(
                teacher_probs.float().sum(dim=-1),
                torch.ones((4, 14), dtype=torch.float32),
                atol=2e-3,
                rtol=2e-3,
            )

            meta = json.loads((dataset_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["train_target_type"], "teacher_probs")
            self.assertEqual(meta["teacher_law"], "distributional_noise")
            self.assertEqual(meta["train_decode_mode"], "sample_then_corrupt")
            self.assertEqual(meta["vocab_size"], VOCAB_SIZE)
            self.assertTrue((dataset_dir / "train_x.pt").exists())
            self.assertTrue((dataset_dir / "train_y.pt").exists())
            self.assertTrue((dataset_dir / "val_x.pt").exists())
            self.assertTrue((dataset_dir / "val_y.pt").exists())

    def test_train_py_offline_s5_teacher_probs_runs_and_writes_eval_artifacts(self):
        dataset_name = "s5_noisy_offline_full_dist_sample_then_corrupt_test_full_dist"
        dataset_dir = DATA_ROOT / dataset_name
        prompt_bank_dir = DATA_ROOT / "test_s5_prompt_bank_for_full_dist_train"
        teacher_dir = Path(tempfile.mkdtemp(prefix="s5-full-dist-teacher-"))
        out_dir = Path(tempfile.mkdtemp(prefix="s5-full-dist-out-"))

        try:
            _write_s5_prompt_bank(prompt_bank_dir, m=2, n_train=4, n_val=2, seed=17)
            _write_teacher_checkpoint(teacher_dir, vocab_size=VOCAB_SIZE, block_size=28)

            render_offline_dataset(
                teacher_checkpoint=str(teacher_dir),
                prompt_bank_dir=str(prompt_bank_dir),
                save_dir=str(dataset_dir),
                subset_size=4,
                eta=0.2,
                rollout_mode="sample_then_corrupt",
                target_mode="teacher_probs",
                gen_batch_size=2,
                device="cpu",
                dtype_name="float32",
                seed=5,
            )

            cmd = [
                sys.executable,
                "train.py",
                "config/train_s5_noisy_bc.py",
                "--dataset=" + dataset_name,
                "--out_dir=" + str(out_dir),
                "--device=cpu",
                "--dtype=float32",
                "--compile=False",
                "--offline_target_type=teacher_probs",
                "--n_layer=1",
                "--n_head=1",
                "--n_embd=16",
                "--block_size=28",
                "--batch_size=2",
                "--gradient_accumulation_steps=1",
                "--learning_rate=0.001",
                "--max_iters=20",
                "--eval_interval=2",
                "--eval_iters=1",
                "--always_save_checkpoint=True",
                "--offline_single_epoch=True",
                "--offline_eval_full=False",
                "--offline_train_shuffle=False",
                "--final_eval_on_exit=True",
                "--s5_eval_metrics=True",
                "--s5_eval_clean_train_loss=True",
                "--s5_eval_n=2",
                "--s5_eval_batch_size=2",
                "--wandb_log=False",
            ]
            subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            checkpoint = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["config"]["offline_target_type"], "teacher_probs")
            self.assertTrue((out_dir / "last_eval.json").exists())
            self.assertTrue((out_dir / "eval_history.jsonl").exists())
            self.assertTrue((out_dir / "completed.txt").exists())
            last_eval = json.loads((out_dir / "last_eval.json").read_text(encoding="utf-8"))
            self.assertIn("val/clean_full_exact", last_eval)
            self.assertIn("val/clean_final_exact", last_eval)
            self.assertIn("train/clean_oracle_loss_eval", last_eval)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
            shutil.rmtree(dataset_dir, ignore_errors=True)
            shutil.rmtree(prompt_bank_dir, ignore_errors=True)
            shutil.rmtree(teacher_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
