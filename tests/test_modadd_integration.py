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

from data.modular_addition.task import sample_cot_example_ids_from_rng
from data.synthetic.prompt_bank import build_xy_from_prompt_and_target
from model import GPT, GPTConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"


def _write_prompt_bank(root: Path, *, p: int, m: int, n_train: int, n_val: int, seed: int) -> None:
    rng = random.Random(seed)
    prompt_len = m + 1
    cot_len = m

    train_prompt = torch.empty((n_train, prompt_len), dtype=torch.int32)
    train_cot = torch.empty((n_train, cot_len), dtype=torch.int32)
    val_prompt = torch.empty((n_val, prompt_len), dtype=torch.int32)
    val_cot = torch.empty((n_val, cot_len), dtype=torch.int32)

    for row in range(n_train):
        prompt_ids, cot_ids = sample_cot_example_ids_from_rng(rng, p=p, m=m)
        train_prompt[row] = torch.tensor(prompt_ids, dtype=torch.int32)
        train_cot[row] = torch.tensor(cot_ids, dtype=torch.int32)
    for row in range(n_val):
        prompt_ids, cot_ids = sample_cot_example_ids_from_rng(rng, p=p, m=m)
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
                "prompt_len": prompt_len,
                "cot_len": cot_len,
                "final_answer_len": 1,
                "n_train": n_train,
                "n_val": n_val,
                "seed": seed,
            },
            f,
            indent=2,
        )


def _write_offline_dataset(root: Path, *, prompt_bank_dir: Path, subset_size: int, eta: float) -> None:
    train_prompt = torch.load(prompt_bank_dir / "clean_train_prompt_ids.pt", map_location="cpu")
    train_cot = torch.load(prompt_bank_dir / "clean_train_cot_ids.pt", map_location="cpu")
    val_prompt = torch.load(prompt_bank_dir / "clean_val_prompt_ids.pt", map_location="cpu")
    val_cot = torch.load(prompt_bank_dir / "clean_val_cot_ids.pt", map_location="cpu")
    subset_idx = torch.arange(subset_size, dtype=torch.long)

    train_x, train_y = build_xy_from_prompt_and_target(train_prompt[:subset_size], train_cot[:subset_size])
    val_x, val_y = build_xy_from_prompt_and_target(val_prompt, val_cot)

    root.mkdir(parents=True, exist_ok=True)
    torch.save(train_x, root / "train_x.pt")
    torch.save(train_y, root / "train_y.pt")
    torch.save(val_x, root / "val_x.pt")
    torch.save(val_y, root / "val_y.pt")
    torch.save(subset_idx, root / "subset_indices.pt")
    torch.save(train_prompt[:subset_size], root / "clean_train_prompt_ids.pt")
    torch.save(train_cot[:subset_size], root / "clean_train_cot_ids.pt")
    torch.save(val_prompt, root / "clean_val_prompt_ids.pt")
    torch.save(val_cot, root / "clean_val_cot_ids.pt")
    with open(root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "modadd",
                "p": int(train_prompt.max().item()),
                "m": int(train_cot.size(1)),
                "prompt_len": int(train_prompt.size(1)),
                "cot_len": int(train_cot.size(1)),
                "final_answer_len": 1,
                "subset_size": subset_size,
                "eta": eta,
                "prompt_bank_dir": str(prompt_bank_dir),
                "train_decode_mode": "greedy_then_corrupt",
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


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_state_dicts_equal(testcase: unittest.TestCase, lhs: dict, rhs: dict) -> None:
    testcase.assertEqual(set(lhs), set(rhs))
    for key in lhs:
        left_value = lhs[key]
        right_value = rhs[key]
        if torch.is_tensor(left_value):
            torch.testing.assert_close(left_value, right_value)
        else:
            testcase.assertEqual(left_value, right_value)


def _run_train_opd(
    *,
    prompt_bank_dir: Path,
    teacher_dir: Path,
    out_dir: Path,
    objective: str,
    max_iters: int,
    init_from: str = "scratch",
    seed: int = 7,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "train_opd.py",
        "--task=modadd",
        "--teacher_checkpoint=" + str(teacher_dir),
        "--prompt_bank_dir=" + str(prompt_bank_dir),
        "--subset_size=8",
        "--eta=0.1",
        "--teacher_law=distributional_noise",
        "--objective=" + objective,
        "--out_dir=" + str(out_dir),
        "--init_from=" + init_from,
        "--batch_size=2",
        "--max_iters=" + str(max_iters),
        "--learning_rate=0.001",
        "--warmup_iters=2",
        "--eval_interval=2",
        "--eval_n=2",
        "--eval_batch_size=2",
        "--log_interval=1",
        "--save_interval=2",
        "--device=cpu",
        "--dtype=float32",
        "--seed=" + str(seed),
    ]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


class ModularAdditionIntegrationTests(unittest.TestCase):
    def test_train_py_offline_modadd_uses_dataset_meta_and_writes_eval_artifacts(self):
        dataset_name = "modadd_clean_offline_p3_m4_datasetmeta_test"
        dataset_dir = DATA_ROOT / dataset_name
        prompt_bank_dir = DATA_ROOT / "test_modadd_prompt_bank_for_train"
        out_dir = Path(tempfile.mkdtemp(prefix="modadd-train-out-"))

        try:
            _write_prompt_bank(prompt_bank_dir, p=3, m=4, n_train=4, n_val=2, seed=7)
            _write_offline_dataset(dataset_dir, prompt_bank_dir=prompt_bank_dir, subset_size=4, eta=0.0)

            cmd = [
                sys.executable,
                "train.py",
                "--dataset=" + dataset_name,
                "--out_dir=" + str(out_dir),
                "--modadd_p=7",
                "--modadd_m=21",
                "--device=cpu",
                "--dtype=float32",
                "--compile=False",
                "--n_layer=1",
                "--n_head=1",
                "--n_embd=16",
                "--block_size=8",
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
                "--modadd_eval_metrics=True",
                "--modadd_eval_clean_train_loss=True",
                "--s5_eval_n=2",
                "--s5_eval_batch_size=2",
                "--wandb_log=False",
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            checkpoint = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["config"]["modadd_p"], 3)
            self.assertEqual(checkpoint["config"]["modadd_m"], 4)
            self.assertEqual(checkpoint["config"]["resolved_modadd_p"], 3)
            self.assertEqual(checkpoint["config"]["resolved_modadd_m"], 4)
            self.assertTrue((out_dir / "last_eval.json").exists())
            self.assertTrue((out_dir / "eval_history.jsonl").exists())
            self.assertTrue((out_dir / "completed.txt").exists())
            last_eval = json.loads((out_dir / "last_eval.json").read_text(encoding="utf-8"))
            self.assertIn("val/clean_full_exact", last_eval)
            self.assertIn("val/clean_final_exact", last_eval)
            self.assertGreaterEqual(len((out_dir / "eval_history.jsonl").read_text(encoding="utf-8").splitlines()), 1)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
            shutil.rmtree(dataset_dir, ignore_errors=True)
            shutil.rmtree(prompt_bank_dir, ignore_errors=True)

    def test_train_opd_modadd_writes_task_metadata_and_eval_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_bank_dir = root / "prompt_bank"
            teacher_dir = root / "teacher"
            out_dir = root / "opd_out"

            _write_prompt_bank(prompt_bank_dir, p=3, m=4, n_train=4, n_val=2, seed=11)
            _write_teacher_checkpoint(teacher_dir, vocab_size=4, block_size=8)

            cmd = [
                sys.executable,
                "train_opd.py",
                "--task=modadd",
                "--teacher_checkpoint=" + str(teacher_dir),
                "--prompt_bank_dir=" + str(prompt_bank_dir),
                "--subset_size=4",
                "--eta=0.1",
                "--teacher_law=distributional_noise",
                "--objective=forward_kl_full",
                "--out_dir=" + str(out_dir),
                "--batch_size=2",
                "--max_iters=1",
                "--learning_rate=0.001",
                "--warmup_iters=0",
                "--eval_interval=1",
                "--eval_n=2",
                "--eval_batch_size=2",
                "--log_interval=1",
                "--device=cpu",
                "--dtype=float32",
                "--seed=7",
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            run_meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(run_meta["task"], "modadd")
            self.assertEqual(run_meta["p"], 3)
            self.assertEqual(run_meta["m"], 4)
            self.assertEqual(run_meta["prompt_len"], 5)
            self.assertEqual(run_meta["cot_len"], 4)
            self.assertEqual(run_meta["final_answer_len"], 1)
            self.assertTrue((out_dir / "subset_indices.pt").exists())
            self.assertTrue((out_dir / "last_eval.json").exists())
            self.assertTrue((out_dir / "eval_history.jsonl").exists())
            self.assertTrue((out_dir / "completed.txt").exists())
            last_eval = json.loads((out_dir / "last_eval.json").read_text(encoding="utf-8"))
            self.assertIn("val/clean_full_exact", last_eval)
            self.assertIn("val/clean_final_exact", last_eval)

    def test_train_opd_modadd_reverse_kl_full_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_bank_dir = root / "prompt_bank"
            teacher_dir = root / "teacher"
            out_dir = root / "opd_reverse_full_out"

            _write_prompt_bank(prompt_bank_dir, p=3, m=4, n_train=4, n_val=2, seed=13)
            _write_teacher_checkpoint(teacher_dir, vocab_size=4, block_size=8)

            cmd = [
                sys.executable,
                "train_opd.py",
                "--task=modadd",
                "--teacher_checkpoint=" + str(teacher_dir),
                "--prompt_bank_dir=" + str(prompt_bank_dir),
                "--subset_size=4",
                "--eta=0.1",
                "--teacher_law=distributional_noise",
                "--objective=reverse_kl_full",
                "--out_dir=" + str(out_dir),
                "--batch_size=2",
                "--max_iters=1",
                "--learning_rate=0.001",
                "--warmup_iters=0",
                "--eval_interval=1",
                "--eval_n=2",
                "--eval_batch_size=2",
                "--log_interval=1",
                "--device=cpu",
                "--dtype=float32",
                "--seed=7",
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            run_meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(run_meta["task"], "modadd")
            self.assertEqual(run_meta["objective"], "reverse_kl_full")
            self.assertTrue((out_dir / "subset_indices.pt").exists())
            self.assertTrue((out_dir / "last_eval.json").exists())
            self.assertTrue((out_dir / "eval_history.jsonl").exists())
            self.assertTrue((out_dir / "completed.txt").exists())
            self.assertTrue((out_dir / "ckpt.pt").exists())
            last_eval = json.loads((out_dir / "last_eval.json").read_text(encoding="utf-8"))
            self.assertIn("val/clean_full_exact", last_eval)
            self.assertIn("val/clean_final_exact", last_eval)

    def test_train_opd_modadd_forward_kl_full_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_bank_dir = root / "prompt_bank"
            teacher_dir = root / "teacher"
            resumed_out_dir = root / "forward_resume"
            continuous_out_dir = root / "forward_continuous"

            _write_prompt_bank(prompt_bank_dir, p=3, m=4, n_train=8, n_val=2, seed=19)
            _write_teacher_checkpoint(teacher_dir, vocab_size=4, block_size=8)

            _run_train_opd(
                prompt_bank_dir=prompt_bank_dir,
                teacher_dir=teacher_dir,
                out_dir=resumed_out_dir,
                objective="forward_kl_full",
                max_iters=2,
                seed=17,
            )
            _run_train_opd(
                prompt_bank_dir=prompt_bank_dir,
                teacher_dir=teacher_dir,
                out_dir=resumed_out_dir,
                objective="forward_kl_full",
                max_iters=4,
                init_from="resume",
                seed=17,
            )
            _run_train_opd(
                prompt_bank_dir=prompt_bank_dir,
                teacher_dir=teacher_dir,
                out_dir=continuous_out_dir,
                objective="forward_kl_full",
                max_iters=4,
                seed=17,
            )

            resumed_ckpt = torch.load(resumed_out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
            continuous_ckpt = torch.load(continuous_out_dir / "ckpt.pt", map_location="cpu", weights_only=False)

            _assert_state_dicts_equal(self, resumed_ckpt["model"], continuous_ckpt["model"])
            _assert_state_dicts_equal(self, resumed_ckpt["prompt_cycle_state"], continuous_ckpt["prompt_cycle_state"])
            self.assertEqual(resumed_ckpt["iter_num"], continuous_ckpt["iter_num"])
            self.assertEqual(_read_json(resumed_out_dir / "last_eval.json"), _read_json(continuous_out_dir / "last_eval.json"))
            self.assertEqual((resumed_out_dir / "completed.txt").read_text(encoding="utf-8"), "iter_num=4\n")
            self.assertEqual((continuous_out_dir / "completed.txt").read_text(encoding="utf-8"), "iter_num=4\n")
            self.assertTrue((continuous_out_dir / "ckpt_0000002.pt").exists())

    def test_train_opd_modadd_reverse_kl_full_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_bank_dir = root / "prompt_bank"
            teacher_dir = root / "teacher"
            resumed_out_dir = root / "reverse_resume"
            continuous_out_dir = root / "reverse_continuous"

            _write_prompt_bank(prompt_bank_dir, p=3, m=4, n_train=8, n_val=2, seed=23)
            _write_teacher_checkpoint(teacher_dir, vocab_size=4, block_size=8)

            _run_train_opd(
                prompt_bank_dir=prompt_bank_dir,
                teacher_dir=teacher_dir,
                out_dir=resumed_out_dir,
                objective="reverse_kl_full",
                max_iters=2,
                seed=29,
            )
            _run_train_opd(
                prompt_bank_dir=prompt_bank_dir,
                teacher_dir=teacher_dir,
                out_dir=resumed_out_dir,
                objective="reverse_kl_full",
                max_iters=4,
                init_from="resume",
                seed=29,
            )
            _run_train_opd(
                prompt_bank_dir=prompt_bank_dir,
                teacher_dir=teacher_dir,
                out_dir=continuous_out_dir,
                objective="reverse_kl_full",
                max_iters=4,
                seed=29,
            )

            resumed_ckpt = torch.load(resumed_out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
            continuous_ckpt = torch.load(continuous_out_dir / "ckpt.pt", map_location="cpu", weights_only=False)

            _assert_state_dicts_equal(self, resumed_ckpt["model"], continuous_ckpt["model"])
            _assert_state_dicts_equal(self, resumed_ckpt["prompt_cycle_state"], continuous_ckpt["prompt_cycle_state"])
            self.assertEqual(resumed_ckpt["iter_num"], continuous_ckpt["iter_num"])
            self.assertEqual(_read_json(resumed_out_dir / "last_eval.json"), _read_json(continuous_out_dir / "last_eval.json"))
            self.assertEqual((resumed_out_dir / "completed.txt").read_text(encoding="utf-8"), "iter_num=4\n")
            self.assertEqual((continuous_out_dir / "completed.txt").read_text(encoding="utf-8"), "iter_num=4\n")
            self.assertTrue((continuous_out_dir / "ckpt_0000002.pt").exists())


if __name__ == "__main__":
    unittest.main()
