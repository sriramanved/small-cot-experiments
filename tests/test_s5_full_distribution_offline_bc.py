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
from data.synthetic.offline_losses import offline_teacher_prob_loss_from_logits
from model import GPT, GPTConfig
from model import causal_lm_loss


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


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_eval_history(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_state_dicts_equal(testcase: unittest.TestCase, lhs: dict, rhs: dict) -> None:
    testcase.assertEqual(set(lhs), set(rhs))
    for key in lhs:
        left_value = lhs[key]
        right_value = rhs[key]
        if torch.is_tensor(left_value):
            torch.testing.assert_close(left_value, right_value)
        else:
            testcase.assertEqual(left_value, right_value)


def _run_train_py(
    *,
    dataset_name: str,
    out_dir: Path,
    offline_target_type: str,
    max_iters: int,
    init_from: str = "scratch",
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "train.py",
        "config/train_s5_noisy_bc.py",
        "--dataset=" + dataset_name,
        "--out_dir=" + str(out_dir),
        "--device=cpu",
        "--dtype=float32",
        "--compile=False",
        "--offline_target_type=" + offline_target_type,
        "--init_from=" + init_from,
        "--n_layer=1",
        "--n_head=1",
        "--n_embd=16",
        "--block_size=28",
        "--batch_size=2",
        "--gradient_accumulation_steps=1",
        "--learning_rate=0.001",
        "--warmup_iters=2",
        "--max_iters=" + str(max_iters),
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
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _run_train_opd_s5(
    *,
    prompt_bank_dir: Path,
    teacher_dir: Path,
    out_dir: Path,
    objective: str,
    max_iters: int,
    seed: int = 37,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "train_opd.py",
        "--task=s5",
        "--teacher_checkpoint=" + str(teacher_dir),
        "--prompt_bank_dir=" + str(prompt_bank_dir),
        "--subset_size=12",
        "--eta=0.2",
        "--teacher_law=distributional_noise",
        "--objective=" + objective,
        "--out_dir=" + str(out_dir),
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


class S5FullDistributionOfflineBCTests(unittest.TestCase):
    def test_train_py_online_s5_eval_only_omits_val_cot_exact(self):
        out_dir = Path(tempfile.mkdtemp(prefix="s5-online-out-"))

        try:
            cmd = [
                sys.executable,
                "train.py",
                "--dataset=s5_cot",
                "--out_dir=" + str(out_dir),
                "--s5_m=2",
                "--device=cpu",
                "--dtype=float32",
                "--compile=False",
                "--n_layer=1",
                "--n_head=1",
                "--n_embd=16",
                "--block_size=28",
                "--batch_size=2",
                "--gradient_accumulation_steps=1",
                "--learning_rate=0.001",
                "--max_iters=1",
                "--eval_interval=1",
                "--eval_iters=1",
                "--always_save_checkpoint=False",
                "--eval_only=True",
                "--s5_eval_metrics=True",
                "--s5_eval_n=2",
                "--s5_eval_batch_size=2",
                "--wandb_log=False",
            ]
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            last_eval = _read_json(out_dir / "last_eval.json")
            self.assertNotIn("val/cot_exact", last_eval)
            self.assertIn("val/clean_full_exact", last_eval)
            self.assertIn("val/clean_final_exact", last_eval)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

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
            _write_s5_prompt_bank(prompt_bank_dir, m=2, n_train=8, n_val=4, seed=17)
            _write_teacher_checkpoint(teacher_dir, vocab_size=VOCAB_SIZE, block_size=28)

            render_offline_dataset(
                teacher_checkpoint=str(teacher_dir),
                prompt_bank_dir=str(prompt_bank_dir),
                save_dir=str(dataset_dir),
                subset_size=8,
                eta=0.2,
                rollout_mode="sample_then_corrupt",
                target_mode="teacher_probs",
                gen_batch_size=2,
                device="cpu",
                dtype_name="float32",
                seed=5,
            )

            _run_train_py(
                dataset_name=dataset_name,
                out_dir=out_dir,
                offline_target_type="teacher_probs",
                max_iters=4,
            )

            checkpoint = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["config"]["offline_target_type"], "teacher_probs")
            self.assertTrue((out_dir / "last_eval.json").exists())
            self.assertTrue((out_dir / "eval_history.jsonl").exists())
            self.assertTrue((out_dir / "completed.txt").exists())
            last_eval = _read_json(out_dir / "last_eval.json")
            self.assertNotIn("val/cot_exact", last_eval)
            self.assertIn("val/clean_full_exact", last_eval)
            self.assertIn("val/clean_final_exact", last_eval)
            self.assertIn("train/clean_oracle_loss_eval", last_eval)
            history = _read_eval_history(out_dir / "eval_history.jsonl")
            self.assertEqual([entry["reason"] for entry in history], ["periodic", "periodic", "periodic", "final"])
            self.assertEqual([entry["iter"] for entry in history], [0, 2, 4, 4])
            self.assertEqual((out_dir / "completed.txt").read_text(encoding="utf-8"), "iter_num=4\n")
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
            shutil.rmtree(dataset_dir, ignore_errors=True)
            shutil.rmtree(prompt_bank_dir, ignore_errors=True)
            shutil.rmtree(teacher_dir, ignore_errors=True)

    def test_train_py_offline_s5_token_targets_runs_and_writes_eval_artifacts(self):
        dataset_name = "s5_noisy_offline_sample_then_corrupt_test_tokens"
        dataset_dir = DATA_ROOT / dataset_name
        prompt_bank_dir = DATA_ROOT / "test_s5_prompt_bank_for_token_train"
        teacher_dir = Path(tempfile.mkdtemp(prefix="s5-token-teacher-"))
        out_dir = Path(tempfile.mkdtemp(prefix="s5-token-out-"))

        try:
            _write_s5_prompt_bank(prompt_bank_dir, m=2, n_train=8, n_val=4, seed=23)
            _write_teacher_checkpoint(teacher_dir, vocab_size=VOCAB_SIZE, block_size=28)

            render_offline_dataset(
                teacher_checkpoint=str(teacher_dir),
                prompt_bank_dir=str(prompt_bank_dir),
                save_dir=str(dataset_dir),
                subset_size=8,
                eta=0.2,
                rollout_mode="sample_then_corrupt",
                target_mode="tokens",
                gen_batch_size=2,
                device="cpu",
                dtype_name="float32",
                seed=11,
            )

            _run_train_py(
                dataset_name=dataset_name,
                out_dir=out_dir,
                offline_target_type="tokens",
                max_iters=4,
            )

            checkpoint = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["config"]["offline_target_type"], "tokens")
            self.assertTrue((out_dir / "last_eval.json").exists())
            self.assertTrue((out_dir / "eval_history.jsonl").exists())
            self.assertTrue((out_dir / "completed.txt").exists())
            history = _read_eval_history(out_dir / "eval_history.jsonl")
            self.assertEqual([entry["reason"] for entry in history], ["periodic", "periodic", "periodic", "final"])
            self.assertEqual([entry["iter"] for entry in history], [0, 2, 4, 4])
            self.assertEqual((out_dir / "completed.txt").read_text(encoding="utf-8"), "iter_num=4\n")
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
            shutil.rmtree(dataset_dir, ignore_errors=True)
            shutil.rmtree(prompt_bank_dir, ignore_errors=True)
            shutil.rmtree(teacher_dir, ignore_errors=True)

    def test_teacher_prob_supervision_differs_from_token_target_cross_entropy(self):
        logits = torch.tensor(
            [[[1.2, 0.1, -0.5], [-0.2, 2.0, 0.3], [0.5, 0.1, -0.4], [2.1, -1.0, 0.2]]],
            dtype=torch.float32,
        )
        y = torch.tensor([[-1, -1, 1, 0]], dtype=torch.long)
        teacher_probs = torch.tensor(
            [[[0.51, 0.48, 0.01], [0.05, 0.90, 0.05]]],
            dtype=torch.float32,
        )

        student_suffix_logits, full_loss, _ = offline_teacher_prob_loss_from_logits(logits, y, teacher_probs)
        token_loss = causal_lm_loss(student_suffix_logits, y[:, -teacher_probs.size(1):], ignore_index=-1)

        self.assertGreater(abs(full_loss.item() - token_loss.item()), 0.05)

    def test_train_py_offline_s5_teacher_probs_resume_matches_continuous(self):
        dataset_name = "s5_noisy_offline_full_dist_sample_then_corrupt_resume_test"
        dataset_dir = DATA_ROOT / dataset_name
        prompt_bank_dir = DATA_ROOT / "test_s5_prompt_bank_for_full_dist_resume"
        teacher_dir = Path(tempfile.mkdtemp(prefix="s5-full-dist-resume-teacher-"))
        resumed_out_dir = Path(tempfile.mkdtemp(prefix="s5-full-dist-resume-out-"))
        continuous_out_dir = Path(tempfile.mkdtemp(prefix="s5-full-dist-cont-out-"))

        try:
            _write_s5_prompt_bank(prompt_bank_dir, m=2, n_train=8, n_val=4, seed=31)
            _write_teacher_checkpoint(teacher_dir, vocab_size=VOCAB_SIZE, block_size=28)

            render_offline_dataset(
                teacher_checkpoint=str(teacher_dir),
                prompt_bank_dir=str(prompt_bank_dir),
                save_dir=str(dataset_dir),
                subset_size=8,
                eta=0.2,
                rollout_mode="sample_then_corrupt",
                target_mode="teacher_probs",
                gen_batch_size=2,
                device="cpu",
                dtype_name="float32",
                seed=13,
            )

            _run_train_py(
                dataset_name=dataset_name,
                out_dir=resumed_out_dir,
                offline_target_type="teacher_probs",
                max_iters=2,
            )
            _run_train_py(
                dataset_name=dataset_name,
                out_dir=resumed_out_dir,
                offline_target_type="teacher_probs",
                max_iters=4,
                init_from="resume",
            )
            _run_train_py(
                dataset_name=dataset_name,
                out_dir=continuous_out_dir,
                offline_target_type="teacher_probs",
                max_iters=4,
            )

            resumed_ckpt = torch.load(resumed_out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
            continuous_ckpt = torch.load(continuous_out_dir / "ckpt.pt", map_location="cpu", weights_only=False)

            _assert_state_dicts_equal(self, resumed_ckpt["model"], continuous_ckpt["model"])
            _assert_state_dicts_equal(self, resumed_ckpt["offline_train_state"], continuous_ckpt["offline_train_state"])
            self.assertEqual(resumed_ckpt["iter_num"], continuous_ckpt["iter_num"])
            resumed_last_eval = _read_json(resumed_out_dir / "last_eval.json")
            continuous_last_eval = _read_json(continuous_out_dir / "last_eval.json")
            self.assertNotIn("val/cot_exact", resumed_last_eval)
            self.assertNotIn("val/cot_exact", continuous_last_eval)
            for key in ("iter", "reason", "val/clean_full_exact", "val/clean_final_exact"):
                self.assertEqual(resumed_last_eval[key], continuous_last_eval[key])
            self.assertTrue(torch.isfinite(torch.tensor(resumed_last_eval["val/loss"])).item())
            self.assertTrue(torch.isfinite(torch.tensor(continuous_last_eval["val/loss"])).item())
            self.assertTrue(torch.isfinite(torch.tensor(resumed_last_eval["train/clean_oracle_loss_eval"])).item())
            self.assertTrue(torch.isfinite(torch.tensor(continuous_last_eval["train/clean_oracle_loss_eval"])).item())
            self.assertEqual((resumed_out_dir / "completed.txt").read_text(encoding="utf-8"), "iter_num=4\n")
            self.assertEqual((continuous_out_dir / "completed.txt").read_text(encoding="utf-8"), "iter_num=4\n")
        finally:
            shutil.rmtree(resumed_out_dir, ignore_errors=True)
            shutil.rmtree(continuous_out_dir, ignore_errors=True)
            shutil.rmtree(dataset_dir, ignore_errors=True)
            shutil.rmtree(prompt_bank_dir, ignore_errors=True)
            shutil.rmtree(teacher_dir, ignore_errors=True)

    def test_train_opd_s5_reverse_kl_full_runs_multiple_evals_and_saves_consistent_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_bank_dir = root / "prompt_bank"
            teacher_dir = root / "teacher"
            out_dir = root / "opd_s5_reverse_full"

            _write_s5_prompt_bank(prompt_bank_dir, m=2, n_train=12, n_val=4, seed=41)
            _write_teacher_checkpoint(teacher_dir, vocab_size=VOCAB_SIZE, block_size=28)

            _run_train_opd_s5(
                prompt_bank_dir=prompt_bank_dir,
                teacher_dir=teacher_dir,
                out_dir=out_dir,
                objective="reverse_kl_full",
                max_iters=4,
                seed=43,
            )

            run_meta = _read_json(out_dir / "run_meta.json")
            self.assertEqual(run_meta["task"], "s5")
            self.assertEqual(run_meta["objective"], "reverse_kl_full")
            self.assertEqual(run_meta["teacher_law"], "distributional_noise")
            self.assertTrue((out_dir / "subset_indices.pt").exists())
            self.assertTrue((out_dir / "ckpt.pt").exists())
            self.assertTrue((out_dir / "ckpt_0000002.pt").exists())
            self.assertEqual((out_dir / "completed.txt").read_text(encoding="utf-8"), "iter_num=4\n")

            history = _read_eval_history(out_dir / "eval_history.jsonl")
            self.assertEqual([entry["reason"] for entry in history], ["periodic", "periodic", "final"])
            self.assertEqual([entry["iter"] for entry in history], [0, 2, 4])
            last_eval = _read_json(out_dir / "last_eval.json")
            self.assertEqual(last_eval["reason"], "final")
            self.assertNotIn("val/cot_exact", last_eval)


if __name__ == "__main__":
    unittest.main()
