from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.s5_cot.prompt_bank import load_prompt_bank, select_train_subset
from data.s5_cot.task import (
    DIGIT_END_ID,
    DIGIT_START_ID,
    estimate_saved_clean_train_loss,
    evaluate_saved_clean_s5_metrics,
)
from nanogpt_checkpoint import load_nanogpt_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose an S5 noisy offline BC dataset/checkpoint by validating "
            "prompt-bank alignment and comparing noisy train targets against clean oracle references."
        )
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--prompt_bank_dir", type=str, required=True)
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--train_decode_mode", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_eval", type=int, default=512)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_meta(dataset_dir: Path) -> dict[str, object]:
    with open(dataset_dir / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def bool_line(name: str, ok: bool) -> str:
    return f"{name}: {'OK' if ok else 'MISMATCH'}"


def summarize_dataset(
    dataset_dir: Path,
    prompt_bank_dir: Path,
    *,
    expected_subset_size: int | None,
    expected_eta: float | None,
    expected_teacher_checkpoint: str | None,
    expected_train_decode_mode: str | None,
) -> dict[str, object]:
    meta = load_meta(dataset_dir)
    prompt_bank = load_prompt_bank(prompt_bank_dir)

    subset_size = int(meta["subset_size"])
    subset_idx_saved = torch.load(dataset_dir / "subset_indices.pt", map_location="cpu").long()
    subset_idx_expected = select_train_subset(prompt_bank, subset_size)

    clean_train_prompt_saved = torch.load(dataset_dir / "clean_train_prompt_ids.pt", map_location="cpu")
    clean_train_cot_saved = torch.load(dataset_dir / "clean_train_cot_ids.pt", map_location="cpu").long()
    clean_val_prompt_saved = torch.load(dataset_dir / "clean_val_prompt_ids.pt", map_location="cpu")
    clean_val_cot_saved = torch.load(dataset_dir / "clean_val_cot_ids.pt", map_location="cpu").long()
    train_y = torch.load(dataset_dir / "train_y.pt", map_location="cpu").long()

    prompt_len = clean_train_prompt_saved.size(1)
    saved_targets = train_y[:, prompt_len - 1:]
    oracle_targets = clean_train_cot_saved

    token_mask = torch.ones_like(saved_targets, dtype=torch.bool)
    digit_mask = (oracle_targets >= DIGIT_START_ID) & (oracle_targets <= DIGIT_END_ID)
    punctuation_mask = token_mask & (~digit_mask)

    token_match = saved_targets.eq(oracle_targets)
    full_exact = token_match.all(dim=1).float().mean().item()
    final_exact = token_match[:, -7:].all(dim=1).float().mean().item()
    token_mismatch_rate = (~token_match).float().mean().item()
    digit_mismatch_rate = (~token_match[digit_mask]).float().mean().item() if torch.any(digit_mask) else 0.0
    punctuation_mismatch_rate = (~token_match[punctuation_mask]).float().mean().item() if torch.any(punctuation_mask) else 0.0

    summary = {
        "meta": meta,
        "subset_idx_matches_prompt_bank": torch.equal(subset_idx_saved, subset_idx_expected),
        "train_prompts_match_prompt_bank": torch.equal(
            clean_train_prompt_saved,
            prompt_bank.clean_train_prompt_ids.index_select(0, subset_idx_expected),
        ),
        "train_clean_refs_match_prompt_bank": torch.equal(
            clean_train_cot_saved,
            prompt_bank.clean_train_cot_ids.index_select(0, subset_idx_expected).long(),
        ),
        "val_prompts_match_prompt_bank": torch.equal(clean_val_prompt_saved, prompt_bank.clean_val_prompt_ids),
        "val_cots_match_prompt_bank": torch.equal(clean_val_cot_saved, prompt_bank.clean_val_cot_ids.long()),
        "rendered_vs_clean_full_exact": full_exact,
        "rendered_vs_clean_final_exact": final_exact,
        "rendered_vs_clean_token_mismatch_rate": token_mismatch_rate,
        "rendered_vs_clean_digit_mismatch_rate": digit_mismatch_rate,
        "rendered_vs_clean_punctuation_mismatch_rate": punctuation_mismatch_rate,
    }

    if expected_subset_size is not None:
        summary["expected_subset_size_matches"] = int(expected_subset_size) == subset_size
    if expected_eta is not None:
        summary["expected_eta_matches"] = abs(float(meta["eta"]) - float(expected_eta)) < 1e-12
    if expected_teacher_checkpoint is not None:
        summary["expected_teacher_checkpoint_matches"] = (
            Path(str(meta["teacher_checkpoint"])) == Path(expected_teacher_checkpoint)
        )
    if expected_train_decode_mode is not None:
        summary["expected_train_decode_mode_matches"] = (
            str(meta["train_decode_mode"]) == expected_train_decode_mode
        )

    return summary


def summarize_checkpoint(
    checkpoint_path: str,
    dataset_dir: Path,
    *,
    device: str,
    eval_iters: int,
    batch_size: int,
    n_eval: int,
) -> dict[str, object]:
    model = load_nanogpt_model(
        checkpoint_path,
        map_location="cpu",
        device=device,
        eval_mode=True,
    )
    clean_val_metrics = evaluate_saved_clean_s5_metrics(
        model,
        device=device,
        data_dir=str(dataset_dir),
        n_eval=n_eval,
        batch_size=batch_size,
    )
    clean_train_loss = estimate_saved_clean_train_loss(
        model,
        device=device,
        data_dir=str(dataset_dir),
        eval_iters=eval_iters,
        batch_size=batch_size,
    )

    train_x = torch.load(dataset_dir / "train_x.pt", map_location="cpu")
    train_y = torch.load(dataset_dir / "train_y.pt", map_location="cpu").long()
    n = train_x.size(0)
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        idx = torch.randint(n, (batch_size,))
        x = train_x.index_select(0, idx).to(device=device, dtype=torch.long)
        y = train_y.index_select(0, idx).to(device=device, dtype=torch.long)
        _, loss = model(x, y)
        losses[k] = loss.item()

    return {
        "checkpoint_clean_train_oracle_loss": float(clean_train_loss),
        "checkpoint_noisy_train_loss": float(losses.mean().item()),
        **{f"checkpoint_{k}": v for k, v in clean_val_metrics.items()},
    }


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    prompt_bank_dir = Path(args.prompt_bank_dir)

    dataset_summary = summarize_dataset(
        dataset_dir,
        prompt_bank_dir,
        expected_subset_size=args.subset_size,
        expected_eta=args.eta,
        expected_teacher_checkpoint=args.teacher_checkpoint,
        expected_train_decode_mode=args.train_decode_mode,
    )

    print("Dataset checks:")
    print(bool_line("subset indices match prompt-bank prefix", dataset_summary["subset_idx_matches_prompt_bank"]))
    print(bool_line("train prompts match prompt bank", dataset_summary["train_prompts_match_prompt_bank"]))
    print(bool_line("train clean refs match prompt bank", dataset_summary["train_clean_refs_match_prompt_bank"]))
    print(bool_line("val prompts match prompt bank", dataset_summary["val_prompts_match_prompt_bank"]))
    print(bool_line("val cots match prompt bank", dataset_summary["val_cots_match_prompt_bank"]))
    if "expected_subset_size_matches" in dataset_summary:
        print(bool_line("meta subset_size matches expected", dataset_summary["expected_subset_size_matches"]))
    if "expected_eta_matches" in dataset_summary:
        print(bool_line("meta eta matches expected", dataset_summary["expected_eta_matches"]))
    if "expected_teacher_checkpoint_matches" in dataset_summary:
        print(bool_line("meta teacher checkpoint matches expected", dataset_summary["expected_teacher_checkpoint_matches"]))
    if "expected_train_decode_mode_matches" in dataset_summary:
        print(bool_line("meta train_decode_mode matches expected", dataset_summary["expected_train_decode_mode_matches"]))

    print("\nDataset meta:")
    print(json.dumps(dataset_summary["meta"], indent=2))

    print("\nNoisy train targets vs clean oracle:")
    print(
        json.dumps(
            {
                "rendered_vs_clean_full_exact": dataset_summary["rendered_vs_clean_full_exact"],
                "rendered_vs_clean_final_exact": dataset_summary["rendered_vs_clean_final_exact"],
                "rendered_vs_clean_token_mismatch_rate": dataset_summary["rendered_vs_clean_token_mismatch_rate"],
                "rendered_vs_clean_digit_mismatch_rate": dataset_summary["rendered_vs_clean_digit_mismatch_rate"],
                "rendered_vs_clean_punctuation_mismatch_rate": dataset_summary["rendered_vs_clean_punctuation_mismatch_rate"],
            },
            indent=2,
        )
    )

    if args.checkpoint is not None:
        print("\nCheckpoint metrics:")
        checkpoint_summary = summarize_checkpoint(
            args.checkpoint,
            dataset_dir,
            device=args.device,
            eval_iters=args.eval_iters,
            batch_size=args.batch_size,
            n_eval=args.n_eval,
        )
        print(json.dumps(checkpoint_summary, indent=2))

    if args.strict:
        required_checks = [
            dataset_summary["subset_idx_matches_prompt_bank"],
            dataset_summary["train_prompts_match_prompt_bank"],
            dataset_summary["train_clean_refs_match_prompt_bank"],
            dataset_summary["val_prompts_match_prompt_bank"],
            dataset_summary["val_cots_match_prompt_bank"],
        ]
        if "expected_subset_size_matches" in dataset_summary:
            required_checks.append(dataset_summary["expected_subset_size_matches"])
        if "expected_eta_matches" in dataset_summary:
            required_checks.append(dataset_summary["expected_eta_matches"])
        if "expected_teacher_checkpoint_matches" in dataset_summary:
            required_checks.append(dataset_summary["expected_teacher_checkpoint_matches"])
        if "expected_train_decode_mode_matches" in dataset_summary:
            required_checks.append(dataset_summary["expected_train_decode_mode_matches"])
        if not all(required_checks):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
