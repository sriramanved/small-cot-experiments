from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.s5_cot.offline_render import generate_teacher_targets
from data.s5_cot.task import (
    evaluate_saved_clean_s5_metrics,
    greedy_generate_target_ids_batched,
)
from hf_checkpoint import DTYPE_LOOKUP, load_nanogpt_checkpoint_as_hf
from nanogpt_checkpoint import load_nanogpt_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose clean offline S5 failures by comparing the original nanoGPT "
            "teacher, the HF-converted teacher, and the rendered offline dataset "
            "against oracle clean CoTs."
        )
    )
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--n_eval", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=sorted(DTYPE_LOOKUP), default="float16")
    return parser.parse_args()


def load_nanogpt_teacher(checkpoint_or_dir: str, device: str):
    return load_nanogpt_model(
        checkpoint_or_dir,
        map_location="cpu",
        device=device,
        eval_mode=True,
    )


def hf_clean_metrics(model, source_dir: Path, n_eval: int, batch_size: int, device: str):
    prompt_ids_all = torch.load(source_dir / "clean_val_prompt_ids.pt", map_location="cpu").long()[:n_eval]
    cot_ids_all = torch.load(source_dir / "clean_val_cot_ids.pt", map_location="cpu").long()[:n_eval]

    full_ok = 0
    final_ok = 0
    disagreements_with_oracle = 0
    n = prompt_ids_all.size(0)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        prompt_ids = prompt_ids_all[start:end]
        cot_ids = cot_ids_all[start:end]
        pred_ids = generate_teacher_targets(
            model,
            prompt_ids,
            target_len=cot_ids.size(1),
            eta=0.0,
            device=device,
        ).long()
        match_full = pred_ids.eq(cot_ids).all(dim=1)
        match_final = pred_ids[:, -7:].eq(cot_ids[:, -7:]).all(dim=1)
        full_ok += int(match_full.sum().item())
        final_ok += int(match_final.sum().item())
        disagreements_with_oracle += int((~match_full).sum().item())

    return {
        "clean_full_exact": full_ok / n,
        "clean_final_exact": final_ok / n,
        "oracle_disagreement_rate": disagreements_with_oracle / n,
    }


def compare_nanogpt_vs_hf(ng_model, hf_model, source_dir: Path, n_eval: int, batch_size: int, device: str):
    prompt_ids_all = torch.load(source_dir / "clean_val_prompt_ids.pt", map_location="cpu").long()[:n_eval]
    cot_len = torch.load(source_dir / "clean_val_cot_ids.pt", map_location="cpu").size(1)

    n = prompt_ids_all.size(0)
    same_full = 0
    same_final = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        prompt_ids = prompt_ids_all[start:end]
        ng_pred = greedy_generate_target_ids_batched(ng_model, prompt_ids, cot_len, device)
        hf_pred = generate_teacher_targets(
            hf_model,
            prompt_ids,
            target_len=cot_len,
            eta=0.0,
            device=device,
        ).long()
        same_full += int(ng_pred.eq(hf_pred).all(dim=1).sum().item())
        same_final += int(ng_pred[:, -7:].eq(hf_pred[:, -7:]).all(dim=1).sum().item())

    return {
        "full_rollout_agreement": same_full / n,
        "final_answer_agreement": same_final / n,
    }


def inspect_rendered_dataset(dataset_dir: Path, n_eval: int):
    meta = json.load(open(dataset_dir / "meta.json", "r", encoding="utf-8"))
    clean_train_prompt_ids = torch.load(dataset_dir / "clean_train_prompt_ids.pt", map_location="cpu")
    clean_train_cot_ids = torch.load(dataset_dir / "clean_train_cot_ids.pt", map_location="cpu").long()
    train_y = torch.load(dataset_dir / "train_y.pt", map_location="cpu").long()

    n = min(n_eval, clean_train_prompt_ids.size(0))
    prompt_len = clean_train_prompt_ids.size(1)
    saved_target_ids = train_y[:n, prompt_len - 1:]
    oracle_target_ids = clean_train_cot_ids[:n]

    full_ok = saved_target_ids.eq(oracle_target_ids).all(dim=1)
    final_ok = saved_target_ids[:, -7:].eq(oracle_target_ids[:, -7:]).all(dim=1)

    return {
        "meta_subset_size": int(meta["subset_size"]),
        "meta_eta": float(meta["eta"]),
        "rendered_vs_oracle_full_exact": float(full_ok.float().mean().item()),
        "rendered_vs_oracle_final_exact": float(final_ok.float().mean().item()),
    }


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir is not None else None
    torch_dtype = DTYPE_LOOKUP[args.dtype]

    print("Loading original nanoGPT teacher...")
    ng_model = load_nanogpt_teacher(args.teacher_checkpoint, args.device)

    print("Loading HF-converted teacher...")
    hf_model = load_nanogpt_checkpoint_as_hf(
        args.teacher_checkpoint,
        map_location="cpu",
        device=args.device,
        torch_dtype=torch_dtype,
        eval_mode=True,
    )

    print("\nOriginal nanoGPT teacher on clean val:")
    print(
        evaluate_saved_clean_s5_metrics(
            ng_model,
            device=args.device,
            data_dir=str(source_dir),
            n_eval=args.n_eval,
            batch_size=args.batch_size,
        )
    )

    print("\nHF teacher on same clean val:")
    print(hf_clean_metrics(hf_model, source_dir, args.n_eval, args.batch_size, args.device))

    print("\nOriginal nanoGPT vs HF greedy agreement on same prompts:")
    print(compare_nanogpt_vs_hf(ng_model, hf_model, source_dir, args.n_eval, args.batch_size, args.device))

    if dataset_dir is not None:
        print("\nRendered dataset targets vs oracle clean CoTs:")
        print(inspect_rendered_dataset(dataset_dir, args.n_eval))


if __name__ == "__main__":
    main()
