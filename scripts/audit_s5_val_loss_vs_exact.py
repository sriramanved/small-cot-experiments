from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.s5_cot.validation_diagnostics import evaluate_s5_offline_validation_diagnostics
from nanogpt_checkpoint import load_nanogpt_model
from torch_dtypes import DTYPE_LOOKUP


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description=(
            "Audit S5 offline-BC validation loss versus clean exact accuracy. "
            "Prints confidence/calibration diagnostics and examples that are "
            "greedy-exact but still have high clean CE."
        )
    )
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory containing ckpt.pt.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to ckpt.pt or a checkpoint directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Offline dataset directory. Defaults to data/<checkpoint config dataset>.",
    )
    parser.add_argument("--eta", type=float, default=None, help="Optional expected eta sanity check.")
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--dtype", choices=sorted(DTYPE_LOOKUP), default=None)
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--loss-threshold", type=float, default=1.0)
    parser.add_argument("--json-out", type=str, default=None)
    return parser.parse_args()


def resolve_checkpoint_arg(args: argparse.Namespace) -> str:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.run_dir is None:
        raise ValueError("Provide --run-dir or --checkpoint.")
    return str(Path(args.run_dir) / "ckpt.pt")


def resolve_data_dir(args: argparse.Namespace, checkpoint: dict[str, object]) -> Path:
    if args.data_dir is not None:
        return Path(args.data_dir)
    config = checkpoint.get("config", {})
    if not isinstance(config, dict) or not config.get("dataset"):
        raise ValueError("Could not infer data dir from checkpoint config; pass --data-dir.")
    return ROOT / "data" / str(config["dataset"])


def load_meta(data_dir: Path) -> dict[str, object]:
    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def print_metrics(metrics: dict[str, float]) -> None:
    print("Diagnostics:")
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float) and math.isnan(value):
            rendered = "nan"
        else:
            rendered = f"{float(value):.6g}"
        print(f"  {key}: {rendered}")


def print_examples(examples: list[dict[str, object]]) -> None:
    if not examples:
        print("\nNo examples matched: clean_full_exact=1 and sequence_nll > threshold.")
        return

    print("\nHigh-loss exact examples:")
    for idx, example in enumerate(examples, start=1):
        print(
            f"\nExample {idx} row={example['row']} "
            f"sequence_nll={float(example['sequence_nll']):.6f} "
            f"full_exact={example['clean_full_exact']} "
            f"final_exact={example['clean_final_exact']}"
        )
        print(f"prompt: {example['prompt']}")
        print(f"clean target: {example['clean_target']}")
        print(f"greedy target: {example['greedy_generated_target']}")
        print("tokens:")
        print("  pos mark clean gen p_clean loss top1 top1_p top2 top2_p")
        for token in example["tokens"]:
            assert isinstance(token, dict)
            marks = []
            if token["region"] == "final":
                marks.append("final")
            if token["corruptible"]:
                marks.append("digit")
            mark = ",".join(marks) if marks else "cot"
            print(
                "  "
                f"{int(token['pos']):03d} "
                f"{mark:11s} "
                f"{token['clean_token']!s:>2s} "
                f"{token['greedy_token']!s:>2s} "
                f"{float(token['clean_prob']):.6f} "
                f"{float(token['loss']):.6f} "
                f"{token['top1_token']!s:>2s} "
                f"{float(token['top1_prob']):.6f} "
                f"{token['top2_token']!s:>2s} "
                f"{float(token['top2_prob']):.6f}"
            )


def main() -> None:
    args = parse_args()
    checkpoint_arg = resolve_checkpoint_arg(args)
    model, checkpoint = load_nanogpt_model(
        checkpoint_arg,
        map_location="cpu",
        device=args.device,
        eval_mode=True,
        return_checkpoint=True,
    )
    if args.dtype is not None:
        model = model.to(dtype=DTYPE_LOOKUP[args.dtype])

    data_dir = resolve_data_dir(args, checkpoint)
    meta = load_meta(data_dir)
    if args.eta is not None:
        meta_eta = float(meta.get("eta", float("nan")))
        if not math.isfinite(meta_eta) or abs(meta_eta - float(args.eta)) > 1e-12:
            print(f"warning: requested eta={args.eta} but dataset meta eta={meta.get('eta')}")

    metrics, examples = evaluate_s5_offline_validation_diagnostics(
        model,
        device=args.device,
        data_dir=data_dir,
        n_eval=args.n_eval,
        batch_size=args.batch_size,
        loss_threshold=args.loss_threshold,
        collect_examples=True,
        max_examples=args.num_examples,
    )

    print(f"checkpoint: {checkpoint_arg}")
    print(f"data_dir: {data_dir}")
    print(f"eta: {meta.get('eta')}")
    print(f"teacher_law: {meta.get('teacher_law', 'distributional_noise')}")
    print(f"target_span: {meta.get('target_span', 'cot_with_final_answer_suffix')}")
    print_metrics(metrics)
    print_examples(examples)

    if args.json_out is not None:
        payload = {
            "checkpoint": checkpoint_arg,
            "data_dir": str(data_dir),
            "meta": meta,
            "metrics": metrics,
            "examples": examples,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
