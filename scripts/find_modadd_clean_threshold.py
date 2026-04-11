from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the smallest modular-addition offline BC subset size whose "
            "last saved eval reaches perfect clean_full_exact and clean_final_exact."
        )
    )
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--subset_sizes", type=int, nargs="+", required=True)
    parser.add_argument("--out_prefix", type=str, default="out-modadd-clean-offline-bc")
    parser.add_argument("--save_path", type=str, default=None)
    return parser.parse_args()


def load_last_eval(path: Path) -> dict[str, float]:
    with open(path / "last_eval.json", "r", encoding="utf-8") as f:
        return json.load(f)


def is_perfect(metrics: dict[str, float]) -> bool:
    return metrics.get("val/clean_full_exact", 0.0) >= 1.0 and metrics.get("val/clean_final_exact", 0.0) >= 1.0


def main() -> None:
    args = parse_args()
    threshold = None
    inspected = []

    for subset_size in sorted(args.subset_sizes):
        out_dir = Path(f"{args.out_prefix}-p{args.p}-m{args.m}-n{subset_size}")
        eval_path = out_dir / "last_eval.json"
        if not eval_path.exists():
            raise FileNotFoundError(f"Missing {eval_path}; train that subset size first.")
        metrics = load_last_eval(out_dir)
        inspected.append(
            {
                "subset_size": subset_size,
                "clean_full_exact": metrics.get("val/clean_full_exact"),
                "clean_final_exact": metrics.get("val/clean_final_exact"),
                "out_dir": str(out_dir),
            }
        )
        if threshold is None and is_perfect(metrics):
            threshold = subset_size

    result = {
        "task": "modadd",
        "p": args.p,
        "m": args.m,
        "perfect_metric": "clean_full_exact == 1.0 and clean_final_exact == 1.0",
        "threshold_subset_size": threshold,
        "inspected": inspected,
    }
    save_path = Path(args.save_path or f"modadd_clean_threshold_p{args.p}_m{args.m}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if threshold is None:
        raise SystemExit(
            f"No perfect subset found for p={args.p}, m={args.m}. Saved inspection summary to {save_path}."
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
