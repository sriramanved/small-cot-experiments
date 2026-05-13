from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate modular-addition LogLossBC and paper-method results into CSV and Markdown tables."
    )
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--output_csv", type=str, default="modadd_results.csv")
    parser.add_argument("--output_md", type=str, default="modadd_results.md")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_filter(row: dict[str, object], *, p: int | None, m: int | None) -> bool:
    if p is not None and int(row.get("p", -1)) != p:
        return False
    if m is not None and int(row.get("m", -1)) != m:
        return False
    return True


def collect_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for out_dir in sorted(root.glob("out-modadd-*")):
        if not out_dir.is_dir():
            continue
        eval_path = out_dir / "last_eval.json"
        if not eval_path.exists():
            continue
        eval_summary = load_json(eval_path)
        row = {
            "out_dir": str(out_dir),
            "iter": eval_summary.get("iter"),
            "val_loss": eval_summary.get("val/loss"),
            "val_clean_full_exact": eval_summary.get("val/clean_full_exact"),
            "val_clean_final_exact": eval_summary.get("val/clean_final_exact"),
            "rollout_mode": "",
        }

        run_meta_path = out_dir / "run_meta.json"
        if run_meta_path.exists():
            run_meta = load_json(run_meta_path)
            row.update(
                {
                    "method": f"opd_{run_meta['objective']}",
                    "task": run_meta["task"],
                    "p": run_meta["p"],
                    "m": run_meta["m"],
                    "subset_size": run_meta["subset_size"],
                    "eta": run_meta["eta"],
                    "objective": run_meta["objective"],
                    "teacher_law": run_meta["teacher_law"],
                    "rollout_mode": "",
                }
            )
            rows.append(row)
            continue

        ckpt_path = out_dir / "ckpt.pt"
        if not ckpt_path.exists():
            continue
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        dataset = str(config.get("dataset", ""))
        dataset_meta = {}
        dataset_meta_path = root / "data" / dataset / "meta.json"
        if dataset_meta_path.exists():
            dataset_meta = load_json(dataset_meta_path)

        rollout_mode = dataset_meta.get("train_decode_mode", "")
        row.update(
            {
                "method": "offline_bc",
                "task": "modadd",
                "p": config.get("modadd_p", dataset_meta.get("p")),
                "m": config.get("modadd_m", dataset_meta.get("m")),
                "subset_size": config.get("offline_train_subset_size", dataset_meta.get("subset_size")),
                "eta": dataset_meta.get("eta", 0.0 if dataset.startswith("modadd_clean_offline") else None),
                "objective": "",
                "teacher_law": "",
                "rollout_mode": rollout_mode,
            }
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "method",
        "task",
        "p",
        "m",
        "subset_size",
        "eta",
        "objective",
        "teacher_law",
        "rollout_mode",
        "val_loss",
        "val_clean_full_exact",
        "val_clean_final_exact",
        "iter",
        "out_dir",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "method",
        "p",
        "m",
        "subset_size",
        "eta",
        "rollout_mode",
        "teacher_law",
        "val_clean_full_exact",
        "val_clean_final_exact",
        "out_dir",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [row for row in collect_rows(Path(args.root)) if maybe_filter(row, p=args.p, m=args.m)]
    rows.sort(key=lambda row: (row.get("method", ""), float(row.get("eta", 0.0)), int(row.get("subset_size", 0))))
    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_md), rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv} and {args.output_md}")


if __name__ == "__main__":
    main()
