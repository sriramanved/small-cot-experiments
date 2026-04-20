from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator


OFFLINE_SAMPLE_RE = re.compile(
    r"^out-modadd-noisy-bc-sample-then-corrupt-"
    r"p(?P<p>\d+)-m(?P<m>\d+)-n(?P<subset_size>\d+)-eta(?P<eta>[^-]+)"
    r"(?:-(?P<run_tag>.+))?$"
)

OPD_RE = re.compile(
    r"^out-modadd-opd-(?P<objective>.+?)-"
    r"p(?P<p>\d+)-m(?P<m>\d+)-n(?P<subset_size>\d+)-eta(?P<eta>[^-]+)-"
    r"(?P<teacher_law>[^-]+)-(?P<temp_tag>[^-]+)"
    r"(?:-(?P<run_tag>.+))?$"
)

OBJECTIVE_TO_METHOD = {
    "forward_kl_simple": "NAIL-forward",
    "reverse_kl_simple": "NAIL-reverse",
    "reverse_kl_tm": "OPD",
}

METHOD_COLORS = {
    "LogLossBC": "#355070",
    "NAIL-forward": "#E76F51",
    "NAIL-reverse": "#F4A261",
    "OPD": "#2A9D8F",
}

METHOD_LINESTYLES = {
    "LogLossBC": "--",
    "NAIL-forward": "-",
    "NAIL-reverse": "-.",
    "OPD": ":",
}

METHOD_MARKERS = {
    "LogLossBC": "o",
    "NAIL-forward": "s",
    "NAIL-reverse": "D",
    "OPD": "^",
}


@dataclass
class RunRecord:
    run_id: str
    method: str
    objective: str
    teacher_law: str
    p: int
    m: int
    subset_size: int
    eta: float
    run_tag: str
    temp_tag: str
    out_dir: Path
    eval_history_path: Path
    last_eval_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize legacy pre-Hydra modular-addition runs from old out-dir naming "
            "conventions (offline BC sample_then_corrupt, forward_kl_simple, reverse_kl_simple)."
        )
    )
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--subset_size", type=int, required=True)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory for plots and CSV summary. Defaults to analysis/figures/modadd_legacy_<...>.",
    )
    parser.add_argument("--show", action="store_true", help="Also show figures interactively.")
    return parser.parse_args()


def parse_eta_tag(tag: str) -> float:
    return float(tag.replace("p", "."))


def eta_tag(eta: float) -> str:
    text = f"{eta:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Legacy eval logs can contain both old-style aliases and newer slash-style keys.
    # After renaming, that can create duplicate column names like val/clean_full_exact.
    if not df.columns.duplicated().any():
        return df

    ordered_cols: list[str] = []
    for col in df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)

    merged = pd.DataFrame(index=df.index)
    for col in ordered_cols:
        cols = df.loc[:, df.columns == col]
        if isinstance(cols, pd.Series):
            merged[col] = cols
            continue
        merged[col] = cols.bfill(axis=1).iloc[:, 0]
    return merged


def normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "val_clean_full_exact": "val/clean_full_exact",
        "val_clean_final_exact": "val/clean_final_exact",
        "val_loss": "val/loss",
        "iter_num": "iter",
    }
    work = coalesce_duplicate_columns(df.rename(columns=rename_map).copy())
    expected = ["iter", "val/clean_full_exact", "val/clean_final_exact"]
    missing = [col for col in expected if col not in work.columns]
    if missing:
        raise ValueError(f"Missing required history columns: {missing}")
    work["iter"] = pd.to_numeric(work["iter"], errors="coerce")
    for metric in ("val/clean_full_exact", "val/clean_final_exact", "val/loss"):
        if metric in work.columns:
            work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=["iter"]).sort_values("iter").reset_index(drop=True)
    return work


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_runs(root: Path, *, p: int, m: int, subset_size: int, run_tag: str) -> list[RunRecord]:
    records: list[RunRecord] = []
    for out_dir in sorted(root.glob("out-modadd-*")):
        if not out_dir.is_dir():
            continue
        eval_history_path = out_dir / "eval_history.jsonl"
        if not eval_history_path.exists():
            continue

        name = out_dir.name
        offline_match = OFFLINE_SAMPLE_RE.match(name)
        if offline_match:
            run_p = int(offline_match.group("p"))
            run_m = int(offline_match.group("m"))
            run_n = int(offline_match.group("subset_size"))
            found_tag = offline_match.group("run_tag") or ""
            if (run_p, run_m, run_n) != (p, m, subset_size):
                continue
            if run_tag and found_tag != run_tag:
                continue
            records.append(
                RunRecord(
                    run_id=name,
                    method="LogLossBC",
                    objective="sample_then_corrupt",
                    teacher_law="distributional_noise",
                    p=run_p,
                    m=run_m,
                    subset_size=run_n,
                    eta=parse_eta_tag(offline_match.group("eta")),
                    run_tag=found_tag,
                    temp_tag="",
                    out_dir=out_dir,
                    eval_history_path=eval_history_path,
                    last_eval_path=(out_dir / "last_eval.json") if (out_dir / "last_eval.json").exists() else None,
                )
            )
            continue

        opd_match = OPD_RE.match(name)
        if opd_match:
            objective = opd_match.group("objective")
            method = OBJECTIVE_TO_METHOD.get(objective)
            if method is None:
                continue
            run_p = int(opd_match.group("p"))
            run_m = int(opd_match.group("m"))
            run_n = int(opd_match.group("subset_size"))
            found_tag = opd_match.group("run_tag") or ""
            if (run_p, run_m, run_n) != (p, m, subset_size):
                continue
            if run_tag and found_tag != run_tag:
                continue
            records.append(
                RunRecord(
                    run_id=name,
                    method=method,
                    objective=objective,
                    teacher_law=opd_match.group("teacher_law"),
                    p=run_p,
                    m=run_m,
                    subset_size=run_n,
                    eta=parse_eta_tag(opd_match.group("eta")),
                    run_tag=found_tag,
                    temp_tag=opd_match.group("temp_tag"),
                    out_dir=out_dir,
                    eval_history_path=eval_history_path,
                    last_eval_path=(out_dir / "last_eval.json") if (out_dir / "last_eval.json").exists() else None,
                )
            )
    return records


def build_runs_df(records: list[RunRecord]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    run_data: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for record in records:
        history_df = normalize_history_df(pd.read_json(record.eval_history_path, lines=True))
        history_df["run_id"] = record.run_id
        history_df["method"] = record.method
        history_df["eta"] = record.eta
        run_data[record.run_id] = history_df

        if record.last_eval_path is not None:
            last_eval = load_json(record.last_eval_path)
            final_full = last_eval.get("val/clean_full_exact", last_eval.get("val_clean_full_exact"))
            final_final = last_eval.get("val/clean_final_exact", last_eval.get("val_clean_final_exact"))
            final_iter = last_eval.get("iter", last_eval.get("iter_num", history_df["iter"].iloc[-1]))
        else:
            final_row = history_df.iloc[-1]
            final_full = float(final_row["val/clean_full_exact"])
            final_final = float(final_row["val/clean_final_exact"])
            final_iter = int(final_row["iter"])

        rows.append(
            {
                "run_id": record.run_id,
                "method": record.method,
                "objective": record.objective,
                "teacher_law": record.teacher_law,
                "p": record.p,
                "m": record.m,
                "subset_size": record.subset_size,
                "eta": record.eta,
                "run_tag": record.run_tag,
                "temp_tag": record.temp_tag,
                "final_iter": final_iter,
                "final_clean_full_exact": final_full,
                "final_clean_final_exact": final_final,
                "out_dir": str(record.out_dir),
                "eval_history_path": str(record.eval_history_path),
            }
        )

    runs_df = pd.DataFrame(rows).sort_values(["method", "eta"]).reset_index(drop=True)
    return runs_df, run_data


def plot_summary_vs_eta(runs_df: pd.DataFrame, *, metric: str, out_path: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    for method in ("LogLossBC", "NAIL-forward", "NAIL-reverse", "OPD"):
        method_df = runs_df[runs_df["method"] == method].sort_values("eta")
        if method_df.empty:
            continue
        ax.plot(
            method_df["eta"],
            method_df[metric],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            linewidth=2.3,
            markersize=6,
            label=method,
        )
    metric_name = metric.replace("final_", "")
    ax.set_title(f"ModAdd p={int(runs_df['p'].iloc[0])}, m={int(runs_df['m'].iloc[0])}: {metric_name} vs eta")
    ax.set_xlabel("eta")
    ax.set_ylabel(metric_name)
    ax.set_ylim(0.0, 1.01)
    ax.grid(alpha=0.35)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_per_eta(run_data: dict[str, pd.DataFrame], runs_df: pd.DataFrame, *, metric: str, out_dir: Path, show: bool) -> None:
    etas = sorted(runs_df["eta"].unique())
    for eta in etas:
        fig, ax = plt.subplots(figsize=(12.5, 5.0), constrained_layout=True)
        method_rows = runs_df[runs_df["eta"] == eta].sort_values("method")
        for _, row in method_rows.iterrows():
            df = run_data[row["run_id"]].sort_values("iter")
            ax.plot(
                df["iter"],
                df[metric],
                color=METHOD_COLORS[row["method"]],
                linestyle=METHOD_LINESTYLES[row["method"]],
                marker=METHOD_MARKERS[row["method"]],
                linewidth=2.2,
                markersize=4,
                label=row["method"],
            )
        metric_name = metric.replace("val/", "")
        ax.set_title(
            f"ModAdd p={int(runs_df['p'].iloc[0])}, m={int(runs_df['m'].iloc[0])}, eta={eta:.2f}: {metric_name}"
        )
        ax.set_xlabel("iter")
        ax.set_ylabel(metric_name)
        ax.set_ylim(0.0, 1.01)
        ax.xaxis.set_major_locator(MultipleLocator(10000))
        ax.xaxis.set_minor_locator(MultipleLocator(5000))
        ax.grid(which="minor", linestyle=":", alpha=0.25)
        ax.grid(which="major", alpha=0.35)
        ax.legend(loc="best")
        out_path = out_dir / f"eta{eta_tag(eta)}_{metric_name}.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        print(f"Saved {out_path}")
        if show:
            plt.show()
        plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else root / "analysis" / "figures" / f"modadd_legacy_p{args.p}_m{args.m}_n{args.subset_size}_{args.run_tag or 'all'}"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = discover_runs(
        root,
        p=args.p,
        m=args.m,
        subset_size=args.subset_size,
        run_tag=args.run_tag,
    )
    if not records:
        raise SystemExit(
            "No matching legacy runs found. "
            "Check --root, --p, --m, --subset_size, and --run_tag against your old out-dir names."
        )

    runs_df, run_data = build_runs_df(records)
    csv_path = out_dir / "discovered_runs.csv"
    runs_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    print(runs_df[[
        "method", "eta", "objective", "teacher_law",
        "final_clean_full_exact", "final_clean_final_exact", "out_dir"
    ]].to_string(index=False))

    plot_summary_vs_eta(
        runs_df,
        metric="final_clean_full_exact",
        out_path=out_dir / "summary_clean_full_exact_vs_eta.png",
        show=args.show,
    )
    plot_summary_vs_eta(
        runs_df,
        metric="final_clean_final_exact",
        out_path=out_dir / "summary_clean_final_exact_vs_eta.png",
        show=args.show,
    )
    plot_per_eta(
        run_data,
        runs_df,
        metric="val/clean_full_exact",
        out_dir=out_dir,
        show=args.show,
    )
    plot_per_eta(
        run_data,
        runs_df,
        metric="val/clean_final_exact",
        out_dir=out_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
