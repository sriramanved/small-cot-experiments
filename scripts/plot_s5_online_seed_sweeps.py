from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
for path in (ROOT, SRC_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from nanogpt.methods.student_prefix import normalize_student_prefix_method


STUDENT_PREFIX_RE = re.compile(
    r"^out-s5-(?P<method_family>opd|nail)-(?P<loss>forward|reverse)-(?P<teacher_signal>mc|full)-"
    r"m(?P<m>\d+)-n(?P<subset_size>\d+)-eta(?P<eta>[^-]+)-"
    r"(?P<teacher_law>[^-]+)(?P<temp_suffix>(?:-.*)?)\-seed(?P<seed>\d+)$"
)

OFFLINE_BC_RE = re.compile(
    r"^out-s5-noisy-bc-m(?P<m>\d+)-n(?P<subset_size>\d+)-eta(?P<eta>[^-]+)"
    r"(?P<suffixes>(?:-[^-]+)*)-seed(?P<seed>\d+)$"
)

LEGACY_OPD_RE = re.compile(
    r"^out-s5-opd-(?P<objective>.+?)-m(?P<m>\d+)-n(?P<subset_size>\d+)-eta(?P<eta>[^-]+)-"
    r"(?P<teacher_law>[^-]+)-(?P<temp_tag>[^-]+)-seed(?P<seed>\d+)$"
)

METHOD_COLORS = {
    "Offline BC": "#4A4E69",
    "NAIL-forward, greedy rollout": "#D1495B",
    "NAIL-reverse, greedy rollout": "#EDA43B",
    "NAIL-forward, sampled rollout": "#00798C",
    "TM OPD": "#5B8E7D",
}

SEED_LINESTYLES = {
    20260417: "-",
    20260418: "--",
    20260419: ":",
}

SEED_MARKERS = {
    20260417: "o",
    20260418: "s",
    20260419: "^",
}

SOURCE_PRIORITY = {
    "local": 0,
    "blocklab": 1,
    "dev_node": 1,
    "aics": 2,
    "cache": 3,
}


@dataclass
class RunRecord:
    run_id: str
    method: str
    teacher_law: str
    teacher_signal: str
    m: int
    subset_size: int
    eta: float
    seed: int
    source_root: Path
    source_kind: str
    out_dir: Path
    eval_history_path: Path
    last_eval_path: Path | None
    run_meta: dict[str, object] | None
    completed: bool


def parse_eta_tag(tag: str) -> float:
    return float(tag.replace("p", ".").replace("neg", "-"))


def eta_tag(eta: float) -> str:
    text = f"{eta:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "neg")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_completed_run(out_dir: Path, last_eval_path: Path | None) -> bool:
    if (out_dir / "completed.txt").exists():
        return True
    if last_eval_path is None:
        return False
    try:
        return str(load_json(last_eval_path).get("reason", "")) == "final"
    except (OSError, json.JSONDecodeError):
        return False


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    return work.dropna(subset=["iter"]).sort_values("iter").reset_index(drop=True)


def method_label_from_state(method_state: dict[str, object]) -> str | None:
    method_family = str(method_state["method_family"])
    loss = str(method_state["loss"])
    teacher_signal = str(method_state["teacher_signal"])
    rollout_temperature = float(method_state["resolved_rollout_temperature"])

    if teacher_signal != "mc":
        return None
    if method_family == "opd" and loss == "reverse":
        return "TM OPD"
    if method_family == "nail" and loss == "forward" and rollout_temperature == 0.0:
        return "NAIL-forward, greedy rollout"
    if method_family == "nail" and loss == "reverse" and rollout_temperature == 0.0:
        return "NAIL-reverse, greedy rollout"
    if method_family == "nail" and loss == "forward" and rollout_temperature > 0.0:
        return "NAIL-forward, sampled rollout"
    return None


def legacy_method_label(
    *,
    objective: str,
    run_meta: dict[str, object] | None,
) -> str | None:
    if run_meta is not None:
        method_state = normalize_student_prefix_method(run_meta)
        return method_label_from_state(method_state)

    if objective == "reverse_kl_tm":
        return "TM OPD"
    if objective.startswith("forward_kl_"):
        return "NAIL-forward, sampled rollout"
    return None


def discover_s5_online_runs(
    search_roots: list[Path],
    *,
    m: int,
    seeds: list[int],
    etas: list[float],
    subset_size: int | None = None,
    teacher_seed: int | None = None,
) -> list[RunRecord]:
    seed_set = {int(seed) for seed in seeds}
    eta_set = {float(eta) for eta in etas}
    seen_dirs: set[Path] = set()
    records: list[RunRecord] = []

    for root in search_roots:
        if not root.exists():
            continue
        root = root.resolve()
        source_kind = "local" if root == ROOT else root.name

        for out_dir in sorted(root.rglob("out-s5-*")):
            if not out_dir.is_dir():
                continue
            rel_parts = out_dir.relative_to(root).parts
            if source_kind == "local" and rel_parts[:2] in {("analysis", "imports"), ("analysis", "cache")}:
                continue
            if source_kind != "local" and rel_parts[:1] in {("analysis",), ("exports",)}:
                continue
            resolved = out_dir.resolve()
            if resolved in seen_dirs:
                continue

            eval_history_path = out_dir / "eval_history.jsonl"
            if not eval_history_path.exists():
                continue

            run_meta_path = out_dir / "run_meta.json"
            run_meta = load_json(run_meta_path) if run_meta_path.exists() else None

            name = out_dir.name
            student_prefix_match = STUDENT_PREFIX_RE.match(name)
            if student_prefix_match:
                run_m = int(student_prefix_match.group("m"))
                run_seed = int(student_prefix_match.group("seed"))
                run_eta = parse_eta_tag(student_prefix_match.group("eta"))
                run_n = int(student_prefix_match.group("subset_size"))
                teacher_law = student_prefix_match.group("teacher_law")
                method_state = normalize_student_prefix_method(
                    run_meta or {
                        "method_family": student_prefix_match.group("method_family"),
                        "loss": student_prefix_match.group("loss"),
                        "teacher_signal": student_prefix_match.group("teacher_signal"),
                    }
                )
                method = method_label_from_state(method_state)
                teacher_signal = str(method_state["teacher_signal"])
            else:
                offline_match = OFFLINE_BC_RE.match(name)
                if offline_match:
                    run_m = int(offline_match.group("m"))
                    run_seed = int(offline_match.group("seed"))
                    run_eta = parse_eta_tag(offline_match.group("eta"))
                    run_n = int(offline_match.group("subset_size"))
                    suffixes = set(filter(None, offline_match.group("suffixes").split("-")))
                    if "full" in suffixes or "full-dist" in offline_match.group("suffixes"):
                        continue
                    teacher_law = "sample_then_corrupt" if "sample" in suffixes else "greedy_then_corrupt"
                    method = "Offline BC"
                    teacher_signal = "offline"
                else:
                    legacy_match = LEGACY_OPD_RE.match(name)
                    if not legacy_match:
                        continue
                    run_m = int(legacy_match.group("m"))
                    run_seed = int(legacy_match.group("seed"))
                    run_eta = parse_eta_tag(legacy_match.group("eta"))
                    run_n = int(legacy_match.group("subset_size"))
                    teacher_law = legacy_match.group("teacher_law")
                    method = legacy_method_label(
                        objective=legacy_match.group("objective"),
                        run_meta=run_meta,
                    )
                    if run_meta is not None:
                        teacher_signal = str(normalize_student_prefix_method(run_meta)["teacher_signal"])
                    else:
                        teacher_signal = "mc"

            if method is None:
                continue
            if teacher_signal != "mc" and method != "Offline BC":
                continue
            if run_m != m or run_seed not in seed_set or run_eta not in eta_set:
                continue
            if subset_size is not None and run_n != subset_size:
                continue
            if run_meta is not None and teacher_seed is not None:
                teacher_checkpoint = str(run_meta.get("teacher_checkpoint", ""))
                if teacher_checkpoint and f"teacher{teacher_seed}" not in teacher_checkpoint:
                    continue

            records.append(
                RunRecord(
                    run_id=str(out_dir),
                    method=method,
                    teacher_law=teacher_law,
                    teacher_signal=teacher_signal,
                    m=run_m,
                    subset_size=run_n,
                    eta=run_eta,
                    seed=run_seed,
                    source_root=root,
                    source_kind=source_kind,
                    out_dir=out_dir,
                    eval_history_path=eval_history_path,
                    last_eval_path=(out_dir / "last_eval.json") if (out_dir / "last_eval.json").exists() else None,
                    run_meta=run_meta,
                    completed=is_completed_run(
                        out_dir,
                        (out_dir / "last_eval.json") if (out_dir / "last_eval.json").exists() else None,
                    ),
                )
            )
            seen_dirs.add(resolved)

    return records


def build_runs_df(records: list[RunRecord]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    run_data: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for record in records:
        history_df = normalize_history_df(pd.read_json(record.eval_history_path, lines=True))
        history_df["run_id"] = record.run_id
        history_df["method"] = record.method
        history_df["eta"] = record.eta
        history_df["seed"] = record.seed
        history_df["source_kind"] = record.source_kind
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
                "teacher_law": record.teacher_law,
                "teacher_signal": record.teacher_signal,
                "m": record.m,
                "subset_size": record.subset_size,
                "eta": record.eta,
                "seed": record.seed,
                "source_kind": record.source_kind,
                "source_root": str(record.source_root),
                "completed": record.completed,
                "final_iter": final_iter,
                "final_clean_full_exact": final_full,
                "final_clean_final_exact": final_final,
                "out_dir": str(record.out_dir),
                "eval_history_path": str(record.eval_history_path),
            }
        )

    runs_df = pd.DataFrame(rows)
    if runs_df.empty:
        return runs_df, run_data

    method_order = {method: index for index, method in enumerate(METHOD_COLORS)}
    runs_df["method_order"] = runs_df["method"].map(method_order)
    runs_df["source_order"] = runs_df["source_kind"].map(SOURCE_PRIORITY).fillna(99)
    runs_df = runs_df.sort_values(
        ["eta", "method_order", "seed", "completed", "source_order", "final_iter"],
        ascending=[True, True, True, False, True, False],
    ).reset_index(drop=True)
    return runs_df, run_data


def dedupe_preferred_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return runs_df
    ordered = runs_df.sort_values(
        ["eta", "method", "seed", "completed", "source_order", "final_iter"],
        ascending=[True, True, True, False, True, False],
    )
    preferred = ordered.drop_duplicates(subset=["eta", "method", "seed"], keep="first").copy()
    return preferred.sort_values(["eta", "method_order", "seed"]).reset_index(drop=True)


def expected_rows(seeds: list[int], etas: list[float]) -> list[dict[str, object]]:
    methods = list(METHOD_COLORS.keys())
    rows: list[dict[str, object]] = []
    for eta in etas:
        for seed in seeds:
            for method in methods:
                rows.append({"eta": eta, "seed": seed, "method": method})
    return rows


def coverage_table(
    runs_df: pd.DataFrame,
    run_data: dict[str, pd.DataFrame],
    *,
    seeds: list[int],
    etas: list[float],
) -> pd.DataFrame:
    expected = pd.DataFrame(expected_rows(seeds, etas))
    if runs_df.empty:
        expected["available"] = False
        expected["n_points"] = 0
        expected["min_iter"] = pd.NA
        expected["max_iter"] = pd.NA
        expected["completed"] = pd.NA
        expected["final_clean_full_exact"] = pd.NA
        expected["final_clean_final_exact"] = pd.NA
        expected["run_id"] = pd.NA
        expected["source_kind"] = pd.NA
        return expected

    summary = (
        runs_df.groupby(["eta", "seed", "method"], as_index=False)
        .agg(
            run_id=("run_id", "first"),
            source_kind=("source_kind", "first"),
            completed=("completed", "first"),
            final_iter=("final_iter", "first"),
            final_clean_full_exact=("final_clean_full_exact", "first"),
            final_clean_final_exact=("final_clean_final_exact", "first"),
        )
    )

    if run_data:
        curve_summary = (
            pd.concat(run_data.values(), ignore_index=True)
            .groupby(["eta", "seed", "method"], as_index=False)
            .agg(
                n_points=("iter", "size"),
                min_iter=("iter", "min"),
                max_iter=("iter", "max"),
            )
        )
    else:
        curve_summary = pd.DataFrame(
            columns=["eta", "seed", "method", "n_points", "min_iter", "max_iter"]
        )

    merged = expected.merge(summary, on=["eta", "seed", "method"], how="left")
    merged = merged.merge(curve_summary, on=["eta", "seed", "method"], how="left")
    merged["available"] = merged["run_id"].notna()
    merged["n_points"] = merged["n_points"].fillna(0).astype(int)
    return merged.sort_values(["eta", "seed", "method"]).reset_index(drop=True)


def default_output_dir(root: Path, *, m: int, teacher_seed: int) -> Path:
    return root / "analysis" / "figures" / f"s5_method_seed_sweeps_m{m}_teacher{teacher_seed}"


def plot_per_eta(
    run_data: dict[str, pd.DataFrame],
    runs_df: pd.DataFrame,
    *,
    metric: str,
    out_dir: Path | None = None,
    show: bool = True,
) -> None:
    if runs_df.empty:
        return

    preferred = dedupe_preferred_runs(runs_df)
    metric_name = metric.split("/")[-1]

    for eta in sorted(preferred["eta"].unique()):
        eta_rows = preferred[preferred["eta"] == eta]
        fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)

        for method in METHOD_COLORS:
            method_rows = eta_rows[eta_rows["method"] == method].copy()
            if method_rows.empty:
                continue

            method_curves: list[pd.DataFrame] = []
            for _, row in method_rows.iterrows():
                df = run_data[row["run_id"]].sort_values("iter").copy()
                df = df[["iter", metric]].rename(columns={metric: "metric"})
                df["seed"] = int(row["seed"])
                method_curves.append(df)

            if not method_curves:
                continue

            combined = pd.concat(method_curves, ignore_index=True)
            summary = (
                combined.groupby("iter", as_index=False)
                .agg(
                    mean=("metric", "mean"),
                    std=("metric", lambda s: s.std(ddof=0)),
                    n_seeds=("seed", "nunique"),
                )
                .sort_values("iter")
            )
            summary["std"] = summary["std"].fillna(0.0)

            color = METHOD_COLORS[method]
            label = f"{method} (n={int(method_rows['seed'].nunique())})"
            ax.plot(
                summary["iter"],
                summary["mean"],
                color=color,
                linewidth=2.6,
                label=label,
            )
            ax.errorbar(
                summary["iter"],
                summary["mean"],
                yerr=summary["std"],
                fmt="none",
                ecolor=color,
                elinewidth=1.2,
                alpha=0.35,
                capsize=2,
            )

        ax.set_title(
            f"S5 m={int(preferred['m'].iloc[0])}, eta={eta:.2f}: methods with seed error bars ({metric_name})"
        )
        ax.set_xlabel("iter")
        ax.set_ylabel(metric)
        ax.set_ylim(0.0, 1.01)
        ax.xaxis.set_major_locator(MultipleLocator(10000))
        ax.xaxis.set_minor_locator(MultipleLocator(5000))
        ax.grid(which="minor", linestyle=":", alpha=0.25)
        ax.grid(which="major", alpha=0.35)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"eta{eta_tag(float(eta))}_{metric_name}_online_seed_sweeps.png"
            fig.savefig(out_path, dpi=220, bbox_inches="tight")
            print(f"Saved {out_path}")

        if show:
            plt.show()
        plt.close(fig)
