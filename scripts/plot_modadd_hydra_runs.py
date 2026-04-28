from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator


NOISY_BC_RE = re.compile(
    r"^out-modadd-noisy-bc-"
    r"p(?P<p>\d+)-m(?P<m>\d+)-n(?P<subset_size>\d+)-eta(?P<eta>[^-]+)"
    r"(?:-(?P<rollout_mode>[^-]+))?-seed(?P<seed>\d+)$"
)

LEGACY_OPD_RE = re.compile(
    r"^out-modadd-opd-(?P<objective>.+?)-"
    r"p(?P<p>\d+)-m(?P<m>\d+)-n(?P<subset_size>\d+)-eta(?P<eta>[^-]+)-"
    r"(?P<teacher_law>[^-]+)-(?P<temp_tag>[^-]+)-seed(?P<seed>\d+)$"
)

STUDENT_PREFIX_RE = re.compile(
    r"^out-modadd-(?P<method_family>opd|nail)-(?P<loss>forward|reverse)-(?P<teacher_signal>mc|full)-"
    r"p(?P<p>\d+)-m(?P<m>\d+)-n(?P<subset_size>\d+)-eta(?P<eta>[^-]+)-"
    r"(?P<teacher_law>[^-]+)(?P<temp_suffix>(?:-.*)?)\-seed(?P<seed>\d+)$"
)

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
    rollout_mode: str
    p: int
    m: int
    subset_size: int
    eta: float
    seed: int
    temp_tag: str
    out_dir: Path
    eval_history_path: Path
    last_eval_path: Path | None
    run_meta: dict[str, object] | None
    selection_rank: int
    selection_reason: str


def parse_eta_tag(tag: str) -> float:
    return float(tag.replace("p", ".").replace("neg", "-"))


def eta_tag(eta: float) -> str:
    text = f"{eta:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "neg")


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


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def method_label(method_family: str, loss: str) -> str:
    if method_family == "opd":
        return "OPD"
    if loss == "forward":
        return "NAIL-forward"
    return "NAIL-reverse"


def unique_run_id(root: Path, out_dir: Path) -> str:
    try:
        return str(out_dir.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(out_dir.resolve())


def classify_run_selection(
    *,
    method: str,
    out_dir: Path,
    run_meta: dict[str, object] | None,
) -> tuple[int, str]:
    if method != "NAIL-reverse":
        return 0, "standard_method"

    reverse_action_source = None if run_meta is None else run_meta.get("reverse_action_source")
    if reverse_action_source == "student_aux_sample":
        return 0, "run_meta.reverse_action_source=student_aux_sample"
    if reverse_action_source == "rollout_action":
        return 40, "legacy_rollout_actions"

    path_text = str(out_dir.resolve())
    if "nail_reverse_mc_fixed" in path_text:
        return 1, "path_contains_nail_reverse_mc_fixed"

    if run_meta is not None and all(key in run_meta for key in ("target_len", "answer_len", "target_span")):
        return 2, "post_patch_run_meta_fields_present"
    return 30, "no_fixed_nail_reverse_marker"


def legacy_method_label(
    *,
    objective: str,
    temp_tag: str,
    run_meta: dict[str, object] | None,
) -> str | None:
    if run_meta is not None and "method_family" in run_meta and "loss" in run_meta:
        return method_label(str(run_meta["method_family"]), str(run_meta["loss"]))
    if objective.startswith("forward_kl_"):
        return "NAIL-forward"
    if objective == "reverse_kl_tm":
        return "OPD"
    rollout_temp = None
    if run_meta is not None:
        rollout_temp = run_meta.get("resolved_rollout_temperature")
        if rollout_temp is None:
            rollout_temp = run_meta.get("student_rollout_temperature", run_meta.get("student_temperature"))
    if rollout_temp is None and temp_tag == "greedy":
        rollout_temp = 0.0
    if rollout_temp is not None and float(rollout_temp) == 0.0:
        return "NAIL-reverse"
    if objective.startswith("reverse_kl_"):
        return "OPD"
    return None


def discover_modadd_runs(
    root: Path,
    *,
    p: int,
    m: int,
    subset_size: int,
    seed: int,
    etas: list[float] | None = None,
) -> list[RunRecord]:
    eta_set = None if etas is None else {float(eta) for eta in etas}
    records: list[RunRecord] = []

    for out_dir in sorted(root.rglob("out-modadd-*")):
        if not out_dir.is_dir():
            continue

        eval_history_path = out_dir / "eval_history.jsonl"
        if not eval_history_path.exists():
            continue

        name = out_dir.name
        offline_match = NOISY_BC_RE.match(name)
        if offline_match:
            run_p = int(offline_match.group("p"))
            run_m = int(offline_match.group("m"))
            run_n = int(offline_match.group("subset_size"))
            run_seed = int(offline_match.group("seed"))
            run_eta = parse_eta_tag(offline_match.group("eta"))
            if (run_p, run_m, run_n, run_seed) != (p, m, subset_size, seed):
                continue
            if eta_set is not None and run_eta not in eta_set:
                continue
            records.append(
                RunRecord(
                    run_id=unique_run_id(root, out_dir),
                    method="LogLossBC",
                    objective="sample_then_corrupt",
                    teacher_law="distributional_noise",
                    rollout_mode=offline_match.group("rollout_mode") or "greedy_then_corrupt",
                    p=run_p,
                    m=run_m,
                    subset_size=run_n,
                    eta=run_eta,
                    seed=run_seed,
                    temp_tag="",
                    out_dir=out_dir,
                    eval_history_path=eval_history_path,
                    last_eval_path=(out_dir / "last_eval.json") if (out_dir / "last_eval.json").exists() else None,
                    run_meta=None,
                    selection_rank=0,
                    selection_reason="standard_method",
                )
            )
            continue

        run_meta_path = out_dir / "run_meta.json"
        run_meta = load_json(run_meta_path) if run_meta_path.exists() else None

        student_prefix_match = STUDENT_PREFIX_RE.match(name)
        if student_prefix_match:
            run_p = int(student_prefix_match.group("p"))
            run_m = int(student_prefix_match.group("m"))
            run_n = int(student_prefix_match.group("subset_size"))
            run_seed = int(student_prefix_match.group("seed"))
            run_eta = parse_eta_tag(student_prefix_match.group("eta"))
            objective = (
                f"{student_prefix_match.group('method_family')}:"
                f"{student_prefix_match.group('loss')}:"
                f"{student_prefix_match.group('teacher_signal')}"
            )
            method = method_label(
                student_prefix_match.group("method_family"),
                student_prefix_match.group("loss"),
            )
            teacher_law = student_prefix_match.group("teacher_law")
            temp_tag = student_prefix_match.group("temp_suffix").lstrip("-")
        else:
            opd_match = LEGACY_OPD_RE.match(name)
            if not opd_match:
                continue
            objective = opd_match.group("objective")
            method = legacy_method_label(
                objective=objective,
                temp_tag=opd_match.group("temp_tag"),
                run_meta=run_meta,
            )
            if method is None:
                continue
            run_p = int(opd_match.group("p"))
            run_m = int(opd_match.group("m"))
            run_n = int(opd_match.group("subset_size"))
            run_seed = int(opd_match.group("seed"))
            run_eta = parse_eta_tag(opd_match.group("eta"))
            teacher_law = opd_match.group("teacher_law")
            temp_tag = opd_match.group("temp_tag")

        if (run_p, run_m, run_n, run_seed) != (p, m, subset_size, seed):
            continue
        if eta_set is not None and run_eta not in eta_set:
            continue

        selection_rank, selection_reason = classify_run_selection(
            method=method,
            out_dir=out_dir,
            run_meta=run_meta,
        )
        records.append(
            RunRecord(
                run_id=unique_run_id(root, out_dir),
                method=method,
                objective=objective,
                teacher_law=teacher_law,
                rollout_mode="",
                p=run_p,
                m=run_m,
                subset_size=run_n,
                eta=run_eta,
                seed=run_seed,
                temp_tag=temp_tag,
                out_dir=out_dir,
                eval_history_path=eval_history_path,
                last_eval_path=(out_dir / "last_eval.json") if (out_dir / "last_eval.json").exists() else None,
                run_meta=run_meta,
                selection_rank=selection_rank,
                selection_reason=selection_reason,
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
                "rollout_mode": record.rollout_mode,
                "p": record.p,
                "m": record.m,
                "subset_size": record.subset_size,
                "eta": record.eta,
                "seed": record.seed,
                "temp_tag": record.temp_tag,
                "final_iter": final_iter,
                "final_clean_full_exact": final_full,
                "final_clean_final_exact": final_final,
                "out_dir": str(record.out_dir),
                "eval_history_path": str(record.eval_history_path),
                "selection_rank": record.selection_rank,
                "selection_reason": record.selection_reason,
            }
        )

    runs_df = pd.DataFrame(rows).sort_values(["eta", "method", "selection_rank", "run_id"]).reset_index(drop=True)
    return runs_df, run_data


def default_output_dir(root: Path, *, p: int, m: int, subset_size: int, seed: int) -> Path:
    return root / "analysis" / "figures" / f"modadd_p{p}_m{m}_n{subset_size}_seed{seed}"


def plot_per_eta(
    run_data: dict[str, pd.DataFrame],
    runs_df: pd.DataFrame,
    *,
    metric: str,
    out_dir: Path | None = None,
    show: bool = True,
) -> None:
    etas = sorted(runs_df["eta"].unique())
    metric_name = metric.split("/")[-1]

    for eta in etas:
        fig, ax = plt.subplots(figsize=(14, 5.25), constrained_layout=True)
        method_rows = runs_df[runs_df["eta"] == eta].sort_values("method")
        missing_methods: list[str] = []

        for method in ("LogLossBC", "NAIL-forward", "NAIL-reverse", "OPD"):
            row_df = method_rows[method_rows["method"] == method]
            if row_df.empty:
                missing_methods.append(method)
                continue
            row = row_df.iloc[0]
            df = run_data[row["run_id"]].sort_values("iter")
            ax.plot(
                df["iter"],
                df[metric],
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                marker=METHOD_MARKERS[method],
                linewidth=2.4,
                markersize=4,
                label=method,
            )

        ax.set_title(
            f"ModAdd p={int(runs_df['p'].iloc[0])}, m={int(runs_df['m'].iloc[0])}, eta={eta:.2f}: "
            f"LogLossBC vs NAIL-forward vs NAIL-reverse vs OPD ({metric_name})"
        )
        ax.set_xlabel("iter")
        ax.set_ylabel(metric)
        ax.set_ylim(0.0, 1.01)
        ax.xaxis.set_major_locator(MultipleLocator(10000))
        ax.xaxis.set_minor_locator(MultipleLocator(5000))
        ax.grid(which="minor", linestyle=":", alpha=0.25)
        ax.grid(which="major", alpha=0.35)
        ax.legend(loc="best")

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"eta{eta_tag(eta)}_{metric_name}_methods.png"
            fig.savefig(out_path, dpi=220, bbox_inches="tight")
            print(f"Saved {out_path}")

        if missing_methods:
            print(f"Missing methods for eta={eta:.2f}: {', '.join(missing_methods)}")

        if show:
            plt.show()
        plt.close(fig)
