from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator


LEGACY_OPD_RE = re.compile(
    r"^out-modadd-opd-(?P<objective>.+?)-"
    r"p(?P<p>\d+)-m(?P<m>\d+)-n(?P<subset_size>\d+)-eta(?P<eta>[^-]+)-"
    r"(?P<teacher_law>[^-]+)-(?P<temp_tag>.+)-seed(?P<seed>\d+)$"
)

STUDENT_PREFIX_RE = re.compile(
    r"^out-modadd-(?P<method_family>opd|nail)-(?P<loss>forward|reverse)-(?P<teacher_signal>mc|full)-"
    r"p(?P<p>\d+)-m(?P<m>\d+)-n(?P<subset_size>\d+)-eta(?P<eta>[^-]+)-"
    r"(?P<teacher_law>[^-]+)(?P<temp_suffix>(?:-.*)?)\-seed(?P<seed>\d+)$"
)

METHOD_ORDER = {
    "NAIL-forward": 0,
    "NAIL-reverse": 1,
    "OPD": 2,
}

CURVE_COLORS = {
    ("NAIL-forward", "greedy"): "#D1495B",
    ("NAIL-forward", "sampled"): "#EDA43B",
    ("NAIL-reverse", "greedy"): "#F4A261",
    ("NAIL-reverse", "sampled"): "#E9C46A",
    ("OPD", "greedy"): "#00798C",
    ("OPD", "sampled"): "#5B8E7D",
}

ROLLOUT_LINESTYLES = {
    "greedy": "--",
    "sampled": "-",
}

ROLLOUT_MARKERS = {
    "greedy": "o",
    "sampled": "s",
}


@dataclass
class RunRecord:
    run_id: str
    objective: str
    method: str
    teacher_law: str
    p: int
    m: int
    subset_size: int
    eta: float
    seed: int
    temp_tag: str
    student_rollout_temperature: float
    student_temperature: float
    rollout_variant: str
    rollout_label: str
    display_label: str
    out_dir: Path
    eval_history_path: Path
    last_eval_path: Path | None
    run_meta: dict[str, object] | None


def parse_eta_tag(tag: str) -> float:
    return float(tag.replace("p", ".").replace("neg", "-"))


def eta_tag(eta: float) -> str:
    text = f"{eta:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "neg")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def method_label(method_family: str, loss: str) -> str:
    if method_family == "opd":
        return "OPD"
    if loss == "forward":
        return "NAIL-forward"
    return "NAIL-reverse"


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


def parse_temp_component(tag: str) -> float:
    if tag == "greedy":
        return 0.0
    if not tag.startswith("t"):
        raise ValueError(f"Unrecognized temperature tag: {tag!r}")
    return parse_eta_tag(tag[1:])


def parse_rollout_student_temps(
    temp_tag: str,
    *,
    run_meta: dict[str, object] | None,
) -> tuple[float, float]:
    rollout_temp = None
    student_temp = None

    if run_meta is not None:
        raw_rollout = run_meta.get("resolved_rollout_temperature", run_meta.get("student_rollout_temperature"))
        raw_student = run_meta.get("resolved_loss_temperature", run_meta.get("student_temperature"))
        if raw_rollout is not None:
            rollout_temp = float(raw_rollout)
        if raw_student is not None:
            student_temp = float(raw_student)

    if rollout_temp is not None and student_temp is not None:
        return rollout_temp, student_temp

    if temp_tag.startswith("roll") and "-stud" in temp_tag:
        rollout_tag, student_tag = temp_tag.split("-stud", maxsplit=1)
        rollout_temp = parse_temp_component(rollout_tag[len("roll"):])
        student_temp = parse_temp_component(student_tag)
        return rollout_temp, student_temp

    shared_temp = parse_temp_component(temp_tag)
    return shared_temp, shared_temp


def rollout_variant_label(rollout_temperature: float) -> tuple[str, str]:
    if float(rollout_temperature) == 0.0:
        return "greedy", "Greedy rollout"
    return "sampled", "Sampled rollout"


def discover_modadd_opd_runs(
    root: Path,
    *,
    p: int,
    m: int,
    subset_size: int,
    seed: int,
    etas: list[float] | None = None,
    objectives: list[str] | None = None,
) -> list[RunRecord]:
    eta_set = None if etas is None else {float(eta) for eta in etas}
    objective_set = None if objectives is None else set(objectives)
    records: list[RunRecord] = []

    for out_dir in sorted(root.rglob("out-modadd-*")):
        if not out_dir.is_dir():
            continue

        eval_history_path = out_dir / "eval_history.jsonl"
        if not eval_history_path.exists():
            continue

        student_prefix_match = STUDENT_PREFIX_RE.match(out_dir.name)
        run_meta_path = out_dir / "run_meta.json"
        run_meta = load_json(run_meta_path) if run_meta_path.exists() else None
        if student_prefix_match:
            run_p = int(student_prefix_match.group("p"))
            run_m = int(student_prefix_match.group("m"))
            run_n = int(student_prefix_match.group("subset_size"))
            run_seed = int(student_prefix_match.group("seed"))
            run_eta = parse_eta_tag(student_prefix_match.group("eta"))
            method = method_label(
                student_prefix_match.group("method_family"),
                student_prefix_match.group("loss"),
            )
            objective = method
            teacher_law = student_prefix_match.group("teacher_law")
            temp_tag = student_prefix_match.group("temp_suffix").lstrip("-")
        else:
            match = LEGACY_OPD_RE.match(out_dir.name)
            if not match:
                continue
            objective = match.group("objective")
            method = legacy_method_label(
                objective=objective,
                temp_tag=match.group("temp_tag"),
                run_meta=run_meta,
            )
            if method is None:
                continue
            run_p = int(match.group("p"))
            run_m = int(match.group("m"))
            run_n = int(match.group("subset_size"))
            run_seed = int(match.group("seed"))
            run_eta = parse_eta_tag(match.group("eta"))
            teacher_law = match.group("teacher_law")
            temp_tag = match.group("temp_tag")

        if (run_p, run_m, run_n, run_seed) != (p, m, subset_size, seed):
            continue
        if eta_set is not None and run_eta not in eta_set:
            continue
        if objective_set is not None and objective not in objective_set:
            continue
        rollout_temp, student_temp = parse_rollout_student_temps(
            temp_tag,
            run_meta=run_meta,
        )
        rollout_variant, rollout_label = rollout_variant_label(rollout_temp)

        records.append(
            RunRecord(
                run_id=out_dir.name,
                objective=objective,
                method=method,
                teacher_law=teacher_law,
                p=run_p,
                m=run_m,
                subset_size=run_n,
                eta=run_eta,
                seed=run_seed,
                temp_tag=temp_tag,
                student_rollout_temperature=rollout_temp,
                student_temperature=student_temp,
                rollout_variant=rollout_variant,
                rollout_label=rollout_label,
                display_label=f"{method}, {rollout_label.lower()}",
                out_dir=out_dir,
                eval_history_path=eval_history_path,
                last_eval_path=(out_dir / "last_eval.json") if (out_dir / "last_eval.json").exists() else None,
                run_meta=run_meta,
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
        history_df["objective"] = record.objective
        history_df["eta"] = record.eta
        history_df["rollout_variant"] = record.rollout_variant
        history_df["rollout_label"] = record.rollout_label
        history_df["display_label"] = record.display_label
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
                "seed": record.seed,
                "temp_tag": record.temp_tag,
                "student_rollout_temperature": record.student_rollout_temperature,
                "student_temperature": record.student_temperature,
                "rollout_variant": record.rollout_variant,
                "rollout_label": record.rollout_label,
                "display_label": record.display_label,
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
    runs_df = runs_df.sort_values(
        ["eta", "method", "student_rollout_temperature", "display_label"]
    ).reset_index(drop=True)
    return runs_df, run_data


def default_output_dir(root: Path, *, p: int, m: int, subset_size: int, seed: int) -> Path:
    return root / "analysis" / "figures" / f"modadd_p{p}_m{m}_opd_rollouts_n{subset_size}_seed{seed}"


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

    expected_combos = [
        ("NAIL-forward", "greedy"),
        ("NAIL-forward", "sampled"),
        ("NAIL-reverse", "greedy"),
        ("NAIL-reverse", "sampled"),
        ("OPD", "greedy"),
        ("OPD", "sampled"),
    ]

    for eta in etas:
        fig, ax = plt.subplots(figsize=(14.5, 5.6), constrained_layout=True)
        eta_rows = runs_df[runs_df["eta"] == eta]
        missing_labels: list[str] = []

        for objective, rollout_variant in expected_combos:
            row_df = eta_rows[
                (eta_rows["method"] == objective)
                & (eta_rows["rollout_variant"] == rollout_variant)
            ]
            if row_df.empty:
                missing_labels.append(f"{objective}, {rollout_variant} rollout")
                continue

            row = row_df.iloc[0]
            df = run_data[row["run_id"]].sort_values("iter")
            ax.plot(
                df["iter"],
                df[metric],
                color=CURVE_COLORS[(objective, rollout_variant)],
                linestyle=ROLLOUT_LINESTYLES[rollout_variant],
                marker=ROLLOUT_MARKERS[rollout_variant],
                linewidth=2.4,
                markersize=4,
                label=row["display_label"],
            )

        ax.set_title(
            f"ModAdd p={int(runs_df['p'].iloc[0])}, m={int(runs_df['m'].iloc[0])}, eta={eta:.2f}: "
            f"NAIL-forward vs NAIL-reverse vs OPD rollouts ({metric_name})"
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
            out_path = out_dir / f"eta{eta_tag(eta)}_{metric_name}_opd_rollouts.png"
            fig.savefig(out_path, dpi=220, bbox_inches="tight")
            print(f"Saved {out_path}")

        if missing_labels:
            print(f"Missing curves for eta={eta:.2f}: {', '.join(missing_labels)}")

        if show:
            plt.show()
        plt.close(fig)
