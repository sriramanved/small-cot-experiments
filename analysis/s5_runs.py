from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "manifests" / "s5_low_noise_comparison.csv"
)

PATH_COLUMNS = ("out_dir", "train_log_path")
NUMERIC_COLUMNS = (
    "eta",
    "subset_size",
    "summary_iter",
    "summary_val_loss",
    "summary_val_cot_exact",
    "summary_val_clean_full_exact",
    "summary_val_clean_final_exact",
)
SUMMARY_COLUMNS = (
    "summary_iter",
    "summary_val_loss",
    "summary_val_cot_exact",
    "summary_val_clean_full_exact",
    "summary_val_clean_final_exact",
)
EVAL_ALIASES = {
    "train_loss": "train/loss_eval",
    "val_loss": "val/loss",
    "train_clean_oracle_loss": "train/clean_oracle_loss_eval",
    "val_cot_exact": "val/cot_exact",
    "val_clean_full_exact": "val/clean_full_exact",
    "val_clean_final_exact": "val/clean_final_exact",
}
TRAIN_LINE_RE = re.compile(r"^iter (?P<iter>\d+): (?P<body>.+)$")
EVAL_LINE_RE = re.compile(r"^(?P<reason>periodic|eval|final) step (?P<iter>\d+): (?P<body>.+)$")


def _resolve_root(root: str | Path) -> Path:
    return Path(root).expanduser().resolve()


def _resolve_path(root: Path, value: Any) -> Path | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _coerce_scalar(text: str) -> float | int | None:
    text = text.strip()
    if not text:
        return None
    if text.endswith("ms"):
        text = text[:-2]
    elif text.endswith("%"):
        text = text[:-1]
    try:
        value = float(text)
    except ValueError:
        return None
    if value.is_integer():
        return int(value)
    return value


def _parse_metric_body(body: str) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for segment in body.split(","):
        segment = segment.strip()
        if not segment or " " not in segment:
            continue
        key_text, value_text = segment.rsplit(" ", 1)
        value = _coerce_scalar(value_text)
        if value is None:
            continue
        key = key_text.strip().replace("/", "_").replace(" ", "_")
        metrics[key] = value
    return metrics


def _load_eval_history(run_dir: Path | None) -> pd.DataFrame:
    if run_dir is None:
        return pd.DataFrame()
    history_path = run_dir / "eval_history.jsonl"
    last_eval_path = run_dir / "last_eval.json"
    rows: list[dict[str, Any]] = []
    if history_path.exists():
        rows = _read_jsonl(history_path)
    elif last_eval_path.exists():
        rows = [_load_json(last_eval_path)]
    if not rows:
        return pd.DataFrame()
    history = pd.DataFrame(rows)
    if "iter" in history.columns:
        history = history.sort_values("iter").reset_index(drop=True)
    return history


def _load_completed_iter(run_dir: Path | None) -> int | None:
    if run_dir is None:
        return None
    completed_path = run_dir / "completed.txt"
    if not completed_path.exists():
        return None
    for line in completed_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("iter_num="):
            try:
                return int(line.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _load_stdout_history(log_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if log_path is None or not log_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        train_match = TRAIN_LINE_RE.match(line)
        if train_match is not None:
            row = {"iter": int(train_match.group("iter"))}
            row.update(_parse_metric_body(train_match.group("body")))
            train_rows.append(row)
            continue

        eval_match = EVAL_LINE_RE.match(line)
        if eval_match is not None:
            row = {
                "iter": int(eval_match.group("iter")),
                "reason": eval_match.group("reason"),
            }
            row.update(_parse_metric_body(eval_match.group("body")))
            for source_col, alias_col in EVAL_ALIASES.items():
                if source_col in row and alias_col not in row:
                    row[alias_col] = row[source_col]
            eval_rows.append(row)

    train_history = pd.DataFrame(train_rows)
    eval_history = pd.DataFrame(eval_rows)
    if not train_history.empty:
        train_history = train_history.sort_values("iter").reset_index(drop=True)
    if not eval_history.empty:
        eval_history = eval_history.sort_values("iter").reset_index(drop=True)
    return train_history, eval_history


def load_manifest(
    root: str | Path = ".",
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the run manifest into a dataframe with resolved filesystem paths."""

    root_path = _resolve_root(root)
    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else DEFAULT_MANIFEST_PATH
    )
    frame = pd.read_csv(manifest)
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in PATH_COLUMNS:
        resolved_col = f"{column}_resolved"
        frame[resolved_col] = [
            _resolve_path(root_path, value) for value in frame.get(column, pd.Series([None] * len(frame)))
        ]
    frame["root_dir"] = root_path
    return frame


def load_run_data(run_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Load per-run history tables and metadata keyed by run_id."""

    run_data: dict[str, dict[str, Any]] = {}
    for row in run_df.to_dict("records"):
        run_id = str(row["run_id"])
        run_dir = row.get("out_dir_resolved")
        train_log_path = row.get("train_log_path_resolved")

        eval_history = _load_eval_history(run_dir)
        train_history, stdout_eval_history = _load_stdout_history(train_log_path)
        if eval_history.empty and not stdout_eval_history.empty:
            eval_history = stdout_eval_history.copy()

        summary = {
            column.removeprefix("summary_"): row.get(column)
            for column in SUMMARY_COLUMNS
            if column in row and not pd.isna(row[column])
        }

        run_meta = None
        if run_dir is not None:
            meta_path = run_dir / "run_meta.json"
            if meta_path.exists():
                run_meta = _load_json(meta_path)

        run_data[run_id] = {
            "run_dir": run_dir,
            "train_log_path": train_log_path,
            "summary": summary,
            "eval_history": eval_history,
            "train_history": train_history,
            "stdout_eval_history": stdout_eval_history,
            "run_meta": run_meta,
            "completed_iter": _load_completed_iter(run_dir),
            "has_local_eval_history": not eval_history.empty,
            "has_local_train_history": not train_history.empty,
        }
    return run_data


def build_summary_table(run_df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact dataframe for quick comparison tables in notebooks."""

    preferred_columns = [
        "run_id",
        "plot_group",
        "method_label",
        "eta",
        "objective",
        "teacher_law",
        "summary_iter",
        "summary_val_loss",
        "summary_val_clean_full_exact",
        "summary_val_clean_final_exact",
        "notes",
    ]
    available_columns = [column for column in preferred_columns if column in run_df.columns]
    return run_df.loc[:, available_columns].sort_values(["eta", "plot_group"]).reset_index(drop=True)


def stack_history(
    run_df: pd.DataFrame,
    run_data: dict[str, dict[str, Any]],
    *,
    history_key: str,
    metric: str,
) -> pd.DataFrame:
    """Stack a requested metric across runs into one long dataframe."""

    rows: list[pd.DataFrame] = []
    for row in run_df.to_dict("records"):
        run_id = str(row["run_id"])
        history = run_data.get(run_id, {}).get(history_key, pd.DataFrame())
        if history is None or history.empty or metric not in history.columns:
            continue
        piece = history.loc[:, ["iter", metric]].rename(columns={metric: "value"}).copy()
        piece["metric"] = metric
        for key in (
            "run_id",
            "plot_group",
            "method_label",
            "method_family",
            "objective",
            "teacher_law",
            "rollout_mode",
            "eta",
        ):
            if key == "run_id":
                piece[key] = run_id
            elif key in row:
                piece[key] = row[key]
        rows.append(piece)

    if not rows:
        return pd.DataFrame(
            columns=[
                "iter",
                "value",
                "metric",
                "run_id",
                "plot_group",
                "method_label",
                "method_family",
                "objective",
                "teacher_law",
                "rollout_mode",
                "eta",
            ]
        )

    return pd.concat(rows, ignore_index=True).sort_values(
        ["plot_group", "eta", "iter"]
    ).reset_index(drop=True)
