from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
for path in (ROOT, SRC_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from nanogpt.methods.student_prefix import normalize_student_prefix_method

try:
    from scripts.plot_style import (
        apply_iteration_axis,
        get_method_style,
        metric_display_label,
        polish_axes,
        save_publication_figure,
        set_publication_style,
    )
except ModuleNotFoundError:
    from plot_style import (
        apply_iteration_axis,
        get_method_style,
        metric_display_label,
        polish_axes,
        save_publication_figure,
        set_publication_style,
    )


DEFAULT_OVERRIDE_PATH = ROOT / "analysis" / "cache" / "s5_online_seed_sweeps" / "run_overrides.json"
DEFAULT_WANDB_CACHE_DIR = ROOT / "analysis" / "cache" / "s5_online_seed_sweeps" / "wandb"
DEFAULT_NAIL_REVERSE_MIN_ARTIFACT_UTC = pd.Timestamp("2026-04-26T00:00:00+00:00")
DEFAULT_EFFECTIVE_BATCH_SIZE = 64
TRUSTED_NAIL_REVERSE_VARIANTS = {
    "manual_preferred",
    "fixed_aux_actions",
    "likely_fixed_recent_code",
    "wandb_explicit",
}

WANDB_EVAL_KEY_GROUPS = (
    ("iter", "val/clean_full_exact", "val/clean_final_exact", "val/loss"),
    ("iter", "val/clean_full_exact", "val/clean_final_exact"),
    ("iter_num", "val_clean_full_exact", "val_clean_final_exact", "val_loss"),
    ("iter_num", "val_clean_full_exact", "val_clean_final_exact"),
)

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

PLOT_METHODS = (
    "LogLossBC",
    "OPD-R",
    "NAIL-F, greedy rollout",
    "OPD-F",
    "NAIL-R, greedy rollout",
)

METHOD_COLORS = {method: get_method_style(method).color for method in PLOT_METHODS}

PLOT_LEGEND_LABELS = {
    "LogLossBC": "LogLossBC",
    "OPD-R": "OPD-R",
    "NAIL-F, greedy rollout": "NAIL-F",
    "OPD-F": "OPD-F",
    "NAIL-R, greedy rollout": "NAIL-R",
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

SOURCE_LOCATION_NAMES = {"local", "blocklab", "dev_node", "aics"}


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
    source_order: int
    reverse_variant: str
    selection_rank: int
    selection_reason: str
    artifact_mtime: float
    artifact_datetime_utc: str


def parse_eta_tag(tag: str) -> float:
    return float(tag.replace("p", ".").replace("neg", "-"))


def eta_tag(eta: float) -> str:
    text = f"{eta:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "neg")


def rollout_temperature_from_suffix(temp_suffix: str) -> float | None:
    match = re.search(r"(?:^|-)rollt(?P<tag>[^-]+)", temp_suffix)
    if match is None:
        return None
    return parse_eta_tag(match.group("tag"))


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def artifact_timestamp_for_run(out_dir: Path) -> tuple[float, str]:
    candidates = [
        out_dir / "last_eval.json",
        out_dir / "eval_history.jsonl",
        out_dir / "run_meta.json",
        out_dir / "launcher_config.json",
        out_dir / "launcher_command.txt",
        out_dir / "completed.txt",
    ]
    mtimes = [path.stat().st_mtime for path in candidates if path.exists()]
    if not mtimes:
        mtime = out_dir.stat().st_mtime
    else:
        mtime = max(mtimes)
    iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    return mtime, iso


def wandb_cache_dir_for_run_path(run_path: str, cache_dir: Path = DEFAULT_WANDB_CACHE_DIR) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "__", run_path.strip("/"))
    return cache_dir / safe_name


def _scan_wandb_eval_history(run: object) -> list[dict[str, object]]:
    for keys in WANDB_EVAL_KEY_GROUPS:
        rows = list(run.scan_history(keys=list(keys), page_size=1000))
        if rows:
            return rows
    rows = list(run.scan_history(page_size=1000))
    return [
        row
        for row in rows
        if any(
            key in row
            for key in (
                "val/clean_full_exact",
                "val_clean_full_exact",
                "val/clean_final_exact",
                "val_clean_final_exact",
            )
        )
    ]


def _last_eval_from_history_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    history_df, _ = normalize_history_df(pd.DataFrame(rows))
    final_row = history_df.iloc[-1]
    last_eval: dict[str, object] = {
        "iter": int(final_row["iter"]),
        "reason": str(final_row.get("reason", "final")),
        "val/clean_full_exact": float(final_row["val/clean_full_exact"]),
        "val/clean_final_exact": float(final_row["val/clean_final_exact"]),
    }
    if "val/loss" in final_row and not pd.isna(final_row["val/loss"]):
        last_eval["val/loss"] = float(final_row["val/loss"])
    return last_eval


def sync_wandb_run_to_cache(
    run_path: str,
    *,
    cache_dir: Path = DEFAULT_WANDB_CACHE_DIR,
    refresh: bool = False,
    api: object | None = None,
    allow_api: bool = True,
    timeout: int = 120,
) -> Path:
    run_cache_dir = wandb_cache_dir_for_run_path(run_path, cache_dir)
    eval_history_path = run_cache_dir / "eval_history.jsonl"
    run_meta_path = run_cache_dir / "run_meta.json"
    last_eval_path = run_cache_dir / "last_eval.json"

    if not refresh and eval_history_path.exists() and run_meta_path.exists() and last_eval_path.exists():
        return run_cache_dir

    if not allow_api:
        raise FileNotFoundError(
            f"No cached W&B history for {run_path} under {run_cache_dir}. "
            "Set ALLOW_WANDB_API=True in the notebook to fetch it live."
        )

    if api is None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb is required to sync explicit W&B run paths. Install wandb or use an existing cache."
            ) from exc
        api = wandb.Api(timeout=timeout)

    run = api.run(run_path)
    rows = _scan_wandb_eval_history(run)
    if not rows:
        raise RuntimeError(f"No eval history rows found for W&B run {run_path}")

    run_cache_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_history_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")

    config = _json_safe(dict(getattr(run, "config", {}) or {}))
    run_path_parts = getattr(run, "path", None)
    canonical_run_path = "/".join(run_path_parts) if isinstance(run_path_parts, (list, tuple)) else run_path
    run_meta = {
        **(config if isinstance(config, dict) else {}),
        "wandb_run_path": canonical_run_path,
        "wandb_requested_run_path": run_path,
        "wandb_run_id": str(getattr(run, "id", run_path.rsplit("/", 1)[-1])),
        "wandb_run_name": str(getattr(run, "name", "")),
        "wandb_state": str(getattr(run, "state", "")),
        "wandb_url": str(getattr(run, "url", "")),
        "source": "wandb_explicit",
    }
    for attr in ("created_at", "updated_at"):
        value = getattr(run, attr, None)
        if value is not None:
            run_meta[f"wandb_{attr}"] = str(value)
    write_json(run_meta_path, run_meta)
    write_json(last_eval_path, _last_eval_from_history_rows(rows))
    with open(run_cache_dir / "completed.txt", "w", encoding="utf-8") as f:
        f.write(f"wandb_state={getattr(run, 'state', '')}\n")
    return run_cache_dir


def source_kind_for_root(root: Path) -> str:
    resolved = root.resolve()
    if resolved == ROOT:
        return "local"

    try:
        rel_parts = resolved.relative_to(ROOT).parts
    except ValueError:
        rel_parts = resolved.parts

    if rel_parts[:1] == ("reruns",):
        return "local"
    for part in rel_parts:
        if part in SOURCE_LOCATION_NAMES:
            return part
    return resolved.name


def source_priority_for_root(root: Path) -> int:
    resolved = root.resolve()
    source_kind = source_kind_for_root(resolved)
    if source_kind == "local":
        return SOURCE_PRIORITY[source_kind]

    try:
        rel_parts = resolved.relative_to(ROOT).parts
    except ValueError:
        rel_parts = resolved.parts

    if "analysis" in rel_parts and "cache" in rel_parts:
        return 10 + SOURCE_PRIORITY.get(source_kind, 9)
    if "analysis" in rel_parts and "imports" in rel_parts:
        return 20 + SOURCE_PRIORITY.get(source_kind, 9)
    return 30 + SOURCE_PRIORITY.get(source_kind, 99)


def _path_suffix_from_parts(parts: tuple[str, ...], anchor: str) -> str | None:
    if anchor not in parts:
        return None
    index = parts.index(anchor)
    return Path(*parts[index:]).as_posix()


def _override_tokens_for_text(text: str) -> set[str]:
    path = Path(text)
    tokens = {path.as_posix()}
    if not path.is_absolute():
        tokens.add((ROOT / path).as_posix())
    else:
        tokens.add(path.as_posix())
    try:
        tokens.add(path.resolve().as_posix())
    except OSError:
        pass
    reruns_suffix = _path_suffix_from_parts(path.parts, "reruns")
    if reruns_suffix is not None:
        tokens.add(reruns_suffix)
    return tokens


def _override_tokens_for_run_dir(path: Path) -> set[str]:
    resolved = path.resolve()
    tokens = {resolved.as_posix()}
    try:
        tokens.add(resolved.relative_to(ROOT).as_posix())
    except ValueError:
        pass
    reruns_suffix = _path_suffix_from_parts(resolved.parts, "reruns")
    if reruns_suffix is not None:
        tokens.add(reruns_suffix)
    return tokens


def run_matches_override(path: Path, override_tokens: set[str]) -> bool:
    return not override_tokens.isdisjoint(_override_tokens_for_run_dir(path))


def load_run_selection_overrides(path: Path | None = None) -> tuple[set[str], set[str]]:
    override_path = DEFAULT_OVERRIDE_PATH if path is None else path
    if not override_path.exists():
        return set(), set()

    payload = load_json(override_path)
    preferred: set[str] = set()
    for value in payload.get("preferred_run_ids", []):
        preferred.update(_override_tokens_for_text(str(value)))
    excluded: set[str] = set()
    for value in payload.get("excluded_run_ids", []):
        excluded.update(_override_tokens_for_text(str(value)))
    return preferred, excluded


def classify_run_selection(
    *,
    method: str,
    out_dir: Path,
    run_meta: dict[str, object] | None,
    preferred_run_ids: set[str],
) -> tuple[str, int, str]:
    resolved_out_dir = out_dir.resolve()
    manual_rank = -100 if run_matches_override(resolved_out_dir, preferred_run_ids) else 0
    manual_reason = "manual_preferred_run_ids" if manual_rank < 0 else ""

    if method == "LogLossBC":
        if manual_reason:
            return "manual_preferred", manual_rank, manual_reason

        offline_match = OFFLINE_BC_RE.match(out_dir.name)
        suffixes = set()
        if offline_match is not None:
            suffixes = set(filter(None, offline_match.group("suffixes").split("-")))
        is_sampled_offline = "sample" in suffixes
        path_text = str(resolved_out_dir)

        if is_sampled_offline and "dev_node" in path_text:
            return "offline_bc_sample_dev_node", -20, "offline_bc_sample_on_dev_node"
        if is_sampled_offline:
            return "offline_bc_sample", -10, "offline_bc_sample_run"
        if "dev_node" in path_text:
            return "offline_bc_dev_node", 5, "offline_bc_on_dev_node"
        return "offline_bc_other", 10, "offline_bc_non_sample_run"

    if method != "NAIL-R, greedy rollout":
        if manual_reason:
            return "manual_preferred", manual_rank, manual_reason
        return "standard", 0, "standard_method"

    reverse_action_source = None if run_meta is None else run_meta.get("reverse_action_source")
    if reverse_action_source == "student_aux_sample":
        if manual_reason:
            return "manual_preferred", manual_rank, manual_reason
        return "fixed_aux_actions", 0, "run_meta.reverse_action_source=student_aux_sample"
    if reverse_action_source == "rollout_action":
        if manual_reason:
            return "manual_preferred", manual_rank, manual_reason
        return "legacy_rollout_actions", 40, "run_meta.reverse_action_source=rollout_action"

    path_text = str(resolved_out_dir)
    if "nail_reverse_mc_fixed" in path_text:
        if manual_reason:
            return "manual_preferred", manual_rank, manual_reason
        return "fixed_aux_actions", 1, "path_contains_nail_reverse_mc_fixed"

    if run_meta is not None and all(key in run_meta for key in ("target_len", "answer_len", "target_span")):
        if manual_reason:
            return "manual_preferred", manual_rank, manual_reason
        return "likely_fixed_recent_code", 2, "post_patch_run_meta_fields_present"

    if manual_reason:
        return "manual_preferred", manual_rank, manual_reason
    return "unknown_or_legacy", 30, "no_fixed_nail_reverse_marker"


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


def _metric_matches_expected(
    row: pd.Series,
    *,
    expected_key: str,
    alias_key: str,
    expected: dict[str, object],
) -> bool:
    expected_value = expected.get(expected_key, expected.get(alias_key))
    if expected_value is None:
        return False
    actual_value = row.get(expected_key, row.get(alias_key))
    if actual_value is None or pd.isna(actual_value):
        return False
    return abs(float(actual_value) - float(expected_value)) <= 1e-12


def _last_eval_match_score(row: pd.Series, last_eval: dict[str, object]) -> int | None:
    expected_final_iter = last_eval.get("iter", last_eval.get("iter_num"))
    if expected_final_iter is None or float(row["iter"]) != float(expected_final_iter):
        return None

    score = 0
    if "reason" in row and row.get("reason") == "final":
        score -= 10
    if _metric_matches_expected(
        row,
        expected_key="val/clean_full_exact",
        alias_key="val_clean_full_exact",
        expected=last_eval,
    ):
        score -= 2
    if _metric_matches_expected(
        row,
        expected_key="val/clean_final_exact",
        alias_key="val_clean_final_exact",
        expected=last_eval,
    ):
        score -= 2
    return score


def select_latest_history_segment(
    df: pd.DataFrame,
    *,
    last_eval: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if df.empty:
        return df.copy(), {
            "history_segment_count": 0,
            "history_selected_segment": 0,
            "history_restart_count": 0,
        }

    work = df.copy()
    segment_ids: list[int] = []
    current_segment = 0
    prev_iter: float | None = None
    for iter_value in work["iter"].tolist():
        numeric_iter = float(iter_value)
        if prev_iter is not None and numeric_iter < prev_iter:
            current_segment += 1
        segment_ids.append(current_segment)
        prev_iter = numeric_iter
    work["segment_id"] = segment_ids

    selected_segment = current_segment
    if last_eval is not None:
        expected_final_iter = last_eval.get("iter", last_eval.get("iter_num"))
        if expected_final_iter is not None:
            expected_final_iter = float(expected_final_iter)
            candidate_segments: list[tuple[int, int, int]] = []
            for segment_id in sorted(work["segment_id"].unique()):
                segment_df = work[work["segment_id"] == segment_id]
                iter_matches = segment_df[segment_df["iter"] == expected_final_iter]
                if iter_matches.empty:
                    continue
                best_match_score = min(
                    score
                    for score in (
                        _last_eval_match_score(row, last_eval) for _, row in iter_matches.iterrows()
                    )
                    if score is not None
                )
                candidate_segments.append((best_match_score, int(segment_id), len(candidate_segments)))
            if candidate_segments:
                candidate_segments.sort(key=lambda item: (item[0], -item[1]))
                selected_segment = candidate_segments[0][1]

    selected = work[work["segment_id"] == selected_segment].copy()
    if last_eval is not None:
        best_row_order: int | None = None
        best_score: int | None = None
        for _, row in selected.iterrows():
            score = _last_eval_match_score(row, last_eval)
            if score is None:
                continue
            row_order = int(row["_row_order"]) if "_row_order" in row else 0
            if best_score is None or score < best_score or (score == best_score and row_order > best_row_order):
                best_score = score
                best_row_order = row_order
        if best_row_order is not None:
            selected = selected[selected["_row_order"] <= best_row_order].copy()
    selected = selected.drop_duplicates(subset=["iter"], keep="last")
    selected = selected.sort_values("iter").reset_index(drop=True)
    return selected, {
        "history_segment_count": int(current_segment + 1),
        "history_selected_segment": int(selected_segment),
        "history_restart_count": int(current_segment),
    }


def normalize_history_df(
    df: pd.DataFrame,
    *,
    last_eval: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rename_map = {
        "val_clean_full_exact": "val/clean_full_exact",
        "val_clean_final_exact": "val/clean_final_exact",
        "val_loss": "val/loss",
        "iter_num": "iter",
    }
    work = coalesce_duplicate_columns(df.rename(columns=rename_map).copy())
    work["_row_order"] = range(len(work))
    expected = ["iter", "val/clean_full_exact", "val/clean_final_exact"]
    missing = [col for col in expected if col not in work.columns]
    if missing:
        raise ValueError(f"Missing required history columns: {missing}")
    work["iter"] = pd.to_numeric(work["iter"], errors="coerce")
    for metric in ("val/clean_full_exact", "val/clean_final_exact", "val/loss"):
        if metric in work.columns:
            work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=["iter"]).sort_values("_row_order").reset_index(drop=True)
    selected, segment_meta = select_latest_history_segment(work, last_eval=last_eval)
    drop_cols = [col for col in ("_row_order", "segment_id") if col in selected.columns]
    if drop_cols:
        selected = selected.drop(columns=drop_cols)
    return selected, segment_meta


def method_label_from_state(method_state: dict[str, object]) -> str | None:
    method_family = str(method_state["method_family"])
    loss = str(method_state["loss"])
    teacher_signal = str(method_state["teacher_signal"])
    rollout_temperature = float(method_state["resolved_rollout_temperature"])

    if teacher_signal != "mc":
        return None
    if method_family == "opd" and loss == "reverse":
        return "OPD-R"
    if method_family == "nail" and loss == "forward" and rollout_temperature == 0.0:
        return "NAIL-F, greedy rollout"
    if method_family == "nail" and loss == "reverse" and rollout_temperature == 0.0:
        return "NAIL-R, greedy rollout"
    if method_family == "nail" and loss == "forward" and rollout_temperature > 0.0:
        return "OPD-F"
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
        return "OPD-R"
    if objective.startswith("forward_kl_"):
        return "OPD-F"
    return None


def iter_run_dirs(root: Path, *, prefix: str = "out-s5-", max_depth: int | None = None) -> list[Path]:
    if root.name.startswith(prefix):
        return [root]

    run_dirs: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            with os.scandir(current) as entries:
                child_dirs = [entry for entry in entries if entry.is_dir(follow_symlinks=False)]
        except OSError:
            continue

        if max_depth is not None and depth >= max_depth:
            continue
        for entry in sorted(child_dirs, key=lambda item: item.name, reverse=True):
            if entry.name.startswith(prefix):
                run_dirs.append(Path(entry.path))
            else:
                stack.append((Path(entry.path), depth + 1))
    return sorted(run_dirs)


def discover_s5_online_runs(
    search_roots: list[Path],
    *,
    m: int,
    seeds: list[int],
    etas: list[float],
    subset_size: int | None = None,
    teacher_seed: int | None = None,
    override_path: Path | None = None,
    max_depth: int | None = None,
) -> list[RunRecord]:
    seed_set = {int(seed) for seed in seeds}
    eta_set = {float(eta) for eta in etas}
    seen_dirs: set[Path] = set()
    records: list[RunRecord] = []
    preferred_run_ids, excluded_run_ids = load_run_selection_overrides(override_path)

    for root in search_roots:
        if not root.exists():
            continue
        root = root.resolve()
        source_kind = source_kind_for_root(root)
        source_order = source_priority_for_root(root)

        for out_dir in iter_run_dirs(root, max_depth=max_depth):
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
            if run_matches_override(resolved, excluded_run_ids):
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
                rollout_temperature = rollout_temperature_from_suffix(
                    student_prefix_match.group("temp_suffix")
                )
                fallback_method_state = {
                    "method_family": student_prefix_match.group("method_family"),
                    "loss": student_prefix_match.group("loss"),
                    "teacher_signal": student_prefix_match.group("teacher_signal"),
                }
                if rollout_temperature is not None:
                    fallback_method_state["rollout_temperature_override"] = rollout_temperature
                method_state = normalize_student_prefix_method(
                    run_meta or fallback_method_state
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
                    method = "LogLossBC"
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
            if teacher_signal != "mc" and method != "LogLossBC":
                continue
            if run_m != m or run_seed not in seed_set or run_eta not in eta_set:
                continue
            if subset_size is not None and run_n != subset_size:
                continue
            if run_meta is not None and teacher_seed is not None:
                teacher_checkpoint = str(run_meta.get("teacher_checkpoint", ""))
                if teacher_checkpoint and f"teacher{teacher_seed}" not in teacher_checkpoint:
                    continue

            reverse_variant, selection_rank, selection_reason = classify_run_selection(
                method=method,
                out_dir=out_dir,
                run_meta=run_meta,
                preferred_run_ids=preferred_run_ids,
            )
            artifact_mtime, artifact_datetime_utc = artifact_timestamp_for_run(out_dir)

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
                    source_order=source_order,
                    reverse_variant=reverse_variant,
                    selection_rank=selection_rank,
                    selection_reason=selection_reason,
                    artifact_mtime=artifact_mtime,
                    artifact_datetime_utc=artifact_datetime_utc,
                )
            )
            seen_dirs.add(resolved)

    return records


def _config_value(payload: dict[str, object], *paths: str) -> object | None:
    for path in paths:
        if path in payload and payload[path] is not None:
            return payload[path]
        current: object = payload
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current is not None:
            return current
    return None


def _optional_int(value: object | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _optional_float(value: object | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _first_not_none(*values: object | None) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def effective_batch_size_from_meta(run_meta: dict[str, object] | None) -> int:
    if run_meta is None:
        return DEFAULT_EFFECTIVE_BATCH_SIZE

    explicit_effective = _optional_int(
        _config_value(
            run_meta,
            "effective_batch_size",
            "optim.effective_batch_size",
            "train.effective_batch_size",
        )
    )
    if explicit_effective is not None:
        return explicit_effective

    batch_size = _optional_int(
        _config_value(run_meta, "batch_size", "optim.batch_size", "train.batch_size")
    )
    grad_accum = _optional_int(
        _config_value(
            run_meta,
            "gradient_accumulation_steps",
            "optim.gradient_accumulation_steps",
            "train.gradient_accumulation_steps",
        )
    )
    world_size = _optional_int(
        _config_value(run_meta, "world_size", "ddp_world_size", "runtime.world_size", "trainer.world_size")
    )
    return int(batch_size or DEFAULT_EFFECTIVE_BATCH_SIZE) * int(grad_accum or 1) * int(world_size or 1)


def _iter_explicit_wandb_specs(
    run_paths_by_eta: dict[float, object],
    *,
    seeds: list[int],
) -> list[tuple[float, int | None, str]]:
    specs: list[tuple[float, int | None, str]] = []
    for eta, value in run_paths_by_eta.items():
        eta_value = float(eta)
        if isinstance(value, dict):
            for seed, run_path in value.items():
                if run_path:
                    specs.append((eta_value, int(seed), str(run_path)))
            continue
        run_paths = [value] if isinstance(value, str) else list(value or [])
        for index, run_path in enumerate(run_paths):
            if not run_path:
                continue
            seed = int(seeds[index]) if index < len(seeds) else None
            specs.append((eta_value, seed, str(run_path)))
    return specs


def explicit_wandb_etas(run_paths_by_eta: dict[float, object]) -> set[float]:
    return {
        float(eta)
        for eta, value in run_paths_by_eta.items()
        if any(run_path for _, _, run_path in _iter_explicit_wandb_specs({float(eta): value}, seeds=[]))
    }


def expected_seeds_by_eta_for_wandb_overrides(
    run_paths_by_eta: dict[float, object],
    *,
    default_seeds: list[int],
    etas: list[float],
) -> dict[float, list[int]]:
    expected: dict[float, list[int]] = {}
    explicit_by_eta: dict[float, list[int]] = {}
    for eta, seed, _ in _iter_explicit_wandb_specs(run_paths_by_eta, seeds=default_seeds):
        if seed is not None:
            explicit_by_eta.setdefault(float(eta), []).append(int(seed))

    for eta in etas:
        eta_value = float(eta)
        explicit_seeds = explicit_by_eta.get(eta_value)
        if explicit_seeds:
            expected[eta_value] = sorted(set(explicit_seeds))
        else:
            expected[eta_value] = list(default_seeds)
    return expected


def discover_wandb_s5_online_runs(
    run_paths_by_eta: dict[float, object],
    *,
    m: int,
    seeds: list[int],
    etas: list[float],
    subset_size: int | None = None,
    teacher_seed: int | None = None,
    cache_dir: Path = DEFAULT_WANDB_CACHE_DIR,
    refresh: bool = False,
    api: object | None = None,
    allow_api: bool = True,
    skip_missing_cache: bool = False,
    method: str = "NAIL-R, greedy rollout",
) -> list[RunRecord]:
    seed_set = {int(seed) for seed in seeds}
    eta_set = {float(eta) for eta in etas}
    records: list[RunRecord] = []

    for fallback_eta, fallback_seed, run_path in _iter_explicit_wandb_specs(
        run_paths_by_eta,
        seeds=seeds,
    ):
        if fallback_eta not in eta_set:
            continue
        try:
            run_cache_dir = sync_wandb_run_to_cache(
                run_path,
                cache_dir=cache_dir,
                refresh=refresh,
                api=api,
                allow_api=allow_api,
            )
        except FileNotFoundError:
            if skip_missing_cache:
                continue
            raise
        run_meta_path = run_cache_dir / "run_meta.json"
        run_meta = load_json(run_meta_path)

        run_name = str(run_meta.get("wandb_run_name", ""))
        name_match = STUDENT_PREFIX_RE.match(run_name)
        parsed_m = _optional_int(name_match.group("m")) if name_match else None
        parsed_seed = _optional_int(name_match.group("seed")) if name_match else None
        parsed_eta = parse_eta_tag(name_match.group("eta")) if name_match else None
        parsed_n = _optional_int(name_match.group("subset_size")) if name_match else None
        parsed_teacher_law = name_match.group("teacher_law") if name_match else None

        run_m = int(
            _first_not_none(
                _optional_int(_config_value(run_meta, "m", "task.m", "s5_m")),
                parsed_m,
                m,
            )
        )
        run_seed = _first_not_none(
            _optional_int(_config_value(run_meta, "seed", "task.seed", "train_seed")),
            parsed_seed,
            fallback_seed,
        )
        run_eta = float(
            _first_not_none(
                _optional_float(_config_value(run_meta, "eta", "task.eta")),
                parsed_eta,
                fallback_eta,
            )
        )
        run_n = int(
            _first_not_none(
                _optional_int(_config_value(run_meta, "subset_size", "task.subset_size")),
                parsed_n,
                subset_size,
                0,
            )
        )
        teacher_law = str(
            _config_value(run_meta, "teacher_law", "task.teacher_law")
            or parsed_teacher_law
            or "distributional_noise"
        )

        if run_seed is None:
            continue
        run_seed = int(run_seed)
        if run_seed not in seed_set or run_m != m or run_eta not in eta_set:
            continue
        if subset_size is not None and run_n != subset_size:
            continue
        if teacher_seed is not None:
            teacher_checkpoint = str(_config_value(run_meta, "teacher_checkpoint", "task.teacher_checkpoint") or "")
            if teacher_checkpoint and f"teacher{teacher_seed}" not in teacher_checkpoint:
                continue

        artifact_mtime, artifact_datetime_utc = artifact_timestamp_for_run(run_cache_dir)
        records.append(
            RunRecord(
                run_id=f"wandb:{run_path}",
                method=method,
                teacher_law=teacher_law,
                teacher_signal="mc",
                m=run_m,
                subset_size=run_n,
                eta=run_eta,
                seed=run_seed,
                source_root=cache_dir,
                source_kind="wandb",
                out_dir=run_cache_dir,
                eval_history_path=run_cache_dir / "eval_history.jsonl",
                last_eval_path=run_cache_dir / "last_eval.json",
                run_meta=run_meta,
                completed=is_completed_run(run_cache_dir, run_cache_dir / "last_eval.json"),
                source_order=-10,
                reverse_variant="wandb_explicit",
                selection_rank=-200,
                selection_reason="explicit_wandb_run_path",
                artifact_mtime=artifact_mtime,
                artifact_datetime_utc=artifact_datetime_utc,
            )
        )

    return records


def keep_only_explicit_wandb_for_configured_nail_reverse_etas(
    runs_df: pd.DataFrame,
    run_paths_by_eta: dict[float, object],
    *,
    method: str = "NAIL-R, greedy rollout",
) -> pd.DataFrame:
    configured_etas = explicit_wandb_etas(run_paths_by_eta)
    if runs_df.empty or not configured_etas:
        return runs_df.copy()

    available_explicit_etas = set(
        runs_df.loc[
            runs_df["method"].eq(method) & runs_df["reverse_variant"].eq("wandb_explicit"),
            "eta",
        ].astype(float)
    )
    configured_etas = configured_etas.intersection(available_explicit_etas)
    if not configured_etas:
        return runs_df.copy()

    keep_mask = ~(
        runs_df["method"].eq(method)
        & runs_df["eta"].astype(float).isin(configured_etas)
        & ~runs_df["reverse_variant"].eq("wandb_explicit")
    )
    return runs_df.loc[keep_mask].copy().reset_index(drop=True)


def build_runs_df(records: list[RunRecord]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    run_data: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for record in records:
        last_eval = load_json(record.last_eval_path) if record.last_eval_path is not None else None
        history_df, history_meta = normalize_history_df(
            pd.read_json(record.eval_history_path, lines=True),
            last_eval=last_eval,
        )
        history_df["run_id"] = record.run_id
        history_df["method"] = record.method
        history_df["eta"] = record.eta
        history_df["seed"] = record.seed
        history_df["source_kind"] = record.source_kind
        effective_batch_size = effective_batch_size_from_meta(record.run_meta)
        history_df["effective_batch_size"] = effective_batch_size
        history_df["expert_trajectories"] = history_df["iter"] * effective_batch_size
        run_data[record.run_id] = history_df

        if last_eval is not None:
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
                "source_order": record.source_order,
                "reverse_variant": record.reverse_variant,
                "selection_rank": record.selection_rank,
                "selection_reason": record.selection_reason,
                "artifact_mtime": record.artifact_mtime,
                "artifact_datetime_utc": record.artifact_datetime_utc,
                "source_root": str(record.source_root),
                "completed": record.completed,
                "effective_batch_size": effective_batch_size,
                "final_iter": final_iter,
                "final_expert_trajectories": int(final_iter) * effective_batch_size,
                "final_clean_full_exact": final_full,
                "final_clean_final_exact": final_final,
                **history_meta,
                "out_dir": str(record.out_dir),
                "eval_history_path": str(record.eval_history_path),
            }
        )

    runs_df = pd.DataFrame(rows)
    if runs_df.empty:
        return runs_df, run_data

    method_order = {method: index for index, method in enumerate(METHOD_COLORS)}
    runs_df["method_order"] = runs_df["method"].map(method_order)
    runs_df = runs_df.sort_values(
        [
            "eta",
            "method_order",
            "seed",
            "selection_rank",
            "completed",
            "artifact_mtime",
            "source_order",
            "final_iter",
        ],
        ascending=[True, True, True, True, False, False, True, False],
    ).reset_index(drop=True)
    return runs_df, run_data


def dedupe_preferred_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return runs_df
    ordered = runs_df.sort_values(
        [
            "eta",
            "method",
            "seed",
            "selection_rank",
            "completed",
            "artifact_mtime",
            "source_order",
            "final_iter",
        ],
        ascending=[True, True, True, True, False, False, True, False],
    )
    preferred = ordered.drop_duplicates(subset=["eta", "method", "seed"], keep="first").copy()
    return preferred.sort_values(["eta", "method_order", "seed"]).reset_index(drop=True)


def default_plot_runs_df(
    runs_df: pd.DataFrame,
    *,
    nail_reverse_min_artifact_utc: str | pd.Timestamp | None = DEFAULT_NAIL_REVERSE_MIN_ARTIFACT_UTC,
) -> pd.DataFrame:
    if runs_df.empty:
        return runs_df.copy()
    if nail_reverse_min_artifact_utc is None:
        return runs_df.copy()

    cutoff = pd.Timestamp(nail_reverse_min_artifact_utc)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    artifact_times = pd.to_datetime(runs_df["artifact_datetime_utc"], utc=True, errors="coerce")
    keep_mask = ~(
        runs_df["method"].eq("NAIL-R, greedy rollout")
        & ~runs_df["reverse_variant"].isin(TRUSTED_NAIL_REVERSE_VARIANTS)
        & artifact_times.lt(cutoff)
    )
    return runs_df.loc[keep_mask].copy().reset_index(drop=True)


def selected_plot_runs_df(
    runs_df: pd.DataFrame,
    *,
    nail_reverse_min_artifact_utc: str | pd.Timestamp | None = DEFAULT_NAIL_REVERSE_MIN_ARTIFACT_UTC,
    require_trusted_nail_reverse: bool = False,
) -> pd.DataFrame:
    filtered = default_plot_runs_df(
        runs_df,
        nail_reverse_min_artifact_utc=nail_reverse_min_artifact_utc,
    )
    if require_trusted_nail_reverse and not filtered.empty:
        trusted_mask = filtered["reverse_variant"].isin(TRUSTED_NAIL_REVERSE_VARIANTS)
        filtered = filtered.loc[
            ~filtered["method"].eq("NAIL-R, greedy rollout") | trusted_mask
        ].copy()
    return dedupe_preferred_runs(filtered)


def missing_method_seed_rows(
    runs_df: pd.DataFrame,
    *,
    method: str,
    seeds: list[int] | dict[float, list[int]],
    etas: list[float],
) -> pd.DataFrame:
    if isinstance(seeds, dict):
        expected_rows_for_method = [
            {"eta": eta, "seed": seed, "method": method}
            for eta in etas
            for seed in seeds.get(float(eta), [])
        ]
    else:
        expected_rows_for_method = [
            {"eta": eta, "seed": seed, "method": method} for eta in etas for seed in seeds
        ]
    expected = pd.DataFrame(
        expected_rows_for_method,
        columns=["eta", "seed", "method"],
    )
    if runs_df.empty:
        expected["missing"] = True
        return expected

    observed = runs_df[["eta", "seed", "method"]].drop_duplicates()
    merged = expected.merge(observed, on=["eta", "seed", "method"], how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"].drop(columns="_merge").copy()
    missing["missing"] = True
    return missing.reset_index(drop=True)


def require_method_seed_coverage(
    runs_df: pd.DataFrame,
    *,
    method: str,
    seeds: list[int] | dict[float, list[int]],
    etas: list[float],
) -> None:
    missing = missing_method_seed_rows(runs_df, method=method, seeds=seeds, etas=etas)
    if missing.empty:
        return
    missing_text = ", ".join(
        f"eta={row.eta}, seed={int(row.seed)}" for row in missing.itertuples(index=False)
    )
    raise RuntimeError(
        f"Missing selected {method} runs for: {missing_text}. "
        "Add explicit W&B run paths or sync the missing output directories before plotting."
    )


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

    preferred = dedupe_preferred_runs(runs_df)
    summary = (
        preferred.groupby(["eta", "seed", "method"], as_index=False)
        .agg(
            run_id=("run_id", "first"),
            source_kind=("source_kind", "first"),
            reverse_variant=("reverse_variant", "first"),
            selection_reason=("selection_reason", "first"),
            artifact_datetime_utc=("artifact_datetime_utc", "first"),
            completed=("completed", "first"),
            final_iter=("final_iter", "first"),
            final_clean_full_exact=("final_clean_full_exact", "first"),
            final_clean_final_exact=("final_clean_final_exact", "first"),
        )
    )

    preferred_run_ids = preferred["run_id"].tolist()
    if preferred_run_ids:
        curve_summary = (
            pd.concat([run_data[run_id] for run_id in preferred_run_ids], ignore_index=True)
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


def summarize_method_curves(
    method_curves: list[pd.DataFrame],
    *,
    require_full_seed_support: bool = True,
) -> pd.DataFrame:
    if not method_curves:
        return pd.DataFrame(columns=["iter", "mean", "std", "n_seeds"])
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
    if require_full_seed_support:
        expected_seed_count = combined["seed"].nunique()
        summary = summary[summary["n_seeds"] == expected_seed_count].copy()
    return summary.reset_index(drop=True)


def plot_per_eta(
    run_data: dict[str, pd.DataFrame],
    runs_df: pd.DataFrame,
    *,
    metric: str,
    out_dir: Path | None = None,
    show: bool = True,
) -> None:
    set_publication_style()
    import matplotlib.pyplot as plt

    if runs_df.empty:
        return

    preferred = dedupe_preferred_runs(runs_df)
    metric_name = metric.split("/")[-1]

    for eta in sorted(preferred["eta"].unique()):
        eta_rows = preferred[preferred["eta"] == eta]
        fig, ax = plt.subplots(figsize=(14.5, 6.0), constrained_layout=False)
        fig.subplots_adjust(left=0.075, right=0.79, bottom=0.14, top=0.88)

        plotted_methods: list[tuple[str, pd.DataFrame, object, int]] = []

        for method in METHOD_COLORS:
            method_rows = eta_rows[eta_rows["method"] == method].copy()
            if method_rows.empty:
                continue

            method_curves: list[pd.DataFrame] = []
            for _, row in method_rows.iterrows():
                df = run_data[row["run_id"]].sort_values("iter").copy()
                if "expert_trajectories" not in df.columns:
                    effective_batch_size = int(row.get("effective_batch_size", DEFAULT_EFFECTIVE_BATCH_SIZE))
                    df["expert_trajectories"] = df["iter"] * effective_batch_size
                df = df[["expert_trajectories", metric]].rename(
                    columns={"expert_trajectories": "iter", metric: "metric"}
                )
                df["seed"] = int(row["seed"])
                method_curves.append(df)

            if not method_curves:
                continue

            summary = summarize_method_curves(method_curves)
            if summary.empty:
                continue

            style = get_method_style(method)
            seed_count = int(summary["n_seeds"].max())
            plotted_methods.append((method, summary, style, seed_count))

        seed_counts = {seed_count for _, _, _, seed_count in plotted_methods}
        show_seed_counts = len(seed_counts) > 1

        for method, summary, style, seed_count in plotted_methods:
            label = PLOT_LEGEND_LABELS.get(method, style.label)
            if show_seed_counts:
                label = f"{label} (n={seed_count})"
            ax.plot(
                summary["iter"],
                summary["mean"],
                color=style.color,
                linestyle=style.linestyle,
                linewidth=style.linewidth,
                label=label,
            )
            ax.fill_between(
                summary["iter"],
                summary["mean"] - summary["std"],
                summary["mean"] + summary["std"],
                color=style.color,
                alpha=0.18,
                linewidth=0,
            )

        ax.set_title(f"S5 m={int(preferred['m'].iloc[0])}, eta={eta:.2f}")
        ax.set_xlabel("# Expert Trajectories")
        ax.set_ylabel(metric_display_label(metric))
        ax.set_ylim(0.0, 1.01)
        ax.set_xlim(left=0)
        apply_iteration_axis(ax, nbins=6)
        polish_axes(ax)
        if plotted_methods:
            ax.legend(
                loc="upper left",
                # bbox_to_anchor=(1.01, 1.0), add these to RC params
                borderaxespad=0.0,
                fontsize=12,
                handlelength=2.2,
                handletextpad=0.55,
                labelspacing=0.45,
                borderpad=0.45,
            )

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"eta{eta_tag(float(eta))}_{metric_name}_online_seed_sweeps.png"
            save_publication_figure(fig, out_path)
            print(f"Saved {out_path} and {out_path.with_suffix('.pdf')}")

        if show:
            plt.show()
        plt.close(fig)
