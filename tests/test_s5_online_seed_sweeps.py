from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.plot_s5_online_seed_sweeps import (
    build_runs_df,
    coverage_table,
    default_plot_runs_df,
    dedupe_preferred_runs,
    discover_s5_online_runs,
    discover_wandb_s5_online_runs,
    effective_batch_size_from_meta,
    expected_seeds_by_eta_for_wandb_overrides,
    keep_only_explicit_wandb_for_configured_nail_reverse_etas,
    iter_run_dirs,
    missing_method_seed_rows,
    normalize_history_df,
    require_method_seed_coverage,
    selected_plot_runs_df,
    summarize_method_curves,
)


def _write_run(
    out_dir: Path,
    *,
    teacher_seed: int,
    run_meta: dict[str, object] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_history.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"iter": 0, "val/clean_full_exact": 0.1, "val/clean_final_exact": 0.2}) + "\n")
        f.write(json.dumps({"iter": 10, "val/clean_full_exact": 0.3, "val/clean_final_exact": 0.4}) + "\n")
    with open(out_dir / "last_eval.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "iter": 10,
                "reason": "final",
                "val/clean_full_exact": 0.3,
                "val/clean_final_exact": 0.4,
            },
            f,
            indent=2,
        )
    with open(out_dir / "completed.txt", "w", encoding="utf-8") as f:
        f.write("iter_num=10\n")

    payload = {
        "task": "s5",
        "method_family": "nail",
        "loss": "reverse",
        "teacher_signal": "mc",
        "teacher_checkpoint": f"reruns/s5_m21_teacher{teacher_seed}/out-s5-cot-m21-depth1-seed{teacher_seed}",
        "resolved_rollout_temperature": 0.0,
    }
    if run_meta is not None:
        payload.update(run_meta)
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class _FakeWandbRun:
    def __init__(
        self,
        *,
        run_path: str,
        name: str,
        config: dict[str, object],
        rows: list[dict[str, object]],
    ) -> None:
        self.id = run_path.rsplit("/", 1)[-1]
        self.path = run_path.split("/")
        self.name = name
        self.config = config
        self.state = "finished"
        self.url = f"https://wandb.ai/{run_path}"
        self._rows = rows

    def scan_history(self, keys=None, page_size: int = 1000):
        del page_size
        if keys is None:
            return iter(self._rows)
        return (
            {key: row[key] for key in keys if key in row}
            for row in self._rows
            if all(key in row for key in keys)
        )


class _FakeWandbApi:
    def __init__(self, runs: dict[str, _FakeWandbRun]) -> None:
        self.runs = runs

    def run(self, run_path: str) -> _FakeWandbRun:
        return self.runs[run_path]


class S5OnlineSeedSweepTests(unittest.TestCase):
    def test_normalize_history_df_keeps_latest_rerun_segment_from_reused_out_dir(self):
        raw_history = [
            {"iter": 0, "reason": "periodic", "val/clean_full_exact": 0.10, "val/clean_final_exact": 0.20},
            {"iter": 100, "reason": "periodic", "val/clean_full_exact": 0.60, "val/clean_final_exact": 0.70},
            {"iter": 200, "reason": "final", "val/clean_full_exact": 0.90, "val/clean_final_exact": 0.95},
            {"iter": 0, "reason": "periodic", "val/clean_full_exact": 0.05, "val/clean_final_exact": 0.10},
            {"iter": 100, "reason": "periodic", "val/clean_full_exact": 0.40, "val/clean_final_exact": 0.50},
            {"iter": 150, "reason": "final", "val/clean_full_exact": 0.50, "val/clean_final_exact": 0.60},
        ]

        history_df, history_meta = normalize_history_df(
            pd.DataFrame(raw_history),
            last_eval={
                "iter": 150,
                "reason": "final",
                "val/clean_full_exact": 0.50,
                "val/clean_final_exact": 0.60,
            },
        )

        self.assertEqual(history_meta["history_segment_count"], 2)
        self.assertEqual(history_meta["history_selected_segment"], 1)
        self.assertEqual(history_meta["history_restart_count"], 1)
        self.assertEqual(history_df["iter"].tolist(), [0, 100, 150])
        self.assertEqual(history_df["val/clean_full_exact"].tolist(), [0.05, 0.40, 0.50])

    def test_normalize_history_df_truncates_trailing_rows_after_last_eval_within_segment(self):
        raw_history = [
            {"iter": 0, "reason": "periodic", "val/clean_full_exact": 0.10, "val/clean_final_exact": 0.20},
            {"iter": 100, "reason": "periodic", "val/clean_full_exact": 0.70, "val/clean_final_exact": 0.80},
            {"iter": 125, "reason": "final", "val/clean_full_exact": 1.00, "val/clean_final_exact": 1.00},
            {"iter": 150, "reason": "periodic", "val/clean_full_exact": 0.60, "val/clean_final_exact": 0.65},
            {"iter": 175, "reason": "periodic", "val/clean_full_exact": 0.20, "val/clean_final_exact": 0.25},
        ]

        history_df, history_meta = normalize_history_df(
            pd.DataFrame(raw_history),
            last_eval={
                "iter": 125,
                "reason": "final",
                "val/clean_full_exact": 1.00,
                "val/clean_final_exact": 1.00,
            },
        )

        self.assertEqual(history_meta["history_segment_count"], 1)
        self.assertEqual(history_meta["history_selected_segment"], 0)
        self.assertEqual(history_meta["history_restart_count"], 0)
        self.assertEqual(history_df["iter"].tolist(), [0, 100, 125])
        self.assertEqual(history_df["val/clean_full_exact"].tolist(), [0.10, 0.70, 1.00])

    def test_prefers_fixed_nail_reverse_run_over_legacy_duplicate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            imports_root = root / "analysis" / "imports" / "s5_online_seed_sweeps" / "aics"
            cache_root = root / "analysis" / "cache" / "s5_online_seed_sweeps" / "dev_node"
            run_name = "out-s5-nail-reverse-mc-m21-n8000000-eta0p0-distributional_noise-seed20260417"

            _write_run(imports_root / "legacy" / run_name, teacher_seed=20260417)
            _write_run(
                cache_root / "fixed" / run_name,
                teacher_seed=20260417,
                run_meta={
                    "target_len": 147,
                    "answer_len": 7,
                    "target_span": "cot_with_final_answer_suffix",
                    "reverse_action_source": "student_aux_sample",
                },
            )

            records = discover_s5_online_runs(
                [imports_root, cache_root],
                m=21,
                seeds=[20260417],
                etas=[0.0],
                teacher_seed=20260417,
                override_path=root / "missing_overrides.json",
            )
            runs_df, _ = build_runs_df(records)
            preferred = dedupe_preferred_runs(runs_df)

            self.assertEqual(len(preferred), 1)
            self.assertEqual(preferred.iloc[0]["reverse_variant"], "fixed_aux_actions")
            self.assertEqual(
                preferred.iloc[0]["selection_reason"],
                "run_meta.reverse_action_source=student_aux_sample",
            )
            self.assertIn("/fixed/", preferred.iloc[0]["run_id"])

    def test_iter_run_dirs_does_not_descend_inside_run_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outer = root / "nested" / "out-s5-nail-reverse-mc-m21-n8000000-eta0p0-distributional_noise-seed20260417"
            inner = outer / "checkpoint_0000010" / "out-s5-should-not-be-seen"
            inner.mkdir(parents=True)

            run_dirs = iter_run_dirs(root)

            self.assertEqual(run_dirs, [outer])

    def test_iter_run_dirs_accepts_exact_run_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "out-s5-nail-reverse-mc-m21-n8000000-eta0p0-distributional_noise-seed20260417"
            nested = run_dir / "checkpoint_0000010" / "out-s5-should-not-be-seen"
            nested.mkdir(parents=True)

            run_dirs = iter_run_dirs(run_dir)

            self.assertEqual(run_dirs, [run_dir])

    def test_discover_infers_source_kind_from_nested_search_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            family_root = (
                root
                / "analysis"
                / "imports"
                / "s5_online_seed_sweeps"
                / "aics"
                / "reruns"
                / "s5_m21_teacher20260417_render1337_train20260418"
            )
            run_dir = (
                family_root
                / "nail_reverse_mc"
                / "out-s5-nail-reverse-mc-m21-n12000000-eta0p7-distributional_noise-seed20260418"
            )
            _write_run(run_dir, teacher_seed=20260417)

            records = discover_s5_online_runs(
                [family_root],
                m=21,
                seeds=[20260418],
                etas=[0.7],
                teacher_seed=20260417,
                override_path=root / "missing_overrides.json",
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].source_kind, "aics")

    def test_sampled_rollout_suffix_is_used_without_run_meta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = (
                root
                / "analysis"
                / "imports"
                / "s5_online_seed_sweeps"
                / "aics"
                / "reruns"
                / "s5_m21_teacher20260417_render1337_train20260418"
                / "nail_forward_mc"
                / "out-s5-nail-forward-mc-m21-n8000000-eta0p1-distributional_noise-rollt1p0-seed20260418"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "eval_history.jsonl", "w", encoding="utf-8") as f:
                f.write(json.dumps({"iter": 0, "val/clean_full_exact": 0.1, "val/clean_final_exact": 0.2}) + "\n")
                f.write(json.dumps({"iter": 10, "val/clean_full_exact": 0.3, "val/clean_final_exact": 0.4}) + "\n")
            with open(run_dir / "last_eval.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "iter": 10,
                        "reason": "final",
                        "val/clean_full_exact": 0.3,
                        "val/clean_final_exact": 0.4,
                    },
                    f,
                    indent=2,
                )

            records = discover_s5_online_runs(
                [run_dir],
                m=21,
                seeds=[20260418],
                etas=[0.1],
                teacher_seed=20260417,
                override_path=root / "missing_overrides.json",
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].method, "OPD-F")

    def test_prefers_newer_unknown_nail_reverse_run_when_dates_differ(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            imports_root = root / "analysis" / "imports" / "s5_online_seed_sweeps" / "aics"
            cache_root = root / "analysis" / "cache" / "s5_online_seed_sweeps" / "dev_node"
            run_name = "out-s5-nail-reverse-mc-m21-n8000000-eta0p0-distributional_noise-seed20260418"

            older_dir = imports_root / "older" / run_name
            newer_dir = cache_root / "newer" / run_name
            _write_run(older_dir, teacher_seed=20260417)
            _write_run(newer_dir, teacher_seed=20260417)

            old_ts = 1777075200  # 2026-04-25T00:00:00Z
            new_ts = 1777248000  # 2026-04-27T00:00:00Z
            for path in (
                older_dir / "eval_history.jsonl",
                older_dir / "last_eval.json",
                older_dir / "run_meta.json",
                older_dir / "completed.txt",
            ):
                os.utime(path, (old_ts, old_ts))
            for path in (
                newer_dir / "eval_history.jsonl",
                newer_dir / "last_eval.json",
                newer_dir / "run_meta.json",
                newer_dir / "completed.txt",
            ):
                os.utime(path, (new_ts, new_ts))

            records = discover_s5_online_runs(
                [imports_root, cache_root],
                m=21,
                seeds=[20260418],
                etas=[0.0],
                teacher_seed=20260417,
                override_path=root / "missing_overrides.json",
            )
            runs_df, _ = build_runs_df(records)
            preferred = dedupe_preferred_runs(runs_df)

            self.assertEqual(len(preferred), 1)
            self.assertEqual(preferred.iloc[0]["reverse_variant"], "unknown_or_legacy")
            self.assertIn("/newer/", preferred.iloc[0]["run_id"])
            self.assertIn("2026-04-27", preferred.iloc[0]["artifact_datetime_utc"])

    def test_default_plot_runs_df_filters_old_nail_reverse_but_keeps_manual_override(self):
        runs_df = pd.DataFrame(
            [
                {
                    "run_id": "old-reverse",
                    "method": "NAIL-R, greedy rollout",
                    "reverse_variant": "unknown_or_legacy",
                    "artifact_datetime_utc": "2026-04-25T00:00:00+00:00",
                },
                {
                    "run_id": "manual-old-reverse",
                    "method": "NAIL-R, greedy rollout",
                    "reverse_variant": "manual_preferred",
                    "artifact_datetime_utc": "2026-04-25T00:00:00+00:00",
                },
                {
                    "run_id": "new-reverse",
                    "method": "NAIL-R, greedy rollout",
                    "reverse_variant": "fixed_aux_actions",
                    "artifact_datetime_utc": "2026-04-27T00:00:00+00:00",
                },
                {
                    "run_id": "forward",
                    "method": "NAIL-F, greedy rollout",
                    "reverse_variant": "standard",
                    "artifact_datetime_utc": "2026-04-01T00:00:00+00:00",
                },
            ]
        )

        filtered = default_plot_runs_df(runs_df)

        self.assertEqual(
            filtered["run_id"].tolist(),
            ["manual-old-reverse", "new-reverse", "forward"],
        )

    def test_selected_plot_runs_df_filters_then_dedupes(self):
        runs_df = pd.DataFrame(
            [
                {
                    "run_id": "fixed-new",
                    "method": "NAIL-R, greedy rollout",
                    "eta": 0.0,
                    "seed": 20260417,
                    "selection_rank": 1,
                    "completed": True,
                    "artifact_mtime": 20.0,
                    "source_order": 0,
                    "final_iter": 125000,
                    "method_order": 2,
                    "reverse_variant": "fixed_aux_actions",
                    "artifact_datetime_utc": "2026-04-26T00:00:00+00:00",
                },
                {
                    "run_id": "legacy-old",
                    "method": "NAIL-R, greedy rollout",
                    "eta": 0.0,
                    "seed": 20260417,
                    "selection_rank": 30,
                    "completed": True,
                    "artifact_mtime": 10.0,
                    "source_order": 1,
                    "final_iter": 125000,
                    "method_order": 2,
                    "reverse_variant": "unknown_or_legacy",
                    "artifact_datetime_utc": "2026-04-22T00:00:00+00:00",
                },
            ]
        )

        selected = selected_plot_runs_df(runs_df)

        self.assertEqual(selected["run_id"].tolist(), ["fixed-new"])

    def test_explicit_wandb_run_path_is_cached_and_preferred_over_local_duplicate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            imports_root = root / "analysis" / "imports" / "s5_online_seed_sweeps" / "aics"
            run_name = "out-s5-nail-reverse-mc-m21-n8000000-eta0p0-distributional_noise-seed20260417"
            _write_run(imports_root / "legacy" / run_name, teacher_seed=20260417)

            run_path = "entity/project/new-run-id"
            api = _FakeWandbApi(
                {
                    run_path: _FakeWandbRun(
                        run_path=run_path,
                        name=run_name,
                        config={
                            "task": {
                                "m": 21,
                                "subset_size": 8000000,
                                "eta": 0.0,
                                "teacher_checkpoint": "reruns/s5_m21_teacher20260417/out-s5-cot-m21-depth1-seed20260417",
                            },
                            "seed": 20260417,
                        },
                        rows=[
                            {
                                "iter": 0,
                                "val/clean_full_exact": 0.2,
                                "val/clean_final_exact": 0.3,
                                "val/loss": 3.0,
                            },
                            {
                                "iter": 100,
                                "val/clean_full_exact": 1.0,
                                "val/clean_final_exact": 1.0,
                                "val/loss": 0.1,
                            },
                        ],
                    )
                }
            )

            records = discover_s5_online_runs(
                [imports_root],
                m=21,
                seeds=[20260417],
                etas=[0.0],
                teacher_seed=20260417,
                override_path=root / "missing_overrides.json",
            )
            records.extend(
                discover_wandb_s5_online_runs(
                    {0.0: {20260417: run_path}},
                    m=21,
                    seeds=[20260417],
                    etas=[0.0],
                    teacher_seed=20260417,
                    cache_dir=root / "wandb-cache",
                    api=api,
                    refresh=True,
                )
            )
            runs_df, run_data = build_runs_df(records)
            selected = selected_plot_runs_df(runs_df)

            self.assertEqual(selected["run_id"].tolist(), [f"wandb:{run_path}"])
            self.assertEqual(selected.iloc[0]["reverse_variant"], "wandb_explicit")
            self.assertEqual(float(selected.iloc[0]["final_clean_full_exact"]), 1.0)
            self.assertEqual(run_data[f"wandb:{run_path}"]["iter"].tolist(), [0, 100])

    def test_explicit_wandb_run_path_can_skip_missing_cache_without_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = discover_wandb_s5_online_runs(
                {0.0: {20260417: "entity/project/missing-run-id"}},
                m=21,
                seeds=[20260417],
                etas=[0.0],
                teacher_seed=20260417,
                cache_dir=Path(tmpdir) / "wandb-cache",
                allow_api=False,
                skip_missing_cache=True,
            )

            self.assertEqual(records, [])

    def test_selected_plot_runs_df_keeps_unknown_nail_reverse_rows_for_complete_seed_set(self):
        runs_df = pd.DataFrame(
            [
                {
                    "run_id": "trusted-seed-a",
                    "method": "NAIL-R, greedy rollout",
                    "eta": 0.7,
                    "seed": 20260417,
                    "selection_rank": -100,
                    "completed": True,
                    "artifact_mtime": 30.0,
                    "source_order": 0,
                    "final_iter": 187500,
                    "method_order": 2,
                    "reverse_variant": "manual_preferred",
                    "artifact_datetime_utc": "2026-04-27T00:00:00+00:00",
                },
                {
                    "run_id": "trusted-seed-b",
                    "method": "NAIL-R, greedy rollout",
                    "eta": 0.7,
                    "seed": 20260418,
                    "selection_rank": -100,
                    "completed": True,
                    "artifact_mtime": 29.0,
                    "source_order": 0,
                    "final_iter": 187500,
                    "method_order": 2,
                    "reverse_variant": "manual_preferred",
                    "artifact_datetime_utc": "2026-04-27T00:00:00+00:00",
                },
                {
                    "run_id": "unknown-third-seed",
                    "method": "NAIL-R, greedy rollout",
                    "eta": 0.7,
                    "seed": 20260419,
                    "selection_rank": 30,
                    "completed": True,
                    "artifact_mtime": 28.0,
                    "source_order": 1,
                    "final_iter": 187500,
                    "method_order": 2,
                    "reverse_variant": "unknown_or_legacy",
                    "artifact_datetime_utc": "2026-04-27T00:00:00+00:00",
                },
            ]
        )

        selected = selected_plot_runs_df(runs_df)

        self.assertEqual(
            selected["run_id"].tolist(),
            ["trusted-seed-a", "trusted-seed-b", "unknown-third-seed"],
        )
        self.assertEqual(selected["seed"].tolist(), [20260417, 20260418, 20260419])

    def test_explicit_wandb_eta_does_not_mix_local_nail_reverse_fallback(self):
        runs_df = pd.DataFrame(
            [
                {
                    "run_id": "wandb:eta07-seed-a",
                    "method": "NAIL-R, greedy rollout",
                    "eta": 0.7,
                    "seed": 20260417,
                    "selection_rank": -200,
                    "completed": True,
                    "artifact_mtime": 30.0,
                    "source_order": -10,
                    "final_iter": 187500,
                    "method_order": 2,
                    "reverse_variant": "wandb_explicit",
                    "artifact_datetime_utc": "2026-04-27T00:00:00+00:00",
                },
                {
                    "run_id": "wandb:eta07-seed-b",
                    "method": "NAIL-R, greedy rollout",
                    "eta": 0.7,
                    "seed": 20260418,
                    "selection_rank": -200,
                    "completed": True,
                    "artifact_mtime": 29.0,
                    "source_order": -10,
                    "final_iter": 187500,
                    "method_order": 2,
                    "reverse_variant": "wandb_explicit",
                    "artifact_datetime_utc": "2026-04-27T00:00:00+00:00",
                },
                {
                    "run_id": "local-third-seed",
                    "method": "NAIL-R, greedy rollout",
                    "eta": 0.7,
                    "seed": 20260419,
                    "selection_rank": -100,
                    "completed": True,
                    "artifact_mtime": 28.0,
                    "source_order": 0,
                    "final_iter": 187500,
                    "method_order": 2,
                    "reverse_variant": "manual_preferred",
                    "artifact_datetime_utc": "2026-04-27T00:00:00+00:00",
                },
            ]
        )
        run_paths_by_eta = {
            0.7: {
                20260417: "entity/project/eta07-seed-a",
                20260418: "entity/project/eta07-seed-b",
            }
        }

        filtered = keep_only_explicit_wandb_for_configured_nail_reverse_etas(
            runs_df,
            run_paths_by_eta,
        )
        expected_seeds = expected_seeds_by_eta_for_wandb_overrides(
            run_paths_by_eta,
            default_seeds=[20260417, 20260418, 20260419],
            etas=[0.7],
        )

        self.assertEqual(filtered["run_id"].tolist(), ["wandb:eta07-seed-a", "wandb:eta07-seed-b"])
        self.assertEqual(expected_seeds, {0.7: [20260417, 20260418]})
        require_method_seed_coverage(
            filtered,
            method="NAIL-R, greedy rollout",
            seeds=expected_seeds,
            etas=[0.7],
        )

    def test_configured_wandb_eta_keeps_local_fallback_when_cache_missing(self):
        runs_df = pd.DataFrame(
            [
                {
                    "run_id": "local-eta07-seed-a",
                    "method": "NAIL-R, greedy rollout",
                    "eta": 0.7,
                    "seed": 20260417,
                    "selection_rank": -100,
                    "completed": True,
                    "artifact_mtime": 30.0,
                    "source_order": 0,
                    "final_iter": 187500,
                    "method_order": 2,
                    "reverse_variant": "manual_preferred",
                    "artifact_datetime_utc": "2026-04-27T00:00:00+00:00",
                }
            ]
        )

        filtered = keep_only_explicit_wandb_for_configured_nail_reverse_etas(
            runs_df,
            {0.7: {20260417: "entity/project/missing-cache-run"}},
        )

        self.assertEqual(filtered["run_id"].tolist(), ["local-eta07-seed-a"])

    def test_require_method_seed_coverage_reports_missing_reverse_seed(self):
        runs_df = pd.DataFrame(
            [
                {"eta": 0.7, "seed": 20260417, "method": "NAIL-R, greedy rollout"},
                {"eta": 0.7, "seed": 20260418, "method": "NAIL-R, greedy rollout"},
            ]
        )

        missing = missing_method_seed_rows(
            runs_df,
            method="NAIL-R, greedy rollout",
            seeds=[20260417, 20260418, 20260419],
            etas=[0.7],
        )

        self.assertEqual(missing[["eta", "seed"]].to_dict("records"), [{"eta": 0.7, "seed": 20260419}])
        with self.assertRaisesRegex(RuntimeError, "eta=0.7, seed=20260419"):
            require_method_seed_coverage(
                runs_df,
                method="NAIL-R, greedy rollout",
                seeds=[20260417, 20260418, 20260419],
                etas=[0.7],
            )

    def test_coverage_table_uses_only_preferred_run_curve(self):
        runs_df = pd.DataFrame(
            [
                {
                    "run_id": "preferred",
                    "method": "NAIL-R, greedy rollout",
                    "eta": 0.0,
                    "seed": 20260418,
                    "selection_rank": 1,
                    "completed": True,
                    "artifact_mtime": 20.0,
                    "source_order": 0,
                    "final_iter": 125000,
                    "method_order": 2,
                    "source_kind": "local",
                    "reverse_variant": "fixed_aux_actions",
                    "selection_reason": "path_contains_nail_reverse_mc_fixed",
                    "artifact_datetime_utc": "2026-04-27T00:00:00+00:00",
                    "final_clean_full_exact": 1.0,
                    "final_clean_final_exact": 1.0,
                },
                {
                    "run_id": "duplicate-long",
                    "method": "NAIL-R, greedy rollout",
                    "eta": 0.0,
                    "seed": 20260418,
                    "selection_rank": 30,
                    "completed": True,
                    "artifact_mtime": 10.0,
                    "source_order": 1,
                    "final_iter": 187500,
                    "method_order": 2,
                    "source_kind": "aics",
                    "reverse_variant": "unknown_or_legacy",
                    "selection_reason": "no_fixed_nail_reverse_marker",
                    "artifact_datetime_utc": "2026-04-23T00:00:00+00:00",
                    "final_clean_full_exact": 0.0,
                    "final_clean_final_exact": 0.0,
                },
            ]
        )
        run_data = {
            "preferred": pd.DataFrame(
                {
                    "iter": [0, 125000],
                    "eta": [0.0, 0.0],
                    "seed": [20260418, 20260418],
                    "method": ["NAIL-R, greedy rollout", "NAIL-R, greedy rollout"],
                }
            ),
            "duplicate-long": pd.DataFrame(
                {
                    "iter": [0, 187500],
                    "eta": [0.0, 0.0],
                    "seed": [20260418, 20260418],
                    "method": ["NAIL-R, greedy rollout", "NAIL-R, greedy rollout"],
                }
            ),
        }

        coverage = coverage_table(run_data=run_data, runs_df=runs_df, seeds=[20260418], etas=[0.0])
        row = coverage.loc[
            (coverage["method"] == "NAIL-R, greedy rollout")
            & (coverage["seed"] == 20260418)
            & (coverage["eta"] == 0.0)
        ].iloc[0]

        self.assertEqual(int(row["n_points"]), 2)
        self.assertEqual(int(row["max_iter"]), 125000)

    def test_summarize_method_curves_truncates_to_common_seed_support(self):
        method_curves = [
            pd.DataFrame({"iter": [0, 100, 200], "metric": [0.1, 0.2, 0.3], "seed": [1, 1, 1]}),
            pd.DataFrame({"iter": [0, 100], "metric": [0.4, 0.5], "seed": [2, 2]}),
        ]

        summary = summarize_method_curves(method_curves)

        self.assertEqual(summary["iter"].tolist(), [0, 100])
        self.assertEqual(summary["n_seeds"].tolist(), [2, 2])
        self.assertAlmostEqual(float(summary["mean"].iloc[0]), 0.25)
        self.assertAlmostEqual(float(summary["mean"].iloc[1]), 0.35)
        self.assertAlmostEqual(float(summary["std"].iloc[0]), 0.15)
        self.assertAlmostEqual(float(summary["std"].iloc[1]), 0.15)

    def test_manual_override_can_pin_exact_duplicate_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            imports_root = root / "analysis" / "imports" / "s5_online_seed_sweeps" / "aics"
            cache_root = root / "analysis" / "cache" / "s5_online_seed_sweeps" / "dev_node"
            run_name = "out-s5-nail-reverse-mc-m21-n8000000-eta0p1-distributional_noise-seed20260418"

            legacy_dir = imports_root / "legacy" / run_name
            fixed_dir = cache_root / "fixed" / run_name
            _write_run(
                legacy_dir,
                teacher_seed=20260417,
                run_meta={
                    "target_len": 147,
                    "answer_len": 7,
                    "target_span": "cot_with_final_answer_suffix",
                },
            )
            _write_run(
                fixed_dir,
                teacher_seed=20260417,
                run_meta={
                    "target_len": 147,
                    "answer_len": 7,
                    "target_span": "cot_with_final_answer_suffix",
                },
            )

            override_path = root / "analysis" / "cache" / "s5_online_seed_sweeps" / "run_overrides.json"
            override_path.parent.mkdir(parents=True, exist_ok=True)
            with open(override_path, "w", encoding="utf-8") as f:
                json.dump({"preferred_run_ids": [str(legacy_dir)]}, f, indent=2)

            records = discover_s5_online_runs(
                [imports_root, cache_root],
                m=21,
                seeds=[20260418],
                etas=[0.1],
                teacher_seed=20260417,
                override_path=override_path,
            )
            runs_df, _ = build_runs_df(records)
            preferred = dedupe_preferred_runs(runs_df)

            self.assertEqual(len(preferred), 1)
            self.assertEqual(preferred.iloc[0]["reverse_variant"], "manual_preferred")
            self.assertEqual(preferred.iloc[0]["selection_reason"], "manual_preferred_run_ids")
            self.assertEqual(Path(preferred.iloc[0]["run_id"]).resolve(), legacy_dir.resolve())

    def test_reruns_relative_override_matches_synced_import_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            imports_root = root / "analysis" / "imports" / "s5_online_seed_sweeps" / "aics"
            run_dir = (
                imports_root
                / "reruns"
                / "s5_m21_teacher20260417_render1337_train20260418"
                / "nail_reverse_mc"
                / "out-s5-nail-reverse-mc-m21-n12000000-eta0p1-distributional_noise-seed20260418"
            )
            _write_run(
                run_dir,
                teacher_seed=20260417,
                run_meta={
                    "target_len": 147,
                    "answer_len": 7,
                    "target_span": "cot_with_final_answer_suffix",
                },
            )

            override_path = root / "analysis" / "cache" / "s5_online_seed_sweeps" / "run_overrides.json"
            override_path.parent.mkdir(parents=True, exist_ok=True)
            with open(override_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "preferred_run_ids": [
                            "reruns/s5_m21_teacher20260417_render1337_train20260418/nail_reverse_mc/out-s5-nail-reverse-mc-m21-n12000000-eta0p1-distributional_noise-seed20260418"
                        ]
                    },
                    f,
                    indent=2,
                )

            records = discover_s5_online_runs(
                [imports_root],
                m=21,
                seeds=[20260418],
                etas=[0.1],
                teacher_seed=20260417,
                override_path=override_path,
            )
            runs_df, _ = build_runs_df(records)
            preferred = dedupe_preferred_runs(runs_df)

            self.assertEqual(len(preferred), 1)
            self.assertEqual(preferred.iloc[0]["reverse_variant"], "manual_preferred")
            self.assertEqual(preferred.iloc[0]["selection_reason"], "manual_preferred_run_ids")

    def test_build_runs_df_trims_appended_history_to_last_eval_segment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "analysis" / "imports" / "s5_online_seed_sweeps" / "aics" / (
                "out-s5-nail-reverse-mc-m21-n8000000-eta0p0-distributional_noise-seed20260417"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "eval_history.jsonl", "w", encoding="utf-8") as f:
                for row in (
                    {"iter": 0, "reason": "periodic", "val/clean_full_exact": 0.10, "val/clean_final_exact": 0.20},
                    {"iter": 200, "reason": "final", "val/clean_full_exact": 0.90, "val/clean_final_exact": 0.95},
                    {"iter": 0, "reason": "periodic", "val/clean_full_exact": 0.05, "val/clean_final_exact": 0.10},
                    {"iter": 150, "reason": "final", "val/clean_full_exact": 0.50, "val/clean_final_exact": 0.60},
                ):
                    f.write(json.dumps(row) + "\n")
            with open(run_dir / "last_eval.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "iter": 150,
                        "reason": "final",
                        "val/clean_full_exact": 0.50,
                        "val/clean_final_exact": 0.60,
                    },
                    f,
                    indent=2,
                )
            with open(run_dir / "completed.txt", "w", encoding="utf-8") as f:
                f.write("iter_num=150\n")
            with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "task": "s5",
                        "method_family": "nail",
                        "loss": "reverse",
                        "teacher_signal": "mc",
                        "teacher_checkpoint": "reruns/s5_m21_teacher20260417/out-s5-cot-m21-depth1-seed20260417",
                        "resolved_rollout_temperature": 0.0,
                        "reverse_action_source": "student_aux_sample",
                    },
                    f,
                    indent=2,
                )

            records = discover_s5_online_runs(
                [run_dir.parent],
                m=21,
                seeds=[20260417],
                etas=[0.0],
                teacher_seed=20260417,
                override_path=root / "missing_overrides.json",
            )
            runs_df, run_data = build_runs_df(records)

            self.assertEqual(int(runs_df.iloc[0]["history_restart_count"]), 1)
            self.assertEqual(int(runs_df.iloc[0]["history_segment_count"]), 2)
            self.assertEqual(int(runs_df.iloc[0]["history_selected_segment"]), 1)
            self.assertEqual(run_data[runs_df.iloc[0]["run_id"]]["iter"].tolist(), [0, 150])

    def test_offline_bc_prefers_sampled_dev_node_runs_for_eta_0p1_and_0p5(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dev_root = root / "analysis" / "cache" / "s5_online_seed_sweeps" / "dev_node"
            aics_root = root / "analysis" / "cache" / "s5_online_seed_sweeps" / "aics"

            for eta_tag in ("0p1", "0p5"):
                desired = (
                    dev_root
                    / "offline_bc"
                    / f"out-s5-noisy-bc-m21-n8000000-eta{eta_tag}-sample-seed20260417"
                )
                duplicate = (
                    aics_root
                    / "offline_bc"
                    / f"out-s5-noisy-bc-m21-n8000000-eta{eta_tag}-seed20260417"
                )
                _write_run(desired, teacher_seed=20260417)
                _write_run(duplicate, teacher_seed=20260417)

                old_ts = 1777075200
                new_ts = 1777248000
                for path in (desired / "eval_history.jsonl", desired / "last_eval.json", desired / "completed.txt"):
                    os.utime(path, (old_ts, old_ts))
                for path in (duplicate / "eval_history.jsonl", duplicate / "last_eval.json", duplicate / "completed.txt"):
                    os.utime(path, (new_ts, new_ts))

            records = discover_s5_online_runs(
                [dev_root, aics_root],
                m=21,
                seeds=[20260417],
                etas=[0.1, 0.5],
                teacher_seed=20260417,
                override_path=root / "missing_overrides.json",
            )
            runs_df, _ = build_runs_df(records)
            preferred = dedupe_preferred_runs(runs_df)

            offline_rows = preferred[preferred["method"] == "LogLossBC"].sort_values("eta")
            self.assertEqual(offline_rows["eta"].tolist(), [0.1, 0.5])
            self.assertTrue(all("-sample-" in run_id for run_id in offline_rows["run_id"]))
            self.assertTrue(all("dev_node" in run_id for run_id in offline_rows["run_id"]))
            self.assertEqual(
                offline_rows["selection_reason"].tolist(),
                ["offline_bc_sample_on_dev_node", "offline_bc_sample_on_dev_node"],
            )

    def test_effective_batch_size_uses_config_when_available_and_defaults_to_64(self):
        self.assertEqual(effective_batch_size_from_meta(None), 64)
        self.assertEqual(
            effective_batch_size_from_meta(
                {
                    "optim": {
                        "batch_size": 32,
                        "gradient_accumulation_steps": 2,
                    },
                    "runtime": {"world_size": 4},
                }
            ),
            256,
        )
        self.assertEqual(effective_batch_size_from_meta({"effective_batch_size": 128}), 128)


if __name__ == "__main__":
    unittest.main()
