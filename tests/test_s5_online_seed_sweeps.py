from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_s5_online_seed_sweeps import (
    build_runs_df,
    dedupe_preferred_runs,
    discover_s5_online_runs,
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


class S5OnlineSeedSweepTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
