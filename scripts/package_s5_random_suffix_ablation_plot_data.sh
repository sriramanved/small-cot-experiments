#!/usr/bin/env bash
set -euo pipefail

# Package lightweight plotting artifacts for the S5 random-suffix ablations.
#
# Usage from repo root on the dev node:
#   scripts/package_s5_random_suffix_ablation_plot_data.sh
#
# Optional knobs:
#   RUN_ROOT=reruns/s5_random_suffix
#   TRANSFER_DIR=transfer/s5_random_suffix_ablation_plot_data
#   ARCHIVE=transfer/s5_random_suffix_ablation_plot_data.tar.gz

RUN_ROOT="${RUN_ROOT:-reruns/s5_random_suffix}"
TRANSFER_DIR="${TRANSFER_DIR:-transfer/s5_random_suffix_ablation_plot_data}"
ARCHIVE="${ARCHIVE:-transfer/s5_random_suffix_ablation_plot_data.tar.gz}"

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "missing run root: $RUN_ROOT" >&2
  exit 1
fi

rm -rf "$TRANSFER_DIR"
mkdir -p "$TRANSFER_DIR/logs"

RUN_ROOT="$RUN_ROOT" TRANSFER_DIR="$TRANSFER_DIR" python - <<'PY'
import os
import re
import shutil
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
transfer_root = Path(os.environ["TRANSFER_DIR"])
files = [
    "eval_history.jsonl",
    "last_eval.json",
    "run_meta.json",
    "launcher_command.txt",
    "launcher_config.json",
    "completed.txt",
    "wandb_state.json",
]

groups: dict[str, list[Path]] = {}
for meta in run_root.rglob("run_meta.json"):
    run_dir = meta.parent
    rel = run_dir.relative_to(run_root)
    key = re.sub(r"_rerun\d+$", "", str(rel))
    groups.setdefault(key, []).append(run_dir)

def rerun_num(path: Path) -> int:
    match = re.search(r"_rerun(\d+)$", path.name)
    return int(match.group(1)) if match else 0

def score(path: Path) -> tuple[bool, bool, int]:
    return (
        (path / "completed.txt").exists(),
        (path / "eval_history.jsonl").exists(),
        rerun_num(path),
    )

for key, run_dirs in sorted(groups.items()):
    best = sorted(run_dirs, key=score)[-1]
    dest = transfer_root / "reruns" / "s5_random_suffix" / key
    dest.mkdir(parents=True, exist_ok=True)
    for filename in files:
        src = best / filename
        if src.exists():
            shutil.copy2(src, dest / filename)
    print(f"{key} <- {best}")
PY

if [[ -d logs ]]; then
  while IFS= read -r log; do
    cp "$log" "$TRANSFER_DIR/logs/"
  done < <(find logs -maxdepth 1 -type f | grep 's5_rsuffix_' || true)
fi

echo "eval histories:"
find "$TRANSFER_DIR/reruns/s5_random_suffix" -name eval_history.jsonl | wc -l
echo "completed markers:"
find "$TRANSFER_DIR/reruns/s5_random_suffix" -name completed.txt | wc -l
echo "run metadata files:"
find "$TRANSFER_DIR/reruns/s5_random_suffix" -name run_meta.json | wc -l

mkdir -p "$(dirname "$ARCHIVE")"
tar -czf "$ARCHIVE" -C "$(dirname "$TRANSFER_DIR")" "$(basename "$TRANSFER_DIR")"
ls -lh "$ARCHIVE"
