#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

BASE_N="${BASE_N:-6000000}"
SUBSET_SIZES="${SUBSET_SIZES:-250000 500000 1000000 2000000 4000000 6000000}"
BASE_DATASET="${BASE_DATASET:-s5_clean_offline_n${BASE_N}}"
LOG_DIR="${LOG_DIR:-logs/clean_offline_sweep}"
mkdir -p "${LOG_DIR}"
BASE_DATASET_DIR="data/${BASE_DATASET}"

if [[ ! -f "${BASE_DATASET_DIR}/train_x.pt" ]]; then
  echo "Missing ${BASE_DATASET_DIR}/train_x.pt"
  echo "Run scripts/generate_clean_offline_sweep.sh first."
  exit 1
fi

export BASE_DATASET_DIR
export BASE_N
python - <<'PY'
import os
import torch
train_x = torch.load(os.path.join(os.environ["BASE_DATASET_DIR"], "train_x.pt"), map_location="cpu")
expected_n = int(os.environ["BASE_N"])
assert train_x.size(0) == expected_n, (
    f"base clean offline dataset has {train_x.size(0)} rows, expected {expected_n}. "
    "Regenerate it before training."
)
print(f"Verified base clean offline dataset has {expected_n} rows.")
PY

for SUBSET_SIZE in ${SUBSET_SIZES}; do
  if (( SUBSET_SIZE > BASE_N )); then
    echo "Skipping SUBSET_SIZE=${SUBSET_SIZE}; exceeds BASE_N=${BASE_N}"
    continue
  fi
  DATASET_NAME="${BASE_DATASET}"
  OUT_DIR="out-s5-clean-offline-bc-n${SUBSET_SIZE}"
  DONE_MARKER="${OUT_DIR}/completed.txt"
  LOG_PATH="${LOG_DIR}/s5_clean_offline_bc_n${SUBSET_SIZE}.log"
  EXTRA_ARGS=()

  if [[ -f "${DONE_MARKER}" ]]; then
    echo "Skipping ${DATASET_NAME}; found ${DONE_MARKER}"
    continue
  elif [[ -f "${OUT_DIR}/ckpt.pt" ]]; then
    echo "Resuming ${DATASET_NAME} from ${OUT_DIR}/ckpt.pt"
    EXTRA_ARGS+=(--init_from=resume)
  else
    echo "Starting ${DATASET_NAME}"
  fi

  python -u train.py config/train_s5_clean_offline_bc.py \
    --dataset="${DATASET_NAME}" \
    --out_dir="${OUT_DIR}" \
    --offline_train_subset_size="${SUBSET_SIZE}" \
    --wandb_log=True \
    --wandb_project=small-cot-experiments \
    --wandb_run_name="s5-clean-offline-bc-n${SUBSET_SIZE}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_PATH}"
done
