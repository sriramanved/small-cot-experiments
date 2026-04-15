#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

P="${P:-7}"
M="${M:-21}"
BLOCK_SIZE="${BLOCK_SIZE:-$((2 * M))}"
RUN_TAG="${RUN_TAG:-}"
BASE_N="${BASE_N:-15000000}"
SUBSET_SIZES="${SUBSET_SIZES:-250000 500000 1000000 2000000 4000000 6000000}"
BASE_DATASET="${BASE_DATASET:-modadd_clean_offline_p${P}_m${M}_n${BASE_N}}"
LOG_DIR="${LOG_DIR:-logs/modadd_clean_offline_sweep}"
EVAL_INTERVAL="${EVAL_INTERVAL:-}"
mkdir -p "${LOG_DIR}"
BASE_DATASET_DIR="data/${BASE_DATASET}"
TAG_SUFFIX=""
if [[ -n "${RUN_TAG}" ]]; then
  TAG_SUFFIX="-${RUN_TAG}"
fi

if [[ ! -f "${BASE_DATASET_DIR}/train_x.pt" ]]; then
  echo "Missing ${BASE_DATASET_DIR}/train_x.pt"
  echo "Run scripts/generate_modadd_clean_offline_sweep.sh first."
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
  OUT_DIR="out-modadd-clean-offline-bc-p${P}-m${M}-n${SUBSET_SIZE}${TAG_SUFFIX}"
  DONE_MARKER="${OUT_DIR}/completed.txt"
  LOG_PATH="${LOG_DIR}/modadd_clean_offline_bc_p${P}_m${M}_n${SUBSET_SIZE}${TAG_SUFFIX}.log"
  EXTRA_ARGS=()

  if [[ -f "${DONE_MARKER}" ]]; then
    echo "Skipping ${OUT_DIR}; found ${DONE_MARKER}"
    continue
  elif [[ -f "${OUT_DIR}/ckpt.pt" ]]; then
    echo "Resuming ${OUT_DIR} from ${OUT_DIR}/ckpt.pt"
    EXTRA_ARGS+=(--init_from=resume)
  else
    echo "Starting ${OUT_DIR}"
  fi

  if [[ -n "${EVAL_INTERVAL}" ]]; then
    EXTRA_ARGS+=(--eval_interval="${EVAL_INTERVAL}")
  fi

  python -u train.py config/train_modadd_clean_offline_bc.py \
    --dataset="${BASE_DATASET}" \
    --out_dir="${OUT_DIR}" \
    --modadd_p="${P}" \
    --modadd_m="${M}" \
    --block_size="${BLOCK_SIZE}" \
    --offline_train_subset_size="${SUBSET_SIZE}" \
    --wandb_log=True \
    --wandb_project=small-cot-experiments \
    --wandb_run_name="modadd-clean-offline-bc-p${P}-m${M}-n${SUBSET_SIZE}${TAG_SUFFIX}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_PATH}"
done
