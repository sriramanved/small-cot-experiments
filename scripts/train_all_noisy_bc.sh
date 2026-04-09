#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

SUBSET_SIZE="${SUBSET_SIZE:-1000000}"  # set this after the clean-N sweep
ETAS="${ETAS:-0.005 0.1 0.2}"
LOG_DIR="${LOG_DIR:-logs/noisy_bc}"
mkdir -p "${LOG_DIR}"

for ETA in ${ETAS}; do
  ETA_TAG="${ETA/./p}"
  DATASET_NAME="s5_noisy_offline_n${SUBSET_SIZE}_eta_${ETA_TAG}"
  OUT_DIR="out-s5-noisy-bc-n${SUBSET_SIZE}-eta${ETA_TAG}"
  DONE_MARKER="${OUT_DIR}/completed.txt"
  LOG_PATH="${LOG_DIR}/s5_noisy_bc_n${SUBSET_SIZE}_eta${ETA_TAG}.log"
  EXTRA_ARGS=()

  if [[ ! -f "data/${DATASET_NAME}/train_x.pt" ]]; then
    echo "Missing data/${DATASET_NAME}/train_x.pt"
    echo "Run scripts/generate_all_noisy_datasets.sh first."
    exit 1
  fi

  if [[ -f "${DONE_MARKER}" ]]; then
    echo "Skipping ${DATASET_NAME}; found ${DONE_MARKER}"
    continue
  elif [[ -f "${OUT_DIR}/ckpt.pt" ]]; then
    echo "Resuming ${DATASET_NAME} from ${OUT_DIR}/ckpt.pt"
    EXTRA_ARGS+=(--init_from=resume)
  else
    echo "Starting ${DATASET_NAME}"
  fi

  python -u train.py config/train_s5_noisy_bc.py \
    --dataset="${DATASET_NAME}" \
    --out_dir="${OUT_DIR}" \
    --wandb_log=True \
    --wandb_project=small-cot-experiments \
    --wandb_run_name="s5-noisy-bc-n${SUBSET_SIZE}-eta${ETA_TAG}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_PATH}"
done
