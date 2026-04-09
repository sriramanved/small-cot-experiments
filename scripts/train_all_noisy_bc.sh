#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

SUBSET_SIZE=1000000  # set this after the clean-N sweep

for ETA in 0.2 0.4 0.6 0.8; do
  ETA_TAG="${ETA/./p}"
  DATASET_NAME="s5_noisy_offline_n${SUBSET_SIZE}_eta_${ETA_TAG}"
  OUT_DIR="out-s5-noisy-bc-n${SUBSET_SIZE}-eta${ETA_TAG}"

  python -u train.py config/train_s5_noisy_bc.py \
    --dataset="${DATASET_NAME}" \
    --out_dir="${OUT_DIR}" \
    --wandb_log=True \
    --wandb_project=small-cot-experiments \
    --wandb_run_name="s5-noisy-bc-n${SUBSET_SIZE}-eta${ETA_TAG}" \
    2>&1 | tee "s5_noisy_bc_n${SUBSET_SIZE}_eta${ETA_TAG}.log"
done
