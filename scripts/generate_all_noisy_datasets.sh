#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

M="${M:-21}"
N_TRAIN="${N_TRAIN:-6000000}"
N_VAL="${N_VAL:-5000}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-out-s5-cot-len21-depth1-400k}"
PROMPT_BANK_DIR="${PROMPT_BANK_DIR:-data/s5_clean_prompt_bank_m${M}_n${N_TRAIN}_val${N_VAL}}"
SUBSET_SIZE="${SUBSET_SIZE:-1000000}"  # set this after the clean-N sweep
ETAS="${ETAS:-0.005 0.1 0.2}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1024}"  # try 512, 1024, 2048, then 4096 on the dev node
SEED="${SEED:-1337}"
LOG_DIR="${LOG_DIR:-logs/noisy_dataset_render}"
mkdir -p "${LOG_DIR}"

for ETA in ${ETAS}; do
  ETA_TAG="${ETA/./p}"
  DATASET_NAME="s5_noisy_offline_n${SUBSET_SIZE}_eta_${ETA_TAG}"
  SAVE_DIR="data/${DATASET_NAME}"
  LOG_PATH="${LOG_DIR}/${DATASET_NAME}.log"

  if [[ -f "${SAVE_DIR}/train_x.pt" && -f "${SAVE_DIR}/train_y.pt" && -f "${SAVE_DIR}/val_x.pt" && -f "${SAVE_DIR}/val_y.pt" ]]; then
    echo "Skipping ${DATASET_NAME}; found existing tensors in ${SAVE_DIR}"
    continue
  fi

  echo "Rendering ${DATASET_NAME}"

  python -u data/s5_cot/generate_noisy_rollouts.py \
    --teacher_checkpoint="${TEACHER_CHECKPOINT}" \
    --prompt_bank_dir="${PROMPT_BANK_DIR}" \
    --save_dir="${SAVE_DIR}" \
    --subset_size="${SUBSET_SIZE}" \
    --eta="${ETA}" \
    --gen_batch_size="${GEN_BATCH_SIZE}" \
    --device="cuda" \
    --dtype="float16" \
    --seed="${SEED}" \
    2>&1 | tee "${LOG_PATH}"
done
