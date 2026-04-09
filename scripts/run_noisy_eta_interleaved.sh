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
SUBSET_SIZE="${SUBSET_SIZE:?Set SUBSET_SIZE to the chosen clean threshold N}"
ETAS="${ETAS:-0.005 0.1 0.2}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1024}"
SEED="${SEED:-1337}"
RENDER_LOG_DIR="${RENDER_LOG_DIR:-logs/noisy_dataset_render}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-logs/noisy_bc}"

mkdir -p "${RENDER_LOG_DIR}" "${TRAIN_LOG_DIR}"

for ETA in ${ETAS}; do
  ETA_TAG="${ETA/./p}"
  DATASET_NAME="s5_noisy_offline_n${SUBSET_SIZE}_eta_${ETA_TAG}"
  SAVE_DIR="data/${DATASET_NAME}"
  OUT_DIR="out-s5-noisy-bc-n${SUBSET_SIZE}-eta${ETA_TAG}"
  DONE_MARKER="${OUT_DIR}/completed.txt"
  RENDER_LOG_PATH="${RENDER_LOG_DIR}/${DATASET_NAME}.log"
  TRAIN_LOG_PATH="${TRAIN_LOG_DIR}/s5_noisy_bc_n${SUBSET_SIZE}_eta${ETA_TAG}.log"
  EXTRA_ARGS=()

  if [[ ! -f "${SAVE_DIR}/train_x.pt" || ! -f "${SAVE_DIR}/train_y.pt" || ! -f "${SAVE_DIR}/val_x.pt" || ! -f "${SAVE_DIR}/val_y.pt" ]]; then
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
      2>&1 | tee "${RENDER_LOG_PATH}"
  else
    echo "Skipping render for ${DATASET_NAME}; found existing tensors in ${SAVE_DIR}"
  fi

  if [[ -f "${DONE_MARKER}" ]]; then
    echo "Skipping train for ${DATASET_NAME}; found ${DONE_MARKER}"
    continue
  elif [[ -f "${OUT_DIR}/ckpt.pt" ]]; then
    echo "Resuming train for ${DATASET_NAME} from ${OUT_DIR}/ckpt.pt"
    EXTRA_ARGS+=(--init_from=resume)
  else
    echo "Starting train for ${DATASET_NAME}"
  fi

  python -u train.py config/train_s5_noisy_bc.py \
    --dataset="${DATASET_NAME}" \
    --out_dir="${OUT_DIR}" \
    --wandb_log=True \
    --wandb_project=small-cot-experiments \
    --wandb_run_name="s5-noisy-bc-n${SUBSET_SIZE}-eta${ETA_TAG}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${TRAIN_LOG_PATH}"
done
