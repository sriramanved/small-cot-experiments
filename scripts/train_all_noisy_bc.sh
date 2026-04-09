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
ROLLOUT_MODE="${ROLLOUT_MODE:-greedy_then_corrupt}"
LOG_DIR="${LOG_DIR:-logs/noisy_bc}"
mkdir -p "${LOG_DIR}"

case "${ROLLOUT_MODE}" in
  greedy_then_corrupt)
    DATASET_PREFIX="s5_noisy_offline"
    OUT_PREFIX="out-s5-noisy-bc"
    RUN_PREFIX="s5-noisy-bc"
    ;;
  sample_then_corrupt)
    DATASET_PREFIX="s5_noisy_offline_sample_then_corrupt"
    OUT_PREFIX="out-s5-noisy-bc-sample-then-corrupt"
    RUN_PREFIX="s5-noisy-bc-sample-then-corrupt"
    ;;
  *)
    echo "Unknown ROLLOUT_MODE=${ROLLOUT_MODE}"
    exit 1
    ;;
esac

for ETA in ${ETAS}; do
  ETA_TAG="${ETA/./p}"
  DATASET_NAME="${DATASET_PREFIX}_n${SUBSET_SIZE}_eta_${ETA_TAG}"
  OUT_DIR="${OUT_PREFIX}-n${SUBSET_SIZE}-eta${ETA_TAG}"
  DONE_MARKER="${OUT_DIR}/completed.txt"
  LOG_PATH="${LOG_DIR}/${RUN_PREFIX}_n${SUBSET_SIZE}_eta${ETA_TAG}.log"
  EXTRA_ARGS=()

  if [[ ! -f "data/${DATASET_NAME}/train_x.pt" ]]; then
    echo "Missing data/${DATASET_NAME}/train_x.pt"
    echo "Run scripts/generate_all_noisy_datasets.sh first."
    exit 1
  fi

  python -u scripts/diagnose_noisy_offline_bc.py \
    --dataset_dir="data/${DATASET_NAME}" \
    --prompt_bank_dir="${PROMPT_BANK_DIR}" \
    --teacher_checkpoint="${TEACHER_CHECKPOINT}" \
    --subset_size="${SUBSET_SIZE}" \
    --eta="${ETA}" \
    --train_decode_mode="${ROLLOUT_MODE}" \
    --strict

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
    --wandb_run_name="${RUN_PREFIX}-n${SUBSET_SIZE}-eta${ETA_TAG}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_PATH}"
done
