#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

P="${P:-7}"
M="${M:-21}"
N_TRAIN="${N_TRAIN:-15000000}"
N_VAL="${N_VAL:-5000}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-out-modadd-cot-p${P}-m${M}-depth1}"
PROMPT_BANK_DIR="${PROMPT_BANK_DIR:-data/modadd_clean_prompt_bank_p${P}_m${M}_n${N_TRAIN}_val${N_VAL}}"
THRESHOLD_FILE="${THRESHOLD_FILE:-modadd_clean_threshold_p${P}_m${M}.json}"
SUBSET_SIZE="${SUBSET_SIZE:-}"
ETAS="${ETAS:-0.05 0.1 0.2}"
ROLLOUT_MODE="${ROLLOUT_MODE:-greedy_then_corrupt}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1024}"
SEED="${SEED:-1337}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
RENDER_LOG_DIR="${RENDER_LOG_DIR:-logs/modadd_noisy_dataset_render}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-logs/modadd_noisy_bc}"

mkdir -p "${RENDER_LOG_DIR}" "${TRAIN_LOG_DIR}"

if [[ -z "${SUBSET_SIZE}" ]]; then
  if [[ ! -f "${THRESHOLD_FILE}" ]]; then
    echo "Set SUBSET_SIZE or create ${THRESHOLD_FILE} with scripts/find_modadd_clean_threshold.py."
    exit 1
  fi
  SUBSET_SIZE="$(python - <<PY
import json
print(json.load(open("${THRESHOLD_FILE}", "r", encoding="utf-8"))["threshold_subset_size"])
PY
)"
fi

case "${ROLLOUT_MODE}" in
  greedy_then_corrupt)
    DATASET_PREFIX="modadd_noisy_offline"
    OUT_PREFIX="out-modadd-noisy-bc"
    RUN_PREFIX="modadd-noisy-bc"
    ;;
  sample_then_corrupt)
    DATASET_PREFIX="modadd_noisy_offline_sample_then_corrupt"
    OUT_PREFIX="out-modadd-noisy-bc-sample-then-corrupt"
    RUN_PREFIX="modadd-noisy-bc-sample-then-corrupt"
    ;;
  *)
    echo "Unknown ROLLOUT_MODE=${ROLLOUT_MODE}"
    exit 1
    ;;
esac

for ETA in ${ETAS}; do
  ETA_TAG="${ETA/./p}"
  DATASET_NAME="${DATASET_PREFIX}_p${P}_m${M}_n${SUBSET_SIZE}_eta_${ETA_TAG}"
  SAVE_DIR="data/${DATASET_NAME}"
  OUT_DIR="${OUT_PREFIX}-p${P}-m${M}-n${SUBSET_SIZE}-eta${ETA_TAG}"
  DONE_MARKER="${OUT_DIR}/completed.txt"
  RENDER_LOG_PATH="${RENDER_LOG_DIR}/${DATASET_NAME}.log"
  TRAIN_LOG_PATH="${TRAIN_LOG_DIR}/${RUN_PREFIX}_p${P}_m${M}_n${SUBSET_SIZE}_eta${ETA_TAG}.log"
  EXTRA_ARGS=()

  if [[ ! -f "${SAVE_DIR}/train_x.pt" || ! -f "${SAVE_DIR}/train_y.pt" || ! -f "${SAVE_DIR}/val_x.pt" || ! -f "${SAVE_DIR}/val_y.pt" ]]; then
    echo "Rendering ${DATASET_NAME}"
    python -u data/modular_addition/generate_noisy_rollouts.py \
      --teacher_checkpoint="${TEACHER_CHECKPOINT}" \
      --prompt_bank_dir="${PROMPT_BANK_DIR}" \
      --save_dir="${SAVE_DIR}" \
      --subset_size="${SUBSET_SIZE}" \
      --eta="${ETA}" \
      --rollout_mode="${ROLLOUT_MODE}" \
      --gen_batch_size="${GEN_BATCH_SIZE}" \
      --device="${DEVICE}" \
      --dtype="${DTYPE}" \
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

  python -u train.py config/train_modadd_noisy_bc.py \
    --dataset="${DATASET_NAME}" \
    --out_dir="${OUT_DIR}" \
    --modadd_p="${P}" \
    --modadd_m="${M}" \
    --wandb_log=True \
    --wandb_project=small-cot-experiments \
    --wandb_run_name="${RUN_PREFIX}-p${P}-m${M}-n${SUBSET_SIZE}-eta${ETA_TAG}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${TRAIN_LOG_PATH}"
done
