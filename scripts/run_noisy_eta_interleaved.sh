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
ROLLOUT_MODE="${ROLLOUT_MODE:-greedy_then_corrupt}"
TARGET_MODE="${TARGET_MODE:-tokens}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1024}"
SEED="${SEED:-1337}"
RENDER_LOG_DIR="${RENDER_LOG_DIR:-logs/noisy_dataset_render}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-logs/noisy_bc}"

mkdir -p "${RENDER_LOG_DIR}" "${TRAIN_LOG_DIR}"

if [[ "${TARGET_MODE}" != "tokens" && "${TARGET_MODE}" != "teacher_probs" ]]; then
  echo "Unknown TARGET_MODE=${TARGET_MODE}"
  exit 1
fi

if [[ "${TARGET_MODE}" == "teacher_probs" ]]; then
  if [[ "${ROLLOUT_MODE}" != "sample_then_corrupt" ]]; then
    echo "TARGET_MODE=teacher_probs currently requires ROLLOUT_MODE=sample_then_corrupt"
    exit 1
  fi
  DATASET_PREFIX="s5_noisy_offline_full_dist_sample_then_corrupt"
  OUT_PREFIX="out-s5-noisy-bc-full-dist-sample-then-corrupt"
  RUN_PREFIX="s5-noisy-bc-full-dist-sample-then-corrupt"
  OFFLINE_TARGET_TYPE="teacher_probs"
else
  OFFLINE_TARGET_TYPE="tokens"
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
fi

for ETA in ${ETAS}; do
  ETA_TAG="${ETA/./p}"
  DATASET_NAME="${DATASET_PREFIX}_n${SUBSET_SIZE}_eta_${ETA_TAG}"
  SAVE_DIR="data/${DATASET_NAME}"
  OUT_DIR="${OUT_PREFIX}-n${SUBSET_SIZE}-eta${ETA_TAG}"
  DONE_MARKER="${OUT_DIR}/completed.txt"
  RENDER_LOG_PATH="${RENDER_LOG_DIR}/${DATASET_NAME}.log"
  TRAIN_LOG_PATH="${TRAIN_LOG_DIR}/${RUN_PREFIX}_n${SUBSET_SIZE}_eta${ETA_TAG}.log"
  EXTRA_ARGS=()
  REQUIRED_RENDER_FILES=(
    "${SAVE_DIR}/train_x.pt"
    "${SAVE_DIR}/train_y.pt"
    "${SAVE_DIR}/val_x.pt"
    "${SAVE_DIR}/val_y.pt"
  )
  if [[ "${TARGET_MODE}" == "teacher_probs" ]]; then
    REQUIRED_RENDER_FILES+=("${SAVE_DIR}/train_teacher_probs.pt")
  fi

  MISSING_RENDER_FILE=0
  for REQUIRED_FILE in "${REQUIRED_RENDER_FILES[@]}"; do
    if [[ ! -f "${REQUIRED_FILE}" ]]; then
      MISSING_RENDER_FILE=1
      break
    fi
  done

  if [[ "${MISSING_RENDER_FILE}" == "1" ]]; then
    echo "Rendering ${DATASET_NAME}"
    python -u data/s5_cot/generate_noisy_rollouts.py \
      --teacher_checkpoint="${TEACHER_CHECKPOINT}" \
      --prompt_bank_dir="${PROMPT_BANK_DIR}" \
      --save_dir="${SAVE_DIR}" \
      --subset_size="${SUBSET_SIZE}" \
      --eta="${ETA}" \
      --rollout_mode="${ROLLOUT_MODE}" \
      --target_mode="${TARGET_MODE}" \
      --gen_batch_size="${GEN_BATCH_SIZE}" \
      --device="cuda" \
      --dtype="float16" \
      --seed="${SEED}" \
      2>&1 | tee "${RENDER_LOG_PATH}"
  else
    python -u scripts/diagnose_noisy_offline_bc.py \
      --dataset_dir="${SAVE_DIR}" \
      --prompt_bank_dir="${PROMPT_BANK_DIR}" \
      --teacher_checkpoint="${TEACHER_CHECKPOINT}" \
      --subset_size="${SUBSET_SIZE}" \
      --eta="${ETA}" \
      --train_decode_mode="${ROLLOUT_MODE}" \
      --train_target_type="${TARGET_MODE}" \
      --strict
    echo "Skipping render for ${DATASET_NAME}; found matching existing tensors in ${SAVE_DIR}"
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
    --offline_target_type="${OFFLINE_TARGET_TYPE}" \
    --wandb_log=True \
    --wandb_project=small-cot-experiments \
    --wandb_run_name="${RUN_PREFIX}-n${SUBSET_SIZE}-eta${ETA_TAG}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${TRAIN_LOG_PATH}"
done
