#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

M="${M:-21}"
N_TRAIN="${N_TRAIN:-15000000}"
N_VAL="${N_VAL:-5000}"
PROMPT_BANK_DIR="${PROMPT_BANK_DIR:-data/s5_clean_prompt_bank_m${M}_n${N_TRAIN}_val${N_VAL}}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-out-s5-cot-len21-depth1-400k}"
SUBSET_SIZE="${SUBSET_SIZE:-8000000}"
ETAS="${ETAS:-0.05 0.1 0.2}"
TEACHER_LAW="${TEACHER_LAW:-distributional_noise}"
OBJECTIVE="${OBJECTIVE:-reverse_kl_tm}"
STUDENT_TEMPERATURE="${STUDENT_TEMPERATURE:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_ITERS="${MAX_ITERS:-110000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_ITERS="${WARMUP_ITERS:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
EVAL_N="${EVAL_N:-5000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
SEED="${SEED:-1337}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
EPS="${EPS:-1e-10}"
SHUFFLE_PROMPTS="${SHUFFLE_PROMPTS:-0}"
COMPILE="${COMPILE:-0}"
WANDB_LOG="${WANDB_LOG:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-small-cot-experiments}"
LOG_DIR="${LOG_DIR:-logs/opd_hf}"

mkdir -p "${LOG_DIR}"

if [[ "${STUDENT_TEMPERATURE}" == "0" || "${STUDENT_TEMPERATURE}" == "0.0" ]]; then
  TEMP_TAG="greedy"
else
  TEMP_TAG="t${STUDENT_TEMPERATURE/./p}"
fi

for ETA in ${ETAS}; do
  ETA_TAG="${ETA/./p}"
  OUT_DIR="out-s5-opd-hf-${OBJECTIVE}-n${SUBSET_SIZE}-eta${ETA_TAG}-${TEACHER_LAW}-${TEMP_TAG}"
  LOG_PATH="${LOG_DIR}/s5_opd_hf_${OBJECTIVE}_n${SUBSET_SIZE}_eta${ETA_TAG}_${TEACHER_LAW}_${TEMP_TAG}.log"
  EXTRA_ARGS=()

  if [[ -f "${OUT_DIR}/completed.txt" ]]; then
    echo "Skipping ${OUT_DIR}; found completed.txt"
    continue
  elif [[ -f "${OUT_DIR}/training_state.pt" ]]; then
    echo "Resuming ${OUT_DIR} from training_state.pt"
    EXTRA_ARGS+=(--init_from=resume)
  else
    echo "Starting ${OUT_DIR}"
  fi

  if [[ "${SHUFFLE_PROMPTS}" == "1" ]]; then
    EXTRA_ARGS+=(--shuffle_prompts)
  fi
  if [[ "${COMPILE}" == "1" ]]; then
    EXTRA_ARGS+=(--compile)
  fi
  if [[ "${WANDB_LOG}" == "1" ]]; then
    EXTRA_ARGS+=(--wandb_log)
  fi

  python -u train_opd_hf.py \
    --teacher_checkpoint="${TEACHER_CHECKPOINT}" \
    --prompt_bank_dir="${PROMPT_BANK_DIR}" \
    --subset_size="${SUBSET_SIZE}" \
    --eta="${ETA}" \
    --teacher_law="${TEACHER_LAW}" \
    --objective="${OBJECTIVE}" \
    --out_dir="${OUT_DIR}" \
    --batch_size="${BATCH_SIZE}" \
    --max_iters="${MAX_ITERS}" \
    --learning_rate="${LEARNING_RATE}" \
    --warmup_iters="${WARMUP_ITERS}" \
    --student_temperature="${STUDENT_TEMPERATURE}" \
    --eval_interval="${EVAL_INTERVAL}" \
    --eval_n="${EVAL_N}" \
    --eval_batch_size="${EVAL_BATCH_SIZE}" \
    --log_interval="${LOG_INTERVAL}" \
    --save_interval="${SAVE_INTERVAL}" \
    --seed="${SEED}" \
    --device="${DEVICE}" \
    --dtype="${DTYPE}" \
    --eps="${EPS}" \
    --wandb_project="${WANDB_PROJECT}" \
    --wandb_run_name="s5-opd-hf-${OBJECTIVE}-n${SUBSET_SIZE}-eta${ETA_TAG}-${TEACHER_LAW}-${TEMP_TAG}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_PATH}"
done
