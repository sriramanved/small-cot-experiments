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
MAX_ITERS="${MAX_ITERS:-125000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_ITERS="${WARMUP_ITERS:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
EVAL_N="${EVAL_N:-5000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
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
LOG_DIR="${LOG_DIR:-logs/opd}"

mkdir -p "${LOG_DIR}"

extract_wandb_run_id() {
  local out_dir="$1"
  local log_path="$2"
  local run_name="$3"
  local state_path="${out_dir}/wandb_state.json"
  local fallback_id
  fallback_id="$(python3 - <<PY
import hashlib
from pathlib import Path
project = "${WANDB_PROJECT}"
out_dir = Path("${out_dir}").resolve()
print(hashlib.sha1(f"{project}:{out_dir}".encode("utf-8")).hexdigest()[:16])
PY
)"

  if [[ -f "${log_path}" ]]; then
    local log_run_id
    log_run_id="$(grep -oE 'runs/[A-Za-z0-9]+' "${log_path}" | tail -n1 | cut -d/ -f2)"
    if [[ -n "${log_run_id}" ]]; then
      echo "${log_run_id}"
      return 0
    fi
  fi
  if compgen -G 'wandb/run-*' > /dev/null; then
    local cache_run_id
    cache_run_id="$(
      grep -R -l -F -- "${run_name}" wandb/run-* 2>/dev/null \
      | sed -nE 's#.*wandb/run-[^/]*-([A-Za-z0-9]+)/.*#\1#p' \
      | grep -v "^${fallback_id}$" \
      | tail -n1
    )"
    if [[ -n "${cache_run_id}" ]]; then
      echo "${cache_run_id}"
      return 0
    fi
  fi
  if [[ -f "${state_path}" ]]; then
    local state_run_id
    state_run_id="$(grep -oE '"run_id"[[:space:]]*:[[:space:]]*"[^"]+"' "${state_path}" | tail -n1 | sed -E 's/.*"([^"]+)".*/\1/')"
    if [[ -n "${state_run_id}" && "${state_run_id}" != "${fallback_id}" ]]; then
      echo "${state_run_id}"
      return 0
    fi
  fi
  return 1
}

if [[ "${STUDENT_TEMPERATURE}" == "0" || "${STUDENT_TEMPERATURE}" == "0.0" ]]; then
  TEMP_TAG="greedy"
else
  TEMP_TAG="t${STUDENT_TEMPERATURE/./p}"
fi

for ETA in ${ETAS}; do
  ETA_TAG="${ETA/./p}"
  OUT_DIR="out-s5-opd-${OBJECTIVE}-n${SUBSET_SIZE}-eta${ETA_TAG}-${TEACHER_LAW}-${TEMP_TAG}"
  LOG_PATH="${LOG_DIR}/s5_opd_${OBJECTIVE}_n${SUBSET_SIZE}_eta${ETA_TAG}_${TEACHER_LAW}_${TEMP_TAG}.log"
  RUN_NAME="s5-opd-${OBJECTIVE}-n${SUBSET_SIZE}-eta${ETA_TAG}-${TEACHER_LAW}-${TEMP_TAG}"
  EXTRA_ARGS=()
  COMPLETED_PATH="${OUT_DIR}/completed.txt"
  CKPT_PATH="${OUT_DIR}/ckpt.pt"

  if [[ -f "${COMPLETED_PATH}" ]]; then
    COMPLETED_ITER="$(sed -n 's/^iter_num=//p' "${COMPLETED_PATH}" | tail -n1)"
    if [[ -n "${COMPLETED_ITER}" ]] && (( COMPLETED_ITER >= MAX_ITERS )); then
      echo "Skipping ${OUT_DIR}; completed at iter ${COMPLETED_ITER} >= MAX_ITERS=${MAX_ITERS}"
      continue
    elif [[ -f "${CKPT_PATH}" ]]; then
      echo "Extending ${OUT_DIR} from iter ${COMPLETED_ITER:-unknown} to MAX_ITERS=${MAX_ITERS}"
      EXTRA_ARGS+=(--init_from=resume)
    else
      echo "Skipping ${OUT_DIR}; found completed.txt but no ckpt.pt to resume from"
      continue
    fi
  elif [[ -f "${CKPT_PATH}" ]]; then
    echo "Resuming ${OUT_DIR} from ckpt.pt"
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
    if WANDB_RUN_ID="$(extract_wandb_run_id "${OUT_DIR}" "${LOG_PATH}" "${RUN_NAME}")" && [[ -n "${WANDB_RUN_ID}" ]]; then
      EXTRA_ARGS+=(--wandb_run_id="${WANDB_RUN_ID}")
    fi
  fi

  echo "==== $(date '+%Y-%m-%d %H:%M:%S') launching ${RUN_NAME} ====" | tee -a "${LOG_PATH}"
  python -u train_opd.py \
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
    --wandb_run_name="${RUN_NAME}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "${LOG_PATH}"
done
