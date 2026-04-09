#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

TEACHER_CHECKPOINT="out-s5-cot-len21-depth1-400k"
PROMPT_BANK_DIR="data/s5_clean_prompt_bank_m21_n6000000_val5000"
SUBSET_SIZE=1000000  # set this after the clean-N sweep
GEN_BATCH_SIZE=1024  # try 512, 1024, 2048, then 4096 on the dev node
SEED=1337

for ETA in 0.2 0.4 0.6 0.8; do
  ETA_TAG="${ETA/./p}"
  DATASET_NAME="s5_noisy_offline_n${SUBSET_SIZE}_eta_${ETA_TAG}"
  SAVE_DIR="data/${DATASET_NAME}"

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
    --seed="${SEED}"
done
