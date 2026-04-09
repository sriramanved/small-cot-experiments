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
SEED="${SEED:-1337}"
PROMPT_BANK_DIR="${PROMPT_BANK_DIR:-data/s5_clean_prompt_bank_m${M}_n${N_TRAIN}_val${N_VAL}}"

python -u data/s5_cot/generate_clean_prompt_bank.py \
  --save_dir="${PROMPT_BANK_DIR}" \
  --m="${M}" \
  --n_train="${N_TRAIN}" \
  --n_val="${N_VAL}" \
  --seed="${SEED}"
