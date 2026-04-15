#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

P="${P:-7}"
M="${M:-30}"
OUT_DIR="${OUT_DIR:-out-modadd-base-p${P}-m${M}-depth1}"
BLOCK_SIZE="${BLOCK_SIZE:-$((M + 1))}"
WANDB_PROJECT="${WANDB_PROJECT:-small-cot-experiments}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${OUT_DIR}}"
EXTRA_ARGS=()

if [[ -f "${OUT_DIR}/completed.txt" ]]; then
  echo "Skipping ${OUT_DIR}; found completed.txt"
  exit 0
elif [[ -f "${OUT_DIR}/ckpt.pt" ]]; then
  echo "Resuming ${OUT_DIR} from ckpt.pt"
  EXTRA_ARGS+=(--init_from=resume)
else
  echo "Starting ${OUT_DIR}"
fi

python -u train.py config/train_modadd_base_p7_m30.py \
  --out_dir="${OUT_DIR}" \
  --modadd_p="${P}" \
  --modadd_m="${M}" \
  --block_size="${BLOCK_SIZE}" \
  --wandb_project="${WANDB_PROJECT}" \
  --wandb_run_name="${WANDB_RUN_NAME}" \
  "${EXTRA_ARGS[@]}"
