#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

P="${P:-7}"
M="${M:-21}"
OUT_DIR="${OUT_DIR:-out-modadd-cot-p${P}-m${M}-depth1}"
BLOCK_SIZE="${BLOCK_SIZE:-$((2 * M))}"
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

python -u train.py config/train_modadd_cot_p7_m21.py \
  --out_dir="${OUT_DIR}" \
  --modadd_p="${P}" \
  --modadd_m="${M}" \
  --block_size="${BLOCK_SIZE}" \
  "${EXTRA_ARGS[@]}"
