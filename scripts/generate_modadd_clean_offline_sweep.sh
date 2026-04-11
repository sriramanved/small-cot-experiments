#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

P="${P:-7}"
M="${M:-21}"
N_TRAIN="${N_TRAIN:-6000000}"
N_VAL="${N_VAL:-5000}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-out-modadd-cot-p${P}-m${M}-depth1}"
PROMPT_BANK_DIR="${PROMPT_BANK_DIR:-data/modadd_clean_prompt_bank_p${P}_m${M}_n${N_TRAIN}_val${N_VAL}}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1024}"
SEED="${SEED:-1337}"
FULL_SUBSET_SIZE="${FULL_SUBSET_SIZE:-${N_TRAIN}}"
DATASET_NAME="${DATASET_NAME:-modadd_clean_offline_p${P}_m${M}_n${FULL_SUBSET_SIZE}}"
SAVE_DIR="${SAVE_DIR:-data/${DATASET_NAME}}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"

if (( FULL_SUBSET_SIZE > N_TRAIN )); then
  echo "FULL_SUBSET_SIZE=${FULL_SUBSET_SIZE} exceeds N_TRAIN=${N_TRAIN}"
  exit 1
fi

if [[ ! -f "${PROMPT_BANK_DIR}/clean_train_prompt_ids.pt" ]]; then
  echo "Generating prompt bank at ${PROMPT_BANK_DIR}"
  python -u data/modular_addition/generate_clean_prompt_bank.py \
    --save_dir="${PROMPT_BANK_DIR}" \
    --p="${P}" \
    --m="${M}" \
    --n_train="${N_TRAIN}" \
    --n_val="${N_VAL}" \
    --seed="${SEED}"
fi

if [[ -f "${SAVE_DIR}/train_x.pt" && -f "${SAVE_DIR}/train_y.pt" && -f "${SAVE_DIR}/val_x.pt" && -f "${SAVE_DIR}/val_y.pt" ]]; then
  export SAVE_DIR
  export FULL_SUBSET_SIZE
  export P
  export M
  python - <<'PY'
import json
import os
from pathlib import Path
import torch

save_dir = Path(os.environ["SAVE_DIR"])
expected_n = int(os.environ["FULL_SUBSET_SIZE"])
expected_p = int(os.environ["P"])
expected_m = int(os.environ["M"])

train_x = torch.load(save_dir / "train_x.pt", map_location="cpu")
train_y = torch.load(save_dir / "train_y.pt", map_location="cpu")
meta = json.load(open(save_dir / "meta.json"))

assert train_x.size(0) == expected_n, f"existing train_x has {train_x.size(0)} rows, expected {expected_n}"
assert train_y.size(0) == expected_n, f"existing train_y has {train_y.size(0)} rows, expected {expected_n}"
assert int(meta["subset_size"]) == expected_n, f"existing meta subset_size={meta['subset_size']}, expected {expected_n}"
assert float(meta["eta"]) == 0.0, f"existing meta eta={meta['eta']}, expected 0.0"
assert str(meta["task"]) == "modadd", f"existing meta task={meta['task']}, expected modadd"
assert int(meta["p"]) == expected_p, f"existing meta p={meta['p']}, expected {expected_p}"
assert int(meta["m"]) == expected_m, f"existing meta m={meta['m']}, expected {expected_m}"
print(f"Verified existing clean offline dataset at {save_dir} with {expected_n} rows.")
PY
  echo "Skipping render; verified existing ${DATASET_NAME} tensors in ${SAVE_DIR}"
  exit 0
fi

echo "Rendering ${DATASET_NAME} once; smaller N sweeps will train on strict prefixes of this dataset."

python -u data/modular_addition/generate_noisy_rollouts.py \
  --teacher_checkpoint="${TEACHER_CHECKPOINT}" \
  --prompt_bank_dir="${PROMPT_BANK_DIR}" \
  --save_dir="${SAVE_DIR}" \
  --subset_size="${FULL_SUBSET_SIZE}" \
  --eta=0.0 \
  --gen_batch_size="${GEN_BATCH_SIZE}" \
  --device="${DEVICE}" \
  --dtype="${DTYPE}" \
  --seed="${SEED}"
