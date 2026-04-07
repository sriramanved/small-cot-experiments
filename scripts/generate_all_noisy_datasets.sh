#!/usr/bin/env bash
set -euo pipefail

cd ~/small-cot-experiments/nanoGPT
source .venv/bin/activate

TEACHER_OUT="out-s5-cot-len21-depth1-400k"
M=21
N_TRAIN=50000
N_VAL=5000
TEMP=1.0
SEED=1337

for ETA in 0.2 0.4 0.6 0.8; do
  ETA_TAG="${ETA/./p}"
  SAVE_DIR="data/s5_noisy_offline_eta_${ETA_TAG}"

  echo "Generating noisy dataset for eta=${ETA} -> ${SAVE_DIR}"

  python -u data/s5_cot/generate_noisy_rollouts.py \
    --teacher_out_dir="${TEACHER_OUT}" \
    --save_dir="${SAVE_DIR}" \
    --eta="${ETA}" \
    --m="${M}" \
    --n_train="${N_TRAIN}" \
    --n_val="${N_VAL}" \
    --temperature="${TEMP}" \
    --seed="${SEED}"
done