#!/usr/bin/env bash
set -euo pipefail

cd ~/small-cot-experiments/nanoGPT
source .venv/bin/activate

for ETA in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  ETA_TAG="${ETA/./p}"
  python -u train.py config/train_s5_noisy_bc.py \
    --dataset="s5_noisy_offline_eta_${ETA_TAG}" \
    --out_dir="out-s5-noisy-bc-eta${ETA_TAG}" \
    --wandb_log=True \
    --wandb_project=small-cot-experiments \
    --wandb_run_name="s5-noisy-bc-eta${ETA_TAG}" \
    2>&1 | tee "s5_noisy_bc_eta${ETA_TAG}.log"
done