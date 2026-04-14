#!/usr/bin/env bash
# Submit with:
#   sbatch --account=<ACCOUNT> --partition=<GPU_PARTITION> run_s5_noisy_bc_mc_eta.sh <eta>
# Optionally export VENV_PATH first if you do not want to rely on repo-local .venv:
#   VENV_PATH=/path/to/venv sbatch --account=<ACCOUNT> --partition=<GPU_PARTITION> run_s5_noisy_bc_mc_eta.sh <eta>
#
# This script intentionally leaves account/partition out of #SBATCH directives so the
# submitter can choose only from values confirmed on the cluster.
#SBATCH --job-name=s5-bc-mc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

ETA="${1:?usage: sbatch --account=<ACCOUNT> --partition=<GPU_PARTITION> run_s5_noisy_bc_mc_eta.sh <eta>}"
LOW_ETA_SUBSET_SIZE="${LOW_ETA_SUBSET_SIZE:-8000000}"
HIGH_ETA_SUBSET_SIZE="${HIGH_ETA_SUBSET_SIZE:-12000000}"
HIGH_ETA_THRESHOLD="${HIGH_ETA_THRESHOLD:-0.2}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${ROOT}"

mkdir -p logs/slurm logs/noisy_dataset_render logs/noisy_bc

VENV_PATH="${VENV_PATH:-}"
if [[ -n "${VENV_PATH}" ]]; then
  if [[ -d "${VENV_PATH}" && -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
  elif [[ -f "${VENV_PATH}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}"
  else
    echo "VENV_PATH=${VENV_PATH} does not look like a virtualenv directory or activate script." >&2
    exit 1
  fi
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ -n "${SUBSET_SIZE:-}" ]]; then
  RESOLVED_SUBSET_SIZE="${SUBSET_SIZE}"
else
  if python3 - <<PY
import sys
sys.exit(0 if float("${ETA}") > float("${HIGH_ETA_THRESHOLD}") else 1)
PY
  then
    RESOLVED_SUBSET_SIZE="${HIGH_ETA_SUBSET_SIZE}"
  else
    RESOLVED_SUBSET_SIZE="${LOW_ETA_SUBSET_SIZE}"
  fi
fi

echo "Using subset_size=${RESOLVED_SUBSET_SIZE} for eta=${ETA}"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

env \
  N_TRAIN="${N_TRAIN:-15000000}" \
  SUBSET_SIZE="${RESOLVED_SUBSET_SIZE}" \
  ETAS="${ETA}" \
  ROLLOUT_MODE="${ROLLOUT_MODE:-sample_then_corrupt}" \
  TARGET_MODE="${TARGET_MODE:-tokens}" \
  GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1024}" \
  TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-out-s5-cot-len21-depth1-400k}" \
  PROMPT_BANK_DIR="${PROMPT_BANK_DIR:-data/s5_clean_prompt_bank_m21_n15000000_val5000}" \
  BC_COMPILE="${BC_COMPILE:-True}" \
  BC_S5_EVAL_BATCH_SIZE="${BC_S5_EVAL_BATCH_SIZE:-512}" \
  BC_SAVE_EVERY="${BC_SAVE_EVERY:-0}" \
  bash scripts/run_noisy_eta_interleaved.sh
