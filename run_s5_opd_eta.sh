#!/usr/bin/env bash
# Submit with:
#   sbatch --account=<ACCOUNT> --partition=<GPU_PARTITION> run_s5_opd_eta.sh <eta>
# Optionally export VENV_PATH first if you do not want to rely on repo-local .venv:
#   VENV_PATH=/path/to/venv sbatch --account=<ACCOUNT> --partition=<GPU_PARTITION> run_s5_opd_eta.sh <eta>
#
# This script intentionally leaves account/partition out of #SBATCH directives so the
# submitter can choose only from values confirmed on the cluster.
#SBATCH --job-name=s5-opd-rkl-full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

ETA="${1:?usage: sbatch --account=<ACCOUNT> --partition=<GPU_PARTITION> run_s5_opd_eta.sh <eta>}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${ROOT}"

mkdir -p logs/slurm logs/opd

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

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

env \
  N_TRAIN="${N_TRAIN:-15000000}" \
  SUBSET_SIZE="${SUBSET_SIZE:-8000000}" \
  ETAS="${ETA}" \
  OBJECTIVE="${OBJECTIVE:-reverse_kl_full}" \
  TEACHER_LAW="${TEACHER_LAW:-distributional_noise}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}" \
  COMPILE="${COMPILE:-0}" \
  bash scripts/run_opd_sweep.sh
