#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/aics_submit_s5_opd_eta.sh <eta> --account ACCOUNT --partition PARTITION [options]

Options:
  --venv-path PATH     Export VENV_PATH for the batch job before submission.
  --repo-path PATH     Repo root to submit from. Defaults to this repo root.
  --dry-run            Print the sbatch command instead of running it.
  --help, -h           Show this help text.

Equivalent direct command:
  mkdir -p logs/slurm logs/opd
  sbatch --account=<ACCOUNT> --partition=<GPU_PARTITION> --chdir="$PWD" run_s5_opd_eta.sh <eta>
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ETA=""
ACCOUNT=""
PARTITION=""
REPO_PATH="${ROOT}"
VENV_PATH=""
DRY_RUN=0

while (($#)); do
  case "$1" in
    --account)
      ACCOUNT="${2:?missing value for --account}"
      shift 2
      ;;
    --partition)
      PARTITION="${2:?missing value for --partition}"
      shift 2
      ;;
    --repo-path)
      REPO_PATH="${2:?missing value for --repo-path}"
      shift 2
      ;;
    --venv-path)
      VENV_PATH="${2:?missing value for --venv-path}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -z "${ETA}" ]]; then
        ETA="$1"
        shift
      else
        echo "Unexpected extra positional argument: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "${ETA}" || -z "${ACCOUNT}" || -z "${PARTITION}" ]]; then
  echo "eta, --account, and --partition are required." >&2
  usage >&2
  exit 1
fi

REPO_PATH="$(cd "${REPO_PATH}" && pwd)"
mkdir -p "${REPO_PATH}/logs/slurm" "${REPO_PATH}/logs/opd"

SBATCH_CMD=(
  sbatch
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --chdir="${REPO_PATH}"
  "${REPO_PATH}/run_s5_opd_eta.sh"
  "${ETA}"
)

if [[ -n "${VENV_PATH}" ]]; then
  CMD=(env "VENV_PATH=${VENV_PATH}" "${SBATCH_CMD[@]}")
else
  CMD=("${SBATCH_CMD[@]}")
fi

printf 'Command:'
for arg in "${CMD[@]}"; do
  printf ' %q' "${arg}"
done
printf '\n'

if (( DRY_RUN )); then
  exit 0
fi

"${CMD[@]}"
