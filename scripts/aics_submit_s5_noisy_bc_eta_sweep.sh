#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/aics_submit_s5_noisy_bc_eta_sweep.sh --mode MODE --account ACCOUNT --partition PARTITION [options] <eta ...>

Modes:
  mc         Offline BC MC with sample_then_corrupt targets
  full_dist  Offline BC full-distribution with teacher_probs targets

Options:
  --venv-path PATH     Pass VENV_PATH through to each batch job.
  --repo-path PATH     Repo root to submit from. Defaults to this repo root.
  --dry-run            Print the commands without submitting them.
  --help, -h           Show this help text.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=""
ACCOUNT=""
PARTITION=""
REPO_PATH="${ROOT}"
VENV_PATH=""
DRY_RUN=0
declare -a ETAS=()

while (($#)); do
  case "$1" in
    --mode)
      MODE="${2:?missing value for --mode}"
      shift 2
      ;;
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
      ETAS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${MODE}" || -z "${ACCOUNT}" || -z "${PARTITION}" ]]; then
  echo "--mode, --account, and --partition are required." >&2
  usage >&2
  exit 1
fi

if ((${#ETAS[@]} == 0)); then
  echo "Provide at least one eta value." >&2
  usage >&2
  exit 1
fi

REPO_PATH="$(cd "${REPO_PATH}" && pwd)"

for eta in "${ETAS[@]}"; do
  CMD=(
    bash
    "${REPO_PATH}/scripts/aics_submit_s5_noisy_bc_eta.sh"
    "${eta}"
    --mode
    "${MODE}"
    --account
    "${ACCOUNT}"
    --partition
    "${PARTITION}"
  )
  if [[ -n "${VENV_PATH}" ]]; then
    CMD+=(
      --venv-path
      "${VENV_PATH}"
    )
  fi
  if (( DRY_RUN )); then
    CMD+=(--dry-run)
  fi

  printf '\nSubmitting mode=%s eta=%s\n' "${MODE}" "${eta}"
  "${CMD[@]}"
done
