#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/aics_gpu_smoke_test.sh --account ACCOUNT --partition PARTITION [options]

Options:
  --repo-path PATH     Repo to enter on the allocated node. Defaults to this repo root.
  --venv-path PATH     Virtualenv directory or activate script to source before testing.
  --cpus N             CPUs per task for the interactive allocation. Default: 4
  --mem SIZE           Memory request for the interactive allocation. Default: 16G
  --time HH:MM:SS      Walltime for the interactive allocation. Default: 00:30:00
  --dry-run            Print the command instead of running it.
  --help, -h           Show this help text.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCOUNT=""
PARTITION=""
REPO_PATH="${ROOT}"
VENV_PATH=""
CPUS="4"
MEM="16G"
TIME_LIMIT="00:30:00"
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
    --cpus)
      CPUS="${2:?missing value for --cpus}"
      shift 2
      ;;
    --mem)
      MEM="${2:?missing value for --mem}"
      shift 2
      ;;
    --time)
      TIME_LIMIT="${2:?missing value for --time}"
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
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${ACCOUNT}" || -z "${PARTITION}" ]]; then
  echo "--account and --partition are required." >&2
  usage >&2
  exit 1
fi

REPO_PATH="$(cd "${REPO_PATH}" && pwd)"
mkdir -p "${REPO_PATH}/logs/slurm" "${REPO_PATH}/logs/opd"

INNER_SCRIPT='
set -euo pipefail
REPO_PATH="$1"
VENV_PATH="${2:-}"
cd "$REPO_PATH"

if [[ -n "$VENV_PATH" ]]; then
  if [[ -d "$VENV_PATH" && -f "$VENV_PATH/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
  elif [[ -f "$VENV_PATH" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_PATH"
  else
    echo "VENV_PATH=$VENV_PATH does not look like a virtualenv directory or activate script." >&2
    exit 1
  fi
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

echo "node=$(hostname)"
echo "cwd=$(pwd)"
echo "python=$(command -v python)"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
nvidia-smi
exec bash -i
'

CMD=(
  srun
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --gres=gpu:1
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
  --time="${TIME_LIMIT}"
  --pty
  bash
  -lc
  "${INNER_SCRIPT}"
  bash
  "${REPO_PATH}"
  "${VENV_PATH}"
)

printf 'Command:'
for arg in "${CMD[@]}"; do
  printf ' %q' "${arg}"
done
printf '\n'

if (( DRY_RUN )); then
  exit 0
fi

"${CMD[@]}"
