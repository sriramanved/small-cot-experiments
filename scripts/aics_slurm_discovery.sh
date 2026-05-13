#!/usr/bin/env bash
set -u

usage() {
  cat <<'EOF'
Usage: bash scripts/aics_slurm_discovery.sh

Runs the AICS discovery commands from the cluster workflow in a fixed order and
continues even if one of them fails.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

COMMANDS=(
  'whoami'
  'hostname'
  'pwd'
  'groups'
  'sinfo -s'
  'sinfo -Nl'
  'scontrol show partition'
  'sacctmgr -n show associations user=$USER format=Account,Partition,QOS'
  'squeue -u $USER'
  'if command -v resource_usage >/dev/null 2>&1; then resource_usage; else echo "resource_usage not found"; fi'
  'if [[ -f ~/bin/interactivejob ]]; then cat ~/bin/interactivejob; else echo "~/bin/interactivejob not found"; fi'
  'find ~ -maxdepth 4 -type d -name small-cot-experiments 2>/dev/null'
  'find ~ -maxdepth 5 -type f -path "*/bin/activate" 2>/dev/null | grep -E "/(\.venv|venv)/bin/activate$"'
)

for cmd in "${COMMANDS[@]}"; do
  printf '\n$ %s\n' "${cmd}"
  bash -lc "${cmd}"
  status=$?
  if (( status != 0 )); then
    printf '[command exited with status %s]\n' "${status}"
  fi
done
