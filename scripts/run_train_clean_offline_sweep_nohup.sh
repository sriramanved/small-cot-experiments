#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LOG_DIR="logs/clean_offline_sweep"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/overnight_clean_offline_sweep_${STAMP}.log"
PID_FILE="${LOG_DIR}/overnight_clean_offline_sweep_${STAMP}.pid"

nohup bash scripts/train_clean_offline_sweep.sh > "${MASTER_LOG}" 2>&1 &
echo $! > "${PID_FILE}"

echo "Started clean offline sweep in the background."
echo "PID: $(cat "${PID_FILE}")"
echo "Master log: ${MASTER_LOG}"
echo "Tail it with: tail -f ${MASTER_LOG}"
