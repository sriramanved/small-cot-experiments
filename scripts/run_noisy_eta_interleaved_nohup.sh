#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LOG_DIR="${LOG_DIR:-logs/noisy_eta_interleaved}"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/overnight_noisy_eta_interleaved_${STAMP}.log"
PID_FILE="${LOG_DIR}/overnight_noisy_eta_interleaved_${STAMP}.pid"

nohup bash scripts/run_noisy_eta_interleaved.sh > "${MASTER_LOG}" 2>&1 &
echo $! > "${PID_FILE}"

echo "Started noisy eta interleaved run in the background."
echo "PID: $(cat "${PID_FILE}")"
echo "Master log: ${MASTER_LOG}"
echo "Tail it with: tail -f ${MASTER_LOG}"
