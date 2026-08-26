#!/usr/bin/env bash
set -Eeuo pipefail

on_error() {
  local exit_code=$?
  local line_no=${BASH_LINENO[0]:-unknown}
  local command=${BASH_COMMAND:-unknown}
  echo "STARTUP ERROR: exit_code=${exit_code} line=${line_no} command=${command}" >&2
  exit "${exit_code}"
}
trap on_error ERR

log() {
  printf 'STARTUP: %s\n' "$*"
}

APP_FILE="${APP_FILE:-app.py}"
PORT="${PORT:-8501}"
STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
MODEL_DIR="${BRO_MODEL_DIR:-/app/models}"
MODEL_FILE="${BRO_MODEL_FILE:-gemma-3-4b-it-Q4_K_M.gguf}"

export BRO_LLM_PATH="${BRO_LLM_PATH:-${MODEL_DIR}/${MODEL_FILE}}"
export STREAMLIT_SERVER_PORT="${PORT}"
export STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS}"

log "startup.sh entered"
log "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "app_file=${APP_FILE}"
log "port=${PORT}"
log "streamlit_address=${STREAMLIT_SERVER_ADDRESS}"
log "BRO_LLM_PATH=${BRO_LLM_PATH}"
log "python=$(python --version 2>&1)"
log "working_directory=$(pwd)"

if [[ ! -f "${BRO_LLM_PATH}" ]]; then
  echo "STARTUP ERROR: baked model file not found at ${BRO_LLM_PATH}" >&2
  exit 90
fi

if [[ ! -s "${BRO_LLM_PATH}" ]]; then
  echo "STARTUP ERROR: baked model file is empty at ${BRO_LLM_PATH}" >&2
  exit 91
fi

log "baked model verification succeeded"
log "model_size_bytes=$(stat -c %s "${BRO_LLM_PATH}")"
log "filesystem state:"
df -h "${MODEL_DIR}" || true

log "launching Streamlit"
log "command=streamlit run ${APP_FILE} --server.address=${STREAMLIT_SERVER_ADDRESS} --server.port=${PORT}"

exec streamlit run "${APP_FILE}" \
  --server.address="${STREAMLIT_SERVER_ADDRESS}" \
  --server.port="${PORT}" \
  --server.headless=true \
  --browser.gatherUsageStats=false
