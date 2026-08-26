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
MODEL_REPO="${BRO_MODEL_REPO:-leeroy-jankins/bro}"
MODEL_FILE="${BRO_MODEL_FILE:-bro-3-4b-it-qat-Q4_K_M.gguf}"

log "startup.sh entered"
log "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "app_file=${APP_FILE}"
log "port=${PORT}"
log "streamlit_address=${STREAMLIT_SERVER_ADDRESS}"
log "model_repo=${MODEL_REPO}"
log "model_file=${MODEL_FILE}"
log "model_dir=${MODEL_DIR}"
log "python=$(python --version 2>&1)"
log "working_directory=$(pwd)"

mkdir -p "${MODEL_DIR}"
export BRO_LLM_PATH="${BRO_LLM_PATH:-${MODEL_DIR}/${MODEL_FILE}}"
log "BRO_LLM_PATH=${BRO_LLM_PATH}"

log "filesystem before model load:"
df -h "${MODEL_DIR}" || true

if [[ ! -s "${BRO_LLM_PATH}" ]]; then
  log "model is not present; beginning Hugging Face download"
  export MODEL_DIR MODEL_REPO MODEL_FILE BRO_LLM_PATH

  python - <<'PY'
import os
import sys
import traceback
from pathlib import Path

print("STARTUP/PYTHON: model download helper entered", flush=True)

try:
    from huggingface_hub import hf_hub_download
    print("STARTUP/PYTHON: huggingface_hub import succeeded", flush=True)

    model_dir = Path(os.environ["MODEL_DIR"])
    model_repo = os.environ["MODEL_REPO"]
    model_file = os.environ["MODEL_FILE"]
    target = Path(os.environ["BRO_LLM_PATH"])

    print(f"STARTUP/PYTHON: repo={model_repo}", flush=True)
    print(f"STARTUP/PYTHON: filename={model_file}", flush=True)
    print(f"STARTUP/PYTHON: target={target}", flush=True)

    model_dir.mkdir(parents=True, exist_ok=True)
    print("STARTUP/PYTHON: invoking hf_hub_download", flush=True)

    downloaded = Path(
        hf_hub_download(
            repo_id=model_repo,
            filename=model_file,
            local_dir=str(model_dir),
        )
    )

    print(f"STARTUP/PYTHON: hf_hub_download returned {downloaded}", flush=True)

    if not downloaded.is_file():
        raise RuntimeError(f"Downloaded model does not exist: {downloaded}")

    size = downloaded.stat().st_size
    if size <= 0:
        raise RuntimeError(f"Downloaded model is empty: {downloaded}")

    if downloaded.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded.replace(target)

    print(
        f"STARTUP/PYTHON: model ready at {target} ({target.stat().st_size} bytes)",
        flush=True,
    )
except Exception as exc:
    print(f"STARTUP/PYTHON ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    raise
PY

  log "Hugging Face download helper completed"
else
  log "using existing model at ${BRO_LLM_PATH}"
fi

if [[ ! -s "${BRO_LLM_PATH}" ]]; then
  echo "STARTUP ERROR: model is still missing or empty at ${BRO_LLM_PATH}" >&2
  exit 90
fi

log "model verification succeeded"
log "model_size_bytes=$(stat -c %s "${BRO_LLM_PATH}")"
log "filesystem after model load:"
df -h "${MODEL_DIR}" || true

export STREAMLIT_SERVER_PORT="${PORT}"
export STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS}"

log "launching Streamlit"
log "command=streamlit run ${APP_FILE} --server.address=${STREAMLIT_SERVER_ADDRESS} --server.port=${PORT}"

exec streamlit run "${APP_FILE}" \
  --server.address="${STREAMLIT_SERVER_ADDRESS}" \
  --server.port="${PORT}" \
  --server.headless=true \
  --browser.gatherUsageStats=false
