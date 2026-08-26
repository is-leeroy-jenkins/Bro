#!/usr/bin/env bash
set -euo pipefail

APP_FILE="${APP_FILE:-app.py}"
PORT="${PORT:-8501}"
STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
MODEL_DIR="${BRO_MODEL_DIR:-/app/models}"
MODEL_REPO="${BRO_MODEL_REPO:-leeroy-jankins/bro}"
MODEL_FILE="${BRO_MODEL_FILE:-bro-3-4b-it-qat-Q4_K_M.gguf}"

mkdir -p "${MODEL_DIR}"
export BRO_LLM_PATH="${BRO_LLM_PATH:-${MODEL_DIR}/${MODEL_FILE}}"

if [[ ! -s "${BRO_LLM_PATH}" ]]; then
  echo "Bro model not found at ${BRO_LLM_PATH}. Downloading ${MODEL_REPO}/${MODEL_FILE} with huggingface_hub..."
  export MODEL_DIR MODEL_REPO MODEL_FILE BRO_LLM_PATH
  python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

model_dir = Path(os.environ["MODEL_DIR"])
model_repo = os.environ["MODEL_REPO"]
model_file = os.environ["MODEL_FILE"]
target = Path(os.environ["BRO_LLM_PATH"])

model_dir.mkdir(parents=True, exist_ok=True)
path = Path(
    hf_hub_download(
        repo_id=model_repo,
        filename=model_file,
        local_dir=str(model_dir),
    )
)

if not path.is_file() or path.stat().st_size == 0:
    raise RuntimeError(f"Downloaded model is missing or empty: {path}")

if path.resolve() != target.resolve():
    target.parent.mkdir(parents=True, exist_ok=True)
    path.replace(target)

print(f"Bro model download complete: {target} ({target.stat().st_size} bytes)")
PY
else
  echo "Using cached Bro model at ${BRO_LLM_PATH}."
fi

export STREAMLIT_SERVER_PORT="${PORT}"
export STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS}"

exec streamlit run "${APP_FILE}" \
  --server.address="${STREAMLIT_SERVER_ADDRESS}" \
  --server.port="${PORT}" \
  --server.headless=true \
  --browser.gatherUsageStats=false
