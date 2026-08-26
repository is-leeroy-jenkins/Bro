#!/usr/bin/env bash
set -euo pipefail

APP_FILE="${APP_FILE:-app.py}"
PORT="${PORT:-8501}"
STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
MODEL_DIR="${BRO_MODEL_DIR:-/app/models}"
MODEL_FILE="${BRO_MODEL_FILE:-bro-3-4b-it-qat-Q4_K_M.gguf}"
MODEL_URL="${BRO_MODEL_URL:-https://huggingface.co/leeroy-jankins/bro/resolve/main/bro-3-4b-it-qat-Q4_K_M.gguf?download=true}"

mkdir -p "${MODEL_DIR}"
export BRO_LLM_PATH="${BRO_LLM_PATH:-${MODEL_DIR}/${MODEL_FILE}}"

if [[ ! -s "${BRO_LLM_PATH}" ]]; then
  echo "Bro model not found at ${BRO_LLM_PATH}. Downloading ${MODEL_FILE}..."
  tmp_file="${BRO_LLM_PATH}.part"
  rm -f "${tmp_file}"
  curl --fail --location --retry 5 --retry-delay 5 --connect-timeout 30 \
    --output "${tmp_file}" "${MODEL_URL}"
  mv "${tmp_file}" "${BRO_LLM_PATH}"
  echo "Bro model download complete."
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
