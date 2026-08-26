FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    APP_FILE=app.py \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    BRO_MODEL_DIR=/app/models \
    BRO_MODEL_REPO=leeroy-jankins/bro \
    BRO_MODEL_FILE=gemma-3-4b-it-Q4_K_M.gguf \
    BRO_LLM_PATH=/app/models/gemma-3-4b-it-Q4_K_M.gguf

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        build-essential \
        cmake \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

# Download the exact Bro GGUF at image-build time so Azure startup has no
# Hugging Face network dependency. The resulting model becomes part of the image.
RUN mkdir -p "${BRO_MODEL_DIR}" \
    && python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

model_dir = Path(os.environ["BRO_MODEL_DIR"])
model_repo = os.environ["BRO_MODEL_REPO"]
model_file = os.environ["BRO_MODEL_FILE"]
target = Path(os.environ["BRO_LLM_PATH"])

print(f"BUILD: downloading {model_repo}/{model_file}", flush=True)
downloaded = Path(
    hf_hub_download(
        repo_id=model_repo,
        filename=model_file,
        local_dir=str(model_dir),
    )
)

if not downloaded.is_file() or downloaded.stat().st_size <= 0:
    raise RuntimeError(f"Bro model download failed or produced an empty file: {downloaded}")

if downloaded.resolve() != target.resolve():
    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded.replace(target)

print(f"BUILD: model ready at {target} ({target.stat().st_size} bytes)", flush=True)
PY

COPY . /app

# Bro's development config contains a Windows-only fallback path. The Azure image
# replaces that one assignment so MODEL_PATH honors the Linux BRO_LLM_PATH value.
RUN sed -i "s|^MODEL_PATH = .*|MODEL_PATH = os.getenv( 'BRO_LLM_PATH', r'/app/models/gemma-3-4b-it-Q4_K_M.gguf' )|" /app/config.py \
    && grep -F "MODEL_PATH = os.getenv" /app/config.py

RUN mkdir -p /app/stores/sqlite /app/logging \
    && test -s "${BRO_LLM_PATH}" \
    && chmod +x /app/startup.sh

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl --fail "http://127.0.0.1:${PORT}/_stcore/health" || exit 1

CMD ["/app/startup.sh"]
