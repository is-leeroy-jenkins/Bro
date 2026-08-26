FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    APP_FILE=app.py \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    BRO_MODEL_DIR=/app/models \
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

COPY . /app

# Bro's development config contains a Windows-only fallback path. The Azure image
# makes MODEL_PATH honor BRO_LLM_PATH while allowing the GGUF itself to be supplied
# separately after deployment.
RUN sed -i "s|^MODEL_PATH = .*|MODEL_PATH = os.getenv( 'BRO_LLM_PATH', r'/app/models/gemma-3-4b-it-Q4_K_M.gguf' )|" /app/config.py \
    && grep -F "MODEL_PATH = os.getenv" /app/config.py

RUN mkdir -p /app/models /app/stores/sqlite /app/logging \
    && chmod +x /app/startup.sh

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl --fail "http://127.0.0.1:${PORT}/_stcore/health" || exit 1

CMD ["/app/startup.sh"]
