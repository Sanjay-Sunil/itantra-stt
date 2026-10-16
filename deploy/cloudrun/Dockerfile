# Cloud Run image: dashboard + built INT8 models baked in (private Artifact Registry image; no download at start)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY . .
RUN pip install --no-cache-dir --no-deps . && chmod +x start.sh

ENV ITANTRA_CLOUD=1 PYTHONUNBUFFERED=1 INDICSTT_MODELS_DIR=/app/models
CMD ["./start.sh"]
