FROM python:3.11-slim

# build-essential: IndicTransToolkit compiles a small Cython extension; libsndfile for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
RUN pip install --no-cache-dir --no-deps . && chmod +x start.sh

USER user
ENV ITANTRA_CLOUD=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface
EXPOSE 7860
CMD ["./start.sh"]
