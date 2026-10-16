#!/usr/bin/env bash
# Cloud Run entrypoint: serve on $PORT (8080 by default). Models are already in /app/models.
set -euo pipefail
exec streamlit run app/streamlit_app.py --server.port "${PORT:-8080}" --server.address 0.0.0.0 --server.headless true
