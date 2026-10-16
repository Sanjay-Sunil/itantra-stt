#!/usr/bin/env bash
# Space entrypoint: download the built models (private repo, HF_TOKEN secret), then serve the dashboard.
set -euo pipefail
python scripts/fetch_models.py --prepared "${MODELS_REPO:-__MODELS_REPO__}"
exec streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
