#!/usr/bin/env bash
# Deploy the dashboard to Google Cloud Run (public URL, scales to zero).
#   PROJECT=<gcp-project-id> [REGION=asia-south1] scripts/deploy_cloudrun.sh
# Needs: gcloud logged in (gcloud auth login), billing enabled on the project,
#        build context staged by `python scripts/publish.py cloudrun`.
set -euo pipefail
G=${GCLOUD:-$HOME/google-cloud-sdk/bin/gcloud}
PROJECT=${PROJECT:?set PROJECT=<gcp project id>}
REGION=${REGION:-asia-south1}
SERVICE=${SERVICE:-itantra-stt}
REPO=itantra
TAG=$(git rev-parse --short HEAD)
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/stt:$TAG"

[ -f build/cloudrun/Dockerfile ] || { echo "run: python scripts/publish.py cloudrun"; exit 1; }
"$G" config set project "$PROJECT" >/dev/null
"$G" services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
"$G" artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 \
  || "$G" artifacts repositories create "$REPO" --repository-format=docker --location "$REGION" \
       --description "iTantra STT images (private)"

# build in Cloud Build (no local Docker needed); the image stays private in Artifact Registry
"$G" builds submit build/cloudrun --tag "$IMAGE" --timeout=3600s

# 4 GiB / 2 vCPU: one engine (~1.7 GB) for all visitors; scale to zero when idle; 1 instance max (cost cap);
# session affinity keeps a visitor's websocket on the same instance; CPU boost speeds the cold start
"$G" run deploy "$SERVICE" --image "$IMAGE" --region "$REGION" --allow-unauthenticated \
  --memory 4Gi --cpu 2 --concurrency 20 --min-instances 0 --max-instances 1 \
  --timeout 3600 --session-affinity --cpu-boost --port 8080

"$G" run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)'
