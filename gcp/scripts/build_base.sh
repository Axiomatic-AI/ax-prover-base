#!/usr/bin/env bash
set -euo pipefail

BENCH_REF="${1:-main}"

# Load env file (required) - scripts run from repo root via Makefile
ENV_FILE="gcp/env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found. Run: cp gcp/env.example gcp/env && edit gcp/env" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID in gcp/env}"
: "${AR_REGION:?Set AR_REGION in gcp/env}"
: "${AR_REPO:?Set AR_REPO in gcp/env}"
: "${BASE_IMAGE_NAME:?Set BASE_IMAGE_NAME in gcp/env}"
: "${LEANBENCH_REPO_SSH:?Set LEANBENCH_REPO_SSH in gcp/env}"

gcloud config set project "${GCP_PROJECT_ID}" >/dev/null

gcloud builds submit \
  --config gcp/cloudbuild.base.yaml \
  --substitutions=_AR_REGION="${AR_REGION}",_AR_REPO="${AR_REPO}",_IMAGE_NAME="${BASE_IMAGE_NAME}",_BENCH_REPO_SSH="${LEANBENCH_REPO_SSH}",_BENCH_REF="${BENCH_REF}"
