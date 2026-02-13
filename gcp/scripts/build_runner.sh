#!/usr/bin/env bash
set -euo pipefail

AX_TAG="${1:-latest}"
BASE_TAG="${2:-main}"

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
: "${RUNNER_IMAGE_NAME:?Set RUNNER_IMAGE_NAME in gcp/env}"
: "${BASE_IMAGE_NAME:?Set BASE_IMAGE_NAME in gcp/env}"

BASE_IMAGE="${AR_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${BASE_IMAGE_NAME}:${BASE_TAG}"

# Build wheel locally (setuptools-scm uses git to determine version)
echo "==> Building wheel"
rm -rf dist/
python3 -m build --wheel
echo "==> Built: $(ls dist/*.whl)"

gcloud config set project "${GCP_PROJECT_ID}" >/dev/null

gcloud builds submit \
  --config gcp/cloudbuild.runner.yaml \
  --substitutions=_AR_REGION="${AR_REGION}",_AR_REPO="${AR_REPO}",_IMAGE_NAME="${RUNNER_IMAGE_NAME}",_BASE_IMAGE="${BASE_IMAGE}",_AX_TAG="${AX_TAG}"
