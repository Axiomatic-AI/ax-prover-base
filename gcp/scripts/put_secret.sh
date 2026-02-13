#!/usr/bin/env bash
set -euo pipefail

# Load env file (required) - scripts run from repo root via Makefile
ENV_FILE="gcp/env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found. Run: cp gcp/env.example gcp/env && edit gcp/env" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID in gcp/env}"

NAME="${1:?Usage: put_secret.sh SECRET_NAME (reads secret from stdin)}"

gcloud config set project "${GCP_PROJECT_ID}" >/dev/null

# Create if missing
if ! gcloud secrets describe "${NAME}" >/dev/null 2>&1; then
  gcloud secrets create "${NAME}" --replication-policy="automatic" >/dev/null
fi

echo "Paste secret value, then press Ctrl-D:"
# Read stdin and add as a new version
gcloud secrets versions add "${NAME}" --data-file=- >/dev/null

echo "✅ Secret updated: ${NAME}"
