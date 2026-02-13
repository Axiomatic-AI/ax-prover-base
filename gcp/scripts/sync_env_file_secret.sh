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
: "${ENV_SECRETS_FILE:=.env.secrets.gcp}"
: "${ENV_SECRETS_SECRET_NAME:=AX_ENV_SECRETS}"

gcloud config set project "${GCP_PROJECT_ID}" >/dev/null

if [[ ! -f "${ENV_SECRETS_FILE}" ]]; then
  echo "Missing ${ENV_SECRETS_FILE}. Create it from .env.secrets.example first." >&2
  exit 1
fi

# Create if missing
if ! gcloud secrets describe "${ENV_SECRETS_SECRET_NAME}" >/dev/null 2>&1; then
  gcloud secrets create "${ENV_SECRETS_SECRET_NAME}" --replication-policy="automatic" >/dev/null
fi

# Upload new version from file
gcloud secrets versions add "${ENV_SECRETS_SECRET_NAME}" --data-file="${ENV_SECRETS_FILE}" >/dev/null
echo "✅ Synced ${ENV_SECRETS_FILE} -> Secret Manager secret ${ENV_SECRETS_SECRET_NAME}"
