#!/usr/bin/env bash
set -euo pipefail

# Load env file (required)
ENV_FILE="gcp/env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found. Run: cp gcp/env.example gcp/env && edit gcp/env" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
: "${GCP_REGION:?Set GCP_REGION}"
: "${GCP_CONFIG_BUCKET:?Set GCP_CONFIG_BUCKET}"
: "${GCP_ARTIFACT_BUCKET:?Set GCP_ARTIFACT_BUCKET}"
: "${AR_REPO:?Set AR_REPO}"
: "${AR_REGION:?Set AR_REGION}"
: "${BATCH_RUNNER_SA_NAME:?Set BATCH_RUNNER_SA_NAME}"
: "${BATCH_RUNNER_SA_EMAIL:?Set BATCH_RUNNER_SA_EMAIL}"

echo "==> Using project: ${GCP_PROJECT_ID}"
gcloud config set project "${GCP_PROJECT_ID}" >/dev/null

echo "==> Enabling required APIs"
gcloud services enable \
  artifactregistry.googleapis.com \
  batch.googleapis.com \
  compute.googleapis.com \
  iam.googleapis.com \
  logging.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  run.googleapis.com \
  >/dev/null

echo "==> Creating buckets (if missing)"
# Config bucket (read-mostly)
if ! gsutil ls -b "gs://${GCP_CONFIG_BUCKET}" >/dev/null 2>&1; then
  gsutil mb -p "${GCP_PROJECT_ID}" -l "${GCP_REGION}" -b on "gs://${GCP_CONFIG_BUCKET}"
fi

# Artifacts bucket (write)
if ! gsutil ls -b "gs://${GCP_ARTIFACT_BUCKET}" >/dev/null 2>&1; then
  gsutil mb -p "${GCP_PROJECT_ID}" -l "${GCP_REGION}" -b on "gs://${GCP_ARTIFACT_BUCKET}"
fi

echo "==> (Optional) Enabling versioning on config bucket"
gsutil versioning set on "gs://${GCP_CONFIG_BUCKET}" >/dev/null || true

echo "==> Creating Artifact Registry repo (if missing)"
if ! gcloud artifacts repositories describe "${AR_REPO}" --location "${AR_REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${AR_REPO}" \
    --repository-format=docker \
    --location="${AR_REGION}" \
    --description="Axiomatic experiment runner images" >/dev/null
fi

echo "==> Creating service account (if missing)"
if ! gcloud iam service-accounts describe "${BATCH_RUNNER_SA_EMAIL}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${BATCH_RUNNER_SA_NAME}" \
    --display-name="Batch runner for ax-prover experiments" >/dev/null
fi

echo "==> Granting IAM roles to runner service account"

# Batch job execution (required for jobs to report state)
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${BATCH_RUNNER_SA_EMAIL}" \
  --role="roles/batch.agentReporter" >/dev/null

# Artifact Registry read (to pull container images)
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${BATCH_RUNNER_SA_EMAIL}" \
  --role="roles/artifactregistry.reader" >/dev/null

# Secret access
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${BATCH_RUNNER_SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor" >/dev/null

echo "==> Granting IAM roles to Cloud Build service accounts"
PROJECT_NUMBER="$(gcloud projects describe "${GCP_PROJECT_ID}" --format='value(projectNumber)')"
CB_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
# Cloud Build uses the Compute Engine default SA to run build steps
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Secret access (for LEANBENCH_DEPLOY_KEY) - grant to both SAs
gcloud secrets add-iam-policy-binding LEANBENCH_DEPLOY_KEY \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/secretmanager.secretAccessor" >/dev/null 2>&1 || true
gcloud secrets add-iam-policy-binding LEANBENCH_DEPLOY_KEY \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/secretmanager.secretAccessor" >/dev/null 2>&1 || true

# Artifact Registry write (to push images)
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/artifactregistry.writer" >/dev/null
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/artifactregistry.writer" >/dev/null

# Logs
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
  --member="serviceAccount:${BATCH_RUNNER_SA_EMAIL}" \
  --role="roles/logging.logWriter" >/dev/null

# Storage: configs bucket read
gsutil iam ch \
  "serviceAccount:${BATCH_RUNNER_SA_EMAIL}:objectViewer" \
  "gs://${GCP_CONFIG_BUCKET}" >/dev/null || true

# Storage: artifacts bucket write
gsutil iam ch \
  "serviceAccount:${BATCH_RUNNER_SA_EMAIL}:objectAdmin" \
  "gs://${GCP_ARTIFACT_BUCKET}" >/dev/null || true

# If LeanSearch is Cloud Run, grant invoker (optional; safe to skip until you fill vars)
if [[ -n "${LEANSEARCH_SERVICE:-}" && -n "${LEANSEARCH_SERVICE_REGION:-}" ]]; then
  echo "==> Granting Cloud Run invoker for LeanSearch (${LEANSEARCH_SERVICE})"
  gcloud run services add-iam-policy-binding "${LEANSEARCH_SERVICE}" \
    --region "${LEANSEARCH_SERVICE_REGION}" \
    --member="serviceAccount:${BATCH_RUNNER_SA_EMAIL}" \
    --role="roles/run.invoker" >/dev/null
else
  echo "==> Skipping Cloud Run invoker grant (LEANSEARCH_SERVICE / REGION not set)"
fi

echo "✅ Bootstrap complete."
echo "Next: put secrets with:  make put-secret NAME=LANGSMITH_API_KEY  (then paste value, ctrl-d)"
