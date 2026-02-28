#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Load environment
# =============================================================================

ENV_FILE="gcp/env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found. Run: cp gcp/env.example gcp/env && edit gcp/env" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

# =============================================================================
# Usage
# =============================================================================

usage() {
  cat <<EOF
Usage:
  bash gcp/scripts/submit_batch_job.sh \\
    --dataset DATASET \\
    --config-file PATH.yaml \\
    [--name EXPERIMENT_NAME] \\
    [--image IMAGE_URI | --image-tag TAG] \\
    [--experiment-prefix PREFIX] \\
    [--max-concurrency N] \\
    [--folder /opt/lean_benchmarks] \\
    [--machine-type n2-standard-16] \\
    [--region REGION] \\
    [--extra-args "raw extra args"] \\
    [-- <extra args passed through>]

Options:
  --name    Short experiment name for grouping runs (e.g., "baseline", "v2")
            Job name format: ax-{name}-{dataset}-{timestamp}
  --region  GCP region for the job. Defaults to "auto" which randomly selects
            from available regions to distribute load across quota pools.

Examples:
  bash gcp/scripts/submit_batch_job.sh --dataset QuantumTheorem_v0 --config-file cfg.yaml
  bash gcp/scripts/submit_batch_job.sh --dataset QuantumTheorem_v0 --config-file cfg.yaml --name baseline
  bash gcp/scripts/submit_batch_job.sh --dataset QuantumTheorem_v0 --config-file cfg.yaml --region auto
  bash gcp/scripts/submit_batch_job.sh --dataset QuantumTheorem_v0 --config-file cfg.yaml -- --verbose
EOF
}

# =============================================================================
# Parse arguments
# =============================================================================

IMAGE_URI="${RUNNER_IMAGE_URI:-}"
IMAGE_TAG=""
DATASET=""
CONFIG_FILE=""
EXPERIMENT_NAME=""
EXPERIMENT_PREFIX=""
MAX_CONCURRENCY="4"
FOLDER="/opt/lean_benchmarks"
MACHINE_TYPE="e2-standard-32"
EXTRA_ARGS_RAW=""
REGION=""

# Regions to rotate through when --region=auto (all GCP Batch supported regions)
AVAILABLE_REGIONS=(
  # Europe
  "europe-west1"      # Belgium
  "europe-west2"      # London
  "europe-west3"      # Frankfurt
  "europe-west4"      # Netherlands
  "europe-west6"      # Zurich
  "europe-west8"      # Milan
  "europe-west9"      # Paris
  "europe-north1"     # Finland
  "europe-central2"   # Warsaw
  # US
  "us-central1"       # Iowa
  "us-east1"          # South Carolina
  "us-east4"          # Virginia
  "us-east5"          # Columbus
  "us-west1"          # Oregon
  "us-west2"          # Los Angeles
  "us-west3"          # Salt Lake City
  "us-west4"          # Las Vegas
  "us-south1"         # Dallas
  # Asia
  # "asia-east1"        # Taiwan
  # "asia-east2"        # Hong Kong
  "asia-northeast1"   # Tokyo
  "asia-northeast2"   # Osaka
  "asia-northeast3"   # Seoul
  "asia-south1"       # Mumbai
  "asia-south2"       # Delhi
  "asia-southeast1"   # Singapore
  "asia-southeast2"   # Jakarta
  # Other
  "australia-southeast1"  # Sydney
  "australia-southeast2"  # Melbourne
  "northamerica-northeast1"  # Montreal
  "northamerica-northeast2"  # Toronto
  "southamerica-east1"       # Sao Paulo
)

PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)             IMAGE_URI="$2"; shift 2 ;;
    --image-tag)         IMAGE_TAG="$2"; shift 2 ;;
    --dataset)           DATASET="$2"; shift 2 ;;
    --config-file)       CONFIG_FILE="$2"; shift 2 ;;
    --name)              EXPERIMENT_NAME="$2"; shift 2 ;;
    --experiment-prefix) EXPERIMENT_PREFIX="$2"; shift 2 ;;
    --max-concurrency)   MAX_CONCURRENCY="$2"; shift 2 ;;
    --folder)            FOLDER="$2"; shift 2 ;;
    --machine-type)      MACHINE_TYPE="$2"; shift 2 ;;
    --region)            REGION="$2"; shift 2 ;;
    --extra-args)        EXTRA_ARGS_RAW="$2"; shift 2 ;;
    -h|--help)           usage; exit 0 ;;
    --)                  shift; PASSTHRU=("$@"); break ;;
    *)                   echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# =============================================================================
# Validate required env vars and args
# =============================================================================

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID in gcp/env}"
: "${GCP_REGION:?Set GCP_REGION in gcp/env}"
: "${BATCH_RUNNER_SA_EMAIL:?Set BATCH_RUNNER_SA_EMAIL in gcp/env}"
: "${ENV_SECRETS_SECRET_NAME:?Set ENV_SECRETS_SECRET_NAME in gcp/env}"
: "${GCP_ARTIFACT_BUCKET:?Set GCP_ARTIFACT_BUCKET in gcp/env}"
: "${AR_REGION:?Set AR_REGION in gcp/env}"
: "${AR_REPO:?Set AR_REPO in gcp/env}"
: "${RUNNER_IMAGE_NAME:?Set RUNNER_IMAGE_NAME in gcp/env}"
: "${DEFAULT_RUNNER_TAG:?Set DEFAULT_RUNNER_TAG in gcp/env}"

[[ -n "${DATASET}" ]] || { echo "ERROR: Missing --dataset"; usage; exit 1; }
[[ -f "${CONFIG_FILE}" ]] || { echo "ERROR: Missing/invalid --config-file: ${CONFIG_FILE}"; exit 1; }

# =============================================================================
# Derive values
# =============================================================================

# Image URI
if [[ -z "${IMAGE_URI}" ]]; then
  TAG="${IMAGE_TAG:-${DEFAULT_RUNNER_TAG}}"
  IMAGE_URI="${AR_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${RUNNER_IMAGE_NAME}:${TAG}"
fi

# Job name (must be lowercase, start with letter, max 63 chars)
SUBMITTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# Timestamp without dashes or year (MMDDHHMMSS)
ts="$(date -u +%m%d%H%M%S)"
# Sanitize: lowercase, remove all non-alphanumeric (no dashes within components)
ds_sanitized="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+//g')"
config_name="$(basename "${CONFIG_FILE}" .yaml)"
config_sanitized="$(echo "${config_name}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+//g')"
# Format: ax-[TIMESTAMP]-[DATASET]-[CONFIGNAME]-[EXPERIMENTNAME] (dashes only as separators)
if [[ -n "${EXPERIMENT_NAME}" ]]; then
  name_sanitized="$(echo "${EXPERIMENT_NAME}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+//g')"
  JOB_NAME="${PREFIX:-ax}-${ts}-${ds_sanitized}-${config_sanitized}-${name_sanitized}"
else
  JOB_NAME="${PREFIX:-ax}-${ts}-${ds_sanitized}-${config_sanitized}"
fi
# GCP Batch job name limit is 63 chars, must start with letter
JOB_NAME="$(echo "${JOB_NAME}" | cut -c1-63)"
[[ "${JOB_NAME}" =~ ^[a-z] ]] || JOB_NAME="j${JOB_NAME:1}"

# Select region
if [[ -z "${REGION}" || "${REGION}" == "auto" ]]; then
  # Randomly select a region (default behavior)
  region_idx=$((RANDOM % ${#AVAILABLE_REGIONS[@]}))
  REGION="${AVAILABLE_REGIONS[$region_idx]}"
  echo "Auto-selected region: ${REGION}"
fi

# Experiment prefix defaults to job name
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-${JOB_NAME}}"

# Git commit for manifest
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")"

# Output location
OUT_PREFIX="gs://${GCP_ARTIFACT_BUCKET}/runs/${JOB_NAME}"

# =============================================================================
# Pre-process config file to resolve imports
# =============================================================================

# Generate unique name to avoid conflicts
RESOLVED_CONFIG_NAME="resolved_$(date +%s)_$$"
RESOLVED_CONFIG=".axiomatic/${RESOLVED_CONFIG_NAME}.yaml"
trap 'rm -f "${RESOLVED_CONFIG}"' EXIT

echo "Resolving config imports from ${CONFIG_FILE}..."
# Run ax-prover without a command to trigger config parsing and saving, then show help
ax-prover --config "${CONFIG_FILE}" --save-config "${RESOLVED_CONFIG_NAME}" >/dev/null 2>&1

if [[ ! -f "${RESOLVED_CONFIG}" ]]; then
  echo "ERROR: Failed to resolve config file imports" >&2
  exit 1
fi

echo "Using resolved config: ${RESOLVED_CONFIG}"

# =============================================================================
# Encode config and extra args as base64
# =============================================================================

CONFIG_B64="$(base64 -w 0 < "${RESOLVED_CONFIG}")"

# Process passthrough args (safely shell-quote them)
PASSTHRU_STR=""
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then
  PASSTHRU_STR="$(python3 -c "import shlex,sys; print(' '.join(shlex.quote(a) for a in sys.argv[1:]))" "${PASSTHRU[@]}")"
fi

# Combine extra args
EXTRA_COMBINED="${EXTRA_ARGS_RAW}"
[[ -z "${PASSTHRU_STR}" ]] || EXTRA_COMBINED="${EXTRA_COMBINED:+${EXTRA_COMBINED} }${PASSTHRU_STR}"
EXTRA_B64="$(printf '%s' "${EXTRA_COMBINED}" | base64 -w 0)"

# =============================================================================
# Build the container command (runs inside the container)
# =============================================================================

read -r -d '' CONTAINER_SCRIPT <<'BASH' || true
set -euo pipefail

cd "$LEAN_FOLDER"

# Decode config
echo "$AX_CONFIG_B64" | base64 -d > /tmp/config.yaml

# Decode extra args
EXTRA="$(echo "$AX_EXTRA_ARGS_B64" | base64 -d)"

# Prepare output directory
mkdir -p /tmp/out

# Run ax-prover
set +e
eval "ax-prover --config /tmp/config.yaml --save-config used_config experiment '$AX_DATASET' \
  --folder '$LEAN_FOLDER' \
  --max-concurrency '$AX_MAX_CONCURRENCY' \
  --experiment-prefix '$AX_EXPERIMENT_PREFIX' \
  $EXTRA" 2>&1 | tee /tmp/out/run.log
EXIT_CODE=${PIPESTATUS[0]}
set -e

# Archive all git-modified and untracked files
cd "$LEAN_FOLDER" && git status --porcelain | awk '{print $2}' | tar -czf /tmp/out/axiomatic.tgz -T - 2>/dev/null || true

# Upload artifacts to GCS
python3 /opt/upload_artifacts.py || true

exit "$EXIT_CODE"
BASH

# JSON-escape the script
CONTAINER_CMD="$(echo "${CONTAINER_SCRIPT}" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")"

# =============================================================================
# Generate job JSON
# =============================================================================

gcloud config set project "${GCP_PROJECT_ID}" >/dev/null

JOB_JSON="$(mktemp)"
cat > "${JOB_JSON}" <<EOF
{
  "taskGroups": [{
    "taskCount": 1,
    "parallelism": 1,
    "taskSpec": {
      "runnables": [{
        "container": {
          "imageUri": "${IMAGE_URI}",
          "commands": ["bash", "-lc", ${CONTAINER_CMD}]
        },
        "environment": {
          "variables": {
            "GCP_PROJECT_ID": "${GCP_PROJECT_ID}",
            "ENV_SECRETS_SECRET_NAME": "${ENV_SECRETS_SECRET_NAME}",
            "AX_CONFIG_B64": "${CONFIG_B64}",
            "AX_EXTRA_ARGS_B64": "${EXTRA_B64}",
            "AX_DATASET": "${DATASET}",
            "AX_MAX_CONCURRENCY": "${MAX_CONCURRENCY}",
            "AX_EXPERIMENT_PREFIX": "${EXPERIMENT_PREFIX}",
            "LEAN_FOLDER": "${FOLDER}",
            "OUT_PREFIX": "${OUT_PREFIX}",
            "AX_JOB_NAME": "${JOB_NAME}",
            "AX_EXPERIMENT_NAME": "${EXPERIMENT_NAME}",
            "AX_IMAGE_URI": "${IMAGE_URI}",
            "AX_GIT_COMMIT": "${GIT_COMMIT}",
            "AX_CONFIG_FILE": "$(basename "${CONFIG_FILE}")",
            "AX_MACHINE_TYPE": "${MACHINE_TYPE}",
            "AX_SUBMITTED_AT": "${SUBMITTED_AT}",
            "AX_REGION": "${REGION}"
          }
        }
      }],
      "computeResource": {
        "cpuMilli": 32000,
        "memoryMib": 128000
      }
    }
  }],
  "allocationPolicy": {
    "instances": [{"policy": {"machineType": "${MACHINE_TYPE}"}}],
    "serviceAccount": {"email": "${BATCH_RUNNER_SA_EMAIL}"}
  },
  "logsPolicy": {"destination": "CLOUD_LOGGING"}
}
EOF

# =============================================================================
# Submit job
# =============================================================================

echo "Submitting Batch job: ${JOB_NAME}"
gcloud batch jobs submit "${JOB_NAME}" \
  --location="${REGION}" \
  --config="${JOB_JSON}"

cat <<EOF

✅ Submitted: ${JOB_NAME}
   Region:     ${REGION}
   Experiment: ${EXPERIMENT_NAME:-"(none)"}
   Dataset:    ${DATASET}
   Image:      ${IMAGE_URI}
   Artifacts:  ${OUT_PREFIX}
EOF
