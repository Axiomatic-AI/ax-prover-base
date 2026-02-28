#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Stop/delete GCP Batch jobs matching a pattern
# =============================================================================

ENV_FILE="gcp/env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

# =============================================================================
# Usage
# =============================================================================

usage() {
  cat <<EOF
Usage: bash gcp/scripts/stop_jobs.sh [OPTIONS] <pattern>

Stop/delete GCP Batch jobs whose names contain the given pattern.

Arguments:
  pattern       String to match in job names (case-insensitive)

Options:
  --dry-run          List matching jobs without deleting them
  --force            Skip confirmation prompt
  --state STATE      Only match jobs in specific state (RUNNING, SUCCEEDED, FAILED, etc.)
  --exclude-succeeded  Exclude jobs that succeeded (delete RUNNING, FAILED, QUEUED, etc.)
  -h, --help         Show this help

Examples:
  bash gcp/scripts/stop_jobs.sh sonnetbase                        # Delete all jobs with 'sonnetbase'
  bash gcp/scripts/stop_jobs.sh --dry-run sonnetbase              # Preview what would be deleted
  bash gcp/scripts/stop_jobs.sh --exclude-succeeded sonnetbase    # Delete non-succeeded jobs only
  bash gcp/scripts/stop_jobs.sh --state RUNNING flash             # Delete only running 'flash' jobs
EOF
}

# =============================================================================
# Parse arguments
# =============================================================================

DRY_RUN=false
FORCE=false
STATE_FILTER=""
EXCLUDE_SUCCEEDED=false
PATTERN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)           DRY_RUN=true; shift ;;
    --force)             FORCE=true; shift ;;
    --state)             STATE_FILTER="$2"; shift 2 ;;
    --exclude-succeeded) EXCLUDE_SUCCEEDED=true; shift ;;
    -h|--help)           usage; exit 0 ;;
    -*)                  echo "Unknown option: $1"; usage; exit 1 ;;
    *)                   PATTERN="$1"; shift ;;
  esac
done

if [[ -z "${PATTERN}" ]]; then
  echo "ERROR: Missing pattern argument"
  usage
  exit 1
fi

# =============================================================================
# Regions to search (same as submit script)
# =============================================================================

REGIONS=(
  "europe-west1" "europe-west2" "europe-west3" "europe-west4" "europe-west6"
  "europe-west8" "europe-west9" "europe-north1" "europe-central2"
  "us-central1" "us-east1" "us-east4" "us-east5" "us-west1" "us-west2"
  "us-west3" "us-west4" "us-south1"
  "asia-east1" "asia-east2" "asia-northeast1" "asia-northeast2" "asia-northeast3"
  "asia-south1" "asia-south2" "asia-southeast1" "asia-southeast2"
  "australia-southeast1" "australia-southeast2"
  "northamerica-northeast1" "northamerica-northeast2" "southamerica-east1"
)

# =============================================================================
# Find matching jobs across all regions
# =============================================================================

echo "Searching for jobs matching '${PATTERN}' across ${#REGIONS[@]} regions..."

gcloud config set project "${GCP_PROJECT_ID}" >/dev/null

declare -a MATCHING_JOBS=()

for region in "${REGIONS[@]}"; do
  # List jobs in this region, filter by pattern (case-insensitive)
  while IFS=$'\t' read -r name state; do
    [[ -z "${name}" ]] && continue

    # Case-insensitive pattern match
    if echo "${name}" | grep -qi "${PATTERN}"; then
      # Apply state filter if specified
      if [[ -n "${STATE_FILTER}" ]] && [[ "${state}" != "${STATE_FILTER}" ]]; then
        continue
      fi
      # Exclude succeeded jobs if requested
      if [[ "${EXCLUDE_SUCCEEDED}" == "true" ]] && [[ "${state}" == "SUCCEEDED" ]]; then
        continue
      fi
      MATCHING_JOBS+=("${name}|${state}|${region}")
    fi
  done < <(gcloud batch jobs list --location="${region}" \
    --format="value(name.basename(),status.state)" 2>/dev/null || true)
done

# =============================================================================
# Display results
# =============================================================================

if [[ ${#MATCHING_JOBS[@]} -eq 0 ]]; then
  echo "No jobs found matching '${PATTERN}'"
  exit 0
fi

echo ""
echo "Found ${#MATCHING_JOBS[@]} job(s) matching '${PATTERN}':"
echo ""
printf "%-60s %-12s %s\n" "JOB NAME" "STATE" "REGION"
printf "%-60s %-12s %s\n" "--------" "-----" "------"

for job_info in "${MATCHING_JOBS[@]}"; do
  IFS='|' read -r name state location <<< "${job_info}"
  printf "%-60s %-12s %s\n" "${name}" "${state}" "${location}"
done
echo ""

# =============================================================================
# Delete jobs
# =============================================================================

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[DRY RUN] Would delete ${#MATCHING_JOBS[@]} job(s)"
  exit 0
fi

if [[ "${FORCE}" != "true" ]]; then
  read -r -p "Delete these ${#MATCHING_JOBS[@]} job(s)? [y/N] " confirm
  if [[ ! "${confirm}" =~ ^[Yy] ]]; then
    echo "Aborted."
    exit 0
  fi
fi

echo ""
echo "Deleting jobs..."

deleted=0
failed=0

for job_info in "${MATCHING_JOBS[@]}"; do
  IFS='|' read -r name state location <<< "${job_info}"
  echo -n "  Deleting ${name} (${location})... "

  if gcloud batch jobs delete "${name}" --location="${location}" --quiet 2>/dev/null; then
    echo "done"
    ((deleted++))
  else
    echo "failed"
    ((failed++))
  fi
done

echo ""
echo "Deleted: ${deleted}, Failed: ${failed}"
