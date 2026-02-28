# GCP Deployment Guide

## 1. Initial Setup

```bash
# Copy and edit environment config
cp gcp/env.example gcp/env
# Edit gcp/env with your project details
```

## 2. Bootstrap GCP Resources

This creates buckets, Artifact Registry, service accounts, and IAM bindings:

```bash
cd gcp
make bootstrap
```

## 3. Setup Secrets

**Important:** Complete this step BEFORE building images.

### Deploy key for lean_benchmarks repo (required for base image build)

```bash
# Generate deploy key
ssh-keygen -t ed25519 -C "cloudbuild-leanbench" -f /tmp/leanbench_deploy_key -N ""

# Add the PUBLIC key to GitHub repo as deploy key:
cat /tmp/leanbench_deploy_key.pub
# Go to: lean_benchmarks repo > Settings > Deploy keys > Add deploy key

# Upload PRIVATE key to Secret Manager (from gcp/ directory):
cd gcp
cat /tmp/leanbench_deploy_key | make put-secret NAME=LEANBENCH_DEPLOY_KEY

# Clean up local key
rm /tmp/leanbench_deploy_key /tmp/leanbench_deploy_key.pub
```

### API keys for ax-prover (required for job runtime)

Sync your local `.env.secrets.gcp` to Secret Manager:

```bash
# From gcp/ directory
make sync-envfile
```

Or add individual secrets:

```bash
make put-secret NAME=ANTHROPIC_API_KEY
# paste value, Ctrl-D
```

## 4. Build Images

```bash
# From gcp/ directory
# Build base image (Lean + mathlib, slow, do once per mathlib version)
# Defaults to BENCH_REF=main (tags image as 'main')
make build-base

# Build runner image (ax-prover, fast, do per code change)
# Defaults to AX_TAG=latest, BASE_TAG=main
make build-runner
```

## 5. Submit Jobs

```bash
# Basic submission
bash gcp/scripts/submit_batch_job.sh \
  --dataset QuantumTheorem_v0 \
  --config-file configs/qt.yaml

# With experiment name (for grouping related runs)
# Job name format: ax-{name}-{dataset}-{timestamp}
bash gcp/scripts/submit_batch_job.sh \
  --dataset QuantumTheorem_v0 \
  --config-file configs/qt.yaml \
  --name baseline

# Multiple runs for statistics
bash gcp/scripts/submit_batch_job.sh --dataset QT --config-file cfg.yaml --name run1
bash gcp/scripts/submit_batch_job.sh --dataset QT --config-file cfg.yaml --name run2
bash gcp/scripts/submit_batch_job.sh --dataset QT --config-file cfg.yaml --name run3

# With extra args passed to ax-prover
bash gcp/scripts/submit_batch_job.sh --dataset QuantumTheorem_v0 --config-file configs/qt.yaml -- --verbose
```

Each job uploads a `manifest.json` with experiment metadata (dataset, config, git commit, timestamps, etc.) to the artifacts bucket for tracking.

```bash
bash gcp/scripts/submit_batch_job.sh --dataset quantum_theorems_v0_2_questions --config-file ../lean-benchmarks/.axiomatic/configs/prover-cloud-vm.yaml
```

## 6. Monitor Jobs

```bash
# List jobs
gcloud batch jobs list --location="${GCP_REGION}"

# Describe job
gcloud batch jobs describe JOB_NAME --location="${GCP_REGION}"

# View logs
gcloud logging read "resource.type=\"batch.googleapis.com/Job\" AND labels.job_uid=\"JOB_UID\"" --limit=100

# Artifacts are uploaded to:
# gs://${GCP_ARTIFACT_BUCKET}/runs/${JOB_NAME}/
```

[Jobs UI](https://console.cloud.google.com/batch/jobs?inv=1&invt=AbkWFg&project=ax-baku)

[Experiment Files](https://console.cloud.google.com/storage/browser/ax-experiment-artifacts/runs?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&inv=1&invt=AbkWFg&project=ax-baku)


## Troubleshooting

### Cloud Build: Permission denied for secret

```
Permission 'secretmanager.versions.access' denied for resource 'projects/.../secrets/LEANBENCH_DEPLOY_KEY/versions/latest'
```

**Cause:** Cloud Build uses the Compute Engine default service account to run builds, and it doesn't have access to secrets.

**Fix:** Grant secret access to the Compute Engine default SA:
```bash
PROJECT_NUMBER=$(gcloud projects describe $GCP_PROJECT_ID --format='value(projectNumber)')
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding LEANBENCH_DEPLOY_KEY \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/secretmanager.secretAccessor"
```

### Cloud Build: Secret not found

**Cause:** The secret hasn't been created yet.

**Fix:** Create the secret (see Step 3 above):
```bash
cd gcp
cat /tmp/leanbench_deploy_key | make put-secret NAME=LEANBENCH_DEPLOY_KEY
```
