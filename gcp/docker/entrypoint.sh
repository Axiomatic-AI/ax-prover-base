#!/usr/bin/env bash
set -euo pipefail

# If configured, fetch .env.secrets from Secret Manager
if [[ -n "${ENV_SECRETS_SECRET_NAME:-}" ]]; then
  python3 /opt/bootstrap_envsecrets.py
fi

# Run exactly what you typed (e.g. ax-prover experiment QuantumTheorem_v0)
exec "$@"
