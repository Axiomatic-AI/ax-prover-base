import os

from google.cloud import secretmanager

project = os.environ["GCP_PROJECT_ID"]
secret_name = os.environ.get("ENV_SECRETS_SECRET_NAME", "AX_ENV_SECRETS")

client = secretmanager.SecretManagerServiceClient()
name = f"projects/{project}/secrets/{secret_name}/versions/latest"
payload = client.access_secret_version(request={"name": name}).payload.data.decode("utf-8")

# Write into CWD; ax-prover will load it from project root or CWD :contentReference[oaicite:2]{index=2}
with open(".env.secrets", "w", encoding="utf-8") as f:
    f.write(payload)

print("Wrote .env.secrets from Secret Manager")
