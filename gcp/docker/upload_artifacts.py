#!/usr/bin/env python3
"""Upload job artifacts to GCS after ax-prover run."""

import json
import os
from datetime import UTC, datetime

from google.cloud import storage


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """Parse gs://bucket/prefix into (bucket, prefix)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}")
    rest = uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
    return bucket, prefix


def upload_file(bucket, base_prefix: str, local_path: str, remote_name: str) -> None:
    """Upload a file if it exists."""
    if not os.path.exists(local_path):
        print(f"Skipping {local_path} (not found)")
        return
    blob = bucket.blob(f"{base_prefix}/{remote_name}")
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} -> {remote_name}")


def upload_json(bucket, base_prefix: str, data: dict, remote_name: str) -> None:
    """Upload a dict as JSON."""
    blob = bucket.blob(f"{base_prefix}/{remote_name}")
    blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")
    print(f"Uploaded {remote_name}")


def build_manifest() -> dict:
    """Build manifest from environment variables."""
    return {
        "job_name": os.environ.get("AX_JOB_NAME", ""),
        "experiment_name": os.environ.get("AX_EXPERIMENT_NAME", ""),
        "dataset": os.environ.get("AX_DATASET", ""),
        "image_uri": os.environ.get("AX_IMAGE_URI", ""),
        "git_commit": os.environ.get("AX_GIT_COMMIT", ""),
        "config_file": os.environ.get("AX_CONFIG_FILE", ""),
        "max_concurrency": os.environ.get("AX_MAX_CONCURRENCY", ""),
        "machine_type": os.environ.get("AX_MACHINE_TYPE", ""),
        "submitted_at": os.environ.get("AX_SUBMITTED_AT", ""),
        "completed_at": datetime.now(UTC).isoformat(),
    }


def main():
    out_prefix = os.environ.get("OUT_PREFIX")
    folder = os.environ.get("LEAN_FOLDER", "/opt/lean_benchmarks")

    if not out_prefix:
        print("OUT_PREFIX not set, skipping upload")
        return

    bucket_name, base = parse_gs_uri(out_prefix)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Upload manifest
    manifest = build_manifest()
    upload_json(bucket, base, manifest, "manifest.json")

    # Upload artifacts
    upload_file(bucket, base, "/tmp/out/run.log", "run.log")
    upload_file(bucket, base, "/tmp/out/axiomatic.tgz", "axiomatic.tgz")
    upload_file(bucket, base, f"{folder}/.axiomatic/used_config.yaml", "used_config.yaml")

    print(f"Artifacts: {out_prefix}")


if __name__ == "__main__":
    main()
