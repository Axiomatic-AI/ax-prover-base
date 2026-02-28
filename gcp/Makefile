SHELL := /bin/bash

# Get repo root (parent of gcp/)
REPO_ROOT := $(shell cd "$(dir $(lastword $(MAKEFILE_LIST)))/.." && pwd)

.PHONY: help bootstrap put-secret sync-envfile build-base build-runner

help:
	@echo "Run from gcp/ directory: cd gcp && make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  make bootstrap                        # enable APIs, create buckets, AR repo, service account + IAM"
	@echo "  make put-secret NAME=...              # set a Secret Manager secret from stdin"
	@echo "  make sync-envfile                     # sync .env.secrets.gcp to Secret Manager"
	@echo "  make build-base                       # build base image (default: BENCH_REF=main)"
	@echo "  make build-runner                     # build runner image (defaults: AX_TAG=latest, BASE_TAG=main)"

bootstrap:
	@cd "$(REPO_ROOT)" && bash gcp/scripts/bootstrap.sh

put-secret:
	@cd "$(REPO_ROOT)" && bash gcp/scripts/put_secret.sh "$(NAME)"

sync-envfile:
	@cd "$(REPO_ROOT)" && bash gcp/scripts/sync_env_file_secret.sh

build-base:
	@cd "$(REPO_ROOT)" && bash gcp/scripts/build_base.sh "$(or $(BENCH_REF),main)"

build-runner:
	@cd "$(REPO_ROOT)" && bash gcp/scripts/build_runner.sh "$(or $(AX_TAG),latest)" "$(or $(BASE_TAG),main)"
