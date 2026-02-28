#!/usr/bin/env bash
set -euo pipefail

# Test Docker images locally before pushing to Cloud Build
# Usage: bash gcp/scripts/test_local.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source gcp/env

echo "==> Building Python wheel..."
uv build

echo ""
echo "==> Cloning lean_benchmarks (if needed)..."
BENCH_DIR="${PROJECT_ROOT}/.build_cache/lean_benchmarks"
if [ ! -d "$BENCH_DIR" ]; then
  mkdir -p "$(dirname "$BENCH_DIR")"
  git clone "${LEANBENCH_REPO_SSH}" "$BENCH_DIR"
fi
cd "$BENCH_DIR"
git fetch origin
git checkout main
git pull
BENCH_COMMIT=$(git rev-parse --short HEAD)

cd "$PROJECT_ROOT"

echo ""
echo "==> Preparing build context..."
# Copy lean_benchmarks to build context (Docker COPY doesn't follow symlinks)
if [ -d lean_benchmarks ] || [ -L lean_benchmarks ]; then
  rm -rf lean_benchmarks
fi
cp -r "$BENCH_DIR" lean_benchmarks

echo ""
echo "==> Building base image (this may take a while)..."
docker build \
  -f gcp/docker/Dockerfile.base \
  -t leanbench-base:local \
  .

# Clean up copy
rm -rf lean_benchmarks

echo ""
echo "==> Building runner image..."
docker build \
  -f gcp/docker/Dockerfile.runner \
  --build-arg BASE_IMAGE=leanbench-base:local \
  -t ax-prover-runner:local \
  .

echo ""
echo "==> Testing images..."
echo ""
echo "Python version:"
docker run --rm ax-prover-runner:local python3

echo ""
echo "Testing StrEnum import:"
docker run --rm ax-prover-runner:local python3 -c "from enum import StrEnum; print('✓ StrEnum works!')"

echo ""
echo "Testing ax-prover installation:"
docker run --rm ax-prover-runner:local ax-prover --version || true

echo ""
echo "Testing ax-prover help:"
docker run --rm ax-prover-runner:local ax-prover --help | head -20

echo ""
echo "==> Success! Images ready:"
echo "  - leanbench-base:local (lean_benchmarks@${BENCH_COMMIT})"
echo "  - ax-prover-runner:local"
echo ""
echo "To run interactively:"
echo "  docker run -it --rm ax-prover-runner:local bash"
