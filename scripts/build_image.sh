#!/usr/bin/env bash
# Build the multi-stage image.
#
# On a normal machine this is just: docker build -t churn-scoring-api:phase3 .
# In this VM Docker cannot start containers (seccomp blocks setns),
# so the build uses direct Buildah commands. Two VM workarounds:
# (1) buildah bud fails at pip install because the egress proxy
# presents a self-signed certificate; the CA bundle is mounted
# read-only for the install step only, so nothing about the proxy
# leaks into the image config. (2) buildah commit stages blobs in
# /var/tmp, a 2 GB tmpfs the image does not fit in; the commit runs
# with TMPDIR=~/buildah-tmp on the overlay filesystem.
#
# Run scripts/smoke_image.sh after this to record the size and
# smoke-test the image in chroot.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=churn-scoring-api:phase3
mkdir -p ~/buildah-tmp
CA_BUNDLE=/run/hatch/egress-tls/ca-bundle.pem

echo "== builder stage =="
builder=$(buildah --storage-driver vfs from python:3.12-slim)
buildah --storage-driver vfs copy "$builder" requirements.txt /install/requirements.txt
buildah --storage-driver vfs run --volume "$CA_BUNDLE:/ca-bundle.pem:ro" \
    --env SSL_CERT_FILE=/ca-bundle.pem "$builder" \
    pip install --no-cache-dir --prefix=/install -r /install/requirements.txt
buildah --storage-driver vfs commit "$builder" churn-scoring-api:builder
buildah --storage-driver vfs rm "$builder" >/dev/null

echo "== runtime stage =="
runtime=$(buildah --storage-driver vfs from python:3.12-slim)
buildah --storage-driver vfs copy --from churn-scoring-api:builder \
    "$runtime" /install /usr/local
buildah --storage-driver vfs copy "$runtime" app/ /app/app/
buildah --storage-driver vfs copy "$runtime" artifacts/ /app/artifacts/
buildah --storage-driver vfs copy "$runtime" data/raw/telco.csv /app/data/raw/telco.csv
buildah --storage-driver vfs copy "$runtime" benchmarks/results/metrics.json \
    /app/benchmarks/results/metrics.json
buildah --storage-driver vfs config --workingdir /app \
    --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
    --port 8000 \
    --cmd '["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]' \
    "$runtime"
TMPDIR="$HOME/buildah-tmp" buildah --storage-driver vfs commit \
    "$runtime" "$TAG"
buildah --storage-driver vfs rm "$runtime" >/dev/null
echo "built $TAG"
