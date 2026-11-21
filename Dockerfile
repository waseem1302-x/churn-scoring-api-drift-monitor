# Multi-stage build. The builder installs the runtime dependencies;
# the runtime stage ships only what the API needs. Evidently is
# reporting-only and pytest is test-only, so neither enters the
# image (see requirements-dev.txt): that is most of the saving
# against the 1.39 GB Phase 2 image.
#
# On a normal machine: docker build -t churn-scoring-api:phase3 .
# In this VM Docker cannot start containers (seccomp blocks setns),
# so the image is built with Buildah instead; see
# scripts/build_image.sh for the exact commands, including the two
# VM workarounds (CA bundle for pip, TMPDIR for the commit).

# Stage 1: builder. Dependencies go to /install, not /usr/local,
# so the runtime stage can copy them in one layer.
FROM python:3.12-slim AS builder
WORKDIR /install
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: runtime. Only the API code, the artifacts, the Telco
# file the drift reference reads, and the metrics file /metrics
# serves. No build tools, no test code, no scripts, no benchmark
# reports, no Evidently.
FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY --from=builder /install /usr/local
COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY data/raw/telco.csv ./data/raw/telco.csv
COPY benchmarks/results/metrics.json ./benchmarks/results/metrics.json
EXPOSE 8000
# Cloud Run sets $PORT (default 8080) and requires the container to
# listen on it; locally $PORT is unset and the default 8000 applies.
# exec keeps uvicorn as PID 1 so it receives the stop signal directly.
CMD ["sh", "-c", "exec uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
