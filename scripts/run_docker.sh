#!/usr/bin/env bash
# Run the image on a normal Docker host.
#
# This script has NOT been run on a Docker host: Docker cannot
# start containers in this development VM (seccomp blocks setns),
# so the image is built and smoke-tested with Buildah here
# (scripts/build_image.sh, scripts/smoke_image.sh). The first
# person with a Docker host runs this script and reports whether
# it works as written.
#
# Usage: CHURN_API_KEY=your-key bash scripts/run_docker.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${CHURN_API_KEY:-}" ]; then
    echo "set CHURN_API_KEY first (the API key /score and /drift require)" >&2
    exit 1
fi

docker build -t churn-scoring-api:phase3 .
docker run -d --name churn-scoring-api -p 8000:8000 \
    -e CHURN_API_KEY="$CHURN_API_KEY" churn-scoring-api:phase3

for _ in $(seq 1 30); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then break; fi
    sleep 1
done
curl -sf http://localhost:8000/health
echo
echo "API is up on http://localhost:8000"
echo "Score contract row 0 with:"
echo "  python3 -c \"import json; row=json.load(open('benchmarks/results/contract_rows.json'))[0]; print(json.dumps({'customer_id': row['id'], **row['inputs']}))\" > /tmp/churn-score-payload.json"
echo "  curl -H \"X-API-Key: \$CHURN_API_KEY\" -H \"Content-Type: application/json\" -d @/tmp/churn-score-payload.json http://localhost:8000/score"
echo "  (expect probability 0.0137, tier LOW)"
echo "Stop it with: docker stop churn-scoring-api && docker rm churn-scoring-api"
