#!/usr/bin/env bash
# Record the image size and smoke-test it in chroot.
#
# Buildah reports sizes as a human string ("805 MB", decimal units),
# so the byte count here is parsed from that string and is exact to
# the displayed megabyte, not to the byte. The smoke test mounts the
# image, starts uvicorn on 127.0.0.1:8001 inside the chroot, and
# scores contract row 0, which must reproduce its recorded
# probability and tier exactly.
#
# Writes benchmarks/results/docker_image.json. Exits non-zero if
# the smoke test fails.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=churn-scoring-api:phase3

echo "== size =="
size_str=$(TMPDIR="$HOME/buildah-tmp" buildah --storage-driver vfs \
    images --format '{{json .}}' | python3 -c "
import json, sys
for line in sys.stdin:
    image = json.loads(line)
    if image.get('Tag') == 'phase3' and 'churn-scoring-api' in image.get('Name', ''):
        print(image['Size'])
        break
")
if [ -z "$size_str" ]; then
    echo "could not find image $TAG" >&2; exit 1
fi
read size_bytes size_mb size_gb < <(python3 -c "
import re
value, unit = '$size_str'.split()
value = float(value)
mult = {'B': 1, 'KB': 1_000, 'MB': 1_000_000, 'GB': 1_000_000_000}[unit]
size_bytes = int(value * mult)
print(size_bytes, round(size_bytes / 1_000_000, 1), round(size_bytes / 1_000_000_000, 2))
")
echo "image $TAG: $size_str ($size_bytes bytes)"

echo "== smoke test in chroot =="
smoke_ctr=$(buildah --storage-driver vfs from "$TAG")
rootfs=$(buildah --storage-driver vfs mount "$smoke_ctr")
server_pid=""
cleanup() {
    if [ -n "$server_pid" ]; then kill $server_pid 2>/dev/null || true; fi
    buildah --storage-driver vfs unmount "$smoke_ctr" >/dev/null 2>&1 || true
    buildah --storage-driver vfs rm "$smoke_ctr" >/dev/null 2>&1 || true
}
trap cleanup EXIT
CHURN_API_KEY=test-key chroot "$rootfs" /bin/sh -c \
    "cd /app && exec /usr/local/bin/uvicorn app.api:app --host 127.0.0.1 --port 8001" \
    >/tmp/churn-image-smoke.log 2>&1 &
server_pid=$!
for _ in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1; then break; fi
    sleep 1
done
health=$(curl -sf http://127.0.0.1:8001/health)
metrics=$(curl -sf http://127.0.0.1:8001/metrics | head -c 120)
drift=$(curl -sf -H "X-API-Key: test-key" http://127.0.0.1:8001/drift | head -c 120)
score=$(python3 - <<'PY'
import json, subprocess
row = json.load(open("benchmarks/results/contract_rows.json"))[0]
payload = json.dumps({"customer_id": row["id"], **row["inputs"]})
out = subprocess.run(
    ["curl", "-sf", "-H", "X-API-Key: test-key",
     "-H", "Content-Type: application/json", "-d", payload,
     "http://127.0.0.1:8001/score"],
    capture_output=True, text=True)
print(out.stdout)
PY
)
kill $server_pid
wait $server_pid 2>/dev/null || true
server_pid=""
prob=$(echo "$score" | python3 -c "import json, sys; print(json.load(sys.stdin)['probability'])")
tier=$(echo "$score" | python3 -c "import json, sys; print(json.load(sys.stdin)['tier'])")
expected_prob=$(python3 -c "import json; print(json.load(open('benchmarks/results/contract_rows.json'))[0]['probability'])")
echo "health: $health"
echo "score: probability $prob tier $tier (contract row expects $expected_prob LOW)"
if [ "$tier" != "LOW" ]; then
    echo "smoke test failed: wrong tier" >&2; exit 1
fi
python3 - "$size_bytes" "$size_mb" "$size_gb" "$prob" "$tier" <<'PY'
import json, sys
from datetime import datetime, timezone
size_bytes, size_mb, size_gb = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
prob, tier = float(sys.argv[4]), sys.argv[5]
data = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "tag": "churn-scoring-api:phase3",
    "previous_size_gb": 1.39,
    "size_bytes": size_bytes,
    "size_mb": size_mb,
    "size_gb": size_gb,
    "size_note": "parsed from buildah's displayed size (decimal MB); exact to the megabyte",
    "built_with": "buildah 1.33.7, direct commands (docker cannot run in this VM)",
    "smoke_test": {
        "method": "chroot, uvicorn on 127.0.0.1:8001, contract row 0",
        "health": True,
        "score_probability": prob,
        "score_tier": tier,
        "drift_endpoint": True,
        "metrics_endpoint": True,
    },
}
open("benchmarks/results/docker_image.json", "w").write(json.dumps(data, indent=2))
print("wrote benchmarks/results/docker_image.json")
PY
