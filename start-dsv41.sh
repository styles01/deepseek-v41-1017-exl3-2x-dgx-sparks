#!/usr/bin/env bash
# start-dsv41.sh — UX launcher target (sparkDash tile): bring up DSV4.1 TP2 ONLY.
# No downloads, no GLM handling beyond a refuse-if-running guard.
# Serves BOTH names: dsv41 + GLM-5.3-Flash-EXL3 (alias, so Hermes pins keep working).
#   MAX_LEN=275000 MAX_SEQS=4 TEXT_ONLY=0 (4 lanes x 275K; KV 2.5GiB/rank fits, sized 10/12)
set -euo pipefail
cd "$(dirname "$0")"

WEIGHTS_DIR="/home/jaita/models/dsv41-exl3"

if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^glm53-exl3-"; then
  echo "REFUSED: GLM (glm53-exl3-*) is serving — the cluster is one-lane-at-a-time."
  echo "Stop GLM first (sparkDash GLM tile stop script), then run this."
  exit 1
fi
kb=$(du -s "$WEIGHTS_DIR" 2>/dev/null | cut -f1 || echo 0)
if [ "$kb" -lt 400000000 ]; then
  echo "REFUSED: weights incomplete at $WEIGHTS_DIR (${kb} KB). Run the cutover"
  echo "download first: ./switch-dsv41.sh switch (or resume it); then retry."
  exit 1
fi

HEAD=10.200.0.1 WORKER=10.200.0.2 MODEL_DIR="$WEIGHTS_DIR" \
  SERVED="dsv41 DeepSeek-4.1-Flash GLM-5.3-Flash-EXL3" MAX_LEN=275000 MAX_SEQS=4 TEXT_ONLY=0 \
  ./scripts/launch/dsv41_tp2.sh up

echo ">> waiting for /v1/models ..."
for i in $(seq 1 120); do
  out=$(curl -s -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null || true)
  if echo "$out" | grep -q dsv41; then
    echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); print('>> UP, serving:', [m['id'] for m in d['data']])"
    exit 0
  fi
  sleep 10
done
echo "!! engine did not report within 20m — check: ssh larryspark 'docker logs dsv41-head --tail 50'"
exit 1