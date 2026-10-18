#!/usr/bin/env bash
# stop-dsv41.sh — UX launcher target (sparkDash tile): stop the DSV4.1 TP2 serve.
set -euo pipefail
cd "$(dirname "$0")"
HEAD=10.200.0.1 WORKER=10.200.0.2 MODEL_DIR="/home/jaita/models/dsv41-exl3" \
  ./scripts/launch/dsv41_tp2.sh down
echo ">> dsv41 containers stopped (weights stay on disk)"