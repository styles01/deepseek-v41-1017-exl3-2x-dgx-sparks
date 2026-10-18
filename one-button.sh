#!/usr/bin/env bash
# One-button: downloads the community pack + fresh 1017 Engram, repacks, boots the lane.
#   MODEL_ROOT=/home/jaita bash one-button.sh
# Requires: hf (huggingface_hub CLI), ~420 GB NVMe, docker, this repo, the published image tag.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL_ROOT=${MODEL_ROOT:-/home/jaita/models}
PACK=${PACK:-$MODEL_ROOT/dsv41-1017-exl3-2.9}
ENGRAM=${ENGRAM:-$MODEL_ROOT/dsv41-1017-engram-src}
OUT=${OUT:-$MODEL_ROOT/dsv41-1017-ours}
IMAGE=${IMAGE:-ghcr.io/styles01/dsv41-1017-exl3-2x-dgx-sparks:v1}
REPO_ID=${REPO_ID:-LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw}
BASE_ID=${BASE_ID:-deepseek-ai/DeepSeek-V4.1-Flash-1017}

echo "== 1/4 image"; docker pull "$IMAGE" || echo "(already present?)"
echo "== 2/4 pack download"; hf download "$REPO_ID" --local-dir "$PACK"
echo "== 3/4 fresh 1017 Engram (shards 47+48; old tables poison)"
hf download "$BASE_ID" \
  --include "model-00047-of-00048.safetensors" "model-00048-of-00048.safetensors" \
  "model.safetensors.index.json" "config.json" --local-dir "$ENGRAM"
echo "== 4/4 repack"; python3 "$HERE/repack/repack_ours.py" --pack "$PACK" --engram "$ENGRAM" --out "$OUT"

export HEAD=${HEAD:-10.200.0.1} WORKER=${WORKER:-10.200.0.2}
SPEC='{"method":"dspark","num_speculative_tokens":6,"draft_parallel_config":{"tensor_parallel_size":2}}' \
  MODEL_DIR="$OUT" TEXT_ONLY=0 IMAGE="$IMAGE" bash "$HERE/scripts/launch/dsv41_tp2.sh" up

echo "Lane up on :8000. Bench with: python3 $HERE/kit_bench.py --base http://127.0.0.1:8000 --model dsv41 decode"