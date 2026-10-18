#!/usr/bin/env bash
# ============================================================================
# switch-dsv41.sh — staged cutover: GLM-5.3-Flash cluster -> DeepSeek-V4.1-Flash
# Control box: LARRYSPARK (head, rank 0). Worker: LARRYSPARK2 (rank 1).
# Recipe: github.com/sudoingX/dgx-spark-deepseek-v41 (repo staged at ~/dsv41_recipe)
#
#   ./switch-dsv41.sh status        # everything: repo, images, weights, GLM, disk
#   ./switch-dsv41.sh bench-setup   # fetch kit_bench.py (sha256-pinned) — SAFE anytime
#   ./switch-dsv41.sh preflight     # run recipe check.sh (read-only) — SAFE anytime
#   ./switch-dsv41.sh switch        # THE CUTOVER: GLM down -> build/download/serve -> bench
#   ./switch-dsv41.sh rollback      # dsv41 down, GLM cluster back up
#
# HARD SAFETY RULE (from recipe FINDINGS/limits): no large file I/O (weights,
# images) while a GPU job runs — on GB10 CUDA allocs cannot reclaim page cache;
# boxes went silent. Therefore ALL heavy I/O (image build, 399 GB weights,
# rsync to worker) happens ONLY after GLM is stopped, inside `switch`.
#
# CUTOVER TARGETS (2026-10-12, per James):
#   SERVED="dsv41 DeepSeek-4.1-Flash GLM-5.3-Flash-EXL3" -> vLLM answers to BOTH names (engine-args:
#     --served-model-name takes multiple names, server responds to any). Every
#     Hermes profile pinned to GLM-5.3-Flash-EXL3 @ :8000 keeps working = zero churn.
#   MAX_LEN=275000 MAX_SEQS=4 -> 4 lanes, up to 275K tokens each.
#     KV math: pool 774K token-slots/rank (2.5 GiB fp8_ds_mla); demand/rank
#     = lanes x MAX_LEN/2 = 4 x 137.5K = 550K <= 774K. Fits, ~220K headroom.
#     (KV_BYTES must stay pinned; do NOT raise UTIL without measuring.)
#   TEXT_ONLY: dsv41 skips the vision tower -> aux.vision must move OFF
#     larryspark at cutover (see Mac-side dsv41-hermes-prep.sh).
# ============================================================================
set -euo pipefail

RECIPE="/home/jaita/dsv41_recipe"
WORKER="10.200.0.2"
SELF="10.200.0.1"
WEIGHTS_DIR="/home/jaita/models/dsv41-exl3"     # same absolute path on BOTH nodes
KIT="$RECIPE/kit_bench.py"
KIT_SHA256="83a4fbeb987032f906cf00d3f6c9dd2fad47dd675e75dd498c4c4188d63a6c0d"
GLM_DIR="/home/jaita/src/GLM-5.3-Flash-EXL3-2x-DGX-Sparks"   # REAL home of glm53-exl3-head/worker
                                                    # (TensorFold dir was wrong — its stop.sh
                                                    #  targets glm53-flash-tf, absent)
MODEL_NAME="dsv41"
BENCH_OUT="$RECIPE/bench-results"
mkdir -p "$BENCH_OUT"

glm_up()     { curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; }
dsv41_up()   { curl -s -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q dsv41; }
dl_done()    { [ -d "$WEIGHTS_DIR" ] && [ -f "$WEIGHTS_DIR/MANIFEST.sha256" ] \
               && du -s "$WEIGHTS_DIR" 2>/dev/null | awk '{exit !($1>400000000)}'; }

cmd="${1:-status}"
case "$cmd" in

status)
  echo "== DSV41 staging on $(hostname) =="
  echo "-- recipe repo --"; [ -d "$RECIPE/.git" ] && echo "  $RECIPE ($(git -C "$RECIPE" log -1 --format=%h 2>/dev/null))" || echo "  MISSING"
  echo "-- bench client --"
  if [ -f "$KIT" ]; then
    echo "  kit_bench.py: $(sha256sum "$KIT" | cut -d' ' -f1)"
    [ "$(sha256sum "$KIT" | cut -d' ' -f1)" = "$KIT_SHA256" ] && echo "  checksum: MATCHES pinned" || echo "  checksum: MISMATCH — delete and re-run bench-setup"
  else echo "  kit_bench.py: not fetched (run: $0 bench-setup)"; fi
  echo "-- weights (BOTH nodes, same path: $WEIGHTS_DIR) --"
  if dl_done; then echo "  head: assembled + manifest present ($(du -s "$WEIGHTS_DIR" | cut -f1) KB)"
  else echo "  head: incomplete/absent ($(du -s "$WEIGHTS_DIR" 2>/dev/null | cut -f1 || echo 0) KB) — downloaded only inside 'switch' window"; fi
  echo "  worker: $(ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER" "du -s $WEIGHTS_DIR 2>/dev/null | cut -f1 || echo absent" 2>/dev/null || echo unreachable)"
  echo "-- images --"; docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>/dev/null | grep -E "dsv41" || echo "  no dsv41 images built yet (built inside 'switch' window)"
  echo "-- live serve --"
  glm_up   && echo "  GLM :8000 UP (cluster is the active rig)" || true
  dsv41_up && echo "  DSV41 :8000 UP" || true
  { glm_up || dsv41_up; } || echo "  :8000 nothing serving"
  echo "-- disk --"
  df -h / | tail -1 | sed 's/^/  head:   /'
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER" "df -h / | tail -1" 2>/dev/null | sed 's/^/  worker: /' || echo "  worker: unreachable"
  echo "-- mem --"; free -g | head -2 | tail -1 | sed 's/^/  /'
  ;;

bench-setup)
  echo ">> fetching pinned kit_bench.py"
  curl -fsSLo "$KIT" https://raw.githubusercontent.com/bertholomus/TensorFold/fe6f9f7b373db76422ddfa121577238474c0a1b7/tools/dsv41/kit_bench.py
  got=$(sha256sum "$KIT" | cut -d' ' -f1)
  [ "$got" = "$KIT_SHA256" ] && echo ">> sha256 MATCH: $got" || { echo ">> MISMATCH ($got) — removing"; rm -f "$KIT"; exit 1; }
  ;;

preflight)
  echo ">> running recipe check.sh (read-only) — self as HEAD, $WORKER as WORKER"
  cd "$RECIPE"
  HEAD="$SELF" WORKER="$WORKER" ./check.sh 2>&1 | tail -40
  ;;

switch)
  echo "THIS WILL: stop the GLM cluster, then build images + download 399 GB weights"
  echo "(head + worker rsync) + launch DSV41-Flash TP2 on :8000. Downtime: hours."
  read -r -p "Type SWITCH to confirm: " c; [ "$c" = "SWITCH" ] || { echo "aborted"; exit 1; }

  echo "== phase 0: stop GLM (both nodes) =="
  cd "$GLM_DIR" && ./stop.sh
  sleep 5
  # VERIFY actual death — never trust stop.sh's exit code (10/12 near-miss:
  # TensorFold dir's stop.sh was a silent no-op on the live glm53-exl3 stack).
  tries=0
  while (( tries < 30 )); do
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^glm53-exl3-" \
       && ! curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
      echo ">> VERIFIED: glm53-exl3 containers gone and :8000 dead"; break
    fi
    sleep 5; tries=$((tries+1))
    if (( tries >= 30 )); then
      echo "!! FATAL: GLM still alive after stop.sh — ABORTING (no GPU-job-beside-I/O risk)"; exit 1
    fi
  done

  restore() {
    echo ">> RESTORE: bringing GLM cluster back up"
    cd "$GLM_DIR" && nohup ./start.sh > "/tmp/glm-restore-dsv41-$(date +%m%d-%H%M).log" 2>&1 \
      && echo ">> start.sh launched (tail /tmp/glm-restore-dsv41-*.log)"
  }
  trap restore EXIT

  echo "== phase 1: recipe install (wheels + 3 images + copy to worker) =="
  cd "$RECIPE"
  PEER="$WORKER" ./install.sh 2>&1 | tail -15

  echo "== phase 2: weights — head download (424 GB incl. Engram) =="
  export HF_HOME=/home/jaita/models/hf
  hf download sudoingx/DeepSeek-V4.1-Flash-EXL3-2.7bpw --local-dir "$WEIGHTS_DIR" 2>&1 | tail -3
  hf download deepseek-ai/DeepSeek-V4.1-Flash model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
    --revision 2cba9e42aa026125f3ed06c6d98c1db82f7ca027 --local-dir "$WEIGHTS_DIR" 2>&1 | tail -3
  ( cd "$WEIGHTS_DIR" && sha256sum -c MANIFEST.sha256 2>&1 | tail -3 )

  echo "== phase 3: rsync weights to worker over ConnectX (~550 MB/s) =="
  ssh -o BatchMode=yes "$WORKER" "mkdir -p $(dirname "$WEIGHTS_DIR")"
  rsync -a --info=progress2 "$WEIGHTS_DIR/" "$WORKER:$WEIGHTS_DIR/"

  echo "== phase 4: launch dsv41 TP2 (4 lanes x 275K, alias GLM name) =="
  echo ">> evicting weights from page cache (GB10 trap: CUDA allocs don't reclaim cache; no sudo -n on this box)"
  python3 - "$WEIGHTS_DIR" <<'PYEOF'
import os, sys, ctypes, functools
libc = ctypes.CDLL("libc.so.6", use_errno=True)
POSIX_FADV_DONTNEED = 4
total = 0
for root, _, files in os.walk(sys.argv[1]):
    for f in files:
        p = os.path.join(root, f)
        try:
            fd = os.open(p, os.O_RDONLY)
            libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
            os.close(fd); total += 1
        except OSError:
            pass
print(f">> fadvise DONTNEED applied to {total} files")
PYEOF
  # SERVED with two names: dsv41 (new) + GLM-5.3-Flash-EXL3 (alias for all
  # existing Hermes pins); MAX_SEQS=4 for 2-4 lanes; 275K per lane.
  HEAD="$SELF" WORKER="$WORKER" MODEL_DIR="$WEIGHTS_DIR" \
    SERVED="dsv41 DeepSeek-4.1-Flash GLM-5.3-Flash-EXL3" MAX_LEN=275000 MAX_SEQS=4 TEXT_ONLY=0 \
    ./scripts/launch/dsv41_tp2.sh up
  ./scripts/launch/dsv41_tp2.sh status

  echo "== phase 5: smoke bench (pinned client, 3 reps) =="
  python3 "$KIT" --base http://127.0.0.1:8000 --model "$MODEL_NAME" decode --set b --tokens 384 --reps 3 2>&1 | tee "$BENCH_OUT/smoke-$(date +%m%d-%H%M).txt"

  trap - EXIT
  echo "== CUTOVER COMPLETE $(date) =="
  echo "   serve: http://$SELF:8000  model: $MODEL_NAME"
  echo "   NOTE: only MAX_SEQS=2, 262K context; GLM restarts via: $0 rollback"
  echo "   Hermes: point profiles' larryspark provider model at '$MODEL_NAME' (or new provider base_url stays :8000)"
  ;;

rollback)
  echo ">> rolling back to GLM cluster"
  cd "$RECIPE" && ./scripts/launch/dsv41_tp2.sh down 2>/dev/null || true
  sleep 3
  if glm_up; then echo ">> GLM already up"; else
    cd "$GLM_DIR" && nohup ./start.sh > "/tmp/glm-rollback-$(date +%m%d-%H%M).log" 2>&1 \
      && echo ">> GLM start.sh launched"
  fi
  echo ">> dsv41 weights/images remain on disk (delete manually to reclaim ~430 GB/box)"
  ;;

*)
  echo "usage: $0 {status|bench-setup|preflight|switch|rollback}"; exit 2
  ;;
esac
