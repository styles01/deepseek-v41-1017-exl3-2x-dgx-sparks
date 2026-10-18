#!/usr/bin/env bash
# arena-dsv41-submit.sh — pre-flight + full Spark Arena benchmark for the dsv41 recipe.
# Designed to run UNATTENDED (cron 2am ET): stops the manual dsv41 serve, runs the
# arena benchmark via sparkrun (detached), then restarts the serving lane.
set -uo pipefail
LOG=/tmp/arena-dsv41-submit.log
exec >> "$LOG" 2>&1
export PATH="$HOME/.local/bin:$PATH"
SPARK_HOST="${1:-larryspark}"

echo "=== ARENA SUBMIT START $(date -u +%FT%TZ) ==="

# 1. Stop our manual serve (frees :8000 + both GPUs for sparkrun's own launch)
ssh -o BatchMode=yes "$SPARK_HOST" 'cd /home/jaita/dsv41_recipe && ./scripts/launch/dsv41_tp2.sh down' || echo "stop rc=$? (continuing)"
sleep 5

# 2. Hard gate before submission (must PASS)
ssh -o BatchMode=yes "$SPARK_HOST" 'python3 ~/verify_arena_recipe.py @styles01/dsv41-tp2-2x'
GATE=$?
echo "gate rc=$GATE"
if [ "$GATE" != "0" ]; then
  echo "!! GATE FAILED — restarting the manual serve and aborting submission"
  ssh -o BatchMode=yes "$SPARK_HOST" 'cd /home/jaita/dsv41_recipe && ./start-dsv41.sh' || true
  exit 1
fi

# 3. Run the benchmark DETACHED on the Spark (survives everything; PATH preserved)
ssh -o BatchMode=yes "$SPARK_HOST" 'nohup env PATH="$HOME/.local/bin:$PATH" python3 -m sparkrun arena benchmark run @styles01/dsv41-tp2-2x --cluster sparkduo > /tmp/bench-dsv41.log 2>&1 &'
echo "benchmark launched detached (log: /tmp/bench-dsv41.log on $SPARK_HOST)"

# 4. Completion watcher: on terminal state, restart the serving lane + report
( sleep 600
  for i in $(seq 1 340); do
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$SPARK_HOST" 'grep -qE "Benchmark complete|error|Traceback|Submission .*failed" /tmp/bench-dsv41.log 2>/dev/null'; then
      echo "=== BENCH TERMINAL STATE DETECTED $(date -u +%FT%TZ) ==="
      ssh -o BatchMode=yes "$SPARK_HOST" 'tail -20 /tmp/bench-dsv41.log'
      echo "=== restarting serving lane ==="
      ssh -o BatchMode=yes "$SPARK_HOST" 'cd /home/jaita/dsv41_recipe && ./start-dsv41.sh > /tmp/dsv41-post-arena.log 2>&1'
      exit 0
    fi
    sleep 60
  done
) &
echo "watcher armed (max ~6h)"
echo "=== SETUP DONE (benchmark runs detached; watcher will restore service) ==="
