#!/bin/bash
# Final test-set evaluation of the best model of each type.
#
#   tmux new -s test
#   bash scripts/run_test.sh          # from the repo root
#   (detach: Ctrl-b d, reattach: tmux attach -t test)
#
# All models run in parallel. Per-episode results go to
# results/<agent>_<tag>_test.csv (written by scripts.evaluate) and each
# model's printed summary to results/test_logs/<name>.log. The LM model is
# the slowest (~3 h); the others finish in under ~1.5 h.
# Set PY to override the interpreter (it needs torch + transformers).

set -u
cd "$(dirname "$0")/.."
PY=${PY:-/usr/local/bin/python3}
LOGS=results/test_logs
mkdir -p "$LOGS"

run() {  # run <name> <evaluate args...>
    local name=$1; shift
    echo "[$(date '+%H:%M')] start  $name"
    if "$PY" -m scripts.evaluate --split test "$@" > "$LOGS/$name.log" 2>&1; then
        echo "[$(date '+%H:%M')] done   $name: $(grep -h 'success rate\|mean capped' "$LOGS/$name.log" | tr -s ' ' | tr '\n' ' ')"
    else
        echo "[$(date '+%H:%M')] FAILED $name (see $LOGS/$name.log)"
    fi
}

run random    --agent random &
run frequency --agent frequency &
run belief    --agent belief --set endgame_conf=0.5 hunt_every=3 --tag end0.5_hunt3 &
run lm        --agent lm --set hunt_every=3 --tag w3_end0.5_hunt3 &
run ppo       --agent ppo --set checkpoint=checkpoints/ppo_hunt/best.pt --tag hunt_best &
wait

echo
echo "== summary (test split) =="
for f in "$LOGS"/*.log; do
    printf '%-10s %s\n' "$(basename "$f" .log)" \
        "$(grep -h 'success rate\|mean capped\|median guesses' "$f" | tr -s ' ' | tr '\n' ' ')"
done | tee "$LOGS/summary.txt"
