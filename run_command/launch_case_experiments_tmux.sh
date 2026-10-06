#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODE="${MODE:-train}"
GPU_IDS_TEXT="${GPU_IDS:-0 1}"
CONTAINER_NAMES_TEXT="${CONTAINER_NAMES:-selfplay-rlboa-training-gpu0 selfplay-rlboa-training-gpu1}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"

read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_TEXT"
read -r -a CONTAINER_NAMES_ARRAY <<< "$CONTAINER_NAMES_TEXT"
SHARD_COUNT="${#GPU_IDS_ARRAY[@]}"

case "$MODE" in
  train)
    SCRIPT="run_command/run_case_experiments.sh"
    SESSION_NAME="${SESSION_NAME:-selfplay-rlboa-cases}"
    LOG_PREFIX="train"
    ;;
  eval)
    SCRIPT="run_command/evaluate_case_experiments.sh"
    SESSION_NAME="${SESSION_NAME:-selfplay-rlboa-eval}"
    LOG_PREFIX="eval"
    ;;
  *)
    echo "MODE must be train or eval: $MODE" >&2
    exit 2
    ;;
esac

if (( SHARD_COUNT < 1 || ${#CONTAINER_NAMES_ARRAY[@]} != SHARD_COUNT )); then
  echo "GPU_IDS and CONTAINER_NAMES must contain the same non-zero number of entries" >&2
  exit 2
fi
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

for container in "${CONTAINER_NAMES_ARRAY[@]}"; do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" != "true" ]]; then
    echo "Docker container is not running: $container" >&2
    exit 1
  fi
done

mkdir -p "$RESULTS_ROOT/experiment_logs"
for ((shard = 0; shard < SHARD_COUNT; shard++)); do
  gpu_id="${GPU_IDS_ARRAY[$shard]}"
  container="${CONTAINER_NAMES_ARRAY[$shard]}"
  log="$RESULTS_ROOT/experiment_logs/${LOG_PREFIX}-gpu${gpu_id}.log"
  command="docker exec --workdir '$PROJECT_DIR' -e HOME=/home/nakata -e USER=nakata -e LOGNAME=nakata -e MPLCONFIGDIR=/tmp/matplotlib -e CUDA_VISIBLE_DEVICES=0 -e SHARD_INDEX=$shard -e SHARD_COUNT=$SHARD_COUNT -e RESULTS_ROOT='$RESULTS_ROOT' '$container' bash '$SCRIPT' 2>&1 | tee -a '$log'"
  if (( shard == 0 )); then
    tmux new-session -d -s "$SESSION_NAME" -n "gpu${gpu_id}" "$command"
  else
    tmux new-window -t "$SESSION_NAME" -n "gpu${gpu_id}" "$command"
  fi
done

echo "Started $MODE with $SHARD_COUNT GPU shards in tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
