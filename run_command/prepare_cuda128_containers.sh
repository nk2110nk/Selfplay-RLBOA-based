#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="${IMAGE:-selfplay-rlboa:cu128}"
GPU_IDS_TEXT="${GPU_IDS:-0 1}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-selfplay-rlboa-training-gpu}"
HOST_ROOT="${HOST_ROOT:-/home/nakata}"
REBUILD="${REBUILD:-0}"

read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_TEXT"
if (( ${#GPU_IDS_ARRAY[@]} < 1 )); then
  echo "GPU_IDS must contain at least one GPU" >&2
  exit 2
fi

if [[ "$REBUILD" == "1" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -f "$PROJECT_DIR/Dockerfile.cuda128" -t "$IMAGE" "$PROJECT_DIR"
fi

for gpu_id in "${GPU_IDS_ARRAY[@]}"; do
  container="${CONTAINER_PREFIX}${gpu_id}"
  if docker inspect "$container" >/dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]]; then
      docker start "$container" >/dev/null
    fi
    echo "ready: $container"
    continue
  fi
  docker run -dit \
    --name "$container" \
    --gpus "device=$gpu_id" \
    --user "$(id -u):$(id -g)" \
    --env HOME="$HOST_ROOT" \
    --env USER=nakata \
    --env LOGNAME=nakata \
    --env MPLCONFIGDIR=/tmp/matplotlib \
    --volume "$HOST_ROOT:$HOST_ROOT" \
    --workdir "$PROJECT_DIR" \
    "$IMAGE" >/dev/null
  echo "created: $container"
done

echo "Containers are ready. No training or evaluation was started."
