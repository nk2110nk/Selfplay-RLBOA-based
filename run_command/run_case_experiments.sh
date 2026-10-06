#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

read -r -a CASE_LIST <<< "${CASES:-case1 case2 case3}"
read -r -a KNOWN_DOMAINS <<< "${DOMAINS:-Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A}"
read -r -a AGENT_LIST <<< "${AGENTS:-Boulware Conceder Linear Atlas3}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
EXPERT_TIMESTEPS="${EXPERT_TIMESTEPS:-100000}"
GENERAL_TIMESTEPS="${GENERAL_TIMESTEPS:-300000}"
N_ENVS="${N_ENVS:-4}"
N_STEPS="${N_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_EPOCHS="${N_EPOCHS:-10}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-0}"

if (( SHARD_COUNT < 1 || SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
  echo "Invalid shard: index=$SHARD_INDEX count=$SHARD_COUNT" >&2
  exit 2
fi
if (( ${#AGENT_LIST[@]} != 4 )); then
  echo "This experiment requires exactly four scripted agents" >&2
  exit 2
fi

checkpoint_step() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
import torch

try:
    try:
        state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(sys.argv[1], map_location="cpu")
    print(int(state.get("global_step", -1)))
except Exception:
    print(-1)
PY
}

run_training() {
  local job_index="$1" model_dir="$2" target="$3"
  shift 3
  if (( job_index % SHARD_COUNT != SHARD_INDEX )); then
    return
  fi
  if (( LIMIT > 0 && handled_jobs >= LIMIT )); then
    return
  fi

  local state_file="$model_dir/training_state.pt" step=-1
  if [[ -f "$state_file" ]]; then
    step="$(checkpoint_step "$state_file")"
  fi
  if (( step >= target )); then
    echo "[job $job_index] complete ($step/$target): $model_dir"
    handled_jobs=$((handled_jobs + 1))
    return
  fi

  local -a command
  if (( step >= 0 )); then
    command=("$PYTHON_BIN" train.py --resume "$model_dir"
             --total-timesteps "$target" --device "$DEVICE")
    echo "[job $job_index] resume ($step/$target): $model_dir"
  else
    command=("$PYTHON_BIN" train.py "$@" --total-timesteps "$target"
             --device "$DEVICE" --save-path "$model_dir")
    echo "[job $job_index] start (0/$target): $model_dir"
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  %q' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
  handled_jobs=$((handled_jobs + 1))
}

common_args=(
  --seed "$SEED"
  --n-envs "$N_ENVS"
  --n-steps "$N_STEPS"
  --batch-size "$BATCH_SIZE"
  --n-epochs "$N_EPOCHS"
  --pool-eval-freq 10000
  --pool-eval-episodes 4
  --snapshot-freq 10000
  --max-pool-size 16
  --self-play-probability 0.0
  --scripted-probability 0.5
  --snapshot-probability 0.5
  --pfsp-alpha 1.0
  --uniform-mix 0.1
)

job_index=0
handled_jobs=0

for case_name in "${CASE_LIST[@]}"; do
  case "$case_name" in
    case1|case2|case3) ;;
    *) echo "Invalid case: $case_name" >&2; exit 2 ;;
  esac

  for domain in "${KNOWN_DOMAINS[@]}"; do
    for ((i = 0; i < ${#AGENT_LIST[@]}; i++)); do
      for ((j = i; j < ${#AGENT_LIST[@]}; j++)); do
        agent0="${AGENT_LIST[$i]}"
        agent1="${AGENT_LIST[$j]}"
        pair="$agent0-$agent1"
        model_dir="$RESULTS_ROOT/$case_name/models/expert/$pair/$domain/RLBOASelfPlay_Negotiator"
        duplicate_option=(--allow-duplicate-opponents)
        if [[ "$agent0" != "$agent1" ]]; then
          duplicate_option=(--no-allow-duplicate-opponents)
        fi
        run_training "$job_index" "$model_dir" "$EXPERT_TIMESTEPS" \
          --agents "$agent0" "$agent1" --issue "$domain" \
          --model-type expert --case "$case_name" \
          "${duplicate_option[@]}" "${common_args[@]}"
        job_index=$((job_index + 1))
      done
    done
  done

  model_dir="$RESULTS_ROOT/$case_name/models/general/RLBOASelfPlay_Negotiator"
  run_training "$job_index" "$model_dir" "$GENERAL_TIMESTEPS" \
    --agents "${AGENT_LIST[@]}" --issue "${KNOWN_DOMAINS[@]}" \
    --model-type general --case "$case_name" --allow-duplicate-opponents \
    "${common_args[@]}"
  job_index=$((job_index + 1))
done

mkdir -p "$RESULTS_ROOT/experiment_status"
if (( LIMIT == 0 )) && [[ "$DRY_RUN" != "1" ]]; then
  touch "$RESULTS_ROOT/experiment_status/train-shard-${SHARD_INDEX}-of-${SHARD_COUNT}.done"
fi
echo "Training shard complete: shard=$SHARD_INDEX/$SHARD_COUNT handled=$handled_jobs total_jobs=$job_index"
