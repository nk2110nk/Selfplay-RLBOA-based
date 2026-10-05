#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

read -r -a CASE_LIST <<< "${CASES:-case1 case2 case3}"
read -r -a KNOWN_DOMAINS <<< "${DOMAINS:-Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A}"
read -r -a UNKNOWN_DOMAINS <<< "${UNKNOWN_DOMAINS:-Coffee Camera Lunch SmartPhone Kitchen}"
read -r -a AGENT_LIST <<< "${AGENTS:-Boulware Conceder Linear Atlas3}"
ALL_DOMAINS=("${KNOWN_DOMAINS[@]}" "${UNKNOWN_DOMAINS[@]}")

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
DEVICE="${DEVICE:-cuda}"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
LIMIT="${LIMIT:-0}"

if (( SHARD_COUNT < 1 || SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
  echo "Invalid shard: index=$SHARD_INDEX count=$SHARD_COUNT" >&2
  exit 2
fi

result_is_current() {
  "$PYTHON_BIN" - "$1" "$2" "$SEED" "$EPISODES" "$3" "$4" "$5" "$6" "$7" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path, checkpoint_path = map(Path, sys.argv[1:3])
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
except (OSError, ValueError):
    raise SystemExit(1)

valid = (
    manifest.get("checkpoint_sha256") == digest
    and manifest.get("seed") == int(sys.argv[3])
    and manifest.get("episodes") == int(sys.argv[4])
    and manifest.get("case") == sys.argv[5]
    and manifest.get("domain") == sys.argv[6]
    and manifest.get("opponents") == [sys.argv[7], sys.argv[8]]
    and manifest.get("model_type") == sys.argv[9]
    and not manifest.get("deterministic")
    and not manifest.get("noise")
)
raise SystemExit(0 if valid else 1)
PY
}

run_evaluation() {
  local job_index="$1" model_dir="$2" model_type="$3" case_name="$4"
  local domain="$5" agent0="$6" agent1="$7"
  if (( job_index % SHARD_COUNT != SHARD_INDEX )); then
    return
  fi
  if (( LIMIT > 0 && handled_jobs >= LIMIT )); then
    return
  fi

  local checkpoint="$model_dir/checkpoint.zip"
  local pair="$agent0-$agent1"
  local file_name="$domain-$pair-dF-nF.tsv"
  local export_file="$RESULTS_ROOT/seed-$SEED/$case_name/evaluation/$model_type/$pair/$domain/$case_name/$file_name"
  if [[ ! -f "$checkpoint" && "$DRY_RUN" != "1" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  if [[ "$FORCE" != "1" && -f "$export_file" && -f "$export_file.json" ]] && \
      result_is_current "$export_file.json" "$checkpoint" "$case_name" \
        "$domain" "$agent0" "$agent1" "$model_type"; then
    echo "[job $job_index] complete: $export_file"
    handled_jobs=$((handled_jobs + 1))
    return
  fi

  local -a command=(
    "$PYTHON_BIN" test_negotiator.py
    --model "$model_dir"
    --domain "$domain"
    --opponent1 "$agent0"
    --opponent2 "$agent1"
    --case "$case_name"
    --model-type "$model_type"
    --episodes "$EPISODES"
    --seed "$SEED"
    --device "$DEVICE"
    --no-deterministic
    --no-noise
    --export-root "$RESULTS_ROOT"
  )
  echo "[job $job_index] evaluate: $model_type $case_name $domain $pair"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  %q' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
  handled_jobs=$((handled_jobs + 1))
}

job_index=0
handled_jobs=0
general_pair="$(IFS=-; echo "${AGENT_LIST[*]}")"

for case_name in "${CASE_LIST[@]}"; do
  for domain in "${KNOWN_DOMAINS[@]}"; do
    for ((i = 0; i < ${#AGENT_LIST[@]}; i++)); do
      for ((j = i; j < ${#AGENT_LIST[@]}; j++)); do
        agent0="${AGENT_LIST[$i]}"
        agent1="${AGENT_LIST[$j]}"
        pair="$agent0-$agent1"
        model_dir="$RESULTS_ROOT/seed-$SEED/$case_name/models/expert/$pair/$domain/RLBOASelfPlay_Negotiator"
        run_evaluation "$job_index" "$model_dir" expert "$case_name" \
          "$domain" "$agent0" "$agent1"
        job_index=$((job_index + 1))
      done
    done
  done

  model_dir="$RESULTS_ROOT/seed-$SEED/$case_name/models/general/$general_pair/general/RLBOASelfPlay_Negotiator"
  for domain in "${ALL_DOMAINS[@]}"; do
    for ((i = 0; i < ${#AGENT_LIST[@]}; i++)); do
      for ((j = i; j < ${#AGENT_LIST[@]}; j++)); do
        run_evaluation "$job_index" "$model_dir" general "$case_name" \
          "$domain" "${AGENT_LIST[$i]}" "${AGENT_LIST[$j]}"
        job_index=$((job_index + 1))
      done
    done
  done
done

mkdir -p "$RESULTS_ROOT/experiment_status"
if (( LIMIT == 0 )) && [[ "$DRY_RUN" != "1" ]]; then
  touch "$RESULTS_ROOT/experiment_status/eval-shard-${SHARD_INDEX}-of-${SHARD_COUNT}.done"
fi
echo "Evaluation shard complete: shard=$SHARD_INDEX/$SHARD_COUNT handled=$handled_jobs total_jobs=$job_index"
