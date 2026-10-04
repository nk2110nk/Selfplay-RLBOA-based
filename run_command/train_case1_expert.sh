#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../Transformer-based/.venv/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
SMOKE_ROOT="${SMOKE_ROOT:-$PROJECT_DIR/smoke_results}"
SEED="${SEED:-0}"
CASE_NAME="${CASE_NAME:-case1}"
TIMESTEPS="${TIMESTEPS:-100000}"
N_ENVS="${N_ENVS:-4}"
N_STEPS="${N_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
SMOKE="${SMOKE:-0}"
read -r -a DOMAINS <<< "${DOMAINS:-Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A}"
read -r -a AGENTS <<< "${AGENTS:-Boulware Linear Conceder Atlas3 CUHKAgent}"

case "$CASE_NAME" in
  case1|case2|case3) ;;
  *) echo "CASE_NAME must be case1, case2, or case3: $CASE_NAME" >&2; exit 2 ;;
esac

root="$RESULTS_ROOT"
if [[ "$SMOKE" == "1" ]]; then root="$SMOKE_ROOT"; fi

for domain in "${DOMAINS[@]}"; do
  for ((i=0; i<${#AGENTS[@]}; i++)); do
    for ((j=i; j<${#AGENTS[@]}; j++)); do
      a0="${AGENTS[$i]}"; a1="${AGENTS[$j]}"; pair="$a0-$a1"
      model_dir="$root/seed-$SEED/$CASE_NAME/models/expert/$pair/$domain/RLBOASelfPlay_Negotiator"
      command=("$PYTHON_BIN" train.py --issue "$domain" --agents "$a0" "$a1"
        --model-type expert --case "$CASE_NAME" --save-path "$root" --seed "$SEED"
        --total-timesteps "$TIMESTEPS" --n-envs "$N_ENVS" --n-steps "$N_STEPS"
        --batch-size "$BATCH_SIZE" --pool-eval-freq 10000 --snapshot-freq 10000
        --pool-eval-episodes 4 --max-pool-size 16 --self-play-probability 0.0
        --scripted-probability 0.5 --snapshot-probability 0.5)
      if [[ -f "$model_dir/checkpoint.zip" ]]; then
        if [[ "$RESUME" == "1" ]]; then
          command=("$PYTHON_BIN" train.py --resume "$model_dir" --total-timesteps "$TIMESTEPS")
        else
          echo "skip existing (set RESUME=1 explicitly): $model_dir"
          continue
        fi
      fi
      if [[ "$DRY_RUN" == "1" ]]; then printf ' %q' "${command[@]}"; printf '\n';
      else "${command[@]}"; fi
    done
  done
done
