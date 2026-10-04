#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DOMAINS="Laptop"
export CASES="${CASES:-case1 case2 case3}"
export TIMESTEPS="${TIMESTEPS:-50000}"
export EPISODES="${EPISODES:-100}"
export SEED="${SEED:-0}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] training started: domain=$DOMAINS cases=$CASES timesteps=$TIMESTEPS seed=$SEED"
"$PROJECT_DIR/run_command/train_cases1_to3_expert.sh"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] training completed; evaluation started: episodes=$EPISODES"
"$PROJECT_DIR/run_command/test_cases1_to3_expert.sh"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] training and evaluation completed"
