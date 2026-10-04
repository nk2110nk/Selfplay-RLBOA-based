#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../Transformer-based/.venv/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
SMOKE_ROOT="${SMOKE_ROOT:-$PROJECT_DIR/smoke_results}"
SEED="${SEED:-0}"
CASE_NAME="${CASE_NAME:-case1}"
EPISODES="${EPISODES:-100}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
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
      checkpoint="$model_dir/checkpoint.zip"
      output="$model_dir/csv/$pair/$domain/det=False_noise=False/$domain-$pair-dF-nF.tsv"
      if [[ ! -f "$checkpoint" && "$DRY_RUN" != "1" ]]; then echo "missing: $checkpoint" >&2; continue; fi
      if [[ "$FORCE" != "1" && -f "$output" && -f "$output.json" ]] && "$PYTHON_BIN" - "$output.json" "$checkpoint" "$SEED" "$EPISODES" "$CASE_NAME" "$domain" "$a0" "$a1" <<'PY'
import hashlib, json, pathlib, sys
m=json.loads(pathlib.Path(sys.argv[1]).read_text())
h=hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()
valid=(
    m.get('checkpoint_sha256') == h
    and m.get('seed') == int(sys.argv[3])
    and m.get('episodes') == int(sys.argv[4])
    and m.get('case') == sys.argv[5]
    and m.get('domain') == sys.argv[6]
    and m.get('opponents') == [sys.argv[7], sys.argv[8]]
    and m.get('model_type') == 'expert'
    and m.get('n_actions') == 10
    and not m.get('deterministic')
    and not m.get('noise')
)
raise SystemExit(0 if valid else 1)
PY
      then echo "validated existing: $output"; continue; fi
      command=("$PYTHON_BIN" test_negotiator.py --model "$model_dir" --domain "$domain"
        --opponent1 "$a0" --opponent2 "$a1" --case "$CASE_NAME" --model-type expert
        --episodes "$EPISODES" --seed "$SEED")
      if [[ "$DRY_RUN" == "1" ]]; then printf ' %q' "${command[@]}"; printf '\n';
      else "${command[@]}"; fi
    done
  done
done
