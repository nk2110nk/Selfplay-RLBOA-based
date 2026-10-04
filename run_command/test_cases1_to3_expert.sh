#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
read -r -a CASES_TO_RUN <<< "${CASES:-case1 case2 case3}"

for case_name in "${CASES_TO_RUN[@]}"; do
  CASE_NAME="$case_name" "$PROJECT_DIR/run_command/test_case1_expert.sh" "$@"
done
