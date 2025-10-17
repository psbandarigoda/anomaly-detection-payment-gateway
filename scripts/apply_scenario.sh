#!/bin/bash
set -euo pipefail
if [ $# -ne 1 ]; then
  echo "Usage: $0 <scenario-folder-name>"
  echo "e.g., $0 01_reject_trivy_high_risk"
  exit 1
fi
SCEN="$1"
echo "Applying scenario: $SCEN"
mkdir -p datasets/payment_set_0001 ml/data
cp -f "test_scenarios/$SCEN/.env" "datasets/payment_set_0001/.env"
cp -f "test_scenarios/$SCEN/eval.csv" "ml/data/eval.csv"
echo "Done. Commit and open a PR to run the pipeline."
