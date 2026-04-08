#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-experiments_quick_5k.yaml}"

python run_experiments.py --config "$CONFIG"
python collect_results.py --config "$CONFIG" --result-root ./results --csv-out results_summary_5k.csv --md-out results_summary_5k.md
