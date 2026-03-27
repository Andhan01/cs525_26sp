#!/bin/bash
set -euo pipefail

INPUT_DIR="${1:-.}"
OUTPUT_DIR="${2:-plots_q4}"

python3 "$(dirname "$0")/plot_q4_comparison.py" --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
#bash plot_q4_comparison.sh . plots_q4