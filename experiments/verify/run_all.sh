#!/bin/sh
# Run every experiment in order and write the raw output to RESULTS_RAW.txt
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
OUT=experiments/verify/RESULTS_RAW.txt
: > "$OUT"
for f in 00_environment 01_bandwidth 02_generate_cli 03_tokens_and_shapes 04_forward_burst_vs_sustained 05_argmax 06_dependent_loop_per_step 00_environment; do
  echo "=== $f ===" | tee -a "$OUT"
  $PY experiments/verify/$f.py 2>&1 | grep -vE "Fetching|it/s\]|Reconstruction|Download complete" | tee -a "$OUT"
  echo | tee -a "$OUT"
done
