#!/usr/bin/env bash
# Needs the probe test, which is archived at the git tag archive/terfc-screening-probe (check it out first).
# usage: MODE=... MOL=... OPS=... bash run.sh LOGNAME   (run from the repo root)
set -u
export OPENBLAS_NUM_THREADS=1
export FERRIC_TERF_TABLE_DIR="$PWD/terf-tables"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-3}"
LOG=benchmarks/terfc_screening_bound/logs/$1.log
start=$(date +%s)
scripts/ferric-limited --max=8G --high=7G -- cargo test -p ferric-mp2 --release \
  --test terfc_screen_probe -j4 -- --ignored --nocapture > "$LOG" 2>&1
echo "WALL $(( $(date +%s) - start )) s" >> "$LOG"
