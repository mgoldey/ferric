#!/usr/bin/env bash
# Detachable one-system tail-sweep piece for bench_direct_alkane_series.
# Usage: run_tail_piece.sh <nc> <logfile>
# Runs under scripts/ferric-limited (3G cap) so it is self-limiting on the
# shared box; meant to be launched via nohup/setsid so a harness-level
# task kill does not lose the measurement.
set -euo pipefail
cd "$(dirname "$0")/../.."
nc="$1"
log="$2"
exec scripts/ferric-limited --max=4G --high=3600M -- \
  env OPENBLAS_NUM_THREADS=1 FERRIC_MEM_BUDGET_GB=2 \
  FERRIC_LMP2_BENCH_MIN_C="$nc" FERRIC_LMP2_BENCH_MAX_C="$nc" \
  cargo test -p ferric-mp2 --release --test lmp2_direct -- \
  --ignored --nocapture bench_direct_alkane_series >>"$log" 2>&1
