#!/usr/bin/env bash
# Run a ferric-cli job with FERRIC_MEM_TRACE=1, which prints "[ferric-mem] ... rss=<bytes>"
# lines at the traced entry points (DLPNO-MP2, amplitude LMP2) to stderr.
# Usage: scripts/measure_mem.sh [ferric-cli args...] <input.toml>
set -euo pipefail
export FERRIC_MEM_TRACE=1
export OPENBLAS_NUM_THREADS=1
exec cargo run --release -p ferric-cli --bin ferric-cli -- "$@"
