#!/usr/bin/env bash
# Capped, detachable gate-test run: full ferric-mp2 suite + downstream
# linlccd tests. Log to $1.
set -uo pipefail
cd "$(dirname "$0")/../.."
log="$1"
{
  scripts/ferric-limited --max=4G --high=3600M -- \
    env OPENBLAS_NUM_THREADS=1 cargo test -p ferric-mp2 --release
  echo "GATE: ferric-mp2 exit $?"
  scripts/ferric-limited --max=4G --high=3600M -- \
    env OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cc --release --test linlccd_amplitude
  echo "GATE: ferric-cc linlccd exit $?"
  echo "GATE-DONE"
} >>"$log" 2>&1
