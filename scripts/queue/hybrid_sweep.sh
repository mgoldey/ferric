#!/usr/bin/env bash
# Hybrid rank x thread ratio sweep driver (scratch; scripts/queue is gitignored).
#
# The honest grid is over the 6 PHYSICAL cores: 1x6, 2x3, 3x2, 6x1. Every one of
# those uses exactly 6 physical cores, so they are directly comparable. The
# 12-logical configurations are run SEPARATELY and must not be mixed into the
# same comparison (SMT would masquerade as a hybrid win).
#
# --bind-to core --map-by slot:PE=<threads> gives each rank a distinct set of
# <threads> physical cores, which is the whole point: without binding, N ranks
# land wherever the OS puts them and the measurement is of the scheduler.
set -u
BIN="$1"
TEST="$2"
export OPENBLAS_NUM_THREADS=1
MPIRUN="${MPIRUN:-$(command -v mpirun || echo "$HOME/.local/bin/mpirun")}"

run_cfg() {
  local ranks="$1" pe="$2" tag="$3"
  # PE=n asks Open MPI for n processing elements per rank; under
  # `--bind-to core` a processing element IS a physical core, so
  # ranks x PE = 6 covers the box exactly once and SMT siblings are never
  # handed out as if they were independent cores. (`--cpu-set 0-5` would be
  # the other way to pin to the six physical cores, but Open MPI 4.1 rejects
  # it as a conflicting binding policy when PE=n is also given.)
  timeout 1800 "$MPIRUN" -np "$ranks" --bind-to core \
      --map-by "slot:PE=$pe" \
      "$BIN" --nocapture --test-threads=1 --ignored "$TEST" 2>&1 \
    | grep -E "RATIO_SWEEP" | sed "s/^/[$tag] /"
}

echo "### 6-physical-core grid (comparable set)"
run_cfg 1 6 "1x6"
run_cfg 2 3 "2x3"
run_cfg 3 2 "3x2"
run_cfg 6 1 "6x1"
