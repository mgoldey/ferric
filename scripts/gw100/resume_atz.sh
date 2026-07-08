#!/usr/bin/env bash
# Resume the GW100 aug-cc-pVTZ sweep.
#
# aug-cc-pVDZ is complete (93/93 + 7 ECP = 100/100). parallel_complete.py is
# idempotent — it skips every already-converged/failed row across BOTH bases, so
# this only runs the remaining aTZ molecules (51 as of 2026-07-07).
#
# Standing constraints baked in:
#   - memory-scoped via systemd-run --user --scope (MemoryMax, no swap)
#   - OPENBLAS_NUM_THREADS=1, rayon owns parallelism
#   - one molecule at a time on all cores (1 worker x RAYON threads), NOT parallel
#     workers — the aTZ tail (aromatics/nucleobases) is memory-bound (a single
#     benzene-class aTZ RPA job ~17 GB RSS), so concurrent workers oversubscribe.
#   - detached with direct file output (NOT `| tail`) so it survives a teardown.
#
# Usage:
#   scripts/gw100/resume_atz.sh                 # build + resume, defaults below
#   RAYON=12 MEM=20G BUDGET=10800 scripts/gw100/resume_atz.sh
#   SKIP_BUILD=1 scripts/gw100/resume_atz.sh    # skip the rebuild step
#
# Knobs (env overrides):
#   RAYON   rayon threads for the single worker      (default 12)
#   MEM     MemoryMax for the sweep scope            (default 20G)
#   BUDGET  GW100_MOL_BUDGET seconds/molecule        (default 5400)
#   WORKERS parallel_complete workers                (default 1; 2 is the safe
#           upper bound on this box, small-molecule end only)
set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

RAYON="${RAYON:-12}"
MEM="${MEM:-20G}"
BUDGET="${BUDGET:-5400}"
WORKERS="${WORKERS:-1}"
LOG="scripts/queue/out/gw100_atz_resume.log"
mkdir -p scripts/queue/out

# 1. Rebuild the driver so the aTZ run uses the current SCF (SAD-default guess).
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo ">> building gw100_full (SAD-default SCF)..."
  systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 -u gw100_atz_build \
    env OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS="$RAYON" \
    cargo build --release --example gw100_full -p ferric-benchmarks
fi

# 2. Resume the aTZ sweep, detached with direct file output (survives teardown).
echo ">> resuming aTZ sweep: WORKERS=$WORKERS RAYON=$RAYON MEM=$MEM BUDGET=${BUDGET}s"
echo ">> log: $LOG"
systemd-run --user --scope -p MemoryMax="$MEM" -p MemorySwapMax=0 -u gw100_atz_sweep \
  env OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS="$RAYON" GW100_MOL_BUDGET="$BUDGET" \
  bash -c "python3 scripts/gw100/parallel_complete.py $WORKERS $RAYON > $LOG 2>&1" &
disown

echo ">> launched (detached). tail progress with:  tail -f $LOG"
