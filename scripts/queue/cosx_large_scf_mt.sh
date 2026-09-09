#!/usr/bin/env bash
# PHASE 1 of the COSX large-system lane: converge a DENSITY on ALL CORES.
#
#   cosx_large_scf_mt.sh <system> <basis> <secs> [tag]
#
# The one-thread rule in this lane exists so that K-BUILD TIMINGS are comparable.
# A density is an INPUT, not a timing, so it may be produced on every core; the
# harness already labels the SCF line "default pool, not a timing". OpenBLAS
# stays at 1 thread (BLAS>1 under rayon is the openblas-rayon-dgetrf-crash
# hazard); the SCF's parallelism is rayon's.
#
# k_builder=link + direct J: DF-JK's two dressed 3-index tensors are 13.7-58 GB
# at these rungs and would spill to a 95%-full partition, which is exactly the
# defect this lane refuses to trigger.
#
# Windows are chained: MAXITER caps the iterations, the unconverged density is
# saved and flagged, and RESTART seeds the next window from it.
set -euo pipefail

cd "$(dirname "$0")/../.."
BIN=$(ls -t target/release/deps/cosx_full_k-* | grep -v '\.d$' | head -1)
SYS="$1"; BAS="$2"; SECS="$3"; TAG="${4:-w1}"

THREADS="${THREADS:-$(nproc)}"
DENS="scripts/queue/out/dens_${SYS}_${BAS//def2-/}.bin"
LOG="scripts/queue/out/cosx_large_${SYS}_${BAS//def2-/}_scfmt_${TAG}.log"

echo "=== SCF-MT $SYS/$BAS  threads=$THREADS  binary=$BIN  window=${SECS}s  tag=$TAG ==="
echo "PRE  PSI: $(grep full /proc/pressure/memory)"
echo "PRE  disk: $(df -h /home/matt | tail -1)"
echo "PRE  load: $(uptime)"

restart_env=()
[[ -n "${RESTART:-}" ]] && restart_env=(COSX_FK_SCF_RESTART_IN="$RESTART")

set +e
env OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS="$THREADS" FERRIC_MEM_BUDGET_GB="${BUDGET:-2}" \
    COSX_FK_SYSTEM="$SYS" COSX_FK_BASIS="$BAS" \
    "${restart_env[@]+"${restart_env[@]}"}" \
    COSX_FK_SCF="${SCFMODE:-link}" COSX_FK_SCF_DCONV="${DCONV:-1e-5}" \
    COSX_FK_DENSITY_OUT="$DENS" \
    COSX_FK_SCF_MAXITER="${MAXITER:-40}" \
    COSX_FK_SCF_SAVE_UNCONVERGED=1 \
    timeout "$SECS" scripts/ferric-limited --max="${MAXMEM:-6G}" --high="${HIGHMEM:-5G}" -- \
    "$BIN" --ignored --nocapture 2>&1 | tee "$LOG" | tail -30
rc=${PIPESTATUS[0]}
set -e

echo "POST rc=$rc"
echo "POST PSI: $(grep full /proc/pressure/memory)"
echo "POST disk: $(df -h /home/matt | tail -1)"
ls -la "$DENS" 2>/dev/null || echo "NO DENSITY WRITTEN"
