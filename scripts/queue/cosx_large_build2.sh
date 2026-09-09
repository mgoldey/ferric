#!/usr/bin/env bash
# PHASE 2 of the COSX large-system lane: time the K builders at ONE THREAD on a
# density that Phase 1 already converged. Two sub-modes, because the harness runs
# direct-K BEFORE LinK in one process and a direct-K timeout would then also cost
# the COSX/LinK numbers:
#
#   cosx_large_build2.sh main   <system> <basis> <secs>   # COSX + LinK + DF-K refusal; saves K_cosx
#   cosx_large_build2.sh direct <system> <basis> <secs>   # direct four-centre K only, vs the saved K_cosx
#
# One thread everywhere, PSI printed before and after, cpu ~ wall asserted by the
# harness's own timed() prints. Never run two of these at once.
set -euo pipefail

cd "$(dirname "$0")/../.."
BIN=$(ls -t target/release/deps/cosx_full_k-* | grep -v '\.d$' | head -1)
SUB="$1"; SYS="$2"; BAS="$3"; SECS="$4"; shift 4

DENS="scripts/queue/out/dens_${SYS}_${BAS//def2-/}.bin"
KCOSX="scripts/queue/out/kcosx_${SYS}_${BAS//def2-/}.bin"
LOG="scripts/queue/out/cosx_large_${SYS}_${BAS//def2-/}_build_${SUB}.log"

[[ -f "$DENS" ]] || { echo "no density at $DENS -- run Phase 1 first" >&2; exit 2; }

echo "=== BUILD/$SUB $SYS/$BAS  binary=$BIN  window=${SECS}s ==="
echo "PRE  PSI: $(grep full /proc/pressure/memory)"
echo "PRE  disk: $(df -h /home/matt | tail -1)"
echo "PRE  load: $(uptime)"

common=(OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 FERRIC_MEM_BUDGET_GB=2
        COSX_FK_SYSTEM="$SYS" COSX_FK_BASIS="$BAS" COSX_FK_DENSITY_IN="$DENS")

case "$SUB" in
  main)   arm=(COSX_FK_LINK=1 COSX_FK_DFK=1 COSX_FK_K_OUT="$KCOSX") ;;
  direct) arm=(COSX_FK_BUILDS=0 COSX_FK_K_IN="$KCOSX" COSX_FK_DIRECT=1) ;;
  *) echo "sub must be main|direct" >&2; exit 2 ;;
esac

set +e
env "${common[@]}" "${arm[@]}" "$@" \
  timeout "$SECS" scripts/ferric-limited --max=6G --high=5G -- \
  "$BIN" --ignored --nocapture 2>&1 | tee "$LOG" | tail -26
rc=${PIPESTATUS[0]}
set -e

echo "POST rc=$rc"
echo "POST PSI: $(grep full /proc/pressure/memory)"
echo "POST disk: $(df -h /home/matt | tail -1)"
