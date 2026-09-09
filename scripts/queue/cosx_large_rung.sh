#!/usr/bin/env bash
# One rung of the COSX large-system ceiling map: ONE (system, basis) cell in ONE
# foreground window under the cgroup cap. See scripts/queue/out/cosx_large_prereg.md.
#
#   cosx_large_rung.sh scf   <system> <basis> <secs>   # converge a density (k_builder=link)
#   cosx_large_rung.sh build <system> <basis> <secs> [extra env...]  # time the K builders
#
# Never run two of these at once: every timing in this lane is one thread on an
# otherwise idle box, and PSI is checked before and after.
set -euo pipefail

cd "$(dirname "$0")/../.."
BIN=$(ls -t target/release/deps/cosx_full_k-* | grep -v '\.d$' | head -1)
MODE="$1"; SYS="$2"; BAS="$3"; SECS="$4"; shift 4

DENS="scripts/queue/out/dens_${SYS}_${BAS//def2-/}.bin"
LOG="scripts/queue/out/cosx_large_${SYS}_${BAS//def2-/}_${MODE}.log"

echo "=== $MODE $SYS/$BAS  binary=$BIN  window=${SECS}s ==="
echo "PRE  PSI: $(grep full /proc/pressure/memory)"
echo "PRE  disk: $(df -h /home/matt | tail -1)"

common=(OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 FERRIC_MEM_BUDGET_GB=2
        COSX_FK_SYSTEM="$SYS" COSX_FK_BASIS="$BAS")

case "$MODE" in
  scf)
    # LinK-K + direct J. Post-a272333c LinK is correct at the SCF level
    # (tests/link_scf_anchor.rs), and it is far cheaper than the direct J+K SCF,
    # which is what makes the upper rungs reachable at all.
    # RESTART is passed only when non-empty: the harness treats the var's mere
    # PRESENCE as "restart from this path", so an empty value would try to read "".
    restart_env=()
    [[ -n "${RESTART:-}" ]] && restart_env=(COSX_FK_SCF_RESTART_IN="$RESTART")
    env "${common[@]}" "${restart_env[@]+"${restart_env[@]}"}" \
      COSX_FK_SCF=link COSX_FK_SCF_DCONV=1e-5 \
      COSX_FK_DENSITY_OUT="$DENS" \
      COSX_FK_SCF_MAXITER="${MAXITER:-40}" \
      COSX_FK_SCF_SAVE_UNCONVERGED="${SAVE_UNCONV:-1}" \
      timeout "$SECS" scripts/ferric-limited --max=6G --high=5G -- \
      "$BIN" --ignored --nocapture 2>&1 | tee "$LOG" | tail -20
    ;;
  build)
    env "${common[@]}" "$@" \
      COSX_FK_DENSITY_IN="$DENS" COSX_FK_DFK=1 \
      timeout "$SECS" scripts/ferric-limited --max=6G --high=5G -- \
      "$BIN" --ignored --nocapture 2>&1 | tee "$LOG" | tail -24
    ;;
  *) echo "mode must be scf|build" >&2; exit 2 ;;
esac

echo "POST PSI: $(grep full /proc/pressure/memory)"
echo "POST disk: $(df -h /home/matt | tail -1)"
