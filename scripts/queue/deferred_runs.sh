#!/usr/bin/env bash
# Deferred-run queue: launch heavy ferric jobs ONLY when the box is free, so they
# never contend with an active benchmark grid. Each job is memory-scoped (8 GB,
# no swap), single-thread (OPENBLAS/RAYON=1), nice'd, and idle-I/O — it yields to
# anything already running.
#
# Gate: refuse to start unless 1-min load < LOAD_MAX AND free mem > MEM_MIN_GB.
# Idempotent per job: skips a job whose output file already exists.
#
# Usage:
#   scripts/queue/deferred_runs.sh check     # report gate status, run nothing
#   scripts/queue/deferred_runs.sh gw100     # run GW100 re-verification if gate open
#   scripts/queue/deferred_runs.sh trunc <mol> <basis>   # deferred truncation spike
#   scripts/queue/deferred_runs.sh all       # gw100 then trunc (benzene), gated
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/scripts/queue/out"
mkdir -p "$OUT"
LOAD_MAX="${LOAD_MAX:-6.0}"     # 1-min load ceiling to start a heavy job
MEM_MIN_GB="${MEM_MIN_GB:-10}"  # free GB floor

load1() { awk '{print $1}' /proc/loadavg; }
freemem_gb() { free -g | awk '/^Mem:/ {print $7}'; }

gate_open() {
  local l m
  l=$(load1); m=$(freemem_gb)
  awk -v l="$l" -v lm="$LOAD_MAX" 'BEGIN{exit !(l < lm)}' || { echo "GATE CLOSED: load $l >= $LOAD_MAX"; return 1; }
  [ "$m" -ge "$MEM_MIN_GB" ] || { echo "GATE CLOSED: free ${m}G < ${MEM_MIN_GB}G"; return 1; }
  echo "GATE OPEN: load $l < $LOAD_MAX, free ${m}G >= ${MEM_MIN_GB}G"
  return 0
}

scoped() {  # scoped <unit> <logfile> <cmd...>
  local unit="$1" log="$2"; shift 2
  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p CPUWeight=20 \
    --quiet --unit="$unit" -- \
    nice -n 15 ionice -c3 \
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
    "$@" > "$log" 2>&1
}

run_gw100() {
  local log="$OUT/gw100_full_$(date +%Y%m%d_%H%M).txt"
  echo "[queue] GW100 re-verification -> $log"
  scoped gw100-run "$log" \
    "$ROOT/target/release/examples/gw100_full"
  echo "[queue] GW100 done. MAE table:"; grep -E "^MAE|^mol|^----" "$log" | head
}

# GW100 (10-mol subset, all methods) at a given basis, via the idempotent runner
# so results.json accumulates and re-runs are skippable.
run_gw100_basis() {
  local basis="$1"
  echo "[queue] GW100 sweep basis=$basis (idempotent) ..."
  scoped "gw100-$basis" "$OUT/gw100_${basis}_$(date +%Y%m%d_%H%M).txt" \
    python3 "$ROOT/scripts/gw100/run_sweep.py" "$basis"
}

run_trunc() {
  local mol="${1:-benzene}" basis="${2:-aug-cc-pvdz}"
  local log="$OUT/trunc_${mol}_${basis}_$(date +%Y%m%d_%H%M).txt"
  echo "[queue] truncation spike $mol/$basis -> $log"
  scoped trunc-run "$log" \
    "$ROOT/target/release/examples/pdep_trunc_trustmap" "$mol" "$basis"
  echo "[queue] trunc done."; grep -E "^#|thresh|----|^[0-9]" "$log" | tail -20
}

case "${1:-check}" in
  check) gate_open ;;
  gw100) gate_open && run_gw100 ;;
  trunc) shift; gate_open && run_trunc "$@" ;;
  # Full method×threshold benchmark: the mid-size + large systems (water already
  # done). Each gated independently is overkill; gate once, run both serially.
  trustmap) gate_open || exit 1
            run_trunc ethylene aug-cc-pvdz
            run_trunc benzene  aug-cc-pvdz ;;
  # GW100 (10-mol subset) at BOTH bases — gate once, run aDZ then aTZ serially.
  gw100-bases) gate_open || exit 1
               run_gw100_basis aug-cc-pvdz
               run_gw100_basis aug-cc-pvtz
               python3 "$ROOT/scripts/gw100/run_sweep.py" --show ;;
  all)   gate_open || exit 1; run_gw100; run_trunc ethylene aug-cc-pvdz; run_trunc benzene aug-cc-pvdz ;;
  *) echo "usage: $0 {check|gw100|trunc <mol> <basis>|trustmap|gw100-bases|all}"; exit 2 ;;
esac
