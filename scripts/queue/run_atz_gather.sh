#!/bin/bash
# Comparison methods on the FULL A24 at aug-cc-pVTZ, CP, published parameters.
#   mp2        plain RI-MP2 (baseline)
#   atterfc    attenuated MP2, erfc, omega = 0.420 A^-1  (dissertation optimal)
#   scs2terfc  SCS-MP2(2terfc), r0 0.75/1.05 A, c_OS 1.27, c_SS 4.05 (JCTC 2015)
#   mp2v       MP2-V, r0 1.00 A, b 11.0, C 0.0089, terfc (JCTC 11, 4159 (2015))
#
# These fill the gap the manuscript's lineage argument rests on: every
# predecessor B/T claims to supersede has been cited but never measured here.
#
# aTZ is the RIGHT basis for these parameters -- all four published
# parameterizations above are fitted at aug-cc-pVTZ specifically.
#
# PARALLEL (3 jobs at RAYON=4). Measured: one aQZ job peaks at ~2.2 GB and
# RAYON=12 does not saturate 12 cores on 6-12 atom systems; aTZ is cheaper
# still. Serial at RAYON=12 was leaving most of the box idle.
#
# Waits for BOTH T scans (r0tscan and the r0text extension) so the box is not
# oversubscribed -- the old version waited only on r0tscan and would have
# collided with the extension.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. scripts/queue/memgate.sh
export FERRIC_TERF_TABLE_DIR="${FERRIC_TERF_TABLE_DIR:-$HOME/qc/terf-tables-data}"
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=4
NPROC=3
LOG() { echo "[$(date +%H:%M)] $*"; }

LOG "waiting for ALL T r0 scans (r0tscan, r0tpar, r0text, r0broad) to drain"
# Gate on BOTH the driver scripts AND any running r0-scan ferric job. The
# driver-only check has a hole: a driver can exit while its last job is still
# finishing, and a killed driver leaves jobs orphaned but running. Either way
# the gather would start alongside them -- 3+3 concurrent jobs on a box where
# 3 already saturate the CPU, at double the memory the per-slot caps assume.
r0_busy() {
  pgrep -f 'run_r0tscan.sh|run_r0tpar.sh|run_r0text.sh|run_r0broad.sh|run_r0bmin.sh' > /dev/null && return 0
  local p c
  for p in $(pgrep -x ferric 2>/dev/null); do
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$c" in *_r0t*_T.toml|*_r0Bmin_B.toml) return 0;; esac
  done
  return 1
}
while r0_busy; do sleep 120; done
sleep 90          # settle, then re-check: quiet may be a gap between dispatches
while r0_busy; do sleep 120; done
LOG "T scans drained; starting the aTZ gather"

run() {
  local key="$1" out="benchmarks/grid/out/$1.out"
  if [ -s "$out" ] && grep -q "Total energy\|Total      =" "$out"; then
    echo "[skip] $key"; return
  fi
  # Gate on AVAILABLE, not free -- most of "used" is reclaimable page cache.
  mem_wait "${SLOT_MB:-2600}" || echo "[mem  ] proceeding anyway for $key"
  local st; st=$(date +%s)
  # Per-SLOT memory cap. ferric-limited's default is 12G per job, which is
  # right for ONE job but lets NPROC=3 collectively reach 36G on a 23GB box --
  # the cgroup bounds a single runaway job, not three cooperating ones.
  # Measured RSS grows with system size (2.2G at 6-8 atoms, 3.1G at 10), so
  # size the slot for the LARGEST system, not the smallest.
  scripts/ferric-limited --max=5G --high=4G -- ./target/release/ferric \
    "benchmarks/grid/toml/${key}.toml" > "$out" 2> "${out}.err"
  local rc=$? el=$(( $(date +%s) - st ))
  if [ $rc -eq 0 ] && grep -q "Total" "$out"; then echo "[ok   ] ${el}s $key"
  else echo "[FAIL ] ${el}s $key rc=$rc"; fi
}
export -f run mem_wait mem_avail_mb mem_psi_some10

# Cheapest systems first so a partial run still yields a usable subset.
for meth in mp2 atterfc scs2terfc mp2v; do
  for s in 20 21 19 15 12 24 22 08 07 18 17 23 13 11 16 06 14 10 05 09 04 03 02 01; do
    for f in dimer mA_cp mB_cp; do
      t="benchmarks/grid/toml/a24-${s}_${f}_atz_${meth}.toml"
      [ -f "$t" ] && echo "a24-${s}_${f}_atz_${meth}"
    done
  done | xargs -P "$NPROC" -I{} bash -c 'run "$@"' _ {}
  LOG "=== ${meth} complete ==="
done
LOG "=== GATHER DONE ==="
