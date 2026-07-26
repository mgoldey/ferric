#!/bin/bash
# Comparison methods on the FULL A24 at aug-cc-pVTZ, CP, published parameters.
#   mp2        plain RI-MP2 (baseline)
#   atterfc    attenuated MP2, erfc, omega = 0.420 A^-1  (dissertation optimal)
#   scs2terfc  SCS-MP2(2terfc), r0 0.75/1.05 A, c_OS 1.27, c_SS 4.05 (JCTC 2015)
#
# These fill the gap the manuscript's lineage argument rests on: every
# predecessor B/T claim to supersede has been cited but never measured here.
# Waits for the T r0-scan to drain first so the box is not oversubscribed.
set -u
cd /home/matt/qc/ferric
export FERRIC_TERF_TABLE_DIR=/home/matt/qc/terf-tables-data
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=12
OK=0; FAIL=0
LOG() { echo "[$(date +%H:%M)] $*"; }

LOG "waiting for the T r0-scan to finish"
while pgrep -f run_r0tscan.sh > /dev/null; do sleep 120; done
LOG "T scan drained; starting the aTZ gather"

run() {
  local key="$1" out="benchmarks/grid/out/$1.out"
  if [ -s "$out" ] && grep -q "Total energy\|Total      =" "$out"; then
    echo "[skip] $key"; return
  fi
  for _ in $(seq 1 90); do
    [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 8 ] && break
    sleep 60
  done
  local st; st=$(date +%s)
  scripts/ferric-limited -- ./target/release/ferric \
    "benchmarks/grid/toml/${key}.toml" > "$out" 2> "${out}.err"
  local rc=$? el=$(( $(date +%s) - st ))
  if [ $rc -eq 0 ] && grep -q "Total" "$out"; then
    echo "[ok   ] ${el}s $key"; OK=$((OK+1))
  else
    echo "[FAIL ] ${el}s $key rc=$rc"; FAIL=$((FAIL+1))
  fi
}

# Cheapest systems first so a partial run still yields a usable subset.
for meth in mp2 atterfc scs2terfc; do
  for s in 20 21 19 15 12 24 22 08 07 18 17 23 13 11 16 06 14 10 05 09 04 03 02 01; do
    for f in dimer mA_cp mB_cp; do
      [ -f "benchmarks/grid/toml/a24-${s}_${f}_atz_${meth}.toml" ] && \
        run "a24-${s}_${f}_atz_${meth}"
    done
  done
  LOG "=== ${meth} complete ==="
done
LOG "=== GATHER DONE: ${OK} ok, ${FAIL} failed ==="
