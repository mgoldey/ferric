#!/bin/bash
# B-formulation fine scan, r0 = 0.76..0.81 step 0.01, aQZ CP, 7-dimer subset.
# Refines the r0 = 0.790 minimum located by mae_spline.py on the coarse grid.
# One job per (system, fragment); each sweeps all 6 r0 on a single SCF.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export FERRIC_TERF_TABLE_DIR="${FERRIC_TERF_TABLE_DIR:-$HOME/qc/terf-tables-data}"
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=12
FAIL=0; OK=0
run() {
  local key="$1" out="benchmarks/grid/out/$1.out"
  # Idempotent: a complete job has all 6 points.
  if [ -s "$out" ] && [ "$(grep -c 'Total energy' "$out")" -ge 6 ]; then
    echo "[skip] $key"; return
  fi
  for _ in $(seq 1 90); do
    [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 5 ] && break
    sleep 60
  done
  local st; st=$(date +%s)
  scripts/ferric-limited -- ./target/release/ferric \
    "benchmarks/grid/toml/${key}.toml" > "$out" 2> "${out}.err"
  local rc=$? el=$(( $(date +%s) - st ))
  local n; n=$(grep -c "Total energy" "$out" 2>/dev/null || echo 0)
  if [ $rc -eq 0 ] && [ "$n" -ge 6 ]; then echo "[ok   ] ${el}s ${n}pts $key"; OK=$((OK+1))
  else echo "[FAIL ] ${el}s ${n}pts $key rc=$rc"; FAIL=$((FAIL+1)); fi
}
# Cheapest first, so a partial run still yields a usable curve.
for s in 20 21 12 24 15 22 23; do
  for f in dimer mA_cp mB_cp; do run "a24-${s}_${f}_aqz_r0fine_B"; done
done
echo "=== DONE: ${OK} ok, ${FAIL} failed ==="
