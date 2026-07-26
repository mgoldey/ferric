#!/bin/bash
# T-formulation scan, r0 = 0.7-1.3 step 0.1, aQZ CP, 7-dimer subset.
# T (coupled-rings) over 0.7-1.3 A. NOTE: T is 1st order in v_lr where B is 3rd,
# so the lane notes put T's own window at LARGE r0 (omega <= 0.1-0.15 A^-1).
# This 0.7-1.3 window is user-specified; if the curve comes back monotonic the
# optimum is likely OUTSIDE it, toward larger r0.
# One job per (system, fragment); each sweeps all 6 r0 on a single SCF.
cd /home/matt/qc/ferric
export FERRIC_TERF_TABLE_DIR=/home/matt/qc/terf-tables-data
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=12
FAIL=0; OK=0
run() {
  local key="$1" out="benchmarks/grid/out/$1.out"
  # Idempotent: a complete job has all 6 points.
  if [ -s "$out" ] && [ "$(grep -c 'Total energy' "$out")" -ge 7 ]; then
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
  local n; n=$(grep -c "Total energy" "$out" 2>/dev/null || echo 0)
  if [ $rc -eq 0 ] && [ "$n" -ge 7 ]; then echo "[ok   ] ${el}s ${n}pts $key"; OK=$((OK+1))
  else echo "[FAIL ] ${el}s ${n}pts $key rc=$rc"; FAIL=$((FAIL+1)); fi
}
# Cheapest first, so a partial run still yields a usable curve.
for s in 20 21 12 24 15 22 23; do
  for f in dimer mA_cp mB_cp; do run "a24-${s}_${f}_aqz_r0tscan_T"; done
done
echo "=== DONE: ${OK} ok, ${FAIL} failed ==="
