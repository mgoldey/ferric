#!/bin/bash
# Spline-informed aQZ r0 scans over the 7-dimer subset, B then T.
# One job per (system, fragment, formulation); each job sweeps all r0 internally.
cd /home/matt/qc/ferric
export FERRIC_TERF_TABLE_DIR=/home/matt/qc/terf-tables-data
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=12
FAIL=0
run() {
  local key="$1" out="benchmarks/grid/out/$1.out"
  if [ -s "$out" ] && grep -q "Total energy" "$out"; then echo "[skip] $key"; return; fi
  # Wait for headroom rather than aborting the whole run.
  for _ in $(seq 1 60); do
    [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 5 ] && break
    sleep 60
  done
  local st; st=$(date +%s)
  scripts/ferric-limited -- ./target/release/ferric \
    "benchmarks/grid/toml/${key}.toml" > "$out" 2> "${out}.err"
  local rc=$? el=$(( $(date +%s) - st ))
  local n; n=$(grep -c "Total energy" "$out" 2>/dev/null || echo 0)
  if [ $rc -eq 0 ] && [ "$n" -gt 0 ]; then echo "[ok   ] ${el}s ${n}pts $key"
  else echo "[FAIL ] ${el}s $key rc=$rc"; FAIL=$((FAIL+1)); fi
}
for form in B T; do
  for s in 20 21 12 24 15 22 23; do        # cheapest first
    for f in dimer mA_cp mB_cp; do
      run "a24-${s}_${f}_aqz_r0scan_${form}"
    done
  done
  echo "=== formulation ${form} complete ==="
done
echo "=== ALL DONE ($FAIL failures) ==="
