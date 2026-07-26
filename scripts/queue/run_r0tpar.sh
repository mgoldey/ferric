#!/bin/bash
# T-formulation scan (PARALLEL TAIL), r0 = 0.7-1.3, aQZ CP.
# Same jobs as run_r0tscan.sh but 3-wide at RAYON=4 instead of serial at
# RAYON=12. Measured 2026-07-26: the serial tail had ~6.9 h left at 1044% CPU
# on ONE job, while 2.2 GB/job against 16 GB available left the box mostly
# idle. Idempotent, so it simply skips the 4 systems already finished.
# T (coupled-rings) over 0.7-1.3 A. NOTE: T is 1st order in v_lr where B is 3rd,
# so the lane notes put T's own window at LARGE r0 (omega <= 0.1-0.15 A^-1).
# This 0.7-1.3 window is user-specified; if the curve comes back monotonic the
# optimum is likely OUTSIDE it, toward larger r0.
# One job per (system, fragment); each sweeps all 6 r0 on a single SCF.
cd /home/matt/qc/ferric
export FERRIC_TERF_TABLE_DIR=/home/matt/qc/terf-tables-data
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=4
NPROC=3
FAIL=0; OK=0
run() {
  local key="$1" out="benchmarks/grid/out/$1.out"
  # Idempotent: a complete job has all 6 points.
  if [ -s "$out" ] && [ "$(grep -c 'Total energy' "$out" 2>/dev/null | head -1)" -ge 7 ]; then
    echo "[skip] $key"; return
  fi
  for _ in $(seq 1 90); do
    [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 8 ] && break
    sleep 60
  done
  local st; st=$(date +%s)
  # Per-SLOT memory cap. ferric-limited's default is 12G per job, which is
  # right for ONE job but lets NPROC=3 collectively reach 36G on a 23GB box --
  # the cgroup bounds a single runaway job, not three cooperating ones.
  # Measured RSS grows with system size (2.2G at 6-8 atoms, 3.1G at 10), so
  # size the slot for the LARGEST system, not the smallest.
  scripts/ferric-limited --max=5G --high=4G -- ./target/release/ferric \
    "benchmarks/grid/toml/${key}.toml" > "$out" 2> "${out}.err"
  local rc=$? el=$(( $(date +%s) - st ))
  local n; n=$(grep -c "Total energy" "$out" 2>/dev/null | head -1); [ -z "$n" ] && n=0
  if [ $rc -eq 0 ] && [ "$n" -ge 7 ]; then echo "[ok   ] ${el}s ${n}pts $key"
  else echo "[FAIL ] ${el}s ${n}pts $key rc=$rc"; fi
}
# Cheapest first, so a partial run still yields a usable curve.
export -f run
# Cheapest first by ATOM COUNT (22 is 10 atoms, 15 is 11) -- run_r0tscan.sh
# had these two transposed.
for s in 20 21 12 24 22 15 23; do
  for f in dimer mA_cp mB_cp; do echo "a24-${s}_${f}_aqz_r0tscan_T"; done
done | xargs -P "$NPROC" -I{} bash -c 'run "$@"' _ {}
echo "=== DONE $(date +%H:%M) ==="
