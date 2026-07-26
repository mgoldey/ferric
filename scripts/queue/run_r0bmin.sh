#!/bin/bash
# B-formulation sweep at the estimated minimum: r0 = 0.8 +/- 0.1, FULL A24.
# Broadens r0 optimization from 7 systems to all 24: these 13 had never
# been scanned at all. Runs 3-wide at RAYON=4, like the other parallel
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
  # A trimmed sweep has 1-3 points, not a fixed count -- read how many this
  # job's OWN toml asks for. A bare '>= 1' would mark a 3-point job complete
  # after one point and silently drop the rest.
  local want; want=$(grep -oP 'r0_sweep = \[\K[^]]*' "benchmarks/grid/toml/${key}.toml" 2>/dev/null | tr ',' '\n' | grep -c .)
  [ -z "$want" ] && want=1
  if [ -s "$out" ] && [ "$(grep -c 'Total energy' "$out" 2>/dev/null | head -1)" -ge "$want" ]; then
    echo "[skip] $key"; return
  fi
  for _ in $(seq 1 90); do
    [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 8 ] && break
    sleep 60
  done
  # LOCK: refuse to run a job another process is already running.
  # xargs -P re-dispatches from a static list, so if a stage is restarted (or
  # an earlier run's xargs survives), two processes can open the SAME output
  # file and truncate each other. Observed 2026-07-26: all three a24-21
  # fragments had duplicate writers and their .out files were clobbered to
  # 48 bytes with zero completed points. flock makes the second one skip.
  exec 9>"${out}.lock"
  if ! flock -n 9; then
    echo "[lock ] $key already running elsewhere"; return
  fi
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
  if [ $rc -eq 0 ] && [ "$n" -ge "$want" ]; then echo "[ok   ] ${el}s ${n}/${want}pts $key"
  else echo "[FAIL ] ${el}s ${n}pts $key rc=$rc"; fi
}
# Cheapest first, so a partial run still yields a usable curve.
export -f run
# Cheapest first by ATOM COUNT (22 is 10 atoms, 15 is 11) -- run_r0tscan.sh
# had these two transposed.
# Cheapest first by atom count (4,6,7,7,8,8,9,9,9,10,10,13,13). Systems are
# zero-padded to two digits in geoms/ (a24-01, not a24-1).
# Cheapest first by atom count. Only jobs the PLANNER kept exist as TOMLs --
# benchmarks/grid/plan_sweep.py trims each sweep to the r0 points not already
# on disk and deletes TOMLs that are fully covered, so this loop skips whatever
# is absent. Re-run the planner (idempotent) before relaunching to re-trim
# against newly-landed data.
for s in 04 02 03 20 01 06 21 05 08 09 12 24 07 10 16 11 13 19 22 15 14 23 17 18; do
  for f in dimer mA_cp mB_cp; do
    t="benchmarks/grid/toml/a24-${s}_${f}_aqz_r0Bmin_B.toml"
    [ -f "$t" ] && echo "a24-${s}_${f}_aqz_r0Bmin_B"
  done
done | xargs -P "$NPROC" -I{} bash -c 'run "$@"' _ {}
echo "=== DONE $(date +%H:%M) ==="
