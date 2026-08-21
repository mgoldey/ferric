#!/bin/bash
# T-formulation r0 EXTENSION scan, r0 = 1.4-1.8 step 0.1, aQZ CP, 7-dimer subset.
#
# The 0.7-1.3 scan is monotonic across its whole window (MAE 0.2608 -> 0.0988
# kcal/mol at r0=1.3), so the optimum is OUTSIDE it. Large r0 is trustworthy:
# the terfc "far-field overshoot" was the RI error of the Coulomb reference,
# not an integral bug (agent a52a1da, merged). Smoke-tested at r0=1.8.
#
# PARALLEL, unlike run_r0tscan.sh's serial loop. Measured: one aQZ job peaks at
# ~2.2 GB RSS, and RAYON=12 does not saturate 12 cores on these small systems.
# 3 concurrent jobs at RAYON=4 => ~6.6 GB, all 12 cores busy. Serial was going
# to take ~7.7 h for this scan alone.
#
# The 8G floor is per-slot AND checked before each dispatch, so a memory spike
# stalls new work instead of stacking onto it (the box OOMed twice on 2026-07-25).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. scripts/queue/memgate.sh
export FERRIC_TERF_TABLE_DIR="${FERRIC_TERF_TABLE_DIR:-$HOME/qc/terf-tables-data}"
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=4
NPROC=3

run() {
  local key="$1" out="benchmarks/grid/out/$1.out"
  # Idempotent: a complete job has all 5 points.
  if [ -s "$out" ] && [ "$(grep -c 'Total energy' "$out" 2>/dev/null | head -1)" -ge 5 ]; then
    echo "[skip] $key"; return
  fi
  # Gate on AVAILABLE (not free) -- most of "used" is reclaimable page cache.
  mem_wait "${SLOT_MB:-2600}" || echo "[mem  ] proceeding anyway for $key"
  # LOCK: see run_r0bmin.sh -- xargs -P can re-dispatch a job whose output is
  # already being written, and the two writers truncate each other.
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
  local n; n=$(grep -c "Total energy" "$out" 2>/dev/null | head -1)
  [ -z "$n" ] && n=0
  if [ $rc -eq 0 ] && [ "$n" -ge 5 ]; then echo "[ok   ] ${el}s ${n}pts $key"
  else echo "[FAIL ] ${el}s ${n}pts $key rc=$rc"; fi
}
export -f run mem_wait mem_avail_mb mem_psi_some10

# Cheapest first (by atom count: 6,7,8,8,10,11,12), so a partial run still
# yields a usable curve. NOTE 22 (10 atoms) precedes 15 (11) -- run_r0tscan.sh
# had these transposed.
for s in 20 21 12 24 22 15 23; do
  for f in dimer mA_cp mB_cp; do echo "a24-${s}_${f}_aqz_r0text_T"; done
done | xargs -P "$NPROC" -I{} bash -c 'run "$@"' _ {}
echo "=== DONE $(date '+%H:%M') ==="
