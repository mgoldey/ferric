#!/bin/bash
# Resolve B and T r0 minima at 0.05 A resolution, near the ALREADY-LOCATED
# optima -- rather than continuing to sweep coarse low-r0 territory.
#
# Where the minima are (measured 2026-07-26, aQZ, CP):
#   T  spline min r0 = 1.433 A  (MAE 0.0414)  on the n=4 set with full 0.7-1.8
#   B  spline min r0 = 0.824 A  (MAE 0.0759)  on the n=10 set with 6 common r0
#
# Both are INTERIOR and bracketed, so they are real turning points, not
# boundary artifacts. But each spline straddles a coarse gap -- B's nearest
# sampled points are 0.750 and 1.000, a 0.25 A span -- so the quoted minimum is
# an interpolation across territory with no data. These runs fill it:
#
#   B: 0.80 / 0.85 / 0.90  on systems 2,5,12,14,15,19,21,22,23,24   (60 pts)
#   T: 1.40 / 1.45 / 1.50  on systems 12,21,22,24                   (12 pts)
#
# 72 points total, versus 273 for the remaining broad set -- and unlike the
# broad set these directly sharpen the numbers the manuscript quotes.
#
# The system sets are the ones with a COMMON r0 grid; mixing in systems that
# lack coverage would change the MAE by changing the sample, not the physics
# (mae_spline refuses to mix system counts for exactly this reason).
#
# TOMLs come from plan_sweep.py, which trims each job to only its MISSING r0
# points, so re-running this is safe and converges.
cd /home/matt/qc/ferric
. scripts/queue/memgate.sh
export FERRIC_TERF_TABLE_DIR=/home/matt/qc/terf-tables-data
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=4
NPROC=3

run() {
  local key="$1" out="benchmarks/grid/out/$1.out"
  local toml="benchmarks/grid/toml/${key}.toml"
  [ -f "$toml" ] || { echo "[skip ] $key (no toml -- already complete)"; return; }
  # How many points does THIS job's toml ask for? The plan trims per job, so a
  # FIXED threshold would be wrong here -- one job may need 3 points and its
  # neighbour 1. Verified against a known-complete run: 7 r0 points produce
  # exactly 7 "Total energy" lines, one per point and no extra header line.
  # (An earlier draft of this script added +1 for a supposed base reference.
  # That would have made every job permanently "incomplete" and rerun forever.)
  local want
  want=$(grep -o '[0-9]\+\.[0-9]\+' <<<"$(grep 'r0_sweep' "$toml")" | wc -l)
  if [ -s "$out" ] && [ "$(grep -c 'Total energy' "$out" 2>/dev/null)" -ge "$want" ]; then
    echo "[skip ] $key (has $want pts)"; return
  fi
  mem_wait "${SLOT_MB:-2600}" || echo "[mem  ] proceeding anyway for $key"
  exec 9>"${out}.lock"
  if ! flock -n 9; then echo "[lock ] $key running elsewhere"; return; fi
  local st; st=$(date +%s)
  scripts/ferric-limited --max=5G --high=4G -- ./target/release/ferric \
    "$toml" > "$out" 2> "${out}.err"
  local rc=$? el=$(( $(date +%s) - st ))
  local n; n=$(grep -c "Total energy" "$out" 2>/dev/null); [ -z "$n" ] && n=0
  if [ $rc -eq 0 ] && [ "$n" -ge "$want" ]; then echo "[ok   ] ${el}s ${n}/${want}pts $key"
  else echo "[FAIL ] ${el}s ${n}/${want}pts $key rc=$rc"; fi
}
export -f run mem_wait mem_avail_mb mem_psi_some10

# T first: only 12 points, and it sharpens the headline number (T is the
# formulation the manuscript leads with). Then B's 60.
{
  for s in 12 21 22 24; do
    for f in dimer mA_cp mB_cp; do echo "a24-${s}_${f}_aqz_r0Tfine_T"; done
  done
  for s in 02 05 12 14 15 19 21 22 23 24; do
    for f in dimer mA_cp mB_cp; do echo "a24-${s}_${f}_aqz_r0Bfine_B"; done
  done
} | xargs -P "$NPROC" -I{} bash -c 'run "$@"' _ {}
echo "=== DONE $(date +%H:%M) ==="
