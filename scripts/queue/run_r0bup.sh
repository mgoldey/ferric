#!/bin/bash
# Extend B upward: the FULL-A24 minimum is above 0.90, outside the +/-0.1
# bracket that was built around the 10-system estimate.
#
# THE FINDING THAT MOTIVATES THIS (2026-07-27)
#
# a24-17/18 -- the two 13-atom systems, the last ones missing -- finally landed,
# giving B a complete 24-system curve for the first time:
#
#   n=24   r0 0.70 -> 0.1222,  0.80 -> 0.0986,  0.90 -> 0.0926   STILL FALLING
#   n=10   r0 0.80 -> 0.0736,  0.90 -> 0.0766   interior min 0.8353
#
# The 10-system subset turns at ~0.835; the full set does not turn at all
# within 0.70-0.90. So the 14 systems added since that estimate pull the
# optimum to LARGER r0, and 0.8353 is a property of the subset, not of A24.
#
# This is the same sampling-bias lesson that already bit the T scan, in the
# other direction: there the broad set made the benchmark HARDER at fixed r0;
# here it MOVES the optimum. Neither is visible without the full set.
#
# So: add 1.00 and 1.10 to bracket the real minimum from above. 10 systems
# already have 1.00 (2,5,12,14,15,19,21,22,23,24), so the planner trims those.
#
# SCOPE: 12 systems here (1,3,4,6,7,8,9,10,11,13,16,20 plus the 10 needing only
# 1.10). a24-17/18 are EXCLUDED -- they need the 10 GB slot and run under
# run_r0big.sh, which has them queued.
#
# CONCURRENCY: NPROC=2 at a 3.4 GB slot. Do not raise without re-measuring;
# capacity on this box has changed hour to hour as job sizes grew.
cd /home/matt/qc/ferric
. scripts/queue/memgate.sh
export FERRIC_TERF_TABLE_DIR=/home/matt/qc/terf-tables-data
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=4
NPROC=2

run() {
  local key="$1" out="benchmarks/grid/out/$1.out"
  local toml="benchmarks/grid/toml/${key}.toml"
  [ -f "$toml" ] || { echo "[skip ] $key (no toml)"; return; }
  local want
  want=$(grep -o '[0-9]\+\.[0-9]\+' <<<"$(grep 'r0_sweep' "$toml")" | wc -l)
  if [ -s "$out" ] && [ "$(grep -c 'Total energy' "$out" 2>/dev/null)" -ge "$want" ]; then
    echo "[skip ] $key (has $want pts)"; return
  fi
  mem_wait "${SLOT_MB:-3400}" || echo "[mem  ] proceeding anyway for $key"
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

# ORDER: the systems that GATE the answer run first.
#
# The n=24 B curve gains its fourth point only when r0=1.00 reaches all 24
# systems. Ten systems already have 1.00 and need only 1.10 -- those jobs are
# cheap (mean 642 s) but they do NOT advance r0=1.00 at all. Running them first
# (the original order) front-loaded ~7 h of work that left the gating number
# pinned at 10/24 for hours. Reordered 2026-07-27 at Matt's request.
#
# Note on why the 1.00/1.10 pair is NOT split into separate jobs: an r0_sweep
# runs every point on ONE SCF. Trimming these TOMLs to 1.00 alone would save
# well under half the wall time and then force a full SCF redo for 1.10 later.
# Reordering systems has no such penalty, so that is the lever used here.
{
  # GATING: these 12 lack r0=1.00 entirely. (a24-17/18 are the other two, on
  # run_r0big.sh -- they need the 10 GB slot.) Cheapest first by atom count.
  for s in 20 16 01 03 04 06 07 08 09 10 11 13; do
    for f in dimer mA_cp mB_cp; do echo "a24-${s}_${f}_aqz_r0Bup_B"; done
  done
  # NON-GATING: already have 1.00, need only 1.10 for the upper bracket.
  for s in 02 05 12 14 15 19 21 22 23 24; do
    for f in dimer mA_cp mB_cp; do echo "a24-${s}_${f}_aqz_r0Bup_B"; done
  done
} | xargs -P "$NPROC" -I{} bash -c 'run "$@"' _ {}
echo "=== DONE $(date +%H:%M) ==="
