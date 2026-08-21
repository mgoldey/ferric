#!/bin/bash
# Complete BOTH minima +/- 0.1 A across ALL of A24.
#
# TARGET
#   B  min 0.8332 A  ->  0.70 / 0.80 / 0.90   (straddles it, +/-~0.13)
#   T  min 1.433  A  ->  1.40 / 1.50          (straddles it, +/-~0.07)
#
# WHY THESE r0 AND NOT 0.73/0.83/0.93 AND 1.33/1.43/1.53
#
# Asking for a literal +/-0.1 grid around each minimum means brand-new r0
# values that reuse NOTHING: 216 points for B and 216 for T, 432 total.
# Snapping to r0 values ALREADY on disk costs 68 points for the same
# bracketing, because B is already at 21-22 of 24 systems on 0.70/0.80/0.90
# and T's 1.40/1.50 already exist for the original 4 systems. The physics
# question -- is the minimum bracketed on both sides across the full
# benchmark -- is answered identically either way.
#
# SCOPE (68 points)
#   B: systems 17, 18 (missing everywhere) + 23 (missing 0.90 only)  = 20 pts
#   T: systems 2, 5, 11, 13, 14, 17, 18, 19                          = 48 pts
#
# T EXCLUDES 1,3,4,6,7,8,9,10,15,16,20,23 ON PURPOSE: run_r0text2.sh is
# ALREADY running 1.4/1.5/1.6/1.7 for exactly those. Queueing them here would
# race that driver for the same output files -- flock would serialize them, but
# the second would still redo finished work.
#
# CONCURRENCY: NPROC=2, 3.4 GB slot. On 2026-07-27 a FOURTH driver on this box
# drove it into thrash within 90 s (so=245-477 sustained, us=1%, PSI 94) even
# though the pre-launch snapshot showed 10.7 GB free and PSI 0.00 -- because
# the RESIDENT jobs were still growing. A snapshot of current RSS does not
# bound future RSS. Run this only when at most two other drivers are active.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. scripts/queue/memgate.sh
export FERRIC_TERF_TABLE_DIR="${FERRIC_TERF_TABLE_DIR:-$HOME/qc/terf-tables-data}"
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

# B first: only 20 points, and it closes B to a full 24-system answer -- the
# strongest result either formulation currently has.
{
  for s in 23 17 18; do
    for f in dimer mA_cp mB_cp; do echo "a24-${s}_${f}_aqz_r0Bpm_B"; done
  done
  for s in 02 05 19 14 11 13 17 18; do
    for f in dimer mA_cp mB_cp; do echo "a24-${s}_${f}_aqz_r0Tpm_T"; done
  done
} | xargs -P "$NPROC" -I{} bash -c 'run "$@"' _ {}
echo "=== DONE $(date +%H:%M) ==="
