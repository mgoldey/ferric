#!/bin/bash
# T at r0 = 1.60 / 1.70 -- T's full-A24 optimum is ABOVE 1.50, not at 1.433.
#
# THE RESULT THAT MOTIVATES THIS (overnight 2026-07-27/28)
#
# All 24 systems now have r0 = 1.40 and 1.50:
#
#   n=24   1.40 -> 0.1050   1.50 -> 0.0903      STILL FALLING
#   n=17   0.70 -> 0.3717   0.80 -> 0.3454   1.40 -> 0.0979   1.50 -> 0.0861
#                                            BOUNDARY minimum at 1.50
#
# Both views agree and both are boundary minima, so the optimum lies beyond
# 1.50. The old n=4 figure of 1.433 A was a subset artifact -- exactly the
# pattern B showed, where the 10-system estimate of 0.835 moved to 0.893 on the
# full set AND the benchmark got 29% harder. Do not quote 1.433.
#
# 1.60 and 1.70 bracket from above: the four original systems (12,21,22,24)
# already have both, and their curves turn between 1.4 and 1.7, so this should
# capture the turn for the full set. If 1.70 is still falling, extend again --
# but the per-system optima measured so far (1.4-1.7) make that unlikely.
#
# 58 jobs, 116 points. 14 jobs already complete (the four original systems), so
# the planner trimmed them.
#
# RAYON=12 / NPROC=1: one job, all 12 cores. Run this ALONE -- two drivers at
# RAYON=12 put 24 threads on 12 cores, the oversubscription behind both crashes
# on 2026-07-27.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. scripts/queue/memgate.sh
export FERRIC_TERF_TABLE_DIR="${FERRIC_TERF_TABLE_DIR:-$HOME/qc/terf-tables-data}"
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=12
NPROC=1

run() {
  local key="$1" out="benchmarks/grid/out/$1.out"
  local toml="benchmarks/grid/toml/${key}.toml"
  [ -f "$toml" ] || { echo "[skip ] $key (no toml)"; return; }
  local want
  want=$(grep -o '[0-9]\+\.[0-9]\+' <<<"$(grep 'r0_sweep' "$toml")" | wc -l)
  if [ -s "$out" ] && [ "$(grep -c 'Total energy' "$out" 2>/dev/null)" -ge "$want" ]; then
    echo "[skip ] $key (has $want pts)"; return
  fi
  mem_wait "${SLOT_MB:-4000}" || echo "[mem  ] proceeding anyway for $key"
  exec 9>"${out}.lock"
  if ! flock -n 9; then echo "[lock ] $key running elsewhere"; return; fi
  local st; st=$(date +%s)
  scripts/ferric-limited --max=12G --high=10G -- ./target/release/ferric \
    "$toml" > "$out" 2> "${out}.err"
  local rc=$? el=$(( $(date +%s) - st ))
  local n; n=$(grep -c "Total energy" "$out" 2>/dev/null); [ -z "$n" ] && n=0
  if [ $rc -eq 0 ] && [ "$n" -ge "$want" ]; then echo "[ok   ] ${el}s ${n}/${want}pts $key"
  else echo "[FAIL ] ${el}s ${n}/${want}pts $key rc=$rc"; fi
}
export -f run mem_wait mem_avail_mb mem_psi_some10

# Cheapest first by atom count so a partial run still yields a usable curve.
for s in 04 02 03 01 06 05 08 09 07 10 16 11 13 19 15 14 23 17 18 12 20 21 22 24; do
  for f in dimer mA_cp mB_cp; do echo "a24-${s}_${f}_aqz_r0Tup_T"; done
done | xargs -P "$NPROC" -I{} bash -c 'run "$@"' _ {}
echo "=== DONE $(date +%H:%M) ==="
