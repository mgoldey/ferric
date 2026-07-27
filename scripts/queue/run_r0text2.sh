#!/bin/bash
# Extend the T r0 scan past 1.3 for the BROAD systems, so the A24-wide optimum
# stops being a boundary artifact.
#
# WHY THIS EXISTS
#
# As of 2026-07-27 the T aggregate has two answers on two different samples:
#
#   n=4  (12,21,22,24)   full 0.7-1.8   spline min r0 = 1.433 A  MAE 0.0414
#   n=16 (broad landed)  only 0.7-1.3   BOUNDARY min at 1.3      MAE 0.1181
#
# The n=16 curve is still FALLING at its edge (slope -0.25 kcal/mol per A), so
# 1.3 is where the data stops, not where the optimum is. And the two samples
# genuinely disagree: at the shared r0=1.3 the n=4 set gives MAE 0.0808 while
# n=16 gives 0.1181 -- 46% higher. The broad sample is harder, exactly as
# [[a24-subset-sampling-bias]] predicted, so the n=4 minimum must NOT be quoted
# as an A24-wide result.
#
# WHERE TO SCAN, AND WHY NOT FURTHER
#
# A quadratic through the seven n=16 points extrapolates a vertex near 1.60 A.
# Extrapolation is not evidence -- this repo has been burned by polynomial
# extrapolation of an r0 scan before -- so the window brackets BOTH candidates
# rather than betting on either:
#
#   1.4  (n=4's minimum)  1.5  1.6 (the extrapolated vertex)  1.7 (margin above)
#
# If the turn lands at 1.6 there is still a sampled point above it, so the
# result is an INTERIOR minimum rather than another boundary.
#
# SCOPE: 12 systems x 3 fragments x 4 r0 = 144 points.
#   - 9 broad systems with a COMPLETE 0.7-1.3 triple: 1,3,4,6,7,8,9,10,16
#   - plus 15,20,23, whose `r0text` outputs turned out to be EMPTY shells from
#     killed jobs (header line only, 0 "Total energy") -- they look like 1.4
#     coverage in a naive filename/header scan but are not data.
#   - EXCLUDED: 11,13,17,18 (0.7-1.3 still in flight under run_r0broad.sh).
#     Adding them here would race that driver for the same outputs.
cd /home/matt/qc/ferric
. scripts/queue/memgate.sh
export FERRIC_TERF_TABLE_DIR=/home/matt/qc/terf-tables-data
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=4
# NPROC=2 and a 3.4 GB slot: sized for a SHARED box. run_r0broad.sh and
# run_r0bmin.sh are still running at NPROC=3 each, and on 2026-07-26 a third
# driver at NPROC=3 with a 2.6 GB slot estimate drove the box into sustained
# swap thrash (si/so ~3000, wa=68%, us=1%, nothing completing for ~1 h).
# Per-job cgroup caps do not prevent that -- only total concurrency does.
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

# Cheapest first by atom count, so a partial run still yields a usable curve:
# 20,23 (4-6 atoms), 15,16 (7-8), 1,3,4,6,7,8,9,10 (larger).
for s in 20 23 15 16 01 03 04 06 07 08 09 10; do
  for f in dimer mA_cp mB_cp; do echo "a24-${s}_${f}_aqz_r0text2_T"; done
done | xargs -P "$NPROC" -I{} bash -c 'run "$@"' _ {}
echo "=== DONE $(date +%H:%M) ==="
