#!/bin/bash
# T at r0 = 1.40 / 1.50 for the 18 systems that lack them -- the last step to a
# full 24-system T curve.
#
# WHY THIS IS WORTH THE COMPUTE
#
# B just finished its first complete 24-system curve and the subset estimate
# was wrong in BOTH location and difficulty:
#
#   n=10 subset   min 0.8353   MAE 0.0718
#   n=24 full     min 0.8934   MAE 0.0926   (+0.058 A, +29% harder)
#
# T's current 1.433 figure rests on FOUR systems, all from the weakly-bound
# tail. The T scan already showed the same effect from the other direction --
# adding the broad set raised MAE from 0.0808 to 0.1181 at fixed r0=1.3 -- so
# 1.433 should be expected to move, not trusted.
#
# COVERAGE ON ENTRY: 1.40 and 1.50 each have 6/24 (systems 12,17,18,21,22,24).
# This runs the other 18. 52 jobs, 104 points.
#
# CONCURRENCY: the box is otherwise idle (both B drivers have exited), so
# NPROC=3 at a 3.4 GB slot. That is the configuration that ran cleanly for
# hours today; it is NOT safe alongside another driver. Capacity here has
# changed hour to hour as job sizes grew -- re-measure before adding anything.
#
# a24-17/18 are absent BY DESIGN: they are the 13-atom systems, they already
# have both points, and they need the 10 GB slot in run_r0big.sh.
cd /home/matt/qc/ferric
. scripts/queue/memgate.sh
export FERRIC_TERF_TABLE_DIR=/home/matt/qc/terf-tables-data
# RAYON=2, NPROC=5, not RAYON=4/NPROC=3. MEASURED 2026-07-27: with RAYON=4 the
# three running jobs drew only ~190% CPU each (572% of 1200% total) while load
# average sat at 14.5 -- i.e. threads were QUEUEING without adding throughput.
# ferric's intra-job scaling is poor at this size (see the
# ferric-single-job-threading-noop note); more concurrent JOBS beats more
# threads per job. 6 slots x ~190% saturates 12 cores, and every REMAINING T
# system is 4-12 atoms (0.5-2.3 GB measured), so 5 slots fit RAM with headroom
# alongside the one S22 job.
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=2
NPROC=5

run() {
  local key="$1" out="benchmarks/grid/out/$1.out"
  local toml="benchmarks/grid/toml/${key}.toml"
  [ -f "$toml" ] || { echo "[skip ] $key (no toml)"; return; }
  local want
  want=$(grep -o '[0-9]\+\.[0-9]\+' <<<"$(grep 'r0_sweep' "$toml")" | wc -l)
  if [ -s "$out" ] && [ "$(grep -c 'Total energy' "$out" 2>/dev/null)" -ge "$want" ]; then
    echo "[skip ] $key (has $want pts)"; return
  fi
  mem_wait "${SLOT_MB:-2500}" || echo "[mem  ] proceeding anyway for $key"
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

# Cheapest first by atom count, so a partial run still yields a usable curve
# and the small systems land while the big ones are still going.
for s in 20 23 02 05 19 14 15 16 01 03 04 06 07 08 09 10 11 13; do
  for f in dimer mA_cp mB_cp; do echo "a24-${s}_${f}_aqz_r0Tpm_T"; done
done | xargs -P "$NPROC" -I{} bash -c 'run "$@"' _ {}
echo "=== DONE $(date +%H:%M) ==="
