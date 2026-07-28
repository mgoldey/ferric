#!/bin/bash
# S22 for formulation B at r0 = 0.90, SMALLEST SYSTEM FIRST.
#
# WHY S22, AND WHY THIS ORDER
#
# A24 turned out to be the wrong benchmark to demonstrate B on. Measured
# 2026-07-27 (docs/b-vs-mp2-a24.md): B beats MP2 by 11.9% in MAE, but that is
# NOT a real effect -- paired t = 0.95, and B is WORSE on 14 of 24 systems. All
# of its benefit sits on the three repulsive pi-stacked contacts (+0.130
# kcal/mol, t = +72, 3/3), where MP2 overbinds and swapping uncoupled for
# coupled long-range dispersion removes the spurious attraction. Only 3 of 24
# A24 systems are in that regime. S22 has many more pi-stacked cases, so it is
# where the claim can actually be tested.
#
# Smallest-first is a MEASUREMENT strategy, not just politeness: S22 spans 6 to
# 30 atoms (~344 to ~2026 aQZ basis functions, vs ~700 for the A24 aQZ jobs
# that run fine here). Nobody knows where this box's ceiling is. Running in
# size order means every completed system measures feasibility for the next,
# and a run that dies at the top still leaves a usable small-to-medium set
# rather than nothing.
#
# NPROC=1 DELIBERATELY. The later systems are 2-3x larger than anything run
# here so far; a second concurrent slot would risk the exact thrash/throttle
# failures that cost most of 2026-07-27. Throughput is not the goal -- getting
# a defensible answer at all is.
#
# It also means this driver COEXISTS safely with the T sweep (run_r0tpm.sh,
# NPROC=3 at 3.4 GB slots): one S22 job at a time, and the early small systems
# (~344-700 aQZ basis functions) are no bigger than the A24 jobs already
# running. By the time this reaches the 24+ atom systems the T sweep should be
# done; if it is not, the memgate below holds this driver rather than
# oversubscribing.
#
# budget_gb = 0 in the TOMLs means "unset" -> auto-detect, which as of commit
# 9a763de finally reads THIS PROCESS'S OWN cgroup limit rather than the root's
# (which is unlimited). Before that fix a --max=NG job budgeted itself from the
# whole box and then silently throttled inside its own cap. So the --max below
# is now genuinely the number ferric plans against, and is deliberately
# generous: this driver owns the machine.
cd /home/matt/qc/ferric
. scripts/queue/memgate.sh
export FERRIC_TERF_TABLE_DIR=/home/matt/qc/terf-tables-data
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=8
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
  mem_wait "${SLOT_MB:-6000}" || echo "[mem  ] proceeding anyway for $key"
  exec 9>"${out}.lock"
  if ! flock -n 9; then echo "[lock ] $key running elsewhere"; return; fi
  local st; st=$(date +%s)
  scripts/ferric-limited --max=16G --high=14G -- ./target/release/ferric \
    "$toml" > "$out" 2> "${out}.err"
  local rc=$? el=$(( $(date +%s) - st ))
  local n; n=$(grep -c "Total energy" "$out" 2>/dev/null); [ -z "$n" ] && n=0
  local rss; rss=$(grep -oE 'peak RSS[^0-9]*([0-9.]+)' "${out}.err" 2>/dev/null | tail -1)
  if [ $rc -eq 0 ] && [ "$n" -ge "$want" ]; then echo "[ok   ] ${el}s ${n}/${want}pts $key $rss"
  else echo "[FAIL ] ${el}s ${n}/${want}pts $key rc=$rc $rss"; fi
}
export -f run mem_wait mem_avail_mb mem_psi_some10

# Atom count ascending: 6,8,10,10,10,12,12,15,15,16,17,20,24,24,24,24,25,26,28,28,30,30
for s in 02 01 03 08 16 04 09 17 19 18 10 12 05 11 13 20 06 22 14 21 07 15; do
  for f in dimer mA_cp mB_cp; do echo "s22-${s}_${f}_aqz_s22b_B"; done
done | xargs -P "$NPROC" -I{} bash -c 'run "$@"' _ {}
echo "=== DONE $(date +%H:%M) ==="
