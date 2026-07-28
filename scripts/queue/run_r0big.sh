#!/bin/bash
# The 13-atom A24 systems (a24-17, a24-18) at BOTH minima, with a slot big
# enough to actually hold them.
#
# WHY A SEPARATE DRIVER
#
# a24-17/18 are the largest systems in A24 (13 atoms vs 10 for a24-11), and at
# aQZ they need MORE THAN 5 GB. Under the shared drivers' standard
# `ferric-limited --max=5G --high=4G` slot they do not fail -- they THROTTLE.
# Diagnosed 2026-07-27: three of them sat at 4.43-4.64 GB against memory.max
# 5368709120 with every thread parked in `mem_cgroup_handle_over_high`, burning
# 46 minutes at **0 completed points** while the machine as a whole had 5+ GB
# free.
#
# That failure is worth recognising by sight because it MIMICS system-wide
# thrash (jobs stalled, low us%) but is NOT the same thing and the fix is the
# opposite. Tell them apart:
#
#   system thrash        vmstat si/so both large, swap filling, MANY jobs slow
#   cgroup over-high     si/so ~0, plenty of free RAM, ONLY the big jobs stall,
#                        `ps -o wchan` shows mem_cgroup_handle_over_high
#
# Reducing concurrency fixes the first and does NOTHING for the second -- I
# stopped two drivers chasing this before checking wchan.
#
# So: 10 GB cap, and NPROC=1. One 13-atom job at a time, with room to breathe.
# Run this ALONE; it is sized to use most of the box.
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
  mem_wait "${SLOT_MB:-9000}" || echo "[mem  ] proceeding anyway for $key"
  exec 9>"${out}.lock"
  if ! flock -n 9; then echo "[lock ] $key running elsewhere"; return; fi
  local st; st=$(date +%s)
  scripts/ferric-limited --max=10G --high=9G -- ./target/release/ferric \
    "$toml" > "$out" 2> "${out}.err"
  local rc=$? el=$(( $(date +%s) - st ))
  local n; n=$(grep -c "Total energy" "$out" 2>/dev/null); [ -z "$n" ] && n=0
  if [ $rc -eq 0 ] && [ "$n" -ge "$want" ]; then echo "[ok   ] ${el}s ${n}/${want}pts $key"
  else echo "[FAIL ] ${el}s ${n}/${want}pts $key rc=$rc"; fi
}
export -f run mem_wait mem_avail_mb mem_psi_some10

# B first (r0Bmin covers 0.70/0.80/0.90 -- the whole +/-0.1 bracket), then the
# T points. a24-17/18 are the ONLY systems still missing from B's 24-system
# answer, so these six jobs close it.
{
  for s in 17 18; do
    for f in dimer mA_cp mB_cp; do echo "a24-${s}_${f}_aqz_r0Bmin_B"; done
  done
  for s in 17 18; do
    for f in dimer mA_cp mB_cp; do echo "a24-${s}_${f}_aqz_r0Tpm_T"; done
  done
  # The full-A24 B minimum turned out to be above r0=0.90 -- see
  # run_r0bup.sh -- so these two large systems need the upward extension too.
  # NOTE: no apostrophes in comments inside this brace group; an unpaired
  # quote here opens a string and bash mis-parses the whole block at runtime,
  # which is exactly how this driver died once already.
  for s in 17 18; do
    for f in dimer mA_cp mB_cp; do echo "a24-${s}_${f}_aqz_r0Bup_B"; done
  done
} | xargs -P "$NPROC" -I{} bash -c 'run "$@"' _ {}
echo "=== DONE $(date +%H:%M) ==="
