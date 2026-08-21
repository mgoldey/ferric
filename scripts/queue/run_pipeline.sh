#!/bin/bash
# Async pipeline, run in order after the B r0-fine scan lands:
#   1. T-formulation r0 sweep (own bracket -- T does NOT share B's r0 scale)
#   2. attenuated MP2 variants on A24/aTZ at PUBLISHED parameters
#   3. rs-mp2-rpa B and T on A24, then S22, at each formulation's optimum
#
# Every stage is idempotent (a job is done only when its output has the
# expected point count), waits for RAM rather than aborting, and logs failures
# instead of exiting silently.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export FERRIC_TERF_TABLE_DIR="${FERRIC_TERF_TABLE_DIR:-$HOME/qc/terf-tables-data}"
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=12
LOG() { echo "[$(date +%H:%M)] $*"; }

wait_for_ram() {   # $1 = GB needed
  for _ in $(seq 1 180); do
    [ "$(free -g | awk '/^Mem:/{print $7}')" -ge "$1" ] && return 0
    sleep 60
  done
  LOG "WARN: proceeding at low RAM after 3h wait"
}

run_job() {        # $1 = toml key, $2 = expected "Total energy" count
  local key="$1" want="${2:-1}" out="benchmarks/grid/out/$1.out"
  if [ -s "$out" ] && [ "$(grep -c 'Total energy' "$out")" -ge "$want" ]; then
    LOG "[skip] $key"; return 0
  fi
  wait_for_ram 5
  local st; st=$(date +%s)
  scripts/ferric-limited -- ./target/release/ferric \
    "benchmarks/grid/toml/${key}.toml" > "$out" 2> "${out}.err"
  local rc=$? el=$(( $(date +%s) - st ))
  local n; n=$(grep -c "Total energy" "$out" 2>/dev/null || echo 0)
  if [ "$rc" -eq 0 ] && [ "$n" -ge "$want" ]; then
    LOG "[ok   ] ${el}s ${n}pts $key"
  else
    LOG "[FAIL ] ${el}s ${n}pts $key rc=$rc"
  fi
}

# ── Wait for the B fine scan ────────────────────────────────────────────────
LOG "waiting for the B r0-fine scan to finish"
while pgrep -f run_r0fine.sh > /dev/null; do sleep 120; done
LOG "B scan drained"
python3 benchmarks/grid/mae_spline.py --form B --systems 12,15,20,21,22,23,24 \
  | tee /tmp/b_min.txt
LOG "=== STAGE 1: T-formulation r0 sweep ==="
LOG "T does not share B's r0 scale (1st vs 3rd order in v_lr): coarse first."
# Coarse T grid spanning small->large r0; the spline picks the refinement.
TPTS="0.5000, 0.7500, 1.0000, 1.5000, 2.0000, 3.0000"
for s in 20 21 12 24 15 22 23; do
  for f in dimer mA_cp mB_cp; do
    src=benchmarks/grid/toml/a24-${s}_${f}_aqz_r0fine_B.toml
    dst=benchmarks/grid/toml/a24-${s}_${f}_aqz_r0coarse_T.toml
    [ -f "$src" ] && sed -e "s|^r0_sweep = .*|r0_sweep = [${TPTS}]|" \
        -e 's|^formulation = .*|formulation = "coupled-rings"|' "$src" > "$dst"
    run_job "a24-${s}_${f}_aqz_r0coarse_T" 6
  done
done
LOG "=== STAGE 1 complete ==="
python3 benchmarks/grid/mae_spline.py --form T --systems 12,15,20,21,22,23,24 \
  --suggest | tee /tmp/t_min.txt
LOG "=== PIPELINE PAUSED: stages 2-3 need review ==="
LOG "Stage 2 (attMP2 variants) blocked: MP2C(dRPA) rejects ghost monomers;"
LOG "MP2-V not yet implemented. Stage 3 needs T's optimum from above."
