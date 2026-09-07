#!/usr/bin/env bash
# Wall-time effect of DFT angular grid pruning on a larger case.
#
# Point-count reduction only matters if it shows up in wall time, so this
# measures the thing that actually gets billed: a full KS-DFT SCF, flat grid vs
# pruned grid, same molecule/basis/functional/thresholds.
#
# Reports total wall time AND per-iteration time (total / iterations), because
# the two can move in opposite directions: pruning perturbs the Fock matrix, so
# it could in principle change the ITERATION COUNT as well as the cost per
# iteration. A per-iteration win that is cancelled by extra iterations is not a
# win, and only the split shows that.
#
# Usage:  scripts/queue/prune_timing.sh [alkane_N] [basis] [functional]
# Default: alkane_8 / cc-pvdz / PBE
#
# Run under scripts/ferric-limited (see the repo memory rules for >= alkane_8):
#   scripts/ferric-limited --max=4G --high=3600M -- scripts/queue/prune_timing.sh
set -uo pipefail

MOL="${1:-alkane_8}"
BASIS="${2:-cc-pvdz}"
FUNC="${3:-PBE}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT/target/release/ferric"
XYZ="testdata/molecules/${MOL}.xyz"

if [[ ! -x "$BIN" ]]; then
  echo "error: $BIN not built. Run: cargo build --release -p ferric-cli" >&2
  exit 1
fi
if [[ ! -f "$ROOT/$XYZ" ]]; then
  echo "error: $ROOT/$XYZ not found" >&2
  exit 1
fi

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# OPENBLAS_NUM_THREADS=1: BLAS>1 under rayon is unsafe in this repo, and a
# varying BLAS thread count would make the two arms incomparable anyway.
export OPENBLAS_NUM_THREADS=1
export FERRIC_MEM_BUDGET_GB="${FERRIC_MEM_BUDGET_GB:-2}"

run_arm () {
  local label="$1" prune="$2" toml="$OUT/$1.toml" log="$OUT/$1.log"
  cat > "$toml" <<EOF
[molecule]
xyz = "$XYZ"

[basis]
name = "$BASIS"

[method]
kind = "ksdft"
task = "energy"

[dft]
functional = "$FUNC"
grid_prune = "$prune"

[scf]
# Deliberately loose. This probe measures COST PER ITERATION, not accuracy —
# accuracy is grid_prune_live_scf.rs. Tight thresholds would only add
# iterations of the same per-iteration work to both arms, lengthening the
# measurement without sharpening it. Both arms use the identical setting, so
# the comparison stays fair.
max_iter = 12
energy_conv = 1e-6
density_conv = 1e-5
EOF

  # `time` on the process, not an internal timer: this is about billed wall
  # time, including grid construction and AO evaluation, not just the SCF loop.
  local t0 t1
  t0=$(date +%s.%N)
  ( cd "$ROOT" && "$BIN" "$toml" ) > "$log" 2>&1
  local rc=$?
  t1=$(date +%s.%N)

  local wall energy iters
  wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  # ferric prints "  energy     = -76.3335101342 Hartree" and
  # "  iterations = 55" -- parse those exact lines.
  energy=$(awk '/^[[:space:]]*energy[[:space:]]*=/ {print $3}' "$log" | tail -1)
  iters=$(awk '/^[[:space:]]*iterations[[:space:]]*=/ {print $3}' "$log" | tail -1)
  [[ -z "$iters" ]] && iters=0

  echo "$label|$rc|$wall|${energy:-NA}|$iters|$log"
}

echo "=== DFT grid pruning wall-time probe ==="
echo "system : $MOL / $BASIS / $FUNC"
echo "budget : FERRIC_MEM_BUDGET_GB=$FERRIC_MEM_BUDGET_GB, OPENBLAS_NUM_THREADS=1"
echo

# Memory-pressure gate. A run taken while the box is swapping measures the
# swap, not the grid: an alkane_6 probe taken at PSI full avg300=36 reported
# the PRUNED arm 20% SLOWER than flat with identical iteration counts, which is
# not a physically possible pruning result. Refuse to measure under stall
# rather than emit a number that looks like data. (Repo memory:
# "Thrashing looks like slowness, not OOM" / "Memory gates must use PSI".)
psi_full_avg10 () {
  awk '/^full/{for(i=2;i<=NF;i++){split($i,a,"="); if(a[1]=="avg10") print a[2]}}' \
    /proc/pressure/memory 2>/dev/null || echo 0
}
p=$(psi_full_avg10)
if awk -v p="${p:-0}" 'BEGIN{exit !(p+0 > 5.0)}'; then
  echo "error: memory PSI full avg10=${p}% -- the box is stalling on memory." >&2
  echo "       Timing taken now would measure swap, not the grid. Wait and retry." >&2
  exit 1
fi

# Interleave the arms (flat, pruned, flat, pruned) and keep the SECOND pair.
# The first pair warms the page cache and the CPU frequency governor; a
# cold-vs-warm comparison would otherwise be charged entirely to whichever arm
# ran first. Reporting the second pair means both arms are equally warm.
run_arm flat none    > /dev/null
run_arm pruned nwchem > /dev/null
flat=$(run_arm flat none)
pruned=$(run_arm pruned nwchem)

IFS='|' read -r _ frc fwall fe fit flog <<< "$flat"
IFS='|' read -r _ prc pwall pe pit plog <<< "$pruned"

printf '%-8s %-4s %10s %18s %6s %12s\n' arm rc "wall(s)" "E_total(Ha)" iters "s/iter"
for row in "flat|$frc|$fwall|$fe|$fit" "pruned|$prc|$pwall|$pe|$pit"; do
  IFS='|' read -r a r w e i <<< "$row"
  spi=$(awk -v w="$w" -v i="$i" 'BEGIN{ if (i+0>0) printf "%.3f", w/i; else print "NA" }')
  printf '%-8s %-4s %10s %18s %6s %12s\n' "$a" "$r" "$w" "$e" "$i" "$spi"
done

echo
if [[ "$frc" != "0" || "$prc" != "0" ]]; then
  echo "WARNING: an arm exited non-zero (flat rc=$frc, pruned rc=$prc). Logs:"
  echo "  flat  : $flog"
  echo "  pruned: $plog"
  # Keep the logs for inspection rather than letting the trap delete them.
  cp "$flog" "$plog" /tmp/ 2>/dev/null || true
  echo "  (copied to /tmp/)"
  exit 1
fi

awk -v fw="$fwall" -v pw="$pwall" -v fe="$fe" -v pe="$pe" 'BEGIN{
  if (pw+0>0) printf "speedup (total wall): %.2fx  (%.1f%% less time)\n", fw/pw, 100*(1-pw/fw);
  d = fe-pe; if (d<0) d=-d;
  printf "energy difference flat vs pruned: %.3e Ha\n", d;
  print "";
  print "NOTE: the energy difference above is a CONSISTENCY check between two";
  print "ferric runs, NOT an accuracy measurement. Accuracy vs PySCF is";
  print "crates/ferric-dft/tests/grid_prune_live_scf.rs.";
}'
