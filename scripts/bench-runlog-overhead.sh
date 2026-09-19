#!/usr/bin/env bash
#
# Worst-case I/O overhead of the JSON run log (crates/ferric-scf/src/runlog.rs).
#
# RUN THIS ON A QUIET BOX. It is a microsecond-scale measurement; a box under
# load reports the box, not the code. Check `uptime` first -- load average
# should be near zero, and nothing else should be building.
#
#   OPENBLAS_NUM_THREADS=1 scripts/bench-runlog-overhead.sh
#
# ---------------------------------------------------------------------------
# WHY TWO MEASUREMENTS, AND WHY THE DIRECT ONE IS THE ANSWER
# ---------------------------------------------------------------------------
#
# Part A times the emit path itself (N records in a loop, sink installed vs
# not). This is the honest instrument: it isolates the thing being measured.
#
# Part B times whole `ferric` processes with the log on and off. It is reported
# as a SANITY CHECK ONLY, not as the overhead number, because an SCF process
# spends most of its life in startup, basis parsing and integral setup -- a
# per-record cost of a few microseconds is far below the run-to-run spread, so
# Part B cannot resolve it. An earlier attempt to use Part B alone produced
# CPU deltas of -0.40% to +2.89% on the SAME code and one physically impossible
# "-214 us/iteration"; that is noise being read as signal. Part B's job is only
# to confirm the overhead is too small to see end to end -- which is the
# actual claim being made.
#
# WHAT TO DO WITH THE RESULT: compare Part A's "added per record" against the
# cost of one real SCF iteration from Part B (wall/iters). The claim under test
# is that the ratio is negligible; if added-per-record ever approaches a
# percent of an iteration, the design needs an in-iteration buffer (still
# flushed per record).
set -euo pipefail

cd "$(dirname "$0")/.."
: "${OPENBLAS_NUM_THREADS:=1}"
export OPENBLAS_NUM_THREADS
OUT="${OUT:-/tmp/ferric-runlog-bench}"
mkdir -p "$OUT"

echo "load average now: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "(if that is not near zero, stop -- the numbers will measure the box)"
echo

echo "=== building (release) ==="
cargo build --release -p ferric-scf --example runlog_overhead
cargo build --release -p ferric-cli
echo

echo "=== Part A: direct cost of the emit path (THE measurement) ==="
for n in 10000 100000; do
  ./target/release/examples/runlog_overhead "$OUT/a.jsonl" "$n"
  echo
done

echo "=== Part B: end-to-end sanity check (NOT the overhead number) ==="
cat > "$OUT/bench.toml" <<'EOF'
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"
[scf]
max_iter = 100
energy_conv = 1e-8
density_conv = 1e-7
EOF

bench() { # $1 = label, $2.. = extra ferric args
  local label="$1"; shift
  local best=999999
  for _ in 1 2 3 4 5 6 7 8 9; do
    local t0 t1 ms
    t0=$(date +%s%N)
    ./target/release/ferric "$@" "$OUT/bench.toml" >/dev/null 2>&1
    t1=$(date +%s%N)
    ms=$(( (t1 - t0) / 1000000 ))
    [ "$ms" -lt "$best" ] && best=$ms
  done
  # MINIMUM over repetitions, not mean: the minimum is the run least disturbed
  # by other load, which is the closest estimate of the true cost. A mean on a
  # shared box is a measure of the interference.
  echo "  $label: ${best} ms (min of 9)"
}

bench "log OFF" --no-json
bench "log ON "
# `grep -c` prints 0 and EXITS NONZERO when nothing matches, so the old
# `|| echo '?'` fired on an empty log as well as a missing one -- and the
# guidance line below then told the reader to divide by '?'. Validate instead
# of papering over it: a benchmark that cannot count its own iterations has
# not measured anything.
iters=$(grep -c '"record":"scf_iter"' "$OUT/bench.ferric.jsonl" 2>/dev/null) || iters=0
if ! [[ "$iters" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: expected a positive scf_iter count in $OUT/bench.ferric.jsonl, got '$iters'." >&2
  echo "       The run produced no logged iterations, so Part B cannot be" >&2
  echo "       divided per iteration and the comparison below is meaningless." >&2
  exit 1
fi
echo "  (water/STO-3G RHF, $iters logged SCF iterations)"
echo
echo "Compare Part A's added-per-record against (Part B ms / $iters) per iteration."
