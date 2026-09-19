#!/usr/bin/env bash
#
# Kill a REAL ferric SCF mid-flight and show what the JSON run log preserved.
#
# This is the end-to-end demonstration of the one property the run log exists
# for. Not a synthetic probe writing records in a loop -- an actual
# benzene/cc-pVDZ PBE calculation doing real Fock builds, SIGKILLed partway
# through its SCF. SIGKILL specifically, not SIGTERM or SIGINT: a signal the
# process cannot catch, arriving the same way an OOM kill does.
#
# The `#[test] a_killed_run_leaves_a_usable_partial_log` in
# crates/ferric-scf/tests/runlog_writer.rs asserts the same property in CI on a
# synthetic probe (fast, no chemistry). This script is the human-facing version:
# it shows the surviving SCF trajectory next to the near-empty stdout capture
# that used to be the only record of a run.
#
#   scripts/demo-runlog-survives-kill.sh
#
# Cheap: one small SCF, killed after ~8 iterations. Safe to run on a busy box
# (it is a durability check, not a measurement).
set -u

cd "$(dirname "$0")/.."
ROOT=$PWD
OUT="${OUT:-/tmp/ferric-killdemo}"
mkdir -p "$OUT"
FERRIC=./target/release/ferric

if [ ! -x "$FERRIC" ]; then
  echo "building the release CLI first..."
  cargo build --release -p ferric-cli || exit 1
fi

cat > "$OUT/kill.toml" <<'EOF'
[molecule]
xyz = "testdata/molecules/benzene.xyz"

[basis]
name = "cc-pvdz"

[method]
kind = "ksdft"

[dft]
functional = "PBE"

# density_conv is deliberately unreachable so the run keeps iterating and can
# be killed mid-SCF rather than finishing first.
[scf]
max_iter = 200
energy_conv = 1e-12
density_conv = 1e-12
EOF

LOG="$OUT/kill.ferric.jsonl"
rm -f "$LOG"

OPENBLAS_NUM_THREADS=1 "$FERRIC" "$OUT/kill.toml" > "$OUT/kill.stdout" 2>&1 &
PID=$!
echo "started ferric, pid=$PID"

# Wait until the SCF is well underway, then kill without warning.
N=0
for _ in $(seq 1 600); do
  N=$(grep -c '"record":"scf_iter"' "$LOG" 2>/dev/null)
  N=${N:-0}
  if [ "$N" -ge 8 ] 2>/dev/null; then break; fi
  sleep 0.5
done
echo "SCF iterations logged before the kill: $N"
kill -9 "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
echo "the process exited with $? (137 = SIGKILL)"
echo

echo "=== what survived on disk ==="
printf 'stdout capture : %s bytes  <- what a "log" used to mean\n' "$(wc -c < "$OUT/kill.stdout")"
printf 'JSON run log   : %s lines, %s bytes\n' "$(wc -l < "$LOG")" "$(wc -c < "$LOG")"
echo

python3 - "$LOG" <<'PYEOF'
import json, sys
ok = bad = 0
for line in open(sys.argv[1]):
    try:
        json.loads(line); ok += 1
    except Exception:
        bad += 1
print(f"parseable records: {ok}   unparseable: {bad}  "
      f"(at most 1 is expected -- a kill can land mid-write)")
print()
print("the SCF trajectory that survived the kill:")
for line in open(sys.argv[1]):
    try:
        r = json.loads(line)
    except Exception:
        print("  <truncated final line>")
        continue
    if r["record"] == "run_start":
        c, m = r["config"], r["molecule"]
        print(f'  run_start  {m["formula"]}/{c["basis"]} {c["functional"]}, '
              f'ferric {r["ferric_version"]} sha {r["git_sha"]}')
    elif r["record"] == "scf_iter":
        print(f'  iter {r["iter"]:3d}   E = {r["energy"]:.10f}   '
              f'dp_rms = {r["dp_rms"]}')
PYEOF
