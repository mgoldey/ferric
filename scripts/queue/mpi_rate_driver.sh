#!/usr/bin/env bash
# Full rate proof for fix/mpi-allreduce-truncate: BASELINE then FIXED, in the
# SAME release build, on the same box.
#
# Why both halves in one script: the failure is a RACE, so its rate depends on
# the build (debug vs release changes the window width) and on box load. A
# post-fix streak is only meaningful against a baseline measured under the same
# conditions — quoting a pre-fix rate from a different build is how the earlier
# attempt produced numbers that could not be compared (handoff §5).
#
# The baseline is produced by REVERTING the fix in a scratch copy of guess.rs,
# not by checking out an old commit, so the two builds differ ONLY in the fix.
set -uo pipefail

cd "$(dirname "$0")/../.."
ROOT="$PWD"
GUESS="crates/ferric-scf/src/guess.rs"
OUT="${1:-/tmp/mpi_rate_proof_out}"
mkdir -p "$OUT"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/driver.log"; }

build_test_bin() {
    # Build the MPI-feature test binary and echo its path.
    OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-scf --features mpi \
        --test mpi_guess_collective_ordering --no-run --message-format=json 2>/dev/null \
      | jq -r 'select(.executable != null) | .executable' | tail -1
}

log "=== build: FIXED (as committed) ==="
BIN_FIXED="$(build_test_bin)"
[ -n "$BIN_FIXED" ] && [ -x "$BIN_FIXED" ] || { log "FATAL: fixed build produced no binary"; exit 1; }
log "fixed binary: $BIN_FIXED"
cp "$BIN_FIXED" "$OUT/bin_fixed"

log "=== build: BASELINE (fix reverted in-place) ==="
cp "$GUESS" "$OUT/guess.rs.orig"
# Revert ONLY the serialization guard: force the parallel path unconditionally.
python3 - "$GUESS" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
# The fix gates the serial branch on a live multi-rank world. Neutralise that
# condition so the per-element builds go back through par_iter, which is
# exactly the pre-fix behaviour.
s2, n = re.subn(r'unique_zs\.len\(\)\s*<=\s*1\s*\|\|\s*mpi_multi_rank',
                'unique_zs.len() <= 1', s)
if n == 0:
    # Fall back: report the region so the driver can fail loudly rather than
    # silently measuring the FIXED code as if it were the baseline.
    sys.stderr.write("REVERT-FAILED: could not find the multi-rank guard\n")
    sys.exit(3)
open(p,'w').write(s2)
print(f"reverted {n} site(s)")
PY
rc=$?
if [ $rc -ne 0 ]; then
    log "FATAL: could not revert the fix -- refusing to measure a baseline that"
    log "       might actually be the fixed code. Inspect $GUESS by hand."
    cp "$OUT/guess.rs.orig" "$GUESS"
    exit 1
fi

BIN_BASE="$(build_test_bin)"
cp "$OUT/guess.rs.orig" "$GUESS"   # restore immediately after building
[ -n "$BIN_BASE" ] && [ -x "$BIN_BASE" ] || { log "FATAL: baseline build produced no binary"; exit 1; }
cp "$BIN_BASE" "$OUT/bin_base"
log "baseline binary captured; $GUESS restored"
git diff --quiet "$GUESS" && log "restore verified clean" || { log "FATAL: $GUESS still modified"; exit 1; }

# ---- measure -----------------------------------------------------------
# Baseline first: if it does NOT fail, the experiment cannot distinguish the
# fix from nothing and the whole proof is void.
log "=== BASELINE np=4, 10 runs (expect failures) ==="
bash scripts/queue/mpi_rate_proof.sh "$OUT/bin_base" 4 10 baseline 2>&1 | tee -a "$OUT/baseline.log"

log "=== FIXED np=4, 20 runs (expect 20/20 clean) ==="
bash scripts/queue/mpi_rate_proof.sh "$OUT/bin_fixed" 4 20 fixed 2>&1 | tee -a "$OUT/fixed.log"

log "=== SUMMARY ==="
grep -h "^SUMMARY" "$OUT/baseline.log" "$OUT/fixed.log" | tee -a "$OUT/driver.log"
