#!/usr/bin/env bash
# Rate-based proof for fix/mpi-allreduce-truncate.
#
# Per the handoff §5: WATER (not benzene), --release (not debug), and a FRESH
# BASELINE measured in the same build, because the build sets the race-window
# width and therefore the rate.
#
# Vehicle is the committed regression test `mpi_guess_collective_ordering`,
# which is water/cc-pVDZ through the MINAO guess with DF aux bases live. It
# checks BOTH that the guess completes AND that every rank's density is
# bit-identical, so a silently-mismatched reduce fails too.
#
# NOTE: ferric never calls MPI_Finalize (handoff §6), so mpirun's exit status
# is unusable. We assert on OUTPUT CONTENT, not $?.
#
# Usage: mpi_rate_proof.sh <binary> <np> <nruns> <label>
set -u

BIN="${1:?test binary}"
NP="${2:?np}"
RUNS="${3:?nruns}"
LABEL="${4:-run}"

pass=0; fail=0; i=0
while [ "$i" -lt "$RUNS" ]; do
    i=$((i+1))
    out=$(OPENBLAS_NUM_THREADS=1 timeout 300 mpirun -np "$NP" --oversubscribe \
              "$BIN" --nocapture --test-threads=1 2>&1)
    # Success is the harness's own summary line, not the exit code.
    if grep -qE "^test result: ok\. 1 passed" <<<"$out"; then
        pass=$((pass+1)); verdict="ok"
    else
        fail=$((fail+1)); verdict="FAIL"
        # Keep the first line of evidence for each distinct failure.
        printf '  [%s run %d] %s\n' "$LABEL" "$i" \
            "$(grep -oE 'MPI_ERR_[A-Z_]+|truncated|signal [0-9]+|panicked at.*' <<<"$out" | head -1)"
    fi
    printf '[%s] np=%s run %2d/%s: %s  (pass %d / fail %d)\n' \
        "$LABEL" "$NP" "$i" "$RUNS" "$verdict" "$pass" "$fail"
done

echo "SUMMARY $LABEL np=$NP: $pass passed, $fail failed of $RUNS"
