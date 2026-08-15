#!/usr/bin/env bash
# G1 correctness CI gate (docs/perf-tasks/G1-correctness-ci-gate.md, triage #27).
#
# This repo has no hosted CI (no .github/workflows, no git remote at all as of
# 2026-07-16) -- this script IS the gate. It is meant to be run:
#   - by hand before a push: scripts/ci-gate.sh
#   - automatically via the pre-push hook installed by scripts/install-hooks.sh
#
# What it checks:
#   1. `cargo test --workspace` (default set, i.e. the ~706/800 non-`#[ignore]`
#      tests -- the 94 `#[ignore]`d tests are deliberately-slow numerical
#      dispersion/CPKS/BSE benchmarks excluded from routine runs by design;
#      this gate does not force them).
#   2. `cargo clippy --workspace --all-targets` warnings, filtered against the
#      SAME allow-list encoded in root Cargo.toml's [workspace.lints] block
#      (every member crate opts in via `[lints] workspace = true`). Because
#      the allow-list already lives in Cargo.toml, `cargo clippy` itself won't
#      emit allow-listed lints in the first place -- this script's clippy step
#      is mostly a safety net (e.g. for anyone who runs clippy with
#      --cap-lints or otherwise bypasses the workspace lints table) and a
#      place to fail loudly with a clear message if anything DOES fire.
#
# Exit code: 0 = clean, nonzero = a real failure (test or clippy) OR a timeout
# under heavy machine load (see LOAD AWARENESS below) -- read the final
# message to tell which.
#
# Env overrides:
#   CI_GATE_TIMEOUT_SECS   per-step timeout (default 1800 = 30 min; a cold
#                          build can take ~29 min per docs/performance.md, so
#                          this assumes a warm/incremental build -- see
#                          scripts/README.md for what to do on a cold-cache OR
#                          heavily-contended box).
#   CI_GATE_JOBS           cargo -j value (default: nproc, capped at 8 to
#                          leave headroom on a shared box).
#   CI_GATE_SKIP_CLIPPY=1  skip the clippy step (test-only gate).
#   CI_GATE_SKIP_TESTS=1   skip the test step (clippy-only gate).
#   CI_GATE_SKIP_COMPLEXITY=1  skip the complexity-regression step.
#
# Step 3, complexity regression (scripts/complexity_gate.py): tracks
# cyclomatic complexity (CC) and maintainability index (MI) per function via
# `rust-code-analysis-cli` (install: `cargo install rust-code-analysis-cli`)
# against a checked-in baseline (scripts/complexity_baseline.json). This is
# NOT an absolute threshold -- several SCF/RPA numerical kernels are already
# legitimately complex (CC 100-134) and explicitly out of scope for
# splitting, same reasoning as the too_many_arguments allow-list above. The
# gate only fails on a REGRESSION (a function getting worse vs. baseline, or
# a brand-new function appearing above a generous ceiling) -- see the
# script's own doc comment for the full rationale. Soft-skips (does not fail
# the gate) if rust-code-analysis-cli isn't installed, since it's a
# machine-local dev tool, not a workspace dependency.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TIMEOUT_SECS="${CI_GATE_TIMEOUT_SECS:-1800}"
NPROC="$(nproc 2>/dev/null || echo 4)"
DEFAULT_JOBS=$(( NPROC < 8 ? NPROC : 8 ))
JOBS="${CI_GATE_JOBS:-$DEFAULT_JOBS}"
LOCK_FILE="/tmp/ferric-cargo.lock"
LOCK_WAIT_SECS=14400

# ---- load awareness ---------------------------------------------------
# Per the brief: a correctness failure (test/clippy) is load-INDEPENDENT --
# a test either passes or fails regardless of box speed -- so the real risk
# here is a hook that TIMES OUT under heavy contention and gets misread as
# "the gate is broken" instead of "the box was busy." We record load at
# start and annotate every exit path with it so a contended run is
# distinguishable from a real correctness issue.
LOAD1="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo "unknown")"
HALF_NPROC=$(awk -v n="$NPROC" 'BEGIN{printf "%.2f", n/2}')
CONTENDED=0
if [[ "$LOAD1" != "unknown" ]] && awk -v l="$LOAD1" -v h="$HALF_NPROC" 'BEGIN{exit !(l>h)}'; then
    CONTENDED=1
fi

echo "== ferric correctness gate =="
echo "   repo:        $REPO_ROOT"
echo "   nproc:       $NPROC (using -j $JOBS)"
echo "   load avg(1m): $LOAD1 (nproc/2 = $HALF_NPROC)"
if [[ "$CONTENDED" == "1" ]]; then
    echo "   NOTE: box is under heavy contention (load > nproc/2)."
    echo "         A timeout below is more likely to mean \"box was busy\""
    echo "         than \"the gate found nothing\" -- see scripts/README.md."
fi
echo "   per-step timeout: ${TIMEOUT_SECS}s (CI_GATE_TIMEOUT_SECS to change)"
echo

FAILED=0
STEP_TIMED_OUT=0

run_step() {
    local name="$1"
    shift
    echo "-- $name --"
    if timeout --signal=TERM --kill-after=30 "${TIMEOUT_SECS}" \
        flock -w "$LOCK_WAIT_SECS" "$LOCK_FILE" -c "$*"; then
        echo "-- $name: PASS --"
        return 0
    else
        local rc=$?
        if [[ $rc -eq 124 || $rc -eq 137 ]]; then
            echo "-- $name: TIMEOUT after ${TIMEOUT_SECS}s (exit $rc) --"
            STEP_TIMED_OUT=1
        else
            echo "-- $name: FAIL (exit $rc) --"
        fi
        FAILED=1
        return 1
    fi
}

# ---- 1. cargo test --workspace (default set, no --ignored) ------------
if [[ "${CI_GATE_SKIP_TESTS:-0}" != "1" ]]; then
    run_step "cargo test --workspace" \
        "OPENBLAS_NUM_THREADS=1 cargo test --workspace -j $JOBS"
else
    echo "-- cargo test --workspace: SKIPPED (CI_GATE_SKIP_TESTS=1) --"
fi
echo

# ---- 2. cargo clippy --workspace --all-targets, allow-list checked -----
# The allow-list lives in root Cargo.toml's [workspace.lints] block (each
# member crate opts in via `[lints] workspace = true`), so a plain
# `cargo clippy --workspace --all-targets` already will not emit these lints.
# This step re-runs clippy and greps its own JSON output for any
# `clippy::`-coded warning/error as a redundant, explicit check -- if the
# lints table and this list ever drift (e.g. someone edits Cargo.toml without
# reading this file), THIS is what catches it.
ALLOWED_LINTS=(
    "clippy::excessive_precision"
    "clippy::neg_multiply"
    "clippy::needless_range_loop"
    "clippy::doc_lazy_continuation"
    "clippy::too_many_arguments"
    "clippy::derivable_impls"
    "clippy::field_reassign_with_default"
    "clippy::redundant_pattern_matching"
    "clippy::unnecessary_map_or"
    "clippy::unnecessary_min_or_max"
    "clippy::type_complexity"
    "clippy::needless_borrows_for_generic_args"
    "clippy::if_same_then_else"
    "clippy::neg_cmp_op_on_partial_ord"
    "unused_parens"
)

if [[ "${CI_GATE_SKIP_CLIPPY:-0}" != "1" ]]; then
    echo "-- cargo clippy --workspace --all-targets --"
    CLIPPY_JSON="$(mktemp /tmp/ferric-ci-gate-clippy.XXXXXX.json)"
    if timeout --signal=TERM --kill-after=30 "${TIMEOUT_SECS}" \
        flock -w "$LOCK_WAIT_SECS" "$LOCK_FILE" -c \
        "OPENBLAS_NUM_THREADS=1 cargo clippy --workspace --all-targets -j $JOBS --message-format=json" \
        > "$CLIPPY_JSON" 2>&1; then
        CLIPPY_RC=0
    else
        CLIPPY_RC=$?
    fi

    if [[ $CLIPPY_RC -eq 124 || $CLIPPY_RC -eq 137 ]]; then
        echo "-- cargo clippy: TIMEOUT after ${TIMEOUT_SECS}s (exit $CLIPPY_RC) --"
        STEP_TIMED_OUT=1
        FAILED=1
    else
        # Build a grep -F -f pattern file from the allow-list, matched against
        # each JSON line's "code":"<lint>" field.
        PATTERN_FILE="$(mktemp /tmp/ferric-ci-gate-allowlist.XXXXXX.txt)"
        for lint in "${ALLOWED_LINTS[@]}"; do
            echo "\"code\":\"${lint}\"" >> "$PATTERN_FILE"
        done

        # Any compiler-message line with level warning/error whose code is NOT
        # in the allow-list is a gate failure. rustc/clippy JSON puts the code
        # at top level of the "message" object as {"code":{"code":"..."}}.
        VIOLATIONS="$(grep '"reason":"compiler-message"' "$CLIPPY_JSON" \
            | grep -E '"level":"(warning|error)"' \
            | grep -vF -f "$PATTERN_FILE" \
            | grep -E '"code":\{"code":"[a-zA-Z_:]+"' \
            || true)"

        if [[ -n "$VIOLATIONS" ]]; then
            echo "-- cargo clippy: FAIL -- non-allow-listed warnings found --"
            echo "$VIOLATIONS" | python3 -c "
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    msg = obj.get('message', {})
    code = (msg.get('code') or {}).get('code', '(unknown)')
    spans = msg.get('spans', [])
    loc = ''
    for s in spans:
        if s.get('is_primary'):
            loc = f\"{s.get('file_name')}:{s.get('line_start')}\"
            break
    print(f'  {code} @ {loc}: {msg.get(\"message\", \"\")}')
" 2>/dev/null || echo "$VIOLATIONS"
            echo
            echo "   Not on the allow-list in scripts/ci-gate.sh / root Cargo.toml"
            echo "   [workspace.lints]. Either fix the code, or -- if this is a"
            echo "   numerical/formatting idiom this repo already tolerates"
            echo "   elsewhere -- add it to BOTH the allow-list here and the"
            echo "   [workspace.lints] block, with a one-line rationale."
            FAILED=1
        elif [[ $CLIPPY_RC -ne 0 ]]; then
            echo "-- cargo clippy: FAIL (exit $CLIPPY_RC, no parseable violations -- check raw output) --"
            tail -40 "$CLIPPY_JSON"
            FAILED=1
        else
            echo "-- cargo clippy: PASS (no warnings outside the allow-list) --"
        fi
        rm -f "$PATTERN_FILE"
    fi
    rm -f "$CLIPPY_JSON"
else
    echo "-- cargo clippy: SKIPPED (CI_GATE_SKIP_CLIPPY=1) --"
fi
echo

# ---- 3. complexity regression (CC/MI vs. checked-in baseline) ----------
# Soft-skip (does not set FAILED) if the tool isn't installed -- this is a
# machine-local dev tool (`cargo install rust-code-analysis-cli`), not a
# workspace dependency every contributor is required to have. A missing
# baseline file also soft-skips (first run before anyone has generated one).
if [[ "${CI_GATE_SKIP_COMPLEXITY:-0}" != "1" ]]; then
    echo "-- complexity regression (scripts/complexity_gate.py) --"
    if command -v rust-code-analysis-cli >/dev/null 2>&1; then
        if python3 "$REPO_ROOT/scripts/complexity_gate.py"; then
            echo "-- complexity regression: PASS --"
        else
            RC=$?
            if [[ $RC -eq 2 ]]; then
                echo "-- complexity regression: SKIPPED (see message above) --"
            else
                echo "-- complexity regression: FAIL --"
                FAILED=1
            fi
        fi
    else
        echo "-- complexity regression: SKIPPED (rust-code-analysis-cli not installed;"
        echo "   cargo install rust-code-analysis-cli to enable this check locally) --"
    fi
else
    echo "-- complexity regression: SKIPPED (CI_GATE_SKIP_COMPLEXITY=1) --"
fi
echo

# ---- summary ------------------------------------------------------------
echo "== gate summary =="
echo "   load avg(1m) at start: $LOAD1 (nproc/2 = $HALF_NPROC, contended=$CONTENDED)"
if [[ "$FAILED" == "1" ]]; then
    if [[ "$STEP_TIMED_OUT" == "1" && "$CONTENDED" == "1" ]]; then
        echo "   RESULT: TIMEOUT under heavy load -- box was busy (load $LOAD1 > nproc/2"
        echo "           = $HALF_NPROC). This is NOT necessarily a correctness problem."
        echo "           Retry when the box is quieter, or raise CI_GATE_TIMEOUT_SECS."
        echo "           If you must push now: git push --no-verify (this SKIPS the"
        echo "           gate entirely -- only do this with an explicit note in your"
        echo "           push/PR that the gate was bypassed, not silently passed)."
    elif [[ "$STEP_TIMED_OUT" == "1" ]]; then
        echo "   RESULT: TIMEOUT (load looked normal -- investigate: hung test? deadlock?)."
    else
        echo "   RESULT: FAIL -- see above for the failing test(s) or clippy violation(s)."
    fi
    exit 1
else
    echo "   RESULT: PASS"
    exit 0
fi
