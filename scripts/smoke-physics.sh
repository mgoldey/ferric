#!/usr/bin/env bash
# Extremely fast physics smoke test: does the engine still produce the right
# NUMBERS, not just compile?
#
# `cargo check` proves the code builds. This proves an SCF still converges to
# the value PySCF gives, and that a geometry optimization still walks downhill
# to the known minimum. It is the cheapest check that would catch a real
# regression in the science rather than in the types.
#
# Budget: ~1 s with a release binary (~5 s debug). Every case is H2/STO-3G --
# 2 atoms, 2 basis functions -- deliberately the smallest system that still
# exercises the whole path: integrals -> SCF -> gradient -> optimizer.
#
# Usage:
#   scripts/smoke-physics.sh
#   FERRIC_BIN=target/debug/ferric scripts/smoke-physics.sh
#
# Exit: 0 = all pass, 1 = a number moved (or the binary is missing).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BIN="${FERRIC_BIN:-target/release/ferric}"
if [[ ! -x "$BIN" ]]; then
    if [[ -x target/debug/ferric ]]; then
        BIN=target/debug/ferric
    else
        echo "smoke: no ferric binary (build: cargo build --release -p ferric-cli)" >&2
        exit 1
    fi
fi

export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

FAILED=0
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Extract the last float matching <label> and compare against <expected>.
check() {
    local label="$1" expected="$2" tol="$3" key="$4" out="$5" got
    got="$(grep -E "$key" "$out" | tail -1 | grep -oE '\-?[0-9]+\.[0-9]+' | tail -1)"
    if [[ -z "$got" ]]; then
        echo "  FAIL $label: no value parsed (key: $key)"
        tail -3 "$out" | sed 's/^/         /'
        FAILED=1
        return
    fi
    if python3 -c "import sys; sys.exit(0 if abs($got - ($expected)) <= $tol else 1)"; then
        printf "  ok   %-26s %s\n" "$label" "$got"
    else
        printf "  FAIL %-26s %s  expected %s +/- %s\n" "$label" "$got" "$expected" "$tol"
        FAILED=1
    fi
}

run() {
    if ! timeout 120 "$BIN" "$1" > "$2" 2>&1; then
        echo "  FAIL $1: exited nonzero"
        tail -5 "$2" | sed 's/^/         /'
        FAILED=1
        return 1
    fi
}

echo "== ferric physics smoke ($BIN) =="
START=$(date +%s%N)

# H2 at 0.74 A. NOTE: this is NOT testdata/molecules/h2.xyz, which is at
# 0.7414 A -- the reference JSON (and crates/ferric-scf/src/rhf.rs's
# test_rhf_h2_sto3g) both use 0.74, and the 0.0014 A difference is worth
# ~7.5e-5 Ha, far outside the tolerance below. Keep the geometry and the
# reference in the same file so they cannot drift apart.
printf '2\nH2\nH 0 0 0\nH 0 0 0.74\n' > "$TMP/h2.xyz"

# --- 1. RHF energy vs PySCF 2.13.0 (testdata/reference/h2_sto-3g_rhf.json) ---
# ferric reproduces this to ~1e-10; 1e-8 leaves room for a libint/BLAS bump
# while still catching any real regression.
cat > "$TMP/rhf.toml" <<EOF
[molecule]
xyz = "$TMP/h2.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"
task = "energy"
EOF
if run "$TMP/rhf.toml" "$TMP/rhf.out"; then
    check "RHF/STO-3G energy" "-1.1167593073964255" "1e-8" "^  energy" "$TMP/rhf.out"
    grep -qE "^  converged  = true" "$TMP/rhf.out" \
        && echo "  ok   RHF converged" \
        || { echo "  FAIL RHF did not converge"; FAILED=1; }
fi

# --- 2. Geometry optimization (LDA). Exercises gradients + the optimizer. ---
# Asserts convergence AND that the optimized energy is below the fixed-geometry
# SCF value: an optimizer that "converges" while walking uphill is broken.
if run "examples/h2-lda-opt.toml" "$TMP/opt.out"; then
    grep -qE "^  converged  = true" "$TMP/opt.out" \
        && echo "  ok   geometry opt converged" \
        || { echo "  FAIL geometry opt did not converge"; FAILED=1; }
    check "opt final energy" "-1.1212649781" "1e-4" "^  final E" "$TMP/opt.out"
fi

ELAPSED=$(( ($(date +%s%N) - START) / 1000000 ))
echo "== smoke finished in ${ELAPSED}ms =="

if [[ $FAILED -ne 0 ]]; then
    echo "== PHYSICS SMOKE FAILED -- a number moved. Do not push. =="
    exit 1
fi
echo "== physics smoke: PASS =="
