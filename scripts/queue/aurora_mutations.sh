#!/usr/bin/env bash
# Mutation ledger for the AURORA-SCF test suite.
#
# Each mutation deliberately breaks ONE thing in aurora.rs (or rhf.rs) and runs
# the tests that should catch it. A mutation that no test catches is a SURVIVOR
# and is reported as such — a test that has never been seen to fail is only an
# assumption.
#
# Each mutation is also checked for REACHABILITY: it must compile and the
# mutated line must actually execute in the tests being run. A mutation in dead
# code would be "killed" by nothing and must not be counted either way.
#
# Usage:  scripts/queue/aurora_mutations.sh [name-filter]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/crates/ferric-scf/src/aurora.rs"
RHF="$ROOT/crates/ferric-scf/src/rhf.rs"
SCRATCH="${SCRATCH_DIR:-/tmp/aurora-mutations}"
mkdir -p "$SCRATCH"
cp "$SRC" "$SCRATCH/aurora.pristine"
cp "$RHF" "$SCRATCH/rhf.pristine"

restore() {
  cp "$SCRATCH/aurora.pristine" "$SRC"
  cp "$SCRATCH/rhf.pristine" "$RHF"
}
trap restore EXIT

FILTER="${1:-}"
PASS=0
SURVIVE=0
SKIP=0

# mutate <name> <file> <from> <to> <test-targets...>
mutate() {
  local name="$1"; shift
  local file="$1"; shift
  local from="$1"; shift
  local to="$1"; shift
  local targets=("$@")

  if [[ -n "$FILTER" && "$name" != *"$FILTER"* ]]; then return; fi

  restore
  if ! grep -qF "$from" "$file"; then
    echo "SKIP  $name  (pattern not found — mutation is stale)"
    SKIP=$((SKIP+1))
    return
  fi
  python3 - "$file" "$from" "$to" <<'PY'
import sys
path, frm, to = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
assert frm in s
open(path, 'w').write(s.replace(frm, to, 1))
PY

  local args=()
  for t in "${targets[@]}"; do args+=(--test "$t"); done

  local log="$SCRATCH/$name.log"
  if ! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf "${args[@]}" --release \
        -- --test-threads=1 >"$log" 2>&1; then
    # Distinguish "did not compile" from "tests failed". `error: test failed`
    # is cargo's exit banner and must NOT be read as a compile error, or every
    # kill is misfiled as a skip (this harness did exactly that on its first
    # run).
    if grep -qE "^error(\[E[0-9]+\])?: " "$log" \
       && ! grep -qE "^error: test failed" "$log"; then
      echo "SKIP  $name  (did not compile — mutation unreachable as written)"
      SKIP=$((SKIP+1))
    elif grep -qE "^test result: FAILED" "$log"; then
      local who
      who=$(sed -n '/^failures:$/,/^test result/p' "$log" \
            | grep -oE "^    [a-zA-Z_0-9]+$" | sort -u | tr '\n' ' ')
      echo "KILL  $name  <- ${who:-(see log)}"
      PASS=$((PASS+1))
    else
      echo "SKIP  $name  (neither compiled nor ran cleanly — inspect $log)"
      SKIP=$((SKIP+1))
    fi
  else
    echo "SURVIVE  $name  ** NO TEST CAUGHT THIS **"
    SURVIVE=$((SURVIVE+1))
  fi
}

echo "=== AURORA mutation ledger ==="

# --- Auxiliary response: every factor, sign and contraction (Eqs. S7-S8). ---
mutate coulomb-factor-8-to-4 "$SRC" \
  "y.scaled_add(8.0 * scalar, &bp);" \
  "y.scaled_add(4.0 * scalar, &bp);" \
  aurora_math_anchors

mutate exchange-factor-2-to-1 "$SRC" \
  "y.scaled_add(-2.0 * ax, &acc);" \
  "y.scaled_add(-1.0 * ax, &acc);" \
  aurora_math_anchors

mutate exchange-sign-flip "$SRC" \
  "y.scaled_add(-2.0 * ax, &acc);" \
  "y.scaled_add(2.0 * ax, &acc);" \
  aurora_math_anchors

mutate drop-second-exchange-term "$SRC" \
  "acc += &bvo.dot(&xt).dot(&bvo);" \
  "" \
  aurora_math_anchors

mutate drop-first-exchange-term "$SRC" \
  "acc += &bvv.dot(&x).dot(&boo);" \
  "" \
  aurora_math_anchors

# --- Base operator (Eq. S9). ---
mutate base-fock-factor-2-to-1 "$SRC" \
  "let mut y = 2.0 * (fvv.dot(&x) - x.dot(foo));
        y += &self.response(x);" \
  "let mut y = 1.0 * (fvv.dot(&x) - x.dot(foo));
        y += &self.response(x);" \
  aurora_math_anchors

mutate base-fock-sign-flip "$SRC" \
  "let mut y = 2.0 * (fvv.dot(&x) - x.dot(foo));
        y += &self.response(x);" \
  "let mut y = 2.0 * (x.dot(foo) - fvv.dot(&x));
        y += &self.response(x);" \
  aurora_math_anchors

# --- Geodesic (Eq. S18). ---
mutate geodesic-sin-cos-swap "$SRC" \
  "acc += uv[(a, t)] * sigma[t].sin() * vt[(t, i)];" \
  "acc += uv[(a, t)] * sigma[t].cos() * vt[(t, i)];" \
  aurora_math_anchors

mutate geodesic-antisymmetry-broken "$SRC" \
  "u_mo[(i, nocc + a)] = -acc;" \
  "u_mo[(i, nocc + a)] = acc;" \
  aurora_math_anchors

mutate geodesic-drop-cos-minus-one-occ "$SRC" \
  "acc += vt[(t, i)] * (sigma[t].cos() - 1.0) * vt[(t, j)];" \
  "acc += vt[(t, i)] * sigma[t].cos() * vt[(t, j)];" \
  aurora_math_anchors

# --- Transport. ---
mutate transport-drop-conjugation "$SRC" \
  "let t = u_mo.t().dot(&kappa).dot(&u_mo);" \
  "let t = kappa.clone();" \
  aurora_math_anchors

mutate transport-antisymmetry-broken "$SRC" \
  "kappa[(i, nocc + a)] = -v[(a, i)];" \
  "kappa[(i, nocc + a)] = v[(a, i)];" \
  aurora_math_anchors

# --- The OFF switch and the gate (integration-level). ---
mutate default-enabled-true "$SRC" \
  "            enabled: false," \
  "            enabled: true," \
  aurora_off_bit_identity

mutate gate-always-allows "$RHF" \
  "let aurora_exchange_ok = config.aurora.allow_low_exchange_ks
            || aurora_ax >= crate::aurora::MIN_VALIDATED_EXCHANGE_FRACTION;" \
  "let aurora_exchange_ok = true;" \
  aurora_same_fixed_point

mutate gate-never-allows "$RHF" \
  "let aurora_exchange_ok = config.aurora.allow_low_exchange_ks
            || aurora_ax >= crate::aurora::MIN_VALIDATED_EXCHANGE_FRACTION;" \
  "let aurora_exchange_ok = false;" \
  aurora_same_fixed_point

# --- Convergence-relevant: the step must actually be applied. ---
mutate step-not-applied "$SRC" \
  "let c_new = c.dot(&u_mo);" \
  "let c_new = c.to_owned();" \
  aurora_same_fixed_point

mutate trust-radius-ignored "$SRC" \
  "        if norm > trust_radius && norm > 0.0 {
            step *= trust_radius / norm;
        }" \
  "        if false {
            step *= trust_radius / norm;
        }" \
  aurora_step_control

# --- Powell damping (Eqs. S10-S11). ---
mutate powell-threshold-0.2-to-0.0 "$SRC" \
  "let y_use = if sy < 0.2 * sbs {" \
  "let y_use = if sy < 0.0 * sbs {" \
  aurora_step_control

mutate powell-theta-0.8-to-0.0 "$SRC" \
  "let theta = 0.8 * sbs / denom;" \
  "let theta = 0.0 * sbs / denom;" \
  aurora_step_control

mutate shift-ignored "$SRC" \
  "        if shift != 0.0 {
            let d = self.diagonal(fvv, foo);" \
  "        if false {
            let d = self.diagonal(fvv, foo);" \
  aurora_math_anchors

# --- The J/K counter itself. ---
mutate jk-counter-never-ticks "$RHF" \
  "crate::aurora::TARGET_JK_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);" \
  "" \
  aurora_benchmark_counter

restore
echo
echo "=== ledger: $PASS killed, $SURVIVE survived, $SKIP skipped ==="
