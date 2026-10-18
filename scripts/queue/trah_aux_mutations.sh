#!/usr/bin/env bash
# Mutation ledger for the TRAH auxiliary-curvature experiment.
#
# Each mutation deliberately breaks ONE thing and names the test that must
# catch it. A mutation nobody catches is a SURVIVOR and is reported as such.
#
# This script exists because four mutations on this stack failed silently in
# one session: one patched a different code path, one was below the observable,
# one had its source restored mid-build, and one had an anchor `cargo fmt` had
# already rewrapped. So every mutation here is VERIFIED APPLIED (grep after
# patching, and again after the build starts) and VERIFIED REVERTED.
#
# Usage:  OPENBLAS_NUM_THREADS=1 bash scripts/queue/trah_aux_mutations.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1

TRAH=crates/ferric-scf/src/trah.rs
RHF=crates/ferric-scf/src/rhf.rs
AUR=crates/ferric-scf/src/aurora.rs
BACKUP=$(mktemp -d)
cp "$TRAH" "$BACKUP/trah.rs"
cp "$RHF" "$BACKUP/rhf.rs"
cp "$AUR" "$BACKUP/aurora.rs"
restore() {
  cp "$BACKUP/trah.rs" "$TRAH"
  cp "$BACKUP/rhf.rs" "$RHF"
  cp "$BACKUP/aurora.rs" "$AUR"
}
trap restore EXIT

pass=0; surv=0

# $1 label | $2 file | $3 sed expr | $4 grep-proof of the MUTATED text | $5 test filter | $6 extra cargo args
mutate() {
  local label=$1 file=$2 expr=$3 proof=$4 filter=$5; shift 5
  # MUT_ONLY=<substring> runs just the mutations whose label contains it.
  if [ -n "${MUT_ONLY:-}" ] && [[ "$label" != *"$MUT_ONLY"* ]]; then return; fi
  echo "=============================================================="
  echo "MUTATION: $label"
  restore
  sed -i "$expr" "$file"
  if ! grep -qF "$proof" "$file"; then
    echo "  NOT APPLIED -- the sed did not change the file. INVALID mutation."
    surv=$((surv+1)); restore; return
  fi
  echo "  applied (verified by grep: '$proof')"
  # Re-verify immediately before the test binary runs, so a concurrent restore
  # or a formatter cannot silently un-mutate the source under us.
  if ! grep -qF "$proof" "$file"; then
    echo "  VANISHED before build. INVALID."
    surv=$((surv+1)); restore; return
  fi
  local out; out=$(cargo test --release -j 3 -p ferric-scf "$@" -- --test-threads=1 --include-ignored "$filter" 2>&1)
  local rc=$?
  # And once more after: proves the tested binary was built from mutated source.
  if ! grep -qF "$proof" "$file"; then
    echo "  source was restored DURING the run. INVALID."
    surv=$((surv+1)); restore; return
  fi
  if [ $rc -ne 0 ]; then
    echo "  CAUGHT by $filter"
    echo "$out" | grep -A1 -E '^test .*(FAILED|ok)|panicked at|assertion|must |expected |rel diff|floor bar|monotone' | head -8
    pass=$((pass+1))
  else
    echo "  *** SURVIVOR *** $filter did not notice"
    echo "$out" | grep -E 'test result' | head -2
    surv=$((surv+1))
  fi
  restore
}

# 1. The scale constant. This is THE one that would produce a clean, wrong rho.
mutate "AURORA_TO_TRAH_CURVATURE 0.5 -> 1.0 (drops the factor-2 convention)" \
  "$TRAH" \
  's/^pub const AURORA_TO_TRAH_CURVATURE: f64 = 0.5;/pub const AURORA_TO_TRAH_CURVATURE: f64 = 1.0;/' \
  'pub const AURORA_TO_TRAH_CURVATURE: f64 = 1.0;' \
  'aux_and_exact_curvature_share_one_scale' \
  --test trah_aux_curvature

# 2. The gate. If aux_curvature=false still takes the aux path, bit-identity dies.
mutate "gate inverted: aux path runs when the flag is FALSE (dispatch in trah.rs)" \
  "$TRAH" \
  's/^    if !config.trah.aux_curvature {/    if config.trah.aux_curvature {/' \
  '    if config.trah.aux_curvature {' \
  'trah_aux_off_is_bit_identical' \
  --test trah_aux_curvature

# 3. Wrong Fock block: f_oo built from the virtual corner. Breaks the gap term.
#    LEDGER 2026-10-17: was a SURVIVOR (the fixed-point test only sees the path
#    to the same fixed point; the operator anchors skip step-level wiring).
#    Now caught by the step-level anchor, which compares a full aux TRAH step
#    with the full-gap exact step from a non-canonical start (#279).
mutate "f_oo read from the VIRTUAL block of F_mo" \
  "$TRAH" \
  's/            f_oo\[(i, j)\] = inp.f_mo\[(i, j)\];/            f_oo[(i, j)] = inp.f_mo[(no + i, no + j)];/' \
  'f_oo[(i, j)] = inp.f_mo[(no + i, no + j)];' \
  'aux_trah_step_matches_full_gap_exact_step_off_canonical' \
  --test trah_aux_response_anchor

# 4. Stale tangent space: never refresh the aux model onto current orbitals.
mutate "refresh_orbitals removed (aux tensor frozen at the first geometry)" \
  "$TRAH" \
  's/^    aux.refresh_orbitals(c)?;/    \/\/ MUTANT: refresh removed/' \
  '// MUTANT: refresh removed' \
  'aux_trah_step_matches_full_gap_exact_step_off_canonical' \
  --test trah_aux_response_anchor

# 5. The rho log. If note_rho_rhf stops recording, the central measurement is empty.
mutate "note_rho_rhf no longer appends to the log" \
  "$TRAH" \
  's/^        v.push(rho);/        let _ = rho;/' \
  'let _ = rho;' \
  'aux_curvature_biases_rho_below_one' \
  --test trah_aux_rho

# 6. Symmetry: make the Bvv factor non-symmetric (zero its upper triangle).
#    `Bvv X Boo` is symmetric as a whole only because Bvv and Boo are; breaking
#    Bvv breaks the operator's symmetry without changing any shape.
mutate "aux response: Bvv upper triangle zeroed (operator no longer symmetric)" \
  "$AUR" \
  's|^                let bvv = self.bvv.slice(s!\[p, .., ..\]);|                let mut bvv = self.bvv.slice(s![p, .., ..]).to_owned();\n                for r in 0..bvv.nrows() { for c in (r + 1)..bvv.ncols() { bvv[(r, c)] = 0.0; } }|' \
  'for c in (r + 1)..bvv.ncols() { bvv[(r, c)] = 0.0; }' \
  'aux_curvature_operator_is_symmetric' \
  --test trah_aux_curvature

# 7-11. Mutations the original ledger LACKED: they touch the CONTENT of the 2e
#       response, which no earlier test compared with the exact matvec (#279).
mutate "aux response: exchange (K) dropped entirely" \
  "$AUR" \
  's/y.scaled_add(-2.0 \* ax, &acc);/y.scaled_add(-0.0 * ax, \&acc);/' \
  'y.scaled_add(-0.0 * ax, &acc);' \
  'rich_aux_response_matches_exact_hessian_matvec' \
  --test trah_aux_response_anchor

mutate "aux response: one exchange contraction (Bvo Xt Bvo) dropped" \
  "$AUR" \
  's/acc += &bvo.dot(&xt).dot(&bvo);/acc += \&(bvo.dot(\&xt).dot(\&bvo) * 0.0);/' \
  'dot(&bvo) * 0.0' \
  'rich_aux_response_matches_exact_hessian_matvec' \
  --test trah_aux_response_anchor

mutate "aux response: exchange term 2 replaced by a copy of term 1 (index pairing wrong)" \
  "$AUR" \
  's/acc += &bvo.dot(&xt).dot(&bvo);/acc += \&bvv.dot(\&x).dot(\&boo);/' \
  'acc += &bvv.dot(&x).dot(&boo);' \
  'rich_aux_response_matches_exact_hessian_matvec' \
  --test trah_aux_response_anchor

mutate "aux response: Coulomb prefactor 8 -> 4" \
  "$AUR" \
  's/y.scaled_add(8.0 \* scalar, &bp);/y.scaled_add(4.0 * scalar, \&bp);/' \
  'y.scaled_add(4.0 * scalar, &bp);' \
  'rich_aux_response_matches_exact_hessian_matvec' \
  --test trah_aux_response_anchor

# Small construction defect: 0.5% on the Coulomb term. Below the def2 fitting
# error, above the qz-jkfit floor, so only the aux-enrichment ladder sees it.
mutate "aux response: Coulomb prefactor 8 -> 8.04 (0.5% defect)" \
  "$AUR" \
  's/y.scaled_add(8.0 \* scalar, &bp);/y.scaled_add(8.04 * scalar, \&bp);/' \
  'y.scaled_add(8.04 * scalar, &bp);' \
  'aux_error_falls_with_aux_enrichment_to_a_derived_floor' \
  --test trah_aux_response_anchor

# The non-canonical anchor must see the one-electron gap confound: zeroing the
# full-gap correction makes the control arm the diagonal-gap matvec.
mutate "full-gap correction zeroed" \
  "$TRAH" \
  's/^    let mut out = f_vv.dot(k) - k.dot(&f_oo);/    let mut out = f_vv.dot(k) * 0.0 - k.dot(\&f_oo) * 0.0;/' \
  'f_vv.dot(k) * 0.0' \
  'rich_aux_matches_full_gap_exact_at_noncanonical_point' \
  --test trah_aux_response_anchor

# The origin/main golden must see a change that perturbs BOTH the default and
# the explicit-false arm (default-vs-explicit comparison alone cannot).
mutate "RHF_ENERGY_SCALE 4.0 -> 3.0 (alters the exact path in both arms)" \
  "$TRAH" \
  's/^const RHF_ENERGY_SCALE: f64 = 4.0;/const RHF_ENERGY_SCALE: f64 = 3.0;/' \
  'const RHF_ENERGY_SCALE: f64 = 3.0;' \
  'trah_aux_off_is_bit_identical' \
  --test trah_aux_curvature

echo "=============================================================="
echo "LEDGER: $pass caught, $surv survivors/invalid"
