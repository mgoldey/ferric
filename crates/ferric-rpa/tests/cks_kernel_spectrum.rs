//! Is the MP2C correction's sign a bug, or a property of the construction?
//!
//! # The question
//!
//! ferric's MP2C correction is positive for every system measured:
//!
//! ```text
//!   system     published corr   ferric corr
//!   a24-12         +0.060         +0.395      right sign, 6.6x too large
//!   methane        -0.060         +0.195      WRONG SIGN
//!   ethene         -0.040         +0.366      WRONG SIGN
//! ```
//!
//! Two readings are possible and they call for opposite responses:
//!
//! 1. **a bug** — something in the Dyson solve, kernel projection, or
//!    quadrature has a sign or factor error, and fixing it recovers the
//!    published numbers; or
//! 2. **a different quantity** — the construction as specified cannot produce a
//!    negative number at all, so no amount of debugging *this* code will
//!    reproduce the published table, and the gap is in the protocol
//!    (asymptotically corrected orbitals, MP2/CBS baseline) rather than here.
//!
//! # What decides between them
//!
//! `E_disp = −(1/2π)∫dω Tr[Π_A Π_B]` with `Π = [I + Π⁰K]⁻¹Π⁰` and `K = I + F̃`.
//!
//! If `K ≻ 0`, the Dyson step is a **strict contraction**: every eigenvalue of
//! `Π⁰` shrinks, hence the trace shrinks, hence `|E^CKS| < |E^UCHF|`, hence
//! `e_correction > 0`. That holds for every system, independent of physics.
//! A negative correction therefore *requires* `K` to have a negative eigenvalue.
//!
//! This was checked over 20 000 randomized PSD `Π⁰` pairs before writing this
//! test: **0** sign violations in 6712 positive-definite-`K` draws, versus 237
//! genuine `corr < 0` cases once `K` was allowed to be indefinite. The
//! implication is sharp and one-directional — measuring `min eig K` on a real
//! molecule settles which of the two readings above is correct.
//!
//! The ALDA `f_xc` **is** negative (that is standard), so `K = I + F̃` going
//! indefinite is not absurd a priori. Whether it does is an empirical question
//! about magnitude, which is exactly what this file measures rather than
//! assumes. Until now the claim "`K` is positive definite in practice" appeared
//! only in prose, asserted by nothing.
//!
//! # THE ANSWER (measured 2026-07-26): the claim is FALSE
//!
//! ```text
//!   water/cc-pVDZ, K = I + F_xc, naux = 84
//!     ALDAX      min eig = -5.588   max eig = 0.968   50 of 84 negative
//!     ALDA X+C   min eig = -6.964
//! ```
//!
//! `K` is **strongly indefinite** — not marginally, and not only for one
//! functional. So the Dyson step is *not* a contraction, `e_correction > 0` is
//! **not** structurally guaranteed, and reading 2 above is refuted:
//!
//! > **The MP2C sign discrepancy is a live bug in this code, not a protocol
//! > difference.** The published -0.060 / -0.040 values ARE reachable by this
//! > construction and remain a legitimate debugging target.
//!
//! Note the corollary that makes this urgent rather than merely interesting: an
//! indefinite `I + Π⁰K` can be near-singular, so individual frequency points may
//! be passing through a near-pole. That is a plausible mechanism for ferric's
//! correction being 6.6x too large on a24-12 (+0.395 vs published +0.060) while
//! the underlying uncoupled machinery validates to 0.5%, and it is the first
//! thing to check.
//!
//! Correlation *deepening* the indefiniteness (-5.588 -> -6.964) is a free
//! cross-check that `project_fxc_to_aux` is assembling the kernel correctly:
//! ALDA correlation is negative, so it must push `min eig` down. It does.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::cks_dispersion::{build_monomer_response, CksDispersionConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Build a monomer response carrying the ALDA kernel, and report `K`'s spectrum.
///
/// Mirrors `cks_dispersion_uncoupled.rs::setup` so the two files exercise the
/// same construction path — a spectrum measured through a different setup would
/// not license any conclusion about the dispersion numbers that motivated it.
fn kernel_spectrum(xyz: &str, xc: &str) -> Vec<f64> {
    let mol = Molecule::parse_xyz(xyz, 0, 1).expect("parse xyz");
    let bs = basis::bundled("cc-pvdz").expect("orbital basis");
    let obs = PreparedBasis::new(&mol, &bs).expect("prepare obs");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let scf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .expect("rhf");
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").expect("aux basis"))
        .expect("prepare aux");

    let cfg = CksDispersionConfig {
        n_freq: 8,
        u0: 1.0,
        frozen_core: 0,
        ..Default::default()
    };
    let resp = build_monomer_response(&mol, &obs, &dfbs, &scf, xc, &cfg).expect("response");

    resp.kernel_eigenvalues()
        .expect("CKS coupling must carry an XC kernel")
}

/// Water/cc-pVDZ, ALDAX — the published MP2C kernel.
///
/// Measures `K = I + F̃` and asserts it is INDEFINITE, i.e. a negative MP2C
/// correction is reachable by this construction, so the published negative
/// values are a legitimate debugging target. If a change ever makes `K`
/// positive definite this fails, and the structural "the correction is always
/// positive" argument must be revisited rather than silently re-adopted.
#[test]
fn alda_kernel_is_indefinite_so_a_negative_correction_is_reachable() {
    let water = "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.467\nH 0.0 -0.757 -0.467\n";
    let eigs = kernel_spectrum(water, "LDA_X");

    let min = eigs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "K = I + F_xc (water/cc-pVDZ, ALDAX): n = {}, min eig = {min:.6}, max eig = {max:.6}",
        eigs.len()
    );
    let n_neg = eigs.iter().filter(|&&e| e < 0.0).count();
    eprintln!("  eigenvalues below zero: {n_neg}");

    // MEASURED 2026-07-26: min eig = -5.588, with 50 of 84 eigenvalues below
    // zero. The positivity claim that the sign conclusion rested on is FALSE.
    //
    // This assertion is deliberately the inverse of the one this file was
    // written to confirm. It is pinned as a regression: if a future change
    // makes K positive definite, the structural argument becomes available
    // again and THAT is a result worth failing a test over.
    assert!(
        min < 0.0,
        "K = I + F_xc is now positive definite (min = {min:.6e}) -- it was measured \
         INDEFINITE (min -5.588, 50 of 84 negative) on 2026-07-26. If this is a \
         deliberate fix, the 'e_correction > 0 is structural' argument in mp2c_cks.rs \
         and the validation page becomes valid again and both must be revisited."
    );
    assert!(
        n_neg > 0 && n_neg < eigs.len(),
        "expected a genuinely INDEFINITE kernel (some but not all eigenvalues negative); \
         got {n_neg} of {} below zero -- an all-negative K would suggest an overall sign \
         error in the projection rather than the physical mix of signs ALDA produces",
        eigs.len()
    );

    // The kernel must actually be doing something -- a K that is numerically
    // indistinguishable from the identity would make this test vacuous (it
    // would pass for the trivial reason that F_xc ~ 0, not because ALDA is too
    // weak to break positivity). Guard against that reading explicitly.
    let dev = eigs.iter().map(|e| (e - 1.0).abs()).fold(0.0, f64::max);
    eprintln!("  max |eig - 1| = {dev:.6}  (0 would mean the kernel is absent)");
    assert!(
        dev > 1e-6,
        "K is indistinguishable from the identity (max |eig-1| = {dev:.3e}) -- the XC kernel \
         is not being applied at all, so this test proves nothing about ALDA's strength"
    );
}

/// Same measurement with LDA correlation added (ALDA X+C).
///
/// Correlation makes `f_xc` more negative, so it should deepen the indefiniteness
/// rather than relieve it. Measured: ALDAX -5.588 -> ALDA X+C **-6.964**.
///
/// That ordering is itself a correctness check on `project_fxc_to_aux`: the
/// correlation part is negative, so if adding it moved `min eig K` *up* the c
/// contribution would be dropped or sign-flipped. It moves down, so the kernel
/// assembly is behaving, and the indefiniteness is a real property of ALDA in
/// this basis rather than an artifact of one functional choice.
#[test]
fn adding_lda_correlation_deepens_the_indefiniteness() {
    let water = "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.467\nH 0.0 -0.757 -0.467\n";
    let x = kernel_spectrum(water, "LDA_X");
    let xc = kernel_spectrum(water, "LDA");

    let min_x = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let min_xc = xc.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!("min eig K: ALDAX = {min_x:.6}, ALDA X+C = {min_xc:.6}");

    // Both are indefinite; correlation deepens it (-5.588 -> -6.964).
    assert!(
        min_xc < 0.0,
        "K with LDA correlation is positive definite (min = {min_xc:.6e}); it was measured \
         at -6.964 on 2026-07-26. See the sibling test."
    );

    // Correlation should push the kernel further from the identity, not less.
    // If it does not, the correlation part is not reaching the projection.
    assert!(
        min_xc <= min_x + 1e-8,
        "adding LDA correlation moved min eig K UP ({min_x:.6} -> {min_xc:.6}); ALDA correlation \
         is negative, so it must not make the kernel more positive -- the c part is likely \
         being dropped or sign-flipped in project_fxc_to_aux"
    );
}
