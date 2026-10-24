//! The sharp internal tests for the CKS dispersion machinery.
//!
//! These are the checks that can be made *without* a published MP2C reference
//! value, and they are the ones that would actually catch an error:
//!
//!  1. **Uncoupled ≡ UCHF dispersion.** The Casimir–Polder trace built from the
//!     bare bubble `Π⁰` must reproduce the non-additive (intermolecular) part of
//!     the direct-MP2 energy `2·E_OS` (the direct ring term of the
//!     second-order energy). It is the only
//!     test here that validates the *physics* rather than the plumbing: it
//!     independently confirms the frequency integral, the `4·e_ia/(ω²+e_ia²)`
//!     bubble convention, the `−1/2π` prefactor, AND the aux-basis dressing all
//!     at once, because getting any one of them wrong breaks the match.
//!
//!  2. **fxc off ⇒ uncoupled, exactly.** `CoupledHartree` with the Dyson
//!     resummation is *not* the same as `Uncoupled` (that is the point), but
//!     `Cks` with a null functional must equal `CoupledHartree` bit-for-bit,
//!     because the only difference is adding a zero matrix.
//!
//!  3. **Sign and ordering.** Dispersion is attractive; coupling *screens* it,
//!     so |E_disp^coupled| < |E_disp^uncoupled| for a normal closed-shell pair.
//!
//! A note on why test 1 is a difference and not an absolute equality: `2·E_OS`
//! on a single system is a supermolecular correlation energy containing
//! intramonomer correlation, while `E_disp^(2)` is interaction-only by
//! construction. Only the dimer-minus-monomers difference is comparable.

//! # Mutation ledger (2026-10-23, against this file, `cks_dispersion_separation.rs`,
//! `cks_kernel_spectrum.rs` and the `cks_dispersion` unit test)
//!
//! * alignment check disabled → `aux_alignment_is_enforced` FAILS
//! * aux-index misalignment reintroduced in the trace (B's aux order reversed)
//!   → `cks_dispersion_decays_with_separation` FAILS
//! * `E_disp` prefactor sign flipped → `cks_dispersion_decays_with_separation` FAILS
//! * XC kernel sign flipped in the Dyson kernel → `coupling_screens_dispersion`,
//!   `dyson_min_eigenvalue_reproduces_its_exact_limits`,
//!   `cross_spin_fxc_term_is_included` FAIL
//! * symmetrized Dyson matrix built with `Π⁰` instead of `Π⁰^{1/2}` →
//!   `dyson_min_eigenvalue_matches_the_unsymmetrized_spectrum` FAILS
//! * spin factor ½ dropped from `f_xc` → `coupling_screens_dispersion`,
//!   `cross_spin_fxc_term_is_included` FAIL
//! * ESCAPES the per-commit tier: the static stability diagnostic reading
//!   monomer A twice (He₂ is symmetric); seen only by the asymmetric
//!   ignored runs in `mp2c_cks_published_sign.rs`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::cks_dispersion::{
    build_monomer_response, cks_dispersion_energy, CksCoupling, CksDispersionConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

/// Two well-separated He atoms: the smallest system with a genuine
/// intermolecular dispersion interaction and no covalent contamination.
/// He/cc-pVDZ has 5 basis functions per atom, so the dimer basis is 10 — small
/// enough to run the full dense Dyson solve in a unit test.
const R_HE: f64 = 4.0;

fn he_dimer_xyz() -> String {
    format!("2\nHe2\nHe 0 0 0\nHe 0 0 {R_HE}\n")
}
/// Monomer A in the *dimer* basis: real He at the origin, ghost He at the
/// partner's position. This is the counterpoise / monomer-in-dimer-basis
/// construction the Casimir-Polder trace requires.
fn he_mono_a_xyz() -> String {
    format!("2\nHe A (dimer basis)\nHe 0 0 0\n@He 0 0 {R_HE}\n")
}
fn he_mono_b_xyz() -> String {
    format!("2\nHe B (dimer basis)\n@He 0 0 0\nHe 0 0 {R_HE}\n")
}

fn setup(xyz: &str) -> (Molecule, PreparedBasis, PreparedBasis, ScfResult) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
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
    .unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    (mol, obs, dfbs, rhf)
}

/// `2·E_OS` — the direct-MP2 (uncoupled dispersion) energy of one system.
fn two_e_os(mol: &Molecule, obs: &PreparedBasis, dfbs: &PreparedBasis, rhf: &ScfResult) -> f64 {
    use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
    let sc = ri_mp2_spin_components(
        mol,
        obs,
        dfbs,
        Operator::coulomb(),
        rhf,
        &RiMp2Config::default(),
    )
    .unwrap();
    2.0 * sc.0.e_os
}

/// TEST 1 — the physics check.
///
/// `E_disp^UCHF` from the Casimir–Polder machinery must equal the non-additive
/// part of `2·E_OS`:
///
/// ```text
///   E_disp^uncoupled(A,B)  ==  2E_OS(dimer) − 2E_OS(A) − 2E_OS(B)
/// ```
///
/// with all three `2E_OS` evaluated in the same dimer basis (CP construction).
#[test]
fn uncoupled_dispersion_matches_direct_mp2_nonadditivity() {
    let (dm, do_, dd, dr) = setup(&he_dimer_xyz());
    let (am, ao, ad, ar) = setup(&he_mono_a_xyz());
    let (bm, bo, bd, br) = setup(&he_mono_b_xyz());

    // Supermolecular reference: the non-additive part of the direct MP2 energy.
    let e_ref =
        two_e_os(&dm, &do_, &dd, &dr) - two_e_os(&am, &ao, &ad, &ar) - two_e_os(&bm, &bo, &bd, &br);

    // Casimir–Polder, uncoupled. A dense frequency grid, because THIS test is
    // about whether the identity holds, not about how cheaply it converges.
    let cfg = CksDispersionConfig {
        coupling: CksCoupling::Uncoupled,
        n_freq: 40,
        u0: 1.0,
        frozen_core: 0,
    };
    let ra = build_monomer_response(&am, &ao, &ad, &ar, "", &cfg).unwrap();
    let rb = build_monomer_response(&bm, &bo, &bd, &br, "", &cfg).unwrap();
    let disp = cks_dispersion_energy(&ra, &rb, &cfg).unwrap();

    // Both must be attractive.
    assert!(
        e_ref < 0.0,
        "reference non-additive 2E_OS should be attractive, got {e_ref:.3e}"
    );
    assert!(
        disp.e_disp < 0.0,
        "E_disp^UCHF should be attractive, got {:.3e}",
        disp.e_disp
    );

    // The identity is exact in the complete-basis/exact-response sense; here it
    // is limited by the frequency quadrature and by the fact that the two
    // constructions distribute the intramonomer/intermonomer split slightly
    // differently in a finite basis. A 5% agreement on a number of order 1e-5
    // Ha already rules out every convention error (a wrong prefactor or a
    // missed dressing is an O(1) or larger discrepancy, not a 5% one).
    let rel = (disp.e_disp - e_ref).abs() / e_ref.abs();
    eprintln!(
        "He2/cc-pVDZ 4 A: E_disp^UCHF = {:.10e}, 2E_OS non-additivity = {:.10e}, rel = {rel:.4e}",
        disp.e_disp, e_ref
    );
    assert!(
        rel < 0.05,
        "uncoupled Casimir-Polder dispersion does not reproduce the direct-MP2 \
         non-additivity:\n  E_disp^UCHF = {:.8e} Ha\n  2E_OS non-add = {:.8e} Ha\n  rel err = {:.3}",
        disp.e_disp,
        e_ref,
        rel
    );
}

/// TEST 2 — switching the XC kernel off must be exact, not approximate.
///
/// `Cks` with a functional whose `f_xc` is identically zero adds a zero matrix
/// to the Dyson kernel, so it must agree with `CoupledHartree` to the last bit.
/// Anything else means the two paths are not the same code.
#[test]
fn cks_with_null_fxc_equals_coupled_hartree_bitwise() {
    let (am, ao, ad, ar) = setup(&he_mono_a_xyz());
    let (bm, bo, bd, br) = setup(&he_mono_b_xyz());

    let base = CksDispersionConfig {
        n_freq: 8,
        u0: 1.0,
        frozen_core: 0,
        ..Default::default()
    };

    let ch = CksDispersionConfig {
        coupling: CksCoupling::CoupledHartree,
        ..base.clone()
    };
    let ra_ch = build_monomer_response(&am, &ao, &ad, &ar, "", &ch).unwrap();
    let rb_ch = build_monomer_response(&bm, &bo, &bd, &br, "", &ch).unwrap();
    let e_ch = cks_dispersion_energy(&ra_ch, &rb_ch, &ch).unwrap().e_disp;

    // Build a CKS response, then zero its kernel. Adding a zero matrix to the
    // Dyson kernel must be a no-op, so this has to reproduce `CoupledHartree`
    // exactly — through the *same* solve path, which is the point: it proves
    // the two couplings differ only by the kernel and not by any other
    // incidental branch.
    let cks = CksDispersionConfig {
        coupling: CksCoupling::Cks,
        ..base
    };
    let mut ra = build_monomer_response(&am, &ao, &ad, &ar, "LDA", &cks).unwrap();
    let mut rb = build_monomer_response(&bm, &bo, &bd, &br, "LDA", &cks).unwrap();
    ra.zero_fxc_for_test();
    rb.zero_fxc_for_test();
    let e_cks = cks_dispersion_energy(&ra, &rb, &cks).unwrap().e_disp;

    assert_eq!(
        e_ch, e_cks,
        "CKS with a zeroed f_xc must equal CoupledHartree bit-for-bit \
         (only difference is adding a zero matrix); got {e_ch:.17e} vs {e_cks:.17e}"
    );
}

/// TEST 3 — coupling screens dispersion.
///
/// The whole premise of MP2C is that the uncoupled response *overestimates*
/// dispersion. If the coupled number were larger in magnitude, the correction
/// would have the wrong sign and MP2C would make overbinding worse.
#[test]
fn coupling_screens_dispersion() {
    let (am, ao, ad, ar) = setup(&he_mono_a_xyz());
    let (bm, bo, bd, br) = setup(&he_mono_b_xyz());

    let mk = |c: CksCoupling, xc: &str| {
        let cfg = CksDispersionConfig {
            coupling: c,
            n_freq: 16,
            u0: 1.0,
            frozen_core: 0,
        };
        let ra = build_monomer_response(&am, &ao, &ad, &ar, xc, &cfg).unwrap();
        let rb = build_monomer_response(&bm, &bo, &bd, &br, xc, &cfg).unwrap();
        cks_dispersion_energy(&ra, &rb, &cfg).unwrap().e_disp
    };

    let e_unc = mk(CksCoupling::Uncoupled, "");
    let e_ch = mk(CksCoupling::CoupledHartree, "");
    let e_cks = mk(CksCoupling::Cks, "LDA");

    assert!(
        e_unc < 0.0 && e_ch < 0.0 && e_cks < 0.0,
        "all dispersion energies must be attractive"
    );
    assert!(
        e_ch.abs() < e_unc.abs(),
        "Hartree coupling must screen (reduce) the dispersion: |{e_ch:.4e}| >= |{e_unc:.4e}|"
    );
    assert!(
        e_cks.abs() < e_unc.abs(),
        "CKS coupling must screen (reduce) the dispersion: |{e_cks:.4e}| >= |{e_unc:.4e}|"
    );

    // ORDERING — this is the assertion that caught the spin-factor bug.
    //
    // The XC kernel is attractive (f_xc < 0), so it *opposes* the repulsive
    // Hartree kernel and partially undoes its screening. The physical ordering
    // is therefore strict:
    //
    //     |E_Hartree-coupled|  <  |E_CKS|  <  |E_uncoupled|
    //
    // A kernel that is too large by any factor pushes E_CKS past E_uncoupled and
    // flips the sign of the whole MP2C correction. Merely asserting "all three
    // are negative" would not have noticed; asserting the ordering does.
    assert!(
        e_ch.abs() < e_cks.abs(),
        "CKS must lie between Hartree-coupled and uncoupled — the attractive f_xc \
         partially offsets Hartree screening, so |E_CKS| must EXCEED \
         |E_Hartree-coupled|. Got |{e_cks:.4e}| <= |{e_ch:.4e}|, which means the \
         XC kernel has the wrong sign or is being applied with the wrong weight."
    );
}

/// TEST 4 — the XC kernel must actually be load-bearing.
///
/// Added after mutation testing: seeding "silently ignore `fxc_aux`" was caught
/// only *incidentally*, by the bitwise null-functional test, and would not have
/// been caught at all had that mutation compiled cleanly. Nothing asserted that
/// a real `f_xc` changes the answer — so `Cks` could have degenerated into an
/// expensive alias for `CoupledHartree` without a single test noticing.
///
/// This closes that hole: with a real semilocal functional the projected kernel
/// is a genuinely non-zero matrix, so the CKS response must differ from the
/// Hartree-only one by well outside quadrature noise.
#[test]
fn xc_kernel_changes_the_answer() {
    let (am, ao, ad, ar) = setup(&he_mono_a_xyz());
    let (bm, bo, bd, br) = setup(&he_mono_b_xyz());

    let mk = |c: CksCoupling, xc: &str| {
        let cfg = CksDispersionConfig {
            coupling: c,
            n_freq: 16,
            u0: 1.0,
            frozen_core: 0,
        };
        let ra = build_monomer_response(&am, &ao, &ad, &ar, xc, &cfg).unwrap();
        let rb = build_monomer_response(&bm, &bo, &bd, &br, xc, &cfg).unwrap();
        cks_dispersion_energy(&ra, &rb, &cfg).unwrap().e_disp
    };

    let e_ch = mk(CksCoupling::CoupledHartree, "");
    let e_lda = mk(CksCoupling::Cks, "LDA");
    let e_xa = mk(CksCoupling::Cks, "LDA_X");

    // The XC kernel must move the number by far more than quadrature noise.
    let rel_lda = (e_lda - e_ch).abs() / e_ch.abs();
    assert!(
        rel_lda > 1e-3,
        "CKS with LDA is indistinguishable from Hartree-only coupling \
         ({e_lda:.8e} vs {e_ch:.8e}, rel {rel_lda:.2e}) — the f_xc kernel is not \
         reaching the Dyson solve"
    );

    // And two different functionals must give two different answers, or the
    // functional argument is being ignored somewhere upstream of the kernel.
    assert!(
        (e_lda - e_xa).abs() / e_lda.abs() > 1e-4,
        "LDA (X+C) and LDA_X (exchange only) give the same CKS dispersion \
         ({e_lda:.8e} vs {e_xa:.8e}) — the functional choice is not reaching f_xc"
    );
}

/// A mismatched auxiliary space must be a hard error, not a meaningless trace.
#[test]
fn mismatched_aux_space_errors() {
    // Monomer A in the DIMER basis vs monomer B in its OWN basis: different naux.
    let (am, ao, ad, ar) = setup(&he_mono_a_xyz());
    let (bm, bo, bd, br) = setup("1\nHe\nHe 0 0 0\n");

    let cfg = CksDispersionConfig {
        coupling: CksCoupling::Uncoupled,
        n_freq: 4,
        ..Default::default()
    };
    let ra = build_monomer_response(&am, &ao, &ad, &ar, "", &cfg).unwrap();
    let rb = build_monomer_response(&bm, &bo, &bd, &br, "", &cfg).unwrap();

    let err = cks_dispersion_energy(&ra, &rb, &cfg).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("different auxiliary spaces"),
        "expected a clean aux-space mismatch error, got: {msg}"
    );
}

/// GGA must be refused, not silently truncated to its rho-rho block.
///
/// This is the regression test for the bug `xc_kernel_changes_the_answer`
/// exposed: `xc_lda_fxc` on a GGA functional returns zeros, so a GGA request
/// used to produce a silently uncoupled (Hartree-only) number.
#[test]
fn gga_fxc_is_refused() {
    let (am, ao, ad, ar) = setup(&he_mono_a_xyz());
    let cfg = CksDispersionConfig {
        coupling: CksCoupling::Cks,
        n_freq: 4,
        ..Default::default()
    };
    let err = build_monomer_response(&am, &ao, &ad, &ar, "PBE", &cfg).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("GGA-family") && msg.contains("ALDA"),
        "expected a clean GGA refusal naming the ALDA limitation, got: {msg}"
    );
}

/// Meta-GGA has no f_xc in ferric — it must refuse, not silently drop the τ term.
#[test]
fn metagga_fxc_is_refused() {
    let (am, ao, ad, ar) = setup(&he_mono_a_xyz());
    let cfg = CksDispersionConfig {
        coupling: CksCoupling::Cks,
        n_freq: 4,
        ..Default::default()
    };
    let err = build_monomer_response(&am, &ao, &ad, &ar, "SCAN", &cfg).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("meta-GGA"),
        "expected a clean meta-GGA refusal, got: {msg}"
    );
}

/// TEST 5 — the cross-spin `f_αβ` term must actually be included.
///
/// Added after a mutation ESCAPED: deleting `f_αβ` from the kernel (keeping only
/// `f_αα`) passed all six preceding tests while shifting the CKS dispersion by
/// ~1.9%. Every structural test still held — the number stayed negative, stayed
/// ordered between the Hartree-coupled and uncoupled limits, and still responded
/// to the functional choice — because dropping a cross-spin term perturbs the
/// magnitude without breaking any *qualitative* property. Only a value check
/// catches it.
///
/// The discriminator used here is physical rather than a hardcoded constant:
/// LDA exchange has **no** cross-spin second derivative (`f_αβ^X = 0` exactly,
/// since `E_x^LDA` is a sum of independent per-spin terms), whereas the VWN
/// correlation in full `LDA` does. So `f_αβ` is by construction the *only*
/// source of difference between these two quantities' cross-spin content, and
/// the gap between them cannot be reproduced if `f_αβ` is dropped.
///
/// Pinned value: He₂ at 4 Å, cc-pVDZ/cc-pVDZ-RI, 16-point grid, ALDA.
/// Recomputing after an intentional physics change is expected; recomputing to
/// make a red test go green is not.
#[test]
fn cross_spin_fxc_term_is_included() {
    let (am, ao, ad, ar) = setup(&he_mono_a_xyz());
    let (bm, bo, bd, br) = setup(&he_mono_b_xyz());

    let mk = |xc: &str| {
        let cfg = CksDispersionConfig {
            coupling: CksCoupling::Cks,
            n_freq: 16,
            u0: 1.0,
            frozen_core: 0,
        };
        let ra = build_monomer_response(&am, &ao, &ad, &ar, xc, &cfg).unwrap();
        let rb = build_monomer_response(&bm, &bo, &bd, &br, xc, &cfg).unwrap();
        cks_dispersion_energy(&ra, &rb, &cfg).unwrap().e_disp
    };

    let e_lda = mk("LDA"); // X + C: has a non-zero f_ab
    let e_x = mk("LDA_X"); // X only: f_ab is identically zero

    // Reference values from the validated implementation.
    const E_LDA_REF: f64 = -8.7858055659e-7;
    const E_X_REF: f64 = -8.7205260677e-7;
    const TOL: f64 = 1e-11;

    assert!(
        (e_lda - E_LDA_REF).abs() < TOL,
        "ALDA (LDA X+C) CKS dispersion drifted: got {e_lda:.10e}, expected {E_LDA_REF:.10e}. \
         If the f_ab cross-spin term were dropped this lands near -8.6216e-7."
    );
    assert!(
        (e_x - E_X_REF).abs() < TOL,
        "ALDA (LDA_X) CKS dispersion drifted: got {e_x:.10e}, expected {E_X_REF:.10e}"
    );

    // And the two must differ: correlation contributes a cross-spin kernel that
    // exchange alone does not. Equality here means f_ab never entered.
    assert!(
        (e_lda - e_x).abs() / e_lda.abs() > 1e-3,
        "LDA and LDA_X give effectively the same CKS dispersion ({e_lda:.6e} vs {e_x:.6e}) — \
         the cross-spin f_ab term is not reaching the kernel"
    );
}

/// ANCHOR — fragment ordering. Which monomer is called "A" is a labelling
/// choice, so swapping the two must not change the dispersion energy (the
/// trace `Tr[Π_A Π_B]` is symmetric and each per-frequency term is the same
/// elementwise products summed in the same order, so this is bit-identical),
/// nor the full CP-corrected MP2C interaction energy (a re-associated sum of
/// the same terms, so equal to roundoff).
#[test]
fn swapping_the_monomers_leaves_the_energies_unchanged() {
    use ferric_rpa::mp2c_cks::{mp2c_cks_interaction, MonomerInput, Mp2cCksConfig};

    let (dm, do_, dd, dr) = setup(&he_dimer_xyz());
    let (am, ao, ad, ar) = setup(&he_mono_a_xyz());
    let (bm, bo, bd, br) = setup(&he_mono_b_xyz());

    let cfg = CksDispersionConfig {
        coupling: CksCoupling::Cks,
        n_freq: 8,
        u0: 1.0,
        frozen_core: 0,
    };
    let ra = build_monomer_response(&am, &ao, &ad, &ar, "LDA", &cfg).unwrap();
    let rb = build_monomer_response(&bm, &bo, &bd, &br, "LDA", &cfg).unwrap();
    let e_ab = cks_dispersion_energy(&ra, &rb, &cfg).unwrap().e_disp;
    let e_ba = cks_dispersion_energy(&rb, &ra, &cfg).unwrap().e_disp;
    assert_eq!(
        e_ab, e_ba,
        "E_disp(A,B) = {e_ab:.17e} but E_disp(B,A) = {e_ba:.17e}"
    );

    let mcfg = Mp2cCksConfig {
        xc: "LDA".to_string(),
        disp: CksDispersionConfig {
            n_freq: 8,
            u0: 1.0,
            frozen_core: 0,
            ..Default::default()
        },
        frozen_core: 0,
    };
    let a = MonomerInput {
        mol: &am,
        obs: &ao,
        dfbs: &ad,
        scf: &ar,
    };
    let b = MonomerInput {
        mol: &bm,
        obs: &bo,
        dfbs: &bd,
        scf: &br,
    };
    let ab = mp2c_cks_interaction(&dm, &do_, &dd, &dr, &a, &b, &mcfg).unwrap();
    let ba = mp2c_cks_interaction(&dm, &do_, &dd, &dr, &b, &a, &mcfg).unwrap();
    let d = (ab.e_int_mp2c - ba.e_int_mp2c).abs();
    eprintln!(
        "swap: e_int_mp2c {:.12e} vs {:.12e} (|diff| {d:.2e} Ha)",
        ab.e_int_mp2c, ba.e_int_mp2c
    );
    assert!(
        d < 1e-12,
        "MP2C interaction energy depends on monomer order: |diff| = {d:.3e} Ha"
    );
    assert_eq!(
        ab.e_correction, ba.e_correction,
        "the dispersion correction must be bit-identical"
    );
}

/// ANCHOR — the Dyson conditioning diagnostic in its exact limits.
///
/// With the Hartree kernel only (`K = I`), `M = I + Π⁰` and `Π⁰ ⪰ 0`, so
/// `min eig M ≥ 1` exactly (up to roundoff), and it approaches 1 as `ω → ∞`
/// where the bubble vanishes. `Uncoupled` has no Dyson step, so `None`.
/// A diagnostic that cannot reproduce these limits would say nothing about
/// near-singular nodes under the indefinite ALDA kernel.
#[test]
fn dyson_min_eigenvalue_reproduces_its_exact_limits() {
    let (am, ao, ad, ar) = setup(&he_mono_a_xyz());
    let cfg = CksDispersionConfig {
        coupling: CksCoupling::Cks,
        n_freq: 4,
        ..Default::default()
    };
    let r = build_monomer_response(&am, &ao, &ad, &ar, "LDA_X", &cfg).unwrap();

    assert!(r
        .dyson_min_eigenvalue(0.5, CksCoupling::Uncoupled)
        .unwrap()
        .is_none());
    for &w in &[0.0, 0.1, 1.0, 10.0] {
        let m = r
            .dyson_min_eigenvalue(w, CksCoupling::CoupledHartree)
            .unwrap()
            .unwrap();
        eprintln!("omega = {w:5.1}: min eig (I + Pi0) = {m:.15}");
        assert!(
            m > 1.0 - 1e-12,
            "Hartree-only Dyson matrix below 1 at omega = {w}: {m:.15e}"
        );
    }
    let far = r
        .dyson_min_eigenvalue(1e4, CksCoupling::CoupledHartree)
        .unwrap()
        .unwrap();
    assert!(
        (far - 1.0).abs() < 1e-6,
        "Pi0 must vanish as omega -> inf; min eig M = {far:.12e}"
    );
    // The ALDA kernel must actually change the spectrum relative to K = I.
    let cks = r
        .dyson_min_eigenvalue(0.0, CksCoupling::Cks)
        .unwrap()
        .unwrap();
    let ch = r
        .dyson_min_eigenvalue(0.0, CksCoupling::CoupledHartree)
        .unwrap()
        .unwrap();
    eprintln!("omega = 0: min eig M  CKS(ALDAX) = {cks:.10}, Hartree-only = {ch:.10}");
    assert!(
        (cks - ch).abs() > 1e-6,
        "the XC kernel does not reach the Dyson diagnostic"
    );
}
