//! Does `E_disp` vanish at infinite monomer separation?
//!
//! # The claim under test
//!
//! [`ferric_rpa::mp2c_cks`]'s module docs assert:
//!
//! > The dispersion terms here are **intrinsically interaction energies** — they
//! > come from a monomer-A × monomer-B trace and vanish identically at infinite
//! > separation, with no cancellation of large intramonomer pieces required.
//!
//! That is the structural argument for preferring this route over the
//! supermolecular substitute in `mp2c.rs`, and it is directly checkable: push the
//! two monomers apart and watch `E_disp`. A genuine SAPT dispersion energy decays
//! as `R^-6`. If instead it plateaus at a large constant, the trace is picking up
//! intramonomer response, `E_disp` is not an interaction energy, and every
//! absolute magnitude the module reports is meaningless — only the *difference*
//! `E_disp^CKS − E_disp^UCHF` could still mean anything, and only if the
//! intramonomer contamination cancelled between the two, which it need not.
//!
//! # Why this test was written, and what it found
//!
//! On the S22 methane and ethene dimers this machinery returned `E_disp^UCHF` of
//! **−270** and **−343 kcal/mol** — three orders of magnitude above any physical
//! dispersion energy for those complexes (~−0.5 and ~−1.6 kcal/mol) — while
//! reproducing the documented a24-12 value of −1.46 kcal/mol exactly. That
//! contrast was read as "the harness is fine, the systems differ". It was the
//! opposite.
//!
//! The scan below, run with the two monomers written in *opposite* center
//! orders, gave `E_disp` = −64.822203, −64.820461, −64.820459, −64.820458
//! kcal/mol at R = 3, 6, 12, 24 Å: flat to five significant figures. Run with a
//! **common** center order it gives −3.877e-3, −5.624e-5, −8.787e-7, −1.373e-8
//! kcal/mol — a factor of 64.0 per doubling, i.e. textbook `R^-6`.
//!
//! The defect was never intramonomer contamination of the trace. It was that
//! `Tr[Π_A Π_B]` pairs aux index `P` of A with aux index `P` of B, and the two
//! monomers' auxiliary bases were ordered by their own XYZ atom order. With the
//! blocks positionally reversed, A's on-A block multiplied B's on-B block and
//! the cross-trace degenerated into A's self-trace — a monomer property, hence
//! the perfectly flat baseline. `a24-12`'s CP fragment files happen to share an
//! atom ordering; `meth_m{A,B}_cp.xyz` and `eth_m{A,B}_cp.xyz` did not.
//!
//! `cks_dispersion_energy` now refuses a misaligned pair outright
//! (`aux_alignment_is_enforced` below pins that), so the original failure is no
//! longer reachable through the public API. This test keeps watching the
//! physics: that a correctly-aligned pair decays as `R^-6`.
//!
//! Uses He₂, the cheapest possible closed-shell dimer, so the scan is affordable
//! and so the `cks_dispersion_uncoupled.rs` He₂ validation is the direct
//! neighbour of this one.

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

const KCAL: f64 = 627.509_474_06;

/// Build one He monomer of a pair separated by `r` Å.
///
/// Both centers are always listed **z = 0 first, z = r second**; `real_is_first`
/// only chooses which of the two carries the nuclear charge. That common
/// ordering is what makes aux index `P` mean the same point in space for both
/// monomers, which is the precondition the Casimir-Polder trace needs.
fn he_monomer(
    r: f64,
    real_is_first: bool,
) -> (
    Molecule,
    PreparedBasis,
    PreparedBasis,
    ferric_scf::ScfResult,
) {
    let (a, b) = if real_is_first {
        ("He", "@He")
    } else {
        ("@He", "He")
    };
    let xyz = format!("2\nHe monomer + ghost\n{a} 0.0 0.0 0.0\n{b} 0.0 0.0 {r:.6}\n");
    let mol = Molecule::parse_xyz(&xyz, 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
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
    .unwrap();
    assert!(scf.converged);
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    (mol, obs, dfbs, scf)
}

#[test]
fn cks_dispersion_decays_with_separation() {
    let cfg_unc = CksDispersionConfig {
        coupling: CksCoupling::Uncoupled,
        n_freq: 16,
        u0: 1.0,
        frozen_core: 0,
    };

    eprintln!("He2/cc-pVDZ uncoupled Casimir-Polder dispersion vs separation");
    eprintln!(
        "{:>8}  {:>18}  {:>14}  {:>10}",
        "R (A)", "E_disp (kcal/mol)", "ratio to R=3", "per-doubling"
    );

    let mut results: Vec<(f64, f64)> = Vec::new();
    for &r in &[3.0_f64, 6.0, 12.0, 24.0] {
        let (ma, oa, da, sa) = he_monomer(r, true);
        let (mb, ob, db, sb) = he_monomer(r, false);
        let ra = build_monomer_response(&ma, &oa, &da, &sa, "LDA_X", &cfg_unc).unwrap();
        let rb = build_monomer_response(&mb, &ob, &db, &sb, "LDA_X", &cfg_unc).unwrap();
        let e = cks_dispersion_energy(&ra, &rb, &cfg_unc).unwrap().e_disp * KCAL;
        let first = if results.is_empty() { e } else { results[0].1 };
        let prev = results.last().map(|&(_, p)| (p / e).abs());
        eprintln!(
            "{r:8.1}  {e:18.8e}  {:14.4e}  {}",
            e / first,
            prev.map_or("     -".to_string(), |x| format!("{x:10.2}"))
        );
        assert!(
            e < 0.0,
            "E_disp must be attractive at every separation, got {e:.4e} at R = {r} A"
        );
        results.push((r, e));
    }

    // A genuine R^-6 dispersion energy falls by 2^6 = 64x from R to 2R.
    //
    // The overall 3 -> 24 A ratio is the headline check, kept at the original
    // deliberately-loose factor-of-100 floor so it fails only on a genuine
    // plateau. But a loose global bound alone would also pass on, say, an R^-2
    // decay, which is not dispersion either -- so each individual doubling is
    // additionally required to land near 64x. The window is wide (32-128, i.e.
    // an effective exponent between 5 and 7) because the shortest separation
    // still carries real overlap-driven (exchange-dispersion-like) contamination
    // that is not asymptotic; it is not wide enough to admit a different power.
    // Measured: 68.95x (3->6 A, effective exponent 6.11, the 3 A point still
    // carrying higher multipoles R^-8, R^-10 and overlap) then 64.00x and
    // 64.00x. The window is centred on the asymptotic 2^6 and spans one unit of
    // exponent each side: the nearest competing powers, R^-4 (16x) or R^-8
    // (256x), fall outside it, and the 3 A point's excess (+0.11 in exponent)
    // sits well inside.
    let e_near = results[0].1;
    let e_far = results[results.len() - 1].1;
    let decay = (e_near / e_far).abs();
    eprintln!("\n|E_disp(3 A)| / |E_disp(24 A)| = {decay:.4e}   (R^-6 predicts ~2.6e5)");
    assert!(
        decay > 100.0,
        "E_disp does NOT decay with separation (only {decay:.2}x over an 8x distance \
         increase, R^-6 predicts ~2.6e5x). The Casimir-Polder trace is therefore NOT an \
         interaction energy. E_disp(3 A) = {e_near:.4e}, E_disp(24 A) = {e_far:.4e} kcal/mol."
    );

    for w in results.windows(2) {
        let ((r0, e0), (r1, e1)) = (w[0], w[1]);
        let ratio = (e0 / e1).abs();
        assert!(
            (32.0..=128.0).contains(&ratio),
            "doubling {r0} A -> {r1} A changed |E_disp| by {ratio:.2}x; R^-6 requires ~64x \
             (accepted window 32-128, i.e. an effective exponent in [5, 7]). \
             E_disp = {e0:.6e} -> {e1:.6e} kcal/mol."
        );
    }
}

/// The misalignment that caused the original plateau must now be a hard error.
///
/// This is the regression test for the actual defect. Monomer B is written with
/// its real center FIRST -- the natural way to write a CP fragment, and what the
/// bundled `meth_m{A,B}_cp.xyz` did -- so its auxiliary basis is the positional
/// reverse of monomer A's. Both have identical `naux`, so the pre-existing
/// dimension check passes; only the center comparison catches it.
///
/// Without this refusal the pair silently returns ~-64.8 kcal/mol at every
/// separation instead of the correct -3.9e-3 kcal/mol at 3 A.
#[test]
fn aux_alignment_is_enforced() {
    let cfg = CksDispersionConfig {
        coupling: CksCoupling::Uncoupled,
        n_freq: 4,
        u0: 1.0,
        frozen_core: 0,
    };
    let r = 3.0_f64;

    // A: real at z=0, ghost at z=r.
    let (ma, oa, da, sa) = he_monomer(r, true);
    // B written the WRONG way round: real center listed first, so its aux block
    // order is reversed relative to A's.
    let xyz = format!("2\nHe B (reversed order)\nHe 0.0 0.0 {r:.6}\n@He 0.0 0.0 0.0\n");
    let mb = Molecule::parse_xyz(&xyz, 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let ob = PreparedBasis::new(&mb, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &ob).unwrap();
    let sb = solve_rhf(
        &ParallelContext::default(),
        &mb,
        &ob,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    let db = PreparedBasis::new(&mb, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();

    let ra = build_monomer_response(&ma, &oa, &da, &sa, "", &cfg).unwrap();
    let rb = build_monomer_response(&mb, &ob, &db, &sb, "", &cfg).unwrap();

    // naux matches -- the old dimension check would have let this through.
    assert_eq!(ra.naux(), rb.naux());

    let err = cks_dispersion_energy(&ra, &rb, &cfg).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("not aligned"),
        "expected a clean aux-alignment refusal, got: {msg}"
    );
}
