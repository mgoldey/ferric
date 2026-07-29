//! EXACTNESS ANCHOR for the TDA-DFT spike.
//!
//! With the f_xc coupling term ZEROED and c_HF = 1, the TDA-DFT A-matrix
//!
//! ```text
//!   A_{ia,jb} = δ(ε_a − ε_i) + 2(ia|jb) − c_HF (ij|ab) + 2(ia|f_xc|jb)
//! ```
//!
//! collapses term-for-term onto the CIS/TDHF-TDA matrix that ferric already
//! has in `bse::run_cis_tda`. This test pins that reduction.
//!
//! It is the load-bearing check that separates the two things that can be
//! wrong in this spike:
//!   * the (ia)-space A-matrix ASSEMBLY (indexing, the 2v − K convention, the
//!     frozen-core window, the RI contraction) — covered here;
//!   * the AO→(ia) f_xc KERNEL ADAPTER — covered by the PySCF comparison in
//!     `tda_dft_pyscf.rs`.
//!
//! If this fails, no amount of kernel work fixes it. **No excitation energy
//! from this module should be quoted until this passes.**
//!
//! Both drivers build the same RI tensor from the same reference in the same
//! order, so agreement is expected at ROUND-OFF, not merely "close" — the
//! assertions below are at 1e-12 Ha absolute on every one of the ~n
//! eigenvalues, with a separate bit-identity report for diagnosis.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::bse::run_cis_tda;
use ferric_gw::tddft::{run_tda_dft, TdaDftConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const WATER: &str = "3\nwater\nO 0.0000 0.0000 0.1173\nH 0.0000 0.7572 -0.4692\nH 0.0000 -0.7572 -0.4692\n";

/// Shared setup: converged RHF on `xyz` in `obs_name`, plus the RI aux basis.
struct Ref {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    op: Operator,
    rhf: ferric_scf::ScfResult,
}

fn build_rhf(xyz: &str, obs_name: &str, aux_name: &str) -> Ref {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-9,
        ..Default::default()
    };
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged, "reference RHF did not converge");
    Ref { mol, obs, dfbs, op, rhf }
}

/// Core anchor, parameterized over basis so one failure mode (a basis-size- or
/// parallel-threshold-dependent indexing bug) cannot hide behind a single case.
fn anchor_at(obs_name: &str, aux_name: &str, frozen_core: usize) {
    let r = build_rhf(WATER, obs_name, aux_name);

    let cis = run_cis_tda(&r.mol, &r.obs, &r.dfbs, r.op, &r.rhf, frozen_core).unwrap();

    // xc_name = None  ⇒  c_HF = 1.0 and NO f_xc kernel is constructed.
    let cfg = TdaDftConfig { frozen_core, ..Default::default() };
    let tda = run_tda_dft(&r.mol, &r.obs, &r.dfbs, r.op, &r.rhf, None, &cfg).unwrap();

    assert_eq!(tda.c_hf, 1.0, "xc=None must give c_HF = 1 (pure CIS)");
    assert!(!tda.fxc_included, "xc=None must not build an f_xc kernel");
    assert_eq!(tda.nocc, cis.nocc);
    assert_eq!(tda.nvir, cis.nvir);
    assert_eq!(tda.omega.len(), cis.omega.len());
    assert!(
        tda.nocc * tda.nvir >= 8,
        "test must exercise the parallel row-fill branch (n >= 8); got n = {}",
        tda.nocc * tda.nvir
    );

    let mut max_diff = 0.0_f64;
    let mut n_bit_identical = 0usize;
    for (k, (a, b)) in tda.omega.iter().zip(cis.omega.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
        if a.to_bits() == b.to_bits() {
            n_bit_identical += 1;
        }
        assert!(
            d < 1e-12,
            "{obs_name} fc={frozen_core}: TDA-DFT(no fxc, c_HF=1) eigenvalue {k} differs \
             from CIS-TDA by {d:e} Ha (want < 1e-12): {a:.17e} vs {b:.17e}"
        );
    }
    eprintln!(
        "ANCHOR {obs_name} fc={frozen_core}: n={} states, max |Omega_TDA-DFT - Omega_CIS| = {max_diff:e} Ha, \
         {n_bit_identical}/{} bit-identical",
        tda.omega.len(),
        tda.omega.len()
    );

    // Oscillator strengths must agree too — they are built from the
    // eigenVECTORS, so this catches an eigenvector-ordering divergence that
    // eigenvalue-only agreement would miss.
    let mut max_f_diff = 0.0_f64;
    for (a, b) in tda
        .oscillator_strength
        .iter()
        .zip(cis.oscillator_strength.iter())
    {
        max_f_diff = max_f_diff.max((a - b).abs());
    }
    assert!(
        max_f_diff < 1e-10,
        "{obs_name} fc={frozen_core}: oscillator strengths differ from CIS by {max_f_diff:e}"
    );
    eprintln!("ANCHOR {obs_name} fc={frozen_core}: max |f_TDA-DFT - f_CIS| = {max_f_diff:e}");
}

#[test]
fn tda_dft_without_fxc_reduces_exactly_to_cis_sto3g() {
    anchor_at("sto-3g", "cc-pvdz-ri", 0);
}

#[test]
fn tda_dft_without_fxc_reduces_exactly_to_cis_ccpvdz() {
    anchor_at("cc-pvdz", "cc-pvdz-ri", 0);
}

/// The frozen-core window must be applied identically in both drivers. A
/// frozen-core off-by-one is exactly the sort of bug that a single fc=0 test
/// cannot see.
#[test]
fn tda_dft_without_fxc_reduces_exactly_to_cis_frozen_core() {
    anchor_at("cc-pvdz", "cc-pvdz-ri", 1);
}

/// MUTATION TEST for the anchor above.
///
/// An exactness anchor that has never been seen to FAIL is an assumption, not
/// evidence. This deliberately perturbs the one knob the anchor holds fixed
/// (c_HF) and asserts the anchor's own tolerance REJECTS the result. If this
/// test ever passes trivially (i.e. the perturbed run still matches CIS to
/// 1e-12), the anchor is vacuous and the other tests in this file mean nothing.
#[test]
fn cis_anchor_is_not_vacuous_c_hf_perturbation_is_detected() {
    let r = build_rhf(WATER, "sto-3g", "cc-pvdz-ri");
    let cis = run_cis_tda(&r.mol, &r.obs, &r.dfbs, r.op, &r.rhf, 0).unwrap();

    // Same path, but c_HF deliberately wrong by 1%.
    let cfg = TdaDftConfig {
        c_hf_override: Some(0.99),
        ..Default::default()
    };
    let perturbed = run_tda_dft(&r.mol, &r.obs, &r.dfbs, r.op, &r.rhf, None, &cfg).unwrap();
    assert_eq!(perturbed.c_hf, 0.99, "c_hf_override must be honoured");

    let max_diff = perturbed
        .omega
        .iter()
        .zip(cis.omega.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("MUTATION: 1% c_HF perturbation moves eigenvalues by {max_diff:e} Ha");
    assert!(
        max_diff > 1e-12,
        "MUTATION TEST FAILED: a 1% change in c_HF did not move any eigenvalue above the \
         anchor's 1e-12 tolerance (max diff {max_diff:e}). The anchor cannot distinguish a \
         correct A-matrix from a wrong one and is therefore worthless."
    );
    // Sanity on the magnitude: 1% of the exchange term should be a
    // milli-Hartree-scale shift on water/STO-3G, not a round-off tickle.
    assert!(
        max_diff > 1e-5,
        "a 1% c_HF change moved eigenvalues only {max_diff:e} Ha — suspiciously small; is the \
         exchange term reaching the matrix at all?"
    );
}

/// The f_xc term must actually CHANGE the answer. Pairs with the anchor: the
/// anchor proves `include_fxc = false` reproduces CIS, this proves
/// `include_fxc = true` is not silently a no-op (which would ALSO reproduce
/// CIS and would ALSO pass the anchor).
#[test]
fn fxc_term_is_not_a_no_op() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    // An LDA reference, so the f_xc kernel is the LDA one.
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-9,
        xc: Some("LDA".to_string()),
        ..Default::default()
    };
    let ks = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(ks.converged, "reference RKS/LDA did not converge");

    let with = run_tda_dft(
        &mol,
        &obs,
        &dfbs,
        op,
        &ks,
        Some("LDA"),
        &TdaDftConfig { include_fxc: true, ..Default::default() },
    )
    .unwrap();
    let without = run_tda_dft(
        &mol,
        &obs,
        &dfbs,
        op,
        &ks,
        Some("LDA"),
        &TdaDftConfig { include_fxc: false, ..Default::default() },
    )
    .unwrap();

    assert!(with.fxc_included && !without.fxc_included);
    assert_eq!(with.c_hf, 0.0, "LDA is a pure functional: c_HF must be 0");
    let max_diff = with
        .omega
        .iter()
        .zip(without.omega.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("f_xc term shifts eigenvalues by up to {max_diff:e} Ha");
    assert!(
        max_diff > 1e-4,
        "the f_xc coupling term changed the eigenvalues by only {max_diff:e} Ha — it is \
         effectively a no-op, so the anchor would pass even with a dead kernel adapter"
    );
}

/// A pure functional must report c_HF = 0 and a plain hybrid its true mixing
/// fraction. A silently-wrong c_HF is invisible in the anchor (which overrides
/// it) but poisons every real excitation energy.
#[test]
fn exact_exchange_fraction_is_read_from_the_functional() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ctx,
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig { energy_conv: 1e-10, density_conv: 1e-9, ..Default::default() },
    )
    .unwrap();

    // Only the c_HF resolution is under test, so skip the (expensive) kernel.
    let no_kernel = TdaDftConfig { include_fxc: false, ..Default::default() };
    for (name, want) in [("LDA", 0.0), ("PBE", 0.0), ("B3LYP", 0.2)] {
        let res = run_tda_dft(&mol, &obs, &dfbs, op, &rhf, Some(name), &no_kernel).unwrap();
        assert!(
            (res.c_hf - want).abs() < 1e-12,
            "{name}: c_HF = {} (want {want})",
            res.c_hf
        );
        eprintln!("c_HF({name}) = {}", res.c_hf);
    }
}

/// Unsupported functional classes must ERROR, never silently drop their
/// missing kernel term. (Config-honesty convention: no silent fallbacks.)
#[test]
fn unsupported_functionals_are_rejected_not_silently_approximated() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ctx,
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig { energy_conv: 1e-10, density_conv: 1e-9, ..Default::default() },
    )
    .unwrap();
    let cfg = TdaDftConfig::default();

    // wB97X-V carries BOTH VV10 and range-separation; it is rejected by the
    // VV10 guard, which fires first. Keep it as a case, but assert the
    // specific reason so the two guards cannot be confused.
    let err = run_tda_dft(&mol, &obs, &dfbs, op, &rhf, Some("wB97X-V"), &cfg)
        .expect_err("wB97X-V must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("VV10"), "wB97X-V should hit the VV10 guard: {msg}");
    eprintln!("VV10 rejected: {msg}");

    // A range-separated hybrid WITHOUT VV10, so the range-separation guard is
    // the one under test rather than being shadowed by the VV10 check above.
    // (CAM-B3LYP has no nonlocal correlation term.)
    let err = run_tda_dft(&mol, &obs, &dfbs, op, &rhf, Some("HYB_GGA_XC_CAM_B3LYP"), &cfg)
        .expect_err("RSH must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("range-separated"),
        "CAM-B3LYP should hit the range-separation guard, got: {msg}"
    );
    eprintln!("RSH rejected: {msg}");

    // Meta-GGA: no tau f_xc kernel.
    let err = run_tda_dft(&mol, &obs, &dfbs, op, &rhf, Some("SCAN"), &cfg)
        .expect_err("meta-GGA must be rejected");
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("meta-gga") || msg.contains("mgga"),
        "unexpected meta-GGA rejection message: {msg}"
    );
    eprintln!("meta-GGA rejected: {msg}");
}

/// An open-shell reference must be rejected outright (singlet closed-shell
/// spike only).
#[test]
fn open_shell_reference_is_rejected() {
    let mol = Molecule::parse_xyz("2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let uhf = ferric_scf::uhf::solve_uhf(
        &ctx,
        &mol,
        &obs,
        &bounds,
        &ferric_scf::uhf::UhfConfig { energy_conv: 1e-9, density_conv: 1e-7, ..Default::default() },
    )
    .unwrap();
    let err = run_tda_dft(&mol, &obs, &dfbs, op, &uhf, None, &TdaDftConfig::default())
        .expect_err("open-shell reference must be rejected");
    assert!(err.to_string().contains("closed-shell"), "unexpected: {err}");
}
