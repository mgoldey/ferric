//! Closed-shell COHSEX on a Kohn–Sham reference: the static shift Σ_x − v_xc
//! (`run_gw(.., vxc_diag)` with `GwMethod::Cohsex`). Cheap (H2O/6-31G PBE,
//! 8-point quadrature that COHSEX never reads), so NOT ignored.
//!
//! # Exactness anchor (independent construction)
//!
//! The unrestricted sibling `u_cohsex.rs` assembles the same QP energy per
//! spin from its own code: its own two-spin B̃ build, its own spin-summed
//! dielectric (`run_u_pdep_rpa`, ε̃ = I + Π_α + Π_β), and its own shift line.
//! Feeding it the converged RKS result re-expressed as UKS (α = β: same MOs,
//! orbital energies and Fock, D_σ = ½D) with `vxc_diag = (v, v)` must give the
//! closed-shell COHSEX@PBE QP energy on every window state, in both spin
//! channels. MEASURED 2026-10-02 (H2O/6-31G, PBE): max |Δε_qp| = 1.1e-15 Ha
//! (rounding: the two paths sum Π and the B̃ build in different orders); bar
//! `TOL_ANCHOR` = 1e-14.
//!
//! The shift itself is far from zero (min |Σx − v_xc| over the window
//! 0.178 Ha = 4.8 eV, measured), so the anchor cannot pass vacuously: a
//! closed-shell path that drops the shift (the defect this file pins) misses
//! by ≥ 0.178 Ha, > 1e13× the bar.
//!
//! # `vxc_diag = None` is today's HF path, bit for bit
//!
//! With no shift, ε_qp must be exactly ε_mf + (ΔΣ_SEX + Σ_COH) in floating
//! point — the pre-fix formula — so the HF reference cannot have moved.
//!
//! MUTATIONS (run by hand 2026-10-02; each makes
//! `cohsex_at_ks_matches_u_cohsex_on_the_same_orbitals` fail):
//! (a) `cohsex.rs` shift `sx − v` → `sx + v` (v_xc sign flipped);
//! (b) `sx − v` → `sx` (Σ_x added without v_xc);
//! (c) shift applied only when `m_loc < mo_b.n_occ_act` (occupied MOs only);
//! (d) `lib.rs` `GwMethod::Cohsex` arm passes `None` (the pre-fix dispatch).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::vxc_mo::vxc_diagonal_mo;
use ferric_gw::{run_gw, run_u_gw, GwConfig, GwMethod, GwResult, UGwResult};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::{ScfResult, Spin};
use ndarray::Array1;

/// Closed-shell vs unrestricted COHSEX on identical orbitals. Measured
/// 1.1e-15 Ha (see the module doc).
const TOL_ANCHOR: f64 = 1e-14;
/// The anchor is only meaningful if the shift it tests is large.
const MIN_SHIFT: f64 = 1e-2; // measured min 0.178 Ha

const H2O: &str = "3
H2O
O  0.0   0.0       0.117790
H  0.0   0.755453 -0.471161
H  0.0  -0.755453 -0.471161
";

struct Case {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rks: ScfResult,
    vxc: Array1<f64>,
}

fn h2o_pbe() -> Case {
    let mol = Molecule::parse_xyz(H2O, 0, 1).unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let cfg = RhfConfig {
        xc: Some("pbe".into()),
        max_iter: 200,
        energy_conv: 1e-10,
        density_conv: 1e-9,
        ..Default::default()
    };
    let rks = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &cfg,
    )
    .unwrap();
    assert!(rks.converged, "H2O/6-31G RKS/PBE did not converge");
    let (vxc, _) = vxc_diagonal_mo(&mol, &obs_bs, "pbe", &rks).unwrap();
    Case {
        mol,
        obs,
        dfbs,
        rks,
        vxc,
    }
}

/// The converged RKS result re-expressed as an unrestricted one with α = β.
fn as_unrestricted(r: &ScfResult) -> ScfResult {
    let mut u = r.clone();
    u.spin = Spin::Unrestricted;
    u.density_alpha = 0.5 * r.density_r();
    u.density_beta = Some(0.5 * r.density_r());
    u.mos_beta = Some(r.mos_alpha.clone());
    u.eps_beta = Some(r.eps_alpha.clone());
    u.fock_beta = Some(r.fock_alpha.clone());
    u
}

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        trunc_thresh: 0.0,
        eigensolver: Eigensolver::Lanczos,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 8,
            u0: 0.5,
        },
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        ..Default::default()
    }
}

fn gw_cfg() -> GwConfig {
    GwConfig {
        method: GwMethod::Cohsex,
        ..Default::default()
    }
}

fn closed(c: &Case, vxc: Option<&Array1<f64>>) -> GwResult {
    run_gw(
        &c.mol,
        &c.obs,
        &c.dfbs,
        Operator::coulomb(),
        &c.rks,
        &pdep_cfg(),
        &gw_cfg(),
        vxc,
    )
    .unwrap()
}

fn open(c: &Case) -> UGwResult {
    run_u_gw(
        &c.mol,
        &c.obs,
        &c.dfbs,
        Operator::coulomb(),
        &as_unrestricted(&c.rks),
        &pdep_cfg(),
        &gw_cfg(),
        Some((&c.vxc, &c.vxc)),
    )
    .unwrap()
}

#[test]
fn cohsex_at_ks_matches_u_cohsex_on_the_same_orbitals() {
    let c = h2o_pbe();
    let cs = closed(&c, Some(&c.vxc));
    let us = open(&c);
    assert_eq!(cs.mo_indices, us.mo_indices, "QP window");

    let mut worst = 0.0_f64;
    let mut min_shift = f64::INFINITY;
    for (k, &p) in cs.mo_indices.iter().enumerate() {
        let shift = cs.sigma_x[k] - c.vxc[p];
        min_shift = min_shift.min(shift.abs());
        for (spin, eqp, sx, sc) in [
            ("alpha", &us.eps_qp_a, &us.sigma_x_a, &us.sigma_c_a),
            ("beta", &us.eps_qp_b, &us.sigma_x_b, &us.sigma_c_b),
        ] {
            let d = (cs.eps_qp[k] - eqp[k]).abs();
            worst = worst.max(d);
            eprintln!(
                "MO {p} {spin}: closed {:.12} open {:.12} |d| {d:.2e} shift {shift:+.6} Ha \
                 ({:+.4} eV)",
                cs.eps_qp[k],
                eqp[k],
                shift * 27.211_386_245_988
            );
            assert!(
                d < TOL_ANCHOR,
                "MO {p} {spin}: closed-shell COHSEX@PBE {:.12} vs U-COHSEX {:.12}, |d| {d:.2e} Ha \
                 > {TOL_ANCHOR:.0e} — the closed-shell KS shift is wrong or missing",
                cs.eps_qp[k],
                eqp[k]
            );
            assert!(
                (cs.sigma_x[k] - sx[k]).abs() < TOL_ANCHOR,
                "MO {p} {spin}: Σx differs ({:.12} vs {:.12})",
                cs.sigma_x[k],
                sx[k]
            );
            assert!(
                (cs.sigma_c[k] - sc[k]).abs() < TOL_ANCHOR,
                "MO {p} {spin}: ΔΣ_SEX + Σ_COH differs ({:.12} vs {:.12})",
                cs.sigma_c[k],
                sc[k]
            );
        }
    }
    eprintln!("anchor: worst |d| {worst:.2e} Ha; min |Σx − v_xc| {min_shift:.3e} Ha");
    assert!(
        min_shift > MIN_SHIFT,
        "the KS shift is too small ({min_shift:.2e} Ha) for this anchor to test anything"
    );
}

#[test]
fn cohsex_without_vxc_is_the_unshifted_formula_bit_for_bit() {
    let c = h2o_pbe();
    let none = closed(&c, None);
    for (k, &p) in none.mo_indices.iter().enumerate() {
        let want = none.eps_mf[k] + none.sigma_c[k];
        assert_eq!(
            none.eps_qp[k].to_bits(),
            want.to_bits(),
            "MO {p}: vxc_diag = None must give ε_mf + Σc exactly: {:.17e} vs {want:.17e}",
            none.eps_qp[k]
        );
    }
    // And the KS shift really is applied when given: same Σx/Σc, ε_qp moved
    // by exactly Σx − v_xc.
    let some = closed(&c, Some(&c.vxc));
    for (k, &p) in some.mo_indices.iter().enumerate() {
        assert_eq!(some.sigma_c[k].to_bits(), none.sigma_c[k].to_bits());
        assert_eq!(some.sigma_x[k].to_bits(), none.sigma_x[k].to_bits());
        let moved = some.eps_qp[k] - none.eps_qp[k];
        let shift = some.sigma_x[k] - c.vxc[p];
        assert!(
            (moved - shift).abs() < TOL_ANCHOR,
            "MO {p}: ε_qp moved by {moved:.12} Ha, expected Σx − v_xc = {shift:.12}"
        );
    }
}

#[test]
fn vxc_diag_of_the_wrong_length_is_an_error_not_a_panic() {
    let c = h2o_pbe();
    let short = Array1::<f64>::zeros(c.vxc.len() - 1);
    let err = run_gw(
        &c.mol,
        &c.obs,
        &c.dfbs,
        Operator::coulomb(),
        &c.rks,
        &pdep_cfg(),
        &gw_cfg(),
        Some(&short),
    )
    .expect_err("a vxc_diag shorter than the MO count must be refused")
    .to_string();
    assert!(err.contains("vxc_diag"), "must name the argument: {err}");
}
