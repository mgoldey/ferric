//! Exactness anchors and the PySCF cross-check for omega-RI RPA
//! (`run_pdep_rpa_fitted` / `run_u_pdep_rpa_fitted`).
//!
//! * `omega = 0`: the omega-RI fit is the Coulomb fit, so the energy equals
//!   `run_pdep_rpa` (<= 1e-10 Eh) and `omega = 1e-5` sits on it.
//! * the omega-RI shift `E(omega) - E(Coulomb)` equals PySCF + numpy on the
//!   explicit matrix `Aᵀ V~⁻¹ V V~⁻¹ A` (scripts/validation/gen_omega_ri.py),
//!   on the same 40-point Gauss-Legendre grid (u0 = 0.5).
//! * open shell: a closed-shell molecule run through UHF must give the
//!   closed-shell omega-RI RPA energy (the spin-resolved fitted builder is an
//!   independent code path from the restricted one).
//!
//! Artifact hypothesis: if the dressing reached only the metric (or only the
//! 3-index integrals) the shift would be orders of magnitude off PySCF; if the
//! open-shell builder ignored `fit`, the UHF value would equal the Coulomb one.

use std::path::PathBuf;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::omega_ri::RiFit;
use ferric_rpa::config::{Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, run_pdep_rpa_fitted, run_u_pdep_rpa_fitted};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf()
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

fn scf_cfg() -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    }
}

fn setup(system: &str, unrestricted: bool) -> Sys {
    let xyz = root()
        .join("testdata/molecules/validation")
        .join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz(xyz.to_str().unwrap()).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = if unrestricted {
        solve_uhf(&ctx, &mol, &obs, &bounds, &scf_cfg()).unwrap()
    } else {
        solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg()).unwrap()
    };
    assert!(rhf.converged);
    Sys {
        mol,
        obs,
        dfbs,
        rhf,
    }
}

fn cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        trunc_thresh: 0.0,
        eigensolver: Eigensolver::Lanczos,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 40,
            u0: 0.5,
        },
        ..Default::default()
    }
}

fn e(s: &Sys, fit: RiFit) -> f64 {
    run_pdep_rpa_fitted(&s.mol, &s.obs, &s.dfbs, Operator::coulomb(), &s.rhf, &cfg(), &fit)
        .unwrap()
        .e_rpa
}

#[test]
fn omega_zero_and_tiny_omega_equal_coulomb_rpa() {
    let s = setup("h2o", false);
    let e_c = run_pdep_rpa(&s.mol, &s.obs, &s.dfbs, Operator::coulomb(), &s.rhf, &cfg())
        .unwrap()
        .e_rpa;
    let d0 = e(&s, RiFit::omega_ri(0.0).unwrap()) - e_c;
    let d5 = e(&s, RiFit::omega_ri(1e-5).unwrap()) - e_c;
    eprintln!("omega=0: {d0:.2e}  omega=1e-5: {d5:.2e}");
    assert!(d0.abs() < 1e-10, "{d0:e}");
    assert!(d5.abs() < 1e-9, "{d5:e}");
}

#[test]
fn omega_ri_rpa_shift_matches_pyscf_explicit_assembly() {
    for system in ["h2o", "nh3"] {
        let s = setup(system, false);
        let p = root().join(format!(
            "testdata/reference/validation/omega_ri/{system}_cc-pvdz.json"
        ));
        let r: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let e0 = e(&s, RiFit::Coulomb);
        for blk in r["omega_ri"].as_array().unwrap() {
            let w = blk["omega_bohr_inv"].as_f64().unwrap();
            let shift = e(&s, RiFit::omega_ri(w).unwrap()) - e0;
            let want = blk["rpa_minus_coulomb"].as_f64().unwrap();
            eprintln!(
                "{system} w={w}: ferric {shift:+.6e} pyscf {want:+.6e} |d| {:.1e}",
                (shift - want).abs()
            );
            assert!(
                (shift - want).abs() < 2e-9 + 1e-4 * want.abs(),
                "{system} w={w}: {shift:e} vs {want:e}"
            );
        }
    }
}

#[test]
fn open_shell_fitted_path_equals_closed_shell_for_a_closed_shell_molecule() {
    let r = setup("h2o", false);
    let u = setup("h2o", true);
    let fit = RiFit::omega_ri(0.1).unwrap();
    let e_r = e(&r, fit);
    let e_u = run_u_pdep_rpa_fitted(&u.mol, &u.obs, &u.dfbs, Operator::coulomb(), &u.rhf, &cfg(), &fit)
        .unwrap()
        .e_rpa;
    let e_uc = run_u_pdep_rpa_fitted(
        &u.mol,
        &u.obs,
        &u.dfbs,
        Operator::coulomb(),
        &u.rhf,
        &cfg(),
        &RiFit::Coulomb,
    )
    .unwrap()
    .e_rpa;
    eprintln!(
        "R {e_r:.12} U {e_u:.12} |d| {:.1e}; U shift {:.2e}",
        (e_r - e_u).abs(),
        e_u - e_uc
    );
    assert!((e_r - e_u).abs() < 1e-8, "{e_r} vs {e_u}");
    assert!((e_u - e_uc).abs() > 1e-7, "open-shell path ignored the fit");
}
