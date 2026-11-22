//! Exactness anchors and cross-checks for omega-RI-MP2 and omega-RI-CDD-MP2.
//!
//! Written BEFORE any sweep (CLAUDE.md, Experimental Protocol). Each anchor
//! names the trivial limit where the approximation does nothing:
//!
//! * `omega = 0`: the omega-RI fit IS the Coulomb fit, so canonical omega-RI-MP2
//!   equals `ri_mp2` (to rounding of a different factorization, <= 1e-10 Eh);
//!   `omega = 1e-5` runs the erfc integrals and must still sit on it (the
//!   omega -> 0 error is O(omega^2)).
//! * untruncated Cholesky (`cdd_tol = 0`): the pseudo-MOs are an orthogonal
//!   rotation of the scaled canonical MOs, so omega-RI-CDD-MP2 equals the MO
//!   Laplace path (`LaplaceMp2::compute_mo`, an independent code path) at
//!   `omega = 0`, and equals the canonical-pseudo-orbital evaluation at
//!   `omega > 0`.
//! * zero Schwarz / pair thresholds: nothing is screened.
//!
//! The external cross-check is PySCF: `testdata/reference/validation/omega_ri/`
//! (scripts/validation/gen_omega_ri.py) assembles the omega-RI matrix
//! `Aᵀ V~⁻¹ V V~⁻¹ A` explicitly with numpy solves on PySCF erfc integrals. The
//! comparison is of the omega-RI SHIFT `E(omega) - E(Coulomb)` from each code's
//! own orbitals, which removes the SCF-convergence difference to first order.
//!
//! Artifact hypothesis (stated before measuring): if the erfc kernel did not
//! reach the 3-index integrals (or the metric), the shift would be 0 (or the
//! naive J^T V_w^-1 J value, orders of magnitude larger); if the dressing were
//! transposed, omega = 0 would still pass (L^-1 vs L^-T give the same Gram) but
//! omega > 0 would miss PySCF. Each is caught by a test below.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::laplace::{laplace_sos_mp2, LaplaceMp2, SosFormulation, SosMp2Config};
use ferric_mp2::omega_ri::RiFit;
use ferric_mp2::omega_ri_cdd::{
    omega_ri_cdd_mp2, CddEnergy, OmegaRiCddConfig, OmegaRiCddResult, PseudoOrbitals,
};
use ferric_mp2::rimp2::{ri_mp2_spin_components, ri_mp2_spin_components_fitted, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
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

fn setup(system: &str) -> Sys {
    let xyz = root()
        .join("testdata/molecules/validation")
        .join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz(xyz.to_str().unwrap()).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged);
    Sys {
        mol,
        obs,
        dfbs,
        rhf,
    }
}

fn canonical(s: &Sys, fit: RiFit) -> (f64, f64) {
    let (sc, _) = ri_mp2_spin_components_fitted(
        &s.mol,
        &s.obs,
        &s.dfbs,
        Operator::coulomb(),
        &s.rhf,
        &RiMp2Config::default(),
        &fit,
    )
    .unwrap();
    (sc.e_os, sc.e_ss)
}

fn cdd(s: &Sys, cfg: &OmegaRiCddConfig) -> OmegaRiCddResult {
    omega_ri_cdd_mp2(&s.mol, &s.obs, &s.dfbs, Operator::coulomb(), &s.rhf, cfg).unwrap()
}

fn json(system: &str) -> serde_json::Value {
    let p = root().join(format!(
        "testdata/reference/validation/omega_ri/{system}_cc-pvdz.json"
    ));
    serde_json::from_str(&std::fs::read_to_string(Path::new(&p)).unwrap()).unwrap()
}

#[test]
fn omega_zero_and_tiny_omega_equal_coulomb_ri_mp2() {
    let s = setup("h2o");
    let (sc, _) = ri_mp2_spin_components(
        &s.mol,
        &s.obs,
        &s.dfbs,
        Operator::coulomb(),
        &s.rhf,
        &RiMp2Config::default(),
    )
    .unwrap();
    let (os0, ss0) = canonical(&s, RiFit::omega_ri(0.0).unwrap());
    eprintln!(
        "omega=0: dOS {:.2e} dSS {:.2e}",
        os0 - sc.e_os,
        ss0 - sc.e_ss
    );
    assert!((os0 - sc.e_os).abs() < 1e-10 && (ss0 - sc.e_ss).abs() < 1e-10);
    let (os5, ss5) = canonical(&s, RiFit::omega_ri(1e-5).unwrap());
    let d = (os5 + ss5) - sc.e_total;
    eprintln!("omega=1e-5: dE {d:.2e}");
    assert!(d.abs() < 1e-9, "omega=1e-5 vs Coulomb RI-MP2: {d:e}");
}

#[test]
fn omega_ri_shift_matches_pyscf_explicit_assembly() {
    for system in ["h2o", "nh3"] {
        let s = setup(system);
        let r = json(system);
        let (os0, ss0) = canonical(&s, RiFit::Coulomb);
        for blk in r["omega_ri"].as_array().unwrap() {
            let w = blk["omega_bohr_inv"].as_f64().unwrap();
            let (os, ss) = canonical(&s, RiFit::omega_ri(w).unwrap());
            let shift = os + ss - os0 - ss0;
            let want = blk["mp2_minus_coulomb"].as_f64().unwrap();
            eprintln!(
                "{system} w={w}: ferric shift {shift:+.6e} pyscf {want:+.6e} |d| {:.1e}",
                (shift - want).abs()
            );
            assert!(
                (shift - want).abs() < 2e-9 + 1e-4 * want.abs(),
                "{system} w={w}: omega-RI shift {shift:e} vs PySCF {want:e}"
            );
        }
    }
}

#[test]
fn untruncated_cdd_equals_mo_laplace_at_omega_zero() {
    let s = setup("h2o");
    for n_quad in [5usize, 7] {
        let mut lap = LaplaceMp2::new(n_quad);
        let e_mo = lap
            .compute_mo(&s.mol, &s.obs, &s.dfbs, Operator::coulomb(), &s.rhf, 0)
            .unwrap();
        let r = cdd(
            &s,
            &OmegaRiCddConfig {
                n_quad,
                ..Default::default()
            },
        );
        eprintln!(
            "n_quad={n_quad}: CDD {:.12} MO-Laplace {e_mo:.12}",
            r.e_corr
        );
        assert!((r.e_corr - e_mo).abs() < 1e-10, "{} vs {e_mo}", r.e_corr);
        let sos = laplace_sos_mp2(
            &s.mol,
            &s.obs,
            &s.dfbs,
            Operator::coulomb(),
            &s.rhf,
            &SosMp2Config {
                c_os: 1.3,
                n_quad,
                ..Default::default()
            },
            SosFormulation::Mo,
        )
        .unwrap();
        let rs = cdd(
            &s,
            &OmegaRiCddConfig {
                n_quad,
                energy: CddEnergy::Sos { c_os: 1.3 },
                ..Default::default()
            },
        );
        assert!((rs.e_corr - sos.sos_corr).abs() < 1e-10);
    }
}

#[test]
fn untruncated_cdd_equals_canonical_pseudo_orbitals_at_omega() {
    let s = setup("h2o");
    let fit = RiFit::omega_ri(0.1).unwrap();
    let base = OmegaRiCddConfig {
        fit,
        ..Default::default()
    };
    let chol = cdd(&s, &base);
    let canon = cdd(
        &s,
        &OmegaRiCddConfig {
            orbitals: PseudoOrbitals::Canonical,
            ..base.clone()
        },
    );
    eprintln!("{chol}\n{canon}");
    let d_canon = chol.e_corr - canon.e_corr;
    eprintln!("Lowdin-Cholesky - canonical pseudo-MOs = {d_canon:.2e}");
    assert!(d_canon.abs() < 1e-12, "{d_canon:e}");
    // tol = 0 stops at roundoff: a pseudo-MO whose weight e^{t(ε-ε_F)} is
    // below m·ε_mach of the largest (the O 1s at large t) is dropped, with no
    // energy effect at all (the equality above is to 1e-10).
    assert!(chol.mean_rank_occ <= chol.nocc as f64 && chol.mean_rank_occ > 0.0);
    assert!(chol.mean_rank_vir <= chol.nvir as f64 && chol.mean_rank_vir > 0.0);
    // and to the canonical omega-RI-MP2 within the 7-point Laplace error
    let (os, ss) = canonical(&s, fit);
    let d = chol.e_corr - (os + ss);
    eprintln!("CDD(7) - canonical omega-RI-MP2 = {d:.2e}");
    assert!(d.abs() < 5e-7, "Laplace error {d:e}");
}

#[test]
fn screening_and_truncation_are_exact_at_zero_and_act_when_set() {
    let s = setup("h2o");
    let fit = RiFit::omega_ri(0.1).unwrap();
    let base = OmegaRiCddConfig {
        fit,
        ..Default::default()
    };
    let r0 = cdd(&s, &base);
    assert_eq!((r0.pair_fraction, r0.virtual_fraction), (1.0, 1.0));
    let rs = cdd(
        &s,
        &OmegaRiCddConfig {
            schwarz_thresh: 1e-9,
            pair_thresh: 1e-6,
            ..base.clone()
        },
    );
    let rt = cdd(
        &s,
        &OmegaRiCddConfig {
            orbitals: PseudoOrbitals::Cholesky { tol: 1e-6 },
            ..base.clone()
        },
    );
    eprintln!(
        "screened: dE {:.2e} pairs {:.3} virt {:.3}; cdd_tol 1e-6: dE {:.2e} rank_vir {:.1}/{}",
        rs.e_corr - r0.e_corr,
        rs.pair_fraction,
        rs.virtual_fraction,
        rt.e_corr - r0.e_corr,
        rt.mean_rank_vir,
        rt.nvir
    );
    assert!(
        rt.mean_rank_vir < rt.nvir as f64,
        "cdd_tol did not truncate"
    );
    assert!((rt.e_corr - r0.e_corr).abs() < 1e-5);
    assert!((rs.e_corr - r0.e_corr).abs() < 1e-6);
}
