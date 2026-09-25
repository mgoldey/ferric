//! `PdepRpaResult::eigenpotentials` are PHYSICAL aux-basis coefficients.
//!
//! Exactness anchor, independent of any external reference: with ferric's own
//! aux metric V = (P|op|Q) (`threeindex::coulomb_metric_2c`) and its own
//! physical-aux static polarizability Π = R diag(4/Δε) Rᵀ (R = raw (Q|ia)
//! from `eri3_mo_ov_blocked`, no metric factor anywhere), the exported
//! columns c_α must satisfy
//!
//! ```text
//!   cᵀ V c       = I
//!   cᵀ (V + Π) c = diag(λ_α(0))      (λ = eigenvalues_static)
//! ```
//!
//! i.e. c are the generalized eigenvectors of (V + Π) c = λ V c, which is the
//! basis-independent definition of the static dielectric eigenpotentials. The
//! checks never touch `v_inv_sqrt`, so they cannot be satisfied by a
//! factor-dependent storage convention.
//!
//! Both factorizations `metric_inverse_sqrt` can return are covered:
//! * Coulomb → lower-triangular Cholesky L⁻¹ (NOT symmetric);
//! * erf(ω)  → symmetric eigh V^{-1/2} with near-null modes dropped (for the
//!   kept modes, λ > 1, the eigenvector lies in the factor's range so the
//!   identities still hold exactly).
//! and the unrestricted path (`run_u_pdep_rpa`, per-spin prefactor 2).
//!
//! Artifact hypothesis: if the export were c = F·u (the dressed vector pushed
//! through the factor rather than its transpose) the Coulomb case gives
//! cᵀVc − I of order 1e3 (measured on water/cc-pVDZ from the CLI NPZ before
//! the fix), while the erf case would still pass (F symmetric), so the erf
//! case pins the factorization generality, not the transpose.
//!
//! MUTATION (documented, run by hand): reverting `.t()` in
//! `run_pdep_rpa_eigensolve` (`inter.v_inv_sqrt.t().dot(..)` →
//! `inter.v_inv_sqrt.dot(..)`) fails `coulomb_cholesky_*`; reverting it at the
//! `run_u_pdep_rpa` site fails `unrestricted_*`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ferric_mp2::rimp2::{eri3_budget_bytes, eri3_mo_ov_blocked};
use ferric_rpa::{run_pdep_rpa, run_u_pdep_rpa, PdepRpaConfig, PdepRpaResult};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::{Array2, Axis};

const TOL: f64 = 1e-9;

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        eigensolver_conv_thresh: 1e-10,
        need_eigenvalues_freq: false,
        ..Default::default()
    }
}

/// Raw (un-dressed) (Q|ia) for one spin channel, flattened to (naux, nocc*nvir)
/// in ferric's i*nvir+a order, plus the matching Δε_ia.
fn raw_ov(
    op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    mos: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
) -> (Array2<f64>, Vec<f64>) {
    let nmo = mos.ncols();
    let nvir = nmo - nocc;
    let c_occ = mos.slice(ndarray::s![.., ..nocc]).to_owned();
    let c_vir = mos.slice(ndarray::s![.., nocc..]).to_owned();
    let r = eri3_mo_ov_blocked(op, obs, dfbs, &c_occ, &c_vir, eri3_budget_bytes(None)).unwrap();
    let naux = r.shape()[0];
    let r = r.into_shape_with_order((naux, nocc * nvir)).unwrap();
    let mut de = Vec::with_capacity(nocc * nvir);
    for i in 0..nocc {
        for a in 0..nvir {
            de.push(eps[nocc + a] - eps[i]);
        }
    }
    (r, de)
}

/// Π += R diag(pref/Δε) Rᵀ.
fn add_pi(pi: &mut Array2<f64>, r: &Array2<f64>, de: &[f64], pref: f64) {
    let mut rs = r.clone();
    for (ia, mut col) in rs.axis_iter_mut(Axis(1)).enumerate() {
        col.mapv_inplace(|x| x * pref / de[ia]);
    }
    *pi += &rs.dot(&r.t());
}

/// max|cᵀVc − I| and max|cᵀ(V+Π)c − diag(λ)| (the latter relative to max λ).
fn check(label: &str, res: &PdepRpaResult, v: &Array2<f64>, pi: &Array2<f64>) {
    let c = &res.eigenpotentials;
    let m = res.n_eigenpotentials;
    assert_eq!(
        c.nrows(),
        v.nrows(),
        "{label}: eigenpotentials rows != naux"
    );
    assert_eq!(
        c.ncols(),
        m,
        "{label}: eigenpotentials cols != n_eigenpotentials"
    );
    assert!(
        m >= 2,
        "{label}: need >= 2 modes for an off-diagonal check, got {m}"
    );
    let gram = c.t().dot(&v.dot(c));
    let vp = v + pi;
    let h = c.t().dot(&vp.dot(c));
    let lam = &res.eigenvalues_static;
    let lam_max = lam.iter().fold(1.0_f64, |a, &b| a.max(b.abs()));
    let mut err_g = 0.0_f64;
    let mut err_h = 0.0_f64;
    for a in 0..m {
        for b in 0..m {
            let id = if a == b { 1.0 } else { 0.0 };
            err_g = err_g.max((gram[(a, b)] - id).abs());
            err_h = err_h.max((h[(a, b)] - id * lam[a]).abs());
        }
    }
    eprintln!(
        "{label}: M={m} naux={} max|cᵀVc-I|={err_g:.3e} max|cᵀ(V+Π)c-diag(λ)|={err_h:.3e} \
         λ[0..3]={:?}",
        v.nrows(),
        &lam[..3.min(m)]
    );
    assert!(
        err_g <= TOL,
        "{label}: cᵀVc deviates from I by {err_g:.3e} (> {TOL:.0e})"
    );
    assert!(
        err_h <= TOL * lam_max,
        "{label}: cᵀ(V+Π)c deviates from diag(λ) by {err_h:.3e} (> {:.1e})",
        TOL * lam_max
    );
}

fn water() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap()
}

fn closed_shell_case(label: &str, obs_name: &str, op: Operator) {
    let ctx = ParallelContext::default();
    let mol = water();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    // Full-Coulomb RHF reference; `op` only enters the RI/RPA kernel.
    let coul = Operator::coulomb();
    let bounds = SchwarzBounds::compute(coul, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, coul, &bounds, &RhfConfig::default()).unwrap();

    let res = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &pdep_cfg()).unwrap();
    assert!(
        res.eigensolver_converged,
        "{label}: eigensolver not converged"
    );

    let v = coulomb_metric_2c(op, &dfbs).unwrap();
    let nocc = mol.nelec() as usize / 2;
    let (r, de) = raw_ov(op, &obs, &dfbs, rhf.mos_r(), rhf.eps_r(), nocc);
    let mut pi = Array2::<f64>::zeros(v.raw_dim());
    add_pi(&mut pi, &r, &de, 4.0);
    check(label, &res, &v, &pi);
}

#[test]
fn coulomb_cholesky_eigenpotentials_are_v_orthonormal_and_diagonalize_v_plus_pi() {
    closed_shell_case("H2O/cc-pVDZ Coulomb", "cc-pvdz", Operator::coulomb());
}

#[test]
fn erf_symmetric_eigenpotentials_are_v_orthonormal_and_diagonalize_v_plus_pi() {
    // ω = 0.2 Bohr⁻¹: the eigh factor drops 28 of 84 near-null metric modes
    // here (PySCF numpy mirror of the same construction: identities hold to
    // 4e-12), so this also exercises the dropped-mode branch.
    closed_shell_case("H2O/STO-3G erf(0.2)", "sto-3g", Operator::erf(0.2));
}

#[test]
fn unrestricted_eigenpotentials_are_v_orthonormal_and_diagonalize_v_plus_pi() {
    let ctx = ParallelContext::default();
    // OH radical (doublet), Bohr-free xyz in Angstrom.
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &RhfConfig::default()).unwrap();

    let res = run_u_pdep_rpa(&mol, &obs, &dfbs, op, &uhf, &pdep_cfg()).unwrap();
    assert!(res.eigensolver_converged, "U: eigensolver not converged");

    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity as usize - 1;
    let nocc_a = (nelec + two_s) / 2;
    let nocc_b = (nelec - two_s) / 2;
    let v = coulomb_metric_2c(op, &dfbs).unwrap();
    let mut pi = Array2::<f64>::zeros(v.raw_dim());
    let (ra, dea) = raw_ov(op, &obs, &dfbs, &uhf.mos_alpha, &uhf.eps_alpha, nocc_a);
    add_pi(&mut pi, &ra, &dea, 2.0);
    let mos_b = uhf.mos_beta.as_ref().expect("UHF has beta MOs");
    let eps_b = uhf.eps_beta.as_ref().expect("UHF has beta energies");
    let (rb, deb) = raw_ov(op, &obs, &dfbs, mos_b, eps_b, nocc_b);
    add_pi(&mut pi, &rb, &deb, 2.0);
    check("OH/STO-3G UHF Coulomb", &res, &v, &pi);
}
