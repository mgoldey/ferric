//! `response_modes`: the dielectric eigenmodes are physical aux densities.
//!
//! Three anchors, each independent of the dressed-basis storage convention:
//!
//! 1. Generalized eigen-equation with ferric's own raw integrals. With
//!    `V = (P|Q)` (`coulomb_metric_2c`) and `Π(iω) = R diag(4Δ/(Δ²+ω²)) Rᵀ`
//!    from the raw `R = (Q|ia)` (`eri3_mo_ov_blocked`, no metric factor), the
//!    columns must satisfy `cᵀVc = I` and `cᵀ(V+Π)c = diag(λ)`. That is the
//!    statement "ρ_α = Σ c_P χ_P is a density whose Hartree potential induces
//!    the Coulomb-fitted density −(λ−1)ρ_α".
//!    Artifact hypothesis: a back-transform `c = F·u` (Cholesky `F = L⁻¹` is
//!    not symmetric) or `c = u` breaks `cᵀVc = I` by O(1), while a correct
//!    transform leaves only round-off.
//!    Bar: both sides are O(1) numbers (`cᵀVc` entries ≤ 1, `cᵀ(V+Π)c`
//!    entries ≤ λ_max ≈ 2.2), so a wrong transform fails by O(1); round-off
//!    of forming them over naux = 84 (one triangular solve, two GEMMs) is
//!    measured at 4e-13 (`cᵀVc`) and 1.1e-12 (`cᵀ(V+Π)c`). The asserted bar,
//!    1e-9 (relative to λ_max), sits three decades above round-off and nine
//!    below the failure.
//! 2. Full-rank polarizability. `α_bare − Σ_α s_α s_αᵀ / λ_α` over every mode
//!    is algebraically the RPA polarizability; at ω = 0 it must equal
//!    `pdep_polarizability_static`, which solves `ε̃ y = w` by Cholesky and
//!    never diagonalizes. Bar: two factorizations of the same SPD matrix with
//!    κ(ε̃) = λ_max/λ_min ≈ 2.2 agree to ~naux·ε·κ ≈ 4e-14 relative (measured
//!    max|Δα| 4.4e-15 a.u. on |α| ≈ 4); asserted 1e-10 relative to |α|.
//! 3. The leading static modes are the PDEP eigenpotentials of
//!    `run_pdep_rpa` (Lanczos at conv 1e-10): eigenvalues agree to 1e-8 (a
//!    residual-norm-1e-10 Ritz value is accurate to the residual squared over
//!    the gap, well below that; 1e-8 also covers its residual itself), and
//!    each column agrees up to sign: `|c_aᵀ V c_b| ≥ 1 − 1e-8`.
//!
//! MUTATIONS (run by hand, each fails a test below):
//! * `inter.v_inv_sqrt.t().dot(&u)` → `inter.v_inv_sqrt.dot(&u)` in
//!   `response_modes`: anchors 1 and 3 fail (cᵀVc off I by O(1)).
//! * dielectric weight `4.0 * d_w` → `2.0 * d_w`: anchors 1, 2, 3 fail.
//! * field couplings `* 4.0` → `* 2.0`: anchor 2 fails.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ferric_mp2::rimp2::{eri3_budget_bytes, eri3_mo_ov_blocked};
use ferric_rpa::properties::pdep_polarizability_static;
use ferric_rpa::response_modes::response_modes;
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array2, Axis};

struct Case {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

fn water_ccpvdz() -> Case {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    Case {
        mol,
        obs,
        dfbs,
        rhf,
    }
}

/// Π(iω) = R diag(4Δ/(Δ²+ω²)) Rᵀ from raw (Q|ia), all occupied orbitals.
fn raw_pi(case: &Case, omega: f64) -> Array2<f64> {
    let nocc = case.mol.nelec() as usize / 2;
    let mos = case.rhf.mos_r();
    let eps = case.rhf.eps_r();
    let nvir = mos.ncols() - nocc;
    let c_occ = mos.slice(ndarray::s![.., ..nocc]).to_owned();
    let c_vir = mos.slice(ndarray::s![.., nocc..]).to_owned();
    let r = eri3_mo_ov_blocked(
        Operator::coulomb(),
        &case.obs,
        &case.dfbs,
        &c_occ,
        &c_vir,
        eri3_budget_bytes(None),
    )
    .unwrap();
    let naux = r.shape()[0];
    let r = r.into_shape_with_order((naux, nocc * nvir)).unwrap();
    let mut rs = r.clone();
    for (ia, mut col) in rs.axis_iter_mut(Axis(1)).enumerate() {
        let (i, a) = (ia / nvir, ia % nvir);
        let de = eps[nocc + a] - eps[i];
        col.mapv_inplace(|x| x * 4.0 * de / (de * de + omega * omega));
    }
    rs.dot(&r.t())
}

fn assert_generalized_eigenvectors(case: &Case, omega: f64) {
    let naux = case.dfbs.nbasis();
    let modes = response_modes(
        &case.mol,
        &case.obs,
        &case.dfbs,
        &case.rhf,
        Operator::coulomb(),
        naux,
        omega,
        None,
    )
    .unwrap();
    assert_eq!(modes.coefficients.dim(), (naux, naux));
    let v = coulomb_metric_2c(Operator::coulomb(), &case.dfbs).unwrap();
    let pi = raw_pi(case, omega);
    let c = &modes.coefficients;
    let gram = c.t().dot(&v.dot(c));
    let h = c.t().dot(&(&v + &pi).dot(c));
    let lam = &modes.eigenvalues;
    let lam_max = lam[0];
    let (mut err_g, mut err_h) = (0.0_f64, 0.0_f64);
    for a in 0..naux {
        for b in 0..naux {
            let id = if a == b { 1.0 } else { 0.0 };
            err_g = err_g.max((gram[(a, b)] - id).abs());
            err_h = err_h.max((h[(a, b)] - id * lam[a]).abs());
        }
    }
    eprintln!(
        "omega={omega}: naux={naux} max|cᵀVc-I|={err_g:.3e} \
         max|cᵀ(V+Π)c-diag λ|={err_h:.3e} λ[0..3]={:?} λ_min={:.6}",
        &lam[..3],
        lam[naux - 1]
    );
    assert!(lam.windows(2).all(|w| w[0] >= w[1]), "λ not descending");
    assert!(lam[naux - 1] >= 1.0 - 1e-12, "λ_min below 1");
    assert!(err_g <= 1e-9, "cᵀVc off I by {err_g:.3e}");
    assert!(
        err_h <= 1e-9 * lam_max,
        "cᵀ(V+Π)c off diag λ by {err_h:.3e}"
    );
}

#[test]
fn modes_are_v_orthonormal_generalized_eigenvectors_static() {
    assert_generalized_eigenvectors(&water_ccpvdz(), 0.0);
}

#[test]
fn modes_are_v_orthonormal_generalized_eigenvectors_imaginary_frequency() {
    assert_generalized_eigenvectors(&water_ccpvdz(), 0.5);
}

#[test]
fn full_rank_modes_reconstruct_the_static_polarizability() {
    let case = water_ccpvdz();
    let naux = case.dfbs.nbasis();
    let modes = response_modes(
        &case.mol,
        &case.obs,
        &case.dfbs,
        &case.rhf,
        Operator::coulomb(),
        naux,
        0.0,
        None,
    )
    .unwrap();
    let from_modes = modes.polarizability_from_modes();
    let cfg = PdepRpaConfig::default();
    let reference = pdep_polarizability_static(
        &case.mol,
        &case.obs,
        &case.dfbs,
        &case.rhf,
        Operator::coulomb(),
        &cfg,
    )
    .unwrap()
    .tensor;
    let scale = (0..3).map(|d| reference[d][d].abs()).fold(0.0, f64::max);
    let mut err = 0.0_f64;
    for d in 0..3 {
        for e in 0..3 {
            err = err.max((from_modes[d][e] - reference[d][e]).abs());
        }
    }
    // Leading-mode truncation, reported for the record (not asserted).
    let lead: f64 = (0..8)
        .map(|k| {
            (0..3)
                .map(|d| modes.field_couplings[(k, d)].powi(2) / modes.eigenvalues[k])
                .sum::<f64>()
        })
        .sum();
    let all: f64 = (0..naux)
        .map(|k| {
            (0..3)
                .map(|d| modes.field_couplings[(k, d)].powi(2) / modes.eigenvalues[k])
                .sum::<f64>()
        })
        .sum();
    eprintln!(
        "alpha diag modes {:?} ref {:?} bare {:?} max|Δ|={err:.3e}; \
         leading 8 modes carry {:.4} of the screening trace",
        [from_modes[0][0], from_modes[1][1], from_modes[2][2]],
        [reference[0][0], reference[1][1], reference[2][2]],
        [
            modes.alpha_bare[0][0],
            modes.alpha_bare[1][1],
            modes.alpha_bare[2][2]
        ],
        lead / all
    );
    assert!(
        err <= 1e-10 * scale,
        "mode-sum α differs from pdep_polarizability_static by {err:.3e}"
    );
}

#[test]
fn leading_static_modes_are_the_pdep_eigenpotentials() {
    let case = water_ccpvdz();
    let m = 8;
    let modes = response_modes(
        &case.mol,
        &case.obs,
        &case.dfbs,
        &case.rhf,
        Operator::coulomb(),
        m,
        0.0,
        None,
    )
    .unwrap();
    let cfg = PdepRpaConfig {
        frozen_core: 0,
        eigensolver_conv_thresh: 1e-10,
        need_eigenvalues_freq: false,
        ..Default::default()
    };
    let pdep = run_pdep_rpa(
        &case.mol,
        &case.obs,
        &case.dfbs,
        Operator::coulomb(),
        &case.rhf,
        &cfg,
    )
    .unwrap();
    assert!(pdep.eigensolver_converged);
    assert!(pdep.n_eigenpotentials >= m);
    let v = coulomb_metric_2c(Operator::coulomb(), &case.dfbs).unwrap();
    let cross = modes
        .coefficients
        .t()
        .dot(&v.dot(&pdep.eigenpotentials.slice(ndarray::s![.., ..m])));
    let mut worst_lam = 0.0_f64;
    let mut worst_overlap = 0.0_f64;
    for k in 0..m {
        worst_lam = worst_lam.max((modes.eigenvalues[k] - pdep.eigenvalues_static[k]).abs());
        worst_overlap = worst_overlap.max(1.0 - cross[(k, k)].abs());
    }
    eprintln!("max|Δλ|={worst_lam:.3e} max(1-|c·Vc'|)={worst_overlap:.3e}");
    assert!(worst_lam <= 1e-8, "eigenvalues differ by {worst_lam:.3e}");
    assert!(
        worst_overlap <= 1e-8,
        "mode vectors differ: 1-|overlap| = {worst_overlap:.3e}"
    );
}
