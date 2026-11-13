//! The direct integral route of CASSCF against the dense `nbas^4` oracle.
//!
//! At one fixed orbital set (the RHF orbitals rotated by a random `κ` with
//! entries of order 0.05, so no block is canonical and every gradient class
//! is nonzero) the two routes give the energy, the orbital gradient, `F^I`
//! and `F^A`. The CAS-CI is the same code on both sides (36 and 400
//! determinants, solved densely to the rounding floor), so every difference
//! comes from the integrals.
//!
//! # Bars
//!
//! Integral level, a priori. `G = J − ½K` (AO) and `(pu|vw)`:
//!
//! * trivial limit (`thresh = 0`, no quartet skipped): only the summation
//!   order differs. A J/K element is a sum of `nbas²` terms bounded by
//!   `e_max |D_ls|` (`e_max = max Q²`, the Schwarz bound on every
//!   integral), so `δG <= 2 · 1.5 · nbas² ε e_max Σ|D|` (`γ_N ≈ N ε`, both
//!   routes) and `δF_pq <= c1² δG` in the MO basis; `(pu|vw)` is four chained
//!   sums of `nbas` terms, so `δP <= 2 · 4 nbas ε e_max c1 ca1³` (`c1`, `ca1`:
//!   largest column 1-norm of all and of the active orbitals);
//! * default threshold: a skipped J/K term is below `thresh` (screen
//!   `Q_bra Q_ket max|D| < thresh`), adding `1.5 nbas² thresh` to `δG`; a
//!   skipped `(pu|vw)` integral is below `thresh` (screen `Q_bra Q_ket <
//!   thresh`), adding `thresh c1 ca1³` to `δP`.
//!
//! Energy and gradient, propagated from the MEASURED integral differences
//! `δF` (max over `F^I`, `F^A`) and `δP`. In the orthonormal MO basis
//! `E_core − E_nuc = Σ_i (h + F^I)_ii`, `h_eff` is the active block of
//! `F^I`, and the CI root is variational, so
//!
//! ```text
//!   |ΔE| <= n_core δF + 2 n_a² δF + n_a⁴ δP + 4 nbas² ε |E|.
//! ```
//!
//! `g_pq = 2 (F_pq − F_qp)` with core columns `2 (F^I + F^A)` and active
//! columns `F^I γ + (pu|vw) Γ`, `|γ|, |Γ| <= 2`. The CI responds to the
//! perturbed Hamiltonian: `‖δH_CI‖ <= n_a² N δF + ½ n_a⁴ N² δP` (`N` active
//! electrons; one- and two-body excitation operators have norms at most `N`
//! and `N²`), `‖δc‖ <= ‖δH_CI‖ / gap` with the gap of the dense CI
//! Hamiltonian, and `|δγ| <= 2 N ‖δc‖`, `|δΓ| <= 2 N² ‖δc‖`. Then
//!
//! ```text
//!   |Δg| <= 4 (4 δF + 2 n_a δF + 2 n_a³ δP
//!              + n_a max|F^I| δγ + n_a³ max|(pu|vw)| δΓ).
//! ```
//!
//! # Orbital Hessian products
//!
//! Both routes build the fixed-RDM Hessian at the dense route's point (same
//! RDMs and Fock matrices) and multiply the same unit `x`, so only the
//! integrals differ: `(pq|jk)`, `(pj|qk)` over the occupied orbitals, whose
//! occupied kernel `L` gives the density responses and whose active blocks
//! give `W` (`src/casscf_newton.rs`). With `co1` the largest column 1-norm of
//! the occupied orbitals, each integral moves by
//! `δQ <= 2 · 4 nbas ε e_max c1² co1² + thresh c1² co1²`; `L` combines three
//! with weights 2, ½, ½ and multiplies `Y` with `Σ|Y^c| <= 2 n_c xs`,
//! `Σ|Y^a| <= 2 n_a² xs` (`xs` the largest column 1-norm of `X`,
//! `|γ| <= 2`), so `δG <= 3 δQ Σ|Y|`; `W` sums `n_a²` terms of each with
//! weights up to 4 and 8, `δW <= 12 n_a² δQ`; the row term `Σ_uvw (pu|vw) Γ`
//! moves by `4 n_a³ δQ`. Then
//!
//! ```text
//!   |δ(Hx)| <= 4 (2 (δG_c + δG_a) + 2 n_a δG_c
//!                 + 12 n_a² δQ ‖x‖₁ + 4 n_a³ δQ xs).
//! ```

mod casscf_common;

use casscf_common::{maxabs, molecule_text, setup_text, EPS};
use ferric_ci::casscf::{CasScfConfig, CasScfIntegrals, CasScfOptimizer, CasScfProblem};
use ndarray::Array2;

fn normals(n: usize, seed: u64) -> Vec<f64> {
    let mut st = seed;
    let mut uni = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (uni(), uni());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect()
}

struct Case {
    nbas: usize,
    eri_max: f64,
    c: Array2<f64>,
    dense: ferric_ci::casscf::CasScfPoint,
    direct: ferric_ci::casscf::CasScfPoint,
    e_dense_energy_only: f64,
    e_direct_energy_only: f64,
    paaa_dense: Vec<f64>,
    paaa_direct: Vec<f64>,
}

fn case(file: &str, n_core: usize, n_active: usize, nel: (usize, usize), thresh: f64) -> Case {
    let (mol, prep, rhf) = setup_text(&molecule_text(file), "6-31g");
    let nmo = rhf.mos_r().ncols();
    let cfg = |integrals| CasScfConfig {
        n_active,
        n_elec_active: nel,
        n_core,
        ci_conv_thresh: 1e-12,
        integrals,
        ..Default::default()
    };
    let dense = CasScfProblem::new(&mol, &prep, nmo, &cfg(CasScfIntegrals::Dense)).unwrap();
    let direct =
        CasScfProblem::new(&mol, &prep, nmo, &cfg(CasScfIntegrals::Direct { thresh })).unwrap();
    assert_eq!(direct.integrals(), CasScfIntegrals::Direct { thresh });
    let kappa: Vec<f64> = normals(dense.pairs().len(), 11)
        .iter()
        .map(|x| 0.05 * x)
        .collect();
    let c = dense.rotate(rhf.mos_r(), &kappa).unwrap();
    let q = ferric_scf::screening::SchwarzBounds::compute(
        ferric_integrals::operator::Operator::coulomb(),
        &prep,
    )
    .unwrap()
    .q;
    let qmax = maxabs(q.iter());
    // (pu|vw) is checked before any CI runs on it: a wrong (tu|vw) can make
    // the CI fail to converge, which would hide the integral error behind a
    // solver error.
    let paaa_dense = dense.paaa(&c).unwrap();
    let paaa_direct = direct.paaa(&c).unwrap();
    let dp = vdiff(&paaa_direct, &paaa_dense);
    let bar_p = paaa_bar(
        prep.nbasis() as f64,
        qmax * qmax,
        &c,
        n_core,
        n_active,
        thresh,
    );
    assert!(dp <= bar_p, "δP = {dp:e} > {bar_p:e}");
    Case {
        nbas: prep.nbasis(),
        eri_max: qmax * qmax,
        dense: dense.evaluate(&c).unwrap(),
        direct: direct.evaluate(&c).unwrap(),
        e_dense_energy_only: dense.energy(&c).unwrap(),
        e_direct_energy_only: direct.energy(&c).unwrap(),
        paaa_dense,
        paaa_direct,
        c,
    }
}

/// A-priori bar on `max |δ(pu|vw)|`: the rounding of the two routes' sums
/// plus the screened ket pairs, both carried through the four coefficient
/// contractions by the column 1-norms.
fn paaa_bar(
    n: f64,
    e_max: f64,
    c: &Array2<f64>,
    n_core: usize,
    n_active: usize,
    thresh: f64,
) -> f64 {
    let c1 = col1(c.view());
    let ca1 = col1(c.slice(ndarray::s![.., n_core..n_core + n_active]));
    2.0 * 4.0 * n * EPS * e_max * c1 * ca1.powi(3) + thresh * c1 * ca1.powi(3)
}

fn diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    maxabs((a - b).iter())
}

fn vdiff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

/// Largest column 1-norm of `c`.
fn col1(c: ndarray::ArrayView2<'_, f64>) -> f64 {
    c.columns()
        .into_iter()
        .map(|col| col.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// Lowest two eigenvalues' gap of the dense CI Hamiltonian of `pt`.
fn ci_gap(pt: &ferric_ci::casscf::CasScfPoint, nel: (usize, usize)) -> f64 {
    use ndarray_linalg::{EigValsh, UPLO};
    let n = pt.ci.integrals.n_active;
    let space = ferric_ci::hamiltonian::DeterminantSpace {
        alpha_strings: ferric_ci::strings::enumerate_strings(n, nel.0),
        beta_strings: ferric_ci::strings::enumerate_strings(n, nel.1),
    };
    let h = ferric_ci::hamiltonian::dense_hamiltonian(&pt.ci.integrals, &space, None).unwrap();
    let w = h.eigvalsh(UPLO::Lower).unwrap();
    w[1] - w[0]
}

fn check(file: &str, n_core: usize, n_active: usize, nel: (usize, usize), thresh: f64) {
    let r = case(file, n_core, n_active, nel, thresh);
    let n = r.nbas as f64;
    let na = n_active as f64;
    let ne = (nel.0 + nel.1) as f64;
    let e_max = r.eri_max;
    let c1 = col1(r.c.view());
    let ca1 = col1(r.c.slice(ndarray::s![.., n_core..n_core + n_active]));
    let sum_d = (2.0 * (n_core as f64) * c1 * c1).max(2.0 * na * ca1 * ca1);
    let dg_ao = 2.0 * 1.5 * n * n * EPS * e_max * sum_d + 1.5 * n * n * thresh;
    let bar_f = c1 * c1 * dg_ao;
    let bar_p = paaa_bar(n, e_max, &r.c, n_core, n_active, thresh);

    let df = diff(&r.direct.f_inactive, &r.dense.f_inactive)
        .max(diff(&r.direct.f_active, &r.dense.f_active));
    let dp = vdiff(&r.paaa_direct, &r.paaa_dense);
    let gap = ci_gap(&r.dense, nel);
    let bar_e = (n_core as f64) * df
        + 2.0 * na * na * df
        + na.powi(4) * dp
        + 4.0 * n * n * EPS * r.dense.e_total.abs();
    let dh_ci = na * na * ne * df + 0.5 * na.powi(4) * ne * ne * dp;
    let dc = dh_ci / gap;
    let (dgam, dgam2) = (2.0 * ne * dc, 2.0 * ne * ne * dc);
    let fmax = maxabs(r.dense.f_inactive.iter());
    let pmax = maxabs(r.paaa_dense.iter());
    let bar_g = 4.0
        * (4.0 * df
            + 2.0 * na * df
            + 2.0 * na.powi(3) * dp
            + na * fmax * dgam
            + na.powi(3) * pmax * dgam2);

    let de = (r.direct.e_total - r.dense.e_total).abs();
    let de2 = (r.e_direct_energy_only - r.e_dense_energy_only).abs();
    let dg = vdiff(&r.direct.gradient, &r.dense.gradient);
    eprintln!(
        "{file} CAS({},{n_active}) thresh={thresh:e}: δF {df:.2e} (bar {bar_f:.1e}) \
         δP {dp:.2e} (bar {bar_p:.1e}) |ΔE| {de:.2e} energy() {de2:.2e} (bar {bar_e:.1e}) \
         |Δg| {dg:.2e} (bar {bar_g:.1e}), CI gap {gap:.3}",
        nel.0 + nel.1
    );
    assert!(
        maxabs(r.dense.gradient.iter()) > 1e-2,
        "gradient is not exercised"
    );
    assert!(df <= bar_f, "δF = {df:e} > {bar_f:e}");
    assert!(de <= bar_e, "|ΔE| = {de:e} > {bar_e:e}");
    assert!(de2 <= bar_e, "|ΔE| (energy()) = {de2:e} > {bar_e:e}");
    assert!(dg <= bar_g, "|Δg| = {dg:e} > {bar_g:e}");
}

fn trivial_limit(file: &str, n_core: usize, n_active: usize, nel: (usize, usize)) {
    check(file, n_core, n_active, nel, 0.0);
}

fn default_threshold(file: &str, n_core: usize, n_active: usize, nel: (usize, usize)) {
    check(
        file,
        n_core,
        n_active,
        nel,
        ferric_ci::casscf_ints::DEFAULT_INTEGRAL_THRESH,
    );
}

#[test]
fn direct_matches_dense_in_the_trivial_limit_h2o_cas44() {
    trivial_limit("water.xyz", 3, 4, (2, 2));
}

#[test]
fn direct_matches_dense_in_the_trivial_limit_n2_cas66() {
    trivial_limit("n2.xyz", 4, 6, (3, 3));
}

#[test]
fn direct_matches_dense_in_the_trivial_limit_o2_triplet_cas86() {
    trivial_limit("o2.xyz", 4, 6, (5, 3));
}

#[test]
fn direct_matches_dense_at_the_default_threshold_h2o_cas44() {
    default_threshold("water.xyz", 3, 4, (2, 2));
}

#[test]
fn direct_matches_dense_at_the_default_threshold_n2_cas66() {
    default_threshold("n2.xyz", 4, 6, (3, 3));
}

fn hessian_check(file: &str, n_core: usize, n_active: usize, nel: (usize, usize), thresh: f64) {
    let (mol, prep, rhf) = setup_text(&molecule_text(file), "6-31g");
    let nmo = rhf.mos_r().ncols();
    let cfg = |integrals| CasScfConfig {
        n_active,
        n_elec_active: nel,
        n_core,
        ci_conv_thresh: 1e-12,
        integrals,
        optimizer: CasScfOptimizer::AugmentedHessian,
        ..Default::default()
    };
    let dense = CasScfProblem::new(&mol, &prep, nmo, &cfg(CasScfIntegrals::Dense)).unwrap();
    let direct =
        CasScfProblem::new(&mol, &prep, nmo, &cfg(CasScfIntegrals::Direct { thresh })).unwrap();
    let npair = dense.pairs().len();
    let kappa: Vec<f64> = normals(npair, 11).iter().map(|x| 0.05 * x).collect();
    let c = dense.rotate(rhf.mos_r(), &kappa).unwrap();
    let pt = dense.evaluate(&c).unwrap();
    let hd = dense.orbital_hessian(&c, &pt).unwrap();
    let hr = direct.orbital_hessian(&c, &pt).unwrap();
    let mut x = normals(npair, 17);
    let xn = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    x.iter_mut().for_each(|v| *v /= xn);
    let a = dense.hessian_vector(&hd, &x).unwrap();
    let b = direct.hessian_vector(&hr, &x).unwrap();

    let q = ferric_scf::screening::SchwarzBounds::compute(
        ferric_integrals::operator::Operator::coulomb(),
        &prep,
    )
    .unwrap()
    .q;
    let e_max = maxabs(q.iter()).powi(2);
    let n = prep.nbasis() as f64;
    let (nc, na) = (n_core as f64, n_active as f64);
    let c1 = col1(c.view());
    let co1 = col1(c.slice(ndarray::s![.., ..n_core + n_active]));
    let mut xm = Array2::<f64>::zeros((nmo, nmo));
    for (&(p, q), &v) in dense.pairs().iter().zip(&x) {
        xm[(p, q)] = v;
        xm[(q, p)] = -v;
    }
    let xs = col1(xm.view());
    let x1: f64 = x.iter().map(|v| v.abs()).sum();
    let dq = 2.0 * 4.0 * n * EPS * e_max * (c1 * co1).powi(2) + thresh * (c1 * co1).powi(2);
    let dg_c = 3.0 * dq * 2.0 * nc * xs;
    let dg_a = 3.0 * dq * 2.0 * na * na * xs;
    let bar = 4.0
        * (2.0 * (dg_c + dg_a)
            + 2.0 * na * dg_c
            + 12.0 * na * na * dq * x1
            + 4.0 * na.powi(3) * dq * xs);
    let d = vdiff(&a, &b);
    eprintln!(
        "{file} thresh={thresh:e}: max|Hx| {:.3}, |δ(Hx)| {d:.2e} (bar {bar:.1e})",
        maxabs(a.iter())
    );
    assert!(maxabs(a.iter()) > 1e-2, "the product is not exercised");
    assert!(d <= bar, "|δ(Hx)| = {d:e} > {bar:e}");
}

#[test]
fn direct_hessian_product_matches_dense_in_the_trivial_limit_h2o_cas44() {
    hessian_check("water.xyz", 3, 4, (2, 2), 0.0);
}

#[test]
fn direct_hessian_product_matches_dense_in_the_trivial_limit_o2_triplet_cas86() {
    hessian_check("o2.xyz", 4, 6, (5, 3), 0.0);
}

#[test]
fn direct_hessian_product_matches_dense_at_the_default_threshold_n2_cas66() {
    hessian_check(
        "n2.xyz",
        4,
        6,
        (3, 3),
        ferric_ci::casscf_ints::DEFAULT_INTEGRAL_THRESH,
    );
}

#[test]
fn a_negative_or_nan_integral_threshold_is_refused() {
    let (mol, prep, rhf) = setup_text(&molecule_text("water.xyz"), "sto-3g");
    for thresh in [-1e-12, f64::NAN, f64::INFINITY] {
        let cfg = CasScfConfig {
            n_active: 4,
            n_elec_active: (2, 2),
            n_core: 3,
            integrals: CasScfIntegrals::Direct { thresh },
            ..Default::default()
        };
        let e = CasScfProblem::new(&mol, &prep, rhf.mos_r().ncols(), &cfg)
            .err()
            .unwrap_or_else(|| panic!("thresh {thresh} accepted"));
        assert!(format!("{e}").contains("integral threshold"), "{e}");
    }
}
