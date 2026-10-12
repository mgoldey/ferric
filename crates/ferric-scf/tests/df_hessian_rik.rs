//! Analytic RHF Hessian for RI-K (`ferric_scf::hessian_df_k`) against
//! INDEPENDENT constructions.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test df_hessian_rik
//! ```
//!
//! # What is compared to what
//!
//! * `K^x` (the AO matrices of the first-order exchange Fock) vs a central
//!   difference of the SCF's own `DfK` matrix at displaced geometries (fixed
//!   AO density).
//! * Each skeleton piece vs a central difference of the matching piece of the
//!   first derivative at FIXED density: the 3c piece (`2α⟨Γ₀, L_x⟩` at the
//!   displaced geometry), the 2c piece (`−α⟨Γ₀, V_x Γ₀⟩`), the cross piece
//!   (`2α⟨Z_x, ∂_yΓ⟩`), and the whole K skeleton vs a central difference of
//!   `df_jk_gradient` (the validated DF gradient, RI-K channel).
//! * The TOTAL Hessian vs a central difference (h and h/2, Richardson) of
//!   ferric's own analytic DF gradient with the SCF re-converged at every
//!   displaced geometry.
//! * Translational and rotational invariance.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * Right: every piece matches its finite difference at the FD floor; the
//!   exact-K and RI-J-only Hessians miss the RI-K reference by far more than
//!   the bar.
//! * Skeleton wrong (dropped term, wrong weight): the piece tests and the full
//!   test fail together while `K^x` stays green.
//! * `K^x` wrong (missing metric/aux-centre part, transposed pair): the piece
//!   tests stay green, the `K^x` test and the full test fail.
//! * FD reference broken (displaced SCF on another state, a metric mode
//!   crossing the DF-K cut inside the step): the full test fails column-wise
//!   while every analytic-piece test is green; the systems here have a clean
//!   spectral gap (no metric mode between the cut and the gradient's noise
//!   floor), checked by `clean_gap`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ferric_scf::df_gradient::{df_jk_gradient, DfKChannel};
use ferric_scf::df_k::DfK;
use ferric_scf::fock::KBuilder;
use ferric_scf::gradient::rhf_gradient;
use ferric_scf::hessian::rij_support::{rik_first, rik_terms, RikMutants};
use ferric_scf::hessian::{rhf_hessian, rhf_hessian_parts, RhfHessianParts};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{s, Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

const WATER_DISTORTED: &str = "3\nH2O distorted off C2v\n\
O      0.010000    -0.020000     0.120000\n\
H      0.030000     0.770000    -0.460000\n\
H     -0.020000    -0.740000    -0.490000\n";

fn has_deriv2() -> bool {
    let order = ferric_integrals::engine::libint_max_deriv_order();
    if order < 2 {
        eprintln!("SKIP: libint2 generated with derivative order {order}; no analytic Hessian");
    }
    order >= 2
}

fn has_deriv1() -> bool {
    let order = ferric_integrals::engine::libint_max_deriv_order();
    if order < 1 {
        eprintln!("SKIP: libint2 generated without first derivatives");
    }
    order >= 1
}

fn water() -> Molecule {
    Molecule::parse_xyz(WATER_DISTORTED, 0, 1).unwrap()
}

/// SCF with RI-K in `k_aux`; RI-J in `j_aux` ("" = exact J).
fn scf_config(j_aux: &str, k_aux: &str) -> RhfConfig {
    RhfConfig {
        max_iter: 200,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        df_j_aux: Some(j_aux.to_string()),
        df_k_aux: Some(k_aux.to_string()),
        ..Default::default()
    }
}

struct Setup {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    rhf: ScfResult,
    cfg: RhfConfig,
    basis: String,
    aux: String,
}

fn setup(mol: &Molecule, basis_name: &str, j_aux: &str, k_aux: &str) -> Setup {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = scf_config(j_aux, k_aux);
    let rhf = solve_rhf(&ParallelContext::default(), mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged, "{basis_name}/{k_aux}: SCF did not converge");
    let route = rhf.df_jk.as_ref().expect("a DF route");
    assert_eq!(route.k_aux.as_deref(), Some(k_aux), "RI-K must be on");
    assert_eq!(
        route.j_aux.as_deref(),
        Some(j_aux).filter(|j| !j.is_empty()),
        "RI-J route"
    );
    Setup {
        mol: mol.clone(),
        prep,
        bounds,
        rhf,
        cfg,
        basis: basis_name.to_string(),
        aux: k_aux.to_string(),
    }
}

fn parts(s: &Setup) -> RhfHessianParts {
    rhf_hessian_parts(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &s.rhf,
        &s.cfg,
    )
    .unwrap()
}

fn displaced(mol: &Molecule, atom: usize, coord: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match coord {
        0 => m.atoms[atom].x += h,
        1 => m.atoms[atom].y += h,
        _ => m.atoms[atom].zpos += h,
    }
    m
}

fn fd_hessian(
    mol: &Molecule,
    h: f64,
    mut grad: impl FnMut(&Molecule) -> Array2<f64>,
) -> Array2<f64> {
    let n3 = 3 * mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n3, n3));
    for b in 0..n3 {
        let gp = grad(&displaced(mol, b / 3, b % 3, h));
        let gm = grad(&displaced(mol, b / 3, b % 3, -h));
        for a in 0..n3 {
            out[(a, b)] = (gp[(a / 3, a % 3)] - gm[(a / 3, a % 3)]) / (2.0 * h);
        }
    }
    out
}

fn fd_hessian_richardson(
    mol: &Molecule,
    h: f64,
    mut grad: impl FnMut(&Molecule) -> Array2<f64>,
) -> Array2<f64> {
    let a = fd_hessian(mol, h, &mut grad);
    let b = fd_hessian(mol, h / 2.0, &mut grad);
    (4.0 * b - a) / 3.0
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn assert_close(ctx: &str, got: &Array2<f64>, want: &Array2<f64>, tol: f64) -> f64 {
    let d = max_abs_diff(got, want);
    eprintln!(
        "{ctx}: max|analytic - FD| = {d:.3e} (max|FD| {:.3e}, tol {tol:.0e})",
        max_abs(want)
    );
    assert!(d < tol, "{ctx}: max|analytic - FD| = {d:.3e} >= {tol:.0e}");
    d
}

fn prep_at(m: &Molecule, basis_name: &str) -> PreparedBasis {
    PreparedBasis::new(m, &basis::bundled(basis_name).unwrap()).unwrap()
}

fn aux_at(m: &Molecule, aux: &str) -> PreparedBasis {
    PreparedBasis::new(m, &basis::bundled(aux).unwrap()).unwrap()
}

/// `X = sqrt(2) C_occ`, so `D = X Xᵀ` is the closed-shell total density.
fn occ_factor(s: &Setup) -> Array2<f64> {
    let nocc = (s.mol.nelec() / 2) as usize;
    s.rhf.mos_r().slice(s![.., ..nocc]).to_owned() * std::f64::consts::SQRT_2
}

/// Smallest and largest metric eigenvalue of the fit at this geometry, and
/// whether any lies in the band the gradient treats as dropped although the
/// energy keeps it (`[1e-10, 1e-7)`): the FD of the own gradient is only a
/// valid reference when none does.
fn clean_gap(mol: &Molecule, aux: &str) -> (f64, f64, bool) {
    let v = coulomb_metric_2c(Operator::coulomb(), &aux_at(mol, aux)).unwrap();
    let (lam, _) = v.eigh(UPLO::Upper).unwrap();
    let band = lam.iter().any(|&l| (1e-10..1e-7).contains(&l));
    (lam[0], lam[lam.len() - 1], !band)
}

const H_PIECE: f64 = 1e-4;
/// Step for finite differences of the fit coefficients `Γ = V⁻¹ L` (the same
/// minimum as the RI-J pieces: noise ~ cond(V)/h², truncation ~ h²).
const H_C: f64 = 3e-4;

// ---------------------------------------------------------------------------
// K^x (first-order exchange Fock) vs the SCF's own DfK
// ---------------------------------------------------------------------------

fn check_first_order_k(basis_name: &str, aux: &str) {
    if !has_deriv1() {
        return;
    }
    let s = setup(&water(), basis_name, "", aux);
    let route = s.rhf.df_jk.as_ref().unwrap();
    let d = s.rhf.density_r();
    let x = occ_factor(&s);
    let terms = rik_terms(&s.mol, &s.prep, route, &x, &RikMutants::default()).unwrap();
    let k_at = |m: &Molecule| {
        let prep = prep_at(m, basis_name);
        let mut dfk = DfK::new(Operator::coulomb(), &prep, &aux_at(m, aux), usize::MAX).unwrap();
        let mut k = Array2::<f64>::zeros(d.dim());
        dfk.build(d, &mut k).unwrap();
        k
    };
    let n3 = 3 * s.mol.atoms.len();
    for xi in 0..n3 {
        let kp = k_at(&displaced(&s.mol, xi / 3, xi % 3, H_PIECE));
        let km = k_at(&displaced(&s.mol, xi / 3, xi % 3, -H_PIECE));
        let fd = (kp - km) / (2.0 * H_PIECE);
        assert_close(
            &format!("{basis_name}/{aux}: K^x, x = {xi}"),
            &terms.first_order_k[xi],
            &fd,
            5e-7,
        );
        assert!(max_abs(&fd) > 1e-3, "vacuous: K^x ~ 0 at x = {xi}");
    }
}

#[test]
fn first_order_k_matches_dfk_fd_sto3g() {
    check_first_order_k("sto-3g", "def2-universal-jkfit");
}

#[test]
fn first_order_k_matches_dfk_fd_631g() {
    check_first_order_k("6-31g", "cc-pvdz-ri");
}

// ---------------------------------------------------------------------------
// Skeleton pieces vs fixed-density finite differences
// ---------------------------------------------------------------------------

fn check_skeleton_pieces(basis_name: &str, aux: &str) {
    if !has_deriv1() {
        return;
    }
    let s = setup(&water(), basis_name, "", aux);
    let route = s.rhf.df_jk.as_ref().unwrap();
    let d = s.rhf.density_r();
    let x = occ_factor(&s);
    let terms = rik_terms(&s.mol, &s.prep, route, &x, &RikMutants::default()).unwrap();
    let n3 = 3 * s.mol.atoms.len();
    let alpha = ferric_scf::hessian_df_k::RIK_ALPHA;
    let f0 = rik_first(&s.mol, &s.prep, route, &x, None).unwrap();
    let g0 = f0.gamma.clone();
    let first_at = |m: &Molecule, fixed: Option<&Array2<f64>>| {
        rik_first(m, &prep_at(m, basis_name), route, &x, fixed).unwrap()
    };
    let mut three = Array2::<f64>::zeros((n3, n3));
    let mut metric = Array2::<f64>::zeros((n3, n3));
    let mut cross = Array2::<f64>::zeros((n3, n3));
    for y in 0..n3 {
        let (mp, mm) = (
            displaced(&s.mol, y / 3, y % 3, H_PIECE),
            displaced(&s.mol, y / 3, y % 3, -H_PIECE),
        );
        let (fp, fm) = (first_at(&mp, Some(&g0)), first_at(&mm, Some(&g0)));
        let (cp, cm) = (
            first_at(&displaced(&s.mol, y / 3, y % 3, H_C), None),
            first_at(&displaced(&s.mol, y / 3, y % 3, -H_C), None),
        );
        let dgamma = (&cp.gamma - &cm.gamma) / (2.0 * H_C);
        for xi in 0..n3 {
            three[(xi, y)] = 2.0 * alpha * (fp.lg[xi] - fm.lg[xi]) / (2.0 * H_PIECE);
            metric[(xi, y)] = -alpha * (fp.gvg[xi] - fm.gvg[xi]) / (2.0 * H_PIECE);
            cross[(xi, y)] = 2.0
                * alpha
                * f0.z
                    .row(xi)
                    .iter()
                    .zip(dgamma.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
        }
    }
    let tol = 2e-6;
    let tag = format!("{basis_name}/{aux}");
    assert_close(
        &format!("{tag}: 3c piece"),
        &terms.three_centre,
        &three,
        tol,
    );
    assert_close(
        &format!("{tag}: 2c piece"),
        &terms.metric_second,
        &metric,
        tol,
    );
    assert_close(&format!("{tag}: cross piece"), &terms.cross, &cross, tol);
    for (name, m) in [("3c", &three), ("2c", &metric), ("cross", &cross)] {
        assert!(max_abs(m) > 1e-3, "vacuous: {name} piece ~ 0");
    }

    // The whole K skeleton vs d/dy of the validated DF gradient at fixed D.
    let natoms = s.mol.atoms.len();
    let fd = fd_hessian(&s.mol, H_PIECE, |m| {
        let prep = prep_at(m, basis_name);
        df_jk_gradient(
            natoms,
            &prep,
            &aux_at(m, aux),
            Operator::coulomb(),
            None,
            &[DfKChannel { density: d, alpha }],
            route.budget_bytes,
        )
        .unwrap()
    });
    assert_close(&format!("{tag}: K skeleton"), &terms.skeleton, &fd, tol);
    assert!(
        terms.skeleton_asymmetry < 1e-9,
        "raw skeleton asymmetry {:.3e}",
        terms.skeleton_asymmetry
    );
}

#[test]
fn skeleton_pieces_match_fd_sto3g() {
    check_skeleton_pieces("sto-3g", "def2-universal-jkfit");
}

#[test]
fn skeleton_pieces_match_fd_631g() {
    check_skeleton_pieces("6-31g", "cc-pvdz-ri");
}

// ---------------------------------------------------------------------------
// Total vs FD of ferric's own DF gradient
// ---------------------------------------------------------------------------

fn df_gradient_at(m: &Molecule, basis_name: &str, j_aux: &str, k_aux: &str) -> Array2<f64> {
    let sm = setup(m, basis_name, j_aux, k_aux);
    rhf_gradient(m, &sm.prep, Operator::coulomb(), &sm.bounds, &sm.rhf, None).unwrap()
}

fn check_full(mol: &Molecule, basis_name: &str, j_aux: &str, k_aux: &str, tol: f64) -> (f64, f64) {
    let s = setup(mol, basis_name, j_aux, k_aux);
    let (lmin, lmax, clean) = clean_gap(mol, k_aux);
    eprintln!(
        "{basis_name}/{k_aux}: metric lambda_min {lmin:.3e}, cond {:.2e}",
        lmax / lmin
    );
    assert!(clean, "metric has a mode in the gradient's noise band");
    let p = parts(&s);
    let total = p.total();
    let fd = fd_hessian_richardson(&s.mol, 2e-3, |m| {
        df_gradient_at(m, basis_name, j_aux, k_aux)
    });
    eprintln!(
        "{basis_name}/J={j_aux:?}/K={k_aux}: CPHF iterations {:?}, residual {:.2e}, response asym {:.2e}",
        p.cphf_iterations, p.cphf_max_residual, p.response_asymmetry
    );
    assert!(
        max_abs_diff(&p.skeleton(), &fd) > 100.0 * tol,
        "skeleton alone matches FD: the full test cannot see the response"
    );
    let miss = assert_close(
        &format!("{basis_name}/J={j_aux:?}/K={k_aux}: full Hessian"),
        &total,
        &fd,
        tol,
    );
    assert!(
        p.response_asymmetry < 1e-9,
        "asym {:.3e}",
        p.response_asymmetry
    );
    (miss, max_abs(&fd))
}

const BAR_OWN: f64 = 1e-8;

#[test]
fn full_hessian_matches_fd_of_df_gradient_sto3g() {
    if !has_deriv2() {
        return;
    }
    check_full(&water(), "sto-3g", "", "def2-universal-jkfit", BAR_OWN);
}

#[test]
fn full_hessian_matches_fd_of_df_gradient_631g() {
    if !has_deriv2() {
        return;
    }
    check_full(&water(), "6-31g", "", "cc-pvdz-ri", BAR_OWN);
}

#[test]
fn full_hessian_with_ri_j_and_ri_k_matches_fd_631g() {
    if !has_deriv1() {
        return;
    }
    check_full(&water(), "6-31g", "cc-pvdz-ri", "cc-pvdz-ri", BAR_OWN);
}

#[test]
fn translational_and_rotational_invariance() {
    if !has_deriv2() {
        return;
    }
    let s = setup(&water(), "6-31g", "", "cc-pvdz-ri");
    let total = parts(&s).total();
    let n = s.mol.atoms.len();
    let n3 = 3 * n;
    let mut worst = 0.0f64;
    for a in 0..n3 {
        for axis in 0..3 {
            let sum: f64 = (0..n).map(|b| total[(a, 3 * b + axis)]).sum();
            worst = worst.max(sum.abs());
        }
    }
    eprintln!("translational invariance: worst row sum {worst:.3e}");
    assert!(worst < 1e-9, "row sum {worst:.3e}");
    let g = rhf_gradient(
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &s.rhf,
        None,
    )
    .unwrap();
    let r: Vec<[f64; 3]> = s.mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let cross = |e: [f64; 3], v: [f64; 3]| {
        [
            e[1] * v[2] - e[2] * v[1],
            e[2] * v[0] - e[0] * v[2],
            e[0] * v[1] - e[1] * v[0],
        ]
    };
    let mut worst_rot = 0.0f64;
    for axis in 0..3 {
        let mut e = [0.0; 3];
        e[axis] = 1.0;
        let u: Vec<f64> = r.iter().flat_map(|&ri| cross(e, ri)).collect();
        for atom in 0..n {
            let gv = cross(e, [g[(atom, 0)], g[(atom, 1)], g[(atom, 2)]]);
            for k in 0..3 {
                let hu: f64 = (0..n3).map(|b| total[(3 * atom + k, b)] * u[b]).sum();
                worst_rot = worst_rot.max((hu - gv[k]).abs());
            }
        }
    }
    eprintln!("rotational invariance: worst |H (e x r) - e x g| = {worst_rot:.3e}");
    assert!(worst_rot < 1e-7, "rotational {worst_rot:.3e}");
}

#[allow(dead_code)]
fn unused(_: &Array1<f64>, _: &Setup) {
    let _ = (rhf_hessian, ());
}
