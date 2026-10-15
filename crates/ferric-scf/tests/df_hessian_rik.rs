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
use ndarray::{s, Array2};
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
        energy_conv: 1e-14,
        density_conv: 1e-12,
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
    assert_eq!(
        route.k_aux.as_deref(),
        Some(k_aux).filter(|k| !k.is_empty()),
        "RI-K route"
    );
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

#[test]
fn every_k_x_mutant_fails_the_dfk_finite_difference_test() {
    // The rank-one anchor only sees the skeleton; K^x mutants are killed by the
    // DfK finite-difference test (bar 5e-7). Each must move K^x by > 10 bars.
    if !has_deriv1() {
        return;
    }
    let s = setup(&water(), "6-31g", "", "cc-pvdz-ri");
    let route = s.rhf.df_jk.as_ref().unwrap();
    let x = occ_factor(&s);
    let good = rik_terms(&s.mol, &s.prep, route, &x, &RikMutants::default()).unwrap();
    let d = RikMutants::default();
    for (name, m) in [
        (
            "drop K^x metric term",
            RikMutants {
                drop_kx_metric: true,
                ..d
            },
        ),
        ("drop K^x", RikMutants { drop_kx: true, ..d }),
        (
            "omit aux-centre",
            RikMutants {
                omit_aux_centre_kx: true,
                ..d
            },
        ),
        (
            "transpose pair",
            RikMutants {
                transpose_pair_kx: true,
                ..d
            },
        ),
        (
            "swap metric",
            RikMutants {
                swap_metric: true,
                ..d
            },
        ),
        (
            "alpha x2",
            RikMutants {
                alpha_factor: 2.0,
                ..d
            },
        ),
    ] {
        let bad = rik_terms(&s.mol, &s.prep, route, &x, &m).unwrap();
        let w2 = good
            .first_order_fock
            .iter()
            .zip(&bad.first_order_fock)
            .fold(0.0f64, |mx, (g, b)| mx.max(max_abs_diff(g, b)));
        eprintln!("K^x mutant {name}: max element change {w2:.3e} (FD bar 5e-7)");
        assert!(w2 > 10.0 * 5e-7, "mutant {name}: {w2:.3e}");
    }
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

/// Agreement bar of the production Hessian with the central FD (Richardson,
/// h = 2e-3 / 1e-3) of ferric's own analytic DF gradient, SCF re-converged at
/// every displaced geometry, converged to energy_conv 1e-14 / density_conv
/// 1e-12. Production floors measured (release, RI-K only): water STO-3G/jkfit
/// 1.1e-10, 6-31G/cc-pVDZ-RI 1.3e-10, cc-pVDZ/jkfit 3.4e-10, NH3 cc-pVDZ/jkfit
/// 5.7e-10, def2-SVP/jkfit 4.8e-10 (worst); RI-J + RI-K 6-31G 1.2e-10,
/// cc-pVDZ/jkfit 3.8e-10. With the displaced SCFs converged only to 1e-12 /
/// 1e-10 the same tests reported up to 3.0e-9 (def2-SVP, identical for RI-J and
/// RI-K): that was the FD reference's SCF noise, not the Hessian.
const BAR_OWN: f64 = 1e-8;
/// Bar of the mutation suite, which runs on water 6-31G/cc-pVDZ-RI. Derived
/// from both sides: the production agreement with the differenced own gradient
/// on this system is 1.4e-10 (1.6e-10 with RI-J), so a bar of 1e-9 is 7x above
/// what a correct Hessian shows; the closest mutant (stencil step x10) misses
/// by 6.8e-8, i.e. 68 bars = 6.8x the 10-bar pass line (the margin stated, not
/// assumed); every other mutant misses by >= 5.6e-2.
const BAR_MUT: f64 = 1e-9;

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
    // The Hessian needs the one-electron/overlap second derivatives, which the
    // shipped first-derivative libint lacks (as for every analytic Hessian).
    if !has_deriv2() {
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

// ---------------------------------------------------------------------------
// Anchor: one occupied column reduces RI-K to RI-J (independent code path)
// ---------------------------------------------------------------------------
//
// An exactly-spanning auxiliary basis (the RI-J anchor) does not exist for
// exchange: E_K sums self-repulsions of PAIR densities rho_ij = phi_i phi_j,
// and with i, j on different atoms rho_ij is a two-centre product, which no
// finite set of atom-centred (rigidly moving) auxiliary functions spans; with
// both on one atom every integral is one-centre and the Hessian is identically
// zero (translational invariance), so that configuration is vacuous (measured
// scale 3e-17). What does hold exactly: for ONE occupied column
// (m = 1, D = x xᵀ) L_P = (P|rho_11) = d_P, so
//
//     E_K = alpha d^T M d = 2 alpha E_J^DF(D),   alpha = -1/4,
//
// and every skeleton piece must equal 2 alpha times the matching piece of the
// merged RI-J Hessian (`rij_terms`, itself validated by an exact-span anchor,
// own-gradient FD and PySCF), through a code path that shares nothing with
// `rik_terms` except the stencil module and the metric. Any aux basis works,
// so the test uses the production fits.
//
// Hypotheses. If right: the three pieces agree to the stencil/Cholesky floor
// (~1e-9). If a RI-K term is dropped or mis-weighted the matching piece
// misses by that term. BLIND to: everything that needs m > 1 (the pair index
// ij, off-diagonal pairs, the transpose structure of K^x) -- covered by the
// FD tests above and the PySCF comparison. K^x is not tested here (K[D] and
// J[D] are different functionals of D even for rank one).

/// A deterministic dense column over all basis functions.
fn dense_column(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, 1), |(i, _)| {
        let t = i as f64;
        (0.9 + 0.6 * (1.3 * t).sin()) / (1.0 + 0.15 * t)
    })
}

fn rank_one_pieces(basis_name: &str, aux: &str, mu: &RikMutants) -> (Vec<f64>, f64) {
    use ferric_scf::hessian::rij_support::{rij_terms_with_aux, rik_terms_with_aux, RijMutants};
    let mol = water();
    let prep = prep_at(&mol, basis_name);
    let x = dense_column(prep.nbasis());
    let d = x.dot(&x.t());
    let op = Operator::coulomb();
    let natoms = mol.atoms.len();
    let rij = rij_terms_with_aux(
        natoms,
        &prep,
        aux_at(&mol, aux),
        op,
        usize::MAX,
        &d,
        &RijMutants::default(),
    )
    .unwrap();
    let rik = rik_terms_with_aux(natoms, &prep, aux_at(&mol, aux), op, usize::MAX, &x, mu).unwrap();
    let f = 2.0 * ferric_scf::hessian_df_k::RIK_ALPHA;
    let misses = vec![
        max_abs_diff(&rik.three_centre, &(&rij.three_centre * f)),
        max_abs_diff(&rik.metric_second, &(&rij.metric_second * f)),
        max_abs_diff(&rik.cross, &(&rij.cross * f)),
        max_abs_diff(&rik.skeleton, &(&rij.skeleton * f)),
    ];
    (misses, max_abs(&(&rij.skeleton * f)))
}

fn check_rank_one(basis_name: &str, aux: &str) {
    if !has_deriv1() {
        return;
    }
    let bar = 1e-8;
    let (m, scale) = rank_one_pieces(basis_name, aux, &RikMutants::default());
    eprintln!(
        "{basis_name}/{aux} rank-one RI-K vs 2*alpha*RI-J: 3c {:.2e}, 2c {:.2e}, cross {:.2e}, \
         skeleton {:.2e} (scale {scale:.2e})",
        m[0], m[1], m[2], m[3]
    );
    assert!(scale > 1e-2, "vacuous anchor");
    assert!(m.iter().all(|&v| v < bar), "rank-one identity: {m:?}");
    let d = RikMutants::default();
    for (name, mu, piece) in [
        (
            "drop metric second",
            RikMutants {
                drop_metric_second: true,
                ..d
            },
            3,
        ),
        (
            "drop cross",
            RikMutants {
                drop_cross: true,
                ..d
            },
            3,
        ),
        (
            "alpha x2",
            RikMutants {
                alpha_factor: 2.0,
                ..d
            },
            3,
        ),
        (
            "alpha x1/2",
            RikMutants {
                alpha_factor: 0.5,
                ..d
            },
            3,
        ),
    ] {
        let (mm, _) = rank_one_pieces(basis_name, aux, &mu);
        eprintln!("  mutant {name}: skeleton miss {:.3e}", mm[piece]);
        assert!(mm[piece] > 10.0 * bar, "mutant {name} survives the anchor");
    }
}

#[test]
fn rank_one_exchange_equals_scaled_ri_j_sto3g() {
    check_rank_one("sto-3g", "def2-universal-jkfit");
}

#[test]
fn rank_one_exchange_equals_scaled_ri_j_631g() {
    check_rank_one("6-31g", "cc-pvdz-ri");
}

// ---------------------------------------------------------------------------
// Mutation suite (production-wired) and negative controls
// ---------------------------------------------------------------------------
//
// BAR_OWN is the agreement bar of the production Hessian against the central
// FD (Richardson, h = 2e-3 / 1e-3) of ferric's own analytic DF gradient with
// re-converged SCFs. Every mutant must miss the production Hessian by more
// than 10 bars; identities are reported as BLIND, never counted.

fn mutant_total(s: &Setup, m: &RikMutants) -> Array2<f64> {
    ferric_scf::hessian::rij_support::rhf_hessian_parts_rik_mutated(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &s.rhf,
        &s.cfg,
        m,
    )
    .unwrap()
    .total()
}

fn all_mutants() -> Vec<(&'static str, RikMutants)> {
    use ferric_scf::hessian_df_stencil::{StencilConfig, Variant};
    let d = RikMutants::default();
    let st = |v| StencilConfig {
        variant: v,
        ..Default::default()
    };
    vec![
        (
            "drop -a<G,V_xy G> (2c second derivative)",
            RikMutants {
                drop_metric_second: true,
                ..d
            },
        ),
        (
            "drop cross terms",
            RikMutants {
                drop_cross: true,
                ..d
            },
        ),
        (
            "drop K^x metric term",
            RikMutants {
                drop_kx_metric: true,
                ..d
            },
        ),
        ("drop K^x altogether", RikMutants { drop_kx: true, ..d }),
        (
            "omit aux-centre part of K^x",
            RikMutants {
                omit_aux_centre_kx: true,
                ..d
            },
        ),
        (
            "transpose of the pair in K^x",
            RikMutants {
                transpose_pair_kx: true,
                ..d
            },
        ),
        (
            "metric swapped for erfc(0.4)",
            RikMutants {
                swap_metric: true,
                ..d
            },
        ),
        (
            "exchange weight x2",
            RikMutants {
                alpha_factor: 2.0,
                ..d
            },
        ),
        (
            "exchange weight x1/2",
            RikMutants {
                alpha_factor: 0.5,
                ..d
            },
        ),
        (
            "stencil step x10",
            RikMutants {
                stencil: st(Variant::Step10),
                ..d
            },
        ),
        (
            "one-shell displacement",
            RikMutants {
                stencil: st(Variant::SingleShell),
                ..d
            },
        ),
        (
            "CPHF kernel: exact K where the SCF fitted it",
            RikMutants {
                cphf_exact_k: true,
                ..d
            },
        ),
        (
            "CPHF kernel: J and K swapped",
            RikMutants {
                cphf_swap_jk: true,
                ..d
            },
        ),
    ]
}

#[test]
fn cphf_kernel_mutants_fail_the_finite_difference_test() {
    // The kernel G of the CPHF is not touched by the skeleton or K^x tests; a
    // wrong kernel changes only the orbital response. Hypothesis: production
    // agrees with the differenced own gradient at the FD floor, each kernel
    // mutant misses it by > 10 bars.
    if !has_deriv2() {
        return;
    }
    let s = setup(&water(), "6-31g", "", "cc-pvdz-ri");
    let fd = fd_hessian_richardson(&s.mol, 2e-3, |m| {
        df_gradient_at(m, "6-31g", "", "cc-pvdz-ri")
    });
    let prod = max_abs_diff(&parts(&s).total(), &fd);
    eprintln!("production vs FD: {prod:.2e} (bar {BAR_MUT:.0e})");
    assert!(prod < BAR_MUT);
    let d = RikMutants::default();
    for (name, m) in [
        (
            "exact K inside G",
            RikMutants {
                cphf_exact_k: true,
                ..d
            },
        ),
        (
            "J and K swapped in G",
            RikMutants {
                cphf_swap_jk: true,
                ..d
            },
        ),
    ] {
        let miss = max_abs_diff(&mutant_total(&s, &m), &fd);
        eprintln!(
            "CPHF mutant {name}: FD miss {miss:.3e} ({:.1e} bars)",
            miss / BAR_MUT
        );
        assert!(miss > 10.0 * BAR_MUT, "{name}: {miss:.3e}");
    }
}

#[test]
fn an_inconsistent_density_is_refused_for_the_exchange_factor() {
    // RI-K builds X from the first nocc MOs; a result whose density is not
    // that (here: the occupied and a virtual MO exchanged) must be a typed
    // error, never a Hessian of the wrong state.
    if !has_deriv2() {
        return;
    }
    let s = setup(&water(), "sto-3g", "", "def2-universal-jkfit");
    let nocc = (s.mol.nelec() / 2) as usize;
    let mut bad = s.rhf.clone();
    let (a, b) = (
        bad.mos_alpha.column(0).to_owned(),
        bad.mos_alpha.column(nocc).to_owned(),
    );
    bad.mos_alpha.column_mut(0).assign(&b);
    bad.mos_alpha.column_mut(nocc).assign(&a);
    let e = rhf_hessian(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &bad,
        &s.cfg,
    )
    .unwrap_err()
    .to_string();
    eprintln!("inconsistent density: {e}");
    assert!(e.contains("first nocc MOs"), "{e}");
}

#[test]
fn the_probe_that_auto_consults_is_stricter_than_the_hessian_itself() {
    // Twin-augmented cc-pVDZ-RI (a near-duplicate s function on each H):
    // eps 1e-2 -> cond 7e7 (accepted), 3e-3 -> lambda_min 1.7e-7, cond 7.9e8
    // (cond above 1e8: probe refuses, the Hessian still runs with a warning),
    // 1e-3 -> lambda_min ~2e-8 (a kept mode in [1e-10, 1e-7): probe refuses),
    // 1e-5 -> modes below 1e-10 (both refuse, naming the count).
    use ferric_scf::hessian::rij_support::{probe_exchange_metric, rik_terms_with_aux};
    if !has_deriv1() {
        return;
    }
    let mol = water();
    let prep = prep_at(&mol, "sto-3g");
    let x = dense_column(prep.nbasis());
    let op = Operator::coulomb();
    let run = |eps: f64| {
        rik_terms_with_aux(
            3,
            &prep,
            twin_aux(&mol, "cc-pvdz-ri", eps),
            op,
            usize::MAX,
            &x,
            &RikMutants::default(),
        )
    };
    let probe = |eps: f64| probe_exchange_metric(&twin_aux(&mol, "cc-pvdz-ri", eps), op);
    probe(1e-2).expect("cond 7e7 is accepted");
    run(1e-2).expect("and the Hessian runs");
    let e = probe(3e-3).unwrap_err().to_string();
    eprintln!("probe cond: {e}");
    assert!(e.contains("cond"), "{e}");
    run(3e-3).expect("Analytic is strict: the Hessian itself runs (with a warning)");
    let e = probe(1e-3).unwrap_err().to_string();
    eprintln!("probe band: {e}");
    assert!(e.contains("lie in"), "{e}");
    run(1e-3).expect("the Hessian itself runs with a kept mode in the band");
    for f in [probe(1e-5).map(|_| ()), run(1e-5).map(|_| ())] {
        let e = f.unwrap_err().to_string();
        assert!(e.contains("dropped mode"), "{e}");
    }
}

#[test]
fn every_wired_mutant_misses_by_more_than_ten_bars() {
    if !has_deriv2() {
        return;
    }
    let s = setup(&water(), "6-31g", "", "cc-pvdz-ri");
    let prod = parts(&s).total();
    let again = mutant_total(&s, &RikMutants::default());
    assert_eq!(
        max_abs_diff(&prod, &again),
        0.0,
        "production is not reproducible"
    );
    for (name, m) in all_mutants() {
        let miss = max_abs_diff(&mutant_total(&s, &m), &prod);
        eprintln!(
            "MUTANT {name}: miss {miss:.3e} ({:.1e} bars)",
            miss / BAR_MUT
        );
        assert!(
            miss > 10.0 * BAR_MUT,
            "mutant {name} misses by only {miss:.3e}"
        );
    }
}

#[test]
fn explicit_inverse_mutant_is_measured_not_claimed() {
    // At the conditioning of a water cc-pVDZ-RI metric (cond ~ 2.5e5) the LU
    // inverse equals the V^{-1/2} V^{-1/2} product to roundoff: BLIND here,
    // not counted. The ignored benzene test measures it where it matters.
    if !has_deriv2() {
        return;
    }
    let s = setup(&water(), "6-31g", "", "cc-pvdz-ri");
    let prod = parts(&s).total();
    let miss = max_abs_diff(
        &mutant_total(
            &s,
            &RikMutants {
                explicit_inverse: true,
                ..Default::default()
            },
        ),
        &prod,
    );
    eprintln!("explicit-inverse mutant, water 6-31G/cc-pVDZ-RI: miss {miss:.3e}");
    assert!(
        miss < BAR_MUT,
        "explicit inverse became detectable here: {miss:.3e}"
    );
}

/// An SCF with explicit RI settings that may leave K (or J) exact.
fn setup_mixed(basis_name: &str, j_aux: &str, k_aux: &str) -> Setup {
    let mol = water();
    let prep = prep_at(&mol, basis_name);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = scf_config(j_aux, k_aux);
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged);
    Setup {
        mol,
        prep,
        bounds,
        rhf,
        cfg,
    }
}

#[test]
fn the_exact_k_and_ri_j_only_hessians_miss_the_ri_k_reference() {
    // Negative controls: a Hessian that does not differentiate the RI-K
    // energy is NOT the reference. (a) the all-exact Hessian, (b) RI-J with
    // exact K. Hypothesis: the fitting error is visible at >= 10 bars.
    if !has_deriv2() {
        return;
    }
    let (basis_name, aux) = ("6-31g", "cc-pvdz-ri");
    let reference = parts(&setup(&water(), basis_name, "", aux)).total();
    let exact = parts(&setup_mixed(basis_name, "", "")).total();
    let ri_j_only = parts(&setup_mixed(basis_name, aux, "")).total();
    for (name, h) in [("all-exact", &exact), ("RI-J, exact K", &ri_j_only)] {
        let miss = max_abs_diff(h, &reference);
        eprintln!(
            "NEGATIVE CONTROL {name}: miss {miss:.3e} ({:.1e} bars)",
            miss / BAR_MUT
        );
        assert!(miss > 10.0 * BAR_MUT, "{name} misses by only {miss:.3e}");
    }
}

// ---------------------------------------------------------------------------
// Refusals, availability, DF off
// ---------------------------------------------------------------------------

#[test]
fn unsupported_exchange_fitting_configurations_are_refused() {
    use ferric_scf::hessian::{
        analytic_hessian_available, analytic_rohf_hessian_available, rhf_hessian_preflight,
        rohf_hessian_preflight, uhf_hessian_preflight,
    };
    let s = setup(&water(), "sto-3g", "", "def2-universal-jkfit");
    let op = Operator::coulomb();
    // Supported: closed-shell RHF RI-K, alone and with RI-J.
    rhf_hessian_preflight(op, &s.cfg).expect("RI-K preflight");
    if has_deriv2() {
        analytic_hessian_available(&s.mol, &s.prep, op, &s.cfg).expect("available");
    }
    // Both fitted: same probe (the one-electron second derivatives are needed).
    if has_deriv2() {
        analytic_hessian_available(
            &s.mol,
            &s.prep,
            op,
            &scf_config("def2-universal-jkfit", "def2-universal-jkfit"),
        )
        .expect("RI-J + RI-K available");
    }
    // ROHF and UHF RI-K are supported (tests/df_hessian_rohf.rs, df_hessian_uhf.rs).
    rohf_hessian_preflight(op, &s.cfg).expect("ROHF RI-K preflight");
    if has_deriv2() {
        analytic_rohf_hessian_available(&s.mol, &s.prep, op, &s.cfg).expect("ROHF RI-K probe");
    }
    uhf_hessian_preflight(op, &s.cfg).expect("UHF RI-K preflight");
    // External potential next to RI-K.
    let with_field = RhfConfig {
        external_potential: Some(ferric_core::external_potential::ExternalPotential {
            point_charges: vec![ferric_core::external_potential::PointCharge {
                q: 0.3,
                x: 5.0,
                y: 0.0,
                z: 0.0,
            }],
            ..Default::default()
        }),
        ..s.cfg.clone()
    };
    let e = rhf_hessian_preflight(op, &with_field)
        .unwrap_err()
        .to_string();
    assert!(e.contains("external potential"), "{e}");
    // A result whose RI-K basis differs from the configuration's.
    let other = scf_config("", "cc-pvdz-ri");
    let e = rhf_hessian(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        op,
        &s.bounds,
        &s.rhf,
        &other,
    )
    .unwrap_err()
    .to_string();
    eprintln!("route/config mismatch: {e}");
    assert!(
        e.contains("RI-K") && e.contains("differs from the configuration"),
        "{e}"
    );
}

#[test]
fn df_off_configurations_are_unchanged_by_the_exchange_branch() {
    // Exact SCFs (df_k_aux unset or "") carry no route, never enter the RI-K
    // code, and give the same Hessian bit for bit; the hash of every part
    // against origin/main is recorded in the PR description.
    if !has_deriv2() {
        return;
    }
    let mol = water();
    let prep = prep_at(&mol, "sto-3g");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let hess = |df_k: Option<String>| {
        let cfg = RhfConfig {
            max_iter: 200,
            energy_conv: 1e-12,
            density_conv: 1e-10,
            df_j_aux: Some(String::new()),
            df_k_aux: df_k,
            ..Default::default()
        };
        let rhf = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
        assert!(rhf.df_jk.is_none());
        rhf_hessian(
            &ParallelContext::default(),
            &mol,
            &prep,
            op,
            &bounds,
            &rhf,
            &cfg,
        )
        .unwrap()
    };
    assert_eq!(max_abs_diff(&hess(None), &hess(Some(String::new()))), 0.0);
}

#[test]
fn a_nearly_dependent_exchange_basis_is_refused_naming_the_dropped_mode_count() {
    // Hypothesis: a fitting basis with a near-duplicate s function on each
    // hydrogen has metric eigenvalues below the DF-K cut (1e-10); DfK DROPS
    // those modes, so the energy is a spectral function of V and its second
    // derivative (Daleckii-Krein to second order) is not implemented: refused
    // with the count. The artifact hypothesis (the guard fires on every basis)
    // is excluded by the accepted twin 1e-3 away.
    use ferric_core::basis::Shell;
    if !has_deriv1() {
        return;
    }
    let mol = water();
    let prep = prep_at(&mol, "sto-3g");
    let x = dense_column(prep.nbasis());
    let try_twin = |eps: f64| {
        let mut aux = basis::bundled("cc-pvdz-ri").unwrap();
        let h = aux.shells.get_mut(&1).unwrap();
        let first = h.iter().find(|s| s.l == 0).unwrap().clone();
        let a = first.exponents[0];
        h.push(Shell {
            exponents: vec![a * (1.0 + eps)],
            coefficients: vec![(2.0 * a / std::f64::consts::PI).powf(0.75)],
            ..first
        });
        ferric_scf::hessian::rij_support::rik_terms_with_aux(
            mol.atoms.len(),
            &prep,
            PreparedBasis::new(&mol, &aux).unwrap(),
            Operator::coulomb(),
            usize::MAX,
            &x,
            &RikMutants::default(),
        )
    };
    try_twin(1e-3).expect("a well-separated twin must be accepted");
    for eps in [1e-5, 1e-7] {
        let msg = try_twin(eps).expect_err("dropped modes").to_string();
        eprintln!("twin {eps:.0e}: {msg}");
        assert!(msg.contains("2 dropped mode(s)"), "{msg}");
    }
}

#[test]
fn the_pre_scf_probe_accepts_healthy_metrics_and_the_long_range_metric_has_dropped_modes() {
    use ferric_scf::hessian::rij_support::probe_exchange_metric;
    let mol = water();
    let op = Operator::coulomb();
    probe_exchange_metric(&aux_at(&mol, "def2-universal-jkfit"), op).expect("healthy metric");
    // Control for the refusal: the erf(0.3) metric of this basis (used by the
    // range-separated exchange of the SCF, not by this Hessian) has modes the
    // DfK cut drops.
    let lr = coulomb_metric_2c(Operator::erf(0.3), &aux_at(&mol, "def2-universal-jkfit")).unwrap();
    let (lam, _) = lr.eigh(UPLO::Upper).unwrap();
    let dropped = lam.iter().filter(|&&l| l < 1e-10).count();
    eprintln!("erf(0.3) metric of def2-universal-jkfit: {dropped} dropped modes");
    assert!(dropped > 0, "the control metric must have dropped modes");
}

// ---------------------------------------------------------------------------
// Larger systems (ignored: measured wall times, release, one run each)
// ---------------------------------------------------------------------------

const NH3_DISTORTED: &str = "4\nNH3 distorted\n\
N      0.010000     0.020000     0.100000\n\
H      0.020000     0.940000    -0.300000\n\
H      0.810000    -0.480000    -0.340000\n\
H     -0.830000    -0.460000    -0.310000\n";

#[test]
#[ignore = "slow: water cc-pVDZ/def2-universal-jkfit RI-K own-gradient FD; measured 26 s (box load ~8)"]
fn full_hessian_matches_fd_of_df_gradient_ccpvdz_jkfit() {
    if !has_deriv2() {
        return;
    }
    check_full(&water(), "cc-pvdz", "", "def2-universal-jkfit", BAR_OWN);
}

#[test]
#[ignore = "slow: water def2-SVP/def2-universal-jkfit RI-K own-gradient FD; measured 6.7 s (box load ~8)"]
fn full_hessian_matches_fd_of_df_gradient_def2svp_jkfit() {
    if !has_deriv2() {
        return;
    }
    check_full(&water(), "def2-svp", "", "def2-universal-jkfit", BAR_OWN);
}

#[test]
#[ignore = "slow: water cc-pVDZ RI-J + RI-K (both jkfit) own-gradient FD; measured 16 s (box load ~8)"]
fn full_hessian_matches_fd_of_df_gradient_ccpvdz_ri_jk() {
    if !has_deriv2() {
        return;
    }
    check_full(
        &water(),
        "cc-pvdz",
        "def2-universal-jkfit",
        "def2-universal-jkfit",
        BAR_OWN,
    );
}

#[test]
#[ignore = "slow: NH3 cc-pVDZ/def2-universal-jkfit RI-K own-gradient FD; measured 33 s (box load ~8)"]
fn full_hessian_matches_fd_of_df_gradient_nh3_ccpvdz_jkfit() {
    if !has_deriv2() {
        return;
    }
    let mol = Molecule::parse_xyz(NH3_DISTORTED, 0, 1).unwrap();
    let (miss, scale) = check_full(&mol, "cc-pvdz", "", "def2-universal-jkfit", BAR_OWN);
    eprintln!("NH3 cc-pVDZ/jkfit RI-K: miss {miss:.3e} of {scale:.3e}");
}

// ---------------------------------------------------------------------------
// Memory pre-flight
// ---------------------------------------------------------------------------

#[test]
fn the_memory_charge_scales_as_documented() {
    use ferric_scf::hessian_df_k::rik_peak_bytes;
    // 3N A m^2 (L^x, Z^x) + 3N A^2 (V^x) dominate; spot-check the arithmetic
    // the reservation uses and the documented examples.
    let small = rik_peak_bytes(3, 24, 5, 113);
    let big = rik_peak_bytes(30, 300, 60, 1500);
    eprintln!("peak bytes: water cc-pVDZ/jkfit {small}, 30 atoms/300 bf/m=60/A=1500 {big}");
    assert!(small < 50 << 20);
    assert!(big > 5usize << 30, "the 30-atom example must exceed 5 GB");
}

// ---------------------------------------------------------------------------
// Accuracy as a function of the conditioning of the exchange metric
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow: accuracy-vs-cond(V) scan for the RI-K path, own-gradient FD per aux basis; measured 86 s for six bases (box load ~8)"]
fn accuracy_versus_metric_conditioning() {
    // Hypothesis: like RI-J, the RI-K Hessian does not degrade with cond(V) up
    // to the dropped-mode refusal. The reference (FD of ferric's own DF-K
    // gradient) is valid only when no metric mode lies in [1e-10, 1e-7), where
    // the gradient's Daleckii-Krein noise floor treats a kept mode as dropped;
    // a basis with such a mode is reported with its band count and NOT used as
    // an accuracy datum. A basis with a mode below 1e-10 is refused by the
    // Hessian (typed error), reported as such.
    if !has_deriv2() {
        return;
    }
    for aux in [
        "cc-pvdz-ri",
        "def2-universal-jkfit",
        "aug-cc-pvdz-rifit",
        "aug-cc-pvtz-rifit",
        "aug-cc-pvqz-rifit",
        "def2-qzvpp-rifit",
    ] {
        let t0 = std::time::Instant::now();
        let mol = water();
        let v = coulomb_metric_2c(Operator::coulomb(), &aux_at(&mol, aux)).unwrap();
        let (lam, _) = v.eigh(UPLO::Upper).unwrap();
        let (lmin, lmax) = (lam[0], lam[lam.len() - 1]);
        let band = lam.iter().filter(|&&l| (1e-10..1e-7).contains(&l)).count();
        let below = lam.iter().filter(|&&l| l < 1e-10).count();
        if below > 0 {
            eprintln!(
                "COND {aux}: lambda_min {lmin:.2e} cond {:.2e}: REFUSED ({below} dropped modes)",
                lmax / lmin
            );
            continue;
        }
        let s = setup(&mol, "6-31g", "", aux);
        let total = parts(&s).total();
        if band > 0 {
            eprintln!(
                "COND {aux}: lambda_min {lmin:.2e} cond {:.2e}: {band} mode(s) in the gradient's \
                 noise band; own-gradient reference INVALID, no accuracy datum",
                lmax / lmin
            );
            continue;
        }
        let fd = fd_hessian_richardson(&s.mol, 2e-3, |m| df_gradient_at(m, "6-31g", "", aux));
        eprintln!(
            "COND {aux}: lambda_min {lmin:.2e} lambda_max {lmax:.2e} cond {:.2e} miss vs own FD \
             {:.3e} ({:.1}s)",
            lmax / lmin,
            max_abs_diff(&total, &fd),
            t0.elapsed().as_secs_f64()
        );
    }
}

/// `max|skeleton(explicit inverse) - skeleton(V^{-1/2} V^{-1/2})|` and the
/// same for `K^x`, for the RI-K pieces at the converged density of `s`.
fn explicit_inverse_difference(s: &Setup) -> (f64, f64) {
    let route = s.rhf.df_jk.as_ref().unwrap();
    let x = occ_factor(s);
    let good = rik_terms(&s.mol, &s.prep, route, &x, &RikMutants::default()).unwrap();
    let bad = rik_terms(
        &s.mol,
        &s.prep,
        route,
        &x,
        &RikMutants {
            explicit_inverse: true,
            ..Default::default()
        },
    )
    .unwrap();
    let kx = good
        .first_order_k
        .iter()
        .zip(&bad.first_order_k)
        .fold(0.0f64, |m, (a, b)| m.max(max_abs_diff(a, b)));
    (max_abs_diff(&good.skeleton, &bad.skeleton), kx)
}

#[test]
#[ignore = "slow: benzene STO-3G/def2-universal-jkfit RI-K pieces; measured 8.2 s (box load ~8)"]
fn explicit_inverse_mutant_on_an_ill_conditioned_metric() {
    // Hypothesis: physics predicts no difference (same algebra); a numerical
    // artifact predicts a difference that grows with cond(V). The assertion is
    // RELATIVE to the water difference measured in the same run.
    if !has_deriv1() {
        return;
    }
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/benzene.xyz"
    );
    let benzene = Molecule::load_xyz(path).unwrap();
    let (sk, kx) =
        explicit_inverse_difference(&setup(&benzene, "sto-3g", "", "def2-universal-jkfit"));
    let (sk_w, kx_w) = explicit_inverse_difference(&setup(&water(), "6-31g", "", "cc-pvdz-ri"));
    eprintln!(
        "explicit inverse: benzene skeleton {sk:.3e} K^x {kx:.3e}; water skeleton {sk_w:.3e} K^x {kx_w:.3e}"
    );
    assert!(
        sk > 100.0 * sk_w.max(1e-14),
        "skeleton {sk:.3e} vs water {sk_w:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Beyond the warning threshold: more points, and the RI-J analogue
// ---------------------------------------------------------------------------

/// `(lambda_min, lambda_max, modes below 1e-10, modes in [1e-10, 1e-7))`.
fn spectrum_of(mol: &Molecule, aux: &str) -> (f64, f64, usize, usize) {
    let v = coulomb_metric_2c(Operator::coulomb(), &aux_at(mol, aux)).unwrap();
    let (lam, _) = v.eigh(UPLO::Upper).unwrap();
    (
        lam[0],
        lam[lam.len() - 1],
        lam.iter().filter(|&&l| l < 1e-10).count(),
        lam.iter().filter(|&&l| (1e-10..1e-7).contains(&l)).count(),
    )
}

#[test]
#[ignore = "survey: metric spectra of every bundled fitting basis on water; a few seconds"]
fn survey_metric_conditioning_of_bundled_fitting_bases() {
    for aux in [
        "cc-pvdz-ri",
        "def2-universal-jkfit",
        "aug-cc-pvdz-rifit",
        "aug-cc-pvtz-rifit",
        "aug-cc-pvqz-rifit",
        "cc-pvtz-rifit",
        "def2-svp-rifit",
        "def2-tzvp-rifit",
        "def2-tzvpp-rifit",
        "def2-qzvp-rifit",
        "def2-qzvpp-rifit",
    ] {
        let (lmin, lmax, dropped, band) = spectrum_of(&water(), aux);
        eprintln!(
            "SURVEY {aux}: lambda_min {lmin:.2e} cond {:.2e} dropped {dropped} band {band}",
            lmax / lmin
        );
    }
}

/// One scan row: the K Hessian against two Richardson references (h = 2e-3
/// and 4e-3: their difference is the reference's own noise scale), and the J
/// analogue (RI-J with the same fitting basis, exact K) against the same.
fn scan_row(aux: &str) {
    let t0 = std::time::Instant::now();
    let mol = water();
    let (lmin, lmax, dropped, band) = spectrum_of(&mol, aux);
    if dropped > 0 || band > 0 {
        eprintln!(
            "SCAN {aux}: cond {:.2e}: {dropped} dropped, {band} band -> refused/invalid reference",
            lmax / lmin
        );
        return;
    }
    let refs = |j: &str, k: &str| {
        let h_an = parts(&setup(&mol, "6-31g", j, k)).total();
        let f2 = fd_hessian_richardson(&mol, 2e-3, |m| df_gradient_at(m, "6-31g", j, k));
        let f4 = fd_hessian_richardson(&mol, 4e-3, |m| df_gradient_at(m, "6-31g", j, k));
        (
            max_abs_diff(&h_an, &f2),
            max_abs_diff(&h_an, &f4),
            max_abs_diff(&f2, &f4),
        )
    };
    let (k2, k4, kref) = refs("", aux);
    let (j2, j4, jref) = refs(aux, "");
    eprintln!(
        "SCAN {aux}: lambda_min {lmin:.2e} cond {:.2e} | K miss vs FD(2e-3) {k2:.2e}, FD(4e-3) \
         {k4:.2e}, FD refs differ {kref:.2e} | J analogue {j2:.2e}, {j4:.2e}, {jref:.2e} ({:.0}s)",
        lmax / lmin,
        t0.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "slow: RI-K accuracy above cond 1e8 with the RI-J analogue and two FD references; measured 124 s (box load ~8)"]
fn accuracy_above_the_warning_threshold() {
    // Hypothesis (physics): the Hessian is the derivative of the energy, so its
    // error is the FD reference's noise plus roundoff ~ eps*cond(V); the RI-J
    // analogue (full inverse, same metric) shares the roundoff but not the
    // K-specific amplification. Artifact: if K misses by much more than J and
    // than the spread between the two FD references, the K path loses accuracy
    // by itself above the threshold.
    if !has_deriv2() {
        return;
    }
    scan_row("aug-cc-pvqz-rifit");
}

/// A water fitting basis (`base` on every atom) with a near-duplicate s
/// function (exponent x (1 + eps)) added on each hydrogen.
fn twin_aux(mol: &Molecule, base: &str, eps: f64) -> PreparedBasis {
    use ferric_core::basis::Shell;
    let mut aux = basis::bundled(base).unwrap();
    let h = aux.shells.get_mut(&1).unwrap();
    let first = h.iter().find(|s| s.l == 0).unwrap().clone();
    let a = first.exponents[0];
    h.push(Shell {
        exponents: vec![a * (1.0 + eps)],
        coefficients: vec![(2.0 * a / std::f64::consts::PI).powf(0.75)],
        ..first
    });
    PreparedBasis::new(mol, &aux).unwrap()
}

/// `m` dense non-orthogonal columns over all basis functions.
fn dense_columns(n: usize, m: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, m), |(i, c)| {
        let t = (i + 3 * c) as f64;
        (0.8 + 0.5 * (1.1 * t + c as f64).sin()) / (1.0 + 0.12 * t)
    })
}

#[test]
#[ignore = "slow: K and J skeleton accuracy on twin-augmented fitting bases above cond 1e8; fixed-density FD of the DF gradient; measured 50 s (box load ~8)"]
fn skeleton_accuracy_on_near_dependent_fitting_bases() {
    // Fixed density, no SCF: the K skeleton against the FD of df_jk_gradient
    // (RI-K channel) at two steps, and the RI-J skeleton against the FD of the
    // J gradient with the SAME metric. Hypothesis (physics): both are exact
    // derivatives, so both agree with their references to the reference's own
    // roundoff, which grows with cond(V) identically for J and K (same V, same
    // 3c integrals). Artifact: K misses by much more than J at equal cond (a
    // K-specific amplification: M = W W built from the eigenvectors).
    use ferric_scf::hessian::rij_support::{rij_terms_with_aux, rik_terms_with_aux, RijMutants};
    if !has_deriv1() {
        return;
    }
    let mol = water();
    let prep = prep_at(&mol, "6-31g");
    let n = prep.nbasis();
    let x = dense_columns(n, 5);
    let d = x.dot(&x.t());
    let op = Operator::coulomb();
    let alpha = ferric_scf::hessian_df_k::RIK_ALPHA;
    for base in ["cc-pvdz-ri", "def2-universal-jkfit"] {
        for eps in [1e-1, 3e-2, 1e-2, 3e-3] {
            let v = coulomb_metric_2c(op, &twin_aux(&mol, base, eps)).unwrap();
            let (lam, _) = v.eigh(UPLO::Upper).unwrap();
            let (lmin, cond) = (lam[0], lam[lam.len() - 1] / lam[0]);
            let below = lam.iter().filter(|&&l| l < 1e-10).count();
            let band = lam.iter().filter(|&&l| (1e-10..1e-7).contains(&l)).count();
            if below > 0 || band > 0 {
                eprintln!(
                    "TWIN {base} eps {eps:.0e}: lambda_min {lmin:.2e} cond {cond:.2e}: {below} \
                     dropped, {band} band -> refused/invalid reference"
                );
                continue;
            }
            let k = rik_terms_with_aux(
                3,
                &prep,
                twin_aux(&mol, base, eps),
                op,
                usize::MAX,
                &x,
                &RikMutants::default(),
            )
            .unwrap();
            let j = rij_terms_with_aux(
                3,
                &prep,
                twin_aux(&mol, base, eps),
                op,
                usize::MAX,
                &d,
                &RijMutants::default(),
            )
            .unwrap();
            let fd = |h: f64, kchan: bool| {
                fd_hessian(&mol, h, |m| {
                    let p = prep_at(m, "6-31g");
                    let aux = twin_aux(m, base, eps);
                    let ch = [DfKChannel { density: &d, alpha }];
                    if kchan {
                        df_jk_gradient(3, &p, &aux, op, None, &ch, usize::MAX).unwrap()
                    } else {
                        df_jk_gradient(3, &p, &aux, op, Some(&d), &[], usize::MAX).unwrap()
                    }
                })
            };
            let (k1, k3) = (fd(1e-4, true), fd(3e-4, true));
            let (j1, j3) = (fd(1e-4, false), fd(3e-4, false));
            eprintln!(
                "TWIN {base} eps {eps:.0e}: lambda_min {lmin:.2e} cond {cond:.2e} | K skeleton miss \
                 {:.2e} (h 1e-4), {:.2e} (h 3e-4), FD spread {:.2e} | J skeleton miss {:.2e}, \
                 {:.2e}, FD spread {:.2e}",
                max_abs_diff(&k.skeleton, &k1),
                max_abs_diff(&k.skeleton, &k3),
                max_abs_diff(&k1, &k3),
                max_abs_diff(&j.skeleton, &j1),
                max_abs_diff(&j.skeleton, &j3),
                max_abs_diff(&j1, &j3),
            );
        }
    }
}
