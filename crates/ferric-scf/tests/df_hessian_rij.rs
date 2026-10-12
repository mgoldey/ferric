//! Analytic RHF Hessian for RI-J with exact K (`ferric_scf::hessian_df`)
//! against INDEPENDENT constructions.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test df_hessian_rij
//! ```
//!
//! # What is compared to what
//!
//! * First-derivative pieces of the fit (`d_x`, `V_x c`, `∂_x c`) vs central
//!   differences of `d(R) = V c`, of `V(R) c₀` and of `c(R)` at really
//!   displaced geometries (no SCF: fixed AO density).
//! * Each skeleton piece vs a central difference of the matching piece of
//!   the first derivative at FIXED density (`E_xy` = d/dy of `E_x`): the 3c
//!   piece (`d_x·c₀` at displaced geometry), the 2c piece (`−½ c₀·V_x c₀`),
//!   the cross piece (`Z_x · ∂_y c`), and the J skeleton as a whole vs a
//!   central difference of `df_jk_gradient` (the validated DF gradient).
//! * `J^x` (the AO matrices of the first-order Fock) vs a central difference
//!   of the SCF's own `DfJ` Coulomb matrix at displaced geometries.
//! * The TOTAL Hessian vs a central difference (h and h/2, Richardson) of
//!   ferric's own analytic DF gradient with the SCF re-converged at every
//!   displaced geometry.
//! * Translational invariance, and rotational invariance as
//!   `H·(e_a × r) = e_a × g` (valid away from stationary points).
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * Right: every piece matches its finite difference at the FD floor
//!   (~1e-7 on pieces without SCF noise, ~1e-6 with re-converged SCFs); the
//!   exact-J Hessian misses the DF Hessian by the fitting error.
//! * Skeleton wrong (a dropped term or a wrong weight): the piece test and
//!   the full test fail together while `J^x` and the first-derivative tests
//!   stay green.
//! * `J^x` wrong (missing aux-fit part): the piece tests stay green, the
//!   `J^x` test and the full test fail, and the CPHF response asymmetry grows.
//! * FD reference broken (a displaced SCF on another state): the full-FD test
//!   fails column-wise while all analytic-piece tests are green.
//! * An integral artifact common to the stencil and the oracle cannot be seen
//!   by the fixed-density pieces; the displaced-geometry gradient FD and the
//!   PySCF comparison are what see it.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ferric_scf::df_gradient::df_jk_gradient;
use ferric_scf::df_j::DfJ;
use ferric_scf::fock::JBuilder;
use ferric_scf::gradient::rhf_gradient;
use ferric_scf::hessian::rij_support::{
    rhf_hessian_parts_mutated, rij_first, rij_terms, RijFirst, RijMutants,
};
use ferric_scf::hessian::{rhf_hessian, rhf_hessian_parts, RhfHessianParts};
use ferric_scf::ks_gradient::twoelectron_k_gradient;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array1, Array2};

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

/// The RI-J pieces that need only first-derivative engines (a stencil of
/// their blocks) run on a first-derivative libint too.
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

fn scf_config(aux: &str) -> RhfConfig {
    RhfConfig {
        max_iter: 200,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        df_j_aux: Some(aux.to_string()),
        df_k_aux: Some(String::new()),
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

fn setup(mol: &Molecule, basis_name: &str, aux: &str) -> Setup {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = scf_config(aux);
    let rhf = solve_rhf(&ParallelContext::default(), mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged, "{basis_name}/{aux}: SCF did not converge");
    assert!(rhf.df_jk.as_ref().is_some_and(|r| r.j_aux.is_some()));
    Setup {
        mol: mol.clone(),
        prep,
        bounds,
        rhf,
        cfg,
        basis: basis_name.to_string(),
        aux: aux.to_string(),
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

/// Central-difference Hessian: column b = (g(x_b + h) - g(x_b - h)) / 2h.
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

/// Richardson (h, h/2) combination: O(h^4).
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

/// Fixed-density first-derivative pieces at a displaced geometry.
fn first_at(s: &Setup, m: &Molecule, c_fixed: Option<&Array1<f64>>) -> RijFirst {
    let prep = prep_at(m, &s.basis);
    rij_first(
        m,
        &prep,
        s.rhf.df_jk.as_ref().unwrap(),
        s.rhf.density_r(),
        c_fixed,
    )
    .unwrap()
}

/// `d(R) = V(R) c(R)` at fixed density, as a `3N`-independent vector.
fn d_at(s: &Setup, m: &Molecule) -> Array1<f64> {
    let f = first_at(s, m, None);
    let v = coulomb_metric_2c(Operator::coulomb(), &aux_at(m, &s.aux)).unwrap();
    v.dot(&f.c)
}

const H_PIECE: f64 = 1e-4;
/// Step for finite differences of the fit vector `c = V^-1 d`. The solve
/// amplifies integral roundoff by cond(V) (jkfit metrics: 1e6-1e8), so the
/// central difference has a noise floor ~ 1/h^2 and a truncation floor ~ h^2;
/// measured at STO-3G/def2-universal-jkfit: 6.2e-6 (3e-5), 7.5e-7 (1e-4),
/// 4.4e-8 (3e-4), 2.6e-7 (1e-3), 2.1e-6 (3e-3) against the analytic
/// derivative. 3e-4 sits at the minimum.
const H_C: f64 = 3e-4;

/// `dc/dx` by central difference of the fit vector (step `H_C`).
fn dc_at(s: &Setup, x: usize) -> Array1<f64> {
    let (mp, mm) = (
        displaced(&s.mol, x / 3, x % 3, H_C),
        displaced(&s.mol, x / 3, x % 3, -H_C),
    );
    (&first_at(s, &mp, None).c - &first_at(s, &mm, None).c) / (2.0 * H_C)
}

// ---------------------------------------------------------------------------
// First-derivative pieces of the fit
// ---------------------------------------------------------------------------

fn check_first_pieces(basis_name: &str, aux: &str) {
    if !has_deriv1() {
        return;
    }
    let s = setup(&water(), basis_name, aux);
    let f0 = first_at(&s, &s.mol, None);
    let n3 = 3 * s.mol.atoms.len();
    let naux = f0.c.len();
    let mut dx_fd = Array2::<f64>::zeros((n3, naux));
    let mut vc_fd = Array2::<f64>::zeros((n3, naux));
    let mut dc_fd = Array2::<f64>::zeros((n3, naux));
    let c0 = f0.c.clone();
    for x in 0..n3 {
        let (mp, mm) = (
            displaced(&s.mol, x / 3, x % 3, H_PIECE),
            displaced(&s.mol, x / 3, x % 3, -H_PIECE),
        );
        let (dp, dm) = (d_at(&s, &mp), d_at(&s, &mm));
        dx_fd.row_mut(x).assign(&((&dp - &dm) / (2.0 * H_PIECE)));
        // V_x c0 by central difference of V(R) c0.
        let vp = coulomb_metric_2c(Operator::coulomb(), &aux_at(&mp, aux)).unwrap();
        let vm = coulomb_metric_2c(Operator::coulomb(), &aux_at(&mm, aux)).unwrap();
        vc_fd
            .row_mut(x)
            .assign(&((vp.dot(&c0) - vm.dot(&c0)) / (2.0 * H_PIECE)));
        dc_fd.row_mut(x).assign(&dc_at(&s, x));
    }
    let tol = 5e-7;
    assert_close(&format!("{basis_name}/{aux}: d_x"), &f0.dx, &dx_fd, tol);
    assert_close(&format!("{basis_name}/{aux}: V_x c"), &f0.vc, &vc_fd, tol);
    // dc/dx carries the cond(V) noise of the finite-difference reference
    // itself (H_C doc): measured 1.2e-6 at STO-3G/def2-universal-jkfit, 5.6e-8
    // with the 6-31G/cc-pVDZ-RI fit. The analytic derivative is checked at
    // 4e-9 by the cross-piece test and 4e-11 by the full-Hessian test.
    assert_close(&format!("{basis_name}/{aux}: dc/dx"), &f0.y, &dc_fd, 3e-6);
    assert!(max_abs(&f0.y) > 1e-2, "vacuous: dc/dx ~ 0");
}

#[test]
fn first_derivative_pieces_match_fd_sto3g() {
    check_first_pieces("sto-3g", "def2-universal-jkfit");
}

#[test]
fn first_derivative_pieces_match_fd_631g() {
    check_first_pieces("6-31g", "cc-pvdz-ri");
}

// ---------------------------------------------------------------------------
// Skeleton pieces vs fixed-density finite differences
// ---------------------------------------------------------------------------

/// `E_xy` pieces = d/dy of the matching piece of `E_x` at fixed density.
fn check_skeleton_pieces(basis_name: &str, aux: &str) {
    if !has_deriv1() {
        return;
    }
    let s = setup(&water(), basis_name, aux);
    let route = s.rhf.df_jk.as_ref().unwrap();
    let d = s.rhf.density_r();
    let terms = rij_terms(&s.mol, &s.prep, route, d, &RijMutants::default()).unwrap();
    let n3 = 3 * s.mol.atoms.len();
    let c0 = terms.first.c.clone();
    let mut three = Array2::<f64>::zeros((n3, n3));
    let mut metric = Array2::<f64>::zeros((n3, n3));
    let mut cross = Array2::<f64>::zeros((n3, n3));
    let z0 = &terms.first.dx - &terms.first.vc;
    for y in 0..n3 {
        let (mp, mm) = (
            displaced(&s.mol, y / 3, y % 3, H_PIECE),
            displaced(&s.mol, y / 3, y % 3, -H_PIECE),
        );
        let (fp, fm) = (first_at(&s, &mp, Some(&c0)), first_at(&s, &mm, Some(&c0)));
        let dc = dc_at(&s, y);
        for x in 0..n3 {
            three[(x, y)] = (fp.dx.row(x).dot(&c0) - fm.dx.row(x).dot(&c0)) / (2.0 * H_PIECE);
            metric[(x, y)] =
                -0.5 * (fp.vc.row(x).dot(&c0) - fm.vc.row(x).dot(&c0)) / (2.0 * H_PIECE);
            cross[(x, y)] = z0.row(x).dot(&dc);
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

    // The whole J skeleton vs d/dy of the validated DF gradient at fixed D.
    let natoms = s.mol.atoms.len();
    let fd = fd_hessian(&s.mol, H_PIECE, |m| {
        let prep = prep_at(m, basis_name);
        df_jk_gradient(
            natoms,
            &prep,
            &aux_at(m, aux),
            Operator::coulomb(),
            Some(d),
            &[],
            route.budget_bytes,
        )
        .unwrap()
    });
    assert_close(&format!("{tag}: J skeleton"), &terms.skeleton, &fd, tol);

    assert!(
        terms.skeleton_asymmetry < 1e-9,
        "raw skeleton asymmetry {:.3e}",
        terms.skeleton_asymmetry
    );

    // The exact-K skeleton needs the four-centre second derivatives.
    if !has_deriv2() {
        return;
    }
    // The exact-K skeleton: total two-electron term minus the RI-J piece.
    let p = parts(&s);
    let k_skel = &p.two_electron - &terms.skeleton;
    let fd_k = fd_hessian(&s.mol, H_PIECE, |m| {
        let prep = prep_at(m, basis_name);
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        twoelectron_k_gradient(&prep, Operator::coulomb(), &bounds, d, 1.0).unwrap()
    });
    assert_close(&format!("{tag}: K skeleton"), &k_skel, &fd_k, 5e-7);
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
// J^x (first-order Fock, Coulomb part) vs the SCF's own DfJ
// ---------------------------------------------------------------------------

fn check_first_order_j(basis_name: &str, aux: &str) {
    if !has_deriv1() {
        return;
    }
    let s = setup(&water(), basis_name, aux);
    let route = s.rhf.df_jk.as_ref().unwrap();
    let d = s.rhf.density_r();
    let terms = rij_terms(&s.mol, &s.prep, route, d, &RijMutants::default()).unwrap();
    let j_at = |m: &Molecule| {
        let prep = prep_at(m, basis_name);
        let mut df = DfJ::new(Operator::coulomb(), &prep, &aux_at(m, aux), usize::MAX).unwrap();
        let mut j = Array2::<f64>::zeros(d.dim());
        df.build(d, &mut j).unwrap();
        j
    };
    let n3 = 3 * s.mol.atoms.len();
    for x in 0..n3 {
        let jp = j_at(&displaced(&s.mol, x / 3, x % 3, H_PIECE));
        let jm = j_at(&displaced(&s.mol, x / 3, x % 3, -H_PIECE));
        let fd = (jp - jm) / (2.0 * H_PIECE);
        assert_close(
            &format!("{basis_name}/{aux}: J^x, x = {x}"),
            &terms.first_order_j[x],
            &fd,
            5e-7,
        );
        assert!(max_abs(&fd) > 1e-3, "vacuous: J^x ~ 0 at x = {x}");
    }
}

#[test]
fn first_order_j_matches_dfj_fd_sto3g() {
    check_first_order_j("sto-3g", "def2-universal-jkfit");
}

#[test]
fn first_order_j_matches_dfj_fd_631g() {
    check_first_order_j("6-31g", "cc-pvdz-ri");
}

// ---------------------------------------------------------------------------
// Total vs FD of ferric's own DF gradient, invariances
// ---------------------------------------------------------------------------

fn df_gradient_at(m: &Molecule, basis_name: &str, aux: &str) -> Array2<f64> {
    let sm = setup(m, basis_name, aux);
    rhf_gradient(m, &sm.prep, Operator::coulomb(), &sm.bounds, &sm.rhf, None).unwrap()
}

fn check_full(mol: &Molecule, basis_name: &str, aux: &str, tol: f64) -> (f64, f64) {
    let s = setup(mol, basis_name, aux);
    let p = parts(&s);
    let total = p.total();
    let fd = fd_hessian_richardson(&s.mol, 2e-3, |m| df_gradient_at(m, basis_name, aux));
    eprintln!(
        "{basis_name}/{aux}: CPHF iterations {:?}, residual {:.2e}, response asym {:.2e}",
        p.cphf_iterations, p.cphf_max_residual, p.response_asymmetry
    );
    assert!(
        max_abs_diff(&p.skeleton(), &fd) > 100.0 * tol,
        "skeleton alone matches FD: the full test cannot see the response"
    );
    let miss = assert_close(
        &format!("{basis_name}/{aux}: full Hessian"),
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

#[test]
fn full_hessian_matches_fd_of_df_gradient_sto3g() {
    if !has_deriv2() {
        return;
    }
    check_full(&water(), "sto-3g", "def2-universal-jkfit", BAR_OWN);
}

#[test]
fn full_hessian_matches_fd_of_df_gradient_631g() {
    if !has_deriv2() {
        return;
    }
    check_full(&water(), "6-31g", "cc-pvdz-ri", BAR_OWN);
}

#[test]
fn translational_and_rotational_invariance() {
    if !has_deriv2() {
        return;
    }
    let s = setup(&water(), "6-31g", "cc-pvdz-ri");
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
    // H (e_a x r) = e_a x g (the gradient rotates with the molecule).
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
// Mutants (production-wired)
// ---------------------------------------------------------------------------

fn mutant_total(s: &Setup, m: &RijMutants) -> Array2<f64> {
    rhf_hessian_parts_mutated(
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

// ---------------------------------------------------------------------------
// Exactness anchor: an auxiliary basis that spans the density's products
// ---------------------------------------------------------------------------
//
// Hypotheses. If the RI-J Hessian is right: for a density confined to the
// hydrogens' own functions (D nonzero only in the H1 x H1 and H2 x H2 blocks)
// and an auxiliary basis that contains, on each hydrogen, an s function for
// every pairwise sum of the orbital exponents, the product density
// rho = sum D chi chi is in the auxiliary span for EVERY geometry (orbital and auxiliary functions move
// rigidly together), so E_J^DF(R) == E_J(R) identically and the fitted
// skeleton and J^x equal the exact four-centre ones up to the stencil and
// Cholesky roundoff (~1e-9). If my implementation is broken (a missing
// term, a wrong weight) the equality fails by the size of the term. The two
// hypotheses give different outcomes, so the anchor can distinguish them. A
// self-consistent artifact (a bug shared by RI-J and the exact path) cannot
// hide here: the exact skeleton and J^x are the validated four-centre ones.
// Teeth: with the ordinary cc-pVDZ-RI functions on H1 the same comparison
// misses by the fitting error, and every mutant misses.

fn exact_span_aux(mol: &Molecule, orbital_basis: &str) -> PreparedBasis {
    exact_span_aux_with_twin(mol, orbital_basis, None)
}

/// [`exact_span_aux`] plus, optionally, a near-duplicate of a middle function
/// on each hydrogen (exponent x (1 + eps)). The span (and so the exact answer)
/// is unchanged; only cond(V) grows.
fn exact_span_aux_with_twin(
    mol: &Molecule,
    orbital_basis: &str,
    twin: Option<f64>,
) -> PreparedBasis {
    use ferric_core::basis::Shell;
    let orbital = basis::bundled(orbital_basis).unwrap();
    let mut exps: Vec<f64> = Vec::new();
    for sh in orbital.for_element(1).unwrap() {
        exps.extend(sh.exponents.iter().copied());
    }
    let mut sums: Vec<f64> = Vec::new();
    for (i, a) in exps.iter().enumerate() {
        for b in &exps[i..] {
            sums.push(a + b);
        }
    }
    sums.sort_by(|x, y| x.partial_cmp(y).unwrap());
    sums.dedup_by(|x, y| (*x - *y).abs() < 1e-9);
    let mut aux = basis::bundled("cc-pvdz-ri").unwrap();
    let shells: Vec<Shell> = sums
        .iter()
        .map(|&a| Shell {
            l: 0,
            pure: false,
            exponents: vec![a],
            coefficients: vec![(2.0 * a / std::f64::consts::PI).powf(0.75)],
        })
        .collect();
    let mut shells = shells;
    if let Some(eps) = twin {
        let a = sums[sums.len() / 2] * (1.0 + eps);
        shells.push(Shell {
            l: 0,
            pure: false,
            exponents: vec![a],
            coefficients: vec![(2.0 * a / std::f64::consts::PI).powf(0.75)],
        });
    }
    aux.shells.insert(1, shells);
    PreparedBasis::new(mol, &aux).unwrap()
}

/// Block-diagonal `D` over the two hydrogens' own functions (zero on the
/// oxygen and on the H1-H2 cross block; arbitrary, non-SCF). Each hydrogen's
/// density is on-site, so its products are exactly representable by the
/// on-site auxiliary functions; the H1-H2 Coulomb interaction keeps the
/// Hessian nonzero (a density on ONE atom has an identically zero skeleton).
fn hydrogens_density(prep: &PreparedBasis) -> Array2<f64> {
    let n = prep.nbasis();
    let mut d = Array2::<f64>::zeros((n, n));
    let (offs, dims, at) = (
        prep.shell_offsets(),
        prep.shell_dims(),
        prep.shell_to_atom(),
    );
    for atom in [1usize, 2] {
        let idx: Vec<usize> = (0..prep.nshells())
            .filter(|&sh| at[sh] == atom)
            .flat_map(|sh| offs[sh]..offs[sh] + dims[sh])
            .collect();
        for (a, &i) in idx.iter().enumerate() {
            for (b, &j) in idx.iter().enumerate() {
                let w = if atom == 1 { 1.3 } else { 0.9 };
                d[(i, j)] = if a == b {
                    w
                } else {
                    0.4 / (1.0 + (a + b) as f64)
                };
            }
        }
    }
    d
}

fn anchor_misses(orbital_basis: &str, exact_aux: bool, m: &RijMutants) -> (f64, f64, f64, f64) {
    use ferric_scf::hessian::rij_support::{exact_j_first_order, exact_j_skeleton};
    let mol = water();
    let prep = prep_at(&mol, orbital_basis);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let natoms = mol.atoms.len();
    let d = hydrogens_density(&prep);
    let dfbs = if exact_aux {
        exact_span_aux(&mol, orbital_basis)
    } else {
        aux_at(&mol, "cc-pvdz-ri")
    };
    let rij = ferric_scf::hessian::rij_support::rij_terms_with_aux(
        natoms,
        &prep,
        dfbs,
        op,
        usize::MAX,
        &d,
        m,
    )
    .unwrap();
    let skel = exact_j_skeleton(&prep, op, &bounds, natoms, &d).unwrap();
    let jx = exact_j_first_order(&prep, op, &bounds, natoms, &d).unwrap();
    let jx_miss = rij
        .first_order_j
        .iter()
        .zip(&jx)
        .fold(0.0f64, |m, (a, b)| m.max(max_abs_diff(a, b)));
    let jx_scale = jx.iter().fold(0.0f64, |m, a| m.max(max_abs(a)));
    (
        max_abs_diff(&rij.skeleton, &skel),
        max_abs(&skel),
        jx_miss,
        jx_scale,
    )
}

fn check_trivial_limit(orbital_basis: &str) {
    if !has_deriv2() {
        return;
    }
    let bar = 1e-8;
    let (sk, sk_scale, jx, jx_scale) = anchor_misses(orbital_basis, true, &RijMutants::default());
    eprintln!(
        "{orbital_basis} exact-span aux: skeleton miss {sk:.3e} (scale {sk_scale:.3e}), \
         J^x miss {jx:.3e} (scale {jx_scale:.3e})"
    );
    assert!(sk_scale > 1e-2 && jx_scale > 1e-1, "vacuous anchor");
    assert!(sk < bar, "skeleton miss {sk:.3e}");
    assert!(jx < bar, "J^x miss {jx:.3e}");

    // Teeth 1: the ordinary RI functions on H1 do not span, so the same
    // comparison misses by the fitting error.
    let (sk2, _, jx2, _) = anchor_misses(orbital_basis, false, &RijMutants::default());
    eprintln!("{orbital_basis} cc-pVDZ-RI aux: skeleton miss {sk2:.3e}, J^x miss {jx2:.3e}");
    assert!(sk2 > 10.0 * bar && jx2 > 10.0 * bar, "anchor has no teeth");

    // Teeth 2: production-wired mutants break the equality.
    let d = RijMutants::default();
    let mutants = [
        (
            "drop -1/2 c V_xy c",
            RijMutants {
                drop_metric_second: true,
                ..d
            },
        ),
        (
            "drop 3c-2c cross terms",
            RijMutants {
                drop_cross: true,
                ..d
            },
        ),
    ];
    for (name, m) in mutants {
        let (sk3, _, jx3, _) = anchor_misses(orbital_basis, true, &m);
        eprintln!("  mutant {name}: skeleton miss {sk3:.3e}, J^x miss {jx3:.3e}");
        assert!(
            sk3.max(jx3) > 10.0 * bar,
            "mutant {name} survives the exactness anchor"
        );
    }

    // BLIND SPOT, asserted so it cannot be mistaken for coverage: the
    // auxiliary-fit part of J^x, (P|mu nu) dc_P/dx, is identically zero in this
    // configuration. Each hydrogen's density and its on-site auxiliary
    // functions translate rigidly together, so the exact fit coefficients c_P
    // do not change under any nuclear displacement. The anchor therefore
    // cannot see `omit_aux_jx`; the DfJ finite-difference test of J^x
    // (first_order_j_matches_dfj_fd_*) and the full-Hessian test kill it.
    let (sk4, _, jx4, _) = anchor_misses(
        orbital_basis,
        true,
        &RijMutants {
            omit_aux_jx: true,
            ..d
        },
    );
    eprintln!("  omit aux-fit J^x (blind here): skeleton {sk4:.3e}, J^x {jx4:.3e}");
    assert!(sk4.max(jx4) < bar, "the aux-fit part is expected to vanish");
}

#[test]
fn trivial_limit_exact_span_sto3g() {
    check_trivial_limit("sto-3g");
}

#[test]
fn trivial_limit_exact_span_631g() {
    check_trivial_limit("6-31g");
}

// ---------------------------------------------------------------------------
// Mutation suite
// ---------------------------------------------------------------------------
//
// BAR_OWN is the agreement bar of the production Hessian against the central
// FD (Richardson, h = 2e-3 / 1e-3) of ferric's own analytic DF gradient with
// re-converged SCFs. Production floors measured 2026-10-11 (release): water
// STO-3G/jkfit 4.4e-11, 6-31G/cc-pVDZ-RI 1.6e-10, cc-pVDZ/cc-pVDZ-RI 9.1e-11,
// def2-SVP/jkfit 3.0e-9 (worst), NH3 cc-pVDZ/cc-pVDZ-RI 1.1e-9. The closest
// mutant is the stencil step x10 at 4.0e-7, which must exceed 10 bars. The bar
// 1e-8 is 3.3x above the worst production floor and 4x below that mutant / 10.
// Every mutant must miss the production Hessian by more than 10 bars.
const BAR_OWN: f64 = 1e-8;

fn mutants_of(stencil: ferric_scf::hessian_df_stencil::Variant) -> Vec<(&'static str, RijMutants)> {
    use ferric_scf::hessian_df_stencil::StencilConfig;
    let d = RijMutants::default();
    let st = |v| StencilConfig {
        variant: v,
        ..Default::default()
    };
    vec![
        (
            "drop -1/2 c V_xy c",
            RijMutants {
                drop_metric_second: true,
                ..d
            },
        ),
        (
            "drop 3c-2c cross terms",
            RijMutants {
                drop_cross: true,
                ..d
            },
        ),
        (
            "omit aux-fit part of J^x",
            RijMutants {
                omit_aux_jx: true,
                ..d
            },
        ),
        (
            "Coulomb metric swapped for erfc(0.4)",
            RijMutants {
                swap_metric: true,
                ..d
            },
        ),
        (
            "stencil step x10",
            RijMutants {
                stencil: st(ferric_scf::hessian_df_stencil::Variant::Step10),
                ..d
            },
        ),
        (
            "one-shell displacement",
            RijMutants {
                stencil: st(stencil),
                ..d
            },
        ),
    ]
}

#[test]
fn every_wired_mutant_misses_by_more_than_ten_bars() {
    if !has_deriv2() {
        return;
    }
    let s = setup(&water(), "6-31g", "cc-pvdz-ri");
    let prod = parts(&s).total();
    let again = mutant_total(&s, &RijMutants::default());
    assert_eq!(
        max_abs_diff(&prod, &again),
        0.0,
        "production is not reproducible"
    );
    for (name, m) in mutants_of(ferric_scf::hessian_df_stencil::Variant::SingleShell) {
        let miss = max_abs_diff(&mutant_total(&s, &m), &prod);
        eprintln!(
            "MUTANT {name}: miss {miss:.3e} ({:.1e} bars)",
            miss / BAR_OWN
        );
        assert!(
            miss > 10.0 * BAR_OWN,
            "mutant {name} misses by only {miss:.3e}"
        );
    }
}

#[test]
fn explicit_inverse_mutant_is_measured_not_claimed() {
    // At the conditioning of a water cc-pVDZ-RI metric (cond ~ 1e4) the
    // explicit inverse is an identity to roundoff: BLIND, not counted. The
    // ignored benzene test below measures it where it matters.
    if !has_deriv2() {
        return;
    }
    let s = setup(&water(), "6-31g", "cc-pvdz-ri");
    let prod = parts(&s).total();
    let miss = max_abs_diff(
        &mutant_total(
            &s,
            &RijMutants {
                explicit_inverse: true,
                ..Default::default()
            },
        ),
        &prod,
    );
    eprintln!("explicit-inverse mutant, water 6-31G/cc-pVDZ-RI: miss {miss:.3e}");
    assert!(
        miss < BAR_OWN,
        "explicit inverse became detectable here: {miss:.3e}"
    );
}

#[test]
fn swapping_dxd_and_dyd_is_an_identity() {
    // Z_x^T M Z_y is symmetric in (x, y) because M is symmetric, so the
    // "swap d_x d and d_y d" mutant transposes a symmetric matrix. Not counted.
    if !has_deriv2() {
        return;
    }
    let s = setup(&water(), "6-31g", "cc-pvdz-ri");
    let terms = rij_terms(
        &s.mol,
        &s.prep,
        s.rhf.df_jk.as_ref().unwrap(),
        s.rhf.density_r(),
        &RijMutants::default(),
    )
    .unwrap();
    let asym = max_abs_diff(&terms.cross, &terms.cross.t().to_owned());
    eprintln!(
        "cross piece asymmetry {asym:.3e} (scale {:.3e}); raw skeleton asymmetry {:.3e}",
        max_abs(&terms.cross),
        terms.skeleton_asymmetry
    );
    assert!(asym < 1e-12 * max_abs(&terms.cross).max(1.0));
}

#[test]
fn mutated_aux_jx_fails_the_dfj_finite_difference_test() {
    // The exactness anchor is blind to the aux-fit part of J^x (see there);
    // this is the test that kills the mutant.
    if !has_deriv2() {
        return;
    }
    let s = setup(&water(), "6-31g", "cc-pvdz-ri");
    let route = s.rhf.df_jk.as_ref().unwrap();
    let d = s.rhf.density_r();
    let good = rij_terms(&s.mol, &s.prep, route, d, &RijMutants::default()).unwrap();
    let bad = rij_terms(
        &s.mol,
        &s.prep,
        route,
        d,
        &RijMutants {
            omit_aux_jx: true,
            ..Default::default()
        },
    )
    .unwrap();
    let worst = good
        .first_order_j
        .iter()
        .zip(&bad.first_order_j)
        .fold(0.0f64, |m, (a, b)| m.max(max_abs_diff(a, b)));
    eprintln!("aux-fit part of J^x: max element {worst:.3e} (FD bar 5e-7)");
    assert!(worst > 10.0 * 5e-7);
}

// ---------------------------------------------------------------------------
// Refusals and availability
// ---------------------------------------------------------------------------

#[test]
fn unsupported_density_fitted_configurations_are_refused() {
    use ferric_scf::hessian::{
        analytic_hessian_available, analytic_uhf_hessian_available, rhf_hessian_preflight,
        uhf_hessian_preflight,
    };
    let s = setup(&water(), "sto-3g", "def2-universal-jkfit");
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let hess = |rhf: &ScfResult, cfg: &RhfConfig| {
        rhf_hessian(&ctx, &s.mol, &s.prep, op, &s.bounds, rhf, cfg)
    };
    // RI-K requested in the configuration is supported for closed-shell RHF
    // (tests/df_hessian_rik.rs) and refused for open-shell references.
    let ri_k = RhfConfig {
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..scf_config("def2-universal-jkfit")
    };
    rhf_hessian_preflight(op, &ri_k).expect("RHF RI-J + RI-K preflight");
    let e = uhf_hessian_preflight(op, &ri_k).unwrap_err().to_string();
    eprintln!("UHF RI-K config: {e}");
    assert!(e.contains("RI"), "{e}");
    // A route that fitted exchange although the configuration did not.
    let mut with_k = s.rhf.clone();
    let route = with_k.df_jk.as_mut().unwrap();
    route.k_aux = Some("def2-universal-jkfit".into());
    let e = hess(&with_k, &s.cfg).unwrap_err().to_string();
    eprintln!("RI-K route: {e}");
    assert!(
        e.contains("RI-K") && e.contains("differs from the configuration"),
        "{e}"
    );
    // Range-separated fitted exchange in the route.
    let mut rsh = s.rhf.clone();
    rsh.df_jk.as_mut().unwrap().rsh_k = Some(("def2-universal-jkfit".into(), 0.3));
    assert!(hess(&rsh, &s.cfg)
        .unwrap_err()
        .to_string()
        .contains("range-separated"));
    // Open-shell reference with RI-J.
    let e = uhf_hessian_preflight(op, &s.cfg).unwrap_err().to_string();
    eprintln!("UHF RI-J config: {e}");
    assert!(e.contains("RI-J"), "{e}");
    // A result whose aux differs from the configuration's.
    let other = scf_config("cc-pvdz-ri");
    let e = hess(&s.rhf, &other).unwrap_err().to_string();
    eprintln!("route/config mismatch: {e}");
    assert!(e.contains("differs from the configuration"), "{e}");
    // RI-J with an external potential is untested and refused.
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

    // Negative controls: the supported configuration is NOT refused anywhere.
    rhf_hessian_preflight(op, &s.cfg).expect("RI-J + exact K preflight");
    if has_deriv2() {
        hess(&s.rhf, &s.cfg).expect("RI-J + exact K must be supported");
        analytic_hessian_available(&s.mol, &s.prep, op, &s.cfg).expect("available");
    }
    // UHF probe still refuses the RI-J config.
    assert!(analytic_uhf_hessian_available(&s.mol, &s.prep, op, &s.cfg).is_err());
}

#[test]
fn df_off_configurations_take_the_exact_path() {
    // df_j_aux unset and df_j_aux = "" are the same exact SCF and the same
    // Hessian, bit for bit (the RI-J code is not entered). The one-time
    // comparison against origin/main is recorded in the PR description.
    if !has_deriv2() {
        return;
    }
    let mol = water();
    let prep = prep_at(&mol, "sto-3g");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let hess = |df_j: Option<String>| {
        let cfg = RhfConfig {
            max_iter: 200,
            energy_conv: 1e-12,
            density_conv: 1e-10,
            df_j_aux: df_j,
            df_k_aux: Some(String::new()),
            ..Default::default()
        };
        let rhf = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
        assert!(rhf.df_jk.is_none(), "exact SCF must carry no DF route");
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

// ---------------------------------------------------------------------------
// Larger systems (ignored: measured wall times, release, one run each)
// ---------------------------------------------------------------------------

const NH3_DISTORTED: &str = "4\nNH3 distorted\n\
N      0.010000     0.020000     0.100000\n\
H      0.020000     0.950000    -0.300000\n\
H      0.830000    -0.470000    -0.320000\n\
H     -0.850000    -0.450000    -0.280000\n";

#[test]
#[ignore = "slow: water cc-pVDZ/cc-pVDZ-RI own-gradient FD; measured 28 s (4 threads, box load ~10)"]
fn full_hessian_matches_fd_of_df_gradient_ccpvdz() {
    if !has_deriv2() {
        return;
    }
    let (miss, scale) = check_full(&water(), "cc-pvdz", "cc-pvdz-ri", BAR_OWN);
    eprintln!("water cc-pVDZ/cc-pVDZ-RI: miss {miss:.3e} of {scale:.3e}");
}

#[test]
#[ignore = "slow: water def2-SVP/def2-universal-jkfit own-gradient FD; measured 6.5 s (4 threads, box load ~10)"]
fn full_hessian_matches_fd_of_df_gradient_def2svp_jkfit() {
    if !has_deriv2() {
        return;
    }
    let (miss, scale) = check_full(&water(), "def2-svp", "def2-universal-jkfit", BAR_OWN);
    eprintln!("water def2-SVP/jkfit: miss {miss:.3e} of {scale:.3e}");
}

#[test]
#[ignore = "slow: NH3 cc-pVDZ/cc-pVDZ-RI own-gradient FD; measured 38 s (4 threads, box load ~10)"]
fn full_hessian_matches_fd_of_df_gradient_nh3_ccpvdz() {
    if !has_deriv2() {
        return;
    }
    let mol = Molecule::parse_xyz(NH3_DISTORTED, 0, 1).unwrap();
    let (miss, scale) = check_full(&mol, "cc-pvdz", "cc-pvdz-ri", BAR_OWN);
    eprintln!("NH3 cc-pVDZ/cc-pVDZ-RI: miss {miss:.3e} of {scale:.3e}");
}

#[test]
#[ignore = "slow: exact-J vs RI-J Hessian at cc-pVDZ; measured 4.6 s (4 threads, box load ~10)"]
fn the_fitting_error_is_visible_to_the_exact_j_hessian() {
    // Negative control for the PySCF comparison: the exact-integral Hessian of
    // the same molecule/basis is NOT the RI-J Hessian. The miss is the RI
    // fitting error (SCF density change included).
    if !has_deriv2() {
        return;
    }
    let df = parts(&setup(&water(), "cc-pvdz", "cc-pvdz-ri")).total();
    let mol = water();
    let prep = prep_at(&mol, "cc-pvdz");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        max_iter: 200,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
    let exact = rhf_hessian(
        &ParallelContext::default(),
        &mol,
        &prep,
        op,
        &bounds,
        &rhf,
        &cfg,
    )
    .unwrap();
    let miss = max_abs_diff(&df, &exact);
    eprintln!("exact-J vs RI-J Hessian, water cc-pVDZ/cc-pVDZ-RI: miss {miss:.3e}");
    assert!(
        miss > 1e-5,
        "fitting error unexpectedly invisible: {miss:.3e}"
    );
}

/// `max|skeleton(explicit inverse) - skeleton(Cholesky)|` and the same for
/// `J^x`, for the RI-J pieces at the converged density of `s`.
fn explicit_inverse_difference(s: &Setup) -> (f64, f64) {
    let route = s.rhf.df_jk.as_ref().unwrap();
    let d = s.rhf.density_r();
    let good = rij_terms(&s.mol, &s.prep, route, d, &RijMutants::default()).unwrap();
    let bad = rij_terms(
        &s.mol,
        &s.prep,
        route,
        d,
        &RijMutants {
            explicit_inverse: true,
            ..Default::default()
        },
    )
    .unwrap();
    let jx = good
        .first_order_j
        .iter()
        .zip(&bad.first_order_j)
        .fold(0.0f64, |m, (a, b)| m.max(max_abs_diff(a, b)));
    (max_abs_diff(&good.skeleton, &bad.skeleton), jx)
}

#[test]
#[ignore = "slow: benzene STO-3G/def2-universal-jkfit RI-J pieces; measured 6.4 s (4 threads, box load ~10)"]
fn explicit_inverse_mutant_on_an_ill_conditioned_metric() {
    // cond(V) = 7.1e8 for benzene/def2-universal-jkfit (df_j.rs). The explicit
    // inverse is blind on the well-conditioned water metric but is detected
    // here, on the RI-J pieces (skeleton and J^x). Hypothesis: physics
    // predicts no difference (same algebra); a numerical artifact predicts a
    // difference that grows with cond(V) -- which is what is measured.
    // The assertion is RELATIVE to the water difference, measured in the same
    // run, not an absolute number: skeleton 9.0e-7 vs water 1.4e-11 (about
    // 6.5e4x); the J^x difference is only 1.3x a 1e-7 absolute bar, so it is
    // reported but not asserted.
    if !has_deriv2() {
        return;
    }
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/benzene.xyz"
    );
    let benzene = Molecule::load_xyz(path).unwrap();
    let (sk, jx) = explicit_inverse_difference(&setup(&benzene, "sto-3g", "def2-universal-jkfit"));
    let (sk_w, jx_w) = explicit_inverse_difference(&setup(&water(), "6-31g", "cc-pvdz-ri"));
    eprintln!(
        "explicit inverse: benzene skeleton {sk:.3e} J^x {jx:.3e}; water skeleton {sk_w:.3e} J^x {jx_w:.3e}"
    );
    assert!(
        sk > 100.0 * sk_w.max(1e-14),
        "skeleton {sk:.3e} vs water {sk_w:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Conditioning guard
// ---------------------------------------------------------------------------

#[test]
fn a_nearly_dependent_fitting_basis_is_refused_naming_the_mode_count() {
    // Hypothesis: a fitting basis with a near-duplicate s function on each
    // hydrogen has metric eigenvalues below the DF-K cut (1e-10); RI-J uses
    // the FULL inverse, so its second derivative would amplify noise without
    // bound and is refused (naming the mode count) instead of returning a
    // number. The artifact hypothesis (the guard fires on every basis) is
    // excluded by the accepted case: a twin 1e-3 away is fine. Measured
    // boundary on this basis: accepted at 1e-4, refused (2 modes, one per H)
    // from 1e-5 down.
    use ferric_core::basis::Shell;
    if !has_deriv2() {
        return;
    }
    let mol = water();
    let prep = prep_at(&mol, "sto-3g");
    let d = hydrogens_density(&prep);
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
        ferric_scf::hessian::rij_support::rij_terms_with_aux(
            mol.atoms.len(),
            &prep,
            PreparedBasis::new(&mol, &aux).unwrap(),
            Operator::coulomb(),
            usize::MAX,
            &d,
            &RijMutants::default(),
        )
    };
    // Accepted AND accurate: the skeleton of the twin-augmented basis agrees
    // with the finite difference of the validated DF gradient at fixed D built
    // with the same basis (bar 2e-6, the piece-test bar).
    let accepted = try_twin(1e-3).expect("a well-separated twin must be accepted");
    let route_budget = usize::MAX;
    let fd = fd_hessian(&mol, H_PIECE, |m| {
        let p = prep_at(m, "sto-3g");
        let mut aux = basis::bundled("cc-pvdz-ri").unwrap();
        let h = aux.shells.get_mut(&1).unwrap();
        let first = h.iter().find(|s| s.l == 0).unwrap().clone();
        let a = first.exponents[0];
        h.push(Shell {
            exponents: vec![a * (1.0 + 1e-3)],
            coefficients: vec![(2.0 * a / std::f64::consts::PI).powf(0.75)],
            ..first
        });
        df_jk_gradient(
            3,
            &p,
            &PreparedBasis::new(m, &aux).unwrap(),
            Operator::coulomb(),
            Some(&d),
            &[],
            route_budget,
        )
        .unwrap()
    });
    let acc = assert_close(
        "twin 1e-3 accepted: J skeleton",
        &accepted.skeleton,
        &fd,
        2e-6,
    );
    eprintln!("twin 1e-3: accuracy {acc:.2e}");
    for eps in [1e-5, 1e-7] {
        let msg = try_twin(eps).expect_err("near-singular metric").to_string();
        eprintln!("twin {eps:.0e}: {msg}");
        assert!(msg.contains("2 eigenvalue(s) below 1e-10"), "{msg}");
    }
}

// ---------------------------------------------------------------------------
// Accuracy as a function of the conditioning of the Coulomb metric
// ---------------------------------------------------------------------------

/// `(lambda_min, lambda_max)` of the Coulomb metric of `aux` on `mol`.
fn metric_extremes(mol: &Molecule, aux: &str) -> (f64, f64) {
    use ndarray_linalg::{Eigh, UPLO};
    let v = coulomb_metric_2c(Operator::coulomb(), &aux_at(mol, aux)).unwrap();
    let (lam, _) = v.eigh(UPLO::Upper).unwrap();
    (lam[0], lam[lam.len() - 1])
}

#[test]
#[ignore = "slow: accuracy-vs-cond(V) scan, own-gradient FD per aux basis; measured 46 s for six bases (4 threads, box load ~10)"]
fn accuracy_versus_metric_conditioning() {
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
        let (lmin, lmax) = metric_extremes(&water(), aux);
        let s = setup(&water(), "6-31g", aux);
        let total = parts(&s).total();
        let fd = fd_hessian_richardson(&s.mol, 2e-3, |m| df_gradient_at(m, "6-31g", aux));
        eprintln!(
            "COND {aux}: lambda_min {lmin:.2e} lambda_max {lmax:.2e} cond {:.2e} miss vs own FD {:.3e} ({:.1}s)",
            lmax / lmin,
            max_abs_diff(&total, &fd),
            t0.elapsed().as_secs_f64()
        );
    }
}

#[test]
fn exact_span_fit_stays_exact_at_extreme_conditioning() {
    // The truth is conditioning-independent: with the exact-span auxiliary
    // basis the fitted skeleton and J^x must equal the exact four-centre ones
    // whatever near-duplicate functions are added. The 6-31G span itself has
    // cond(V) = 8.7e11 (lambda_min 1.8e-10); twins at 1e-1, 1e-2, 1e-3 take it
    // to 3.6e12, 4.8e12, 2.0e14 (measured). Misses stay at 2.2e-13 (skeleton)
    // and 6e-13 .. 2.9e-12 (J^x). Benign case: the data are consistent with
    // the metric. The generic-density accuracy is measured separately
    // (accuracy_versus_metric_conditioning, ignored).
    use ferric_scf::hessian::rij_support::{exact_j_first_order, exact_j_skeleton};
    if !has_deriv2() {
        return;
    }
    let mol = water();
    let prep = prep_at(&mol, "6-31g");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let d = hydrogens_density(&prep);
    let skel = exact_j_skeleton(&prep, op, &bounds, 3, &d).unwrap();
    let jx = exact_j_first_order(&prep, op, &bounds, 3, &d).unwrap();
    for eps in [None, Some(1e-1), Some(1e-2), Some(1e-3)] {
        let r = ferric_scf::hessian::rij_support::rij_terms_with_aux(
            3,
            &prep,
            exact_span_aux_with_twin(&mol, "6-31g", eps),
            op,
            usize::MAX,
            &d,
            &RijMutants {
                allow_ill_conditioned: true,
                ..Default::default()
            },
        )
        .unwrap();
        let jx_miss = r
            .first_order_j
            .iter()
            .zip(&jx)
            .fold(0.0f64, |m, (a, b)| m.max(max_abs_diff(a, b)));
        let sk_miss = max_abs_diff(&r.skeleton, &skel);
        eprintln!("twin {eps:?}: skeleton miss {sk_miss:.2e}, J^x miss {jx_miss:.2e}");
        assert!(
            sk_miss < 1e-10 && jx_miss < 1e-10,
            "{eps:?}: {sk_miss:.2e} {jx_miss:.2e}"
        );
    }
}

#[test]
fn the_pre_scf_probe_sees_an_ill_conditioned_metric() {
    // The metric depends on geometry and fitting basis only, so the probe the
    // frequency dispatch consults before the SCF refuses it (Auto then runs
    // finite differences instead of failing after the SCF). The exact-span
    // 6-31G set with a 1e-2 twin has lambda_min 3.3e-11 < 1e-10; a healthy
    // fitting basis passes.
    let mol = water();
    let op = Operator::coulomb();
    let bad = exact_span_aux_with_twin(&mol, "6-31g", Some(1e-2));
    let msg = ferric_scf::hessian::rij_support::probe_metric(&bad, op)
        .expect_err("ill-conditioned metric")
        .to_string();
    eprintln!("probe: {msg}");
    assert!(msg.contains("eigenvalue(s) below 1e-10"), "{msg}");
    ferric_scf::hessian::rij_support::probe_metric(&aux_at(&mol, "cc-pvdz-ri"), op)
        .expect("a healthy fitting basis passes the probe");
}
