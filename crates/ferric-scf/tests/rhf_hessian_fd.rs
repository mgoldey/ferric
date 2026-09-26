//! Analytic RHF Hessian (`ferric_scf::hessian`) against INDEPENDENT
//! constructions: central finite differences of ferric's analytic gradients.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test rhf_hessian_fd
//! ```
//!
//! # What is compared to what
//!
//! * Term 1 (nuclear repulsion) vs a four-point second difference of
//!   `Molecule::nuclear_repulsion` — no integrals, no SCF: the exactness
//!   anchor for the index/units convention every other test relies on.
//! * Skeleton terms 2–4 vs central differences of the gradient pieces at a
//!   FROZEN AO density and energy-weighted density (the skeleton IS that
//!   derivative: FD of PySCF's frozen-density gradient reproduces PySCF's
//!   skeleton to 6e-9):
//!   - `nuclear + one_electron` vs FD of `oneelectron_gradient(D, W = 0)`
//!   - `nuclear + overlap`      vs FD of `oneelectron_gradient(D = 0, W)`
//!   - `two_electron`           vs FD of `twoelectron_gradient(D)`
//!     No SCF runs at the displaced geometries, so the only noise is O(h²)
//!     and integral screening; each term fails on its own.
//! * The full Hessian vs central differences of `rhf_gradient` with a fresh
//!   SCF at every displaced geometry. Only this comparison sees term 5 (CPHF).
//! * Internal consistency: symmetry of the raw response term (terms 1–4 are
//!   symmetric by construction, the response is not), and translational
//!   invariance of the total (`Σ_B H[(A,a),(B,b)] = 0`).
//!
//! H2O is DISTORTED off C2v so every Cartesian block is populated (a
//! symmetric geometry would zero whole blocks and hide a coordinate mix-up).
//! Bases: STO-3G, 6-31G, and cc-pVDZ (d functions: the second-derivative
//! integral path at l = 2).
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * Right: each skeleton term matches its fixed-density FD at the O(h²)
//!   level (~1e-8); the full Hessian matches FD of `rhf_gradient` at the SCF
//!   noise/truncation floor (~1e-6); the raw response asymmetry is ~CPHF_TOL.
//! * libint2 block ORDER wrong (the i ≤ j pair index, or the operator-centre
//!   blocks of the nuclear engine): coordinates mix, the affected skeleton
//!   test misses by O(0.1–1) Ha/Bohr² even though translational invariance
//!   may survive (a coordinate permutation can preserve row sums).
//! * CPHF wrong (sign, factor, occupied block): every skeleton test passes,
//!   the full-FD test fails and the response asymmetry grows far above
//!   CPHF_TOL — the two failures together localize it to term 5.
//! * FD reference broken (a displaced SCF on another state, too loose SCF):
//!   the full-FD test fails column-wise while the skeleton tests and the
//!   response symmetry stay clean.
//!
//! # MUTATION (each must turn a named test red; see `src/hessian.rs`)
//!
//! * skeleton 1e: delete `h += &contract_1e_deriv2(prep, natoms, ffi::OP_NUCLEAR, d)?;`
//!   in `skeleton_hess_1e` → `skeleton_one_electron_matches_fd_*` misses by
//!   O(10) Ha/Bohr² (the nuclear-attraction curvature).
//! * overlap/W: drop the leading `-` in `skeleton_hess_overlap` →
//!   `skeleton_overlap_matches_fd_*` misses by 2·|term| (~0.5 Ha/Bohr²).
//! * 2e skeleton: in `perm_summed_gamma` pass `blk.sym34` where `blk.sym12`
//!   goes → `skeleton_two_electron_matches_fd_*` misses (wrong permutation
//!   weights on every s1 ≠ s2 quartet).
//! * CPHF: in `solve_cphf_one` change `rhs -= &frame.c_vir.t().dot(&g_oo)...`
//!   to `+=` → skeleton tests stay green, `full_hessian_matches_fd_*` misses
//!   by ~1e-2 and `response_asymmetry` exceeds its bar.
//! * scatter: drop the `if i != j { h[(c, r)] += v; }` mirror in
//!   `scatter_unique_pairs` → every skeleton test misses (half of each
//!   off-diagonal second derivative lost).
//!
//! Run 2026-09-25, each exactly as predicted: skeleton 1e → the two
//! `skeleton_one_electron` tests and the three full-Hessian tests fail;
//! overlap sign → `skeleton_overlap` ×2 + full ×3; 2e weights →
//! `skeleton_two_electron` ×2 + full ×3; CPHF sign → full ×3 and
//! `hessian_is_symmetric_and_translationally_invariant`, skeletons green;
//! scatter mirror → all ten Hessian tests (only the V_nn anchor and the
//! refusal test pass).

use ferric_core::basis;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{
    build_energy_weighted_density, oneelectron_gradient, rhf_gradient, twoelectron_gradient,
};
use ferric_scf::hessian::{
    hess_nuclear_repulsion, rhf_hessian, rhf_hessian_parts, RhfHessianParts,
};
use ferric_scf::result::{DfJkRoute, ScfResult};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Nuclear-repulsion term vs 4-point second difference of V_nn, Ha/Bohr².
/// Richardson-extrapolated stencil: roundoff ~1e-8 (the plain h = 1e-3 stencil
/// missed by 2.7e-6, its own O(h²) truncation).
const TOL_NUCLEAR_ANCHOR: f64 = 1e-7;
/// Skeleton term vs FD of the frozen-density gradient piece, Ha/Bohr².
/// Measured ≤ 3.4e-8 (cc-pVDZ two-electron term).
const TOL_SKELETON_FD: f64 = 5e-7;
/// Full analytic Hessian vs FD of `rhf_gradient` with re-converged SCFs,
/// Ha/Bohr². Measured ≤ 5.2e-7, the FD's own floor (O(h²) truncation at
/// h = 1e-3 plus SCF noise ~density_conv/h).
const TOL_FULL_FD: f64 = 5e-6;
/// Raw response-term asymmetry, Ha/Bohr². Measured ≤ 2.4e-11.
const TOL_RESPONSE_ASYM: f64 = 1e-9;
/// Translational invariance of the total Hessian, Ha/Bohr². Measured 7.2e-12.
const TOL_TRANSLATION: f64 = 1e-9;
/// The response term must be at least this multiple of `TOL_FULL_FD` in
/// max-abs, so the full-FD test cannot pass with term 5 silently zero.
const MUST_MATTER_FACTOR: f64 = 100.0;

/// FD step for the frozen-density skeleton checks (no SCF noise), Bohr.
const H_SKELETON: f64 = 1e-4;
/// FD step for the full Hessian (re-converged SCF at each point), Bohr.
const H_FULL: f64 = 1e-3;

const WATER_DISTORTED: &str = "3\nH2O distorted off C2v\n\
O      0.010000    -0.020000     0.120000\n\
H      0.030000     0.770000    -0.460000\n\
H     -0.020000    -0.740000    -0.490000\n";

fn water() -> Molecule {
    Molecule::parse_xyz(WATER_DISTORTED, 0, 1).unwrap()
}

fn scf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 200,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        // Explicit exact J/K: the analytic Hessian refuses RI.
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        ..Default::default()
    }
}

struct Setup {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    rhf: ScfResult,
}

fn setup(mol: &Molecule, basis_name: &str) -> Setup {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        mol,
        &prep,
        op,
        &bounds,
        &scf_config(),
    )
    .unwrap();
    assert!(
        rhf.converged,
        "{basis_name}: reference SCF did not converge"
    );
    Setup {
        mol: mol.clone(),
        prep,
        bounds,
        rhf,
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
        &scf_config(),
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

/// Central-difference Hessian: column b = (g(x_b + h) − g(x_b − h)) / 2h,
/// `grad` returning the (natoms, 3) gradient at a geometry.
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

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn assert_close(ctx: &str, got: &Array2<f64>, want: &Array2<f64>, tol: f64) {
    let d = max_abs_diff(got, want);
    eprintln!(
        "{ctx}: max|analytic - FD| = {d:.3e} (max|FD| {:.3e}, tol {tol:.0e})",
        max_abs(want)
    );
    assert!(d < tol, "{ctx}: max|analytic - FD| = {d:.3e} >= {tol:.0e}");
}

/// Frozen-density gradient with the basis (and nuclei) at `m`.
fn frozen_prep(m: &Molecule, basis_name: &str) -> (PreparedBasis, SchwarzBounds) {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(m, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (prep, bounds)
}

// ---------------------------------------------------------------------------
// Term 1: exactness anchor
// ---------------------------------------------------------------------------

#[test]
fn nuclear_repulsion_term_matches_second_difference_of_energy() {
    let mol = water();
    let an = hess_nuclear_repulsion(&mol);
    // Four-point second difference of V_nn, Richardson-extrapolated from h and
    // h/2: the plain h = 1e-3 stencil carries O(h²) truncation of ~3e-6 here
    // (the O–H repulsion's fourth derivative), which is the stencil's error,
    // not the analytic term's. The extrapolated value is O(h⁴) (~1e-12) plus
    // roundoff ~ eps·V_nn/h² ≈ 1e-8.
    let h = 1e-3;
    let fd = (4.0 * second_difference_vnn(&mol, h / 2.0) - second_difference_vnn(&mol, h)) / 3.0;
    assert!(max_abs(&an) > 1.0, "vacuous: nuclear Hessian is ~0");
    assert_close("V_nn anchor", &an, &fd, TOL_NUCLEAR_ANCHOR);
}

fn second_difference_vnn(mol: &Molecule, h: f64) -> Array2<f64> {
    let n3 = 3 * mol.atoms.len();
    let e = |a: usize, sa: f64, b: usize, sb: f64| {
        displaced(&displaced(mol, a / 3, a % 3, sa), b / 3, b % 3, sb).nuclear_repulsion()
    };
    let mut fd = Array2::<f64>::zeros((n3, n3));
    for a in 0..n3 {
        for b in 0..n3 {
            fd[(a, b)] =
                (e(a, h, b, h) - e(a, h, b, -h) - e(a, -h, b, h) + e(a, -h, b, -h)) / (4.0 * h * h);
        }
    }
    fd
}

// ---------------------------------------------------------------------------
// Terms 2-4: skeleton vs frozen-density FD
// ---------------------------------------------------------------------------

fn check_skeleton_one_electron(basis_name: &str) {
    let s = setup(&water(), basis_name);
    let p = parts(&s);
    let d = s.rhf.density_r().clone();
    let zero = Array2::<f64>::zeros(d.dim());
    let fd = fd_hessian(&s.mol, H_SKELETON, |m| {
        let (prep, _) = frozen_prep(m, basis_name);
        oneelectron_gradient(m, &prep, &d, &zero, None).unwrap()
    });
    let an = &p.nuclear + &p.one_electron;
    assert_close(
        &format!("{basis_name}: V_nn + D·h''"),
        &an,
        &fd,
        TOL_SKELETON_FD,
    );
}

fn check_skeleton_overlap(basis_name: &str) {
    let s = setup(&water(), basis_name);
    let p = parts(&s);
    let nocc = (s.mol.nelec() / 2) as usize;
    let w = build_energy_weighted_density(&s.rhf, nocc);
    let zero = Array2::<f64>::zeros(w.dim());
    let fd = fd_hessian(&s.mol, H_SKELETON, |m| {
        let (prep, _) = frozen_prep(m, basis_name);
        oneelectron_gradient(m, &prep, &zero, &w, None).unwrap()
    });
    let an = &p.nuclear + &p.overlap;
    assert!(max_abs(&p.overlap) > 1e-2, "vacuous: overlap term is ~0");
    assert_close(
        &format!("{basis_name}: V_nn - W·S''"),
        &an,
        &fd,
        TOL_SKELETON_FD,
    );
}

fn check_skeleton_two_electron(basis_name: &str) {
    let s = setup(&water(), basis_name);
    let p = parts(&s);
    let d = s.rhf.density_r().clone();
    let fd = fd_hessian(&s.mol, H_SKELETON, |m| {
        let (prep, bounds) = frozen_prep(m, basis_name);
        twoelectron_gradient(&prep, Operator::coulomb(), &bounds, &d).unwrap()
    });
    assert!(
        max_abs(&p.two_electron) > 1e-1,
        "vacuous: 2e skeleton is ~0"
    );
    assert_close(
        &format!("{basis_name}: Γ·ERI''"),
        &p.two_electron,
        &fd,
        TOL_SKELETON_FD,
    );
}

#[test]
fn skeleton_one_electron_matches_fd_sto3g() {
    check_skeleton_one_electron("sto-3g");
}

#[test]
fn skeleton_one_electron_matches_fd_ccpvdz() {
    check_skeleton_one_electron("cc-pvdz");
}

#[test]
fn skeleton_overlap_matches_fd_sto3g() {
    check_skeleton_overlap("sto-3g");
}

#[test]
fn skeleton_overlap_matches_fd_ccpvdz() {
    check_skeleton_overlap("cc-pvdz");
}

#[test]
fn skeleton_two_electron_matches_fd_sto3g() {
    check_skeleton_two_electron("sto-3g");
}

#[test]
fn skeleton_two_electron_matches_fd_ccpvdz() {
    check_skeleton_two_electron("cc-pvdz");
}

// ---------------------------------------------------------------------------
// Full Hessian (term 5 included) vs FD of the analytic gradient
// ---------------------------------------------------------------------------

fn check_full(basis_name: &str) {
    let s = setup(&water(), basis_name);
    let p = parts(&s);
    let total = p.total();
    let op = Operator::coulomb();
    let fd = fd_hessian(&s.mol, H_FULL, |m| {
        let sm = setup(m, basis_name);
        rhf_gradient(m, &sm.prep, op, &sm.bounds, &sm.rhf, None).unwrap()
    });
    eprintln!(
        "{basis_name}: CPHF iterations {:?}, max residual {:.2e}, response asym {:.2e}",
        p.cphf_iterations, p.cphf_max_residual, p.response_asymmetry
    );
    // TEETH: the response term must matter at this bar, and the skeleton
    // alone must miss the FD Hessian.
    assert!(
        max_abs(&p.response) > MUST_MATTER_FACTOR * TOL_FULL_FD,
        "{basis_name}: response term max {:.3e} too small for the full-FD test to see it",
        max_abs(&p.response)
    );
    assert!(
        max_abs_diff(&p.skeleton(), &fd) > MUST_MATTER_FACTOR * TOL_FULL_FD,
        "{basis_name}: skeleton alone matches FD — the test cannot see term 5"
    );
    assert_close(
        &format!("{basis_name}: full Hessian"),
        &total,
        &fd,
        TOL_FULL_FD,
    );
    assert!(
        p.response_asymmetry < TOL_RESPONSE_ASYM,
        "{basis_name}: raw response asymmetry {:.3e} >= {TOL_RESPONSE_ASYM:.0e}",
        p.response_asymmetry
    );
}

#[test]
fn full_hessian_matches_fd_sto3g() {
    check_full("sto-3g");
}

#[test]
fn full_hessian_matches_fd_631g() {
    check_full("6-31g");
}

#[test]
fn full_hessian_matches_fd_ccpvdz() {
    check_full("cc-pvdz");
}

#[test]
fn hessian_is_symmetric_and_translationally_invariant() {
    let s = setup(&water(), "6-31g");
    let p = parts(&s);
    let total = p.total();
    let n3 = total.nrows();
    let asym = max_abs_diff(&total, &total.t().to_owned());
    assert!(
        asym < 1e-12,
        "symmetrized total is not symmetric: {asym:.3e}"
    );
    assert!(
        p.response_asymmetry < TOL_RESPONSE_ASYM,
        "raw response asymmetry {:.3e}",
        p.response_asymmetry
    );
    let mut worst = 0.0f64;
    for a in 0..n3 {
        for axis in 0..3 {
            let sum: f64 = (0..n3 / 3).map(|b| total[(a, 3 * b + axis)]).sum();
            worst = worst.max(sum.abs());
        }
    }
    eprintln!("translational invariance: worst row sum {worst:.3e}");
    assert!(worst < TOL_TRANSLATION, "Σ_B H[(A,a),(B,b)] = {worst:.3e}");
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn unsupported_configurations_are_refused() {
    let s = setup(&water(), "sto-3g");
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let refused = |rhf: &ScfResult, cfg: &RhfConfig, op: Operator, what: &str| {
        let err = rhf_hessian(&ctx, &s.mol, &s.prep, op, &s.bounds, rhf, cfg)
            .expect_err(&format!("{what} must be refused"));
        eprintln!("{what}: {err}");
    };

    let mut ri = s.rhf.clone();
    ri.df_jk = DfJkRoute::from_scf(Some("cc-pvdz-ri"), None, None, op, 0);
    assert!(ri.df_jk.is_some());
    refused(&ri, &scf_config(), op, "RI SCF");

    let ks = RhfConfig {
        xc: Some("PBE".into()),
        ..scf_config()
    };
    refused(&s.rhf, &ks, op, "KS config");

    let ext = RhfConfig {
        external_potential: Some(ExternalPotential {
            field: Some([0.0, 0.0, 1e-3]),
            ..Default::default()
        }),
        ..scf_config()
    };
    refused(&s.rhf, &ext, op, "external field");

    refused(&s.rhf, &scf_config(), Operator::erf(0.4), "erf operator");

    let mut unconverged = s.rhf.clone();
    unconverged.converged = false;
    refused(&unconverged, &scf_config(), op, "unconverged SCF");

    // Negative control: the supported configuration is NOT refused.
    rhf_hessian(&ctx, &s.mol, &s.prep, op, &s.bounds, &s.rhf, &scf_config())
        .expect("plain RHF must be supported");
}
