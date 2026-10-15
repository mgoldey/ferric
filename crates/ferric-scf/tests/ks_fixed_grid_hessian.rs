//! Opt-in analytic Hessian of closed-shell Kohn-Sham LDA with exact J on a grid
//! FIXED IN SPACE (`ferric_scf::hessian::rks_lda_fixed_grid_hessian`, unit 1 of
//! issue #407).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test ks_fixed_grid_hessian
//! ```
//!
//! # What is compared to what
//!
//! The energy is `E[R] = E_KS(R; grid frozen at the reference geometry)`: the
//! grid points and weights never move, only the AOs do. Every reference below
//! differentiates THAT energy, so none of them contains a Becke weight
//! response.
//!
//! * XC pieces at a FROZEN density (no SCF anywhere): the XC skeleton vs a
//!   central difference of `FixedGridLda::gradient`; the first-order XC Fock
//!   matrix vs a central difference of `V_xc` built at displaced nuclei; the
//!   kernel response vs a central difference of `V_xc` in the density.
//! * The gradient vs a central difference of the fixed-grid ENERGY from SCFs
//!   run at displaced nuclei on the reference grid (through the injected-XC
//!   SCF, [`ferric_scf::ks_fixed_grid::FixedGridXc`]). This is what validates
//!   the gradient the Hessian is then differenced against: a gradient defect
//!   shared by Hessian and reference would be invisible otherwise.
//! * The full Hessian vs a central difference of that validated gradient with
//!   a fresh fixed-grid SCF at every displaced geometry. Only this comparison
//!   sees the CPHF response with the XC kernel.
//!
//! # Artifact hypothesis next to the physics hypothesis
//!
//! * Right: every XC piece matches its frozen-density difference at the
//!   `O(h^2)` level; the full Hessian matches the gradient difference at the
//!   SCF-noise / truncation floor.
//! * Broken kernel response in the CPHF: the three frozen-density pieces still
//!   pass, the full-Hessian comparison fails by the response size.
//! * The reference secretly differentiates the production (moving-grid)
//!   energy: the fixed-grid Hessian would miss it by the weight-response
//!   scale. The negative control below measures exactly that scale and
//!   requires the miss to be large against the agreement bar.
//! * FD of an SCF that silently re-built the grid at the displaced geometry
//!   would absorb the weight response and make the fixed-grid Hessian miss;
//!   the displaced SCFs here take the grid from the reference geometry by
//!   construction (the grid is an argument), and the displaced energy at the
//!   reference geometry reproduces the production SCF energy.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::hessian::{
    rks_lda_fixed_grid_hessian, rks_lda_fixed_grid_hessian_parts_with, KsFixedGridMutants,
};
use ferric_scf::ks_fixed_grid::{
    rks_lda_fixed_grid_gradient, solve_rks_fixed_grid, FixedGridLda, XcGeometry, XcMutants,
};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const WATER: &str = "3\nH2O distorted off C2v\n\
O      0.010000    -0.020000     0.120000\n\
H      0.030000     0.770000    -0.460000\n\
H     -0.020000    -0.740000    -0.490000\n";

/// Grid for the SCF-based anchors: 50 radial x 110 angular per atom (the
/// physics is grid-independent; this keeps the 18+ displaced SCFs cheap).
fn grid_cfg() -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial: 50,
        n_angular: 110,
        prune: None,
    }
}

fn has_deriv2() -> bool {
    let order = ferric_integrals::engine::libint_max_deriv_order();
    if order < 2 {
        eprintln!("SKIP: libint2 generated with derivative order {order}; no analytic Hessian");
    }
    order >= 2
}

fn water() -> Molecule {
    Molecule::parse_xyz(WATER, 0, 1).unwrap()
}

/// Config handed to the Hessian: names the functional and states that J is
/// exact (the KS default is RI-J).
fn hessian_config() -> RhfConfig {
    RhfConfig {
        xc: Some("LDA".into()),
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        dft_grid: Some(grid_cfg()),
        ..Default::default()
    }
}

/// Production SCF config (grid following the atoms, exact J).
fn production_config(grid: Option<AtomicGridConfig>) -> RhfConfig {
    RhfConfig {
        max_iter: 200,
        energy_conv: 1e-13,
        density_conv: 1e-11,
        dft_grid: grid,
        ..hessian_config()
    }
}

struct Pt {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    res: ScfResult,
}

/// SCF at `mol` whose XC lives on the reference `grid` (frozen in space).
fn fixed_grid_scf(grid: &FixedGridLda, mol: &Molecule, basis_name: &str) -> Pt {
    let ctx = ParallelContext::default();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        use_sad_guess: false,
        max_iter: 200,
        energy_conv: 1e-13,
        density_conv: 1e-11,
        ..Default::default()
    };
    let res = solve_rks_fixed_grid(&ctx, mol, &prep, op, &bounds, grid, &cfg).unwrap();
    assert!(
        res.converged,
        "{basis_name}: fixed-grid SCF did not converge"
    );
    Pt {
        mol: mol.clone(),
        prep,
        bounds,
        res,
    }
}

fn displaced(mol: &Molecule, coord: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match coord % 3 {
        0 => m.atoms[coord / 3].x += h,
        1 => m.atoms[coord / 3].y += h,
        _ => m.atoms[coord / 3].zpos += h,
    }
    m
}

/// Central difference `(f(+h) - f(-h)) / 2h` of an array-valued function of a
/// step, Richardson-extrapolated from `h` and `h/2` (truncation `O(h^4)`).
fn rich(h: f64, mut f: impl FnMut(f64) -> Array2<f64>) -> Array2<f64> {
    let mut d = |h: f64| (f(h) - f(-h)) / (2.0 * h);
    let coarse = d(h);
    let fine = d(0.5 * h);
    (4.0 * fine - coarse) / 3.0
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    max_abs(&(a - b))
}

fn flat(g: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_vec((g.len(), 1), g.iter().copied().collect()).unwrap()
}

/// Reference-geometry objects shared by the tests.
struct Ref {
    mol: Molecule,
    bs_name: String,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    grid: FixedGridLda,
    fixed: Pt,
}

fn reference(basis_name: &str) -> Ref {
    let mol = water();
    let grid = FixedGridLda::new(&mol, "LDA", &grid_cfg()).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let fixed = fixed_grid_scf(&grid, &mol, basis_name);
    Ref {
        mol,
        bs_name: basis_name.to_string(),
        prep,
        bounds,
        grid,
        fixed,
    }
}

/// The fixed-grid energy gradient at the reference geometry, (natoms, 3).
fn gradient_at(r: &Ref, pt: &Pt) -> Array2<f64> {
    rks_lda_fixed_grid_gradient(
        &pt.mol,
        &pt.prep,
        Operator::coulomb(),
        &pt.bounds,
        &r.grid,
        &pt.res,
    )
    .unwrap()
}

/// Central difference of the fixed-grid gradient, each from its own SCF.
fn fd_hessian_of_gradient(r: &Ref, h: f64) -> Array2<f64> {
    let n3 = 3 * r.mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n3, n3));
    for b in 0..n3 {
        let col = rich(h, |s| {
            let pt = fixed_grid_scf(&r.grid, &displaced(&r.mol, b, s), &r.bs_name);
            flat(&gradient_at(r, &pt))
        });
        out.column_mut(b).assign(&col.column(0));
    }
    out
}

/// Central difference of the fixed-grid ENERGY, (natoms, 3).
fn fd_gradient_of_energy(r: &Ref, h: f64) -> Array2<f64> {
    let n3 = 3 * r.mol.atoms.len();
    let mut out = Array2::<f64>::zeros((r.mol.atoms.len(), 3));
    for b in 0..n3 {
        let e = rich(h, |s| {
            let pt = fixed_grid_scf(&r.grid, &displaced(&r.mol, b, s), &r.bs_name);
            Array2::from_elem((1, 1), pt.res.energy)
        });
        out[(b / 3, b % 3)] = e[(0, 0)];
    }
    out
}

fn analytic(r: &Ref, m: &KsFixedGridMutants) -> Array2<f64> {
    let cfg = hessian_config();
    rks_lda_fixed_grid_hessian_parts_with(
        &ParallelContext::default(),
        &r.mol,
        &r.fixed.prep,
        Operator::coulomb(),
        &r.fixed.bounds,
        &r.fixed.res,
        &cfg,
        &r.grid,
        m,
    )
    .unwrap()
    .total()
}

// ---------------------------------------------------------------------------
// Frozen-density pieces (no SCF)
// ---------------------------------------------------------------------------

/// Tolerance for the frozen-density comparisons, Ha/Bohr^2 (or Ha/Bohr for the
/// Fock matrix): central difference with Richardson at h = 1e-3, no SCF noise.
const TOL_FROZEN: f64 = 1e-7;
/// Step of the frozen-density differences, Bohr (measured error 1.9e-8 at 6-31G,
/// h^4 truncation of the tight O 1s functions).
const H_FROZEN: f64 = 5e-4;

#[test]
fn xc_pieces_match_frozen_density_differences() {
    for basis_name in ["sto-3g", "6-31g"] {
        let r = reference(basis_name);
        let d = r.fixed.res.density_r().clone();
        let geo = XcGeometry::new(&r.mol, &r.prep).unwrap();
        let n3 = 3 * r.mol.atoms.len();
        let bs = basis::bundled(basis_name).unwrap();
        let skel = r
            .grid
            .skeleton_hessian(&geo, &d, &XcMutants::default())
            .unwrap();
        let fock = r
            .grid
            .first_order_fock(&geo, &d, &XcMutants::default())
            .unwrap();
        let mut worst_skel = 0.0f64;
        let mut worst_fock = 0.0f64;
        for b in 0..n3 {
            let col = rich(H_FROZEN, |s| {
                let m = displaced(&r.mol, b, s);
                let prep = PreparedBasis::new(&m, &bs).unwrap();
                let g = XcGeometry::new(&m, &prep).unwrap();
                flat(&r.grid.gradient(&g, &d).unwrap())
            });
            let an = skel.column(b).to_owned().insert_axis(ndarray::Axis(1));
            worst_skel = worst_skel.max(max_abs_diff(&an, &col));
            let v = rich(H_FROZEN, |s| {
                let m = displaced(&r.mol, b, s);
                let prep = PreparedBasis::new(&m, &bs).unwrap();
                let g = XcGeometry::new(&m, &prep).unwrap();
                r.grid.energy_vxc(&g, &d).unwrap().1
            });
            worst_fock = worst_fock.max(max_abs_diff(&fock[b], &v));
        }
        eprintln!(
            "{basis_name}: XC skeleton vs FD of frozen-D gradient {worst_skel:.3e}; first-order \
             Fock vs FD of V_xc {worst_fock:.3e} (max |skeleton| {:.3e})",
            max_abs(&skel)
        );
        assert!(
            worst_skel < TOL_FROZEN,
            "{basis_name}: skeleton {worst_skel:.3e}"
        );
        assert!(
            worst_fock < TOL_FROZEN,
            "{basis_name}: Fock {worst_fock:.3e}"
        );
    }
}

/// The frozen-density skeleton error must be finite-difference truncation, not
/// a missing term: halving the step must shrink it by close to 2^4 = 16 (the
/// Richardson-extrapolated stencil), which a wrong or missing term (a constant
/// miss, independent of the step) cannot do.
#[test]
fn xc_skeleton_difference_error_is_truncation_that_scales_with_the_step() {
    let r = reference("6-31g");
    let d = r.fixed.res.density_r().clone();
    let geo = XcGeometry::new(&r.mol, &r.prep).unwrap();
    let bs = basis::bundled("6-31g").unwrap();
    let skel = r
        .grid
        .skeleton_hessian(&geo, &d, &XcMutants::default())
        .unwrap();
    let err = |h: f64| {
        let mut worst = 0.0f64;
        for b in 0..9 {
            let col = rich(h, |s| {
                let m = displaced(&r.mol, b, s);
                let prep = PreparedBasis::new(&m, &bs).unwrap();
                let g = XcGeometry::new(&m, &prep).unwrap();
                flat(&r.grid.gradient(&g, &d).unwrap())
            });
            let an = skel.column(b).to_owned().insert_axis(ndarray::Axis(1));
            worst = worst.max(max_abs_diff(&an, &col));
        }
        worst
    };
    let e = [4e-3, 2e-3, 1e-3, 5e-4].map(err);
    eprintln!("skeleton error at h = 4e-3, 2e-3, 1e-3, 5e-4: {e:?}");
    for k in 0..3 {
        let ratio = e[k] / e[k + 1];
        assert!(
            ratio > 12.0,
            "halving h shrank the error only {ratio:.1}x: {e:?}"
        );
    }
}

#[test]
fn xc_kernel_response_matches_difference_of_vxc_in_the_density() {
    let r = reference("6-31g");
    let d = r.fixed.res.density_r().clone();
    let geo = XcGeometry::new(&r.mol, &r.prep).unwrap();
    let n = d.nrows();
    // A smooth deterministic symmetric perturbation (not a density: the kernel
    // is linear, so any symmetric matrix is a valid probe).
    let delta = Array2::from_shape_fn((n, n), |(i, j)| {
        let (a, b) = (i.min(j) as f64, i.max(j) as f64);
        (0.37 * a + 0.11 * b + 0.5).sin() * 0.1
    });
    let kernel = r.grid.response(&geo, &d, 1.0).unwrap();
    let an = kernel.apply(&delta);
    let fd = rich(H_FROZEN, |s| {
        let dd = &d + &(s * &delta);
        r.grid.energy_vxc(&geo, &dd).unwrap().1
    });
    let diff = max_abs_diff(&an, &fd);
    eprintln!(
        "kernel response vs FD of V_xc(D): {diff:.3e} (max |V1| {:.3e})",
        max_abs(&an)
    );
    assert!(diff < TOL_FROZEN, "kernel response {diff:.3e}");
    assert!(
        max_abs(&an) > 1e3 * TOL_FROZEN,
        "the probe must exercise the kernel"
    );
}

// ---------------------------------------------------------------------------
// SCF-level anchors
// ---------------------------------------------------------------------------

/// Fixed-grid gradient vs central difference of the fixed-grid energy, Ha/Bohr
/// (measured max 3.7e-10 over STO-3G and 6-31G, h = 5e-4).
const TOL_GRAD_FD: f64 = 2e-9;
/// Analytic Hessian vs central difference of the validated fixed-grid gradient,
/// Ha/Bohr^2 (measured max 1.9e-8 over STO-3G and 6-31G at h = 5e-4, which is
/// the stencil's h^4 truncation: the identical number appears for the
/// frozen-density skeleton alone, and it falls 16x per halving of h).
const TOL_HESS_FD: f64 = 1e-7;

#[test]
fn fixed_grid_scf_reproduces_the_production_energy_at_the_reference_geometry() {
    let r = reference("6-31g");
    let ctx = ParallelContext::default();
    let prod = solve_rhf(
        &ctx,
        &r.mol,
        &r.prep,
        Operator::coulomb(),
        &r.bounds,
        &production_config(Some(grid_cfg())),
    )
    .unwrap();
    let de = (prod.energy - r.fixed.res.energy).abs();
    eprintln!("production vs injected fixed-grid SCF energy: {de:.3e}");
    assert!(de < 1e-8, "{de:.3e}");
}

#[test]
fn gradient_matches_central_difference_of_the_fixed_grid_energy() {
    for basis_name in ["sto-3g", "6-31g"] {
        let r = reference(basis_name);
        let an = gradient_at(&r, &r.fixed);
        let fd = fd_gradient_of_energy(&r, H_FROZEN);
        let diff = max_abs_diff(&an, &fd);
        eprintln!(
            "{basis_name}: |grad_analytic - FD(E_fixed)| = {diff:.3e} (max |g| {:.3e})",
            max_abs(&an)
        );
        assert!(diff < TOL_GRAD_FD, "{basis_name}: {diff:.3e}");
    }
}

#[test]
fn hessian_matches_central_difference_of_the_fixed_grid_gradient() {
    if !has_deriv2() {
        return;
    }
    for basis_name in ["sto-3g", "6-31g"] {
        let r = reference(basis_name);
        let an = analytic(&r, &KsFixedGridMutants::default());
        let fd = fd_hessian_of_gradient(&r, H_FROZEN);
        let diff = max_abs_diff(&an, &fd);
        eprintln!(
            "{basis_name}: |H_analytic - FD(g_fixed)| = {diff:.3e} (max |H| {:.3e})",
            max_abs(&an)
        );
        assert!(diff < TOL_HESS_FD, "{basis_name}: {diff:.3e}");
    }
}

/// The entry point driven by the PRODUCTION SCF (grid built by `solve_rhf`,
/// not injected) agrees with the Hessian from the injected fixed-grid SCF: the
/// same energy surface, two SCF constructions.
#[test]
fn production_scf_result_gives_the_same_hessian() {
    if !has_deriv2() {
        return;
    }
    let r = reference("6-31g");
    let ctx = ParallelContext::default();
    let prod = solve_rhf(
        &ctx,
        &r.mol,
        &r.prep,
        Operator::coulomb(),
        &r.bounds,
        &production_config(Some(grid_cfg())),
    )
    .unwrap();
    let from_prod = rks_lda_fixed_grid_hessian(
        &ctx,
        &r.mol,
        &r.prep,
        Operator::coulomb(),
        &r.bounds,
        &prod,
        &hessian_config(),
    )
    .unwrap();
    let from_inj = analytic(&r, &KsFixedGridMutants::default());
    let diff = max_abs_diff(&from_prod, &from_inj);
    eprintln!("Hessian from production SCF vs injected fixed-grid SCF: {diff:.3e}");
    assert!(diff < TOL_SCF_NOISE, "{diff:.3e}");
}

/// Two SCFs of one energy surface: agreement bar for converged-density noise.
const TOL_SCF_NOISE: f64 = 5e-11;

/// A deliberately broken construction is caught by the Hessian anchor with a
/// miss of at least `MUTANT_FACTOR` times the agreement bar.
const MUTANT_FACTOR: f64 = 10.0;

#[test]
fn production_wired_mutants_each_fail_the_hessian_anchor() {
    if !has_deriv2() {
        return;
    }
    let r = reference("6-31g");
    let fd = fd_hessian_of_gradient(&r, H_FROZEN);
    let good = max_abs_diff(&analytic(&r, &KsFixedGridMutants::default()), &fd);
    assert!(good < TOL_HESS_FD, "production {good:.3e}");
    let xc = |f: fn(&mut XcMutants)| {
        let mut m = KsFixedGridMutants::default();
        f(&mut m.xc);
        m
    };
    let resp = |scale: f64| KsFixedGridMutants {
        response_fxc: scale,
        ..Default::default()
    };
    let cases: Vec<(&str, KsFixedGridMutants)> = vec![
        (
            "drop the XC skeleton (both terms)",
            xc(|m| {
                m.skeleton_fxc = 0.0;
                m.skeleton_vxc = 0.0;
            }),
        ),
        (
            "drop f'' rho_x rho_y in the skeleton",
            xc(|m| m.skeleton_fxc = 0.0),
        ),
        (
            "drop f' rho_xy in the skeleton",
            xc(|m| m.skeleton_vxc = 0.0),
        ),
        ("f' rho_xy skeleton x2", xc(|m| m.skeleton_vxc = 2.0)),
        ("drop f'' rho_x chi chi in F^x", xc(|m| m.fock_fxc = 0.0)),
        ("drop f' d(chi chi) in F^x", xc(|m| m.fock_vxc = 0.0)),
        ("drop the XC kernel in the CPHF", resp(0.0)),
        ("XC kernel in the CPHF x2", resp(2.0)),
        ("XC kernel in the CPHF x0.5", resp(0.5)),
    ];
    let mut weakest = f64::INFINITY;
    for (name, m) in cases {
        let miss = max_abs_diff(&analytic(&r, &m), &fd);
        eprintln!(
            "mutant [{name}]: misses the FD Hessian by {miss:.3e} ({:.1e} x bar)",
            miss / TOL_HESS_FD
        );
        weakest = weakest.min(miss);
        assert!(
            miss >= MUTANT_FACTOR * TOL_HESS_FD,
            "mutant [{name}] misses by only {miss:.3e}"
        );
    }
    eprintln!(
        "weakest mutant: {weakest:.3e} = {:.1e} x bar",
        weakest / TOL_HESS_FD
    );
}

/// NEGATIVE CONTROL. The production KS SCF puts its grid on the atoms, so its
/// gradient carries the Becke weight response. Differencing THAT gradient must
/// disagree with the fixed-grid analytic Hessian by the weight-response scale,
/// far above the fixed-grid agreement bar: the fixed-grid Hessian is a
/// different (and, for frequencies, not the production) surface.
#[test]
fn production_gradient_difference_differs_by_the_weight_response_scale() {
    if !has_deriv2() {
        return;
    }
    use ferric_scf::frequencies::{atom_masses, frequencies_from_cartesian_hessian};
    use ferric_scf::ks_gradient::ks_gradient_closed;
    let ctx = ParallelContext::default();
    let mol = water();
    let bs = basis::bundled("6-31g").unwrap();
    let op = Operator::coulomb();
    let cfg = production_config(None);
    let setup = |m: &Molecule| {
        let prep = PreparedBasis::new(m, &bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let res = solve_rhf(&ctx, m, &prep, op, &bounds, &cfg).unwrap();
        (prep, bounds, res)
    };
    let (prep, bounds, res) = setup(&mol);
    let fixed = rks_lda_fixed_grid_hessian(
        &ctx,
        &mol,
        &prep,
        op,
        &bounds,
        &res,
        &RhfConfig {
            dft_grid: None,
            ..hessian_config()
        },
    )
    .unwrap();
    let n3 = 9;
    let mut prod = Array2::<f64>::zeros((n3, n3));
    for b in 0..n3 {
        let col = rich(H_FROZEN, |s| {
            let m = displaced(&mol, b, s);
            let (p, bd, r) = setup(&m);
            flat(&ks_gradient_closed(&m, &p, &bs, op, &bd, "LDA", &r, None).unwrap())
        });
        prod.column_mut(b).assign(&col.column(0));
    }
    let prod = 0.5 * (&prod + &prod.t());
    let diff = max_abs_diff(&fixed, &prod);
    let masses = atom_masses(&mol).unwrap();
    let f_fixed = frequencies_from_cartesian_hessian(&mol, &fixed, &masses).unwrap();
    let f_prod = frequencies_from_cartesian_hessian(&mol, &prod, &masses).unwrap();
    let dnu = f_fixed
        .frequencies
        .iter()
        .zip(&f_prod.frequencies)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!(
        "fixed-grid analytic vs FD of the production (weight-response) gradient: max |dH| = \
         {diff:.3e} Ha/Bohr^2 ({:.1e} x the fixed-grid bar), max |d nu| = {dnu:.2} cm^-1",
        diff / TOL_HESS_FD
    );
    assert!(diff > MUTANT_FACTOR * TOL_HESS_FD, "{diff:.3e}");
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

fn refusal_with(config: &RhfConfig, r: &Ref, res: &ScfResult) -> String {
    let err = rks_lda_fixed_grid_hessian(
        &ParallelContext::default(),
        &r.mol,
        &r.fixed.prep,
        Operator::coulomb(),
        &r.fixed.bounds,
        res,
        config,
    )
    .expect_err("must be refused");
    err.to_string()
}

#[test]
fn unsupported_configurations_are_refused_naming_the_next_unit() {
    let r = reference("sto-3g");
    let res = &r.fixed.res;
    let with_xc = |name: &str| RhfConfig {
        xc: Some(name.into()),
        ..hessian_config()
    };
    for (xc, needle) in [
        ("PBE", "GGA"),
        ("PBE0", "GGA"),
        ("B3LYP", "GGA"),
        ("HSE06", "range-separated"),
        ("wB97X-V", "VV10"),
        ("SCAN", "meta-GGA"),
    ] {
        let msg = refusal_with(&with_xc(xc), &r, res);
        eprintln!("{xc}: {msg}");
        assert!(
            msg.contains(needle) && msg.contains("issue #407"),
            "{xc}: {msg}"
        );
    }
    let msg = refusal_with(
        &RhfConfig {
            xc: None,
            ..hessian_config()
        },
        &r,
        res,
    );
    assert!(msg.contains("Kohn-Sham configuration"), "{msg}");
    let msg = refusal_with(
        &RhfConfig {
            df_j_aux: Some("def2-universal-jkfit".into()),
            ..hessian_config()
        },
        &r,
        res,
    );
    assert!(
        msg.contains("density-fitted") && msg.contains("RI-J/RI-K"),
        "{msg}"
    );
    let msg = refusal_with(
        &RhfConfig {
            external_potential: Some(ferric_core::external_potential::ExternalPotential {
                field: Some([0.0, 0.0, 0.01]),
                ..Default::default()
            }),
            ..hessian_config()
        },
        &r,
        res,
    );
    assert!(msg.contains("external potential"), "{msg}");
}

#[test]
fn an_open_shell_reference_is_refused() {
    let r = reference("sto-3g");
    let uks = ferric_scf::uhf::solve_uhf(
        &ParallelContext::default(),
        &r.mol,
        &r.fixed.prep,
        &r.fixed.bounds,
        &production_config(Some(grid_cfg())),
    )
    .unwrap();
    let msg = refusal_with(&hessian_config(), &r, &uks);
    eprintln!("{msg}");
    assert!(
        msg.contains("open-shell") && msg.contains("issue #407"),
        "{msg}"
    );
}

/// On a first-derivative libint2 the second-derivative engines are typed
/// errors, never a panic (the wheel's libint has no deriv-2 integrals).
#[test]
fn without_second_derivative_integrals_the_hessian_is_a_typed_error() {
    if ferric_integrals::engine::libint_max_deriv_order() >= 2 {
        eprintln!("SKIP: this libint2 has second derivatives");
        return;
    }
    let r = reference("sto-3g");
    let out = rks_lda_fixed_grid_hessian(
        &ParallelContext::default(),
        &r.mol,
        &r.fixed.prep,
        Operator::coulomb(),
        &r.fixed.bounds,
        &r.fixed.res,
        &hessian_config(),
    );
    assert!(out.is_err());
}

/// d functions (the second-derivative path at l = 2 for both the XC AO
/// derivatives and the integrals) at a step small enough for the tight cc-pVDZ
/// oxygen s functions.
#[test]
fn hessian_with_d_functions_matches_the_fixed_grid_gradient_difference() {
    if !has_deriv2() {
        return;
    }
    let r = reference("cc-pvdz");
    let an = analytic(&r, &KsFixedGridMutants::default());
    let fd = fd_hessian_of_gradient(&r, 2.5e-4);
    let diff = max_abs_diff(&an, &fd);
    eprintln!(
        "cc-pvdz: |H_analytic - FD(g_fixed)| = {diff:.3e} (max |H| {:.3e})",
        max_abs(&an)
    );
    assert!(diff < TOL_HESS_FD, "{diff:.3e}");
}
