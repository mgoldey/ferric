//! Analytic ROHF Hessian (`ferric_scf::hessian::rohf_hessian`, issue #291
//! PR-3): the orbital response and the full matrix, against independent
//! constructions. The derivation is in the module doc of
//! `crates/ferric-scf/src/hessian/rohf.rs`. The public entry points still
//! refuse ROHF (PR-4), so everything here goes through the test-visible
//! functions.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test rohf_hessian_response
//! ```
//!
//! # What is compared to what
//!
//! * Closed-shell anchor: a singlet ROHF (real `solve_rohf`, and an RHF result
//!   relabelled as ROHF) gives the `rhf_hessian` FULL matrix.
//! * ferric's own central FD of `rohf_gradient` (SCF re-converged at every
//!   displaced geometry), Richardson over h = 2e-3/1e-3, on NH2 (doublet), CH2
//!   (triplet) and HO2 (doublet), at geometries distorted off every symmetry
//!   so all Cartesian blocks are populated. The comparison is the whole
//!   matrix, reported per atom column (central atom and each non-special atom).
//! * PySCF: central FD of PySCF's analytic ROHF gradient
//!   (`testdata/reference/validation/rohf_hessian/`), cc-pVDZ, Hessian and
//!   harmonic frequencies in cm^-1.
//! * Invariances: symmetry, translation (row sums) and rotation
//!   (`sum_b H_ab (w x r_b) = w x g_a`, which needs the gradient).
//!
//! # Hypotheses (written before measuring)
//!
//! * Physics (the response is right): analytic == FD to the FD floor on every
//!   system; PySCF-FD agreement at PySCF's own FD error (~1e-6); invariances
//!   hold to the CPHF tolerance.
//! * Artifact (the comparison cannot see the response): the skeleton alone (no
//!   response) would already agree. It does not: the response is the dominant
//!   part of the matrix, asserted below, so agreement is a statement about the
//!   response and not about the skeleton PR-2 already validated.
//! * Artifact (FD shares a defect with the analytic code): both use the same
//!   `rohf_gradient`, whose own validation is against PySCF's analytic
//!   gradient (<= 6.6e-11, in `rohf_hessian_skeleton`); the independent
//!   third leg is PySCF's own FD Hessian.
//!
//! # Negative controls and mutants
//!
//! Each must miss the FD reference by more than `MUST_MISS_FACTOR` x the bar.
//! The UHF Hessian evaluated at the ROHF orbitals (eigenvalue-form `W`,
//! spin-diagonal CPHF) is the control the skeleton PR did not run.
//!
//! IDENTITY (blind) candidates, algebra first: `D_a F_b D_b -> D_b F_b D_b` in
//! `W`: the difference is `sym(P_o F_b P_c)`, the closed-open block of `F_b`,
//! which `g_oc = 0` keeps zero along every perturbation, so it is invisible
//! also in `W^y` (the test records its size and does not count it).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::frequencies_from_cartesian_hessian;
use ferric_scf::gradient::rohf_gradient;
use ferric_scf::hessian::{
    rhf_hessian, rohf_hessian, rohf_hessian_parts, rohf_hessian_parts_mutated, uhf_hessian,
    RohfMutation,
};
use ferric_scf::result::{ScfResult, Spin};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use serde_json::Value;

/// Analytic vs ferric FD of `rohf_gradient`, Ha/Bohr^2.
/// Measured max 3.2e-10 (HO2/STO-3G), FD self-floor up to 5.7e-10; bar ~30x that.
const TOL_FD: f64 = 1e-8;
/// Singlet ROHF vs RHF full Hessian, Ha/Bohr^2.
/// Measured 1.2e-11.
const TOL_ANCHOR: f64 = 1e-8;
/// Row sums / symmetry of the full Hessian.
/// Measured <= 1.1e-11 (translation and rotation); the UNSYMMETRISED response
/// asymmetry, the real consistency probe, measured <= 2.7e-11.
const TOL_INVARIANCE: f64 = 1e-9;
/// Analytic vs PySCF FD Hessian (cc-pVDZ), Ha/Bohr^2.
/// Measured 5.6e-7..1.0e-6 (PySCF's own FD h-vs-2h error 0.7e-6..2.6e-6).
const TOL_PYSCF: f64 = 3e-6;
/// Frequencies vs PySCF FD frequencies, cm^-1.
/// Measured <= 0.0032 (PySCF's own h-vs-2h frequency gap is 0.0096 cm^-1).
const TOL_FREQ_CM: f64 = 0.03;
/// A mutant / control must miss by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 10.0;

/// FD step (Bohr); Richardson with h/2.
const H_FD: f64 = 2e-3;

const CH2_DISTORTED: &str = "3\nCH2 triplet distorted off C2v\n\
C      0.000000     0.000000     0.000000\n\
H      0.990000     0.020000     0.420000\n\
H     -0.970000    -0.030000     0.440000\n";

const NH2_DISTORTED: &str = "3\nNH2 doublet distorted off C2v\n\
N      0.000000     0.000000     0.000000\n\
H      0.010000     0.810000    -0.600000\n\
H      0.020000    -0.790000    -0.640000\n";

const HO2_LIFTED: &str = "3\nHO2 doublet, H out of plane\n\
O      0.000000     0.000000     0.000000\n\
O      1.331000     0.000000     0.050000\n\
H     -0.239672     0.940956     0.300000\n";

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

fn mol_of(xyz: &str, mult: usize) -> Molecule {
    Molecule::parse_xyz(xyz, 0, mult).unwrap()
}

/// Exact J/K, tight convergence.
fn scf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 500,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        ..Default::default()
    }
}

fn prep_for(m: &Molecule, basis_name: &str) -> (PreparedBasis, SchwarzBounds) {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(m, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (prep, bounds)
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

/// ROHF gradient with the SCF re-converged at `m`.
fn scf_gradient(m: &Molecule, basis_name: &str) -> Array2<f64> {
    let (prep, bounds) = prep_for(m, basis_name);
    let op = Operator::coulomb();
    let res = solve_rohf(
        &ParallelContext::default(),
        m,
        &prep,
        op,
        &bounds,
        &scf_config(),
    )
    .unwrap();
    assert!(res.converged, "displaced ROHF not converged");
    rohf_gradient(m, &prep, op, &bounds, &res, None).unwrap()
}

/// Central-difference Hessian: column b = (g(x_b + h) - g(x_b - h)) / 2h.
fn fd_hessian(
    mol: &Molecule,
    h: f64,
    grad: &mut impl FnMut(&Molecule) -> Array2<f64>,
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

/// `(4 D(h/2) - D(h)) / 3`.
fn fd_richardson(
    mol: &Molecule,
    h: f64,
    grad: &mut impl FnMut(&Molecule) -> Array2<f64>,
) -> Array2<f64> {
    let coarse = fd_hessian(mol, h, grad);
    let fine = fd_hessian(mol, 0.5 * h, grad);
    (4.0 * fine - coarse) / 3.0
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
        "{ctx}: max|got - want| = {d:.3e} (max|want| {:.3e}, tol {tol:.0e})",
        max_abs(want)
    );
    assert!(d < tol, "{ctx}: max|got - want| = {d:.3e} >= {tol:.0e}");
}

/// A mutant must miss `want` by more than `MUST_MISS_FACTOR` x `bar`.
fn assert_miss(ctx: &str, mutant: &Array2<f64>, want: &Array2<f64>, bar: f64) {
    let d = max_abs_diff(mutant, want);
    eprintln!(
        "MUTANT {ctx}: misses by {d:.3e} = {:.1}x the bar {bar:.0e}",
        d / bar
    );
    assert!(
        d > MUST_MISS_FACTOR * bar,
        "MUTANT {ctx} missed by only {d:.3e} (<= {MUST_MISS_FACTOR} x {bar:.0e}): the test \
         cannot see this defect"
    );
}

/// Per-atom-column maxima of `|a - b|`: column block `j` collects the columns
/// `3j..3j+3` (the perturbed atom `j`), all rows.
fn per_atom_column_miss(a: &Array2<f64>, b: &Array2<f64>) -> Vec<f64> {
    (0..a.ncols() / 3)
        .map(|j| {
            let mut m = 0.0f64;
            for r in 0..a.nrows() {
                for c in 3 * j..3 * j + 3 {
                    m = m.max((a[(r, c)] - b[(r, c)]).abs());
                }
            }
            m
        })
        .collect()
}

// ---------------------------------------------------------------------------
// One cached case per system: SCF, analytic Hessian, FD references
// ---------------------------------------------------------------------------

struct Case {
    mol: Molecule,
    basis_name: &'static str,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    rohf: ScfResult,
    analytic: ferric_scf::hessian::RhfHessianParts,
    /// Richardson FD (h = 1e-3, 5e-4) of the re-converged analytic gradient.
    fd: Array2<f64>,
    /// The FD's own self-disagreement: Richardson (1e-3, 5e-4) vs (2e-3, 1e-3).
    fd_floor: f64,
}

fn build_case(xyz: &str, mult: usize, basis_name: &'static str) -> Case {
    let mol = mol_of(xyz, mult);
    let (prep, bounds) = prep_for(&mol, basis_name);
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let rohf = solve_rohf(&ctx, &mol, &prep, op, &bounds, &scf_config()).unwrap();
    assert!(rohf.converged);
    let analytic =
        rohf_hessian_parts(&ctx, &mol, &prep, op, &bounds, &rohf, &scf_config()).unwrap();
    let mut grad = |m: &Molecule| scf_gradient(m, basis_name);
    let fd = fd_richardson(&mol, H_FD, &mut grad);
    let fd_fine = fd_richardson(&mol, 0.5 * H_FD, &mut grad);
    let fd_floor = max_abs_diff(&fd, &fd_fine);
    eprintln!(
        "{basis_name}/{}atoms mult {mult}: FD self-floor (Richardson 2e-3 vs 1e-3) {fd_floor:.3e}, \
         response CPHF iterations max {}, residual {:.2e}, asymmetry {:.2e}",
        mol.atoms.len(),
        analytic.cphf_iterations.iter().max().unwrap(),
        analytic.cphf_max_residual,
        analytic.response_asymmetry
    );
    Case {
        mol,
        basis_name,
        prep,
        bounds,
        rohf,
        analytic,
        fd: fd_fine,
        fd_floor,
    }
}

fn nh2() -> &'static Case {
    static C: OnceLock<Case> = OnceLock::new();
    C.get_or_init(|| build_case(NH2_DISTORTED, 2, "6-31g"))
}

fn ch2() -> &'static Case {
    static C: OnceLock<Case> = OnceLock::new();
    C.get_or_init(|| build_case(CH2_DISTORTED, 3, "6-31g"))
}

fn ho2() -> &'static Case {
    static C: OnceLock<Case> = OnceLock::new();
    C.get_or_init(|| build_case(HO2_LIFTED, 2, "sto-3g"))
}

impl Case {
    fn total(&self) -> Array2<f64> {
        self.analytic.total()
    }

    fn mutated(&self, m: RohfMutation) -> Array2<f64> {
        rohf_hessian_parts_mutated(
            &ParallelContext::default(),
            &self.mol,
            &self.prep,
            Operator::coulomb(),
            &self.bounds,
            &self.rohf,
            &scf_config(),
            m,
        )
        .unwrap()
        .total()
    }

    fn tag(&self) -> String {
        format!("{}/{}atoms", self.basis_name, self.mol.atoms.len())
    }
}

// ---------------------------------------------------------------------------
// Closed-shell anchor
// ---------------------------------------------------------------------------

#[test]
fn singlet_rohf_hessian_equals_rhf_hessian() {
    if !has_deriv2() {
        return;
    }
    let mol = mol_of(WATER_DISTORTED, 1);
    let (prep, bounds) = prep_for(&mol, "6-31g");
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let cfg = scf_config();
    let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let want = rhf_hessian(&ctx, &mol, &prep, op, &bounds, &rhf, &cfg).unwrap();

    let rohf = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let got = rohf_hessian(&ctx, &mol, &prep, op, &bounds, &rohf, &cfg).unwrap();
    assert_close(
        "singlet solve_rohf vs RHF (full matrix)",
        &got,
        &want,
        TOL_ANCHOR,
    );

    let mut relabelled = rhf.clone();
    let half = 0.5 * rhf.density_r();
    relabelled.spin = Spin::RestrictedOpen;
    relabelled.density_alpha = half.clone();
    relabelled.density_beta = Some(half);
    relabelled.rohf_spin_focks = Some((rhf.fock_r().clone(), rhf.fock_r().clone()));
    let got2 = rohf_hessian(&ctx, &mol, &prep, op, &bounds, &relabelled, &cfg).unwrap();
    assert_close(
        "relabelled RHF vs RHF (full matrix)",
        &got2,
        &want,
        TOL_ANCHOR,
    );
    assert!(
        max_abs(&(&want - &rohf_hessian_skeleton_only(&mol, &prep, &bounds, &rohf))) > 1e-2,
        "vacuous: the response is ~0 in the closed-shell anchor"
    );
}

fn rohf_hessian_skeleton_only(
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rohf: &ScfResult,
) -> Array2<f64> {
    ferric_scf::hessian::rohf_skeleton_parts(
        mol,
        prep,
        Operator::coulomb(),
        bounds,
        rohf,
        &scf_config(),
    )
    .unwrap()
    .skeleton()
}

// ---------------------------------------------------------------------------
// FD of rohf_gradient
// ---------------------------------------------------------------------------

fn check_fd(c: &Case) {
    let h = c.total();
    let per_atom = per_atom_column_miss(&h, &c.fd);
    eprintln!(
        "{}: per-perturbed-atom max|H - FD| = {per_atom:?}, FD floor {:.3e}",
        c.tag(),
        c.fd_floor
    );
    assert_close(&format!("{} analytic vs FD", c.tag()), &h, &c.fd, TOL_FD);
    assert!(
        c.analytic.response_asymmetry < TOL_INVARIANCE,
        "{}: unsymmetrised response asymmetry {:.3e}",
        c.tag(),
        c.analytic.response_asymmetry
    );
    // The response is a large part of the answer (the artifact hypothesis:
    // the skeleton alone would agree).
    let resp = max_abs(&c.analytic.response);
    assert!(resp > 1e-1, "vacuous: response max {resp:.3e}");
}

#[test]
fn analytic_matches_fd_of_rohf_gradient_nh2() {
    if has_deriv2() {
        check_fd(nh2());
    }
}

#[test]
fn analytic_matches_fd_of_rohf_gradient_ch2_triplet() {
    if has_deriv2() {
        check_fd(ch2());
    }
}

#[test]
fn analytic_matches_fd_of_rohf_gradient_ho2() {
    if has_deriv2() {
        check_fd(ho2());
    }
}

// ---------------------------------------------------------------------------
// Invariances
// ---------------------------------------------------------------------------

fn check_invariances(c: &Case) {
    let h = c.total();
    let n3 = h.nrows();
    let asym = max_abs_diff(&h, &h.t().to_owned());
    let mut worst_t = 0.0f64;
    for a in 0..n3 {
        for axis in 0..3 {
            let s: f64 = (0..n3 / 3).map(|b| h[(a, 3 * b + axis)]).sum();
            worst_t = worst_t.max(s.abs());
        }
    }
    // Rotation: sum_b H_ab (w x r_b) = w x g_a for each axis w.
    let g = rohf_gradient(
        &c.mol,
        &c.prep,
        Operator::coulomb(),
        &c.bounds,
        &c.rohf,
        None,
    )
    .unwrap();
    let natoms = n3 / 3;
    let mut worst_r = 0.0f64;
    for w in 0..3 {
        let mut e = [0.0f64; 3];
        e[w] = 1.0;
        let cross = |v: [f64; 3]| {
            [
                e[1] * v[2] - e[2] * v[1],
                e[2] * v[0] - e[0] * v[2],
                e[0] * v[1] - e[1] * v[0],
            ]
        };
        let rot: Vec<f64> = (0..natoms)
            .flat_map(|b| {
                let a = &c.mol.atoms[b];
                cross([a.x, a.y, a.zpos])
            })
            .collect();
        for a in 0..natoms {
            let rhs = cross([g[(a, 0)], g[(a, 1)], g[(a, 2)]]);
            for i in 0..3 {
                let lhs: f64 = (0..n3).map(|b| h[(3 * a + i, b)] * rot[b]).sum();
                worst_r = worst_r.max((lhs - rhs[i]).abs());
            }
        }
    }
    eprintln!(
        "{}: asymmetry {asym:.3e}, translation {worst_t:.3e}, rotation {worst_r:.3e} (max|g| {:.2e})",
        c.tag(),
        g.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    );
    assert!(g.iter().any(|v| v.abs() > 1e-3), "vacuous: gradient ~0");
    assert!(asym < TOL_INVARIANCE, "asymmetry {asym:.3e}");
    assert!(worst_t < TOL_INVARIANCE, "translation {worst_t:.3e}");
    assert!(worst_r < TOL_INVARIANCE, "rotation {worst_r:.3e}");
}

#[test]
fn hessian_is_symmetric_and_translation_rotation_invariant() {
    if !has_deriv2() {
        return;
    }
    for c in [nh2(), ch2(), ho2()] {
        check_invariances(c);
    }
}

// ---------------------------------------------------------------------------
// Negative controls and mutants
// ---------------------------------------------------------------------------

/// The UHF Hessian evaluated at the ROHF orbitals: MO energies are the
/// diagonal of the spin Focks (the best a UHF code can read from them).
fn uhf_at_rohf_orbitals(
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    rohf: &ScfResult,
) -> Result<Array2<f64>, ferric_core::FerricError> {
    let (fa, fb) = rohf.rohf_spin_focks.clone().unwrap();
    let cm = &rohf.mos_alpha;
    let diag = |f: &Array2<f64>| {
        let m = cm.t().dot(f).dot(cm);
        (0..m.nrows()).map(|i| m[(i, i)]).collect::<Vec<f64>>()
    };
    let mut u = rohf.clone();
    u.spin = Spin::Unrestricted;
    u.mos_beta = Some(cm.clone());
    u.eps_alpha = diag(&fa);
    u.eps_beta = Some(diag(&fb));
    uhf_hessian(
        &ParallelContext::default(),
        mol,
        prep,
        Operator::coulomb(),
        bounds,
        &u,
        &scf_config(),
    )
}

/// The UHF-at-ROHF-orbitals control: a miss by > 10x the bar, OR a typed error
/// (its spin-diagonal CPHF is not positive definite at an ROHF point: seen on
/// HO2/cc-pVDZ, where the control is caught by refusing to answer).
fn assert_uhf_control(
    ctx: &str,
    got: Result<Array2<f64>, ferric_core::FerricError>,
    want: &Array2<f64>,
    bar: f64,
) {
    match got {
        Ok(h) => assert_miss(ctx, &h, want, bar),
        Err(e) => eprintln!("MUTANT {ctx}: refused with a typed error ({e})"),
    }
}

fn check_controls(c: &Case) {
    let tag = c.tag();
    let skeleton_only = rohf_hessian_skeleton_only(&c.mol, &c.prep, &c.bounds, &c.rohf);
    assert_miss(
        &format!("{tag} NO response (skeleton only)"),
        &skeleton_only,
        &c.fd,
        TOL_FD,
    );
    assert_uhf_control(
        &format!("{tag} UHF Hessian at ROHF orbitals"),
        uhf_at_rohf_orbitals(&c.mol, &c.prep, &c.bounds, &c.rohf),
        &c.fd,
        TOL_FD,
    );
}

#[test]
fn negative_controls_miss_on_all_three_systems() {
    if !has_deriv2() {
        return;
    }
    for c in [nh2(), ch2(), ho2()] {
        check_controls(c);
    }
}

fn check_mutants(c: &Case) {
    let tag = c.tag();
    let real = [
        ("drop the oc block", RohfMutation::DropOcBlock),
        (
            "diagonal-gap operator instead of the commutator",
            RohfMutation::DiagonalGapOperator,
        ),
        ("drop -Tr[W^y S^x]", RohfMutation::DropWOverlap),
        (
            "one spin's F^y in the D_a F_b^y D_b term of W^y",
            RohfMutation::OneSpinFockInW,
        ),
        ("F_a^x for both spins", RohfMutation::OneSpinFockX),
        ("alpha/beta F^x swapped", RohfMutation::SwapSpinFockX),
        (
            "drop G(D^S) (overlap part of the density change)",
            RohfMutation::DropOverlapDensityG,
        ),
        (
            "flip the sign of the overlap-Fock term -1/2(S f + f S)",
            RohfMutation::FlipOverlapFockSign,
        ),
    ];
    for (name, m) in real {
        assert_miss(&format!("{tag} {name}"), &c.mutated(m), &c.fd, TOL_FD);
    }
    // Identity candidate: recorded, never counted.
    let delta = max_abs_diff(&c.mutated(RohfMutation::DbFbDbInW), &c.total());
    eprintln!(
        "{tag} IDENTITY mutant D_a F_b D_b -> D_b F_b D_b in W^y: |dH| = {delta:.3e} \
         (blind: closed-open block of F_b stays zero)"
    );
    assert!(
        delta < TOL_FD,
        "the identity mutant moved the Hessian by {delta:.3e}"
    );
}

#[test]
fn response_mutants_are_caught_nh2() {
    if has_deriv2() {
        check_mutants(nh2());
    }
}

#[test]
fn response_mutants_are_caught_ch2_triplet() {
    if has_deriv2() {
        check_mutants(ch2());
    }
}

#[test]
fn response_mutants_are_caught_ho2() {
    if has_deriv2() {
        check_mutants(ho2());
    }
}

// ---------------------------------------------------------------------------
// PySCF references (cc-pVDZ): Hessian and frequencies
// ---------------------------------------------------------------------------

const ROW_DIR: &str = "testdata/reference/validation/rohf_hessian";
const MOL_DIR: &str = "testdata/molecules/validation";

fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("testdata").is_dir() && p.join("crates").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-scf manifest dir should be <root>/crates/ferric-scf")
        .to_path_buf()
}

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_cc-pvdz.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing reference {} ({e}); regenerate with scripts/validation/gen_rohf_hessian.py \
             — a missing reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn matrix(v: &Value, ptr: &str) -> Array2<f64> {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("reference field {ptr} missing"));
    let (n, m) = (rows.len(), rows[0].as_array().unwrap().len());
    let mut out = Array2::<f64>::zeros((n, m));
    for (i, row) in rows.iter().enumerate() {
        for (j, x) in row.as_array().unwrap().iter().enumerate() {
            out[(i, j)] = x.as_f64().unwrap();
        }
    }
    out
}

fn floats(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("reference field {key} missing"))
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn check_pyscf(system: &str) {
    if !has_deriv2() {
        return;
    }
    let r = reference(system);
    let mult = r["multiplicity"].as_u64().unwrap() as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, mult).unwrap();
    let (prep, bounds) = prep_for(&mol, "cc-pvdz");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let rohf = solve_rohf(&ctx, &mol, &prep, op, &bounds, &scf_config()).unwrap();
    let t0 = std::time::Instant::now();
    let h = rohf_hessian(&ctx, &mol, &prep, op, &bounds, &rohf, &scf_config()).unwrap();
    eprintln!(
        "{system}/cc-pVDZ analytic ROHF Hessian: {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    let h = 0.5 * (&h + &h.t());
    let h1 = matrix(&r, "/hessian_fd");
    let h2 = matrix(&r, "/hessian_fd_2h");
    eprintln!(
        "{system}: PySCF's own FD error (h vs 2h) {:.3e}",
        max_abs_diff(&h1, &h2)
    );
    let d1 = max_abs_diff(&h, &h1);
    let d2 = max_abs_diff(&h, &h2);
    let per_atom = per_atom_column_miss(&h, &h1);
    eprintln!("{system}: analytic vs PySCF FD (h) {d1:.3e}, (2h) {d2:.3e}, per atom {per_atom:?}");
    assert!(d1 < TOL_PYSCF, "{system}: Hessian vs PySCF FD {d1:.3e}");
    // Negative controls against the same reference.
    assert_miss(
        &format!("{system}/cc-pVDZ NO response (skeleton only) vs PySCF"),
        &rohf_hessian_skeleton_only(&mol, &prep, &bounds, &rohf),
        &h1,
        TOL_PYSCF,
    );
    assert_uhf_control(
        &format!("{system}/cc-pVDZ UHF Hessian at ROHF orbitals vs PySCF"),
        uhf_at_rohf_orbitals(&mol, &prep, &bounds, &rohf),
        &h1,
        TOL_PYSCF,
    );

    let masses = floats(&r, "masses_amu");
    let f = frequencies_from_cartesian_hessian(&mol, &h, &masses).unwrap();
    let f_ref = floats(&r, "freq_fd_cm");
    let f_ref2 = floats(&r, "freq_fd_2h_cm");
    let dev = f
        .frequencies
        .iter()
        .zip(&f_ref)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!(
        "{system}: ferric analytic frequencies {:?} vs PySCF FD {:?} (2h {:?}): max|d| {dev:.4} cm^-1",
        f.frequencies, f_ref, f_ref2
    );
    assert_eq!(f.frequencies.len(), f_ref.len());
    assert!(
        dev < TOL_FREQ_CM,
        "{system}: frequencies off by {dev:.4} cm^-1"
    );
}

#[test]
fn analytic_matches_pyscf_fd_nh2() {
    check_pyscf("nh2");
}

#[test]
fn analytic_matches_pyscf_fd_ch2_triplet() {
    check_pyscf("ch2_triplet");
}

#[test]
#[ignore = "HO2/cc-pVDZ analytic Hessian + two control Hessians: ~15 s release, minutes in debug; run once"]
fn analytic_matches_pyscf_fd_ho2() {
    check_pyscf("ho2");
}

// ---------------------------------------------------------------------------
// Degenerate open shell and typed errors
// ---------------------------------------------------------------------------

/// Linear OH (2Pi): the open pi orbital is degenerate with the empty one. The
/// orbital Hessian has a zero mode (the hole rotating about the axis), but no
/// nuclear perturbation of the A' hole state drives it (the A' and A''
/// components do not couple), so the response is well defined and must equal
/// the FD of the gradient. Artifact hypothesis: the CG would stall or return a
/// huge rotation if the zero mode were excited; it converges in 11-14 CG
/// iterations with max |kappa| <= 0.45 per Bohr (measured, linear and slightly
/// bent geometries).
#[test]
fn linear_oh_zero_mode_is_not_driven_and_matches_fd() {
    if !has_deriv2() {
        return;
    }
    let mol = mol_of("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 2);
    let (prep, bounds) = prep_for(&mol, "6-31g");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let rohf = solve_rohf(&ctx, &mol, &prep, op, &bounds, &scf_config()).unwrap();
    let h = rohf_hessian(&ctx, &mol, &prep, op, &bounds, &rohf, &scf_config()).unwrap();
    let mut grad = |m: &Molecule| scf_gradient(m, "6-31g");
    let fd = fd_richardson(&mol, H_FD, &mut grad);
    assert_close("linear OH analytic vs FD", &h, &fd, TOL_FD);
}

/// The typed-error path of the response solve. No physical near-singular ROHF
/// reference was found (linear OH above is benign), so this exercises it on an
/// artificially indefinite operator: spin Focks with the sign flipped make
/// every orbital-Hessian diagonal negative, so the CG must report
/// non-positive curvature instead of returning a Hessian.
#[test]
fn an_indefinite_orbital_hessian_is_a_typed_error() {
    if !has_deriv2() {
        return;
    }
    let c = ch2();
    let mut bad = c.rohf.clone();
    let (fa, fb) = bad.rohf_spin_focks.clone().unwrap();
    bad.rohf_spin_focks = Some((-fa, -fb));
    let err = rohf_hessian(
        &ParallelContext::default(),
        &c.mol,
        &c.prep,
        Operator::coulomb(),
        &c.bounds,
        &bad,
        &scf_config(),
    )
    .expect_err("an indefinite orbital Hessian must not return a Hessian");
    eprintln!("typed error: {err}");
    assert!(
        err.to_string().contains("not positive definite")
            || err.to_string().contains("did not converge"),
        "{err}"
    );
}
