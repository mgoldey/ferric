//! Analytic UHF Hessian (`ferric_scf::hessian::uhf_hessian`) against
//! INDEPENDENT constructions: central finite differences of ferric's analytic
//! UHF gradient, and the already-validated RHF Hessian in the closed-shell
//! limit.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test uhf_hessian_fd
//! ```
//!
//! # What is compared to what
//!
//! * EXACTNESS ANCHOR (first): a converged closed-shell RHF result re-labelled
//!   as UHF (`C_α = C_β`, `D_α = D_β = D/2`) must give the RHF Hessian term by
//!   term. Every UHF-specific construction — the UHF Γ, the per-spin
//!   first-order Fock matrices, the coupled α/β CPHF, the per-spin response
//!   assembly with unit occupations — is exercised, and its trivial limit is
//!   the RHF code validated against PySCF. A second anchor runs a real
//!   `solve_uhf` on the closed-shell molecule (it converges back to RHF), so
//!   the SCF-side plumbing (β fields, occupations) is covered too.
//! * Skeleton terms 2–4 vs central differences of the UHF gradient pieces at a
//!   FROZEN D, W (and D_α, D_β for the 2e term):
//!   - `nuclear + one_electron` vs FD of `oneelectron_gradient(D_α + D_β, W = 0)`
//!   - `nuclear + overlap`      vs FD of `oneelectron_gradient(D = 0, W_UHF)`
//!   - `two_electron`           vs FD of `twoelectron_gradient_uhf(D, D_α, D_β)`
//! * The full Hessian vs central differences of `uhf_gradient`, a UHF SCF
//!   re-converged at every displaced geometry, each seeded from the
//!   undisplaced α/β MOs (a fresh guess can land another spin state or a
//!   symmetry-broken twin and fabricate an FD "Hessian"), with a basin check
//!   on the energy. Only this comparison sees the open-shell response.
//! * Internal consistency: raw response-term symmetry and translational
//!   invariance of the total.
//!
//! Systems: OH doublet (²Π, a tilted bond so the off-axis blocks are
//! populated) and CH₂ triplet (³B₁) distorted off C2v, STO-3G and 6-31G. The
//! reference SCF runs with the stability check and descent on and must come
//! out not-proven-unstable: the CPHF needs a stable minimum.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * Right: the closed-shell anchor agrees at CPHF_TOL (~1e-10); each skeleton
//!   term matches its frozen-density FD at O(h²) (~1e-8); the full Hessian
//!   matches FD of `uhf_gradient` at the SCF-noise/truncation floor (~1e-6);
//!   the raw response asymmetry is ~CPHF_TOL.
//! * Spin coupling wrong (cross-spin J dropped or doubled, K on the wrong
//!   spin, a factor 2 left over from RHF occupations): the closed-shell anchor
//!   misses by O(0.1) Ha/Bohr² — in that limit the α and β CPHF blocks are
//!   equal, so the error cannot cancel — and the open-shell FD test misses.
//! * Γ or W wrong for D_α ≠ D_β only (e.g. `gamma` with D_total in place of
//!   `gamma_uhf`): the closed-shell anchor PASSES (the two forms coincide there)
//!   and the open-shell skeleton FD test fails. That is why both kinds of test
//!   exist.
//! * FD reference broken (a displaced SCF in another basin): the basin check
//!   or the full-FD test fails column-wise while the skeleton tests and the
//!   response symmetry stay clean.
//! * OH ²Π caveat: rotating the β π hole about the bond is an exact zero mode
//!   of the UHF orbital Hessian. Every nuclear perturbation is orthogonal to it
//!   by symmetry, so the CPHF right-hand side has no component along it and
//!   conjugate gradient stays in the range; a CPHF failure or an OH-only miss
//!   points here first.
//!
//! # TOLERANCES
//!
//! Each bar sits above the maximum measured on this box, written next to it.
//!
//! # MUTATION (each must turn a named test red; see `src/hessian.rs`)
//!
//! * UHF Γ: in `uhf_hessian_parts` use `gamma(&d, ..)` in place of
//!   `gamma_uhf(&d, da, db, ..)` → `skeleton_two_electron_*` fail, the
//!   closed-shell anchors stay green (the two Γ coincide at D_α = D_β).
//! * UHF W: in `uhf_hessian_parts` build `w` with `2.0 *` → `skeleton_overlap`
//!   and every full/anchor test fail.
//! * per-spin first-order Fock: in `uhf_hessian_parts` set `k_scale: 0.5` →
//!   anchors and `full_hessian_*` fail, skeletons green.
//! * spin coupling: in `TwoElectronBuilder::g_uhf` return
//!   `[&ja - &ka, &jb - &kb]` (cross-spin J dropped) → both closed-shell
//!   anchors and every `full_hessian_*` fail, skeletons green.
//! * CPHF occupation: in `ucphf_response` build both frames with `occ` 2.0 →
//!   anchors and full tests fail.
//! * response assembly: in `ucphf_response` drop the `+ assemble_response(..β..)`
//!   term → anchors and full tests fail, `response_asymmetry` may stay small.
//! * occupation guard: in `state_refusal` return `None` before the aufbau
//!   check → `non_aufbau_occupations_are_refused` fails.
//!
//! Run 2026-09-25 (all seven caught): UHF Γ → the two-electron skeleton and
//! full tests fail, anchors green; W doubled → overlap skeleton, full and
//! anchors; k_scale 0.5 and cross-spin J dropped → full and anchors,
//! skeletons green; CPHF occupation 2 → 13 of 14; β response dropped → full,
//! anchors and invariance; occupation guard removed →
//! `non_aufbau_occupations_are_refused` only.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{
    build_energy_weighted_density_uhf, oneelectron_gradient, twoelectron_gradient_uhf, uhf_gradient,
};
use ferric_scf::hessian::{
    analytic_hessian_available, analytic_uhf_hessian_available, rhf_hessian, rhf_hessian_parts,
    uhf_hessian, uhf_hessian_parts, UhfHessianParts,
};
use ferric_scf::result::{ScfResult, Spin};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf_best_effort;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess};
use ndarray::Array2;

/// Closed-shell RHF re-labelled as UHF vs the RHF Hessian, per term,
/// Ha/Bohr². Same arithmetic up to summation order in the skeleton; the
/// response differs only by the CPHF residual. Measured 2.8e-17 (response;
/// the skeleton terms are bit-identical).
const TOL_ANCHOR_EXACT: f64 = 1e-12;
/// `solve_uhf` on the closed-shell molecule vs `rhf_hessian`, Ha/Bohr²: the two
/// SCFs agree only to their convergence. Measured 1.5e-11.
const TOL_ANCHOR_SCF: f64 = 1e-9;
/// Skeleton term vs FD of the frozen-density gradient piece, Ha/Bohr².
/// Measured ≤ 1.7e-8.
const TOL_SKELETON_FD: f64 = 3e-7;
/// Full analytic Hessian vs FD of `uhf_gradient` with re-converged SCFs,
/// Ha/Bohr². Measured ≤ 1.0e-6 (OH/6-31G), the FD's own floor.
const TOL_FULL_FD: f64 = 5e-6;
/// Raw response-term asymmetry, Ha/Bohr². Measured ≤ 3.3e-11.
const TOL_RESPONSE_ASYM: f64 = 1e-9;
/// Translational invariance of the total Hessian, Ha/Bohr². Measured 5.7e-12.
const TOL_TRANSLATION: f64 = 1e-9;
/// The response term must be at least this multiple of the bar it is judged
/// against, so a test cannot pass with term 5 silently zero.
const MUST_MATTER_FACTOR: f64 = 100.0;
/// A displaced SCF whose energy moves further than this from the reference
/// (Ha) at `H_FULL` has changed state; gradients here are ≲ 0.2 Ha/Bohr.
const BASIN_TOL: f64 = 1e-3;

/// FD step for the frozen-density skeleton checks (no SCF noise), Bohr.
const H_SKELETON: f64 = 1e-4;
/// FD step for the full Hessian (re-converged SCF at each point), Bohr.
const H_FULL: f64 = 1e-3;

/// OH ²Π with the bond tilted off every axis.
const OH_TILTED: &str = "2\nOH doublet, tilted bond\n\
O      0.000000     0.000000     0.000000\n\
H      0.120000    -0.080000     0.960000\n";

/// CH₂ ³B₁ distorted off C2v.
const CH2_DISTORTED: &str = "3\nCH2 triplet distorted off C2v\n\
C      0.000000     0.000000     0.000000\n\
H      0.990000     0.020000     0.420000\n\
H     -0.970000    -0.030000     0.440000\n";

const WATER_DISTORTED: &str = "3\nH2O distorted off C2v\n\
O      0.010000    -0.020000     0.120000\n\
H      0.030000     0.770000    -0.460000\n\
H     -0.020000    -0.740000    -0.490000\n";

fn oh() -> Molecule {
    Molecule::parse_xyz(OH_TILTED, 0, 2).unwrap()
}

fn ch2() -> Molecule {
    Molecule::parse_xyz(CH2_DISTORTED, 0, 3).unwrap()
}

fn water() -> Molecule {
    Molecule::parse_xyz(WATER_DISTORTED, 0, 1).unwrap()
}

/// Exact J/K, tight convergence. The analytic Hessian refuses RI.
fn scf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 300,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        ..Default::default()
    }
}

/// The reference SCF: stability-checked and descended to a stable state.
fn reference_config() -> RhfConfig {
    RhfConfig {
        check_stability: true,
        scf_stability_descent: true,
        ..scf_config()
    }
}

struct Setup {
    mol: Molecule,
    basis_name: String,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    uhf: ScfResult,
}

fn prep_for(m: &Molecule, basis_name: &str) -> (PreparedBasis, SchwarzBounds) {
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(m, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (prep, bounds)
}

fn setup(mol: &Molecule, basis_name: &str) -> Setup {
    let (prep, bounds) = prep_for(mol, basis_name);
    let uhf = solve_uhf(
        &ParallelContext::default(),
        mol,
        &prep,
        &bounds,
        &reference_config(),
    )
    .unwrap();
    assert!(
        uhf.converged,
        "{basis_name}: reference UHF did not converge"
    );
    let st = uhf
        .stability
        .as_ref()
        .expect("check_stability was set: the reference must carry a verdict");
    eprintln!(
        "{basis_name}: reference UHF E {:.10}, stability λ_min {:.3e} (noise floor {:.1e})",
        uhf.energy, st.lowest_eigenvalue, st.noise_floor
    );
    assert!(
        st.is_stable,
        "{basis_name}: reference UHF is proven unstable (λ_min {:.3e}); the CPHF needs a minimum",
        st.lowest_eigenvalue
    );
    Setup {
        mol: mol.clone(),
        basis_name: basis_name.to_string(),
        prep,
        bounds,
        uhf,
    }
}

fn parts(s: &Setup) -> UhfHessianParts {
    uhf_hessian_parts(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &s.uhf,
        &scf_config(),
    )
    .unwrap()
}

fn nocc(mol: &Molecule) -> (usize, usize) {
    let nelec = mol.nelec() as usize;
    let unpaired = mol.multiplicity - 1;
    ((nelec + unpaired) / 2, (nelec - unpaired) / 2)
}

fn beta_density(r: &ScfResult) -> Array2<f64> {
    r.density_beta.clone().expect("UHF β density")
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

/// Central-difference Hessian: column b = (g(x_b + h) − g(x_b − h)) / 2h.
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
        "{ctx}: max|got - want| = {d:.3e} (max|want| {:.3e}, tol {tol:.0e})",
        max_abs(want)
    );
    assert!(d < tol, "{ctx}: max|got - want| = {d:.3e} >= {tol:.0e}");
}

// ---------------------------------------------------------------------------
// Exactness anchors: closed-shell UHF == RHF
// ---------------------------------------------------------------------------

/// A converged RHF result presented as a UHF one with identical α/β orbitals.
fn rhf_as_uhf(rhf: &ScfResult) -> ScfResult {
    let mut u = rhf.clone();
    let half = 0.5 * rhf.density_r();
    u.spin = Spin::Unrestricted;
    u.density_alpha = half.clone();
    u.density_beta = Some(half);
    u.mos_beta = Some(rhf.mos_r().clone());
    u.eps_beta = Some(rhf.eps_r().to_vec());
    u.fock_beta = Some(rhf.fock_r().clone());
    u
}

fn rhf_setup(basis_name: &str) -> (Molecule, PreparedBasis, SchwarzBounds, ScfResult) {
    let mol = water();
    let (prep, bounds) = prep_for(&mol, basis_name);
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &scf_config(),
    )
    .unwrap();
    assert!(rhf.converged);
    (mol, prep, bounds, rhf)
}

#[test]
fn closed_shell_uhf_equals_rhf_term_by_term() {
    let (mol, prep, bounds, rhf) = rhf_setup("6-31g");
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let r = rhf_hessian_parts(&ctx, &mol, &prep, op, &bounds, &rhf, &scf_config()).unwrap();
    let u = uhf_hessian_parts(
        &ctx,
        &mol,
        &prep,
        op,
        &bounds,
        &rhf_as_uhf(&rhf),
        &scf_config(),
    )
    .unwrap();
    assert!(
        max_abs(&r.response) > MUST_MATTER_FACTOR * TOL_ANCHOR_EXACT,
        "vacuous: RHF response is ~0"
    );
    let terms = [
        ("nuclear", &u.nuclear, &r.nuclear),
        ("one_electron", &u.one_electron, &r.one_electron),
        ("overlap", &u.overlap, &r.overlap),
        ("two_electron", &u.two_electron, &r.two_electron),
        ("response", &u.response, &r.response),
    ];
    for (what, got, want) in terms {
        assert_close(
            &format!("closed-shell anchor {what}"),
            got,
            want,
            TOL_ANCHOR_EXACT,
        );
    }
    eprintln!(
        "anchor CPHF iterations: RHF {:?} UHF {:?}",
        r.cphf_iterations, u.cphf_iterations
    );
}

#[test]
fn closed_shell_solve_uhf_hessian_equals_rhf_hessian() {
    let (mol, prep, bounds, rhf) = rhf_setup("sto-3g");
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let uhf = solve_uhf(&ctx, &mol, &prep, &bounds, &scf_config()).unwrap();
    assert!(uhf.converged);
    assert!(
        (uhf.energy - rhf.energy).abs() < 1e-9,
        "closed-shell UHF did not return to RHF: {:.12} vs {:.12}",
        uhf.energy,
        rhf.energy
    );
    let hr = rhf_hessian(&ctx, &mol, &prep, op, &bounds, &rhf, &scf_config()).unwrap();
    let hu = uhf_hessian(&ctx, &mol, &prep, op, &bounds, &uhf, &scf_config()).unwrap();
    assert_close("solve_uhf closed-shell anchor", &hu, &hr, TOL_ANCHOR_SCF);
}

// ---------------------------------------------------------------------------
// Terms 2-4: skeleton vs frozen-density FD
// ---------------------------------------------------------------------------

fn check_skeleton_one_electron(mol: &Molecule, basis_name: &str) {
    let s = setup(mol, basis_name);
    let p = parts(&s);
    let d = &s.uhf.density_alpha + &beta_density(&s.uhf);
    let zero = Array2::<f64>::zeros(d.dim());
    let fd = fd_hessian(&s.mol, H_SKELETON, |m| {
        let (prep, _) = prep_for(m, basis_name);
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

fn check_skeleton_overlap(mol: &Molecule, basis_name: &str) {
    let s = setup(mol, basis_name);
    let p = parts(&s);
    let (na, nb) = nocc(&s.mol);
    let w = build_energy_weighted_density_uhf(&s.uhf, na, nb);
    let zero = Array2::<f64>::zeros(w.dim());
    let fd = fd_hessian(&s.mol, H_SKELETON, |m| {
        let (prep, _) = prep_for(m, basis_name);
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

fn check_skeleton_two_electron(mol: &Molecule, basis_name: &str) {
    let s = setup(mol, basis_name);
    let p = parts(&s);
    let da = s.uhf.density_alpha.clone();
    let db = beta_density(&s.uhf);
    let d = &da + &db;
    let fd = fd_hessian(&s.mol, H_SKELETON, |m| {
        let (prep, bounds) = prep_for(m, basis_name);
        twoelectron_gradient_uhf(&prep, Operator::coulomb(), &bounds, &d, &da, &db).unwrap()
    });
    assert!(
        max_abs(&p.two_electron) > 1e-1,
        "vacuous: 2e skeleton is ~0"
    );
    assert_close(
        &format!("{basis_name}: Γ_UHF·ERI''"),
        &p.two_electron,
        &fd,
        TOL_SKELETON_FD,
    );
}

#[test]
fn skeleton_one_electron_matches_fd_ch2_631g() {
    check_skeleton_one_electron(&ch2(), "6-31g");
}

#[test]
fn skeleton_overlap_matches_fd_ch2_631g() {
    check_skeleton_overlap(&ch2(), "6-31g");
}

#[test]
fn skeleton_two_electron_matches_fd_ch2_sto3g() {
    check_skeleton_two_electron(&ch2(), "sto-3g");
}

#[test]
fn skeleton_two_electron_matches_fd_ch2_631g() {
    check_skeleton_two_electron(&ch2(), "6-31g");
}

#[test]
fn skeleton_two_electron_matches_fd_oh_631g() {
    check_skeleton_two_electron(&oh(), "6-31g");
}

// ---------------------------------------------------------------------------
// Full Hessian (term 5 included) vs FD of the analytic UHF gradient
// ---------------------------------------------------------------------------

/// FD of `uhf_gradient`, every displaced UHF seeded from the reference MOs.
fn fd_full(s: &Setup) -> Array2<f64> {
    let op = Operator::coulomb();
    let ca = s.uhf.mos_alpha.clone();
    let cb = s.uhf.mos_beta.clone().expect("UHF β MOs");
    let e0 = s.uhf.energy;
    fd_hessian(&s.mol, H_FULL, |m| {
        let (prep, bounds) = prep_for(m, &s.basis_name);
        let r = solve_uhf_with_guess(
            &ParallelContext::default(),
            m,
            &prep,
            &bounds,
            &scf_config(),
            Some((&ca, &cb)),
        )
        .unwrap();
        assert!(r.converged, "displaced UHF did not converge");
        assert!(
            (r.energy - e0).abs() < BASIN_TOL,
            "displaced UHF changed state: E {:.10} vs reference {e0:.10}",
            r.energy
        );
        uhf_gradient(m, &prep, op, &bounds, &r, None).unwrap()
    })
}

fn check_full(mol: &Molecule, basis_name: &str) {
    let s = setup(mol, basis_name);
    let p = parts(&s);
    let fd = fd_full(&s);
    eprintln!(
        "{basis_name}: CPHF iterations {:?}, max residual {:.2e}, response asym {:.2e}",
        p.cphf_iterations, p.cphf_max_residual, p.response_asymmetry
    );
    // TEETH: the response must matter at this bar, and the skeleton alone
    // must miss the FD Hessian.
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
        &format!("{basis_name}: full UHF Hessian"),
        &p.total(),
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
fn full_hessian_matches_fd_oh_sto3g() {
    check_full(&oh(), "sto-3g");
}

#[test]
fn full_hessian_matches_fd_oh_631g() {
    check_full(&oh(), "6-31g");
}

#[test]
fn full_hessian_matches_fd_ch2_sto3g() {
    check_full(&ch2(), "sto-3g");
}

#[test]
fn full_hessian_matches_fd_ch2_631g() {
    check_full(&ch2(), "6-31g");
}

#[test]
fn hessian_is_symmetric_and_translationally_invariant() {
    let s = setup(&ch2(), "6-31g");
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
fn wrong_references_and_unsupported_configurations_are_refused() {
    let s = setup(&ch2(), "sto-3g");
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let cfg = scf_config();
    let refused_uhf = |res: &ScfResult, cfg: &RhfConfig, what: &str| {
        let err = uhf_hessian(&ctx, &s.mol, &s.prep, op, &s.bounds, res, cfg)
            .expect_err(&format!("{what} must be refused"));
        eprintln!("{what}: {err}");
    };

    // Best effort: the refusal is on the spin kind, whatever the ROHF exit.
    let rohf = solve_rohf_best_effort(&ctx, &s.mol, &s.prep, op, &s.bounds, &cfg).unwrap();
    refused_uhf(&rohf, &cfg, "ROHF result");
    let ks = RhfConfig {
        xc: Some("PBE".into()),
        ..scf_config()
    };
    refused_uhf(&s.uhf, &ks, "UKS config");
    let mut unconverged = s.uhf.clone();
    unconverged.converged = false;
    refused_uhf(&unconverged, &cfg, "unconverged SCF");

    // The RHF entry point refuses a UHF result, and the RHF probe refuses an
    // open shell that the UHF probe accepts.
    assert!(rhf_hessian(&ctx, &s.mol, &s.prep, op, &s.bounds, &s.uhf, &cfg).is_err());
    assert!(analytic_hessian_available(&s.mol, &s.prep, op, &cfg).is_err());
    analytic_uhf_hessian_available(&s.mol, &s.prep, op, &cfg)
        .expect("the UHF probe accepts a triplet");
    let (oh_prep, _) = prep_for(&oh(), "sto-3g");
    analytic_uhf_hessian_available(&oh(), &oh_prep, op, &cfg)
        .expect("the UHF probe accepts an odd electron count");

    // Negative control: the supported configuration is NOT refused.
    uhf_hessian(&ctx, &s.mol, &s.prep, op, &s.bounds, &s.uhf, &cfg)
        .expect("plain UHF must be supported");
}

#[test]
fn non_aufbau_occupations_are_refused() {
    let s = setup(&ch2(), "sto-3g");
    let (_, nb) = nocc(&s.mol);
    // Swap the β HOMO and LUMO columns without touching the density: the MOs
    // no longer describe the occupied space the density was built from (what
    // a MOM/excited-state result looks like to the response).
    let mut bad = s.uhf.clone();
    let cb = bad.mos_beta.as_mut().unwrap();
    let homo = cb.column(nb - 1).to_owned();
    let lumo = cb.column(nb).to_owned();
    cb.column_mut(nb - 1).assign(&lumo);
    cb.column_mut(nb).assign(&homo);
    let err = uhf_hessian(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &bad,
        &scf_config(),
    )
    .expect_err("a β MO set inconsistent with the β density must be refused");
    eprintln!("non-aufbau: {err}");
    assert!(
        err.to_string().contains("aufbau"),
        "refused for the wrong reason: {err}"
    );
}
