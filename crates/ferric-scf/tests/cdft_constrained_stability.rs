//! Are the HeNe⁺ cDFT diabats A and B genuine MINIMA of their own constrained
//! manifolds, or merely stationary points?
//!
//! Hypotheses were written down BEFORE measuring, in
//! `tests/HYPOTHESES-constrained-stability.md`. Read that first; it also states
//! the operator question below, so the answer cannot be retrofitted.
//!
//! # The operator (this is the whole task)
//!
//! A constrained SCF solution is NOT expected to be a minimum of the bare
//! `E[ρ]`. The cDFT Lagrangian `W[ρ,λ] = E[ρ] + λ(N_C[ρ] − N_target)` is a
//! saddle in the combined (ρ, λ) space BY CONSTRUCTION — minimized over ρ,
//! maximized over λ. An ordinary `E[ρ]` stability analysis at a constrained
//! solution would report spurious instabilities on perfectly good diabats.
//!
//! The correct check is internal stability of the λ-AUGMENTED problem AT FIXED
//! CONVERGED λ: is the state a minimum of `E[ρ] + λ N_C[ρ]` over orbital
//! rotations?
//!
//! # The structural simplification, VERIFIED here and not assumed
//!
//! In `cdft_driver.rs` the constraint enters as `lw = l * &w_mats[ci]` — a
//! FIXED one-electron operator. `W` is built once from geometry + the Becke
//! grid, OUTSIDE the λ loop, and is never rebuilt from the density. A
//! density-independent one-electron term contributes to the Fock but NOT to the
//! Fock response `δF/δD`. So the λ-augmented orbital Hessian should EQUAL the
//! ordinary UHF orbital Hessian evaluated at the constrained density and
//! orbitals, with the CONSTRAINED orbital energies — which is exactly what
//! `uhf_newton::hessian_matvec` computes when fed the constrained `C` and the
//! constrained MO-basis Fock.
//!
//! `augmented_hessian_equals_plain_hessian_at_fixed_lambda` FINITE-DIFFERENCES
//! that claim against the analytic matvec before any verdict below relies on
//! it. It is a prerequisite of every stability number in this file.
//!
//! # What the constrained solve hands back
//!
//! `solve_uhf_fockmod` applies the Fock modifier AFTER forming `e_elec_no_xc`,
//! so `ScfResult::energy` is the ordinary (bare) energy at the constrained
//! density, while `fock_alpha`/`fock_beta` — and therefore `eps_alpha`/
//! `eps_beta`, which come from diagonalizing them — ARE λ-augmented. That is
//! the combination the augmented Hessian needs, and it is why no library change
//! is required.
//!
//! # Eigensolver, and why it is the LIBRARY's
//!
//! `lowest_eigenvalue` delegates to `ferric_scf::stability::
//! uhf_internal_stability`, the validated block Davidson driven by
//! `hessian_matvec`. It is NOT a local eigensolver, and must not become one
//! again.
//!
//! This file originally carried a private ~110-line single-root Davidson, and
//! that solver returned a CONVERGED WRONG ANSWER on the water/STO-3G anchor:
//!
//! ```text
//!   dense (ground truth)   λ_min = +3.6243948468e-1
//!   private single-root    λ_min = +3.6883351924e-1  resid 6.63e-11, 2 iters
//!   library block solver   λ_min = +3.6243948468e-1  resid 3.72e-15, 7 iters
//! ```
//!
//! The orbital Hessian is block diagonal in the molecule's irreps. On water
//! (C2v) the unit-vector seed lands inside a closed 2×2 block whose own lowest
//! root is +3.6883e-1, and a single-root iteration converges there to 6.63e-11
//! without ever seeing the true λ_min. A RESIDUAL CHECK CANNOT CATCH THIS — the
//! iteration genuinely converged, just on the wrong root. The old assertion
//! (`λ_min > 1e-3`) then accepted the wrong value by a factor of 360.
//!
//! The library solver defends with three safeguards its own docs call
//! load-bearing: block tracking (`n_block >= 2`), a dense symmetry-breaking
//! seed, and block-wide convergence.
//!
//! # The durable guard: a DENSE cross-check on EVERY anchor
//!
//! Swapping the solver fixes today's bug. What catches tomorrow's is that every
//! λ_min in this file is additionally checked against a DENSE Hessian built by
//! applying the SAME `hessian_matvec` to every unit vector and diagonalizing
//! (`dense_lowest_eigenvalue` / `assert_matches_dense`). Every system here has
//! a rotation space of at most 148 dimensions, so this costs essentially
//! nothing, and it is an INDEPENDENT construction — which is the only kind of
//! agreement that distinguishes signal from systematic error. That the guard
//! is REACHABLE (it fails on the wrong value, not merely passes on the right
//! one) is pinned by `dense_cross_check_rejects_the_symmetry_block_root`.
//!
//! Verdicts are read from `StabilityResult::verdict()` — the total four-way
//! `StabilityVerdict` — never from the `is_stable` bool, which is a
//! "not-proven-unstable" flag that reads `true` for a negative λ_min anywhere
//! in the marginal band.
//!
//! Geometry convention: He at the origin, Ne at (0, 0, R), so the bond axis is
//! z.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::ao_grid::eval_basis_on_points;
use ferric_dft::cdft::{build_weight_matrix, population, Constraint, SpinChannel};
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::cdft_driver::solve_cdft_uhf;
use ferric_scf::engine_pool::EnginePool;
use ferric_scf::rhf::{build_jk, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::{
    uhf_internal_stability, StabilityConfig, StabilityResult, StabilityVerdict,
};
use ferric_scf::uhf::{solve_uhf, solve_uhf_fockmod};
use ferric_scf::uhf_newton::{hessian_matvec, UhfNewtonInputs};
use ndarray::Array2;
use ndarray_linalg::{Eigh, Solve, UPLO};

/// Same tolerance as the lane under test. NOT tuned here.
const HENE_LAMBDA_TOL: f64 = 1e-5;
/// Same level shift as the lane under test, for the same reason.
const HENE_LEVEL_SHIFT: f64 = 0.5;
const R_ANG: f64 = 2.0;
/// Natural (promolecule) Becke population of He at R = 2.0 Å, measured on
/// `test/cdft-atomic-ip-anchor` (794a1551). State B's integer target of 2.000
/// drags the density 0.046 e past this.
const HENE_NATURAL_N_HE: f64 = 1.954484;

// ===========================================================================
// Deterministic PRNG (no external rand dep) — same generator as
// uhf_newton_smoke.rs so the two FD checks are directly comparable.
// ===========================================================================

struct Xorshift64(u64);
impl Xorshift64 {
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

// ===========================================================================
// Shared scaffolding
// ===========================================================================

struct Sys {
    mol: Molecule,
    bs: basis::BasisSet,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
    nocc_a: usize,
    nocc_b: usize,
}

fn build_sys(xyz: &str, charge: i32, mult: usize, basis_name: &str) -> Sys {
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    let nocc_a = (nelec + two_s) / 2;
    let nocc_b = (nelec - two_s) / 2;
    Sys {
        mol,
        bs,
        prep,
        bounds,
        ctx,
        nocc_a,
        nocc_b,
    }
}

fn hene_xyz() -> String {
    format!("2\nHeNe+\nHe 0.0 0.0 0.0\nNe 0.0 0.0 {R_ANG}\n")
}

/// The lane-under-test's config, reproduced EXACTLY.
///
/// `energy_conv`/`density_conv` are deliberately left at their defaults, as on
/// `test/cdft-et-second-system` (04a7ea88). This is not cosmetic: tightening
/// them to 1e-11/1e-9 changes the λ-Newton path enough that state B's outer
/// loop no longer converges in its 30-iteration budget. The states audited here
/// must be the lane's states, so the lane's knobs are used verbatim.
///
/// # `cdft_stability_descent: false` — why this file OPTS OUT of the fix
///
/// ADDED 2026-09-16 on `fix/cdft-state-selection`, when the driver gained a
/// stability-guided descent that DEFAULTS ON. This file is the AUDIT of the
/// pre-fix state: every verdict in it is a measurement OF the saddle that
/// descent now escapes. Leaving the descent enabled here would make the file
/// silently re-measure the post-descent state and its documented λ_min values
/// (−3.999e-2 for B, the 0.6671 eV drop, the natural-vs-integer comparison)
/// would evaporate — not because the defect was understood differently, but
/// because the thing being audited would no longer be running.
///
/// So the audit is pinned to the UNFIXED solver deliberately. That keeps it a
/// historical record AND keeps it live as a regression test: if the saddle ever
/// stops being a saddle with the descent OFF, that is a real change in the
/// underlying SCF and this file will say so. The FIXED behavior is asserted
/// separately, in `tests/cdft_state_selection.rs`.
fn hene_cfg() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        level_shift: HENE_LEVEL_SHIFT,
        cdft_lambda_tol: HENE_LAMBDA_TOL,
        cdft_stability_descent: false,
        dft_grid: Some(AtomicGridConfig {
            n_radial: 99,
            n_angular: 302,
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Tightly-converged variant for the UNCONSTRAINED reference states and the FD
/// check. A stability verdict read off a loosely-converged state describes a
/// point that is not actually stationary, so the anchors get 1e-11/1e-9 even
/// though the cDFT lane itself runs at the defaults.
fn hene_cfg_tight() -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..hene_cfg()
    }
}

/// Build the constraint weight matrix `W` for a fragment, on the SAME grid the
/// cDFT driver uses by default (99 radial × 302 angular). Duplicated here
/// rather than exported, so this test never perturbs the lane it audits.
fn hene_weight_matrix(sys: &Sys, fragment: &[usize]) -> Array2<f64> {
    let grid_cfg = AtomicGridConfig {
        n_radial: 99,
        n_angular: 302,
        ..Default::default()
    };
    let grid = build_atomic_grid(&sys.mol, &grid_cfg);
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(&sys.mol, &sys.bs, &pts).unwrap();
    build_weight_matrix(&sys.mol, &grid, &chi, fragment)
}

/// Occupied-block AO density for one spin from MO coefficients.
fn density_from_mos(c: &Array2<f64>, nocc: usize) -> Array2<f64> {
    let occ = c.slice(ndarray::s![.., ..nocc]);
    occ.dot(&occ.t())
}

/// Occ→virt block (rows = virt, cols = occ) of a square MO matrix.
fn ov_block(m: &Array2<f64>, nocc: usize, n: usize) -> Array2<f64> {
    let nv = n - nocc;
    let mut out = Array2::<f64>::zeros((nv, nocc));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            out[(ir, i)] = m[(a, i)];
        }
    }
    out
}

/// Cayley rotation of C by ε·κ_ov (exactly orthonormality-preserving).
fn rotate(c: &Array2<f64>, k_ov: &Array2<f64>, nocc: usize, eps: f64) -> Array2<f64> {
    let n = c.nrows();
    let mut kappa = Array2::<f64>::zeros((n, n));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            let v = eps * k_ov[(ir, i)];
            kappa[(a, i)] = v;
            kappa[(i, a)] = -v;
        }
    }
    let half = 0.5 * &kappa;
    let eye = Array2::<f64>::eye(n);
    let am = &eye - &half;
    let bm = &eye + &half;
    let mut u = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let sol = am.solve(&bm.column(col).to_owned()).unwrap();
        for row in 0..n {
            u[(row, col)] = sol[row];
        }
    }
    c.dot(&u)
}

/// Both spin Focks in AO basis at MO coefficients (c_a, c_b), OPTIONALLY with a
/// fixed one-electron constraint potential `λW` added to each spin (the cDFT
/// `SpinChannel::Total` convention: same potential to both spins).
///
/// This mirrors `solve_uhf_fockmod`'s Fock exactly for pure UHF: F^σ = h + J[D]
/// − K[D^σ] (+ λW).
fn ao_focks(
    sys: &Sys,
    h: &Array2<f64>,
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    lam_w: Option<&Array2<f64>>,
) -> (Array2<f64>, Array2<f64>) {
    let n = sys.prep.nbasis();
    let da = density_from_mos(c_a, sys.nocc_a);
    let db = density_from_mos(c_b, sys.nocc_b);
    let dt = &da + &db;
    let mut j = Array2::<f64>::zeros((n, n));
    let mut kdum = Array2::<f64>::zeros((n, n));
    build_jk(
        &sys.ctx,
        &sys.prep,
        &sys.bounds,
        1e-12,
        &dt,
        &mut j,
        &mut kdum,
    )
    .unwrap();
    let mut ka = Array2::<f64>::zeros((n, n));
    let mut kb = Array2::<f64>::zeros((n, n));
    let mut jd = Array2::<f64>::zeros((n, n));
    build_jk(
        &sys.ctx,
        &sys.prep,
        &sys.bounds,
        1e-12,
        &da,
        &mut jd,
        &mut ka,
    )
    .unwrap();
    jd.fill(0.0);
    build_jk(
        &sys.ctx,
        &sys.prep,
        &sys.bounds,
        1e-12,
        &db,
        &mut jd,
        &mut kb,
    )
    .unwrap();
    let mut fa = h + &j - &ka;
    let mut fb = h + &j - &kb;
    if let Some(lw) = lam_w {
        fa += lw;
        fb += lw;
    }
    (fa, fb)
}

/// The λ-AUGMENTED energy `E[ρ] + λ N_C[ρ]` at MO coefficients (c_a, c_b).
///
/// `E[ρ]` is the ordinary UHF electronic energy ½ Tr[(h + F_bare) D] summed over
/// spins, plus nuclear repulsion; `λ N_C[ρ]` is `λ Tr[W (D_α + D_β)]` for a
/// `SpinChannel::Total` constraint. Note the bare F (no λW) enters the ½-trace
/// energy — adding λW there would double-count the linear constraint term.
fn augmented_energy(
    sys: &Sys,
    h: &Array2<f64>,
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    lam: f64,
    w: &Array2<f64>,
) -> f64 {
    let da = density_from_mos(c_a, sys.nocc_a);
    let db = density_from_mos(c_b, sys.nocc_b);
    let (fa, fb) = ao_focks(sys, h, c_a, c_b, None);
    let e_elec = 0.5 * ((&(h + &fa) * &da).sum() + (&(h + &fb) * &db).sum());
    let n_c = population(w, &da, &db, &SpinChannel::Total);
    e_elec + sys.mol.nuclear_repulsion() + lam * n_c
}

/// Packed occ→virt gradient of the λ-AUGMENTED functional: the (virt, occ)
/// block of the λ-augmented MO-basis Fock, per spin.
fn augmented_gradient(
    sys: &Sys,
    h: &Array2<f64>,
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    lam_w: Option<&Array2<f64>>,
) -> (Array2<f64>, Array2<f64>) {
    let n = sys.prep.nbasis();
    let (fa, fb) = ao_focks(sys, h, c_a, c_b, lam_w);
    let fa_mo = c_a.t().dot(&fa).dot(c_a);
    let fb_mo = c_b.t().dot(&fb).dot(c_b);
    (
        ov_block(&fa_mo, sys.nocc_a, n),
        ov_block(&fb_mo, sys.nocc_b, n),
    )
}

// ===========================================================================
// Eigensolver: lowest eigenvalue of the orbital Hessian via Davidson
// ===========================================================================

/// Flatten the two spin blocks into one vector, and back.
fn pack(a: &Array2<f64>, b: &Array2<f64>) -> Vec<f64> {
    a.iter().chain(b.iter()).copied().collect()
}
fn unpack(v: &[f64], sh_a: (usize, usize), sh_b: (usize, usize)) -> (Array2<f64>, Array2<f64>) {
    let na = sh_a.0 * sh_a.1;
    (
        Array2::from_shape_vec(sh_a, v[..na].to_vec()).unwrap(),
        Array2::from_shape_vec(sh_b, v[na..].to_vec()).unwrap(),
    )
}

/// Result of an extremal-eigenvalue solve on the orbital Hessian.
///
/// A thin, lossless view of [`ferric_scf::stability::StabilityResult`] in the
/// packed-vector shape this file's rotation helpers want. The full verdict
/// (including `StabilityVerdict` and the noise floor) is retained in
/// `stability`, and the four-way `verdict()` — NOT the `is_stable` bool — is
/// what any branching here reads.
struct EigResult {
    /// Lowest eigenvalue λ_min (Hartree per unit rotation²).
    lambda_min: f64,
    /// ‖H v − λ v‖₂ for the returned pair. A verdict whose |λ_min| is not
    /// comfortably above this residual is NOT a verdict.
    residual: f64,
    /// Eigenvector, packed (α block then β block).
    vec: Vec<f64>,
    iters: usize,
    /// The library verdict this row came from, kept so callers can branch on
    /// the total four-way `StabilityVerdict` rather than re-deriving a sign
    /// test against a hand-rolled threshold.
    stability: StabilityResult,
}

/// Lowest eigenvalue of the (λ-augmented) UHF orbital Hessian.
///
/// **This delegates to the library solver, `ferric_scf::stability::
/// uhf_internal_stability`, and must keep doing so.** It used to be a private
/// ~110-line single-root Davidson written inside this file, and that solver
/// returned a CONVERGED WRONG ANSWER on water/STO-3G:
///
/// ```text
///   dense (ground truth)   λ_min = +3.6243948468e-1
///   private single-root    λ_min = +3.6883351924e-1  resid 6.63e-11, 2 iters
///   library block solver   λ_min = +3.6243948468e-1  resid 3.72e-15, 7 iters
/// ```
///
/// The orbital Hessian is block diagonal in the molecule's irreps. On water
/// (C2v) `H·e_4` has support only on `{4, 14}` — a closed 2×2 irrep block whose
/// own lowest root is +3.6883e-1. A single-root Davidson seeded from a unit
/// vector never leaves that block, so it converges to 6.63e-11 *inside* it and
/// reports the wrong λ_min with a beautiful residual. No residual check can
/// detect that, because the iteration genuinely converged.
///
/// `uhf_internal_stability` defends against exactly this with three safeguards
/// its own docs call load-bearing: block tracking (`n_block >= 2`), a
/// symmetry-breaking dense seed, and block-wide convergence. Do NOT reintroduce
/// a local eigensolver here; and every verdict in this file is additionally
/// cross-checked against a DENSE Hessian built from the same matvec (see
/// `dense_lowest_eigenvalue` / `assert_matches_dense`), which is the only thing
/// that catches this class of bug when it recurs somewhere new.
///
/// `inp.f_a_mo` / `inp.f_b_mo` decide WHICH Hessian this is: pass the
/// λ-augmented MO Fock and the constrained orbitals and you get the λ-augmented
/// Hessian (see `augmented_hessian_equals_plain_hessian_at_fixed_lambda`).
fn lowest_eigenvalue(
    ctx: &ParallelContext,
    inp: &UhfNewtonInputs,
    max_iter: usize,
    conv: f64,
) -> EigResult {
    let cfg = StabilityConfig {
        conv_thresh: conv,
        max_iter,
        ..Default::default()
    };
    let st = uhf_internal_stability(ctx, inp, &cfg).expect("UHF stability analysis");
    let vec = pack(
        &st.eigenvector_alpha,
        st.eigenvector_beta
            .as_ref()
            .expect("UHF stability returns both spin blocks"),
    );
    EigResult {
        lambda_min: st.lowest_eigenvalue,
        residual: st.residual,
        vec,
        iters: st.iterations,
        stability: st,
    }
}

// ===========================================================================
// DENSE CROSS-CHECK — the durable guard
// ===========================================================================

/// The exact lowest eigenvalue of the orbital Hessian, by building it DENSELY:
/// apply the SAME `hessian_matvec` to every unit vector of the rotation space
/// and diagonalize the resulting matrix.
///
/// This is `dim` matvecs, and every system in this file has `dim <= 148`, so it
/// costs essentially nothing. It exists because it is the ONLY check that
/// catches a converged-but-wrong iterative eigenvalue: a symmetry-block-trapped
/// Davidson root has a tiny residual and cannot be distinguished from the truth
/// by any self-consistency test on the iteration itself.
///
/// Returns `(lambda_min, all_eigenvalues, matvec_asymmetry)`. The asymmetry is
/// reported because the dense build is also a free check that `hessian_matvec`
/// really is symmetric — if it were not, neither solver would be measuring an
/// eigenvalue problem at all.
fn dense_lowest_eigenvalue(ctx: &ParallelContext, inp: &UhfNewtonInputs) -> (f64, Vec<f64>, f64) {
    let n = inp.c_a.nrows();
    let (na, nb) = (inp.nocc_a, inp.nocc_b);
    let sh_a = (n - na, na);
    let sh_b = (n - nb, nb);
    let dim_a = sh_a.0 * sh_a.1;
    let dim = dim_a + sh_b.0 * sh_b.1;
    let pool = EnginePool::new(inp.bounds.op, inp.prep, 1e-14).unwrap();

    let mut dense = Array2::<f64>::zeros((dim, dim));
    for col in 0..dim {
        let mut v = vec![0.0; dim];
        v[col] = 1.0;
        let (ka, kb) = unpack(&v, sh_a, sh_b);
        let (ha, hb) = hessian_matvec(ctx, inp, &ka, &kb, &pool).unwrap();
        for (row, val) in ha.iter().chain(hb.iter()).enumerate() {
            dense[(row, col)] = *val;
        }
    }
    let asym = (&dense - &dense.t())
        .iter()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    let (vals, _) = dense.eigh(UPLO::Lower).unwrap();
    (vals[0], vals.to_vec(), asym)
}

/// Assert an iterative λ_min against the DENSE ground truth, and report both.
///
/// The tolerance is deliberately NOT a loose "is it positive" bar: the defect
/// this guard exists for produced a wrong λ_min that was positive, clearly
/// above any sign threshold, and only 1.8% off. `tol` must therefore be tight
/// enough that the WRONG value fails — which is checked directly by
/// `dense_cross_check_rejects_the_symmetry_block_root`.
fn assert_matches_dense(
    label: &str,
    ctx: &ParallelContext,
    inp: &UhfNewtonInputs,
    e: &EigResult,
    tol: f64,
) {
    let (dense_min, spectrum, asym) = dense_lowest_eigenvalue(ctx, inp);
    let lowest: Vec<String> = spectrum
        .iter()
        .take(4)
        .map(|v| format!("{v:+.8e}"))
        .collect();
    eprintln!(
        "[dense x-check] {label}: dense λ_min = {dense_min:+.10e}  iterative = {:+.10e}  \
         |Δ| = {:.3e}  (dim {}, matvec asymmetry {asym:.2e}, lowest 4: {})",
        e.lambda_min,
        (e.lambda_min - dense_min).abs(),
        spectrum.len(),
        lowest.join(", ")
    );
    assert!(
        asym < 1e-9,
        "{label}: hessian_matvec is not symmetric (max |H - Hᵀ| = {asym:.3e}); \
         an eigenvalue from it would not mean anything"
    );
    assert!(
        (e.lambda_min - dense_min).abs() <= tol,
        "{label}: iterative λ_min = {:.10e} does NOT match the dense Hessian's \
         {dense_min:.10e} (|Δ| = {:.3e} > tol {tol:.1e}). This is the symmetry-block \
         trap: an iterative root can converge to a tiny residual ({:.2e} here) inside \
         the wrong irrep block. Dense spectrum starts: {}",
        e.lambda_min,
        (e.lambda_min - dense_min).abs(),
        e.residual,
        lowest.join(", ")
    );
}

/// Convenience: build `UhfNewtonInputs` from MOs + MO-basis Focks.
#[allow(clippy::too_many_arguments)]
fn newton_inputs<'a>(
    sys: &'a Sys,
    c_a: &'a Array2<f64>,
    c_b: &'a Array2<f64>,
    f_a_mo: &'a Array2<f64>,
    f_b_mo: &'a Array2<f64>,
) -> UhfNewtonInputs<'a> {
    UhfNewtonInputs {
        prep: &sys.prep,
        bounds: &sys.bounds,
        c_a,
        c_b,
        f_a_mo,
        f_b_mo,
        nocc_a: sys.nocc_a,
        nocc_b: sys.nocc_b,
        k_mix_sr: 1.0,
        fxc: None,
        thresh: 1e-12,
        ooc_budget: 0,
    }
}

// ===========================================================================
// 1. THE PREREQUISITE: is the λ-augmented Hessian the plain Hessian?
// ===========================================================================

/// **Prerequisite of every stability verdict in this file.**
///
/// The claim under test: because the cDFT constraint enters as a FIXED
/// one-electron operator `λW` (built once from geometry + grid, never rebuilt
/// from the density), it contributes to the Fock but NOT to the Fock response.
/// Therefore the λ-augmented orbital Hessian equals the ordinary UHF orbital
/// Hessian at the constrained density/orbitals with constrained orbital
/// energies — i.e. `hessian_matvec` fed the constrained state already IS the
/// augmented Hessian.
///
/// Verified by central finite difference of the λ-augmented GRADIENT (the
/// occ→virt block of the λ-augmented MO Fock) along a random rotation, at FIXED
/// λ, compared to the analytic matvec. If `W` were density-dependent anywhere,
/// or the Becke weights carried a density dependence, this would disagree.
///
/// Two things are checked, and they are different:
///   (a) FD of the augmented gradient == analytic matvec  → the matvec IS the
///       augmented Hessian;
///   (b) the augmented and bare matvecs DIFFER (the λW term is really present
///       in the gradient/Fock), so (a) is not passing vacuously.
///
/// Also anchors the GRADIENT itself against FD of the augmented ENERGY, which
/// is what makes "this Hessian belongs to that functional" a closed statement
/// rather than two unconnected checks.
#[test]
fn augmented_hessian_equals_plain_hessian_at_fixed_lambda() {
    let sys = build_sys(&hene_xyz(), 1, 2, "def2-svp");
    let n = sys.prep.nbasis();
    let h = oneelectron::hcore(&sys.prep);

    // A representative constrained state: the He-fragment constraint at the
    // λ of state A. We do NOT need it converged for this check — the Hessian
    // identity is a property of the OPERATOR, not of stationarity — but using
    // the real converged state keeps the numbers in the regime that matters.
    let w = hene_weight_matrix(&sys, &[0]);
    let lam = 1.631409_f64;
    let lam_w = lam * &w;

    let cfg = hene_cfg_tight();
    let fm = |f_a: &mut Array2<f64>, f_b: &mut Array2<f64>| {
        *f_a += &lam_w;
        *f_b += &lam_w;
    };
    let res = solve_uhf_fockmod(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        &cfg,
        None,
        Some(&fm),
    )
    .unwrap();
    assert!(
        res.converged,
        "fixed-λ inner UHF must converge for the FD check"
    );
    let c_a = res.mos_alpha.clone();
    let c_b = res.mos_beta.clone().unwrap();

    // ---- (0) gradient-of-the-functional anchor -----------------------------
    // FD of the λ-augmented ENERGY along a random rotation must equal the
    // directional derivative read off the λ-augmented GRADIENT. Factor 2:
    // the occ→virt block is counted once but the antisymmetric κ touches both
    // the (a,i) and (i,a) corners.
    //
    // CRUCIAL: this anchor is evaluated at a DISPLACED point, not at the
    // converged one. At a converged constrained state the augmented gradient is
    // zero by Brillouin, so BOTH sides of this comparison would be ~1e-13 and
    // the assert would pass on nothing at all — measuring noise against noise.
    // (That is exactly what the first run of this test did: FD = 1.42e-10,
    // analytic = -3.60e-14, and it "passed".) Displacing by `OFFSET` puts the
    // reference point off the stationary point so the gradient is O(1e-2) and
    // the comparison has something to compare.
    let mut rng = Xorshift64(0x243F_6A88_85A3_08D3);
    let mk = |rng: &mut Xorshift64, nocc: usize| -> Array2<f64> {
        Array2::<f64>::from_shape_fn((n - nocc, nocc), |_| 0.01 * rng.next_f64())
    };
    const OFFSET: f64 = 0.05;
    let off_a = mk(&mut rng, sys.nocc_a);
    let off_b = mk(&mut rng, sys.nocc_b);
    let d_a = rotate(&c_a, &off_a, sys.nocc_a, OFFSET);
    let d_b = rotate(&c_b, &off_b, sys.nocc_b, OFFSET);
    let ka = mk(&mut rng, sys.nocc_a);
    let kb = mk(&mut rng, sys.nocc_b);

    let eps_e = 1e-4;
    let e_p = augmented_energy(
        &sys,
        &h,
        &rotate(&d_a, &ka, sys.nocc_a, eps_e),
        &rotate(&d_b, &kb, sys.nocc_b, eps_e),
        lam,
        &w,
    );
    let e_m = augmented_energy(
        &sys,
        &h,
        &rotate(&d_a, &ka, sys.nocc_a, -eps_e),
        &rotate(&d_b, &kb, sys.nocc_b, -eps_e),
        lam,
        &w,
    );
    let de_fd = (e_p - e_m) / (2.0 * eps_e);
    let (g_a, g_b) = augmented_gradient(&sys, &h, &d_a, &d_b, Some(&lam_w));
    let de_analytic = 2.0
        * (g_a.iter().zip(ka.iter()).map(|(x, y)| x * y).sum::<f64>()
            + g_b.iter().zip(kb.iter()).map(|(x, y)| x * y).sum::<f64>());
    eprintln!(
        "[aug-grad anchor] at displaced point: dE/deps  FD = {de_fd:.10e}   \
         analytic = {de_analytic:.10e}   rel = {:.3e}",
        (de_fd - de_analytic).abs() / de_analytic.abs().max(1e-12)
    );
    // Reachability guard: if the reference point were stationary this whole
    // check would be vacuous, so REQUIRE a gradient big enough to test.
    assert!(
        de_analytic.abs() > 1e-4,
        "the anchor point must be NON-stationary or this check is vacuous: \
         |dE/deps| = {:.3e}",
        de_analytic.abs()
    );
    assert!(
        (de_fd - de_analytic).abs() <= 1e-6 * de_analytic.abs(),
        "λ-augmented gradient must be the derivative of the λ-augmented energy: \
         FD {de_fd:.6e} vs analytic {de_analytic:.6e}"
    );

    // ---- (1) FD of the augmented gradient vs the analytic matvec -----------
    let f_a_mo = {
        let (fa, _) = ao_focks(&sys, &h, &c_a, &c_b, Some(&lam_w));
        c_a.t().dot(&fa).dot(&c_a)
    };
    let f_b_mo = {
        let (_, fb) = ao_focks(&sys, &h, &c_a, &c_b, Some(&lam_w));
        c_b.t().dot(&fb).dot(&c_b)
    };
    let inp = newton_inputs(&sys, &c_a, &c_b, &f_a_mo, &f_b_mo);
    let pool = EnginePool::new(sys.bounds.op, &sys.prep, 1e-14).unwrap();
    let (ha, hb) = hessian_matvec(&sys.ctx, &inp, &ka, &kb, &pool).unwrap();

    let eps = 1e-4;
    let (gp_a, gp_b) = augmented_gradient(
        &sys,
        &h,
        &rotate(&c_a, &ka, sys.nocc_a, eps),
        &rotate(&c_b, &kb, sys.nocc_b, eps),
        Some(&lam_w),
    );
    let (gm_a, gm_b) = augmented_gradient(
        &sys,
        &h,
        &rotate(&c_a, &ka, sys.nocc_a, -eps),
        &rotate(&c_b, &kb, sys.nocc_b, -eps),
        Some(&lam_w),
    );
    let fd_a = (&gp_a - &gm_a) / (2.0 * eps);
    let fd_b = (&gp_b - &gm_b) / (2.0 * eps);

    let max_dev = (&fd_a - &ha)
        .iter()
        .chain((&fd_b - &hb).iter())
        .fold(0.0f64, |m, &v| m.max(v.abs()));
    let scale = fd_a
        .iter()
        .chain(fd_b.iter())
        .fold(0.0f64, |m, &v| m.max(v.abs()));
    eprintln!(
        "[aug-Hessian FD] max|FD − analytic| = {max_dev:.3e}   scale = {scale:.3e}   \
         rel = {:.3e}",
        max_dev / scale
    );
    assert!(
        max_dev <= 1e-5 * scale.max(1e-3),
        "λ-augmented Hessian must equal hessian_matvec at the constrained state: \
         max deviation {max_dev:.3e} vs scale {scale:.3e}"
    );

    // ---- (2) the check is not vacuous: λW really does change the state -----
    // Same matvec at the SAME orbitals but with the BARE MO Fock must differ,
    // otherwise (1) would be comparing two copies of the same thing.
    let f_a_bare = {
        let (fa, _) = ao_focks(&sys, &h, &c_a, &c_b, None);
        c_a.t().dot(&fa).dot(&c_a)
    };
    let f_b_bare = {
        let (_, fb) = ao_focks(&sys, &h, &c_a, &c_b, None);
        c_b.t().dot(&fb).dot(&c_b)
    };
    let inp_bare = newton_inputs(&sys, &c_a, &c_b, &f_a_bare, &f_b_bare);
    let (ba, bb) = hessian_matvec(&sys.ctx, &inp_bare, &ka, &kb, &pool).unwrap();
    let diff = (&ba - &ha)
        .iter()
        .chain((&bb - &hb).iter())
        .fold(0.0f64, |m, &v| m.max(v.abs()));
    eprintln!("[non-vacuous] max|H_bare·κ − H_aug·κ| = {diff:.3e}  (must be >> 0)");
    assert!(
        diff > 1e-3 * scale.max(1e-3),
        "bare and λ-augmented matvecs must DIFFER, else the FD check above is vacuous: \
         diff {diff:.3e}"
    );
}

// ===========================================================================
// 2. EXACTNESS ANCHOR: λ = 0 must reduce to the unconstrained problem
// ===========================================================================

/// **Exactness anchor.** At λ = 0 the constraint is vacuous, so the constrained
/// path must reduce EXACTLY to the unconstrained one.
///
/// Checked at the level that matters: the λ-augmented Hessian built with λ = 0
/// must be BIT-IDENTICAL to the plain Hessian, because `λW` with λ = 0 is
/// exactly the zero matrix and `f += 0.0 * W` changes no bit of a float. This
/// is the strongest possible form of the anchor, and it is what licenses
/// reading the λ ≠ 0 numbers as "the same operator, with the constraint on".
#[test]
fn lambda_zero_augmented_hessian_is_bit_identical_to_unconstrained() {
    let sys = build_sys(&hene_xyz(), 1, 2, "def2-svp");
    let h = oneelectron::hcore(&sys.prep);
    let cfg = hene_cfg_tight();

    // Unconstrained UHF.
    let res = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg).unwrap();
    assert!(res.converged);
    let c_a = res.mos_alpha.clone();
    let c_b = res.mos_beta.clone().unwrap();

    let w = hene_weight_matrix(&sys, &[0]);
    let zero_w = 0.0 * &w;

    let (fa_b, fb_b) = ao_focks(&sys, &h, &c_a, &c_b, None);
    let (fa_0, fb_0) = ao_focks(&sys, &h, &c_a, &c_b, Some(&zero_w));
    // The AO Focks must already be bit-identical.
    let fock_dev = (&fa_b - &fa_0)
        .iter()
        .chain((&fb_b - &fb_0).iter())
        .fold(0.0f64, |m, &v| m.max(v.abs()));
    eprintln!("[λ=0 anchor] max|F_bare − F_(λ=0)| = {fock_dev:.3e}  (must be exactly 0)");
    assert_eq!(fock_dev, 0.0, "λ = 0 must leave the Fock bit-identical");

    let f_a_mo = c_a.t().dot(&fa_b).dot(&c_a);
    let f_b_mo = c_b.t().dot(&fb_b).dot(&c_b);
    let f_a_mo0 = c_a.t().dot(&fa_0).dot(&c_a);
    let f_b_mo0 = c_b.t().dot(&fb_0).dot(&c_b);

    let inp = newton_inputs(&sys, &c_a, &c_b, &f_a_mo, &f_b_mo);
    let inp0 = newton_inputs(&sys, &c_a, &c_b, &f_a_mo0, &f_b_mo0);
    let e_bare = lowest_eigenvalue(&sys.ctx, &inp, 60, 1e-7);
    let e_zero = lowest_eigenvalue(&sys.ctx, &inp0, 60, 1e-7);
    // Both sides against the dense Hessian: bit-identity between two solves is
    // NOT evidence that either is right — a construction error is deterministic
    // and would reproduce on both. Only the dense build is an independent
    // construction.
    assert_matches_dense("HeNe+ λ=0 bare", &sys.ctx, &inp, &e_bare, 1e-8);
    assert_matches_dense("HeNe+ λ=0 constrained-path", &sys.ctx, &inp0, &e_zero, 1e-8);
    eprintln!(
        "[λ=0 anchor] λ_min unconstrained = {:.10e} (resid {:.2e}), \
         λ_min via constrained path at λ=0 = {:.10e} (resid {:.2e})",
        e_bare.lambda_min, e_bare.residual, e_zero.lambda_min, e_zero.residual
    );
    assert_eq!(
        e_bare.lambda_min, e_zero.lambda_min,
        "the λ = 0 constrained path must reproduce the unconstrained λ_min bit-identically"
    );
}

/// **Exactness anchor, external half.** At λ = 0 on HeNe⁺ the analysis must
/// reproduce the KNOWN answer: the unconstrained UHF state ferric converges to
/// is INTERNALLY UNSTABLE.
///
/// This is the only externally-checkable statement available — PySCF has no
/// cDFT, so there is no external reference for the constrained case. The
/// `test/cdft-state-identity` audit (d1d30fa5) established that ferric's
/// unconstrained HeNe⁺ UHF matches PySCF's MOM-forced ²Π state to 1.26e-9 Ha,
/// and that PySCF's `stability()` rejects that state as 0.136 eV above the true
/// ²Σ⁺ minimum. So λ_min MUST come out negative here.
///
/// Passing this validates the SIGN CONVENTION as well as the reduction: a
/// Hessian with a flipped sign would report this known-unstable state as
/// stable.
#[test]
fn unconstrained_hene_is_unstable_known_reference() {
    let sys = build_sys(&hene_xyz(), 1, 2, "def2-svp");
    let h = oneelectron::hcore(&sys.prep);
    let cfg = hene_cfg_tight();
    let res = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg).unwrap();
    assert!(res.converged, "the reference state must be a CONVERGED one");
    let c_a = res.mos_alpha.clone();
    let c_b = res.mos_beta.clone().unwrap();
    let (fa, fb) = ao_focks(&sys, &h, &c_a, &c_b, None);
    let f_a_mo = c_a.t().dot(&fa).dot(&c_a);
    let f_b_mo = c_b.t().dot(&fb).dot(&c_b);
    let inp = newton_inputs(&sys, &c_a, &c_b, &f_a_mo, &f_b_mo);
    let e = lowest_eigenvalue(&sys.ctx, &inp, 80, 1e-8);
    assert_matches_dense("HeNe+ unconstrained UHF", &sys.ctx, &inp, &e, 1e-8);
    eprintln!(
        "[KNOWN-UNSTABLE] HeNe+ unconstrained UHF: E = {:.10}, λ_min = {:.8e} Ha, \
         residual = {:.2e}, iters = {}",
        res.energy, e.lambda_min, e.residual, e.iters
    );
    assert!(
        e.residual < 1e-6,
        "eigensolver must actually converge before its verdict is read: residual {:.3e}",
        e.residual
    );
    assert!(
        e.lambda_min < -1e-4,
        "the unconstrained HeNe+ UHF state is KNOWN unstable (PySCF stability() puts it \
         0.136 eV above the σ minimum): λ_min must be clearly negative, got {:.6e}",
        e.lambda_min
    );
}

/// **Both directions, positive half.** The analysis must report STABLE on
/// something that is. Water/STO-3G at its equilibrium closed-shell UHF solution
/// is the standard known-stable reference (a closed-shell singlet well away
/// from any RHF→UHF instability).
#[test]
fn known_stable_reference_reports_positive_lambda_min() {
    // `lam_ref` is the DENSE-Hessian λ_min, recorded to 10 digits. Asserting
    // against it — rather than against a "clearly positive" threshold — is what
    // makes this test able to fail. The old bar was `λ_min > 1e-3`, which the
    // WRONG water value (+3.6883351924e-1, the C2v symmetry-block root) cleared
    // by 360×; see `dense_cross_check_rejects_the_symmetry_block_root`.
    for (label, xyz, charge, mult, bset, lam_ref) in [
        (
            "H2O/STO-3G",
            "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587\n",
            0,
            1,
            "sto-3g",
            3.6243948468e-1,
        ),
        (
            "H2/STO-3G @ r_e",
            "2\nH2\nH 0 0 0\nH 0 0 0.741\n",
            0,
            1,
            "sto-3g",
            4.0344721450e-1,
        ),
    ] {
        let sys = build_sys(xyz, charge, mult, bset);
        let h = oneelectron::hcore(&sys.prep);
        let cfg = RhfConfig {
            energy_conv: 1e-11,
            density_conv: 1e-9,
            max_iter: 200,
            ..Default::default()
        };
        let res = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg).unwrap();
        assert!(res.converged);
        let c_a = res.mos_alpha.clone();
        let c_b = res.mos_beta.clone().unwrap();
        let (fa, fb) = ao_focks(&sys, &h, &c_a, &c_b, None);
        let f_a_mo = c_a.t().dot(&fa).dot(&c_a);
        let f_b_mo = c_b.t().dot(&fb).dot(&c_b);
        let inp = newton_inputs(&sys, &c_a, &c_b, &f_a_mo, &f_b_mo);
        let e = lowest_eigenvalue(&sys.ctx, &inp, 80, 1e-8);
        assert_matches_dense(label, &sys.ctx, &inp, &e, 1e-8);
        eprintln!(
            "[KNOWN-STABLE] {label}: E = {:.10}, λ_min = {:.8e} Ha, residual = {:.2e}, \
             iters = {}",
            res.energy, e.lambda_min, e.residual, e.iters
        );
        assert!(
            e.residual < 1e-6,
            "{label}: eigensolver residual {:.3e}",
            e.residual
        );
        assert_eq!(
            e.stability.verdict(),
            StabilityVerdict::Stable,
            "{label} is a known-stable minimum, so the verdict must be STABLE: {}",
            e.stability.summary()
        );
        assert!(
            (e.lambda_min - lam_ref).abs() < 1e-8,
            "{label}: lambda_min = {:.10e} but the dense Hessian's value is \
             {lam_ref:.10e} (|delta| = {:.3e}). A 'clearly positive' bar would NOT catch \
             this: the symmetry-block root on water is +3.6883351924e-1, which is \
             positive, 1.8% off, and 360x above the old 1e-3 threshold.",
            e.lambda_min,
            (e.lambda_min - lam_ref).abs()
        );
    }
}

/// **REACHABILITY OF THE DENSE CROSS-CHECK.** The guard that catches the F1
/// defect must be shown to FAIL on the wrong value, not merely to pass on the
/// right one — a test nobody has seen fail is an assumption.
///
/// This reproduces the superseded private single-root Davidson from `8e5699fb`
/// (unit-vector seed on the smallest Hessian diagonal, one tracked root, break
/// on that root's own residual) on the same water/STO-3G Hessian the anchor
/// above uses, and asserts THREE things:
///
///   1. the private solver really does return the WRONG λ_min, and returns it
///      with a tiny residual and `converged`-looking behaviour, so no residual
///      test could have caught it;
///   2. the OLD assertion bar (`λ_min > 1e-3`) accepts that wrong value — which
///      is why the defect survived review;
///   3. `assert_matches_dense`, the new guard, REJECTS it.
///
/// If a future change makes the private-style solver accidentally correct on
/// this system, (1) fails and this test must be re-pointed at a system where
/// the trap still fires — the test is not allowed to go quietly vacuous.
#[test]
fn dense_cross_check_rejects_the_symmetry_block_root() {
    let sys = build_sys(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587\n",
        0,
        1,
        "sto-3g",
    );
    let h = oneelectron::hcore(&sys.prep);
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    };
    let res = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg).unwrap();
    assert!(res.converged);
    let c_a = res.mos_alpha.clone();
    let c_b = res.mos_beta.clone().unwrap();
    let (fa, fb) = ao_focks(&sys, &h, &c_a, &c_b, None);
    let f_a_mo = c_a.t().dot(&fa).dot(&c_a);
    let f_b_mo = c_b.t().dot(&fb).dot(&c_b);
    let inp = newton_inputs(&sys, &c_a, &c_b, &f_a_mo, &f_b_mo);

    let (dense_min, spectrum, _asym) = dense_lowest_eigenvalue(&sys.ctx, &inp);
    let (bad_lambda, bad_resid, bad_iters) = single_root_davidson_the_old_way(&sys.ctx, &inp);
    eprintln!(
        "[reachability] dense λ_min = {dense_min:+.10e}  |  8e5699fb-style single-root \
         λ_min = {bad_lambda:+.10e} (resid {bad_resid:.2e}, {bad_iters} iters)  |  \
         error {:+.3e} = {:.3}%  |  dense spectrum[1] = {:+.10e}",
        bad_lambda - dense_min,
        100.0 * (bad_lambda - dense_min).abs() / dense_min.abs(),
        spectrum[1]
    );

    // (1) The trap still fires, and it fires CONVERGED.
    assert!(
        (bad_lambda - dense_min).abs() > 1e-3,
        "the symmetry-block trap no longer fires on water/STO-3G (single-root gave \
         {bad_lambda:.10e} vs dense {dense_min:.10e}). This test has gone VACUOUS and \
         must be re-pointed at a system where it still fires — it is not allowed to \
         pass by accident."
    );
    assert!(
        bad_resid < 1e-8,
        "the wrong root must arrive with a SMALL residual, else a residual check \
         would already have caught it: got {bad_resid:.3e}"
    );
    assert!(
        (bad_lambda - spectrum[1]).abs() < 1e-9,
        "the wrong value must be the SECOND dense eigenvalue (the C2v block root), \
         which is what makes this a symmetry-block trap and not generic \
         non-convergence: got {bad_lambda:.10e} vs spectrum[1] = {:.10e}",
        spectrum[1]
    );

    // (2) The OLD bar accepts it. This is the review finding, asserted.
    assert!(
        bad_lambda > 1e-3,
        "the old `λ_min > 1e-3` bar is supposed to ACCEPT the wrong value \
         ({bad_lambda:.6e}); if it no longer does, the story recorded here is stale"
    );
    assert!(
        dense_min > 1e-3,
        "the old bar accepts the RIGHT value too ({dense_min:.6e}) — which is exactly \
         why it could not distinguish them"
    );

    // (3) The NEW guard rejects it. Run `assert_matches_dense` on a fabricated
    // EigResult carrying the wrong eigenvalue and confirm it panics.
    let bad = EigResult {
        lambda_min: bad_lambda,
        residual: bad_resid,
        vec: vec![0.0; spectrum.len()],
        iters: bad_iters,
        stability: uhf_internal_stability(&sys.ctx, &inp, &StabilityConfig::default()).unwrap(),
    };
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_matches_dense("INJECTED WRONG VALUE", &sys.ctx, &inp, &bad, 1e-8);
    }));
    assert!(
        caught.is_err(),
        "assert_matches_dense ACCEPTED the symmetry-block root {bad_lambda:.10e} against \
         the dense λ_min {dense_min:.10e}. The guard is inert and the F1 defect would \
         recur undetected."
    );
    eprintln!(
        "[reachability] assert_matches_dense correctly REJECTED the injected wrong \
         value {bad_lambda:+.10e}"
    );
}

/// The superseded eigensolver from `8e5699fb`, reproduced verbatim in shape:
/// one unit-vector seed on the smallest Hessian diagonal, ONE tracked Ritz root
/// (`theta = vals[0]`), and a break on that single root's residual.
///
/// It exists ONLY as the wrong-value generator for
/// `dense_cross_check_rejects_the_symmetry_block_root`. It is deliberately not
/// used for any verdict: `lowest_eigenvalue` delegates to the library block
/// solver. Returns `(lambda_min, residual, iterations)`.
fn single_root_davidson_the_old_way(
    ctx: &ParallelContext,
    inp: &UhfNewtonInputs,
) -> (f64, f64, usize) {
    let n = inp.c_a.nrows();
    let (na, nb) = (inp.nocc_a, inp.nocc_b);
    let sh_a = (n - na, na);
    let sh_b = (n - nb, nb);
    let dim = sh_a.0 * sh_a.1 + sh_b.0 * sh_b.1;
    let pool = EnginePool::new(inp.bounds.op, inp.prep, 1e-14).unwrap();

    let mut diag = Vec::with_capacity(dim);
    for (f_mo, nocc) in [(inp.f_a_mo, na), (inp.f_b_mo, nb)] {
        for a in nocc..n {
            for i in 0..nocc {
                diag.push(f_mo[(a, a)] - f_mo[(i, i)]);
            }
        }
    }
    let matvec = |v: &[f64]| -> Vec<f64> {
        let (ka, kb) = unpack(v, sh_a, sh_b);
        let (ha, hb) = hessian_matvec(ctx, inp, &ka, &kb, &pool).unwrap();
        pack(&ha, &hb)
    };

    let seed = diag
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let mut basis: Vec<Vec<f64>> = {
        let mut v = vec![0.0; dim];
        v[seed] = 1.0;
        vec![v]
    };
    let mut sigma: Vec<Vec<f64>> = vec![matvec(&basis[0])];
    let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let nrm = |a: &[f64]| -> f64 { dot(a, a).sqrt() };

    let mut theta = dot(&basis[0], &sigma[0]);
    let mut resid_norm = f64::INFINITY;
    let mut used = 0usize;

    for it in 1..=80 {
        used = it;
        let m = basis.len();
        let mut sub = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                sub[(i, j)] = dot(&basis[i], &sigma[j]);
            }
        }
        let sub_sym = 0.5 * (&sub + &sub.t());
        let (vals, vecs) = sub_sym.eigh(UPLO::Lower).unwrap();
        theta = vals[0];
        let y = vecs.index_axis(ndarray::Axis(1), 0).to_owned();

        let mut x = vec![0.0; dim];
        let mut hx = vec![0.0; dim];
        for i in 0..m {
            let yi = y[i];
            for k in 0..dim {
                x[k] += yi * basis[i][k];
                hx[k] += yi * sigma[i][k];
            }
        }
        let r: Vec<f64> = (0..dim).map(|k| hx[k] - theta * x[k]).collect();
        resid_norm = nrm(&r);
        if resid_norm < 1e-8 {
            break;
        }
        let mut t: Vec<f64> = (0..dim)
            .map(|k| {
                let d = theta - diag[k];
                r[k] / if d.abs() < 1e-8 {
                    1e-8_f64.copysign(d)
                } else {
                    d
                }
            })
            .collect();
        for _ in 0..2 {
            for bvec in basis.iter() {
                let c = dot(&t, bvec);
                for k in 0..dim {
                    t[k] -= c * bvec[k];
                }
            }
        }
        let tn = nrm(&t);
        if tn < 1e-12 {
            break;
        }
        for v in t.iter_mut() {
            *v /= tn;
        }
        sigma.push(matvec(&t));
        basis.push(t);
    }
    (theta, resid_norm, used)
}

// ===========================================================================
// 3. THE VERDICT: are the diabats minima of their constrained manifolds?
// ===========================================================================

/// Run the real cDFT driver for one fragment/target, then measure λ_min of the
/// λ-AUGMENTED Hessian at the converged constrained state.
///
/// Returns (E_bare, λ, N_C, λ_min, residual, outer_iters).
fn constrained_state_stability(
    sys: &Sys,
    fragment: &[usize],
    target: f64,
    label: &str,
) -> (f64, f64, f64, f64, f64, usize) {
    let h = oneelectron::hcore(&sys.prep);
    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: fragment.to_vec(),
            target,
            spin: SpinChannel::Total,
        }],
        ..hene_cfg()
    };
    let cd = solve_cdft_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bs, &sys.bounds, &cfg).unwrap();
    assert!(cd.scf.converged, "{label}: inner SCF must be converged");
    let lam = cd.lambdas[0];
    let w = hene_weight_matrix(sys, fragment);
    let lam_w = lam * &w;

    let c_a = cd.scf.mos_alpha.clone();
    let c_b = cd.scf.mos_beta.clone().unwrap();
    let (fa, fb) = ao_focks(sys, &h, &c_a, &c_b, Some(&lam_w));
    let f_a_mo = c_a.t().dot(&fa).dot(&c_a);
    let f_b_mo = c_b.t().dot(&fb).dot(&c_b);

    // Sanity: the constrained state must be STATIONARY on the augmented
    // functional (Brillouin), else λ_min describes a non-solution.
    let gmax = ov_block(&f_a_mo, sys.nocc_a, sys.prep.nbasis())
        .iter()
        .chain(ov_block(&f_b_mo, sys.nocc_b, sys.prep.nbasis()).iter())
        .fold(0.0f64, |m, &v| m.max(v.abs()));

    let inp = newton_inputs(sys, &c_a, &c_b, &f_a_mo, &f_b_mo);
    let e = lowest_eigenvalue(&sys.ctx, &inp, 100, 1e-8);
    // Every constrained verdict in this file (A, B, and the natural-target row)
    // comes through here, so the dense cross-check lands on all of them at once.
    assert_matches_dense(label, &sys.ctx, &inp, &e, 1e-8);
    eprintln!(
        "[{label}] E_bare = {:.8}  λ = {:+.6}  N_C = {:.6}  outer = {}  \
         |g|_max = {gmax:.2e}  λ_min = {:+.8e} Ha  resid = {:.2e}  eig-iters = {}",
        cd.scf.energy, lam, cd.populations[0], cd.outer_iters, e.lambda_min, e.residual, e.iters
    );
    (
        cd.scf.energy,
        lam,
        cd.populations[0],
        e.lambda_min,
        e.residual,
        cd.outer_iters,
    )
}

/// **THE VERDICT.** λ_min of the λ-augmented Hessian for diabat A (hole on He,
/// N_He → 1) and diabat B (hole on Ne, N_He → 2).
///
/// H-PHYSICS predicts both positive (genuine constrained minima, and the
/// 1.0975 eV gap gets its first variational support). H-ARTIFACT predicts at
/// least one negative (a saddle, and the reported diabat energy — hence the
/// gap — is wrong).
///
/// MEASURED (2026-09-16): **H-ARTIFACT, for state B only.**
///   * STATE A: λ_min = +1.3194 Ha, residual 2.0e-9 → a GENUINE MINIMUM of its
///     constrained manifold.
///   * STATE B: λ_min = −3.9992e-2 Ha, residual 9.3e-9 → a SADDLE. The negative
///     eigenvalue is ~7 orders of magnitude above the eigensolver residual, so
///     this is not a marginal or noise-floor result.
///
/// Because B is not a minimum, the reported E_B = −130.40219057 is not the
/// lowest energy available at that constraint, and the 1.0975 eV A−B gap built
/// on it does not mean what the lane claims. How much lower B can actually go
/// is measured by `following_state_b_instability_reaches_a_lower_energy`.
///
/// This test ASSERTS the signs it measures, so it is a pinned result, not a
/// print: if either state's stability ever flips, this fails.
#[test]
fn hene_diabats_are_minima_of_their_constrained_manifolds() {
    let sys = build_sys(&hene_xyz(), 1, 2, "def2-svp");

    let (e_a, lam_a, n_a, lmin_a, res_a, _) =
        constrained_state_stability(&sys, &[0], 1.0, "STATE A (hole on He, N_He→1)");
    let (e_b, lam_b, n_b, lmin_b, res_b, _) =
        constrained_state_stability(&sys, &[0], 2.0, "STATE B (hole on Ne, N_He→2)");

    // The lane's own reported numbers, reproduced here so a drift in the
    // underlying states is caught rather than silently re-interpreted.
    eprintln!(
        "[GAP] E_A − E_B = {:.8} Ha = {:.4} eV",
        e_a - e_b,
        (e_a - e_b) * 27.211_386_245_988
    );
    assert!(
        (e_a - (-130.361_859_5)).abs() < 1e-5,
        "state A energy drifted: {e_a:.8}"
    );
    assert!(
        (e_b - (-130.402_190_57)).abs() < 1e-5,
        "state B energy drifted: {e_b:.8}"
    );
    assert!(
        (lam_a - 1.631_409).abs() < 1e-3,
        "state A λ drifted: {lam_a}"
    );
    assert!(
        (lam_b - (-2.753_705)).abs() < 1e-3,
        "state B λ drifted: {lam_b}"
    );
    assert!((n_a - 1.0).abs() < 1e-4 && (n_b - 2.0).abs() < 1e-4);

    // Verdicts, each gated on its own residual: a λ_min that is not comfortably
    // clear of the residual is not a verdict at all.
    assert!(res_a < 1e-6, "state A eigensolver residual {res_a:.3e}");
    assert!(res_b < 1e-6, "state B eigensolver residual {res_b:.3e}");
    assert!(
        lmin_a.abs() > 100.0 * res_a && lmin_b.abs() > 100.0 * res_b,
        "each λ_min must clear its own residual by a wide margin before its SIGN is read: \
         A {lmin_a:.3e} vs {res_a:.3e}, B {lmin_b:.3e} vs {res_b:.3e}"
    );

    // STATE A: a genuine minimum of its constrained manifold.
    assert!(
        lmin_a > 1e-4,
        "STATE A was measured STABLE (λ_min = +1.3194 Ha); a flip means the verdict \
         changed: λ_min = {lmin_a:.6e}"
    );
    // STATE B: a SADDLE. This asserts the NEGATIVE sign, i.e. it pins the
    // refutation. If B ever becomes stable this fails and the finding must be
    // revisited — which is the point.
    assert!(
        lmin_b < -1e-4,
        "STATE B was measured UNSTABLE (λ_min = −3.999e-2 Ha, a saddle of its own \
         constrained manifold); a flip means the verdict changed: λ_min = {lmin_b:.6e}"
    );
}

/// **Secondary, pre-registered experiment.** State B is constrained to the
/// INTEGER target N_He = 2.000, but the natural (promolecule) Becke population
/// at R = 2.0 Å is 1.954484 (`test/cdft-atomic-ip-anchor`, 794a1551) — so B is
/// dragged 0.046 e past its natural value.
///
/// H-ARTIFACT-OVERCONSTRAINT predicts stability differs between the integer and
/// natural targets (instability only at the over-constrained integer one), which
/// would localize any defect to the target choice rather than to cDFT itself.
/// H-PHYSICS predicts both stable.
///
/// Reported either way; asserted only on what is measured.
#[test]
fn state_b_natural_target_vs_integer_target() {
    let sys = build_sys(&hene_xyz(), 1, 2, "def2-svp");
    let (e_i, lam_i, n_i, lmin_i, res_i, _) =
        constrained_state_stability(&sys, &[0], 2.0, "B @ INTEGER target 2.000000");
    let (e_n, lam_n, n_n, lmin_n, res_n, _) =
        constrained_state_stability(&sys, &[0], HENE_NATURAL_N_HE, "B @ NATURAL target 1.954484");
    eprintln!(
        "[B target sweep] integer: N={n_i:.6} λ={lam_i:+.6} E={e_i:.8} λ_min={lmin_i:+.6e}\n\
         [B target sweep] natural: N={n_n:.6} λ={lam_n:+.6} E={e_n:.8} λ_min={lmin_n:+.6e}\n\
         [B target sweep] ΔE(natural − integer) = {:+.8} Ha, Δλ_min = {:+.6e}",
        e_n - e_i,
        lmin_n - lmin_i
    );
    assert!(
        res_i < 1e-6 && res_n < 1e-6,
        "residuals {res_i:.2e} / {res_n:.2e}"
    );
    // The natural target is LESS constrained, so its λ must be smaller in
    // magnitude — a structural check that the two runs really are different
    // points on the constraint path and not the same solve twice.
    assert!(
        lam_n.abs() < lam_i.abs(),
        "the natural target must need a weaker constraint than the integer one: \
         |λ_nat| = {:.4} vs |λ_int| = {:.4}",
        lam_n.abs(),
        lam_i.abs()
    );
    // MEASURED (2026-09-16): H-ARTIFACT-OVERCONSTRAINT is REFUTED as the sole
    // cause. B is unstable at BOTH targets —
    //   integer 2.000000: λ = −2.753705, E = −130.40219057, λ_min = −3.999e-2
    //   natural 1.954484: λ = −0.060075, E = −130.50031848, λ_min = −3.718e-3
    // — so the negative mode is NOT created by dragging the density 0.046 e past
    // its natural value. Over-constraint makes it ~10x WORSE (and costs 0.0981
    // Ha of energy), but it does not cause it.
    //
    // Read the natural-target row carefully before drawing comfort from its
    // smaller |λ_min|: λ = −0.060 is nearly a vacuous constraint and its energy
    // −130.50031848 sits 2.8e-5 Ha from the UNCONSTRAINED −130.50034664. At the
    // natural target the "diabat" is essentially the unconstrained state, which
    // is itself the known ²Π saddle — consistent with, not independent of, the
    // λ = 0 anchor above.
    assert!(
        lmin_i < -1e-4,
        "B @ integer target was measured UNSTABLE (λ_min = −3.999e-2): {lmin_i:.6e}"
    );
    assert!(
        lmin_n < -1e-4,
        "B @ natural target was ALSO measured UNSTABLE (λ_min = −3.718e-3), which is \
         what refutes over-constraint as the sole cause: {lmin_n:.6e}"
    );
    assert!(
        lmin_i < lmin_n,
        "over-constraint should make the instability WORSE, not better: \
         integer {lmin_i:.3e} vs natural {lmin_n:.3e}"
    );
}

// ===========================================================================
// 4. FOLLOWING THE INSTABILITY — what the saddle actually costs
// ===========================================================================

/// **The consequence.** State B is a SADDLE of its own constrained manifold
/// (λ_min = −3.999e-2 Ha). So there is a downhill direction AT THE SAME
/// CONSTRAINT, and the energy the lane reports for B is not the lowest one
/// available there.
///
/// This test follows that direction: rotate the converged B orbitals along the
/// negative eigenvector, re-converge the SAME constrained problem from the
/// rotated guess, and compare. Two things must hold for the new point to count
/// as a better B rather than a different state:
///   * it must satisfy the SAME constraint (N_He → 2.000 to the same tolerance);
///   * it must be a converged constrained solution in its own right.
///
/// The energy difference it finds is how wrong the reported diabat energy — and
/// therefore the 1.0975 eV gap — is.
#[test]
fn following_state_b_instability_reaches_a_lower_energy() {
    let sys = build_sys(&hene_xyz(), 1, 2, "def2-svp");
    let n = sys.prep.nbasis();
    let h = oneelectron::hcore(&sys.prep);
    let fragment = [0usize];
    let target = 2.0;

    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: fragment.to_vec(),
            target,
            spin: SpinChannel::Total,
        }],
        ..hene_cfg()
    };
    let cd = solve_cdft_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bs, &sys.bounds, &cfg).unwrap();
    assert!(cd.scf.converged);
    let lam = cd.lambdas[0];
    let w = hene_weight_matrix(&sys, &fragment);
    let lam_w = lam * &w;
    let c_a = cd.scf.mos_alpha.clone();
    let c_b = cd.scf.mos_beta.clone().unwrap();

    let (fa, fb) = ao_focks(&sys, &h, &c_a, &c_b, Some(&lam_w));
    let f_a_mo = c_a.t().dot(&fa).dot(&c_a);
    let f_b_mo = c_b.t().dot(&fb).dot(&c_b);
    let inp = newton_inputs(&sys, &c_a, &c_b, &f_a_mo, &f_b_mo);
    let eig = lowest_eigenvalue(&sys.ctx, &inp, 100, 1e-8);
    assert_matches_dense("state B (descent start)", &sys.ctx, &inp, &eig, 1e-8);
    eprintln!(
        "[follow] state B as reported: E = {:.8}  λ = {:+.6}  N_C = {:.6}  \
         λ_min = {:+.6e} (resid {:.2e})",
        cd.scf.energy, lam, cd.populations[0], eig.lambda_min, eig.residual
    );
    assert!(
        eig.lambda_min < -1e-4,
        "this test only has meaning if B is unstable; λ_min = {:.3e}",
        eig.lambda_min
    );

    // Unpack the negative eigenvector into the two spin rotation blocks.
    let sh_a = (n - sys.nocc_a, sys.nocc_a);
    let sh_b = (n - sys.nocc_b, sys.nocc_b);
    let (va, vb) = unpack(&eig.vec, sh_a, sh_b);

    // Descend along it. The step size is SWEPT rather than guessed: a saddle is
    // downhill in this direction for small enough steps, but the re-converged
    // basin can depend on how far the guess is thrown, so we report the sweep
    // and take the best CONSTRAINT-SATISFYING result.
    let mut best: Option<(f64, f64, f64, f64)> = None; // (step, E, λ, N_C)
    for &step in &[0.05_f64, 0.1, 0.2, 0.4, 0.8] {
        let g_a = rotate(&c_a, &va, sys.nocc_a, step);
        let g_b = rotate(&c_b, &vb, sys.nocc_b, step);
        // A restart that fails to re-converge is DATA, not a crash: record it
        // and keep sweeping, so the report shows which steps found a solution
        // and which did not, rather than dying on the first awkward one.
        let (e2, lam2, n2, conv) = match solve_cdft_uhf_from_guess(&sys, &cfg, (&g_a, &g_b)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[follow] step {step:>4}: restart did NOT re-converge ({e:?})");
                continue;
            }
        };
        eprintln!(
            "[follow] step {step:>4}: re-converged E = {e2:.8}  λ = {lam2:+.6}  \
             N_C = {n2:.6}  converged = {conv}  ΔE vs reported = {:+.8} Ha",
            e2 - cd.scf.energy
        );
        if conv && (n2 - target).abs() < 1e-4 {
            let better = match best {
                None => true,
                Some((_, be, _, _)) => e2 < be,
            };
            if better {
                best = Some((step, e2, lam2, n2));
            }
        }
    }

    let (step, e_lower, lam_lower, n_lower) =
        best.expect("at least one rotated restart must re-converge at the same constraint");
    let drop_ha = cd.scf.energy - e_lower;
    let drop_ev = drop_ha * 27.211_386_245_988;
    eprintln!(
        "[follow] BEST constraint-satisfying restart (step {step}): E = {e_lower:.8}  \
         λ = {lam_lower:+.6}  N_C = {n_lower:.6}\n\
         [follow] LOWER than the reported B by {drop_ha:.8} Ha = {drop_ev:.4} eV\n\
         [follow] reported gap E_A − E_B = 1.0975 eV; using the lower B it becomes \
         {:.4} eV",
        (-130.361_859_5 - e_lower) * 27.211_386_245_988
    );

    assert!(
        drop_ha > 1e-6,
        "following B's negative eigenvector must reach a LOWER constrained energy \
         (that is what λ_min < 0 means): ΔE = {drop_ha:.3e} Ha"
    );
    // MEASURED: 0.02451658 Ha = 0.6671 eV lower, at the SAME constraint
    // (N_He = 2.000000). Pin the magnitude so a future change that quietly
    // loses this downhill direction is caught.
    assert!(
        drop_ha > 1e-2,
        "the measured drop was 0.0245 Ha; a much smaller one means the descent \
         changed: ΔE = {drop_ha:.6e} Ha"
    );

    // Is the LOWER state itself a minimum, or is there yet more below? Report
    // it: "we found something lower" and "we found the bottom" are different
    // claims and only the first is established by the drop above.
    let cd2_lam = lam_lower;
    let lw2 = cd2_lam * &w;
    let guess2_a = rotate(&c_a, &va, sys.nocc_a, step);
    let guess2_b = rotate(&c_b, &vb, sys.nocc_b, step);
    let fm2 = |f_a: &mut Array2<f64>, f_b: &mut Array2<f64>| {
        *f_a += &lw2;
        *f_b += &lw2;
    };
    let scf2 = solve_uhf_fockmod(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        &cfg,
        Some((&guess2_a, &guess2_b)),
        Some(&fm2),
    )
    .unwrap();
    let c2a = scf2.mos_alpha.clone();
    let c2b = scf2.mos_beta.clone().unwrap();
    let (f2a, f2b) = ao_focks(&sys, &h, &c2a, &c2b, Some(&lw2));
    let f2a_mo = c2a.t().dot(&f2a).dot(&c2a);
    let f2b_mo = c2b.t().dot(&f2b).dot(&c2b);
    let inp2 = newton_inputs(&sys, &c2a, &c2b, &f2a_mo, &f2b_mo);
    let eig2 = lowest_eigenvalue(&sys.ctx, &inp2, 100, 1e-8);
    assert_matches_dense("descent endpoint", &sys.ctx, &inp2, &eig2, 1e-8);

    // THIS POINT IS WHERE THE ITERATIVE SOLVER RUNS OUT OF EVIDENCE, and the
    // library says so rather than guessing. The endpoint carries a near-exact
    // zero mode (dense λ_min = −1.0102213317e-9 Ha), so the Davidson block
    // stagnates in a proper subspace at a lowest-root residual of ~1.2e-8 —
    // just above the 1e-8 request — and reports INDETERMINATE.
    //
    // That is the correct report for an ITERATIVE solve, and it is why the
    // verdict below is read off the DENSE eigenvalue, which is exact by
    // construction (`dim` matvecs, no iteration to stall). Loosening
    // `conv_thresh` until the iterative solver said "converged" would be
    // manufacturing the verdict, not measuring it.
    //
    // NOTE (changed number, recorded deliberately): the superseded private
    // single-root solver in `8e5699fb` reported λ_min = −5.91e-9 here against a
    // residual of 4.83e-9. The dense Hessian says −1.0102213317e-9. Both are
    // inside the noise floor and both give the SAME verdict — MARGINAL — so the
    // conclusion ("we found something lower" is established, "we found the
    // bottom" is NOT) is unchanged. Only the unresolvable digits moved.
    let (dense2_min, _spec2, _asym2) = dense_lowest_eigenvalue(&sys.ctx, &inp2);
    let dense_verdict = StabilityResult {
        lowest_eigenvalue: dense2_min,
        converged: true,
        residual: 0.0,
        ..eig2.stability.clone()
    };
    eprintln!(
        "[follow] the LOWER state's own stability: E = {:.8}  iterative λ_min = \
         {:+.8e} Ha (resid {:.2e}, {})  |  DENSE λ_min = {:+.8e} Ha ⇒ {}",
        scf2.energy,
        eig2.lambda_min,
        eig2.residual,
        eig2.stability.verdict().label(),
        dense2_min,
        dense_verdict.verdict().label()
    );
    assert_eq!(
        dense_verdict.verdict(),
        StabilityVerdict::Marginal,
        "the lower state was measured MARGINAL (|λ_min| at or below the noise \
         floor, so no sign can be read; the descent has reached a near-flat \
         direction, and whether a still lower B exists is NOT established here). \
         A decisive verdict here would be a NEW result: dense λ_min = {dense2_min:.6e}, \
         noise floor = {:.1e}",
        dense_verdict.noise_floor
    );
}

/// Re-run the cDFT driver from an explicit orbital guess. The driver itself
/// takes no guess, so this reproduces its λ-Newton loop around
/// `solve_uhf_fockmod`, which DOES — everything else (grid, weight matrix,
/// residual, damping, tolerance) matches `cdft_driver::solve_cdft_uhf`.
///
/// Returns (E_bare, λ, N_C, converged).
fn solve_cdft_uhf_from_guess(
    sys: &Sys,
    cfg: &RhfConfig,
    guess: (&Array2<f64>, &Array2<f64>),
) -> Result<(f64, f64, f64, bool), ferric_core::FerricError> {
    let con = &cfg.constraints[0];
    let w = hene_weight_matrix(sys, &con.fragment);
    let target = con.target;

    let run = |lam: f64| -> Result<(f64, f64, f64, bool), ferric_core::FerricError> {
        let lw = lam * &w;
        let fm = |f_a: &mut Array2<f64>, f_b: &mut Array2<f64>| {
            *f_a += &lw;
            *f_b += &lw;
        };
        let scf = solve_uhf_fockmod(
            &sys.ctx,
            &sys.mol,
            &sys.prep,
            &sys.bounds,
            cfg,
            Some(guess),
            Some(&fm),
        )?;
        let d_a = &scf.density_alpha;
        let d_b = scf.density_beta.as_ref().unwrap_or(d_a);
        let n_c = population(&w, d_a, d_b, &con.spin);
        Ok((scf.energy, n_c - target, n_c, scf.converged))
    };

    let mut lam = 0.0_f64;
    let fd = 1e-3_f64;
    for _ in 0..30 {
        let (e, resid, n_c, conv) = run(lam)?;
        if resid.abs() < cfg.cdft_lambda_tol {
            return Ok((e, lam, n_c, conv));
        }
        let (_, resid_p, _, _) = run(lam + fd)?;
        let jac = (resid_p - resid) / fd;
        if jac.abs() < 1e-14 {
            return Err(ferric_core::FerricError::Lapack(
                "guess-restart Jacobian singular".into(),
            ));
        }
        lam -= (resid / jac).clamp(-1.0, 1.0);
    }
    Err(ferric_core::FerricError::Convergence(
        "guess-restart outer loop did not converge in 30 iters".into(),
    ))
}
