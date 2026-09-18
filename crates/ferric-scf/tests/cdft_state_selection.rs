//! **Part 1 — the discriminating experiment.** Is ferric's constrained HeNe⁺
//! state B multi-valued across orbital guesses at a fixed Becke target, and how
//! does that interact with the target itself?
//!
//! Hypotheses were pre-registered in `tests/HYPOTHESES-cdft-state-selection.md`
//! BEFORE any number in this file was measured. Read that first.
//!
//! # Why this exists
//!
//! Three external codes agree the UNCONSTRAINED HeNe⁺/def2-SVP state at
//! R = 2.0 Å is the σ-hole state (E = −130.5053405); ferric finds the ²Π state
//! 0.136 eV higher, and ORCA's `STABPerform` independently calls ferric's state
//! unstable. NWChem's CONSTRAINED solve at the same integer target N_He = 2.000
//! is itself multi-valued across its own guesses (−130.447402 from the default
//! atomic guess vs −130.423436 from hcore+swap), while at the natural
//! promolecule target the two guesses agree to <1e-7 Ha.
//!
//! # The two hypotheses, and the statistic that separates them
//!
//! * **H-GUESS** — the constrained problem has several solutions and the guess
//!   picks one. Then E is a function of the GUESS at fixed (target, N_final):
//!   two runs that both hit the target to 1e-8 can still differ by ~0.02 Ha.
//! * **H-SATURATE** — the Becke coordinate saturates below 2.000, so E is a
//!   function of the residual population error alone, and runs reaching the
//!   same `N_final` land at the same E regardless of guess.
//!
//! **The pre-registered discriminator:** partition the target = 2.000 runs by
//! `N_final` agreeing to 1e-7. If E still differs by ≫1e-6 Ha *within* such a
//! group, H-GUESS is supported and H-SATURATE refuted for that group.
//!
//! H-SATURATE makes a second, independent prediction the energy spread cannot
//! make: `dc/dλ` (the driver's own finite-difference Jacobian) must COLLAPSE
//! toward zero as the target approaches 2.000. `constraint_jacobian_does_not_
//! collapse_at_the_integer_target` measures exactly that, so the saturation
//! hypothesis is tested on its own terms and not merely by elimination.
//!
//! # Scope
//!
//! Everything here is `SpinChannel::Total`, fragment = [He] (atom 0), HeNe⁺ at
//! R = 2.0 Å in def2-SVP, at the lane's own knobs (`hene_cfg`). The driver
//! itself is not modified by this file; guesses are injected through
//! `solve_uhf_fockmod`, which already takes them.

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
use ferric_scf::stability::{uhf_internal_stability, StabilityConfig, StabilityVerdict};
use ferric_scf::uhf::solve_uhf_fockmod;
use ferric_scf::uhf_newton::{hessian_matvec, UhfNewtonInputs};
use ndarray::Array2;
use ndarray_linalg::{Eigh, Solve, UPLO};

const HENE_LAMBDA_TOL: f64 = 1e-5;

/// How close two runs' `N_final` must be for the H-GUESS/H-SATURATE
/// discriminator to treat them as "at the same point on the constraint
/// coordinate".
///
/// # Why 2e-7 and not the original 1e-7 (widened 2026-09-16)
///
/// The discriminator asks: do two runs that reached the SAME `N_final` still
/// land at different E? For that question to be answerable, two runs must
/// actually agree on `N_final` to within this tolerance.
///
/// The original 1e-7 was tight enough that a small perturbation of the guesses
/// left NO pair inside it. That happened when `fix/scf-unconstrained-state-
/// selection` changed the unconstrained reference: the integer-target runs
/// still occupy TWO clearly separated energy levels (−130.40219 and −130.42671,
/// 0.0245 Ha apart — the multi-valuedness is entirely intact), but their
/// `N_final` values spread to 1.99999955 … 2.00000034, i.e. pairwise gaps of a
/// few 1e-7. With no pair inside 1e-7 the maximum over an empty set is 0.0,
/// which the assertion then read as "H-SATURATE not excluded" — a verdict
/// produced by having measured NOTHING.
///
/// 2e-7 restores a populated comparison (SAD vs post-descent, |ΔE| = 0.0245 Ha)
/// and remains 50× TIGHTER than the solver's own `cdft_lambda_tol` of 1e-5, so
/// it is still a genuine "same constraint coordinate" claim and not a widening
/// that manufactures agreement. Measured sensitivity, so the choice is not
/// load-bearing: the within-group |ΔE| is 0.02451674 Ha at EVERY tolerance from
/// 2e-7 through 1e-6 — the verdict is flat across the whole band and only the
/// empty-set case at 1e-7 differs.
///
/// The accompanying reachability guard (`n_pairs > 0`) makes a recurrence
/// impossible to misread: an empty comparison now FAILS LOUDLY instead of
/// silently returning zero.
const N_PAIR_TOL: f64 = 2e-7;
const HENE_LEVEL_SHIFT: f64 = 0.5;
const R_ANG: f64 = 2.0;
/// Natural (promolecule) Becke population of He at R = 2.0 Å, from
/// `test/cdft-atomic-ip-anchor` (794a1551).
const NATURAL_N_HE: f64 = 1.954_484;

// ===========================================================================
// Scaffolding — mirrors cdft_constrained_stability.rs so the two are directly
// comparable. Duplicated rather than exported: neither test may perturb the
// lane it audits.
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

fn build_sys() -> Sys {
    let xyz = format!("2\nHeNe+\nHe 0.0 0.0 0.0\nNe 0.0 0.0 {R_ANG}\n");
    let mol = Molecule::parse_xyz(&xyz, 1, 2).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    Sys {
        mol,
        bs,
        prep,
        bounds,
        ctx,
        nocc_a: (nelec + two_s) / 2,
        nocc_b: (nelec - two_s) / 2,
    }
}

/// The lane's config, verbatim (see `cdft_constrained_stability::hene_cfg`).
///
/// `cdft_stability_descent: false` because PART 1 MEASURES THE UNFIXED SOLVER.
/// The whole point of the guess sweep is to characterise the landscape the
/// driver was navigating badly; running it with the descent on would measure
/// the fix instead and the multi-valuedness would be hidden by the very
/// mechanism that resolves it. The FIXED driver is exercised separately, by
/// `the_fixed_driver_reaches_the_lower_state` below, which sets the default.
///
/// `use_sad_guess: false` is pinned HERE, at the base config, rather than only
/// in [`cfg_with_target`] — see that function's doc for the full reasoning. It
/// belongs at this level because EVERY baseline in this file was recorded
/// against the hcore-started solver, and configs derived straight from
/// `hene_cfg` (via `..hene_cfg()` / `..hene_cfg_fixed()`) bypass
/// `cfg_with_target` entirely. When the pin lived only there, four tests that
/// build on this config ran from MINAO instead and died on
/// `cDFT outer loop did not converge in 30 iters` — the very non-convergence
/// `cfg_with_target`'s doc already records. The pin was incomplete, not the
/// observation wrong.
///
/// What this DOES NOT do: it does not claim the MINAO-started constrained loop
/// converges. It does not, and that gap is held open on purpose by
/// [`minao_started_cdft_does_not_converge_at_the_integer_target`].
fn hene_cfg() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        level_shift: HENE_LEVEL_SHIFT,
        cdft_lambda_tol: HENE_LAMBDA_TOL,
        // PINNED, not inherited: this file asserts "the constrained path
        // converges", and that assertion must not double as an assertion
        // about how many outer iterations one particular CPU needs.
        //
        // Measured requirement for every guess that converges at all
        // (this box; CI's counts differ -- that is the whole point):
        //
        //   post-descent  7 | MINAO 10 | target 1.98  12 | SAD        14
        //   driver/none  16 | target 1.99 14 | target 1.995 20 | hcore 23
        //
        // CI needed >30 for SAD where this box needs 14, so the old
        // hardcoded 30 failed the build on a machine difference. 40 is ~1.7x
        // the worst LOCAL count (23) and clear of CI's observed spread.
        //
        // Not higher: the cap is ALSO the price paid by paths that never
        // converge, since each wasted outer iteration runs a full inner SCF.
        // At 64 `cdft_coupling_hene` took 462 s vs 17 s at 30, because its
        // non-convergent sigma points burn the entire cap before failing.
        //
        // This is NOT a "make it converge eventually" cap: `state A (N_He=1)`
        // still does not converge at 200 (verified 2026-09-17), and the
        // catalogue documents it as a genuine non-converger at lines 893/902.
        // Raising the cap does not rescue it and is not meant to.
        cdft_max_outer: 40,
        cdft_stability_descent: false,
        use_sad_guess: false,
        dft_grid: Some(AtomicGridConfig {
            n_radial: 99,
            n_angular: 302,
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// The lane's constrained config at a given Becke target.
///
/// # Why `use_sad_guess: false` is pinned here (2026-09-16)
///
/// Every measured baseline in this file — the two constrained levels at the
/// integer target, their λ values, their stability verdicts, and the guess
/// catalogue's `state A` / `natural target` / `post-descent` reference
/// orbitals — was recorded against a solver whose open-shell path ALWAYS
/// started from hcore, because `uhf.rs` ignored `RhfConfig::use_sad_guess`
/// entirely.
///
/// `fix/scf-unconstrained-state-selection` made that field live. Left
/// unpinned, the λ-Newton loop here would start from MINAO instead, which is a
/// DIFFERENT numerical experiment from the one these baselines describe — and
/// measurably so: at the intermediate targets 1.990 and 1.995 the outer loop
/// stops converging within its 30-iteration cap, so `natural target` and
/// `state A` drop out of the catalogue and the Jacobian sweep loses half its
/// points.
///
/// This file's job is to AUDIT the constrained lane against its recorded
/// numbers, so it pins the guess those numbers were taken with rather than
/// silently re-baselining onto a different one. That the MINAO-started
/// constrained loop converges less readily at intermediate targets is a REAL
/// observation about the cDFT driver's outer-loop robustness, not a property
/// of the unconstrained fix — it is recorded here and left for the cDFT lane
/// to pursue, deliberately NOT tuned away by widening `max_outer`.
fn cfg_with_target(target: f64) -> RhfConfig {
    RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![0],
            target,
            spin: SpinChannel::Total,
        }],
        // Redundant since `hene_cfg` now pins this too, and KEPT redundant: it
        // is the pin this function's doc above describes, and a reader who
        // deletes it should have to reckon with that doc rather than discover
        // the behaviour changed two call-levels away.
        use_sad_guess: false,
        ..hene_cfg()
    }
}

fn weight_matrix(sys: &Sys, fragment: &[usize]) -> Array2<f64> {
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

fn density_from_mos(c: &Array2<f64>, nocc: usize) -> Array2<f64> {
    let occ = c.slice(ndarray::s![.., ..nocc]);
    occ.dot(&occ.t())
}

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

/// Both spin AO Focks at (c_a, c_b), optionally with the fixed λW added to each
/// spin (the `SpinChannel::Total` convention).
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

/// The λ-augmented energy `E[ρ] + λ N_C[ρ]`. The BARE Fock enters the ½-trace
/// electronic energy; adding λW there would double-count the linear term.
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

/// Exact lowest Hessian eigenvalue by DENSE construction from the same matvec.
/// `dim <= 148` here, so this is free, and it is the only check that catches a
/// converged-but-wrong iterative root (see `cdft_constrained_stability.rs`).
fn dense_lowest_eigenvalue(ctx: &ParallelContext, inp: &UhfNewtonInputs) -> f64 {
    let n = inp.c_a.nrows();
    let (na, nb) = (inp.nocc_a, inp.nocc_b);
    let sh_a = (n - na, na);
    let sh_b = (n - nb, nb);
    let dim_a = sh_a.0 * sh_a.1;
    let dim = dim_a + sh_b.0 * sh_b.1;
    let pool = EnginePool::new(inp.bounds.op, inp.prep, 1e-14).unwrap();
    let mut dense = Array2::<f64>::zeros((dim, dim));
    for col in 0..dim {
        let mut ka = Array2::<f64>::zeros(sh_a);
        let mut kb = Array2::<f64>::zeros(sh_b);
        if col < dim_a {
            ka.as_slice_mut().unwrap()[col] = 1.0;
        } else {
            kb.as_slice_mut().unwrap()[col - dim_a] = 1.0;
        }
        let (ha, hb) = hessian_matvec(ctx, inp, &ka, &kb, &pool).unwrap();
        for (row, val) in ha.iter().chain(hb.iter()).enumerate() {
            dense[(row, col)] = *val;
        }
    }
    let (vals, _) = dense.eigh(UPLO::Lower).unwrap();
    vals[0]
}

/// λ-augmented stability at a converged constrained state. Returns
/// `(dense λ_min, iterative λ_min, iterative residual, verdict label)`.
///
/// The verdict is read from the DENSE eigenvalue, which is exact by
/// construction; the iterative pair is reported alongside so a disagreement is
/// visible rather than absorbed.
fn augmented_stability(
    sys: &Sys,
    h: &Array2<f64>,
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    lam: f64,
    w: &Array2<f64>,
) -> (f64, f64, f64, &'static str) {
    let lam_w = lam * w;
    let (fa, fb) = ao_focks(sys, h, c_a, c_b, Some(&lam_w));
    let f_a_mo = c_a.t().dot(&fa).dot(c_a);
    let f_b_mo = c_b.t().dot(&fb).dot(c_b);
    let inp = newton_inputs(sys, c_a, c_b, &f_a_mo, &f_b_mo);
    let dense_min = dense_lowest_eigenvalue(&sys.ctx, &inp);
    let cfg = StabilityConfig {
        conv_thresh: 1e-8,
        max_iter: 100,
        ..Default::default()
    };
    let st = uhf_internal_stability(&sys.ctx, &inp, &cfg).expect("stability analysis");
    let dense_verdict = ferric_scf::stability::StabilityResult {
        lowest_eigenvalue: dense_min,
        converged: true,
        residual: 0.0,
        ..st.clone()
    };
    let label = match dense_verdict.verdict() {
        StabilityVerdict::Stable => "STABLE",
        StabilityVerdict::Unstable => "UNSTABLE",
        StabilityVerdict::Marginal => "MARGINAL",
        StabilityVerdict::Indeterminate => "INDETERMINATE",
    };
    (dense_min, st.lowest_eigenvalue, st.residual, label)
}

// ===========================================================================
// The λ-Newton loop, reproduced around an EXPLICIT guess
// ===========================================================================

/// One constrained run: the driver's λ-Newton loop, but every inner
/// `solve_uhf_fockmod` starts from `guess` (or hcore when `None`, which is what
/// the driver does today).
///
/// Everything else — grid, weight matrix, residual, FD Jacobian, ±1 clamp,
/// 30-iteration budget, tolerance — matches `cdft_driver::solve_cdft_uhf`
/// exactly. `jacobians` records `dc/dλ` at every outer step, which is the
/// saturation observable.
#[allow(clippy::type_complexity)]
#[derive(Clone)]
struct Run {
    e_bare: f64,
    lambda: f64,
    n_final: f64,
    outer_iters: usize,
    converged: bool,
    c_a: Array2<f64>,
    c_b: Array2<f64>,
    jacobians: Vec<f64>,
}

fn constrained_run(
    sys: &Sys,
    cfg: &RhfConfig,
    w: &Array2<f64>,
    guess: Option<(&Array2<f64>, &Array2<f64>)>,
) -> Result<Run, ferric_core::FerricError> {
    let con = &cfg.constraints[0];
    let target = con.target;
    let run = |lam: f64| -> Result<
        (f64, f64, f64, bool, Array2<f64>, Array2<f64>),
        ferric_core::FerricError,
    > {
        let lw = lam * w;
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
            guess,
            Some(&fm),
        )?;
        let d_a = &scf.density_alpha;
        let d_b = scf.density_beta.as_ref().unwrap_or(d_a);
        let n_c = population(w, d_a, d_b, &con.spin);
        let cb = scf
            .mos_beta
            .clone()
            .unwrap_or_else(|| scf.mos_alpha.clone());
        Ok((
            scf.energy,
            n_c - target,
            n_c,
            scf.converged,
            scf.mos_alpha,
            cb,
        ))
    };

    let mut lam = 0.0_f64;
    let fd = 1e-3_f64;
    let mut jacobians = Vec::new();
    // Read the SAME cap the library driver reads, so this hand-rolled mirror
    // of the λ-Newton loop cannot diverge from `solve_cdft_uhf` on the one
    // axis this file's tests are most sensitive to. (It was a hardcoded 30
    // until 2026-09-17, which is why CI reported "SAD DID NOT CONVERGE in 30"
    // on a path that needs 14 iterations locally -- see `cdft_max_outer`.)
    for outer in 1..=cfg.cdft_max_outer {
        let (e, resid, n_c, conv, ca, cb) = run(lam)?;
        if resid.abs() < cfg.cdft_lambda_tol {
            return Ok(Run {
                e_bare: e,
                lambda: lam,
                n_final: n_c,
                outer_iters: outer,
                converged: conv,
                c_a: ca,
                c_b: cb,
                jacobians,
            });
        }
        let (_, resid_p, _, _, _, _) = run(lam + fd)?;
        let jac = (resid_p - resid) / fd;
        jacobians.push(jac);
        if jac.abs() < 1e-14 {
            return Err(ferric_core::FerricError::Lapack(
                "state-selection Jacobian singular".into(),
            ));
        }
        lam -= (resid / jac).clamp(-1.0, 1.0);
    }
    Err(ferric_core::FerricError::Convergence(format!(
        "state-selection outer loop did not converge in {} iters",
        cfg.cdft_max_outer
    )))
}

// ===========================================================================
// The guess catalogue
// ===========================================================================

/// An orbital guess, named so the table can be partitioned by guess IDENTITY —
/// the observable that separates H-GUESS from H-SATURATE.
struct Guess {
    name: &'static str,
    c_a: Array2<f64>,
    c_b: Array2<f64>,
}

/// Canonical orthogonalizer `X = U s^{-1/2}` from the overlap matrix.
///
/// `ferric_scf::rhf::canonical_orthogonalizer` is private, so this is rebuilt
/// here. It is NOT lindep-filtered, which is safe ONLY because the smallest
/// overlap eigenvalue on HeNe⁺/def2-SVP is far above the library's 1e-6
/// threshold — asserted here rather than assumed, so this helper cannot be
/// silently reused on an ill-conditioned basis.
fn orthogonalizer(s: &Array2<f64>) -> Array2<f64> {
    let (vals, vecs) = s.eigh(UPLO::Lower).unwrap();
    assert!(
        vals[0] > 1e-5,
        "unfiltered orthogonalizer used on a near-linearly-dependent basis \
         (s_min = {:.3e}); this helper is only valid where lindep filtering is a \
         no-op",
        vals[0]
    );
    let mut x = vecs.clone();
    for (j, &v) in vals.iter().enumerate() {
        let inv = 1.0 / v.sqrt();
        for i in 0..x.nrows() {
            x[(i, j)] *= inv;
        }
    }
    x
}

/// hcore MOs: diagonalize h in the canonically-orthogonalized basis. This is
/// what the driver uses today (via `solve_uhf_fockmod`'s `None` branch), so it
/// reproduces the lane's own starting point explicitly.
fn hcore_mos(sys: &Sys) -> (Array2<f64>, Array2<f64>) {
    let h = oneelectron::hcore(&sys.prep);
    let s = oneelectron::overlap(&sys.prep);
    let x = orthogonalizer(&s);
    let hp = x.t().dot(&h).dot(&x);
    let (_, cp) = hp.eigh(UPLO::Lower).unwrap();
    let c = x.dot(&cp);
    (c.clone(), c)
}

/// SAD MOs: build the superposition-of-atomic-densities guess, form the Fock at
/// that density, and diagonalize it. `sad_guess` returns a DENSITY, so one Fock
/// build is needed to turn it into orbitals.
fn sad_mos(sys: &Sys) -> (Array2<f64>, Array2<f64>) {
    let n = sys.prep.nbasis();
    let h = oneelectron::hcore(&sys.prep);
    let s = oneelectron::overlap(&sys.prep);
    let x = orthogonalizer(&s);
    let d_tot = ferric_scf::guess::sad_guess(&sys.mol, &sys.prep, &sys.bs).unwrap();
    // SAD's D is spin-summed; split it evenly to form a spin Fock.
    let d_spin = 0.5 * &d_tot;
    let mut j = Array2::<f64>::zeros((n, n));
    let mut kdum = Array2::<f64>::zeros((n, n));
    build_jk(
        &sys.ctx,
        &sys.prep,
        &sys.bounds,
        1e-12,
        &d_tot,
        &mut j,
        &mut kdum,
    )
    .unwrap();
    let mut jd = Array2::<f64>::zeros((n, n));
    let mut k = Array2::<f64>::zeros((n, n));
    build_jk(
        &sys.ctx,
        &sys.prep,
        &sys.bounds,
        1e-12,
        &d_spin,
        &mut jd,
        &mut k,
    )
    .unwrap();
    let f = &h + &j - &k;
    let fp = x.t().dot(&f).dot(&x);
    let (_, cp) = fp.eigh(UPLO::Lower).unwrap();
    let c = x.dot(&cp);
    (c.clone(), c)
}

/// The UNCONSTRAINED ferric UHF solution's orbitals — what NWChem's default
/// atomic guess effectively hands its constrained solve.
/// The unconstrained UHF solution reached from the BARE HCORE guess — the ²Π
/// state, E = −130.50034664.
///
/// # Why `use_sad_guess: false` is pinned here (2026-09-16)
///
/// This entry is one of SIX guesses whose whole purpose is to present the
/// constrained solver with GENUINELY DIFFERENT starting orbitals, so that the
/// pre-registered discriminator — "two runs reaching the same `N_final` to
/// 1e-7 that still land at different E" — has distinct inputs to compare.
///
/// Until 2026-09-16 `solve_uhf_fockmod` ignored `RhfConfig::use_sad_guess` and
/// always started from hcore, so this returned the ²Π state.
/// `fix/scf-unconstrained-state-selection` made the open-shell path honour that
/// field (default MINAO), and this entry silently became the σ state instead —
/// which sits in the SAME constrained basin as the natural-target and
/// post-descent guesses. The catalogue lost a basin, the discriminator lost its
/// distinguishing pair, and the test failed.
///
/// That was the fix working on an input this test did not intend to change.
/// The repair is to keep this entry the ²Π-referenced guess it was written to
/// be, and to add the new σ reference as a SEPARATE entry
/// ([`unconstrained_sigma_mos`]) rather than replacing one with the other —
/// so the catalogue now spans MORE basins than before, not fewer.
fn unconstrained_mos(sys: &Sys) -> (Array2<f64>, Array2<f64>) {
    let cfg = RhfConfig {
        use_sad_guess: false,
        ..hene_cfg()
    };
    let scf =
        solve_uhf_fockmod(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg, None, None).unwrap();
    let cb = scf.mos_beta.clone().unwrap();
    (scf.mos_alpha, cb)
}

/// The unconstrained UHF solution reached from the DEFAULT (MINAO) guess — the
/// σ state, E = −130.5053405386, which NWChem/ORCA/PySCF all agree is the true
/// unconstrained ground state.
///
/// New on 2026-09-16: before the guess fix this state was not reachable from
/// any default path, so the catalogue had no entry referencing it.
fn unconstrained_sigma_mos(sys: &Sys) -> (Array2<f64>, Array2<f64>) {
    let scf = solve_uhf_fockmod(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        &hene_cfg(),
        None,
        None,
    )
    .unwrap();
    let cb = scf.mos_beta.clone().unwrap();
    (scf.mos_alpha, cb)
}

/// Converged STATE A orbitals (hole on He, N_He → 1).
fn state_a_mos(sys: &Sys, w: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let r = constrained_run(sys, &cfg_with_target(1.0), w, None).unwrap();
    (r.c_a, r.c_b)
}

/// Converged orbitals at the NATURAL promolecule target.
fn natural_target_mos(sys: &Sys, w: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let r = constrained_run(sys, &cfg_with_target(NATURAL_N_HE), w, None).unwrap();
    (r.c_a, r.c_b)
}

/// The POST-DESCENT state: default state B, rotated along its negative
/// λ-augmented eigenvector by ~1 radian and re-converged. The caveat recorded
/// on `test/cdft-constrained-stability` is honored — small steps (ε ≤ 0.5) fall
/// back into the saddle's own DIIS basin, so the step here is 0.8 rad, the one
/// that was MEASURED to escape.
fn post_descent_mos(sys: &Sys, w: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let h = oneelectron::hcore(&sys.prep);
    let cfg = cfg_with_target(2.0);
    let r = constrained_run(sys, &cfg, w, None).unwrap();
    let lam_w = r.lambda * w;
    let (fa, fb) = ao_focks(sys, &h, &r.c_a, &r.c_b, Some(&lam_w));
    let f_a_mo = r.c_a.t().dot(&fa).dot(&r.c_a);
    let f_b_mo = r.c_b.t().dot(&fb).dot(&r.c_b);
    let inp = newton_inputs(sys, &r.c_a, &r.c_b, &f_a_mo, &f_b_mo);
    let stab_cfg = StabilityConfig {
        conv_thresh: 1e-8,
        max_iter: 100,
        ..Default::default()
    };
    let st = uhf_internal_stability(&sys.ctx, &inp, &stab_cfg).unwrap();
    let va = st.eigenvector_alpha.clone();
    let vb = st.eigenvector_beta.clone().unwrap();
    (
        rotate(&r.c_a, &va, sys.nocc_a, 0.8),
        rotate(&r.c_b, &vb, sys.nocc_b, 0.8),
    )
}

fn guess_catalogue(sys: &Sys, w: &Array2<f64>) -> Vec<Guess> {
    let mut g = Vec::new();
    let (a, b) = hcore_mos(sys);
    g.push(Guess {
        name: "hcore (= driver default)",
        c_a: a,
        c_b: b,
    });
    let (a, b) = sad_mos(sys);
    g.push(Guess {
        name: "SAD",
        c_a: a,
        c_b: b,
    });
    let (a, b) = unconstrained_mos(sys);
    g.push(Guess {
        name: "unconstrained UHF (pi)",
        c_a: a,
        c_b: b,
    });
    let (a, b) = unconstrained_sigma_mos(sys);
    g.push(Guess {
        name: "unconstrained UHF (sigma)",
        c_a: a,
        c_b: b,
    });
    let (a, b) = state_a_mos(sys, w);
    g.push(Guess {
        name: "state A (N_He=1)",
        c_a: a,
        c_b: b,
    });
    let (a, b) = natural_target_mos(sys, w);
    g.push(Guess {
        name: "natural target (1.954484)",
        c_a: a,
        c_b: b,
    });
    let (a, b) = post_descent_mos(sys, w);
    g.push(Guess {
        name: "post-descent (0.8 rad)",
        c_a: a,
        c_b: b,
    });
    g
}

// ===========================================================================
// PART 1a — the guess sweep at a fixed target
// ===========================================================================

/// One row of the guess × target table.
struct Row {
    guess: &'static str,
    target: f64,
    e: f64,
    n_final: f64,
    lambda: f64,
    outer: usize,
    dense_lmin: f64,
    verdict: &'static str,
}

fn sweep_guesses_at(sys: &Sys, w: &Array2<f64>, target: f64, guesses: &[Guess]) -> Vec<Row> {
    let h = oneelectron::hcore(&sys.prep);
    let cfg = cfg_with_target(target);
    let mut rows = Vec::new();
    // The driver's OWN run (guess = None), first: this is the lane's answer.
    match constrained_run(sys, &cfg, w, None) {
        Ok(r) => {
            // A row from an inner SCF that did not converge is not a
            // measurement of a solution — it would put a non-stationary point
            // into the guess table and the Hessian verdict would describe
            // nothing. Fail loudly rather than average it in.
            assert!(
                r.converged,
                "driver default (None) at target {target}: the inner SCF did not \
                 converge, so this row is not a solution"
            );
            let (dmin, imin, ires, verdict) =
                augmented_stability(sys, &h, &r.c_a, &r.c_b, r.lambda, w);
            eprintln!(
                "  {:<28} E = {:.8}  N = {:.8}  λ = {:+.6}  outer = {:>2}  \
                 λ_min(dense) = {:+.6e}  [{}]  (iter {:+.6e}, resid {:.1e})",
                "driver default (None)",
                r.e_bare,
                r.n_final,
                r.lambda,
                r.outer_iters,
                dmin,
                verdict,
                imin,
                ires
            );
            rows.push(Row {
                guess: "driver default (None)",
                target,
                e: r.e_bare,
                n_final: r.n_final,
                lambda: r.lambda,
                outer: r.outer_iters,
                dense_lmin: dmin,
                verdict,
            });
        }
        Err(e) => eprintln!("  {:<28} DID NOT CONVERGE: {e:?}", "driver default (None)"),
    }
    for g in guesses {
        match constrained_run(sys, &cfg, w, Some((&g.c_a, &g.c_b))) {
            Ok(r) => {
                assert!(
                    r.converged,
                    "guess {:?} at target {target}: the inner SCF did not converge, \
                     so this row is not a solution",
                    g.name
                );
                let (dmin, imin, ires, verdict) =
                    augmented_stability(sys, &h, &r.c_a, &r.c_b, r.lambda, w);
                eprintln!(
                    "  {:<28} E = {:.8}  N = {:.8}  λ = {:+.6}  outer = {:>2}  \
                     λ_min(dense) = {:+.6e}  [{}]  (iter {:+.6e}, resid {:.1e})",
                    g.name, r.e_bare, r.n_final, r.lambda, r.outer_iters, dmin, verdict, imin, ires
                );
                rows.push(Row {
                    guess: g.name,
                    target,
                    e: r.e_bare,
                    n_final: r.n_final,
                    lambda: r.lambda,
                    outer: r.outer_iters,
                    dense_lmin: dmin,
                    verdict,
                });
            }
            Err(e) => eprintln!("  {:<28} DID NOT CONVERGE: {e:?}", g.name),
        }
    }
    rows
}

/// **THE PART 1 EXPERIMENT.** Guess sweep at the INTEGER target (2.000) and at
/// the NATURAL target (1.954484), with the pre-registered discriminator applied
/// to the result.
///
/// MEASURED (2026-09-16), full table on stderr. **H-GUESS CONFIRMED,
/// H-SATURATE REFUTED.** Every run below reached its target to ≤ 6e-7.
///
/// **ADDENDUM 2026-09-17.** The two `state A` rows below read "did not
/// converge in 30 outer iters" because 30 was the hardcoded cap when the
/// table was recorded. Re-running at `cdft_max_outer = 200` leaves them
/// UNCONVERGED, so `state A` is a genuine non-converger, not an
/// iteration-starved one. The cap is now a config knob pinned to 40 in
/// `hene_cfg()`; every other row's `outer` count below is unchanged by it.
///
/// ```text
/// target = 2.000000 (INTEGER)
///   guess                       E             N_final     λ         outer  λ_min(dense)  verdict
///   driver default (None)   -130.40219057  1.99999977  -2.753705   16   -3.999187e-2  UNSTABLE
///   hcore (= driver dflt)   -130.40219117  1.99999955  -2.753704   23   -3.999169e-2  UNSTABLE
///   SAD                     -130.40219036  1.99999985  -2.753705   14   -3.999187e-2  UNSTABLE
///   unconstrained UHF       -130.42670616  2.00000034  -2.439015   16   -1.017364e-8  MARGINAL
///   state A (N_He=1)        did not converge in 30 outer iters
///   natural target          -130.42670586  2.00000046  -2.439016   20   -9.803376e-9  MARGINAL
///   post-descent (0.8 rad)  -130.42670716  1.99999993  -2.439011    7   -1.010221e-9  MARGINAL
///
/// target = 1.954484 (NATURAL)
///   driver default (None)   -130.50031848  1.95448398  -0.060075    3   -3.718355e-3  UNSTABLE
///   hcore                   -130.50031848  1.95448398  -0.060075    3   -3.718355e-3  UNSTABLE
///   SAD                     -130.50065655  1.95448398  -0.625186    6   -1.795124e-3  UNSTABLE
///   unconstrained UHF       -130.50031848  1.95448398  -0.060075    3   -3.718363e-3  UNSTABLE
///   state A (N_He=1)        did not converge in 30 outer iters
///   natural target          -130.50031848  1.95448398  -0.060075    3   -3.718357e-3  UNSTABLE
///   post-descent (0.8 rad)  -130.50065655  1.95448398  -0.625186    6   -1.795111e-3  UNSTABLE
///
///   [spread] integer 0.02451680 Ha over 6 runs | natural 0.00033807 Ha over 6 runs
///   [discriminator] largest |ΔE| at N_final-matched (1e-7) integer pairs:
///                   0.02451680 Ha  (SAD vs post-descent)
/// ```
///
/// **The discriminator fires.** SAD and post-descent both land on N_final =
/// 1.9999998/1.9999999 — matching to 8e-8 — and still differ in energy by
/// 0.0245 Ha, 24 500× the 1e-6 bar. E is therefore a function of the GUESS at
/// fixed (target, N_final), which is H-GUESS and is incompatible with
/// H-SATURATE's "E is a function of the residual population error alone".
///
/// **Two distinct constrained solutions exist at the integer target**, each with
/// its own λ, and the guesses partition cleanly between them:
///   * the UPPER one, E = −130.402190, λ = −2.7537, which is the state the
///     driver reports today and which its own λ-augmented stability calls a
///     SADDLE (λ_min = −4.0e-2, seven orders above the eigensolver residual);
///   * the LOWER one, E = −130.426706, λ = −2.4390, 0.0245 Ha = 0.667 eV below,
///     MARGINAL (|λ_min| ≤ 1e-8, at the noise floor — "lower" is established,
///     "the bottom" is not).
/// The λ values differ by 0.31 between the two levels, so these are genuinely
/// different points of the λ-Newton problem, not one solution reached twice.
///
/// **Which guesses reach which** is the actionable part: hcore (what the driver
/// uses) and SAD land on the SADDLE; the UNCONSTRAINED UHF solution, the
/// natural-target solution, and the post-descent state all land on the LOWER
/// one. That is exactly the asymmetry the brief predicted from NWChem, whose
/// default atomic guess reaches its own LOW member.
#[test]
fn state_b_energy_is_multi_valued_across_guesses_at_the_integer_target() {
    let sys = build_sys();
    let w = weight_matrix(&sys, &[0]);
    let guesses = guess_catalogue(&sys, &w);

    eprintln!("\n=== target = 2.000000 (INTEGER, over-constrained) ===");
    let integer_rows = sweep_guesses_at(&sys, &w, 2.0, &guesses);
    eprintln!("\n=== target = {NATURAL_N_HE:.6} (NATURAL promolecule) ===");
    let natural_rows = sweep_guesses_at(&sys, &w, NATURAL_N_HE, &guesses);

    assert!(
        integer_rows.len() >= 4,
        "too few converged runs at the integer target to judge multi-valuedness: {}",
        integer_rows.len()
    );

    let spread = |rows: &[Row]| -> f64 {
        let lo = rows.iter().fold(f64::INFINITY, |m, r| m.min(r.e));
        let hi = rows.iter().fold(f64::NEG_INFINITY, |m, r| m.max(r.e));
        hi - lo
    };
    let int_spread = spread(&integer_rows);
    let nat_spread = spread(&natural_rows);
    eprintln!(
        "\n[spread] integer target: {int_spread:.8} Ha over {} runs\n\
         [spread] natural target: {nat_spread:.8} Ha over {} runs",
        integer_rows.len(),
        natural_rows.len()
    );

    // --- THE PRE-REGISTERED DISCRIMINATOR ---------------------------------
    // Partition the integer-target runs by N_final agreeing to N_PAIR_TOL. If E
    // still differs by >> 1e-6 Ha WITHIN such a group, H-GUESS is supported and
    // H-SATURATE is refuted for that group: the runs reached the same point on
    // the constraint coordinate and still landed at different energies, so the
    // residual population error cannot be what selects the state.
    let mut worst_within_group = 0.0_f64;
    let mut worst_pair = ("", "");
    let mut n_pairs = 0usize;
    for i in 0..integer_rows.len() {
        for j in (i + 1)..integer_rows.len() {
            let (a, b) = (&integer_rows[i], &integer_rows[j]);
            if (a.n_final - b.n_final).abs() < N_PAIR_TOL {
                n_pairs += 1;
                let de = (a.e - b.e).abs();
                if de > worst_within_group {
                    worst_within_group = de;
                    worst_pair = (a.guess, b.guess);
                }
            }
        }
    }
    eprintln!(
        "[discriminator] {n_pairs} pair(s) of integer-target runs agree on N_final to \
         {N_PAIR_TOL:.0e}; largest |ΔE| among them: {worst_within_group:.8} Ha \
         ({} vs {})",
        worst_pair.0, worst_pair.1
    );

    // REACHABILITY GUARD. Without this, a run in which NO two guesses land
    // within N_PAIR_TOL of each other yields worst_within_group = 0.0 — which
    // reads exactly like "H-SATURATE not excluded" while actually meaning "the
    // discriminator had nothing to compare". That is the repo's documented
    // failure mode: a gate whose GO condition is unreachable returns
    // ARITHMETIC, NOT MEASUREMENT. The distinction is not hypothetical here —
    // it is precisely what happened on 2026-09-16 when the guess fix perturbed
    // every N_final by a few 1e-7 (see N_PAIR_TOL's comment). Assert the
    // comparison actually took place BEFORE reading its verdict.
    assert!(
        n_pairs > 0,
        "the discriminator compared NOTHING: no two integer-target runs agreed on \
         N_final to {N_PAIR_TOL:.0e}, so worst_within_group = 0.0 is an empty maximum, \
         not evidence about H-SATURATE. Widen N_PAIR_TOL toward the solver's own \
         cdft_lambda_tol ({HENE_LAMBDA_TOL:.0e}) or tighten the solve; do NOT read \
         this as a refutation."
    );

    assert!(
        int_spread > 1e-6,
        "H-GUESS REFUTED: E at the integer target is SINGLE-VALUED across every \
         guess (spread {int_spread:.3e} Ha). The premise of the guess-dependence \
         fix has not been reproduced and no guess-based fix is warranted."
    );
    assert!(
        worst_within_group > 1e-6,
        "H-SATURATE not excluded: every pair of integer-target runs reaching the \
         same N_final (to {N_PAIR_TOL:.0e}) also reached the same E (worst {worst_within_group:.3e} \
         Ha). E would then be a function of the residual population error alone, \
         not of the guess."
    );
    // --- THE CONTROL, AND ITS REFUTATION ----------------------------------
    //
    // PRE-REGISTERED EXPECTATION, NOT MET, RECORDED RATHER THAN RELAXED.
    //
    // The hypotheses file predicted the natural target would be SINGLE-VALUED
    // (< 1e-6 Ha), because NWChem's two guesses agree to < 1e-7 there. The
    // original assertion was `nat_spread < 1e-5` and it FAILED:
    //
    // ```text
    //   [spread] natural target: 0.00033807 Ha over 6 runs
    //   panicked: ferric spread = 3.381e-4 Ha
    // ```
    //
    // MEASURED: the natural target is multi-valued in ferric too, with two
    // levels, and they are ALSO separated by guess identity at an identical
    // N_final (1.95448398 for every run):
    //   hcore / driver default / unconstrained / natural  E = -130.50031848  λ = -0.060075
    //   SAD / post-descent                                E = -130.50065655  λ = -0.625186
    //
    // So guess-dependence is NOT confined to the over-constrained integer
    // target. It is just 72x smaller there (3.4e-4 vs 2.5e-2 Ha), which is why
    // NWChem's two guesses — a much narrower pair than these six — did not
    // resolve it. That is a WIDER defect than the brief assumed, and the bar is
    // therefore re-stated as what was measured rather than loosened until the
    // original claim survived.
    //
    // Note both natural-target solutions are UNSTABLE (λ_min = -3.72e-3 and
    // -1.80e-3), so neither is the bottom; the natural target is not a "safe"
    // regime, merely a less dramatic one.
    assert!(
        nat_spread > 1e-6,
        "the natural target was measured MULTI-VALUED (spread 3.381e-4 Ha across \
         six guesses at an identical N_final). A spread of {nat_spread:.3e} Ha \
         means that finding no longer reproduces — which would be a NEW result, \
         not a pass."
    );
    // …and the integer target's spread is far larger, which is the ordering the
    // brief's external evidence implies even though the natural target is not
    // clean.
    assert!(
        int_spread > 10.0 * nat_spread,
        "the integer target's guess spread ({int_spread:.3e} Ha) was measured 72x \
         the natural target's ({nat_spread:.3e} Ha); that ordering has changed"
    );

    // --- PIN THE GUESS → SOLUTION MAP -------------------------------------
    //
    // This is the fix's whole justification, so it is asserted rather than
    // printed. Each guess must land on the level it was MEASURED to land on,
    // with the λ and the stability verdict that go with that level. A future
    // change that makes hcore reach the lower state (the intended fix) will
    // fail HERE, which is correct: it must be a deliberate, reviewed update of
    // the measured baseline, not a silent drift.
    let find = |g: &str| -> &Row {
        integer_rows
            .iter()
            .find(|r| r.guess == g)
            .unwrap_or_else(|| panic!("integer-target run for guess {g:?} is missing"))
    };
    for g in ["driver default (None)", "hcore (= driver default)", "SAD"] {
        let r = find(g);
        assert!(
            (r.e - (-130.402_190_5)).abs() < 1e-5 && (r.lambda - (-2.753_70)).abs() < 1e-3,
            "{g} was measured on the UPPER (saddle) solution E = -130.4021905, \
             λ = -2.75370; got E = {:.8}, λ = {:+.6}",
            r.e,
            r.lambda
        );
        assert_eq!(
            r.verdict, "UNSTABLE",
            "{g}'s upper solution was measured a SADDLE (λ_min = -4.0e-2); got \
             {} at λ_min = {:.6e}",
            r.verdict, r.dense_lmin
        );
        assert!((r.target - 2.0).abs() < 1e-12);
        assert!(r.outer <= 30);
    }
    // "unconstrained UHF (pi)" is the row that used to be called
    // "unconstrained UHF": before the 2026-09-16 guess fix the unconstrained
    // solve returned the pi state unconditionally. It still lands on the LOWER
    // constrained solution, which is the measured fact this pins; only the
    // label changed, because a SECOND unconstrained reference (the sigma state)
    // now exists and the two must be distinguishable. The sigma row is NOT
    // listed here: it does not converge at the integer target (recorded in the
    // sweep output), so there is no measured baseline to pin it against.
    for g in [
        "unconstrained UHF (pi)",
        "natural target (1.954484)",
        "post-descent (0.8 rad)",
    ] {
        let r = find(g);
        assert!(
            (r.e - (-130.426_706_5)).abs() < 1e-5 && (r.lambda - (-2.439_01)).abs() < 1e-3,
            "{g} was measured on the LOWER solution E = -130.4267065, λ = -2.43901; \
             got E = {:.8}, λ = {:+.6}",
            r.e,
            r.lambda
        );
        assert_eq!(
            r.verdict, "MARGINAL",
            "{g}'s lower solution was measured MARGINAL (|λ_min| <= 1e-8, at the \
             noise floor); got {} at λ_min = {:.6e}",
            r.verdict, r.dense_lmin
        );
    }
}

// ===========================================================================
// PART 1b — the target sweep, and H-SATURATE's own prediction
// ===========================================================================

/// **H-SATURATE's independent prediction, tested on its own terms.**
///
/// If the Becke He population saturates below 2.000, the constraint Jacobian
/// `dc/dλ` must COLLAPSE toward zero as the target approaches the integer, and
/// |λ| must grow without bound. That is a prediction about the CONSTRAINT
/// COORDINATE, not about energies, so it does not reduce to the energy-spread
/// discriminator and can refute H-SATURATE even where the spread cannot.
///
/// Reported for every target in 1.954484 → 2.000. `dc/dλ` is the driver's own
/// finite-difference Jacobian, taken from the LAST outer step (the one nearest
/// the solution).
#[test]
fn constraint_jacobian_does_not_collapse_at_the_integer_target() {
    let sys = build_sys();
    let w = weight_matrix(&sys, &[0]);
    let targets = [NATURAL_N_HE, 1.98, 1.99, 1.995, 2.0];
    let mut last_jacs = Vec::new();
    eprintln!("\n=== target sweep (driver default guess) ===");
    for &t in &targets {
        match constrained_run(&sys, &cfg_with_target(t), &w, None) {
            Ok(r) => {
                let jac = r.jacobians.last().copied().unwrap_or(f64::NAN);
                eprintln!(
                    "  target {t:.6}  E = {:.8}  N = {:.8}  λ = {:+.6}  outer = {:>2}  \
                     |N−target| = {:.2e}  dc/dλ(last) = {jac:+.6e}",
                    r.e_bare,
                    r.n_final,
                    r.lambda,
                    r.outer_iters,
                    (r.n_final - t).abs()
                );
                last_jacs.push((t, jac, r.lambda, r.n_final));
            }
            Err(e) => eprintln!("  target {t:.6}  DID NOT CONVERGE: {e:?}"),
        }
    }
    assert!(
        last_jacs.len() >= 4,
        "need most of the target sweep to converge to judge saturation"
    );

    // The constraint is SATISFIED at every target: |N_final − target| is at the
    // λ-tolerance, not stuck at an asymptote short of it. A saturating
    // coordinate could not do this.
    for &(t, _, _, n) in &last_jacs {
        assert!(
            (n - t).abs() < 10.0 * HENE_LAMBDA_TOL,
            "target {t:.6} was NOT reached (N = {n:.8}); that WOULD be saturation"
        );
    }

    // And dc/dλ does not collapse: the integer target's Jacobian is within an
    // order of magnitude of the natural target's. Saturation would drive this
    // ratio toward zero.
    let jac_nat = last_jacs[0].1.abs();
    let jac_int = last_jacs.last().unwrap().1.abs();
    eprintln!(
        "[saturation] |dc/dλ| natural = {jac_nat:.6e}, integer = {jac_int:.6e}, \
         ratio = {:.4}",
        jac_int / jac_nat
    );
    // MEASURED ratio is 15.43 — dc/dλ GROWS toward the integer target, the
    // OPPOSITE of saturation.
    //
    // The bar is `> 2.0`, not the original `> 0.1`, and that is a CORRECTION
    // rather than a tightening-for-its-own-sake. Mutation 10 replaced
    // `jacobians.last()` with `jacobians.first()` and the test STILL PASSED, at
    // a ratio of exactly 1.0000 — because the FIRST outer step is always taken
    // from λ = 0 and therefore measures the same quantity at every target,
    // carrying no information about the target at all. A pass condition that a
    // target-INDEPENDENT constant clears is not measuring the target. `> 2.0`
    // rejects that constant (and any ratio consistent with saturation) while
    // sitting 7.7x below the measured 15.43.
    assert!(
        jac_int / jac_nat > 2.0,
        "H-SATURATE predicts dc/dλ COLLAPSING at the integer target; it was measured \
         GROWING 15.43x. A ratio of {:.4e} (natural {jac_nat:.3e} → integer \
         {jac_int:.3e}) is neither — note that a ratio near 1.0 is the signature of \
         reading a target-INDEPENDENT Jacobian (e.g. the first outer step, always \
         taken from λ = 0) rather than the one nearest the solution.",
        jac_int / jac_nat
    );
}

// ===========================================================================
// λ ≠ 0 VALIDATION — the check the λ = 0 anchor is structurally blind to
// ===========================================================================

/// **The λ ≠ 0 validation, which does NOT reduce to the λ = 0 anchor.**
///
/// ferric's cDFT exactness anchor sits at λ = 0, so it cannot see any defect
/// proportional to λ — the exact class of bug found in QE's `epcdft`, which
/// double-adds a λ-proportional term and still passes its own λ = 0 anchor
/// bit-identically.
///
/// This test finite-differences the AUGMENTED ENERGY `E[ρ] + λ N_C[ρ]` along a
/// random orbital rotation at the converged state B (λ ≈ −2.75, the largest
/// |λ| in the lane) and compares against the ANALYTIC augmented gradient (the
/// occ→virt block of the λ-augmented MO Fock). A term double-added at strength
/// λ shifts the analytic gradient by exactly that amount while the FD of the
/// correctly-written functional does not follow, so the two disagree at
/// O(λ) — invisible at λ = 0 by construction.
///
/// # Why the check is run AWAY from the converged point
///
/// First attempt evaluated at the converged state itself and its own
/// non-vacuity guard rejected it:
///
/// ```text
///   [λ≠0 FD] λ = -2.753705  max|analytic| = 3.385056e-8  rel = 6.459e-3
///   panicked: the augmented gradient is ~0 at this point, so agreement is vacuous
/// ```
///
/// That is Brillouin's theorem doing its job: at a converged solution the
/// occ→virt gradient IS zero, so "analytic == FD" there is `0 == 0` and carries
/// no information about the λ term. The comparison therefore has to happen at a
/// NON-STATIONARY point, which is what `off_stationary_orbitals` produces — a
/// deterministic pseudo-random rotation off the solution, large enough that the
/// gradient is O(0.1) rather than O(1e-8). The λ is still the converged
/// λ = −2.7537, so it is still a genuine λ ≠ 0 test.
///
/// Its REACHABILITY (that it fails on a deliberately λ-doubled gradient, rather
/// than merely passing on the right one) is pinned by
/// `lambda_nonzero_fd_check_rejects_a_doubled_constraint_term`.
#[test]
fn augmented_gradient_matches_fd_at_large_negative_lambda() {
    let sys = build_sys();
    let w = weight_matrix(&sys, &[0]);
    let h = oneelectron::hcore(&sys.prep);
    let r0 = constrained_run(&sys, &cfg_with_target(2.0), &w, None).unwrap();
    let r = off_stationary_orbitals(&sys, &r0);
    let lam = r.lambda;
    assert!(
        lam.abs() > 1.0,
        "this check is only meaningful at large |λ|; got {lam:+.6}"
    );

    let (ga, gb, fd_a, fd_b) = fd_vs_analytic_gradient(&sys, &h, &r.c_a, &r.c_b, lam, &w, 1.0);
    let num = (&ga - &fd_a)
        .iter()
        .chain((&gb - &fd_b).iter())
        .fold(0.0_f64, |m, &v| m.max(v.abs()));
    let den = ga
        .iter()
        .chain(gb.iter())
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1e-12);
    eprintln!(
        "[λ≠0 FD] λ = {lam:+.6}  max|analytic| = {den:.6e}  max|analytic − FD| = \
         {num:.6e}  rel = {:.3e}",
        num / den
    );
    // The gradient must be BIG enough that agreeing with it is informative.
    assert!(
        den > 1e-4,
        "the augmented gradient is ~0 at this point, so agreement is vacuous: \
         max|g| = {den:.3e}"
    );
    assert!(
        num / den < 1e-5,
        "λ-augmented analytic gradient disagrees with FD of the augmented energy \
         at λ = {lam:+.6}: rel = {:.3e}. A term double-added at strength λ looks \
         exactly like this and is INVISIBLE to the λ = 0 anchor.",
        num / den
    );
}

/// The λ ≠ 0 FD check's reachability: feed it a gradient whose constraint term
/// is DOUBLED (the `epcdft`-class defect) and confirm it is rejected.
///
/// Without this, `augmented_gradient_matches_fd_at_large_negative_lambda` is an
/// assumption — a test never seen to fail.
#[test]
fn lambda_nonzero_fd_check_rejects_a_doubled_constraint_term() {
    let sys = build_sys();
    let w = weight_matrix(&sys, &[0]);
    let h = oneelectron::hcore(&sys.prep);
    let r0 = constrained_run(&sys, &cfg_with_target(2.0), &w, None).unwrap();
    let r = off_stationary_orbitals(&sys, &r0);
    let lam = r.lambda;

    // `lam_scale = 2.0` builds the ANALYTIC gradient with 2λW instead of λW,
    // while the FD still differentiates the correct functional E + λN_C.
    let (ga, gb, fd_a, fd_b) = fd_vs_analytic_gradient(&sys, &h, &r.c_a, &r.c_b, lam, &w, 2.0);
    let num = (&ga - &fd_a)
        .iter()
        .chain((&gb - &fd_b).iter())
        .fold(0.0_f64, |m, &v| m.max(v.abs()));
    let den = ga
        .iter()
        .chain(gb.iter())
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1e-12);
    eprintln!("[λ≠0 FD mutation] doubled λW ⇒ rel = {:.3e}", num / den);
    assert!(
        num / den > 1e-3,
        "the λ ≠ 0 FD check FAILED TO DETECT a doubled constraint term (rel = \
         {:.3e}). It is then inert and proves nothing about λ-proportional \
         defects.",
        num / den
    );
}

/// Deterministic PRNG (no external `rand` dep) — the same generator as
/// `cdft_constrained_stability.rs`/`uhf_newton_smoke.rs`, so the FD checks in
/// all three are directly comparable.
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

/// Rotate a converged run's orbitals OFF the stationary point by a fixed
/// pseudo-random direction, keeping its λ.
///
/// Needed because the occ→virt gradient VANISHES at a converged solution
/// (Brillouin), which makes any gradient comparison there vacuous — the first
/// version of the λ ≠ 0 check hit exactly that and its own non-vacuity guard
/// rejected it. The rotation angle (0.15 rad) is chosen to put the gradient at
/// O(0.1), comfortably above both the FD truncation error and the SCF
/// convergence floor, while staying small enough that the central difference is
/// still in its quadratic regime.
fn off_stationary_orbitals(sys: &Sys, r: &Run) -> Run {
    let n = sys.prep.nbasis();
    let mut rng = Xorshift64(0x5EED_1234_ABCD_0001);
    let mut da = Array2::<f64>::zeros((n - sys.nocc_a, sys.nocc_a));
    let mut db = Array2::<f64>::zeros((n - sys.nocc_b, sys.nocc_b));
    for v in da.iter_mut() {
        *v = rng.next_f64();
    }
    for v in db.iter_mut() {
        *v = rng.next_f64();
    }
    Run {
        c_a: rotate(&r.c_a, &da, sys.nocc_a, 0.15),
        c_b: rotate(&r.c_b, &db, sys.nocc_b, 0.15),
        ..r.clone()
    }
}

/// Analytic augmented gradient (with the constraint term scaled by
/// `lam_scale`) and its central finite difference, both packed as occ→virt
/// blocks. `lam_scale = 1.0` is the correct functional; `2.0` is the
/// double-add mutation.
#[allow(clippy::type_complexity)]
fn fd_vs_analytic_gradient(
    sys: &Sys,
    h: &Array2<f64>,
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    lam: f64,
    w: &Array2<f64>,
    lam_scale: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let n = sys.prep.nbasis();
    let lam_w = (lam_scale * lam) * w;
    let (fa, fb) = ao_focks(sys, h, c_a, c_b, Some(&lam_w));
    let fa_mo = c_a.t().dot(&fa).dot(c_a);
    let fb_mo = c_b.t().dot(&fb).dot(c_b);
    // dE/dκ_ai = 4 F_ai for a real UHF ½-trace energy under the Cayley
    // parameterization used by `rotate`; the factor is irrelevant to the
    // comparison so long as BOTH sides carry it, so the FD below is scaled to
    // match rather than the analytic side being un-scaled.
    let ga = ov_block(&fa_mo, sys.nocc_a, n);
    let gb = ov_block(&fb_mo, sys.nocc_b, n);

    // Central FD of the augmented energy along each of a few random directions,
    // reconstructed into the same occ→virt shape by differencing one element at
    // a time. Full element-wise FD over a 148-dim space is 296 augmented-energy
    // evaluations, each one Fock build — cheap at nbf = 19.
    let mut fd_a = Array2::<f64>::zeros(ga.dim());
    let mut fd_b = Array2::<f64>::zeros(gb.dim());
    let eps = 1e-4_f64;
    for ((ir, i), slot) in fd_a.indexed_iter_mut() {
        let mut dir = Array2::<f64>::zeros(ga.dim());
        dir[(ir, i)] = 1.0;
        let zero_b = Array2::<f64>::zeros(gb.dim());
        let ep = augmented_energy(
            sys,
            h,
            &rotate(c_a, &dir, sys.nocc_a, eps),
            &rotate(c_b, &zero_b, sys.nocc_b, eps),
            lam,
            w,
        );
        let em = augmented_energy(
            sys,
            h,
            &rotate(c_a, &dir, sys.nocc_a, -eps),
            &rotate(c_b, &zero_b, sys.nocc_b, -eps),
            lam,
            w,
        );
        // dE/dε = 2·F_ai for one spin block under this rotation convention.
        *slot = (ep - em) / (2.0 * eps) / 2.0;
    }
    for ((ir, i), slot) in fd_b.indexed_iter_mut() {
        let mut dir = Array2::<f64>::zeros(gb.dim());
        dir[(ir, i)] = 1.0;
        let zero_a = Array2::<f64>::zeros(ga.dim());
        let ep = augmented_energy(
            sys,
            h,
            &rotate(c_a, &zero_a, sys.nocc_a, eps),
            &rotate(c_b, &dir, sys.nocc_b, eps),
            lam,
            w,
        );
        let em = augmented_energy(
            sys,
            h,
            &rotate(c_a, &zero_a, sys.nocc_a, -eps),
            &rotate(c_b, &dir, sys.nocc_b, -eps),
            lam,
            w,
        );
        *slot = (ep - em) / (2.0 * eps) / 2.0;
    }
    (ga, gb, fd_a, fd_b)
}

// ===========================================================================
// PART 3 — PROOF: the FIXED driver, at its own default
// ===========================================================================

/// The lane's config with the fix ENABLED — i.e. at `RhfConfig`'s own default
/// for `cdft_stability_descent`. Everything else matches `hene_cfg`.
fn hene_cfg_fixed() -> RhfConfig {
    RhfConfig {
        cdft_stability_descent: true,
        ..hene_cfg()
    }
}

/// **THE PROOF.** The fixed `solve_cdft_uhf` must reach a state at or below the
/// old −130.40219057 at the integer target, and that state must be stable (or
/// marginal, reported honestly) under the λ-augmented check.
///
/// MEASURED (2026-09-16):
/// ```text
///   before (descent off): E = -130.40219057  λ = -2.753705  λ_min = -3.999187e-2  UNSTABLE (saddle)
///   after  (descent on):  E = -130.42670694  λ = -2.439011  λ_min = -1.319983e-10 MARGINAL
///   ΔE = 0.02451637 Ha = 0.6671 eV lower, at N_C = 2.000000021 (same constraint)
/// ```
///
/// The λ_min moves from −4.0e-2 — seven orders of magnitude above the
/// eigensolver residual, an unambiguous saddle — to −1.3e-10, which is AT the
/// noise floor. **That is reported as MARGINAL, not as stable.** The honest
/// claim is "the saddle is gone and the energy is 0.667 eV lower"; "this is the
/// global minimum of the constrained manifold" is NOT established here and the
/// assertion below does not pretend otherwise.
#[test]
fn the_fixed_driver_reaches_the_lower_state_at_the_integer_target() {
    let sys = build_sys();
    let w = weight_matrix(&sys, &[0]);
    let h = oneelectron::hcore(&sys.prep);

    let before = solve_cdft_uhf(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bs,
        &sys.bounds,
        &RhfConfig {
            constraints: vec![Constraint {
                fragment: vec![0],
                target: 2.0,
                spin: SpinChannel::Total,
            }],
            ..hene_cfg()
        },
    )
    .unwrap();
    let after = solve_cdft_uhf(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bs,
        &sys.bounds,
        &RhfConfig {
            constraints: vec![Constraint {
                fragment: vec![0],
                target: 2.0,
                spin: SpinChannel::Total,
            }],
            ..hene_cfg_fixed()
        },
    )
    .unwrap();

    let (d_b, _, _, v_b) = augmented_stability(
        &sys,
        &h,
        &before.scf.mos_alpha,
        before.scf.mos_beta.as_ref().unwrap(),
        before.lambdas[0],
        &w,
    );
    let (d_a, i_a, r_a, v_a) = augmented_stability(
        &sys,
        &h,
        &after.scf.mos_alpha,
        after.scf.mos_beta.as_ref().unwrap(),
        after.lambdas[0],
        &w,
    );
    eprintln!(
        "[fix] before: E = {:.8}  λ = {:+.6}  N = {:.8}  λ_min = {d_b:+.6e}  [{v_b}]\n\
         [fix] after : E = {:.8}  λ = {:+.6}  N = {:.8}  λ_min = {d_a:+.6e}  [{v_a}]  \
         (iter {i_a:+.3e}, resid {r_a:.1e})\n\
         [fix] ΔE = {:.8} Ha = {:.4} eV LOWER",
        before.scf.energy,
        before.lambdas[0],
        before.populations[0],
        after.scf.energy,
        after.lambdas[0],
        after.populations[0],
        before.scf.energy - after.scf.energy,
        (before.scf.energy - after.scf.energy) * 27.211_386_245_988
    );

    // 1. The pre-fix solution is the known saddle, so the comparison is against
    //    the right baseline and not a drifted one.
    assert_eq!(
        v_b, "UNSTABLE",
        "the descent-OFF path must still reproduce the saddle this fix exists to \
         escape, else the baseline moved: λ_min = {d_b:.6e}"
    );
    assert!((before.scf.energy - (-130.402_190_57)).abs() < 1e-5);

    // 2. The fixed driver is at or below it — by the measured 0.0245 Ha.
    assert!(
        after.scf.energy <= before.scf.energy,
        "the fixed driver must not return a HIGHER state: {:.8} vs {:.8}",
        after.scf.energy,
        before.scf.energy
    );
    assert!(
        before.scf.energy - after.scf.energy > 1e-2,
        "the measured drop was 0.02451637 Ha; a much smaller one means the descent \
         changed: ΔE = {:.6e} Ha",
        before.scf.energy - after.scf.energy
    );

    // 3. It satisfies the SAME constraint, else it is a different problem's
    //    answer rather than a better state B.
    assert!(
        (after.populations[0] - 2.0).abs() < HENE_LAMBDA_TOL,
        "the descended state drifted off the constraint: N = {:.8}",
        after.populations[0]
    );

    // 4. …and it is no longer a saddle. MARGINAL is accepted and UNSTABLE is
    //    not: the claim being pinned is "the negative mode is gone", not "this
    //    is the bottom", and STABLE is not asserted because it was not measured.
    assert!(
        v_a == "MARGINAL" || v_a == "STABLE",
        "the descended state must not still be a saddle; got [{v_a}] at λ_min = \
         {d_a:.6e}. MEASURED was MARGINAL at -1.32e-10 (at the noise floor), so \
         'the bottom' is NOT claimed — only that the -4.0e-2 mode is gone."
    );
}

/// `cdft_stability_descent: false` must reproduce the PREVIOUS behavior exactly,
/// so the fix is a strictly opt-outable change and the audit file
/// `cdft_constrained_stability.rs` — which pins the pre-fix saddle — keeps
/// measuring what it was written about.
///
/// This is the fix's own vacuous-limit anchor: with the knob off, the descent
/// block is skipped ENTIRELY (an early `return`), not merely made a no-op.
#[test]
fn descent_off_reproduces_the_old_saddle() {
    let sys = build_sys();
    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![0],
            target: 2.0,
            spin: SpinChannel::Total,
        }],
        ..hene_cfg()
    };
    let r = solve_cdft_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bs, &sys.bounds, &cfg).unwrap();
    eprintln!(
        "[descent off] E = {:.8}  λ = {:+.6}  N = {:.8}  outer = {}",
        r.scf.energy, r.lambdas[0], r.populations[0], r.outer_iters
    );
    assert!(
        (r.scf.energy - (-130.402_190_57)).abs() < 1e-5,
        "descent OFF must reproduce the pre-fix E = -130.40219057; got {:.8}",
        r.scf.energy
    );
    assert!(
        (r.lambdas[0] - (-2.753_705)).abs() < 1e-3,
        "descent OFF must reproduce the pre-fix λ = -2.753705; got {:+.6}",
        r.lambdas[0]
    );
}

/// **STATE A MUST NOT REGRESS.** State A is a genuine minimum of its own
/// constrained manifold (λ_min = +1.3194), so the descent must LOOK at it, find
/// it STABLE, and change nothing. A fix that perturbed a good diabat while
/// repairing a bad one would be a worse trade than the defect.
///
/// MEASURED: identical E and λ with the descent on and off, and the driver
/// prints `the constrained solution is STABLE (λ_min = +1.3194e0); no descent
/// taken.`
#[test]
fn state_a_is_untouched_by_the_descent() {
    let sys = build_sys();
    let mk = |descent: bool| {
        let cfg = RhfConfig {
            constraints: vec![Constraint {
                fragment: vec![0],
                target: 1.0,
                spin: SpinChannel::Total,
            }],
            cdft_stability_descent: descent,
            ..hene_cfg()
        };
        solve_cdft_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bs, &sys.bounds, &cfg).unwrap()
    };
    let off = mk(false);
    let on = mk(true);
    eprintln!(
        "[state A] descent off: E = {:.10}  λ = {:+.8}\n\
         [state A] descent on : E = {:.10}  λ = {:+.8}  (ΔE = {:.3e})",
        off.scf.energy,
        off.lambdas[0],
        on.scf.energy,
        on.lambdas[0],
        (on.scf.energy - off.scf.energy).abs()
    );
    assert!(
        (on.scf.energy - off.scf.energy).abs() < 1e-10,
        "the descent must leave the STABLE state A untouched: {:.10} vs {:.10}",
        on.scf.energy,
        off.scf.energy
    );
    assert!((on.lambdas[0] - off.lambdas[0]).abs() < 1e-8);
    assert!((on.scf.energy - (-130.361_859_5)).abs() < 1e-5);
}

/// **THE LIMITATION, ASSERTED SO IT CANNOT BE FORGOTTEN.** The fix does NOT
/// reach NWChem's LOW member, and this test exists to keep that on the record
/// rather than let the 0.667 eV improvement read as "solved".
///
/// External reference (NWChem 7.2.2, same Becke coordinate, same integer target
/// N_He = 2.000):
/// ```text
///   NWChem LOW  (default atomic guess)  E = -130.447402218  λ = -2.2808
///   NWChem HIGH (hcore guess + swap)    E = -130.423435597  λ = -2.6662
///   ferric BEFORE this fix              E = -130.40219057   λ = -2.753705
///   ferric AFTER  this fix              E = -130.42670694   λ = -2.439011
/// ```
///
/// So the fix moves ferric 0.6671 eV down, past NWChem's HIGH member (by 0.089
/// eV) and toward — but NOT to — its LOW member, which remains 0.563 eV below.
/// The λ ordering moves the same way: −2.7537 → −2.4390, heading for NWChem's
/// LOW λ = −2.2808.
///
/// **That is consistent with, and corroborated by, the MARGINAL verdict.** The
/// descended state's λ_min = −1.32e-10 sits AT the noise floor, which is
/// precisely the report "no downhill direction is resolvable from here" and NOT
/// "this is the global minimum". A third, lower constrained solution very
/// likely exists and this fix does not find it: the λ-augmented Hessian is a
/// LOCAL object, and a saddle-following descent can only reach what is
/// connected to the current point by a single negative mode.
///
/// TOO-CLEAN CHECK: if this ever starts agreeing with NWChem's LOW member to
/// within a few tenths of a meV, that is a reason to AUDIT before celebrating —
/// nothing in the current construction earns that agreement.
#[test]
fn the_fix_does_not_reach_nwchems_low_member() {
    const NWCHEM_LOW: f64 = -130.447_402_218;
    const NWCHEM_HIGH: f64 = -130.423_435_597;
    let sys = build_sys();
    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![0],
            target: 2.0,
            spin: SpinChannel::Total,
        }],
        ..hene_cfg_fixed()
    };
    let r = solve_cdft_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bs, &sys.bounds, &cfg).unwrap();
    let ev = 27.211_386_245_988;
    eprintln!(
        "[vs NWChem] ferric (fixed) E = {:.8}  λ = {:+.6}\n\
         [vs NWChem]   vs HIGH {NWCHEM_HIGH:.9}: {:+.6} Ha = {:+.4} eV\n\
         [vs NWChem]   vs LOW  {NWCHEM_LOW:.9}: {:+.6} Ha = {:+.4} eV",
        r.scf.energy,
        r.lambdas[0],
        r.scf.energy - NWCHEM_HIGH,
        (r.scf.energy - NWCHEM_HIGH) * ev,
        r.scf.energy - NWCHEM_LOW,
        (r.scf.energy - NWCHEM_LOW) * ev
    );

    // It IS below NWChem's HIGH member — the fix is a real improvement measured
    // against an INDEPENDENT code, not only against ferric's own baseline.
    assert!(
        r.scf.energy < NWCHEM_HIGH,
        "the fixed state was measured 0.089 eV BELOW NWChem's HIGH member; it is \
         now {:.8} vs {NWCHEM_HIGH:.9}",
        r.scf.energy
    );
    // And it is still ABOVE NWChem's LOW member. This assertion is the POINT of
    // the test: it FAILS if ferric ever reaches the low member, which would be
    // a NEW RESULT requiring its own audit rather than a quiet pass.
    assert!(
        r.scf.energy > NWCHEM_LOW,
        "ferric now reaches or passes NWChem's LOW member ({:.8} vs \
         {NWCHEM_LOW:.9}). That is a NEW RESULT, not a pass — the current \
         construction (a single-negative-mode descent from the hcore basin) does \
         not earn it, so AUDIT before believing it.",
        r.scf.energy
    );
    // The measured gap, pinned so a drift in either direction is visible.
    let gap_ev = (r.scf.energy - NWCHEM_LOW) * ev;
    assert!(
        (gap_ev - 0.5631).abs() < 0.05,
        "the measured residual gap to NWChem's LOW member was 0.5631 eV; got \
         {gap_ev:.4} eV"
    );
}

/// **THE DEFAULT PATH IS STILL EXERCISED, despite this file pinning hcore.**
///
/// Every other test here pins `use_sad_guess: false` (see [`hene_cfg`]) because
/// every recorded baseline was taken against the hcore-started solver. That is
/// right for an AUDIT, but it leaves a hole: since
/// `fix/scf-unconstrained-state-selection` made `use_sad_guess` live, MINAO is
/// what the DEFAULT path actually uses, so a file pinned entirely to hcore would
/// stop testing what real callers get. This test covers that hole.
///
/// # What this measures, and a correction to the record
///
/// MEASURED (2026-09-17), MINAO start, integer target N_He = 2.0:
///
/// ```text
///   E = -130.40219085   λ = -2.753704   outer = 10
/// ```
///
/// That is the SAME upper state the hcore path reaches (−130.40219117, λ =
/// −2.753704, 23 outer), agreeing to 3e-7 Ha — and it converges in FEWER outer
/// iterations, not more.
///
/// This CORRECTS a reading of `1a1eeddd`'s note that the MINAO-started
/// constrained loop "stops converging within its 30-iteration cap". That note is
/// accurate for the two INTERMEDIATE targets it names (1.990 and 1.995) and for
/// the `state A (N_He=1)` entry of the guess catalogue above — it is NOT a
/// property of MINAO at the integer target, where MINAO converges fine. The
/// non-convergence is TARGET- and GUESS-specific, not a blanket property of the
/// MINAO start, and the catalogue in
/// [`state_b_energy_is_multi_valued_across_guesses_at_the_integer_target`] shows
/// the same thing: its `SAD` row converges at the integer target in 14 outer
/// iterations.
///
/// Recorded because the broader claim, left uncorrected, would licence widening
/// `max_outer` to "fix" a loop that is not broken in the way the claim suggests.
///
/// # What would make this test fail
///
/// It fails if the default path stops converging here, or if it starts landing
/// on a DIFFERENT state than hcore does. Either is a real change in what users
/// get by default, and neither should be discovered by a baseline drifting
/// silently in some other test.
#[test]
fn the_default_minao_path_converges_to_the_same_state_as_hcore() {
    // The hcore baseline this is compared against, from the guess catalogue
    // above: E = -130.40219117 at the integer target.
    const HCORE_UPPER: f64 = -130.40219117;
    // 3e-6 Ha. The measured hcore-vs-MINAO spread is ~3e-7 (two solvers reaching
    // the same stationary point from different starts, each at its own SCF exit
    // criteria); 10x that is loose enough not to be brittle and still ~4 orders
    // of magnitude tighter than the 0.0245 Ha gap to the OTHER state, which is
    // what this test must never silently accept.
    const TOL: f64 = 3e-6;

    let sys = build_sys();
    let cfg = RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![0],
            target: 2.0,
            spin: SpinChannel::Total,
        }],
        // The ONE place in this file that deliberately does NOT pin the guess:
        // the default path is the whole subject of the test.
        use_sad_guess: true,
        ..hene_cfg()
    };

    let r = solve_cdft_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bs, &sys.bounds, &cfg).expect(
        "the MINAO-started constrained loop must converge at the integer \
             target -- it did (10 outer iters) when this test was written. If it \
             now fails, the DEFAULT path has regressed, which is what this test \
             exists to catch. Do NOT pin this to hcore to make it pass: that \
             would delete the only coverage of the path real callers use.",
    );

    eprintln!(
        "[MINAO start, integer target] E = {:.8}  λ = {:+.6}  N = {:.8}  outer = {}",
        r.scf.energy, r.lambdas[0], r.populations[0], r.outer_iters
    );

    assert!(
        (r.scf.energy - HCORE_UPPER).abs() < TOL,
        "the default (MINAO) path reached {:.8}, but hcore reaches \
         {HCORE_UPPER:.8} (Δ = {:.2e} Ha, tol {TOL:.0e}). The two starts landing \
         on DIFFERENT states is a real finding about what users get by default -- \
         audit it, do not widen this tolerance. NB the other constrained state \
         here is ~0.0245 Ha away, so a Δ of that order means a basin flip.",
        r.scf.energy,
        (r.scf.energy - HCORE_UPPER).abs()
    );
}
