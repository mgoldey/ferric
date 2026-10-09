//! **Issue #225: the HeNe⁺ integer-target guess catalogue, through the real
//! driver, at libint precision 1e-14 AND 1e-20.**
//!
//! HeNe⁺/def2-SVP, R = 2.0 Å, fragment = [He], `SpinChannel::Total`, target
//! N_He = 2 (the over-constrained integer target). Eight starts: the driver's
//! own default and seven explicit orbital guesses, each seeded into EVERY inner
//! solve of the λ loop through `solve_cdft_uhf_seeded` (the same re-seeding the
//! catalogue in `cdft_state_selection.rs` uses). Stability descent is off, so
//! the state reached is the one the λ loop itself lands on.
//!
//! The ERI precision is a process-wide setting, so this file is its own test
//! binary, each test holds `PRECISION_LOCK` for its whole run (cargo runs the
//! tests of one binary as threads of one process), and the default is restored
//! afterwards. Under nextest each test is its own process anyway.
//!
//! What is asserted, at each precision:
//! * every start that converges lands on one of the two documented integer-
//!   target states (`cdft_state_selection.rs`): UPPER E = −130.4021905,
//!   λ = −2.75370 or LOWER E = −130.4267065, λ = −2.43901 (E corrected to the
//!   exact target, 1e-5 Ha / 1e-3 in λ, the bars that file uses);
//! * converging is meeting the constraint: |N − 2| < `cdft_lambda_tol`;
//! * each converged start lands on the state it was measured to reach
//!   (`EXPECTED_LEVEL`, the same at both precisions);
//! * every start converges except those listed in `allowed_failures`, which is
//!   now empty at BOTH precisions (it held `state A (N_He=1)` at 1e-20 until
//!   #282).
//!
//! MEASURED (release, this box):
//!
//! ```text
//!                               1e-14          1e-20
//!   driver default (None)       UPPER  8       UPPER  8
//!   hcore (explicit MOs)        UPPER  8       UPPER  8
//!   SAD                         UPPER 10       UPPER 10
//!   unconstrained UHF (pi)      LOWER 11       LOWER 11
//!   unconstrained UHF (sigma)   UPPER 10       UPPER 10
//!   state A (N_He=1)            LOWER  7       LOWER  7
//!   natural target (1.954484)   LOWER 11       LOWER 11
//!   post-descent (0.8 rad)      LOWER  7       LOWER  7
//!   (state, outer iterations)
//! ```
//!
//! UPPER is E(target) = −130.40218994, LOWER −130.42670699 at both precisions.
//!
//! `state A` used to fail at 1e-20 and was listed in `allowed_failures` so that
//! a fix would fail this test and get recorded. Fixed in #282: the first 23
//! inner solves hit their 400-iteration cap, and `ScalarStepper::trusts`
//! short-circuited on an unconverged main point, so Jacobians of ±310 built
//! from capped DIIS snapshots drove Newton steps for more than half the budget
//! (guard 2's `hf_mismatch` was BYPASSED, not absent — it would have rejected
//! every pair). With the short-circuit closed the start reaches LOWER in 7
//! outer iterations at both precisions, and the 1e-14 column moved 9 → 7 for
//! the same reason.
//!
//! Note the row is now the FASTEST LOWER start in the table. That is not
//! suspicious: it begins from constrained orbitals at the neighbouring target,
//! which is a better guess for λ ≈ −2.44 than hcore or MINAO, and it was only
//! ever slow because the loop spent 23 iterations in a region with no root.
//!
//! Run (release; ~minutes per precision):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-scf \
//!     --test cdft_guess_catalogue_precision -- --ignored --nocapture --test-threads=1
//! ```
//! `FERRIC_CDFT_TRACE=1` adds the per-iteration outer-loop trace.

use std::sync::Mutex;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::cdft::{Constraint, SpinChannel};
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine_pool::set_eri_precision;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::cdft_driver::{solve_cdft_uhf_seeded, CdftResult, CdftSeed};
use ferric_scf::rhf::{build_jk, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::{uhf_internal_stability, StabilityConfig};
use ferric_scf::uhf::solve_uhf_fockmod;
use ferric_scf::uhf_newton::UhfNewtonInputs;
use ndarray::Array2;
use ndarray_linalg::{Eigh, Solve, UPLO};

const R_ANG: f64 = 2.0;
const LAMBDA_TOL: f64 = 1e-5;
const NATURAL_N_HE: f64 = 1.954_484;

/// The two constrained states at the integer target, as measured and pinned in
/// `cdft_state_selection.rs::state_b_energy_is_multi_valued_across_guesses_at_
/// the_integer_target`.
const UPPER: (f64, f64) = (-130.402_190_5, -2.753_70);
const LOWER: (f64, f64) = (-130.426_706_5, -2.439_01);
const TOL_E: f64 = 1e-5;
const TOL_LAMBDA: f64 = 1e-3;

/// The state each start reaches, measured identically at 1e-14 and 1e-20.
const EXPECTED_LEVEL: [(&str, &str); 8] = [
    ("driver default (None)", "UPPER"),
    ("hcore (explicit MOs)", "UPPER"),
    ("SAD", "UPPER"),
    ("unconstrained UHF (pi)", "LOWER"),
    ("unconstrained UHF (sigma)", "UPPER"),
    ("state A (N_He=1)", "LOWER"),
    ("natural target (1.954484)", "LOWER"),
    ("post-descent (0.8 rad)", "LOWER"),
];

static PRECISION_LOCK: Mutex<()> = Mutex::new(());

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
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    Sys {
        mol,
        bs,
        prep,
        bounds,
        ctx: ParallelContext::default(),
        nocc_a: (nelec + two_s) / 2,
        nocc_b: (nelec - two_s) / 2,
    }
}

/// `cdft_state_selection.rs::hene_cfg`, verbatim, at a target.
fn cfg(target: f64) -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        level_shift: 0.5,
        cdft_lambda_tol: LAMBDA_TOL,
        cdft_max_outer: 40,
        cdft_stability_descent: false,
        use_sad_guess: false,
        dft_grid: Some(AtomicGridConfig {
            n_radial: 99,
            n_angular: 302,
            ..Default::default()
        }),
        constraints: vec![Constraint {
            fragment: vec![0],
            target,
            spin: SpinChannel::Total,
        }],
        ..Default::default()
    }
}

fn solve(
    sys: &Sys,
    target: f64,
    guess: Option<(&Array2<f64>, &Array2<f64>)>,
) -> Result<CdftResult, ferric_core::FerricError> {
    solve_cdft_uhf_seeded(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bs,
        &sys.bounds,
        &cfg(target),
        CdftSeed {
            mos: guess,
            lambdas: None,
        },
    )
}

fn mos(r: &CdftResult) -> (Array2<f64>, Array2<f64>) {
    let cb = r.scf.mos_beta.clone().expect("UHF beta MOs");
    (r.scf.mos_alpha.clone(), cb)
}

/// Canonical orthogonalizer (unfiltered: valid only while s_min ≫ 1e-6).
fn orthogonalizer(s: &Array2<f64>) -> Array2<f64> {
    let (vals, vecs) = s.eigh(UPLO::Lower).unwrap();
    assert!(
        vals[0] > 1e-5,
        "near-linear dependence, s_min = {:.3e}",
        vals[0]
    );
    let mut x = vecs;
    for (j, &v) in vals.iter().enumerate() {
        x.column_mut(j).mapv_inplace(|e| e / v.sqrt());
    }
    x
}

fn hcore_mos(sys: &Sys) -> (Array2<f64>, Array2<f64>) {
    let h = oneelectron::hcore(&sys.prep);
    let x = orthogonalizer(&oneelectron::overlap(&sys.prep));
    let (_, cp) = x.t().dot(&h).dot(&x).eigh(UPLO::Lower).unwrap();
    let c = x.dot(&cp);
    (c.clone(), c)
}

/// One Fock build at the SAD density (split evenly by spin), diagonalized.
fn sad_mos(sys: &Sys) -> (Array2<f64>, Array2<f64>) {
    let n = sys.prep.nbasis();
    let h = oneelectron::hcore(&sys.prep);
    let x = orthogonalizer(&oneelectron::overlap(&sys.prep));
    let d_tot = ferric_scf::guess::sad_guess(&sys.mol, &sys.prep, &sys.bs).unwrap();
    let d_spin = 0.5 * &d_tot;
    let (mut j, mut kdum) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
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
    let (mut jd, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
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
    let (_, cp) = x.t().dot(&f).dot(&x).eigh(UPLO::Lower).unwrap();
    let c = x.dot(&cp);
    (c.clone(), c)
}

/// Unconstrained UHF from hcore (²Π) or from MINAO (σ).
///
/// The σ entry needs `use_sad_guess: true` explicitly: the lane config pins
/// it to false, and inheriting that would make the σ entry the ²Π one again.
fn unconstrained_mos(sys: &Sys, minao: bool) -> (Array2<f64>, Array2<f64>) {
    let c = RhfConfig {
        constraints: vec![],
        use_sad_guess: minao,
        ..cfg(2.0)
    };
    let scf =
        solve_uhf_fockmod(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &c, None, None).unwrap();
    let cb = scf.mos_beta.clone().unwrap();
    (scf.mos_alpha, cb)
}

/// Cayley rotation of C by ε·κ_ov.
fn rotate(c: &Array2<f64>, k_ov: &Array2<f64>, nocc: usize, eps: f64) -> Array2<f64> {
    let n = c.nrows();
    let mut kappa = Array2::<f64>::zeros((n, n));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            kappa[(a, i)] = eps * k_ov[(ir, i)];
            kappa[(i, a)] = -eps * k_ov[(ir, i)];
        }
    }
    let eye = Array2::<f64>::eye(n);
    let am = &eye - &(0.5 * &kappa);
    let bm = &eye + &(0.5 * &kappa);
    let mut u = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        u.column_mut(col)
            .assign(&am.solve(&bm.column(col).to_owned()).unwrap());
    }
    c.dot(&u)
}

/// The driver-default integer-target state rotated 0.8 rad along its lowest
/// λ-augmented Hessian eigenvector (the step measured to leave the saddle's
/// DIIS basin). The driver's Fock matrices are already λ-augmented.
fn post_descent_mos(sys: &Sys, upper: &CdftResult) -> (Array2<f64>, Array2<f64>) {
    let (c_a, c_b) = mos(upper);
    let f_a = upper.scf.fock_alpha.clone();
    let f_b = upper.scf.fock_beta.clone().unwrap();
    let f_a_mo = c_a.t().dot(&f_a).dot(&c_a);
    let f_b_mo = c_b.t().dot(&f_b).dot(&c_b);
    let inp = UhfNewtonInputs {
        prep: &sys.prep,
        bounds: &sys.bounds,
        c_a: &c_a,
        c_b: &c_b,
        f_a_mo: &f_a_mo,
        f_b_mo: &f_b_mo,
        nocc_a: sys.nocc_a,
        nocc_b: sys.nocc_b,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: 1e-12,
        ooc_budget: 0,
    };
    let st = uhf_internal_stability(
        &sys.ctx,
        &inp,
        &StabilityConfig {
            conv_thresh: 1e-8,
            max_iter: 100,
            ..Default::default()
        },
    )
    .unwrap();
    let vb = st.eigenvector_beta.clone().unwrap();
    (
        rotate(&c_a, &st.eigenvector_alpha, sys.nocc_a, 0.8),
        rotate(&c_b, &vb, sys.nocc_b, 0.8),
    )
}

/// Run the catalogue at the current precision and check it.
fn catalogue_at(precision: f64, allowed_failures: &[&str]) {
    let _lock = PRECISION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_eri_precision(Some(precision)).unwrap();
    // Restore the default even if an assertion below panics.
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = set_eri_precision(None);
        }
    }
    let _restore = Restore;

    let sys = build_sys();
    eprintln!("\n=== HeNe+ integer-target catalogue, ERI precision {precision:e} ===");

    // Reference orbitals that are themselves constrained solves. A failure
    // here is a failure of the driver on a start the catalogue depends on.
    let upper = solve(&sys, 2.0, None).expect("driver default at the integer target");
    let state_a = solve(&sys, 1.0, None).expect("state A (N_He = 1) from the driver default");
    let natural = solve(&sys, NATURAL_N_HE, None).expect("natural target from the driver default");

    let mut guesses: Vec<(&str, Option<(Array2<f64>, Array2<f64>)>)> = vec![
        ("driver default (None)", None),
        ("hcore (explicit MOs)", Some(hcore_mos(&sys))),
        ("SAD", Some(sad_mos(&sys))),
        (
            "unconstrained UHF (pi)",
            Some(unconstrained_mos(&sys, false)),
        ),
        (
            "unconstrained UHF (sigma)",
            Some(unconstrained_mos(&sys, true)),
        ),
        ("state A (N_He=1)", Some(mos(&state_a))),
        ("natural target (1.954484)", Some(mos(&natural))),
    ];
    guesses.push((
        "post-descent (0.8 rad)",
        Some(post_descent_mos(&sys, &upper)),
    ));

    let mut converged = Vec::new();
    let mut failed = Vec::new();
    for (name, g) in &guesses {
        let r = match g {
            // The driver default was already solved above; re-solving would
            // repeat it bit for bit.
            None => Ok(upper.clone()),
            Some((a, b)) => solve(&sys, 2.0, Some((a, b))),
        };
        match r {
            Ok(r) => {
                let (e, lam, n) = (r.scf.energy, r.lambdas[0], r.populations[0]);
                let e_t = e + lam * (n - 2.0);
                let level = if (e_t - UPPER.0).abs() < TOL_E && (lam - UPPER.1).abs() < TOL_LAMBDA {
                    "UPPER"
                } else if (e_t - LOWER.0).abs() < TOL_E && (lam - LOWER.1).abs() < TOL_LAMBDA {
                    "LOWER"
                } else {
                    "NEW STATE"
                };
                eprintln!(
                    "  {name:<28} E = {e:.8}  E@target = {e_t:.8}  N = {n:.8}  λ = {lam:+.6}  \
                     outer = {:>2}  inner_conv = {}  [{level}]",
                    r.outer_iters, r.scf.converged
                );
                converged.push((*name, level, n, r.scf.converged));
            }
            Err(e) => {
                eprintln!("  {name:<28} DID NOT CONVERGE: {e:?}");
                failed.push(*name);
            }
        }
    }
    eprintln!(
        "[catalogue {precision:e}] {} of {} converged; failed: {failed:?}",
        converged.len(),
        guesses.len()
    );

    for &(name, level, n, inner) in &converged {
        assert!(
            level != "NEW STATE",
            "precision {precision:e}: {name} converged to neither documented integer-target \
             state; the outer loop must change how a state is reached, not which one"
        );
        assert!(
            (n - 2.0).abs() < LAMBDA_TOL,
            "precision {precision:e}: {name} returned N = {n:.10}, off the target"
        );
        assert!(
            inner,
            "precision {precision:e}: {name} returned an unconverged inner SCF"
        );
    }
    for &(name, level, _, _) in &converged {
        let want = EXPECTED_LEVEL
            .iter()
            .find(|(g, _)| *g == name)
            .map(|(_, l)| *l)
            .unwrap_or_else(|| panic!("{name} has no measured level"));
        assert_eq!(
            level, want,
            "precision {precision:e}: {name} was measured to reach the {want} state"
        );
    }
    assert_eq!(
        converged.len() + failed.len(),
        EXPECTED_LEVEL.len(),
        "the catalogue did not run every start"
    );
    assert_eq!(
        failed, allowed_failures,
        "precision {precision:e}: the starts that failed differ from the measured \
         ones. A start that newly converges is progress: record it here."
    );
}

#[test]
#[ignore = "8+3 constrained HeNe+ solves on a 99x302 grid; minutes"]
fn catalogue_converges_at_eri_precision_1e_14() {
    catalogue_at(1e-14, &[]);
}

#[test]
#[ignore = "8+3 constrained HeNe+ solves on a 99x302 grid; minutes"]
fn catalogue_converges_at_eri_precision_1e_20() {
    catalogue_at(1e-20, &[]);
}
