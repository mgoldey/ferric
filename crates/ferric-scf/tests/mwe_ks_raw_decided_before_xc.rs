//! RI-J/K is decided before XC takes any memory, in RHF, UHF and ROHF.
//!
//! # The defect
//!
//! XC used to charge first. Raw then chose `InCore` from the budget alone and
//! was refused by a pool XC had already spent, at
//! `P` in `[max(T, L), L + T)` and `[max(T, R + L), R + L + T)`, with `T` the raw
//! tensor, `R` the resident AO blocks and `L` the XC working set. So raising the
//! budget could turn a running job into a refused one, and since `L` scales with
//! the thread count, the bands moved with it.
//!
//! # Artifact hypothesis
//!
//! If XC commits last, no refusal at `P >= T` and the raw plane's ledger row is
//! the same at every thread count. If the sweep cannot bind, it passes anyway:
//! so the guards below require that the sweep reached both XC modes, and the
//! butane window requires a pool tier that XC-first demonstrably changes.
//!
//! Pure functional only: on a tree where the dressed tensor charges hard, a
//! hybrid has a separate refusal at `[T, 2T)` that has nothing to do with XC.
//!
//! Own binary: these install the process-global pool, and the mutex keeps them
//! apart.

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, solve_rhf_with_hook, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;
use std::sync::{Mutex, MutexGuard};

const AUX: &str = "def2-universal-jkfit";
const THREADS: [usize; 3] = [1, 4, 12];
static SLOT: Mutex<()> = Mutex::new(());

fn slot() -> MutexGuard<'static, ()> {
    let g = SLOT.lock().unwrap_or_else(|e| e.into_inner());
    clear_global();
    g
}

fn in_pool<R: Send>(t: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(t)
        .build()
        .unwrap()
        .install(f)
}

struct Sys {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    tensor: usize,
}

fn sys(xyz: &str, name: &str) -> Sys {
    let mol = Molecule::load_xyz(xyz).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled(name).unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let tensor = aux.nbasis() * prep.nbasis() * prep.nbasis() * 8;
    Sys {
        mol,
        prep,
        bounds,
        tensor,
    }
}

fn water() -> Sys {
    sys("../../testdata/molecules/water.xyz", "cc-pvdz")
}

fn cfg(budget: usize, max_iter: usize) -> RhfConfig {
    RhfConfig {
        xc: Some("PBE".into()),
        df_j_aux: Some(AUX.into()),
        three_index_budget_bytes: budget,
        max_iter,
        ..Default::default()
    }
}

/// `P/T` from below the tensor through past the highest old band, at half-`T`
/// steps across the old bands (each one `T` wide) on water at 1, 4 and 12 threads.
fn ratios() -> Vec<f64> {
    let across = |lo: u32, hi: u32| (2 * lo..=2 * hi).map(|i| f64::from(i) * 0.5);
    let mut r: Vec<f64> = [0.25, 0.5, 1.0, 2.0, 5.0]
        .into_iter()
        .chain(across(10, 20))
        .chain([25.0])
        .chain(across(46, 57))
        .chain([60.0])
        .collect();
    r.dedup();
    r
}
#[derive(Clone, Copy, Debug)]
enum Solver {
    Rhf,
    Uhf,
    Rohf,
}

fn admits(s: &Sys, solver: Solver, c: &RhfConfig) -> Result<(), String> {
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    match solver {
        Solver::Rhf => solve_rhf(&ctx, &s.mol, &s.prep, op, &s.bounds, c),
        Solver::Uhf => solve_uhf(&ctx, &s.mol, &s.prep, &s.bounds, c),
        Solver::Rohf => solve_rohf(&ctx, &s.mol, &s.prep, op, &s.bounds, c),
    }
    .map(|_| ())
    .or_else(|e| {
        // One iteration is all the sweep asks for: UHF and ROHF report that as
        // an error after the planes were admitted, and only a refusal counts.
        let e = e.to_string();
        if e.contains("did not converge") {
            Ok(())
        } else {
            Err(e)
        }
    })
}

fn sweep(solver: Solver) {
    let s = water();
    for t in THREADS {
        let mut admitted_at = None;
        for r in ratios() {
            let p = (r * s.tensor as f64) as usize;
            install_global(MemoryPool::with_capacity_bytes(p));
            let got = in_pool(t, || admits(&s, solver, &cfg(p, 1)));
            clear_global();
            match got {
                Ok(()) => admitted_at = admitted_at.or(Some(r)),
                Err(e) => {
                    assert!(
                        admitted_at.is_none(),
                        "{solver:?} t={t}: ran at P/T={admitted_at:?}, refused at higher {r}: {e}"
                    );
                    assert!(r < 1.0, "{solver:?} t={t}: refused at P/T={r} >= 1: {e}");
                }
            }
        }
        assert!(admitted_at.is_some(), "{solver:?} t={t}: never ran");
    }
}

#[test]
fn raising_the_budget_never_turns_a_running_rhf_job_into_a_refused_one() {
    let _s = slot();
    sweep(Solver::Rhf);
}

#[test]
fn raising_the_budget_never_turns_a_running_uhf_job_into_a_refused_one() {
    let _s = slot();
    sweep(Solver::Uhf);
}

#[test]
fn raising_the_budget_never_turns_a_running_rohf_job_into_a_refused_one() {
    let _s = slot();
    sweep(Solver::Rohf);
}

/// The ledger at the first Fock build: what each plane holds once all are in.
fn ledger(s: &Sys, c: &RhfConfig, pool: &MemoryPool) -> String {
    let mut seen = None;
    let mut hook = |d: &Array2<f64>| {
        seen.get_or_insert_with(|| pool.occupancy_report());
        Ok((0.0, Array2::zeros(d.raw_dim())))
    };
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let _ = solve_rhf_with_hook(&ctx, &s.mol, &s.prep, op, &s.bounds, c, &mut hook).expect("runs");
    seen.expect("hook ran")
}

fn df_rows(report: &str) -> Vec<String> {
    report
        .lines()
        .filter(|l| l.contains("DF 3-index"))
        .map(|l| l.trim().to_string())
        .collect()
}

/// Inertness guard: the sweep's range really does take XC through both modes.
#[test]
fn the_sweep_reaches_recompute_and_resident_xc() {
    let _s = slot();
    let s = water();
    let at = |r: f64| {
        let p = (r * s.tensor as f64) as usize;
        let pool = MemoryPool::with_capacity_bytes(p);
        install_global(pool.clone());
        let rep = in_pool(4, || ledger(&s, &cfg(p, 1), &pool));
        clear_global();
        rep
    };
    let low = at(20.0);
    assert!(low.contains("recompute"), "{low}");
    let high = at(60.0);
    assert!(high.contains("grid AO cache"), "{high}");
    assert!(!df_rows(&high).is_empty(), "{high}");
}

/// Below `T` the raw tier reads the pool. On this grid an XC-first order gives
/// `InCorePacked` at 1 thread and a soft tier at 4 and 12.
#[test]
fn the_raw_tier_does_not_read_the_thread_count() {
    let _s = slot();
    let s = sys("../../testdata/molecules/alkane_4.xyz", "def2-svp");
    let grid = AtomicGridConfig {
        n_radial: 30,
        n_angular: 50,
        prune: None,
    };
    let p = s.tensor * 95 / 100;
    let mut rows = Vec::new();
    for t in THREADS {
        let pool = MemoryPool::with_capacity_bytes(p);
        install_global(pool.clone());
        let mut c = cfg(p, 1);
        c.dft_grid = Some(grid.clone());
        rows.push((t, df_rows(&in_pool(t, || ledger(&s, &c, &pool)))));
        clear_global();
    }
    for (t, r) in &rows {
        assert!(
            r.iter().any(|l| l.contains("packed")),
            "t={t}: the window must be a packed one: {r:?}"
        );
        assert_eq!(*r, rows[0].1, "raw plane moved with t={t}");
    }
}

/// XC's choice must not move an energy: a pool that sends it to recompute beside
/// raw matches an unpooled run, and neither moves with the thread count.
#[test]
fn energy_is_the_same_for_every_xc_mode_and_thread_count() {
    let _s = slot();
    let s = water();
    let e = |p: Option<usize>, t: usize| {
        if let Some(p) = p {
            install_global(MemoryPool::with_capacity_bytes(p));
        }
        let budget = p.unwrap_or(8_000_000_000);
        let c = cfg(budget, 100);
        let r = in_pool(t, || {
            solve_rhf(
                &ParallelContext::default(),
                &s.mol,
                &s.prep,
                Operator::coulomb(),
                &s.bounds,
                &c,
            )
        })
        .unwrap();
        clear_global();
        assert!(r.converged);
        (r.energy, r.iterations)
    };
    let (e0, it0) = e(None, 1);
    let ample = 8_000_000_000;
    let recompute = 20 * s.tensor;
    for (p, t) in [
        (ample, 1),
        (ample, 4),
        (ample, 12),
        (recompute, 1),
        (recompute, 12),
    ] {
        let (e1, it1) = e(Some(p), t);
        assert!((e1 - e0).abs() < 1e-10, "P={p} t={t}: {e1} vs {e0}");
        assert_eq!(it1, it0, "P={p} t={t}");
    }
}
