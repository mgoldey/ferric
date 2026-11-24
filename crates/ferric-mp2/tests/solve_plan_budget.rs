//! The ragged-PCG gather plan ([`SolvePlan`]) is budgeted: its size is
//! known before it is built ([`SolvePlan::bytes_for`]), and a solve whose
//! plan does not fit the budget falls back to the plan-free
//! [`matvec_indexed`] path with a bitwise-identical result.
//!
//! The plan stores one `(u32, u32)` per gather element visited per matvec,
//! which with the pair gate OFF grows like `no³ · overlap` (every pair has
//! ~`2·no` Fock-coupled partners); the indexed path allocates nothing extra.
//!
//! The allocation is MEASURED with a thread-local counting allocator (an
//! independent construction: it sees the heap, not the formula), so the
//! estimate cannot silently drift from what `SolvePlan::new` allocates.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{checked_vvhv, hylleraas_energy, linear_energy, localized_spaces};
use ferric_mp2::lmp2_direct::{assemble_ragged_direct_local, DirectConfig};
use ferric_mp2::ragged::{
    solve_ragged_budgeted, solve_ragged_budgeted_with, solve_resident_bytes, MatvecPath, Ragged,
    RaggedSolve, SolvePlan,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Counts the live heap bytes allocated BY THE CURRENT THREAD. The plan is
/// built on rayon, so [`build_counted`] builds it on a one-thread pool and reads
/// the counter on that pool's thread: the plan's whole allocation lands
/// there, while the other test threads of this binary are not counted.
struct ThreadCounting;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
}

fn bump(delta: isize) {
    let _ = LIVE.try_with(|c| c.set(c.get() + delta));
}

unsafe impl GlobalAlloc for ThreadCounting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            bump(l.size() as isize);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        bump(-(l.size() as isize));
    }
}

#[global_allocator]
static GLOBAL: ThreadCounting = ThreadCounting;

fn live() -> isize {
    LIVE.with(|c| c.get())
}

/// `build()` and the live heap bytes it left behind, run on a one-thread
/// rayon pool so that every allocation of its parallel loops is counted.
fn build_counted<T: Send>(build: impl FnOnce() -> T + Send) -> (T, isize) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let before = live();
        let x = build();
        (x, live() - before)
    })
}

struct Case {
    rg: Ragged,
    f_oo: ndarray::Array2<f64>,
    f_vv: ndarray::Array2<f64>,
    no: usize,
    nv: usize,
}

/// C_n/6-31G, cc-pvdz-ri aux, Coulomb, frozen core = n carbons, assembled
/// on the integral-direct path with the narrow maps of the original
/// measurements here (aux 10, virt 12 Bohr, ao_tail 1e-3).
fn case(xyz: &str, fc: usize, eps: f64, gate: Option<f64>) -> Case {
    let dcfg = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 1e-5,
        ..Default::default()
    };
    case_with(xyz, fc, eps, gate, &dcfg)
}

/// [`case`] on the maps `dcfg`.
fn case_with(xyz: &str, fc: usize, eps: f64, gate: Option<f64>, dcfg: &DirectConfig) -> Case {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig::default().with_energy_conv(1e-10),
    )
    .unwrap();
    let (vvhv, _) = checked_vvhv(&mol, &obs, &obs_bs, &rhf, "test").unwrap();
    let spaces = localized_spaces(&mol, &obs, &rhf, fc, &vvhv).unwrap();
    let (rg, _, _) =
        assemble_ragged_direct_local(&mol, &obs, &dfbs, op, &spaces, eps, 1.0, gate, dcfg).unwrap();
    let no = spaces.f_oo.nrows();
    let nv = vvhv.c_vloc.ncols();
    Case {
        rg,
        f_oo: spaces.f_oo,
        f_vv: spaces.f_vv,
        no,
        nv,
    }
}

/// The estimate equals the heap the plan holds. `SolvePlan::new` sizes
/// every vector exactly, so the stated factor is 1 (asserted to 1%); the
/// counting allocator is the independent measurement. Gate off and on.
#[ignore = "tier slow (>=60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn solve_plan_bytes_for_matches_the_measured_allocation() {
    for gate in [None, Some(0.7)] {
        let c = case("alkane_8.xyz", 8, 1e-4, gate);
        let est = SolvePlan::bytes_for(&c.rg, &c.f_oo);
        let (plan, measured) = build_counted(|| SolvePlan::new(&c.rg, &c.f_oo));
        eprintln!(
            "C8 gate={gate:?}: SolvePlan estimate {est} B, measured {measured} B, \
             {} pairs",
            c.rg.pairs.len()
        );
        assert!(est > 0, "plan-bytes counter is zero");
        let ratio = est as f64 / measured as f64;
        assert!(
            (0.99..=1.01).contains(&ratio),
            "estimate {est} vs measured {measured} (ratio {ratio:.4})"
        );
        drop(plan);
        // Stage D: with every block on the sparse Fvv product the plan also
        // holds each block's retained (row, col) list; the estimate tracks it.
        let est_s = SolvePlan::bytes_for_with_threshold(&c.rg, &c.f_oo, 1.5);
        let (plan_s, measured_s) =
            build_counted(|| SolvePlan::with_fvv_threshold(&c.rg, &c.f_oo, 1.5));
        eprintln!("C8 gate={gate:?}: sparse-Fvv plan estimate {est_s} B, measured {measured_s} B");
        assert_eq!(plan_s.n_fvv_sparse(), c.rg.pairs.len());
        assert!(est_s > est, "the sparse lists are not counted");
        let ratio = est_s as f64 / measured_s as f64;
        assert!(
            (0.99..=1.01).contains(&ratio),
            "sparse-Fvv estimate {est_s} vs measured {measured_s} (ratio {ratio:.4})"
        );
        drop(plan_s);
    }
}

/// A budget smaller than the plan forces the plan-free indexed matvec, an
/// ample one builds the plan, and the two solves are bitwise identical:
/// iterations, final residual, every amplitude and the energy.
#[test]
fn tiny_budget_falls_back_to_indexed_and_is_bitwise() {
    let c = case("alkane_4.xyz", 4, 1e-4, None);
    let (planned, pp) = solve_ragged_budgeted(&c.rg, &c.f_oo, 1e-11, 400, usize::MAX);
    let (indexed, pi) = solve_ragged_budgeted(&c.rg, &c.f_oo, 1e-11, 400, 1);
    assert_eq!(pp.path, MatvecPath::Planned, "ample budget did not plan");
    assert_eq!(
        pi.path,
        MatvecPath::Indexed,
        "tiny budget did not fall back"
    );
    assert!(pp.plan_bytes > 1 && pp.plan_bytes == pi.plan_bytes);
    assert!(planned.converged && indexed.converged);
    assert_eq!(planned.iterations, indexed.iterations);
    assert_eq!(planned.relres.to_bits(), indexed.relres.to_bits());
    assert_eq!(planned.flops_per_matvec, indexed.flops_per_matvec);
    for (a, b) in planned.t.iter().zip(&indexed.t) {
        assert!(
            a.iter()
                .zip(b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "amplitudes differ between the planned and indexed paths"
        );
    }
    let (ea, eb) = (
        linear_energy(&c.rg, &planned.t),
        linear_energy(&c.rg, &indexed.t),
    );
    assert_eq!(ea.to_bits(), eb.to_bits(), "energy differs: {ea} vs {eb}");
    // The stationary energy's extra matvec runs on the operator the solve
    // used (the plan, or the indexed matvec on the fallback): bitwise too.
    let bp = solve_ragged_budgeted_with(&c.rg, &c.f_oo, 1e-11, 400, usize::MAX, 0.0);
    let bi = solve_ragged_budgeted_with(&c.rg, &c.f_oo, 1e-11, 400, 1, 0.0);
    assert!(bp.plan().is_some() && bi.plan().is_none());
    let (ha, hb) = (
        hylleraas_energy(&c.rg, &c.f_oo, bp.plan(), &bp.solve.t).unwrap(),
        hylleraas_energy(&c.rg, &c.f_oo, bi.plan(), &bi.solve.t).unwrap(),
    );
    assert_eq!(
        ha.to_bits(),
        hb.to_bits(),
        "Hylleraas energy differs: {ha} vs {hb}"
    );
}

fn assert_bitwise(c_rg: &Ragged, a: &RaggedSolve, b: &RaggedSolve) {
    assert!(a.converged && b.converged);
    assert_eq!(a.iterations, b.iterations);
    assert_eq!(a.relres.to_bits(), b.relres.to_bits());
    for (x, y) in a.t.iter().zip(&b.t) {
        assert!(
            x.iter()
                .zip(y.iter())
                .all(|(u, v)| u.to_bits() == v.to_bits()),
            "amplitudes differ between the planned and indexed paths"
        );
    }
    let (ea, eb) = (linear_energy(c_rg, &a.t), linear_energy(c_rg, &b.t));
    assert_eq!(ea.to_bits(), eb.to_bits(), "energy differs: {ea} vs {eb}");
}

/// The plan is admitted against the budget LEFT after the solve's resident
/// footprint (`rg` + the PCG vectors), not the whole budget. A budget that
/// holds the plan alone but not plan + resident must fall back to the
/// indexed matvec (bitwise the same energy); a budget holding both plans.
/// Mutation-checked: reverting the gate to `plan_bytes <= budget_bytes`
/// makes the first half plan, and this test fails.
#[test]
fn plan_is_admitted_against_budget_minus_resident() {
    const MAX_IT: usize = 400;
    let c = case("alkane_4.xyz", 4, 1e-4, None);
    let plan = SolvePlan::bytes_for(&c.rg, &c.f_oo);
    let resident = solve_resident_bytes(&c.rg, MAX_IT);
    let amps: usize = c.rg.pairs.iter().map(|pb| pb.j_blk.len()).sum();
    assert!(
        resident >= 5 * 8 * amps + c.rg.heap_bytes(),
        "resident {resident} B misses the PCG vectors or rg"
    );
    // the window that distinguishes the gates must exist
    assert!(plan > 0 && resident > 0);

    let fits_both = plan + resident;
    let (on_plan, d_on) = solve_ragged_budgeted(&c.rg, &c.f_oo, 1e-11, MAX_IT, fits_both);
    assert_eq!(
        d_on.path,
        MatvecPath::Planned,
        "plan + resident fits, yet no plan"
    );
    assert_eq!((d_on.plan_bytes, d_on.resident_bytes), (plan, resident));

    let plan_only = fits_both - 1; // >= plan, < plan + resident
    assert!(plan_only >= plan);
    let (off_plan, d_off) = solve_ragged_budgeted(&c.rg, &c.f_oo, 1e-11, MAX_IT, plan_only);
    assert_eq!(
        d_off.path,
        MatvecPath::Indexed,
        "budget {plan_only} B holds the plan ({plan} B) but not plan + resident \
         ({fits_both} B); the plan must not be built"
    );
    assert_eq!((d_off.plan_bytes, d_off.resident_bytes), (plan, resident));
    assert_bitwise(&c.rg, &on_plan, &off_plan);
}

/// A plan built for a STRICT subset of the blocks (a share, the other
/// blocks hollow as on a real rank) gives every owned output block bit for
/// bit what the unshared plan and the indexed matvec give it, with the same
/// flop count as the shared indexed matvec, and leaves the other blocks
/// empty. Its size estimate is again its measured allocation, and smaller
/// than the whole plan's. Dense Fvv and every block on the sparse Fvv.
#[test]
fn plan_over_owned_blocks_is_bitwise_on_those_blocks() {
    use ferric_core::parallel::ParallelContext;
    use ferric_mp2::ragged::{
        apply_pattern, matvec_indexed, matvec_indexed_into, matvec_planned, RaggedShare,
    };
    let c = case("alkane_4.xyz", 4, 1e-4, None);
    let n = c.rg.pairs.len();
    let owned: Vec<bool> = (0..n).map(|p| p % 3 != 1).collect();
    assert!(owned.iter().any(|&o| o) && owned.iter().any(|&o| !o));
    let mut shared = c.rg.clone();
    for (pb, &own) in shared.pairs.iter_mut().zip(&owned) {
        if !own {
            for a in [&mut pb.fvv_aa, &mut pb.fvv_bb, &mut pb.j_blk, &mut pb.denom] {
                *a = ndarray::Array2::zeros((0, 0));
            }
        }
    }
    shared.share = Some(RaggedShare {
        ctx: ParallelContext::for_rank(0, 1),
        owned: owned.clone(),
    });
    let mut seed = 11u64;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    // whole on every block, as the PCG's search direction is
    let t: Vec<ndarray::Array2<f64>> =
        c.rg.pairs
            .iter()
            .map(|pb| {
                let mut x = ndarray::Array2::from_shape_fn((pb.da.len(), pb.db.len()), |_| rnd());
                apply_pattern(&mut x, &pb.pat, pb.db.len());
                x
            })
            .collect();
    let empty = || -> Vec<ndarray::Array2<f64>> {
        (0..n).map(|_| ndarray::Array2::zeros((0, 0))).collect()
    };
    let mut f_idx = 0u64;
    let mut idx_shared = empty();
    matvec_indexed_into(&shared, &c.f_oo, &t, &mut idx_shared, &mut f_idx);
    let mut f0 = 0u64;
    let want_dense = matvec_indexed(&c.rg, &c.f_oo, &t, &mut f0);
    for rho in [0.0, 1.5] {
        let whole = SolvePlan::with_fvv_threshold(&c.rg, &c.f_oo, rho);
        let mut want = empty();
        matvec_planned(&c.rg, &whole, &t, &mut want, &mut f0);
        let est = SolvePlan::bytes_for_with_threshold(&shared, &c.f_oo, rho);
        let (plan, measured) =
            build_counted(|| SolvePlan::with_fvv_threshold(&shared, &c.f_oo, rho));
        assert_eq!(est as isize, measured, "rho {rho}: estimate vs allocation");
        assert!(est < SolvePlan::bytes_for_with_threshold(&c.rg, &c.f_oo, rho));
        let (mut got, mut fl) = (empty(), 0u64);
        matvec_planned(&shared, &plan, &t, &mut got, &mut fl);
        if rho == 0.0 {
            assert_eq!(
                fl, f_idx,
                "flop count differs from the shared indexed matvec"
            );
        }
        for p in 0..n {
            if owned[p] {
                let same = |a: &ndarray::Array2<f64>| {
                    a.dim() == got[p].dim()
                        && a.iter()
                            .zip(&got[p])
                            .all(|(u, v)| u.to_bits() == v.to_bits())
                };
                assert!(
                    same(&want[p]),
                    "rho {rho}: block {p} differs from the whole plan"
                );
                if rho == 0.0 {
                    assert!(same(&want_dense[p]), "block {p} differs from indexed");
                    assert!(
                        same(&idx_shared[p]),
                        "block {p} differs from shared indexed"
                    );
                }
            } else {
                assert_eq!(got[p].len(), 0, "another rank's block {p} was computed");
            }
        }
    }
}

/// MEASUREMENT (estimate only, nothing is built, no timing): the plan bytes
/// at C4/C8, gate off/on, across ε. Run with `--ignored --nocapture`.
#[test]
#[ignore = "measurement report; run explicitly"]
fn solve_plan_bytes_report() {
    for (xyz, fc) in [("alkane_4.xyz", 4usize), ("alkane_8.xyz", 8)] {
        for eps in [0.0, 1e-5, 1e-4, 1e-3] {
            for gate in [None, Some(0.7)] {
                if eps == 0.0 && gate.is_some() {
                    continue;
                }
                let c = case(xyz, fc, eps, gate);
                let amps: usize = c.rg.pairs.iter().map(|pb| pb.j_blk.len()).sum();
                println!(
                    "PLAN {xyz} eps={eps:e} gate={} no={} nv={} pairs={} amps={} \
                     plan_bytes={} plan/amps8={:.2}",
                    u8::from(gate.is_some()),
                    c.no,
                    c.nv,
                    c.rg.pairs.len(),
                    amps,
                    SolvePlan::bytes_for(&c.rg, &c.f_oo),
                    SolvePlan::bytes_for(&c.rg, &c.f_oo) as f64 / (8 * amps) as f64
                );
            }
        }
    }
}

/// MEASUREMENT (timing, single runs): the planned and the indexed matvec on
/// the same solve, through the driver's tolerance policy (adaptive default)
/// and its Hylleraas energy; production maps, eps = 1e-4, gate off.
/// `FERRIC_PLAN_TIMING_ALKANES` lists the chain lengths (default "12").
/// Run with `--ignored --nocapture --test-threads=1`.
#[test]
#[ignore = "timing report; run explicitly"]
fn planned_vs_indexed_solve_timing_report() {
    use ferric_mp2::lmp2_amplitude::{solve_under_rtol_policy, CgRtolPolicy};
    let ns: Vec<usize> = std::env::var("FERRIC_PLAN_TIMING_ALKANES")
        .unwrap_or_else(|_| "12".into())
        .split(',')
        .map(|s| s.trim().parse().expect("chain length"))
        .collect();
    for n in ns {
        let dcfg = DirectConfig::production();
        let c = case_with(&format!("alkane_{n}.xyz"), n, 1e-4, None, &dcfg);
        let mut energies = Vec::new();
        for (label, budget) in [("planned", usize::MAX), ("indexed", 1)] {
            let t0 = std::time::Instant::now();
            let s = solve_under_rtol_policy(
                &c.rg,
                &c.f_oo,
                &c.f_vv,
                CgRtolPolicy::from_config(None),
                400,
                budget,
                0.0,
            )
            .unwrap();
            let secs = t0.elapsed().as_secs_f64();
            let d = s.budgeted.decision;
            let iters = s.budgeted.solve.iterations;
            println!(
                "TIMING alkane_{n} {label}: path={} iters={iters} solve+energy {secs:.3} s \
                 ({:.4} s/matvec) plan_bytes={} resident_bytes={} pairs={} e={:.12}",
                d.path.as_str(),
                secs / (iters + 1) as f64,
                d.plan_bytes,
                d.resident_bytes,
                c.rg.pairs.len(),
                s.e_hylleraas
            );
            energies.push(s.e_hylleraas);
        }
        assert_eq!(energies[0].to_bits(), energies[1].to_bits());
        // The two costs separately: building the plan, and one matvec each
        // way (one warm-up, then the mean of `reps`), on a whole vector.
        let t0 = std::time::Instant::now();
        let plan = SolvePlan::new(&c.rg, &c.f_oo);
        let t_build = t0.elapsed().as_secs_f64();
        let x: Vec<ndarray::Array2<f64>> = c.rg.pairs.iter().map(|pb| pb.j_blk.clone()).collect();
        let mut out: Vec<ndarray::Array2<f64>> =
            c.rg.pairs
                .iter()
                .map(|_| ndarray::Array2::zeros((0, 0)))
                .collect();
        let reps = 5;
        let mut fl = 0u64;
        let mut per = |planned: bool| {
            let mut run = |out: &mut [ndarray::Array2<f64>]| {
                if planned {
                    ferric_mp2::ragged::matvec_planned(&c.rg, &plan, &x, out, &mut fl);
                } else {
                    ferric_mp2::ragged::matvec_indexed_into(&c.rg, &c.f_oo, &x, out, &mut fl);
                }
            };
            run(&mut out);
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                run(&mut out);
            }
            t0.elapsed().as_secs_f64() / reps as f64
        };
        let (mi, mp) = (per(false), per(true));
        println!(
            "MATVEC alkane_{n}: plan build {t_build:.3} s; per matvec indexed {mi:.4} s, \
             planned {mp:.4} s (indexed/planned {:.2}); break-even after {:.1} matvecs",
            mi / mp,
            t_build / (mi - mp)
        );
    }
}
