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

/// Counts the live heap bytes allocated BY THE CURRENT THREAD.
/// `SolvePlan::new` is serial, so the plan's whole allocation lands here,
/// while the other test threads of this binary are not counted.
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

struct Case {
    rg: Ragged,
    f_oo: ndarray::Array2<f64>,
    no: usize,
    nv: usize,
}

/// C_n/6-31G, cc-pvdz-ri aux, Coulomb, frozen core = n carbons, assembled
/// on the integral-direct path with production maps.
fn case(xyz: &str, fc: usize, eps: f64, gate: Option<f64>) -> Case {
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
        &RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    let (vvhv, _) = checked_vvhv(&mol, &obs, &obs_bs, &rhf, "test").unwrap();
    let spaces = localized_spaces(&mol, &obs, &rhf, fc, &vvhv).unwrap();
    let dcfg = DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 1e-5,
        ..Default::default()
    };
    let (rg, _, _) =
        assemble_ragged_direct_local(&mol, &obs, &dfbs, op, &spaces, eps, 1.0, gate, &dcfg)
            .unwrap();
    let no = spaces.f_oo.nrows();
    let nv = vvhv.c_vloc.ncols();
    Case {
        rg,
        f_oo: spaces.f_oo,
        no,
        nv,
    }
}

/// The estimate equals the heap the plan holds. `SolvePlan::new` sizes
/// every vector exactly, so the stated factor is 1 (asserted to 1%); the
/// counting allocator is the independent measurement. Gate off and on.
#[test]
fn solve_plan_bytes_for_matches_the_measured_allocation() {
    for gate in [None, Some(0.7)] {
        let c = case("alkane_8.xyz", 8, 1e-4, gate);
        let est = SolvePlan::bytes_for(&c.rg, &c.f_oo);
        let before = live();
        let plan = SolvePlan::new(&c.rg, &c.f_oo);
        let measured = live() - before;
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
        let before = live();
        let plan_s = SolvePlan::with_fvv_threshold(&c.rg, &c.f_oo, 1.5);
        let measured_s = live() - before;
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
        hylleraas_energy(&c.rg, &c.f_oo, bp.plan(), &bp.solve.t),
        hylleraas_energy(&c.rg, &c.f_oo, bi.plan(), &bi.solve.t),
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
