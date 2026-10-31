//! Allocation-level check of the spilled DF J/K path against its budget: a
//! counting allocator measures what the plan arithmetic only models. Heap only,
//! so BLAS and libint internals are outside it; those are small and not budgeted.
//! Nothing is subtracted: blocks are allocated on first use, so an untouched one
//! is not there to count.
//!
//! Own binary, one test: the allocator is process-wide.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_scf::df_j::DfJ;
use ferric_scf::df_k::DfK;
use ferric_scf::fock::{JBuilder, KBuilder};
use ndarray::Array2;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn add(n: usize) {
    let now = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(now, Relaxed);
}

// SAFETY: forwards every call to `System` unchanged and only keeps counters.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        add(l.size());
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        // Counted in full, though the kernel maps these pages lazily, so this
        // over-states RSS for a block that is never touched.
        add(l.size());
        System.alloc_zeroed(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        add(n);
        LIVE.fetch_sub(l.size(), Relaxed);
        System.realloc(p, l, n)
    }
}

#[global_allocator]
static A: Counting = Counting;

#[test]
fn spilled_dressing_and_one_fock_build_stay_inside_the_budget() {
    let mol = Molecule::load_xyz("../../testdata/molecules/alkane_12.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
    let aux = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let (naux, nao) = (aux.nbasis(), obs.nbasis());
    let needed = naux * nao * nao * 8;
    let budget = (needed as f64 * 0.45) as usize;

    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let mut raw = ThreeIndexSource::build(op, &obs, &aux, budget).unwrap();
    assert!(!raw.is_incore(), "raw must not be unpacked in core");
    let dfk = DfK::from_full_raw(&mut raw, &obs, &aux, op, budget, Some(&ctx));
    let mut dfk = dfk.unwrap();
    assert_eq!(
        raw.scratch_bytes(),
        0,
        "dressing must leave raw's block cold"
    );
    let mut dfj = DfJ::from_source(raw, op, &aux, budget, Some(&ctx)).unwrap();
    let d = Array2::from_shape_fn((nao, nao), |(i, j)| {
        0.5 / ((i as f64 - j as f64).abs() + 1.0)
    });
    let mut f = Array2::<f64>::zeros((nao, nao));
    // Twice, in the SCF's order, so a block left resident by one K pass is still
    // there for the J stream of the next build.
    for _ in 0..2 {
        dfj.build(&d, &mut f).unwrap();
        dfk.build(&d, &mut f).unwrap();
    }
    let peak = PEAK.load(Relaxed) - base;
    eprintln!(
        "naux={naux} nao={nao} budget={budget} peak={peak} ({:.3} of budget)",
        peak as f64 / budget as f64
    );
    assert!(
        peak as f64 <= budget as f64 * 1.05,
        "peak {peak} over budget {budget}"
    );
    assert!(
        peak as f64 >= budget as f64 * 0.25,
        "peak {peak} is far under {budget}: this budget is not what bound it"
    );
    drop((dfj, dfk));

    // A range-separated hybrid holds two K instances, each sized to the whole
    // budget. They run in sequence, so together they must still fit.
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let mut sr = DfK::new(Operator::erfc(0.3), &obs, &aux, budget).unwrap();
    let mut lr = DfK::new(Operator::erf(0.3), &obs, &aux, budget).unwrap();
    for _ in 0..2 {
        sr.build(&d, &mut f).unwrap();
        lr.build(&d, &mut f).unwrap();
    }
    let peak = PEAK.load(Relaxed) - base;
    eprintln!(
        "RSH pair: peak={peak} ({:.3} of budget)",
        peak as f64 / budget as f64
    );
    assert!(
        peak as f64 <= budget as f64 * 1.05,
        "RSH peak {peak} over budget {budget}"
    );
}
