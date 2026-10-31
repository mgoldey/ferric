//! Allocation-level check of a KS job at the tightest budget XC-last admits:
//! `P = T + L/2`, so XC cannot be fully charged. The ledger is arithmetic; the
//! counting allocator measures what is really live. Heap only, so BLAS and
//! libint internals are outside it.
//!
//! Own binary, one test: the allocator is process-wide.

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
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
fn a_ks_job_with_xc_uncharged_peaks_near_its_budget() {
    let mol = Molecule::load_xyz("../../testdata/molecules/alkane_4.xyz").unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let aux_name = "def2-universal-jkfit";
    let aux = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let tensor = aux.nbasis() * prep.nbasis() * prep.nbasis() * 8;
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let main = AtomicGridConfig::default();

    // L from the charge the evaluator itself takes, forced into recompute by a
    // pool below its resident size.
    let nlc = AtomicGridConfig::default_nlc();
    let threads = 2;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let probe = MemoryPool::with_capacity_bytes(tensor);
    install_global(probe.clone());
    let ks = pool.install(|| {
        ferric_dft::ks::KsXc::new_with_omega_budgeted(&mol, &bs, "PBE", &main, &nlc, None, None)
            .unwrap()
    });
    let l = probe.outstanding_bytes();
    assert!(
        l > 0 && ks.batch_pts_for_test().is_some(),
        "must be recompute"
    );
    drop(ks);
    clear_global();

    let p = tensor + l / 2;
    install_global(MemoryPool::with_capacity_bytes(p));
    let c = RhfConfig {
        xc: Some("PBE".into()),
        df_j_aux: Some(aux_name.into()),
        three_index_budget_bytes: p,
        ..Default::default()
    };
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let r = pool
        .install(|| solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &c))
        .expect("the budget holds raw, so the job must run");
    let peak = PEAK.load(Relaxed) - base;
    clear_global();
    assert!(r.converged);
    println!(
        "PEAK heap={peak} T={tensor} L={l} P={p} over_P={:.3}",
        peak as f64 / p as f64
    );
    // Everything the ledger knows about, charged or not. MEASURED 68.1 MB against
    // this 76.0 MB; P is 59.6 MB, so the overshoot is real and bounded by L.
    let bound = tensor + l;
    assert!(peak <= bound, "peak {peak} exceeds {bound} (P={p}, L={l})");
}
