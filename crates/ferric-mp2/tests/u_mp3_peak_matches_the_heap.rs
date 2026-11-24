//! The peak `u_mp3` pre-flights and charges is checked against the Rust heap,
//! through a counting global allocator.
//!
//! `u_mp3_peak_bytes` is derived by reading the driver, so a misread (a copy
//! a permutation makes, a block alive one stage longer than it looks) would
//! sit in both the formula and any re-derivation of it. The live-heap
//! high-water of the call is the independent construction.
//!
//! # The window, derived
//!
//! `measured − charged` must lie in `[0, SLACK]`:
//!
//! * `>= 0`: the charge never claims more than the heap held.
//! * `<= SLACK = 16 KiB`: what the formula deliberately excludes, O(n)
//!   bookkeeping (orbital-energy vectors, labels, shell lists, the engine
//!   pool) and, on a ONE-thread rayon pool, one worker's transform scratch
//!   (`d1·nbas` per aux row, ≤ 24·5 f64 here). Every counted term is larger:
//!   the smallest held block is OH's ββ OVOV, 4²·19² f64 = 46 KiB.
//!
//! The fixture is OH ²Π UHF / cc-pVDZ (+ cc-pVDZ-RI), RI and exact.
//!
//! Measured 2026-11-14: gap 466 B (RI) and 468 B (exact). Mutation: dropping
//! one doubles block from the ring stage of `amplitude_stage_elems`
//! (`5 * s` → `4 * s`) fails here with a gap of 102 066 B.
//!
//! Keep this file to ONE test: the counting allocator sees every thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::u_mp3::{u_mp3, UMp3Config, UMp3Eri};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static HIGH: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let now = LIVE.fetch_add(by, SeqCst) + by;
    HIGH.fetch_max(now, SeqCst);
}

// SAFETY: every method forwards to `System` unchanged and only updates two
// atomics afterwards; no allocation happens inside the allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            grew(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        if !p.is_null() {
            grew(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        LIVE.fetch_sub(l.size(), SeqCst);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = System.realloc(p, l, new);
        if !q.is_null() {
            if new >= l.size() {
                grew(new - l.size());
            } else {
                LIVE.fetch_sub(l.size() - new, SeqCst);
            }
        }
        q
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

const SLACK: usize = 16 * 1024;
const AMPLE: usize = 64_000_000_000;

fn heap_peak<T>(f: impl FnOnce() -> T) -> usize {
    let base = LIVE.load(SeqCst);
    HIGH.store(base, SeqCst);
    let out = f();
    let peak = HIGH.load(SeqCst) - base;
    drop(out);
    peak
}

fn charged<T>(f: impl FnOnce() -> T) -> usize {
    let pool = MemoryPool::with_capacity_bytes(AMPLE);
    install_global(pool.clone());
    let out = f();
    clear_global();
    drop(out);
    pool.peak_bytes()
}

#[test]
fn u_mp3_charge_sits_just_under_the_measured_heap_peak() {
    std::env::set_var("FERRIC_MEM_BUDGET_GB", "64");
    std::env::remove_var("FERRIC_OOC_BUDGET_GB");
    std::env::remove_var("FERRIC_ERI3_BUDGET_GB");
    clear_global();

    let mol = Molecule::load_xyz_with_charge("../../testdata/molecules/validation/oh.xyz", 0, 2)
        .expect("oh.xyz");
    let obs = PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux =
        PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let uhf = solve_uhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        &bounds,
        &RhfConfig::default(),
    )
    .expect("uhf");

    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");
    for (label, eri) in [("RI", UMp3Eri::Ri { aux: &aux }), ("exact", UMp3Eri::Exact)] {
        let call = || u_mp3(&mol, &obs, op, &uhf, eri, &UMp3Config::default()).unwrap();
        let measured = one.install(|| heap_peak(call));
        let charge = one.install(|| charged(call));
        eprintln!(
            "U-MP3 {label}: heap high-water {measured} B, charge {charge} B, gap {} B",
            measured as i64 - charge as i64
        );
        assert!(
            charge <= measured && measured - charge <= SLACK,
            "U-MP3 {label}: heap high-water {measured} B vs charge {charge} B \
             (gap {} B, allowed [0, {SLACK}])",
            measured as i64 - charge as i64
        );
    }
}
