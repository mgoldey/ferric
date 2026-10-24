//! The peaks `canonical_mp2` and `mp3_energy` charge are checked against the
//! Rust heap itself, through a counting global allocator.
//!
//! `mwe_canonical_mp3_charge_the_pool.rs` pins the charge to a formula, but
//! that formula and the crate's are both derived by reading the same code, so
//! a misread (a copy `einsum!` makes, a buffer alive one stage longer than it
//! looks) would sit in both. This file is the independent construction: it
//! measures the live-heap high-water of each call and requires the charge to
//! lie within a fixed window below it.
//!
//! # The window, derived
//!
//! `measured - charged` must be in `[0, SLACK]`:
//!
//! * `>= 0`: the charge never claims more than the heap actually held.
//! * `<= SLACK = 16 KiB`: what the peak formulas deliberately exclude. The
//!   largest item is `matrixmultiply`'s packing buffer for the k-blocked
//!   (128-wide) scalar `einsum!` contractions, `128·(8+4)` f64 = 12 288 B
//!   (mr=8, nr=4), which is live while MP3 sits at its amplitude-stage peak.
//!   The rest is O(n) bookkeeping (labels, shell-pair lists, `eo`/`ev`),
//!   measured at ~0.2-0.6 KiB.
//!
//! The fixture is water/cc-pVDZ + cc-pVDZ-RI, where every counted term is
//! larger than `SLACK` (smallest: MP3's einsum copy of `b_vo`, A·no·nv = 84·5·19
//! f64 = 63 840 B; canonical's `c_occ + c_vir`, 24·24 f64 = 4 608 B, is the
//! one exception and is pinned exactly by the pool test instead), so dropping
//! or double-counting any one of them leaves the window.
//!
//! Measured on a ONE-thread rayon pool: per-worker scratch is excluded from
//! the formulas by design (a hard gate must not read the thread count), so
//! more workers would widen the gap for reasons unrelated to the charge.
//!
//! Keep this file to ONE test. The counting allocator sees every thread in the
//! process, so a second test running alongside would allocate inside the
//! measured window and make it flaky.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

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

/// Live-heap high-water of `f`, above the heap live when it starts.
fn heap_peak<T>(f: impl FnOnce() -> T) -> usize {
    let base = LIVE.load(SeqCst);
    HIGH.store(base, SeqCst);
    let out = f();
    let peak = HIGH.load(SeqCst) - base;
    drop(out);
    peak
}

/// What the call charges the pool (its high-water under an ample pool).
fn charged<T>(f: impl FnOnce() -> T) -> usize {
    let pool = MemoryPool::with_capacity_bytes(AMPLE);
    install_global(pool.clone());
    let out = f();
    clear_global();
    drop(out);
    pool.peak_bytes()
}

fn assert_in_window(what: &str, measured: usize, charge: usize) {
    assert!(
        charge <= measured && measured - charge <= SLACK,
        "{what}: heap high-water {measured} B vs charge {charge} B \
         (gap {} B, allowed [0, {SLACK}])",
        measured as i64 - charge as i64
    );
}

#[test]
fn canonical_and_mp3_charges_sit_just_under_the_measured_heap_peak() {
    std::env::set_var("FERRIC_MEM_BUDGET_GB", "64");
    std::env::remove_var("FERRIC_OOC_BUDGET_GB");
    std::env::remove_var("FERRIC_ERI3_BUDGET_GB");
    clear_global();

    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz");
    let obs = PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz").expect("obs"))
        .expect("prepared obs");
    let dfbs = PreparedBasis::new(
        &mol,
        &ferric_core::basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri"),
    )
    .expect("prepared dfbs");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig::default(),
    )
    .expect("rhf");

    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");
    let canonical = || ferric_mp2::canonical::canonical_mp2(&mol, &obs, op, &rhf, 0).unwrap();
    let mp3 = || ferric_mp2::mp3::mp3_energy(&mol, &obs, &dfbs, op, &rhf, 0, None).unwrap();

    let measured = one.install(|| heap_peak(canonical));
    let charge = one.install(|| charged(canonical));
    assert_in_window("canonical MP2", measured, charge);

    let measured = one.install(|| heap_peak(mp3));
    let charge = one.install(|| charged(mp3));
    assert_in_window("MP3", measured, charge);
}
