//! The LR (G ≠ 0) derivative pass of [`fit_derivatives`]: BIT-IDENTITY of the
//! chunk-parallel evaluation to the serial pass it replaced.
//!
//! The golden FNV-1a hashes below were recorded from the UNCHANGED serial pass
//! (`f64::to_bits` of every element of the LR orbital / aux-centre / metric
//! force arrays, on a fabricated symmetric `Y` and `Wm` — the pass is linear in
//! them and bit-exactness does not need a converged density). Each case runs
//! the pass at 1 / 2 / 6 rayon threads and at several in-flight chunk counts;
//! all must reproduce the golden hash, with the pass split into many chunks
//! (asserted) so the ordered replay is exercised.

use super::*;
use crate::hcore::{periodic_hcore, PeriodicHcoreConfig};
use crate::rsgdf::{RangeSplit, RsGdfConfig};
use crate::sr_rotation::SrColumnRotation;
use ferric_core::basis;
use ferric_core::mol::Molecule;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

/// FNV-1a over the bits of `a` (row-major).
fn fnv(a: &Array2<f64>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for v in a.iter() {
        h ^= v.to_bits();
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
}

/// A LR-pass test system: the build, its bases and a fabricated `(Y, Wm)`.
struct Sys {
    cell: Cell,
    obs: PreparedBasis,
    aux: PreparedBasis,
    gdf: RsGdf,
    y: Array2<f64>,
    wm: Array2<f64>,
}

fn sys(xyz: &str, lattice: [[f64; 3]; 3], bs: &str, split: bool, rot: SrColumnRotation) -> Sys {
    let cell = Cell::new(Molecule::parse_xyz(xyz, 0, 1).unwrap(), lattice).unwrap();
    let obs = PreparedBasis::new(cell.mol(), &basis::bundled(bs).unwrap()).unwrap();
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let hc = periodic_hcore(&cell, &obs, &PeriodicHcoreConfig::with_omega(0.8)).unwrap();
    let cfg = RsGdfConfig {
        omega: 1.0,
        budget_bytes: Some(1 << 30),
        range_split: split.then(|| RangeSplit::new(1.0)),
        sr_column_rotation: rot,
        ..Default::default()
    };
    let gdf = RsGdf::build_for_gradient(&cell, &obs, &aux, &hc.s, &cfg).unwrap();
    let (n, naux) = (obs.nbasis(), aux.nbasis());
    let mut s = 0x5eed_u64;
    let mut y = Array2::<f64>::zeros((naux, n * n));
    for p in 0..naux {
        for m in 0..n {
            for k in 0..=m {
                let v = lcg(&mut s);
                y[(p, m * n + k)] = v;
                y[(p, k * n + m)] = v;
            }
        }
    }
    let mut wm = Array2::<f64>::zeros((naux, naux));
    for p in 0..naux {
        for q in 0..=p {
            let v = lcg(&mut s);
            wm[(p, q)] = v;
            wm[(q, p)] = v;
        }
    }
    Sys {
        cell,
        obs,
        aux,
        gdf,
        y,
        wm,
    }
}

fn in_pool<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
        .install(f)
}

/// `[orb, aux3, metric]` LR hashes and the chunk count of one pass with at
/// most `cap` chunks in flight.
fn lr_hashes(s: &Sys, budget: usize, cap: usize) -> ([u64; 3], usize) {
    let mut ledger = Ledger::new(budget);
    let d = fit_derivatives_with(
        &s.gdf,
        &s.cell,
        &s.obs,
        &s.aux,
        (&s.y, &s.wm),
        (false, false, DerivRotMutant::None),
        cap,
        &mut ledger,
    )
    .unwrap();
    (
        [fnv(&d.orb_lr), fnv(&d.aux3_lr), fnv(&d.metric_lr)],
        d.n_chunks,
    )
}

/// The mutant switch is process-global: the tests of this module take turns.
static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

const TILT: [[f64; 3]; 3] = [[6.0, 0.0, 0.0], [1.1, 6.4, 0.0], [0.7, 0.9, 6.9]];

/// `(name, split, ledger budget, golden [orb, aux3, metric] hashes)` of the
/// tilted-water STO-3G system (the cc-pVDZ / rotation cases are the end-to-end
/// `tests/pbc_lr_force_bitwise.rs`).
const CASES: [(&str, bool, usize, [u64; 3]); 2] = [
    (
        "water sto-3g unsplit",
        false,
        3 << 20,
        [0x68b7f8223c84e2b3, 0x93ae4f0b6e63b0fc, 0x1761195a8ff029f2],
    ),
    (
        "water sto-3g split",
        true,
        3 << 20,
        [0x0269f7aa218eea25, 0x0953641627319b59, 0x0b0b8b5c88748518],
    ),
];

/// `(threads, in-flight cap)` runs: serial, narrow, and the production cap.
const RUNS: [(usize, usize); 3] = [(1, 1), (2, 3), (6, 12)];

/// The LR pass reproduces the golden bits of the serial pass at every thread
/// count and in-flight cap, with the pass split into many chunks.
#[test]
fn lr_pass_matches_the_serial_goldens_at_any_threads_and_inflight() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    for (name, split, budget, golden) in CASES {
        let s = sys(WATER, TILT, "sto-3g", split, SrColumnRotation::Off);
        for (threads, cap) in RUNS {
            let (h, nc) = in_pool(threads, || lr_hashes(&s, budget, cap));
            assert!(
                nc >= 8,
                "{name}: only {nc} chunks, the window would not bind"
            );
            assert_eq!(h, golden, "{name} at {threads} threads, cap {cap}");
        }
    }
}

/// MUTANT: an unordered reduction (per-thread partials folded by `reduce`)
/// must NOT reproduce the goldens at 6 threads, in the unsplit and the split
/// pass alike — the bitwise comparison above can tell the difference.
#[test]
fn unordered_reduce_mutant_breaks_the_goldens() {
    use crate::rsgdf::split::UNORDERED_LR_REDUCE;
    use std::sync::atomic::Ordering;
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    for (name, split, budget, golden) in CASES {
        let s = sys(WATER, TILT, "sto-3g", split, SrColumnRotation::Off);
        UNORDERED_LR_REDUCE.store(true, Ordering::SeqCst);
        let (h, _) = in_pool(6, || lr_hashes(&s, budget, 12));
        UNORDERED_LR_REDUCE.store(false, Ordering::SeqCst);
        assert_ne!(h, golden, "{name}: the unordered mutant was not detected");
    }
}
