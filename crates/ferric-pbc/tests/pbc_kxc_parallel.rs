//! The k-point XC build ([`KPeriodicXc`], [`KPeriodicXcPolarized`]) runs its
//! grid chunks in parallel and folds their results in ASCENDING chunk order
//! (issue #356, Performance P2). Each chunk's `E_xc` and `V_xc(k)` are summed
//! from zero, so the reduction order is fixed by the chunking alone and the
//! result must be BIT-IDENTICAL to the serial fold (no rayon at all) at every
//! thread count.
//!
//! What can go wrong, and what catches it:
//!
//! * a reduction whose association follows the thread schedule (rayon
//!   `reduce` / `sum`, a per-thread partial sum, completion-order folding):
//!   the 1 / 2 / 4-thread results differ from the serial fold in the last
//!   bits (`*_is_bit_identical_*`; mutation-checked by swapping the ordered
//!   fold for a rayon `reduce`, which fails these tests);
//! * a phase clock that is never wired, or a density integration that
//!   leaks into the build clocks (`build_timings_*`).
//!
//! The densities are random Hermitian positive semi-definite `D(k)` (complex,
//! no symmetry between `k` and `−k`), on a triclinic four-hydrogen cell with
//! an s+p basis, a `1 × 2 × 3` mesh and a grid of ~17 chunks, so every chunk
//! boundary and every k contributes.

mod common;

use common::*;
use ferric_pbc::dft::{PeriodicGrid, PeriodicGridConfig, PeriodicXcConfig};
use ferric_pbc::kuks::KPeriodicXcPolarized;
use ferric_pbc::kuscf::KPointXcPolarized;
use ferric_pbc::{KPeriodicXc, KPointMesh, KPointXc};
use ndarray::Array2;
use num_complex::Complex64;

type Mats = Vec<Array2<Complex64>>;

fn grid_cfg() -> PeriodicGridConfig {
    PeriodicGridConfig {
        n_radial: 40,
        n_angular: 50,
        ..Default::default()
    }
}

/// `(cell, mesh, grid)` of the fixture.
fn fixture() -> (ferric_pbc::Cell, KPointMesh, PeriodicGrid) {
    let cell = triclinic_cell();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 2, 3]).unwrap();
    let grid = PeriodicGrid::build(&cell, &grid_cfg()).unwrap();
    (cell, mesh, grid)
}

/// Deterministic pseudo-random Hermitian PSD `D(k) = s A Aᴴ`.
fn densities(nk: usize, n: usize, seed: u64, s: f64) -> Mats {
    let mut x = seed;
    let mut next = || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    (0..nk)
        .map(|_| {
            let a = Array2::from_shape_fn((n, n), |_| Complex64::new(next(), next()));
            let ah = a.t().mapv(|z| z.conj());
            a.dot(&ah).mapv(|z| z * s)
        })
        .collect()
}

fn bits(e: f64, vs: &[&Mats]) -> (u64, Vec<u64>) {
    let v = vs
        .iter()
        .flat_map(|m| m.iter())
        .flat_map(|m| m.iter())
        .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
        .collect();
    (e.to_bits(), v)
}

fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

/// The fixture's XC object for `functional`, and its `nk`.
fn closed(functional: &str) -> (KPeriodicXc, usize) {
    let (cell, mesh, grid) = fixture();
    let xc = KPeriodicXc::new(
        &cell,
        &sp_basis_h(),
        functional,
        &grid,
        &mesh,
        &PeriodicXcConfig::default(),
    )
    .unwrap();
    (xc, mesh.nk())
}

#[test]
fn closed_shell_build_is_bit_identical_to_the_serial_fold_at_any_thread_count() {
    for functional in ["LDA", "PBE", "SCAN"] {
        let (mut xc, nk) = closed(functional);
        let chunks = xc.xc_timings().counter("k XC chunks").unwrap();
        assert!(chunks >= 8, "{chunks} chunks: too few to exercise the fold");
        let d = densities(nk, xc.nbasis(), 7, 0.05);
        xc.set_serial_reduction(true);
        let (e0, v0) = xc.build(&d).unwrap();
        let reference = bits(e0, &[&v0]);
        xc.set_serial_reduction(false);
        for threads in [1, 2, 4] {
            let (e, v) = in_pool(threads, || xc.build(&d).unwrap());
            assert_eq!(
                bits(e, &[&v]),
                reference,
                "{functional}: {threads} threads vs serial (E {e:.17e} vs {e0:.17e})"
            );
        }
        assert!(e0.is_finite() && e0 < 0.0, "{functional}: E_xc {e0}");
    }
}

#[test]
fn open_shell_build_is_bit_identical_to_the_serial_fold_at_any_thread_count() {
    for functional in ["PBE", "SCAN"] {
        let (mut inner, nk) = closed(functional);
        let n = inner.nbasis();
        let da = densities(nk, n, 11, 0.03);
        let db = densities(nk, n, 13, 0.02);
        inner.set_serial_reduction(true);
        let mut xc = KPeriodicXcPolarized::from_closed(inner).unwrap();
        let (e0, a0, b0) = xc.build_polarized(&da, &db).unwrap();
        let reference = bits(e0, &[&a0, &b0]);
        let (inner, _) = closed(functional);
        let mut xc = KPeriodicXcPolarized::from_closed(inner).unwrap();
        for threads in [1, 2, 4] {
            let (e, a, b) = in_pool(threads, || xc.build_polarized(&da, &db).unwrap());
            assert_eq!(
                bits(e, &[&a, &b]),
                reference,
                "{functional}: {threads} threads vs serial (E {e:.17e} vs {e0:.17e})"
            );
        }
    }
}

#[test]
fn build_timings_count_every_build_and_no_density_integration() {
    let (mut xc, nk) = closed("SCAN");
    let d = densities(nk, xc.nbasis(), 3, 0.05);
    xc.integrate_density(&d).unwrap();
    let t = xc.xc_timings();
    assert!(t.stage("k XC AO cache").is_some_and(|s| s.calls == 1));
    assert_eq!(t.stage("k XC build").map(|s| s.calls), Some(0));
    assert!(
        t.substages.is_empty(),
        "integrate_density fed the build clocks: {:?}",
        t.substages
    );
    let e1 = xc.build(&d).unwrap().0;
    let e2 = xc.build(&d).unwrap().0;
    assert_eq!(
        e1.to_bits(),
        e2.to_bits(),
        "timed builds must not move a bit"
    );
    let t = xc.xc_timings();
    assert_eq!(t.stage("k XC build").map(|s| s.calls), Some(2));
    for phase in [
        "density GEMM",
        "density contraction",
        "tau",
        "libxc kernel",
        "Vxc scale",
        "Vxc GEMM",
        "Vxc add",
    ] {
        let name = format!("k XC build/{phase}");
        let s = t
            .substage(&name)
            .unwrap_or_else(|| panic!("{name} missing: {:?}", t.substages));
        assert!(s.calls > 0 && s.wall_s >= 0.0, "{name}: {s:?}");
    }
    assert_eq!(
        t.counter("k XC grid points"),
        Some(xc.npoints() as u64),
        "grid-point counter"
    );
}
