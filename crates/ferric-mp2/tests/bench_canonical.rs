//! Wall-clock benchmark for canonical MP2 (AO ERI build + AO->MO transform +
//! energy). Not a correctness gate -- the correctness gates are the unit tests
//! in `canonical.rs`. Run explicitly:
//!
//! ```text
//!   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo test -p ferric-mp2 --release --test bench_canonical -- --ignored --nocapture
//! ```
//!
//! Times only the `canonical_mp2` call (SCF is excluded, solved once up front),
//! best-of-3, matching `scripts`-side PySCF `mp.MP2` timing which likewise
//! excludes SCF.
//!
//! ## Measured 2026-07-25 (12-core box, OPENBLAS=1 RAYON=1, best of 3)
//!
//! | system | nbas | before | after | speedup | PySCF |
//! |---|---|---|---|---|---|
//! | water/cc-pVDZ    | 24 | 4.48 s  | 0.025 s | 179x  | 0.002 s |
//! | methane/cc-pVDZ  | 34 | ~42 s*  | 0.067 s | ~630x | 0.005 s |
//! | water/aug-cc-pVDZ| 41 | ~137 s* | 0.099 s | ~1400x| 0.010 s |
//! | ethane/cc-pVDZ   | 58 | ~55 min*| 0.531 s | ~6200x| 0.043 s |
//!
//! `*` extrapolated from the measured water/cc-pVDZ point along the old code's
//! `O(nbas^4 * (nocc*nvir)^2)` scaling -- only water/cc-pVDZ was timed directly
//! on the old path (the larger ones were impractically slow to run 3x).
//! Correlation energies are unchanged to ~1e-10 (RHF-convergence-limited) and
//! agree with PySCF on every system.
//!
//! ## The "PySCF" column above is NOT comparable (corrected 2026-07-25)
//!
//! Those PySCF times were `mp.MP2(mf).kernel()` on an `mf` whose SCF had
//! already cached the full AO ERI tensor in `mf._eri`. `mp.MP2` reuses that
//! cache, so the timed region builds **no integrals at all**, while
//! `canonical_mp2` builds every one from scratch. Evidence: on ethane,
//! `mol.intor("int2e")` alone is 0.146 s -- 4x the 0.035 s "total MP2 time".
//!
//! Forcing PySCF to build its own (`mf._eri = None`), same box, same
//! OPENBLAS/OMP/MKL=1 pinning:
//!
//! | system | ferric | PySCF cold (builds ERIs) | PySCF warm (reuses SCF's) |
//! |---|---|---|---|
//! | methane/cc-pVDZ | 0.067 s | 0.101 s | 0.004 s |
//! | ethane/cc-pVDZ  | 0.522 s | 0.691 s | 0.033 s |
//!
//! Like for like, ferric is already ahead. Keep the distinction in mind before
//! quoting a "PySCF is Nx faster" number off this file.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::canonical::canonical_mp2;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

fn bench(label: &str, xyz: &str, basis_name: &str, reps: usize) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let rhf = solve_rhf(
        &ctx,
        &mol,
        &prep,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-9,
            ..Default::default()
        },
    )
    .unwrap();

    let mut times = Vec::new();
    let mut e = 0.0;
    for _ in 0..reps {
        let t = Instant::now();
        e = canonical_mp2(&mol, &prep, op, &rhf, 0).unwrap();
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "FERRIC {label}/{basis_name} nbas={} e_corr={:.12} best={:.3}s median={:.3}s all={:?}",
        prep.nbasis(),
        e,
        times[0],
        times[times.len() / 2],
        times.iter().map(|t| format!("{t:.3}")).collect::<Vec<_>>()
    );
}

const H2O: &str = "3\nwater\nO   0.000000   0.000000   0.117790\nH   0.000000   0.755453  -0.471161\nH   0.000000  -0.755453  -0.471161\n";

#[test]
#[ignore]
fn bench_h2o_ccpvdz() {
    bench("water", H2O, "cc-pvdz", 3);
}

#[test]
#[ignore]
fn bench_h2o_augccpvdz() {
    bench("water", H2O, "aug-cc-pvdz", 3);
}

#[test]
#[ignore]
fn bench_ch4_ccpvdz() {
    let xyz = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/methane.xyz"
    ))
    .unwrap();
    bench("methane", &xyz, "cc-pvdz", 3);
}

#[test]
#[ignore]
fn bench_c2h6_ccpvdz() {
    let xyz = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/c2h6.xyz"
    ))
    .unwrap();
    bench("ethane", &xyz, "cc-pvdz", 3);
}
