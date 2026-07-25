//! Head-to-head timing: scalar-loop RI-MP2 vs einsum!-framework RI-MP2.
//!
//! The scalar path streams the (ia|jb) inner product as a hand loop
//! O(o^2 v^2 naux); the einsum! path materializes (ia|jb) and routes the
//! contractions through BLAS3 GEMM. This benchmark measures whether that is a
//! net win and at what size.
//!
//! Run with:
//!   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo test -p ferric-mp2 --release --test rimp2_einsum_bench -- --ignored --nocapture

use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2_einsum, ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn bench_one(name: &str, xyz: &str, obs_name: &str, dfbs_name: &str, reps: usize) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled(obs_name).unwrap();
    let dfbs_bs = basis::bundled(dfbs_name).unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    let cfg = RiMp2Config { frozen_core: 0, memory_budget_bytes: None, ..Default::default() };

    // Warm up + correctness cross-check.
    let (sc_ref, _) = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let sc_ein = ri_mp2_einsum(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    assert!((sc_ref.e_total - sc_ein.e_total).abs() < 1e-9);

    let nbas = obs.nbasis();

    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    }
    let t_scalar = t0.elapsed().as_secs_f64() / reps as f64;

    let t1 = Instant::now();
    for _ in 0..reps {
        let _ = ri_mp2_einsum(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    }
    let t_einsum = t1.elapsed().as_secs_f64() / reps as f64;

    println!(
        "{name:24} nbas={nbas:3}  scalar={:8.2}ms  einsum={:8.2}ms  speedup={:.2}x  E={:.8}",
        t_scalar * 1e3,
        t_einsum * 1e3,
        t_scalar / t_einsum,
        sc_ein.e_total
    );
}

#[test]
#[ignore = "benchmark: run with --release --ignored --nocapture"]
fn rimp2_scalar_vs_einsum_timing() {
    println!();
    println!("=== RI-MP2: scalar-loop vs einsum! (both include integral build) ===");
    // Note: both paths rebuild the RI integrals each call, so timings include
    // the AO->MO + Cholesky build (shared, identical), plus the differing
    // energy-contraction step. The contraction is the only part that differs.
    bench_one(
        "water/cc-pVDZ",
        "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
        "cc-pvdz",
        "cc-pvdz-ri",
        5,
    );
    bench_one(
        "water/def2-TZVP",
        "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
        "def2-tzvp",
        "def2-tzvp-rifit",
        3,
    );
    bench_one(
        "water/aug-cc-pVTZ",
        "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
        "aug-cc-pvtz",
        "aug-cc-pvtz-rifit",
        3,
    );
    bench_one(
        "methane/def2-TZVP",
        "5\n\nC 0.0 0.0 0.0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629\n",
        "def2-tzvp",
        "def2-tzvp-rifit",
        3,
    );
}
