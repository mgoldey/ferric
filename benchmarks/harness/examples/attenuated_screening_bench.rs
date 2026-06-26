//! Wall-time benchmark for QQR-3 screened attenuated MP2.
//!
//! Tests whether the distance-aware screening in the 3-index ERI build
//! actually translates to a measurable wall-time win on decane (C10H22),
//! the smallest molecule where Schwarz-derived QQR was observed to drop
//! a meaningful fraction of shell triples (37.6% retention at thresh=1e-6).
//!
//! Reports per-threshold: total attenuated-MP2 wall time, screened-triple
//! retention rate, and the energy delta vs the unscreened reference.
//!
//! Usage:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-mp2 \
//!       --example attenuated_screening_bench

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::attenuated::{attenuated_ri_mp2, AttenuatedMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

fn run_one(xyz_path: &str, label: &str) {
    println!("\n=== {label} ({xyz_path}) ===");
    let mol = Molecule::load_xyz(xyz_path).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op_c = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op_c, &obs).unwrap();

    println!("  nbas = {}, nshells = {}", obs.nbasis(), obs.nshells());

    let t0 = Instant::now();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol, &obs, op_c, &bounds,
        &RhfConfig { energy_conv: 1e-9, ..Default::default() },
    ).unwrap();
    let t_scf = t0.elapsed();
    println!("  RHF: E = {:.10} Ha ({:.2}s)", rhf.energy, t_scf.as_secs_f64());

    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    println!("  naux = {}", dfbs.nbasis());

    // Dense (unscreened) attenuated MP2 — reference.
    let cfg_dense = AttenuatedMp2Config {
        omega: 0.222,           // 0.420 Å⁻¹ — dissertation erfc optimum
        scaling: 1.0,
        frozen_core: 0,
        screen_thresh: None,
    };
    let t0 = Instant::now();
    let r_dense = attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &cfg_dense).unwrap();
    let t_dense = t0.elapsed();
    println!("  ─ dense (unscreened):   E_corr = {:.10} Ha, t = {:.2}s",
        r_dense.mp2_corr, t_dense.as_secs_f64());

    // Screened sweep.
    for &thresh in &[1e-12, 1e-10, 1e-8, 1e-6] {
        let cfg = AttenuatedMp2Config {
            omega: 0.222,
            scaling: 1.0,
            frozen_core: 0,
            screen_thresh: Some(thresh),
        };
        let t0 = Instant::now();
        let r = attenuated_ri_mp2(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();
        let t = t0.elapsed();
        let de = r.mp2_corr - r_dense.mp2_corr;
        let speedup = t_dense.as_secs_f64() / t.as_secs_f64();
        println!(
            "  ─ QQR3 thresh={thresh:.0e}: E_corr = {:.10} Ha, t = {:.2}s ({:.2}×), ΔE = {:+.2e} Ha",
            r.mp2_corr, t.as_secs_f64(), speedup, de
        );
    }
}

fn main() {
    run_one("testdata/molecules/water.xyz", "water (sanity check)");
    run_one("testdata/molecules/alkane_10.xyz", "decane C10H22");
}
