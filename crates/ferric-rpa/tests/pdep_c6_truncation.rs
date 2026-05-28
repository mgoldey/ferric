//! Spike: compare full-rank vs PDEP-truncated per-atom C6 on water/cc-pVDZ.
//!
//! Measures accuracy (C6 error vs full solve) and timing as a function of
//! the PDEP truncation threshold (which controls M, the number of eigenpotentials
//! retained).  Run with:
//!   cargo test -p ferric-rpa --test pdep_c6_truncation -- --nocapture

use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::{
    casimir_polder_c6, pdep_dynamic_polarizability, pdep_dynamic_polarizability_truncated,
    DispersionPartition,
};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn build_water() -> (Molecule, PreparedBasis, PreparedBasis, Operator, ferric_scf::ScfResult) {
    let xyz = include_str!("../../../testdata/molecules/water.xyz");
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    (mol, obs, dfbs, op, rhf)
}

#[test]
fn pdep_truncation_accuracy_and_timing() {
    let (mol, obs, dfbs, op, rhf) = build_water();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();

    // -----------------------------------------------------------------------
    // 1. Full-rank reference
    // -----------------------------------------------------------------------
    let mut cfg_full = PdepRpaConfig::default();
    cfg_full.frozen_core = 0;
    cfg_full.trunc_thresh = 0.0;  // keep all modes → full-rank equivalent

    let t0 = Instant::now();
    let dp_full = pdep_dynamic_polarizability(
        &mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg_full, DispersionPartition::Becke,
    ).unwrap();
    let t_full = t0.elapsed();
    let res_full = casimir_polder_c6(&dp_full);
    let c6_full_oo = res_full.c6_iso_pair[(0, 0)];
    let c6_full_hh = res_full.c6_iso_pair[(1, 1)];
    let naux = dfbs.nbasis();

    println!("\n=== PDEP-truncated C6 spike — water/cc-pVDZ ===");
    println!("naux={naux}  n_quad={}  t_full={:.1}ms",
        dp_full.freqs.len(), t_full.as_secs_f64()*1e3);
    println!("Full-rank  C6(O-O)={c6_full_oo:.4}  C6(H-H)={c6_full_hh:.4}");
    println!();
    println!("{:>12} {:>6} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "trunc_thresh", "M", "C6(O-O)", "C6(H-H)",
        "t_trunc(ms)", "err%(O-O)", "err%(H-H)");
    println!("{}", "-".repeat(80));

    // -----------------------------------------------------------------------
    // 2. Truncated path at several thresholds
    // -----------------------------------------------------------------------
    for &thresh in &[0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.2] {
        let mut cfg = PdepRpaConfig::default();
        cfg.frozen_core = 0;
        cfg.trunc_thresh = thresh;

        // Run PDEP to get eigenpotentials + eigenvalues_freq.
        let t1 = Instant::now();
        let rpa = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
        let t_rpa = t1.elapsed();
        let m = rpa.n_eigenpotentials;

        // Truncated C6.
        let t2 = Instant::now();
        let dp_trunc = pdep_dynamic_polarizability_truncated(
            &rpa, &mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg, DispersionPartition::Becke,
        ).unwrap();
        let t_trunc = t2.elapsed();
        let res_trunc = casimir_polder_c6(&dp_trunc);

        let c6_oo = res_trunc.c6_iso_pair[(0, 0)];
        let c6_hh = res_trunc.c6_iso_pair[(1, 1)];
        let err_oo = (c6_oo - c6_full_oo).abs() / c6_full_oo * 100.0;
        let err_hh = if c6_full_hh.abs() > 1e-10 {
            (c6_hh - c6_full_hh).abs() / c6_full_hh * 100.0
        } else { 0.0 };

        println!("{:>12.0e} {:>6} {:>10.4} {:>10.4} {:>10.1}+{:.0} {:>11.2}% {:>11.2}%",
            thresh, m, c6_oo, c6_hh,
            t_trunc.as_secs_f64()*1e3, t_rpa.as_secs_f64()*1e3,
            err_oo, err_hh);

        // Sanity: at thresh=0 the truncated path should be close to full.
        if thresh == 0.0 {
            // The spike uses approximate projection (V metric), so allow 20%.
            assert!(err_oo < 20.0, "thresh=0 should be near full-rank: err={err_oo:.1}%");
        }
    }

    println!();
    println!("t_trunc = grid+dipoles+freq_loop (C^αᵀ B g μ rank-M update)");
    println!("t_rpa   = Davidson eigensolver (run once; not repeated in production)");
    println!("In production: run_pdep_rpa once, reuse eigenpotentials for C6.");
}
