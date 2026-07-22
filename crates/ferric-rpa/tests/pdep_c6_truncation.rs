//! Spike: full-rank vs PDEP-truncated per-atom C6 accuracy and timing.
//!
//! Runs two molecules:
//!   - water / cc-pVDZ-RI      (naux ~ 84,  solve cost negligible vs grid)
//!   - methane / aug-cc-pVTZ-RI (naux ~ 249, solve cost meaningful)
//!
//! For each: sweeps trunc_thresh to vary M (# modes kept), measures
//! C6 error vs the full-rank reference and wall time for the truncated path.
//!
//! Run with:
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --test pdep_c6_truncation \
//!     --release -- --nocapture

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

struct System {
    name: &'static str,
    mol: Molecule,
    obs: PreparedBasis,
    obs_bs: ferric_core::basis::BasisSet,
    dfbs: PreparedBasis,
    op: Operator,
    rhf: ferric_scf::ScfResult,
    /// Reference C6 pair indices to report: (label, i, j).
    pairs: Vec<(&'static str, usize, usize)>,
    thresholds: Vec<f64>,
}

fn make_system(
    xyz: &str,
    obs_name: &'static str,
    dfbs_name: &'static str,
    name: &'static str,
    pairs: Vec<(&'static str, usize, usize)>,
    thresholds: Vec<f64>,
) -> System {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled(obs_name).unwrap();
    let dfbs_bs = basis::bundled(dfbs_name).unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    System { name, mol, obs, obs_bs, dfbs, op, rhf, pairs, thresholds }
}

fn run_system(sys: &System) {
    // Full-rank reference (trunc_thresh = 0.0).
    let cfg_full = PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 0.0,
        ..Default::default()
    };

    let t0 = Instant::now();
    let dp_full = pdep_dynamic_polarizability(
        &sys.mol, &sys.obs, &sys.obs_bs, &sys.dfbs, &sys.rhf, sys.op,
        &cfg_full, DispersionPartition::Becke, None,
    ).unwrap();
    let t_full = t0.elapsed();
    let res_full = casimir_polder_c6(&dp_full);
    let naux = sys.dfbs.nbasis();
    let nfreq = dp_full.freqs.len();

    println!("\n=== {} ===", sys.name);
    println!("naux={naux}  n_quad={nfreq}  t_full={:.1}ms", t_full.as_secs_f64()*1e3);
    print!("Full-rank: ");
    for &(lbl, i, j) in &sys.pairs {
        print!("  C6({lbl})={:.4}", res_full.c6_iso_pair[(i, j)]);
    }
    println!();
    println!();

    // Header
    let pair_hdrs: String = sys.pairs.iter()
        .map(|(lbl, ..)| format!("{:>10}", format!("C6({lbl})")))
        .collect::<Vec<_>>().join("");
    let err_hdrs: String = sys.pairs.iter()
        .map(|(lbl, ..)| format!("{:>11}", format!("err%({lbl})")))
        .collect::<Vec<_>>().join("");
    println!("{:>12} {:>6}{}{:>13}{}", "thresh", "M", pair_hdrs, "t_trunc+rpa", err_hdrs);
    println!("{}", "-".repeat(12 + 6 + sys.pairs.len()*10 + 13 + sys.pairs.len()*11 + 4));

    for &thresh in &sys.thresholds {
        let cfg = PdepRpaConfig {
            frozen_core: 0,
            trunc_thresh: thresh,
            ..Default::default()
        };

        let t1 = Instant::now();
        let rpa = run_pdep_rpa(&sys.mol, &sys.obs, &sys.dfbs, sys.op, &sys.rhf, &cfg).unwrap();
        let t_rpa = t1.elapsed();
        let m = rpa.n_eigenpotentials;

        let t2 = Instant::now();
        let dp = pdep_dynamic_polarizability_truncated(
            &rpa, &sys.mol, &sys.obs, &sys.obs_bs, &sys.dfbs, &sys.rhf, sys.op,
            &cfg, DispersionPartition::Becke,
        ).unwrap();
        let t_trunc = t2.elapsed();
        let res = casimir_polder_c6(&dp);

        let vals: String = sys.pairs.iter()
            .map(|&(_, i, j)| format!("{:>10.4}", res.c6_iso_pair[(i, j)]))
            .collect::<Vec<_>>().join("");
        let errs: String = sys.pairs.iter()
            .map(|&(_, i, j)| {
                let full = res_full.c6_iso_pair[(i, j)];
                let got  = res.c6_iso_pair[(i, j)];
                let pct  = if full.abs() > 1e-10 { (got - full).abs() / full * 100.0 } else { 0.0 };
                format!("{:>11.2}%", pct)
            })
            .collect::<Vec<_>>().join("");

        println!("{:>12.0e} {:>6}{}  {:>8.1}+{:<5.0}{}",
            thresh, m, vals,
            t_trunc.as_secs_f64()*1e3, t_rpa.as_secs_f64()*1e3, errs);

        if thresh == 0.0 {
            for &(lbl, i, j) in &sys.pairs {
                let full = res_full.c6_iso_pair[(i, j)];
                let got  = res.c6_iso_pair[(i, j)];
                if full.abs() > 1e-10 {
                    let err = (got - full).abs() / full * 100.0;
                    assert!(err < 20.0,
                        "{} C6({lbl}) thresh=0 err={err:.1}% (should be <20%)", sys.name);
                }
            }
        }
    }
}

#[test]
#[ignore = "slow: 7-threshold sweep on 2 systems incl. methane/aug-cc-pVTZ \
            (naux~249) PDEP-RPA + per-atom dispersion each -- this file's own \
            module doc already labels it 'Spike:' and instructs running with \
            --release, so it wasn't meant for the default cargo test pass"]
fn pdep_truncation_sweep() {
    // System 1: water / cc-pVDZ-RI  (small naux, baseline)
    let water_xyz = include_str!("../../../testdata/molecules/water.xyz");
    let water = make_system(
        water_xyz, "cc-pvdz", "cc-pvdz-ri",
        "water / cc-pVDZ / cc-pVDZ-RI",
        vec![("O-O", 0, 0), ("H-H", 1, 1), ("O-H", 0, 1)],
        vec![0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.2],
    );

    // System 2: methane / aug-cc-pVTZ / aug-cc-pVTZ-RI  (large naux, interesting regime)
    let methane_xyz = include_str!("../../../testdata/molecules/methane.xyz");
    let methane = make_system(
        methane_xyz, "aug-cc-pvtz", "aug-cc-pvtz-rifit",
        "methane / aug-cc-pVTZ / aug-cc-pVTZ-RI",
        vec![("C-C", 0, 0), ("H-H", 1, 1), ("C-H", 0, 1)],
        vec![0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.2],
    );

    run_system(&water);
    run_system(&methane);

    println!("\nNote: t_trunc = Becke grid + per-atom dipoles + M×M freq loop");
    println!("      t_rpa  = Davidson solver (one-time cost; reusable for many properties)");
    println!("      In production, run_pdep_rpa once and call truncated C6 with the result.");
}
