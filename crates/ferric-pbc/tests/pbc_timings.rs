//! Stage timers (`ferric_pbc::timing`; FINDINGS "Performance plan (research)
//! — 2026-09-25", item 0) are populated, internally consistent, and do not
//! move a number.
//!
//! The timers have no off switch: they read the wall and process-CPU clocks at
//! stage boundaries and never touch an integral, a matrix or a loop order.
//! What CAN go wrong, and what each test guards:
//!
//! * a stage is never recorded, or two stages overlap (then `Σ stages` could
//!   exceed the component's total) — `*_timings_are_populated_and_disjoint`;
//! * a counter disagrees with the struct field it copies (a counter taken at
//!   the wrong point, e.g. before the screen ran) — the same tests compare
//!   each counter with its `RsGdfStats` / `PeriodicHcore` field;
//! * the per-call J/K/XC clocks are not wired (0 calls after an SCF);
//! * instrumentation perturbs the numerics (e.g. a reordered accumulation
//!   while splitting the LR pair FT from its GEMMs): two builds of the same B
//!   and two SCFs on it must agree BIT for bit, and the RS-GDF energy must
//!   still match the dense-AFT oracle to its pinned fitting error
//!   (`pbc_rsgdf.rs`, cc-pvdz-ri: −1.974228505452e-06 Ha, exxdiv none).

mod common;

use common::*;
use ferric_core::basis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{gamma_rks, GammaRksConfig, PeriodicGridConfig};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
use ferric_pbc::timing::PbcTimings;
use ferric_pbc::uhf::GammaUhfIntegrals;

const HCORE_OMEGA: f64 = 0.8;
/// `pbc_rsgdf.rs` H2_PROTO_DE (cc-pvdz-ri, exxdiv none).
const H2_RI_DE_NONE: f64 = -1.974228505452e-06;

/// Every stage non-negative, the leaves' sum within the total (plus clock
/// granularity), the total positive.
fn assert_consistent(t: &PbcTimings, who: &str) {
    assert!(t.wall_s > 0.0, "{who}: total wall {}", t.wall_s);
    for s in &t.stages {
        assert!(s.wall_s >= 0.0, "{who}: stage {} wall {}", s.name, s.wall_s);
        assert!(s.calls >= 1, "{who}: stage {} has no calls", s.name);
        if let Some(c) = s.cpu_s {
            assert!(c >= 0.0, "{who}: stage {} cpu {c}", s.name);
        }
    }
    let sum = t.stage_wall_sum();
    assert!(
        sum <= t.wall_s + 1e-3,
        "{who}: stages sum to {sum} s > total {} s (overlapping stages?)",
        t.wall_s
    );
}

fn stage_names(t: &PbcTimings) -> Vec<&'static str> {
    t.stages.iter().map(|s| s.name).collect()
}

#[test]
fn hcore_and_rsgdf_timings_are_populated_and_disjoint() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc =
        periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(HCORE_OMEGA)).expect("hcore");
    let ht = &hc.timings;
    assert_consistent(ht, "hcore");
    for name in [
        "hcore setup (shells, pair images)",
        "hcore S/T",
        "hcore SR attraction",
        "hcore LR attraction",
        "hcore Ewald E_nn",
    ] {
        assert!(
            ht.stage(name).is_some(),
            "hcore stage {name:?} missing: {:?}",
            stage_names(ht)
        );
    }
    assert!(
        ht.stage("hcore ECP").is_none(),
        "all-electron basis timed an ECP stage"
    );
    assert_eq!(
        ht.counter("hcore SR triplets"),
        Some(hc.n_sr_triplets as u64)
    );
    assert_eq!(ht.counter("hcore pair images"), Some(hc.n_images as u64));
    assert_eq!(ht.counter("hcore LR half-G"), Some(hc.n_g_half as u64));
    assert_eq!(ht.counter("hcore LR chunks"), Some(hc.n_lr_chunks as u64));
    // Every computed triplet passed a segment test first.
    let tests = ht
        .counter("hcore SR segment tests")
        .expect("segment-test counter");
    assert!(tests >= hc.n_sr_triplets as u64 && hc.n_sr_triplets > 0);

    let aux = prep_for(&cell, &basis::bundled("cc-pvdz-ri").unwrap());
    let cfg = RsGdfConfig {
        exxdiv: ExxDiv::None,
        budget_bytes: Some(1 << 30),
        ..Default::default()
    };
    let gdf = RsGdf::build(&cell, &prep, &aux, &hc.s, &cfg).expect("rsgdf");
    let gt = gdf.timings();
    for name in [
        "rsgdf setup (shells, pair images, G list)",
        "rsgdf SR metric (2-centre)",
        "rsgdf SR 3-centre",
        "rsgdf LR pair FT",
        "rsgdf LR aux FT + GEMM",
        "rsgdf G=0 + symmetrise",
        "rsgdf metric eigh",
        "rsgdf B = (J3 W)^T",
    ] {
        assert!(
            gt.stage(name).is_some(),
            "rsgdf stage {name:?} missing: {:?}",
            stage_names(&gt)
        );
    }
    // Before any SCF the J/K clocks exist but have run nothing; the build
    // total covers the build stages only.
    assert_eq!(gt.stage("scf J (rsgdf)").map(|s| s.calls), Some(0));
    assert_eq!(gt.stage("scf K (rsgdf)").map(|s| s.calls), Some(0));
    let mut build_only = gt.clone();
    build_only.stages.retain(|s| !s.name.starts_with("scf "));
    assert_consistent(&build_only, "rsgdf build");
    let st = gdf.stats();
    assert_eq!(
        gt.counter("rsgdf SR3 triplets"),
        Some(st.n_sr3_triplets as u64)
    );
    assert_eq!(gt.counter("rsgdf SR2 pairs"), Some(st.n_sr2_pairs as u64));
    assert_eq!(gt.counter("rsgdf LR half-G"), Some(st.n_g_half as u64));
    assert_eq!(gt.counter("rsgdf LR chunks"), Some(st.n_g_chunks as u64));
    assert_eq!(gt.counter("rsgdf naux"), Some(st.naux as u64));
    assert_eq!(gt.counter("rsgdf aux dropped"), Some(st.n_dropped as u64));
    assert_eq!(
        gt.stage("rsgdf LR aux FT + GEMM").unwrap().calls,
        st.n_g_chunks as u64,
        "one GEMM sink call per pair-FT chunk"
    );

    // Numbers do not move: a second build is bitwise the first, two SCFs on
    // it agree bit for bit, and the energy keeps its pinned fitting error.
    let gdf2 = RsGdf::build(&cell, &prep, &aux, &hc.s, &cfg).expect("rsgdf again");
    assert!(
        gdf.b()
            .iter()
            .zip(gdf2.b().iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "a repeated RS-GDF build changed B"
    );
    let e1 = gamma_rhf_jk(
        &cell,
        &prep,
        &hc,
        Box::new(gdf.j_builder()),
        Box::new(gdf.k_builder()),
    );
    let e2 = gamma_rhf_jk(
        &cell,
        &prep,
        &hc,
        Box::new(gdf.j_builder()),
        Box::new(gdf.k_builder()),
    );
    assert_eq!(e1.energy.to_bits(), e2.energy.to_bits());
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("dense AFT");
    let e_dense = gamma_rhf(&cell, &prep, &hc, &eri).energy;
    let de = e1.energy - e_dense;
    assert!(
        (de - H2_RI_DE_NONE).abs() < 1e-9,
        "RS-GDF fitting error {de:e} vs pinned {H2_RI_DE_NONE:e}"
    );

    // The per-call clocks saw both SCFs.
    let gt = gdf.timings();
    let (j, k) = (
        gt.stage("scf J (rsgdf)").unwrap(),
        gt.stage("scf K (rsgdf)").unwrap(),
    );
    assert!(
        j.calls >= 2 && k.calls >= 2,
        "J {} / K {} calls",
        j.calls,
        k.calls
    );
    assert!(
        j.calls as usize >= e1.iterations,
        "{} J calls, {} iterations",
        j.calls,
        e1.iterations
    );

    // Dense AFT: build stages + its own J/K clocks.
    let dt = eri.timings();
    for name in ["dense AFT pair FT", "dense AFT GEMM"] {
        assert!(
            dt.stage(name).is_some(),
            "dense stage {name:?} missing: {:?}",
            stage_names(&dt)
        );
    }
    assert!(dt.stage("scf J (dense AFT)").unwrap().calls >= 1);
    assert_eq!(dt.counter("dense AFT half-G"), Some(eri.n_g_half() as u64));
}

#[test]
fn rks_timings_count_the_xc_calls() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc =
        periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(HCORE_OMEGA)).expect("hcore");
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::Ewald,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("dense AFT");
    let mut cfg = GammaRksConfig::new("LDA");
    cfg.grid = PeriodicGridConfig {
        neighbour_cutoff: Some(10.0),
        ..PeriodicGridConfig::with_size(30, 110)
    };
    let r = gamma_rks(&cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &cfg).expect("rks");
    let t = &r.timings;
    assert_consistent(t, "gamma_rks");
    for name in ["xc grid build", "xc AO cache", "scf XC eval"] {
        assert!(
            t.stage(name).is_some(),
            "rks stage {name:?} missing: {:?}",
            stage_names(t)
        );
    }
    // One XC call per SCF Fock build plus the post-SCF E_xc evaluation.
    let calls = t.stage("scf XC eval").unwrap().calls as usize;
    assert!(
        calls >= r.scf.iterations,
        "{calls} XC calls, {} iterations",
        r.scf.iterations
    );
    assert_eq!(t.counter("xc grid points"), Some(r.n_grid_points as u64));
}
