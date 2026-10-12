//! Schwarz triple cut (`DirectConfig::schwarz_skip`) under the terfc operator.
//!
//! Anchors first (CLAUDE.md protocol):
//!  * trivial limit: `schwarz_skip = 0` and a threshold so small nothing can
//!    fall under it evaluate the SAME triples and give bit-identical energies
//!    (the cut is a pure filter on which triples are computed);
//!  * the cut bites: a mutant threshold far too large moves the energy by
//!    orders more than the production threshold, and skips most triples;
//!  * the production threshold skips a real share of triples for a relative
//!    energy change far below the amplitude truncation (eps).
//!
//! Needs the terf tables; falls back to the repo `terf-tables/` directory.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::AmplitudeLmp2Config;
use ferric_mp2::lmp2_direct::{amplitude_lmp2_direct, DirectConfig, DirectStats};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::sync::Once;

const ANG: f64 = ferric_core::units::ANGSTROM_TO_BOHR;

fn ensure_tables() -> bool {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
            let d = concat!(env!("CARGO_MANIFEST_DIR"), "/../../terf-tables");
            if std::path::Path::new(d).is_dir() {
                std::env::set_var("FERRIC_TERF_TABLE_DIR", d);
            }
        }
    });
    std::env::var("FERRIC_TERF_TABLE_DIR").is_ok()
}

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    obs_bs: basis::BasisSet,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
    nc: usize,
}

fn setup(xyz: &str, nc: usize) -> Setup {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    Setup {
        mol,
        obs,
        obs_bs,
        dfbs,
        rhf,
        nc,
    }
}

fn maps(skip: f64) -> DirectConfig {
    DirectConfig {
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: skip,
        // pinned: the auto budget (0.8*RAM) ignores a cgroup cap
        scratch_budget_bytes: 1usize << 30,
        ..Default::default()
    }
}

fn run(su: &Setup, op: Operator, skip: f64) -> (f64, DirectStats) {
    let cfg = AmplitudeLmp2Config {
        eps: 5e-5,
        frozen_core: su.nc,
        compute_reference: false,
        ..Default::default()
    };
    let (r, st) = amplitude_lmp2_direct(
        &su.mol,
        &su.obs,
        &su.obs_bs,
        &su.dfbs,
        op,
        &su.rhf,
        &cfg,
        &maps(skip),
    )
    .unwrap();
    (r.e_corr, st)
}

fn ops() -> [(&'static str, Operator); 2] {
    [
        (
            "linked",
            Operator::terfc_with_omega(1.0 * ANG, std::f64::consts::FRAC_1_SQRT_2 / (1.0 * ANG)),
        ),
        ("NCP", Operator::terfc_with_omega(0.991 * ANG, 1.3663 / ANG)),
    ]
}

fn check(xyz: &str, nc: usize, min_frac: f64, which: &[usize]) {
    if !ensure_tables() {
        eprintln!("SKIPPED: no terf tables");
        return;
    }
    let su = setup(xyz, nc);
    for &i in which {
        let (name, op) = ops()[i];
        let (e0, s0) = run(&su, op, 0.0);
        let (et, st) = run(&su, op, 1e-300);
        assert_eq!(s0.n_eri3_skipped, 0);
        assert_eq!(
            st.n_eri3_skipped, 0,
            "{name}: tiny threshold skipped triples"
        );
        assert_eq!(st.n_eri3_shell_triples, s0.n_eri3_shell_triples);
        assert_eq!(
            e0.to_bits(),
            et.to_bits(),
            "{name}: not bit-identical in the trivial limit"
        );

        let (e5, s5) = run(&su, op, 1e-5);
        let (e2, s2) = run(&su, op, 1e-2);
        let frac5 = s5.n_eri3_skipped as f64 / (s5.n_eri3_shell_triples + s5.n_eri3_skipped) as f64;
        let frac2 = s2.n_eri3_skipped as f64 / (s2.n_eri3_shell_triples + s2.n_eri3_skipped) as f64;
        let rel5 = ((e5 - e0) / e0).abs();
        let rel2 = ((e2 - e0) / e0).abs();
        eprintln!(
            "{xyz} {name}: E0 {e0:.10} skipped {frac5:.3} (1e-5) rel dE {rel5:.2e} | \
             mutant 1e-2: skipped {frac2:.3} rel dE {rel2:.2e}"
        );
        assert!(
            frac5 > min_frac,
            "{name}: production cut skipped only {frac5:.3}"
        );
        assert!(rel5 < 1e-5, "{name}: production cut moved E by {rel5:.2e}");
        assert!(
            frac2 > frac5 && rel2 > 100.0 * rel5,
            "{name}: mutant not loud: {rel2:.2e}"
        );
    }
}

/// Cheap: C4 linked.
#[test]
fn terfc_schwarz_trivial_limit_and_mutant_c4() {
    check("alkane_4.xyz", 4, 0.04, &[0]);
}

/// C8, linked and NCP; 215 s release on a loaded box (2026-10-11).
#[test]
#[ignore = "C8 x 2 operators x 4 runs, ~215 s release"]
fn terfc_schwarz_trivial_limit_and_mutant_c8() {
    check("alkane_8.xyz", 8, 0.18, &[0, 1]);
}

/// A/B wall + energy, plain Schwarz, alternating run order, repeated.
///   FERRIC_BENCH_C=12|16  FERRIC_BENCH_TAUS=0,1e-8,1e-6,1e-5,1e-4
///   FERRIC_BENCH_GATE=0.7 (pair gate on; default off)  FERRIC_BENCH_REPS=2
///   RAYON_NUM_THREADS=6 scripts/ferric-limited -- cargo test --release -p ferric-mp2 \
///     --test lmp2_direct_terfc_schwarz -- --ignored --nocapture bench_terfc_schwarz_ab
#[test]
#[ignore = "benchmark"]
fn bench_terfc_schwarz_ab() {
    if !ensure_tables() {
        return;
    }
    let envf = |k: &str| std::env::var(k).ok();
    let nc: usize = envf("FERRIC_BENCH_C")
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let reps: usize = envf("FERRIC_BENCH_REPS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let gate: Option<f64> = envf("FERRIC_BENCH_GATE").and_then(|v| v.parse().ok());
    let taus: Vec<f64> = envf("FERRIC_BENCH_TAUS")
        .unwrap_or("0,1e-8,1e-6,1e-5,1e-4".into())
        .split(',')
        .map(|t| t.parse().unwrap())
        .collect();
    let only: Option<usize> = envf("FERRIC_BENCH_OP").and_then(|v| v.parse().ok());
    let su = setup(&format!("alkane_{nc}.xyz"), nc);
    println!("sys op tau rep order E_corr wall_s t_eri3_s eval_Mtriples skipped_Mtriples");
    for (oi, (name, op)) in ops().into_iter().enumerate() {
        if only.is_some_and(|o| o != oi) {
            continue;
        }
        for rep in 0..reps {
            let mut order = taus.clone();
            if rep % 2 == 1 {
                order.reverse();
            }
            for (k, &tau) in order.iter().enumerate() {
                let cfg = AmplitudeLmp2Config {
                    eps: 5e-5,
                    frozen_core: nc,
                    compute_reference: false,
                    pair_gate_cal: gate,
                    ..Default::default()
                };
                let t = std::time::Instant::now();
                let (r, st) = amplitude_lmp2_direct(
                    &su.mol,
                    &su.obs,
                    &su.obs_bs,
                    &su.dfbs,
                    op,
                    &su.rhf,
                    &cfg,
                    &maps(tau),
                )
                .unwrap();
                println!(
                    "C{nc} {name} {tau:.0e} rep{rep} pos{k} {:.12} {:.1} {:.1} {:.3} {:.3}",
                    r.e_corr,
                    t.elapsed().as_secs_f64(),
                    st.t_eri3_s,
                    st.n_eri3_shell_triples as f64 / 1e6,
                    st.n_eri3_skipped as f64 / 1e6
                );
            }
        }
    }
}
