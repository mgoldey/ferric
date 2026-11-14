//! Timing harness for OO-RI-MP2's J/K routes (ignored; run by hand).
//!
//! ```text
//! FERRIC_OO_BENCH_XYZ=testdata/molecules/alkane_6.xyz FERRIC_OO_BENCH_ROUTE=plain \
//!   FERRIC_TIMING=1 cargo test --release -p ferric-mp2 --test bench_oo_rimp2_jk \
//!   -- --ignored --nocapture
//! ```
//!
//! `FERRIC_OO_BENCH_ROUTE`: `plain` (four-centre J/K built from scratch at
//! every point, the default config), `difference` (four-centre, difference
//! builds) or `rijk` (RI-JK reference on `def2-universal-jkfit`, so the
//! functional is the fitted one). `FERRIC_OO_BENCH_BASIS` (default cc-pvdz)
//! and `FERRIC_OO_BENCH_AUX` (default cc-pvdz-ri) pick the bases;
//! `FERRIC_OO_BENCH_GRADIENT=1` also times the nuclear gradient. Prints one
//! `RESULT` line with wall and process CPU seconds per stage.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config};
use ferric_mp2::oo_rimp2_gradient::oo_ri_mp2_gradient;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

/// Process CPU seconds (user + system) from `/proc/self/stat`.
fn cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = stat.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    let ticks = |i: usize| {
        f.get(i)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(f64::NAN)
    };
    // Fields after the command name: state is index 0, utime 11, stime 12.
    (ticks(11) + ticks(12)) / 100.0
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

#[test]
#[ignore = "timing harness; needs FERRIC_OO_BENCH_XYZ"]
fn bench_oo_rimp2_jk_routes() {
    let xyz = std::env::var("FERRIC_OO_BENCH_XYZ").expect("FERRIC_OO_BENCH_XYZ");
    let route = env_or("FERRIC_OO_BENCH_ROUTE", "difference");
    let basis_name = env_or("FERRIC_OO_BENCH_BASIS", "cc-pvdz");
    let aux_name = env_or("FERRIC_OO_BENCH_AUX", "cc-pvdz-ri");
    let jk_aux = match route.as_str() {
        "plain" | "difference" => None,
        "rijk" => Some("def2-universal-jkfit".to_string()),
        other => panic!("FERRIC_OO_BENCH_ROUTE = {other}: plain | difference | rijk"),
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(&xyz);
    let mol = Molecule::load_xyz(path.to_str().unwrap()).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(&basis_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(&aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();

    let (w0, c0) = (Instant::now(), cpu_seconds());
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            df_j_aux: jk_aux.clone(),
            df_k_aux: jk_aux,
            ..Default::default()
        }
        .for_geometry_task(),
    )
    .unwrap();
    assert!(rhf.converged);
    let (w_scf, c_scf) = (w0.elapsed().as_secs_f64(), cpu_seconds() - c0);

    let cfg = OoRiMp2Config {
        difference_jk: route != "plain",
        ..Default::default()
    };
    let (w1, c1) = (Instant::now(), cpu_seconds());
    let oo = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &cfg, None).unwrap();
    let (w_oo, c_oo) = (w1.elapsed().as_secs_f64(), cpu_seconds() - c1);

    let (mut w_g, mut c_g) = (0.0, 0.0);
    if std::env::var("FERRIC_OO_BENCH_GRADIENT").is_ok_and(|v| v == "1") {
        let (w2, c2) = (Instant::now(), cpu_seconds());
        let g = oo_ri_mp2_gradient(&mol, &obs, &dfbs, op, &bounds, &oo, 0, None).unwrap();
        (w_g, c_g) = (w2.elapsed().as_secs_f64(), cpu_seconds() - c2);
        let gmax = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        println!("gradient max|g| = {gmax:.10e}");
    }
    println!(
        "RESULT route={route} nbas={} E={:.15e} iters={} converged={} scf_wall={w_scf:.2} \
         scf_cpu={c_scf:.2} oo_wall={w_oo:.2} oo_cpu={c_oo:.2} grad_wall={w_g:.2} grad_cpu={c_g:.2}",
        obs.nbasis(),
        oo.total_energy,
        oo.iterations,
        oo.converged
    );
}
