//! COSX density-driven screen threshold sweep (ignored; run explicitly).
//!
//! For one (system, basis) cell: converge RHF, build the unscreened COSX K on
//! the production grid (50,110)+fit, then the screened K at a list of
//! thresholds, and print per threshold `max|K_scr - K_unscr|`, the kept
//! (pair, batch) fraction (density-driven) next to the geometry-only kept
//! fraction (what an integral-magnitude screen would keep), and the wall time
//! on one thread. When `COSX_SS_DIRECT=1` the analytic K is also built so the
//! grid error is on the same line — the default threshold is the loosest one
//! whose screen error stays far (>= 10x) below that grid error.
//!
//! Env:
//!   COSX_SS_SYSTEM   xyz stem under testdata/molecules (default alkane_4); "water" = inline water
//!   COSX_SS_BASIS    bundled basis (default def2-svp)
//!   COSX_SS_THRESHES comma list (default 1e-5,1e-6,1e-7,1e-8,1e-9,1e-10)
//!   COSX_SS_DIRECT   "1": also build the analytic (direct) K for the grid error
//!   COSX_SS_DFJK     "1": converge with DF-JK (jkfit) instead of direct JK

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::{CosxConfig, CosxK};
use ferric_scf::fock::KBuilder;
use ferric_scf::rhf::{build_jk, solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use std::time::Instant;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

fn testdata(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}

fn env_flag(name: &str) -> bool {
    matches!(std::env::var(name).ok().as_deref(), Some("1") | Some("true") | Some("yes"))
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v))
}

fn build_k(ctx: &ParallelContext, mol: &Molecule, prep: &PreparedBasis, cfg: CosxConfig, d: &Array2<f64>) -> (Array2<f64>, ferric_scf::cosx_k::CosxTimings, f64) {
    let n = prep.nbasis();
    let mut kb = CosxK::new(ctx, mol, prep, cfg, usize::MAX).expect("CosxK::new");
    let mut k = Array2::zeros((n, n));
    let t0 = Instant::now();
    kb.build(d, &mut k).expect("cosx build");
    (k, *kb.last_timings(), t0.elapsed().as_secs_f64())
}

#[test]
#[ignore = "threshold sweep; run explicitly with --ignored --nocapture"]
fn cosx_screen_sweep_cell() {
    let system = std::env::var("COSX_SS_SYSTEM").unwrap_or_else(|_| "alkane_4".into());
    let basis = std::env::var("COSX_SS_BASIS").unwrap_or_else(|_| "def2-svp".into());
    let threshes: Vec<f64> = std::env::var("COSX_SS_THRESHES")
        .unwrap_or_else(|_| "1e-5,1e-6,1e-7,1e-8,1e-9,1e-10".into())
        .split(',')
        .map(|s| s.trim().parse().expect("threshold"))
        .collect();
    let mol = if system == "water" {
        Molecule::parse_xyz(WATER, 0, 1).expect("water")
    } else {
        Molecule::load_xyz(&testdata(&format!("testdata/molecules/{system}.xyz"))).expect("xyz")
    };
    let bs = bundled(&basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let nbf = prep.nbasis();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let schwarz = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("pool");

    // DF-JK: the default convergence (energy_conv is a sanity bound; a tight
    // one is unreachable on the RI noise floor). Direct: tighter is cheap.
    let scf_cfg = if env_flag("COSX_SS_DFJK") {
        RhfConfig {
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("def2-universal-jkfit".into()),
            ..Default::default()
        }
    } else {
        RhfConfig { energy_conv: 1e-10, density_conv: 1e-8, ..Default::default() }
    };
    let res = solve_rhf(&ctx, &mol, &prep, op, &schwarz, &scf_cfg).expect("rhf");
    assert!(res.converged, "reference RHF did not converge");
    let d = res.density_total.clone();
    println!("\n=== COSX screen sweep: {system} / {basis}  natoms={} nsh={} nbf={nbf}  E={:.8} ===", mol.atoms.len(), prep.nshells(), res.energy);

    let k_direct = env_flag("COSX_SS_DIRECT").then(|| {
        let mut j = Array2::zeros((nbf, nbf));
        let mut k = Array2::zeros((nbf, nbf));
        build_jk(&ctx, &prep, &schwarz, 1e-14, &d, &mut j, &mut k).expect("direct jk");
        k
    });

    let (k_un, t_un, w_un) = pool.install(|| build_k(&ctx, &mol, &prep, CosxConfig { screen_thresh: None, ..CosxConfig::default() }, &d));
    let grid_err = k_direct.as_ref().map(|kd| max_abs_diff(&k_un, kd));
    println!(
        "unscreened: wall {w_un:.3} s (A-build {:.3} s = {:.1}%), pairs {}/{}; grid error vs direct K = {}",
        t_un.a_build_s,
        100.0 * t_un.a_build_s / t_un.total_s,
        t_un.pairs_kept,
        t_un.pairs_total,
        grid_err.map(|e| format!("{e:.3e}")).unwrap_or_else(|| "n/a".into())
    );
    println!("ROWHDR | system | basis | t | max|dK| | dK/grid_err | kept_dd | kept_geom | wall_s | A_build_s | speedup");
    for &t in &threshes {
        let (k_sc, ts, w_sc) = pool.install(|| build_k(&ctx, &mol, &prep, CosxConfig { screen_thresh: Some(t), ..CosxConfig::default() }, &d));
        let dev = max_abs_diff(&k_un, &k_sc);
        let kept = ts.pairs_kept as f64 / ts.pairs_total as f64;
        let geom = ts.pairs_kept_geom as f64 / ts.pairs_total as f64;
        println!(
            "ROW | {system} | {basis} | {t:.0e} | {dev:.3e} | {} | {kept:.4} | {geom:.4} | {w_sc:.3} | {:.3} | {:.2}x",
            grid_err.map(|e| format!("{:.3e}", dev / e)).unwrap_or_else(|| "n/a".into()),
            ts.a_build_s,
            w_un / w_sc
        );
    }
}
