//! SCRATCH diagnostic: sweep table-interpolation accuracy against the exact
//! Poisson series over the WHOLE reachable (S,s) domain, not just the S=20
//! seam. The series is the defining formula and is valid everywhere (it is
//! only *used* for S>20 because it is slower), so it is ground truth here.
use ferric_integrals as _;
use std::os::raw::{c_char, c_double, c_int};
use std::path::PathBuf;

extern "C" {
    fn scf_terfc_debug_series_G(s_big: c_double, s_small: c_double, m: c_int) -> c_double;
    fn scf_terfc_debug_interp_G(
        dir: *const c_char,
        s_big: c_double,
        s_small: c_double,
        m: c_int,
        n: c_int,
    ) -> c_double;
}

fn table_dir() -> Option<PathBuf> {
    let d = std::env::var("FERRIC_TERF_TABLE_DIR").ok()?;
    let p = PathBuf::from(d);
    if p.join("16_4_2.bin").exists() {
        Some(p)
    } else {
        None
    }
}

#[test]
#[ignore = "benchmark: terfc interpolation sweep; --release --ignored --nocapture"]
fn sweep_interp_vs_series_full_domain() {
    let Some(dir) = table_dir() else {
        eprintln!("skip: no tables");
        return;
    };
    let cdir = std::ffi::CString::new(dir.to_str().unwrap()).unwrap();

    // Reachable s: curvature constraint r0*omega = 1/sqrt2 => s = phi^2 r0^2
    // <= omega^2 r0^2 = 1/2. Sample the whole reachable band.
    let s_vals = [0.0_f64, 0.05, 0.15, 0.25, 0.35, 0.45, 0.4999];

    // Per decade of S, track worst relative error and where it occurs.
    // Bucket by which table covers: (0,4] pts=16, (4,10] pts=8, (10,20] pts=4.
    let buckets: [(f64, f64, &str); 3] = [
        (0.01, 4.0, "table0 pts=16 (S<=4)"),
        (4.0, 10.0, "table1 pts=8  (4<S<=10)"),
        (10.0, 20.0, "table2 pts=4  (10<S<=20)"),
    ];

    for &(lo, hi, name) in &buckets {
        for &m in &[0_i32, 1, 2, 4, 7, 10] {
            let mut worst_rel = 0.0f64;
            let mut worst_abs = 0.0f64;
            let mut at = (0.0f64, 0.0f64);
            let mut worst_val = 0.0f64;
            // 200 points across the bucket, deliberately off-node.
            for i in 0..=200 {
                let s_big = lo + (hi - lo) * (i as f64) / 200.0;
                if s_big <= 0.0 {
                    continue;
                }
                for &s_small in &s_vals {
                    let gi =
                        unsafe { scf_terfc_debug_interp_G(cdir.as_ptr(), s_big, s_small, m, 0) };
                    let gs = unsafe { scf_terfc_debug_series_G(s_big, s_small, m) };
                    if !gi.is_finite() || !gs.is_finite() {
                        eprintln!("NONFINITE at S={s_big} s={s_small} m={m}: {gi} {gs}");
                        continue;
                    }
                    let abs = (gi - gs).abs();
                    let rel = if gs.abs() > 0.0 { abs / gs.abs() } else { 0.0 };
                    if rel > worst_rel {
                        worst_rel = rel;
                        worst_abs = abs;
                        at = (s_big, s_small);
                        worst_val = gs;
                    }
                }
            }
            eprintln!(
                "{name} m={m:2}: worst rel {worst_rel:.3e} (abs {worst_abs:.3e}, G={worst_val:.3e}) at S={:.4} s={:.4}",
                at.0, at.1
            );
        }
    }
}

/// Same sweep for the DECOUPLED seams (`Operator::terfc_with_omega`), where
/// s = phi^2 r0^2 reaches (r0*omega)^2 = 0.5..79 for r0*omega = 0.7..8.9. The
/// error found here sets the relative inflation of the terfc Schwarz factors
/// (`schwarz::TERFC_TABLE_Q_INFLATION`).
#[test]
#[ignore = "benchmark: terfc interpolation sweep on sharp seams; --release --ignored --nocapture"]
fn sweep_interp_vs_series_sharp_seams() {
    let Some(dir) = table_dir() else {
        eprintln!("skip: no tables");
        return;
    };
    let cdir = std::ffi::CString::new(dir.to_str().unwrap()).unwrap();
    let s_vals = [0.5_f64, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 30.0, 50.0, 79.0];
    let mut overall_abs = 0.0f64;
    for &(lo, hi) in &[(0.01_f64, 4.0_f64), (4.0, 10.0), (10.0, 20.0)] {
        for &m in &[0_i32, 1, 2, 4, 7, 10] {
            let (mut worst_rel, mut worst_abs) = (0.0f64, 0.0f64);
            let mut at = (0.0f64, 0.0f64);
            for i in 0..=200 {
                let s_big = lo + (hi - lo) * (i as f64) / 200.0;
                for &s_small in &s_vals {
                    let gi =
                        unsafe { scf_terfc_debug_interp_G(cdir.as_ptr(), s_big, s_small, m, 0) };
                    let gs = unsafe { scf_terfc_debug_series_G(s_big, s_small, m) };
                    if !gi.is_finite() || !gs.is_finite() || gs == 0.0 {
                        continue;
                    }
                    let abs = (gi - gs).abs();
                    worst_abs = worst_abs.max(abs);
                    let rel = abs / gs.abs();
                    if rel > worst_rel {
                        worst_rel = rel;
                        at = (s_big, s_small);
                    }
                }
            }
            overall_abs = overall_abs.max(worst_abs);
            eprintln!(
                "S in ({lo},{hi}] m={m:2}: worst rel {worst_rel:.3e} (worst abs {worst_abs:.3e}) \
                 at S={:.3} s={:.2}",
                at.0, at.1
            );
        }
    }
    eprintln!("OVERALL worst ABSOLUTE interpolation error in G: {overall_abs:.3e}");
}
