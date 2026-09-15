//! How much slower is terfc than erfc, and where does the gap come from?
//!
//! Both are attenuated Coulomb operators used by MP2, but they reach the
//! integrals by structurally different routes:
//!
//!   erfc  -> a libint2-NATIVE operator (`Operator::erfc_coulomb`, shim
//!            op_kind 2). One engine, one pass, no tables.
//!   terfc -> a table-interpolated kernel with NO libint2 operator. Every
//!            block is `compute_cart_eri3` TWICE (Coulomb with Boys, terf with
//!            the tables) plus an element-wise subtraction.
//!
//! So a 1.0x ratio is probably not reachable: terfc pays for a second integral
//! pass that erfc never makes. The point of this harness is to measure the gap
//! honestly, attribute it, and find the floor -- not to assert a target.
//!
//! `#[ignore]`d: timing harness, not a correctness gate. Run with
//! `--ignored --nocapture`, and only on a QUIET box -- this repo has a standing
//! lesson that wall times taken under load are inflated up to 1.6x and that
//! `cpu_s/wall_s` is what discriminates contention from real cost.

use ferric_core::{basis, mol::Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

fn tables_available() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR").is_ok()
}

/// Read this process's CPU time (utime+stime, seconds) so the harness can
/// report cpu/wall. A ratio far below the worker count means the timing was
/// taken under contention and must be discarded.
fn cpu_seconds() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // field 14 = utime, 15 = stime (1-indexed, after the comm field in parens)
    let tail = match s.rfind(')') {
        Some(i) => s[i + 1..].to_string(),
        None => return 0.0,
    };
    let f: Vec<&str> = tail.split_whitespace().collect();
    let tps = 100.0f64; // USER_HZ
    let utime: f64 = f.get(11).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let stime: f64 = f.get(12).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    (utime + stime) / tps
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
}

fn setup(name: &str) -> Sys {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{name}.xyz",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ferric_core::parallel::ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    Sys { mol, obs, dfbs, rhf }
}

/// Time one ri_mp2 call, warm pass first. Returns (wall_s, cpu_s, energy).
fn timed(s: &Sys, op: Operator) -> (f64, f64, f64) {
    let cfg = RiMp2Config::default();
    let _ = ri_mp2(&s.mol, &s.obs, &s.dfbs, op, &s.rhf, &cfg).unwrap();
    let c0 = cpu_seconds();
    let t = Instant::now();
    let e = ri_mp2(&s.mol, &s.obs, &s.dfbs, op, &s.rhf, &cfg).unwrap();
    (t.elapsed().as_secs_f64(), cpu_seconds() - c0, e.mp2_corr)
}

#[test]
#[ignore = "timing harness; run with --ignored --nocapture on a QUIET box"]
fn erfc_vs_terfc_ri_mp2() {
    if !tables_available() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset");
        return;
    }
    let name = std::env::var("FERRIC_BENCH_MOL").unwrap_or_else(|_| "alkane_10".into());
    let s = setup(&name);
    eprintln!(
        "\n=== {name} / cc-pVDZ + cc-pVDZ-RI: nbf={} naux={} ===",
        s.obs.nbasis(),
        s.dfbs.nbasis()
    );
    eprintln!("load1={:.2}", {
        let l = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
        l.split_whitespace()
            .next()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(-1.0)
    });

    // erfc at the dissertation-default omega; terfc at a matched-ish r0.
    let (w_e, c_e, e_e) = timed(&s, Operator::erfc(0.222234));
    eprintln!("erfc(w=0.222234)   wall {w_e:8.3}s  cpu {c_e:8.3}s  cpu/wall {:5.2}  E {e_e:.10}",
              c_e / w_e.max(1e-9));

    for r0 in [1.0_f64, 2.0] {
        let (w_t, c_t, e_t) = timed(&s, Operator::terfc(r0));
        eprintln!(
            "terfc(r0={r0})        wall {w_t:8.3}s  cpu {c_t:8.3}s  cpu/wall {:5.2}  E {e_t:.10}  \
             => terfc/erfc {:6.2}x  (gap {:+.1} %pt)",
            c_t / w_t.max(1e-9),
            w_t / w_e.max(1e-9),
            100.0 * (w_t / w_e.max(1e-9) - 1.0)
        );
    }

    eprintln!(
        "\nNOTE: erfc is a libint2-native single-pass operator; terfc is two \
         compute_cart_eri3 passes (Coulomb + terf tables) plus a subtraction. \
         A 1.0x ratio is structurally unlikely -- the floor is ~2x on integral \
         work alone unless the second pass is eliminated."
    );
}
