//! STEP 5: does libint2-native terf actually beat the hand-rolled MD driver?
//!
//! Baselines to beat (decane/cc-pVDZ+RI, serial, 3-index block):
//!   libint2 Coulomb 1.358 s | libint2 erfc 2.005 s (1.48x)
//!   MD Coulomb 34.46 s (25.4x) | MD terf r0=2 74.70 s (55.0x)
//!
//! Success = terf approaching erfc's 1.48x. Anything under ~3x is a large win.
#![cfg(feature = "libint2_terf")]

use ferric_core::{basis, mol::Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::{engine::Engine, ffi, operator::Operator};
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::time::Instant;

fn load1() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|l| l.split_whitespace().next().and_then(|v| v.parse().ok()))
        .unwrap_or(-1.0)
}

/// Full (P|s1 s2) sweep through the libint2-native terf engine.
fn sweep_libint2(prep: &PreparedBasis, dfbs: &PreparedBasis, r0: f64, omega: f64) -> f64 {
    let dir = std::env::var("FERRIC_TERF_TABLE_DIR").unwrap();
    let cdir = CString::new(dir).unwrap();
    // SAFETY: valid metadata; handle destroyed below.
    let h = unsafe {
        ffi::scf_engine_create_terf_libint2(
            // lmax/max_nprim must span BOTH bases -- a 3-center engine sees
            // aux shells too, and the aux set goes higher than obs here.
            // Sizing from obs alone aborts libint2 with
            // "the angular momentum limit is exceeded". Matches Engine::new_3center.
            r0,
            omega,
            3,
            prep.max_nprim().max(dfbs.max_nprim()),
            prep.max_l().max(dfbs.max_l()),
            0.0,
            cdir.as_ptr() as *const c_char,
        )
    };
    assert!(!h.is_null(), "libint2 terf engine construction failed");
    let dims_o = prep.shell_dims();
    let dims_d = dfbs.shell_dims();
    let maxn = dims_d.iter().max().unwrap() * dims_o.iter().max().unwrap().pow(2);
    let mut buf = vec![0.0f64; maxn];
    let mut acc = 0.0f64;
    let t = Instant::now();
    for p in 0..dims_d.len() {
        for s1 in 0..dims_o.len() {
            for s2 in 0..=s1 {
                // SAFETY: in-bounds shells; buf sized for the largest block.
                let n = unsafe {
                    ffi::scf_compute_eri3(
                        h,
                        prep.handle(),
                        dfbs.handle(),
                        p as c_int,
                        s1 as c_int,
                        s2 as c_int,
                        buf.as_mut_ptr(),
                    )
                };
                if n > 0 {
                    acc += buf[0];
                }
            }
        }
    }
    let dt = t.elapsed().as_secs_f64();
    // SAFETY: handle from the matching constructor, not used after this.
    unsafe { ffi::scf_engine_destroy(h) };
    std::hint::black_box(acc);
    dt
}

/// Same sweep through an ordinary ferric Engine (MD driver for terf).
fn sweep_md(prep: &PreparedBasis, dfbs: &PreparedBasis, op: Operator) -> f64 {
    let mut eng = Engine::new_3center(op, prep, dfbs, 0.0).expect("3c engine");
    let dims_o = prep.shell_dims();
    let dims_d = dfbs.shell_dims();
    let mut acc = 0.0f64;
    let t = Instant::now();
    for p in 0..dims_d.len() {
        for s1 in 0..dims_o.len() {
            for s2 in 0..=s1 {
                if let Some(b) = eng.compute_eri3(prep, dfbs, p, s1, s2) {
                    acc += b[0];
                }
            }
        }
    }
    let dt = t.elapsed().as_secs_f64();
    std::hint::black_box(acc);
    dt
}

#[test]
#[ignore = "timing; run with --ignored --nocapture on a QUIET box"]
fn libint2_terf_vs_md_timing() {
    let Ok(_) = std::env::var("FERRIC_TERF_TABLE_DIR") else {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset");
        return;
    };
    let name = std::env::var("FERRIC_BENCH_MOL").unwrap_or_else(|_| "alkane_10".into());
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{name}.xyz",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obsname = std::env::var("FERRIC_BENCH_BASIS").unwrap_or_else(|_| "cc-pvdz".into());
    let auxname = std::env::var("FERRIC_BENCH_AUX").unwrap_or_else(|_| "cc-pvdz-ri".into());
    let obs = PreparedBasis::new(&mol, &basis::bundled(&obsname).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(&auxname).unwrap()).unwrap();
    let r0: f64 = std::env::var("FERRIC_BENCH_R0")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.0);
    let omega = 1.0 / (r0 * 2.0_f64.sqrt());

    eprintln!(
        "\n=== {name} / {obsname}+{auxname} r0={r0}: nbf={} naux={}  load1={:.2} ===",
        obs.nbasis(),
        dfbs.nbasis(),
        load1()
    );

    // Interleaved, minima of 3 -- the box drifts, so alternate every arm.
    let (mut c, mut e, mut md, mut li) = (f64::MAX, f64::MAX, f64::MAX, f64::MAX);
    let rounds: usize = std::env::var("FERRIC_BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    for _ in 0..rounds {
        c = c.min(sweep_md(&obs, &dfbs, Operator::coulomb()));
        e = e.min(sweep_md(&obs, &dfbs, Operator::erfc(0.222234)));
        md = md.min(sweep_md(&obs, &dfbs, Operator::terf(r0)));
        li = li.min(sweep_libint2(&obs, &dfbs, r0, omega));
    }
    eprintln!("  libint2 coulomb   {c:8.3}s   1.00x");
    eprintln!("  libint2 erfc      {e:8.3}s  {:5.2}x", e / c);
    eprintln!("  MD terf (today)   {md:8.3}s  {:5.2}x", md / c);
    eprintln!(
        "  libint2 terf NEW  {li:8.3}s  {:5.2}x   <-- speedup vs MD: {:.1}x",
        li / c,
        md / li
    );
    eprintln!("  load1 after={:.2}", load1());
}
