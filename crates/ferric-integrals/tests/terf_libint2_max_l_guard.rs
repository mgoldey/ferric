//! #234: the libint2-native terf engine must refuse an angular momentum whose
//! Boys-like order reaches the terf table depth, instead of reading past the
//! table rows and overflowing the series buffer.
//!
//! libint2 asks the core evaluator for m up to `braket_rank * max_l` (the
//! engine here is deriv_order 0), and the tables hold m < 24 (`TERFC_DIMM`).
//! Only builds under the `libint2_terf` feature against a terf-patched libint2
//! with FERRIC_LIBINT2_TERF set, like the other terf_libint2_* tests.
#![cfg(feature = "libint2_terf")]

use ferric_core::{basis, mol::Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::{Engine, TerfLibint2Braket};
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use std::ffi::CString;
use std::os::raw::c_int;

/// Table depth (`TERFC_DIMM` in shim.cc), restated so the limits below are
/// checked against an independent number rather than read back from the shim.
const TABLE_DEPTH: i32 = 24;
const BRAKETS: [TerfLibint2Braket; 3] = [
    TerfLibint2Braket::XsXs,
    TerfLibint2Braket::XsXx,
    TerfLibint2Braket::XxXx,
];

/// The terf tables: FERRIC_TERF_TABLE_DIR, else the repo's terf-tables/.
/// Exported once (every test calls this first) so the MD engine, which only
/// reads the env var, finds the same tables without a racing set_var.
fn table_dir() -> String {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
            std::env::set_var(
                "FERRIC_TERF_TABLE_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/../../terf-tables"),
            );
        }
    });
    std::env::var("FERRIC_TERF_TABLE_DIR").unwrap()
}

#[test]
fn max_l_limit_is_the_last_l_below_the_table_depth() {
    // xs_xs: 2*11 = 22, xs_xx: 3*7 = 21, xx_xx: 4*5 = 20; one more L reaches 24.
    let expected = [
        (TerfLibint2Braket::XsXs, 11),
        (TerfLibint2Braket::XsXx, 7),
        (TerfLibint2Braket::XxXx, 5),
    ];
    for (bk, lim) in expected {
        assert_eq!(bk.max_l(), lim, "{bk:?}");
        assert!(bk.rank() * lim < TABLE_DEPTH, "{bk:?}");
        assert!(bk.rank() * (lim + 1) >= TABLE_DEPTH, "{bk:?}");
    }
}

#[test]
fn engine_refuses_max_l_past_the_table_depth() {
    let dir = table_dir();
    let r0 = 2.0_f64;
    let omega = 1.0 / (r0 * 2.0_f64.sqrt());
    for bk in BRAKETS {
        let lim = bk.max_l();
        for max_l in [lim + 1, lim + 2, 12, -1] {
            let err = Engine::new_terf_libint2(r0, omega, bk, 3, max_l, 0.0, Some(&dir))
                .expect_err(&format!("{bk:?} max_l={max_l} must be refused"));
            let msg = err.to_string();
            assert!(msg.contains("refused max_l"), "{bk:?} max_l={max_l}: {msg}");
        }
    }
    // The 4-center case from the issue: (ii|ii) at max_L = 6 asks for m = 24.
    let err = Engine::new_terf_libint2(r0, omega, TerfLibint2Braket::XxXx, 3, 6, 0.0, Some(&dir))
        .unwrap_err()
        .to_string();
    assert!(err.contains("4*max_l = 24"), "{err}");
}

#[test]
fn engine_accepts_max_l_at_the_limit() {
    let dir = table_dir();
    let r0 = 2.0_f64;
    let omega = 1.0 / (r0 * 2.0_f64.sqrt());
    // The xs_xs limit (11) sits above the AM this libint2 was generated for
    // (7 for 2-center ERIs here), which libint2 itself refuses with a clean
    // lmax_exceeded error -- not a terf concern. So check each braket up to
    // the smaller of its terf limit and 7.
    for bk in BRAKETS {
        for max_l in [0, bk.max_l().min(7)] {
            Engine::new_terf_libint2(r0, omega, bk, 3, max_l, 0.0, Some(&dir))
                .unwrap_or_else(|e| panic!("{bk:?} max_l={max_l} must be accepted: {e}"));
        }
    }
}

#[test]
fn hook_refuses_mmax_at_the_table_depth() {
    // The second line of defence: even if an over-deep mmax reached the hook,
    // it must refuse rather than read past the table rows (table point) or
    // overflow its series buffer (series point, S > 20).
    let cdir = CString::new(table_dir()).unwrap();
    let r0 = 2.0_f64;
    let omega = 1.0 / (r0 * 2.0_f64.sqrt());
    // rho = 1: phi2 = 1/9, S = T/9, s = 4/9. T = 10 -> table, T = 400 -> series.
    for t in [10.0_f64, 400.0] {
        for (mmax, want) in [
            (0, 0),
            (TABLE_DEPTH - 1, 0),
            (TABLE_DEPTH, ffi::SCF_EINVAL),
            (40, ffi::SCF_EINVAL),
        ] {
            let mut gm = vec![0.0_f64; mmax as usize + 1];
            // SAFETY: valid C string; gm holds mmax + 1 doubles, as declared.
            let st = unsafe {
                ffi::scf_terf_libint2_hook_probe(
                    cdir.as_ptr(),
                    1.0,
                    t,
                    mmax as c_int,
                    omega,
                    r0,
                    gm.as_mut_ptr(),
                    gm.len() as c_int,
                )
            };
            assert_eq!(st, want, "T={t} mmax={mmax}");
            if st == 0 {
                // G_m of the tempered kernel changes sign with m; only finiteness
                // and a nonzero leading term are generic.
                assert!(
                    gm.iter().all(|g| g.is_finite()) && gm[0] > 0.0,
                    "T={t} mmax={mmax}: {gm:?}"
                );
            }
        }
    }
}

/// A valid max_l still works: the libint2 engine at the 4-center limit matches
/// the hand-rolled MD terf path on water/cc-pVDZ to the same 1e-10 bar the
/// existing 3-center gate (terf_libint2_vs_md.rs) uses.
#[test]
fn valid_max_l_quartets_match_the_md_path() {
    let dir = table_dir();
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/water.xyz"
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let r0 = 2.0_f64;
    let omega = 1.0 / (r0 * 2.0_f64.sqrt());
    let mut li = Engine::new_terf_libint2(
        r0,
        omega,
        TerfLibint2Braket::XxXx,
        obs.max_nprim() as usize,
        TerfLibint2Braket::XxXx.max_l(),
        0.0,
        Some(&dir),
    )
    .unwrap();
    let mut md = Engine::new_2e(Operator::terf(r0), &obs, 0.0).unwrap();
    let nsh = obs.shell_dims().len();
    let (mut worst, mut checked, mut fp) = (0.0_f64, 0usize, 0u64);
    for a in 0..nsh {
        for b in 0..=a {
            for c in (0..nsh).step_by(3) {
                for d in (0..=c).step_by(2) {
                    let x = li
                        .compute_quartet(&obs, a, b, c, d)
                        .map(<[f64]>::to_vec)
                        .unwrap_or_default();
                    let y = md
                        .compute_quartet(&obs, a, b, c, d)
                        .map(<[f64]>::to_vec)
                        .unwrap_or_default();
                    let n = x.len().max(y.len());
                    let scale = x.iter().chain(&y).fold(0.0_f64, |m, v| m.max(v.abs()));
                    for i in 0..n {
                        let (xi, yi) = (
                            x.get(i).copied().unwrap_or(0.0),
                            y.get(i).copied().unwrap_or(0.0),
                        );
                        fp = fp.rotate_left(5) ^ xi.to_bits();
                        if scale > 1e-12 {
                            worst = worst.max((xi - yi).abs() / scale);
                        }
                    }
                    checked += 1;
                }
            }
        }
    }
    eprintln!("libint2 terf xx_xx vs MD: {checked} quartets, worst rel {worst:.3e}, bits fingerprint {fp:016x}");
    assert!(checked > 50, "only {checked} quartets");
    assert!(
        worst < 1e-10,
        "libint2 terf quartets disagree with MD by {worst:.3e}"
    );
}
