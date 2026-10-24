//! #234: the libint2-native terf engine must refuse an angular momentum whose
//! Boys-like order reaches the terf table depth, instead of reading past the
//! table rows and overflowing the series buffer.
//!
//! libint2 asks the core evaluator for m up to `braket_rank * max_l` (the
//! engine here is deriv_order 0), and the tables hold m < 24 (`TERFC_DIMM`).
//! Only builds under the `libint2_terf` feature against a terf-patched libint2
//! with FERRIC_LIBINT2_TERF set, like the other terf_libint2_* tests.
#![cfg(feature = "libint2_terf")]

use ferric_core::basis::{self, BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::{Engine, TerfLibint2Braket};
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use std::collections::HashMap;
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
        for max_l in [lim + 1, lim + 2, 40] {
            let err = Engine::new_terf_libint2(r0, omega, bk, 3, max_l, 0.0, Some(&dir))
                .expect_err(&format!("{bk:?} max_l={max_l} must be refused"));
            let msg = err.to_string();
            assert!(msg.contains("refused max_l"), "{bk:?} max_l={max_l}: {msg}");
        }
    }
    let err = Engine::new_terf_libint2(r0, omega, TerfLibint2Braket::XxXx, 3, -1, 0.0, Some(&dir))
        .unwrap_err()
        .to_string();
    assert!(err.contains("max_l must be >= 0"), "{err}");
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
    // The terf limit can sit above the AM the linked libint2 was generated
    // for (xs_xs: 11 vs 6 or 7). Accept up to the smaller of the two.
    for bk in BRAKETS {
        let top = bk.max_l().min(bk.libint_max_l());
        assert!(top >= 2, "{bk:?}: libint2 limit {}", bk.libint_max_l());
        for max_l in [0, top] {
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

#[test]
fn engine_refuses_max_l_past_the_libint_build() {
    // Below the terf limit but past what libint2 was generated for: its own
    // status and message, not libint2's lmax_exceeded as SCF_EINTERNAL.
    let dir = table_dir();
    let r0 = 2.0_f64;
    let omega = 1.0 / (r0 * 2.0_f64.sqrt());
    let mut seen = 0;
    for bk in BRAKETS {
        let l = bk.libint_max_l() + 1;
        if l > bk.max_l() {
            continue;
        }
        seen += 1;
        let err = Engine::new_terf_libint2(r0, omega, bk, 3, l, 0.0, Some(&dir))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("generated for max_l <="),
            "{bk:?} max_l={l}: {err}"
        );
    }
    // xs_xs (terf limit 11) is past every libint2 build ferric uses (6 or 7).
    assert!(seen >= 1);
}

#[test]
fn engine_refuses_bad_omega_r0_and_nprim() {
    let dir = table_dir();
    let bk = TerfLibint2Braket::XsXx;
    for omega in [0.0, -0.5, f64::NAN, f64::INFINITY] {
        let err = Engine::new_terf_libint2(2.0, omega, bk, 3, 2, 0.0, Some(&dir))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("omega must be finite and > 0"),
            "omega={omega}: {err}"
        );
    }
    let err = Engine::new_terf_libint2(f64::NAN, 0.35, bk, 3, 2, 0.0, Some(&dir))
        .unwrap_err()
        .to_string();
    assert!(err.contains("r0 must be finite"), "{err}");
    let err = Engine::new_terf_libint2(2.0, 0.35, bk, 0, 2, 0.0, Some(&dir))
        .unwrap_err()
        .to_string();
    assert!(err.contains("max_nprim must be >= 1"), "{err}");
    // The shim enforces the same on its own (the Rust checks above would mask
    // a missing shim check): a raw call with omega <= 0 / NaN gets SCF_EINVAL.
    // A valid engine first: it initializes libint2, which the raw constructor
    // below does not do (libint2 would assert instead of the check firing).
    Engine::new_terf_libint2(2.0, 0.35, bk, 3, 2, 0.0, Some(&dir)).unwrap();
    let cdir = CString::new(dir).unwrap();
    for (r0, omega) in [(2.0, 0.0), (2.0, -1.0), (2.0, f64::NAN), (f64::NAN, 0.35)] {
        let mut st: c_int = 0;
        // SAFETY: valid C string and out-param; a non-null handle is destroyed.
        let h = unsafe {
            ffi::scf_engine_create_terf_libint2(r0, omega, 3, 3, 2, 0.0, cdir.as_ptr(), &mut st)
        };
        if !h.is_null() {
            unsafe { ffi::scf_engine_destroy(h) };
        }
        assert!(
            h.is_null() && st == ffi::SCF_EINVAL,
            "r0={r0} omega={omega}: status {st}"
        );
    }
}

#[test]
fn hook_refuses_nonpositive_omega_and_rho() {
    // libint2 zero-fills Gm when the hook returns false, so these must be
    // refusals (SCF_EINVAL through the probe), never a quiet `false`.
    let cdir = CString::new(table_dir()).unwrap();
    for (rho, omega) in [
        (1.0, 0.0),
        (1.0, -0.3),
        (1.0, f64::NAN),
        (0.0, 0.35),
        (-1.0, 0.35),
    ] {
        let mut gm = vec![0.0_f64; 4];
        // SAFETY: valid C string; gm holds mmax + 1 = 4 doubles.
        let st = unsafe {
            ffi::scf_terf_libint2_hook_probe(
                cdir.as_ptr(),
                rho,
                10.0,
                3,
                omega,
                2.0,
                gm.as_mut_ptr(),
                4,
            )
        };
        assert_eq!(st, ffi::SCF_EINVAL, "rho={rho} omega={omega}");
    }
}

/// One single-primitive pure shell of every l in 0..=lmax on each of two
/// atoms 1.4 A apart: quartets over these reach m = 4*lmax, the band
/// water/cc-pVDZ (l <= 2, m <= 8) never touches.
fn high_l_basis(lmax: i32) -> (Molecule, BasisSet) {
    let mol = Molecule::parse_xyz("2\n\nH 0 0 0\nH 0 0 1.4\n", 0, 1).unwrap();
    let shells = (0..=lmax)
        .map(|l| Shell {
            l,
            pure: true,
            exponents: vec![0.9 + 0.3 * f64::from(l)],
            coefficients: vec![1.0],
        })
        .collect();
    let mut map = HashMap::new();
    map.insert(1, shells);
    (
        mol,
        BasisSet {
            name: "terf-high-l".into(),
            shells: map,
            ecps: HashMap::new(),
        },
    )
}

/// Running comparison: max |x - y| and max |x|,|y| over a whole sweep, plus
/// the highest m among nonzero blocks.
#[derive(Default)]
struct Sweep {
    max_diff: f64,
    scale: f64,
    top_m: i32,
    blocks: usize,
}

impl Sweep {
    fn add(&mut self, m: i32, x: &[f64], y: &[f64]) {
        let n = x.len().max(y.len());
        for i in 0..n {
            let (a, b) = (
                x.get(i).copied().unwrap_or(0.0),
                y.get(i).copied().unwrap_or(0.0),
            );
            self.max_diff = self.max_diff.max((a - b).abs());
            self.scale = self.scale.max(a.abs()).max(b.abs());
        }
        if n > 0 {
            self.top_m = self.top_m.max(m);
        }
        self.blocks += 1;
    }

    fn check(&self, what: &str, want_top_m: i32) {
        let rel = self.max_diff / self.scale;
        eprintln!(
            "{what}: {} blocks, top m {}, max |diff| {:.3e} of sweep scale {:.3e} ({rel:.3e})",
            self.blocks, self.top_m, self.max_diff, self.scale
        );
        assert_eq!(
            self.top_m, want_top_m,
            "{what}: the top of the m band was never compared"
        );
        assert!(
            rel < 1e-12,
            "{what}: libint2 terf disagrees with MD by {rel:.3e} of the sweep scale"
        );
    }
}

/// The high-m band (m up to 4*5 = 20 for quartets, 3*l for triplets) against
/// the hand-written MD terf path, at the largest l both the terf tables and
/// the linked libint2 allow.
///
/// The bar is the absolute difference over the SWEEP's largest integral, not
/// per block. Measured (2.13.1, l <= 5): the two recurrences agree to an
/// absolute floor of ~1e-14 of the sweep scale whatever the block size, so a
/// ~1e-7 block shows a per-block "relative" error up to ~4e-8 that is the
/// same roundoff floor (recurrence order, as in the 1e-10 water gate). A wrong
/// high-m kernel value errs in proportion to the block it feeds instead,
/// which this bar still catches for any block above ~1e-11 of the scale;
/// checked by perturbing the hook's m >= 9 output by 1e-4 (this test fails;
/// 1e-6 sits under the roundoff floor and passes, so the sensitivity is
/// ~1e-5 relative in the high-m G_m, far below any real kernel defect).
#[test]
fn high_l_blocks_match_the_md_path() {
    let dir = table_dir();
    let r0 = 2.0_f64;
    let omega = 1.0 / (r0 * 2.0_f64.sqrt());
    let l_of = |obs: &PreparedBasis, i: usize| obs.shell_dims()[i] as i32 / 2; // pure: 2l+1

    // 4-center.
    let bk = TerfLibint2Braket::XxXx;
    let l4 = bk.max_l().min(bk.libint_max_l());
    let (mol, bs) = high_l_basis(l4);
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let mut li = Engine::new_terf_libint2(r0, omega, bk, 1, l4, 0.0, Some(&dir)).unwrap();
    let mut md = Engine::new_2e(Operator::terf(r0), &obs, 0.0).unwrap();
    let nsh = obs.shell_dims().len();
    let mut sw = Sweep::default();
    for a in 0..nsh {
        for b in 0..=a {
            for c in 0..nsh {
                for d in 0..=c {
                    let m = l_of(&obs, a) + l_of(&obs, b) + l_of(&obs, c) + l_of(&obs, d);
                    if m < 9 && (a + b + c + d) % 5 != 0 {
                        continue; // keep the run short; the low band is covered above
                    }
                    let x = li
                        .compute_quartet_checked(&obs, a, b, c, d)
                        .unwrap()
                        .map(<[f64]>::to_vec);
                    let y = md
                        .compute_quartet_checked(&obs, a, b, c, d)
                        .unwrap()
                        .map(<[f64]>::to_vec);
                    sw.add(m, &x.unwrap_or_default(), &y.unwrap_or_default());
                }
            }
        }
    }
    sw.check(&format!("xx_xx l<={l4}"), 4 * l4);

    // 3-center.
    let bk = TerfLibint2Braket::XsXx;
    let l3 = bk.max_l().min(bk.libint_max_l());
    let (mol, bs) = high_l_basis(l3);
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let mut li = Engine::new_terf_libint2(r0, omega, bk, 1, l3, 0.0, Some(&dir)).unwrap();
    let mut md = Engine::new_3center(Operator::terf(r0), &obs, &obs, 0.0).unwrap();
    let nsh = obs.shell_dims().len();
    let mut sw = Sweep::default();
    for p in 0..nsh {
        for a in 0..nsh {
            for b in 0..=a {
                let m = l_of(&obs, p) + l_of(&obs, a) + l_of(&obs, b);
                let x = li
                    .compute_eri3_checked(&obs, &obs, p, a, b)
                    .unwrap()
                    .map(<[f64]>::to_vec);
                let y = md.compute_eri3(&obs, &obs, p, a, b).map(<[f64]>::to_vec);
                sw.add(m, &x.unwrap_or_default(), &y.unwrap_or_default());
            }
        }
    }
    sw.check(&format!("xs_xx l<={l3}"), 3 * l3);
}
