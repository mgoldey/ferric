//! Oracle anchors for the terfc FAR-FIELD series path (`terf_G_series` in
//! shim.cc) — the exact code `terf_aux` runs for out-of-table (S,s), i.e.
//! S > 20 (the curvature constraint r0·ω = 1/√2 bounds s = φ²r0² ≤ 1/2, so
//! every table covers the reachable s-range and coverage reduces to S ≤ 20).
//!
//! History: the shim used to SKIP the terf subtraction for out-of-table
//! primitives, leaving full-Coulomb contamination in far-field terfc
//! integrals and making (P|Q)_terfc spuriously indefinite (commit 95aa709).
//! These anchors pin the replacement series against ground truth so the
//! far-field path can never silently regress again.
//!
//! All three tests call the shipped series through the test-only FFI hook
//! `scf_terfc_debug_series_G` (a thin try/catch wrapper around
//! `terf_G_series`), NOT a Rust reimplementation.

// Link the crate (and thereby the shim static library that carries the
// debug hooks); this test calls the C hooks directly.
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

/// Same table-dir gating as terfc_base_validation.rs: env var first, then the
/// in-repo terf-tables/ directory; skip (return None) when tables are absent.
fn table_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("FERRIC_TERF_TABLE_DIR") {
        let p = PathBuf::from(d);
        if p.join("16_4_2.bin").exists() {
            return Some(p);
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo = manifest.parent()?.parent()?.join("terf-tables");
    if repo.join("16_4_2.bin").exists() {
        return Some(repo);
    }
    None
}

fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// Boys function F_m(25) computed in-repo: F_0(T) = √(π/(4T))·erf(√T) with the
/// upward recursion F_{m+1} = ((2m+1)F_m − e^{−T})/(2T), stable for T > m.
/// erf(5) hardcoded from mpmath (`mp.erf(5)`, prec=256).
fn boys_25(mmax: usize) -> Vec<f64> {
    let t = 25.0_f64;
    const ERF_5: f64 = 0.999_999_999_998_462_540_21;
    let mut f = vec![0.0; mmax + 1];
    f[0] = (std::f64::consts::PI / (4.0 * t)).sqrt() * ERF_5;
    let e = (-t).exp();
    for m in 0..mmax {
        f[m + 1] = ((2 * m + 1) as f64 * f[m] - e) / (2.0 * t);
    }
    f
}

/// Anchor 1 — the s→0 identity: the terf kernel degenerates to erf, whose
/// MD auxiliary IS the Boys function, so G_m(S, 0) = F_m(S) exactly
/// (verified to 1e-77 in mpmath during the fix). Pins the series' m-difference
/// structure against an independently computable function. S=25 is in the
/// series-only region (out of table).
#[test]
fn series_g_at_s_zero_equals_boys_s25() {
    // Series path needs no tables; no gating required.
    let f = boys_25(8);
    for &m in &[0usize, 4, 8] {
        let g = unsafe { scf_terfc_debug_series_G(25.0, 0.0, m as c_int) };
        assert!(g.is_finite(), "series G_{m}(25,0) returned NaN");
        let rel = rel_diff(g, f[m]);
        assert!(
            rel < 1e-10,
            "G_{m}(25, 0) = {g:.16e} != F_{m}(25) = {:.16e} (rel {rel:.2e})",
            f[m]
        );
    }
}

/// Anchor 2 — hardcoded mpmath ground truth at s > 0 (the true terf series,
/// not reachable by any Boys identity). Values generated at prec=256 with the
/// table generator's own machinery:
///
/// ```text
/// python3 - <<'EOF'
/// import sys; sys.path.insert(0, "terf-tables")
/// import mpmath as mp; mp.mp.prec = 256
/// from generate_tables import _df_precompute, _build_fd_table, DIMI
/// def G(S, s, m):
///     df = _df_precompute(DIMI)
///     gS = _build_fd_table(mp.mpf(S), m + 2, DIMI)
///     gs = _build_fd_table(mp.mpf(s), 2, DIMI)
///     return sum(df[i] * gS[m+1][i] * gs[0][i] for i in range(DIMI))
/// for (S, s, m) in [(20.5, 0.45, 0), (25.0, 0.3, 4), (32.0, 0.5, 8)]:
///     print(S, s, m, mp.nstr(G(S, s, m), 20))
/// EOF
/// ```
#[test]
fn series_g_matches_mpmath_anchors() {
    // (S, s, m, G_{m,0}(S,s) @ mpmath prec=256)
    const ANCHORS: &[(f64, f64, i32, f64)] = &[
        (20.5, 0.45, 0, 0.195_734_778_844_768_55),
        (25.0, 0.3, 4, 2.977_702_189_458_724_8e-6),
        (32.0, 0.5, 8, 1.128_118_765_692_328_1e-9),
    ];
    for &(s_big, s_small, m, g_ref) in ANCHORS {
        let g = unsafe { scf_terfc_debug_series_G(s_big, s_small, m) };
        assert!(g.is_finite(), "series G_{m}({s_big},{s_small}) returned NaN");
        let rel = rel_diff(g, g_ref);
        assert!(
            rel < 1e-10,
            "G_{m}({s_big}, {s_small}) = {g:.16e} != mpmath {g_ref:.16e} (rel {rel:.2e})"
        );
    }
}

/// Anchor 3 — seam continuity at the table boundary S = 20. In production,
/// terf_aux uses poly-10 table interpolation for S ≤ 20 and the series for
/// S > 20; the jump at the seam is bounded by their disagreement where both
/// are defined. Compare the two SHIPPED evaluators at the same points just
/// inside the boundary (and at 19.5 as an interior control).
///
/// Tolerance: relative < 1e-9 OR absolute < 1e-13. The absolute escape covers
/// high-m entries on the coarse 4_20_20 table, where poly-10 interpolation of
/// float64 table data bottoms out near the data's own precision floor
/// (measured: m=8 at S=19.5 agrees to 3e-15 ABSOLUTE = 4e-8 relative on a
/// 7.5e-8 value — table-data noise, not a seam artifact; the historical
/// far-field bug this guards against produced ~1e-1 absolute errors).
/// Requires tables (interp side) — gated like terfc_base_validation.rs.
#[test]
fn series_matches_table_interp_at_seam() {
    let Some(dir) = table_dir() else {
        eprintln!("skipping: terfc tables not found (set FERRIC_TERF_TABLE_DIR)");
        return;
    };
    let cdir = std::ffi::CString::new(dir.to_str().unwrap()).unwrap();
    for &s_big in &[19.5_f64, 19.99] {
        for &s_small in &[0.1_f64, 0.45] {
            for &m in &[0_i32, 4, 8] {
                let gi =
                    unsafe { scf_terfc_debug_interp_G(cdir.as_ptr(), s_big, s_small, m, 0) };
                let gs = unsafe { scf_terfc_debug_series_G(s_big, s_small, m) };
                assert!(
                    gi.is_finite() && gs.is_finite(),
                    "NaN at S={s_big} s={s_small} m={m}: interp={gi} series={gs}"
                );
                let rel = rel_diff(gi, gs);
                let abs = (gi - gs).abs();
                assert!(
                    rel < 1e-9 || abs < 1e-13,
                    "seam mismatch at S={s_big} s={s_small} m={m}: \
                     interp={gi:.16e} series={gs:.16e} (rel {rel:.2e}, abs {abs:.2e})"
                );
            }
        }
    }
    // And the series must remain finite/smooth just outside the seam (the
    // production-only region): monotone-decreasing G_0 across the boundary.
    let g_in = unsafe { scf_terfc_debug_series_G(19.99, 0.45, 0) };
    let g_out = unsafe { scf_terfc_debug_series_G(20.01, 0.45, 0) };
    assert!(
        g_out.is_finite() && g_out < g_in && rel_diff(g_in, g_out) < 1e-3,
        "series not smooth across S=20: G(19.99)={g_in:.12e} G(20.01)={g_out:.12e}"
    );
}
