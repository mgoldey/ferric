//! libint2 2.7.2's `TennoGmEval` (the evaluator behind `Operator::yukawa` and
//! `Operator::slater_geminal`) indexed its Chebyshev table out of range at the
//! table edges (issue #436).
//!
//! ROOT CAUSE. `interpolate_Gm` picks the table cell with
//! `Tint = floor(log2(T))` (T-octave, valid 0..=9: the last octave is
//! [512, 1024]) and `Uint = floor(log10(U / Umin))` (U-decade, valid 0..=9:
//! the last is [1e2, 1e3]), with no guard. `eval_yukawa` / `eval_slater`
//! interpolate for `T <= Tmax = 1024` and U up to `Umax = 1000`, so
//! `T == 1024` exactly (and the doubles just below it whose log2 rounds to
//! 10) gives `Tint == 10`, and `U == 1000` (or the doubles within ~2 ulp below,
//! whose `log10(U / Umin)` rounds to 10) gives `Uint == 10`. Both index
//! `one_over_2K[10]` / `one_over_10K[10]` (stack arrays of 10) and the
//! coefficient block `c_[interval * ...]` one cell past the table. AddressSanitizer
//! reports a stack-buffer-overflow; a non-NDEBUG build aborts on the `assert`.
//! 2.7.2 has no runtime `U > Umax` check either (assert only).
//!
//! A second, independent defect sits in the same corner of the domain and is
//! fixed by the same override: the explicit `G_0` / `G_{-1}` evaluation forms
//! `exp(x*x - T) * erfc(x)`, which for `x` above ~26.5 is `inf * 0 = NaN`
//! (it overflows once `U + 2 sqrt(U T) > ~709`). So `Gm[0]` was NaN across the
//! whole upper part of the nominal domain, e.g. at `U = 1000` for any `T`.
//! The edge cannot be tested against a finite closed form without fixing it.
//!
//! ARTIFACT vs PHYSICS hypothesis (written before measuring). Physics: the
//! Yukawa core integral at (T = 1024, U) is a smooth, tiny, positive number
//! and varies continuously across T = 1024; at U = 1000 it is ~1.8e-4 and
//! varies continuously. Artifact: an out-of-table read returns a value that
//! depends on stack/heap contents (changes with `MALLOC_PERTURB_`, with the
//! optimiser) and is discontinuous in T or U at the edge, or NaN.
//!
//! Tests here:
//! * `reference_quadrature_is_anchored`: the independent reference (composite
//!   Gauss-Legendre of the DEFINING integral
//!   `G_m = int_0^1 t^2m exp(U(1 - t^-2) - T t^2) dt`) against three
//!   high-precision (mpmath, 30 digits) values. A wrong reference would fail
//!   here, before it is used on the edge.
//! * `evaluator_matches_the_defining_integral_at_the_table_edges`: the direct
//!   evaluator (`tenno_gm`) at `T`/`U` exactly on, one ulp either side of, and
//!   away from the edges, both kernels, against the reference.
//! * `engine_u_edge_matches_the_fourier_closed_form`: `(P|AA)` with integer
//!   data that makes `U == 1000.0` and `T == 10.0` exactly under any
//!   rounding, through the real engine, against a Fourier-space quadrature
//!   (independent of libint2's `Gm`).
//! * `engine_t_edge_is_continuous_and_matches_the_erfc_closed_form`: `(P|AA)`
//!   with `T == 1024.0` exactly.
//! * `u_above_umax_is_rejected_only_below_tmax`: on the 2.7.2 override
//!   `U > Umax` with `T <= Tmax` is an `Err`, with `T > Tmax` a finite 0.
//! * `large_u_plus_large_t_is_finite_on_the_override`: `T = 2000, U = 100` (an
//!   upward-recursion overflow, `U + 2 sqrt(U T) > 709`) is 0 on both overrides
//!   (stock libint 2.13.1 returns NaN there, issue #439).
//! * `nan_region_grid_is_finite_and_vanishingly_small`: every `T > Tmax` grid
//!   point with `U + 2 sqrt(U T) > 725` (the stock-NaN region of `eval_urr`,
//!   2.7.2 and 2.13.1 alike) is finite and below 1e-200, both kernels.
//! * `engine_far_pair_is_finite_on_the_override`: exact-integer engine
//!   geometries whose `U + zeta R` exceeds 709 (R = 70 Bohr at zeta = 10,
//!   R = 40 at zeta = 20): finite and vanishingly small, not NaN.
//! * `sweep_is_independent_of_allocator_fill`: a hashed sweep of everything
//!   above in a child process under `MALLOC_PERTURB_` 1, 170 and 2 (255 and 0
//!   do not exercise uninitialised reads).
//!
//! WHICH LIBINT. The edge defects are 2.7.2-specific (2.13.1 guards the
//! indices and falls back to a Gaussian fit above `Umax`), so on 2.13.1 these
//! tests are version-independent regressions that must also pass. The
//! `exp(x*x - T) * erfc(x)` overflow of `eval_urr` (`T > Tmax`) is in BOTH
//! stock versions; both ferric overrides fix it (#436 for 2.7.2, #439 for
//! 2.13.1), and a libint2 build with neither override (stock 2.13.1, distro
//! builds) skips or excuses the NaN slots. The 2.7.2
//! build is exercised with `LIBINT2_PREFIX=$HOME/.local` and a private
//! `CARGO_TARGET_DIR` (see `libint2_overrides/README.md`).

use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::{Atom, Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::{tenno_gm, Engine, TennoKind};
use ferric_integrals::operator::Operator;
use std::collections::HashMap;
use std::f64::consts::PI;

/// `Some("2.7.2" | "2.13.1")` when build.rs compiled that version's patched
/// `boys.h` into the shim, `None` for a stock header.
const BOYS_OVERRIDE: Option<&str> = option_env!("FERRIC_LIBINT2_BOYS_OVERRIDE");

/// libint2 2.7.2 WITHOUT the override aborts (assert) or reads out of range at
/// the table edges, so the edge tests cannot run on it: skip with a message.
fn skip_unpatched_272() -> bool {
    if ferric_integrals::libint_version() == "2.7.2" && BOYS_OVERRIDE != Some("2.7.2") {
        eprintln!(
            "SKIP: libint2 2.7.2 without the ferric boys.h override (issue #436 not fixed in this build)"
        );
        return true;
    }
    false
}

extern "C" {
    fn erfc(x: f64) -> f64;
}

// ------------------------------------------------------------ the reference

/// `n`-point Gauss-Legendre nodes and weights on [-1, 1] (Newton on P_n).
fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-16 {
                break;
            }
        }
        out.push((x, 2.0 / ((1.0 - x * x) * dp * dp)));
    }
    out
}

/// `G_m(T, U) = int_0^1 t^(2m) exp(U (1 - t^-2) - T t^2) dt` by composite
/// Gauss-Legendre: uniform panels on [0, 0.99] and geometric panels
/// approaching t = 1, where the `U (1 - t^-2)` factor makes a boundary layer of
/// width ~ 1 / (2 U). The integrand is smooth and positive.
fn gm_reference(m: usize, t_arg: f64, u_arg: f64) -> f64 {
    let gl = gauss_legendre(16);
    let f = |t: f64| {
        if t <= 0.0 {
            return 0.0;
        }
        t.powi(2 * m as i32) * (u_arg * (1.0 - 1.0 / (t * t)) - t_arg * t * t).exp()
    };
    let mut edges: Vec<f64> = (0..=1980).map(|i| 0.99 * f64::from(i) / 1980.0).collect();
    let mut w = 0.01;
    while w > 1e-13 {
        edges.push(1.0 - w);
        w *= 0.5;
    }
    edges.push(1.0);
    edges.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    let mut total = 0.0;
    for pair in edges.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let (c, h) = (0.5 * (a + b), 0.5 * (b - a));
        total += gl.iter().map(|&(x, wt)| wt * f(c + h * x)).sum::<f64>() * h;
    }
    total
}

/// The value `eval_yukawa` / `eval_slater` should return for output slot `m`.
fn reference(kind: TennoKind, m: usize, t: f64, u: f64, zeta_over_two_rho: f64) -> f64 {
    match kind {
        TennoKind::Yukawa => gm_reference(m, t, u),
        TennoKind::Slater => {
            let gm1 = if m == 0 {
                // G_{-1}: int_0^1 t^-2 exp(...) dt (finite for U > 0)
                gm_minus_one(t, u)
            } else {
                gm_reference(m - 1, t, u)
            };
            (gm1 - gm_reference(m, t, u)) * zeta_over_two_rho
        }
    }
}

fn gm_minus_one(t_arg: f64, u_arg: f64) -> f64 {
    // int_0^1 t^-2 exp(U(1 - t^-2) - T t^2) dt = int_1^inf exp(U(1 - s^2) - T / s^2) ds
    // with s = 1/t; substitute s = 1 + v, integrate v over [0, v_max].
    let gl = gauss_legendre(16);
    let f = |v: f64| {
        let s = 1.0 + v;
        (u_arg * (1.0 - s * s) - t_arg / (s * s)).exp()
    };
    // exp(-U (s^2 - 1)) < e^-45 beyond s = sqrt(1 + 45 / U)
    let v_max = (1.0 + 45.0 / u_arg).sqrt();
    let mut edges = vec![0.0];
    let mut w = 1e-13;
    while w < v_max {
        edges.push(w);
        w *= 1.3;
    }
    edges.push(v_max);
    let mut total = 0.0;
    for pair in edges.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let (c, h) = (0.5 * (a + b), 0.5 * (b - a));
        total += gl.iter().map(|&(x, wt)| wt * f(c + h * x)).sum::<f64>() * h;
    }
    total
}

/// Three 30-digit mpmath values of the defining integral (the closed form
/// `sqrt(pi)/(4 sqrt T) (e^{k^2-T} erfc k - e^{l^2-T} erfc l)` + upward
/// recursion, itself checked against mpmath's `quad` to 1e-25).
#[test]
fn reference_quadrature_is_anchored() {
    let cases: [(f64, f64, [f64; 3]); 3] = [
        (
            1.0,
            1000.0,
            [1.83847705403e-4, 1.83664315988e-4, 1.8348129171e-4],
        ),
        (
            10.0,
            700.0,
            [3.28258722102e-8, 3.27785420062e-8, 3.27313478925e-8],
        ),
        (0.3, 5.0, [0.0609621901149, 0.0531234978804, 0.046956956848]),
    ];
    for (t, u, want) in cases {
        for (m, &w) in want.iter().enumerate() {
            let got = gm_reference(m, t, u);
            let rel = (got - w).abs() / w;
            assert!(rel < 1e-10, "G_{m}({t}, {u}): {got:e} vs mpmath {w:e}");
        }
    }
    // T = 1024, U = 1 (mpmath closed form 1.20737617704e-29): the quadrature
    // peak is narrow here, so only the magnitude is asserted.
    let g0 = gm_reference(0, 1024.0, 1.0);
    assert!(
        (g0 - 1.20737617704e-29).abs() < 1e-33,
        "G_0(1024, 1) = {g0:e}"
    );
}

// ------------------------------------------------------ direct evaluator

fn ulp_down(x: f64) -> f64 {
    f64::from_bits(x.to_bits() - 1)
}
fn ulp_up(x: f64) -> f64 {
    f64::from_bits(x.to_bits() + 1)
}

/// `(zeta, one_over_rho)` giving `U = 0.25 * zeta * zeta * one_over_rho` (libint's
/// operation order) equal to `u`, or at most a few ulp below it (never above:
/// a U above `Umax` is rejected by the 2.7.2 override). `u == 1000` uses the
/// exact pair (200, 0.1), for which the product is exactly 1000.0.
fn zeta_for(u: f64) -> (f64, f64) {
    if u == U_EDGE {
        return (200.0, 0.1);
    }
    let mut zeta = 2.0 * u.sqrt();
    while 0.25 * zeta * zeta * 1.0 > u {
        zeta = ulp_down(zeta);
    }
    (zeta, 1.0)
}

/// Tolerance of the table-interpolated slots: the table guarantees an absolute
/// 1.4e-14 for m > 0; the explicit slot 0 is far better, but at tiny T and large
/// U its `exp(x*x)` rounding amplifies (~x^2 eps), so allow 1e-9 relative there.
fn tol(ref_val: f64) -> f64 {
    2e-14 + 1e-9 * ref_val.abs()
}

const T_EDGE: f64 = 1024.0;
const U_EDGE: f64 = 1000.0;

fn edge_points() -> Vec<(f64, f64)> {
    let ts = [
        T_EDGE,
        ulp_down(T_EDGE),
        ulp_down(ulp_down(T_EDGE)),
        ulp_up(T_EDGE),
        1023.0,
        512.0,
        ulp_down(512.0),
        ulp_up(512.0),
        2.0,
        0.0,
        1e-3,
        1.0,
        30.0,
    ];
    let us = [
        1.0,
        0.25,
        500.0,
        900.0,
        999.0,
        ulp_down(U_EDGE),
        ulp_down(ulp_down(U_EDGE)),
        ulp_down(ulp_down(ulp_down(U_EDGE))),
        U_EDGE,
        1e-6,
        100.0,
    ];
    let mut v = Vec::new();
    for &t in &ts {
        for &u in &us {
            v.push((t, u));
        }
    }
    v
}

#[test]
fn evaluator_matches_the_defining_integral_at_the_table_edges() {
    if skip_unpatched_272() {
        return;
    }
    let mmax = 4;
    let mut checked = 0;
    let mut nan_on_other_build = 0;
    for &(t, u) in &edge_points() {
        // T == 0 with a tiny U needs the unstable upward recursion (stock
        // restriction, see Operator::yukawa); the grid keeps U >= 1e-6 there.
        let (zeta, oor) = zeta_for(u);
        let u_eff = 0.25 * zeta * zeta * oor;
        let z2r = 0.5 * zeta * oor;
        for kind in [TennoKind::Yukawa, TennoKind::Slater] {
            let got = tenno_gm(kind, oor, t, mmax, zeta)
                .unwrap_or_else(|e| panic!("{kind:?} T={t:e} U={u_eff:e}: {e}"));
            for (m, &g) in got.iter().enumerate() {
                if !g.is_finite() && other_build_overflows(t, u_eff) {
                    nan_on_other_build += 1;
                    continue;
                }
                assert!(g.is_finite(), "{kind:?} m={m} T={t:e} U={u_eff:e}: {g}");
                let want = reference(kind, m, t.max(1e-300), u_eff, z2r);
                assert!(
                    (g - want).abs()
                        <= tol(want) * if kind == TennoKind::Slater { 1e3 } else { 1.0 },
                    "{kind:?} m={m} T={t:e} U={u_eff:e}: {g:e} vs defining integral {want:e}"
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 200, "{checked}");
    if nan_on_other_build > 0 {
        eprintln!(
            "NOTE: {nan_on_other_build} slots are NaN on this libint2 (T > Tmax with U + 2 sqrt(U T) > 700: \
             the same exp * erfc overflow in its upward recursion; the ferric boys.h overrides fix it)"
        );
    }
}

/// Stock libint2 (2.13.1 and 2.7.2) has the `exp(x*x - T) * erfc(x)` overflow
/// in `eval_urr`, i.e. for `T > Tmax` once `U + 2 sqrt(U T)` exceeds ~709. Both
/// ferric overrides remove it (#436, #439); a build with no override is excused.
fn other_build_overflows(t: f64, u: f64) -> bool {
    BOYS_OVERRIDE.is_none() && t > T_EDGE && u + 2.0 * (u * t).sqrt() > 700.0
}

/// The input that hits the U edge must really be U == Umax under libint's own
/// arithmetic, and the one that hits the T edge must really be T == Tmax.
#[test]
fn the_edge_inputs_are_exactly_on_the_edge() {
    // zeta = 200, one_over_rho = 0.1: U = 0.25 * 200 * 200 * 0.1
    let u = 0.25 * 200.0_f64 * 200.0 * 0.1;
    assert_eq!(u, 1000.0, "U must be exactly Umax = 1000");
    // T is passed through verbatim.
    assert_eq!(T_EDGE, 1024.0);
}

// ---------------------------------------------------------------- the engine

fn s_shell(exp: f64) -> Shell {
    // one primitive, ferric's convention: unit-normalised primitive
    Shell {
        l: 0,
        pure: false,
        exponents: vec![exp],
        coefficients: vec![1.0],
    }
}

fn prep_at(pos: [f64; 3], shell: Shell) -> PreparedBasis {
    let mol = Molecule {
        atoms: vec![Atom {
            symbol: "I".into(),
            z: 53,
            x: pos[0],
            y: pos[1],
            zpos: pos[2],
            ghost: false,
            n_core_ecp: 0,
        }],
        charge: 0,
        multiplicity: 1,
    };
    let mut shells = HashMap::new();
    shells.insert(53, vec![shell]);
    let bs = BasisSet {
        name: "tenno-edge".into(),
        shells,
        ecps: HashMap::new(),
    };
    PreparedBasis::new(&mol, &bs).expect("prep")
}

/// `(P | AA)`: P an s primitive of exponent `p` at the origin, `AA` the
/// product of two s primitives of exponent `a` at `pos_a`, Yukawa `zeta`.
fn p_aa(p: f64, a: f64, pos_a: [f64; 3], op: Operator) -> Option<f64> {
    let obs = prep_at(pos_a, s_shell(a));
    let aux = prep_at([0.0; 3], s_shell(p));
    let mut eng = Engine::new_3center(op, &obs, &aux, f64::MIN_POSITIVE).expect("engine");
    eng.compute_eri3(&obs, &aux, 0, 0, 0).map(|b| b[0])
}

/// Interaction of two UNIT-normalised s Gaussian charges of exponents `p` and
/// `q` at distance `r` under `exp(-zeta r) / r`, by quadrature in Fourier
/// space: `(2 / (pi r)) int_0^inf k sin(k r) / (k^2 + zeta^2) exp(-k^2 / (4 rho)) dk`,
/// `rho = p q / (p + q)`. Independent of libint2's `Gm`.
fn yukawa_gaussians_fourier(p: f64, q: f64, r: f64, zeta: f64) -> f64 {
    let rho = p * q / (p + q);
    let gl = gauss_legendre(24);
    let kmax = 2.0 * (rho * 50.0).sqrt() + 5.0;
    let panels = 4000;
    let mut total = 0.0;
    for i in 0..panels {
        let (a, b) = (
            kmax * i as f64 / f64::from(panels),
            kmax * (i + 1) as f64 / f64::from(panels),
        );
        let (c, h) = (0.5 * (a + b), 0.5 * (b - a));
        total += gl
            .iter()
            .map(|&(x, w)| {
                let k = c + h * x;
                w * k * (k * r).sin() / (k * k + zeta * zeta) * (-k * k / (4.0 * rho)).exp()
            })
            .sum::<f64>()
            * h;
    }
    2.0 / (PI * r) * total
}

/// Closed form of the same interaction in terms of complementary error
/// functions (the form used where the Fourier integral oscillates too fast:
/// valid for any r > 0; the `e^{+zeta r} erfc` term underflows harmlessly).
fn yukawa_gaussians_erfc(p: f64, q: f64, r: f64, zeta: f64) -> f64 {
    let rho = p * q / (p + q);
    let a = zeta / (2.0 * rho.sqrt());
    let b = rho.sqrt() * r;
    let e = a * a;
    let first = (e - zeta * r).exp() * unsafe { erfc(a - b) };
    let second = (e + zeta * r).exp() * unsafe { erfc(a + b) };
    0.5 * (first - second) / r
}

/// `(P|AA)` closed form: P is a unit-normalised s primitive (weight
/// `N_p (pi/p)^(3/2) = (2 pi / p)^(3/4)`... we reuse the unit-charge result:
/// the engine integral is `[N_p (pi/p)^{3/2}] * [unit-charge interaction]`
/// because `AA` is a unit charge (`N_a^2 (pi/(2a))^{3/2} = 1`).
fn p_aa_closed_form(p: f64, a: f64, r: f64, zeta: f64, fourier: bool) -> f64 {
    let weight_p = (2.0 * p / PI).powf(0.75) * (PI / p).powf(1.5);
    let q = 2.0 * a;
    let i = if fourier {
        yukawa_gaussians_fourier(p, q, r, zeta)
    } else {
        yukawa_gaussians_erfc(p, q, r, zeta)
    };
    weight_p * i
}

#[test]
fn closed_forms_agree_at_a_generic_geometry() {
    // anchor: Fourier quadrature == erfc closed form == engine
    let (p, a, r, zeta) = (1.3, 0.7, 1.9, 0.8);
    let f = p_aa_closed_form(p, a, r, zeta, true);
    let c = p_aa_closed_form(p, a, r, zeta, false);
    assert!((f - c).abs() < 1e-12 * c.abs(), "{f:e} vs {c:e}");
    let got = p_aa(p, a, [r, 0.0, 0.0], Operator::yukawa(zeta)).expect("not screened");
    assert!((got - c).abs() < 1e-11 * c.abs(), "engine {got:e} vs {c:e}");
}

/// P exponent 20 at the origin; AA = two s of exponent 10 at (1, 0, 0), so
/// rho = 20 * 20 / 40 = 10 and |PQ|^2 = 1, T = 10; zeta = 200 gives
/// U = zeta^2 / (4 rho) = 40000 / 40 = 1000. In binary floating point
/// `0.25 * 200 * 200 * 0.1` is exactly 1000.0 (checked above); libint forms
/// `one_over_rho` as 1/rho with rho = 10 exactly, so U lands on Umax or at most
/// an ulp below it, both inside the stock defect window.
#[test]
fn engine_u_edge_matches_the_fourier_closed_form() {
    if skip_unpatched_272() {
        return;
    }
    let (p, a, zeta) = (20.0, 10.0, 200.0);
    let rho = p * (2.0 * a) / (p + 2.0 * a);
    assert_eq!(rho, 10.0);
    let u = 0.25 * zeta * zeta * (1.0 / rho);
    assert!(
        (999.9999999999998..=1000.0).contains(&u),
        "U = {u:.17} must be in the defect window [999.9999999999998, 1000]"
    );
    let r = 1.0;
    // (the erfc closed form overflows here: exp(zeta^2 / (4 rho)) = e^1000)
    let want = p_aa_closed_form(p, a, r, zeta, true);
    let got = p_aa(p, a, [r, 0.0, 0.0], Operator::yukawa(zeta)).expect("not screened");
    assert!(got.is_finite(), "{got}");
    assert!(
        (got - want).abs() < 1e-9 * want,
        "engine {got:e} vs Fourier closed form {want:e}"
    );
}

/// P exponent 2 at the origin; AA = two s of exponent 1 at (32, 0, 0):
/// rho = 1, |PQ|^2 = 1024, so T == 1024.0 exactly (every intermediate is a
/// small integer). zeta = 1.
#[test]
fn engine_t_edge_is_continuous_and_matches_the_erfc_closed_form() {
    if skip_unpatched_272() {
        return;
    }
    let (p, a, zeta) = (2.0, 1.0, 1.0);
    let rho = p * (2.0 * a) / (p + 2.0 * a);
    assert_eq!(rho, 1.0);
    let t = rho * (32.0_f64 * 32.0);
    assert_eq!(t, 1024.0, "T must be exactly Tmax");
    let op = || Operator::yukawa(zeta);
    let got = p_aa(p, a, [32.0, 0.0, 0.0], op()).expect("not screened");
    let want = p_aa_closed_form(p, a, 32.0, zeta, false);
    assert!(
        (got - want).abs() < 1e-9 * want.abs() + 1e-30,
        "engine {got:e} vs erfc closed form {want:e}"
    );
    // continuity across T = 1024 (distance scaled by 1 +- 1e-9)
    for s in [1.0 - 1e-9, 1.0 + 1e-9] {
        let near = p_aa(p, a, [32.0 * s, 0.0, 0.0], op()).expect("not screened");
        let want_near = p_aa_closed_form(p, a, 32.0 * s, zeta, false);
        assert!((near - want_near).abs() < 1e-9 * want_near.abs() + 1e-30);
        assert!((near - got).abs() < 1e-6 * got.abs(), "{near:e} vs {got:e}");
    }
    // Slater geminal (s s s: only Gm[0] and G_{-1}) at the same geometry
    let sg = p_aa(p, a, [32.0, 0.0, 0.0], Operator::slater_geminal(zeta)).expect("not screened");
    assert!(sg.is_finite());
}

/// `U > Umax`. On the 2.7.2 override: with `T <= Tmax` it is rejected (`Err`); with
/// `T > Tmax` (above both limits) the true value is exactly 0 and the upward
/// recursion returns it, finite (on 2.13.1 with the override too, issue #439;
/// stock 2.13.1 returns NaN there). For `T <= Tmax` 2.13.1 falls back to a
/// Gaussian fit, which is asserted finite only. Builds with no override are
/// only exercised.
#[test]
fn u_above_umax_is_rejected_only_below_tmax() {
    if skip_unpatched_272() {
        return;
    }
    // U = 0.25 * 400 * 400 * 0.1 = 4000 > Umax
    for kind in [TennoKind::Yukawa, TennoKind::Slater] {
        for &t in &[0.5, 30.0, 1024.0] {
            let r = tenno_gm(kind, 0.1, t, 3, 400.0);
            if BOYS_OVERRIDE == Some("2.7.2") {
                assert!(
                    r.is_err(),
                    "{kind:?} T={t}: U > Umax must be rejected, got {r:?}"
                );
            } else if BOYS_OVERRIDE == Some("2.13.1") {
                // 2.13.1 falls back to a Gaussian fit: finite, not rejected
                let v = r.unwrap_or_else(|e| panic!("{kind:?} T={t}: {e}"));
                assert!(v.iter().all(|x| x.is_finite()), "{kind:?} T={t}: {v:?}");
            }
        }
        for &t in &[ulp_up(T_EDGE), 3000.0] {
            let r = tenno_gm(kind, 0.1, t, 3, 400.0);
            if BOYS_OVERRIDE.is_some() {
                let v = r.unwrap_or_else(|e| panic!("{kind:?} T={t}: {e}"));
                assert!(v.iter().all(|x| *x == 0.0), "{kind:?} T={t}: {v:?}");
            }
        }
    }
    if BOYS_OVERRIDE == Some("2.7.2") {
        // Slater below Umin with T <= Tmax: formerly a negative table index
        assert!(tenno_gm(TennoKind::Slater, 1.0, 1.0, 3, 1e-4).is_err());
        // ... but T > Tmax keeps its working upward recursion
        let v = tenno_gm(TennoKind::Slater, 1.0, 2000.0, 3, 1e-4).expect("T > Tmax");
        assert!(v.iter().all(|x| x.is_finite()), "{v:?}");
    }
}

/// `T = 2000, U = 100`: `U + 2 sqrt(U T) = 994 > 709`, the `exp(x*x - T) * erfc(x)`
/// overflow of the upward recursion. The true values are < 1e-400, so 0. Fixed by
/// the 2.7.2 override (#436) and the 2.13.1 override (#439).
#[test]
fn large_u_plus_large_t_is_finite_on_the_override() {
    if BOYS_OVERRIDE.is_none() {
        eprintln!("SKIP: no ferric boys.h override in this build; stock libint2 returns NaN here");
        return;
    }
    for kind in [TennoKind::Yukawa, TennoKind::Slater] {
        let v = tenno_gm(kind, 1.0, 2000.0, 4, 20.0).expect("evaluates");
        assert!(v.iter().all(|x| *x == 0.0), "{kind:?}: {v:?}");
    }
}

/// `T > Tmax` and `U + 2 sqrt(U T) > 725`: the region where stock `eval_urr`'s
/// explicit `exp(x*x - T) * erfc(x)` overflows (NaN on every measured point
/// with `U + 2 sqrt(U T) > 725`). The true values are below 1e-230 (the largest
/// value the override returns anywhere on the swept grid is 3.7e-234, from
/// `exp(U - 2 sqrt(U T))`; the `e^-T` part underflows). Finite and < 1e-200 for
/// both kernels, every `m`, `T` up to 1e12 and `U` from 1 to 1e8 (`U > Umax` is
/// above both limits here, where the upward recursion is used).
#[test]
fn nan_region_grid_is_finite_and_vanishingly_small() {
    if BOYS_OVERRIDE.is_none() {
        eprintln!("SKIP: no ferric boys.h override in this build; stock libint2 returns NaN here");
        return;
    }
    let mut checked = 0;
    for &(t, u) in &nan_region_points() {
        let (zeta, oor) = (2.0 * u.sqrt(), 1.0);
        let u_eff = 0.25 * zeta * zeta * oor;
        for kind in [TennoKind::Yukawa, TennoKind::Slater] {
            let v = tenno_gm(kind, oor, t, 6, zeta)
                .unwrap_or_else(|e| panic!("{kind:?} T={t:e} U={u_eff:e}: {e}"));
            for (m, &g) in v.iter().enumerate() {
                assert!(
                    g.is_finite() && g.abs() < 1e-200,
                    "{kind:?} m={m} T={t:e} U={u_eff:e}: {g:e}"
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 100, "{checked}");
}

/// `(T, U)` in the NaN region of stock `eval_urr`: `T > Tmax = 1024` and
/// `U + 2 sqrt(U T) > 725`.
fn nan_region_points() -> Vec<(f64, f64)> {
    let ts = [
        ulp_up(T_EDGE),
        1025.0,
        1100.0,
        1500.0,
        2000.0,
        5000.0,
        1e4,
        1e5,
        1e8,
        1e12,
    ];
    let us = [
        1.0, 25.0, 99.0, 100.0, 150.0, 500.0, 1000.0, 4000.0, 1e5, 1e8,
    ];
    let mut v = Vec::new();
    for &t in &ts {
        for &u in &us {
            if u + 2.0 * (u * t).sqrt() > 725.0 {
                v.push((t, u));
            }
        }
    }
    v
}

/// Exact-integer far pairs through the real engine: P exponent 2 at the
/// origin and AA two s of exponent 1 at `(R, 0, 0)` give `rho = 1`,
/// `T = R^2`, `U = zeta^2 / 4`, all exact in binary floating point.
/// * zeta = 10, R = 70: T = 4900, U = 25, `U + zeta R` = 725 (U < 100).
/// * zeta = 20, R = 40: T = 1600, U = 100, `U + zeta R` = 900 (U >= 100).
///
/// Stock libint 2.13.1 and 2.7.2 return NaN on both; the true value is below
/// 1e-200 (two Gaussian charges 40+ Bohr apart under exp(-zeta r)/r with
/// zeta >= 10 is e^-400 or smaller).
#[test]
fn engine_far_pair_is_finite_on_the_override() {
    if BOYS_OVERRIDE.is_none() {
        eprintln!("SKIP: no ferric boys.h override in this build; stock libint2 returns NaN here");
        return;
    }
    for (zeta, r) in [(10.0_f64, 70.0_f64), (20.0, 40.0)] {
        let u = 0.25 * zeta * zeta;
        assert!(u + zeta * r > 709.0, "geometry must be in the NaN region");
        assert!(r * r > T_EDGE, "T must exceed Tmax");
        for op in [Operator::yukawa(zeta), Operator::slater_geminal(zeta)] {
            let got = p_aa(2.0, 1.0, [r, 0.0, 0.0], op).expect("not screened");
            assert!(
                got.is_finite() && got.abs() < 1e-200,
                "zeta={zeta} R={r}: {got:e}"
            );
        }
    }
}

/// Through the engine, the 2.7.2 override's `U > Umax` rejection surfaces as the
/// shim's `SCF_EINTERNAL` status, which the compute wrapper turns into a panic
/// (never a silent NaN integral). rho = 10, zeta = 300: U = 2250.
#[test]
fn engine_u_above_umax_panics_instead_of_returning_nan() {
    if BOYS_OVERRIDE != Some("2.7.2") {
        eprintln!("SKIP: only the 2.7.2 override rejects U > Umax");
        return;
    }
    let outcome =
        std::panic::catch_unwind(|| p_aa(20.0, 10.0, [1.0, 0.0, 0.0], Operator::yukawa(300.0)));
    assert!(
        outcome.is_err(),
        "U = 2250 must be rejected, got {outcome:?}"
    );
}

// ----------------------------------------------------------------- sweep

fn fnv(h: &mut u64, x: f64) {
    for b in x.to_bits().to_le_bytes() {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn sweep() -> (u64, u64, usize) {
    let (mut h_edge, mut h_engine, mut n) =
        (0xcbf2_9ce4_8422_2325_u64, 0xcbf2_9ce4_8422_2325_u64, 0);
    for &(t, u) in &edge_points() {
        let (zeta, oor) = zeta_for(u);
        for kind in [TennoKind::Yukawa, TennoKind::Slater] {
            match tenno_gm(kind, oor, t, 6, zeta) {
                Ok(v) => v.iter().for_each(|&x| fnv(&mut h_edge, x)),
                Err(_) => fnv(&mut h_edge, f64::NAN),
            }
            n += 1;
        }
    }
    // engine blocks on the exact-edge integer geometries
    for (p, a, pos, zeta) in [
        (20.0, 10.0, [1.0, 0.0, 0.0], 200.0),
        (2.0, 1.0, [32.0, 0.0, 0.0], 1.0),
        (2.0, 1.0, [32.0, 0.0, 0.0], 3.0),
        (20.0, 10.0, [0.0, 0.0, 0.0], 200.0),
        (20.0, 10.0, [0.0, 1.0, 0.0], 199.0),
    ] {
        for op in [Operator::yukawa(zeta), Operator::slater_geminal(zeta)] {
            match p_aa(p, a, pos, op) {
                Some(x) => fnv(&mut h_engine, x),
                None => fnv(&mut h_engine, f64::NAN),
            }
            n += 1;
        }
    }
    (h_edge, h_engine, n)
}

/// Child entry point: prints the hashes. Run by
/// `sweep_is_independent_of_allocator_fill` under several allocator fills.
#[test]
fn sweep_child_prints_hashes() {
    if skip_unpatched_272() {
        println!("SWEEP_HASH skipped");
        return;
    }
    let (h_edge, h_engine, n) = sweep();
    let h_nan = nan_region_hash();
    println!("SWEEP_HASH edge={h_edge:016x} engine={h_engine:016x} n={n} nanreg={h_nan:016x}");
}

/// Hash of the `T > Tmax` NaN-region evaluator grid plus the exact-integer far
/// pairs (kept out of `sweep()` so its `edge`/`engine`/`n` stay comparable with
/// the values recorded before #439).
fn nan_region_hash() -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for &(t, u) in &nan_region_points() {
        for kind in [TennoKind::Yukawa, TennoKind::Slater] {
            match tenno_gm(kind, 1.0, t, 6, 2.0 * u.sqrt()) {
                Ok(v) => v.iter().for_each(|&x| fnv(&mut h, x)),
                Err(_) => fnv(&mut h, f64::NAN),
            }
        }
    }
    for (zeta, r) in [(10.0_f64, 70.0_f64), (20.0, 40.0)] {
        for op in [Operator::yukawa(zeta), Operator::slater_geminal(zeta)] {
            fnv(
                &mut h,
                p_aa(2.0, 1.0, [r, 0.0, 0.0], op).unwrap_or(f64::NAN),
            );
        }
    }
    h
}

#[test]
fn sweep_is_independent_of_allocator_fill() {
    let exe = std::env::current_exe().expect("current exe");
    let run = |perturb: Option<&str>| -> String {
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(["sweep_child_prints_hashes", "--exact", "--nocapture"]);
        cmd.env_remove("MALLOC_PERTURB_");
        if let Some(p) = perturb {
            cmd.env("MALLOC_PERTURB_", p);
        }
        let out = cmd.output().expect("spawn");
        assert!(out.status.success(), "child failed: {:?}", out.status);
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .find(|l| l.starts_with("SWEEP_HASH"))
            .expect("hash line")
            .to_string()
    };
    let base = run(None);
    for p in ["1", "170", "2"] {
        let got = run(Some(p));
        assert_eq!(
            got, base,
            "MALLOC_PERTURB_={p} changed integral bits (uninitialised read)"
        );
    }
}
