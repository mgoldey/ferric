//! libint2's `TennoGmEval::eval_yukawa` for `U < Umin` (issue #443).
//!
//! DEFECT. For `U = 0.25 zeta^2 / rho < Umin = 1e-7` stock libint2 (2.7.2 and
//! 2.13.1) always evaluates the core integral
//! `G_m = int_0^1 t^{2m} exp(U(1 - t^-2) - T t^2) dt` with `eval_urr`, the
//! upward recursion `G_{m+1} = [(2m+1) G_m + 2U G_{m-1} - e^-T] / (2T)`. It
//! multiplies the rounding error of `G_0` by `prod (2m+1)/(2T)` per step, so
//! it is garbage for `T` below roughly `0.35 mmax`: O(1) at `m <= 20`, `T = 1`;
//! inf/NaN at `m = 36`, `T = 1e-8`. The error does not depend on `U`.
//!
//! The override sends `U < Umin` and `T < 0.5 mmax + 6` (`T < 1e-2` at
//! `mmax == 0`, where only the closed form of `G_0` cancels) to a graded
//! Gauss-Legendre quadrature of the defining integral; at or above that `T`
//! the recursion, whose result is already accurate there, is untouched.
//!
//! ARTIFACT vs PHYSICS hypothesis (written before measuring). Physics: for
//! `U -> 0` the kernel is exp(-zeta r)/r -> 1/r, so `G_m(T, U) -> F_m(T)` (the
//! Boys function) with `0 <= G_m <= 1/(2m+1)` and monotone decay in m, smooth
//! in T. Artifact (the recursion): errors that grow geometrically in m and
//! blow up as T -> 0 independent of U, violating `G_m <= 1/(2m+1)`. They
//! cannot be confused: the first is bounded and U-independent in the m
//! direction only through F_m, the second is unbounded.
//!
//! References: `data/yukawa_urr_ref.rs`, mpmath (50 digits) quadrature of the
//! defining integral. The exactness anchor is `U -> 0 => F_m(T)`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::{boys_fm_chebyshev7, tenno_gm, Engine, TennoKind};
use ferric_integrals::operator::Operator;

#[path = "data/yukawa_urr_ref.rs"]
mod reference;

/// Libint's stock evaluation stays unchanged from this `T` (see the override).
fn t_safe(mmax: usize) -> f64 {
    if mmax == 0 {
        1e-2
    } else {
        0.5 * mmax as f64 + 6.0
    }
}

/// Largest `mmax` libint's Tenno table supports: 36 on 2.13.1, 20 on 2.7.2
/// (a larger request throws, which the shim reports as an error).
fn max_m() -> usize {
    if ferric_integrals::libint_version() == "2.7.2" {
        20
    } else {
        36
    }
}

extern "C" {
    fn erfc(x: f64) -> f64;
}

/// Port of stock `eval_urr<false>` for the bit-identity side of the dispatch.
fn stock_urr(t: f64, u: f64, mmax: usize) -> Vec<f64> {
    stock_urr_perturbed(t, u, mmax, 0.0)
}

/// `stock_urr` with `G_0` scaled by `1 + eps`: the response to
/// one rounding error, which the recursion amplifies.
fn stock_urr_perturbed(t: f64, u: f64, mmax: usize, eps: f64) -> Vec<f64> {
    let (su, st) = (u.sqrt(), t.sqrt());
    let (kappa, lambda) = (su - st, su + st);
    // SAFETY: pure libm function.
    let ek = (kappa * kappa - t).exp() * unsafe { erfc(kappa) };
    let el = (lambda * lambda - t).exp() * unsafe { erfc(lambda) };
    let pfac = 0.443_113_462_726_379_f64;
    let mut gmm1 = pfac * (ek + el) / su;
    let mut gm = pfac * (ek - el) / st * (1.0 + eps);
    let mut out = vec![gm];
    let (oo2t, two_u, emt) = (0.5 / t, 2.0 * u, (-t).exp());
    for m in 0..mmax {
        let gp = oo2t * ((2 * m + 1) as f64 * gm + two_u * gmm1 - emt);
        out.push(gp);
        gmm1 = gm;
        gm = gp;
    }
    out
}

#[test]
fn matches_mpmath_and_respects_the_analytic_bounds() {
    let mut worst = 0.0f64;
    for &(zeta, t, u, ref want) in reference::REF {
        for mmax in [0usize, 1, 6, 12, 20, 36]
            .into_iter()
            .filter(|&m| m <= max_m())
        {
            let got = tenno_gm(TennoKind::Yukawa, 1.0, t, mmax, zeta).expect("evaluates");
            for m in 0..=mmax {
                let g = got[m];
                assert!(g.is_finite(), "T={t:e} U={u:e} mmax={mmax} m={m}: {g}");
                let err = (g - want[m]).abs();
                worst = worst.max(err);
                assert!(
                    err <= 1e-13,
                    "T={t:e} U={u:e} mmax={mmax} m={m}: got {g:e} want {:e} err {err:e}",
                    want[m]
                );
                let hi = 1.0 / (2 * m + 1) as f64;
                assert!(
                    g >= -1e-15 && g <= hi + 1e-15,
                    "T={t:e} U={u:e} mmax={mmax} m={m}: {g:e} outside [0, {hi:e}]"
                );
            }
        }
    }
    eprintln!("worst abs err vs mpmath: {worst:e}");
}

/// Exactness anchor: as U -> 0, G_m -> F_m(T). For m >= 1 the U-dependence is
/// O(U^{m+1/2}) (below 1e-20), for m = 0 it is O(sqrt U) = 1e-7.
#[test]
fn u_to_zero_gives_the_boys_function() {
    for &t in &[1e-8, 1e-6, 1e-3, 0.1, 1.0, 3.0, 5.0, 10.0, 20.0] {
        let mmax = max_m();
        let zeta = 2e-10; // U = 1e-20 (one_over_rho = 1)
        let g = tenno_gm(TennoKind::Yukawa, 1.0, t, mmax, zeta).expect("evaluates");
        let f = boys_fm_chebyshev7(t, mmax).expect("boys");
        for m in 0..=mmax {
            let tol = if m == 0 { 1e-9 } else { 1e-13 };
            assert!(
                (g[m] - f[m]).abs() <= tol,
                "T={t:e} m={m}: G={:e} F={:e}",
                g[m],
                f[m]
            );
        }
    }
}

/// The grid of Yukawa arguments the override must leave on the stock path:
/// `U < Umin` with `T >= t_safe(mmax)`. Yields `(zeta, mmax, T)` for
/// `one_over_rho = 1`. Shared by the gating comparison and the opt-in golden.
fn untouched_grid() -> Vec<(f64, usize, f64)> {
    let mut out = Vec::new();
    for &zeta in &[2e-8, 2e-7, 2e-5, 6e-4] {
        for mmax in 0..=max_m() {
            for k in 0..60 {
                let t = if mmax == 0 {
                    1e-2 * 1.3f64.powi(k)
                } else {
                    t_safe(mmax) * (1.0 + 0.05 * f64::from(k))
                };
                if t <= 1000.0 {
                    out.push((zeta, mmax, t));
                }
            }
        }
    }
    out
}

/// GATING. At and above the dispatch threshold the stock recursion is what
/// runs (its result is already accurate there), so it must agree with an
/// in-binary port of it to a few ulp (libm- and FMA-independent). Raising the
/// threshold or sending everything to the fallback moves values by the
/// recursion's own error, which this bar (2e-15 relative) resolves.
#[test]
fn above_the_threshold_the_stock_recursion_is_unchanged() {
    for (zeta, mmax, t) in untouched_grid() {
        let u = 0.25 * zeta * zeta;
        assert!(u < 1e-7, "U={u:e} must be below Umin");
        let got = tenno_gm(TennoKind::Yukawa, 1.0, t, mmax, zeta).expect("evaluates");
        let want = stock_urr(t, u, mmax);
        // libm/FMA differences enter as rounding of G_0, which the recursion
        // amplifies; scale the allowance by the measured response to one ulp.
        let resp = stock_urr_perturbed(t, u, mmax, f64::EPSILON);
        for m in 0..=mmax {
            let tol = 16.0 * (resp[m] - want[m]).abs() + 5e-14 * want[m].abs() + 1e-300;
            assert!(
                (got[m] - want[m]).abs() <= tol,
                "zeta={zeta:e} T={t:e} mmax={mmax} m={m}: {:e} vs {:e}",
                got[m],
                want[m]
            );
        }
    }
}

/// GATING. The `U >= Umin` table region is not touched by the override: it
/// matches 50-digit mpmath values and the physical bounds `0 <= G_m <=
/// 1/(2m+1)` with monotone decay in m.
#[test]
fn table_region_matches_mpmath_and_is_physical() {
    let mmax = 20.min(max_m());
    for &(zeta, t, u, ref want) in reference::REF_TABLE {
        let got = tenno_gm(TennoKind::Yukawa, 1.0, t, mmax, zeta).expect("evaluates");
        for m in 0..=mmax {
            let g = got[m];
            assert!(
                (g - want[m]).abs() <= 1e-12,
                "U={u:e} T={t:e} m={m}: {g:e} vs {:e}",
                want[m]
            );
            assert!(
                g >= -1e-15 && g <= 1.0 / (2 * m + 1) as f64 + 1e-15,
                "U={u:e} T={t:e} m={m}: {g:e} out of bounds"
            );
            if m > 0 {
                assert!(
                    g <= got[m - 1] + 1e-13,
                    "U={u:e} T={t:e} m={m}: not decaying"
                );
            }
        }
    }
}

/// OPT-IN GOLDEN, never gating. FNV-1a over the bit patterns of every Yukawa
/// value on `untouched_grid()` plus `U >= Umin` points at small T. Bit patterns
/// depend on libm and compiler contraction, so another glibc/CPU can differ
/// with no regression; the gating checks are the two tests above. Run with
/// `--ignored --nocapture` on the dev box to refresh or compare (the recorded
/// values are stock-equal: identical with the fallback disabled).
#[test]
#[ignore = "bit-identity golden: recorded on dev box libint 2.13.1/2.7.2; run to refresh"]
fn untouched_region_hash() {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut n = 0usize;
    let mut feed = |x: f64| {
        for b in x.to_bits().to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        n += 1;
    };
    for (zeta, mmax, t) in untouched_grid() {
        for v in tenno_gm(TennoKind::Yukawa, 1.0, t, mmax, zeta).expect("evaluates") {
            feed(v);
        }
    }
    for &zeta in &[1.4, 3.0, 20.0] {
        for &t in &[0.0, 1e-8, 1e-3, 0.5, 2.0] {
            for v in tenno_gm(TennoKind::Yukawa, 1.0, t, 12, zeta).expect("evaluates") {
                feed(v);
            }
        }
    }
    eprintln!("UNTOUCHED_REGION_HASH n={n} h={h:016x}");
    let (want_n, want_h) = match ferric_integrals::libint_version() {
        "2.7.2" => (55579, 0xda40_0bb5_bddb_82a4_u64),
        "2.13.1" => (168_859, 0x84f8_33f1_8f8c_180e_u64),
        v => {
            eprintln!("SKIP hash assertion: no recorded hash for libint {v}");
            return;
        }
    };
    assert_eq!(
        (n, h),
        (want_n, want_h),
        "untouched-region values changed bits"
    );
}

/// The reported blow-up: inf/NaN at m = 36, T ~ 1e-8, and T = 0 exactly.
#[test]
fn no_inf_or_nan_at_tiny_t_and_high_m() {
    for &t in &[0.0, 1e-300, 1e-12, 1e-8, 1e-4] {
        for &zeta in &[2e-7, 2e-5, 2e-10] {
            let g = tenno_gm(TennoKind::Yukawa, 1.0, t, max_m(), zeta).expect("evaluates");
            for (m, &v) in g.iter().enumerate() {
                assert!(
                    v.is_finite() && v >= 0.0 && v <= 1.0 / (2 * m + 1) as f64 + 1e-15,
                    "T={t:e} zeta={zeta:e} m={m}: {v:e}"
                );
            }
        }
    }
}

/// Through the real 4-centre engine: with `zeta = 1e-8`, `U` is far below
/// `Umin` for every primitive pair, so every quartet takes the patched path,
/// including `T == 0` and `T` at rounding level (coincident bra/ket centres;
/// `mmax == 0` there, the closed-form `G_0` cancels to O(1) noise) and d shells
/// (`m` up to 8). The Yukawa ERI is then the Coulomb ERI plus
/// `-zeta S_ab S_cd + O(zeta^2)` with normalised-function overlaps
/// `|S| <= 1`, so every quartet is finite and within `1.5 zeta` of Coulomb. The
/// recursion this replaces is off by 4e-3 on H2/STO-3G (10|10) here.
#[test]
fn tiny_zeta_yukawa_eris_track_coulomb_through_the_engine() {
    let zeta = 1e-8;
    let h2 = "2\nh2\nH 0 0 0\nH 0 0 0.74\n";
    let water = "3\nw\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n";
    for (xyz, bname) in [
        (h2, "sto-3g"),
        (h2, "cc-pvdz"),
        (water, "sto-3g"),
        (water, "cc-pvdz"),
    ] {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled(bname).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let mut coul = Engine::new_2e(Operator::coulomb(), &prep, 1e-14).unwrap();
        let mut yuk = Engine::new_2e(Operator::yukawa(zeta), &prep, 1e-14).unwrap();
        let n = prep.nshells();
        let (mut worst, mut count) = (0.0f64, 0usize);
        for a in 0..n {
            for b in 0..=a {
                for c in 0..=a {
                    for d in 0..=c {
                        let vc = coul.compute_quartet(&prep, a, b, c, d).map(<[f64]>::to_vec);
                        let vy = yuk.compute_quartet(&prep, a, b, c, d).map(<[f64]>::to_vec);
                        let (Some(vc), Some(vy)) = (vc, vy) else {
                            continue;
                        };
                        for (x, y) in vc.iter().zip(&vy) {
                            assert!(y.is_finite(), "{bname} ({a}{b}|{c}{d}): {y}");
                            worst = worst.max((x - y).abs());
                            count += 1;
                        }
                    }
                }
            }
        }
        eprintln!(
            "{bname} ({} atoms): {count} integrals, worst |Yukawa - Coulomb| = {worst:e}",
            mol.atoms.len()
        );
        assert!(
            worst < 1.5 * zeta,
            "{bname}: worst |Yukawa - Coulomb| = {worst:e}"
        );
    }
}
