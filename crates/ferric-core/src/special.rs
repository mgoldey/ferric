//! Special functions: the error function `erf` and complementary error
//! function `erfc`.
//!
//! Both follow W. J. Cody, "Rational Chebyshev approximations for the error
//! function", Math. Comp. 23, 631 (1969) (the scheme behind most libm
//! implementations), in three regions:
//!
//! - `|x| <= 0.46875`: `erf(x) = x·R(x²)` (direct, so `erf` keeps full
//!   *relative* accuracy near zero);
//! - `0.46875 < |x| <= 4`: `erfc(|x|) = exp(-x²)·R(|x|)`;
//! - `|x| > 4`: asymptotic `erfc(|x|) = exp(-x²)/|x|·(1/√π − R(1/x²)/x²)`.
//!
//! `exp(-x²)` is evaluated with Cody's split `x² = ys² + (x−ys)(x+ys)`,
//! `ys = trunc(16x)/16`, which avoids the cancellation of a single rounding of
//! `x²`. `erfc` keeps full relative accuracy through the tail (it underflows to
//! 0 near `x ≈ 26.5`). Maximum relative error is a few ulp across the real line
//! (see the tests, which compare to 40-digit mpmath values).
//!
//! Hand-rolled because `f64::erf` is not in stable Rust and `ferric-core`
//! carries no special-function dependency (and no `unsafe`
//! libm binding).

/// Cody region 1 numerator (`|x| <= 0.46875`), in powers of `x²`.
const A: [f64; 5] = [
    3.161_123_743_870_565_6e0,
    1.138_641_541_510_501_56e2,
    3.774_852_376_853_020_21e2,
    3.209_377_589_138_469_47e3,
    1.857_777_061_846_031_53e-1,
];
/// Cody region 1 denominator.
const B: [f64; 4] = [
    2.360_129_095_234_412_09e1,
    2.440_246_379_344_441_73e2,
    1.282_616_526_077_372_28e3,
    2.844_236_833_439_170_62e3,
];
/// Cody region 2 numerator (`0.46875 < x <= 4`).
const C: [f64; 9] = [
    5.641_884_969_886_700_89e-1,
    8.883_149_794_388_375_94e0,
    6.611_919_063_714_162_95e1,
    2.986_351_381_974_001_31e2,
    8.819_522_212_417_690_90e2,
    1.712_047_612_634_070_58e3,
    2.051_078_377_826_071_47e3,
    1.230_339_354_797_997_25e3,
    2.153_115_354_744_038_46e-8,
];
/// Cody region 2 denominator.
const D: [f64; 8] = [
    1.574_492_611_070_983_47e1,
    1.176_939_508_913_124_99e2,
    5.371_811_018_620_098_58e2,
    1.621_389_574_566_690_19e3,
    3.290_799_235_733_459_63e3,
    4.362_619_090_143_247_16e3,
    3.439_367_674_143_721_64e3,
    1.230_339_354_803_749_42e3,
];
/// Cody region 3 numerator (`x > 4`), in powers of `1/x²`.
const P: [f64; 6] = [
    3.053_266_349_612_323_44e-1,
    3.603_448_999_498_044_39e-1,
    1.257_817_261_112_292_46e-1,
    1.608_378_514_874_227_66e-2,
    6.587_491_615_298_378_03e-4,
    1.631_538_713_730_209_78e-2,
];
/// Cody region 3 denominator.
const Q: [f64; 5] = [
    2.568_520_192_289_822_42e0,
    1.872_952_849_923_460_47e0,
    5.279_051_029_514_284_12e-1,
    6.051_834_131_244_131_91e-2,
    2.335_204_976_268_691_85e-3,
];

/// `1/√π`.
const FRAC_1_SQRT_PI: f64 = 0.564_189_583_547_756_3;
/// Region 1 / region 2 boundary.
const X_SMALL: f64 = 0.468_75;
/// Region 2 / region 3 boundary.
const X_LARGE: f64 = 4.0;
/// `erfc(x)` underflows to 0 beyond this (`x² > ln(f64::MIN_POSITIVE)` margin).
const X_ERFC_ZERO: f64 = 26.543;

/// `erf(x)·` for `|x| <= X_SMALL`: `x·R(x²)`.
#[inline]
fn erf_small(x: f64) -> f64 {
    let z = x * x;
    let num = ((((A[4] * z + A[0]) * z + A[1]) * z + A[2]) * z + A[3]) * x;
    let den = (((z + B[0]) * z + B[1]) * z + B[2]) * z + B[3];
    num / den
}

/// `exp(-x²)` with Cody's split of `x²` (see module docs).
#[inline]
fn exp_neg_sq(x: f64) -> f64 {
    let ys = (x * 16.0).trunc() / 16.0;
    let del = (x - ys) * (x + ys);
    (-ys * ys).exp() * (-del).exp()
}

/// `erfc(x)` for `x > X_SMALL` (regions 2 and 3, `x` finite positive).
#[inline]
fn erfc_pos(x: f64) -> f64 {
    if x > X_ERFC_ZERO {
        return 0.0;
    }
    if x <= X_LARGE {
        let mut num = C[8] * x;
        let mut den = x;
        for k in 0..7 {
            num = (num + C[k]) * x;
            den = (den + D[k]) * x;
        }
        (num + C[7]) / (den + D[7]) * exp_neg_sq(x)
    } else {
        let zinv = 1.0 / (x * x);
        let mut num = P[5] * zinv;
        let mut den = zinv;
        for k in 0..4 {
            num = (num + P[k]) * zinv;
            den = (den + Q[k]) * zinv;
        }
        let r = zinv * (num + P[4]) / (den + Q[4]);
        (FRAC_1_SQRT_PI - r) / x * exp_neg_sq(x)
    }
}

/// Error function `erf(x) = (2/√π)∫₀ˣ exp(−t²) dt`.
///
/// Odd (`erf(−x) = −erf(x)` exactly), `erf(±∞) = ±1`, NaN propagates.
/// Near zero it is evaluated directly (not as `1 − erfc`), so it keeps full
/// relative accuracy for small `|x|`.
#[inline]
pub fn erf(x: f64) -> f64 {
    let a = x.abs();
    if a.is_nan() {
        return x;
    }
    if a <= X_SMALL {
        return erf_small(x);
    }
    // erf(a) = 1 - erfc(a); erfc(a) < 1.2e-16 beyond a ≈ 5.9, so this is 1.
    let v = 1.0 - erfc_pos(a);
    if x < 0.0 {
        -v
    } else {
        v
    }
}

/// Complementary error function `erfc(x) = 1 − erf(x)`.
///
/// Keeps full *relative* accuracy for large positive `x` (it is computed from
/// the Gaussian-weighted rational forms, never as `1 − erf`); underflows to 0
/// for `x ≳ 26.5`. Negative arguments use `erfc(−x) = 2 − erfc(x)`. `erfc(+∞)
/// = 0`, `erfc(−∞) = 2`, NaN propagates.
#[inline]
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x.abs() <= X_SMALL {
        return 1.0 - erf_small(x);
    }
    if x > 0.0 {
        erfc_pos(x)
    } else {
        2.0 - erfc_pos(-x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (x, erf(x), erfc(x)) from 40-digit mpmath.
    const REF: [(f64, f64, f64); 20] = [
        (0.0, 0.0, 1.0),
        (1e-8, 1.1283791670955125599e-8, 0.99999998871620832904),
        (1e-3, 0.0011283787909692364034, 0.9988716212090307636),
        (0.1, 0.1124629160182848984, 0.8875370839817151016),
        (0.3, 0.32862675945912741619, 0.67137324054087258381),
        (0.46875, 0.49261347321793799159, 0.50738652678206200841),
        (0.5, 0.52049987781304653768, 0.47950012218695346232),
        (1.0, 0.84270079294971486934, 0.15729920705028513066),
        (1.5, 0.96610514647531072707, 0.033894853524689272933),
        (2.0, 0.99532226501895273416, 0.0046777349810472658379),
        (3.0, 0.99997790950300141456, 0.000022090496998585441373),
        (4.0, 0.99999998458274209972, 1.5417257900280018852e-8),
        (5.0, 0.99999999999846254021, 1.5374597944280348502e-12),
        (6.0, 0.99999999999999997848, 2.1519736712498913117e-17),
        (10.0, 1.0, 2.088487583762544757e-45),
        (15.0, 1.0, 7.2129941724512066666e-100),
        (20.0, 1.0, 5.3958656116079009289e-176),
        (26.0, 1.0, 5.6631924088561428465e-296),
        (4.5, 0.99999999980338395585, 1.9661604415428874763e-10),
        (0.46876, 0.49262253110684652927, 0.50737746889315347073),
    ];

    fn rel(got: f64, want: f64) -> f64 {
        if want == 0.0 {
            got.abs()
        } else {
            ((got - want) / want).abs()
        }
    }

    #[test]
    fn erf_matches_high_precision_values() {
        for (x, e, _) in REF {
            let r = rel(erf(x), e);
            assert!(r < 5e-16, "erf({x}) = {:e}, want {e:e}, rel {r:e}", erf(x));
        }
    }

    /// The tail test: erfc must be relatively accurate where erfc << 1
    /// (a `1 - erf` implementation would return 0 or noise for x >~ 6).
    #[test]
    fn erfc_matches_high_precision_values_including_tail() {
        for (x, _, c) in REF {
            let r = rel(erfc(x), c);
            assert!(
                r < 1e-15,
                "erfc({x}) = {:e}, want {c:e}, rel {r:e}",
                erfc(x)
            );
        }
    }

    #[test]
    fn erf_is_odd_and_erfc_reflects() {
        for (x, e, c) in REF {
            assert_eq!(erf(-x), -erf(x), "erf odd at {x}");
            let r = rel(erfc(-x), 2.0 - c);
            assert!(r < 2e-16, "erfc(-{x}) rel {r:e}");
            let _ = e;
        }
    }

    #[test]
    fn erf_plus_erfc_is_one() {
        let mut x = -7.0;
        while x < 7.0 {
            assert!((erf(x) + erfc(x) - 1.0).abs() < 4e-16, "x={x}");
            x += 0.0137;
        }
    }

    /// Region joins at 0.46875 and 4 are continuous to rounding.
    #[test]
    fn region_boundaries_are_continuous() {
        for b in [X_SMALL, X_LARGE] {
            let (lo, hi) = (b - 1e-12, b + 1e-12);
            assert!((erf(lo) - erf(hi)).abs() < 1e-11);
            assert!(rel(erfc(lo), erfc(hi)) < 1e-10);
        }
    }

    #[test]
    fn special_values() {
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert_eq!(erf(f64::NEG_INFINITY), -1.0);
        assert_eq!(erfc(f64::INFINITY), 0.0);
        assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
        assert!(erf(f64::NAN).is_nan());
        assert!(erfc(f64::NAN).is_nan());
        assert_eq!(erfc(30.0), 0.0);
    }
}
