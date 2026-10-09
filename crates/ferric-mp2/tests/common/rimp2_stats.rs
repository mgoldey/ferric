//! Measurement helpers shared by the mixed RI-MP2 tests: per-element error
//! statistics of the G_i blocks, the log-log fit, and the two rounding modes.
use ndarray::{s, Array2, ArrayView2};

/// Per-element statistics of err/S over elements with S > 0, where
/// err = |approx − exact| and S_ab = Σ_P |B_P,ia||B_P,jb|. Thousands of
/// elements per block; the bound says err ≤ eps·S + eta.
pub struct Elem {
    eps: f64,
    eta: f64,
    pub n: usize,
    sum_sq: f64,
    /// max err / (eps·S + eta): ≤ 1 is the per-element bound.
    pub worst: f64,
}

impl Elem {
    pub fn new(eps: f64, eta: f64) -> Self {
        Self {
            eps,
            eta,
            n: 0,
            sum_sq: 0.0,
            worst: 0.0,
        }
    }

    pub fn add(
        &mut self,
        approx: &ArrayView2<f64>,
        exact: &ArrayView2<f64>,
        sab: &ArrayView2<f64>,
    ) {
        for ((a, e), sv) in approx.iter().zip(exact.iter()).zip(sab.iter()) {
            if *sv > 0.0 {
                let err = (a - e).abs();
                self.sum_sq += (err / sv) * (err / sv);
                self.n += 1;
                self.worst = self.worst.max(err / (self.eps * sv + self.eta));
            }
        }
    }

    /// Root mean square of err/S.
    pub fn rms(&self) -> f64 {
        (self.sum_sq / self.n.max(1) as f64).sqrt()
    }
}

/// (g, S) of block `i`: g = B_iᵀ·B_tail exact in f64, S the absolute-value
/// product (`babs` = |B_ov|).
pub fn block_g_s(
    b_ov: &Array2<f64>,
    babs: &Array2<f64>,
    i: usize,
    nvir: usize,
) -> (Array2<f64>, Array2<f64>) {
    let lo = i * nvir;
    let g = b_ov
        .slice(s![.., lo..lo + nvir])
        .t()
        .dot(&b_ov.slice(s![.., lo..]));
    let sab = babs
        .slice(s![.., lo..lo + nvir])
        .t()
        .dot(&babs.slice(s![.., lo..]));
    (g, sab)
}

pub fn max_abs(b: &Array2<f64>) -> f64 {
    b.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// `x` rounded to f32 by TRUNCATION (toward zero), back to f64: the mutant of
/// round-to-nearest-even used to measure what the rounding mode is worth.
pub fn truncate_f32(v: &Array2<f64>) -> Array2<f64> {
    v.mapv(|x| {
        let r = x as f32;
        let r = if f64::from(r).abs() > x.abs() {
            f32::from_bits(r.to_bits() - 1) // one ulp toward zero (same sign)
        } else {
            r
        };
        f64::from(r)
    })
}

/// Least-squares line through (ln x, ln y): (slope, standard error of the
/// slope). Needs at least three points for a standard error (n − 2 degrees of
/// freedom); with two it is reported as infinite.
pub fn loglog_fit(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let (lx, ly): (Vec<f64>, Vec<f64>) = points.iter().map(|&(x, y)| (x.ln(), y.ln())).unzip();
    let (mx, my) = (lx.iter().sum::<f64>() / n, ly.iter().sum::<f64>() / n);
    let sxx: f64 = lx.iter().map(|x| (x - mx) * (x - mx)).sum();
    let sxy: f64 = lx.iter().zip(&ly).map(|(x, y)| (x - mx) * (y - my)).sum();
    let slope = sxy / sxx;
    let icpt = my - slope * mx;
    let rss: f64 = lx
        .iter()
        .zip(&ly)
        .map(|(x, y)| (y - icpt - slope * x).powi(2))
        .sum();
    let se = if points.len() > 2 {
        (rss / (n - 2.0) / sxx).sqrt()
    } else {
        f64::INFINITY
    };
    (slope, se)
}

/// One-sided 95% Student-t quantile t(0.95; df), the standard tabulated values
/// (a property of the t distribution, not a tuned number). The slope's
/// one-sided 95% upper confidence limit is slope + t·SE.
pub fn t95(df: usize) -> f64 {
    const T: [f64; 10] = [
        6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812,
    ];
    T[df.clamp(1, 10) - 1]
}
