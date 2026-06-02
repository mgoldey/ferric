//! Minimax-Laplace quadrature for `1/x` on `[1, R]`.
//!
//! Provides exponents `t_k` and weights `w_k` such that
//! `1/x ≈ Σ_k w_k exp(-t_k x)` is uniformly accurate on a chosen range.
//!
//! Used by:
//! - Laplace-MP2 (`ferric-mp2::laplace`) — denominator decomposition of MP2.
//! - Laplace-separable χ₀ for PDEP-RPA (`ferric-rpa::laplace_chi0`) — orbital
//!   energy-gap factorization of the static/imaginary-frequency response.
//!
//! Tables are taken from Takatsuka, Ten-no, Hackbusch (JCP 129, 044112, 2008)
//! as bundled in Helmich-Paris' `laplace-minimax` library.

/// A normalized minimax-Laplace quadrature: exponents `t_k` and weights `w_k`
/// rescaled to a problem-specific energy range `[ymin, ymax]`.
///
/// After construction, `1/x ≈ Σ_k w_k exp(-t_k x)` holds with minimax error on
/// `[ymin, ymax]`.
#[derive(Debug, Clone)]
pub struct LaplaceQuadrature {
    /// Requested number of quadrature points (table size may differ if unsupported).
    pub n_quad: usize,
    /// Rescaled exponents `t_k` (length = number of points actually used).
    pub points: Vec<f64>,
    /// Rescaled weights `w_k`.
    pub weights: Vec<f64>,
}

impl LaplaceQuadrature {
    /// Construct a quadrature for the energy-gap range `[ymin, ymax]`.
    ///
    /// Picks the tabulated minimax data with `R_tab ≥ ymax/ymin` (with a 1%
    /// slack to absorb floating-point round-off) and rescales:
    ///   `t_actual = t_table / ymin`, `w_actual = w_table / ymin`.
    pub fn new(n_quad: usize, ymin: f64, ymax: f64) -> Self {
        let r = ymax / ymin;
        let (raw_t, raw_w) = select_minimax_points(n_quad, r);
        let points: Vec<f64> = raw_t.iter().map(|&t| t / ymin).collect();
        let weights: Vec<f64> = raw_w.iter().map(|&w| w / ymin).collect();
        Self { n_quad, points, weights }
    }

    /// Length of the actual quadrature (may differ from `n_quad` for
    /// unsupported sizes — `select_minimax_points` falls back to k=7).
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Select minimax quadrature exponents and weights for `1/x` on `[1, R]`.
///
/// Data from Takatsuka, Ten-no, Hackbusch (JCP 129, 044112, 2008)
/// via Helmich-Paris `laplace-minimax`.
/// Returns `(exponents, weights)` for the unnormalized interval `[1, R]`.
///
/// If `k` is not directly tabulated, falls back to the k=7 table — callers
/// should validate convergence externally.
pub fn select_minimax_points(k: usize, r: f64) -> (Vec<f64>, Vec<f64>) {
    let table: &[MinimaxEntry] = match k {
        3 => MINIMAX_K3,
        5 => MINIMAX_K5,
        7 => MINIMAX_K7,
        _ => MINIMAX_K7,
    };

    for (r_tab, t, w) in table.iter() {
        if *r_tab >= r * 0.99 {
            return (t.to_vec(), w.to_vec());
        }
    }
    let (_, t, w) = &table[table.len() - 1];
    (t.to_vec(), w.to_vec())
}

type MinimaxEntry = (f64, &'static [f64], &'static [f64]);

static MINIMAX_K3: &[MinimaxEntry] = &[
    (5.0,   &[1.6607313750141492e-01, 9.7843720854537763e-01, 3.0530159808682455e+00],
            &[4.3657921050336840e-01, 1.2723305210744182e+00, 3.2151540832501007e+00]),
    (10.0,  &[1.0644554843011034e-01, 6.7919336467354152e-01, 2.4024163101166760e+00],
            &[2.8473486758368682e-01, 9.5831151180023555e-01, 2.8443772254878890e+00]),
    (20.0,  &[6.7901368940589263e-02, 4.8979488750896610e-01, 1.9962015809195739e+00],
            &[1.8718659453518208e-01, 7.6338571884375950e-01, 2.6169711142660255e+00]),
    (50.0,  &[3.9082098432306915e-02, 3.5092306837783149e-01, 1.6921947262166477e+00],
            &[1.1533107566965783e-01, 6.1942325781831142e-01, 2.4405036326787770e+00]),
    (100.0, &[2.8403804819953280e-02, 2.9983005709471117e-01, 1.5761907592561200e+00],
            &[8.9215707908966990e-02, 5.6479425010878515e-01, 2.3699428171677330e+00]),
];

static MINIMAX_K5: &[MinimaxEntry] = &[
    (5.0,   &[1.0333512117595546e-01, 5.6589985547987387e-01, 1.5002594864446179e+00, 3.1563889823905420e+00, 6.1935841722085909e+00],
            &[2.6751449382571008e-01, 6.7316481854106480e-01, 1.2341495375414908e+00, 2.1730749987606481e+00, 4.2400491045957720e+00]),
    (10.0,  &[6.4797316294974178e-02, 3.6329423979113412e-01, 1.0098084995269310e+00, 2.2813077174805523e+00, 4.8771539519428151e+00],
            &[1.6861003324210627e-01, 4.4474622263913965e-01, 8.9143002224546830e-01, 1.7540776690871602e+00, 3.7773593670209080e+00]),
    (20.0,  &[3.9791035518631973e-02, 2.3131332253827694e-01, 6.8992455340431735e-01, 1.7098435082055961e+00, 4.0149225768154402e+00],
            &[1.0434198456045442e-01, 2.9555856650050433e-01, 6.6783133886464374e-01, 1.4788816616452081e+00, 3.4728708916802091e+00]),
    (50.0,  &[2.0598516936418058e-02, 1.2946507144298772e-01, 4.3979723967050899e-01, 1.2499372840249323e+00, 3.2997722744581939e+00],
            &[5.4928968471960854e-02, 1.7972260846046623e-01, 4.8760926843571956e-01, 1.2421875649935035e+00, 3.2030368198904093e+00]),
    (100.0, &[1.2552727540467989e-02, 8.6549394877289257e-02, 3.3100124293713873e-01, 1.0393832416814095e+00, 2.9584699518773347e+00],
            &[3.4210303487463151e-02, 1.3026229608206730e-01, 4.0415841564587901e-01, 1.1232181675451096e+00, 3.0638169728620763e+00]),
    (500.0, &[4.4700910206532437e-03, 4.3042233518954816e-02, 2.1332569755781333e-01, 7.9513224013521722e-01, 2.5436793036816292e+00],
            &[1.3537186918918044e-02, 7.8245171363032009e-02, 3.0503984426641567e-01, 9.7044129821768366e-01, 2.8813614470814399e+00]),
    (1000.0,&[3.4004600489073895e-03, 3.7149816796614998e-02, 1.9603210903786736e-01, 7.5684838765183138e-01, 2.4761708329681671e+00],
            &[1.0838272657023281e-02, 7.0764497808673207e-02, 2.8913820910474880e-01, 9.4454172629160815e-01, 2.8500005633404064e+00]),
];

static MINIMAX_K7: &[MinimaxEntry] = &[
    (5.0,   &[7.5178012053935581e-02, 4.0394057377108084e-01, 1.0300953839123785e+00, 2.0257208844822334e+00, 3.5300416178849785e+00, 5.8202600945046035e+00, 9.5718793022987914e+00],
            &[1.9380100796794852e-01, 4.6924355863005290e-01, 7.9463765966869560e-01, 1.2188241997775222e+00, 1.8330663504453952e+00, 2.8441417219183061e+00, 5.0081428000286365e+00]),
    (10.0,  &[4.6784971919695058e-02, 2.5413109177030713e-01, 6.6202063661281685e-01, 1.3468236945914096e+00, 2.4608175652259985e+00, 4.3018718689144126e+00, 7.5591271744403050e+00],
            &[1.2090244093952099e-01, 2.9921214711072491e-01, 5.2876236081329564e-01, 8.6523254061261390e-01, 1.4096599453909606e+00, 2.3721982300004751e+00, 4.4850705608595405e+00]),
    (20.0,  &[2.8397875019357810e-02, 1.5669465451979245e-01, 4.2128353664098478e-01, 9.0071033948500667e-01, 1.7551920884002583e+00, 3.2943879648814645e+00, 6.2167521169331161e+00],
            &[7.3640943793741087e-02, 1.8809607758877431e-01, 3.5358561229255181e-01, 6.3102636982655691e-01, 1.1268723268357688e+00, 2.0532076958089367e+00, 4.1312146140908652e+00]),
    (50.0,  &[1.4346616954119040e-02, 8.1686636965124057e-02, 2.3392918256659598e-01, 5.4781736961611505e-01, 1.1828967187208399e+00, 2.4526048384096861e+00, 5.0658409539581051e+00],
            &[3.7455686176677001e-02, 1.0185840534484672e-01, 2.1485110258922235e-01, 4.3851262496240967e-01, 8.8110126016728485e-01, 1.7621025260651733e+00, 3.8032676122312763e+00]),
    (100.0, &[8.4890821340208086e-03, 5.0116642014961708e-02, 1.5366998285929265e-01, 3.9160355318450513e-01, 9.1783364668505174e-01, 2.0447798621093103e+00, 4.4889127432163773e+00],
            &[2.2333642551176373e-02, 6.5162010730403855e-02, 1.5346948447776387e-01, 3.4692296202841277e-01, 7.5471793703896095e-01, 1.6044221299260411e+00, 3.6234684847661183e+00]),
    (500.0, &[2.5357978643651921e-03, 1.7578759091398184e-02, 6.7613168610773614e-02, 2.1262834171859676e-01, 5.9156023005630420e-01, 1.5113889410817196e+00, 3.7022956650360443e+00],
            &[6.9203048135253430e-03, 2.6567898179860355e-02, 8.2863747425280715e-02, 2.2897941405638794e-01, 5.7693244244983499e-01, 1.3704988341020230e+00, 3.3539045747733187e+00]),
    (1000.0,&[1.5465907100920744e-03, 1.2056643172650844e-02, 5.1852489933254427e-02, 1.7661795467136937e-01, 5.2036352547560127e-01, 1.3877091798262136e+00, 3.5126938632453744e+00],
            &[4.3590321081121135e-03, 1.9739937643250896e-02, 6.8465314092830382e-02, 2.0200683741399486e-01, 5.3305670896838087e-01, 1.3100924544159886e+00, 3.2836589740207804e+00]),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadrature_rescaling_basic() {
        let q = LaplaceQuadrature::new(7, 1.0, 50.0);
        assert_eq!(q.points.len(), 7);
        assert_eq!(q.weights.len(), 7);
        let (raw_t, _) = select_minimax_points(7, 50.0);
        assert!((q.points[0] - raw_t[0]).abs() < 1e-15);
    }

    #[test]
    fn approximates_one_over_x() {
        // Σ_k w_k exp(-t_k x) ≈ 1/x on [ymin, ymax].
        let ymin = 0.5;
        let ymax = 20.0;
        let q = LaplaceQuadrature::new(7, ymin, ymax);
        for &x in &[ymin, 1.0, 5.0, ymax] {
            let approx: f64 =
                q.points.iter().zip(q.weights.iter()).map(|(&t, &w)| w * (-t * x).exp()).sum();
            let exact = 1.0 / x;
            let rel = ((approx - exact) / exact).abs();
            assert!(rel < 1e-3, "x={x}: approx={approx}, exact={exact}, rel={rel}");
        }
    }
}
