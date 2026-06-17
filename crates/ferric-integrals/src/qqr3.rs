//! QQR-style distance-aware screening bounds for 3-index integrals (P|μν).
//!
//! Refines the 3-index Schwarz bound  `|(P|μν)| ≤ Q3[P] · Q(μ,ν)`  with a
//! distance/extent envelope. The bound is
//!
//! ```text
//!   |(P|μν)|  ≤  Q3[P] · Q(μ,ν) · min(1, (ext_aux[P] + ext_obs(μν)) / R_eff)
//! ```
//!
//! where `R_eff = max(0, R − ext_aux[P] − ext_obs(μν))` is the EDGE-TO-EDGE
//! distance between the aux blob P and the obs-pair blob μν, and R is the
//! distance between their charge centers.
//!
//! ## Why this form (and why the earlier forms were INVALID)
//!
//! `(P|μν)` is the Coulomb interaction of two Gaussian charge clouds. The
//! leading multipole term decays as 1/R, so the distance factor must scale as
//! 1/R_eff — NOT `ext·ext/R` (which under-states the cloud charges and
//! collapses far too fast, ratio 14.8) nor `ext·ext/R²` (wrong power, 1/R² is
//! the FIELD not the potential). The numerator is `ext_sum` so the factor is
//! continuous and equals 1 at contact (R = ext_sum), reducing smoothly to the
//! Schwarz bound for overlapping/penetrating clouds where the multipole
//! expansion is invalid. Validated against the true dense integral on water +
//! alkane_6: worst |true|/bound = 0.9999 ≤ 1 (see `tests` + `examples/`).
//!
//! ## erfc attenuation is carried by the Schwarz factors, not a distance factor
//!
//! For an `ErfcCoulomb` operator the per-operator Schwarz self-integrals
//! `Q3[P]` and `Q(μ,ν)` are computed with the erfc kernel and are therefore
//! already smaller than their Coulomb counterparts — that is where the
//! attenuation enters. The Coulomb `ext_sum/R_eff` envelope is already nearly
//! tight (ratio 0.9999), leaving NO headroom to multiply in a separate
//! `erfc(ω̃·R)` long-range factor: doing so over-suppresses real long-range
//! triples and makes the bound INVALID (measured worst ratio 1.7–5.3). So the
//! same Coulomb envelope is used for both operators; erfc screening benefit
//! comes entirely from the smaller erfc Schwarz factors.

use crate::basis_bridge::PreparedBasis;
use crate::engine::Engine;
use crate::operator::Operator;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ndarray::Array2;

/// Tight integral precision for the Schwarz self-integrals that seed the QQR-3
/// bound. The default screening precision (1e-14) lets libint round a diffuse
/// obs-pair self-integral `(μν|μν)` down to EXACTLY 0 (verified on alkane_6:
/// shells 70,10 give true 8.8e-11 but 0.0 at 1e-14). A zero Schwarz factor
/// would zero the whole bound and silently drop a triple whose true 3-index
/// integral is ~1e-4 — an invalid bound. Computing the seed self-integrals at
/// 1e-30 recovers the true tiny value so the Cauchy–Schwarz product stays a
/// genuine upper bound.
const QQR3_SEED_PREC: f64 = 1e-30;

/// Distance-aware screening for 3-index ERIs.
///
/// Holds per-aux-shell self-Schwarz and pair-center/extent, plus per-obs-pair
/// Schwarz and pair-center/extent. `estimate3(P, s1, s2)` returns the QQR bound.
pub struct QqrBounds3 {
    /// Per-aux-shell Schwarz: Q3[P] = sqrt(max_a |(P_a|P_a)|).
    q3: Vec<f64>,
    /// Aux-shell self-pair extents:  ext_aux[P] = 1/sqrt(2·α_min(P)).
    aux_extents: Vec<f64>,
    /// Aux-shell centers (= atom centers of the aux shell).
    aux_centers: Vec<[f64; 3]>,
    /// Aux-shell minimum (most diffuse) primitive exponent α_P. Used to form
    /// the effective erfc range parameter ω̃ (the finite blob width softens the
    /// attenuation, so the valid envelope uses ω̃ < ω, not bare ω).
    aux_min_exp: Vec<f64>,

    /// Obs-pair Schwarz matrix, Q(s1,s2) = sqrt(|(s1,s2|s1,s2)|).
    q_obs: Array2<f64>,
    /// Obs-pair extents and centers, indexed `i*nsh_obs + j`.
    obs_pair_extents: Vec<f64>,
    obs_pair_centers: Vec<[f64; 3]>,
    /// Obs-pair summed min exponents α_i_min + α_j_min, indexed `i*nsh_obs + j`.
    obs_pair_alpha: Vec<f64>,

    op: Operator,
    nsh_obs: usize,
    nsh_aux: usize,
}

impl QqrBounds3 {
    /// Build the QQR-3 bounds for an obs/aux basis pair under operator `op`.
    pub fn new(
        op: Operator,
        mol: &Molecule,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
    ) -> Result<Self, FerricError> {
        // Seed Schwarz factors at TIGHT precision so diffuse-pair self-integrals
        // do not underflow to exactly 0 (which would zero the bound and drop a
        // real triple — see QQR3_SEED_PREC).
        let q3 = q3_aux_tight(op, aux)?;
        let q_obs = q_obs_tight(op, obs)?;

        // Per-shell minimum exponent and atom center for each basis, mirroring
        // the construction in ferric_scf::qqr::QqrBounds::new. Basis sets are
        // recovered from the PreparedBasis (which retains a reference to them).
        let (obs_min, obs_origin) = collect_min_exponents_and_origins(mol, obs.basis_set());
        let (aux_min, aux_origin) = collect_min_exponents_and_origins(mol, aux.basis_set());
        assert_eq!(obs_min.len(), obs.nshells());
        assert_eq!(aux_min.len(), aux.nshells());
        let nsh_obs = obs.nshells();
        let nsh_aux = aux.nshells();

        // Obs pair centers/extents: same definition as 4-center QQR.
        let mut obs_pair_centers = vec![[0.0; 3]; nsh_obs * nsh_obs];
        let mut obs_pair_extents = vec![0.0; nsh_obs * nsh_obs];
        let mut obs_pair_alpha = vec![0.0; nsh_obs * nsh_obs];
        for i in 0..nsh_obs {
            let ai = obs_min[i];
            let ri = obs_origin[i];
            for j in 0..nsh_obs {
                let aj = obs_min[j];
                let rj = obs_origin[j];
                let sum = ai + aj;
                let idx = i * nsh_obs + j;
                obs_pair_centers[idx] = [
                    (ai * ri[0] + aj * rj[0]) / sum,
                    (ai * ri[1] + aj * rj[1]) / sum,
                    (ai * ri[2] + aj * rj[2]) / sum,
                ];
                obs_pair_extents[idx] = 1.0 / sum.sqrt();
                obs_pair_alpha[idx] = sum;
            }
        }

        // Aux self-pair extents/centers (P with itself).
        let mut aux_extents = vec![0.0; nsh_aux];
        let mut aux_centers = vec![[0.0; 3]; nsh_aux];
        let mut aux_min_exp = vec![0.0; nsh_aux];
        for p in 0..nsh_aux {
            let ap = aux_min[p];
            aux_centers[p] = aux_origin[p];
            aux_extents[p] = 1.0 / (2.0 * ap).sqrt();
            aux_min_exp[p] = ap;
        }

        Ok(QqrBounds3 {
            q3,
            aux_extents,
            aux_centers,
            aux_min_exp,
            q_obs,
            obs_pair_extents,
            obs_pair_centers,
            obs_pair_alpha,
            op,
            nsh_obs,
            nsh_aux,
        })
    }

    pub fn nsh_obs(&self) -> usize { self.nsh_obs }
    pub fn nsh_aux(&self) -> usize { self.nsh_aux }
    pub fn op(&self) -> Operator { self.op }

    #[doc(hidden)]
    pub fn debug_distance(&self, p: usize, s1: usize, s2: usize) -> f64 {
        let idx = s1 * self.nsh_obs + s2;
        let c = self.obs_pair_centers[idx];
        let a = self.aux_centers[p];
        ((c[0]-a[0]).powi(2)+(c[1]-a[1]).powi(2)+(c[2]-a[2]).powi(2)).sqrt()
    }
    #[doc(hidden)]
    pub fn debug_aux_extent(&self, p: usize) -> f64 { self.aux_extents[p] }
    #[doc(hidden)]
    pub fn debug_obs_extent(&self, s1: usize, s2: usize) -> f64 {
        self.obs_pair_extents[s1 * self.nsh_obs + s2]
    }
    #[doc(hidden)]
    pub fn debug_omega_tilde(&self, p: usize, s1: usize, s2: usize) -> f64 {
        let a = self.aux_min_exp[p];
        let b = self.obs_pair_alpha[s1 * self.nsh_obs + s2];
        let gamma = a * b / (a + b);
        let w2 = self.op.omega * self.op.omega;
        (w2 * gamma / (w2 + gamma)).sqrt()
    }

    /// Upper bound estimate for |(P | s1 s2)|.
    pub fn estimate3(&self, p: usize, s1: usize, s2: usize) -> f64 {
        let schwarz_est = self.q3[p] * self.q_obs[(s1, s2)];
        if schwarz_est == 0.0 {
            return 0.0;
        }
        let idx_obs = s1 * self.nsh_obs + s2;
        let c_obs = self.obs_pair_centers[idx_obs];
        let c_aux = self.aux_centers[p];
        let dx = c_obs[0] - c_aux[0];
        let dy = c_obs[1] - c_aux[1];
        let dz = c_obs[2] - c_aux[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r < 1e-14 {
            return schwarz_est;
        }
        // QQR multipole distance envelope. `(P|μν)` is the Coulomb interaction
        // of two Gaussian charge clouds; its leading (monopole) term decays as
        // 1/R, so the distance factor scales as 1/R_eff. The numerator is the
        // sum of cloud extents so the factor is continuous and equals 1 at
        // contact (R = ext_sum), reducing to the Schwarz bound for
        // overlapping/penetrating clouds where the multipole expansion breaks
        // down. The earlier `ext·ext/R_eff` (and `/R_eff²`) forms under-stated
        // the cloud charges and collapsed far too fast — invalid bounds that
        // silently dropped real integrals (worst |true|/bound 3.8–14.8).
        let ext_obs_e = self.obs_pair_extents[idx_obs];
        let ext_aux_e = self.aux_extents[p];
        let ext_sum = ext_obs_e + ext_aux_e;
        // Edge-to-edge separation: when the clouds overlap (r ≤ ext_sum) the
        // integral is dominated by the penetration region where the 1/R decay
        // has not set in, so the factor falls back to 1 (Schwarz).
        let r_eff = (r - ext_sum).max(0.0);
        let decay = if r_eff > 0.0 {
            (ext_sum / r_eff).min(1.0)
        } else {
            1.0
        };
        // NOTE: no separate erfc(ω̃·R) factor. For an ErfcCoulomb operator the
        // attenuation is already carried by the per-operator erfc Schwarz
        // factors (`q3`, `q_obs` computed with the erfc kernel). The Coulomb
        // ext_sum/R_eff envelope is already nearly tight (ratio 0.9999), so an
        // extra erfc factor < 1 would over-suppress real long-range triples and
        // break validity (measured worst ratio 1.7–5.3).
        schwarz_est * decay
    }
}

/// Complementary error function `erfc(x) = 1 − erf(x)` for `x ≥ 0`.
///
/// Rust's std has no erfc; this is W. J. Cody's rational Chebyshev
/// approximation (the same scheme used by most libm implementations),
/// accurate to ~1e-15 — far more than a screening bound needs. Only the
/// `x ≥ 0` branch is exercised here (R·ω̃ is always non-negative), but the
/// full odd reflection is included for safety.
pub fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    // erfc(x) = exp(-x²) · (erfc-scaled rational). Use the standard
    // continued-fraction-free rational forms split at x = 0.5 and x = 4.0.
    let z = x * x;
    if x < 0.5 {
        // erf via series-equivalent rational (Cody region 1), then 1 − erf.
        const A: [f64; 5] = [
            3.16112374387056560e0,
            1.13864154151050156e2,
            3.77485237685302021e2,
            3.20937758913846947e3,
            1.85777706184603153e-1,
        ];
        const B: [f64; 4] = [
            2.36012909523441209e1,
            2.44024637934444173e2,
            1.28261652607737228e3,
            2.84423683343917062e3,
        ];
        let num = ((((A[4] * z + A[0]) * z + A[1]) * z + A[2]) * z + A[3]) * x;
        let den = (((z + B[0]) * z + B[1]) * z + B[2]) * z + B[3];
        return 1.0 - num / den;
    }
    if x < 4.0 {
        // Cody region 2.
        const C: [f64; 9] = [
            5.64188496988670089e-1,
            8.88314979438837594e0,
            6.61191906371416295e1,
            2.98635138197400131e2,
            8.81952221241769090e2,
            1.71204761263407058e3,
            2.05107837782607147e3,
            1.23033935479799725e3,
            2.15311535474403846e-8,
        ];
        const D: [f64; 8] = [
            1.57449261107098347e1,
            1.17693950891312499e2,
            5.37181101862009858e2,
            1.62138957456669019e3,
            3.29079923573345963e3,
            4.36261909014324716e3,
            3.43936767414372164e3,
            1.23033935480374942e3,
        ];
        let mut num = C[8] * x + C[0];
        let mut den = x + D[0];
        for k in 1..8 {
            num = num * x + C[k];
            den = den * x + D[k];
        }
        return (-z).exp() * num / den;
    }
    // Cody region 3 (large x asymptotic).
    const P: [f64; 6] = [
        3.05326634961232344e-1,
        3.60344899949804439e-1,
        1.25781726111229246e-1,
        1.60837851487422766e-2,
        6.58749161529837803e-4,
        1.63153871373020978e-2,
    ];
    const Q: [f64; 5] = [
        2.56852019228982242e0,
        1.87295284992346047e0,
        5.27905102951428412e-1,
        6.05183413124413191e-2,
        2.33520497626869185e-3,
    ];
    let zinv = 1.0 / z;
    let mut num = P[5] * zinv + P[0];
    let mut den = zinv + Q[0];
    for k in 1..5 {
        num = num * zinv + P[k];
        den = den * zinv + Q[k];
    }
    let r = zinv * num / den;
    const SQRT_PI_INV: f64 = 0.564189583547756287; // 1/sqrt(π)
    (-z).exp() / x * (SQRT_PI_INV - r)
}

/// Per-aux-shell Schwarz `Q3[P] = sqrt(max_a |(P_a|P_a)|)` at tight precision.
///
/// Same quantity as [`crate::schwarz::schwarz3_aux`] but computed at
/// [`QQR3_SEED_PREC`] so diffuse self-integrals are not screened to 0.
fn q3_aux_tight(op: Operator, dfbs: &PreparedBasis) -> Result<Vec<f64>, FerricError> {
    let nsh = dfbs.nshells();
    let dims = dfbs.shell_dims();
    let mut eng = Engine::new_2center(op, dfbs, QQR3_SEED_PREC)?;
    let mut q3 = vec![0.0f64; nsh];
    for p in 0..nsh {
        let block = eng.compute_eri2(dfbs, p, p);
        let np = dims[p];
        let mut maxv = 0.0f64;
        for a in 0..np {
            let v = block[a * np + a].abs();
            if v > maxv {
                maxv = v;
            }
        }
        q3[p] = maxv.sqrt();
    }
    Ok(q3)
}

/// Obs-pair Schwarz matrix `Q(s1,s2) = sqrt(max |(s1 s2|s1 s2)|)` at tight
/// precision, so diffuse pair self-integrals are not screened to 0.
///
/// Returns a dense `nsh × nsh` matrix mirroring [`crate::schwarz::schwarz`],
/// but built from the diagonal `(μν|μν)` elements at [`QQR3_SEED_PREC`].
fn q_obs_tight(op: Operator, obs: &PreparedBasis) -> Result<Array2<f64>, FerricError> {
    let nsh = obs.nshells();
    let dims = obs.shell_dims();
    let mut eng = Engine::new_2e(op, obs, QQR3_SEED_PREC)?;
    let mut q = Array2::zeros((nsh, nsh));
    for s1 in 0..nsh {
        for s2 in 0..=s1 {
            let mut maxv = 0.0f64;
            if let Some(block) = eng.compute_quartet(obs, s1, s2, s1, s2) {
                let n1 = dims[s1];
                let n2 = dims[s2];
                // Diagonal element (μν|μν) lives at flat index ((i*n2+j)*n1+i)*n2+j.
                for i in 0..n1 {
                    for j in 0..n2 {
                        let idx = ((i * n2 + j) * n1 + i) * n2 + j;
                        if idx < block.len() {
                            let v = block[idx].abs();
                            if v > maxv {
                                maxv = v;
                            }
                        }
                    }
                }
            }
            let val = maxv.sqrt();
            q[(s1, s2)] = val;
            q[(s2, s1)] = val;
        }
    }
    Ok(q)
}

fn collect_min_exponents_and_origins(
    mol: &Molecule,
    bs: &BasisSet,
) -> (Vec<f64>, Vec<[f64; 3]>) {
    let mut min_exps = Vec::new();
    let mut origins = Vec::new();
    for atom in &mol.atoms {
        let shells = bs.for_element(atom.z).unwrap();
        for sh in shells {
            let alpha_min = sh.exponents.iter().cloned().fold(f64::INFINITY, f64::min);
            min_exps.push(alpha_min);
            origins.push([atom.x, atom.y, atom.zpos]);
        }
    }
    (min_exps, origins)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threeindex::eri3_tensor;
    use ferric_core::basis;

    #[test]
    fn test_erfc_accuracy() {
        // Reference erfc values (computed in high precision) — the Cody
        // approximation must match to ~1e-12.
        let cases = [
            (0.0, 1.0),
            (0.1, 0.8875370839817152),
            (0.25, 0.7236736098317631),
            (0.5, 0.4795001221869535),
            (1.0, 0.15729920705028513),
            (2.0, 0.004677734981047266),
            (3.0, 2.209049699858544e-5),
            (5.0, 1.5374597944280349e-12),
        ];
        for (x, expect) in cases {
            let got = erfc(x);
            let err = (got - expect).abs();
            // Relative tolerance for the tiny tail values.
            let tol = 1e-12 * expect.max(1e-12) + 1e-13;
            assert!(err < tol.max(1e-12),
                "erfc({x}) = {got}, expected {expect}, err {err:.3e}");
        }
        assert!((erfc(0.0) - 1.0).abs() < 1e-15);
        // monotone decreasing
        let mut prev = erfc(0.0);
        for i in 1..=200 {
            let x = i as f64 * 0.05;
            let v = erfc(x);
            assert!(v <= prev + 1e-15, "erfc not monotone at x={x}");
            assert!(v >= 0.0, "erfc negative at x={x}");
            prev = v;
        }
    }

    /// THE missing validity test: estimate3 must be a TRUE upper bound on the
    /// actual dense erfc integral |(P|μν)| — not just ≤ Schwarz. Its absence is
    /// why the bare-Gaussian under-estimating bound shipped. We assert
    /// estimate3 ≥ max-element |(P|s1 s2)| for EVERY shell triple.
    fn assert_estimate3_bounds_true_integral(path: &str) {
        let mol = Molecule::load_xyz(path).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::erfc(0.222);
        let qqr3 = QqrBounds3::new(op, &mol, &obs, &aux).unwrap();
        let dense = eri3_tensor(op, &obs, &aux).unwrap();

        let dims_obs = obs.shell_dims();
        let offs_obs = obs.shell_offsets();
        let dims_aux = aux.shell_dims();
        let offs_aux = aux.shell_offsets();

        let mut worst_ratio = 0.0f64; // |true| / bound; must stay ≤ 1
        for p in 0..qqr3.nsh_aux() {
            for s1 in 0..qqr3.nsh_obs() {
                for s2 in 0..=s1 {
                    let bound = qqr3.estimate3(p, s1, s2);
                    // Max |integral| over the AO functions in this shell triple.
                    let mut tru = 0.0f64;
                    for pp in 0..dims_aux[p] {
                        for ii in 0..dims_obs[s1] {
                            for jj in 0..dims_obs[s2] {
                                let v = dense[(offs_aux[p] + pp,
                                               offs_obs[s1] + ii,
                                               offs_obs[s2] + jj)].abs();
                                if v > tru { tru = v; }
                            }
                        }
                    }
                    assert!(bound >= tru - 1e-12,
                        "estimate3({p},{s1},{s2}) = {bound:.6e} UNDER-estimates \
                         true |(P|μν)| = {tru:.6e} (deficit {:.3e})",
                        tru - bound);
                    if tru > 1e-14 {
                        let ratio = tru / bound;
                        if ratio > worst_ratio { worst_ratio = ratio; }
                    }
                }
            }
        }
        eprintln!("{path}: worst |true|/bound = {worst_ratio:.4} (must be ≤ 1)");
        assert!(worst_ratio <= 1.0 + 1e-9,
            "bound is invalid: worst ratio {worst_ratio} > 1");
    }

    #[test]
    fn test_estimate3_is_valid_upper_bound_water() {
        assert_estimate3_bounds_true_integral("../../testdata/molecules/water.xyz");
    }

    #[test]
    fn test_estimate3_is_valid_upper_bound_alkane6() {
        assert_estimate3_bounds_true_integral("../../testdata/molecules/alkane_6.xyz");
    }

    #[test]
    fn test_qqr3_le_schwarz3() {
        // QQR-3 estimate must be ≤ the Schwarz-3 estimate (Q3·Q_obs) for every
        // shell triple — the distance/operator factor is always ≤ 1.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::erfc(0.222);
        let qqr3 = QqrBounds3::new(op, &mol, &obs, &aux).unwrap();
        for p in 0..qqr3.nsh_aux() {
            for s1 in 0..qqr3.nsh_obs() {
                for s2 in 0..qqr3.nsh_obs() {
                    let schwarz3 = qqr3.q3[p] * qqr3.q_obs[(s1, s2)];
                    let q = qqr3.estimate3(p, s1, s2);
                    assert!(
                        q <= schwarz3 + 1e-14,
                        "QQR3({p},{s1},{s2}) = {q} > Schwarz3 = {schwarz3}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_qqr3_erfc_tighter_than_qqr3_coulomb() {
        // Same Schwarz factors, but ErfcCoulomb adds the exp(-ω²R²) decay,
        // so erfc-QQR3 must be ≤ Coulomb-QQR3 at every triple, strictly
        // smaller for some distant triple.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let qqr3_c = QqrBounds3::new(
            Operator::coulomb(), &mol, &obs, &aux).unwrap();
        let qqr3_e = QqrBounds3::new(
            Operator::erfc(0.5), &mol, &obs, &aux).unwrap();
        let mut found_strictly_smaller = false;
        for p in 0..qqr3_c.nsh_aux() {
            for s1 in 0..qqr3_c.nsh_obs() {
                for s2 in 0..qqr3_c.nsh_obs() {
                    let c = qqr3_c.estimate3(p, s1, s2);
                    let e = qqr3_e.estimate3(p, s1, s2);
                    // Note: Q3 and Q_obs differ per operator (libint computes
                    // erfc Schwarz factors that are themselves smaller), so we
                    // only check the distance envelope on a normalized ratio
                    // when both are nonzero. The strict-smaller condition is
                    // about the *combined* bound being tighter.
                    assert!(e <= c + 1e-14,
                        "erfc QQR3 not ≤ Coulomb QQR3 at ({p},{s1},{s2}): {e} vs {c}");
                    if c > 1e-10 && (e / c) < 0.99 {
                        found_strictly_smaller = true;
                    }
                }
            }
        }
        assert!(found_strictly_smaller,
            "erfc QQR3 should be strictly smaller than Coulomb QQR3 somewhere");
    }
}
