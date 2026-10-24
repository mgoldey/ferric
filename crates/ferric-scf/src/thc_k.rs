//! THC-RI-K: exchange via least-squares tensor hypercontraction (Lee, Lin &
//! Head-Gordon, JCTC 16, 243 (2020), Eq. 38). AO-THC (`M^[nn]`) variant.
//!
//! **EXPERIMENTAL, feature `thc`, never auto-selected, not wired into any SCF
//! driver, CLI or Python surface.** The measured THC-K vs DF-K comparison on
//! ferric is recorded in `reference/thc/thc-lane-2026-10.md`; read it before
//! using this builder for anything but correctness work.
//!
//! # What this computes
//!
//! ```text
//!   (μλ|νσ) ≈ Σ_KL ω_μK ω_λK M_KL ω_νL ω_σL
//!   K_μν    = Σ_λσ (μλ|νσ) D_λσ = Σ_KL ω_μK [M ∘ (ωᵀ D ω)]_KL ω_νL
//! ```
//!
//! `ω_μK = χ_μ(r_K)` is the AO collocation on `N_IP` interpolation points and
//! `M` the THC core. `K` follows the [`KBuilder`](crate::fock::KBuilder) convention (no ½, no sign).
//! The per-build work is `O(nbf · N_IP²)` dense GEMM.
//!
//! # Fit target and the trivial limit
//!
//! `M` is the least-squares fit of the SAME density-fitted ERIs [`DfK`] uses,
//! `G = BᵀB` with `B = V^{-1/2}(P|μν)` and the identical lindep-filtered
//! `V^{-1/2}`:
//!
//! ```text
//!   X[(μν), K] = ω_μK ω_νK,   S = XᵀX = (ωᵀω) ∘ (ωᵀω),
//!   W = B X  (naux × N_IP),   M = S⁺ WᵀW S⁺
//! ```
//!
//! When the selected columns of `X` span the whole symmetric pair space,
//! `X S⁺ Xᵀ` is the projector onto it and `X M Xᵀ = G` exactly, so THC-K
//! reproduces DF-K to round-off. That is the exactness anchor
//! (`thc_k_matches_exact_in_the_trivial_limit`). Neither `X` (`nbf² × npts`)
//! nor `G` (`nbf² × nbf²`) is ever formed: both identities above are exact
//! rewrites, which is what makes the setup run past a few dozen basis
//! functions at all.
//!
//! # Interpolation-point selection
//!
//! Greedy pivoted Cholesky of the product Gram matrix
//! `Γ_pq = (Σ_μ χ_μp χ_μq)²` over a Becke parent grid, columns generated on
//! demand. In exact arithmetic this picks the same pivots as column-pivoted QR
//! of `X` (Businger-Golub), which is what the pre-port branch used — at
//! `O(npts · N_IP)` memory instead of `O(nbf² · npts)`. It is NOT the ISDF/CVT
//! selection of the reference, which is expected to be more compact.
//!
//! # AO-THC, not MO-THC
//!
//! `M^[on]` depends on the density, which forces re-deriving the Fock matrix
//! from the THC energy (Lee et al. §3: ignoring it "leads to numerically
//! unstable behavior"). `M` here is density-independent and built once.
//!
//! [`DfK`]: crate::df_k::DfK

use crate::df_k::v_inv_sqrt_lindep;
use crate::fock::KBuilder;
use ferric_core::basis::BasisSet;
use ferric_core::error::FerricError;
use ferric_core::linalg::{eigh_dc, Uplo};
use ferric_core::mol::Molecule;
use ferric_integrals::ao_grid::eval_basis_on_points;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ndarray::{s, Array1, Array2, Axis};

/// Lee et al. §8: recommended `c_ISDF` for HF exchange. THC-RI-K is reported
/// non-variational (SCF collapse) below it, so it is a stability floor.
pub const DEFAULT_C_ISDF_HF: f64 = 8.5;
/// Lee et al. §8: recommended value for ωB97X-V exchange.
pub const DEFAULT_C_ISDF_RSH: f64 = 6.5;

/// Configuration for [`ThcK`].
#[derive(Debug, Clone)]
pub struct ThcKConfig {
    /// Requested `N_IP = c_isdf · naux` (clamped to the parent grid and to the
    /// numerical rank of the product space).
    pub c_isdf: f64,
    /// Becke parent-grid radial points.
    pub n_radial: usize,
    /// Becke parent-grid Lebedev order (ferric's table: 6/14/26/50/110/302).
    pub n_angular: usize,
    /// Pivoted-Cholesky stop: residual Gram diagonal below
    /// `rank_tol · max initial diagonal` ends the selection. The default
    /// 1e-18 is BELOW the ~1e-16 relative resolution of the `resid -= c²`
    /// downdate, so in practice the stop is set by rounding noise, not by an
    /// exact rank test: STO-3G water selects 37 points against a pair-space
    /// rank of 28 (the surplus columns are harmless, `S⁺` truncates them).
    pub rank_tol: f64,
    /// Relative eigenvalue drop for `S⁺`.
    pub pinv_tol: f64,
}

impl Default for ThcKConfig {
    fn default() -> Self {
        Self {
            c_isdf: DEFAULT_C_ISDF_HF,
            n_radial: 50,
            n_angular: 110,
            rank_tol: 1e-18,
            pinv_tol: 1e-12,
        }
    }
}

/// THC-RI-K exchange builder. Off by default; see the module docs.
pub struct ThcK {
    /// AO collocation on the selected points, `(nbf, n_ip)`.
    omega: Array2<f64>,
    /// THC core `M`, `(n_ip, n_ip)`, dense by construction.
    m: Array2<f64>,
    nbf: usize,
}

impl ThcK {
    /// Build the factorization (one-time setup, density-independent).
    pub fn new(
        mol: &Molecule,
        obs_bs: &BasisSet,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
        cfg: &ThcKConfig,
    ) -> Result<Self, FerricError> {
        if !(cfg.c_isdf > 0.0) {
            return Err(FerricError::General(format!(
                "ThcK: c_isdf must be positive, got {}",
                cfg.c_isdf
            )));
        }
        let chi = parent_collocation(mol, obs_bs, cfg)?;
        let want = (cfg.c_isdf * dfbs.nbasis() as f64).ceil() as usize;
        let pivots = select_points_pivoted_cholesky(&chi, want, cfg.rank_tol);
        let omega = unit_columns(chi.select(Axis(1), &pivots));
        let m = fit_core(obs, dfbs, op, &omega, cfg.pinv_tol)?;
        Self::from_parts(omega, m)
    }

    /// Assemble from an explicit collocation `omega` `(nbf, n_ip)` and core
    /// `m` `(n_ip, n_ip)`. Used by the anchor tests and by the build-cost
    /// harness, whose per-build cost depends only on the shapes.
    pub fn from_parts(omega: Array2<f64>, m: Array2<f64>) -> Result<Self, FerricError> {
        let n_ip = omega.ncols();
        if m.dim() != (n_ip, n_ip) {
            return Err(FerricError::General(format!(
                "ThcK: core is {:?}, expected ({n_ip}, {n_ip})",
                m.dim()
            )));
        }
        let nbf = omega.nrows();
        Ok(Self { omega, m, nbf })
    }

    /// Interpolation points actually used.
    pub fn n_interpolation_points(&self) -> usize {
        self.omega.ncols()
    }

    /// `K = ω (M ∘ P) ωᵀ` for a symmetric grid-space density `P = ωᵀDω`.
    /// The Hadamard product is the shared `(K,L)` index of Eq. 38, NOT a GEMM.
    fn contract(&self, mut p: Array2<f64>, k: &mut Array2<f64>) {
        p.zip_mut_with(&self.m, |a, &b| *a *= b);
        let tmp = self.omega.dot(&p);
        k.assign(&tmp.dot(&self.omega.t()));
    }

    fn check_nbf(&self, rows: usize, what: &str) -> Result<(), FerricError> {
        if rows != self.nbf {
            return Err(FerricError::General(format!(
                "ThcK: {what} has {rows} rows, expected nbf={}",
                self.nbf
            )));
        }
        Ok(())
    }
}

impl KBuilder for ThcK {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        self.check_nbf(d.nrows(), "density")?;
        let wd = self.omega.t().dot(&d.dot(&self.omega));
        // D symmetric => P symmetric; symmetrize away GEMM-order drift.
        let p = (&wd + &wd.t()) * 0.5;
        self.contract(p, k);
        Ok(0)
    }

    /// Eq. 38 ordering: `Φ = C_occᵀ ω`, `P = ΦᵀΦ`, then the shared contraction.
    fn build_from_occ(
        &mut self,
        c_occ: &Array2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        self.check_nbf(c_occ.nrows(), "c_occ")?;
        let phi = c_occ.t().dot(&self.omega);
        self.contract(phi.t().dot(&phi), k);
        Ok(0)
    }

    fn update_density(&mut self, _d: &Array2<f64>) {}

    fn reset(&mut self) {}
}

/// AO values on the Becke parent grid, `(nbf, npts)`.
fn parent_collocation(
    mol: &Molecule,
    obs_bs: &BasisSet,
    cfg: &ThcKConfig,
) -> Result<Array2<f64>, FerricError> {
    use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
    let gcfg = AtomicGridConfig {
        n_radial: cfg.n_radial,
        n_angular: cfg.n_angular,
        prune: None,
    };
    let points: Vec<[f64; 3]> = build_atomic_grid(mol, &gcfg)
        .iter()
        .map(|p| p.xyz)
        .collect();
    eval_basis_on_points(mol, obs_bs, &points)
        .map_err(|e| FerricError::General(format!("ThcK: AO grid eval: {e:?}")))
}

/// Greedy pivoted Cholesky of `Γ_pq = (χ_pᵀχ_q)²` with columns on demand.
/// Returns at most `want` pivots, fewer if the residual diagonal drops below
/// `rank_tol · max_p Γ_pp` (numerical rank of the product space reached).
pub fn select_points_pivoted_cholesky(chi: &Array2<f64>, want: usize, rank_tol: f64) -> Vec<usize> {
    let npts = chi.ncols();
    let kmax = want.min(npts);
    let mut resid: Array1<f64> = chi.map_axis(Axis(0), |c| c.dot(&c).powi(2));
    let stop = rank_tol * resid.iter().cloned().fold(0.0, f64::max);
    let mut l = Array2::<f64>::zeros((npts, kmax));
    let mut pivots = Vec::with_capacity(kmax);
    for j in 0..kmax {
        let (p, dp) = argmax(&resid);
        if !(dp > stop) {
            break;
        }
        let mut col = chi.t().dot(&chi.column(p)).mapv(|g| g * g);
        if j > 0 {
            let lp = l.slice(s![p, ..j]).to_owned();
            col -= &l.slice(s![.., ..j]).dot(&lp);
        }
        col /= dp.sqrt();
        resid.zip_mut_with(&col, |r, &c| *r -= c * c);
        resid[p] = 0.0;
        l.column_mut(j).assign(&col);
        pivots.push(p);
    }
    pivots
}

/// Rescale each collocation column to unit norm. THC is invariant under
/// `ω_K → a_K ω_K` (the fitted `M_KL` absorbs `1/(a_K² a_L²)`), but the
/// normal-equation metric `S` is not: near-nucleus points carry values many
/// orders larger than valence ones, and Jacobi-scaling them out is what lets
/// the trivial-limit anchor reach round-off.
fn unit_columns(mut omega: Array2<f64>) -> Array2<f64> {
    for mut col in omega.columns_mut() {
        let nrm = col.dot(&col).sqrt();
        if nrm > 0.0 {
            col /= nrm;
        }
    }
    omega
}

fn argmax(v: &Array1<f64>) -> (usize, f64) {
    v.iter().enumerate().fold(
        (0, f64::NEG_INFINITY),
        |acc, (i, &x)| if x > acc.1 { (i, x) } else { acc },
    )
}

/// `M = S⁺ WᵀW S⁺`, `W = V^{-1/2} Σ_μν (P|μν) ω_μK ω_νK`, `S = (ωᵀω)∘(ωᵀω)`.
fn fit_core(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    omega: &Array2<f64>,
    pinv_tol: f64,
) -> Result<Array2<f64>, FerricError> {
    let v_is = v_inv_sqrt_lindep(&threeindex::coulomb_metric_2c(op, dfbs)?)?;
    let eri3 = threeindex::eri3_tensor(op, obs, dfbs)?;
    let w = v_is.dot(&collocate_eri3(&eri3, omega));
    let gram = omega.t().dot(omega);
    let s_pinv = pseudo_inverse_psd(&(&gram * &gram), pinv_tol)?;
    let ws = w.dot(&s_pinv);
    Ok(ws.t().dot(&ws))
}

/// `R[P,K] = Σ_μν E[P,μ,ν] ω_μK ω_νK` without forming `(nbf², N_IP)`.
fn collocate_eri3(eri3: &ndarray::Array3<f64>, omega: &Array2<f64>) -> Array2<f64> {
    let naux = eri3.shape()[0];
    let mut r = Array2::<f64>::zeros((naux, omega.ncols()));
    for (p, mut row) in r.axis_iter_mut(Axis(0)).enumerate() {
        let t = eri3.index_axis(Axis(0), p).dot(omega);
        row.assign(&(&t * omega).sum_axis(Axis(0)));
    }
    r
}

/// Truncated pseudo-inverse of a symmetric PSD matrix.
fn pseudo_inverse_psd(s: &Array2<f64>, tol: f64) -> Result<Array2<f64>, FerricError> {
    let (evals, evecs) = eigh_dc(s, Uplo::Upper)?;
    let cut = tol * evals.iter().cloned().fold(0.0f64, f64::max).max(1e-300);
    let inv = Array1::from_iter(evals.iter().map(|&l| if l > cut { 1.0 / l } else { 0.0 }));
    Ok((&evecs * &inv).dot(&evecs.t()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df_k::DfK;
    use ferric_core::basis;

    fn water() -> Molecule {
        Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap()
    }

    struct Fixture {
        mol: Molecule,
        obs_set: BasisSet,
        obs: PreparedBasis,
        dfbs: PreparedBasis,
    }

    fn fixture(obs_name: &str) -> Fixture {
        let mol = water();
        let obs_set = basis::bundled(obs_name).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        Fixture {
            mol,
            obs_set,
            obs,
            dfbs,
        }
    }

    /// Deterministic non-orthogonal mock occupied block.
    fn mock_c_occ(n: usize, nocc: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, nocc), |(mu, i)| {
            ((mu * 7 + i * 13) % 11) as f64 / 11.0 - 0.5
        })
    }

    fn rel_max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let d = (a - b).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        d / b.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    fn dfk_reference(f: &Fixture, c: &Array2<f64>) -> Array2<f64> {
        let n = f.obs.nbasis();
        let mut dfk = DfK::new(Operator::coulomb(), &f.obs, &f.dfbs, usize::MAX / 4).unwrap();
        let mut k = Array2::zeros((n, n));
        <DfK as KBuilder>::build(&mut dfk, &c.dot(&c.t()), &mut k).unwrap();
        k
    }

    /// EXACTNESS ANCHOR. With the selection allowed to exhaust the rank of the
    /// symmetric AO-pair space (STO-3G water: 28), `X S⁺ Xᵀ` is the projector
    /// onto it and THC-K must equal DF-K on the same aux basis and the same
    /// `V^{-1/2}`, on BOTH contraction paths.
    ///
    /// The bar is the measured conditioning floor of the normal-equation fit,
    /// not machine epsilon: `S` squares the collocation condition number and
    /// `M` carries large cancelling entries. Measured 2026-10-24: 4.9e-10 at
    /// `rank_tol = 1e-18` (n_ip 37), 2.3e-9 at 1e-14 (n_ip 28). The
    /// construction mutants (Hadamard -> GEMM, one `S⁺` dropped, `V^{-1/2}`
    /// dropped, `S` not squared) fail it by >= 8 orders of magnitude; see
    /// `reference/thc/thc-lane-2026-10.md`.
    #[test]
    fn thc_k_matches_exact_in_the_trivial_limit() {
        let f = fixture("sto-3g");
        let n = f.obs.nbasis();
        let cfg = ThcKConfig {
            c_isdf: 1e6,
            ..Default::default()
        };
        let mut thc = ThcK::new(
            &f.mol,
            &f.obs_set,
            &f.obs,
            &f.dfbs,
            Operator::coulomb(),
            &cfg,
        )
        .unwrap();
        assert!(
            thc.n_interpolation_points() >= n * (n + 1) / 2,
            "must reach the pair-space rank"
        );

        let c = mock_c_occ(n, 5);
        let k_ref = dfk_reference(&f, &c);
        let mut k_den = Array2::zeros((n, n));
        thc.build(&c.dot(&c.t()), &mut k_den).unwrap();
        let mut k_occ = Array2::zeros((n, n));
        thc.build_from_occ(&c, &mut k_occ).unwrap();
        let (e_den, e_occ) = (rel_max_diff(&k_den, &k_ref), rel_max_diff(&k_occ, &k_ref));
        assert!(
            e_den < 1e-9 && e_occ < 1e-9,
            "THC vs DF-K: density {e_den:.2e}, occ {e_occ:.2e}"
        );
    }

    /// Full-rank selection at cc-pVDZ. The product space is far worse
    /// conditioned than at STO-3G: the Gram-based selection stops at n_ip 293
    /// of 300 and the fit floor is 1.63e-4 of max|K| (measured 2026-10-24) —
    /// a REGRESSION bar for the realistic-basis floor, not an exactness claim.
    #[test]
    fn thc_k_full_rank_floor_ccpvdz() {
        let f = fixture("cc-pvdz");
        let n = f.obs.nbasis();
        let cfg = ThcKConfig {
            c_isdf: 6.0,
            ..Default::default()
        };
        let mut thc = ThcK::new(
            &f.mol,
            &f.obs_set,
            &f.obs,
            &f.dfbs,
            Operator::coulomb(),
            &cfg,
        )
        .unwrap();
        let c = mock_c_occ(n, 5);
        let k_ref = dfk_reference(&f, &c);
        let mut k = Array2::zeros((n, n));
        thc.build_from_occ(&c, &mut k).unwrap();
        let e = rel_max_diff(&k, &k_ref);
        assert!(
            e < 3e-4,
            "full-rank cc-pVDZ THC vs DF-K {e:.2e}, n_ip {}",
            thc.n_interpolation_points()
        );
    }

    /// TRUNCATED regime. Water/cc-pVDZ-RI has naux 84, so the production
    /// `c_isdf = 8.5` asks for 714 points and is clamped at the 293 the
    /// product space holds — that is the full-rank floor above, not a
    /// truncation. Here 200 points (`c_isdf ≈ 2.38`) are requested, well below
    /// the rank. Bounds are derived from both sides of the measured 3.17e-3
    /// (2026-10-24): it must stay below 2× that (regression), and above 5×
    /// the 1.63e-4 full-rank floor so the test cannot silently fall back into
    /// the clamped regime.
    #[test]
    fn thc_k_truncated_rank_error_is_bounded() {
        let f = fixture("cc-pvdz");
        let n = f.obs.nbasis();
        let cfg = ThcKConfig {
            c_isdf: 199.5 / f.dfbs.nbasis() as f64,
            ..Default::default()
        };
        let mut thc = ThcK::new(
            &f.mol,
            &f.obs_set,
            &f.obs,
            &f.dfbs,
            Operator::coulomb(),
            &cfg,
        )
        .unwrap();
        assert_eq!(thc.n_interpolation_points(), 200);
        let c = mock_c_occ(n, 5);
        let mut k = Array2::zeros((n, n));
        thc.build_from_occ(&c, &mut k).unwrap();
        let e = rel_max_diff(&k, &dfk_reference(&f, &c));
        assert!(
            e < 6.4e-3 && e > 8.2e-4,
            "truncated (n_ip 200) THC vs DF-K {e:.3e}"
        );
    }

    /// Independent construction for the SELECTION: greedy column-pivoted QR
    /// (Businger-Golub, by Gram-Schmidt on the explicit `nbf² × npts` product
    /// matrix `X`, never forming the Gram) must pick the same pivots as the
    /// pivoted Cholesky of `Γ = XᵀX`. Equal in exact arithmetic; checked at
    /// k = 60 on water/cc-pVDZ.
    fn qrcp_pivots(chi: &Array2<f64>, k: usize) -> Vec<usize> {
        let (nbf, npts) = chi.dim();
        let mut x = Array2::<f64>::zeros((nbf * nbf, npts));
        for mu in 0..nbf {
            for nu in 0..nbf {
                let row = &chi.row(mu) * &chi.row(nu);
                x.row_mut(mu * nbf + nu).assign(&row);
            }
        }
        let mut norms2: Array1<f64> = x.map_axis(Axis(0), |c| c.dot(&c));
        let mut q: Vec<Array1<f64>> = Vec::with_capacity(k);
        let mut piv = Vec::with_capacity(k);
        for _ in 0..k {
            let (p, _) = argmax(&norms2);
            let mut v = x.column(p).to_owned();
            for _ in 0..2 {
                for qi in &q {
                    let d = qi.dot(&v);
                    v.scaled_add(-d, qi);
                }
            }
            v /= v.dot(&v).sqrt();
            let proj = x.t().dot(&v);
            norms2.zip_mut_with(&proj, |r, &c| *r -= c * c);
            for &done in &piv {
                norms2[done] = f64::NEG_INFINITY;
            }
            norms2[p] = f64::NEG_INFINITY;
            q.push(v);
            piv.push(p);
        }
        piv
    }

    #[test]
    fn thc_k_selection_matches_column_pivoted_qr() {
        let f = fixture("cc-pvdz");
        let chi = parent_collocation(&f.mol, &f.obs_set, &ThcKConfig::default()).unwrap();
        let chol = select_points_pivoted_cholesky(&chi, 60, ThcKConfig::default().rank_tol);
        assert_eq!(chol, qrcp_pivots(&chi, 60));
    }

    #[test]
    fn thc_k_rejects_nonpositive_c_isdf() {
        let f = fixture("sto-3g");
        let cfg = ThcKConfig {
            c_isdf: 0.0,
            ..Default::default()
        };
        assert!(ThcK::new(
            &f.mol,
            &f.obs_set,
            &f.obs,
            &f.dfbs,
            Operator::coulomb(),
            &cfg
        )
        .is_err());
    }

    #[test]
    fn thc_k_from_parts_rejects_shape_mismatch() {
        assert!(ThcK::from_parts(Array2::zeros((4, 6)), Array2::zeros((6, 5))).is_err());
    }
}
