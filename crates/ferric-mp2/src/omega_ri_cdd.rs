//! omega-RI-CDD-MP2 and omega-RI-CDD-SOS-MP2.
//!
//! Glasbrenner, Graf, Ochsenfeld, JCTC 16, 6856 (2020): Laplace-transformed
//! RI-MP2 in the basis of Cholesky-decomposed pseudo-densities (CDD, Maurer,
//! Clin, Ochsenfeld, JCP 140, 224112 (2014)), with the erfc-attenuated fitting
//! metric of [`crate::omega_ri`]. Book: `book/src/12-oo-laplace/05-omega-ri-cdd-mp2.md`.
//!
//! # Energy
//!
//! For each minimax Laplace point `(t, w)` (Fermi level `ε_F` midway in the
//! gap), the occupied and virtual pseudo-densities
//!
//! ```text
//! P_occ(t) = C_o e^{+t(ε_i-ε_F)} C_oᵀ,   P_vir(t) = C_v e^{-t(ε_a-ε_F)} C_vᵀ
//! ```
//!
//! are factored by pivoted Cholesky, `P = L Lᵀ`, and the fitted 3-index tensor
//! is carried to the Cholesky pseudo-MOs `i̲, a̅`. With `(i̲a̅|j̲b̅) = Σ_P B_P,i̲a̅ B_P,j̲b̅`,
//!
//! ```text
//! E_C = -2 Σ_t w Σ_PQ J_PQ(t)²,   J(t) = B(t) B(t)ᵀ               (paper eqs 34-37)
//! E_X = +Σ_t w Σ_ij Σ_ab (i̲a̅|j̲b̅)(i̲b̅|j̲a̅)                         (paper eqs 38-39)
//! E_MP2 = E_C + E_X,   E_OS = E_C/2,   E_SS = E_C/2 + E_X,   E_SOS = c_os E_OS
//! ```
//!
//! The energy is invariant under any orthogonal rotation of the occupied and
//! virtual pseudo-MOs, so the UNTRUNCATED Cholesky factor reproduces the MO
//! Laplace energy exactly ([`PseudoOrbitals::Canonical`] is that reference).
//!
//! # What is and is not implemented
//!
//! The algebra here is DENSE. The paper's cost advantage comes from natural
//! blocking and aIPB screening of the sparse erfc 3-index integrals in the AO
//! and pseudo-MO bases; none of that is built. The Cholesky factor is applied
//! to the dressed MO tensor (`L = C X` with `X` in the MO basis, the same
//! transformation the paper's eqs 22-27 perform through `T = Lᵀ S L`), so the
//! AO 3-index tensor is read once, not once per Laplace point. The
//! approximations that DO change the cost here are the Cholesky truncation
//! (`cdd_tol`: fewer virtual pseudo-MOs at large `t`) and the paper's Schwarz
//! screening of the exchange term (`schwarz_thresh` = ϑ_schwarz, eq 43, and
//! `pair_thresh` = ϑ_ij, eq 46). Each is exact at 0, which is its default.
//!
//! # Pseudo-MO construction and order
//!
//! The pivoted Cholesky runs on the Löwdin-orthogonalized pseudo-density
//! `S^{1/2} P S^{1/2}` (the paper's construction, eqs 22-27), and the factor is
//! carried back with `S^{-1/2}`. In the factored form this is a pivoted
//! Cholesky of `G = S^{1/2} C e^{±t(ε-ε_F)/2}` ([`lowdin_coefficients`]), which
//! gives pseudo-MOs centred on their (orthogonalized) AO pivot. They are then
//! sorted by the position of that pivot along the molecule's principal axis
//! ([`principal_axis_ao_keys`]), so pseudo-MOs that are near in space are near
//! in index and the Schwarz-kept virtuals of a pair form a short index range.
//! The exchange GEMM runs on that contiguous range as a strided view of `B`
//! ([`screened_exchange`]). Neither choice moves the untruncated energy: any
//! full-rank pivoted Cholesky factor is an orthogonal rotation of the scaled
//! MOs, and a permutation is one too. They change which columns a nonzero
//! `cdd_tol` drops and which entries the Schwarz screens drop.

use crate::omega_ri::RiFit;
use crate::rimp2::{active_occ, eri3_budget_bytes, stream_dressed_mo_band_budgeted};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_quadrature::LaplaceQuadrature;
use ferric_scf::PreparedAux;
use ferric_scf::ScfResult;
use ndarray::{s, Array2, Axis};
use rayon::prelude::*;

/// How the occupied/virtual pseudo-MOs are built at each Laplace point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PseudoOrbitals {
    /// Pivoted Cholesky of the pseudo-densities (CDD). `tol` is the absolute
    /// residual-diagonal stopping threshold; `0.0` stops only at the rank of
    /// the space (machine precision), which is exact.
    Cholesky {
        /// Residual-diagonal threshold of the pivoted Cholesky decomposition.
        tol: f64,
    },
    /// Canonical MOs scaled by `e^{±t(ε-ε_F)/2}`: the MO-Laplace reference the
    /// untruncated Cholesky factor must reproduce.
    Canonical,
}

/// Which energy to evaluate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CddEnergy {
    /// Full MP2: Coulomb and exchange terms.
    Mp2,
    /// SOS-MP2: `c_os · E_OS`, the Coulomb term only (no exchange).
    Sos {
        /// Opposite-spin scaling coefficient (Jung/Head-Gordon: 1.3).
        c_os: f64,
    },
}

/// Configuration of [`omega_ri_cdd_mp2`].
#[derive(Debug, Clone)]
pub struct OmegaRiCddConfig {
    /// Fitting metric; `RiFit::Coulomb` gives RI-CDD-MP2.
    pub fit: RiFit,
    /// Number of frozen core orbitals.
    pub frozen_core: usize,
    /// Minimax Laplace points, one of {3, 5, 7} (hard error otherwise).
    pub n_quad: usize,
    /// Pseudo-MO construction.
    pub orbitals: PseudoOrbitals,
    /// MP2 or SOS-MP2.
    pub energy: CddEnergy,
    /// ϑ_schwarz: drop virtual `a` from pair `ij` when `Q_ia Q_ja · max_b Q_ib Q_jb`
    /// is below it (every dropped product is then bounded by it). `0` = off.
    pub schwarz_thresh: f64,
    /// ϑ_ij: drop pair `ij` when the bound `(Σ_a Q_ia Q_ja)²` on its exchange
    /// energy is below it. `0` = off.
    pub pair_thresh: f64,
    /// Resident-bytes ceiling; `None` resolves as everywhere else.
    pub memory_budget_bytes: Option<usize>,
}

impl Default for OmegaRiCddConfig {
    fn default() -> Self {
        Self {
            fit: RiFit::Coulomb,
            frozen_core: 0,
            n_quad: 7,
            orbitals: PseudoOrbitals::Cholesky { tol: 0.0 },
            energy: CddEnergy::Mp2,
            schwarz_thresh: 0.0,
            pair_thresh: 0.0,
            memory_budget_bytes: None,
        }
    }
}

/// Result of [`omega_ri_cdd_mp2`].
#[derive(Debug, Clone)]
#[must_use]
pub struct OmegaRiCddResult {
    /// `E_SCF + e_corr`.
    pub total_energy: f64,
    /// MP2 (or scaled SOS) correlation energy.
    pub e_corr: f64,
    /// Coulomb term `E_C` (= 2 E_OS).
    pub e_coulomb: f64,
    /// Exchange term `E_X`; `None` for SOS (not computed).
    pub e_exchange: Option<f64>,
    /// Unscaled opposite-spin energy `E_C / 2`.
    pub e_os: f64,
    /// Same-spin energy `E_C/2 + E_X`; `None` for SOS.
    pub e_ss: Option<f64>,
    /// Laplace points used.
    pub n_quad: usize,
    /// Fit provenance label.
    pub fit: String,
    /// Mean occupied / virtual pseudo-MO counts over the Laplace points.
    pub mean_rank_occ: f64,
    /// See [`Self::mean_rank_occ`].
    pub mean_rank_vir: f64,
    /// Active occupied / virtual canonical counts (the untruncated ranks).
    pub nocc: usize,
    /// See [`Self::nocc`].
    pub nvir: usize,
    /// Fraction of `(i<=j)` pairs kept by the ϑ_ij prescreen (1 for SOS).
    pub pair_fraction: f64,
    /// Fraction of `(ij, a)` virtual entries kept by ϑ_schwarz (1 for SOS).
    pub virtual_fraction: f64,
    /// Exchange GEMM work of the kept entries, `Σ_ij k_ij² / (n_pairs · r_vir²)`
    /// summed over points (1 unscreened, 0 for SOS): the flop fraction a
    /// gather of exactly the kept virtuals would do.
    pub exchange_kept_work: f64,
    /// Exchange GEMM work actually done on the contiguous index ranges,
    /// `Σ_ij w_ij² / (n_pairs · r_vir²)` (`w_ij >= k_ij` is the range width).
    pub exchange_range_work: f64,
    /// Wall seconds per stage.
    pub timings: CddTimings,
}

/// Stage wall times of one [`omega_ri_cdd_mp2`] call, seconds; the per-point
/// stages are summed over the Laplace points.
#[derive(Debug, Clone, Copy, Default)]
pub struct CddTimings {
    /// Fit dressing `D` (metrics, Cholesky, solves).
    pub dress: f64,
    /// 3-index integrals and the dressed MO transform to `B_ov`.
    pub b_ov: f64,
    /// Pivoted Cholesky pseudo-MO factors and their ordering.
    pub cholesky: f64,
    /// `B_ov -> B(t)` pseudo-MO transform.
    pub transform: f64,
    /// Coulomb term `tr J²`.
    pub coulomb: f64,
    /// Exchange term with its Schwarz screens.
    pub exchange: f64,
}

impl std::fmt::Display for OmegaRiCddResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "omega-RI-CDD total: {:.10} Ha (corr {:.10}; fit {}, n_quad {}, \
             pseudo-MO ranks {:.1}/{} occ {:.1}/{} vir, pairs {:.3}, virtuals {:.3})",
            self.total_energy,
            self.e_corr,
            self.fit,
            self.n_quad,
            self.mean_rank_occ,
            self.nocc,
            self.mean_rank_vir,
            self.nvir,
            self.pair_fraction,
            self.virtual_fraction
        )
    }
}

/// Pivoted Cholesky of `P = G Gᵀ` (`G` is `n x m`), returned in the coordinates
/// of `G`: `M` (`m x r`) with `L = G M`, `L Lᵀ ≈ P`, plus the AO pivots.
///
/// Never forms the `n x n` matrix `P`: column `k` of the factor is
/// `G (I - M Mᵀ) g_p / sqrt(d_p)` for pivot `p` (largest residual diagonal
/// `d_p`, `g_p` = row `p` of `G`). Stops when `max d <= max(tol, m·ε·d_0)` or
/// at `r = m`, so `tol = 0` stops only at roundoff and the factor spans the
/// same space as `G`: it is then an exact factorization up to rounding.
pub fn pivoted_cholesky_factored(g: &Array2<f64>, tol: f64) -> (Vec<usize>, Array2<f64>) {
    let (n, m) = g.dim();
    let mut d: Vec<f64> = g.rows().into_iter().map(|r| r.dot(&r)).collect();
    let d0 = d.iter().cloned().fold(0.0f64, f64::max);
    let floor = tol.max(m as f64 * f64::EPSILON * d0);
    let mut cols: Vec<ndarray::Array1<f64>> = Vec::with_capacity(m);
    let mut pivots = Vec::with_capacity(m);
    for _ in 0..m {
        let (p, &dp) =
            d.iter().enumerate().fold(
                (0, &f64::NEG_INFINITY),
                |b, x| if x.1 > b.1 { x } else { b },
            );
        if !(dp > floor) {
            break;
        }
        let gp = g.row(p).to_owned();
        // (I - M Mᵀ) g_p
        let mut mk = gp.clone();
        for c in &cols {
            let proj = c.dot(&gp);
            mk.scaled_add(-proj, c);
        }
        mk /= dp.sqrt();
        let l = g.dot(&mk);
        for i in 0..n {
            d[i] -= l[i] * l[i];
        }
        d[p] = 0.0;
        pivots.push(p);
        cols.push(mk);
    }
    let r = cols.len();
    let mut mmat = Array2::<f64>::zeros((m, r));
    for (k, c) in cols.iter().enumerate() {
        mmat.column_mut(k).assign(c);
    }
    (pivots, mmat)
}

/// Pseudo-MO coefficients at one Laplace / imaginary-time point, in the
/// canonical MO basis: `L_occ = C_o x_occ`, `L_vir = C_v x_vir`.
#[derive(Debug, Clone)]
pub struct CddFactors {
    /// `(nocc, r_occ)`.
    pub x_occ: Array2<f64>,
    /// `(nvir, r_vir)`.
    pub x_vir: Array2<f64>,
}

/// Löwdin MO coefficients `S^{1/2} C` (orthonormal columns: `Cᵀ S C = I`).
///
/// The pivoted Cholesky of `S^{1/2} P S^{1/2} = (S^{1/2} G)(S^{1/2} G)ᵀ` is the
/// paper's orthogonalized CDD. In the factored form only the rows that are
/// pivoted on change: [`pivoted_cholesky_factored`] of `S^{1/2} G` returns `M`
/// with `L' = S^{1/2} G M`, and the AO factor is `L = S^{-1/2} L' = G M`, the
/// same coordinates as before.
pub fn lowdin_coefficients(s_half: &Array2<f64>, c: &Array2<f64>) -> Array2<f64> {
    s_half.dot(c)
}

/// Position of each AO's centre along the principal axis of the atomic
/// coordinates (the eigenvector of the largest eigenvalue of their covariance;
/// for a chain, the chain direction). Sorting pseudo-MOs by the key of their
/// AO pivot lays them out along the molecule. Any key is exact: it only
/// permutes pseudo-MOs.
pub fn principal_axis_ao_keys(mol: &Molecule, obs: &PreparedBasis) -> Vec<f64> {
    let n = mol.atoms.len().max(1) as f64;
    let xyz: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();
    let mut c = [0.0; 3];
    for p in &xyz {
        for k in 0..3 {
            c[k] += p[k] / n;
        }
    }
    let mut cov = Array2::<f64>::zeros((3, 3));
    for p in &xyz {
        for a in 0..3 {
            for b in 0..3 {
                cov[(a, b)] += (p[a] - c[a]) * (p[b] - c[b]);
            }
        }
    }
    let axis = match ferric_core::linalg::eigh_dc(&cov, ferric_core::linalg::Uplo::Upper) {
        Ok((_, v)) => [v[(0, 2)], v[(1, 2)], v[(2, 2)]],
        Err(_) => [0.0, 0.0, 1.0],
    };
    let centres = obs.shell_centers();
    let mut keys = vec![0.0; obs.nbasis()];
    for ((&off, &dim), r) in obs
        .shell_offsets()
        .iter()
        .zip(obs.shell_dims())
        .zip(&centres)
    {
        let k = (0..3).map(|q| (r[q] - c[q]) * axis[q]).sum::<f64>();
        keys[off..off + dim].fill(k);
    }
    keys
}

/// Build the pseudo-MO coefficients of `P_occ(t) = C_o e^{t(ε_i-ε_F)} C_oᵀ` and
/// `P_vir(t) = C_v e^{-t(ε_a-ε_F)} C_vᵀ`.
///
/// `c_occ`, `c_vir` are the Löwdin coefficients ([`lowdin_coefficients`]):
/// the Cholesky pivots on the orthogonalized pseudo-density. The pseudo-MOs
/// are returned sorted by `ao_key` of their pivot (stable, so ties keep the
/// pivot order); `ao_key` has one entry per AO row of `c_occ`.
#[allow(clippy::too_many_arguments)]
pub fn cdd_factors(
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    eps_f: f64,
    t: f64,
    orbitals: PseudoOrbitals,
    ao_key: &[f64],
) -> CddFactors {
    let so: Vec<f64> = eps_occ
        .iter()
        .map(|&e| (0.5 * t * (e - eps_f)).exp())
        .collect();
    let sv: Vec<f64> = eps_vir
        .iter()
        .map(|&e| (-0.5 * t * (e - eps_f)).exp())
        .collect();
    let side = |c: &Array2<f64>, sc: &[f64]| -> Array2<f64> {
        match orbitals {
            PseudoOrbitals::Canonical => Array2::from_diag(&ndarray::Array1::from(sc.to_vec())),
            PseudoOrbitals::Cholesky { tol } => {
                let mut g = c.clone();
                for (mut col, &f) in g.axis_iter_mut(Axis(1)).zip(sc) {
                    col *= f;
                }
                let (piv, m) = pivoted_cholesky_factored(&g, tol);
                let mut order: Vec<usize> = (0..piv.len()).collect();
                order.sort_by(|&x, &y| ao_key[piv[x]].total_cmp(&ao_key[piv[y]]));
                let mut m = m.select(Axis(1), &order);
                for (mut row, &f) in m.axis_iter_mut(Axis(0)).zip(sc) {
                    row *= f;
                }
                m
            }
        }
    };
    CddFactors {
        x_occ: side(c_occ, &so),
        x_vir: side(c_vir, &sv),
    }
}

/// `B_cdd[P, i̲a̅] = Σ_ia x_occ[i,i̲] x_vir[a,a̅] B[P, ia]` for a `(naux, nocc·nvir)`
/// tensor; returns `(naux, r_occ·r_vir)` in the same `i̲·r_vir + a̅` layout.
pub fn cdd_transform(b_ov: &Array2<f64>, f: &CddFactors) -> Array2<f64> {
    let naux = b_ov.nrows();
    let (nocc, ro) = f.x_occ.dim();
    let (nvir, rv) = f.x_vir.dim();
    assert_eq!(b_ov.ncols(), nocc * nvir, "cdd_transform: B width");
    let mut out = Array2::<f64>::zeros((naux, ro * rv));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(b_ov.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut orow, brow)| {
            let bp = brow
                .into_shape_with_order((nocc, nvir))
                .expect("B row is contiguous");
            let y = f.x_occ.t().dot(&bp).dot(&f.x_vir);
            orow.assign(
                &y.into_shape_with_order(ro * rv)
                    .expect("fresh dot output is contiguous"),
            );
        });
    out
}

/// Tally of one [`screened_exchange`] call.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExchangeTally {
    /// `Σ_ij Σ_ab (ia|jb)(ib|ja)` over the kept entries.
    pub k: f64,
    /// `(i<=j)` pairs kept by ϑ_ij / all pairs.
    pub pairs_kept: usize,
    /// See [`Self::pairs_kept`].
    pub pairs_total: usize,
    /// `(ij, a)` entries kept by ϑ_schwarz / `pairs_total · nv`.
    pub virt_kept: usize,
    /// See [`Self::virt_kept`].
    pub virt_total: usize,
    /// `Σ_ij k_ij²` over kept pairs (`k_ij` kept virtuals).
    pub kept_sq: f64,
    /// `Σ_ij w_ij²` over kept pairs (`w_ij` the kept index range width).
    pub range_sq: f64,
}

/// Exchange sum `Σ_ij Σ_ab (ia|jb)(ib|ja)` with the paper's Schwarz screening,
/// `(ia|jb) = Σ_P B_P,ia B_P,jb`, `b` of shape `(naux, no·nv)`.
///
/// For each kept pair the GEMM `G = B_iᵀ B_j` runs on the contiguous virtual
/// index range `[min keep, max keep]`, read as a strided view of `b` (no
/// copy), and the sum is restricted to the kept entries inside it. The result
/// is the sum over exactly the kept set whatever the order of the virtuals;
/// the order only decides how much of the range is wasted (pseudo-MOs sorted
/// along the molecule make the range close to the kept set). With both
/// thresholds 0 nothing is dropped and the sum is exact.
pub fn screened_exchange(
    b: &Array2<f64>,
    no: usize,
    nv: usize,
    schwarz_thresh: f64,
    pair_thresh: f64,
) -> ExchangeTally {
    // Q_ia = sqrt((ia|ia)) = |B_:,ia|
    let mut q = vec![0.0f64; no * nv];
    for row in b.rows() {
        for (qc, &x) in q.iter_mut().zip(row) {
            *qc += x * x;
        }
    }
    q.iter_mut().for_each(|x| *x = x.sqrt());
    let per_i: Vec<ExchangeTally> = (0..no)
        .into_par_iter()
        .map(|i| {
            let mut acc = ExchangeTally::default();
            let qi = &q[i * nv..(i + 1) * nv];
            let mut mask = vec![false; nv];
            for j in i..no {
                acc.pairs_total += 1;
                acc.virt_total += nv;
                let qj = &q[j * nv..(j + 1) * nv];
                let sa: Vec<f64> = qi.iter().zip(qj).map(|(x, y)| x * y).collect();
                let a_ij: f64 = sa.iter().sum();
                if a_ij * a_ij < pair_thresh {
                    continue;
                }
                acc.pairs_kept += 1;
                let smax = sa.iter().cloned().fold(0.0f64, f64::max);
                let (mut lo, mut hi, mut k) = (nv, 0usize, 0usize);
                for (a, m) in mask.iter_mut().enumerate() {
                    *m = sa[a] * smax >= schwarz_thresh;
                    if *m {
                        lo = lo.min(a);
                        hi = a + 1;
                        k += 1;
                    }
                }
                acc.virt_kept += k;
                if k == 0 {
                    continue;
                }
                let w = hi - lo;
                acc.kept_sq += (k * k) as f64;
                acc.range_sq += (w * w) as f64;
                let bi = b.slice(s![.., i * nv + lo..i * nv + hi]);
                let bj = b.slice(s![.., j * nv + lo..j * nv + hi]);
                // G[a,b] = (ia|jb) on the range; K_ij = Σ_{a,b kept} G[a,b] G[b,a]
                let g = bi.t().dot(&bj);
                let m = &mask[lo..hi];
                let mut kij = 0.0;
                for a in 0..w {
                    if !m[a] {
                        continue;
                    }
                    for bb in 0..w {
                        if m[bb] {
                            kij += g[(a, bb)] * g[(bb, a)];
                        }
                    }
                }
                acc.k += if i == j { kij } else { 2.0 * kij };
            }
            acc
        })
        .collect();
    per_i
        .iter()
        .fold(ExchangeTally::default(), |a, x| ExchangeTally {
            k: a.k + x.k,
            pairs_kept: a.pairs_kept + x.pairs_kept,
            pairs_total: a.pairs_total + x.pairs_total,
            virt_kept: a.virt_kept + x.virt_kept,
            virt_total: a.virt_total + x.virt_total,
            kept_sq: a.kept_sq + x.kept_sq,
            range_sq: a.range_sq + x.range_sq,
        })
}

/// [`omega_ri_cdd_mp2`] on the system `aux` was prepared on (see [`PreparedAux`]):
/// `rhf` is checked against that system first
/// ([`PreparedAux::check_reference`]).
/// The reference must be a closed-shell restricted (RHF / RKS) result
/// ([`PreparedAux::check_restricted_reference`]).
/// A system with no virtual orbitals is refused
/// ([`PreparedAux::check_virtual_orbitals`]).
pub fn omega_ri_cdd_mp2_on(
    aux: &PreparedAux<'_>,
    rhf: &ScfResult,
    config: &OmegaRiCddConfig,
) -> Result<OmegaRiCddResult, FerricError> {
    let sys = aux.system();
    aux.check_restricted_reference(rhf)?;
    aux.check_virtual_orbitals("omega_ri_cdd_mp2_on", rhf)?;
    omega_ri_cdd_mp2(sys.mol(), sys.basis(), aux.basis(), sys.op(), rhf, config)
}

/// omega-RI-CDD-MP2 / -SOS-MP2 correlation energy (closed-shell RHF reference).
///
/// `op` must be the Coulomb operator. See the module docs for the energy, the
/// exactness limits and what is not implemented.
pub fn omega_ri_cdd_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &OmegaRiCddConfig,
) -> Result<OmegaRiCddResult, FerricError> {
    if let PseudoOrbitals::Cholesky { tol } = config.orbitals {
        if !(tol.is_finite() && tol >= 0.0) {
            return Err(FerricError::General(format!(
                "omega-RI-CDD: cdd_tol must be finite and >= 0 (got {tol})"
            )));
        }
    }
    for (name, v) in [
        ("schwarz_thresh", config.schwarz_thresh),
        ("pair_thresh", config.pair_thresh),
    ] {
        if !(v.is_finite() && v >= 0.0) {
            return Err(FerricError::General(format!(
                "omega-RI-CDD: {name} must be finite and >= 0 (got {v})"
            )));
        }
    }
    if let CddEnergy::Sos { c_os } = config.energy {
        if !c_os.is_finite() {
            return Err(FerricError::General(format!(
                "omega-RI-CDD-SOS: c_os must be finite (got {c_os})"
            )));
        }
        if config.schwarz_thresh > 0.0 || config.pair_thresh > 0.0 {
            return Err(FerricError::General(
                "omega-RI-CDD-SOS has no exchange term: schwarz_thresh / pair_thresh \
                 would be ignored, so they are refused"
                    .into(),
            ));
        }
    }
    let nbas = obs.nbasis();
    let nocc_total = mol.nelec() as usize / 2;
    let fc = config.frozen_core;
    let nocc = active_occ(nocc_total, fc)?;
    let nvir = nbas - nocc_total;
    let naux = dfbs.nbasis();
    let eps = rhf.eps_r();
    let nmo = eps.len();
    let c = rhf.mos_r();
    let c_occ = c.slice(s![.., fc..nocc_total]).to_owned();
    let c_vir = c.slice(s![.., nocc_total..]).to_owned();
    let eps_occ: Vec<f64> = eps[fc..nocc_total].to_vec();
    let eps_vir: Vec<f64> = eps[nocc_total..].to_vec();

    // Memory preflight: B_ov and one transformed B(t), both (naux, nocc*nvir)
    // at most (the exchange reads strided views of B(t), no copy).
    let budget = ferric_core::memory::resolve_budget_bytes(config.memory_budget_bytes);
    let nov_bytes = naux
        .saturating_mul(nocc)
        .saturating_mul(nvir)
        .saturating_mul(8);
    let planes = 2;
    ferric_core::memory::check_alloc(
        &format!(
            "omega-RI-CDD (naux={naux}, nocc={nocc}, nvir={nvir}; {planes} (naux, nocc*nvir) planes)"
        ),
        nov_bytes.saturating_mul(planes),
        budget,
    )?;

    // Fitted MO tensor, built once: B = D (P|ia)_w.
    let mut tm = CddTimings::default();
    let clock = std::time::Instant::now;
    let eri_budget = eri3_budget_bytes(config.memory_budget_bytes);
    let t0 = clock();
    let (src_op, dress) = crate::omega_ri::fit_dressing(&config.fit, op, dfbs, budget)?;
    tm.dress = t0.elapsed().as_secs_f64();
    let t0 = clock();
    let b_ov = {
        let mut src = ThreeIndexSource::build(src_op, obs, dfbs, eri_budget)?;
        stream_dressed_mo_band_budgeted(&mut src, &dress, &c_occ, &c_vir, None, Some(eri_budget))?
    };
    drop(dress);
    tm.b_ov = t0.elapsed().as_secs_f64();

    // Löwdin coefficients for the orthogonalized pivoted Cholesky, and the
    // principal-axis AO keys that order the pseudo-MOs along the molecule.
    let t0 = clock();
    let s_half = ferric_core::linalg::symmetric_sqrt(&ferric_integrals::oneelectron::overlap(obs))?;
    let lc_occ = lowdin_coefficients(&s_half, &c_occ);
    let lc_vir = lowdin_coefficients(&s_half, &c_vir);
    drop(s_half);
    let ao_key = principal_axis_ao_keys(mol, obs);
    tm.cholesky = t0.elapsed().as_secs_f64();

    // Laplace grid: the same range LaplaceMp2::compute_mo fits.
    let ymin = 2.0 * (eps[nocc_total] - eps[nocc_total - 1]);
    let ymax = 2.0 * (eps[nmo - 1] - eps[0]);
    let quad = LaplaceQuadrature::new(config.n_quad, ymin, ymax)?;
    let eps_f = 0.5 * (eps[nocc_total] + eps[nocc_total - 1]);

    let mut e_c = 0.0;
    let mut e_x = 0.0;
    let (mut ro_sum, mut rv_sum) = (0usize, 0usize);
    let (mut pk, mut pt, mut vk, mut vt) = (0usize, 0usize, 0usize, 0usize);
    let (mut kept_work, mut range_work, mut full_work) = (0.0f64, 0.0f64, 0.0f64);
    for (&t, &w) in quad.points.iter().zip(&quad.weights) {
        let t0 = clock();
        let f = cdd_factors(
            &lc_occ,
            &lc_vir,
            &eps_occ,
            &eps_vir,
            eps_f,
            t,
            config.orbitals,
            &ao_key,
        );
        tm.cholesky += t0.elapsed().as_secs_f64();
        let (ro, rv) = (f.x_occ.ncols(), f.x_vir.ncols());
        ro_sum += ro;
        rv_sum += rv;
        let t0 = clock();
        let bt = cdd_transform(&b_ov, &f);
        tm.transform += t0.elapsed().as_secs_f64();
        let t0 = clock();
        e_c += -2.0 * w * crate::laplace::sos_j_square_trace(&bt);
        tm.coulomb += t0.elapsed().as_secs_f64();
        if matches!(config.energy, CddEnergy::Mp2) {
            let t0 = clock();
            let x = screened_exchange(&bt, ro, rv, config.schwarz_thresh, config.pair_thresh);
            tm.exchange += t0.elapsed().as_secs_f64();
            e_x += w * x.k;
            pk += x.pairs_kept;
            pt += x.pairs_total;
            vk += x.virt_kept;
            vt += x.virt_total;
            kept_work += x.kept_sq;
            range_work += x.range_sq;
            full_work += (x.pairs_total * rv * rv) as f64;
        }
    }
    let nq = quad.points.len() as f64;
    let e_os = 0.5 * e_c;
    let (e_corr, e_exchange, e_ss) = match config.energy {
        CddEnergy::Mp2 => (e_c + e_x, Some(e_x), Some(0.5 * e_c + e_x)),
        CddEnergy::Sos { c_os } => (c_os * e_os, None, None),
    };
    let frac = |a: usize, b: usize| if b == 0 { 1.0 } else { a as f64 / b as f64 };
    Ok(OmegaRiCddResult {
        total_energy: rhf.energy + e_corr,
        e_corr,
        e_coulomb: e_c,
        e_exchange,
        e_os,
        e_ss,
        n_quad: quad.points.len(),
        fit: config.fit.label(),
        mean_rank_occ: ro_sum as f64 / nq,
        mean_rank_vir: rv_sum as f64 / nq,
        nocc,
        nvir,
        pair_fraction: frac(pk, pt),
        virtual_fraction: frac(vk, vt),
        exchange_kept_work: if full_work > 0.0 {
            kept_work / full_work
        } else {
            0.0
        },
        exchange_range_work: if full_work > 0.0 {
            range_work / full_work
        } else {
            0.0
        },
        timings: tm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_mat(n: usize, m: usize, seed: u64) -> Array2<f64> {
        let mut s = seed;
        Array2::from_shape_fn((n, m), |_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        })
    }

    /// The factored pivoted Cholesky reproduces `G Gᵀ` at tol = 0 and its
    /// pivots are those of a textbook pivoted Cholesky on the explicit matrix.
    #[test]
    fn factored_pivoted_cholesky_is_exact_and_matches_the_textbook_pivots() {
        let g = rand_mat(9, 4, 3);
        let p = g.dot(&g.t());
        let (piv, m) = pivoted_cholesky_factored(&g, 0.0);
        assert_eq!(m.ncols(), 4);
        let l = g.dot(&m);
        let err = (&l.dot(&l.t()) - &p)
            .iter()
            .fold(0.0f64, |a, x| a.max(x.abs()));
        assert!(err < 1e-13, "L Lᵀ != P: {err:e}");
        // textbook pivoted Cholesky
        let mut r = p.clone();
        let mut want = vec![];
        for _ in 0..4 {
            let (k, _) = (0..9).fold((0, f64::MIN), |b, i| {
                if r[(i, i)] > b.1 {
                    (i, r[(i, i)])
                } else {
                    b
                }
            });
            want.push(k);
            let col = r.column(k).to_owned() / r[(k, k)].sqrt();
            for i in 0..9 {
                for j in 0..9 {
                    r[(i, j)] -= col[i] * col[j];
                }
            }
        }
        assert_eq!(piv, want);
        // a threshold drops columns and the error is bounded by it
        let (_, mt) = pivoted_cholesky_factored(&g, 0.3);
        assert!(mt.ncols() < 4);
    }

    /// Screened exchange at zero thresholds equals the Laplace module's
    /// blocked wide-GEMM exchange (an independent construction).
    #[test]
    fn unscreened_pair_exchange_matches_the_blocked_gemm_exchange() {
        let (naux, no, nv) = (11, 4, 5);
        let b = rand_mat(naux, no * nv, 9);
        let x = screened_exchange(&b, no, nv, 0.0, 0.0);
        let want = crate::laplace::laplace_exchange_energy(&b, naux, no, nv, 1 << 20);
        assert!(
            (x.k - want).abs() < 1e-12 * want.abs().max(1.0),
            "{} vs {want}",
            x.k
        );
        assert_eq!((x.pairs_kept, x.virt_kept), (x.pairs_total, x.virt_total));
        assert_eq!(x.kept_sq, x.range_sq);
        // a positive pair threshold drops something
        let x2 = screened_exchange(&b, no, nv, 0.0, 1e6);
        assert!(x2.pairs_kept < x.pairs_total);
    }

    /// Brute-force screened exchange from the explicit 4-index integrals with
    /// the paper's keep rule (the oracle for the contiguous-range GEMM).
    fn screened_exchange_oracle(b: &Array2<f64>, no: usize, nv: usize, ts: f64, tp: f64) -> f64 {
        let g = b.t().dot(b); // (ia|jb), ia = i*nv + a
        let q: Vec<f64> = (0..no * nv).map(|c| g[(c, c)].sqrt()).collect();
        let mut e = 0.0;
        for i in 0..no {
            for j in i..no {
                let sa: Vec<f64> = (0..nv).map(|a| q[i * nv + a] * q[j * nv + a]).collect();
                if sa.iter().sum::<f64>().powi(2) < tp {
                    continue;
                }
                let smax = sa.iter().cloned().fold(0.0f64, f64::max);
                let keep: Vec<usize> = (0..nv).filter(|&a| sa[a] * smax >= ts).collect();
                let mut kij = 0.0;
                for &a in &keep {
                    for &bb in &keep {
                        kij += g[(i * nv + a, j * nv + bb)] * g[(i * nv + bb, j * nv + a)];
                    }
                }
                e += if i == j { kij } else { 2.0 * kij };
            }
        }
        e
    }

    /// The contiguous-range exchange sums exactly the kept set: it equals the
    /// brute-force oracle at thresholds that drop scattered entries, and it is
    /// unchanged when the virtuals and occupieds are permuted.
    #[test]
    fn range_exchange_equals_the_kept_set_oracle_under_any_order() {
        let (naux, no, nv) = (13, 5, 9);
        let mut b = rand_mat(naux, no * nv, 21);
        // spread the column norms over decades so the screens bite unevenly
        for c in 0..no * nv {
            let f = 10f64.powf(-3.0 * ((c * 7) % (no * nv)) as f64 / (no * nv) as f64);
            b.column_mut(c).mapv_inplace(|x| x * f);
        }
        let (ts, tp) = (1e-4, 1e-7);
        let x = screened_exchange(&b, no, nv, ts, tp);
        let want = screened_exchange_oracle(&b, no, nv, ts, tp);
        assert!(
            x.virt_kept < x.virt_total && x.range_sq > x.kept_sq,
            "screen did not bite"
        );
        assert!((x.k - want).abs() < 1e-13 * want.abs(), "{} vs {want}", x.k);
        // permute occupieds and virtuals (one permutation for every occupied)
        let po: Vec<usize> = vec![3, 0, 4, 2, 1];
        let pv: Vec<usize> = vec![5, 2, 8, 0, 7, 1, 6, 3, 4];
        let cols: Vec<usize> = po
            .iter()
            .flat_map(|&i| pv.iter().map(move |&a| i * nv + a))
            .collect();
        let bp = b.select(Axis(1), &cols);
        let xp = screened_exchange(&bp, no, nv, ts, tp);
        assert!(
            (xp.k - x.k).abs() < 1e-13 * x.k.abs(),
            "{} vs {}",
            xp.k,
            x.k
        );
        assert_eq!((xp.pairs_kept, xp.virt_kept), (x.pairs_kept, x.virt_kept));
    }

    /// The orthogonalized pivoted Cholesky: pivoting the factored form on
    /// `S^{1/2} G` gives the pivots of a textbook pivoted Cholesky of the
    /// explicit `S^{1/2} P S^{1/2}`, which differ from the AO pivots for this
    /// `S`; `G M` reproduces `P`; the pseudo-MOs come out sorted by key.
    #[test]
    fn lowdin_pivots_match_the_explicit_orthogonalized_density() {
        let (n, m) = (8, 4);
        let a = rand_mat(n, n, 4);
        let s_ao = a.dot(&a.t()) * 0.3 + Array2::<f64>::eye(n);
        let s_half = ferric_core::linalg::symmetric_sqrt(&s_ao).unwrap();
        let g = rand_mat(n, m, 17);
        let gl = lowdin_coefficients(&s_half, &g);
        let (piv, mm) = pivoted_cholesky_factored(&gl, 0.0);
        let textbook = |p: &Array2<f64>| -> Vec<usize> {
            let mut r = p.clone();
            let mut want = vec![];
            for _ in 0..m {
                let (k, _) = (0..n).fold((0, f64::MIN), |b, i| {
                    if r[(i, i)] > b.1 {
                        (i, r[(i, i)])
                    } else {
                        b
                    }
                });
                want.push(k);
                let col = r.column(k).to_owned() / r[(k, k)].sqrt();
                for i in 0..n {
                    for j in 0..n {
                        r[(i, j)] -= col[i] * col[j];
                    }
                }
            }
            want
        };
        let p_ao = g.dot(&g.t());
        let p_orth = s_half.dot(&p_ao).dot(&s_half);
        assert_eq!(piv, textbook(&p_orth));
        assert_ne!(piv, textbook(&p_ao), "S does not move the pivots: vacuous");
        let l = g.dot(&mm);
        let err = (&l.dot(&l.t()) - &p_ao)
            .iter()
            .fold(0.0f64, |a, x| a.max(x.abs()));
        assert!(err < 1e-12, "G M (G M)^T != P: {err:e}");
        // cdd_factors sorts the pseudo-MOs by the key of their pivot
        let key: Vec<f64> = (0..n).map(|i| ((i * 5) % n) as f64).collect();
        let eps_o = vec![-1.0, -0.7, -0.5, -0.2];
        let f = cdd_factors(
            &gl,
            &gl,
            &eps_o,
            &eps_o,
            0.0,
            0.0,
            PseudoOrbitals::Cholesky { tol: 0.0 },
            &key,
        );
        let sorted: Vec<f64> = {
            let mut k: Vec<f64> = piv.iter().map(|&p| key[p]).collect();
            k.sort_by(f64::total_cmp);
            k
        };
        // at t = 0 the factor is mm itself up to the column order
        for (c, &kk) in sorted.iter().enumerate() {
            let col = f.x_occ.column(c);
            let src = (0..m)
                .find(|&q| {
                    key[piv[q]] == kk && (&mm.column(q) - &col).iter().all(|x| x.abs() < 1e-14)
                })
                .expect("column is a permuted pivot column");
            assert_eq!(key[piv[src]], kk);
        }
    }
}
