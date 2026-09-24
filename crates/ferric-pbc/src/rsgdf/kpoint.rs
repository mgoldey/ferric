//! Stage 3: k-point range-separated Gaussian density fitting (RS-GDF per
//! momentum transfer q) and the [`KPointJk`] builder on it. Rust port of the
//! validated prototype `reference/pbc/pbc_kgdf.py` (FINDINGS "Iteration 11
//! (Python, k-point RS-GDF)"; conventions of "Iteration 9").
//!
//! # Target
//!
//! For every mesh pair `(k, k')`, `q = k' − k` (a Gamma-centred class), fit
//! the Bloch pair density `a_ml(K) = P^{k'}_ml(K)` ([`crate::pair_ft::residues`],
//! `K = G + q`) in the q-Bloch aux `X^q_P = Σ_T e^{iq·T} χ_P(r − T)`, in the
//! Coulomb metric of the SAME `K = 0`-dropped kernel the dense oracle
//! ([`crate::kdense_aft`]) uses:
//!
//! ```text
//! J2(q)[P,Q]      = SR Σ_T e^{+iq·T} (P_0|Q_T)_erfc          + (1/Ω) Σ_{K∈G+q, K≠0} v_lr(K) conj X_P(K) X_Q(K)
//! J3(k,k')[ml,P]  = SR Σ_{L,T} e^{ik'·L} e^{−iq·T} (m_0 l_L|P_T)_erfc + (1/Ω) Σ_K v_lr(K) conj X_P(K) a_ml(K)
//! J2(q) = U s U^H (eigh_herm, keep s > lindep PER q),  B(k,k')[a, ml] = Σ_P conj(U_Pa) s_a^{−1/2} J3[ml, P]
//! Kker[k,k'][(ml),(ns)] ≈ Σ_a B_a,ml conj B_a,ns,       Jker[k,k'][(mn),(ls)] ≈ Σ_a B_a,mn(k,k) conj B_a,ls(k',k')
//! ```
//!
//! # What is new relative to Gamma ([`super::RsGdf`])
//!
//! * SR integrals are q- and k-INDEPENDENT: the Gamma walk (same triplets,
//!   same screen) is computed ONCE and binned by `(L mod mesh, T mod mesh)`
//!   (reals); each `(q, k')` is a phase contraction of the bins.
//! * G = 0: `v_erfc` is finite at `K = 0`, and `K = 0` lies only on the
//!   q = 0 lattice, so ONLY q = 0 subtracts `c0 q qᵀ` / `c0 S(k) q`
//!   ([`subtract_g0`] for the metric; the complex-S three-index twin here).
//! * q = 0 Hermitisation of `J3(k,k)` in (m,n) (the Gamma
//!   `symmetrize_pairs`): REQUIRED — truncated SR image sets break it at the
//!   precision level, J turns non-Hermitian and the SCF stalls at loose
//!   precision (prototype, prec 1e-8).
//! * LR: at a time-reversal-invariant q (`2q ∈ G`, incl. q = 0) the `±K`
//!   partners are both on the q lattice, so the (G, −G) half set with weight
//!   2 and REAL `2 Re[conj(X) Q_r]` per residue applies (q = 0 uses exactly
//!   the Gamma half set); every other q sums the FULL complex `G + q` set.
//! * Time reversal: `J2(−q) = conj J2(q)`, `J3(−k,−k') = conj J3(k,k')`;
//!   with `U(−q) := conj U(q)`, `B(−k,−k') = conj B(k,k')`, so only one of
//!   each `(q, −q)` is built and the partner is read by conjugation.
//! * K: `(1/N_k) Σ_{k'} Σ_a B_a(k,k') D(k') B_a(k,k')^H`, one q block at a
//!   time; J from the q = 0 block; `v_M S(k) D(k) S(k)` with the SUPERCELL
//!   Madelung constant ([`KPointMesh::madelung`]) for `exxdiv = ewald`.
//!
//! k-point RS-GDF IS the Gamma RS-GDF of the diag(N) supercell (block-diagonal
//! metric in q with the same eigenvalues, so the same lindep cut): no fitting
//! error of its own (prototype 4.7e-15).
//!
//! # Memory
//!
//! Every big buffer is reserved on a [`crate::budget`] ledger first: S(k)
//! copies, the SR residue bins `8 (R_T naux² + R_L R_T nao² naux)`, the
//! resident B `16 N_built N_k naux nao²` (upper bound, before the lindep
//! cut), the per-q working set, the pair-image and K lists; the LR pair FT
//! is K-chunked within what remains.

use super::{
    aux_ft_shells, check_obs_on_cell, dot3, gshells, pair_image_radius, subtract_g0, G0Handling,
    LatticeWalker, RsGdfConfig, Stage,
};
use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::hcore::{gvector_list_bytes, half_gvectors, G_CHUNK_BYTES};
use crate::kpts::{lattice_coords, unit_root, KPointMesh};
use crate::kscf::{eigh_herm, KPointJk};
use crate::lattice::Cell;
use crate::pair_ft::residues::{pair_ft_residues_chunked, residue_coords, residue_index};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2, Array3, ArrayView1};
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// Default hard cap for [`KRsGdf::fitted_kernels`] (test/diagnostic
/// `2 N_k² nao⁴` complex).
pub const DEFAULT_K_FITTED_KERNELS_MAX_BYTES: usize = 512 << 20;

/// TEST-ONLY deliberate defects / references for `tests/pbc_krsgdf.rs` (the
/// prototype's `_MUTANT` switch). Never set in production.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KRsGdfMutation {
    /// `e^{+iq·T}` instead of `e^{−iq·T}` on the SR aux images of J3
    /// (q ≠ 0 blocks only; prototype max|ΔKker| 1.8e2 on the anchor).
    QPhaseSign,
    /// Gamma G = 0 bookkeeping at EVERY q (with `S(k')` at q ≠ 0; prototype
    /// 6.4e-2 and an indefinite J2(q ≠ 0)).
    G0AllQ,
    /// Skip the q = 0 (m,n) Hermitisation of `J3(k,k)`.
    NoHermQ0,
    /// NOT a defect: build every q class explicitly (the reference for the
    /// time-reversal fill).
    NoTimeReversal,
}

/// Settings for [`KRsGdf::build`]: the Gamma [`RsGdfConfig`] (ω, precision,
/// lindep, exxdiv, SR screen, budget, G = 0 mode — all with the same meaning,
/// per q) plus the test-only mutation switch.
#[derive(Debug, Clone, Copy)]
pub struct KRsGdfConfig {
    pub gdf: RsGdfConfig,
    #[doc(hidden)]
    pub mutation: Option<KRsGdfMutation>,
}

impl Default for KRsGdfConfig {
    fn default() -> Self {
        Self {
            gdf: RsGdfConfig::default(),
            mutation: None,
        }
    }
}

/// Per-q-class build record (mirrored classes copy their partner's numbers).
#[derive(Debug, Clone)]
pub struct KRsGdfQStats {
    /// q-class index ([`KPointMesh::q_class`]).
    pub iq: usize,
    /// Cartesian q (Bohr⁻¹).
    pub q: [f64; 3],
    /// `2q ∈ G` (half-set LR, real metric).
    pub trim: bool,
    /// `Some(p)`: not built, read as the conjugate of class `p`.
    pub mirrored_from: Option<usize>,
    pub naux: usize,
    /// Metric eigenvectors kept (rows of every B of this class).
    pub naux_kept: usize,
    /// Metric eigenvalues `<= lindep` dropped at this q.
    pub n_dropped: usize,
    pub metric_eig_min: f64,
    pub metric_eig_max: f64,
    /// LR K vectors summed (half set at TRIM q, full set otherwise).
    pub n_k_vectors: usize,
    /// max |J2 − J2^H| before Hermitisation.
    pub asym_j2: f64,
    /// q = 0 only: max |J3\[mn\] − conj J3\[nm\]| over k before Hermitisation.
    pub asym_j3: f64,
}

/// What the build did (counts and conditioning).
#[derive(Debug, Clone)]
pub struct KRsGdfStats {
    pub nk: usize,
    pub naux: usize,
    /// One entry per q class, in class order.
    pub per_q: Vec<KRsGdfQStats>,
    /// q classes built explicitly (the rest by time reversal).
    pub n_q_built: usize,
    pub n_pair_images: usize,
    /// SR 3-centre shell triplets / 2-centre shell pairs (computed ONCE).
    pub n_sr3_triplets: usize,
    pub n_sr2_pairs: usize,
    /// Residue counts of the pair-image (`R_L`) and aux-image (`R_T`) bins.
    pub residues_l: usize,
    pub residues_t: usize,
    pub n_lr_chunks: usize,
    /// Complex elements of the resident B.
    pub b_elements: usize,
    pub budget_bytes: usize,
    pub resident_bytes: usize,
}

/// One built q class: `b[k']` is `B(k'−q, k')`, `(naux_kept, nao²)`.
#[derive(Debug, Clone)]
struct QBlock {
    kof: Vec<usize>,
    b: Vec<Array2<C64>>,
    /// Also serves `(−k, −k')` by conjugation.
    mirror: bool,
}

/// Complex k-point RS-GDF tensors for a mesh and what the J/K builder needs.
#[derive(Debug, Clone)]
pub struct KRsGdf {
    nao: usize,
    nk: usize,
    blocks: Vec<QBlock>,
    /// Index into `blocks` of the q = 0 class.
    q0: usize,
    /// `[k * N_k + k']` → (block, k' slot, conjugate?).
    pair_loc: Vec<(usize, usize, bool)>,
    minus: Vec<usize>,
    s: Vec<Array2<C64>>,
    madelung: f64,
    madelung_ewald: f64,
    stats: KRsGdfStats,
}

fn sat_prod(xs: &[usize]) -> u64 {
    xs.iter().fold(1u64, |a, &x| a.saturating_mul(x as u64))
}

/// SR metric binned by the residue of `T` modulo `moduli`: `bins[r]` =
/// `Σ_{T ≡ r} (P_0|Q_T)_erfc` (reals). Same walk and per-bin addition order as
/// the Gamma `Stage::sr_metric`.
fn sr_metric_binned(
    st: &Stage<'_>,
    moduli: [usize; 3],
) -> Result<(Vec<Array2<f64>>, usize), FerricError> {
    let naux = st.aux.nbasis();
    let nr = moduli[0] * moduli[1] * moduli[2];
    let b = st.cell.reciprocal();
    let mut bins: Vec<Array2<f64>> = (0..nr).map(|_| Array2::zeros((naux, naux))).collect();
    let count = st.sr_metric_each(|p, q, t, blk| {
        let j2 = &mut bins[residue_index(lattice_coords(&b, &t), moduli)];
        for i in 0..p.nfun {
            for j in 0..q.nfun {
                j2[(p.off + i, q.off + j)] += blk[i * q.nfun + j];
            }
        }
    })?;
    Ok((bins, count))
}

/// SR 3-index binned by `(L mod mod_l, T mod mod_t)`: `bins[rL * R_T + rT]`
/// = `Σ (m_0 l_L|P_T)_erfc`, `(nao², naux)` reals. Same walk as the Gamma
/// `Stage::sr_three_index`.
fn sr_three_index_binned(
    st: &Stage<'_>,
    images: &[[f64; 3]],
    mod_l: [usize; 3],
    mod_t: [usize; 3],
) -> Result<(Vec<Array2<f64>>, usize), FerricError> {
    let n = st.obs.nbasis();
    let naux = st.aux.nbasis();
    let rt = mod_t[0] * mod_t[1] * mod_t[2];
    let nr = mod_l[0] * mod_l[1] * mod_l[2] * rt;
    let b = st.cell.reciprocal();
    let mut bins: Vec<Array2<f64>> = (0..nr).map(|_| Array2::zeros((n * n, naux))).collect();
    let count = st.sr_three_index_each(images, |a, bs, p, l, t, blk| {
        let r = residue_index(lattice_coords(&b, &l), mod_l) * rt
            + residue_index(lattice_coords(&b, &t), mod_t);
        let j3 = &mut bins[r];
        for pp in 0..p.nfun {
            for i in 0..a.nfun {
                let row0 = (a.off + i) * n + bs.off;
                let src = (pp * a.nfun + i) * bs.nfun;
                for j in 0..bs.nfun {
                    j3[(row0 + j, p.off + pp)] += blk[src + j];
                }
            }
        }
    })?;
    Ok((bins, count))
}

/// TEST/DIAGNOSTIC: the Gamma SR sums (`(J2_SR, J3_SR)` of [`super::RsGdf`])
/// and the SAME sums from the residue-binned k-point walk at a single bin
/// (1×1×1). They must agree BIT FOR BIT: the binned mode visits the same
/// blocks in the same order, and the Gamma sums are what `RsGdf::build`
/// consumes (the refactor-safety pin for the Gamma path).
#[doc(hidden)]
pub fn sr_sums_gamma_and_single_bin(
    cell: &Cell,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    cfg: &RsGdfConfig,
) -> Result<[(Array2<f64>, Array2<f64>); 2], FerricError> {
    cfg.validate()?;
    check_obs_on_cell(cell, obs)?;
    let st = Stage {
        cell,
        obs,
        aux,
        obs_sh: gshells(obs, "RsGdf orbital basis")?,
        aux_sh: gshells(aux, "RsGdf aux basis")?,
        omega: cfg.omega,
        thresh: cfg.precision,
        sr_screen: cfg.sr_screen,
        walker: LatticeWalker::new(cell),
    };
    let images = cell.translations(pair_image_radius(&st, cfg.precision))?;
    let (j2g, _) = st.sr_metric()?;
    let (j3g, _) = st.sr_three_index(&images)?;
    let one = [1usize; 3];
    let (mut j2b, _) = sr_metric_binned(&st, one)?;
    let (mut j3b, _) = sr_three_index_binned(&st, &images, one, one)?;
    Ok([(j2g, j3g), (j2b.remove(0), j3b.remove(0))])
}

/// LR K vectors of class `iq`: q = 0 → the Gamma half sphere (weight 2);
/// TRIM q → the canonical half of `{G + q}` by doubled integer coordinates
/// (exact, no float sign test); otherwise the full `{G + q}` set. `K = 0`
/// never included; `|K| <= gcut`.
fn lr_kvectors(
    cell: &Cell,
    mesh: &KPointMesh,
    iq: usize,
    trim: bool,
    gcut: f64,
) -> Result<(Vec<[f64; 3]>, bool), FerricError> {
    let (q, mq) = mesh.q_class(iq);
    if mq == [0, 0, 0] {
        return Ok((half_gvectors(cell, gcut)?, true));
    }
    let nm = mesh.n();
    let mut q2 = [0i64; 3];
    for i in 0..3 {
        if trim {
            let x = 2 * mq[i];
            if x % nm[i] as i64 != 0 {
                return Err(FerricError::General(format!(
                    "KRsGdf: q class {iq} flagged time-reversal invariant but 2q is not in G \
                     (internal error)"
                )));
            }
            q2[i] = x / nm[i] as i64;
        }
    }
    let qn = dot3(&q, &q).sqrt();
    let g2cut = gcut * gcut;
    let a = *cell.lattice();
    let mut out = Vec::new();
    for g in cell.gvectors(gcut + qn)? {
        let k = [g[0] + q[0], g[1] + q[1], g[2] + q[2]];
        let k2 = dot3(&k, &k);
        if !(k2 > 0.0 && k2 <= g2cut) {
            continue;
        }
        if trim {
            // K = Σ (n_i + m_i/N_i) b_i; doubled coordinates are integers.
            let ng = lattice_coords(&a, &g);
            let d = [2 * ng[0] + q2[0], 2 * ng[1] + q2[1], 2 * ng[2] + q2[2]];
            let pos = if d[0] != 0 {
                d[0] > 0
            } else if d[1] != 0 {
                d[1] > 0
            } else {
                d[2] > 0
            };
            if !pos {
                continue;
            }
        }
        out.push(k);
    }
    Ok((out, trim))
}

/// LR contribution of the K set `kv` to `J2(q)` and to the per-residue
/// accumulators `acc[rL]` (`(nao², naux)`, the `e^{ik'·L}`-free part of J3):
/// `half`: `(2/Ω) Σ v Re[conj(A) X]` (reals); otherwise `(1/Ω) Σ v conj(X) A`.
fn lr_accumulate_q(
    st: &Stage<'_>,
    kv: &[[f64; 3]],
    half: bool,
    moduli: [usize; 3],
    j2: &mut Array2<C64>,
    acc: &mut [Array2<C64>],
    chunk_budget: usize,
) -> Result<usize, FerricError> {
    let n = st.obs.nbasis();
    let n2 = n * n;
    let naux = st.aux.nbasis();
    let vol = st.cell.volume();
    let omega = st.omega;
    // Re/Im pair copies (one residue at a time) + X, conj(X)·v and the four
    // real aux copies of the half-set path, per K.
    let extra_per_g = n2
        .saturating_mul(16)
        .saturating_add(naux.saturating_mul(64))
        .saturating_add(64);
    let thresh = (0.01 * st.thresh).min(crate::pair_ft::DEFAULT_PAIR_FT_THRESH);
    let aux_sh = &st.aux_sh;
    let one = C64::new(1.0, 0.0);
    let sink = |_k0: usize, ks: &[[f64; 3]], qs: &[Array3<C64>]| -> Result<(), FerricError> {
        let ng = ks.len();
        let x = aux_ft_shells(aux_sh, naux, ks);
        let fac = if half { 2.0 } else { 1.0 };
        let wts: Vec<f64> = ks
            .iter()
            .map(|k| {
                let k2 = dot3(k, k);
                fac / vol * 4.0 * PI / k2 * (-k2 / (4.0 * omega * omega)).exp()
            })
            .collect();
        if half {
            let mut xr = Array2::<f64>::zeros((naux, ng));
            let mut xi = Array2::<f64>::zeros((naux, ng));
            let mut xrw = Array2::<f64>::zeros((naux, ng));
            let mut xiw = Array2::<f64>::zeros((naux, ng));
            for g in 0..ng {
                for p in 0..naux {
                    let z = x[(p, g)];
                    xr[(p, g)] = z.re;
                    xi[(p, g)] = z.im;
                    xrw[(p, g)] = wts[g] * z.re;
                    xiw[(p, g)] = wts[g] * z.im;
                }
            }
            let mut j2r = Array2::<f64>::zeros((naux, naux));
            general_mat_mul(1.0, &xrw, &xr.t(), 0.0, &mut j2r);
            general_mat_mul(1.0, &xiw, &xi.t(), 1.0, &mut j2r);
            j2.zip_mut_with(&j2r, |z, &v| z.re += v);
            let mut pr = Array2::<f64>::zeros((n2, ng));
            let mut pim = Array2::<f64>::zeros((n2, ng));
            let mut m = Array2::<f64>::zeros((n2, naux));
            for (qr, a) in qs.iter().zip(acc.iter_mut()) {
                for mm in 0..n {
                    for l in 0..n {
                        let row = mm * n + l;
                        for g in 0..ng {
                            let z = qr[[mm, l, g]];
                            pr[(row, g)] = wts[g] * z.re;
                            pim[(row, g)] = wts[g] * z.im;
                        }
                    }
                }
                // Re[conj(X) Q] = X.re Q.re + X.im Q.im
                general_mat_mul(1.0, &pr, &xr.t(), 0.0, &mut m);
                general_mat_mul(1.0, &pim, &xi.t(), 1.0, &mut m);
                a.zip_mut_with(&m, |z, &v| z.re += v);
            }
        } else {
            let mut xcw = Array2::<C64>::zeros((naux, ng));
            for p in 0..naux {
                for g in 0..ng {
                    xcw[(p, g)] = x[(p, g)].conj() * wts[g];
                }
            }
            // J2[P,Q] += Σ_K v conj X_P X_Q
            general_mat_mul(one, &xcw, &x.t(), one, &mut *j2);
            for (qr, a) in qs.iter().zip(acc.iter_mut()) {
                let q2 = qr
                    .view()
                    .into_shape_with_order((n2, ng))
                    .map_err(|e| FerricError::General(format!("KRsGdf LR reshape: {e}")))?;
                // acc[ml,P] += Σ_K Q_ml(K) v conj X_P(K)
                general_mat_mul(one, &q2, &xcw.t(), one, a);
            }
        }
        Ok(())
    };
    pair_ft_residues_chunked(
        st.cell,
        st.obs,
        kv,
        moduli,
        thresh,
        chunk_budget,
        extra_per_g,
        sink,
    )
}

/// `J3[mn,P] −= c0 S_mn q_P` (complex S; the three-index half of
/// [`subtract_g0`], same mode switch).
fn subtract_g0_three_index(
    j3: &mut Array2<C64>,
    s: &Array2<C64>,
    q: &[f64],
    c0: f64,
    mode: G0Handling,
) {
    if mode != G0Handling::Consistent {
        return;
    }
    let n = s.nrows();
    for m in 0..n {
        for k in 0..n {
            let smn = s[(m, k)];
            let row = m * n + k;
            for (p, qp) in q.iter().enumerate() {
                j3[(row, p)] -= smn * (c0 * qp);
            }
        }
    }
}

/// max |J3[mn,P] − conj J3[nm,P]|.
fn pair_herm_asym(j3: &Array2<C64>, n: usize) -> f64 {
    let mut w = 0.0_f64;
    for m in 0..n {
        for k in 0..=m {
            let (r1, r2) = (m * n + k, k * n + m);
            for p in 0..j3.ncols() {
                w = w.max((j3[(r1, p)] - j3[(r2, p)].conj()).norm());
            }
        }
    }
    w
}

/// `J3[mn,P] ← ½ (J3[mn,P] + conj J3[nm,P])` (the q = 0 Hermitisation).
fn hermitize_pairs(j3: &mut Array2<C64>, n: usize) {
    for m in 0..n {
        for k in 0..=m {
            let (r1, r2) = (m * n + k, k * n + m);
            for p in 0..j3.ncols() {
                let avg = (j3[(r1, p)] + j3[(r2, p)].conj()) * 0.5;
                j3[(r1, p)] = avg;
                j3[(r2, p)] = avg.conj();
            }
        }
    }
}

impl KRsGdf {
    /// Build the per-q fitted tensors for `mesh` on `cell`: orbital basis
    /// `obs` (built from `cell.mol()`), aux `aux` (any centres, as
    /// [`super::RsGdf::build`]), `s_k` = `S(k)` per mesh point (G = 0 term
    /// and Madelung shift). See the module doc.
    pub fn build(
        cell: &Cell,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
        mesh: &KPointMesh,
        s_k: &[Array2<C64>],
        cfg: &KRsGdfConfig,
    ) -> Result<Self, FerricError> {
        let g = &cfg.gdf;
        g.validate()?;
        super::require_pure_aux(aux, "KRsGdf")?;
        let n = obs.nbasis();
        let n2 = n * n;
        let naux = aux.nbasis();
        let nk = mesh.nk();
        if naux == 0 || n == 0 {
            return Err(FerricError::General(format!(
                "KRsGdf: empty basis (nao = {n}, naux = {naux})"
            )));
        }
        if s_k.len() != nk || s_k.iter().any(|s| s.dim() != (n, n)) {
            return Err(FerricError::General(format!(
                "KRsGdf: need {nk} overlap matrices of shape ({n}, {n}), got {}",
                s_k.len()
            )));
        }
        check_obs_on_cell(cell, obs)?;
        let st = Stage {
            cell,
            obs,
            aux,
            obs_sh: gshells(obs, "KRsGdf orbital basis")?,
            aux_sh: gshells(aux, "KRsGdf aux basis")?,
            omega: g.omega,
            thresh: g.precision,
            sr_screen: g.sr_screen,
            walker: LatticeWalker::new(cell),
        };
        let mod_l = mesh.residue_moduli();
        let mod_t = mesh.n();
        let rl = mod_l[0] * mod_l[1] * mod_l[2];
        let rt = mod_t[0] * mod_t[1] * mod_t[2];
        let use_tr = cfg.mutation != Some(KRsGdfMutation::NoTimeReversal);
        let reps: Vec<usize> = (0..nk)
            .filter(|&iq| !use_tr || mesh.q_minus(iq) >= iq)
            .collect();
        let n_built = reps.len();

        let mut ledger = Ledger::new(crate::budget::resolve(g.budget_bytes));
        ledger.reserve(
            &format!("KRsGdf S(k) copies + Madelung temporaries (nao = {n}, N_k = {nk})"),
            bytes_of(sat_prod(&[nk + 2, n2]), 16),
        )?;
        ledger.reserve(
            &format!("KRsGdf SR residue bins (R_L = {rl}, R_T = {rt}, naux = {naux}, nao = {n})"),
            bytes_of(
                sat_prod(&[rt, naux, naux]).saturating_add(sat_prod(&[rl, rt, n2, naux])),
                8,
            ),
        )?;
        ledger.reserve(
            &format!(
                "KRsGdf resident B ({n_built} q classes x N_k = {nk} x naux = {naux} x nao² = {n2})"
            ),
            bytes_of(sat_prod(&[n_built, nk, naux, n2]), 16),
        )?;
        ledger.reserve(
            &format!("KRsGdf per-q working set (metric, eigenvectors, R_L = {rl} LR accumulators)"),
            bytes_of(
                sat_prod(&[4, naux, naux])
                    .saturating_add(sat_prod(&[rl, n2, naux]))
                    .saturating_add(sat_prod(&[2, n2, naux])),
                16,
            )
            .saturating_add(bytes_of(
                sat_prod(&[naux, naux]).saturating_add(sat_prod(&[n2, naux])),
                8,
            )),
        )?;
        let rpair = pair_image_radius(&st, g.precision);
        ledger.reserve(
            &format!("KRsGdf pair-image list (r_pair = {rpair:.2} Bohr)"),
            bytes_of(cell.translation_count_bound(rpair)?, 24),
        )?;
        let images = cell.translations(rpair)?;
        let gcut = 2.0 * g.omega * (1.0 / g.precision).ln().sqrt();
        let qmax = (0..nk)
            .map(|iq| {
                let q = mesh.q_class(iq).0;
                dot3(&q, &q).sqrt()
            })
            .fold(0.0_f64, f64::max);
        ledger.reserve(
            &format!("KRsGdf LR K list (|K| <= {gcut:.3}, |q| <= {qmax:.3})"),
            gvector_list_bytes(cell, gcut + qmax)?.saturating_mul(2),
        )?;
        let resident_bytes = ledger.resident();
        let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);

        // --- SR, once for every q: residue bins.
        let (j2res, n_sr2) = sr_metric_binned(&st, mod_t)?;
        let (j3res, n_sr3) = sr_three_index_binned(&st, &images, mod_l, mod_t)?;

        let qv: Vec<f64> = aux_ft_shells(&st.aux_sh, naux, &[[0.0; 3]])
            .column(0)
            .iter()
            .map(|z| z.re)
            .collect();
        let c0 = PI / (g.omega * g.omega * cell.volume());
        let phk: Vec<Vec<C64>> = (0..nk)
            .map(|k| {
                (0..rl)
                    .map(|r| mesh.phase(k, residue_coords(r, mod_l)))
                    .collect()
            })
            .collect();
        let dq = rt as i64;
        // e^{+iq·t_r} for aux-image residue r (q = Σ m_i/N_i b_i).
        let q_phase = |mq: [i64; 3], r: usize| -> C64 {
            let c = residue_coords(r, mod_t);
            let mut t = 0i64;
            for i in 0..3 {
                t += mq[i] * c[i] * (dq / mod_t[i] as i64);
            }
            unit_root(t, dq)
        };

        let mut blocks: Vec<QBlock> = Vec::with_capacity(n_built);
        let mut per_q: Vec<Option<KRsGdfQStats>> = vec![None; nk];
        let mut n_lr_chunks = 0usize;
        let mut b_elements = 0usize;
        let mut q0 = usize::MAX;
        for &iq in &reps {
            let iqm = mesh.q_minus(iq);
            let trim = iqm == iq;
            let (q, mq) = mesh.q_class(iq);
            let php: Vec<C64> = (0..rt).map(|r| q_phase(mq, r)).collect();
            let flip = cfg.mutation == Some(KRsGdfMutation::QPhaseSign);

            // SR: J2(q) = Σ_T e^{+iq·T} bins; acc[rL] = Σ_T e^{−iq·T} bins.
            let mut j2 = Array2::<C64>::zeros((naux, naux));
            for (bin, ph) in j2res.iter().zip(&php) {
                j2.zip_mut_with(bin, |z, &x| *z += *ph * x);
            }
            let mut acc: Vec<Array2<C64>> = (0..rl)
                .map(|r_l| {
                    let mut a = Array2::<C64>::zeros((n2, naux));
                    for (r_t, ph) in php.iter().enumerate() {
                        let ph = if flip { *ph } else { ph.conj() };
                        a.zip_mut_with(&j3res[r_l * rt + r_t], |z, &x| *z += ph * x);
                    }
                    a
                })
                .collect();

            // LR on K = G + q.
            let (kv, half) = lr_kvectors(cell, mesh, iq, trim, gcut)?;
            n_lr_chunks += lr_accumulate_q(&st, &kv, half, mod_l, &mut j2, &mut acc, chunk_budget)?;

            // G = 0: q = 0 only (the mutant: every q).
            let g0_here = mq == [0, 0, 0] || cfg.mutation == Some(KRsGdfMutation::G0AllQ);
            if g0_here {
                let mut re = j2.mapv(|z| z.re);
                let mut no_j3 = Array2::<f64>::zeros((0, naux));
                subtract_g0(&mut re, &mut no_j3, &[], &qv, c0, g.g0);
                j2.zip_mut_with(&re, |z, &x| z.re = x);
            }

            // Hermitian metric solve with the per-q lindep cut.
            let mut asym_j2 = 0.0_f64;
            for i in 0..naux {
                for j in 0..=i {
                    asym_j2 = asym_j2.max((j2[(i, j)] - j2[(j, i)].conj()).norm());
                }
            }
            let j2h = {
                let h = j2.t().mapv(|z| z.conj());
                (&j2 + &h).mapv(|z| z * 0.5)
            };
            drop(j2);
            let (evals, evecs) = eigh_herm(&j2h).map_err(|e| {
                FerricError::Lapack(format!("KRsGdf metric eigh at q class {iq}: {e}"))
            })?;
            drop(j2h);
            let keep: Vec<usize> = (0..naux).filter(|&k| evals[k] > g.lindep).collect();
            if keep.is_empty() {
                return Err(FerricError::General(format!(
                    "KRsGdf: every metric eigenvalue at q class {iq} is <= lindep {:e} (max {:e})",
                    g.lindep,
                    evals[naux - 1]
                )));
            }
            // W[P,a] = conj(U[P,a]) s_a^{-1/2}; B = (J3 W)ᵀ.
            let mut w = Array2::<C64>::zeros((naux, keep.len()));
            for (c, &k) in keep.iter().enumerate() {
                let f = 1.0 / evals[k].sqrt();
                for r in 0..naux {
                    w[(r, c)] = evecs[(r, k)].conj() * f;
                }
            }
            drop(evecs);

            let kof: Vec<usize> = (0..nk).map(|j| mesh.k_minus_q(j, mq)).collect();
            let mut bq: Vec<Array2<C64>> = Vec::with_capacity(nk);
            let mut asym_j3 = 0.0_f64;
            for (j, ph_j) in phk.iter().enumerate() {
                let mut j3 = Array2::<C64>::zeros((n2, naux));
                for (a, ph) in acc.iter().zip(ph_j) {
                    j3.zip_mut_with(a, |z, &x| *z += *ph * x);
                }
                if g0_here {
                    // q = 0: k = k', a_ml(0) = S_ml(k').
                    subtract_g0_three_index(&mut j3, &s_k[j], &qv, c0, g.g0);
                }
                if mq == [0, 0, 0] {
                    asym_j3 = asym_j3.max(pair_herm_asym(&j3, n));
                    if cfg.mutation != Some(KRsGdfMutation::NoHermQ0) {
                        hermitize_pairs(&mut j3, n);
                    }
                }
                let bt = j3.dot(&w); // (n², kept)
                bq.push(bt.t().as_standard_layout().into_owned());
            }
            drop(acc);
            b_elements += nk * keep.len() * n2;
            let mirror = use_tr && iqm != iq;
            let qs = KRsGdfQStats {
                iq,
                q,
                trim,
                mirrored_from: None,
                naux,
                naux_kept: keep.len(),
                n_dropped: naux - keep.len(),
                metric_eig_min: evals[0],
                metric_eig_max: evals[naux - 1],
                n_k_vectors: kv.len(),
                asym_j2,
                asym_j3,
            };
            if mirror {
                per_q[iqm] = Some(KRsGdfQStats {
                    iq: iqm,
                    q: mesh.q_class(iqm).0,
                    mirrored_from: Some(iq),
                    ..qs.clone()
                });
            }
            per_q[iq] = Some(qs);
            if mq == [0, 0, 0] {
                q0 = blocks.len();
            }
            blocks.push(QBlock { kof, b: bq, mirror });
        }
        if q0 == usize::MAX {
            return Err(FerricError::General(
                "KRsGdf: the q = 0 class was not built (internal error)".into(),
            ));
        }
        let minus: Vec<usize> = (0..nk).map(|k| mesh.minus(k)).collect();
        let unset = (usize::MAX, 0, false);
        let mut pair_loc = vec![unset; nk * nk];
        for (bi, blk) in blocks.iter().enumerate() {
            for (j, &k) in blk.kof.iter().enumerate() {
                pair_loc[k * nk + j] = (bi, j, false);
                if blk.mirror {
                    pair_loc[minus[k] * nk + minus[j]] = (bi, j, true);
                }
            }
        }
        if pair_loc.iter().any(|p| p.0 == usize::MAX) {
            return Err(FerricError::General(
                "KRsGdf: some (k, k') pair has no q block (internal error)".into(),
            ));
        }
        let per_q: Vec<KRsGdfQStats> = per_q
            .into_iter()
            .enumerate()
            .map(|(iq, s)| {
                s.ok_or_else(|| {
                    FerricError::General(format!("KRsGdf: q class {iq} has no record (internal)"))
                })
            })
            .collect::<Result<_, _>>()?;
        let madelung_ewald = mesh.madelung(cell)?;
        let madelung = match g.exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => madelung_ewald,
        };
        ferric_core::memory::warn_if_rss_over("ferric-pbc KRsGdf", ledger.budget(), 1.1);
        let stats = KRsGdfStats {
            nk,
            naux,
            per_q,
            n_q_built: n_built,
            n_pair_images: images.len(),
            n_sr3_triplets: n_sr3,
            n_sr2_pairs: n_sr2,
            residues_l: rl,
            residues_t: rt,
            n_lr_chunks,
            b_elements,
            budget_bytes: ledger.budget(),
            resident_bytes,
        };
        Ok(Self {
            nao: n,
            nk,
            blocks,
            q0,
            pair_loc,
            minus,
            s: s_k.to_vec(),
            madelung,
            madelung_ewald,
            stats,
        })
    }

    /// Build statistics (per-q kept/dropped counts, conditioning, counts).
    pub fn stats(&self) -> &KRsGdfStats {
        &self.stats
    }

    /// Number of k-points.
    pub fn nk(&self) -> usize {
        self.nk
    }

    /// `v_M` applied in K (0 for `ExxDiv::None`).
    pub fn madelung(&self) -> f64 {
        self.madelung
    }

    /// The mesh (supercell) Madelung constant `ExxDiv::Ewald` applies.
    pub fn madelung_ewald(&self) -> f64 {
        self.madelung_ewald
    }

    /// The same tensors with another exchange-divergence treatment.
    pub fn with_exxdiv(mut self, exxdiv: ExxDiv) -> Self {
        self.madelung = match exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => self.madelung_ewald,
        };
        self
    }

    /// `B(k, k')`, `(naux_kept(q), nao²)`, row a, column `m·nao+l`
    /// (conjugated from the partner class when mirrored). Copies.
    pub fn block(&self, k: usize, kp: usize) -> Array2<C64> {
        let (bi, j, conj) = self.pair_loc[k * self.nk + kp];
        let b = &self.blocks[bi].b[j];
        if conj {
            b.mapv(|z| z.conj())
        } else {
            b.clone()
        }
    }

    /// TEST/DIAGNOSTIC: fitted dense kernels `(Jker, Kker)`, `[k * N_k + k']`
    /// each `(nao², nao²)` in the [`crate::kdense_aft::KDenseAftEri`] layout.
    /// Errors above `max_bytes`.
    #[allow(clippy::type_complexity)]
    pub fn fitted_kernels(
        &self,
        max_bytes: usize,
    ) -> Result<(Vec<Array2<C64>>, Vec<Array2<C64>>), FerricError> {
        let (nk, n2) = (self.nk, self.nao * self.nao);
        let bytes = 2u128 * (nk as u128).pow(2) * (n2 as u128).pow(2) * 16;
        if bytes > max_bytes as u128 {
            return Err(FerricError::General(format!(
                "KRsGdf::fitted_kernels: {bytes} bytes (nao = {}, N_k = {nk}) > cap {max_bytes}",
                self.nao
            )));
        }
        let b0 = &self.blocks[self.q0];
        let mut jk = Vec::with_capacity(nk * nk);
        let mut kk = Vec::with_capacity(nk * nk);
        for k in 0..nk {
            for kp in 0..nk {
                let bk = &b0.b[k];
                let bkp = b0.b[kp].mapv(|z| z.conj());
                jk.push(bk.t().dot(&bkp));
                let b = self.block(k, kp);
                kk.push(b.t().dot(&b.mapv(|z| z.conj())));
            }
        }
        Ok((jk, kk))
    }

    /// J/K builder with this tensor's `v_M`.
    pub fn jk_builder(&self) -> KRsGdfJk<'_> {
        KRsGdfJk {
            gdf: self,
            madelung: self.madelung,
        }
    }

    /// J/K builder with an explicit `v_M` (0 = `exxdiv=None`).
    pub fn jk_builder_with_madelung(&self, madelung: f64) -> KRsGdfJk<'_> {
        KRsGdfJk {
            gdf: self,
            madelung,
        }
    }

    /// `J(k)`, `K(k)` (K including `v_M S D S`) for densities `dm` (one per
    /// mesh point). J from the q = 0 block; K one q block at a time.
    /// Overwrites `j`, `k`.
    pub fn contract(
        &self,
        dm: &[Array2<C64>],
        madelung: f64,
        j: &mut [Array2<C64>],
        k: &mut [Array2<C64>],
    ) -> Result<(), FerricError> {
        let (n, nk) = (self.nao, self.nk);
        if dm.len() != nk || j.len() != nk || k.len() != nk {
            return Err(FerricError::General(format!(
                "KRsGdfJk: {} densities / {} J / {} K, expected {nk}",
                dm.len(),
                j.len(),
                k.len()
            )));
        }
        for x in dm.iter().chain(j.iter()).chain(k.iter()) {
            if x.dim() != (n, n) {
                return Err(FerricError::General(format!(
                    "KRsGdfJk: matrix of shape {:?}, expected ({n}, {n})",
                    x.dim()
                )));
            }
        }
        let inv = 1.0 / nk as f64;
        let zero = C64::new(0.0, 0.0);

        // J(k) = Σ_a B_a(k,k) ρ_a, ρ_a = (1/N_k) Σ_k' Σ_ls conj B_a,ls(k',k') D_ls(k').
        let b0 = &self.blocks[self.q0];
        let mut rho = Array1::<C64>::zeros(b0.b[0].nrows());
        for (kk, d) in dm.iter().enumerate() {
            let d_std = d.as_standard_layout();
            let dflat = ArrayView1::from(d_std.as_slice().expect("standard layout"));
            // q = 0: kof[k'] = k'.
            let bc = b0.b[kk].mapv(|z| z.conj());
            rho += &bc.dot(&dflat);
        }
        rho.mapv_inplace(|z| z * inv);
        for (kk, jk) in j.iter_mut().enumerate() {
            let jf = b0.b[kk].t().dot(&rho);
            for m in 0..n {
                for nu in 0..n {
                    jk[(m, nu)] = jf[m * n + nu];
                }
            }
        }

        for kx in k.iter_mut() {
            kx.fill(zero);
        }
        for blk in &self.blocks {
            for (jj, &kk) in blk.kof.iter().enumerate() {
                add_exchange(&blk.b[jj], &dm[jj], false, n, inv, &mut k[kk])?;
                if blk.mirror {
                    let (km, jm) = (self.minus[kk], self.minus[jj]);
                    add_exchange(&blk.b[jj], &dm[jm], true, n, inv, &mut k[km])?;
                }
            }
        }
        if madelung != 0.0 {
            for (kk, kx) in k.iter_mut().enumerate() {
                let s = &self.s[kk];
                let sds = s.dot(&dm[kk]).dot(s);
                kx.scaled_add(C64::new(madelung, 0.0), &sds);
            }
        }
        Ok(())
    }
}

/// `out += scale Σ_a B_a D B_a^H` (`B_a` the `(n, n)` row a of `b`, or its
/// conjugate when `conj`).
fn add_exchange(
    b: &Array2<C64>,
    d: &Array2<C64>,
    conj: bool,
    n: usize,
    scale: f64,
    out: &mut Array2<C64>,
) -> Result<(), FerricError> {
    let sc = C64::new(scale, 0.0);
    let one = C64::new(1.0, 0.0);
    for row in b.rows() {
        let ba = row
            .into_shape_with_order((n, n))
            .map_err(|e| FerricError::General(format!("KRsGdfJk: B row reshape: {e}")))?;
        let ba = if conj {
            ba.mapv(|z| z.conj())
        } else {
            ba.to_owned()
        };
        let tmp = ba.dot(d);
        let bh = ba.t().mapv(|z| z.conj());
        general_mat_mul(sc, &tmp, &bh, one, out);
    }
    Ok(())
}

/// [`KPointJk`] over a [`KRsGdf`], including the Madelung shift.
pub struct KRsGdfJk<'a> {
    gdf: &'a KRsGdf,
    madelung: f64,
}

impl KPointJk for KRsGdfJk<'_> {
    fn build(
        &mut self,
        dm: &[Array2<C64>],
        j: &mut [Array2<C64>],
        k: &mut [Array2<C64>],
    ) -> Result<(), FerricError> {
        self.gdf.contract(dm, self.madelung, j, k)
    }
}
