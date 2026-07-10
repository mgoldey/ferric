//! # RI-Laplace MP2
//!
//! This module implements the Resolution of Identity (RI) Laplace-transform MP2 method.
//!
//! ## Theory
//! The canonical MP2 correlation energy is given by:
//! $$E_{corr} = -\sum_{iajb} \frac{(ia|jb)[2(ia|jb) - (ib|ja)]}{\epsilon_a + \epsilon_b - \epsilon_i - \epsilon_j}$$
//!
//! Using the Laplace transform identity for the denominator:
//! $$\frac{1}{x} = \int_0^\infty e^{-tx} dt \approx \sum_k w_k e^{-t_k x}$$
//!
//! The correlation energy can be expressed as an integral over the Laplace parameter $t$.
//! In the RI approximation, the energy factorizes into Coulomb ($J$) and Exchange ($K$)
//! traces that can be evaluated efficiently in either the MO or AO basis.
//!
//! ## Implementation
//! This module provides two implementations:
//! 1. **MO-based (`compute_mo`)**: Transforms the 3-center integrals to the MO basis.
//!    This is $O(N^4)$ and used primarily for validating the quadrature convergence.
//! 2. **AO/MO hybrid (`compute_ao`)**: J term in AO basis via pseudo-densities (supports
//!    future sparse path); K term in MO basis via blocked wide-GEMM Gram contraction
//!    (`laplace_exchange_energy`) on τ-weighted amplitudes hoisted out of the point loop.
//!
//! ## Reference
//! Häser & Almlöf, Chem. Phys. Lett. 191, 299 (1992).
//! Takatsuka, Ten-no, Hackbusch, J. Chem. Phys. 129, 044112 (2008).

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_quadrature::LaplaceQuadrature;
use ferric_scf::ScfResult;
use ndarray::Array2;

use crate::rimp2::active_occ;
use crate::boys::{boys_localize, build_domains, build_pseudo_density_occ_sparse,
                  build_pseudo_density_vir_sparse};

/// Effective per-task memory budget for the quadrature-point closures.
///
/// The J and K terms both allocate large per-point intermediates INSIDE the
/// `par_iter` over quadrature points, so every resident buffer is multiplied by
/// the number of rayon worker threads. We derive a per-task ceiling by dividing
/// the process-wide 3-index budget (`FERRIC_ERI3_BUDGET_GB`, else unlimited) by
/// the active thread count, then reserve a fraction of that for the blocked
/// intermediates. Falls back to a fixed default when the budget is unlimited so
/// the blocking still bounds a single point's footprint.
fn per_task_budget_bytes() -> usize {
    // Default per-task ceiling when no explicit process budget is set. 512 MiB
    // is generous for the (naux, nbas²)-panel intermediates yet still forces
    // blocking at the nbf=900/naux=2200 scale (a full-width buffer is 14 GB).
    const DEFAULT_PER_TASK: usize = 512 * 1024 * 1024;
    let threads = rayon::current_num_threads().max(1);
    match ferric_integrals::three_index_source::env_budget_bytes() {
        usize::MAX => DEFAULT_PER_TASK,
        total => (total / threads).max(64 * 1024 * 1024),
    }
}

/// Row-sparse representation of a B^P slice (nbas × nbas matrix).
///
/// For each row μ, stores only the column indices and values with |B^P_{μν}| > threshold.
/// Enables O(nnz) sparse-dense matrix products M^P = B^P @ P(t) when B^P is sparse.
struct SparseBSlice {
    /// For each row μ: (col_indices, values)
    rows: Vec<(Vec<u16>, Vec<f64>)>,
    nbas: usize,
}

impl SparseBSlice {
    fn from_dense(b: &Array2<f64>, thresh: f64) -> Self {
        let nbas = b.nrows();
        let rows = (0..nbas).map(|mu| {
            let row = b.row(mu);
            let mut cols = Vec::new();
            let mut vals = Vec::new();
            for (nu, &v) in row.iter().enumerate() {
                if v.abs() > thresh {
                    cols.push(nu as u16);
                    vals.push(v);
                }
            }
            (cols, vals)
        }).collect();
        Self { rows, nbas }
    }

    /// Compute rows [m0, m1) of M = self @ rhs (dense) into `out`, which must be
    /// exactly (m1-m0)*nbas long, row-major over (μ∈[m0,m1), ν).
    /// Output M[μ,ν] = Σ_{σ∈nnz(row μ)} B^P_{μσ} rhs_{σν}.
    ///
    /// `out` is caller-provided and MUST be zeroed for the columns this fills
    /// (this method overwrites, not accumulates) — we zero it here so panel
    /// buffers can be reused across μ-panels without a separate clear.
    fn mat_mul_flat_rows(&self, rhs: &Array2<f64>, m0: usize, m1: usize, out: &mut [f64]) {
        let nbas = self.nbas;
        debug_assert_eq!(out.len(), (m1 - m0) * nbas);
        out.fill(0.0);
        for mu in m0..m1 {
            let (cols, vals) = &self.rows[mu];
            let out_row = &mut out[(mu - m0) * nbas..(mu - m0 + 1) * nbas];
            for (&nu, &b_val) in cols.iter().zip(vals.iter()) {
                let rhs_row = rhs.row(nu as usize);
                for (&r, o) in rhs_row.iter().zip(out_row.iter_mut()) {
                    *o += b_val * r;
                }
            }
        }
    }

    /// Compute rows [m0, m1) of (rhs · Bᵀ) into `out` ((m1-m0)*nbas long, row-major):
    /// out[μ,ν] = Σ_{σ∈nnz(B row ν)} rhs_{μσ} B_{νσ}.
    ///
    /// With a SYMMETRIC `rhs` (the pseudo-densities P(t)/Q(t) are symmetric by
    /// construction), this equals the transposed slab (B·rhs)ᵀ[μ∈[m0,m1), ν] —
    /// i.e. N^Q[ν, μ] laid out with μ as the panel row, which is exactly the
    /// pairing the J-term trace Tr(M^P·N^Q) = Σ_μν M^P_μν N^Q_νμ needs.
    fn mat_mul_t_rows(&self, rhs: &Array2<f64>, m0: usize, m1: usize, out: &mut [f64]) {
        let nbas = self.nbas;
        debug_assert_eq!(out.len(), (m1 - m0) * nbas);
        for mu in m0..m1 {
            let rhs_row = rhs.row(mu);
            let rhs_row = rhs_row.as_slice().expect("pseudo-density rows are contiguous");
            let out_row = &mut out[(mu - m0) * nbas..(mu - m0 + 1) * nbas];
            for (nu, o) in out_row.iter_mut().enumerate() {
                let (cols, vals) = &self.rows[nu];
                let mut acc = 0.0f64;
                for (&sigma, &v) in cols.iter().zip(vals.iter()) {
                    acc += rhs_row[sigma as usize] * v;
                }
                *o = acc;
            }
        }
    }
}

pub struct LaplaceMp2Result {
    pub total_energy: f64,
    pub mp2_corr: f64,
    pub e_os: f64,
    pub e_ss: f64,
}

/// Laplace-MP2 exchange (K) trace for one quadrature point.
///
/// Given the τ-weighted RI amplitudes `b_t` in MO basis, shaped
/// `(naux, nocc*nvir)` row-major so that `b_t[P, i*nvir + a] = B^P_ia(t)`,
/// compute
/// ```text
/// e_exch = Σ_{P,i,Q,j} G_PQ[i,j] · G_PQ[j,i],   G_PQ[i,j] = Σ_a B^P_ia B^Q_ja.
/// ```
///
/// Reshaping `b_t` to `X[(P,i), a]` of shape `(naux*nocc) × nvir` is free
/// (the storage is already contiguous in that order), so `Y = X Xᵀ` gives
/// `Y[(P,i),(Q,j)] = G_PQ[i,j]` in a SINGLE wide GEMM per row-block. We block
/// the leading `(P,i)` index so the intermediate `Y_blk` stays bounded while the
/// GEMM contraction dimension (`nvir`) and inner dimension (`naux*nocc`) stay
/// wide — one large GEMM per block instead of `naux²` tiny `nocc×nocc` GEMMs.
///
/// The contraction `Σ Y[Pi,Qj]·Y[Pj,Qi]` swaps the occupied indices *within*
/// each `(P,Q)` block, so it is evaluated per `nocc×nocc` sub-block of `Y`.
fn laplace_exchange_energy(
    b_t: &Array2<f64>,
    naux: usize,
    nocc: usize,
    nvir: usize,
    budget_bytes: usize,
) -> f64 {
    // X[(P,i), a]: same buffer as b_t, reinterpreted (naux*nocc) × nvir.
    let x = b_t
        .view()
        .into_shape_with_order((naux * nocc, nvir))
        .expect("b_t is contiguous (naux, nocc*nvir)");
    let rows = naux * nocc;

    // The intermediate `y_blk = x_blk.dot(&x.t())` has shape
    // (block_p*nocc) × (naux*nocc), so its true footprint is
    //   block_p * nocc * (naux*nocc) * 8  =  block_rows × (naux·nocc) × 8.
    // Choose block_p (whole P's — the occupied-swap contraction needs the full
    // (P,Q) nocc×nocc sub-block inside one Y_blk) so that footprint fits the
    // per-task budget, clamped to [1, naux].
    // Bytes contributed to y_blk by one P of the block = nocc rows × ncol cols × 8.
    let row_bytes = rows.max(1) * nocc.max(1) * 8;
    let block_p = (budget_bytes / row_bytes.max(1)).clamp(1, naux.max(1));

    (0..naux)
        .step_by(block_p)
        .map(|p0| {
            let p_end = (p0 + block_p).min(naux);
            let x_blk = x.slice(ndarray::s![p0 * nocc..p_end * nocc, ..]);
            // Y_blk[(P,i),(Q,j)] = G_PQ[i,j], shape (block*nocc) × (naux*nocc):
            // one wide GEMM with inner dimension nvir.
            let y_blk = x_blk.dot(&x.t());

            // Contract: for each P in this block and every Q, sum over the
            // nocc×nocc sub-block G_PQ[i,j]*G_PQ[j,i].
            //
            // Layout: ndarray's `dot` allocates its output COLUMN-major when
            // both operands have row-stride 1 (impl_linalg.rs `set_f`), which
            // happens here exactly when nvir == 1 (x_blk strides [nvir, 1],
            // x.t() row-stride always 1). The flat-index contraction below
            // assumes C order, so normalize first — a borrow (free) in the
            // common C-order case, a copy only for the tiny nvir == 1 case.
            let mut e_blk = 0.0f64;
            let y_std = y_blk.as_standard_layout();
            let ys = y_std.as_slice().expect("as_standard_layout is C-contiguous");
            let ncol = naux * nocc;
            for pi_local in 0..(p_end - p0) {
                for q in 0..naux {
                    // G_PQ[i,j] = y_blk[pi_local*nocc + i, q*nocc + j]
                    // G_PQ[j,i] = y_blk[pi_local*nocc + j, q*nocc + i]
                    let base_i = pi_local * nocc; // block-local row base for occ i
                    let col_base = q * nocc;
                    for i in 0..nocc {
                        let row_i = (base_i + i) * ncol + col_base;
                        for j in 0..nocc {
                            let g_ij = ys[row_i + j];
                            let g_ji = ys[(base_i + j) * ncol + col_base + i];
                            e_blk += g_ij * g_ji;
                        }
                    }
                }
            }
            e_blk
        })
        .sum()
}

pub fn laplace_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    n_quad: usize,
    frozen_core: usize,
) -> Result<LaplaceMp2Result, FerricError> {
    let mut laplace = LaplaceMp2::new(n_quad);
    let (mp2_corr, e_os, e_ss) = laplace.compute_ao(mol, obs, dfbs, op, rhf, frozen_core, None)?;
    Ok(LaplaceMp2Result {
        total_energy: rhf.energy + mp2_corr,
        mp2_corr,
        e_os,
        e_ss,
    })
}

/// Local MP2 with Boys-localized occupied orbitals and spatial domains.
///
/// `domain_cutoff_bohr`: radius around each Boys center that defines its AO domain.
/// Orbitals whose centers are far apart contribute zero to P(t) between their domains,
/// giving linear-scaling pseudo-densities for large molecules.
// Each arg is a distinct input (system, two bases, operator, reference, and
// three independent Laplace/locality knobs) — no natural grouping.
#[allow(clippy::too_many_arguments)]
pub fn laplace_lmp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    n_quad: usize,
    frozen_core: usize,
    domain_cutoff_bohr: f64,
) -> Result<LaplaceMp2Result, FerricError> {
    let mut laplace = LaplaceMp2::new(n_quad);
    let (mp2_corr, e_os, e_ss) = laplace.compute_ao(
        mol, obs, dfbs, op, rhf, frozen_core, Some(domain_cutoff_bohr),
    )?;
    Ok(LaplaceMp2Result {
        total_energy: rhf.energy + mp2_corr,
        mp2_corr,
        e_os,
        e_ss,
    })
}

/// Apply the τ-weighting `B^P_ia(t) = B^P_ia · exp(-t(ε_a - ε_i)/2)` to a copy
/// of the hoisted MO amplitudes.
///
/// `b_mo_flat` is `(naux, nocc*nvir)` row-major (`[P, i*nvir + a]`) and stays
/// constant across quadrature points — only the diagonal scaling depends on `t`.
/// The factor separates as `exp(t ε_i/2) · exp(-t ε_a/2)`; we precompute the
/// per-occ and per-vir exponentials (nocc + nvir `exp` calls) and scale by their
/// outer product instead of one `exp` per (i,a) element.
fn weighted_b_mo(
    b_mo_flat: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    nocc: usize,
    nvir: usize,
    t: f64,
) -> Array2<f64> {
    // Per-point scalings: occ_e[i] = exp(t ε_i/2), vir_e[a] = exp(-t ε_a/2).
    let occ_e: Vec<f64> = eps_occ.iter().map(|&e| (0.5 * t * e).exp()).collect();
    let vir_e: Vec<f64> = eps_vir.iter().map(|&e| (-0.5 * t * e).exp()).collect();
    let mut b_t = b_mo_flat.clone();
    for mut row in b_t.rows_mut() {
        let r = row.as_slice_mut().unwrap();
        for i in 0..nocc {
            let oi = occ_e[i];
            let base = i * nvir;
            for a in 0..nvir {
                r[base + a] *= oi * vir_e[a];
            }
        }
    }
    b_t
}

/// Laplace-transform MP2 energy builder.
pub struct LaplaceMp2 {
    pub n_quad: usize,
    pub points: Vec<f64>,
    pub weights: Vec<f64>,
}

use rayon::prelude::*;

impl LaplaceMp2 {
    /// Create a Laplace-MP2 builder from orbital energies.
    ///
    /// Selects minimax quadrature points for the range [ymin, ymax] where
    /// ymin = 2*(LUMO - HOMO) and ymax = 2*(eps_max - eps_min).
    /// Points are from Takatsuka, Ten-no, Hackbusch (JCP 129, 044112, 2008)
    /// via the Helmich-Paris laplace-minimax library.
    pub fn new(n_quad: usize) -> Self {
        // Default: will be reinitialized by compute() using actual orbital energies.
        LaplaceMp2 { n_quad, points: vec![], weights: vec![] }
    }

    /// Initialize quadrature for orbital energy range [ymin, ymax].
    ///
    /// ymin = 2*(LUMO - HOMO), ymax = 2*(eps_max_vir - eps_min_occ).
    /// The exponents (t) and weights (w) approximate 1/x on [ymin, ymax] as
    /// 1/x ≈ Σ_k w_k exp(-t_k x). Points are scaled: t_actual = t_table / ymin,
    /// w_actual = w_table / ymin.
    fn init_quadrature(&mut self, ymin: f64, ymax: f64) {
        let q = LaplaceQuadrature::new(self.n_quad, ymin, ymax);
        self.points = q.points;
        self.weights = q.weights;
    }

    /// Compute the MP2 energy using MO-based RI-Laplace transform.
    ///
    /// This method transforms the 3-center integrals to the MO basis and is
    /// useful for verifying the accuracy of the Laplace quadrature against
    /// canonical RI-MP2 results.
    pub fn compute_mo(
        &mut self,
        mol: &Molecule,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
        rhf: &ScfResult,
        frozen_core: usize,
    ) -> Result<f64, FerricError> {
        let nbas = obs.nbasis();
        let nocc_total = mol.nelec() as usize / 2;
        let nocc = active_occ(nocc_total, frozen_core)?;
        let nvir = nbas - nocc_total;
        let naux = dfbs.nbasis();

        let eps = rhf.eps_r();
        let nmo = eps.len();
        let ymin = 2.0 * (eps[nocc_total] - eps[nocc_total - 1]);
        let ymax = 2.0 * (eps[nmo - 1] - eps[0]);
        self.init_quadrature(ymin, ymax);

        let c = rhf.mos_r();
        let c_occ = c.slice(ndarray::s![.., frozen_core..nocc_total]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

        // 1. Get (P|ia) RI amplitudes. The raw AO 3-index tensor is generated in
        // budget-sized aux blocks and transformed to MO immediately
        // (bit-identical to the dense eri3_tensor + transform_3center_ov path),
        // so the naux·nbas² AO tensor is never materialized here.
        let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
        let v_inv_sqrt = crate::rimp2::cholesky_inverse_sqrt(&v2c)?;
        let eri3_mo = crate::rimp2::eri3_mo_ov_blocked(
            op, obs, dfbs, &c_occ, &c_vir, crate::rimp2::eri3_budget_bytes(None),
        )?;
        let b_flat = v_inv_sqrt.dot(&eri3_mo.into_shape_with_order((naux, nocc * nvir)).unwrap());

        // Hoist per-orbital energies: only the diagonal τ-scaling depends on the
        // quadrature point, `b_flat` (geometry/basis) is constant.
        let occ_scale: Vec<f64> = (0..nocc).map(|i| eps[frozen_core + i]).collect();
        let vir_scale: Vec<f64> = (0..nvir).map(|a| eps[nocc_total + a]).collect();
        let k_budget = per_task_budget_bytes();

        // 2. Parallel quadrature over points
        let e_corr: f64 = self.points.par_iter().zip(self.weights.par_iter()).map(|(&t, &w)| {
            // Weighted amplitudes: B_ia(t) = B_ia * exp(-t * (eps_a - eps_i) / 2)
            let b_t = weighted_b_mo(&b_flat, &occ_scale, &vir_scale, nocc, nvir, t);

            // J_PQ = sum_{ia} B_ia^P B_ia^Q
            let j_mat = b_t.dot(&b_t.t());
            let e_coul = j_mat.iter().map(|&x| x * x).sum::<f64>();

            // Exchange: single blocked wide GEMM instead of the dense (naux·nocc)² Gram.
            let e_exch = laplace_exchange_energy(&b_t, naux, nocc, nvir, k_budget);

            -w * (2.0 * e_coul - e_exch)
        }).sum();

        Ok(e_corr)
    }

    /// Compute the MP2 energy using a hybrid AO/MO Laplace-transform approach.
    ///
    /// - J term: AO pseudo-density formulation. With `domain_cutoff_bohr = Some(r)`,
    ///   Boys-localizes the occupied MOs and restricts P(t) to spatial domains,
    ///   enabling linear-scaling pseudo-densities for large molecules.
    /// - K term: MO basis, blocked wide-GEMM Gram contraction — O(naux² × nocc² × nvir)
    ///   FLOPs but executed as (naux·nocc)×nvir GEMM blocks with a ~64 MiB intermediate cap,
    ///   with the un-weighted (P|ia) amplitudes hoisted out of the quadrature loop.
    // Distinct inputs (system, two bases, operator, reference, and two locality
    // knobs); nothing to bundle beyond what `self` already holds.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_ao(
        &mut self,
        mol: &Molecule,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
        rhf: &ScfResult,
        frozen_core: usize,
        domain_cutoff_bohr: Option<f64>,
    ) -> Result<(f64, f64, f64), FerricError> {
        let nbas = obs.nbasis();
        let nocc_total = mol.nelec() as usize / 2;
        let nocc = active_occ(nocc_total, frozen_core)?;
        let nvir = nbas - nocc_total;
        let naux = dfbs.nbasis();

        let eps = rhf.eps_r();
        let nmo = eps.len();
        let ymin = 2.0 * (eps[nocc_total] - eps[nocc_total - 1]);
        let ymax = 2.0 * (eps[nmo - 1] - eps[0]);
        self.init_quadrature(ymin, ymax);

        // Build RI-fitted 3-center integrals: b_ao[P, μ, ν] = Σ_Q V^{-1/2}_{PQ} (Q|μν)
        //
        // NOTE: the AO J term random-accesses `b_sparse[P]` for every P at every
        // quadrature point (points are the parallel axis), so the dressed 3-index
        // tensor must be held resident — a streaming ThreeIndexSource would force
        // re-reading the whole tensor per point or serializing the quadrature.
        // We therefore keep it dense and fail fast if it would not fit the budget,
        // rather than silently allocating a ~14 GB tensor per process. (The former
        // per-point 28.5 GB J-term buffers are what the μ-panel blocking below
        // eliminates; this resident tensor is the irreducible O(naux·nbas²) cost.)
        {
            let dense_bytes = naux.saturating_mul(nbas).saturating_mul(nbas).saturating_mul(8);
            let budget = ferric_integrals::three_index_source::env_budget_bytes();
            if dense_bytes > budget {
                return Err(FerricError::General(format!(
                    "laplace-MP2 AO path needs a resident dressed 3-index tensor of \
                     {:.2} GB (naux={naux}, nbas={nbas}) but FERRIC_ERI3_BUDGET_GB caps \
                     it at {:.2} GB. Raise the budget or use compute_mo.",
                    dense_bytes as f64 / 1e9, budget as f64 / 1e9,
                )));
            }
        }
        let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
        let v_inv_sqrt = crate::rimp2::cholesky_inverse_sqrt(&v2c)?;
        let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;
        let eri3_flat = eri3_ao.into_shape_with_order((naux, nbas * nbas)).unwrap();
        let b_flat_ao = v_inv_sqrt.dot(&eri3_flat);
        let b_ao = b_flat_ao.into_shape_with_order((naux, nbas, nbas)).unwrap();

        let c = rhf.mos_r();
        let c_occ = c.slice(ndarray::s![.., frozen_core..nocc_total]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

        // Boys localization + domain construction (when requested).
        // Boys centers define orbital domains for spatial screening of P(t) and Q(t).
        // The pseudo-densities still use canonical MO coefficients and orbital energies —
        // the Boys rotation is unitary so the AO P(t) is invariant to it, but here we
        // use the domains to decide which (μ,ν) pairs to include (AO sparsity).
        let eps_occ: Vec<f64> = (frozen_core..nocc_total).map(|k| eps[k]).collect();
        let boys_domains = if let Some(cutoff) = domain_cutoff_bohr {
            let dip = ferric_integrals::oneelectron::dipole(obs, [0.0, 0.0, 0.0]);
            let boys = boys_localize(&c_occ, &dip, 200);
            let shell_centers = obs.shell_centers();
            let nshells = obs.nshells();
            let mut offs = vec![0usize; nshells + 1];
            for s in 0..nshells {
                offs[s + 1] = offs[s] + obs.shell_dims()[s];
            }
            let domains = build_domains(&boys.centers, &shell_centers, &offs, cutoff);
            Some(domains)
        } else {
            None
        };

        // Build sparse B^P slices for AO J computation.
        // Threshold 1e-12 retains all numerically significant RI integrals.
        // With localized orbitals and diffuse bases this will be much sparser.
        // Sparse B^P representation: threshold small elements of each B^P slice.
        // sto-3g decane: ~32% fill at 1e-12 (14% at 1e-6 with <1e-6 Ha error).
        // True linear scaling requires sparse P(t)/Q(t), which needs localized MOs.
        let b_sparse: Vec<SparseBSlice> = (0..naux)
            .map(|p| SparseBSlice::from_dense(&b_ao.slice(ndarray::s![p, .., ..]).to_owned(), 1e-12))
            .collect();

        // MO-basis integrals for K: b_mo[P, i*nvir+a] = (P|ia). Hoisted — constant
        // across quadrature points; only the diagonal τ-scaling depends on t.
        let eri3_mo = crate::mo_transform::transform_3center_ov(&b_ao, &c_occ, &c_vir);
        // b_ao's last use is the MO transform above; free the dense (naux, nbas²)
        // tensor now instead of holding it across the whole quadrature loop next
        // to b_sparse (which duplicates its significant entries).
        drop(b_ao);
        let b_mo_flat = eri3_mo.into_shape_with_order((naux, nocc * nvir)).unwrap();
        let occ_scale: Vec<f64> = (0..nocc).map(|i| eps[frozen_core + i]).collect();
        let vir_scale: Vec<f64> = (0..nvir).map(|a| eps[nocc_total + a]).collect();

        // Per-task memory ceiling. Both the J-term panels and the K-term Gram
        // are allocated INSIDE the per-point par_iter, so every buffer is
        // multiplied by the active rayon thread count — the budget is already
        // divided by that count in per_task_budget_bytes().
        let task_budget = per_task_budget_bytes();
        // J-term panel width over the μ (leading AO) axis. Each open μ-row of the
        // M and N panels is (naux · nbas · 8) bytes; hold two panels (M, N), so
        //   block_mu · naux · nbas · 8 · 2 ≤ task_budget.
        // Blocking over μ keeps the P and Q (aux) axes full so the Gram
        //   J[P,Q] = Σ_μν M^P_μν N^Q_μν = Σ_{μ-panel} Σ_ν M_panel N_panel^T
        // accumulates exactly across panels. Clamp to [1, nbas].
        let mu_row_bytes = naux.max(1) * nbas.max(1) * 8 * 2;
        let block_mu = (task_budget / mu_row_bytes.max(1)).clamp(1, nbas.max(1));

        // Parallel over quadrature points — each point is independent.
        // Inner BLAS calls use multithreaded DGEMM; no nested rayon.
        let (e_os, e_ss): (f64, f64) = self.points.par_iter().zip(self.weights.par_iter()).map(|(&t, &w)| {
            // --- J term in AO basis ---
            // Build pseudo-densities: sparse (domain-restricted) when Boys-localized,
            // dense (canonical) otherwise.
            let (pt, qt) = if let Some(ref domains) = boys_domains {
                let pt = build_pseudo_density_occ_sparse(&c_occ, &eps_occ, t, domains);
                let qt = build_pseudo_density_vir_sparse(&c_vir, eps, t, nocc_total, domains);
                (pt, qt)
            } else {
                let pt = build_pseudo_density_occ(c, eps, t, nocc, frozen_core);
                let qt = build_pseudo_density_vir(c, eps, t, nvir, nocc_total);
                (pt, qt)
            };

            // J[P,Q] = Tr(M^P·N^Q) = Σ_μ Σ_ν M^P_μν N^Q_νμ, with M^P = B^P·P(t),
            // N^Q = B^Q·Q(t). Block the μ axis: for each μ-panel [m0,m1), build the
            // (naux, pw·nbas) slabs
            //   m_panel[P, μν] = M^P[μ, ν]          (rows of B^P·P)
            //   n_panel[Q, μν] = N^Q[ν, μ]          (transposed slab, via Q(t)·B^Qᵀ
            //                                        with Q(t) symmetric)
            // and accumulate their Gram into j_mat — the shared μν column index then
            // realizes exactly the μ↔ν-swapped trace pairing of the old full-width
            // n_t_buf packing. This bounds the per-task footprint to
            // block_mu·naux·nbas·8·2 instead of the old (naux, nbas²) full-width
            // buffers (naux·nbas²·8 each — ~14 GB at nbf=900/naux=2200, ×2 ×threads
            // inside the par_iter).
            let mut j_mat = Array2::<f64>::zeros((naux, naux));
            let mut m0 = 0;
            while m0 < nbas {
                let m1 = (m0 + block_mu).min(nbas);
                let pw = m1 - m0; // panel width in μ rows
                // Panel buffers sized exactly pw·nbas so each Array2 row is
                // contiguous (needed for as_slice_mut in the sparse fill methods).
                let mut m_panel = Array2::<f64>::zeros((naux, pw * nbas));
                let mut n_panel = Array2::<f64>::zeros((naux, pw * nbas));
                for p in 0..naux {
                    // Sparse B^P times the dense pseudo-densities, restricted to the
                    // μ rows in this panel (avoids materializing the full nbas² row).
                    b_sparse[p].mat_mul_flat_rows(&pt, m0, m1,
                        m_panel.row_mut(p).as_slice_mut().unwrap());
                    b_sparse[p].mat_mul_t_rows(&qt, m0, m1,
                        n_panel.row_mut(p).as_slice_mut().unwrap());
                }
                // j_mat += M_panel · N_panelᵀ: Σ_{μ∈panel,ν} M^P[μ,ν]·N^Q[ν,μ]
                j_mat += &m_panel.dot(&n_panel.t());
                m0 = m1;
            }
            let e_os_k: f64 = j_mat.iter().map(|&x| x * x).sum();

            // --- K term in MO basis ---
            // Apply Laplace weights B_ia(t) = B_ia·exp(-t(ε_a-ε_i)/2) to the
            // hoisted amplitudes, then contract via one blocked wide GEMM
            // (Y = X Xᵀ, X = b_t viewed as (naux·nocc)×nvir) instead of the
            // naux² tiny nocc×nocc GEMMs the previous loop used.
            let b_t = weighted_b_mo(&b_mo_flat, &occ_scale, &vir_scale, nocc, nvir, t);
            let e_exch_k = laplace_exchange_energy(&b_t, naux, nocc, nvir, task_budget);

            let e_ss_k = e_os_k - e_exch_k;
            (-w * e_os_k, -w * e_ss_k)
        }).reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

        Ok((e_os + e_ss, e_os, e_ss))
    }
}

/// Build the occupied pseudo-density P(t)_{μν} = Σ_i C_{μi} exp(t ε_i) C_{νi}.
pub fn build_pseudo_density_occ(
    c: &Array2<f64>,
    eps: &[f64],
    t: f64,
    nocc: usize,
    first_occ: usize,
) -> Array2<f64> {
    let n = c.nrows();
    let mut p = Array2::zeros((n, n));
    for i in 0..nocc {
        let factor = (t * eps[first_occ + i]).exp();
        for mu in 0..n {
            let c_mu_i = c[(mu, first_occ + i)] * factor;
            for nu in 0..n {
                p[(mu, nu)] += c_mu_i * c[(nu, first_occ + i)];
            }
        }
    }
    p
}

/// Build the virtual pseudo-density Q(t)_{μν} = Σ_a C_{μa} exp(-t ε_a) C_{νa}.
pub fn build_pseudo_density_vir(
    c: &Array2<f64>,
    eps: &[f64],
    t: f64,
    nvir: usize,
    nocc_total: usize,
) -> Array2<f64> {
    let n = c.nrows();
    let mut q = Array2::zeros((n, n));
    for a in 0..nvir {
        let factor = (-t * eps[nocc_total + a]).exp();
        for mu in 0..n {
            let c_mu_a = c[(mu, nocc_total + a)] * factor;
            for nu in 0..n {
                q[(mu, nu)] += c_mu_a * c[(nu, nocc_total + a)];
            }
        }
    }
    q
}


#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use ferric_integrals::operator::Operator;

    #[test]
    fn test_laplace_mp2_mo_vs_ao() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        ).unwrap();

        let mut laplace = LaplaceMp2::new(3);
        let e_mo = laplace.compute_mo(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();
        let (e_ao, _, _) = laplace.compute_ao(&mol, &obs, &dfbs, op, &rhf, 0, None).unwrap();

        eprintln!("Laplace RI-MP2 MO: {e_mo:.10}");
        eprintln!("Laplace RI-MP2 AO: {e_ao:.10}");

        assert!((e_mo - e_ao).abs() < 1e-8,
            "MO and AO Laplace methods should give identical results: {e_mo} vs {e_ao}");
    }

    /// nvir = 1 regression: with a single virtual orbital (H2/STO-3G) BOTH
    /// operands of the exchange-energy GEMM have row-stride 1, and ndarray's
    /// `dot` then allocates its output in COLUMN-major order
    /// (`(m, n).set_f(lhs_s0 == 1 && rhs_s0 == 1)` in impl_linalg.rs). The
    /// flat-index contraction assumed C order and `as_slice()` panicked.
    #[test]
    fn test_laplace_mp2_h2_sto3g_single_virtual() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() },
        ).unwrap();

        let mut laplace = LaplaceMp2::new(7);
        let e_laplace = laplace.compute_mo(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();

        let ri = crate::rimp2::ri_mp2(
            &mol, &obs, &dfbs, op, &rhf, &crate::rimp2::RiMp2Config::default(),
        ).unwrap();
        eprintln!("Laplace: {e_laplace:.10}  RI-MP2: {:.10}", ri.mp2_corr);
        assert!(
            (e_laplace - ri.mp2_corr).abs() < 1e-5,
            "Laplace-MP2 must match RI-MP2 on a single-virtual system: {e_laplace} vs {}",
            ri.mp2_corr
        );
    }

    #[test]
    fn test_laplace_mp2_water_ccpvdz() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        ).unwrap();

        let mut laplace = LaplaceMp2::new(7);
        let e_mo = laplace.compute_mo(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();
        let (e_ao, _, _) = laplace.compute_ao(&mol, &obs, &dfbs, op, &rhf, 0, None).unwrap();

        eprintln!("H2O Laplace RI-MP2 MO: {e_mo:.10}");
        eprintln!("H2O Laplace RI-MP2 AO: {e_ao:.10}");

        assert!((e_mo - e_ao).abs() < 1e-8);

        // Reference RI-MP2 for H2O/cc-pVDZ is -0.20403347
        let ri_mp2_ref = -0.20403347;
        assert!((e_mo - ri_mp2_ref).abs() < 1e-3,
            "Laplace RI-MP2 ({e_mo:.6}) should be close to RI-MP2 ({ri_mp2_ref:.6})");
    }

    /// With a large domain cutoff (whole molecule), Boys LMP2 must reproduce
    /// the canonical Laplace result.  The Boys rotation is unitary so the
    /// energy is invariant; failure here means the pseudo-density build or
    /// domain masking is broken.
    #[test]
    fn test_lmp2_large_cutoff_matches_canonical() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() },
        ).unwrap();

        let mut laplace = LaplaceMp2::new(7);
        let (e_canonical, _, _) = laplace.compute_ao(&mol, &obs, &dfbs, op, &rhf, 0, None).unwrap();
        // 20 Bohr (~10 Å) encompasses water entirely — Boys domains include all AOs
        let (e_lmp2, _, _) = laplace.compute_ao(&mol, &obs, &dfbs, op, &rhf, 0, Some(20.0)).unwrap();

        eprintln!("H2O Laplace canonical: {e_canonical:.10}");
        eprintln!("H2O Laplace LMP2 (20 Bohr cutoff): {e_lmp2:.10}");

        assert!((e_canonical - e_lmp2).abs() < 1e-6,
            "LMP2 with full-molecule domain ({e_lmp2:.8}) should match canonical ({e_canonical:.8})");
    }

    #[test]
    fn test_laplace_quadrature_convergence() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() },
        ).unwrap();

        // 3 points vs 5 points vs 7 points
        let mut lap3 = LaplaceMp2::new(3);
        let mut lap5 = LaplaceMp2::new(5);
        let mut lap7 = LaplaceMp2::new(7);

        let e3 = lap3.compute_mo(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();
        let e5 = lap5.compute_mo(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();
        let e7 = lap7.compute_mo(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();

        eprintln!("H2/cc-pVDZ Laplace MP2: k=3: {e3:.10}, k=5: {e5:.10}, k=7: {e7:.10}");

        // They should all be within ~0.001 Ha of each other for H2
        assert!((e3 - e5).abs() < 1e-3);
        assert!((e5 - e7).abs() < 1e-4);
    }
}
