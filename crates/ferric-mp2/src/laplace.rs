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
/// intermediates.
///
/// # Two defects fixed here
///
/// **1. The caller's budget was discarded.** This function used to call
/// `resolve_budget_bytes(None)`, throwing away `LaplaceMp2::memory_budget_bytes`
/// and re-resolving from the environment. That is the exact bug
/// [`LaplaceMp2::memory_budget_bytes`]'s own doc records fixing for the other
/// paths in this file (`compute_mo`, `compute_ao`, `compute_sos_*` all thread it
/// correctly) — this helper was simply missed in that pass, so a caller passing
/// an explicit ceiling still had its per-task blocking sized against whatever
/// the environment happened to say. It now takes `explicit` and threads it
/// through, and every call site passes `self.memory_budget_bytes`.
///
/// **2. The 64 MiB floor could EXCEED the per-task share, silently.** The old
/// `(total / threads).max(64 MiB)` looks conservative but inverts at small
/// budgets: at a 2 GiB budget with 32 threads the derived share is exactly
/// 64 MiB and the floor binds, so the real aggregate peak is
/// `32 × 64 MiB = 2 GiB` — the entire budget, with the per-thread division
/// defeated and nothing said about it. Below that the floor is strictly larger
/// than the share and the aggregate OVERSHOOTS the budget outright.
///
/// The floor still exists — a per-task ceiling of a few bytes would block the
/// panels down to width 1 and make no progress — but when it binds, that is now
/// stated on stderr instead of silently absorbed. The message names the
/// actionable fix (raise the budget or lower the thread count), matching how the
/// floored-band warnings in `ferric-cc`'s (T) drivers read.
///
/// Note the old `usize::MAX` arm is gone: `resolve_budget_bytes` returns
/// `usize::MAX` only for an explicitly infinite budget, never from the
/// auto-detect or fallback paths, so `DEFAULT_PER_TASK` was effectively dead
/// code. `usize::MAX / threads` is already an enormous per-task ceiling, which
/// is the correct reading of "unlimited" anyway.
fn per_task_budget_bytes(explicit: Option<usize>) -> usize {
    /// Smallest per-task ceiling that still lets the blocked panels make
    /// meaningful progress.
    const MIN_PER_TASK: usize = 64 * 1024 * 1024;
    let threads = rayon::current_num_threads().max(1);
    let total = ferric_core::memory::resolve_budget_bytes(explicit);
    let share = total / threads;
    if share < MIN_PER_TASK {
        eprintln!(
            "ferric WARNING [Laplace-MP2]: the {:.2} GiB memory budget split over {threads} \
             rayon threads gives {:.0} MiB per task, below the {:.0} MiB floor the blocked \
             quadrature panels need. Using the floor, so the aggregate peak may reach {:.2} GiB \
             — ABOVE the budget. Raise [memory] budget_gb / FERRIC_MEM_BUDGET_GB, or lower \
             RAYON_NUM_THREADS.",
            total as f64 / (1024.0 * 1024.0 * 1024.0),
            share as f64 / (1024.0 * 1024.0),
            MIN_PER_TASK as f64 / (1024.0 * 1024.0),
            (MIN_PER_TASK.saturating_mul(threads)) as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        return MIN_PER_TASK;
    }
    share
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

/// Result of a Laplace-transform MP2 energy calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct LaplaceMp2Result {
    pub total_energy: f64,
    pub mp2_corr: f64,
    pub e_os: f64,
    pub e_ss: f64,
}

impl std::fmt::Display for LaplaceMp2Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Laplace-MP2 total: {:.10} Ha (corr: {:.10})",
            self.total_energy, self.mp2_corr)
    }
}

/// Laplace-transform SOS-MP2 (scaled-opposite-spin MP2) configuration.
///
/// Follows [`crate::scs::ScsMp2Config`]'s style: an explicit scaling
/// coefficient, a frozen-core count, and an optional resident-bytes ceiling.
/// There is deliberately no `c_ss` — SOS-MP2 *is* the c_ss = 0 limit, and that
/// is exactly what makes the Laplace denominator factorize (see the module-level
/// derivation on [`LaplaceMp2::compute_sos_mo`]).
#[derive(Debug, Clone)]
pub struct SosMp2Config {
    /// Opposite-spin scaling coefficient. Jung/Head-Gordon (JCP 121, 9793
    /// (2004)) fitted c_os = 1.3 for SOS-MP2; c_os = 1.0 recovers the bare
    /// opposite-spin MP2 energy `ri_mp2_spin_components(..).e_os` (up to the
    /// Laplace quadrature error) and is what the validation tests use.
    pub c_os: f64,
    /// Number of frozen core orbitals.
    pub frozen_core: usize,
    /// Number of minimax Laplace quadrature points. Must be one of {3, 5, 7}:
    /// [`ferric_quadrature::LaplaceQuadrature::new`] hard-errors otherwise
    /// rather than silently capping (see `docs`/TD-QUAD).
    pub n_quad: usize,
    /// Optional resident-bytes ceiling, threaded into the 3-index budget
    /// exactly as [`LaplaceMp2::memory_budget_bytes`]. `None` resolves via
    /// [`ferric_core::memory::resolve_budget_bytes`].
    pub memory_budget_bytes: Option<usize>,
    /// Domain radius (Bohr) for [`SosFormulation::AoSparse`]; ignored otherwise.
    ///
    /// Carried here so the CLI/Python surfaces can accept it as a plain scalar
    /// knob, but the FORMULATION is what selects sparse-vs-dense — the enum
    /// variant carries the cutoff, so an inconsistent pair is rejected at
    /// [`SosFormulation::parse_config_str`] rather than resolved by precedence.
    pub domain_cutoff_bohr: Option<f64>,
}

impl Default for SosMp2Config {
    /// Jung/Head-Gordon SOS-MP2: c_os = 1.3, all electrons correlated,
    /// 7-point minimax quadrature (the tightest supported grid).
    fn default() -> Self {
        Self {
            c_os: 1.3,
            frozen_core: 0,
            n_quad: 7,
            memory_budget_bytes: None,
            domain_cutoff_bohr: None,
        }
    }
}

/// Result of a Laplace SOS-MP2 calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct SosMp2Result {
    /// `E_SCF + sos_corr`.
    pub total_energy: f64,
    /// The scaled correlation energy `c_os * e_os`.
    pub sos_corr: f64,
    /// The UNSCALED Laplace opposite-spin correlation energy. Comparable
    /// directly against `ri_mp2_spin_components(..).e_os`.
    pub e_os: f64,
    /// The `c_os` that was applied, echoed for provenance.
    pub c_os: f64,
    /// Quadrature points actually used.
    pub n_quad: usize,
}

impl std::fmt::Display for SosMp2Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SOS-MP2 total: {:.10} Ha (c_os={:.2}, n_quad={})",
            self.total_energy, self.c_os, self.n_quad)
    }
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

/// Laplace J-term (opposite-spin) energy contribution for ONE quadrature point,
/// evaluated entirely in the AO basis from the pseudo-densities.
///
/// Returns `Σ_PQ J_PQ(t)²` with
/// ```text
/// J_PQ(t) = Σ_μν M^P_μν N^Q_νμ,   M^P = B^P·P(t),   N^Q = B^Q·Q(t).
/// ```
/// This is exactly the AO-basis form of the MO Coulomb trace
/// `Σ_PQ (Σ_ia B^P_ia(t) B^Q_ia(t))²`: substituting the pseudo-densities
/// `P(t)_μν = Σ_i C_μi e^{tε_i} C_νi` and `Q(t)_μν = Σ_a C_μa e^{-tε_a} C_νa`
/// into `Tr(B^P P B^Q Q)` reproduces `Σ_ia B^P_ia e^{-t(ε_a-ε_i)} B^Q_ia`,
/// which is `J_PQ` for the τ-weighted amplitudes. No MO transform is needed,
/// which is the whole point of the AO path.
///
/// The μ axis is blocked to `block_mu` rows so the two panel buffers stay within
/// the per-task budget; the P and Q (aux) axes stay full so the Gram
/// accumulates exactly across panels.
///
/// Extracted verbatim from `compute_ao`'s per-point closure so the SOS AO path
/// reuses the identical algebra rather than restating it.
fn laplace_ao_coulomb_energy(
    b_sparse: &[SparseBSlice],
    pt: &Array2<f64>,
    qt: &Array2<f64>,
    naux: usize,
    nbas: usize,
    block_mu: usize,
) -> f64 {
    let mut j_mat = Array2::<f64>::zeros((naux, naux));
    let mut m0 = 0;
    while m0 < nbas {
        let m1 = (m0 + block_mu).min(nbas);
        let pw = m1 - m0; // panel width in μ rows
        // Panel buffers sized exactly pw·nbas so each Array2 row is contiguous
        // (needed for as_slice_mut in the sparse fill methods).
        let mut m_panel = Array2::<f64>::zeros((naux, pw * nbas));
        let mut n_panel = Array2::<f64>::zeros((naux, pw * nbas));
        for p in 0..naux {
            b_sparse[p].mat_mul_flat_rows(pt, m0, m1, m_panel.row_mut(p).as_slice_mut().unwrap());
            b_sparse[p].mat_mul_t_rows(qt, m0, m1, n_panel.row_mut(p).as_slice_mut().unwrap());
        }
        j_mat += &m_panel.dot(&n_panel.t());
        m0 = m1;
    }
    j_mat.iter().map(|&x| x * x).sum()
}

/// AO-Laplace RI-MP2 via pseudo-density factorization of the denominator.
pub fn laplace_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    n_quad: usize,
    frozen_core: usize,
    memory_budget_bytes: Option<usize>,
) -> Result<LaplaceMp2Result, FerricError> {
    let mut laplace = LaplaceMp2::new(n_quad);
    // Without this the CLI's `laplace-mp2` arm discarded `[memory] budget_gb`
    // entirely: `LaplaceMp2::memory_budget_bytes` exists precisely so callers
    // can set a budget without changing `compute_ao`'s signature (see the
    // field's own doc), but this entry point never set it — so only the
    // sibling `laplace-sos-mp2` arm was ever rewired when that bug was fixed.
    laplace.memory_budget_bytes = memory_budget_bytes;
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
#[derive(Debug, Clone)]
pub struct LaplaceMp2 {
    pub n_quad: usize,
    pub points: Vec<f64>,
    pub weights: Vec<f64>,
    /// Caller's explicit memory ceiling, threaded into the 3-index budget and
    /// the AO-path pre-flight. `None` resolves through
    /// `FERRIC_MEM_BUDGET_GB` -> detected RAM -> 2 GiB, exactly as elsewhere.
    ///
    /// Carried on the struct rather than added to `compute_mo`/`compute_ao`'s
    /// signatures so the several in-crate call sites stay unchanged. Before
    /// this, both paths called `resolve_budget_bytes(None)`, silently
    /// discarding a user's `[memory] budget_gb`.
    pub memory_budget_bytes: Option<usize>,
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
        LaplaceMp2 { n_quad, points: vec![], weights: vec![], memory_budget_bytes: None }
    }

    /// Initialize quadrature for orbital energy range [ymin, ymax].
    ///
    /// ymin = 2*(LUMO - HOMO), ymax = 2*(eps_max_vir - eps_min_occ).
    /// The exponents (t) and weights (w) approximate 1/x on [ymin, ymax] as
    /// 1/x ≈ Σ_k w_k exp(-t_k x). Points are scaled: t_actual = t_table / ymin,
    /// w_actual = w_table / ymin.
    fn init_quadrature(&mut self, ymin: f64, ymax: f64) -> Result<(), FerricError> {
        let q = LaplaceQuadrature::new(self.n_quad, ymin, ymax)?;
        self.points = q.points;
        self.weights = q.weights;
        Ok(())
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
        self.init_quadrature(ymin, ymax)?;

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
            op, obs, dfbs, &c_occ, &c_vir,
            crate::rimp2::eri3_budget_bytes(self.memory_budget_bytes),
        )?;
        let b_flat = v_inv_sqrt.dot(&eri3_mo.into_shape_with_order((naux, nocc * nvir)).unwrap());

        // Hoist per-orbital energies: only the diagonal τ-scaling depends on the
        // quadrature point, `b_flat` (geometry/basis) is constant.
        let occ_scale: Vec<f64> = (0..nocc).map(|i| eps[frozen_core + i]).collect();
        let vir_scale: Vec<f64> = (0..nvir).map(|a| eps[nocc_total + a]).collect();
        // Stage-seam RSS safety net: the AO->MO transform is complete and
        // `b_flat` is resident, before the per-point quadrature fan-out
        // multiplies its scratch by the thread count. Observational only.
        ferric_core::memory::warn_if_rss_over(
            "Laplace-MP2 AO->MO transform complete",
            ferric_core::memory::resolve_budget_bytes(self.memory_budget_bytes),
            1.1,
        );

        let k_budget = per_task_budget_bytes(self.memory_budget_bytes);

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
        self.init_quadrature(ymin, ymax)?;

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
            // TWO tensors of this size are co-resident, not one: the dressing
            // below is `v_inv_sqrt.dot(&eri3_flat)`, so the raw `eri3_flat`
            // input is still live while the `b_flat_ao` output is allocated.
            // Counting a single copy under-reported the peak by ~2x -- the same
            // "guard checks less than the code allocates" defect fixed in the
            // (T) drivers. The naux^2 metric and its inverse-sqrt ride along.
            let dense_bytes = naux.saturating_mul(nbas).saturating_mul(nbas).saturating_mul(8);
            let metric_bytes = naux.saturating_mul(naux).saturating_mul(2).saturating_mul(8);
            let peak_bytes = dense_bytes.saturating_mul(2).saturating_add(metric_bytes);
            // Honor the CALLER's budget. This resolved `None`, which silently
            // discarded an explicit `[memory] budget_gb` in favour of an
            // env/auto-detected value.
            let budget = ferric_core::memory::resolve_budget_bytes(self.memory_budget_bytes);
            if peak_bytes > budget {
                return Err(FerricError::General(format!(
                    "laplace-MP2 AO path needs {:.2} GB resident (naux={naux}, nbas={nbas}: \
                     two co-resident naux*nbas^2 tensors during the metric dressing, plus \
                     the naux^2 metric) but the budget is {:.2} GB. Raise \
                     [memory] budget_gb / FERRIC_MEM_BUDGET_GB, or use compute_mo.",
                    peak_bytes as f64 / 1e9, budget as f64 / 1e9,
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
        //
        // FIXED 2026-07-27. This comment used to read "the pseudo-densities still
        // use canonical MO coefficients and orbital energies — the Boys rotation is
        // unitary so the AO P(t) is invariant to it". That invariance holds for the
        // UNTRUNCATED P(t) and is FALSE once a domain mask is applied: masking
        // canonical orbital i by LOCALIZED orbital i's domain mixes two unrelated
        // index sets. We therefore carry the localized coefficients together with
        // the occupied-block Fock in that same basis, so the orbital index and the
        // domain index refer to the same orbital.
        // NOTE: canonical occupied energies are no longer used by the sparse
        // path — the localized construction carries exp(+t F_loc) instead (see
        // build_pseudo_density_occ_sparse). Kept only where a dense/canonical
        // consumer still needs it.
        let _eps_occ: Vec<f64> = (frozen_core..nocc_total).map(|k| eps[k]).collect();
        let boys_domains = if let Some(cutoff) = domain_cutoff_bohr {
            let dip = ferric_integrals::oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
            let boys = boys_localize(&c_occ, &dip, 200);
            let f_loc = boys.c_loc.t().dot(rhf.fock_r()).dot(&boys.c_loc);
            let shell_centers = obs.shell_centers();
            let nshells = obs.nshells();
            let mut offs = vec![0usize; nshells + 1];
            for s in 0..nshells {
                offs[s + 1] = offs[s] + obs.shell_dims()[s];
            }
            let domains = build_domains(&boys.centers, &shell_centers, &offs, cutoff);
            Some((domains, boys.c_loc, f_loc))
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
        let task_budget = per_task_budget_bytes(self.memory_budget_bytes);
        // J-term panel width over the μ (leading AO) axis. Each open μ-row of the
        // M and N panels is (naux · nbas · 8) bytes; hold two panels (M, N), so
        //   block_mu · naux · nbas · 8 · 2 ≤ task_budget.
        // Blocking over μ keeps the P and Q (aux) axes full so the Gram
        //   J[P,Q] = Σ_μν M^P_μν N^Q_μν = Σ_{μ-panel} Σ_ν M_panel N_panel^T
        // accumulates exactly across panels. Clamp to [1, nbas].
        let mu_row_bytes = naux.max(1) * nbas.max(1) * 8 * 2;
        let block_mu = (task_budget / mu_row_bytes.max(1)).clamp(1, nbas.max(1));

        // Parallel over quadrature points — each point is independent. The
        // J-term panel fill (mat_mul_flat_rows/mat_mul_t_rows) is hand-rolled
        // sparse-times-dense, not BLAS. The two GEMMs that do run per point
        // (m_panel.dot(&n_panel.t()) below, and x_blk.dot(&x.t()) inside
        // laplace_exchange_energy) execute under this rayon map, i.e. nested
        // BLAS-under-rayon — callers must run with OPENBLAS_NUM_THREADS=1 (or
        // an equivalent with_blas_threads(1, ..) scope) per the project's
        // rayon/BLAS threading convention; nothing here raises BLAS threads.
        let (e_os, e_ss): (f64, f64) = self.points.par_iter().zip(self.weights.par_iter()).map(|(&t, &w)| {
            // --- J term in AO basis ---
            // Build pseudo-densities: sparse (domain-restricted) when Boys-localized,
            // dense (canonical) otherwise.
            let (pt, qt) = if let Some((ref domains, ref c_loc, ref f_loc)) = boys_domains {
                let pt = build_pseudo_density_occ_sparse(c_loc, f_loc, t, domains);
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
            let e_os_k =
                laplace_ao_coulomb_energy(&b_sparse, &pt, &qt, naux, nbas, block_mu);

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

    /// Laplace SOS-MP2, MO formulation. Returns the UNSCALED opposite-spin
    /// correlation energy `E_OS` (apply `c_os` at the caller).
    ///
    /// # Derivation
    ///
    /// For a closed-shell RHF reference the opposite-spin MP2 correlation energy
    /// in spatial orbitals is
    /// ```text
    /// E_OS = Σ_ijab (ia|jb)² / D_ij^ab,     D_ij^ab = ε_i + ε_j - ε_a - ε_b < 0.
    /// ```
    /// (the same-spin partner carries the exchange-like `-(ib|ja)` numerator).
    /// Substituting the Laplace identity for the POSITIVE denominator
    /// `x = ε_a + ε_b - ε_i - ε_j = -D`,
    /// ```text
    /// 1/x = ∫₀^∞ e^{-x t} dt ≈ Σ_k w_k e^{-t_k x},
    /// ```
    /// and splitting the exponential over the (i,a) and (j,b) index pairs —
    /// which is possible because `x` is a plain SUM of one-orbital energies:
    /// ```text
    /// e^{-t x} = e^{-t(ε_a-ε_i)} · e^{-t(ε_b-ε_j)}
    ///          = [e^{-t(ε_a-ε_i)/2}]² · [e^{-t(ε_b-ε_j)/2}]²
    /// ```
    /// gives, with the RI amplitude `(ia|jb) = Σ_P B^P_ia B^P_jb` and the
    /// τ-weighted amplitude `B^P_ia(t) = B^P_ia e^{-t(ε_a-ε_i)/2}`,
    /// ```text
    /// E_OS ≈ -Σ_k w_k Σ_PQ [ Σ_ia B^P_ia(t_k) B^Q_ia(t_k) ]²
    ///      = -Σ_k w_k Σ_PQ J_PQ(t_k)².
    /// ```
    ///
    /// **This is the whole reason SOS is the Laplace-friendly variant.** The
    /// (i,a) and (j,b) sums have completely decoupled: `J_PQ(t)` is a single
    /// contraction over one occupied-virtual pair, and the energy is the
    /// Frobenius norm of that `naux × naux` matrix. The same-spin numerator
    /// `(ia|jb)(ib|ja)` instead entangles the two pairs through the swapped
    /// index pattern (the `laplace_exchange_energy` Gram above), and no
    /// factorization of the denominator removes that coupling. Dropping SS is
    /// what leaves a product of independent contractions.
    ///
    /// Cost, in tensor dimensions rather than time: the τ-weighting is
    /// `naux·nocc·nvir` elements, `J = B(t)B(t)ᵀ` is one GEMM of shape
    /// `(naux × nocc·nvir) × (nocc·nvir × naux)`, and the energy is a sum over
    /// `naux²` entries. No `nocc²nvir²` object and no `naux²nocc²` Gram is ever
    /// formed — those belong to the same-spin term this method omits.
    pub fn compute_sos_mo(
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
        self.init_quadrature(ymin, ymax)?;

        let c = rhf.mos_r();
        let c_occ = c.slice(ndarray::s![.., frozen_core..nocc_total]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

        // (P|ia) RI amplitudes, dressed with V^{-1/2} — identical construction
        // to `compute_mo` (blocked aux generation, never materializing the AO
        // naux·nbas² tensor).
        let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
        let v_inv_sqrt = crate::rimp2::cholesky_inverse_sqrt(&v2c)?;
        let eri3_mo = crate::rimp2::eri3_mo_ov_blocked(
            op, obs, dfbs, &c_occ, &c_vir,
            crate::rimp2::eri3_budget_bytes(self.memory_budget_bytes),
        )?;
        let b_flat = v_inv_sqrt.dot(&eri3_mo.into_shape_with_order((naux, nocc * nvir)).unwrap());

        let occ_scale: Vec<f64> = (0..nocc).map(|i| eps[frozen_core + i]).collect();
        let vir_scale: Vec<f64> = (0..nvir).map(|a| eps[nocc_total + a]).collect();

        // Parallel over quadrature points. Only the J (Coulomb) trace survives —
        // the `laplace_exchange_energy` call that `compute_mo` makes here is the
        // same-spin piece and is deliberately absent.
        let e_os: f64 = self
            .points
            .par_iter()
            .zip(self.weights.par_iter())
            .map(|(&t, &w)| {
                let b_t = weighted_b_mo(&b_flat, &occ_scale, &vir_scale, nocc, nvir, t);
                let j_mat = b_t.dot(&b_t.t());
                -w * j_mat.iter().map(|&x| x * x).sum::<f64>()
            })
            .sum();

        Ok(e_os)
    }

    /// Laplace SOS-MP2, AO formulation via pseudo-densities. Returns the
    /// UNSCALED opposite-spin correlation energy `E_OS`.
    ///
    /// Same energy as [`Self::compute_sos_mo`], different algebra. Starting from
    /// the point-wise MO trace `J_PQ(t) = Σ_ia B^P_ia(t) B^Q_ia(t)`, expand the
    /// MO amplitudes back into AOs, `B^P_ia = Σ_μν C_μi B^P_μν C_νa`, and absorb
    /// the τ weights into the coefficient products. The occupied and virtual
    /// sums then close independently into the pseudo-densities
    /// ```text
    /// P(t)_μν = Σ_i C_μi e^{+t ε_i} C_νi        (build_pseudo_density_occ)
    /// Q(t)_μν = Σ_a C_μa e^{-t ε_a} C_νa        (build_pseudo_density_vir)
    /// ```
    /// giving a pure-AO expression with NO occupied or virtual index left:
    /// ```text
    /// J_PQ(t) = Tr[ B^P P(t) B^Q Q(t) ] = Σ_μν (B^P P)_μν (B^Q Q)_νμ
    /// E_OS ≈ -Σ_k w_k Σ_PQ J_PQ(t_k)².
    /// ```
    ///
    /// This is the formulation that can in principle go sub-quintic: `P(t)` and
    /// `Q(t)` become sparse for localized occupied orbitals and local basis
    /// sets, and `B^P` is already stored row-sparse
    /// (`SparseBSlice`) — so the `B^P·P(t)` products are O(nnz) rather than
    /// O(nbas³), and no MO transform appears anywhere in the quadrature loop.
    /// The dense-`P`/`Q` path implemented here is the correctness reference for
    /// that sparse limit; `domain_cutoff_bohr` switches on the Boys-localized,
    /// domain-restricted pseudo-densities the way `compute_ao` does.
    ///
    /// Note the AO path's irreducible resident cost is the dressed
    /// `naux·nbas²` 3-index tensor (the quadrature points are the parallel axis
    /// and each randomly accesses every `B^P`), gated below exactly as in
    /// `compute_ao`.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_sos_ao(
        &mut self,
        mol: &Molecule,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
        rhf: &ScfResult,
        frozen_core: usize,
        domain_cutoff_bohr: Option<f64>,
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
        self.init_quadrature(ymin, ymax)?;

        // Same resident-tensor pre-flight as compute_ao: two co-resident
        // naux·nbas² tensors during the metric dressing plus the naux² metric.
        {
            let dense_bytes = naux.saturating_mul(nbas).saturating_mul(nbas).saturating_mul(8);
            let metric_bytes = naux.saturating_mul(naux).saturating_mul(2).saturating_mul(8);
            let peak_bytes = dense_bytes.saturating_mul(2).saturating_add(metric_bytes);
            let budget = ferric_core::memory::resolve_budget_bytes(self.memory_budget_bytes);
            if peak_bytes > budget {
                return Err(FerricError::General(format!(
                    "laplace-SOS-MP2 AO path needs {:.2} GB resident (naux={naux}, nbas={nbas}: \
                     two co-resident naux*nbas^2 tensors during the metric dressing, plus \
                     the naux^2 metric) but the budget is {:.2} GB. Raise \
                     [memory] budget_gb / FERRIC_MEM_BUDGET_GB, or use compute_sos_mo.",
                    peak_bytes as f64 / 1e9,
                    budget as f64 / 1e9,
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
        // NOTE: canonical occupied energies are no longer used by the sparse
        // path — the localized construction carries exp(+t F_loc) instead (see
        // build_pseudo_density_occ_sparse). Kept only where a dense/canonical
        // consumer still needs it.
        let _eps_occ: Vec<f64> = (frozen_core..nocc_total).map(|k| eps[k]).collect();

        let boys_domains = if let Some(cutoff) = domain_cutoff_bohr {
            let dip = ferric_integrals::oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
            let boys = boys_localize(&c_occ, &dip, 200);
            // See compute_ao: the domain index must refer to the SAME orbital as
            // the coefficient index, so carry the localized coefficients and the
            // occupied-block Fock in that basis rather than canonical ones.
            let f_loc = boys.c_loc.t().dot(rhf.fock_r()).dot(&boys.c_loc);
            let shell_centers = obs.shell_centers();
            let nshells = obs.nshells();
            let mut offs = vec![0usize; nshells + 1];
            for s in 0..nshells {
                offs[s + 1] = offs[s] + obs.shell_dims()[s];
            }
            let domains = build_domains(&boys.centers, &shell_centers, &offs, cutoff);
            Some((domains, boys.c_loc, f_loc))
        } else {
            None
        };

        let b_sparse: Vec<SparseBSlice> = (0..naux)
            .map(|p| SparseBSlice::from_dense(&b_ao.slice(ndarray::s![p, .., ..]).to_owned(), 1e-12))
            .collect();
        // b_ao's only consumer on this path is the sparse conversion above (the
        // SOS energy needs no MO transform at all — unlike compute_ao, which
        // still needs (P|ia) for its exchange term). Free the dense tensor
        // before entering the quadrature loop.
        drop(b_ao);

        let task_budget = per_task_budget_bytes(self.memory_budget_bytes);
        let mu_row_bytes = naux.max(1) * nbas.max(1) * 8 * 2;
        let block_mu = (task_budget / mu_row_bytes.max(1)).clamp(1, nbas.max(1));

        // Parallel over quadrature points. Same nested-BLAS-under-rayon caveat
        // as compute_ao: run with OPENBLAS_NUM_THREADS=1.
        let e_os: f64 = self
            .points
            .par_iter()
            .zip(self.weights.par_iter())
            .map(|(&t, &w)| {
                let (pt, qt) = if let Some((ref domains, ref c_loc, ref f_loc)) = boys_domains {
                    (
                        build_pseudo_density_occ_sparse(c_loc, f_loc, t, domains),
                        build_pseudo_density_vir_sparse(&c_vir, eps, t, nocc_total, domains),
                    )
                } else {
                    (
                        build_pseudo_density_occ(c, eps, t, nocc, frozen_core),
                        build_pseudo_density_vir(c, eps, t, nvir, nocc_total),
                    )
                };
                -w * laplace_ao_coulomb_energy(&b_sparse, &pt, &qt, naux, nbas, block_mu)
            })
            .sum();

        Ok(e_os)
    }
}

/// Which Laplace SOS-MP2 formulation to evaluate.
///
/// Not `Eq`: the `AoSparse` payload is an `f64` cutoff. `PartialEq` is enough
/// for the tests, which compare against explicit variants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SosFormulation {
    /// MO basis: τ-weighted `(P|ia)` amplitudes, `J = B(t)B(t)ᵀ`.
    Mo,
    /// AO basis via occupied/virtual pseudo-densities. No MO transform inside
    /// the quadrature loop; the path that can exploit AO sparsity.
    Ao,
    /// AO basis with Boys-localized, domain-RESTRICTED pseudo-densities.
    ///
    /// The cutoff (Bohr) is the radius around each Boys center defining that
    /// orbital's AO domain. This is the only variant that is not exact: it
    /// DISCARDS contributions from AO pairs outside every domain, so it is a
    /// controlled approximation converging to [`Self::Ao`] as the cutoff grows.
    ///
    /// See [`SosMp2Config::domain_cutoff_bohr`] for why this is exposed as a
    /// separate variant rather than an option on `Ao`.
    AoSparse(f64),
}

impl SosFormulation {
    /// Parse a user-facing config string, STRICTLY.
    ///
    /// `None` resolves to the default (`Mo`). An unrecognized value is a hard
    /// error rather than a silent fallback — same convention as
    /// `QuadratureScheme`/`C6Source`/`Chi0Sparsity::parse_config_str`, and the
    /// reason CLI TOML uses `deny_unknown_fields`: a typo must never quietly
    /// give the user a different method than the one they asked for.
    /// `cutoff` supplies the domain radius for `"ao-sparse"`, which is
    /// meaningless for the other two — passing it with `"mo"`/`"ao"` is an
    /// error rather than a silently-ignored knob.
    pub fn parse_config_str(
        s: Option<&str>,
        cutoff: Option<f64>,
    ) -> Result<Self, FerricError> {
        let form = match s {
            None | Some("mo") => Self::Mo,
            Some("ao") => Self::Ao,
            Some("ao-sparse") => {
                let r = cutoff.ok_or_else(|| {
                    FerricError::General(
                        "SOS-MP2 formulation \"ao-sparse\" requires a domain cutoff \
                         (CLI: [mp2] domain_cutoff_bohr; Python: domain_cutoff_bohr=). \
                         There is no safe default: the right radius depends on the \
                         system and basis, and a wrong one silently changes the energy."
                            .to_string(),
                    )
                })?;
                if !(r.is_finite() && r > 0.0) {
                    return Err(FerricError::General(format!(
                        "SOS-MP2 domain_cutoff_bohr must be finite and > 0 (got {r})"
                    )));
                }
                Self::AoSparse(r)
            }
            Some(other) => {
                return Err(FerricError::General(format!(
                    "unknown SOS-MP2 formulation \"{other}\"; expected \"mo\" (default, \
                     tau-weighted (P|ia) amplitudes), \"ao\" (occupied/virtual \
                     pseudo-densities) or \"ao-sparse\" (domain-restricted AO, \
                     APPROXIMATE — needs domain_cutoff_bohr)"
                )));
            }
        };
        if cutoff.is_some() && !matches!(form, Self::AoSparse(_)) {
            return Err(FerricError::General(format!(
                "domain_cutoff_bohr is only meaningful with formulation \
                 \"ao-sparse\" (got formulation {:?}). Set formulation = \
                 \"ao-sparse\" or drop the cutoff — silently ignoring it would \
                 hand back a different (exact) method than the one configured.",
                s.unwrap_or("mo")
            )));
        }
        Ok(form)
    }
}

/// Laplace SOS-MP2: `E = c_os · E_OS`, with `E_OS` from the Laplace-transformed
/// opposite-spin MP2 expression.
///
/// `formulation` selects the MO or AO algebra; both compute the same quantity
/// and agree to quadrature/RI round-off (this is asserted in the module tests).
/// With `c_os = 1.0` the returned `e_os` reproduces
/// [`crate::rimp2::ri_mp2_spin_components`]'s `e_os` to the Laplace quadrature
/// error, which is the hard internal reference for this method.
pub fn laplace_sos_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    config: &SosMp2Config,
    formulation: SosFormulation,
) -> Result<SosMp2Result, FerricError> {
    if !config.c_os.is_finite() {
        return Err(FerricError::General(format!(
            "laplace_sos_mp2: c_os must be finite (got {})",
            config.c_os
        )));
    }
    let mut laplace = LaplaceMp2::new(config.n_quad);
    laplace.memory_budget_bytes = config.memory_budget_bytes;
    let e_os = match formulation {
        SosFormulation::Mo => {
            laplace.compute_sos_mo(mol, obs, dfbs, op, rhf, config.frozen_core)?
        }
        SosFormulation::Ao => {
            laplace.compute_sos_ao(mol, obs, dfbs, op, rhf, config.frozen_core, None)?
        }
        SosFormulation::AoSparse(cutoff) => laplace.compute_sos_ao(
            mol,
            obs,
            dfbs,
            op,
            rhf,
            config.frozen_core,
            Some(cutoff),
        )?,
    };
    let sos_corr = config.c_os * e_os;
    Ok(SosMp2Result {
        total_energy: rhf.energy + sos_corr,
        sos_corr,
        e_os,
        c_os: config.c_os,
        n_quad: config.n_quad,
    })
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

    /// Widen the AO-Laplace-vs-RI-MP2 cross-check beyond H2/cc-pVDZ (see
    /// `docs/VALIDATION.md`'s "AO-Laplace MP2 (O(N))" row, previously graded
    /// "one molecule"). Methane/cc-pVDZ is a genuinely different test: Td
    /// symmetry, 5 occupied / 4 heavy-plus-H centers, and a much larger
    /// HOMO-LUMO gap than H2 or water — Laplace-quadrature error is
    /// gap-dependent (the minimax range is set by ymin = 2*(LUMO-HOMO)), so
    /// this exercises a different part of the quadrature's operating range.
    ///
    /// Cross-checked against ferric's OWN live `rimp2::ri_mp2` (not a
    /// hardcoded literature number) — this is an internal two-code-path
    /// agreement check, matching how `test_laplace_mp2_h2_sto3g_single_virtual`
    /// already validates against live RI-MP2 rather than a stored reference.
    #[test]
    fn test_laplace_mp2_methane_ccpvdz_vs_live_rimp2() {
        let mol = Molecule::load_xyz("../../testdata/molecules/methane.xyz").unwrap();
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
        let e_mo = laplace.compute_mo(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();
        let (e_ao, _, _) = laplace.compute_ao(&mol, &obs, &dfbs, op, &rhf, 0, None).unwrap();

        let ri = crate::rimp2::ri_mp2(
            &mol, &obs, &dfbs, op, &rhf, &crate::rimp2::RiMp2Config::default(),
        ).unwrap();

        eprintln!("CH4/cc-pVDZ Laplace MO: {e_mo:.10}  AO: {e_ao:.10}  live RI-MP2: {:.10}", ri.mp2_corr);

        assert!((e_mo - e_ao).abs() < 1e-8,
            "MO and AO Laplace methods should agree on methane: {e_mo} vs {e_ao}");
        assert!((e_mo - ri.mp2_corr).abs() < 1e-3,
            "Laplace RI-MP2 ({e_mo:.6}) should be within 1e-3 Ha of live RI-MP2 ({:.6}) on methane/cc-pVDZ",
            ri.mp2_corr);
    }

    /// Widen basis coverage: water/aug-cc-pVDZ adds diffuse functions (a
    /// wider, more diffuse virtual space than the plain cc-pVDZ case already
    /// covered), cross-checked against ferric's own live RI-MP2 rather than
    /// a stored reference number.
    #[test]
    fn test_laplace_mp2_water_augccpvdz_vs_live_rimp2() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("aug-cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let dfbs_set = basis::bundled("aug-cc-pvdz-rifit").unwrap();
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
        let e_mo = laplace.compute_mo(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();
        let (e_ao, _, _) = laplace.compute_ao(&mol, &obs, &dfbs, op, &rhf, 0, None).unwrap();

        let ri = crate::rimp2::ri_mp2(
            &mol, &obs, &dfbs, op, &rhf, &crate::rimp2::RiMp2Config::default(),
        ).unwrap();

        eprintln!("H2O/aug-cc-pVDZ Laplace MO: {e_mo:.10}  AO: {e_ao:.10}  live RI-MP2: {:.10}", ri.mp2_corr);

        assert!((e_mo - e_ao).abs() < 1e-8,
            "MO and AO Laplace methods should agree on water/aug-cc-pVDZ: {e_mo} vs {e_ao}");
        assert!((e_mo - ri.mp2_corr).abs() < 1e-3,
            "Laplace RI-MP2 ({e_mo:.6}) should be within 1e-3 Ha of live RI-MP2 ({:.6}) on water/aug-cc-pVDZ",
            ri.mp2_corr);
    }

    // -----------------------------------------------------------------------
    // Laplace SOS-MP2
    //
    // The hard reference is `ri_mp2_spin_components(..).e_os` — the SAME
    // opposite-spin energy computed by an unrelated code path (canonical
    // denominators, no Laplace transform). c_os = 1.0 must reproduce it to the
    // quadrature error, and the MO and AO formulations must reproduce each
    // other to round-off.
    // -----------------------------------------------------------------------

    /// Small closed-shell setups. water/STO-3G and water/6-31G only — the
    /// algebra is what is under test, not scaling.
    fn setup_sos(basis_name: &str) -> (Molecule, PreparedBasis, PreparedBasis, ScfResult) {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled(basis_name).unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        // cc-pVDZ-RI is the bundled aux set; the RI error is common to both the
        // Laplace and the canonical reference, so it cancels in the comparison.
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
        )
        .unwrap();
        (mol, obs, dfbs, rhf)
    }

    /// c_os = 1.0 must reproduce `ri_mp2_spin_components(..).e_os`, and the
    /// deviation must SHRINK as n_quad goes 3 -> 5 -> 7. Both formulations.
    #[test]
    fn sos_cos_one_reproduces_ri_mp2_e_os_and_converges_in_n_quad() {
        for basis_name in ["sto-3g", "6-31g"] {
            let (mol, obs, dfbs, rhf) = setup_sos(basis_name);
            let op = Operator::coulomb();

            // Reference: canonical-denominator opposite-spin MP2.
            let (sc, _) = crate::rimp2::ri_mp2_spin_components(
                &mol, &obs, &dfbs, op, &rhf, &crate::rimp2::RiMp2Config::default(),
            )
            .unwrap();
            eprintln!("\n=== water/{basis_name}: reference E_OS = {:.12} ===", sc.e_os);
            assert!(
                sc.e_os.abs() > 1e-4,
                "reference E_OS is ~0 — the comparison below would be vacuous"
            );

            let mut prev_mo = f64::INFINITY;
            for &n_quad in &[3usize, 5, 7] {
                let cfg = SosMp2Config { c_os: 1.0, n_quad, ..Default::default() };
                let mo = laplace_sos_mp2(
                    &mol, &obs, &dfbs, op, &rhf, &cfg, SosFormulation::Mo,
                )
                .unwrap();
                let ao = laplace_sos_mp2(
                    &mol, &obs, &dfbs, op, &rhf, &cfg, SosFormulation::Ao,
                )
                .unwrap();

                let dev_mo = (mo.e_os - sc.e_os).abs();
                let dev_ao = (ao.e_os - sc.e_os).abs();
                eprintln!(
                    "n_quad={n_quad}: MO E_OS={:.12} (dev {dev_mo:.3e})  \
                     AO E_OS={:.12} (dev {dev_ao:.3e})  |MO-AO|={:.3e}",
                    mo.e_os,
                    ao.e_os,
                    (mo.e_os - ao.e_os).abs()
                );

                // MO and AO are different algebra for the same quantity.
                assert!(
                    (mo.e_os - ao.e_os).abs() < 1e-9,
                    "MO ({}) and AO ({}) SOS formulations must agree at n_quad={n_quad}",
                    mo.e_os,
                    ao.e_os
                );
                // c_os = 1 must be the bare OS energy, not a scaled one.
                assert!((mo.sos_corr - mo.e_os).abs() < 1e-14);
                assert!(
                    dev_mo < 1e-3,
                    "n_quad={n_quad}: Laplace SOS E_OS ({}) must match canonical \
                     E_OS ({}) to the quadrature error",
                    mo.e_os,
                    sc.e_os
                );
                // Quadrature must actually converge, not just be close once.
                assert!(
                    dev_mo <= prev_mo,
                    "increasing n_quad to {n_quad} made the deviation WORSE \
                     ({dev_mo:.3e} vs {prev_mo:.3e})"
                );
                prev_mo = dev_mo;
            }
            // The tightest grid must be tight.
            assert!(prev_mo < 1e-5, "n_quad=7 deviation {prev_mo:.3e} is too large");
        }
    }

    /// The SOS energy must be the OPPOSITE-SPIN part alone — numerically
    /// distinct from the full MP2 correlation energy and from the same-spin
    /// part. Without this a mutation returning the total MP2 energy would still
    /// pass a loose "close to reference" check.
    #[test]
    fn sos_is_the_opposite_spin_part_not_the_total() {
        let (mol, obs, dfbs, rhf) = setup_sos("6-31g");
        let op = Operator::coulomb();
        let (sc, _) = crate::rimp2::ri_mp2_spin_components(
            &mol, &obs, &dfbs, op, &rhf, &crate::rimp2::RiMp2Config::default(),
        )
        .unwrap();
        let cfg = SosMp2Config { c_os: 1.0, ..Default::default() };
        let got =
            laplace_sos_mp2(&mol, &obs, &dfbs, op, &rhf, &cfg, SosFormulation::Mo).unwrap();

        eprintln!(
            "water/6-31G: E_OS={:.12}, E_SS={:.12}, E_total={:.12}, Laplace SOS={:.12}",
            sc.e_os, sc.e_ss, sc.e_total, got.e_os
        );
        assert!(sc.e_ss.abs() > 1e-4, "E_SS must be nonzero for this test to discriminate");
        assert!(
            (got.e_os - sc.e_total).abs() > 1e-4,
            "SOS E_OS ({}) must NOT be the total MP2 correlation energy ({})",
            got.e_os,
            sc.e_total
        );
        assert!(
            (got.e_os - sc.e_ss).abs() > 1e-4,
            "SOS E_OS ({}) must NOT be the same-spin energy ({})",
            got.e_os,
            sc.e_ss
        );
        // Opposite-spin correlation is negative and, for a closed shell, the
        // dominant share of the total.
        assert!(got.e_os < 0.0, "E_OS must be negative, got {}", got.e_os);
    }

    /// `c_os` must actually scale, the default must be Jung/Head-Gordon 1.3, and
    /// the total energy must be `E_SCF + c_os·E_OS`.
    #[test]
    fn sos_config_scales_and_defaults_to_jung_head_gordon() {
        let d = SosMp2Config::default();
        assert_eq!(d.c_os, 1.3, "SOS-MP2 default c_os must be 1.3 (Jung/Head-Gordon)");
        assert_eq!(d.frozen_core, 0);
        assert_eq!(d.n_quad, 7);

        let (mol, obs, dfbs, rhf) = setup_sos("sto-3g");
        let op = Operator::coulomb();
        let unit = laplace_sos_mp2(
            &mol, &obs, &dfbs, op, &rhf,
            &SosMp2Config { c_os: 1.0, ..Default::default() },
            SosFormulation::Mo,
        )
        .unwrap();
        let scaled =
            laplace_sos_mp2(&mol, &obs, &dfbs, op, &rhf, &d, SosFormulation::Mo).unwrap();

        eprintln!(
            "water/STO-3G: E_OS={:.12}, c_os=1.3 -> sos_corr={:.12}",
            unit.e_os, scaled.sos_corr
        );
        // The UNSCALED component is the same regardless of c_os.
        assert!((scaled.e_os - unit.e_os).abs() < 1e-14, "e_os must be reported unscaled");
        assert!(
            (scaled.sos_corr - 1.3 * unit.e_os).abs() < 1e-12,
            "sos_corr must be c_os * e_os"
        );
        // Discrimination: 1.3x must be numerically distinguishable from 1.0x.
        assert!((scaled.sos_corr - unit.sos_corr).abs() > 1e-4);
        assert!(
            (scaled.total_energy - (rhf.energy + scaled.sos_corr)).abs() < 1e-12,
            "total_energy must be E_SCF + sos_corr"
        );
        assert_eq!(scaled.c_os, 1.3);
    }

    /// `select_minimax_points` hard-errors outside {3,5,7} by design. The SOS
    /// wrapper must propagate that as an error, not panic or silently coerce.
    #[test]
    fn sos_rejects_unsupported_n_quad() {
        let (mol, obs, dfbs, rhf) = setup_sos("sto-3g");
        let op = Operator::coulomb();
        for bad in [0usize, 1, 4, 6, 9] {
            let cfg = SosMp2Config { n_quad: bad, ..Default::default() };
            let r = laplace_sos_mp2(&mol, &obs, &dfbs, op, &rhf, &cfg, SosFormulation::Mo);
            assert!(r.is_err(), "n_quad={bad} must be rejected, not silently coerced");
        }
        // And a non-finite c_os is config, so it errors rather than producing NaN.
        for bad in [f64::NAN, f64::INFINITY] {
            let cfg = SosMp2Config { c_os: bad, ..Default::default() };
            assert!(laplace_sos_mp2(&mol, &obs, &dfbs, op, &rhf, &cfg, SosFormulation::Mo).is_err());
        }
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

    /// The SOS formulation selector must be STRICT.
    ///
    /// Both the CLI (`[mp2] sos_formulation`) and the Python binding
    /// (`formulation=`) route through this, so a silent fallback here would
    /// hand a user a different algebra than the one they asked for. `"MO"` is
    /// the realistic typo: right word, wrong case.
    #[test]
    fn sos_formulation_parses_strictly() {
        let p = |s, c| SosFormulation::parse_config_str(s, c);
        assert_eq!(p(None, None).unwrap(), SosFormulation::Mo);
        assert_eq!(p(Some("mo"), None).unwrap(), SosFormulation::Mo);
        assert_eq!(p(Some("ao"), None).unwrap(), SosFormulation::Ao);
        assert_eq!(
            p(Some("ao-sparse"), Some(8.0)).unwrap(),
            SosFormulation::AoSparse(8.0)
        );

        for bad in ["MO", "AO", "Mo", "mo ", "", "molecular-orbital", "sos", "aosparse"] {
            let msg = p(Some(bad), None).unwrap_err().to_string();
            assert!(
                msg.contains("unknown SOS-MP2 formulation") && msg.contains(bad),
                "{bad:?} must be rejected by name, got: {msg}"
            );
        }

        // "ao-sparse" without a cutoff must ERROR, not pick a default radius:
        // the right radius is system- and basis-dependent, and a wrong one
        // silently changes the energy.
        let msg = p(Some("ao-sparse"), None).unwrap_err().to_string();
        assert!(msg.contains("requires a domain cutoff"), "got: {msg}");

        for bad_cut in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let msg = p(Some("ao-sparse"), Some(bad_cut)).unwrap_err().to_string();
            assert!(msg.contains("must be finite and > 0"), "{bad_cut}: {msg}");
        }

        // A cutoff on an EXACT formulation is a configuration error, not a
        // no-op — silently ignoring it would run a different method than the
        // one the user configured.
        for exact in [None, Some("mo"), Some("ao")] {
            let msg = p(exact, Some(8.0)).unwrap_err().to_string();
            assert!(
                msg.contains("only meaningful with formulation"),
                "{exact:?} + cutoff must be rejected, got: {msg}"
            );
        }
    }

    /// The sparse AO path converges to the exact AO path as the domain radius
    /// grows, and reproduces it EXACTLY once the cutoff spans the molecule.
    ///
    /// Uses butane, not water: water's diameter is 2.9 Bohr, so every cutoff
    /// >= 3 already covers the whole molecule and the test would pass
    /// vacuously (measured: |d| = 0.0 at EVERY radius >= 3). Butane is 10.5
    /// Bohr across, so the small radii here genuinely truncate.
    ///
    /// The `r = 4` deviation is asserted to be LARGE on purpose. Domain
    /// truncation is not a mild perturbation at these radii — see
    /// `sos_ao_sparse_truncation_radius_tracks_molecular_diameter` for why
    /// that matters.
    #[test]
    fn sos_ao_sparse_converges_to_dense_as_cutoff_grows() {
        let mol = Molecule::load_xyz(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/molecules/alkane_4.xyz"
        ))
        .unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let aux = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() },
        )
        .unwrap();

        let sos = |form| {
            laplace_sos_mp2(
                &mol,
                &obs,
                &dfbs,
                op,
                &rhf,
                &SosMp2Config { c_os: 1.0, ..Default::default() },
                form,
            )
            .unwrap()
            .e_os
        };

        let dense = sos(SosFormulation::Ao);
        let radii = [4.0, 6.0, 8.0, 12.0];
        let devs: Vec<f64> =
            radii.iter().map(|&r| (sos(SosFormulation::AoSparse(r)) - dense).abs()).collect();
        eprintln!("butane/STO-3G dense e_os = {dense:.12}");
        for (r, d) in radii.iter().zip(&devs) {
            eprintln!("  cutoff {r:5.1} Bohr -> |d| = {d:.3e}  ({:.1}%)", 100.0 * d / dense.abs());
        }

        // Monotone: a larger domain can only ADD terms back.
        for w in devs.windows(2) {
            assert!(
                w[1] <= w[0] * 1.001,
                "sparse must approach dense as the cutoff grows, got {devs:?}"
            );
        }
        // A cutoff spanning the molecule is a vacuous mask => exact.
        assert!(
            devs[3] < 1e-10,
            "a cutoff spanning the molecule must reproduce the dense AO path \
             exactly, got |d| = {:.3e}",
            devs[3]
        );

        // THE POSITIVE CLAIM (rewritten 2026-07-27, see the index-mismatch fix
        // in boys::build_pseudo_density_occ_sparse). Truncation now PAYS: a
        // 4 Bohr domain on a 10.5 Bohr molecule already reaches well inside
        // chemical accuracy. Before the fix this same row was 63.3% error.
        assert!(
            devs[0] < 1e-3 * dense.abs(),
            "cutoff 4 Bohr on butane should be near-exact after the localized \
             exp(tF) fix; got {:.3e} of {dense:.6} ({:.3}%) — a large deviation \
             here means the canonical/localized index mismatch is BACK",
            devs[0],
            100.0 * devs[0] / dense.abs()
        );

        // TEETH: the sweep must still contain a radius that genuinely masks
        // something, or "near-exact everywhere" would be trivially true and
        // this test would prove nothing. r=4 must be a real restriction, i.e.
        // strictly worse than the vacuous r=12.
        assert!(
            devs[0] > devs[3],
            "r=4 must be a GENUINE restriction (worse than the vacuous r=12): \
             {:.3e} vs {:.3e} — if these are equal the mask is doing nothing \
             and the test is vacuous",
            devs[0],
            devs[3]
        );
    }

    /// TRUNCATION IS TRANSFERABLE — a fixed radius works across system sizes.
    ///
    /// # History: this test used to pin the OPPOSITE claim
    ///
    /// It formerly asserted that the radius needed for 1% accuracy TRACKS THE
    /// MOLECULAR DIAMETER, with a table showing r(1%) == r(exact) at every
    /// size, and concluded there was no accurate-and-profitable regime. That
    /// was an artifact of an index-mismatch bug in
    /// `boys::build_pseudo_density_occ_sparse`, which masked CANONICAL orbital
    /// i by LOCALIZED orbital i's domain. Because the two index sets are
    /// unrelated after a Jacobi rotation, exactness required every ball to
    /// cover every AO — which PREDICTS r(exact) ~ diameter a priori, with no
    /// physics involved. The old test faithfully pinned that prediction.
    ///
    /// The test did its job: it FAILED the moment the construction was fixed,
    /// with its own message saying the negative result "needs revisiting".
    /// This is what a tripwire is for.
    ///
    /// # What is asserted now
    ///
    /// A radius that is exact for butane (10.5 Bohr across) must ALSO be
    /// essentially exact for octane (19.9 Bohr) — i.e. the usable radius does
    /// NOT grow with the molecule. Measured after the fix: octane at 12 Bohr
    /// is exact to round-off, and danuglipron (71 atoms, 31.3 Bohr) reaches
    /// 0.05% at r = 4 Bohr, a domain spanning ~13% of its diameter.
    ///
    /// If this test ever fails again with octane badly wrong at a radius that
    /// suffices for butane, the index mismatch (or an equivalent masking
    /// error) has returned.
    #[test]
    fn sos_ao_sparse_truncation_radius_is_transferable_across_sizes() {
        let run = |name: &str, cutoff: f64| -> (f64, f64) {
            let mol = Molecule::load_xyz(&format!(
                "{}/../../testdata/molecules/{name}.xyz",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let bs = basis::bundled("sto-3g").unwrap();
            let aux = basis::bundled("cc-pvdz-ri").unwrap();
            let obs = PreparedBasis::new(&mol, &bs).unwrap();
            let dfbs = PreparedBasis::new(&mol, &aux).unwrap();
            let op = Operator::coulomb();
            let bounds = SchwarzBounds::compute(op, &obs).unwrap();
            let rhf = solve_rhf(
                &ferric_core::parallel::ParallelContext::default(),
                &mol,
                &obs,
                op,
                &bounds,
                &RhfConfig { energy_conv: 1e-10, ..Default::default() },
            )
            .unwrap();
            let cfg = SosMp2Config { c_os: 1.0, ..Default::default() };
            let dense =
                laplace_sos_mp2(&mol, &obs, &dfbs, op, &rhf, &cfg, SosFormulation::Ao)
                    .unwrap()
                    .e_os;
            let sparse = laplace_sos_mp2(
                &mol,
                &obs,
                &dfbs,
                op,
                &rhf,
                &cfg,
                SosFormulation::AoSparse(cutoff),
            )
            .unwrap()
            .e_os;
            (dense, (sparse - dense).abs() / dense.abs())
        };

        // A radius that is EXACT for butane (10.5 Bohr across) must still be
        // badly wrong for octane (19.9 Bohr across). That is the whole point:
        // the usable radius is not transferable between system sizes.
        let (_, rel_butane) = run("alkane_4", 12.0);
        let (_, rel_octane) = run("alkane_8", 12.0);
        eprintln!(
            "cutoff 12 Bohr: butane rel err {rel_butane:.3e}, octane rel err {rel_octane:.3e}"
        );
        assert!(rel_butane < 1e-9, "12 Bohr spans butane, expected exact: {rel_butane:.3e}");
        // THE CLAIM: the same radius transfers to a molecule ~2x the size.
        assert!(
            rel_octane < 1e-6,
            "a 12 Bohr radius that is exact for butane must also be essentially \
             exact for octane (transferability); got {rel_octane:.3e}. A LARGE \
             value here means the canonical/localized index mismatch in \
             build_pseudo_density_occ_sparse has returned — that bug made the \
             usable radius track the molecular diameter"
        );
        // TEETH: a radius far BELOW both diameters must still be a genuine
        // restriction on the larger molecule, or the sweep proves nothing.
        let (_, rel_octane_tight) = run("alkane_8", 3.0);
        eprintln!("cutoff  3 Bohr: octane rel err {rel_octane_tight:.3e}");
        assert!(
            rel_octane_tight > rel_octane,
            "a 3 Bohr domain must be a stricter restriction than 12 Bohr on \
             octane ({rel_octane_tight:.3e} vs {rel_octane:.3e}); if equal, the \
             mask is inert and this test is vacuous"
        );
    }
}
