//! Orbital-optimized RI-MP2 (OO-RI-MP2).
//!
//! Minimizes E_HF + E_MP2 jointly by optimizing orbital rotation parameters
//! using a level-shifted approximate Newton step with DIIS extrapolation
//! and Cayley orbital rotations.
//!
//! The level-shifted diagonal Hessian uses orbital energy differences as the
//! approximate Hessian: kappa_{ai} = -g_{ai} / (eps_a - eps_i + mu), following
//! Bozkaya & Sherrill, JCP 135, 104103 (2011). DIIS (Pulay extrapolation)
//! accelerates convergence of the orbital rotation parameters.
//!
//! The analytic orbital gradient uses the Hylleraas functional derivative,
//! which includes both the 1-PDM/Fock terms and the 2-electron integral
//! response terms from the MO integral derivatives.

use crate::rimp2::{active_occ, cholesky_inverse_sqrt};
use ferric_core::mol::Molecule;
use ferric_core::orbitals::OrbitalSpace;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::{env_budget_bytes, ThreeIndexSource};
use ferric_integrals::threeindex;
use ferric_scf::diis::Diis;
use ferric_scf::rhf::build_jk;
use ferric_scf::ScfResult;
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array2, Array3};
use ndarray_linalg::Solve;
use std::cell::RefCell;

/// Configuration for OO-RI-MP2.
#[derive(Debug, Clone)]
pub struct OoRiMp2Config {
    pub max_iter: usize,
    pub grad_conv: f64,
    pub energy_conv: f64,
    pub step_size: f64,
    pub frozen_core: usize,
    /// Level shift for the approximate diagonal Hessian (Ha).
    /// Regularizes the Newton step when orbital energy gaps are small.
    pub level_shift: f64,
    /// Maximum DIIS subspace size for orbital rotation extrapolation.
    pub diis_size: usize,
    /// Whether to use DIIS for orbital rotations.
    pub use_diis: bool,
    /// Optional resident-bytes ceiling for the 3-index MO transform. `None` →
    /// resolved via [`ferric_core::memory::resolve_budget_bytes`].
    pub memory_budget_bytes: Option<usize>,
}

impl Default for OoRiMp2Config {
    fn default() -> Self {
        Self {
            max_iter: 100,
            grad_conv: 1e-4,
            energy_conv: 1e-8,
            step_size: 0.5,
            frozen_core: 0,
            level_shift: 0.1,
            diis_size: 6,
            use_diis: true,
            memory_budget_bytes: None,
        }
    }
}

/// Result from OO-RI-MP2.
#[derive(Debug)]
pub struct OoRiMp2Result {
    /// Total energy: E_HF(optimized) + E_MP2(optimized).
    pub total_energy: f64,
    /// Re-optimized HF energy component.
    pub hf_energy: f64,
    /// MP2 correlation energy with optimized orbitals.
    pub mp2_corr: f64,
    /// Whether gradient and energy convergence thresholds were met.
    pub converged: bool,
    /// Number of orbital optimization iterations.
    pub iterations: usize,
    /// Final orbital gradient norm.
    pub grad_norm: f64,
    /// Optimized MO coefficients.
    pub mos: Array2<f64>,
    /// Orbital energies from the optimized Fock matrix.
    pub orbital_energies: Vec<f64>,
}

/// AO-side invariants for OO-RI-MP2, built once and reused across every
/// orbital-rotation iteration. These depend only on `(obs, dfbs, op)` — not on
/// the MO coefficients — so rebuilding them per iteration (and per line-search
/// backtrack) was pure waste.
///
/// The `(naux, nao, nao)` AO 3-index tensor is served through a
/// memory-budgeted [`ThreeIndexSource`] (`FERRIC_ERI3_BUDGET_GB`): in-core when
/// it fits the budget (identical to the old resident `Array3`), disk-spilled in
/// aux-blocks when it does not. Consumers pull raw aux-blocks via
/// `for_each_block` and dress each block with `V^{-1/2}` on the fly, so the peak
/// resident 3-index footprint is one aux-block, not the full tensor.
///
/// `RefCell` gives the `for_each_block` iterator the `&mut` it needs (disk seek
/// + scratch reuse) while `OoRiMp2AoTensors` is shared as `&self` across the
/// hot orbital-optimization loop. Borrows are non-overlapping (each transform
/// takes the borrow, streams, drops it), so no runtime borrow conflict arises.
pub struct OoRiMp2AoTensors {
    /// V^{-1/2}, shape (naux, naux).
    pub v2c_inv_sqrt: Array2<f64>,
    /// Budget-aware raw AO 3-center integral source (P|mu nu), (naux, nao, nao).
    pub eri3_ao: RefCell<ThreeIndexSource>,
    naux: usize,
    nao: usize,
}

impl OoRiMp2AoTensors {
    /// Build the AO-side invariants once, budget from `FERRIC_ERI3_BUDGET_GB`.
    pub fn build(
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
    ) -> Result<Self, FerricError> {
        Self::build_with_budget(obs, dfbs, op, env_budget_bytes())
    }

    /// Build with an explicit resident-bytes budget for the raw 3-index tensor.
    pub fn build_with_budget(
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
        budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        let v2c = threeindex::coulomb_metric_2c(op, dfbs)?;
        let v2c_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
        let src = ThreeIndexSource::build(op, obs, dfbs, budget_bytes)?;
        let naux = src.naux();
        let nao = src.nao();
        Ok(Self { v2c_inv_sqrt, eri3_ao: RefCell::new(src), naux, nao })
    }

    /// Number of auxiliary basis functions (rows of the 3-index tensor).
    pub fn naux(&self) -> usize {
        self.naux
    }
    /// Number of AO basis functions.
    pub fn nao(&self) -> usize {
        self.nao
    }
}

/// Compute the full-MO 3-center B tensor: B^P_{pq} for all MO pairs p,q.
///
/// Returns b_full of shape (naux, nmo, nmo) where:
///   b_full[(P, p, q)] = sum_Q V^{-1/2}_{PQ} sum_{mu,nu} (Q|mu nu) C_{mu,p} C_{nu,q}
///
/// AO-side objects are rebuilt each call; prefer [`compute_b_full_mo_with`] in
/// hot loops where the AO invariants are hoisted.
pub fn compute_b_full_mo(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    c: &Array2<f64>,
) -> Result<Array3<f64>, FerricError> {
    let ao = OoRiMp2AoTensors::build(obs, dfbs, op)?;
    compute_b_full_mo_with(&ao, c)
}

/// Aux-row chunk for the streamed MO transform + metric dressing. Caps the
/// MO-transformed transient at `MO_CHUNK · width · 8` bytes regardless of the
/// raw source's block size (the in-core backend serves one block spanning all
/// of naux, which would otherwise reintroduce a full-size transient). 256 keeps
/// the dressing GEMM's inner dimension wide (k = 256) while bounding the panel.
const MO_CHUNK: usize = 256;

/// Full-MO 3-center B tensor from pre-built AO invariants.
///
/// Memory: streams raw AO aux-blocks from the budgeted [`ThreeIndexSource`],
/// MO-transforms them in chunks of at most [`MO_CHUNK`] aux rows (BLAS3:
/// half = (B^Q_AO)·C, then C^T·half), and dresses each chunk into the output
/// with `V^{-1/2}` on the fly. The output `(naux, nmo, nmo)` tensor is
/// allocated once; the transient is one `(≤MO_CHUNK, nmo²)` panel. This removes
/// the former 2× peak (the resident full `eri3_mo` AND its `b_flat` copy held
/// simultaneously during the metric GEMM).
///
/// Exactness: `b_full[P,p,q] = Σ_Q V^{-1/2}[P,Q] · (C^T (Q|μν) C)[p,q]`, the
/// same contraction as before — reordered, not approximated.
pub fn compute_b_full_mo_with(
    ao: &OoRiMp2AoTensors,
    c: &Array2<f64>,
) -> Result<Array3<f64>, FerricError> {
    let naux = ao.naux();
    let nmo = c.ncols();

    let mut b_full = Array3::<f64>::zeros((naux, nmo, nmo));
    let m = &ao.v2c_inv_sqrt;
    ao.eri3_ao.borrow_mut().for_each_block(|blk| {
        let qb = blk.data.shape()[0];
        let mut q0 = 0;
        while q0 < qb {
            let q1 = (q0 + MO_CHUNK).min(qb);
            let qc = q1 - q0;
            // MO-transform this chunk: mo[q,p,r] = C^T (Q|μν) C  (BLAS3 per q).
            let mut mo_blk = Array2::<f64>::zeros((qc, nmo * nmo));
            for q in 0..qc {
                let bq_ao = blk.data.slice(ndarray::s![q0 + q, .., ..]);
                let half = bq_ao.dot(c); // (nao, nmo)
                let bq_mo = c.t().dot(&half); // (nmo, nmo)
                mo_blk
                    .slice_mut(ndarray::s![q, ..])
                    .assign(&bq_mo.into_shape_with_order(nmo * nmo).unwrap());
            }
            // Dress into every output aux row, accumulating IN PLACE (beta=1):
            //   b_full[:, p, q] += V^{-1/2}[:, Qchunk] · mo_blk.
            // general_mat_mul writes straight into b_full — no (naux, nmo²)
            // `contrib` allocation, which would itself be a second full copy.
            let msub = m.slice(ndarray::s![.., blk.p0 + q0..blk.p0 + q1]);
            let mut b_flat = b_full
                .view_mut()
                .into_shape_with_order((naux, nmo * nmo))
                .unwrap();
            ndarray::linalg::general_mat_mul(1.0, &msub, &mo_blk, 1.0, &mut b_flat);
            q0 = q1;
        }
        Ok(())
    })?;
    Ok(b_full)
}

/// Compute the RI-MP2 energy for a given set of MO coefficients.
///
/// Returns (e_mp2, b_ov_flat) where b_ov_flat is (naux, nocc*nvir) for reuse.
/// Uses pre-built AO invariants (see [`OoRiMp2AoTensors`]); only the MO
/// transform + fitting contraction depend on `c`.
fn compute_rimp2_with_orbitals(
    ao: &OoRiMp2AoTensors,
    c: &Array2<f64>,
    eps: &[f64],
    orb: &OrbitalSpace,
) -> Result<(f64, Array2<f64>), FerricError> {
    let OrbitalSpace { nocc, nocc_total, first_occ, nvir } = *orb;
    let naux = ao.naux();

    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    // MO transform (P|μν) -> (P|ia) and dress with V^{-1/2} on the fly, streaming
    // raw AO aux-blocks in ≤MO_CHUNK-row chunks.
    //   b_flat[P, ia] = Σ_Q V^{-1/2}[P,Q] (C_occ^T (Q|μν) C_vir)[ia]
    // The dressing GEMM accumulates in place (beta=1); the peak transient is one
    // (≤MO_CHUNK, nocc·nvir) MO panel, never a second full-size copy.
    let nov = nocc * nvir;
    let mut b_flat = Array2::<f64>::zeros((naux, nov));
    let m = &ao.v2c_inv_sqrt;
    ao.eri3_ao.borrow_mut().for_each_block(|blk| {
        let qb = blk.data.shape()[0];
        let mut q0 = 0;
        while q0 < qb {
            let q1 = (q0 + MO_CHUNK).min(qb);
            let qc = q1 - q0;
            let mut mo_blk = Array2::<f64>::zeros((qc, nov));
            for q in 0..qc {
                let bq_ao = blk.data.slice(ndarray::s![q0 + q, .., ..]);
                // (P|ia) = C_occ^T (Q|μν) C_vir
                let tmp = bq_ao.dot(&c_vir); // (nao, nvir)
                let bq_mo = c_occ.t().dot(&tmp); // (nocc, nvir)
                mo_blk
                    .slice_mut(ndarray::s![q, ..])
                    .assign(&bq_mo.into_shape_with_order(nov).unwrap());
            }
            let msub = m.slice(ndarray::s![.., blk.p0 + q0..blk.p0 + q1]);
            ndarray::linalg::general_mat_mul(1.0, &msub, &mo_blk, 1.0, &mut b_flat);
            q0 = q1;
        }
        Ok(())
    })?;

    // MP2 energy via i-blocked wide GEMMs (same path as the main RI-MP2 lane).
    let sc = crate::rimp2::spin_components_from_b_ov(
        &b_flat, eps, nocc, nvir, first_occ, nocc_total,
    );
    Ok((sc.e_total, b_flat))
}

/// Build the HF energy from MO coefficients + 1e integrals + J/K.
fn compute_hf_energy(
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    c: &Array2<f64>,
    nocc_total: usize,
    h: &Array2<f64>,
) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
    let n = prep.nbasis();

    // Build density: D = 2 * C_occ C_occ^T
    let mut d = Array2::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for i in 0..nocc_total {
                sum += c[(mu, i)] * c[(nu, i)];
            }
            d[(mu, nu)] = 2.0 * sum;
        }
    }

    // Build J, K
    let mut j_mat = Array2::zeros((n, n));
    let mut k_mat = Array2::zeros((n, n));
    let ctx = ferric_core::parallel::ParallelContext::default();
    build_jk(&ctx, prep, bounds, 1e-12, &d, &mut j_mat, &mut k_mat)?;

    // F = H + J - 0.5*K
    let f = h + &j_mat - &(0.5 * &k_mat);

    // E_elec = 0.5 * tr(D * (H + F))
    let hpf = h + &f;
    let e_elec: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| 0.5 * d[(i, j)] * hpf[(i, j)])
        .sum();
    let vnn = mol.nuclear_repulsion();
    let e_hf = e_elec + vnn;

    Ok((e_hf, f, d))
}

/// Compute orbital energies as diagonal of C^T F C.
fn orbital_energies(c: &Array2<f64>, f: &Array2<f64>) -> Vec<f64> {
    let n = c.ncols();
    let f_mo = c.t().dot(f).dot(c);
    (0..n).map(|i| f_mo[(i, i)]).collect()
}

/// Compute t2 amplitudes and (ia|jb) integrals from B tensor.
///
/// t2 is stored as flat vec of length (nocc*nvir)^2 with indexing t2[ia*nov + jb]
/// where ia = i*nvir + a, jb = j*nvir + b.
///
/// Returns (t2, eri_ov) where eri_ov[ia*nov + jb] = (ia|jb).
pub fn compute_t2_and_integrals(
    b_flat: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    nocc_total: usize,
    first_occ: usize,
    naux: usize,
) -> (Vec<f64>, Vec<f64>) {
    let nov = nocc * nvir;
    let mut t2 = vec![0.0f64; nov * nov];
    let mut eri_ov = vec![0.0f64; nov * nov];

    for i in 0..nocc {
        for a in 0..nvir {
            let ia = i * nvir + a;
            for j in 0..nocc {
                for b in 0..nvir {
                    let jb = j * nvir + b;
                    let eri_iajb: f64 =
                        (0..naux).map(|p| b_flat[(p, ia)] * b_flat[(p, jb)]).sum();
                    let denom = eps[first_occ + i] + eps[first_occ + j]
                        - eps[nocc_total + a]
                        - eps[nocc_total + b];
                    eri_ov[ia * nov + jb] = eri_iajb;
                    t2[ia * nov + jb] = eri_iajb / denom;
                }
            }
        }
    }
    (t2, eri_ov)
}

/// Compute only the t2 amplitudes from the B tensor, without materializing the
/// (ia|jb) integral array.
///
/// Identical numerics to [`compute_t2_and_integrals`] for its first return value,
/// but allocates a single `nov²` buffer instead of two — the `eri_ov` tensor is
/// never built. Callers that discard the integrals (e.g. OSV/PNO construction in
/// ferric-rpa) should prefer this to halve the transient footprint (~10 GB → ~5 GB
/// at dimer/aTZ scale). Indexing matches: t2[ia*nov + jb], ia = i*nvir + a.
pub fn compute_t2_only(
    b_flat: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
    nocc_total: usize,
    first_occ: usize,
    naux: usize,
) -> Vec<f64> {
    let nov = nocc * nvir;
    let mut t2 = vec![0.0f64; nov * nov];

    for i in 0..nocc {
        for a in 0..nvir {
            let ia = i * nvir + a;
            for j in 0..nocc {
                for b in 0..nvir {
                    let jb = j * nvir + b;
                    let eri_iajb: f64 =
                        (0..naux).map(|p| b_flat[(p, ia)] * b_flat[(p, jb)]).sum();
                    let denom = eps[first_occ + i] + eps[first_occ + j]
                        - eps[nocc_total + a]
                        - eps[nocc_total + b];
                    t2[ia * nov + jb] = eri_iajb / denom;
                }
            }
        }
    }
    t2
}

/// Build the full relaxed 1-PDM for OO-MP2 in MO basis.
///
/// For OO-MP2, the density is already "relaxed" because it is a stationary
/// point w.r.t. orbital rotations. The MO-basis density is:
///   P_pq = delta_pq (for occ) + P^MP2_pq
pub fn build_oo_mp2_relaxed_density(
    t2: &[f64],
    nocc: usize,
    nvir: usize,
    nmo: usize,
    first_occ: usize,
) -> Array2<f64> {
    let (p_oo, p_vv) = build_mp2_density(t2, nocc, nvir);
    let mut p = Array2::zeros((nmo, nmo));
    
    // HF occupied part
    for i in 0..nocc {
        let idx = first_occ + i;
        p[(idx, idx)] = 2.0;
    }
    
    // MP2 correction
    for i in 0..nocc {
        for j in 0..nocc {
            p[(first_occ + i, first_occ + j)] += p_oo[(i, j)];
        }
    }
    for a in 0..nvir {
        for b in 0..nvir {
            let nocc_total = nmo - nvir;
            p[(nocc_total + a, nocc_total + b)] += p_vv[(a, b)];
        }
    }
    p
}

/// Build the MP2 unrelaxed 1-particle density matrix in MO basis.
pub fn build_mp2_density(
    t2: &[f64],
    nocc: usize,
    nvir: usize,
) -> (Array2<f64>, Array2<f64>) {
    let nov = nocc * nvir;

    // P^MP2_ij = -sum_{kab} t_{ik,ab} (2 t_{jk,ab} - t_{jk,ba})
    let mut p_oo = Array2::zeros((nocc, nocc));
    for i in 0..nocc {
        for j in 0..nocc {
            let mut sum = 0.0;
            for k in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let ik_ab = (i * nvir + a) * nov + k * nvir + b;
                        let jk_ab = (j * nvir + a) * nov + k * nvir + b;
                        let jk_ba = (j * nvir + b) * nov + k * nvir + a;
                        sum += t2[ik_ab] * (2.0 * t2[jk_ab] - t2[jk_ba]);
                    }
                }
            }
            p_oo[(i, j)] = -sum;
        }
    }

    // P^MP2_ab = sum_{ijc} t_{ij,ac} (2 t_{ij,bc} - t_{ij,cb})
    let mut p_vv = Array2::zeros((nvir, nvir));
    for a in 0..nvir {
        for b in 0..nvir {
            let mut sum = 0.0;
            for i in 0..nocc {
                for j in 0..nocc {
                    for c in 0..nvir {
                        let ij_ac = (i * nvir + a) * nov + j * nvir + c;
                        let ij_bc = (i * nvir + b) * nov + j * nvir + c;
                        let ij_cb = (i * nvir + c) * nov + j * nvir + b;
                        sum += t2[ij_ac] * (2.0 * t2[ij_bc] - t2[ij_cb]);
                    }
                }
            }
            p_vv[(a, b)] = sum;
        }
    }

    (p_oo, p_vv)
}

/// Compute the full OO-MP2 orbital gradient g_{ai} for occupied-virtual rotations.
///
/// This implements the Hylleraas functional derivative which includes:
/// 1. The 1-PDM/Fock terms (response of Fock matrix to density change)
/// 2. The 2-electron integral response terms (response of (ia|jb) to orbital rotation)
///
/// The formula is derived from the Hylleraas functional:
///   L[t] = 2 * sum_{ijab} t_{ij,ab} * (2*(ia|jb) - (ib|ja)) - sum_{ijab} t_{ij,ab} * D_{ijab} * tau_{ijab}
///
/// At the stationary point L = E_MP2, and by the 2n+1 rule the gradient of the energy
/// w.r.t. orbital rotation kappa_{ck} (c=virtual, k=occupied) at fixed amplitudes is:
///
///   dE_total/d kappa_{ck} = 4*F_{ck}  (HF part, = 0 at convergence)
///     + 2 * sum_{ijab} t_{ij,ab} * [2*d(ia|jb)/dk - d(ib|ja)/dk]   (MP2 integral response)
///
/// The orbital energy denominator terms vanish at the HF solution (dD/dk = 0 because
/// Brillouin condition makes d eps_p/d kappa_{ck} = 0).
///
/// The integral response d(ia|jb)/d kappa_{ck} = delta_{ik}*(ca|jb) + delta_{jk}*(ia|cb)
///   - delta_{ac}*(ik|jb) - delta_{bc}*(ia|jk), computed using the full-MO B tensor.
fn compute_orbital_gradient(
    f_mo: &Array2<f64>,
    t2: &[f64],
    b_full: &Array3<f64>,
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
) -> Array2<f64> {
    // VVOV panel width from the resident-bytes budget: one c-value costs
    // nvir·nocc·nvir·8 bytes of VVOV rows. Unset budget = one full-width panel
    // (bit-identical to the former unblocked path).
    let nov = nocc * nvir;
    let row_bytes = nvir.saturating_mul(nov).saturating_mul(8).max(1);
    let panel_c = (env_budget_bytes() / row_bytes).max(1).min(nvir.max(1));
    compute_orbital_gradient_panelled(
        f_mo, t2, b_full, nocc, nvir, first_occ, nocc_total, panel_c,
    )
}

/// [`compute_orbital_gradient`] with an explicit VVOV c-panel width. The
/// panelled evaluation is exact for any `panel_c >= 1` — panels change memory
/// shape only, never the contraction. Split out so tests can force multi-panel
/// execution regardless of the environment budget.
// Orbital-space sizes plus the panel width are all irreducibly distinct.
#[allow(clippy::too_many_arguments)]
fn compute_orbital_gradient_panelled(
    f_mo: &Array2<f64>,
    t2: &[f64],
    b_full: &Array3<f64>,
    nocc: usize,
    nvir: usize,
    first_occ: usize,
    nocc_total: usize,
    panel_c: usize,
) -> Array2<f64> {
    use rayon::prelude::*;

    let naux = b_full.shape()[0];
    let nov = nocc * nvir;

    // Dressed B blocks (contiguous, aux-major) sliced out of the full-MO tensor.
    //   Bov[P, i*nvir+a] = B^P_{i,a}   (occ, vir)
    //   Bvv[P, a*nvir+b] = B^P_{a,b}   (vir, vir)
    //   Boo[P, i*nocc+j] = B^P_{i,j}   (occ, occ)
    // Building the dense MO-ERI blocks below is then a single wide GEMM each,
    // replacing the former per-element O(naux) dot inside the ijab loops
    // (which cost O(naux · nvir³ nocc³)).
    let mut b_ov = Array2::<f64>::zeros((naux, nov));
    let mut b_vv = Array2::<f64>::zeros((naux, nvir * nvir));
    let mut b_oo = Array2::<f64>::zeros((naux, nocc * nocc));
    for p in 0..naux {
        for i in 0..nocc {
            let i_mo = first_occ + i;
            for a in 0..nvir {
                b_ov[(p, i * nvir + a)] = b_full[(p, i_mo, nocc_total + a)];
            }
            for j in 0..nocc {
                b_oo[(p, i * nocc + j)] = b_full[(p, i_mo, first_occ + j)];
            }
        }
        for a in 0..nvir {
            let a_mo = nocc_total + a;
            for b in 0..nvir {
                b_vv[(p, a * nvir + b)] = b_full[(p, a_mo, nocc_total + b)];
            }
        }
    }

    // OOOV is small (nocc²·nov·8 ≈ 0.44 GB at the audit scale) — build once.
    //   OOOV[(i*nocc+k), (j*nvir+b)] = (ik|jb)
    let ooov = b_oo.t().dot(&b_ov); // (nocc², nocc·nvir)

    // VVOV[(c*nvir+a), (j*nvir+b)] = (ca|jb), shape (nvir², nocc·nvir), is the
    // single largest transient in the crate (~203 GB at the audit scale). Every
    // VVOV read inside the c_idx loop below is confined to the `nvir` rows
    // [c_idx*nvir, c_idx*nvir+nvir), so we never need more than a panel of c
    // rows resident. Block over c-panels: build vvov_panel for a panel of
    // c-values (a wide GEMM: b_vv[:, panel].t() · b_ov), consume it, discard it.
    // Peak VVOV footprint is one panel instead of the full (nvir², nov) square.
    let panel_c = panel_c.max(1).min(nvir.max(1));

    // g_{ai} has shape (nvir, nocc) -- virtual index a, occupied index i
    let mut g = Array2::zeros((nvir, nocc));

    // HF contribution: -4 * F_{ai} (sign follows the Cayley rotation convention
    // where kappa_{ai}>0 mixes occupied into virtual: C_new ≈ C(I-K))
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            g[(a, i)] -= 4.0 * f_mo[(a_mo, i_mo)];
        }
    }

    // MP2 integral response:
    // dE_MP2/d kappa_{ck} = 2 * sum_{ijab} t_{ij,ab} * [2*d(ia|jb)/dk_{ck} - d(ib|ja)/dk_{ck}]
    //
    // d(ia|jb)/dk_{ck} = delta_{ik}*(ca|jb) + delta_{jk}*(ia|cb) - delta_{ac}*(ik|jb) - delta_{bc}*(ia|jk)
    // d(ib|ja)/dk_{ck} = delta_{ik}*(cb|ja) + delta_{jk}*(ib|ca) - delta_{bc}*(ik|ja) - delta_{ac}*(ib|jk)
    //
    // Combined: 2*d(ia|jb)/dk - d(ib|ja)/dk =
    //   delta_{ik} * [2*(ca|jb) - (cb|ja)]
    // + delta_{jk} * [2*(ia|cb) - (ib|ca)]
    // - delta_{ac} * [2*(ik|jb) - (ib|jk)]
    // - delta_{bc} * [2*(ia|jk) - (ik|ja)]
    //
    // For each (c, k) pair, we sum over ijab with the appropriate delta contractions.
    // ERIs are read from the (panelled) VVOV / (full) OOOV blocks:
    //   (ca|jb) = vvov[c*nvir+a, j*nvir+b]  (vvov row = local c-panel offset)
    //   (ik|jb) = ooov[i*nocc+k, j*nvir+b]     ((ib|jk)=(jk|ib)=ooov[j*nocc+k, i*nvir+b])

    let mut c0 = 0;
    while c0 < nvir {
        let c1 = (c0 + panel_c).min(nvir);
        // vvov_panel rows correspond to c in [c0, c1): local row (c-c0)*nvir + a.
        let bvv_panel = b_vv.slice(ndarray::s![.., c0 * nvir..c1 * nvir]);
        let vvov_panel = bvv_panel.t().dot(&b_ov); // ((c1-c0)·nvir, nov)

        // Parallelize over the panel's c-values. Each c_idx reads only shared,
        // read-only inputs (vvov_panel/ooov/t2) and produces its own row of nocc
        // gradient contributions — a disjoint-write pattern. We collect each
        // c_idx's row independently and scatter into `g` serially afterward.
        // The per-c_idx `grad_ck` accumulation order is byte-for-byte the same
        // as the serial version regardless of thread count, so the result is
        // bit-identical (no cross-c_idx summation whose order could vary).
        let panel_rows: Vec<(usize, Vec<f64>)> = (c0..c1)
            .into_par_iter()
            .map(|c_idx| {
                let cbase = (c_idx - c0) * nvir; // local vvov_panel row base for this c
                let mut row = vec![0.0_f64; nocc];
                for (k, row_k) in row.iter_mut().enumerate() {
                    let mut grad_ck = 0.0;

                    // Term 1: delta_{ik} -> i=k, sum over j,a,b
                    // 2 * sum_{jab} t_{kj,ab} * [2*(ca|jb) - (cb|ja)]
                    for j in 0..nocc {
                        for a in 0..nvir {
                            let ca = cbase + a;
                            for b in 0..nvir {
                                let t_kj_ab = t2[(k * nvir + a) * nov + j * nvir + b];
                                let eri_cajb = vvov_panel[(ca, j * nvir + b)];
                                let eri_cbja = vvov_panel[(cbase + b, j * nvir + a)];
                                grad_ck += t_kj_ab * (2.0 * eri_cajb - eri_cbja);
                            }
                        }
                    }

                    // Term 2: delta_{jk} -> j=k, sum over i,a,b
                    // 2 * sum_{iab} t_{ik,ab} * [2*(ia|cb) - (ib|ca)]
                    // (ia|cb) = (cb|ia) = vvov[c*nvir+b, i*nvir+a];
                    // (ib|ca) = (ca|ib) = vvov[c*nvir+a, i*nvir+b]
                    for i in 0..nocc {
                        for a in 0..nvir {
                            for b in 0..nvir {
                                let t_ik_ab = t2[(i * nvir + a) * nov + k * nvir + b];
                                let eri_iacb = vvov_panel[(cbase + b, i * nvir + a)];
                                let eri_ibca = vvov_panel[(cbase + a, i * nvir + b)];
                                grad_ck += t_ik_ab * (2.0 * eri_iacb - eri_ibca);
                            }
                        }
                    }

                    // Term 3: delta_{ac} -> a=c, sum over i,j,b
                    // -2 * sum_{ijb} t_{ij,cb} * [2*(ik|jb) - (ib|jk)]
                    // (ik|jb) = ooov[i*nocc+k, j*nvir+b];
                    // (ib|jk) = (jk|ib) = ooov[j*nocc+k, i*nvir+b]
                    for i in 0..nocc {
                        for j in 0..nocc {
                            for b in 0..nvir {
                                let t_ij_cb = t2[(i * nvir + c_idx) * nov + j * nvir + b];
                                let eri_ikjb = ooov[(i * nocc + k, j * nvir + b)];
                                let eri_ibjk = ooov[(j * nocc + k, i * nvir + b)];
                                grad_ck -= t_ij_cb * (2.0 * eri_ikjb - eri_ibjk);
                            }
                        }
                    }

                    // Term 4: delta_{bc} -> b=c, sum over i,j,a
                    // -2 * sum_{ija} t_{ij,ac} * [2*(ia|jk) - (ik|ja)]
                    // (ia|jk) = (jk|ia) = ooov[j*nocc+k, i*nvir+a];
                    // (ik|ja) = ooov[i*nocc+k, j*nvir+a]
                    for i in 0..nocc {
                        for j in 0..nocc {
                            for a in 0..nvir {
                                let t_ij_ac = t2[(i * nvir + a) * nov + j * nvir + c_idx];
                                let eri_iajk = ooov[(j * nocc + k, i * nvir + a)];
                                let eri_ikja = ooov[(i * nocc + k, j * nvir + a)];
                                grad_ck -= t_ij_ac * (2.0 * eri_iajk - eri_ikja);
                            }
                        }
                    }

                    *row_k = -2.0 * grad_ck;
                }
                (c_idx, row)
            })
            .collect();

        for (c_idx, row) in panel_rows {
            for (k, &val) in row.iter().enumerate() {
                g[(c_idx, k)] += val;
            }
        }
        c0 = c1;
    }

    g
}

/// Cayley transform for orbital rotation.
///
/// Given antisymmetric kappa (nmo x nmo), compute U = (I + kappa/2)^{-1} (I - kappa/2).
/// U is exactly unitary for any antisymmetric kappa.
fn cayley_rotation(kappa: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let n = kappa.nrows();
    let eye = Array2::eye(n);
    let half_k = 0.5 * kappa;
    let lhs = &eye + &half_k; // I + kappa/2
    let rhs = &eye - &half_k; // I - kappa/2

    // Solve (I + kappa/2) U = (I - kappa/2) for U, column by column
    let mut u = Array2::zeros((n, n));
    for col in 0..n {
        let rhs_col = rhs.column(col).to_owned();
        let u_col = lhs
            .solve(&rhs_col)
            .map_err(|e| FerricError::Lapack(format!("Cayley solve col {col}: {e}")))?;
        u.column_mut(col).assign(&u_col);
    }
    Ok(u)
}

/// Run OO-RI-MP2.
///
/// Starting from converged RHF orbitals, iteratively optimize MO coefficients
/// to minimize E_HF + E_MP2 jointly.
pub fn oo_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    config: &OoRiMp2Config,
) -> Result<OoRiMp2Result, FerricError> {
    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let nocc = active_occ(nocc_total, config.frozen_core)?;
    let first_occ = config.frozen_core;
    let nvir = nbas - nocc_total;
    let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);

    // One-electron integrals (fixed)
    let h = oneelectron::hcore(obs);

    // AO-side invariants: built once, reused every iteration + backtrack.
    // Thread the config budget (M1 resolver) rather than the env-only default.
    let ao = OoRiMp2AoTensors::build_with_budget(
        obs,
        dfbs,
        op,
        ferric_core::memory::resolve_budget_bytes(config.memory_budget_bytes),
    )?;

    // Start from converged RHF orbitals
    let mut c = rhf.mos_r().clone();

    // Initial energies
    let (mut e_hf, mut f_ao, _d) = compute_hf_energy(mol, obs, bounds, &c, nocc_total, &h)?;
    let mut eps = orbital_energies(&c, &f_ao);
    let (mut e_mp2, mut b_ov) = compute_rimp2_with_orbitals(&ao, &c, &eps, &orb)?;
    let mut total_energy = e_hf + e_mp2;
    let mut grad_norm = f64::MAX;
    let nmo = nbas;
    let max_kappa = 0.3; // cap individual rotation angles (radians)

    // DIIS for orbital rotation extrapolation.
    //
    // We apply DIIS to the MO coefficient matrix C itself (the state variable),
    // using the orbital gradient mapped into the AO basis as the error vector.
    // This mirrors how SCF DIIS works: the Fock matrix is replaced by C, and
    // the commutator error is replaced by the orbital gradient.  The gradient
    // is mapped to a (nbas, nbas) matrix G_AO = C * g_full * C^T where g_full
    // is the antisymmetric gradient in the MO basis (g_full[a,i] = g[a,i],
    // g_full[i,a] = -g[a,i]).  This ensures the DIIS error has the same shape
    // as the trial vector (C).
    let mut diis = if config.use_diis {
        Some(Diis::new(config.diis_size))
    } else {
        None
    };

    for iter in 1..=config.max_iter {
        // Compute full-MO B tensor for gradient evaluation
        let b_full = compute_b_full_mo_with(&ao, &c)?;

        // Build t2 amplitudes
        let naux = dfbs.nbasis();
        let (t2, _eri_ov) = compute_t2_and_integrals(
            &b_ov, &eps, nocc, nvir, nocc_total, first_occ, naux,
        );

        // Fock matrix in MO basis
        let f_mo = c.t().dot(&f_ao).dot(&c);

        // Orbital gradient g_{ai} with full 2e response terms
        let g = compute_orbital_gradient(
            &f_mo, &t2, &b_full, nocc, nvir, first_occ, nocc_total,
        );

        // Check gradient norm
        grad_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();

        eprintln!(
            "OO-RI-MP2 iter {:3}: E_HF={:.10} E_MP2={:.10} E_tot={:.10} |g|={:.2e}",
            iter, e_hf, e_mp2, total_energy, grad_norm
        );

        if grad_norm < config.grad_conv {
            return Ok(OoRiMp2Result {
                total_energy,
                hf_energy: e_hf,
                mp2_corr: e_mp2,
                converged: true,
                iterations: iter,
                grad_norm,
                mos: c,
                orbital_energies: eps,
            });
        }

        // Level-shifted approximate Newton step:
        //   kappa_{ai} = -g_{ai} / (eps_a - eps_i + mu)
        // The diagonal Hessian is the orbital energy gap; the level shift mu
        // regularizes small gaps (Bozkaya & Sherrill, JCP 135, 104103, 2011).
        let mut kappa_ov = Array2::zeros((nvir, nocc));
        for a in 0..nvir {
            for i in 0..nocc {
                let gap = eps[nocc_total + a] - eps[first_occ + i];
                kappa_ov[(a, i)] = -g[(a, i)] / (gap + config.level_shift);
            }
        }

        // Cap the step by scaling uniformly if any element exceeds max_kappa.
        let kappa_max_abs = kappa_ov.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        if kappa_max_abs > max_kappa {
            let scale = max_kappa / kappa_max_abs;
            kappa_ov *= scale;
        }

        // Build full antisymmetric kappa matrix (nmo x nmo) from ov block
        let mut kappa = Array2::zeros((nmo, nmo));
        for a in 0..nvir {
            let a_mo = nocc_total + a;
            for i in 0..nocc {
                let i_mo = first_occ + i;
                kappa[(a_mo, i_mo)] = kappa_ov[(a, i)];
                kappa[(i_mo, a_mo)] = -kappa_ov[(a, i)];
            }
        }

        // Cayley rotation
        let u = cayley_rotation(&kappa)?;
        let mut c_new = c.dot(&u);

        // DIIS extrapolation on the MO coefficients.
        // Error vector: map the orbital gradient to the AO basis as an
        // antisymmetric matrix in MO space, then project to AO:
        //   err_AO = C * g_antisym * C^T
        if let Some(ref mut diis_obj) = diis {
            let mut g_antisym = Array2::zeros((nmo, nmo));
            for a in 0..nvir {
                let a_mo = nocc_total + a;
                for i in 0..nocc {
                    let i_mo = first_occ + i;
                    g_antisym[(a_mo, i_mo)] = g[(a, i)];
                    g_antisym[(i_mo, a_mo)] = -g[(a, i)];
                }
            }
            let err_ao = c_new.dot(&g_antisym).dot(&c_new.t());
            c_new = diis_obj.step(&c_new, &err_ao);
        }

        // Evaluate energy at the new (possibly DIIS-extrapolated) orbitals
        let (ehf, fao, _d) =
            compute_hf_energy(mol, obs, bounds, &c_new, nocc_total, &h)?;
        let epsnew = orbital_energies(&c_new, &fao);
        let (emp2, bov) = compute_rimp2_with_orbitals(&ao, &c_new, &epsnew, &orb)?;
        let total_new = ehf + emp2;
        let de = (total_new - total_energy).abs();

        // Backtracking if energy increased by more than a small tolerance.
        // DIIS can produce small uphill steps; we tolerate those.
        if total_new > total_energy + 1e-4 {
            // Fall back to a damped Newton step without DIIS extrapolation.
            let mut bt_kappa_ov = kappa_ov.clone();
            let mut bt_c = c.dot(&u);
            let mut bt_ehf = ehf;
            let mut bt_fao = fao.clone();
            let mut bt_eps = epsnew.clone();
            let mut bt_emp2 = emp2;
            let mut bt_bov = bov.clone();
            let mut bt_total = total_new;

            for _bt in 0..10 {
                bt_kappa_ov *= 0.5;
                let mut k = Array2::zeros((nmo, nmo));
                for a in 0..nvir {
                    let a_mo = nocc_total + a;
                    for i in 0..nocc {
                        let i_mo = first_occ + i;
                        k[(a_mo, i_mo)] = bt_kappa_ov[(a, i)];
                        k[(i_mo, a_mo)] = -bt_kappa_ov[(a, i)];
                    }
                }
                let u2 = cayley_rotation(&k)?;
                bt_c = c.dot(&u2);
                let (eh, fa, _) =
                    compute_hf_energy(mol, obs, bounds, &bt_c, nocc_total, &h)?;
                let en = orbital_energies(&bt_c, &fa);
                let (em, bo) = compute_rimp2_with_orbitals(&ao, &bt_c, &en, &orb)?;
                bt_total = eh + em;
                if bt_total <= total_energy + 1e-12 {
                    bt_ehf = eh;
                    bt_fao = fa;
                    bt_eps = en;
                    bt_emp2 = em;
                    bt_bov = bo;
                    break;
                }
                bt_ehf = eh;
                bt_fao = fa;
                bt_eps = en;
                bt_emp2 = em;
                bt_bov = bo;
            }

            // The backtracking loop commits bt_* to its last trial step on every
            // path (break or exhaustion), so bt_c is always the step to take.
            c = bt_c.clone();
            e_hf = bt_ehf;
            e_mp2 = bt_emp2;
            total_energy = bt_total;
            f_ao = bt_fao;
            eps = bt_eps;
            b_ov = bt_bov;

            // Reset DIIS after backtracking since the extrapolated
            // subspace produced an uphill step.
            if let Some(ref mut diis_obj) = diis {
                diis_obj.reset();
            }
        } else {
            // Accept the (possibly DIIS-extrapolated) step
            c = c_new;
            e_hf = ehf;
            e_mp2 = emp2;
            total_energy = total_new;
            f_ao = fao;
            eps = epsnew;
            b_ov = bov;
        }

        if de < config.energy_conv && iter > 1 {
            // Energy converged; recompute gradient to check convergence.
            let b_full2 = compute_b_full_mo_with(&ao, &c)?;
            let naux2 = dfbs.nbasis();
            let (t2_2, _) = compute_t2_and_integrals(
                &b_ov, &eps, nocc, nvir, nocc_total, first_occ, naux2,
            );
            let f_mo2 = c.t().dot(&f_ao).dot(&c);
            let g2 = compute_orbital_gradient(
                &f_mo2, &t2_2, &b_full2, nocc, nvir, first_occ, nocc_total,
            );
            grad_norm = g2.iter().map(|x| x * x).sum::<f64>().sqrt();

            if grad_norm < config.grad_conv * 10.0 {
                return Ok(OoRiMp2Result {
                    total_energy,
                    hf_energy: e_hf,
                    mp2_corr: e_mp2,
                    converged: true,
                    iterations: iter,
                    grad_norm,
                    mos: c,
                    orbital_energies: eps,
                });
            }
        }
    }

    Ok(OoRiMp2Result {
        total_energy,
        hf_energy: e_hf,
        mp2_corr: e_mp2,
        converged: false,
        iterations: config.max_iter,
        grad_norm,
        mos: c,
        orbital_energies: eps,
    })
}

/// Compute RI-MP2 energy for orbitals rotated by kappa (for finite-difference testing).
///
/// Takes initial MO coefficients, applies a Cayley rotation with the given kappa,
/// rebuilds Fock / density, and returns E_HF + E_MP2.
// System context (mol, two bases, operator, bounds) plus the rotation inputs
// (c_init, kappa) and orbital partition — all distinct, nothing left to bundle.
#[allow(clippy::too_many_arguments)]
pub fn energy_at_kappa(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    c_init: &Array2<f64>,
    kappa: &Array2<f64>,
    orb: &OrbitalSpace,
) -> Result<f64, FerricError> {
    let nocc_total = orb.nocc_total;
    let h = oneelectron::hcore(obs);
    let ao = OoRiMp2AoTensors::build(obs, dfbs, op)?;
    let u = cayley_rotation(kappa)?;
    let c_rot = c_init.dot(&u);
    let (e_hf, f_ao, _) = compute_hf_energy(mol, obs, bounds, &c_rot, nocc_total, &h)?;
    let eps = orbital_energies(&c_rot, &f_ao);
    let (e_mp2, _) = compute_rimp2_with_orbitals(&ao, &c_rot, &eps, orb)?;
    Ok(e_hf + e_mp2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    fn setup_h2() -> (Molecule, PreparedBasis, PreparedBasis, Operator, SchwarzBounds, ScfResult) {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
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
        )
        .unwrap();
        assert!(rhf.converged);
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        (mol, obs, dfbs, op, bounds, rhf)
    }

    #[test]
    fn test_oo_rimp2_lowers_energy() {
        let (mol, obs, dfbs, op, bounds, rhf) = setup_h2();

        // Standard RI-MP2
        let ri_result = crate::rimp2::ri_mp2(
            &mol,
            &obs,
            &dfbs,
            op,
            &rhf,
            &crate::rimp2::RiMp2Config::default(),
        )
        .unwrap();

        // OO-RI-MP2
        let oo_result = oo_ri_mp2(
            &mol,
            &obs,
            &dfbs,
            op,
            &bounds,
            &rhf,
            &OoRiMp2Config::default(),
        )
        .unwrap();

        eprintln!("Standard RI-MP2 total: {:.10}", ri_result.total_energy);
        eprintln!(
            "OO-RI-MP2 total: {:.10} (HF={:.10}, MP2={:.10})",
            oo_result.total_energy, oo_result.hf_energy, oo_result.mp2_corr
        );
        eprintln!(
            "OO converged: {}, iters: {}, |g|: {:.2e}",
            oo_result.converged, oo_result.iterations, oo_result.grad_norm
        );
        eprintln!(
            "Energy lowering: {:.2e}",
            ri_result.total_energy - oo_result.total_energy
        );

        // OO-RI-MP2 total energy should be <= standard RI-MP2 total energy
        // (variational principle for orbital optimization)
        assert!(
            oo_result.total_energy <= ri_result.total_energy + 1e-10,
            "OO total ({:.10}) should be <= RI total ({:.10})",
            oo_result.total_energy,
            ri_result.total_energy
        );
        assert!(oo_result.converged, "OO-RI-MP2 should converge");
    }

    #[test]
    fn test_oo_rimp2_gradient_finite_difference() {
        let (mol, obs, dfbs, op, bounds, rhf) = setup_h2();

        let nbas = obs.nbasis();
        let nelec = mol.nelec() as usize;
        let nocc_total = nelec / 2;
        let nocc = nocc_total;
        let first_occ = 0;
        let nvir = nbas - nocc_total;
        let naux = dfbs.nbasis();
        let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);

        // Compute analytic gradient at RHF orbitals
        let c = rhf.mos_r();
        let h = oneelectron::hcore(&obs);
        let ao = OoRiMp2AoTensors::build(&obs, &dfbs, op).unwrap();
        let (_e_hf, f_ao, _) = compute_hf_energy(&mol, &obs, &bounds, c, nocc_total, &h).unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (_e_mp2, b_ov) = compute_rimp2_with_orbitals(&ao, c, &eps, &orb).unwrap();

        // Build t2 amplitudes
        let (t2, _) = compute_t2_and_integrals(
            &b_ov, &eps, nocc, nvir, nocc_total, first_occ, naux,
        );

        // Build full-MO B tensor for gradient
        let b_full = compute_b_full_mo_with(&ao, c).unwrap();

        let f_mo = c.t().dot(&f_ao).dot(c);
        let g = compute_orbital_gradient(
            &f_mo, &t2, &b_full, nocc, nvir, first_occ, nocc_total,
        );

        // Finite difference check for each (a, i) component
        let delta = 1e-5;
        let mut max_err = 0.0f64;
        for a in 0..nvir {
            let a_mo = nocc_total + a;
            for i in 0..nocc {
                let i_mo = first_occ + i;

                // kappa+ : perturb (a,i) by +delta
                let mut kappa_plus = Array2::zeros((nbas, nbas));
                kappa_plus[(a_mo, i_mo)] = delta;
                kappa_plus[(i_mo, a_mo)] = -delta;

                let e_plus = energy_at_kappa(
                    &mol, &obs, &dfbs, op, &bounds, c, &kappa_plus, &orb,
                )
                .unwrap();

                // kappa- : perturb (a,i) by -delta
                let mut kappa_minus = Array2::zeros((nbas, nbas));
                kappa_minus[(a_mo, i_mo)] = -delta;
                kappa_minus[(i_mo, a_mo)] = delta;

                let e_minus = energy_at_kappa(
                    &mol, &obs, &dfbs, op, &bounds, c, &kappa_minus, &orb,
                )
                .unwrap();

                let fd_grad = (e_plus - e_minus) / (2.0 * delta);
                let analytic = g[(a, i)];
                let err = (fd_grad - analytic).abs();
                max_err = max_err.max(err);

                eprintln!(
                    "grad[a={},i={}]: analytic={:+.8e}, FD={:+.8e}, err={:.2e}",
                    a, i, analytic, fd_grad, err
                );
            }
        }

        eprintln!("Max gradient error: {:.2e}", max_err);
        // Allow generous tolerance for FD vs analytic (FD has O(delta^2) truncation
        // plus numerical noise from integral recomputation)
        assert!(
            max_err < 1e-3,
            "Gradient FD check failed: max_err={:.2e}",
            max_err
        );
    }

    #[test]
    fn test_oo_rimp2_h2o_ccpvdz() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
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
        )
        .unwrap();
        assert!(rhf.converged);
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();

        let ri = crate::rimp2::ri_mp2(
            &mol,
            &obs,
            &dfbs,
            op,
            &rhf,
            &crate::rimp2::RiMp2Config::default(),
        )
        .unwrap();

        let config = OoRiMp2Config::default();
        let oo = oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();

        eprintln!("H2O RI-MP2 total:    {:.10}", ri.total_energy);
        eprintln!(
            "H2O OO-RI-MP2 total: {:.10} (HF={:.10}, MP2={:.10})",
            oo.total_energy, oo.hf_energy, oo.mp2_corr
        );
        eprintln!(
            "OO converged: {}, iters: {}, |g|: {:.2e}",
            oo.converged, oo.iterations, oo.grad_norm
        );

        assert!(
            oo.converged,
            "OO-RI-MP2 H2O did not converge: {} iters, |g|={:.2e}",
            oo.iterations, oo.grad_norm
        );
        assert!(
            oo.total_energy <= ri.total_energy + 1e-10,
            "OO={:.10} should be <= RI={:.10}",
            oo.total_energy, ri.total_energy
        );
    }

    /// The aux-blocked (disk-spill) path through the ThreeIndexSource must give
    /// bit-comparable results to the in-core path: b_full, the OV-dressed MP2
    /// energy, and the orbital gradient all agree to machine precision.
    #[test]
    fn test_spill_budget_paths_match_incore() {
        let (mol, obs, dfbs, op, bounds, rhf) = setup_h2();
        let nbas = obs.nbasis();
        let nocc_total = (mol.nelec() as usize) / 2;
        let nvir = nbas - nocc_total;
        let orb = OrbitalSpace::new(nocc_total, nvir, nocc_total, 0);
        let c = rhf.mos_r();
        let h = oneelectron::hcore(&obs);

        // In-core reference (unlimited budget).
        let ao_ref = OoRiMp2AoTensors::build_with_budget(&obs, &dfbs, op, usize::MAX).unwrap();
        assert_eq!(ao_ref.eri3_ao.borrow().n_blocks(), 1);
        // Tiny budget: ~3 aux rows per block, forces disk spill + many blocks.
        let tiny = obs.nbasis() * obs.nbasis() * 8 * 3;
        let ao_spill = OoRiMp2AoTensors::build_with_budget(&obs, &dfbs, op, tiny).unwrap();
        assert!(
            ao_spill.eri3_ao.borrow().n_blocks() > 1,
            "expected multi-block spill, got {}",
            ao_spill.eri3_ao.borrow().n_blocks()
        );

        // b_full identical.
        let b_ref = compute_b_full_mo_with(&ao_ref, c).unwrap();
        let b_spill = compute_b_full_mo_with(&ao_spill, c).unwrap();
        let maxdiff = (&b_ref - &b_spill).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(maxdiff < 1e-12, "b_full spill vs in-core maxdiff={maxdiff:.2e}");

        // MP2 energy + b_ov identical.
        let (_e_hf, f_ao, _) = compute_hf_energy(&mol, &obs, &bounds, c, nocc_total, &h).unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (e_ref, bov_ref) = compute_rimp2_with_orbitals(&ao_ref, c, &eps, &orb).unwrap();
        let (e_spill, bov_spill) = compute_rimp2_with_orbitals(&ao_spill, c, &eps, &orb).unwrap();
        assert!((e_ref - e_spill).abs() < 1e-12, "E_MP2 spill vs in-core: {:.3e}", (e_ref - e_spill).abs());
        let bovdiff = (&bov_ref - &bov_spill).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(bovdiff < 1e-12, "b_ov spill vs in-core maxdiff={bovdiff:.2e}");
    }

    /// The VVOV c-panelled gradient must be exact for any panel width:
    /// panel_c = 1 (max blocking) vs panel_c = nvir (single panel, the former
    /// unblocked path).
    #[test]
    fn test_vvov_panelled_gradient_exact() {
        let (mol, obs, dfbs, op, bounds, rhf) = setup_h2();
        let nbas = obs.nbasis();
        let nocc_total = (mol.nelec() as usize) / 2;
        let nocc = nocc_total;
        let first_occ = 0;
        let nvir = nbas - nocc_total;
        let naux = dfbs.nbasis();
        let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);
        let c = rhf.mos_r();
        let h = oneelectron::hcore(&obs);
        let ao = OoRiMp2AoTensors::build(&obs, &dfbs, op).unwrap();
        let (_e_hf, f_ao, _) = compute_hf_energy(&mol, &obs, &bounds, c, nocc_total, &h).unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (_e, b_ov) = compute_rimp2_with_orbitals(&ao, c, &eps, &orb).unwrap();
        let (t2, _) = compute_t2_and_integrals(&b_ov, &eps, nocc, nvir, nocc_total, first_occ, naux);
        let b_full = compute_b_full_mo_with(&ao, c).unwrap();
        let f_mo = c.t().dot(&f_ao).dot(c);

        let g_full = compute_orbital_gradient_panelled(
            &f_mo, &t2, &b_full, nocc, nvir, first_occ, nocc_total, nvir,
        );
        for panel in [1usize, 2, 3] {
            let g_p = compute_orbital_gradient_panelled(
                &f_mo, &t2, &b_full, nocc, nvir, first_occ, nocc_total, panel,
            );
            let maxdiff = (&g_full - &g_p).iter().map(|v| v.abs()).fold(0.0, f64::max);
            assert!(
                maxdiff < 1e-13,
                "panelled gradient (panel_c={panel}) differs from full: {maxdiff:.2e}"
            );
        }
    }

    /// The rayon-parallelized c_idx response-term loop (P6) must produce a
    /// byte-for-byte identical gradient regardless of thread count. Each c_idx
    /// writes only its own row via a disjoint-write collect, so there is no
    /// summation whose order varies with scheduling — bit-identity, not merely
    /// close-agreement, is the correct assertion. Mirrors
    /// `whole_pipeline_rhf_gradient_bit_identical_across_thread_counts` in
    /// ferric-scf/src/rhf.rs.
    #[test]
    fn test_oo_gradient_bit_identical_across_thread_counts() {
        // Water/cc-pVDZ gives nocc=5, nvir=19 — enough c-values for rayon to
        // actually split work across threads, unlike H2 (nvir=9, nocc=1).
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
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
        )
        .unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();

        let nbas = obs.nbasis();
        let nocc_total = (mol.nelec() as usize) / 2;
        let nocc = nocc_total;
        let first_occ = 0;
        let nvir = nbas - nocc_total;
        let naux = dfbs.nbasis();
        let orb = OrbitalSpace::new(nocc, nvir, nocc_total, first_occ);
        let c = rhf.mos_r();
        let h = oneelectron::hcore(&obs);
        let ao = OoRiMp2AoTensors::build(&obs, &dfbs, op).unwrap();
        let (_e_hf, f_ao, _) = compute_hf_energy(&mol, &obs, &bounds, c, nocc_total, &h).unwrap();
        let eps = orbital_energies(c, &f_ao);
        let (_e, b_ov) = compute_rimp2_with_orbitals(&ao, c, &eps, &orb).unwrap();
        let (t2, _) = compute_t2_and_integrals(&b_ov, &eps, nocc, nvir, nocc_total, first_occ, naux);
        let b_full = compute_b_full_mo_with(&ao, c).unwrap();
        let f_mo = c.t().dot(&f_ao).dot(c);

        // Force a multi-panel width (panel_c=3) so the parallel region runs
        // inside more than one GEMM panel, exercising the interaction of
        // panelling and rayon scheduling together.
        let run_with_threads = |n: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
            pool.install(|| {
                compute_orbital_gradient_panelled(
                    &f_mo, &t2, &b_full, nocc, nvir, first_occ, nocc_total, 3,
                )
            })
        };

        let g1 = run_with_threads(1);
        let g4 = run_with_threads(4);
        let g8 = run_with_threads(8);

        for a in 0..nvir {
            for i in 0..nocc {
                assert_eq!(
                    g1[(a, i)].to_bits(),
                    g4[(a, i)].to_bits(),
                    "OO gradient not bit-identical 1 vs 4 threads at (a={a}, i={i}): \
                     1={:.17e} (0x{:016x}), 4={:.17e} (0x{:016x})",
                    g1[(a, i)], g1[(a, i)].to_bits(),
                    g4[(a, i)], g4[(a, i)].to_bits(),
                );
                assert_eq!(
                    g1[(a, i)].to_bits(),
                    g8[(a, i)].to_bits(),
                    "OO gradient not bit-identical 1 vs 8 threads at (a={a}, i={i}): \
                     1={:.17e} (0x{:016x}), 8={:.17e} (0x{:016x})",
                    g1[(a, i)], g1[(a, i)].to_bits(),
                    g8[(a, i)], g8[(a, i)].to_bits(),
                );
            }
        }
    }

    #[test]
    fn test_cayley_is_unitary() {
        let n = 5;
        // Build a random antisymmetric matrix
        let mut kappa = Array2::zeros((n, n));
        let vals = [0.1, -0.2, 0.05, -0.15, 0.3, 0.08, -0.12, 0.25, -0.07, 0.18];
        let mut idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                kappa[(i, j)] = vals[idx % vals.len()];
                kappa[(j, i)] = -vals[idx % vals.len()];
                idx += 1;
            }
        }

        let u = cayley_rotation(&kappa).unwrap();
        // U^T U should be identity
        let utu = u.t().dot(&u);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (utu[(i, j)] - expected).abs() < 1e-12,
                    "U^T U[{},{}] = {}, expected {}",
                    i, j, utu[(i, j)], expected
                );
            }
        }
    }
}
