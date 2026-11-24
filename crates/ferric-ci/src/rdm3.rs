//! Spin-free 3-RDM, the two `(aa|aa)`-contracted 4-RDM intermediates and the
//! hole RDMs of a CI vector: the active-space inputs of strongly contracted
//! NEVPT2.
//!
//! # Conventions
//!
//! `E_pq = Σ_σ a†_{pσ} a_{qσ}`, real CI vector `C`, `(pq|rs)` the active
//! chemist-notation integrals, `N` the active electron count.
//!
//! ```text
//!   dm3[p1,q1,p2,q2,p3,q3] = Σ_{στυ} ⟨a†_{p1σ} a†_{p2τ} a†_{p3υ} a_{q3υ} a_{q2τ} a_{q1σ}⟩
//!   f3ca[a,b,c,d,e,f]      = Σ_{ijk} (ki|fj) ⟨E_ab E_cd E_ej E_ki⟩
//!   f3ac[a,b,c,d,e,f]      = Σ_{ijk} (ik|je) ⟨E_ab E_cd E_jf E_ik⟩
//!   hdm1[p,a]              = Σ_σ ⟨a_{pσ} a†_{aσ}⟩
//!   hdm2[p,q,a,b]          = Σ_{στ} ⟨a_{pσ} a_{qτ} a†_{bτ} a†_{aσ}⟩
//!   hdm3[p,q,r,a,b,c]      = Σ_{στυ} ⟨a_{pσ} a_{qτ} a_{rυ} a†_{cυ} a†_{bτ} a†_{aσ}⟩
//! ```
//!
//! `dm3` is PySCF's normal-ordered convention (`fci.rdm.make_dm1234` followed
//! by `reorder_dm1234`), the convention of
//! `scripts/prototypes/nevpt2_derivation.md`; [`NevptRdms::dm1`] and
//! [`NevptRdms::dm2`] are those of [`crate::rdm`]. `f3ca` and `f3ac` are
//! PySCF's `mrpt.nevpt2._contract4pdm` with kernels `NEVPTkern_cedf_aedf` and
//! `NEVPTkern_aedf_ecdf` on the chemist `(aa|aa)` block: contractions of the
//! E-product 4-RDM `⟨E E E E⟩`, not of the normal-ordered one. The hole RDMs
//! are PySCF's `mrpt.nevpt2.make_hdm1/2/3` fed with the E-product 1- to
//! 3-RDMs (`fci.rdm.make_dm123` without reordering), which is what PySCF's
//! NEVPT2 driver passes them; their index layout groups the annihilators
//! first, unlike `dm_k`.
//!
//! # Construction
//!
//! With `X_rs = E_rs C` (as in [`crate::rdm`]) and `⟨C|E_ab E_cd = (E_dc E_ba C)ᵀ`:
//!
//! ```text
//!   ⟨E_ab E_cd E_ef⟩ = ⟨E_dc X_ba | X_ef⟩
//!   f3ca[a..f]       = ⟨E_dc X_ba | R_ef⟩,   R_ef  = Σ_j E_ej W_fj
//!   f3ac[a..f]       = ⟨E_dc X_ba | R'_ef⟩,  R'_ef = Σ_j E_jf W_ej
//!   W_fj             = Σ_ki (ki|fj) X_ki
//! ```
//!
//! `X`, `W`, `R`, `R'` are `n²` vectors each over all determinants, built from
//! the single-replacement lists of [`crate::sigma::DirectSigma`] (an entry
//! `E_pq|S⟩ = ε|T⟩` gives `(E_qp Y)(S) += ε Y(T)`), and `W` is one GEMM. The
//! `n⁴` vectors `L_{dc,ba} = E_dc X_ba` exist only for one block of α rows at
//! a time; per block, `Lᵀ [X | R | R']` is one GEMM of shape `n⁴ × 3n²`. No
//! 4-RDM is formed. The block results are summed in block order, so the
//! output does not depend on the rayon thread count.
//!
//! The normal-ordered 3-RDM follows from `G3 = ⟨E_pq E_rs E_tu⟩`
//! (PySCF `reorder_dm123`):
//!
//! ```text
//!   dm3[p,q,r,s,t,u] = G3 − δ_qr dm2[p,s,t,u] − δ_st dm2[p,q,r,u]
//!                         − δ_qt dm2[p,u,r,s] − δ_qr δ_st dm1[u,p].
//! ```
//!
//! A β replacement carries `(−1)^{N_α}` twice in the α-before-β determinant
//! order, so the lists' single-channel phase is the full phase (as in σ1).
//!
//! Cost: `O(N_det · n_exc · n²)` to build the `L` blocks and
//! `O(3 N_det n⁶)` for the GEMMs. Memory: [`nevpt_rdm_memory_elems`](crate::rdm3::nevpt_rdm_memory_elems).

use crate::hamiltonian::DeterminantSpace;
use crate::integrals::ActiveSpaceIntegrals;
use crate::rdm::{rdm_concurrent_blocks, rdm_memory_elems, spin_summed_rdms, BATCH_CHUNKS};
use crate::sigma::DirectSigma;
use crate::strings;
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::FerricError;
use ndarray::{s, Array2, Array4, Array6, ArrayView2};
use rayon::prelude::*;

/// Target number of `f64` elements in one block of `L = E_dc X_ba`
/// (`n⁴ × determinants of the block`), 16 MiB.
pub const L_BLOCK_ELEMS: usize = 1 << 21;

/// The 1- to 3-RDMs and the two contracted 4-RDM intermediates (see the
/// module documentation for every definition).
#[derive(Debug, Clone)]
pub struct NevptRdms {
    /// `dm1[p,q] = ⟨E_pq⟩`.
    pub dm1: Array2<f64>,
    /// `dm2[p,q,r,s] = Σ_στ ⟨a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}⟩`.
    pub dm2: Array4<f64>,
    /// Normal-ordered spin-free 3-RDM. `Σ_x dm3[p,q,r,s,x,x] = (N−2) dm2[p,q,r,s]`.
    pub dm3: Array6<f64>,
    /// `Σ_{ijk} (ki|fj) ⟨E_ab E_cd E_ej E_ki⟩` (PySCF `f3ca`).
    pub f3ca: Array6<f64>,
    /// `Σ_{ijk} (ik|je) ⟨E_ab E_cd E_jf E_ik⟩` (PySCF `f3ac`).
    pub f3ac: Array6<f64>,
}

/// Spin-free hole RDMs (PySCF `make_hdm1/2/3`; see the module documentation).
#[derive(Debug, Clone)]
pub struct HoleRdms {
    /// `hdm1[p,a] = Σ_σ ⟨a_{pσ} a†_{aσ}⟩ = 2δ_pa − dm1[a,p]`.
    pub hdm1: Array2<f64>,
    /// `hdm2[p,q,a,b] = Σ_στ ⟨a_{pσ} a_{qτ} a†_{bτ} a†_{aσ}⟩`.
    pub hdm2: Array4<f64>,
    /// `hdm3[p,q,r,a,b,c] = Σ_στυ ⟨a_{pσ} a_{qτ} a_{rυ} a†_{cυ} a†_{bτ} a†_{aσ}⟩`.
    pub hdm3: Array6<f64>,
}

/// α rows per `L` block for `n_active` orbitals and `n_beta_str` β strings.
pub fn l_block_rows(n_active: usize, n_beta_str: usize) -> usize {
    let dets = (L_BLOCK_ELEMS / n_active.pow(4).max(1)).max(1);
    (dets / n_beta_str.max(1)).max(1)
}

/// Number of `L` blocks built at once on `threads` rayon workers. At least 1.
pub fn nevpt_rdm_concurrent_blocks(
    n_active: usize,
    n_alpha_str: usize,
    n_beta_str: usize,
    threads: usize,
) -> usize {
    let rows = l_block_rows(n_active, n_beta_str);
    let n_chunks = n_alpha_str.div_ceil(rows);
    threads.min(BATCH_CHUNKS).min(n_chunks).max(1)
}

/// Element counts `(per_worker, resident)` of [`nevpt_rdms`], in `f64`
/// units, excluding the 1-/2-RDM build ([`crate::rdm::rdm_memory_elems`],
/// reserved separately) and the excitation lists.
///
/// * per worker: one `L` block (`n⁴ × rows · n_beta_str`) and its result
///   (`n⁴ × 3n²`);
/// * resident: `X`, `W`, `R`, `R'` (`4 n² N_det`), the two block-result
///   accumulators (`3 n⁶`) and the outputs `dm3`, `f3ca`, `f3ac` (`3 n⁶`).
pub fn nevpt_rdm_memory_elems(
    n_active: usize,
    n_alpha_str: usize,
    n_beta_str: usize,
) -> (usize, usize) {
    let n2 = n_active * n_active;
    let n4 = n2.saturating_mul(n2);
    let n6 = n4.saturating_mul(n2);
    let ndet = n_alpha_str.saturating_mul(n_beta_str);
    let rows = l_block_rows(n_active, n_beta_str).min(n_alpha_str.max(1));
    let per_worker = n4
        .saturating_mul(rows.saturating_mul(n_beta_str))
        .saturating_add(n6.saturating_mul(3));
    let resident = n2
        .saturating_mul(ndet)
        .saturating_mul(4)
        .saturating_add(n6.saturating_mul(6));
    (per_worker, resident)
}

/// Single replacements `E_pq|S⟩ = sign|T⟩` of determinant `S = ia·n_β + ib`,
/// as `f(pq, T, sign)`.
#[inline]
fn for_each_replacement(
    direct: &DirectSigma<'_>,
    ia: usize,
    ib: usize,
    mut f: impl FnMut(usize, usize, f64),
) {
    let nb = direct.n_beta_str;
    for e in &direct.alpha[ia] {
        f(e.pq as usize, e.target as usize * nb + ib, e.sign);
    }
    for e in &direct.beta[ib] {
        f(e.pq as usize, ia * nb + e.target as usize, e.sign);
    }
}

/// 1- to 3-RDMs and `f3ca`/`f3ac` of the CI vector `c` (layout of
/// [`crate::CasCiResult::ci_vector`], `ia * n_beta_str + ib`, strings from
/// [`crate::strings::enumerate_strings`]) in the `n_elec = (n_α, n_β)`
/// sector, with `ints.eri` as the `(aa|aa)` block. `c` is used as given:
/// pass a normalized vector.
///
/// The scratch is reserved up front through a [`MemoryPlan`] on
/// `memory_budget_bytes` (resolved as in [`crate::CasCiConfig`]); an
/// oversized request is refused before any allocation.
pub fn nevpt_rdms(
    ints: &ActiveSpaceIntegrals,
    n_elec: (usize, usize),
    c: &[f64],
    memory_budget_bytes: Option<usize>,
) -> Result<NevptRdms, FerricError> {
    let n = ints.n_active;
    let (n_alpha, n_beta) = n_elec;
    if n == 0 || n > strings::MAX_ACTIVE_ORBITALS || n_alpha > n || n_beta > n {
        return Err(FerricError::General(format!(
            "NEVPT RDMs: invalid active space (n_active = {n}, n_elec = ({n_alpha}, {n_beta}))"
        )));
    }
    let n2 = n * n;
    if ints.eri.len() != n2 * n2 {
        return Err(FerricError::General(format!(
            "NEVPT RDMs: eri has length {} but n_active^4 = {}",
            ints.eri.len(),
            n2 * n2
        )));
    }
    let na = strings::count_strings(n, n_alpha).unwrap_or(usize::MAX);
    let nb = strings::count_strings(n, n_beta).unwrap_or(usize::MAX);
    let ndet = na.saturating_mul(nb);
    if c.len() != ndet {
        return Err(FerricError::General(format!(
            "NEVPT RDMs: CI vector has length {} but the space has {ndet} determinants",
            c.len()
        )));
    }

    let threads = rayon::current_num_threads();
    let mut plan = MemoryPlan::from_global_pool(
        memory_budget_bytes,
        format!("NEVPT RDMs (n_active={n}, n_elec=({n_alpha},{n_beta}), N_det={ndet})"),
    );
    plan.reserve_sized(
        "alpha and beta determinant string lists",
        na.saturating_add(nb),
        std::mem::size_of::<u64>(),
        Lifetime::Resident,
        1,
    );
    let n_exc = |k: usize| k * (n - k) + k;
    plan.reserve(
        "direct sigma excitation lists",
        2usize.saturating_mul(
            na.saturating_mul(n_exc(n_alpha))
                .saturating_add(nb.saturating_mul(n_exc(n_beta)).saturating_mul(2)),
        ),
        Lifetime::Resident,
    );
    let (w12, t12, r12) = rdm_memory_elems(n, na, nb);
    plan.reserve_per_worker(
        "RDM block D = E_rs C",
        w12,
        rdm_concurrent_blocks(na, nb, threads),
    );
    plan.reserve("RDM block results (one batch)", t12, Lifetime::Transient);
    plan.reserve("RDM accumulators and dm1/dm2", r12, Lifetime::Resident);
    let (w3, r3) = nevpt_rdm_memory_elems(n, na, nb);
    plan.reserve_per_worker(
        "L = E_dc X_ba block and its n^4 x 3n^2 result",
        w3,
        nevpt_rdm_concurrent_blocks(n, na, nb, threads),
    );
    plan.reserve(
        "X, W, R, R' vectors, accumulators and dm3/f3ca/f3ac",
        r3,
        Lifetime::Resident,
    );
    // `_mem` must stay a named binding: it holds the reservation until the
    // outputs are built.
    let _mem = plan.commit()?;

    let space = DeterminantSpace {
        alpha_strings: strings::enumerate_strings(n, n_alpha),
        beta_strings: strings::enumerate_strings(n, n_beta),
    };
    let direct = DirectSigma::new(ints, &space);
    let low = spin_summed_rdms(&direct, c)?;

    // X[S, qp] = (E_qp C)(S): an entry E_pq|S⟩ = ε|T⟩ gives ⟨S|E_qp|T⟩ = ε.
    let mut x = Array2::<f64>::zeros((ndet, n2));
    x.as_slice_mut()
        .expect("standard layout")
        .par_chunks_mut(nb * n2)
        .enumerate()
        .for_each(|(ia, rows)| {
            for ib in 0..nb {
                let row = &mut rows[ib * n2..(ib + 1) * n2];
                for_each_replacement(&direct, ia, ib, |pq, t, sign| {
                    let (p, q) = (pq / n, pq % n);
                    row[q * n + p] += sign * c[t];
                });
            }
        });
    // W[S, fj] = Σ_ki (ki|fj) X[S, ki].
    let g = ArrayView2::from_shape((n2, n2), &ints.eri).expect("n^2 x n^2 eri");
    // `w` is read as a row-major slice below; `dot` does not promise that
    // layout for every operand shape.
    let w = x.dot(&g).as_standard_layout().into_owned();
    // RR[S, ef] = Σ_j (E_ej W_fj)(S) and RR[S, n² + ef] = Σ_j (E_jf W_ej)(S).
    let mut rr = Array2::<f64>::zeros((ndet, 2 * n2));
    {
        let ws = w.as_slice().expect("standard layout");
        rr.as_slice_mut()
            .expect("standard layout")
            .par_chunks_mut(nb * 2 * n2)
            .enumerate()
            .for_each(|(ia, rows)| {
                for ib in 0..nb {
                    let row = &mut rows[ib * 2 * n2..(ib + 1) * 2 * n2];
                    for_each_replacement(&direct, ia, ib, |pq, t, sign| {
                        let (p, q) = (pq / n, pq % n);
                        let wt = &ws[t * n2..(t + 1) * n2];
                        // E_qp with q = e, p = j: R[e = q, f] += ε W[f, p].
                        for f in 0..n {
                            row[q * n + f] += sign * wt[f * n + p];
                        }
                        // E_qp with q = j, p = f: R'[e, f = p] += ε W[e, q].
                        for e in 0..n {
                            row[n2 + e * n + p] += sign * wt[e * n + q];
                        }
                    });
                }
            });
    }
    drop(w);

    let rows = l_block_rows(n, nb);
    let n_chunks = na.div_ceil(rows);
    let batch = nevpt_rdm_concurrent_blocks(n, na, nb, threads);
    let mut g3 = Array2::<f64>::zeros((n2 * n2, n2));
    let mut f3 = Array2::<f64>::zeros((n2 * n2, 2 * n2));
    let chunk_ids: Vec<usize> = (0..n_chunks).collect();
    for ids in chunk_ids.chunks(batch) {
        let parts: Vec<(Array2<f64>, Array2<f64>)> = ids
            .par_iter()
            .map(|&k| l_block(&direct, &x, &rr, k * rows, ((k + 1) * rows).min(na)))
            .collect();
        for (gk, fk) in parts {
            g3 += &gk;
            f3 += &fk;
        }
    }
    drop(x);
    drop(rr);

    let dm1 = low.dm1;
    let dm2 = low.dm2;
    let lrow = |a: usize, b: usize, c: usize, d: usize| ((d * n + c) * n + b) * n + a;
    let mut dm3 = Array6::<f64>::zeros((n, n, n, n, n, n));
    let mut f3ca = Array6::<f64>::zeros((n, n, n, n, n, n));
    let mut f3ac = Array6::<f64>::zeros((n, n, n, n, n, n));
    for ((p, q, r, s_, t, u), v) in dm3.indexed_iter_mut() {
        let row = lrow(p, q, r, s_);
        let mut val = g3[[row, t * n + u]];
        if q == r {
            val -= dm2[[p, s_, t, u]];
        }
        if s_ == t {
            val -= dm2[[p, q, r, u]];
        }
        if q == t {
            val -= dm2[[p, u, r, s_]];
        }
        if q == r && s_ == t {
            val -= dm1[[u, p]];
        }
        *v = val;
    }
    for ((a, b, c, d, e, f), v) in f3ca.indexed_iter_mut() {
        *v = f3[[lrow(a, b, c, d), e * n + f]];
    }
    for ((a, b, c, d, e, f), v) in f3ac.indexed_iter_mut() {
        *v = f3[[lrow(a, b, c, d), n2 + e * n + f]];
    }
    Ok(NevptRdms {
        dm1,
        dm2,
        dm3,
        f3ca,
        f3ac,
    })
}

/// `(Lᵀ X, Lᵀ RR)` over the α rows `a0..a1`, with
/// `L[S, (qp)(rs)] = (E_qp X_rs)(S)`.
fn l_block(
    direct: &DirectSigma<'_>,
    x: &Array2<f64>,
    rr: &Array2<f64>,
    a0: usize,
    a1: usize,
) -> (Array2<f64>, Array2<f64>) {
    let n = direct.n_active();
    let n2 = n * n;
    let nb = direct.n_beta_str;
    let len = (a1 - a0) * nb;
    let xs = x.as_slice().expect("standard layout");
    let mut l = Array2::<f64>::zeros((len, n2 * n2));
    {
        let ls = l.as_slice_mut().expect("standard layout");
        for ia in a0..a1 {
            for ib in 0..nb {
                let off = ((ia - a0) * nb + ib) * n2 * n2;
                for_each_replacement(direct, ia, ib, |pq, t, sign| {
                    let (p, q) = (pq / n, pq % n);
                    let dst = &mut ls[off + (q * n + p) * n2..off + (q * n + p + 1) * n2];
                    for (o, &v) in dst.iter_mut().zip(&xs[t * n2..(t + 1) * n2]) {
                        *o += sign * v;
                    }
                });
            }
        }
    }
    let d0 = a0 * nb;
    let gk = l.t().dot(&x.slice(s![d0..d0 + len, ..]));
    let fk = l.t().dot(&rr.slice(s![d0..d0 + len, ..]));
    (gk, fk)
}

/// Hole RDMs from the normal-ordered `dm1`, `dm2`, `dm3` of [`NevptRdms`],
/// by PySCF's `make_hdm1/2/3` on the E-product RDMs
/// (`G2[p,q,r,s] = dm2 + δ_qr dm1[p,s]`, `G3` the inverse of the reordering
/// in the module documentation).
///
/// Memory: `G2`, `G3`, `hdm2`, `hdm3` (`2 n⁴ + 2 n⁶` doubles), reserved
/// through a [`MemoryPlan`] on `memory_budget_bytes`.
pub fn hole_rdms(
    dm1: &Array2<f64>,
    dm2: &Array4<f64>,
    dm3: &Array6<f64>,
    memory_budget_bytes: Option<usize>,
) -> Result<HoleRdms, FerricError> {
    let n = dm1.nrows();
    if dm1.dim() != (n, n) || dm2.dim() != (n, n, n, n) || dm3.dim() != (n, n, n, n, n, n) {
        return Err(FerricError::General(format!(
            "hole RDMs: shapes dm1 {:?}, dm2 {:?}, dm3 {:?} disagree",
            dm1.dim(),
            dm2.dim(),
            dm3.dim()
        )));
    }
    let n4 = n.pow(4);
    let mut plan =
        MemoryPlan::from_global_pool(memory_budget_bytes, format!("hole RDMs (n_active={n})"));
    plan.reserve(
        "E-product G2, G3 and hdm1..hdm3",
        n4.saturating_mul(2)
            .saturating_add(n4.saturating_mul(n * n).saturating_mul(2))
            .saturating_add(n * n),
        Lifetime::Resident,
    );
    let _mem = plan.commit()?;

    let (g2, g3) = e_product_rdms(dm1, dm2, dm3);
    let dl = |i: usize, j: usize| if i == j { 1.0 } else { 0.0 };

    // make_hdm1: 2δ − dm1ᵀ.
    let mut hdm1 = Array2::<f64>::zeros((n, n));
    for ((p, a), v) in hdm1.indexed_iter_mut() {
        *v = 2.0 * dl(p, a) - dm1[[a, p]];
    }
    // make_hdm2 with t[i,j,k,l] = G2[i,k,j,l] − δ_jk dm1[i,l].
    let t2 = |i: usize, j: usize, k: usize, l: usize| g2[[i, k, j, l]] - dl(j, k) * dm1[[i, l]];
    let mut hdm2 = Array4::<f64>::zeros((n, n, n, n));
    for ((i, j, k, l), v) in hdm2.indexed_iter_mut() {
        *v = t2(k, l, i, j) + dl(i, l) * dm1[[k, j]] + dl(j, k) * dm1[[l, i]]
            - 2.0 * dl(i, k) * dm1[[l, j]]
            - 2.0 * dl(j, l) * dm1[[k, i]]
            - 2.0 * dl(i, l) * dl(j, k)
            + 4.0 * dl(i, k) * dl(j, l);
    }
    // make_hdm3.
    let mut hdm3 = Array6::<f64>::zeros((n, n, n, n, n, n));
    for ((p, q, r, a, b, c), v) in hdm3.indexed_iter_mut() {
        *v = -dl(p, b) * hdm2[[q, r, a, c]] - dl(b, r) * hdm2[[p, q, a, c]]
            + 2.0 * dl(b, q) * hdm2[[p, r, a, c]]
            + 2.0 * dl(a, p) * g2[[b, q, c, r]]
            - 4.0 * dl(a, p) * dl(c, r) * dm1[[b, q]]
            + 2.0 * dl(c, r) * g2[[b, q, a, p]]
            - g3[[b, q, a, p, c, r]]
            + 2.0 * dl(a, r) * dl(p, c) * dm1[[b, q]]
            - dl(a, r) * g2[[b, q, c, p]];
    }
    Ok(HoleRdms { hdm1, hdm2, hdm3 })
}

/// E-product RDMs `G2[p,q,r,s] = ⟨E_pq E_rs⟩` and
/// `G3[p,q,r,s,t,u] = ⟨E_pq E_rs E_tu⟩` (PySCF `fci.rdm.make_dm123` without
/// `reorder_dm123`) from the normal-ordered `dm1`, `dm2`, `dm3`:
///
/// ```text
///   G2 = dm2[p,q,r,s] + δ_qr dm1[p,s]
///   G3 = dm3 + δ_qr dm2[p,s,t,u] + δ_st dm2[p,q,r,u] + δ_qt dm2[p,u,r,s] + δ_qr δ_st dm1[u,p]
/// ```
///
/// (the inverse of the reordering in the module documentation). Shapes are
/// not checked here; [`hole_rdms`] checks them.
pub fn e_product_rdms(
    dm1: &Array2<f64>,
    dm2: &Array4<f64>,
    dm3: &Array6<f64>,
) -> (Array4<f64>, Array6<f64>) {
    let n = dm1.nrows();
    let dl = |i: usize, j: usize| if i == j { 1.0 } else { 0.0 };
    let mut g2 = Array4::<f64>::zeros((n, n, n, n));
    for ((p, q, r, s_), v) in g2.indexed_iter_mut() {
        *v = dm2[[p, q, r, s_]] + dl(q, r) * dm1[[p, s_]];
    }
    let mut g3 = Array6::<f64>::zeros((n, n, n, n, n, n));
    for ((p, q, r, s_, t, u), v) in g3.indexed_iter_mut() {
        let mut val = dm3[[p, q, r, s_, t, u]];
        if q == r {
            val += dm2[[p, s_, t, u]];
        }
        if s_ == t {
            val += dm2[[p, q, r, u]];
        }
        if q == t {
            val += dm2[[p, u, r, s_]];
        }
        if q == r && s_ == t {
            val += dm1[[u, p]];
        }
        *v = val;
    }
    (g2, g3)
}
