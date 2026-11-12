//! Spin-summed one- and two-particle reduced density matrices of a CI vector.
//!
//! Definitions (`E_pq = Σ_σ a†_{pσ} a_{qσ}`, real CI vector `C`, active
//! orbital indices; the convention of PySCF `fci.direct_spin1.make_rdm12`):
//!
//! ```text
//!   dm1[p,q]     = ⟨C| E_pq |C⟩
//!   dm2[p,q,r,s] = Σ_στ ⟨C| a†_{pσ} a†_{rτ} a_{sτ} a_{qσ} |C⟩
//!                = ⟨C| E_pq E_rs |C⟩ − δ_qr dm1[p,s]
//!   E_active     = Σ_pq h_pq dm1[p,q] + ½ Σ_pqrs (pq|rs) dm2[p,q,r,s]
//! ```
//!
//! with `h` and `(pq|rs)` the active-space integrals of
//! [`crate::integrals::ActiveSpaceIntegrals`] (inactive-core potential folded
//! into `h`), so `E_core + E_active` is the CI energy.
//!
//! Construction. For every pair `rs` the vector `D_rs = E_rs C` is built from
//! the single-replacement lists of [`crate::sigma::DirectSigma`]: a list entry
//! `E_pq|I⟩ = ε|T⟩` gives `⟨I|E_qp|T⟩ = ε`, so
//!
//! ```text
//!   D_qp(Iα, Iβ) += ε C(T, Iβ)    for each α entry of Iα,
//!   D_qp(Iα, Iβ) += ε C(Iα, T)    for each β entry of Iβ.
//! ```
//!
//! A β operator pair carries `(−1)^{N_α}` twice in the α-before-β order, so the
//! β phase is the single-channel one (as in σ1). Then
//!
//! ```text
//!   dm1[r,s]          = Σ_I C(I) D_rs(I)
//!   ⟨C|E_pq E_rs|C⟩   = Σ_I D_qp(I) D_rs(I)        (E_pq† = E_qp)
//! ```
//!
//! The second line is one GEMM `D Dᵀ` of shape `n² × n²`. The α-α, β-β and
//! α-β parts of the 2-RDM are all inside it: `D = D^α + D^β`, and the cross
//! products `D^α D^βᵀ` are the α-β terms. No determinant pair is visited.
//!
//! The determinants are processed in fixed blocks of α rows (about
//! [`crate::rdm::CHUNK_DETS`] determinants each, independent of the thread count); each
//! block's `D` is built, contracted and dropped. Block results are summed in
//! block order, so the RDMs do not depend on the rayon thread count.
//!
//! Cost: `O(N_det · (n_exc_α + n_exc_β))` to build `D`, `O(N_det · n⁴)` for
//! the GEMM. Memory: one `n² × block` `D` per worker, [`crate::rdm::BATCH_CHUNKS`] block
//! results of `n⁴ + n²` doubles, and the outputs; see [`crate::rdm::rdm_memory_elems`].

use crate::integrals::ActiveSpaceIntegrals;
use crate::sigma::DirectSigma;
use ferric_core::FerricError;
use ndarray::{Array1, Array2, Array4, ArrayView1};
use rayon::prelude::*;

/// Target number of determinants per block of α rows.
pub const CHUNK_DETS: usize = 4096;
/// Number of blocks evaluated in parallel before their results are summed.
pub const BATCH_CHUNKS: usize = 64;

/// Spin-summed reduced density matrices over the active orbitals.
#[derive(Debug, Clone)]
pub struct CasCiRdms {
    /// `dm1[p,q] = ⟨E_pq⟩`, shape `(n, n)`. Trace = number of active electrons.
    pub dm1: Array2<f64>,
    /// `dm2[p,q,r,s] = Σ_στ ⟨a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}⟩`, shape
    /// `(n, n, n, n)`. `Σ_pr dm2[p,p,r,r] = N(N−1)`.
    pub dm2: Array4<f64>,
}

/// The active energy rebuilt from the RDMs, with the magnitude needed to
/// bound its rounding error.
#[derive(Debug, Clone, Copy)]
pub struct RdmEnergy {
    /// `E_core + e_one + e_two`.
    pub e_total: f64,
    /// `Σ h_pq dm1[p,q]`.
    pub e_one: f64,
    /// `½ Σ (pq|rs) dm2[p,q,r,s]`.
    pub e_two: f64,
    /// `|E_core| + Σ |h_pq dm1[p,q]| + ½ Σ |(pq|rs) dm2[p,q,r,s]|`: the sum
    /// of absolute terms, which bounds the rounding error of `e_total`
    /// (relative to machine epsilon).
    pub abs_terms: f64,
}

/// α rows per block for `n_beta_str` β strings.
pub fn chunk_rows(n_beta_str: usize) -> usize {
    (CHUNK_DETS / n_beta_str.max(1)).max(1)
}

/// Element counts `(per_worker, transient, resident)` of the RDM build for
/// `n_active` orbitals and the given string counts, in `f64` units.
///
/// * per worker: one block of `D`, `n² × rows · n_beta_str`;
/// * transient: [`crate::rdm::BATCH_CHUNKS`] block results, `n⁴ + n²` each;
/// * resident: the `n⁴` accumulator, `dm2` (`n⁴`) and `dm1` plus its
///   accumulator (`2 n²`).
pub fn rdm_memory_elems(
    n_active: usize,
    n_alpha_str: usize,
    n_beta_str: usize,
) -> (usize, usize, usize) {
    let n2 = n_active * n_active;
    let rows = chunk_rows(n_beta_str).min(n_alpha_str.max(1));
    let per_worker = n2.saturating_mul(rows.saturating_mul(n_beta_str));
    let n_chunks = n_alpha_str.div_ceil(rows.max(1));
    let transient = BATCH_CHUNKS
        .min(n_chunks)
        .saturating_mul(n2.saturating_mul(n2).saturating_add(n2));
    let resident = n2
        .saturating_mul(n2)
        .saturating_mul(2)
        .saturating_add(2 * n2);
    (per_worker, transient, resident)
}

/// Spin-summed 1- and 2-RDM of the CI vector `c` (indexed `ia * n_beta + ib`,
/// the layout of [`crate::CasCiResult::ci_vector`]) over the strings of
/// `direct`. `c` is used as given: pass a normalized vector.
pub fn spin_summed_rdms(direct: &DirectSigma<'_>, c: &[f64]) -> Result<CasCiRdms, FerricError> {
    let n = direct.n_active();
    let n2 = n * n;
    let na = direct.n_alpha_str;
    let nb = direct.n_beta_str;
    if c.len() != na * nb {
        return Err(FerricError::General(format!(
            "CAS-CI RDM: CI vector has length {} but the space has {} determinants",
            c.len(),
            na * nb
        )));
    }
    let rows = chunk_rows(nb);
    let n_chunks = na.div_ceil(rows);
    let mut g = Array2::<f64>::zeros((n2, n2));
    let mut d1 = Array1::<f64>::zeros(n2);
    let chunk_ids: Vec<usize> = (0..n_chunks).collect();
    for batch in chunk_ids.chunks(BATCH_CHUNKS) {
        let parts: Vec<(Array2<f64>, Array1<f64>)> = batch
            .par_iter()
            .map(|&k| block_contribution(direct, c, k * rows, ((k + 1) * rows).min(na)))
            .collect();
        for (gk, dk) in parts {
            g += &gk;
            d1 += &dk;
        }
    }

    let mut dm1 = Array2::<f64>::zeros((n, n));
    for r in 0..n {
        for s in 0..n {
            dm1[[r, s]] = d1[r * n + s];
        }
    }
    let mut dm2 = Array4::<f64>::zeros((n, n, n, n));
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    let mut v = g[[q * n + p, r * n + s]];
                    if q == r {
                        v -= dm1[[p, s]];
                    }
                    dm2[[p, q, r, s]] = v;
                }
            }
        }
    }
    Ok(CasCiRdms { dm1, dm2 })
}

/// `(D Dᵀ, D c)` over the α rows `a0..a1`, with `D[rs, I] = (E_rs C)(I)`.
fn block_contribution(
    direct: &DirectSigma<'_>,
    c: &[f64],
    a0: usize,
    a1: usize,
) -> (Array2<f64>, Array1<f64>) {
    let n = direct.n_active();
    let n2 = n * n;
    let nb = direct.n_beta_str;
    let len = (a1 - a0) * nb;
    let mut d = Array2::<f64>::zeros((n2, len));
    {
        let ds = d
            .as_slice_mut()
            .expect("freshly allocated Array2 is in standard layout");
        for ia in a0..a1 {
            let off = (ia - a0) * nb;
            // α: E_pq|Iα⟩ = ε|T⟩  ⇒  D_qp(Iα, :) += ε C(T, :).
            for e in &direct.alpha[ia] {
                let pq = e.pq as usize;
                let (p, q) = (pq / n, pq % n);
                let base = (q * n + p) * len + off;
                let ct = &c[e.target as usize * nb..(e.target as usize + 1) * nb];
                for (o, &x) in ds[base..base + nb].iter_mut().zip(ct) {
                    *o += e.sign * x;
                }
            }
            // β: E_pq|Iβ⟩ = ε|T⟩  ⇒  D_qp(Iα, Iβ) += ε C(Iα, T).
            let crow = &c[ia * nb..(ia + 1) * nb];
            for (ib, list) in direct.beta.iter().enumerate() {
                for e in list {
                    let pq = e.pq as usize;
                    let (p, q) = (pq / n, pq % n);
                    ds[(q * n + p) * len + off + ib] += e.sign * crow[e.target as usize];
                }
            }
        }
    }
    let cb = ArrayView1::from(&c[a0 * nb..a1 * nb]);
    let gk = d.dot(&d.t());
    let dk = d.dot(&cb);
    (gk, dk)
}

/// `E_core + Σ h dm1 + ½ Σ (pq|rs) dm2` over the active integrals `ints`.
pub fn rdm_energy(ints: &ActiveSpaceIntegrals, rdms: &CasCiRdms) -> RdmEnergy {
    let n = ints.n_active;
    let mut e_one = 0.0;
    let mut abs_terms = ints.e_core.abs();
    for p in 0..n {
        for q in 0..n {
            let t = ints.h[(p, q)] * rdms.dm1[[p, q]];
            e_one += t;
            abs_terms += t.abs();
        }
    }
    let mut e_two = 0.0;
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    let t = 0.5 * ints.g(p, q, r, s) * rdms.dm2[[p, q, r, s]];
                    e_two += t;
                    abs_terms += t.abs();
                }
            }
        }
    }
    RdmEnergy {
        e_total: ints.e_core + e_one + e_two,
        e_one,
        e_two,
        abs_terms,
    }
}
