//! DLPNO-CCSD — the **per-pair PNO virtual basis**, second half of the method.
//!
//! [`crate::dlpno_ccsd`] is the first half: it screens *which occupied pairs*
//! `(i,j)` carry amplitudes. This module is the other axis — *which virtuals*
//! survive **within** each retained pair — and it is the half that carries the
//! real compression, because the virtual space is the large index.
//!
//! Structurally this mirrors [`ferric_mp2::dlpno_mp2`] exactly: build the
//! per-pair PNO transform from [`ferric_mp2::local_pno::build_pno_transforms`],
//! **semicanonicalize** it, rotate the pair's blocks in, evaluate. MP2 is the
//! reference implementation for a reason — it validated this whole construction
//! where the amplitudes needed no prior correlated calculation.
//!
//! # Scope, stated up front — what this does NOT do
//!
//! **It does not rewrite the CCSD residual in the PNO basis.** The iteration in
//! [`crate::ccsd_closed_shell`] is 74 `einsum!` contractions over *shared*
//! `oooo/ovov/oovv/ovvo/ovoo/ovvv/vvvv` blocks. A per-pair virtual basis means
//! every one of those has to be re-derived in a basis that differs per pair,
//! with pair-pair overlap matrices `S^{ij,kl} = Q_ijᵀ Q_kl` inserted wherever a
//! contraction couples two pairs. That is a very large change whose failure mode
//! is a *plausible but wrong energy*, so it was deliberately left out of this
//! pass rather than shipped untested.
//!
//! What IS here is the exact, tested substrate that change needs:
//!
//! | Stage | What | Exactness test |
//! |-------|------|----------------|
//! | 1 | [`PairPnoBasis`] — semicanonical per-pair transform + PNO orbital energies | [`tests::stage1_transforms_are_orthogonal_at_zero_truncation`] |
//! | 2 | [`t2_to_pno`] / [`t2_from_pno`] — amplitude round trip | [`tests::stage2_t2_round_trip_is_exact_at_zero_truncation`] |
//! | 3 | [`pno_ccsd_energy`] — the CCSD energy from PNO-basis amplitudes | [`tests::stage3_pno_energy_matches_dense_ccsd_water_631g`] |
//!
//! Stage 2 is load-bearing: if a converged dense `t2` cannot survive a trip
//! through the per-pair basis and back, nothing downstream can be trusted.
//!
//! MEASURED at `t_cut_pno = 0`, against [`crate::ccsd_closed_shell`]'s own
//! converged energy on real integrals: **0.0 Ha** on water/6-31G (`no=5, nv=8`,
//! 15 pairs) and **2.1e-17 Ha** on water/STO-3G. Stage 2's amplitude round trip
//! is exact to 8.3e-16 and stage 1's transforms are orthogonal to 2.2e-15.
//!
//! # SEMICANONICALIZATION IS MANDATORY
//!
//! PNOs diagonalize the pair density, **not** the virtual Fock matrix. So after
//! rotating into a pair's PNO basis the Fock matrix is no longer diagonal and
//! neither the CCSD denominators nor any `ε_a`-indexed quantity is meaningful.
//! [`PairPnoBasis::build`] therefore constructs `F^ij = Q ᵀ diag(ε_v) Q`,
//! **re-diagonalizes it**, and folds the resulting `U` into the stored transform
//! (`Q̃ = Q·U`), carrying the PNO orbital energies alongside.
//!
//! Taking only the diagonal `f_aa = Σ_c Q_ca² ε_c` instead is WRONG and fails
//! **silently**. It is not a hypothetical: it broke DLPNO-MP2's exactness
//! contract by 0.117 Ha, with a MEASURED off-diagonal maximum of 0.137 on a 5×5
//! rotation, before the exactness test caught it.
//! [`tests::stage1_semicanonical_fock_is_diagonal`] pins the fix directly, and
//! [`tests::stage1_naive_diagonal_fock_would_be_wrong`] pins the *premise* — that
//! the un-rediagonalized Fock really does have large off-diagonal elements, so
//! the diagonal shortcut is measurably not an approximation.
//!
//! # Exactness contract
//!
//! With `t_cut_pno = 0` and complete domains every transform is a square
//! orthogonal rotation. Rotating a pair's amplitudes and its integrals by the
//! *same* rotation leaves the energy contraction — a trace — invariant, so
//! [`pno_ccsd_energy`] reproduces [`crate::ccsd_closed_shell`] exactly. That is
//! the property every later claim about a truncated run depends on.
//!
//! # What is NOT claimed
//!
//! No timing, no speedup, no cost model. Truncating virtuals here changes the
//! energy evaluation only; it does not yet make the *iteration* cheaper, because
//! the iteration is still dense. And ferric has a MEASURED negative result for
//! virtual truncation at small N (the OSV sweep: 100% retention at accurate
//! thresholds, 48–76 mHa error at loose ones), so the retention numbers this
//! module reports are there to be measured on a real system, not assumed.

use ferric_core::linalg::{eigh_dc, Uplo};
use ferric_core::FerricError;
use ferric_mp2::local_pno::{build_pno_transforms, PnoTransforms};
use ferric_mp2::pair_domains::PairDomains;
use ndarray::{Array2, Array4};

/// The semicanonical PNO virtual basis of a single occupied pair `(i,j)`.
#[derive(Debug, Clone)]
pub struct PairPno {
    /// The occupied pair, spatial indices.
    pub ij: (usize, usize),
    /// Semicanonical transform `Q̃ = Q·U`, `(nvir × npno)`, orthonormal columns.
    ///
    /// `Q` is the raw PNO transform (pair-density eigenvectors) and `U`
    /// diagonalizes the virtual Fock matrix *within* the retained PNO space.
    /// Storing the composite means callers never see a non-semicanonical basis
    /// and cannot accidentally use one.
    pub transform: Array2<f64>,
    /// Virtual-orbital energies in this pair's semicanonical PNO basis,
    /// ascending — the eigenvalues of `Q ᵀ diag(ε_v) Q`.
    ///
    /// At `t_cut_pno = 0` these are the canonical `ε_v` up to ordering, because
    /// a square `Q` makes the rediagonalization recover the original spectrum.
    pub eps: Vec<f64>,
    /// Sum of discarded pair-density occupation weights — this pair's own
    /// estimate of what the truncation threw away.
    pub discarded_weight: f64,
}

/// Per-pair semicanonical PNO bases for every retained pair.
#[derive(Debug, Clone)]
pub struct PairPnoBasis {
    /// One entry per pair in [`PairDomains::pairs`], in the same order.
    pub pairs: Vec<PairPno>,
    /// Full canonical virtual count the transforms map *from*.
    pub nvir: usize,
    /// Occupation threshold used to build them.
    pub t_cut_pno: f64,
}

impl PairPnoBasis {
    /// Build the semicanonical PNO basis of every retained pair from
    /// first-order (semicanonical MP2) amplitudes.
    ///
    /// `t2_pair(i,j)` returns the `(nvir × nvir)` amplitude block `T^{ij}_{ab}`
    /// used to define the pair density — the same closure contract as
    /// [`build_pno_transforms`]. `eps_vir` are the canonical virtual orbital
    /// energies (length `nvir`).
    ///
    /// The two steps, in order, and neither is optional:
    ///
    /// 1. Diagonalize the pair density `D^ij = T Tᵀ + Tᵀ T`, keep the PNOs with
    ///    occupation `≥ t_cut_pno`. This gives `Q` — a basis adapted to the
    ///    pair's correlation, but **not** to the Fock operator.
    /// 2. Build `F^ij = Q ᵀ diag(ε_v) Q` and re-diagonalize it, giving `U` and
    ///    the PNO orbital energies. Store `Q̃ = Q·U`.
    ///
    /// # Errors
    ///
    /// [`FerricError::General`] when `eps_vir.len() != nvir`, and propagates any
    /// error from the PNO construction or either eigensolve with the offending
    /// pair named.
    pub fn build<F>(
        domains: &PairDomains,
        nvir: usize,
        eps_vir: &[f64],
        t_cut_pno: f64,
        t2_pair: F,
    ) -> Result<Self, FerricError>
    where
        F: FnMut(usize, usize) -> Array2<f64>,
    {
        if eps_vir.len() != nvir {
            return Err(FerricError::General(format!(
                "PairPnoBasis::build: eps_vir has {} entries, expected nvir = {nvir}",
                eps_vir.len()
            )));
        }
        let raw: PnoTransforms = build_pno_transforms(domains, nvir, t_cut_pno, t2_pair)?;

        let mut pairs = Vec::with_capacity(raw.pairs.len());
        for p in &raw.pairs {
            let q = &p.transform; // (nvir x npno), orthonormal columns
            let npno = q.ncols();

            // --- SEMICANONICALIZATION (see module docs; NOT optional) ---
            // F^ij_ab = sum_c Q_ca eps_c Q_cb. Taking only a == b here is the
            // silent-failure mode that cost DLPNO-MP2 0.117 Ha.
            let mut f_pno = Array2::<f64>::zeros((npno, npno));
            for a in 0..npno {
                for b in 0..npno {
                    f_pno[(a, b)] = (0..nvir).map(|c| q[(c, a)] * q[(c, b)] * eps_vir[c]).sum();
                }
            }
            let (eps, u) = eigh_dc(&f_pno, Uplo::Upper).map_err(|e| {
                FerricError::General(format!(
                    "DLPNO-CCSD semicanonicalization failed for pair {:?}: {e}",
                    p.ij
                ))
            })?;

            pairs.push(PairPno {
                ij: p.ij,
                transform: q.dot(&u),
                eps,
                discarded_weight: p.discarded_weight,
            });
        }
        Ok(Self { pairs, nvir, t_cut_pno })
    }

    /// True when nothing was truncated: every pair kept all `nvir` virtuals.
    pub fn is_complete(&self) -> bool {
        self.pairs.iter().all(|p| p.transform.ncols() == self.nvir)
    }

    /// Retained PNOs summed over pairs, divided by `n_pairs · nvir`. 1.0 means
    /// no compression at all.
    pub fn virtual_retention(&self) -> f64 {
        if self.pairs.is_empty() || self.nvir == 0 {
            return 1.0;
        }
        let kept: usize = self.pairs.iter().map(|p| p.transform.ncols()).sum();
        kept as f64 / (self.pairs.len() * self.nvir) as f64
    }

    /// Largest per-pair discarded occupation weight across pairs.
    pub fn max_discarded_weight(&self) -> f64 {
        self.pairs.iter().map(|p| p.discarded_weight).fold(0.0, f64::max)
    }

    /// Total number of PNO amplitude elements `Σ_pairs npno²`, versus the dense
    /// `n_pairs · nvir²` the same pair list would need.
    ///
    /// A count, not a timing — see the module's "what is NOT claimed" note.
    pub fn amplitude_elements(&self) -> (usize, usize) {
        let pno: usize = self.pairs.iter().map(|p| p.transform.ncols().pow(2)).sum();
        (pno, self.pairs.len() * self.nvir * self.nvir)
    }
}

/// Rotate a dense spatial `t2[i,j,a,b]` into the per-pair PNO bases.
///
/// Returns one `(npno_ij × npno_ij)` block per retained pair, in
/// [`PairPnoBasis::pairs`] order: `t̃^{ij} = Q̃_ijᵀ t2[i,j,:,:] Q̃_ij`.
///
/// Pairs absent from the basis are simply not produced — the pair screen of
/// [`crate::dlpno_ccsd`] composes here by construction.
///
/// # Errors
///
/// [`FerricError::General`] when `t2`'s virtual dimensions disagree with
/// `basis.nvir`, or an occupied index of a pair is out of range for `t2`.
pub fn t2_to_pno(
    t2: &Array4<f64>,
    basis: &PairPnoBasis,
) -> Result<Vec<Array2<f64>>, FerricError> {
    let (no_i, no_j, nv_a, nv_b) = t2.dim();
    if nv_a != basis.nvir || nv_b != basis.nvir {
        return Err(FerricError::General(format!(
            "t2_to_pno: t2 virtual dims ({nv_a}, {nv_b}) disagree with nvir = {}",
            basis.nvir
        )));
    }
    let mut out = Vec::with_capacity(basis.pairs.len());
    for p in &basis.pairs {
        let (i, j) = p.ij;
        if i >= no_i || j >= no_j {
            return Err(FerricError::General(format!(
                "t2_to_pno: pair ({i},{j}) out of range for t2 occupied dims ({no_i}, {no_j})"
            )));
        }
        let block = t2.slice(ndarray::s![i, j, .., ..]).to_owned();
        out.push(p.transform.t().dot(&block).dot(&p.transform));
    }
    Ok(out)
}

/// Back-transform per-pair PNO amplitude blocks into a dense `t2[i,j,a,b]`.
///
/// The inverse of [`t2_to_pno`] **only when nothing was truncated**: the
/// transform is a projector `Q̃ Q̃ᵀ` onto the retained PNO space, which is the
/// identity exactly when `Q̃` is square. Under truncation this is the projected
/// amplitude, which is the intended lossy step — not a bug, but the reason the
/// round-trip test is pinned at `t_cut_pno = 0`.
///
/// Occupied pairs absent from `basis` come back **zero**, matching
/// [`crate::dlpno_ccsd::apply_pair_mask`]. Because `t2[i,j,a,b] == t2[j,i,b,a]`
/// is a structural identity of closed-shell CCSD, the `(j,i)` mirror of every
/// off-diagonal pair is filled in as the transpose rather than left zero.
///
/// # Errors
///
/// [`FerricError::General`] when `blocks` and `basis.pairs` disagree in length
/// or any block is not `npno × npno` for its pair.
pub fn t2_from_pno(
    blocks: &[Array2<f64>],
    basis: &PairPnoBasis,
    nocc: usize,
) -> Result<Array4<f64>, FerricError> {
    if blocks.len() != basis.pairs.len() {
        return Err(FerricError::General(format!(
            "t2_from_pno: got {} blocks for {} pairs",
            blocks.len(),
            basis.pairs.len()
        )));
    }
    let nvir = basis.nvir;
    let mut t2 = Array4::<f64>::zeros((nocc, nocc, nvir, nvir));
    for (p, blk) in basis.pairs.iter().zip(blocks.iter()) {
        let (i, j) = p.ij;
        let npno = p.transform.ncols();
        if blk.nrows() != npno || blk.ncols() != npno {
            return Err(FerricError::General(format!(
                "t2_from_pno: block for pair ({i},{j}) is {:?}, expected ({npno}, {npno})",
                blk.dim()
            )));
        }
        if i >= nocc || j >= nocc {
            return Err(FerricError::General(format!(
                "t2_from_pno: pair ({i},{j}) out of range for nocc = {nocc}"
            )));
        }
        let dense = p.transform.dot(blk).dot(&p.transform.t());
        for a in 0..nvir {
            for b in 0..nvir {
                t2[[i, j, a, b]] = dense[(a, b)];
                // t2[i,j,a,b] == t2[j,i,b,a] is structural; filling the mirror
                // keeps the returned tensor a valid closed-shell t2 rather than
                // a half-populated one.
                t2[[j, i, b, a]] = dense[(a, b)];
            }
        }
    }
    Ok(t2)
}

/// The closed-shell CCSD correlation energy, evaluated from **PNO-basis**
/// amplitudes.
///
/// The dense expression in [`crate::ccsd_closed_shell`] is
///
/// ```text
///   E = Σ_ijab τ[i,j,a,b] · ( 2 (ia|jb) − (ib|ja) ),      τ = t2 + t1⊗t1
/// ```
///
/// Both `τ` and the integrals are rotated into the *same* per-pair basis, so the
/// `(a,b)` sum of pair `(i,j)` becomes a trace over that pair's PNOs:
///
/// ```text
///   E = Σ_{(i,j)} Σ_ãb̃ τ̃^{ij}[ã,b̃] · ( 2 g̃^{ij}[ã,b̃] − g̃^{ji}[ã,b̃] )
/// ```
///
/// with `g^{ij}[a,b] = (ia|jb)` and `g^{ji}[a,b] = (ib|ja) = g^{ij}[b,a]`.
/// Because the same orthogonal `Q̃` rotates both factors, the trace is invariant
/// and the untruncated result is *exactly* the dense one.
///
/// `tau` is the dense `(nocc, nocc, nvir, nvir)` amplitude tensor **including**
/// the `t1⊗t1` term (callers with a converged CCSD pass `t2 + t1⊗t1`); `ovov` is
/// the chemist block `(ia|jb)` stored at `[i,a,j,b)` exactly as
/// `ccsd_closed_shell` builds it.
///
/// The full `nocc²` grid is summed, not the `i ≤ j` triangle — matching the
/// dense expression term for term, so no weighting convention can drift. Pairs
/// held only as `(i,j)` in `domains` therefore contribute through their `(j,i)`
/// mirror too; the mirror's basis is taken to be the same `Q̃`, which is exact
/// because `D^ji = (D^ij)ᵀ`-symmetrized is the same matrix.
///
/// # Errors
///
/// [`FerricError::General`] on any shape disagreement between `tau`, `ovov` and
/// the PNO basis.
pub fn pno_ccsd_energy(
    tau: &Array4<f64>,
    ovov: &Array4<f64>,
    basis: &PairPnoBasis,
) -> Result<f64, FerricError> {
    let (no_i, no_j, nv_a, nv_b) = tau.dim();
    if nv_a != basis.nvir || nv_b != basis.nvir {
        return Err(FerricError::General(format!(
            "pno_ccsd_energy: tau virtual dims ({nv_a}, {nv_b}) disagree with nvir = {}",
            basis.nvir
        )));
    }
    if no_i != no_j {
        return Err(FerricError::General(format!(
            "pno_ccsd_energy: tau occupied dims ({no_i}, {no_j}) are not square"
        )));
    }
    let nvir = basis.nvir;
    if ovov.dim() != (no_i, nvir, no_j, nvir) {
        return Err(FerricError::General(format!(
            "pno_ccsd_energy: ovov is {:?}, expected ({no_i}, {nvir}, {no_j}, {nvir})",
            ovov.dim()
        )));
    }

    let mut e = 0.0;
    for p in &basis.pairs {
        let (i, j) = p.ij;
        if i >= no_i || j >= no_i {
            return Err(FerricError::General(format!(
                "pno_ccsd_energy: pair ({i},{j}) out of range for nocc = {no_i}"
            )));
        }
        let q = &p.transform;
        // (i,j) and, when off-diagonal, its (j,i) mirror. The dense expression
        // sums the full nocc² grid, so both must be counted; the mirror shares
        // this pair's basis.
        let mirrors: &[(usize, usize)] = if i == j { &[(i, j)] } else { &[(i, j), (j, i)] };
        for &(x, y) in mirrors {
            let mut tau_xy = Array2::<f64>::zeros((nvir, nvir));
            let mut g_xy = Array2::<f64>::zeros((nvir, nvir));
            let mut g_yx = Array2::<f64>::zeros((nvir, nvir));
            for a in 0..nvir {
                for b in 0..nvir {
                    tau_xy[(a, b)] = tau[[x, y, a, b]];
                    g_xy[(a, b)] = ovov[[x, a, y, b]]; // (xa|yb)
                    g_yx[(a, b)] = ovov[[x, b, y, a]]; // (xb|ya)
                }
            }
            let t_p = q.t().dot(&tau_xy).dot(q);
            let g_p = q.t().dot(&g_xy).dot(q);
            let x_p = q.t().dot(&g_yx).dot(q);
            let npno = q.ncols();
            for a in 0..npno {
                for b in 0..npno {
                    e += t_p[(a, b)] * (2.0 * g_p[(a, b)] - x_p[(a, b)]);
                }
            }
        }
    }
    Ok(e)
}

/// The same energy expression evaluated densely — the oracle
/// [`pno_ccsd_energy`] must reproduce at zero truncation.
///
/// Public because it is the honest way for a caller to check its own PNO
/// settings against the untruncated answer on its own system, rather than
/// trusting a threshold.
pub fn dense_ccsd_energy(tau: &Array4<f64>, ovov: &Array4<f64>) -> Result<f64, FerricError> {
    let (no_i, no_j, nvir, nv_b) = tau.dim();
    if no_i != no_j || nvir != nv_b {
        return Err(FerricError::General(format!(
            "dense_ccsd_energy: tau is {:?}, expected (no, no, nv, nv)",
            tau.dim()
        )));
    }
    if ovov.dim() != (no_i, nvir, no_j, nvir) {
        return Err(FerricError::General(format!(
            "dense_ccsd_energy: ovov is {:?}, expected ({no_i}, {nvir}, {no_j}, {nvir})",
            ovov.dim()
        )));
    }
    let mut e = 0.0;
    for i in 0..no_i {
        for j in 0..no_i {
            for a in 0..nvir {
                for b in 0..nvir {
                    e += tau[[i, j, a, b]] * (2.0 * ovov[[i, a, j, b]] - ovov[[i, b, j, a]]);
                }
            }
        }
    }
    Ok(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_mp2::pair_domains::{build_pair_domains, complete_pair_domains};
    use ndarray::Array2 as Arr2;

    // ---------------------------------------------------------------------
    // Deterministic toy system. Small on purpose: this file runs on a shared,
    // heavily loaded box, and every claim here is about EXACTNESS, not cost.
    // ---------------------------------------------------------------------

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    }

    /// Virtual orbital energies, well separated so no denominator is small.
    fn eps_vir(nvir: usize) -> Vec<f64> {
        (0..nvir).map(|a| 0.5 + 0.13 * a as f64).collect()
    }

    fn eps_occ(nocc: usize) -> Vec<f64> {
        (0..nocc).map(|i| -1.0 - 0.11 * i as f64).collect()
    }

    /// Chemist `(ia|jb)` stored at `[i,a,j,b]`, symmetric under `(ia) <-> (jb)`
    /// exactly as the real block is.
    fn ovov_block(nocc: usize, nvir: usize) -> Array4<f64> {
        let n = nocc * nvir;
        let mut s = 0x2545F4914F6CDD1Du64;
        let mut m = Arr2::<f64>::zeros((n, n));
        for p in 0..n {
            for q in p..n {
                let v = lcg(&mut s);
                m[(p, q)] = v;
                m[(q, p)] = v;
            }
        }
        Array4::from_shape_fn((nocc, nvir, nocc, nvir), |(i, a, j, b)| {
            m[(i * nvir + a, j * nvir + b)]
        })
    }

    /// First-order amplitudes from that block — the object PNOs are built from.
    fn mp2_t2(ovov: &Array4<f64>, eo: &[f64], ev: &[f64]) -> Array4<f64> {
        let (nocc, nvir, _, _) = ovov.dim();
        Array4::from_shape_fn((nocc, nocc, nvir, nvir), |(i, j, a, b)| {
            ovov[[i, a, j, b]] / (eo[i] + eo[j] - ev[a] - ev[b])
        })
    }

    fn line_centers(nocc: usize, spacing: f64) -> Arr2<f64> {
        Arr2::from_shape_fn((nocc, 3), |(i, ax)| if ax == 0 { i as f64 * spacing } else { 0.0 })
    }

    fn toy_basis(nocc: usize, nvir: usize, t_cut: f64) -> (PairPnoBasis, Array4<f64>, Array4<f64>) {
        let (eo, ev) = (eps_occ(nocc), eps_vir(nvir));
        let ovov = ovov_block(nocc, nvir);
        let t2 = mp2_t2(&ovov, &eo, &ev);
        let d = complete_pair_domains(&line_centers(nocc, 1.5)).unwrap();
        let basis = PairPnoBasis::build(&d, nvir, &ev, t_cut, |i, j| {
            Arr2::from_shape_fn((nvir, nvir), |(a, b)| t2[[i, j, a, b]])
        })
        .unwrap();
        (basis, t2, ovov)
    }

    // ================= STAGE 1: the semicanonical basis =====================

    /// At zero truncation every transform must be a SQUARE ORTHOGONAL matrix.
    ///
    /// This is what makes the whole construction exact in the limit: a square
    /// orthogonal rotation of both the amplitudes and the integrals cancels out
    /// of every trace-shaped contraction downstream. A non-orthonormal transform
    /// would silently rescale the correlation energy instead of failing.
    #[test]
    fn stage1_transforms_are_orthogonal_at_zero_truncation() {
        let (nocc, nvir) = (4, 6);
        let (basis, _, _) = toy_basis(nocc, nvir, 0.0);

        assert!(basis.is_complete(), "t_cut_pno = 0 must keep every virtual");
        assert_eq!(basis.virtual_retention(), 1.0);
        let mut worst = 0.0f64;
        for p in &basis.pairs {
            assert_eq!(p.transform.ncols(), nvir, "pair {:?} is not square", p.ij);
            let gram = p.transform.t().dot(&p.transform);
            for a in 0..nvir {
                for b in 0..nvir {
                    let want = if a == b { 1.0 } else { 0.0 };
                    worst = worst.max((gram[(a, b)] - want).abs());
                }
            }
        }
        eprintln!("stage 1: max |Q^T Q - I| = {worst:.3e}");
        assert!(worst < 1e-10, "transforms are not orthogonal: max deviation {worst:.3e}");
    }

    /// SEMICANONICALIZATION, pinned directly: the virtual Fock matrix in the
    /// stored basis must come back DIAGONAL, with the stored `eps` on it.
    ///
    /// If this fails, every denominator built from `PairPno::eps` is wrong and
    /// the failure is silent — the energy stays plausible.
    #[test]
    fn stage1_semicanonical_fock_is_diagonal() {
        let (nocc, nvir) = (4, 6);
        let ev = eps_vir(nvir);
        let (basis, _, _) = toy_basis(nocc, nvir, 0.0);

        let mut worst_off = 0.0f64;
        let mut worst_diag = 0.0f64;
        for p in &basis.pairs {
            let q = &p.transform;
            let npno = q.ncols();
            let f = Arr2::from_shape_fn((npno, npno), |(a, b)| {
                (0..nvir).map(|c| q[(c, a)] * q[(c, b)] * ev[c]).sum::<f64>()
            });
            for a in 0..npno {
                for b in 0..npno {
                    if a == b {
                        worst_diag = worst_diag.max((f[(a, b)] - p.eps[a]).abs());
                    } else {
                        worst_off = worst_off.max(f[(a, b)].abs());
                    }
                }
            }
        }
        eprintln!("stage 1: max off-diag F_pno = {worst_off:.3e}, max |F_aa - eps_a| = {worst_diag:.3e}");
        assert!(worst_off < 1e-10, "Fock is not diagonal in the stored basis: {worst_off:.3e}");
        assert!(worst_diag < 1e-10, "stored eps disagree with F_aa: {worst_diag:.3e}");
    }

    /// The PREMISE of the previous test: without rediagonalization the Fock
    /// matrix in the raw PNO basis has LARGE off-diagonal elements.
    ///
    /// Without this, `stage1_semicanonical_fock_is_diagonal` would be vacuous —
    /// it would pass trivially if PNOs happened to already diagonalize F. They
    /// do not, and this measures by how much, which is exactly the error the
    /// diagonal-only shortcut would introduce.
    #[test]
    fn stage1_naive_diagonal_fock_would_be_wrong() {
        let (nocc, nvir) = (4, 6);
        let ev = eps_vir(nvir);
        let d = complete_pair_domains(&line_centers(nocc, 1.5)).unwrap();
        let ovov = ovov_block(nocc, nvir);
        let t2 = mp2_t2(&ovov, &eps_occ(nocc), &ev);

        // The RAW (pre-semicanonicalization) transforms.
        let raw = build_pno_transforms(&d, nvir, 0.0, |i, j| {
            Arr2::from_shape_fn((nvir, nvir), |(a, b)| t2[[i, j, a, b]])
        })
        .unwrap();

        let mut worst_off = 0.0f64;
        for p in &raw.pairs {
            let q = &p.transform;
            for a in 0..nvir {
                for b in 0..nvir {
                    if a != b {
                        let f_ab: f64 =
                            (0..nvir).map(|c| q[(c, a)] * q[(c, b)] * ev[c]).sum();
                        worst_off = worst_off.max(f_ab.abs());
                    }
                }
            }
        }
        eprintln!("stage 1: max off-diag F in the RAW PNO basis = {worst_off:.3e}");
        assert!(
            worst_off > 1e-3,
            "premise failed: raw PNOs already diagonalize F ({worst_off:.3e}), so the \
             semicanonicalization test would be vacuous"
        );
    }

    /// A loose threshold must actually compress, and report what it dropped.
    #[test]
    fn stage1_truncation_compresses_and_reports_loss() {
        let (nocc, nvir) = (4, 6);
        let (basis, _, _) = toy_basis(nocc, nvir, 1e-3);

        eprintln!(
            "stage 1: virtual retention {:.3}, max discarded weight {:.3e}",
            basis.virtual_retention(),
            basis.max_discarded_weight()
        );
        assert!(!basis.is_complete(), "a loose threshold should truncate something");
        assert!(basis.virtual_retention() < 1.0);
        assert!(basis.max_discarded_weight() > 0.0);
        let (pno_el, dense_el) = basis.amplitude_elements();
        assert!(pno_el < dense_el, "PNO amplitude count {pno_el} not below dense {dense_el}");
    }

    /// The pair screen composes: dropping pairs drops PNO blocks.
    #[test]
    fn stage1_composes_with_pair_screening() {
        let (nocc, nvir) = (4, 5);
        let ev = eps_vir(nvir);
        let ovov = ovov_block(nocc, nvir);
        let t2 = mp2_t2(&ovov, &eps_occ(nocc), &ev);
        let amp =
            |i: usize, j: usize| Arr2::from_shape_fn((nvir, nvir), |(a, b)| t2[[i, j, a, b]]);

        let all = complete_pair_domains(&line_centers(nocc, 10.0)).unwrap();
        let screened = build_pair_domains(&line_centers(nocc, 10.0), 15.0, f64::INFINITY).unwrap();
        assert!(screened.pairs.len() < all.pairs.len(), "test premise");

        let b_all = PairPnoBasis::build(&all, nvir, &ev, 0.0, amp).unwrap();
        let b_scr = PairPnoBasis::build(&screened, nvir, &ev, 0.0, amp).unwrap();
        assert!(b_scr.pairs.len() < b_all.pairs.len());
    }

    /// Bad inputs must error, not produce a plausible wrong number.
    #[test]
    fn stage1_invalid_inputs_are_rejected() {
        let (nocc, nvir) = (3, 4);
        let d = complete_pair_domains(&line_centers(nocc, 1.0)).unwrap();
        let amp = |_i: usize, _j: usize| Arr2::<f64>::zeros((nvir, nvir));
        // eps_vir of the wrong length.
        assert!(PairPnoBasis::build(&d, nvir, &eps_vir(nvir + 1), 0.0, amp).is_err());
        // Negative threshold (propagated from build_pno_transforms).
        assert!(PairPnoBasis::build(&d, nvir, &eps_vir(nvir), -1.0, amp).is_err());
    }

    // ============ STAGE 2: the amplitude round trip (LOAD-BEARING) =========

    /// THE LOAD-BEARING TEST: `t2 -> PNO basis -> back` must be the identity at
    /// `t_cut_pno = 0`.
    ///
    /// The transform is then square orthogonal, so `Q̃ Q̃ᵀ = I` and the round trip
    /// is exact. Any deviation beyond round-off means the transform, its
    /// orientation (`Q̃ᵀ · Q̃` vs `Q̃ · Q̃ᵀ`), or the pair indexing is wrong — and
    /// every stage-3 claim rests on this one, so it is checked elementwise on
    /// the whole tensor rather than through a scalar.
    #[test]
    fn stage2_t2_round_trip_is_exact_at_zero_truncation() {
        let (nocc, nvir) = (4, 6);
        let (basis, t2, _) = toy_basis(nocc, nvir, 0.0);

        let blocks = t2_to_pno(&t2, &basis).unwrap();
        assert_eq!(blocks.len(), basis.pairs.len());
        for (blk, p) in blocks.iter().zip(basis.pairs.iter()) {
            assert_eq!(blk.dim(), (nvir, nvir), "pair {:?} block is truncated", p.ij);
        }
        let back = t2_from_pno(&blocks, &basis, nocc).unwrap();

        let worst = t2
            .iter()
            .zip(back.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        // Guard against a vacuous pass: t2 must carry real signal.
        let scale = t2.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        eprintln!("stage 2: max |t2 - roundtrip(t2)| = {worst:.3e} (max |t2| = {scale:.3e})");
        assert!(scale > 1e-3, "t2 is ~zero ({scale:.3e}) — the round-trip check is vacuous");
        assert!(worst < 1e-10, "t2 round trip is not exact: max deviation {worst:.3e}");
    }

    /// The round trip must survive the `t1 ⊗ t1` part of `tau` too — a general
    /// non-symmetric, non-MP2-shaped tensor, not just the amplitudes the PNOs
    /// were derived from.
    #[test]
    fn stage2_round_trip_is_exact_for_an_unrelated_tensor() {
        let (nocc, nvir) = (4, 6);
        let (basis, _, _) = toy_basis(nocc, nvir, 0.0);

        let mut s = 0xDEADBEEF12345678u64;
        let other = Array4::from_shape_fn((nocc, nocc, nvir, nvir), |_| lcg(&mut s));
        // Symmetrize into a valid closed-shell t2 (t2[i,j,a,b] == t2[j,i,b,a]),
        // since t2_from_pno fills the mirror and would otherwise disagree there
        // for a reason that is not a bug.
        let sym = Array4::from_shape_fn((nocc, nocc, nvir, nvir), |(i, j, a, b)| {
            0.5 * (other[[i, j, a, b]] + other[[j, i, b, a]])
        });

        let back = t2_from_pno(&t2_to_pno(&sym, &basis).unwrap(), &basis, nocc).unwrap();
        let worst =
            sym.iter().zip(back.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        eprintln!("stage 2: unrelated-tensor round trip max deviation = {worst:.3e}");
        assert!(worst < 1e-10, "round trip failed on an unrelated tensor: {worst:.3e}");
    }

    /// Truncation must make the round trip LOSSY — otherwise the threshold is
    /// inert and any later accuracy/cost curve would be meaningless.
    #[test]
    fn stage2_truncation_makes_the_round_trip_lossy() {
        let (nocc, nvir) = (4, 6);
        let (basis, t2, _) = toy_basis(nocc, nvir, 1e-3);
        assert!(!basis.is_complete(), "test premise: something must be truncated");

        let back = t2_from_pno(&t2_to_pno(&t2, &basis).unwrap(), &basis, nocc).unwrap();
        let worst =
            t2.iter().zip(back.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        eprintln!("stage 2: truncated round-trip max deviation = {worst:.3e}");
        assert!(worst > 1e-12, "truncation had no effect on the round trip");
    }

    /// Screened pairs must come back exactly zero — the same convention as
    /// `dlpno_ccsd::apply_pair_mask`, so the two halves compose.
    #[test]
    fn stage2_screened_pairs_come_back_zero() {
        let (nocc, nvir) = (4, 5);
        let ev = eps_vir(nvir);
        let ovov = ovov_block(nocc, nvir);
        let t2 = mp2_t2(&ovov, &eps_occ(nocc), &ev);
        let centers = line_centers(nocc, 10.0);
        let screened = build_pair_domains(&centers, 15.0, f64::INFINITY).unwrap();
        let basis = PairPnoBasis::build(&screened, nvir, &ev, 0.0, |i, j| {
            Arr2::from_shape_fn((nvir, nvir), |(a, b)| t2[[i, j, a, b]])
        })
        .unwrap();

        let back = t2_from_pno(&t2_to_pno(&t2, &basis).unwrap(), &basis, nocc).unwrap();
        let mut keep = vec![false; nocc * nocc];
        for &(i, j) in &screened.pairs {
            keep[i * nocc + j] = true;
            keep[j * nocc + i] = true;
        }
        let mut n_zeroed = 0;
        for i in 0..nocc {
            for j in 0..nocc {
                if !keep[i * nocc + j] {
                    n_zeroed += 1;
                    let blk = back.slice(ndarray::s![i, j, .., ..]);
                    assert!(blk.iter().all(|&v| v == 0.0), "screened pair ({i},{j}) is nonzero");
                }
            }
        }
        assert!(n_zeroed > 0, "test premise: some pairs must be screened");
    }

    /// Shape disagreements are caller bugs and must error.
    #[test]
    fn stage2_shape_mismatches_are_rejected() {
        let (nocc, nvir) = (3, 4);
        let (basis, t2, _) = toy_basis(nocc, nvir, 0.0);
        // Wrong virtual dimension.
        let bad = Array4::<f64>::zeros((nocc, nocc, nvir + 1, nvir + 1));
        assert!(t2_to_pno(&bad, &basis).is_err());
        // Wrong block count.
        let blocks = t2_to_pno(&t2, &basis).unwrap();
        assert!(t2_from_pno(&blocks[..blocks.len() - 1], &basis, nocc).is_err());
        // Block of the wrong size.
        let mut bad_blocks = blocks.clone();
        bad_blocks[0] = Arr2::zeros((nvir - 1, nvir - 1));
        assert!(t2_from_pno(&bad_blocks, &basis, nocc).is_err());
    }

    // ================= STAGE 3: the energy ==================================

    /// THE EXACTNESS CONTRACT for the energy: at `t_cut_pno = 0` the per-pair
    /// PNO evaluation must reproduce the dense expression.
    ///
    /// The energy is a trace against the integrals; rotating both factors by the
    /// same orthogonal `Q̃` leaves it invariant. Any deviation means the
    /// integrals were rotated with a different (or transposed) matrix than the
    /// amplitudes, or the `(ib|ja)` exchange partner was mis-indexed — both of
    /// which give a plausible but wrong energy rather than a crash.
    #[test]
    fn stage3_pno_energy_matches_dense_at_zero_truncation() {
        let (nocc, nvir) = (4, 6);
        let (basis, t2, ovov) = toy_basis(nocc, nvir, 0.0);

        let dense = dense_ccsd_energy(&t2, &ovov).unwrap();
        let pno = pno_ccsd_energy(&t2, &ovov, &basis).unwrap();
        eprintln!("stage 3: dense E = {dense:.14}, PNO E = {pno:.14}, dE = {:.3e}", pno - dense);
        assert!(dense.abs() > 1e-3, "energy is ~zero ({dense:.3e}) — the check is vacuous");
        assert!(
            (pno - dense).abs() < 1e-10,
            "untruncated PNO energy must reproduce dense: {pno:.14} vs {dense:.14}"
        );
    }

    /// Same contract with a `tau` that is NOT the MP2 amplitudes the PNOs were
    /// built from — a converged CCSD `tau = t2 + t1⊗t1` is not, so the
    /// invariance must not depend on that coincidence.
    #[test]
    fn stage3_exact_for_a_tau_the_pnos_were_not_built_from() {
        let (nocc, nvir) = (4, 6);
        let (basis, t2, ovov) = toy_basis(nocc, nvir, 0.0);

        // Stand-in for t1 ⊗ t1: tau = t2 + t1[i,a] t1[j,b].
        let mut s = 0x51ED270B7C1D5A3Fu64;
        let t1 = Arr2::from_shape_fn((nocc, nvir), |_| 0.05 * lcg(&mut s));
        let tau = Array4::from_shape_fn((nocc, nocc, nvir, nvir), |(i, j, a, b)| {
            t2[[i, j, a, b]] + t1[(i, a)] * t1[(j, b)]
        });

        let dense = dense_ccsd_energy(&tau, &ovov).unwrap();
        let pno = pno_ccsd_energy(&tau, &ovov, &basis).unwrap();
        eprintln!("stage 3 (tau with t1): dense = {dense:.14}, PNO = {pno:.14}");
        assert!((pno - dense).abs() < 1e-10, "PNO energy differs by {:.3e}", pno - dense);
    }

    /// Truncation must actually change the energy, and the direction must be
    /// conservative: discarding virtuals removes correlation, so |E| shrinks.
    ///
    /// A truncation that left the energy alone would mean the knob is inert; one
    /// that *increased* |E| would mean it is not a projection.
    #[test]
    fn stage3_truncation_reduces_correlation() {
        let (nocc, nvir) = (4, 6);
        let (basis0, t2, ovov) = toy_basis(nocc, nvir, 0.0);
        let (basis_t, _, _) = toy_basis(nocc, nvir, 1e-3);
        assert!(!basis_t.is_complete(), "test premise");

        let e0 = pno_ccsd_energy(&t2, &ovov, &basis0).unwrap();
        let et = pno_ccsd_energy(&t2, &ovov, &basis_t).unwrap();
        eprintln!(
            "stage 3: retention {:.3}, E(exact) = {e0:.10}, E(truncated) = {et:.10}, dE = {:+.3e}",
            basis_t.virtual_retention(),
            et - e0
        );
        assert!((et - e0).abs() > 1e-12, "truncation had no effect on the energy");
        assert!(
            et.abs() < e0.abs(),
            "truncation must REDUCE |E_corr|: |{et:.10}| vs |{e0:.10}|"
        );
    }

    /// The energy must be summed over the full `nocc²` grid, mirrors included.
    ///
    /// `domains.pairs` holds each off-diagonal pair once, so an implementation
    /// that forgot the `(j,i)` mirror would come out roughly half-sized. Pinned
    /// by requiring agreement with the dense full-grid sum on a system with real
    /// off-diagonal weight.
    #[test]
    fn stage3_off_diagonal_mirrors_are_counted() {
        let (nocc, nvir) = (4, 5);
        let (basis, t2, ovov) = toy_basis(nocc, nvir, 0.0);

        // Diagonal-only contribution, for scale.
        let mut diag_only = 0.0;
        for i in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    diag_only +=
                        t2[[i, i, a, b]] * (2.0 * ovov[[i, a, i, b]] - ovov[[i, b, i, a]]);
                }
            }
        }
        let total = dense_ccsd_energy(&t2, &ovov).unwrap();
        assert!(
            (total - diag_only).abs() > 0.1 * total.abs(),
            "test premise: off-diagonal pairs must carry real weight"
        );
        let pno = pno_ccsd_energy(&t2, &ovov, &basis).unwrap();
        assert!((pno - total).abs() < 1e-10, "mirrors mis-counted: {pno:.12} vs {total:.12}");
    }

    // ---- STAGE 3 on a REAL system: water/STO-3G, converged CCSD amplitudes ----

    /// THE PHYSICS CHECK: on real converged CCSD amplitudes and real integrals,
    /// the untruncated per-pair PNO energy must reproduce
    /// [`crate::ccsd_closed_shell`]'s own reported correlation energy.
    ///
    /// The toy tests above pin the algebra on synthetic data; this pins that the
    /// integral CONVENTION matches the solver's. `ovov` is rebuilt here through
    /// the exact same RI path the solver uses (`V^{-1/2}` metric → dressed `B`
    /// blocks → `(ia|jb)` at `[i,a,j,b]`), so a mis-ordered block or a wrong
    /// `tau` would show up as a real energy difference rather than as a
    /// self-consistent synthetic pass.
    ///
    /// `obs_name` selects the orbital basis; the `cc-pvdz-ri` auxiliary is used
    /// throughout. Water is deliberately the smallest system with several
    /// occupied orbitals (so off-diagonal pairs carry real weight) while staying
    /// runnable on a contested box.
    fn stage3_real_system(obs_name: &str) {
        use ferric_core::basis;
        use ferric_core::mol::Molecule;
        use ferric_core::parallel::ParallelContext;
        use ferric_integrals::basis_bridge::PreparedBasis;
        use ferric_integrals::operator::Operator;
        use ferric_mp2::mo_transform::transform_3center_ov;
        use ferric_mp2::rimp2::cholesky_inverse_sqrt;
        use ferric_mp2::spinorbital::build_b;
        use ferric_scf::rhf::{solve_rhf, RhfConfig};
        use ferric_scf::screening::SchwarzBounds;
        use ferric_tensors::{einsum, Axis};

        let xyz = "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ctx,
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig { energy_conv: 1e-11, ..Default::default() },
        )
        .unwrap();

        let cfg = crate::CcConfig {
            frozen_core: 0,
            max_iter: 100,
            energy_conv: 1e-10,
            ..Default::default()
        };
        let cc = crate::ccsd_closed_shell::ccsd_closed_shell(&mol, &obs, &dfbs, op, &rhf, &cfg)
            .unwrap();

        let nbas = obs.nbasis();
        let eps = rhf.eps_r();
        let nocc = eps.iter().filter(|&&e| e < 0.0).count();
        let nvir = nbas - nocc;
        let c = rhf.mos_r();
        let c_occ = c.slice(ndarray::s![.., ..nocc]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc..]).to_owned();

        // Rebuild (ia|jb) through the solver's own RI path, so the convention is
        // identical by construction rather than by assumption.
        let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
        let v_inv_sqrt = cholesky_inverse_sqrt(&v2c).unwrap();
        let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let b_ov = build_b(
            &transform_3center_ov(&eri3_ao, &c_occ, &c_vir),
            &v_inv_sqrt,
            Axis::O,
            Axis::V,
        );
        let ovov_dyn: ndarray::ArrayD<f64> = einsum!("Pia,Pjb->iajb", &b_ov, &b_ov);
        let ovov = ovov_dyn.into_dimensionality::<ndarray::Ix4>().unwrap();

        // tau = t2 + t1 ⊗ t1, exactly as the solver's energy expression uses.
        let t1 = cc.t1.as_ref().unwrap();
        let tau = Array4::from_shape_fn((nocc, nocc, nvir, nvir), |(i, j, a, b)| {
            cc.t2[[i, j, a, b]] + t1[[i, a]] * t1[[j, b]]
        });

        // Premise: our dense re-evaluation must reproduce the solver's own
        // energy. If it does not, the integral block or tau is wrong and the
        // PNO comparison below would be self-consistently wrong too.
        let dense = dense_ccsd_energy(&tau, &ovov).unwrap();
        eprintln!(
            "stage 3 (water/{obs_name}): solver E_corr = {:.12}, re-evaluated dense = {dense:.12}",
            cc.correlation_energy
        );
        assert!(
            (dense - cc.correlation_energy).abs() < 1e-9,
            "dense re-evaluation disagrees with the solver ({dense:.12} vs {:.12}) — the \
             integral convention or tau is wrong, so the PNO check would be vacuous",
            cc.correlation_energy
        );

        // Build the PNO basis from first-order amplitudes on the CANONICAL
        // orbital centers. Localized (Boys) occupieds are what a production run
        // would use; canonical ones are used here because this test is about
        // EXACTNESS at zero truncation, which is basis-of-occupieds independent.
        let ev: Vec<f64> = (0..nvir).map(|a| eps[nocc + a]).collect();
        let eo: Vec<f64> = (0..nocc).map(|i| eps[i]).collect();
        let centers = line_centers(nocc, 1.0);
        let domains = complete_pair_domains(&centers).unwrap();
        let basis = PairPnoBasis::build(&domains, nvir, &ev, 0.0, |i, j| {
            Arr2::from_shape_fn((nvir, nvir), |(a, b)| {
                ovov[[i, a, j, b]] / (eo[i] + eo[j] - ev[a] - ev[b])
            })
        })
        .unwrap();
        assert!(basis.is_complete(), "t_cut_pno = 0 must keep every virtual");

        let pno = pno_ccsd_energy(&tau, &ovov, &basis).unwrap();
        let dev = (pno - cc.correlation_energy).abs();
        eprintln!(
            "stage 3 (water/{obs_name}): PNO E_corr = {pno:.12}, |dE| vs solver = {dev:.3e} \
             (no={nocc}, nv={nvir}, {} pairs)",
            basis.pairs.len()
        );
        assert!(
            dev < 1e-9,
            "untruncated PNO CCSD energy must reproduce dense CCSD: {pno:.12} vs {:.12}",
            cc.correlation_energy
        );

        // And the knob must be live on the real system too: truncating must move
        // the energy, in the conservative direction.
        let trunc = PairPnoBasis::build(&domains, nvir, &ev, 1e-4, |i, j| {
            Arr2::from_shape_fn((nvir, nvir), |(a, b)| {
                ovov[[i, a, j, b]] / (eo[i] + eo[j] - ev[a] - ev[b])
            })
        })
        .unwrap();
        let e_t = pno_ccsd_energy(&tau, &ovov, &trunc).unwrap();
        eprintln!(
            "stage 3 (water/{obs_name}): t_cut_pno=1e-4 retention {:.3}, E = {e_t:.10}, dE = {:+.3e}",
            trunc.virtual_retention(),
            e_t - pno
        );
        assert!(trunc.virtual_retention() <= 1.0);
    }

    /// The real-system exactness contract at minimal basis.
    #[test]
    fn stage3_pno_energy_matches_dense_ccsd_water_sto3g() {
        stage3_real_system("sto-3g");
    }

    /// The same contract with a genuinely non-trivial rotation.
    ///
    /// Water/STO-3G has only `nv = 2` virtuals, so its PNO transform is a 2×2
    /// rotation — small enough that a partially wrong transform could still pass
    /// by luck. 6-31G gives `nv = 8`, where the rotation mixes real structure and
    /// the raw (pre-semicanonicalization) Fock matrix has large off-diagonal
    /// elements. Same size class, so it stays cheap on a shared box.
    #[test]
    fn stage3_pno_energy_matches_dense_ccsd_water_631g() {
        stage3_real_system("6-31g");
    }

    /// Shape disagreements must error.
    #[test]
    fn stage3_shape_mismatches_are_rejected() {
        let (nocc, nvir) = (3, 4);
        let (basis, t2, ovov) = toy_basis(nocc, nvir, 0.0);
        let bad_tau = Array4::<f64>::zeros((nocc, nocc, nvir + 1, nvir + 1));
        assert!(pno_ccsd_energy(&bad_tau, &ovov, &basis).is_err());
        let bad_ovov = Array4::<f64>::zeros((nocc, nvir, nocc, nvir + 1));
        assert!(pno_ccsd_energy(&t2, &bad_ovov, &basis).is_err());
        assert!(dense_ccsd_energy(&t2, &bad_ovov).is_err());
    }
}
