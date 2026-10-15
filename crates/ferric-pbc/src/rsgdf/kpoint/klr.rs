//! Range-separated exact exchange on the k-point RS-GDF J/K ([`crate::rsh`]).
//!
//! # What the attenuated exchange needs
//!
//! The range-separated hybrid exchange kernel is
//! `v(r) = c_sr erfc(ωr)/r + c_lr erf(ωr)/r = c_sr/r + (c_lr − c_sr) erf(ωr)/r`
//! (`crate::rsh` module doc). The exchange matrix is linear in the kernel, so
//!
//! ```text
//! K = c_sr K[1/r] + (c_lr − c_sr) K[erf(ωr)/r] + v_M^{mix} S D S
//! ```
//!
//! * `K[1/r]` is the existing full-Coulomb k-point RS-GDF exchange
//!   ([`KRsGdf`], whose Ewald split parameter `ω_0` is unrelated to the
//!   attenuation `ω`), reused unchanged with the Madelung term left out.
//! * `K[erf(ωr)/r]` is built here by [`KRsGdf::build_attenuated`]. The erf
//!   kernel has NO short-range part: in reciprocal space it is
//!   `v_ω(K) = 4π/K² e^{−K²/4ω²}`, a Gaussian-damped function that vanishes
//!   past `K ≈ 2ω√ln(1/ε)`, so the whole fit lives in G-space. The aux fit is
//!   the existing one with the SR real-space sums absent and the split
//!   parameter replaced by the attenuation `ω`: `J2(q)`, `J3(k,k')` are
//!   [`lr_accumulate_q`] on `K = G + q` (the `K = 0` term dropped, exactly the
//!   dense oracle's kernel, so there is no `c0` subtraction to make), the
//!   metric is solved with the same per-q lindep cut, and the same
//!   `B(k,k')` layout feeds the same exchange contraction. The fit is exact
//!   when the aux functions' Fourier vectors span the kept K set (then
//!   `J3 J2^{−1} J3^H` reproduces `Σ_K v a a^H` identically), so for small
//!   `ω` (few K vectors) the attenuated K carries no fit error. When the K
//!   set is larger than the aux rank it is the usual Coulomb-metric fit
//!   error, second order in the fit residual.
//! * The long-range (`erf`) part is the one that diverges at `K = 0`; the
//!   `erfc` part does not. The mixed exchange-divergence constant
//!   `v_M^{mix} = c_sr v_M^{full} + (c_lr − c_sr) v_M^{LR}(ω)` of the
//!   `diag(N)` supercell ([`RshParams::madelung`]) multiplies `S D S`; with
//!   `K = 0` dropped from every fitted piece this is the complete `G = 0`
//!   treatment (HSE06, `c_lr = 0`: `c_sr [v_M^{full} − v_M^{LR}]`, the
//!   finite short-range constant).
//!
//! The erf piece is skipped (zero to the build precision) when no q class has
//! a K vector inside the Gaussian cut, e.g. `ω → 0`.
//!
//! Derivatives (forces, stress) of the attenuated exchange are NOT
//! implemented; the force and stress entry points refuse a [`KRsGdf`] that
//! carries it ([`KRsGdf::rsh`]).

use super::{
    hermitize_pairs, lr_accumulate_q, lr_kvectors, sat_prod, KRsGdf, KRsGdfConfig, KRsGdfMutation,
    KRsGdfQStats, KRsGdfStats, QBlock,
};
use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::hcore::{gvector_list_bytes, G_CHUNK_BYTES};
use crate::kpts::KPointMesh;
use crate::kscf::eigh_herm;
use crate::lattice::Cell;
use crate::pair_ft::residues::residue_coords;
use crate::rsgdf::{check_obs_on_cell, dot3, gshells, LatticeWalker, Stage};
use crate::rsh::RshParams;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;
use num_complex::Complex64 as C64;

/// TEST-ONLY defects of the range-separated exchange on RS-GDF
/// ([`KRsGdfMutation::Rsh`]); each must break an anchor of
/// `tests/pbc_krsh_rsgdf.rs`.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KRshMutant {
    /// The erf piece with no Gaussian damping (a truncated full Coulomb).
    NoAttenuation,
    /// `ω` taken in Å⁻¹ where Bohr⁻¹ is meant (×1.8897 in the attenuated fit).
    OmegaUnits,
    /// The erf and erfc roles swapped: `c_lr K[1/r] + (c_sr − c_lr) K_erf`.
    SwappedRoles,
    /// The mixed Madelung constant dropped.
    NoMadelung,
    /// The mixed Madelung constant of the PRIMITIVE cell, not the supercell.
    PrimitiveMadelung,
}

/// The attenuated-exchange companion a [`KRsGdf`] carries.
#[derive(Debug, Clone)]
pub(super) struct RshAttached {
    pub(super) params: RshParams,
    /// `None`: the erf piece is zero to the build precision.
    pub(super) lr: Option<KRsGdf>,
    pub(super) swapped: bool,
}

/// Budget the attenuated fit may use: what is left of the resolved budget
/// after the full-Coulomb fit's resident B and `S(k)` (floored at 1 byte; 0
/// would mean "unset" to the ledger).
pub(super) fn attenuated_budget(total: usize, resident: usize) -> usize {
    total.saturating_sub(resident).max(1)
}

impl RshAttached {
    /// `(coefficient of K[1/r], coefficient of K[erf(ωr)/r])`.
    pub(super) fn coefficients(&self) -> (f64, f64) {
        let p = &self.params;
        if self.swapped {
            (p.c_lr, p.c_sr - p.c_lr)
        } else {
            (p.c_sr, p.c_lr - p.c_sr)
        }
    }
}

impl KRsGdf {
    /// RS-GDF k-point tensors for a range-separated hybrid: the full-Coulomb
    /// fit of [`KRsGdf::build`] plus the attenuated fit
    /// (`KRsGdf::build_attenuated`), with the mixed Madelung constant. The
    /// J/K builders then return
    /// `K = c_sr K[1/r] + (c_lr − c_sr) K[erf] + v_M S D S` (module doc).
    /// `cfg.gdf.exxdiv` picks `v_M^{mix}` or 0.
    pub fn build_rsh(
        cell: &Cell,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
        mesh: &KPointMesh,
        s_k: &[Array2<C64>],
        cfg: &KRsGdfConfig,
        rsh: RshParams,
    ) -> Result<Self, FerricError> {
        rsh.validate()?;
        let mutant = match cfg.mutation {
            Some(KRsGdfMutation::Rsh(m)) => Some(m),
            _ => None,
        };
        let mut full = Self::build(cell, obs, aux, mesh, s_k, cfg)?;
        // The full-Coulomb B stays resident while the attenuated fit is built.
        let budget = attenuated_budget(
            crate::budget::resolve(cfg.gdf.budget_bytes),
            full.resident_bytes(),
        );
        let needs_lr = rsh.c_lr != rsh.c_sr || mutant == Some(KRshMutant::SwappedRoles);
        let lr = if needs_lr {
            let w = match mutant {
                Some(KRshMutant::OmegaUnits) => rsh.omega * ferric_core::units::ANGSTROM_TO_BOHR,
                _ => rsh.omega,
            };
            Self::build_attenuated(cell, obs, aux, mesh, s_k, cfg, (w, budget), mutant)?
        } else {
            None
        };
        let lat = match mutant {
            Some(KRshMutant::PrimitiveMadelung) => *cell.lattice(),
            _ => mesh.supercell_lattice(),
        };
        let mixed = match mutant {
            Some(KRshMutant::NoMadelung) => 0.0,
            _ => rsh.madelung(&Cell::new(cell.mol().clone(), lat)?)?,
        };
        full.madelung_ewald = mixed;
        full.madelung = match cfg.gdf.exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => mixed,
        };
        full.rsh = Some(Box::new(RshAttached {
            params: rsh,
            lr,
            swapped: mutant == Some(KRshMutant::SwappedRoles),
        }));
        Ok(full)
    }

    /// The exchange kernel `Σ_a B_a ⊗ conj B_a` of `(k, k')` with the
    /// range-separation mix applied (the bare kernel without a companion).
    pub(super) fn mixed_k_kernel(&self, k: usize, kp: usize) -> Array2<C64> {
        let mut kern = self.plain_k_kernel(k, kp);
        if let Some(rsh) = &self.rsh {
            let (c_full, c_lr) = rsh.coefficients();
            kern.mapv_inplace(|z| z * c_full);
            if let Some(lr) = &rsh.lr {
                kern.scaled_add(C64::new(c_lr, 0.0), &lr.plain_k_kernel(k, kp));
            }
        }
        kern
    }

    /// Turn the accumulated bare full-Coulomb exchange `k` into the
    /// range-separated one: `K = c_full K[1/r] + c_lr K[erf(ωr)/r]` (module
    /// doc); a no-op without a companion.
    pub(super) fn mix_rsh_exchange(
        &self,
        dm: &[Array2<C64>],
        k: &mut [Array2<C64>],
    ) -> Result<(), FerricError> {
        let Some(rsh) = &self.rsh else {
            return Ok(());
        };
        let (c_full, c_lr) = rsh.coefficients();
        for kx in k.iter_mut() {
            kx.mapv_inplace(|z| z * c_full);
        }
        if let Some(lr) = rsh.lr.as_ref().filter(|_| c_lr != 0.0) {
            let mut kl = vec![Array2::<C64>::zeros((self.nao, self.nao)); self.nk];
            lr.check_exchange_scratch()?;
            lr.add_k_blocks(dm, &mut kl)?;
            for (kx, kb) in k.iter_mut().zip(&kl) {
                kx.scaled_add(C64::new(c_lr, 0.0), kb);
            }
        }
        Ok(())
    }

    /// The range-separation parameters of an attached attenuated exchange.
    pub fn rsh(&self) -> Option<RshParams> {
        self.rsh.as_ref().map(|r| r.params)
    }

    /// Fit of the erf-attenuated Coulomb kernel `erf(ωr)/r` with the `K = 0`
    /// term dropped (module doc), in the same `B(k,k')` layout as
    /// [`KRsGdf::build`]. `None` when no q class has a K vector inside the
    /// Gaussian cut `|K| <= 2ω√ln(1/ε)`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_attenuated(
        cell: &Cell,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
        mesh: &KPointMesh,
        s_k: &[Array2<C64>],
        cfg: &KRsGdfConfig,
        (omega, budget): (f64, usize),
        mutant: Option<KRshMutant>,
    ) -> Result<Option<Self>, FerricError> {
        let g = &cfg.gdf;
        g.validate()?;
        if !(omega > 0.0 && omega.is_finite()) {
            return Err(FerricError::General(format!(
                "KRsGdf attenuated exchange: omega must be finite and > 0, got {omega}"
            )));
        }
        crate::rsgdf::require_pure_aux(aux, "KRsGdf attenuated exchange")?;
        let n = obs.nbasis();
        let n2 = n * n;
        let naux = aux.nbasis();
        let nk = mesh.nk();
        if naux == 0 || n == 0 || s_k.len() != nk || s_k.iter().any(|s| s.dim() != (n, n)) {
            return Err(FerricError::General(format!(
                "KRsGdf attenuated exchange: bad shapes (nao = {n}, naux = {naux}, {} overlaps, \
                 N_k = {nk})",
                s_k.len()
            )));
        }
        check_obs_on_cell(cell, obs)?;
        // The cut is set by the TRUE omega; the NoAttenuation mutant removes the
        // Gaussian damping by evaluating the weights at a huge omega.
        let gcut = 2.0 * omega * (1.0 / g.precision).ln().sqrt();
        let w_eval = if mutant == Some(KRshMutant::NoAttenuation) {
            1e6
        } else {
            omega
        };
        let st = Stage {
            cell,
            obs,
            aux,
            obs_sh: gshells(obs, "KRsGdf attenuated orbital basis")?,
            aux_sh: gshells(aux, "KRsGdf attenuated aux basis")?,
            omega: w_eval,
            thresh: g.precision,
            sr_screen: g.sr_screen,
            walker: LatticeWalker::new(cell),
        };
        let mod_l = mesh.residue_moduli();
        let rl = mod_l[0] * mod_l[1] * mod_l[2];
        let reps: Vec<usize> = (0..nk).filter(|&iq| mesh.q_minus(iq) >= iq).collect();
        let kvs: Vec<(Vec<[f64; 3]>, bool)> = reps
            .iter()
            .map(|&iq| lr_kvectors(cell, mesh, iq, mesh.q_minus(iq) == iq, gcut))
            .collect::<Result<_, _>>()?;
        if kvs.iter().all(|(kv, _)| kv.is_empty()) {
            return Ok(None);
        }

        let mut ledger = Ledger::new(budget);
        ledger.reserve(
            &format!("KRsGdf attenuated S(k) copies (nao = {n}, N_k = {nk})"),
            bytes_of(sat_prod(&[nk + 2, n2]), 16),
        )?;
        ledger.reserve(
            &format!(
                "KRsGdf attenuated resident B ({} q classes x N_k = {nk} x naux = {naux} x \
                 nao² = {n2})",
                reps.len()
            ),
            bytes_of(sat_prod(&[reps.len(), nk, naux, n2]), 16),
        )?;
        ledger.reserve(
            &format!("KRsGdf attenuated per-q working set (R_L = {rl} accumulators)"),
            bytes_of(
                sat_prod(&[4, naux, naux])
                    .saturating_add(sat_prod(&[rl, n2, naux]))
                    .saturating_add(sat_prod(&[2, n2, naux])),
                16,
            ),
        )?;
        let qmax = (0..nk)
            .map(|iq| {
                let q = mesh.q_class(iq).0;
                dot3(&q, &q).sqrt()
            })
            .fold(0.0_f64, f64::max);
        ledger.reserve(
            &format!("KRsGdf attenuated K list (|K| <= {gcut:.3}, |q| <= {qmax:.3})"),
            gvector_list_bytes(cell, gcut + qmax)?.saturating_mul(2),
        )?;
        let resident_bytes = ledger.resident();
        let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);

        let phk: Vec<Vec<C64>> = (0..nk)
            .map(|k| {
                (0..rl)
                    .map(|r| mesh.phase(k, residue_coords(r, mod_l)))
                    .collect()
            })
            .collect();
        let mut blocks: Vec<QBlock> = Vec::with_capacity(reps.len());
        let mut per_q: Vec<Option<KRsGdfQStats>> = vec![None; nk];
        let (mut n_lr_chunks, mut b_elements, mut q0) = (0usize, 0usize, usize::MAX);
        for (&iq, (kv, half)) in reps.iter().zip(&kvs) {
            let iqm = mesh.q_minus(iq);
            let trim = iqm == iq;
            let (q, mq) = mesh.q_class(iq);
            let mut j2 = Array2::<C64>::zeros((naux, naux));
            let mut acc: Vec<Array2<C64>> =
                (0..rl).map(|_| Array2::<C64>::zeros((n2, naux))).collect();
            if !kv.is_empty() {
                n_lr_chunks +=
                    lr_accumulate_q(&st, kv, *half, mod_l, &mut j2, &mut acc, chunk_budget)?;
            }
            // Σ_K weight: bounds every element of the erf exchange this class
            // carries (|a_ml(K)| <= ~1 for normalised orbitals), see `fit_block`.
            let wfac = if *half { 2.0 } else { 1.0 } / cell.volume() * 4.0 * std::f64::consts::PI;
            let wsum: f64 = kv
                .iter()
                .map(|k| {
                    let k2 = dot3(k, k);
                    wfac / k2 * (-k2 / (4.0 * w_eval * w_eval)).exp()
                })
                .sum();
            let (bq, kept, evals) = fit_block(
                (&j2, &acc, &phk),
                (g.lindep, wsum),
                (mq == [0, 0, 0], n, iq),
            )?;
            drop(acc);
            b_elements += nk * kept * n2;
            let kof: Vec<usize> = (0..nk).map(|j| mesh.k_minus_q(j, mq)).collect();
            let mirror = iqm != iq;
            let qs = KRsGdfQStats {
                iq,
                q,
                trim,
                mirrored_from: None,
                naux,
                naux_kept: kept,
                n_dropped: naux - kept,
                metric_eig_min: evals.first().copied().unwrap_or(0.0),
                metric_eig_max: evals.last().copied().unwrap_or(0.0),
                n_k_vectors: kv.len(),
                asym_j2: 0.0,
                asym_j3: 0.0,
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
                "KRsGdf attenuated exchange: the q = 0 class was not built (internal error)".into(),
            ));
        }
        let minus: Vec<usize> = (0..nk).map(|k| mesh.minus(k)).collect();
        let mut pair_loc = vec![(usize::MAX, 0, false); nk * nk];
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
                "KRsGdf attenuated exchange: some (k, k') pair has no q block (internal error)"
                    .into(),
            ));
        }
        let per_q: Vec<KRsGdfQStats> = per_q
            .into_iter()
            .map(|s| s.ok_or_else(|| FerricError::General("attenuated q record missing".into())))
            .collect::<Result<_, _>>()?;
        let stats = KRsGdfStats {
            nk,
            naux,
            per_q,
            n_q_built: reps.len(),
            n_pair_images: 0,
            n_sr3_triplets: 0,
            n_sr3_triplets_ordered: 0,
            n_sr2_pairs: 0,
            residues_l: rl,
            residues_t: 1,
            n_lr_chunks,
            b_elements,
            budget_bytes: ledger.budget(),
            resident_bytes,
            split_counters: Vec::new(),
            sr_rotated_columns: 0,
        };
        Ok(Some(Self {
            nao: n,
            nk,
            blocks,
            q0,
            pair_loc,
            minus,
            s: s_k.to_vec(),
            madelung: 0.0,
            madelung_ewald: 0.0,
            rsh: None,
            stats,
        }))
    }
}

/// Metric solve and `B(k,k')` of one q class from the LR accumulators:
/// returns `(B per k', kept count, metric eigenvalues)`. An empty K set gives
/// zero-row `B` blocks.
fn fit_block(
    (j2, acc, phk): (&Array2<C64>, &[Array2<C64>], &[Vec<C64>]),
    (lindep, wsum): (f64, f64),
    (q_is_zero, n, iq): (bool, usize, usize),
) -> Result<(Vec<Array2<C64>>, usize, Vec<f64>), FerricError> {
    let (n2, naux) = (n * n, j2.nrows());
    if j2.iter().all(|z| z.norm() == 0.0) {
        let zero = Array2::<C64>::zeros((0, n2));
        return Ok((vec![zero; phk.len()], 0, Vec::new()));
    }
    let j2h = {
        let h = j2.t().mapv(|z| z.conj());
        (j2 + &h).mapv(|z| z * 0.5)
    };
    let (evals, evecs) = eigh_herm(&j2h).map_err(|e| {
        FerricError::Lapack(format!(
            "KRsGdf attenuated metric eigh at q class {iq}: {e}"
        ))
    })?;
    let keep: Vec<usize> = (0..naux).filter(|&k| evals[k] > lindep).collect();
    if keep.is_empty() {
        // Every eigenvalue is under the absolute cut. If the kernel weights
        // themselves are negligible (`wsum < lindep`; the K vectors sit in the
        // far Gaussian tail) the class contributes below the build precision:
        // every element of `K_erf` is bounded by `wsum·max|a|²` with
        // `|a_ml(K)| <~ 1`, the same size as the directions the cut drops
        // elsewhere. Zero-row B, as for an empty K set. With non-negligible
        // weights the smallness is the aux basis's (it does not span the K
        // set): a genuinely singular metric, which stays a typed error.
        if wsum < lindep {
            let zero = Array2::<C64>::zeros((0, n2));
            return Ok((vec![zero; phk.len()], 0, evals.to_vec()));
        }
        return Err(FerricError::General(format!(
            "KRsGdf attenuated exchange: every metric eigenvalue at q class {iq} is <= lindep \
             {lindep:e} (max {:e}) although the kernel weights are not negligible (sum {wsum:e})",
            evals[naux - 1]
        )));
    }
    let mut w = Array2::<C64>::zeros((naux, keep.len()));
    for (c, &k) in keep.iter().enumerate() {
        let f = 1.0 / evals[k].sqrt();
        for r in 0..naux {
            w[(r, c)] = evecs[(r, k)].conj() * f;
        }
    }
    let mut bq = Vec::with_capacity(phk.len());
    for ph_j in phk {
        let mut j3 = Array2::<C64>::zeros((n2, naux));
        for (a, ph) in acc.iter().zip(ph_j) {
            j3.zip_mut_with(a, |z, &x| *z += *ph * x);
        }
        if q_is_zero {
            hermitize_pairs(&mut j3, n);
        }
        bq.push(j3.dot(&w).t().as_standard_layout().into_owned());
    }
    Ok((bq, keep.len(), evals.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::attenuated_budget;

    #[test]
    fn attenuated_budget_is_the_remainder_floored_at_one() {
        assert_eq!(attenuated_budget(1000, 400), 600);
        assert_eq!(attenuated_budget(1000, 1000), 1);
        assert_eq!(attenuated_budget(1000, 5000), 1);
        assert_eq!(attenuated_budget(0, 0), 1);
        // The resident bytes are charged: a mutant ignoring them returns the total.
        assert_ne!(attenuated_budget(1000, 400), 1000);
    }
}
