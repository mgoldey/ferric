//! True pair-density PNOs for PDEP-RPA — and the measurement of whether they beat
//! the diagonal-only OSVs already in [`crate::pno`].
//!
//! # What is different from `pno.rs`, precisely
//!
//! [`crate::pno`] builds **OSVs**: it diagonalizes the *diagonal* pair density
//! `D^ii = 2 T^ii (T^ii)ᵀ`, one virtual set per occupied orbital, then concatenates
//! and QRs them into a single shared reduced basis. Only `i = j` amplitude blocks
//! ever enter.
//!
//! This module builds **true PNOs** via [`ferric_mp2::local_pno::build_pno_transforms`]:
//! for every occupied pair `(i, j)` it diagonalizes the *off-diagonal* pair density
//!
//! ```text
//! D^ij = T^ij (T^ij)ᵀ + (T^ij)ᵀ T^ij
//! ```
//!
//! which is strictly more information than the OSV construction sees. That is the
//! whole reason PNO thresholds compress harder than OSV thresholds at equal error in
//! the DLPNO literature, and it is the hypothesis this module exists to test on
//! ferric.
//!
//! # The structural obstacle, stated honestly
//!
//! RPA's independent-particle response is a sum over **single** excitations:
//!
//! ```text
//! ε̃(iω) = I + Σ_{ia} B_{ia} s²_{ia}(ω) B_{ia}ᵀ
//! ```
//!
//! There is exactly one occupied index per term. A PNO basis, by construction, is
//! indexed by a *pair* `(i, j)`. So unlike MP2 — where `dlpno_mp2.rs` can rotate each
//! pair's `(ia|jb)` block into that pair's own PNOs because the energy expression
//! carries a pair index — **RPA has no pair index to hang a per-pair virtual basis
//! on.** This is not an implementation gap; it is what the RPA energy functional is.
//!
//! The construction used here is the natural resolution, and is what makes the
//! comparison to OSVs apples-to-apples:
//!
//! 1. Build true per-pair PNOs `P^{ij}` from the off-diagonal pair densities.
//! 2. For each occupied `i`, form the **union** of the PNO subspaces of every retained
//!    pair that `i` participates in — that is the virtual space orbital `i` actually
//!    needs in order to be described by all of its pairs. Concretely: eigendecompose
//!    `M^i = Σ_{j : (i,j) retained} P^{ij} (P^{ij})ᵀ`, whose range is exactly that union.
//! 3. Concatenate the per-orbital union bases and orthogonalize once, giving a single
//!    shared reduced virtual basis, then semicanonicalize it.
//!
//! Step 3 mirrors `pno.rs` deliberately: the two paths differ **only** in where the
//! per-orbital subspace came from (off-diagonal pair densities vs the diagonal one),
//! so any difference in the measured retention/error curve is attributable to true
//! PNOs and nothing else.
//!
//! # Semicanonicalization is not optional
//!
//! The reduced virtuals are NOT eigenvectors of the virtual Fock matrix, and RPA's
//! `s_{ia}(ω)` denominators are only valid in a Fock-diagonal basis. So the virtual
//! Fock matrix is built in the reduced basis and **re-diagonalized**. Taking only its
//! diagonal is wrong and fails silently — in the MP2 sibling that mistake broke the
//! exactness contract by 0.117 Ha before it was caught (see
//! [`ferric_mp2::dlpno_mp2`]). [`dlpno_rpa_zero_threshold_matches_canonical_h2o`]
//! pins the fix here.
//!
//! # Exactness contract
//!
//! At `t_cut_pno = 0` every `P^{ij}` is a square orthogonal rotation, so each union
//! subspace is the full virtual space, the shared basis spans all `nvir` virtuals, and
//! after semicanonicalization the composite transform lands back on the canonical
//! virtuals up to ordering and sign. The RPA energy must then reproduce untruncated
//! PDEP-RPA. Two tests pin this.
//!
//! # What this is honestly worth — read before quoting a number
//!
//! ferric has a MEASURED negative result for the OSV path: 100% virtual retention at
//! accurate thresholds (zero compression), and 48–76 mHa error at `t_osv = 1e-3`. A
//! cliff, not a tradeoff. This module does not assume PNOs fix that. It measures it,
//! side by side on the same system, via
//! [`compare_osv_vs_pno`] and the `osv_vs_pno_retention_and_error_sweep` test.
//!
//! **No wall-clock quantity is measured or reported anywhere in this module**, by
//! design: retention counts and energies are reproducible, timings on a contested box
//! are not.

use crate::PdepRpaConfig;
use ferric_core::linalg::{eigh_dc, Uplo};
use ferric_core::FerricError;
use ferric_mp2::local_pno::{build_pno_transforms, PnoTransforms};
use ferric_mp2::oo_rimp2::compute_t2_and_integrals;
use ferric_mp2::pair_domains::{complete_pair_domains, PairDomains};
use ferric_mp2::rimp2::RpaIntermediates;
use ndarray::{s, Array2};

/// Tolerance on the union-projector eigenvalues that decides which directions are in
/// the span of a per-orbital union of PNO subspaces.
///
/// `M^i = Σ_j P^{ij} (P^{ij})ᵀ` is a sum of orthogonal projectors, so every nonzero
/// eigenvalue is ≥ 1 for a direction lying in at least one `P^{ij}` and exactly 0 for a
/// direction orthogonal to all of them. The gap is therefore O(1), not O(ε) — this
/// threshold sits deep inside it and is not a tunable knob.
const UNION_RANK_TOL: f64 = 1e-8;

/// Shared reduced virtual basis built from TRUE per-pair PNOs.
///
/// Structurally parallel to [`crate::pno::DnvTransform`] so the two can be compared
/// field by field; the difference is entirely in how the per-orbital subspaces were
/// obtained.
#[derive(Debug, Clone)]
pub struct DlpnoRpaTransform {
    /// Dimension of the per-occupied-orbital union subspace, before the shared
    /// orthogonalization. Index `i` is the active occupied orbital.
    pub n_pno_per_orbital: Vec<usize>,
    /// Final shared reduced virtual count.
    pub n_vir_reduced: usize,
    /// Full (untruncated) virtual count, for retention ratios.
    pub nvir: usize,
    /// Transformed RI tensor `B^P_{i, ã}`, shape `(naux, nocc · n_vir_reduced)`.
    pub b_ov_pno: Array2<f64>,
    /// Semicanonical (Fock-diagonal) reduced virtual energies, length `n_vir_reduced`.
    pub eps_vir_reduced: Vec<f64>,
    /// Per-pair retention diagnostics straight from the PNO construction, before the
    /// union/orthogonalization steps. This is the quantity DLPNO papers quote.
    pub pair_retention: f64,
    /// Mean per-pair virtual retention from [`PnoTransforms::virtual_retention`].
    pub pno_virtual_retention: f64,
    /// Largest per-pair discarded occupation weight.
    pub max_discarded_weight: f64,
    /// Number of occupied pairs PNOs were built for.
    pub n_pairs: usize,
}

impl DlpnoRpaTransform {
    /// Fraction of the canonical virtual space surviving into the shared reduced
    /// basis. `1.0` means no compression at all — the OSV path's measured outcome at
    /// accurate thresholds, and the number to check before believing any PNO win.
    pub fn shared_retention(&self) -> f64 {
        if self.nvir == 0 {
            return 1.0;
        }
        self.n_vir_reduced as f64 / self.nvir as f64
    }
}

/// Build true per-pair PNO transforms for an RPA reference.
///
/// `domains` selects which occupied pairs get PNOs; pass
/// [`ferric_mp2::pair_domains::complete_pair_domains`] to disable occupied-side
/// screening and isolate the virtual-side question, which is what the comparison
/// against OSVs requires.
///
/// The amplitudes are the semicanonical first-order (MP2) ones built from the same
/// `B^P_{ia}` tensor the RPA dielectric uses, via
/// [`ferric_mp2::oo_rimp2::compute_t2_and_integrals`] — the identical source
/// [`crate::pno::build_dnv_transform`] draws on, so the OSV/PNO comparison differs
/// only in which pair densities are diagonalized.
///
/// # Errors
///
/// Propagates PNO eigensolver failures and errors when `domains` was built for a
/// different `nocc` than `inter` carries.
pub fn build_pair_pnos(
    inter: &RpaIntermediates,
    eps: &[f64],
    domains: &PairDomains,
    t_cut_pno: f64,
) -> Result<PnoTransforms, FerricError> {
    let (nocc, nvir) = (inter.nocc, inter.nvir);
    if domains.nocc != nocc {
        return Err(FerricError::General(format!(
            "build_pair_pnos: domains were built for nocc={}, but the RPA \
             intermediates carry nocc={nocc}",
            domains.nocc
        )));
    }

    let (t2, _eri) = compute_t2_and_integrals(
        &inter.b_ov,
        eps,
        nocc,
        nvir,
        inter.nocc_total,
        inter.first_occ,
        inter.naux,
    );
    let nov = nocc * nvir;

    // T^ij_ab = t2[(i·nvir + a)·nov + (j·nvir + b)]. For i != j this block is NOT
    // symmetric, which is exactly why the off-diagonal pair density needs both
    // T Tᵀ and Tᵀ T terms (local_pno handles that).
    let amp = |i: usize, j: usize| -> Array2<f64> {
        Array2::from_shape_fn((nvir, nvir), |(a, b)| {
            t2[(i * nvir + a) * nov + (j * nvir + b)]
        })
    };

    build_pno_transforms(domains, nvir, t_cut_pno, amp)
}

/// Per-occupied-orbital union of the PNO subspaces of every pair containing `i`.
///
/// Returns an orthonormal `(nvir × k_i)` basis whose range is
/// `span{ range(P^{ij}) : (i,j) or (j,i) retained }`. Obtained by eigendecomposing
/// `M^i = Σ_j P^{ij} (P^{ij})ᵀ`, a sum of orthogonal projectors whose nonzero
/// eigenvalues are ≥ 1 — so the rank test against [`UNION_RANK_TOL`] sits in an O(1)
/// gap rather than resolving a fuzzy singular-value tail.
///
/// An occupied orbital appearing in no retained pair cannot happen through
/// [`build_pair_domains`][ferric_mp2::pair_domains::build_pair_domains] (diagonal
/// pairs are never screened), but is handled rather than silently producing an empty
/// basis: it falls back to the full virtual space.
fn union_subspace_for_orbital(
    pnos: &PnoTransforms,
    i: usize,
    nvir: usize,
) -> Result<Array2<f64>, FerricError> {
    let mut m = Array2::<f64>::zeros((nvir, nvir));
    let mut n_contributing = 0usize;
    for pair in &pnos.pairs {
        if pair.ij.0 != i && pair.ij.1 != i {
            continue;
        }
        n_contributing += 1;
        let p = &pair.transform;
        // M += P Pᵀ  (the orthogonal projector onto this pair's PNO space).
        m = m + p.dot(&p.t());
    }
    if n_contributing == 0 {
        // No pair touches this orbital: keep everything rather than zero its response.
        return Ok(Array2::<f64>::eye(nvir));
    }

    let (eigs, vecs) = eigh_dc(&m, Uplo::Upper).map_err(|e| {
        FerricError::General(format!("PNO union eigh failed for occupied orbital {i}: {e}"))
    })?;
    let keep: Vec<usize> = (0..nvir).filter(|&k| eigs[k] > UNION_RANK_TOL).collect();
    if keep.is_empty() {
        return Err(FerricError::General(format!(
            "PNO union subspace for occupied orbital {i} came out empty (largest \
             projector eigenvalue {:.3e}); this should be impossible for a sum of \
             orthogonal projectors over {n_contributing} pairs",
            eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        )));
    }
    let mut basis = Array2::<f64>::zeros((nvir, keep.len()));
    for (slot, &k) in keep.iter().enumerate() {
        for a in 0..nvir {
            basis[(a, slot)] = vecs[(a, k)];
        }
    }
    Ok(basis)
}

/// Build a shared, semicanonical reduced virtual basis from TRUE per-pair PNOs.
///
/// Pipeline: per-pair PNOs → per-orbital union subspaces → concatenate →
/// orthogonalize → semicanonicalize → transform `B`.
///
/// The last two steps are the load-bearing ones. Orthogonalization uses the same
/// projector-eigendecomposition trick as the union step (`Σ_i Q^i (Q^i)ᵀ`), which is
/// numerically better behaved than QR-ing a wide, highly redundant concatenation and
/// guessing its rank from `R`'s diagonal. Semicanonicalization re-diagonalizes the
/// virtual Fock matrix in the shared basis; skipping it silently breaks exactness (see
/// the module docs).
///
/// # Errors
///
/// Propagates PNO construction and eigensolver failures, and errors if the reduced
/// basis comes out empty.
pub fn build_dlpno_rpa_transform(
    inter: &RpaIntermediates,
    eps: &[f64],
    domains: &PairDomains,
    t_cut_pno: f64,
) -> Result<DlpnoRpaTransform, FerricError> {
    let (nocc, nvir, naux) = (inter.nocc, inter.nvir, inter.naux);
    let nocc_total = inter.nocc_total;

    let pnos = build_pair_pnos(inter, eps, domains, t_cut_pno)?;

    // --- Per-orbital union subspaces from the OFF-DIAGONAL pair densities. ---
    let mut n_pno_per_orbital = Vec::with_capacity(nocc);
    let mut per_orbital: Vec<Array2<f64>> = Vec::with_capacity(nocc);
    for i in 0..nocc {
        let basis = union_subspace_for_orbital(&pnos, i, nvir)?;
        n_pno_per_orbital.push(basis.ncols());
        per_orbital.push(basis);
    }

    // --- Shared basis: range of Σ_i Q^i (Q^i)ᵀ, again a sum of projectors. ---
    let mut acc = Array2::<f64>::zeros((nvir, nvir));
    for q_i in &per_orbital {
        acc = acc + q_i.dot(&q_i.t());
    }
    let (acc_eigs, acc_vecs) = eigh_dc(&acc, Uplo::Upper).map_err(|e| {
        FerricError::General(format!("DLPNO-RPA shared-basis eigh failed: {e}"))
    })?;
    let keep: Vec<usize> = (0..nvir).filter(|&k| acc_eigs[k] > UNION_RANK_TOL).collect();
    if keep.is_empty() {
        return Err(FerricError::General(
            "DLPNO-RPA: shared reduced virtual basis is empty".into(),
        ));
    }
    let n_vir_reduced = keep.len();
    let mut q = Array2::<f64>::zeros((nvir, n_vir_reduced));
    for (slot, &k) in keep.iter().enumerate() {
        for a in 0..nvir {
            q[(a, slot)] = acc_vecs[(a, k)];
        }
    }

    // --- Semicanonicalize: RPA denominators are only valid in a Fock-diagonal
    // basis, and PNOs are not Fock eigenvectors. Build F_vir in the reduced basis
    // and RE-DIAGONALIZE. Taking the diagonal alone is silently wrong. ---
    let mut f_red = Array2::<f64>::zeros((n_vir_reduced, n_vir_reduced));
    for k in 0..n_vir_reduced {
        for l in 0..n_vir_reduced {
            f_red[(k, l)] =
                (0..nvir).map(|a| q[(a, k)] * eps[nocc_total + a] * q[(a, l)]).sum();
        }
    }
    let (eps_red, u_red) = eigh_dc(&f_red, Uplo::Upper).map_err(|e| {
        FerricError::General(format!("DLPNO-RPA semicanonicalization eigh failed: {e}"))
    })?;
    let q_canonical = q.dot(&u_red);

    // --- Transform B^P_{ia} into the semicanonical reduced basis. ---
    let mut b_ov_pno = Array2::<f64>::zeros((naux, nocc * n_vir_reduced));
    for i in 0..nocc {
        let b_i = inter.b_ov.slice(s![.., i * nvir..(i + 1) * nvir]);
        let b_red = b_i.dot(&q_canonical); // (naux, n_vir_reduced)
        b_ov_pno
            .slice_mut(s![.., i * n_vir_reduced..(i + 1) * n_vir_reduced])
            .assign(&b_red);
    }

    Ok(DlpnoRpaTransform {
        n_pno_per_orbital,
        n_vir_reduced,
        nvir,
        b_ov_pno,
        eps_vir_reduced: eps_red,
        pair_retention: domains.pair_retention(),
        pno_virtual_retention: pnos.virtual_retention(),
        max_discarded_weight: pnos.max_discarded_weight(),
        n_pairs: pnos.pairs.len(),
    })
}

/// RPA correlation energy on a reduced virtual space.
///
/// Shared by the PNO path and (through [`compare_osv_vs_pno`]) the OSV path so the two
/// are integrated by *identical* code — the comparison would be worthless if the two
/// energies came from different quadrature or eigensolve settings.
///
/// Mirrors [`crate::pno::run_pdep_rpa_osv`]'s integration stage: full-rank identity
/// seed → Lanczos → log-det trace over the imaginary-frequency quadrature.
fn rpa_energy_in_reduced_basis(
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    naux: usize,
    config: &PdepRpaConfig,
) -> Result<f64, FerricError> {
    let seed = Array2::<f64>::eye(naux);
    let max_iter =
        if config.eigensolver_max_vecs == 0 { 3 * naux } else { config.eigensolver_max_vecs };

    let matvec = |v: &Array2<f64>| -> Array2<f64> {
        crate::sternheimer::dielectric_apply(v, b_ov, eps_occ, eps_vir, 0.0)
    };
    let lz = crate::lanczos::run_lanczos_seeded(
        seed,
        matvec,
        naux,
        max_iter,
        config.eigensolver_conv_thresh,
        config.verbose,
    )?;
    if !lz.converged {
        eprintln!(
            "warning: DLPNO-RPA Lanczos eigensolve did NOT converge (max Ritz residual \
             {:.3e} > {:.3e}); the reported energy is best-effort",
            lz.max_resid, config.eigensolver_conv_thresh
        );
    }

    let n_keep = lz
        .eigenvalues
        .iter()
        .filter(|&&lam| (lam - 1.0).abs() > config.trunc_thresh)
        .count()
        .max(1);
    let v_kept = lz.eigenvectors.slice(s![.., ..n_keep]).to_owned();

    let (quad_freqs, quad_weights) = crate::quadrature::build_quadrature(&config.quadrature);
    let summands = crate::energy::eval_trace_log_summands_budgeted(
        &v_kept,
        b_ov,
        eps_occ,
        eps_vir,
        &quad_freqs,
        config.memory_budget_bytes,
    )?;
    Ok(crate::energy::rpa_correlation_energy_from_summands(&quad_weights, &summands))
}

/// PDEP-RPA correlation energy in a TRUE per-pair-PNO-derived reduced virtual space.
///
/// Returns `(E_c, n_vir_reduced, transform)`. `t_cut_pno = 0` must reproduce
/// untruncated PDEP-RPA — that is the exactness contract, pinned by
/// [`dlpno_rpa_zero_threshold_matches_canonical_h2o`].
///
/// Occupied-side pair screening is deliberately disabled here (complete domains): this
/// entry point exists to answer the *virtual-side* question, and mixing in a second
/// approximation would make the OSV comparison uninterpretable. Callers wanting both
/// screens can use [`build_dlpno_rpa_transform`] with their own `PairDomains`.
///
/// No timing is measured or returned. Retention counts and energies only.
pub fn run_pdep_rpa_pno(
    mol: &ferric_core::mol::Molecule,
    obs: &ferric_integrals::basis_bridge::PreparedBasis,
    dfbs: &ferric_integrals::basis_bridge::PreparedBasis,
    op: ferric_integrals::operator::Operator,
    rhf: &ferric_scf::ScfResult,
    config: &PdepRpaConfig,
    t_cut_pno: f64,
) -> Result<(f64, usize, DlpnoRpaTransform), FerricError> {
    use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config};

    let mp2_cfg = RiMp2Config {
        frozen_core: config.frozen_core,
        memory_budget_bytes: config.memory_budget_bytes,
        ..Default::default()
    };
    let inter = compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;
    let eps = rhf.eps_r();

    // Boys centers are only needed as diagnostics when no occupied screening is
    // applied, so a zero-filled placeholder is honest here — `complete_pair_domains`
    // applies no distance test whatsoever.
    let centers = Array2::<f64>::zeros((inter.nocc, 3));
    let domains = complete_pair_domains(&centers)?;

    let tr = build_dlpno_rpa_transform(&inter, eps, &domains, t_cut_pno)?;

    let eps_occ: Vec<f64> =
        eps[inter.first_occ..inter.first_occ + inter.nocc].to_vec();
    let e_c = rpa_energy_in_reduced_basis(
        &tr.b_ov_pno,
        &eps_occ,
        &tr.eps_vir_reduced,
        inter.naux,
        config,
    )?;
    let n = tr.n_vir_reduced;
    Ok((e_c, n, tr))
}

/// One row of the OSV-vs-PNO comparison: both truncations at the same threshold, on
/// the same reference, integrated by the same code.
#[derive(Debug, Clone)]
pub struct TruncationComparison {
    /// The threshold applied to both schemes (`t_osv` for OSV, `t_cut_pno` for PNO).
    pub threshold: f64,
    /// Full canonical virtual count.
    pub nvir: usize,
    /// Shared reduced virtual count from the OSV (diagonal-density) path.
    pub n_vir_osv: usize,
    /// Shared reduced virtual count from the true-PNO (off-diagonal density) path.
    pub n_vir_pno: usize,
    /// Mean PER-PAIR virtual retention before the union/shared step — the quantity
    /// DLPNO papers quote. Diverges from `n_vir_pno / nvir` exactly to the extent that
    /// different pairs select different virtuals.
    pub pno_pair_retention: f64,
    /// OSV correlation energy error vs the untruncated result (Hartree, signed).
    pub err_osv: f64,
    /// True-PNO correlation energy error vs the untruncated result (Hartree, signed).
    pub err_pno: f64,
}

/// Sweep thresholds and measure OSV retention/error against true-PNO retention/error
/// on one system.
///
/// This is the module's reason to exist: ferric's OSV path has a measured cliff (100%
/// retention at accurate thresholds, tens of mHa error at loose ones), and the honest
/// question is whether true pair-density PNOs change that verdict. Both schemes are
/// run against the same reference, the same untruncated baseline, and the same
/// integration code, so the only difference is which pair densities were diagonalized.
///
/// Returns `(e_canonical, rows)`. Reports energies and retention counts only — no
/// wall-clock quantity is measured.
pub fn compare_osv_vs_pno(
    mol: &ferric_core::mol::Molecule,
    obs: &ferric_integrals::basis_bridge::PreparedBasis,
    dfbs: &ferric_integrals::basis_bridge::PreparedBasis,
    op: ferric_integrals::operator::Operator,
    rhf: &ferric_scf::ScfResult,
    config: &PdepRpaConfig,
    thresholds: &[f64],
) -> Result<(f64, Vec<TruncationComparison>), FerricError> {
    let e_canonical = crate::run_pdep_rpa(mol, obs, dfbs, op, rhf, config)?.e_rpa;

    let mut rows = Vec::with_capacity(thresholds.len());
    for &t in thresholds {
        let (e_osv, n_osv, _naux) =
            crate::pno::run_pdep_rpa_osv(mol, obs, dfbs, op, rhf, config, t)?;
        let (e_pno, n_pno, tr) =
            run_pdep_rpa_pno(mol, obs, dfbs, op, rhf, config, t)?;
        rows.push(TruncationComparison {
            threshold: t,
            nvir: tr.nvir,
            n_vir_osv: n_osv,
            n_vir_pno: n_pno,
            pno_pair_retention: tr.pno_virtual_retention,
            err_osv: e_osv - e_canonical,
            err_pno: e_pno - e_canonical,
        });
    }
    Ok((e_canonical, rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{QuadratureConfig, QuadratureScheme};
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config};
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use ferric_scf::ScfResult;

    const H2O: &str =
        "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";

    /// H2O/STO-3G has nvir = 2. That is enough to pin exactness and error handling
    /// cheaply, but FAR too small to say anything about compression — with two
    /// virtuals there is nothing to discard and every scheme trivially "retains 100%".
    /// Anything claiming to measure retention must use [`benzene`] instead.
    fn water() -> Ref {
        reference_from_xyz(H2O)
    }

    /// Benzene/STO-3G: 36 basis functions, 21 occupied, **nvir = 15**, 231 occupied
    /// pairs. The smallest system in the tree with a virtual space big enough for a
    /// retention question to be meaningful, and still tiny in RAM.
    fn benzene() -> Ref {
        let xyz = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/molecules/benzene.xyz"
        ))
        .expect("testdata/molecules/benzene.xyz");
        reference_from_xyz(&xyz)
    }

    /// STO-3G / cc-pvdz-ri throughout. The box this runs on is contested; a bigger
    /// basis would buy nothing the comparison needs.
    struct Ref {
        mol: Molecule,
        obs: PreparedBasis,
        dfbs: PreparedBasis,
        op: Operator,
        rhf: ScfResult,
    }

    fn reference_from_xyz(xyz: &str) -> Ref {
        let ctx = ParallelContext::default();
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("sto-3g").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        Ref { mol, obs, dfbs, op, rhf }
    }

    fn cfg() -> PdepRpaConfig {
        PdepRpaConfig {
            quadrature: QuadratureConfig {
                scheme: QuadratureScheme::GaussLegendre,
                n_points: 20,
                u0: 0.5,
            },
            frozen_core: 0,
            trunc_thresh: 0.0,
            eigensolver_conv_thresh: 1e-9,
            ..Default::default()
        }
    }

    fn intermediates(r: &Ref) -> RpaIntermediates {
        compute_rpa_intermediates(
            &r.mol,
            &r.obs,
            &r.dfbs,
            r.op,
            &r.rhf,
            &RiMp2Config { frozen_core: 0, memory_budget_bytes: None, ..Default::default() },
        )
        .unwrap()
    }

    /// The PNOs must actually come from OFF-DIAGONAL pair densities, i.e. the pair
    /// list must contain `i != j` pairs and their densities must differ from the
    /// diagonal ones. Without this the module would be an OSV implementation wearing
    /// a PNO name, and every comparison below would be vacuous.
    #[test]
    fn pnos_are_built_from_off_diagonal_pairs() {
        let r = benzene();
        let inter = intermediates(&r);
        let centers = Array2::<f64>::zeros((inter.nocc, 3));
        let domains = complete_pair_domains(&centers).unwrap();
        let pnos = build_pair_pnos(&inter, r.rhf.eps_r(), &domains, 0.0).unwrap();

        let n_offdiag = pnos.pairs.iter().filter(|p| p.ij.0 != p.ij.1).count();
        assert!(
            n_offdiag > 0,
            "premise: a true-PNO build must include off-diagonal pairs (nocc={})",
            inter.nocc
        );
        assert_eq!(
            pnos.pairs.len(),
            inter.nocc * (inter.nocc + 1) / 2,
            "complete domains must give every i<=j pair"
        );

        // An off-diagonal pair's occupations must differ from the diagonal pair's —
        // otherwise D^ij carries no information beyond D^ii.
        let diag = pnos.pairs.iter().find(|p| p.ij == (0, 0)).unwrap();
        let offd = pnos.pairs.iter().find(|p| p.ij.0 != p.ij.1).unwrap();
        let differ = diag
            .occupations
            .iter()
            .zip(offd.occupations.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(
            differ,
            "off-diagonal pair {:?} has the same occupations as the diagonal pair; \
             the off-diagonal density is not being used",
            offd.ij
        );
    }

    /// THE EXACTNESS CONTRACT, part 1: at `t_cut_pno = 0` every transform is a square
    /// orthogonal rotation, so the shared reduced basis must span the whole virtual
    /// space.
    #[test]
    fn zero_threshold_spans_the_full_virtual_space() {
        let r = water();
        let inter = intermediates(&r);
        let centers = Array2::<f64>::zeros((inter.nocc, 3));
        let domains = complete_pair_domains(&centers).unwrap();
        let tr = build_dlpno_rpa_transform(&inter, r.rhf.eps_r(), &domains, 0.0).unwrap();

        eprintln!(
            "t_cut_pno=0: n_pno_per_orbital={:?}, n_vir_reduced={} (nvir={})",
            tr.n_pno_per_orbital, tr.n_vir_reduced, inter.nvir
        );
        assert_eq!(
            tr.n_vir_reduced, inter.nvir,
            "a lossless PNO transform must span the full virtual space"
        );
        assert_eq!(tr.pno_virtual_retention, 1.0);
        assert_eq!(tr.max_discarded_weight, 0.0);
    }

    /// THE EXACTNESS CONTRACT, part 2 — the one that catches a missing
    /// semicanonicalization.
    ///
    /// Spanning the full space is necessary but NOT sufficient: the reduced basis is
    /// an arbitrary orthogonal rotation of the canonical virtuals, and RPA's
    /// `s_{ia}(ω)` denominators are only valid where the virtual Fock matrix is
    /// diagonal. Re-diagonalizing it restores that. Skipping the re-diagonalization
    /// leaves this test failing by far more than its tolerance while the span test
    /// above still passes — which is exactly how the MP2 sibling's 0.117 Ha bug hid.
    #[test]
    fn dlpno_rpa_zero_threshold_matches_canonical_h2o() {
        let r = water();
        let c = cfg();

        let e_canonical =
            crate::run_pdep_rpa(&r.mol, &r.obs, &r.dfbs, r.op, &r.rhf, &c).unwrap().e_rpa;
        let (e_pno, n_vir, tr) =
            run_pdep_rpa_pno(&r.mol, &r.obs, &r.dfbs, r.op, &r.rhf, &c, 0.0).unwrap();

        eprintln!(
            "H2O/STO-3G  canonical={e_canonical:.12}  pno(t=0)={e_pno:.12}  \
             n_vir_reduced={n_vir} (nvir={})",
            tr.nvir
        );
        let dev = (e_pno - e_canonical).abs();
        assert!(
            dev < 1e-8,
            "true-PNO RPA at t_cut_pno=0 must reproduce canonical PDEP-RPA; \
             deviation {dev:.3e} Ha (a missing semicanonicalization looks exactly \
             like this)"
        );
    }

    /// Semicanonicalization is load-bearing, demonstrated directly rather than
    /// asserted in prose: the virtual Fock matrix in the raw (pre-semicanonical)
    /// reduced basis has substantial off-diagonal elements, so using its diagonal as
    /// orbital energies — the tempting shortcut — would be wrong.
    #[test]
    fn reduced_basis_fock_is_not_diagonal_before_semicanonicalization() {
        let r = benzene();
        let inter = intermediates(&r);
        let eps = r.rhf.eps_r();
        let nvir = inter.nvir;
        let centers = Array2::<f64>::zeros((inter.nocc, 3));
        let domains = complete_pair_domains(&centers).unwrap();
        let pnos = build_pair_pnos(&inter, eps, &domains, 0.0).unwrap();

        // Take a single pair's raw PNO transform and build F_vir in it.
        let p = &pnos.pairs.iter().find(|p| p.ij.0 != p.ij.1).unwrap().transform;
        let n = p.ncols();
        let mut f = Array2::<f64>::zeros((n, n));
        for a in 0..n {
            for b in 0..n {
                f[(a, b)] =
                    (0..nvir).map(|c| p[(c, a)] * eps[inter.nocc_total + c] * p[(c, b)]).sum();
            }
        }
        let max_offdiag = (0..n)
            .flat_map(|a| (0..n).map(move |b| (a, b)))
            .filter(|(a, b)| a != b)
            .map(|(a, b)| f[(a, b)].abs())
            .fold(0.0, f64::max);
        eprintln!("max |F_offdiag| in the raw PNO basis = {max_offdiag:.6}");
        assert!(
            max_offdiag > 1e-3,
            "premise: the raw PNO basis should NOT be Fock-diagonal (max offdiag \
             {max_offdiag:.3e}); if it were, semicanonicalization would be a no-op \
             and this module's central caveat would be vacuous"
        );
    }

    /// A large threshold must actually compress and must report the loss, otherwise
    /// the knob is inert and the sweep below would be measuring nothing.
    #[test]
    fn large_threshold_compresses_and_reports_loss() {
        let r = benzene();
        let inter = intermediates(&r);
        let centers = Array2::<f64>::zeros((inter.nocc, 3));
        let domains = complete_pair_domains(&centers).unwrap();
        let tr = build_dlpno_rpa_transform(&inter, r.rhf.eps_r(), &domains, 1e-2).unwrap();

        eprintln!(
            "t_cut_pno=1e-2: per-pair retention={:.3}, shared retention={:.3} \
             ({}/{}), max discarded weight={:.3e}",
            tr.pno_virtual_retention,
            tr.shared_retention(),
            tr.n_vir_reduced,
            tr.nvir,
            tr.max_discarded_weight
        );
        assert!(
            tr.pno_virtual_retention < 1.0,
            "a 1e-2 occupation threshold should truncate SOME pair's virtuals"
        );
        assert!(tr.max_discarded_weight > 0.0, "truncation must report what it discarded");
        assert!(tr.n_vir_reduced <= tr.nvir);
    }

    /// Bad inputs error rather than producing a plausible wrong number.
    #[test]
    fn mismatched_domains_are_rejected() {
        let r = water();
        let inter = intermediates(&r);
        let centers = Array2::<f64>::zeros((inter.nocc + 1, 3));
        let domains = complete_pair_domains(&centers).unwrap();
        assert!(build_pair_pnos(&inter, r.rhf.eps_r(), &domains, 0.0).is_err());
        assert!(build_dlpno_rpa_transform(&inter, r.rhf.eps_r(), &domains, 0.0).is_err());
    }

    /// A negative threshold is a caller bug and must error, not silently keep
    /// everything.
    #[test]
    fn negative_threshold_is_rejected() {
        let r = water();
        let inter = intermediates(&r);
        let centers = Array2::<f64>::zeros((inter.nocc, 3));
        let domains = complete_pair_domains(&centers).unwrap();
        assert!(build_pair_pnos(&inter, r.rhf.eps_r(), &domains, -1e-6).is_err());
    }

    /// THE MEASUREMENT this module exists for: OSV retention/error vs true-PNO
    /// retention/error, same system, same baseline, same integration code.
    ///
    /// ferric's prior OSV result was a cliff — 100% retention at accurate thresholds,
    /// tens of mHa error at loose ones. This test reports whether true pair-density
    /// PNOs change that. It asserts only the exactness end of the sweep and the
    /// direction of the retention/error tradeoff; the verdict itself is the printed
    /// table, because forcing a threshold assertion would be pre-judging the answer.
    #[test]
    fn osv_vs_pno_retention_and_error_sweep() {
        let r = benzene();
        let c = cfg();
        let thresholds = [1e-8_f64, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2];

        let (e_canonical, rows) =
            compare_osv_vs_pno(&r.mol, &r.obs, &r.dfbs, r.op, &r.rhf, &c, &thresholds).unwrap();

        eprintln!("\n=== OSV vs TRUE PNO — H2O/STO-3G + cc-pvdz-ri ===");
        eprintln!("canonical PDEP-RPA E_c = {e_canonical:.10} Ha, nvir = {}", rows[0].nvir);
        eprintln!(
            "{:>9}  {:>11}  {:>11}  {:>13}  {:>12}  {:>12}",
            "thresh", "n_vir(OSV)", "n_vir(PNO)", "PNO pair-ret", "dE OSV(mHa)", "dE PNO(mHa)"
        );
        for row in &rows {
            eprintln!(
                "{:>9.0e}  {:>11}  {:>11}  {:>13.3}  {:>12.4}  {:>12.4}",
                row.threshold,
                row.n_vir_osv,
                row.n_vir_pno,
                row.pno_pair_retention,
                row.err_osv * 1e3,
                row.err_pno * 1e3
            );
        }

        // Exactness end of the sweep: at 1e-8 both schemes must be essentially lossless.
        let tight = &rows[0];
        assert!(
            tight.err_pno.abs() < 1e-6,
            "true PNO at threshold 1e-8 should be sub-uHa, got {:.3e} Ha",
            tight.err_pno
        );
        assert!(
            tight.err_osv.abs() < 1e-6,
            "OSV at threshold 1e-8 should be sub-uHa, got {:.3e} Ha",
            tight.err_osv
        );

        // Retention must be monotone-ish in the threshold and bounded — a scheme whose
        // reduced space GREW with a looser threshold would be a bug, not a tradeoff.
        for row in &rows {
            assert!(row.n_vir_pno <= row.nvir, "PNO cannot exceed nvir");
            assert!(row.n_vir_osv <= row.nvir, "OSV cannot exceed nvir");
            assert!(row.n_vir_pno >= 1 && row.n_vir_osv >= 1);
        }
        assert!(
            rows.last().unwrap().n_vir_pno <= rows[0].n_vir_pno,
            "loosening the PNO threshold must not increase the retained virtual count"
        );
    }
}
