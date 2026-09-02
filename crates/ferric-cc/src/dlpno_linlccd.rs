//! DLPNO-LinLCCD(hh) — **a closed amplitude iteration in per-pair PNO bases.**
//!
//! This is the thing [`crate::dlpno_ccsd_kernel`] explicitly did not do. That
//! module's "Stage D" — a residual and an iteration that never leave the
//! per-pair virtual spaces — was left undone for two named reasons: CCSD's
//! singles channel has no pair-shaped home, and its `ovvv`/`vvvv` blocks are the
//! engineering bulk of a production DLPNO code.
//!
//! **Neither reason applies to LinLCCD(hh)**, and that is the whole argument for
//! this module. Read [`crate::linlccd`]'s residual: for
//! [`LadderVariant::Hh`](crate::linlccd::LadderVariant::Hh) it is three lines.
//!
//! ```text
//!   R = <ij||ab>  +  ½ Σ_kl <kl||ij> t[k,l,a,b]        (the hh ladder)
//!   t_new = R / D                                      (Jacobi)
//! ```
//!
//! No `t1`. No `tau`. No `ovvv`. No `vvvv` — the block is never even built. The
//! equation is **linear in T2**, and the only virtual-index structure it has is
//! `[a,b]` carried untouched from `t` to `R`. So every quantity in the iteration
//! is *pair-shaped*, which is exactly the shape
//! [`crate::dlpno_ccsd_kernel`] established stays compressed end-to-end.
//!
//! # The one contraction, and its S-insertion
//!
//! The hh ladder couples occupied pair `(k,l)` to occupied pair `(i,j)` through
//! `<kl||ij>`, summing nothing over virtuals — the virtual labels `a,b` are
//! spectators. In a per-pair basis those spectators live in *different* spaces on
//! the two sides, so the amplitude must be converted from `(k,l)`'s PNOs into
//! `(i,j)`'s by the pair-pair overlap `S^{ij,kl} = Q_ijᵀ Q_kl`:
//!
//! ```text
//!   R̃^{ij}[ã,b̃] = ṽ^{ij}[ã,b̃] + ½ Σ_kl <kl||ij> · ( S^{ij,kl} t̃^{kl} (S^{ij,kl})ᵀ )[ã,b̃]
//! ```
//!
//! This is the same conversion [`crate::dlpno_ccsd_kernel::pno_loo_t2_term`]
//! performs — a `t2`-shaped output driven by an occupied-space matrix — and it is
//! structurally identical to the `pno_woooo`-style pattern that module solved. The
//! derivation is the resolution of identity between two pairs' virtual spaces:
//! writing `t[k,l,a,b] = Σ_ãb̃ Q_kl[a,ã] t̃^{kl}[ã,b̃] Q_kl[b,b̃]` and projecting the
//! result onto `Q_ij`, the `a` sum gives `Σ_a Q_ij[a,ã] Q_kl[a,c̃] = S^{ij,kl}[ã,c̃]`.
//!
//! At `t_cut_pno = 0` every `Q` is square orthogonal, so every `S` is orthogonal
//! and the insertion collapses to the dense contraction. That is the exactness
//! contract, and it is pinned at every stage below.
//!
//! # Spin orbitals
//!
//! [`crate::linlccd`] is a **spin-orbital** method: `no2 = 2·no`, `nv2 = 2·nv`,
//! even index = α, odd = β. The PNO machinery in
//! [`crate::dlpno_ccsd_virtual`] is written against a generic
//! `(nocc, nvir, eps_vir)` and is agnostic to what those indices *mean*, so it is
//! used here over the **spin-orbital** indices directly: `nvir = nv2`, `eps_vir`
//! the spin-orbital virtual energies, and [`PairDomains`] built over `no2`
//! occupied spin orbitals whose centers are the spatial centers duplicated.
//!
//! That choice is deliberate and is what makes the amplitude blocks map 1:1 onto
//! `t[i,j,:,:]` with no re-indexing. It also means the antisymmetry
//! `t[i,j,a,b] = −t[i,j,b,a] = −t[j,i,a,b]` is the structural identity in play —
//! **not** the closed-shell `t[i,j,a,b] = t[j,i,b,a]` that
//! [`crate::dlpno_ccsd_virtual::t2_from_pno`] assumes. So this module does its own
//! per-pair transforms rather than reusing `t2_to_pno`/`t2_from_pno`, whose mirror
//! convention is wrong for a spin-orbital tensor. Reusing them would have been a
//! plausible-but-wrong energy; see [`so_t2_to_pno`].
//!
//! # Staging
//!
//! | Stage | What | Exactness test |
//! |-------|------|----------------|
//! | 1 | [`pno_hh_ladder`] — the hh ladder with `S` inserted | [`tests::stage1_hh_ladder_is_exact_at_zero_truncation`] |
//! | 2 | [`dlpno_linlccd_hh`] — a **CLOSED iteration** | [`tests::stage2_closed_iteration_reproduces_dense_water_sto3g`] |
//! | 3 | [`HhFlopCount`] — structural cost as a function of `npno` | [`tests::stage3_cost_strictly_decreases_with_truncation`] |
//!
//! # SEMICANONICALIZATION
//!
//! Jacobi denominators use [`PairPno::eps`] — the eigenvalues of `Q ᵀ diag(ε_v) Q`
//! — via [`crate::dlpno_ccsd_kernel::pno_denominators`]. The diagonal-only
//! shortcut `f_aa = Σ_c Q_ca² ε_c` is a MEASURED 0.117 Ha error in the DLPNO-MP2
//! sibling and is not available here.
//!
//! # No wall clock
//!
//! Cost is claimed **structurally**: [`HhFlopCount`] is a pure function of the
//! per-pair `npno`, performs no arithmetic, and its dense column uses the same
//! pair list so the ratio isolates *virtual* truncation. It also reports the
//! **transform overhead** separately, because for the DLPNO-(T) sibling the
//! transform cost more than the kernel it saved.

use crate::dlpno_ccsd_kernel::{pno_denominators, PairIndex, PairOverlaps};
use crate::dlpno_ccsd_virtual::{PairPno, PairPnoBasis};
use crate::linlccd::LadderVariant;
use crate::{CcConfig, CcResult};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::mo_transform::{transform_3center_oo, transform_3center_ov};
use ferric_mp2::pair_domains::{complete_pair_domains, PairDomains};
use ferric_mp2::rimp2::cholesky_inverse_sqrt;
use ferric_mp2::spinorbital::{asym_oovv, asym_same, build_b};
use ferric_scf::ScfResult;
use ferric_tensors::{einsum, Axis};
use ndarray::{Array2, Array4, ArrayD};

// =====================================================================
// Spin-orbital amplitude <-> PNO transforms
// =====================================================================
//
// These are the spin-orbital analogues of `dlpno_ccsd_virtual::t2_to_pno` /
// `t2_from_pno`.
//
// MEASURED, on a valid antisymmetric spin-orbital tensor: the DOUBLE mirror
// `t[j,i,b,a]` is the same in both conventions, because spin-orbital `t` is
// antisymmetric SEPARATELY in the occupied and the virtual pair:
//
//   t[j,i,a,b] = -t[i,j,a,b]      (occupied swap)
//   t[i,j,b,a] = -t[i,j,a,b]      (virtual swap)
//   t[j,i,b,a] = +t[i,j,a,b]      (BOTH — the two minus signs cancel)
//
// So `dlpno_ccsd_virtual::t2_from_pno`'s `t2[j,i,b,a] = +t2[i,j,a,b]` fill is
// numerically CORRECT for a spin-orbital tensor too. An earlier version of this
// module wrote a minus there on the reasoning that "spin orbitals are
// antisymmetric", and every mirror block came back at exactly 2x its own
// magnitude with the wrong sign (MEASURED: dev 5.2e-1 against scale 2.6e-1 on a
// no2=4/nv2=4 probe, i.e. -x vs +x). The blocks stored as (i,j) with i<=j were
// simultaneously exact to 3.3e-16, which is precisely the failure signature that
// would have produced a plausible-but-wrong energy had the residual not been
// compared elementwise.
//
// These helpers therefore exist NOT because the sign differs but because the
// SHAPE contract does: `t2_from_pno` is documented against closed-shell spatial
// amplitudes, and relying on a coincidence of conventions across two methods
// with different physics is exactly the kind of implicit coupling that breaks
// silently when either side is edited. `so_t2_from_pno` states the spin-orbital
// derivation in its own right and `stage1_mirror_convention_is_derived` pins it.

/// Rotate a spin-orbital `t[i,j,a,b]` into the per-pair PNO bases.
///
/// Returns one `(npno × npno)` block per pair in [`PairPnoBasis::pairs`] order:
/// `t̃^{ij} = Q̃_ijᵀ t[i,j,:,:] Q̃_ij`. Occupied indices here are **spin
/// orbitals**, so `basis` must have been built over `no2` of them.
///
/// # Errors
///
/// [`FerricError::General`] when the virtual dimensions disagree with
/// `basis.nvir` or a pair index is out of range.
pub fn so_t2_to_pno(
    t: &Array4<f64>,
    basis: &PairPnoBasis,
) -> Result<Vec<Array2<f64>>, FerricError> {
    let (no_i, no_j, nv_a, nv_b) = t.dim();
    if nv_a != basis.nvir || nv_b != basis.nvir {
        return Err(FerricError::General(format!(
            "so_t2_to_pno: virtual dims ({nv_a}, {nv_b}) disagree with nvir = {}",
            basis.nvir
        )));
    }
    let mut out = Vec::with_capacity(basis.pairs.len());
    for p in &basis.pairs {
        let (i, j) = p.ij;
        if i >= no_i || j >= no_j {
            return Err(FerricError::General(format!(
                "so_t2_to_pno: pair ({i},{j}) out of range for occupied dims ({no_i}, {no_j})"
            )));
        }
        let blk = t.slice(ndarray::s![i, j, .., ..]).to_owned();
        out.push(p.transform.t().dot(&blk).dot(&p.transform));
    }
    Ok(out)
}

/// Back-transform per-pair PNO blocks into a dense spin-orbital `t[i,j,a,b]`.
///
/// The inverse of [`so_t2_to_pno`] only when nothing was truncated.
///
/// The `(j,i)` mirror is filled with `t[j,i,b,a] = +t[i,j,a,b]`: spin-orbital `t`
/// is antisymmetric separately in the occupied and the virtual pair, so swapping
/// BOTH applies two minus signs that cancel. Writing a minus here — the naive
/// reading of "spin orbitals are antisymmetric" — puts every mirror block in at
/// twice its magnitude with the wrong sign; see the module-level note for the
/// measured signature.
///
/// # Errors
///
/// [`FerricError::General`] on block-count or block-shape disagreement.
pub fn so_t2_from_pno(
    blocks: &[Array2<f64>],
    basis: &PairPnoBasis,
    no2: usize,
) -> Result<Array4<f64>, FerricError> {
    if blocks.len() != basis.pairs.len() {
        return Err(FerricError::General(format!(
            "so_t2_from_pno: got {} blocks for {} pairs",
            blocks.len(),
            basis.pairs.len()
        )));
    }
    let nvir = basis.nvir;
    let mut t = Array4::<f64>::zeros((no2, no2, nvir, nvir));
    for (p, blk) in basis.pairs.iter().zip(blocks.iter()) {
        let (i, j) = p.ij;
        let npno = p.transform.ncols();
        if blk.dim() != (npno, npno) {
            return Err(FerricError::General(format!(
                "so_t2_from_pno: block for pair ({i},{j}) is {:?}, expected ({npno}, {npno})",
                blk.dim()
            )));
        }
        if i >= no2 || j >= no2 {
            return Err(FerricError::General(format!(
                "so_t2_from_pno: pair ({i},{j}) out of range for no2 = {no2}"
            )));
        }
        let dense = p.transform.dot(blk).dot(&p.transform.t());
        for a in 0..nvir {
            for b in 0..nvir {
                t[[i, j, a, b]] = dense[(a, b)];
                // Occupied swap AND virtual swap: two sign flips that cancel.
                t[[j, i, b, a]] = dense[(a, b)];
            }
        }
    }
    Ok(t)
}

/// `t̃^{(x,y)}` for the requested orientation of a stored pair block.
///
/// `basis.pairs` holds each pair once as `(i,j)` with `i <= j`; the hh ladder
/// runs over the full `no2²` grid and asks for both orientations.
///
/// The virtual labels `a,b` are **spectators** in the hh ladder — they are
/// carried untouched from `t` to the output — so the requested `(l,k)` block is
/// the occupied swap ALONE:
///
/// ```text
///   t[l,k,a,b] = −t[k,l,a,b]        (NO transpose: a,b keep their slots)
/// ```
///
/// Writing `−blocks[p].t()` here instead would additionally swap the virtuals,
/// which is a *different* tensor and, because the double swap is the identity,
/// silently equals `+blocks[p]` — i.e. it drops the sign entirely. Both the
/// transpose and the sign have to be right, and they are not independent.
///
/// Note this holds in the PNO basis because both orientations share `Q`, so
/// `Q ᵀ(−X)Q = −(Q ᵀXQ)`.
fn so_oriented_amp(blocks: &[Array2<f64>], basis: &PairPnoBasis, p: usize, x: usize) -> Array2<f64> {
    let (i, j) = basis.pairs[p].ij;
    if x == i || i == j {
        blocks[p].clone()
    } else {
        -&blocks[p]
    }
}

// =====================================================================
// STAGE 1 — the hh ladder in the PNO basis
// =====================================================================

/// The hh ladder `x[i,j,ã,b̃] = Σ_{kl} <kl||ij> · t[k,l,a,b]`, evaluated entirely
/// in per-pair PNO bases and returned **pair-shaped**.
///
/// This is the whole kernel of LinLCCD(hh). The dense original is
/// `einsum!("klij,klab->ijab", oooo, t)`. Here it becomes, per retained pair
/// `(i,j)`:
///
/// ```text
///   x̃^{ij} = Σ_{kl} <kl||ij> · S^{ij,kl} t̃^{kl} (S^{ij,kl})ᵀ
/// ```
///
/// The virtual indices never leave the PNO spaces and never touch `nvir` — the
/// output is `npno²` per pair, not `nv2²`. That is the structural difference from
/// [`crate::dlpno_ccsd_kernel::pno_fvv_direct`] and friends, whose free virtual
/// indices forced a canonical-basis assembly.
///
/// `oooo` is the antisymmetrized spin-orbital `<kl||ij>` at `[k,l,i,j]`, exactly
/// as [`crate::linlccd`] builds it. Pairs absent from `basis` contribute zero,
/// matching the screened-amplitude convention.
///
/// # Errors
///
/// [`FerricError::General`] on block-count or `oooo`-dimension disagreement.
pub fn pno_hh_ladder(
    t_pno: &[Array2<f64>],
    oooo: &ArrayD<f64>,
    basis: &PairPnoBasis,
    overlaps: &PairOverlaps,
    index: &PairIndex,
) -> Result<Vec<Array2<f64>>, FerricError> {
    if t_pno.len() != basis.pairs.len() {
        return Err(FerricError::General(format!(
            "pno_hh_ladder: got {} amplitude blocks for {} pairs",
            t_pno.len(),
            basis.pairs.len()
        )));
    }
    let no2 = index.nocc();
    if oooo.ndim() != 4 || oooo.shape().iter().any(|&d| d != no2) {
        return Err(FerricError::General(format!(
            "pno_hh_ladder: oooo is {:?}, expected all dims = {no2}",
            oooo.shape()
        )));
    }

    let mut out = Vec::with_capacity(basis.pairs.len());
    for (p_ij, pair) in basis.pairs.iter().enumerate() {
        let (i, j) = pair.ij;
        let npno = pair.transform.ncols();
        let mut x = Array2::<f64>::zeros((npno, npno));
        for k in 0..no2 {
            for l in 0..no2 {
                let v = oooo[[k, l, i, j]];
                if v == 0.0 {
                    continue;
                }
                let Some(p_kl) = index.get(k, l) else { continue };
                let s = overlaps.get(p_ij, p_kl);
                let t = so_oriented_amp(t_pno, basis, p_kl, k);
                let conv = s.dot(&t).dot(&s.t());
                x = x + v * &conv;
            }
        }
        out.push(x);
    }
    Ok(out)
}

/// Dense oracle for [`pno_hh_ladder`]: `Σ_{kl} <kl||ij> t[k,l,a,b]`.
///
/// Written as explicit loops rather than `einsum!` so the index convention cannot
/// drift between oracle and implementation. This is exactly what
/// [`crate::linlccd`] runs as `einsum!("klij,klab->ijab", oooo, &t_t)`.
pub fn dense_hh_ladder(oooo: &ArrayD<f64>, t: &Array4<f64>) -> Result<Array4<f64>, FerricError> {
    let no2 = oooo.shape()[0];
    let (t_i, t_j, nv2, nv_b) = t.dim();
    if t_i != no2 || t_j != no2 || nv2 != nv_b {
        return Err(FerricError::General(format!(
            "dense_hh_ladder: t is {:?}, expected ({no2}, {no2}, nv, nv)",
            t.dim()
        )));
    }
    let mut x = Array4::<f64>::zeros((no2, no2, nv2, nv2));
    for i in 0..no2 {
        for j in 0..no2 {
            for k in 0..no2 {
                for l in 0..no2 {
                    let v = oooo[[k, l, i, j]];
                    if v == 0.0 {
                        continue;
                    }
                    for a in 0..nv2 {
                        for b in 0..nv2 {
                            x[[i, j, a, b]] += v * t[[k, l, a, b]];
                        }
                    }
                }
            }
        }
    }
    Ok(x)
}

// =====================================================================
// STAGE 2 — the CLOSED iteration
// =====================================================================

/// What a converged DLPNO-LinLCCD(hh) run reports, beyond the energy.
#[derive(Debug, Clone)]
#[must_use]
pub struct DlpnoLinlccdResult {
    /// Converged correlation energy, evaluated from PNO-basis amplitudes.
    pub correlation_energy: f64,
    /// Iterations taken.
    pub iterations: usize,
    /// Whether the energy-change criterion was met.
    pub converged: bool,
    /// Fraction of virtuals retained, `Σ npno / (n_pairs · nv2)`.
    pub virtual_retention: f64,
    /// Structural cost counts — no timings.
    pub flops: HhFlopCount,
    /// `max |S Sᵀ − 1|` over all pair-pair overlaps: the ONLY approximation the
    /// S-insertion introduces, and zero to round-off at `t_cut_pno = 0`.
    pub max_nonorthogonality: f64,
}

/// **THE PRIZE: a closed DLPNO iteration.**
///
/// Solves the LinLCCD(hh) amplitude equations with the amplitudes living in
/// per-pair PNO bases from the first iterate to the last. Nothing is
/// back-transformed to the canonical `nv2²` virtual space inside the loop —
/// residual, Jacobi update, DIIS error vector, and energy are all evaluated on
/// `npno²` blocks.
///
/// The loop, term for term against [`crate::linlccd`]'s:
///
/// ```text
///   E    = ¼ Σ_(ij) Σ_ãb̃ ṽ^{ij}[ã,b̃] t̃^{ij}[ã,b̃]      (over the FULL no2² grid)
///   R̃^ij = ṽ^{ij} + ½ ( S t̃ Sᵀ contracted with <kl||ij> )
///   t̃_new = R̃ / D̃                                      D̃ from PairPno::eps
/// ```
///
/// At `t_cut_pno = 0` this reproduces `linlccd(.., LadderVariant::Hh)` exactly,
/// because every `S` is orthogonal and every trace is invariant under the shared
/// rotation.
///
/// # Arguments
///
/// `t_cut_pno = 0.0` gives the exact baseline. `domains` may screen the occupied
/// pair list; passing `None` builds complete domains over all `no2` occupied spin
/// orbitals.
///
/// # Errors
///
/// [`FerricError::General`] on any shape disagreement, propagated eigensolver
/// failures, and [`FerricError::Convergence`] when `cfg.max_iter` is exhausted.
#[allow(clippy::too_many_arguments)]
pub fn dlpno_linlccd_hh(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &CcConfig,
    t_cut_pno: f64,
    domains: Option<&PairDomains>,
) -> Result<DlpnoLinlccdResult, FerricError> {
    // ---- Integrals and spin-orbital blocks: IDENTICAL provenance to linlccd ----
    let nbas = obs.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let no = ferric_mp2::rimp2::active_occ(nocc_total, cfg.frozen_core)?;
    let first_occ = cfg.frozen_core;
    let nv = nbas - nocc_total;
    let (no2, nv2) = (2 * no, 2 * nv);

    // ---- MEMORY PRE-FLIGHT: the setup, not the iteration ----
    //
    // WHY THIS EXISTS. Everything this module *claims* about memory is about the
    // ITERATION, and that claim is true: `HhFlopCount::of` and
    // `PairPnoBasis::virtual_retention` describe per-pair `npno²` blocks, and the
    // loop below never leaves them. What neither describes is the SETUP that has
    // to run before the first PNO exists — which materializes several dense
    // canonical tensors on the full `no2²·nv2²` and `naux·nbas²` grids.
    //
    // That gap is a real production incident, not a hypothetical: the LNO-coupled
    // path reported a predicted 0.055 GB working set and peaked at 7.3 GB, because
    // the number it reported was the compressed pair-shaped one while the
    // allocator was serving the dense setup. `CcConfig::memory_budget_bytes` was
    // threaded into this crate but never read here, so nothing reconciled the two
    // and the only feedback was the OOM killer.
    //
    // A `MemoryPlan` closes it by making the estimate and the allocation the same
    // statement: every dense block below is declared here, BEFORE the first one is
    // allocated, and `check()` reports a per-reservation breakdown naming the term
    // that blew up rather than a bare total. Lifetimes are read off the code, not
    // guessed:
    //
    //   * `eri3_ao` is Transient — explicitly `drop`ped once `oooo` is built.
    //   * `g_iajb` / `g_ijkl` are Transient — scoped to the block that folds them
    //     into `v_oovv` / `oooo`.
    //   * `v_oovv` AND `v4` are BOTH Resident: `v4` is a `.clone()` of `v_oovv`
    //     into a static-rank view and `v_oovv` is never dropped, so the pair is
    //     genuinely co-resident. Declaring only one would under-estimate by a
    //     factor of two on the largest term in the method.
    //   * `t_guess` is Resident — the closure handed to `PairPnoBasis::build`
    //     borrows it, so it outlives the whole PNO construction.
    //
    // Deliberately NOT declared: the per-pair PNO blocks themselves. They are
    // bounded by the dense terms already counted (`Σ_pairs npno² ≤ n_pairs·nv2²`)
    // and adding them would over-estimate — and an over-estimating guard is also
    // a bug, since it refuses jobs that would have fit.
    let naux = dfbs.nbasis();
    let no2_sq = no2.saturating_mul(no2);
    let nv2_sq = nv2.saturating_mul(nv2);
    let oovv_elems = no2_sq.saturating_mul(nv2_sq);
    let mut plan = ferric_core::memory::plan::MemoryPlan::resolve(
        cfg.memory_budget_bytes,
        format!("DLPNO-LinLCCD(hh) setup (no={no}, nv={nv} spatial, naux={naux})"),
    );
    {
        use ferric_core::memory::plan::Lifetime::{Resident, Transient};
        plan.reserve("eri3_ao (P|mu nu)", naux.saturating_mul(nbas).saturating_mul(nbas), Transient);
        plan.reserve("g_iajb (ia|jb)", (no * nv).saturating_pow(2), Transient);
        plan.reserve("g_ijkl (ij|kl)", no.saturating_pow(4), Transient);
        plan.reserve("b_ov B(P|ia)", naux.saturating_mul(no).saturating_mul(nv), Resident);
        plan.reserve("v_oovv <ij||ab>", oovv_elems, Resident);
        plan.reserve("v4 <ij||ab> (clone)", oovv_elems, Resident);
        plan.reserve("oooo <ij||kl>", no2_sq.saturating_mul(no2_sq), Resident);
        plan.reserve("t_guess t2[i,j,a,b]", oovv_elems, Resident);
    }
    plan.check()?;

    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + no]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();

    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;

    let b_ov = build_b(
        &transform_3center_ov(&eri3_ao, &c_occ, &c_vir),
        &v_inv_sqrt,
        Axis::O,
        Axis::V,
    );
    let v_oovv = {
        let g_iajb: ArrayD<f64> = einsum!("Pia,Pjb->iajb", &b_ov, &b_ov);
        asym_oovv(&g_iajb, no, nv)
    };
    let oooo = {
        let b_oo = build_b(&transform_3center_oo(&eri3_ao, &c_occ), &v_inv_sqrt, Axis::O, Axis::O);
        let g_ijkl: ArrayD<f64> = einsum!("Pij,Pkl->ijkl", &b_oo, &b_oo);
        asym_same(&g_ijkl, no)
    };
    drop(eri3_ao);

    // Spin-orbital energies: even = alpha, odd = beta (linlccd's convention).
    let mut eo = vec![0.0f64; no2];
    let mut ev = vec![0.0f64; nv2];
    for i in 0..no {
        eo[2 * i] = eps[first_occ + i];
        eo[2 * i + 1] = eps[first_occ + i];
    }
    for a in 0..nv {
        ev[2 * a] = eps[nocc_total + a];
        ev[2 * a + 1] = eps[nocc_total + a];
    }

    let v4 = v_oovv
        .clone()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| FerricError::General(format!("dlpno_linlccd_hh: oovv reshape: {e}")))?;

    // ---- MP2 guess, in the canonical basis, used to DEFINE the PNOs ----
    let t_guess = Array4::<f64>::from_shape_fn((no2, no2, nv2, nv2), |(i, j, a, b)| {
        v4[[i, j, a, b]] / (eo[i] + eo[j] - ev[a] - ev[b])
    });

    // ---- Pair domains over SPIN-ORBITAL occupieds ----
    // Default: complete domains. Centers are a synthetic line — they are used
    // only for the distance screen, which is disabled at infinite cutoff, and
    // are carried for diagnostics.
    let owned_domains;
    let domains: &PairDomains = match domains {
        Some(d) => {
            if d.nocc != no2 {
                return Err(FerricError::General(format!(
                    "dlpno_linlccd_hh: domains were built for nocc = {} but this system has \
                     no2 = {no2} occupied SPIN orbitals — domains for this method must be \
                     built over spin orbitals, not spatial ones",
                    d.nocc
                )));
            }
            d
        }
        None => {
            let centers = Array2::<f64>::from_shape_fn((no2, 3), |(i, ax)| {
                if ax == 0 {
                    i as f64
                } else {
                    0.0
                }
            });
            owned_domains = complete_pair_domains(&centers)?;
            &owned_domains
        }
    };

    // ---- The semicanonical per-pair PNO basis ----
    let basis = PairPnoBasis::build(domains, nv2, &ev, t_cut_pno, |i, j| {
        Array2::from_shape_fn((nv2, nv2), |(a, b)| t_guess[[i, j, a, b]])
    })?;
    // The `S^{P,Q}` cache is the worst-scaling allocation on this path
    // (`O(nocc⁴·nvir²)`) and is NOT covered by the setup plan above, whose terms
    // are all canonical-grid. Charge it against what the setup plan LEFT, rather
    // than against the whole budget: the dense blocks declared above are still
    // resident here, and two guards that each pass in isolation are exactly how
    // a composed peak escapes both.
    let overlaps = PairOverlaps::build_within_budget(&basis, plan.remaining())?;
    let index = PairIndex::new(&basis, no2)?;
    let denoms = pno_denominators(&basis, &eo)?;

    // ---- Per-pair driver integrals, rotated in ONCE, outside the loop ----
    let v_pno = so_t2_to_pno(&v4, &basis)?;

    // ---- Amplitudes: PNO-basis from here on. Nothing leaves. ----
    let mut t_pno: Vec<Array2<f64>> = v_pno
        .iter()
        .zip(denoms.iter())
        .map(|(v, d)| v / d)
        .collect();

    // DIIS over the concatenated per-pair blocks.
    let dim: usize = basis.pairs.iter().map(|p| p.transform.ncols().pow(2)).sum();
    let mut diis = ferric_scf::diis::Diis::new(cfg.diis_subspace.max(1));
    let mut e_old = 0.0;
    let mut converged = false;
    let mut iterations = cfg.max_iter;
    let mut e_corr = 0.0;

    for iter in 0..cfg.max_iter {
        // Energy: E = ¼ Σ_ijab <ij||ab> t[i,j,a,b], over the FULL no2² grid.
        // `basis.pairs` holds each pair once, so the off-diagonal mirror must be
        // counted too. Both v and t mirror with the SAME sign flip, so the
        // product is mirror-invariant and the mirror simply doubles the term.
        e_corr = 0.25 * pno_energy(&v_pno, &t_pno, &basis);
        let d_e = (e_corr - e_old).abs();
        if iter > 0 && d_e < cfg.energy_conv {
            converged = true;
            iterations = iter;
            break;
        }
        e_old = e_corr;

        // Residual: driver + ½ hh ladder, all pair-shaped.
        let ladder = pno_hh_ladder(&t_pno, &oooo, &basis, &overlaps, &index)?;
        let r: Vec<Array2<f64>> =
            v_pno.iter().zip(ladder.iter()).map(|(v, x)| v + 0.5 * x).collect();

        // Jacobi update on SEMICANONICAL denominators.
        let t_new: Vec<Array2<f64>> =
            r.iter().zip(denoms.iter()).map(|(ri, d)| ri / d).collect();

        // DIIS on the flattened per-pair concatenation.
        let mut flat = Array2::<f64>::zeros((dim, 1));
        let mut err = Array2::<f64>::zeros((dim, 1));
        let mut off = 0;
        for (new, old) in t_new.iter().zip(t_pno.iter()) {
            for (k, (&n, &o)) in new.iter().zip(old.iter()).enumerate() {
                flat[(off + k, 0)] = n;
                err[(off + k, 0)] = n - o;
            }
            off += new.len();
        }
        let ext = diis.step(&flat, &err);

        let mut off = 0;
        for (blk, pair) in t_pno.iter_mut().zip(basis.pairs.iter()) {
            let n = pair.transform.ncols();
            for a in 0..n {
                for b in 0..n {
                    blk[(a, b)] = ext[(off + a * n + b, 0)];
                }
            }
            off += n * n;
        }
    }

    if !converged {
        return Err(FerricError::Convergence(format!(
            "DLPNO-LinLCCD(hh) did not converge in {} iterations (last dE = {:.3e})",
            cfg.max_iter,
            (e_corr - e_old).abs()
        )));
    }

    Ok(DlpnoLinlccdResult {
        correlation_energy: e_corr,
        iterations,
        converged,
        virtual_retention: basis.virtual_retention(),
        flops: HhFlopCount::of(&basis, no2),
        max_nonorthogonality: overlaps.max_nonorthogonality(),
    })
}

/// `Σ_ijab <ij||ab> t[i,j,a,b]` over the full `no2²` grid, from PNO blocks.
///
/// `basis.pairs` holds each pair once with `i <= j`; the dense sum runs over both
/// orientations. Both `v` and `t` mirror with the same sign
/// (`X[j,i,b,a] = −X[i,j,a,b]` for both, being antisymmetrized), so the mirror
/// term equals the stored one and off-diagonal pairs count twice.
fn pno_energy(v_pno: &[Array2<f64>], t_pno: &[Array2<f64>], basis: &PairPnoBasis) -> f64 {
    let mut e = 0.0;
    for (p, pair) in basis.pairs.iter().enumerate() {
        let (i, j) = pair.ij;
        let w = if i == j { 1.0 } else { 2.0 };
        let s: f64 = v_pno[p].iter().zip(t_pno[p].iter()).map(|(a, b)| a * b).sum();
        e += w * s;
    }
    e
}

/// A dense LinLCCD(hh) reference that shares this module's PNO-independent path.
///
/// Returned so a caller can difference the two on its own system rather than
/// trusting a threshold. This runs the *same* integrals through the *dense*
/// residual, so any difference from [`dlpno_linlccd_hh`] at `t_cut_pno = 0` is
/// attributable to the PNO machinery alone and not to a different integral build.
pub fn dense_linlccd_hh(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &CcConfig,
) -> Result<CcResult, FerricError> {
    crate::linlccd::linlccd(mol, obs, dfbs, op, rhf, cfg, LadderVariant::Hh)
}

// =====================================================================
// STAGE 3 — structural cost. NO WALL CLOCK.
// =====================================================================

/// FLOP and working-set counts for the hh-ladder iteration, as **pure functions
/// of the basis**. No arithmetic is performed to obtain them.
///
/// The dense column is computed over the SAME pair list, so the ratio isolates
/// *virtual* truncation and does not silently credit the occupied pair screen —
/// that is [`ferric_mp2::pair_domains`]'s separate claim.
///
/// [`Self::transform_flops`] is reported separately and deliberately: for the
/// DLPNO-(T) sibling the per-pair transform cost **more** than the kernel it
/// saved (1.76× even at zero truncation), which made the whole construction a
/// net loss. The honest way to present a PNO cost claim is therefore
/// kernel-versus-kernel *and* transform-versus-kernel, both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HhFlopCount {
    /// `Σ_pairs npno²` versus `n_pairs · nv2²` — the amplitude working set,
    /// which is also the residual, driver, and denominator working set.
    pub amplitude_elements: (usize, usize),
    /// Largest single per-pair block, `max npno²` versus `nv2²`.
    pub max_pair_block: (usize, usize),
    /// The hh ladder itself: per `(pair, k, l)` with both sides retained, the
    /// conversion `S t̃ Sᵀ` plus the scaled accumulation.
    pub ladder_flops: (usize, usize),
    /// Building the PNO basis and rotating the driver integrals in — the
    /// **overhead** the ladder saving has to beat. One eigendecomposition of an
    /// `nv2×nv2` pair density plus an `nv2×nv2` semicanonical Fock solve plus two
    /// `nv2`-sided rotations, per pair.
    ///
    /// Paid **once**, versus `ladder_flops` paid **per iteration** — so the honest
    /// comparison is `transform_flops` against `n_iter · ladder_flops`, which
    /// [`Self::transform_vs_kernel`] computes.
    pub transform_flops: usize,
}

impl HhFlopCount {
    /// Derive the counts from a basis and an occupied spin-orbital count.
    pub fn of(basis: &PairPnoBasis, no2: usize) -> Self {
        let nv = basis.nvir;
        let npno: Vec<usize> = basis.pairs.iter().map(|p| p.transform.ncols()).collect();
        let n_pairs = npno.len();

        let amp_pno: usize = npno.iter().map(|&n| n * n).sum();
        let amp_dense = n_pairs * nv * nv;
        let max_pno = npno.iter().map(|&n| n * n).max().unwrap_or(0);

        // Conversion S(m×n) · t̃(n×n) · Sᵀ(n×m): m·n² + m²·n, plus m² for the
        // scaled accumulation. The dense analogue is the same expression at
        // m = n = nv2 — which is what the ORIGINAL einsum does per (k,l,i,j).
        let conv = |m: usize, n: usize| m * n * n + m * m * n + m * m;
        let dense_conv = conv(nv, nv);

        let idx = pair_lookup(basis);
        let mut lad_pno = 0usize;
        let mut lad_n = 0usize;
        for (p_ij, _pair) in basis.pairs.iter().enumerate() {
            for k in 0..no2 {
                for l in 0..no2 {
                    let Some(&p_kl) = idx.get(&(k, l)) else { continue };
                    lad_pno += conv(npno[p_ij], npno[p_kl]);
                    lad_n += 1;
                }
            }
        }

        // Per pair: pair density T Tᵀ + Tᵀ T (2·nv³), its eigendecomposition
        // (~9·nv³ for a symmetric QR/DC solve), the semicanonical Fock build
        // (nv·npno² for Q ᵀ diag(ε) Q exploiting the diagonal) and its eigensolve
        // (~9·npno³), Q·U (nv·npno²), and the driver rotation Qᵀ v Q
        // (nv²·npno + nv·npno²). Constants are the standard leading-order
        // flop counts; the point of the number is its SCALING in nv and npno,
        // and it is compared only against a kernel count derived the same way.
        let mut transform = 0usize;
        for &n in &npno {
            transform += 2 * nv * nv * nv; // pair density
            transform += 9 * nv * nv * nv; // eigh of the nv×nv density
            transform += nv * n * n; // semicanonical Fock
            transform += 9 * n * n * n; // eigh of the npno×npno Fock
            transform += nv * n * n; // Q·U
            transform += nv * nv * n + nv * n * n; // driver rotation
        }

        Self {
            amplitude_elements: (amp_pno, amp_dense),
            max_pair_block: (max_pno, nv * nv),
            ladder_flops: (lad_pno, lad_n * dense_conv),
            transform_flops: transform,
        }
    }

    /// PNO cost as a fraction of dense, per row. 1.0 = no compression.
    pub fn ratios(&self) -> [f64; 3] {
        let r = |(a, b): (usize, usize)| if b == 0 { 1.0 } else { a as f64 / b as f64 };
        [r(self.amplitude_elements), r(self.max_pair_block), r(self.ladder_flops)]
    }

    /// The number that decides whether the whole construction is worth anything:
    /// one-off transform cost divided by the **per-iteration** kernel saving.
    ///
    /// `n_iter` is the iteration count the saving is amortized over. Returns
    /// `None` when truncation saved nothing (the denominator is zero or
    /// negative), which is itself the answer: at that threshold the transform is
    /// pure overhead.
    ///
    /// A value below 1.0 means the transform pays for itself over `n_iter`
    /// iterations; above 1.0 means the PNO construction costs more than the
    /// kernel it accelerates. The DLPNO-(T) sibling measured 1.76 at zero
    /// truncation — pure loss — which is why this is reported rather than buried.
    pub fn transform_vs_kernel(&self, n_iter: usize) -> Option<f64> {
        let (pno, dense) = self.ladder_flops;
        if dense <= pno {
            return None;
        }
        let saved = (dense - pno) * n_iter;
        Some(self.transform_flops as f64 / saved as f64)
    }

    /// A human-readable table, for a caller reporting its own numbers.
    pub fn table(&self) -> String {
        let rows: [(&str, (usize, usize)); 3] = [
            ("amplitude elements", self.amplitude_elements),
            ("max pair block    ", self.max_pair_block),
            ("hh ladder flops   ", self.ladder_flops),
        ];
        let mut s = String::from("  quantity              PNO           dense      ratio\n");
        for (name, (a, b)) in rows {
            let ratio = if b == 0 { 1.0 } else { a as f64 / b as f64 };
            s.push_str(&format!("  {name}  {a:>12}  {b:>12}   {ratio:>6.4}\n"));
        }
        s.push_str(&format!("  transform (once)    {:>12}\n", self.transform_flops));
        s
    }
}

fn pair_lookup(basis: &PairPnoBasis) -> std::collections::HashMap<(usize, usize), usize> {
    let mut m = std::collections::HashMap::new();
    for (p, pair) in basis.pairs.iter().enumerate() {
        let (i, j) = pair.ij;
        m.insert((i, j), p);
        m.insert((j, i), p);
    }
    m
}

/// Accessor used by tests and callers wanting the per-pair PNO count directly.
pub fn pair_npno(p: &PairPno) -> usize {
    p.transform.ncols()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    // ------------------------------------------------------------------
    // Small real systems only. Every claim here is EXACTNESS or a COUNT,
    // neither of which needs size, and the box is contested.
    // ------------------------------------------------------------------

    struct Sys {
        mol: Molecule,
        obs: PreparedBasis,
        dfbs: PreparedBasis,
        rhf: ScfResult,
        op: Operator,
    }

    fn water(obs_name: &str) -> Sys {
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
        Sys { mol, obs, dfbs, rhf, op }
    }

    fn cfg() -> CcConfig {
        CcConfig { frozen_core: 0, max_iter: 100, energy_conv: 1e-10, ..Default::default() }
    }

    /// The spin-orbital pieces of a system, for the stage-1 fixtures.
    struct Blocks {
        oooo: ArrayD<f64>,
        t: Array4<f64>,
        no2: usize,
        nv2: usize,
        ev: Vec<f64>,
    }

    fn blocks(s: &Sys) -> Blocks {
        let nbas = s.obs.nbasis();
        let no = s.mol.nelec() as usize / 2;
        let nv = nbas - no;
        let (no2, nv2) = (2 * no, 2 * nv);
        let eps = s.rhf.eps_r();
        let c = s.rhf.mos_r();
        let c_occ = c.slice(ndarray::s![.., ..no]).to_owned();
        let c_vir = c.slice(ndarray::s![.., no..]).to_owned();

        let v2c = ferric_integrals::threeindex::coulomb_metric_2c(s.op, &s.dfbs).unwrap();
        let v_inv_sqrt = cholesky_inverse_sqrt(&v2c).unwrap();
        let eri3 = ferric_integrals::threeindex::eri3_tensor(s.op, &s.obs, &s.dfbs).unwrap();
        let b_ov = build_b(
            &transform_3center_ov(&eri3, &c_occ, &c_vir),
            &v_inv_sqrt,
            Axis::O,
            Axis::V,
        );
        let g_iajb: ArrayD<f64> = einsum!("Pia,Pjb->iajb", &b_ov, &b_ov);
        let v_oovv = asym_oovv(&g_iajb, no, nv);
        let b_oo = build_b(&transform_3center_oo(&eri3, &c_occ), &v_inv_sqrt, Axis::O, Axis::O);
        let g_ijkl: ArrayD<f64> = einsum!("Pij,Pkl->ijkl", &b_oo, &b_oo);
        let oooo = asym_same(&g_ijkl, no);

        let mut eo = vec![0.0; no2];
        let mut ev = vec![0.0; nv2];
        for i in 0..no {
            eo[2 * i] = eps[i];
            eo[2 * i + 1] = eps[i];
        }
        for a in 0..nv {
            ev[2 * a] = eps[no + a];
            ev[2 * a + 1] = eps[no + a];
        }
        let v4 = v_oovv.into_dimensionality::<ndarray::Ix4>().unwrap();
        let t = Array4::from_shape_fn((no2, no2, nv2, nv2), |(i, j, a, b)| {
            v4[[i, j, a, b]] / (eo[i] + eo[j] - ev[a] - ev[b])
        });
        Blocks { oooo, t, no2, nv2, ev }
    }

    fn so_centers(no2: usize) -> Array2<f64> {
        Array2::from_shape_fn((no2, 3), |(i, ax)| if ax == 0 { i as f64 } else { 0.0 })
    }

    fn build_basis(b: &Blocks, t_cut: f64) -> PairPnoBasis {
        let d = complete_pair_domains(&so_centers(b.no2)).unwrap();
        PairPnoBasis::build(&d, b.nv2, &b.ev, t_cut, |i, j| {
            Array2::from_shape_fn((b.nv2, b.nv2), |(a, bb)| b.t[[i, j, a, bb]])
        })
        .unwrap()
    }

    // =================== STAGE 1 ===================

    /// **STAGE 1, THE LOAD-BEARING RESULT.** The hh ladder, rewritten with a
    /// pair-pair overlap inserted on both virtual indices, must reproduce the
    /// dense `einsum!("klij,klab->ijab", …)` at `t_cut_pno = 0`.
    ///
    /// If this fails, nothing downstream is worth attempting. Compared
    /// elementwise against the dense oracle after back-transforming, because a
    /// wrong `S` orientation or a dropped antisymmetry sign gives a plausible
    /// tensor of the right shape rather than a crash.
    #[test]
    fn stage1_hh_ladder_is_exact_at_zero_truncation() {
        let s = water("sto-3g");
        let b = blocks(&s);
        let basis = build_basis(&b, 0.0);
        assert!(basis.is_complete(), "t_cut_pno = 0 must keep every virtual");

        let overlaps = PairOverlaps::build(&basis);
        let index = PairIndex::new(&basis, b.no2).unwrap();
        let t_pno = so_t2_to_pno(&b.t, &basis).unwrap();

        let pno = pno_hh_ladder(&t_pno, &b.oooo, &basis, &overlaps, &index).unwrap();
        let got = so_t2_from_pno(&pno, &basis, b.no2).unwrap();
        let want = dense_hh_ladder(&b.oooo, &b.t).unwrap();

        let worst =
            got.iter().zip(want.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max);
        let scale = want.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        eprintln!(
            "stage 1 (water/STO-3G, no2={}, nv2={}): max |hh(dense) - hh(PNO)| = {worst:.3e} \
             (scale {scale:.3e})",
            b.no2, b.nv2
        );
        assert!(scale > 1e-4, "the hh ladder is ~zero ({scale:.3e}) — the check is vacuous");
        assert!(worst < 1e-12, "S-inserted hh ladder is NOT exact: {worst:.3e}");
    }

    /// The same contract on a system where the PNO rotation is genuinely
    /// non-trivial. STO-3G water has `nv2 = 4`; 6-31G gives `nv2 = 16`, where the
    /// per-pair rotations mix real structure.
    #[test]
    fn stage1_hh_ladder_is_exact_water_631g() {
        let s = water("6-31g");
        let b = blocks(&s);
        let basis = build_basis(&b, 0.0);
        let overlaps = PairOverlaps::build(&basis);
        let index = PairIndex::new(&basis, b.no2).unwrap();
        let t_pno = so_t2_to_pno(&b.t, &basis).unwrap();

        let got = so_t2_from_pno(
            &pno_hh_ladder(&t_pno, &b.oooo, &basis, &overlaps, &index).unwrap(),
            &basis,
            b.no2,
        )
        .unwrap();
        let want = dense_hh_ladder(&b.oooo, &b.t).unwrap();
        let worst =
            got.iter().zip(want.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max);
        eprintln!(
            "stage 1 (water/6-31G, no2={}, nv2={}): max deviation = {worst:.3e}",
            b.no2, b.nv2
        );
        assert!(worst < 1e-12, "hh ladder not exact at 6-31G: {worst:.3e}");
    }

    /// The PREMISE that makes stage 1 non-vacuous: the off-diagonal pair-pair
    /// overlaps must be genuinely non-trivial rotations.
    ///
    /// If every pair happened to share PNOs, `S = 1`, the insertion would never
    /// be exercised, and the exactness test would pass without testing anything.
    #[test]
    fn stage1_pair_overlaps_are_nontrivial() {
        let s = water("6-31g");
        let b = blocks(&s);
        let basis = build_basis(&b, 0.0);
        let overlaps = PairOverlaps::build(&basis);

        assert!(
            overlaps.max_nonorthogonality() < 1e-10,
            "S must be orthogonal at zero truncation"
        );
        let mut worst = 0.0f64;
        for p in 0..overlaps.n_pairs() {
            for q in 0..overlaps.n_pairs() {
                if p == q {
                    continue;
                }
                let sm = overlaps.get(p, q);
                for a in 0..sm.nrows() {
                    for c in 0..sm.ncols() {
                        let want = if a == c { 1.0 } else { 0.0 };
                        worst = worst.max((sm[(a, c)] - want).abs());
                    }
                }
            }
        }
        eprintln!("stage 1: max |S^(P!=Q) - 1| = {worst:.3e} (must be LARGE)");
        assert!(
            worst > 1e-2,
            "premise failed: pair overlaps are ~identity ({worst:.3e}), so the S-insertion \
             is never exercised and stage 1 is vacuous"
        );
    }

    /// The mirror convention, pinned by an elementwise round trip over the WHOLE
    /// tensor including mirrors.
    ///
    /// This is the test that caught the module's one real bug: an earlier
    /// `so_t2_from_pno` wrote `t[j,i,b,a] = −t[i,j,a,b]`, and every mirror block
    /// came back at twice its magnitude with the wrong sign while the stored
    /// `i <= j` blocks stayed exact to 3.3e-16. A scalar check would have seen a
    /// plausible number.
    #[test]
    fn stage1_spin_orbital_round_trip_is_exact() {
        let s = water("6-31g");
        let b = blocks(&s);
        let basis = build_basis(&b, 0.0);

        let back = so_t2_from_pno(&so_t2_to_pno(&b.t, &basis).unwrap(), &basis, b.no2).unwrap();
        let worst =
            b.t.iter().zip(back.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max);
        let scale = b.t.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        eprintln!("stage 1: spin-orbital round trip max deviation = {worst:.3e} (scale {scale:.3e})");
        assert!(scale > 1e-4, "t is ~zero — the round trip check is vacuous");
        assert!(worst < 1e-12, "spin-orbital round trip is not exact: {worst:.3e}");
    }

    /// The mirror algebra `so_oriented_amp` and `so_t2_from_pno` rest on, pinned
    /// against the ACTUAL amplitudes rather than asserted in a doc comment.
    ///
    /// Three separate claims, because they are the three ways to get this wrong
    /// and only the third is the one a "spin orbitals are antisymmetric" reflex
    /// gets right:
    ///
    /// 1. occupied swap alone FLIPS sign — what the hh ladder's orientation needs;
    /// 2. virtual swap alone FLIPS sign;
    /// 3. **both** swaps KEEP sign — the two flips cancel, so the double mirror
    ///    is `+`, which is why the closed-shell fill in
    ///    [`crate::dlpno_ccsd_virtual::t2_from_pno`] is numerically fine here and
    ///    why writing a minus in `so_t2_from_pno` was a bug rather than a
    ///    convention difference.
    ///
    /// An earlier version of this module asserted (3) was a sign FLIP. It is not,
    /// and the measured consequence was every mirror block at twice its magnitude
    /// with the wrong sign.
    #[test]
    fn stage1_mirror_convention_is_derived() {
        let s = water("sto-3g");
        let b = blocks(&s);
        let (mut occ, mut vir, mut both, mut scale) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in 0..b.no2 {
            for j in 0..b.no2 {
                for a in 0..b.nv2 {
                    for c in 0..b.nv2 {
                        let x = b.t[[i, j, a, c]];
                        scale = scale.max(x.abs());
                        occ = occ.max((b.t[[j, i, a, c]] + x).abs());
                        vir = vir.max((b.t[[i, j, c, a]] + x).abs());
                        both = both.max((b.t[[j, i, c, a]] - x).abs());
                    }
                }
            }
        }
        eprintln!(
            "stage 1 mirror algebra (scale {scale:.3e}): |t[j,i,a,b]+t| = {occ:.3e}, \
             |t[i,j,b,a]+t| = {vir:.3e}, |t[j,i,b,a]-t| = {both:.3e}"
        );
        assert!(scale > 1e-4, "t is ~zero — the derivation check is vacuous");
        assert!(occ < 1e-14, "occupied swap does not flip sign: {occ:.3e}");
        assert!(vir < 1e-14, "virtual swap does not flip sign: {vir:.3e}");
        assert!(both < 1e-14, "the DOUBLE swap must KEEP sign, deviation {both:.3e}");
    }

    /// Truncation must actually change the ladder — an inert knob would make the
    /// stage-3 cost curve fictional.
    #[test]
    fn stage1_truncation_changes_the_ladder() {
        let s = water("6-31g");
        let b = blocks(&s);
        let b0 = build_basis(&b, 0.0);
        let bt = build_basis(&b, 1e-5);
        assert!(!bt.is_complete(), "test premise: 1e-5 must truncate something");

        let go = |basis: &PairPnoBasis| {
            let ov = PairOverlaps::build(basis);
            let ix = PairIndex::new(basis, b.no2).unwrap();
            let tp = so_t2_to_pno(&b.t, basis).unwrap();
            so_t2_from_pno(&pno_hh_ladder(&tp, &b.oooo, basis, &ov, &ix).unwrap(), basis, b.no2)
                .unwrap()
        };
        let x0 = go(&b0);
        let xt = go(&bt);
        let dev = x0.iter().zip(xt.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max);
        eprintln!(
            "stage 1: truncated ladder (retention {:.3}) differs by {dev:.3e}",
            bt.virtual_retention()
        );
        assert!(dev > 1e-14, "truncation had no effect on the ladder — the knob is inert");
    }

    /// Bad inputs must error rather than produce a plausible wrong number.
    #[test]
    fn stage1_invalid_inputs_are_rejected() {
        let s = water("sto-3g");
        let b = blocks(&s);
        let basis = build_basis(&b, 0.0);
        let overlaps = PairOverlaps::build(&basis);
        let index = PairIndex::new(&basis, b.no2).unwrap();
        let t_pno = so_t2_to_pno(&b.t, &basis).unwrap();

        // Wrong block count.
        assert!(
            pno_hh_ladder(&t_pno[..t_pno.len() - 1], &b.oooo, &basis, &overlaps, &index).is_err()
        );
        // oooo with the wrong dimension.
        let bad = ArrayD::<f64>::zeros(ndarray::IxDyn(&[b.no2 + 1; 4]));
        assert!(pno_hh_ladder(&t_pno, &bad, &basis, &overlaps, &index).is_err());
        // Wrong virtual dims into the transform.
        let bad_t = Array4::<f64>::zeros((b.no2, b.no2, b.nv2 + 1, b.nv2 + 1));
        assert!(so_t2_to_pno(&bad_t, &basis).is_err());
    }

    // =================== STAGE 2 — THE PRIZE ===================

    /// **THE PRIZE.** A CLOSED iteration — driver, hh ladder, Jacobi update, and
    /// DIIS all with amplitudes living in per-pair PNO bases — must converge to
    /// the same correlation energy as dense `linlccd(.., LadderVariant::Hh)`.
    ///
    /// Nothing else in this codebase has achieved a closed DLPNO iteration:
    /// [`crate::dlpno_ccsd_kernel`] proved the S-insertion contraction-by-
    /// contraction but stopped short of an iteration, for reasons that do not
    /// apply to this method (no t1, no ovvv, no vvvv, linear in T2).
    ///
    /// The comparison is against the dense solver's OWN reported energy through
    /// its OWN integral path, so a mis-built block would show as a real
    /// difference rather than as a self-consistent synthetic pass.
    #[test]
    fn stage2_closed_iteration_reproduces_dense_water_sto3g() {
        let s = water("sto-3g");
        let c = cfg();
        let dense = dense_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &c).unwrap();
        let pno =
            dlpno_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &c, 0.0, None).unwrap();

        let dev = (pno.correlation_energy - dense.correlation_energy).abs();
        eprintln!(
            "STAGE 2 (water/STO-3G): dense E_corr = {:.14}, DLPNO E_corr = {:.14}, |dE| = {dev:.3e} \
             ({} iters, retention {:.3}, max|SSᵀ-1| = {:.3e})",
            dense.correlation_energy,
            pno.correlation_energy,
            pno.iterations,
            pno.virtual_retention,
            pno.max_nonorthogonality
        );
        assert!(
            dense.correlation_energy.abs() > 1e-4,
            "E_corr is ~zero — the comparison is vacuous"
        );
        assert_eq!(pno.virtual_retention, 1.0, "t_cut_pno = 0 must keep every virtual");
        assert!(
            dev < 1e-9,
            "CLOSED PNO iteration must reproduce dense LinLCCD(hh): {:.14} vs {:.14}",
            pno.correlation_energy,
            dense.correlation_energy
        );
    }

    /// The same contract at 6-31G, where the per-pair rotations are non-trivial
    /// (`nv2 = 16` rather than STO-3G's 4) and a partially wrong transform could
    /// not pass by luck.
    #[test]
    fn stage2_closed_iteration_reproduces_dense_water_631g() {
        let s = water("6-31g");
        let c = cfg();
        let dense = dense_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &c).unwrap();
        let pno =
            dlpno_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &c, 0.0, None).unwrap();

        let dev = (pno.correlation_energy - dense.correlation_energy).abs();
        eprintln!(
            "STAGE 2 (water/6-31G): dense E_corr = {:.14}, DLPNO E_corr = {:.14}, |dE| = {dev:.3e} \
             ({} iters, retention {:.3})",
            dense.correlation_energy, pno.correlation_energy, pno.iterations, pno.virtual_retention
        );
        assert!(dev < 1e-9, "closed PNO iteration is not exact at 6-31G: {dev:.3e}");
    }

    /// The iteration must actually iterate. If the hh ladder contributed nothing,
    /// the "closed iteration" would be a one-shot MP2 and stage 2 would be
    /// pinning the driver term alone.
    #[test]
    fn stage2_the_ladder_actually_contributes() {
        let s = water("6-31g");
        let c = cfg();
        let hh = dlpno_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &c, 0.0, None).unwrap();
        let mp2 =
            crate::linlccd::linlccd(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &c, LadderVariant::DriversOnly)
                .unwrap();
        let shift = (hh.correlation_energy - mp2.correlation_energy).abs();
        eprintln!(
            "stage 2: E(hh) = {:.12}, E(drivers only / MP2) = {:.12}, ladder shift = {shift:.3e}, \
             iters = {}",
            hh.correlation_energy, mp2.correlation_energy, hh.iterations
        );
        assert!(hh.iterations > 1, "the iteration converged in {} steps — it never iterated", hh.iterations);
        assert!(
            shift > 1e-5,
            "the hh ladder shifted the energy by only {shift:.3e} — stage 2 would be pinning \
             the driver term alone"
        );
    }

    /// Domains built over SPATIAL orbitals must be REJECTED, not silently
    /// misinterpreted.
    ///
    /// This method's `PairDomains` are over spin orbitals (`no2`), a genuine
    /// difference from every other consumer of that type. A spatial-domain object
    /// has half as many occupieds and would produce a plausible wrong energy, so
    /// the mismatch is a hard error.
    #[test]
    fn stage2_spatial_domains_are_rejected() {
        let s = water("sto-3g");
        let c = cfg();
        let no = s.mol.nelec() as usize / 2;
        let spatial = complete_pair_domains(&so_centers(no)).unwrap();
        let r = dlpno_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &c, 0.0, Some(&spatial));
        assert!(r.is_err(), "spatial domains must be rejected, got {:?}", r.map(|x| x.correlation_energy));
    }

    /// Truncation must move the energy — an inert knob makes the stage-3 curve
    /// meaningless.
    #[test]
    fn stage2_truncation_changes_the_converged_energy() {
        let s = water("6-31g");
        let c = cfg();
        let e0 = dlpno_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &c, 0.0, None).unwrap();
        let et = dlpno_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &c, 1e-5, None).unwrap();
        assert!(et.virtual_retention < 1.0, "test premise: 1e-5 must truncate");
        let d = (et.correlation_energy - e0.correlation_energy).abs();
        eprintln!(
            "stage 2: E(exact) = {:.12}, E(t_cut=1e-5, retention {:.3}) = {:.12}, dE = {:+.3e}",
            e0.correlation_energy, et.virtual_retention, et.correlation_energy,
            et.correlation_energy - e0.correlation_energy
        );
        assert!(d > 1e-12, "truncation had no effect on the converged energy");
        assert!(
            et.correlation_energy.abs() < e0.correlation_energy.abs(),
            "truncation must REDUCE |E_corr|: {:.12} vs {:.12}",
            et.correlation_energy,
            e0.correlation_energy
        );
    }

    // =================== STAGE 3 — cost, structurally ===================

    /// The cost of the ladder and the working set must STRICTLY DECREASE as
    /// `t_cut_pno` rises, while the dense count stays fixed.
    ///
    /// This is the module's entire cost claim: a count derived from `npno` alone,
    /// with no arithmetic performed and no clock consulted.
    #[test]
    fn stage3_cost_strictly_decreases_with_truncation() {
        let s = water("6-31g");
        let b = blocks(&s);
        let mut prev: Option<HhFlopCount> = None;
        for &t_cut in &[0.0, 1e-6, 1e-5, 1e-4] {
            let basis = build_basis(&b, t_cut);
            let f = HhFlopCount::of(&basis, b.no2);
            eprintln!(
                "stage 3: t_cut = {t_cut:.0e}  retention {:.4}\n{}",
                basis.virtual_retention(),
                f.table()
            );
            if let Some(p) = &prev {
                assert_eq!(
                    p.amplitude_elements.1, f.amplitude_elements.1,
                    "the dense column must not move with the threshold"
                );
                assert_eq!(p.ladder_flops.1, f.ladder_flops.1);
                assert!(
                    f.amplitude_elements.0 <= p.amplitude_elements.0,
                    "working set grew with a tighter threshold"
                );
                assert!(f.ladder_flops.0 <= p.ladder_flops.0, "ladder flops grew");
            }
            prev = Some(f);
        }
        // And the loosest threshold must have STRICTLY beaten the exact one.
        let f0 = HhFlopCount::of(&build_basis(&b, 0.0), b.no2);
        let ft = HhFlopCount::of(&build_basis(&b, 1e-4), b.no2);
        assert!(
            ft.ladder_flops.0 < f0.ladder_flops.0,
            "truncation bought no ladder flops: {} vs {}",
            ft.ladder_flops.0,
            f0.ladder_flops.0
        );
        assert!(ft.amplitude_elements.0 < f0.amplitude_elements.0);
        assert_eq!(f0.ratios()[0], 1.0, "zero truncation must be exactly the dense count");
    }

    /// **THE MOST IMPORTANT NUMBER IN THE REPORT.** The one-off PNO transform
    /// cost versus the per-iteration kernel saving.
    ///
    /// The DLPNO-(T) sibling measured a transform costing 1.76× the kernel it
    /// saved *even at zero truncation* — a pure loss. This test does not assert a
    /// direction, because the honest answer might be that this method loses too;
    /// it asserts only that the number is REPORTED and computed from the counts
    /// rather than assumed. The eprintln is the deliverable.
    #[test]
    fn stage3_transform_overhead_is_reported() {
        let s = water("6-31g");
        let b = blocks(&s);
        // Iteration count from a real converged run, so the amortization is not
        // a guess.
        let c = cfg();
        let run =
            dlpno_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &c, 0.0, None).unwrap();
        let n_iter = run.iterations.max(1);

        eprintln!("stage 3: converged in {n_iter} iterations at t_cut_pno = 0");
        for &t_cut in &[0.0, 1e-6, 1e-5, 1e-4] {
            let basis = build_basis(&b, t_cut);
            let f = HhFlopCount::of(&basis, b.no2);
            let saved = f.ladder_flops.1 as i128 - f.ladder_flops.0 as i128;
            match f.transform_vs_kernel(n_iter) {
                Some(r) => eprintln!(
                    "  t_cut = {t_cut:.0e}  retention {:.4}  ladder saved {saved} flops/iter  \
                     transform {} flops  transform/(n_iter*saved) = {r:.3}  {}",
                    basis.virtual_retention(),
                    f.transform_flops,
                    if r < 1.0 { "PAYS FOR ITSELF" } else { "NET LOSS" }
                ),
                None => eprintln!(
                    "  t_cut = {t_cut:.0e}  retention {:.4}  ladder saved NOTHING — the \
                     transform ({} flops) is pure overhead",
                    basis.virtual_retention(),
                    f.transform_flops
                ),
            }
        }
        // At zero truncation the ladder saves nothing by construction, so the
        // ratio must be None. That is the honest baseline: the transform is
        // pure overhead until the threshold does something.
        let f0 = HhFlopCount::of(&build_basis(&b, 0.0), b.no2);
        assert_eq!(
            f0.transform_vs_kernel(n_iter),
            None,
            "at t_cut_pno = 0 the ladder cost equals dense, so there is no saving to amortize"
        );
        assert!(f0.transform_flops > 0, "the transform must have a nonzero cost");
    }

    /// The accuracy/cost curve on a real system, reported as data.
    ///
    /// Not an assertion about where the sweet spot is — that is a measurement,
    /// and this test's job is to produce it reproducibly. The only assertions are
    /// monotonicity of the cost and that the exact end of the curve is exact.
    #[test]
    fn stage3_accuracy_cost_curve_water_631g() {
        let s = water("6-31g");
        let c = cfg();
        let dense = dense_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &c)
            .unwrap()
            .correlation_energy;
        eprintln!("stage 3 accuracy/cost curve — water/6-31G, dense E_corr = {dense:.12}");
        eprintln!("  t_cut_pno   retention   E_corr           dE (Ha)      ladder ratio  amp ratio");

        let mut prev_flops = usize::MAX;
        for &t_cut in &[0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3] {
            let r = dlpno_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &c, t_cut, None)
                .unwrap();
            let ratios = r.flops.ratios();
            eprintln!(
                "  {t_cut:9.0e}   {:.4}     {:.12}  {:+.3e}    {:.4}        {:.4}",
                r.virtual_retention,
                r.correlation_energy,
                r.correlation_energy - dense,
                ratios[2],
                ratios[0]
            );
            if t_cut == 0.0 {
                assert!(
                    (r.correlation_energy - dense).abs() < 1e-9,
                    "the exact end of the curve is not exact: {:.3e}",
                    r.correlation_energy - dense
                );
            }
            assert!(
                r.flops.ladder_flops.0 <= prev_flops,
                "ladder cost is not monotone in t_cut_pno"
            );
            prev_flops = r.flops.ladder_flops.0;
        }
    }

    // ==================================================================
    // MEMORY BUDGET — the 0.055 GB vs 7.3 GB gap, pinned in both directions
    // ==================================================================
    //
    // `CcConfig::memory_budget_bytes` was threaded through this crate but never
    // read by the DLPNO family. The reported cost was pair-shaped
    // (`HhFlopCount`, `virtual_retention`) while the setup allocated on the dense
    // canonical grid, so the LNO-coupled sibling predicted 0.055 GB and peaked at
    // 7.3 GB with nothing in between to notice. These two tests pin BOTH
    // failure directions, because only pinning the first produces the opposite
    // defect: a guard that refuses jobs which would have fit.

    /// A budget far below the dense setup must be refused BEFORE allocating, and
    /// the message must name the term that blew up — a bare total is what made
    /// the historical incidents slow to diagnose.
    #[test]
    fn tiny_budget_is_refused_and_the_breakdown_names_the_largest_term() {
        let s = water("sto-3g");
        let cfg = CcConfig {
            // 1 kB: smaller than any block this method forms.
            memory_budget_bytes: Some(1_000),
            ..cfg()
        };
        let err = dlpno_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &cfg, 0.0, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("DLPNO-LinLCCD(hh) setup"), "must name the method: {err}");
        assert!(
            err.contains("v_oovv") || err.contains("t_guess") || err.contains("v4"),
            "breakdown must name a dense setup term: {err}"
        );
        assert!(err.contains("budget"), "must say what the ceiling was: {err}");
    }

    /// AN OVER-ESTIMATING GUARD IS ALSO A BUG. A budget that comfortably holds
    /// the real working set must still run to the same energy as the unguarded
    /// path — the guard may not cost a single job that would have fit.
    #[test]
    fn ample_budget_still_runs_and_is_unchanged() {
        let s = water("sto-3g");
        let base = cfg();
        let reference =
            dlpno_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &base, 0.0, None).unwrap();

        // 4 GB is ample for water/STO-3G by orders of magnitude, and is an
        // EXPLICIT budget, so it does not depend on the ambient env/cgroup.
        let cfg = CcConfig {
            memory_budget_bytes: Some(ferric_core::memory::gib_to_bytes(4.0)),
            ..base
        };
        let guarded =
            dlpno_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &cfg, 0.0, None).unwrap();

        assert_eq!(
            guarded.correlation_energy, reference.correlation_energy,
            "the budget guard must not perturb the numerics"
        );
        assert_eq!(guarded.iterations, reference.iterations);
    }

    /// The setup plan and the `PairOverlaps` guard must COMPOSE.
    ///
    /// The overlap cache is charged against `plan.remaining()`, not against the
    /// whole budget, because the dense setup blocks are still resident when it is
    /// built. Two guards that each pass in isolation are exactly how a composed
    /// peak escapes both — the first of the three defects `MemoryPlan`'s module
    /// docs enumerate.
    ///
    /// The boundary is found by search rather than by a hand-mirrored byte count:
    /// re-deriving the plan's arithmetic in the test would reintroduce the second
    /// estimator whose drift is the defect under repair.
    #[test]
    fn overlap_cache_is_charged_against_the_setup_plans_remainder() {
        let s = water("sto-3g");
        let base = cfg();
        let run = |bytes: usize| {
            let c = CcConfig { memory_budget_bytes: Some(bytes), ..base.clone() };
            dlpno_linlccd_hh(&s.mol, &s.obs, &s.dfbs, s.op, &s.rhf, &c, 0.0, None)
                .err()
                .map(|e| e.to_string())
        };

        // Walk the budget up from "nothing fits" and record which guard speaks.
        // Somewhere below the fully-ample budget the setup must fit while the
        // overlap cache does not; if it never does, the overlaps are being
        // charged against the whole budget and the guards are not composing.
        let ample = ferric_core::memory::gib_to_bytes(4.0);
        let mut saw_overlap_failure = false;
        let mut probe = 1_024usize;
        while probe < ample {
            match run(probe) {
                None => break, // everything fits from here up
                Some(err) if err.contains("PairOverlaps") => {
                    saw_overlap_failure = true;
                    break;
                }
                Some(err) => assert!(
                    err.contains("DLPNO-LinLCCD(hh) setup"),
                    "the only two guards on this path are the setup plan and the \
                     overlap cache; got: {err}"
                ),
            }
            probe *= 2;
        }
        assert!(
            saw_overlap_failure,
            "no budget refused the overlap cache while admitting the setup — the \
             S^(P,Q) cache is not being charged against the setup plan's remainder"
        );
    }
}
