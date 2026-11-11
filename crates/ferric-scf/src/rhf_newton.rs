//! Second-order RHF / RKS step: damped Newton via Hessian-vector products.
//!
//! The RESTRICTED closed-shell analogue of `uhf_newton.rs`. Where UHF carries
//! two independent MO sets (α, β) each with its own occ↔virt rotation κ^σ, a
//! closed-shell system has ONE MO set C with a single occupied↔virtual rotation
//! κ_{ai} (a ∈ virt, i ∈ occ). Both spins share C and share the occupation, so
//! the α/β bookkeeping of `uhf_newton.rs` collapses to one block plus explicit
//! factor-2 restricted-density bookkeeping.
//!
//! The gradient is the occupied→virtual block of the MO-basis Fock,
//!   g_{ai} = F_{ai}   (a ∈ virt, i ∈ occ),
//! which is exactly the RHF Brillouin condition (F_{ai} = 0 at convergence).
//! (Sign/scale note: the restricted orbital gradient is 4·F_ai; the constant
//! prefactor cancels between g and the diagonal-gap Hessian term below — the
//! Newton step κ = −g/(gap) and the full PCG solve are prefactor-invariant, so
//! we use g_ai = F_ai and gap = F_aa − F_ii consistently, matching the UHF path.)
//!
//! The Hessian matvec builds the AO restricted density perturbation
//!   δD = 2 · (C δD_MO Cᵀ),   δD_MO\[a,i\] = δD_MO\[i,a\] = κ_{ai}
//! (the factor 2 is the closed-shell double occupation, matching D = 2·C_occ·C_occᵀ),
//! rebuilds J on δD and K on δD, and — for KS — adds the XC-kernel response using
//! the SAME `FxcKernelStore`/`FxcResponse` closure the ROKS/UKS paths use, called
//! at the restricted point δD_α = δD_β = ½·δD (each spin carries half the total
//! restricted density perturbation). It then forms
//!   δF = δJ − ½·k_mix·δK + δV_xc,
//! projects δF to MO basis, reads the occ→virt block, and adds the diagonal
//! orbital-energy-gap term (F_aa − F_ii)·κ_ai.
//!
//! Rotation is applied via the Cayley unitary U = (I − κ/2)^{−1}(I + κ/2), which
//! exactly preserves orthonormality (identical to `uhf_newton.rs`).
//!
//! # Two Hessians in this module
//!
//! [`hessian_matvec`](crate::rhf_newton::hessian_matvec) is the SINGLET channel
//! (κ_α = +κ_β), the one the Newton solver drives and the one RHF's own energy
//! is stationary in.
//! [`triplet_hessian_matvec`](crate::rhf_newton::triplet_hessian_matvec) is the
//! TRIPLET channel (κ_β = −κ_α), used only
//! by the external stability analysis
//! ([`stability::rhf_external_stability`](crate::stability::rhf_external_stability)):
//! it differs by the ABSENCE of the δJ term, because equal-and-opposite spin
//! density perturbations cancel in δD_total. Keeping them as separate
//! functions rather than one flag is deliberate — the Newton path must never
//! be able to pick up the triplet operator by accident.

use crate::df_k::DfK;
use crate::engine_pool::EnginePool;
use crate::fock::{JBuilder, KBuilder};
use crate::rhf::build_jk_with_pool_homogeneous;
use crate::rohf_newton::FxcResponse;
use crate::screening::SchwarzBounds;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;
use ndarray_linalg::Solve;
use std::cell::{Cell, RefCell};

thread_local! {
    /// Singlet Hessian-vector products computed on this thread (test seam).
    static HVP_CALLS: Cell<usize> = const { Cell::new(0) };
    /// Direct four-centre response J/K builds made by closed-shell (singlet
    /// and triplet) HVPs (test seam).
    static DIRECT_RESPONSE_BUILDS: Cell<usize> = const { Cell::new(0) };
    /// Response builds made through one of the SCF's own UNFITTED builders
    /// (DirectJ, DirectK, COSX) by any second-order product (test seam).
    static UNFITTED_RESPONSE_BUILDS: Cell<usize> = const { Cell::new(0) };
}

/// Number of singlet [`hessian_matvec`] calls made on the calling thread so
/// far. A test seam: the Newton PCG loop is sequential on its caller's thread,
/// so a per-thread count is race-free under the parallel test harness.
#[doc(hidden)]
pub fn hvp_calls_on_this_thread() -> usize {
    HVP_CALLS.with(Cell::get)
}

/// Number of direct four-centre J/K builds the closed-shell (singlet and
/// triplet) HVPs made on the calling thread so far (test seam). Zero across a
/// Newton step means every response term came from the SCF's own builders
/// (fitted or unfitted) or was not needed.
#[doc(hidden)]
pub fn direct_response_builds_on_this_thread() -> usize {
    DIRECT_RESPONSE_BUILDS.with(Cell::get)
}

/// Number of response J or K builds that went through one of the SCF's own
/// UNFITTED builders (`DirectJ`, `DirectK`, COSX) on the calling thread so far,
/// from any closed- or open-shell second-order product (test seam; see
/// `unfitted_response`).
#[doc(hidden)]
pub fn unfitted_response_builds_on_this_thread() -> usize {
    UNFITTED_RESPONSE_BUILDS.with(Cell::get)
}

/// `M[δD]` from one of the SCF's own unfitted, linear J or K builders (`DirectJ`,
/// `DirectK`, COSX), made exactly homogeneous in `δD`.
///
/// Those builders screen on ABSOLUTE thresholds (Schwarz × `max|D|`, COSX's
/// density-driven pair screen and `|D|` row mask), which is right for a Fock
/// build and wrong for a response operand whose norm shrinks with the PCG /
/// Davidson residual. As in [`crate::rhf::build_jk_with_pool_homogeneous`],
/// `δD` is scaled by the [`crate::rhf::pow2_normalizer`] of `max|δD|` (exact,
/// a power of two) so every screen sees `max|δD| ∈ (½, 1]`, and the result is
/// scaled back. A zero `δD` gives zero without a build. `build` receives a
/// zeroed output (builders may add to it or overwrite it).
///
/// The builder must be a plain linear contraction for an INDEFINITE `δD`:
/// `DirectJ`, `DirectK` and COSX are; LinK's density-pair list is not, and is
/// never lent.
pub(crate) fn unfitted_response(
    dd: &Array2<f64>,
    build: impl FnOnce(&Array2<f64>, &mut Array2<f64>) -> Result<usize, FerricError>,
) -> Result<Array2<f64>, FerricError> {
    let mut out = Array2::<f64>::zeros(dd.raw_dim());
    let dmax = dd.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let Some((scale, inv_scale)) = crate::rhf::pow2_normalizer(dmax) else {
        return Ok(out);
    };
    UNFITTED_RESPONSE_BUILDS.with(|c| c.set(c.get() + 1));
    if scale == 1.0 {
        build(dd, &mut out)?;
    } else {
        build(&(inv_scale * dd), &mut out)?;
        out *= scale;
    }
    Ok(out)
}

/// The SCF's own J/K builders, lent to the closed-shell second-order products
/// (Newton [`hessian_matvec_jk`], TRAH, singlet and triplet stability) so the
/// response `δJ`/`δK` is the derivative of the Fock the SCF actually builds.
///
/// # Why
///
/// An RI-J / DF-K SCF minimises the density-FITTED energy. Its orbital Hessian
/// is the response of the same fitted J and K, on the same `B[P,μ,ν]`. A direct
/// four-centre response is the second derivative of a different functional:
/// it is off by the fitting error, which does not shrink with a
/// finite-difference step, and it cost 55-85 SCF iterations per product at
/// C8H18/def2-TZVP, where the SCF itself never builds a four-centre integral.
/// A mixed route (DF-J + direct K, DF-K + conventional J) builds its other
/// half with an unfitted builder; that half of the response comes from the
/// SAME builder ([`Self::with_unfitted_j`], [`Self::with_unfitted_k`]). Any
/// linear K builder can be lent (COSX included, FD-anchored); the SCF does not
/// lend COSX, whose K costs more than the direct build (`rhf::lend_scf_jk`).
///
/// # Which builder answers which term
///
/// * `δJ`: the SCF's RI-J when lent, else its unfitted J builder (`DirectJ`)
///   when lent, else the direct build.
/// * `δK`: only when the Fock consumes exact exchange through the ω = 0 kernel
///   (`inp.rsh` is `None` and `inp.k_mix_sr != 0`). Then the SCF's DF-K
///   (occupied-factor path) when lent, else its unfitted K builder
///   (`DirectK`, density path) when lent, else the direct build. A pure
///   functional builds no K at all.
///
/// The unfitted builders run through `unfitted_response`, which makes their
/// absolute screens homogeneous in `δD`. When a needed term has no lent
/// builder, the combined direct `build_jk_with_pool_homogeneous` runs once and
/// supplies it, as it did for every term before this type existed.
///
/// # Interior mutability
///
/// [`JBuilder::build`] and [`KBuilder::build_from_occ`] take `&mut self`; the
/// matvec takes `&self` inside the sequential PCG loop, so each builder sits in
/// a [`RefCell`] borrowed and released inside one matvec (the same pattern as
/// [`crate::rsh_response::RshResponse`]). The builders are BORROWED from the
/// SCF loop, not rebuilt, so the Newton step reads the identical fitted
/// tensors the Fock was assembled from.
pub struct ScfJkResponse<'a, 'd> {
    j: Option<RefCell<&'a mut (dyn JBuilder + 'd)>>,
    k: Option<RefCell<&'a mut DfK<'d>>>,
    j_unfitted: Option<RefCell<&'a mut (dyn JBuilder + 'd)>>,
    k_unfitted: Option<RefCell<&'a mut (dyn KBuilder + 'd)>>,
}

impl<'a, 'd> ScfJkResponse<'a, 'd> {
    /// Lend the SCF's RI-J builder (`j`) and ω = 0 DF-K builder (`k`); either
    /// may be `None`, in which case that term comes from an unfitted builder
    /// lent with [`Self::with_unfitted_j`] / [`Self::with_unfitted_k`], else
    /// from the direct build.
    pub fn new<J: JBuilder + 'd>(j: Option<&'a mut J>, k: Option<&'a mut DfK<'d>>) -> Self {
        Self {
            j: j.map(|j| RefCell::new(j as &'a mut (dyn JBuilder + 'd))),
            k: k.map(RefCell::new),
            j_unfitted: None,
            k_unfitted: None,
        }
    }

    /// Lend the SCF's unfitted Coulomb builder (`DirectJ` on a DF-K +
    /// conventional-J route). Read only when no fitted J is lent.
    pub fn with_unfitted_j(mut self, j: Option<&'a mut (dyn JBuilder + 'd)>) -> Self {
        self.j_unfitted = j.map(RefCell::new);
        self
    }

    /// Lend the SCF's unfitted ω = 0 exchange builder (`DirectK` on DF-J +
    /// direct K; any linear builder works, the SCF does not lend COSX). Read
    /// only when no DF-K is lent. Never LinK: its density-pair list is not
    /// linear in an indefinite `δD`.
    pub fn with_unfitted_k(mut self, k: Option<&'a mut (dyn KBuilder + 'd)>) -> Self {
        self.k_unfitted = k.map(RefCell::new);
        self
    }

    /// `δJ[δD]` from a lent builder, or `None` when none is lent.
    fn j_response(&self, dd: &Array2<f64>) -> Option<Result<Array2<f64>, FerricError>> {
        if let Some(cell) = self.j.as_ref() {
            let mut dj = Array2::<f64>::zeros(dd.raw_dim());
            return Some(cell.borrow_mut().build(dd, &mut dj).map(|_| dj));
        }
        let cell = self.j_unfitted.as_ref()?;
        Some(unfitted_response(dd, |d, out| {
            cell.borrow_mut().build(d, out)
        }))
    }

    /// `δK` of the total restricted perturbation `δD = 2·(Y·Xᵀ + X·Yᵀ)` from a
    /// lent builder (`dd` is that `δD`), or `None` when none is lent.
    fn k_response(
        &self,
        c: &Array2<f64>,
        k_ov: &Array2<f64>,
        nocc: usize,
        dd: &Array2<f64>,
    ) -> Option<Result<Array2<f64>, FerricError>> {
        if let Some(cell) = self.k.as_ref() {
            return Some(fitted_response_k(cell, c, k_ov, nocc));
        }
        let cell = self.k_unfitted.as_ref()?;
        Some(unfitted_response(dd, |d, out| {
            let mut kb = cell.borrow_mut();
            kb.update_density(d);
            kb.build(d, out)
        }))
    }
}

impl std::fmt::Debug for ScfJkResponse<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScfJkResponse")
            .field("fitted_j", &self.j.is_some())
            .field("fitted_k", &self.k.is_some())
            .field("unfitted_j", &self.j_unfitted.is_some())
            .field("unfitted_k", &self.k_unfitted.is_some())
            .finish()
    }
}

/// Inputs to one RHF/RKS Newton step.
pub struct RhfNewtonInputs<'a> {
    /// The AO basis on the molecule.
    pub prep: &'a PreparedBasis,
    /// Schwarz bounds for the response J/K builds.
    pub bounds: &'a SchwarzBounds,
    /// MO coeffs (n × n), columns ordered occ | virt.
    pub c: &'a Array2<f64>,
    /// Fock in the MO basis (current iter).
    pub f_mo: &'a Array2<f64>,
    /// Number of doubly occupied MOs.
    pub nocc: usize,
    /// K mixing coefficient for the plain-Coulomb exchange response
    /// (1.0 for HF, c_HF for a global hybrid). Read ONLY when `rsh` is `None`.
    pub k_mix_sr: f64,
    /// Range-separated exchange response (ω ≠ 0 only). `Some(..)` replaces the
    /// single Coulomb-kernel `k_mix_sr · δK` with the SR/LR combination
    /// `c_SR·δK[erfc(ω)] + c_LR·δK[erf(ω)]` built from the SAME `DfK` fitters
    /// the converged Fock was assembled from — see [`crate::rsh_response`].
    /// `None` (ω = 0) leaves the Coulomb path bit-identical to the pre-#314 code.
    pub rsh: Option<&'a crate::rsh_response::RshResponse<'a>>,
    /// Optional XC-kernel response closure (None for pure RHF). Called with the
    /// per-spin restricted density perturbation (δD_α = δD_β = ½·δD_total).
    pub fxc: Option<&'a FxcResponse<'a>>,
    /// Integral screening threshold for the response J/K builds (the solvers
    /// pass `RhfConfig::integral_thresh`).
    pub thresh: f64,
    /// Solver-resolved memory budget (see `rhf::resolve_three_index_budget`),
    /// used to size the `build_jk_with_pool` reduction band via
    /// `reduce::resolve_band_bytes` — never affects results, only the
    /// live-set/parallel width of the deterministic reduction.
    pub ooc_budget: usize,
}

/// One damped-Newton step on RHF/RKS MO coefficients.
///
/// Returns `(C_new, kmax)` where `kmax` is the ∞-norm of the rotation (a
/// step-size diagnostic). `level_shift` is added to the diagonal preconditioner
/// gap; `max_step` is the componentwise trust radius on κ. `jk` lends the
/// SCF's density-fitted J/K builders to every Hessian-vector product (see
/// [`ScfJkResponse`]); `None` builds the response J and K directly.
pub fn rhf_newton_step(
    ctx: &ParallelContext,
    inp: &RhfNewtonInputs,
    jk: Option<&ScfJkResponse<'_, '_>>,
    level_shift: f64,
    max_step: f64,
    cg_max_iter: usize,
    cg_conv: f64,
) -> Result<(Array2<f64>, f64), FerricError> {
    let n = inp.c.nrows();
    let no = inp.nocc;

    // EnginePool is geometry/basis-only (density-independent) — build ONCE
    // here and reuse across every hessian_matvec call in the PCG loop below,
    // instead of build_jk constructing a fresh pool per call. Reduction order
    // (grouped_deterministic_sum, inside build_jk_with_pool) is unchanged, so
    // results stay bit-identical across thread counts.
    let pool = EnginePool::new(
        inp.bounds.op,
        inp.prep,
        ferric_integrals::engine_pool::eri_precision(),
    )?;

    // Gradient g_{ai} = F_{ai}  (rows = virt, cols = occ).
    let g = occ_virt_block(inp.f_mo, no, n);

    // Diagonal preconditioner gap (F[a,a] − F[i,i]) + shift.
    let f_diag: Vec<f64> = (0..n).map(|i| inp.f_mo[(i, i)]).collect();
    let diag = build_gap(&f_diag, no, n, level_shift);

    // Initial guess: diagonal solve κ⁰ = −g / gap.
    let mut k = neg_div(&g, &diag);

    // Gradient already below tolerance: the step is κ⁰ itself, and the CG
    // residual (whose first term is an HVP) would never be read, so skip it.
    // Each HVP is a full response J/K plus f_xc build, and near convergence
    // this is the common case (wiki/notes/scf-second-order-cost-2026-11).
    let gmax = max_abs(&g);
    if gmax >= cg_conv {
        pcg_refine(ctx, inp, jk, &pool, &g, &diag, &mut k, cg_max_iter, cg_conv)?;
    }

    // Trust-radius clip.
    let kmax = max_abs(&k);
    if kmax > max_step {
        let s = max_step / kmax;
        k.mapv_inplace(|x| x * s);
    }

    let c_new = apply_cayley(inp.c, &k, no, n)?;
    Ok((c_new, kmax))
}

/// Preconditioned conjugate-gradient refinement of `k` (on entry the diagonal
/// guess κ⁰) towards H·κ = −g.
#[allow(clippy::too_many_arguments)]
fn pcg_refine(
    ctx: &ParallelContext,
    inp: &RhfNewtonInputs,
    jk: Option<&ScfJkResponse<'_, '_>>,
    pool: &EnginePool,
    g: &Array2<f64>,
    diag: &Array2<f64>,
    k: &mut Array2<f64>,
    cg_max_iter: usize,
    cg_conv: f64,
) -> Result<(), FerricError> {
    let h0 = hessian_matvec_jk(ctx, inp, k, pool, jk)?;
    let mut r = &neg(g) - &h0;
    let mut z = &r / diag;
    let mut p = z.clone();
    let mut rz_old = inner(&r, &z);

    for _it in 0..cg_max_iter {
        if max_abs(&r) < cg_conv {
            break;
        }
        let ap = hessian_matvec_jk(ctx, inp, &p, pool, jk)?;
        let p_ap = inner(&p, &ap);
        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rz_old / p_ap;
        k.zip_mut_with(&p, |k, &v| *k += alpha * v);
        r.zip_mut_with(&ap, |r, &v| *r -= alpha * v);
        z = &r / diag;
        let rz_new = inner(&r, &z);
        let beta = rz_new / rz_old;
        rz_old = rz_new;
        p.zip_mut_with(&z, |p, &v| *p = v + beta * *p);
    }
    Ok(())
}

/// H · κ for the occ→virt block, with the response J and K built directly.
/// Public so tests can finite-difference-validate the matvec directly against
/// the orbital-gradient derivative. Identical to [`hessian_matvec_jk`] with
/// `jk = None`.
pub fn hessian_matvec(
    ctx: &ParallelContext,
    inp: &RhfNewtonInputs,
    k: &Array2<f64>,
    pool: &EnginePool,
) -> Result<Array2<f64>, FerricError> {
    hessian_matvec_jk(ctx, inp, k, pool, None)
}

/// H · κ for the occ→virt block, with the response J/K from the SCF's own
/// density-fitted builders when `jk` lends them (see [`ScfJkResponse`] for
/// which builder answers which term). `None` is the direct build of
/// [`hessian_matvec`], bit for bit.
///
/// # Errors
///
/// Propagates a failed J/K build or exchange response.
pub fn hessian_matvec_jk(
    ctx: &ParallelContext,
    inp: &RhfNewtonInputs,
    k: &Array2<f64>,
    pool: &EnginePool,
    jk: Option<&ScfJkResponse<'_, '_>>,
) -> Result<Array2<f64>, FerricError> {
    HVP_CALLS.with(|c| c.set(c.get() + 1));
    let n = inp.c.nrows();
    let no = inp.nocc;

    // Single-set MO density perturbation, then AO transform.
    // δD_single[a,i] = δD_single[i,a] = κ[a,i]; δD_single_ao = C·δD_MO·Cᵀ.
    let dd_single_ao = ao_from_ov(inp.c, k, no, n);
    // Restricted (closed-shell) total density perturbation: D = 2·D_closed
    // ⇒ δD = 2·δD_single. J and K in the RHF Fock are built on this total D.
    let dd_ao = 2.0 * &dd_single_ao;

    // δJ and δK on the total restricted density perturbation. `dk` is `None`
    // only when `jk` is lent and the Fock consumes no ω = 0 exact exchange.
    let (dj, dk) = response_jk(ctx, inp, k, pool, jk, &dd_ao)?;

    // F = H + J − ½·K_total  ⇒  δF = δJ − ½·δK_total. For ω = 0, K_total =
    // k_mix·K[Coulomb]; for ω ≠ 0 it is c_SR·K[erfc(ω)] + c_LR·K[erf(ω)] from
    // the Fock's own fitters (`scale = 0.5` in `subtract_rsh_exchange`'s
    // restricted call, which is the ½ here). With `jk = None` the ω = 0 arm is
    // the untouched pre-#314 expression (δK always built, even at k_mix = 0),
    // so a direct run is bit-identical; `dk` is unused on the RSH arm.
    let mut df: Array2<f64> = match (inp.rsh, dk) {
        (Some(rsh), _) => &dj - &(0.5 * &rsh.exchange_response(&dd_ao)?),
        (None, Some(dk)) => &dj - &(0.5 * inp.k_mix_sr * &dk),
        (None, None) => dj,
    };

    if let Some(fxc) = inp.fxc {
        // The f_xc response expects the per-spin perturbation. For a restricted
        // point δD_α = δD_β = ½·δD_total = δD_single. Both spin outputs are
        // identical here; add one spin's δV_xc (the closed-shell restricted
        // response, matching V_xc appearing once in the restricted Fock).
        let (dvxc, _dvxc_b) = fxc(&dd_single_ao, &dd_single_ao);
        df = &df + &dvxc;
    }

    // Project to MO basis and read the occ→virt block.
    let df_mo = inp.c.t().dot(&df).dot(inp.c);
    let mut h = occ_virt_block(&df_mo, no, n);

    // Diagonal orbital-energy-gap term: + (F[a,a] − F[i,i]) · κ[a,i].
    let f_diag: Vec<f64> = (0..n).map(|i| inp.f_mo[(i, i)]).collect();
    for (ir, a) in (no..n).enumerate() {
        for (ic, i) in (0..no).enumerate() {
            h[(ir, ic)] += (f_diag[a] - f_diag[i]) * k[(ir, ic)];
        }
    }

    Ok(h)
}

/// `(δJ, δK)` of the total restricted density perturbation `dd_ao`; see
/// [`ScfJkResponse`]. `δK` is `None` only when `jk` is lent and no ω = 0 exact
/// exchange is consumed.
fn response_jk(
    ctx: &ParallelContext,
    inp: &RhfNewtonInputs,
    k_ov: &Array2<f64>,
    pool: &EnginePool,
    jk: Option<&ScfJkResponse<'_, '_>>,
    dd_ao: &Array2<f64>,
) -> Result<(Array2<f64>, Option<Array2<f64>>), FerricError> {
    let Some(jk) = jk else {
        let (dj, dk) = direct_response_jk(ctx, inp, pool, dd_ao)?;
        return Ok((dj, Some(dk)));
    };
    // The lent builds below run OUTSIDE the SCF's J/K ladder scope; without
    // this guard a mixed-precision device kernel would treat them as mixed, and
    // the ± occupied-factor δK would carry an f32 error independent of |κ|.
    let _f64 = crate::jk_ladder::F64BuildScope::enter();
    let need_k = inp.rsh.is_none() && inp.k_mix_sr != 0.0;
    let mut direct: Option<(Array2<f64>, Array2<f64>)> = None;
    let dj = match jk.j_response(dd_ao) {
        Some(r) => r?,
        None => direct
            .insert(direct_response_jk(ctx, inp, pool, dd_ao)?)
            .0
            .clone(),
    };
    let dk = if need_k {
        Some(match jk.k_response(inp.c, k_ov, inp.nocc, dd_ao) {
            Some(r) => r?,
            None => match direct.take() {
                Some((_, dk)) => dk,
                None => direct_response_jk(ctx, inp, pool, dd_ao)?.1,
            },
        })
    } else {
        None
    };
    Ok((dj, dk))
}

/// The direct four-centre `(δJ, δK)` (the only path before [`ScfJkResponse`]).
fn direct_response_jk(
    ctx: &ParallelContext,
    inp: &RhfNewtonInputs,
    pool: &EnginePool,
    dd_ao: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    DIRECT_RESPONSE_BUILDS.with(|c| c.set(c.get() + 1));
    let n = dd_ao.nrows();
    let mut dj = Array2::<f64>::zeros((n, n));
    let mut dk = Array2::<f64>::zeros((n, n));
    let band_bytes = crate::reduce::resolve_band_bytes(inp.ooc_budget);
    build_jk_with_pool_homogeneous(
        ctx, inp.prep, inp.bounds, inp.thresh, dd_ao, &mut dj, &mut dk, pool, band_bytes,
    )?;
    Ok((dj, dk))
}

/// DF-K of the total restricted perturbation `δD = 2·(Y·Xᵀ + X·Yᵀ)` with
/// `X = C_occ`, `Y = C_virt·κ`, through the SCF's occupied-factor path
/// ([`occ_pair_response_k`]).
fn fitted_response_k(
    dfk: &RefCell<&mut DfK<'_>>,
    c: &Array2<f64>,
    k_ov: &Array2<f64>,
    nocc: usize,
) -> Result<Array2<f64>, FerricError> {
    let mut dfk = dfk.borrow_mut();
    occ_pair_response_k(
        &mut |c_occ, out| dfk.build_from_occ(c_occ, out).map(|_| ()),
        c,
        k_ov,
        nocc,
    )
}

/// `K[(X+Y)(X+Y)ᵀ − (X−Y)(X−Y)ᵀ] = K[2·(Y·Xᵀ + X·Yᵀ)]` with `X = C[:, ..nocc]`
/// and `Y = C[:, nocc..]·κ` (`κ` is the `(n − nocc) × nocc` virt×occ block),
/// from an occupied-factor exchange build `k_occ(C_f, out)` that writes
/// `K[C_f·C_fᵀ]` into a zeroed `out` ([`KBuilder::build_from_occ`]).
///
/// The difference of squares is an exact identity and K is linear, so this is
/// two `nocc`-column builds (each the cost of one SCF DF-K build) instead of
/// the density path's `O(naux·n³)`. The closed-shell δD is exactly this; a
/// single spin's `δD_σ = Y·Xᵀ + X·Yᵀ` is half of it (callers scale by ½,
/// a power of two). `nocc == 0` gives zero.
pub(crate) fn occ_pair_response_k(
    k_occ: &mut dyn FnMut(&Array2<f64>, &mut Array2<f64>) -> Result<(), FerricError>,
    c: &Array2<f64>,
    k_ov: &Array2<f64>,
    nocc: usize,
) -> Result<Array2<f64>, FerricError> {
    let n = c.nrows();
    let mut k_plus = Array2::<f64>::zeros((n, n));
    if nocc == 0 {
        return Ok(k_plus);
    }
    let x = c.slice(ndarray::s![.., ..nocc]);
    let y = c.slice(ndarray::s![.., nocc..]).dot(k_ov);
    let plus = &x + &y;
    let minus = &x - &y;
    let mut k_minus = Array2::<f64>::zeros((n, n));
    k_occ(&plus, &mut k_plus)?;
    k_occ(&minus, &mut k_minus)?;
    k_plus -= &k_minus;
    Ok(k_plus)
}

/// **Triplet (RHF→UHF) orbital-Hessian matvec** — the external-stability
/// operator, `H_triplet · κ`.
///
/// This is the operator that answers "is this RHF solution a minimum against
/// breaking SPIN symmetry?", i.e. against the rotation `κ_β = −κ_α` that
/// [`hessian_matvec`](crate::rhf_newton::hessian_matvec) (which hard-codes
/// `κ_β = +κ_α`) cannot express. It is a
/// genuinely different Hessian block, not a scaled version of the singlet one:
///
/// ```text
///   singlet  (κ_α = +κ_β):   (ε_a − ε_i)κ + [ δJ(δD) − ½·k_mix·δK(δD) ]_ai
///   triplet  (κ_α = −κ_β):   (ε_a − ε_i)κ         − ½·k_mix·δK(δD)    _ai
/// ```
///
/// with `δD = 2·C(κ_MO)Cᵀ` in both cases (see
/// [`hessian_matvec`](crate::rhf_newton::hessian_matvec)). **The
/// Coulomb term is ABSENT**, and that is the whole physical content of the
/// channel: in the triplet rotation the α and β density perturbations are
/// equal and opposite, so `δD_total = δD_α + δD_β = 0` and the Coulomb
/// response — which sees only the total density — cancels exactly. Exchange
/// does not cancel, because each spin's `K` sees only its own
/// `δD_σ`, and `K(−δD) = −K(δD)` flips sign together with `κ_β`, leaving
/// `−½·K(δD)` on the α block and its negative on the β block. The two blocks
/// therefore carry the same operator and the problem reduces to the single
/// α block implemented here.
///
/// Equal, with no overall factor, to PySCF's
/// `scf.stability._gen_hop_rhf_external` → `hop_rhf2uhf`, which builds
/// `(ε_a − ε_i)κ − ½K(2C(κ+κᵀ)Cᵀ)`. The in-tree references
/// (`testdata/reference/validation/scf_stability/*.json`,
/// `/rhf/triplet_rhf_to_uhf/lowest`) are stored in exactly this convention.
///
/// # Equivalently: the triplet block of the UHF Hessian at the RHF point
///
/// Evaluating [`uhf_newton::hessian_matvec`](crate::uhf_newton::hessian_matvec)
/// with `C_α = C_β = C_RHF` and `F_α = F_β = F_RHF` gives a Hessian that
/// block-diagonalizes EXACTLY into the singlet `(κ, +κ)/√2` and triplet
/// `(κ, −κ)/√2` channels (the coupling block vanishes identically). This
/// function reproduces the triplet block of that construction, and that
/// agreement — two independent constructions of the same operator — is the
/// exactness anchor in `scf_stability_external.rs`.
///
/// # Cost
///
/// `build_jk_with_pool` is a combined J+K builder, so `δJ` is BUILT AND
/// DISCARDED here: one J matrix of wasted work per matvec. The crate's K-only
/// path (`direct_k::DirectK`) screens on a global density maximum
/// (`DensityScreen::Global`) where `build_jk_with_pool` screens on the
/// shell-blocked `d_max_shell` table, so swapping it in would change `δK` in
/// the last digits and break the bit-level agreement with the UHF-at-RHF
/// construction that the anchor rests on. The combined build is kept
/// deliberately: the anchor is worth more than the J.
///
/// # KS references
///
/// There is no `fxc` term here, and `inp.fxc` is IGNORED rather than applied:
/// the triplet XC response kernel is `f_αα − f_αβ`, which the closed-shell
/// [`FxcResponse`] does not provide (it returns the SINGLET combination
/// `f_αα + f_αβ`). Callers must not reach this function with a KS reference;
/// [`stability::rhf_external_stability`](crate::stability::rhf_external_stability)
/// refuses them with
/// [`StabilitySkip::TripletXcKernel`](crate::stability::StabilitySkip::TripletXcKernel)
/// before any matvec runs. `k_mix_sr` IS honoured, so the hybrid exact-exchange
/// fraction is correct — it is only the semilocal response that is missing.
///
/// # Errors
///
/// Propagates a failed J/K build.
pub fn triplet_hessian_matvec(
    ctx: &ParallelContext,
    inp: &RhfNewtonInputs,
    k: &Array2<f64>,
    pool: &EnginePool,
) -> Result<Array2<f64>, FerricError> {
    triplet_hessian_matvec_jk(ctx, inp, k, pool, None)
}

/// [`triplet_hessian_matvec`] with the exchange response from the SCF's own
/// ω = 0 exchange builder when `jk` lends one (DF-K through the occupied-factor
/// path, or COSX / `DirectK` through `unfitted_response`; see
/// [`ScfJkResponse`]). The triplet channel has no `δJ`, so a lent J builder is
/// never read and no Coulomb response is built. `None` is
/// [`triplet_hessian_matvec`] bit for bit (the combined direct build whose
/// `δJ` is discarded, kept for the UHF-at-RHF anchor; see its Cost section).
///
/// # Errors
///
/// Propagates a failed J/K build or exchange response.
pub fn triplet_hessian_matvec_jk(
    ctx: &ParallelContext,
    inp: &RhfNewtonInputs,
    k: &Array2<f64>,
    pool: &EnginePool,
    jk: Option<&ScfJkResponse<'_, '_>>,
) -> Result<Array2<f64>, FerricError> {
    let n = inp.c.nrows();
    let no = inp.nocc;

    // SAME density perturbation as the singlet channel: δD = 2·C(κ+κᵀ)Cᵀ.
    // The triplet character is NOT in δD, it is in which response terms act
    // on it (see the doc above) — building a different δD here would be the
    // classic way to get a plausible-looking wrong operator.
    let dd_ao = 2.0 * &ao_from_ov(inp.c, k, no, n);

    // δF_triplet = −½·δK_total. No δJ, no δV_xc. The exchange that does NOT
    // cancel in this channel is the SAME range-separated combination the Fock
    // builds, so ω ≠ 0 takes the SR/LR response here too; ω = 0 keeps the
    // untouched single-kernel expression.
    let df: Array2<f64> = match inp.rsh {
        Some(rsh) => -0.5 * &rsh.exchange_response(&dd_ao)?,
        None => -0.5 * inp.k_mix_sr * &triplet_exchange(ctx, inp, k, pool, jk, &dd_ao)?,
    };

    let df_mo = inp.c.t().dot(&df).dot(inp.c);
    let mut h = occ_virt_block(&df_mo, no, n);

    // Diagonal orbital-energy-gap term, identical to the singlet channel: the
    // one-electron part of the Hessian does not know about spin coupling.
    let f_diag: Vec<f64> = (0..n).map(|i| inp.f_mo[(i, i)]).collect();
    for (ir, a) in (no..n).enumerate() {
        for (ic, i) in (0..no).enumerate() {
            h[(ir, ic)] += (f_diag[a] - f_diag[i]) * k[(ir, ic)];
        }
    }

    Ok(h)
}

/// `δK[δD]` for the triplet channel: from the SCF's lent ω = 0 exchange
/// builder when there is one and exact exchange is consumed, else the combined
/// direct build with its `δJ` discarded (see [`triplet_hessian_matvec`]'s
/// Cost section). A lent route with `k_mix_sr == 0` builds nothing.
fn triplet_exchange(
    ctx: &ParallelContext,
    inp: &RhfNewtonInputs,
    k_ov: &Array2<f64>,
    pool: &EnginePool,
    jk: Option<&ScfJkResponse<'_, '_>>,
    dd_ao: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    if let Some(jk) = jk {
        if inp.k_mix_sr == 0.0 {
            return Ok(Array2::<f64>::zeros(dd_ao.raw_dim()));
        }
        let _f64 = crate::jk_ladder::F64BuildScope::enter();
        if let Some(r) = jk.k_response(inp.c, k_ov, inp.nocc, dd_ao) {
            return r;
        }
    }
    DIRECT_RESPONSE_BUILDS.with(|c| c.set(c.get() + 1));
    let n = dd_ao.nrows();
    let mut dj = Array2::<f64>::zeros((n, n));
    let mut dk = Array2::<f64>::zeros((n, n));
    let band_bytes = crate::reduce::resolve_band_bytes(inp.ooc_budget);
    build_jk_with_pool_homogeneous(
        ctx, inp.prep, inp.bounds, inp.thresh, dd_ao, &mut dj, &mut dk, pool, band_bytes,
    )?;
    // `dj` is deliberately dropped: see the Cost section. The triplet response
    // is exchange-only.
    drop(dj);
    Ok(dk)
}

// ---- helpers (mirror uhf_newton.rs) ----

/// The occ→virt block (rows = virt [nocc..n], cols = occ [0..nocc]) of a
/// square MO matrix.
fn occ_virt_block(m: &Array2<f64>, nocc: usize, n: usize) -> Array2<f64> {
    let nv = n - nocc;
    let mut out = Array2::<f64>::zeros((nv, nocc));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            out[(ir, i)] = m[(a, i)];
        }
    }
    out
}

/// AO single-set δD from an occ→virt block: δD^MO symmetric with the block in
/// both (virt,occ) and (occ,virt) corners, then C·δD^MO·Cᵀ.
fn ao_from_ov(c: &Array2<f64>, k_ov: &Array2<f64>, nocc: usize, n: usize) -> Array2<f64> {
    let mut dd_mo = Array2::<f64>::zeros((n, n));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            let v = k_ov[(ir, i)];
            dd_mo[(a, i)] = v;
            dd_mo[(i, a)] = v;
        }
    }
    c.dot(&dd_mo).dot(&c.t())
}

/// Diagonal gap (F[a,a] − F[i,i]) + shift, floored at 1e-6 for near-degeneracy.
fn build_gap(f_diag: &[f64], nocc: usize, n: usize, shift: f64) -> Array2<f64> {
    let nv = n - nocc;
    let mut out = Array2::<f64>::zeros((nv, nocc));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            let d = (f_diag[a] - f_diag[i]) + shift;
            out[(ir, i)] = if d.abs() < 1e-6 { 1e-6 } else { d };
        }
    }
    out
}

/// Apply the Cayley unitary built from the occ→virt block to C: C ← C·U.
fn apply_cayley(
    c: &Array2<f64>,
    k_ov: &Array2<f64>,
    nocc: usize,
    n: usize,
) -> Result<Array2<f64>, FerricError> {
    // Antisymmetric κ in MO basis from the occ→virt block.
    let mut kappa = Array2::<f64>::zeros((n, n));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            let v = k_ov[(ir, i)];
            kappa[(a, i)] = v;
            kappa[(i, a)] = -v;
        }
    }
    let half = 0.5 * &kappa;
    let eye = Array2::<f64>::eye(n);
    let a_mat = &eye - &half;
    let b_mat = &eye + &half;
    let mut u = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let bcol = b_mat.column(col).to_owned();
        let sol = a_mat
            .solve(&bcol)
            .map_err(|e| FerricError::Lapack(format!("RHF Cayley solve: {e}")))?;
        for row in 0..n {
            u[(row, col)] = sol[row];
        }
    }
    Ok(c.dot(&u))
}

fn neg(a: &Array2<f64>) -> Array2<f64> {
    -a
}
fn neg_div(num: &Array2<f64>, den: &Array2<f64>) -> Array2<f64> {
    -num / den
}
fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, &v| m.max(v.abs()))
}
fn inner(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
