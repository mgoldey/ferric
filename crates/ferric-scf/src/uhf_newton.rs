//! Second-order UHF / UKS step: damped Newton via Hessian-vector products.
//!
//! The unrestricted analogue of `rohf_newton.rs`. Where ROHF rotates a single
//! set of MOs partitioned into closed/open/virtual (Roothaan coupling), UHF has
//! two INDEPENDENT MO sets — α and β — each with its own occupied↔virtual
//! rotation κ^σ. The two spins are NOT decoupled, however: the Coulomb response
//! δJ depends on the total δD = δD_α + δD_β, and (for KS) the XC kernel carries
//! the f_αβ cross term. So we solve the coupled linear system
//!
//!   H · (κ_α, κ_β) = −(g_α, g_β)
//!
//! for both spin blocks simultaneously with one preconditioned-CG loop.
//!
//! Per spin the gradient is the occupied→virtual block of the MO-basis Fock,
//!   g^σ_{ai} = F^σ_{ai}   (a ∈ virt_σ, i ∈ occ_σ),
//! which is exactly the UHF Brillouin condition (F^σ_{ai} = 0 at convergence).
//!
//! The Hessian matvec builds the AO density perturbation δD^σ from κ^σ, rebuilds
//! J (on δD_total) and K^σ (on δD^σ), adds the optional XC-kernel response
//! (δV_xc^α, δV_xc^β) — the SAME `LdaFxcKernel`/`GgaFxcKernel` closure the ROKS
//! path uses — projects the resulting δF^σ to MO basis, reads the occ→virt
//! block, and adds the diagonal orbital-energy-gap term.
//!
//! Rotation is applied via the Cayley unitary U = (I − κ/2)^{−1}(I + κ/2) per
//! spin, which exactly preserves orthonormality. Sign convention matches
//! `rohf_newton.rs` (κ = −g/(gap+μ); C ← C·U).

use crate::engine_pool::EnginePool;
use crate::rhf::build_jk_with_pool_homogeneous;
use crate::rohf_newton::FxcResponse;
use crate::screening::SchwarzBounds;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;
use ndarray_linalg::Solve;
use std::cell::Cell;

thread_local! {
    /// UHF/UKS Hessian-vector products computed on this thread (test seam).
    static HVP_CALLS: Cell<usize> = const { Cell::new(0) };
    /// Direct four-centre response J/K builds made by the UHF and ROHF products
    /// (test seam).
    static DIRECT_RESPONSE_BUILDS: Cell<usize> = const { Cell::new(0) };
}

/// Number of [`hessian_matvec`] calls made on the calling thread so far (test
/// seam; the PCG loop is sequential on its caller's thread, so a per-thread
/// count is race-free under the parallel test harness).
#[doc(hidden)]
pub fn hvp_calls_on_this_thread() -> usize {
    HVP_CALLS.with(Cell::get)
}

/// Number of direct four-centre J/K builds the open-shell Hessian-vector
/// products ([`hessian_matvec`] and `rohf_newton::hessian_matvec`) made on the
/// calling thread so far (test seam). Zero across a Newton step means every
/// response term came from the SCF's fitted builders or was not needed.
#[doc(hidden)]
pub fn direct_response_builds_on_this_thread() -> usize {
    DIRECT_RESPONSE_BUILDS.with(Cell::get)
}

/// An AO response linear in its density argument, `δD ↦ δJ[δD]`, built by the
/// SAME (density-fitted) builder the Fock was assembled from.
pub type AoResponse<'a> = dyn Fn(&Array2<f64>) -> Result<Array2<f64>, FerricError> + 'a;

/// An exchange build from an occupied-type factor: `C_f ↦ K[C_f·C_fᵀ]` (the
/// [`crate::fock::KBuilder::build_from_occ`] contract, weight 1 per column),
/// by the SAME density-fitted builder the Fock was assembled from. The matvec
/// forms each spin's `δK[δD_σ]` from two such builds
/// (`rhf_newton::occ_pair_response_k`), never from the `n × n` δD.
pub type OccKResponse<'a> = dyn Fn(&Array2<f64>) -> Result<Array2<f64>, FerricError> + 'a;

/// The exchange response a second-order product takes from the SCF's own ω = 0
/// exchange builder.
#[derive(Clone, Copy)]
pub enum KResponse<'a> {
    /// The SCF's DF-K through its occupied-factor path ([`OccKResponse`]):
    /// each spin's `δK[δD_σ]` from two `nocc`-column builds.
    Occ(&'a OccKResponse<'a>),
    /// The SCF's unfitted exchange builder (`DirectK` on DF-J + direct K;
    /// any linear builder works) on the density: `δD_σ ↦ K[δD_σ]`, one build per spin,
    /// made homogeneous in `δD_σ` by `rhf_newton::unfitted_response`.
    Density(&'a AoResponse<'a>),
}

impl std::fmt::Debug for KResponse<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            KResponse::Occ(_) => "KResponse::Occ",
            KResponse::Density(_) => "KResponse::Density",
        })
    }
}

/// The SCF's own J/K builders as the response hooks of
/// [`UhfNewtonInputs::j_response`] / [`UhfNewtonInputs::k_response`] (and the
/// ROHF ones), so the open-shell orbital Hessian differentiates the Fock the
/// SCF builds:
///
/// * `δJ`: the fitted RI-J when lent, else the unfitted J builder (`DirectJ`,
///   on DF-K + conventional J), else `None` (the matvec's direct build).
/// * `δK`: the fitted DF-K (occupied-factor path) when lent, else the unfitted
///   ω = 0 exchange builder (COSX or `DirectK`, density path), else `None`.
///
/// Unfitted builders run through `rhf_newton::unfitted_response`, which makes
/// their absolute screens homogeneous in the response density. Never lend LinK:
/// its density-pair list is not linear in an indefinite `δD`.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub struct ScfResponses<'b> {
    j: Option<Box<AoResponse<'b>>>,
    k_occ: Option<Box<OccKResponse<'b>>>,
    k_density: Option<Box<AoResponse<'b>>>,
}

impl<'b> ScfResponses<'b> {
    /// Wrap the lent builders; see the type doc for which one answers which
    /// term. A fitted builder takes precedence over an unfitted one.
    pub fn lend<'e: 'b>(
        fitted_j: Option<&'b mut (dyn crate::fock::JBuilder + 'e)>,
        fitted_k: Option<&'b mut crate::df_k::DfK<'e>>,
        unfitted_j: Option<&'b mut (dyn crate::fock::JBuilder + 'e)>,
        unfitted_k: Option<&'b mut (dyn crate::fock::KBuilder + 'e)>,
    ) -> Self {
        use crate::fock::{JBuilder, KBuilder};
        use crate::rhf_newton::unfitted_response;
        use std::cell::RefCell;
        let j = match (fitted_j, unfitted_j) {
            (Some(b), _) => {
                let cell = RefCell::new(b);
                Some(Box::new(move |dd: &Array2<f64>| {
                    let mut out = Array2::<f64>::zeros(dd.raw_dim());
                    JBuilder::build(&mut **cell.borrow_mut(), dd, &mut out)?;
                    Ok(out)
                }) as Box<AoResponse<'b>>)
            }
            (None, Some(b)) => {
                let cell = RefCell::new(b);
                Some(Box::new(move |dd: &Array2<f64>| {
                    unfitted_response(dd, |d, out| {
                        JBuilder::build(&mut **cell.borrow_mut(), d, out)
                    })
                }) as Box<AoResponse<'b>>)
            }
            (None, None) => None,
        };
        let (k_occ, k_density) = match (fitted_k, unfitted_k) {
            (Some(b), _) => {
                let cell = RefCell::new(b);
                let k = Box::new(move |c_f: &Array2<f64>| {
                    let n = c_f.nrows();
                    let mut out = Array2::<f64>::zeros((n, n));
                    KBuilder::build_from_occ(&mut **cell.borrow_mut(), c_f, &mut out)?;
                    Ok(out)
                }) as Box<OccKResponse<'b>>;
                (Some(k), None)
            }
            (None, Some(b)) => {
                let cell = RefCell::new(b);
                let k = Box::new(move |dd: &Array2<f64>| {
                    unfitted_response(dd, |d, out| {
                        let mut kb = cell.borrow_mut();
                        kb.update_density(d);
                        kb.build(d, out)
                    })
                }) as Box<AoResponse<'b>>;
                (None, Some(k))
            }
            (None, None) => (None, None),
        };
        Self {
            j,
            k_occ,
            k_density,
        }
    }

    /// The `j_response` hook.
    pub fn j(&self) -> Option<&AoResponse<'b>> {
        self.j.as_deref()
    }

    /// The `k_response` hook.
    pub fn k(&self) -> Option<KResponse<'_>> {
        match (self.k_occ.as_deref(), self.k_density.as_deref()) {
            (Some(k), _) => Some(KResponse::Occ(k)),
            (None, Some(k)) => Some(KResponse::Density(k)),
            (None, None) => None,
        }
    }
}

impl std::fmt::Debug for ScfResponses<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScfResponses")
            .field("j", &self.j.is_some())
            .field("k_occ", &self.k_occ.is_some())
            .field("k_density", &self.k_density.is_some())
            .finish()
    }
}

/// Inputs to one UHF/UKS Newton step.
pub struct UhfNewtonInputs<'a> {
    /// The AO basis on the molecule.
    pub prep: &'a PreparedBasis,
    /// Schwarz bounds for the response J/K builds.
    pub bounds: &'a SchwarzBounds,
    /// α MO coeffs (n × n), columns ordered occ | virt.
    pub c_a: &'a Array2<f64>,
    /// β MO coeffs (n × n), columns ordered occ | virt.
    pub c_b: &'a Array2<f64>,
    /// α Fock in the α-MO basis (current iter).
    pub f_a_mo: &'a Array2<f64>,
    /// β Fock in the β-MO basis (current iter).
    pub f_b_mo: &'a Array2<f64>,
    /// Number of occupied α MOs.
    pub nocc_a: usize,
    /// Number of occupied β MOs.
    pub nocc_b: usize,
    /// K mixing coefficient for the plain-Coulomb exchange response
    /// (1.0 for HF, c_HF for a global hybrid). Read ONLY when `rsh` is `None`.
    pub k_mix_sr: f64,
    /// Range-separated exchange response (ω ≠ 0 only). `Some(..)` replaces the
    /// single Coulomb-kernel `k_mix_sr · δK` with the SR/LR combination
    /// `c_SR·δK[erfc(ω)] + c_LR·δK[erf(ω)]` built from the SAME `DfK` fitters
    /// the converged Fock was assembled from — see [`crate::rsh_response`] for
    /// why the fitters are shared rather than rebuilt. `None` (ω = 0) leaves the
    /// Coulomb path below bit-identical to the pre-#314 code.
    pub rsh: Option<&'a crate::rsh_response::RshResponse<'a>>,
    /// Optional XC-kernel response closure (None for pure UHF).
    pub fxc: Option<&'a FxcResponse<'a>>,
    /// Integral screening threshold for the response J/K builds (the solvers
    /// pass `RhfConfig::integral_thresh`).
    pub thresh: f64,
    /// Solver-resolved memory budget (see `rhf::resolve_three_index_budget`),
    /// used to size the `build_jk_with_pool` reduction band via
    /// `reduce::resolve_band_bytes` — never affects results, only the
    /// live-set/parallel width of the deterministic reduction.
    pub ooc_budget: usize,
    /// Coulomb response of the Fock's own DF-J (`δJ[δD_total]`). `None`: the
    /// exact four-centre δJ, which is what a direct-J Fock differentiates.
    /// Under DF-J the exact δJ is a DIFFERENT kernel from the one the energy
    /// uses: MEASURED on OH ²Π UKS-LDA/STO-3G with def2-universal-jkfit, the
    /// mismatch puts ~2e-5 Ha/rad² of curvature on the hole-orientation mode
    /// whose true curvature is ~1e-6, so every TRAH step there realised twice
    /// its prediction (ρ = 2.00) and the run crept to `max_iter`.
    pub j_response: Option<&'a AoResponse<'a>>,
    /// Exchange response of the Fock's own ω = 0 exchange builder
    /// ([`KResponse`]): the DF-K occupied-factor build, or the unfitted COSX /
    /// `DirectK` density build of a mixed route. The matvec forms `δK[δD_σ]`
    /// per spin from it and scales by `k_mix_sr` as it does the exact one.
    /// `None`: exact δK. Ignored on the RSH path (`rsh`), which already uses
    /// the Fock's fitters, and when `k_mix_sr == 0` (no exact exchange is
    /// consumed, so none is built by either path).
    pub k_response: Option<KResponse<'a>>,
}

/// One damped-Newton step on UHF/UKS MO coefficients.
///
/// Returns `(C_α_new, C_β_new, kmax)` where `kmax` is the ∞-norm of the packed
/// rotation (a step-size diagnostic). `level_shift` is added to the diagonal
/// preconditioner gap; `max_step` is the componentwise trust radius on κ.
#[allow(clippy::too_many_arguments)]
pub fn uhf_newton_step(
    ctx: &ParallelContext,
    inp: &UhfNewtonInputs,
    level_shift: f64,
    max_step: f64,
    cg_max_iter: usize,
    cg_conv: f64,
) -> Result<(Array2<f64>, Array2<f64>, f64), FerricError> {
    let n = inp.c_a.nrows();
    let na = inp.nocc_a;
    let nb = inp.nocc_b;

    // EnginePool is geometry/basis-only (density-independent) — build ONCE
    // here and reuse across every hessian_matvec call in the PCG loop below
    // (3 build_jk calls per matvec), instead of each call constructing its
    // own pool. Reduction order is unchanged, so results stay bit-identical
    // across thread counts.
    let pool = EnginePool::new(
        inp.bounds.op,
        inp.prep,
        ferric_integrals::engine_pool::eri_precision(),
    )?;

    // Gradient blocks g^σ_{ai} = F^σ_{ai}  (rows = virt, cols = occ).
    let g_a = occ_virt_block(inp.f_a_mo, na, n);
    let g_b = occ_virt_block(inp.f_b_mo, nb, n);

    // Diagonal preconditioner gaps: (F[a,a] − F[i,i]) + shift, per spin.
    let f_a_diag: Vec<f64> = (0..n).map(|i| inp.f_a_mo[(i, i)]).collect();
    let f_b_diag: Vec<f64> = (0..n).map(|i| inp.f_b_mo[(i, i)]).collect();
    let diag_a = build_gap(&f_a_diag, na, n, level_shift);
    let diag_b = build_gap(&f_b_diag, nb, n, level_shift);

    // Initial guess: diagonal solve κ⁰ = −g / gap.
    let mut k_a = neg_div(&g_a, &diag_a);
    let mut k_b = neg_div(&g_b, &diag_b);

    // Gradient already below tolerance: the step is κ⁰ itself, and the CG
    // residual (whose first term is an HVP) would never be read, so skip it
    // (as `rhf_newton_step` does; wiki/notes/scf-second-order-cost-2026-11).
    let gmax = max_abs(&g_a).max(max_abs(&g_b));
    if gmax >= cg_conv {
        let diag = (&diag_a, &diag_b);
        pcg_refine(
            ctx,
            inp,
            &pool,
            (&g_a, &g_b),
            diag,
            (&mut k_a, &mut k_b),
            cg_max_iter,
            cg_conv,
        )?;
    }

    // Trust-radius clip (shared scale so the α/β step stays consistent).
    let kmax = max_abs(&k_a).max(max_abs(&k_b));
    if kmax > max_step {
        let s = max_step / kmax;
        k_a.mapv_inplace(|x| x * s);
        k_b.mapv_inplace(|x| x * s);
    }

    let c_a_new = apply_cayley(inp.c_a, &k_a, na, n)?;
    let c_b_new = apply_cayley(inp.c_b, &k_b, nb, n)?;
    Ok((c_a_new, c_b_new, kmax))
}

/// Preconditioned conjugate-gradient refinement of `(κ_α, κ_β)` (on entry the
/// diagonal guess κ⁰) towards the coupled system H·κ = −g.
#[allow(clippy::too_many_arguments)]
fn pcg_refine(
    ctx: &ParallelContext,
    inp: &UhfNewtonInputs,
    pool: &EnginePool,
    (g_a, g_b): (&Array2<f64>, &Array2<f64>),
    (diag_a, diag_b): (&Array2<f64>, &Array2<f64>),
    (k_a, k_b): (&mut Array2<f64>, &mut Array2<f64>),
    cg_max_iter: usize,
    cg_conv: f64,
) -> Result<(), FerricError> {
    let (h0_a, h0_b) = hessian_matvec(ctx, inp, k_a, k_b, pool)?;
    let mut r_a = &neg(g_a) - &h0_a;
    let mut r_b = &neg(g_b) - &h0_b;
    let mut z_a = &r_a / diag_a;
    let mut z_b = &r_b / diag_b;
    let mut p_a = z_a.clone();
    let mut p_b = z_b.clone();
    let mut rz_old = inner(&r_a, &z_a) + inner(&r_b, &z_b);

    for _it in 0..cg_max_iter {
        let max_resid = max_abs(&r_a).max(max_abs(&r_b));
        if max_resid < cg_conv {
            break;
        }
        let (ap_a, ap_b) = hessian_matvec(ctx, inp, &p_a, &p_b, pool)?;
        let p_ap = inner(&p_a, &ap_a) + inner(&p_b, &ap_b);
        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rz_old / p_ap;
        k_a.zip_mut_with(&p_a, |k, &v| *k += alpha * v);
        k_b.zip_mut_with(&p_b, |k, &v| *k += alpha * v);
        r_a.zip_mut_with(&ap_a, |r, &v| *r -= alpha * v);
        r_b.zip_mut_with(&ap_b, |r, &v| *r -= alpha * v);
        z_a = &r_a / diag_a;
        z_b = &r_b / diag_b;
        let rz_new = inner(&r_a, &z_a) + inner(&r_b, &z_b);
        let beta = rz_new / rz_old;
        rz_old = rz_new;
        p_a.zip_mut_with(&z_a, |p, &v| *p = v + beta * *p);
        p_b.zip_mut_with(&z_b, |p, &v| *p = v + beta * *p);
    }
    Ok(())
}

/// H · (κ_α, κ_β) for the two occ→virt blocks.
///
/// Public so tests can finite-difference-validate the matvec directly against
/// the orbital-gradient derivative. `k_a`/`k_b` are the occ→virt (virt×occ)
/// rotation blocks; returns the corresponding Hessian-vector-product blocks.
pub fn hessian_matvec(
    ctx: &ParallelContext,
    inp: &UhfNewtonInputs,
    k_a: &Array2<f64>,
    k_b: &Array2<f64>,
    pool: &EnginePool,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    HVP_CALLS.with(|c| c.set(c.get() + 1));
    let n = inp.c_a.nrows();
    let na = inp.nocc_a;
    let nb = inp.nocc_b;

    // δD^σ in AO from κ^σ. In σ-MO basis, δD^σ[a,i] = δD^σ[i,a] = κ^σ[a,i]
    // (occ→virt block only; occ=1, virt=0 ⇒ (occ[i]−occ[a])·κ = +κ). Transform
    // to AO via C_σ · δD_σ^MO · C_σ^T.
    let dd_a_ao = ao_from_ov(inp.c_a, k_a, na, n);
    let dd_b_ao = ao_from_ov(inp.c_b, k_b, nb, n);

    let (mut df_a, mut df_b) = response_fock(ctx, inp, pool, (k_a, k_b), (&dd_a_ao, &dd_b_ao))?;

    if let Some(fxc) = inp.fxc {
        let (dvxc_a, dvxc_b) = fxc(&dd_a_ao, &dd_b_ao);
        df_a = &df_a + &dvxc_a;
        df_b = &df_b + &dvxc_b;
    }

    // Project to each spin's own MO basis and read the occ→virt block.
    let df_a_mo = inp.c_a.t().dot(&df_a).dot(inp.c_a);
    let df_b_mo = inp.c_b.t().dot(&df_b).dot(inp.c_b);
    let mut h_a = occ_virt_block(&df_a_mo, na, n);
    let mut h_b = occ_virt_block(&df_b_mo, nb, n);

    // Diagonal orbital-energy-gap term: + (F[a,a] − F[i,i]) · κ[a,i].
    let f_a_diag: Vec<f64> = (0..n).map(|i| inp.f_a_mo[(i, i)]).collect();
    let f_b_diag: Vec<f64> = (0..n).map(|i| inp.f_b_mo[(i, i)]).collect();
    for (ir, a) in (na..n).enumerate() {
        for (ic, i) in (0..na).enumerate() {
            h_a[(ir, ic)] += (f_a_diag[a] - f_a_diag[i]) * k_a[(ir, ic)];
        }
    }
    for (ir, a) in (nb..n).enumerate() {
        for (ic, i) in (0..nb).enumerate() {
            h_b[(ir, ic)] += (f_b_diag[a] - f_b_diag[i]) * k_b[(ir, ic)];
        }
    }

    Ok((h_a, h_b))
}

/// `(δF_α, δF_β)` without the XC kernel: `δJ[δD_α + δD_β] − δK_σ` per spin.
///
/// * `δJ`: the Fock's own J builder (`inp.j_response`) when lent, else direct.
/// * `δK_σ`: the RSH SR/LR pair when `inp.rsh` is set; otherwise only when the
///   Fock consumes ω = 0 exact exchange (`k_mix_sr != 0`), from the Fock's own
///   DF-K through the occupied-factor path when lent
///   (`½·occ_pair_response_k`, since `δD_σ = Y·Xᵀ + X·Yᵀ`), or from its
///   unfitted builder on `δD_σ` ([`KResponse::Density`]), else direct. A pure
///   functional builds no K: the old code built two direct K's and scaled them
///   by 0, so dropping them leaves `δF` bit-identical.
fn response_fock(
    ctx: &ParallelContext,
    inp: &UhfNewtonInputs,
    pool: &EnginePool,
    (k_a, k_b): (&Array2<f64>, &Array2<f64>),
    (dd_a_ao, dd_b_ao): (&Array2<f64>, &Array2<f64>),
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    // Every fitted build below (RI-J, the ± occupied-factor DF-K, the RSH
    // pair) runs OUTSIDE the SCF's J/K ladder scope; without this guard a
    // mixed-precision device kernel would treat them as mixed, and the ±
    // occupied-factor δK would carry an f32 error independent of |κ| (as in
    // `rhf_newton::response_jk`). Direct builds do not consult the scope.
    let _f64 = crate::jk_ladder::F64BuildScope::enter();
    let dd_tot = dd_a_ao + dd_b_ao;
    let dj = match inp.j_response {
        Some(jr) => jr(&dd_tot)?,
        None => {
            direct_jk(
                ctx,
                inp.prep,
                inp.bounds,
                inp.thresh,
                inp.ooc_budget,
                pool,
                &dd_tot,
            )?
            .0
        }
    };

    // Per-spin exchange response. The RSH branch is the ONLY branch that
    // touches ω: it builds c_SR·δK[erfc(ω)] + c_LR·δK[erf(ω)] from the Fock's
    // own DF-K fitters, matching F_σ = H + J − (c_SR·K_SR + c_LR·K_LR)_σ (the
    // `occ_factor`/`scale` are both 1.0 for a per-spin Fock, see
    // `fock_assembly::subtract_rsh_exchange`'s call in `solve_uhf`).
    if let Some(rsh) = inp.rsh {
        let dk_a = rsh.exchange_response(dd_a_ao)?;
        let dk_b = rsh.exchange_response(dd_b_ao)?;
        return Ok((&dj - &dk_a, &dj - &dk_b));
    }
    let c_k = inp.k_mix_sr;
    if c_k == 0.0 {
        return Ok((dj.clone(), dj));
    }
    let (dk_a, dk_b) = match inp.k_response {
        Some(KResponse::Density(kr)) => (kr(dd_a_ao)?, kr(dd_b_ao)?),
        Some(KResponse::Occ(kr)) => {
            let mut k_occ = |c: &Array2<f64>, out: &mut Array2<f64>| -> Result<(), FerricError> {
                *out = kr(c)?;
                Ok(())
            };
            let mut spin = |c: &Array2<f64>, k_ov: &Array2<f64>, nocc: usize| {
                crate::rhf_newton::occ_pair_response_k(&mut k_occ, c, k_ov, nocc).map(|k| 0.5 * k)
            };
            (
                spin(inp.c_a, k_a, inp.nocc_a)?,
                spin(inp.c_b, k_b, inp.nocc_b)?,
            )
        }
        None => {
            let direct_k = |dd: &Array2<f64>| {
                direct_jk(
                    ctx,
                    inp.prep,
                    inp.bounds,
                    inp.thresh,
                    inp.ooc_budget,
                    pool,
                    dd,
                )
                .map(|(_, k)| k)
            };
            (direct_k(dd_a_ao)?, direct_k(dd_b_ao)?)
        }
    };
    Ok((&dj - &(c_k * &dk_a), &dj - &(c_k * &dk_b)))
}

/// One direct four-centre `(J, K)` build of `dd` (counted by
/// [`direct_response_builds_on_this_thread`]). Shared with `rohf_newton`.
pub(crate) fn direct_jk(
    ctx: &ParallelContext,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    thresh: f64,
    ooc_budget: usize,
    pool: &EnginePool,
    dd: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    DIRECT_RESPONSE_BUILDS.with(|c| c.set(c.get() + 1));
    let n = dd.nrows();
    let mut j = Array2::<f64>::zeros((n, n));
    let mut k = Array2::<f64>::zeros((n, n));
    let band_bytes = crate::reduce::resolve_band_bytes(ooc_budget);
    build_jk_with_pool_homogeneous(
        ctx, prep, bounds, thresh, dd, &mut j, &mut k, pool, band_bytes,
    )?;
    Ok((j, k))
}

// ---- helpers ----

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

/// AO δD from an occ→virt block: δD^MO symmetric with the block in both
/// (virt,occ) and (occ,virt) corners, then C·δD^MO·Cᵀ.
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
            .map_err(|e| FerricError::Lapack(format!("UHF Cayley solve: {e}")))?;
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
