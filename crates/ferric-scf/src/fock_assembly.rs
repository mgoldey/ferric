//! Shared exchange-assembly helpers for all six SCF variants.
//!
//! `solve_rhf` (RHF + RKS), `solve_uhf` (UHF + UKS) and `solve_rohf`
//! (ROHF + ROKS) each assemble F = H + J − (exact-exchange mix) with the same
//! two exchange shapes:
//!
//! - **RSH** (ω > 0): K_total(D) = c_SR·K[erfc(ω)](D) + c_LR·K[erf(ω)](D),
//!   contracted per iteration from two geometry-only [`DfK`] fitters that are
//!   built once before the SCF loop.
//! - **HF / plain hybrid**: one K(D) folded in with a scalar exchange fraction
//!   (a single `f.scaled_add(-c, &k)` at the call site — no helper needed).
//!
//! Before this module each solver carried its own verbatim copy of the RSH
//! fitter-pair construction and the per-density RSH fold, so an exchange-
//! assembly fix had to be applied three times in parallel (2026-07-22: the
//! in-place no-clone Fock assembly landed as three identical edits). New
//! exchange-assembly changes go HERE, once, and reach RHF/UHF/ROHF and their
//! KS variants together.
//!
//! Known remaining duplication, deliberately NOT unified here: each solver
//! still hand-rolls its builder-selection preamble (combined `DirectJK` in
//! RHF, per-spin `DfK`/`DirectK` in UHF, `DirectJ`+`DirectK` in ROHF) and its
//! iteration-loop scaffolding (DIIS error, COSMO/PCM hooks, convergence
//! bookkeeping). Those differences are behavioral, not cosmetic, and the loops
//! are convergence-critical — folding them into one spin-generic driver is a
//! separate, carefully-validated refactor, not a drive-by extraction.

use crate::df_j::DfJ;
use crate::df_k::DfK;
use crate::fock::KBuilder;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::Array2;

/// Default JK-fit auxiliary basis when the caller hasn't set one (the same
/// default the solvers use for RI-J/RI-K auto-selection).
pub(crate) const DEFAULT_JK_AUX: &str = "def2-universal-jkfit";

/// Build the (optional) DF-J and DF-K fitters for one non-RSH (ω = 0) SCF
/// setup, sharing a single [`PreparedBasis`] when `j_aux == k_aux`.
///
/// `rhf::solve_rhf` and `uhf::solve_uhf` each independently did
/// `bundled(aux)? -> PreparedBasis::new` once for J and once for K; when the
/// caller set the same aux name for both (the common case — one JK-fit
/// basis serving both J and K), that built two byte-identical
/// `PreparedBasis` instances. Neither [`DfJ`] nor [`DfK`] retains a reference
/// to the aux `PreparedBasis` after construction (both copy out only what
/// they need into their own `ThreeIndexSource`/metric fields — see their
/// struct docs), so building it once here and handing `&dfbs` to both
/// constructors is safe and changes nothing about either builder's state.
///
/// `j_aux`/`k_aux` are `None` exactly when the caller doesn't want that
/// builder (mirrors the existing per-solver `if let Some(aux_name) = ...`
/// gating) — returns `(None, None)` for `(None, None)`, and independently
/// gates each output on its own `Option`, so `(Some(aux), None)` still skips
/// building a DF-K fitter entirely (no wasted `PreparedBasis` when only one
/// side is requested).
pub(crate) fn build_df_jk<'a>(
    ctx: &'a ParallelContext,
    mol: &Molecule,
    op: Operator,
    prep: &PreparedBasis,
    j_aux: Option<&str>,
    k_aux: Option<&str>,
    ooc_budget: usize,
) -> Result<(Option<DfJ<'a>>, Option<DfK<'a>>), FerricError> {
    if j_aux.is_none() && k_aux.is_none() {
        return Ok((None, None));
    }
    // Shared PreparedBasis only when both names are set AND identical —
    // otherwise each requested side builds its own (still only once per
    // side, same as before).
    if let (Some(ja), Some(ka)) = (j_aux, k_aux) {
        if ja == ka {
            let dfbs_set = ferric_core::basis::bundled(ja)?;
            let dfbs = PreparedBasis::new(mol, &dfbs_set)?;
            // Same aux basis on both sides: build the raw (P|μν) tensor ONCE and
            // let both consumers share it. DfJ and DfK were each calling into
            // ThreeIndexSource independently, generating the identical ~95.6M
            // integrals twice (~600 ms per build at benzene/aug-cc-pVTZ).
            //
            // Order matters: DfK dresses from `&mut raw` and keeps only the
            // V^{-1/2}-dressed copy, so raw survives and is then MOVED into DfJ,
            // which does retain it (it applies V^{-1} every iteration).
            //
            // Single-rank ONLY. Under MPI the two sides need different extents —
            // DfJ holds just its own band `ctx.aux_band(naux)`, while DfK needs
            // the FULL aux range because dressing any band sums over all Q — so
            // a shared source would either over-allocate DfJ or under-feed DfK.
            // Multi-rank keeps the independent builds below.
            if ctx.size <= 1 {
                let naux = dfbs.nbasis();
                let mut raw =
                    ferric_integrals::three_index_source::ThreeIndexSource::build_band(
                        op, prep, &dfbs, ooc_budget, 0, naux,
                    )?;
                let df_k = Some(DfK::from_full_raw(
                    &mut raw, prep, &dfbs, op, ooc_budget, Some(ctx),
                )?);
                let df_j = Some(DfJ::from_source(raw, op, &dfbs, ooc_budget, Some(ctx))?);
                return Ok((df_j, df_k));
            }
            let df_j = Some(DfJ::new_banded(op, prep, &dfbs, ooc_budget, Some(ctx))?);
            let df_k = Some(DfK::new_banded(op, prep, &dfbs, ooc_budget, Some(ctx))?);
            return Ok((df_j, df_k));
        }
    }
    let df_j = if let Some(aux_name) = j_aux {
        let dfbs_set = ferric_core::basis::bundled(aux_name)?;
        let dfbs = PreparedBasis::new(mol, &dfbs_set)?;
        Some(DfJ::new_banded(op, prep, &dfbs, ooc_budget, Some(ctx))?)
    } else {
        None
    };
    let df_k = if let Some(aux_name) = k_aux {
        let dfbs_set = ferric_core::basis::bundled(aux_name)?;
        let dfbs = PreparedBasis::new(mol, &dfbs_set)?;
        Some(DfK::new_banded(op, prep, &dfbs, ooc_budget, Some(ctx))?)
    } else {
        None
    };
    Ok((df_j, df_k))
}

/// Build the geometry-only SR/LR [`DfK`] fitter pair for a range-separated
/// hybrid: `(K[erfc(ω)], K[erf(ω)])`. Called once before the SCF loop; only
/// the D-dependent contraction (see [`subtract_rsh_exchange`]) runs per
/// iteration. `df_k_aux = None` falls back to [`DEFAULT_JK_AUX`].
pub(crate) fn build_rsh_dfk_pair<'a>(
    ctx: &'a ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    df_k_aux: Option<&str>,
    omega: f64,
    ooc_budget: usize,
) -> Result<(DfK<'a>, DfK<'a>), FerricError> {
    let aux_name = df_k_aux.unwrap_or(DEFAULT_JK_AUX);
    let dfbs_set = ferric_core::basis::bundled(aux_name)?;
    let dfbs_prep = PreparedBasis::new(mol, &dfbs_set)?;
    Ok((
        DfK::new_banded(Operator::erfc(omega), prep, &dfbs_prep, ooc_budget, Some(ctx))?,
        DfK::new_banded(Operator::erf(omega), prep, &dfbs_prep, ooc_budget, Some(ctx))?,
    ))
}

/// Contract the RSH exchange for one density and fold it into F in place:
/// `F −= scale · (c_sr·K_SR(D) + c_lr·K_LR(D))`.
///
/// `scale` is 0.5 for the closed-shell restricted convention (F = H + J −
/// ½·K_total over the total density) and 1.0 for per-spin Focks (UHF/ROHF).
/// `f += (−scale)·K_total` is bit-identical to the former per-solver
/// `f −= &k_total` / scaled-assign paths: scaling by −1 is an exact sign flip
/// and by 0.5 an exact exponent shift, so no rounding is introduced. Only two
/// K-sized scratch matrices are live (allocated here, dropped on return) —
/// never a retained k_total clone.
///
/// Validate `config.k_builder` and say which pluggable builder (if any) the
/// caller should construct.
///
/// Shared by `solve_rhf`, `solve_uhf` and `solve_rohf` so the whitelist, the
/// error text and the DF-J/DF-K precedence rule live in ONE place. Before this
/// existed only `solve_rhf` read the field at all, so `k_builder = "link"` /
/// `"cosx"` was silently ignored for every open-shell run.
///
/// * `Err` — the value is not one of `direct` / `link` / `cosx`.
/// * `Ok(None)` — no pluggable builder: the field is unset, is `"direct"`, or
///   is overridden by an active density-fitted J/K path (warned about, below).
/// * `Ok(Some(kind))` — construct that builder.
///
/// `df_active` is `df_j.is_some() || df_k.is_some()` AFTER the solver's own
/// auto-defaulting (a hybrid/RSH functional silently turns DF-K on). Exchange
/// then comes from DF-K or the direct 4-centre builder and a pluggable K would
/// be built and thrown away, so it is skipped WITH A WARNING rather than
/// silently no-op'ing. `df_k_present` only picks the wording.
pub(crate) fn resolve_k_builder(
    k_builder: Option<&str>,
    df_active: bool,
    df_k_present: bool,
) -> Result<Option<&str>, FerricError> {
    let Some(kb) = k_builder else {
        return Ok(None);
    };
    if kb != "direct" && kb != "link" && kb != "cosx" {
        return Err(FerricError::General(format!(
            "unknown k_builder '{kb}': valid options are 'direct', 'link', 'cosx'"
        )));
    }
    let pluggable = matches!(kb, "link" | "cosx").then_some(kb);
    if df_active {
        if let Some(kind) = pluggable {
            eprintln!(
                "[ferric] warning: k_builder = \"{kind}\" is IGNORED because density-fitted J/K is active \
                 (df_j_aux/df_k_aux set, or auto-defaulted for a functional); exchange comes from {}",
                if df_k_present { "DF-K" } else { "the direct 4-centre builder" }
            );
        }
        return Ok(None);
    }
    Ok(pluggable)
}

/// Construct the pluggable exchange builder named by [`resolve_k_builder`].
///
/// Returns `Ok(None)` for `kind == None` so a caller can pass the resolver's
/// output straight through. The returned builder is built ONCE and reused for
/// every iteration (and, in the open-shell solvers, for BOTH spins):
///
/// * **LinK** — its `SignificantPairs` list is geometry-only, but its
///   `DensityPairs` list is DENSITY-DEPENDENT, so the caller MUST call
///   [`KBuilder::update_density`] with the density it is about to contract
///   before each `build`. For two spins that means a per-spin
///   `update_density(D_σ)` + `build(D_σ)` pair: the α and β densities have
///   different sparsity, and a list built from `D_α` (or from `D_α + D_β`)
///   can drop pairs that are significant for `D_β`. Building the list from the
///   total density would be a screening APPROXIMATION whose error is invisible
///   in the energy until it is not; per-spin lists are exact to the same
///   threshold the closed-shell path holds, at the cost of one extra (cheap)
///   list build per iteration. Correctness first — see
///   `tests/k_builder_open_shell.rs`.
/// * **COSX** — has NO density-dependent state (`update_density` is a no-op)
///   and its `S_num` overlap-fit factor is geometry-only, so ONE instance
///   serves both spins with no leaked state between builds. This is anchored
///   directly (`cosx_k_alpha_beta_independent_from_one_instance`).
///
/// The `link_bound` reference is the caller's own `SchwarzBounds`, kept alive
/// by the caller for the builder's lifetime.
pub(crate) fn build_pluggable_k<'a, B: crate::screening::Bound + Sync>(
    kind: Option<&str>,
    ctx: &'a ParallelContext,
    mol: &'a Molecule,
    prep: &'a PreparedBasis,
    link_bound: &'a B,
    op: Operator,
    cosx_cfg: &crate::cosx_k::CosxConfig,
    integral_thresh: f64,
    ooc_budget: usize,
) -> Result<Option<Box<dyn KBuilder + 'a>>, FerricError> {
    match kind {
        Some("link") => Ok(Some(Box::new(crate::link_k::LinkK::new(
            ctx, prep, link_bound, op, integral_thresh, ooc_budget,
        )) as Box<dyn KBuilder + 'a>)),
        Some("cosx") => {
            // Seminumerical exchange is defined against the 1/r kernel only;
            // an attenuated (erf/erfc) or geminal operator has no COSX form
            // here. Refused rather than silently computing Coulomb exchange.
            if op != Operator::coulomb() {
                return Err(FerricError::General(
                    "k_builder = \"cosx\" supports the Coulomb operator only".into(),
                ));
            }
            let ck = crate::cosx_k::CosxK::new(ctx, mol, prep, cosx_cfg.clone(), ooc_budget)?;
            Ok(Some(Box::new(ck) as Box<dyn KBuilder + 'a>))
        }
        _ => Ok(None),
    }
}

/// Narrow the resolved pluggable-K choice to the exchange shapes it can
/// actually serve, warning when it cannot.
///
/// A pluggable builder supplies ONE plain Coulomb-kernel K(D). Two exchange
/// shapes cannot consume that:
///
/// * **pure DFT** (`need_k == false`, k_mix all zero) — no exact exchange at
///   all, so any K built would be multiplied by zero and discarded;
/// * **RSH** (`omega > 0`) — exchange is `c_SR·K[erfc(ω)] + c_LR·K[erf(ω)]`,
///   contracted from the dedicated SR/LR `DfK` fitter pair. Note the open-shell
///   solvers reach this with `df_j`/`df_k` BOTH `None` (their `k_aux_eff` gate
///   deliberately excludes ω > 0), so [`resolve_k_builder`]'s `df_active`
///   warning does not fire for RSH the way it does in `solve_rhf`, where an RSH
///   functional auto-defaults `df_k_aux_eff` and trips it. Without this second
///   warning an RSH open-shell run would skip the builder SILENTLY — the exact
///   class of no-op this wiring exists to remove.
pub(crate) fn narrow_k_builder_to_supported(
    kind: Option<&str>,
    need_k: bool,
    omega: f64,
) -> Option<&str> {
    let kind = kind?;
    let reason = if !need_k {
        "the functional uses no exact exchange"
    } else if omega > 0.0 {
        "exchange for a range-separated functional comes from the SR/LR density-fitted fitters"
    } else {
        return Some(kind);
    };
    eprintln!("[ferric] warning: k_builder = \"{kind}\" is IGNORED because {reason}");
    None
}

/// Build K_α and K_β from ONE pluggable builder instance, refreshing any
/// density-dependent state per spin.
///
/// The `update_density(D_σ)` immediately before `build(D_σ)` is what makes a
/// shared LinK instance correct across two spins (see [`build_pluggable_k`]);
/// it is a no-op for COSX. Returns the summed quartet count.
pub(crate) fn build_open_shell_pluggable_k(
    kb: &mut dyn KBuilder,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    k_a: &mut Array2<f64>,
    k_b: &mut Array2<f64>,
) -> Result<usize, FerricError> {
    kb.update_density(d_a);
    let mut q = kb.build(d_a, k_a)?;
    kb.update_density(d_b);
    q += kb.build(d_b, k_b)?;
    Ok(q)
}

/// `c_occ`: when `Some`, the BARE occupied MO coefficients with
/// `D = occ_factor · c_occ·c_occᵀ`; both SR and LR exchange are then contracted
/// via the O(naux·n²·nocc) DF-K half-transform ([`KBuilder::build_from_occ`])
/// instead of the O(naux·n³) density path, and the linear factor `occ_factor` is
/// folded into the Fock scale (`build_from_occ` returns K for `c_occ·c_occᵀ`;
/// K is linear in D so K(D) = occ_factor·K(c_occ·c_occᵀ)). `occ_factor` is 2.0
/// for the closed-shell restricted density (`D = 2·C_occ·C_occᵀ`) and 1.0 for a
/// per-spin UHF/ROHF density (`D_σ = C_occ,σ·C_occ,σᵀ`). Applying the factor to
/// the Fock scale (an exact power-of-2) rather than pre-scaling C avoids a √2
/// rounding on the coefficients. `None` (guess iteration / smearing, where `D`
/// is not a plain `C·Cᵀ`) uses the density-based [`KBuilder::build`] with the
/// factor already baked into `d`. Both routes contract the same fitted B and
/// agree to the DF-K reassociation floor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn subtract_rsh_exchange(
    dfk_sr: &mut DfK,
    dfk_lr: &mut DfK,
    d: &Array2<f64>,
    c_occ: Option<&Array2<f64>>,
    occ_factor: f64,
    f: &mut Array2<f64>,
    c_sr: f64,
    c_lr: f64,
    scale: f64,
) -> Result<(), FerricError> {
    let n = f.nrows();
    let mut k_sr = Array2::<f64>::zeros((n, n));
    let mut k_lr = Array2::<f64>::zeros((n, n));
    // Effective Fock scale: the density path bakes the occupation factor into
    // `d`, so it uses `scale` directly; the occ path builds K for c_occ·c_occᵀ
    // and folds `occ_factor` in here (scale·occ_factor is exact for the 0.5 / 1.0
    // / 2.0 powers of two used by RHF/UHF/ROHF).
    let eff_scale = if let Some(c) = c_occ {
        dfk_sr.build_from_occ(c, &mut k_sr)?;
        dfk_lr.build_from_occ(c, &mut k_lr)?;
        scale * occ_factor
    } else {
        dfk_sr.build(d, &mut k_sr)?;
        dfk_lr.build(d, &mut k_lr)?;
        scale
    };
    // In-place: k_sr *= c_sr; k_sr += c_lr·k_lr; f −= eff_scale·k_sr. Same
    // element-wise association as `c_sr*&k_sr + c_lr*&k_lr` followed by
    // `f.scaled_add(-eff_scale, &k_total)` — no `k_total` temp.
    k_sr *= c_sr;
    k_sr.scaled_add(c_lr, &k_lr);
    f.scaled_add(-eff_scale, &k_sr);
    Ok(())
}
