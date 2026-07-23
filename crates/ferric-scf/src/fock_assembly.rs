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
pub(crate) fn subtract_rsh_exchange(
    dfk_sr: &mut DfK,
    dfk_lr: &mut DfK,
    d: &Array2<f64>,
    f: &mut Array2<f64>,
    c_sr: f64,
    c_lr: f64,
    scale: f64,
) -> Result<(), FerricError> {
    let n = f.nrows();
    let mut k_sr = Array2::<f64>::zeros((n, n));
    dfk_sr.build(d, &mut k_sr)?;
    let mut k_lr = Array2::<f64>::zeros((n, n));
    dfk_lr.build(d, &mut k_lr)?;
    // In-place: k_sr *= c_sr; k_sr += c_lr·k_lr; f −= scale·k_sr. Same
    // element-wise association as `c_sr*&k_sr + c_lr*&k_lr` followed by
    // `f.scaled_add(-scale, &k_total)` — bit-identical, no `k_total` temp.
    k_sr *= c_sr;
    k_sr.scaled_add(c_lr, &k_lr);
    f.scaled_add(-scale, &k_sr);
    Ok(())
}
