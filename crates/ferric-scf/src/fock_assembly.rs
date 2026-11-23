//! Shared J/K route decision and exchange-assembly helpers for all six SCF
//! variants.
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
//! # The J/K route
//!
//! [`resolve_jk_route`] turns `df_j_aux`, `df_k_aux`, `k_builder`, the
//! exchange mix and the spin case into the builders the SCF runs
//! ([`JkRoute`]). The three solvers build exactly what it returns, record it
//! in `ScfResult::df_jk` (via [`JkRoute::record`]), and the certificate and
//! the CLI's `SCF J/K` line read the same decision back. Every combination:
//!
//! | Functional | Spin | `df_j_aux` unset | `df_j_aux = ""` | `df_j_aux = name` |
//! |---|---|---|---|---|
//! | none (HF) | any | four-centre J | four-centre J | RI-J on `name` |
//! | any functional | restricted | RI-J on `def2-universal-jkfit` | four-centre J | RI-J on `name` |
//! | any functional | open | four-centre J | four-centre J | RI-J on `name` |
//!
//! J does not depend on ω: a range-separated hybrid fits J exactly as a
//! global hybrid does, in all three solvers.
//!
//! | Functional | Spin | `df_k_aux` unset | `df_k_aux = ""` | `df_k_aux = name` |
//! |---|---|---|---|---|
//! | none (HF) | any | exact K | exact K | RI-K on `name` |
//! | global hybrid | restricted | RI-K on `def2-universal-jkfit` | exact K | RI-K on `name` |
//! | global hybrid | open | exact K | exact K | RI-K on `name` |
//! | pure (no exact exchange) | any | no K | no K | no K (the fit is dropped) |
//! | range-separated (ω > 0) | any | SR/LR pair on `def2-universal-jkfit` | refused | SR/LR pair on `name` |
//!
//! "exact K" is the four-centre builder, or the pluggable one `k_builder`
//! names. `k_builder` is honoured only where exchange is not density-fitted:
//!
//! - `"cosx"`: replaces the ω = 0 exchange of HF and global hybrids (auto
//!   default included), with RI-J kept where the J row above fits it (the
//!   RIJCOSX composite). A NAMED `df_k_aux` next to it is an error. Ignored
//!   (with a warning) for pure and range-separated functionals.
//! - `"link"`: needs exact J as well as exact K (it rides the four-centre J
//!   pass); ignored, with a warning, under any RI-J or RI-K.
//! - `"direct"`: the default builder, named.
//!
//! The open-shell solvers take no auto default because their callers (CLI,
//! Python `run_uks`/`run_roks`/`run_dft`, the geometry drivers) name the KS
//! default aux explicitly; a library caller passing `None` to `solve_uhf`
//! gets the four-centre route it has always had.
//!
//! Before this module each solver carried its own verbatim copy of the RSH
//! fitter-pair construction and the per-density RSH fold, so an exchange-
//! assembly fix had to be applied three times in parallel (2026-07-22: the
//! in-place no-clone Fock assembly landed as three identical edits). New
//! exchange-assembly changes go HERE, once, and reach RHF/UHF/ROHF and their
//! KS variants together.
//!
//! Known remaining duplication, deliberately NOT unified here: each solver
//! still hand-rolls its builder construction (combined `DirectJK` in RHF,
//! per-spin `DfK`/`DirectK` in UHF, `DirectJ`+`DirectK` in ROHF) and its
//! iteration-loop scaffolding (DIIS error, COSMO/PCM hooks, convergence
//! bookkeeping). Those differences are behavioral, not cosmetic, and the loops
//! are convergence-critical — folding them into one spin-generic driver is a
//! separate, carefully-validated refactor, not a drive-by extraction.

use crate::df_j::{DfJ, ResolvedRiJPrecision, RiJPrecision};
use crate::df_k::DfK;
use crate::fock::KBuilder;
use crate::options::KBuilderKind;
use crate::split_rij::RiJ;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::{jk_storage, JkStorage};
use ndarray::Array2;

/// Default JK-fit auxiliary basis when the caller hasn't set one (the same
/// default the solvers use for RI-J/RI-K auto-selection).
pub(crate) const DEFAULT_JK_AUX: &str = crate::options::DEFAULT_JK_AUX;

/// The exchange a solver's functional asks for, as the J/K route reads it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JkExchange {
    /// Whether a molecular XC functional is active (`config.xc`); `false` for
    /// Hartree-Fock and for an injected (periodic) XC, whose route is all
    /// exact by construction (`rhf::validate_injected`).
    pub functional: bool,
    /// The exact-exchange mix (`KMix::default()` = HF).
    pub k_mix: ferric_dft::xc_trait::KMix,
}

impl JkExchange {
    /// The exchange `config` asks for, read the way the solvers read it:
    /// `config.xc` through libxc, with `config.xc_omega` replacing ω. An
    /// unresolvable name counts as Hartree-Fock exchange (the SCF itself
    /// reports the bad name).
    pub fn from_config(config: &crate::rhf::RhfConfig) -> Self {
        let Some(name) = config.xc.as_deref() else {
            return JkExchange {
                functional: false,
                k_mix: ferric_dft::xc_trait::KMix::default(),
            };
        };
        let def = match config.xc_omega {
            Some(w) => ferric_dft::libxc::xc_def_from_name_nspin_omega(name, 1, w),
            None => ferric_dft::libxc::xc_def_from_name(name),
        };
        JkExchange {
            functional: true,
            k_mix: def.map_or_else(
                |_| ferric_dft::xc_trait::KMix::default(),
                |d| ferric_dft::libxc::k_mix_from_xc_def(&d),
            ),
        }
    }

    /// Whether the SCF consumes exact exchange at all: HF, a global hybrid or
    /// a range-separated hybrid. `false` only for a pure functional.
    pub fn k_consumed(&self) -> bool {
        !self.functional || self.k_mix.sr > 0.0 || self.k_mix.omega > 0.0
    }
}

/// The J/K builders an SCF runs, resolved once by [`resolve_jk_route`] (see
/// the module table).
#[derive(Clone, Debug, PartialEq)]
pub struct JkRoute {
    /// Auxiliary basis of RI-J, or `None` for four-centre J.
    pub j_aux: Option<String>,
    /// Auxiliary basis of the ω = 0 RI-K (HF / global-hybrid exchange), or
    /// `None` when that exchange is exact, COSX, or not consumed.
    pub k_aux: Option<String>,
    /// Range-separated exchange: aux basis and ω of the SR/LR `DfK` pair.
    pub rsh_pair: Option<(String, f64)>,
    /// `k_builder = "cosx"` supplies the ω = 0 exchange in place of RI-K.
    pub cosx_k: bool,
    /// Exchange is density-fitted, or a requested fit was dropped because
    /// the functional uses no exact exchange: a pluggable `k_builder` is
    /// then overridden (`resolve_k_builder`'s `df_k_active`).
    pub k_df_routed: bool,
    /// Whether the SCF consumes exact exchange ([`JkExchange::k_consumed`]).
    pub k_consumed: bool,
    /// ω of the functional (0 for HF, pure and global hybrids).
    pub omega: f64,
}

/// The aux basis of a range-separated SR/LR fitter pair: `df_k_aux`, else
/// `DEFAULT_JK_AUX`. One rule for `build_rsh_dfk_pair` and the route.
fn rsh_pair_aux(df_k_aux: Option<&str>) -> &str {
    df_k_aux.unwrap_or(DEFAULT_JK_AUX)
}

/// Resolve the J/K route of an SCF: what `config`'s `df_j_aux`, `df_k_aux`
/// and `k_builder` mean for `exchange` in the `spin` solver. The table in
/// the module doc lists every case. `Spin::Unrestricted` and
/// `Spin::RestrictedOpen` resolve identically.
///
/// `Err` for `k_builder = "cosx"` next to a named `df_k_aux` (two builders
/// for one exchange matrix, `cosx_replaces_df_k`).
pub fn resolve_jk_route(
    config: &crate::rhf::RhfConfig,
    exchange: JkExchange,
    spin: crate::result::Spin,
) -> Result<JkRoute, FerricError> {
    let restricted = spin == crate::result::Spin::Restricted;
    let omega = exchange.k_mix.omega;
    let k_consumed = exchange.k_consumed();
    // The restricted solver auto-defaults a functional's fits; `""` is the
    // explicit four-centre opt-out on every solver.
    let auto_j = restricted && exchange.functional;
    let auto_k = auto_j && (exchange.k_mix.sr > 0.0 || omega > 0.0);
    let resolve = |requested: Option<&str>, auto: bool| -> Option<String> {
        match requested {
            None => auto.then(|| DEFAULT_JK_AUX.to_string()),
            Some("") => None,
            Some(name) => Some(name.to_string()),
        }
    };
    // The K that a single ω = 0 exchange builder supplies reaches F only for
    // HF and global hybrids; a range-separated hybrid takes its exchange
    // from the SR/LR pair, a pure functional takes none.
    let k_single_consumed = k_consumed && omega == 0.0;
    let cosx_k = cosx_replaces_df_k(
        config.k_builder,
        config.df_k_aux.as_deref(),
        k_single_consumed,
    )?;
    let k_aux = if k_single_consumed && !cosx_k {
        resolve(config.df_k_aux.as_deref(), auto_k)
    } else {
        None
    };
    let named_k = config.df_k_aux.as_deref().is_some_and(|s| !s.is_empty());
    let rsh_pair = (omega > 0.0)
        .then(|| rsh_pair_aux(config.df_k_aux.as_deref()))
        .filter(|a| !a.is_empty())
        .map(|a| (a.to_string(), omega));
    Ok(JkRoute {
        j_aux: resolve(config.df_j_aux.as_deref(), auto_j),
        k_df_routed: k_aux.is_some() || omega > 0.0 || (!k_consumed && named_k),
        k_aux,
        rsh_pair,
        cosx_k,
        k_consumed,
        omega,
    })
}

impl JkRoute {
    /// The record of this route in `ScfResult::df_jk` (`None` when nothing is
    /// fitted). `ri_j_precision` is what the RI-J tensor was stored at.
    pub fn record(
        &self,
        op: Operator,
        budget_bytes: usize,
        ri_j_precision: RiJPrecision,
    ) -> Option<crate::result::DfJkRoute> {
        crate::result::DfJkRoute::from_scf(
            self.j_aux.as_deref(),
            self.k_aux.as_deref(),
            self.rsh_pair.as_ref().map(|(a, w)| (a.as_str(), *w)),
            op,
            budget_bytes,
            ri_j_precision,
        )
    }

    /// Whether `record` (an SCF's `df_jk`) names the same fitted builders as
    /// this route: the RI-J aux, the ω = 0 RI-K aux and the SR/LR pair.
    /// Storage precisions are not part of the route.
    pub fn matches_record(&self, record: Option<&crate::result::DfJkRoute>) -> bool {
        let rsh = self.rsh_pair.as_ref().map(|(a, w)| (a.as_str(), *w));
        match record {
            None => self.j_aux.is_none() && self.k_aux.is_none() && rsh.is_none(),
            Some(r) => {
                r.j_aux == self.j_aux
                    && r.k_aux == self.k_aux
                    && r.rsh_k.as_ref().map(|(a, w)| (a.as_str(), *w)) == rsh
            }
        }
    }

    /// The pluggable exchange builder (`"link"` / `"cosx"`) this route runs
    /// for `k_builder`, or `None`: the decision of `resolve_k_builder` and
    /// `narrow_k_builder_to_supported` without their warnings, for readers
    /// that describe a finished run (the certificate).
    pub fn pluggable_k(&self, k_builder: Option<KBuilderKind>) -> Option<KBuilderKind> {
        let kind = k_builder.filter(|k| *k != KBuilderKind::Direct)?;
        let overridden = self.k_df_routed || (self.j_aux.is_some() && kind != KBuilderKind::Cosx);
        (!overridden && self.k_consumed && self.omega <= 0.0).then_some(kind)
    }
}

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
///
/// `j_pair_thresh` is the Schwarz threshold for RI-J's pair-screened storage
/// (callers pass `[scf] integral_thresh`): a J-only source keeps just the AO
/// pairs whose rows can exceed it, see
/// [`PairScreen`](ferric_integrals::three_index_source::PairScreen). `<= 0`
/// keeps every pair. A source shared with DF-K is never screened.
///
/// `j_precision` is the RESOLVED [`RiJPrecision`] of the RI-J tensor and why
/// (`config.resolved_ri_j_precision()`); the builder's own
/// [`RiJ::precision`](crate::split_rij::RiJ::precision) says what was used
/// (an f32 request falls back to f64 when the f32 tier declines, and
/// Split-RI-J has no tensor).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_df_jk<'a>(
    ctx: &'a ParallelContext,
    mol: &Molecule,
    op: Operator,
    prep: &PreparedBasis,
    j_aux: Option<&str>,
    k_aux: Option<&str>,
    ooc_budget: usize,
    j_pair_thresh: f64,
    resolved: ResolvedRiJPrecision,
    k_precision: crate::df_k::RiKPrecision,
) -> Result<(Option<RiJ<'a>>, Option<DfK<'a>>), FerricError> {
    let j_precision = resolved.precision;
    let mut out = crate::df_j::with_request_source(resolved.source, || {
        build_df_jk_impl(
            ctx,
            mol,
            op,
            prep,
            (j_aux, k_aux),
            ooc_budget,
            (j_pair_thresh, resolved.site),
            j_precision,
            k_precision,
        )
    })?;
    // Every stored route (including a tensor shared with DF-K) gates its
    // device build on the SCF's site.
    if let Some(j) = out.0.as_mut().and_then(RiJ::as_stored_mut) {
        j.route_device_by(resolved.site);
    }
    note_df_k_built(out.1.is_some());
    if let (Some(j), Some(rl)) = (out.0.as_ref(), crate::runlog::log()) {
        rl.note(
            "ri_j_tensor",
            serde_json::json!({
                "aux": j_aux,
                "precision_requested": j_precision.as_str(),
                "precision_source": resolved.source.as_str(),
                "precision": j.precision().as_str(),
            }),
        );
    }
    Ok(out)
}

// Test-only observable for "was a DF-K fitter constructed on THIS thread".
// Thread-local rather than a global atomic so parallel `cargo test` threads
// cannot see each other's builds; the SAD free-atom solves run inside their
// own single-thread rayon pool (`guess::run_serial_pool`), so they do not
// increment the caller's counter either. Used by `rhf::tests` to prove a pure
// functional never builds the DfK whose K it would discard.
#[cfg(test)]
thread_local! {
    pub(crate) static DF_K_BUILT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn note_df_k_built(built: bool) {
    if built {
        DF_K_BUILT.with(|c| c.set(c.get() + 1));
    }
}

#[cfg(not(test))]
#[inline(always)]
fn note_df_k_built(_built: bool) {}

#[allow(clippy::too_many_arguments)]
fn build_df_jk_impl<'a>(
    ctx: &'a ParallelContext,
    mol: &Molecule,
    op: Operator,
    prep: &PreparedBasis,
    (j_aux, k_aux): (Option<&str>, Option<&str>),
    ooc_budget: usize,
    (j_pair_thresh, j_site): (f64, ferric_core::gpu::DeviceSite),
    j_precision: RiJPrecision,
    k_precision: crate::df_k::RiKPrecision,
) -> Result<(Option<RiJ<'a>>, Option<DfK<'a>>), FerricError> {
    // `Some("")` is the explicit "do not density-fit" sentinel (see
    // `solve_rhf`); UHF/ROHF pass the config straight through, so honour it
    // here rather than letting it reach `basis::bundled("")`.
    let j_aux = j_aux.filter(|s| !s.is_empty());
    let k_aux = k_aux.filter(|s| !s.is_empty());
    if j_aux.is_none() && k_aux.is_none() {
        return Ok((None, None));
    }
    // Shared PreparedBasis only when both names are set AND identical —
    // otherwise each requested side builds its own (still only once per
    // side, same as before).
    if let (Some(ja), Some(ka)) = (j_aux, k_aux) {
        if ja == ka {
            // Several ranks: the shared tensor is striped for DF-K, and
            // Split-RI-J runs on one rank only.
            if ctx.size > 1 {
                warn_split_bypassed("it runs on one rank only, and this run has several");
            }
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
                use ferric_integrals::three_index_source::ThreeIndexSource;
                let naux = dfbs.nbasis();
                // Leave room for the packed dressed band DF-K will hold beside
                // the raw tensor (`DfK::packed_band_bytes`).
                let beside = DfK::packed_band_bytes(naux, prep.nbasis(), k_precision);
                let mut raw = if dress_first(op, (prep, &dfbs), ooc_budget, beside, k_precision) {
                    ThreeIndexSource::build_dress_stream(op, mol, prep, &dfbs, ooc_budget)?
                } else {
                    ThreeIndexSource::build_for_jk_beside(
                        op,
                        mol,
                        prep,
                        &dfbs,
                        ooc_budget,
                        (0, naux),
                        true,
                        beside,
                    )?
                };
                let df_k = Some(DfK::from_full_raw_with_precision(
                    &mut raw,
                    prep,
                    &dfbs,
                    op,
                    ooc_budget,
                    Some(ctx),
                    k_precision,
                )?);
                // DF-K has its dressed copy now. A raw tensor that was spilled
                // or is recomputed becomes RI-J's f32 tensor by one streamed
                // pass under `auto` when f32 is asked for and fits beside the
                // dressed band (the values the resident route stores).
                // Otherwise Split-RI-J for J when asked, or under `auto` when
                // raw is not resident (J would pay that every pass): the raw
                // tensor is then freed here instead of kept for J.
                let raw = if jk_storage() == JkStorage::Auto {
                    match DfJ::from_streamed_source_f32(
                        raw,
                        op,
                        &dfbs,
                        ooc_budget,
                        rij_f32_beside(df_k.as_ref(), beside, ooc_budget),
                        Some(ctx),
                        j_precision,
                    )? {
                        Ok(j) => {
                            let mut df_j = Some(RiJ::Stored(j));
                            yield_device_to_k(&mut df_j, &df_k);
                            return Ok((df_j, df_k));
                        }
                        Err(raw) => raw,
                    }
                } else {
                    raw
                };
                let mut df_j = Some(
                    if split_j_for_shared_raw(op, prep, &dfbs, raw.is_resident()) {
                        drop(raw);
                        RiJ::Split(Box::new(crate::split_rij::SplitRiJ::new(
                            op,
                            prep,
                            &dfbs,
                            j_pair_thresh,
                        )?))
                    } else {
                        RiJ::Stored(DfJ::from_source_beside(
                            raw,
                            op,
                            &dfbs,
                            (
                                ooc_budget,
                                rij_f32_beside(df_k.as_ref(), beside, ooc_budget),
                            ),
                            Some(ctx),
                            j_precision,
                        )?)
                    },
                );
                yield_device_to_k(&mut df_j, &df_k);
                return Ok((df_j, df_k));
            }
            let df_j = Some(RiJ::Stored(DfJ::new_for_jk(
                op,
                mol,
                prep,
                &dfbs,
                ooc_budget,
                Some(ctx),
                j_pair_thresh,
                j_precision,
                j_site,
            )?));
            let df_k = Some(DfK::new_banded_with_precision(
                op,
                prep,
                &dfbs,
                ooc_budget,
                Some(ctx),
                k_precision,
            )?);
            return Ok((df_j, df_k));
        }
    }
    let mut df_j = if let Some(aux_name) = j_aux {
        let dfbs_set = ferric_core::basis::bundled(aux_name)?;
        let dfbs = PreparedBasis::new(mol, &dfbs_set)?;
        Some(j_only_builder(
            ctx,
            mol,
            op,
            (prep, &dfbs),
            ooc_budget,
            (j_pair_thresh, j_site),
            j_precision,
        )?)
    } else {
        None
    };
    let df_k = if let Some(aux_name) = k_aux {
        let dfbs_set = ferric_core::basis::bundled(aux_name)?;
        let dfbs = PreparedBasis::new(mol, &dfbs_set)?;
        Some(DfK::new_banded_with_precision(
            op,
            prep,
            &dfbs,
            ooc_budget,
            Some(ctx),
            k_precision,
        )?)
    } else {
        None
    };
    yield_device_to_k(&mut df_j, &df_k);
    Ok((df_j, df_k))
}

/// The dressed DF-K band an f32 RI-J conversion counts beside it: its
/// packed size `band_bytes` (`DfK::packed_band_bytes`, the figure the raw
/// placement reserves room for) when the budget alone holds it, else 0
/// (spilled). Not the live `DfK::resident_tensor_bytes`: whether the band
/// stayed in core also reads the pool, and the f32/f64 choice must depend on
/// the shapes and the budget only.
/// What a shared RI-JK set-up's f32 RI-J conversion counts beside it: the
/// band at its budget placement ([`budget_placed_band_bytes`]) when DF-K was
/// built, else nothing. The live `df_k` says only whether there is a band;
/// whether it stayed in core or spilled is deliberately not read.
fn rij_f32_beside(df_k: Option<&DfK<'_>>, band_bytes: usize, budget: usize) -> usize {
    df_k.map_or(0, |_| budget_placed_band_bytes(band_bytes, budget))
}

fn budget_placed_band_bytes(band_bytes: usize, budget: usize) -> usize {
    if band_bytes <= budget {
        band_bytes
    } else {
        0
    }
}

/// Dress-first RI-JK ([`ThreeIndexSource::dress_first_wanted`]) for a shared
/// raw tensor: its memory tiers would hold it, but not beside DF-K's dressed
/// band of `dressed` bytes, which fits alone. Raw is then recomputed in one
/// streamed pass to dress K (bitwise the resident dressing) and not held; J
/// takes RI-J's f32 tensor by a second streamed pass when that fits, else
/// Split-RI-J. The run this reroutes held raw and spilled the dressed band,
/// re-reading it every iteration.
///
/// `auto` and `split` only (the others name raw's placement), single rank, a
/// DF-K that dresses a streamed raw into its packed f64 band
/// ([`DfK::dresses_streamed_raw_packed`]), and an operator and bases Split-RI-J
/// supports (so J never falls back to recomputing raw every pass).
///
/// [`ThreeIndexSource::dress_first_wanted`]: ferric_integrals::three_index_source::ThreeIndexSource::dress_first_wanted
fn dress_first(
    op: Operator,
    (prep, dfbs): (&PreparedBasis, &PreparedBasis),
    budget: usize,
    dressed: usize,
    k_precision: crate::df_k::RiKPrecision,
) -> bool {
    use ferric_integrals::three_index_source::ThreeIndexSource;
    let (nao, naux) = (prep.nbasis(), dfbs.nbasis());
    let taken = matches!(jk_storage(), JkStorage::Auto | JkStorage::Split)
        && DfK::dresses_streamed_raw_packed(k_precision, nao)
        && ThreeIndexSource::dress_first_wanted(nao, naux, budget, dressed)
        && crate::split_rij::SplitRiJ::supports(op, prep, dfbs);
    if taken {
        DRESS_FIRST_TAKEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let packed = naux * nao * (nao + 1) / 2 * 8;
        eprintln!(
            "[jk_storage] {}: the raw 3-index tensor ({:.2} GB packed) and DF-K's dressed \
             band ({:.2} GB) do not fit the memory budget ({:.2} GB) together; raw is \
             recomputed once to dress K and not held",
            jk_storage(),
            packed as f64 / 1e9,
            dressed as f64 / 1e9,
            budget as f64 / 1e9,
        );
    }
    taken
}

static DRESS_FIRST_TAKEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// How many shared RI-JK setups in this process took the dress-first route
/// ([`dress_first`]). Test hook.
#[doc(hidden)]
pub fn dress_first_count_for_test() -> usize {
    DRESS_FIRST_TAKEN.load(std::sync::atomic::Ordering::SeqCst)
}

/// `jk_storage = "split"` asked for, but J is built from a stored tensor:
/// say why, once per process. No-op for every other policy.
fn warn_split_bypassed(why: &str) {
    use ferric_integrals::three_index_source::{jk_storage, JkStorage};
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if jk_storage() == JkStorage::Split && !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed)
    {
        eprintln!(
            "[ferric] warning: jk_storage = \"split\" not used for RI-J: {why}; \
             the RI-J tensor is stored as under \"auto\""
        );
    }
}

/// The RI-J builder when J has its own tensor (no DF-K sharing it).
///
/// Split-RI-J for `jk_storage = "split"`, and under `auto` when the stored
/// tensor would not be held in memory in any tier (`DfJ::stored_fits_in_memory`,
/// the builders' own fit checks), single rank, for an operator and bases Split-RI-J
/// supports. Measured on danuglipron PBE/def2-SVP: Split-RI-J 81 s against
/// 584 s recomputing and 600 s spilling the tensor (46 s held in memory).
/// Otherwise the stored [`DfJ`], which picks its own tier.
fn j_only_builder<'a>(
    ctx: &'a ParallelContext,
    mol: &Molecule,
    op: Operator,
    (prep, dfbs): (&PreparedBasis, &PreparedBasis),
    ooc_budget: usize,
    (j_pair_thresh, j_site): (f64, ferric_core::gpu::DeviceSite),
    j_precision: RiJPrecision,
) -> Result<RiJ<'a>, FerricError> {
    use crate::split_rij::SplitRiJ;
    use ferric_integrals::three_index_source::{jk_storage, JkStorage};
    let single = ctx.size <= 1;
    match jk_storage() {
        // Single rank only: it does not stripe the aux band.
        JkStorage::Split if !single => {
            warn_split_bypassed("it runs on one rank only, and this run has several");
        }
        JkStorage::Split => {
            return Ok(RiJ::Split(Box::new(SplitRiJ::new(
                op,
                prep,
                dfbs,
                j_pair_thresh,
            )?)));
        }
        JkStorage::Auto if single && SplitRiJ::supports(op, prep, dfbs) => {
            let v = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
            if !DfJ::stored_fits_in_memory(
                op,
                (prep, dfbs),
                ooc_budget,
                (j_pair_thresh, j_site),
                &v,
                j_precision,
            ) {
                note_auto_split();
                return Ok(RiJ::Split(Box::new(SplitRiJ::with_metric(
                    op,
                    prep,
                    dfbs,
                    v,
                    j_pair_thresh,
                )?)));
            }
        }
        _ => {}
    }
    Ok(RiJ::Stored(DfJ::new_for_jk(
        op,
        mol,
        prep,
        dfbs,
        ooc_budget,
        Some(ctx),
        j_pair_thresh,
        j_precision,
        j_site,
    )?))
}

/// Shared J/K raw tensor (single rank): J by Split-RI-J for `"split"`, or
/// under `auto` when the raw tensor is not resident (spilled or recomputed)
/// and Split-RI-J supports the operator and bases.
fn split_j_for_shared_raw(
    op: Operator,
    prep: &PreparedBasis,
    dfbs: &PreparedBasis,
    raw_resident: bool,
) -> bool {
    use ferric_integrals::three_index_source::{jk_storage, JkStorage};
    match jk_storage() {
        JkStorage::Split => true,
        JkStorage::Auto => {
            let split = !raw_resident && crate::split_rij::SplitRiJ::supports(op, prep, dfbs);
            if split {
                note_auto_split();
            }
            split
        }
        _ => false,
    }
}

/// `auto` chose Split-RI-J: say so once per build (stderr).
fn note_auto_split() {
    eprintln!(
        "[jk_storage] auto: the RI-J tensor does not fit the memory budget; \
         J by Split-RI-J (recomputed in contracted form each build, no stored J tensor)"
    );
}

/// RI-J is built before DF-K every iteration, so under `gpu = on` its device
/// upload would take the pool first. Tell it how much DF-K will want.
#[cfg(feature = "gpu")]
fn yield_device_to_k(df_j: &mut Option<RiJ<'_>>, df_k: &Option<DfK<'_>>) {
    if let (Some(j), Some(k)) = (df_j.as_mut().and_then(RiJ::as_stored_mut), df_k.as_ref()) {
        j.reserve_device_for_k(k.device_footprint_bytes());
    }
}

#[cfg(not(feature = "gpu"))]
fn yield_device_to_k(_df_j: &mut Option<RiJ<'_>>, _df_k: &Option<DfK<'_>>) {}

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
    k_precision: crate::df_k::RiKPrecision,
) -> Result<(DfK<'a>, DfK<'a>), FerricError> {
    let aux_name = rsh_pair_aux(df_k_aux);
    let dfbs_set = ferric_core::basis::bundled(aux_name)?;
    let dfbs_prep = PreparedBasis::new(mol, &dfbs_set)?;
    Ok((
        DfK::new_banded_with_precision(
            Operator::erfc(omega),
            prep,
            &dfbs_prep,
            ooc_budget,
            Some(ctx),
            k_precision,
        )?,
        DfK::new_banded_with_precision(
            Operator::erf(omega),
            prep,
            &dfbs_prep,
            ooc_budget,
            Some(ctx),
            k_precision,
        )?,
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
/// Say which pluggable builder (if any) the caller should construct for
/// `config.k_builder` (already parsed at the frontend, by
/// [`KBuilderKind::parse`]).
///
/// Shared by `solve_rhf`, `solve_uhf` and `solve_rohf` so the DF-J/DF-K
/// precedence rule lives in ONE place. Before this existed only `solve_rhf`
/// read the field at all, so `k_builder = "link"` / `"cosx"` was silently
/// ignored for every open-shell run.
///
/// * `None` — no pluggable builder: the field is unset, is
///   [`KBuilderKind::Direct`], or is overridden by an active density-fitted
///   path (warned about, below).
/// * `Some(kind)` — construct that builder ([`KBuilderKind::Link`] or
///   [`KBuilderKind::Cosx`]).
///
/// The precedence rule (RIJCOSX):
///
/// * an active DF-K (`df_k_active`) always wins — a pluggable K would be
///   built and thrown away, so it is skipped WITH A WARNING;
/// * an active DF-J alone skips `"link"` (its J comes from the combined
///   four-centre pass, which DF-J replaces), but NOT `"cosx"`: COSX supplies
///   only K, so DF-J + COSX is the RIJCOSX composite (Neese et al. 2009;
///   ORCA RIJCOSX, Psi4 `DFDIRJ+COSX`). The solvers then take J from `DfJ`
///   and K from COSX.
///
/// The IGNORED warning fires for every explicitly passed value the
/// density-fitted path overrides, `"direct"` included: an inert `"direct"`
/// is as invisible in the energy as an inert `"link"` (both equal the
/// default bitwise), so only the warning tells a caller the knob did nothing.
/// The one `"direct"` it does not fire for is DF-J alone with an ω = 0 exact
/// exchange the SCF consumes: K then comes from the direct
/// four-centre builder, so the request is honoured. See
/// [`k_builder_ignored_warning`].
///
/// `df_k_active` is [`JkRoute::k_df_routed`]: a DF-K the route builds, the
/// SR/LR pair of a range-separated functional, or a requested fit dropped
/// because the functional uses no exact exchange. With `k_builder = "cosx"` the solvers
/// never build the ω = 0 DF-K in the first place (see
/// [`cosx_replaces_df_k`]). `df_k_present`, `need_k` (the functional consumes
/// exact exchange at all) and `omega` only pick the wording — see
/// [`exchange_route_when_df_active`].
pub(crate) fn resolve_k_builder(
    k_builder: Option<KBuilderKind>,
    df_j_active: bool,
    df_k_active: bool,
    df_k_present: bool,
    need_k: bool,
    omega: f64,
) -> Option<KBuilderKind> {
    let kb = k_builder?;
    let pluggable = (kb != KBuilderKind::Direct).then_some(kb);
    let overridden = df_k_active || (df_j_active && kb != KBuilderKind::Cosx);
    if overridden {
        if let Some(msg) = k_builder_ignored_warning(kb, df_k_active, df_k_present, need_k, omega) {
            eprintln!("{msg}");
        }
        return None;
    }
    pluggable
}

/// The IGNORED warning [`resolve_k_builder`] prints for a validated `kb` that
/// an active density-fitted path overrides (the caller has already decided
/// `overridden`), or `None` when the request is in fact honoured.
///
/// `"link"` / `"cosx"` are never honoured once overridden. `"direct"` is
/// honoured exactly when exchange still comes from the direct four-centre
/// builder: no DF-K (active or dropped), an exact exchange the SCF consumes,
/// and ω = 0, i.e. DF-J alone next to HF or a global hybrid with
/// `df_k_aux = ""`. Everywhere else (DF-K active, a pure functional, a
/// range-separated one) `"direct"` names a builder that never runs.
///
/// Pure so the decision is testable without capturing stderr.
pub(crate) fn k_builder_ignored_warning(
    kb: KBuilderKind,
    df_k_active: bool,
    df_k_present: bool,
    need_k: bool,
    omega: f64,
) -> Option<String> {
    let direct_k_runs = !df_k_active && need_k && omega <= 0.0;
    if kb == KBuilderKind::Direct && direct_k_runs {
        return None;
    }
    if !need_k {
        // No exchange builder runs, density-fitted or not; a named df_k_aux
        // next to a pure functional builds no fitter.
        return Some(format!(
            "[ferric] warning: k_builder = \"{kb}\" is IGNORED because the functional uses no exact exchange"
        ));
    }
    Some(format!(
        "[ferric] warning: k_builder = \"{kb}\" is IGNORED because density-fitted J/K is active \
         (df_j_aux/df_k_aux set, or auto-defaulted for a functional); {}",
        exchange_route_when_df_active(df_k_present, need_k, omega)
    ))
}

/// RIJCOSX: does `k_builder = "cosx"` REPLACE the ω = 0 density-fitted
/// exchange (DF-K) for this run?
///
/// `k_consumed_w0` is whether the SCF consumes a plain (ω = 0) exact-exchange
/// matrix at all: Hartree–Fock and global hybrids yes; pure functionals (no
/// exact exchange) and range-separated hybrids (exchange from the SR/LR DF
/// fitters, which have no COSX form) no — those keep their existing routing
/// and warnings.
///
/// * `Ok(true)` — COSX supplies K: the solver must NOT build the ω = 0 DF-K
///   (neither auto-defaulted nor requested) and takes J from DF-J if one is
///   active, else from the four-centre builder.
/// * `Ok(false)` — not a COSX run, or COSX does not apply to this exchange.
/// * `Err` — `k_builder = "cosx"` together with an EXPLICITLY NAMED `df_k_aux`
///   (a non-empty name): two exchange builders were requested for one K. A
///   hard error rather than a silent pick, per the config-honesty convention;
///   `df_k_aux = ""` (the exact sentinel) or unset are accepted.
pub(crate) fn cosx_replaces_df_k(
    k_builder: Option<KBuilderKind>,
    df_k_aux: Option<&str>,
    k_consumed_w0: bool,
) -> Result<bool, FerricError> {
    if k_builder != Some(KBuilderKind::Cosx) || !k_consumed_w0 {
        return Ok(false);
    }
    if let Some(name) = df_k_aux.filter(|s| !s.is_empty()) {
        return Err(FerricError::General(format!(
            "k_builder = \"cosx\" conflicts with df_k_aux = \"{name}\": both name a builder for the \
             same exchange matrix. COSX is the exchange builder of the RIJCOSX composite (DF-J for \
             Coulomb, COSX for exchange); leave df_k_aux unset (or \"\") with k_builder = \"cosx\", \
             or drop k_builder to use RI-K"
        )));
    }
    Ok(true)
}

/// Where exchange actually comes from when [`resolve_k_builder`] skips a
/// pluggable builder because density fitting is active.
///
/// Before this existed the warning said "the direct 4-centre builder" whenever
/// DF-K was absent, which was wrong for the two cases where `solve_rhf` drops
/// DF-K on purpose: a pure functional builds no K at all, and an RSH
/// functional contracts K from its SR/LR DF fitters (there is no four-centre
/// erf/erfc exchange path). Same wording as [`narrow_k_builder_to_supported`].
pub(crate) fn exchange_route_when_df_active(
    df_k_present: bool,
    need_k: bool,
    omega: f64,
) -> &'static str {
    if !need_k {
        "the functional uses no exact exchange"
    } else if omega > 0.0 {
        "exchange for a range-separated functional comes from the SR/LR density-fitted fitters"
    } else if df_k_present {
        "exchange comes from DF-K"
    } else {
        "exchange comes from the direct 4-centre builder"
    }
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pluggable_k<'a, B: crate::screening::Bound + Sync>(
    kind: Option<KBuilderKind>,
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
        Some(KBuilderKind::Link) => Ok(Some(Box::new(crate::link_k::LinkK::new(
            ctx,
            prep,
            link_bound,
            op,
            integral_thresh,
            ooc_budget,
        )) as Box<dyn KBuilder + 'a>)),
        Some(KBuilderKind::Cosx) => {
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
        Some(KBuilderKind::Direct) | None => Ok(None),
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
///   contracted from the dedicated SR/LR `DfK` fitter pair. [`resolve_k_builder`]
///   already overrides a pluggable K there (the route marks the SR/LR pair
///   as density-fitted exchange); this second check keeps the warning for a
///   caller that reaches it otherwise, so an RSH run never skips the builder
///   silently.
pub(crate) fn narrow_k_builder_to_supported(
    kind: Option<KBuilderKind>,
    need_k: bool,
    omega: f64,
) -> Option<KBuilderKind> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The band an f32 RI-J conversion counts beside it is placed by the
    /// budget alone: the same figure whether the live DF-K band stayed in core
    /// or was spilled (the live residency is not an input), the band when the
    /// budget holds it, else nothing.
    #[test]
    fn f32_conversion_counts_the_band_at_its_budget_placement() {
        let band = 1_000_000;
        assert_eq!(budget_placed_band_bytes(band, band), band);
        assert_eq!(budget_placed_band_bytes(band, 10 * band), band);
        assert_eq!(budget_placed_band_bytes(band, band - 1), 0);
        assert_eq!(budget_placed_band_bytes(0, 0), 0);
    }

    /// The RI-J precision a shared set-up picks is the same whether the live
    /// DF-K band stayed in core or spilled (which also reads the pool): one
    /// input, one budget, one precision.
    #[test]
    fn rij_precision_is_the_same_with_the_band_resident_or_spilled() {
        use crate::df_j::{DfJ, RiJPrecision};
        use crate::df_k::RiKPrecision;
        use ferric_integrals::three_index_source::ThreeIndexSource;
        let mol = Molecule::parse_xyz(
            "3\nwater\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n",
            0,
            1,
        )
        .unwrap();
        let prep =
            PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(
            &mol,
            &ferric_core::basis::bundled("def2-universal-jkfit").unwrap(),
        )
        .unwrap();
        let op = Operator::coulomb();
        let (naux, nao) = (dfbs.nbasis(), prep.nbasis());
        let band = DfK::packed_band_bytes(naux, nao, RiKPrecision::F64);
        let resident =
            DfK::new_with_precision(op, &prep, &dfbs, 1 << 40, RiKPrecision::F64).unwrap();
        let spilled =
            DfK::new_with_precision(op, &prep, &dfbs, band / 8, RiKPrecision::F64).unwrap();
        assert_ne!(
            resident.resident_tensor_bytes(),
            spilled.resident_tensor_bytes(),
            "the two bands must differ in residency for this test to bite"
        );
        let f32_bytes = naux * nao * (nao + 1) / 2 * 4;
        for budget in [f32_bytes + band / 2, f32_bytes + band + band / 2, 1 << 40] {
            let precision = |k: &DfK<'_>| {
                let beside = rij_f32_beside(Some(k), band, budget);
                let raw = ThreeIndexSource::build(op, &prep, &dfbs, 1 << 40).unwrap();
                DfJ::from_source_beside(raw, op, &dfbs, (budget, beside), None, RiJPrecision::F32)
                    .unwrap()
                    .precision()
            };
            assert_eq!(precision(&resident), precision(&spilled), "budget {budget}");
        }
    }

    /// Dress first, at a budget where raw and DF-K's packed band do not fit
    /// together but the band and the dressing's stream do: DF-K holds just the
    /// packed f64 band, RI-J its f32 tensor, and both give bitwise the J and K
    /// of the resident route (ample budget). Exactness anchor of the route:
    /// at one BLAS thread nothing it changes may reach a bit.
    #[test]
    fn dress_first_is_bitwise_the_resident_route() {
        use crate::df_k::RiKPrecision;
        use crate::fock::{JBuilder, KBuilder};
        use ferric_integrals::three_index_source::ThreeIndexSource;
        assert_eq!(
            std::env::var("OPENBLAS_NUM_THREADS").as_deref(),
            Ok("1"),
            "bit-identity holds at one BLAS thread; run with OPENBLAS_NUM_THREADS=1"
        );
        let path = format!(
            "{}/../../testdata/molecules/alkane_6.xyz",
            env!("CARGO_MANIFEST_DIR")
        );
        let mol = Molecule::load_xyz(&path).unwrap();
        let prep =
            PreparedBasis::new(&mol, &ferric_core::basis::bundled("def2-svp").unwrap()).unwrap();
        let aux = ferric_core::basis::bundled(DEFAULT_JK_AUX).unwrap();
        let (nao, naux) = (
            prep.nbasis(),
            PreparedBasis::new(&mol, &aux).unwrap().nbasis(),
        );
        let dressed = DfK::packed_band_bytes(naux, nao, RiKPrecision::F64);
        let stream = ThreeIndexSource::dress_stream_bytes_for(nao, naux);
        // Raw packed is the band's size: the route's budget band is
        // (dressed + stream, 2·dressed], reachable only when stream < dressed.
        assert!(
            stream < dressed,
            "fixture too small: stream {stream} vs band {dressed}"
        );
        let budget = dressed + stream + (dressed - stream) / 2;
        assert!(ThreeIndexSource::dress_first_wanted(
            nao, naux, budget, dressed
        ));
        let ctx = ParallelContext::default();
        let op = Operator::coulomb();
        let build = |budget: usize| {
            build_df_jk_impl(
                &ctx,
                &mol,
                op,
                &prep,
                (Some(DEFAULT_JK_AUX), Some(DEFAULT_JK_AUX)),
                budget,
                (0.0, ferric_core::gpu::DeviceSite::plain(nao)),
                RiJPrecision::F32,
                RiKPrecision::F64,
            )
            .unwrap()
        };
        let taken = dress_first_count_for_test();
        let (j_ref, k_ref) = build(usize::MAX / 4);
        assert_eq!(
            dress_first_count_for_test(),
            taken,
            "ample budget took the route"
        );
        let (j_new, k_new) = build(budget);
        assert_eq!(dress_first_count_for_test(), taken + 1, "route not taken");
        let (mut j_ref, mut k_ref) = (j_ref.unwrap(), k_ref.unwrap());
        let (mut j_new, mut k_new) = (j_new.unwrap(), k_new.unwrap());
        assert!(k_new.is_packed(), "dressed band not held packed");
        assert_eq!(k_new.resident_tensor_bytes(), dressed);
        assert!(
            matches!(j_new, RiJ::Stored(_)),
            "J not from a stored tensor"
        );
        assert_eq!(j_new.precision(), RiJPrecision::F32);
        // A symmetric, non-idempotent density with mixed signs.
        let d = Array2::from_shape_fn((nao, nao), |(i, j)| {
            let (a, b) = (i.min(j) as f64, i.max(j) as f64);
            0.4 / (b - a + 1.0) - 0.03 * ((a + 2.0 * b) % 5.0)
        });
        let run_j = |j: &mut RiJ<'_>| {
            let mut out = Array2::zeros((nao, nao));
            j.build(&d, &mut out).unwrap();
            out
        };
        let run_k = |k: &mut DfK<'_>| {
            let mut out = Array2::zeros((nao, nao));
            k.build(&d, &mut out).unwrap();
            out
        };
        let bits = |a: &Array2<f64>| a.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&run_k(&mut k_ref)),
            bits(&run_k(&mut k_new)),
            "K moved"
        );
        assert_eq!(
            bits(&run_j(&mut j_ref)),
            bits(&run_j(&mut j_new)),
            "J moved"
        );
    }

    /// The four routes `resolve_k_builder`'s warning can name. The two
    /// `df_k_present = false` cases with `need_k = false` or `omega > 0` are
    /// exactly `solve_rhf`'s `df_k_skipped` runs, which used to be told
    /// exchange came from "the direct 4-centre builder".
    #[test]
    fn exchange_route_names_the_builder_actually_used() {
        let pure = exchange_route_when_df_active(false, false, 0.0);
        assert!(pure.contains("no exact exchange"), "{pure}");
        let rsh = exchange_route_when_df_active(false, true, 0.3);
        assert!(rsh.contains("SR/LR density-fitted"), "{rsh}");
        for route in [pure, rsh] {
            assert!(!route.contains("4-centre"), "{route}");
        }
        assert_eq!(
            exchange_route_when_df_active(true, true, 0.0),
            "exchange comes from DF-K"
        );
        assert_eq!(
            exchange_route_when_df_active(false, true, 0.0),
            "exchange comes from the direct 4-centre builder"
        );
    }

    /// `k_builder = "direct"` under an active DF-K (the `run_dft` hybrid
    /// default) is as inert as `"link"` and must say so: a warning limited to
    /// `"link"`/`"cosx"` leaves a `"direct"` that never ran silent.
    ///
    /// MUTATION: restrict the warning to `matches!(kb, "link" | "cosx")`
    /// and the `"direct"` arms go RED; drop the `!need_k` lead and the two
    /// pure-functional arms go RED.
    #[test]
    fn direct_overridden_by_density_fitting_is_warned_like_link() {
        // (kb, df_k_active, df_k_present, need_k, omega, expected route)
        let cases = [
            // Global hybrid / HF with RI-K: exchange from DF-K.
            ("direct", true, true, true, 0.0, "exchange comes from DF-K"),
            ("link", true, true, true, 0.0, "exchange comes from DF-K"),
            // Pure functional under DF-J (+ a dropped DF-K): no K at all.
            ("direct", true, false, false, 0.0, "no exact exchange"),
            ("direct", false, false, false, 0.0, "no exact exchange"),
            // Range-separated: SR/LR fitters, the main DF-K dropped.
            ("direct", true, false, true, 0.3, "SR/LR density-fitted"),
        ];
        for (kb, k_active, k_present, need_k, omega, route) in cases {
            let msg = k_builder_ignored_warning(
                KBuilderKind::parse(kb).unwrap(),
                k_active,
                k_present,
                need_k,
                omega,
            )
            .unwrap_or_else(|| panic!("{kb} {k_active} {need_k} {omega}: no warning"));
            // A pure functional builds no exchange fitter, so the lead names
            // the missing exact exchange rather than active density fitting.
            let lead = if need_k {
                "density-fitted J/K is active"
            } else {
                "the functional uses no exact exchange"
            };
            assert!(
                msg.contains(&format!("k_builder = \"{kb}\" is IGNORED because {lead}")),
                "{msg}"
            );
            assert!(msg.contains(route), "{msg}");
        }
    }

    /// The one overridden-path `"direct"` that is honoured: DF-J alone next
    /// to a consumed ω = 0 exact exchange (HF, or a global hybrid with
    /// `df_k_aux = ""`). K then comes from the direct four-centre builder, so
    /// a warning would be false. `"link"` in the same state is still skipped
    /// (DF-J replaces the combined pass it rides on) and still warned.
    ///
    /// MUTATION: drop the `direct_k_runs` early return and the first assert
    /// goes RED.
    #[test]
    fn direct_beside_df_j_alone_is_honoured_and_silent() {
        assert_eq!(
            k_builder_ignored_warning(KBuilderKind::Direct, false, false, true, 0.0),
            None
        );
        let link = k_builder_ignored_warning(KBuilderKind::Link, false, false, true, 0.0)
            .expect("link under DF-J alone is skipped, so it must warn");
        assert!(link.contains("direct 4-centre builder"), "{link}");
    }

    /// Without any density fitting nothing overrides `"direct"`: it resolves
    /// to the (default) direct builder, `Ok(None)`, with no warning path
    /// reached; `"link"` resolves to itself. Unchanged by the fix.
    #[test]
    fn direct_without_density_fitting_resolves_unchanged() {
        for need_k in [true, false] {
            assert_eq!(
                resolve_k_builder(Some(KBuilderKind::Direct), false, false, false, need_k, 0.0),
                None
            );
        }
        assert_eq!(
            resolve_k_builder(Some(KBuilderKind::Link), false, false, false, true, 0.0),
            Some(KBuilderKind::Link)
        );
    }
}
