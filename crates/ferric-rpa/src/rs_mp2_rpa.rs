//! SR-MP2 + LR-RPA range-separated correlation (Δ-form B and coupled-rings T).
//! See wiki/superpowers/specs/2026-06-09-sr-mp2-lr-rpa-design.md.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{
    compute_rpa_intermediates, spin_components_from_b_ov, spin_components_from_signed_b_ov,
    RiMp2Config, SignedBOv,
};
use ferric_scf::PreparedAux;
use ferric_scf::ScfResult;

use crate::{run_pdep_rpa_from_intermediates, PdepRpaConfig};

/// Which algebraic form of the range-separated MP2 + RPA functional to use.
///
/// Both forms are exact through 2nd order in the full Coulomb interaction and
/// contain all pure long-range rings. They differ in whether mixed SR×LR rings
/// (3rd order and higher) are included.
///
/// ## Formulation B — Δ-form (`DeltaLr`, default)
///
/// ```text
/// E_c^B = E_MP2[Coulomb] + (E_dRPA[erf_ω] − 2·E_OS[erf_ω])
/// ```
///
/// Resums pure long-range rings via a single dRPA\[erf\] solve. Drops all mixed
/// SR×LR rings at 3rd order and above (leading residual: 3·k_s·k_l·(k_s+k_l)/(4Δ²)
/// in the one-mode ring model). Cost: 1 dRPA\[erf\] call.
///
/// ## Formulation T — coupled rings (`CoupledRings`)
///
/// ```text
/// E_c^T = E_MP2[Coulomb] + (E_dRPA[Coulomb] − 2·E_OS[Coulomb])
///                         − (E_dRPA[erfc_ω] − 2·E_OS[erfc_ω])
/// ```
///
/// Screens ALL rings (full-Coulomb ΔdRPA), then un-screens the short-range-only
/// rings (subtracts erfc ΔdRPA). Contains exact 2nd order + all pure-LR rings +
/// ALL mixed SR×LR rings; excludes pure-SR rings beyond 2nd order (avoiding
/// dRPA's short-range self-correlation hole).
///
/// Same exact limits as B: ω→0 ⇒ erfc→Coulomb, the two ΔdRPA terms cancel ⇒
/// plain MP2; ω→∞ ⇒ erfc→0 ⇒ MP2 + ΔdRPA\[Coulomb\]. Cost: 2 dRPA calls
/// (Coulomb + erfc).
///
/// ## TOML / Python knob
///
/// TOML: `[mp2] formulation = "delta-lr"` (default) or `"coupled-rings"`
/// Python: `run_rs_mp2_rpa(..., formulation="delta-lr")` (default) or
///         `formulation="coupled-rings"`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RsMp2RpaFormulation {
    /// Δ-form (formulation B): E_MP2\[Coulomb\] + (E_dRPA\[erf\] − 2·E_OS\[erf\]).
    /// Pure-LR rings only; mixed SR×LR rings dropped. Default.
    #[default]
    DeltaLr,
    /// Coupled-rings (formulation T): MP2 + ΔdRPA\[Coulomb\] − ΔdRPA\[erfc\].
    /// Adds all mixed SR×LR rings vs DeltaLr; no pure-SR rings ≥3rd order.
    CoupledRings,
}

/// Which range-separation kernel splits Coulomb into short + long range.
///
/// Both choices satisfy the SAME split identity (SR + LR = Coulomb) and thus give
/// the SAME exact limits, so either drops into formulations B and T without
/// changing the ring-diagram telescoping. They differ only in the *shape* of the
/// attenuator, which changes intermediate-ω behavior (basis-set convergence,
/// π-stack over/under-binding), not the ω→0 / ω→∞ endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Attenuator {
    /// Standard error-function split: LR = erf(ωr)/r, SR = erfc(ωr)/r.
    /// Parameterized by ω (Bohr⁻¹). Default — preserves all existing behavior.
    #[default]
    Erf,
    /// Tempered (Dutoi/Goldey) split: LR = terf(r,r0)/r, SR = terfc(r,r0)/r,
    /// with terf + terfc = Coulomb exactly. Parameterized by r0 (Bohr); the
    /// curvature ω = 1/(r0·√2) is derived unless `terf_omega` overrides it.
    /// Exact via 2D interpolation tables (needs FERRIC_TERF_TABLE_DIR).
    ///
    /// The terf kernel is not positive definite, so it is never fitted in its
    /// own metric: the long-range integrals are Coulomb RI minus terfc RI (the
    /// signed factor `ferric_mp2::rimp2::SignedBOv`), the LR dRPA is the
    /// signed log-det ([`crate::energy::signed_drpa_energy`]), and formulation
    /// B refuses PDEP truncation, the Laplace χ₀ and the Boys-screened χ₀.
    Terf,
}

/// Configuration for SR-MP2 + LR-RPA.
#[derive(Debug, Clone)]
pub struct RsMp2RpaConfig {
    /// Range-separation parameter ω in Bohr⁻¹ (CLI/Python boundary is Å⁻¹,
    /// matching att-rimp2). Default 0.222 Bohr⁻¹ = 0.420 Å⁻¹ (erfc-optimal,
    /// Goldey & Head-Gordon JPCL 2012).
    ///
    /// Small-ω caveat: the erf 2-center metric loses rank as ω→0 (regularized
    /// eigh inverse handles it); ω=0.05 Bohr⁻¹ is the tested floor.
    pub omega: f64,
    /// Which attenuator splits Coulomb (see [`Attenuator`]). Default `Erf`.
    ///
    /// When `Terf`, the long/short-range operators become `terf(r0)`/`terfc(r0)`
    /// and `omega` is IGNORED — the curvature is derived from `r0` as
    /// ω = 1/(r0·√2). Use [`RsMp2RpaConfig::r0`] as the single knob then.
    pub attenuator: Attenuator,
    /// Range-separation length r0 in Bohr, used ONLY when `attenuator == Terf`.
    /// By default ω is derived (ω = 1/(r0·√2), the Dutoi curvature link);
    /// see `terf_omega` to decouple. Ignored for `Erf`. Default 3.18 Bohr ⇒
    /// ω ≈ 0.222 Bohr⁻¹ = 0.42 Å⁻¹ (the erf operating point), so the terf and
    /// erf arms are directly comparable at the default.
    pub r0: f64,
    /// Terf-only: OVERRIDE the curvature-linked sharpness with an independent
    /// ω (Bohr⁻¹). `None` (default) keeps the Dutoi link ω = 1/(r0·√2). The
    /// split identity terf + terfc = Coulomb holds for every (r0, ω)
    /// (anchor-tested), and the tables cover s = (ωr0)² ≤ 80. Whether the
    /// curvature constraint binds once LR-RPA computes the complement is open
    /// for terf: the Ne2 terf curves behind the decoupling were computed with
    /// the terf metric's negative modes dropped (see
    /// `Operator::terfc_with_omega`). Ignored for `Erf`.
    pub terf_omega: Option<f64>,
    /// Number of frozen core orbitals for every MP2 and dRPA call (default 0).
    pub frozen_core: usize,
    /// dRPA solver knobs. Default forces trunc_thresh = 0.0 (full rank):
    /// this is an energy method; PDEP truncation is a production-size opt-in.
    ///
    /// Note: the nested `drpa.frozen_core` field is ignored — `cfg.frozen_core`
    /// is authoritative and overwrites it before each RPA call.
    pub drpa: PdepRpaConfig,
    /// Which algebraic formulation to use. Default is `DeltaLr` (formulation B),
    /// which preserves all existing behavior.
    pub formulation: RsMp2RpaFormulation,
}

impl Default for RsMp2RpaConfig {
    fn default() -> Self {
        Self {
            omega: 0.420 * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
            attenuator: Attenuator::Erf,
            r0: 3.18,
            terf_omega: None,
            frozen_core: 0,
            drpa: PdepRpaConfig {
                trunc_thresh: 0.0,
                ..Default::default()
            },
            formulation: RsMp2RpaFormulation::DeltaLr,
        }
    }
}

/// Component breakdown.
///
/// `e_corr` is the selected formulation's correlation energy; semantics follow
/// `formulation`. `total_energy = rhf + e_corr`.
///
/// ## Always-present fields (all formulations)
///
/// - `e_mp2_full`, `e_sr_mp2`, `e_lr_mp2`, `e_dmp2_lr` come from the three
///   spin-component RI-MP2 calls (Coulomb/erfc/erf) which are always run.
///
/// ## Formulation-specific fields
///
/// - `e_drpa_lr` (`Some` for `DeltaLr`, `None` for `CoupledRings`): dRPA\[erf\] energy.
/// - `e_corr_naive` (`Some` for `DeltaLr`, `None` for `CoupledRings`): diagnostic
///   formulation A = E_MP2\[erfc\] + E_dRPA\[erf\]; missing SR×LR cross-range terms.
/// - `e_delta_drpa_full` (`Some` for `CoupledRings`, `None` for `DeltaLr`):
///   ΔdRPA\[Coulomb\] = E_dRPA\[Coulomb\] − 2·E_OS\[Coulomb\].
/// - `e_delta_drpa_sr` (`Some` for `CoupledRings`, `None` for `DeltaLr`):
///   ΔdRPA\[erfc\] = E_dRPA\[erfc\] − 2·E_OS\[erfc\] (the short-range ring contribution
///   that is subtracted to avoid double-counting pure-SR rings).
#[derive(Debug, Clone)]
#[must_use]
pub struct RsMp2RpaResult {
    /// RI-MP2 correlation energy with the full Coulomb operator, Ha.
    pub e_mp2_full: f64,
    /// RI-MP2 correlation energy with the short-range operator (erfc, or
    /// terfc for the terf attenuator), Ha.
    pub e_sr_mp2: f64,
    /// RI-MP2 correlation energy with the long-range operator (erf, or terf), Ha.
    pub e_lr_mp2: f64,
    /// Direct (ring) second-order term with the erf kernel = 2·E_OS\[erf\].
    pub e_dmp2_lr: f64,
    /// E_dRPA\[erf\] (formulation B / DeltaLr only; None for CoupledRings).
    pub e_drpa_lr: Option<f64>,
    /// Naive sum E_MP2\[erfc\] + E_dRPA\[erf\] (formulation A, diagnostic, DeltaLr only).
    /// Missing the 2·v_sr·v_lr cross-range correlation; reported to make that visible.
    pub e_corr_naive: Option<f64>,
    /// ΔdRPA\[Coulomb\] = E_dRPA\[Coulomb\] − 2·E_OS\[Coulomb\] (CoupledRings only).
    pub e_delta_drpa_full: Option<f64>,
    /// ΔdRPA\[erfc\] = E_dRPA\[erfc\] − 2·E_OS\[erfc\] (CoupledRings only).
    /// Subtracted from ΔdRPA\[Coulomb\] to exclude pure-SR rings beyond 2nd order.
    pub e_delta_drpa_sr: Option<f64>,
    /// Correlation energy of the selected formulation.
    pub e_corr: f64,
    /// Reference SCF energy plus `e_corr`, Ha.
    pub total_energy: f64,
}

impl std::fmt::Display for RsMp2RpaResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RS-MP2+RPA total: {:.10} Ha (corr: {:.10})",
            self.total_energy, self.e_corr
        )
    }
}

/// Preflight estimate (bytes) for [`rs_mp2_lr_rpa_scan`] under `budget_bytes`,
/// with the frequency scratch at full width.
///
/// The driver builds one `RpaIntermediates` at a time and drops it before the
/// next is built: the Coulomb set (MP2 spin components, plus the Coulomb dRPA
/// for formulation T) first, then per ω the erf set and the erfc set in turn.
/// Inside each set the AO tensor is dropped when `b_ov` is built, before the
/// spin components and the dRPA run. The estimate is therefore the largest
/// phase of one set ([`crate::budget::scan_charge`]), with the AO tensor at
/// the tier the build takes under the budget left for it, for any number of
/// ω points. `n_omega` is taken so that independence is stated (and tested)
/// rather than implied.
pub fn scan_peak_bytes(
    shape: crate::budget::PeakEstimateShape,
    n_omega: usize,
    budget_bytes: usize,
) -> usize {
    let _ = n_omega;
    let nov = shape.nocc.saturating_mul(shape.nvir);
    let stream = ferric_mp2::rimp2::mo_stream_scratch_bytes(shape.naux, nov);
    crate::budget::scan_charge(shape, budget_bytes, nov, stream).total
}

/// The preflight shape and [`crate::budget::ScanCharge`] of an rs-mp2-rpa run
/// of `cfg` under its resolved budget, as the driver computes them.
///
/// The packed-tier test reads the shared pool, and the driver plans after it
/// has reserved the persistent planes; with no pool installed the two agree.
pub fn scan_charge_for(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    cfg: &RsMp2RpaConfig,
) -> Result<(crate::budget::PeakEstimateShape, crate::budget::ScanCharge), FerricError> {
    let shape = scan_shape(mol, obs, dfbs, cfg)?;
    let budget = ferric_core::memory::resolve_budget_bytes(cfg.drpa.memory_budget_bytes);
    Ok((shape, charge_of(shape, cfg, budget)))
}

/// The closed-shell preflight shape (the formula `compute_rpa_intermediates`
/// uses for nocc/nvir). Cheap shape values only.
fn scan_shape(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    cfg: &RsMp2RpaConfig,
) -> Result<crate::budget::PeakEstimateShape, FerricError> {
    let naux = dfbs.nbasis();
    let nbas = obs.nbasis();
    let nocc_total = (mol.nelec() as usize) / 2;
    let nocc = ferric_mp2::rimp2::active_occ(nocc_total, cfg.frozen_core)?;
    Ok(crate::budget::PeakEstimateShape {
        naux,
        nocc,
        nvir: nbas.saturating_sub(nocc_total),
        n_quad: cfg.drpa.quadrature.n_points,
        n_workers: rayon::current_num_threads().max(1),
        // trunc_thresh is unknown pre-eigensolve; the default is 0.0 (keep-all).
        n_keep: naux,
        grid: None,
        nao: nbas,
        // rs-mp2-rpa consumes the RPA correlation energy only; it never asks
        // for inv_dielectric_freq, so the per-frequency stack is built,
        // consumed and dropped rather than retained.
        need_inv_dielectric: false,
    })
}

/// [`crate::budget::scan_charge`] with the panel width the frequency loop
/// will allocate under `cfg`'s budget and the unbudgeted stream chunk that
/// `compute_rpa_intermediates` uses.
///
/// With `cfg.attenuator == Terf` the charge is
/// [`crate::budget::with_signed_lr`]: the Coulomb `b_ov` stays resident and
/// the long-range dRPA is the signed log-det.
fn charge_of(
    shape: crate::budget::PeakEstimateShape,
    cfg: &RsMp2RpaConfig,
    budget_bytes: usize,
) -> crate::budget::ScanCharge {
    let nov = shape.nocc.saturating_mul(shape.nvir);
    let panel_width = crate::energy::quad_panel_width(
        shape.n_keep,
        nov,
        shape.n_workers,
        cfg.drpa.memory_budget_bytes,
    );
    let stream = ferric_mp2::rimp2::mo_stream_scratch_bytes(shape.naux, nov);
    let base = crate::budget::scan_charge(shape, budget_bytes, panel_width, stream);
    if cfg.attenuator != Attenuator::Terf {
        return base;
    }
    crate::budget::with_signed_lr(
        base,
        (shape.nao, shape.naux),
        budget_bytes,
        held_b_ov_bytes(&[shape]),
        signed_drpa_bytes(&[shape], cfg),
    )
}

/// The Coulomb `b_ov` planes a terf scan keeps resident, one per shape.
fn held_b_ov_bytes(shapes: &[crate::budget::PeakEstimateShape]) -> usize {
    shapes
        .iter()
        .map(|s| s.naux.saturating_mul(s.nocc).saturating_mul(s.nvir))
        .fold(0usize, |a, b| a.saturating_add(b))
        .saturating_mul(8)
}

/// The signed log-det scratch at the panel width the loop will take.
fn signed_drpa_bytes(shapes: &[crate::budget::PeakEstimateShape], cfg: &RsMp2RpaConfig) -> usize {
    let Some(first) = shapes.first() else {
        return 0;
    };
    let nov = shapes
        .iter()
        .map(|s| s.nocc.saturating_mul(s.nvir))
        .max()
        .unwrap_or(0);
    let width = crate::energy::quad_panel_width(
        2 * first.naux,
        nov,
        first.n_workers,
        cfg.drpa.memory_budget_bytes,
    );
    crate::budget::signed_quad_scratch_bytes(first.naux, width, first.n_workers)
}

/// The terf long-range dRPA is the signed log-det
/// ([`crate::energy::signed_drpa_energy`]): it forms no eigenvalues and no
/// static eigenbasis, so the dRPA options that need one have no meaning for it
/// and are refused rather than ignored.
fn refuse_unsupported_signed_drpa(cfg: &RsMp2RpaConfig, mol: &Molecule) -> Result<(), FerricError> {
    let why = if cfg.drpa.trunc_thresh > 0.0 {
        Some(format!(
            "trunc_thresh = {} (PDEP truncation)",
            cfg.drpa.trunc_thresh
        ))
    } else if !matches!(cfg.drpa.chi0_backend, crate::config::Chi0Backend::Dense) {
        Some(format!("chi0_backend = {:?}", cfg.drpa.chi0_backend))
    } else if let crate::Chi0Sparsity::BoysScreened { .. } =
        cfg.drpa.chi0_sparsity.resolve(mol.atoms.len())
    {
        Some(format!(
            "chi0_sparsity = {:?} (the Boys-screened chi0 for this {}-atom molecule)",
            cfg.drpa.chi0_sparsity,
            mol.atoms.len()
        ))
    } else if cfg.drpa.need_inv_dielectric_freq {
        Some("need_inv_dielectric_freq = true".to_string())
    } else {
        None
    };
    match why {
        None => Ok(()),
        Some(w) => Err(FerricError::unsupported(
            format!("rs-mp2-rpa terf, formulation B, with {w}"),
            "the long-range dRPA is the signed log-det of Coulomb RI minus terfc RI (the terf \
             kernel is not positive definite), which forms no static eigenbasis. Use \
             trunc_thresh = 0, chi0_backend = dense and chi0_sparsity = dense",
        )),
    }
}

/// [`rs_mp2_lr_rpa`] on the system `aux` was prepared on (see [`PreparedAux`]):
/// `rhf` is checked against that system first
/// ([`PreparedAux::check_reference`]).
/// A system with no virtual orbitals is refused
/// ([`PreparedAux::check_virtual_orbitals`]).
pub fn rs_mp2_lr_rpa_on(
    aux: &PreparedAux<'_>,
    rhf: &ScfResult,
    cfg: &RsMp2RpaConfig,
) -> Result<RsMp2RpaResult, FerricError> {
    let sys = aux.system();
    aux.check_reference(rhf)?;
    aux.check_virtual_orbitals("rs_mp2_lr_rpa_on", rhf)?;
    rs_mp2_lr_rpa(sys.mol(), sys.basis(), aux.basis(), rhf, cfg)
}

/// SR-MP2 + LR-RPA, Δ-form (B) or coupled-rings (T).
///
/// **DeltaLr (B)**: replaces MP2's long-range direct ring series with its dRPA\[erf\]
/// resummation. Exact limits: ω→0 ⇒ plain MP2; ω→∞ ⇒ MP2 + (dRPA − dMP2).
///
/// **CoupledRings (T)**: screens all rings (ΔdRPA\[Coulomb\]) then un-screens the
/// pure-SR rings (−ΔdRPA\[erfc\]). Same exact limits; additionally includes all
/// mixed SR×LR ring diagrams. Cost: 2 dRPA calls instead of 1.
///
/// This is the one-point case of [`rs_mp2_lr_rpa_scan`].
pub fn rs_mp2_lr_rpa(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    cfg: &RsMp2RpaConfig,
) -> Result<RsMp2RpaResult, FerricError> {
    let ops = [operators_from_cfg(cfg)];
    let mut out = scan_impl(mol, obs, dfbs, rhf, cfg, &ops, &[cfg.formulation])?;
    out.pop()
        .ok_or_else(|| FerricError::General("rs-mp2-rpa produced no result".into()))
}

/// The (long-range, short-range) operator pair selected by `cfg`.
///
/// `Erf` uses erf(ω)/erfc(ω); `Terf` uses the tempered terf(r0)/terfc(r0)
/// (ω derived from r0 unless `terf_omega` overrides it). terf + terfc =
/// Coulomb exactly, so the telescoping and exact limits are unchanged.
fn operators_from_cfg(cfg: &RsMp2RpaConfig) -> (Operator, Operator) {
    match cfg.attenuator {
        Attenuator::Erf => (Operator::erf(cfg.omega), Operator::erfc(cfg.omega)),
        Attenuator::Terf => match cfg.terf_omega {
            Some(w) => (
                Operator::terf_with_omega(cfg.r0, w),
                Operator::terfc_with_omega(cfg.r0, w),
            ),
            None => (Operator::terf(cfg.r0), Operator::terfc(cfg.r0)),
        },
    }
}

/// [`rs_mp2_lr_rpa_scan`] on the system `aux` was prepared on (see [`PreparedAux`]):
/// `rhf` is checked against that system first
/// ([`PreparedAux::check_reference`]).
/// A system with no virtual orbitals is refused
/// ([`PreparedAux::check_virtual_orbitals`]).
pub fn rs_mp2_lr_rpa_scan_on(
    aux: &PreparedAux<'_>,
    rhf: &ScfResult,
    cfg: &RsMp2RpaConfig,
    omegas: &[f64],
    formulations: &[RsMp2RpaFormulation],
) -> Result<Vec<RsMp2RpaResult>, FerricError> {
    let sys = aux.system();
    aux.check_reference(rhf)?;
    aux.check_virtual_orbitals("rs_mp2_lr_rpa_scan_on", rhf)?;
    rs_mp2_lr_rpa_scan(
        sys.mol(),
        sys.basis(),
        aux.basis(),
        rhf,
        cfg,
        omegas,
        formulations,
    )
}

/// SR-MP2 + LR-RPA at several ω values and formulations in one run.
///
/// Returns one [`RsMp2RpaResult`] per (ω, formulation), ω-major in the order
/// given (`out[i * formulations.len() + j]` is `omegas[i]` with
/// `formulations[j]`). Each entry is bit-identical to a separate
/// [`rs_mp2_lr_rpa`] call with that ω and formulation.
///
/// `omegas` are Bohr⁻¹. With `cfg.attenuator == Erf` each is the erf/erfc
/// range-separation parameter and `cfg.omega` is ignored. With `Terf`, `cfg.r0`
/// is fixed and each value is the independent sharpness override
/// (`terf_omega`); `cfg.terf_omega` and `cfg.omega` are ignored. `cfg.formulation`
/// is ignored in favour of `formulations`.
///
/// What is built once: the Coulomb RI intermediates and MP2 spin components,
/// and (when T is requested) the Coulomb dRPA, none of which depend on ω. Per
/// ω: the erf and erfc intermediates, shared by B and T; B's erf dRPA and T's
/// erfc dRPA. Each intermediate is dropped before the next is built, so peak
/// memory does not grow with `omegas.len()` (see [`scan_peak_bytes`]). The
/// reference SCF is the caller's, so it is also done once.
///
/// With `Terf` the Coulomb intermediates stay resident for the whole scan and
/// only the terfc set is built per ω: the long range is Coulomb RI minus terfc
/// RI ([`Attenuator::Terf`]). [`scan_peak_bytes`] does not include that
/// held plane; [`scan_charge_for`] does.
///
/// # Errors
/// Empty `omegas` or `formulations`, a non-finite or non-positive ω, or a
/// repeated formulation are [`FerricError::General`].
pub fn rs_mp2_lr_rpa_scan(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    cfg: &RsMp2RpaConfig,
    omegas: &[f64],
    formulations: &[RsMp2RpaFormulation],
) -> Result<Vec<RsMp2RpaResult>, FerricError> {
    if omegas.is_empty() {
        return Err(FerricError::General(
            "rs-mp2-rpa scan needs at least one omega".into(),
        ));
    }
    if let Some(w) = omegas.iter().find(|w| !w.is_finite() || **w <= 0.0) {
        return Err(FerricError::General(format!(
            "rs-mp2-rpa scan omega must be finite and > 0 (got {w})"
        )));
    }
    let ops: Vec<(Operator, Operator)> = omegas
        .iter()
        .map(|&w| match cfg.attenuator {
            Attenuator::Erf => (Operator::erf(w), Operator::erfc(w)),
            Attenuator::Terf => (
                Operator::terf_with_omega(cfg.r0, w),
                Operator::terfc_with_omega(cfg.r0, w),
            ),
        })
        .collect();
    scan_impl(mol, obs, dfbs, rhf, cfg, &ops, formulations)
}

/// Shared driver: one result per (operator pair, formulation), pair-major.
fn scan_impl(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    cfg: &RsMp2RpaConfig,
    ops: &[(Operator, Operator)],
    formulations: &[RsMp2RpaFormulation],
) -> Result<Vec<RsMp2RpaResult>, FerricError> {
    if formulations.is_empty() {
        return Err(FerricError::General(
            "rs-mp2-rpa needs at least one formulation".into(),
        ));
    }
    for (i, f) in formulations.iter().enumerate() {
        if formulations[..i].contains(f) {
            return Err(FerricError::General(format!(
                "rs-mp2-rpa formulation {f:?} listed twice"
            )));
        }
    }
    if !matches!(rhf.spin, ferric_scf::Spin::Restricted) {
        return scan_impl_unrestricted(mol, obs, dfbs, rhf, cfg, ops, formulations);
    }
    let want_b = formulations.contains(&RsMp2RpaFormulation::DeltaLr);
    let want_t = formulations.contains(&RsMp2RpaFormulation::CoupledRings);
    let signed = signed_lr(ops, cfg)?;
    if signed && want_b {
        refuse_unsupported_signed_drpa(cfg, mol)?;
    }

    // Resolved ONCE: the preflight budget check and the stage-seam RSS
    // checks compare against the same configured ceiling.
    let resolved_budget_bytes =
        ferric_core::memory::resolve_budget_bytes(cfg.drpa.memory_budget_bytes);

    // Pre-flight peak-memory gate (see `budget::scan_charge`). Cheap shape
    // values only, so this runs before ANY large allocation.
    //
    // Phases. Each operator's set runs build -> spin -> dRPA on top of the
    // persistent planes (metric + one b_ov), and the AO tensor is dropped
    // when b_ov is built. The ceiling check charges the largest phase, with
    // the AO tensor at the tier `ThreeIndexSource::build` takes under the
    // budget left after the persistent planes and the stream chunk. That
    // budget is handed to the build (`ri_cfg` below), so the tier charged is
    // the tier taken: in core, packed, or spilled.
    //
    // Pool. The persistent planes are reserved for the whole call, and first,
    // because the packed-tier test reads the pool's free bytes. The AO
    // tensor is charged by `ThreeIndexSource` itself for as long as it lives.
    // The dRPA planes are reserved around each dRPA call and the spin scratch
    // around each spin-component call, so no phase holds another's bytes.
    let shape = scan_shape(mol, obs, dfbs, cfg)?;
    let label = format!(
        "RS-MP2-RPA preflight (naux={}, nocc={}, nvir={}, n_workers={}, points={}, \
         formulations={formulations:?})",
        shape.naux,
        shape.nocc,
        shape.nvir,
        shape.n_workers,
        ops.len()
    );
    let held = if signed { held_b_ov_bytes(&[shape]) } else { 0 };
    let _persistent = ferric_core::memory::pool::reserve_global(
        &format!("{label} [metric + b_ov]"),
        crate::budget::scan_persistent_bytes(shape).saturating_add(held),
    )?;
    let charge = charge_of(shape, cfg, resolved_budget_bytes);
    // Decides admissibility on the unbudgeted path too, where the pool is
    // inert.
    ferric_core::memory::check_alloc(
        &format!("{label} (AO tensor tier {})", charge.ao_tier.name()),
        charge.total,
        resolved_budget_bytes,
    )?;

    let ri_cfg = RiMp2Config {
        frozen_core: cfg.frozen_core,
        // The AO tier's budget; never 0, which would read as unset.
        memory_budget_bytes: Some(charge.ao_budget.max(1)),
        ..Default::default()
    };

    // SHARED-INTERMEDIATE FUSION. The MP2 spin components and the dRPA solves
    // both need the dressed b_ov = V^{-1/2}(P|op|ia) for the SAME operator;
    // building it is the expensive step. Each operator's intermediates are
    // built once and feed both `spin_components_from_b_ov` and
    // `run_pdep_rpa_from_intermediates`.
    let eps = rhf.eps_r();
    let inter_of = |op: Operator| -> Result<ferric_mp2::rimp2::RpaIntermediates, FerricError> {
        let it = compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &ri_cfg)?;
        // Stage-seam RSS observability: a stderr line, never an error.
        ferric_core::memory::warn_if_rss_over(
            &format!("RS-MP2-RPA RpaIntermediates build (op={op:?})"),
            resolved_budget_bytes,
            1.1,
        );
        Ok(it)
    };
    let sc_of = |it: &ferric_mp2::rimp2::RpaIntermediates| {
        // Soft: the per-worker transient runs at whatever width is active.
        let _spin = ferric_core::memory::pool::try_reserve_global(
            &format!("{label} [spin g_i]"),
            charge.spin_scratch,
        );
        spin_components_from_b_ov(&it.b_ov, eps, it.nocc, it.nvir, it.first_occ, it.nocc_total)
    };

    let mut drpa_cfg = cfg.drpa.clone();
    drpa_cfg.frozen_core = cfg.frozen_core;
    let drpa_of =
        |it: &ferric_mp2::rimp2::RpaIntermediates, op: Operator| -> Result<f64, FerricError> {
            let _hard = ferric_core::memory::pool::reserve_global(
                &format!("{label} [dRPA eigensolve]"),
                charge.drpa_hard,
            )?;
            // Soft: `energy::quad_panel_width` narrows the loop under a budget.
            let _soft = ferric_core::memory::pool::try_reserve_global(
                &format!("{label} [dRPA freq scratch]"),
                charge.drpa_soft,
            );
            Ok(run_pdep_rpa_from_intermediates(it, mol, obs, dfbs, op, rhf, &drpa_cfg)?.e_rpa)
        };

    // The terf long range: the signed factor [B_C; B_terfc] with signature
    // (+, -) (`SignedBOv`); terf is never fitted in its own indefinite metric.
    let signed_sc_of = |pos: &ferric_mp2::rimp2::RpaIntermediates,
                        neg: &ferric_mp2::rimp2::RpaIntermediates| {
        let _spin = ferric_core::memory::pool::try_reserve_global(
            &format!("{label} [spin g_i]"),
            charge.spin_scratch,
        );
        spin_components_from_signed_b_ov(
            SignedBOv {
                pos: &pos.b_ov,
                neg: &neg.b_ov,
            },
            eps,
            pos.nocc,
            pos.nvir,
            pos.first_occ,
            pos.nocc_total,
        )
    };
    let signed_drpa_of = |pos: &ferric_mp2::rimp2::RpaIntermediates,
                          neg: &ferric_mp2::rimp2::RpaIntermediates|
     -> Result<f64, FerricError> {
        let _soft = ferric_core::memory::pool::try_reserve_global(
            &format!("{label} [signed dRPA freq scratch]"),
            charge.drpa_soft,
        );
        let channel = crate::energy::SignedChannel {
            pos: &pos.b_ov,
            neg: &neg.b_ov,
            eps_occ: &eps[pos.first_occ..pos.first_occ + pos.nocc],
            eps_vir: &eps[pos.nocc_total..pos.nocc_total + pos.nvir],
            prefactor: 4.0,
        };
        crate::energy::signed_drpa_energy(
            &[channel],
            &cfg.drpa.quadrature,
            cfg.drpa.memory_budget_bytes,
        )
    };

    // ω-independent stage: Coulomb spin components, and for T the Coulomb dRPA.
    // `it_full` is dropped at the end of this block, unless the terf long
    // range needs it for every point.
    let (sc_full, drpa_coul_e, held_full) = {
        let it_full = inter_of(Operator::coulomb())?;
        let sc_full = sc_of(&it_full);
        let drpa_coul_e = if want_t {
            Some(drpa_of(&it_full, Operator::coulomb())?)
        } else {
            None
        };
        (sc_full, drpa_coul_e, signed.then_some(it_full))
    };

    let mut out = Vec::with_capacity(ops.len() * formulations.len());
    for &(op_lr, op_sr) in ops {
        if let Some(full) = held_full.as_ref() {
            // terf: one terfc build per point feeds the LR (signed, with the
            // held Coulomb factor) and the SR numbers.
            let it_sr = inter_of(op_sr)?;
            let sc_lr = signed_sc_of(full, &it_sr);
            let drpa_lr_e = if want_b {
                Some(signed_drpa_of(full, &it_sr)?)
            } else {
                None
            };
            let sc_sr = sc_of(&it_sr);
            let drpa_sr_e = if want_t {
                Some(drpa_of(&it_sr, op_sr)?)
            } else {
                None
            };
            let rings = Rings {
                full: closed_shell_parts(&sc_full),
                lr: closed_shell_parts(&sc_lr),
                sr: closed_shell_parts(&sc_sr),
                drpa_lr: drpa_lr_e,
                drpa_coulomb: drpa_coul_e,
                drpa_sr: drpa_sr_e,
            };
            for &formulation in formulations {
                out.push(rings.assemble(formulation, rhf.energy));
            }
            continue;
        }
        // LR: spin components always (e_dmp2_lr); the LR dRPA only for B.
        // op_lr is erf(ω) for Erf, terf(r0) for Terf, and the LR dRPA must use
        // that same operator.
        let (sc_lr, drpa_lr_e) = {
            let it_lr = inter_of(op_lr)?;
            let sc_lr = sc_of(&it_lr);
            let e = if want_b {
                Some(drpa_of(&it_lr, op_lr)?)
            } else {
                None
            };
            (sc_lr, e)
        };
        // SR: spin components always; the SR dRPA only for T.
        let (sc_sr, drpa_sr_e) = {
            let it_sr = inter_of(op_sr)?;
            let sc_sr = sc_of(&it_sr);
            let e = if want_t {
                Some(drpa_of(&it_sr, op_sr)?)
            } else {
                None
            };
            (sc_sr, e)
        };
        let rings = Rings {
            full: closed_shell_parts(&sc_full),
            lr: closed_shell_parts(&sc_lr),
            sr: closed_shell_parts(&sc_sr),
            drpa_lr: drpa_lr_e,
            drpa_coulomb: drpa_coul_e,
            drpa_sr: drpa_sr_e,
        };
        for &formulation in formulations {
            out.push(rings.assemble(formulation, rhf.energy));
        }
    }
    Ok(out)
}

/// Closed shell: E_MP2 = Σ (ia|jb)[2(ia|jb)−(ib|ja)]/Δ; the direct (ring)
/// part 2Σ(ia|jb)²/Δ — the 2nd-order truncation of dRPA — equals 2·E_OS.
fn closed_shell_parts(sc: &ferric_mp2::rimp2::SpinComponents) -> OpParts {
    OpParts {
        e_total: sc.e_total,
        dmp2: 2.0 * sc.e_os,
    }
}

/// Whether the long-range operators of `ops` are terf, which the scan forms as
/// Coulomb RI minus terfc RI. All points share one attenuator.
fn signed_lr(ops: &[(Operator, Operator)], cfg: &RsMp2RpaConfig) -> Result<bool, FerricError> {
    use ferric_integrals::operator::OperatorKind;
    let terf = |op: &Operator| matches!(op.kind, OperatorKind::Terf);
    let n = ops.iter().filter(|(lr, _)| terf(lr)).count();
    if n != 0 && n != ops.len() {
        return Err(FerricError::General(
            "rs-mp2-rpa: a scan mixes terf and non-terf long-range operators".into(),
        ));
    }
    let signed = n != 0;
    if signed != (cfg.attenuator == Attenuator::Terf) {
        return Err(FerricError::General(format!(
            "rs-mp2-rpa: the operators do not match the {:?} attenuator",
            cfg.attenuator
        )));
    }
    Ok(signed)
}

/// One operator's MP2 numbers as the assembly reads them: the MP2
/// correlation energy and its direct (ring) second-order part `dMP2`, the
/// second-order term of the dRPA under the same operator.
///
/// Closed shell `dMP2 = 2·E_OS`. Open shell
/// `dMP2 = E_αβ + ½(D_αα + D_ββ)` with `D_σσ = Σ_{ijab∈σ} (ia|jb)²/Δ` (the
/// same-spin direct sum, no exchange); `2·E_αβ` would be wrong whenever the
/// two spins differ (`book/src/26-rs-mp2-rpa/04-two-formulations.md#two-formulations`).
#[derive(Debug, Clone, Copy)]
struct OpParts {
    e_total: f64,
    dmp2: f64,
}

/// Every number one (ω, formulation) result is assembled from.
#[derive(Debug, Clone, Copy)]
struct Rings {
    full: OpParts,
    lr: OpParts,
    sr: OpParts,
    drpa_lr: Option<f64>,
    drpa_coulomb: Option<f64>,
    drpa_sr: Option<f64>,
}

impl Rings {
    /// B: `E_MP2[C] + E_dRPA[lr] − dMP2[lr]`;
    /// T: `E_MP2[C] + (E_dRPA[C] − dMP2[C]) − (E_dRPA[sr] − dMP2[sr])`.
    fn assemble(&self, formulation: RsMp2RpaFormulation, e_ref: f64) -> RsMp2RpaResult {
        let (e_corr, e_drpa_lr, e_corr_naive, e_delta_drpa_full, e_delta_drpa_sr) =
            match formulation {
                RsMp2RpaFormulation::DeltaLr => {
                    let e_drpa = self.drpa_lr.expect("B requested => LR dRPA computed");
                    (
                        self.full.e_total + e_drpa - self.lr.dmp2,
                        Some(e_drpa),
                        Some(self.sr.e_total + e_drpa),
                        None,
                        None,
                    )
                }
                RsMp2RpaFormulation::CoupledRings => {
                    let e_coul = self
                        .drpa_coulomb
                        .expect("T requested => Coulomb dRPA computed");
                    let e_sr = self.drpa_sr.expect("T requested => SR dRPA computed");
                    let delta_full = e_coul - self.full.dmp2;
                    let delta_sr = e_sr - self.sr.dmp2;
                    (
                        self.full.e_total + delta_full - delta_sr,
                        None,
                        None,
                        Some(delta_full),
                        Some(delta_sr),
                    )
                }
            };
        RsMp2RpaResult {
            e_mp2_full: self.full.e_total,
            e_sr_mp2: self.sr.e_total,
            e_lr_mp2: self.lr.e_total,
            e_dmp2_lr: self.lr.dmp2,
            e_drpa_lr,
            e_corr_naive,
            e_delta_drpa_full,
            e_delta_drpa_sr,
            e_corr,
            total_energy: e_ref + e_corr,
        }
    }
}

/// One operator's open-shell [`OpParts`]: `E_MP2 = E_αα + E_ββ + E_αβ`
/// (`u_ri_mp2_from_parts`) and `dMP2 = E_αβ + ½(D_αα + D_ββ)`.
///
/// `D_σσ = Σ_{ijab∈σ} (ia|jb)²/Δ` is the `e_os` of the closed-shell kernel
/// run on spin σ's `b_ov` and energies; an empty channel contributes 0.
fn open_shell_parts(
    a: &ferric_mp2::rimp2::RpaIntermediates,
    b: &ferric_mp2::rimp2::RpaIntermediates,
    eps_a: &[f64],
    eps_b: &[f64],
) -> Result<OpParts, FerricError> {
    let direct_same_spin = |it: &ferric_mp2::rimp2::RpaIntermediates, eps: &[f64]| -> f64 {
        if it.nocc == 0 || it.nvir == 0 {
            return 0.0;
        }
        spin_components_from_b_ov(&it.b_ov, eps, it.nocc, it.nvir, it.first_occ, it.nocc_total).e_os
    };
    let c = ferric_mp2::u_rimp2::u_ri_mp2_from_parts(a, b, eps_a, eps_b)?;
    let d_aa = direct_same_spin(a, eps_a);
    let d_bb = direct_same_spin(b, eps_b);
    Ok(OpParts {
        e_total: c.e_total,
        dmp2: c.e_ab + 0.5 * (d_aa + d_bb),
    })
}

/// [`open_shell_parts`] of the signed factors `(pos_σ, neg_σ)` (Coulomb RI
/// minus terfc RI per spin): `E_MP2` from
/// `ferric_mp2::u_rimp2::u_ri_mp2_signed_from_parts` and `D_σσ` from the
/// signed closed-shell kernel on spin σ.
fn open_shell_signed_parts(
    (pa, na): (
        &ferric_mp2::rimp2::RpaIntermediates,
        &ferric_mp2::rimp2::RpaIntermediates,
    ),
    (pb, nb): (
        &ferric_mp2::rimp2::RpaIntermediates,
        &ferric_mp2::rimp2::RpaIntermediates,
    ),
    eps_a: &[f64],
    eps_b: &[f64],
) -> Result<OpParts, FerricError> {
    let direct_same_spin = |p: &ferric_mp2::rimp2::RpaIntermediates,
                            n: &ferric_mp2::rimp2::RpaIntermediates,
                            eps: &[f64]|
     -> f64 {
        if p.nocc == 0 || p.nvir == 0 {
            return 0.0;
        }
        spin_components_from_signed_b_ov(
            SignedBOv {
                pos: &p.b_ov,
                neg: &n.b_ov,
            },
            eps,
            p.nocc,
            p.nvir,
            p.first_occ,
            p.nocc_total,
        )
        .e_os
    };
    let c = ferric_mp2::u_rimp2::u_ri_mp2_signed_from_parts((pa, na), (pb, nb), eps_a, eps_b)?;
    let d_aa = direct_same_spin(pa, na, eps_a);
    let d_bb = direct_same_spin(pb, nb, eps_b);
    Ok(OpParts {
        e_total: c.e_total,
        dmp2: c.e_ab + 0.5 * (d_aa + d_bb),
    })
}

/// The open-shell preflight shapes `(alpha, beta)`: the occupation counts
/// `compute_rpa_intermediates_spin` uses (`nocc_σ = (N ± 2S)/2`).
fn scan_shapes_unrestricted(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    cfg: &RsMp2RpaConfig,
) -> Result<
    (
        crate::budget::PeakEstimateShape,
        crate::budget::PeakEstimateShape,
    ),
    FerricError,
> {
    let closed = scan_shape(mol, obs, dfbs, cfg)?;
    let nbas = obs.nbasis();
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    let shape_for = |nocc_total: usize| -> Result<_, FerricError> {
        Ok(crate::budget::PeakEstimateShape {
            nocc: ferric_mp2::rimp2::active_occ(nocc_total, cfg.frozen_core)?,
            nvir: nbas.saturating_sub(nocc_total),
            ..closed
        })
    };
    let a = shape_for(((nelec + two_s) / 2) as usize)?;
    // The AO tensor is spin-independent and charged once, on alpha.
    let b = crate::budget::PeakEstimateShape {
        nao: 0,
        ..shape_for(((nelec - two_s) / 2) as usize)?
    };
    Ok((a, b))
}

/// [`crate::budget::scan_charge_unrestricted`] for `cfg` under `budget_bytes`.
fn charge_of_unrestricted(
    (a, b): (
        crate::budget::PeakEstimateShape,
        crate::budget::PeakEstimateShape,
    ),
    cfg: &RsMp2RpaConfig,
    budget_bytes: usize,
) -> crate::budget::ScanCharge {
    let panel = |s: crate::budget::PeakEstimateShape| {
        crate::energy::quad_panel_width(
            s.n_keep,
            s.nocc.saturating_mul(s.nvir),
            s.n_workers,
            cfg.drpa.memory_budget_bytes,
        )
    };
    let stream = |s: crate::budget::PeakEstimateShape| {
        ferric_mp2::rimp2::mo_stream_scratch_bytes(s.naux, s.nocc.saturating_mul(s.nvir))
    };
    let base = crate::budget::scan_charge_unrestricted(
        (a, b),
        budget_bytes,
        (panel(a), panel(b)),
        stream(a).max(stream(b)),
    );
    if cfg.attenuator != Attenuator::Terf {
        return base;
    }
    crate::budget::with_signed_lr(
        base,
        (a.nao.max(b.nao), a.naux),
        budget_bytes,
        held_b_ov_bytes(&[a, b]),
        signed_drpa_bytes(&[a, b], cfg),
    )
}

/// Open-shell (UHF, or ROHF semi-canonicalized per spin) driver of
/// [`scan_impl`].
///
/// Per operator one pair of spin intermediates `B_σ = V^{-1/2}(P|op|ia)_σ`
/// feeds the U-MP2 blocks (`u_ri_mp2_from_parts`: `E_αα`, `E_ββ`, `E_αβ`),
/// the same-spin direct sums `D_σσ` and the spin-summed dRPA
/// (`ε̃ = I + Π_α + Π_β`, [`crate::u_pdep_rpa_from_intermediates`]). The pair
/// is dropped before the next operator's is built. The assembly is the
/// closed-shell one with `dMP2 = E_αβ + ½(D_αα + D_ββ)`.
///
/// Memory: the closed-shell charge model ([`crate::budget::scan_charge`]),
/// extended to two co-resident spin sets
/// ([`crate::budget::scan_charge_unrestricted`]).
fn scan_impl_unrestricted(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    cfg: &RsMp2RpaConfig,
    ops: &[(Operator, Operator)],
    formulations: &[RsMp2RpaFormulation],
) -> Result<Vec<RsMp2RpaResult>, FerricError> {
    use ferric_mp2::rimp2::{compute_rpa_intermediates_spin, RpaIntermediates};
    let want_b = formulations.contains(&RsMp2RpaFormulation::DeltaLr);
    let want_t = formulations.contains(&RsMp2RpaFormulation::CoupledRings);
    let signed = signed_lr(ops, cfg)?;
    if signed && want_b {
        refuse_unsupported_signed_drpa(cfg, mol)?;
    }
    // The screened chi0 is closed-shell only (run_u_pdep_rpa refuses it too).
    if let crate::Chi0Sparsity::BoysScreened { .. } =
        cfg.drpa.chi0_sparsity.resolve(mol.atoms.len())
    {
        return Err(FerricError::General(format!(
            "rs-mp2-rpa: chi0_sparsity {:?} resolves to the Boys-screened chi0 for this \
             {}-atom molecule, which has no open-shell version; use chi0_sparsity = \"dense\"",
            cfg.drpa.chi0_sparsity,
            mol.atoms.len()
        )));
    }
    let view = ferric_scf::semicanonical::unrestricted_reference(mol, rhf)?;
    let uhf: &ScfResult = &view;
    let (eps_a, eps_b) = (uhf.eps_a(), uhf.eps_b());

    let resolved_budget_bytes =
        ferric_core::memory::resolve_budget_bytes(cfg.drpa.memory_budget_bytes);
    let shapes = scan_shapes_unrestricted(mol, obs, dfbs, cfg)?;
    let label = format!(
        "U-RS-MP2-RPA preflight (naux={}, nocc_a={}, nvir_a={}, nocc_b={}, nvir_b={}, \
         n_workers={}, points={}, formulations={formulations:?})",
        shapes.0.naux,
        shapes.0.nocc,
        shapes.0.nvir,
        shapes.1.nocc,
        shapes.1.nvir,
        shapes.0.n_workers,
        ops.len()
    );
    let charge = charge_of_unrestricted(shapes, cfg, resolved_budget_bytes);
    let _persistent = ferric_core::memory::pool::reserve_global(
        &format!("{label} [metric + b_ov, both spins]"),
        charge.persistent,
    )?;
    ferric_core::memory::check_alloc(
        &format!("{label} (AO tensor tier {})", charge.ao_tier.name()),
        charge.total,
        resolved_budget_bytes,
    )?;
    let ri_cfg = RiMp2Config {
        frozen_core: cfg.frozen_core,
        memory_budget_bytes: Some(charge.ao_budget.max(1)),
        ..Default::default()
    };

    let pair_of = |op: Operator| -> Result<(RpaIntermediates, RpaIntermediates), FerricError> {
        let a = compute_rpa_intermediates_spin(mol, obs, dfbs, op, uhf, &ri_cfg, true)?;
        let b = compute_rpa_intermediates_spin(mol, obs, dfbs, op, uhf, &ri_cfg, false)?;
        ferric_core::memory::warn_if_rss_over(
            &format!("U-RS-MP2-RPA spin intermediates (op={op:?})"),
            resolved_budget_bytes,
            1.1,
        );
        Ok((a, b))
    };
    let parts_of = |(a, b): &(RpaIntermediates, RpaIntermediates)| -> Result<OpParts, FerricError> {
        let _spin = ferric_core::memory::pool::try_reserve_global(
            &format!("{label} [spin g_i]"),
            charge.spin_scratch,
        );
        open_shell_parts(a, b, eps_a, eps_b)
    };
    let mut drpa_cfg = cfg.drpa.clone();
    drpa_cfg.frozen_core = cfg.frozen_core;
    let drpa_of = |(a, b): &(RpaIntermediates, RpaIntermediates)| -> Result<f64, FerricError> {
        let _hard = ferric_core::memory::pool::reserve_global(
            &format!("{label} [dRPA eigensolve]"),
            charge.drpa_hard,
        )?;
        let _soft = ferric_core::memory::pool::try_reserve_global(
            &format!("{label} [dRPA freq scratch]"),
            charge.drpa_soft,
        );
        Ok(crate::u_pdep_rpa_from_intermediates(a, b, uhf, &drpa_cfg)?.e_rpa)
    };

    // terf long range: signed factors [B_C,σ; B_terfc,σ], signature (+, -).
    let signed_parts_of = |(pa, pb): &(RpaIntermediates, RpaIntermediates),
                           (na, nb): &(RpaIntermediates, RpaIntermediates)|
     -> Result<OpParts, FerricError> {
        let _spin = ferric_core::memory::pool::try_reserve_global(
            &format!("{label} [spin g_i]"),
            charge.spin_scratch,
        );
        open_shell_signed_parts((pa, na), (pb, nb), eps_a, eps_b)
    };
    let signed_drpa_of = |(pa, pb): &(RpaIntermediates, RpaIntermediates),
                          (na, nb): &(RpaIntermediates, RpaIntermediates)|
     -> Result<f64, FerricError> {
        let _soft = ferric_core::memory::pool::try_reserve_global(
            &format!("{label} [signed dRPA freq scratch]"),
            charge.drpa_soft,
        );
        fn channel<'a>(
            p: &'a RpaIntermediates,
            n: &'a RpaIntermediates,
            eps: &'a [f64],
        ) -> crate::energy::SignedChannel<'a> {
            crate::energy::SignedChannel {
                pos: &p.b_ov,
                neg: &n.b_ov,
                eps_occ: &eps[p.first_occ..p.first_occ + p.nocc],
                eps_vir: &eps[p.nocc_total..p.nocc_total + p.nvir],
                prefactor: 2.0,
            }
        }
        crate::energy::signed_drpa_energy(
            &[channel(pa, na, eps_a), channel(pb, nb, eps_b)],
            &cfg.drpa.quadrature,
            cfg.drpa.memory_budget_bytes,
        )
    };

    let (full, drpa_coulomb, held_full) = {
        let p = pair_of(Operator::coulomb())?;
        let full = parts_of(&p)?;
        let e = if want_t { Some(drpa_of(&p)?) } else { None };
        (full, e, signed.then_some(p))
    };
    let mut out = Vec::with_capacity(ops.len() * formulations.len());
    for &(op_lr, op_sr) in ops {
        if let Some(fp) = held_full.as_ref() {
            let p = pair_of(op_sr)?;
            let lr = signed_parts_of(fp, &p)?;
            let drpa_lr = if want_b {
                Some(signed_drpa_of(fp, &p)?)
            } else {
                None
            };
            let sr = parts_of(&p)?;
            let drpa_sr = if want_t { Some(drpa_of(&p)?) } else { None };
            let rings = Rings {
                full,
                lr,
                sr,
                drpa_lr,
                drpa_coulomb,
                drpa_sr,
            };
            for &formulation in formulations {
                out.push(rings.assemble(formulation, rhf.energy));
            }
            continue;
        }
        let (lr, drpa_lr) = {
            let p = pair_of(op_lr)?;
            let lr = parts_of(&p)?;
            let e = if want_b { Some(drpa_of(&p)?) } else { None };
            (lr, e)
        };
        let (sr, drpa_sr) = {
            let p = pair_of(op_sr)?;
            let sr = parts_of(&p)?;
            let e = if want_t { Some(drpa_of(&p)?) } else { None };
            (sr, e)
        };
        let rings = Rings {
            full,
            lr,
            sr,
            drpa_lr,
            drpa_coulomb,
            drpa_sr,
        };
        for &formulation in formulations {
            out.push(rings.assemble(formulation, rhf.energy));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    // These reference-cross-check tests call the standalone entry points
    // directly (the production path uses the fused intermediates above).
    use crate::run_pdep_rpa;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_mp2::rimp2::ri_mp2_spin_components;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use ferric_scf::ScfResult;

    fn setup_h2() -> (Molecule, PreparedBasis, PreparedBasis, ScfResult) {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig::default().with_energy_conv(1e-10),
        )
        .unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        (mol, obs, dfbs, rhf)
    }

    /// ω→0: erf(ωr)/r → 0, so dRPA[erf] and dMP2[erf] both vanish and the
    /// Δ-form must reduce to plain RI-MP2. The correction is 4th-order small
    /// in v_lr, so even ω=0.05 must be tight.
    ///
    /// Note: ω=0.05 Bohr⁻¹ is the tested floor — at this value the erf metric
    /// has ≥1 significant eigenvalue (the LR dielectric ε̃ is non-trivially
    /// above identity) and the Δ-correction is <1e-5 Ha from MP2.
    #[test]
    fn omega_to_zero_reduces_to_mp2() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let full = ferric_mp2::rimp2::ri_mp2(
            &mol,
            &obs,
            &dfbs,
            Operator::coulomb(),
            &rhf,
            &ferric_mp2::rimp2::RiMp2Config::default(),
        )
        .unwrap();
        let cfg = RsMp2RpaConfig {
            omega: 0.05,
            ..Default::default()
        };
        let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();
        let drpa_lr = r.e_drpa_lr.unwrap();
        eprintln!(
            "MP2 corr {:.10}  Δ-form corr {:.10}  ΔLR {:.2e}",
            full.mp2_corr,
            r.e_corr,
            drpa_lr - r.e_dmp2_lr
        );
        assert!(
            (r.e_corr - full.mp2_corr).abs() < 1e-5,
            "omega→0 must reduce to MP2: {} vs {}",
            r.e_corr,
            full.mp2_corr
        );
        assert!((r.total_energy - (rhf.energy + r.e_corr)).abs() < 1e-12);
    }

    /// ω→∞: erf → Coulomb, so the Δ-form must equal
    /// E_MP2[Coulomb] + (E_dRPA[Coulomb] − 2·E_OS[Coulomb]) computed explicitly.
    /// Pins the dRPA energy convention against the MP2 spin components.
    ///
    /// ω=200 Bohr⁻¹ is needed for 1e-6 Ha tolerance: at ω=50 the residual
    /// 2-center integral mismatch (P|erf(50r)/r|Q) vs (P|1/r|Q) is ~7 µHa for
    /// H₂/cc-pVDZ — a finite-ω truncation artefact, not a convention error.
    /// At ω=200 the same artefact drops to ~5e-7 Ha, safely below 1e-6.
    /// The tolerance is still 4 orders of magnitude tighter than any factor-2/4
    /// convention trap (~18 mHa = 1.8e-2 Ha), so it robustly catches spin-convention bugs.
    #[test]
    fn omega_to_infinity_is_mp2_plus_delta_drpa() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let cfg = RsMp2RpaConfig {
            omega: 200.0,
            ..Default::default()
        };
        let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();

        let ri_cfg = RiMp2Config::default();
        let (sc, _) =
            ri_mp2_spin_components(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &ri_cfg).unwrap();
        let drpa_cfg = PdepRpaConfig {
            trunc_thresh: 0.0,
            ..Default::default()
        };
        let drpa_coul =
            run_pdep_rpa(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &drpa_cfg).unwrap();
        let expected = sc.e_total + drpa_coul.e_rpa - 2.0 * sc.e_os;
        eprintln!(
            "Δ-form(ω=200) {:.10}  MP2+ΔdRPA[Coulomb] {:.10}",
            r.e_corr, expected
        );
        assert!(
            (r.e_corr - expected).abs() < 1e-6,
            "omega→∞ limit broken: {} vs {}",
            r.e_corr,
            expected
        );
        assert!((r.total_energy - (rhf.energy + r.e_corr)).abs() < 1e-12);
    }

    /// CoupledRings, ω→0: the two ΔdRPA terms (Coulomb and erfc) become equal
    /// (erfc → Coulomb as ω→0), so their difference cancels and e_corr → plain MP2.
    /// NOTE: the margin is cancellation-limited, not slack — at ω=0.05 the two
    /// ΔdRPA terms are each ~7.9e-3 Ha and the residual is ~7.9e-6 against the
    /// 1e-5 tolerance (79% of budget). A wider test ω would fail this limit.
    #[test]
    fn coupled_rings_omega_to_zero_reduces_to_mp2() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let full = ferric_mp2::rimp2::ri_mp2(
            &mol,
            &obs,
            &dfbs,
            Operator::coulomb(),
            &rhf,
            &ferric_mp2::rimp2::RiMp2Config::default(),
        )
        .unwrap();
        let cfg = RsMp2RpaConfig {
            omega: 0.05,
            formulation: RsMp2RpaFormulation::CoupledRings,
            ..Default::default()
        };
        let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();
        let delta_full = r.e_delta_drpa_full.unwrap();
        let delta_sr = r.e_delta_drpa_sr.unwrap();
        eprintln!(
            "CoupledRings(ω=0.05): MP2={:.10}  e_corr={:.10}  ΔdRPA_full={:.4e}  ΔdRPA_sr={:.4e}  diff={:.2e}",
            full.mp2_corr, r.e_corr, delta_full, delta_sr, r.e_corr - full.mp2_corr
        );
        assert!(
            (r.e_corr - full.mp2_corr).abs() < 1e-5,
            "CoupledRings ω→0 must reduce to MP2: {} vs {}",
            r.e_corr,
            full.mp2_corr
        );
        assert!((r.total_energy - (rhf.energy + r.e_corr)).abs() < 1e-12);
    }

    /// CoupledRings, ω→∞: erfc→0, so ΔdRPA[erfc]→0 and e_corr →
    /// E_MP2[Coulomb] + ΔdRPA[Coulomb], which must equal the same explicit
    /// MP2+ΔdRPA[Coulomb] value as the DeltaLr omega→∞ test.
    #[test]
    fn coupled_rings_omega_to_infinity_is_mp2_plus_delta_drpa() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let cfg = RsMp2RpaConfig {
            omega: 200.0,
            formulation: RsMp2RpaFormulation::CoupledRings,
            ..Default::default()
        };
        let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();

        let ri_cfg = RiMp2Config::default();
        let (sc, _) =
            ri_mp2_spin_components(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &ri_cfg).unwrap();
        let drpa_cfg = PdepRpaConfig {
            trunc_thresh: 0.0,
            ..Default::default()
        };
        let drpa_coul =
            run_pdep_rpa(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &drpa_cfg).unwrap();
        let expected = sc.e_total + drpa_coul.e_rpa - 2.0 * sc.e_os;
        eprintln!(
            "CoupledRings(ω=200): e_corr={:.10}  MP2+ΔdRPA[Coulomb]={:.10}  diff={:.2e}",
            r.e_corr,
            expected,
            r.e_corr - expected
        );
        assert!(
            (r.e_corr - expected).abs() < 1e-6,
            "CoupledRings ω→∞ limit broken: {} vs {}",
            r.e_corr,
            expected
        );
        assert!((r.total_energy - (rhf.energy + r.e_corr)).abs() < 1e-12);
    }

    /// The shared-intermediate fusion must change NOTHING. Reconstruct the
    /// CoupledRings energy at a realistic ω from independent
    /// ri_mp2_spin_components + run_pdep_rpa calls (the pre-fusion recipe) and
    /// demand bit-level agreement with the fused production path.
    #[test]
    fn coupled_rings_fusion_is_bit_identical() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let omega = 0.42;
        let cfg = RsMp2RpaConfig {
            omega,
            formulation: RsMp2RpaFormulation::CoupledRings,
            ..Default::default()
        };
        let fused = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();

        // Pre-fusion reconstruction: separate transforms for each operator.
        let ri_cfg = RiMp2Config::default();
        let (sc_full, _) =
            ri_mp2_spin_components(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &ri_cfg).unwrap();
        let (sc_sr, _) =
            ri_mp2_spin_components(&mol, &obs, &dfbs, Operator::erfc(omega), &rhf, &ri_cfg)
                .unwrap();
        let drpa_cfg = cfg.drpa.clone();
        let drpa_coul =
            run_pdep_rpa(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &drpa_cfg).unwrap();
        let drpa_erfc =
            run_pdep_rpa(&mol, &obs, &dfbs, Operator::erfc(omega), &rhf, &drpa_cfg).unwrap();
        let delta_full = drpa_coul.e_rpa - 2.0 * sc_full.e_os;
        let delta_sr = drpa_erfc.e_rpa - 2.0 * sc_sr.e_os;
        let expected = sc_full.e_total + delta_full - delta_sr;

        eprintln!(
            "fusion check: fused={:.12} unfused={:.12} diff={:.2e}",
            fused.e_corr,
            expected,
            fused.e_corr - expected
        );
        assert!(
            (fused.e_corr - expected).abs() < 1e-10,
            "fusion not bit-identical: {} vs {}",
            fused.e_corr,
            expected
        );
    }

    // ── terf/terfc attenuator arm ─────────────────────────────────────────
    //
    // The tempered split satisfies terf + terfc = Coulomb exactly, so it has the
    // SAME exact limits as erf/erfc — but the mapping to r0 is INVERTED vs ω:
    //   large r0 ⇒ ω=1/(r0√2)→0 ⇒ terfc→Coulomb, terf→0  ⇒ plain MP2
    //   small r0 ⇒ ω→∞          ⇒ terfc→0, terf→Coulomb  ⇒ MP2 + ΔdRPA[Coulomb]
    // These tests need the interpolation tables. They resolve FERRIC_TERF_TABLE_DIR
    // (or the main-repo `terf-tables/` — the worktree copy is generators-only, the
    // .bin tables are uncommitted), and SKIP with a note if absent, matching
    // crates/ferric-integrals/tests/terfc_base_validation.rs.

    /// Locate the terf .bin tables; None (→ skip) if not found.
    fn terf_tables_available() -> bool {
        if let Ok(d) = std::env::var("FERRIC_TERF_TABLE_DIR") {
            if std::path::Path::new(&d).join("16_4_2.bin").exists() {
                return true;
            }
        }
        false
    }

    /// terf, large r0 (ω→0): terf→0 ⇒ both DeltaLr and CoupledRings reduce to
    /// plain RI-MP2. r0=20 Bohr ⇒ ω≈0.035 Bohr⁻¹.
    #[test]
    fn terf_large_r0_reduces_to_mp2() {
        if !terf_tables_available() {
            eprintln!("SKIP terf_large_r0_reduces_to_mp2: terf tables absent");
            return;
        }
        let (mol, obs, dfbs, rhf) = setup_h2();
        let full = ferric_mp2::rimp2::ri_mp2(
            &mol,
            &obs,
            &dfbs,
            Operator::coulomb(),
            &rhf,
            &ferric_mp2::rimp2::RiMp2Config::default(),
        )
        .unwrap();
        for form in [
            RsMp2RpaFormulation::DeltaLr,
            RsMp2RpaFormulation::CoupledRings,
        ] {
            let cfg = RsMp2RpaConfig {
                attenuator: Attenuator::Terf,
                r0: 20.0,
                formulation: form,
                ..Default::default()
            };
            let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();
            eprintln!(
                "terf(r0=20, {form:?}): MP2={:.10} e_corr={:.10} diff={:.2e}",
                full.mp2_corr,
                r.e_corr,
                r.e_corr - full.mp2_corr
            );
            assert!(
                (r.e_corr - full.mp2_corr).abs() < 1e-4,
                "terf large-r0 must reduce to MP2 ({form:?}): {} vs {}",
                r.e_corr,
                full.mp2_corr
            );
            assert!((r.total_energy - (rhf.energy + r.e_corr)).abs() < 1e-12);
        }
    }

    /// terf, small r0 (ω→∞): terf→Coulomb ⇒ both formulations equal
    /// E_MP2[Coulomb] + (E_dRPA[Coulomb] − 2·E_OS[Coulomb]). r0=0.05 Bohr ⇒ ω≈14.
    ///
    /// Tolerance is 5e-4, not tighter, because r0=0.05 is the finite-range floor:
    /// the CLI sweep converges monotonically toward the target as r0 shrinks
    /// (-0.02258 @0.30 → -0.01861 @0.05, target -0.01843), the SAME finite-range
    /// truncation the erf ω→∞ test documents (it needs ω=200 for 1e-6; ω≈14 here
    /// leaves ~1.8e-4). This confirms the limit direction, not machine-precision
    /// equality — pushing r0 smaller runs off the terf table domain.
    #[test]
    fn terf_small_r0_is_mp2_plus_delta_drpa() {
        if !terf_tables_available() {
            eprintln!("SKIP terf_small_r0_is_mp2_plus_delta_drpa: terf tables absent");
            return;
        }
        let (mol, obs, dfbs, rhf) = setup_h2();
        let ri_cfg = RiMp2Config::default();
        let (sc, _) =
            ri_mp2_spin_components(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &ri_cfg).unwrap();
        let rpa_cfg = PdepRpaConfig {
            trunc_thresh: 0.0,
            ..Default::default()
        };
        let rpa_coul =
            run_pdep_rpa(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &rpa_cfg).unwrap();
        let expected = sc.e_total + rpa_coul.e_rpa - 2.0 * sc.e_os;
        for form in [
            RsMp2RpaFormulation::DeltaLr,
            RsMp2RpaFormulation::CoupledRings,
        ] {
            let cfg = RsMp2RpaConfig {
                attenuator: Attenuator::Terf,
                r0: 0.05,
                formulation: form,
                ..Default::default()
            };
            let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();
            eprintln!(
                "terf(r0=0.05, {form:?}): e_corr={:.10} MP2+ΔdRPA[Coul]={:.10} diff={:.2e}",
                r.e_corr,
                expected,
                r.e_corr - expected
            );
            assert!(
                (r.e_corr - expected).abs() < 5e-4,
                "terf small-r0 limit broken ({form:?}): {} vs {}",
                r.e_corr,
                expected
            );
            assert!((r.total_energy - (rhf.energy + r.e_corr)).abs() < 1e-12);
        }
    }

    /// The erf default must be untouched by the terf plumbing: an explicit
    /// Attenuator::Erf config equals the pre-change default at ω=0.42.
    #[test]
    fn erf_default_unchanged_by_terf_plumbing() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let a = rs_mp2_lr_rpa(
            &mol,
            &obs,
            &dfbs,
            &rhf,
            &RsMp2RpaConfig {
                omega: 0.42,
                ..Default::default()
            },
        )
        .unwrap();
        let b = rs_mp2_lr_rpa(
            &mol,
            &obs,
            &dfbs,
            &rhf,
            &RsMp2RpaConfig {
                omega: 0.42,
                attenuator: Attenuator::Erf,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            a.e_corr.to_bits(),
            b.e_corr.to_bits(),
            "explicit Erf must be bit-identical to the default"
        );
    }

    /// Item 3 smoke test: a normal small H2/cc-pVDZ run under a generous
    /// (default auto-resolved) budget must complete successfully — i.e. the
    /// new preflight gate (Item 1) does not reject it, and the stage-seam RSS
    /// warnings (Item 3), which are pure stderr observability and never
    /// affect control flow, do not disrupt the computation. This is the
    /// "normal small run produces no disruption" contract; the "no warning
    /// fires" half of that claim is unit-tested directly in
    /// ferric-core::memory (`warn_if_rss_over_does_not_panic_and_is_a_pure_observer`)
    /// since capturing this test binary's own stderr for a substring check
    /// would be flaky under the parallel test harness.
    #[test]
    fn small_system_run_completes_under_default_budget_item3_smoke() {
        let (mol, obs, dfbs, rhf) = setup_h2();
        let cfg = RsMp2RpaConfig::default();
        let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg)
            .expect("small H2/cc-pVDZ run must complete under the default auto-resolved budget");
        assert!(r.total_energy.is_finite());
    }

    // ── Open shell (scan_impl_unrestricted) ──────────────────────────────

    const WATER: &str =
        "3\nwater\nO 0.000 0.000 0.118\nH 0.000 0.755 -0.471\nH 0.000 -0.755 -0.471\n";
    const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";

    /// The RHF `rhf` written as a UHF with identical α and β orbitals,
    /// energies, densities and Fock matrices: a UHF stationary point.
    fn lift_to_unrestricted(rhf: &ScfResult) -> ScfResult {
        let mut u = rhf.clone();
        u.spin = ferric_scf::Spin::Unrestricted;
        u.density_beta = Some(rhf.density_alpha.clone());
        u.mos_beta = Some(rhf.mos_alpha.clone());
        u.eps_beta = Some(rhf.eps_alpha.clone());
        u.fock_beta = Some(rhf.fock_alpha.clone());
        u
    }

    fn setup_water_cs() -> (Molecule, PreparedBasis, PreparedBasis, ScfResult) {
        let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig::default()
                .with_energy_conv(1e-11)
                .with_density_conv(1e-10),
        )
        .unwrap();
        assert!(rhf.converged);
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        (mol, obs, dfbs, rhf)
    }

    fn setup_oh_uhf() -> (Molecule, PreparedBasis, PreparedBasis, ScfResult) {
        let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
        let uhf = ferric_scf::uhf::solve_uhf(
            &ParallelContext::default(),
            &mol,
            &obs,
            &bounds,
            &RhfConfig::default()
                .with_energy_conv(1e-11)
                .with_density_conv(1e-9)
                .with_max_iter(300),
        )
        .unwrap();
        assert!(uhf.converged);
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        (mol, obs, dfbs, uhf)
    }

    /// EXACTNESS ANCHOR (singlet limit). A closed-shell water RHF written as a
    /// UHF (α ≡ β) takes `scan_impl_unrestricted`; every field of both
    /// formulations must equal the restricted driver's.
    ///
    /// Bar 1e-10 Ha. Same orbitals on both sides, so what differs is only the
    /// arithmetic: (i) the MP2 blocks are summed in a different order
    /// (`u_ri_mp2_from_parts` vs `spin_components_from_b_ov`), O(ε·Σ|terms|)
    /// ≈ 1e-16 · 0.3 Ha; (ii) the dRPA takes the open-shell static-eigenbasis
    /// route, whose ε̃_U(0) is assembled as Π_α + Π_β (prefactor 2 + 2)
    /// instead of one prefactor-4 Π, and is diagonalized at full rank: the
    /// eigenvalues agree to n·ε·λ_max ≈ 84 · 2.2e-16 · O(10) ≈ 2e-13 each,
    /// and the energy sums ln λ + 1 − λ over ≤ 84 modes on 20 nodes, so its
    /// change is ≲ 84 · 20 · 2e-13 / (2π) · w_max ≈ 1e-11.
    #[test]
    fn unrestricted_matches_restricted_in_the_singlet_limit() {
        let (mol, obs, dfbs, rhf) = setup_water_cs();
        let uhf = lift_to_unrestricted(&rhf);
        for formulation in [
            RsMp2RpaFormulation::DeltaLr,
            RsMp2RpaFormulation::CoupledRings,
        ] {
            let cfg = RsMp2RpaConfig {
                formulation,
                ..Default::default()
            };
            let r = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();
            let u = rs_mp2_lr_rpa(&mol, &obs, &dfbs, &uhf, &cfg).unwrap();
            let opt = |a: Option<f64>, b: Option<f64>| match (a, b) {
                (Some(x), Some(y)) => (x - y).abs(),
                (None, None) => 0.0,
                _ => f64::INFINITY,
            };
            let diffs = [
                ("e_mp2_full", (r.e_mp2_full - u.e_mp2_full).abs()),
                ("e_sr_mp2", (r.e_sr_mp2 - u.e_sr_mp2).abs()),
                ("e_lr_mp2", (r.e_lr_mp2 - u.e_lr_mp2).abs()),
                ("e_dmp2_lr", (r.e_dmp2_lr - u.e_dmp2_lr).abs()),
                ("e_drpa_lr", opt(r.e_drpa_lr, u.e_drpa_lr)),
                (
                    "e_delta_drpa_full",
                    opt(r.e_delta_drpa_full, u.e_delta_drpa_full),
                ),
                ("e_delta_drpa_sr", opt(r.e_delta_drpa_sr, u.e_delta_drpa_sr)),
                ("e_corr", (r.e_corr - u.e_corr).abs()),
            ];
            for (name, d) in diffs {
                eprintln!("{formulation:?} {name}: |R - U| = {d:.3e}");
                assert!(
                    d <= 1e-10,
                    "{formulation:?} {name}: |R - U| = {d:.3e} > 1e-10"
                );
            }
        }
    }

    /// `open_shell_parts`' dMP2 is the second-order term of the spin-summed
    /// U-dRPA it is subtracted from, on a genuine open shell (OH doublet).
    ///
    /// Independent route: the U-dRPA kernel itself on the Coulomb pair with
    /// every b_ov scaled by `s`, so each integral scales by t = s². Then
    /// E(t) = d t² + c3 t³ + c4 t⁴ + …, and the two-point Richardson value
    /// 2E(t)/t² − E(2t)/(2t)² = d − 2 c4 t² (the t³ term cancels).
    ///
    /// Bar. At t = 5e-3 the truncation is 2|c4| t² = 5e-5 |c4|; with
    /// |c4| ≤ |d| (the ring series of a gapped molecule converges, |c_{n+1}/c_n|
    /// < 1, measured |c3/d| is printed) that is ≤ 5e-5 |d|. The frequency grid is
    /// 80 Gauss–Legendre nodes, on which the rational t² integrand is exact to
    /// ~1e-12 relative. Rounding: E(t) ≈ 2.5e-5 d, computed to ~84·ε absolute, so
    /// ~1e-9 relative. Bar 1e-4 · |d|. The closed-shell rule 2·E_αβ must miss by
    /// more than ten bars, or this test could not tell them apart.
    #[test]
    fn unrestricted_dmp2_is_the_second_order_of_the_udrpa() {
        use ferric_mp2::rimp2::compute_rpa_intermediates_spin;
        let (mol, obs, dfbs, uhf) = setup_oh_uhf();
        let ri = RiMp2Config::default();
        let op = Operator::coulomb();
        let a = compute_rpa_intermediates_spin(&mol, &obs, &dfbs, op, &uhf, &ri, true).unwrap();
        let b = compute_rpa_intermediates_spin(&mol, &obs, &dfbs, op, &uhf, &ri, false).unwrap();
        let parts = open_shell_parts(&a, &b, uhf.eps_a(), uhf.eps_b()).unwrap();
        let comps =
            ferric_mp2::u_rimp2::u_ri_mp2_from_parts(&a, &b, uhf.eps_a(), uhf.eps_b()).unwrap();
        let mut cfg = PdepRpaConfig {
            trunc_thresh: 0.0,
            ..Default::default()
        };
        cfg.quadrature.n_points = 80;
        let e_of = |t: f64| {
            let s = t.sqrt();
            let mut a2 = a.clone();
            let mut b2 = b.clone();
            a2.b_ov *= s;
            b2.b_ov *= s;
            crate::u_pdep_rpa_from_intermediates(&a2, &b2, &uhf, &cfg)
                .unwrap()
                .e_rpa
        };
        let t = 5e-3;
        let (e1, e2) = (e_of(t), e_of(2.0 * t));
        let d_rich = 2.0 * e1 / (t * t) - e2 / (4.0 * t * t);
        let c3 = (e2 / (4.0 * t * t) - e1 / (t * t)) / t;
        let bar = 1e-4 * parts.dmp2.abs();
        let miss = (2.0 * comps.e_ab - d_rich).abs();
        eprintln!(
            "dMP2 {:.12}  Richardson {:.12}  |diff| {:.3e}  bar {:.3e}  c3/d {:.3}  \
             |2 E_ab - d| {:.3e}",
            parts.dmp2,
            d_rich,
            (parts.dmp2 - d_rich).abs(),
            bar,
            c3 / d_rich,
            miss
        );
        assert!((parts.dmp2 - d_rich).abs() <= bar);
        assert!(
            miss > 10.0 * bar,
            "2 E_ab is not distinguishable: {miss:.3e}"
        );
        assert_eq!(parts.e_total.to_bits(), comps.e_total.to_bits());
    }

    /// Open-shell ω limits (OH doublet, UHF), both formulations, mirroring the
    /// closed-shell tests above: as ω → 0 (erf → 0) the energy tends to
    /// U-RI-MP2 at least linearly in ω; as ω → ∞ (erf → 1/r) the energy tends to
    /// E_MP2[C] + E_U-dRPA[C] − dMP2[C], with E_U-dRPA from the standalone
    /// `run_u_pdep_rpa` driver, and the gap shrinks as 1/ω².
    #[test]
    fn unrestricted_omega_limits() {
        use ferric_mp2::rimp2::compute_rpa_intermediates_spin;
        let (mol, obs, dfbs, uhf) = setup_oh_uhf();
        let ri = RiMp2Config::default();
        let ump2 = ferric_mp2::u_rimp2::u_ri_mp2(&mol, &obs, &dfbs, Operator::coulomb(), &uhf, &ri)
            .unwrap();
        let drpa_cfg = PdepRpaConfig {
            trunc_thresh: 0.0,
            ..Default::default()
        };
        let e_urpa = crate::run_u_pdep_rpa(&mol, &obs, &dfbs, Operator::coulomb(), &uhf, &drpa_cfg)
            .unwrap()
            .e_rpa;
        let op = Operator::coulomb();
        let a = compute_rpa_intermediates_spin(&mol, &obs, &dfbs, op, &uhf, &ri, true).unwrap();
        let b = compute_rpa_intermediates_spin(&mol, &obs, &dfbs, op, &uhf, &ri, false).unwrap();
        let d_c = open_shell_parts(&a, &b, uhf.eps_a(), uhf.eps_b())
            .unwrap()
            .dmp2;
        let full = ump2.mp2_corr + e_urpa - d_c;
        for formulation in [
            RsMp2RpaFormulation::DeltaLr,
            RsMp2RpaFormulation::CoupledRings,
        ] {
            let at = |omega: f64| {
                rs_mp2_lr_rpa(
                    &mol,
                    &obs,
                    &dfbs,
                    &uhf,
                    &RsMp2RpaConfig {
                        omega,
                        formulation,
                        ..Default::default()
                    },
                )
                .unwrap()
            };
            let (lo, lo4) = (at(0.05), at(0.0125));
            let (hi, hi4) = (at(200.0), at(800.0));
            let (l, l4) = (
                (lo.e_corr - ump2.mp2_corr).abs(),
                (lo4.e_corr - ump2.mp2_corr).abs(),
            );
            let (g, g4) = ((hi.e_corr - full).abs(), (hi4.e_corr - full).abs());
            eprintln!(
                "{formulation:?}: w->0 gap {l:.3e} (w=0.05) {l4:.3e} (w=0.0125); \
                 w->inf gap {g:.3e} (w=200) {g4:.3e} (w=800)"
            );
            // ω → 0: the leading long-range term is at least linear in ω (the
            // fitted erf(ωr)/r ≈ 2ω/√π − …), so a fourfold smaller ω cuts the
            // gap by ≥ 4, and at ω = 0.0125 it is below 1e-5 (closed shell:
            // 1e-5 at ω = 0.05 on H2, where T is cancellation-limited). A gap
            // already at the 1e-11 rounding floor needs no ratio.
            assert!(
                l4 < 1e-5 && (l4 * 4.0 <= l || l4 < 1e-11),
                "{formulation:?} omega->0"
            );
            // ω → ∞: erf(ωr)/r − 1/r integrated against Gaussians is O(1/ω²)
            // once ω² exceeds the tightest exponents (O 1s ~1e4 Bohr⁻² here),
            // so a fourfold ω cuts the gap by ≥ 8 (16 expected), and at ω = 800
            // it is below 1e-6. A construction error (wrong dMP2, a dropped
            // spin) leaves an ω-independent gap ≥ 1e-4 and fails both.
            assert!(
                g4 < 1e-6 && (g4 * 8.0 <= g || g4 < 1e-11),
                "{formulation:?} omega->inf"
            );
            assert!((hi.total_energy - (uhf.energy + hi.e_corr)).abs() < 1e-12);
        }
    }
}
