//! Supermolecular ΔdRPA-corrected MP2 (an MP2C-*inspired* construction).
//!
//! Algebraically `E_MP2 + ΔdRPA[v_d]`, with `ΔdRPA = E_dRPA − 2·E_OS`. The
//! Rust items keep the `mp2c` name for continuity; **this is not MP2C**.
//!
//! # What published MP2C does, and what this does instead
//!
//! Published MP2C (Pitoňák & Heßelmann, *J. Chem. Theory Comput.* **6**, 168
//! (2010); Huang, Goldey, Head-Gordon & Beran, *J. Chem. Theory Comput.* **10**,
//! 2054 (2014)) swaps **intermolecular SAPT dispersion terms**:
//!
//! ```text
//!     E_MP2C = E_MP2 − E_disp^(20),UCHF + E_disp^TDDFT                    (1)
//! ```
//!
//! both dispersion terms built from *monomer* responses, so the swap touches
//! only the dispersion interaction. That construction is [`crate::mp2c_cks`]
//! (experimental). This module instead swaps the **whole supermolecular
//! direct-ring correlation energy** of each system:
//!
//! ```text
//!     E_c = E_MP2[v] − 2·E_OS[v_d] + E_dRPA[v_d]                          (4)
//! ```
//!
//! `2·E_OS` is the second-order direct-ring (dMP2) energy (closed-shell spin
//! summation of the exchange-free second-order term) and `E_dRPA` is its
//! infinite-order ring resummation, so (4) replaces MP2's direct-ring
//! correlation with dRPA's. The interaction energy is then recovered by
//! counterpoise-corrected supermolecular subtraction (dimer − monomers).
//!
//! **The non-additivity of (4) is not the dispersion swap of (1).** Besides
//! the coupled-minus-uncoupled dispersion, the dimer-minus-monomers difference
//! of `E_dRPA − 2·E_OS` also contains ring-coupling corrections to the
//! intramonomer correlation as it responds to the partner (the correlation
//! parts of electrostatics and induction) and changes driven by exchange and
//! overlap. Those extra pieces vanish only as `R → ∞`; at van der Waals
//! contact they need not be small, and nothing here separates them.
//!
//! Consistent with that, the measured corrections are much larger than
//! published MP2C corrections on comparable complexes: A24-19 (methane dimer)
//! and A24-20 (methane–Ar) give +0.29 and +0.22 kcal/mol at aug-cc-pVDZ
//! (`tests/validation_mp2c.rs`), against published MP2C corrections of order
//! 0.0x kcal/mol for the smallest S22 complexes (methane dimer −0.060).
//!
//! The coupling source is dRPA, not TDDFT: no functional enters. The same
//! substitution is `rs_mp2_rpa`'s formulation B.
//!
//! ## Relationship to formulation B
//!
//! With [`Mp2cDispersionRange::LongRange`], Eq. (4) is *algebraically identical*
//! to `rs_mp2_rpa`'s formulation B:
//!
//! ```text
//!     E_c^B = E_MP2[v] + (E_dRPA[v_lr] − 2·E_OS[v_lr])
//! ```
//!
//! [`mp2c_energy`] with `LongRange` and `rs_mp2_lr_rpa` with
//! `DeltaLr` must agree to the floating-point noise floor at matched ω; that
//! equivalence is asserted in the tests (`long_range_mp2c_equals_formulation_b`)
//! rather than assumed, because it is the load-bearing structural claim.
//!
//! The default is [`Mp2cDispersionRange::FullRange`] (no range separation,
//! as in published MP2C), which is the ω→∞ limit of B, i.e.
//! `E_MP2 + ΔdRPA[Coulomb]`.
//!
//! # Counterpoise and ghost atoms
//!
//! Everything in (4) is a correlation energy built from occupied/virtual MO
//! blocks, and the orbital counts come from `mol.nelec()` and `obs.nbasis()`.
//! A ghost center contributes basis functions (raising `nbasis`, hence `nvir`)
//! and zero electrons (leaving `nelec`, hence `nocc`, untouched), which is
//! precisely the counterpoise construction. No term in (4) reads atomic
//! positions or nuclear charges directly, so there is no ghost-specific code
//! path here and none is needed. [`mp2c_interaction_energy`] does the CP
//! subtraction; the ghost-correctness of the pieces is tested in
//! `tests/mp2c_ghost.rs`.
//!
//! # A shared-machinery trap worth knowing about
//!
//! [`run_pdep_rpa_from_intermediates`] takes both an `op: Operator` and a
//! `config.frozen_core`, but on that entry point **neither is load-bearing for
//! the energy**: the correlation energy is determined entirely by the prebuilt
//! `inter` (its `b_ov` already encodes the operator, its `nocc`/`first_occ`
//! already encode the frozen core). `op` and `config.frozen_core` only reach
//! the off-by-default screened-b_ov path.
//!
//! This was established by mutation testing during development: seeding a wrong
//! `op` argument, or zeroing `drpa_cfg.frozen_core`, into the calls below
//! changes *nothing* and no test can catch it, because there is nothing to
//! catch — the mutations are semantically inert. The mutations that *are*
//! harmful are passing the wrong **intermediates** (`it_full` where `it_lr`
//! belongs) or dropping `frozen_core` from **`ri_cfg`**, and both are covered
//! by `tests/mp2c_limits.rs`.
//!
//! The two assignments below are therefore kept as defensive consistency (they
//! cost nothing and would matter the moment the screened path is enabled, or if
//! the function's contract changes), not because they currently affect a
//! number. `rs_mp2_rpa.rs` carries the identical exposure on the same function.
//!
//! # Scope
//!
//! Library-level energy only, closed-shell RHF references. No CLI, no Python
//! bindings, no gradients, no open-shell path. See the validation page
//! (`site/src/reference/validation.md`).

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{compute_rpa_intermediates, spin_components_from_b_ov, RiMp2Config};
use ferric_scf::result::Spin;
use ferric_scf::ScfResult;

use crate::{run_pdep_rpa_from_intermediates, PdepRpaConfig};

/// Which two-electron operator the uncoupled→coupled dispersion swap acts
/// through.
///
/// Both choices leave `E_MP2[v]` (the full Coulomb MP2 energy) untouched; they
/// differ only in the kernel used for the `−2E_OS + E_dRPA` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mp2cDispersionRange {
    /// Swap the **full-range** dispersion: `v_d = 1/r₁₂`.
    ///
    /// ```text
    ///     E_c = E_MP2[v] − 2·E_OS[v] + E_dRPA[v]
    /// ```
    ///
    /// Full range like published MP2C (which has no range separation), but
    /// swapping the whole supermolecular direct-ring energy rather than only
    /// the intermolecular dispersion (see the module docs). `omega` is not
    /// consulted.
    ///
    /// Note this is also the ω→∞ limit of the long-range form, and equals
    /// `E_MP2 + ΔdRPA[Coulomb]`.
    #[default]
    FullRange,
    /// Swap only the **long-range** dispersion: `v_d = erf(ωr₁₂)/r₁₂`.
    ///
    /// ```text
    ///     E_c = E_MP2[v] − 2·E_OS[v_lr] + E_dRPA[v_lr]
    /// ```
    ///
    /// Algebraically identical to `rs_mp2_rpa`'s formulation B at the same ω
    /// (see the module docs). Rationale: dRPA's ring series has a
    /// short-range self-correlation error, so restricting the swap to the long
    /// range keeps MP2's (better) short-range correlation intact. Uses
    /// [`Mp2cConfig::omega`].
    LongRange,
}

/// Configuration for [`mp2c_energy`].
#[derive(Debug, Clone)]
pub struct Mp2cConfig {
    /// Which range the dispersion swap acts through. Default
    /// [`Mp2cDispersionRange::FullRange`] (the literature grouping).
    pub range: Mp2cDispersionRange,
    /// Range-separation parameter ω in **Bohr⁻¹**, used only when
    /// `range == LongRange`. Default 0.222 Bohr⁻¹ = 0.420 Å⁻¹, the erfc-optimal
    /// value of Goldey & Head-Gordon JPCL 2012 and the same default
    /// `RsMp2RpaConfig` carries, so the two are directly comparable.
    ///
    /// Ignored entirely for `FullRange`. A `LongRange` run with ω ≤ 0 or
    /// non-finite is a hard error rather than a NaN energy.
    pub omega: f64,
    /// Frozen-core orbitals. Forwarded verbatim to both the MP2 and the dRPA
    /// halves so the two cannot disagree about the correlated space (a
    /// mismatch would leave a spurious residue in `−2E_OS + E_dRPA`, which is
    /// supposed to cancel to zero at second order).
    ///
    /// Resolved downstream through `rimp2::active_occ`, which errors on
    /// `frozen_core >= nocc` instead of underflowing.
    pub frozen_core: usize,
    /// dRPA solver knobs. Defaults to full rank (`trunc_thresh = 0.0`): MP2C is
    /// an energy method and the `−2E_OS + E_dRPA` difference is a small
    /// residual between two much larger numbers, so PDEP truncation error
    /// enters it undamped. Truncation is an explicit opt-in, never a default.
    ///
    /// The nested `drpa.frozen_core` is ignored — `frozen_core` above is
    /// authoritative and overwrites it before the RPA call.
    pub drpa: PdepRpaConfig,
}

impl Default for Mp2cConfig {
    fn default() -> Self {
        Self {
            range: Mp2cDispersionRange::FullRange,
            omega: 0.420 * ferric_mp2::attenuated::BOHR_INV_PER_ANG_INV,
            frozen_core: 0,
            drpa: PdepRpaConfig {
                trunc_thresh: 0.0,
                ..Default::default()
            },
        }
    }
}

impl Mp2cConfig {
    /// The dispersion-swap operator this configuration selects.
    fn dispersion_operator(&self) -> Operator {
        match self.range {
            Mp2cDispersionRange::FullRange => Operator::coulomb(),
            Mp2cDispersionRange::LongRange => Operator::erf(self.omega),
        }
    }

    /// ω is only meaningful — and only validated — for the long-range form.
    /// A `FullRange` run with a garbage ω is fine, because nothing reads it.
    fn validate(&self) -> Result<(), FerricError> {
        if self.range == Mp2cDispersionRange::LongRange
            && (self.omega <= 0.0 || !self.omega.is_finite())
        {
            return Err(FerricError::General(format!(
                "mp2c: LongRange requires a finite omega > 0 (got {} Bohr^-1). \
                 Use Mp2cDispersionRange::FullRange if you do not want range separation.",
                self.omega
            )));
        }
        Ok(())
    }
}

/// Component breakdown of an MP2C calculation.
///
/// `e_corr = e_mp2 − e_disp_uncoupled + e_disp_coupled` exactly; asserted, not
/// assumed (see [`Mp2cResult::components_sum_to_corr`]).
#[derive(Debug, Clone)]
pub struct Mp2cResult {
    /// `E_MP2[Coulomb]` — the full, exchange-inclusive second-order energy.
    /// Always the full Coulomb operator regardless of `range`; the range
    /// selection only touches the dispersion swap.
    pub e_mp2: f64,
    /// `2·E_OS[v_d]`: the second-order direct-ring correlation energy of THIS
    /// system (removed; subtracted in `e_corr`). A per-system correlation
    /// energy, not a dispersion energy: only part of its dimer-minus-monomers
    /// difference is dispersion (see the module docs).
    pub e_disp_uncoupled: f64,
    /// `E_dRPA[v_d]`: the dRPA correlation energy of THIS system (added). Same
    /// caveat as `e_disp_uncoupled`: a correlation energy, not a dispersion
    /// energy.
    pub e_disp_coupled: f64,
    /// `e_disp_coupled − e_disp_uncoupled`, the net MP2C correction to MP2.
    /// Exposed because it is the quantity that must vanish in the reduction
    /// limit and the one whose sign carries the physics (MP2C should *weaken*
    /// dispersion binding, so this is typically positive).
    pub e_correction: f64,
    /// `e_mp2 + e_correction` — the MP2C correlation energy.
    pub e_corr: f64,
    /// `scf.energy + e_corr`.
    pub total_energy: f64,
    /// Opposite-spin part of `E_MP2[Coulomb]`, reported for diagnostics and
    /// because `2·e_os_full` is the *full-range* uncoupled dispersion even when
    /// the swap was long-range only.
    pub e_os_full: f64,
    /// Same-spin part of `E_MP2[Coulomb]`.
    pub e_ss_full: f64,
    /// Whether the dRPA eigensolve met its convergence tolerance. `false` means
    /// `e_disp_coupled` came from unconverged best-effort Ritz pairs; MP2C does
    /// not silently swallow that (see the reliability conventions).
    pub drpa_converged: bool,
    /// Which range the swap acted through, recorded so a consumer can tell a
    /// full-range from a long-range result after the fact.
    pub range: Mp2cDispersionRange,
}

impl Mp2cResult {
    /// Residual of the additivity identity
    /// `e_corr − (e_mp2 − e_disp_uncoupled + e_disp_coupled)`.
    ///
    /// **Not exactly zero.** `e_corr` is assembled as
    /// `e_mp2 + (e_disp_coupled − e_disp_uncoupled)` — the correction is formed
    /// first, so it can be reported as `e_correction` — whereas this method
    /// re-associates the same three terms in a different order. In IEEE-754
    /// that is a 1-ULP-scale difference, not an identity. Assert against a
    /// tolerance (a few ULP of `e_corr`), never `== 0.0`.
    pub fn components_sum_to_corr(&self) -> f64 {
        self.e_corr - (self.e_mp2 - self.e_disp_uncoupled + self.e_disp_coupled)
    }
}

/// Supermolecular ΔdRPA-corrected MP2 for one system: MP2 with its
/// direct-ring correlation `2·E_OS` replaced by dRPA. Named `mp2c_*` for
/// continuity; it is not published MP2C, which swaps only intermolecular SAPT
/// dispersion terms (see the module docs).
///
/// ```text
///     E = E_HF + E_MP2[v] − 2·E_OS[v_d] + E_dRPA[v_d]
/// ```
///
/// `scf` must be a **closed-shell restricted** reference; anything else is a
/// hard error rather than a plausible, meaningless number (the closed-shell
/// `2·E_OS = E_dMP2` identity that the whole construction rests on is a
/// closed-shell spin-summation result and does not hold for an unrestricted
/// reference).
///
/// # Reduction limit
///
/// With `LongRange` and ω→0 the operator `erf(ωr)/r → 0`, so both dispersion
/// terms vanish and this reduces to plain RI-MP2. That is the limit tested in
/// `long_range_omega_to_zero_reduces_to_mp2`. `FullRange` has **no** such
/// limit — it has no parameter to turn off — which is why the reduction test
/// necessarily runs on the long-range arm.
///
/// # Cost
///
/// One dRPA solve plus the MP2 spin components. For `FullRange` both halves
/// share a single set of RI intermediates (one aux-blocked `(P|op|ia)`
/// transform, the expensive step); `LongRange` needs two (Coulomb for E_MP2,
/// erf for the swap).
pub fn mp2c_energy(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    scf: &ScfResult,
    cfg: &Mp2cConfig,
) -> Result<Mp2cResult, FerricError> {
    if !matches!(scf.spin, Spin::Restricted) {
        return Err(FerricError::General(format!(
            "mp2c_energy requires a closed-shell restricted (RHF) reference, got {:?}. \
             The E_dMP2 = 2*E_OS identity the uncoupled-dispersion term relies on is a \
             closed-shell spin-summation result; there is no open-shell MP2C path in ferric.",
            scf.spin
        )));
    }
    cfg.validate()?;

    let ri_cfg = RiMp2Config {
        frozen_core: cfg.frozen_core,
        memory_budget_bytes: cfg.drpa.memory_budget_bytes,
        ..Default::default()
    };
    let mut drpa_cfg = cfg.drpa.clone();
    drpa_cfg.frozen_core = cfg.frozen_core;

    let eps = scf.eps_r();
    let op_d = cfg.dispersion_operator();

    // SHARED-INTERMEDIATE FUSION, same as rs_mp2_rpa: the MP2 spin components
    // and the dRPA solve both need the dressed b_ov = V^{-1/2}(P|op|ia), and
    // building it is the expensive step. For FullRange the two halves want the
    // SAME operator, so exactly one transform is built and fed to both — the
    // dRPA and the E_OS it is differenced against then come from bit-identical
    // intermediates, which is what makes the second-order cancellation in
    // (E_dRPA − 2E_OS) clean rather than noise-limited.
    let it_full = compute_rpa_intermediates(mol, obs, dfbs, Operator::coulomb(), scf, &ri_cfg)?;
    let sc_full = spin_components_from_b_ov(
        &it_full.b_ov,
        eps,
        it_full.nocc,
        it_full.nvir,
        it_full.first_occ,
        it_full.nocc_total,
    );

    let (e_disp_uncoupled, drpa) = match cfg.range {
        Mp2cDispersionRange::FullRange => {
            // Reuse it_full: the swap operator IS Coulomb here.
            let drpa = run_pdep_rpa_from_intermediates(
                &it_full,
                mol,
                obs,
                dfbs,
                Operator::coulomb(),
                scf,
                &drpa_cfg,
            )?;
            (2.0 * sc_full.e_os, drpa)
        }
        Mp2cDispersionRange::LongRange => {
            let it_lr = compute_rpa_intermediates(mol, obs, dfbs, op_d, scf, &ri_cfg)?;
            let sc_lr = spin_components_from_b_ov(
                &it_lr.b_ov,
                eps,
                it_lr.nocc,
                it_lr.nvir,
                it_lr.first_occ,
                it_lr.nocc_total,
            );
            let drpa =
                run_pdep_rpa_from_intermediates(&it_lr, mol, obs, dfbs, op_d, scf, &drpa_cfg)?;
            (2.0 * sc_lr.e_os, drpa)
        }
    };

    let e_disp_coupled = drpa.e_rpa;
    let e_correction = e_disp_coupled - e_disp_uncoupled;
    let e_corr = sc_full.e_total + e_correction;

    Ok(Mp2cResult {
        e_mp2: sc_full.e_total,
        e_disp_uncoupled,
        e_disp_coupled,
        e_correction,
        e_corr,
        total_energy: scf.energy + e_corr,
        e_os_full: sc_full.e_os,
        e_ss_full: sc_full.e_ss,
        drpa_converged: drpa.eigensolver_converged,
        range: cfg.range,
    })
}

/// A counterpoise-corrected ΔdRPA-corrected-MP2 interaction energy from three
/// already-computed [`mp2c_energy`] results.
///
/// ```text
///     ΔE = E(dimer) − E(monomer A, dimer basis) − E(monomer B, dimer basis)
/// ```
///
/// All three [`Mp2cResult`]s must come from runs in the **same basis at the
/// same geometry**, with the monomers carrying ghost centers at the absent
/// fragment's atomic positions (`@C`, `@H`, … in the XYZ). This function cannot
/// verify that — it only sees three numbers — so [`Mp2cInteraction`] records
/// the per-term breakdown and the caller is responsible for having fed it a
/// genuine CP triple. `tests/mp2c_ghost.rs` exercises the real path.
///
/// # What the supermolecular subtraction does and does not isolate
///
/// Both dRPA and `E_OS` are size-consistent, so the correction's non-additive
/// residue vanishes for infinitely separated fragments and tends to the
/// coupled-minus-uncoupled dispersion as `R → ∞`. At finite `R` it also
/// carries ring-coupling corrections to the intramonomer correlation
/// (electrostatics/induction correlation) and exchange–overlap changes, which
/// published MP2C's monomer-response SAPT terms exclude by construction. This
/// is therefore not the published MP2C interaction energy.
pub fn mp2c_interaction_energy(
    dimer: &Mp2cResult,
    monomer_a: &Mp2cResult,
    monomer_b: &Mp2cResult,
) -> Mp2cInteraction {
    let d = |f: fn(&Mp2cResult) -> f64| f(dimer) - f(monomer_a) - f(monomer_b);
    // `total_energy - e_corr` recovers the SCF energy each result was built on.
    let e_int_hf = d(|r| r.total_energy - r.e_corr);
    let e_int_correction = d(|r| r.e_correction);
    Mp2cInteraction {
        e_int_mp2c: d(|r| r.total_energy),
        e_int_mp2: e_int_hf + d(|r| r.e_mp2),
        e_int_correction,
        e_int_hf,
    }
}

/// Counterpoise-corrected interaction-energy breakdown, all in Hartree.
///
/// `e_int_mp2c = e_int_mp2 + e_int_correction` exactly.
#[derive(Debug, Clone)]
pub struct Mp2cInteraction {
    /// MP2C interaction energy: `ΔE(HF) + ΔE(MP2 corr) + ΔE(correction)`.
    pub e_int_mp2c: f64,
    /// Plain MP2 interaction energy (HF + MP2 correlation), for comparison —
    /// this is the number MP2C is supposed to improve on.
    pub e_int_mp2: f64,
    /// The correction's own contribution to the interaction energy,
    /// `Δ(E_dRPA[v_d] − 2E_OS[v_d])`, dimer minus monomers. Positive
    /// (repulsive) on the dispersion-bound A24 complexes measured; it is not
    /// the published MP2C dispersion correction (see the module docs).
    pub e_int_correction: f64,
    /// Hartree–Fock interaction energy alone.
    pub e_int_hf: f64,
}

impl Mp2cInteraction {
    /// Hartree → kcal/mol.
    pub const HARTREE_TO_KCAL: f64 = 627.509_474_06;

    /// `e_int_mp2c` in kcal/mol, the unit A24/S22/S66 reference values use.
    pub fn e_int_mp2c_kcal(&self) -> f64 {
        self.e_int_mp2c * Self::HARTREE_TO_KCAL
    }

    /// `e_int_mp2` in kcal/mol.
    pub fn e_int_mp2_kcal(&self) -> f64 {
        self.e_int_mp2 * Self::HARTREE_TO_KCAL
    }

    /// Residual of `e_int_mp2c − (e_int_mp2 + e_int_correction)`.
    ///
    /// **Not exactly zero, and the scale that matters is not the interaction
    /// energy.** Each of these terms is a difference of total energies of order
    /// 10²  Ha, so the residual carries ULP noise at ~1e-14 Ha *absolute* even
    /// though the interaction energies themselves are ~1e-3 Ha. Assert against
    /// a tolerance scaled to the total energies involved (or simply use a
    /// ~1e-12 Ha absolute bound), never `== 0.0` and never a bound derived from
    /// the size of the interaction energy.
    pub fn components_sum_to_total(&self) -> f64 {
        self.e_int_mp2c - (self.e_int_mp2 + self.e_int_correction)
    }
}
