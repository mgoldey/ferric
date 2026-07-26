//! Attenuated MP2 with a long-range VV10 dispersion correction ("MP2-V").
//!
//! Goldey, Belzunces & Head-Gordon, *J. Chem. Theory Comput.* **11**, 4159
//! (2015), DOI 10.1021/acs.jctc.5b00509.
//!
//! # The construction
//!
//! Attenuated MP2 replaces `1/r₁₂` in the MP2 correlation integrals with a
//! short-range kernel, which **deletes** the long-range correlation tail. The
//! price is that the long-range C₆ coefficients go to zero — attenuated MP2 has
//! no dispersion at all beyond the attenuation length (paper §1: "the long-range
//! C6 coefficients are zero for attenuated MP2"). MP2-V **pastes a model tail
//! back** using the VV10 nonlocal correlation functional:
//!
//! ```text
//!     E_MP2-V = E_HF + E_c^attMP2(r₀) + E_nl^VV10[ρ; b, C, r₀]              (1)
//! ```
//!
//! ## The VV10 half is NOT plain VV10
//!
//! This is the single most important correctness point in this module. The
//! paper's Eq. 11 damps the VV10 pair kernel at short range so it does not
//! double-count the correlation attenuated MP2 already carries:
//!
//! ```text
//!     E_nlc = ∫dr ρ(r) ∫dr' ρ(r') Φ_VV10(|r−r'|, b, C) · [1 − terfc(|r−r'|, r₀)²]
//! ```
//!
//! and — critically — "**The r₀ parameter is shared with the attenuated
//! short-range MP2 part rather than being adjusted separately**" (paper p.
//! 4161). So the *same* r₀ appears in the MP2 operator and in the VV10 damping.
//! Feeding bare VV10 (the `Vv10Damping::None` path that ωB97X-V uses) into Eq. 1
//! would double-count short-range correlation and is NOT the published method;
//! [`AttVv10Config::vv10_damping`] exists so that difference is measurable, but
//! [`AttVv10Config::mp2_v_terfc_atz`] — the only parameterization the paper
//! actually fits — turns the damping on.
//!
//! ## Which attenuator
//!
//! The paper uses **terfc** (the "tempered" erfc of Dutoi/Goldey), not erfc:
//!
//! ```text
//!     terf(r, r₀)  = ½ [ erf((r−r₀)/(r₀√2)) + erf((r+r₀)/(r₀√2)) ]
//!     terfc(r, r₀) = 1 − terf(r, r₀)
//! ```
//!
//! ferric has both. [`AttVv10Attenuator::Terfc`] is the published operator and
//! requires the interpolation tables (`FERRIC_TERF_TABLE_DIR`);
//! [`AttVv10Attenuator::Erfc`] is offered as a table-free variant for smoke
//! testing and for comparison against ferric's PySCF-validated erfc path.
//! **The fitted (r₀, b, C) below belong to terfc and do NOT transfer to erfc** —
//! see [`AttVv10Config`].
//!
//! # Published parameters
//!
//! From the paper, §3 (Training) and Table 1, p. 4161–4162:
//!
//! | quantity | value | provenance |
//! |----------|-------|------------|
//! | basis    | aug-cc-pVTZ ("aTZ") | §3: "our previous work on attenuated MP2 achieved the greatest success with the Dunning aug-cc-pVTZ (aTZ) basis, we shall use it again here" |
//! | attenuator | terfc | method label "MP2-V(terfc, aTZ)" |
//! | r₀ | **1.00 Å** | Table 1 row `r0/Å = 1.00`, the RMSD minimum (0.199 kcal/mol); confirmed in text: "The optimal r₀ for MP2-V(terfc, aTZ), 1.00 Å" |
//! | b  | **11.0** | Table 1, same row; text: "The optimal damping parameter b is found to be 11.0" |
//! | C  | **0.0089** | §3: "We chose to fix the long-range correlation parameter C at the value optimized for LC-VV10 …, namely, C = 0.0089" |
//! | counterpoise | **NOT applied** | §2 Methods: "Counterpoise corrections were not performed unless otherwise indicated" |
//! | frozen core | **applied** | §2: "The frozen core approximation was used for all wave function results reported" |
//! | grid | SG-1 | §2: "Grid-based VV10 calculations use the SG1 grid" |
//!
//! Table 1 also reports the shallow (r₀, b) valley around that minimum — `b`
//! rises steeply with `r₀`, so the two are **not** independently tunable. If you
//! change `r₀`, move `b` with it:
//!
//! | r₀/Å | b | S66 RMSD | MSE | MUE |
//! |------|------|-------|--------|-------|
//! | 0.85 | 8.0  | 0.206 | −0.043 | 0.172 |
//! | 0.90 | 9.0  | 0.200 | 0.015  | 0.169 |
//! | 0.95 | 9.5  | 0.212 | −0.077 | 0.178 |
//! | **1.00** | **11.0** | **0.199** | **−0.019** | **0.167** |
//! | 1.05 | 12.5 | 0.201 | −0.024 | 0.167 |
//! | 1.10 | 14.5 | 0.204 | −0.026 | 0.169 |
//!
//! For context if this is ever benchmarked on S66 (Table 2, kcal/mol RMSD):
//! MP2/aTZ 1.533 → MP2(terfc, aTZ) 0.251 → **MP2-V(terfc, aTZ) 0.199**.
//!
//! **These are the ONLY (r₀, b, C) values this module ships.** The paper fits
//! MP2-V for aug-cc-pVTZ and no other basis. In particular there is **no
//! published MP2-V(terfc, aDZ) parameterization** — the aDZ numbers that appear
//! in the paper's Table 2 (`MP2(terfc, aDZ)`, r₀ = 1.05 Å) are for *uncorrected*
//! attenuated MP2, a different method with no VV10 term. Using the aTZ (r₀, b, C)
//! in a smaller basis is unparameterized extrapolation; this module lets you do
//! it (the config is public) but will not pretend it is the published method.
//!
//! Do **not** mix MP2-V's r₀ with uncorrected attenuated MP2's: the same method
//! family at the same aTZ basis uses r₀ = **1.35 Å** without the VV10 term
//! (no-CP; Huang 2014 gives 1.75 Å for the CP case). MP2-V's shorter 1.00 Å is
//! the whole point — the VV10 tail lets you attenuate harder.
//!
//! Two CP-convention caveats worth carrying:
//!  * These values are **no-counterpoise**. Huang, Goldey & Head-Gordon
//!    (the companion "achieving high accuracy … Coulomb-attenuated …" paper)
//!    document that the optimal r₀ shifts materially between CP and no-CP
//!    conventions — terfc r₀ = 1.35 Å (no-CP) vs 1.75 Å (CP) at aTZ for
//!    uncorrected attenuated MP2. Expect the same sensitivity here: applying
//!    these no-CP parameters to counterpoise-corrected interaction energies is
//!    a convention mismatch, not a free choice.
//!  * The paper's Table 1 numbers are the **self-consistent** (HF+VV10 orbitals)
//!    variant. It also reports the non-self-consistent (post-HF) value as
//!    "essentially identical" (0.202 vs 0.199 kcal/mol RMSD), explicitly
//!    concluding "Hence, VV10 can be used as a post-HF correction if desired".
//!    **This implementation is the post-HF variant**: it evaluates E_nl on the
//!    converged plain-HF density and adds it, with no VV10 term in the Fock
//!    matrix. That is the paper-sanctioned 0.202 kcal/mol path, not the 0.199.
//!
//! # Known convention mismatches (read before quoting a number)
//!
//! Even with the published (r₀, b, C), a run here is not bit-for-bit the
//! paper's protocol. The gaps, all deliberate and none silent:
//!
//!  * **Grid.** The paper used SG-1. ferric has no SG-1; the default is
//!    ferric's own NLC grid (50 radial × 50 angular, matching what
//!    `ferric_scf`'s KS drivers pass for wB97X-V). Configurable via
//!    [`AttVv10Config::nlc_grid`].
//!  * **Frozen core.** The paper froze cores; [`AttVv10Config::frozen_core`]
//!    defaults to `0` because ferric has no element-aware core counter and
//!    guessing one silently is worse than defaulting to all-electron. Set it.
//!  * **Self-consistency.** This is the post-HF variant (see above): E_nl is
//!    evaluated on the converged plain-HF density with no VV10 term in the
//!    Fock matrix.
//!  * **Basis.** The parameters are aug-cc-pVTZ-only. Nothing stops you running
//!    them in another basis; the result is unparameterized extrapolation.
//!
//! # Scope (Phase A + Phase B open-shell)
//!
//! Library-level energy only. No CLI, no Python bindings, no gradients, no
//! self-consistent VV10-in-Fock. See the crate docs / VALIDATION.md for what is
//! proven.
//!
//! [`att_mp2_vv10`] accepts **restricted (RHF)** references;
//! [`u_att_mp2_vv10`] accepts **unrestricted (UHF) and restricted-open (ROHF)**
//! references. Both share [`AttVv10Config`] and the identical VV10 evaluator.
//!
//! ## OPEN SHELL IS UNPARAMETERIZED — read this before quoting an open-shell number
//!
//! The paper fits (r₀ = 1.00 Å, b = 11.0, C = 0.0089) on **S66**, which is
//! **entirely closed-shell dimers**, at aug-cc-pVTZ, no counterpoise, frozen
//! core. There is **no open-shell MP2-V parameterization anywhere in that
//! paper** — no open-shell training set, no re-fit, not even a spot check.
//!
//! [`u_att_mp2_vv10`] therefore runs the **closed-shell-fitted parameters on an
//! open-shell reference**. That is unparameterized extrapolation, in the same
//! sense (and for the same reason) as running the aTZ parameters in a different
//! basis. This module will not adjust the parameters "for open shell" — there is
//! nothing published to adjust them *to*, and inventing a spin-dependent tweak
//! would be fabrication. The spin bookkeeping of the U-MP2 half is validated
//! (see the tests); the *numerical value of (r₀, b, C) for radicals* is not, and
//! cannot be from this paper.
//!
//! Note also that the paper reports only aggregate S66/G2 statistics (RMSD/MSE
//! in kcal/mol), never a single total energy, so there is **no external
//! reference total energy to match** for any spin case, closed or open. Both
//! paths are graded *Smoke* in `docs/VALIDATION.md` for exactly that reason.
//!
//! The paper's own recommendation is worth carrying: MP2-V "cannot be
//! recommended for evaluation of bonded interactions" (§4, G2/97 RMSD 9.69
//! kcal/mol vs plain MP2/aTZ's 8.28). It is a noncovalent-interaction method.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_dft::libxc::Vv10Params;
use ferric_dft::vv10::Vv10Damping;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::Spin;
use ferric_scf::ScfResult;

use crate::rimp2::{ri_mp2_spin_components, RiMp2Config, SpinComponents};
use crate::u_rimp2::{u_ri_mp2, URiMp2Components};

/// Bohr per Ångström. The paper quotes r₀ in Å; every ferric internal length
/// (grid coordinates, `Operator::terfc`'s `distance`) is in Bohr.
pub const BOHR_PER_ANG: f64 = 1.8897259886;

/// Which short-range attenuator multiplies the MP2 correlation operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttVv10Attenuator {
    /// `terfc(r₁₂, r₀)/r₁₂` — the operator the paper fits. Exact, via ferric's
    /// Dutoi/Goldey 2D interpolation tables, which must be present at runtime
    /// (`FERRIC_TERF_TABLE_DIR`, or the manifest-relative `terf-tables/`).
    Terfc,
    /// `erfc(ω r₁₂)/r₁₂` with `ω = 1/(r₀√2)` (the same curvature constraint
    /// `Operator::terfc` uses to derive its ω from r₀).
    ///
    /// **NOT the published operator.** Offered because it needs no tables and
    /// because ferric's erfc attenuated-MP2 path is cross-validated against
    /// PySCF to ~1e-8 Ha, which makes it a useful control. The fitted
    /// (r₀, b, C) do not transfer: erfc and terfc have different tails at
    /// matched r₀, so this combination is unparameterized.
    Erfc,
}

/// Configuration for [`att_mp2_vv10`].
#[derive(Debug, Clone)]
pub struct AttVv10Config {
    /// Range-separation length r₀ in **Bohr**. Use [`AttVv10Config::from_r0_angstrom`]
    /// or the named constructors rather than converting by hand.
    pub r0_bohr: f64,
    /// VV10 semiempirical parameters (C, b).
    pub vv10: Vv10Params,
    /// Short-range damping of the VV10 kernel.
    ///
    /// The published method requires `Vv10Damping::Terfc { r0_bohr }` with the
    /// SAME r₀ as the MP2 attenuation (paper Eq. 11 + p. 4161). `Vv10Damping::None`
    /// reproduces bare ωB97X-V-style VV10 and is provided only so the
    /// double-counting the damping removes can be measured.
    pub vv10_damping: Vv10Damping,
    /// Which attenuator multiplies the MP2 correlation operator.
    pub attenuator: AttVv10Attenuator,
    /// Frozen-core orbitals.
    ///
    /// **Convention mismatch to be aware of:** the paper used the frozen-core
    /// approximation for all wave-function results (§2 Methods), but ferric has
    /// no element-aware "count the core orbitals" helper, so this defaults to
    /// `0` (all-electron) rather than silently guessing a core size. Set it
    /// explicitly to match the paper's convention (e.g. 1 for a first-row
    /// hydride like water, 5 for a second-row atom). Since (r₀, b, C) were
    /// fitted with frozen core on, an all-electron run is a convention
    /// mismatch, not just a tighter calculation.
    pub frozen_core: usize,
    /// Grid for the VV10 nonlocal integration. The paper used SG-1; ferric has
    /// no SG-1, so the default here is ferric's own NLC grid setting (the same
    /// shape `ks.rs` passes for wB97X-V). Documented mismatch, not a silent one.
    pub nlc_grid: AtomicGridConfig,
    /// Resident-bytes ceiling forwarded to the RI-MP2 3-index MO transform.
    pub memory_budget_bytes: Option<usize>,
}

impl AttVv10Config {
    /// The published **MP2-V(terfc, aTZ)** parameterization.
    ///
    /// r₀ = 1.00 Å, b = 11.0, C = 0.0089, terfc attenuator, VV10 damped by
    /// `1 − terfc(R, r₀)²`, frozen core on. Fitted on S66 with **no
    /// counterpoise correction** in the **aug-cc-pVTZ** basis. See the module
    /// docs for the per-value provenance table.
    ///
    /// Applying this to any other basis, or to counterpoise-corrected
    /// interaction energies, is extrapolation beyond what was fitted.
    pub fn mp2_v_terfc_atz() -> Self {
        let r0_bohr = 1.00 * BOHR_PER_ANG;
        Self {
            r0_bohr,
            vv10: Vv10Params { c: 0.0089, b: 11.0 },
            vv10_damping: Vv10Damping::Terfc { r0_bohr },
            attenuator: AttVv10Attenuator::Terfc,
            frozen_core: 0,
            nlc_grid: default_nlc_grid(),
            memory_budget_bytes: None,
        }
    }

    /// The six (r₀/Å, b) pairs along Table 1's shallow valley, in the paper's
    /// order. `C = 0.0089` throughout (it was fixed, not fitted).
    ///
    /// `b` rises steeply with `r₀` — 8.0 at 0.85 Å to 14.5 at 1.10 Å — so these
    /// are NOT independently tunable. Exposed so a caller sweeping r₀ moves `b`
    /// with it instead of holding b = 11.0 and silently leaving the valley.
    pub const TABLE1_R0_B_PAIRS: [(f64, f64); 6] = [
        (0.85, 8.0),
        (0.90, 9.0),
        (0.95, 9.5),
        (1.00, 11.0),
        (1.05, 12.5),
        (1.10, 14.5),
    ];

    /// An MP2-V configuration at one of the published Table 1 (r₀, b) valley
    /// points, selected by r₀ in Ångström.
    ///
    /// Returns `None` for any r₀ not in [`Self::TABLE1_R0_B_PAIRS`] rather than
    /// interpolating `b` — the valley was sampled on a 0.05 Å grid and the
    /// paper gives no functional form, so an interpolated `b` would be an
    /// invented parameter. Use [`Self::from_r0_angstrom`] if you deliberately
    /// want an off-table r₀ (and pick `b` yourself, knowingly).
    pub fn mp2_v_terfc_atz_at_r0(r0_ang: f64) -> Option<Self> {
        let (_, b) = Self::TABLE1_R0_B_PAIRS
            .iter()
            .find(|(r, _)| (r - r0_ang).abs() < 1e-9)?;
        let r0_bohr = r0_ang * BOHR_PER_ANG;
        Some(Self {
            r0_bohr,
            vv10: Vv10Params { c: 0.0089, b: *b },
            vv10_damping: Vv10Damping::Terfc { r0_bohr },
            ..Self::mp2_v_terfc_atz()
        })
    }

    /// Same VV10 parameters and r₀ as [`Self::mp2_v_terfc_atz`] but with the
    /// **erfc** attenuator, which needs no interpolation tables.
    ///
    /// **Unparameterized.** erfc has a different tail from terfc at matched r₀,
    /// so the fitted (b, C) are not the right ones for this operator. Useful
    /// for smoke tests and for isolating table availability from method
    /// correctness; not a method to publish numbers from.
    pub fn erfc_control_at_atz_params() -> Self {
        Self {
            attenuator: AttVv10Attenuator::Erfc,
            ..Self::mp2_v_terfc_atz()
        }
    }

    /// Set r₀ from an Ångström value, keeping the VV10 damping's r₀ in sync
    /// **if** the damping is the terfc form (the paper shares one r₀ between
    /// the two halves). An explicitly-`None` damping is left alone.
    pub fn from_r0_angstrom(mut self, r0_ang: f64) -> Self {
        let r0_bohr = r0_ang * BOHR_PER_ANG;
        self.r0_bohr = r0_bohr;
        if let Vv10Damping::Terfc { .. } = self.vv10_damping {
            self.vv10_damping = Vv10Damping::Terfc { r0_bohr };
        }
        self
    }

    /// r₀ in Ångström (the unit the paper quotes).
    pub fn r0_angstrom(&self) -> f64 {
        self.r0_bohr / BOHR_PER_ANG
    }

    /// The two-electron operator this configuration attenuates MP2 with.
    ///
    /// Both branches derive ω from r₀ through the same curvature constraint
    /// `r₀ · ω = 1/√2` that [`Operator::terfc`] uses, so the two attenuators
    /// are compared at matched r₀ rather than at matched-but-unrelated ω.
    fn mp2_operator(&self) -> Operator {
        match self.attenuator {
            AttVv10Attenuator::Terfc => Operator::terfc(self.r0_bohr),
            AttVv10Attenuator::Erfc => {
                Operator::erfc(1.0 / (self.r0_bohr * std::f64::consts::SQRT_2))
            }
        }
    }

    /// The RI-MP2 configuration both the closed- and open-shell halves run
    /// under. Shared so `frozen_core` and the memory budget cannot diverge
    /// between the two spin paths.
    ///
    /// `frozen_core` is forwarded verbatim; the downstream
    /// `compute_rpa_intermediates_spin` / `ri_mp2_spin_components` resolve it
    /// through `rimp2::active_occ`, which errors on `frozen_core > nocc`
    /// instead of underflowing (never `nocc - frozen_core` — see the repo's
    /// reliability conventions). For the unrestricted path the same
    /// `frozen_core` is applied to **each** spin channel independently, which
    /// is the standard convention (freeze the same core shells in α and β) but
    /// means a value larger than `nocc_β` errors even when it fits in `nocc_α`.
    fn ri_mp2_config(&self) -> RiMp2Config {
        RiMp2Config {
            frozen_core: self.frozen_core,
            memory_budget_bytes: self.memory_budget_bytes,
            ..Default::default()
        }
    }
}

/// ferric's NLC grid shape for VV10, matching what `ferric_scf`'s KS drivers
/// pass as the `nlc` grid for wB97X-V. Deliberately a function, not a `Default`
/// impl on `AtomicGridConfig`, so the choice stays visible at the call site.
fn default_nlc_grid() -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial: 50,
        n_angular: 50,
        prune: None,
    }
}

/// Spin decomposition of the attenuated-MP2 correlation half.
///
/// The two spin cases genuinely have different natural decompositions and
/// forcing them into one struct would require inventing a mapping:
///
///  * Closed shell: opposite-/same-spin (`e_os`, `e_ss`) — the split SCS-MP2 and
///    the attenuated-MP2 literature use.
///  * Open shell: αα / ββ / αβ (`e_aa`, `e_bb`, `e_ab`) — the three genuinely
///    distinct blocks of U-MP2, where αα ≠ ββ in general.
///
/// One could report `e_ss = e_aa + e_bb`, but that discards the α/β asymmetry
/// that is the *entire* physical content of an open-shell calculation, so this
/// enum keeps both shapes intact rather than lossily unifying them.
#[derive(Debug, Clone)]
pub enum AttVv10SpinComponents {
    /// From a restricted reference, via [`ri_mp2_spin_components`].
    Restricted(SpinComponents),
    /// From an unrestricted or restricted-open reference, via [`u_ri_mp2`].
    Unrestricted(URiMp2Components),
}

impl AttVv10SpinComponents {
    /// Total correlation energy, whichever decomposition this is.
    pub fn e_total(&self) -> f64 {
        match self {
            Self::Restricted(s) => s.e_total,
            Self::Unrestricted(u) => u.e_total,
        }
    }
}

/// Energy decomposition returned by [`att_mp2_vv10`] and [`u_att_mp2_vv10`].
///
/// `total` is exactly `e_hf + e_c_att_mp2 + e_nl_vv10` — asserted, not assumed
/// (see [`AttVv10Result::components_sum_to_total`]).
#[derive(Debug, Clone)]
pub struct AttVv10Result {
    /// Reference Hartree–Fock total energy (from the supplied `ScfResult`).
    /// For the open-shell entry point this is the UHF/ROHF total energy.
    pub e_hf: f64,
    /// Attenuated MP2 correlation energy under the configured attenuator.
    pub e_c_att_mp2: f64,
    /// VV10 nonlocal correlation energy, damped per [`AttVv10Config::vv10_damping`].
    pub e_nl_vv10: f64,
    /// `e_hf + e_c_att_mp2 + e_nl_vv10`.
    pub total: f64,
    /// Spin split of `e_c_att_mp2` — shape depends on the reference (see
    /// [`AttVv10SpinComponents`]).
    pub spin_components: AttVv10SpinComponents,
    /// Number of grid points in the VV10 nonlocal integration (reported so a
    /// caller can tell a suspiciously small E_nl from a suspiciously small grid).
    pub n_nlc_points: usize,
    /// Spin type of the reference this result came from. Recorded because the
    /// open-shell path runs closed-shell-fitted parameters (see the module
    /// docs), so a consumer must be able to tell the two apart after the fact.
    pub reference_spin: Spin,
}

impl AttVv10Result {
    /// Residual of the additivity identity `total − (e_hf + e_c + e_nl)`.
    /// Exactly zero by construction; exposed so tests can assert it rather than
    /// trust it.
    pub fn components_sum_to_total(&self) -> f64 {
        self.total - (self.e_hf + self.e_c_att_mp2 + self.e_nl_vv10)
    }

    /// `true` when the (r₀, b, C) in play were used outside what the paper
    /// fitted *on the spin axis alone* — i.e. an open-shell reference.
    ///
    /// This does NOT cover the basis axis (the parameters are aug-cc-pVTZ-only)
    /// or the attenuator axis (they are terfc-only); those mismatches are
    /// documented on [`AttVv10Config`] and are the caller's to track.
    pub fn is_open_shell_extrapolation(&self) -> bool {
        !matches!(self.reference_spin, Spin::Restricted)
    }
}

/// Attenuated MP2 + long-range VV10 dispersion (MP2-V), post-HF.
///
/// ```text
///     E = E_HF + E_c^attMP2(r₀) + E_nl^VV10[ρ_HF; b, C, r₀]
/// ```
///
/// `rhf` must be a **closed-shell restricted** reference; anything else is a
/// hard [`FerricError`] pointing at [`u_att_mp2_vv10`], rather than a silently
/// wrong number (the RI-MP2 half here assumes doubly-occupied spatial orbitals,
/// so an unrestricted density would produce a plausible, meaningless energy).
///
/// The VV10 term is evaluated on the converged HF density with no feedback into
/// the Fock matrix (the paper's post-HF variant; see the module docs).
pub fn att_mp2_vv10(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    rhf: &ScfResult,
    config: &AttVv10Config,
) -> Result<AttVv10Result, FerricError> {
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(format!(
            "att_mp2_vv10 requires a closed-shell restricted (RHF) reference, got {:?}. \
             Use u_att_mp2_vv10 for UHF/ROHF references — but note that the published \
             (r0, b, C) were fitted on closed-shell S66 only, so open-shell MP2-V is \
             unparameterized extrapolation.",
            rhf.spin
        )));
    }
    validate_r0(config, "att_mp2_vv10")?;

    // ---- Half 1: attenuated MP2 correlation -------------------------------
    let (spin_components, _) = ri_mp2_spin_components(
        mol,
        obs,
        dfbs,
        config.mp2_operator(),
        rhf,
        &config.ri_mp2_config(),
    )?;

    assemble(
        mol,
        obs_bs,
        rhf,
        config,
        AttVv10SpinComponents::Restricted(spin_components),
    )
}

/// Attenuated MP2 + long-range VV10 dispersion (MP2-V) from an **open-shell**
/// UHF or ROHF reference, post-HF.
///
/// ```text
///     E = E_UHF + E_c^U-attMP2(r₀) + E_nl^VV10[ρ_α + ρ_β; b, C, r₀]
/// ```
///
/// # ⚠ The parameters are not validated for this case
///
/// The paper's (r₀, b, C) were fitted on **S66, which contains only closed-shell
/// dimers**, at aug-cc-pVTZ with no counterpoise and frozen core. There is no
/// published open-shell MP2-V parameterization. This function runs the
/// closed-shell-fitted values on an open-shell reference and does **not** adjust
/// them; the result is unparameterized extrapolation, and the returned
/// [`AttVv10Result::is_open_shell_extrapolation`] is `true` so a consumer cannot
/// lose track of that. See the module docs.
///
/// # What is and is not spin-dependent here
///
/// * The **MP2 half is spin-dependent** and goes through [`u_ri_mp2`], which
///   builds separate α and β `B^P_{ia}` tensors and sums the three genuinely
///   distinct αα / ββ / αβ blocks. It is fed the *same* `config.mp2_operator()`
///   (terfc or erfc at the same r₀) the closed-shell path uses — `u_ri_mp2`
///   takes an arbitrary [`Operator`], so no new integral machinery was needed.
/// * The **VV10 half is spin-agnostic**: VV10 is a functional of the *total*
///   density ρ = ρ_α + ρ_β and |∇ρ|² alone. This calls the identical
///   [`vv10_energy_on_density`] on `scf.density_total()`, exactly as
///   `ferric_dft::ks::KsXcUks::add_xc` calls the closed-shell `add_vv10_scratch`
///   on its own spin-summed density. No open-shell VV10 code exists or is
///   needed; a per-spin VV10 would be a *different functional*, not a
///   generalization of this one.
///
/// # ROHF
///
/// Accepted. [`u_ri_mp2`] supports ROHF by construction: ROHF stores a single
/// MO set (`mos_alpha`) that both spin channels share, and
/// `compute_rpa_intermediates_spin` falls back to it for the β request, with the
/// α/β occupation split taken from `mol.multiplicity`. `eps_beta` is likewise
/// absent for ROHF and falls back to `eps_alpha`. The resulting energy is
/// therefore ROHF-MP2 in the "use the ROHF canonical orbitals as if they were
/// UHF orbitals" (semicanonical-free) sense, which is what the rest of ferric's
/// open-shell MP2 stack already does — not a Z-averaged or fully semicanonical
/// ROHF-MP2.
pub fn u_att_mp2_vv10(
    mol: &Molecule,
    obs: &PreparedBasis,
    obs_bs: &ferric_core::basis::BasisSet,
    dfbs: &PreparedBasis,
    scf: &ScfResult,
    config: &AttVv10Config,
) -> Result<AttVv10Result, FerricError> {
    match scf.spin {
        Spin::Unrestricted | Spin::RestrictedOpen => {}
        Spin::Restricted => {
            return Err(FerricError::General(
                "u_att_mp2_vv10 requires an open-shell (UHF or ROHF) reference, got \
                 Restricted. Use att_mp2_vv10 for closed-shell references — routing a \
                 restricted result through the unrestricted path would silently take the \
                 alpha orbitals as if they were an independent spin channel."
                    .into(),
            ));
        }
    }
    validate_r0(config, "u_att_mp2_vv10")?;

    // ---- Half 1: unrestricted attenuated MP2 correlation ------------------
    // Same operator as the closed-shell path: u_ri_mp2 takes an arbitrary
    // Operator, so terfc/erfc attenuation needs nothing new here.
    let u = u_ri_mp2(
        mol,
        obs,
        dfbs,
        config.mp2_operator(),
        scf,
        &config.ri_mp2_config(),
    )?;

    assemble(
        mol,
        obs_bs,
        scf,
        config,
        AttVv10SpinComponents::Unrestricted(u.components),
    )
}

/// Shared tail of both entry points: the (damped) VV10 half plus the
/// three-way sum. Factored out so the VV10 evaluation is provably the *same
/// call* for both spin cases — the closed-shell and open-shell paths cannot
/// drift apart in the dispersion term.
fn assemble(
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    scf: &ScfResult,
    config: &AttVv10Config,
    spin_components: AttVv10SpinComponents,
) -> Result<AttVv10Result, FerricError> {
    let e_c_att_mp2 = spin_components.e_total();

    // VV10 is a functional of the TOTAL density only. `density_total()` is
    // D_α + D_β for U/RO and 2·D_α for R, i.e. the electron density in both
    // cases — so this line is spin-agnostic by construction, not by coincidence.
    let (e_nl_vv10, n_nlc_points) = vv10_energy_on_density(
        mol,
        obs_bs,
        scf.density_total(),
        &config.vv10,
        config.vv10_damping,
        &config.nlc_grid,
    )?;

    let e_hf = scf.energy;
    Ok(AttVv10Result {
        e_hf,
        e_c_att_mp2,
        e_nl_vv10,
        total: e_hf + e_c_att_mp2 + e_nl_vv10,
        spin_components,
        n_nlc_points,
        reference_spin: scf.spin,
    })
}

/// r₀ appears in a division (`1/(r₀√2)`) and as a length scale in the damping,
/// so a non-finite or non-positive value must be rejected, not propagated as a
/// NaN energy.
fn validate_r0(config: &AttVv10Config, who: &str) -> Result<(), FerricError> {
    if config.r0_bohr <= 0.0 || !config.r0_bohr.is_finite() {
        return Err(FerricError::General(format!(
            "{who}: r0 must be finite and > 0 (got {} Bohr)",
            config.r0_bohr
        )));
    }
    Ok(())
}

/// VV10 nonlocal correlation energy for a closed-shell total density matrix.
///
/// Standalone because it is exactly the quantity the ωB97X-V KS path computes
/// internally (`ferric_dft::ks` → `add_vv10_scratch`), which makes it the
/// cross-check seam: feeding this the same density, grid and `Vv10Params` that
/// a wB97X-V run uses, with `Vv10Damping::None`, must reproduce that run's E_nl.
pub fn vv10_energy_on_density(
    mol: &Molecule,
    obs_bs: &ferric_core::basis::BasisSet,
    d_total: &ndarray::Array2<f64>,
    params: &Vv10Params,
    damping: Vv10Damping,
    grid_cfg: &AtomicGridConfig,
) -> Result<(f64, usize), FerricError> {
    let grid = build_atomic_grid(mol, grid_cfg);
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = ferric_dft::ao_grid::eval_basis_and_grad_on_points(mol, obs_bs, &pts)
        .map_err(|e| FerricError::General(format!("att_mp2_vv10 AO grid evaluation: {e:?}")))?;
    let dens = ferric_dft::density_on_grid::eval_density_closed(d_total, &chi, &dchi);
    let (e_nl, _vrho, _vsig) =
        ferric_dft::vv10::compute_vv10_damped_energy_and_potentials(&grid, &dens, params, damping);
    Ok((e_nl, grid.len()))
}
