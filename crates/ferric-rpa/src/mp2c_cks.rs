//! MP2C with a **genuine** coupled-Kohn–Sham dispersion term (EXPERIMENTAL).
//!
//! Pitoňák & Heßelmann, *J. Chem. Theory Comput.* **6**, 168 (2010):
//!
//! ```text
//!     E_MP2C = E_MP2 − E_disp^UCHF + E_disp^CKS                           (1)
//! ```
//!
//! # Open defect: the coupled response is unstable in the dimer basis
//!
//! The MP2C correction `E_disp^CKS − E_disp^UCHF` does not reproduce the
//! published table (Huang, PhD dissertation, UC Riverside 2015, Table 2.1).
//! aug-cc-pVDZ / aug-cc-pVDZ-RI, ALDAX, 16 frequencies, RHF monomer
//! references, kcal/mol (`tests/mp2c_cks_published_sign.rs`):
//!
//! ```text
//!   system            E_disp^UCHF   E_disp^CKS   correction   published   min eig M(0) A / B
//!   methane dimer       −0.8787      −0.6935      +0.1852      −0.060      −0.038 / −0.038
//!   ethene dimer        −0.5121      −0.8468      −0.3348      −0.040      +0.323 / −0.401
//!   A24-12 / S22-16     −1.4618      −1.0710      +0.3908      +0.060      −1.975 / +0.751
//! ```
//!
//! The last column is the smallest eigenvalue of the static Dyson matrix
//! `M = I + Π⁰^{1/2} K Π⁰^{1/2}` of each monomer
//! ([`crate::cks_dispersion::MonomerResponse::dyson_min_eigenvalue`]). On
//! every system at least one monomer has `M(0)` negative, so its coupled
//! response has a negative mode and the imaginary-frequency integrand crosses
//! a pole (the eigenvalue passes through zero between quadrature nodes). The
//! Casimir–Polder integral is then not defined, and the value the quadrature
//! returns is arbitrary: the uncoupled term reproduces to four digits across
//! small upstream changes while the ethene CKS term moved from −0.146 to
//! −0.847 kcal/mol. [`Mp2cCksInteraction::static_dyson_min_eigenvalue`]
//! reports this and a warning is printed when it is `≤ 0`.
//!
//! The instability comes from the monomer-in-dimer-basis construction, not the
//! kernel assembly: the isolated monomers are stable (methane/aug-cc-pVDZ
//! `M(0)` = 0.969 RHF, 0.957 KS-LDA_X; water/cc-pVDZ 0.982 / 0.975), while the
//! same methane with ghost partner centers gives −0.038 (RHF) and +0.020 (a
//! KS reference built with the kernel's own functional, `LDA_X`: positive but
//! near-singular). `tests/cks_dyson_stability.rs` measures this. The leading
//! hypothesis, not yet tested, is the ALDA kernel's `ρ^{-2/3}` growth where
//! the monomer density is small, sampled by the auxiliary and virtual
//! functions on the ghost centers. The published protocol (asymptotically
//! corrected KS orbitals, MP2/CBS baseline) is also not implemented here.
//!
//! # How this differs from [`crate::mp2c`]
//!
//! `mp2c.rs` implements (1) with **dRPA substituted for CKS**, evaluated
//! supermolecularly. This module implements the published construction:
//!
//! | | `mp2c.rs` | this module |
//! |---|---|---|
//! | coupled term | `E_dRPA` (ab-initio ring resummation) | `E_disp^CKS` (TDDFT monomer response) |
//! | uncoupled term | `2·E_OS` (supermolecular) | `E_disp^UCHF` (same Casimir–Polder machinery, fxc off) |
//! | partitioning | supermolecular, recovered by CP subtraction | SAPT-style, monomer responses |
//! | functional | none | required (enters `f_xc`) |
//!
//! The dispersion terms here are **intrinsically interaction energies** — they
//! come from a monomer-A × monomer-B trace and vanish at infinite separation.
//! `tests/cks_dispersion_separation.rs` scans He₂/cc-pVDZ and measures a factor
//! of 64.00 per doubling from 6 Å to 24 Å (68.95 from 3 Å to 6 Å, where overlap
//! still contributes), i.e. `R⁻⁶`.
//!
//! # Auxiliary-index alignment
//!
//! The trace pairs auxiliary index `P` of monomer A with auxiliary index `P` of
//! monomer B, and each monomer's auxiliary basis is ordered by its own XYZ atom
//! order. Both CP fragments must therefore list every center — real and ghost —
//! in the same order, differing only in which centers carry the `@` marker.
//! Fragments written own-real-atoms-first have identical `naux` and permuted
//! aux indices, and the cross-trace then collapses into a monomer self-trace
//! (flat in `R`, about 10⁴× too large on He₂).
//! [`crate::cks_dispersion::cks_dispersion_energy`] compares the two monomers'
//! per-function auxiliary centers and refuses a misaligned pair.
//!
//! # Why both dispersion terms come from the same machinery
//!
//! `E_disp^UCHF` is computed by the *same* [`crate::cks_dispersion`] code path as
//! `E_disp^CKS`, with the coupling switched to [`CksCoupling::Uncoupled`], so
//! any shared error — the frequency grid, the RI auxiliary basis, the
//! aux-basis dressing — cancels in the correction to the extent it is shared.
//!
//! # Functional choice
//!
//! The published MP2C dispersion correction uses the **ALDAX** kernel
//! (adiabatic LDA, exchange-only). ferric's CKS kernel is LDA-family only:
//! `xc = "LDA_X"` is the published choice; `xc = "LDA"` (X+C, the default)
//! adds the VWN correlation kernel. The published protocol also builds the
//! monomer response on asymptotically corrected KS orbitals and applies the
//! correction to an MP2/CBS baseline; neither is implemented here.
//!
//! # Scope
//!
//! Library-level interaction energies only, closed-shell restricted references.
//! No CLI, no Python bindings, no gradients. Requires an explicit CP triple
//! (dimer + both monomers in the dimer basis).

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_scf::ScfResult;

use crate::cks_dispersion::{
    build_monomer_response, cks_dispersion_energy, CksCoupling, CksDispersionConfig,
};

/// Configuration for [`mp2c_cks_interaction`].
#[derive(Debug, Clone)]
pub struct Mp2cCksConfig {
    /// Functional supplying the adiabatic `f_xc` for the CKS response.
    ///
    /// Must be LDA-family (see the module docs); anything else is refused with a
    /// clean error rather than silently approximated. Default `"LDA"`.
    pub xc: String,
    /// Frequency-grid and coupling knobs shared by both dispersion evaluations.
    ///
    /// `coupling` is **overwritten** by the driver — it evaluates the uncoupled
    /// and CKS terms itself and would otherwise be able to disagree with its own
    /// construction. Everything else (`n_freq`, `u0`, `frozen_core`) is honored
    /// and, critically, is applied *identically* to both terms.
    pub disp: CksDispersionConfig,
    /// Frozen-core orbitals for the supermolecular `E_MP2` term. Kept separate
    /// from `disp.frozen_core` only so the two can be reported independently;
    /// [`mp2c_cks_interaction`] requires them to agree, because a mismatch
    /// would mean the MP2 term and the dispersion terms correlate different
    /// orbital spaces.
    pub frozen_core: usize,
}

impl Default for Mp2cCksConfig {
    fn default() -> Self {
        Self {
            xc: "LDA".to_string(),
            disp: CksDispersionConfig::default(),
            frozen_core: 0,
        }
    }
}

impl Mp2cCksConfig {
    fn validate(&self) -> Result<(), FerricError> {
        if self.frozen_core != self.disp.frozen_core {
            return Err(FerricError::General(format!(
                "mp2c_cks: frozen_core mismatch — MP2 term uses {} but the dispersion terms use \
                 {}. These must agree; a mismatch means the subtracted uncoupled dispersion and \
                 the MP2 energy it is subtracted from describe different orbital spaces.",
                self.frozen_core, self.disp.frozen_core
            )));
        }
        Ok(())
    }
}

/// One monomer of a counterpoise triple: geometry, bases, and a converged SCF.
///
/// Both monomers must be **in the dimer basis** (ghost centers at the partner's
/// positions) and share the same auxiliary basis, so their responses live in a
/// common auxiliary space. [`cks_dispersion_energy`] enforces the dimension
/// match; the physical correctness of the ghost placement is the caller's.
pub struct MonomerInput<'a> {
    /// Monomer geometry with ghost centers at the partner's positions.
    pub mol: &'a Molecule,
    /// Orbital basis over the dimer's centers.
    pub obs: &'a PreparedBasis,
    /// Auxiliary (RI) basis over the dimer's centers, in the dimer's order.
    pub dfbs: &'a PreparedBasis,
    /// Converged closed-shell SCF. For a genuine CKS response this should be a
    /// **Kohn–Sham** reference; an RHF reference yields a coupled-Hartree–Fock
    /// response, which is a different object.
    pub scf: &'a ScfResult,
}

/// Full MP2C interaction-energy breakdown, all in Hartree.
#[derive(Debug, Clone)]
pub struct Mp2cCksInteraction {
    /// `ΔE_MP2 − E_disp^UCHF + E_disp^CKS` — the MP2C interaction energy.
    pub e_int_mp2c: f64,
    /// Counterpoise-corrected supermolecular MP2 interaction energy.
    pub e_int_mp2: f64,
    /// `E_disp^UCHF`, the uncoupled dispersion that is removed (negative).
    pub e_disp_uncoupled: f64,
    /// `E_disp^CKS`, the coupled dispersion that is added back (negative).
    pub e_disp_cks: f64,
    /// `e_disp_cks − e_disp_uncoupled`, the net MP2C correction.
    ///
    /// Published MP2C corrections are repulsive for most complexes but
    /// negative for some small ones (methane and ethene dimers); see the
    /// module docs for the open sign discrepancy.
    pub e_correction: f64,
    /// Counterpoise-corrected Hartree–Fock/SCF interaction energy.
    pub e_int_scf: f64,
    /// Smallest eigenvalue of the static CKS Dyson matrix over both monomers
    /// (see [`crate::cks_dispersion::CksDispersionResult`]). `≤ 0` means
    /// `e_disp_cks`, and therefore the correction, is not meaningful.
    pub static_dyson_min_eigenvalue: f64,
}

impl Mp2cCksInteraction {
    /// Hartree → kcal/mol.
    pub const HARTREE_TO_KCAL: f64 = 627.509_474_06;

    /// `e_int_mp2c` in kcal/mol — the unit A24/S22/S66 references use.
    pub fn e_int_mp2c_kcal(&self) -> f64 {
        self.e_int_mp2c * Self::HARTREE_TO_KCAL
    }

    /// `e_int_mp2` in kcal/mol.
    pub fn e_int_mp2_kcal(&self) -> f64 {
        self.e_int_mp2 * Self::HARTREE_TO_KCAL
    }
}

/// MP2C interaction energy with a genuine CKS coupled-dispersion term.
///
/// ```text
///     ΔE_MP2C = ΔE_MP2(CP) − E_disp^UCHF + E_disp^CKS
/// ```
///
/// # Arguments
///
/// The counterpoise triple. All three must use the **same orbital and auxiliary
/// basis over the same union of centers**, with the monomers carrying ghost
/// atoms at the partner fragment's positions.
///
/// # Errors
///
/// Non-restricted references, a frozen-core mismatch between the MP2 and
/// dispersion halves, a non-LDA functional, or an auxiliary-space mismatch
/// between the two monomers.
pub fn mp2c_cks_interaction(
    dimer_mol: &Molecule,
    dimer_obs: &PreparedBasis,
    dimer_dfbs: &PreparedBasis,
    dimer_scf: &ScfResult,
    mono_a: &MonomerInput<'_>,
    mono_b: &MonomerInput<'_>,
    cfg: &Mp2cCksConfig,
) -> Result<Mp2cCksInteraction, FerricError> {
    use ferric_integrals::operator::Operator;
    use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
    use ferric_scf::result::Spin;

    cfg.validate()?;
    for (label, s) in [
        ("dimer", dimer_scf),
        ("monomer A", mono_a.scf),
        ("monomer B", mono_b.scf),
    ] {
        if !matches!(s.spin, Spin::Restricted) {
            return Err(FerricError::General(format!(
                "mp2c_cks_interaction: {label} reference is {:?}, but only closed-shell \
                 restricted references are supported.",
                s.spin
            )));
        }
    }

    // --- Supermolecular MP2 interaction energy (counterpoise-corrected) ------
    let ri_cfg = RiMp2Config {
        frozen_core: cfg.frozen_core,
        ..Default::default()
    };
    let mp2_corr = |mol: &Molecule, obs: &PreparedBasis, dfbs: &PreparedBasis, scf: &ScfResult| {
        ri_mp2_spin_components(mol, obs, dfbs, Operator::coulomb(), scf, &ri_cfg)
            .map(|(sc, _)| sc.e_total)
    };
    let e_mp2_d = mp2_corr(dimer_mol, dimer_obs, dimer_dfbs, dimer_scf)?;
    let e_mp2_a = mp2_corr(mono_a.mol, mono_a.obs, mono_a.dfbs, mono_a.scf)?;
    let e_mp2_b = mp2_corr(mono_b.mol, mono_b.obs, mono_b.dfbs, mono_b.scf)?;

    let e_int_scf = dimer_scf.energy - mono_a.scf.energy - mono_b.scf.energy;
    let e_int_mp2 = e_int_scf + (e_mp2_d - e_mp2_a - e_mp2_b);

    // --- Dispersion terms, both from the SAME machinery ---------------------
    //
    // Building the monomer responses is the expensive step and it is
    // coupling-independent for the bubble itself, but the CKS level additionally
    // needs the projected f_xc. Rather than build the responses twice, build
    // them once at the CKS level (which carries the kernel) and evaluate the
    // uncoupled trace from the same objects — `CksCoupling::Uncoupled` simply
    // never consults `fxc_aux`. This guarantees the two terms share bit-identical
    // b_ov / orbital energies / aux dressing, which is exactly the shared-error
    // cancellation the module docs argue for.
    let cks_cfg = CksDispersionConfig {
        coupling: CksCoupling::Cks,
        ..cfg.disp.clone()
    };
    let resp_a = build_monomer_response(
        mono_a.mol,
        mono_a.obs,
        mono_a.dfbs,
        mono_a.scf,
        &cfg.xc,
        &cks_cfg,
    )?;
    let resp_b = build_monomer_response(
        mono_b.mol,
        mono_b.obs,
        mono_b.dfbs,
        mono_b.scf,
        &cfg.xc,
        &cks_cfg,
    )?;

    let unc_cfg = CksDispersionConfig {
        coupling: CksCoupling::Uncoupled,
        ..cfg.disp.clone()
    };
    let e_disp_uncoupled = cks_dispersion_energy(&resp_a, &resp_b, &unc_cfg)?.e_disp;
    let cks = cks_dispersion_energy(&resp_a, &resp_b, &cks_cfg)?;
    let e_disp_cks = cks.e_disp;

    let e_correction = e_disp_cks - e_disp_uncoupled;

    Ok(Mp2cCksInteraction {
        e_int_mp2c: e_int_mp2 + e_correction,
        e_int_mp2,
        e_disp_uncoupled,
        e_disp_cks,
        e_correction,
        e_int_scf,
        static_dyson_min_eigenvalue: cks.static_dyson_min_eigenvalue.unwrap_or(f64::NAN),
    })
}
