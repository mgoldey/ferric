//! Estimated noise floor of the SCF energy change `|dE|`, and the advisory
//! "your `energy_conv` is below it" warning (issue #410).
//!
//! # What the floor is
//!
//! Once the density has settled, `dE = E_n - E_(n-1)` no longer shrinks: it
//! jitters at a level set by the numerics, not by the physics. A tolerance
//! `energy_conv` below that level is met only by chance, so the SCF runs on
//! until the jitter happens to dip under it (12-27 extra iterations on the
//! 71-atom case of #410).
//!
//! Two measured contributions (data and fit: PR description of this module;
//! `FERRIC_SCF_TRACE` runs at `density_conv = 1e-12`, `energy_conv = 0`):
//!
//! * **Integral screening** (conventional J, the dominant term). The direct
//!   builder skips a quartet when `Q_ab * Q_cd * max|D| <= integral_thresh`;
//!   as the density moves in its last digits, borderline quartets flip in and
//!   out and each flip moves the energy by up to ~`integral_thresh`. The RMS of
//!   `dE` scales linearly with `integral_thresh` (alkane C10/STO-3G: 4.8e-7,
//!   2.4e-8, 9.1e-11, 1.1e-12 at 1e-8, 1e-10, 1e-12, 1e-14) and grows with the
//!   number of quartets (~`sqrt(N_quartets)`, i.e. ~`nbf^1.7`). It does NOT grow with `|E|`
//!   (HBr, E = -2573, is quieter than SiH4, E = -291, at equal `nbf`), and it is
//!   bit-identical across 1, 2 and 4 threads. Density-fitted J removes it.
//! * **Rounding** of the energy sum, `~ a few * eps * |E|`, the floor that
//!   remains with density-fitted J.
//!
//! The estimate is the SUM of the two (screening term only when J is
//! conventional). It is an estimate good to a factor of a few, not a bound:
//! see [`WARN_FACTOR`] for how that is accounted for.
//!
//! The functions here are pure: nothing reads them back into the SCF, so
//! energies and iteration counts are unchanged.

use ferric_core::mol::Molecule;

/// Coefficient of the screening term `SCREEN_COEFF * integral_thresh * nbf^SCREEN_EXP`.
pub const SCREEN_COEFF: f64 = 0.05;
/// Exponent of `nbf` in the screening term.
pub const SCREEN_EXP: f64 = 1.7;
/// Coefficient of the rounding term `ROUND_COEFF * eps * |E| * sqrt(nbf)`.
pub const ROUND_COEFF: f64 = 0.5;
/// `energy_conv` is reported as below the floor when `energy_conv < WARN_FACTOR * floor`.
pub const WARN_FACTOR: f64 = 0.3;

/// Inputs of the floor estimate, all known before the first SCF iteration
/// except `e_abs`, which can be a rough estimate (see [`estimate_total_energy_abs`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseFloorInputs {
    /// `|E_total|` in Hartree (rounding term only).
    pub e_abs: f64,
    /// Number of basis functions.
    pub nbf: usize,
    /// Direct-SCF integral screening threshold (`RhfConfig::integral_thresh`).
    pub integral_thresh: f64,
    /// Whether the Coulomb matrix is density-fitted (no screening noise).
    pub j_fitted: bool,
}

/// Estimated RMS of `|dE|` once the density has settled, in Hartree.
pub fn de_noise_floor(i: &NoiseFloorInputs) -> f64 {
    let nbf = i.nbf.max(1) as f64;
    let rounding = ROUND_COEFF * f64::EPSILON * i.e_abs.abs() * nbf.sqrt();
    let screening = if i.j_fitted {
        0.0
    } else {
        SCREEN_COEFF * i.integral_thresh.max(0.0) * nbf.powf(SCREEN_EXP)
    };
    rounding + screening
}

/// True when `energy_conv` is low enough that the `dE` criterion is met only
/// by chance: `energy_conv < WARN_FACTOR * floor`. A non-positive or NaN
/// `energy_conv` is below any positive floor and also unreachable, so it counts.
pub fn energy_conv_below_floor(energy_conv: f64, floor: f64) -> bool {
    !(energy_conv >= WARN_FACTOR * floor)
}

/// Crude `|E_total|` from the nuclear charges alone (Thomas-Fermi atom,
/// `0.7687 Z^(7/3)` Hartree), good to ~15 % for Z <= 54. Only the rounding
/// term reads it, and that term is a few eps*|E|, so the accuracy is immaterial.
pub fn estimate_total_energy_abs(mol: &Molecule) -> f64 {
    mol.atoms
        .iter()
        .filter(|a| !a.ghost)
        .map(|a| 0.7687 * f64::from(a.z.max(0)).powf(7.0 / 3.0))
        .sum()
}

/// Whether the floor model covers this configuration. Functional (grid) and
/// COSX (seminumerical K) runs add noise sources that were not measured, so no
/// claim is made for them.
pub fn is_characterised(config: &crate::rhf::RhfConfig) -> bool {
    config.xc.is_none() && config.k_builder.as_deref() != Some("cosx")
}

/// True when a Coulomb aux request means "fit J" (`None` and the empty
/// opt-out sentinel do not).
pub fn j_aux_is_fitted(aux: Option<&str>) -> bool {
    aux.is_some_and(|s| !s.is_empty())
}

/// The floor for a configuration, or `None` outside the measured scope
/// ([`is_characterised`]). `j_fitted` is the EFFECTIVE Coulomb treatment (the
/// solver's resolved aux name, not the raw request).
pub fn floor_for_config(
    config: &crate::rhf::RhfConfig,
    e_abs: f64,
    nbf: usize,
    j_fitted: bool,
) -> Option<f64> {
    is_characterised(config).then(|| {
        de_noise_floor(&NoiseFloorInputs {
            e_abs,
            nbf,
            integral_thresh: config.integral_thresh,
            j_fitted,
        })
    })
}

/// The advisory text for a user-supplied `energy_conv`, or `None` when the
/// setting is above the estimated floor or the configuration is outside the
/// measured scope.
///
/// Called ONCE, before the first SCF iteration, by the Python and CLI front
/// ends: `nbf` and the molecule are known by then, and the one input that is
/// not (`|E|`) enters only the rounding term, where the nuclear-charge
/// estimate ([`estimate_total_energy_abs`]) is accurate enough. (Waiting for
/// the first iteration's energy would warn only after the first Fock build,
/// and would need a side channel out of the solver for no gain in accuracy.)
pub fn energy_conv_warning(
    mol: &Molecule,
    nbf: usize,
    config: &crate::rhf::RhfConfig,
) -> Option<String> {
    let j_fitted = j_aux_is_fitted(config.df_j_aux.as_deref());
    let floor = floor_for_config(config, estimate_total_energy_abs(mol), nbf, j_fitted)?;
    energy_conv_below_floor(config.energy_conv, floor)
        .then(|| floor_message(config.energy_conv, floor, nbf))
}

/// The clause shared by the pre-SCF warning and the exit report.
pub fn floor_sentence(energy_conv: f64, floor: f64) -> String {
    format!(
        "energy_conv {energy_conv:.1e} is below the estimated noise floor of dE (~{floor:.1e} Ha \
         for this system), so the dE criterion can be met only by chance"
    )
}

/// The full advisory.
pub fn floor_message(energy_conv: f64, floor: f64, nbf: usize) -> String {
    format!(
        "{} ({nbf} basis functions). The SCF is judged on the density change (density_conv); \
         dE is only a loose not-still-descending bound, so a tighter energy_conv adds \
         iterations without improving the result. Use an energy_conv above ~{:.0e}, or the \
         default.",
        floor_sentence(energy_conv, floor),
        floor
    )
}
