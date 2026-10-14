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
//! # The model (measured; table and fit in the PR that added this module)
//!
//! `floor = rounding + screening`, where
//!
//! * `rounding = ROUND_COEFF * eps * |E| * sqrt(nbf)` is the floating-point
//!   floor of the energy sum. It is what remains with density-fitted J and
//!   in the open-shell solvers (RI-J alkanes C10/C20 and benzene: 3e-13 to
//!   1.3e-12 Ha, within 25 % of this formula).
//! * `screening = SCREEN_COEFF * integral_thresh * nbf^2` is added ONLY for the
//!   closed-shell DIRECT path (conventional J and K, default K builder). That
//!   path updates the Fock matrix incrementally from the density change
//!   (`rhf.rs`, full rebuild every 8th iteration); each update drops quartets
//!   whose bound falls below `integral_thresh` for the SMALL density change, and
//!   the dropped pieces accumulate until the next full rebuild. The result is
//!   100-1000x above the rounding floor, grows linearly with `integral_thresh`
//!   (alkane C10/STO-3G: 4.8e-7, 2.4e-8, 9.1e-11, 1.1e-12 Ha at 1e-8, 1e-10,
//!   1e-12, 1e-14) and with the quartet count (`~sqrt(nbf^4)`), does NOT grow with
//!   `|E|` (HBr, E = -2573, is quieter than SiH4, E = -291, at equal `nbf`), and
//!   is bit-identical across 1, 2 and 4 threads.
//!
//! It is an ESTIMATE good to a factor of a few (the windowed RMS of one run
//! varies by ~3x between windows; see [`WARN_FACTOR`]), not a bound. Functional
//! (grid) and non-default K-builder (LinK, COSX) runs are outside what was
//! measured; [`floor_for_config`] returns `None` for them rather than guess.
//!
//! The functions here are pure and nothing in the solvers reads them back, so
//! energies and iteration counts are unchanged.

use ferric_core::mol::Molecule;

/// Coefficient of the screening term `SCREEN_COEFF * integral_thresh * nbf^2`.
pub const SCREEN_COEFF: f64 = 0.022;
/// Coefficient of the rounding term `ROUND_COEFF * eps * |E| * sqrt(nbf)`.
pub const ROUND_COEFF: f64 = 0.5;
/// `energy_conv` is reported as below the floor when `energy_conv < WARN_FACTOR * floor`.
pub const WARN_FACTOR: f64 = 0.3;

/// Which solver family runs the SCF. Only the closed-shell solver updates the
/// direct Fock matrix incrementally by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverFamily {
    /// `solve_rhf` (RHF and the closed-shell references of the post-HF kinds).
    ClosedShell,
    /// `solve_uhf` / `solve_rohf`.
    OpenShell,
}

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
    /// Whether the Fock matrix is built by the incremental direct J/K path
    /// (the only source of the screening term).
    pub incremental_direct: bool,
}

/// Estimated RMS of `|dE|` once the density has settled, in Hartree.
pub fn de_noise_floor(i: &NoiseFloorInputs) -> f64 {
    let nbf = i.nbf.max(1) as f64;
    let rounding = ROUND_COEFF * f64::EPSILON * i.e_abs.abs() * nbf.sqrt();
    let screening = if i.incremental_direct {
        SCREEN_COEFF * i.integral_thresh.max(0.0) * nbf * nbf
    } else {
        0.0
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

/// Whether the floor model covers this configuration. Functional (grid) runs
/// and the LinK / COSX exchange builders add noise sources that were not
/// measured, so no claim is made for them.
pub fn is_characterised(config: &crate::rhf::RhfConfig) -> bool {
    config.xc.is_none() && matches!(config.k_builder.as_deref(), None | Some("direct"))
}

/// True when a density-fitting aux request means "fit" (`None` and the empty
/// opt-out sentinel do not).
pub fn aux_is_fitted(aux: Option<&str>) -> bool {
    aux.is_some_and(|s| !s.is_empty())
}

/// Whether the closed-shell solver will take the incremental direct J/K path
/// for this configuration, mirroring `solve_rhf`'s own condition (no density
/// fitting of either matrix, default K builder, incremental Fock not disabled
/// by `FERRIC_SCF_INCREMENTAL`). Always false for the open-shell solvers.
pub fn incremental_direct_expected(config: &crate::rhf::RhfConfig, family: SolverFamily) -> bool {
    family == SolverFamily::ClosedShell
        && is_characterised(config)
        && !aux_is_fitted(config.df_j_aux.as_deref())
        && !aux_is_fitted(config.df_k_aux.as_deref())
        && crate::rhf::closed_shell_incremental_enabled()
}

/// The floor for a configuration, or `None` outside the measured scope
/// ([`is_characterised`]). `incremental_direct` is whether the incremental
/// direct path is (or will be) in use.
pub fn floor_for_config(
    config: &crate::rhf::RhfConfig,
    e_abs: f64,
    nbf: usize,
    incremental_direct: bool,
) -> Option<f64> {
    is_characterised(config).then(|| {
        de_noise_floor(&NoiseFloorInputs {
            e_abs,
            nbf,
            integral_thresh: config.integral_thresh,
            incremental_direct,
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
/// the first iteration's energy would need a side channel out of the solver
/// for no gain in accuracy.)
pub fn energy_conv_warning(
    mol: &Molecule,
    nbf: usize,
    config: &crate::rhf::RhfConfig,
    family: SolverFamily,
) -> Option<String> {
    let floor = floor_for_config(
        config,
        estimate_total_energy_abs(mol),
        nbf,
        incremental_direct_expected(config, family),
    )?;
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
