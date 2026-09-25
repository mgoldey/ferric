//! PCM configuration surface, threaded through the SCF configs (analogous
//! to `ferric_core::external_potential::ExternalPotential`).

use serde::{Deserialize, Serialize};

use ferric_core::FerricError;

use crate::cavity::{CavityConfig, Tessera};
use crate::matrices::SdKind;

/// Named solvents and their dielectric constants at 298 K, the ONE table
/// behind [`PcmConfig::for_solvent`] (and so behind both the CLI's `[pcm]
/// solvent` key and the Python `solvent=` kwarg). Names are matched
/// case-insensitively. `"dcm"` is an alias of `"dichloromethane"`.
pub const NAMED_SOLVENTS: &[(&str, f64)] = &[
    ("water", 78.4),
    ("dmso", 46.7),
    ("methanol", 32.6),
    ("ethanol", 24.9),
    ("acetone", 20.7),
    ("dichloromethane", 8.93),
    ("dcm", 8.93),
    ("thf", 7.43),
    ("chloroform", 4.71),
    ("toluene", 2.38),
    ("hexane", 1.88),
];

/// Lebedev orders the PCM cavity (`cavity::build_cavity`) and the
/// Gaussian-smeared S/D ξ table support.
pub const SUPPORTED_LEBEDEV_ORDERS: [usize; 6] = [6, 14, 26, 50, 110, 302];

/// Dielectric constant of a named solvent from [`NAMED_SOLVENTS`]
/// (case-insensitive, surrounding whitespace ignored), or `None`.
pub fn solvent_epsilon(name: &str) -> Option<f64> {
    let key = name.trim().to_ascii_lowercase();
    NAMED_SOLVENTS
        .iter()
        .find(|(n, _)| *n == key)
        .map(|(_, eps)| *eps)
}

/// Configuration for an IEF-PCM implicit-solvent calculation.
///
/// `None` (in the consuming `RhfConfig.pcm: Option<PcmConfig>` field) means
/// "no solvent" and MUST be byte-identical to a vacuum calculation — see
/// the `pcm_none_matches_vacuum_*` regression tests in `ferric-scf`.
///
/// `#[serde(deny_unknown_fields)]` per the repo's config-honesty convention:
/// a typo'd TOML key must hard-error, never silently no-op.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PcmConfig {
    /// Solvent dielectric constant (must be > 1.0; e.g. water = 78.4).
    pub epsilon: f64,
    /// Scale factor applied to modified Bondi van der Waals radii when building the
    /// cavity spheres. Default 1.2 (matches PySCF/Q-Chem convention).
    #[serde(default = "default_vdw_scale")]
    pub vdw_scale: f64,
    /// Lebedev order used to tessellate each atomic sphere (6, 14, 26, 50,
    /// 110, or 302). Default 110.
    #[serde(default = "default_lebedev_order")]
    pub lebedev_order: usize,
    /// Maximum number of outer SCF-coupled PCM re-solves per SCF iteration.
    /// PCM charges are solved exactly (LAPACK, not a fixed-point loop) from
    /// the CURRENT density each SCF iteration, so a value of 1 (default) is
    /// standard practice (PySCF/Psi4 default): the outer SCF/DIIS loop is
    /// itself the self-consistency driver between q and D. A value > 1 is
    /// accepted for experimentation but is not required for correctness in
    /// the current implementation (see `pcm.rs`'s per-iteration hook).
    #[serde(default = "default_inner_iters")]
    pub inner_iters: usize,
    /// Boundary-element S/D matrix formulation. Default
    /// [`SdKind::GaussianSmeared`] (PySCF `pcm.py` convention, added
    /// 2026-07-19 -- see `matrices.rs`'s module doc). [`SdKind::PointCharge`]
    /// reproduces the original bare-point-charge formula this crate shipped
    /// with.
    #[serde(default)]
    pub sd_kind: SdKind,
    /// How the solute potential at a tessera, and the reaction-field
    /// operator from the tessera charges, are evaluated. Default
    /// [`ProbeKind::Point`]. [`ProbeKind::GaussianSmeared`] is PySCF
    /// `pcm.py`'s convention (see [`ProbeKind`]).
    #[serde(default)]
    pub probe: ProbeKind,
    /// An explicit cavity to use instead of building one from the molecule.
    /// `None` (the default) builds the cavity from `vdw_scale` and
    /// `lebedev_order`. `Some(tess)` uses `tess` as given and IGNORES
    /// `vdw_scale` and `lebedev_order`; every [`Tessera`] field that the S/D
    /// matrices read (`position`, `normal`, `area`, `sphere_radius`,
    /// `charge_exp`, `switch_fun`) must be set. This exists so another
    /// code's cavity can be injected to test the solver separately from the
    /// cavity construction (`crates/ferric-scf/tests/validation_pcm.rs`).
    /// Not settable from TOML.
    #[serde(skip)]
    pub cavity: Option<Vec<Tessera>>,
}

/// How the tessera charges couple to the solute's charge density.
///
/// * [`ProbeKind::Point`]: each tessera is a point charge. The potential at
///   tessera `k` is `Σ_A Z_A/|r_k − R_A| − Σ_μν D_μν ⟨μ|1/|r − r_k||ν⟩` and
///   the reaction-field operator is `Σ_k q_k ⟨μ|−1/|r − r_k||ν⟩`.
/// * [`ProbeKind::GaussianSmeared`]: each tessera is a normalized Gaussian
///   charge of exponent `charge_exp²` (the same Gaussian the
///   [`SdKind::GaussianSmeared`] S/D matrices assume), so every `1/r` above
///   becomes `erf(ξ_k r)/r`. This is PySCF `pcm.py`'s `_get_v`/`_get_vmat`/
///   `v_grids_n` (int3c2e/int2c2e against `fakemol_for_charges(expnt=ξ²)`).
///   It changes the solvation energy by ~1e-6 Ha on water and NH3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ProbeKind {
    #[default]
    Point,
    GaussianSmeared,
}

fn default_vdw_scale() -> f64 {
    1.2
}
fn default_lebedev_order() -> usize {
    110
}
fn default_inner_iters() -> usize {
    1
}

impl Default for PcmConfig {
    /// Defaults to water (ε = 78.4) — the most common PCM solvent.
    fn default() -> Self {
        Self::water()
    }
}

impl PcmConfig {
    /// Water at room temperature — the standard PCM validation solvent.
    pub fn water() -> Self {
        Self {
            epsilon: 78.4,
            vdw_scale: default_vdw_scale(),
            lebedev_order: default_lebedev_order(),
            inner_iters: default_inner_iters(),
            sd_kind: SdKind::default(),
            probe: ProbeKind::default(),
            cavity: None,
        }
    }

    /// IEF-PCM in the named solvent, every other knob at its default.
    ///
    /// The single solvent table shared by the CLI's `[pcm] solvent` key and
    /// the Python `solvent=` kwarg ([`NAMED_SOLVENTS`], dielectric constants
    /// at 298 K; case-insensitive). An unrecognised name is an ERROR, never a
    /// silent fallback to vacuum or to water: either would look like a
    /// successful solvated run of a different solvent.
    pub fn for_solvent(name: &str) -> Result<Self, FerricError> {
        let epsilon = solvent_epsilon(name).ok_or_else(|| {
            let known: Vec<&str> = NAMED_SOLVENTS.iter().map(|(n, _)| *n).collect();
            FerricError::General(format!(
                "solvent '{name}' not recognised; known solvents: {}. Give a dielectric \
                 constant directly instead for any other solvent",
                known.join(", ")
            ))
        })?;
        Self::with_epsilon(epsilon)
    }

    /// IEF-PCM at dielectric constant `epsilon`, every other knob at its
    /// default. `epsilon` must be finite and `> 1.0` (1.0 is vacuum). NaN and
    /// infinity are refused explicitly: `NaN <= 1.0` and `inf <= 1.0` are both
    /// false, so a bare comparison would let them through to fail later inside
    /// the cavity solve.
    pub fn with_epsilon(epsilon: f64) -> Result<Self, FerricError> {
        if !epsilon.is_finite() || epsilon <= 1.0 {
            return Err(FerricError::General(format!(
                "PCM dielectric must be > 1.0 and finite, got {epsilon} (vacuum is 1.0; \
                 omit the solvent for no solvation)"
            )));
        }
        Ok(Self {
            epsilon,
            ..Self::water()
        })
    }

    /// Set the per-sphere Lebedev order, refusing any order outside
    /// [`SUPPORTED_LEBEDEV_ORDERS`] (the set the cavity and Gaussian-ξ tables
    /// support) up front rather than as an error from inside SCF setup.
    pub fn with_lebedev_order(mut self, order: usize) -> Result<Self, FerricError> {
        if !SUPPORTED_LEBEDEV_ORDERS.contains(&order) {
            return Err(FerricError::General(format!(
                "PCM lebedev_order must be one of {SUPPORTED_LEBEDEV_ORDERS:?}; got {order}"
            )));
        }
        self.lebedev_order = order;
        Ok(self)
    }

    pub(crate) fn cavity_config(&self) -> CavityConfig {
        CavityConfig {
            vdw_scale: self.vdw_scale,
            lebedev_order: self.lebedev_order,
            skip_ghost_atoms: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deny_unknown_fields_rejects_typo() {
        let toml_str = r#"
            epsilon = 78.4
            vdwscale = 1.2
        "#;
        let result: Result<PcmConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err(), "typo'd key 'vdwscale' should be rejected");
    }

    #[test]
    fn for_solvent_matches_the_table_and_is_case_insensitive() {
        assert_eq!(PcmConfig::for_solvent("water").unwrap().epsilon, 78.4);
        assert_eq!(PcmConfig::for_solvent("  Water ").unwrap().epsilon, 78.4);
        assert_eq!(
            PcmConfig::for_solvent("DCM").unwrap().epsilon,
            PcmConfig::for_solvent("dichloromethane").unwrap().epsilon
        );
        // Every other knob stays at the water() default.
        let c = PcmConfig::for_solvent("toluene").unwrap();
        let w = PcmConfig::water();
        assert_eq!(c.epsilon, 2.38);
        assert_eq!(c.vdw_scale, w.vdw_scale);
        assert_eq!(c.lebedev_order, w.lebedev_order);
        assert_eq!(c.inner_iters, w.inner_iters);
    }

    #[test]
    fn unknown_solvent_is_an_error_naming_it() {
        let e = PcmConfig::for_solvent("watr").unwrap_err().to_string();
        assert!(e.contains("'watr' not recognised"), "{e}");
        assert!(
            e.contains("water"),
            "the message must list the known names: {e}"
        );
    }

    #[test]
    fn with_epsilon_refuses_vacuum_and_non_finite() {
        for eps in [1.0, 0.5, -3.0, f64::NAN, f64::INFINITY] {
            assert!(PcmConfig::with_epsilon(eps).is_err(), "{eps} accepted");
        }
        assert_eq!(PcmConfig::with_epsilon(4.0).unwrap().epsilon, 4.0);
    }

    #[test]
    fn with_lebedev_order_refuses_unsupported_orders() {
        for order in [0, 7, 194] {
            assert!(PcmConfig::water().with_lebedev_order(order).is_err());
        }
        for order in SUPPORTED_LEBEDEV_ORDERS {
            assert_eq!(
                PcmConfig::water()
                    .with_lebedev_order(order)
                    .unwrap()
                    .lebedev_order,
                order
            );
        }
    }

    #[test]
    fn defaults_fill_in_when_omitted() {
        let toml_str = "epsilon = 78.4\n";
        let cfg: PcmConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.vdw_scale, 1.2);
        assert_eq!(cfg.lebedev_order, 110);
        assert_eq!(cfg.inner_iters, 1);
        assert_eq!(cfg.probe, ProbeKind::Point);
        assert!(cfg.cavity.is_none());
    }
}
