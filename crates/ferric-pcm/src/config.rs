//! PCM configuration surface, threaded through the SCF configs (analogous
//! to `ferric_core::external_potential::ExternalPotential`).

use serde::{Deserialize, Serialize};

use crate::cavity::CavityConfig;

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
    /// Scale factor applied to Bondi van der Waals radii when building the
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

impl PcmConfig {
    /// Water at room temperature — the standard PCM validation solvent.
    pub fn water() -> Self {
        Self {
            epsilon: 78.4,
            vdw_scale: default_vdw_scale(),
            lebedev_order: default_lebedev_order(),
            inner_iters: default_inner_iters(),
        }
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
    fn defaults_fill_in_when_omitted() {
        let toml_str = "epsilon = 78.4\n";
        let cfg: PcmConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.vdw_scale, 1.2);
        assert_eq!(cfg.lebedev_order, 110);
        assert_eq!(cfg.inner_iters, 1);
    }
}
