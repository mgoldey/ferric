//! [`ScfOptionsExt`](crate::options::ScfOptionsExt): the SCF knobs only the
//! CLI `[scf]` section lowers today, layered over
//! [`ScfOptions`](crate::options::ScfOptions).

use ferric_core::FerricError;

use super::ScfOptions;
use crate::rhf::RhfConfig;

/// [`ScfOptions`] plus the SCF knobs that only the CLI `[scf]` section
/// exposes today. Lowered by [`ScfOptionsExt::into_rhf_config`], which runs
/// [`ScfOptions::into_rhf_config`] and then applies these fields.
///
/// A separate type rather than new [`ScfOptions`] fields, so the Python entry
/// points' exhaustive `ScfOptions` literals are untouched by the CLI
/// migration; when those literals are next edited, these fields can move
/// into `ScfOptions` itself. As there, `None` keeps the
/// [`RhfConfig::default`] value of the field it lowers to.
#[derive(Debug, Clone, Default)]
pub struct ScfOptionsExt {
    /// The shared options.
    pub base: ScfOptions,
    /// [`RhfConfig::diis_switch_thresh`].
    pub diis_switch_thresh: Option<f64>,
    /// [`RhfConfig::ri_j_task`]: the task class an unset
    /// [`RhfConfig::ri_j_precision`] resolves against. A frontend derives it
    /// from the task it runs; it is not a user knob.
    pub ri_j_task: Option<crate::df_j::RiJTask>,
    /// [`RhfConfig::screening`].
    pub screening: Option<crate::screening::ScreeningKind>,
    /// [`RhfConfig::verbose`].
    pub verbose: Option<bool>,
}

impl ScfOptionsExt {
    /// Lower onto [`RhfConfig::default`]: [`ScfOptions::into_rhf_config`]
    /// for [`ScfOptionsExt::base`], then each `Some` field here copied into
    /// its `RhfConfig` field. Pure: no I/O, no environment.
    pub fn into_rhf_config(self) -> Result<RhfConfig, FerricError> {
        // Exhaustive destructure, as in `ScfOptions::into_rhf_config`.
        let ScfOptionsExt {
            base,
            diis_switch_thresh,
            ri_j_task,
            screening,
            verbose,
        } = self;
        let mut c = base.into_rhf_config()?;
        if let Some(v) = diis_switch_thresh {
            c.diis_switch_thresh = v;
        }
        if let Some(v) = ri_j_task {
            c.ri_j_task = v;
        }
        if let Some(v) = screening {
            c.screening = v;
        }
        if let Some(v) = verbose {
            c.verbose = v;
        }
        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::df_j::RiJTask;
    use crate::screening::ScreeningKind;

    fn debug(c: &RhfConfig) -> String {
        format!("{c:#?}")
    }

    #[test]
    fn unset_extension_is_the_base_lowering() {
        let base = ScfOptions {
            max_iter: Some(17),
            level_shift: Some(0.25),
            ..Default::default()
        };
        let ext = ScfOptionsExt {
            base: base.clone(),
            ..Default::default()
        };
        assert_eq!(
            debug(&ext.into_rhf_config().unwrap()),
            debug(&base.into_rhf_config().unwrap())
        );
        assert_eq!(
            debug(&ScfOptionsExt::default().into_rhf_config().unwrap()),
            debug(&RhfConfig::default())
        );
    }

    #[test]
    fn each_extension_field_lowers_to_its_config_field() {
        let c = ScfOptionsExt {
            diis_switch_thresh: Some(3e-2),
            ri_j_task: Some(RiJTask::Geometry),
            screening: Some(ScreeningKind::Csb),
            verbose: Some(true),
            ..Default::default()
        }
        .into_rhf_config()
        .unwrap();
        let expected = RhfConfig {
            diis_switch_thresh: 3e-2,
            ri_j_task: RiJTask::Geometry,
            screening: ScreeningKind::Csb,
            verbose: true,
            ..RhfConfig::default()
        };
        assert_eq!(debug(&c), debug(&expected));
    }
}
