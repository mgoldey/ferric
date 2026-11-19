//! The MD driver: velocity Verlet, Berendsen and Langevin BAOAB.

use crate::config::MdConfig;
use crate::provider::ForceProvider;
use crate::state::{MdObserver, MdResult};
use crate::system::MdSystem;
use ferric_core::error::FerricError;

/// Propagate `system` with forces from `provider` according to `config`,
/// calling each observer on every recorded frame.
pub fn run_md(
    system: &MdSystem,
    provider: &mut dyn ForceProvider,
    config: &MdConfig,
    observers: &mut [&mut dyn MdObserver],
) -> Result<MdResult, FerricError> {
    let _ = (system, provider, config, observers);
    todo!()
}
