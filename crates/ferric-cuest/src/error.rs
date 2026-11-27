//! Error conversion between `CuestError` and `ferric_core::FerricError`.

use crate::CuestError;
use ferric_core::FerricError;

impl From<CuestError> for FerricError {
    fn from(e: CuestError) -> Self {
        match e {
            CuestError::FeatureDisabled => FerricError::Unsupported {
                feature: "cuest backend".into(),
                reason: "cuEST feature not enabled at build time".into(),
            },
            CuestError::DeviceUnavailable(why) => FerricError::Unsupported {
                feature: "cuest backend".into(),
                reason: why,
            },
            CuestError::Unsupported(why) => FerricError::Unsupported {
                feature: "cuest backend".into(),
                reason: why,
            },
            other => FerricError::General(other.to_string()),
        }
    }
}
