use crate::error::FerricError;

pub(crate) fn parse_float_list(ss: &[String]) -> Result<Vec<f64>, FerricError> {
    ss.iter()
        .map(|s| s.parse::<f64>().map_err(|e| FerricError::Basis(format!("bad float {s:?}: {e}"))))
        .collect()
}

pub(crate) fn canonical_name(name: &str) -> String {
    name.to_ascii_lowercase()
}
