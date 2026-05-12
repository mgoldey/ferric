use pyo3::prelude::*;

#[pymodule]
fn ferric(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
