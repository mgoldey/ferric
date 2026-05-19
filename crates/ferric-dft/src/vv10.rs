//! VV10 nonlocal correlation. Stub — real implementation in Task 12.

use ndarray::{Array2, Array3};

use crate::density_on_grid::DensityGrid;
use crate::grid::GridPoint;
use crate::libxc::Vv10Params;

/// Stub `add_vv10`: returns 0.0 and does not modify `f`. Real implementation
/// in Task 12 will accumulate V_nl into `f` and return E_nl.
pub fn add_vv10(
    _grid: &[GridPoint],
    _chi: &Array2<f64>,
    _dchi: &Array3<f64>,
    _dens: &DensityGrid,
    _params: &Vv10Params,
    _f: &mut Array2<f64>,
) -> f64 {
    0.0
}
