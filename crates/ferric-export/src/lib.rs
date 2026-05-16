pub mod cube;
pub mod gto_eval;
pub mod ml;

pub use cube::{export_cube, export_mo_cubes, GridSpec};
pub use gto_eval::{eval_basis_on_grid, nbasis as eval_nbasis, GtoEvalError};
pub use ml::export_npz;

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ndarray::Array3;

/// Export a real-space function f(r) = Σ_P c_P · χ_P(r) to a Gaussian cube file.
///
/// `coeffs` are the basis-set coefficients (one per basis function, in the order
/// matching `eval_basis_on_grid`). `bs` is the basis the coefficients are in
/// (typically the auxiliary basis for PDEP eigenpotentials).
pub fn export_basis_function_cube(
    path: &str,
    mol: &Molecule,
    bs: &BasisSet,
    grid: &GridSpec,
    coeffs: &[f64],
    comment: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let chi = eval_basis_on_grid(mol, bs, grid)?;
    let nbf = chi.nrows();
    if coeffs.len() != nbf {
        return Err(format!(
            "coefficient length {} does not match basis size {}",
            coeffs.len(), nbf
        ).into());
    }
    // f[g] = Σ_P c_P · χ[P, g]
    let coeff_arr = ndarray::ArrayView1::from(coeffs);
    let f_flat = coeff_arr.dot(&chi); // shape (n_grid,)

    let mut data = Array3::<f64>::zeros((grid.n_x, grid.n_y, grid.n_z));
    for ix in 0..grid.n_x {
        for iy in 0..grid.n_y {
            for iz in 0..grid.n_z {
                let g = (ix * grid.n_y + iy) * grid.n_z + iz;
                data[[ix, iy, iz]] = f_flat[g];
            }
        }
    }
    cube::export_cube(path, mol, grid, &data, comment)?;
    Ok(())
}
