//! Evaluate Gaussian-type orbital (GTO) basis functions on a 3D Cartesian grid.
//!
//! Used to render molecular orbitals, PDEP eigenpotentials, and other
//! quantities expanded in a basis set onto a real-space grid for cube-file export.
//!
//! Supports both pure (solid-harmonic) and Cartesian Gaussians for s, p, d.
//! Higher angular momenta (f, g, ...) are not yet implemented; will return an error.

use ferric_core::basis::{num_functions, BasisSet};
use ferric_core::mol::Molecule;
use ndarray::Array2;
use thiserror::Error;

use crate::cube::GridSpec;

#[derive(Error, Debug)]
pub enum GtoEvalError {
    #[error("no basis shells for Z={z} in basis set")]
    MissingElement { z: i32 },
    #[error("angular momentum l={l} not supported for grid evaluation (only s, p, d)")]
    UnsupportedL { l: i32 },
}

/// One contracted GTO shell located in space (centered on its parent atom).
#[derive(Debug, Clone)]
struct LocatedShell<'a> {
    l: i32,
    pure: bool,
    exponents: &'a [f64],
    coefficients: &'a [f64],
    center: [f64; 3],
}

/// Build the flat list of basis functions in the same order as `PreparedBasis`
/// (atom-major, shell-within-atom in basis-set order, function-within-shell).
fn collect_shells<'a>(mol: &Molecule, bs: &'a BasisSet) -> Result<Vec<LocatedShell<'a>>, GtoEvalError> {
    let mut out = Vec::new();
    for atom in &mol.atoms {
        let shells = bs.for_element(atom.z).ok_or(GtoEvalError::MissingElement { z: atom.z })?;
        for sh in shells {
            out.push(LocatedShell {
                l: sh.l,
                pure: sh.pure,
                exponents: &sh.exponents,
                coefficients: &sh.coefficients,
                center: [atom.x, atom.y, atom.zpos],
            });
        }
    }
    Ok(out)
}

/// Number of basis functions in `mol` × `bs` (matches `PreparedBasis::nbasis`).
pub fn nbasis(mol: &Molecule, bs: &BasisSet) -> Result<usize, GtoEvalError> {
    let shells = collect_shells(mol, bs)?;
    Ok(shells.iter().map(|s| num_functions(s.l, s.pure)).sum())
}

/// Evaluate a contracted radial part: Σ_p c_p · exp(-α_p r²)
#[inline]
fn radial(shell: &LocatedShell, r2: f64) -> f64 {
    let mut v = 0.0;
    for (a, c) in shell.exponents.iter().zip(shell.coefficients.iter()) {
        v += c * (-a * r2).exp();
    }
    v
}

/// Cartesian → pure (solid-harmonic) transforms for d shells.
///
/// Convention matches libint2's pure shell ordering (m = -l, -l+1, ..., l-1, l):
/// d: [d_{-2}, d_{-1}, d_0, d_1, d_2] = [dxy, dyz, dz2, dxz, dx2-y2] (un-normalized
/// proportionality; the contraction coefficients absorb the radial normalization).
///
/// For Cartesian d: order is xx, xy, xz, yy, yz, zz (libint2 convention).
fn eval_shell(shell: &LocatedShell, dx: f64, dy: f64, dz: f64, out: &mut [f64]) -> Result<(), GtoEvalError> {
    let r2 = dx * dx + dy * dy + dz * dz;
    let rad = radial(shell, r2);

    match (shell.l, shell.pure) {
        (0, _) => {
            out[0] = rad;
        }
        (1, _) => {
            // p shells: libint2 uses Cartesian order [px, py, pz] for both pure and cart
            // (pure with l=1 reduces to Cartesian)
            out[0] = rad * dx;
            out[1] = rad * dy;
            out[2] = rad * dz;
        }
        (2, true) => {
            // Pure d: 5 real solid harmonics (libint2 m = -2 .. +2 order).
            //   S(2,-2) = √3 xy
            //   S(2,-1) = √3 yz
            //   S(2, 0) = (2z² - x² - y²) / 2
            //   S(2,+1) = √3 xz
            //   S(2,+2) = √3 (x² - y²) / 2
            let sqrt3 = 3.0_f64.sqrt();
            out[0] = rad * sqrt3 * dx * dy;
            out[1] = rad * sqrt3 * dy * dz;
            out[2] = rad * (2.0 * dz * dz - dx * dx - dy * dy) * 0.5;
            out[3] = rad * sqrt3 * dx * dz;
            out[4] = rad * sqrt3 * (dx * dx - dy * dy) * 0.5;
        }
        (2, false) => {
            // Cartesian d: 6 functions in libint2 order xx, xy, xz, yy, yz, zz
            out[0] = rad * dx * dx;
            out[1] = rad * dx * dy;
            out[2] = rad * dx * dz;
            out[3] = rad * dy * dy;
            out[4] = rad * dy * dz;
            out[5] = rad * dz * dz;
        }
        (3, true) => {
            // Pure f: 7 real solid harmonics, m = -3 .. +3 (libint2 order).
            // Normalizations from Helgaker/Jørgensen/Olsen Table 6.3 (real Y_lm).
            let sqrt15 = 15.0_f64.sqrt();
            let sqrt10_4 = 10.0_f64.sqrt() * 0.25; // √10 / 4
            let sqrt6_4 = 6.0_f64.sqrt() * 0.25;
            let x2 = dx * dx; let y2 = dy * dy; let z2 = dz * dz;
            // m = -3: √(5/8) · y(3x² − y²)  (proportional)
            out[0] = rad * sqrt10_4 * dy * (3.0 * x2 - y2);
            // m = -2: √15 · xyz
            out[1] = rad * sqrt15 * dx * dy * dz;
            // m = -1: √(3/8) · y(5z² − r²) = √(3/8) · y(4z² − x² − y²)
            out[2] = rad * sqrt6_4 * dy * (4.0 * z2 - x2 - y2);
            // m =  0: ½ · z(2z² − 3x² − 3y²) = ½ · z(5z² − 3r²)
            out[3] = rad * 0.5 * dz * (2.0 * z2 - 3.0 * x2 - 3.0 * y2);
            // m = +1: √(3/8) · x(5z² − r²)
            out[4] = rad * sqrt6_4 * dx * (4.0 * z2 - x2 - y2);
            // m = +2: ½ · √15 · z(x² − y²)
            out[5] = rad * 0.5 * sqrt15 * dz * (x2 - y2);
            // m = +3: √(5/8) · x(x² − 3y²)
            out[6] = rad * sqrt10_4 * dx * (x2 - 3.0 * y2);
        }
        (3, false) => {
            // Cartesian f: 10 functions in libint2 order
            //   xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
            let x2 = dx * dx; let y2 = dy * dy; let z2 = dz * dz;
            out[0] = rad * x2 * dx;
            out[1] = rad * x2 * dy;
            out[2] = rad * x2 * dz;
            out[3] = rad * dx * y2;
            out[4] = rad * dx * dy * dz;
            out[5] = rad * dx * z2;
            out[6] = rad * y2 * dy;
            out[7] = rad * y2 * dz;
            out[8] = rad * dy * z2;
            out[9] = rad * z2 * dz;
        }
        (l, _) => return Err(GtoEvalError::UnsupportedL { l }),
    }
    Ok(())
}

/// Evaluate all basis functions on every grid point.
///
/// Returns `Array2` of shape `(nbasis, n_x*n_y*n_z)` where rows are basis functions
/// and columns are grid points in (ix, iy, iz) row-major order matching `export_cube`.
///
/// Memory: `nbasis × n_grid × 8` bytes. For a 50³ grid + 100 basis functions ≈ 100 MB.
/// Caller should keep this in mind for large systems / fine grids.
pub fn eval_basis_on_grid(
    mol: &Molecule,
    bs: &BasisSet,
    grid: &GridSpec,
) -> Result<Array2<f64>, GtoEvalError> {
    let shells = collect_shells(mol, bs)?;
    let nbf = shells.iter().map(|s| num_functions(s.l, s.pure)).sum();
    let npts = grid.n_x * grid.n_y * grid.n_z;
    let mut chi = Array2::<f64>::zeros((nbf, npts));

    // Assume orthogonal grid (step vectors are axis-aligned). The cube exporter
    // already enforces this; nonorthogonal grids would need a more general loop.
    let hx = grid.step_x[0];
    let hy = grid.step_y[1];
    let hz = grid.step_z[2];

    // Max num_functions for l ≤ 3: pure 7, Cartesian 10.
    let mut shell_buf = [0.0f64; 10];

    for ix in 0..grid.n_x {
        let x = grid.origin[0] + ix as f64 * hx;
        for iy in 0..grid.n_y {
            let y = grid.origin[1] + iy as f64 * hy;
            for iz in 0..grid.n_z {
                let z = grid.origin[2] + iz as f64 * hz;
                let g = (ix * grid.n_y + iy) * grid.n_z + iz;

                let mut row_offset = 0usize;
                for sh in &shells {
                    let n = num_functions(sh.l, sh.pure);
                    let buf = &mut shell_buf[..n];
                    let dx = x - sh.center[0];
                    let dy = y - sh.center[1];
                    let dz = z - sh.center[2];
                    eval_shell(sh, dx, dy, dz, buf)?;
                    for (i, &v) in buf.iter().enumerate() {
                        chi[(row_offset + i, g)] = v;
                    }
                    row_offset += n;
                }
            }
        }
    }
    Ok(chi)
}

/// Evaluate all basis functions on an arbitrary list of points (not necessarily
/// a regular grid).
///
/// Returns `Array2` of shape `(nbasis, npts)`. Used by Becke-Lebedev atomic
/// integration where the points are atom-centered radial × angular nodes.
pub fn eval_basis_on_points(
    mol: &Molecule,
    bs: &BasisSet,
    points: &[[f64; 3]],
) -> Result<Array2<f64>, GtoEvalError> {
    let shells = collect_shells(mol, bs)?;
    let nbf = shells.iter().map(|s| num_functions(s.l, s.pure)).sum();
    let npts = points.len();
    let mut chi = Array2::<f64>::zeros((nbf, npts));

    let mut shell_buf = [0.0f64; 10];

    for (g, p) in points.iter().enumerate() {
        let mut row_offset = 0usize;
        for sh in &shells {
            let n = num_functions(sh.l, sh.pure);
            let buf = &mut shell_buf[..n];
            let dx = p[0] - sh.center[0];
            let dy = p[1] - sh.center[1];
            let dz = p[2] - sh.center[2];
            eval_shell(sh, dx, dy, dz, buf)?;
            for (i, &v) in buf.iter().enumerate() {
                chi[(row_offset + i, g)] = v;
            }
            row_offset += n;
        }
    }
    Ok(chi)
}

/// Index from (ix, iy, iz) to flat grid offset (matches cube row-major order).
pub fn grid_index(grid: &GridSpec, ix: usize, iy: usize, iz: usize) -> usize {
    (ix * grid.n_y + iy) * grid.n_z + iz
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    fn h2_mol() -> Molecule {
        Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.7408\n", 0, 1).unwrap()
    }

    #[test]
    fn h2_sto3g_s_function_normalization() {
        // At the H nucleus, the STO-3G 1s contracted GTO should be positive and finite.
        let mol = h2_mol();
        let bs = basis::bundled("sto-3g").unwrap();
        let grid = GridSpec {
            origin: [0.0, 0.0, 0.0],
            n_x: 1, n_y: 1, n_z: 1,
            step_x: [1.0, 0.0, 0.0],
            step_y: [0.0, 1.0, 0.0],
            step_z: [0.0, 0.0, 1.0],
        };
        let chi = eval_basis_on_grid(&mol, &bs, &grid).unwrap();
        // 2 H atoms × 1 STO-3G 1s function each = 2 basis functions
        assert_eq!(chi.nrows(), 2);
        assert_eq!(chi.ncols(), 1);
        // First H at origin: 1s should be largest there
        assert!(chi[(0, 0)] > 0.0);
        // Second H at z=1.4: its 1s should be smaller at origin
        assert!(chi[(1, 0)] > 0.0);
        assert!(chi[(0, 0)] > chi[(1, 0)]);
    }
}
