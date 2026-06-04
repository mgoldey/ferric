//! Evaluate Gaussian-type orbital (GTO) basis functions on a 3D Cartesian grid.
//!
//! Core AO evaluator shared by `ferric-dft` (Becke-Lebedev grids) and
//! `ferric-export` (cube-file grids).

use ferric_core::basis::{num_functions, BasisSet};
use ferric_core::mol::Molecule;
use ndarray::{Array2, Array3};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GtoEvalError {
    #[error("no basis shells for Z={z} in basis set")]
    MissingElement { z: i32 },
    #[error("angular momentum l={l} not supported for grid evaluation (only s, p, d)")]
    UnsupportedL { l: i32 },
}

/// One contracted GTO shell located in space (centered on its parent atom).
#[derive(Debug, Clone)]
pub struct LocatedShell<'a> {
    pub l: i32,
    pub pure: bool,
    pub exponents: &'a [f64],
    pub coefficients: &'a [f64],
    pub center: [f64; 3],
}

/// Build the flat list of basis functions in the same order as `PreparedBasis`
/// (atom-major, shell-within-atom in basis-set order, function-within-shell).
pub fn collect_shells<'a>(mol: &Molecule, bs: &'a BasisSet) -> Result<Vec<LocatedShell<'a>>, GtoEvalError> {
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

/// Evaluate a contracted radial part: Σ_p c_p · N(α_p, l) · exp(-α_p r²).
///
/// The basis-set JSON files store contraction coefficients `c_p` that
/// multiply un-normalized primitives `exp(-α r²) · r^l · Y_lm`. To
/// evaluate the actual normalized AO we need to multiply each primitive
/// by its individual normalization constant
///
/// ```text
///   N(α, l) = (2α/π)^(3/4) · (4α)^(l/2) · 1/√((2l-1)!!)
/// ```
///
/// For l = 0 this is the familiar `(2α/π)^{3/4}`. Without this factor,
/// AO evaluations are wrong by ~100-400× for tight 1s primitives on
/// heavy atoms, which silently breaks any grid integration that
/// relies on `eval_basis_on_*` (cube density visualization happened to
/// look "fine" because the relative shape is qualitatively correct
/// even with the wrong absolute scale).
#[inline]
fn radial(shell: &LocatedShell, r2: f64) -> f64 {
    let l = shell.l;
    let pi = std::f64::consts::PI;
    // (2l-1)!! for l = 0..3
    let dbl_fact: f64 = match l {
        0 => 1.0,
        1 => 1.0,
        2 => 3.0,
        3 => 15.0,
        _ => 1.0,
    };
    let mut v = 0.0;
    for (a, c) in shell.exponents.iter().zip(shell.coefficients.iter()) {
        let n = (2.0 * a / pi).powf(0.75)
            * (4.0 * a).powi(l) .sqrt()
            / dbl_fact.sqrt();
        v += c * n * (-a * r2).exp();
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
pub fn eval_shell(shell: &LocatedShell, dx: f64, dy: f64, dz: f64, out: &mut [f64]) -> Result<(), GtoEvalError> {
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

/// Helper returning both the contracted radial value and its derivative w.r.t. r².
///
/// rad(r²)        = Σ_p c_p · N(α_p, l) · exp(-α_p r²)
/// d_rad/d(r²)    = -Σ_p c_p · N(α_p, l) · α_p · exp(-α_p r²)
///
/// Then ∂(rad)/∂x = 2x · d_rad/d(r²).
#[inline]
fn radial_and_d(shell: &LocatedShell, r2: f64) -> (f64, f64) {
    let l = shell.l;
    let pi = std::f64::consts::PI;
    // (2l-1)!! for l = 0..3
    let dbl_fact: f64 = match l {
        0 | 1 => 1.0,
        2 => 3.0,
        3 => 15.0,
        _ => 1.0,
    };
    let mut rad = 0.0_f64;
    let mut drad = 0.0_f64;
    for (a, c) in shell.exponents.iter().zip(shell.coefficients.iter()) {
        let n = (2.0 * a / pi).powf(0.75) * (4.0 * a).powi(l).sqrt() / dbl_fact.sqrt();
        let e = (-a * r2).exp();
        rad += c * n * e;
        drad += -c * n * a * e;
    }
    (rad, drad)
}

/// Like `radial_and_d` but also returns d²rad/d(r²)² = Σ c·N·α²·exp(-α r²).
#[inline]
fn radial_and_d_d2(shell: &LocatedShell, r2: f64) -> (f64, f64, f64) {
    let l = shell.l;
    let pi = std::f64::consts::PI;
    let dbl_fact: f64 = match l {
        0 | 1 => 1.0,
        2 => 3.0,
        3 => 15.0,
        _ => 1.0,
    };
    let mut rad = 0.0_f64;
    let mut drad = 0.0_f64;
    let mut d2rad = 0.0_f64;
    for (a, c) in shell.exponents.iter().zip(shell.coefficients.iter()) {
        let n = (2.0 * a / pi).powf(0.75) * (4.0 * a).powi(l).sqrt() / dbl_fact.sqrt();
        let e = (-a * r2).exp();
        rad += c * n * e;
        drad += -c * n * a * e;
        d2rad += c * n * a * a * e;
    }
    (rad, drad, d2rad)
}

/// Per-shell evaluation of χ and ∇χ at a relative offset (dx, dy, dz) from the
/// shell center.
///
/// Convention (must match `eval_shell` exactly):
///   * s shell: out[0] = rad
///   * p shell: [px, py, pz] · rad (Cartesian order)
///   * pure-d:  5 solid harmonics, m = -2..+2 order
///   * cart-d:  6 functions in libint2 xx, xy, xz, yy, yz, zz order
///   * pure-f:  7 solid harmonics, m = -3..+3 order
///   * cart-f:  10 functions in libint2 xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz order
///   * l ≥ 4:   not supported (UnsupportedL)
fn eval_shell_and_grad(
    sh: &LocatedShell,
    dx: f64,
    dy: f64,
    dz: f64,
    out: &mut [f64],
    out_grad: &mut [[f64; 10]; 3],
) -> Result<(), GtoEvalError> {
    let r2 = dx * dx + dy * dy + dz * dz;
    let (rad, drad_dr2) = radial_and_d(sh, r2);
    let two_dr = 2.0 * drad_dr2;

    match (sh.l, sh.pure) {
        (0, _) => {
            // A = 1 → ∂A = 0; χ = rad, ∇χ = 2·drad·(dx, dy, dz)
            out[0] = rad;
            out_grad[0][0] = two_dr * dx;
            out_grad[1][0] = two_dr * dy;
            out_grad[2][0] = two_dr * dz;
        }
        (1, _) => {
            // A_i = (dx, dy, dz); χ_i = rad · A_i.
            // ∂(rad · A_i)/∂a = 2·drad·a·A_i + rad·∂A_i/∂a = 2·drad·a·A_i + rad·δ_{ai}
            out[0] = rad * dx;
            out[1] = rad * dy;
            out[2] = rad * dz;
            // i = 0 (px):
            out_grad[0][0] = two_dr * dx * dx + rad;
            out_grad[1][0] = two_dr * dy * dx;
            out_grad[2][0] = two_dr * dz * dx;
            // i = 1 (py):
            out_grad[0][1] = two_dr * dx * dy;
            out_grad[1][1] = two_dr * dy * dy + rad;
            out_grad[2][1] = two_dr * dz * dy;
            // i = 2 (pz):
            out_grad[0][2] = two_dr * dx * dz;
            out_grad[1][2] = two_dr * dy * dz;
            out_grad[2][2] = two_dr * dz * dz + rad;
        }
        (2, true) => {
            // Pure d (libint2 m = -2 .. +2):
            //   m=-2: A = √3 xy        ∂x=√3 y,  ∂y=√3 x,  ∂z=0
            //   m=-1: A = √3 yz        ∂x=0,     ∂y=√3 z,  ∂z=√3 y
            //   m= 0: A = (2z²−x²−y²)/2 ∂x=−x,   ∂y=−y,    ∂z=2z
            //   m=+1: A = √3 xz        ∂x=√3 z,  ∂y=0,     ∂z=√3 x
            //   m=+2: A = √3 (x²−y²)/2 ∂x=√3 x,  ∂y=−√3 y, ∂z=0
            let s3 = 3.0_f64.sqrt();
            let a_m = [
                s3 * dx * dy,
                s3 * dy * dz,
                (2.0 * dz * dz - dx * dx - dy * dy) * 0.5,
                s3 * dx * dz,
                s3 * (dx * dx - dy * dy) * 0.5,
            ];
            let amx = [s3 * dy, 0.0, -dx, s3 * dz, s3 * dx];
            let amy = [s3 * dx, s3 * dz, -dy, 0.0, -s3 * dy];
            let amz = [0.0, s3 * dy, 2.0 * dz, s3 * dx, 0.0];
            for m in 0..5 {
                out[m] = rad * a_m[m];
                out_grad[0][m] = two_dr * dx * a_m[m] + rad * amx[m];
                out_grad[1][m] = two_dr * dy * a_m[m] + rad * amy[m];
                out_grad[2][m] = two_dr * dz * a_m[m] + rad * amz[m];
            }
        }
        (2, false) => {
            // Cartesian d (libint2 order: xx, xy, xz, yy, yz, zz):
            let a_m = [dx * dx, dx * dy, dx * dz, dy * dy, dy * dz, dz * dz];
            let amx = [2.0 * dx, dy, dz, 0.0, 0.0, 0.0];
            let amy = [0.0, dx, 0.0, 2.0 * dy, dz, 0.0];
            let amz = [0.0, 0.0, dx, 0.0, dy, 2.0 * dz];
            for m in 0..6 {
                out[m] = rad * a_m[m];
                out_grad[0][m] = two_dr * dx * a_m[m] + rad * amx[m];
                out_grad[1][m] = two_dr * dy * a_m[m] + rad * amy[m];
                out_grad[2][m] = two_dr * dz * a_m[m] + rad * amz[m];
            }
        }
        (3, true) => {
            // Pure f (libint2 m = -3 .. +3), normalizations matching `eval_shell`.
            let s15 = 15.0_f64.sqrt();
            let c10 = 10.0_f64.sqrt() * 0.25; // √10 / 4
            let c6  = 6.0_f64.sqrt() * 0.25;  // √6  / 4
            let x2 = dx * dx; let y2 = dy * dy; let z2 = dz * dz;
            // A_m
            let a_m = [
                c10 * dy * (3.0 * x2 - y2),                  // m = -3
                s15 * dx * dy * dz,                          // m = -2
                c6  * dy * (4.0 * z2 - x2 - y2),             // m = -1
                0.5 * dz * (2.0 * z2 - 3.0 * x2 - 3.0 * y2), // m =  0
                c6  * dx * (4.0 * z2 - x2 - y2),             // m = +1
                0.5 * s15 * dz * (x2 - y2),                  // m = +2
                c10 * dx * (x2 - 3.0 * y2),                  // m = +3
            ];
            // ∂A_m/∂x
            let amx = [
                c10 * 6.0 * dx * dy,
                s15 * dy * dz,
                c6  * (-2.0) * dx * dy,
                -3.0 * dx * dz,
                c6  * (4.0 * z2 - 3.0 * x2 - y2),
                s15 * dx * dz,
                c10 * (3.0 * x2 - 3.0 * y2),
            ];
            // ∂A_m/∂y
            let amy = [
                c10 * (3.0 * x2 - 3.0 * y2),
                s15 * dx * dz,
                c6  * (4.0 * z2 - x2 - 3.0 * y2),
                -3.0 * dy * dz,
                c6  * (-2.0) * dx * dy,
                -s15 * dy * dz,
                c10 * (-6.0) * dx * dy,
            ];
            // ∂A_m/∂z
            let amz = [
                0.0,
                s15 * dx * dy,
                c6  * 8.0 * dy * dz,
                0.5 * (6.0 * z2 - 3.0 * x2 - 3.0 * y2),
                c6  * 8.0 * dx * dz,
                0.5 * s15 * (x2 - y2),
                0.0,
            ];
            for m in 0..7 {
                out[m] = rad * a_m[m];
                out_grad[0][m] = two_dr * dx * a_m[m] + rad * amx[m];
                out_grad[1][m] = two_dr * dy * a_m[m] + rad * amy[m];
                out_grad[2][m] = two_dr * dz * a_m[m] + rad * amz[m];
            }
        }
        (3, false) => {
            // Cartesian f (libint2 order: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz).
            let x2 = dx * dx; let y2 = dy * dy; let z2 = dz * dz;
            let a_m = [
                x2 * dx, x2 * dy, x2 * dz, dx * y2, dx * dy * dz,
                dx * z2, y2 * dy, y2 * dz, dy * z2, z2 * dz,
            ];
            let amx = [
                3.0 * x2, 2.0 * dx * dy, 2.0 * dx * dz, y2, dy * dz,
                z2, 0.0, 0.0, 0.0, 0.0,
            ];
            let amy = [
                0.0, x2, 0.0, 2.0 * dx * dy, dx * dz,
                0.0, 3.0 * y2, 2.0 * dy * dz, z2, 0.0,
            ];
            let amz = [
                0.0, 0.0, x2, 0.0, dx * dy,
                2.0 * dx * dz, 0.0, y2, 2.0 * dy * dz, 3.0 * z2,
            ];
            for m in 0..10 {
                out[m] = rad * a_m[m];
                out_grad[0][m] = two_dr * dx * a_m[m] + rad * amx[m];
                out_grad[1][m] = two_dr * dy * a_m[m] + rad * amy[m];
                out_grad[2][m] = two_dr * dz * a_m[m] + rad * amz[m];
            }
        }
        (l, _) => return Err(GtoEvalError::UnsupportedL { l }),
    }
    Ok(())
}

/// Evaluate χ_μ(r_g) AND ∇χ_μ(r_g) at a set of points.
///
/// Returns `(chi, dchi)` where:
///   `chi`  has shape `(nbasis, npts)`
///   `dchi` has shape `(3, nbasis, npts)` with axis order [x, y, z]
///
/// Each AO has the form χ = rad(r²) · A(x, y, z) where A is the angular
/// polynomial. Its gradient is:
///   ∂χ/∂a = 2 · (a − A_a) · drad_dr2 · A + rad · ∂A/∂a
/// where `a` ranges over {x, y, z} and A_a is the shell center coordinate.
pub fn eval_basis_and_grad_on_points(
    mol: &Molecule,
    bs: &BasisSet,
    points: &[[f64; 3]],
) -> Result<(Array2<f64>, Array3<f64>), GtoEvalError> {
    use rayon::prelude::*;

    let shells = collect_shells(mol, bs)?;
    let nbf: usize = shells.iter().map(|s| num_functions(s.l, s.pure)).sum();
    let npts = points.len();

    // Per-point AO + ∇AO evaluation (pure scalar; the parallelizable unit).
    let eval_point = |p: &[f64; 3]| -> Result<(Vec<f64>, [Vec<f64>; 3]), GtoEvalError> {
        let mut chi_col = vec![0.0f64; nbf];
        let mut dchi_col = [vec![0.0f64; nbf], vec![0.0f64; nbf], vec![0.0f64; nbf]];
        let mut buf = [0.0f64; 10];
        let mut gradbuf: [[f64; 10]; 3] = [[0.0; 10]; 3];

        let mut row_offset = 0usize;
        for sh in &shells {
            buf.fill(0.0);
            for row in gradbuf.iter_mut() { row.fill(0.0); }

            let n = num_functions(sh.l, sh.pure);
            let dx = p[0] - sh.center[0];
            let dy = p[1] - sh.center[1];
            let dz = p[2] - sh.center[2];

            eval_shell_and_grad(sh, dx, dy, dz, &mut buf[..n], &mut gradbuf)?;
            for i in 0..n {
                chi_col[row_offset + i] = buf[i];
                dchi_col[0][row_offset + i] = gradbuf[0][i];
                dchi_col[1][row_offset + i] = gradbuf[1][i];
                dchi_col[2][row_offset + i] = gradbuf[2][i];
            }
            row_offset += n;
        }
        Ok((chi_col, dchi_col))
    };

    // Grid points are independent. Parallelize over points (pure scalar work, no
    // BLAS inside → no oversubscription against BLAS threads), then assemble into
    // the (nbf, npts) / (3, nbf, npts) arrays. BUT parallelism POISONS tiny
    // workloads: rayon spawn/join/steal dwarfs the work on small grids (the
    // free-atom/proatom SCF case — single atom, few points — was ~18× slower
    // under threads). Below a work threshold, run serially. The free-atom SCF
    // path additionally pins rayon to one thread, but this guard makes the
    // function never-slower regardless of caller threading.
    const PAR_WORK_THRESHOLD: usize = 50_000; // ~npts·nbf flops-ish
    let per_point: Vec<(Vec<f64>, [Vec<f64>; 3])> = if npts * nbf >= PAR_WORK_THRESHOLD {
        points
            .par_iter()
            .map(&eval_point)
            .collect::<Result<Vec<_>, _>>()?
    } else {
        points
            .iter()
            .map(&eval_point)
            .collect::<Result<Vec<_>, _>>()?
    };

    let mut chi = Array2::<f64>::zeros((nbf, npts));
    let mut dchi = Array3::<f64>::zeros((3, nbf, npts));
    for (g, (chi_col, dchi_col)) in per_point.iter().enumerate() {
        for i in 0..nbf {
            chi[(i, g)] = chi_col[i];
            dchi[(0, i, g)] = dchi_col[0][i];
            dchi[(1, i, g)] = dchi_col[1][i];
            dchi[(2, i, g)] = dchi_col[2][i];
        }
    }
    Ok((chi, dchi))
}

/// Per-shell evaluation of χ, ∇χ, and ∇∇χ (the Hessian) for s and p shells only.
///
/// `hess_buf[a*3+b][i]` = ∂²χ_i/∂x_a ∂x_b for the i-th basis function.
fn eval_shell_grad_hess(
    sh: &LocatedShell,
    dx: f64, dy: f64, dz: f64,
    out: &mut [f64],
    out_grad: &mut [[f64; 10]; 3],
    out_hess: &mut [[f64; 10]; 9],   // axis-pair index 3*a+b, function index i
) -> Result<(), GtoEvalError> {
    let r2 = dx * dx + dy * dy + dz * dz;
    let (rad, drad, d2rad) = radial_and_d_d2(sh, r2);
    let two_dr = 2.0 * drad;
    let four_d2r = 4.0 * d2rad;
    let d = [dx, dy, dz];

    match (sh.l, sh.pure) {
        (0, _) => {
            // χ = rad
            // ∂χ/∂xa = 2·drad·xa
            // ∂²χ/∂xa∂xb = 2·drad·δ_ab + 4·d²rad·xa·xb
            out[0] = rad;
            for a in 0..3 {
                out_grad[a][0] = two_dr * d[a];
                for b in 0..3 {
                    let delta_ab = if a == b { 1.0 } else { 0.0 };
                    out_hess[a * 3 + b][0] =
                        two_dr * delta_ab + four_d2r * d[a] * d[b];
                }
            }
        }
        (1, _) => {
            // χ_i = rad · d_i  (i = 0,1,2 for px,py,pz)
            // ∂χ_i/∂xa = 2·drad·xa·d_i + rad·δ_{ai}
            // ∂²χ_i/∂xa∂xb = 2·drad·δ_ab·d_i + 4·d²rad·xa·xb·d_i
            //              + 2·drad·xa·δ_{bi} + 2·drad·xb·δ_{ai}
            for i in 0..3 {
                out[i] = rad * d[i];
            }
            for a in 0..3 {
                for i in 0..3 {
                    let d_ai = if a == i { 1.0 } else { 0.0 };
                    out_grad[a][i] = two_dr * d[a] * d[i] + rad * d_ai;
                }
                for b in 0..3 {
                    let delta_ab = if a == b { 1.0 } else { 0.0 };
                    for i in 0..3 {
                        let d_ai = if a == i { 1.0 } else { 0.0 };
                        let d_bi = if b == i { 1.0 } else { 0.0 };
                        out_hess[a * 3 + b][i] =
                            two_dr * delta_ab * d[i]
                          + four_d2r * d[a] * d[b] * d[i]
                          + two_dr * d[a] * d_bi
                          + two_dr * d[b] * d_ai;
                    }
                }
            }
        }
        (2, true) => {
            // Pure d, libint2 m = −2..+2 (matches eval_shell_and_grad).
            //   m=−2: A = √3 dx dy
            //   m=−1: A = √3 dy dz
            //   m= 0: A = (2dz² − dx² − dy²) / 2
            //   m=+1: A = √3 dx dz
            //   m=+2: A = √3 (dx² − dy²) / 2
            let s3 = 3.0_f64.sqrt();
            let a_m = [
                s3 * dx * dy,
                s3 * dy * dz,
                (2.0 * dz * dz - dx * dx - dy * dy) * 0.5,
                s3 * dx * dz,
                s3 * (dx * dx - dy * dy) * 0.5,
            ];
            // Angular gradient ∂A_m/∂x_a: rows = axis (x, y, z), cols = m.
            let amx = [s3 * dy,  0.0,        -dx,        s3 * dz,    s3 * dx];
            let amy = [s3 * dx,  s3 * dz,    -dy,        0.0,       -s3 * dy];
            let amz = [0.0,      s3 * dy,    2.0 * dz,   s3 * dx,    0.0];
            // Angular Hessian ∂²A_m/∂x_a∂x_b: rows = (a*3+b), cols = m.
            // m_xy:  d²/dxdy = √3,  others 0
            // m_yz:  d²/dydz = √3,  others 0
            // m_z²:  d²/dxx = -1, d²/dyy = -1, d²/dzz = 2,  off-diag 0
            // m_xz:  d²/dxdz = √3,  others 0
            // m_x²−y²: d²/dxx = √3, d²/dyy = -√3, others 0
            let ah: [[f64; 5]; 9] = [
                // xx
                [0.0,  0.0,  -1.0,  0.0,   s3],
                // xy
                [s3,   0.0,   0.0,  0.0,   0.0],
                // xz
                [0.0,  0.0,   0.0,  s3,    0.0],
                // yx (= xy by symmetry)
                [s3,   0.0,   0.0,  0.0,   0.0],
                // yy
                [0.0,  0.0,  -1.0,  0.0,  -s3],
                // yz
                [0.0,  s3,    0.0,  0.0,   0.0],
                // zx (= xz)
                [0.0,  0.0,   0.0,  s3,    0.0],
                // zy (= yz)
                [0.0,  s3,    0.0,  0.0,   0.0],
                // zz
                [0.0,  0.0,   2.0,  0.0,   0.0],
            ];
            let ag = [amx, amy, amz];
            for m in 0..5 {
                out[m] = rad * a_m[m];
                for a in 0..3 {
                    out_grad[a][m] = two_dr * d[a] * a_m[m] + rad * ag[a][m];
                }
                for a in 0..3 {
                    for b in 0..3 {
                        let delta_ab = if a == b { 1.0 } else { 0.0 };
                        out_hess[a * 3 + b][m] =
                              two_dr * delta_ab * a_m[m]
                            + four_d2r * d[a] * d[b] * a_m[m]
                            + two_dr * d[a] * ag[b][m]
                            + two_dr * d[b] * ag[a][m]
                            + rad * ah[a * 3 + b][m];
                    }
                }
            }
        }
        (2, false) => {
            // Cartesian d, libint2 order: xx, xy, xz, yy, yz, zz.
            let a_m = [dx * dx, dx * dy, dx * dz, dy * dy, dy * dz, dz * dz];
            let amx = [2.0 * dx, dy,       dz,       0.0,      0.0,      0.0];
            let amy = [0.0,      dx,       0.0,      2.0 * dy, dz,       0.0];
            let amz = [0.0,      0.0,      dx,       0.0,      dy,       2.0 * dz];
            // Angular Hessians ∂²A_m/∂x_a∂x_b for each m:
            //   xx: only Hxx = 2
            //   xy: Hxy = Hyx = 1
            //   xz: Hxz = Hzx = 1
            //   yy: only Hyy = 2
            //   yz: Hyz = Hzy = 1
            //   zz: only Hzz = 2
            let ah: [[f64; 6]; 9] = [
                // xx
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                // xy
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                // xz
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                // yx
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                // yy
                [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                // yz
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                // zx
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                // zy
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                // zz
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
            ];
            let ag = [amx, amy, amz];
            for m in 0..6 {
                out[m] = rad * a_m[m];
                for a in 0..3 {
                    out_grad[a][m] = two_dr * d[a] * a_m[m] + rad * ag[a][m];
                }
                for a in 0..3 {
                    for b in 0..3 {
                        let delta_ab = if a == b { 1.0 } else { 0.0 };
                        out_hess[a * 3 + b][m] =
                              two_dr * delta_ab * a_m[m]
                            + four_d2r * d[a] * d[b] * a_m[m]
                            + two_dr * d[a] * ag[b][m]
                            + two_dr * d[b] * ag[a][m]
                            + rad * ah[a * 3 + b][m];
                    }
                }
            }
        }
        (3, true) => {
            // Pure f (libint2 m = -3 .. +3), matches `eval_shell` normalizations.
            let s15 = 15.0_f64.sqrt();
            let c10 = 10.0_f64.sqrt() * 0.25;
            let c6  = 6.0_f64.sqrt() * 0.25;
            let x2 = dx * dx; let y2 = dy * dy; let z2 = dz * dz;
            let a_m = [
                c10 * dy * (3.0 * x2 - y2),
                s15 * dx * dy * dz,
                c6  * dy * (4.0 * z2 - x2 - y2),
                0.5 * dz * (2.0 * z2 - 3.0 * x2 - 3.0 * y2),
                c6  * dx * (4.0 * z2 - x2 - y2),
                0.5 * s15 * dz * (x2 - y2),
                c10 * dx * (x2 - 3.0 * y2),
            ];
            let amx = [
                c10 * 6.0 * dx * dy,
                s15 * dy * dz,
                c6  * (-2.0) * dx * dy,
                -3.0 * dx * dz,
                c6  * (4.0 * z2 - 3.0 * x2 - y2),
                s15 * dx * dz,
                c10 * (3.0 * x2 - 3.0 * y2),
            ];
            let amy = [
                c10 * (3.0 * x2 - 3.0 * y2),
                s15 * dx * dz,
                c6  * (4.0 * z2 - x2 - 3.0 * y2),
                -3.0 * dy * dz,
                c6  * (-2.0) * dx * dy,
                -s15 * dy * dz,
                c10 * (-6.0) * dx * dy,
            ];
            let amz = [
                0.0,
                s15 * dx * dy,
                c6  * 8.0 * dy * dz,
                0.5 * (6.0 * z2 - 3.0 * x2 - 3.0 * y2),
                c6  * 8.0 * dx * dz,
                0.5 * s15 * (x2 - y2),
                0.0,
            ];
            // Angular Hessians ∂²A_m/∂a∂b (a*3+b indexing), m = 0..7
            // m=-3: y(3x²-y²)·c10 → Hxx=6c10·y, Hyy=-6c10·y, Hzz=0, Hxy=6c10·x, Hxz=Hyz=0
            // m=-2: xyz·s15       → Hxy=s15·z, Hxz=s15·y, Hyz=s15·x; diag = 0
            // m=-1: y(4z²-x²-y²)·c6 → Hxx=-2c6·y, Hyy=-6c6·y, Hzz=8c6·y, Hxy=-2c6·x, Hxz=0, Hyz=8c6·z
            // m= 0: z(2z²-3x²-3y²)·½ → Hxx=-3z, Hyy=-3z, Hzz=6z, Hxy=0, Hxz=-3x, Hyz=-3y
            // m=+1: x(4z²-x²-y²)·c6  → Hxx=-6c6·x, Hyy=-2c6·x, Hzz=8c6·x, Hxy=-2c6·y, Hxz=8c6·z, Hyz=0
            // m=+2: z(x²-y²)·s15/2  → Hxx=s15·z, Hyy=-s15·z, Hzz=0, Hxy=0, Hxz=s15·x, Hyz=-s15·y
            // m=+3: x(x²-3y²)·c10   → Hxx=6c10·x, Hyy=-6c10·x, Hzz=0, Hxy=-6c10·y, Hxz=Hyz=0
            let ah: [[f64; 7]; 9] = [
                // xx: m=-3..+3
                [ 6.0*c10*dy,         0.0,          -2.0*c6*dy,        -3.0*dz,        -6.0*c6*dx,        s15*dz,         6.0*c10*dx ],
                // xy
                [ 6.0*c10*dx,         s15*dz,       -2.0*c6*dx,         0.0,           -2.0*c6*dy,        0.0,           -6.0*c10*dy ],
                // xz
                [ 0.0,                 s15*dy,       0.0,              -3.0*dx,         8.0*c6*dz,        s15*dx,         0.0 ],
                // yx (= xy)
                [ 6.0*c10*dx,         s15*dz,       -2.0*c6*dx,         0.0,           -2.0*c6*dy,        0.0,           -6.0*c10*dy ],
                // yy
                [-6.0*c10*dy,         0.0,          -6.0*c6*dy,        -3.0*dz,        -2.0*c6*dx,       -s15*dz,        -6.0*c10*dx ],
                // yz
                [ 0.0,                 s15*dx,       8.0*c6*dz,        -3.0*dy,         0.0,             -s15*dy,         0.0 ],
                // zx (= xz)
                [ 0.0,                 s15*dy,       0.0,              -3.0*dx,         8.0*c6*dz,        s15*dx,         0.0 ],
                // zy (= yz)
                [ 0.0,                 s15*dx,       8.0*c6*dz,        -3.0*dy,         0.0,             -s15*dy,         0.0 ],
                // zz
                [ 0.0,                 0.0,          8.0*c6*dy,         6.0*dz,         8.0*c6*dx,        0.0,            0.0 ],
            ];
            let ag = [amx, amy, amz];
            for m in 0..7 {
                out[m] = rad * a_m[m];
                for a in 0..3 {
                    out_grad[a][m] = two_dr * d[a] * a_m[m] + rad * ag[a][m];
                }
                for a in 0..3 {
                    for b in 0..3 {
                        let delta_ab = if a == b { 1.0 } else { 0.0 };
                        out_hess[a * 3 + b][m] =
                              two_dr * delta_ab * a_m[m]
                            + four_d2r * d[a] * d[b] * a_m[m]
                            + two_dr * d[a] * ag[b][m]
                            + two_dr * d[b] * ag[a][m]
                            + rad * ah[a * 3 + b][m];
                    }
                }
            }
        }
        (3, false) => {
            // Cartesian f (libint2 order): xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz.
            let x2 = dx * dx; let y2 = dy * dy; let z2 = dz * dz;
            let a_m = [
                x2 * dx, x2 * dy, x2 * dz, dx * y2, dx * dy * dz,
                dx * z2, y2 * dy, y2 * dz, dy * z2, z2 * dz,
            ];
            let amx = [
                3.0 * x2, 2.0 * dx * dy, 2.0 * dx * dz, y2, dy * dz,
                z2, 0.0, 0.0, 0.0, 0.0,
            ];
            let amy = [
                0.0, x2, 0.0, 2.0 * dx * dy, dx * dz,
                0.0, 3.0 * y2, 2.0 * dy * dz, z2, 0.0,
            ];
            let amz = [
                0.0, 0.0, x2, 0.0, dx * dy,
                2.0 * dx * dz, 0.0, y2, 2.0 * dy * dz, 3.0 * z2,
            ];
            // Angular Hessians by monomial (xxx ... zzz)
            //   xxx (x³): Hxx=6x
            //   xxy (x²y): Hxx=2y, Hxy=2x
            //   xxz (x²z): Hxx=2z, Hxz=2x
            //   xyy (xy²): Hyy=2x, Hxy=2y
            //   xyz (xyz): Hxy=z, Hxz=y, Hyz=x
            //   xzz (xz²): Hzz=2x, Hxz=2z
            //   yyy (y³): Hyy=6y
            //   yyz (y²z): Hyy=2z, Hyz=2y
            //   yzz (yz²): Hzz=2y, Hyz=2z
            //   zzz (z³): Hzz=6z
            let ah: [[f64; 10]; 9] = [
                // xx
                [6.0*dx, 2.0*dy, 2.0*dz, 0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0 ],
                // xy
                [0.0,    2.0*dx, 0.0,    2.0*dy, dz,     0.0,    0.0,    0.0,    0.0,    0.0 ],
                // xz
                [0.0,    0.0,    2.0*dx, 0.0,    dy,     2.0*dz, 0.0,    0.0,    0.0,    0.0 ],
                // yx (= xy)
                [0.0,    2.0*dx, 0.0,    2.0*dy, dz,     0.0,    0.0,    0.0,    0.0,    0.0 ],
                // yy
                [0.0,    0.0,    0.0,    2.0*dx, 0.0,    0.0,    6.0*dy, 2.0*dz, 0.0,    0.0 ],
                // yz
                [0.0,    0.0,    0.0,    0.0,    dx,     0.0,    0.0,    2.0*dy, 2.0*dz, 0.0 ],
                // zx (= xz)
                [0.0,    0.0,    2.0*dx, 0.0,    dy,     2.0*dz, 0.0,    0.0,    0.0,    0.0 ],
                // zy (= yz)
                [0.0,    0.0,    0.0,    0.0,    dx,     0.0,    0.0,    2.0*dy, 2.0*dz, 0.0 ],
                // zz
                [0.0,    0.0,    0.0,    0.0,    0.0,    2.0*dx, 0.0,    0.0,    2.0*dy, 6.0*dz ],
            ];
            let ag = [amx, amy, amz];
            for m in 0..10 {
                out[m] = rad * a_m[m];
                for a in 0..3 {
                    out_grad[a][m] = two_dr * d[a] * a_m[m] + rad * ag[a][m];
                }
                for a in 0..3 {
                    for b in 0..3 {
                        let delta_ab = if a == b { 1.0 } else { 0.0 };
                        out_hess[a * 3 + b][m] =
                              two_dr * delta_ab * a_m[m]
                            + four_d2r * d[a] * d[b] * a_m[m]
                            + two_dr * d[a] * ag[b][m]
                            + two_dr * d[b] * ag[a][m]
                            + rad * ah[a * 3 + b][m];
                    }
                }
            }
        }
        (l, _) => return Err(GtoEvalError::UnsupportedL { l }),
    }
    Ok(())
}

/// Evaluate χ_μ(r_g), ∇χ_μ(r_g), and ∇∇χ_μ(r_g) at a set of points.
///
/// Returns `(chi, dchi, ddchi)` where:
///   `chi`   has shape `(nbasis, npts)`
///   `dchi`  has shape `(3, nbasis, npts)` with axis order [x, y, z]
///   `ddchi` has shape `(3, 3, nbasis, npts)` — full 3×3 Hessian
///
/// Supports s, p, d, and f (pure + Cartesian) shells. g and higher still
/// return `UnsupportedL`.
// Returns (values, gradients, hessians) as rank-2/3/4 arrays; the tuple is
// self-documenting and used once, so a type alias would only add indirection.
#[allow(clippy::type_complexity)]
pub fn eval_basis_grad_hess_on_points(
    mol: &ferric_core::mol::Molecule,
    bs: &ferric_core::basis::BasisSet,
    points: &[[f64; 3]],
) -> Result<(Array2<f64>, Array3<f64>, ndarray::Array4<f64>), GtoEvalError> {
    let shells = collect_shells(mol, bs)?;
    let nbf: usize = shells.iter().map(|s| num_functions(s.l, s.pure)).sum();
    let npts = points.len();
    let mut chi = Array2::<f64>::zeros((nbf, npts));
    let mut dchi = Array3::<f64>::zeros((3, nbf, npts));
    let mut ddchi = ndarray::Array4::<f64>::zeros((3, 3, nbf, npts));

    let mut buf = [0.0f64; 10];
    let mut gradbuf: [[f64; 10]; 3] = [[0.0; 10]; 3];
    let mut hessbuf: [[f64; 10]; 9] = [[0.0; 10]; 9];

    for (g, p) in points.iter().enumerate() {
        let mut row_offset = 0usize;
        for sh in &shells {
            buf.fill(0.0);
            for row in gradbuf.iter_mut() { row.fill(0.0); }
            for row in hessbuf.iter_mut() { row.fill(0.0); }

            let n = num_functions(sh.l, sh.pure);
            let dx = p[0] - sh.center[0];
            let dy = p[1] - sh.center[1];
            let dz = p[2] - sh.center[2];

            eval_shell_grad_hess(sh, dx, dy, dz, &mut buf[..n], &mut gradbuf, &mut hessbuf)?;
            for i in 0..n {
                chi[(row_offset + i, g)] = buf[i];
                for a in 0..3 {
                    dchi[(a, row_offset + i, g)] = gradbuf[a][i];
                    for b in 0..3 {
                        ddchi[(a, b, row_offset + i, g)] = hessbuf[a * 3 + b][i];
                    }
                }
            }
            row_offset += n;
        }
    }
    Ok((chi, dchi, ddchi))
}
