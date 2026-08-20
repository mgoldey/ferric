//! Evaluate Gaussian-type orbital (GTO) basis functions on a 3D Cartesian grid.
//!
//! Core AO evaluator shared by `ferric-dft` (Becke-Lebedev grids) and
//! `ferric-export` (cube-file grids).

use ferric_core::basis::{num_functions, BasisSet};
use ferric_core::mol::Molecule;
use ndarray::{Array2, Array3};
use thiserror::Error;

/// Errors from GTO evaluation on a real-space grid.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum GtoEvalError {
    #[error("no basis shells for Z={z} in basis set")]
    MissingElement { z: i32 },
    #[error("angular momentum l={l} not supported for grid evaluation (only s, p, d, f, g)")]
    UnsupportedL { l: i32 },
    /// Pre-flight size guard: the dense AO-on-grid buffer would exceed the memory
    /// budget. Carries the fully-formatted message from
    /// `ferric_core::memory::check_alloc`.
    #[error("{0}")]
    OutOfBudget(String),
}

impl From<GtoEvalError> for ferric_core::error::FerricError {
    fn from(e: GtoEvalError) -> Self { Self::General(e.to_string()) }
}

/// Number of resident `f64` "planes" of shape `(nbf, npts)` a dense AO-grid
/// evaluation keeps alive at once.
///
/// - `chi` alone: 1 plane.
/// - `chi` + `dchi` (value + 3-component gradient): 1 + 3 = 4 planes — this is
///   the formula `ks.rs`'s `check_grid_budget` has used since the SCF energy
///   path was first guarded.
/// - `chi` + `dchi` + `ddchi` (value + gradient + full 3×3 Hessian): 1 + 3 + 9
///   = 13 planes in principle, but `ddchi` is symmetric (∂²/∂a∂b = ∂²/∂b∂a) —
///   [`eval_basis_grad_hess_on_points`] nonetheless *materializes* the full
///   redundant 3×3 (9-plane) tensor (see its `Array4<f64>` return shape), so
///   the resident allocation really is 1 + 3 + 9 = 13 planes at peak — NOT the
///   10 a naive "value + grad + unique Hessian components" count would give.
///   We size the guard for what is actually allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AoGridKind {
    /// `chi` only.
    ValueOnly,
    /// `chi` + `dchi`.
    ValueAndGrad,
    /// `chi` + `dchi` + `ddchi` (full, non-deduplicated 3×3 Hessian).
    ValueGradHess,
}

impl AoGridKind {
    /// Number of `(nbf, npts)` `f64` planes resident at once for this kind.
    /// `const fn` so callers (e.g. `ks.rs`'s batched-fallback sizing) can fold
    /// it into a `const` at compile time.
    pub const fn planes(self) -> usize {
        match self {
            AoGridKind::ValueOnly => 1,
            AoGridKind::ValueAndGrad => 4,
            AoGridKind::ValueGradHess => 13,
        }
    }
}

/// Fail fast if a dense AO-grid evaluation of `kind` for `nbf` basis functions
/// on `npts` grid points would exceed the resolved memory budget.
///
/// This is the single shared formula behind every dense AO/∇AO/∇∇AO buffer in
/// the crate: `ks.rs`'s SCF-energy grid cache (`ValueAndGrad`, doubled by the
/// caller when a VV10 NLC grid is also resident), and the DFT analytic-
/// gradient (`gradient.rs`) and f_xc Newton-kernel (`fxc.rs`) paths, which
/// this function itself now guards internally so every call site is covered
/// without having to thread a budget parameter through `ks_gradient.rs` /
/// `rohf.rs`.
///
/// The budget comes from the unified M1 resolver
/// [`ferric_core::memory::resolve_budget_bytes`] (no explicit config field
/// reaches these evaluators today, so `None` → `FERRIC_MEM_BUDGET_GB` >
/// legacy env vars > 0.8×RAM, same precedence as every other unconfigured
/// budget check in the workspace).
pub fn check_ao_grid_budget(kind: AoGridKind, nbf: usize, npts: usize) -> Result<(), GtoEvalError> {
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let needed = kind
        .planes()
        .saturating_mul(nbf)
        .saturating_mul(npts)
        .saturating_mul(8);
    if needed > budget {
        return Err(GtoEvalError::OutOfBudget(format!(
            "DFT AO-grid buffer ({kind:?}) needs {needed_gb:.2} GB (nbf={nbf}, npts={npts}) \
             but the budget is {budget_gb:.2} GB — raise [memory] budget_gb / \
             FERRIC_MEM_BUDGET_GB, use a smaller grid, or a smaller basis",
            needed_gb = needed as f64 / 1e9,
            budget_gb = budget as f64 / 1e9,
        )));
    }
    Ok(())
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
    // (2l-1)!! for l = 0..4
    let dbl_fact: f64 = match l {
        0 => 1.0,
        1 => 1.0,
        2 => 3.0,
        3 => 15.0,
        4 => 105.0,
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
///
/// Supports s, p, d, f, and g (pure + Cartesian); l ≥ 5 returns `UnsupportedL`.
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
        (4, true) => {
            // Pure g: 9 real solid harmonics, m = -4 .. +4 (libint2 STANDARD
            // order). Angular normalizations are the raw libint2
            // solidharmonics::coeff() applied to the bare monomials (the same
            // convention the d/f arms use); `radial` supplies the uniform l=4
            // shell normalization. Verified against the analytic overlap matrix.
            let s35 = 35.0_f64.sqrt();
            let s70 = 70.0_f64.sqrt();
            let s5 = 5.0_f64.sqrt();
            let s10 = 10.0_f64.sqrt();
            let x2 = dx * dx; let y2 = dy * dy; let z2 = dz * dz;
            let r2v = x2 + y2 + z2;
            // m = -4: (√35/2) · xy(x² − y²)
            out[0] = rad * 0.5 * s35 * dx * dy * (x2 - y2);
            // m = -3: (√70/4) · yz(3x² − y²)
            out[1] = rad * 0.25 * s70 * dy * dz * (3.0 * x2 - y2);
            // m = -2: (√5/2) · xy(7z² − r²)
            out[2] = rad * 0.5 * s5 * dx * dy * (7.0 * z2 - r2v);
            // m = -1: (√10/4) · yz(7z² − 3r²)
            out[3] = rad * 0.25 * s10 * dy * dz * (7.0 * z2 - 3.0 * r2v);
            // m =  0: (1/8) · (35z⁴ − 30z²r² + 3r⁴)
            out[4] = rad * 0.125 * (35.0 * z2 * z2 - 30.0 * z2 * r2v + 3.0 * r2v * r2v);
            // m = +1: (√10/4) · xz(7z² − 3r²)
            out[5] = rad * 0.25 * s10 * dx * dz * (7.0 * z2 - 3.0 * r2v);
            // m = +2: (√5/4) · (x² − y²)(7z² − r²)
            out[6] = rad * 0.25 * s5 * (x2 - y2) * (7.0 * z2 - r2v);
            // m = +3: (√70/4) · xz(x² − 3y²)
            out[7] = rad * 0.25 * s70 * dx * dz * (x2 - 3.0 * y2);
            // m = +4: (√35/8) · (x⁴ − 6x²y² + y⁴)
            out[8] = rad * 0.125 * s35 * (x2 * x2 - 6.0 * x2 * y2 + y2 * y2);
        }
        (4, false) => {
            // Cartesian g: 15 functions in libint2 STANDARD order
            //   xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz,
            //   yyyy, yyyz, yyzz, yzzz, zzzz
            let x2 = dx * dx; let y2 = dy * dy; let z2 = dz * dz;
            out[0] = rad * x2 * x2;
            out[1] = rad * x2 * dx * dy;
            out[2] = rad * x2 * dx * dz;
            out[3] = rad * x2 * y2;
            out[4] = rad * x2 * dy * dz;
            out[5] = rad * x2 * z2;
            out[6] = rad * dx * y2 * dy;
            out[7] = rad * dx * y2 * dz;
            out[8] = rad * dx * dy * z2;
            out[9] = rad * dx * z2 * dz;
            out[10] = rad * y2 * y2;
            out[11] = rad * y2 * dy * dz;
            out[12] = rad * y2 * z2;
            out[13] = rad * dy * z2 * dz;
            out[14] = rad * z2 * z2;
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

    let mut shell_buf = [0.0f64; 15];

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
    // (2l-1)!! for l = 0..4
    let dbl_fact: f64 = match l {
        0 | 1 => 1.0,
        2 => 3.0,
        3 => 15.0,
        4 => 105.0,
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
        4 => 105.0,
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
///   * pure-g:  9 solid harmonics, m = -4..+4 order
///   * cart-g:  15 functions in libint2 xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz, yyyy, yyyz, yyzz, yzzz, zzzz order
///   * l ≥ 5:   not supported (UnsupportedL)
fn eval_shell_and_grad(
    sh: &LocatedShell,
    dx: f64,
    dy: f64,
    dz: f64,
    out: &mut [f64],
    out_grad: &mut [[f64; 15]; 3],
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
        (4, true) => {
            // Pure g (libint2 m = -4 .. +4), normalizations matching `eval_shell`.
            // Angular polynomials and their derivatives are generated from the
            // real solid harmonics via sympy (see docs) so no term is
            // hand-transcribed; validated against grid overlap + FD.
            let s35 = 35.0_f64.sqrt();
            let s70 = 70.0_f64.sqrt();
            let s5 = 5.0_f64.sqrt();
            let s10 = 10.0_f64.sqrt();
            let a_m = [
                0.5*s35 * (1.0*dx*dx*dx*dy - 1.0*dx*dy*dy*dy),  // m=-4
                0.25*s70 * (-1.0*dy*dy*dy*dz + 3.0*dx*dx*dy*dz),  // m=-3
                0.5*s5 * (-1.0*dx*dy*dy*dy - 1.0*dx*dx*dx*dy + 6.0*dx*dy*dz*dz),  // m=-2
                0.25*s10 * (-3.0*dy*dy*dy*dz + 4.0*dy*dz*dz*dz - 3.0*dx*dx*dy*dz),  // m=-1
                0.125 * (3.0*dx*dx*dx*dx + 3.0*dy*dy*dy*dy + 8.0*dz*dz*dz*dz - 24.0*dx*dx*dz*dz - 24.0*dy*dy*dz*dz + 6.0*dx*dx*dy*dy),  // m= 0
                0.25*s10 * (-3.0*dx*dx*dx*dz + 4.0*dx*dz*dz*dz - 3.0*dx*dy*dy*dz),  // m=+1
                0.25*s5 * (1.0*dy*dy*dy*dy - 1.0*dx*dx*dx*dx - 6.0*dy*dy*dz*dz + 6.0*dx*dx*dz*dz),  // m=+2
                0.25*s70 * (1.0*dx*dx*dx*dz - 3.0*dx*dy*dy*dz),  // m=+3
                0.125*s35 * (1.0*dx*dx*dx*dx + 1.0*dy*dy*dy*dy - 6.0*dx*dx*dy*dy),  // m=+4
            ];
            let amx = [
                0.5*s35 * (-1.0*dy*dy*dy + 3.0*dx*dx*dy),  // m=-4
                0.25*s70 * (6.0*dx*dy*dz),  // m=-3
                0.5*s5 * (-1.0*dy*dy*dy - 3.0*dx*dx*dy + 6.0*dy*dz*dz),  // m=-2
                0.25*s10 * (-6.0*dx*dy*dz),  // m=-1
                0.125 * (12.0*dx*dx*dx - 48.0*dx*dz*dz + 12.0*dx*dy*dy),  // m= 0
                0.25*s10 * (4.0*dz*dz*dz - 9.0*dx*dx*dz - 3.0*dy*dy*dz),  // m=+1
                0.25*s5 * (-4.0*dx*dx*dx + 12.0*dx*dz*dz),  // m=+2
                0.25*s70 * (-3.0*dy*dy*dz + 3.0*dx*dx*dz),  // m=+3
                0.125*s35 * (4.0*dx*dx*dx - 12.0*dx*dy*dy),  // m=+4
            ];
            let amy = [
                0.5*s35 * (1.0*dx*dx*dx - 3.0*dx*dy*dy),  // m=-4
                0.25*s70 * (-3.0*dy*dy*dz + 3.0*dx*dx*dz),  // m=-3
                0.5*s5 * (-1.0*dx*dx*dx - 3.0*dx*dy*dy + 6.0*dx*dz*dz),  // m=-2
                0.25*s10 * (4.0*dz*dz*dz - 9.0*dy*dy*dz - 3.0*dx*dx*dz),  // m=-1
                0.125 * (12.0*dy*dy*dy - 48.0*dy*dz*dz + 12.0*dx*dx*dy),  // m= 0
                0.25*s10 * (-6.0*dx*dy*dz),  // m=+1
                0.25*s5 * (4.0*dy*dy*dy - 12.0*dy*dz*dz),  // m=+2
                0.25*s70 * (-6.0*dx*dy*dz),  // m=+3
                0.125*s35 * (4.0*dy*dy*dy - 12.0*dx*dx*dy),  // m=+4
            ];
            let amz = [
                0.0,  // m=-4
                0.25*s70 * (-1.0*dy*dy*dy + 3.0*dx*dx*dy),  // m=-3
                0.5*s5 * (12.0*dx*dy*dz),  // m=-2
                0.25*s10 * (-3.0*dy*dy*dy - 3.0*dx*dx*dy + 12.0*dy*dz*dz),  // m=-1
                0.125 * (32.0*dz*dz*dz - 48.0*dx*dx*dz - 48.0*dy*dy*dz),  // m= 0
                0.25*s10 * (-3.0*dx*dx*dx - 3.0*dx*dy*dy + 12.0*dx*dz*dz),  // m=+1
                0.25*s5 * (-12.0*dy*dy*dz + 12.0*dx*dx*dz),  // m=+2
                0.25*s70 * (1.0*dx*dx*dx - 3.0*dx*dy*dy),  // m=+3
                0.0,  // m=+4
            ];
            for m in 0..9 {
                out[m] = rad * a_m[m];
                out_grad[0][m] = two_dr * dx * a_m[m] + rad * amx[m];
                out_grad[1][m] = two_dr * dy * a_m[m] + rad * amy[m];
                out_grad[2][m] = two_dr * dz * a_m[m] + rad * amz[m];
            }
        }
        (4, false) => {
            // Cartesian g (libint2 STANDARD order):
            //   xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz,
            //   yyyy, yyyz, yyzz, yzzz, zzzz
            let x2 = dx * dx; let y2 = dy * dy; let z2 = dz * dz;
            let x3 = x2 * dx; let y3 = y2 * dy; let z3 = z2 * dz;
            let a_m = [
                x2 * x2, x3 * dy, x3 * dz, x2 * y2, x2 * dy * dz,
                x2 * z2, dx * y3, dx * y2 * dz, dx * dy * z2, dx * z3,
                y2 * y2, y3 * dz, y2 * z2, dy * z3, z2 * z2,
            ];
            let amx = [
                4.0 * x3, 3.0 * x2 * dy, 3.0 * x2 * dz, 2.0 * dx * y2, 2.0 * dx * dy * dz,
                2.0 * dx * z2, y3, y2 * dz, dy * z2, z3,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ];
            let amy = [
                0.0, x3, 0.0, 2.0 * x2 * dy, x2 * dz,
                0.0, 3.0 * dx * y2, 2.0 * dx * dy * dz, dx * z2, 0.0,
                4.0 * y3, 3.0 * y2 * dz, 2.0 * dy * z2, z3, 0.0,
            ];
            let amz = [
                0.0, 0.0, x3, 0.0, x2 * dy,
                2.0 * x2 * dz, 0.0, dx * y2, 2.0 * dx * dy * dz, 3.0 * dx * z2,
                0.0, y3, 2.0 * y2 * dz, 3.0 * dy * z2, 4.0 * z3,
            ];
            for m in 0..15 {
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
    let shells = collect_shells(mol, bs)?;
    let nbf: usize = shells.iter().map(|s| num_functions(s.l, s.pure)).sum();
    let npts = points.len();
    check_ao_grid_budget(AoGridKind::ValueAndGrad, nbf, npts)?;
    eval_basis_and_grad_on_points_unchecked(&shells, nbf, points)
}

/// Same evaluation as [`eval_basis_and_grad_on_points`], but WITHOUT its own
/// `check_ao_grid_budget` re-resolution — the caller has already sized this
/// exact call against a memory budget it resolved itself and must guarantee
/// stays valid. Still returns `Result`: shell evaluation can still fail for a
/// genuine reason (`UnsupportedL`), unrelated to the memory budget — only the
/// budget re-check is skipped, not error propagation in general.
///
/// Exists for `ferric_dft::ks`'s batched V_xc fallback: `KsXc`/`KsXcUks`
/// resolve the budget ONCE in `new()` and use it both to decide Full-vs-Batched
/// AND to size `resolve_batch_size`'s `batch_pts` so that a batch of exactly
/// `batch_pts` points is guaranteed to fit. `check_ao_grid_budget` resolves
/// [`ferric_core::memory::resolve_budget_bytes`] itself — a *live* 0.8×
/// MemAvailable reading in the auto-detect case — so calling it again from
/// inside the per-batch loop re-reads a budget that has been shrinking as the
/// SCF allocates, and can spuriously reject a batch the caller already sized
/// correctly (this was the mid-run `.expect(...)` panic site before this
/// function existed: `check_ao_grid_budget` firing on a drifted reading
/// protects nothing once the caller has already accounted for the real
/// budget with better information). Callers outside `ks.rs`'s batched path
/// should use the checked [`eval_basis_and_grad_on_points`] instead.
pub fn eval_basis_and_grad_on_points_unchecked(
    shells: &[LocatedShell],
    nbf: usize,
    points: &[[f64; 3]],
) -> Result<(Array2<f64>, Array3<f64>), GtoEvalError> {
    use rayon::prelude::*;

    let npts = points.len();

    // Output arrays, allocated ONCE. chi is (nbf, npts) row-major, so
    // chi[(i, g)] lives at offset i*npts + g. dchi is (3, nbf, npts), so
    // dchi[(a, i, g)] lives at offset a*nbf*npts + i*npts + g. We scatter each
    // grid column's values directly into these buffers (no per-point Vec collect
    // + copy), halving the construction peak vs. the old materialize-then-copy.
    let mut chi = Array2::<f64>::zeros((nbf, npts));
    let mut dchi = Array3::<f64>::zeros((3, nbf, npts));

    // Evaluate all AO + ∇AO values for grid point `g` and scatter them into the
    // pre-allocated columns via raw pointers. Each grid point owns a disjoint
    // set of column offsets (column `g` of chi, and column `g` of each of the 3
    // dchi planes), so distinct points never write overlapping addresses — the
    // scatter is data-race-free and bit-identical to a serial fill.
    let dchi_plane = nbf * npts;
    let eval_into = |g: usize,
                     p: &[f64; 3],
                     chi_base: *mut f64,
                     dchi_base: *mut f64|
     -> Result<(), GtoEvalError> {
        let mut buf = [0.0f64; 15];
        let mut gradbuf: [[f64; 15]; 3] = [[0.0; 15]; 3];

        let mut row_offset = 0usize;
        for sh in shells {
            buf.fill(0.0);
            for row in gradbuf.iter_mut() { row.fill(0.0); }

            let n = num_functions(sh.l, sh.pure);
            let dx = p[0] - sh.center[0];
            let dy = p[1] - sh.center[1];
            let dz = p[2] - sh.center[2];

            eval_shell_and_grad(sh, dx, dy, dz, &mut buf[..n], &mut gradbuf)?;
            for i in 0..n {
                let row = row_offset + i;
                // SAFETY: `row < nbf` and `g < npts`, so every offset lies in
                // bounds of the respective array, and column `g` is written by
                // this grid point alone (see the disjointness argument above).
                unsafe {
                    *chi_base.add(row * npts + g) = buf[i];
                    *dchi_base.add(row * npts + g) = gradbuf[0][i];
                    *dchi_base.add(dchi_plane + row * npts + g) = gradbuf[1][i];
                    *dchi_base.add(2 * dchi_plane + row * npts + g) = gradbuf[2][i];
                }
            }
            row_offset += n;
        }
        Ok(())
    };

    // Grid points are independent. Parallelize over points (pure scalar work, no
    // BLAS inside → no oversubscription against BLAS threads). BUT parallelism
    // POISONS tiny workloads: rayon spawn/join/steal dwarfs the work on small
    // grids (the free-atom/proatom SCF case — single atom, few points — was ~18×
    // slower under threads). Below a work threshold, run serially. The free-atom
    // SCF path additionally pins rayon to one thread, but this guard makes the
    // function never-slower regardless of caller threading.
    const PAR_WORK_THRESHOLD: usize = 50_000; // ~npts·nbf flops-ish
    let chi_addr = chi.as_mut_ptr() as usize;
    let dchi_addr = dchi.as_mut_ptr() as usize;
    if npts * nbf >= PAR_WORK_THRESHOLD {
        points
            .par_iter()
            .enumerate()
            .try_for_each(|(g, p)| {
                eval_into(g, p, chi_addr as *mut f64, dchi_addr as *mut f64)
            })?;
    } else {
        for (g, p) in points.iter().enumerate() {
            eval_into(g, p, chi_addr as *mut f64, dchi_addr as *mut f64)?;
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
    out_grad: &mut [[f64; 15]; 3],
    out_hess: &mut [[f64; 15]; 9],   // axis-pair index 3*a+b, function index i
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
        (4, true) => {
            // Pure g (libint2 m = -4 .. +4), matches `eval_shell` normalizations.
            let s35 = 35.0_f64.sqrt();
            let s70 = 70.0_f64.sqrt();
            let s5 = 5.0_f64.sqrt();
            let s10 = 10.0_f64.sqrt();
            let a_m = [
                0.5*s35 * (1.0*dx*dx*dx*dy - 1.0*dx*dy*dy*dy),  // m=-4
                0.25*s70 * (-1.0*dy*dy*dy*dz + 3.0*dx*dx*dy*dz),  // m=-3
                0.5*s5 * (-1.0*dx*dy*dy*dy - 1.0*dx*dx*dx*dy + 6.0*dx*dy*dz*dz),  // m=-2
                0.25*s10 * (-3.0*dy*dy*dy*dz + 4.0*dy*dz*dz*dz - 3.0*dx*dx*dy*dz),  // m=-1
                0.125 * (3.0*dx*dx*dx*dx + 3.0*dy*dy*dy*dy + 8.0*dz*dz*dz*dz - 24.0*dx*dx*dz*dz - 24.0*dy*dy*dz*dz + 6.0*dx*dx*dy*dy),  // m= 0
                0.25*s10 * (-3.0*dx*dx*dx*dz + 4.0*dx*dz*dz*dz - 3.0*dx*dy*dy*dz),  // m=+1
                0.25*s5 * (1.0*dy*dy*dy*dy - 1.0*dx*dx*dx*dx - 6.0*dy*dy*dz*dz + 6.0*dx*dx*dz*dz),  // m=+2
                0.25*s70 * (1.0*dx*dx*dx*dz - 3.0*dx*dy*dy*dz),  // m=+3
                0.125*s35 * (1.0*dx*dx*dx*dx + 1.0*dy*dy*dy*dy - 6.0*dx*dx*dy*dy),  // m=+4
            ];
            let amx = [
                0.5*s35 * (-1.0*dy*dy*dy + 3.0*dx*dx*dy),  // m=-4
                0.25*s70 * (6.0*dx*dy*dz),  // m=-3
                0.5*s5 * (-1.0*dy*dy*dy - 3.0*dx*dx*dy + 6.0*dy*dz*dz),  // m=-2
                0.25*s10 * (-6.0*dx*dy*dz),  // m=-1
                0.125 * (12.0*dx*dx*dx - 48.0*dx*dz*dz + 12.0*dx*dy*dy),  // m= 0
                0.25*s10 * (4.0*dz*dz*dz - 9.0*dx*dx*dz - 3.0*dy*dy*dz),  // m=+1
                0.25*s5 * (-4.0*dx*dx*dx + 12.0*dx*dz*dz),  // m=+2
                0.25*s70 * (-3.0*dy*dy*dz + 3.0*dx*dx*dz),  // m=+3
                0.125*s35 * (4.0*dx*dx*dx - 12.0*dx*dy*dy),  // m=+4
            ];
            let amy = [
                0.5*s35 * (1.0*dx*dx*dx - 3.0*dx*dy*dy),  // m=-4
                0.25*s70 * (-3.0*dy*dy*dz + 3.0*dx*dx*dz),  // m=-3
                0.5*s5 * (-1.0*dx*dx*dx - 3.0*dx*dy*dy + 6.0*dx*dz*dz),  // m=-2
                0.25*s10 * (4.0*dz*dz*dz - 9.0*dy*dy*dz - 3.0*dx*dx*dz),  // m=-1
                0.125 * (12.0*dy*dy*dy - 48.0*dy*dz*dz + 12.0*dx*dx*dy),  // m= 0
                0.25*s10 * (-6.0*dx*dy*dz),  // m=+1
                0.25*s5 * (4.0*dy*dy*dy - 12.0*dy*dz*dz),  // m=+2
                0.25*s70 * (-6.0*dx*dy*dz),  // m=+3
                0.125*s35 * (4.0*dy*dy*dy - 12.0*dx*dx*dy),  // m=+4
            ];
            let amz = [
                0.0,  // m=-4
                0.25*s70 * (-1.0*dy*dy*dy + 3.0*dx*dx*dy),  // m=-3
                0.5*s5 * (12.0*dx*dy*dz),  // m=-2
                0.25*s10 * (-3.0*dy*dy*dy - 3.0*dx*dx*dy + 12.0*dy*dz*dz),  // m=-1
                0.125 * (32.0*dz*dz*dz - 48.0*dx*dx*dz - 48.0*dy*dy*dz),  // m= 0
                0.25*s10 * (-3.0*dx*dx*dx - 3.0*dx*dy*dy + 12.0*dx*dz*dz),  // m=+1
                0.25*s5 * (-12.0*dy*dy*dz + 12.0*dx*dx*dz),  // m=+2
                0.25*s70 * (1.0*dx*dx*dx - 3.0*dx*dy*dy),  // m=+3
                0.0,  // m=+4
            ];
            // Angular Hessians ∂²A_m/∂a∂b (a*3+b indexing), cols m=-4..+4.
            // Generated from the same sympy solid-harmonic derivation.
            let ah: [[f64; 9]; 9] = [
                // xx
                [ 0.5*s35*(6.0*dx*dy), 0.25*s70*(6.0*dy*dz), 0.5*s5*(-6.0*dx*dy), 0.25*s10*(-6.0*dy*dz), 0.125*(-48.0*dz*dz + 12.0*dy*dy + 36.0*dx*dx), 0.25*s10*(-18.0*dx*dz), 0.25*s5*(-12.0*dx*dx + 12.0*dz*dz), 0.25*s70*(6.0*dx*dz), 0.125*s35*(-12.0*dy*dy + 12.0*dx*dx) ],
                // xy
                [ 0.5*s35*(-3.0*dy*dy + 3.0*dx*dx), 0.25*s70*(6.0*dx*dz), 0.5*s5*(-3.0*dx*dx - 3.0*dy*dy + 6.0*dz*dz), 0.25*s10*(-6.0*dx*dz), 0.125*(24.0*dx*dy), 0.25*s10*(-6.0*dy*dz), 0.0, 0.25*s70*(-6.0*dy*dz), 0.125*s35*(-24.0*dx*dy) ],
                // xz
                [ 0.0, 0.25*s70*(6.0*dx*dy), 0.5*s5*(12.0*dy*dz), 0.25*s10*(-6.0*dx*dy), 0.125*(-96.0*dx*dz), 0.25*s10*(-9.0*dx*dx - 3.0*dy*dy + 12.0*dz*dz), 0.25*s5*(24.0*dx*dz), 0.25*s70*(-3.0*dy*dy + 3.0*dx*dx), 0.0 ],
                // yx (= xy)
                [ 0.5*s35*(-3.0*dy*dy + 3.0*dx*dx), 0.25*s70*(6.0*dx*dz), 0.5*s5*(-3.0*dx*dx - 3.0*dy*dy + 6.0*dz*dz), 0.25*s10*(-6.0*dx*dz), 0.125*(24.0*dx*dy), 0.25*s10*(-6.0*dy*dz), 0.0, 0.25*s70*(-6.0*dy*dz), 0.125*s35*(-24.0*dx*dy) ],
                // yy
                [ 0.5*s35*(-6.0*dx*dy), 0.25*s70*(-6.0*dy*dz), 0.5*s5*(-6.0*dx*dy), 0.25*s10*(-18.0*dy*dz), 0.125*(-48.0*dz*dz + 12.0*dx*dx + 36.0*dy*dy), 0.25*s10*(-6.0*dx*dz), 0.25*s5*(-12.0*dz*dz + 12.0*dy*dy), 0.25*s70*(-6.0*dx*dz), 0.125*s35*(-12.0*dx*dx + 12.0*dy*dy) ],
                // yz
                [ 0.0, 0.25*s70*(-3.0*dy*dy + 3.0*dx*dx), 0.5*s5*(12.0*dx*dz), 0.25*s10*(-9.0*dy*dy - 3.0*dx*dx + 12.0*dz*dz), 0.125*(-96.0*dy*dz), 0.25*s10*(-6.0*dx*dy), 0.25*s5*(-24.0*dy*dz), 0.25*s70*(-6.0*dx*dy), 0.0 ],
                // zx (= xz)
                [ 0.0, 0.25*s70*(6.0*dx*dy), 0.5*s5*(12.0*dy*dz), 0.25*s10*(-6.0*dx*dy), 0.125*(-96.0*dx*dz), 0.25*s10*(-9.0*dx*dx - 3.0*dy*dy + 12.0*dz*dz), 0.25*s5*(24.0*dx*dz), 0.25*s70*(-3.0*dy*dy + 3.0*dx*dx), 0.0 ],
                // zy (= yz)
                [ 0.0, 0.25*s70*(-3.0*dy*dy + 3.0*dx*dx), 0.5*s5*(12.0*dx*dz), 0.25*s10*(-9.0*dy*dy - 3.0*dx*dx + 12.0*dz*dz), 0.125*(-96.0*dy*dz), 0.25*s10*(-6.0*dx*dy), 0.25*s5*(-24.0*dy*dz), 0.25*s70*(-6.0*dx*dy), 0.0 ],
                // zz
                [ 0.0, 0.0, 0.5*s5*(12.0*dx*dy), 0.25*s10*(24.0*dy*dz), 0.125*(-48.0*dx*dx - 48.0*dy*dy + 96.0*dz*dz), 0.25*s10*(24.0*dx*dz), 0.25*s5*(-12.0*dy*dy + 12.0*dx*dx), 0.0, 0.0 ],
            ];
            let ag = [amx, amy, amz];
            for m in 0..9 {
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
        (4, false) => {
            // Cartesian g (libint2 STANDARD order):
            //   xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz,
            //   yyyy, yyyz, yyzz, yzzz, zzzz
            let x2 = dx * dx; let y2 = dy * dy; let z2 = dz * dz;
            let x3 = x2 * dx; let y3 = y2 * dy; let z3 = z2 * dz;
            let a_m = [
                x2 * x2, x3 * dy, x3 * dz, x2 * y2, x2 * dy * dz,
                x2 * z2, dx * y3, dx * y2 * dz, dx * dy * z2, dx * z3,
                y2 * y2, y3 * dz, y2 * z2, dy * z3, z2 * z2,
            ];
            let amx = [
                4.0 * x3, 3.0 * x2 * dy, 3.0 * x2 * dz, 2.0 * dx * y2, 2.0 * dx * dy * dz,
                2.0 * dx * z2, y3, y2 * dz, dy * z2, z3,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ];
            let amy = [
                0.0, x3, 0.0, 2.0 * x2 * dy, x2 * dz,
                0.0, 3.0 * dx * y2, 2.0 * dx * dy * dz, dx * z2, 0.0,
                4.0 * y3, 3.0 * y2 * dz, 2.0 * dy * z2, z3, 0.0,
            ];
            let amz = [
                0.0, 0.0, x3, 0.0, x2 * dy,
                2.0 * x2 * dz, 0.0, dx * y2, 2.0 * dx * dy * dz, 3.0 * dx * z2,
                0.0, y3, 2.0 * y2 * dz, 3.0 * dy * z2, 4.0 * z3,
            ];
            // Angular Hessians ∂²A_m/∂a∂b, rows = axis-pair (a*3+b), cols = 15 g cart.
            let ah: [[f64; 15]; 9] = [
                // xx
                [12.0*x2, 6.0*dx*dy, 6.0*dx*dz, 2.0*y2, 2.0*dy*dz, 2.0*z2,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                // xy
                [0.0, 3.0*x2, 0.0, 4.0*dx*dy, 2.0*dx*dz, 0.0,
                 3.0*y2, 2.0*dy*dz, z2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                // xz
                [0.0, 0.0, 3.0*x2, 0.0, 2.0*dx*dy, 4.0*dx*dz,
                 0.0, y2, 2.0*dy*dz, 3.0*z2, 0.0, 0.0, 0.0, 0.0, 0.0],
                // yx (= xy)
                [0.0, 3.0*x2, 0.0, 4.0*dx*dy, 2.0*dx*dz, 0.0,
                 3.0*y2, 2.0*dy*dz, z2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                // yy
                [0.0, 0.0, 0.0, 2.0*x2, 0.0, 0.0,
                 6.0*dx*dy, 2.0*dx*dz, 0.0, 0.0, 12.0*y2, 6.0*dy*dz, 2.0*z2, 0.0, 0.0],
                // yz
                [0.0, 0.0, 0.0, 0.0, x2, 0.0,
                 0.0, 2.0*dx*dy, 2.0*dx*dz, 0.0, 0.0, 3.0*y2, 4.0*dy*dz, 3.0*z2, 0.0],
                // zx (= xz)
                [0.0, 0.0, 3.0*x2, 0.0, 2.0*dx*dy, 4.0*dx*dz,
                 0.0, y2, 2.0*dy*dz, 3.0*z2, 0.0, 0.0, 0.0, 0.0, 0.0],
                // zy (= yz)
                [0.0, 0.0, 0.0, 0.0, x2, 0.0,
                 0.0, 2.0*dx*dy, 2.0*dx*dz, 0.0, 0.0, 3.0*y2, 4.0*dy*dz, 3.0*z2, 0.0],
                // zz
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0*x2,
                 0.0, 0.0, 2.0*dx*dy, 6.0*dx*dz, 0.0, 0.0, 2.0*y2, 6.0*dy*dz, 12.0*z2],
            ];
            let ag = [amx, amy, amz];
            for m in 0..15 {
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
/// Supports s, p, d, f, and g (pure + Cartesian) shells. l ≥ 5 still
/// returns `UnsupportedL`.
// Returns (values, gradients, hessians) as rank-2/3/4 arrays; the tuple is
// self-documenting and used once, so a type alias would only add indirection.
#[allow(clippy::type_complexity)]
pub fn eval_basis_grad_hess_on_points(
    mol: &ferric_core::mol::Molecule,
    bs: &ferric_core::basis::BasisSet,
    points: &[[f64; 3]],
) -> Result<(Array2<f64>, Array3<f64>, ndarray::Array4<f64>), GtoEvalError> {
    use rayon::prelude::*;

    let shells = collect_shells(mol, bs)?;
    let nbf: usize = shells.iter().map(|s| num_functions(s.l, s.pure)).sum();
    let npts = points.len();
    check_ao_grid_budget(AoGridKind::ValueGradHess, nbf, npts)?;

    // Output arrays, allocated once; each grid point scatters into its own
    // column `g` of every plane, so writes of distinct points are disjoint —
    // same raw-pointer scatter (and same disjointness argument) as
    // `eval_basis_and_grad_on_points`. This function was always single-copy
    // (it never had the per-point collect); the scatter adds the same
    // point-parallelism without any extra buffer.
    let mut chi = Array2::<f64>::zeros((nbf, npts));
    let mut dchi = Array3::<f64>::zeros((3, nbf, npts));
    let mut ddchi = ndarray::Array4::<f64>::zeros((3, 3, nbf, npts));

    let plane = nbf * npts; // stride between axis-planes of dchi / (a,b)-planes of ddchi
    let eval_into = |g: usize,
                     p: &[f64; 3],
                     chi_base: *mut f64,
                     dchi_base: *mut f64,
                     ddchi_base: *mut f64|
     -> Result<(), GtoEvalError> {
        let mut buf = [0.0f64; 15];
        let mut gradbuf: [[f64; 15]; 3] = [[0.0; 15]; 3];
        let mut hessbuf: [[f64; 15]; 9] = [[0.0; 15]; 9];

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
                let row = row_offset + i;
                // SAFETY: `row < nbf`, `g < npts`, and plane indices < 3
                // (resp. 9), so every offset is in bounds; column `g` is
                // written by this grid point alone (disjointness above).
                unsafe {
                    *chi_base.add(row * npts + g) = buf[i];
                    for a in 0..3 {
                        *dchi_base.add(a * plane + row * npts + g) = gradbuf[a][i];
                        for b in 0..3 {
                            *ddchi_base.add((a * 3 + b) * plane + row * npts + g) =
                                hessbuf[a * 3 + b][i];
                        }
                    }
                }
            }
            row_offset += n;
        }
        Ok(())
    };

    // Same small-workload guard as `eval_basis_and_grad_on_points`: rayon
    // overhead poisons tiny grids, so run those serially.
    const PAR_WORK_THRESHOLD: usize = 50_000; // ~npts·nbf flops-ish
    let chi_addr = chi.as_mut_ptr() as usize;
    let dchi_addr = dchi.as_mut_ptr() as usize;
    let ddchi_addr = ddchi.as_mut_ptr() as usize;
    if npts * nbf >= PAR_WORK_THRESHOLD {
        points.par_iter().enumerate().try_for_each(|(g, p)| {
            eval_into(
                g,
                p,
                chi_addr as *mut f64,
                dchi_addr as *mut f64,
                ddchi_addr as *mut f64,
            )
        })?;
    } else {
        for (g, p) in points.iter().enumerate() {
            eval_into(
                g,
                p,
                chi_addr as *mut f64,
                dchi_addr as *mut f64,
                ddchi_addr as *mut f64,
            )?;
        }
    }
    Ok((chi, dchi, ddchi))
}

#[cfg(test)]
mod budget_guard_tests {
    use super::*;

    // FERRIC_MEM_BUDGET_GB is process-global; the default test harness runs
    // tests in parallel (same convention as ferric_core::memory's own tests).
    // Shared crate-wide lock (see lib.rs) — a module-local lock cannot stop
    // cross-module races on the process-global budget env var.
    use crate::TEST_BUDGET_ENV_LOCK as ENV_LOCK;
    const VAR: &str = ferric_core::memory::ENV_UNIFIED;

    fn clear() {
        std::env::remove_var(VAR);
    }

    #[test]
    fn planes_formula_matches_documented_counts() {
        assert_eq!(AoGridKind::ValueOnly.planes(), 1);
        assert_eq!(AoGridKind::ValueAndGrad.planes(), 4);
        assert_eq!(AoGridKind::ValueGradHess.planes(), 13);
    }

    /// A 50-heavy-atom / aTZ-scale / fine-grid job (nbf~1500, npts~412_500)
    /// with the full value+grad+hess accounting must be rejected against a
    /// small configured budget — this is the exact shape from the production
    /// incident this guard exists to catch (gradient.rs's Hessian path).
    #[test]
    fn large_scale_hess_allocation_rejected_under_tiny_budget() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        std::env::set_var(VAR, "1"); // 1 GiB budget

        let nbf = 1500usize;
        let npts = 412_500usize;
        // 13 planes * 1500 * 412500 * 8 bytes ≈ 64.4 GB — far over 1 GiB.
        let err = check_ao_grid_budget(AoGridKind::ValueGradHess, nbf, npts).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("nbf=1500"), "message should cite nbf: {msg}");
        assert!(msg.contains("npts=412500"), "message should cite npts: {msg}");
        assert!(
            msg.contains("GB"),
            "message should state the estimated/budgeted sizes in GB: {msg}"
        );
        assert!(
            msg.contains("FERRIC_MEM_BUDGET_GB"),
            "message should point at the remediation knob: {msg}"
        );

        clear();
    }

    /// The same large-scale shape must also be rejected for the plainer
    /// value+grad (4-plane) accounting used by `fxc.rs`'s Newton kernels and
    /// `xc_gradient_closed_lda_from_density` — a smaller budget still catches
    /// it since 4 planes at this scale is still ~19.8 GB.
    #[test]
    fn large_scale_grad_allocation_rejected_under_tiny_budget() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        std::env::set_var(VAR, "1"); // 1 GiB budget

        let nbf = 1500usize;
        let npts = 412_500usize;
        let err = check_ao_grid_budget(AoGridKind::ValueAndGrad, nbf, npts).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("nbf=1500"));
        assert!(msg.contains("npts=412500"));
        assert!(msg.contains("GB"));

        clear();
    }

    /// Small/typical systems (water/cc-pVDZ scale: nbf~25, npts~35_000, a
    /// (75,110) atomic grid on 3 atoms) must NOT be rejected at a realistic
    /// budget — a wrong/overly conservative formula must not break currently-
    /// working DFT gradient / ROKS-Newton jobs. This is the critical
    /// regression guard: the formula must undershoot real small jobs by a
    /// wide margin, not just barely pass.
    #[test]
    fn small_scale_allocation_not_rejected_under_realistic_budget() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        std::env::set_var(VAR, "2"); // 2 GiB budget — a modest laptop-scale default

        let nbf = 25usize; // water / cc-pVDZ
        let npts = 35_000usize; // 3 atoms * (75 radial * 110 angular)-ish grid

        // ValueGradHess (the largest, gradient.rs's Hessian path): 13*25*35000*8
        // ≈ 91 MB — trivially under 2 GiB.
        assert!(
            check_ao_grid_budget(AoGridKind::ValueGradHess, nbf, npts).is_ok(),
            "water/cc-pVDZ-scale Hessian AO grid must fit a 2 GiB budget"
        );
        // ValueAndGrad (fxc.rs / LDA gradient path): even smaller.
        assert!(
            check_ao_grid_budget(AoGridKind::ValueAndGrad, nbf, npts).is_ok(),
            "water/cc-pVDZ-scale grad-only AO grid must fit a 2 GiB budget"
        );

        clear();
    }

    /// A somewhat larger but still very ordinary system (small organic
    /// molecule, def2-SVP-scale nbf~150, a fine (99,590) grid npts~150_000)
    /// must also pass comfortably under the auto-detected/fallback budget
    /// (no explicit env override) — guards against the formula being so
    /// conservative it rejects everyday jobs on a modest developer machine.
    #[test]
    fn moderate_scale_allocation_not_rejected_under_default_budget() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear(); // fall through to auto-detect/fallback (>= 2 GiB fallback floor)

        let nbf = 150usize;
        let npts = 150_000usize;
        // 13*150*150000*8 ≈ 2.34 GB — comfortably under any real machine's
        // auto-detected budget, and under the final 2 GiB fallback only in
        // the ValueAndGrad case; assert the actually-relevant (smaller,
        // fxc/LDA-path) formula here to avoid flaking on a sandboxed CI box
        // with no /proc/meminfo (which lands on the 2 GiB fallback).
        assert!(
            check_ao_grid_budget(AoGridKind::ValueAndGrad, nbf, npts).is_ok(),
            "moderate-scale grad-only AO grid must fit the default budget"
        );

        clear();
    }
}
