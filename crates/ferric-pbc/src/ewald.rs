//! Ewald sums for point charges in a periodic cell.
//!
//! With splitting parameter `ω` (Bohr⁻¹), for charges `Z_j` at `R_j`:
//!
//! ```text
//! E = ½ Σ'_{L,i,j} Z_i Z_j erfc(ω|R_i − R_j − L|)/|R_i − R_j − L|       (SR, real space)
//!   + (2π/Ω) Σ_{G≠0} e^{−G²/4ω²}/G² |Σ_j Z_j e^{−iG·R_j}|²              (LR, reciprocal)
//!   − ω/√π Σ_j Z_j²                                                    (self)
//!   − π (Σ_j Z_j)² / (2 Ω ω²)                                          (G=0 / background)
//! ```
//!
//! (the prime drops `i = j` at `L = 0`). The last term is the uniform
//! neutralising background for a charged cell (zero for a neutral one); it is
//! the convention of PySCF's `Cell.ewald()`/`energy_nuc()`. The result is
//! independent of `ω`; tests assert that to 1e-12 at ω = 0.8, 1.5, 3.
//!
//! Port of `ewald_nn` in `reference/pbc/pbc_gamma.py`, with two changes:
//! the cutoffs are derived from a precision target, and the SR/LR sums use
//! compensated (Neumaier) summation. The latter is load-bearing: at ω = 3 the
//! self and LR terms are each ~10² Ha for H₂O in a 7-Bohr cell and cancel to
//! ~10¹, and naive summation of the ~10⁵ G terms left a 2.6e-11 ω-dependent
//! residual (MEASURED in numpy against PySCF, identical at precision 1e-16,
//! 1e-20 and 1e-24, i.e. roundoff, not truncation; `math.fsum` removed it
//! to 3e-14).

use crate::lattice::Cell;
use ferric_core::FerricError;
use ferric_integrals::qqr3::erfc;

/// Default Ewald precision target: the neglected SR (`erfc`) and LR
/// (`e^{−G²/4ω²}`) terms are each below this, per pair / per G.
pub const DEFAULT_EWALD_PRECISION: f64 = 1e-16;

/// Neumaier compensated accumulator.
#[derive(Default)]
struct Neumaier {
    sum: f64,
    c: f64,
}

impl Neumaier {
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.c += (self.sum - t) + x;
        } else {
            self.c += (x - t) + self.sum;
        }
        self.sum = t;
    }
    fn value(&self) -> f64 {
        self.sum + self.c
    }
}

/// A balanced splitting parameter `ω = √π / Ω^{1/3}` (Bohr⁻¹): equalises the
/// real- and reciprocal-space work for a roughly isotropic cell. Any `ω > 0`
/// gives the same energy.
pub fn default_ewald_omega(cell: &Cell) -> f64 {
    std::f64::consts::PI.sqrt() / cell.volume().cbrt()
}

/// Ewald nuclear repulsion of the cell's nuclei (charges
/// `Atom::effective_z`, so ghosts contribute nothing), at splitting `omega`
/// and [`DEFAULT_EWALD_PRECISION`]. Matches PySCF `Cell.energy_nuc()`.
pub fn ewald_nuclear_repulsion(cell: &Cell, omega: f64) -> Result<f64, FerricError> {
    ewald_nuclear_repulsion_with_precision(cell, omega, DEFAULT_EWALD_PRECISION)
}

/// [`ewald_nuclear_repulsion`] with an explicit precision target in `(0, 1)`.
pub fn ewald_nuclear_repulsion_with_precision(
    cell: &Cell,
    omega: f64,
    precision: f64,
) -> Result<f64, FerricError> {
    ewald_point_charges(
        cell,
        &cell.nuclear_charges(),
        &cell.positions(),
        omega,
        precision,
    )
}

/// Gamma-point Madelung constant in PySCF's convention
/// (`pyscf.pbc.tools.madelung(cell, kpts=zeros((1,3)))` with `cell.omega = 0`):
/// `−2 ×` the Ewald energy of ONE unit point charge per cell with a
/// neutralising background. Depends only on the lattice. Positive for
/// ordinary cells (simple cubic: `2.8372974794806 / a`). This is the constant
/// `v_M` of the exchange-divergence correction `K += v_M S D S`
/// (`exxdiv='ewald'`).
pub fn madelung_constant(cell: &Cell) -> Result<f64, FerricError> {
    let e = ewald_point_charges(
        cell,
        &[1.0],
        &[[0.0; 3]],
        default_ewald_omega(cell),
        DEFAULT_EWALD_PRECISION,
    )?;
    Ok(-2.0 * e)
}

/// Ewald energy of arbitrary point charges `charges[j]` at `positions[j]`
/// (Bohr) in `cell`'s lattice (the cell's own atoms are ignored). See the
/// module doc for the formula. Errors on a non-positive/non-finite `omega`,
/// a precision outside `(0, 1)`, mismatched lengths, or two coincident charges.
pub fn ewald_point_charges(
    cell: &Cell,
    charges: &[f64],
    positions: &[[f64; 3]],
    omega: f64,
    precision: f64,
) -> Result<f64, FerricError> {
    if !(omega > 0.0) || !omega.is_finite() {
        return Err(FerricError::General(format!(
            "ewald: omega must be finite and > 0, got {omega}"
        )));
    }
    if !(f64::MIN_POSITIVE..1.0).contains(&precision) {
        return Err(FerricError::General(format!(
            "ewald: precision must lie in (0, 1), got {precision}"
        )));
    }
    if charges.len() != positions.len() || charges.is_empty() {
        return Err(FerricError::General(format!(
            "ewald: {} charges vs {} positions (need equal, non-zero)",
            charges.len(),
            positions.len()
        )));
    }
    // erfc(x) < e^{-x²} and e^{-G²/4ω²} = e^{-x²} at x = G/2ω: both tails are
    // below `precision` once x >= sqrt(ln(1/precision)); +1 is a safety margin.
    let s = (1.0 / precision).ln().sqrt() + 1.0;
    let rcut = s / omega;
    let gcut = 2.0 * omega * s;

    // --- SR: real-space erfc sum over images.
    let mut sr = Neumaier::default();
    for l in cell.translations_for(positions, rcut)? {
        let l_is_zero = l == [0.0; 3];
        for (i, (zi, ri)) in charges.iter().zip(positions).enumerate() {
            for (j, (zj, rj)) in charges.iter().zip(positions).enumerate() {
                // A zero charge (ghost atom, effective_z = 0) contributes
                // nothing, and may legitimately sit on a real nucleus.
                if *zi == 0.0 || *zj == 0.0 {
                    continue;
                }
                let dv = [
                    ri[0] - rj[0] - l[0],
                    ri[1] - rj[1] - l[1],
                    ri[2] - rj[2] - l[2],
                ];
                let d = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
                if d < 1e-10 {
                    if l_is_zero && i == j {
                        continue;
                    }
                    return Err(FerricError::General(format!(
                        "ewald: charges {i} and {j} coincide (under translation {l:?})"
                    )));
                }
                sr.add(0.5 * zi * zj * erfc(omega * d) / d);
            }
        }
    }

    // --- LR: reciprocal-space sum over G != 0.
    let vol = cell.volume();
    let mut lr = Neumaier::default();
    for g in cell.gvectors(gcut)? {
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        if g2 < 1e-24 {
            continue; // G = 0 handled by the background term
        }
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for (z, r) in charges.iter().zip(positions) {
            let ph = g[0] * r[0] + g[1] * r[1] + g[2] * r[2];
            re += z * ph.cos();
            im -= z * ph.sin();
        }
        lr.add((-g2 / (4.0 * omega * omega)).exp() / g2 * (re * re + im * im));
    }
    let e_lr = 2.0 * std::f64::consts::PI / vol * lr.value();

    let z2: f64 = charges.iter().map(|z| z * z).sum();
    let zsum: f64 = charges.iter().sum();
    let e_self = -omega / std::f64::consts::PI.sqrt() * z2;
    let e_g0 = -std::f64::consts::PI * zsum * zsum / (2.0 * vol * omega * omega);

    let mut tot = Neumaier::default();
    tot.add(sr.value());
    tot.add(e_lr);
    tot.add(e_self);
    tot.add(e_g0);
    Ok(tot.value())
}

/// Real-space (SR, `erfc`) and reciprocal-space (LR) parts of the Ewald
/// nuclear-repulsion gradient `dE_nn/dR_i` (`natoms × 3`, Hartree/Bohr),
/// returned separately so a mutation test can drop one. The self and
/// background terms are position-independent and contribute nothing.
///
/// ```text
/// SR:  g_i += Σ'_{j,L} Z_i Z_j f'(d) (R_i − R_j − L)/d,
///      f'(d) = −[erfc(ωd)/d² + (2ω/√π) e^{−ω²d²}/d]
/// LR:  g_i += (4π/Ω) Σ_{G≠0} e^{−G²/4ω²}/G² Z_i G Re[S*(G) (−i) e^{−iG·R_i}]
/// ```
///
/// Same cutoffs as [`ewald_point_charges`] at `precision`, Neumaier sums.
/// Model: PySCF `pbc.grad.krhf.grad_nuc` (pinned in `tests/pbc_grad.rs`).
pub fn ewald_nuclear_gradient_parts(
    cell: &Cell,
    omega: f64,
    precision: f64,
) -> Result<(Vec<[f64; 3]>, Vec<[f64; 3]>), FerricError> {
    if !(omega > 0.0) || !omega.is_finite() {
        return Err(FerricError::General(format!(
            "ewald gradient: omega must be finite and > 0, got {omega}"
        )));
    }
    if !(f64::MIN_POSITIVE..1.0).contains(&precision) {
        return Err(FerricError::General(format!(
            "ewald gradient: precision must lie in (0, 1), got {precision}"
        )));
    }
    let charges = cell.nuclear_charges();
    let positions = cell.positions();
    let n = charges.len();
    let s = (1.0 / precision).ln().sqrt() + 1.0;
    let rcut = s / omega;
    let gcut = 2.0 * omega * s;
    let two_w_sqrtpi = 2.0 * omega / std::f64::consts::PI.sqrt();

    let mut sr: Vec<[Neumaier; 3]> = (0..n).map(|_| Default::default()).collect();
    for l in cell.translations_for(&positions, rcut)? {
        for i in 0..n {
            if charges[i] == 0.0 {
                continue;
            }
            for j in 0..n {
                if charges[j] == 0.0 {
                    continue;
                }
                let dv = [
                    positions[i][0] - positions[j][0] - l[0],
                    positions[i][1] - positions[j][1] - l[1],
                    positions[i][2] - positions[j][2] - l[2],
                ];
                let d = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
                if d < 1e-10 {
                    if i == j && l == [0.0; 3] {
                        continue;
                    }
                    return Err(FerricError::General(format!(
                        "ewald gradient: charges {i} and {j} coincide (under translation {l:?})"
                    )));
                }
                let fp = -(erfc(omega * d) / (d * d)
                    + two_w_sqrtpi * (-(omega * d) * (omega * d)).exp() / d);
                let f = charges[i] * charges[j] * fp / d;
                for c in 0..3 {
                    sr[i][c].add(f * dv[c]);
                }
            }
        }
    }

    let vol = cell.volume();
    let mut lr: Vec<[Neumaier; 3]> = (0..n).map(|_| Default::default()).collect();
    for g in cell.gvectors(gcut)? {
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        if g2 < 1e-24 {
            continue;
        }
        let (mut sre, mut sim) = (0.0_f64, 0.0_f64);
        for (z, r) in charges.iter().zip(&positions) {
            let ph = g[0] * r[0] + g[1] * r[1] + g[2] * r[2];
            sre += z * ph.cos();
            sim -= z * ph.sin();
        }
        let k = 4.0 * std::f64::consts::PI / vol * (-g2 / (4.0 * omega * omega)).exp() / g2;
        for i in 0..n {
            if charges[i] == 0.0 {
                continue;
            }
            let ph = g[0] * positions[i][0] + g[1] * positions[i][1] + g[2] * positions[i][2];
            // Re[S* (−i) e^{−iφ}] = −S_re sin φ − S_im cos φ
            let re = -sre * ph.sin() - sim * ph.cos();
            let f = k * charges[i] * re;
            for c in 0..3 {
                lr[i][c].add(f * g[c]);
            }
        }
    }
    let fold = |v: Vec<[Neumaier; 3]>| -> Vec<[f64; 3]> {
        v.iter()
            .map(|a| [a[0].value(), a[1].value(), a[2].value()])
            .collect()
    };
    Ok((fold(sr), fold(lr)))
}

/// Ewald nuclear-repulsion gradient `dE_nn/dR` (`natoms` rows of `[x, y, z]`,
/// Hartree/Bohr): [`ewald_nuclear_gradient_parts`] summed, at
/// [`DEFAULT_EWALD_PRECISION`]. Independent of `omega` up to truncation.
pub fn ewald_nuclear_gradient(cell: &Cell, omega: f64) -> Result<Vec<[f64; 3]>, FerricError> {
    let (sr, lr) = ewald_nuclear_gradient_parts(cell, omega, DEFAULT_EWALD_PRECISION)?;
    Ok(sr
        .iter()
        .zip(&lr)
        .map(|(a, b)| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
        .collect())
}

// ---------------------------------------------------------------------------
// Strain derivative (the Gamma stress, `crate::stress`).
// ---------------------------------------------------------------------------

/// `dE/dε_ij` (Hartree, NOT divided by Ω) of an Ewald energy under the
/// homogeneous strain `r → (1 + ε) r` of the lattice and the charges (fixed
/// fractional coordinates), by part:
///
/// ```text
/// SR:          ½ Σ'_{i,j,L} Z_i Z_j f'(d) d_a d_b / d,   d = R_i − R_j − L,
///              f'(d) = −[erfc(ωd)/d² + (2ω/√π) e^{−ω²d²}/d]
/// LR:          −δ_ab E_LR + (2π/Ω) Σ_{G≠0} |S(G)|² k'(G²) (−2 G_a G_b),
///              k = e^{−G²/4ω²}/G²,  k' = −k (1/4ω² + 1/G²)
/// background:  −δ_ab E_G0   (E_G0 = −π (Σ Z)²/(2Ωω²) ∝ 1/Ω)
/// self:        0
/// ```
///
/// (`G → (1 + ε)^{−T} G` at fixed Miller index, so `dG²/dε_ab = −2 G_a G_b`;
/// `dΩ/dε_ab = Ω δ_ab`; `S(G) = Σ Z e^{−iG·R}` is strain-invariant.) Every
/// part is ω-independent in total, not separately.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EwaldStrainParts {
    /// Real-space (erfc) part.
    pub sr: [[f64; 3]; 3],
    /// Reciprocal-space part including its `−δ E_LR` volume term.
    pub lr: [[f64; 3]; 3],
    /// Neutralising-background (G = 0) volume term.
    pub background: [[f64; 3]; 3],
}

impl EwaldStrainParts {
    /// `sr + lr + background`.
    pub fn total(&self) -> [[f64; 3]; 3] {
        let mut t = [[0.0; 3]; 3];
        for (a, row) in t.iter_mut().enumerate() {
            for (b, v) in row.iter_mut().enumerate() {
                *v = self.sr[a][b] + self.lr[a][b] + self.background[a][b];
            }
        }
        t
    }
}

/// [`EwaldStrainParts`] of [`ewald_point_charges`] with the same cutoffs
/// (`precision`) and Neumaier sums. Errors as [`ewald_point_charges`].
pub fn ewald_point_charges_strain(
    cell: &Cell,
    charges: &[f64],
    positions: &[[f64; 3]],
    omega: f64,
    precision: f64,
) -> Result<EwaldStrainParts, FerricError> {
    if !(omega > 0.0) || !omega.is_finite() {
        return Err(FerricError::General(format!(
            "ewald strain: omega must be finite and > 0, got {omega}"
        )));
    }
    if !(f64::MIN_POSITIVE..1.0).contains(&precision) {
        return Err(FerricError::General(format!(
            "ewald strain: precision must lie in (0, 1), got {precision}"
        )));
    }
    if charges.len() != positions.len() || charges.is_empty() {
        return Err(FerricError::General(format!(
            "ewald strain: {} charges vs {} positions (need equal, non-zero)",
            charges.len(),
            positions.len()
        )));
    }
    let s = (1.0 / precision).ln().sqrt() + 1.0;
    let rcut = s / omega;
    let gcut = 2.0 * omega * s;
    let two_w_sqrtpi = 2.0 * omega / std::f64::consts::PI.sqrt();
    let new9 = || -> [[Neumaier; 3]; 3] { Default::default() };
    let fold9 = |v: &[[Neumaier; 3]; 3]| -> [[f64; 3]; 3] {
        let mut o = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                o[a][b] = v[a][b].value();
            }
        }
        o
    };

    // --- SR.
    let mut sr = new9();
    for l in cell.translations_for(positions, rcut)? {
        let l_is_zero = l == [0.0; 3];
        for (i, (zi, ri)) in charges.iter().zip(positions).enumerate() {
            for (j, (zj, rj)) in charges.iter().zip(positions).enumerate() {
                if *zi == 0.0 || *zj == 0.0 {
                    continue;
                }
                let dv = [
                    ri[0] - rj[0] - l[0],
                    ri[1] - rj[1] - l[1],
                    ri[2] - rj[2] - l[2],
                ];
                let d = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
                if d < 1e-10 {
                    if l_is_zero && i == j {
                        continue;
                    }
                    return Err(FerricError::General(format!(
                        "ewald strain: charges {i} and {j} coincide (under translation {l:?})"
                    )));
                }
                let fp = -(erfc(omega * d) / (d * d)
                    + two_w_sqrtpi * (-(omega * d) * (omega * d)).exp() / d);
                let f = 0.5 * zi * zj * fp / d;
                for a in 0..3 {
                    for b in 0..3 {
                        sr[a][b].add(f * dv[a] * dv[b]);
                    }
                }
            }
        }
    }

    // --- LR.
    let vol = cell.volume();
    let tp_vol = 2.0 * std::f64::consts::PI / vol;
    let mut e_lr = Neumaier::default();
    let mut lr_g = new9();
    for g in cell.gvectors(gcut)? {
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        if g2 < 1e-24 {
            continue;
        }
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for (z, r) in charges.iter().zip(positions) {
            let ph = g[0] * r[0] + g[1] * r[1] + g[2] * r[2];
            re += z * ph.cos();
            im -= z * ph.sin();
        }
        let s2 = re * re + im * im;
        let k = (-g2 / (4.0 * omega * omega)).exp() / g2;
        e_lr.add(k * s2);
        let kp = -k * (1.0 / (4.0 * omega * omega) + 1.0 / g2);
        let f = s2 * kp * -2.0;
        for a in 0..3 {
            for b in 0..3 {
                lr_g[a][b].add(f * g[a] * g[b]);
            }
        }
    }
    let e_lr = tp_vol * e_lr.value();
    let mut lr = fold9(&lr_g);
    for (a, row) in lr.iter_mut().enumerate() {
        for v in row.iter_mut() {
            *v *= tp_vol;
        }
        row[a] -= e_lr;
    }

    // --- background.
    let zsum: f64 = charges.iter().sum();
    let e_g0 = -std::f64::consts::PI * zsum * zsum / (2.0 * vol * omega * omega);
    let mut background = [[0.0; 3]; 3];
    for (a, row) in background.iter_mut().enumerate() {
        row[a] = -e_g0;
    }
    Ok(EwaldStrainParts {
        sr: fold9(&sr),
        lr,
        background,
    })
}

/// [`ewald_point_charges_strain`] of the cell's nuclei (charges
/// `Atom::effective_z`) at `precision`: the strain derivative of
/// [`ewald_nuclear_repulsion_with_precision`].
pub fn ewald_nuclear_strain(
    cell: &Cell,
    omega: f64,
    precision: f64,
) -> Result<EwaldStrainParts, FerricError> {
    ewald_point_charges_strain(
        cell,
        &cell.nuclear_charges(),
        &cell.positions(),
        omega,
        precision,
    )
}

/// `dv_M/dε_ij` of [`madelung_constant`] (`v_M = −2 E_ewald(one unit
/// charge)`, so `dv_M/dε = −2 ×` its [`EwaldStrainParts::total`]). By Euler
/// homogeneity (`v_M` is degree −1 in lengths) `tr dv_M/dε = −v_M` — asserted
/// in `tests/pbc_stress.rs`.
pub fn madelung_strain(cell: &Cell) -> Result<[[f64; 3]; 3], FerricError> {
    let p = ewald_point_charges_strain(
        cell,
        &[1.0],
        &[[0.0; 3]],
        default_ewald_omega(cell),
        DEFAULT_EWALD_PRECISION,
    )?
    .total();
    let mut out = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            out[a][b] = -2.0 * p[a][b];
        }
    }
    Ok(out)
}
