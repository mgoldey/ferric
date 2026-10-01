//! MBD@rsSCS many-body dispersion energy for finite molecules.
//!
//! Ambrosetti, Reilly, DiStasio & Tkatchenko, J. Chem. Phys. 140, 18A508
//! (2014). The construction follows libMBD / pymbd 0.15.0 (`pymbd.screening`,
//! `pymbd.mbd_energy`, libMBD `variant='rsscs'`) term for term, so that the two
//! can be compared with identical inputs:
//!
//! 1. TS inputs per atom: α₀, C6, R_vdW (from volume ratios r_A:
//!    α₀ = r α_free, C6 = r² C6_free, R_vdW = r^{1/3} R_vdW^free), and
//!    ω = 4 C6 / (3 α₀²), α(iu) = α₀ / (1 + (u/ω)²).
//! 2. Range-separated self-consistent screening at every node u_k of the
//!    frequency grid:
//!    α^rsSCS(iu) = Tr[Σ_B ((diag 1/α(iu) + T_SR)⁻¹)_AB] / 3, with
//!    T_SR = (1 − f) T_GG. T_GG is the dipole tensor between Gaussian charge
//!    densities of width σ_A(iu) = (√(2/π) α_A(iu)/3)^{1/3},
//!    σ_AB = √(σ_A² + σ_B²); f is the Fermi function below with the TS radii.
//! 3. α₀^rsSCS = α^rsSCS(u = 0), C6^rsSCS = (3/π) Σ_k w_k α^rsSCS(iu_k)²,
//!    ω^rsSCS = 4 C6^rsSCS / (3 (α₀^rsSCS)²),
//!    R_vdW^rsSCS = R_vdW (α₀^rsSCS/α₀)^{1/3}.
//! 4. Long-range coupled quantum harmonic oscillators:
//!    H = diag(ω²) + ω_A ω_B √(α_A α_B) f T_bare (all rsSCS quantities, f with
//!    the rsSCS radii), E = ½ Σ_p √λ_p − (3/2) Σ_A ω_A.
//!
//! Fermi damping: f(R) = 1 / (1 + exp(−a (R/S_AB − 1))), S_AB = β (R_A + R_B),
//! a = 6 by default (the SUM of the two radii, as libMBD and pymbd do).
//! Note f(0) = 1/(1 + e^a) ≈ 2.5e-3 at a = 6, so β → ∞ does NOT switch the
//! long-range coupling off completely; it scales it by 1/(1 + e^a).
//!
//! Frequency grid: libMBD's default — n = 15 Gauss–Legendre nodes mapped by
//! u = L(1+x)/(1−x), L = 0.6, plus a u = 0 node of weight 0 that supplies the
//! static α (see [`mbd_freq_grid`]).
//!
//! T_bare sign: T_ij = (δ_ij R² − 3 R_i R_j)/R⁵, the convention in which
//! A⁻¹ + T is the screening operator (same as [`super::mbd`]).
//!
//! Unlike [`super::mbd::dipole_coupling_tensor`] (which keeps its
//! Abramowitz–Stegun erf for the `[rpa] c6_source = "mbd"` path), this module
//! uses the C library `erf` (double precision).
//!
//! Scope: energy only, finite systems (no lattice, no gradients).

use ferric_core::FerricError;
use ndarray::Array2;
use ndarray_linalg::{Eigh, Inverse, UPLO};

use crate::dispersion::free_atom_ref::{ts_free_atom, ts_free_atom_r_vdw};
use crate::quadrature::gauss_legendre_nodes;

// libm's `erf` (C99), already linked into every ferric binary through
// libint2/OpenBLAS — full double precision, the same choice as
// ferric-scf/src/cosmo.rs and ferric-pcm.
extern "C" {
    fn erf(x: f64) -> f64;
}

fn erf_exact(x: f64) -> f64 {
    // SAFETY: `erf` is a pure C99 libm function of one double.
    unsafe { erf(x) }
}

/// Fermi damping steepness a of MBD@rsSCS (Ambrosetti 2014; libMBD default).
pub const MBD_FERMI_A_DEFAULT: f64 = 6.0;
/// Number of Gauss–Legendre frequency nodes (libMBD default; the grid also
/// carries a u = 0 node, see [`mbd_freq_grid`]).
pub const MBD_N_FREQ_DEFAULT: usize = 15;
/// Scale L of libMBD's frequency map u = L(1+x)/(1−x).
pub const MBD_FREQ_GRID_L: f64 = 0.6;

/// Published MBD@rsSCS range-separation parameter β for a functional.
///
/// Ambrosetti, Reilly, DiStasio & Tkatchenko, JCP 140, 18A508 (2014):
/// PBE 0.83, PBE0 0.85, HSE06 0.85. Matching is case-insensitive and ignores
/// `-`, `_` and spaces. Any other name is an error: β is fitted per functional,
/// and substituting another functional's value would silently change the
/// energy. Pass β explicitly for an unlisted functional.
pub fn mbd_rsscs_beta_for_functional(name: &str) -> Result<f64, FerricError> {
    let key = name.to_ascii_lowercase().replace(['-', '_', ' '], "");
    match key.as_str() {
        "pbe" => Ok(0.83),
        "pbe0" => Ok(0.85),
        "hse06" => Ok(0.85),
        _ => Err(FerricError::General(format!(
            "no published MBD@rsSCS beta for functional '{name}' (known: PBE 0.83, \
             PBE0 0.85, HSE06 0.85; Ambrosetti et al., JCP 140, 18A508 (2014)). \
             Supply beta explicitly; refusing to substitute another functional's value."
        ))),
    }
}

/// Settings of an MBD@rsSCS evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MbdRsscsConfig {
    /// Range-separation parameter β (functional dependent).
    pub beta: f64,
    /// Fermi damping steepness a.
    pub a: f64,
    /// Number of Gauss–Legendre frequency nodes (the u = 0 node is extra).
    pub n_freq: usize,
}

impl MbdRsscsConfig {
    /// Explicit β with the libMBD defaults a = 6, 15 frequency nodes.
    pub fn with_beta(beta: f64) -> Self {
        Self {
            beta,
            a: MBD_FERMI_A_DEFAULT,
            n_freq: MBD_N_FREQ_DEFAULT,
        }
    }

    /// The published β of `functional` (see [`mbd_rsscs_beta_for_functional`]).
    pub fn for_functional(functional: &str) -> Result<Self, FerricError> {
        Ok(Self::with_beta(mbd_rsscs_beta_for_functional(functional)?))
    }

    fn validate(&self) -> Result<(), FerricError> {
        if !(self.beta.is_finite() && self.beta > 0.0) {
            return Err(FerricError::General(format!(
                "MBD@rsSCS: beta must be finite and > 0, got {}",
                self.beta
            )));
        }
        if !(self.a.is_finite() && self.a > 0.0) {
            return Err(FerricError::General(format!(
                "MBD@rsSCS: Fermi steepness a must be finite and > 0, got {}",
                self.a
            )));
        }
        if self.n_freq == 0 {
            return Err(FerricError::General(
                "MBD@rsSCS: n_freq must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Per-atom TS inputs of MBD@rsSCS (atomic units).
#[derive(Debug, Clone, PartialEq)]
pub struct MbdAtomParams {
    /// Static polarizability α₀ per atom.
    pub alpha_0: Vec<f64>,
    /// C6 coefficient per atom.
    pub c6: Vec<f64>,
    /// van der Waals radius (Bohr) per atom.
    pub r_vdw: Vec<f64>,
}

/// TS per-atom (α₀, C6, R_vdW) from Hirshfeld volume ratios r_A = V_A/V_free:
/// α₀ = r α_free, C6 = r² C6_free, R_vdW = r^{1/3} R_vdW^free
/// ([`ts_free_atom`], [`ts_free_atom_r_vdw`]).
///
/// # Errors
///
/// Length mismatch, a non-positive or non-finite ratio, or an element outside
/// the Z = 1..=54 table (no fallback value is substituted).
pub fn ts_params_from_volume_ratios(
    z: &[usize],
    volume_ratios: &[f64],
) -> Result<MbdAtomParams, FerricError> {
    if z.len() != volume_ratios.len() {
        return Err(FerricError::General(format!(
            "MBD@rsSCS: {} atomic numbers but {} volume ratios",
            z.len(),
            volume_ratios.len()
        )));
    }
    let mut p = MbdAtomParams {
        alpha_0: Vec::with_capacity(z.len()),
        c6: Vec::with_capacity(z.len()),
        r_vdw: Vec::with_capacity(z.len()),
    };
    for (i, (&za, &r)) in z.iter().zip(volume_ratios).enumerate() {
        if !(r.is_finite() && r > 0.0) {
            return Err(FerricError::General(format!(
                "MBD@rsSCS: volume ratio of atom {i} (Z={za}) must be finite and > 0, got {r}"
            )));
        }
        let (alpha_free, c6_free, _) = ts_free_atom(za).ok_or_else(|| out_of_table(i, za))?;
        let r_free = ts_free_atom_r_vdw(za).ok_or_else(|| out_of_table(i, za))?;
        p.alpha_0.push(r * alpha_free);
        p.c6.push(r * r * c6_free);
        p.r_vdw.push(r.cbrt() * r_free);
    }
    Ok(p)
}

fn out_of_table(i: usize, z: usize) -> FerricError {
    FerricError::General(format!(
        "MBD@rsSCS: no free-atom TS reference (alpha, C6, R_vdW) for atom {i} (Z={z}); \
         the table covers Z=1..=54 only"
    ))
}

/// libMBD's imaginary-frequency grid: `[0, u_n, …, u_1]` with weights
/// `[0, w_n, …, w_1]`, where (u_k, w_k) are `n` Gauss–Legendre nodes mapped by
/// u = L(1+x)/(1−x), L = [`MBD_FREQ_GRID_L`] (pymbd `freq_grid`: same order).
/// The u = 0 node has weight 0; it only supplies the static polarizability.
pub fn mbd_freq_grid(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (u, w) = gauss_legendre_nodes(n, MBD_FREQ_GRID_L);
    let mut freqs = Vec::with_capacity(n + 1);
    let mut weights = Vec::with_capacity(n + 1);
    freqs.push(0.0);
    weights.push(0.0);
    freqs.extend(u.iter().rev());
    weights.extend(w.iter().rev());
    (freqs, weights)
}

/// Result of [`mbd_rsscs_energy`] / [`mbd_rsscs_energy_from_params`]
/// (atomic units; per-atom vectors in input order).
#[derive(Debug, Clone)]
pub struct MbdRsscsResult {
    /// MBD@rsSCS dispersion energy (Hartree).
    pub energy: f64,
    /// TS inputs (before screening).
    pub ts: MbdAtomParams,
    /// Screened static polarizability α₀^rsSCS.
    pub alpha_0_rsscs: Vec<f64>,
    /// Screened C6^rsSCS.
    pub c6_rsscs: Vec<f64>,
    /// Screened radius R_vdW^rsSCS (Bohr).
    pub r_vdw_rsscs: Vec<f64>,
    /// Screened characteristic frequency ω^rsSCS = 4 C6^rsSCS / (3 α₀^rsSCS²).
    pub omega_rsscs: Vec<f64>,
    /// Smallest eigenvalue of the coupled-oscillator matrix (> 0 by the guard).
    pub min_eigenvalue: f64,
    /// The β, a and grid size used.
    pub config: MbdRsscsConfig,
}

/// MBD@rsSCS energy from atomic numbers, coordinates (Bohr) and Hirshfeld
/// volume ratios. See the module doc for the construction.
pub fn mbd_rsscs_energy(
    z: &[usize],
    positions: &[[f64; 3]],
    volume_ratios: &[f64],
    config: &MbdRsscsConfig,
) -> Result<MbdRsscsResult, FerricError> {
    let params = ts_params_from_volume_ratios(z, volume_ratios)?;
    mbd_rsscs_energy_from_params(positions, &params, config)
}

/// Fermi damping f(R) = 1/(1 + exp(−a (R/S − 1))).
fn fermi(r: f64, s: f64, a: f64) -> f64 {
    1.0 / (1.0 + (-a * (r / s - 1.0)).exp())
}

/// Bare dipole tensor block (δ_ij R² − 3 R_i R_j)/R⁵ for separation `d`.
fn t_bare(d: &[f64; 3], r: f64) -> [[f64; 3]; 3] {
    let r2 = r * r;
    let r5 = r2 * r2 * r;
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let kron = if i == j { r2 } else { 0.0 };
            (kron - 3.0 * d[i] * d[j]) / r5
        })
    })
}

/// Gaussian-damped dipole tensor (pymbd `T_erf_coulomb`): with ζ = R/σ,
/// θ = 2ζ/√π e^{−ζ²}: T_GG = (erf ζ − θ) T_bare + 2ζ²θ R R^T / R⁵.
fn t_gg(d: &[f64; 3], r: f64, sigma_ab: f64) -> [[f64; 3]; 3] {
    const SQRT_PI: f64 = 1.772_453_850_905_516;
    let bare = t_bare(d, r);
    let zeta = r / sigma_ab;
    let theta = 2.0 * zeta / SQRT_PI * (-zeta * zeta).exp();
    let erf_theta = erf_exact(zeta) - theta;
    let extra = 2.0 * zeta * zeta * theta / r.powi(5);
    std::array::from_fn(|i| std::array::from_fn(|j| erf_theta * bare[i][j] + extra * d[i] * d[j]))
}

fn check_inputs(positions: &[[f64; 3]], p: &MbdAtomParams) -> Result<(), FerricError> {
    let n = positions.len();
    if n == 0 {
        return Err(FerricError::General("MBD@rsSCS: no atoms".to_string()));
    }
    if p.alpha_0.len() != n || p.c6.len() != n || p.r_vdw.len() != n {
        return Err(FerricError::General(format!(
            "MBD@rsSCS: {n} positions but {} alpha_0, {} C6, {} R_vdW",
            p.alpha_0.len(),
            p.c6.len(),
            p.r_vdw.len()
        )));
    }
    for a in 0..n {
        for (what, v) in [
            ("alpha_0", p.alpha_0[a]),
            ("C6", p.c6[a]),
            ("R_vdW", p.r_vdw[a]),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(FerricError::General(format!(
                    "MBD@rsSCS: {what} of atom {a} must be finite and > 0, got {v}"
                )));
            }
        }
        if positions[a].iter().any(|x| !x.is_finite()) {
            return Err(FerricError::General(format!(
                "MBD@rsSCS: non-finite coordinate for atom {a}"
            )));
        }
        for b in 0..a {
            let d = sep(positions, a, b);
            if norm(&d) < 1e-8 {
                return Err(FerricError::General(format!(
                    "MBD@rsSCS: atoms {b} and {a} coincide (separation < 1e-8 Bohr)"
                )));
            }
        }
    }
    Ok(())
}

fn sep(positions: &[[f64; 3]], a: usize, b: usize) -> [f64; 3] {
    std::array::from_fn(|i| positions[a][i] - positions[b][i])
}

fn norm(d: &[f64; 3]) -> f64 {
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Range-separated SCS at every node: returns α^rsSCS(iu_k) as `[k][A]`.
fn rsscs_screen(
    positions: &[[f64; 3]],
    p: &MbdAtomParams,
    cfg: &MbdRsscsConfig,
    freqs: &[f64],
) -> Result<Vec<Vec<f64>>, FerricError> {
    let n = positions.len();
    let omega: Vec<f64> = (0..n)
        .map(|a| 4.0 / 3.0 * p.c6[a] / (p.alpha_0[a] * p.alpha_0[a]))
        .collect();
    let frac = (2.0 / std::f64::consts::PI).sqrt() / 3.0;
    let mut out = Vec::with_capacity(freqs.len());
    for (k, &u) in freqs.iter().enumerate() {
        let alpha: Vec<f64> = (0..n)
            .map(|a| p.alpha_0[a] / (1.0 + (u / omega[a]).powi(2)))
            .collect();
        let sigma: Vec<f64> = alpha.iter().map(|&x| (frac * x).cbrt()).collect();
        let mut c = Array2::<f64>::zeros((3 * n, 3 * n));
        for a in 0..n {
            for i in 0..3 {
                c[(3 * a + i, 3 * a + i)] = 1.0 / alpha[a];
            }
            for b in 0..a {
                let d = sep(positions, a, b);
                let r = norm(&d);
                let s = cfg.beta * (p.r_vdw[a] + p.r_vdw[b]);
                let sr = 1.0 - fermi(r, s, cfg.a);
                let sab = (sigma[a] * sigma[a] + sigma[b] * sigma[b]).sqrt();
                let t = t_gg(&d, r, sab);
                for i in 0..3 {
                    for j in 0..3 {
                        let v = sr * t[i][j];
                        c[(3 * a + i, 3 * b + j)] = v;
                        c[(3 * b + j, 3 * a + i)] = v;
                    }
                }
            }
        }
        let cinv = c.inv().map_err(|e| {
            FerricError::Lapack(format!(
                "MBD@rsSCS screening: matrix diag(1/alpha) + T_SR is singular at \
                 frequency node {k} (u={u}): {e}"
            ))
        })?;
        let mut a_rs = vec![0.0; n];
        for (a, slot) in a_rs.iter_mut().enumerate() {
            let mut tr = 0.0;
            for b in 0..n {
                for i in 0..3 {
                    tr += cinv[(3 * a + i, 3 * b + i)];
                }
            }
            *slot = tr / 3.0;
            if !(slot.is_finite() && *slot > 0.0) {
                return Err(FerricError::General(format!(
                    "MBD@rsSCS screening: screened polarizability of atom {a} is {} at \
                     frequency node {k} (u={u}) — the short-range screening is \
                     unphysical for this geometry/parameter set (atoms too close or \
                     beta too large); refusing to continue with a non-positive alpha",
                    *slot
                )));
            }
        }
        out.push(a_rs);
    }
    Ok(out)
}

/// MBD@rsSCS energy from per-atom TS inputs (α₀, C6, R_vdW) and coordinates
/// (Bohr). See the module doc for the construction.
///
/// # Errors
///
/// Invalid inputs or config; a singular screening matrix; a non-positive
/// screened α; and a coupled-oscillator matrix that is not positive definite
/// (polarization catastrophe: an imaginary mode makes the energy undefined).
pub fn mbd_rsscs_energy_from_params(
    positions: &[[f64; 3]],
    params: &MbdAtomParams,
    config: &MbdRsscsConfig,
) -> Result<MbdRsscsResult, FerricError> {
    config.validate()?;
    check_inputs(positions, params)?;
    let n = positions.len();
    let (freqs, weights) = mbd_freq_grid(config.n_freq);
    debug_assert_eq!(freqs[0], 0.0);

    let a_dyn = rsscs_screen(positions, params, config, &freqs)?;
    let alpha_0_rsscs = a_dyn[0].clone();
    let c6_rsscs: Vec<f64> = (0..n)
        .map(|a| {
            3.0 / std::f64::consts::PI
                * weights
                    .iter()
                    .zip(&a_dyn)
                    .map(|(w, ak)| w * ak[a] * ak[a])
                    .sum::<f64>()
        })
        .collect();
    let r_vdw_rsscs: Vec<f64> = (0..n)
        .map(|a| params.r_vdw[a] * (alpha_0_rsscs[a] / params.alpha_0[a]).cbrt())
        .collect();
    let omega_rsscs: Vec<f64> = (0..n)
        .map(|a| 4.0 / 3.0 * c6_rsscs[a] / (alpha_0_rsscs[a] * alpha_0_rsscs[a]))
        .collect();

    // Long-range coupled QHOs.
    let mut h = Array2::<f64>::zeros((3 * n, 3 * n));
    for a in 0..n {
        for i in 0..3 {
            h[(3 * a + i, 3 * a + i)] = omega_rsscs[a] * omega_rsscs[a];
        }
        for b in 0..a {
            let d = sep(positions, a, b);
            let r = norm(&d);
            let s = config.beta * (r_vdw_rsscs[a] + r_vdw_rsscs[b]);
            let pref = omega_rsscs[a]
                * omega_rsscs[b]
                * (alpha_0_rsscs[a] * alpha_0_rsscs[b]).sqrt()
                * fermi(r, s, config.a);
            let t = t_bare(&d, r);
            for i in 0..3 {
                for j in 0..3 {
                    let v = pref * t[i][j];
                    h[(3 * a + i, 3 * b + j)] = v;
                    h[(3 * b + j, 3 * a + i)] = v;
                }
            }
        }
    }
    let (evals, _) = h
        .eigh(UPLO::Lower)
        .map_err(|e| FerricError::Lapack(format!("MBD@rsSCS: eigendecomposition failed: {e}")))?;
    let min_eigenvalue = evals.iter().copied().fold(f64::INFINITY, f64::min);
    if !(min_eigenvalue > 0.0) {
        return Err(FerricError::General(format!(
            "MBD@rsSCS: the coupled-oscillator matrix is not positive definite \
             (smallest eigenvalue {min_eigenvalue:.3e}): polarization catastrophe, the \
             dispersion energy is undefined for this geometry/parameter set (atoms too \
             close for this beta, or unphysical alpha/C6 inputs)"
        )));
    }
    let energy =
        0.5 * evals.iter().map(|&l| l.sqrt()).sum::<f64>() - 1.5 * omega_rsscs.iter().sum::<f64>();

    Ok(MbdRsscsResult {
        energy,
        ts: params.clone(),
        alpha_0_rsscs,
        c6_rsscs,
        r_vdw_rsscs,
        omega_rsscs,
        min_eigenvalue,
        config: *config,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(beta: f64) -> MbdRsscsConfig {
        MbdRsscsConfig::with_beta(beta)
    }

    /// Closed form for two isotropic QHOs whose coupling is diagonal on the z
    /// axis (x, y: t_perp; z: t_par): per direction a 2×2 eigenproblem.
    /// Independent of the 3N×3N eigh in the production path.
    fn dimer_energy_closed_form(a: [f64; 2], w: [f64; 2], t_perp: f64, t_par: f64) -> f64 {
        let mut e = 0.0;
        for t in [t_perp, t_perp, t_par] {
            let c = w[0] * w[1] * (a[0] * a[1]).sqrt() * t;
            let m = 0.5 * (w[0] * w[0] + w[1] * w[1]);
            let dd = (0.25 * (w[0] * w[0] - w[1] * w[1]).powi(2) + c * c).sqrt();
            e += 0.5 * ((m + dd).sqrt() + (m - dd).sqrt() - w[0] - w[1]);
        }
        e
    }

    /// Per-direction screened α of a dimer: row sums of [[1/a, t],[t, 1/b]]⁻¹.
    fn dimer_screen(a: f64, b: f64, t: f64) -> (f64, f64) {
        let det = 1.0 / (a * b) - t * t;
        ((1.0 / b - t) / det, (1.0 / a - t) / det)
    }

    /// Whole-pipeline closed form for a dimer on the z axis: screening per
    /// direction (2×2), C6/α/R/ω^rsSCS, then the per-direction energy.
    fn dimer_rsscs_closed_form(z: [usize; 2], r: f64, c: &MbdRsscsConfig) -> f64 {
        let p = ts_params_from_volume_ratios(&z, &[1.0, 1.0]).unwrap();
        let (f, w) = mbd_freq_grid(c.n_freq);
        let om: Vec<f64> = (0..2)
            .map(|i| 4.0 / 3.0 * p.c6[i] / p.alpha_0[i].powi(2))
            .collect();
        let s = c.beta * (p.r_vdw[0] + p.r_vdw[1]);
        let sr = 1.0 - fermi(r, s, c.a);
        let mut a_iso = vec![[0.0f64; 2]; f.len()];
        for (k, &u) in f.iter().enumerate() {
            let al: Vec<f64> = (0..2)
                .map(|i| p.alpha_0[i] / (1.0 + (u / om[i]).powi(2)))
                .collect();
            let sig: Vec<f64> = al
                .iter()
                .map(|&x| ((2.0 / std::f64::consts::PI).sqrt() * x / 3.0).cbrt())
                .collect();
            let zeta = r / (sig[0] * sig[0] + sig[1] * sig[1]).sqrt();
            let theta = 2.0 * zeta / std::f64::consts::PI.sqrt() * (-zeta * zeta).exp();
            let e = erf_exact(zeta) - theta;
            // T_GG on the z axis: perp = e/R³, par = (−2e + 2ζ²θ)/R³.
            let tp = sr * e / r.powi(3);
            let tz = sr * (-2.0 * e + 2.0 * zeta * zeta * theta) / r.powi(3);
            let (ap, bp) = dimer_screen(al[0], al[1], tp);
            let (az, bz) = dimer_screen(al[0], al[1], tz);
            a_iso[k] = [(2.0 * ap + az) / 3.0, (2.0 * bp + bz) / 3.0];
        }
        let a0 = a_iso[0];
        let c6: Vec<f64> = (0..2)
            .map(|i| {
                3.0 / std::f64::consts::PI
                    * w.iter()
                        .zip(&a_iso)
                        .map(|(wk, ak)| wk * ak[i] * ak[i])
                        .sum::<f64>()
            })
            .collect();
        let rr: Vec<f64> = (0..2)
            .map(|i| p.r_vdw[i] * (a0[i] / p.alpha_0[i]).cbrt())
            .collect();
        let ww = [
            4.0 / 3.0 * c6[0] / (a0[0] * a0[0]),
            4.0 / 3.0 * c6[1] / (a0[1] * a0[1]),
        ];
        let flr = fermi(r, c.beta * (rr[0] + rr[1]), c.a);
        dimer_energy_closed_form(a0, ww, flr / r.powi(3), -2.0 * flr / r.powi(3))
    }

    #[test]
    fn beta_table_is_strict() {
        assert_eq!(mbd_rsscs_beta_for_functional("PBE").unwrap(), 0.83);
        assert_eq!(mbd_rsscs_beta_for_functional("pbe0").unwrap(), 0.85);
        assert_eq!(mbd_rsscs_beta_for_functional("HSE-06").unwrap(), 0.85);
        let err = mbd_rsscs_beta_for_functional("b3lyp")
            .unwrap_err()
            .to_string();
        assert!(err.contains("b3lyp") && err.contains("beta"), "{err}");
        assert!(mbd_rsscs_beta_for_functional("hse").is_err());
    }

    #[test]
    fn freq_grid_matches_libmbd_layout() {
        let (f, w) = mbd_freq_grid(15);
        assert_eq!(f.len(), 16);
        assert_eq!((f[0], w[0]), (0.0, 0.0));
        // Descending after the u=0 node, like pymbd's freq_grid.
        assert!(f[1..].windows(2).all(|p| p[0] > p[1]));
        // ∫_0^∞ du / (1 + u²) = π/2 on the mapped grid: pymbd's grid gives
        // 4.5e-12 abs error.
        let s: f64 = f.iter().zip(&w).map(|(u, wk)| wk / (1.0 + u * u)).sum();
        assert!((s - std::f64::consts::FRAC_PI_2).abs() < 1e-10, "{s}");
    }

    /// EXACTNESS ANCHOR: one atom has no partner — no screening, no coupling,
    /// E = 0 and α₀^rsSCS = α₀, R^rsSCS = R exactly.
    #[test]
    fn single_atom_energy_is_zero() {
        for z in [1usize, 6, 8, 17, 35] {
            let res = mbd_rsscs_energy(&[z], &[[0.3, -0.2, 1.0]], &[0.87], &cfg(0.83)).unwrap();
            let w = res.omega_rsscs[0];
            assert!(res.energy.abs() < 1e-14 * w, "Z={z}: E={}", res.energy);
            let da = (res.alpha_0_rsscs[0] - res.ts.alpha_0[0]).abs() / res.ts.alpha_0[0];
            assert!(da < 1e-15, "Z={z}: alpha_0^rsSCS vs alpha_0 {da:.2e}");
            assert!((res.r_vdw_rsscs[0] - res.ts.r_vdw[0]).abs() < 1e-15 * res.ts.r_vdw[0]);
        }
    }

    /// EXACTNESS ANCHOR: no long-range coupling ⇒ E = 0. With f(0) =
    /// 1/(1+e^a), β → ∞ alone leaves 1/(1+e^6) ≈ 2.5e-3 of the coupling, so the
    /// trivial limit needs a steep Fermi function too: β = 1e3, a = 200 makes
    /// f ≤ e^{-199} for every pair here.
    #[test]
    fn no_long_range_coupling_gives_zero_energy() {
        let z = [8usize, 1, 1];
        let pos = [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]];
        let c = MbdRsscsConfig {
            beta: 1e3,
            a: 200.0,
            n_freq: 15,
        };
        let res = mbd_rsscs_energy(&z, &pos, &[0.85, 0.6, 0.6], &c).unwrap();
        let scale: f64 = res.omega_rsscs.iter().sum();
        assert!(res.energy.abs() < 1e-14 * scale, "E={}", res.energy);
        // And the default a = 6 with huge β does NOT vanish (f(0) = 1/(1+e^6)).
        let c6 = MbdRsscsConfig { a: 6.0, ..c };
        let e6 = mbd_rsscs_energy(&z, &pos, &[0.85, 0.6, 0.6], &c6)
            .unwrap()
            .energy;
        assert!(
            e6 < -1e-7,
            "a=6, beta=1e3 should keep a scaled coupling: {e6}"
        );
    }

    /// EXACTNESS ANCHOR: far-apart atoms recover the London limit
    /// E → −C6_AB/R⁶ with C6_AB = (3/2) α_A α_B ω_A ω_B/(ω_A+ω_B) built from
    /// the rsSCS per-atom values; the leading correction is O(R⁻⁶) relative
    /// (≈ (α/R³)² ≈ 4e-8 for C at 40 Bohr). Screening is off too: (1 − f) ≈ e^{-34}.
    #[test]
    fn far_dimer_recovers_london_limit() {
        for (za, zb) in [(6usize, 6usize), (6, 1), (8, 7)] {
            let r = 40.0;
            let res = mbd_rsscs_energy(
                &[za, zb],
                &[[0.0, 0.0, 0.0], [0.0, 0.0, r]],
                &[1.0, 1.0],
                &cfg(0.83),
            )
            .unwrap();
            for i in 0..2 {
                let d = (res.alpha_0_rsscs[i] - res.ts.alpha_0[i]).abs() / res.ts.alpha_0[i];
                assert!(d < 1e-12, "Z={za}-{zb}: screening at 40 Bohr {d:.2e}");
            }
            let (a, w) = (&res.alpha_0_rsscs, &res.omega_rsscs);
            let c6ab = 1.5 * a[0] * a[1] * w[0] * w[1] / (w[0] + w[1]);
            let ratio = -res.energy * r.powi(6) / c6ab;
            assert!(
                (ratio - 1.0).abs() < 1e-6,
                "Z={za}-{zb}: E R^6/(-C6) = {ratio}"
            );
        }
    }

    /// INDEPENDENT CONSTRUCTION: a dimer on the z axis decouples into 2×2
    /// problems per Cartesian direction for both the screening and the energy;
    /// the production 3N×3N inverse/eigh path must agree with it at every
    /// separation regime (strong SR screening, crossover, β huge with a = 6).
    #[test]
    fn dimer_matches_per_direction_closed_form() {
        for (z, r, beta) in [
            ([6usize, 6usize], 2.5, 0.83),
            ([6, 6], 5.0, 0.83),
            ([6, 1], 2.0, 0.85),
            ([8, 1], 1.8, 0.83),
            ([6, 6], 8.0, 0.83),
            ([6, 6], 4.0, 1e6),
        ] {
            let c = cfg(beta);
            let res =
                mbd_rsscs_energy(&z, &[[0.0, 0.0, 0.0], [0.0, 0.0, r]], &[1.0, 1.0], &c).unwrap();
            let want = dimer_rsscs_closed_form(z, r, &c);
            // E = ½Σ√λ − (3/2)Σω cancels ~0.9 Ha of Σω down to 1e-4..1e-7 Ha,
            // so the floor is ABSOLUTE: ~eps·Σω ≈ 1e-16 Ha. The same closed form
            // in numpy agrees with pymbd to 1e-17..1.7e-16 Ha on these six
            // cases (6e-13..1.5e-12 rel; 7.6e-10 rel at beta=1e6 where
            // E = -7.3e-8). Bar 1e-14 Ha = ~100x that floor (see the mutations
            // listed in tests/validation_mbd_rsscs.rs for what it catches).
            let diff = (res.energy - want).abs();
            assert!(
                diff < 1e-14,
                "{z:?} R={r} beta={beta}: E={} closed form {want} |diff| {diff:.2e}",
                res.energy
            );
        }
    }

    /// Off-axis orientation: the energy is rotation invariant (catches a T
    /// built from one Cartesian component only).
    #[test]
    fn energy_is_rotation_invariant() {
        let z = [6usize, 8, 1];
        let base = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.3], [1.9, 0.0, -0.6]];
        let (c, s) = (0.6_f64.cos(), 0.6_f64.sin());
        let rot: Vec<[f64; 3]> = base
            .iter()
            .map(|p| [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])
            .map(|p| [p[0], c * p[1] - s * p[2], s * p[1] + c * p[2]])
            .collect();
        let e0 = mbd_rsscs_energy(&z, &base, &[0.9, 0.8, 0.6], &cfg(0.83))
            .unwrap()
            .energy;
        let e1 = mbd_rsscs_energy(&z, &rot, &[0.9, 0.8, 0.6], &cfg(0.83))
            .unwrap()
            .energy;
        // Roundoff alone measured 1.2e-12 relative (4.4e-16 Ha); a T built from one
        // Cartesian component would miss at O(1).
        assert!((e0 - e1).abs() < 1e-10 * e0.abs(), "{e0} vs {e1}");
    }

    #[test]
    fn invalid_inputs_error_instead_of_panicking() {
        let pos = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]];
        assert!(mbd_rsscs_energy(&[6, 55], &pos, &[1.0, 1.0], &cfg(0.83)).is_err());
        assert!(mbd_rsscs_energy(&[6, 6], &pos, &[1.0, 0.0], &cfg(0.83)).is_err());
        assert!(mbd_rsscs_energy(&[6, 6], &pos, &[1.0], &cfg(0.83)).is_err());
        assert!(mbd_rsscs_energy(&[6, 6], &pos, &[1.0, 1.0], &cfg(0.0)).is_err());
        assert!(mbd_rsscs_energy(&[], &[], &[], &cfg(0.83)).is_err());
        let same = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let e = mbd_rsscs_energy(&[6, 6], &same, &[1.0, 1.0], &cfg(0.83)).unwrap_err();
        assert!(e.to_string().contains("coincide"), "{e}");
    }

    /// POLARIZATION CATASTROPHE GUARD: two very polarizable atoms (Na-like α
    /// scaled up) close together with almost no damping (tiny β) drive the
    /// lowest coupled mode through zero. Must be an Err naming the cause,
    /// never a NaN energy.
    #[test]
    fn polarization_catastrophe_is_an_error() {
        let p = MbdAtomParams {
            alpha_0: vec![400.0, 400.0],
            c6: vec![4000.0, 4000.0],
            r_vdw: vec![3.0, 3.0],
        };
        let c = MbdRsscsConfig {
            beta: 0.05,
            a: 6.0,
            n_freq: 15,
        };
        let res = mbd_rsscs_energy_from_params(&[[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], &p, &c);
        let msg = res.expect_err("catastrophe must error").to_string();
        assert!(
            msg.contains("polarization catastrophe") || msg.contains("screened polarizability"),
            "{msg}"
        );
    }
}
