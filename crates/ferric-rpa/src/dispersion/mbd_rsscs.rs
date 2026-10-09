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
//! Analytic gradient: [`mbd_rsscs_gradient_from_params`] /
//! [`mbd_rsscs_gradient`] differentiate the same forward pass in reverse mode
//! (adjoint; derivation on `rsscs_backward`), giving dE/dR_A with the TS
//! inputs fixed plus dE/dα₀, dE/dC6, dE/dR_vdW (and dE/dr for volume ratios),
//! at O(n_freq (3N)³) cost — the cost class of the energy.
//!
//! Scope: finite systems only (no lattice).

use ferric_core::FerricError;
use ndarray::{Array1, Array2, Axis};
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

/// Range-separated SCS at every node: returns α^rsSCS(iu_k) as `[k][A]`, and,
/// when `keep` is set, the inverse screening matrices B(u_k) = C(u_k)⁻¹ that
/// the backward pass of [`mbd_rsscs_gradient_from_params`] needs (otherwise
/// an empty vector; the energy path never holds them).
fn rsscs_screen(
    positions: &[[f64; 3]],
    p: &MbdAtomParams,
    cfg: &MbdRsscsConfig,
    freqs: &[f64],
    keep: bool,
) -> Result<(Vec<Vec<f64>>, Vec<Array2<f64>>), FerricError> {
    let n = positions.len();
    let omega: Vec<f64> = (0..n)
        .map(|a| 4.0 / 3.0 * p.c6[a] / (p.alpha_0[a] * p.alpha_0[a]))
        .collect();
    let frac = (2.0 / std::f64::consts::PI).sqrt() / 3.0;
    let mut out = Vec::with_capacity(freqs.len());
    let mut kept = Vec::with_capacity(if keep { freqs.len() } else { 0 });
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
        if keep {
            kept.push(cinv);
        }
    }
    Ok((out, kept))
}

/// Forward-pass intermediates the gradient's backward pass reads.
struct Tape {
    freqs: Vec<f64>,
    weights: Vec<f64>,
    /// α^rsSCS(iu_k) as `[k][A]`.
    a_dyn: Vec<Vec<f64>>,
    /// B(u_k) = (diag 1/α(iu_k) + T_SR)⁻¹ per node.
    binv: Vec<Array2<f64>>,
    /// Eigenpairs of the coupled-oscillator matrix H.
    h_evals: Array1<f64>,
    h_evecs: Array2<f64>,
}

/// The one forward pass shared by the energy and the gradient. With
/// `keep = false` it holds no per-node matrices and returns no tape.
fn rsscs_forward(
    positions: &[[f64; 3]],
    params: &MbdAtomParams,
    config: &MbdRsscsConfig,
    keep: bool,
) -> Result<(MbdRsscsResult, Option<Tape>), FerricError> {
    config.validate()?;
    check_inputs(positions, params)?;
    let n = positions.len();
    let (freqs, weights) = mbd_freq_grid(config.n_freq);
    debug_assert_eq!(freqs[0], 0.0);

    let (a_dyn, binv) = rsscs_screen(positions, params, config, &freqs, keep)?;
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
    let (evals, evecs) = h
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

    let result = MbdRsscsResult {
        energy,
        ts: params.clone(),
        alpha_0_rsscs,
        c6_rsscs,
        r_vdw_rsscs,
        omega_rsscs,
        min_eigenvalue,
        config: *config,
    };
    let tape = keep.then(|| Tape {
        freqs,
        weights,
        a_dyn,
        binv,
        h_evals: evals,
        h_evecs: evecs,
    });
    Ok((result, tape))
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
    Ok(rsscs_forward(positions, params, config, false)?.0)
}

/// Analytic gradient of the MBD@rsSCS energy.
#[derive(Debug, Clone)]
pub struct MbdRsscsGradient {
    /// The forward result; `result.energy` is BIT-IDENTICAL to
    /// [`mbd_rsscs_energy_from_params`] for the same inputs (both run the
    /// same forward code).
    pub result: MbdRsscsResult,
    /// dE/dR_A (Hartree/Bohr), per-atom TS inputs held fixed.
    pub d_positions: Vec<[f64; 3]>,
    /// dE/dα₀_A of the UNSCREENED TS input (`MbdAtomParams::alpha_0`).
    pub d_alpha_0: Vec<f64>,
    /// dE/dC6_A of the UNSCREENED TS input (`MbdAtomParams::c6`).
    pub d_c6: Vec<f64>,
    /// dE/dR_vdW_A of the UNSCREENED TS input (`MbdAtomParams::r_vdw`).
    pub d_r_vdw: Vec<f64>,
}

/// Fermi partials (∂f/∂R, ∂f/∂S) of f = 1/(1 + e^{−z}), z = a (R/S − 1):
/// f' = e^{−z}/(1 + e^{−z})² (= f(1 − f) without its cancellation),
/// ∂f/∂R = f' a/S, ∂f/∂S = −f' a R/S².
fn fermi_partials(r: f64, s: f64, a: f64) -> (f64, f64) {
    let e = (-a * (r / s - 1.0)).exp();
    // e = +inf (R ≪ S, f = 0 to double precision): f' = 0, not inf/inf.
    let fp = if e.is_finite() {
        e / ((1.0 + e) * (1.0 + e))
    } else {
        0.0
    };
    (fp * a / s, -fp * a * r / (s * s))
}

/// Symmetric part of a 3×3 adjoint; the tensors it is contracted with are
/// symmetric, so only this part carries a derivative.
fn sym3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| 0.5 * (m[i][j] + m[j][i])))
}

/// Vector–Jacobian product of the bare dipole tensor. With M symmetric,
/// L = Σ M_ij T_ij = tr M / R³ − 3 q/R⁵, q = dᵀMd, and
/// ∂L/∂d = −3 tr(M) d/R⁵ − 6 M d/R⁵ + 15 q d/R⁷.
/// Returns (L, ∂L/∂d, q, M d).
fn t_bare_vjp(d: &[f64; 3], r: f64, m: &[[f64; 3]; 3]) -> (f64, [f64; 3], f64, [f64; 3]) {
    let ms = sym3(m);
    let tr = ms[0][0] + ms[1][1] + ms[2][2];
    let md: [f64; 3] = std::array::from_fn(|i| (0..3).map(|j| ms[i][j] * d[j]).sum());
    let q = d[0] * md[0] + d[1] * md[1] + d[2] * md[2];
    let r2 = r * r;
    let r5 = r2 * r2 * r;
    let r7 = r5 * r2;
    let l = tr / (r2 * r) - 3.0 * q / r5;
    let g =
        std::array::from_fn(|x| -3.0 * tr * d[x] / r5 - 6.0 * md[x] / r5 + 15.0 * q * d[x] / r7);
    (l, g, q, md)
}

/// Vector–Jacobian product of [`t_gg`]: for the adjoint M returns
/// (∂L/∂d, ∂L/∂σ) of L = Σ M_ij T_GG,ij(d, σ).
///
/// T_GG = g₁ T_bare + g₂ d dᵀ with ζ = R/σ, θ = 2ζ/√π e^{−ζ²},
/// g₁ = erf ζ − θ, g₂ = 2ζ²θ/R⁵. Using d erf/dζ = 2/√π e^{−ζ²},
/// dθ/dζ = 2/√π e^{−ζ²}(1 − 2ζ²), d(ζ²θ)/dζ = θ(3ζ − 2ζ³), ∂ζ/∂R = ζ/R,
/// ∂ζ/∂σ = −ζ/σ:
///   ∂g₁/∂R = 2ζ²θ/R,              ∂g₁/∂σ = −2ζ²θ/σ,
///   ∂g₂/∂R = −4ζ²θ(1 + ζ²)/R⁶,    ∂g₂/∂σ = −2ζ²θ(3 − 2ζ²)/(σ R⁵).
/// With L_b = Σ M T_bare and q = dᵀMd (M symmetrized):
///   ∂L/∂d = g₁ ∂L_b/∂d + 2 g₂ M d + (L_b ∂g₁/∂R + q ∂g₂/∂R) d/R,
///   ∂L/∂σ = L_b ∂g₁/∂σ + q ∂g₂/∂σ.
fn t_gg_vjp(d: &[f64; 3], r: f64, sigma: f64, m: &[[f64; 3]; 3]) -> ([f64; 3], f64) {
    const SQRT_PI: f64 = 1.772_453_850_905_516;
    let (lb, dlb, q, md) = t_bare_vjp(d, r, m);
    let zeta = r / sigma;
    let theta = 2.0 * zeta / SQRT_PI * (-zeta * zeta).exp();
    let g1 = erf_exact(zeta) - theta;
    let r5 = r.powi(5);
    let z2t = zeta * zeta * theta;
    let g2 = 2.0 * z2t / r5;
    let dg1_dr = 2.0 * z2t / r;
    let dg1_ds = -2.0 * z2t / sigma;
    let dg2_dr = -4.0 * z2t * (1.0 + zeta * zeta) / (r5 * r);
    let dg2_ds = -2.0 * z2t * (3.0 - 2.0 * zeta * zeta) / (sigma * r5);
    let radial = (lb * dg1_dr + q * dg2_dr) / r;
    let gd = std::array::from_fn(|x| g1 * dlb[x] + 2.0 * g2 * md[x] + radial * d[x]);
    (gd, lb * dg1_ds + q * dg2_ds)
}

/// Reverse-mode (adjoint) pass over the tape. Cost O(n_freq (3N)²) on top of
/// one (3N)³ product for dE/dH; no per-coordinate matrices are formed.
///
/// Adjoint steps (x̄ = ∂E/∂x; every step mirrors a line of `rsscs_forward`):
/// 1. E = ½ Σ√λ − (3/2) Σ ω̃ with dλ_p = v_pᵀ dH v_p ⇒ ∂E/∂λ_p = ¼ λ_p^{−1/2},
///    H̄ = G = ¼ V diag(λ^{−1/2}) Vᵀ (= ¼ H^{−1/2}, valid with degeneracies
///    since E = ½ tr H^{1/2}), and ω̄̃ = −3/2. Check: one atom, H = ω̃² I₃,
///    ω̄̃ = 2ω̃·3/(4ω̃) − 3/2 = 0, as E ≡ 0 requires.
/// 2. H diagonal ω̃_A²: ω̄̃_A += 2 ω̃_A Σ_i G_(A,i)(A,i). Off-diagonal pair
///    A > B: the block value P t_ij sits at (A,i),(B,j) AND (B,j),(A,i), so
///    its adjoint is v̄_ij = G_(A,i)(B,j) + G_(B,j)(A,i). P̄ = Σ v̄·T_bare,
///    T̄_bare = P v̄ (→ positions); P = ω̃_A ω̃_B √(α̃_A α̃_B) f(R; β(R̃_A+R̃_B))
///    → ω̄̃, ᾱ̃ (½ P̄ P/α̃), f̄ → (R̄, S̄) via [`fermi_partials`], R̄̃_{A,B} += β S̄.
/// 3. ω̃ = 4/3 C̃6/α̃² → C̄̃6 += ω̄̃·4/(3α̃²), ᾱ̃ −= ω̄̃·2ω̃/α̃.
///    R̃ = R (α̃/α₀)^{1/3} → R̄ += R̄̃ R̃/R, ᾱ̃ += R̄̃ R̃/(3α̃), ᾱ₀ −= R̄̃ R̃/(3α₀).
/// 4. Per node k: ā_A = C̄̃6_A (6/π) w_k ã_A(u_k) (+ ᾱ̃_A at the u = 0 node).
///    ã_A = ⅓ Σ_{B,i} B_(A,i)(B,i) with B = C⁻¹ ⇒ C̄ = −Bᵀ Ḃ Bᵀ,
///    Ḃ_(A,i)(B,j) = ā_A/3 δ_ij. Ḃ = diag(ā/3) (1 1ᵀ ⊗ I₃) factors, so
///    C̄_mn = −Σ_i z_mi w_ni with z_mi = Σ_A B_(A,i)m ā_A/3 and
///    w_ni = Σ_B B_n(B,i) (rank 3, only the entries needed are formed).
///    C diagonal 1/α_A → ᾱ_A −= Σ_i C̄_(A,i)(A,i)/α_A². Off-diagonal pair:
///    v̄_ij = C̄_(A,i)(B,j) + C̄_(B,j)(A,i), v = (1 − f(R; β(R_A+R_B))) T_GG(d, σ_AB)
///    → f̄ = −Σ v̄·T_GG, T̄_GG = (1 − f) v̄ → [`t_gg_vjp`] (d, σ_AB);
///    σ_AB = √(σ_A²+σ_B²) → σ̄_A += σ̄_AB σ_A/σ_AB; σ_A = (frac α_A)^{1/3}
///    → ᾱ_A += σ̄_A σ_A/(3α_A); α_A = α₀/(1 + (u/ω)²) → ᾱ₀ += ᾱ_A α_A/α₀,
///    ω̄ += ᾱ_A 2 α_A² u²/(α₀ ω³).
/// 5. ω = 4/3 C6/α₀² → C̄6 += ω̄·4/(3α₀²), ᾱ₀ −= ω̄·2ω/α₀.
/// Positions: every pair term depends on d = R_A − R_B only, so ∂/∂d goes
/// to A with + and to B with −; R = |d| contributes R̄ d/R.
fn rsscs_backward(
    positions: &[[f64; 3]],
    p: &MbdAtomParams,
    cfg: &MbdRsscsConfig,
    res: MbdRsscsResult,
    tape: &Tape,
) -> MbdRsscsGradient {
    let n = positions.len();
    let n3 = 3 * n;
    let (beta, fa) = (cfg.beta, cfg.a);
    let mut d_pos = vec![[0.0_f64; 3]; n];
    let mut d_a0 = vec![0.0_f64; n];
    let mut d_c6 = vec![0.0_f64; n];
    let mut d_r = vec![0.0_f64; n];

    // Step 1: G = ¼ V diag(λ^{-1/2}) Vᵀ.
    let mut vs = tape.h_evecs.clone();
    for (pidx, mut col) in vs.axis_iter_mut(Axis(1)).enumerate() {
        let s = 0.25 / tape.h_evals[pidx].sqrt();
        col.mapv_inplace(|x| x * s);
    }
    let g = vs.dot(&tape.h_evecs.t());
    drop(vs);

    // Step 2: adjoint through H.
    let (a0t, rt, omt) = (&res.alpha_0_rsscs, &res.r_vdw_rsscs, &res.omega_rsscs);
    let mut om_bar = vec![-1.5_f64; n];
    let mut a0t_bar = vec![0.0_f64; n];
    let mut rt_bar = vec![0.0_f64; n];
    for a in 0..n {
        let diag: f64 = (0..3).map(|i| g[(3 * a + i, 3 * a + i)]).sum();
        om_bar[a] += 2.0 * omt[a] * diag;
        for b in 0..a {
            let d = sep(positions, a, b);
            let r = norm(&d);
            let s = beta * (rt[a] + rt[b]);
            let f = fermi(r, s, fa);
            let sq = (a0t[a] * a0t[b]).sqrt();
            let pref = omt[a] * omt[b] * sq * f;
            let tb = t_bare(&d, r);
            let mut vbar = [[0.0_f64; 3]; 3];
            let mut pbar = 0.0;
            for i in 0..3 {
                for j in 0..3 {
                    vbar[i][j] = g[(3 * a + i, 3 * b + j)] + g[(3 * b + j, 3 * a + i)];
                    pbar += vbar[i][j] * tb[i][j];
                }
            }
            let m: [[f64; 3]; 3] =
                std::array::from_fn(|i| std::array::from_fn(|j| pref * vbar[i][j]));
            let (_, gd, _, _) = t_bare_vjp(&d, r, &m);
            om_bar[a] += pbar * omt[b] * sq * f;
            om_bar[b] += pbar * omt[a] * sq * f;
            let half = 0.5 * pbar * omt[a] * omt[b] * sq * f;
            a0t_bar[a] += half / a0t[a];
            a0t_bar[b] += half / a0t[b];
            let fbar = pbar * omt[a] * omt[b] * sq;
            let (df_dr, df_ds) = fermi_partials(r, s, fa);
            let rbar = fbar * df_dr;
            let sbar = fbar * df_ds;
            rt_bar[a] += beta * sbar;
            rt_bar[b] += beta * sbar;
            for x in 0..3 {
                let gx = gd[x] + rbar * d[x] / r;
                d_pos[a][x] += gx;
                d_pos[b][x] -= gx;
            }
        }
    }
    drop(g);

    // Step 3: ω̃ and R̃.
    let mut c6t_bar = vec![0.0_f64; n];
    for a in 0..n {
        c6t_bar[a] = om_bar[a] * 4.0 / (3.0 * a0t[a] * a0t[a]);
        a0t_bar[a] -= om_bar[a] * 2.0 * omt[a] / a0t[a];
        a0t_bar[a] += rt_bar[a] * rt[a] / (3.0 * a0t[a]);
        d_r[a] += rt_bar[a] * rt[a] / p.r_vdw[a];
        d_a0[a] -= rt_bar[a] * rt[a] / (3.0 * p.alpha_0[a]);
    }

    // Step 4: per frequency node (buffers reused across nodes).
    let omega_ts: Vec<f64> = (0..n)
        .map(|a| 4.0 / 3.0 * p.c6[a] / (p.alpha_0[a] * p.alpha_0[a]))
        .collect();
    let frac = (2.0 / std::f64::consts::PI).sqrt() / 3.0;
    let mut om_ts_bar = vec![0.0_f64; n];
    let mut abar = vec![0.0_f64; n];
    let mut alpha = vec![0.0_f64; n];
    let mut sigma = vec![0.0_f64; n];
    let mut alpha_bar = vec![0.0_f64; n];
    let mut sigma_bar = vec![0.0_f64; n];
    let mut z3 = Array2::<f64>::zeros((n3, 3));
    let mut w3 = Array2::<f64>::zeros((n3, 3));
    for (k, &u) in tape.freqs.iter().enumerate() {
        let bm = &tape.binv[k];
        let ak = &tape.a_dyn[k];
        for a in 0..n {
            abar[a] = c6t_bar[a] * (6.0 / std::f64::consts::PI) * tape.weights[k] * ak[a];
            if k == 0 {
                // α̃₀ = a_dyn[0] (the u = 0 node), as in `rsscs_forward`.
                abar[a] += a0t_bar[a];
            }
        }
        z3.fill(0.0);
        w3.fill(0.0);
        for m in 0..n3 {
            for bb in 0..n {
                for i in 0..3 {
                    w3[(m, i)] += bm[(m, 3 * bb + i)];
                    z3[(m, i)] += bm[(3 * bb + i, m)] * abar[bb] / 3.0;
                }
            }
        }
        let cbar = |m: usize, q: usize| -> f64 {
            -(z3[(m, 0)] * w3[(q, 0)] + z3[(m, 1)] * w3[(q, 1)] + z3[(m, 2)] * w3[(q, 2)])
        };
        for a in 0..n {
            alpha[a] = p.alpha_0[a] / (1.0 + (u / omega_ts[a]).powi(2));
            sigma[a] = (frac * alpha[a]).cbrt();
        }
        alpha_bar.fill(0.0);
        sigma_bar.fill(0.0);
        for a in 0..n {
            let dsum: f64 = (0..3).map(|i| cbar(3 * a + i, 3 * a + i)).sum();
            alpha_bar[a] -= dsum / (alpha[a] * alpha[a]);
            for b in 0..a {
                let d = sep(positions, a, b);
                let r = norm(&d);
                let s = beta * (p.r_vdw[a] + p.r_vdw[b]);
                let sr = 1.0 - fermi(r, s, fa);
                let sab = (sigma[a] * sigma[a] + sigma[b] * sigma[b]).sqrt();
                let t = t_gg(&d, r, sab);
                let mut vbar = [[0.0_f64; 3]; 3];
                let mut srbar = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        vbar[i][j] = cbar(3 * a + i, 3 * b + j) + cbar(3 * b + j, 3 * a + i);
                        srbar += vbar[i][j] * t[i][j];
                    }
                }
                let m: [[f64; 3]; 3] =
                    std::array::from_fn(|i| std::array::from_fn(|j| sr * vbar[i][j]));
                let (gd, gsig) = t_gg_vjp(&d, r, sab, &m);
                let fbar = -srbar;
                let (df_dr, df_ds) = fermi_partials(r, s, fa);
                let rbar = fbar * df_dr;
                let sbar = fbar * df_ds;
                d_r[a] += beta * sbar;
                d_r[b] += beta * sbar;
                for x in 0..3 {
                    let gx = gd[x] + rbar * d[x] / r;
                    d_pos[a][x] += gx;
                    d_pos[b][x] -= gx;
                }
                sigma_bar[a] += gsig * sigma[a] / sab;
                sigma_bar[b] += gsig * sigma[b] / sab;
            }
        }
        for a in 0..n {
            alpha_bar[a] += sigma_bar[a] * sigma[a] / (3.0 * alpha[a]);
            d_a0[a] += alpha_bar[a] * alpha[a] / p.alpha_0[a];
            om_ts_bar[a] += alpha_bar[a] * 2.0 * alpha[a] * alpha[a] * u * u
                / (p.alpha_0[a] * omega_ts[a].powi(3));
        }
    }

    // Step 5: ω = 4/3 C6/α₀².
    for a in 0..n {
        d_c6[a] += om_ts_bar[a] * 4.0 / (3.0 * p.alpha_0[a] * p.alpha_0[a]);
        d_a0[a] -= om_ts_bar[a] * 2.0 * omega_ts[a] / p.alpha_0[a];
    }

    MbdRsscsGradient {
        result: res,
        d_positions: d_pos,
        d_alpha_0: d_a0,
        d_c6,
        d_r_vdw: d_r,
    }
}

/// Analytic gradient of the MBD@rsSCS energy with respect to the coordinates
/// (Bohr) and the unscreened per-atom TS inputs, by one reverse-mode pass
/// over the same forward computation as [`mbd_rsscs_energy_from_params`]
/// (cost O(n_freq (3N)³), dominated by the forward inverses).
///
/// Memory: the forward pass keeps the n_freq + 1 inverse screening matrices,
/// (n_freq + 1)·(3N)²·8 bytes.
///
/// # Errors
///
/// As [`mbd_rsscs_energy_from_params`].
pub fn mbd_rsscs_gradient_from_params(
    positions: &[[f64; 3]],
    params: &MbdAtomParams,
    config: &MbdRsscsConfig,
) -> Result<MbdRsscsGradient, FerricError> {
    // The tape keeps one dense (3N)² screening inverse per frequency node
    // (n_freq + 1 of them) plus H's eigenvectors, and the forward/backward
    // passes hold about two more (3N)² work matrices: charged against the
    // memory budget before anything is allocated, held for the whole call.
    let dim = 3 * positions.len();
    let est = (config.n_freq + 1 + 3)
        .saturating_mul(dim.saturating_mul(dim))
        .saturating_mul(std::mem::size_of::<f64>());
    let label = format!(
        "mbd_rsscs_gradient (natoms={}, n_freq={})",
        positions.len(),
        config.n_freq
    );
    ferric_core::memory::check_alloc(&label, est, ferric_core::memory::resolve_budget_bytes(None))?;
    let _charge = ferric_core::memory::pool::reserve_global(&label, est)?;
    let (res, tape) = rsscs_forward(positions, params, config, true)?;
    let tape = tape.ok_or_else(|| {
        FerricError::General("MBD@rsSCS gradient: forward pass returned no tape".to_string())
    })?;
    Ok(rsscs_backward(positions, params, config, res, &tape))
}

/// [`mbd_rsscs_gradient_from_params`] from atomic numbers and Hirshfeld
/// volume ratios (argument order of [`mbd_rsscs_energy`]). Also returns
/// dE/dr_A through α₀ = r α_free, C6 = r² C6_free, R_vdW = r^{1/3} R_free:
/// dE/dr = α_free dE/dα₀ + 2 r C6_free dE/dC6 + ⅓ r^{−2/3} R_free dE/dR_vdW.
///
/// # Errors
///
/// As [`mbd_rsscs_energy`].
pub fn mbd_rsscs_gradient(
    z: &[usize],
    positions: &[[f64; 3]],
    volume_ratios: &[f64],
    config: &MbdRsscsConfig,
) -> Result<(MbdRsscsGradient, Vec<f64>), FerricError> {
    let params = ts_params_from_volume_ratios(z, volume_ratios)?;
    let grad = mbd_rsscs_gradient_from_params(positions, &params, config)?;
    let mut d_ratio = Vec::with_capacity(z.len());
    for (i, (&za, &r)) in z.iter().zip(volume_ratios).enumerate() {
        let (alpha_free, c6_free, _) = ts_free_atom(za).ok_or_else(|| out_of_table(i, za))?;
        let r_free = ts_free_atom_r_vdw(za).ok_or_else(|| out_of_table(i, za))?;
        let rc = r.cbrt();
        d_ratio.push(
            alpha_free * grad.d_alpha_0[i]
                + 2.0 * r * c6_free * grad.d_c6[i]
                + r_free / (3.0 * rc * rc) * grad.d_r_vdw[i],
        );
    }
    Ok((grad, d_ratio))
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

    // ---------------------------------------------------------------------
    // Analytic gradient (reverse mode) vs central finite differences.
    //
    // Expected FD error, central difference with step h:
    //   truncation h²/6 · E''' and roundoff ~ eps·Σω/h (E cancels
    //   ½Σ√λ against (3/2)Σω, Σω ~ 1–3 Ha here).
    // Positions, h = 1e-4 Bohr: truncation ~1e-11 (|E| ~ 1e-3 Ha, E''' ~
    // |E|·336/L³ at L ~ 2–3 Bohr), roundoff ~1e-15·3/2e-4 ~ 1e-11, so
    // ~1e-10 absolute against max|g| ~ 1e-4..1e-3, i.e. ≲1e-6 of the scale.
    // Parameters, h = 1e-5·x (x ~ 3–50): roundoff ~1e-14/(2h) ~ 1e-10..1e-9
    // against |dE/dx| ~ 1e-5..1e-4, i.e. ≲1e-5..1e-6 of the scale.
    //
    // WHAT EACH TEST CATCHES (mutations of `rsscs_backward`):
    // * gradient_matches_fd_*: positions — dropping the T_SR (screening)
    //   adjoint of step 4, dropping either Fermi R-derivative, a wrong
    //   ∂T_GG/∂d or ∂T_GG/∂σ term, using only the (A,B) block instead of
    //   (A,B)+(B,A) (factor 2 on every pair adjoint), C̄ = +B Ḃ B (sign),
    //   dropping the R̃ dependence on α̃₀ (α̃₀ depends on positions through
    //   screening). Parameters — dropping the ω(C6, α₀) dependence of α(iu)
    //   (C6 enters ONLY via ω, so dE/dC6 would be identically zero),
    //   dropping −R̄̃ R̃/(3α₀) or the β S̄ term on the UNSCREENED R_vdW, the
    //   wrong weight 6/π w_k in ā.
    // * ratio_gradient_matches_fd: chain-rule factors (2r C6_free,
    //   ⅓ r^{-2/3} R_free).
    // * gradient_is_translation_and_rotation_invariant: a position adjoint
    //   not built from the separation vector (e.g. M·d replaced by a single
    //   component, or the ± split between A and B lost).
    // * gradient_energy_is_bit_identical: the gradient's forward pass
    //   drifting from the energy's.

    /// PROVISIONAL: max |analytic − FD| / max |analytic| per input family
    /// (positions, α₀, C6, R_vdW). Expected ≲1e-6 (above); the main
    /// session tightens this from the measured maxima.
    const FD_REL_BAR: f64 = 1e-5;
    /// PROVISIONAL: |Σ_A g_A| and |Σ_A R_A × g_A| relative to max|g| (and
    /// max|R|·max|g| for the torque). Pure roundoff is expected (~1e-14).
    const INVARIANCE_REL_BAR: f64 = 1e-10;
    const FD_H_POS: f64 = 1e-4;
    const FD_H_PARAM_REL: f64 = 1e-5;

    fn dimer_case() -> (Vec<usize>, Vec<[f64; 3]>, Vec<f64>) {
        (
            vec![6, 8],
            vec![[0.0, 0.0, 0.0], [0.3, -0.4, 2.6]],
            vec![0.9, 0.8],
        )
    }

    /// Non-symmetric 6-atom cluster (formamide-like, bonded distances).
    fn cluster_case() -> (Vec<usize>, Vec<[f64; 3]>, Vec<f64>) {
        (
            vec![6, 8, 7, 1, 1, 1],
            vec![
                [0.0, 0.0, 0.0],
                [2.27, 0.1, 0.05],
                [-1.3, 2.1, -0.2],
                [-0.9, -1.95, 0.3],
                [-3.1, 2.0, 0.4],
                [-0.4, 3.8, -0.6],
            ],
            vec![0.9, 0.85, 0.88, 0.6, 0.65, 0.7],
        )
    }

    fn max_abs(v: impl IntoIterator<Item = f64>) -> f64 {
        v.into_iter().fold(0.0_f64, |m, x| m.max(x.abs()))
    }

    fn energy(pos: &[[f64; 3]], p: &MbdAtomParams, c: &MbdRsscsConfig) -> f64 {
        mbd_rsscs_energy_from_params(pos, p, c).unwrap().energy
    }

    fn check_family(label: &str, analytic: &[f64], fd: &[f64]) {
        let scale = max_abs(analytic.iter().copied());
        assert!(
            scale > 0.0,
            "{label}: analytic gradient is identically zero"
        );
        let err = max_abs(analytic.iter().zip(fd).map(|(a, f)| a - f));
        eprintln!(
            "{label}: max|an-fd| {err:.2e} scale {scale:.2e} rel {:.2e}",
            err / scale
        );
        assert!(
            err < FD_REL_BAR * scale,
            "{label}: max|analytic - FD| {err:.2e} >= {FD_REL_BAR:.0e} * {scale:.2e}\n\
             analytic {analytic:?}\nfd {fd:?}"
        );
    }

    fn fd_check_all(label: &str, pos: &[[f64; 3]], p: &MbdAtomParams, c: &MbdRsscsConfig) {
        let g = mbd_rsscs_gradient_from_params(pos, p, c).unwrap();
        let n = pos.len();
        // Positions.
        let mut an = Vec::new();
        let mut fd = Vec::new();
        for a in 0..n {
            for x in 0..3 {
                let mut pp = pos.to_vec();
                let mut pm = pos.to_vec();
                pp[a][x] += FD_H_POS;
                pm[a][x] -= FD_H_POS;
                fd.push((energy(&pp, p, c) - energy(&pm, p, c)) / (2.0 * FD_H_POS));
                an.push(g.d_positions[a][x]);
            }
        }
        check_family(&format!("{label} positions"), &an, &fd);
        // Unscreened TS inputs.
        fn f_alpha(q: &mut MbdAtomParams) -> &mut Vec<f64> {
            &mut q.alpha_0
        }
        fn f_c6(q: &mut MbdAtomParams) -> &mut Vec<f64> {
            &mut q.c6
        }
        fn f_r(q: &mut MbdAtomParams) -> &mut Vec<f64> {
            &mut q.r_vdw
        }
        type Field = fn(&mut MbdAtomParams) -> &mut Vec<f64>;
        let fields: [(&str, Field, &Vec<f64>); 3] = [
            ("alpha_0", f_alpha, &g.d_alpha_0),
            ("C6", f_c6, &g.d_c6),
            ("R_vdW", f_r, &g.d_r_vdw),
        ];
        for (name, field, analytic) in fields {
            let mut fd = Vec::with_capacity(n);
            for a in 0..n {
                let mut qp = p.clone();
                let mut qm = p.clone();
                let h = FD_H_PARAM_REL * field(&mut qp)[a];
                field(&mut qp)[a] += h;
                field(&mut qm)[a] -= h;
                fd.push((energy(pos, &qp, c) - energy(pos, &qm, c)) / (2.0 * h));
            }
            check_family(&format!("{label} {name}"), analytic, &fd);
        }
    }

    #[test]
    fn gradient_matches_fd_dimer() {
        let (z, pos, r) = dimer_case();
        let p = ts_params_from_volume_ratios(&z, &r).unwrap();
        fd_check_all("dimer", &pos, &p, &cfg(0.83));
    }

    #[test]
    fn gradient_matches_fd_cluster() {
        let (z, pos, r) = cluster_case();
        let p = ts_params_from_volume_ratios(&z, &r).unwrap();
        fd_check_all("cluster", &pos, &p, &cfg(0.83));
    }

    #[test]
    fn ratio_gradient_matches_fd() {
        let c = cfg(0.83);
        for (label, (z, pos, r)) in [("dimer", dimer_case()), ("cluster", cluster_case())] {
            let (g, d_ratio) = mbd_rsscs_gradient(&z, &pos, &r, &c).unwrap();
            let e_ref = mbd_rsscs_energy(&z, &pos, &r, &c).unwrap().energy;
            assert_eq!(g.result.energy, e_ref, "{label}: ratio front door energy");
            let fd: Vec<f64> = (0..r.len())
                .map(|a| {
                    let h = FD_H_PARAM_REL * r[a];
                    let mut rp = r.clone();
                    let mut rm = r.clone();
                    rp[a] += h;
                    rm[a] -= h;
                    (mbd_rsscs_energy(&z, &pos, &rp, &c).unwrap().energy
                        - mbd_rsscs_energy(&z, &pos, &rm, &c).unwrap().energy)
                        / (2.0 * h)
                })
                .collect();
            check_family(&format!("{label} ratios"), &d_ratio, &fd);
        }
    }

    #[test]
    fn gradient_is_translation_and_rotation_invariant() {
        let (z, pos, r) = cluster_case();
        let (g, _) = mbd_rsscs_gradient(&z, &pos, &r, &cfg(0.83)).unwrap();
        let gmax = max_abs(g.d_positions.iter().flatten().copied());
        let rmax = max_abs(pos.iter().flatten().copied());
        assert!(gmax > 0.0);
        let mut force_sum = [0.0_f64; 3];
        let mut torque = [0.0_f64; 3];
        for (ra, ga) in pos.iter().zip(&g.d_positions) {
            for x in 0..3 {
                force_sum[x] += ga[x];
            }
            torque[0] += ra[1] * ga[2] - ra[2] * ga[1];
            torque[1] += ra[2] * ga[0] - ra[0] * ga[2];
            torque[2] += ra[0] * ga[1] - ra[1] * ga[0];
        }
        let ft = max_abs(force_sum);
        let tq = max_abs(torque);
        eprintln!("sum g {ft:.2e}, torque {tq:.2e}, max|g| {gmax:.2e}");
        assert!(ft < INVARIANCE_REL_BAR * gmax, "sum_A g_A = {force_sum:?}");
        assert!(
            tq < INVARIANCE_REL_BAR * gmax * rmax,
            "sum_A R_A x g_A = {torque:?}"
        );
    }

    #[test]
    fn gradient_energy_is_bit_identical() {
        for (z, pos, r) in [dimer_case(), cluster_case()] {
            let p = ts_params_from_volume_ratios(&z, &r).unwrap();
            let c = cfg(0.83);
            let g = mbd_rsscs_gradient_from_params(&pos, &p, &c).unwrap();
            let e = mbd_rsscs_energy_from_params(&pos, &p, &c).unwrap();
            assert_eq!(g.result.energy, e.energy);
            assert_eq!(g.result.alpha_0_rsscs, e.alpha_0_rsscs);
            assert_eq!(g.result.min_eigenvalue, e.min_eigenvalue);
        }
    }

    /// EXACTNESS ANCHOR for the gradient: a single atom has E ≡ 0 for every
    /// position and TS input, so every derivative is exactly zero.
    #[test]
    fn single_atom_gradient_is_zero() {
        let (g, d_ratio) =
            mbd_rsscs_gradient(&[6], &[[0.3, -0.2, 1.0]], &[0.87], &cfg(0.83)).unwrap();
        let all = g
            .d_positions
            .iter()
            .flatten()
            .chain(&g.d_alpha_0)
            .chain(&g.d_c6)
            .chain(&g.d_r_vdw)
            .chain(&d_ratio)
            .copied();
        // E = ½·3√(ω̃²) − (3/2)ω̃ cancels exactly in the adjoint too, up to the
        // rounding of 1/√λ against ω̃ (a few ulp of ω̃·(dω̃/dx)).
        let m = max_abs(all);
        assert!(m < 1e-12, "single atom gradient {m:.2e}");
    }
}
