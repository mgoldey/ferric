//! Harmonic vibrational frequencies from numerical differentiation of
//! **analytic** nuclear gradients.
//!
//! ferric has analytic gradients but no analytic second derivatives, so the
//! Hessian here is built by central-differencing the analytic gradient with
//! respect to each of the 3N nuclear coordinates:
//!
//! ```text
//! H[a, b] = ( g_a(x_b + delta) - g_a(x_b - delta) ) / (2 * delta)
//! ```
//!
//! That is `6N` gradient evaluations (each a full SCF plus a gradient), the
//! same displacement pattern `ferric_mp2::gradient::rimp2_gradient_fd` uses
//! one derivative order lower. Differentiating an analytic gradient once is
//! numerically far better conditioned than differencing energies twice: the
//! error is `O(delta^2)` truncation plus `O(eps_scf / delta)` noise, rather
//! than `O(eps_scf / delta^2)`.
//!
//! **Analytic gradients only.** There is deliberately no finite-difference-of-
//! finite-difference fallback. A method without an analytic gradient produces a
//! clean typed error naming the method rather than a silently noisy number.
//!
//! All three references have an analytic KS gradient, verified against
//! [`crate::ks_gradient`] rather than assumed:
//!
//! | reference | HF gradient | KS gradient (`xc` set) |
//! |---|---|---|
//! | [`FrequencyReference::Rhf`]  | `rhf_gradient`  | `ks_gradient_closed` |
//! | [`FrequencyReference::Uhf`]  | `uhf_gradient`  | `ks_gradient_uks` |
//! | [`FrequencyReference::Rohf`] | `rohf_gradient` | `ks_gradient_roks` |
//!
//! meta-GGA is implemented for **all three** (τ-dependent terms in
//! `xc_gradient_closed_mgga_from_density` and its polarized sibling), so
//! RKS/UKS/ROKS SCAN frequencies work. Older notes in this repo describe
//! meta-GGA gradients — and open-shell KS gradients generally — as
//! unimplemented; that is stale. Each `ks_gradient_*` raises its own named
//! error for a combination it genuinely cannot handle, and frequencies inherit
//! it rather than falling back to differencing energies.
//!
//! # Pipeline
//!
//! 1. Central-difference the analytic gradient into `H[3N, 3N]` (Cartesian,
//!    Hartree / Bohr^2).
//! 2. Symmetrize `H <- (H + H^T)/2`. The two mixed partials are equal in exact
//!    arithmetic; their difference is a useful noise diagnostic, reported as
//!    [`FrequencyResult::asymmetry`].
//! 3. Mass-weight `H_ij <- H_ij / sqrt(m_i m_j)`, with masses in electron
//!    masses so eigenvalues come out in atomic units.
//! 4. **Project out translations and rotations** (6 modes, or 5 if linear)
//!    before diagonalizing. See [`projection`](self#projection) below.
//! 5. Diagonalize with [`ferric_core::linalg::eigh_dc`] and convert
//!    eigenvalues to wavenumbers.
//!
//! # Projection
//!
//! Skipping step 4 is the classic silent failure: the six zero-frequency modes
//! come out as small nonzero numbers of arbitrary sign, and — worse — they mix
//! into the genuine vibrations when the geometry is not exactly at a
//! stationary point. We build an orthonormal basis for the translation/
//! rotation subspace in *mass-weighted* coordinates and apply
//! `P = I - sum_k |t_k><t_k|` on both sides of the Hessian, which drives those
//! modes to numerically exact zero eigenvalues and leaves the vibrational
//! block untouched.
//!
//! The translation vectors are `sqrt(m_A)` times a unit Cartesian direction on
//! every atom; the rotation vectors are `sqrt(m_A) * (e_k x (R_A - R_com))`.
//! For a linear molecule the rotation about the molecular axis vanishes
//! identically, so Gram-Schmidt finds only 2 independent rotations and the
//! subspace has rank 5 — [`crate::frequencies::detect_linear`] and the rank found by
//! orthonormalization are cross-checked against each other.
//!
//! # Mass convention
//!
//! Masses are the isotope-averaged IUPAC 2013 standard atomic weights from
//! [`ferric_core::elements::atomic_mass`]. NOTE that PySCF's
//! `Mole.atom_mass_list()` returns *integer isotope mass numbers* by default
//! (O -> 16, not 15.999); its `hessian.thermo.harmonic_analysis` internally
//! uses averaged masses. Comparing against PySCF requires matching this
//! choice — the discrepancy is ~0.4% in frequency, which is far larger than
//! the numerical error of this module and would otherwise look like a bug.

use ferric_core::elements::atomic_mass;
use ferric_core::linalg::{eigh_dc, Uplo};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::{Array1, Array2};

use crate::gradient::{rhf_gradient, rohf_gradient, uhf_gradient};
use crate::ks_gradient::{ks_gradient_closed, ks_gradient_roks, ks_gradient_uks};
use crate::rhf::{solve_rhf, RhfConfig};
use crate::rohf::solve_rohf;
use crate::screening::SchwarzBounds;
use crate::uhf::solve_uhf;

/// Electron masses per unified atomic mass unit (u).
///
/// CODATA 2018: the electron mass is 5.485 799 090 65(16) x 10^-4 u, so one u
/// is the reciprocal of that. Masses must be converted to electron masses
/// (a.u.) before mass-weighting so the Hessian eigenvalues come out in atomic
/// units of angular-frequency-squared.
const AMU_TO_ELECTRON_MASS: f64 = 1.0 / 5.485_799_090_65e-4;

/// Hartree-atomic-unit angular frequency expressed in wavenumbers (cm^-1).
///
/// omega_au -> nu_tilde = omega_au * E_h / (2 pi c hbar), evaluated with CODATA
/// 2018 constants: E_h = 4.359 744 722 21e-18 J, hbar = 1.054 571 817e-34 J s,
/// c = 2.997 924 58e10 cm/s. This equals PySCF's
/// `nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI` route to within a
/// rounding of the last digit.
const AU_FREQ_TO_CM: f64 = 219_474.631_363_2;

/// Default displacement for the central difference, in Bohr.
///
/// 5e-3 Bohr balances `O(delta^2)` truncation against gradient noise. Tighter
/// displacements amplify SCF convergence error; looser ones bite into the
/// harmonic approximation of the numerical derivative itself.
pub const DEFAULT_DELTA: f64 = 5.0e-3;

/// Threshold (in Bohr) on the smallest moment of inertia below which a
/// molecule is treated as linear.
const LINEARITY_THRESH: f64 = 1.0e-6;

/// Which SCF reference to differentiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyReference {
    /// Closed-shell RHF, or closed-shell KS-DFT when `RhfConfig::xc` is set.
    Rhf,
    /// Spin-unrestricted UHF, or UKS when `RhfConfig::xc` is set.
    Uhf,
    /// Restricted open-shell ROHF, or ROKS when `RhfConfig::xc` is set.
    Rohf,
}

impl FrequencyReference {
    #[allow(dead_code)] // diagnostic name used by callers-to-be; harmless to keep
    fn label(&self) -> &'static str {
        match self {
            FrequencyReference::Rhf => "RHF/RKS",
            FrequencyReference::Uhf => "UHF",
            FrequencyReference::Rohf => "ROHF",
        }
    }
}

/// Configuration for a frequency calculation.
#[derive(Debug, Clone)]
pub struct FrequencyConfig {
    /// Central-difference displacement in Bohr. See [`DEFAULT_DELTA`].
    pub delta: f64,
    /// Which SCF reference to use.
    pub reference: FrequencyReference,
}

impl Default for FrequencyConfig {
    fn default() -> Self {
        Self {
            delta: DEFAULT_DELTA,
            reference: FrequencyReference::Rhf,
        }
    }
}

/// Result of a harmonic frequency calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct FrequencyResult {
    /// Vibrational wavenumbers in cm^-1, ascending. Length `3N - 6`
    /// (`3N - 5` if linear). A negative entry denotes an **imaginary**
    /// frequency (a mode with a negative force constant), reported as a
    /// negative number by the usual convention.
    pub frequencies: Vec<f64>,
    /// Wavenumbers of the projected-out translation/rotation modes, in cm^-1.
    /// These should be numerically zero; they are retained as a diagnostic.
    pub trans_rot_frequencies: Vec<f64>,
    /// Normal-mode displacement vectors in *Cartesian* coordinates, one row
    /// per mode, `3N` entries each (mass-weighted eigenvectors divided back
    /// through by `sqrt(m)`). Not normalized to any particular convention.
    pub normal_modes: Array2<f64>,
    /// Whether the molecule was detected as linear (so `3N - 5` modes).
    pub is_linear: bool,
    /// The symmetrized mass-weighted Hessian actually diagonalized, in atomic
    /// units.
    pub mass_weighted_hessian: Array2<f64>,
    /// Cartesian Hessian before mass-weighting, Hartree/Bohr^2, symmetrized.
    pub cartesian_hessian: Array2<f64>,
    /// Largest `|H_ij - H_ji|` in the raw Cartesian Hessian before
    /// symmetrization, Hartree/Bohr^2. This is a direct measure of the
    /// numerical noise floor: it is zero in exact arithmetic, so a large value
    /// means the displacement or the SCF convergence is badly chosen.
    pub asymmetry: f64,
    /// Number of analytic gradient evaluations performed (`6N`).
    pub n_gradient_evaluations: usize,
    /// Electronic energy at the *undisplaced* input geometry.
    pub energy: f64,
}

impl FrequencyResult {
    /// Zero-point vibrational energy in Hartree, `0.5 * sum(h nu)`.
    ///
    /// Imaginary modes (negative entries) are skipped — a ZPE is only
    /// physically meaningful at a true minimum, and including an imaginary
    /// mode would silently produce a complex or wrong number.
    pub fn zero_point_energy(&self) -> f64 {
        0.5 * self
            .frequencies
            .iter()
            .filter(|f| **f > 0.0)
            .map(|f| f / AU_FREQ_TO_CM)
            .sum::<f64>()
    }

    /// Number of imaginary (negative) frequencies. Zero at a minimum, one at a
    /// first-order saddle point.
    pub fn n_imaginary(&self) -> usize {
        self.frequencies.iter().filter(|f| **f < 0.0).count()
    }
}

/// Compute harmonic vibrational frequencies by central-differencing the
/// analytic nuclear gradient.
///
/// Performs `6N` SCF-plus-gradient evaluations. The input geometry should be a
/// converged stationary point; nothing here verifies that, and frequencies at
/// a non-stationary geometry are not physically meaningful (the residual
/// gradient contaminates the projection).
///
/// Returns a typed error if the requested method has no analytic gradient
/// (meta-GGA), naming the method.
pub fn harmonic_frequencies(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    scf_config: &RhfConfig,
    freq_config: &FrequencyConfig,
) -> Result<FrequencyResult, FerricError> {
    let natoms = mol.atoms.len();
    if natoms < 2 {
        return Err(FerricError::General(format!(
            "harmonic frequencies require at least 2 atoms, got {natoms}"
        )));
    }
    if !(freq_config.delta.is_finite() && freq_config.delta > 0.0) {
        return Err(FerricError::General(format!(
            "frequency displacement delta must be finite and positive, got {}",
            freq_config.delta
        )));
    }

    // Fail fast on a missing mass before spending 6N SCF evaluations.
    let masses = atom_masses(mol)?;

    let n_coord = 3 * natoms;

    // Energy at the undisplaced geometry, and an early check that the method
    // combination actually produces a gradient at all.
    let (energy, _) = energy_and_gradient(ctx, mol, basis_name, op, scf_config, freq_config.reference)?;

    // --- Step 1: central-difference the analytic gradient -----------------
    // Column b of the Hessian is d(g)/d(x_b), so we displace coordinate b and
    // collect the whole 3N-vector gradient at each of the two displacements.
    let delta = freq_config.delta;
    let mut hess = Array2::<f64>::zeros((n_coord, n_coord));
    let mut n_evals = 0usize;

    for b in 0..n_coord {
        let atom = b / 3;
        let coord = b % 3;

        let mut mol_p = mol.clone();
        let mut mol_m = mol.clone();
        displace(&mut mol_p, atom, coord, delta);
        displace(&mut mol_m, atom, coord, -delta);

        let (_, g_p) = energy_and_gradient(ctx, &mol_p, basis_name, op, scf_config, freq_config.reference)?;
        let (_, g_m) = energy_and_gradient(ctx, &mol_m, basis_name, op, scf_config, freq_config.reference)?;
        n_evals += 2;

        for a in 0..n_coord {
            let (ai, ac) = (a / 3, a % 3);
            hess[(a, b)] = (g_p[(ai, ac)] - g_m[(ai, ac)]) / (2.0 * delta);
        }
    }

    // --- Step 2: symmetrize, recording the asymmetry as a noise probe -----
    let mut asymmetry: f64 = 0.0;
    for i in 0..n_coord {
        for j in (i + 1)..n_coord {
            asymmetry = asymmetry.max((hess[(i, j)] - hess[(j, i)]).abs());
        }
    }
    let cartesian_hessian = {
        let mut h = hess.clone();
        for i in 0..n_coord {
            for j in 0..n_coord {
                h[(i, j)] = 0.5 * (hess[(i, j)] + hess[(j, i)]);
            }
        }
        h
    };

    // --- Steps 3-5 --------------------------------------------------------
    let mut result = frequencies_from_cartesian_hessian(mol, &cartesian_hessian, &masses)?;
    result.asymmetry = asymmetry;
    result.n_gradient_evaluations = n_evals;
    result.energy = energy;
    Ok(result)
}

/// Mass-weight, project out translations/rotations, diagonalize, and convert to
/// wavenumbers.
///
/// Split out from [`harmonic_frequencies`] so the (cheap, purely linear-algebra)
/// half can be tested against an analytically-known Hessian without running any
/// SCF, and so a Hessian obtained some other way can be fed in.
pub fn frequencies_from_cartesian_hessian(
    mol: &Molecule,
    cartesian_hessian: &Array2<f64>,
    masses_amu: &[f64],
) -> Result<FrequencyResult, FerricError> {
    let natoms = mol.atoms.len();
    let n_coord = 3 * natoms;
    if cartesian_hessian.dim() != (n_coord, n_coord) {
        return Err(FerricError::General(format!(
            "cartesian Hessian must be {n_coord}x{n_coord} for {natoms} atoms, got {:?}",
            cartesian_hessian.dim()
        )));
    }
    if masses_amu.len() != natoms {
        return Err(FerricError::General(format!(
            "expected {natoms} masses, got {}",
            masses_amu.len()
        )));
    }

    // --- Step 3: mass-weight ---------------------------------------------
    // Masses in electron masses so eigenvalues land in atomic units.
    let mass_au: Vec<f64> = masses_amu.iter().map(|m| m * AMU_TO_ELECTRON_MASS).collect();
    for (i, m) in mass_au.iter().enumerate() {
        if !(m.is_finite() && *m > 0.0) {
            return Err(FerricError::General(format!(
                "atom {i} has non-positive mass {m}"
            )));
        }
    }
    // inv_sqrt_m[c] for Cartesian coordinate c (atom c/3).
    let inv_sqrt_m: Vec<f64> = (0..n_coord).map(|c| 1.0 / mass_au[c / 3].sqrt()).collect();

    let mut mw = Array2::<f64>::zeros((n_coord, n_coord));
    for i in 0..n_coord {
        for j in 0..n_coord {
            mw[(i, j)] = cartesian_hessian[(i, j)] * inv_sqrt_m[i] * inv_sqrt_m[j];
        }
    }

    // --- Step 4: project out translations and rotations -------------------
    let is_linear = detect_linear(mol, &mass_au);
    let tr_basis = translation_rotation_basis(mol, &mass_au)?;
    let n_tr = tr_basis.len();
    let expected_tr = if is_linear { 5 } else { 6 };
    if n_tr != expected_tr {
        return Err(FerricError::General(format!(
            "translation/rotation subspace has rank {n_tr}, expected {expected_tr} for a \
             {} molecule -- geometry may be degenerate (coincident atoms?)",
            if is_linear { "linear" } else { "nonlinear" }
        )));
    }

    // P H P with P = I - sum_k |t_k><t_k|. Applied as
    //   H' = H - T(T^T H) - (H T)T^T + T(T^T H T)T^T
    // but the direct form below is clearer at these sizes and O(n_tr * n^2).
    let mut proj = mw.clone();
    for t in &tr_basis {
        // H <- H - t (t^T H) - (H t) t^T + t (t^T H t) t^T
        let ht: Array1<f64> = proj.dot(t);
        let tht: f64 = t.dot(&ht);
        for i in 0..n_coord {
            for j in 0..n_coord {
                proj[(i, j)] -= t[i] * ht[j] + ht[i] * t[j] - t[i] * t[j] * tht;
            }
        }
    }
    // Re-symmetrize: the rank-1 sweeps above accumulate O(eps) asymmetry.
    for i in 0..n_coord {
        for j in (i + 1)..n_coord {
            let v = 0.5 * (proj[(i, j)] + proj[(j, i)]);
            proj[(i, j)] = v;
            proj[(j, i)] = v;
        }
    }

    // --- Step 5: diagonalize and convert ----------------------------------
    let (evals, evecs) = eigh_dc(&proj, Uplo::Lower)?;

    // The projected translation/rotation modes now have numerically zero
    // eigenvalues. Identify them by overlap with the known T/R subspace rather
    // than by "the n_tr smallest |eigenvalue|" -- at a transition state a
    // genuine imaginary mode can be smaller in magnitude than the numerical
    // zeros, and magnitude-sorting would misclassify it as a rotation.
    let mut is_tr = vec![false; n_coord];
    for k in 0..n_coord {
        let col = evecs.column(k);
        let overlap: f64 = tr_basis
            .iter()
            .map(|t| {
                let d: f64 = t.iter().zip(col.iter()).map(|(a, b)| a * b).sum();
                d * d
            })
            .sum();
        // A vector lying wholly in the T/R subspace has overlap 1; a genuine
        // vibration is orthogonal to it (overlap ~0). 0.5 is the natural split.
        is_tr[k] = overlap > 0.5;
    }
    let found_tr = is_tr.iter().filter(|b| **b).count();
    if found_tr != n_tr {
        return Err(FerricError::General(format!(
            "projection produced {found_tr} translation/rotation eigenvectors but the \
             subspace has rank {n_tr} -- projection is inconsistent"
        )));
    }

    let to_cm = |lambda: f64| -> f64 {
        // omega = sqrt(lambda); a negative force constant gives an imaginary
        // frequency, reported as a negative wavenumber by convention.
        let w = lambda.abs().sqrt() * AU_FREQ_TO_CM;
        if lambda < 0.0 {
            -w
        } else {
            w
        }
    };

    let mut frequencies = Vec::with_capacity(n_coord - n_tr);
    let mut trans_rot_frequencies = Vec::with_capacity(n_tr);
    let mut modes: Vec<f64> = Vec::with_capacity((n_coord - n_tr) * n_coord);
    for k in 0..n_coord {
        if is_tr[k] {
            trans_rot_frequencies.push(to_cm(evals[k]));
        } else {
            frequencies.push(to_cm(evals[k]));
            // Back-transform the mass-weighted eigenvector to Cartesian
            // displacements: q = M^{1/2} x  =>  x = M^{-1/2} q.
            for i in 0..n_coord {
                modes.push(evecs[(i, k)] * inv_sqrt_m[i]);
            }
        }
    }

    let n_vib = frequencies.len();
    let normal_modes = Array2::from_shape_vec((n_vib, n_coord), modes).map_err(|e| {
        FerricError::General(format!("normal-mode array shape error: {e}"))
    })?;

    Ok(FrequencyResult {
        frequencies,
        trans_rot_frequencies,
        normal_modes,
        is_linear,
        mass_weighted_hessian: proj,
        cartesian_hessian: cartesian_hessian.clone(),
        asymmetry: 0.0,
        n_gradient_evaluations: 0,
        energy: 0.0,
    })
}

/// Isotope-averaged masses (u) for every atom, erroring on an unknown element.
pub fn atom_masses(mol: &Molecule) -> Result<Vec<f64>, FerricError> {
    mol.atoms
        .iter()
        .map(|a| {
            atomic_mass(a.z).ok_or_else(|| {
                FerricError::General(format!(
                    "no atomic mass available for {} (Z={}); the mass table covers Z=1..=54",
                    a.symbol, a.z
                ))
            })
        })
        .collect()
}

/// Center of mass, in Bohr.
fn center_of_mass(mol: &Molecule, mass_au: &[f64]) -> [f64; 3] {
    let total: f64 = mass_au.iter().sum();
    let mut com = [0.0; 3];
    for (a, m) in mol.atoms.iter().zip(mass_au.iter()) {
        com[0] += m * a.x;
        com[1] += m * a.y;
        com[2] += m * a.zpos;
    }
    for c in com.iter_mut() {
        *c /= total;
    }
    com
}

/// Detect linearity from the inertia tensor: a linear molecule has one zero
/// principal moment (the axis itself), a nonlinear one has three nonzero.
///
/// Uses a *relative* threshold so the test is scale-free — an absolute cutoff
/// would misclassify either very small or very extended molecules.
pub fn detect_linear(mol: &Molecule, mass_au: &[f64]) -> bool {
    if mol.atoms.len() < 3 {
        // Any 2-atom system is linear by construction.
        return true;
    }
    let com = center_of_mass(mol, mass_au);
    let mut inertia = Array2::<f64>::zeros((3, 3));
    for (a, m) in mol.atoms.iter().zip(mass_au.iter()) {
        let r = [a.x - com[0], a.y - com[1], a.zpos - com[2]];
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        for i in 0..3 {
            for j in 0..3 {
                inertia[(i, j)] += m * ((if i == j { r2 } else { 0.0 }) - r[i] * r[j]);
            }
        }
    }
    let Ok((evals, _)) = eigh_dc(&inertia, Uplo::Lower) else {
        return false;
    };
    let max = evals.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    if max <= 0.0 {
        return false;
    }
    evals[0].abs() / max < LINEARITY_THRESH
}

/// Orthonormal basis (in mass-weighted Cartesian coordinates) for the
/// translation + rotation subspace.
///
/// Translations: `t_k[3A + i] = sqrt(m_A) * delta_ik`.
/// Rotations:    `r_k[3A + .] = sqrt(m_A) * (e_k x (R_A - R_com))`.
///
/// The six raw vectors are orthonormalized by modified Gram-Schmidt; vectors
/// whose norm collapses are dropped, which is exactly what happens to the
/// third rotation of a linear molecule. The returned length is therefore 6 for
/// a nonlinear molecule and 5 for a linear one, determined by the numerics
/// rather than asserted up front.
fn translation_rotation_basis(
    mol: &Molecule,
    mass_au: &[f64],
) -> Result<Vec<Array1<f64>>, FerricError> {
    let natoms = mol.atoms.len();
    let n_coord = 3 * natoms;
    let com = center_of_mass(mol, mass_au);
    let sqrt_m: Vec<f64> = mass_au.iter().map(|m| m.sqrt()).collect();

    let mut raw: Vec<Array1<f64>> = Vec::with_capacity(6);

    // Three translations.
    for k in 0..3 {
        let mut v = Array1::<f64>::zeros(n_coord);
        for a in 0..natoms {
            v[3 * a + k] = sqrt_m[a];
        }
        raw.push(v);
    }

    // Three rotations: e_k x (R_A - R_com), mass-weighted.
    for k in 0..3 {
        let mut v = Array1::<f64>::zeros(n_coord);
        for (a, atom) in mol.atoms.iter().enumerate() {
            let r = [atom.x - com[0], atom.y - com[1], atom.zpos - com[2]];
            // e_k x r
            let cross = match k {
                0 => [0.0, -r[2], r[1]],
                1 => [r[2], 0.0, -r[0]],
                _ => [-r[1], r[0], 0.0],
            };
            for i in 0..3 {
                v[3 * a + i] = sqrt_m[a] * cross[i];
            }
        }
        raw.push(v);
    }

    // Modified Gram-Schmidt. The drop threshold is relative to the vector's
    // original norm so it is invariant to the overall mass/length scale.
    let mut basis: Vec<Array1<f64>> = Vec::with_capacity(6);
    for v in raw.iter() {
        let mut w = v.clone();
        let norm0 = w.dot(&w).sqrt();
        if norm0 <= 0.0 {
            continue;
        }
        for b in basis.iter() {
            let proj = b.dot(&w);
            w = &w - &(b * proj);
        }
        let norm = w.dot(&w).sqrt();
        // 1e-6 relative: the axial rotation of a linear molecule collapses to
        // ~1e-16 here, while a genuinely independent rotation retains O(1).
        if norm / norm0 > 1.0e-6 {
            basis.push(w / norm);
        }
    }
    Ok(basis)
}

fn displace(mol: &mut Molecule, atom: usize, coord: usize, d: f64) {
    match coord {
        0 => mol.atoms[atom].x += d,
        1 => mol.atoms[atom].y += d,
        _ => mol.atoms[atom].zpos += d,
    }
}

/// Reject method/reference combinations with no analytic gradient, naming the
/// method. Mirrors the guards in [`crate::optimize`].
///
/// All three references now have an analytic KS gradient
/// (`ks_gradient_closed` / `ks_gradient_uks` / `ks_gradient_roks`), so there is
/// nothing left to reject at this level: each of those functions raises its own
/// named error for a combination it cannot handle, and frequencies inherit it.
/// This hook is kept so a future gradient-less method has an obvious place to
/// declare itself rather than silently producing a noisy Hessian.
/// One SCF + analytic gradient at a geometry, dispatched on the reference.
fn energy_and_gradient(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    config: &RhfConfig,
    reference: FrequencyReference,
) -> Result<(f64, Array2<f64>), FerricError> {
    let bs = ferric_core::basis::bundled(basis_name)?;
    let prep = PreparedBasis::new(mol, &bs)?;
    let bounds = SchwarzBounds::compute(op, &prep)?;
    let ext = config.external_potential.as_ref();

    match reference {
        FrequencyReference::Rhf => {
            let res = solve_rhf(ctx, mol, &prep, op, &bounds, config)?;
            let grad = if let Some(xc_name) = config.xc.as_deref() {
                ks_gradient_closed(mol, &prep, &bs, op, &bounds, xc_name, &res, ext)?
            } else {
                rhf_gradient(mol, &prep, op, &bounds, &res, ext)?
            };
            Ok((res.energy, grad))
        }
        FrequencyReference::Uhf => {
            let res = solve_uhf(ctx, mol, &prep, &bounds, config)?;
            let grad = if let Some(xc_name) = config.xc.as_deref() {
                ks_gradient_uks(mol, &prep, &bs, op, &bounds, xc_name, &res, ext)?
            } else {
                uhf_gradient(mol, &prep, op, &bounds, &res, ext)?
            };
            Ok((res.energy, grad))
        }
        FrequencyReference::Rohf => {
            let res = solve_rohf(ctx, mol, &prep, op, &bounds, config)?;
            let grad = if let Some(xc_name) = config.xc.as_deref() {
                ks_gradient_roks(mol, &prep, &bs, op, &bounds, xc_name, &res, ext)?
            } else {
                rohf_gradient(mol, &prep, op, &bounds, &res, ext)?
            };
            Ok((res.energy, grad))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn water() -> Molecule {
        // Near-equilibrium RHF/cc-pVDZ water, the geometry the PySCF
        // reference below was generated at.
        Molecule::parse_xyz(
            "3\nwater\nO 0.000000 0.000000 0.117300\n\
             H 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n",
            0,
            1,
        )
        .unwrap()
    }

    fn co2() -> Molecule {
        Molecule::load_xyz("../../testdata/molecules/co2.xyz").unwrap()
    }

    /// Tolerance (cm^-1) for a projected-out translation/rotation mode built
    /// from a NOISELESS analytic Hessian.
    ///
    /// A frequency is `sqrt(eigenvalue)`, so the square root *amplifies*
    /// eigenvalue noise near zero: an eigenvalue at double-precision epsilon
    /// relative to the vibrational scale (~1e-4 a.u. for a 2000 cm^-1 mode)
    /// is ~1e-20, which maps to ~1e-4 cm^-1 rather than to zero. Asserting
    /// 1e-6 cm^-1 here would be asserting ~1e-24 on the eigenvalue, well below
    /// what f64 can represent at this scale. 1e-2 cm^-1 is still ~5 orders of
    /// magnitude below the softest chemically meaningful vibration.
    const ZERO_MODE_TOL_CM: f64 = 1.0e-2;

    /// Tolerance for zero modes coming out of a real SCF Hessian, where the
    /// noise floor is set by SCF convergence and the finite displacement
    /// rather than by machine epsilon.
    const ZERO_MODE_TOL_CM_SCF: f64 = 15.0;

    // ---- projection unit tests (no SCF) ---------------------------------

    #[test]
    fn test_detect_linear() {
        let m = water();
        let mass = atom_masses(&m)
            .unwrap()
            .iter()
            .map(|x| x * AMU_TO_ELECTRON_MASS)
            .collect::<Vec<_>>();
        assert!(!detect_linear(&m, &mass), "water is bent, not linear");

        let c = co2();
        let cmass = atom_masses(&c)
            .unwrap()
            .iter()
            .map(|x| x * AMU_TO_ELECTRON_MASS)
            .collect::<Vec<_>>();
        assert!(detect_linear(&c, &cmass), "CO2 is linear");
    }

    #[test]
    fn test_tr_basis_rank_6_for_bent_5_for_linear() {
        // This is THE test for the projection: a nonlinear molecule must have
        // a rank-6 translation/rotation subspace, a linear one rank 5.
        let m = water();
        let mass: Vec<f64> = atom_masses(&m).unwrap().iter().map(|x| x * AMU_TO_ELECTRON_MASS).collect();
        let b = translation_rotation_basis(&m, &mass).unwrap();
        assert_eq!(b.len(), 6, "bent water must have 6 trans/rot modes");

        let c = co2();
        let cmass: Vec<f64> = atom_masses(&c).unwrap().iter().map(|x| x * AMU_TO_ELECTRON_MASS).collect();
        let cb = translation_rotation_basis(&c, &cmass).unwrap();
        assert_eq!(cb.len(), 5, "linear CO2 must have 5 trans/rot modes");

        // And the basis must actually be orthonormal.
        for (i, u) in b.iter().enumerate() {
            for (j, v) in b.iter().enumerate() {
                let d = u.dot(v);
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((d - want).abs() < 1e-12, "<{i}|{j}> = {d}, want {want}");
            }
        }
    }

    /// A Hessian built from a pairwise harmonic (spring) model is exactly
    /// translation- and rotation-invariant at a stationary geometry, so its
    /// mass-weighted spectrum must contain exactly 6 (or 5) zeros. This tests
    /// the projection machinery against a known-clean Hessian, with no SCF
    /// noise in the way.
    fn spring_hessian(mol: &Molecule, k: f64) -> Array2<f64> {
        let n = mol.atoms.len();
        let mut h = Array2::<f64>::zeros((3 * n, 3 * n));
        let pos = |a: usize| {
            let at = &mol.atoms[a];
            [at.x, at.y, at.zpos]
        };
        for a in 0..n {
            for b in 0..n {
                if a == b {
                    continue;
                }
                let (ra, rb) = (pos(a), pos(b));
                let d: Vec<f64> = (0..3).map(|i| ra[i] - rb[i]).collect();
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                // Bond-stretch Hessian block: k * (unit outer unit).
                for i in 0..3 {
                    for j in 0..3 {
                        let v = k * (d[i] / r) * (d[j] / r);
                        h[(3 * a + i, 3 * a + j)] += v;
                        h[(3 * a + i, 3 * b + j)] -= v;
                    }
                }
            }
        }
        h
    }

    #[test]
    fn test_projection_gives_exactly_six_zeros_water() {
        let m = water();
        let masses = atom_masses(&m).unwrap();
        let h = spring_hessian(&m, 0.5);
        let r = frequencies_from_cartesian_hessian(&m, &h, &masses).unwrap();
        assert!(!r.is_linear);
        assert_eq!(r.trans_rot_frequencies.len(), 6, "water needs 6 zero modes");
        assert_eq!(r.frequencies.len(), 3, "water has 3N-6 = 3 vibrations");
        for f in &r.trans_rot_frequencies {
            assert!(f.abs() < ZERO_MODE_TOL_CM, "projected trans/rot mode not zero: {f} cm^-1");
        }
    }

    #[test]
    fn test_projection_gives_exactly_five_zeros_co2() {
        let m = co2();
        let masses = atom_masses(&m).unwrap();
        let h = spring_hessian(&m, 0.5);
        let r = frequencies_from_cartesian_hessian(&m, &h, &masses).unwrap();
        assert!(r.is_linear, "CO2 must be detected as linear");
        assert_eq!(r.trans_rot_frequencies.len(), 5, "linear CO2 needs 5 zero modes");
        assert_eq!(r.frequencies.len(), 4, "CO2 has 3N-5 = 4 vibrations");
        for f in &r.trans_rot_frequencies {
            assert!(f.abs() < ZERO_MODE_TOL_CM, "projected trans/rot mode not zero: {f} cm^-1");
        }
    }

    /// Translating and rigidly rotating the molecule (and its Hessian) must
    /// leave the frequencies unchanged. This is the strong test of the
    /// projection: a projection built in the wrong frame (e.g. about the
    /// origin instead of the center of mass) passes the "6 zeros" test at a
    /// centered geometry and fails here.
    #[test]
    fn test_frequencies_invariant_to_translation_and_rotation() {
        let m = water();
        let masses = atom_masses(&m).unwrap();
        let h = spring_hessian(&m, 0.5);
        let base = frequencies_from_cartesian_hessian(&m, &h, &masses).unwrap();

        // Translate far from the origin.
        let mut t = m.clone();
        for a in t.atoms.iter_mut() {
            a.x += 3.7;
            a.y -= 1.9;
            a.zpos += 12.25;
        }
        let ht = spring_hessian(&t, 0.5);
        let rt = frequencies_from_cartesian_hessian(&t, &ht, &masses).unwrap();
        for (a, b) in base.frequencies.iter().zip(rt.frequencies.iter()) {
            assert!((a - b).abs() < 1e-8, "translation changed a frequency: {a} vs {b}");
        }
        assert_eq!(rt.trans_rot_frequencies.len(), 6);

        // Rigidly rotate by a generic angle about a generic axis.
        let (ca, sa) = (0.6f64, 0.8f64); // exact 3-4-5 rotation about z
        let (cb, sb) = (0.8f64, 0.6f64); // then about x
        let mut r = m.clone();
        for at in r.atoms.iter_mut() {
            let (x, y, z) = (at.x, at.y, at.zpos);
            let (x1, y1, z1) = (ca * x - sa * y, sa * x + ca * y, z);
            let (x2, y2, z2) = (x1, cb * y1 - sb * z1, sb * y1 + cb * z1);
            at.x = x2;
            at.y = y2;
            at.zpos = z2;
        }
        let hr = spring_hessian(&r, 0.5);
        let rr = frequencies_from_cartesian_hessian(&r, &hr, &masses).unwrap();
        for (a, b) in base.frequencies.iter().zip(rr.frequencies.iter()) {
            assert!((a - b).abs() < 1e-8, "rotation changed a frequency: {a} vs {b}");
        }
        assert_eq!(rr.trans_rot_frequencies.len(), 6, "rotated water still needs 6 zeros");
        for f in &rr.trans_rot_frequencies {
            assert!(f.abs() < ZERO_MODE_TOL_CM, "rotated trans/rot mode not zero: {f}");
        }
    }

    #[test]
    fn test_mass_lookup_error_names_element() {
        // Z=92 (U) is outside the table; the error must name the element.
        let mut m = water();
        m.atoms[0].z = 92;
        m.atoms[0].symbol = "U".into();
        let e = atom_masses(&m).unwrap_err().to_string();
        assert!(e.contains("U"), "error should name the element: {e}");
        assert!(e.contains("92"), "error should name Z: {e}");
    }

    /// CLOSED-shell meta-GGA gradients ARE implemented (the tau terms landed in
    /// `xc_gradient_closed_mgga_from_density`), so a closed-shell SCAN
    /// frequency run must SUCCEED rather than error.
    ///
    /// This test exists because the obvious assumption -- "meta-GGA has no
    /// analytic gradient" -- is true only for the OPEN-shell paths. Asserting
    /// a rejection here would lock in a stale premise and would start failing
    /// the moment someone read the code.
    #[test]
    fn test_closed_shell_metagga_frequencies_are_supported() {
        let m = water();
        let cfg = RhfConfig {
            xc: Some("SCAN".into()),
            ..Default::default()
        };
        let fc = FrequencyConfig {
            reference: FrequencyReference::Rhf,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let r = harmonic_frequencies(&ctx, &m, "sto-3g", Operator::coulomb(), &cfg, &fc)
            .expect("closed-shell meta-GGA gradients are implemented, so this must succeed");
        assert_eq!(r.frequencies.len(), 3);
        assert_eq!(r.trans_rot_frequencies.len(), 6);
        eprintln!("water/STO-3G SCAN freqs (cm^-1): {:?}", r.frequencies);
    }

    /// OPEN-shell meta-GGA gradients are NOT implemented -- `ks_gradient_uks`
    /// / `ks_gradient_roks` still reject them -- and in any case ferric has no
    /// UKS/ROKS analytic gradient at all, so the open-shell guard fires first
    /// and names the reference.
    #[test]
    /// Open-shell meta-GGA frequencies now RUN (they were previously rejected).
    ///
    /// `ks_gradient_uks`/`_roks` handle MetaGga, so the old rejection is gone.
    /// But `ks_gradient.rs`'s module doc records a **pre-existing
    /// spin-polarized SCAN SCF energy defect** (~2e-4 Ha on OH/STO-3G, versus
    /// 3e-8 for OH/PBE and ~1e-8 for closed-shell SCAN) that limits the
    /// open-shell meta-GGA gradient to ~5e-4 Ha/Bohr against PySCF.
    ///
    /// So this asserts the frequencies are PRODUCED and physically sane, and
    /// deliberately does NOT assert tight accuracy — quoting an open-shell
    /// SCAN frequency as validated would be exactly the silent-wrong pattern
    /// the guard used to prevent. Tighten this once the SCF defect is fixed.
    fn test_open_shell_metagga_frequencies_run_but_are_not_tightly_validated() {
        let m = water();
        let cfg = RhfConfig {
            xc: Some("SCAN".into()),
            energy_conv: 1e-7,
            density_conv: 1e-6,
            max_iter: 600,
            ..Default::default()
        };
        let fc = FrequencyConfig {
            reference: FrequencyReference::Rohf,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let r = harmonic_frequencies(&ctx, &m, "sto-3g", Operator::coulomb(), &cfg, &fc)
            .expect("ROKS/SCAN frequencies should run now that MetaGga gradients are wired");
        assert_eq!(r.frequencies.len(), 3, "water has 3N-6 = 3 modes");
        for w in &r.frequencies {
            assert!(
                *w > 500.0 && *w < 6000.0,
                "SCAN water frequency {w} cm^-1 is outside any physical range"
            );
        }
    }

    #[test]
    /// UKS frequencies now RUN — this test previously asserted they were
    /// rejected, on a guard whose comment ("not implemented") was stale.
    fn test_uks_frequencies_run() {
        let m = water();
        let cfg = RhfConfig {
            xc: Some("PBE".into()),
            energy_conv: 1e-7,
            density_conv: 1e-6,
            max_iter: 600,
            ..Default::default()
        };
        let fc = FrequencyConfig {
            reference: FrequencyReference::Uhf,
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let r = harmonic_frequencies(&ctx, &m, "sto-3g", Operator::coulomb(), &cfg, &fc)
            .expect("UKS frequencies should run now that ks_gradient_uks is wired");
        assert_eq!(r.frequencies.len(), 3, "water has 3N-6 = 3 modes");
        assert!(
            r.asymmetry < 1e-3,
            "Hessian asymmetry {:.3e} above the FD noise floor",
            r.asymmetry
        );
    }

    // ---- end-to-end SCF tests -------------------------------------------

    /// Water / STO-3G RHF against PySCF `hessian.rhf` + `thermo`.
    ///
    /// PySCF reference (isotope-averaged masses, matching ferric's table):
    ///   1854.9 cm^-1 is NOT it -- see the literal values asserted below.
    #[test]
    fn test_water_sto3g_frequencies_vs_pyscf() {
        let m = water();
        let cfg = RhfConfig { energy_conv: 1e-11, ..Default::default() };
        let fc = FrequencyConfig::default();
        let ctx = ParallelContext::default();
        let r = harmonic_frequencies(&ctx, &m, "sto-3g", Operator::coulomb(), &cfg, &fc).unwrap();

        eprintln!("water/STO-3G freqs (cm^-1): {:?}", r.frequencies);
        eprintln!("  trans/rot: {:?}", r.trans_rot_frequencies);
        eprintln!("  asymmetry: {:.3e} Ha/Bohr^2, {} gradient evals", r.asymmetry, r.n_gradient_evaluations);

        assert_eq!(r.n_gradient_evaluations, 6 * 3);
        assert_eq!(r.trans_rot_frequencies.len(), 6);
        assert_eq!(r.frequencies.len(), 3);

        // PySCF RHF/STO-3G, isotope_avg masses (generated locally 2026-07-26):
        //   [2043.1061, 4488.0531, 4790.2952]
        const PYSCF: [f64; 3] = [2043.1061, 4488.0531, 4790.2952];
        for (got, want) in r.frequencies.iter().zip(PYSCF.iter()) {
            assert!(
                (got - want).abs() < 1.0,
                "freq {got:.4} vs PySCF {want:.4} (diff {:.4} cm^-1)",
                got - want
            );
        }
    }

    /// Water / cc-pVDZ RHF against PySCF -- the headline external reference.
    #[test]
    fn test_water_ccpvdz_frequencies_vs_pyscf() {
        let m = water();
        let cfg = RhfConfig { energy_conv: 1e-11, ..Default::default() };
        let fc = FrequencyConfig::default();
        let ctx = ParallelContext::default();
        let r = harmonic_frequencies(&ctx, &m, "cc-pvdz", Operator::coulomb(), &cfg, &fc).unwrap();

        eprintln!("water/cc-pVDZ freqs (cm^-1): {:?}", r.frequencies);
        eprintln!("  trans/rot: {:?}", r.trans_rot_frequencies);
        eprintln!("  asymmetry: {:.3e} Ha/Bohr^2", r.asymmetry);
        eprintln!("  ZPE: {:.8} Ha", r.zero_point_energy());

        assert_eq!(r.trans_rot_frequencies.len(), 6);
        assert_eq!(r.frequencies.len(), 3);
        assert_eq!(r.n_imaginary(), 0, "equilibrium water has no imaginary modes");

        // PySCF RHF/cc-pVDZ, isotope_avg masses (generated locally 2026-07-26):
        //   [1804.2202, 3953.5732, 4050.4122]
        const PYSCF: [f64; 3] = [1804.2202, 3953.5732, 4050.4122];
        for (got, want) in r.frequencies.iter().zip(PYSCF.iter()) {
            assert!(
                (got - want).abs() < 2.0,
                "freq {got:.4} vs PySCF {want:.4} (diff {:.4} cm^-1)",
                got - want
            );
        }
    }

    /// CO2 / STO-3G: the linear case, end to end. Exactly 5 zero modes and
    /// 4 vibrations, with the bend doubly degenerate.
    #[test]
    fn test_co2_sto3g_linear_five_zero_modes() {
        let m = co2();
        let cfg = RhfConfig { energy_conv: 1e-11, ..Default::default() };
        let fc = FrequencyConfig::default();
        let ctx = ParallelContext::default();
        let r = harmonic_frequencies(&ctx, &m, "sto-3g", Operator::coulomb(), &cfg, &fc).unwrap();

        eprintln!("CO2/STO-3G freqs (cm^-1): {:?}", r.frequencies);
        eprintln!("  trans/rot: {:?}", r.trans_rot_frequencies);

        assert!(r.is_linear);
        assert_eq!(r.trans_rot_frequencies.len(), 5, "linear molecule: 5 zero modes");
        assert_eq!(r.frequencies.len(), 4, "CO2 has 3N-5 = 4 vibrations");
        for f in &r.trans_rot_frequencies {
            assert!(
                f.abs() < ZERO_MODE_TOL_CM_SCF,
                "SCF trans/rot mode not zero: {f} cm^-1"
            );
        }

        // PySCF RHF/STO-3G (generated locally 2026-07-26):
        //   [434.711, 434.711, 1561.3089, 2807.7276]
        const PYSCF: [f64; 4] = [434.711, 434.711, 1561.3089, 2807.7276];
        for (got, want) in r.frequencies.iter().zip(PYSCF.iter()) {
            assert!(
                (got - want).abs() < 5.0,
                "freq {got:.4} vs PySCF {want:.4} (diff {:.4} cm^-1)",
                got - want
            );
        }
        // The two bends are degenerate by symmetry.
        assert!(
            (r.frequencies[0] - r.frequencies[1]).abs() < 1.0,
            "CO2 bend should be doubly degenerate: {:?}",
            r.frequencies
        );
    }

    /// End-to-end rotation invariance, through the real SCF+gradient path.
    /// Rotating the molecule rotates every integral, so this exercises far
    /// more than the linear-algebra-only test above.
    #[test]
    fn test_water_frequencies_invariant_to_rigid_rotation_scf() {
        let m = water();
        let cfg = RhfConfig { energy_conv: 1e-11, ..Default::default() };
        let fc = FrequencyConfig::default();
        let ctx = ParallelContext::default();
        let base = harmonic_frequencies(&ctx, &m, "sto-3g", Operator::coulomb(), &cfg, &fc).unwrap();

        // Rotate (3-4-5 about z, then about x) and translate.
        let (ca, sa) = (0.6f64, 0.8f64);
        let (cb, sb) = (0.8f64, 0.6f64);
        let mut rot = m.clone();
        for at in rot.atoms.iter_mut() {
            let (x, y, z) = (at.x, at.y, at.zpos);
            let (x1, y1, z1) = (ca * x - sa * y, sa * x + ca * y, z);
            let (x2, y2, z2) = (x1, cb * y1 - sb * z1, sb * y1 + cb * z1);
            at.x = x2 + 2.5;
            at.y = y2 - 1.25;
            at.zpos = z2 + 0.75;
        }
        let r = harmonic_frequencies(&ctx, &rot, "sto-3g", Operator::coulomb(), &cfg, &fc).unwrap();

        eprintln!("water/STO-3G rotated freqs: {:?}", r.frequencies);
        assert_eq!(r.trans_rot_frequencies.len(), 6);
        for (a, b) in base.frequencies.iter().zip(r.frequencies.iter()) {
            assert!(
                (a - b).abs() < 0.5,
                "rigid rotation changed a frequency: {a:.4} vs {b:.4} cm^-1"
            );
        }
    }
}
