//! Harmonic infrared intensities of SCF references.
//!
//! The intensity of normal mode `k` is the integrated absorption coefficient
//!
//! ```text
//! A_k = N_A / (12 eps0 c^2) * |d mu / d Q_k|^2,
//! d mu_a / d Q_k = sum_i P[a, i] x_k[i],
//! ```
//!
//! with `P[a, i] = d mu_a / d R_i` the atomic polar tensor (the dipole
//! derivative with respect to the Cartesian nuclear coordinate `i`) and `x_k`
//! the Cartesian displacement of mode `k` per unit mass-weighted normal
//! coordinate `Q_k`. [`crate::frequencies::FrequencyResult::normal_modes`] holds
//! exactly those `x_k`: unit vectors in electron-mass-weighted coordinates
//! divided back through by `sqrt(m)`, so `sum_i m_i x_k[i]^2 = 1` with `m` in
//! electron masses and no further mass factor is applied here. In atomic units
//! `d mu / d Q` is in `e / sqrt(m_e)`, and [`AU_TO_KM_PER_MOL`](crate::infrared::AU_TO_KM_PER_MOL) converts
//! `|d mu / d Q|^2` to km/mol. Derivation:
//! `book/src/14-derivatives/06-frequencies.md#frequencies-thermochemistry-ir-intensities`; proof:
//! `scripts/ir_intensity_sympy.py`.
//!
//! # Two routes to the dipole derivative
//!
//! In a uniform field `F` ferric's energy is `E(F) = E(0) - mu . F + ...`
//! (a charge `q` at `R` gains `-q F . R`), so `mu = -dE/dF` and
//!
//! ```text
//! P[a, i] = d mu_a / d R_i = - d^2 E / (dF_a dR_i) = - d g_i / d F_a .
//! ```
//!
//! [`dipole_derivatives_field`](crate::infrared::dipole_derivatives_field) (the production route) central-differences
//! the ANALYTIC nuclear gradient in a field `+-h e_a`: six SCF + gradient
//! evaluations whatever the molecule size. It is exact in the SCF's own
//! response, because the gradient of an SCF solved in the field carries the
//! field's orbital relaxation. [`dipole_derivatives_dipole_fd`](crate::infrared::dipole_derivatives_dipole_fd) instead
//! central-differences the SCF dipole along each Cartesian displacement
//! (`6N` SCFs); the two share no code beyond the SCF solver, so their
//! agreement tests both.
//!
//! The field route runs its SCFs at `density_conv <= IR_DENSITY_CONV`: the
//! APT error from SCF exits is `K n density_conv / h` (the gradient bound of
//! the frequencies module, divided by the field step), and at the default
//! `h = 1e-3` a loose threshold would dominate the `O(h^2)` truncation.
//!
//! The translational sum rule `sum_A P[a, (A, b)] = Q delta_ab` (`Q` the
//! molecular charge) holds for both routes in exact arithmetic when no
//! external point charges are present; [`IrSpectrum::sum_rule_residual`](crate::infrared::IrSpectrum::sum_rule_residual)
//! records how far the computed tensor is from it.

use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ndarray::Array2;

use crate::frequencies::{energy_gradient_and_result, FrequencyReference, FrequencyResult};
use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;
use crate::warm_start::{solve_rhf_warm, solve_rohf_warm, solve_uhf_warm, ScfWarmStart};

/// `N_A e^2 / (12 eps0 c^2 m_e)` in km/mol: the intensity of a mode with
/// `|d mu / d Q| = 1 e / sqrt(m_e)` (CODATA 2018). Equals
/// 974.880 km/mol per `e^2 / u` times `m_u / m_e`.
pub const AU_TO_KM_PER_MOL: f64 = 1_777_097.731_306;

/// Default field step of [`dipole_derivatives_field`], atomic units.
///
/// The central difference has truncation `h^2 / 6 |d^4 E / dF^3 dR|` (a
/// hyperpolarizability derivative, `O(10-100)` au for small molecules) and
/// SCF noise `K n density_conv / h`; `1e-3` keeps both near `1e-5` e at
/// [`IR_DENSITY_CONV`].
pub const DEFAULT_IR_FIELD: f64 = 1.0e-3;

/// Largest `density_conv` the field route's SCFs run at (tighter user values
/// are kept).
pub const IR_DENSITY_CONV: f64 = 1.0e-9;

/// Intensities of a frequency run's normal modes and the dipole derivatives
/// they came from.
#[derive(Debug, Clone)]
pub struct IrSpectrum {
    /// Intensity of each mode of the frequency result, km/mol, in its order
    /// (imaginary modes included).
    pub intensities_km_mol: Vec<f64>,
    /// Atomic polar tensor `d mu_a / d R_i`, `3 x 3N`, atomic units (e),
    /// columns ordered `[x0, y0, z0, x1, ...]`.
    pub dipole_derivatives: Array2<f64>,
    /// Field step used, atomic units.
    pub field_step: f64,
    /// `max |sum_A P[a, (A, b)] - Q delta_ab|`, e; `None` with external point
    /// charges (the sum rule does not hold when the environment stays put).
    pub sum_rule_residual: Option<f64>,
}

/// Refuse what the intensity routes do not cover, before any SCF.
fn preflight(scf_config: &RhfConfig, h: f64) -> Result<(), FerricError> {
    crate::x2c::refuse_relativistic(scf_config.relativistic, "IR intensities")?;
    if !(h.is_finite() && h > 0.0) {
        return Err(FerricError::General(format!(
            "IR intensities: the finite-difference step must be finite and > 0, got {h}"
        )));
    }
    Ok(())
}

/// `config` with `delta` added to its uniform field (the other external
/// charges kept).
fn with_extra_field(config: &RhfConfig, delta: [f64; 3]) -> RhfConfig {
    let mut cfg = config.clone();
    let mut ext: ExternalPotential = cfg.external_potential.clone().unwrap_or_default();
    let f = ext.field.unwrap_or([0.0; 3]);
    ext.field = Some([f[0] + delta[0], f[1] + delta[1], f[2] + delta[2]]);
    cfg.external_potential = Some(ext);
    cfg
}

/// Production route: `P[a, i] = -(g_i(+h e_a) - g_i(-h e_a)) / (2 h)` from
/// the analytic gradient of SCFs solved in the field. `3 x 3N`, e.
pub fn dipole_derivatives_field(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    scf_config: &RhfConfig,
    reference: FrequencyReference,
    h: f64,
) -> Result<Array2<f64>, FerricError> {
    preflight(scf_config, h)?;
    let mut base = scf_config.for_geometry_task();
    base.density_conv = base.density_conv.min(IR_DENSITY_CONV);
    let n = 3 * mol.atoms.len();
    let mut apt = Array2::<f64>::zeros((3, n));
    for a in 0..3 {
        let mut grads = Vec::with_capacity(2);
        for sign in [1.0, -1.0] {
            let mut df = [0.0; 3];
            df[a] = sign * h;
            let cfg = with_extra_field(&base, df);
            let mut warm = ScfWarmStart::anchored(false);
            let (_, g, scf) =
                energy_gradient_and_result(ctx, mol, basis_name, op, &cfg, reference, &mut warm)?;
            crate::optimize::require_converged(&scf)?;
            grads.push(g);
        }
        for (i, (gp, gm)) in grads[0].iter().zip(grads[1].iter()).enumerate() {
            apt[(a, i)] = -(gp - gm) / (2.0 * h);
        }
    }
    Ok(apt)
}

/// The SCF dipole (origin 0, a.u.) at `mol`.
fn scf_dipole(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    config: &RhfConfig,
    reference: FrequencyReference,
) -> Result<[f64; 3], FerricError> {
    let bs = ferric_core::basis::bundled(basis_name)?;
    let prep = PreparedBasis::new(mol, &bs)?;
    let bounds = SchwarzBounds::compute(op, &prep)?;
    let mut warm = ScfWarmStart::anchored(false);
    let res = match reference {
        FrequencyReference::Rhf => solve_rhf_warm(ctx, mol, &prep, op, &bounds, config, &mut warm)?,
        FrequencyReference::Uhf => solve_uhf_warm(ctx, mol, &prep, &bounds, config, &mut warm)?,
        FrequencyReference::Rohf => {
            solve_rohf_warm(ctx, mol, &prep, op, &bounds, config, &mut warm)?
        }
    };
    crate::optimize::require_converged(&res)?;
    crate::properties::dipole_moment(mol, &prep, res.density_total())
}

/// Validation route: `P[a, i] = (mu_a(R + delta e_i) - mu_a(R - delta e_i))
/// / (2 delta)` from SCF dipoles at displaced geometries (`6N` SCFs, `delta`
/// in Bohr). `3 x 3N`, e.
pub fn dipole_derivatives_dipole_fd(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    scf_config: &RhfConfig,
    reference: FrequencyReference,
    delta: f64,
) -> Result<Array2<f64>, FerricError> {
    preflight(scf_config, delta)?;
    let cfg = scf_config.for_geometry_task();
    let n = 3 * mol.atoms.len();
    let mut apt = Array2::<f64>::zeros((3, n));
    for i in 0..n {
        let mut mu = Vec::with_capacity(2);
        for sign in [1.0, -1.0] {
            let mut m = mol.clone();
            let atom = &mut m.atoms[i / 3];
            match i % 3 {
                0 => atom.x += sign * delta,
                1 => atom.y += sign * delta,
                _ => atom.zpos += sign * delta,
            }
            mu.push(scf_dipole(ctx, &m, basis_name, op, &cfg, reference)?);
        }
        for a in 0..3 {
            apt[(a, i)] = (mu[0][a] - mu[1][a]) / (2.0 * delta);
        }
    }
    Ok(apt)
}

/// Intensities (km/mol) of the modes `normal_modes` (rows, Cartesian
/// displacements per unit electron-mass-weighted normal coordinate, as
/// [`FrequencyResult::normal_modes`] stores them) from the atomic polar
/// tensor `apt` (`3 x 3N`, e).
pub fn ir_intensities(
    apt: &Array2<f64>,
    normal_modes: &Array2<f64>,
) -> Result<Vec<f64>, FerricError> {
    if apt.nrows() != 3 || apt.ncols() != normal_modes.ncols() {
        return Err(FerricError::General(format!(
            "IR intensities: dipole derivatives {:?} do not match normal modes {:?}",
            apt.dim(),
            normal_modes.dim()
        )));
    }
    // dmu[a, k] = sum_i P[a, i] x_k[i]
    let dmu = apt.dot(&normal_modes.t());
    Ok((0..normal_modes.nrows())
        .map(|k| AU_TO_KM_PER_MOL * (0..3).map(|a| dmu[(a, k)].powi(2)).sum::<f64>())
        .collect())
}

/// `max |sum_A P[a, (A, b)] - Q delta_ab|` for molecular charge `charge`.
pub fn sum_rule_residual(apt: &Array2<f64>, charge: f64) -> f64 {
    let natoms = apt.ncols() / 3;
    let mut worst = 0.0f64;
    for a in 0..3 {
        for b in 0..3 {
            let s: f64 = (0..natoms).map(|at| apt[(a, 3 * at + b)]).sum();
            let want = if a == b { charge } else { 0.0 };
            worst = worst.max((s - want).abs());
        }
    }
    worst
}

/// IR intensities of `freq`'s modes for the SCF `reference` at `mol`, by the
/// field route ([`dipole_derivatives_field`], step `h`; [`DEFAULT_IR_FIELD`]
/// is the default). `scf_config` is the frequency run's.
#[allow(clippy::too_many_arguments)]
pub fn ir_spectrum(
    ctx: &ParallelContext,
    mol: &Molecule,
    basis_name: &str,
    op: Operator,
    scf_config: &RhfConfig,
    reference: FrequencyReference,
    freq: &FrequencyResult,
    h: f64,
) -> Result<IrSpectrum, FerricError> {
    let apt = dipole_derivatives_field(ctx, mol, basis_name, op, scf_config, reference, h)?;
    let intensities_km_mol = ir_intensities(&apt, &freq.normal_modes)?;
    let has_charges = scf_config
        .external_potential
        .as_ref()
        .is_some_and(|e| !e.point_charges.is_empty() || !e.smeared_charges.is_empty());
    let sum_rule_residual = (!has_charges).then(|| sum_rule_residual(&apt, f64::from(mol.charge)));
    Ok(IrSpectrum {
        intensities_km_mol,
        dipole_derivatives: apt,
        field_step: h,
        sum_rule_residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequencies::{atom_masses, frequencies_from_cartesian_hessian};

    /// The trivial limit: fixed point charges `+q`, `-q` on a diatomic spring
    /// make the dipole exactly linear, `mu = sum_A q_A R_A`, so `P = q_A I`
    /// with no truncation. The stretch then has
    /// `d mu / d Q = q / sqrt(mu_red)` (`mu_red` the reduced mass in electron
    /// masses), so `A = AU_TO_KM_PER_MOL q^2 / mu_red` in closed form.
    #[test]
    fn ir_intensity_matches_exact_in_the_trivial_limit() {
        let mol = Molecule::parse_xyz("2\n\nH 0 0 0\nF 0 0 0.92\n", 0, 1).unwrap();
        let k = 0.6;
        let mut hess = Array2::<f64>::zeros((6, 6));
        hess[(2, 2)] = k;
        hess[(5, 5)] = k;
        hess[(2, 5)] = -k;
        hess[(5, 2)] = -k;
        let masses = atom_masses(&mol).unwrap();
        let freq = frequencies_from_cartesian_hessian(&mol, &hess, &masses).unwrap();
        assert_eq!(freq.frequencies.len(), 1);
        let q = 0.4;
        let mut apt = Array2::<f64>::zeros((3, 6));
        for b in 0..3 {
            apt[(b, b)] = q;
            apt[(b, 3 + b)] = -q;
        }
        let a = ir_intensities(&apt, &freq.normal_modes).unwrap();
        let me_per_u = 1.0 / 5.485_799_090_65e-4;
        let mu_red = masses[0] * masses[1] / (masses[0] + masses[1]) * me_per_u;
        let exact = AU_TO_KM_PER_MOL * q * q / mu_red;
        assert!((a[0] - exact).abs() / exact < 1e-12, "{} vs {exact}", a[0]);
        assert!(sum_rule_residual(&apt, 0.0) < 1e-15);
    }

    /// A homonuclear diatomic with equal charges has no dipole change on
    /// stretching: zero intensity.
    #[test]
    fn symmetric_stretch_of_equal_charges_is_dark() {
        let mol = Molecule::parse_xyz("2\n\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let mut hess = Array2::<f64>::zeros((6, 6));
        hess[(2, 2)] = 0.4;
        hess[(5, 5)] = 0.4;
        hess[(2, 5)] = -0.4;
        hess[(5, 2)] = -0.4;
        let freq =
            frequencies_from_cartesian_hessian(&mol, &hess, &atom_masses(&mol).unwrap()).unwrap();
        let mut apt = Array2::<f64>::zeros((3, 6));
        for b in 0..3 {
            apt[(b, b)] = 0.3;
            apt[(b, 3 + b)] = 0.3;
        }
        let a = ir_intensities(&apt, &freq.normal_modes).unwrap();
        assert!(a[0].abs() < 1e-20);
    }

    #[test]
    fn mismatched_shapes_are_refused() {
        let apt = Array2::<f64>::zeros((3, 6));
        let modes = Array2::<f64>::zeros((1, 9));
        assert!(ir_intensities(&apt, &modes).is_err());
    }
}
