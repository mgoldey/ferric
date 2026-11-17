//! Rigid-rotor / harmonic-oscillator (RRHO) thermochemistry of an ideal gas.
//!
//! [`rrho_thermochemistry`](crate::thermo::rrho_thermochemistry) turns harmonic wavenumbers, the geometry and the
//! electronic energy into translational, rotational, vibrational and
//! electronic internal energy `E`, enthalpy `H`, entropy `S`, Gibbs energy
//! `G`, and heat capacities `Cv`, `Cp`, at a temperature `T` and pressure `P`.
//! The formulas and conventions are those of PySCF `pyscf.hessian.thermo`:
//!
//! * translation: `q = (2 pi M k T / h^2)^(3/2) k T / P`,
//!   `S = R (5/2 + ln q)`, `E = 3/2 R T`, `H = 5/2 R T`;
//! * rotation, linear: `q = k T / (sigma h B)`, `S = R (1 + ln q)`,
//!   `E = H = R T`; nonlinear: `q = sqrt(pi) (k T / h)^(3/2) / (sigma
//!   sqrt(A B C))`, `S = R (3/2 + ln q)`, `E = H = 3/2 R T`; a single atom has
//!   no rotational part;
//! * vibration, per real mode with `x = h c nu / (k T)`:
//!   `S = R (x / (e^x - 1) - ln(1 - e^-x))`,
//!   `E = H = ZPE + R T x / (e^x - 1)`, `ZPE = 1/2 sum h c nu`;
//! * electronic: `S = R ln(2S + 1)` (the spin multiplicity), `E = H = E0`;
//! * `G = H - T S` for every part; the totals are sums of the parts.
//!
//! Every energy is per molecule in Hartree and every entropy and heat
//! capacity in Hartree/K, so `R` here is `k_B / E_h`. Derivation:
//! `book/src/14-derivatives/06-frequencies.md#frequencies-thermochemistry-ir-intensities`; proof:
//! `scripts/rrho_thermo_sympy.py`.
//!
//! Imaginary modes (negative wavenumbers) are left out of the vibrational
//! sums, as in PySCF, and each run that drops one records a warning in
//! [`Thermochemistry::warnings`](crate::thermo::Thermochemistry::warnings). The rotational symmetry number is an
//! input ([`ThermoConfig::symmetry_number`](crate::thermo::ThermoConfig::symmetry_number), default 1): ferric does not
//! detect non-Abelian point groups, so it cannot count the proper rotations
//! of, say, NH3 (sigma 3) or CH4 (sigma 12). With sigma = 1 the rotational
//! entropy is too large by `R ln sigma`.
//!
//! Constants are CODATA 2018. Lengths are converted to metres with ferric's
//! own Bohr radius ([`ferric_core::units::BOHR_RADIUS_ANGSTROM`]), the value
//! that read the input geometry, so the moments of inertia are those of the
//! geometry the user gave in Angstrom.

use ferric_core::mol::Molecule;
use ferric_core::units::BOHR_RADIUS_ANGSTROM;
use ferric_core::FerricError;

use crate::frequencies::{atom_masses, detect_linear};

/// Boltzmann constant, J/K (CODATA 2018, exact).
pub const BOLTZMANN_J_PER_K: f64 = 1.380_649e-23;
/// Planck constant, J s (CODATA 2018, exact).
pub const PLANCK_J_S: f64 = 6.626_070_15e-34;
/// Speed of light, m/s (exact).
pub const LIGHT_SPEED_M_PER_S: f64 = 299_792_458.0;
/// Hartree energy, J (CODATA 2018).
pub const HARTREE_J: f64 = 4.359_744_722_207_1e-18;
/// Atomic mass constant, kg (CODATA 2018).
pub const ATOMIC_MASS_KG: f64 = 1.660_539_066_60e-27;
/// One standard atmosphere, Pa (exact).
pub const ATMOSPHERE_PA: f64 = 101_325.0;
/// Default temperature, K.
pub const DEFAULT_TEMPERATURE_K: f64 = 298.15;

/// `k_B / E_h`: the gas constant per molecule in Hartree/K.
const R_EH: f64 = BOLTZMANN_J_PER_K / HARTREE_J;

/// Temperature, pressure and rotational symmetry number of a thermochemistry
/// evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermoConfig {
    /// Temperature in K (default 298.15).
    pub temperature: f64,
    /// Pressure in Pa (default 101325, one atmosphere).
    pub pressure: f64,
    /// Rotational symmetry number: the number of proper rotations of the
    /// point group (1 for C1/Cs/Ci/C_inf_v, 2 for C2v and D_inf_h, 3 for C3v,
    /// 12 for Td). Default 1; ferric does not detect it.
    pub symmetry_number: u32,
}

impl Default for ThermoConfig {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE_K,
            pressure: ATMOSPHERE_PA,
            symmetry_number: 1,
        }
    }
}

impl ThermoConfig {
    /// Error on a non-positive or non-finite temperature or pressure, or a
    /// zero symmetry number.
    pub fn validate(&self) -> Result<(), FerricError> {
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(FerricError::General(format!(
                "thermochemistry temperature must be finite and > 0 K, got {}",
                self.temperature
            )));
        }
        if !(self.pressure.is_finite() && self.pressure > 0.0) {
            return Err(FerricError::General(format!(
                "thermochemistry pressure must be finite and > 0 Pa, got {}",
                self.pressure
            )));
        }
        if self.symmetry_number == 0 {
            return Err(FerricError::General(
                "rotational symmetry number must be >= 1".into(),
            ));
        }
        Ok(())
    }
}

/// One contribution to the thermochemistry. Energies in Hartree, entropies
/// and heat capacities in Hartree/K, per molecule.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ThermoTerm {
    /// Internal energy `E`.
    pub energy: f64,
    /// Enthalpy `H`.
    pub enthalpy: f64,
    /// Entropy `S`.
    pub entropy: f64,
    /// Gibbs energy `G = H - T S`.
    pub gibbs: f64,
    /// Constant-volume heat capacity.
    pub cv: f64,
    /// Constant-pressure heat capacity.
    pub cp: f64,
}

impl ThermoTerm {
    fn new(energy: f64, enthalpy: f64, entropy: f64, cv: f64, cp: f64, t: f64) -> Self {
        Self {
            energy,
            enthalpy,
            entropy,
            gibbs: enthalpy - t * entropy,
            cv,
            cp,
        }
    }
}

/// Rotor type used for the rotational partition function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotor {
    /// A single atom: no rotational part.
    Atom,
    /// Linear molecule: one rotational constant.
    Linear,
    /// Nonlinear molecule: three rotational constants.
    Nonlinear,
}

impl Rotor {
    /// `"atom"`, `"linear"` or `"nonlinear"`.
    pub fn label(&self) -> &'static str {
        match self {
            Rotor::Atom => "atom",
            Rotor::Linear => "linear",
            Rotor::Nonlinear => "nonlinear",
        }
    }
}

/// RRHO thermochemistry at one temperature and pressure.
#[derive(Debug, Clone, PartialEq)]
pub struct Thermochemistry {
    /// The inputs.
    pub config: ThermoConfig,
    /// Electronic energy at the geometry, Hartree.
    pub e0: f64,
    /// Zero-point vibrational energy, Hartree (real modes only).
    pub zpe: f64,
    /// `E0`, `S = R ln(2S + 1)`.
    pub electronic: ThermoTerm,
    /// Translational part.
    pub translational: ThermoTerm,
    /// Rotational part (zero for an atom).
    pub rotational: ThermoTerm,
    /// Vibrational part, ZPE included in `energy` and `enthalpy`.
    pub vibrational: ThermoTerm,
    /// Sum of the four parts.
    pub total: ThermoTerm,
    /// Rotor type.
    pub rotor: Rotor,
    /// Rotational constants in GHz, largest first: none for an atom, one for
    /// a linear molecule, three otherwise.
    pub rotational_constants_ghz: Vec<f64>,
    /// Number of imaginary modes left out of the vibrational sums.
    pub n_imaginary_excluded: usize,
    /// Warnings about the evaluation (imaginary modes dropped).
    pub warnings: Vec<String>,
}

/// Principal moments of inertia in u Bohr^2, ascending, about the centre of
/// mass.
fn principal_moments(mol: &Molecule, masses_amu: &[f64]) -> Result<[f64; 3], FerricError> {
    let total: f64 = masses_amu.iter().sum();
    let mut com = [0.0; 3];
    for (a, m) in mol.atoms.iter().zip(masses_amu) {
        for (c, x) in com.iter_mut().zip([a.x, a.y, a.zpos]) {
            *c += m * x / total;
        }
    }
    let mut inertia = ndarray::Array2::<f64>::zeros((3, 3));
    for (a, m) in mol.atoms.iter().zip(masses_amu) {
        let r = [a.x - com[0], a.y - com[1], a.zpos - com[2]];
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        for i in 0..3 {
            for j in 0..3 {
                let delta = if i == j { r2 } else { 0.0 };
                inertia[(i, j)] += m * (delta - r[i] * r[j]);
            }
        }
    }
    let (evals, _) = ferric_core::linalg::eigh_dc(&inertia, ferric_core::linalg::Uplo::Lower)?;
    let mut e = [evals[0], evals[1], evals[2]];
    e.sort_by(f64::total_cmp);
    Ok(e)
}

/// Rotational constant `B = h / (8 pi^2 I)` in Hz for a moment `I` in
/// u Bohr^2.
fn rotational_constant_hz(moment_amu_bohr2: f64) -> f64 {
    let bohr_m = BOHR_RADIUS_ANGSTROM * 1.0e-10;
    let i_si = moment_amu_bohr2 * ATOMIC_MASS_KG * bohr_m * bohr_m;
    PLANCK_J_S / (8.0 * std::f64::consts::PI * std::f64::consts::PI * i_si)
}

/// Translational part: Sackur-Tetrode entropy of an ideal gas of mass
/// `mass_amu` at `t` K and `p` Pa.
fn translational(mass_amu: f64, t: f64, p: f64) -> ThermoTerm {
    let m = mass_amu * ATOMIC_MASS_KG;
    let kt = BOLTZMANN_J_PER_K * t;
    let q = (2.0 * std::f64::consts::PI * m * kt / (PLANCK_J_S * PLANCK_J_S)).powf(1.5) * kt / p;
    ThermoTerm::new(
        1.5 * R_EH * t,
        2.5 * R_EH * t,
        R_EH * (2.5 + q.ln()),
        1.5 * R_EH,
        2.5 * R_EH,
        t,
    )
}

/// Rotational part for rotational constants `b_hz` (one for a linear rotor,
/// three otherwise, none for an atom).
fn rotational(rotor: Rotor, b_hz: &[f64], sigma: f64, t: f64) -> ThermoTerm {
    let kt_over_h = BOLTZMANN_J_PER_K * t / PLANCK_J_S;
    match rotor {
        Rotor::Atom => ThermoTerm::default(),
        Rotor::Linear => {
            let q = kt_over_h / (sigma * b_hz[0]);
            ThermoTerm::new(R_EH * t, R_EH * t, R_EH * (1.0 + q.ln()), R_EH, R_EH, t)
        }
        Rotor::Nonlinear => {
            let abc: f64 = b_hz.iter().product();
            let q = kt_over_h.powf(1.5) * std::f64::consts::PI.sqrt() / (sigma * abc.sqrt());
            ThermoTerm::new(
                1.5 * R_EH * t,
                1.5 * R_EH * t,
                R_EH * (1.5 + q.ln()),
                1.5 * R_EH,
                1.5 * R_EH,
                t,
            )
        }
    }
}

/// Vibrational part from the real wavenumbers (cm^-1), and the ZPE.
fn vibrational(real_cm: &[f64], t: f64) -> (ThermoTerm, f64) {
    // h c nu / k_B in K per cm^-1 (the second radiation constant).
    let c2 = 100.0 * PLANCK_J_S * LIGHT_SPEED_M_PER_S / BOLTZMANN_J_PER_K;
    let mut zpe = 0.0;
    let (mut s, mut e_th, mut cv) = (0.0, 0.0, 0.0);
    for &nu in real_cm {
        let theta = c2 * nu;
        let x = theta / t;
        // exp_m1 keeps x/(e^x - 1) accurate for soft modes (x -> 0).
        let bose = 1.0 / x.exp_m1();
        zpe += 0.5 * R_EH * theta;
        s += x * bose - (-(-x).exp_m1()).ln();
        e_th += x * bose;
        // x^2 e^-x / (1 - e^-x)^2: no overflow for stiff modes at low T.
        let em1 = (-x).exp_m1();
        cv += x * x * (-x).exp() / (em1 * em1);
    }
    let energy = zpe + R_EH * t * e_th;
    (
        ThermoTerm::new(energy, energy, R_EH * s, R_EH * cv, R_EH * cv, t),
        zpe,
    )
}

/// RRHO thermochemistry of `mol` with harmonic wavenumbers
/// `frequencies_cm` (cm^-1; negative = imaginary, left out with a warning)
/// and electronic energy `e0` (Hartree). Masses are
/// [`atom_masses`]'s isotope-averaged standard atomic weights, as in the
/// frequency calculation; the spin multiplicity is `mol.multiplicity`.
pub fn rrho_thermochemistry(
    mol: &Molecule,
    frequencies_cm: &[f64],
    e0: f64,
    config: &ThermoConfig,
) -> Result<Thermochemistry, FerricError> {
    config.validate()?;
    let masses = atom_masses(mol)?;
    rrho_thermochemistry_with_masses(mol, &masses, frequencies_cm, e0, config)
}

/// [`rrho_thermochemistry`] with explicit masses (u), one per atom.
pub fn rrho_thermochemistry_with_masses(
    mol: &Molecule,
    masses_amu: &[f64],
    frequencies_cm: &[f64],
    e0: f64,
    config: &ThermoConfig,
) -> Result<Thermochemistry, FerricError> {
    config.validate()?;
    let natoms = mol.atoms.len();
    if masses_amu.len() != natoms || natoms == 0 {
        return Err(FerricError::General(format!(
            "thermochemistry: {} masses for {natoms} atoms",
            masses_amu.len()
        )));
    }
    let t = config.temperature;
    let rotor = if natoms == 1 {
        Rotor::Atom
    } else if detect_linear(mol, masses_amu) {
        Rotor::Linear
    } else {
        Rotor::Nonlinear
    };
    let moments = principal_moments(mol, masses_amu)?;
    let b_hz: Vec<f64> = match rotor {
        Rotor::Atom => Vec::new(),
        // The two equal nonzero moments; the largest is the better-conditioned.
        Rotor::Linear => vec![rotational_constant_hz(moments[2])],
        Rotor::Nonlinear => moments.iter().map(|&i| rotational_constant_hz(i)).collect(),
    };

    let real: Vec<f64> = frequencies_cm
        .iter()
        .copied()
        .filter(|f| *f > 0.0)
        .collect();
    let n_imaginary_excluded = frequencies_cm.iter().filter(|f| **f < 0.0).count();
    let mut warnings = Vec::new();
    if n_imaginary_excluded > 0 {
        warnings.push(format!(
            "thermochemistry: {n_imaginary_excluded} imaginary mode(s) left out of the \
             vibrational sums; the geometry is not a minimum, so the vibrational terms \
             (and G) are not those of a stable species"
        ));
    }

    let electronic = ThermoTerm::new(
        e0,
        e0,
        R_EH * (mol.multiplicity.max(1) as f64).ln(),
        0.0,
        0.0,
        t,
    );
    let translational = translational(masses_amu.iter().sum(), t, config.pressure);
    let rotational = rotational(rotor, &b_hz, f64::from(config.symmetry_number), t);
    let (vibrational, zpe) = vibrational(&real, t);
    let parts = [electronic, translational, rotational, vibrational];
    let sum = |f: fn(&ThermoTerm) -> f64| parts.iter().map(f).sum::<f64>();
    let total = ThermoTerm {
        energy: sum(|p| p.energy),
        enthalpy: sum(|p| p.enthalpy),
        entropy: sum(|p| p.entropy),
        gibbs: sum(|p| p.gibbs),
        cv: sum(|p| p.cv),
        cp: sum(|p| p.cp),
    };
    let mut rotational_constants_ghz: Vec<f64> = b_hz.iter().map(|b| b * 1e-9).collect();
    rotational_constants_ghz.sort_by(|a, b| b.total_cmp(a));
    Ok(Thermochemistry {
        config: *config,
        e0,
        zpe,
        electronic,
        translational,
        rotational,
        vibrational,
        total,
        rotor,
        rotational_constants_ghz,
        n_imaginary_excluded,
        warnings,
    })
}

impl ThermoTerm {
    /// `{energy, enthalpy, entropy, gibbs, cv, cp}`, Hartree and Hartree/K.
    pub fn to_json(&self) -> serde_json::Value {
        use crate::runlog::json_f64;
        serde_json::json!({
            "energy": json_f64(self.energy),
            "enthalpy": json_f64(self.enthalpy),
            "entropy": json_f64(self.entropy),
            "gibbs": json_f64(self.gibbs),
            "cv": json_f64(self.cv),
            "cp": json_f64(self.cp),
        })
    }
}

impl Thermochemistry {
    /// The result as JSON (the CLI run log's `thermochemistry` field and the
    /// Python `FrequencyResult.thermo` dict): inputs, rotor, rotational
    /// constants (GHz), `e0` and `zpe` (Hartree), the excluded-mode count,
    /// warnings, and each part's [`ThermoTerm::to_json`].
    pub fn to_json(&self) -> serde_json::Value {
        use crate::runlog::json_f64;
        serde_json::json!({
            "temperature_k": self.config.temperature,
            "pressure_pa": self.config.pressure,
            "symmetry_number": self.config.symmetry_number,
            "rotor": self.rotor.label(),
            "rotational_constants_ghz": self.rotational_constants_ghz,
            "e0": json_f64(self.e0),
            "zpe": json_f64(self.zpe),
            "n_imaginary_excluded": self.n_imaginary_excluded,
            "warnings": self.warnings,
            "electronic": self.electronic.to_json(),
            "translational": self.translational.to_json(),
            "rotational": self.rotational.to_json(),
            "vibrational": self.vibrational.to_json(),
            "total": self.total.to_json(),
        })
    }
}

impl crate::frequencies::FrequencyResult {
    /// RRHO thermochemistry from these frequencies at `mol` (the geometry the
    /// frequencies were computed at), with [`Self::energy`] as `E0`. See
    /// [`rrho_thermochemistry`].
    pub fn thermochemistry(
        &self,
        mol: &Molecule,
        config: &ThermoConfig,
    ) -> Result<Thermochemistry, FerricError> {
        rrho_thermochemistry(mol, &self.frequencies, self.energy, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(sym: &str) -> Molecule {
        Molecule::parse_xyz(&format!("1\n\n{sym} 0 0 0\n"), 0, 1).unwrap()
    }

    /// The trivial limit: one atom has no rotation or vibration, so S is the
    /// Sackur-Tetrode entropy, written here from SI constants independently
    /// of the module (`k ln[(2 pi m k T)^(3/2) k T / (h^3 P)] + 5/2 k`).
    /// NIST-JANAF gives 154.845 J/(mol K) for argon at 298.15 K and 1 bar.
    #[test]
    fn thermo_matches_exact_in_the_trivial_limit() {
        let mol = atom("Ar");
        let m_u = 39.948;
        let cfg = ThermoConfig {
            pressure: 1.0e5,
            ..ThermoConfig::default()
        };
        let th = rrho_thermochemistry_with_masses(&mol, &[m_u], &[], -526.0, &cfg).unwrap();
        let (k, h, t, p): (f64, f64, f64, f64) = (1.380649e-23, 6.62607015e-34, 298.15, 1.0e5);
        let m = m_u * 1.66053906660e-27;
        let s_si = k
            * (((2.0 * std::f64::consts::PI * m * k * t).powf(1.5) * k * t / (h.powi(3) * p)).ln()
                + 2.5);
        let s_jmolk = s_si * 6.02214076e23;
        let ours = th.total.entropy * HARTREE_J * 6.02214076e23;
        assert!((ours - s_jmolk).abs() < 1e-9, "{ours} vs {s_jmolk}");
        assert!((ours - 154.845).abs() < 0.01, "{ours} vs NIST 154.845");
        assert_eq!(th.rotor, Rotor::Atom);
        assert_eq!(th.rotational, ThermoTerm::default());
        assert_eq!(th.vibrational, ThermoTerm::default());
        assert!((th.total.enthalpy - (-526.0 + 2.5 * R_EH * t)).abs() < 1e-15);
    }

    /// High-temperature (classical) limits: a harmonic mode tends to
    /// E - ZPE -> k T and S -> k (1 + ln(k T / h c nu)); a rotor's E is
    /// already classical.
    #[test]
    fn vibrational_terms_reach_the_classical_limit() {
        let nu = 100.0;
        let t = 1.0e6;
        let (v, zpe) = vibrational(&[nu], t);
        let c2 = 100.0 * PLANCK_J_S * LIGHT_SPEED_M_PER_S / BOLTZMANN_J_PER_K;
        let x = c2 * nu / t;
        assert!(((v.energy - zpe) / (R_EH * t) - 1.0).abs() < x);
        let s_classical = R_EH * (1.0 - x.ln());
        assert!((v.entropy - s_classical).abs() / s_classical < x * x);
        assert!((v.cv / R_EH - 1.0).abs() < x * x);
    }

    /// A wrong symmetry number moves S_rot by exactly R ln sigma.
    #[test]
    fn symmetry_number_shifts_the_rotational_entropy_by_r_ln_sigma() {
        let mol = Molecule::parse_xyz(
            "3\n\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
            0,
            1,
        )
        .unwrap();
        let f = [1600.0, 3700.0, 3800.0];
        let one = rrho_thermochemistry(&mol, &f, 0.0, &ThermoConfig::default()).unwrap();
        let two = rrho_thermochemistry(
            &mol,
            &f,
            0.0,
            &ThermoConfig {
                symmetry_number: 2,
                ..ThermoConfig::default()
            },
        )
        .unwrap();
        let d = one.rotational.entropy - two.rotational.entropy;
        assert!((d - R_EH * 2f64.ln()).abs() < 1e-18);
        assert_eq!(one.rotor, Rotor::Nonlinear);
        assert_eq!(one.rotational_constants_ghz.len(), 3);
    }

    /// Imaginary modes are dropped with a warning; the ZPE matches
    /// `FrequencyResult::zero_point_energy`'s formula.
    #[test]
    fn imaginary_modes_are_excluded_with_a_warning() {
        let mol = Molecule::parse_xyz("2\n\nH 0 0 0\nF 0 0 0.92\n", 0, 1).unwrap();
        let th =
            rrho_thermochemistry(&mol, &[-50.0, 4000.0], 0.0, &ThermoConfig::default()).unwrap();
        assert_eq!(th.n_imaginary_excluded, 1);
        assert_eq!(th.warnings.len(), 1);
        assert_eq!(th.rotor, Rotor::Linear);
        assert!((th.zpe - 0.5 * 4000.0 / 219_474.631_363_2).abs() < 1e-12);
        let clean = rrho_thermochemistry(&mol, &[4000.0], 0.0, &ThermoConfig::default()).unwrap();
        assert!(clean.warnings.is_empty());
        assert_eq!(clean.vibrational, th.vibrational);
    }

    #[test]
    fn bad_inputs_are_refused() {
        let mol = atom("He");
        for cfg in [
            ThermoConfig {
                temperature: 0.0,
                ..Default::default()
            },
            ThermoConfig {
                pressure: -1.0,
                ..Default::default()
            },
            ThermoConfig {
                symmetry_number: 0,
                ..Default::default()
            },
        ] {
            assert!(rrho_thermochemistry(&mol, &[], 0.0, &cfg).is_err());
        }
    }
}
