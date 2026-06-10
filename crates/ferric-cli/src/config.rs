use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    pub molecule: MoleculeCfg,
    pub basis: BasisCfg,
    pub method: MethodCfg,
    #[serde(default)]
    pub scf: ScfCfg,
    #[serde(default)]
    pub mp2: Mp2Cfg,
    #[serde(default)]
    pub optimize: OptimizeCfg,
    #[serde(default)]
    pub rpa: RpaCfg,
    #[serde(default)]
    pub dft: DftCfg,
}

#[derive(Deserialize, Default)]
pub struct DftCfg {
    /// XC functional name: "LDA", "PBE", "B3LYP", "wB97X-V", or any libxc name.
    pub functional: Option<String>,
}

#[derive(Deserialize, Default)]
pub struct OptimizeCfg {
    pub max_steps: Option<usize>,
    pub g_max_thresh: Option<f64>,
    pub g_rms_thresh: Option<f64>,
    pub e_conv: Option<f64>,
    pub trust_radius: Option<f64>,
}

#[derive(Deserialize, Default)]
pub struct Mp2Cfg {
    pub auxbasis: Option<String>,
    #[serde(default)]
    pub frozen_core: usize,
    #[allow(dead_code)]
    #[serde(default)]
    pub orbital_optimize: bool,
    /// Range-separation parameter ω in Å⁻¹ (for att-rimp2 and rs-mp2-rpa). Default 0.420.
    pub omega: Option<f64>,
    /// SCS opposite-spin scaling coefficient.
    pub c_os: Option<f64>,
    /// SCS same-spin scaling coefficient.
    pub c_ss: Option<f64>,
    /// Number of Laplace quadrature points (for laplace-mp2).
    pub n_quad: Option<usize>,
}

#[derive(Deserialize, Default)]
pub struct RpaCfg {
    pub auxbasis: Option<String>,
    #[serde(default)]
    pub frozen_core: usize,
    pub n_quad: Option<usize>,
    pub quadrature: Option<String>,
    pub trunc_thresh: Option<f64>,
    pub davidson_conv_thresh: Option<f64>,
    pub u0: Option<f64>,
    #[serde(default)]
    pub run_diagnostics: bool,
    /// If set, write PDEP eigenpotentials to `<prefix>_eigpot_NNN.cube`.
    pub export_eigpot_prefix: Option<String>,
    /// Number of leading eigenpotentials to export (default: 10).
    pub export_eigpot_count: Option<usize>,
    /// Cube grid spacing in Bohr (default: 0.2).
    pub cube_spacing: Option<f64>,
    /// Cube grid margin in Bohr beyond bounding box (default: 4.0).
    pub cube_margin: Option<f64>,
    /// If set, write per-molecule features to this NPZ path (eigenpotentials,
    /// coords, atomic numbers, optional ESP-at-atoms and α tensor).
    pub export_npz: Option<String>,
    /// Compute and include ESP at each nuclear position in the NPZ bundle.
    /// Default: true when `export_npz` is set.
    pub compute_esp: Option<bool>,
    /// Compute and include the static polarizability tensor in the NPZ bundle.
    /// Default: true when `export_npz` is set.
    pub compute_polarizability: Option<bool>,
    /// Compute and include the per-atom Hirshfeld polarizability decomposition
    /// (`alpha_atomic`, shape (N, 3, 3), additive to `alpha_tensor`).
    /// Default: true when `export_npz` is set.
    pub compute_alpha_atomic: Option<bool>,
    /// Compute and include the electric field at each nuclear position in the
    /// NPZ bundle (stored as `electric_field`, shape (natoms, 3), a.u.).
    /// Default: true when `export_npz` is set.
    pub compute_electric_field: Option<bool>,
    /// Include the AO-basis density matrix in the NPZ bundle (stored as
    /// `density_matrix`, shape (n_bf, n_bf), float64). Needed downstream for
    /// CM5 charge derivation and density-derived properties.
    /// Default: true when `export_npz` is set.
    pub compute_density_matrix: Option<bool>,
    /// Compute and include Hirshfeld atomic charges in the NPZ bundle
    /// (stored as `hirshfeld_charges`, shape (natoms,), float64, units of e).
    /// These are the Hirshfeld baseline charges; downstream CM5 pair-correction
    /// is applied in the consumer.
    /// Default: true when `export_npz` is set.
    pub compute_hirshfeld_charges: Option<bool>,
    /// Compute and include Löwdin atomic charges (from symmetrically-
    /// orthogonalized AOs) in the NPZ bundle as `lowdin_charges`,
    /// shape (natoms,), float64, e. Recommended baseline charges for CM5
    /// (no proatom approximation, basis-set-stable).
    /// Default: true when `export_npz` is set.
    pub compute_lowdin_charges: Option<bool>,
    /// Compute per-atom anisotropic C6 dispersion coefficients and include them
    /// in the NPZ bundle (`c6_iso`, `c6_aniso`, `alpha_atomic_dynamic`,
    /// `c6_freqs`, `c6_weights`). Default: true when `export_npz` is set.
    pub compute_c6: Option<bool>,
    /// C6 polarizability source: "ts" (Tkatchenko-Scheffler single-pole,
    /// default) or "pdep" (PDEP-RPA dynamic α(iω), Phase 2).
    pub c6_source: Option<String>,
    /// Per-atom partition for C6: "hirshfeld" (default for pdep) or "becke".
    /// Hirshfeld is required for correct anisotropy in pdep C6 — Becke atom-centred
    /// dipoles lose charge-transfer contributions and invert bond-axis ordering.
    /// For TS, partition only affects alpha_static shape; Hirshfeld volumes are
    /// always used for the volume ratio regardless of this setting.
    pub c6_partition: Option<String>,
    /// XC functional for the RPA *reference* orbitals (e.g. "PBE0", "PBE").
    /// `None` (default) uses a Hartree-Fock reference (RPA@HF). Setting this
    /// runs the closed-shell KS-DFT solver first, so the RPA/PDEP response and
    /// C6 are built on KS orbitals (RPA@PBE0 etc.) — KS orbitals have smaller
    /// HOMO-LUMO gaps, raising the polarizability toward experiment.
    pub xc: Option<String>,
}

#[derive(Deserialize)]
pub struct MoleculeCfg {
    pub xyz: String,
    #[serde(default)]
    pub charge: i32,
    #[serde(default = "default_multiplicity")]
    pub multiplicity: usize,
}

fn default_multiplicity() -> usize { 1 }

#[derive(Deserialize)]
pub struct BasisCfg {
    pub name: Option<String>,
    pub path: Option<String>,
}

#[derive(Deserialize)]
pub struct MethodCfg {
    pub kind: String,
    #[serde(default = "default_task")]
    pub task: String,
}

fn default_task() -> String { "energy".into() }

#[derive(Deserialize)]
pub struct ScfCfg {
    #[serde(default = "default_max_iter")]
    pub max_iter: usize,
    #[serde(default = "default_energy_conv")]
    pub energy_conv: f64,
    #[serde(default = "default_density_conv")]
    pub density_conv: f64,
    #[serde(default = "default_diis_size")]
    pub diis_size: usize,
    #[serde(default = "default_integral_thresh")]
    pub integral_thresh: f64,
    pub k_builder: Option<String>,
    pub df_j_aux: Option<String>,
    pub df_k_aux: Option<String>,
    /// Optional virtual-virtual block level shift (Ha) for open-shell SCF
    /// (UHF / ROHF / UKS / ROKS). The shift is rational-damped by the DIIS
    /// error so the converged Fock is the unshifted stationary point. A
    /// value of 0.2 is a useful default for OH-like doublets at LDA/PBE
    /// where DIIS otherwise plateaus.
    pub level_shift: Option<f64>,
}

impl Default for ScfCfg {
    fn default() -> Self {
        ScfCfg {
            max_iter: 100,
            energy_conv: 1e-8,
            density_conv: 1e-7,
            diis_size: 8,
            integral_thresh: 1e-12,
            k_builder: None,
            df_j_aux: None,
            df_k_aux: None,
            level_shift: None,
        }
    }
}

fn default_max_iter() -> usize { 100 }
fn default_energy_conv() -> f64 { 1e-8 }
fn default_density_conv() -> f64 { 1e-7 }
fn default_diis_size() -> usize { 8 }
fn default_integral_thresh() -> f64 { 1e-12 }

pub fn load_config(path: &str) -> Result<Config, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{e}"))?;
    toml::from_str(&text).map_err(|e| format!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_config() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "sto-3g"
[method]
kind = "rhf"
[scf]
max_iter = 50
energy_conv = 1e-9
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.molecule.xyz, "water.xyz");
        assert_eq!(cfg.basis.name.as_deref(), Some("sto-3g"));
        assert_eq!(cfg.method.kind, "rhf");
        assert_eq!(cfg.scf.max_iter, 50);
    }

    #[test]
    fn test_parse_attenuated_config() {
        let toml_str = r#"
[molecule]
xyz = "water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "att-rimp2"
[mp2]
auxbasis = "cc-pvdz-ri"
omega = 0.420
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.method.kind, "att-rimp2");
        assert!((cfg.mp2.omega.unwrap() - 0.420).abs() < 1e-10);
    }

    #[test]
    fn test_parse_rs_mp2_rpa_config() {
        let toml_str = r#"
[molecule]
xyz = "testdata/molecules/water.xyz"
[basis]
name = "cc-pvdz"
[method]
kind = "rs-mp2-rpa"
[mp2]
auxbasis = "cc-pvdz-ri"
omega = 0.3
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.method.kind, "rs-mp2-rpa");
        assert_eq!(cfg.mp2.omega, Some(0.3));
    }

}
