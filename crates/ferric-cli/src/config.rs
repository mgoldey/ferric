use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    pub molecule: MoleculeCfg,
    pub basis: BasisCfg,
    pub method: MethodCfg,
    #[serde(default)]
    pub scf: ScfCfg,
}

#[derive(Deserialize)]
pub struct MoleculeCfg {
    pub xyz: String,
}

#[derive(Deserialize)]
pub struct BasisCfg {
    pub name: Option<String>,
    pub path: Option<String>,
}

#[derive(Deserialize)]
pub struct MethodCfg {
    pub kind: String,
}

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
}

impl Default for ScfCfg {
    fn default() -> Self {
        ScfCfg {
            max_iter: 100,
            energy_conv: 1e-8,
            density_conv: 1e-7,
            diis_size: 8,
            integral_thresh: 1e-12,
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
}
