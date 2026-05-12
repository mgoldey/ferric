use crate::elements::symbol_to_z;
use crate::FerricError;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Clone)]
pub struct Shell {
    pub l: i32,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct BasisSet {
    pub name: String,
    pub shells: HashMap<i32, Vec<Shell>>,
}

impl BasisSet {
    pub fn for_element(&self, z: i32) -> Option<&[Shell]> {
        self.shells.get(&z).map(|v| v.as_slice())
    }
}

pub fn num_functions(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

// --- BSE-JSON parser ---

#[derive(Deserialize)]
struct BseFile {
    name: Option<String>,
    elements: HashMap<String, BseElement>,
}

#[derive(Deserialize)]
struct BseElement {
    electron_shells: Vec<BseShell>,
}

#[derive(Deserialize)]
struct BseShell {
    angular_momentum: Vec<i32>,
    exponents: Vec<String>,
    coefficients: Vec<Vec<String>>,
}

pub fn load_bse_json(path: &str) -> Result<BasisSet, FerricError> {
    let text = fs::read_to_string(path).map_err(FerricError::Io)?;
    parse_bse_json(&text, path)
}

fn parse_bse_json(text: &str, name: &str) -> Result<BasisSet, FerricError> {
    let bf: BseFile = serde_json::from_str(text).map_err(|e| FerricError::Basis(format!("BSE JSON: {e}")))?;
    let mut shells: HashMap<i32, Vec<Shell>> = HashMap::new();
    let bs_name = bf.name.unwrap_or_else(|| name.to_string());
    for (z_str, elem) in &bf.elements {
        let z: i32 = z_str.parse().map_err(|e| FerricError::Basis(format!("bad element key {z_str:?}: {e}")))?;
        for sh in &elem.electron_shells {
            let exps = parse_float_list(&sh.exponents)?;
            for (k, &l) in sh.angular_momentum.iter().enumerate() {
                if k >= sh.coefficients.len() {
                    return Err(FerricError::Basis(format!("shell missing coefficient column {k}")));
                }
                let coefs = parse_float_list(&sh.coefficients[k])?;
                if coefs.len() != exps.len() {
                    return Err(FerricError::Basis(format!("{} coeffs vs {} exps", coefs.len(), exps.len())));
                }
                shells.entry(z).or_default().push(Shell { l, exponents: exps.clone(), coefficients: coefs });
            }
        }
    }
    Ok(BasisSet { name: bs_name, shells })
}

fn parse_float_list(ss: &[String]) -> Result<Vec<f64>, FerricError> {
    ss.iter().map(|s| s.parse::<f64>().map_err(|e| FerricError::Basis(format!("bad float {s:?}: {e}")))).collect()
}

// --- Gaussian-94 parser ---

pub fn load_g94(path: &str) -> Result<BasisSet, FerricError> {
    let text = fs::read_to_string(path).map_err(FerricError::Io)?;
    parse_g94(&text, path)
}

fn parse_g94(text: &str, name: &str) -> Result<BasisSet, FerricError> {
    let mut shells: HashMap<i32, Vec<Shell>> = HashMap::new();
    let mut cur_z: Option<i32> = None;
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('!') { continue; }
        if line == "****" { cur_z = None; continue; }
        if cur_z.is_none() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() >= 2 {
                if let Some(z) = symbol_to_z(fields[0]) { cur_z = Some(z); }
            }
            continue;
        }
        let z = cur_z.unwrap();
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 { continue; }
        let shell_type = fields[0].to_uppercase();
        let nprim: usize = fields[1].parse().map_err(|e| FerricError::Basis(format!("bad nprim: {e}")))?;
        let is_sp = shell_type == "SP";
        let ncol = if is_sp { 2 } else { 1 };
        let mut exps = Vec::with_capacity(nprim);
        let mut coefs: Vec<Vec<f64>> = (0..ncol).map(|_| Vec::with_capacity(nprim)).collect();
        for _ in 0..nprim {
            let pline = lines.next().ok_or_else(|| FerricError::Basis("unexpected EOF in shell".into()))?;
            let pf: Vec<&str> = pline.split_whitespace().collect();
            if pf.len() < 1 + ncol {
                return Err(FerricError::Basis(format!("primitive line has {} fields, want {}", pf.len(), 1 + ncol)));
            }
            exps.push(pf[0].parse::<f64>().map_err(|e| FerricError::Basis(format!("bad exponent: {e}")))?);
            for c in 0..ncol {
                coefs[c].push(pf[1 + c].parse::<f64>().map_err(|e| FerricError::Basis(format!("bad coefficient: {e}")))?);
            }
        }
        let l = match shell_type.as_str() {
            "S" | "SP" => 0, "P" => 1, "D" => 2, "F" => 3, "G" => 4,
            other => return Err(FerricError::Basis(format!("unknown shell type {other:?}"))),
        };
        if is_sp {
            shells.entry(z).or_default().push(Shell { l: 0, exponents: exps.clone(), coefficients: coefs[0].clone() });
            shells.entry(z).or_default().push(Shell { l: 1, exponents: exps, coefficients: coefs[1].clone() });
        } else {
            shells.entry(z).or_default().push(Shell { l, exponents: exps, coefficients: coefs[0].clone() });
        }
    }
    Ok(BasisSet { name: name.to_string(), shells })
}

// --- Bundled basis sets ---

fn canonical_name(name: &str) -> String { name.to_ascii_lowercase() }

pub fn bundled(name: &str) -> Result<BasisSet, FerricError> {
    let cn = canonical_name(name);
    let json = match cn.as_str() {
        "sto-3g" => include_str!("basis/bundled/sto-3g.json"),
        "6-31g" => include_str!("basis/bundled/6-31g.json"),
        "cc-pvdz" => include_str!("basis/bundled/cc-pvdz.json"),
        "def2-svp" => include_str!("basis/bundled/def2-svp.json"),
        _ => return Err(FerricError::Basis(format!("unknown bundled basis {name:?}"))),
    };
    let mut bs = parse_bse_json(json, &cn)?;
    bs.name = cn;
    Ok(bs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundled_sto3g() {
        let bs = bundled("STO-3G").unwrap();
        let h_shells = bs.for_element(1).unwrap();
        assert_eq!(h_shells.len(), 1);
        assert_eq!(h_shells[0].l, 0);
        let c_shells = bs.for_element(6).unwrap();
        assert!(c_shells.len() >= 2);
    }

    #[test]
    fn test_bundled_def2svp_has_d_shells() {
        let bs = bundled("def2-svp").unwrap();
        let c_shells = bs.for_element(6).unwrap();
        assert!(c_shells.iter().any(|s| s.l == 2), "def2-SVP carbon should have d shells");
    }

    #[test]
    fn test_bundled_case_insensitive() {
        let bs1 = bundled("sto-3g").unwrap();
        let bs2 = bundled("STO-3G").unwrap();
        assert_eq!(bs1.name, bs2.name);
    }

    #[test]
    fn test_bundled_631g_oxygen() {
        let bs = bundled("6-31g").unwrap();
        let o_shells = bs.for_element(8).unwrap();
        assert!(o_shells.len() >= 3, "6-31G oxygen needs >=3 shells");
    }

    #[test]
    fn test_num_functions() {
        assert_eq!(num_functions(0), 1);
        assert_eq!(num_functions(1), 3);
        assert_eq!(num_functions(2), 6);
    }
}
