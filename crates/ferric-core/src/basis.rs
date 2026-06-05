//! Gaussian basis sets: data structures, parsers, and bundled library.
//!
//! Supports BSE-JSON and Gaussian-94 input formats, plus a set of bundled
//! basis sets compiled into the binary via `include_str!`.

use crate::elements::symbol_to_z;
use crate::FerricError;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

/// A single contracted Gaussian shell.
///
/// `l` is the angular momentum quantum number (0=s, 1=p, 2=d, ...).
/// When `pure` is true, spherical harmonics are used (2l+1 functions);
/// otherwise Cartesian Gaussians are used ((l+1)(l+2)/2 functions).
#[derive(Debug, Clone)]
pub struct Shell {
    pub l: i32,
    pub pure: bool,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}

/// A named basis set mapping atomic numbers to their shells.
#[derive(Debug, Clone)]
pub struct BasisSet {
    pub name: String,
    /// Map from atomic number Z to the list of shells for that element.
    pub shells: HashMap<i32, Vec<Shell>>,
}

impl BasisSet {
    /// Return the shells for element with atomic number `z`, or `None` if absent.
    pub fn for_element(&self, z: i32) -> Option<&[Shell]> {
        self.shells.get(&z).map(|v| v.as_slice())
    }
}

/// Number of basis functions for a shell with angular momentum `l`.
///
/// Spherical: 2l+1. Cartesian: (l+1)(l+2)/2.
pub fn num_functions(l: i32, pure: bool) -> usize {
    if pure { (2 * l + 1) as usize } else { ((l + 1) * (l + 2) / 2) as usize }
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
    #[serde(default)]
    function_type: Option<String>,
}

/// Load a basis set from a Basis Set Exchange JSON file.
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
            let pure = match sh.function_type.as_deref() {
                Some("gto_spherical") => true,
                Some("gto_cartesian") => false,
                _ => false, // "gto" or absent defaults to Cartesian
            };
            if sh.angular_momentum.len() == 1 {
                // Single angular momentum — each coefficient column is a separate contraction
                let l = sh.angular_momentum[0];
                let shell_pure = pure && l >= 2;
                for col in &sh.coefficients {
                    let mut coefs = parse_float_list(col)?;
                    if coefs.len() != exps.len() {
                        return Err(FerricError::Basis(format!("{} coeffs vs {} exps", coefs.len(), exps.len())));
                    }
                    renormalize_contraction(&exps, &mut coefs, l);
                    shells.entry(z).or_default().push(Shell { l, pure: shell_pure, exponents: exps.clone(), coefficients: coefs });
                }
            } else {
                // Multiple angular momenta (e.g. SP) — column k corresponds to angular_momentum[k]
                for (k, &l) in sh.angular_momentum.iter().enumerate() {
                    if k >= sh.coefficients.len() {
                        return Err(FerricError::Basis(format!("shell missing coefficient column {k}")));
                    }
                    let mut coefs = parse_float_list(&sh.coefficients[k])?;
                    if coefs.len() != exps.len() {
                        return Err(FerricError::Basis(format!("{} coeffs vs {} exps", coefs.len(), exps.len())));
                    }
                    renormalize_contraction(&exps, &mut coefs, l);
                    let shell_pure = pure && l >= 2;
                    shells.entry(z).or_default().push(Shell { l, pure: shell_pure, exponents: exps.clone(), coefficients: coefs });
                }
            }
        }
    }
    Ok(BasisSet { name: bs_name, shells })
}

/// Rescale a contraction so its contracted AO has unit self-overlap.
///
/// JSON basis files use one of two conventions for the per-primitive
/// normalization absorbed into the coefficients. Gaussian-style BSE
/// downloads (cc-pVDZ, aug-cc-pVTZ) already ship coefs with the
/// contraction normalized to unit overlap. Turbomole-style BSE downloads
/// (def2-SVP, def2-TZVP, def2-QZVP, …) ship raw coefs and expect the
/// reader to renormalize.
///
/// libint2 internally normalizes contracted shells to unit overlap regardless,
/// so its analytic-integral path is unaffected. ferric's DFT-grid path
/// (`radial`) only applies the per-primitive `N(α, l)` factor, not the
/// contraction-level renormalization — so this function brings both paths
/// into agreement at load time. After this rescaling libint's internal
/// renorm becomes idempotent (S is already 1).
///
/// Self-overlap formula for a contracted shell of angular momentum `l`:
/// ```text
///   S = Σ_p Σ_q c_p c_q · ( 2 √(α_p α_q) / (α_p + α_q) )^(l + 3/2)
/// ```
/// then divide every c_p by √S.
fn renormalize_contraction(exps: &[f64], coefs: &mut [f64], l: i32) {
    let lf = l as f64;
    let mut s = 0.0_f64;
    for (a, ca) in exps.iter().zip(coefs.iter()) {
        for (b, cb) in exps.iter().zip(coefs.iter()) {
            let prim_ov = (2.0 * (a * b).sqrt() / (a + b)).powf(lf + 1.5);
            s += ca * cb * prim_ov;
        }
    }
    if s > 0.0 {
        let scale = 1.0 / s.sqrt();
        for c in coefs.iter_mut() { *c *= scale; }
    }
}

fn parse_float_list(ss: &[String]) -> Result<Vec<f64>, FerricError> {
    ss.iter().map(|s| s.parse::<f64>().map_err(|e| FerricError::Basis(format!("bad float {s:?}: {e}")))).collect()
}

// --- Gaussian-94 parser ---

/// Load a basis set from a Gaussian-94 format file.
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
        // G94 convention: spherical for L>=2
        let pure = l >= 2;
        if is_sp {
            let mut c_s = coefs[0].clone();
            let mut c_p = coefs[1].clone();
            renormalize_contraction(&exps, &mut c_s, 0);
            renormalize_contraction(&exps, &mut c_p, 1);
            shells.entry(z).or_default().push(Shell { l: 0, pure: false, exponents: exps.clone(), coefficients: c_s });
            shells.entry(z).or_default().push(Shell { l: 1, pure: false, exponents: exps, coefficients: c_p });
        } else {
            let mut c0 = coefs[0].clone();
            renormalize_contraction(&exps, &mut c0, l);
            shells.entry(z).or_default().push(Shell { l, pure, exponents: exps, coefficients: c0 });
        }
    }
    Ok(BasisSet { name: name.to_string(), shells })
}

// --- Bundled basis sets ---

fn canonical_name(name: &str) -> String { name.to_ascii_lowercase() }

/// Load a bundled basis set by name (case-insensitive).
///
/// Automatically supports all `.json` files in the bundled folder.
///
/// # Examples
///
/// ```
/// use ferric_core::basis::bundled;
///
/// let bs = bundled("STO-3G").unwrap();
/// assert!(bs.for_element(1).is_some());
/// ```
pub fn bundled(name: &str) -> Result<BasisSet, FerricError> {
    let cn = canonical_name(name);
    let json = match cn.as_str() {
        "sto-3g" => include_str!("basis/bundled/sto-3g.json"),
        "6-31g" => include_str!("basis/bundled/6-31g.json"),
        "cc-pvdz" => include_str!("basis/bundled/cc-pvdz.json"),
        "def2-svp" => include_str!("basis/bundled/def2-svp.json"),
        "cc-pvdz-ri" => include_str!("basis/bundled/cc-pvdz-ri.json"),
        "cc-pvdz-f12" => include_str!("basis/bundled/cc-pvdz-f12.json"),
        "cc-pvdz-f12-optri" => include_str!("basis/bundled/cc-pvdz-f12-optri.json"),
        "def2-svp-rifit" => include_str!("basis/bundled/def2-svp-rifit.json"),
        "def2-tzvp" => include_str!("basis/bundled/def2-tzvp.json"),
        "def2-tzvp-rifit" => include_str!("basis/bundled/def2-tzvp-rifit.json"),
        "def2-tzvpp-rifit" => include_str!("basis/bundled/def2-tzvpp-rifit.json"),
        "def2-qzvp" => include_str!("basis/bundled/def2-qzvp.json"),
        "def2-qzvp-rifit" => include_str!("basis/bundled/def2-qzvp-rifit.json"),
        "def2-qzvpp-rifit" => include_str!("basis/bundled/def2-qzvpp-rifit.json"),
        "aug-cc-pvdz" => include_str!("basis/bundled/aug-cc-pvdz.json"),
        "aug-cc-pvdz-rifit" => include_str!("basis/bundled/aug-cc-pvdz-rifit.json"),
        "aug-cc-pvtz" => include_str!("basis/bundled/aug-cc-pvtz.json"),
        "aug-cc-pvtz-rifit" => include_str!("basis/bundled/aug-cc-pvtz-rifit.json"),
        "def2-universal-jkfit" => include_str!("basis/bundled/def2-universal-jkfit.json"),
        _ => return Err(FerricError::Basis(format!("unknown bundled basis: {name}"))),
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
        assert!(!h_shells[0].pure);
        let c_shells = bs.for_element(6).unwrap();
        assert!(c_shells.len() >= 2);
    }

    #[test]
    fn test_bundled_ccpvdz_f12_loads() {
        let obs = bundled("cc-pVDZ-F12").unwrap();
        // F12 orbital set: H has s,p (>= cc-pVDZ); O reaches d.
        assert!(obs.for_element(1).is_some(), "H missing from cc-pVDZ-F12");
        let o = obs.for_element(8).unwrap();
        assert!(o.iter().any(|s| s.l == 2), "O should have d shells");

        // OptRI (CABS aux): larger, reaches g (L=4).
        let ri = bundled("cc-pVDZ-F12-OptRI").unwrap();
        let o_ri = ri.for_element(8).unwrap();
        let max_l = o_ri.iter().map(|s| s.l).max().unwrap();
        assert!(max_l >= 3, "OptRI O should reach >= f, got max L={max_l}");
    }

    #[test]
    fn test_bundled_def2svp_has_d_shells() {
        let bs = bundled("def2-svp").unwrap();
        let c_shells = bs.for_element(6).unwrap();
        let d_shell = c_shells.iter().find(|s| s.l == 2);
        assert!(d_shell.is_some(), "def2-SVP carbon should have d shells");
        assert!(d_shell.unwrap().pure, "def2-SVP d shells should be spherical");
    }

    #[test]
    fn test_bundled_ccpvdz_spherical() {
        let bs = bundled("cc-pvdz").unwrap();
        let c_shells = bs.for_element(6).unwrap();
        for sh in c_shells {
            if sh.l >= 2 {
                assert!(sh.pure, "cc-pVDZ L={} shell should be spherical", sh.l);
            } else {
                assert!(!sh.pure, "cc-pVDZ L={} shell should be Cartesian", sh.l);
            }
        }
    }

    #[test]
    fn test_bundled_sto3g_all_cartesian() {
        let bs = bundled("sto-3g").unwrap();
        for shells in bs.shells.values() {
            for sh in shells {
                assert!(!sh.pure, "STO-3G should be all Cartesian");
            }
        }
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
    fn test_num_functions_cartesian() {
        assert_eq!(num_functions(0, false), 1);  // s
        assert_eq!(num_functions(1, false), 3);  // p
        assert_eq!(num_functions(2, false), 6);  // 6d
        assert_eq!(num_functions(3, false), 10); // 10f
    }

    #[test]
    fn test_num_functions_spherical() {
        assert_eq!(num_functions(0, true), 1);  // s
        assert_eq!(num_functions(1, true), 3);  // p
        assert_eq!(num_functions(2, true), 5);  // 5d
        assert_eq!(num_functions(3, true), 7);  // 7f
    }

    #[test]
    fn test_bundled_ccpvdz_ri() {
        let bs = bundled("cc-pvdz-ri").unwrap();
        let o_shells = bs.for_element(8).unwrap();
        assert!(!o_shells.is_empty(), "cc-pVDZ-RI should have oxygen shells");
        let max_l = o_shells.iter().map(|s| s.l).max().unwrap();
        assert!(max_l >= 3, "cc-pVDZ-RI oxygen should have at least f functions, got max_l={max_l}");
    }

    #[test]
    fn test_bundled_def2svp_rifit() {
        let bs = bundled("def2-svp-rifit").unwrap();
        let h_shells = bs.for_element(1).unwrap();
        assert!(!h_shells.is_empty(), "def2-SVP-RIFIT should have hydrogen shells");
    }
}
