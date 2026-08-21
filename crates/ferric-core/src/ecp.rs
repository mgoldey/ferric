//! Effective core potential (ECP) data: structures, BSE-JSON parser, bundled library.
//!
//! Scalar-relativistic ECPs (def2-ECP, cc-pVnZ-PP) replace the chemically inert
//! core electrons of heavy atoms with an effective potential. The BSE-JSON format
//! carries two pieces per element:
//!
//! - `ecp_electrons`: the number of core electrons replaced (`n_core`).
//! - `ecp_potentials`: the semilocal expansion. Each entry is one angular-momentum
//!   channel `l`, expanded as a sum of terms
//!   `U_l(r) = Σ_k d_k · r^(n_k − 2) · exp(−ζ_k r²)`.
//!   (BSE stores `r_exponents` as the literal power `n_k` on `r`; the conventional
//!   ECP radial form carries `r^(n_k-2)`, but we store `n_k` verbatim and leave the
//!   convention to the consumer — libecpint and PySCF both take the raw `n`.)
//!
//! The channel with the **largest** angular momentum is the local (`U_L`) term;
//! the lower channels are the projected semilocal corrections. We preserve the
//! per-channel `angular_momentum` tag so the integral shim can map them to
//! libecpint without re-deriving which channel is local.

use crate::basis_util::{parse_float_list, canonical_name};
use crate::FerricError;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

/// One radial term of an ECP angular-momentum channel:
/// `d · r^(n) · exp(−ζ r²)`.
///
/// `coef` = `d` (contraction coefficient), `r_exp` = `n` (power on r as stored by
/// BSE, i.e. the literal `r_exponents` value), `gexp` = `ζ` (Gaussian exponent).
#[derive(Debug, Clone, PartialEq)]
pub struct EcpTerm {
    pub coef: f64,
    pub r_exp: i32,
    pub gexp: f64,
}

/// One angular-momentum channel of an ECP, e.g. the s, p, d, ... projector or the
/// local term (highest l).
#[derive(Debug, Clone, PartialEq)]
pub struct EcpShell {
    /// Angular momentum `l` of this channel (0=s, 1=p, ...). The channel with the
    /// maximum `angular_momentum` across an [`EcpDef`] is the local term.
    pub angular_momentum: i32,
    pub terms: Vec<EcpTerm>,
}

/// A complete ECP definition for one element.
#[derive(Debug, Clone, PartialEq)]
pub struct EcpDef {
    /// Number of core electrons replaced by this ECP.
    pub n_core: i32,
    /// Per-angular-momentum channels (local + semilocal projectors).
    pub shells: Vec<EcpShell>,
}

impl EcpDef {
    /// The largest angular momentum among the channels — the local term `l`.
    pub fn max_angular_momentum(&self) -> i32 {
        self.shells.iter().map(|s| s.angular_momentum).max().unwrap_or(0)
    }
}

/// A named collection of ECP definitions keyed by atomic number `Z`.
#[derive(Debug, Clone, Default)]
pub struct EcpSet {
    pub name: String,
    pub defs: HashMap<i32, EcpDef>,
}

impl EcpSet {
    /// Return the ECP definition for element `z`, or `None` if absent.
    pub fn for_element(&self, z: i32) -> Option<&EcpDef> {
        self.defs.get(&z)
    }
}

// --- BSE-JSON parser ---

#[derive(Deserialize)]
struct BseEcpFile {
    name: Option<String>,
    elements: HashMap<String, BseEcpElement>,
}

#[derive(Deserialize)]
struct BseEcpElement {
    #[serde(default)]
    ecp_electrons: Option<i32>,
    #[serde(default)]
    ecp_potentials: Option<Vec<BseEcpPotential>>,
}

#[derive(Deserialize)]
struct BseEcpPotential {
    angular_momentum: Vec<i32>,
    r_exponents: Vec<i32>,
    gaussian_exponents: Vec<String>,
    coefficients: Vec<Vec<String>>,
}

/// Load an ECP set from a Basis Set Exchange JSON file on disk.
pub fn load_ecp_json(path: &str) -> Result<EcpSet, FerricError> {
    let text = fs::read_to_string(path)
        .map_err(|e| FerricError::General(format!("cannot read ECP file {path:?}: {e}")))?;
    parse_ecp_json(&text, path)
}

/// Parse a BSE-format ECP JSON string. Elements without ECP data are skipped.
pub fn parse_ecp_json(text: &str, name: &str) -> Result<EcpSet, FerricError> {
    let bf: BseEcpFile =
        serde_json::from_str(text).map_err(|e| FerricError::Basis(format!("ECP JSON: {e}")))?;
    let set_name = bf.name.unwrap_or_else(|| name.to_string());
    let mut defs: HashMap<i32, EcpDef> = HashMap::new();
    for (z_str, elem) in &bf.elements {
        // An element entry may legitimately carry only basis shells (no ECP) when
        // the same file mixes both — skip those.
        let (n_core, pots) = match (elem.ecp_electrons, elem.ecp_potentials.as_ref()) {
            (Some(n), Some(p)) => (n, p),
            _ => continue,
        };
        let z: i32 = z_str
            .parse()
            .map_err(|e| FerricError::Basis(format!("bad element key {z_str:?}: {e}")))?;
        let mut shells = Vec::with_capacity(pots.len());
        for pot in pots {
            if pot.angular_momentum.len() != 1 {
                return Err(FerricError::Basis(format!(
                    "ECP channel for Z={z} has {} angular momenta, expected 1",
                    pot.angular_momentum.len()
                )));
            }
            let l = pot.angular_momentum[0];
            let gexps = parse_float_list(&pot.gaussian_exponents)?;
            if pot.coefficients.len() != 1 {
                return Err(FerricError::Basis(format!(
                    "ECP channel l={l} for Z={z} has {} coefficient columns, expected 1",
                    pot.coefficients.len()
                )));
            }
            let coefs = parse_float_list(&pot.coefficients[0])?;
            let n = gexps.len();
            if pot.r_exponents.len() != n || coefs.len() != n {
                return Err(FerricError::Basis(format!(
                    "ECP channel l={l} for Z={z}: ragged term lists ({} r_exp, {} gexp, {} coef)",
                    pot.r_exponents.len(),
                    n,
                    coefs.len()
                )));
            }
            let terms = (0..n)
                .map(|k| EcpTerm {
                    coef: coefs[k],
                    r_exp: pot.r_exponents[k],
                    gexp: gexps[k],
                })
                .collect();
            shells.push(EcpShell { angular_momentum: l, terms });
        }
        defs.insert(z, EcpDef { n_core, shells });
    }
    Ok(EcpSet { name: set_name, defs })
}

// --- Bundled ECP sets ---

/// Load a bundled ECP set by name (case-insensitive).
///
/// # Examples
///
/// ```
/// use ferric_core::ecp::bundled;
///
/// let ecp = bundled("def2-ECP").unwrap();
/// let iodine = ecp.for_element(53).unwrap();
/// assert_eq!(iodine.n_core, 28);
/// ```
pub fn bundled(name: &str) -> Result<EcpSet, FerricError> {
    let cn = canonical_name(name);
    let json = match cn.as_str() {
        "def2-ecp" => include_str!("basis/bundled/def2-ecp.json"),
        _ => return Err(FerricError::Basis(format!("unknown bundled ECP: {name}"))),
    };
    let mut ecp = parse_ecp_json(json, &cn)?;
    ecp.name = cn;
    Ok(ecp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundled_def2ecp_iodine_ncore() {
        let ecp = bundled("def2-ECP").unwrap();
        let i = ecp.for_element(53).expect("iodine ECP present");
        // def2-ECP replaces the [Kr]4d10 = 28-electron core of iodine.
        assert_eq!(i.n_core, 28);
    }

    #[test]
    fn test_bundled_def2ecp_iodine_channels() {
        let ecp = bundled("def2-ECP").unwrap();
        let i = ecp.for_element(53).unwrap();
        // def2-ECP for I has 4 channels: l = 3 (local), 0, 1, 2.
        assert_eq!(i.shells.len(), 4);
        let mut ls: Vec<i32> = i.shells.iter().map(|s| s.angular_momentum).collect();
        ls.sort();
        assert_eq!(ls, vec![0, 1, 2, 3]);
        // The local channel is l = 3 (the maximum).
        assert_eq!(i.max_angular_momentum(), 3);
    }

    #[test]
    fn test_bundled_def2ecp_iodine_local_terms() {
        let ecp = bundled("def2-ECP").unwrap();
        let i = ecp.for_element(53).unwrap();
        // Local channel l=3: 4 terms, all r^2, exact coefs/exponents from the BSE JSON.
        let local = i.shells.iter().find(|s| s.angular_momentum == 3).unwrap();
        assert_eq!(local.terms.len(), 4);
        for t in &local.terms {
            assert_eq!(t.r_exp, 2);
        }
        // First term: d = -21.84204, ζ = 19.458609.
        assert!((local.terms[0].coef - (-21.842_040)).abs() < 1e-9);
        assert!((local.terms[0].gexp - 19.458_609).abs() < 1e-9);
        // Last term: d = -0.320804, ζ = 4.884315.
        assert!((local.terms[3].coef - (-0.320_804)).abs() < 1e-9);
        assert!((local.terms[3].gexp - 4.884_315).abs() < 1e-9);
    }

    #[test]
    fn test_bundled_def2ecp_iodine_s_channel() {
        let ecp = bundled("def2-ECP").unwrap();
        let i = ecp.for_element(53).unwrap();
        // s channel (l=0): 7 terms; first d=49.994293, ζ=40.015835.
        let s = i.shells.iter().find(|s| s.angular_momentum == 0).unwrap();
        assert_eq!(s.terms.len(), 7);
        assert!((s.terms[0].coef - 49.994_293).abs() < 1e-9);
        assert!((s.terms[0].gexp - 40.015_835).abs() < 1e-9);
        assert_eq!(s.terms[0].r_exp, 2);
    }

    #[test]
    fn test_parse_skips_elements_without_ecp() {
        // An element block with only basis shells (no ecp_* keys) must be skipped,
        // not error — back-compat for mixed basis+ECP files.
        let json = r#"{
            "name": "test",
            "elements": {
                "1":  { "electron_shells": [] },
                "53": { "ecp_electrons": 28, "ecp_potentials": [
                    { "angular_momentum": [0], "r_exponents": [2],
                      "gaussian_exponents": ["1.5"], "coefficients": [["3.0"]] } ] }
            }
        }"#;
        let set = parse_ecp_json(json, "test").unwrap();
        assert!(set.for_element(1).is_none());
        let i = set.for_element(53).unwrap();
        assert_eq!(i.n_core, 28);
        assert_eq!(i.shells[0].terms[0], EcpTerm { coef: 3.0, r_exp: 2, gexp: 1.5 });
    }

    #[test]
    fn test_unknown_bundled_errors() {
        assert!(bundled("no-such-ecp").is_err());
    }
}
