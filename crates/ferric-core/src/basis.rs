//! Gaussian basis sets: data structures, parsers, and bundled library.
//!
//! Supports BSE-JSON and Gaussian-94 input formats, plus a set of bundled
//! basis sets compiled into the binary via `include_str!`.

use crate::basis_util::{canonical_name, parse_float_list};
use crate::ecp::{EcpDef, EcpShell, EcpTerm};
use crate::elements::symbol_to_z;
use crate::FerricError;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::str::FromStr;

/// A single contracted Gaussian shell.
///
/// `l` is the angular momentum quantum number (0=s, 1=p, 2=d, ...).
/// When `pure` is true, spherical harmonics are used (2l+1 functions);
/// otherwise Cartesian Gaussians are used ((l+1)(l+2)/2 functions).
#[derive(Debug, Clone, PartialEq)]
pub struct Shell {
    pub l: i32,
    pub pure: bool,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}

/// A named basis set mapping atomic numbers to their shells.
///
/// # Examples
///
/// ```
/// use ferric_core::basis::BasisSet;
///
/// // Look up a bundled basis set by name:
/// let bs: BasisSet = "cc-pvdz".parse().unwrap();
/// assert!(bs.for_element(1).is_some()); // hydrogen
///
/// // Or via the explicit function:
/// let bs = ferric_core::basis::bundled("sto-3g").unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BasisSet {
    pub name: String,
    /// Map from atomic number Z to the list of shells for that element.
    pub shells: HashMap<i32, Vec<Shell>>,
    /// Optional ECP definitions keyed by Z. Populated only when the source JSON
    /// carries `ecp_electrons`/`ecp_potentials` blocks (e.g. cc-pVnZ-PP). Empty
    /// for ECP-free basis sets — existing loads are unaffected.
    pub ecps: HashMap<i32, EcpDef>,
}

impl BasisSet {
    /// Return the shells for element with atomic number `z`, or `None` if absent.
    pub fn for_element(&self, z: i32) -> Option<&[Shell]> {
        self.shells.get(&z).map(|v| v.as_slice())
    }

    /// Return the ECP definition for element `z`, or `None` if this basis carries
    /// no ECP for it.
    pub fn ecp_for_element(&self, z: i32) -> Option<&EcpDef> {
        self.ecps.get(&z)
    }

    /// Atomic numbers this basis has shells for, ascending.
    pub fn elements(&self) -> Vec<i32> {
        let mut zs: Vec<i32> = self.shells.keys().copied().collect();
        zs.sort_unstable();
        zs
    }

    /// The covered elements as compact symbol ranges, e.g. `"H-Ar, Ca-Kr"`.
    ///
    /// ```
    /// let bs = ferric_core::basis::bundled("cc-pvtz").unwrap();
    /// assert_eq!(bs.coverage_summary(), "H-Ar, Ca-Kr"); // no K upstream
    /// ```
    pub fn coverage_summary(&self) -> String {
        let zs = self.elements();
        let sym = |z: i32| {
            crate::elements::z_to_symbol(z).map_or_else(|| format!("Z{z}"), str::to_string)
        };
        let mut parts = Vec::new();
        let mut i = 0;
        while i < zs.len() {
            let mut j = i;
            while j + 1 < zs.len() && zs[j + 1] == zs[j] + 1 {
                j += 1;
            }
            parts.push(if i == j {
                sym(zs[i])
            } else {
                format!("{}-{}", sym(zs[i]), sym(zs[j]))
            });
            i = j + 1;
        }
        parts.join(", ")
    }

    /// The error for an atom whose element this basis does not cover. Names
    /// the element, the basis and what the basis does cover, so the user can
    /// pick another set instead of guessing.
    pub fn missing_element_error(&self, z: i32) -> FerricError {
        let sym = crate::elements::z_to_symbol(z).unwrap_or("?");
        FerricError::Basis(format!(
            "basis {:?} has no shells for {sym} (Z={z}); it covers {}",
            self.name,
            self.coverage_summary()
        ))
    }
}

/// Number of basis functions for a shell with angular momentum `l`.
///
/// Spherical: 2l+1. Cartesian: (l+1)(l+2)/2.
pub fn num_functions(l: i32, pure: bool) -> usize {
    if pure {
        (2 * l + 1) as usize
    } else {
        ((l + 1) * (l + 2) / 2) as usize
    }
}

// --- BSE-JSON parser ---

#[derive(Deserialize)]
struct BseFile {
    name: Option<String>,
    elements: HashMap<String, BseElement>,
}

#[derive(Deserialize)]
struct BseElement {
    #[serde(default)]
    electron_shells: Vec<BseShell>,
    /// Number of core electrons replaced by an ECP (present only in *-PP / def2-ECP files).
    #[serde(default)]
    ecp_electrons: Option<i32>,
    /// Semilocal ECP expansion (present only when `ecp_electrons` is).
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
    let text = fs::read_to_string(path)
        .map_err(|e| FerricError::General(format!("cannot read basis file {path:?}: {e}")))?;
    parse_bse_json(&text, path)
}

fn parse_bse_json(text: &str, name: &str) -> Result<BasisSet, FerricError> {
    let bf: BseFile =
        serde_json::from_str(text).map_err(|e| FerricError::Basis(format!("BSE JSON: {e}")))?;
    let mut shells: HashMap<i32, Vec<Shell>> = HashMap::new();
    let mut ecps: HashMap<i32, EcpDef> = HashMap::new();
    let bs_name = bf.name.unwrap_or_else(|| name.to_string());
    for (z_str, elem) in &bf.elements {
        let z: i32 = z_str
            .parse()
            .map_err(|e| FerricError::Basis(format!("bad element key {z_str:?}: {e}")))?;
        if let (Some(n_core), Some(pots)) = (elem.ecp_electrons, elem.ecp_potentials.as_ref()) {
            ecps.insert(z, parse_ecp_block(z, n_core, pots)?);
        }
        for sh in &elem.electron_shells {
            convert_bse_shell(sh, shells.entry(z).or_default())?;
        }
    }
    Ok(BasisSet {
        name: bs_name,
        shells,
        ecps,
    })
}

/// Convert one BSE shell into ferric [`Shell`]s, appending to `out`.
///
/// A single-l shell yields one [`Shell`] per coefficient column (general
/// contraction); a multi-l shell (Pople `SP`) yields one per angular momentum,
/// column k belonging to `angular_momentum[k]`. Every contraction is
/// renormalized to unit self-overlap. `gto_spherical` makes l >= 2 pure;
/// "gto", `gto_cartesian` or no type is Cartesian, and l < 2 is never pure.
fn convert_bse_shell(sh: &BseShell, out: &mut Vec<Shell>) -> Result<(), FerricError> {
    let exps = parse_float_list(&sh.exponents)?;
    let pure = sh.function_type.as_deref() == Some("gto_spherical");
    let multi_l = sh.angular_momentum.len() > 1;
    let ncol = if multi_l {
        sh.angular_momentum.len()
    } else {
        sh.coefficients.len()
    };
    for k in 0..ncol {
        let l = sh.angular_momentum[if multi_l { k } else { 0 }];
        let col = sh
            .coefficients
            .get(k)
            .ok_or_else(|| FerricError::Basis(format!("shell missing coefficient column {k}")))?;
        let mut coefs = parse_float_list(col)?;
        if coefs.len() != exps.len() {
            return Err(FerricError::Basis(format!(
                "{} coeffs vs {} exps",
                coefs.len(),
                exps.len()
            )));
        }
        renormalize_contraction(&exps, &mut coefs, l);
        out.push(Shell {
            l,
            pure: pure && l >= 2,
            exponents: exps.clone(),
            coefficients: coefs,
        });
    }
    Ok(())
}

/// Parse a single element's `ecp_potentials` block (BSE-JSON) into an [`EcpDef`].
fn parse_ecp_block(z: i32, n_core: i32, pots: &[BseEcpPotential]) -> Result<EcpDef, FerricError> {
    let mut ecp_shells = Vec::with_capacity(pots.len());
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
                "ECP channel l={l} for Z={z}: ragged term lists"
            )));
        }
        let terms = (0..n)
            .map(|k| EcpTerm {
                coef: coefs[k],
                r_exp: pot.r_exponents[k],
                gexp: gexps[k],
            })
            .collect();
        ecp_shells.push(EcpShell {
            angular_momentum: l,
            terms,
        });
    }
    Ok(EcpDef {
        n_core,
        shells: ecp_shells,
    })
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
        for c in coefs.iter_mut() {
            *c *= scale;
        }
    }
}

// --- Gaussian-94 parser ---

/// Load a basis set from a Gaussian-94 format file.
pub fn load_g94(path: &str) -> Result<BasisSet, FerricError> {
    let text = fs::read_to_string(path)
        .map_err(|e| FerricError::General(format!("cannot read basis file {path:?}: {e}")))?;
    parse_g94(&text, path)
}

fn parse_g94(text: &str, name: &str) -> Result<BasisSet, FerricError> {
    let mut shells: HashMap<i32, Vec<Shell>> = HashMap::new();
    let mut cur_z: Option<i32> = None;
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('!') {
            continue;
        }
        if line == "****" {
            cur_z = None;
            continue;
        }
        if cur_z.is_none() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() >= 2 {
                if let Some(z) = symbol_to_z(fields[0]) {
                    cur_z = Some(z);
                }
            }
            continue;
        }
        let z = cur_z.unwrap();
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            continue;
        }
        let shell_type = fields[0].to_uppercase();
        let nprim: usize = fields[1]
            .parse()
            .map_err(|e| FerricError::Basis(format!("bad nprim: {e}")))?;
        let is_sp = shell_type == "SP";
        let ncol = if is_sp { 2 } else { 1 };
        let mut exps = Vec::with_capacity(nprim);
        let mut coefs: Vec<Vec<f64>> = (0..ncol).map(|_| Vec::with_capacity(nprim)).collect();
        for _ in 0..nprim {
            let pline = lines
                .next()
                .ok_or_else(|| FerricError::Basis("unexpected EOF in shell".into()))?;
            let pf: Vec<&str> = pline.split_whitespace().collect();
            if pf.len() < 1 + ncol {
                return Err(FerricError::Basis(format!(
                    "primitive line has {} fields, want {}",
                    pf.len(),
                    1 + ncol
                )));
            }
            exps.push(
                pf[0]
                    .parse::<f64>()
                    .map_err(|e| FerricError::Basis(format!("bad exponent: {e}")))?,
            );
            for c in 0..ncol {
                coefs[c].push(
                    pf[1 + c]
                        .parse::<f64>()
                        .map_err(|e| FerricError::Basis(format!("bad coefficient: {e}")))?,
                );
            }
        }
        let l = match shell_type.as_str() {
            "S" | "SP" => 0,
            "P" => 1,
            "D" => 2,
            "F" => 3,
            "G" => 4,
            other => return Err(FerricError::Basis(format!("unknown shell type {other:?}"))),
        };
        // G94 convention: spherical for L>=2
        let pure = l >= 2;
        if is_sp {
            let mut c_s = coefs[0].clone();
            let mut c_p = coefs[1].clone();
            renormalize_contraction(&exps, &mut c_s, 0);
            renormalize_contraction(&exps, &mut c_p, 1);
            shells.entry(z).or_default().push(Shell {
                l: 0,
                pure: false,
                exponents: exps.clone(),
                coefficients: c_s,
            });
            shells.entry(z).or_default().push(Shell {
                l: 1,
                pure: false,
                exponents: exps,
                coefficients: c_p,
            });
        } else {
            let mut c0 = coefs[0].clone();
            renormalize_contraction(&exps, &mut c0, l);
            shells.entry(z).or_default().push(Shell {
                l,
                pure,
                exponents: exps,
                coefficients: c0,
            });
        }
    }
    Ok(BasisSet {
        name: name.to_string(),
        shells,
        ecps: HashMap::new(),
    })
}

// --- Bundled basis sets ---

#[cfg(test)]
mod frozen_fingerprints;
mod packed;

/// Every loadable bundled basis set: `(file stem in basis/bundled/, names)`.
///
/// The first name is the primary one ([`bundled_names`] lists it); the rest
/// are aliases. Names are matched after ASCII-lowercasing and trimming, so
/// `"6-31G*"`, `"6-31g(d)"` and `" 6-31G(D) "` are the same set. Aliases
/// follow the spellings other codes use: `(d,p)` for the Pople stars, `-ri`
/// and ORCA's `/c` for RIFIT sets, `/jk` for JKFIT sets, and PySCF's
/// `def2-<x>-jkfit` for the one def2 universal JK set.
///
/// Every JSON in `basis/bundled/` must be either a stem here or on
/// `UNREGISTERED_ALLOWLIST` (enforced by a test). Data, element coverage and
/// the BSE version of every file are recorded in `basis/provenance.json`,
/// written by `scripts/fetch_bse_bases.py`.
const REGISTRY: &[(&str, &[&str])] = &[
    // Pople, in Gaussian's d convention: the 6-31G family is Cartesian (6D),
    // as BSE marks it; the 6-311G family is spherical (5D) on every element,
    // which overrides BSE's Cartesian Na-Ar (see `bundled`).
    ("sto-3g", &["sto-3g"]),
    ("6-31g", &["6-31g"]),
    ("6-31g_d", &["6-31g*", "6-31g(d)"]),
    ("6-31g_d_p", &["6-31g**", "6-31g(d,p)"]),
    ("6-31+g_d", &["6-31+g*", "6-31+g(d)"]),
    ("6-31+g_d_p", &["6-31+g**", "6-31+g(d,p)"]),
    ("6-31++g_d_p", &["6-31++g**", "6-31++g(d,p)"]),
    ("6-311g_d_p", &["6-311g**", "6-311g(d,p)"]),
    ("6-311++g_d_p", &["6-311++g**", "6-311++g(d,p)"]),
    // Dunning, all-electron
    ("cc-pvdz", &["cc-pvdz"]),
    ("cc-pvtz", &["cc-pvtz"]),
    ("cc-pvqz", &["cc-pvqz"]),
    ("aug-cc-pvdz", &["aug-cc-pvdz"]),
    ("aug-cc-pvtz", &["aug-cc-pvtz"]),
    ("aug-cc-pvqz", &["aug-cc-pvqz"]),
    ("cc-pvdz-f12", &["cc-pvdz-f12"]),
    ("cc-pvdz-f12-optri", &["cc-pvdz-f12-optri"]),
    // Dunning, small-core pseudopotential (ECP inline) for Z >= 29; the
    // light elements are the matching all-electron set (Psi4 convention).
    ("cc-pvdz-pp", &["cc-pvdz-pp"]),
    ("cc-pvtz-pp", &["cc-pvtz-pp"]),
    ("cc-pvqz-pp", &["cc-pvqz-pp"]),
    ("aug-cc-pvdz-pp", &["aug-cc-pvdz-pp"]),
    ("aug-cc-pvtz-pp", &["aug-cc-pvtz-pp"]),
    ("aug-cc-pvqz-pp", &["aug-cc-pvqz-pp"]),
    // Dunning RI fitting. `cc-pvdz-ri.json` IS the BSE set "cc-pVDZ-RIFIT".
    ("cc-pvdz-ri", &["cc-pvdz-rifit", "cc-pvdz-ri", "cc-pvdz/c"]),
    (
        "cc-pvtz-rifit",
        &["cc-pvtz-rifit", "cc-pvtz-ri", "cc-pvtz/c"],
    ),
    (
        "cc-pvqz-rifit",
        &["cc-pvqz-rifit", "cc-pvqz-ri", "cc-pvqz/c"],
    ),
    (
        "aug-cc-pvdz-rifit",
        &["aug-cc-pvdz-rifit", "aug-cc-pvdz-ri", "aug-cc-pvdz/c"],
    ),
    (
        "aug-cc-pvtz-rifit",
        &["aug-cc-pvtz-rifit", "aug-cc-pvtz-ri", "aug-cc-pvtz/c"],
    ),
    (
        "aug-cc-pvqz-rifit",
        &["aug-cc-pvqz-rifit", "aug-cc-pvqz-ri", "aug-cc-pvqz/c"],
    ),
    (
        "cc-pvdz-pp-rifit",
        &["cc-pvdz-pp-rifit", "cc-pvdz-pp-ri", "cc-pvdz-pp/c"],
    ),
    (
        "cc-pvtz-pp-rifit",
        &["cc-pvtz-pp-rifit", "cc-pvtz-pp-ri", "cc-pvtz-pp/c"],
    ),
    (
        "cc-pvqz-pp-rifit",
        &["cc-pvqz-pp-rifit", "cc-pvqz-pp-ri", "cc-pvqz-pp/c"],
    ),
    (
        "aug-cc-pvdz-pp-rifit",
        &[
            "aug-cc-pvdz-pp-rifit",
            "aug-cc-pvdz-pp-ri",
            "aug-cc-pvdz-pp/c",
        ],
    ),
    (
        "aug-cc-pvtz-pp-rifit",
        &[
            "aug-cc-pvtz-pp-rifit",
            "aug-cc-pvtz-pp-ri",
            "aug-cc-pvtz-pp/c",
        ],
    ),
    (
        "aug-cc-pvqz-pp-rifit",
        &[
            "aug-cc-pvqz-pp-rifit",
            "aug-cc-pvqz-pp-ri",
            "aug-cc-pvqz-pp/c",
        ],
    ),
    // Dunning JK fitting (Weigend). Coverage is only H, B-F, Al-Cl, Ga-Br.
    ("cc-pvtz-jkfit", &["cc-pvtz-jkfit", "cc-pvtz/jk"]),
    ("cc-pvqz-jkfit", &["cc-pvqz-jkfit", "cc-pvqz/jk"]),
    // Weigend's aug-cc-pVXZ-JKFIT. NOT in the Basis Set Exchange: imported
    // from Psi4 (pinned commit) and checked bit-identical to PySCF's copy by
    // scripts/import_psi4_jkfit.py. H, B-F, Al-Cl, Ga-Br plus Ne/Ar/Kr, whose
    // functions are EXTRAPOLATED, not published; the DZ set is the TZ set
    // truncated, not published either (see provenance.json).
    (
        "aug-cc-pvdz-jkfit",
        &["aug-cc-pvdz-jkfit", "aug-cc-pvdz/jk"],
    ),
    (
        "aug-cc-pvtz-jkfit",
        &["aug-cc-pvtz-jkfit", "aug-cc-pvtz/jk"],
    ),
    (
        "aug-cc-pvqz-jkfit",
        &["aug-cc-pvqz-jkfit", "aug-cc-pvqz/jk"],
    ),
    // Karlsruhe (ECP inline for Z >= 37)
    ("def2-svp", &["def2-svp"]),
    ("def2-svpd", &["def2-svpd"]),
    ("def2-tzvp", &["def2-tzvp"]),
    ("def2-tzvpp", &["def2-tzvpp"]),
    ("def2-tzvpd", &["def2-tzvpd"]),
    ("def2-tzvppd", &["def2-tzvppd"]),
    ("def2-qzvp", &["def2-qzvp"]),
    ("def2-qzvpp", &["def2-qzvpp"]),
    ("def2-qzvppd", &["def2-qzvppd"]),
    (
        "def2-svp-rifit",
        &["def2-svp-rifit", "def2-svp-ri", "def2-svp/c"],
    ),
    (
        "def2-svpd-rifit",
        &["def2-svpd-rifit", "def2-svpd-ri", "def2-svpd/c"],
    ),
    (
        "def2-tzvp-rifit",
        &["def2-tzvp-rifit", "def2-tzvp-ri", "def2-tzvp/c"],
    ),
    (
        "def2-tzvpp-rifit",
        &["def2-tzvpp-rifit", "def2-tzvpp-ri", "def2-tzvpp/c"],
    ),
    (
        "def2-tzvpd-rifit",
        &["def2-tzvpd-rifit", "def2-tzvpd-ri", "def2-tzvpd/c"],
    ),
    (
        "def2-tzvppd-rifit",
        &["def2-tzvppd-rifit", "def2-tzvppd-ri", "def2-tzvppd/c"],
    ),
    (
        "def2-qzvp-rifit",
        &["def2-qzvp-rifit", "def2-qzvp-ri", "def2-qzvp/c"],
    ),
    (
        "def2-qzvpp-rifit",
        &["def2-qzvpp-rifit", "def2-qzvpp-ri", "def2-qzvpp/c"],
    ),
    (
        "def2-qzvppd-rifit",
        &["def2-qzvppd-rifit", "def2-qzvppd-ri", "def2-qzvppd/c"],
    ),
    (
        "def2-universal-jkfit",
        &[
            "def2-universal-jkfit",
            "def2/jk",
            "def2-svp-jkfit",
            "def2-svpd-jkfit",
            "def2-tzvp-jkfit",
            "def2-tzvpp-jkfit",
            "def2-tzvpd-jkfit",
            "def2-tzvppd-jkfit",
            "def2-qzvp-jkfit",
            "def2-qzvpp-jkfit",
            "def2-qzvppd-jkfit",
        ],
    ),
    // Jensen segmented polarization-consistent
    ("pcseg-0", &["pcseg-0"]),
    ("pcseg-1", &["pcseg-1"]),
    ("pcseg-2", &["pcseg-2"]),
    ("aug-pcseg-1", &["aug-pcseg-1"]),
    ("aug-pcseg-2", &["aug-pcseg-2"]),
];

/// JSON files in `basis/bundled/` that are deliberately NOT loadable through
/// [`bundled`], each with the reason. A test fails if a file is neither here
/// nor in `REGISTRY`, so a set cannot silently ship unregistered again.
#[cfg(test)]
const UNREGISTERED_ALLOWLIST: &[(&str, &str)] = &[(
    "def2-ecp",
    "ECP-only data (no electron shells); loaded by ferric_core::ecp::bundled",
)];

/// Primary names of every bundled basis set, in registry order.
///
/// ```
/// let names = ferric_core::basis::bundled_names();
/// assert!(names.contains(&"def2-tzvp"));
/// ```
pub fn bundled_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|(_, names)| names[0]).collect()
}

/// File stem backing a bundled name (after canonicalization), if any.
fn registry_stem(canonical: &str) -> Option<&'static str> {
    REGISTRY
        .iter()
        .find(|(_, names)| names.contains(&canonical))
        .map(|(stem, _)| *stem)
}

/// Load a bundled basis set by name (case-insensitive).
///
/// The loadable names are [`bundled_names`] plus the aliases in the registry
/// (e.g. `"6-31g(d)"` for `"6-31g*"`, `"def2-tzvp/c"` for
/// `"def2-tzvp-rifit"`). The returned set's `name` is the canonicalized name
/// that was asked for. An unknown name is an error that lists the known ones.
///
/// The d-shell type follows Gaussian's convention for the Pople sets: the
/// 6-31G family (6-31G through 6-31++G**) is Cartesian 6D, as BSE types it;
/// the 6-311G family is spherical 5D on EVERY element, including Na-Ar,
/// which BSE types Cartesian (the one deliberate departure from BSE, recorded
/// in `basis/provenance.json`). Every other set is typed as BSE types it
/// (spherical for the Dunning, Karlsruhe and Jensen sets). PySCF applies one
/// `cart` flag to a whole molecule and defaults to spherical, so to match
/// ferric's 6-31G-family numbers build the PySCF molecule with `cart=True`
/// (water/6-31G*: about 1.4 mHa apart otherwise); the 6-311G family matches
/// PySCF's default.
///
/// Element coverage is whatever the source set has (see
/// `basis/provenance.json`); an element the set lacks is not patched, so a
/// calculation on it fails at basis setup, naming the element and the
/// covered range (see [`BasisSet::missing_element_error`]).
///
/// # Examples
///
/// ```
/// use ferric_core::basis::bundled;
///
/// let bs = bundled("STO-3G").unwrap();
/// assert!(bs.for_element(1).is_some());
/// let pople = bundled("6-31G(d,p)").unwrap();
/// assert_eq!(pople.name, "6-31g(d,p)");
/// ```
pub fn bundled(name: &str) -> Result<BasisSet, FerricError> {
    let cn = canonical_name(name);
    let stem = registry_stem(&cn).ok_or_else(|| {
        FerricError::Basis(format!(
            "unknown bundled basis: {name} (known: {})",
            bundled_names().join(", ")
        ))
    })?;
    packed::basis_set(stem, &cn)
}

/// Look up a bundled basis set by name.
///
/// ```
/// use ferric_core::basis::BasisSet;
///
/// let bs: BasisSet = "cc-pvdz".parse().unwrap();
/// assert!(bs.for_element(1).is_some());
/// ```
impl FromStr for BasisSet {
    type Err = FerricError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        bundled(s)
    }
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
    fn test_bundled_augccpvdz_pp_carries_ecp() {
        // aug-cc-pVDZ-PP bundles heavy atoms (Z >= 29, e.g. I/Xe/Ag) WITH their
        // inline ECP and light atoms (Z < 29, e.g. H/C/Al/Cl) from regular
        // aug-cc-pVDZ (no ECP). The GW100
        // ECP molecules (I2, CH3I, AlI3, Xe, Ag2, AgCl) need exactly this mix.
        let bs = bundled("aug-cc-pVDZ-PP").unwrap();
        // Heavy atoms present with ECP (28-electron core for all three).
        for &z in &[47, 53, 54] {
            assert!(bs.for_element(z).is_some(), "Z={z} orbital shells missing");
            let ecp = bs
                .ecp_for_element(z)
                .unwrap_or_else(|| panic!("Z={z} ECP missing"));
            assert_eq!(ecp.n_core, 28, "Z={z} should replace a 28-electron core");
        }
        // Light atoms present WITHOUT ECP.
        for &z in &[1, 6, 13, 17] {
            assert!(bs.for_element(z).is_some(), "light Z={z} shells missing");
            assert!(
                bs.ecp_for_element(z).is_none(),
                "light Z={z} must not carry an ECP"
            );
        }
    }

    #[test]
    fn test_bundled_augccpvtz_pp_carries_ecp_and_grows_over_dz() {
        // aug-cc-pVTZ-PP mirrors the DZ-PP merge: heavy (e.g. I/Xe/Ag) WITH
        // inline 28-core ECP, light (e.g. H/C/Al/Cl) from plain aug-cc-pVTZ
        // (no ECP). Each
        // element must have strictly MORE orbital shells than at DZ (TZ > DZ).
        let dz = bundled("aug-cc-pVDZ-PP").unwrap();
        let tz = bundled("aug-cc-pVTZ-PP").unwrap();
        // Heavy atoms present with 28-core ECP, and TZ shell count > DZ.
        for &z in &[47, 53, 54] {
            let tz_sh = tz
                .for_element(z)
                .unwrap_or_else(|| panic!("Z={z} orbital shells missing"));
            let ecp = tz
                .ecp_for_element(z)
                .unwrap_or_else(|| panic!("Z={z} ECP missing"));
            assert_eq!(ecp.n_core, 28, "Z={z} should replace a 28-electron core");
            let dz_n = dz.for_element(z).unwrap().len();
            assert!(
                tz_sh.len() > dz_n,
                "Z={z} TZ shells ({}) should exceed DZ ({dz_n})",
                tz_sh.len()
            );
        }
        // Light atoms present WITHOUT ECP, and TZ shell count > DZ.
        for &z in &[1, 6, 13, 17] {
            let tz_sh = tz
                .for_element(z)
                .unwrap_or_else(|| panic!("light Z={z} shells missing"));
            assert!(
                tz.ecp_for_element(z).is_none(),
                "light Z={z} must not carry an ECP"
            );
            let dz_n = dz.for_element(z).unwrap().len();
            assert!(
                tz_sh.len() > dz_n,
                "light Z={z} TZ shells ({}) should exceed DZ ({dz_n})",
                tz_sh.len()
            );
        }
    }

    #[test]
    fn test_bundled_ccpvtz_loads_and_grows_over_dz() {
        // cc-pVTZ and its RI-fit aux were present as JSON but unregistered,
        // so bundled() errored on them. This pins the registration: both must
        // load, and TZ must have strictly more shells than DZ for the light
        // elements the DZ/TZ extrapolation actually runs on (H, C, N, O).
        let dz = bundled("cc-pvdz").unwrap();
        let tz = bundled("cc-pvtz").unwrap();
        let tz_ri = bundled("cc-pvtz-rifit").unwrap();
        for &z in &[1, 6, 7, 8] {
            let dz_sh = dz
                .for_element(z)
                .unwrap_or_else(|| panic!("Z={z} missing from cc-pVDZ"));
            let tz_sh = tz
                .for_element(z)
                .unwrap_or_else(|| panic!("Z={z} missing from cc-pVTZ"));
            assert!(
                tz_sh.len() > dz_sh.len(),
                "Z={z} cc-pVTZ shells ({}) should exceed cc-pVDZ ({})",
                tz_sh.len(),
                dz_sh.len()
            );
            // The RI aux must cover every element the orbital basis does, or
            // an RI run would fail mid-calculation instead of at setup.
            assert!(
                tz_ri.for_element(z).is_some(),
                "Z={z} missing from cc-pvtz-rifit"
            );
        }
        // Documented upstream gaps, asserted so the doc comment cannot drift
        // from the data: K(19) is absent from cc-pVTZ, and K(19)+Ca(20) from
        // the rifit. If upstream data is ever refreshed this fails loudly.
        assert!(
            tz.for_element(19).is_none(),
            "cc-pVTZ unexpectedly has K(19)"
        );
        assert!(
            tz_ri.for_element(19).is_none(),
            "cc-pvtz-rifit unexpectedly has K(19)"
        );
        assert!(
            tz_ri.for_element(20).is_none(),
            "cc-pvtz-rifit unexpectedly has Ca(20)"
        );
    }

    #[test]
    fn test_bundled_augccpvqz_loads_and_grows_over_tz() {
        // aug-cc-pVQZ (BSE, ccRepo/Grant Hill, same source/date as aug-cc-pvtz)
        // must load for benzene/A24's elements (H, C, N, O) and have strictly
        // more orbital shells than aTZ (QZ > TZ), same pattern as the DZ->TZ
        // ECP-carrying test above.
        let tz = bundled("aug-cc-pvtz").unwrap();
        let qz = bundled("aug-cc-pvqz").unwrap();
        for &z in &[1, 6, 7, 8] {
            let tz_sh = tz
                .for_element(z)
                .unwrap_or_else(|| panic!("Z={z} missing from aTZ"));
            let qz_sh = qz
                .for_element(z)
                .unwrap_or_else(|| panic!("Z={z} missing from aQZ"));
            assert!(
                qz_sh.len() > tz_sh.len(),
                "Z={z} aQZ shells ({}) should exceed aTZ ({})",
                qz_sh.len(),
                tz_sh.len()
            );
        }
    }

    #[test]
    fn test_bundled_augccpvqz_rifit_loads() {
        // RI-fit aux (BSE, Turbomole 7.3) is missing Li(3)/K(19) relative to
        // aTZ's rifit -- a real upstream gap, not silently patched. Benzene
        // and A24's elements (H, C, N, O) are all present.
        let aux = bundled("aug-cc-pvqz-rifit").unwrap();
        for &z in &[1, 6, 7, 8] {
            assert!(aux.for_element(z).is_some(), "Z={z} missing from aQZ-rifit");
        }
    }

    #[test]
    fn test_bundled_def2svp_has_d_shells() {
        let bs = bundled("def2-svp").unwrap();
        let c_shells = bs.for_element(6).unwrap();
        let d_shell = c_shells.iter().find(|s| s.l == 2);
        assert!(d_shell.is_some(), "def2-SVP carbon should have d shells");
        assert!(
            d_shell.unwrap().pure,
            "def2-SVP d shells should be spherical"
        );
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
    fn test_bundled_sto3g_light_elements_all_cartesian() {
        // H-Ar have only s/p shells. The d shells BSE added for Sc-Xe are
        // `gto_spherical` and are loaded as such (BSE's per-shell type wins).
        let bs = bundled("sto-3g").unwrap();
        for z in 1..=18 {
            for sh in bs.for_element(z).unwrap() {
                assert!(!sh.pure, "STO-3G Z={z} should be all Cartesian");
            }
        }
        let fe_d = bs
            .for_element(26)
            .unwrap()
            .iter()
            .find(|s| s.l == 2)
            .unwrap();
        assert!(fe_d.pure, "BSE marks STO-3G Fe d spherical");
    }

    /// The Pople d-shell convention is Gaussian's, pinned per family and for
    /// every element: the 6-31G family is Cartesian 6D (as BSE types it), the
    /// 6-311G family is spherical 5D everywhere, including Na-Ar, which BSE
    /// types Cartesian (`SPHERICAL_OVERRIDE` in scripts/fetch_bse_bases.py).
    #[test]
    fn pople_d_shell_type_is_pinned_per_family() {
        let families: [(&[&str], bool); 2] = [
            (
                &[
                    "6-31g",
                    "6-31g*",
                    "6-31g**",
                    "6-31+g*",
                    "6-31+g**",
                    "6-31++g**",
                ],
                false,
            ),
            (&["6-311g**", "6-311++g**"], true),
        ];
        for (names, want_pure) in families {
            for &name in names {
                let bs = bundled(name).unwrap();
                let mut n_d = 0usize;
                for (z, shells) in &bs.shells {
                    for sh in shells.iter().filter(|s| s.l == 2) {
                        assert_eq!(sh.pure, want_pure, "{name} Z={z}: d shell pure={}", sh.pure);
                        n_d += 1;
                    }
                }
                // reachability: the family must actually carry d shells
                assert!(n_d >= 8, "{name}: only {n_d} d shells checked");
            }
        }
        // The override's target elements are present (Na-Ar d shells exist).
        let bs = bundled("6-311g**").unwrap();
        for z in 11..=18 {
            assert!(bs
                .for_element(z)
                .unwrap()
                .iter()
                .any(|s| s.l == 2 && s.pure));
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
        assert_eq!(num_functions(0, false), 1); // s
        assert_eq!(num_functions(1, false), 3); // p
        assert_eq!(num_functions(2, false), 6); // 6d
        assert_eq!(num_functions(3, false), 10); // 10f
    }

    #[test]
    fn test_num_functions_spherical() {
        assert_eq!(num_functions(0, true), 1); // s
        assert_eq!(num_functions(1, true), 3); // p
        assert_eq!(num_functions(2, true), 5); // 5d
        assert_eq!(num_functions(3, true), 7); // 7f
    }

    #[test]
    fn test_bundled_ccpvdz_ri() {
        let bs = bundled("cc-pvdz-ri").unwrap();
        let o_shells = bs.for_element(8).unwrap();
        assert!(!o_shells.is_empty(), "cc-pVDZ-RI should have oxygen shells");
        let max_l = o_shells.iter().map(|s| s.l).max().unwrap();
        assert!(
            max_l >= 3,
            "cc-pVDZ-RI oxygen should have at least f functions, got max_l={max_l}"
        );
    }

    /// `cc-pvdz-rifit` (the BSE name) must load and be the SAME data as
    /// `cc-pvdz-ri`. Three shipped examples (water-tda, water-tddft-pbe,
    /// water-b2plyp) needed this name and failed with "unknown bundled basis"
    /// before it was registered.
    #[test]
    fn test_bundled_ccpvdz_rifit_alias() {
        let rifit = bundled("cc-pvdz-rifit").unwrap();
        let ri = bundled("cc-pvdz-ri").unwrap();
        for z in [1i32, 6, 7, 8] {
            let a = rifit
                .for_element(z)
                .unwrap_or_else(|| panic!("cc-pvdz-rifit has no shells for Z={z}"));
            assert!(!a.is_empty(), "cc-pvdz-rifit: empty shell list for Z={z}");
            let b = ri.for_element(z).unwrap();
            assert_eq!(a.len(), b.len(), "Z={z}: alias must be the same data");
            for (sa, sb) in a.iter().zip(b) {
                assert_eq!(sa.l, sb.l, "Z={z}: shell l differs");
                assert_eq!(sa.exponents, sb.exponents, "Z={z}: exponents differ");
            }
        }
        // An RI set for an oxygen-containing molecule needs f functions.
        let o_max_l = rifit
            .for_element(8)
            .unwrap()
            .iter()
            .map(|s| s.l)
            .max()
            .unwrap();
        assert!(
            o_max_l >= 3,
            "cc-pvdz-rifit O max_l = {o_max_l}, expected >= 3"
        );
        assert_eq!(rifit.name, "cc-pvdz-rifit");
    }

    #[test]
    fn test_bundled_def2svp_rifit() {
        let bs = bundled("def2-svp-rifit").unwrap();
        let h_shells = bs.for_element(1).unwrap();
        assert!(
            !h_shells.is_empty(),
            "def2-SVP-RIFIT should have hydrogen shells"
        );
    }

    /// s and p shells must NEVER be pure, in ANY bundled basis, even when the
    /// basis-set JSON declares `gto_spherical` (def2-SVP does exactly that for
    /// iodine's p shells). The loader enforces this via
    /// `shell_pure = pure && l >= 2`.
    ///
    /// This is load-bearing beyond bookkeeping: `ferric_integrals::ao_grid`'s
    /// l=1 arm emits Cartesian (x, y, z) order for BOTH the pure and Cartesian
    /// cases. That is correct only while no pure p-shell exists, because
    /// libint2 orders pure p as (y, z, x) — the m = -1, 0, +1 solid-harmonic
    /// order. If this test ever fails, ao_grid's l=1 arm silently produces
    /// permuted p-orbital values on the grid, which corrupts every KS-DFT
    /// density, Vxc and XC gradient touching that basis.
    #[test]
    fn l_below_2_is_never_pure() {
        // every bundled orbital/aux set, not a hand-picked few
        let all = bundled_names();
        let mut checked = 0usize;
        for &name in &all {
            let bs = bundled(name).unwrap_or_else(|e| panic!("bundled({name}): {e}"));
            checked += 1;
            for (z, shells) in &bs.shells {
                for sh in shells {
                    assert!(
                        !(sh.pure && sh.l < 2),
                        "{name}: Z={z} has a PURE l={} shell — ao_grid's l=1 arm \
                         assumes Cartesian ordering and would silently permute \
                         p values (libint2 pure p is y,z,x). See the comment at \
                         ferric-integrals/src/ao_grid.rs's (1, _) arm.",
                        sh.l
                    );
                }
            }
        }
        // reachability: a typo'd list that loads nothing would pass vacuously
        assert_eq!(checked, all.len(), "not every bundled basis was checked");
        assert!(checked >= 60, "registry shrank to {checked} sets");
    }

    fn bundled_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/basis/bundled")
    }

    fn json_stems_on_disk() -> Vec<String> {
        let mut v: Vec<String> = fs::read_dir(bundled_dir())
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    /// Every JSON in basis/bundled/ is registered or allowlisted with a
    /// reason, every registered stem has a file, and the build packed exactly
    /// the files it should. 21 sets once shipped unregistered for months.
    #[test]
    fn every_bundled_json_is_registered_or_allowlisted() {
        let on_disk = json_stems_on_disk();
        let allow: Vec<&str> = UNREGISTERED_ALLOWLIST.iter().map(|(s, _)| *s).collect();
        for stem in &on_disk {
            let registered = REGISTRY.iter().any(|(s, _)| s == stem);
            let allowed = allow.contains(&stem.as_str());
            assert!(
                registered ^ allowed,
                "{stem}.json must be EITHER registered in basis::REGISTRY OR on \
                 UNREGISTERED_ALLOWLIST with a reason (registered={registered}, \
                 allowlisted={allowed})"
            );
        }
        for (stem, names) in REGISTRY {
            assert!(
                on_disk.iter().any(|s| s == stem),
                "registered {stem}.json is missing"
            );
            assert!(!names.is_empty(), "{stem}: no names");
        }
        for (stem, why) in UNREGISTERED_ALLOWLIST {
            assert!(
                on_disk.iter().any(|s| s == stem),
                "allowlisted {stem}.json is missing"
            );
            assert!(!why.is_empty(), "{stem}: allowlist entry needs a reason");
        }
        let expected_packed: Vec<String> =
            on_disk.into_iter().filter(|s| s != "def2-ecp").collect();
        assert_eq!(packed::packed_stems(), expected_packed);
    }

    /// No name or alias may resolve to two different files, and every name
    /// must already be in canonical form (or it could never match).
    #[test]
    fn bundled_names_are_canonical_and_unambiguous() {
        let mut seen: HashMap<&str, &str> = HashMap::new();
        for (stem, names) in REGISTRY {
            for &n in *names {
                assert_eq!(canonical_name(n), n, "{n:?} is not canonical");
                if let Some(prev) = seen.insert(n, stem) {
                    panic!("{n:?} maps to both {prev} and {stem}");
                }
                bundled(n).unwrap_or_else(|e| panic!("bundled({n:?}): {e}"));
            }
        }
        assert_eq!(
            bundled("6-31G(D,P)").unwrap().shells,
            bundled("6-31g**").unwrap().shells
        );
        assert_eq!(
            bundled(" def2-TZVP/C ").unwrap().shells,
            bundled("def2-tzvp-rifit").unwrap().shells
        );
    }

    /// The packed library is a faithful copy: every registered set equals the
    /// same JSON parsed from disk by `load_bse_json`, shell for shell and ECP
    /// for ECP, so the build-time pooling cannot drop, reorder or alter data.
    #[test]
    fn packed_sets_equal_their_json_parsed_from_disk() {
        for (stem, names) in REGISTRY {
            let path = bundled_dir().join(format!("{stem}.json"));
            let disk = load_bse_json(path.to_str().unwrap()).unwrap();
            let packed = bundled(names[0]).unwrap();
            assert_eq!(packed.shells, disk.shells, "{stem}: shells differ");
            assert_eq!(packed.ecps, disk.ecps, "{stem}: ECPs differ");
            assert!(!packed.shells.is_empty(), "{stem}: no elements");
        }
    }

    /// Every contracted shell of every bundled set has unit self-overlap,
    /// evaluated here from the closed-form Gaussian overlap, independently of
    /// libint (which renormalizes internally and so cannot see a missing
    /// renormalization) and of `renormalize_contraction` itself.
    #[test]
    fn every_bundled_contraction_has_unit_self_overlap() {
        fn self_overlap(sh: &Shell) -> f64 {
            // <g_a|g_b> for unit-normalized primitives of angular momentum l
            // is (2 sqrt(a b) / (a + b))^(l + 3/2).
            let l = f64::from(sh.l);
            let mut s = 0.0;
            for (a, ca) in sh.exponents.iter().zip(&sh.coefficients) {
                for (b, cb) in sh.exponents.iter().zip(&sh.coefficients) {
                    s += ca * cb * ((4.0 * a * b).sqrt() / (a + b)).powf(l + 1.5);
                }
            }
            s
        }
        let mut n = 0usize;
        for name in bundled_names() {
            let bs = bundled(name).unwrap();
            for (z, shells) in &bs.shells {
                for sh in shells {
                    let s = self_overlap(sh);
                    assert!((s - 1.0).abs() < 1e-12, "{name} Z={z} l={}: S = {s}", sh.l);
                    n += 1;
                }
            }
        }
        assert!(n > 10_000, "only {n} shells checked");
    }

    #[test]
    fn unknown_name_lists_known_sets() {
        let msg = bundled("6-31g***").unwrap_err().to_string();
        assert!(
            msg.contains("unknown bundled basis") && msg.contains("def2-tzvp"),
            "{msg}"
        );
    }

    #[test]
    fn missing_element_error_names_element_and_coverage() {
        let bs = bundled("cc-pvtz-jkfit").unwrap();
        assert!(bs.for_element(3).is_none());
        let msg = bs.missing_element_error(3).to_string();
        assert!(
            msg.contains("Li (Z=3)") && msg.contains("H, B-F, Al-Cl, Ga-Br"),
            "{msg}"
        );
    }
}
