//! Basis-set conversion between ferric's `Shell`/`BasisSet` and cuEST's
//! primitive-and-L-normalized shell representation.
//!
//! cuEST does **not** normalize for you: `cuestAOShellCreate()` takes
//! "finished" primitive-and-L-normalized contraction coefficients (see the
//! cuEST "Primitive Basis Function Normalization" docs). ferric stores
//! contraction coefficients for *unnormalized* primitives, with the contraction
//! renormalized under the normalized-primitive metric (V = 1, see
//! `renormalize_contraction` in ferric-core). ferric's primitive normalization
//! factor
//!   N(α,l) = (2α/π)^{3/4} (4α)^{l/2} / sqrt((2l-1)!!)
//! is exactly cuEST's `primitive_factor`, so converting a ferric shell to a
//! cuEST shell is just multiplying each coefficient by `primitive_factor(l,e)`.
//! The contraction is already normalized (V = 1), so no further scaling is
//! needed.

use ferric_core::basis::{BasisSet, Shell};

/// A cuEST-ready shell: angular momentum, spherical flag, exponents, and
/// finished (primitive-and-L-normalized, contraction-normalized) coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct CuestShell {
    pub l: i32,
    pub pure: bool,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}

/// Errors from basis conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BasisConvertError {
    EmptyShell,
    LengthMismatch,
    NonPositiveExponent,
    NonFinite,
    NegativeL,
}

impl std::fmt::Display for BasisConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyShell => write!(f, "shell has no primitives"),
            Self::LengthMismatch => write!(f, "exponents and coefficients length mismatch"),
            Self::NonPositiveExponent => write!(f, "exponent must be > 0"),
            Self::NonFinite => write!(f, "non-finite exponent or coefficient"),
            Self::NegativeL => write!(f, "angular momentum L must be >= 0"),
        }
    }
}

impl std::error::Error for BasisConvertError {}

/// Convert a single ferric shell into cuEST form.
pub fn convert_shell(shell: &Shell) -> Result<CuestShell, BasisConvertError> {
    if shell.l < 0 {
        return Err(BasisConvertError::NegativeL);
    }
    if shell.exponents.is_empty() {
        return Err(BasisConvertError::EmptyShell);
    }
    if shell.exponents.len() != shell.coefficients.len() {
        return Err(BasisConvertError::LengthMismatch);
    }
    for (e, c) in shell.exponents.iter().zip(shell.coefficients.iter()) {
        if !e.is_finite() || !c.is_finite() {
            return Err(BasisConvertError::NonFinite);
        }
        if *e <= 0.0 {
            return Err(BasisConvertError::NonPositiveExponent);
        }
    }
    let coeffs: Vec<f64> = shell
        .exponents
        .iter()
        .zip(shell.coefficients.iter())
        .map(|(e, c)| c * primitive_factor(shell.l, *e))
        .collect();
    Ok(CuestShell {
        l: shell.l,
        pure: shell.pure,
        exponents: shell.exponents.clone(),
        coefficients: coeffs,
    })
}

/// Convert all shells of one element (Z) from a ferric basis.
pub fn convert_element(bs: &BasisSet, z: i32) -> Result<Vec<CuestShell>, BasisConvertError> {
    let shells = bs.for_element(z).ok_or(BasisConvertError::EmptyShell)?;
    shells.iter().map(convert_shell).collect()
}

/// Number of basis functions in a shell (CCA ordering count).
pub fn shell_nfunctions(l: i32, pure: bool) -> usize {
    if pure {
        (2 * l + 1) as usize
    } else {
        ((l + 1) * (l + 2) / 2) as usize
    }
}

/// `d = prod_{k=1}^{L} (2k - 1)` (empty product = 1 for L = 0).
pub fn d_factor(l: i32) -> f64 {
    let mut d = 1.0;
    for k in 1..=l {
        d *= (2 * k - 1) as f64;
    }
    d
}

/// Primitive normalization factor for a single exponent `e` at angular momentum
/// `l`, per the cuEST formula:
///   sqrt( 2^L * (2e)^{L+3/2} / (pi^{3/2} * d) ).
pub fn primitive_factor(l: i32, e: f64) -> f64 {
    let d = d_factor(l);
    let num = 2.0_f64.powi(l) * (2.0 * e).powf(l as f64 + 1.5);
    let den = std::f64::consts::PI.powf(1.5) * d;
    (num / den).sqrt()
}

/// Normalize a shell given **raw** (unnormalized) coefficients `raw` and
/// exponents `exps`, producing cuEST "finished" coefficients.
pub fn normalize_raw_shell(
    l: i32,
    raw: &[f64],
    exps: &[f64],
    n_fact: f64,
) -> Result<Vec<f64>, BasisConvertError> {
    if l < 0 {
        return Err(BasisConvertError::NegativeL);
    }
    if raw.is_empty() || exps.is_empty() {
        return Err(BasisConvertError::EmptyShell);
    }
    if raw.len() != exps.len() {
        return Err(BasisConvertError::LengthMismatch);
    }
    for (e, c) in exps.iter().zip(raw.iter()) {
        if !e.is_finite() || !c.is_finite() {
            return Err(BasisConvertError::NonFinite);
        }
        if *e <= 0.0 {
            return Err(BasisConvertError::NonPositiveExponent);
        }
    }
    let mut v = 0.0;
    for (i, ei) in exps.iter().enumerate() {
        for (j, ej) in exps.iter().enumerate() {
            let ratio = 2.0 * (ei * ej).sqrt() / (ei + ej);
            v += raw[i] * raw[j] * ratio.powf(l as f64 + 1.5);
        }
    }
    if v <= 0.0 || !v.is_finite() {
        return Err(BasisConvertError::NonFinite);
    }
    let scale = (n_fact / v).sqrt();
    let mut out = Vec::with_capacity(raw.len());
    for (k, ek) in exps.iter().enumerate() {
        out.push(raw[k] * primitive_factor(l, *ek) * scale);
    }
    Ok(out)
}
