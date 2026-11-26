//! ECP conversion helpers for cuEST (CPU-side, no device).
//!
//! cuEST's ECP representation uses radial powers, exponents, coefficients and
//! a top shell `L_max`. ferric's `EcpDef` already stores terms as
//! `d · r^n · exp(-ζ r^2)`, so the conversion is a field-for-field copy; the
//! risk area (§5.4) is whether cuEST's allowed powers/top-shell conventions
//! match ferric's. These helpers expose the metadata so it can be audited and
//! tested without a device.

use ferric_core::ecp::{EcpDef, EcpShell, EcpTerm};

#[derive(Debug, Clone, PartialEq)]
pub struct CuestEcpTerm {
    pub coef: f64,
    pub r_exp: i32,
    pub gexp: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CuestEcpShell {
    pub angular_momentum: i32,
    pub terms: Vec<CuestEcpTerm>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CuestEcpDef {
    pub n_core: i32,
    pub shells: Vec<CuestEcpShell>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EcpConvertError {
    NonFinite,
    NonPositiveExponent,
    NegativeAngularMomentum,
}

impl std::fmt::Display for EcpConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite => write!(f, "non-finite ECP coefficient/exponent"),
            Self::NonPositiveExponent => write!(f, "ECP Gaussian exponent must be > 0"),
            Self::NegativeAngularMomentum => write!(f, "ECP angular momentum must be >= 0"),
        }
    }
}

impl std::error::Error for EcpConvertError {}

pub fn convert_term(t: &EcpTerm) -> Result<CuestEcpTerm, EcpConvertError> {
    if !t.coef.is_finite() || !t.gexp.is_finite() {
        return Err(EcpConvertError::NonFinite);
    }
    if t.gexp <= 0.0 {
        return Err(EcpConvertError::NonPositiveExponent);
    }
    Ok(CuestEcpTerm {
        coef: t.coef,
        r_exp: t.r_exp,
        gexp: t.gexp,
    })
}

pub fn convert_shell(s: &EcpShell) -> Result<CuestEcpShell, EcpConvertError> {
    if s.angular_momentum < 0 {
        return Err(EcpConvertError::NegativeAngularMomentum);
    }
    let terms = s.terms.iter().map(convert_term).collect::<Result<_, _>>()?;
    Ok(CuestEcpShell {
        angular_momentum: s.angular_momentum,
        terms,
    })
}

pub fn convert_def(def: &EcpDef) -> Result<CuestEcpDef, EcpConvertError> {
    let shells = def
        .shells
        .iter()
        .map(convert_shell)
        .collect::<Result<_, _>>()?;
    Ok(CuestEcpDef {
        n_core: def.n_core,
        shells,
    })
}

impl CuestEcpDef {
    pub fn max_angular_momentum(&self) -> i32 {
        self.shells
            .iter()
            .map(|s| s.angular_momentum)
            .max()
            .unwrap_or(0)
    }

    pub fn term_count(&self) -> usize {
        self.shells.iter().map(|s| s.terms.len()).sum()
    }
}
