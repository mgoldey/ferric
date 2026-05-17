//! Spin-aware SCF result container.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Spin {
    Restricted,
    Unrestricted,
    RestrictedOpen,
}

#[derive(Debug, Clone)]
pub struct ScfResult {
    pub spin: Spin,
    pub energy: f64,
    /// AO total density (D_α + D_β). For Restricted this equals 2·D_α.
    pub density_total: Array2<f64>,
    /// α-spin density. Always populated.
    pub density_alpha: Array2<f64>,
    /// β-spin density. Populated for Unrestricted/RestrictedOpen; None for Restricted.
    pub density_beta: Option<Array2<f64>>,
    /// α MO coefficients (or restricted MOs).
    pub mos_alpha: Array2<f64>,
    pub mos_beta: Option<Array2<f64>>,
    pub eps_alpha: Vec<f64>,
    pub eps_beta: Option<Vec<f64>>,
    pub fock_alpha: Array2<f64>,
    pub fock_beta: Option<Array2<f64>>,
    pub converged: bool,
    pub iterations: usize,
    pub computed_quartets: usize,
}

impl ScfResult {
    /// Restricted accessor: panics if spin != Restricted.
    pub fn mos_r(&self) -> &Array2<f64> {
        assert!(matches!(self.spin, Spin::Restricted), "mos_r() called on non-restricted result");
        &self.mos_alpha
    }
    pub fn eps_r(&self) -> &[f64] {
        assert!(matches!(self.spin, Spin::Restricted), "eps_r() called on non-restricted result");
        &self.eps_alpha
    }
    pub fn fock_r(&self) -> &Array2<f64> {
        assert!(matches!(self.spin, Spin::Restricted), "fock_r() called on non-restricted result");
        &self.fock_alpha
    }
    pub fn density_r(&self) -> &Array2<f64> {
        assert!(matches!(self.spin, Spin::Restricted), "density_r() called on non-restricted result");
        &self.density_total
    }
    /// Spin-summed AO density D_α + D_β. Available for all spin types
    /// (equals 2·D_α for Restricted; D_α + D_β for U/RO). Use for properties
    /// like ESP, electric field, Löwdin/Hirshfeld charges that take a
    /// total-electron density.
    pub fn density_total(&self) -> &Array2<f64> {
        &self.density_total
    }
    /// Unrestricted/ROHF accessors. Panic if called on a Restricted result.
    pub fn mos_a(&self) -> &Array2<f64> { &self.mos_alpha }
    pub fn mos_b(&self) -> &Array2<f64> {
        self.mos_beta.as_ref().expect("mos_b() called on Restricted result")
    }
    pub fn eps_a(&self) -> &[f64] { &self.eps_alpha }
    pub fn eps_b(&self) -> &[f64] {
        self.eps_beta.as_deref().expect("eps_b() called on Restricted result")
    }
}
