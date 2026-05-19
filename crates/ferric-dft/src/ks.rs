//! `KsXc`: caches the molecular grid + AO + ∇AO + (optional) NLC grid + AO,
//! and implements `XcContribution` for use in the closed-shell KS SCF.

use ndarray::{Array2, Array3};
use thiserror::Error;

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;

use crate::ao_grid::{eval_basis_and_grad_on_points, GtoEvalError};
use crate::density_on_grid::{eval_density_closed, eval_density_uks, DensityGrid};
use crate::grid::{build_atomic_grid, AtomicGridConfig, GridPoint};
use crate::libxc::{xc_def_from_name, xc_def_from_name_nspin, LibxcError, XcDef};
use crate::vv10::add_vv10;
use crate::vxc::{semilocal_vxc_closed, semilocal_vxc_polarized};
use crate::xc_trait::{KMix, UksXcContribution, XcContribution};

#[derive(Error, Debug)]
pub enum KsXcError {
    #[error("AO evaluation failed: {0:?}")]
    Eval(GtoEvalError),
    #[error("libxc resolver failed: {0}")]
    Libxc(LibxcError),
}

impl From<GtoEvalError> for KsXcError { fn from(e: GtoEvalError) -> Self { Self::Eval(e) } }
impl From<LibxcError>  for KsXcError { fn from(e: LibxcError)  -> Self { Self::Libxc(e) } }

/// Caches everything needed to compute V_xc and V_nl per SCF iteration:
/// a Becke-Lebedev grid plus precomputed χ, ∇χ on its points. If the XC
/// definition includes VV10, also caches a smaller NLC grid + AO.
pub struct KsXc {
    pub xc: XcDef,
    pub grid: Vec<GridPoint>,
    pub chi: Array2<f64>,
    pub dchi: Array3<f64>,
    pub nlc_grid: Option<Vec<GridPoint>>,
    pub nlc_chi: Option<Array2<f64>>,
    pub nlc_dchi: Option<Array3<f64>>,
}

impl KsXc {
    pub fn new(
        mol: &Molecule,
        bs: &BasisSet,
        xc_name: &str,
        main: &AtomicGridConfig,
        nlc: &AtomicGridConfig,
    ) -> Result<Self, KsXcError> {
        let xc = xc_def_from_name(xc_name)?;

        let grid = build_atomic_grid(mol, main);
        let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        let (chi, dchi) = eval_basis_and_grad_on_points(mol, bs, &pts)?;

        let (nlc_grid, nlc_chi, nlc_dchi) = if xc.vv10.is_some() {
            let g = build_atomic_grid(mol, nlc);
            let p: Vec<[f64; 3]> = g.iter().map(|gp| gp.xyz).collect();
            let (c, dc) = eval_basis_and_grad_on_points(mol, bs, &p)?;
            (Some(g), Some(c), Some(dc))
        } else {
            (None, None, None)
        };

        Ok(Self { xc, grid, chi, dchi, nlc_grid, nlc_chi, nlc_dchi })
    }
}

impl XcContribution for KsXc {
    fn add_xc(&self, d: &Array2<f64>, f: &mut Array2<f64>) -> f64 {
        // Semilocal piece.
        let dens = eval_density_closed(d, &self.chi, &self.dchi);
        let (e_xc, vxc) = semilocal_vxc_closed(&self.grid, &self.chi, &self.dchi, &dens, &self.xc);
        *f += &vxc;

        // VV10 nonlocal correlation (stub returns 0.0 for now).
        let e_nl = if let (Some(g), Some(c), Some(dc), Some(params)) = (
            self.nlc_grid.as_ref(),
            self.nlc_chi.as_ref(),
            self.nlc_dchi.as_ref(),
            self.xc.vv10.as_ref(),
        ) {
            let nlc_dens = eval_density_closed(d, c, dc);
            add_vv10(g, c, dc, &nlc_dens, params, f)
        } else {
            0.0
        };

        e_xc + e_nl
    }

    fn k_mix(&self) -> KMix {
        if let Some(cam) = self.xc.cam {
            return KMix { sr: cam.c_sr, lr: cam.c_lr, omega: cam.omega };
        }
        if let Some(mix) = self.xc.b3lyp_mix {
            return KMix { sr: mix, lr: mix, omega: 0.0 };
        }
        // Pure functional (LDA, PBE): no exact exchange
        KMix { sr: 0.0, lr: 0.0, omega: 0.0 }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Spin-polarized counterpart (UKS / ROKS)
// ────────────────────────────────────────────────────────────────────────────

/// UKS analog of `KsXc`. Caches the same grids + AO data, but the
/// `XcDef` is built with `nspin=2` so libxc returns spin-resolved
/// v_ρ / v_σ. VV10 is closed-shell-friendly (function of total ρ and |∇ρ|²);
/// the V_nl piece is added equally to V_α and V_β.
pub struct KsXcUks {
    pub xc: XcDef,
    pub grid: Vec<GridPoint>,
    pub chi: Array2<f64>,
    pub dchi: Array3<f64>,
    pub nlc_grid: Option<Vec<GridPoint>>,
    pub nlc_chi: Option<Array2<f64>>,
    pub nlc_dchi: Option<Array3<f64>>,
}

impl KsXcUks {
    pub fn new(
        mol: &Molecule,
        bs: &BasisSet,
        xc_name: &str,
        main: &AtomicGridConfig,
        nlc: &AtomicGridConfig,
    ) -> Result<Self, KsXcError> {
        let xc = xc_def_from_name_nspin(xc_name, 2)?;

        let grid = build_atomic_grid(mol, main);
        let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        let (chi, dchi) = eval_basis_and_grad_on_points(mol, bs, &pts)?;

        let (nlc_grid, nlc_chi, nlc_dchi) = if xc.vv10.is_some() {
            let g = build_atomic_grid(mol, nlc);
            let p: Vec<[f64; 3]> = g.iter().map(|gp| gp.xyz).collect();
            let (c, dc) = eval_basis_and_grad_on_points(mol, bs, &p)?;
            (Some(g), Some(c), Some(dc))
        } else {
            (None, None, None)
        };

        Ok(Self { xc, grid, chi, dchi, nlc_grid, nlc_chi, nlc_dchi })
    }
}

impl UksXcContribution for KsXcUks {
    fn add_xc_uks(
        &self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
        f_a: &mut Array2<f64>,
        f_b: &mut Array2<f64>,
    ) -> f64 {
        // Semilocal: build (ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ) on the main grid
        // then call the polarized libxc path.
        let dens = eval_density_uks(d_a, d_b, &self.chi, &self.dchi);
        let (e_xc, vxc_a, vxc_b) = semilocal_vxc_polarized(
            &self.grid, &self.chi, &self.dchi, &dens, &self.xc,
        );
        *f_a += &vxc_a;
        *f_b += &vxc_b;

        // VV10 nonlocal correlation: function of (ρ_tot, |∇ρ_tot|²); the
        // V_nl matrix is the same for both spins. Cache the d_total → DensityGrid
        // and reuse the closed-shell add_vv10.
        let e_nl = if let (Some(g), Some(c), Some(dc), Some(params)) = (
            self.nlc_grid.as_ref(),
            self.nlc_chi.as_ref(),
            self.nlc_dchi.as_ref(),
            self.xc.vv10.as_ref(),
        ) {
            let d_total = d_a + d_b;
            let dens_total: DensityGrid = eval_density_closed(&d_total, c, dc);
            // Apply VV10 to a single Fock buffer, then add to both spins.
            let mut v_nl = Array2::<f64>::zeros(f_a.dim());
            let e = add_vv10(g, c, dc, &dens_total, params, &mut v_nl);
            *f_a += &v_nl;
            *f_b += &v_nl;
            e
        } else {
            0.0
        };

        e_xc + e_nl
    }

    fn k_mix(&self) -> KMix {
        if let Some(cam) = self.xc.cam {
            return KMix { sr: cam.c_sr, lr: cam.c_lr, omega: cam.omega };
        }
        if let Some(mix) = self.xc.b3lyp_mix {
            return KMix { sr: mix, lr: mix, omega: 0.0 };
        }
        KMix { sr: 0.0, lr: 0.0, omega: 0.0 }
    }
}
