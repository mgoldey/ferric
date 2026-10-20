//! Range-separated exact exchange for the periodic KS-DFT paths (k-point,
//! and the Gamma point through [`crate::gamma_rsh`]).
//!
//! ferric-dft describes a range-separated hybrid by [`CamCoeffs`]: the
//! exact-exchange operator is `c_sr` at short range and `c_lr` at long range,
//! `v(r) = c_sr erfc(ωr)/r + c_lr erf(ωr)/r`. In reciprocal space
//!
//! ```text
//! v(K) = 4π/K² [ c_sr (1 − e^{−K²/4ω²}) + c_lr e^{−K²/4ω²} ]
//!      = 4π/K² [ c_sr + (c_lr − c_sr) e^{−K²/4ω²} ]
//! ```
//!
//! i.e. `c_sr` × (full Coulomb) + `(c_lr − c_sr)` × (erf-attenuated), which is
//! exactly how PySCF's `pbc.dft.rks._get_jk` forms the hybrid K
//! (`hyb · K_full + (α − hyb) · K_LR`; HSE is `c_lr = 0`, whose
//! `K_SR = K_full − K_LR` is linear in the same two pieces).
//!
//! # G = 0 / Madelung
//!
//! The exchange-divergence (`exxdiv = ewald`) constant is linear in the same
//! two pieces: `v_M = c_sr v_M^{full} + (c_lr − c_sr) v_M^{LR}(ω)`, with the
//! supercell constants. `v_M^{LR}(ω)` is PySCF's `tools.pbc.madelung(cell,
//! kpts, omega=ω)` for `ω > 0`,
//!
//! ```text
//! v_M^{LR}(ω) = 2ω/√π − (4π/Ω_sc) Σ_{G≠0} e^{−G²/4ω²}/G²
//! ```
//!
//! (finite: the erf kernel is regular at `G = 0`), and the SR constant is
//! `v_M^{full} − v_M^{LR}` (PySCF's `omega < 0` branch). `ExxDiv::None` drops
//! the `K = 0` term of every piece, as PySCF.
//!
//! # Strain (the k-point RKS stress, [`crate::kxcstress`])
//!
//! Under `a_i → (1 + ε) a_i` with Miller indices and fractional `q` fixed,
//! `K → (1 + ε)^{−T} K`, so `dK²/dε_ab = −2 K_a K_b` and
//!
//! ```text
//! dv/dK² = −v/K² + (4π/K²) f'(K²),   f'(K²) = −(c_lr − c_sr) e^{−K²/4ω²}/4ω²
//! ```
//!
//! ([`RshParams::kernel_slope`]). The mixed Madelung constant strains as
//! `c_sr dv_M^{full}/dε + (c_lr − c_sr) dv_M^{LR}/dε` ([`RshParams::madelung_strain`]),
//! with ([`madelung_lr_strain`], `g(G²) = e^{−G²/4ω²}/G²`)
//!
//! ```text
//! dv_M^{LR}/dε_ab = (4π/Ω) [ δ_ab Σ_{G≠0} g + 2 Σ_{G≠0} g'(G²) G_a G_b ]
//! ```
//!
//! (the `1/Ω` volume term and the `G` shape; the `2ω/√π` self term is
//! strain-free). Both were prototyped and checked against central FD on a
//! triclinic toy lattice before the Rust (h² convergence to 3e-11).

use crate::dft::PeriodicDftError;
use crate::lattice::Cell;
use ferric_core::FerricError;
use ferric_dft::libxc::{xc_def_from_name, CamCoeffs, FunctionalFamily, XcDef};
use std::f64::consts::PI;

/// Range-separation parameters of the exact exchange: see the module doc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RshParams {
    /// Attenuation `ω` (Bohr⁻¹), `> 0`.
    pub omega: f64,
    /// Short-range exact-exchange fraction.
    pub c_sr: f64,
    /// Long-range exact-exchange fraction.
    pub c_lr: f64,
}

impl RshParams {
    /// From ferric-dft's CAM coefficients.
    pub fn from_cam(cam: &CamCoeffs) -> Self {
        Self {
            omega: cam.omega,
            c_sr: cam.c_sr,
            c_lr: cam.c_lr,
        }
    }

    /// Validate `ω` and the fractions.
    pub fn validate(&self) -> Result<(), FerricError> {
        if !(self.omega > 0.0 && self.omega.is_finite())
            || !self.c_sr.is_finite()
            || !self.c_lr.is_finite()
        {
            return Err(FerricError::General(format!(
                "range-separated exchange: need finite omega > 0 and finite fractions, got \
                 omega {}, c_sr {}, c_lr {}",
                self.omega, self.c_sr, self.c_lr
            )));
        }
        Ok(())
    }

    /// The multiplier of `4π/K²`: `c_sr + (c_lr − c_sr) e^{−K²/4ω²}`.
    pub fn kernel_factor(&self, k2: f64) -> f64 {
        self.c_sr + (self.c_lr - self.c_sr) * (-k2 / (4.0 * self.omega * self.omega)).exp()
    }

    /// `dv/dK²` of the exchange kernel `v(K) = 4π/K² · f(K²)`
    /// ([`Self::kernel_factor`]): `−v/K² + (4π/K²) f'(K²)` (module doc, "Strain").
    pub fn kernel_slope(&self, k2: f64) -> f64 {
        let v = 4.0 * PI / k2 * self.kernel_factor(k2);
        -v / k2 + 4.0 * PI / k2 * self.kernel_factor_slope(k2)
    }

    /// `f'(K²) = −(c_lr − c_sr) e^{−K²/4ω²} / 4ω²`, the slope of
    /// [`Self::kernel_factor`].
    pub fn kernel_factor_slope(&self, k2: f64) -> f64 {
        let w2 = 4.0 * self.omega * self.omega;
        -(self.c_lr - self.c_sr) * (-k2 / w2).exp() / w2
    }

    /// The mixed exchange-divergence constant `c_sr v_M^{full} + (c_lr − c_sr)
    /// v_M^{LR}(ω)` of the lattice of `supercell` (the `diag(N)` supercell of
    /// a k-mesh, see [`crate::kpts::KPointMesh::supercell_lattice`]).
    pub fn madelung(&self, supercell: &Cell) -> Result<f64, FerricError> {
        self.validate()?;
        let full = crate::ewald::madelung_constant(supercell)?;
        let lr = madelung_lr(supercell, self.omega)?;
        Ok(self.c_sr * full + (self.c_lr - self.c_sr) * lr)
    }

    /// Strain derivative `dv_M/dε_ab` of [`Self::madelung`] for the lattice
    /// of `supercell` (module doc, "Strain").
    pub fn madelung_strain(&self, supercell: &Cell) -> Result<[[f64; 3]; 3], FerricError> {
        self.validate()?;
        let full = crate::ewald::madelung_strain(supercell)?;
        let lr = madelung_lr_strain(supercell, self.omega)?;
        let mut out = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                out[a][b] = self.c_sr * full[a][b] + (self.c_lr - self.c_sr) * lr[a][b];
            }
        }
        Ok(out)
    }
}

/// `v_M^{LR}(ω)` of `cell`'s lattice (module doc); PySCF
/// `tools.pbc.madelung(cell, omega=ω)`, `ω > 0`. The sum is carried to
/// `e^{−G²/4ω²} < 1e-18`.
pub fn madelung_lr(cell: &Cell, omega: f64) -> Result<f64, FerricError> {
    if !(omega > 0.0 && omega.is_finite()) {
        return Err(FerricError::General(format!(
            "madelung_lr: omega must be finite and > 0, got {omega}"
        )));
    }
    let gcut = 2.0 * omega * (18.0 * std::f64::consts::LN_10).sqrt();
    let vol = cell.volume();
    let mut s = 0.0;
    for g in cell.gvectors(gcut)? {
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        if g2 > 1e-20 {
            s += (-g2 / (4.0 * omega * omega)).exp() / g2;
        }
    }
    Ok(2.0 * omega / PI.sqrt() - 4.0 * PI / vol * s)
}

/// Strain derivative `dv_M^{LR}/dε_ab` of [`madelung_lr`] (module doc,
/// "Strain"; the same `G` sphere, whose edge terms are below 1e-18).
pub fn madelung_lr_strain(cell: &Cell, omega: f64) -> Result<[[f64; 3]; 3], FerricError> {
    if !(omega > 0.0 && omega.is_finite()) {
        return Err(FerricError::General(format!(
            "madelung_lr_strain: omega must be finite and > 0, got {omega}"
        )));
    }
    let gcut = 2.0 * omega * (18.0 * std::f64::consts::LN_10).sqrt();
    let w2 = 4.0 * omega * omega;
    let mut s = 0.0;
    let mut t = [[0.0; 3]; 3];
    for g in cell.gvectors(gcut)? {
        let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        if g2 > 1e-20 {
            let e = (-g2 / w2).exp() / g2;
            let de = -e * (1.0 / g2 + 1.0 / w2);
            s += e;
            for a in 0..3 {
                for b in 0..3 {
                    t[a][b] += 2.0 * de * g[a] * g[b];
                }
            }
        }
    }
    let fac = 4.0 * PI / cell.volume();
    let mut out = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            out[a][b] = fac * (if a == b { s } else { 0.0 } + t[a][b]);
        }
    }
    Ok(out)
}

/// The exchange-divergence constant a k-point KS-DFT run with `functional`
/// APPLIES under `exxdiv = ewald`: the mesh (supercell) Madelung constant for
/// every functional but a range-separated hybrid, whose constant is the mixed
/// `c_sr v_M^{full} + (c_lr − c_sr) v_M^{LR}(ω)` ([`RshParams::madelung`]).
pub fn kpoint_madelung(
    functional: &str,
    cell: &Cell,
    mesh: &crate::kpts::KPointMesh,
) -> Result<f64, FerricError> {
    match resolve_periodic_functional_rsh(functional)?.2 {
        Some(r) => r.madelung(&Cell::new(cell.mol().clone(), mesh.supercell_lattice())?),
        None => mesh.madelung(cell),
    }
}

/// `true` iff `name` resolves to a range-separated (CAM) hybrid that the
/// k-point path accepts (no VV10, no meta-GGA, no double hybrid).
pub fn is_range_separated(name: &str) -> bool {
    matches!(resolve_periodic_functional_rsh(name), Ok((_, _, Some(_))))
}

/// [`crate::dft::resolve_periodic_functional`] that ACCEPTS range-separated
/// hybrids: `(XcDef, global exact-exchange fraction, Some(RshParams) for a
/// CAM functional)`. For a CAM functional the global fraction is `0.0` (the
/// exchange is described by the returned [`RshParams`]). A meta-GGA without
/// range separation (SCAN, r2SCAN) is accepted; a range-separated meta-GGA,
/// VV10 and double hybrids are still refused.
pub fn resolve_periodic_functional_rsh(
    name: &str,
) -> Result<(XcDef, f64, Option<RshParams>), FerricError> {
    let def = xc_def_from_name(name)
        .map_err(|e| FerricError::General(format!("periodic DFT: functional {name:?}: {e:?}")))?;
    let Some(cam) = def.cam else {
        // Not range-separated: the global-exchange rules, with meta-GGA
        // allowed (the k-point path supplies tau, see `crate::kdft`).
        let (def, a) = crate::dft::resolve_periodic_functional_mgga(name, true)?;
        return Ok((def, a, None));
    };
    if def
        .funcs
        .iter()
        .any(|f| matches!(f.family(), FunctionalFamily::MetaGga))
    {
        return Err(PeriodicDftError::Unsupported {
            feature: "meta-GGA",
            reason: format!("{name}: the tau path was not prototyped periodically"),
        }
        .into());
    }
    if def.vv10.is_some() {
        return Err(PeriodicDftError::Unsupported {
            feature: "VV10 nonlocal correlation",
            reason: format!("{name}: no periodic NLC grid"),
        }
        .into());
    }
    if def.weights.is_some() {
        return Err(PeriodicDftError::Unsupported {
            feature: "scaled composite / double hybrid",
            reason: format!("{name}: only LDA, GGA and (range-separated) hybrids are supported"),
        }
        .into());
    }
    let p = RshParams::from_cam(&cam);
    p.validate()?;
    Ok((def, 0.0, Some(p)))
}
