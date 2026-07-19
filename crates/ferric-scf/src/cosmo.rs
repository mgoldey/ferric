//! COSMO (COnductor-like Screening MOdel) implicit solvation.
//!
//! Implements the classic Klamt & Schuurmann apparent-surface-charge (ASC)
//! continuum solvation model:
//!
//!   Klamt, A.; Schuurmann, G. "COSMO: A New Approach to Dielectric
//!   Screening in Solvents with Explicit Expressions for the Screening
//!   Energy and its Gradient." J. Chem. Soc., Perkin Trans. 2, 1993,
//!   799-805.
//!
//! # Model summary
//!
//! 1. **Cavity**: the solute is enclosed in a cavity built from atom-centered
//!    spheres, radius = (element van der Waals radius) x `radius_scale`
//!    (default 1.17, the original COSMO scale factor -- distinct from the
//!    1.2 "Q-Chem convention" used by some PCM variants). Each sphere is
//!    discretized with a fixed Lebedev angular grid (reusing
//!    [`ferric_dft::lebedev`]); grid points buried inside a neighboring
//!    atom's sphere are discarded (simple visibility trim, no GEPOL
//!    re-tessellation of partially-buried segments).
//! 2. **Segment interaction matrix** `S` (called `A` in some papers): for
//!    two distinct segments k != l, `S_kl = 1 / |s_k - s_l|` (bare Coulomb
//!    interaction between point charges at the segment centers). The
//!    diagonal (self-interaction of a segment with itself) uses the
//!    standard closed-form COSMO self-term
//!    `S_kk = xi * sqrt(4*pi / a_k)`, `xi = 3.8`, derived by treating each
//!    segment as a small disk of area `a_k` (Klamt & Schuurmann, eq. 7).
//! 3. **Apparent surface charges**: given the solute's electrostatic
//!    potential `v_k` at each segment (nuclear + electronic), solve the
//!    linear system `S q = -f(eps) v` for the segment charges `q`, where
//!    `f(eps) = (eps - 1) / (eps + x)` is the dielectric scaling function.
//!    This module uses `x = 0.5` (the COSMO "ideal conductor boundary
//!    condition" value; C-PCM instead uses `x = 0` -- see Klamt & Schuurmann
//!    section 2, and cross-checked against PySCF's `solvent/pcm.py`, which
//!    implements exactly `f_epsilon = (eps - 1)/(eps + 0.5)` for
//!    `method="COSMO"`).
//! 4. **Reaction-field potential**: each segment charge acts on the solute
//!    exactly like an external point charge, contributing
//!    `V_reaction = -sum_k q_k <mu| 1/|r - s_k| |nu>` to the one-electron
//!    Hamiltonian (sign convention matches the nuclear-attraction operator:
//!    a positive point charge is *attractive* to electrons). This must be
//!    recomputed every SCF iteration since `q` depends on the converged
//!    density through `v_k`.
//! 5. **Solvation energy**: `E_cosmo = 0.5 * sum_k q_k v_k` (the standard
//!    ASC self-energy expression; the factor 1/2 accounts for the
//!    linear-response relationship between charge and potential).
//!
//! # What is deferred (explicitly out of scope for this module)
//!
//! * **Outlying charge correction** (Klamt & Jonas 1996): a small
//!   correction for solute electron density that leaks outside the cavity
//!   surface. Not implemented; the energy will be *slightly* under- or
//!   over-stabilized relative to a production COSMO code for exposed
//!   heteroatom lone pairs. Left as a documented gap.
//! * **Non-electrostatic terms** (cavitation, dispersion, repulsion -- the
//!   "CDS" terms in some SMx-flavored models): not implemented. This module
//!   is the pure electrostatic COSMO reaction field only.
//! * **Analytic gradients**: not implemented (no `cosmo_gradient` function).
//! * **GEPOL-style segment re-tessellation** of partially-buried spheres:
//!   segments are either fully kept or fully discarded based on their
//!   center point only (no partial-area accounting for a segment whose
//!   Lebedev "tile" straddles a burial boundary). This is a well-known
//!   simplification relative to production GEPOL cavities but converges to
//!   the same physics as the angular grid is refined.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi::{self, CAtom};
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

/// COSMO self-term prefactor (Klamt & Schuurmann 1993, eq. 7): `S_kk = xi *
/// sqrt(4*pi / a_k)`. This is the standard constant quoted throughout the
/// COSMO literature for a segment modeled as a small circular disk.
const COSMO_XI: f64 = 3.8;

/// Default COSMO cavity radius scale factor (Klamt & Schuurmann 1993).
pub const DEFAULT_RADIUS_SCALE: f64 = 1.17;

/// Default Lebedev angular grid order per atomic sphere. 110 points is a
/// reasonable accuracy/cost tradeoff for a fixed (non-adaptive) cavity grid;
/// see [`lebedev`](ferric_dft::lebedev) for supported orders.
pub const DEFAULT_LEBEDEV_ORDER: usize = 110;

/// Bondi (1964) van der Waals radii in Bohr, for the subset of elements this
/// module supports. Values in Angstrom (converted to Bohr below) match the
/// widely used Bondi table (also the basis PySCF's `pyscf.data.radii.VDW`
/// draws from for H-Cl; consistent to 3 decimal places against a local PySCF
/// checkout's `radii.VDW`, spot-checked during development).
const ANGSTROM_TO_BOHR: f64 = 1.0 / 0.529_177_210_92;

fn bondi_radius_angstrom(z: i32) -> Option<f64> {
    let r = match z {
        1 => 1.20,  // H
        2 => 1.40,  // He
        3 => 1.82,  // Li
        4 => 1.53,  // Be
        5 => 1.92,  // B
        6 => 1.70,  // C
        7 => 1.55,  // N
        8 => 1.52,  // O
        9 => 1.47,  // F
        10 => 1.54, // Ne
        11 => 2.27, // Na
        12 => 1.73, // Mg
        13 => 1.84, // Al
        14 => 2.10, // Si
        15 => 1.80, // P
        16 => 1.80, // S
        17 => 1.75, // Cl
        18 => 1.88, // Ar
        19 => 2.75, // K
        20 => 2.31, // Ca
        35 => 1.85, // Br
        36 => 2.02, // Kr
        53 => 1.98, // I
        54 => 2.16, // Xe
        _ => return None,
    };
    Some(r)
}

/// Look up the Bondi van der Waals radius for element `z`, in Bohr.
/// Returns `Err` (never a fabricated fallback) for unsupported elements —
/// per the repo's "no silent fallback" convention, an unsupported element
/// must abort cavity construction rather than guess a radius.
pub fn bondi_radius_bohr(z: i32) -> Result<f64, FerricError> {
    bondi_radius_angstrom(z)
        .map(|r| r * ANGSTROM_TO_BOHR)
        .ok_or_else(|| {
            FerricError::General(format!(
                "cosmo: no Bondi van der Waals radius tabulated for Z={z}; \
                 add an entry to cosmo::bondi_radius_angstrom or exclude this \
                 element from the solvated region"
            ))
        })
}

/// User-facing COSMO configuration. `#[serde(deny_unknown_fields)]` per the
/// repo's config-honesty convention (a typo'd TOML key must hard-error, not
/// silently no-op).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CosmoConfig {
    /// Solvent dielectric constant (e.g. 78.39 for water at 298 K, PySCF/
    /// Q-Chem default 78.3553). Must be finite and > 1.0 (a vacuum has no
    /// solvation and should use `external_potential`/`cosmo: None` instead).
    pub epsilon: f64,
    /// Cavity radius scale factor applied to each atom's Bondi vdW radius.
    /// Default [`DEFAULT_RADIUS_SCALE`] (1.17, the original Klamt-Schuurmann
    /// value) when not specified in TOML (see `Default` impl).
    #[serde(default = "default_radius_scale")]
    pub radius_scale: f64,
    /// Lebedev angular grid order per atomic sphere. Default
    /// [`DEFAULT_LEBEDEV_ORDER`]. Must be one of the orders supported by
    /// [`ferric_dft::lebedev::lebedev`] (6, 14, 26, 50, 110, 302).
    #[serde(default = "default_lebedev_order")]
    pub lebedev_order: usize,
}

fn default_radius_scale() -> f64 {
    DEFAULT_RADIUS_SCALE
}
fn default_lebedev_order() -> usize {
    DEFAULT_LEBEDEV_ORDER
}

impl Default for CosmoConfig {
    fn default() -> Self {
        Self {
            epsilon: 78.39,
            radius_scale: DEFAULT_RADIUS_SCALE,
            lebedev_order: DEFAULT_LEBEDEV_ORDER,
        }
    }
}

impl CosmoConfig {
    /// COSMO dielectric scaling function `f(eps) = (eps - 1) / (eps + 0.5)`
    /// (the ideal-conductor-boundary-condition form; x = 0.5). Cross-checked
    /// against PySCF `solvent/pcm.py`'s `method="COSMO"` branch:
    /// `f_epsilon = (epsilon - 1.0)/(epsilon + 1.0/2.0)`.
    pub fn f_epsilon(&self) -> f64 {
        (self.epsilon - 1.0) / (self.epsilon + 0.5)
    }

    fn validate(&self) -> Result<(), FerricError> {
        if !self.epsilon.is_finite() || self.epsilon <= 1.0 {
            return Err(FerricError::General(format!(
                "cosmo: epsilon must be finite and > 1.0, got {}",
                self.epsilon
            )));
        }
        if !self.radius_scale.is_finite() || self.radius_scale <= 0.0 {
            return Err(FerricError::General(format!(
                "cosmo: radius_scale must be finite and > 0, got {}",
                self.radius_scale
            )));
        }
        if !matches!(self.lebedev_order, 6 | 14 | 26 | 50 | 110 | 302) {
            return Err(FerricError::General(format!(
                "cosmo: lebedev_order must be one of 6, 14, 26, 50, 110, 302, got {}",
                self.lebedev_order
            )));
        }
        Ok(())
    }
}

/// One discretized surface segment ("tessera") of the cavity.
#[derive(Debug, Clone, Copy)]
struct Segment {
    pos: [f64; 3],
    area: f64,
}

/// A constructed COSMO cavity: exposed surface segments over all atomic
/// spheres, ready for the `S` interaction matrix.
#[derive(Debug, Clone)]
pub struct CosmoCavity {
    segments: Vec<Segment>,
}

impl CosmoCavity {
    /// Number of surface segments ("tesserae") in the cavity.
    pub fn n_segments(&self) -> usize {
        self.segments.len()
    }

    /// Total exposed surface area, in Bohr^2.
    pub fn total_area(&self) -> f64 {
        self.segments.iter().map(|s| s.area).sum()
    }

    /// Build a cavity from atom-centered Lebedev-discretized spheres,
    /// scaled-Bondi radii, with a simple point-visibility trim: a grid point
    /// on atom A's sphere is discarded if it falls strictly inside another
    /// atom B's sphere (`|point - R_B| < R_B`). Ghost atoms (basis-only,
    /// zero nuclear charge) are excluded from the cavity — they carry no
    /// physical volume to screen.
    ///
    /// Returns `Err` if any real atom's element has no tabulated Bondi
    /// radius, or if the resulting cavity has zero segments (e.g. every atom
    /// is a ghost, or is fully buried — degenerate/unphysical input).
    pub fn build(mol: &Molecule, config: &CosmoConfig) -> Result<Self, FerricError> {
        config.validate()?;

        let real_atoms: Vec<(usize, [f64; 3], f64)> = mol
            .atoms
            .iter()
            .enumerate()
            .filter(|(_, a)| !a.ghost)
            .map(|(i, a)| -> Result<(usize, [f64; 3], f64), FerricError> {
                let r = bondi_radius_bohr(a.z)? * config.radius_scale;
                Ok((i, [a.x, a.y, a.zpos], r))
            })
            .collect::<Result<Vec<_>, _>>()?;

        if real_atoms.is_empty() {
            return Err(FerricError::General(
                "cosmo: no real (non-ghost) atoms to build a cavity around".into(),
            ));
        }

        let (unit_pts, weights) = ferric_dft::lebedev::lebedev(config.lebedev_order);

        let mut segments = Vec::new();
        for &(ia, center_a, r_a) in &real_atoms {
            let sphere_area = 4.0 * std::f64::consts::PI * r_a * r_a;
            for (pt, w) in unit_pts.iter().zip(weights.iter()) {
                let p = [
                    center_a[0] + r_a * pt[0],
                    center_a[1] + r_a * pt[1],
                    center_a[2] + r_a * pt[2],
                ];
                // Visibility trim: discard if buried inside any OTHER real
                // atom's scaled sphere.
                let mut buried = false;
                for &(ib, center_b, r_b) in &real_atoms {
                    if ib == ia {
                        continue;
                    }
                    let dx = p[0] - center_b[0];
                    let dy = p[1] - center_b[1];
                    let dz = p[2] - center_b[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < r_b * r_b {
                        buried = true;
                        break;
                    }
                }
                if buried {
                    continue;
                }
                segments.push(Segment {
                    pos: p,
                    // Lebedev weights sum to 1 over the sphere, so w*sphere_area
                    // is exactly the area subtended by this node.
                    area: w * sphere_area,
                });
            }
        }

        if segments.is_empty() {
            return Err(FerricError::General(
                "cosmo: cavity construction produced zero exposed surface segments \
                 (fully buried geometry?)"
                    .into(),
            ));
        }

        Ok(Self { segments })
    }

    /// Build the symmetric segment-interaction matrix `S` (n_seg x n_seg):
    /// off-diagonal bare Coulomb `1/|s_k - s_l|`, diagonal COSMO self-term
    /// `xi * sqrt(4*pi/a_k)`.
    fn build_s_matrix(&self) -> Array2<f64> {
        let n = self.segments.len();
        let mut s = Array2::<f64>::zeros((n, n));
        for k in 0..n {
            let ak = self.segments[k].area;
            s[(k, k)] = COSMO_XI * (4.0 * std::f64::consts::PI / ak).sqrt();
            for l in 0..k {
                let dx = self.segments[k].pos[0] - self.segments[l].pos[0];
                let dy = self.segments[k].pos[1] - self.segments[l].pos[1];
                let dz = self.segments[k].pos[2] - self.segments[l].pos[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let v = 1.0 / r;
                s[(k, l)] = v;
                s[(l, k)] = v;
            }
        }
        s
    }
}

/// Result of a single COSMO reaction-field evaluation for a given density.
#[derive(Debug, Clone)]
pub struct CosmoResult {
    /// Reaction-field addition to the one-electron Hamiltonian (hcore/Fock),
    /// shape (nbasis, nbasis). Symmetric.
    pub v_reaction: Array2<f64>,
    /// COSMO solvation (free) energy contribution, in Hartree. Negative
    /// (stabilizing) for a polar solute in a polar solvent.
    pub e_cosmo: f64,
    /// Converged apparent surface charges, one per cavity segment.
    pub charges: Array1<f64>,
}

/// Evaluate the COSMO reaction field for the given solute density.
///
/// This is the per-SCF-iteration entry point: computes the solute's
/// electrostatic potential at each cavity segment (nuclear + electronic,
/// via the same "nuclear attraction with a point charge at an arbitrary
/// point" primitive used by [`ferric_rpa::properties::esp_at_atoms`], just
/// evaluated at cavity segments instead of atoms), solves the COSMO linear
/// system for the apparent surface charges, and builds the reaction-field
/// matrix to add to the Fock/hcore.
///
/// Returns `Err` if the segment-interaction-matrix solve is singular/fails
/// (LAPACK failure) or if `density`'s shape doesn't match `prep.nbasis()` —
/// never a fabricated/clamped charge vector.
pub fn cosmo_reaction_field(
    mol: &Molecule,
    prep: &PreparedBasis,
    cavity: &CosmoCavity,
    config: &CosmoConfig,
    density: &Array2<f64>,
) -> Result<CosmoResult, FerricError> {
    let nbas = prep.nbasis();
    if density.shape() != [nbas, nbas] {
        return Err(FerricError::General(format!(
            "cosmo_reaction_field: density shape {:?} != ({nbas},{nbas})",
            density.shape()
        )));
    }
    config.validate()?;

    let n_seg = cavity.segments.len();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let nsh = prep.nshells();

    // Per-segment solute potential v_k = V_nuc(s_k) + V_elec(s_k), and the
    // per-segment AO "M matrices" contracted with the density inline (we
    // don't need to materialize each M_k separately — only its contraction
    // with D for v_k, and later q_k * M_k summed for V_reaction).
    let mut v = Array1::<f64>::zeros(n_seg);
    // Accumulate V_reaction directly is wasteful before we know q; instead
    // cache each segment's dense M_k matrix is too memory-heavy for large
    // bases * many segments, so we do two passes: pass 1 computes v (via
    // D-contraction only, no matrix storage); pass 2 (after solving for q)
    // recomputes M_k on the fly and accumulates -q_k * M_k into V_reaction.
    // This trades some redundant integral evaluation for O(nbasis^2) memory
    // instead of O(n_seg * nbasis^2).
    let mut engine = Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14)
        .map_err(|e| FerricError::General(format!("cosmo: engine init failed: {e}")))?;

    for (k, seg) in cavity.segments.iter().enumerate() {
        // Point charge of +1 at the segment position: libint's nuclear
        // operator returns <mu| -Z/|r-R| |nu>; with Z=+1 that's -M_k, so
        // v_elec = -sum D*(-M_k) block = +sum D*block... concretely:
        // V_elec(s_k) = -integral rho(r)/|r-s_k| dr = -sum_uv D_uv T_uv
        // where T_uv = <mu|1/|r-s_k||nu>. The engine with Z=+1 returns
        // M_uv = <mu|-1/|r-s_k|nu> = -T_uv, so V_elec = +sum D_uv M_uv.
        let probe = [CAtom {
            atomic_number: 1.0,
            x: seg.pos[0],
            y: seg.pos[1],
            z: seg.pos[2],
        }];
        let rc = unsafe {
            ffi::scf_engine_set_point_charges(engine.handle_mut(), probe.as_ptr(), probe.len() as std::os::raw::c_int)
        };
        if rc < 0 {
            return Err(FerricError::General(format!(
                "cosmo_reaction_field: set_point_charges failed (rc={rc}) for segment {k}"
            )));
        }

        let mut v_elec = 0.0_f64;
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                let block = engine.compute_1e_block(prep, s1, s2);
                let n1 = dims[s1];
                let n2 = dims[s2];
                let o1 = offs[s1];
                let o2 = offs[s2];
                if s1 == s2 {
                    for i in 0..n1 {
                        for j in 0..n2 {
                            v_elec += density[(o1 + i, o2 + j)] * block[i * n2 + j];
                        }
                    }
                } else {
                    for i in 0..n1 {
                        for j in 0..n2 {
                            v_elec += 2.0 * density[(o1 + i, o2 + j)] * block[i * n2 + j];
                        }
                    }
                }
            }
        }

        // Nuclear contribution: V_nuc(s_k) = sum_A Z_A_eff / |s_k - R_A|
        // (real atoms only; ghosts contribute zero effective charge).
        let mut v_nuc = 0.0_f64;
        for atom in &mol.atoms {
            if atom.ghost {
                continue;
            }
            let dx = seg.pos[0] - atom.x;
            let dy = seg.pos[1] - atom.y;
            let dz = seg.pos[2] - atom.zpos;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-10 {
                return Err(FerricError::General(format!(
                    "cosmo_reaction_field: segment {k} coincides with an atom center"
                )));
            }
            v_nuc += atom.effective_z() as f64 / r;
        }

        v[k] = v_nuc + v_elec;
    }

    // Solve S q = -f(eps) v for the apparent surface charges.
    let s_mat = cavity.build_s_matrix();
    let rhs = -config.f_epsilon() * &v;
    let charges = s_mat.solve(&rhs).map_err(|e| {
        FerricError::Lapack(format!(
            "cosmo: segment-interaction-matrix solve failed (singular S — degenerate cavity?): {e}"
        ))
    })?;

    // E_cosmo = 1/2 sum_k q_k v_k.
    let e_cosmo = 0.5 * charges.dot(&v);

    // Pass 2: V_reaction_uv = -sum_k q_k * M_k_uv (M_k as defined above,
    // <mu|-1/|r-s_k||nu> with Z=+1... wait: M_k = engine output with Z=+1 =
    // <mu|-1/|r-s_k||nu>. We want the electron-charge interaction operator
    // for a REAL charge q_k at s_k: <mu| -q_k/|r-s_k| |nu> = q_k * M_k.
    // Summed over segments: V_reaction = sum_k q_k * M_k.
    let mut v_reaction = Array2::<f64>::zeros((nbas, nbas));
    for (k, seg) in cavity.segments.iter().enumerate() {
        let qk = charges[k];
        if qk == 0.0 {
            continue;
        }
        let probe = [CAtom {
            atomic_number: 1.0,
            x: seg.pos[0],
            y: seg.pos[1],
            z: seg.pos[2],
        }];
        let rc = unsafe {
            ffi::scf_engine_set_point_charges(engine.handle_mut(), probe.as_ptr(), probe.len() as std::os::raw::c_int)
        };
        if rc < 0 {
            return Err(FerricError::General(format!(
                "cosmo_reaction_field: set_point_charges failed (rc={rc}) for segment {k} (pass 2)"
            )));
        }
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                let block = engine.compute_1e_block(prep, s1, s2);
                let n1 = dims[s1];
                let n2 = dims[s2];
                let o1 = offs[s1];
                let o2 = offs[s2];
                for i in 0..n1 {
                    for j in 0..n2 {
                        let val = qk * block[i * n2 + j];
                        v_reaction[(o1 + i, o2 + j)] += val;
                        if s1 != s2 {
                            v_reaction[(o2 + j, o1 + i)] += val;
                        }
                    }
                }
            }
        }
    }

    Ok(CosmoResult {
        v_reaction,
        e_cosmo,
        charges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::mol::Molecule;

    fn water() -> Molecule {
        Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap()
    }

    #[test]
    fn bondi_radius_known_elements() {
        // H = 1.20 Angstrom, O = 1.52 Angstrom (Bondi 1964).
        let rh = bondi_radius_bohr(1).unwrap();
        let ro = bondi_radius_bohr(8).unwrap();
        assert!((rh - 1.20 * ANGSTROM_TO_BOHR).abs() < 1e-10);
        assert!((ro - 1.52 * ANGSTROM_TO_BOHR).abs() < 1e-10);
    }

    #[test]
    fn bondi_radius_unsupported_element_errors() {
        // No fabricated fallback: an element with no tabulated radius must
        // hard-error, e.g. Z=118 (Oganesson, absurdly out of scope).
        let err = bondi_radius_bohr(118);
        assert!(err.is_err());
    }

    #[test]
    fn f_epsilon_matches_pyscf_cosmo_convention() {
        // f(eps) = (eps-1)/(eps+0.5); water eps=78.39 -> ~0.9873.
        let cfg = CosmoConfig { epsilon: 78.39, ..Default::default() };
        let f = cfg.f_epsilon();
        let expected = (78.39 - 1.0) / (78.39 + 0.5);
        assert!((f - expected).abs() < 1e-12);
        assert!(f > 0.98 && f < 1.0, "f(eps) for water should be close to 1 (near-conductor limit), got {f}");
    }

    #[test]
    fn f_epsilon_vacuum_limit_is_zero() {
        // eps -> 1 (vacuum): f(eps) -> 0, no screening.
        let cfg = CosmoConfig { epsilon: 1.0 + 1e-9, ..Default::default() };
        assert!(cfg.f_epsilon().abs() < 1e-6);
    }

    #[test]
    fn config_rejects_invalid_epsilon() {
        let cfg = CosmoConfig { epsilon: 0.5, ..Default::default() };
        assert!(cfg.validate().is_err());
        let cfg2 = CosmoConfig { epsilon: f64::NAN, ..Default::default() };
        assert!(cfg2.validate().is_err());
    }

    #[test]
    fn config_rejects_invalid_lebedev_order() {
        let cfg = CosmoConfig { lebedev_order: 7, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn cavity_build_water_nonzero_segments() {
        let mol = water();
        let cfg = CosmoConfig::default();
        let cavity = CosmoCavity::build(&mol, &cfg).unwrap();
        assert!(cavity.n_segments() > 0);
        assert!(cavity.total_area() > 0.0);
        // Sanity: total area should be less than the sum of full sphere
        // areas (some area is buried at the O-H bonds) but not by too much
        // for a small bent triatomic (most of each sphere stays exposed).
        let r_o = bondi_radius_bohr(8).unwrap() * cfg.radius_scale;
        let r_h = bondi_radius_bohr(1).unwrap() * cfg.radius_scale;
        let full_area = 4.0 * std::f64::consts::PI * (r_o * r_o + 2.0 * r_h * r_h);
        assert!(cavity.total_area() < full_area);
        assert!(cavity.total_area() > 0.5 * full_area);
    }

    #[test]
    fn cavity_build_ghost_only_errors() {
        let mut mol = water();
        for atom in mol.atoms.iter_mut() {
            atom.ghost = true;
        }
        let cfg = CosmoConfig::default();
        assert!(CosmoCavity::build(&mol, &cfg).is_err());
    }

    #[test]
    fn s_matrix_symmetric_and_diagonal_positive() {
        let mol = water();
        let cfg = CosmoConfig { lebedev_order: 26, ..Default::default() };
        let cavity = CosmoCavity::build(&mol, &cfg).unwrap();
        let s = cavity.build_s_matrix();
        let n = cavity.n_segments();
        for k in 0..n {
            assert!(s[(k, k)] > 0.0, "diagonal must be positive");
            for l in 0..n {
                assert!((s[(k, l)] - s[(l, k)]).abs() < 1e-12, "S not symmetric at ({k},{l})");
            }
        }
    }
}
