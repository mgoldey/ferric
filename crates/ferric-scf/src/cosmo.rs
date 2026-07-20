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
//!    [`ferric_dft::lebedev`]); each grid point's area is down-weighted by a
//!    smooth switching function of its distance into any neighboring atom's
//!    sphere (SWIG-style, Lange & Herbert, J. Chem. Phys. 133, 244111
//!    (2010), eq. 3.19 -- the same scheme PySCF's `pcm.py` uses), rather than
//!    a hard binary keep/discard cut. See `switch_h`/`switching_weight`
//!    below.
//! 2. **Segment interaction matrix** `S` (called `A` in some papers), default
//!    formulation [`SMatrixKind::GaussianSmeared`] (PySCF `pcm.py`
//!    convention, Li/Scalmani/Frisch, J. Chem. Phys. 122, 194110 (2005)):
//!    each segment is a Gaussian charge distribution of width `xi_k` set by
//!    the LOCAL Lebedev grid density, giving off-diagonal `S_kl = erf(xi_kl *
//!    r_kl) / r_kl` (`xi_kl` the harmonic-combined width of segments k,l) and
//!    diagonal `S_kk = xi_k * sqrt(2/pi) / switch_fun_k`. The original
//!    [`SMatrixKind::PointCharge`] formulation is still available (for two
//!    distinct segments k != l, `S_kl = 1 / |s_k - s_l|`, bare Coulomb
//!    interaction between point charges at the segment centers; diagonal
//!    `S_kk = xi * sqrt(4*pi / a_k)`, `xi = 3.8`, derived by treating each
//!    segment as a small disk of area `a_k`, Klamt & Schuurmann eq. 7) but is
//!    no longer the default — see "Point-charge vs Gaussian-smeared segment
//!    representation" below for the measured effect of this choice.
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
//! * **Point-charge vs Gaussian-smeared segment representation
//!   (RESOLVED 2026-07-19)**: the `S`-matrix now defaults to
//!   [`SMatrixKind::GaussianSmeared`] (PySCF `pcm.py` convention). This was
//!   the single largest lever found across two investigation rounds on this
//!   module: on water/cc-pVDZ/eps=78.39 it moved the self-consistent
//!   solvation energy from -3.39 kcal/mol (old `PointCharge` default, ~43%
//!   of a from-scratch-reproduced PySCF SWIG-COSMO target of -5.94 kcal/mol)
//!   to -5.96 kcal/mol (~0.4% off that target) — bigger than the SWIG
//!   switching-function change (<0.5% effect, see the cavity section above).
//!   Two caveats worth recording precisely: (1) `V`/`V_reaction` (the
//!   solute<->segment potential and reaction-field integrals below) still
//!   use bare nuclear-attraction-type point-charge integrals, NOT PySCF's
//!   Gaussian-smeared `int3c2e`-based potential — only the `S`-matrix
//!   construction was ported, and that alone was already enough to close
//!   the gap to <1%, so the V/V_reaction smearing was NOT pursued (would add
//!   real complexity — a finite-exponent point-charge integral kernel ferric
//!   does not currently have — for a lever that measured near-negligible
//!   incremental value here). (2) An EARLIER same-day investigation
//!   documented a different pair of reference numbers (-9.30 kcal/mol
//!   target, -10.6 kcal/mol isolated-comparison result) that did NOT
//!   reproduce when independently re-derived for this pass, via two
//!   different methods (PySCF's own `PCM` class API, and a from-scratch
//!   port of `pcm.py`'s `gen_surface`/`get_D_S`) that agree with each other
//!   to <10% (-5.94 vs -5.41 kcal/mol, the latter being the closer bare-V/
//!   smeared-S isolation point) — see `crates/ferric-scf/tests/cosmo_water.rs`
//!   for the full methodology. The cavity itself (228 segments, 166.6819084
//!   Bohr^2 total area) is confirmed bit-for-bit identical between ferric
//!   and the re-derivation, ruling out a cavity mismatch as the source of
//!   that earlier discrepancy.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi::{self, CAtom};
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

/// COSMO self-term prefactor (Klamt & Schuurmann 1993, eq. 7): `S_kk = xi *
/// sqrt(4*pi / a_k)`. This is the standard constant quoted throughout the
/// COSMO literature for a segment modeled as a small circular disk. Retained
/// as the bare-point-charge `S`-matrix formula (see [`SMatrixKind::PointCharge`]);
/// superseded by default with [`SMatrixKind::GaussianSmeared`] below.
const COSMO_XI: f64 = 3.8;

/// Per-Lebedev-order Gaussian-charge-width prefactor `XI[ng]` (Table II, Li &
/// Frisch/Scalmani, "Continuous Surface Charge Polarizable Continuum Models",
/// J. Chem. Phys. 122, 194110 (2005), as used by PySCF's `pcm.py`). Each
/// segment's Gaussian width is `xi_k = XI[ng] / (r_vdw_k * sqrt(w_k))`, `w_k`
/// the segment's *unnormalized* Lebedev weight (PySCF convention: sums to
/// `4*pi` over a sphere, i.e. `4*pi * ferric's normalized weight`). Only the
/// six Lebedev orders this module supports (see [`CosmoConfig::validate`])
/// are tabulated; an order outside this set is rejected by config
/// validation before this table is ever consulted.
fn gaussian_xi_table(lebedev_order: usize) -> f64 {
    match lebedev_order {
        6 => 4.84566077868,
        14 => 4.86458714334,
        26 => 4.85478226219,
        50 => 4.89250673295,
        110 => 4.90101060987,
        302 => 4.90498088169,
        _ => unreachable!(
            "gaussian_xi_table: lebedev_order {lebedev_order} not in the validated set \
             {{6,14,26,50,110,302}} -- CosmoConfig::validate should have rejected this earlier"
        ),
    }
}

/// Which `S`-matrix (segment-interaction) formulation to use.
///
/// * [`SMatrixKind::PointCharge`] -- the original ferric formulation: each
///   segment is a bare point charge (`S_kl = 1/|s_k - s_l|` off-diagonal,
///   `S_kk = xi_COSMO * sqrt(4*pi/a_k)` diagonal, `xi_COSMO = 3.8` fixed).
/// * [`SMatrixKind::GaussianSmeared`] -- PySCF `pcm.py`'s formulation: each
///   segment is a Gaussian charge distribution of width `xi_k` set by the
///   LOCAL grid density (`gaussian_xi_table`), giving `S_kl = erf(xi_kl *
///   r_kl) / r_kl` off-diagonal (`xi_kl` the harmonic-sum combined width of
///   segments k,l) and a density-dependent diagonal `S_kk = xi_k *
///   sqrt(2/pi) / switch_fun_k`. This is the default (see
///   [`CosmoConfig::default`]); `PointCharge` is kept only for the
///   regression test that measures the isolated S-matrix effect and for any
///   future need to reproduce the pre-2026-07-19 numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SMatrixKind {
    PointCharge,
    GaussianSmeared,
}

impl Default for SMatrixKind {
    fn default() -> Self {
        SMatrixKind::GaussianSmeared
    }
}

/// SWIG (switching/Gaussian) transition-zone shape parameter (Lange &
/// Herbert, JCP 133, 244111 (2010), eq. 3.19): the quintic smoothstep
/// `h(x) = x^3 (10 - 15x + 6x^2)`, `h(x)=0` for `x<=0`, `h(x)=1` for `x>=1`.
/// This is a faithful port of PySCF's `pcm.py::switch_h` (same functional
/// form; see the `switch_h_*` unit tests below for direct value checks).
fn switch_h(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)
    }
}

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
    /// Segment-interaction (`S`-matrix) formulation. Default
    /// [`SMatrixKind::GaussianSmeared`] (PySCF `pcm.py` convention, added
    /// 2026-07-19 -- see the module doc-comment's "Point-charge segment
    /// representation" section for the measured effect of this choice).
    /// [`SMatrixKind::PointCharge`] reproduces the original bare-point-charge
    /// formula this module shipped with.
    #[serde(default)]
    pub s_matrix_kind: SMatrixKind,
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
            s_matrix_kind: SMatrixKind::default(),
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
    /// Gaussian charge-distribution width `xi_k` for this segment (only
    /// meaningful for [`SMatrixKind::GaussianSmeared`]; computed
    /// unconditionally at build time since it's cheap and needed by the
    /// diagonal self-term too). `xi_k = XI[ng] / (r_vdw * sqrt(w_k))`, PySCF
    /// `gen_surface` convention (`w_k` = unnormalized Lebedev weight, sums to
    /// `4*pi`).
    charge_exp: f64,
    /// Raw switching-function value at this segment's center (`swf` in
    /// PySCF, `h`-product over all other spheres) BEFORE folding into area —
    /// needed separately by the Gaussian-smeared diagonal self-term
    /// (`S_kk = xi_k * sqrt(2/pi) / switch_fun_k`).
    switch_fun: f64,
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
    /// scaled-Bondi radii, using a SWIG-style smooth switching function
    /// (Lange & Herbert, JCP 133, 244111 (2010), eq. 3.19 — the same scheme
    /// PySCF's `pcm.py` uses for `method="COSMO"`) instead of a hard
    /// point-in-sphere visibility trim: a grid point on atom A's sphere gets
    /// its area scaled by `prod_{B != A} h(d_AB)`, where `d_AB` measures how
    /// far the point sits into (or out of) atom B's switching zone
    /// (`h`=[`switch_h`]). A point that is far outside every other sphere
    /// keeps its full area (`h -> 1`); a point deep inside another sphere is
    /// smoothly suppressed to zero area (`h -> 0`) rather than discarded in
    /// one hard step. Points whose total switching weight underflows
    /// (`< 1e-16`, matching PySCF's own cutoff) are dropped as an
    /// optimization only — this is a numerical floor, not a physics cutoff.
    /// Ghost atoms (basis-only, zero nuclear charge) are excluded from the
    /// cavity — they carry no physical volume to screen.
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
        let n_grid = unit_pts.len() as f64;

        // Per-atom switching-zone geometry (PySCF gen_surface / Lange &
        // Herbert eq. 3.19-3.21): the transition band width R_sw scales with
        // the LOCAL Lebedev point density (sqrt(14/N)) so a finer angular
        // grid gets a narrower (more resolved) switching zone, and R_in is
        // the radius at which the switching function starts to turn on.
        let r_sw: Vec<f64> = real_atoms
            .iter()
            .map(|&(_, _, r)| r * (14.0 / n_grid).sqrt())
            .collect();
        let r_in: Vec<f64> = real_atoms
            .iter()
            .zip(r_sw.iter())
            .map(|(&(_, _, r), &rsw)| {
                let ratio = r / rsw;
                let alpha = 0.5 + ratio - (ratio * ratio - 1.0 / 28.0).sqrt();
                r - alpha * rsw
            })
            .collect();

        let xi_prefactor = gaussian_xi_table(config.lebedev_order);

        let mut segments = Vec::new();
        for &(ia, center_a, r_a) in real_atoms.iter() {
            let sphere_area = 4.0 * std::f64::consts::PI * r_a * r_a;
            for (pt, w) in unit_pts.iter().zip(weights.iter()) {
                let p = [
                    center_a[0] + r_a * pt[0],
                    center_a[1] + r_a * pt[1],
                    center_a[2] + r_a * pt[2],
                ];
                // PySCF's `w` is the Lebedev weight in ITS convention, which
                // sums to 4*pi over a sphere (ferric's `lebedev()` sums to 1
                // -- see `ferric_dft::lebedev` doc-comment); convert once
                // here so `xi_k = XI[ng] / (r_vdw * sqrt(w_pyscf))` matches
                // `gen_surface`'s `xi = XI[ng] / (r_vdw * w**0.5)` exactly.
                let w_pyscf = w * 4.0 * std::f64::consts::PI;
                let charge_exp = xi_prefactor / (r_a * w_pyscf.sqrt());
                // Smooth switching weight: product of h(d) over every OTHER
                // real atom's sphere (own-atom factor is exactly 1, matching
                // PySCF's diJ[:, ia] = 1.0).
                let mut swf = 1.0_f64;
                for (ib_idx, &(ib, center_b, _)) in real_atoms.iter().enumerate() {
                    if ib == ia {
                        continue;
                    }
                    let dx = p[0] - center_b[0];
                    let dy = p[1] - center_b[1];
                    let dz = p[2] - center_b[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    let d = (dist - r_in[ib_idx]) / r_sw[ib_idx];
                    // PySCF clamps sub-1e-8 differences to exactly 0 before
                    // calling switch_h — a numerical tidiness step, not part
                    // of the physical switching shape.
                    let d = if d.abs() < 1e-8 { 0.0 } else { d };
                    swf *= switch_h(d);
                    if swf == 0.0 {
                        break;
                    }
                }
                // Numerical floor matching PySCF's `w*swf > 1e-16` keep
                // criterion — points below this contribute negligible area
                // and would otherwise bloat the segment count for free.
                if w * swf <= 1e-16 {
                    continue;
                }
                segments.push(Segment {
                    pos: p,
                    // Lebedev weights sum to 1 over the sphere, so w*sphere_area
                    // is exactly the area subtended by this node; swf smoothly
                    // down-scales it near a neighboring sphere's boundary.
                    area: w * sphere_area * swf,
                    charge_exp,
                    switch_fun: swf,
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

    /// Build the symmetric segment-interaction matrix `S` (n_seg x n_seg),
    /// per `kind`:
    ///
    /// * [`SMatrixKind::PointCharge`]: off-diagonal bare Coulomb
    ///   `1/|s_k - s_l|`, diagonal COSMO self-term `COSMO_XI *
    ///   sqrt(4*pi/a_k)` (the module's original formula).
    /// * [`SMatrixKind::GaussianSmeared`]: off-diagonal `erf(xi_kl * r_kl) /
    ///   r_kl` where `xi_kl = xi_k*xi_l / sqrt(xi_k^2 + xi_l^2)` is the
    ///   harmonic-combined Gaussian width of the two segments, diagonal
    ///   `xi_k * sqrt(2/pi) / switch_fun_k` -- both exactly PySCF
    ///   `pcm.py::get_D_S`'s formula.
    fn build_s_matrix(&self, kind: SMatrixKind) -> Array2<f64> {
        let n = self.segments.len();
        let mut s = Array2::<f64>::zeros((n, n));
        match kind {
            SMatrixKind::PointCharge => {
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
            }
            SMatrixKind::GaussianSmeared => {
                for k in 0..n {
                    let xi_k = self.segments[k].charge_exp;
                    let swf_k = self.segments[k].switch_fun;
                    s[(k, k)] = xi_k * (2.0 / std::f64::consts::PI).sqrt() / swf_k;
                    for l in 0..k {
                        let xi_l = self.segments[l].charge_exp;
                        let dx = self.segments[k].pos[0] - self.segments[l].pos[0];
                        let dy = self.segments[k].pos[1] - self.segments[l].pos[1];
                        let dz = self.segments[k].pos[2] - self.segments[l].pos[2];
                        let r = (dx * dx + dy * dy + dz * dz).sqrt();
                        let xi_kl = xi_k * xi_l / (xi_k * xi_k + xi_l * xi_l).sqrt();
                        let v = unsafe { erf(xi_kl * r) } / r;
                        s[(k, l)] = v;
                        s[(l, k)] = v;
                    }
                }
            }
        }
        s
    }
}

// libm's `erf` (C99) is already transitively linked into every ferric binary
// (libint2/OpenBLAS both pull in libm), so bind it directly instead of
// adding a new crate dependency or a lower-precision hand-rolled
// approximation -- full ~1e-16 double precision, exactly matching what
// PySCF's `scipy.special.erf` (itself backed by the same C library family)
// produces to numerical noise.
extern "C" {
    fn erf(x: f64) -> f64;
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
    let s_mat = cavity.build_s_matrix(config.s_matrix_kind);
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
    fn switch_h_boundary_values() {
        assert_eq!(switch_h(-0.5), 0.0);
        assert_eq!(switch_h(0.0), 0.0);
        assert_eq!(switch_h(1.0), 1.0);
        assert_eq!(switch_h(1.5), 1.0);
    }

    #[test]
    fn switch_h_midpoint_and_monotonic() {
        // h(0.5) = 0.5^3 * (10 - 7.5 + 1.5) = 0.125 * 4.0 = 0.5 (symmetric
        // quintic smoothstep is centered at x=0.5).
        assert!((switch_h(0.5) - 0.5).abs() < 1e-12);
        // Monotonically non-decreasing on [0,1].
        let mut prev = switch_h(0.0);
        for i in 1..=20 {
            let x = i as f64 / 20.0;
            let cur = switch_h(x);
            assert!(cur >= prev - 1e-12, "switch_h not monotonic at x={x}");
            prev = cur;
        }
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
    fn s_matrix_symmetric_and_diagonal_positive_point_charge() {
        let mol = water();
        let cfg = CosmoConfig { lebedev_order: 26, ..Default::default() };
        let cavity = CosmoCavity::build(&mol, &cfg).unwrap();
        let s = cavity.build_s_matrix(SMatrixKind::PointCharge);
        let n = cavity.n_segments();
        for k in 0..n {
            assert!(s[(k, k)] > 0.0, "diagonal must be positive");
            for l in 0..n {
                assert!((s[(k, l)] - s[(l, k)]).abs() < 1e-12, "S not symmetric at ({k},{l})");
            }
        }
    }

    #[test]
    fn s_matrix_symmetric_and_diagonal_positive_gaussian_smeared() {
        let mol = water();
        let cfg = CosmoConfig { lebedev_order: 26, ..Default::default() };
        let cavity = CosmoCavity::build(&mol, &cfg).unwrap();
        let s = cavity.build_s_matrix(SMatrixKind::GaussianSmeared);
        let n = cavity.n_segments();
        for k in 0..n {
            assert!(s[(k, k)] > 0.0, "diagonal must be positive");
            for l in 0..n {
                assert!((s[(k, l)] - s[(l, k)]).abs() < 1e-12, "S not symmetric at ({k},{l})");
            }
        }
    }
}
