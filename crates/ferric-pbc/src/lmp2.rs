//! Stage 8b: Gamma-point amplitude-threshold local MP2 (Rust port of
//! `reference/pbc/pbc_lmp2.py`; FINDINGS "Iteration 5 (Python, Gamma LMP2)",
//! recommendations 1-5 of "For the Rust port onto ferric's amplitude_lmp2").
//!
//! The solver half is ferric-mp2's molecular amplitude-threshold machinery,
//! reused unchanged: the swap-closed Eq-8 pair blocks
//! ([`ferric_mp2::ragged::pair_block_from_g_cand_gated`]), the ragged
//! per-pair PCG ([`ferric_mp2::ragged::solve_ragged`]), the Hylleraas energy
//! ([`ferric_mp2::lmp2_amplitude::hylleraas_energy`]), the R⁻⁶ pair gate
//! ([`ferric_mp2::lmp2_amplitude::pair_gate_keep_with`]) and the VV-HV
//! orthonormalisations (`canonical_orth`, `lowdin`, `pivoted_cholesky_order`).
//! What periodicity changes is implemented here:
//!
//! 1. **Localisation: Berghold/Resta, not Boys.** At Gamma the orbitals are
//!    supercell-periodic; `⟨μ|r|ν⟩` is not a periodic operator and the L = 0
//!    molecular dipole integrals see the cell boundary. We maximise
//!    `Σ_k w_k Σ_i |z_k,ii|²`, `z_k = ⟨φ_i| e^{i b_k·r} |φ_j⟩`, `b_k` the
//!    cell's reciprocal vectors, `w_k = (|a_k|/2π)²` (orthorhombic cells ONLY:
//!    general cells are refused, not silently mis-weighted). The AO matrix of
//!    `e^{i b_k·r}` is the lattice-summed pair FT at `G = −b_k`
//!    ([`crate::pair_ft::pair_ft`]); `Re z_k`, `Im z_k` are real symmetric, so
//!    the Jacobi sweep is Boys' with six weighted "coordinates"
//!    ([`berghold_localize`]). Centroids are phases,
//!    `f_k = arg(z_k,ii)/2π (mod 1)`; spreads `Ω_i = Σ_k w_k (1 − |z_k,ii|²)`
//!    (Bohr², the Wannier spread up to `O((bσ)⁴)`).
//! 2. **VV-HV virtuals** from the lattice-summed `S` (the SCF's) and the
//!    lattice-summed obs × minimal-basis cross overlap (pair FT at G = 0 on a
//!    combined basis); valence virtuals Berghold-localised, hard-virtual
//!    selection weights `1/Ω`. Everything else is the molecular recipe.
//! 3. **Every distance is minimum-image** ([`MinImage`]): the pair gate, the
//!    optional pair cutoff and the fit domains. The Eq-8 integral mask needs no
//!    distance (Gamma integrals already sum every image of j).
//!    [`PeriodicDistance::RawCartesianMutant`] exists ONLY as the negative
//!    control proving the tests can see the difference — which they cannot on
//!    a supercell with two cells along every axis (each raw distance already
//!    IS a minimum image; prototype finding), hence the >= 3-cell test systems.
//! 4. **Integrals**: `(ia|jb) = Σ_k B_k,ia B_k,jb` from the SCF's own RS-GDF
//!    `B` (global), or a per-pair domain-local fit in the PERIODIC metric
//!    (`J_ij = A_i,D J2_DD⁺ A_j,Dᵀ`, eig/`lindep` pseudo-inverse,
//!    [`crate::rsgdf::PeriodicFitParts`]); the domain is every aux function
//!    within `fit_radius_bohr` (minimum image) of either occupied centroid. At
//!    the trivial radius the fit IS the global `B·B` (anchor, tested).
//! 5. **Denominators**: `F_oo`, `F_vv` from the reference Fock, plus exactly
//!    the occupied shift of [`crate::mp2::occupied_shift`] (same convention
//!    table as `gamma_mp2`; `F_vv` is unshifted because `C_vᵀ S D S C_v = 0`).
//!
//! # Exactness anchor
//!
//! `eps = 0` (full mask, no gate/cutoff) must equal [`crate::mp2::gamma_mp2`]
//! on the same B and the same shifted denominators: canonical orbitals +
//! closed-form denominators vs Berghold LMOs + periodic VV-HV + ragged CG,
//! sharing only B and F (`tests/pbc_lmp2.rs`, 1e-10). The anchor is BLIND to
//! localisation (unitary invariance); translation equivalence of the LMOs,
//! virtuals and pair energies is its guard (also tested, with a molecule
//! wrapped across the boundary, and with molecular Boys as the negative
//! control).
//!
//! # KNOWN LIMITATION (Gamma): the eps gate is volume-limited, not local
//!
//! At Gamma every occupied pair carries a distance-INDEPENDENT coupling:
//! the G → 0 components of j's image lattice are a uniform field, so far pairs
//! have `(ia|jb) → −(4π/Ω_cell) μ_ia μ_jb` (needle supercells; `(4π/3Ω) μ·μ`
//! in cubic ones). With the DEFAULT gate ([`GammaEpsGate::RawIntegral`]) the
//! Eq-8 mask therefore keeps ALL N² pairs until
//! `Ω_cell > 4π μ*²/eps` (measured onsets bracket the prediction for
//! eps = 1e-2, 3e-3, 1e-3; below the onset no locality statement can be
//! made in either direction). Consequences for a caller of that gate:
//! * at fixed `eps`, `pairs_kept == nocc²` for every supercell below the onset
//!   (pinned by `eps_gate_keeps_every_pair_below_the_uniform_field_onset` in
//!   `tests/pbc_lmp2.rs`, which also asserts the flip under A-drop);
//! * the truncation error past the onset is `a N + b` with `b = O(1)` the
//!   uniform-field energy of the dropped pairs — report per-molecule errors
//!   with their `a + b/N` split, never as a single number;
//! * a minimum-image distance cutoff (`pair_cutoff_bohr`) IS local (partners
//!   `2⌊R_c/a⌋+1` once the supercell exceeds `2 R_c`).
//!
//! # The fix, OPT-IN: [`GammaEpsGate::UniformHeadRestored`] ("A-drop")
//!
//! FINDINGS "Iteration 5b": the uniform coupling is exactly MINUS the q = 0
//! head that the G = 0-dropped kernel omits (a quadrature hole of weight 1/N,
//! a finite-size artifact like Madelung, not correlation that survives the
//! thermodynamic limit). A-drop restores it in the integrals,
//! `J' = J + (4π/Ω_sc) μ μᵀ` — a positive rank-1 update, i.e. ONE extra RI
//! column `B_extra,ia = √(4π/Ω_sc) μ_ia` ([`UniformHead`]) — and then gates
//! AND solves on `J'`. Far pairs lose their coupling, so the eps gate becomes
//! local (prototype: 3 partners/molecule at eps 1e-4 from N = 4 to 32, where
//! the raw gate keeps all N²). There is nothing to add back.
//! * `μ_ia` along the needle axis from the Resta matrix at `b` AND `2b`,
//!   Richardson `(4 f(b) − f(2b))/3` (O(b⁴); single-b is O(b²) and ~20×
//!   worse on the far-pair residual at 1x1x8).
//! * **Its eps = 0 target is NOT canonical Gamma MP2.** It is `E_head`, the
//!   closed-form MP2 on the head-augmented B with the unchanged (shifted)
//!   Fock ([`mp2_closed_form_local`]), which differs from canonical Gamma MP2
//!   by the uniform term (prototype +2.955e-3 Ha at 1x1x4, +2.890e-3 at
//!   1x1x32; `E_head` is also the better TDL estimator). The result reports it
//!   as `e_corr_target`; `e_corr_canonical` stays canonical Gamma MP2.
//! * **Needle supercells only** ([`needle_axis`]): the head is `μ·W·μ` with
//!   `W = ẑẑ` only when every small-|G| reciprocal vector lies along one axis.
//!   Other shapes (cubic `W = I/3`, Iteration 3's c3) are REFUSED with an
//!   error until their W is derived and measured.
//! * The Fock-side heads (FINDINGS 5b item 3: `F_vv`, diagonal `F_oo`) are
//!   NOT implemented; they are a finite-size correction on top of the LMP2.
//! * To reproduce a canonical Gamma number instead, the recipe is a
//!   pair-level screen plus the analytic add-back (FINDINGS 5b item 4) — also
//!   NOT implemented here; never combine an element-level `J'` gate with a
//!   pair-level add-back.
//!
//! **Default unchanged** ([`GammaLmp2Config::shifted`] keeps
//! [`GammaEpsGate::RawIntegral`]): switching would silently move every eps = 0
//! result off canonical Gamma MP2 and would make every non-needle supercell an
//! error. The hook remains [`GammaEpsGate::gate_quantity`]; for A-drop the
//! block it receives already contains the head (the gate and the solve read
//! the same `J'`).
//!
//! # Not implemented / not measured
//!
//! Non-orthorhombic cells (Silvestrelli weights), k-points, open shell, cost
//! or scaling claims (every block is assembled serially from a GLOBAL B; the
//! domain fit reads a resident `J3`), finite-radius fit accuracy, the
//! uniform head for non-needle supercells, the Fock-side heads.
//!
//! Units: Bohr and Hartree.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::ewald::madelung_constant;
use crate::lattice::Cell;
use crate::mp2::{
    b_ov_from_ao_b, gamma_mp2, occupied_shift, GammaMp2Config, GammaMp2Integrals, Mp2Denominators,
};
use crate::pair_ft::pair_ft;
use crate::rsgdf::{PeriodicFitParts, RsGdf};
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_mp2::lmp2_amplitude::{
    canonical_orth, hylleraas_energy, lowdin, pair_gate_keep_with, pivoted_cholesky_order,
};
use ferric_mp2::ragged::{pair_block_from_g_cand_gated, solve_ragged, PairBlock, Ragged};
use ferric_mp2::rimp2::active_occ;
use ferric_scf::result::{ScfResult, Spin};
use ndarray::{s, Array1, Array2, Array3, ArrayView1};
use ndarray_linalg::{Eigh, Solve, UPLO};
use num_complex::Complex64;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;

/// How distances between centroids / aux centres are measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeriodicDistance {
    /// Minimum-image convention (production).
    MinimumImage,
    /// MUTATION: plain Cartesian distance of the stored positions. Exists
    /// only as the negative control of the minimum-image tests.
    RawCartesianMutant,
}

/// What the Eq-8 amplitude-threshold test acts on, and which integrals are
/// solved with (module doc, "KNOWN LIMITATION" and "The fix, OPT-IN").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GammaEpsGate {
    /// DEFAULT. Keep `(i,a,j,b)` iff `|(ia|jb)| > eps` or `|(ib|ja)| > eps`
    /// on the raw Gamma integrals — the molecular rule. eps = 0 target:
    /// canonical Gamma MP2. At Gamma this keeps every pair below the
    /// uniform-field onset `Ω_cell ~ 4π μ*²/eps`.
    RawIntegral,
    /// "A-drop" (FINDINGS Iteration 5b): restore the omitted q = 0 head,
    /// `J' = J + (4π/Ω_sc) μ μᵀ` (one extra RI column, [`UniformHead`]), and
    /// gate AND solve on `J'`. eps = 0 target: `E_head` =
    /// [`mp2_closed_form_local`] on the head-augmented B, NOT canonical Gamma
    /// MP2 (they differ by the uniform term). Needle supercells only
    /// ([`needle_axis`]); anything else is refused.
    UniformHeadRestored,
}

impl GammaEpsGate {
    /// Whether the integrals themselves carry the restored q = 0 head.
    pub fn restores_uniform_head(&self) -> bool {
        matches!(self, GammaEpsGate::UniformHeadRestored)
    }

    /// THE HOOK: the `(nv, nv)` quantity whose elements the Eq-8 test
    /// compares to `eps` for the ordered pair (i, j), given the integral
    /// block `g[a,b]` the amplitudes are solved with. Both variants borrow
    /// `g` unchanged: for `RawIntegral` `g = (ia|jb)` (bitwise the molecular
    /// rule); for `UniformHeadRestored` `g` is already `J'` (the head is in
    /// the integral source), so gate and solve read the same tensor — which
    /// is what makes the A-drop error purely eps-controlled with nothing to
    /// add back. A gate that differed from the solved integrals (FINDINGS
    /// 5b "A-add") would need a pair-level add-back and is NOT clean.
    pub fn gate_quantity<'g>(
        &self,
        _i: usize,
        _j: usize,
        g: &'g Array2<f64>,
    ) -> Cow<'g, Array2<f64>> {
        match self {
            GammaEpsGate::RawIntegral | GammaEpsGate::UniformHeadRestored => Cow::Borrowed(g),
        }
    }
}

/// Berghold/Resta Jacobi settings.
#[derive(Debug, Clone, Copy)]
pub struct LocalizationConfig {
    /// Maximum Jacobi sweeps; not converging is an error.
    pub max_sweeps: usize,
    /// Convergence: largest rotation angle of a sweep below this (radians).
    pub angle_tol: f64,
    /// `Some(seed)`: start from `C Q` with a random orthogonal `Q` (to test
    /// that the maximum is start-independent). `None`: canonical start.
    pub random_start_seed: Option<u64>,
}

impl Default for LocalizationConfig {
    fn default() -> Self {
        Self {
            max_sweeps: 1000,
            angle_tol: 1e-11,
            random_start_seed: None,
        }
    }
}

/// Settings for [`gamma_lmp2`]. No `Default`: the reference exxdiv must be
/// stated (as for [`crate::mp2::GammaMp2Config`]).
#[derive(Debug, Clone, Copy)]
pub struct GammaLmp2Config {
    /// Single amplitude threshold ε (Eq-8). 0 = full mask (anchor limit).
    pub eps: f64,
    /// Core orbitals excluded from correlation (validated by `active_occ`).
    pub frozen_core: usize,
    /// The exxdiv the RHF reference was converged with.
    pub reference_exxdiv: ExxDiv,
    /// Denominator convention (the physical one is `MadelungShifted`).
    pub denominators: Mp2Denominators,
    pub cg_rtol: f64,
    pub cg_max_iter: usize,
    /// Integral-free R⁻⁶ pair gate constant (molecular `pair_gate_cal`),
    /// with MINIMUM-IMAGE centroid distances and `σ_i = √Ω_i`. `None` = off.
    /// UNCALIBRATED at Gamma (the prototype never measured it); off by default.
    pub pair_gate_cal: Option<f64>,
    /// Minimum-image occupied-centroid pair cutoff (Bohr): pairs farther
    /// apart are dropped whole. `None` = off.
    pub pair_cutoff_bohr: Option<f64>,
    /// Per-pair domain-local fit radius (Bohr) in the periodic metric;
    /// requires [`GammaLmp2Inputs::fit`]. `None` = global B.
    pub fit_radius_bohr: Option<f64>,
    /// The Eq-8 gate quantity (module doc, KNOWN LIMITATION). Default
    /// ([`GammaLmp2Config::shifted`]) is [`GammaEpsGate::RawIntegral`],
    /// deliberately unchanged by the A-drop port: `UniformHeadRestored`
    /// changes the eps = 0 target and refuses non-needle supercells.
    pub eps_gate: GammaEpsGate,
    /// Distance convention for every distance above.
    pub distance: PeriodicDistance,
    pub localization: LocalizationConfig,
    /// Also run canonical [`gamma_mp2`] on the same B (honesty printout;
    /// `e_corr_canonical` is NaN when false).
    pub compute_reference: bool,
    /// Memory budget in bytes (`None` = ferric's unified budget).
    pub budget_bytes: Option<usize>,
}

impl GammaLmp2Config {
    /// Shifted denominators, threshold `eps`, no gate/cutoff/domain fit,
    /// minimum-image distances, canonical-start localisation, reference on.
    pub fn shifted(reference_exxdiv: ExxDiv, eps: f64) -> Self {
        Self {
            eps,
            frozen_core: 0,
            reference_exxdiv,
            denominators: Mp2Denominators::MadelungShifted,
            cg_rtol: 1e-11,
            cg_max_iter: 400,
            pair_gate_cal: None,
            pair_cutoff_bohr: None,
            fit_radius_bohr: None,
            eps_gate: GammaEpsGate::RawIntegral,
            distance: PeriodicDistance::MinimumImage,
            localization: LocalizationConfig::default(),
            compute_reference: true,
            budget_bytes: None,
        }
    }

    fn validate(&self) -> Result<(), FerricError> {
        let bad = |what: String| -> Result<(), FerricError> {
            Err(FerricError::General(format!("gamma_lmp2: {what}")))
        };
        if !(self.eps >= 0.0) || !self.eps.is_finite() {
            return bad(format!("eps must be finite and >= 0, got {}", self.eps));
        }
        if !(self.cg_rtol > 0.0) || self.cg_max_iter == 0 {
            return bad(format!(
                "cg_rtol must be > 0 and cg_max_iter >= 1 (got {}, {})",
                self.cg_rtol, self.cg_max_iter
            ));
        }
        for (name, v) in [
            ("pair_gate_cal", self.pair_gate_cal),
            ("pair_cutoff_bohr", self.pair_cutoff_bohr),
            ("fit_radius_bohr", self.fit_radius_bohr),
        ] {
            if let Some(x) = v {
                if !(x > 0.0) || !x.is_finite() {
                    return bad(format!("{name} must be finite and > 0, got {x}"));
                }
            }
        }
        if self.localization.max_sweeps == 0 || !(self.localization.angle_tol > 0.0) {
            return bad("localization needs max_sweeps >= 1 and angle_tol > 0".into());
        }
        Ok(())
    }
}

/// Everything [`gamma_lmp2`] reads besides the cell, reference and config.
#[derive(Clone, Copy)]
pub struct GammaLmp2Inputs<'a> {
    /// Orbital basis, built from `cell.mol()` (the SCF's).
    pub obs: &'a PreparedBasis,
    /// The `BasisSet` `obs` was built from (for the combined cross overlap).
    pub obs_bs: &'a BasisSet,
    /// Minimal basis for the valence virtuals (e.g. bundled `sto-3g`).
    pub minimal_bs: &'a BasisSet,
    /// The SCF's RS-GDF (its `B` and its lattice overlap).
    pub gdf: &'a RsGdf,
    /// Pre-solve fit pieces ([`RsGdf::build_with_fit_parts`]); required iff
    /// `fit_radius_bohr` is set.
    pub fit: Option<&'a PeriodicFitParts>,
}

// ---------------------------------------------------------------------------
// Geometry: minimum image
// ---------------------------------------------------------------------------

/// Distance under a [`PeriodicDistance`] convention for one cell.
#[derive(Debug, Clone)]
pub struct MinImage {
    a: [[f64; 3]; 3],
    b: [[f64; 3]; 3],
    metric: PeriodicDistance,
}

impl MinImage {
    pub fn new(cell: &Cell, metric: PeriodicDistance) -> Self {
        Self {
            a: *cell.lattice(),
            b: cell.reciprocal(),
            metric,
        }
    }

    /// `|d|` under the convention: reduce to fractional `f − round(f)`, then
    /// take the shortest of the 27 neighbouring images (exact for any cell
    /// whose reduced vectors are not pathologically skewed).
    pub fn distance(&self, d: [f64; 3]) -> f64 {
        let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if self.metric == PeriodicDistance::RawCartesianMutant {
            return norm(d);
        }
        let mut f = [0.0; 3];
        for (k, fk) in f.iter_mut().enumerate() {
            let x = (d[0] * self.b[k][0] + d[1] * self.b[k][1] + d[2] * self.b[k][2]) / (2.0 * PI);
            *fk = x - x.round();
        }
        let mut best = f64::INFINITY;
        for n0 in -1i32..=1 {
            for n1 in -1i32..=1 {
                for n2 in -1i32..=1 {
                    let g = [f[0] + n0 as f64, f[1] + n1 as f64, f[2] + n2 as f64];
                    let mut v = [0.0; 3];
                    for (x, vx) in v.iter_mut().enumerate() {
                        *vx = g[0] * self.a[0][x] + g[1] * self.a[1][x] + g[2] * self.a[2][x];
                    }
                    best = best.min(norm(v));
                }
            }
        }
        best
    }

    pub fn between(&self, p: [f64; 3], q: [f64; 3]) -> f64 {
        self.distance([p[0] - q[0], p[1] - q[1], p[2] - q[2]])
    }
}

/// `(n, n)` matrix of centroid distances under `metric`.
pub fn centroid_distances(
    cell: &Cell,
    centers: &[[f64; 3]],
    metric: PeriodicDistance,
) -> Array2<f64> {
    let mi = MinImage::new(cell, metric);
    let n = centers.len();
    Array2::from_shape_fn((n, n), |(i, j)| mi.between(centers[i], centers[j]))
}

// ---------------------------------------------------------------------------
// Resta / Berghold operator and localisation
// ---------------------------------------------------------------------------

/// `Z_k = ⟨μ| e^{i b_k·r} |ν⟩` (lattice-summed) split into real symmetric
/// parts, with the Berghold weights.
#[derive(Debug, Clone)]
pub struct RestaOperator {
    /// `Re Z_k`, symmetrised.
    pub re: [Array2<f64>; 3],
    /// `Im Z_k`, symmetrised.
    pub im: [Array2<f64>; 3],
    /// `w_k = (|a_k|/2π)²`.
    pub weights: [f64; 3],
    lattice: [[f64; 3]; 3],
}

/// Build the Resta matrices for `cell` in the basis `obs` (pair FT at
/// `G = −b_k`). Refuses non-orthorhombic cells (the weights would be wrong).
pub fn resta_operator(cell: &Cell, obs: &PreparedBasis) -> Result<RestaOperator, FerricError> {
    let a = *cell.lattice();
    check_orthorhombic(&a)?;
    let b = cell.reciprocal();
    let gs: Vec<[f64; 3]> = b.iter().map(|bk| [-bk[0], -bk[1], -bk[2]]).collect();
    let p = pair_ft(cell, obs, &gs)?;
    let n = obs.nbasis();
    let mk = |k: usize, im: bool| resta_part(&p, k, n, im);
    let w = |k: usize| (vec_len(&a[k]) / (2.0 * PI)).powi(2);
    Ok(RestaOperator {
        re: [mk(0, false), mk(1, false), mk(2, false)],
        im: [mk(0, true), mk(1, true), mk(2, true)],
        weights: [w(0), w(1), w(2)],
        lattice: a,
    })
}

fn vec_len(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Refuse non-orthorhombic lattices (Berghold weights, needle geometry).
fn check_orthorhombic(a: &[[f64; 3]; 3]) -> Result<(), FerricError> {
    for i in 0..3 {
        for j in (i + 1)..3 {
            let d = a[i][0] * a[j][0] + a[i][1] * a[j][1] + a[i][2] * a[j][2];
            if d.abs() > 1e-10 * vec_len(&a[i]) * vec_len(&a[j]) {
                return Err(FerricError::General(format!(
                    "gamma_lmp2: Berghold/Resta weights are implemented for orthorhombic cells \
                     only (a_{i}·a_{j} = {d:.3e}); general cells need Silvestrelli's weights"
                )));
            }
        }
    }
    Ok(())
}

/// Symmetrised real (`im = false`) or imaginary part of the pair FT column
/// `k` of `p`: `½ (P_mn + P_nm)`.
fn resta_part(p: &Array3<Complex64>, k: usize, n: usize, im: bool) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(m, v)| {
        let (x, y) = (p[(m, v, k)], p[(v, m, k)]);
        if im {
            0.5 * (x.im + y.im)
        } else {
            0.5 * (x.re + y.re)
        }
    })
}

/// `(Re Z, Im Z)` of `Z = ⟨μ| e^{i m b_axis·r} |ν⟩` (lattice-summed, pair FT
/// at `G = −m b_axis`, the same convention and symmetrisation as
/// [`resta_operator`]; `mult = 1` reproduces its `re/im[axis]`). Used for the
/// `(b, 2b)` Richardson dipoles of [`uniform_head`].
pub fn resta_matrix_at(
    cell: &Cell,
    obs: &PreparedBasis,
    axis: usize,
    mult: u32,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    if axis > 2 || mult == 0 {
        return Err(FerricError::General(format!(
            "resta_matrix_at: axis must be 0..=2 and mult >= 1 (got {axis}, {mult})"
        )));
    }
    let b = cell.reciprocal()[axis];
    let m = mult as f64;
    let p = pair_ft(cell, obs, &[[-m * b[0], -m * b[1], -m * b[2]]])?;
    let n = obs.nbasis();
    Ok((resta_part(&p, 0, n, false), resta_part(&p, 0, n, true)))
}

// ---------------------------------------------------------------------------
// Uniform (q = 0 head) coupling: FINDINGS Iteration 5b, "A-drop"
// ---------------------------------------------------------------------------

/// Smallest supercell aspect ratio `|a_needle| / max |a_⊥|` accepted as a
/// needle by [`needle_axis`]. A structural proxy, NOT a derived accuracy
/// bound: `W = ẑẑ` needs every small-|G| reciprocal vector along one axis,
/// and 4 is the smallest aspect the prototype measured (1x1x4 of a cubic
/// primitive; 1x1x4..32 in Iteration 5b).
pub const NEEDLE_MIN_ASPECT: f64 = 4.0;

/// The needle axis of an orthorhombic supercell whose one lattice vector is
/// at least [`NEEDLE_MIN_ASPECT`] times longer than both others. Every other
/// shape is an ERROR: the uniform head is `μ·W·μ` with a shape-dependent
/// depolarisation tensor `W`, derived and measured only for the needle
/// (`W = ẑẑ`); cubic supercells have `W = I/3` (Iteration 3's c3) but no
/// supercell sweep was run for them, so they are refused, not guessed.
pub fn needle_axis(cell: &Cell) -> Result<usize, FerricError> {
    let a = cell.lattice();
    check_orthorhombic(a)?;
    let len = [vec_len(&a[0]), vec_len(&a[1]), vec_len(&a[2])];
    let k = (0..3)
        .max_by(|&x, &y| len[x].total_cmp(&len[y]))
        .unwrap_or(2);
    let perp = (0..3)
        .filter(|&x| x != k)
        .map(|x| len[x])
        .fold(0.0_f64, f64::max);
    if len[k] < NEEDLE_MIN_ASPECT * perp {
        return Err(FerricError::General(format!(
            "gamma_lmp2: the uniform-head (A-drop) gate is implemented for NEEDLE supercells \
             only (one axis >= {NEEDLE_MIN_ASPECT} x the other two, W = zz); lattice lengths \
             {:.4} / {:.4} / {:.4} Bohr. Other shapes need their own depolarisation tensor W \
             (cubic: I/3), which is not implemented — use GammaEpsGate::RawIntegral",
            len[0], len[1], len[2]
        )));
    }
    Ok(k)
}

/// The restored q = 0 head for one set of localised spaces: `J'_ij =
/// J_ij + scale · μ_i μ_jᵀ`, i.e. the extra RI column `√scale · μ`.
#[derive(Debug, Clone)]
pub struct UniformHead {
    /// Needle axis (lattice-vector index).
    pub axis: usize,
    /// `4π / Ω_sc` (Bohr⁻³).
    pub scale: f64,
    /// Transition dipoles `μ_ia` along the needle, `(no, nv)`: Richardson
    /// `(4 f(b) − f(2b)) / 3` of the Resta estimate (O(b⁴)).
    pub mu: Array2<f64>,
    /// `f(b) = Im(z_ia e^{−i arg z_ii}) / b` alone (O(b²); diagnostics and
    /// the accuracy test only — never used for the integrals).
    pub mu_single_b: Array2<f64>,
    /// `f(2b)` (diagnostics).
    pub mu_double_b: Array2<f64>,
}

impl UniformHead {
    /// The extra RI column `√scale · μ`, `(no, nv)` (row i = `B_extra,i·`).
    pub fn column(&self) -> Array2<f64> {
        let r = self.scale.sqrt();
        self.mu.mapv(|x| r * x)
    }

    /// The uniform coupling `J_unif(i, j)[a, b] = −scale μ_ia μ_jb` that the
    /// raw Gamma integrals carry at every pair (far pairs: `J → J_unif`).
    pub fn coupling(&self, i: usize, j: usize) -> Array2<f64> {
        uniform_coupling(&self.mu, self.scale, i, j)
    }
}

/// `−scale · μ_i μ_jᵀ` for any dipole estimate `mu` (`(no, nv)`).
pub fn uniform_coupling(mu: &Array2<f64>, scale: f64, i: usize, j: usize) -> Array2<f64> {
    let nv = mu.ncols();
    Array2::from_shape_fn((nv, nv), |(a, b)| -scale * mu[(i, a)] * mu[(j, b)])
}

/// Resta transition-dipole estimate `f(b)_ia = Im(z_ia e^{−i arg z_ii}) / b`
/// (the `_resta_dipole` of the prototype).
fn resta_dipoles(
    c_occ: &Array2<f64>,
    c_vir: &Array2<f64>,
    re: &Array2<f64>,
    im: &Array2<f64>,
    b: f64,
) -> Array2<f64> {
    let (no, nv) = (c_occ.ncols(), c_vir.ncols());
    let (zr_o, zi_o) = (re.dot(c_occ), im.dot(c_occ));
    let zr_ia = c_occ.t().dot(&re.dot(c_vir));
    let zi_ia = c_occ.t().dot(&im.dot(c_vir));
    let ph: Vec<f64> = (0..no)
        .map(|i| {
            let ci = c_occ.column(i);
            ci.dot(&zi_o.column(i)).atan2(ci.dot(&zr_o.column(i)))
        })
        .collect();
    // Im[(zr + i zi) e^{−i ph}] = zi cos ph − zr sin ph
    Array2::from_shape_fn((no, nv), |(i, a)| {
        (zi_ia[(i, a)] * ph[i].cos() - zr_ia[(i, a)] * ph[i].sin()) / b
    })
}

/// [`UniformHead`] for `spaces` on a needle supercell (refuses anything
/// else, [`needle_axis`]): Resta matrices at `b` and `2b` along the needle,
/// Richardson dipoles, `scale = 4π/Ω_sc`. Valid while the LMO spread is
/// small against the needle length (the Resta estimate's premise).
pub fn uniform_head(
    cell: &Cell,
    obs: &PreparedBasis,
    spaces: &GammaLocalSpaces,
) -> Result<UniformHead, FerricError> {
    let axis = needle_axis(cell)?;
    if spaces.c_occ.nrows() != obs.nbasis() || spaces.c_vir.nrows() != obs.nbasis() {
        return Err(FerricError::General(format!(
            "uniform_head: spaces have {} AO rows, obs {} functions",
            spaces.c_occ.nrows(),
            obs.nbasis()
        )));
    }
    let b = vec_len(&cell.reciprocal()[axis]);
    let (re1, im1) = resta_matrix_at(cell, obs, axis, 1)?;
    let f1 = resta_dipoles(&spaces.c_occ, &spaces.c_vir, &re1, &im1, b);
    drop((re1, im1));
    let (re2, im2) = resta_matrix_at(cell, obs, axis, 2)?;
    let f2 = resta_dipoles(&spaces.c_occ, &spaces.c_vir, &re2, &im2, 2.0 * b);
    let mu = Array2::from_shape_fn(f1.dim(), |(i, a)| (4.0 * f1[(i, a)] - f2[(i, a)]) / 3.0);
    if mu.iter().any(|x| !x.is_finite()) {
        return Err(FerricError::General(
            "uniform_head: non-finite transition dipole".into(),
        ));
    }
    Ok(UniformHead {
        axis,
        scale: 4.0 * PI / cell.volume(),
        mu,
        mu_single_b: f1,
        mu_double_b: f2,
    })
}

/// Closed-form MP2 of a LOCAL-basis RI tensor: `b_ov[P, i·nv + a]` (plus the
/// optional extra column `head_column[i, a]`, e.g. [`UniformHead::column`])
/// with Fock blocks `f_oo`, `f_vv` (shift included by the caller), rotated
/// to their eigenbases: `E = Σ J(2J − Jᵀ) / (f_i + f_j − f_a − f_b)`.
/// The independent reference for the eps = 0 anchor of any gate (no mask,
/// no CG, no ragged blocks); with `head_column = None` and canonical-span
/// spaces it equals [`gamma_mp2`] (unitary invariance, tested).
pub fn mp2_closed_form_local(
    b_ov: &Array2<f64>,
    head_column: Option<&Array2<f64>>,
    f_oo: &Array2<f64>,
    f_vv: &Array2<f64>,
) -> Result<f64, FerricError> {
    let (no, nv) = (f_oo.nrows(), f_vv.nrows());
    if f_oo.ncols() != no
        || f_vv.ncols() != nv
        || b_ov.ncols() != no * nv
        || head_column.is_some_and(|h| h.dim() != (no, nv))
    {
        return Err(FerricError::General(format!(
            "mp2_closed_form_local: B {:?}, head {:?}, F_oo {:?}, F_vv {:?} inconsistent",
            b_ov.dim(),
            head_column.map(|h| h.dim()),
            f_oo.dim(),
            f_vv.dim()
        )));
    }
    let (eo, uo) = eigh(f_oo, "closed-form F_oo")?;
    let (ev, uv) = eigh(f_vv, "closed-form F_vv")?;
    let nrow = b_ov.nrows() + usize::from(head_column.is_some());
    let mut bc = Array2::<f64>::zeros((nrow, no * nv));
    let rotate_into = |bc: &mut Array2<f64>, p: usize, m: Array2<f64>| {
        let r = uo.t().dot(&m).dot(&uv);
        for i in 0..no {
            for a in 0..nv {
                bc[(p, i * nv + a)] = r[(i, a)];
            }
        }
    };
    for p in 0..b_ov.nrows() {
        let m = Array2::from_shape_fn((no, nv), |(i, a)| b_ov[(p, i * nv + a)]);
        rotate_into(&mut bc, p, m);
    }
    if let Some(h) = head_column {
        rotate_into(&mut bc, nrow - 1, h.clone());
    }
    let mut e = 0.0;
    for i in 0..no {
        let bi = bc.slice(s![.., i * nv..(i + 1) * nv]);
        for j in 0..no {
            let bj = bc.slice(s![.., j * nv..(j + 1) * nv]);
            let g = bi.t().dot(&bj);
            for a in 0..nv {
                for b in 0..nv {
                    let d = eo[i] + eo[j] - ev[a] - ev[b];
                    if !(d < 0.0) {
                        return Err(FerricError::General(format!(
                            "mp2_closed_form_local: non-negative denominator {d:.3e} at \
                             ({i},{a},{j},{b})"
                        )));
                    }
                    e += g[(a, b)] * (2.0 * g[(a, b)] - g[(b, a)]) / d;
                }
            }
        }
    }
    Ok(e)
}

impl RestaOperator {
    /// `(Re z_k,ii, Im z_k,ii)` for one orbital column.
    fn diag(&self, c: ArrayView1<'_, f64>) -> [(f64, f64); 3] {
        std::array::from_fn(|k| (c.dot(&self.re[k].dot(&c)), c.dot(&self.im[k].dot(&c))))
    }

    /// Periodic centroid `Σ_k (arg z_k,ii / 2π mod 1) a_k` (Bohr).
    pub fn centroid(&self, c: ArrayView1<'_, f64>) -> [f64; 3] {
        let z = self.diag(c);
        let mut out = [0.0; 3];
        for (k, (re, im)) in z.iter().enumerate() {
            let f = (im.atan2(*re) / (2.0 * PI)).rem_euclid(1.0);
            for (x, o) in out.iter_mut().enumerate() {
                *o += f * self.lattice[k][x];
            }
        }
        out
    }

    /// Resta spread `Ω = Σ_k w_k (1 − |z_k,ii|²)` (Bohr²).
    pub fn spread(&self, c: ArrayView1<'_, f64>) -> f64 {
        self.diag(c)
            .iter()
            .zip(self.weights)
            .map(|((re, im), w)| w * (1.0 - (re * re + im * im)))
            .sum()
    }
}

/// Outcome of [`berghold_localize`].
#[derive(Debug, Clone)]
pub struct LocalizationInfo {
    /// Functional `Σ_c w_c Σ_i X_c,ii²` at the start / at the end.
    pub f_initial: f64,
    pub f_final: f64,
    pub sweeps: usize,
    /// max over i<j of |∂f/∂θ_ij| at the end (0 at a stationary point).
    pub max_gradient: f64,
    pub converged: bool,
}

/// Deterministic random orthogonal matrix (LCG + modified Gram-Schmidt).
fn random_orthogonal(n: usize, seed: u64) -> Array2<f64> {
    let mut st = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    let mut rnd = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let mut q = Array2::from_shape_fn((n, n), |_| rnd());
    for j in 0..n {
        for k in 0..j {
            let d = q.column(j).dot(&q.column(k));
            let ck = q.column(k).to_owned();
            q.column_mut(j).scaled_add(-d, &ck);
        }
        let nrm = q.column(j).dot(&q.column(j)).sqrt();
        q.column_mut(j).mapv_inplace(|x| x / nrm);
    }
    q
}

/// Maximise `f = Σ_c w_c Σ_i (C_iᵀ M_c C_i)²` over rotations of the columns
/// of `c` by 2×2 Jacobi sweeps, `M_c` = the six matrices `Re Z_k`, `Im Z_k`
/// with weights `w_k` (Berghold/Resta). Port of `pbc_lmp2.jacobi_localize`:
/// `Δf(θ) = A (1 − cos 4θ) + B sin 4θ`, `A = Σ w [X_ij² − (X_ii − X_jj)²/4]`,
/// `B = Σ w X_ij (X_ii − X_jj)`, optimum `4θ = atan2(B, −A)` with
/// `c_i ← cos θ c_i + sin θ c_j`, `c_j ← −sin θ c_i + cos θ c_j`.
/// Not converging within `max_sweeps` is an error.
pub fn berghold_localize(
    c: &Array2<f64>,
    op: &RestaOperator,
    cfg: &LocalizationConfig,
) -> Result<(Array2<f64>, LocalizationInfo), FerricError> {
    let n = c.ncols();
    let mut c = c.clone();
    if let (Some(seed), true) = (cfg.random_start_seed, n > 1) {
        c = c.dot(&random_orthogonal(n, seed));
    }
    let mats: Vec<(&Array2<f64>, f64)> = (0..3)
        .flat_map(|k| [(&op.re[k], op.weights[k]), (&op.im[k], op.weights[k])])
        .collect();
    let w: Vec<f64> = mats.iter().map(|m| m.1).collect();
    let mut x: Vec<Array2<f64>> = mats.iter().map(|(m, _)| c.t().dot(&m.dot(&c))).collect();
    let fval = |x: &[Array2<f64>]| -> f64 {
        x.iter()
            .zip(&w)
            .map(|(xc, wc)| wc * (0..n).map(|i| xc[(i, i)] * xc[(i, i)]).sum::<f64>())
            .sum()
    };
    let f_initial = fval(x.as_slice());
    let mut sweeps = 0usize;
    let mut converged = n <= 1;
    while !converged && sweeps < cfg.max_sweeps {
        sweeps += 1;
        let mut tmax = 0.0_f64;
        for i in 0..n.saturating_sub(1) {
            for j in (i + 1)..n {
                let (mut aa, mut bb) = (0.0, 0.0);
                for (xc, wc) in x.iter().zip(&w) {
                    let (xij, dd) = (xc[(i, j)], xc[(i, i)] - xc[(j, j)]);
                    aa += wc * (xij * xij - 0.25 * dd * dd);
                    bb += wc * xij * dd;
                }
                if aa.hypot(bb) < 1e-15 {
                    continue;
                }
                let t = 0.25 * bb.atan2(-aa);
                if t.abs() < 1e-15 {
                    continue;
                }
                tmax = tmax.max(t.abs());
                let (cs, sn) = (t.cos(), t.sin());
                for mu in 0..c.nrows() {
                    let (ci, cj) = (c[(mu, i)], c[(mu, j)]);
                    c[(mu, i)] = cs * ci + sn * cj;
                    c[(mu, j)] = -sn * ci + cs * cj;
                }
                for xc in x.iter_mut() {
                    for p in 0..n {
                        let (xi, xj) = (xc[(i, p)], xc[(j, p)]);
                        xc[(i, p)] = cs * xi + sn * xj;
                        xc[(j, p)] = -sn * xi + cs * xj;
                    }
                    for p in 0..n {
                        let (xi, xj) = (xc[(p, i)], xc[(p, j)]);
                        xc[(p, i)] = cs * xi + sn * xj;
                        xc[(p, j)] = -sn * xi + cs * xj;
                    }
                }
            }
        }
        if tmax < cfg.angle_tol {
            converged = true;
        }
    }
    let mut max_gradient = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let g: f64 = x
                .iter()
                .zip(&w)
                .map(|(xc, wc)| 4.0 * wc * xc[(i, j)] * (xc[(i, i)] - xc[(j, j)]))
                .sum();
            max_gradient = max_gradient.max(g.abs());
        }
    }
    let info = LocalizationInfo {
        f_initial,
        f_final: fval(x.as_slice()),
        sweeps,
        max_gradient,
        converged,
    };
    if !converged {
        return Err(FerricError::General(format!(
            "gamma_lmp2: Berghold localisation did not converge in {} sweeps (|grad| {:.2e})",
            cfg.max_sweeps, info.max_gradient
        )));
    }
    Ok((c, info))
}

// ---------------------------------------------------------------------------
// Periodic VV-HV virtuals
// ---------------------------------------------------------------------------

/// Lattice-summed `⟨obs μ | minimal ν⟩`, `(nao, nmin)`: pair FT at G = 0 on
/// ONE combined basis (obs shells then minimal shells per element — the
/// molecular `cross_overlap_with_minimal` construction), sliced.
fn lattice_cross_overlap_with_minimal(
    cell: &Cell,
    obs: &PreparedBasis,
    obs_bs: &BasisSet,
    min_bs: &BasisSet,
) -> Result<Array2<f64>, FerricError> {
    let mol = cell.mol();
    let mut merged: HashMap<i32, Vec<Shell>> = HashMap::new();
    let mut n_obs_shells: HashMap<i32, usize> = HashMap::new();
    for atom in &mol.atoms {
        let z = atom.z;
        if merged.contains_key(&z) {
            continue;
        }
        let o = obs_bs
            .for_element(z)
            .ok_or_else(|| FerricError::Basis(format!("gamma_lmp2: no obs shells for Z={z}")))?;
        let m = min_bs.for_element(z).ok_or_else(|| {
            FerricError::Basis(format!("gamma_lmp2: no minimal shells for Z={z}"))
        })?;
        let mut v = o.to_vec();
        v.extend(m.iter().cloned());
        n_obs_shells.insert(z, o.len());
        merged.insert(z, v);
    }
    let merged_bs = BasisSet {
        name: "gamma-lmp2-obs+minimal".to_string(),
        shells: merged,
        ecps: obs_bs.ecps.clone(),
    };
    let comb = PreparedBasis::new(mol, &merged_bs)?;
    let p = pair_ft(cell, &comb, &[[0.0; 3]])?;
    let (mut obs_idx, mut min_idx) = (Vec::new(), Vec::new());
    let mut count = vec![0usize; mol.atoms.len()];
    for (sh, &ai) in comb.shell_to_atom().iter().enumerate() {
        let is_obs = count[ai] < n_obs_shells[&mol.atoms[ai].z];
        let off = comb.shell_offsets()[sh];
        for f in 0..comb.shell_dims()[sh] {
            if is_obs {
                obs_idx.push(off + f);
            } else {
                min_idx.push(off + f);
            }
        }
        count[ai] += 1;
    }
    if obs_idx.len() != obs.nbasis() {
        return Err(FerricError::General(format!(
            "gamma_lmp2: combined-basis bookkeeping mismatch ({} obs AOs vs {})",
            obs_idx.len(),
            obs.nbasis()
        )));
    }
    Ok(Array2::from_shape_fn(
        (obs_idx.len(), min_idx.len()),
        |(r, c)| p[(obs_idx[r], min_idx[c], 0)].re,
    ))
}

fn ao_to_atom(prep: &PreparedBasis) -> Vec<usize> {
    let mut out = vec![0usize; prep.nbasis()];
    for (sh, &ai) in prep.shell_to_atom().iter().enumerate() {
        let off = prep.shell_offsets()[sh];
        for f in 0..prep.shell_dims()[sh] {
            out[off + f] = ai;
        }
    }
    out
}

fn eigh(m: &Array2<f64>, who: &str) -> Result<(Array1<f64>, Array2<f64>), FerricError> {
    m.eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("gamma_lmp2 {who} eigh: {e}")))
}

struct Vvhv {
    c: Array2<f64>,
    n_l: usize,
    n_h: usize,
    loc: Option<LocalizationInfo>,
}

/// Periodic VV-HV ([WSHG23] §2.1 with the molecular rig's documented
/// deviations): projected minimal basis minus the occupied span,
/// Berghold-localised; hard virtuals by `1/Ω`-weighted pivoted Cholesky over
/// the projected AOs, Löwdin, per-atom pseudo-canonicalisation.
#[allow(clippy::too_many_arguments)]
fn periodic_vvhv(
    cell: &Cell,
    obs: &PreparedBasis,
    ints: &GammaLmp2Inputs<'_>,
    s: &Array2<f64>,
    fock: &Array2<f64>,
    c_occ_all: &Array2<f64>,
    nvir_can: usize,
    op: &RestaOperator,
    loc_cfg: &LocalizationConfig,
) -> Result<Vvhv, FerricError> {
    let nao = s.nrows();
    let nocc = c_occ_all.ncols();
    let s_x = lattice_cross_overlap_with_minimal(cell, obs, ints.obs_bs, ints.minimal_bs)?;
    let nmin = s_x.ncols();
    // T = S⁻¹ S_x (column by column; S is SPD)
    let mut t = Array2::<f64>::zeros((nao, nmin));
    for col in 0..nmin {
        let x = s
            .solve(&s_x.column(col).to_owned())
            .map_err(|e| FerricError::Lapack(format!("gamma_lmp2 S solve: {e}")))?;
        t.column_mut(col).assign(&x);
    }
    let q_occ = c_occ_all.dot(&c_occ_all.t().dot(s));
    let tv = &t - &q_occ.dot(&t);
    let n_l = nmin.checked_sub(nocc).ok_or_else(|| {
        FerricError::General(format!(
            "gamma_lmp2: minimal basis ({nmin}) smaller than nocc ({nocc})"
        ))
    })?;
    if n_l > nvir_can {
        return Err(FerricError::General(format!(
            "gamma_lmp2: {n_l} valence virtuals exceed the {nvir_can} canonical virtuals"
        )));
    }
    let (c_l, loc) = if n_l > 0 {
        let cl = canonical_orth(&tv, s, n_l)?;
        if n_l > 1 {
            let (cl, info) = berghold_localize(&cl, op, loc_cfg)?;
            (cl, Some(info))
        } else {
            (cl, None)
        }
    } else {
        (Array2::<f64>::zeros((nao, 0)), None)
    };
    let n_h = nvir_can - n_l;
    let c_h = if n_h > 0 {
        let mut c_e = Array2::<f64>::zeros((nao, nocc + n_l));
        c_e.slice_mut(s![.., ..nocc]).assign(c_occ_all);
        c_e.slice_mut(s![.., nocc..]).assign(&c_l);
        let mut x = Array2::<f64>::eye(nao);
        x -= &c_e.dot(&c_e.t().dot(s));
        let sx = s.dot(&x);
        let a2a = ao_to_atom(obs);
        let mut cols = Vec::new();
        for c in 0..nao {
            let nrm2 = x.column(c).dot(&sx.column(c));
            if nrm2 > 1e-8 {
                cols.push((c, nrm2.sqrt()));
            }
        }
        let ncand = cols.len();
        if ncand < n_h {
            return Err(FerricError::General(format!(
                "gamma_lmp2: {ncand} projected-AO candidates for {n_h} hard virtuals"
            )));
        }
        let mut xn = Array2::<f64>::zeros((nao, ncand));
        let mut parents = Vec::with_capacity(ncand);
        for (k, &(c, nrm)) in cols.iter().enumerate() {
            xn.column_mut(k).assign(&x.column(c).mapv(|v| v / nrm));
            parents.push(a2a[c]);
        }
        let w: Vec<f64> = (0..ncand)
            .map(|k| 1.0 / op.spread(xn.column(k)).max(1e-6))
            .collect();
        let wmax2 = w.iter().fold(0.0_f64, |m, &v| m.max(v)).powi(2);
        let mut ov = xn.t().dot(&s.dot(&xn));
        for r in 0..ncand {
            for c in 0..ncand {
                ov[(r, c)] *= w[r] * w[c] / wmax2;
            }
        }
        let piv = pivoted_cholesky_order(&ov, n_h)?;
        let mut sel = Array2::<f64>::zeros((nao, n_h));
        let mut sel_parents = Vec::with_capacity(n_h);
        for (k, &p) in piv.iter().enumerate() {
            sel.column_mut(k).assign(&xn.column(p));
            sel_parents.push(parents[p]);
        }
        let mut ch = lowdin(&sel, s)?;
        let fh = ch.t().dot(&fock.dot(&ch));
        let mut by_atom: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (k, &a) in sel_parents.iter().enumerate() {
            by_atom.entry(a).or_default().push(k);
        }
        for idx in by_atom.values() {
            let m = idx.len();
            let blk = Array2::from_shape_fn((m, m), |(r, c)| fh[(idx[r], idx[c])]);
            let (_, u) = eigh(&blk, "hard-virtual block")?;
            let old: Vec<Array1<f64>> = idx.iter().map(|&k| ch.column(k).to_owned()).collect();
            for (cnew, &k) in idx.iter().enumerate() {
                let mut acc = Array1::<f64>::zeros(nao);
                for (r, col) in old.iter().enumerate() {
                    acc.scaled_add(u[(r, cnew)], col);
                }
                ch.column_mut(k).assign(&acc);
            }
        }
        ch
    } else {
        Array2::<f64>::zeros((nao, 0))
    };
    let mut c = Array2::<f64>::zeros((nao, n_l + n_h));
    c.slice_mut(s![.., ..n_l]).assign(&c_l);
    c.slice_mut(s![.., n_l..]).assign(&c_h);
    Ok(Vvhv { c, n_l, n_h, loc })
}

/// (max |C_vᵀ S C_v − 1|, max deviation of the canonical-virtual overlap
/// `U = C_canᵀ S C_v` from orthogonal) — the construction half of the anchor.
fn check_virtuals(s: &Array2<f64>, c_v: &Array2<f64>, c_can: &Array2<f64>) -> (f64, f64) {
    let dev_eye = |m: &Array2<f64>| {
        let mut d = 0.0_f64;
        for ((r, c), v) in m.indexed_iter() {
            d = d.max((v - if r == c { 1.0 } else { 0.0 }).abs());
        }
        d
    };
    let o = c_v.t().dot(&s.dot(c_v));
    let u = c_can.t().dot(&s.dot(c_v));
    (
        dev_eye(&o),
        dev_eye(&u.dot(&u.t())).max(dev_eye(&u.t().dot(&u))),
    )
}

// ---------------------------------------------------------------------------
// Localised spaces
// ---------------------------------------------------------------------------

/// The localised problem before any 4-index work.
#[derive(Debug, Clone)]
pub struct GammaLocalSpaces {
    /// Berghold-localised ACTIVE occupieds, `(nao, no)`.
    pub c_occ: Array2<f64>,
    /// VV-HV virtuals, `(nao, nv)`: valence first, then hard.
    pub c_vir: Array2<f64>,
    pub n_valence: usize,
    pub n_hard: usize,
    /// `C_oᵀ F C_o + occ_shift·1` (shifted per the denominator table).
    pub f_oo: Array2<f64>,
    /// `C_vᵀ F C_v`.
    pub f_vv: Array2<f64>,
    /// Periodic centroids of the occupied / virtual orbitals (Bohr, in the
    /// cell's parallelepiped; compare only via minimum image).
    pub occ_centers: Vec<[f64; 3]>,
    pub vir_centers: Vec<[f64; 3]>,
    /// Resta spreads Ω_i of the occupieds (Bohr²).
    pub occ_spreads: Vec<f64>,
    pub madelung: f64,
    pub occ_shift: f64,
    pub frozen_core: usize,
    pub nocc_total: usize,
    pub loc_occ: LocalizationInfo,
    pub loc_valence_virtuals: Option<LocalizationInfo>,
    /// VV-HV construction checks (errors above 1e-8).
    pub vvhv_dev_orth: f64,
    pub vvhv_dev_span: f64,
}

impl GammaLocalSpaces {
    pub fn no(&self) -> usize {
        self.c_occ.ncols()
    }
    pub fn nv(&self) -> usize {
        self.c_vir.ncols()
    }
}

fn check_reference(cell: &Cell, rhf: &ScfResult) -> Result<usize, FerricError> {
    if !matches!(rhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "gamma_lmp2: closed-shell RHF reference required".into(),
        ));
    }
    if !rhf.converged {
        return Err(FerricError::General(
            "gamma_lmp2: the RHF reference did not converge".into(),
        ));
    }
    let nelec = cell.mol().nelec();
    if nelec <= 0 || nelec % 2 != 0 {
        return Err(FerricError::General(format!(
            "gamma_lmp2: closed shell needs an even, positive electron count (got {nelec})"
        )));
    }
    Ok((nelec / 2) as usize)
}

/// Berghold occupieds, periodic VV-HV virtuals, shifted Fock blocks,
/// centroids and spreads. Public so tests can mutate the spaces (e.g. drop a
/// hard virtual) and run [`gamma_lmp2_with_spaces`] on them.
pub fn gamma_localized_spaces(
    cell: &Cell,
    rhf: &ScfResult,
    ints: &GammaLmp2Inputs<'_>,
    cfg: &GammaLmp2Config,
) -> Result<GammaLocalSpaces, FerricError> {
    cfg.validate()?;
    let nocc_total = check_reference(cell, rhf)?;
    let no = active_occ(nocc_total, cfg.frozen_core)?;
    let c = rhf.mos_r();
    let (nao, nmo) = c.dim();
    let s = ints.gdf.overlap();
    let fock = rhf.fock_r();
    if ints.obs.nbasis() != nao || s.dim() != (nao, nao) || fock.dim() != (nao, nao) {
        return Err(FerricError::General(format!(
            "gamma_lmp2: obs has {} functions, S {:?}, F {:?}, C {:?} — inconsistent",
            ints.obs.nbasis(),
            s.dim(),
            fock.dim(),
            c.dim()
        )));
    }
    if nmo <= nocc_total {
        return Err(FerricError::General(format!(
            "gamma_lmp2: {nmo} MOs with {nocc_total} occupied: no virtuals"
        )));
    }
    let madelung = madelung_constant(cell)?;
    let occ_shift = occupied_shift(cfg.reference_exxdiv, cfg.denominators, madelung);

    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    ledger.reserve(
        &format!("Gamma LMP2 Resta matrices + VV-HV n×n work (nao = {nao})"),
        bytes_of((nao as u64).saturating_mul(nao as u64), 8 * 14),
    )?;
    let op = resta_operator(cell, ints.obs)?;
    let (c_occ, loc_occ) = berghold_localize(
        &c.slice(s![.., cfg.frozen_core..nocc_total]).to_owned(),
        &op,
        &cfg.localization,
    )?;
    let c_occ_all = c.slice(s![.., ..nocc_total]).to_owned();
    let vv = periodic_vvhv(
        cell,
        ints.obs,
        ints,
        s,
        fock,
        &c_occ_all,
        nmo - nocc_total,
        &op,
        &cfg.localization,
    )?;
    let c_can = c.slice(s![.., nocc_total..]).to_owned();
    let (dev_orth, dev_span) = check_virtuals(s, &vv.c, &c_can);
    if dev_orth > 1e-8 || dev_span > 1e-8 {
        return Err(FerricError::General(format!(
            "gamma_lmp2: VV-HV construction check failed (orth {dev_orth:.2e}, span {dev_span:.2e})"
        )));
    }
    let mut f_oo = c_occ.t().dot(&fock.dot(&c_occ));
    for i in 0..no {
        f_oo[(i, i)] += occ_shift;
    }
    let f_vv = vv.c.t().dot(&fock.dot(&vv.c));
    let occ_centers = (0..no).map(|i| op.centroid(c_occ.column(i))).collect();
    let occ_spreads = (0..no).map(|i| op.spread(c_occ.column(i))).collect();
    let vir_centers = (0..vv.c.ncols())
        .map(|a| op.centroid(vv.c.column(a)))
        .collect();
    Ok(GammaLocalSpaces {
        c_occ,
        c_vir: vv.c,
        n_valence: vv.n_l,
        n_hard: vv.n_h,
        f_oo,
        f_vv,
        occ_centers,
        vir_centers,
        occ_spreads,
        madelung,
        occ_shift,
        frozen_core: cfg.frozen_core,
        nocc_total,
        loc_occ,
        loc_valence_virtuals: vv.loc,
        vvhv_dev_orth: dev_orth,
        vvhv_dev_span: dev_span,
    })
}

// ---------------------------------------------------------------------------
// Pair integrals: global B or per-pair periodic-metric domain fit
// ---------------------------------------------------------------------------

enum PairSource {
    Global {
        b_ov: Array2<f64>,
    },
    Domain {
        /// `A[P, i·nv+a] = (ia|P)'`.
        a: Array2<f64>,
        j2: Array2<f64>,
        /// `in_r[i][P]`: aux P within the radius of occupied centroid i.
        in_r: Vec<Vec<bool>>,
        lindep: f64,
        radius: f64,
    },
}

/// `(ia|jb)` blocks in the localised basis.
pub struct GammaPairIntegrals {
    src: PairSource,
    /// The restored q = 0 head as one extra RI column `√scale · μ`,
    /// `(no, nv)` ([`GammaPairIntegrals::with_uniform_head`]); `None` = raw.
    head: Option<Array2<f64>>,
    no: usize,
    nv: usize,
    naux: usize,
}

impl GammaPairIntegrals {
    /// Global (`fit_radius = None`, rows of the SCF's B) or domain-local
    /// periodic-metric fit (`Some(r)`, needs `fit`). Distances by `metric`.
    pub fn new(
        cell: &Cell,
        spaces: &GammaLocalSpaces,
        gdf: &RsGdf,
        fit: Option<&PeriodicFitParts>,
        fit_radius: Option<f64>,
        metric: PeriodicDistance,
        budget_bytes: Option<usize>,
    ) -> Result<Self, FerricError> {
        let mut ledger = Ledger::new(crate::budget::resolve(budget_bytes));
        Self::new_on(cell, spaces, gdf, fit, fit_radius, metric, &mut ledger)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_on(
        cell: &Cell,
        spaces: &GammaLocalSpaces,
        gdf: &RsGdf,
        fit: Option<&PeriodicFitParts>,
        fit_radius: Option<f64>,
        metric: PeriodicDistance,
        ledger: &mut Ledger,
    ) -> Result<Self, FerricError> {
        let (no, nv) = (spaces.no(), spaces.nv());
        let nao = spaces.c_occ.nrows();
        let nov = no * nv;
        match fit_radius {
            None => {
                let naux = gdf.b().nrows();
                ledger.reserve(
                    &format!("Gamma LMP2 half-transform + B[k,ia] (naux = {naux}, nao = {nao}, nov = {nov})"),
                    bytes_of(naux as u64, (nao * nv + nov).saturating_mul(8)),
                )?;
                let b_ov = b_ov_from_ao_b(gdf.b(), nao, spaces.c_occ.view(), spaces.c_vir.view())?;
                Ok(Self {
                    src: PairSource::Global { b_ov },
                    head: None,
                    no,
                    nv,
                    naux,
                })
            }
            Some(radius) => {
                let parts = fit.ok_or_else(|| {
                    FerricError::General(
                        "gamma_lmp2: fit_radius_bohr needs the RS-GDF fit parts \
                         (RsGdf::build_with_fit_parts)"
                            .into(),
                    )
                })?;
                let naux = parts.j2.nrows();
                if parts.j3.dim() != (nao * nao, naux) || parts.aux_centers.len() != naux {
                    return Err(FerricError::General(format!(
                        "gamma_lmp2: fit parts J3 {:?} / {} aux centres inconsistent with nao = {nao}, naux = {naux}",
                        parts.j3.dim(),
                        parts.aux_centers.len()
                    )));
                }
                ledger.reserve(
                    &format!(
                        "Gamma LMP2 domain fit: J3ᵀ copy + half-transform + A[P,ia] \
                         (naux = {naux}, nao = {nao}, nov = {nov})"
                    ),
                    bytes_of(naux as u64, (nao * nao + nao * nv + nov).saturating_mul(8)),
                )?;
                let j3t = parts.j3.t().as_standard_layout().into_owned();
                let a = b_ov_from_ao_b(&j3t, nao, spaces.c_occ.view(), spaces.c_vir.view())?;
                drop(j3t);
                let mi = MinImage::new(cell, metric);
                let in_r = spaces
                    .occ_centers
                    .iter()
                    .map(|ci| {
                        parts
                            .aux_centers
                            .iter()
                            .map(|p| mi.between(*ci, *p) <= radius)
                            .collect()
                    })
                    .collect();
                Ok(Self {
                    src: PairSource::Domain {
                        a,
                        j2: parts.j2.clone(),
                        in_r,
                        lindep: parts.lindep,
                        radius,
                    },
                    head: None,
                    no,
                    nv,
                    naux,
                })
            }
        }
    }

    /// Aux functions (global) / aux functions available to domains; the
    /// extra head column is NOT counted here (see [`Self::has_uniform_head`]).
    pub fn naux(&self) -> usize {
        self.naux
    }

    /// Append the restored q = 0 head ("A-drop", FINDINGS Iteration 5b) as
    /// one extra RI column: every block becomes `J_ij + scale μ_i μ_jᵀ`, for
    /// the global B and the domain fit alike (the head is a property of the
    /// G = 0-dropped kernel, not of the fit). Errors on a shape mismatch or
    /// if a head was already applied (applying it twice would double it).
    pub fn with_uniform_head(mut self, head: &UniformHead) -> Result<Self, FerricError> {
        if head.mu.dim() != (self.no, self.nv) {
            return Err(FerricError::General(format!(
                "GammaPairIntegrals::with_uniform_head: mu {:?} vs (no, nv) = ({}, {})",
                head.mu.dim(),
                self.no,
                self.nv
            )));
        }
        if self.head.is_some() {
            return Err(FerricError::General(
                "GammaPairIntegrals::with_uniform_head: head already applied".into(),
            ));
        }
        self.head = Some(head.column());
        Ok(self)
    }

    /// Whether the blocks carry the restored head.
    pub fn has_uniform_head(&self) -> bool {
        self.head.is_some()
    }

    /// `J_ij[a, b] = (ia|jb)` (plus the head column when applied), `(nv, nv)`,
    /// and the aux contraction length used (naux_kept globally, the domain
    /// size for a domain fit; +1 for the head column).
    pub fn block(&self, i: usize, j: usize) -> Result<(Array2<f64>, usize), FerricError> {
        let (mut g, mut n) = self.raw_block(i, j)?;
        if let Some(h) = &self.head {
            for a in 0..self.nv {
                let x = h[(i, a)];
                for b in 0..self.nv {
                    g[(a, b)] += x * h[(j, b)];
                }
            }
            n += 1;
        }
        Ok((g, n))
    }

    fn raw_block(&self, i: usize, j: usize) -> Result<(Array2<f64>, usize), FerricError> {
        let nv = self.nv;
        if i >= self.no || j >= self.no {
            return Err(FerricError::General(format!(
                "GammaPairIntegrals::block({i}, {j}) with no = {}",
                self.no
            )));
        }
        match &self.src {
            PairSource::Global { b_ov } => {
                let bi = b_ov.slice(s![.., i * nv..(i + 1) * nv]);
                let bj = b_ov.slice(s![.., j * nv..(j + 1) * nv]);
                Ok((bi.t().dot(&bj), b_ov.nrows()))
            }
            PairSource::Domain {
                a,
                j2,
                in_r,
                lindep,
                radius,
            } => {
                let dom: Vec<usize> = (0..self.naux)
                    .filter(|&p| in_r[i][p] || in_r[j][p])
                    .collect();
                let d = dom.len();
                if d == 0 {
                    return Err(FerricError::General(format!(
                        "gamma_lmp2: empty aux domain for pair ({i},{j}) at radius {radius} Bohr"
                    )));
                }
                let jdd = Array2::from_shape_fn((d, d), |(r, c)| j2[(dom[r], dom[c])]);
                let (ev, u) = eigh(&jdd, "domain metric")?;
                let keep: Vec<usize> = (0..d).filter(|&k| ev[k] > *lindep).collect();
                if keep.is_empty() {
                    return Err(FerricError::General(format!(
                        "gamma_lmp2: every domain-metric eigenvalue of pair ({i},{j}) is <= lindep"
                    )));
                }
                let w = Array2::from_shape_fn((d, keep.len()), |(r, c)| {
                    u[(r, keep[c])] / ev[keep[c]].sqrt()
                });
                let ai = Array2::from_shape_fn((d, nv), |(r, x)| a[(dom[r], i * nv + x)]);
                let aj = Array2::from_shape_fn((d, nv), |(r, x)| a[(dom[r], j * nv + x)]);
                let (ai_w, aj_w) = (ai.t().dot(&w), aj.t().dot(&w));
                Ok((ai_w.dot(&aj_w.t()), d))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Stage timings (s).
#[derive(Debug, Clone, Default)]
pub struct GammaLmp2Timings {
    pub t_spaces_s: f64,
    pub t_assembly_s: f64,
    pub t_solve_s: f64,
    pub t_reference_s: f64,
}

/// Diagnostics of the restored q = 0 head ([`UniformHead`]).
#[derive(Debug, Clone, Copy)]
pub struct UniformHeadInfo {
    /// Needle axis.
    pub axis: usize,
    /// `4π/Ω_sc`.
    pub scale: f64,
    /// `max |μ_ia|` (Richardson).
    pub mu_max_abs: f64,
    /// `max |μ_ia − f(b)_ia|`: what the Richardson step moved.
    pub richardson_shift_max: f64,
}

/// Gamma LMP2 result: energies and the molecular driver's counters.
#[derive(Debug, Clone)]
#[must_use]
pub struct GammaLmp2Result {
    pub e_corr: f64,
    /// `rhf.energy + e_corr`.
    pub e_total: f64,
    /// Canonical [`gamma_mp2`] on the same B/convention (NaN when disabled).
    /// For [`GammaEpsGate::UniformHeadRestored`] this is NOT the eps = 0
    /// limit (see `e_corr_target`); `e_corr − e_corr_canonical` then mixes
    /// the truncation error with the uniform term.
    pub e_corr_canonical: f64,
    /// The eps = 0 limit of THIS configuration's gate on the global B (NaN
    /// when `compute_reference` is false): `e_corr_canonical` for
    /// `RawIntegral`; [`mp2_closed_form_local`] on the head-augmented B
    /// (`E_head`) for `UniformHeadRestored`. The truncation error is
    /// `e_corr − e_corr_target`.
    pub e_corr_target: f64,
    /// The restored head's diagnostics (`None` for `RawIntegral`).
    pub uniform_head: Option<UniformHeadInfo>,
    /// `e_ij` (ordered, `(no, no)`), summing to `e_corr`: per retained block
    /// `Σ (2 t_iajb − t_ibja) (ia|jb)`; 0 for dropped pairs.
    pub pair_energies: Array2<f64>,
    /// Retained ORDERED pair blocks (diagonal included); `no²` = all.
    pub pairs_kept: usize,
    /// Retained partners per occupied i (ordered pairs (i, ·)).
    pub partners: Vec<usize>,
    pub pair_fraction: f64,
    /// Retained (i,a,j,b) elements / (no·nv)².
    pub keep_fraction: f64,
    /// Unique off-diagonal pairs removed by the R⁻⁶ gate / by the cutoff.
    pub n_pairs_gated: usize,
    pub n_pairs_cut: usize,
    pub dom_mean: f64,
    pub dom_max: usize,
    pub aux_dom_mean: f64,
    pub aux_dom_max: usize,
    pub cg_iterations: usize,
    pub cg_relres: f64,
    pub cg_converged: bool,
    pub n_valence_virt: usize,
    pub n_hard_virt: usize,
    pub nocc_active: usize,
    pub nvir: usize,
    pub madelung: f64,
    pub occ_shift: f64,
    pub ragged_flops_per_matvec: u64,
    pub dense_flops_per_matvec: u64,
    pub timings: GammaLmp2Timings,
}

/// Gamma-point amplitude-threshold LMP2 (module doc). Builds the localised
/// spaces ([`gamma_localized_spaces`]) and runs [`gamma_lmp2_with_spaces`].
pub fn gamma_lmp2(
    cell: &Cell,
    rhf: &ScfResult,
    ints: &GammaLmp2Inputs<'_>,
    cfg: &GammaLmp2Config,
) -> Result<GammaLmp2Result, FerricError> {
    cfg.validate()?;
    if cfg.eps_gate.restores_uniform_head() {
        // Refuse a non-needle supercell BEFORE the localisation work.
        needle_axis(cell)?;
    }
    let t0 = std::time::Instant::now();
    let spaces = gamma_localized_spaces(cell, rhf, ints, cfg)?;
    let t_spaces_s = t0.elapsed().as_secs_f64();
    let mut r = gamma_lmp2_with_spaces(cell, rhf, ints, cfg, &spaces)?;
    r.timings.t_spaces_s = t_spaces_s;
    Ok(r)
}

/// [`gamma_lmp2`] on caller-supplied localised spaces (mutation-test entry
/// point: a deliberately broken space must fail the eps = 0 anchor).
pub fn gamma_lmp2_with_spaces(
    cell: &Cell,
    rhf: &ScfResult,
    ints: &GammaLmp2Inputs<'_>,
    cfg: &GammaLmp2Config,
    spaces: &GammaLocalSpaces,
) -> Result<GammaLmp2Result, FerricError> {
    cfg.validate()?;
    if cfg.eps_gate.restores_uniform_head() {
        needle_axis(cell)?;
    }
    check_reference(cell, rhf)?;
    let (no, nv) = (spaces.no(), spaces.nv());
    if spaces.f_oo.dim() != (no, no)
        || spaces.f_vv.dim() != (nv, nv)
        || spaces.occ_centers.len() != no
    {
        return Err(FerricError::General(
            "gamma_lmp2: localised spaces are internally inconsistent".into(),
        ));
    }
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    let t0 = std::time::Instant::now();
    let src = GammaPairIntegrals::new_on(
        cell,
        spaces,
        ints.gdf,
        ints.fit,
        cfg.fit_radius_bohr,
        cfg.distance,
        &mut ledger,
    )?;
    // A-drop: restore the q = 0 head as one extra RI column (module doc).
    let nao = spaces.c_occ.nrows();
    let head = if cfg.eps_gate.restores_uniform_head() {
        ledger.reserve(
            &format!("Gamma LMP2 uniform head: 2 Resta pair FTs + Re/Im (nao = {nao})"),
            bytes_of((nao as u64).saturating_mul(nao as u64), 16 + 8 * 4)
                .saturating_add(bytes_of((no * nv) as u64, 8 * 6)),
        )?;
        Some(uniform_head(cell, ints.obs, spaces)?)
    } else {
        None
    };
    let src = match &head {
        Some(h) => src.with_uniform_head(h)?,
        None => src,
    };

    // Pair-level screens: both on MINIMUM-IMAGE centroid distances.
    let dist = centroid_distances(cell, &spaces.occ_centers, cfg.distance);
    let mut n_pairs_gated = 0usize;
    let gate_keep: Option<Vec<bool>> = cfg.pair_gate_cal.map(|cal| {
        let sigma: Vec<f64> = spaces
            .occ_spreads
            .iter()
            .map(|w| w.max(0.0).sqrt())
            .collect();
        let (keep, gated) =
            pair_gate_keep_with(|i, j| dist[(i, j)] * dist[(i, j)], &sigma, no, cfg.eps, cal);
        n_pairs_gated = gated;
        keep
    });
    let mut n_pairs_cut = 0usize;

    let fo: Vec<f64> = (0..no).map(|i| spaces.f_oo[(i, i)]).collect();
    let fv: Vec<f64> = (0..nv).map(|a| spaces.f_vv[(a, a)]).collect();
    let cand: Vec<usize> = (0..nv).collect();
    let mut pairs: Vec<PairBlock> = Vec::new();
    let mut aux_sizes: Vec<usize> = Vec::new();
    for i in 0..no {
        for j in i..no {
            if gate_keep.as_ref().is_some_and(|k| !k[i * no + j]) {
                continue;
            }
            if let Some(rc) = cfg.pair_cutoff_bohr {
                if i != j && dist[(i, j)] > rc {
                    n_pairs_cut += 1;
                    continue;
                }
            }
            let (g, naux_used) = src.block(i, j)?;
            let gate = cfg.eps_gate.gate_quantity(i, j, &g);
            let mut any = false;
            if let Some(pb) = pair_block_from_g_cand_gated(
                i,
                j,
                &g,
                &gate,
                &cand,
                nv,
                &spaces.f_vv,
                &fo,
                &fv,
                cfg.eps,
            ) {
                pairs.push(pb);
                any = true;
            }
            if i != j {
                let gt = g.t().to_owned();
                let gate_t = gate.t().to_owned();
                if let Some(pb) = pair_block_from_g_cand_gated(
                    j,
                    i,
                    &gt,
                    &gate_t,
                    &cand,
                    nv,
                    &spaces.f_vv,
                    &fo,
                    &fv,
                    cfg.eps,
                ) {
                    pairs.push(pb);
                    any = true;
                }
            }
            if any {
                aux_sizes.push(naux_used);
            }
        }
    }
    let mut by_i: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut by_j: HashMap<usize, Vec<usize>> = HashMap::new();
    for (p, pb) in pairs.iter().enumerate() {
        by_i.entry(pb.i).or_default().push(p);
        by_j.entry(pb.j).or_default().push(p);
    }
    let rg = Ragged { pairs, by_i, by_j };
    let t_assembly_s = t0.elapsed().as_secs_f64();

    // CG working set: t, r, z, p, A·p per retained block.
    let elems: u64 = rg
        .pairs
        .iter()
        .map(|pb| (pb.da.len() * pb.db.len()) as u64)
        .sum();
    ledger.reserve(
        &format!("Gamma LMP2 ragged CG vectors ({elems} retained block elements)"),
        bytes_of(elems, 8 * 5),
    )?;

    let t0 = std::time::Instant::now();
    let (t, iters, relres, converged, flops_mv) =
        solve_ragged(&rg, &spaces.f_oo, cfg.cg_rtol, cfg.cg_max_iter);
    if !converged {
        return Err(FerricError::General(format!(
            "gamma_lmp2: ragged CG failed to converge (relres {relres:.2e} after {iters} iters)"
        )));
    }
    let e_corr = hylleraas_energy(&rg, &t);
    if !e_corr.is_finite() {
        return Err(FerricError::General(format!(
            "gamma_lmp2: non-finite correlation energy {e_corr}"
        )));
    }
    let mut pair_energies = Array2::<f64>::zeros((no, no));
    for (p, pb) in rg.pairs.iter().enumerate() {
        let nb = pb.db.len();
        let (mut e_dir, mut e_exx) = (0.0, 0.0);
        for (r, &a) in pb.da.iter().enumerate() {
            for (c, &b) in pb.db.iter().enumerate() {
                if !pb.pat[r * nb + c] {
                    continue;
                }
                let jv = pb.j_blk[(r, c)];
                e_dir += t[p][(r, c)] * jv;
                let (sr, sc) = (pb.pos_da[b], pb.pos_db[a]);
                if sr != usize::MAX && sc != usize::MAX {
                    e_exx += t[p][(sr, sc)] * jv;
                }
            }
        }
        pair_energies[(pb.i, pb.j)] += 2.0 * e_dir - e_exx;
    }
    let t_solve_s = t0.elapsed().as_secs_f64();

    let t0 = std::time::Instant::now();
    let e_ref = if cfg.compute_reference {
        gamma_mp2(
            cell,
            rhf,
            GammaMp2Integrals::RsGdf(ints.gdf),
            &GammaMp2Config {
                frozen_core: cfg.frozen_core,
                reference_exxdiv: cfg.reference_exxdiv,
                denominators: cfg.denominators,
                budget_bytes: cfg.budget_bytes,
            },
        )?
        .mp2_corr
    } else {
        f64::NAN
    };
    // The eps = 0 target of THIS configuration (module doc): canonical Gamma
    // MP2 for the raw gate; closed-form MP2 on the head-augmented global B
    // for A-drop.
    let e_target = match (&head, cfg.compute_reference) {
        (_, false) => f64::NAN,
        (None, true) => e_ref,
        (Some(h), true) => {
            let naux = ints.gdf.b().nrows();
            ledger.reserve(
                &format!(
                    "Gamma LMP2 closed-form target: half-transform + B[k,ia] + rotated copy \
                     (naux = {naux}, nov = {})",
                    no * nv
                ),
                bytes_of(naux as u64 + 1, (nao * nv + 2 * no * nv).saturating_mul(8)),
            )?;
            let b_ov = b_ov_from_ao_b(ints.gdf.b(), nao, spaces.c_occ.view(), spaces.c_vir.view())?;
            mp2_closed_form_local(&b_ov, Some(&h.column()), &spaces.f_oo, &spaces.f_vv)?
        }
    };
    let t_reference_s = if cfg.compute_reference {
        t0.elapsed().as_secs_f64()
    } else {
        0.0
    };

    let total_el = (no * nv) as u64 * (no * nv) as u64;
    let kept: u64 = rg
        .pairs
        .iter()
        .map(|pb| pb.pat.iter().filter(|&&x| x).count() as u64)
        .sum();
    let mut partners = vec![0usize; no];
    let mut per_i: Vec<Vec<bool>> = vec![vec![false; nv]; no];
    for pb in &rg.pairs {
        partners[pb.i] += 1;
        for &a in &pb.da {
            per_i[pb.i][a] = true;
        }
    }
    let dom: Vec<usize> = per_i
        .iter()
        .map(|v| v.iter().filter(|&&x| x).count())
        .collect();
    let dom_max = dom.iter().copied().max().unwrap_or(0);
    let dom_mean = if no > 0 {
        dom.iter().sum::<usize>() as f64 / no as f64
    } else {
        0.0
    };
    let aux_dom_mean = if aux_sizes.is_empty() {
        0.0
    } else {
        aux_sizes.iter().sum::<usize>() as f64 / aux_sizes.len() as f64
    };
    let aux_dom_max = aux_sizes.iter().copied().max().unwrap_or(0);
    let dense_flops =
        2 * ((no * no) as u64 * (nv as u64).pow(3) + (no as u64).pow(3) * (nv * nv) as u64);
    let uniform_head_info = head.as_ref().map(|h| UniformHeadInfo {
        axis: h.axis,
        scale: h.scale,
        mu_max_abs: h.mu.iter().fold(0.0_f64, |m, &x| m.max(x.abs())),
        richardson_shift_max: h
            .mu
            .iter()
            .zip(h.mu_single_b.iter())
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs())),
    });
    Ok(GammaLmp2Result {
        e_corr,
        e_total: rhf.energy + e_corr,
        e_corr_canonical: e_ref,
        e_corr_target: e_target,
        uniform_head: uniform_head_info,
        pair_energies,
        pairs_kept: rg.pairs.len(),
        partners,
        pair_fraction: rg.pairs.len() as f64 / (no * no) as f64,
        keep_fraction: kept as f64 / total_el as f64,
        n_pairs_gated,
        n_pairs_cut,
        dom_mean,
        dom_max,
        aux_dom_mean,
        aux_dom_max,
        cg_iterations: iters,
        cg_relres: relres,
        cg_converged: converged,
        n_valence_virt: spaces.n_valence,
        n_hard_virt: spaces.n_hard,
        nocc_active: no,
        nvir: nv,
        madelung: spaces.madelung,
        occ_shift: spaces.occ_shift,
        ragged_flops_per_matvec: flops_mv,
        dense_flops_per_matvec: dense_flops,
        timings: GammaLmp2Timings {
            t_spaces_s: 0.0,
            t_assembly_s,
            t_solve_s,
            t_reference_s,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::mol::{Atom, Molecule};

    fn h_cell(lat: [[f64; 3]; 3]) -> Cell {
        let mol = Molecule {
            atoms: vec![Atom {
                symbol: "H".into(),
                z: 1,
                x: 0.1,
                y: 0.2,
                zpos: 0.3,
                ghost: false,
                n_core_ecp: 0,
            }],
            charge: 0,
            multiplicity: 2,
        };
        Cell::new(mol, lat).unwrap()
    }

    /// Minimum image vs brute force over a ±2 image shell, on an
    /// orthorhombic and a skewed cell; and the raw mutant is really raw.
    #[test]
    fn min_image_matches_brute_force() {
        for lat in [
            [[7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 21.0]],
            [[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]],
        ] {
            let cell = h_cell(lat);
            let mi = MinImage::new(&cell, PeriodicDistance::MinimumImage);
            let raw = MinImage::new(&cell, PeriodicDistance::RawCartesianMutant);
            for d in [
                [0.3, -4.1, 18.0],
                [13.0, 2.0, -9.5],
                [0.0, 0.0, 0.0],
                [3.4, 3.6, 2.2],
            ] {
                let mut best = f64::INFINITY;
                for n0 in -6i32..=6 {
                    for n1 in -6i32..=6 {
                        for n2 in -6i32..=6 {
                            let v: Vec<f64> = (0..3)
                                .map(|x| {
                                    d[x] + n0 as f64 * lat[0][x]
                                        + n1 as f64 * lat[1][x]
                                        + n2 as f64 * lat[2][x]
                                })
                                .collect();
                            best = best.min((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
                        }
                    }
                }
                assert!(
                    (mi.distance(d) - best).abs() < 1e-12,
                    "{d:?}: {} vs {best}",
                    mi.distance(d)
                );
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                assert_eq!(raw.distance(d), r);
            }
        }
    }

    #[test]
    fn random_orthogonal_is_orthogonal() {
        let q = random_orthogonal(7, 3);
        let d = q.t().dot(&q) - Array2::<f64>::eye(7);
        assert!(d.iter().all(|x| x.abs() < 1e-13));
    }

    #[test]
    fn config_validation_is_strict() {
        let ok = GammaLmp2Config::shifted(ExxDiv::Ewald, 1e-4);
        assert!(ok.validate().is_ok());
        for bad in [
            GammaLmp2Config { eps: -1.0, ..ok },
            GammaLmp2Config {
                eps: f64::NAN,
                ..ok
            },
            GammaLmp2Config {
                pair_cutoff_bohr: Some(0.0),
                ..ok
            },
            GammaLmp2Config {
                fit_radius_bohr: Some(f64::INFINITY),
                ..ok
            },
            GammaLmp2Config {
                pair_gate_cal: Some(-0.7),
                ..ok
            },
            GammaLmp2Config {
                cg_max_iter: 0,
                ..ok
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }
}
