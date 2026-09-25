//! Stage 2: Gamma-point closed-shell Kohn-Sham DFT (LDA, GGA, global
//! hybrids), ported from `reference/pbc/pbc_dft.py` (FINDINGS "Iteration 8
//! (Python, Gamma KS-DFT)").
//!
//! # Grid (construction "A2" of the prototype)
//!
//! Atom-centred Treutler–Ahlrichs M4 radial × Lebedev grids on the CELL's
//! atoms only (ferric's molecular radial/angular rules), each point weighted by
//! its home atom's fuzzy-cell weight evaluated in the INFINITE crystal: the
//! cell functions run over every IMAGE atom within a cutoff `D` of the point
//! ([`ferric_dft::becke::partition_weight_over`]). With `w_{A,L}` the weight
//! of image `A + L`, translation covariance gives `w_{A,L}(r) = w_{A,0}(r − L)`
//! and `Σ_{A,L} w_{A,L} = 1`, so for a lattice-periodic `f`
//!
//! ```text
//! ∫_cell f = Σ_{A ∈ cell} ∫_{R³} w_{A,0} f  ≈  Σ_{A ∈ cell} Σ_{g ∈ grid(A)} w_g f(r_g)
//! ```
//!
//! — the full atomic grids, no cut at the cell boundary. The `D` truncation
//! keeps an exact partition of unity (the same neighbour set for every home
//! at a given `r`) PROVIDED every point of space has an atom within `D`,
//! i.e. `D ≥` the covering radius of the atom lattice. [`covering_radius_bound`]
//! bounds that radius from above and [`PeriodicGrid::build`] refuses a
//! smaller `D` with [`PeriodicDftError::NeighbourCutoffTooSmall`].
//!
//! Partition: SSF by default ([`PartitionScheme::Ssf`], compact support, so a
//! finite image list is exact and the huge-box limit reproduces the molecular
//! weights to 1e-16), Becke available. Size-adjustment caveat: with the Bragg
//! adjustment clipped at `|a| = ½` (e.g. Li–H) the SSF exact zone around an
//! isolated molecule is only ~0.09 of the lattice spacing (measured).
//!
//! # Default grid: (75, 302), NOT the molecular (75, 110)
//!
//! Prototype table, `E(grid) − E(uniform, spectrally converged)` (Becke A2,
//! LDA): H2/STO-3G a = 4: 50×146 −7.8e-4, 75×302 −4.2e-4, 100×590 −6.0e-5;
//! triclinic 4H s+p: −2.0e-4 / −2.9e-5 / −3.9e-6. The dominant error is the
//! partition in a dense lattice and it is ANGULAR-limited (100×302 ≈ 75×302).
//! ferric's molecular default (75, 110) has `|∫ρ − N|` 1.9e-3 on H2 a = 4.
//! 590 would be the next step, but ferric's Lebedev table stops at 302
//! (6/14/26/50/110/302), so the default is the finest angular order available
//! with 75 radial shells. PROVISIONAL: both cells are H-only toys; a real solid
//! with core electrons has not been swept (FINDINGS says so explicitly).
//!
//! # AOs
//!
//! Lattice-summed `χ^Γ_μ(r) = Σ_L χ_μ(r − L)` (and `∇χ^Γ`), evaluated per
//! spatial chunk over the image shells whose extent (value AND gradient below
//! `ao_threshold`) reaches the chunk, folded into cell AO indices. The cache
//! (`4 · nbf · npts` f64) is reserved on the budget before allocation.
//!
//! # XC and the SCF
//!
//! [`PeriodicXc`] implements [`ferric_scf::rhf::XcBuilder`] over ferric-dft's
//! unchanged `semilocal_vxc_closed` kernel; [`gamma_rks`] injects it with the
//! lattice `(S, h, E_nn)` and J/K. Hybrids: the SCF scales the injected K —
//! Madelung term included — by the exact-exchange fraction (measured in the
//! prototype: the box-limit `a⁻³` coefficient is `hyb ×` the HF one).
//!
//! # Open shell (Stage 5, FINDINGS "Iteration 10")
//!
//! [`PeriodicXc`] also evaluates the spin-polarized kernel
//! ([`PeriodicXc::eval_polarized`], ferric-dft's unchanged
//! `semilocal_vxc_polarized` on `nspin = 2` libxc handles) and implements
//! `XcBuilder::build_polarized`; [`gamma_uks`] injects it into
//! `ferric_scf::uhf::solve_uhf_injected`, which forms per spin
//! `F_σ = h + J − a·K_inj(D_σ) + V_σ` (the injected K carries the Madelung
//! term, linear in D_σ: no per-spin ½, no Madelung on the `(1 − a)` part).
//! For `a > 0` and `exxdiv = ewald` the default start is
//! [`EwaldStart::Staged`] (none, then ewald from those MOs), and the
//! OCCUPATION-AWARE per-spin gap is checked against `a·v_M`
//! ([`crate::uhf::occupation_gaps`]). ROKS is NOT implemented (ferric-scf's
//! ROHF has no injected path).
//!
//! Refused by name: range-separated hybrids (need an attenuated periodic K),
//! meta-GGA (not prototyped), VV10 / double hybrids, grid pruning, Newton /
//! TRAH / stability (they rebuild molecular J/K and the f_xc kernel on the
//! molecular grid — rejected by `validate_injected`), the molecular grid knobs
//! `RhfConfig.{xc, xc_omega, dft_grid, nlc_grid}`. Nuclear gradients (with
//! the full periodic grid response) live in `crate::grad`
//! (`gamma_rks_gradient` / `gamma_uks_gradient`), on the dense-AFT J/K.
//!
//! Units: Bohr and Hartree.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::ewald::madelung_constant;
use crate::hcore::PeriodicHcore;
use crate::lattice::Cell;
use crate::uhf::{
    nocc_ab, occupation_gaps, spin_square, EwaldStart, GammaUhfIntegrals, SpinGapReport,
};
use ferric_core::basis::{num_functions, BasisSet};
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_dft::becke::{
    partition_weight_over, partition_weight_over_and_grad, NeighbourAtom, PartitionScheme,
};
use ferric_dft::density_on_grid::{eval_density_closed, eval_density_uks};
use ferric_dft::grid::GridPoint;
use ferric_dft::lebedev::lebedev;
use ferric_dft::libxc::{xc_def_from_name, xc_def_from_name_nspin, FunctionalFamily, XcDef};
use ferric_dft::prune::PruneScheme;
use ferric_dft::radial::treutler_ahlrichs_m4;
use ferric_dft::vxc::{semilocal_vxc_closed_scratch, semilocal_vxc_polarized_scratch, VxcScratch};
use ferric_integrals::ao_grid::{
    collect_shells, eval_shell_and_grad, eval_shell_grad_hess, LocatedShell,
};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{
    solve_rhf_injected, validate_injected, PeriodicInjection, RhfConfig, XcBuilder,
};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf_injected;
use ndarray::{Array2, Array3, Array4};
use rayon::prelude::*;

/// Default radial shells (see the module doc's table).
pub const DEFAULT_PERIODIC_N_RADIAL: usize = 75;
/// Default Lebedev order: the finest in ferric's table (module doc).
pub const DEFAULT_PERIODIC_N_ANGULAR: usize = 302;
/// Floor of the automatic neighbour cutoff `D` (Bohr): the prototype's value
/// for H2 a = 4 (`D = 10` and `14` gave identical grids there).
pub const DEFAULT_NEIGHBOUR_CUTOFF: f64 = 10.0;
/// Default AO truncation: every image shell whose value and gradient bound
/// is below this at a point is skipped (prototype `ao_thresh`).
pub const DEFAULT_AO_THRESHOLD: f64 = 1e-15;
/// Lebedev orders ferric's table implements (`lebedev()` panics otherwise,
/// so the config is checked against this list first).
pub const SUPPORTED_LEBEDEV_ORDERS: [usize; 6] = [6, 14, 26, 50, 110, 302];
/// Points per AO/XC chunk.
const CHUNK: usize = 512;
/// Spatial sort box edge (Bohr) that groups points into compact chunks.
const SORT_BOX: f64 = 2.0;

/// A Stage-2 periodic-DFT refusal or grid-construction error, by name.
#[derive(Debug, Clone, PartialEq)]
pub enum PeriodicDftError {
    /// `D` is below the covering-radius bound: some point of space would have
    /// no atom within `D`, so the partition would be empty there and that
    /// region's integrand silently lost.
    NeighbourCutoffTooSmall {
        /// The requested cutoff (Bohr).
        cutoff: f64,
        /// [`covering_radius_bound`] of the cell (Bohr).
        required: f64,
    },
    /// A feature Stage 2 does not implement.
    Unsupported {
        /// The config field / feature, as spelled in the API.
        feature: &'static str,
        /// Why.
        reason: String,
    },
    /// An invalid grid parameter.
    InvalidGrid {
        /// The config field.
        field: &'static str,
        /// Why.
        reason: String,
    },
}

impl std::fmt::Display for PeriodicDftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NeighbourCutoffTooSmall { cutoff, required } => write!(
                f,
                "periodic DFT grid: neighbour_cutoff D = {cutoff} Bohr is below the \
                 covering-radius bound {required:.6} Bohr (the largest point-to-nearest-atom \
                 distance); points with no atom within D would get zero weight. Use \
                 neighbour_cutoff >= {required:.6} (or None for the automatic value)"
            ),
            Self::Unsupported { feature, reason } => {
                write!(f, "periodic DFT: {feature} is not supported: {reason}")
            }
            Self::InvalidGrid { field, reason } => {
                write!(f, "periodic DFT grid: invalid {field}: {reason}")
            }
        }
    }
}

impl std::error::Error for PeriodicDftError {}

impl From<PeriodicDftError> for FerricError {
    fn from(e: PeriodicDftError) -> Self {
        FerricError::General(e.to_string())
    }
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    dot3(&d, &d).sqrt()
}

/// `Σ_i f_i a_i`.
fn frac_to_cart(a: &[[f64; 3]; 3], f: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (i, fi) in f.iter().enumerate() {
        for k in 0..3 {
            out[k] += fi * a[i][k];
        }
    }
    out
}

/// Fractional coordinates of the Cartesian vector `d`: `f_i = d·b_i / 2π`.
fn cart_to_frac(b: &[[f64; 3]; 3], d: [f64; 3]) -> [f64; 3] {
    let tp = 2.0 * std::f64::consts::PI;
    [
        dot3(&d, &b[0]) / tp,
        dot3(&d, &b[1]) / tp,
        dot3(&d, &b[2]) / tp,
    ]
}

/// Upper bound (Bohr) on the covering radius of the cell's atom lattice,
/// `max_r min_{A, L} |r − R_A − L|` — the smallest neighbour cutoff `D` for
/// which every point of space has an atom within `D`.
///
/// `min` of two valid upper bounds:
/// * the Bravais-lattice bound `max_{s ∈ {±½}³} |Σ s_i a_i|` (rounding a
///   point's fractional offset from any one atom to the nearest integer
///   leaves each component within ½);
/// * a probe mesh: the largest point-to-nearest-atom distance over `m³`
///   mesh-cell centres (nearest image searched in the 27 cells around the
///   rounded fractional offset — any candidate is an upper bound on the true
///   nearest distance), plus the mesh cell's half body diagonal (the
///   nearest-atom distance is 1-Lipschitz).
pub fn covering_radius_bound(cell: &Cell) -> f64 {
    let a = cell.lattice();
    let b = cell.reciprocal();
    let mut lattice_bound = 0.0_f64;
    for s0 in [-0.5, 0.5] {
        for s1 in [-0.5, 0.5] {
            for s2 in [-0.5, 0.5] {
                let v = frac_to_cart(a, [s0, s1, s2]);
                lattice_bound = lattice_bound.max(dot3(&v, &v).sqrt());
            }
        }
    }
    let longest = a.iter().map(|v| dot3(v, v).sqrt()).fold(0.0, f64::max);
    let m = ((longest / 0.5).ceil() as usize).clamp(4, 48);
    let pos = cell.positions();
    let probe_max = (0..m * m * m)
        .into_par_iter()
        .map(|idx| {
            let (i, j, k) = (idx / (m * m), (idx / m) % m, idx % m);
            let f = [
                (i as f64 + 0.5) / m as f64,
                (j as f64 + 0.5) / m as f64,
                (k as f64 + 0.5) / m as f64,
            ];
            let p = frac_to_cart(a, f);
            let mut near = f64::INFINITY;
            for r in &pos {
                let fr = cart_to_frac(&b, [p[0] - r[0], p[1] - r[1], p[2] - r[2]]);
                let base = [
                    fr[0] - fr[0].round(),
                    fr[1] - fr[1].round(),
                    fr[2] - fr[2].round(),
                ];
                for n0 in -1..=1 {
                    for n1 in -1..=1 {
                        for n2 in -1..=1 {
                            let v = frac_to_cart(
                                a,
                                [
                                    base[0] + n0 as f64,
                                    base[1] + n1 as f64,
                                    base[2] + n2 as f64,
                                ],
                            );
                            near = near.min(dot3(&v, &v).sqrt());
                        }
                    }
                }
            }
            near
        })
        .reduce(|| 0.0_f64, f64::max);
    lattice_bound.min(probe_max + lattice_bound / m as f64)
}

/// Configuration of a [`PeriodicGrid`].
#[derive(Debug, Clone)]
pub struct PeriodicGridConfig {
    /// Treutler–Ahlrichs M4 radial shells per atom.
    pub n_radial: usize,
    /// Lebedev order (one of [`SUPPORTED_LEBEDEV_ORDERS`]).
    pub n_angular: usize,
    /// Fuzzy-cell partition (default SSF).
    pub partition: PartitionScheme,
    /// Neighbour cutoff `D` (Bohr). `None` = `max(DEFAULT_NEIGHBOUR_CUTOFF,
    /// covering_radius_bound)`; `Some(d)` is validated against
    /// [`covering_radius_bound`] (typed error if smaller).
    pub neighbour_cutoff: Option<f64>,
    /// Angular pruning: must be `None` (refused by name — the pruning tables
    /// are molecular and unvalidated periodically).
    pub prune: Option<PruneScheme>,
    /// Budget for the grid-point storage (bytes); `None` = ferric's unified
    /// budget.
    pub budget_bytes: Option<usize>,
}

impl Default for PeriodicGridConfig {
    fn default() -> Self {
        Self {
            n_radial: DEFAULT_PERIODIC_N_RADIAL,
            n_angular: DEFAULT_PERIODIC_N_ANGULAR,
            partition: PartitionScheme::Ssf,
            neighbour_cutoff: None,
            prune: None,
            budget_bytes: None,
        }
    }
}

impl PeriodicGridConfig {
    /// `(n_radial, n_angular)` with the other fields at their defaults.
    pub fn with_size(n_radial: usize, n_angular: usize) -> Self {
        Self {
            n_radial,
            n_angular,
            ..Self::default()
        }
    }

    fn validate(&self) -> Result<(), PeriodicDftError> {
        if self.prune.is_some() {
            return Err(PeriodicDftError::Unsupported {
                feature: "prune",
                reason: "angular pruning is not implemented for periodic grids (the \
                         region tables are molecular and unvalidated here); use None"
                    .into(),
            });
        }
        if self.n_radial == 0 {
            return Err(PeriodicDftError::InvalidGrid {
                field: "n_radial",
                reason: "must be >= 1".into(),
            });
        }
        if !SUPPORTED_LEBEDEV_ORDERS.contains(&self.n_angular) {
            return Err(PeriodicDftError::InvalidGrid {
                field: "n_angular",
                reason: format!(
                    "{} is not a Lebedev order ferric implements ({:?})",
                    self.n_angular, SUPPORTED_LEBEDEV_ORDERS
                ),
            });
        }
        if let Some(d) = self.neighbour_cutoff {
            if !(d.is_finite() && d > 0.0) {
                return Err(PeriodicDftError::InvalidGrid {
                    field: "neighbour_cutoff",
                    reason: format!("must be finite and > 0, got {d}"),
                });
            }
        }
        Ok(())
    }
}

/// Resolve (and validate) the neighbour cutoff for `cell`: `Ok((D, bound))`.
pub fn resolve_neighbour_cutoff(
    cell: &Cell,
    requested: Option<f64>,
) -> Result<(f64, f64), PeriodicDftError> {
    let bound = covering_radius_bound(cell);
    match requested {
        None => Ok((DEFAULT_NEIGHBOUR_CUTOFF.max(bound), bound)),
        Some(d) if d >= bound => Ok((d, bound)),
        Some(d) => Err(PeriodicDftError::NeighbourCutoffTooSmall {
            cutoff: d,
            required: bound,
        }),
    }
}

/// Integration grid for lattice-periodic functions over one cell.
#[derive(Debug, Clone)]
pub struct PeriodicGrid {
    points: Vec<GridPoint>,
    neighbour_cutoff: f64,
    covering_bound: f64,
    n_generated: usize,
    mean_neighbours: f64,
}

impl PeriodicGrid {
    /// Atom-centred periodic grid (module doc). Points with an exactly zero
    /// weight (SSF: outside the home atom's support) are dropped — exact.
    pub fn build(cell: &Cell, cfg: &PeriodicGridConfig) -> Result<Self, FerricError> {
        cfg.validate()?;
        let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
        let gen = generate_points(
            cell,
            cfg,
            &mut ledger,
            0,
            |nb, _cells, h, xyz, _home, _w_rl| {
                (partition_weight_over(cfg.partition, nb, h, xyz), ())
            },
        )?;
        Ok(Self {
            points: gen.points.into_iter().map(|(g, ())| g).collect(),
            neighbour_cutoff: gen.d_cut,
            covering_bound: gen.bound,
            n_generated: gen.n_generated,
            mean_neighbours: gen.nb_total as f64 / gen.n_generated.max(1) as f64,
        })
    }

    /// Construction B (oracle): `n[0]·n[1]·n[2]` uniform points
    /// `Σ_i (k_i/n_i) a_i` with weight `Ω/N` — spectrally accurate for smooth
    /// all-Gaussian densities, useless for cusps; the independent reference
    /// PySCF's `UniformGrids` also uses.
    pub fn uniform(cell: &Cell, n: [usize; 3]) -> Result<Self, FerricError> {
        if n.contains(&0) {
            return Err(PeriodicDftError::InvalidGrid {
                field: "uniform mesh",
                reason: format!("every dimension must be >= 1, got {n:?}"),
            }
            .into());
        }
        let total = n[0] * n[1] * n[2];
        let mut ledger = Ledger::new(crate::budget::resolve(None));
        ledger.reserve(
            "uniform grid points",
            bytes_of(total as u64, std::mem::size_of::<GridPoint>()),
        )?;
        let a = cell.lattice();
        let w = cell.volume() / total as f64;
        let mut points = Vec::with_capacity(total);
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    let f = [
                        i as f64 / n[0] as f64,
                        j as f64 / n[1] as f64,
                        k as f64 / n[2] as f64,
                    ];
                    points.push(GridPoint {
                        xyz: frac_to_cart(a, f),
                        weight: w,
                        home_atom: 0,
                    });
                }
            }
        }
        Ok(Self {
            points,
            neighbour_cutoff: 0.0,
            covering_bound: 0.0,
            n_generated: total,
            mean_neighbours: 0.0,
        })
    }

    /// The weighted points.
    pub fn points(&self) -> &[GridPoint] {
        &self.points
    }

    /// Number of (nonzero-weight) points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// No points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The neighbour cutoff `D` used (Bohr; 0 for a uniform grid).
    pub fn neighbour_cutoff(&self) -> f64 {
        self.neighbour_cutoff
    }

    /// [`covering_radius_bound`] of the cell (0 for a uniform grid).
    pub fn covering_bound(&self) -> f64 {
        self.covering_bound
    }

    /// Points generated before dropping zero weights.
    pub fn n_generated(&self) -> usize {
        self.n_generated
    }

    /// Mean image atoms within `D` per generated point (the O(n²) partition
    /// cost driver).
    pub fn mean_neighbours(&self) -> f64 {
        self.mean_neighbours
    }

    /// `Σ_g w_g` (= Ω exactly for the uniform grid; the partition error for
    /// an atom-centred one).
    pub fn weight_sum(&self) -> f64 {
        self.points.iter().map(|g| g.weight).sum()
    }
}

/// Output of [`generate_points`]: the nonzero-weight points (with the
/// per-point payload) and the construction statistics.
struct GeneratedPoints<T> {
    points: Vec<(GridPoint, T)>,
    d_cut: f64,
    bound: f64,
    n_generated: usize,
    nb_total: usize,
}

/// The A2 construction shared by [`PeriodicGrid::build`] and
/// [`build_response_grid`] (one code path, so both produce the same points
/// and weights). For every candidate point `per_point(nb, nb_cells, h, xyz,
/// home, w_rl)` gets the image atoms within `D` (the home atom always
/// included, at index `h`), the CELL atom each image belongs to, the point,
/// its home cell atom and the radial × angular weight; it returns the
/// partition weight `w` and a payload. The point's weight is `w_rl · w`, and
/// exactly-zero weights are dropped. `extra_bytes_per_point` is added to the
/// per-point reservation (payload storage).
fn generate_points<T, F>(
    cell: &Cell,
    cfg: &PeriodicGridConfig,
    ledger: &mut Ledger,
    extra_bytes_per_point: usize,
    per_point: F,
) -> Result<GeneratedPoints<T>, FerricError>
where
    T: Send,
    F: Fn(&[NeighbourAtom], &[usize], usize, [f64; 3], usize, f64) -> (f64, T) + Sync,
{
    let (d_cut, bound) = resolve_neighbour_cutoff(cell, cfg.neighbour_cutoff)?;
    let pos = cell.positions();
    let zs: Vec<i32> = cell.mol().atoms.iter().map(|a| a.z).collect();
    let (leb_pts, leb_w) = lebedev(cfg.n_angular);

    // Upper bound on the point count before anything is generated.
    let n_max = pos.len() * cfg.n_radial * leb_pts.len();
    ledger.reserve(
        "periodic grid points",
        bytes_of(
            n_max as u64,
            (std::mem::size_of::<GridPoint>() + 8).saturating_add(extra_bytes_per_point),
        ),
    )?;

    // Every image atom within 2D of a home atom can be within D of one of
    // its (|offset| <= D) points; translations(2D) is a superset of the
    // needed L (min atom–atom distance <= 2D).
    let trans = cell.translations(2.0 * d_cut)?;
    let mut points: Vec<(GridPoint, T)> = Vec::new();
    let mut n_generated = 0usize;
    let mut nb_total = 0usize;
    for (home, r_a) in pos.iter().enumerate() {
        let mut cands: Vec<NeighbourAtom> = Vec::new();
        let mut cand_cell: Vec<usize> = Vec::new();
        let mut home_c = usize::MAX;
        for (li, l) in trans.iter().enumerate() {
            for (b, r_b) in pos.iter().enumerate() {
                let xyz = [r_b[0] + l[0], r_b[1] + l[1], r_b[2] + l[2]];
                if dist3(&xyz, r_a) <= 2.0 * d_cut {
                    if b == home && li == 0 {
                        home_c = cands.len();
                    }
                    cands.push(NeighbourAtom { xyz, z: zs[b] });
                    cand_cell.push(b);
                }
            }
        }
        debug_assert!(home_c != usize::MAX, "translations()[0] is the zero vector");
        let (rs, ws) = treutler_ahlrichs_m4(zs[home], cfg.n_radial);
        let mut pre: Vec<([f64; 3], f64)> = Vec::new();
        for (r, w_r) in rs.iter().zip(&ws) {
            if *r > d_cut {
                continue; // home not within D: weight 0 by construction
            }
            for (u, w_l) in leb_pts.iter().zip(&leb_w) {
                pre.push((
                    [r_a[0] + r * u[0], r_a[1] + r * u[1], r_a[2] + r * u[2]],
                    w_r * w_l,
                ));
            }
        }
        n_generated += pre.len();
        let weighted: Vec<(GridPoint, usize, T)> = pre
            .par_iter()
            .map(|&(xyz, w_rl)| {
                let mut nb: Vec<NeighbourAtom> = Vec::new();
                let mut nb_cell: Vec<usize> = Vec::new();
                let mut h = usize::MAX;
                for (ci, c) in cands.iter().enumerate() {
                    if ci == home_c {
                        h = nb.len();
                        nb.push(*c);
                        nb_cell.push(cand_cell[ci]);
                    } else if dist3(&c.xyz, &xyz) <= d_cut {
                        nb.push(*c);
                        nb_cell.push(cand_cell[ci]);
                    }
                }
                let (w, payload) = per_point(&nb, &nb_cell, h, xyz, home, w_rl);
                (
                    GridPoint {
                        xyz,
                        weight: w_rl * w,
                        home_atom: home,
                    },
                    nb.len(),
                    payload,
                )
            })
            .collect();
        for (g, nn, payload) in weighted {
            nb_total += nn;
            if g.weight != 0.0 {
                points.push((g, payload));
            }
        }
    }
    Ok(GeneratedPoints {
        points,
        d_cut,
        bound,
        n_generated,
        nb_total,
    })
}

/// The [`PeriodicGrid`] points of a config together with the analytic
/// derivative of every point's final weight with respect to every cell atom
/// (the Gamma KS-DFT grid response, `crate::grad`).
#[derive(Debug, Clone)]
pub(crate) struct ResponseGrid {
    /// Exactly the points and weights of [`PeriodicGrid::build`].
    pub(crate) points: Vec<GridPoint>,
    /// `dweight[g · natoms + A] = d(w_rl · w_g)/dR_A`: the TOTAL derivative —
    /// every image of `A` moves, and when `A` is the point's home atom the
    /// point moves with it (`∇_r w = −Σ_k ∂w/∂X_k`).
    pub(crate) dweight: Vec<[f64; 3]>,
    /// Cell atoms.
    pub(crate) natoms: usize,
}

/// [`PeriodicGrid::build`]'s points plus the weight derivatives
/// ([`ResponseGrid`]; `ferric_dft::becke::partition_weight_over_and_grad`
/// folded from image atoms onto cell atoms). The derivative storage
/// (`npts · natoms · 24` bytes) is reserved on `ledger` with the points.
pub(crate) fn build_response_grid(
    cell: &Cell,
    cfg: &PeriodicGridConfig,
    ledger: &mut Ledger,
) -> Result<ResponseGrid, FerricError> {
    cfg.validate()?;
    let natoms = cell.positions().len();
    let per_point_payload = natoms
        .saturating_mul(std::mem::size_of::<[f64; 3]>())
        .saturating_add(std::mem::size_of::<Vec<[f64; 3]>>());
    let gen = generate_points(
        cell,
        cfg,
        ledger,
        per_point_payload,
        |nb, nb_cell, h, xyz, home, w_rl| {
            // The weight through the energy's own kernel (bit-identical grid).
            let w = partition_weight_over(cfg.partition, nb, h, xyz);
            let (_, dw) = partition_weight_over_and_grad(cfg.partition, nb, h, xyz);
            let mut out = vec![[0.0_f64; 3]; natoms];
            let mut tot = [0.0_f64; 3];
            for (k, dk) in dw.iter().enumerate() {
                for x in 0..3 {
                    out[nb_cell[k]][x] += w_rl * dk[x];
                    tot[x] += dk[x];
                }
            }
            for x in 0..3 {
                out[home][x] -= w_rl * tot[x];
            }
            (w, out)
        },
    )?;
    let mut points = Vec::with_capacity(gen.points.len());
    let mut dweight = Vec::with_capacity(gen.points.len() * natoms);
    for (g, dw) in gen.points {
        points.push(g);
        dweight.extend_from_slice(&dw);
    }
    Ok(ResponseGrid {
        points,
        dweight,
        natoms,
    })
}

/// The [`PeriodicGrid`] points of a config with the STRAIN derivative of
/// every point's weight and the point's anchor (the Gamma KS-DFT stress,
/// `crate::stress`; FINDINGS "Iteration 19").
#[derive(Debug, Clone)]
pub(crate) struct StrainGrid {
    /// Exactly the points and weights of [`PeriodicGrid::build`].
    pub(crate) points: Vec<GridPoint>,
    /// `dweight[g][a][b] = d(w_rl · w_g)/dε_ab` for a point rigidly attached
    /// to its home atom (offset unstrained) while every image atom moves
    /// with the strain:
    /// `w_rl Σ_k ∂w/∂X_{k,a} (X_k − R_home)_b` over the image neighbour list
    /// (`partition_weight_over_and_grad`'s per-IMAGE derivative, before the
    /// fold onto cell atoms that [`build_response_grid`] does).
    pub(crate) dweight: Vec<[[f64; 3]; 3]>,
    /// The home atom's position `O_g` (the point moves as `dr/dε_ab = e_a O_b`).
    pub(crate) anchors: Vec<[f64; 3]>,
}

/// [`PeriodicGrid::build`]'s points plus the per-point strain derivative of
/// the weight ([`StrainGrid`]). The derivative storage (`npts · 96` bytes) is
/// reserved on `ledger` with the points.
pub(crate) fn build_strain_grid(
    cell: &Cell,
    cfg: &PeriodicGridConfig,
    ledger: &mut Ledger,
) -> Result<StrainGrid, FerricError> {
    cfg.validate()?;
    let per_point_payload = std::mem::size_of::<([[f64; 3]; 3], [f64; 3])>();
    let gen = generate_points(
        cell,
        cfg,
        ledger,
        per_point_payload,
        |nb, _nb_cell, h, xyz, _home, w_rl| {
            // The weight through the energy's own kernel (bit-identical grid).
            let w = partition_weight_over(cfg.partition, nb, h, xyz);
            let (_, dw) = partition_weight_over_and_grad(cfg.partition, nb, h, xyz);
            let xh = nb[h].xyz;
            let mut out = [[0.0_f64; 3]; 3];
            for (k, dk) in dw.iter().enumerate() {
                let rel = [
                    nb[k].xyz[0] - xh[0],
                    nb[k].xyz[1] - xh[1],
                    nb[k].xyz[2] - xh[2],
                ];
                for a in 0..3 {
                    for b in 0..3 {
                        out[a][b] += w_rl * dk[a] * rel[b];
                    }
                }
            }
            (w, (out, xh))
        },
    )?;
    let mut points = Vec::with_capacity(gen.points.len());
    let mut dweight = Vec::with_capacity(gen.points.len());
    let mut anchors = Vec::with_capacity(gen.points.len());
    for (g, (dw, o)) in gen.points {
        points.push(g);
        dweight.push(dw);
        anchors.push(o);
    }
    Ok(StrainGrid {
        points,
        dweight,
        anchors,
    })
}

/// Image-resolved AO strain moments of [`LatticeAoHess::eval_strain`].
pub(crate) struct AoStrainMoments {
    /// `χ^Γ` `(nbf, np)`.
    pub(crate) chi: Array2<f64>,
    /// `∇χ^Γ` `(3, nbf, np)`.
    pub(crate) dchi: Array3<f64>,
    /// `m1[(a, b, μ, g)] = Σ_L ∂_aχ_μ(r_g − X_L) (O_g − X_L)_b` `(3, 3, nbf, np)`:
    /// `dχ^Γ_μ(r_g)/dε_ab` when the point moves as its anchor `O_g`.
    pub(crate) m1: Array4<f64>,
    /// `m2[(3k + a, b, μ, g)] = Σ_L ∂_k∂_aχ_μ(r_g − X_L) (O_g − X_L)_b`
    /// `(9, 3, nbf, np)` = `d(∂_kχ^Γ_μ)/dε_ab` (GGA only).
    pub(crate) m2: Option<Array4<f64>>,
}

/// Radius beyond which a shell's value AND gradient are below `thresh`
/// (every primitive, with the same normalisation as ferric's AO evaluator;
/// a factor 10 covers the pure-harmonic prefactor).
fn shell_extent(sh: &LocatedShell, thresh: f64) -> f64 {
    let l = sh.l;
    let dbl_fact: f64 = match l {
        0 | 1 => 1.0,
        2 => 3.0,
        3 => 15.0,
        _ => 105.0,
    };
    let nprim = sh.exponents.len().max(1) as f64;
    let mut ext = 0.0_f64;
    for (&alpha, &c) in sh.exponents.iter().zip(sh.coefficients) {
        let norm = (2.0 * alpha / std::f64::consts::PI).powf(0.75) * (4.0 * alpha).powi(l).sqrt()
            / dbl_fact.sqrt();
        let pref = 10.0 * nprim * (c * norm).abs() * (1.0 + l as f64 + 2.0 * alpha);
        // g(r) = pref (1+r)^{l+1} e^{-α r²} >= |χ|, |∇χ|; solve g(r) = thresh.
        let mut r = 1.0_f64;
        for _ in 0..50 {
            let arg = (pref * (1.0 + r).powi(l + 1) / thresh).ln();
            r = (arg.max(1e-3) / alpha).sqrt();
        }
        ext = ext.max(r);
    }
    ext
}

/// One spatial chunk of the lattice-summed AO cache.
#[derive(Debug, Clone)]
struct AoChunk {
    points: Vec<GridPoint>,
    /// `(nbf, npts)`.
    chi: Array2<f64>,
    /// `(3, nbf, npts)`.
    dchi: Array3<f64>,
    /// Distinct lattice translations with a live shell in this chunk.
    n_images: usize,
}

/// Build the chunked `χ^Γ`, `∇χ^Γ` cache over `grid` (budget-gated).
fn lattice_ao_chunks(
    cell: &Cell,
    bs: &BasisSet,
    grid: &PeriodicGrid,
    thresh: f64,
    ledger: &mut Ledger,
) -> Result<(Vec<AoChunk>, usize), FerricError> {
    if !(thresh.is_finite() && thresh > 0.0 && thresh < 1.0) {
        return Err(PeriodicDftError::InvalidGrid {
            field: "ao_threshold",
            reason: format!("must be in (0, 1), got {thresh}"),
        }
        .into());
    }
    let shells = collect_shells(cell.mol(), bs)?;
    let mut offsets = Vec::with_capacity(shells.len());
    let mut nbf = 0usize;
    for sh in &shells {
        if sh.l > 4 {
            return Err(PeriodicDftError::Unsupported {
                feature: "basis angular momentum",
                reason: format!("l = {} (the AO evaluator supports s..g)", sh.l),
            }
            .into());
        }
        offsets.push(nbf);
        nbf += num_functions(sh.l, sh.pure);
    }
    let npts = grid.len();
    ledger.reserve(
        "periodic XC AO cache (chi + grad chi)",
        bytes_of((npts as u64).saturating_mul(nbf as u64), 4 * 8),
    )?;
    let ext: Vec<f64> = shells.iter().map(|s| shell_extent(s, thresh)).collect();
    let r_ao = ext.iter().copied().fold(0.0, f64::max);

    // Spatially sorted chunks.
    let pts = grid.points();
    let mut order: Vec<usize> = (0..npts).collect();
    let key = |p: &[f64; 3]| {
        [
            (p[0] / SORT_BOX).floor() as i64,
            (p[1] / SORT_BOX).floor() as i64,
            (p[2] / SORT_BOX).floor() as i64,
        ]
    };
    order.sort_by_key(|&i| key(&pts[i].xyz));

    // Lattice translations reaching any point: |L| <= rho + R_ao + spread.
    let pos = cell.positions();
    let nat = pos.len() as f64;
    let c0 = [
        pos.iter().map(|p| p[0]).sum::<f64>() / nat,
        pos.iter().map(|p| p[1]).sum::<f64>() / nat,
        pos.iter().map(|p| p[2]).sum::<f64>() / nat,
    ];
    let rho = pts.iter().map(|g| dist3(&g.xyz, &c0)).fold(0.0, f64::max);
    let spread = pos.iter().map(|p| dist3(p, &c0)).fold(0.0, f64::max);
    let rcut_l = rho + r_ao + spread;
    let n_l = cell.translation_count_bound(rcut_l + 2.0 * spread)?;
    ledger.check(
        "periodic XC AO translation list",
        bytes_of(n_l, std::mem::size_of::<[f64; 3]>()),
    )?;
    let trans: Vec<[f64; 3]> = cell
        .translations(rcut_l + 2.0 * spread)?
        .into_iter()
        .filter(|l| dot3(l, l).sqrt() <= rcut_l + spread)
        .collect();

    let chunk_idx: Vec<Vec<usize>> = order.chunks(CHUNK).map(|c| c.to_vec()).collect();
    let chunks: Result<Vec<AoChunk>, FerricError> = chunk_idx
        .par_iter()
        .map(|idx| -> Result<AoChunk, FerricError> {
            let cp: Vec<GridPoint> = idx.iter().map(|&i| pts[i]).collect();
            let m = cp.len() as f64;
            let cen = [
                cp.iter().map(|g| g.xyz[0]).sum::<f64>() / m,
                cp.iter().map(|g| g.xyz[1]).sum::<f64>() / m,
                cp.iter().map(|g| g.xyz[2]).sum::<f64>() / m,
            ];
            let rad = cp.iter().map(|g| dist3(&g.xyz, &cen)).fold(0.0, f64::max);
            // Live (shell, shifted centre) pairs for this chunk.
            let mut live: Vec<(usize, [f64; 3])> = Vec::new();
            let mut n_images = 0usize;
            for l in &trans {
                let mut any = false;
                for (s, sh) in shells.iter().enumerate() {
                    let c = [
                        sh.center[0] + l[0],
                        sh.center[1] + l[1],
                        sh.center[2] + l[2],
                    ];
                    if dist3(&c, &cen) <= rad + ext[s] {
                        live.push((s, c));
                        any = true;
                    }
                }
                n_images += usize::from(any);
            }
            let np = cp.len();
            let mut chi = Array2::<f64>::zeros((nbf, np));
            let mut dchi = Array3::<f64>::zeros((3, nbf, np));
            let mut buf = [0.0_f64; 15];
            let mut gbuf = [[0.0_f64; 15]; 3];
            for (g, pt) in cp.iter().enumerate() {
                for &(s, c) in &live {
                    let (dx, dy, dz) = (pt.xyz[0] - c[0], pt.xyz[1] - c[1], pt.xyz[2] - c[2]);
                    if dx * dx + dy * dy + dz * dz > ext[s] * ext[s] {
                        continue;
                    }
                    let sh = &shells[s];
                    let n = num_functions(sh.l, sh.pure);
                    buf.fill(0.0);
                    for row in gbuf.iter_mut() {
                        row.fill(0.0);
                    }
                    eval_shell_and_grad(sh, dx, dy, dz, &mut buf[..n], &mut gbuf)?;
                    let o = offsets[s];
                    for i in 0..n {
                        chi[(o + i, g)] += buf[i];
                        for ax in 0..3 {
                            dchi[(ax, o + i, g)] += gbuf[ax][i];
                        }
                    }
                }
            }
            Ok(AoChunk {
                points: cp,
                chi,
                dchi,
                n_images,
            })
        })
        .collect();
    Ok((chunks?, nbf))
}

/// Lattice-summed AO values, gradients AND Hessians on arbitrary point
/// chunks (`χ^Γ`, `∇χ^Γ`, `∇∇χ^Γ` folded into cell AO indices) — the
/// `ValueGradHess` analogue of the energy's AO cache, for the periodic XC
/// gradient (`crate::grad`). The image-shell list and the per-shell extents
/// are those of the energy path (value AND gradient below `thresh`); the
/// Hessian of a skipped shell is below `thresh` times a modest factor and
/// only ever multiplies `(D χ)`.
pub(crate) struct LatticeAoHess<'a> {
    shells: Vec<LocatedShell<'a>>,
    offsets: Vec<usize>,
    nbf: usize,
    ext: Vec<f64>,
    trans: Vec<[f64; 3]>,
}

impl<'a> LatticeAoHess<'a> {
    /// Image translations covering every point of `pts` (the energy path's
    /// rule).
    pub(crate) fn new(
        cell: &Cell,
        bs: &'a BasisSet,
        pts: &[GridPoint],
        thresh: f64,
        ledger: &Ledger,
    ) -> Result<Self, FerricError> {
        if !(thresh.is_finite() && thresh > 0.0 && thresh < 1.0) {
            return Err(PeriodicDftError::InvalidGrid {
                field: "ao_threshold",
                reason: format!("must be in (0, 1), got {thresh}"),
            }
            .into());
        }
        let shells = collect_shells(cell.mol(), bs)?;
        let mut offsets = Vec::with_capacity(shells.len());
        let mut nbf = 0usize;
        for sh in &shells {
            if sh.l > 4 {
                return Err(PeriodicDftError::Unsupported {
                    feature: "basis angular momentum",
                    reason: format!("l = {} (the AO evaluator supports s..g)", sh.l),
                }
                .into());
            }
            offsets.push(nbf);
            nbf += num_functions(sh.l, sh.pure);
        }
        let ext: Vec<f64> = shells.iter().map(|s| shell_extent(s, thresh)).collect();
        let r_ao = ext.iter().copied().fold(0.0, f64::max);
        let pos = cell.positions();
        let nat = pos.len() as f64;
        let c0 = [
            pos.iter().map(|p| p[0]).sum::<f64>() / nat,
            pos.iter().map(|p| p[1]).sum::<f64>() / nat,
            pos.iter().map(|p| p[2]).sum::<f64>() / nat,
        ];
        let rho = pts.iter().map(|g| dist3(&g.xyz, &c0)).fold(0.0, f64::max);
        let spread = pos.iter().map(|p| dist3(p, &c0)).fold(0.0, f64::max);
        let rcut_l = rho + r_ao + spread;
        let n_l = cell.translation_count_bound(rcut_l + 2.0 * spread)?;
        ledger.check(
            "periodic XC gradient AO translation list",
            bytes_of(n_l, std::mem::size_of::<[f64; 3]>()),
        )?;
        let trans: Vec<[f64; 3]> = cell
            .translations(rcut_l + 2.0 * spread)?
            .into_iter()
            .filter(|l| dot3(l, l).sqrt() <= rcut_l + spread)
            .collect();
        Ok(Self {
            shells,
            offsets,
            nbf,
            ext,
            trans,
        })
    }

    /// Cell AO count.
    pub(crate) fn nbf(&self) -> usize {
        self.nbf
    }

    /// Spatially compact chunks of at most `chunk` point indices (the energy
    /// path's sort box).
    pub(crate) fn chunks(pts: &[GridPoint], chunk: usize) -> Vec<Vec<usize>> {
        let mut order: Vec<usize> = (0..pts.len()).collect();
        let key = |p: &[f64; 3]| {
            [
                (p[0] / SORT_BOX).floor() as i64,
                (p[1] / SORT_BOX).floor() as i64,
                (p[2] / SORT_BOX).floor() as i64,
            ]
        };
        order.sort_by_key(|&i| key(&pts[i].xyz));
        order.chunks(chunk.max(1)).map(|c| c.to_vec()).collect()
    }

    /// `(χ (nbf, np), ∇χ (3, nbf, np), ∇∇χ (3, 3, nbf, np))` at `pts`.
    #[allow(clippy::type_complexity)]
    pub(crate) fn eval(
        &self,
        pts: &[[f64; 3]],
    ) -> Result<(Array2<f64>, Array3<f64>, Array4<f64>), FerricError> {
        let np = pts.len();
        let nbf = self.nbf;
        let mut chi = Array2::<f64>::zeros((nbf, np));
        let mut dchi = Array3::<f64>::zeros((3, nbf, np));
        let mut ddchi = Array4::<f64>::zeros((3, 3, nbf, np));
        if np == 0 {
            return Ok((chi, dchi, ddchi));
        }
        let m = np as f64;
        let cen = [
            pts.iter().map(|p| p[0]).sum::<f64>() / m,
            pts.iter().map(|p| p[1]).sum::<f64>() / m,
            pts.iter().map(|p| p[2]).sum::<f64>() / m,
        ];
        let rad = pts.iter().map(|p| dist3(p, &cen)).fold(0.0, f64::max);
        let mut live: Vec<(usize, [f64; 3])> = Vec::new();
        for l in &self.trans {
            for (s, sh) in self.shells.iter().enumerate() {
                let c = [
                    sh.center[0] + l[0],
                    sh.center[1] + l[1],
                    sh.center[2] + l[2],
                ];
                if dist3(&c, &cen) <= rad + self.ext[s] {
                    live.push((s, c));
                }
            }
        }
        let mut buf = [0.0_f64; 15];
        let mut gbuf = [[0.0_f64; 15]; 3];
        let mut hbuf = [[0.0_f64; 15]; 9];
        for (g, p) in pts.iter().enumerate() {
            for &(s, c) in &live {
                let (dx, dy, dz) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
                if dx * dx + dy * dy + dz * dz > self.ext[s] * self.ext[s] {
                    continue;
                }
                let sh = &self.shells[s];
                let n = num_functions(sh.l, sh.pure);
                buf.fill(0.0);
                for row in gbuf.iter_mut() {
                    row.fill(0.0);
                }
                for row in hbuf.iter_mut() {
                    row.fill(0.0);
                }
                eval_shell_grad_hess(sh, dx, dy, dz, &mut buf[..n], &mut gbuf, &mut hbuf)?;
                let o = self.offsets[s];
                for i in 0..n {
                    chi[(o + i, g)] += buf[i];
                    for a in 0..3 {
                        dchi[(a, o + i, g)] += gbuf[a][i];
                        for b in 0..3 {
                            ddchi[(a, b, o + i, g)] += hbuf[a * 3 + b][i];
                        }
                    }
                }
            }
        }
        Ok((chi, dchi, ddchi))
    }
}

impl LatticeAoHess<'_> {
    /// Lattice-summed AOs, gradients and the image-resolved strain moments
    /// ([`AoStrainMoments`]) at `pts` with anchors `anchors` (`O_g`: the home
    /// atom for an atom-centred grid point, the point itself for a point at
    /// fixed fractional coordinates). Same live-shell list and extents as
    /// [`LatticeAoHess::eval`]; `hess` adds the Hessian moments (GGA).
    pub(crate) fn eval_strain(
        &self,
        pts: &[[f64; 3]],
        anchors: &[[f64; 3]],
        hess: bool,
    ) -> Result<AoStrainMoments, FerricError> {
        let np = pts.len();
        let nbf = self.nbf;
        if anchors.len() != np {
            return Err(FerricError::General(format!(
                "LatticeAoHess::eval_strain: {np} points but {} anchors",
                anchors.len()
            )));
        }
        let mut chi = Array2::<f64>::zeros((nbf, np));
        let mut dchi = Array3::<f64>::zeros((3, nbf, np));
        let mut m1 = Array4::<f64>::zeros((3, 3, nbf, np));
        let mut m2 = hess.then(|| Array4::<f64>::zeros((9, 3, nbf, np)));
        if np == 0 {
            return Ok(AoStrainMoments { chi, dchi, m1, m2 });
        }
        let m = np as f64;
        let cen = [
            pts.iter().map(|p| p[0]).sum::<f64>() / m,
            pts.iter().map(|p| p[1]).sum::<f64>() / m,
            pts.iter().map(|p| p[2]).sum::<f64>() / m,
        ];
        let rad = pts.iter().map(|p| dist3(p, &cen)).fold(0.0, f64::max);
        let mut live: Vec<(usize, [f64; 3])> = Vec::new();
        for l in &self.trans {
            for (s, sh) in self.shells.iter().enumerate() {
                let c = [
                    sh.center[0] + l[0],
                    sh.center[1] + l[1],
                    sh.center[2] + l[2],
                ];
                if dist3(&c, &cen) <= rad + self.ext[s] {
                    live.push((s, c));
                }
            }
        }
        let mut buf = [0.0_f64; 15];
        let mut gbuf = [[0.0_f64; 15]; 3];
        let mut hbuf = [[0.0_f64; 15]; 9];
        for (g, p) in pts.iter().enumerate() {
            let o = anchors[g];
            for &(s, c) in &live {
                let (dx, dy, dz) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
                if dx * dx + dy * dy + dz * dz > self.ext[s] * self.ext[s] {
                    continue;
                }
                let rel = [o[0] - c[0], o[1] - c[1], o[2] - c[2]];
                let sh = &self.shells[s];
                let n = num_functions(sh.l, sh.pure);
                buf.fill(0.0);
                for row in gbuf.iter_mut() {
                    row.fill(0.0);
                }
                for row in hbuf.iter_mut() {
                    row.fill(0.0);
                }
                eval_shell_grad_hess(sh, dx, dy, dz, &mut buf[..n], &mut gbuf, &mut hbuf)?;
                let off = self.offsets[s];
                for i in 0..n {
                    let mu = off + i;
                    chi[(mu, g)] += buf[i];
                    for a in 0..3 {
                        let da = gbuf[a][i];
                        dchi[(a, mu, g)] += da;
                        for b in 0..3 {
                            m1[(a, b, mu, g)] += da * rel[b];
                        }
                    }
                    if let Some(m2) = m2.as_mut() {
                        for k in 0..3 {
                            for a in 0..3 {
                                let h = hbuf[k * 3 + a][i];
                                for b in 0..3 {
                                    m2[(3 * k + a, b, mu, g)] += h * rel[b];
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(AoStrainMoments { chi, dchi, m1, m2 })
    }
}

/// Resolve a functional name for the periodic path: `(XcDef, exact-exchange
/// fraction)`. Refuses RSH, meta-GGA, VV10 and scaled composites (double
/// hybrids) by name.
pub fn resolve_periodic_functional(name: &str) -> Result<(XcDef, f64), FerricError> {
    let def = xc_def_from_name(name)
        .map_err(|e| FerricError::General(format!("periodic DFT: functional {name:?}: {e:?}")))?;
    if def.cam.is_some() {
        return Err(PeriodicDftError::Unsupported {
            feature: "range-separated hybrid",
            reason: format!(
                "{name} needs the attenuated (erf/erfc) periodic exchange, which Stage 2 does \
                 not build; use a global hybrid (e.g. PBE0) or a semilocal functional"
            ),
        }
        .into());
    }
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
            reason: format!("{name}: only LDA, GGA and global hybrids are supported"),
        }
        .into());
    }
    let a = def.b3lyp_mix.unwrap_or(0.0);
    Ok((def, a))
}

/// Configuration of [`PeriodicXc::new`].
#[derive(Debug, Clone, Copy)]
pub struct PeriodicXcConfig {
    /// AO value/gradient truncation per image shell.
    pub ao_threshold: f64,
    /// Budget (bytes) the AO cache is reserved against; `None` = unified.
    pub budget_bytes: Option<usize>,
}

impl Default for PeriodicXcConfig {
    fn default() -> Self {
        Self {
            ao_threshold: DEFAULT_AO_THRESHOLD,
            budget_bytes: None,
        }
    }
}

/// Semilocal XC on a [`PeriodicGrid`] with lattice-summed AOs: the
/// [`XcBuilder`] the Gamma RKS injects.
pub struct PeriodicXc {
    name: String,
    xc: XcDef,
    /// The same functional on `nspin = 2` libxc handles (UKS).
    xc_pol: XcDef,
    exx: f64,
    chunks: Vec<AoChunk>,
    nbf: usize,
    scratch: VxcScratch,
}

impl std::fmt::Debug for PeriodicXc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeriodicXc")
            .field("functional", &self.name)
            .field("exact_exchange_fraction", &self.exx)
            .field("nbf", &self.nbf)
            .field("npoints", &self.npoints())
            .finish()
    }
}

impl PeriodicXc {
    /// Resolve `functional`, evaluate and cache `χ^Γ`/`∇χ^Γ` of `bs` (the
    /// basis `prep` was built from, on `cell.mol()`) on `grid`.
    pub fn new(
        cell: &Cell,
        bs: &BasisSet,
        functional: &str,
        grid: &PeriodicGrid,
        cfg: &PeriodicXcConfig,
    ) -> Result<Self, FerricError> {
        let (xc, exx) = resolve_periodic_functional(functional)?;
        let xc_pol = xc_def_from_name_nspin(functional, 2).map_err(|e| {
            FerricError::General(format!(
                "periodic DFT: functional {functional:?} (spin-polarized): {e:?}"
            ))
        })?;
        let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
        let (chunks, nbf) = lattice_ao_chunks(cell, bs, grid, cfg.ao_threshold, &mut ledger)?;
        Ok(Self {
            name: functional.to_string(),
            xc,
            xc_pol,
            exx,
            chunks,
            nbf,
            scratch: VxcScratch::new(),
        })
    }

    /// The functional name as given.
    pub fn functional(&self) -> &str {
        &self.name
    }

    /// AO count.
    pub fn nbasis(&self) -> usize {
        self.nbf
    }

    /// Grid points cached.
    pub fn npoints(&self) -> usize {
        self.chunks.iter().map(|c| c.points.len()).sum()
    }

    /// Mean number of lattice translations with a live shell per chunk.
    pub fn mean_live_images(&self) -> f64 {
        let n = self.chunks.len().max(1) as f64;
        self.chunks.iter().map(|c| c.n_images as f64).sum::<f64>() / n
    }

    fn check_d(&self, d: &Array2<f64>) -> Result<(), FerricError> {
        if d.dim() != (self.nbf, self.nbf) {
            return Err(FerricError::General(format!(
                "PeriodicXc: density has shape {:?}, expected ({n}, {n})",
                d.dim(),
                n = self.nbf
            )));
        }
        Ok(())
    }

    /// `∫_cell ρ = Σ_g w_g ρ(r_g)` for the Gamma density `d` (= `tr(D S)` = N
    /// up to the grid error).
    pub fn integrate_density(&self, d: &Array2<f64>) -> Result<f64, FerricError> {
        self.check_d(d)?;
        Ok(self
            .chunks
            .iter()
            .map(|c| {
                let dens = eval_density_closed(d, &c.chi, &c.dchi);
                c.points
                    .iter()
                    .zip(dens.rho.iter())
                    .map(|(g, r)| g.weight * r)
                    .sum::<f64>()
            })
            .sum())
    }

    /// `S^grid_μν = Σ_g w_g χ^Γ_μ χ^Γ_ν` (≈ the lattice overlap).
    pub fn overlap_on_grid(&self) -> Array2<f64> {
        let mut s = Array2::<f64>::zeros((self.nbf, self.nbf));
        for c in &self.chunks {
            let mut wchi = c.chi.clone();
            for (g, p) in c.points.iter().enumerate() {
                wchi.column_mut(g).mapv_inplace(|v| v * p.weight);
            }
            s += &wchi.dot(&c.chi.t());
        }
        s
    }

    /// `(E_xc, V_xc)` at `d` (chunk sums of ferric-dft's
    /// `semilocal_vxc_closed`, which also symmetrises each chunk's V).
    pub fn eval(&mut self, d: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        self.check_d(d)?;
        let mut e = 0.0;
        let mut v = Array2::<f64>::zeros((self.nbf, self.nbf));
        for c in &self.chunks {
            let dens = eval_density_closed(d, &c.chi, &c.dchi);
            let (ec, vc) = semilocal_vxc_closed_scratch(
                &c.points,
                &c.chi,
                &c.dchi,
                &dens,
                None,
                &self.xc,
                &mut self.scratch,
            );
            e += ec;
            v += &vc;
        }
        if !e.is_finite() || v.iter().any(|x| !x.is_finite()) {
            return Err(FerricError::General(format!(
                "PeriodicXc: non-finite E_xc/V_xc ({e}) for {}",
                self.name
            )));
        }
        Ok((e, v))
    }
}

impl PeriodicXc {
    /// Spin-polarized `(E_xc, V_α, V_β)` at `(d_a, d_b)` (chunk sums of
    /// ferric-dft's `semilocal_vxc_polarized`, libxc `nspin = 2`; each chunk's
    /// `V_σ` symmetrised by the kernel). GGA: `V_σ` includes the `σ_αβ` cross
    /// term through `∇ρ_{σ'}`. Closed shell (`d_a = d_b = D/2`) reproduces
    /// [`PeriodicXc::eval`] up to the libxc polarized/unpolarized rounding.
    pub fn eval_polarized(
        &mut self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
        self.check_d(d_a)?;
        self.check_d(d_b)?;
        let mut e = 0.0;
        let mut v_a = Array2::<f64>::zeros((self.nbf, self.nbf));
        let mut v_b = Array2::<f64>::zeros((self.nbf, self.nbf));
        for c in &self.chunks {
            let dens = eval_density_uks(d_a, d_b, &c.chi, &c.dchi);
            let (ec, va, vb) = semilocal_vxc_polarized_scratch(
                &c.points,
                &c.chi,
                &c.dchi,
                &dens,
                None,
                &self.xc_pol,
                &mut self.scratch,
            );
            e += ec;
            v_a += &va;
            v_b += &vb;
        }
        if !e.is_finite() || v_a.iter().chain(v_b.iter()).any(|x| !x.is_finite()) {
            return Err(FerricError::General(format!(
                "PeriodicXc: non-finite polarized E_xc/V_xc ({e}) for {}",
                self.name
            )));
        }
        Ok((e, v_a, v_b))
    }

    /// The functional's global exact-exchange fraction.
    pub fn exact_exchange_fraction(&self) -> f64 {
        self.exx
    }
}

impl XcBuilder for PeriodicXc {
    fn build(&mut self, d: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        self.eval(d)
    }

    fn exact_exchange_fraction(&self) -> f64 {
        self.exx
    }

    fn build_polarized(
        &mut self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
        self.eval_polarized(d_a, d_b)
    }

    fn supports_polarized(&self) -> bool {
        true
    }
}

/// Borrowing adapter so [`gamma_rks`] keeps the [`PeriodicXc`] after the SCF.
struct XcRef<'a>(&'a mut PeriodicXc);

impl XcBuilder for XcRef<'_> {
    fn build(&mut self, d: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        self.0.eval(d)
    }

    fn exact_exchange_fraction(&self) -> f64 {
        self.0.exx
    }

    fn build_polarized(
        &mut self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
        self.0.eval_polarized(d_a, d_b)
    }

    fn supports_polarized(&self) -> bool {
        true
    }
}

/// Borrowing adapter over ANY [`XcBuilder`] (so [`gamma_uks_with_xc`] can
/// inject the caller's builder into two SCF stages and keep it afterwards).
pub(crate) struct DynXcRef<'b, 'c>(pub(crate) &'b mut (dyn XcBuilder + 'c));

impl XcBuilder for DynXcRef<'_, '_> {
    fn build(&mut self, d: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        self.0.build(d)
    }

    fn exact_exchange_fraction(&self) -> f64 {
        self.0.exact_exchange_fraction()
    }

    fn build_polarized(
        &mut self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
        self.0.build_polarized(d_a, d_b)
    }

    fn supports_polarized(&self) -> bool {
        self.0.supports_polarized()
    }
}

/// Configuration for [`gamma_rks`].
#[derive(Debug, Clone)]
pub struct GammaRksConfig {
    /// LDA / GGA / global-hybrid name (ferric-dft friendly names or libxc
    /// identifiers), e.g. `"LDA"` (= PySCF `"LDA,VWN"`), `"PBE"`, `"PBE0"`.
    pub functional: String,
    /// The periodic grid.
    pub grid: PeriodicGridConfig,
    /// Exchange-divergence treatment of the injected K (only a hybrid
    /// consumes it, scaled by its exact-exchange fraction).
    pub exxdiv: ExxDiv,
    /// AO truncation and budget for the XC AO cache.
    pub xc: PeriodicXcConfig,
    /// SCF knobs, validated by `validate_injected` plus the molecular-grid
    /// fields refused here.
    pub scf: RhfConfig,
}

impl GammaRksConfig {
    /// Defaults (grid (75, 302) SSF, exxdiv ewald, hcore guess, tight
    /// convergence) for `functional`.
    pub fn new(functional: &str) -> Self {
        Self {
            functional: functional.to_string(),
            grid: PeriodicGridConfig::default(),
            exxdiv: ExxDiv::Ewald,
            xc: PeriodicXcConfig::default(),
            scf: RhfConfig {
                use_sad_guess: false,
                density_conv: 1e-10,
                max_iter: 200,
                ..Default::default()
            },
        }
    }
}

/// Result of [`gamma_rks`].
#[derive(Debug, Clone)]
#[must_use]
pub struct GammaRksResult {
    /// The converged SCF (energy includes E_xc and E_nn).
    pub scf: ScfResult,
    /// `v_M` of the cell (applied to K iff `exxdiv = ewald`).
    pub madelung: f64,
    /// The functional's global exact-exchange fraction.
    pub exact_exchange_fraction: f64,
    /// Grid points used.
    pub n_grid_points: usize,
    /// Neighbour cutoff `D` used (Bohr).
    pub neighbour_cutoff: f64,
    /// `Σ_g w_g ρ(r_g)` at the converged density (vs N: the grid error).
    pub electrons_on_grid: f64,
    /// `E_xc` at the converged density.
    pub e_xc: f64,
}

/// Gamma-point closed-shell KS-DFT of `cell` on the lattice one-electron
/// terms `hc` and the J/K of `ints` (module doc). `prep` must be the basis
/// `hc`/`ints` were built in, from `cell.mol()`.
///
/// Errors (by name) on open-shell cells, refused functionals / grid knobs /
/// SCF features, a neighbour cutoff below the covering bound, budget
/// overruns, and a non-converged SCF.
pub fn gamma_rks(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    cfg: &GammaRksConfig,
) -> Result<GammaRksResult, FerricError> {
    validate_injected(&cfg.scf)?;
    let refuse = |feature: &'static str, reason: &str| -> Result<(), FerricError> {
        Err(PeriodicDftError::Unsupported {
            feature,
            reason: reason.to_string(),
        }
        .into())
    };
    if cfg.scf.xc_omega.is_some() {
        refuse(
            "scf.xc_omega",
            "range-separated functionals are refused on the periodic path",
        )?;
    }
    if cfg.scf.dft_grid.is_some() {
        refuse(
            "scf.dft_grid",
            "the molecular grid is never built here; set GammaRksConfig.grid",
        )?;
    }
    if cfg.scf.nlc_grid.is_some() {
        refuse("scf.nlc_grid", "no periodic VV10/NLC grid exists")?;
    }
    let mol = cell.mol();
    if mol.nelec() % 2 != 0 || mol.multiplicity != 1 {
        refuse(
            "open shell",
            "gamma_rks is closed-shell RKS only; use gamma_uks for an open-shell cell",
        )?;
    }
    // Cheap name checks before the grid is built.
    resolve_periodic_functional(&cfg.functional)?;
    let grid = PeriodicGrid::build(cell, &cfg.grid)?;
    let mut pxc = PeriodicXc::new(cell, prep.basis_set(), &cfg.functional, &grid, &cfg.xc)?;
    let v_m = madelung_constant(cell)?;
    let applied = match cfg.exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => v_m,
    };
    let ctx = ParallelContext::default();
    // Required by the solver signature; never read for integrals here.
    let bounds = SchwarzBounds::compute(Operator::coulomb(), prep)?;
    let (j, k) = crate::uhf::builders(ints, applied);
    let scf = {
        let inj = PeriodicInjection {
            s: hc.s.clone(),
            h: hc.h.clone(),
            vnn: hc.enn,
            j,
            k,
            xc: Some(Box::new(XcRef(&mut pxc))),
        };
        solve_rhf_injected(&ctx, mol, prep, Operator::coulomb(), &bounds, &cfg.scf, inj)?
    };
    if !scf.converged {
        return Err(FerricError::General(format!(
            "gamma_rks: {} SCF did not converge in {} iterations (last E = {})",
            cfg.functional, scf.iterations, scf.energy
        )));
    }
    let electrons_on_grid = pxc.integrate_density(&scf.density_total)?;
    let (e_xc, _) = pxc.eval(&scf.density_total)?;
    Ok(GammaRksResult {
        madelung: v_m,
        exact_exchange_fraction: pxc.exx,
        n_grid_points: grid.len(),
        neighbour_cutoff: grid.neighbour_cutoff(),
        electrons_on_grid,
        e_xc,
        scf,
    })
}

// ───────────────────────────────────────────────────── Stage 5: Gamma UKS

/// Configuration for [`gamma_uks`] / [`gamma_uks_with_xc`].
#[derive(Debug, Clone)]
pub struct GammaUksConfig {
    /// LDA / GGA / global-hybrid name, as [`GammaRksConfig::functional`].
    /// Ignored by [`gamma_uks_with_xc`] (the caller's builder is the XC).
    pub functional: String,
    /// The periodic grid (ignored by [`gamma_uks_with_xc`]).
    pub grid: PeriodicGridConfig,
    /// Exchange-divergence treatment of the injected K (consumed only by a
    /// hybrid, scaled by its exact-exchange fraction).
    pub exxdiv: ExxDiv,
    /// Start strategy for `exxdiv = ewald` with `a > 0` (default
    /// [`EwaldStart::Staged`], FINDINGS "Iteration 10": the staged start never
    /// stagnated and is always safe). Ignored for `a = 0` (no K is built, so
    /// the Madelung term cannot act) and for `exxdiv = none`.
    pub ewald_start: EwaldStart,
    /// AO truncation and budget for the XC AO cache (ignored by
    /// [`gamma_uks_with_xc`]).
    pub xc: PeriodicXcConfig,
    /// SCF knobs, validated by `validate_injected_uhf` plus the molecular-grid
    /// fields refused here.
    pub scf: RhfConfig,
    /// Optional per-spin starting MOs `(C_α, C_β)`, each `(nao, nao)`; used
    /// by the first stage only.
    pub initial_mos: Option<(Array2<f64>, Array2<f64>)>,
}

impl GammaUksConfig {
    /// Defaults (grid (75, 302) SSF, exxdiv ewald, staged start, hcore
    /// guess, tight convergence) for `functional`.
    pub fn new(functional: &str) -> Self {
        let rks = GammaRksConfig::new(functional);
        Self {
            functional: rks.functional,
            grid: rks.grid,
            exxdiv: rks.exxdiv,
            ewald_start: EwaldStart::Staged,
            xc: rks.xc,
            scf: rks.scf,
            initial_mos: None,
        }
    }
}

/// Grid diagnostics of [`gamma_uks`] (absent from [`gamma_uks_with_xc`]).
#[derive(Debug, Clone, PartialEq)]
pub struct GammaUksGridInfo {
    /// Grid points used.
    pub n_grid_points: usize,
    /// Neighbour cutoff `D` used (Bohr).
    pub neighbour_cutoff: f64,
    /// `Σ_g w_g ρ(r_g)` of the converged TOTAL density (vs N: grid error).
    pub electrons_on_grid: f64,
}

/// Result of [`gamma_uks`] / [`gamma_uks_with_xc`].
#[derive(Debug, Clone)]
#[must_use]
pub struct GammaUksResult {
    /// The final SCF (requested `exxdiv`; energy includes E_xc and E_nn).
    pub scf: ScfResult,
    /// The `exxdiv = none` first stage, when the staged start ran.
    pub none_stage: Option<ScfResult>,
    /// `v_M` of the cell (applied, times `a`, iff `exxdiv = ewald`).
    pub madelung: f64,
    /// The functional's global exact-exchange fraction `a`.
    pub exact_exchange_fraction: f64,
    /// `(N_α, N_β)` from the cell's charge and multiplicity.
    pub nocc: (usize, usize),
    /// `⟨S²⟩` of the KS determinant with the lattice overlap.
    pub s2: f64,
    /// Occupation-aware per-spin gaps against `a·v_M_applied`
    /// ([`crate::uhf::occupation_gaps`]).
    pub gaps: SpinGapReport,
    /// Semilocal `E_xc` at the converged `(D_α, D_β)`.
    pub e_xc: f64,
    /// Grid diagnostics ([`gamma_uks`] only).
    pub grid: Option<GammaUksGridInfo>,
}

/// Refuse the molecular-grid `RhfConfig` knobs the periodic KS paths never
/// honour (shared by [`gamma_rks`]-style entries; `validate_injected*` covers
/// the rest).
pub(crate) fn refuse_molecular_grid_knobs(scf: &RhfConfig) -> Result<(), FerricError> {
    let refuse = |feature: &'static str, reason: &str| -> Result<(), FerricError> {
        Err(PeriodicDftError::Unsupported {
            feature,
            reason: reason.to_string(),
        }
        .into())
    };
    if scf.xc_omega.is_some() {
        refuse(
            "scf.xc_omega",
            "range-separated functionals are refused on the periodic path",
        )?;
    }
    if scf.dft_grid.is_some() {
        refuse(
            "scf.dft_grid",
            "the molecular grid is never built here; set GammaUksConfig.grid",
        )?;
    }
    if scf.nlc_grid.is_some() {
        refuse("scf.nlc_grid", "no periodic VV10/NLC grid exists")?;
    }
    Ok(())
}

/// Gamma-point open-shell (or closed-shell) UKS of `cell` (spin state from
/// `cell.mol()`'s charge and multiplicity) on the lattice one-electron terms
/// `hc` and the J/K of `ints`, with [`PeriodicXc`] built from
/// `cfg.functional` on `cfg.grid` (module doc, "Open shell").
///
/// `prep` must be the basis `hc`/`ints` were built in, from `cell.mol()`.
/// Errors (by name) on refused functionals / grid knobs / SCF features, a
/// neighbour cutoff below the covering bound, budget overruns, and a
/// non-converged SCF stage. Warns on stderr when the occupation-aware
/// per-spin gap is below `a·v_M` (possible Ewald trap).
pub fn gamma_uks(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    cfg: &GammaUksConfig,
) -> Result<GammaUksResult, FerricError> {
    ferric_scf::uhf::validate_injected_uhf(&cfg.scf).map_err(FerricError::from)?;
    refuse_molecular_grid_knobs(&cfg.scf)?;
    nocc_ab(cell.mol())?;
    // Cheap name checks before the grid is built.
    resolve_periodic_functional(&cfg.functional)?;
    let grid = PeriodicGrid::build(cell, &cfg.grid)?;
    let mut pxc = PeriodicXc::new(cell, prep.basis_set(), &cfg.functional, &grid, &cfg.xc)?;
    let mut out = gamma_uks_with_xc(cell, prep, hc, ints, &mut pxc, cfg)?;
    let electrons_on_grid = pxc.integrate_density(&out.scf.density_total)?;
    out.grid = Some(GammaUksGridInfo {
        n_grid_points: grid.len(),
        neighbour_cutoff: grid.neighbour_cutoff(),
        electrons_on_grid,
    });
    Ok(out)
}

/// [`gamma_uks`] with a caller-supplied spin-polarized [`XcBuilder`] instead
/// of a [`PeriodicXc`] built from `cfg.functional` (`cfg.functional`,
/// `cfg.grid` and `cfg.xc` are ignored). Everything else — the injected SCF,
/// the staged ewald start for `a > 0`, `⟨S²⟩` and the occupation-aware gap
/// check against `a·v_M` — is identical. A builder with `a = 1` and zero
/// `E_xc`/`V_σ` is exactly the Gamma UHF (the equivalence test).
pub fn gamma_uks_with_xc(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    xc: &mut dyn XcBuilder,
    cfg: &GammaUksConfig,
) -> Result<GammaUksResult, FerricError> {
    ferric_scf::uhf::validate_injected_uhf(&cfg.scf).map_err(FerricError::from)?;
    refuse_molecular_grid_knobs(&cfg.scf)?;
    if !xc.supports_polarized() {
        return Err(PeriodicDftError::Unsupported {
            feature: "closed-shell XcBuilder",
            reason: "gamma_uks needs XcBuilder::build_polarized (supports_polarized() is false)"
                .into(),
        }
        .into());
    }
    let mol = cell.mol();
    let (na, nb) = nocc_ab(mol)?;
    let a = xc.exact_exchange_fraction();
    if !(a.is_finite() && (0.0..=1.0).contains(&a)) {
        return Err(FerricError::General(format!(
            "gamma_uks: exact-exchange fraction must be in [0, 1], got {a}"
        )));
    }
    let v_m = madelung_constant(cell)?;
    let applied = match cfg.exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => v_m,
    };
    let ctx = ParallelContext::default();
    // Required by the solver signature; never read for integrals here.
    let bounds = SchwarzBounds::compute(Operator::coulomb(), prep)?;
    let mut run = |vm: f64, init: Option<(&Array2<f64>, &Array2<f64>)>| {
        let (j, k) = crate::uhf::builders(ints, vm);
        let inj = PeriodicInjection {
            s: hc.s.clone(),
            h: hc.h.clone(),
            vnn: hc.enn,
            j,
            k,
            xc: Some(Box::new(DynXcRef(&mut *xc))),
        };
        solve_uhf_injected(&ctx, mol, prep, &bounds, &cfg.scf, inj, init)
    };
    let init = cfg.initial_mos.as_ref().map(|(ca, cb)| (ca, cb));
    let staged = cfg.exxdiv == ExxDiv::Ewald && cfg.ewald_start == EwaldStart::Staged && a > 0.0;
    let (scf, none_stage) = if staged {
        let first = run(0.0, init)?;
        let cb = first.mos_beta.clone().ok_or_else(|| {
            FerricError::General("gamma_uks: the none stage returned no beta MOs".into())
        })?;
        let ca = first.mos_alpha.clone();
        let second = run(applied, Some((&ca, &cb)))?;
        (second, Some(first))
    } else {
        (run(applied, init)?, None)
    };
    let shift = a * applied;
    let gaps = occupation_gaps(&scf, &hc.s, shift);
    if !gaps.satisfied() {
        eprintln!(
            "gamma_uks WARNING: occupation-aware per-spin gap below a*v_M (gap_alpha {:?}, \
             gap_beta {:?}, a*v_M {shift:.6}, margin {:?}). Under exxdiv=none this state has a \
             hole below the Fermi level: likely the Gamma Ewald trap (FINDINGS Iterations 6, \
             10). Use EwaldStart::Staged or a better initial_mos.",
            gaps.gap_alpha, gaps.gap_beta, gaps.margin
        );
    }
    let s2 = spin_square(&scf, &hc.s, na, nb)?;
    let d_b = scf.density_beta.as_ref().ok_or_else(|| {
        FerricError::General("gamma_uks: the SCF returned no beta density".into())
    })?;
    let (e_xc, _, _) = xc.build_polarized(&scf.density_alpha, d_b)?;
    Ok(GammaUksResult {
        madelung: v_m,
        exact_exchange_fraction: a,
        nocc: (na, nb),
        s2,
        gaps,
        e_xc,
        grid: None,
        none_stage,
        scf,
    })
}
