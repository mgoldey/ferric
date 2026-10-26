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
//! The default stays at 302; 434 and 590 (and every other order in
//! [`SUPPORTED_LEBEDEV_ORDERS`], the one list `ferric_dft::prune` owns) are
//! accepted as refinement grids. ferric on H2 a = 4 LDA reproduces the
//! prototype's trend (`tests/pbc_rks.rs::
//! larger_lebedev_orders_refine_the_periodic_energy`). PROVISIONAL: both cells
//! are H-only toys; a real solid with core electrons has not been swept
//! (FINDINGS says so explicitly).
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
//! ([`crate::uhf::occupation_gaps`]). ROKS is [`crate::rohf`] (`gamma_roks`,
//! and `gamma_roks_rsh` for range-separated hybrids).
//!
//! Refused by name: range-separated hybrids in every entry but closed-shell
//! [`gamma_rks_rsh`] (the attenuated K is the k = 0 of the 1x1x1 k-point machinery, [`crate::gamma_rsh`]; open-shell [`gamma_uks_rsh`] and [`crate::rohf::gamma_roks_rsh`] take the same exchange per spin; the closed-shell dense-AFT force and stress are [`crate::gamma_rsh::gamma_rks_rsh_gradient`] and [`crate::gamma_rsh::gamma_rks_rsh_stress`], the open-shell UKS dense-AFT force is [`crate::gamma_rsh::gamma_uks_rsh_gradient`] (wired to the bindings through [`check_gamma_uks_request`]) and the ROKS one [`crate::gamma_rsh::gamma_roks_rsh_gradient`] (library only: the bindings refuse it); the open-shell UKS stress is [`crate::gamma_rsh::gamma_uks_rsh_stress`] (library only)),
//! meta-GGA except the validated SCAN / r2SCAN / TPSS in closed-shell [`gamma_rks`] (tau from D and ∇χ^Γ, energy anchored to the 1x1x1 k-mesh and PySCF; the force and stress are the 1x1x1 k-point route [`crate::gamma_mgga`] on the dense-AFT J, the native Gamma XC derivatives refuse it; UKS and ROKS refuse it), VV10 / double hybrids, grid pruning, Newton /
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
use crate::timing::{CallClock, PbcTimings, StageClock, StageTiming};
use crate::uhf::{
    nocc_ab, occupation_gaps, spin_square, EwaldStart, GammaUhfIntegrals, SpinGapReport,
};
use ferric_core::basis::{num_functions, BasisSet};
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_dft::becke::{
    partition_weight_over, partition_weight_over_and_grad, partition_weight_over_switched,
    partition_weight_over_switched_and_grad, NeighbourAtom, NeighbourSwitch, PartitionScheme,
};
use ferric_dft::density_on_grid::{eval_density_closed, eval_density_uks, eval_tau_closed};
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
use num_complex::Complex64;
use rayon::prelude::*;

/// Default radial shells (see the module doc's table).
pub const DEFAULT_PERIODIC_N_RADIAL: usize = 75;
/// Default Lebedev order (module doc); 434 / 590 are refinement grids.
pub const DEFAULT_PERIODIC_N_ANGULAR: usize = 302;
/// Floor of the automatic neighbour cutoff `D` (Bohr): the prototype's value
/// for H2 a = 4 (`D = 10` and `14` gave identical grids there).
pub const DEFAULT_NEIGHBOUR_CUTOFF: f64 = 10.0;
/// Default AO truncation: every image shell whose value and gradient bound
/// is below this at a point is skipped (prototype `ao_thresh`).
pub const DEFAULT_AO_THRESHOLD: f64 = 1e-15;
/// Lebedev orders ferric's table implements (`lebedev()` panics otherwise,
/// so the config is checked against this list first). Re-exported from
/// `ferric_dft::prune`, the single list every grid consumer validates against.
pub use ferric_dft::prune::SUPPORTED_LEBEDEV_ORDERS;
/// Points per AO/XC chunk.
pub(crate) const CHUNK: usize = 512;
/// Spatial sort box edge (Bohr) that groups points into compact chunks.
pub(crate) const SORT_BOX: f64 = 2.0;

/// A Stage-2 periodic-DFT refusal or grid-construction error, by name.
#[derive(Debug, Clone, PartialEq)]
pub enum PeriodicDftError {
    /// `D` is below the covering-radius bound plus the neighbour switch
    /// width: some point of space would have no fully present atom, so the
    /// partition could be empty there and that region's integrand silently
    /// lost.
    NeighbourCutoffTooSmall {
        /// The requested cutoff (Bohr).
        cutoff: f64,
        /// [`covering_radius_bound`] of the cell plus
        /// `PeriodicGridConfig::neighbour_switch` (Bohr).
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
                 covering-radius bound + neighbour_switch = {required:.6} Bohr (the bound is the \
                 largest point-to-nearest-atom distance); points with no fully present atom \
                 would get zero weight. Use neighbour_cutoff >= {required:.6} (or None for the \
                 automatic value)"
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

pub(crate) fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
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
    /// Width `W` (Bohr) of the smooth taper on the neighbour mask: an image
    /// atom at distance `d` from a point takes part in the partition with
    /// presence `f(d)` that falls from 1 at `D − W` to 0 at `D`
    /// ([`NeighbourSwitch`], `ferric_dft::becke::partition_weight_over_switched`),
    /// so the energy, force and stress stay continuous as images cross `D`;
    /// the force and stress carry the taper's derivative. `0.0` (default) is
    /// the hard cutoff, bit-identical to the unswitched grid. With `W > 0`
    /// the automatic `D` is `max(DEFAULT_NEIGHBOUR_CUTOFF, bound) + W` (the
    /// fully-present zone is the hard default's `D`) and an explicit `D` must
    /// satisfy `D − W >= covering_radius_bound`. The taper's energy cost is
    /// that of a cutoff between `D − W` and `D` (measured,
    /// `tests/validation_pbc_grid_switch.rs`). Build cost: the switched
    /// kernel at the same `D` is ~1.6x the hard one, and the automatic `D`
    /// grows by `W`, so the neighbour count rises ~`(1 + W/D)³` (measured
    /// Li–H, cubic 4 Bohr, (75, 302), 3 threads: hard D = 10 0.29 s, 133
    /// neighbours/point; switched D = 10, W = 2 0.46 s; automatic D = 12,
    /// W = 2 0.87 s, 232 neighbours/point; hard D = 12 0.52 s).
    pub neighbour_switch: f64,
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
            neighbour_switch: 0.0,
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

    /// Rejects configurations the periodic grid does not support: any angular
    /// pruning, `n_radial == 0`, an `n_angular` outside `SUPPORTED_LEBEDEV_ORDERS`,
    /// and a non-finite or non-positive `neighbour_cutoff`.
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
        if !(self.neighbour_switch.is_finite() && self.neighbour_switch >= 0.0) {
            return Err(PeriodicDftError::InvalidGrid {
                field: "neighbour_switch",
                reason: format!("must be finite and >= 0, got {}", self.neighbour_switch),
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
    resolve_neighbour_cutoff_switched(cell, requested, 0.0)
}

/// [`resolve_neighbour_cutoff`] for a neighbour switch of width `width`
/// (Bohr): the fully-present zone `D − width` must cover every point of
/// space, so the requirement on `D` is `bound + width` (the error reports
/// that as `required`), and the automatic value is
/// `max(DEFAULT_NEIGHBOUR_CUTOFF, bound) + width`, so the fully-present zone
/// is the hard default's cutoff. `width = 0` is [`resolve_neighbour_cutoff`].
pub fn resolve_neighbour_cutoff_switched(
    cell: &Cell,
    requested: Option<f64>,
    width: f64,
) -> Result<(f64, f64), PeriodicDftError> {
    let bound = covering_radius_bound(cell);
    let required = bound + width;
    match requested {
        None => Ok((DEFAULT_NEIGHBOUR_CUTOFF.max(bound) + width, bound)),
        Some(d) if d >= required => Ok((d, bound)),
        Some(d) => Err(PeriodicDftError::NeighbourCutoffTooSmall {
            cutoff: d,
            required,
        }),
    }
}

/// The partition kernel a grid build uses: the unswitched one for a hard
/// cutoff (width 0, bit-identical to the pre-switch grid) or the switched one.
#[derive(Debug, Clone, Copy)]
struct PartitionKernel {
    scheme: PartitionScheme,
    switch: Option<NeighbourSwitch>,
}

impl PartitionKernel {
    fn new(cfg: &PeriodicGridConfig, d_cut: f64) -> Self {
        let switch = (cfg.neighbour_switch > 0.0).then_some(NeighbourSwitch {
            cutoff: d_cut,
            width: cfg.neighbour_switch,
        });
        Self {
            scheme: cfg.partition,
            switch,
        }
    }

    fn weight(&self, nb: &[NeighbourAtom], h: usize, xyz: [f64; 3]) -> f64 {
        match self.switch {
            None => partition_weight_over(self.scheme, nb, h, xyz),
            Some(sw) => partition_weight_over_switched(self.scheme, nb, h, xyz, sw),
        }
    }

    /// `(w, ∂w/∂X_k)`; `w` is bit-identical to [`Self::weight`].
    fn weight_and_grad(
        &self,
        nb: &[NeighbourAtom],
        h: usize,
        xyz: [f64; 3],
    ) -> (f64, Vec<[f64; 3]>) {
        match self.switch {
            None => {
                let w = partition_weight_over(self.scheme, nb, h, xyz);
                (w, partition_weight_over_and_grad(self.scheme, nb, h, xyz).1)
            }
            Some(sw) => partition_weight_over_switched_and_grad(self.scheme, nb, h, xyz, sw),
        }
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
            |kern, nb, _cells, h, xyz, _home, _w_rl| (kern.weight(nb, h, xyz), ()),
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

    /// The grid of a supercell built from translated copies of this one:
    /// every point is repeated at `r + t` for each `t` in `shifts`, weights
    /// unchanged (so `Σ w` is `shifts.len()` times this grid's). For a
    /// lattice-periodic integrand `Σ_t Σ_g w_g f(r_g + t) = N Σ_g w_g f(r_g)`
    /// exactly, which makes a Gamma-point supercell KS calculation on it an
    /// exact anchor for the k-mesh one on this grid. `home_atom` keeps the
    /// CELL atom index (energy use only; the gradient bookkeeping is not
    /// valid for the copies).
    pub fn replicated(&self, shifts: &[[f64; 3]]) -> Self {
        let mut points = Vec::with_capacity(self.points.len() * shifts.len());
        for t in shifts {
            for p in &self.points {
                points.push(GridPoint {
                    xyz: [p.xyz[0] + t[0], p.xyz[1] + t[1], p.xyz[2] + t[2]],
                    weight: p.weight,
                    home_atom: p.home_atom,
                });
            }
        }
        Self {
            points,
            neighbour_cutoff: self.neighbour_cutoff,
            covering_bound: self.covering_bound,
            n_generated: self.n_generated * shifts.len(),
            mean_neighbours: self.mean_neighbours,
        }
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
    F: Fn(&PartitionKernel, &[NeighbourAtom], &[usize], usize, [f64; 3], usize, f64) -> (f64, T)
        + Sync,
{
    let (d_cut, bound) =
        resolve_neighbour_cutoff_switched(cell, cfg.neighbour_cutoff, cfg.neighbour_switch)?;
    let kern = PartitionKernel::new(cfg, d_cut);
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
                let (w, payload) = per_point(&kern, &nb, &nb_cell, h, xyz, home, w_rl);
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
        |kern, nb, nb_cell, h, xyz, home, w_rl| {
            // The weight through the energy's own kernel (bit-identical grid).
            let (w, dw) = kern.weight_and_grad(nb, h, xyz);
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
        |kern, nb, _nb_cell, h, xyz, _home, w_rl| {
            // The weight through the energy's own kernel (bit-identical grid).
            let (w, dw) = kern.weight_and_grad(nb, h, xyz);
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
pub(crate) fn shell_extent(sh: &LocatedShell, thresh: f64) -> f64 {
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

/// Lattice translations `L` that can bring an AO image within `r_ao` of any
/// of `pts` (`|L| <= rho + R_ao + spread`), shared by the Gamma and Bloch AO
/// caches.
pub(crate) fn image_translations(
    cell: &Cell,
    pts: &[GridPoint],
    r_ao: f64,
    ledger: &mut Ledger,
) -> Result<Vec<[f64; 3]>, FerricError> {
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
    Ok(cell
        .translations(rcut_l + 2.0 * spread)?
        .into_iter()
        .filter(|l| dot3(l, l).sqrt() <= rcut_l + spread)
        .collect())
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

    let trans = image_translations(cell, pts, r_ao, ledger)?;

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

/// Which Bloch phase [`LatticeAoHess::eval_bloch`] applies to an image
/// translation `L` at wave vector `k`: the energy's `e^{+ik·L}`, or one of the
/// two deliberate defects of the k-point XC force tests.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlochPhase {
    /// `e^{+ik·L}` ([`crate::kdft`]'s convention).
    True,
    /// `1` (no phase).
    None,
    /// `e^{-ik·L}`.
    Conj,
}

/// Bloch AOs on one point chunk, one entry per k (`φ^k_μ = Σ_L e^{ik·L}
/// χ_μ(r − L)` and its derivatives).
pub(crate) struct BlochAoHess {
    /// `chi[k]`: `(nbf, np)`.
    pub(crate) chi: Vec<Array2<Complex64>>,
    /// `dchi[k]`: `(3, nbf, np)`.
    pub(crate) dchi: Vec<Array3<Complex64>>,
    /// `ddchi[k]`: `(3, 3, nbf, np)`; empty without `hess`.
    pub(crate) ddchi: Vec<Array4<Complex64>>,
}

impl LatticeAoHess<'_> {
    /// Bloch-phased AOs, gradients and (with `hess`) Hessians at `pts` for
    /// every wave vector in `kpts`. The live image-shell list, the per-shell
    /// extents and the image translations are those of [`Self::eval`] (and so
    /// of the energy's Bloch AO cache, [`crate::kdft`]); the phase of an image
    /// shell centred at `c_s + L` is applied once per image and looped over k.
    pub(crate) fn eval_bloch(
        &self,
        pts: &[[f64; 3]],
        kpts: &[[f64; 3]],
        phase: BlochPhase,
        hess: bool,
    ) -> Result<BlochAoHess, FerricError> {
        let np = pts.len();
        let nbf = self.nbf;
        let nk = kpts.len();
        let mut chi = vec![Array2::<Complex64>::zeros((nbf, np)); nk];
        let mut dchi = vec![Array3::<Complex64>::zeros((3, nbf, np)); nk];
        let mut ddchi = if hess {
            vec![Array4::<Complex64>::zeros((3, 3, nbf, np)); nk]
        } else {
            Vec::new()
        };
        if np == 0 {
            return Ok(BlochAoHess { chi, dchi, ddchi });
        }
        let m = np as f64;
        let cen = [
            pts.iter().map(|p| p[0]).sum::<f64>() / m,
            pts.iter().map(|p| p[1]).sum::<f64>() / m,
            pts.iter().map(|p| p[2]).sum::<f64>() / m,
        ];
        let rad = pts.iter().map(|p| dist3(p, &cen)).fold(0.0, f64::max);
        // e^{ik·L} per (translation, k), filled for translations with a live
        // shell; live = (shell, shifted centre, translation index).
        let mut table: Vec<Vec<Complex64>> = vec![Vec::new(); self.trans.len()];
        let mut live: Vec<(usize, [f64; 3], usize)> = Vec::new();
        for (il, l) in self.trans.iter().enumerate() {
            for (s, sh) in self.shells.iter().enumerate() {
                let c = [
                    sh.center[0] + l[0],
                    sh.center[1] + l[1],
                    sh.center[2] + l[2],
                ];
                if dist3(&c, &cen) <= rad + self.ext[s] {
                    if table[il].is_empty() {
                        table[il] = kpts
                            .iter()
                            .map(|k| {
                                let a = k[0] * l[0] + k[1] * l[1] + k[2] * l[2];
                                match phase {
                                    BlochPhase::True => Complex64::new(a.cos(), a.sin()),
                                    BlochPhase::None => Complex64::new(1.0, 0.0),
                                    BlochPhase::Conj => Complex64::new(a.cos(), -a.sin()),
                                }
                            })
                            .collect();
                    }
                    live.push((s, c, il));
                }
            }
        }
        let mut buf = [0.0_f64; 15];
        let mut gbuf = [[0.0_f64; 15]; 3];
        let mut hbuf = [[0.0_f64; 15]; 9];
        for (g, p) in pts.iter().enumerate() {
            for &(s, c, il) in &live {
                let ph = &table[il];
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
                if hess {
                    for row in hbuf.iter_mut() {
                        row.fill(0.0);
                    }
                    eval_shell_grad_hess(sh, dx, dy, dz, &mut buf[..n], &mut gbuf, &mut hbuf)?;
                } else {
                    eval_shell_and_grad(sh, dx, dy, dz, &mut buf[..n], &mut gbuf)?;
                }
                let o = self.offsets[s];
                for (k, f) in ph.iter().enumerate() {
                    for i in 0..n {
                        chi[k][(o + i, g)] += *f * buf[i];
                        for a in 0..3 {
                            dchi[k][(a, o + i, g)] += *f * gbuf[a][i];
                            if hess {
                                for b in 0..3 {
                                    ddchi[k][(a, b, o + i, g)] += *f * hbuf[a * 3 + b][i];
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(BlochAoHess { chi, dchi, ddchi })
    }
}

/// Bloch AOs and their image-resolved strain moments on one point chunk, one
/// entry per k ([`LatticeAoHess::eval_bloch_strain`]).
pub(crate) struct BlochAoStrain {
    /// `chi[k]`: `(nbf, np)`.
    pub(crate) chi: Vec<Array2<Complex64>>,
    /// `dchi[k]`: `(3, nbf, np)`; `(3, nbf, 0)` without `hess`.
    pub(crate) dchi: Vec<Array3<Complex64>>,
    /// `m1[k][(a, b, μ, g)] = Σ_L e^{ik·L} ∂_aχ_μ(r_g − X_L) (O_g − X_L)_b`
    /// `(3, 3, nbf, np)` = `dφ^k_μ(r_g)/dε_ab` at fixed fractional `k`.
    pub(crate) m1: Vec<Array4<Complex64>>,
    /// `m2[k][(3i + a, b, μ, g)] = Σ_L e^{ik·L} ∂_i∂_aχ_μ(r_g − X_L) (O_g − X_L)_b`
    /// `(9, 3, nbf, np)` = `d(∂_iφ^k_μ)/dε_ab`; empty without `hess`.
    pub(crate) m2: Vec<Array4<Complex64>>,
}

impl LatticeAoHess<'_> {
    /// Bloch-phased AOs and strain moments at `pts` with anchors `anchors`
    /// (`O_g`: the home atom of an atom-centred grid point, the point itself on
    /// a uniform grid) for every wave vector in `kpts`: the complex twin of
    /// [`Self::eval_strain`] with the phase `e^{ik·L}` applied once per image
    /// (the phase is strain-invariant at fixed fractional k, so it multiplies
    /// the moments unchanged). Same live-shell list, extents and translations
    /// as [`Self::eval`] and [`Self::eval_bloch`]. `hess` adds `∇φ` and the
    /// Hessian moments (GGA).
    pub(crate) fn eval_bloch_strain(
        &self,
        pts: &[[f64; 3]],
        anchors: &[[f64; 3]],
        kpts: &[[f64; 3]],
        phase: BlochPhase,
        hess: bool,
    ) -> Result<BlochAoStrain, FerricError> {
        let np = pts.len();
        let nbf = self.nbf;
        let nk = kpts.len();
        if anchors.len() != np {
            return Err(FerricError::General(format!(
                "LatticeAoHess::eval_bloch_strain: {np} points but {} anchors",
                anchors.len()
            )));
        }
        let nd = if hess { np } else { 0 };
        let mut chi = vec![Array2::<Complex64>::zeros((nbf, np)); nk];
        let mut dchi = vec![Array3::<Complex64>::zeros((3, nbf, nd)); nk];
        let mut m1 = vec![Array4::<Complex64>::zeros((3, 3, nbf, np)); nk];
        let mut m2 = if hess {
            vec![Array4::<Complex64>::zeros((9, 3, nbf, np)); nk]
        } else {
            Vec::new()
        };
        if np == 0 {
            return Ok(BlochAoStrain { chi, dchi, m1, m2 });
        }
        let m = np as f64;
        let cen = [
            pts.iter().map(|p| p[0]).sum::<f64>() / m,
            pts.iter().map(|p| p[1]).sum::<f64>() / m,
            pts.iter().map(|p| p[2]).sum::<f64>() / m,
        ];
        let rad = pts.iter().map(|p| dist3(p, &cen)).fold(0.0, f64::max);
        let mut table: Vec<Vec<Complex64>> = vec![Vec::new(); self.trans.len()];
        let mut live: Vec<(usize, [f64; 3], usize)> = Vec::new();
        for (il, l) in self.trans.iter().enumerate() {
            for (s, sh) in self.shells.iter().enumerate() {
                let c = [
                    sh.center[0] + l[0],
                    sh.center[1] + l[1],
                    sh.center[2] + l[2],
                ];
                if dist3(&c, &cen) <= rad + self.ext[s] {
                    if table[il].is_empty() {
                        table[il] = kpts
                            .iter()
                            .map(|k| {
                                let a = k[0] * l[0] + k[1] * l[1] + k[2] * l[2];
                                match phase {
                                    BlochPhase::True => Complex64::new(a.cos(), a.sin()),
                                    BlochPhase::None => Complex64::new(1.0, 0.0),
                                    BlochPhase::Conj => Complex64::new(a.cos(), -a.sin()),
                                }
                            })
                            .collect();
                    }
                    live.push((s, c, il));
                }
            }
        }
        let mut buf = [0.0_f64; 15];
        let mut gbuf = [[0.0_f64; 15]; 3];
        let mut hbuf = [[0.0_f64; 15]; 9];
        for (g, p) in pts.iter().enumerate() {
            let o = anchors[g];
            for &(s, c, il) in &live {
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
                for (k, f) in table[il].iter().enumerate() {
                    for i in 0..n {
                        let mu = off + i;
                        chi[k][(mu, g)] += *f * buf[i];
                        for a in 0..3 {
                            let da = gbuf[a][i];
                            if hess {
                                dchi[k][(a, mu, g)] += *f * da;
                            }
                            for b in 0..3 {
                                m1[k][(a, b, mu, g)] += *f * (da * rel[b]);
                            }
                        }
                        if hess {
                            for kk in 0..3 {
                                for a in 0..3 {
                                    let h = hbuf[kk * 3 + a][i];
                                    for b in 0..3 {
                                        m2[k][(3 * kk + a, b, mu, g)] += *f * (h * rel[b]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(BlochAoStrain { chi, dchi, m1, m2 })
    }
}

/// Resolve a functional name for the periodic path: `(XcDef, exact-exchange
/// fraction)`. Refuses RSH, meta-GGA, VV10 and scaled composites (double
/// hybrids) by name.
pub fn resolve_periodic_functional(name: &str) -> Result<(XcDef, f64), FerricError> {
    resolve_periodic_functional_mgga(name, false)
}

/// Whether `name` resolves to a functional with a meta-GGA component (raw
/// libxc names included). Unresolvable names are `false`.
pub fn is_meta_gga_name(name: &str) -> bool {
    xc_def_from_name(name).is_ok_and(|d| {
        d.funcs
            .iter()
            .any(|f| matches!(f.family(), FunctionalFamily::MetaGga))
    })
}

/// The functional gate of the closed-shell Gamma [`gamma_rks`]: everything
/// [`resolve_periodic_functional`] accepts, plus ONLY the validated pure
/// meta-GGAs SCAN / r2SCAN / TPSS ([`crate::kxcgrad::is_validated_mgga_energy`]);
/// any other meta-GGA (raw `MGGA_*` names) is refused by name. Shared by
/// [`gamma_rks`] and the Python / CLI bindings so they cannot drift.
pub fn resolve_gamma_rks_functional(name: &str) -> Result<(XcDef, f64), FerricError> {
    let validated = crate::kxcgrad::is_validated_mgga_energy(name);
    if is_meta_gga_name(name) && !validated {
        return Err(PeriodicDftError::Unsupported {
            feature: "meta-GGA",
            reason: format!(
                "{name}: Gamma-point RKS supports only the validated SCAN / r2SCAN / TPSS \
                 meta-GGAs"
            ),
        }
        .into());
    }
    resolve_periodic_functional_mgga(name, validated)
}

/// The whole Gamma RKS request gate, one call for the bindings: the
/// [`resolve_gamma_rks_functional`] allowlist for a meta-GGA, plus
/// [`check_gamma_derivs_functional`] and [`check_gamma_mgga_deriv_jk`] (a
/// meta-GGA force / stress is the 1x1x1 route, dense-AFT J only) when a
/// gradient, stress or optimization is requested (`derivs`). A range-separated hybrid runs as an energy on
/// either J/K and with derivatives on the dense-AFT J/K only (`rsgdf` = the
/// RS-GDF fit; [`crate::gamma_rsh::gamma_rks_rsh_gradient`] /
/// [`crate::gamma_rsh::gamma_rks_rsh_stress`]). Plain (non-meta-GGA) names
/// pass untouched.
pub fn check_gamma_rks_request(name: &str, derivs: bool, rsgdf: bool) -> Result<(), FerricError> {
    if crate::rsh::is_range_separated(name) {
        if derivs && rsgdf {
            return Err(gamma_rsh_rsgdf_refusal(name, "RKS", "force and stress run"));
        }
        if derivs {
            crate::kxcgrad::check_validated_rsh_deriv(name)?;
        }
        return Ok(());
    }
    if !is_meta_gga_name(name) {
        return Ok(());
    }
    resolve_gamma_rks_functional(name)?;
    if derivs {
        check_gamma_derivs_functional(name)?;
        check_gamma_mgga_deriv_jk(name, rsgdf)?;
    }
    Ok(())
}

/// The refusal of a Gamma-point range-separated derivative on RS-GDF (the
/// Gamma routes differentiate the dense-AFT exchange only).
fn gamma_rsh_rsgdf_refusal(name: &str, method: &str, what: &str) -> FerricError {
    // Only the closed-shell k-point force and stress differentiate an RS-GDF
    // range-separated exchange (the UKS force refuses it at every mesh).
    let hint = if method == "RKS" {
        "; the closed-shell k-point one on RS-GDF is wired at a k-point mesh, e.g. 1x1x1"
    } else {
        ""
    };
    PeriodicDftError::Unsupported {
        feature: "range-separated hybrid gradient / stress on RS-GDF",
        reason: format!(
            "{name}: the Gamma-point {method} range-separated hybrid {what} on the \
             dense-AFT J/K only (the Gamma route has no RS-GDF exchange derivative{hint}); \
             use jk = \"dense\""
        ),
    }
    .into()
}

/// The Gamma UKS request gate, one call for the bindings (Python, CLI), run
/// before any integral is built. `derivs` lists the requested derivatives
/// (empty = energy only) and `rsgdf` names the J/K. A range-separated hybrid
/// runs as an energy on either J/K ([`crate::gamma_rsh::gamma_uks_any`]); its
/// FORCE runs on the dense-AFT J/K for the validated functionals
/// ([`crate::gamma_rsh::gamma_uks_rsh_gradient`],
/// [`crate::kxcgrad::check_validated_rsh_deriv`]); its STRESS is refused by
/// name (library only, [`crate::gamma_rsh::gamma_uks_rsh_stress`], not
/// wired). A meta-GGA is refused
/// by name (the Gamma UKS has no tau term), pointing at the k-point UKS on a
/// 1x1x1 mesh, which is the Gamma point and has SCAN / r2SCAN. Everything
/// else passes to [`gamma_uks`]'s own functional check.
pub fn check_gamma_uks_request(
    name: &str,
    derivs: &[crate::kxcgrad::Deriv],
    rsgdf: bool,
) -> Result<(), FerricError> {
    use crate::kxcgrad::Deriv;
    if crate::rsh::is_range_separated(name) && !derivs.is_empty() {
        if derivs.contains(&Deriv::Stress) {
            return Err(PeriodicDftError::Unsupported {
                feature: "open-shell range-separated hybrid stress",
                reason: format!(
                    "{name}: the Gamma-point UKS range-separated hybrid stress is not \
                     implemented in the bindings (the library stress, gamma_uks_rsh_stress, is \
                     not wired yet); the force is available on the dense-AFT J/K"
                ),
            }
            .into());
        }
        if rsgdf {
            return Err(gamma_rsh_rsgdf_refusal(name, "UKS", "force runs"));
        }
        crate::kxcgrad::check_validated_rsh_deriv(name)?;
    }
    if is_meta_gga_name(name) {
        return Err(PeriodicDftError::Unsupported {
            feature: "meta-GGA",
            reason: format!(
                "{name}: the Gamma-point UKS has no tau term; the k-point UKS on a 1x1x1 \
                 mesh (the Gamma point) has SCAN / r2SCAN energies, forces and stress"
            ),
        }
        .into());
    }
    Ok(())
}

/// The Gamma ROKS request gate (Python, CLI): a range-separated hybrid runs
/// as an energy ([`crate::gamma_rsh::gamma_roks_any`]) and is refused with a
/// gradient, stress or optimization (the library ROKS force
/// [`crate::gamma_rsh::gamma_roks_rsh_gradient`] exists on the dense-AFT J/K
/// but is not wired here; there is no ROKS range-separated stress). Everything else passes to [`crate::rohf::gamma_roks`]'s own
/// functional check (no meta-GGA).
pub fn check_gamma_roks_request(name: &str, derivs: bool) -> Result<(), FerricError> {
    if derivs && crate::rsh::is_range_separated(name) {
        return Err(PeriodicDftError::Unsupported {
            feature: "restricted-open-shell range-separated hybrid gradient / stress",
            reason: format!(
                "{name}: the Gamma-point ROKS range-separated hybrid force and stress are not \
                 implemented in the bindings (energy only; the library has the dense-AFT ROKS \
                 force, gamma_roks_rsh_gradient, unwired); the UKS force is available on the \
                 dense-AFT J/K"
            ),
        }
        .into());
    }
    Ok(())
}

/// The meta-GGA gate of a closed-shell Gamma-point nuclear gradient, stress
/// or geometry optimization: a meta-GGA must be one of the validated pure
/// SCAN / r2SCAN / TPSS ([`crate::kxcgrad::is_validated_mgga_deriv`]), whose
/// derivatives are routed through the 1x1x1 k-point force and stress on the
/// dense-AFT J ([`crate::gamma_mgga`]). Checked before any SCF.
pub fn check_gamma_derivs_functional(name: &str) -> Result<(), FerricError> {
    if is_meta_gga_name(name) && !crate::kxcgrad::is_validated_mgga_deriv(name) {
        return Err(PeriodicDftError::Unsupported {
            feature: "meta-GGA gradient / stress",
            reason: format!(
                "{name}: the Gamma-point meta-GGA gradient and stress cover the pure SCAN / \
                 r2SCAN / TPSS only"
            ),
        }
        .into());
    }
    Ok(())
}

/// The J/K gate of a Gamma-point meta-GGA gradient, stress or optimization:
/// the routed derivative ([`crate::gamma_mgga`]) rebuilds the Fock with the
/// dense-AFT J, so an RS-GDF Gamma SCF (`rsgdf = true`) is refused before
/// any SCF instead of by the Fock guard after it. Plain functionals pass.
pub fn check_gamma_mgga_deriv_jk(name: &str, rsgdf: bool) -> Result<(), FerricError> {
    if rsgdf && is_meta_gga_name(name) {
        return Err(PeriodicDftError::Unsupported {
            feature: "meta-GGA gradient / stress on RS-GDF",
            reason: format!(
                "{name}: the Gamma-point meta-GGA gradient and stress run on the dense-AFT J \
                 only (jk = \"dense\"); the RS-GDF route is not wired"
            ),
        }
        .into());
    }
    Ok(())
}

/// [`resolve_periodic_functional`] with the meta-GGA refusal optional
/// (`allow_mgga = true` lets ANY meta-GGA through, so callers apply their own
/// allowlist: the k-point path supplies tau, [`crate::kdft`], and the Gamma
/// entry is [`resolve_gamma_rks_functional`]). Every other refusal is unchanged.
pub fn resolve_periodic_functional_mgga(
    name: &str,
    allow_mgga: bool,
) -> Result<(XcDef, f64), FerricError> {
    let def = xc_def_from_name(name)
        .map_err(|e| FerricError::General(format!("periodic DFT: functional {name:?}: {e:?}")))?;
    // VV10 first: wB97X-V is range-separated AND VV10-carrying, and the
    // blocker that no entry point removes is the missing periodic NLC grid.
    if def.vv10.is_some() {
        return Err(PeriodicDftError::Unsupported {
            feature: "VV10 nonlocal correlation",
            reason: format!("{name}: no periodic NLC grid"),
        }
        .into());
    }
    if def.cam.is_some() {
        return Err(PeriodicDftError::Unsupported {
            feature: "range-separated hybrid",
            reason: format!(
                "{name} needs the attenuated (erf/erfc) periodic exchange, which this entry \
                 point does not build: it is built by closed-shell RKS (k-point, or the \
                 Gamma-point gamma_rks_rsh / run_rks_gamma) and by open-shell UKS (solve_kuks, \
                 gamma_uks_rsh / run_uks_gamma) and ROKS (solve_kroks, gamma_roks_rsh / \
                 run_roks_gamma). Use a global hybrid (e.g. \
                 PBE0) or a semilocal functional here"
            ),
        }
        .into());
    }
    if !allow_mgga
        && def
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
    /// Accumulated [`PeriodicXc::eval`] / [`PeriodicXc::eval_polarized`]
    /// calls ([`crate::timing`]).
    clock: CallClock,
    /// The functional is a meta-GGA: [`PeriodicXc::eval`] builds tau from the
    /// cached `∇χ^Γ` (closed shell only; the spin-resolved path refuses).
    mgga: bool,
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
        Self::build_xc(cell, bs, functional, grid, cfg, false, false)
    }

    /// [`PeriodicXc::new`] with the meta-GGA refusal lifted for the CLOSED-shell
    /// [`PeriodicXc::eval`] (tau from `D` and `∇χ^Γ`, as the k-point path does).
    /// [`PeriodicXc::eval_polarized`] errors on a meta-GGA built this way.
    pub fn new_closed_mgga(
        cell: &Cell,
        bs: &BasisSet,
        functional: &str,
        grid: &PeriodicGrid,
        cfg: &PeriodicXcConfig,
    ) -> Result<Self, FerricError> {
        Self::build_xc(cell, bs, functional, grid, cfg, true, false)
    }

    pub(crate) fn build_xc(
        cell: &Cell,
        bs: &BasisSet,
        functional: &str,
        grid: &PeriodicGrid,
        cfg: &PeriodicXcConfig,
        allow_mgga: bool,
        allow_rsh: bool,
    ) -> Result<Self, FerricError> {
        // A range-separated hybrid (closed-shell Gamma RKS only) keeps its
        // exact-exchange fractions INSIDE the K kernel, so the SCF scales K
        // by 1, as the k-point path does (`kdft::solve_krks_on_grid`).
        let (xc, exx) = if allow_rsh {
            let (xc, a, rsh) = crate::rsh::resolve_periodic_functional_rsh(functional)?;
            (xc, if rsh.is_some() { 1.0 } else { a })
        } else {
            resolve_periodic_functional_mgga(functional, allow_mgga)?
        };
        let mgga = xc
            .funcs
            .iter()
            .any(|f| matches!(f.family(), FunctionalFamily::MetaGga));
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
            clock: CallClock::default(),
            mgga,
        })
    }

    /// Accumulated time of every [`PeriodicXc::eval`] /
    /// [`PeriodicXc::eval_polarized`] call so far (stage `"scf XC eval"`).
    pub fn eval_timing(&self) -> StageTiming {
        self.clock.timing("scf XC eval")
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

    /// Shape guard: errors unless the density is `(nbf, nbf)`.
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
        let clock = StageClock::start();
        let out = self.eval_untimed(d);
        self.clock.record(&clock);
        out
    }

    /// Closed-shell `(E_xc, V_xc)` for the total density `d`, summed over the grid
    /// chunks, with no clock. `V_xc` is `(nbf, nbf)`. Errors if `d` has the wrong
    /// shape or the energy or any `V_xc` element is non-finite.
    fn eval_untimed(&mut self, d: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        self.check_d(d)?;
        let mut e = 0.0;
        let mut v = Array2::<f64>::zeros((self.nbf, self.nbf));
        for c in &self.chunks {
            let dens = eval_density_closed(d, &c.chi, &c.dchi);
            let tau = self.mgga.then(|| eval_tau_closed(d, &c.dchi));
            let (ec, vc) = semilocal_vxc_closed_scratch(
                &c.points,
                &c.chi,
                &c.dchi,
                &dens,
                tau.as_ref(),
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
        let clock = StageClock::start();
        let out = self.eval_polarized_untimed(d_a, d_b);
        self.clock.record(&clock);
        out
    }

    /// The Gamma-point spin-resolved path has no tau term: error on a meta-GGA
    /// (the k-point UKS has one, see `crate::kuks`).
    fn refuse_if_mgga(&self) -> Result<(), FerricError> {
        if self.mgga {
            return Err(PeriodicDftError::Unsupported {
                feature: "meta-GGA",
                reason: format!(
                    "{}: the Gamma-point spin-resolved periodic XC has no tau term",
                    self.name
                ),
            }
            .into());
        }
        Ok(())
    }

    /// Spin-polarized `(E_xc, V_α, V_β)` for `(d_a, d_b)`, summed over the grid
    /// chunks, with no clock. Errors on a wrong-shaped density or a non-finite
    /// energy / potential element.
    fn eval_polarized_untimed(
        &mut self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
        self.check_d(d_a)?;
        self.check_d(d_b)?;
        self.refuse_if_mgga()?;
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
    /// identifiers), e.g. `"LDA"` (= PySCF `"LDA,VWN"`), `"PBE"`, `"PBE0"`, or
    /// the pure meta-GGAs `"SCAN"` / `"r2SCAN"` / `"TPSS"` (their force and stress
    /// are [`crate::gamma_mgga`], dense-AFT J only; other meta-GGAs are refused
    /// by name).
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
    /// Driver stages: `"xc grid build"`, `"xc AO cache"`, `"scf XC eval"`
    /// (every XC call, including the one post-SCF `E_xc` evaluation) and the
    /// grid-point counter; `wall_s` is the whole call. The SCF's J/K calls
    /// are on the integral source (`GammaUhfIntegrals::timings`), not here.
    pub timings: PbcTimings,
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
    gamma_rks_impl(cell, prep, hc, ints, cfg, None, None)
}

/// [`gamma_rks`] for a range-separated hybrid (HSE06, CAM-B3LYP, wB97X):
/// J from `ints`, the attenuated K from `rsh_k` (the Gamma = 1x1x1-mesh
/// exchange of [`crate::gamma_rsh`], whose exxdiv must equal `cfg.exxdiv`).
/// `cfg.functional` must be range-separated; the global-hybrid and semilocal
/// functionals go through [`gamma_rks`]. The returned `madelung` is the MIXED
/// constant the K applied and `exact_exchange_fraction` is `c_sr`.
pub fn gamma_rks_rsh(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    rsh_k: &crate::gamma_rsh::GammaRshExchange,
    cfg: &GammaRksConfig,
) -> Result<GammaRksResult, FerricError> {
    gamma_rks_impl(cell, prep, hc, ints, cfg, Some(rsh_k), None)
}

/// [`gamma_rks_rsh`] on a caller-supplied XC grid (e.g. a
/// [`PeriodicGrid::uniform`] one, to match a PySCF `UniformGrids` reference);
/// `cfg.grid` is ignored.
pub fn gamma_rks_rsh_on_grid(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    rsh_k: &crate::gamma_rsh::GammaRshExchange,
    grid: &PeriodicGrid,
    cfg: &GammaRksConfig,
) -> Result<GammaRksResult, FerricError> {
    gamma_rks_impl(cell, prep, hc, ints, cfg, Some(rsh_k), Some(grid))
}

fn gamma_rks_impl(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    cfg: &GammaRksConfig,
    rsh_k: Option<&crate::gamma_rsh::GammaRshExchange>,
    grid_override: Option<&PeriodicGrid>,
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
    if let Some(ex) = rsh_k {
        ex.check_for_rks(&cfg.functional, cfg.exxdiv)?;
    } else {
        resolve_gamma_rks_functional(&cfg.functional)?;
    }
    let total = StageClock::start();
    let (grid, mut pxc, mut timings) = gated_grid_and_xc(
        cell,
        prep,
        &cfg.functional,
        GridSource::new(&cfg.grid, grid_override),
        &cfg.xc,
        true,
        rsh_k.is_some(),
    )?;
    let v_m = match rsh_k {
        Some(ex) => ex.mixed_madelung(),
        None => madelung_constant(cell)?,
    };
    let applied = match cfg.exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => v_m,
    };
    let ctx = ParallelContext::default();
    // Required by the solver signature; never read for integrals here.
    let bounds = SchwarzBounds::compute(Operator::coulomb(), prep)?;
    let (j, mut k) = crate::uhf::builders(ints, applied);
    // The range-separated K replaces the Coulomb K wholesale (fractions and
    // the mixed Madelung term are inside it; the SCF scales it by 1).
    if let Some(ex) = rsh_k {
        k = Box::new(ex.k_builder());
    }
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
    timings.add_stage(&pxc.eval_timing());
    timings.finish(&total);
    Ok(GammaRksResult {
        madelung: v_m,
        exact_exchange_fraction: rsh_k.map_or(pxc.exx, |ex| ex.params().c_sr),
        n_grid_points: grid.len(),
        neighbour_cutoff: grid.neighbour_cutoff(),
        electrons_on_grid,
        e_xc,
        scf,
        timings,
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
    /// Driver stages as [`GammaRksResult::timings`] ([`gamma_uks`] only;
    /// empty from [`gamma_uks_with_xc`], whose XC builder is the caller's).
    pub timings: PbcTimings,
}

/// The periodic grid and the [`PeriodicXc`] AO cache of a KS driver, with
/// their build stages (`"xc grid build"`, `"xc AO cache"`) and the grid-point
/// counter.
pub(crate) fn ks_grid_and_xc(
    cell: &Cell,
    prep: &PreparedBasis,
    functional: &str,
    grid_cfg: &PeriodicGridConfig,
    xc_cfg: &PeriodicXcConfig,
) -> Result<(PeriodicGrid, PeriodicXc, PbcTimings), FerricError> {
    gated_grid_and_xc(
        cell,
        prep,
        functional,
        GridSource::config(grid_cfg),
        xc_cfg,
        false,
        false,
    )
}

/// [`ks_grid_and_xc`] with the meta-GGA refusal optional (closed-shell
/// [`gamma_rks`] only, after its own allowlist).
/// Where the XC grid comes from: built from a config, or supplied.
#[derive(Clone, Copy)]
pub(crate) struct GridSource<'a> {
    cfg: &'a PeriodicGridConfig,
    given: Option<&'a PeriodicGrid>,
}

impl<'a> GridSource<'a> {
    pub(crate) fn new(cfg: &'a PeriodicGridConfig, given: Option<&'a PeriodicGrid>) -> Self {
        Self { cfg, given }
    }

    pub(crate) fn config(cfg: &'a PeriodicGridConfig) -> Self {
        Self { cfg, given: None }
    }

    fn resolve(self, cell: &Cell) -> Result<PeriodicGrid, FerricError> {
        match self.given {
            Some(g) => Ok(g.clone()),
            None => PeriodicGrid::build(cell, self.cfg),
        }
    }
}

pub(crate) fn gated_grid_and_xc(
    cell: &Cell,
    prep: &PreparedBasis,
    functional: &str,
    grid_src: GridSource<'_>,
    xc_cfg: &PeriodicXcConfig,
    allow_mgga: bool,
    allow_rsh: bool,
) -> Result<(PeriodicGrid, PeriodicXc, PbcTimings), FerricError> {
    let mut timings = PbcTimings::default();
    let clock = StageClock::start();
    let grid = grid_src.resolve(cell)?;
    timings.stop("xc grid build", &clock);
    let clock = StageClock::start();
    let pxc = PeriodicXc::build_xc(
        cell,
        prep.basis_set(),
        functional,
        &grid,
        xc_cfg,
        allow_mgga,
        allow_rsh,
    )?;
    timings.stop("xc AO cache", &clock);
    timings.set_counter("xc grid points", grid.len() as u64);
    Ok((grid, pxc, timings))
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
    let total = StageClock::start();
    let (grid, mut pxc, mut timings) =
        ks_grid_and_xc(cell, prep, &cfg.functional, &cfg.grid, &cfg.xc)?;
    let mut out = gamma_uks_with_xc(cell, prep, hc, ints, &mut pxc, cfg)?;
    let electrons_on_grid = pxc.integrate_density(&out.scf.density_total)?;
    out.grid = Some(GammaUksGridInfo {
        n_grid_points: grid.len(),
        neighbour_cutoff: grid.neighbour_cutoff(),
        electrons_on_grid,
    });
    timings.add_stage(&pxc.eval_timing());
    timings.finish(&total);
    out.timings = timings;
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
    gamma_uks_core(cell, prep, hc, ints, xc, cfg, None)
}

/// [`gamma_uks`] for a range-separated hybrid (HSE06, CAM-B3LYP validated against PySCF; wB97X runs the same code but has no open-shell pin): J
/// from `ints`, the attenuated K of each spin from `rsh_k` (the Gamma =
/// 1x1x1-mesh exchange of [`crate::gamma_rsh`], whose exxdiv must equal
/// `cfg.exxdiv`). `K_σ = K[D_σ]` with the fractions and the MIXED Madelung
/// constant inside, so the SCF scales it by 1. `cfg.functional` must be
/// range-separated. The dense-AFT nuclear force is
/// [`crate::gamma_rsh::gamma_uks_rsh_gradient`] (the bindings reach it through
/// [`crate::gamma_rsh::gamma_uks_any_keep`]); the stress refuses RSH.
pub fn gamma_uks_rsh(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    rsh_k: &crate::gamma_rsh::GammaRshExchange,
    cfg: &GammaUksConfig,
) -> Result<GammaUksResult, FerricError> {
    gamma_uks_rsh_impl(cell, prep, hc, ints, rsh_k, None, cfg)
}

/// [`gamma_uks_rsh`] on a caller-supplied XC grid (`cfg.grid` is ignored).
pub fn gamma_uks_rsh_on_grid(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    rsh_k: &crate::gamma_rsh::GammaRshExchange,
    grid: &PeriodicGrid,
    cfg: &GammaUksConfig,
) -> Result<GammaUksResult, FerricError> {
    gamma_uks_rsh_impl(cell, prep, hc, ints, rsh_k, Some(grid), cfg)
}

fn gamma_uks_rsh_impl(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    rsh_k: &crate::gamma_rsh::GammaRshExchange,
    grid_override: Option<&PeriodicGrid>,
    cfg: &GammaUksConfig,
) -> Result<GammaUksResult, FerricError> {
    ferric_scf::uhf::validate_injected_uhf(&cfg.scf).map_err(FerricError::from)?;
    refuse_molecular_grid_knobs(&cfg.scf)?;
    nocc_ab(cell.mol())?;
    // Cheap name checks before the grid is built.
    rsh_k.check_for("gamma_uks_rsh", &cfg.functional, cfg.exxdiv)?;
    let total = StageClock::start();
    let (grid, mut pxc, mut timings) = gated_grid_and_xc(
        cell,
        prep,
        &cfg.functional,
        GridSource::new(&cfg.grid, grid_override),
        &cfg.xc,
        false,
        true,
    )?;
    let mut out = gamma_uks_core(cell, prep, hc, ints, &mut pxc, cfg, Some(rsh_k))?;
    let electrons_on_grid = pxc.integrate_density(&out.scf.density_total)?;
    out.grid = Some(GammaUksGridInfo {
        n_grid_points: grid.len(),
        neighbour_cutoff: grid.neighbour_cutoff(),
        electrons_on_grid,
    });
    timings.add_stage(&pxc.eval_timing());
    timings.finish(&total);
    out.timings = timings;
    Ok(out)
}

fn gamma_uks_core(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    xc: &mut dyn XcBuilder,
    cfg: &GammaUksConfig,
    rsh_k: Option<&crate::gamma_rsh::GammaRshExchange>,
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
    // A range-separated K carries the MIXED constant c_sr v_M + (c_lr - c_sr)
    // v_M(omega) (gamma_rsh); everything else the full-Coulomb one.
    let v_m = match rsh_k {
        Some(ex) => ex.mixed_madelung(),
        None => madelung_constant(cell)?,
    };
    let applied = match cfg.exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => v_m,
    };
    let ctx = ParallelContext::default();
    // Required by the solver signature; never read for integrals here.
    let bounds = SchwarzBounds::compute(Operator::coulomb(), prep)?;
    let mut run = |vm: f64, init: Option<(&Array2<f64>, &Array2<f64>)>| {
        let (j, mut k) = crate::uhf::builders(ints, vm);
        if let Some(ex) = rsh_k {
            // Called once per spin on D_sigma; fractions are inside it.
            k = Box::new(ex.k_builder_with_madelung(vm));
        }
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
        exact_exchange_fraction: rsh_k.map_or(a, |ex| ex.params().c_sr),
        nocc: (na, nb),
        s2,
        gaps,
        e_xc,
        grid: None,
        none_stage,
        scf,
        timings: PbcTimings::default(),
    })
}

/// Grid-level anchors of the neighbour switch (#495), SCF-free: the grid
/// functional `E = Σ_g W_g F(r_g)` of a fixed lab-frame `F`, whose geometry
/// dependence is exactly the grid's (weights and point motion). The SCF-level
/// FD and scans are `tests/validation_pbc_grid_switch.rs`.
///
/// Mutation ledger (measured; each applied, these tests run, reverted):
/// * kernel drops the neighbour's presence derivative (`ferric_dft` M1):
///   dense-cell force FD miss 9.8e-3 (SSF) / 1.2e-2 (Becke), stress 0.23 / 0.36;
/// * kernel flips its sign (M3): force 2.0e-2 / 2.3e-2, stress 0.46 / 0.72;
/// * kernel drops a tapered atom's OWN presence derivative (M2) or its own
///   `f_B` factor (M5): invisible in the dense cell (2.3e-10, the floor: a
///   tapered atom's `P_B` is 0 or negligible there); the sparse-cell test
///   misses by 1.03 (M2) and 0.30 (M5);
/// * switch never selected (`PartitionKernel::new` returns the hard
///   kernel): force 2.4e-4, stress 2.6e-3, and the scan shows the hard jumps.
#[cfg(test)]
mod switch_tests {
    use super::*;
    use ferric_core::mol::{Atom, Molecule};

    const A: f64 = 4.0;
    const CUT: f64 = 6.5;
    const WIDTH: f64 = 1.5;

    fn atom(sym: &str, z: i32, r: [f64; 3]) -> Atom {
        Atom {
            symbol: sym.into(),
            z,
            x: r[0],
            y: r[1],
            zpos: r[2],
            ghost: false,
            n_core_ecp: 0,
        }
    }

    /// Li–H in a cubic cell: the clipped size adjustment (`|a| = ½`) puts far
    /// images inside the SSF support, so the neighbour mask is live.
    fn lih(h_shift: [f64; 3]) -> Cell {
        let mol = Molecule {
            atoms: vec![
                atom("Li", 3, [0.1, -0.05, 0.02]),
                atom(
                    "H",
                    1,
                    [2.07 + h_shift[0], 1.96 + h_shift[1], 2.04 + h_shift[2]],
                ),
            ],
            charge: 0,
            multiplicity: 1,
        };
        Cell::new(mol, [[A, 0.0, 0.0], [0.0, A, 0.0], [0.0, 0.0, A]]).unwrap()
    }

    fn cfg(width: f64) -> PeriodicGridConfig {
        cfg_with(PartitionScheme::Ssf, width)
    }

    fn cfg_with(scheme: PartitionScheme, width: f64) -> PeriodicGridConfig {
        PeriodicGridConfig {
            neighbour_cutoff: Some(CUT),
            neighbour_switch: width,
            partition: scheme,
            ..PeriodicGridConfig::with_size(12, 26)
        }
    }

    /// `F(r) = exp(−0.05 |r − c|²)` and its gradient.
    fn f_and_grad(r: [f64; 3]) -> (f64, [f64; 3]) {
        let c = [1.3, 0.7, 2.2];
        let d = [r[0] - c[0], r[1] - c[1], r[2] - c[2]];
        let v = (-0.05 * dot3(&d, &d)).exp();
        (v, [-0.1 * d[0] * v, -0.1 * d[1] * v, -0.1 * d[2] * v])
    }

    fn functional(cell: &Cell, width: f64) -> f64 {
        functional_with(cell, PartitionScheme::Ssf, width)
    }

    fn functional_with(cell: &Cell, scheme: PartitionScheme, width: f64) -> f64 {
        let g = PeriodicGrid::build(cell, &cfg_with(scheme, width)).unwrap();
        g.points()
            .iter()
            .map(|p| p.weight * f_and_grad(p.xyz).0)
            .sum()
    }

    /// Fourth-order central difference `(−E(2h) + 8E(h) − 8E(−h) + E(−2h)) / 12h`:
    /// at h = 2e-4 its truncation is far below the ~1e-13 roundoff of `Σ W F`.
    fn fd4(h: f64, e: impl Fn(f64) -> f64) -> f64 {
        (-e(2.0 * h) + 8.0 * e(h) - 8.0 * e(-h) + e(-2.0 * h)) / (12.0 * h)
    }

    fn ledger() -> Ledger {
        Ledger::new(1 << 30)
    }

    /// EXACTNESS ANCHOR: width 0 is the hard cutoff, bit for bit (same points,
    /// same weights, same response derivatives as the default config).
    #[test]
    fn zero_width_is_the_hard_cutoff_bitwise() {
        let cell = lih([0.0; 3]);
        let hard = PeriodicGridConfig {
            neighbour_cutoff: Some(CUT),
            ..PeriodicGridConfig::with_size(12, 26)
        };
        assert_eq!(hard.neighbour_switch, 0.0);
        let a = PeriodicGrid::build(&cell, &hard).unwrap();
        let b = PeriodicGrid::build(&cell, &cfg(0.0)).unwrap();
        assert_eq!(a.len(), b.len());
        for (p, q) in a.points().iter().zip(b.points()) {
            assert_eq!(p.weight.to_bits(), q.weight.to_bits());
            assert_eq!(p.xyz, q.xyz);
        }
        // The response grid's weights are the energy grid's, bit for bit.
        let r = build_response_grid(&cell, &cfg(0.0), &mut ledger()).unwrap();
        for (p, q) in a.points().iter().zip(&r.points) {
            assert_eq!(p.weight.to_bits(), q.weight.to_bits());
        }
    }

    /// REGRESSION GUARD for the default (W = 0) grid: an FNV-style hash of every
    /// point's weight and coordinates (bits), triclinic Li–H at (20, 50),
    /// pinned to the values measured on main before the switch existed
    /// (auto D, D = 6.5, D = 12). A change to the hard path's arithmetic —
    /// not only to the switch — fails here. Libm-sensitive inputs (radial
    /// grid logs) are part of the hash: a different platform libm can move
    /// it, which is a re-pin, not a defect, once the in-process anchor above
    /// still holds. Seen failing: `DEFAULT_NEIGHBOUR_CUTOFF` 10 → 10.5 moves
    /// the auto-D hash.
    #[test]
    fn default_grid_weights_match_the_pre_switch_hashes() {
        let mol = Molecule {
            atoms: vec![
                atom("Li", 3, [0.1, -0.05, 0.02]),
                atom("H", 1, [2.07, 1.96, 2.04]),
            ],
            charge: 0,
            multiplicity: 1,
        };
        let cell = Cell::new(mol, [[4.0, 0.0, 0.0], [0.3, 4.2, 0.0], [0.0, 0.1, 3.9]]).unwrap();
        for (d, want) in [
            (None, 0x4ce1_c320_01a9_51e7_u64),
            (Some(6.5), 0xc291_1dff_6c16_959f),
            (Some(12.0), 0xc944_8985_25c5_ac3f),
        ] {
            let cfg = PeriodicGridConfig {
                neighbour_cutoff: d,
                ..PeriodicGridConfig::with_size(20, 50)
            };
            let g = PeriodicGrid::build(&cell, &cfg).unwrap();
            // The probe's offset basis (not the standard FNV one; kept so the
            // pinned values stay comparable with the main-branch measurement).
            let mut h: u64 = 1_469_598_103_934_665_603;
            for p in g.points() {
                for v in [
                    p.weight.to_bits(),
                    p.xyz[0].to_bits(),
                    p.xyz[1].to_bits(),
                    p.xyz[2].to_bits(),
                ] {
                    h ^= v;
                    h = h.wrapping_mul(0x0100_0000_01b3);
                }
            }
            assert_eq!(h, want, "D = {d:?}: {h:016x}");
        }
    }

    /// With the switch on, the response and strain grids carry exactly the
    /// energy grid's weights, and Σ w misses the cell volume by about the
    /// hard grid's partition error (this deliberately coarse (12, 26) grid:
    /// measured 11.3 % switched vs 12.0 % hard).
    #[test]
    fn switched_grids_share_weights_and_integrate_the_volume() {
        let cell = lih([0.0; 3]);
        let e = PeriodicGrid::build(&cell, &cfg(WIDTH)).unwrap();
        let r = build_response_grid(&cell, &cfg(WIDTH), &mut ledger()).unwrap();
        let s = build_strain_grid(&cell, &cfg(WIDTH), &mut ledger()).unwrap();
        assert_eq!(e.len(), r.points.len());
        assert_eq!(e.len(), s.points.len());
        for ((p, q), t) in e.points().iter().zip(&r.points).zip(&s.points) {
            assert_eq!(p.weight.to_bits(), q.weight.to_bits());
            assert_eq!(p.weight.to_bits(), t.weight.to_bits());
        }
        let hard = PeriodicGrid::build(&cell, &cfg(0.0)).unwrap();
        let v = cell.volume();
        let (es, eh) = (
            (e.weight_sum() - v).abs() / v,
            (hard.weight_sum() - v).abs() / v,
        );
        eprintln!("relative volume error: switched {es:.2e}, hard {eh:.2e}");
        assert!(es < 1.5 * eh && eh < 1.5 * es, "{es:e} vs {eh:e}");
    }

    /// The response grid's weight derivative (images folded onto cell atoms
    /// plus the home point's motion) reproduces the FD of `Σ W F` with the
    /// taper live, for SSF and Becke. SSF alone cannot see the derivative of
    /// a non-home atom's OWN presence `f_B` (a tapered atom's `P_B` is
    /// exactly 0 under SSF here); Becke's nonzero tails can. Measured worst
    /// 2.3e-10 (fourth-order FD, h = 2e-4, roundoff-limited); mutants in the
    /// module ledger.
    #[test]
    fn switched_response_grid_matches_fd_of_the_grid_functional() {
        let cell = lih([0.0; 3]);
        let mut worst = 0.0_f64;
        for scheme in [PartitionScheme::Ssf, PartitionScheme::Becke] {
            let rg = build_response_grid(&cell, &cfg_with(scheme, WIDTH), &mut ledger()).unwrap();
            let n = rg.natoms;
            let mut an = [[0.0_f64; 3]; 2];
            for (g, p) in rg.points.iter().enumerate() {
                let (f, df) = f_and_grad(p.xyz);
                for (a, row) in an.iter_mut().enumerate() {
                    for x in 0..3 {
                        row[x] += rg.dweight[g * n + a][x] * f;
                        if a == p.home_atom {
                            row[x] += p.weight * df[x];
                        }
                    }
                }
            }
            let h = 2e-4;
            let mut miss = 0.0_f64;
            for (a, row) in an.iter().enumerate() {
                for x in 0..3 {
                    let shift = |s: f64| {
                        let mut pos = cell.positions();
                        pos[a][x] += s;
                        let mut mol = cell.mol().clone();
                        for (at, p) in mol.atoms.iter_mut().zip(&pos) {
                            (at.x, at.y, at.zpos) = (p[0], p[1], p[2]);
                        }
                        Cell::new(mol, *cell.lattice()).unwrap()
                    };
                    let fd = fd4(h, |s| functional_with(&shift(s), scheme, WIDTH));
                    miss = miss.max((fd - row[x]).abs());
                }
            }
            eprintln!("{scheme:?}: switched grid force vs FD: {miss:.2e}");
            worst = worst.max(miss);
        }
        assert!(worst < 1e-8, "{worst:e}");
    }

    /// A SPARSE cell (Li–H, a = 9 Bohr) with the tightest legal cutoff
    /// `D = bound + W`: near the covering-radius holes the nearest image is
    /// barely fully present and the next ones sit in the taper with
    /// non-negligible cell functions, so the derivative of a tapered atom's
    /// OWN presence `f_B` (invisible in the dense cell above) reaches the
    /// force. Becke partition (its tails keep those `P_B` nonzero).
    #[test]
    fn sparse_cell_force_sees_the_own_presence_derivative() {
        let mol = Molecule {
            atoms: vec![
                atom("Li", 3, [0.1, -0.05, 0.02]),
                atom("H", 1, [2.07, 1.96, 2.04]),
            ],
            charge: 0,
            multiplicity: 1,
        };
        let lat = [[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]];
        let cell = Cell::new(mol, lat).unwrap();
        let w = 1.5;
        let c = PeriodicGridConfig {
            // + 0.05: the FD displacements move the covering bound slightly.
            neighbour_cutoff: Some(covering_radius_bound(&cell) + w + 0.05),
            neighbour_switch: w,
            partition: PartitionScheme::Becke,
            ..PeriodicGridConfig::with_size(12, 26)
        };
        let e = |cl: &Cell| -> f64 {
            PeriodicGrid::build(cl, &c)
                .unwrap()
                .points()
                .iter()
                .map(|p| p.weight * f_and_grad(p.xyz).0)
                .sum()
        };
        let rg = build_response_grid(&cell, &c, &mut ledger()).unwrap();
        let n = rg.natoms;
        let mut an = [[0.0_f64; 3]; 2];
        for (g, p) in rg.points.iter().enumerate() {
            let (f, df) = f_and_grad(p.xyz);
            for (a, row) in an.iter_mut().enumerate() {
                for x in 0..3 {
                    row[x] += rg.dweight[g * n + a][x] * f;
                    if a == p.home_atom {
                        row[x] += p.weight * df[x];
                    }
                }
            }
        }
        let mut miss = 0.0_f64;
        for (a, row) in an.iter().enumerate() {
            for x in 0..3 {
                let shift = |s: f64| {
                    let mut mol = cell.mol().clone();
                    match x {
                        0 => mol.atoms[a].x += s,
                        1 => mol.atoms[a].y += s,
                        _ => mol.atoms[a].zpos += s,
                    }
                    Cell::new(mol, lat).unwrap()
                };
                miss = miss.max((fd4(2e-4, |s| e(&shift(s))) - row[x]).abs());
            }
        }
        eprintln!("sparse Becke: switched grid force vs FD: {miss:.2e}");
        assert!(miss < 1e-8, "{miss:e}");
    }

    /// The strain grid's weight derivative plus the anchored point motion
    /// `dr/dε_ab = e_a O_b` reproduces the FD of `Σ W F` under
    /// `Cell::strained`, all nine components, taper live, SSF and Becke.
    /// Measured worst 2.3e-10; mutants in the module ledger.
    #[test]
    fn switched_strain_grid_matches_fd_of_the_grid_functional() {
        let cell = lih([0.0; 3]);
        let mut worst = 0.0_f64;
        for scheme in [PartitionScheme::Ssf, PartitionScheme::Becke] {
            let sg = build_strain_grid(&cell, &cfg_with(scheme, WIDTH), &mut ledger()).unwrap();
            let mut an = [[0.0_f64; 3]; 3];
            for (g, p) in sg.points.iter().enumerate() {
                let (f, df) = f_and_grad(p.xyz);
                let o = sg.anchors[g];
                for a in 0..3 {
                    for b in 0..3 {
                        an[a][b] += sg.dweight[g][a][b] * f + p.weight * df[a] * o[b];
                    }
                }
            }
            let h = 2e-4;
            let mut miss = 0.0_f64;
            for a in 0..3 {
                for b in 0..3 {
                    let at = |s: f64| {
                        let mut e = [[0.0; 3]; 3];
                        e[a][b] = s;
                        functional_with(&cell.strained(&e).unwrap(), scheme, WIDTH)
                    };
                    miss = miss.max((fd4(h, at) - an[a][b]).abs());
                }
            }
            eprintln!("{scheme:?}: switched grid stress vs FD: {miss:.2e}");
            worst = worst.max(miss);
        }
        assert!(worst < 1e-8, "{worst:e}");
    }

    /// The defect itself, SCF-free: a dense scan of the H position over a
    /// fixed 0.2 Bohr path at two step sizes. A smooth `E` has third
    /// differences `E‴ δ³` (÷8 when δ halves); a one-step jump `J` leaves
    /// `|Δ³E| ≈ J` at any δ. Measured (δ = 1e-3 → 5e-4): hard 1.1e-3 →
    /// 4.1e-4 (ratio 0.37: jumps), switched 2.2e-8 → 2.7e-9 (ratio 0.125).
    #[test]
    fn displacement_scan_has_no_jumps_with_the_switch() {
        let worst_d3 = |width: f64, delta: f64| {
            let n = (0.2 / delta).round() as usize;
            let e: Vec<f64> = (0..=n)
                .map(|i| {
                    let s = i as f64 * delta;
                    functional(&lih([s, 0.6 * s, -0.3 * s]), width)
                })
                .collect();
            e.windows(4)
                .map(|w| (w[3] - 3.0 * w[2] + 3.0 * w[1] - w[0]).abs())
                .fold(0.0_f64, f64::max)
        };
        let (h1, h2) = (worst_d3(0.0, 1e-3), worst_d3(0.0, 5e-4));
        let (s1, s2) = (worst_d3(WIDTH, 1e-3), worst_d3(WIDTH, 5e-4));
        eprintln!(
            "max |third difference| (delta 1e-3 -> 5e-4): hard {h1:.2e} -> {h2:.2e}, switched {s1:.2e} -> {s2:.2e}"
        );
        // Hard: a jump, so halving the step leaves it (ratio ~1, not 1/8).
        assert!(h1 > 1e-6 && h2 > 0.25 * h1, "{h1:e} {h2:e}");
        // Switched: the smooth δ³ law (1/8; bar 1/4) and no jump-sized value.
        assert!(s2 < 0.25 * s1, "{s1:e} {s2:e}");
        assert!(s1 < 1e-3 * h1, "{s1:e} vs {h1:e}");
    }

    /// Refusals: the fully-present zone must cover space, and the width
    /// must be finite and non-negative.
    #[test]
    fn switch_refusals_and_automatic_cutoff() {
        let cell = lih([0.0; 3]);
        let bound = covering_radius_bound(&cell);
        let too_small = PeriodicGridConfig {
            neighbour_cutoff: Some(bound + 0.5),
            neighbour_switch: 1.0,
            ..PeriodicGridConfig::with_size(12, 26)
        };
        match PeriodicGrid::build(&cell, &too_small) {
            Err(e) => assert!(e.to_string().contains("covering-radius"), "{e}"),
            Ok(_) => panic!("D - W below the covering bound must be refused"),
        }
        let (d, _) = resolve_neighbour_cutoff_switched(&cell, None, 9.0).unwrap();
        assert_eq!(d, DEFAULT_NEIGHBOUR_CUTOFF.max(bound) + 9.0);
        for w in [-1.0, f64::NAN, f64::INFINITY] {
            let bad = PeriodicGridConfig {
                neighbour_switch: w,
                ..PeriodicGridConfig::with_size(12, 26)
            };
            let e = PeriodicGrid::build(&cell, &bad).unwrap_err().to_string();
            assert!(e.contains("neighbour_switch"), "{e}");
        }
    }
}
