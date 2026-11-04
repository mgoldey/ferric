//! Atom-centered molecular grid assembled from Treutler-Ahlrichs M4
//! radial + Lebedev angular + Becke fuzzy weights.
//!
//! Each grid point carries:
//!   * `xyz`: Cartesian position (Bohr)
//!   * `weight`: full quadrature weight including 4π r² Jacobian and
//!     Becke partition for the home atom
//!   * `home_atom`: index of the atom the radial shell is centered on
//!
//! Spherical integral of a function `f(r)` over all space:
//! ```text
//!     ∫ f dV ≈ Σ_g w_g · f(r_g)
//! ```
//!
//! Per-atom A partition integrals:
//! ```text
//!     ∫_A f dV = Σ_{g: home=A} w_g · f(r_g)
//! ```
//! (Becke partition is already baked into `w_g`; the home-atom restriction
//! is what distinguishes per-atom from total.)

use ferric_core::error::FerricError;
use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::mol::Molecule;
use rayon::prelude::*;

use crate::becke::{becke_weights_all, becke_weights_and_grad, BeckeScratch, BeckeTable};
use crate::lebedev::lebedev;
use crate::prune::{angular_orders_for_atom, PruneScheme};
use crate::radial::treutler_ahlrichs_m4;

/// Below this many points, run the Becke-weight pass serially. Rayon
/// spawn/join/steal overhead dominates on small (e.g. free-atom SAD) grids —
/// same rationale and threshold order-of-magnitude as `ao_grid.rs`'s
/// `PAR_WORK_THRESHOLD` (per-point Becke weight is O(natoms) to O(natoms²)
/// work, comparable to a handful of AO evaluations per point).
const PAR_WORK_THRESHOLD: usize = 2_000;

/// Everything [`build_atomic_grid_with_response`] holds resident, declared as
/// a plan so the check names the term that blew up rather than just a total.
///
/// Exposed to `gradient.rs` (via [`grid_response_bytes`]) so a gradient's own
/// plan can account for the grid it is handed instead of pretending it starts
/// from an empty budget — the composition failure the `MemoryPlan` type was
/// introduced to fix.
pub(crate) fn grid_response_plan(npts: usize, natoms: usize) -> MemoryPlan {
    let mut plan = MemoryPlan::resolve(None, "DFT grid + Becke weight response");
    // `pre`: the (home atom, xyz, w_r·w_l) staging vector, dropped once the
    // grid is built — but live *alongside* grid and weight1 while `unzip`
    // runs, so it is not a transient in the peak-forming sense.
    plan.reserve_sized(
        "grid staging (home, xyz, w)",
        npts,
        std::mem::size_of::<(usize, [f64; 3], f64)>(),
        Lifetime::Resident,
        1,
    );
    plan.reserve_sized(
        "grid points",
        npts,
        std::mem::size_of::<GridPoint>(),
        Lifetime::Resident,
        1,
    );
    // The payload: weight1[g][b][α], npts · natoms · 3 f64.
    plan.reserve(
        "weight1 dw/dR [point][atom][xyz]",
        npts.saturating_mul(natoms).saturating_mul(3),
        Lifetime::Resident,
    );
    // ...and one Vec header per grid point, which is NOT negligible: 24 bytes
    // × npts is ~12 MB at 500k points, and it is 500k separate heap blocks.
    plan.reserve_sized(
        "weight1 per-point Vec headers",
        npts,
        std::mem::size_of::<Vec<[f64; 3]>>(),
        Lifetime::Resident,
        1,
    );
    plan
}

/// Bytes [`build_atomic_grid_with_response`] leaves resident for its caller
/// (grid + `weight1`), for callers that must budget around it.
pub(crate) fn grid_response_bytes(npts: usize, natoms: usize) -> usize {
    grid_response_plan(npts, natoms)
        .reservations()
        .iter()
        .filter(|r| r.label != "grid staging (home, xyz, w)")
        .map(|r| r.bytes())
        .fold(0usize, |a, b| a.saturating_add(b))
}

/// One grid point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPoint {
    pub xyz: [f64; 3],
    pub weight: f64,
    pub home_atom: usize,
}

/// Configuration for a Becke-Lebedev atomic grid.
#[derive(Debug, Clone)]
pub struct AtomicGridConfig {
    pub n_radial: usize,
    pub n_angular: usize,
    /// Angular-order pruning scheme, or `None` for the flat (unpruned) grid.
    ///
    /// `None` is the default and reproduces the historical grid exactly. Opting
    /// into [`PruneScheme::NwchemLike`] cuts ~23% of grid points (measured on
    /// H2O/CH4/benzene at 75x110; 33.5% at 99x302) for a ΔE_xc of ~1e-10 Ha —
    /// roughly 10^5 inside the reference-test budget, i.e. effectively free.
    ///
    /// Read by `KsXc::new*` / `KsXcUks::new*` (both the main and the NLC grid
    /// honour their own setting), and settable from the CLI via
    /// `[dft] grid_prune = "nwchem"`.
    ///
    /// NOT the default. See `crates/ferric-dft/tests/grid_prune_live_scf.rs`
    /// for the live-SCF evidence behind that choice: the accuracy is fine, but
    /// a pruned grid is not a free swap for the flat one everywhere (see the
    /// two restrictions below), so it stays opt-in.
    ///
    /// CAUTION 1 — `n_angular = 50`: pruning ERRORS rather than silently
    /// returning a flat grid (NWChem's middle regions want order 74, which
    /// ferric would have to snap UP to 110, enlarging the grid — the
    /// config-honesty convention forbids the silent fallback). The NLC/fallback
    /// grids in `rhf.rs`/`uhf.rs`/`rohf.rs` use 50x50, so those must keep this
    /// `None`. They do: each is constructed with an explicit `..Default::
    /// default()` over a `Default` whose `prune` is `None`. This is also why
    /// flipping that `Default` is NOT a safe way to turn pruning on globally —
    /// it would poison every 50x50 NLC grid in the workspace at once.
    ///
    /// Gradients: [`build_atomic_grid_with_response`] honours a pruned config
    /// (it takes [`build_atomic_grid_with_response_pruned`]), so the analytic KS
    /// gradient differentiates the pruned energy when the caller passes the
    /// SCF's grid (`ferric_scf::ks_gradient::KsGradGrids::from_config`).
    pub prune: Option<crate::prune::PruneScheme>,
    /// Grid weight cutoff (ORCA's `WeightCut`), or `None` for no cutoff.
    ///
    /// With `Some(cut)`, every builder in this module drops each grid point
    /// whose FINAL weight `w_g = w_r · w_l · w_becke^{home}(r_g)` (the value
    /// stored in [`GridPoint::weight`]) has `|w_g| < cut`. The comparison is on
    /// the absolute value (Becke weights are non-negative, so the sign never
    /// matters in practice); a NaN weight is never `< cut`, so it is kept and
    /// still poisons the integral visibly. Kept points keep their order and
    /// their bits. `None` and `Some(0.0)` are both "off" and build the grid
    /// bit for bit as without this field. A negative, NaN or infinite cutoff is
    /// an error from the fallible builders ([`build_atomic_grid_pruned`],
    /// [`atomic_grid_point_count`], the response builders) and a panic from
    /// the infallible ones ([`build_atomic_grid`],
    /// [`atomic_quadrature_points`]); the CLI and Python front ends validate it
    /// first. See [`weight_cut_threshold`].
    ///
    /// Which points go: (a) the innermost radial shells, where `w_r ∝ r² dr`
    /// is tiny, and (b) points deep inside another atom's Becke cell, where the
    /// home atom's partition weight underflows towards zero. (b) is most of a
    /// large molecule's grid, so the fraction dropped grows with molecule size.
    /// A dropped point contributes `|w_g · f(r_g)| < cut · |f(r_g)|` to every
    /// quadrature, so at `1e-14` the change in any integral is far below the
    /// grid's own quadrature error.
    ///
    /// Gradients: the response builders drop exactly the same points (same
    /// weight bits, same predicate), so the analytic gradient is the gradient
    /// of the cut energy the SCF converged. A dropped point's weight
    /// DERIVATIVE is not necessarily small; it is dropped anyway, so that the
    /// gradient stays consistent with the energy. The cut energy is
    /// discontinuous where a point's weight crosses `cut` under a nuclear
    /// displacement, so a finite-difference gradient can disagree with the
    /// analytic one by such a jump (of order `cut · |f|`); nothing smooths it.
    ///
    /// Cost: with a cutoff set, [`atomic_grid_point_count`] must evaluate the
    /// Becke weight of every point (it is no longer the cheap closed-form
    /// count); see its doc.
    ///
    /// Settable from the CLI via `[dft] grid_weight_cut = 1e-14` and from
    /// Python via `grid_weight_cut=`. NOT the default.
    pub weight_cut: Option<f64>,
}

/// The production KS grid's weight cutoff ([`AtomicGridConfig::weight_cut`]):
/// ORCA's `WeightCut` value.
pub const DEFAULT_WEIGHT_CUT: f64 = 1e-14;

impl Default for AtomicGridConfig {
    fn default() -> Self {
        // "Fine" production grid: 75 radial × 110 angular = 8250 pts/atom.
        // Comparable to ORCA "GRID5" or PySCF (75, 302).
        Self {
            n_radial: 75,
            n_angular: 110,
            // Flat grid: byte-identical to the historical behavior. See the
            // field doc for why pruning is opt-in until it is validated
            // through a live KS SCF against the reference suites.
            prune: None,
            // No weight cutoff: every (shell, angular point) pair is kept.
            weight_cut: None,
        }
    }
}

impl AtomicGridConfig {
    /// The KS SCF's default XC grid, used wherever a KS code path finds no
    /// explicit grid (`RhfConfig::dft_grid = None`: the SCF, its gradients,
    /// Hessians, response and TDDFT, so they all see the same grid): 75 radial
    /// × 110 angular per atom, pruned with [`PruneScheme::NwchemRadial`] (H/He
    /// 50 and Li-Ne 66 radial shells, NWChem angular regions) and a
    /// [`DEFAULT_WEIGHT_CUT`] weight cutoff. 42-47% fewer points than the flat
    /// 75x110 grid on organic molecules. Measured on live PBE/def2-SVP SCFs:
    /// within 2e-7 to 4e-6 Ha of the flat grid, whose own error against a
    /// 99x590 grid is 4.5e-4 to 3.3e-3 Ha on C10-C32 alkanes; the cutoff moved
    /// energies by under 1e-9 Ha.
    ///
    /// [`Default`] stays the flat, uncut 75x110 grid: the analysis quadratures
    /// (Hirshfeld volumes, partitions, property grids) and the tests that pin
    /// flat-grid numbers build on it.
    #[must_use]
    pub fn production() -> Self {
        Self {
            n_radial: 75,
            n_angular: 110,
            prune: Some(PruneScheme::NwchemRadial),
            weight_cut: Some(DEFAULT_WEIGHT_CUT),
        }
    }

    /// [`production`](Self::production) at `n_radial` x `n_angular`: the
    /// default scheme when it has a region table for `n_angular` (it has none
    /// at 50, and leaves orders below 50 flat), else the flat grid, with the
    /// default weight cutoff either way. What an explicit grid size with no
    /// explicit `grid_prune` means, in the CLI and in Python.
    pub fn production_sized(n_radial: usize, n_angular: usize) -> Self {
        let p = Self::production();
        let prune = p
            .prune
            .filter(|&s| crate::prune::validate_xc_scheme(s, n_angular).is_ok());
        Self {
            n_radial,
            n_angular,
            prune,
            weight_cut: p.weight_cut,
        }
    }

    /// The default VV10 nonlocal grid (50 x 50, unpruned): the one every SCF
    /// driver and the KS gradient use when `nlc_grid` is `None`.
    #[must_use]
    pub fn default_nlc() -> Self {
        Self {
            n_radial: 50,
            n_angular: 50,
            prune: None,
            weight_cut: None,
        }
    }

    /// Opt-in pruned production grid: same 75×110 base as [`Self::default`],
    /// but with [`PruneScheme::NwchemLike`] angular pruning enabled (~23% fewer
    /// points). Not valid where a 50-point angular order is needed; see the
    /// `prune` field's CAUTION notes. A gradient taken on this grid must be
    /// built from the same config (see the field's "Gradients" note).
    #[must_use]
    pub fn nwchem_pruned() -> Self {
        Self {
            n_radial: 75,
            n_angular: 110,
            prune: Some(PruneScheme::NwchemLike),
            weight_cut: None,
        }
    }
}

/// Build the full molecular grid for `mol`. Returns a flat Vec of GridPoints
/// summed across all atoms.
///
/// The Becke-weight evaluation at each point is independent of every other
/// point (it only reads `mol` and that point's `xyz`), so this is a pure map
/// over the point axis: build the un-weighted `(home_atom, xyz, w_r*w_l)`
/// list first (cheap, serial — radial/angular quadrature only), then map the
/// weight over it with an order-preserving parallel map. Collecting in index
/// order makes the output bit-identical to a serial loop regardless of thread
/// count — no reduction, so no nondeterminism to guard against.
///
/// Honours `cfg.weight_cut` (dropping points after weighing them, order
/// kept) but NOT `cfg.prune`: this is the flat grid by contract.
///
/// # Panics
///
/// On an invalid `cfg.weight_cut` (negative, NaN or infinite); see
/// [`weight_cut_threshold`].
pub fn build_atomic_grid(mol: &Molecule, cfg: &AtomicGridConfig) -> Vec<GridPoint> {
    let cut = active_weight_cut(cfg);
    let pre = flat_quadrature_points(mol, cfg);

    apply_weight_cut(weigh_points(mol, &pre), cut)
}

/// The active weight cutoff of `cfg`: `Ok(None)` when off (`weight_cut` unset
/// or `Some(0.0)`), `Ok(Some(cut))` with a finite `cut > 0` when on, and `Err`
/// for a negative, NaN or infinite cutoff (config honesty: an invalid cutoff is
/// never silently treated as off). See [`AtomicGridConfig::weight_cut`].
pub fn weight_cut_threshold(cfg: &AtomicGridConfig) -> Result<Option<f64>, FerricError> {
    match cfg.weight_cut {
        None => Ok(None),
        Some(c) if c.is_finite() && c >= 0.0 => Ok((c > 0.0).then_some(c)),
        Some(c) => Err(FerricError::General(format!(
            "grid weight cutoff (AtomicGridConfig::weight_cut) must be finite and >= 0, got {c}"
        ))),
    }
}

/// [`weight_cut_threshold`] for the infallible builders.
fn active_weight_cut(cfg: &AtomicGridConfig) -> Option<f64> {
    weight_cut_threshold(cfg).unwrap_or_else(|e| panic!("{e}"))
}

/// Whether a point of final weight `weight` survives the cutoff `cut`:
/// everything except `|weight| < cut` (so a NaN weight is kept).
#[inline]
fn survives_weight_cut(weight: f64, cut: f64) -> bool {
    weight.is_nan() || weight.abs() >= cut
}

/// Drop the points of `grid` below `cut` (in place, order kept); `None` returns
/// `grid` untouched.
fn apply_weight_cut(mut grid: Vec<GridPoint>, cut: Option<f64>) -> Vec<GridPoint> {
    if let Some(cut) = cut {
        let before = grid.len();
        grid.retain(|p| survives_weight_cut(p.weight, cut));
        if grid.len() < before {
            // The grid is held for the whole SCF; give the dropped tail back.
            grid.shrink_to_fit();
        }
    }
    grid
}

/// `w_rl * becke_weights_all(mol, xyz)[home]` for each point of `pre`, in order,
/// with the same bits, through one [`BeckeTable`] and per-thread scratch.
fn weigh_points(mol: &Molecule, pre: &[(usize, [f64; 3], f64)]) -> Vec<GridPoint> {
    let table = BeckeTable::new(mol);
    let build_point =
        |s: &mut BeckeScratch, &(a_idx, xyz, w_rl): &(usize, [f64; 3], f64)| GridPoint {
            xyz,
            weight: w_rl * table.weight(xyz, a_idx, s),
            home_atom: a_idx,
        };
    if pre.len() >= PAR_WORK_THRESHOLD {
        pre.par_iter()
            .map_init(|| table.scratch(), build_point)
            .collect()
    } else {
        let mut s = table.scratch();
        pre.iter().map(|p| build_point(&mut s, p)).collect()
    }
}

/// `atomic_quadrature_points` before the weight cutoff. With `cfg.prune` set,
/// the pruned grid's points in [`build_atomic_grid_pruned`]'s order (the same
/// per-atom radial counts and per-shell Lebedev orders); each point still
/// rides its home atom, so the moving-grid Hessian applies as is.
fn uncut_quadrature_points(mol: &Molecule, cfg: &AtomicGridConfig) -> Vec<(usize, [f64; 3], f64)> {
    if cfg.prune.is_none() {
        return flat_quadrature_points(mol, cfg);
    }
    let n = uncut_point_count(mol, cfg, cfg.prune)
        .expect("atomic_quadrature_points: a pruned cfg its caller validated");
    response_staging(mol, cfg, n)
        .expect("atomic_quadrature_points: a pruned cfg its caller validated")
}

/// The un-partitioned points of the flat grid [`build_atomic_grid`] builds:
/// `(home atom, xyz, w_r · w_l)` in its order (atom-major, then radial, then
/// angular), before the Becke weight of the home atom multiplies the last
/// entry. Each point rides its home atom (`xyz = R_home + r · Ω`), which is
/// what the moving-grid Hessian differentiates.
///
/// With `cfg.weight_cut` set, it keeps exactly the points
/// [`build_atomic_grid`] keeps (same order), so the two stay index-aligned;
/// deciding that needs each point's Becke weight, i.e. one full weight pass
/// (the cost of [`build_atomic_grid`] itself). Without a cutoff it computes no
/// Becke weights. With `cfg.prune` set it gives the pruned grid's points, in
/// [`build_atomic_grid_pruned`]'s order (which ignores nothing; the flat
/// [`build_atomic_grid`] ignores `cfg.prune`).
///
/// # Panics
///
/// On an invalid `cfg.weight_cut`; see [`weight_cut_threshold`].
pub fn atomic_quadrature_points(
    mol: &Molecule,
    cfg: &AtomicGridConfig,
) -> Vec<(usize, [f64; 3], f64)> {
    let mut pre = uncut_quadrature_points(mol, cfg);
    if let Some(cut) = active_weight_cut(cfg) {
        let weighed = weigh_points(mol, &pre);
        let mut weights = weighed.iter().map(|p| p.weight);
        // `retain` visits every element once, in order.
        pre.retain(|_| survives_weight_cut(weights.next().expect("same length"), cut));
        pre.shrink_to_fit();
    }
    pre
}

/// [`atomic_quadrature_points`] before any weight cutoff: every (atom, shell,
/// angular point) of the flat grid.
fn flat_quadrature_points(mol: &Molecule, cfg: &AtomicGridConfig) -> Vec<(usize, [f64; 3], f64)> {
    let (lebedev_pts, lebedev_w) = lebedev(cfg.n_angular);
    let mut pre: Vec<(usize, [f64; 3], f64)> =
        Vec::with_capacity(mol.atoms.len() * cfg.n_radial * lebedev_pts.len());
    for (a_idx, atom) in mol.atoms.iter().enumerate() {
        let (rs, ws) = treutler_ahlrichs_m4(atom.z, cfg.n_radial);
        for (r, w_r) in rs.iter().zip(ws.iter()) {
            for (pt, w_l) in lebedev_pts.iter().zip(lebedev_w.iter()) {
                let xyz = [
                    atom.x + r * pt[0],
                    atom.y + r * pt[1],
                    atom.zpos + r * pt[2],
                ];
                pre.push((a_idx, xyz, w_r * w_l));
            }
        }
    }
    pre
}

/// Build the molecular grid with **opt-in angular pruning**.
///
/// `prune = None` is exactly [`build_atomic_grid`] (bit-identical — it
/// delegates), so this is a strict superset of the flat path and the flat grid
/// remains the default everywhere. `prune = Some(scheme)` reduces the Lebedev
/// order on inner-core and far-tail radial shells per
/// [`crate::prune::angular_orders_for_atom`], keeping the full order only in
/// the chemically-active middle regions.
///
/// Returns `Err` (never a silent fallback to the flat grid) if the requested
/// angular order has no valid pruned region table — see
/// [`crate::prune::region_orders`].
///
/// # Weight correctness
///
/// Lebedev weights are normalised to `Σ w = 1` at every order, so mixing
/// orders across radial shells leaves `Σ_r Σ_Ω w_r w_Ω` unchanged shell by
/// shell. No renormalisation is applied or needed.
///
/// `cfg.weight_cut` is honoured in both branches (see
/// [`AtomicGridConfig::weight_cut`]); an invalid cutoff is an `Err`.
pub fn build_atomic_grid_pruned(
    mol: &Molecule,
    cfg: &AtomicGridConfig,
    prune: Option<PruneScheme>,
) -> std::result::Result<Vec<GridPoint>, FerricError> {
    let cut = weight_cut_threshold(cfg)?;
    let Some(scheme) = prune else {
        return Ok(build_atomic_grid(mol, cfg));
    };

    // Cache one Lebedev table per distinct order actually used, so a 75-shell
    // atom does not regenerate the same rule 75 times.
    let mut cache: Vec<(usize, Vec<[f64; 3]>, Vec<f64>)> = Vec::new();
    let mut pre: Vec<(usize, [f64; 3], f64)> = Vec::new();

    for (a_idx, atom) in mol.atoms.iter().enumerate() {
        let (rs, ws) = treutler_ahlrichs_m4(
            atom.z,
            crate::prune::radial_shells(atom.z, cfg.n_radial, Some(scheme)),
        );
        let orders = shell_angular_orders(atom.z, &rs, cfg, Some(scheme))?;
        for ((r, w_r), &order) in rs.iter().zip(ws.iter()).zip(orders.iter()) {
            if !cache.iter().any(|(o, _, _)| *o == order) {
                let (p, w) = lebedev(order);
                cache.push((order, p, w));
            }
            let (_, lebedev_pts, lebedev_w) = cache
                .iter()
                .find(|(o, _, _)| *o == order)
                .expect("just inserted");
            for (pt, w_l) in lebedev_pts.iter().zip(lebedev_w.iter()) {
                let xyz = [
                    atom.x + r * pt[0],
                    atom.y + r * pt[1],
                    atom.zpos + r * pt[2],
                ];
                pre.push((a_idx, xyz, w_r * w_l));
            }
        }
    }

    Ok(apply_weight_cut(weigh_points(mol, &pre), cut))
}

/// Lebedev order of each radial shell `rs` of an atom with charge `z`: the
/// flat `cfg.n_angular` everywhere when `prune` is `None`, otherwise the
/// scheme's per-shell orders. [`build_atomic_grid_pruned`] and
/// [`atomic_grid_point_count`] both resolve orders through this one function,
/// so the count cannot drift from the grid the builder produces.
fn shell_angular_orders(
    z: i32,
    rs: &[f64],
    cfg: &AtomicGridConfig,
    prune: Option<PruneScheme>,
) -> std::result::Result<Vec<usize>, FerricError> {
    match prune {
        None => Ok(vec![cfg.n_angular; rs.len()]),
        Some(scheme) => angular_orders_for_atom(z, rs, cfg.n_angular, scheme),
    }
}

/// Number of points [`build_atomic_grid_pruned`]`(mol, cfg, prune)` returns
/// (and, for `prune = None`, [`build_atomic_grid`]`(mol, cfg)`), exactly.
///
/// Without a weight cutoff (`cfg.weight_cut` `None` or `Some(0.0)`) every
/// (shell, angular point) pair is kept, so the count needs neither the points
/// nor their Becke weights: the cost is the radial quadrature per atom plus one
/// Lebedev rule per distinct order — O(natoms · n_radial), independent of the
/// grid size.
///
/// WITH a cutoff the count depends on every point's Becke weight, so it
/// evaluates all of them: the same O(npts · natoms²) Becke pass as the
/// builder (parallel over radial shells), but streamed — no point list, no
/// grid, nothing npts-sized is allocated. Callers that count and then build
/// therefore pay the Becke pass twice. The kept-or-dropped decision uses the
/// builder's exact weight bits, so the count cannot drift from the grid.
///
/// Errors exactly where the builder errors (e.g. pruning at `n_angular = 50`,
/// or an invalid cutoff).
pub fn atomic_grid_point_count(
    mol: &Molecule,
    cfg: &AtomicGridConfig,
    prune: Option<PruneScheme>,
) -> std::result::Result<usize, FerricError> {
    match weight_cut_threshold(cfg)? {
        None => uncut_point_count(mol, cfg, prune),
        Some(cut) => count_points_surviving_cut(mol, cfg, prune, cut),
    }
}

/// [`atomic_grid_point_count`] ignoring `cfg.weight_cut`: the closed-form
/// count of every (shell, angular point) pair. Also an upper bound on the cut
/// grid's size.
fn uncut_point_count(
    mol: &Molecule,
    cfg: &AtomicGridConfig,
    prune: Option<PruneScheme>,
) -> std::result::Result<usize, FerricError> {
    // Size of each distinct Lebedev rule, taken from the rule itself rather
    // than assumed equal to its order.
    let mut rule_len: Vec<(usize, usize)> = Vec::new();
    let mut total = 0usize;
    for atom in &mol.atoms {
        let (rs, _) = treutler_ahlrichs_m4(
            atom.z,
            crate::prune::radial_shells(atom.z, cfg.n_radial, prune),
        );
        for order in shell_angular_orders(atom.z, &rs, cfg, prune)? {
            let n = match rule_len.iter().find(|(o, _)| *o == order) {
                Some(&(_, n)) => n,
                None => {
                    let n = lebedev(order).0.len();
                    rule_len.push((order, n));
                    n
                }
            };
            total += n;
        }
    }
    Ok(total)
}

/// The number of points of the `prune` grid whose final weight survives
/// `cut` (> 0), streamed shell by shell: each point's `xyz`, `w_r · w_l` and
/// Becke weight are formed exactly as the builders form them (same
/// expressions, same [`BeckeTable`]), so the decision matches theirs bit for
/// bit. Integer sum, so the parallel reduction is exact.
fn count_points_surviving_cut(
    mol: &Molecule,
    cfg: &AtomicGridConfig,
    prune: Option<PruneScheme>,
    cut: f64,
) -> std::result::Result<usize, FerricError> {
    // One Lebedev rule per distinct order, and one (atom, r, w_r, rule) entry
    // per radial shell — O(natoms · n_radial), the only allocations.
    let mut rules: Vec<(usize, Vec<[f64; 3]>, Vec<f64>)> = Vec::new();
    let mut shells: Vec<(usize, f64, f64, usize)> = Vec::new();
    for (a_idx, atom) in mol.atoms.iter().enumerate() {
        let (rs, ws) = treutler_ahlrichs_m4(
            atom.z,
            crate::prune::radial_shells(atom.z, cfg.n_radial, prune),
        );
        let orders = shell_angular_orders(atom.z, &rs, cfg, prune)?;
        for ((&r, &w_r), &order) in rs.iter().zip(ws.iter()).zip(orders.iter()) {
            let rule = match rules.iter().position(|(o, _, _)| *o == order) {
                Some(i) => i,
                None => {
                    let (p, w) = lebedev(order);
                    rules.push((order, p, w));
                    rules.len() - 1
                }
            };
            shells.push((a_idx, r, w_r, rule));
        }
    }

    let table = BeckeTable::new(mol);
    let count_shell = |s: &mut BeckeScratch, &(a_idx, r, w_r, rule): &(usize, f64, f64, usize)| {
        let atom = &mol.atoms[a_idx];
        let (_, lebedev_pts, lebedev_w) = &rules[rule];
        lebedev_pts
            .iter()
            .zip(lebedev_w.iter())
            .filter(|(pt, w_l)| {
                let xyz = [
                    atom.x + r * pt[0],
                    atom.y + r * pt[1],
                    atom.zpos + r * pt[2],
                ];
                let w_rl = w_r * *w_l;
                survives_weight_cut(w_rl * table.weight(xyz, a_idx, s), cut)
            })
            .count()
    };
    let n_pre: usize = shells.iter().map(|sh| rules[sh.3].1.len()).sum();
    let total: usize = if n_pre >= PAR_WORK_THRESHOLD {
        shells
            .par_iter()
            .map_init(|| table.scratch(), count_shell)
            .sum()
    } else {
        let mut s = table.scratch();
        shells.iter().map(|sh| count_shell(&mut s, sh)).sum()
    };
    Ok(total)
}

/// Build the molecular grid AND the per-grid-point full quadrature-weight
/// nuclear-coordinate gradient (PySCF "weight1" convention).
///
/// Returns `(grid, weight1)` where `weight1[g][b][α]` is the total derivative
/// of `w_g = w_r · w_l · w_becke^{home(g)}(r_g(R); R)` with respect to `R_b^α`,
/// treating `r_g` as **moving rigidly with its home atom** (the convention
/// PySCF uses in `grids_response_cc`). Equivalent to:
///
/// ```text
///   weight1[g][b][α] = (w_r · w_l) · [ ∂w_becke^{home}/∂R_b^α|_{r lab-fixed}
///                                      + δ_{b, home} · ∇_r w_becke^{home}(r_g) ]
/// ```
///
/// The `∇_r w_becke` piece is reconstructed via translational invariance
/// `Σ_c ∂w_becke/∂R_c|_{r fixed} + ∇_r w_becke = 0` ⇒
/// `∇_r w_becke = -Σ_c ∂w_becke/∂R_c|_{r fixed}`, applied only to the home
/// row. After this fix, `Σ_b weight1[g][b][α] = 0` at every grid point
/// (rigid-translation invariance, exact).
///
/// Used by the XC gradient grid-response correction (P2.1).
///
/// # Memory
///
/// `weight1` is `npts × natoms × 3` `f64` — it grows with the *product* of the
/// grid size and the atom count, which makes it the largest single allocation
/// in this crate for anything past a few dozen atoms: at 50 atoms and 500k
/// grid points it is ~600 MB of payload plus ~12 MB of per-point `Vec`
/// headers. It was previously built with no guard at all, so an over-sized
/// gradient job died in the allocator rather than at a budget check. The
/// [`MemoryPlan`] below is that check; it runs before `pre` (the first big
/// allocation) is filled, so nothing large is resident when it fires.
///
/// Follow-up, deliberately *not* done here: `weight1` is a
/// `Vec<Vec<[f64; 3]>>`, i.e. one heap allocation **per grid point** (the
/// `Vec::with_capacity(natoms)` in the per-point closure below). That is 500k
/// separate allocations and 24 bytes of header each on the 50-atom case, and a
/// flat `Array3<f64>` of `(npts, natoms, 3)` would remove both. Changing the
/// layout would change the access pattern of every consumer in `gradient.rs`
/// and is out of scope for a memory-accounting fix, so the plan below simply
/// counts the headers as the real cost they are.
pub fn build_atomic_grid_with_response(
    mol: &Molecule,
    cfg: &AtomicGridConfig,
) -> Result<(Vec<GridPoint>, Vec<Vec<[f64; 3]>>), FerricError> {
    // The weight-response algebra is order-agnostic (each point's derivative
    // depends only on its own position and home atom), so a pruned config is
    // differentiated like a flat one: the gradient is that of the pruned energy
    // the SCF converged. `prune = None` is the flat grid exactly.
    build_atomic_grid_with_response_pruned(mol, cfg)
}

/// [`build_atomic_grid_with_response`] honouring `cfg.prune`: the same points
/// (in the same order) and weights as [`build_atomic_grid_pruned`]`(mol, cfg,
/// cfg.prune)`, plus `weight1`. With `cfg.prune = None` it is exactly the
/// flat-grid response builder. Per-shell Lebedev orders change only WHICH
/// points exist; each point's weight derivative is the same Becke expression
/// (`weight1` treats every point as riding its home atom rigidly, whatever
/// its angular rule), so no extra term arises from pruning.
///
/// `cfg.weight_cut` is honoured the same way: a point is dropped from BOTH
/// `grid` and `weight1` exactly when the energy builders drop it (same weight
/// bits, same predicate), so the gradient differentiates the cut energy. A
/// dropped point's `weight1` row is not necessarily small; dropping it anyway
/// is what keeps energy and gradient consistent, and the cut energy is
/// discontinuous where a weight crosses the threshold under displacement (see
/// [`AtomicGridConfig::weight_cut`]). Dropped points skip the O(natoms³)
/// weight-derivative evaluation. The memory check is sized on the UNCUT point
/// count (an upper bound, and what the staging list really holds), so it never
/// needs the Becke pass that an exact cut count would cost.
pub fn build_atomic_grid_with_response_pruned(
    mol: &Molecule,
    cfg: &AtomicGridConfig,
) -> Result<(Vec<GridPoint>, Vec<Vec<[f64; 3]>>), FerricError> {
    let natoms = mol.atoms.len();
    let cut = weight_cut_threshold(cfg)?;
    // Exact without a cutoff; with one, an upper bound (the staging list `pre`
    // really is this long, and the cut grid and `weight1` are no longer).
    let npts = uncut_point_count(mol, cfg, cfg.prune)?;
    grid_response_plan(npts, natoms).check()?;

    let pre = response_staging(mol, cfg, npts)?;

    // Per-point work is O(natoms³) via becke_weights_and_grad and independent
    // across points — same order-preserving parallel map as build_atomic_grid
    // (bit-identical to the serial loop; no reduction).
    //
    // `None` = dropped by the weight cutoff (never, when `cut` is `None`).
    let build_point =
        |&(a_idx, xyz, scale): &(usize, [f64; 3], f64)| -> Option<(GridPoint, Vec<[f64; 3]>)> {
            // The weight VALUE comes from the same `becke_weights_all` the
            // energy grids use, so this grid is the energy grid bit for bit
            // (`becke_weights_and_grad`'s own value differs by ulps), and the
            // cutoff decision below is the energy builders' decision.
            let weight = scale * becke_weights_all(mol, xyz)[a_idx];
            if let Some(cut) = cut {
                if !survives_weight_cut(weight, cut) {
                    return None;
                }
            }
            let (_, dw_lab) = becke_weights_and_grad(mol, xyz);
            let gp = GridPoint {
                xyz,
                weight,
                home_atom: a_idx,
            };

            let row = home_weight_row(&dw_lab, a_idx, natoms, scale);
            Some((gp, row))
        };

    let (grid, weight1) = map_response_points(&pre, cut.is_some(), build_point);
    Ok((grid, weight1))
}

/// The un-partitioned points of [`build_atomic_grid_with_response_pruned`]:
/// `(home atom, xyz, w_r · w_l)` atom-major, then radial, then angular, with
/// the per-atom radial count and per-shell Lebedev orders of `cfg.prune`.
fn response_staging(
    mol: &Molecule,
    cfg: &AtomicGridConfig,
    npts: usize,
) -> Result<Vec<(usize, [f64; 3], f64)>, FerricError> {
    let mut cache: Vec<(usize, Vec<[f64; 3]>, Vec<f64>)> = Vec::new();
    let mut pre: Vec<(usize, [f64; 3], f64)> = Vec::with_capacity(npts);

    for (a_idx, atom) in mol.atoms.iter().enumerate() {
        let (rs, ws) = treutler_ahlrichs_m4(
            atom.z,
            crate::prune::radial_shells(atom.z, cfg.n_radial, cfg.prune),
        );
        let orders = shell_angular_orders(atom.z, &rs, cfg, cfg.prune)?;
        for ((r, w_r), &order) in rs.iter().zip(ws.iter()).zip(orders.iter()) {
            if !cache.iter().any(|(o, _, _)| *o == order) {
                let (p, w) = lebedev(order);
                cache.push((order, p, w));
            }
            let (_, lebedev_pts, lebedev_w) = cache
                .iter()
                .find(|(o, _, _)| *o == order)
                .expect("just inserted");
            for (pt, w_l) in lebedev_pts.iter().zip(lebedev_w.iter()) {
                let xyz = [
                    atom.x + r * pt[0],
                    atom.y + r * pt[1],
                    atom.zpos + r * pt[2],
                ];
                pre.push((a_idx, xyz, w_r * w_l));
            }
        }
    }

    Ok(pre)
}

/// Row `a_idx` of `weight1` for one point: the lab-fixed partition derivative
/// of the home atom's Becke weight, plus `∇_r w` on the home row (PySCF's
/// `weight1`), times the radial-angular Jacobian `scale`.
fn home_weight_row(
    dw_lab: &[Vec<[f64; 3]>],
    a_idx: usize,
    natoms: usize,
    scale: f64,
) -> Vec<[f64; 3]> {
    // Lab-fixed partition derivative for the HOME atom's weight:
    // dw_lab[a_idx][c][k] is ∂w_becke^{a_idx}/∂R_c^k at fixed r.
    // PySCF's weight1 adds ∇_r w_becke to the home-atom row, then
    // multiplies through by the radial-angular Jacobian.
    let mut grad_r = [0.0_f64; 3]; // = -Σ_c dw_lab[a_idx][c]
    for c in 0..natoms {
        grad_r[0] -= dw_lab[a_idx][c][0];
        grad_r[1] -= dw_lab[a_idx][c][1];
        grad_r[2] -= dw_lab[a_idx][c][2];
    }
    let mut row = Vec::with_capacity(natoms);
    for b in 0..natoms {
        let mut entry = [
            scale * dw_lab[a_idx][b][0],
            scale * dw_lab[a_idx][b][1],
            scale * dw_lab[a_idx][b][2],
        ];
        if b == a_idx {
            entry[0] += scale * grad_r[0];
            entry[1] += scale * grad_r[1];
            entry[2] += scale * grad_r[2];
        }
        row.push(entry);
    }
    row
}

/// Map `build_point` over the staged points in order: the indexed map when no
/// cutoff can drop a point (the historical path), `filter_map` otherwise;
/// parallel above [`PAR_WORK_THRESHOLD`]. rayon's `filter_map(..).unzip()`
/// keeps the input order, like the serial one.
fn map_response_points<F>(
    pre: &[(usize, [f64; 3], f64)],
    has_cut: bool,
    build_point: F,
) -> (Vec<GridPoint>, Vec<Vec<[f64; 3]>>)
where
    F: Fn(&(usize, [f64; 3], f64)) -> Option<(GridPoint, Vec<[f64; 3]>)> + Sync + Send,
{
    let par = pre.len() >= PAR_WORK_THRESHOLD;
    match (has_cut, par) {
        (false, true) => pre
            .par_iter()
            .map(|p| build_point(p).expect("no weight cutoff: every point is kept"))
            .unzip(),
        (false, false) => pre
            .iter()
            .map(|p| build_point(p).expect("no weight cutoff: every point is kept"))
            .unzip(),
        (true, true) => pre.par_iter().filter_map(build_point).unzip(),
        (true, false) => pre.iter().filter_map(build_point).unzip(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::mol::{Atom, Molecule};

    /// Distorted water (Bohr) for the grid-identity anchors.
    fn water() -> Molecule {
        let at = |symbol: &str, z: i32, x: f64, y: f64, zpos: f64| Atom {
            symbol: symbol.into(),
            z,
            x,
            y,
            zpos,
            ghost: false,
            n_core_ecp: 0,
        };
        Molecule {
            atoms: vec![
                at("O", 8, 0.0, 0.04, 0.21),
                at("H", 1, 0.06, 1.44, -0.89),
                at("H", 1, -0.04, -1.42, -0.85),
            ],
            charge: 0,
            multiplicity: 1,
        }
    }

    /// EXACTNESS ANCHOR: the response builder returns the energy builder's
    /// grid BIT FOR BIT — flat (the pre-pruning response grid, now routed
    /// through the pruned builder) and pruned (`sgx`, peak 194). A gradient
    /// built on a different grid than the energy would miss FD by the grid
    /// difference; this pins that it cannot. Fails if the response builder
    /// ignores `prune` (point count differs) or orders points differently.
    #[test]
    fn response_grid_is_the_energy_grid_bit_for_bit() {
        let mol = water();
        for prune in [None, Some(PruneScheme::Sgx)] {
            let cfg = AtomicGridConfig {
                n_radial: 35,
                n_angular: 194,
                prune,
                weight_cut: None,
            };
            let e = build_atomic_grid_pruned(&mol, &cfg, cfg.prune).unwrap();
            let (g, w1) = build_atomic_grid_with_response_pruned(&mol, &cfg).unwrap();
            assert_eq!(e.len(), g.len(), "{prune:?}: point counts differ");
            assert_eq!(w1.len(), g.len());
            assert_eq!(
                e.len(),
                atomic_grid_point_count(&mol, &cfg, cfg.prune).unwrap()
            );
            for (a, b) in e.iter().zip(&g) {
                assert_eq!(a.home_atom, b.home_atom);
                for k in 0..3 {
                    assert_eq!(a.xyz[k].to_bits(), b.xyz[k].to_bits(), "{prune:?}");
                }
                assert_eq!(a.weight.to_bits(), b.weight.to_bits(), "{prune:?} weight");
            }
        }
        // The pruned grid really is smaller (the scheme is live).
        let flat = AtomicGridConfig {
            n_radial: 35,
            n_angular: 194,
            prune: None,
            weight_cut: None,
        };
        let sgx = AtomicGridConfig {
            prune: Some(PruneScheme::Sgx),
            ..flat.clone()
        };
        let (nf, ns) = (
            atomic_grid_point_count(&mol, &flat, flat.prune).unwrap(),
            atomic_grid_point_count(&mol, &sgx, sgx.prune).unwrap(),
        );
        assert!(ns < nf, "sgx {ns} vs flat {nf}");
        // The XC-path wrapper (#428) honours a pruned config: same count.
        let (xg, _) = build_atomic_grid_with_response(&mol, &sgx).unwrap();
        assert_eq!(xg.len(), ns);
    }

    fn h2() -> Molecule {
        Molecule {
            atoms: vec![
                Atom {
                    symbol: "H".into(),
                    z: 1,
                    x: 0.0,
                    y: 0.0,
                    zpos: 0.0,
                    ghost: false,
                    n_core_ecp: 0,
                },
                Atom {
                    symbol: "H".into(),
                    z: 1,
                    x: 0.0,
                    y: 0.0,
                    zpos: 1.4,
                    ghost: false,
                    n_core_ecp: 0,
                },
            ],
            charge: 0,
            multiplicity: 1,
        }
    }

    #[test]
    fn grid_integrates_unit_density_to_electron_count_h2() {
        // ∫ ρ dV = N_e for any normalized density. For H2 with ρ = 2 * 1s
        // Slater (ξ=1) localized on each atom averaged, just check that a
        // simple uniform-density-like test integrand integrates correctly.
        let mol = h2();
        let grid = build_atomic_grid(
            &mol,
            &AtomicGridConfig {
                n_radial: 75,
                n_angular: 110,
                ..Default::default()
            },
        );

        // Total weight ≈ infinity for ∫ 1 dV (whole space), so check
        // something normalizable: ∫ exp(-α r²) at H2's bond midpoint
        // (Gaussian centered at midpoint).
        let alpha = 1.0_f64;
        let center = [0.0, 0.0, 0.7];
        let approx: f64 = grid
            .iter()
            .map(|g| {
                let dx = g.xyz[0] - center[0];
                let dy = g.xyz[1] - center[1];
                let dz = g.xyz[2] - center[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                g.weight * (-alpha * r2).exp()
            })
            .sum();
        let exact = (std::f64::consts::PI / alpha).powf(1.5);
        let err = (approx - exact).abs() / exact;
        eprintln!("H2 grid: ∫ Gaussian = {approx:.6}, exact = {exact:.6}, relerr={err:.2e}");
        assert!(err < 1e-3, "H2 Becke-Lebedev Gaussian relerr {err:.2e}");
    }

    /// Serial reference: the pre-P2 per-point loop, verbatim.
    fn build_atomic_grid_serial_ref(mol: &Molecule, cfg: &AtomicGridConfig) -> Vec<GridPoint> {
        let (lebedev_pts, lebedev_w) = lebedev(cfg.n_angular);
        let mut grid = Vec::new();
        for (a_idx, atom) in mol.atoms.iter().enumerate() {
            let (rs, ws) = treutler_ahlrichs_m4(atom.z, cfg.n_radial);
            for (r, w_r) in rs.iter().zip(ws.iter()) {
                for (pt, w_l) in lebedev_pts.iter().zip(lebedev_w.iter()) {
                    let xyz = [
                        atom.x + r * pt[0],
                        atom.y + r * pt[1],
                        atom.zpos + r * pt[2],
                    ];
                    let becke = becke_weights_all(mol, xyz);
                    let weight = w_r * w_l * becke[a_idx];
                    grid.push(GridPoint {
                        xyz,
                        weight,
                        home_atom: a_idx,
                    });
                }
            }
        }
        grid
    }

    #[test]
    fn parallel_grid_bit_identical_to_serial() {
        // Default fine grid on H2: 2 * 75 * 110 = 16,500 points — well above
        // PAR_WORK_THRESHOLD, so build_atomic_grid takes the rayon path.
        let mol = h2();
        let cfg = AtomicGridConfig::default();
        assert!(
            mol.atoms.len() * cfg.n_radial * 110 >= PAR_WORK_THRESHOLD,
            "test must exercise the parallel path"
        );
        let par = build_atomic_grid(&mol, &cfg);
        let ser = build_atomic_grid_serial_ref(&mol, &cfg);
        assert_eq!(par.len(), ser.len());
        for (g, (p, s)) in par.iter().zip(ser.iter()).enumerate() {
            assert_eq!(p.home_atom, s.home_atom, "home_atom mismatch at point {g}");
            for k in 0..3 {
                assert_eq!(
                    p.xyz[k].to_bits(),
                    s.xyz[k].to_bits(),
                    "xyz[{k}] not bit-identical at point {g}"
                );
            }
            assert_eq!(
                p.weight.to_bits(),
                s.weight.to_bits(),
                "weight not bit-identical at point {g}: par {} vs ser {}",
                p.weight,
                s.weight
            );
        }
    }

    #[test]
    fn parallel_grid_with_response_bit_identical_to_serial() {
        // Serial reference for the with-response variant: pre-P2 loop, verbatim.
        let mol = h2();
        let cfg = AtomicGridConfig::default();
        let (par_grid, par_w1) = build_atomic_grid_with_response(&mol, &cfg).unwrap();

        let (lebedev_pts, lebedev_w) = lebedev(cfg.n_angular);
        let natoms = mol.atoms.len();
        let mut ser_grid = Vec::new();
        let mut ser_w1: Vec<Vec<[f64; 3]>> = Vec::new();
        for (a_idx, atom) in mol.atoms.iter().enumerate() {
            let (rs, ws) = treutler_ahlrichs_m4(atom.z, cfg.n_radial);
            for (r, w_r) in rs.iter().zip(ws.iter()) {
                for (pt, w_l) in lebedev_pts.iter().zip(lebedev_w.iter()) {
                    let xyz = [
                        atom.x + r * pt[0],
                        atom.y + r * pt[1],
                        atom.zpos + r * pt[2],
                    ];
                    let (_, dw_lab) = becke_weights_and_grad(&mol, xyz);
                    // Weight value from `becke_weights_all`, as the energy grids.
                    let weight = w_r * w_l * becke_weights_all(&mol, xyz)[a_idx];
                    ser_grid.push(GridPoint {
                        xyz,
                        weight,
                        home_atom: a_idx,
                    });
                    let scale = w_r * w_l;
                    let mut grad_r = [0.0_f64; 3];
                    for c in 0..natoms {
                        grad_r[0] -= dw_lab[a_idx][c][0];
                        grad_r[1] -= dw_lab[a_idx][c][1];
                        grad_r[2] -= dw_lab[a_idx][c][2];
                    }
                    let mut row = Vec::with_capacity(natoms);
                    for b in 0..natoms {
                        let mut entry = [
                            scale * dw_lab[a_idx][b][0],
                            scale * dw_lab[a_idx][b][1],
                            scale * dw_lab[a_idx][b][2],
                        ];
                        if b == a_idx {
                            entry[0] += scale * grad_r[0];
                            entry[1] += scale * grad_r[1];
                            entry[2] += scale * grad_r[2];
                        }
                        row.push(entry);
                    }
                    ser_w1.push(row);
                }
            }
        }

        assert_eq!(par_grid.len(), ser_grid.len());
        assert_eq!(par_w1.len(), ser_w1.len());
        for (g, (p, s)) in par_grid.iter().zip(ser_grid.iter()).enumerate() {
            assert_eq!(p.home_atom, s.home_atom);
            for k in 0..3 {
                assert_eq!(p.xyz[k].to_bits(), s.xyz[k].to_bits(), "xyz at point {g}");
            }
            assert_eq!(
                p.weight.to_bits(),
                s.weight.to_bits(),
                "weight at point {g}"
            );
        }
        for (g, (pr, sr)) in par_w1.iter().zip(ser_w1.iter()).enumerate() {
            assert_eq!(pr.len(), sr.len());
            for (b, (pe, se)) in pr.iter().zip(sr.iter()).enumerate() {
                for k in 0..3 {
                    assert_eq!(
                        pe[k].to_bits(),
                        se[k].to_bits(),
                        "weight1 not bit-identical at point {g}, atom {b}, axis {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn prune_none_is_bit_identical_to_flat_grid() {
        // The whole opt-in guarantee: passing None must not perturb a single
        // bit of the existing default path.
        let mol = h2();
        let cfg = AtomicGridConfig::default();
        let flat = build_atomic_grid(&mol, &cfg);
        let none = build_atomic_grid_pruned(&mol, &cfg, None).unwrap();
        assert_eq!(flat.len(), none.len());
        for (g, (a, b)) in flat.iter().zip(none.iter()).enumerate() {
            assert_eq!(a.home_atom, b.home_atom, "home_atom at {g}");
            assert_eq!(a.weight.to_bits(), b.weight.to_bits(), "weight at {g}");
            for k in 0..3 {
                assert_eq!(a.xyz[k].to_bits(), b.xyz[k].to_bits(), "xyz[{k}] at {g}");
            }
        }
    }

    #[test]
    fn pruned_grid_is_smaller_and_still_integrates_a_gaussian() {
        // Point-count reduction plus the integration invariant: pruning must
        // shrink the grid without breaking the quadrature. A weight
        // normalisation bug would blow this relerr up by orders of magnitude.
        let mol = h2();
        let cfg = AtomicGridConfig::default();
        let flat = build_atomic_grid(&mol, &cfg);
        let pruned = build_atomic_grid_pruned(&mol, &cfg, Some(PruneScheme::NwchemLike)).unwrap();
        let frac = 1.0 - pruned.len() as f64 / flat.len() as f64;
        eprintln!(
            "H2 75x110: flat {} pts -> pruned {} pts ({:.1}% fewer)",
            flat.len(),
            pruned.len(),
            100.0 * frac
        );
        assert!(
            pruned.len() < flat.len(),
            "pruning did not remove any points"
        );
        assert!(frac > 0.15, "pruning saved only {:.1}%", 100.0 * frac);

        let alpha = 1.0_f64;
        let center = [0.0, 0.0, 0.7];
        let integrate = |g: &[GridPoint]| -> f64 {
            g.iter()
                .map(|p| {
                    let dx = p.xyz[0] - center[0];
                    let dy = p.xyz[1] - center[1];
                    let dz = p.xyz[2] - center[2];
                    p.weight * (-alpha * (dx * dx + dy * dy + dz * dz)).exp()
                })
                .sum()
        };
        let exact = (std::f64::consts::PI / alpha).powf(1.5);
        let err_flat = (integrate(&flat) - exact).abs() / exact;
        let err_pruned = (integrate(&pruned) - exact).abs() / exact;
        eprintln!("Gaussian relerr: flat {err_flat:.2e}, pruned {err_pruned:.2e}");
        assert!(err_pruned < 1e-3, "pruned Gaussian relerr {err_pruned:.2e}");
    }

    #[test]
    fn nwchem_pruned_is_smaller() {
        // Opt-in constructor: 75x110 with NWChem-like pruning must yield fewer
        // points than the (unpruned) default grid.
        let mol = h2();
        let pruned_cfg = AtomicGridConfig::nwchem_pruned();
        assert_eq!(pruned_cfg.n_radial, 75);
        assert_eq!(pruned_cfg.n_angular, 110);
        assert!(pruned_cfg.prune.is_some());
        let flat = build_atomic_grid(&mol, &AtomicGridConfig::default());
        let pruned = build_atomic_grid_pruned(&mol, &pruned_cfg, pruned_cfg.prune).unwrap();
        assert!(
            pruned.len() < flat.len(),
            "pruned {} not smaller than flat {}",
            pruned.len(),
            flat.len()
        );
    }

    #[test]
    fn nwchem_pruned_is_smaller_on_water() {
        // Medium-system sanity check for the pruning reduction (Knuth: a 3-atom
        // H2 test is not evidence for larger systems).
        let mol = water();
        let flat = build_atomic_grid(&mol, &AtomicGridConfig::default());
        let pruned_cfg = AtomicGridConfig::nwchem_pruned();
        let pruned = build_atomic_grid_pruned(&mol, &pruned_cfg, pruned_cfg.prune).unwrap();
        assert!(
            pruned.len() < flat.len(),
            "pruned {} not smaller than flat {}",
            pruned.len(),
            flat.len()
        );
    }

    #[test]
    fn pruned_grid_partition_still_sums_to_full_space() {
        // Becke partition exactness is independent of the angular order used
        // at each shell -- confirm pruning does not disturb it.
        let mol = h2();
        let grid = build_atomic_grid_pruned(
            &mol,
            &AtomicGridConfig::default(),
            Some(PruneScheme::NwchemLike),
        )
        .unwrap();
        let f = |p: &GridPoint| {
            let dz = p.xyz[2] - 0.7;
            (-0.5 * (p.xyz[0].powi(2) + p.xyz[1].powi(2) + dz * dz)).exp()
        };
        let total: f64 = grid.iter().map(|p| p.weight * f(p)).sum();
        let mut per_atom = [0.0_f64; 2];
        for p in &grid {
            per_atom[p.home_atom] += p.weight * f(p);
        }
        let sum_atoms: f64 = per_atom.iter().sum();
        assert!(
            (sum_atoms - total).abs() < 1e-12,
            "pruned Becke partition mismatch: {sum_atoms} vs {total}"
        );
    }

    #[test]
    fn point_count_equals_built_grid_len_flat_and_pruned() {
        // O + 2 H: two element rows, so the pruned count exercises the
        // Z-dependent region boundaries, not just hydrogen's.
        let mut water = h2();
        water.atoms[1].zpos = 0.0;
        water.atoms[1].x = 1.43;
        water.atoms[1].y = 1.11;
        water.atoms[0].x = -1.43;
        water.atoms[0].y = 1.11;
        water.atoms.push(Atom {
            symbol: "O".into(),
            z: 8,
            x: 0.0,
            y: 0.0,
            zpos: 0.0,
            ghost: false,
            n_core_ecp: 0,
        });
        for mol in [h2(), water] {
            for (n_radial, n_angular) in [(75, 110), (40, 302)] {
                let cfg = AtomicGridConfig {
                    n_radial,
                    n_angular,
                    ..Default::default()
                };
                for prune in [None, Some(PruneScheme::NwchemLike)] {
                    let built = build_atomic_grid_pruned(&mol, &cfg, prune).unwrap().len();
                    let counted = atomic_grid_point_count(&mol, &cfg, prune).unwrap();
                    assert_eq!(
                        counted,
                        built,
                        "{} atoms, {n_radial}x{n_angular}, prune {prune:?}",
                        mol.atoms.len()
                    );
                }
            }
        }
        // And it refuses exactly where the builder refuses.
        let cfg50 = AtomicGridConfig {
            n_radial: 50,
            n_angular: 50,
            ..Default::default()
        };
        assert!(atomic_grid_point_count(&h2(), &cfg50, Some(PruneScheme::NwchemLike)).is_err());
    }

    #[test]
    fn pruning_errors_instead_of_silently_falling_back() {
        // n_angular = 50 has no useful pruned table on ferric's Lebedev set.
        // It must Err, not quietly return the flat grid.
        let mol = h2();
        let cfg = AtomicGridConfig {
            n_radial: 50,
            n_angular: 50,
            ..Default::default()
        };
        assert!(build_atomic_grid_pruned(&mol, &cfg, Some(PruneScheme::NwchemLike)).is_err());
        // ... but with pruning off, the same config still works.
        assert!(build_atomic_grid_pruned(&mol, &cfg, None).is_ok());
    }

    #[test]
    fn grid_partition_sums_to_full_space() {
        // Σ_A ∫_A f dV = ∫ f dV — the Becke partition is exact at every
        // grid point.
        let mol = h2();
        let grid = build_atomic_grid(
            &mol,
            &AtomicGridConfig {
                n_radial: 50,
                n_angular: 50,
                ..Default::default()
            },
        );
        let alpha = 0.5_f64;
        let center = [0.0, 0.0, 0.7];
        let total: f64 = grid
            .iter()
            .map(|g| {
                let dx = g.xyz[0] - center[0];
                let dy = g.xyz[1] - center[1];
                let dz = g.xyz[2] - center[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                g.weight * (-alpha * r2).exp()
            })
            .sum();
        // Per-atom sum to check partition reproduces total.
        let mut per_atom = [0.0_f64; 2];
        for g in &grid {
            let dx = g.xyz[0] - center[0];
            let dy = g.xyz[1] - center[1];
            let dz = g.xyz[2] - center[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            per_atom[g.home_atom] += g.weight * (-alpha * r2).exp();
        }
        let sum_atoms: f64 = per_atom.iter().sum();
        assert!(
            (sum_atoms - total).abs() < 1e-12,
            "Becke partition sum mismatch: per-atom {sum_atoms}, total {total}"
        );
    }

    /// Bitwise equality of two grids: same length, order, home atoms, points
    /// and weights.
    fn assert_grids_bit_identical(got: &[GridPoint], want: &[GridPoint], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: point counts differ");
        for (g, (a, b)) in got.iter().zip(want).enumerate() {
            assert_eq!(a.home_atom, b.home_atom, "{what}: home_atom at {g}");
            for k in 0..3 {
                assert_eq!(
                    a.xyz[k].to_bits(),
                    b.xyz[k].to_bits(),
                    "{what}: xyz[{k}] at {g}"
                );
            }
            assert_eq!(
                a.weight.to_bits(),
                b.weight.to_bits(),
                "{what}: weight at {g}"
            );
        }
    }

    fn with_cut(cfg: &AtomicGridConfig, weight_cut: Option<f64>) -> AtomicGridConfig {
        AtomicGridConfig {
            weight_cut,
            ..cfg.clone()
        }
    }

    /// Serial reference for the (optionally pruned) grid, written against the
    /// primitives (radial rule, per-shell orders, Lebedev rule,
    /// `becke_weights_all`) rather than `weigh_points` / the cutoff helpers.
    fn pruned_serial_ref(
        mol: &Molecule,
        cfg: &AtomicGridConfig,
        prune: Option<PruneScheme>,
    ) -> Vec<GridPoint> {
        let mut grid = Vec::new();
        for (a_idx, atom) in mol.atoms.iter().enumerate() {
            let (rs, ws) = treutler_ahlrichs_m4(
                atom.z,
                crate::prune::radial_shells(atom.z, cfg.n_radial, prune),
            );
            let orders = shell_angular_orders(atom.z, &rs, cfg, prune).unwrap();
            for ((r, w_r), &order) in rs.iter().zip(ws.iter()).zip(orders.iter()) {
                let (pts, wl) = lebedev(order);
                for (pt, w_l) in pts.iter().zip(wl.iter()) {
                    let xyz = [
                        atom.x + r * pt[0],
                        atom.y + r * pt[1],
                        atom.zpos + r * pt[2],
                    ];
                    grid.push(GridPoint {
                        xyz,
                        weight: w_r * w_l * becke_weights_all(mol, xyz)[a_idx],
                        home_atom: a_idx,
                    });
                }
            }
        }
        grid
    }

    /// THE OPT-IN GUARANTEE: with the cutoff off (`None` or `Some(0.0)`) every
    /// builder returns the historical grid bit for bit — flat (against the
    /// verbatim pre-P2 loop) and pruned (against a primitive-level serial
    /// reference) — and the count is the closed-form one.
    #[test]
    fn weight_cut_off_is_bit_identical_to_the_uncut_grid() {
        let mol = water();
        let base = AtomicGridConfig {
            n_radial: 35,
            n_angular: 194,
            ..Default::default()
        };
        let flat_ref = build_atomic_grid_serial_ref(&mol, &base);
        assert_grids_bit_identical(&pruned_serial_ref(&mol, &base, None), &flat_ref, "ref/ref");
        for off in [None, Some(0.0)] {
            let cfg = with_cut(&base, off);
            assert_eq!(weight_cut_threshold(&cfg).unwrap(), None);
            assert_grids_bit_identical(&build_atomic_grid(&mol, &cfg), &flat_ref, "flat");
            assert_grids_bit_identical(
                &build_atomic_grid_pruned(&mol, &cfg, None).unwrap(),
                &flat_ref,
                "flat via pruned builder",
            );
            assert_eq!(
                atomic_grid_point_count(&mol, &cfg, None).unwrap(),
                flat_ref.len()
            );
            let quad = atomic_quadrature_points(&mol, &cfg);
            assert_eq!(quad.len(), flat_ref.len(), "quadrature points");
            let (rg, rw1) = build_atomic_grid_with_response_pruned(&mol, &cfg).unwrap();
            assert_grids_bit_identical(&rg, &flat_ref, "flat response");
            assert_eq!(rw1.len(), flat_ref.len());
            for prune in [Some(PruneScheme::NwchemLike), Some(PruneScheme::Sgx)] {
                let want = pruned_serial_ref(&mol, &base, prune);
                let got = build_atomic_grid_pruned(&mol, &cfg, prune).unwrap();
                assert_grids_bit_identical(&got, &want, &format!("{prune:?} off={off:?}"));
                assert_eq!(
                    atomic_grid_point_count(&mol, &cfg, prune).unwrap(),
                    want.len()
                );
                let pcfg = AtomicGridConfig {
                    prune,
                    ..cfg.clone()
                };
                let (rg, _) = build_atomic_grid_with_response_pruned(&mol, &pcfg).unwrap();
                assert_grids_bit_identical(&rg, &want, &format!("{prune:?} response"));
            }
        }
    }

    /// With a cutoff: the cut grid is EXACTLY the uncut grid's subsequence of
    /// points with `|w| >= cut` (so every kept `|w| >= cut`, every dropped
    /// `|w| < cut`, order and bits kept); the count equals the built length;
    /// the response builder keeps the same points with the uncut `weight1`
    /// rows of those points; and the quadrature points stay index-aligned
    /// with the flat cut grid. Large (parallel) and small (serial) grids.
    #[test]
    fn weight_cut_drops_exactly_the_points_below_the_cut() {
        let mol = water();
        let cases: [(usize, usize, Option<PruneScheme>); 4] = [
            (35, 194, None),
            (35, 194, Some(PruneScheme::NwchemLike)),
            (35, 194, Some(PruneScheme::Sgx)),
            // 3 · 8 · 26 = 624 points: below PAR_WORK_THRESHOLD, serial paths.
            (8, 26, None),
        ];
        for (n_radial, n_angular, prune) in cases {
            let base = AtomicGridConfig {
                n_radial,
                n_angular,
                prune,
                weight_cut: None,
            };
            let full = build_atomic_grid_pruned(&mol, &base, prune).unwrap();
            let (full_g, full_w1) = build_atomic_grid_with_response_pruned(&mol, &base).unwrap();
            assert_grids_bit_identical(&full_g, &full, "uncut response");
            for cut in [1e-14, 1e-8] {
                let what = format!("{n_radial}x{n_angular} {prune:?} cut {cut:e}");
                let cfg = with_cut(&base, Some(cut));
                let keep = |p: &GridPoint| p.weight.abs() >= cut;
                let kept: Vec<GridPoint> = full.iter().copied().filter(keep).collect();
                let dropped: Vec<GridPoint> = full.iter().copied().filter(|p| !keep(p)).collect();
                assert!(dropped.iter().all(|p| p.weight.abs() < cut), "{what}");
                assert_eq!(kept.len() + dropped.len(), full.len());

                let got = build_atomic_grid_pruned(&mol, &cfg, prune).unwrap();
                assert!(
                    got.iter().all(|p| p.weight.abs() >= cut),
                    "{what}: kept below cut"
                );
                assert_grids_bit_identical(&got, &kept, &what);
                assert_eq!(
                    atomic_grid_point_count(&mol, &cfg, prune).unwrap(),
                    got.len(),
                    "{what}: count"
                );
                // 35 shells put the innermost r ~ 1e-4 bohr, so w_r ~ r^2 dr << 1e-8.
                if cut == 1e-8 && n_radial >= 35 {
                    assert!(!dropped.is_empty(), "{what}: nothing dropped at 1e-8");
                }

                let (rg, rw1) = build_atomic_grid_with_response_pruned(&mol, &cfg).unwrap();
                assert_grids_bit_identical(&rg, &got, &format!("{what} response"));
                let want_w1: Vec<&Vec<[f64; 3]>> = full_g
                    .iter()
                    .zip(&full_w1)
                    .filter(|(p, _)| keep(p))
                    .map(|(_, r)| r)
                    .collect();
                assert_eq!(rw1.len(), want_w1.len(), "{what}: weight1 rows");
                for (g, (a, b)) in rw1.iter().zip(&want_w1).enumerate() {
                    for (ea, eb) in a.iter().zip(b.iter()) {
                        for k in 0..3 {
                            assert_eq!(ea[k].to_bits(), eb[k].to_bits(), "{what}: weight1 at {g}");
                        }
                    }
                }

                if prune.is_none() {
                    assert_grids_bit_identical(&build_atomic_grid(&mol, &cfg), &got, &what);
                    let quad = atomic_quadrature_points(&mol, &cfg);
                    assert_eq!(quad.len(), got.len(), "{what}: quadrature points");
                    for (q, p) in quad.iter().zip(&got) {
                        assert_eq!(q.0, p.home_atom);
                        assert_eq!(q.1, p.xyz);
                    }
                }
            }
        }
    }

    /// A negative, NaN or infinite cutoff is an error (never silently "off"),
    /// from every fallible entry point, and a panic from the infallible ones.
    #[test]
    fn invalid_weight_cut_is_refused() {
        let mol = h2();
        for bad in [-1e-14, f64::NAN, f64::INFINITY] {
            let cfg = AtomicGridConfig {
                n_radial: 8,
                n_angular: 26,
                weight_cut: Some(bad),
                ..Default::default()
            };
            assert!(weight_cut_threshold(&cfg).is_err(), "{bad}");
            assert!(build_atomic_grid_pruned(&mol, &cfg, None).is_err(), "{bad}");
            assert!(atomic_grid_point_count(&mol, &cfg, None).is_err(), "{bad}");
            assert!(
                build_atomic_grid_with_response_pruned(&mol, &cfg).is_err(),
                "{bad}"
            );
            assert!(std::panic::catch_unwind(|| build_atomic_grid(&mol, &cfg)).is_err());
            assert!(std::panic::catch_unwind(|| atomic_quadrature_points(&mol, &cfg)).is_err());
        }
    }

    /// Benzene at the default 75x110 grid, flat and NWChem-pruned: ORCA's
    /// `WeightCut 1e-14` drops a nonzero number of points, and the count
    /// equals the built length. Prints the fraction dropped.
    #[test]
    fn weight_cut_drops_points_on_benzene() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/molecules/benzene.xyz");
        let mol = Molecule::load_xyz(path.to_str().unwrap()).unwrap();
        for prune in [None, Some(PruneScheme::NwchemLike)] {
            let base = AtomicGridConfig {
                prune,
                ..Default::default()
            };
            let cfg = with_cut(&base, Some(1e-14));
            let n_full = atomic_grid_point_count(&mol, &base, prune).unwrap();
            let grid = build_atomic_grid_pruned(&mol, &cfg, prune).unwrap();
            assert_eq!(
                atomic_grid_point_count(&mol, &cfg, prune).unwrap(),
                grid.len()
            );
            assert!(grid.iter().all(|p| p.weight.abs() >= 1e-14));
            let dropped = n_full - grid.len();
            eprintln!(
                "benzene 75x110 {prune:?}: {n_full} pts, {dropped} dropped at 1e-14 ({:.2}%)",
                100.0 * dropped as f64 / n_full as f64
            );
            assert!(
                dropped > 0,
                "{prune:?}: weight cut 1e-14 dropped nothing on benzene"
            );
        }
    }
}
