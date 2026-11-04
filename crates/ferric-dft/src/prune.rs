//! Radial-shell-dependent angular grid pruning (opt-in).
//!
//! The unpruned Becke-Lebedev grid built by [`crate::grid::build_atomic_grid`]
//! uses the SAME Lebedev order at every radial shell of every atom. That is
//! wasteful at both ends of the radial range: deep in the core the density is
//! essentially spherical (a low-order rule is exact enough), and far in the
//! tail the density is negligible. Only the chemically-active valence shell
//! needs the full angular order.
//!
//! This module mirrors PySCF's `nwchem_prune` (`pyscf/dft/gen_grid.py`), which
//! is PySCF's *default* pruning scheme at grid level 3. Each atom's radial
//! shells are placed into one of five concentric regions by the reduced radius
//! `r / R_bragg`, with element-dependent region boundaries:
//!
//! ```text
//!   H, He     : (0.25,   0.5, 1.0, 4.5)
//!   Li .. Ne  : (0.1667, 0.5, 0.9, 3.5)
//!   Na and up : (0.1,    0.4, 0.8, 2.5)
//! ```
//!
//! and each region gets its own Lebedev order.
//!
//! # Why ferric's region table is not byte-identical to NWChem's
//!
//! For a requested full order `n_ang`, NWChem/PySCF pick the region orders by
//! *index* into the full Lebedev-Laikov ladder
//! `[38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, ...]` as
//! `[1, 3, idx-1, idx, idx-1]`. At ferric's default `n_ang = 110` that gives
//! `[50, 86, 86, 110, 86]`; at `n_ang = 302` it gives `[50, 86, 266, 302, 266]`.
//!
//! [`crate::lebedev::lebedev`] only implements the subset
//! `{6, 14, 26, 50, 110, 194, 302, 434, 590}` — it has no
//! 74/86/146/170/230/266/350 rule. We
//! therefore SNAP each NWChem-requested order onto a supported one. The snap
//! direction was **chosen from measurements, not from taste**:
//!
//! * **Innermost region** snaps down, all the way to order 26. NWChem asks for
//!   50; ferric uses 26. Measured cost on the H2O/CH4 E_xc probes below the
//!   1e-9 Ha level — the density is spherical enough there that even a 26-point
//!   rule is exact for practical purposes. This region is also where most of
//!   the savings live (~23 of 75 radial shells).
//! * **All four outer regions** snap up. In particular the outermost region,
//!   where NWChem asks for 86, uses the full order rather than dropping to 50.
//!
//! The outermost region is the one that matters and the reason is not obvious,
//! so it is recorded here. A first implementation snapped the outermost region
//! *down* (86 -> 50), giving ~30% point savings. That looked fine on H2O
//! (ΔE_xc ≈ 7e-6 Ha) but cost **4.7e-5 Ha on CH4/PBE** — more than double the
//! 2e-5 Ha tolerance the PySCF PBE reference tests in `ferric-scf` gate on.
//! The cause: hydrogen's Bragg radius is small (0.661 Bohr), so hydrogen's
//! outermost region begins at only ~3.06 Bohr, and in CH4 the *neighbouring*
//! hydrogens sit ~3.3 Bohr away — squarely inside a region that had been
//! coarsened to order 50. In other words the "far tail where the density is
//! negligible" assumption is false for hydrogen in a polyatomic molecule.
//! Rounding that region up instead makes the pruned E_xc agree with the flat
//! grid to ~1e-10 Ha on every probe while still saving ~23%.
//!
//! Concretely, at `n_ang = 110` ferric's region table is
//! `[26, 110, 110, 110, 110]`, and at `n_ang = 302` it is
//! `[26, 110, 302, 302, 302]` (434 -> `[26, 110, 434, 434, 434]`, 590 ->
//! `[26, 110, 434, 590, 434]`; point reductions not measured for those two).
//! Measured reductions at 75 radial shells:
//! H2O 23.4%, CH4 23.2%, benzene 22.9%; at 99x302, H2O 33.5%.
//!
//! # Weight correctness
//!
//! [`crate::lebedev::lebedev`] returns weights normalised to `Σ w = 1` for
//! EVERY order. Combined with the radial weights from
//! [`crate::radial::treutler_ahlrichs_m4`] (which already carry the `4π r² dr`
//! Jacobian), the spherical integral `Σ_r Σ_Ω w_r w_Ω f` is correct regardless
//! of which order is used at which shell. Switching order per shell is
//! therefore weight-safe by construction; `prune_weights_are_normalised` in the
//! tests below asserts that property directly for every supported order.
//!
//! # ORCA DefGrid (`grid_prune = "defgrid1" | "defgrid2" | "defgrid3"`)
//!
//! [`PruneScheme::DefGrid`](crate::prune::PruneScheme::DefGrid) gives the SCF XC grid of ORCA 6.1's `DEFGRID1/2/3`
//! (manual, "Numerical Integration", Table 2.48: XC AngularGrid 3 / 4 / 6).
//! The grid FIXES both sizes, so the config's `n_radial` and `n_angular` are
//! ignored (see [`validate_xc_scheme`](crate::prune::validate_xc_scheme)). Per atom:
//!
//! * **Radial shells**: `floor((15·IntAcc − 40) + b·ROW)`, `ROW` = period of
//!   the element (H-He 1, Li-Ne 2, ..., Cs-Rn 6, Fr+ 7). The formula and the
//!   XC IntAcc per AngularGrid (3: 4.159, 4: 4.388, 6: 4.959; Table 2.47) are
//!   from the manual. The manual says `b` was re-optimised per AngularGrid
//!   but does not list it, so `b` here (3: 8.18, 4: 9.57, 6: 9.75) was
//!   FITTED to ORCA 6.1.1's own radial counts: single-atom runs with
//!   `GridPruning Unpruned` (every shell one order) give the shell count as
//!   total points / that order, for H-Rn (rows 1-6). Those counts pin `b` to
//!   [8.154, 8.205), [9.545, 9.590), [9.723, 9.769), and they fix the
//!   rounding as `floor` (nearest-integer rounding fits no single `b` at
//!   AngularGrid 4). Row 7 is the formula's extrapolation, as in ORCA.
//! * **Angular orders**: the five region orders of the atom's AngularGrid
//!   (Table 2.46 = [`SGX_REGION_ORDERS`](crate::prune::SGX_REGION_ORDERS), all on ferric's Lebedev set, so
//!   nothing is snapped). H and He use the AngularGrid one level lower
//!   (`HGridReduced true`, ORCA's default) with the SAME radial count:
//!   ORCA's H atom at DEFGRID2 has 35 shells either way and only the
//!   angular orders drop (6790 = 35·194 points → 3850 = 35·110).
//! * **Region boundaries**: ferric's NWChem `r / R_bragg` boundaries
//!   (`region_alphas`), the same as every other scheme here. The manual says
//!   ORCA's region cutoffs were optimised but does not give them, and ORCA's
//!   default "Adaptive" pruning (which enlarges grids for steep/diffuse
//!   basis functions) is not reproduced: this is the non-adaptive
//!   (`OLDPRUNING`-like) grid. On single atoms ORCA's Adaptive and
//!   OLDPRUNING totals agree for H, C, O, Cl with def2-SVP (not for Br).
//!   Ferric's radial mapping stays Treutler-Ahlrichs M4 (ORCA uses an
//!   optimised M3), so even the shell COUNTS per region differ from ORCA's:
//!   single-atom DefGrid sizes here are 4% to 38% larger than ORCA's
//!   OLDPRUNING ones (e.g. C at DEFGRID2: 7036 vs 6136 points).

use ferric_core::error::FerricError;

use crate::becke::bragg_slater_bohr;

/// Local alias — `ferric_core::error` exposes `FerricError` but no `Result`
/// alias of its own.
type Result<T> = std::result::Result<T, FerricError>;

/// Angular Lebedev orders this crate can actually generate, ascending.
/// Mirrors the `match` arms in [`crate::lebedev::lebedev`].
pub const SUPPORTED_LEBEDEV_ORDERS: [usize; 9] = [6, 14, 26, 50, 110, 194, 302, 434, 590];

/// The full Lebedev-Laikov ladder PySCF indexes into (`LEBEDEV_NGRID[4:18]`, through 590).
/// Only used to reproduce NWChem's *requested* per-region orders before
/// snapping them onto [`SUPPORTED_LEBEDEV_ORDERS`].
const NWCHEM_LADDER: [usize; 14] = [
    38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590,
];

/// Lebedev order used in the innermost region (`r/R_bragg` below the first
/// NWChem boundary). NWChem asks for 50 there; ferric drops to 26 because the
/// measured cost is nil (see the module docs) and this is where the bulk of
/// the savings comes from — the innermost region holds ~23 of 75 radial
/// shells on a typical atom.
const INNER_CORE_ORDER: usize = 26;

/// Which angular-pruning scheme to apply when building an atomic grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneScheme {
    /// NWChem-style 5-region radial pruning, as used by PySCF's default
    /// `nwchem_prune`, with orders snapped onto ferric's supported Lebedev
    /// set (see the module docs for the snapping rule).
    NwchemLike,
    /// The seminumerical-exchange (COSX/SGX) scheme of ORCA's `GridX` grids
    /// as transcribed by PySCF's `sgx_prune` (`pyscf/dft/gen_grid.py`,
    /// `SGX_ANG_MAPPING`, v2.13-2.14): the same five NWChem regions
    /// (`r / R_bragg` against the element-row boundaries above), but each
    /// region gets a fixed Lebedev order from one row of [`SGX_REGION_ORDERS`],
    /// selected by the grid's peak order `n_angular`. At `n_angular = 194`
    /// the regions are `26/50/110/194/110` — ORCA 6 `GridX3` (its DefGrid2
    /// FINAL grid) and PySCF's SGX level 2. Every order in the table is
    /// tabulated, so no snapping happens.
    Sgx,
    /// [`NwchemLike`](Self::NwchemLike)'s angular table plus fewer radial
    /// shells for light elements ([`radial_shells`]): H and He get
    /// `round(n_radial · 50/75)`, Li-Ne `round(n_radial · 66/75)`, Na and
    /// heavier the full `n_radial` (50 / 66 / 75 at the default 75).
    /// Measured against the flat 75x110 grid (def2-SVP): live PBE SCFs moved
    /// by -1.8e-7 Ha on n-C20H42 and -4.3e-6 Ha on danuglipron (71 atoms,
    /// C/H/N/O/F), against the flat grid's own 4.5e-4 to 3.3e-3 Ha error
    /// relative to a 99x590 reference on the alkanes; iteration counts
    /// unchanged; C20 wall time 17.7 -> 13.6 s. Li-Ne at 60 shells moved
    /// danuglipron by 9.2e-6 Ha, and 50 shells for Cl and S cost 4e-5 Ha on
    /// CCl4 (fixed-density E_xc probe), so second-row atoms keep the full grid.
    NwchemRadial,
    /// ORCA 6.1's `DEFGRID1/2/3` SCF XC grid (see the module docs): a fixed
    /// per-element radial count and ORCA's five region orders, independent
    /// of the config's `n_radial` / `n_angular`.
    DefGrid(DefGridLevel),
}

/// Which ORCA `DEFGRID` an [`PruneScheme::DefGrid`] grid reproduces. A
/// closed enum (rather than a bare integer) so an invalid level cannot be
/// constructed and [`radial_shells`] needs no error path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefGridLevel {
    /// `DEFGRID1`: XC AngularGrid 3.
    One,
    /// `DEFGRID2` (ORCA's default): XC AngularGrid 4.
    Two,
    /// `DEFGRID3`: XC AngularGrid 6.
    Three,
}

impl DefGridLevel {
    /// ORCA AngularGrid of the SCF XC grid (manual Table 2.48) for
    /// heavy atoms; H and He use one less ([`defgrid_angular_grid`]).
    pub fn xc_angular_grid(self) -> usize {
        match self {
            Self::One => 3,
            Self::Two => 4,
            Self::Three => 6,
        }
    }

    /// `(IntAcc, b)` of the radial formula for this level's AngularGrid:
    /// IntAcc from manual Table 2.47, `b` fitted to ORCA 6.1.1 (module docs).
    fn radial_params(self) -> (f64, f64) {
        match self {
            Self::One => (4.159, 8.18),
            Self::Two => (4.388, 9.57),
            Self::Three => (4.959, 9.75),
        }
    }
}

/// Periodic-table row (period) of element `z`, as ORCA's radial formula
/// uses it; 7 for everything past radon.
fn periodic_row(z: i32) -> usize {
    match z {
        i32::MIN..=2 => 1,
        3..=10 => 2,
        11..=18 => 3,
        19..=36 => 4,
        37..=54 => 5,
        55..=86 => 6,
        _ => 7,
    }
}

/// Radial shells of element `z` on ORCA's `DEFGRID` `level`:
/// `floor((15·IntAcc − 40) + b·ROW)`. The same for H/He as for the
/// unreduced level (`HGridReduced` reduces only the angular grid).
pub fn defgrid_radial_shells(z: i32, level: DefGridLevel) -> usize {
    let (int_acc, b) = level.radial_params();
    let n = (15.0 * int_acc - 40.0) + b * periodic_row(z) as f64;
    // n > 20 for every row and level, so the cast cannot truncate a negative.
    n.floor() as usize
}

/// ORCA AngularGrid used for element `z` on `DEFGRID` `level`: the level's
/// XC AngularGrid, one less for H and He (`HGridReduced true`).
pub fn defgrid_angular_grid(z: i32, level: DefGridLevel) -> usize {
    let ag = level.xc_angular_grid();
    if z <= 2 {
        ag - 1
    } else {
        ag
    }
}

/// Region orders (innermost -> outermost) of element `z` on `DEFGRID`
/// `level`: row `AngularGrid − 1` of [`SGX_REGION_ORDERS`] (= ORCA manual
/// Table 2.46).
pub fn defgrid_region_orders(z: i32, level: DefGridLevel) -> [usize; 5] {
    SGX_REGION_ORDERS[defgrid_angular_grid(z, level) - 1]
}

/// Radial shells for an atom of nuclear charge `z` on a grid of `n_radial`
/// shells under `prune` (see [`PruneScheme::NwchemRadial`]);
/// [`PruneScheme::DefGrid`] ignores `n_radial` ([`defgrid_radial_shells`]);
/// every other scheme, and the flat grid, use `n_radial` for every atom.
/// At least 1.
pub fn radial_shells(z: i32, n_radial: usize, prune: Option<PruneScheme>) -> usize {
    let scaled = |num: usize| ((n_radial * num + 37) / 75).max(1);
    match prune {
        Some(PruneScheme::NwchemRadial) if z <= 2 => scaled(50),
        Some(PruneScheme::NwchemRadial) if z <= 10 => scaled(66),
        Some(PruneScheme::DefGrid(level)) => defgrid_radial_shells(z, level),
        _ => n_radial,
    }
}

/// Check that `scheme` can build the main XC grid at peak order `n_angular`.
///
/// * `NwchemLike` / `NwchemRadial`: `n_angular` must have a
///   [`region_orders`] table.
/// * `DefGrid`: always `Ok`; the scheme fixes its own sizes and ignores
///   `n_angular` (and `n_radial`). Refusing an explicitly-set
///   `grid_radial` / `grid_angular` alongside it is the config layer's job,
///   since only it can tell "set" from "default".
/// * `Sgx`: an error; it is the COSX exchange grid's scheme.
pub fn validate_xc_scheme(scheme: PruneScheme, n_angular: usize) -> Result<()> {
    match scheme {
        PruneScheme::NwchemLike | PruneScheme::NwchemRadial => region_orders(n_angular).map(|_| ()),
        PruneScheme::DefGrid(_) => Ok(()),
        PruneScheme::Sgx => Err(FerricError::General(
            "grid pruning 'sgx' is the COSX exchange grid's scheme; the XC grid takes \
             'none', 'nwchem', 'nwchem-radial' or 'defgrid1'/'defgrid2'/'defgrid3'"
                .to_string(),
        )),
    }
}

/// Region orders (innermost -> outermost) of [`PruneScheme::Sgx`], one row
/// per peak order. PySCF's `SGX_ANG_MAPPING` lists Lebedev DEGREES
/// (`[5,7,11,11,7]`, `[5,7,11,17,11]`, `[7,11,17,23,17]`, `[7,17,23,29,23]`,
/// `[7,23,29,35,29]`, `[11,29,35,41,35]`); these are the matching point
/// counts (degree 5/7/11/17/23/29/35/41 = 14/26/50/110/194/302/434/590).
/// ORCA 6.1's manual gives the same per-region orders for AngularGrid 1-3
/// (14/26/50/50/26, 14/26/50/110/50, 26/50/110/194/110). PySCF's last row
/// (peak 770) is omitted: ferric has no 770-point rule.
pub const SGX_REGION_ORDERS: [[usize; 5]; 6] = [
    [14, 26, 50, 50, 26],
    [14, 26, 50, 110, 50],
    [26, 50, 110, 194, 110],
    [26, 110, 194, 302, 194],
    [26, 194, 302, 434, 302],
    [50, 302, 434, 590, 434],
];

/// The [`PruneScheme::Sgx`] row whose peak (region 3) order is `n_ang`.
pub fn sgx_region_orders(n_ang: usize) -> Result<[usize; 5]> {
    SGX_REGION_ORDERS
        .iter()
        .find(|row| row[3] == n_ang)
        .copied()
        .ok_or_else(|| {
            FerricError::General(format!(
                "grid pruning 'sgx': n_angular = {n_ang} is not the peak order of an SGX/ORCA \
                 GridX row (allowed: 50, 110, 194, 302, 434, 590)"
            ))
        })
}

impl PruneScheme {
    /// Strict string parser for config plumbing. Unknown values are a hard
    /// error, never a silent default (repo "config honesty" convention).
    pub fn parse_config_str(s: &str) -> Result<Option<Self>> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "flat" => Ok(None),
            "nwchem" | "nwchem-like" | "nwchem_like" => Ok(Some(Self::NwchemLike)),
            "nwchem-radial" | "nwchem_radial" => Ok(Some(Self::NwchemRadial)),
            "sgx" => Ok(Some(Self::Sgx)),
            "defgrid1" => Ok(Some(Self::DefGrid(DefGridLevel::One))),
            "defgrid2" => Ok(Some(Self::DefGrid(DefGridLevel::Two))),
            "defgrid3" => Ok(Some(Self::DefGrid(DefGridLevel::Three))),
            other => Err(FerricError::General(format!(
                "unknown grid prune scheme '{other}' (expected 'none', 'nwchem', \
                 'nwchem-radial', 'sgx', 'defgrid1', 'defgrid2' or 'defgrid3')"
            ))),
        }
    }
}

/// Snap `requested` onto the nearest supported Lebedev order, breaking ties
/// and rounding in the direction given by `round_up`.
///
/// `round_up = true` returns the smallest supported order `>= requested`
/// (falling back to the largest supported order if `requested` exceeds it);
/// `round_up = false` returns the largest supported order `<= requested`
/// (falling back to the smallest supported order).
fn snap_order(requested: usize, round_up: bool) -> usize {
    if SUPPORTED_LEBEDEV_ORDERS.contains(&requested) {
        return requested;
    }
    if round_up {
        SUPPORTED_LEBEDEV_ORDERS
            .iter()
            .copied()
            .find(|&o| o >= requested)
            .unwrap_or(SUPPORTED_LEBEDEV_ORDERS[SUPPORTED_LEBEDEV_ORDERS.len() - 1])
    } else {
        SUPPORTED_LEBEDEV_ORDERS
            .iter()
            .copied()
            .rev()
            .find(|&o| o <= requested)
            .unwrap_or(SUPPORTED_LEBEDEV_ORDERS[0])
    }
}

/// The five per-region Lebedev orders for a full angular order `n_ang`,
/// ordered innermost -> outermost.
///
/// Errors if `n_ang` is not one of [`SUPPORTED_LEBEDEV_ORDERS`] (the caller
/// would not be able to build the unpruned grid either), following the repo
/// convention that unsupported config values are a hard error.
pub fn region_orders(n_ang: usize) -> Result<[usize; 5]> {
    if !SUPPORTED_LEBEDEV_ORDERS.contains(&n_ang) {
        return Err(FerricError::General(format!(
            "grid pruning: unsupported angular order {n_ang} \
             (supported: {SUPPORTED_LEBEDEV_ORDERS:?})"
        )));
    }
    // NWChem leaves very small grids alone.
    if n_ang < 50 {
        return Ok([n_ang; 5]);
    }
    if n_ang == 50 {
        // PySCF special-cases n_ang == 50 to ladder indices [1, 2, 2, 2, 1]
        // = [50, 74, 74, 74, 50]. Snapped: inner/outer down to 50, middle up
        // to 110 -- which would make the "pruned" grid *larger* than the flat
        // one. Refuse rather than silently pessimise.
        return Err(FerricError::General(
            "grid pruning: n_angular = 50 has no useful pruned table on ferric's \
             Lebedev set (NWChem's middle regions want order 74, which ferric \
             would have to snap up to 110, enlarging the grid). Use n_angular \
             = 110 or 302, or disable pruning."
                .to_string(),
        ));
    }
    let idx = NWCHEM_LADDER
        .iter()
        .position(|&o| o == n_ang)
        .ok_or_else(|| {
            FerricError::General(format!(
                "grid pruning: angular order {n_ang} is not on the Lebedev-Laikov \
                 ladder used by the NWChem prune table"
            ))
        })?;
    // NWChem region ladder indices: [1, 3, idx-1, idx, idx-1].
    let requested = [
        NWCHEM_LADDER[1],
        NWCHEM_LADDER[3],
        NWCHEM_LADDER[idx - 1],
        NWCHEM_LADDER[idx],
        NWCHEM_LADDER[idx - 1],
    ];
    // Innermost region: snap DOWN, and further to ferric's order-26 rule.
    // Regions 1..4 (everything from the core shoulder outwards, including the
    // tail): snap UP. See the module docs for the measurements that forced
    // the outermost region to round up rather than down.
    Ok([
        INNER_CORE_ORDER,
        snap_order(requested[1], true),
        snap_order(requested[2], true),
        snap_order(requested[3], true),
        snap_order(requested[4], true),
    ])
}

/// NWChem reduced-radius region boundaries for element `z`.
fn region_alphas(z: i32) -> [f64; 4] {
    if z <= 2 {
        [0.25, 0.5, 1.0, 4.5]
    } else if z <= 10 {
        [0.1667, 0.5, 0.9, 3.5]
    } else {
        [0.1, 0.4, 0.8, 2.5]
    }
}

/// Per-radial-shell Lebedev order for atom `z` at radii `rs`, under `scheme`.
///
/// Returns a vector the same length as `rs`. Every returned order is in
/// [`SUPPORTED_LEBEDEV_ORDERS`], so [`crate::lebedev::lebedev`] cannot panic on
/// the result.
pub fn angular_orders_for_atom(
    z: i32,
    rs: &[f64],
    n_ang: usize,
    scheme: PruneScheme,
) -> Result<Vec<usize>> {
    let table = match scheme {
        PruneScheme::NwchemLike | PruneScheme::NwchemRadial => region_orders(n_ang)?,
        PruneScheme::Sgx => sgx_region_orders(n_ang)?,
        // DefGrid ignores `n_ang`: the level fixes the orders.
        PruneScheme::DefGrid(level) => defgrid_region_orders(z, level),
    };
    let alphas = region_alphas(z);
    // bragg_slater_bohr is strictly positive for every Z (fallback 1.0 Angstrom).
    let r_atom = bragg_slater_bohr(z);
    Ok(rs
        .iter()
        .map(|&r| {
            let x = r / r_atom;
            let region = alphas.iter().filter(|&&a| x > a).count();
            table[region]
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lebedev::lebedev;
    use crate::radial::treutler_ahlrichs_m4;

    #[test]
    fn region_table_at_110_matches_documented_snap() {
        // NWChem wants [50, 86, 86, 110, 86]; ferric drops the core to 26 and
        // rounds every outer region up -> [26, 110, 110, 110, 110].
        assert_eq!(region_orders(110).unwrap(), [26, 110, 110, 110, 110]);
    }

    #[test]
    fn region_table_at_302_matches_documented_snap() {
        // NWChem wants [50, 86, 266, 302, 266]; snapped -> [26, 110, 302, 302, 302].
        assert_eq!(region_orders(302).unwrap(), [26, 110, 302, 302, 302]);
    }

    #[test]
    fn region_table_at_434_and_590_snaps_onto_supported_orders() {
        // NWChem wants [50, 86, 350, 434, 350]; 350 has no ferric rule and
        // snaps UP to 434.
        assert_eq!(region_orders(434).unwrap(), [26, 110, 434, 434, 434]);
        // NWChem wants [50, 86, 434, 590, 434]; all of those are supported.
        assert_eq!(region_orders(590).unwrap(), [26, 110, 434, 590, 434]);
    }

    #[test]
    fn region_table_at_194_uses_the_new_rule() {
        // NWChem wants [50, 86, 170, 194, 170]; 170 has no ferric rule and
        // snaps UP to 194.
        assert_eq!(region_orders(194).unwrap(), [26, 110, 194, 194, 194]);
    }

    /// The SGX rows are PySCF's `SGX_ANG_MAPPING` (Lebedev degrees) in point
    /// counts; level 2 (peak 194) is ORCA GridX3's 26/50/110/194/110.
    #[test]
    fn sgx_rows_match_pyscf_and_orca() {
        assert_eq!(sgx_region_orders(194).unwrap(), [26, 50, 110, 194, 110]);
        assert_eq!(sgx_region_orders(110).unwrap(), [14, 26, 50, 110, 50]);
        assert_eq!(sgx_region_orders(302).unwrap(), [26, 110, 194, 302, 194]);
        assert!(sgx_region_orders(146).is_err());
        assert!(sgx_region_orders(26).is_err());
        for row in SGX_REGION_ORDERS {
            for o in row {
                assert!(SUPPORTED_LEBEDEV_ORDERS.contains(&o), "{o}");
            }
            assert_eq!(row.iter().max(), Some(&row[3]), "peak must be region 3");
        }
        // Region assignment for carbon (Li-Ne boundaries 0.1667/0.5/0.9/3.5):
        // x = 0.8 -> region 2, 2.0 -> 3, 0.05 -> 0, 4.0 -> 4.
        let r_c = crate::becke::bragg_slater_bohr(6);
        let o = angular_orders_for_atom(
            6,
            &[0.8 * r_c, 2.0 * r_c, 0.05 * r_c, 4.0 * r_c],
            194,
            PruneScheme::Sgx,
        )
        .unwrap();
        assert_eq!(o, vec![110, 194, 26, 110]);
    }

    #[test]
    fn unsupported_orders_are_hard_errors_not_silent_defaults() {
        // Not a ferric-supported Lebedev order at all.
        assert!(region_orders(86).is_err());
        assert!(region_orders(146).is_err());
        // Supported by lebedev() but has no useful pruned table.
        assert!(region_orders(50).is_err());
        // Small orders are left flat rather than errored (NWChem behaviour).
        assert_eq!(region_orders(26).unwrap(), [26; 5]);
    }

    #[test]
    fn every_pruned_order_is_generatable() {
        // The whole point of snapping: lebedev() must never be handed an
        // order it cannot build. Panics here would be a live bug.
        for n_ang in [110, 302, 434, 590] {
            for z in [1, 6, 8, 16, 26] {
                let (rs, _) = treutler_ahlrichs_m4(z, 75);
                let orders =
                    angular_orders_for_atom(z, &rs, n_ang, PruneScheme::NwchemLike).unwrap();
                assert_eq!(orders.len(), rs.len());
                for o in orders {
                    assert!(
                        SUPPORTED_LEBEDEV_ORDERS.contains(&o),
                        "order {o} not supported"
                    );
                    let (pts, _) = lebedev(o);
                    assert_eq!(pts.len(), o);
                }
            }
        }
    }

    #[test]
    fn prune_weights_are_normalised() {
        // THE weight-correctness invariant: switching Lebedev order per radial
        // shell is only safe because every order's weights sum to exactly 1.
        // If this ever regresses, pruned grids silently mis-integrate.
        for &o in SUPPORTED_LEBEDEV_ORDERS.iter() {
            let (_, w) = lebedev(o);
            let s: f64 = w.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-13,
                "Lebedev order {o} weights sum to {s}, not 1"
            );
        }
    }

    #[test]
    fn orders_are_monotone_within_the_core_and_never_exceed_full_order() {
        // No shell may ever be assigned a HIGHER order than the unpruned grid
        // would use -- pruning must only ever remove points.
        for n_ang in [110, 302, 434, 590] {
            for z in [1, 8, 17] {
                let (rs, _) = treutler_ahlrichs_m4(z, 75);
                let orders =
                    angular_orders_for_atom(z, &rs, n_ang, PruneScheme::NwchemLike).unwrap();
                for &o in &orders {
                    assert!(o <= n_ang, "pruned order {o} exceeds full order {n_ang}");
                }
                // The innermost shell must be coarse and the peak must be the
                // full order somewhere in the middle.
                assert!(orders[0] < n_ang);
                assert!(orders.contains(&n_ang));
            }
        }
    }

    /// Catches: a scheme that thins the radial grid of every element, of
    /// second-row atoms, or of the flat grid, and a rounding that drops a
    /// light atom to zero shells.
    #[test]
    fn nwchem_radial_thins_only_light_elements() {
        let r = |z, n| radial_shells(z, n, Some(PruneScheme::NwchemRadial));
        assert_eq!(
            [r(1, 75), r(2, 75), r(3, 75), r(8, 75), r(10, 75)],
            [50, 50, 66, 66, 66]
        );
        assert_eq!(
            [r(11, 75), r(16, 75), r(17, 75), r(35, 75)],
            [75, 75, 75, 75]
        );
        assert_eq!([r(1, 99), r(6, 99), r(17, 99)], [66, 87, 99]);
        assert_eq!(r(1, 1), 1);
        for p in [None, Some(PruneScheme::NwchemLike), Some(PruneScheme::Sgx)] {
            assert_eq!(radial_shells(1, 75, p), 75, "{p:?}");
        }
        assert_eq!(
            PruneScheme::parse_config_str("nwchem-radial").unwrap(),
            Some(PruneScheme::NwchemRadial)
        );
    }

    #[test]
    fn parse_config_str_is_strict() {
        assert_eq!(PruneScheme::parse_config_str("none").unwrap(), None);
        assert_eq!(
            PruneScheme::parse_config_str("nwchem").unwrap(),
            Some(PruneScheme::NwchemLike)
        );
        assert_eq!(
            PruneScheme::parse_config_str("sgx").unwrap(),
            Some(PruneScheme::Sgx)
        );
        assert!(PruneScheme::parse_config_str("sg1").is_err());
        assert!(PruneScheme::parse_config_str("").is_err());
    }

    const LEVELS: [DefGridLevel; 3] = [DefGridLevel::One, DefGridLevel::Two, DefGridLevel::Three];

    /// Catches: a wrong IntAcc, `b`, row assignment or rounding in the DefGrid
    /// radial formula. Expected counts are MEASURED, not computed: ORCA 6.1.1
    /// single-atom UKS/def2-SVP runs at `!DEFGRIDn` with `GridPruning
    /// Unpruned`, `HGridReduced false`, `BFCut 1e-300`, `WeightCut 1e-300`;
    /// shells = total points / the unpruned grid's single order (110 / 194 /
    /// 434 at DEFGRID1 / 2 / 3), e.g. C at DEFGRID2: 8536 = 44 x 194. Every
    /// element of a row gave the same count (H-He, Li-Ne, Na-Ar, K/Fe/Br/Kr,
    /// I/Xe, Cs/Au/Rn measured). Row 7 is the formula's extrapolation only.
    #[test]
    fn defgrid_radial_counts_match_orca_6_1_1() {
        // (elements of the row probed, [DEFGRID1, DEFGRID2, DEFGRID3] shells).
        let measured: [(&[i32], [usize; 3]); 6] = [
            (&[1, 2], [30, 35, 44]),
            (&[3, 6, 7, 8, 9, 10], [38, 44, 53]),
            (&[11, 14, 16, 17, 18], [46, 54, 63]),
            (&[19, 26, 35, 36], [55, 64, 73]),
            (&[53, 54], [63, 73, 83]),
            (&[55, 79, 86], [71, 83, 92]),
        ];
        for (zs, want) in measured {
            for &z in zs {
                for (level, &n) in LEVELS.iter().zip(want.iter()) {
                    assert_eq!(defgrid_radial_shells(z, *level), n, "Z={z} {level:?}");
                    // radial_shells must route the scheme there and ignore n_radial.
                    for n_radial in [1, 75, 99] {
                        assert_eq!(
                            radial_shells(z, n_radial, Some(PruneScheme::DefGrid(*level))),
                            n
                        );
                    }
                }
            }
        }
        // Row 7 (Fr, Z = 87): floor(22.385 + 7*8.18), floor(25.82 + 7*9.57),
        // floor(34.385 + 7*9.75).
        let fr: Vec<usize> = LEVELS
            .iter()
            .map(|&l| defgrid_radial_shells(87, l))
            .collect();
        assert_eq!(fr, vec![79, 92, 102]);
    }

    /// Catches: the wrong AngularGrid per DefGrid level, a missing H/He
    /// reduction, or a reduction applied to heavy atoms. Tables are ORCA 6.1
    /// manual Table 2.46 (AngularGrid 2/3/4/5/6), levels Table 2.48 (XC
    /// AngularGrid 3/4/6), H/He one level lower (`HGridReduced true`).
    #[test]
    fn defgrid_region_tables_per_level() {
        let heavy = [
            [26, 50, 110, 194, 110],  // DEFGRID1 = AngularGrid 3
            [26, 110, 194, 302, 194], // DEFGRID2 = AngularGrid 4
            [50, 302, 434, 590, 434], // DEFGRID3 = AngularGrid 6
        ];
        let light = [
            [14, 26, 50, 110, 50],    // AngularGrid 2
            [26, 50, 110, 194, 110],  // AngularGrid 3
            [26, 194, 302, 434, 302], // AngularGrid 5
        ];
        for (i, level) in LEVELS.iter().enumerate() {
            for z in [1, 2] {
                assert_eq!(
                    defgrid_region_orders(z, *level),
                    light[i],
                    "Z={z} {level:?}"
                );
            }
            for z in [3, 6, 17, 35, 86] {
                assert_eq!(
                    defgrid_region_orders(z, *level),
                    heavy[i],
                    "Z={z} {level:?}"
                );
            }
        }
        // angular_orders_for_atom ignores n_ang and uses the NWChem regions:
        // carbon (Li-Ne boundaries 0.1667/0.5/0.9/3.5) at x = 0.05, 0.3, 0.8,
        // 2.0, 4.0 -> regions 0..4 of AngularGrid 4.
        let r_c = crate::becke::bragg_slater_bohr(6);
        let rs: Vec<f64> = [0.05, 0.3, 0.8, 2.0, 4.0].iter().map(|x| x * r_c).collect();
        for n_ang in [26, 110, 590] {
            let o = angular_orders_for_atom(6, &rs, n_ang, PruneScheme::DefGrid(DefGridLevel::Two))
                .unwrap();
            assert_eq!(o, vec![26, 110, 194, 302, 194]);
        }
        // Every order a DefGrid atom can get is generatable.
        for level in LEVELS {
            for z in [1, 2, 6, 17, 35, 54, 86, 87] {
                let (rs, _) = treutler_ahlrichs_m4(z, defgrid_radial_shells(z, level));
                for o in angular_orders_for_atom(z, &rs, 110, PruneScheme::DefGrid(level)).unwrap()
                {
                    assert!(SUPPORTED_LEBEDEV_ORDERS.contains(&o), "{o}");
                }
            }
        }
    }

    #[test]
    fn defgrid_parse_round_trips_and_rejects_other_levels() {
        for (s, level) in [
            ("defgrid1", DefGridLevel::One),
            ("defgrid2", DefGridLevel::Two),
            ("DefGrid3", DefGridLevel::Three),
            (" DEFGRID2 ", DefGridLevel::Two),
        ] {
            assert_eq!(
                PruneScheme::parse_config_str(s).unwrap(),
                Some(PruneScheme::DefGrid(level)),
                "{s}"
            );
        }
        for bad in ["defgrid0", "defgrid4", "defgrid", "defgrid-2", "defgrid 2"] {
            assert!(PruneScheme::parse_config_str(bad).is_err(), "{bad}");
        }
    }

    /// Catches: `validate_xc_scheme` letting the COSX scheme through, refusing
    /// DefGrid over an `n_angular` it ignores, or skipping the NWChem table
    /// check.
    #[test]
    fn validate_xc_scheme_accepts_and_rejects() {
        for level in LEVELS {
            for n_ang in [50, 110, 302, 590, 7] {
                assert!(validate_xc_scheme(PruneScheme::DefGrid(level), n_ang).is_ok());
            }
        }
        assert!(validate_xc_scheme(PruneScheme::NwchemLike, 110).is_ok());
        assert!(validate_xc_scheme(PruneScheme::NwchemRadial, 302).is_ok());
        assert!(validate_xc_scheme(PruneScheme::NwchemLike, 50).is_err());
        assert!(validate_xc_scheme(PruneScheme::NwchemRadial, 86).is_err());
        assert!(validate_xc_scheme(PruneScheme::Sgx, 194).is_err());
    }
}
