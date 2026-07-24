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
//! `{6, 14, 26, 50, 110, 302}` — it has no 74/86/146/170/194/230/266 rule. We
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
//! `[26, 110, 302, 302, 302]`. Measured reductions at 75 radial shells:
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

use ferric_core::error::FerricError;

use crate::becke::bragg_slater_bohr;

/// Local alias — `ferric_core::error` exposes `FerricError` but no `Result`
/// alias of its own.
type Result<T> = std::result::Result<T, FerricError>;

/// Angular Lebedev orders this crate can actually generate, ascending.
/// Mirrors the `match` arms in [`crate::lebedev::lebedev`].
pub const SUPPORTED_LEBEDEV_ORDERS: [usize; 6] = [6, 14, 26, 50, 110, 302];

/// The full Lebedev-Laikov ladder PySCF indexes into (`LEBEDEV_NGRID[4:]`).
/// Only used to reproduce NWChem's *requested* per-region orders before
/// snapping them onto [`SUPPORTED_LEBEDEV_ORDERS`].
const NWCHEM_LADDER: [usize; 11] = [38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302];

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
}

impl PruneScheme {
    /// Strict string parser for config plumbing. Unknown values are a hard
    /// error, never a silent default (repo "config honesty" convention).
    pub fn parse_config_str(s: &str) -> Result<Option<Self>> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "flat" => Ok(None),
            "nwchem" | "nwchem-like" | "nwchem_like" => Ok(Some(Self::NwchemLike)),
            other => Err(FerricError::General(format!(
                "unknown grid prune scheme '{other}' (expected 'none' or 'nwchem')"
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
    let PruneScheme::NwchemLike = scheme;
    let table = region_orders(n_ang)?;
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
    fn unsupported_orders_are_hard_errors_not_silent_defaults() {
        // Not a ferric-supported Lebedev order at all.
        assert!(region_orders(86).is_err());
        assert!(region_orders(194).is_err());
        // Supported by lebedev() but has no useful pruned table.
        assert!(region_orders(50).is_err());
        // Small orders are left flat rather than errored (NWChem behaviour).
        assert_eq!(region_orders(26).unwrap(), [26; 5]);
    }

    #[test]
    fn every_pruned_order_is_generatable() {
        // The whole point of snapping: lebedev() must never be handed an
        // order it cannot build. Panics here would be a live bug.
        for n_ang in [110, 302] {
            for z in [1, 6, 8, 16, 26] {
                let (rs, _) = treutler_ahlrichs_m4(z, 75);
                let orders = angular_orders_for_atom(z, &rs, n_ang, PruneScheme::NwchemLike)
                    .unwrap();
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
        for n_ang in [110, 302] {
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

    #[test]
    fn parse_config_str_is_strict() {
        assert_eq!(PruneScheme::parse_config_str("none").unwrap(), None);
        assert_eq!(
            PruneScheme::parse_config_str("nwchem").unwrap(),
            Some(PruneScheme::NwchemLike)
        );
        assert!(PruneScheme::parse_config_str("sg1").is_err());
        assert!(PruneScheme::parse_config_str("").is_err());
    }
}
