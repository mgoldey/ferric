//! MWE: the closed-shell CCSD plan charges each ladder route for what that
//! route allocates, and sizes the RI block from the same plan.
//!
//! # Why this tests the plan and not a CCSD run
//!
//! Same trap as `mwe_ccsd_charges_both_vvvv_blocks.rs`: at any shape small
//! enough to execute, `eri3_ao` sets every accept/refuse threshold and the
//! `nv⁴` and ladder-scratch terms are inert. So the arithmetic is exercised
//! through `cs_ccsd_memory_plan` / `plan_ladder` at a production shape, by
//! bisecting for the smallest accepted budget rather than re-deriving the sum.

use ferric_cc::ccsd_closed_shell::{cs_ccsd_memory_plan, plan_ladder, CsCcsdShape, LadderRoute};
use ferric_cc::{CcConfig, CcsdLadder};

/// benzene/aug-cc-pVTZ-scale, the fixture the spin-orbital plan test uses.
const NO: usize = 21;
const NV: usize = 393;
const NBAS: usize = 414;
const NAUX: usize = 1512;
const F64: usize = 8;

fn shape(diis_subspace: usize) -> CsCcsdShape {
    CsCcsdShape {
        no: NO,
        nv: NV,
        nbas: NBAS,
        naux: NAUX,
        diis_subspace,
    }
}

fn nv4_bytes() -> usize {
    NV.pow(4) * F64
}

/// Smallest budget a route's plan accepts, by bisection on `check()`.
fn smallest_accepted(route: LadderRoute, diis_subspace: usize) -> usize {
    let ok = |b: usize| {
        cs_ccsd_memory_plan(shape(diis_subspace), route, b)
            .check()
            .is_ok()
    };
    let (mut lo, mut hi) = (0usize, usize::MAX / 4);
    assert!(ok(hi), "fixture is broken: an enormous budget was refused");
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if ok(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

fn cfg(ladder: CcsdLadder) -> CcConfig {
    CcConfig {
        ladder,
        ..Default::default()
    }
}

/// CONTRACT 1: the dense route charges the resident `vvvv` AND the three
/// `nv⁴` temporaries of the `wvvvv` build — four blocks at the peak.
#[test]
fn dense_route_charges_four_nv4_blocks() {
    let t = smallest_accepted(LadderRoute::Dense, 6);
    assert!(
        t >= 4 * nv4_bytes(),
        "dense plan accepts {t} bytes but vvvv plus the a1/a2/a3 build is 4 x {}",
        nv4_bytes()
    );
}

/// CONTRACT 2: the RI route charges NO `nv⁴` block. Even at the widest
/// block (`blk = nv`, whose panel is exactly one `nv⁴`) it must need less
/// than the dense route, and at `blk = 1` less than a single `nv⁴` block.
#[test]
fn ri_route_charges_no_nv4_block() {
    let narrow = smallest_accepted(LadderRoute::Ri { block: 1 }, 6);
    let wide = smallest_accepted(LadderRoute::Ri { block: NV }, 6);
    let dense = smallest_accepted(LadderRoute::Dense, 6);
    assert!(
        narrow < nv4_bytes(),
        "RI at blk=1 demands {narrow} bytes, at or above one nv^4 block ({})",
        nv4_bytes()
    );
    assert!(
        wide < dense,
        "RI at blk=nv ({wide}) must undercut dense ({dense})"
    );
}

/// CONTRACT 3: the `blk·nv³` panel is charged per unit of width. At this
/// shape the ladder scratch is the largest transient at every width, so the
/// threshold moves by exactly one `nv³` panel per unit of `blk`.
#[test]
fn ri_panel_is_charged_per_unit_of_block_width() {
    let nv3 = NV.pow(3) * F64;
    let t1 = smallest_accepted(LadderRoute::Ri { block: 1 }, 6);
    let t2 = smallest_accepted(LadderRoute::Ri { block: 2 }, 6);
    let tn = smallest_accepted(LadderRoute::Ri { block: NV }, 6);
    assert_eq!(
        t2 - t1,
        nv3,
        "one more unit of width must cost one nv^3 panel"
    );
    assert_eq!(
        tn - t1,
        (NV - 1) * nv3,
        "width nv must cost nv-1 more panels than width 1"
    );
}

/// CONTRACT 3b: the block-independent RI scratch is charged in full —
/// `Bx` and `L` (`2·naux·nv²`), the `nv³` transpose slab and the `no²·nv²`
/// output `Z`. Both routes share every other reservation, so the RI
/// threshold minus the dense threshold's shared part (dense threshold less
/// its four `nv⁴` blocks, which dominate `eri3_ao` here) is exactly the RI
/// scratch. Halving or dropping any fixed term moves this difference.
#[test]
fn ri_fixed_scratch_terms_are_charged_exactly() {
    let shared = smallest_accepted(LadderRoute::Dense, 6) - 4 * nv4_bytes();
    for blk in [1usize, 7] {
        let ri = smallest_accepted(LadderRoute::Ri { block: blk }, 6);
        let want = (2 * NAUX * NV * NV + blk * NV.pow(3) + NV.pow(3) + NO * NO * NV * NV) * F64;
        assert_eq!(ri - shared, want, "blk={blk}: RI scratch charge");
    }
}

/// CONTRACT 4: `plan_ladder` sizes the RI block from the plan — at the exact
/// threshold of width `k` it picks `k`, one byte under it picks `k - 1`, and
/// the plan it returns passes `check()` at that budget.
#[test]
fn ri_block_width_is_the_widest_the_plan_fits() {
    for k in [2usize, 17, 200, NV] {
        let at = smallest_accepted(LadderRoute::Ri { block: k }, 6);
        for (budget, want) in [(at, k), (at - 1, k - 1)] {
            let c = plan_ladder(shape(6), &cfg(CcsdLadder::Ri), budget).unwrap();
            assert_eq!(c.route, LadderRoute::Ri { block: want }, "budget {budget}");
            assert!(
                c.plan.check().is_ok(),
                "chosen plan must fit budget {budget}"
            );
        }
    }
    let floor = smallest_accepted(LadderRoute::Ri { block: 1 }, 6);
    let c = plan_ladder(shape(6), &cfg(CcsdLadder::Ri), floor - 1).unwrap();
    assert!(
        c.plan.check().is_err(),
        "below the blk=1 threshold the RI plan must refuse, not run at an unaffordable width"
    );
}

/// CONTRACT 5: Auto takes Dense exactly when the dense plan fits, RI (at the
/// widest fitting block) just below that, and RI's refusal when nothing fits.
#[test]
fn auto_switches_at_the_dense_threshold() {
    let dense = smallest_accepted(LadderRoute::Dense, 6);
    let auto = |b| plan_ladder(shape(6), &cfg(CcsdLadder::Auto), b).unwrap();
    assert_eq!(auto(dense).route, LadderRoute::Dense);
    let below = auto(dense - 1);
    assert_eq!(
        below.route,
        LadderRoute::Ri { block: NV },
        "ample for RI at full width"
    );
    assert!(below.plan.check().is_ok());
    let floor = smallest_accepted(LadderRoute::Ri { block: 1 }, 6);
    let starved = auto(floor - 1);
    assert!(matches!(starved.route, LadderRoute::Ri { block: 1 }));
    let msg = starved.plan.check().unwrap_err().to_string();
    assert!(
        msg.contains("RI ladder scratch") && !msg.contains("vvvv"),
        "{msg}"
    );
}

/// CONTRACT 6: the DIIS ring is charged on the RI route too.
#[test]
fn ri_route_charges_the_diis_history() {
    let r = LadderRoute::Ri { block: 1 };
    assert!(smallest_accepted(r, 12) > smallest_accepted(r, 1));
}

/// CONTRACT 7: a pinned width is validated, never clamped or ignored.
#[test]
fn pinned_block_width_is_validated() {
    let big = usize::MAX / 4;
    let pinned = |ladder, b| {
        plan_ladder(
            shape(6),
            &CcConfig {
                ladder,
                ri_ladder_block: Some(b),
                ..Default::default()
            },
            big,
        )
    };
    assert_eq!(
        pinned(CcsdLadder::Ri, 5).unwrap().route,
        LadderRoute::Ri { block: 5 }
    );
    assert!(pinned(CcsdLadder::Ri, 0).is_err());
    assert!(pinned(CcsdLadder::Ri, NV + 1).is_err());
    assert!(
        pinned(CcsdLadder::Auto, 5).is_err(),
        "Auto may not use the pin"
    );
    assert!(
        pinned(CcsdLadder::Dense, 5).is_err(),
        "Dense never uses the pin"
    );
}

/// CONTRACT 8 (reachability): at this shape the `nv⁴` and ladder terms, not
/// `eri3_ao`, set the thresholds above — otherwise contracts 1-3 could pass
/// with those reservations deleted.
#[test]
fn ladder_terms_dominate_eri3_at_this_shape() {
    let eri3 = NAUX * NBAS * NBAS * F64;
    let ri_fixed = (2 * NAUX * NV * NV + 2 * NV.pow(3) + NO * NO * NV * NV) * F64;
    assert!(nv4_bytes() > 5 * eri3, "nv^4 must dominate eri3_ao");
    assert!(ri_fixed > eri3, "the blk=1 RI scratch must exceed eri3_ao");
}

/// The config spelling is strict.
#[test]
fn ladder_parse_is_strict() {
    for l in [CcsdLadder::Auto, CcsdLadder::Dense, CcsdLadder::Ri] {
        assert_eq!(CcsdLadder::parse_config_str(l.as_str()).unwrap(), l);
    }
    for bad in ["", "Auto", "RI", "direct", "dense "] {
        assert!(CcsdLadder::parse_config_str(bad).is_err(), "{bad:?}");
    }
    assert_eq!(CcsdLadder::default(), CcsdLadder::Auto);
}
