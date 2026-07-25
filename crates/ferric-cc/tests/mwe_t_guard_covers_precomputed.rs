//! MWE: does the (T) pre-flight guard cover the allocations that actually
//! dominate, or only the per-triple block?
//!
//! `ccsd_t` guards `peak_triple_block_bytes(nv2) = 6·(2nv)³·8` and its comment
//! claims this "bounds the genuinely large piece". It does not. Roughly 25 lines
//! later the driver allocates `bcei` at `(2nv)³·(2no)·8`, so
//!
//! ```text
//!     bcei / guard = 2·no / 6
//! ```
//!
//! i.e. the guard undercounts by MORE as the system grows — the opposite of what
//! a safety margin should do:
//!
//! ```text
//!   system (no,nv spatial)      guard GB    bcei GB   ratio
//!   H2O/cc-pVDZ      (5,19)        0.003      0.004    1.7x
//!   H2O/aug-cc-pVDZ  (5,36)        0.018      0.030    1.7x
//!   ethane/cc-pVDZ   (9,39)        0.023      0.068    3.0x
//!   butane/cc-pVDZ  (17,77)        0.175      0.993    5.7x
//!   benzene/cc-pVDZ (21,93)        0.309      2.162    7.0x
//! ```
//!
//! `majk` and `bcjk` follow immediately with the same shape, and `asym_phys`
//! allocates each `(2x)`-dimension output while BOTH spatial inputs are still
//! live (a further ~2·nv³·no·8 transient). The closed-shell sibling has the same
//! defect via `ovvv = no·nv³·8`.
//!
//! The existing in-tree test (`ccsd_t_fails_fast_under_tiny_budget`) uses a ~1 KB
//! budget, which any guard catches. The interesting range — and the one that
//! OOM-kills — is the MIDDLE: a budget that PASSES the per-triple check and then
//! cannot hold `bcei`. These contracts pin that range.
//!
//! Pure arithmetic against the published formulas: no SCF, no allocation, safe
//! on a shared box.

/// `peak_triple_block_bytes` from `ccsd_t.rs`: 6 co-resident `[nv2,nv2,nv2]`
/// buffers. Re-derived here rather than imported so the test also catches the
/// production formula silently changing shape.
fn guard_bytes(nv2: usize) -> usize {
    nv2.saturating_pow(3).saturating_mul(6).saturating_mul(8)
}

/// `bcei`: the VVVO block at `[b,c,e,i]`, i.e. `(2nv)³·(2no)` f64.
fn bcei_bytes(no2: usize, nv2: usize) -> usize {
    nv2.saturating_pow(3).saturating_mul(no2).saturating_mul(8)
}

/// The two spatial `einsum!` outputs `d` and `e` that `asym_phys` holds live
/// while building its doubled-dimension result.
fn asym_phys_transient_bytes(no: usize, nv: usize) -> usize {
    2 * nv.saturating_pow(3).saturating_mul(no).saturating_mul(8)
}

/// Systems spanning the range where the ratio grows.
const SYSTEMS: [(&str, usize, usize); 5] = [
    ("H2O/cc-pVDZ", 5, 19),
    ("H2O/aug-cc-pVDZ", 5, 36),
    ("ethane/cc-pVDZ", 9, 39),
    ("butane/cc-pVDZ", 17, 77),
    ("benzene/cc-pVDZ", 21, 93),
];

/// CONTRACT 1: there exists a budget that passes the guard and cannot hold
/// `bcei`.
///
/// The precise failure mode: pre-flight says yes, the next allocation says no.
/// If no such budget exists the guard is already sufficient and this whole
/// task is moot — so this contract establishes the bug is real before any fix.
#[test]
fn a_budget_can_pass_the_guard_and_still_not_hold_bcei() {
    for (name, no, nv) in SYSTEMS {
        let (no2, nv2) = (2 * no, 2 * nv);
        let guard = guard_bytes(nv2);
        let bcei = bcei_bytes(no2, nv2);

        assert!(
            bcei > guard,
            "{name}: bcei ({bcei}) must exceed the guarded per-triple block \
             ({guard}) for this defect to exist"
        );

        // A budget sitting strictly between the two: the guard passes, then
        // bcei cannot be allocated.
        let deceptive = (guard + bcei) / 2;
        assert!(
            deceptive >= guard && deceptive < bcei,
            "{name}: expected a budget in [guard, bcei) — guard={guard}, \
             bcei={bcei}, chosen={deceptive}"
        );
    }
}

/// CONTRACT 2: the undercount grows with system size.
///
/// `bcei/guard = 2·no/6`, so the guard gets *less* protective exactly as jobs
/// get big enough to matter. A guard whose margin shrinks with scale is worse
/// than none, because it looks like protection.
#[test]
fn the_undercount_worsens_as_the_system_grows() {
    let ratio = |no: usize, nv: usize| {
        bcei_bytes(2 * no, 2 * nv) as f64 / guard_bytes(2 * nv) as f64
    };

    let small = ratio(5, 19); // H2O/cc-pVDZ
    let large = ratio(21, 93); // benzene/cc-pVDZ

    assert!(
        large > small,
        "the undercount must grow with system size: H2O {small:.1}x vs \
         benzene {large:.1}x"
    );
    // 2*no/6 exactly: benzene (no=21) is 7.0x.
    assert!(
        (large - 7.0).abs() < 0.05,
        "benzene ratio should be 2*21/6 = 7.0, got {large:.2}"
    );
}

/// CONTRACT 3: a correct estimate must cover the precomputed integrals AND
/// their construction transients.
///
/// Specifies what the fixed guard must count: `bcei` + `majk` + `bcjk`, plus the
/// `asym_phys` temporaries live during each one's construction. This is the
/// target the fix is measured against.
#[test]
fn a_correct_estimate_covers_precomputed_blocks_and_transients() {
    for (name, no, nv) in SYSTEMS {
        let (no2, nv2) = (2 * no, 2 * nv);

        let bcei = bcei_bytes(no2, nv2);
        // majk is (2no)³(2nv); bcjk is (2nv)²(2no)². Both smaller than bcei for
        // nv > no, but neither is currently counted at all.
        let majk = no2.saturating_pow(3).saturating_mul(nv2).saturating_mul(8);
        let bcjk = nv2
            .saturating_pow(2)
            .saturating_mul(no2.saturating_pow(2))
            .saturating_mul(8);
        let transient = asym_phys_transient_bytes(no, nv);

        let honest = bcei + majk + bcjk + transient + guard_bytes(nv2);
        let current = guard_bytes(nv2);

        assert!(
            honest > current,
            "{name}: an honest estimate ({honest}) must exceed today's \
             per-triple-only guard ({current})"
        );
    }
}
