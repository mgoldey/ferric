//! MWE: are the production RI-MP2 lanes guarded at all?
//!
//! `rimp2.rs` and `u_rimp2.rs` contain **zero** `check_alloc` call sites. The
//! guarded paths in this crate are the OO-optimized and reference/validation
//! ones (`canonical.rs:46`, `oo_rimp2.rs:{484,706,1095,1273}`,
//! `laplace.rs:457`); the hot lanes every user actually hits —
//! `ri_mp2`, `ri_mp2_spin_components`, `compute_rpa_intermediates`,
//! `u_ri_mp2`, `compute_u_mp2_amplitudes` — allocate and let the OOM killer
//! decide.
//!
//! Concretely, a user passing `memory_budget_gb = 4` gets the AO tensor spilled
//! to disk to respect 4 GB, and then:
//!
//! ```text
//!   b_flat  naux · nocc · nvir · 8        (rimp2.rs:247)
//!   g_i     nvir · (nocc−i) · nvir · 8    (rimp2.rs:436, PER RAYON WORKER)
//!   b_vv    naux · nvir² · 8              (~13 GB at nvir≈860/naux≈2200 —
//!                                          documented in-code as "the M4-audit
//!                                          peak", still unguarded)
//! ```
//!
//! allocated on top with no check, no warning, and no log line. `BudgetResolution::
//! audit_line()` exists and is never called from this crate.
//!
//! These contracts are arithmetic against the published shapes — no SCF, no
//! allocation — so they are safe on a shared box and fast enough to run every
//! time. They specify what a guard must cover before one is written.

/// `b_flat`: the dressed `B^P_ia` output, `(naux, nocc·nvir)` f64.
fn b_flat_bytes(naux: usize, nocc: usize, nvir: usize) -> usize {
    naux.saturating_mul(nocc).saturating_mul(nvir).saturating_mul(8)
}

/// `b_vv`: the vir-vir block, `(naux, nvir²)` f64.
fn b_vv_bytes(naux: usize, nvir: usize) -> usize {
    naux.saturating_mul(nvir).saturating_mul(nvir).saturating_mul(8)
}

/// `g_i` at its widest (`i = 0`), held by EVERY rayon worker concurrently in the
/// `into_par_iter()` energy loop.
fn g_i_peak_bytes(nocc: usize, nvir: usize, n_workers: usize) -> usize {
    nocc.saturating_mul(nvir)
        .saturating_mul(nvir)
        .saturating_mul(8)
        .saturating_mul(n_workers)
}

/// Shapes spanning "fits anywhere" to the documented ~13 GB `b_vv` case.
/// (naux, nocc, nvir)
const SHAPES: [(&str, usize, usize, usize); 4] = [
    ("H2O/cc-pVDZ", 116, 5, 19),
    ("H2O/aug-cc-pVDZ", 246, 5, 36),
    ("benzene/cc-pVDZ", 505, 21, 93),
    ("M4-audit peak", 2200, 60, 860),
];

/// CONTRACT 1: the documented `b_vv` hazard really is ~13 GB.
///
/// Pins the in-code claim at `rimp2.rs:502` ("at nvir≈860/naux≈2200 this block
/// alone is ~13 GB") so the formula and the comment cannot drift apart. If this
/// ever fails, either the shape changed or the doc was wrong.
#[test]
fn b_vv_reaches_the_documented_13gb() {
    let bytes = b_vv_bytes(2200, 860);
    let gb = bytes as f64 / 1e9;
    assert!(
        (12.0..14.5).contains(&gb),
        "b_vv at naux=2200, nvir=860 should be ~13 GB per the in-code doc, got {gb:.2} GB"
    );
}

/// CONTRACT 2: the per-worker energy transient scales with thread count.
///
/// `g_i` is allocated inside `into_par_iter()`, so resident bytes multiply by
/// the worker count — the same defect class as the RPA dipole banding, and
/// invisible to any single-buffer estimate. A guard must multiply by
/// `n_workers`, not count one copy.
#[test]
fn g_i_transient_scales_with_worker_count() {
    let (nocc, nvir) = (60usize, 860usize);
    let one = g_i_peak_bytes(nocc, nvir, 1);
    let twelve = g_i_peak_bytes(nocc, nvir, 12);

    assert_eq!(
        twelve,
        one * 12,
        "the g_i transient must scale linearly with worker count"
    );
    // At the audit shape this alone is multi-GB per worker.
    assert!(
        one as f64 / 1e9 > 0.3,
        "one worker's g_i should already be sizeable at the audit shape, got {:.2} GB",
        one as f64 / 1e9
    );
}

/// CONTRACT 3: an honest floor must exceed any single buffer.
///
/// Specifies what an `ri_mp2` guard must count: `b_flat` plus the per-worker
/// `g_i` fan-out, and `b_vv` when that block is built. Each alone understates
/// the peak, which is why guarding one buffer would not have helped.
#[test]
fn an_honest_floor_exceeds_every_single_buffer() {
    let n_workers = 12;
    for (name, naux, nocc, nvir) in SHAPES {
        let b_flat = b_flat_bytes(naux, nocc, nvir);
        let g_i = g_i_peak_bytes(nocc, nvir, n_workers);
        let b_vv = b_vv_bytes(naux, nvir);
        let honest = b_flat.saturating_add(g_i).saturating_add(b_vv);

        for (label, single) in [("b_flat", b_flat), ("g_i", g_i), ("b_vv", b_vv)] {
            assert!(
                honest > single,
                "{name}: an honest floor ({honest}) must exceed {label} alone ({single})"
            );
        }
    }
}

/// CONTRACT 4: a budget can be "respected" on the AO side and blown on the MO
/// side.
///
/// The exact shape of the bug. `ThreeIndexSource::build` checks the budget
/// against `naux·nao²` ONLY and spills to disk if needed — so the AO tensor
/// honors a 4 GB budget. `b_flat`/`b_vv` are then allocated unconditionally on
/// top. Peak is therefore always strictly greater than the budget, by
/// construction, and nothing notices.
#[test]
fn ao_side_can_fit_while_the_mo_side_blows_the_budget() {
    let (naux, nocc, nvir) = (2200usize, 60usize, 860usize);
    let budget = 4 * 1000 * 1000 * 1000usize; // 4 GB, as a user might set

    // The MO-side blocks the budget never sees.
    let mo_side = b_flat_bytes(naux, nocc, nvir).saturating_add(b_vv_bytes(naux, nvir));

    assert!(
        mo_side > budget,
        "the MO-side blocks ({:.2} GB) must exceed a 4 GB budget for this to be \
         the documented failure; got {:.2} GB",
        mo_side as f64 / 1e9,
        mo_side as f64 / 1e9,
    );
}
