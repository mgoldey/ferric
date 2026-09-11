//! MWE: the UHF/UKS Newton f_xc kernel duplicates the live KS grid cache and
//! its allocation was never gated at all.
//!
//! # The defect
//!
//! `solve_uhf_fockmod` (`crates/ferric-scf/src/uhf.rs`) builds a `KsXcUks`
//! grid AO cache (`xc_contrib`) once, near the top of the function, and holds
//! it live for the whole SCF loop. When the second-order Newton update
//! engages (`config.newton_trigger > 0.0` and `err_max` has dropped below it),
//! it calls `crate::rohf::FxcKernelStore::build`, which — for any non-LDA
//! functional — constructs a `ferric_dft::fxc::GgaFxcKernel`. That kernel's
//! `new()` calls `eval_basis_and_grad_on_points` on the SAME molecule, basis,
//! and (`config.dft_grid`) grid config as `xc_contrib` already used, i.e. it
//! allocates a SECOND (nbf, npts) chi + (3, nbf, npts) dchi cache — 4 planes —
//! duplicating the one already resident. This reproduces on every Newton
//! step (the kernel is rebuilt per step, not cached across them).
//!
//! `eval_basis_and_grad_on_points`'s own internal gate
//! (`ferric_integrals::ao_grid::check_ao_grid_budget`) cannot see any of this:
//! it re-resolves `ferric_core::memory::resolve_budget_bytes(None)` — the
//! WHOLE ceiling, ignoring both `config.three_index_budget_bytes` and the
//! live KS cache — so it independently re-approves the duplicate against
//! 100% of the same budget the KS cache was already charged against. Before
//! this fix, `uhf.rs` had NO gate of its own at this call site at all.
//!
//! # Measured magnitude
//!
//! One (nbf, npts) plane is `nbf * npts * 8` bytes, `npts = natoms *
//! n_radial * n_angular` (the kernel builds an UNPRUNED atomic grid: 75*110 =
//! 8250 points/atom at the library default).
//!
//! - benzene/aug-cc-pVTZ (nbf=414, 12 atoms, npts=12*8250=99000): one plane
//!   is 414*99000*8 = 327,888,000 B ≈ 0.328 GB. The duplicate 4-plane kernel
//!   costs 4x that ≈ 1.31 GB, on top of the ≈1.31 GB `xc_contrib` KS cache
//!   already resident — 2.62 GB peak where either gate alone only ever
//!   independently certified 1.31 GB as affordable.
//! - danuglipron/def2-SVP (nbf=700, 73 atoms, npts=73*8250=602250): one plane
//!   is 700*602250*8 = 3,372,600,000 B ≈ 3.37 GB. The duplicate kernel costs
//!   ≈13.49 GB on top of the ≈13.49 GB KS cache already resident — 26.98 GB
//!   peak.
//!
//! These are not rounding terms: they are exactly the same order of
//! magnitude as the KS cache itself, because they ARE the same shape.
//!
//! # Scope of this fix
//!
//! This MWE pins the arithmetic of the new pure helper,
//! `ferric_scf::uhf::fxc_kernel_duplicate_bytes`, that the `solve_uhf_fockmod`
//! Newton branch now uses to gate the duplicate allocation via
//! `ferric_core::memory::{available_budget_now, check_alloc}` BEFORE calling
//! `FxcKernelStore::build`, rather than relying solely on the unfixed,
//! nested `check_ao_grid_budget` (out of scope for this file — it lives in
//! `ferric-integrals::ao_grid`, and it still resolves the whole ceiling with
//! no residency awareness). This does not eliminate the duplication itself
//! (that requires `GgaFxcKernel`/`LdaFxcKernel` to borrow `xc_contrib`'s
//! cache instead of rebuilding it, out of scope for `ferric-scf`) — it only
//! makes the allocation honestly accounted for before it happens.
//!
//! An explicit end-to-end reproduction (build a UKS Newton step at a shape
//! large enough to exceed a small budget) is not exercised here: constructing
//! `solve_uhf_fockmod`'s Newton branch needs a real molecule/basis/DFT grid
//! and several SCF iterations to reach `err_max < newton_trigger`, which is
//! far from "milliseconds." The pure-arithmetic contracts below are the
//! affordable proxy; the call-site wiring itself is covered by inspection
//! (quoted in the module doc above) rather than a second, slow integration
//! test.

use ferric_dft::grid::AtomicGridConfig;
use ferric_scf::uhf::fxc_kernel_duplicate_bytes;

const GIB: usize = 1024 * 1024 * 1024;

fn default_grid() -> AtomicGridConfig {
    AtomicGridConfig::default() // n_radial=75, n_angular=110
}

/// CONTRACT 1 (reachability / magnitude): the benzene/aug-cc-pVTZ shape from
/// the brief reproduces the measured ~1.31 GB, not a rounding artifact.
#[test]
fn benzene_shape_matches_measured_magnitude() {
    let nbf = 414;
    let natoms = 12;
    let bytes = fxc_kernel_duplicate_bytes(nbf, natoms, &default_grid());
    let gb = bytes as f64 / 1e9;
    assert!(
        (gb - 1.311552).abs() < 1e-3,
        "benzene/aug-cc-pVTZ duplicate kernel should cost ~1.312 GB, got {gb:.6} GB"
    );
}

/// CONTRACT 1b: danuglipron/def2-SVP reproduces the measured ~13.49 GB.
#[test]
fn danuglipron_shape_matches_measured_magnitude() {
    let nbf = 700;
    let natoms = 73;
    let bytes = fxc_kernel_duplicate_bytes(nbf, natoms, &default_grid());
    let gb = bytes as f64 / 1e9;
    assert!(
        (gb - 13.4904).abs() < 1e-2,
        "danuglipron/def2-SVP duplicate kernel should cost ~13.49 GB, got {gb:.6} GB"
    );
}

/// CONTRACT 2 (the defect this fixes): the duplicate kernel plus the already-
/// resident KS cache of the SAME shape must NOT both be admitted against the
/// full budget independently. Using `fxc_kernel_duplicate_bytes` to size the
/// SECOND allocation and gating it against `available_budget_now` (which
/// subtracts what the first allocation already left resident, via live RSS)
/// must refuse a case where the sum exceeds budget even though either half
/// alone would fit.
#[test]
fn duplicate_plus_resident_ks_cache_exceeds_a_budget_that_fits_either_alone() {
    let nbf = 700;
    let natoms = 73;
    let kernel_bytes = fxc_kernel_duplicate_bytes(nbf, natoms, &default_grid());
    // The KS cache (xc_contrib) is the SAME shape (chi+dchi on the same grid).
    let ks_cache_bytes = kernel_bytes;

    let budget = 16 * GIB; // fits ONE ~13.49 GB copy comfortably, not two
    assert!(
        kernel_bytes < budget,
        "sanity: either allocation alone must fit the chosen budget"
    );

    // Simulate what `available_budget_now` would see once the KS cache is
    // already resident: `available_budget_bytes(budget, Some(ks_cache_bytes))`.
    let avail = ferric_core::memory::available_budget_bytes(budget, Some(ks_cache_bytes));
    let verdict = ferric_core::memory::check_alloc("newton f_xc kernel", kernel_bytes, avail);
    assert!(
        verdict.is_err(),
        "a second ~13.49 GB allocation on top of an already-resident ~13.49 GB KS cache must \
         be refused against a 16 GiB budget (sum ~27 GB), but check_alloc admitted it \
         (avail={avail}, needed={kernel_bytes})"
    );
}

/// CONTRACT 3 (over-rejection guard): an AMPLE budget with nothing else
/// resident must still admit the duplicate kernel. A guard that refuses jobs
/// which would have fit is as much a defect as one that admits an OOM.
#[test]
fn ample_budget_with_nothing_resident_still_admits_the_kernel() {
    let nbf = 414;
    let natoms = 12;
    let kernel_bytes = fxc_kernel_duplicate_bytes(nbf, natoms, &default_grid());

    let budget = 64 * GIB; // ample: room for many multiples of ~1.31 GB
    let avail = ferric_core::memory::available_budget_bytes(budget, Some(0));
    let verdict = ferric_core::memory::check_alloc("newton f_xc kernel", kernel_bytes, avail);
    assert!(
        verdict.is_ok(),
        "an ample budget with nothing resident must still admit the ~1.31 GB duplicate kernel, \
         got {verdict:?}"
    );
}

/// CONTRACT 4 (reachability of the helper's inputs): the byte count scales
/// linearly in `nbf` and in `natoms` (through npts), so this is not an inert
/// constant that would pass regardless of shape.
#[test]
fn byte_count_scales_with_nbf_and_natoms() {
    let cfg = default_grid();
    let base = fxc_kernel_duplicate_bytes(100, 1, &cfg);
    let double_nbf = fxc_kernel_duplicate_bytes(200, 1, &cfg);
    let double_natoms = fxc_kernel_duplicate_bytes(100, 2, &cfg);
    assert_eq!(double_nbf, base * 2, "must scale linearly with nbf");
    assert_eq!(double_natoms, base * 2, "must scale linearly with natoms (via npts)");
}

/// CONTRACT 5: exactly 4 planes (chi + dchi_x/y/z) of `nbf*npts` `f64`s — not
/// 1 (chi only, which would undercount by 4x and miss most of the defect) and
/// not more (which would be an over-estimating guard, itself a bug per this
/// session's house rules).
#[test]
fn byte_count_is_exactly_four_planes() {
    let nbf = 10;
    let natoms = 1;
    let cfg = AtomicGridConfig { n_radial: 5, n_angular: 6, ..AtomicGridConfig::default() };
    let npts = natoms * cfg.n_radial * cfg.n_angular;
    let expected = 4 * nbf * npts * std::mem::size_of::<f64>();
    assert_eq!(fxc_kernel_duplicate_bytes(nbf, natoms, &cfg), expected);
}
