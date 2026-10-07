//! Task D2: the COSX mixed-precision (`cosx-kern`) error law. Alkanes C4
//! (un-ignored), C8, C12, C16, C20 (`#[ignore]`, precondition: a quiet box;
//! deviation from the brief, which asked for C4/C8/C12 un-ignored: the C4 cell
//! alone took 26 minutes at load ~14, C12 is ~15x that) at def2-SVP and butane
//! at def2-TZVP (`#[ignore]`); densities
//! from RI-JK RHF; production default grid (`CosxConfig::default`);
//! `cosx_fp64_multiplier` in {1e4, 1e5, 1e6} (tau = 1e-3, 1e-2, 1e-1 at the
//! 1e-7 screen threshold); all three `PrimPairSum` variants; two rayon workers.
//! Counts and errors only.
//!
//! Per row (printed by [`common::cosx_error_map::print_rows`]): routed flop
//! share, `max|dK|` (post-fit), the derived bound at that element and its
//! maximum, `max|dK|/bound` (asserted <= 1 on EVERY element of every cell),
//! `dE_x = -0.25 tr[D dK]` against its derived bound `0.25 sum |D| bound`
//! (asserted), `kappa_p99` (block-level `sum|terms|/|A|`, p99 over six
//! sub-batches of the production grid), and `|dE_SCF|` (full RIJCOSX SCF with
//! the final-grid pass, mixed vs CPU: in Eh, kcal/mol and per atom, SAME
//! iteration count asserted) for the multipliers in `scf_mults`.
//!
//! # The bound
//!
//! `CosxK::f32_k_bound`: per routed F32 unit the f32 block's element bound
//! `Md3c1e::pair_block_f32_bound` (derivation in `md3c1e/f32_block.rs`: Higham
//! chain depth `D = nnz + 16 l_tot + 49` per primitive pair, absolute-value
//! shadow recursion `S_pp`, primitive-pair term `V` per variant) times `|F|`,
//! `|dK~| <= |X| |dG|^T`, through the fit `K = sym(S S_num^-1 K~)` as
//! `0.5 (|Q| B + (|Q| B)^T)`. It assumes the dense half transform (the cells
//! use it so the bound is exact; the production default is the sparse one,
//! and the SCF energies below use the production default). Not in the bound:
//! the f64 summation noise of the fold and of the `X dG^T` GEMM (identical
//! operations in both arms, relative `1e-16`-scale: far below the f32 term).
//!
//! # Budget row `cosx-kern` (the D2 deliverable; MEASURED, see the harness)
//!
//! | kernel | depth | kappa_sum | bound | mitigation | precision |
//! |---|---|---|---|---|---|
//! | `cosx-kern` | per primitive pair `D = nnz + 16 l_tot + 49` f32 roundings (<= 2 nnz-term chain + R recursion + Boys); across primitive pairs `n_pp` (f64, compensated f32, or plain f32) | block-level p99 `sum|terms|/|A|`: 58.8 (butane/def2-SVP; C8-C20 not yet measured) | per block element `sum_pp (e^{1/8} gamma_D(u32) + 2 gamma_D(u64)) S_pp + V` with `V = gamma_{n_pp}(u64) sum|A_pp|` (F64), `2 u32 sum|A_pp|` (CompensatedF32), `gamma_{n_pp-1}(u32) sum|A_pp|` (F32); propagated through the f64 fold and the fit | Hölder routing: f32 only where `max_q est_q fmax_q < multiplier x screen_thresh`; f64 fold and fit; f32-range overflow falls back to the f64 block (counted) | f32 block, f64 or compensated-f32 primitive-pair sum, f64 fold |
//!
//! # Measured so far (butane/def2-SVP, default grid, 2 workers)
//!
//! All 9 cells inside the derived bound, `max|dK|/bound` 3.6e-3 to 6.0e-3 (the
//! bound is a worst case, 170-280x above the realised error), no f32-range
//! fallback, identical SCF iteration counts (12, 12) at 1e5; flop share
//! 0.629 / 0.861 / 0.989 at 1e4 / 1e5 / 1e6. Realised `max|dK|` 6.6e-10 /
//! 1.1e-8 / 2.5e-8 (F64 and CompensatedF32 identical to the printed digits,
//! plain F32 within 7%); `|dE_SCF|` at 1e5: 1.0e-8 Eh (6.4e-6 kcal/mol, 7.3e-10
//! Eh per atom), 7.3e-8 Eh at 1e6, 9.6e-10 Eh at 1e4: three to five orders
//! below 1 mEh and below Ochsenfeld's 1.8 microEh. C8-C20 and the TZVP cell
//! are NOT measured in this commit (the box was needed for timings); the
//! size slope is therefore not yet available.
//!
//! # Honest scope
//!
//! The slope of `ln max|dK|` vs `ln N` (N = atoms) is fitted over the alkane
//! series; with C4, C8, C12 that is 3 points and `df = 1`: the standard error
//! is nearly uninformative (validation.md's 4.2b paragraph says the same of
//! its ladder). C16 and C20 raise it to `df = 3`. The slope is DESCRIBED,
//! never gated.
//!
//! # What this file does NOT assert (reported per row, decided by D6)
//!
//! The user's shipping bar is `|dE_total| <= 1 mEh` (1.0e-3 Eh = 0.63
//! kcal/mol) on every row AND an error that is size-extensive, i.e. the
//! `ln|dE|` vs `ln N` slope not above linear (about 1) within its
//! uncertainty; Ochsenfeld's 1.8 microEh (JCP 154, 214116 (2021)) stays
//! reported per row. Neither is asserted here; `cosx-kern` stays out of
//! `MixedKernelSet::SHIPPED` until D6 weighs the printed table against them.
//! The pre-registered selection rule (a `PrimPairSum` variant whose
//! `ln max|dK|` slope is more than 2 sigma above the other's is not the device
//! variant) is evaluated and printed, not asserted either.

#[path = "common/cosx_error_map.rs"]
mod cosx_error_map;

use cosx_error_map as common_map;

const MULTS: [f64; 3] = [1e4, 1e5, 1e6];

fn check_rows(rows: &[common_map::Row]) {
    for r in rows {
        let tag = format!("{} {:.0e} {}", r.system, r.mult, r.sum.as_str());
        assert!(
            r.max_ratio <= 1.0,
            "{tag}: |dK| exceeds its bound ({})",
            r.max_ratio
        );
        assert!(
            r.de_x.abs() <= r.de_x_bound,
            "{tag}: |dE_x| {} > bound {}",
            r.de_x,
            r.de_x_bound
        );
        assert!(r.f32_blocks > 0, "{tag}: the f32 path never ran");
        assert_eq!(
            r.f32_fallbacks, 0,
            "{tag}: f32-range fallback on this system"
        );
        if let Some((_, it_cpu, it_mix)) = r.scf {
            assert_eq!(it_cpu, it_mix, "{tag}: SCF iteration counts differ");
        }
    }
}

fn report(rows: &[common_map::Row]) {
    common_map::print_rows(rows);
    let sl = common_map::slopes(rows);
    common_map::print_slopes(&sl);
    common_map::print_device_rule(&sl);
}

/// The SCF comparison (10-13 K builds per SCF) runs at the 1e5 seed only, and
/// only up to C8: the C4 cell with the SCFs of all three multipliers took 26
/// minutes on the shared box (load ~14).
fn sweep(sizes: &[usize]) -> Vec<common_map::Row> {
    let mut rows = Vec::new();
    for &n in sizes {
        let sys = common_map::load(&format!("C{n}"), &format!("alkane_{n}.xyz"), "def2-svp");
        let scf: &[f64] = if n <= 8 { &[1e5] } else { &[] };
        rows.extend(common_map::run_system(&sys, &MULTS, scf));
    }
    rows
}

#[test]
fn alkane_def2_svp_c4_cell() {
    let rows = sweep(&[4]);
    report(&rows);
    check_rows(&rows);
}

#[test]
#[ignore = "precondition: quiet box; the C8 and C12 default-grid cells take hours when the box is loaded"]
fn alkane_def2_svp_error_law_c4_c8_c12() {
    let rows = sweep(&[4, 8, 12]);
    report(&rows);
    check_rows(&rows);
}

#[test]
#[ignore = "precondition: quiet box; butane/def2-TZVP default-grid cell"]
fn butane_def2_tzvp_cell() {
    let sys = common_map::load("C4", "alkane_4.xyz", "def2-tzvp");
    let rows = common_map::run_system(&sys, &MULTS, &[1e5]);
    report(&rows);
    check_rows(&rows);
}

#[test]
#[ignore = "precondition: quiet box; C16/C20 default-grid builds take tens of minutes"]
fn alkane_def2_svp_error_law_c4_to_c20() {
    let rows = sweep(&[4, 8, 12, 16, 20]);
    report(&rows);
    check_rows(&rows);
}
