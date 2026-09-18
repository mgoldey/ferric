//! The (T) triple-band width must be a PERFORMANCE knob, never an energy knob.
//!
//! # Why this test exists before the pool migration, not after
//!
//! `ccsd_t.rs` and `ccsd_t_closed_shell.rs` size their parallel triple band
//! (`triple_chunk_len`) from a BYTE BUDGET. The pool migration changes where
//! that byte number comes from: it stops being "the whole resolved ceiling,
//! re-read" and becomes "what the pool has LEFT after the precomputed blocks
//! were charged". That is a strictly smaller, and job-dependent, number -- so
//! the band width WILL change under the migration.
//!
//! The ksdft path already paid for getting this wrong once: the KS grid batch
//! width was sized from a budget minus LIVE RSS, batch width set accumulation
//! order, and the same input gave -390.3794282913 and -390.3794337741 Ha. The
//! rule that came out of it is that a budget-derived width may depend on the
//! budget and the problem, never on transient state -- and that the only way
//! to know a width is safe is to pin that it does not move the answer.
//!
//! So: before the (T) band width is allowed to depend on the pool ledger, this
//! file pins that the width does not move `et` AT ALL, at ANY width, to the
//! BIT.
//!
//! # Why this is true here (read the loop, not the docstring)
//!
//! ```text
//! for chunk in triples.chunks(chunk_len) {
//!     let partials: Vec<f64> = chunk.par_iter().map(..).collect();
//!     for p in partials { et += p; }
//! }
//! ```
//!
//! `par_iter().collect()` preserves index order regardless of worker count,
//! and the fold is serial in ascending triple order. Chunk boundaries split
//! that ascending sequence into contiguous runs but never REORDER it, so the
//! total addition order is the same ascending order for every `chunk_len`.
//! Floating-point `+` is non-associative but it is being applied in an
//! identical sequence, so the bit pattern is identical.
//!
//! ## Artifact hypothesis (stated before measuring)
//!
//! If the above reading of the loop is RIGHT, `et` is bit-identical across
//! widths spanning serial (1) to whole-list (all triples in one chunk).
//! If it is WRONG -- e.g. if a future refactor makes each chunk produce a
//! chunk-local partial sum that is then folded (a tree fold in disguise) --
//! `et` moves in the last few bits, which `to_bits()` equality catches and an
//! `abs() < 1e-12` assertion would not.
//!
//! ## Reachability of the pass condition -- and what it caught
//!
//! A test that only ever exercises ONE width is arithmetic, not measurement.
//! The first draft of this file swept four round-number budgets and asserted
//! bit-identity. It PASSED. `band_widths_are_actually_distinct` then showed
//! why that pass was worthless: the four budgets produced band widths
//! `[1942, 6500, 65093, 1302072]` on a fixture with only 120 triples, so
//! `triples.chunks(w)` yielded ONE chunk in all four cases. The test compared
//! a single execution shape against itself.
//!
//! The budgets below are therefore SOLVED BACKWARDS from the driver's own
//! width function to hit target widths, and the reachability assertion is on
//! the EFFECTIVE width (`min(width, n_triples)`) -- the thing that actually
//! changes the chunk decomposition -- not on the raw width.
//!
//! ## Why no test in THIS file installs a process-global pool
//!
//! It did, briefly, and that cost eight spurious failures. The pool slot is
//! process-global and cargo runs a binary's tests CONCURRENTLY, so a test that
//! installs a tight pool silently refuses its siblings' CC drivers -- the
//! exact leak `e9949eed` fixed for the integrals tests. A serializing mutex is
//! not enough: it orders the pool-installing tests against each other while
//! the siblings run outside the lock.
//!
//! So this file is pool-FREE (every test drives the unbudgeted path through an
//! explicit `memory_budget_bytes`), and the pooled-width determinism test
//! lives in `mwe_cc_charge_lifetime.rs`, which installs pools under a
//! `CleanSlot` guard throughout.

use ferric_cc::ccsd::ccsd;
use ferric_cc::ccsd_closed_shell::ccsd_closed_shell;
use ferric_cc::ccsd_t::ccsd_t;
use ferric_cc::ccsd_t_closed_shell::ccsd_t_closed_shell;
use ferric_cc::{CcConfig, CcResult};
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// H2O / STO-3G: no = 5, nv = 2 spatial, so no2 = 10, nv2 = 4.
const NO: usize = 5;
const NV: usize = 2;
const NO2: usize = 2 * NO;
const NV2: usize = 2 * NV;
/// Unique i<j<k over 10 occupied spin-orbitals.
const N_TRIPLES_SO: usize = 120;
/// i<=j<=k with repeats over 5 occupied spatial orbitals.
const N_TRIPLES_CS: usize = 35;

struct Fixture {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

fn fixture() -> Fixture {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz");
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").expect("sto-3g")).expect("obs");
    let dfbs =
        PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri")).expect("dfbs");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).expect("RHF");
    Fixture {
        mol,
        obs,
        dfbs,
        rhf,
    }
}

fn cfg_with_budget(bytes: usize) -> CcConfig {
    CcConfig {
        frozen_core: 0,
        max_iter: 30,
        memory_budget_bytes: Some(bytes),
        ..Default::default()
    }
}

/// Converge the amplitudes ONCE with an ample budget, so the budgets swept
/// below vary only the (T) band and not the CCSD iteration that feeds it.
fn converged(f: &Fixture, spin_orbital: bool) -> CcResult {
    let cfg = cfg_with_budget(8_000_000_000);
    let driver = if spin_orbital {
        ccsd
    } else {
        ccsd_closed_shell
    };
    driver(&f.mol, &f.obs, &f.dfbs, Operator::coulomb(), &f.rhf, &cfg).expect("CCSD must converge")
}

/// Budgets that hit band widths 1, 2, 5, 17, 60 and 200 on the SPIN-ORBITAL
/// path, solved backwards from `ccsd_t::t_band_width`:
/// `budget = 2 * width * per_triple + precomputed`, with per_triple = 3072 B
/// and precomputed = 63680 B at (no2, nv2) = (10, 4).
///
/// Width 1 is fully serial; width 200 exceeds the 120-triple list, i.e. one
/// chunk. So the sweep spans the whole decomposition range.
const BUDGETS_SO: [usize; 6] = [69_824, 75_968, 94_400, 168_128, 432_320, 1_292_480];

/// Same construction on the CLOSED-SHELL path: per_triple = 384 B and
/// precomputed = 4000 B at (no, nv) = (5, 2), target widths 1, 2, 5, 17, 40.
const BUDGETS_CS: [usize; 5] = [4_768, 5_536, 7_840, 17_056, 34_720];

/// The effective band width: the raw width capped at the number of triples,
/// because that is what actually determines the chunk decomposition.
fn effective(width: usize, n_triples: usize) -> usize {
    width.min(n_triples).max(1)
}

#[test]
fn band_widths_are_actually_distinct() {
    // REACHABILITY GUARD. Without this, the bit-identity tests below can pass
    // while comparing one execution shape against itself -- which is exactly
    // what the first draft of this file did (widths 1942..1302072 on a
    // 120-triple list: one chunk, four times over).
    let so: Vec<usize> = BUDGETS_SO
        .iter()
        .map(|&b| effective(ferric_cc::ccsd_t::t_band_width(NO2, NV2, b), N_TRIPLES_SO))
        .collect();
    let cs: Vec<usize> = BUDGETS_CS
        .iter()
        .map(|&b| {
            effective(
                ferric_cc::ccsd_t_closed_shell::t_band_width(NO, NV, b),
                N_TRIPLES_CS,
            )
        })
        .collect();

    for (name, widths, n) in [
        ("spin-orbital", &so, N_TRIPLES_SO),
        ("closed-shell", &cs, N_TRIPLES_CS),
    ] {
        let distinct: std::collections::BTreeSet<usize> = widths.iter().copied().collect();
        assert!(
            distinct.len() >= 4,
            "{name}: the budget sweep must produce at least 4 DISTINCT effective \
             band widths or the bit-identity assertions are vacuous; got {widths:?}"
        );
        assert_eq!(
            widths.first().copied(),
            Some(1),
            "{name}: the sweep must include the fully-serial width 1: {widths:?}"
        );
        assert_eq!(
            widths.last().copied(),
            Some(n),
            "{name}: the sweep must include the whole-list width {n} (one chunk): \
             {widths:?}"
        );
    }
}

#[test]
fn spin_orbital_t_energy_is_bit_identical_across_band_widths() {
    let f = fixture();
    let cc = converged(&f, true);

    let mut seen: Option<(usize, f64)> = None;
    for &b in &BUDGETS_SO {
        let w = effective(ferric_cc::ccsd_t::t_band_width(NO2, NV2, b), N_TRIPLES_SO);
        let et = ccsd_t(
            &f.mol,
            &f.obs,
            &f.dfbs,
            Operator::coulomb(),
            &f.rhf,
            &cc,
            &cfg_with_budget(b),
        )
        .unwrap_or_else(|e| panic!("(T) refused budget {b}: {e}"));
        match seen {
            None => seen = Some((w, et)),
            Some((w0, et0)) => assert_eq!(
                et0.to_bits(),
                et.to_bits(),
                "spin-orbital (T) moved with the band width: width {w0} gave \
                 {et0:.17e}, width {w} gave {et:.17e}. The band width is a \
                 PERFORMANCE knob; if it moves the energy, sizing it from the \
                 pool ledger is not safe."
            ),
        }
    }
}

#[test]
fn closed_shell_t_energy_is_bit_identical_across_band_widths() {
    let f = fixture();
    let cc = converged(&f, false);

    let mut seen: Option<(usize, f64)> = None;
    for &b in &BUDGETS_CS {
        let w = effective(
            ferric_cc::ccsd_t_closed_shell::t_band_width(NO, NV, b),
            N_TRIPLES_CS,
        );
        let et = ccsd_t_closed_shell(
            &f.mol,
            &f.obs,
            &f.dfbs,
            Operator::coulomb(),
            &f.rhf,
            &cc,
            &cfg_with_budget(b),
        )
        .unwrap_or_else(|e| panic!("closed-shell (T) refused budget {b}: {e}"));
        match seen {
            None => seen = Some((w, et)),
            Some((w0, et0)) => assert_eq!(
                et0.to_bits(),
                et.to_bits(),
                "closed-shell (T) moved with the band width: width {w0} gave \
                 {et0:.17e}, width {w} gave {et:.17e}"
            ),
        }
    }
}
