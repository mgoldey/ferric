//! PAIR-SUM validation for the C6 lane (policy decision, 2026-09-16).
//!
//! BACKGROUND. `wiki/VALIDATION.md` carried "Per-atom C6 magnitudes (any
//! partition) — partition-dependent, ~10x spread" under "Claimed-but-unproven".
//! That item is UNCLOSEABLE as stated, and this file records why and what
//! replaced it:
//!
//!   * A per-atom C6 is not a physical observable. It is a PARTITION
//!     CONVENTION — a choice of how to carve one molecular density into
//!     atoms. Becke and Hirshfeld give different numbers for the same
//!     molecule (~10x on the per-atom magnitudes; see
//!     `s9_per_atom_c6_consistency.rs::partition_dependence_becke_vs_hirshfeld_water`)
//!     and NEITHER is "wrong". There is no experiment that measures the C6
//!     of an atom inside a molecule, so no amount of validation can make one
//!     convention correct. Asking to validate the magnitude is a category
//!     error, not an open task.
//!   * What IS observable is the MOLECULAR C6, and it has DOSD references in
//!     this repo. So the validation effort goes there.
//!
//! This file therefore validates the thing that can be validated
//! (`C6Result::c6_molecular_iso` vs DOSD) and pins the construction, so the
//! wrong one cannot quietly become the convention later.
//!
//! MEASUREMENT vs INTERPRETATION (kept separate per the repo's Experimental
//! Protocol). The measurements are the recorded 2026-07-20 dosd2 sweep
//! (`scripts/dosd2/results.json`, 80/80 entries) against
//! `scripts/dosd2/refs.json` (Toulouse et al. arXiv:1305.0107 Table III,
//! Meath-school DOSD). The interpretation — that the residual is ferric's
//! documented RPA@PBE/aug-cc-pVDZ systematic underbinding and not a bug — is
//! argued in `docs/dosd-c6-rpa-vs-ts.md` and is NOT re-derived here.
//!
//! Two tests:
//!   1. `partition_and_source_labels_round_trip_through_the_strict_parsers`
//!      (cheap, always runs) — the provenance tag exported to NPZ parses back
//!      to the variant it came from.
//!   2. `molecular_c6_matches_dosd_n2o` (#[ignore]d, needs a real RPA@PBE
//!      run) — the DOSD anchor, on the SAME production seam the NPZ and the
//!      CLI use.
//!
//! Run the anchor:
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --release \
//!     --test c6_pair_sum_vs_dosd -- --ignored --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
use ferric_rpa::dispersion::{
    casimir_polder_c6, pdep_dynamic_polarizability, C6Source, DispersionPartition,
};
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

// ---------------------------------------------------------------------------
// 1. Provenance labels (cheap).
// ---------------------------------------------------------------------------

/// `as_config_str` must be the exact inverse of `parse_config_str` for EVERY
/// variant. This is what makes the NPZ `c6_partition`/`c6_source` tags
/// trustworthy: the exported string is guaranteed to name the variant that
/// actually ran, and a consumer (or a re-run) that feeds it back through the
/// strict parser lands on the same setting.
///
/// Without this, the label and the parser are two independent `match`
/// statements that can silently drift — and a DRIFTED tag is worse than no
/// tag, because it asserts a provenance that is false.
#[test]
fn partition_and_source_labels_round_trip_through_the_strict_parsers() {
    for p in [DispersionPartition::Becke, DispersionPartition::Hirshfeld] {
        let s = p.as_config_str();
        let back = DispersionPartition::parse_config_str(Some(s))
            .unwrap_or_else(|e| panic!("label {s:?} for {p:?} does not parse: {e}"))
            .unwrap_or_else(|| panic!("label {s:?} for {p:?} parsed as None"));
        assert_eq!(
            back, p,
            "partition label {s:?} round-tripped to {back:?}, not {p:?} — the NPZ \
             c6_partition tag would assert a FALSE provenance"
        );
    }

    for s in [C6Source::Ts, C6Source::Pdep, C6Source::Mbd] {
        let label = s.as_config_str();
        let back = C6Source::parse_config_str(Some(label))
            .unwrap_or_else(|e| panic!("label {label:?} for {s:?} does not parse: {e}"));
        assert_eq!(
            back, s,
            "source label {label:?} round-tripped to {back:?}, not {s:?}"
        );
    }

    // REACHABILITY: the labels must be DISTINCT, or the assertions above pass
    // vacuously (a single constant string would round-trip for exactly one
    // variant and the loop would fail — but a shared label between two
    // variants of the same enum is the failure mode worth naming).
    assert_ne!(
        DispersionPartition::Becke.as_config_str(),
        DispersionPartition::Hirshfeld.as_config_str(),
        "the two partitions share a label — the tag cannot distinguish them"
    );
    let src_labels = [
        C6Source::Ts.as_config_str(),
        C6Source::Pdep.as_config_str(),
        C6Source::Mbd.as_config_str(),
    ];
    for i in 0..src_labels.len() {
        for j in (i + 1)..src_labels.len() {
            assert_ne!(
                src_labels[i], src_labels[j],
                "two C6Source variants share the label {:?}",
                src_labels[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. DOSD anchor (real SCF, #[ignore]d).
// ---------------------------------------------------------------------------

/// N2O, aug-cc-pVDZ, PDEP-RPA@PBE. Chosen because it is the CHEAPEST entry in
/// the recorded dosd2 sweep that has a DOSD reference (6.1 s wall for
/// `augccpvdz/n2o/rpa_pbe`), and the sweep's own TOML
/// (`scripts/dosd2/runs/augccpvdz/n2o_rpa_pbe.toml`) is reproduced field for
/// field below.
const N2O_XYZ: &str = "3\nnitrous oxide linear N-N-O (rNN=1.128, rNO=1.184)\n\
                       N 0.000000 0.000000 0.000000\n\
                       N 0.000000 0.000000 1.128000\n\
                       O 0.000000 0.000000 -1.184000\n";

/// DOSD reference C6(N2O-N2O), a.u. Source: `scripts/dosd2/refs.json`
/// (Toulouse et al. arXiv:1305.0107 Table III, Meath-school DOSD) — the SAME
/// reference table the TS row in VALIDATION.md already cites. Not re-derived.
const N2O_C6_DOSD: f64 = 184.9;

/// Recorded ferric value from the 2026-07-20 dosd2 sweep
/// (`scripts/dosd2/results.json`, key `augccpvdz/n2o/rpa_pbe`): 156.472 a.u.,
/// i.e. -15.4% vs DOSD.
const N2O_C6_FERRIC_RECORDED: f64 = 156.472;

/// Tolerance on |C6_ferric - C6_DOSD| / C6_DOSD.
///
/// NOT a round number picked because it passes. Derivation:
///   * The ACTUAL observed agreement is -15.4% (156.472 vs 184.9), recorded
///     2026-07-20. That is the measurement.
///   * That residual is ferric's DOCUMENTED, systematic RPA@PBE/aug-cc-pVDZ
///     underbinding, not scatter: `docs/dosd-c6-rpa-vs-ts.md` reports a mean
///     aug-cc-pVDZ C6 bias of ~18.6% over the DOSD set, and the sibling test
///     `dispersion_c6.rs::anisotropic_c6_vs_kumar_meath` already uses a 30%
///     bar for exactly this quantity at exactly this level of theory.
///   * So the bar is set at the SAME 30% as that sibling test, which leaves
///     ~14.6 percentage points of headroom over the observed 15.4%. The guard
///     band is deliberately wide in the DIRECTION of the known bias and is
///     sized to the documented ~18.6% mean, not to today's single N2O digit:
///     pinning near 15.4% would make the test fail on ordinary SCF/quadrature
///     drift and on the basis-set sensitivity the bias itself has.
///
/// This test is therefore a BUG DETECTOR for the molecular-C6 construction
/// (a factor-of-2 convention slip, a lost 3/pi, a dropped weight, the pair
/// sum substituted for the molecular response), not a precision claim about
/// RPA@PBE. The precision question is the separate, already-documented
/// systematic-bias story.
const C6_REL_TOL: f64 = 0.30;

/// How far the run may drift from the RECORDED sweep value before we stop
/// believing this test reproduces the recorded measurement. Tighter than the
/// DOSD bar because it compares ferric to ITSELF at the same settings: only
/// SCF/quadrature noise and library drift separate them, not physics.
const RECORDED_REL_TOL: f64 = 0.02;

#[test]
#[ignore = "needs a real RPA@PBE run (~10-60s release): cargo test -p ferric-rpa --release --test c6_pair_sum_vs_dosd -- --ignored --nocapture"]
fn molecular_c6_matches_dosd_n2o() {
    let mol = Molecule::parse_xyz(N2O_XYZ, 0, 1).unwrap();
    let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("aug-cc-pvdz-rifit").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();

    let scf_cfg = RhfConfig {
        xc: Some("PBE".to_string()),
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        df_k_aux: Some("def2-universal-jkfit".to_string()),
        ..Default::default()
    };
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).unwrap();

    // Mirrors scripts/dosd2/runs/augccpvdz/n2o_rpa_pbe.toml exactly:
    // n_quad = 40, quadrature = "gauss-legendre", frozen_core = 0,
    // trunc_thresh = 0.0. The quadrature scheme matters — the crate default
    // is MiniMax, so leaving it out would NOT reproduce the recorded number.
    let cfg = PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 0.0,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 40,
            ..Default::default()
        },
        ..Default::default()
    };

    // c6_partition = "becke" in that TOML. NOTE this is deliberately NOT
    // C6Source::Pdep's default partition (Hirshfeld) — see the tag assertion
    // at the end.
    let partition = DispersionPartition::Becke;

    let dp =
        pdep_dynamic_polarizability(&mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg, partition, None)
            .unwrap();
    let res = casimir_polder_c6(&dp);

    let molecular = res.c6_molecular_iso;
    let pair_sum: f64 = res.c6_iso_pair.sum();

    println!("\n=== N2O molecular C6 (aug-cc-pVDZ, PDEP-RPA@PBE, Becke) ===");
    println!("  DOSD reference (arXiv:1305.0107 Tab. III) = {N2O_C6_DOSD:.1} a.u.");
    println!("  recorded dosd2 sweep (2026-07-20)         = {N2O_C6_FERRIC_RECORDED:.3} a.u.");
    println!("  this run: c6_molecular_iso                = {molecular:.3} a.u.");
    println!("  this run: c6_iso_pair.sum()  [WRONG]      = {pair_sum:.3} a.u.");
    println!(
        "  vs DOSD: {:+.1}%   vs recorded: {:+.2}%",
        100.0 * (molecular - N2O_C6_DOSD) / N2O_C6_DOSD,
        100.0 * (molecular - N2O_C6_FERRIC_RECORDED) / N2O_C6_FERRIC_RECORDED
    );

    assert!(
        molecular.is_finite() && molecular > 0.0,
        "molecular C6 must be finite and positive, got {molecular}"
    );

    // --- (c) THE OBSERVABLE: molecular C6 vs the DOSD reference. ---
    let rel = (molecular - N2O_C6_DOSD).abs() / N2O_C6_DOSD;
    assert!(
        rel < C6_REL_TOL,
        "N2O molecular C6 = {molecular:.3} a.u. vs DOSD {N2O_C6_DOSD:.1} a.u. \
         (rel err {:.1}%) exceeds the {:.0}% RPA@PBE/aug-cc-pVDZ bar. See \
         C6_REL_TOL for how that bar was derived (documented ~18.6% systematic \
         underbinding; observed here 15.4% on 2026-07-20).",
        100.0 * rel,
        100.0 * C6_REL_TOL
    );

    // --- Reproduces the RECORDED measurement, not just "some plausible
    // number in the window". Without this, a construction bug that happened
    // to land inside the 30% DOSD band would pass. ---
    let rel_rec = (molecular - N2O_C6_FERRIC_RECORDED).abs() / N2O_C6_FERRIC_RECORDED;
    assert!(
        rel_rec < RECORDED_REL_TOL,
        "N2O molecular C6 = {molecular:.3} a.u. drifted {:.2}% from the recorded \
         2026-07-20 dosd2 sweep value {N2O_C6_FERRIC_RECORDED:.3} a.u. (bar {:.0}%). \
         Either the molecular-C6 construction changed, or the sweep's recorded \
         number no longer describes this code path.",
        100.0 * rel_rec,
        100.0 * RECORDED_REL_TOL
    );

    // --- THE NAIVE CONSTRUCTION IS PINNED AS WRONG. `c6_iso_pair.sum()` is
    // what the CONSUMER WARNING on `C6Result` and on
    // `ferric_export::ml::DispersionBundle` tell consumers not to use. Here it
    // is asserted on REAL physics, not just structurally: the pair sum must
    // NOT be the DOSD-comparable quantity. If the two ever coincide, either
    // the molecular path has been (wrongly) rederived from the pair sum, or
    // the per-atom operator has silently changed definition. ---
    let naive_rel = (pair_sum - N2O_C6_DOSD).abs() / N2O_C6_DOSD;
    assert!(
        naive_rel > C6_REL_TOL,
        "c6_iso_pair.sum() = {pair_sum:.3} a.u. lands INSIDE the {:.0}% DOSD band \
         (rel err {:.1}%). The per-atom pair sum is not supposed to be a valid \
         molecular C6 (it omits inter-atomic coupling; measured -20% to -58% on \
         water). If this now agrees, the two constructions have converged and \
         the CONSUMER WARNING on C6Result/DispersionBundle is stale — \
         investigate before relaxing this assertion.",
        100.0 * C6_REL_TOL,
        100.0 * naive_rel
    );
    assert!(
        (pair_sum - molecular).abs() / molecular > 1e-3,
        "c6_iso_pair.sum() ({pair_sum:.4}) is numerically indistinguishable from \
         c6_molecular_iso ({molecular:.4}) — the molecular total has been \
         rederived from the per-atom pair sum, the exact wrong construction."
    );

    // --- The tag that would be exported alongside these numbers names the
    // partition this run ACTUALLY used, and is NOT the source's default
    // (C6Source::Pdep defaults to Hirshfeld) — so a tag produced by a
    // hardcoded default would be caught here. ---
    assert_eq!(
        partition.as_config_str(),
        "becke",
        "the NPZ c6_partition tag must name the partition that ran"
    );
    assert_ne!(
        partition,
        C6Source::Pdep.default_partition(),
        "this case deliberately uses the NON-default partition for its source, \
         so a hardcoded/defaulted tag cannot pass by coincidence"
    );
}
