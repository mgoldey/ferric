//! CSAM screening-ESTIMATE anchors, per this repo's "EXACTNESS ANCHOR FIRST"
//! experimental protocol.
//!
//! # CSAM is NOT a rigorous bound (2026-09-14 adjudication)
//!
//! This file originally asserted that CSAM never underestimates. That
//! assertion FAILED — correctly — and the failure is the finding: CSAM is a
//! non-rigorous *estimate*, not an upper bound. Confirmed against the full
//! text (Thompson's LMU dissertation, Publication I, reprinting Thompson &
//! Ochsenfeld, J. Chem. Phys. 147, 144101 (2017)): Eq. (8) CSB is the rigorous
//! bound, while Eq. (9)/(11)/(12) CSA1/CSA2/**CSAM** are explicitly labelled
//! non-rigorous. Psi4 ships CSAM. See `ferric_integrals::csam`'s module header
//! for that citation and for the independent numerical reproduction using
//! Psi4's own formula.
//!
//! Non-rigor is a property of the FORMULA, for every kernel including
//! Coulomb — not a per-operator hazard. What it costs, measured by the paper
//! in HF SCF energies: **-0.20 to +1.80 nanohartree** at `theta = 1e-12`
//! (Table V, aug-cc-pVDZ), comparable to the rigorous QQ bound's own
//! -1.50..+0.70 nH on the same systems; 0.05-9.35 microhartree at `1e-10`
//! (Table III, cc-pVDZ), roughly 2x QQ's. Threshold-controlled and small — but
//! with no guarantee, and with an error term that grows linearly with system
//! size at fixed threshold (Fig. 2). That last property, not any measured
//! blow-up, is why a rigorous bound is preferred wherever one is available.
//!
//! Operator coverage is therefore a ROUTING decision: Coulomb and `erf` are
//! estimated here; SHORT-RANGE `erfc` goes to the rigorous CSB bound, which
//! the paper's own conclusion recommends as the default for short-range
//! operators — see (d) below, which also records the correction to the
//! earlier, unsupported "1e8x / 218493x" justification for refusing both.
//!
//! Accordingly `CsamBounds` no longer implements the `Bound` trait, and the
//! validity assertions here have been replaced by CHARACTERIZATION of the
//! underestimate plus the error-vs-threshold test (k) that is the only
//! defensible justification for offering the mode at all.
//!
//! What is asserted here:
//!
//! (a) TRIVIAL LIMIT — at threshold 0, CSAM must retain every quartet plain
//!     Schwarz retains. Psi4 sidesteps this by forcing Schwarz at
//!     threshold == 0; this file checks it directly for ferric's own CSAM.
//! (b) NON-RIGOR, CHARACTERIZED — the worst `|true|/estimate` ratio is
//!     measured and railed (it is > 1, i.e. underestimating, by design), not
//!     forbidden. A ratio that silently became <= 1 would signal the
//!     refinement had gone inert and is itself flagged.
//! (c) TIGHTENING, NOT LOOSENING — `csam_estimate <= schwarz_estimate`
//!     everywhere, with a STRICT inequality demanded somewhere so an
//!     all-ones `X` table (CSAM degenerating to Schwarz) is caught.
//! (d) PER-OPERATOR ROUTING — short-range `erfc` is routed to the rigorous CSB
//!     bound at both the `csam_x_table` and `compute_for_screening` seams;
//!     long-range `erf` and Coulomb are ACCEPTED. Rigorous Schwarz stays
//!     available for all three. See that test's doc for the correction to the
//!     earlier blanket refusal (ratios without a magnitude filter, erfc formed
//!     by float64 subtraction, and `erf` never measured at all).
//! (e) MUTATION-RESISTANCE — folded into (c)'s strictness requirement and
//!     into (b)'s reachability check (see inline comments at each site).
//! (f) SCF ENERGY INVARIANCE — an RHF energy built with CSAM-screened LinK
//!     matches the Schwarz-screened one well inside the convergence gate.
//! (k) ERROR VS THRESHOLD — the CSAM-vs-Schwarz energy error must SHRINK as
//!     the integral threshold tightens. This is the load-bearing test for
//!     the estimate mode; see its own doc comment.
//!
//! ## The hot-path retrofit (2026-09-13)
//!
//! (a)-(f) above validate the BOUND (`CsamBounds`/`csam_x_table`) in
//! isolation and at the LinK seam. They do NOT prove CSAM screening reaches
//! the DEFAULT (`k_builder` unset) `DirectJ`/`DirectK`/`DirectJK` path — the
//! one the benzene/aug-cc-pVTZ perf profile that motivated CSAM actually
//! measured. Before this section, (f)'s own test constructs its RHF runs with
//! `k_builder: Some("link")`, so every one of these tests could pass while
//! CSAM was completely inert on the default path — exactly the "a passing
//! test measuring inertness" failure mode. (g)-(j) below close that gap by
//! exercising `SchwarzBounds::compute_for_screening` (which attaches the
//! `csam_x` refinement table `quartet_scatter::scatter_bra_pair` consults)
//! with `k_builder` LEFT UNSET.
//!
//! (g) MECHANISM ACTIVE ON THE DEFAULT PATH — CSAM must compute strictly
//!     fewer quartets than Schwarz at the same threshold, with `k_builder`
//!     unset. This is the test that would have caught the original gap: if
//!     `compute_for_screening`'s `csam_x` were never threaded through to
//!     `scatter_bra_pair` (e.g. a future edit accidentally dropped the
//!     parameter at one of the five call sites), quartet counts would be
//!     identical and this test would fail.
//! (h) ENERGY INVARIANCE ON THE DEFAULT PATH — same comparison as (f), but
//!     through `DirectJK` rather than LinK, on a real (multi-atom) system.
//! (i) EXACTNESS ANCHOR — `screening = "schwarz"` (the default `ScreeningKind`
//!     AND the default of `compute_for_screening`) must be BYTE-IDENTICAL
//!     (not just numerically close) to a plain `SchwarzBounds::compute` +
//!     `RhfConfig::default()` run — i.e. this retrofit changes nothing for
//!     anyone who does not opt in.
//! (j) THREAD-COUNT BIT-IDENTITY — CSAM-screened `DirectJK::build` must
//!     remain bit-identical across `RAYON_NUM_THREADS`, matching the existing
//!     guarantee for the Schwarz-screened path (see `reduce.rs`'s fold-order
//!     contract and `direct_builders_bit_identical_across_thread_counts` in
//!     `direct_jk.rs`). The CSAM refinement factor is a pure function of
//!     `(s1,s2,s3,s4)` and the precomputed `csam_x` table — it does not touch
//!     the shell-pair work list, the group partition, or the fold order, so
//!     this is expected to hold by construction; this test is the regression
//!     guard for that claim.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::pairs::SignificantPairs;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
// No `Bound` import: `CsamBounds` deliberately does NOT implement that trait
// (it underestimates -- see the header and the note at the call site below).
use ferric_scf::screening::{CsamBounds, ScreeningKind, SchwarzBounds};

fn load_mol(stem: &str) -> Molecule {
    Molecule::load_xyz(&format!("../../testdata/molecules/{stem}.xyz"))
        .unwrap_or_else(|e| panic!("cannot load {stem}.xyz: {e}"))
}

/// ---------------------------------------------------------------------
/// (a) TRIVIAL LIMIT: CSAM at threshold 0 must retain every quartet plain
/// Schwarz retains at threshold 0.
///
/// `SignificantPairs::build` uses a strict `estimate(..) > threshold`
/// comparison (see `schwarz.rs`'s trivial-limit discussion), so this is
/// checked at the PAIR-LIST level, which is what every screening consumer
/// (LinK, DensityPairs) actually gates on. If CSAM's refinement factor ever
/// stored an invalid (spuriously small, e.g. underflowed) entry, some pair
/// Schwarz retains at threshold 0 would be dropped by CSAM here — this is
/// exactly the failure mode `SCHWARZ_TABLE_PRECISION`/`SCHWARZ_Q_FLOOR` exist
/// to prevent on the Schwarz side, and `csam.rs`'s own precision/floor
/// choices exist to prevent on this side.
/// ---------------------------------------------------------------------
#[test]
fn csam_matches_schwarz_pairlist_at_threshold_zero() {
    let mol = load_mol("alkane_8");
    let bs = basis::bundled("cc-pvdz").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("PreparedBasis");
    let nsh = prep.nshells();
    let op = Operator::coulomb();

    let schwarz = SchwarzBounds::compute(op, &prep).expect("Schwarz bounds");
    let csam = CsamBounds::compute(op, &prep).expect("CSAM bounds");

    let sp_schwarz = SignificantPairs::build(&schwarz, nsh, 0.0);
    // `CsamBounds` deliberately does NOT implement `Bound` (it underestimates;
    // see the file header), so the pair list is built through the same
    // `LinkBound::Csam` wrapper production uses — which is also the only place
    // the non-rigorous estimate is allowed to enter bound-typed code.
    let csam_as_bound = ferric_scf::screening::LinkBound::Csam(csam);
    let sp_csam = SignificantPairs::build(&csam_as_bound, nsh, 0.0);

    assert_eq!(
        sp_schwarz.total_pairs(),
        sp_csam.total_pairs(),
        "at thresh=0 CSAM retained {} ordered shell pairs vs Schwarz's {} — the trivial limit \
         (screening does nothing) is violated for CSAM",
        sp_csam.total_pairs(),
        sp_schwarz.total_pairs()
    );

    // Both must ALSO retain the full square — otherwise this would just be
    // "CSAM agrees with a Schwarz that is itself already broken at thresh=0"
    // (see schwarz.rs's own trivial-limit test for why that distinction
    // matters).
    let full_square = nsh * nsh;
    assert_eq!(
        sp_schwarz.total_pairs(),
        full_square,
        "Schwarz itself does not retain the full pair list at thresh=0 on alkane_8/cc-pVDZ; \
         this system was chosen because the Schwarz side is known-good here \
         (schwarz_table_never_stores_a_zero_alkane_8), so a failure here means the harness \
         itself regressed, not CSAM"
    );
}

/// A representative sample of shell quartets: all `(i,j|i,j)`-style diagonal
/// combinations plus a spread of off-diagonal ones, without the full O(nsh^4)
/// loop qqr.rs's water-scale test uses (this file also runs on the larger,
/// diffuse aug-cc-pVDZ water case, where nsh^4 would be considerably more
/// expensive per quartet).
fn sample_quartets(nsh: usize) -> Vec<(usize, usize, usize, usize)> {
    let mut v = Vec::new();
    for i in 0..nsh {
        for j in 0..=i {
            for k in 0..=i {
                for l in 0..=k {
                    v.push((i, j, k, l));
                }
            }
        }
    }
    v
}

/// Shared body for (b) bound validity and (c) tightening, for one
/// (system, basis, operator) triple. Returns
/// `(worst_true_over_bound, found_strictly_tighter)` so callers can assert
/// on both without recomputing.
fn check_validity_and_tightening(path: &str, basis_name: &str, op: Operator) -> (f64, bool) {
    let mol = Molecule::load_xyz(path).unwrap_or_else(|e| panic!("cannot load {path}: {e}"));
    let bs = basis::bundled(basis_name).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("PreparedBasis");
    let nsh = prep.nshells();

    let schwarz = SchwarzBounds::compute(op, &prep).expect("Schwarz bounds");
    let csam = CsamBounds::compute(op, &prep).expect("CSAM bounds");

    // Tight engine precision so libint does not prescreen a small-but-real
    // quartet to nothing and hand us a spuriously "satisfied" bound (same
    // precaution qqr.rs's validity test takes).
    let mut eng = ferric_integrals::engine::Engine::new_2e(op, &prep, 1e-30)
        .expect("tight-precision engine");

    let mut worst_true_over_bound = 0.0f64;
    let mut found_strictly_tighter = false;
    let mut any_engaged = false; // (e) reachability: the factor must ever be < 1 somewhere sampled.

    for (i, j, k, l) in sample_quartets(nsh) {
        let bound = csam.estimate_nonrigorous(i, j, k, l);
        let s = schwarz.estimate(i, j, k, l);
        let tru = match eng.compute_quartet(&prep, i, j, k, l) {
            Some(block) => block.iter().fold(0.0f64, |m, v| m.max(v.abs())),
            None => 0.0,
        };

        // --- (c) tightening: CSAM must never exceed Schwarz -------------
        assert!(
            bound <= s + 1e-12 * s.max(1.0),
            "{path}/{basis_name} {:?} ({i},{j},{k},{l}): CSAM = {bound:.6e} > Schwarz = {s:.6e} \
             — CSAM must only ever TIGHTEN the bound, never loosen it",
            op.kind
        );
        if s > 1e-12 && bound < s * 0.999 {
            found_strictly_tighter = true;
        }
        if bound < s * 0.999 {
            any_engaged = true;
        }

        // --- (b) NON-RIGOR IS EXPECTED: CSAM is an estimate, not a bound,
        // and DOES underestimate (see `ferric_integrals::csam`'s header for
        // the citation and the independent numerical reproduction). There is
        // therefore NO assertion here that `bound >= true` — such an
        // assertion failed, correctly, and its failure is what established
        // that CSAM is not a Coulomb upper bound. What IS measured is the
        // worst underestimate ratio, returned to the caller so the
        // characterization tests below can pin how large it gets.
        let floor = s.min(tru);
        if floor > 1e-14 && bound > 0.0 {
            let ratio = floor / bound;
            if ratio > worst_true_over_bound {
                worst_true_over_bound = ratio;
            }
        }
    }

    assert!(
        any_engaged,
        "{path}/{basis_name} {:?}: CSAM's refinement factor was never < 1 on ANY sampled \
         quartet — either the system/basis is too small to exercise it, or the X table has \
         collapsed to all-ones (which would make CSAM silently degenerate to plain Schwarz)",
        op.kind
    );

    (worst_true_over_bound, found_strictly_tighter)
}

/// (b) NON-RIGOR, DOCUMENTED: on water/cc-pVDZ, Coulomb, CSAM is known to
/// underestimate. This test PINS that fact rather than forbidding it: if the
/// estimate ever became a true upper bound (worst ratio <= 1), that would be
/// a meaningful change in behavior — CSAM is not supposed to be rigorous, and
/// silently becoming so would mean the refinement had gone inert (an
/// all-ones X table degenerates CSAM to plain Schwarz, which IS rigorous and
/// would pass a naive `<= 1` bar while proving nothing).
///
/// The specific quartet that first exposed this was (1,0|0,0) — all shells
/// oxygen-centered, i.e. zero bra-ket separation, where a distance-decay
/// refinement has no justification.
#[test]
fn csam_underestimates_on_water_coulomb_as_expected_for_a_nonrigorous_estimate() {
    let (worst_ratio, _) =
        check_validity_and_tightening("../../testdata/molecules/water.xyz", "cc-pvdz", Operator::coulomb());
    eprintln!("water/cc-pVDZ Coulomb: worst |true|/estimate = {worst_ratio:.4}");
    assert!(
        worst_ratio > 1.0,
        "water/cc-pVDZ Coulomb: worst |true|/estimate = {worst_ratio} <= 1, i.e. CSAM behaved as \
         a rigorous bound here. That contradicts the documented non-rigor and most likely means \
         the refinement went inert (all-ones X table => CSAM == Schwarz). Investigate before \
         relaxing this assertion."
    );
    // Upper sanity rail: a ~2x underestimate is consistent with the
    // independent reproduction (1.39-2.53 across geometries). An order of
    // magnitude worse would indicate a NEW defect on top of the known
    // non-rigor, not the documented behavior.
    assert!(
        worst_ratio < 10.0,
        "water/cc-pVDZ Coulomb: worst |true|/estimate = {worst_ratio} is far beyond the ~2x \
         underestimate the CSAM construction is known to produce — suspect a separate defect"
    );
}

/// (b) BOUND VALIDITY with a DIFFUSE basis (aug-cc-pVDZ), where Cauchy-Schwarz
/// bounds are known to be looser and therefore a better stress test for a
/// refinement that could over-tighten into invalidity (the exact landmine the
/// task brief calls out: a square-root bookkeeping error would show up here
/// first, since diffuse shells have the widest dynamic range in the X ratios).
#[test]
fn csam_underestimate_is_bounded_on_water_aug_cc_pvdz_coulomb() {
    let (worst_ratio, found_tighter) = check_validity_and_tightening(
        "../../testdata/molecules/water.xyz",
        "aug-cc-pvdz",
        Operator::coulomb(),
    );
    eprintln!("water/aug-cc-pVDZ Coulomb: worst |true|/estimate = {worst_ratio:.4}");
    // Diffuse basis: the widest dynamic range in the X ratios, so the worst
    // underestimate here is the stress-test value. Recorded, not forbidden.
    assert!(
        worst_ratio < 10.0,
        "water/aug-cc-pVDZ Coulomb: worst |true|/estimate = {worst_ratio} far exceeds the ~2x \
         underestimate CSAM is known to produce — suspect a defect beyond the documented non-rigor"
    );
    assert!(
        found_tighter,
        "aug-cc-pVDZ: CSAM was never strictly tighter than Schwarz anywhere — an all-ones X \
         table would produce exactly this symptom (see the mutation-resistance note on (c))"
    );
}

/// (c) TIGHTENING, explicit standalone assertion (the per-quartet check
/// inside `check_validity_and_tightening` already enforces `<=` everywhere;
/// this test isolates the STRICT half so a regression to "CSAM == Schwarz
/// always" fails here specifically, independent of (b)'s validity bar).
///
/// (e) MUTATION-RESISTANCE: if `ferric_integrals::csam::csam_x_table` were
/// mutated to always return an all-ones table (X collapses to the trivial
/// Cauchy-Schwarz maximum), `refinement_factor` would be `sqrt(1*1) = 1`
/// everywhere and CSAM would be bit-identical to Schwarz — `found_tighter`
/// below would be `false` and this test would fail. This is the test that
/// specific mutant is designed to be caught by.
#[test]
fn csam_is_strictly_tighter_than_schwarz_somewhere_benzene() {
    let (_, found_tighter) =
        check_validity_and_tightening("../../testdata/molecules/benzene.xyz", "cc-pvdz", Operator::coulomb());
    assert!(
        found_tighter,
        "benzene/cc-pVDZ: CSAM was never strictly tighter than Schwarz anywhere sampled — \
         either the composition is a no-op (all-ones X table mutant) or the sample is too small"
    );
}

/// (d) PER-OPERATOR: `erfc` is ROUTED TO CSB; `erf` and Coulomb are accepted.
///
/// # History, and a correction (2026-09-14)
///
/// An earlier revision of this test refused BOTH attenuated operators, citing
/// a `218493x` "underestimate ratio" on benzene/cc-pVDZ at omega=1.0 and a
/// Psi4-formula chain measurement running `5.3x -> 49x -> 1.9e3 -> 2.7e5 ->
/// 1.0e8`. Those figures do not support the conclusion drawn from them:
///
/// 1. **They are ratios, not energy errors — and no energy error was ever
///    computed.** The harness (`scripts/csam_bound_validity_proto.py`)
///    maximizes `true/estimate` over every quartet with no magnitude filter.
///    A ratio grows without bound; an energy error does not. (Honest scope:
///    adding the paper's own `1e-12` filter leaves those figures unchanged, so
///    this gap is real but is not what produced them — point 2 is.)
/// 2. **The big rows are float64 cancellation.** The harness forms erfc as
///    `Coulomb - erf` in double precision — the exact instability the paper
///    warns about on PDF p. 66. Re-evaluated at 200-bit precision, the
///    harness's own `true` value at the 6-center geometry (13.2 bohr) is wrong
///    by **447%**, and by 3.2e-6 at the 5-center one. Those rows measure
///    IEEE754.
/// 3. **`erf` was never measured at all** — the harness has no `erf` path —
///    and `erf` is the one attenuated kernel Psi4 screens with CSAM in
///    production (`Libint2ErfERI` builds compute AND sieve engines on
///    `libint2::Operator::erf_coulomb`, `psi4/src/psi4/libmints/eri.cc:179,182`).
///
/// # What this test pins now
///
/// `erfc` (SHORT-range) routes to the rigorous CSB bound. That is a routing
/// choice grounded in the paper's own conclusion — CSB "could already be
/// considered the new default estimate for short-range operators", and under
/// erfc it reaches a 2.5x speedup at threshold 1e-14 with virtually no error
/// increase — not a claim that CSAM was measured dangerous there. `erf`
/// (LONG-range, decaying as 1/r like Coulomb) is accepted.
///
/// CSAM remains NON-RIGOROUS for every kernel including Coulomb. Its measured
/// HF energy error is -0.20..+1.80 nanohartree at `theta = 1e-12` (paper Table
/// V) and 0.05-9.35 microhartree at `1e-10` (Table III) — threshold-controlled
/// and comparable to the rigorous QQ bound's own error, but with no guarantee
/// and a term that grows linearly with system size (Fig. 2).
#[test]
fn csam_routes_erfc_to_csb_and_accepts_erf_at_the_scf_seam() {
    let mol = load_mol("water");
    let bs = basis::bundled("cc-pvdz").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("PreparedBasis");

    // erfc: routed away, at BOTH seams, rather than silently degrading.
    let erfc = Operator::erfc(1.0);
    assert!(
        CsamBounds::compute(erfc, &prep).is_err(),
        "CsamBounds::compute must route the short-range erfc kernel to CSB"
    );
    assert!(
        SchwarzBounds::compute_for_screening(erfc, &prep, ScreeningKind::Csam).is_err(),
        "compute_for_screening(.., Csam) must propagate the erfc routing rather than silently \
         returning a bounds object with no csam_x table attached (which would degrade to plain \
         Schwarz without telling the caller)"
    );

    // erf: ACCEPTED at both seams. This is the arm that changed; it is asserted
    // positively so a regression to blanket refusal fails here.
    let erf = Operator::erf(1.0);
    assert!(
        CsamBounds::compute(erf, &prep).is_ok(),
        "CsamBounds::compute must ACCEPT erf — it is long-range (1/r decay, like Coulomb) and is \
         Psi4's own production CSAM kernel. The previous refusal was never measured."
    );
    assert!(
        SchwarzBounds::compute_for_screening(erf, &prep, ScreeningKind::Csam).is_ok(),
        "compute_for_screening(.., Csam) must attach an X table for erf"
    );

    // Controls: the rigorous path must still accept both attenuated operators
    // (it is the alternative the erfc message points at), and CSAM must still
    // work for Coulomb — otherwise the assertions above could pass because
    // everything is uniformly broken or uniformly permissive.
    for op in [erfc, erf] {
        assert!(
            SchwarzBounds::compute_for_screening(op, &prep, ScreeningKind::Schwarz).is_ok(),
            "plain Schwarz (rigorous) must remain available for {:?}",
            op.kind
        );
    }
    assert!(
        CsamBounds::compute(Operator::coulomb(), &prep).is_ok(),
        "CSAM must still work for Coulomb — a blanket failure would make this test vacuous"
    );
}

/// ---------------------------------------------------------------------
/// (f) SCF ENERGY INVARIANCE: an RHF energy computed with `k_builder =
/// "link"` + `screening = "csam"` must match the same run with
/// `screening = "schwarz"` well inside the SCF convergence threshold.
///
/// This is a CORRECTNESS anchor, not a benefit measurement: CSAM screens
/// MORE aggressively than Schwarz at a fixed threshold (a strictly tighter
/// bound survives to a smaller retained set at the same cutoff), so some
/// numerical difference between the two runs is expected and acceptable —
/// what this test rules out is a GROSS divergence (wrong-basin SCF, a
/// mis-wired factor, or an invalid bound silently discarding a
/// non-negligible contribution).
///
/// Uses a tight `integral_thresh` (matching production, `1e-14`) on a small
/// molecule so the screening residual itself is tiny for BOTH bounds and the
/// comparison isolates "did wiring CSAM through solve_rhf change anything
/// structural", not "how much does screening error differ between the two
/// bounds at a loose threshold" (that question belongs to a cost/benefit
/// sweep, not a correctness anchor, and is explicitly out of scope here —
/// see `RhfConfig::screening`'s doc for why only the LinK path is wired at
/// all in this change).
/// ---------------------------------------------------------------------
#[test]
fn csam_screening_rhf_energy_matches_schwarz() {
    let mol = load_mol("alkane_4");
    let bs = basis::bundled("cc-pvdz").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("PreparedBasis");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("Schwarz bounds");
    let ctx = ParallelContext::default();

    let base = RhfConfig {
        density_conv: 1e-9,
        integral_thresh: 1e-14,
        k_builder: Some("link".to_string()),
        ..Default::default()
    };

    let cfg_schwarz = RhfConfig { screening: ScreeningKind::Schwarz, ..base.clone() };
    let cfg_csam = RhfConfig { screening: ScreeningKind::Csam, ..base };

    let res_schwarz = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_schwarz).expect("RHF (Schwarz) solve");
    let res_csam = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_csam).expect("RHF (CSAM) solve");

    assert!(res_schwarz.converged, "Schwarz-screened LinK RHF did not converge");
    assert!(res_csam.converged, "CSAM-screened LinK RHF did not converge");

    let de = (res_schwarz.energy - res_csam.energy).abs();
    eprintln!(
        "alkane_4/cc-pVDZ LinK: E(Schwarz) = {:.12} Ha, E(CSAM) = {:.12} Ha, |dE| = {de:.3e} Ha",
        res_schwarz.energy, res_csam.energy
    );
    // Two orders of magnitude inside density_conv's own scale (1e-9): a real
    // wiring defect (wrong table, unattenuated cross-operator reuse, an
    // inverted factor) would show up as a change many orders larger than
    // this, not a borderline miss.
    assert!(
        de < 1e-7,
        "CSAM-screened LinK RHF energy {:.12} deviates from Schwarz-screened {:.12} by \
         {de:.3e} Ha, exceeding the 1e-7 bar — this is more than a screening-residual-scale \
         difference and indicates a wiring defect, not benign approximation noise",
        res_csam.energy,
        res_schwarz.energy
    );
}

/// Serializes every test below that mutates `FERRIC_SCF_INCREMENTAL` — this
/// is a SEPARATE integration-test binary from `rhf.rs`'s own `#[cfg(test)]`
/// module (its `ENV_LOCK` is a private item of that module and not reachable
/// here), but `cargo test` still runs the `#[test]` functions WITHIN this one
/// binary concurrently by default, and the env var is process-global. Every
/// test that touches `FERRIC_SCF_INCREMENTAL` in this file must hold this
/// lock for the full set-var .. solve .. remove-var span.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Solve RHF on the DEFAULT (`k_builder` unset, `DirectJK`) path with
/// full-rebuild-every-iteration Fock builds (incremental OFF), so
/// `computed_quartets` is a clean function of the screening bound alone —
/// not confounded by the incremental scheme's density-driven screen or by any
/// difference in iteration count between the Schwarz and CSAM runs.
fn solve_rhf_direct_full_rebuild(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
) -> ferric_scf::result::ScfResult {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("FERRIC_SCF_INCREMENTAL", "0");
    let result = solve_rhf(ctx, mol, prep, op, bounds, config);
    std::env::remove_var("FERRIC_SCF_INCREMENTAL");
    result.expect("RHF solve failed")
}

/// ---------------------------------------------------------------------
/// (g) MECHANISM ACTIVE ON THE DEFAULT PATH: with `k_builder` UNSET (the
/// benchmarked default — `DirectJK`, not LinK), CSAM screening must compute
/// STRICTLY FEWER quartets than Schwarz screening at the same threshold.
///
/// This is the load-bearing regression guard for the whole retrofit: before
/// `SchwarzBounds::compute_for_screening` existed and `csam_x` was threaded
/// through `quartet_scatter::scatter_bra_pair`'s five call sites, this
/// comparison would have shown IDENTICAL quartet counts (CSAM's own tests all
/// used `k_builder = "link"` and so never exercised this path) — that is
/// exactly the "passing test measuring inertness" failure mode the task
/// brief calls out. If a future edit re-introduces that gap (e.g. drops
/// `csam_x` at one of `direct_j.rs`/`direct_k.rs`/`direct_jk.rs` (x2)
/// /`rhf.rs::build_jk_with_pool`, or `compute_for_screening` stops attaching
/// the table), the two counts collapse back to equal and this test fails.
///
/// `alkane_8`/cc-pVDZ is used (rather than the smaller alkane_4/cc-pVDZ (f)
/// uses) because a larger system gives the bra-pair screen more shell pairs
/// to discriminate between — this is the same system
/// `csam_matches_schwarz_pairlist_at_threshold_zero` above already uses for
/// its own known-good Schwarz-at-threshold-zero baseline, so a failure here
/// cannot be blamed on an unfamiliar system/basis combination.
/// ---------------------------------------------------------------------
#[test]
fn csam_screening_computes_fewer_quartets_than_schwarz_on_default_path() {
    let mol = load_mol("alkane_8");
    let bs = basis::bundled("cc-pvdz").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("PreparedBasis");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();

    let bounds_schwarz = SchwarzBounds::compute_for_screening(op, &prep, ScreeningKind::Schwarz)
        .expect("Schwarz bounds");
    let bounds_csam = SchwarzBounds::compute_for_screening(op, &prep, ScreeningKind::Csam)
        .expect("CSAM bounds");
    assert!(
        bounds_csam.csam_x.is_some(),
        "compute_for_screening(.., Csam) must attach a csam_x table"
    );
    assert!(
        bounds_schwarz.csam_x.is_none(),
        "compute_for_screening(.., Schwarz) must NOT attach a csam_x table (default must stay \
         byte-identical to plain SchwarzBounds::compute)"
    );

    // A loose-ish integral threshold so screening actually discards a
    // meaningful fraction of quartets on this system/basis (too tight a
    // threshold would retain nearly everything under EITHER bound and this
    // comparison would degenerate to "both compute the full list").
    let config = RhfConfig {
        integral_thresh: 1e-9,
        density_conv: 1e-7,
        k_builder: None, // the DEFAULT path — DirectJK, not LinK.
        ..Default::default()
    };

    let res_schwarz =
        solve_rhf_direct_full_rebuild(&ctx, &mol, &prep, op, &bounds_schwarz, &config);
    let res_csam = solve_rhf_direct_full_rebuild(&ctx, &mol, &prep, op, &bounds_csam, &config);

    assert!(res_schwarz.converged, "Schwarz-screened default-path RHF did not converge");
    assert!(res_csam.converged, "CSAM-screened default-path RHF did not converge");

    eprintln!(
        "alkane_8/cc-pVDZ DirectJK (full rebuild): Schwarz computed_quartets = {}, \
         CSAM computed_quartets = {}",
        res_schwarz.computed_quartets, res_csam.computed_quartets
    );
    assert!(
        res_csam.computed_quartets < res_schwarz.computed_quartets,
        "CSAM screening on the DEFAULT (k_builder unset) path computed {} quartets, not fewer \
         than Schwarz's {} — the CSAM refinement is not reaching quartet_scatter::scatter_bra_pair \
         on this path (the exact gap this retrofit exists to close)",
        res_csam.computed_quartets,
        res_schwarz.computed_quartets
    );
}

/// ---------------------------------------------------------------------
/// (h) ENERGY INVARIANCE ON THE DEFAULT PATH: the same comparison as (f)
/// above, but through `DirectJK` (via `compute_for_screening`) rather than
/// LinK — the actual path (g) just proved CSAM changes the quartet count on.
/// ---------------------------------------------------------------------
#[test]
fn csam_screening_rhf_energy_matches_schwarz_on_default_path() {
    let mol = load_mol("alkane_4");
    let bs = basis::bundled("cc-pvdz").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("PreparedBasis");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();

    let bounds_schwarz = SchwarzBounds::compute_for_screening(op, &prep, ScreeningKind::Schwarz)
        .expect("Schwarz bounds");
    let bounds_csam = SchwarzBounds::compute_for_screening(op, &prep, ScreeningKind::Csam)
        .expect("CSAM bounds");

    let config = RhfConfig {
        density_conv: 1e-9,
        integral_thresh: 1e-14,
        k_builder: None, // the DEFAULT path.
        ..Default::default()
    };

    let res_schwarz = solve_rhf(&ctx, &mol, &prep, op, &bounds_schwarz, &config)
        .expect("RHF (Schwarz, default path) solve");
    let res_csam = solve_rhf(&ctx, &mol, &prep, op, &bounds_csam, &config)
        .expect("RHF (CSAM, default path) solve");

    assert!(res_schwarz.converged, "Schwarz-screened default-path RHF did not converge");
    assert!(res_csam.converged, "CSAM-screened default-path RHF did not converge");

    let de = (res_schwarz.energy - res_csam.energy).abs();
    eprintln!(
        "alkane_4/cc-pVDZ DirectJK: E(Schwarz) = {:.12} Ha, E(CSAM) = {:.12} Ha, |dE| = {de:.3e} Ha",
        res_schwarz.energy, res_csam.energy
    );
    assert!(
        de < 1e-7,
        "CSAM-screened default-path RHF energy {:.12} deviates from Schwarz-screened {:.12} by \
         {de:.3e} Ha, exceeding the 1e-7 bar — this is more than a screening-residual-scale \
         difference and indicates a wiring defect, not benign approximation noise",
        res_csam.energy,
        res_schwarz.energy
    );
}

/// ---------------------------------------------------------------------
/// (i) EXACTNESS ANCHOR: `ScreeningKind::Schwarz` (the default variant AND
/// the default of `compute_for_screening`) must be BYTE-IDENTICAL to plain
/// `SchwarzBounds::compute` — not just numerically close. This is the anchor
/// that proves the retrofit changes NOTHING for the untouched default
/// configuration: no new field access, no new branch outcome, no new
/// floating-point operation on the path anyone not opting into `"csam"`
/// exercises.
/// ---------------------------------------------------------------------
#[test]
fn schwarz_screening_default_path_is_byte_identical_to_pre_csam_behavior() {
    let mol = load_mol("alkane_4");
    let bs = basis::bundled("cc-pvdz").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("PreparedBasis");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();

    let bounds_plain = SchwarzBounds::compute(op, &prep).expect("plain Schwarz bounds");
    let bounds_via_kind =
        SchwarzBounds::compute_for_screening(op, &prep, ScreeningKind::Schwarz)
            .expect("compute_for_screening(Schwarz) bounds");
    assert!(bounds_via_kind.csam_x.is_none());
    assert_eq!(
        bounds_plain.q, bounds_via_kind.q,
        "compute_for_screening(.., Schwarz) must produce the identical q table as compute()"
    );

    let config = RhfConfig::default();
    let res_plain = solve_rhf(&ctx, &mol, &prep, op, &bounds_plain, &config)
        .expect("RHF (plain SchwarzBounds::compute) solve");
    let res_via_kind = solve_rhf(&ctx, &mol, &prep, op, &bounds_via_kind, &config)
        .expect("RHF (compute_for_screening Schwarz) solve");

    assert!(res_plain.converged && res_via_kind.converged);
    assert_eq!(
        res_plain.energy.to_bits(),
        res_via_kind.energy.to_bits(),
        "default screening must be BYTE-IDENTICAL (not just numerically close) between plain \
         SchwarzBounds::compute and compute_for_screening(.., Schwarz) — any difference means \
         this retrofit perturbed the untouched default path"
    );
    assert_eq!(
        res_plain.computed_quartets, res_via_kind.computed_quartets,
        "default screening must retain the identical quartet count"
    );
}

/// ---------------------------------------------------------------------
/// (j) THREAD-COUNT BIT-IDENTITY WITH CSAM ACTIVE: `DirectJK::build`'s J/K
/// output must be bit-identical across `RAYON_NUM_THREADS` when CSAM
/// screening is active, matching the pre-existing Schwarz-only guarantee
/// (`direct_builders_bit_identical_across_thread_counts`,
/// `crates/ferric-scf/src/direct_jk.rs`). The shell-pair work list, the
/// deterministic group partition, and the per-group fold order are all pure
/// functions of `(nsh, thresh, d_max_shell)` — `csam_x` only changes WHICH
/// candidates a bra pair's inner loop admits, never the outer work list or
/// the reduction structure — so this is expected to hold by construction;
/// this test is the regression guard for that claim, exercised directly
/// against `DirectJK` (bypassing the SCF loop, matching how the existing
/// Schwarz-only thread-count guards in `direct_jk.rs` are structured).
/// ---------------------------------------------------------------------
#[test]
fn csam_screening_direct_jk_bit_identical_across_thread_counts() {
    use ferric_scf::direct_jk::DirectJK;

    let mol = load_mol("alkane_4");
    let bs = basis::bundled("cc-pvdz").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("PreparedBasis");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute_for_screening(op, &prep, ScreeningKind::Csam)
        .expect("CSAM bounds");
    assert!(bounds.csam_x.is_some(), "test requires CSAM to actually be active");
    let n = prep.nbasis();

    // Dense symmetric density so every surviving quartet actually contributes
    // (matches the existing thread-count guards' density construction).
    let mut d = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            d[(i, j)] = 0.01 * ((i * 7 + j * 3) % 11) as f64;
        }
    }
    let d = 0.5 * (&d + &d.t());

    let run = |threads: usize| -> (ndarray::Array2<f64>, ndarray::Array2<f64>, usize) {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            let ctx = ParallelContext::default();
            let mut j = ndarray::Array2::zeros((n, n));
            let mut k = ndarray::Array2::zeros((n, n));
            let mut djk = DirectJK::new(&ctx, &prep, &bounds, 1e-14, usize::MAX);
            let cq = djk.build(&d, &mut j, &mut k).unwrap();
            (j, k, cq)
        })
    };
    let r1 = run(1);
    let r4 = run(4);
    assert_eq!(r1.2, r4.2, "CSAM-screened quartet count must be identical across thread counts");
    assert_eq!(r1.0, r4.0, "CSAM-screened DirectJK J must be bit-identical across thread counts");
    assert_eq!(r1.1, r4.1, "CSAM-screened DirectJK K must be bit-identical across thread counts");
}

/// ---------------------------------------------------------------------
/// (k) ERROR-VS-THRESHOLD CHARACTERIZATION — the test that justifies
/// shipping a NON-RIGOROUS estimate at all.
///
/// CSAM is not an upper bound (see `ferric_integrals::csam`'s module header
/// for the citation and the independent numerical reproduction). The ONLY
/// defensible reason to offer it is the claim the 2017 abstract makes for
/// the non-rigorous estimate: that its errors are "completely controllable
/// through the integral screening threshold". That claim is MEASURABLE, and
/// this test measures it instead of asserting it.
///
/// Method: run the same molecule on the default (DirectJK) path at a
/// sequence of tightening integral thresholds, once under rigorous Schwarz
/// and once under CSAM, and compare the converged RHF energies. The
/// CSAM-vs-Schwarz energy difference is the error the non-rigorous estimate
/// introduces. If the estimate is usable, that difference must FALL as the
/// threshold tightens — and in particular must be smaller at the tightest
/// threshold than at the loosest.
///
/// Why an INVARIANT (error decreases) rather than a point bar (error < X):
/// per this repo's experimental protocol, a point bar can pass because the
/// mechanism is inert in that regime, whereas a monotone-decrease
/// requirement can only pass if the threshold actually controls the error.
/// A CSAM whose error were threshold-INDEPENDENT (the failure mode that
/// would make it unshippable) fails this test while comfortably passing any
/// single loose bar.
///
/// Mutation-resistance: if `csam_x` were forced to an all-ones table, CSAM
/// would degenerate to Schwarz, every difference would be ~0, and the
/// "estimate is actually engaged" assertion below (nonzero error at the
/// LOOSEST threshold) would fail — so this test cannot pass by inertness.
/// ---------------------------------------------------------------------
#[test]
fn csam_energy_error_shrinks_with_the_screening_threshold() {
    let mol = load_mol("alkane_4");
    let bs = basis::bundled("cc-pvdz").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("PreparedBasis");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();

    // Loose -> tight. The loosest is deliberately loose enough that the
    // non-rigorous estimate discards real weight and a measurable error
    // appears; the tightest is production-scale.
    let thresholds = [1e-6, 1e-8, 1e-10, 1e-12];

    let mut errors = Vec::new();
    // MEASURED INTERACTION (2026-09-14): with the libint2 ShellPair cache
    // ACTIVE, the CSAM run does not converge at `thresh = 1e-6` on this system,
    // while Schwarz does and while CSAM itself converges at every tighter
    // threshold. Diagnosed by bisection, not guessed:
    //
    //   * the test file is byte-identical to the pre-merge CSAM branch, where
    //     all four thresholds pass;
    //   * `FERRIC_SHELLPAIR_CACHE=0` on THIS branch reproduces the pre-merge
    //     numbers exactly (same energies AND same quartet counts), so the
    //     screening merge is not implicated;
    //   * the cache is deliberately NOT bit-identical (191 ULP / ~3.8e-14
    //     relative per integral — libint2 rebuilds a `nullptr` shell pair in
    //     its swapped canonical order but negates a PRECOMPUTED pair's `AB`).
    //
    // A ~1e-14 integral perturbation is far below any production threshold, but
    // at 1e-6 it flips screening decisions, and CSAM at that threshold keeps
    // only ~half the quartets Schwarz does (13.9M vs 26.6M) — the least
    // redundant configuration in the sweep, hence the one that stalls DIIS
    // short of the 1e-10 density target.
    //
    // The 1e-6 point is retained because the SHRINKING claim is what this test
    // exists to check and a loose anchor makes it meaningful; the cache is
    // pinned off so the assertion measures CSAM's threshold behaviour rather
    // than a cache-vs-screening interaction at a threshold nobody runs.
    let _cache_guard = ShellPairCacheOff::new();

    for &thresh in &thresholds {
        let config = RhfConfig {
            integral_thresh: thresh,
            density_conv: 1e-10,
            k_builder: None, // DEFAULT path
            ..Default::default()
        };
        let b_schwarz = SchwarzBounds::compute_for_screening(op, &prep, ScreeningKind::Schwarz)
            .expect("Schwarz bounds");
        let b_csam = SchwarzBounds::compute_for_screening(op, &prep, ScreeningKind::Csam)
            .expect("CSAM bounds");

        let r_schwarz = solve_rhf_direct_full_rebuild(&ctx, &mol, &prep, op, &b_schwarz, &config);
        let r_csam = solve_rhf_direct_full_rebuild(&ctx, &mol, &prep, op, &b_csam, &config);
        assert!(r_schwarz.converged, "Schwarz run did not converge at thresh {thresh:.0e}");
        assert!(r_csam.converged, "CSAM run did not converge at thresh {thresh:.0e}");

        let err = (r_csam.energy - r_schwarz.energy).abs();
        eprintln!(
            "alkane_4/cc-pVDZ thresh {thresh:.0e}: E(Schwarz) = {:.12}, E(CSAM) = {:.12}, \
             |dE| = {err:.3e} Ha  (quartets {} vs {})",
            r_schwarz.energy, r_csam.energy, r_csam.computed_quartets, r_schwarz.computed_quartets
        );
        errors.push(err);
    }

    // (1) The estimate must actually be ENGAGED at the loosest threshold —
    // otherwise this whole test is measuring inertness.
    assert!(
        errors[0] > 0.0,
        "CSAM introduced EXACTLY zero energy error at the loosest threshold ({:.0e}) — the \
         estimate is inert here (an all-ones X table produces exactly this), so this test \
         would prove nothing about threshold control",
        thresholds[0]
    );

    // (2) THE LOAD-BEARING CLAIM: the error must be controlled by the
    // threshold. Compare tightest against loosest rather than demanding
    // strict monotonicity at every step, because SCF is a nonlinear fixed
    // point and a single intermediate step can move non-monotonically
    // without invalidating threshold control.
    let loosest = errors[0];
    let tightest = errors[errors.len() - 1];
    assert!(
        tightest < loosest,
        "CSAM energy error did NOT fall as the integral threshold tightened \
         ({:.0e} -> {:.0e} gave {loosest:.3e} -> {tightest:.3e} Ha). The sole justification for \
         shipping a non-rigorous estimate is that its error is controllable by the threshold; \
         if that fails, `screening = \"csam\"` must not be offered.",
        thresholds[0],
        thresholds[thresholds.len() - 1]
    );

    // (3) At production threshold the error must be negligible against the
    // chemical-accuracy scale it would otherwise corrupt.
    assert!(
        tightest < 1e-6,
        "CSAM energy error at the tightest threshold ({:.0e}) is {tightest:.3e} Ha, which is not \
         negligible on a chemical-accuracy scale — the estimate is not safe to offer even as an \
         opt-in",
        thresholds[thresholds.len() - 1]
    );
}

/// Pins `FERRIC_SHELLPAIR_CACHE=0` for one test and restores the prior value on
/// drop. An RAII guard rather than bare `set_var`/`remove_var` calls because a
/// panicking assertion between them would otherwise leak the setting into every
/// test sharing this process — the same class of cross-test env leak that made
/// three `shellpair_cache_*` tests fail under the parallel pre-push gate.
struct ShellPairCacheOff(Option<String>);

impl ShellPairCacheOff {
    fn new() -> Self {
        let prev = std::env::var("FERRIC_SHELLPAIR_CACHE").ok();
        std::env::set_var("FERRIC_SHELLPAIR_CACHE", "0");
        Self(prev)
    }
}

impl Drop for ShellPairCacheOff {
    fn drop(&mut self) {
        match self.0.take() {
            Some(v) => std::env::set_var("FERRIC_SHELLPAIR_CACHE", v),
            None => std::env::remove_var("FERRIC_SHELLPAIR_CACHE"),
        }
    }
}
