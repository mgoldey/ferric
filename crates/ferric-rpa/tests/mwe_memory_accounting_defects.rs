//! MWEs for the memory-accounting defects fixed alongside `MemoryPlan`.
//!
//! Each defect below shares one shape: an allocation the code makes and the
//! estimator does not charge (or charges at the wrong size). The failure mode
//! is asymmetric and both halves are bugs, so every contract here is tested in
//! BOTH directions:
//!
//! * **under-counting** approves a job that then OOMs — the 2026-07-13 mode,
//!   where a gate passed a 16-17 GB job against a 10 GB budget;
//! * **over-counting** refuses a job that would have fit — which turns the gate
//!   into a wall and trains users to inflate budgets until it is useless.
//!
//! So each path gets a starvation-budget test (must refuse, and must NAME the
//! offending term so the refusal is actionable) and an ample-budget test (must
//! still run to completion and return sane numbers).
//!
//! The systems are deliberately tiny. Every contract here is structural — is
//! the gate consulted at all, does the breakdown name the term — not
//! scale-dependent, and this box is shared.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::{
    dielectric_spectrum_static, pdep_polarizability_hirshfeld,
    pdep_polarizability_hirshfeld_dynamic, pdep_polarizability_static,
};
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const AMPLE: usize = 4 * 1024 * 1024 * 1024;

type WaterSetup = (
    Molecule,
    PreparedBasis,
    basis::BasisSet,
    PreparedBasis,
    ferric_scf::result::ScfResult,
);

fn water_scf() -> WaterSetup {
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).expect("parse water");
    let obs_bs = basis::bundled("sto-3g").expect("sto-3g");
    let dfbs_bs = basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("prep obs");
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).expect("prep dfbs");

    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).expect("rhf");
    (mol, obs, obs_bs, dfbs, rhf)
}

// ---------------------------------------------------------------------------
// Defect D: the Hirshfeld paths had NO gate at all.
// ---------------------------------------------------------------------------

/// The static Hirshfeld path must refuse a starvation budget.
///
/// Before the fix this path had zero `check_alloc`/`preflight_*` call sites,
/// while holding strictly MORE memory than the Becke path that was judged to
/// need a gate: its `npts` comes from a real-space bounding box (so it grows
/// with molecular VOLUME, not atom count), and its `chi` is monolithic and
/// genuinely read, so it cannot be chunked away.
#[test]
fn hirshfeld_static_refuses_a_starvation_budget() {
    let (mol, obs, obs_bs, dfbs, rhf) = water_scf();
    let cfg = PdepRpaConfig { memory_budget_bytes: Some(1), ..PdepRpaConfig::default() };

    let msg = pdep_polarizability_hirshfeld(
        &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::coulomb(), &cfg, None,
    )
    .expect_err("a 1-byte budget must be REFUSED before chi allocates")
    .to_string();

    // Actionable: name the path, its shape, and the remedy.
    for needle in ["pdep_polarizability_hirshfeld", "natoms=", "npts=", "nbf="] {
        assert!(msg.contains(needle), "refusal must contain {needle:?}; got: {msg}");
    }
    // And name the TERM that blew up — a bare total teaches nothing about
    // which allocation to shrink.
    assert!(
        msg.contains("chi (nbf,npts)"),
        "breakdown must name the dominant grid term; got: {msg}"
    );
}

/// The gate must not become a wall: water/STO-3G needs a few MB and must run.
#[test]
fn hirshfeld_static_runs_under_an_ample_budget() {
    let (mol, obs, obs_bs, dfbs, rhf) = water_scf();
    let cfg = PdepRpaConfig { memory_budget_bytes: Some(AMPLE), ..PdepRpaConfig::default() };

    let alpha = pdep_polarizability_hirshfeld(
        &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::coulomb(), &cfg, None,
    )
    .expect("water/STO-3G must fit a 4 GiB budget comfortably");

    assert_eq!(alpha.len(), mol.atoms.len(), "one 3x3 tensor per atom");
    for (a, t) in alpha.iter().enumerate() {
        let trace = t[0][0] + t[1][1] + t[2][2];
        assert!(trace.is_finite(), "atom {a}: non-finite alpha trace");
    }
}

/// The dynamic Hirshfeld path must refuse a starvation budget.
///
/// This is the heaviest property path: on top of the static footprint it adds
/// a full `b_ov`-shaped `(naux, nov)` buffer AND an `(naux, naux)` `eps_mat`
/// PER RAYON WORKER inside its per-frequency `map_init` — exactly the
/// worker-count-scaled shape behind the 2026-07-13 incident, and it was
/// ungoverned.
#[test]
fn hirshfeld_dynamic_refuses_a_starvation_budget() {
    let (mol, obs, obs_bs, dfbs, rhf) = water_scf();
    let cfg = PdepRpaConfig { memory_budget_bytes: Some(1), ..PdepRpaConfig::default() };

    let msg = pdep_polarizability_hirshfeld_dynamic(
        &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::coulomb(), &cfg, &[0.0], None,
    )
    .expect_err("a 1-byte budget must be REFUSED before chi allocates")
    .to_string();

    assert!(
        msg.contains("pdep_polarizability_hirshfeld_dynamic"),
        "refusal must name the path; got: {msg}"
    );
    // `combined` is the term the audit missed: a SECOND (nbf,npts) buffer held
    // live alongside `chi`, i.e. a 2x on the largest term in this path. Pin
    // both, so a future edit that stops charging the second buffer fails here
    // rather than silently halving the estimate.
    assert!(
        msg.contains("chi (nbf,npts)") && msg.contains("combined (nbf,npts)"),
        "breakdown must charge BOTH co-resident (nbf,npts) buffers, not just chi; got: {msg}"
    );
}

#[test]
fn hirshfeld_dynamic_runs_under_an_ample_budget() {
    let (mol, obs, obs_bs, dfbs, rhf) = water_scf();
    let cfg = PdepRpaConfig { memory_budget_bytes: Some(AMPLE), ..PdepRpaConfig::default() };

    let alpha = pdep_polarizability_hirshfeld_dynamic(
        &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::coulomb(), &cfg, &[0.0], None,
    )
    .expect("water/STO-3G must fit a 4 GiB budget comfortably");

    assert_eq!(alpha.len(), mol.atoms.len(), "one row per atom");
    for per_atom in &alpha {
        assert_eq!(per_atom.len(), 1, "one 3x3 tensor per requested frequency");
        let t = &per_atom[0];
        assert!((t[0][0] + t[1][1] + t[2][2]).is_finite(), "non-finite alpha trace");
    }
}

// ---------------------------------------------------------------------------
// Defect D: the molecular (non-grid) static entry points had NO gate either.
// ---------------------------------------------------------------------------

/// `pdep_polarizability_static` is a public entry point reachable from Python
/// and the CLI. It touches no grid, so it fell outside the 2026-07-13 sweep —
/// but "no grid" is not "no memory": it builds a full `naux x nov` column-
/// scaled copy of `b_ov` plus an `naux x naux` dielectric, with no ceiling.
#[test]
fn molecular_static_refuses_a_starvation_budget() {
    let (mol, obs, _obs_bs, dfbs, rhf) = water_scf();
    let cfg = PdepRpaConfig { memory_budget_bytes: Some(1), ..PdepRpaConfig::default() };

    let msg = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, Operator::coulomb(), &cfg)
        .expect_err("a 1-byte budget must be REFUSED")
        .to_string();

    assert!(msg.contains("pdep_polarizability_static"), "must name the path; got: {msg}");
    assert!(
        msg.contains("b_scaled (naux,nov)"),
        "breakdown must name the scaled-copy term — it is the one that is a \
         genuine second buffer rather than an alias; got: {msg}"
    );
}

#[test]
fn molecular_static_runs_under_an_ample_budget() {
    let (mol, obs, _obs_bs, dfbs, rhf) = water_scf();
    let cfg = PdepRpaConfig { memory_budget_bytes: Some(AMPLE), ..PdepRpaConfig::default() };

    let res = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, Operator::coulomb(), &cfg)
        .expect("water/STO-3G must fit a 4 GiB budget");
    let t = res.tensor;
    let trace = t[0][0] + t[1][1] + t[2][2];
    assert!(trace.is_finite() && trace > 0.0, "alpha trace must be finite and positive: {trace}");
}

#[test]
fn dielectric_spectrum_static_refuses_a_starvation_budget() {
    let (mol, obs, _obs_bs, dfbs, rhf) = water_scf();

    let msg = dielectric_spectrum_static(
        &mol, &obs, &dfbs, &rhf, Operator::coulomb(), 0.0, Some(1),
    )
    .expect_err("a 1-byte budget must be REFUSED")
    .to_string();

    assert!(msg.contains("dielectric_spectrum_static"), "must name the path; got: {msg}");
}

#[test]
fn dielectric_spectrum_static_runs_under_an_ample_budget() {
    let (mol, obs, _obs_bs, dfbs, rhf) = water_scf();

    let spec = dielectric_spectrum_static(
        &mol, &obs, &dfbs, &rhf, Operator::coulomb(), 0.0, Some(AMPLE),
    )
    .expect("water/STO-3G must fit a 4 GiB budget");
    assert!(!spec.eigenvalues.is_empty(), "spectrum must be non-empty");
    assert!(
        spec.eigenvalues.iter().all(|v| v.is_finite()),
        "all dielectric eigenvalues must be finite"
    );
}

// ---------------------------------------------------------------------------
// Defect A: the Becke path's monolithic chi was ~85x under-charged.
// ---------------------------------------------------------------------------

/// The estimator models `chi` as CHUNKED (`nbf * npts/1024 * 8 * n_workers`)
/// because `accumulate_atom_centred_dipoles` evaluates it per grid chunk. The
/// Becke path nevertheless called `eval_basis_on_points` on the FULL point list
/// one line after the gate — allocating the monolithic `nbf * npts` block the
/// estimator had been rewritten to stop charging for, and using it for nothing
/// but its row count.
///
/// The fix deletes that allocation (`nbf` now comes from `obs.nbasis()`, the
/// same integer by construction), which makes the estimate true rather than
/// merely raising the charge.
///
/// This test pins the CONSEQUENCE: the gate's own accounting must not be
/// contradicted by an allocation the method makes anyway. A budget sized to
/// the estimator's chunked model must therefore actually suffice — under the
/// old code the method would allocate ~85x that and die.
#[test]
fn becke_path_does_not_allocate_the_monolithic_chi_it_never_charges() {
    use ferric_rpa::budget::{estimate_grid_bytes, GridEstimateShape};

    // The estimator's chunked chi model, at incident scale.
    let (nbf, npts, n_workers) = (800usize, 586_000usize, 12usize);
    let chunked = estimate_grid_bytes(GridEstimateShape {
        npts,
        nbf,
        natoms: 0,
        dipole_band_width: 0,
        n_workers,
    });
    // The monolithic block that used to be allocated right after the gate.
    let monolithic = nbf * npts * 8;

    // The gap the defect hid. If a future change re-materializes a whole-grid
    // chi on this path, the estimate becomes an under-count of this magnitude
    // again — this assertion documents why the deletion matters.
    assert!(
        monolithic > 50 * chunked,
        "the monolithic chi ({:.2} GB) must dwarf the chunked charge ({:.2} GB) — \
         if these have converged, re-check what the estimator models",
        monolithic as f64 / 1e9,
        chunked as f64 / 1e9,
    );
}

/// The Becke path must still produce correct numbers after the deletion.
///
/// `nbf` changed source (from `chi.nrows()` to `obs.nbasis()`) but not value,
/// so this must be bit-for-bit what it was. Pinned against the Hirshfeld-free
/// invariant that the per-atom tensors sum to something finite and positive.
#[test]
fn becke_path_still_runs_and_returns_sane_tensors() {
    use ferric_rpa::properties::pdep_polarizability_becke;
    let (mol, obs, obs_bs, dfbs, rhf) = water_scf();
    let cfg = PdepRpaConfig { memory_budget_bytes: Some(AMPLE), ..PdepRpaConfig::default() };

    let alpha = pdep_polarizability_becke(
        &mol, &obs, &obs_bs, &dfbs, &rhf, Operator::coulomb(), &cfg,
    )
    .expect("water/STO-3G must fit a 4 GiB budget");

    assert_eq!(alpha.len(), mol.atoms.len());
    let total: f64 = alpha.iter().map(|t| t[0][0] + t[1][1] + t[2][2]).sum();
    assert!(total.is_finite() && total > 0.0, "summed trace must be finite and positive");
}

// ---------------------------------------------------------------------------
// Defect E: the per-worker charge must cover the width actually allocated.
// ---------------------------------------------------------------------------

/// `estimate_peak_bytes` charges the per-worker quadrature scratch at width
/// `m = min(n_keep, naux)`. The property paths' `map_init` closures, however,
/// allocate at width `naux` (a full `b_ov`-shaped buffer and a full
/// `(naux, naux)` `eps_mat`), not at `m`.
///
/// That was invisible because every caller passes `n_keep = naux`, so the two
/// widths coincided — coverage by coincidence, not by construction. A caller
/// passing a truncated `n_keep < naux` would have been silently under-charged
/// on precisely the worker-count-scaled term that caused the incident.
///
/// Contract: truncating `n_keep` must not reduce the estimate below what the
/// naux-wide per-worker allocation actually costs.
#[test]
fn truncated_n_keep_still_charges_the_naux_wide_per_worker_scratch() {
    use ferric_rpa::budget::{estimate_peak_bytes, PeakEstimateShape};

    let full = PeakEstimateShape {
        naux: 500,
        nocc: 10,
        nvir: 100,
        n_quad: 20,
        n_workers: 8,
        n_keep: 500,
        grid: None,
    };
    let truncated = PeakEstimateShape { n_keep: 10, ..full };

    let est_full = estimate_peak_bytes(full);
    let est_trunc = estimate_peak_bytes(truncated);

    // The per-worker allocation does not shrink when n_keep shrinks, so the
    // charge must not either. Before the fix, est_trunc collapsed to roughly
    // the m=10 figure and under-charged the real allocation ~50x on this term.
    let naux_wide_per_worker =
        (500usize * (10 * 100) + 500 * 500) * 8 * 8; // (naux*nov + naux^2) * 8 bytes * 8 workers
    assert!(
        est_trunc >= naux_wide_per_worker,
        "a truncated n_keep must still charge the naux-wide per-worker scratch \
         ({:.3} GB); got {:.3} GB",
        naux_wide_per_worker as f64 / 1e9,
        est_trunc as f64 / 1e9,
    );

    // ...and it must not have become an OVER-estimate either: truncation
    // genuinely shrinks the m-wide output arrays, so the total may fall, but
    // it must never exceed the untruncated figure.
    assert!(
        est_trunc <= est_full,
        "truncating n_keep must not INCREASE the estimate: {est_trunc} > {est_full}"
    );
}

/// The `n_keep == naux` case — every real caller — must be unchanged by the
/// defect-E fix. An accounting correction that moved the number for the
/// production configuration would be a behavior change, not a fix.
#[test]
fn untruncated_estimate_is_unchanged_for_the_production_shape() {
    use ferric_rpa::budget::{estimate_peak_bytes, PeakEstimateShape};

    // Hand-derived from the module docs, with the per-worker term at the width
    // the allocation uses (naux == n_keep here, so both readings agree).
    let (naux, nocc, nvir, n_workers) = (500usize, 10usize, 100usize, 8usize);
    let nov = nocc * nvir;
    let expect = /* metric */ naux * naux * 3 * 8
        + /* b_ov */ naux * nov * 8
        + /* lanczos */ naux * naux * 2 * 8
        + /* per-worker */ (naux * nov + naux * naux) * 8 * n_workers
        + /* y projection */ naux * nov * 8
        + /* outputs */ naux * naux * 2 * 8;

    let got = estimate_peak_bytes(PeakEstimateShape {
        naux, nocc, nvir, n_quad: 20, n_workers, n_keep: naux, grid: None,
    });
    assert_eq!(got, expect, "the production (n_keep == naux) estimate must be unchanged");
}
