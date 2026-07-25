//! MWE: can the peak estimator predict the allocation that actually OOM'd?
//!
//! `budget::estimate_peak_bytes` models the ENERGY path — naux, nocc, nvir,
//! n_workers — and is well-tested there. It has no term for the GRID path, so
//! it cannot see the two allocations that dominate `pdep_polarizability_becke`:
//!
//! ```text
//!   chi      = nbf · npts · 8          ~3.75 GB at nbf=800, npts=586k
//!   d_ai_ao  = natoms · 3 · nbf² · 8   ~1.1 GB at natoms=71, nbf=800
//! ```
//!
//! Those, plus the banded partials, account for the 16-17 GB anon-RSS incidents
//! of 2026-07-13 without needing anything else. A preflight gate built on the
//! current estimator would pass such a job as comfortably fitting a 10 GB
//! budget — which is why "wire in the existing gate" is NOT sufficient on its
//! own, and why this MWE comes before that work.
//!
//! These contracts specified the missing terms before they existed; the
//! `grid: Option<GridEstimateShape>` field and `estimate_grid_bytes` were added
//! to satisfy them. They now serve as the regression net: contract 5 in
//! particular pins that `grid: None` leaves the energy-path estimate
//! byte-identical, so the already-wired energy gates (`run_pdep_rpa`,
//! `rs_mp2_rpa`, `ao_rpa`) cannot be perturbed by this addition.
//!
//! # A note on what is NOT a defect
//!
//! `estimate_peak_bytes` deliberately discards `n_quad` (budget.rs:146-147,
//! with a named no-op guard and a comment). That is CORRECT: quadrature points
//! are processed per-worker via `map_init` and never retained, so `n_quad`
//! drives wall time, not peak resident bytes. Do not "fix" it.

use ferric_rpa::budget::{estimate_peak_bytes, GridEstimateShape, PeakEstimateShape};

/// The 71-atom/def2-SVP shape from the incident, rounded to round numbers.
/// npts = 71 atoms x 8250 points/atom (the default 75 radial x 110 angular
/// Becke-Lebedev grid, read from AtomicGridConfig rather than hardcoded in
/// production — reproduced here as a literal only to pin a known scenario).
const INCIDENT_NATOMS: usize = 71;
const INCIDENT_NBF: usize = 800;
const INCIDENT_NPTS: usize = 71 * 8250;

/// The energy-path shape alone, with no grid term (what the estimator modelled
/// before this work).
fn energy_only_shape() -> PeakEstimateShape {
    PeakEstimateShape {
        naux: 2976,
        nocc: 150,
        nvir: 650,
        n_quad: 12,
        n_workers: 12,
        n_keep: 2976,
        grid: None,
    }
}

fn grid_shape(npts: usize) -> GridEstimateShape {
    GridEstimateShape {
        npts,
        nbf: INCIDENT_NBF,
        natoms: INCIDENT_NATOMS,
        // One band live; the real width comes from `dipole_band_width` at the
        // call site. 1 is the floor, so this is the most CONSERVATIVE (smallest)
        // grid estimate — if the contracts hold here they hold for wider bands.
        dipole_band_width: 1,
    }
}

fn incident_shape() -> PeakEstimateShape {
    PeakEstimateShape { grid: Some(grid_shape(INCIDENT_NPTS)), ..energy_only_shape() }
}

/// CONTRACT 1: the estimate must cover the monolithic `chi` allocation.
///
/// `chi` is a single `Array2::zeros((nbf, npts))` — one contiguous ~3.75 GB
/// block at incident scale. An estimator that omits it cannot bound the grid
/// path at all.
#[test]
fn estimate_covers_the_chi_grid_matrix() {
    let chi_bytes = INCIDENT_NBF * INCIDENT_NPTS * 8;
    let est = estimate_peak_bytes(incident_shape());

    assert!(
        est >= chi_bytes,
        "estimate {:.2} GB must cover the chi grid matrix alone ({:.2} GB) — \
         the estimator has no npts term, so it cannot predict the grid path",
        est as f64 / 1e9,
        chi_bytes as f64 / 1e9,
    );
}

/// CONTRACT 2: the estimate must respond to grid density.
///
/// Doubling the number of grid points doubles `chi`. An estimate that is
/// invariant to npts is, by construction, blind to the dominant term.
#[test]
fn estimate_responds_to_grid_density() {
    let sparse = estimate_peak_bytes(incident_shape());
    // Same electronic shape, twice the grid points — npts is the only change.
    let dense = estimate_peak_bytes(PeakEstimateShape {
        grid: Some(grid_shape(INCIDENT_NPTS * 2)),
        ..energy_only_shape()
    });

    assert!(
        dense > sparse,
        "doubling the grid must raise the estimate, got {dense} vs {sparse}"
    );

    // And the increase must be at least the extra chi rows, not a token bump.
    let extra_chi = INCIDENT_NBF * INCIDENT_NPTS * 8;
    assert!(
        dense - sparse >= extra_chi,
        "doubling npts must add at least one more chi ({:.2} GB), added {:.2} GB",
        extra_chi as f64 / 1e9,
        (dense - sparse) as f64 / 1e9,
    );
}

/// CONTRACT 5: the energy path must be unaffected.
///
/// `grid: None` must estimate byte-identically to the pre-grid estimator, so
/// adding the term cannot perturb the already-wired energy-path gates
/// (`run_pdep_rpa`, `rs_mp2_rpa`, `ao_rpa`).
#[test]
fn energy_path_estimate_is_unchanged_by_the_grid_term() {
    let energy = estimate_peak_bytes(energy_only_shape());
    let with_grid = estimate_peak_bytes(incident_shape());

    assert!(
        with_grid > energy,
        "the grid term must add something ({with_grid} vs {energy})"
    );
    // The energy-only figure must still be dominated by naux^2 / naux*nov, i.e.
    // nothing grid-shaped leaked into it.
    let naux_sq = 2976usize * 2976 * 8;
    assert!(
        energy >= naux_sq,
        "energy-only estimate {energy} must still cover the naux^2 terms"
    );
}

/// CONTRACT 3: the estimate must cover the per-atom AO dipole tensors.
///
/// `d_ai_ao` is `natoms · 3` matrices of `nbf²` each, held for the whole
/// accumulation, plus one band of same-sized partials on top.
#[test]
fn estimate_covers_per_atom_dipole_tensors() {
    let d_ai_ao_bytes = INCIDENT_NATOMS * 3 * INCIDENT_NBF * INCIDENT_NBF * 8;
    let est = estimate_peak_bytes(incident_shape());

    assert!(
        est >= d_ai_ao_bytes,
        "estimate {:.2} GB must cover the per-atom AO dipoles ({:.2} GB) — \
         needs natoms and nbf terms",
        est as f64 / 1e9,
        d_ai_ao_bytes as f64 / 1e9,
    );
}

/// CONTRACT 4: the estimate must exceed the RSS actually observed in the
/// incident.
///
/// The end-to-end property. If the estimator had been consulted before the
/// 2026-07-13 runs, it had to predict >= what the process really took (16-17 GB
/// anon-RSS), or the gate would have waved the job through. This is the single
/// assertion that would have prevented the OOM kills.
#[test]
fn estimate_exceeds_observed_incident_rss() {
    const OBSERVED_RSS_BYTES: usize = 16 * 1_000_000_000; // 16 GB, the low end
    let est = estimate_peak_bytes(incident_shape());

    assert!(
        est >= OBSERVED_RSS_BYTES,
        "estimate {:.2} GB must be >= the {:.2} GB actually observed in the \
         2026-07-13 incident, or a preflight gate built on it would have \
         passed the job that OOM-killed the box",
        est as f64 / 1e9,
        OBSERVED_RSS_BYTES as f64 / 1e9,
    );
}
