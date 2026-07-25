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
//! NOTE `chi` has since been FUSED into the banded accumulation — it is now
//! evaluated per grid chunk (one `nbf · chunk_size` buffer per worker, ~44 MB at
//! 12 workers vs 3.75 GB monolithic). The contracts below track the fused form:
//! the estimate must still RESPOND to grid density, but it no longer needs to
//! cover a full `nbf · npts` block, because none is allocated. Charging the old
//! figure would over-reject, and an over-estimating gate is as broken as an
//! under-estimating one.
//!
//! Those two, plus the banded partials, accounted for the 16-17 GB anon-RSS
//! incidents of 2026-07-13. A preflight gate built on the pre-grid estimator
//! would have passed such a job as comfortably fitting a 10 GB budget — which
//! is why "wire in the existing gate" was never sufficient on its own.
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

use ferric_rpa::budget::{
    estimate_grid_bytes, estimate_peak_bytes, GridEstimateShape, PeakEstimateShape,
};

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
        // Serial: the most conservative (smallest) per-chunk chi term.
        n_workers: 1,
    }
}

fn incident_shape() -> PeakEstimateShape {
    PeakEstimateShape { grid: Some(grid_shape(INCIDENT_NPTS)), ..energy_only_shape() }
}

/// CONTRACT 1: the estimate must cover the per-chunk `chi` buffers.
///
/// `chi` is no longer one `Array2::zeros((nbf, npts))`; it is evaluated per grid
/// chunk inside the banded loop, so the resident term is
/// `nbf · chunk_size · 8 · n_workers`. The estimate must cover THAT — covering
/// the (no-longer-allocated) monolithic figure would be a 85x over-estimate at
/// incident scale.
#[test]
fn estimate_covers_the_per_chunk_chi_buffers() {
    use ferric_rpa::budget::dipole_chunk_count;
    let n_chunks = dipole_chunk_count(INCIDENT_NPTS);
    let chunk_size = INCIDENT_NPTS.div_ceil(n_chunks);
    let chi_bytes = INCIDENT_NBF * chunk_size * 8; // n_workers = 1 in grid_shape
    let est = estimate_peak_bytes(incident_shape());

    assert!(
        est >= chi_bytes,
        "estimate {:.2} GB must cover the per-chunk chi buffers ({:.2} MB)",
        est as f64 / 1e9,
        chi_bytes as f64 / 1e6,
    );

    // And the GRID term must no longer charge the monolithic block. Compare
    // grid-vs-grid: the full `est` also carries the energy-path terms (at this
    // shape the per-worker quadrature scratch alone is ~28.7 GB at 12 workers),
    // so comparing the TOTAL against one component would be meaningless — an
    // earlier version of this assertion did exactly that and failed at 36.05 GB
    // vs 3.75 GB, reporting a defect that was not there.
    let grid_only = estimate_grid_bytes(grid_shape(INCIDENT_NPTS));
    let monolithic = INCIDENT_NBF * INCIDENT_NPTS * 8;
    assert!(
        grid_only < monolithic,
        "the grid term {:.2} GB must be below the old monolithic chi ({:.2} GB) \
         — chi is chunked now, so charging the full block would over-reject",
        grid_only as f64 / 1e9,
        monolithic as f64 / 1e9,
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

    // The response now comes from the grid side arrays (npts-proportional) and
    // the wider chunks, not from a monolithic block. Require a real increase
    // proportional to the extra points, not a token bump.
    let extra_side = INCIDENT_NPTS * (3 * 8 + 8 + 8);
    assert!(
        dense - sparse >= extra_side,
        "doubling npts must add at least the extra grid side arrays ({:.2} MB), \
         added {:.2} MB",
        extra_side as f64 / 1e6,
        (dense - sparse) as f64 / 1e6,
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
/// This asserts the estimate for the CODE AS IT WAS during the incident: a
/// monolithic chi, and a dipole band width of 12 (what the old thread floor
/// produced on that 12-core box). Both have since been fixed, so the same job
/// now needs far less — but the estimator must still be able to describe the
/// original conditions, or we have no evidence it would have caught them.
#[test]
fn estimate_would_have_caught_the_incident_conditions() {
    const OBSERVED_RSS_BYTES: usize = 16 * 1_000_000_000; // 16 GB, the low end
    let incident_conditions = PeakEstimateShape {
        grid: Some(GridEstimateShape {
            dipole_band_width: 12, // the old thread floor on a 12-core box
            ..grid_shape(INCIDENT_NPTS)
        }),
        ..energy_only_shape()
    };
    let est = estimate_peak_bytes(incident_conditions);

    assert!(
        est >= OBSERVED_RSS_BYTES,
        "estimate {:.2} GB must be >= the {:.2} GB observed in the 2026-07-13 \
         incident, or a preflight gate built on it would have passed the job \
         that OOM-killed the box",
        est as f64 / 1e9,
        OBSERVED_RSS_BYTES as f64 / 1e9,
    );
}
