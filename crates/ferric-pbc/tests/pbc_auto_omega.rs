//! `gdf_omega = "auto"`: the per-cell RS-GDF ω chooser
//! (`ferric_pbc::rsgdf::auto_omega`, issue #227).
//!
//! The pure decision rule is unit-tested in the module. Here, against the
//! real build:
//!
//! (a) TRIVIAL-LIMIT ANCHOR: an explicit ω = 1.0 config is bit for bit the
//!     default build (`DEFAULT_RSGDF_OMEGA`), so `auto` that lands on 1.0 is
//!     today's build, and an explicit value is never touched.
//! (b) COUNT ANCHOR: the chooser's SR triplet count at a candidate ω equals
//!     the triplet count the build itself reports at that ω (same walk), and
//!     its closed-form G count is within 2% of the build's half-sphere.
//! (c) The choice is one of the candidates, deterministic, and does not
//!     increase as the same atoms are put in a larger cell (a sparser cell
//!     is more LR-heavy, never less).
//! (d) The pair-FT work proxy is positive and does not depend on how far
//!     the probe sphere extends.
//!
//! Artifact hypothesis, stated before running: if the chooser read a stale
//! or ω-independent SR count, (b) fails at every ω != the one it read; if it
//! priced the wrong sphere, the G count misses by the cube of the ratio.

mod common;

use common::*;
use ferric_core::basis;
use ferric_core::mol::{Atom, Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::pair_ft::pair_ft_work;
use ferric_pbc::rsgdf::auto_omega::{auto_rsgdf_omega, half_sphere_g_count, AUTO_OMEGA_CANDIDATES};
use ferric_pbc::rsgdf::{
    sr_triplet_estimate, sr_walk_counts, RangeSplit, RsGdf, RsGdfConfig, DEFAULT_RSGDF_OMEGA,
};
use ferric_pbc::SrColumnRotation;

const AMPLE: usize = 1 << 30;

fn base() -> RsGdfConfig {
    RsGdfConfig {
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

fn fixtures(a: f64) -> (Cell, PreparedBasis, PreparedBasis) {
    let cell = h2_cell(a);
    let obs = prep_for(&cell, &pyscf_sto3g_h());
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    (cell, obs, aux)
}

#[test]
fn explicit_default_omega_is_the_default_build_bit_for_bit() {
    let (cell, obs, aux) = fixtures(4.0);
    let hc = periodic_hcore(&cell, &obs, &PeriodicHcoreConfig::with_omega(0.8)).unwrap();
    assert_eq!(RsGdfConfig::default().omega, DEFAULT_RSGDF_OMEGA);
    let dflt = RsGdf::build(&cell, &obs, &aux, &hc.s, &base()).unwrap();
    let explicit = RsGdf::build(
        &cell,
        &obs,
        &aux,
        &hc.s,
        &RsGdfConfig {
            omega: 1.0,
            ..base()
        },
    )
    .unwrap();
    assert_eq!(
        dflt.b(),
        explicit.b(),
        "ω = 1.0 explicit must be the default"
    );
}

#[test]
fn chooser_counts_are_the_builds_own_counts() {
    let (cell, obs, aux) = fixtures(4.0);
    let hc = periodic_hcore(&cell, &obs, &PeriodicHcoreConfig::with_omega(0.8)).unwrap();
    let r = auto_rsgdf_omega(&cell, &obs, &aux, &base()).unwrap();
    assert_eq!(r.candidates.len(), AUTO_OMEGA_CANDIDATES.len());
    // Two candidates on opposite sides of the default.
    for &w in &[0.5, 1.4] {
        let c = r.candidates.iter().find(|c| c.omega == w).unwrap();
        let g = RsGdf::build(
            &cell,
            &obs,
            &aux,
            &hc.s,
            &RsGdfConfig { omega: w, ..base() },
        )
        .unwrap();
        let st = g.stats();
        assert_eq!(
            c.n_sr3 as usize, st.n_sr3_triplets,
            "SR triplets at ω = {w}"
        );
        let rel = c.n_g / st.n_g_half as f64 - 1.0;
        assert!(
            rel.abs() < 0.02,
            "G count at ω = {w}: {} vs {}",
            c.n_g,
            st.n_g_half
        );
    }
    // The SR count must actually depend on ω (a stale count would not).
    let n: Vec<f64> = r.candidates.iter().map(|c| c.n_sr3).collect();
    assert!(
        n.windows(2).all(|w| w[0] >= w[1]),
        "SR count not monotone: {n:?}"
    );
    assert!(n[0] > n[n.len() - 1], "SR count independent of ω: {n:?}");
}

#[test]
fn choice_is_a_candidate_deterministic_and_nonincreasing_with_cell_size() {
    let mut last = f64::INFINITY;
    for a in [4.0, 8.0, 14.0, 24.0] {
        let (cell, obs, aux) = fixtures(a);
        let r1 = auto_rsgdf_omega(&cell, &obs, &aux, &base()).unwrap();
        let r2 = auto_rsgdf_omega(&cell, &obs, &aux, &base()).unwrap();
        assert_eq!(r1, r2, "a = {a}: not deterministic");
        assert!(
            AUTO_OMEGA_CANDIDATES.contains(&r1.omega),
            "a = {a}: {}",
            r1.omega
        );
        assert!(
            r1.omega <= last,
            "a = {a}: ω rose from {last} to {} in a larger cell",
            r1.omega
        );
        last = r1.omega;
    }
}

#[test]
fn pair_ft_work_is_positive_and_independent_of_the_probe_extent() {
    let (cell, obs, _) = fixtures(8.0);
    let probe = |gcut: f64| {
        let g: Vec<[f64; 3]> = cell
            .gvectors(gcut)
            .unwrap()
            .into_iter()
            .filter(|g| g[0] > 0.0 || (g[0] == 0.0 && (g[1] > 0.0 || (g[1] == 0.0 && g[2] > 0.0))))
            .collect();
        assert!(!g.is_empty());
        pair_ft_work(&cell, &obs, &g, 1e-13).unwrap()
    };
    let (w1, w2) = (probe(2.0), probe(6.0));
    assert!(w1 > 0);
    assert_eq!(
        w1, w2,
        "survivors reach min|G|, so the sphere extent is irrelevant"
    );
}

#[test]
fn g_count_formula_tracks_the_build() {
    let (cell, obs, aux) = fixtures(6.0);
    let hc = periodic_hcore(&cell, &obs, &PeriodicHcoreConfig::with_omega(0.8)).unwrap();
    let g = RsGdf::build(
        &cell,
        &obs,
        &aux,
        &hc.s,
        &RsGdfConfig {
            omega: 1.4,
            ..base()
        },
    )
    .unwrap();
    let est = half_sphere_g_count(cell.volume(), 1.4, base().precision);
    let rel = est / g.stats().n_g_half as f64 - 1.0;
    assert!(rel.abs() < 0.05, "{est} vs {}", g.stats().n_g_half);
}

// ---- the SR triplet estimator ------------------------------------------

/// A C2 dimer in a 7.5 Bohr cube, cc-pVDZ / cc-pVDZ-RI: generally contracted
/// carbon shells, so the column rotation changes the walk (H2 / Pople bases
/// have nothing to rotate).
fn tri_fixtures() -> (Cell, PreparedBasis, PreparedBasis) {
    let atom = |z: f64| Atom {
        symbol: "C".into(),
        z: 6,
        x: 0.4,
        y: 0.5,
        zpos: z,
        ghost: false,
        n_core_ecp: 0,
    };
    let mol = Molecule {
        atoms: vec![atom(0.3), atom(2.7)],
        charge: 0,
        multiplicity: 1,
    };
    let cell = Cell::new(mol, cubic(7.5)).expect("C2 cell");
    let obs = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    (cell, obs, aux)
}

/// The SR walks the chooser may price: unsplit and range split, each
/// unrotated and with the column rotation (`Off` / the default `Auto`).
fn walk_variants() -> Vec<(&'static str, RsGdfConfig)> {
    let off = SrColumnRotation::Off;
    let split = Some(RangeSplit::default());
    vec![
        (
            "unsplit",
            RsGdfConfig {
                sr_column_rotation: off,
                ..base()
            },
        ),
        ("unsplit+rotation", base()),
        (
            "split",
            RsGdfConfig {
                range_split: split,
                sr_column_rotation: off,
                ..base()
            },
        ),
        (
            "split+rotation",
            RsGdfConfig {
                range_split: split,
                ..base()
            },
        ),
    ]
}

#[test]
fn estimator_is_exact_when_the_budget_is_not_reached() {
    // TRIVIAL-LIMIT ANCHOR: with a budget above the total every cell is
    // visited, and the estimate IS the build's s2 count, for every walk.
    let (cell, obs, aux) = tri_fixtures();
    for (name, cfg) in walk_variants() {
        for omega in [0.5, 1.0] {
            let cfg = RsGdfConfig { omega, ..cfg };
            let exact = sr_walk_counts(&cell, &obs, &aux, &cfg)
                .unwrap()
                .n_sr3_triplets;
            let est = sr_triplet_estimate(&cell, &obs, &aux, &cfg, u64::MAX).unwrap();
            assert_eq!(est, exact as f64, "{name} at ω = {omega}");
            assert!(exact > 0, "{name}: nothing to count");
        }
    }
    // The rotation really is in the walks above (it removes triplets), so
    // the equality is not an unrotated-twice tautology.
    let n = |name: &str| {
        let (_, cfg) = walk_variants().into_iter().find(|v| v.0 == name).unwrap();
        sr_walk_counts(&cell, &obs, &aux, &cfg)
            .unwrap()
            .n_sr3_triplets
    };
    assert!(n("unsplit+rotation") < n("unsplit"));
    assert!(n("split") < n("unsplit"));
}

#[test]
fn sampled_estimate_tracks_the_exact_count() {
    let (cell, obs, aux) = tri_fixtures();
    for (name, cfg) in walk_variants() {
        let exact = sr_walk_counts(&cell, &obs, &aux, &cfg)
            .unwrap()
            .n_sr3_triplets as f64;
        // A budget of ~5% of the triplets forces real sampling.
        let budget = (exact / 20.0) as u64;
        let est = sr_triplet_estimate(&cell, &obs, &aux, &cfg, budget).unwrap();
        let rel = est / exact - 1.0;
        eprintln!("{name}: exact {exact:.0}, budget {budget}, est {est:.0}, rel {rel:+.4}");
        assert!(rel.abs() < SAMPLE_BAR, "{name}: {rel:+.4}");
        // Deterministic: the same call gives the same bits.
        let again = sr_triplet_estimate(&cell, &obs, &aux, &cfg, budget).unwrap();
        assert_eq!(est.to_bits(), again.to_bits(), "{name}");
    }
}

/// Derived from measuring both sides on this fixture (the sampler is
/// deterministic, so these are fixed numbers, not a flaky statistic). The
/// correct sampler at a ~5% budget: |rel| = 0.19%, 0.52%, 0.47%, 0.05%
/// (unsplit, +rotation, split, split+rotation). Mutants of the sampler:
/// sequential visiting order (stride 1) -4.95% / -53.7%, no scale-up -75%,
/// scaling by cells/(visited + wave) -50%. The bar sits between: 2%.
const SAMPLE_BAR: f64 = 0.02;
