//! RS-GDF RANGE SPLIT (`RsGdfConfig::range_split`, the `rsgdf::split`
//! module): Rust port of `reference/pbc/pbc_gdf_split.py`, FINDINGS
//! "Iteration 23 (Python, RS-GDF range split)".
//!
//! # Exactness anchors (written first)
//!
//! 1. A split that moves NOTHING (λ = 0 and λ below every exponent) runs the
//!    split machinery (combined piece bases, the split walks, the kept-input
//!    G = 0 subtract) and must reproduce today's J2, J3, B, SR counts and
//!    SCF energy BIT FOR BIT.
//! 2. The exact-span anchor of `pbc_rsgdf.rs` with EVERYTHING moved (no SR
//!    3-centre or 2-centre call left) must reproduce the dense pure-AFT ERI
//!    to ≤ 1e-12 (prototype 2.5e-16..6.1e-16).
//!
//! # Artifact hypotheses (stated before measuring)
//!
//! * "Split and unsplit agree because nothing moved": every agreement test
//!   asserts the moved primitive counters are > 0 (and the SR counts drop).
//! * "The span anchor passes because a G = 0 error cancels": the two G = 0
//!   mutants must FAIL it (`ThreeIndexFullG0` misses; `MetricFullG0` makes
//!   the metric strongly indefinite, which the guard must REFUSE instead of
//!   the lindep cut silently dropping it — prototype −12.7 at ω = 1.2).
//! * "Moving a block is harmless whatever its FT": moving compact blocks
//!   (C 1s, diamond STO-3G) must miss by ≥ 1e-2 in J3 (prototype 6.7e-2);
//!   on the soft triclinic H cell even moving everything misses by only
//!   3.9e-7 (prototype), so that cell is NOT used for this control.
//!
//! # Mutation plan (for the main agent; the in-test mutants run every time)
//!
//! * Drop the libint piece normalisation (`scale` = 1 for pieces, in
//!   `split::Side::new`) → the triclinic and diamond split-vs-unsplit tests
//!   fail (pieces carry `1/‖χ^c‖`).
//! * Drop call B `(χ_i^c, χ_j^s)` in `SplitPlan::calls` → triclinic ΔJ3
//!   jumps far above 1e-10.
//! * Put the moved-aux weight `v_SR X_s` on X instead of X_s in
//!   `lr_moved_aux` → the span anchor fails.
//! * Skip the scatter in `lr_smooth_pairs` → triclinic ΔJ3 fails.
//! * Replace `S − S_ss` by `S` in `subtract_g0_build` → span anchor fails.

mod common;

use common::*;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::site_basis::SiteBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::ewald::default_ewald_omega;
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig, DEFAULT_K_FITTED_KERNELS_MAX_BYTES};
use ferric_pbc::rsgdf::{
    sr_walk_counts, PeriodicFitParts, RangeSplit, RangeSplitMutant, RsGdf, RsGdfConfig,
    DEFAULT_FITTED_ERI_MAX_BYTES, RANGE_SPLIT_NEG_EIG_GUARD,
};
use ferric_pbc::SrColumnRotation;
use ndarray::Array2;
use num_complex::Complex64;

const HCORE_OMEGA: f64 = 0.8;
const ANCHOR_ALPHA: f64 = 0.5;
const AMPLE: usize = 1 << 30;

/// `reference/pbc/bench/diamond_prim.xyz` (Å) and `.lattice` (Bohr rows).
const DIAMOND_PRIM_XYZ: &str = "2\ndiamond_prim\nC 0.0 0.0 0.0\nC 0.89175 0.89175 0.89175\n";
const DIAMOND_PRIM_LATTICE: [[f64; 3]; 3] = [
    [0.0, 3.3703265431617879, 3.3703265431617879],
    [3.3703265431617879, 0.0, 3.3703265431617879],
    [3.3703265431617879, 3.3703265431617879, 0.0],
];
/// ferric CLI (libint, unsplit) on the STO-3G preflight, cc-pvdz-ri,
/// exxdiv ewald (`reference/pbc/bench/out/ferric_pre_diamond_prim_sto3g_ri_rhf_t1.log`);
/// the prototype's split reproduces it to all 10 printed digits.
const DIAMOND_STO3G_E: f64 = -74.0034040288;

fn hcore(cell: &Cell, prep: &PreparedBasis) -> PeriodicHcore {
    periodic_hcore(cell, prep, &PeriodicHcoreConfig::with_omega(HCORE_OMEGA)).expect("hcore")
}

fn cfg(omega: f64, split: Option<RangeSplit>) -> RsGdfConfig {
    RsGdfConfig {
        omega,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        range_split: split,
        ..Default::default()
    }
}

fn mutant(m: RangeSplitMutant) -> Option<RangeSplit> {
    Some(RangeSplit {
        mutant: m,
        ..RangeSplit::default()
    })
}

fn bitwise_eq(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.dim() == b.dim()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn energy(cell: &Cell, prep: &PreparedBasis, hc: &PeriodicHcore, gdf: &RsGdf) -> f64 {
    gamma_rhf_jk(
        cell,
        prep,
        hc,
        Box::new(gdf.j_builder()),
        Box::new(gdf.k_builder()),
    )
    .energy
}

fn counter(gdf: &RsGdf, name: &str) -> u64 {
    gdf.timings()
        .counter(name)
        .unwrap_or_else(|| panic!("counter {name:?} missing (split path did not run?)"))
}

/// `(orbital smooth, orbital total, aux smooth, aux total)` primitives.
fn moved(gdf: &RsGdf) -> [u64; 4] {
    [
        counter(gdf, "rsgdf split orbital prims smooth"),
        counter(gdf, "rsgdf split orbital prims"),
        counter(gdf, "rsgdf split aux prims smooth"),
        counter(gdf, "rsgdf split aux prims"),
    ]
}

fn build_parts(
    cell: &Cell,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    s: &Array2<f64>,
    c: &RsGdfConfig,
) -> (RsGdf, PeriodicFitParts) {
    RsGdf::build_with_fit_parts(cell, obs, aux, s, c).expect("RsGdf build")
}

// --------------------------------------------------------------- anchor 1 --

#[test]
fn a_split_that_moves_nothing_is_bitwise_todays_build() {
    let cell = h2_cell(4.0);
    let obs = prep_for(&cell, &pyscf_sto3g_h());
    let hc = hcore(&cell, &obs);
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let (g0, p0) = build_parts(&cell, &obs, &aux, &hc.s, &cfg(1.0, None));
    assert_eq!(g0.range_split(), None);
    let e0 = energy(&cell, &obs, &hc, &g0);
    // λ = 0 and a λ below every exponent (orbital λω²/2 = 5e-9, aux 1e-8).
    for lam in [0.0, 1e-8] {
        let rs = RangeSplit::new(lam);
        let (g1, p1) = build_parts(&cell, &obs, &aux, &hc.s, &cfg(1.0, Some(rs)));
        let m = moved(&g1);
        eprintln!("λ = {lam}: moved [orb smooth, orb, aux smooth, aux] = {m:?}");
        assert!(m[1] > 0 && m[3] > 0, "split path did not run: {m:?}");
        assert_eq!((m[0], m[2]), (0, 0), "λ = {lam} moved something: {m:?}");
        assert_eq!(g1.range_split(), Some(rs));
        assert!(bitwise_eq(&p1.j2, &p0.j2), "λ = {lam}: J2 not bitwise");
        assert!(bitwise_eq(&p1.j3, &p0.j3), "λ = {lam}: J3 not bitwise");
        assert!(bitwise_eq(g1.b(), g0.b()), "λ = {lam}: B not bitwise");
        let (s0, s1) = (g0.stats(), g1.stats());
        assert_eq!(s1.n_sr3_triplets, s0.n_sr3_triplets);
        assert_eq!(s1.n_sr2_pairs, s0.n_sr2_pairs);
        let e1 = energy(&cell, &obs, &hc, &g1);
        assert_eq!(e1.to_bits(), e0.to_bits(), "λ = {lam}: E {e1} vs {e0}");
    }
}

// --------------------------------------------------------------- anchor 2 --

fn anchor_sites(cell: &Cell, classes: &[usize]) -> Vec<[f64; 4]> {
    let r = cell.positions();
    let a = cell.lattice();
    let mut out = Vec::new();
    for (i, j) in [(0usize, 0usize), (1, 1), (0, 1)] {
        for k in 0..8usize {
            if !classes.contains(&k) {
                continue;
            }
            let h = [(k >> 2) & 1, (k >> 1) & 1, k & 1];
            let mut c = [0.0; 3];
            for d in 0..3 {
                let ha: f64 = (0..3).map(|x| h[x] as f64 * a[x][d]).sum();
                c[d] = 0.5 * (r[i][d] + r[j][d] + ha);
            }
            out.push([c[0], c[1], c[2], 2.0 * ANCHOR_ALPHA]);
        }
    }
    out
}

struct Anchor {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
    eri: DenseAftEri,
    site: SiteBasis,
}

/// `pbc_rsgdf.rs`'s exact-span anchor: one s (α = 0.5) per H, cubic a = 4,
/// the 24 pair-product aux (α = 1) on ghost sites.
fn anchor() -> Anchor {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    let hc = hcore(&cell, &prep);
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("dense AFT");
    let site = SiteBasis::new(&anchor_sites(&cell, &[0, 1, 2, 3, 4, 5, 6, 7]), 0).unwrap();
    Anchor {
        cell,
        prep,
        hc,
        eri,
        site,
    }
}

impl Anchor {
    fn build(&self, c: &RsGdfConfig) -> Result<RsGdf, ferric_core::FerricError> {
        RsGdf::build(&self.cell, &self.prep, &self.site.prep, &self.hc.s, c)
    }

    fn miss(&self, gdf: &RsGdf) -> f64 {
        max_abs_diff(
            &gdf.fitted_eri(DEFAULT_FITTED_ERI_MAX_BYTES).unwrap(),
            self.eri.eri(),
        )
    }
}

#[test]
fn everything_moved_reproduces_the_exact_span_anchor() {
    let an = anchor();
    // λ = 1: orbital 0.5 ≤ λω²/2 and aux 1 ≤ λω² for ω ≥ √1 — ω = 1 sits
    // ON the boundary, so use 1.2 and 1.5 (the prototype's "everything").
    for omega in [1.2, 1.5] {
        let gdf = an.build(&cfg(omega, Some(RangeSplit::default()))).unwrap();
        let m = moved(&gdf);
        let st = gdf.stats();
        let d = an.miss(&gdf);
        let e_ref = gamma_rhf(&an.cell, &an.prep, &an.hc, &an.eri).energy;
        let e = energy(&an.cell, &an.prep, &an.hc, &gdf);
        eprintln!(
            "span ω={omega}: moved {m:?}; SR3 {} SR2 {}; kept {}/{} (eig {:.3e}..{:.3e}); \
             max|I_fit − I| = {d:.2e}; dE = {:.2e}",
            st.n_sr3_triplets,
            st.n_sr2_pairs,
            st.naux_kept,
            st.naux,
            st.metric_eig_min,
            st.metric_eig_max,
            e - e_ref
        );
        assert_eq!((m[0], m[2]), (m[1], m[3]), "not everything moved: {m:?}");
        assert_eq!(st.n_sr3_triplets, 0, "an SR 3-centre call survived");
        assert_eq!(st.n_sr2_pairs, 0, "an SR metric call survived");
        assert!(d <= 1e-12, "ω={omega}: fitted ERI off by {d:.3e}");
        assert!(
            (e - e_ref).abs() <= 1e-10,
            "ω={omega}: dE {:.3e}",
            e - e_ref
        );
    }
    // λ = 1 at ω = 0.8 moves nothing (0.5 > 0.32, 1 > 0.64): today's anchor.
    let gdf = an.build(&cfg(0.8, Some(RangeSplit::default()))).unwrap();
    let m = moved(&gdf);
    assert_eq!((m[0], m[2]), (0, 0), "{m:?}");
    let d = an.miss(&gdf);
    eprintln!("span ω=0.8 (nothing moved): max|I_fit − I| = {d:.2e}");
    assert!(d < 1e-10, "{d:.3e}");
}

// ------------------------------------------------------------- mutants ----

#[test]
fn split_g0_and_double_count_mutants_fail_the_span_anchor() {
    let an = anchor();
    let omega = 1.2;
    for (m, floor) in [
        // J3's subtract with the full (S, q): the metric stays correct.
        (RangeSplitMutant::ThreeIndexFullG0, 1e-4),
        // Moved blocks also left in real space (prototype "both": 2.9).
        (RangeSplitMutant::DoubleCount, 1e-2),
    ] {
        let gdf = an
            .build(&cfg(omega, mutant(m)))
            .unwrap_or_else(|e| panic!("{m:?} build failed: {e}"));
        let d = an.miss(&gdf);
        eprintln!(
            "{m:?}: max|ΔI| = {d:.3e} (eig min {:.3e})",
            gdf.stats().metric_eig_min
        );
        assert!(d > floor, "{m:?} not detected by the span anchor: {d:.3e}");
    }
}

#[test]
fn metric_guard_refuses_a_wrong_g0_metric() {
    let an = anchor();
    for omega in [1.2, 1.5] {
        // The production split passes the guard …
        let ok = an.build(&cfg(omega, Some(RangeSplit::default()))).unwrap();
        assert!(ok.stats().metric_eig_min >= -RANGE_SPLIT_NEG_EIG_GUARD);
        // … the metric-only G = 0 mutant (prototype −12.7 at ω = 1.2) does not.
        let err = an
            .build(&cfg(omega, mutant(RangeSplitMutant::MetricFullG0)))
            .expect_err("the wrong-G=0 metric was accepted");
        let msg = err.to_string();
        eprintln!("ω={omega}: {msg}");
        assert!(msg.contains("smallest metric eigenvalue"), "{msg}");
    }
}

/// The Gamma gradient build ACCEPTS a split (its forces/stress follow the
/// partition: `tests/pbc_grad_rsgdf_split.rs`). The k-point build accepts
/// the production split (`tests/pbc_krsgdf_split.rs` owns its physics) and
/// REFUSES the split mutants (they are Gamma energy anchors). With
/// everything moved on the exact span, its 1x1x1 fitted kernels are the
/// Gamma split's fitted ERI (and hence the dense AFT one).
#[test]
fn kpoint_build_takes_the_production_split_only_and_the_gradient_build_accepts_it() {
    let an = anchor();
    let c = cfg(1.2, Some(RangeSplit::default()));
    let g = RsGdf::build_for_gradient(&an.cell, &an.prep, &an.site.prep, &an.hc.s, &c)
        .expect("the gradient build must accept a range split");
    assert!(g.has_gradient_parts());
    assert_eq!(g.range_split(), Some(RangeSplit::default()));
    let mesh = KPointMesh::gamma_centred(&an.cell, [1, 1, 1]).unwrap();
    let s_k = vec![an.hc.s.mapv(|x| Complex64::new(x, 0.0))];
    let kc = KRsGdfConfig {
        gdf: c,
        ..Default::default()
    };
    let kg = KRsGdf::build(&an.cell, &an.prep, &an.site.prep, &mesh, &s_k, &kc)
        .expect("the k-point build must accept the production range split");
    let names: Vec<&str> = kg.stats().split_counters.iter().map(|(n, _)| *n).collect();
    assert!(
        names.contains(&"rsgdf split aux prims smooth"),
        "split counters missing: {names:?}"
    );
    let (jf, _) = kg
        .fitted_kernels(DEFAULT_K_FITTED_KERNELS_MAX_BYTES)
        .unwrap();
    let eri = g.fitted_eri(DEFAULT_FITTED_ERI_MAX_BYTES).unwrap();
    let d = jf[0].iter().zip(eri.iter()).fold(0.0_f64, |m, (x, y)| {
        m.max((x - Complex64::new(*y, 0.0)).norm())
    });
    let dd = jf[0]
        .iter()
        .zip(an.eri.eri().iter())
        .fold(0.0_f64, |m, (x, y)| {
            m.max((x - Complex64::new(*y, 0.0)).norm())
        });
    eprintln!("span ω=1.2 split, 1x1x1 k vs Gamma split: {d:.2e}; vs dense AFT {dd:.2e}");
    assert!(d < 1e-12, "k 1x1x1 split vs Gamma split {d:.3e}");
    assert!(dd < 1e-12, "k 1x1x1 split vs dense AFT {dd:.3e}");
    for m in [
        RangeSplitMutant::ThreeIndexFullG0,
        RangeSplitMutant::MetricFullG0,
        RangeSplitMutant::DoubleCount,
        RangeSplitMutant::AllOrbitalSmooth,
    ] {
        let kc = KRsGdfConfig {
            gdf: cfg(1.2, mutant(m)),
            ..Default::default()
        };
        let msg = KRsGdf::build(&an.cell, &an.prep, &an.site.prep, &mesh, &s_k, &kc)
            .expect_err("k-point build accepted a range-split mutant")
            .to_string();
        assert!(msg.contains("Production"), "{m:?}: {msg}");
    }
    assert!(an.build(&cfg(1.2, Some(RangeSplit::new(-1.0)))).is_err());
    assert!(an
        .build(&cfg(1.2, Some(RangeSplit::new(f64::NAN))))
        .is_err());
}

// ---------------------------------------------- split vs unsplit (cheap) --

#[test]
fn split_matches_unsplit_and_is_omega_independent_on_h2() {
    let cell = h2_cell(4.0);
    let obs = prep_for(&cell, &pyscf_sto3g_h());
    let hc = hcore(&cell, &obs);
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let eri = |g: &RsGdf| g.fitted_eri(DEFAULT_FITTED_ERI_MAX_BYTES).unwrap();
    let un = RsGdf::build(&cell, &obs, &aux, &hc.s, &cfg(1.0, None)).unwrap();
    let mut split = Vec::new();
    for omega in [0.7, 1.0, 1.4] {
        let g = RsGdf::build(
            &cell,
            &obs,
            &aux,
            &hc.s,
            &cfg(omega, Some(RangeSplit::default())),
        )
        .unwrap();
        let m = moved(&g);
        eprintln!(
            "H2 ω={omega}: moved {m:?}, SR3 {} (unsplit ω=1: {})",
            g.stats().n_sr3_triplets,
            un.stats().n_sr3_triplets
        );
        assert!(m[0] > 0 && m[2] > 0, "ω={omega}: nothing moved {m:?}");
        split.push(eri(&g));
    }
    let d_un = max_abs_diff(&split[1], &eri(&un));
    let d_w = max_abs_diff(&split[0], &split[2]);
    eprintln!("H2: max|I_split − I_unsplit| (ω=1) = {d_un:.2e}; split ω 0.7 vs 1.4 = {d_w:.2e}");
    assert!(d_un < 1e-10, "{d_un:.3e}");
    assert!(d_w < 1e-9, "{d_w:.3e}");
}

// --------------------------------------------- split vs unsplit (triclinic) --

#[test]
fn split_matches_unsplit_on_the_triclinic_cell() {
    let cell = triclinic_cell();
    let obs = prep_for(&cell, &sp_basis_h());
    let hc = hcore(&cell, &obs);
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let c0 = cfg(1.0, None);
    let c1 = cfg(1.0, Some(RangeSplit::default()));
    let (g0, p0) = build_parts(&cell, &obs, &aux, &hc.s, &c0);
    let (g1, p1) = build_parts(&cell, &obs, &aux, &hc.s, &c1);
    let m = moved(&g1);
    let dj2 = max_abs_diff(&p1.j2, &p0.j2);
    let dj3 = max_abs_diff(&p1.j3, &p0.j3);
    let di = max_abs_diff(
        &g1.fitted_eri(DEFAULT_FITTED_ERI_MAX_BYTES).unwrap(),
        &g0.fitted_eri(DEFAULT_FITTED_ERI_MAX_BYTES).unwrap(),
    );
    let (e0, e1) = (energy(&cell, &obs, &hc, &g0), energy(&cell, &obs, &hc, &g1));
    let (s0, s1) = (g0.stats(), g1.stats());
    eprintln!(
        "tri s+p/cc-pvdz-ri ω=1: moved {m:?}; max|ΔJ2| {dj2:.2e} |ΔJ3| {dj3:.2e} |ΔI| {di:.2e}; \
         E {e0:.13} → {e1:.13} (ΔE {:.2e}); SR3 {} → {} ({:.3}x), SR2 {} → {}",
        e1 - e0,
        s0.n_sr3_triplets,
        s1.n_sr3_triplets,
        s1.n_sr3_triplets as f64 / s0.n_sr3_triplets as f64,
        s0.n_sr2_pairs,
        s1.n_sr2_pairs
    );
    // Prototype: 4/16 orbital prims, 8/24 aux shells moved; ΔJ3 1.35e-12,
    // ΔJ2 6.29e-12, ΔI 1.56e-12, ΔE −1.9e-14.
    assert!(m[0] > 0 && m[0] < m[1] && m[2] > 0 && m[2] < m[3], "{m:?}");
    assert!(dj3 < 1e-10, "ΔJ3 {dj3:.3e}");
    assert!(dj2 < 1e-10, "ΔJ2 {dj2:.3e}");
    assert!(di < 1e-10, "ΔI {di:.3e}");
    assert!((e1 - e0).abs() <= 1e-12, "ΔE {:.3e}", e1 - e0);
    // The walks shrink, and the integral-free counter reproduces both builds.
    assert!(s1.n_sr3_triplets < s0.n_sr3_triplets);
    assert!(s1.n_sr2_pairs < s0.n_sr2_pairs);
    for (c, s) in [(&c0, s0), (&c1, s1)] {
        let n = sr_walk_counts(&cell, &obs, &aux, c).unwrap();
        assert_eq!(n.n_sr3_triplets, s.n_sr3_triplets);
        assert_eq!(n.n_sr3_triplets_ordered, s.n_sr3_triplets_ordered);
        assert_eq!(n.n_sr2_pairs, s.n_sr2_pairs);
    }
}

// ------------------------------------------------------ diamond (release) --

fn diamond(orbital: &str) -> (Cell, PreparedBasis, PreparedBasis) {
    let mol = Molecule::parse_xyz(DIAMOND_PRIM_XYZ, 0, 1).unwrap();
    let cell = Cell::new(mol, DIAMOND_PRIM_LATTICE).unwrap();
    let obs = PreparedBasis::new(cell.mol(), &basis::bundled(orbital).unwrap()).unwrap();
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    (cell, obs, aux)
}

#[test]
#[ignore = "slow: four diamond_prim STO-3G/cc-pvdz-ri RS-GDF builds (unsplit ~18 s SR3 in release \
            at 1 thread) plus two SCFs; run with --release -- --ignored, serially, on a quiet box"]
fn diamond_sto3g_split_matches_ferric_and_moving_the_core_misses() {
    let (cell, obs, aux) = diamond("sto-3g");
    let hc = periodic_hcore(
        &cell,
        &obs,
        &PeriodicHcoreConfig::with_omega(default_ewald_omega(&cell)),
    )
    .unwrap();
    let ewald = |split| RsGdfConfig {
        exxdiv: ExxDiv::Ewald,
        ..cfg(1.0, split)
    };
    let (g0, p0) = build_parts(&cell, &obs, &aux, &hc.s, &ewald(None));
    let (g1, p1) = build_parts(
        &cell,
        &obs,
        &aux,
        &hc.s,
        &ewald(Some(RangeSplit::default())),
    );
    let (s0, s1) = (g0.stats(), g1.stats());
    let m = moved(&g1);
    let (e0, e1) = (energy(&cell, &obs, &hc, &g0), energy(&cell, &obs, &hc, &g1));
    let dj3 = max_abs_diff(&p1.j3, &p0.j3);
    let dj2 = max_abs_diff(&p1.j2, &p0.j2);
    eprintln!(
        "diamond STO-3G/cc-pvdz-ri ω=1: moved {m:?}; E unsplit {e0:.10} split {e1:.10} \
         (ΔE {:.2e}; bench {DIAMOND_STO3G_E}); max|ΔJ3| {dj3:.2e} |ΔJ2| {dj2:.2e}; \
         SR3 {} → {} ({:.3}x), SR2 {} → {}",
        e1 - e0,
        s0.n_sr3_triplets,
        s1.n_sr3_triplets,
        s1.n_sr3_triplets as f64 / s0.n_sr3_triplets as f64,
        s0.n_sr2_pairs,
        s1.n_sr2_pairs
    );
    // Today's build is the benchmark's (counters and energy). The pin is
    // the pre-s2 ORDERED count; the s2 build reports it as the
    // ordered-equivalent (rsgdf module doc "Orbital-pair symmetry").
    assert_eq!(s0.n_sr3_triplets_ordered, 16_023_080);
    assert_eq!(s0.n_sr2_pairs, 159_064);
    assert!((e0 - DIAMOND_STO3G_E).abs() <= 1e-9, "unsplit {e0:.12}");
    // Prototype: 4 smooth orbital prims (the 0.222 C 2sp), ΔE 2.6e-10.
    assert_eq!(m[0], 4, "{m:?}");
    assert!((e1 - e0).abs() <= 1e-9, "ΔE {:.3e}", e1 - e0);
    assert!((e1 - DIAMOND_STO3G_E).abs() <= 1e-9, "split {e1:.12}");

    // Moving compact blocks (C 1s) must miss (prototype: orbital-all 6.7e-2
    // in J3 with J2 untouched; everything 6.7e-2 / 3.4e-2).
    let (_, pm) = build_parts(
        &cell,
        &obs,
        &aux,
        &hc.s,
        &ewald(mutant(RangeSplitMutant::AllOrbitalSmooth)),
    );
    let d_orb = max_abs_diff(&pm.j3, &p0.j3);
    eprintln!("AllOrbitalSmooth: max|ΔJ3| = {d_orb:.3e}");
    assert!(
        d_orb >= 1e-2,
        "moving the C 1s pairs went undetected: {d_orb:.3e}"
    );
    match RsGdf::build_with_fit_parts(&cell, &obs, &aux, &hc.s, &ewald(Some(RangeSplit::new(1e3))))
    {
        Ok((_, pl)) => {
            let d = max_abs_diff(&pl.j3, &p0.j3);
            eprintln!("λ = 1e3 (everything moved): max|ΔJ3| = {d:.3e}");
            assert!(d >= 1e-2, "moving everything went undetected: {d:.3e}");
        }
        Err(e) => eprintln!("λ = 1e3 (everything moved): refused by the guard: {e}"),
    }
}

#[test]
#[ignore = "slow: integral-free walk over ~2.3e8 diamond_prim cc-pVDZ/cc-pvdz-ri SR triplets; \
            run with --release -- --ignored"]
fn diamond_ccpvdz_split_visits_under_a_quarter_of_the_sr3_triplets() {
    let (cell, obs, aux) = diamond("cc-pvdz");
    // Column rotation OFF: the pins below are the UNROTATED walk (FINDINGS
    // "Iteration 23"); the default `Auto` would count the rotated walk
    // (cc-pVDZ C rotates), `pbc_sr_rotation.rs` covers that one.
    let unrotated = |split| RsGdfConfig {
        sr_column_rotation: SrColumnRotation::Off,
        ..cfg(1.0, split)
    };
    let un = sr_walk_counts(&cell, &obs, &aux, &unrotated(None)).unwrap();
    let sp = sr_walk_counts(&cell, &obs, &aux, &unrotated(Some(RangeSplit::default()))).unwrap();
    // Ordered-walk (s1) units: the prototype counters' unit.
    let r3 = sp.n_sr3_triplets_s1 as f64 / un.n_sr3_triplets_s1 as f64;
    let r2 = sp.n_sr2_pairs as f64 / un.n_sr2_pairs as f64;
    eprintln!(
        "diamond cc-pVDZ/cc-pvdz-ri ω=1: SR3 {} → {} ({r3:.4}x; prototype twocall 35,921,128 = \
         0.182x, trim 0.147x); SR2 {} → {} ({r2:.4}x; prototype 17,288)",
        un.n_sr3_triplets_s1, sp.n_sr3_triplets_s1, un.n_sr2_pairs, sp.n_sr2_pairs
    );
    eprintln!(
        "  s2 (computed / ordered-equivalent): unsplit {} / {}, split {} / {}",
        un.n_sr3_triplets, un.n_sr3_triplets_ordered, sp.n_sr3_triplets, sp.n_sr3_triplets_ordered
    );
    // ferric's own counter (FINDINGS "Iteration 23" calibration, exact).
    assert_eq!(un.n_sr3_triplets_s1, 197_033_528);
    assert_eq!(un.n_sr3_triplets_ordered, un.n_sr3_triplets_s1);
    assert_eq!(un.n_sr2_pairs, 159_064);
    assert!(r3 < 0.25, "SR3 ratio {r3:.4}");
    assert!(r2 < 0.25, "SR2 ratio {r2:.4}");
}
