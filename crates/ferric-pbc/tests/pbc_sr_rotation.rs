//! Column rotation of generally contracted orbital shells inside the Gamma
//! SR walks (`ferric_pbc::sr_rotation`; `rsgdf` / `hcore` module docs
//! "Column rotation"; design `reference/pbc/sr-general-contraction-design.md`
//! §3 (d), §4 step 2).
//!
//! Construction under test: within each (element, l, exponent list) group,
//! the single-primitive columns are subtracted from the other columns; the
//! Gamma SR 3-centre walk (RS-GDF, unsplit and range split) and the hcore SR
//! attraction run on the rotated shells, and the finished blocks are
//! transformed back exactly, `J3 = (T ⊗ T) J3'`, `V_SR = T V' Tᵀ`.
//!
//! What each test pins:
//!
//! (a) TRIVIAL-LIMIT ANCHOR (written first): on a basis with nothing to
//!     rotate the rotation is the identity and every build — flag on, and
//!     every mutant — is BITWISE today's build (J3, B, S, T, V_SR, h,
//!     counters). The counter `… rotated columns` = 0 proves the flag was on.
//! (b) COVARIANCE ANCHOR (an independent construction: the parent walk on
//!     the parent shells vs the rotated walk + back-transform): at a tight
//!     truncation `P_TIGHT` = 1e-16, `max|X_rot − X| ≤ TOL_COV · max|X|`
//!     for X = J3 (unsplit and split) and V_SR. Derivation of
//!     `TOL_COV` = 1e-13 (measure-free; the measured values are printed):
//!     * truncation: each build skips only triplets whose bound is below
//!       `P_TIGHT`; the two kept sets differ only near the radius, so the
//!       difference is at most (triplets that differ per element) ×
//!       `P_TIGHT` ≲ 1e3 × 1e-16 = 1e-13 absolute, against `max|J3|`,
//!       `max|V_SR|` ≥ 1 for cc-pVDZ core functions — a ceiling, the bound
//!       overestimates real tails by orders (`pbc_sr_screening`);
//!     * round-off: libint evaluates a rotated contraction from a different
//!       primitive sum (relative ≲ 1e-14 per integral, times
//!       `1 + Σ|T_ks|` ≤ 1.6 on cc-pVDZ), and the back-transform adds
//!       ≤ 4 terms (`γ_4 ≈ 9e-16` relative);
//!     so a correct rotation sits at ~1e-15..1e-14 relative, a decade below
//!     the bar. If a first run measures more than `TOL_COV`, the anchor has
//!     found a construction problem: do NOT raise the bar.
//! (c) The Gamma RHF energy with rotated hcore + RS-GDF (default screening,
//!     unsplit and split) matches the unrotated build to `E_BAR` = 1e-10 Ha
//!     on a C/O/H cell (formaldehyde / cc-pVDZ).
//! (d) The rotated builds are BITWISE identical across 1/2/6 threads.
//! (e) Counters: the rotated walk computes fewer SR triplets (every rotated
//!     primitive set is a subset of its parent's, so no pair radius grows;
//!     with the range split at ω = 1, λ = 1 every rotated C/O column is
//!     wholly compact or wholly smooth, so their pairs need one kept call
//!     instead of two — H's rotated s keeps its smooth 0.4446 primitive next
//!     to two compact ones and stays split), `sr_walk_counts` reproduces the
//!     rotated build's counters without integrals, and the rotated-column
//!     counter is exact (H2O: O 1s, 2s, p and one s per H = 5).
//! (f) Mutants, each must miss `TOL_COV` by ≥ `MUTANT_FLOOR` = 1e-6
//!     relative on the fixture where production passes: wrong sign of `T_ks`,
//!     `T` without libint's renormalisation, the back-transform on one index
//!     only, and the AUX basis rotated by mistake (run with a generally
//!     contracted aux basis — cc-pVDZ itself — since cc-pvdz-ri is segmented
//!     and the mutant would be a no-op there; production on the same aux
//!     passes (b)'s bar).
//!
//! Artifact hypothesis, stated before measuring. Correct: (b) at the
//! ~1e-15 level and shrinking with `P_TIGHT`. Missing renormalisation:
//! O(1) relative (cc-pVDZ `T_kk` 0.50..1.0006), independent of precision.
//! Wrong sign: O(|T_ks|) ≈ 0.5 relative on the rows of rotated columns.
//! One-sided transform: O(|T_ks|) on pairs of two rotated columns. Aux
//! rotated: O(1) on the aux columns of rotated aux shells. Smooth piece not
//! rotated (a split built on the parent shells but back-transformed): an
//! ω-dependent error that does not shrink with precision (not a mutant
//! here: the rotated split plan is built from the rotated stage by
//! construction).

mod common;

use common::*;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::hcore::{
    periodic_hcore, periodic_hcore_pair_s1_oracle, PeriodicHcore, PeriodicHcoreConfig,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::{sr_walk_counts, RangeSplit, RsGdf, RsGdfConfig};
use ferric_pbc::sr_rotation::{ColumnRotation, ColumnRotationMutant};
use ndarray::Array2;

const THREADS: [usize; 3] = [1, 2, 6];
const HCORE_OMEGA: f64 = 0.8;
/// (b): the tight truncation of the covariance anchor.
const P_TIGHT: f64 = 1e-16;
/// (b): relative bar of the covariance anchor (module doc derivation).
const TOL_COV: f64 = 1e-13;
/// (f): smallest relative miss a mutant must show.
const MUTANT_FLOOR: f64 = 1e-6;
/// (c): RHF energy bar (Ha).
const E_BAR: f64 = 1e-10;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

const FORMALDEHYDE: &str = "4
formaldehyde
C  0.0000  0.0000 -0.5290
O  0.0000  0.0000  0.6760
H  0.0000  0.9430 -1.1160
H  0.0000 -0.9430 -1.1160
";

fn in_pool<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("rayon pool")
        .install(f)
}

fn cell_of(xyz: &str, a: f64) -> Cell {
    Cell::new(Molecule::parse_xyz(xyz, 0, 1).expect("xyz"), cubic(a)).expect("cell")
}

fn bundled(cell: &Cell, name: &str) -> PreparedBasis {
    prep_for(cell, &basis::bundled(name).expect("bundled basis"))
}

fn rotation(m: ColumnRotationMutant) -> Option<ColumnRotation> {
    Some(ColumnRotation { mutant: m })
}

fn production() -> Option<ColumnRotation> {
    Some(ColumnRotation::new())
}

fn gdf_cfg(split: Option<RangeSplit>, rot: Option<ColumnRotation>, precision: f64) -> RsGdfConfig {
    RsGdfConfig {
        omega: 1.0,
        precision,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(1 << 30),
        range_split: split,
        sr_column_rotation: rot,
        ..Default::default()
    }
}

fn hcore_cfg(rot: Option<ColumnRotation>, precision: f64) -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision,
        sr_column_rotation: rot,
        ..PeriodicHcoreConfig::with_omega(HCORE_OMEGA)
    }
}

/// `(RsGdf, symmetrised J3 before the metric solve)`.
fn gdf_j3(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: &PreparedBasis,
    s: &Array2<f64>,
    cfg: &RsGdfConfig,
) -> (RsGdf, Array2<f64>) {
    let (g, parts) = RsGdf::build_with_fit_parts(cell, prep, aux, s, cfg).expect("rsgdf");
    (g, parts.j3)
}

fn bit_diffs(a: &Array2<f64>, b: &Array2<f64>) -> usize {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn assert_bitwise(a: &Array2<f64>, b: &Array2<f64>, what: &str) {
    let d = bit_diffs(a, b);
    assert_eq!(d, 0, "{what}: {d} of {} elements differ in bits", a.len());
}

fn max_abs(m: &Array2<f64>) -> f64 {
    m.iter().fold(0.0_f64, |a, x| a.max(x.abs()))
}

/// `max|a − b| / max|b|`.
fn rel(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let s = max_abs(b);
    assert!(
        s > 0.0,
        "reference is identically zero (vacuous comparison)"
    );
    max_abs_diff(a, b) / s
}

fn energy(cell: &Cell, prep: &PreparedBasis, hc: &PeriodicHcore, g: &RsGdf) -> f64 {
    gamma_rhf_jk(
        cell,
        prep,
        hc,
        Box::new(g.j_builder()),
        Box::new(g.k_builder()),
    )
    .energy
}

// ------------------------------------------------------------------ (a)

#[test]
fn identity_rotation_is_bitwise_todays_build() {
    let cell = triclinic_cell();
    // STO-3G s + one Cartesian p: the two shells have different l, so no
    // group has two columns and nothing rotates.
    let prep = prep_for(&cell, &sp_basis_h());
    let aux = bundled(&cell, "cc-pvdz-ri");
    let hc0 = periodic_hcore(&cell, &prep, &hcore_cfg(None, 1e-14)).expect("hcore");
    assert_eq!(hc0.timings.counter("hcore SR rotated columns"), None);
    for m in [
        ColumnRotationMutant::Production,
        ColumnRotationMutant::FlipSign,
        ColumnRotationMutant::NoRenormalization,
        ColumnRotationMutant::OneSidedBackTransform,
    ] {
        let hc1 = periodic_hcore(&cell, &prep, &hcore_cfg(rotation(m), 1e-14)).expect("hcore");
        for (a, b, what) in [
            (&hc1.s, &hc0.s, "S"),
            (&hc1.t, &hc0.t, "T"),
            (&hc1.v_sr, &hc0.v_sr, "V_SR"),
            (&hc1.h, &hc0.h, "h"),
        ] {
            assert_bitwise(a, b, &format!("{m:?}: identity-rotation {what}"));
        }
        assert_eq!(hc1.n_sr_triplets, hc0.n_sr_triplets, "{m:?}: hcore count");
        assert_eq!(
            hc1.timings.counter("hcore SR rotated columns"),
            Some(0),
            "{m:?}: the flag was on and detected nothing"
        );
    }
    for split in [None, Some(RangeSplit::default())] {
        let (g0, j0) = gdf_j3(&cell, &prep, &aux, &hc0.s, &gdf_cfg(split, None, 1e-13));
        assert_eq!(g0.timings().counter("rsgdf SR3 rotated columns"), None);
        for m in [
            ColumnRotationMutant::Production,
            ColumnRotationMutant::FlipSign,
            ColumnRotationMutant::NoRenormalization,
            ColumnRotationMutant::OneSidedBackTransform,
            ColumnRotationMutant::RotateAux,
        ] {
            let cfg = gdf_cfg(split, rotation(m), 1e-13);
            let (g1, j1) = gdf_j3(&cell, &prep, &aux, &hc0.s, &cfg);
            let tag = format!("split {:?}, {m:?}", split.map(|s| s.lambda));
            assert_bitwise(&j1, &j0, &format!("{tag}: identity-rotation J3"));
            assert_bitwise(g1.b(), g0.b(), &format!("{tag}: identity-rotation B"));
            let (s1, s0) = (g1.stats(), g0.stats());
            assert_eq!(
                (s1.n_sr3_triplets, s1.n_sr3_triplets_ordered, s1.n_sr2_pairs),
                (s0.n_sr3_triplets, s0.n_sr3_triplets_ordered, s0.n_sr2_pairs),
                "{tag}: counters"
            );
            assert_eq!(
                g1.timings().counter("rsgdf SR3 rotated columns"),
                Some(0),
                "{tag}: the flag was on and detected nothing"
            );
            let wc = sr_walk_counts(&cell, &prep, &aux, &cfg).expect("walk counts");
            assert_eq!(wc.n_sr3_triplets, s0.n_sr3_triplets, "{tag}: walk counts");
        }
    }
}

// ------------------------------------------------------- (b) and (e)

#[test]
fn rotated_sr_blocks_back_transform_to_the_parent_basis() {
    let cell = cell_of(WATER, 8.0);
    let prep = bundled(&cell, "cc-pvdz");
    let aux = bundled(&cell, "cc-pvdz-ri");
    // hcore: V_SR covariance; S, T, V_LR untouched bit for bit.
    let hc0 = periodic_hcore(&cell, &prep, &hcore_cfg(None, P_TIGHT)).expect("hcore");
    let hc1 = periodic_hcore(&cell, &prep, &hcore_cfg(production(), P_TIGHT)).expect("hcore");
    for (a, b, what) in [
        (&hc1.s, &hc0.s, "S"),
        (&hc1.t, &hc0.t, "T"),
        (&hc1.v_lr, &hc0.v_lr, "V_LR"),
    ] {
        assert_bitwise(a, b, &format!("{what} outside the rotated walk"));
    }
    assert_bitwise(&hc1.v_sr, &hc1.v_sr.t().to_owned(), "rotated V_SR symmetry");
    let dv = rel(&hc1.v_sr, &hc0.v_sr);
    eprintln!(
        "V_SR: max|V| {:.3e}, rel diff {dv:.2e}; SR triplets rotated {} vs parent {}",
        max_abs(&hc0.v_sr),
        hc1.n_sr_triplets,
        hc0.n_sr_triplets
    );
    assert!(
        dv > 0.0,
        "rotated V_SR bitwise the parent's: rotation not applied?"
    );
    assert!(dv <= TOL_COV, "V_SR covariance: rel {dv:.3e} > {TOL_COV:e}");
    assert_eq!(hc1.timings.counter("hcore SR rotated columns"), Some(5));
    assert!(hc1.n_sr_triplets < hc0.n_sr_triplets, "(e) hcore count");
    for split in [None, Some(RangeSplit::default())] {
        let tag = format!("split {:?}", split.map(|s| s.lambda));
        let (g0, j0) = gdf_j3(&cell, &prep, &aux, &hc0.s, &gdf_cfg(split, None, P_TIGHT));
        let (g1, j1) = gdf_j3(
            &cell,
            &prep,
            &aux,
            &hc0.s,
            &gdf_cfg(split, production(), P_TIGHT),
        );
        let dj = rel(&j1, &j0);
        let (s0, s1) = (g0.stats(), g1.stats());
        eprintln!(
            "{tag}: max|J3| {:.3e}, rel diff {dj:.2e}; asym_j3 {:.2e} vs {:.2e}; SR3 rotated {} \
             (ordered-equivalent {}) vs parent {} ({})",
            max_abs(&j0),
            s1.asym_j3,
            s0.asym_j3,
            s1.n_sr3_triplets,
            s1.n_sr3_triplets_ordered,
            s0.n_sr3_triplets,
            s0.n_sr3_triplets_ordered
        );
        assert!(dj > 0.0, "{tag}: rotated J3 bitwise the parent's");
        assert!(
            dj <= TOL_COV,
            "{tag}: J3 covariance rel {dj:.3e} > {TOL_COV:e}"
        );
        assert_eq!(
            g1.timings().counter("rsgdf SR3 rotated columns"),
            Some(5),
            "{tag}"
        );
        // (e) the walk shrinks; the integral-free counter is the build's.
        assert!(
            s1.n_sr3_triplets < s0.n_sr3_triplets,
            "{tag}: rotated walk {} vs parent {}",
            s1.n_sr3_triplets,
            s0.n_sr3_triplets
        );
        assert_eq!(s1.n_sr2_pairs, s0.n_sr2_pairs, "{tag}: SR metric untouched");
        let wc = sr_walk_counts(&cell, &prep, &aux, &gdf_cfg(split, production(), P_TIGHT))
            .expect("walk counts");
        assert_eq!(
            (wc.n_sr3_triplets, wc.n_sr3_triplets_ordered, wc.n_sr2_pairs),
            (s1.n_sr3_triplets, s1.n_sr3_triplets_ordered, s1.n_sr2_pairs),
            "{tag}: integral-free counts vs the rotated build"
        );
    }
}

/// (e) at the production screen, split and unsplit (integral-free counts).
#[test]
fn rotated_walk_counts_shrink() {
    let cell = cell_of(WATER, 8.0);
    let prep = bundled(&cell, "cc-pvdz");
    let aux = bundled(&cell, "cc-pvdz-ri");
    let split = Some(RangeSplit::default());
    let w0 = sr_walk_counts(&cell, &prep, &aux, &gdf_cfg(split, None, 1e-13)).unwrap();
    let w1 = sr_walk_counts(&cell, &prep, &aux, &gdf_cfg(split, production(), 1e-13)).unwrap();
    let u0 = sr_walk_counts(&cell, &prep, &aux, &gdf_cfg(None, None, 1e-13)).unwrap();
    let u1 = sr_walk_counts(&cell, &prep, &aux, &gdf_cfg(None, production(), 1e-13)).unwrap();
    eprintln!(
        "H2O/cc-pVDZ SR3 calls (s2 / ordered-equivalent / s1): split parent {} / {} / {}, \
         split rotated {} / {} / {} (ratio {:.3}); unsplit parent {} / {} / {}, unsplit rotated \
         {} / {} / {} (ratio {:.3})",
        w0.n_sr3_triplets,
        w0.n_sr3_triplets_ordered,
        w0.n_sr3_triplets_s1,
        w1.n_sr3_triplets,
        w1.n_sr3_triplets_ordered,
        w1.n_sr3_triplets_s1,
        w0.n_sr3_triplets as f64 / w1.n_sr3_triplets as f64,
        u0.n_sr3_triplets,
        u0.n_sr3_triplets_ordered,
        u0.n_sr3_triplets_s1,
        u1.n_sr3_triplets,
        u1.n_sr3_triplets_ordered,
        u1.n_sr3_triplets_s1,
        u0.n_sr3_triplets as f64 / u1.n_sr3_triplets as f64,
    );
    assert!(w1.n_sr3_triplets < w0.n_sr3_triplets, "split: no drop");
    // Subset primitives never enlarge a pair's radius (smaller charge
    // bound, larger smallest exponent), so the unsplit rotated walk keeps a
    // subset of the parent walk's triplets.
    assert!(u1.n_sr3_triplets < u0.n_sr3_triplets, "unsplit: no drop");
    assert_eq!(w1.n_sr2_pairs, w0.n_sr2_pairs, "SR metric untouched");
}

// ------------------------------------------------------------------ (c)

#[test]
fn rhf_energy_matches_the_unrotated_build() {
    let cell = cell_of(FORMALDEHYDE, 9.0);
    let prep = bundled(&cell, "cc-pvdz");
    let aux = bundled(&cell, "cc-pvdz-ri");
    let hc0 = periodic_hcore(&cell, &prep, &hcore_cfg(None, 1e-14)).expect("hcore");
    let hc1 = periodic_hcore(&cell, &prep, &hcore_cfg(production(), 1e-14)).expect("hcore");
    // C 1s, 2s, p; O 1s, 2s, p; one s per H.
    assert_eq!(hc1.timings.counter("hcore SR rotated columns"), Some(8));
    for split in [None, Some(RangeSplit::default())] {
        let tag = format!("H2CO/cc-pVDZ split {:?}", split.map(|s| s.lambda));
        let g0 =
            RsGdf::build(&cell, &prep, &aux, &hc0.s, &gdf_cfg(split, None, 1e-13)).expect("rsgdf");
        let g1 = RsGdf::build(
            &cell,
            &prep,
            &aux,
            &hc1.s,
            &gdf_cfg(split, production(), 1e-13),
        )
        .expect("rsgdf");
        let (e0, e1) = (
            energy(&cell, &prep, &hc0, &g0),
            energy(&cell, &prep, &hc1, &g1),
        );
        eprintln!(
            "{tag}: E rotated {e1:.15} vs parent {e0:.15} (ΔE {:.2e}); SR3 {} vs {}; hcore SR \
             {} vs {}",
            e1 - e0,
            g1.stats().n_sr3_triplets,
            g0.stats().n_sr3_triplets,
            hc1.n_sr_triplets,
            hc0.n_sr_triplets
        );
        assert!((e1 - e0).abs() <= E_BAR, "{tag}: ΔE {:.3e}", e1 - e0);
    }
}

// ------------------------------------------------------------------ (d)

#[test]
fn rotated_builds_are_bitwise_across_thread_counts() {
    let cell = cell_of(WATER, 8.0);
    let prep = bundled(&cell, "cc-pvdz");
    let aux = bundled(&cell, "cc-pvdz-ri");
    let hcfg = hcore_cfg(production(), 1e-14);
    let cfg = gdf_cfg(Some(RangeSplit::default()), production(), 1e-13);
    let runs: Vec<(PeriodicHcore, RsGdf, Array2<f64>)> = THREADS
        .iter()
        .map(|&t| {
            in_pool(t, || {
                let hc = periodic_hcore(&cell, &prep, &hcfg).expect("hcore");
                let (g, j3) = gdf_j3(&cell, &prep, &aux, &hc.s, &cfg);
                (hc, g, j3)
            })
        })
        .collect();
    let (hc_ref, g_ref, j_ref) = &runs[0];
    for (&t, (hc, g, j3)) in THREADS.iter().zip(&runs) {
        assert_bitwise(&hc.v_sr, &hc_ref.v_sr, &format!("V_SR at {t} threads"));
        assert_bitwise(&hc.h, &hc_ref.h, &format!("h at {t} threads"));
        assert_bitwise(j3, j_ref, &format!("J3 at {t} threads"));
        assert_bitwise(g.b(), g_ref.b(), &format!("B at {t} threads"));
        assert_eq!(hc.n_sr_triplets, hc_ref.n_sr_triplets, "{t} threads");
        assert_eq!(
            g.stats().n_sr3_triplets,
            g_ref.stats().n_sr3_triplets,
            "{t} threads"
        );
    }
}

// ------------------------------------------------------------------ (f)

#[test]
fn mutants_fail_the_covariance_anchor() {
    let cell = cell_of(WATER, 8.0);
    let prep = bundled(&cell, "cc-pvdz");
    let aux = bundled(&cell, "cc-pvdz-ri");
    let hc0 = periodic_hcore(&cell, &prep, &hcore_cfg(None, P_TIGHT)).expect("hcore");
    let (_, j0) = gdf_j3(&cell, &prep, &aux, &hc0.s, &gdf_cfg(None, None, P_TIGHT));
    for m in [
        ColumnRotationMutant::FlipSign,
        ColumnRotationMutant::NoRenormalization,
        ColumnRotationMutant::OneSidedBackTransform,
    ] {
        let hc = periodic_hcore(&cell, &prep, &hcore_cfg(rotation(m), P_TIGHT)).expect("hcore");
        let (_, j3) = gdf_j3(
            &cell,
            &prep,
            &aux,
            &hc0.s,
            &gdf_cfg(None, rotation(m), P_TIGHT),
        );
        let (dv, dj) = (rel(&hc.v_sr, &hc0.v_sr), rel(&j3, &j0));
        eprintln!("{m:?}: V_SR rel {dv:.2e}, J3 rel {dj:.2e}");
        assert!(
            dv >= MUTANT_FLOOR,
            "{m:?} survived the V_SR anchor ({dv:.2e})"
        );
        assert!(
            dj >= MUTANT_FLOOR,
            "{m:?} survived the J3 anchor ({dj:.2e})"
        );
    }
    // Aux rotated by mistake: needs a generally contracted AUX basis
    // (cc-pVDZ as the fitting basis). Production on the same aux passes the
    // bar, so the failure is the mutant's, not the fixture's.
    let aux_gc = bundled(&cell, "cc-pvdz");
    let (_, jg0) = gdf_j3(&cell, &prep, &aux_gc, &hc0.s, &gdf_cfg(None, None, P_TIGHT));
    let (_, jg1) = gdf_j3(
        &cell,
        &prep,
        &aux_gc,
        &hc0.s,
        &gdf_cfg(None, production(), P_TIGHT),
    );
    let (_, jgm) = gdf_j3(
        &cell,
        &prep,
        &aux_gc,
        &hc0.s,
        &gdf_cfg(None, rotation(ColumnRotationMutant::RotateAux), P_TIGHT),
    );
    let (dp, dm) = (rel(&jg1, &jg0), rel(&jgm, &jg0));
    eprintln!("aux = cc-pVDZ: production J3 rel {dp:.2e}; RotateAux J3 rel {dm:.2e}");
    assert!(dp <= TOL_COV, "production with a contracted aux: {dp:.3e}");
    assert!(
        dm >= MUTANT_FLOOR,
        "RotateAux survived the J3 anchor ({dm:.2e})"
    );
}

// ------------------------------------------------------------ refusals

#[test]
fn builds_that_must_walk_the_parent_shells_refuse_the_rotation() {
    let cell = cell_of(WATER, 8.0);
    let prep = bundled(&cell, "cc-pvdz");
    let aux = bundled(&cell, "cc-pvdz-ri");
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg(None, 1e-14)).expect("hcore");
    let cfg = gdf_cfg(None, production(), 1e-13);
    assert!(RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, &cfg).is_err());
    assert!(RsGdf::build_pair_s1_oracle(&cell, &prep, &aux, &hc.s, &cfg).is_err());
    assert!(periodic_hcore_pair_s1_oracle(&cell, &prep, &hcore_cfg(production(), 1e-14)).is_err());
    assert!(periodic_hcore(
        &cell,
        &prep,
        &hcore_cfg(rotation(ColumnRotationMutant::RotateAux), 1e-14)
    )
    .is_err());
}
