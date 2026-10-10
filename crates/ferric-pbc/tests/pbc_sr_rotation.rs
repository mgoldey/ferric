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
//! (g) Default resolution (`SrColumnRotation::Auto`): the Gamma energy and
//!     gradient builds rotate (bitwise explicit `On`); the s1 oracles run
//!     unrotated (bitwise explicit `Off`); explicit `On` on those paths is
//!     refused by name. The unrotated references of (a)-(f) are explicit
//!     `Off`. (The k-point energy builds rotate too: `pbc_sr_rotation_k.rs`.)
//! (h) FORCES AND STRESS (module `sr_rotation`, "Forces and stress"): the
//!     derivative walks run on the rotated shells with `Tᵀ W T` weights.
//!     * EXACTNESS ANCHOR: rotation on vs off equal to `D_BAR` = 1e-10,
//!       measured both as a WALK anchor (one density through both builds)
//!       and end to end (each build's own SCF), on H2 (unsplit, split) and
//!       off-symmetry H2O (split) for forces, H2 for stress, hcore and fit
//!       terms each anchored on their own. Measured correct values 5e-15 ..
//!       1e-12; the smallest MUTANT miss is 1.4e-1 (`D_BAR` sits between).
//!     * FD of the ROTATED energy (forces: all of H2's components; stress:
//!       all nine strains), bars `FD_F_BAR` / `FD_S_BAR` = measured O(h²)
//!       truncation (1e-9 / 4e-9, the unrotated control gives the same) ×10.
//!     * TRIVIAL LIMIT: a basis with nothing to rotate gives BITWISE the
//!       unrotated force and stress, also under every forward-transform
//!       mutant (they cannot act).
//!     * MUTANTS of the forward transform (`GradMutation::Rot*` /
//!       `StressMutation::Rot*`: no `T`, `T` where `Tᵀ` belongs, one AO
//!       index only, wrong sign of `T_ks`): each misses BOTH the hcore and
//!       the fit term by ≥ `D_MUTANT_FLOOR` and the finite difference.
//!     * Refusals: a rotated hcore with `Off` in `hcore_cfg`, an explicit
//!       `On` over an unrotated hcore, rotation mutants in `hcore_cfg` or
//!       the `RsGdf` (a deliberately wrong energy has no derivative).
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
use ferric_pbc::grad::{
    gamma_rhf_gradient_rsgdf, GammaGradConfig, GammaGradient, GradMutation, RsGdfGradSource,
};
use ferric_pbc::hcore::{
    periodic_hcore, periodic_hcore_pair_s1_oracle, PeriodicHcore, PeriodicHcoreConfig,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::{sr_walk_counts, RangeSplit, RsGdf, RsGdfConfig};
use ferric_pbc::sr_rotation::{ColumnRotation, ColumnRotationMutant, SrColumnRotation};
use ferric_pbc::stress::{
    gamma_rhf_stress_rsgdf, GammaStress, GammaStressConfig, Mat3, StressMutation,
};
use ferric_scf::result::ScfResult;
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
/// (h): rotated vs unrotated force (Ha/Bohr) and stress dE/dε (Ha). Measured
/// (release, 3 threads): force 4.9e-15 .. 7.4e-14, stress 7.6e-14 .. 1.0e-12;
/// the smallest forward-transform mutant miss is 1.4e-1 (force) / 3.7e-1
/// (stress), so the bar sits 3 decades above the worst correct value and 9
/// below the smallest mutant.
const D_BAR: f64 = 1e-10;
/// (h): smallest miss a forward-transform mutant must show, per term.
const D_MUTANT_FLOOR: f64 = 1e-3;
/// (h): analytic vs central FD (h = 1e-4) of the ROTATED energy. Measured
/// force 1.0e-9 (rotated) and 1.1e-9 (unrotated control: the O(h²) FD
/// truncation, it scales ×4..7 at h = 2e-4), stress 4.2e-9.
const FD_F_BAR: f64 = 1e-8;
const FD_S_BAR: f64 = 5e-8;

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

/// Runs `f` inside a fresh rayon pool of `n` threads and returns its result.
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

fn rotation(m: ColumnRotationMutant) -> SrColumnRotation {
    SrColumnRotation::On(ColumnRotation { mutant: m })
}

fn production() -> SrColumnRotation {
    SrColumnRotation::on()
}

/// The unrotated reference (explicit: the default `Auto` rotates the Gamma
/// energy builds).
const OFF: SrColumnRotation = SrColumnRotation::Off;

fn gdf_cfg(split: Option<RangeSplit>, rot: SrColumnRotation, precision: f64) -> RsGdfConfig {
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

fn hcore_cfg(rot: SrColumnRotation, precision: f64) -> PeriodicHcoreConfig {
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

/// The number of elements whose bit patterns differ; panics if the shapes
/// differ.
fn bit_diffs(a: &Array2<f64>, b: &Array2<f64>) -> usize {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// Asserts that `a` and `b` agree in every bit (see [`bit_diffs`]); `what`
/// labels a failure.
fn assert_bitwise(a: &Array2<f64>, b: &Array2<f64>, what: &str) {
    let d = bit_diffs(a, b);
    assert_eq!(d, 0, "{what}: {d} of {} elements differ in bits", a.len());
}

/// Largest element-wise `|m|` (0 for an empty matrix).
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
    let hc0 = periodic_hcore(&cell, &prep, &hcore_cfg(OFF, 1e-14)).expect("hcore");
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
        let (g0, j0) = gdf_j3(&cell, &prep, &aux, &hc0.s, &gdf_cfg(split, OFF, 1e-13));
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
    let hc0 = periodic_hcore(&cell, &prep, &hcore_cfg(OFF, P_TIGHT)).expect("hcore");
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
        let (g0, j0) = gdf_j3(&cell, &prep, &aux, &hc0.s, &gdf_cfg(split, OFF, P_TIGHT));
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
    let w0 = sr_walk_counts(&cell, &prep, &aux, &gdf_cfg(split, OFF, 1e-13)).unwrap();
    let w1 = sr_walk_counts(&cell, &prep, &aux, &gdf_cfg(split, production(), 1e-13)).unwrap();
    let u0 = sr_walk_counts(&cell, &prep, &aux, &gdf_cfg(None, OFF, 1e-13)).unwrap();
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
    let hc0 = periodic_hcore(&cell, &prep, &hcore_cfg(OFF, 1e-14)).expect("hcore");
    let hc1 = periodic_hcore(&cell, &prep, &hcore_cfg(production(), 1e-14)).expect("hcore");
    // C 1s, 2s, p; O 1s, 2s, p; one s per H.
    assert_eq!(hc1.timings.counter("hcore SR rotated columns"), Some(8));
    for split in [None, Some(RangeSplit::default())] {
        let tag = format!("H2CO/cc-pVDZ split {:?}", split.map(|s| s.lambda));
        let g0 =
            RsGdf::build(&cell, &prep, &aux, &hc0.s, &gdf_cfg(split, OFF, 1e-13)).expect("rsgdf");
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
    let hc0 = periodic_hcore(&cell, &prep, &hcore_cfg(OFF, P_TIGHT)).expect("hcore");
    let (_, j0) = gdf_j3(&cell, &prep, &aux, &hc0.s, &gdf_cfg(None, OFF, P_TIGHT));
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
    let (_, jg0) = gdf_j3(&cell, &prep, &aux_gc, &hc0.s, &gdf_cfg(None, OFF, P_TIGHT));
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
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg(OFF, 1e-14)).expect("hcore");
    let cfg = gdf_cfg(None, production(), 1e-13);
    // An explicit request is refused by name, never silently ignored.
    for (what, r) in [
        (
            "build_pair_s1_oracle",
            RsGdf::build_pair_s1_oracle(&cell, &prep, &aux, &hc.s, &cfg).map(|_| ()),
        ),
        (
            "periodic_hcore_pair_s1_oracle",
            periodic_hcore_pair_s1_oracle(&cell, &prep, &hcore_cfg(production(), 1e-14))
                .map(|_| ()),
        ),
    ] {
        let msg = r.expect_err(what).to_string();
        assert!(
            msg.contains("sr_column_rotation") && msg.contains("explicitly"),
            "{what}: {msg}"
        );
    }
    assert!(periodic_hcore(
        &cell,
        &prep,
        &hcore_cfg(rotation(ColumnRotationMutant::RotateAux), 1e-14)
    )
    .is_err());
    // The default `Auto` resolves OFF on the same builds: no refusal, and
    // bitwise the explicit-Off build (the oracles stay frozen).
    let auto = gdf_cfg(None, SrColumnRotation::Auto, 1e-13);
    let off = gdf_cfg(None, OFF, 1e-13);
    let (ga, go) = (
        RsGdf::build_pair_s1_oracle(&cell, &prep, &aux, &hc.s, &auto).expect("s1 oracle, Auto"),
        RsGdf::build_pair_s1_oracle(&cell, &prep, &aux, &hc.s, &off).expect("s1 oracle, Off"),
    );
    assert_bitwise(ga.b(), go.b(), "s1 oracle B: Auto vs Off");
    let (ha, ho) = (
        periodic_hcore_pair_s1_oracle(&cell, &prep, &hcore_cfg(SrColumnRotation::Auto, 1e-14))
            .expect("hcore s1 oracle, Auto"),
        periodic_hcore_pair_s1_oracle(&cell, &prep, &hcore_cfg(OFF, 1e-14))
            .expect("hcore s1 oracle, Off"),
    );
    assert_bitwise(&ha.h, &ho.h, "hcore s1 oracle h: Auto vs Off");
    assert_eq!(ha.sr_rotated_columns, 0);
}

// ------------------------------------------- default resolution (Auto)

/// The library default (`SrColumnRotation::Auto`) rotates the Gamma energy
/// builds: bitwise the explicit `On` build, counters set.
#[test]
fn default_gamma_energy_build_rotates() {
    let cell = cell_of(WATER, 8.0);
    let prep = bundled(&cell, "cc-pvdz");
    let aux = bundled(&cell, "cc-pvdz-ri");
    let hdef = PeriodicHcoreConfig::for_cell(&cell);
    assert_eq!(hdef.sr_column_rotation, SrColumnRotation::Auto);
    let hc_def = periodic_hcore(&cell, &prep, &hdef).expect("hcore default");
    let hc_on = periodic_hcore(&cell, &prep, &hdef.with_sr_column_rotation(production()))
        .expect("hcore on");
    let hc_off =
        periodic_hcore(&cell, &prep, &hdef.with_sr_column_rotation(OFF)).expect("hcore off");
    assert_eq!(hc_def.sr_rotated_columns, 5, "O 1s, 2s, p and one s per H");
    assert_eq!(hc_def.timings.counter("hcore SR rotated columns"), Some(5));
    assert_eq!(hc_off.sr_rotated_columns, 0);
    assert_eq!(hc_off.timings.counter("hcore SR rotated columns"), None);
    assert_bitwise(&hc_def.v_sr, &hc_on.v_sr, "default V_SR vs explicit On");
    assert_bitwise(&hc_def.h, &hc_on.h, "default h vs explicit On");
    assert!(
        bit_diffs(&hc_def.v_sr, &hc_off.v_sr) > 0,
        "default V_SR bitwise the unrotated one: rotation not applied"
    );
    let gdef = RsGdfConfig {
        budget_bytes: Some(1 << 30),
        ..Default::default()
    };
    assert_eq!(gdef.sr_column_rotation, SrColumnRotation::Auto);
    let gon = RsGdfConfig {
        sr_column_rotation: production(),
        ..gdef
    };
    let (g_def, j_def) = gdf_j3(&cell, &prep, &aux, &hc_def.s, &gdef);
    let (g_on, j_on) = gdf_j3(&cell, &prep, &aux, &hc_def.s, &gon);
    assert_eq!(
        g_def.timings().counter("rsgdf SR3 rotated columns"),
        Some(5)
    );
    assert_bitwise(&j_def, &j_on, "default J3 vs explicit On");
    assert_bitwise(g_def.b(), g_on.b(), "default B vs explicit On");
    let g_plain = RsGdf::build(&cell, &prep, &aux, &hc_def.s, &gdef).expect("rsgdf");
    assert_bitwise(g_plain.b(), g_def.b(), "build vs build_with_fit_parts B");
    assert_eq!(
        g_plain.timings().counter("rsgdf SR3 rotated columns"),
        Some(5)
    );
}

// ------------------------------------------------- forces and stress (h)

/// Off-symmetry water (every force and every stress component non-zero).
const WATER_TILT: &str = "3
water
O  0.0300  0.0200  0.1173
H  0.0400  0.7572 -0.4692
H  0.0900 -0.7100 -0.4300
";

/// Off-axis H2 (Bohr), a = 5: the cheapest cell with something to rotate
/// (one s column per H), nine non-zero stress components.
const H2_OFF_AXIS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

fn h2_off_axis(pos: &[[f64; 3]]) -> Cell {
    Cell::new(hydrogens(pos), cubic(5.0)).expect("cell")
}

fn water_tilt() -> Cell {
    cell_of(WATER_TILT, 8.0)
}

/// SCF + gradient-capable `RsGdf` + hcore of one cell at one rotation
/// request (`rot` goes to BOTH the hcore and the fit, like every driver).
struct Fx {
    cell: Cell,
    prep: PreparedBasis,
    aux: PreparedBasis,
    hc: PeriodicHcore,
    hcfg: PeriodicHcoreConfig,
    gdf: RsGdf,
    scf: ScfResult,
}

fn fixture(
    cell: Cell,
    prep: PreparedBasis,
    rot: SrColumnRotation,
    split: Option<RangeSplit>,
) -> Fx {
    let aux = bundled(&cell, "cc-pvdz-ri");
    let hcfg = hcore_cfg(rot, 1e-14);
    let hc = periodic_hcore(&cell, &prep, &hcfg).expect("hcore");
    let gdf = RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, &gdf_cfg(split, rot, 1e-13))
        .expect("build_for_gradient");
    let scf = gamma_rhf_jk(
        &cell,
        &prep,
        &hc,
        Box::new(gdf.j_builder()),
        Box::new(gdf.k_builder()),
    );
    Fx {
        cell,
        prep,
        aux,
        hc,
        hcfg,
        gdf,
        scf,
    }
}

fn cc_fixture(cell: Cell, rot: SrColumnRotation, split: Option<RangeSplit>) -> Fx {
    let prep = bundled(&cell, "cc-pvdz");
    fixture(cell, prep, rot, split)
}

fn gcfg(m: Option<GradMutation>) -> GammaGradConfig {
    GammaGradConfig {
        mutation: m,
        budget_bytes: Some(1 << 30),
        ..Default::default()
    }
}

fn scfg(m: Option<StressMutation>) -> GammaStressConfig {
    GammaStressConfig {
        mutation: m,
        budget_bytes: Some(1 << 30),
        ..Default::default()
    }
}

fn try_grad(
    fx: &Fx,
    scf: &ScfResult,
    hc: &PeriodicHcore,
    hcfg: &PeriodicHcoreConfig,
    m: Option<GradMutation>,
) -> Result<GammaGradient, ferric_core::FerricError> {
    let src = RsGdfGradSource {
        gdf: &fx.gdf,
        aux: &fx.aux,
        aux_jac: None,
    };
    gamma_rhf_gradient_rsgdf(
        &fx.cell,
        &fx.prep,
        hcfg,
        hc,
        &src,
        scf,
        ExxDiv::None,
        &gcfg(m),
    )
}

/// The force of `fx` at density `scf` (its own by default).
fn grad_at(fx: &Fx, scf: &ScfResult, m: Option<GradMutation>) -> GammaGradient {
    try_grad(fx, scf, &fx.hc, &fx.hcfg, m).expect("gradient")
}

fn stress_at(fx: &Fx, scf: &ScfResult, m: Option<StressMutation>) -> GammaStress {
    let src = RsGdfGradSource {
        gdf: &fx.gdf,
        aux: &fx.aux,
        aux_jac: None,
    };
    gamma_rhf_stress_rsgdf(
        &fx.cell,
        &fx.prep,
        &fx.hcfg,
        &fx.hc,
        &src,
        scf,
        ExxDiv::None,
        &scfg(m),
    )
    .expect("stress")
}

fn mat3_max_diff(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

fn mat3_bits_equal(a: &Mat3, b: &Mat3) -> bool {
    (0..3).all(|i| (0..3).all(|j| a[i][j].to_bits() == b[i][j].to_bits()))
}

const GRAD_MUTANTS: [GradMutation; 4] = [
    GradMutation::RotDropT,
    GradMutation::RotUseT,
    GradMutation::RotOneSided,
    GradMutation::RotFlipSign,
];

const STRESS_MUTANTS: [StressMutation; 4] = [
    StressMutation::RotDropT,
    StressMutation::RotUseT,
    StressMutation::RotOneSided,
    StressMutation::RotFlipSign,
];

/// The rotated fixtures really rotate (so the comparisons are not vacuous).
fn assert_rotates(fx: &Fx, tag: &str) {
    assert!(fx.hc.sr_rotated_columns > 0, "{tag}: hcore did not rotate");
    assert_eq!(
        fx.gdf.timings().counter("rsgdf SR3 rotated columns"),
        Some(fx.hc.sr_rotated_columns as u64),
        "{tag}: the gradient build did not rotate"
    );
}

/// The force fixtures: (tag, cell, split).
fn force_fixtures() -> Vec<(&'static str, Cell, Option<RangeSplit>)> {
    vec![
        ("H2 unsplit", h2_off_axis(&H2_OFF_AXIS), None),
        (
            "H2 split",
            h2_off_axis(&H2_OFF_AXIS),
            Some(RangeSplit::default()),
        ),
        ("H2O split", water_tilt(), Some(RangeSplit::default())),
    ]
}

/// EXACTNESS ANCHOR (forces): with the rotation requested, the analytic
/// force equals the unrotated force. Two measures: the WALK anchor (one
/// fixed density through both builds: only the derivative walks differ) and
/// the END-TO-END anchor (each build's own SCF). Mutants of the forward
/// transform must miss the walk anchor.
#[test]
fn rotated_forces_equal_the_unrotated_forces_and_mutants_miss() {
    for (tag, cell, split) in force_fixtures() {
        let on = cc_fixture(cell.clone(), production(), split);
        let off = cc_fixture(cell, OFF, split);
        assert_rotates(&on, tag);
        let g_off = grad_at(&off, &off.scf, None);
        // Walk anchor: the SAME density through the rotated and the plain walks.
        let g_walk = grad_at(&on, &off.scf, None);
        let walk = max_abs_diff(&g_walk.grad, &g_off.grad);
        // End-to-end: each build's own converged density.
        let g_own = grad_at(&on, &on.scf, None);
        let own = max_abs_diff(&g_own.grad, &g_off.grad);
        let scale = max_abs(&g_off.grad);
        eprintln!(
            "{tag}: max|F| {scale:.3e}; rot-vs-off walk {walk:.2e}, own-SCF {own:.2e}; \
             SR3 deriv calls {} vs {}",
            g_walk.fit.as_ref().map_or(0, |f| f.n_sr3_deriv),
            g_off.fit.as_ref().map_or(0, |f| f.n_sr3_deriv),
        );
        assert!(walk < D_BAR, "{tag}: walk anchor {walk:e}");
        assert!(own < D_BAR, "{tag}: own-SCF anchor {own:e}");
        // Each path is anchored on its own term, not only on the sum.
        let hcore_term = |g: &GammaGradient| &g.parts.vsr_basis + &g.parts.vsr_nuc;
        let hc_walk = max_abs_diff(&hcore_term(&g_walk), &hcore_term(&g_off));
        let fit_walk = max_abs_diff(&g_walk.parts.fit_orb_sr, &g_off.parts.fit_orb_sr);
        assert!(
            hc_walk < D_BAR && fit_walk < D_BAR,
            "{tag}: {hc_walk:e} {fit_walk:e}"
        );
        assert!(
            amax_part(&g_off.parts.fit_orb_sr) > 1e-3 && amax_part(&hcore_term(&g_off)) > 1e-3,
            "{tag}: vacuous SR terms"
        );
        // The rotated walks do fewer derivative calls (the payoff exists).
        let n3 = |g: &GammaGradient| g.fit.as_ref().expect("fit").n_sr3_deriv;
        assert!(n3(&g_walk) < n3(&g_off), "{tag}: SR3 calls");
        assert!(
            g_walk.n_sr_triplets < g_off.n_sr_triplets,
            "{tag}: hcore calls"
        );
        // Mutants: every one misses BOTH the hcore and the fit term.
        for m in GRAD_MUTANTS {
            let gm = grad_at(&on, &off.scf, Some(m));
            let tot = max_abs_diff(&gm.grad, &g_off.grad);
            let hc_miss = max_abs_diff(&hcore_term(&gm), &hcore_term(&g_off));
            let fit_miss = max_abs_diff(&gm.parts.fit_orb_sr, &g_off.parts.fit_orb_sr);
            assert!(
                tot > D_MUTANT_FLOOR && hc_miss > D_MUTANT_FLOOR && fit_miss > D_MUTANT_FLOOR,
                "{tag}: mutant {m:?} escaped: total {tot:e}, hcore {hc_miss:e}, fit {fit_miss:e}"
            );
        }
    }
}

/// Largest element-wise `|m|` of a part.
fn amax_part(m: &Array2<f64>) -> f64 {
    max_abs(m)
}

/// The RHF energy of `cell` through the rotated (or not) pipeline at `rot`.
fn pipeline_energy(cell: &Cell, bs: &str, rot: SrColumnRotation, split: Option<RangeSplit>) -> f64 {
    let prep = bundled(cell, bs);
    let aux = bundled(cell, "cc-pvdz-ri");
    let hc = periodic_hcore(cell, &prep, &hcore_cfg(rot, 1e-14)).expect("hcore");
    let g = RsGdf::build(cell, &prep, &aux, &hc.s, &gdf_cfg(split, rot, 1e-13)).expect("rsgdf");
    energy(cell, &prep, &hc, &g)
}

/// ANCHOR (finite differences): the analytic force of the ROTATED energy
/// equals the central FD of the rotated energy.
#[test]
fn rotated_force_matches_fd_of_the_rotated_energy() {
    let h = 1e-4;
    for split in [None, Some(RangeSplit::default())] {
        let cell = h2_off_axis(&H2_OFF_AXIS);
        let on = cc_fixture(cell, production(), split);
        assert_rotates(&on, "H2");
        let g = grad_at(&on, &on.scf, None);
        let mut worst = 0.0_f64;
        let mut worst_mut = [0.0_f64; 4];
        let gm: Vec<_> = GRAD_MUTANTS
            .iter()
            .map(|&m| grad_at(&on, &on.scf, Some(m)))
            .collect();
        for (a, x) in [(0usize, 0usize), (0, 2), (1, 1)] {
            let mut p = H2_OFF_AXIS;
            p[a][x] += h;
            let ep = pipeline_energy(&h2_off_axis(&p), "cc-pvdz", production(), split);
            p[a][x] -= 2.0 * h;
            let em = pipeline_energy(&h2_off_axis(&p), "cc-pvdz", production(), split);
            let fd = (ep - em) / (2.0 * h);
            worst = worst.max((g.grad[(a, x)] - fd).abs());
            for (k, gk) in gm.iter().enumerate() {
                worst_mut[k] = worst_mut[k].max((gk.grad[(a, x)] - fd).abs());
            }
        }
        assert!(
            worst < FD_F_BAR,
            "split={split:?}: |analytic - FD| {worst:e}"
        );
        for (m, w) in GRAD_MUTANTS.iter().zip(worst_mut) {
            assert!(
                w > D_MUTANT_FLOOR,
                "split={split:?}: mutant {m:?} escaped FD: {w:e}"
            );
        }
    }
}

/// EXACTNESS ANCHOR (stress), same structure as the forces'.
#[test]
fn rotated_stress_equals_the_unrotated_stress_and_matches_fd() {
    let h = 1e-4;
    for split in [None, Some(RangeSplit::default())] {
        let cell = h2_off_axis(&H2_OFF_AXIS);
        let on = cc_fixture(cell.clone(), production(), split);
        let off = cc_fixture(cell.clone(), OFF, split);
        assert_rotates(&on, "H2");
        let s_off = stress_at(&off, &off.scf, None);
        let s_walk = stress_at(&on, &off.scf, None);
        let s_own = stress_at(&on, &on.scf, None);
        let walk = mat3_max_diff(&s_walk.de_deps, &s_off.de_deps);
        let own = mat3_max_diff(&s_own.de_deps, &s_off.de_deps);
        // FD of the rotated energy under all nine strains.
        let mut fd = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut e = [[0.0; 3]; 3];
                e[i][j] = h;
                let ep =
                    pipeline_energy(&cell.strained(&e).unwrap(), "cc-pvdz", production(), split);
                e[i][j] = -h;
                let em =
                    pipeline_energy(&cell.strained(&e).unwrap(), "cc-pvdz", production(), split);
                fd[i][j] = (ep - em) / (2.0 * h);
            }
        }
        let fd_err = mat3_max_diff(&s_own.de_deps, &fd);
        assert!(walk < D_BAR, "split={split:?}: stress walk anchor {walk:e}");
        assert!(
            own < D_BAR,
            "split={split:?}: stress own-SCF anchor {own:e}"
        );
        assert!(
            fd_err < FD_S_BAR,
            "split={split:?}: stress analytic-vs-FD {fd_err:e}"
        );
        let vsr = |s: &GammaStress| s.parts.vsr;
        let zero = [[0.0; 3]; 3];
        assert!(mat3_max_diff(&vsr(&s_walk), &vsr(&s_off)) < D_BAR);
        assert!(mat3_max_diff(&s_walk.parts.fit_j3_sr, &s_off.parts.fit_j3_sr) < D_BAR);
        assert!(
            mat3_max_diff(&vsr(&s_off), &zero) > 1e-3
                && mat3_max_diff(&s_off.parts.fit_j3_sr, &zero) > 1e-3,
            "vacuous SR strain terms"
        );
        for m in STRESS_MUTANTS {
            let sm = stress_at(&on, &off.scf, Some(m));
            let tot = mat3_max_diff(&sm.de_deps, &s_off.de_deps);
            let hc_miss = mat3_max_diff(&vsr(&sm), &vsr(&s_off));
            let fit_miss = mat3_max_diff(&sm.parts.fit_j3_sr, &s_off.parts.fit_j3_sr);
            assert!(
                tot > D_MUTANT_FLOOR && hc_miss > D_MUTANT_FLOOR && fit_miss > D_MUTANT_FLOOR,
                "split={split:?}: mutant {m:?} escaped: total {tot:e}, hcore {hc_miss:e}, \
                 fit {fit_miss:e}"
            );
        }
    }
}

/// TRIVIAL LIMIT: a basis with nothing to rotate. Force and stress with the
/// rotation requested (and every forward-transform mutant, which cannot act)
/// are BITWISE the unrotated ones.
///
/// The unmutated check runs on the triclinic cell (non-orthogonal images);
/// the mutants cannot act on a basis with nothing to rotate, so they are
/// checked on the cheap cubic H2 cell (the full triclinic mutant sweep took
/// ~4.5 min and timed out the CI shard).
#[test]
fn identity_rotation_forces_and_stress_are_bitwise_unrotated() {
    for (cell, mutants, splits) in [
        (triclinic_cell(), false, vec![None]),
        (
            h2_off_axis(&H2_OFF_AXIS),
            true,
            vec![None, Some(RangeSplit::default())],
        ),
    ] {
        for split in splits {
            let on = fixture(
                cell.clone(),
                prep_for(&cell, &sp_basis_h()),
                production(),
                split,
            );
            let off = fixture(cell.clone(), prep_for(&cell, &sp_basis_h()), OFF, split);
            assert_eq!(on.hc.sr_rotated_columns, 0);
            assert_eq!(
                on.gdf.timings().counter("rsgdf SR3 rotated columns"),
                Some(0)
            );
            let g_off = grad_at(&off, &off.scf, None);
            let s_off = stress_at(&off, &off.scf, None);
            let gm: Vec<Option<GradMutation>> = if mutants {
                std::iter::once(None)
                    .chain(GRAD_MUTANTS.iter().map(|&m| Some(m)))
                    .collect()
            } else {
                vec![None]
            };
            for m in gm {
                let g = grad_at(&on, &on.scf, m);
                assert_bitwise(
                    &g.grad,
                    &g_off.grad,
                    &format!("force {m:?} split={split:?}"),
                );
            }
            let sm: Vec<Option<StressMutation>> = if mutants {
                std::iter::once(None)
                    .chain(STRESS_MUTANTS.iter().map(|&m| Some(m)))
                    .collect()
            } else {
                vec![None]
            };
            for m in sm {
                let s = stress_at(&on, &on.scf, m);
                assert!(
                    mat3_bits_equal(&s.de_deps, &s_off.de_deps),
                    "stress {m:?} split={split:?}"
                );
            }
        }
    }
}

/// `Auto` (the library default) rotates the force path: hcore, fit and the
/// derivative walks, bitwise the explicit-`On` run, and the force really
/// used fewer SR derivative calls than the unrotated walk.
#[test]
fn default_force_path_rotates() {
    let cell = h2_off_axis(&H2_OFF_AXIS);
    let auto = cc_fixture(cell.clone(), SrColumnRotation::Auto, None);
    let on = cc_fixture(cell.clone(), production(), None);
    let off = cc_fixture(cell, OFF, None);
    assert_eq!(auto.hc.sr_rotated_columns, 2, "one s column per H");
    assert_eq!(
        auto.gdf.timings().counter("rsgdf SR3 rotated columns"),
        Some(2)
    );
    let (ga, go, gf) = (
        grad_at(&auto, &auto.scf, None),
        grad_at(&on, &on.scf, None),
        grad_at(&off, &off.scf, None),
    );
    assert_bitwise(&ga.grad, &go.grad, "Auto vs On force");
    assert!(
        bit_diffs(&ga.grad, &gf.grad) > 0,
        "rotated force bitwise the unrotated one: the derivative walk did not rotate"
    );
    let n = |g: &GammaGradient| g.fit.as_ref().expect("fit").n_sr3_deriv;
    assert!(n(&ga) < n(&gf), "SR3 deriv calls {} vs {}", n(&ga), n(&gf));
    assert!(ga.n_sr_triplets < gf.n_sr_triplets, "hcore SR deriv calls");
}

/// Typed refusals of the force path: configurations that would differentiate
/// another walk than the energy ran.
#[test]
fn force_path_refuses_inconsistent_rotation_requests() {
    let cell = h2_off_axis(&H2_OFF_AXIS);
    let on = cc_fixture(cell.clone(), production(), None);
    let off = cc_fixture(cell.clone(), OFF, None);
    // Control: both consistent pairs run.
    grad_at(&on, &on.scf, None);
    grad_at(&off, &off.scf, None);
    // A rotated hcore with a config asking for no rotation.
    let msg = try_grad(&on, &on.scf, &on.hc, &hcore_cfg(OFF, 1e-14), None)
        .expect_err("rotated hc, cfg Off")
        .to_string();
    assert!(msg.contains("hcore_cfg.sr_column_rotation is Off"), "{msg}");
    // An explicit On over an unrotated hcore of a basis that rotates.
    let msg = try_grad(
        &off,
        &off.scf,
        &off.hc,
        &hcore_cfg(production(), 1e-14),
        None,
    )
    .expect_err("unrotated hc, cfg On")
    .to_string();
    assert!(msg.contains("built unrotated"), "{msg}");
    // A rotation mutant's energy has no derivative.
    let mutant = hcore_cfg(rotation(ColumnRotationMutant::FlipSign), 1e-14);
    let msg = try_grad(&on, &on.scf, &on.hc, &mutant, None)
        .expect_err("mutant cfg")
        .to_string();
    assert!(msg.contains("mutant"), "{msg}");
    // An RsGdf built with a rotation mutant, differentiated.
    let aux = bundled(&cell, "cc-pvdz-ri");
    let prep = bundled(&cell, "cc-pvdz");
    let g_mut = RsGdf::build_for_gradient(
        &cell,
        &prep,
        &aux,
        &on.hc.s,
        &gdf_cfg(None, rotation(ColumnRotationMutant::FlipSign), 1e-13),
    )
    .expect("mutant build");
    let fx = Fx {
        cell: cell.clone(),
        prep: bundled(&cell, "cc-pvdz"),
        aux,
        hc: on.hc.clone(),
        hcfg: on.hcfg,
        gdf: g_mut,
        scf: on.scf.clone(),
    };
    let msg = try_grad(&fx, &fx.scf, &fx.hc, &fx.hcfg, None)
        .expect_err("RsGdf rotation mutant")
        .to_string();
    assert!(msg.contains("column-rotation mutant"), "{msg}");
}

/// The rotated force and stress are BITWISE identical across thread counts
/// (the ordered-parallel derivative walks and the per-element `Tᵀ Y T`
/// read are deterministic), and the frozen serial hcore oracle
/// (`SerialDerivWalks`) agrees bitwise with the ordered walk on the rotated
/// shells.
#[test]
fn rotated_forces_and_stress_are_bitwise_across_thread_counts() {
    for split in [None, Some(RangeSplit::default())] {
        let cell = h2_off_axis(&H2_OFF_AXIS);
        let fx = cc_fixture(cell, production(), split);
        assert_rotates(&fx, "H2");
        let ref_g = in_pool(1, || grad_at(&fx, &fx.scf, None));
        let ref_s = in_pool(1, || stress_at(&fx, &fx.scf, None));
        for n in [2, 6] {
            let g = in_pool(n, || grad_at(&fx, &fx.scf, None));
            let s = in_pool(n, || stress_at(&fx, &fx.scf, None));
            assert_bitwise(&g.grad, &ref_g.grad, &format!("force, {n} threads"));
            assert!(
                mat3_bits_equal(&s.de_deps, &ref_s.de_deps),
                "stress, {n} threads, split={split:?}"
            );
        }
        let serial = grad_at(&fx, &fx.scf, Some(GradMutation::SerialDerivWalks));
        assert_bitwise(&serial.grad, &ref_g.grad, "SerialDerivWalks vs ordered");
        let serial = stress_at(&fx, &fx.scf, Some(StressMutation::SerialDerivWalks));
        assert!(
            mat3_bits_equal(&serial.de_deps, &ref_s.de_deps),
            "stress SerialDerivWalks vs ordered, split={split:?}"
        );
    }
}
