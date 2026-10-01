//! k-point orbital-pair symmetry (s2) of the short-range integral stages
//! (`rsgdf::kpoint` module doc "Orbital-pair symmetry at k",
//! `hcore::kpoint` module doc "Orbital-pair symmetry"). The Gamma twin is
//! `pbc_pair_symmetry.rs`.
//!
//! # The relations under test (derived, not measured)
//!
//! * RS-GDF SR residue bins. `B[r_L, r_T][μν, P] = Σ_{L ≡ r_L} Σ_{T ≡ r_T}
//!   (μ_0 ν_L | P_T)`. With `(μ_0 ν_L | P_T) = (ν_0 μ_{−L} | P_{T−L})`:
//!   `B[b][μν] = B[M(b)][νμ]`, `M(r_L, r_T) = (−r_L mod mod_l,
//!   (r_T − r_L) mod mod_t)` (an involution; `mod_t | mod_l`). The s2 walk
//!   computes each unordered shell pair once per residue and writes the
//!   transposed half into bin `M(b)`. Diagonal pairs: canonical residues
//!   `r_L ≤ r_{−L}` only; at a self-conjugate residue the lexicographically
//!   smaller element of each `M`-orbit is written, directly and mirrored.
//!   `common::kpair_mirror` / `kpair_s2_expected` re-derive both rules here,
//!   independently of the library, and the library's own map
//!   (`kpair_mirror_bin`) is checked against them.
//! * hcore `S(k)`, `T(k)`, `V_SR(k)`: `X(k)[νμ] = conj X(k)[μν]` (the phase
//!   pairing `e^{±ik·L}` IS the conjugation; no bins).
//!
//! # What each stage must satisfy against the FROZEN ordered oracles
//!
//! (`sr3_kbins_s2_and_s1`'s ordered branch = `KRsGdf::build_pair_s1_oracle`'s
//! walk; `*_kpts_parallel_and_serial`'s serial loops =
//! `periodic_hcore_kpts_pair_s1_oracle`):
//!
//! 1. EXACT `M`-symmetry / off-diagonal Hermiticity, bitwise.
//! 2. Every element bitwise one of the oracle's two ordered evaluations
//!    `x = s1[e]`, `y = s1[M(e)]` (hcore: `y = conj s1[(ν, μ)]`); unsplit and
//!    hcore: EXACTLY the prescribed one.
//! 3. Bitwise across 1/2/6 threads.
//! 4. `|new − avg| ≤ ½|x − y| + ε|avg|`, `avg = fl(½ fl(x + y))`
//!    (derivation of `pbc_pair_symmetry.rs` item 4: `new ∈ {x, y}` and
//!    `avg = ½(x + y)(1 + δ)`, `|δ| ≤ ε/2` per real component, so
//!    `|new − avg| ≤ ½|x − y| + ½ε|x + y|`; the complex norm obeys the same
//!    bound). Scale: the ordered walk's own round-off asymmetry (printed).
//! 5. k-mesh RHF / UHF / range-split RHF energies within `E_BAR` = 1e-11 Ha of
//!    the frozen pre-s2 build (the task bar; the expected shift is
//!    first-order in a ~1e-16 relative change of J3 / h, i.e. ~1e-15 Ha),
//!    on a TRIM mesh (1×1×2) and a non-TRIM one (1×1×3).
//! 6. Counters: the ordered-equivalent count equals the ordered oracle's
//!    count (unsplit and hcore).
//! 7. The RELATION itself: the ordered walk's `x` and `y` agree to
//!    `MAP_BAR` (items 1, 2 and 4 hold for ANY map the library and this file
//!    share, right or wrong — `new ∈ {x, y}` makes item 4 automatic; only
//!    `x ≈ y` says the map pairs equal lattice sums).
//!
//! # Artifact hypothesis (stated before measuring)
//!
//! A correct s2 walk: 0 symmetry violations, 0 elements outside `{x, y}`,
//! `max|new − avg|` ~1e-17 absolute. A broken one is O(|J3|) ~ 1e-2..1e-1
//! off somewhere: a dropped transposed write leaves rows `νμ` zero; a
//! skipped diagonal zeroes the diagonal blocks; a missing aux shift or a
//! wrong pair-image bin puts the right numbers in the wrong bin. Items 1, 2
//! and 4 fail by ~15 orders of magnitude, so the mutants below are asserted
//! against item 4 with a margin, not near the bound.
//!
//! # Mutants (in-library switches, `KPairSymMutant` / `KHcorePairMutant`)
//!
//! * `WrongPairBin` (transposed half into bin `r_L`, not `r_{−L}`) and hcore
//!   `NoConj` (transposed element unconjugated): IDENTITIES on a mesh of TRIM
//!   points only. At Gamma-centred 1×1×2, `mod_l = 2`, so `−r ≡ r` for every
//!   residue (`M` moves only `r_T`), and every phase is exactly ±1, so every
//!   `S(k)`, `V(k)` element is real (conj flips only the sign of a zero).
//!   Asserted blind there (bitwise bins, equal values, equal energy) and
//!   caught on the non-TRIM 1×1×3 (`mod_l = 3`, `−1 ≡ 2`, phases
//!   `e^{±2πi/3}`), at the bins, the matrices and the KRHF energy.
//! * `NoAuxShift`, `DropTransposed`, `SkipDiagonal`, `DiagonalNoMirror`
//!   (bins) and `DropTransposed`, `SkipDiagonal` (hcore): caught on BOTH
//!   meshes (`NoAuxShift` too: at 1×1×2 `r_L = 1` shifts `r_T` by 1 mod 2).
//!
//! Not covered: meshes beyond 1×1×3 / MP 1×2×2, d shells, the k force /
//! stress walks (their own ordered loops, unchanged).

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::hcore::kpoint::{
    hcore_kpts_s2_mutant, overlap_kinetic_kpts_parallel_and_serial, periodic_hcore_kpts,
    periodic_hcore_kpts_pair_s1_oracle, sr_attraction_kpts_parallel_and_serial, HcoreKS2Parts,
    KHcorePairMutant, PeriodicHcoreK,
};
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KPointJk, KScfConfig};
use ferric_pbc::kuscf::solve_kuhf_injected;
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{
    kpair_mirror_bin, sr3_kbins_s2_and_s1, KPairSymMutant, KRsGdf, KRsGdfConfig, KRsGdfMutation,
    KSr3Parts,
};
use ferric_pbc::rsgdf::{RangeSplit, RsGdfConfig};
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

const THREADS: [usize; 3] = [1, 2, 6];
/// Nuclear-attraction ω and precision (as `pbc_krsgdf_split.rs`).
const HCORE_OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
/// RS-GDF ω (λ = 1 moves orbital a ≤ 0.5, aux α ≤ 1).
const GDF_OMEGA: f64 = 1.0;
const AMPLE: usize = 1 << 31;
/// Item 5 of the module doc.
const E_BAR: f64 = 1e-11;
/// A caught mutant is off by O(|J3|) ~ 1e-2 (artifact hypothesis); this is
/// 5+ decades below that and 7+ above the round-off scale of item 4.
const MUTANT_ELEMENT_BAR: f64 = 1e-8;
/// Item 7: max ½|x − y| of the ordered walk's two evaluations under the map.
/// They differ by summation order (~1e-16 relative) and by a triplet the
/// screen keeps in one orientation only, exactly at its radius, which
/// contributes at most ~the screen precision (1e-13 RS-GDF, 1e-14 hcore,
/// absolute). A WRONG map pairs different lattice sums: O(|J3|) ~ 1e-2. The
/// bar sits 3 decades above the first and 8 below the second.
const MAP_BAR: f64 = 1e-10;
/// Item 7 with a range split: a compact × split pair's two orientations run
/// DIFFERENT kept calls (1 vs 2) with their own piece screens, so `x` and `y`
/// agree only to the accumulated screen tolerance (many sub-threshold
/// triplets), not to round-off. Still 6 decades below a wrong map.
const MAP_BAR_SPLIT: f64 = 1e-8;
/// A caught energy mutant: 3 decades above `E_BAR`.
const MUTANT_E_BAR: f64 = 1e-8;

const TRIM: [usize; 3] = [1, 1, 2];
const NON_TRIM: [usize; 3] = [1, 1, 3];

const KPAIR_MUTANTS: [KPairSymMutant; 5] = [
    KPairSymMutant::WrongPairBin,
    KPairSymMutant::NoAuxShift,
    KPairSymMutant::DropTransposed,
    KPairSymMutant::SkipDiagonal,
    KPairSymMutant::DiagonalNoMirror,
];
const HCORE_MUTANTS: [KHcorePairMutant; 3] = [
    KHcorePairMutant::NoConj,
    KHcorePairMutant::DropTransposed,
    KHcorePairMutant::SkipDiagonal,
];

/// H3 (Bohr), a = 4.5, STO-3G, doublet (2, 1) per cell (`pbc_krsgdf_split.rs`).
const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
const H3_A: f64 = 4.5;

fn in_pool<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("rayon pool")
        .install(f)
}

fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(HCORE_OMEGA)
    }
}

fn kscf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    }
}

fn gdf_cfg(split: Option<RangeSplit>) -> RsGdfConfig {
    RsGdfConfig {
        omega: GDF_OMEGA,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        range_split: split,
        ..Default::default()
    }
}

fn kgdf_cfg(split: Option<RangeSplit>, mutation: Option<KRsGdfMutation>) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: gdf_cfg(split),
        mutation,
    }
}

/// The prototype's ET-sp aux on H (`pbc_krsgdf_split.rs`): s(4.7, 1.9, 0.75,
/// 0.3) + p(1.25, 0.5), one unit-normalised primitive per shell. Small, so
/// the bins tests stay cheap; λ = 1 moves s 0.75, 0.3 and p 0.5.
fn et_sp_aux() -> BasisSet {
    let mut shells = Vec::new();
    for (l, a) in [(0, 4.7), (0, 1.9), (0, 0.75), (0, 0.3), (1, 1.25), (1, 0.5)] {
        shells.push(Shell {
            l,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "et-sp-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

/// Triclinic 4 H, STO-3G s + one Cartesian p: 8 shells with 1×1, 1×3, 3×3
/// blocks (within-shell `μ ≠ ν` pairs exist, so the diagonal orbit rule is
/// live). At ω = 1, λ = 1 the s shell is split and the p shell compact, so
/// the range split has compact × split pairs in both index orders.
fn tri_sp() -> (Cell, PreparedBasis, PreparedBasis) {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    let aux = PreparedBasis::new(cell.mol(), &et_sp_aux()).expect("aux");
    (cell, prep, aux)
}

fn cell_at(pos: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

fn bits_eq(x: f64, y: f64) -> bool {
    x.to_bits() == y.to_bits()
}

fn cbits_eq(x: Complex64, y: Complex64) -> bool {
    bits_eq(x.re, y.re) && bits_eq(x.im, y.im)
}

fn bins_bit_diffs(a: &[Array2<f64>], b: &[Array2<f64>]) -> usize {
    assert_eq!(a.len(), b.len(), "bin counts");
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            assert_eq!(x.dim(), y.dim());
            x.iter()
                .zip(y.iter())
                .filter(|(p, q)| !bits_eq(**p, **q))
                .count()
        })
        .sum()
}

fn cmats_bit_diffs(a: &[Array2<Complex64>], b: &[Array2<Complex64>]) -> usize {
    assert_eq!(a.len(), b.len(), "matrix counts");
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            assert_eq!(x.dim(), y.dim());
            x.iter()
                .zip(y.iter())
                .filter(|(p, q)| !cbits_eq(**p, **q))
                .count()
        })
        .sum()
}

/// What item 4 measures for one comparison.
#[derive(Debug, Default)]
struct VsAvg {
    /// max |new − avg|.
    dmax: f64,
    /// max ½|x − y| (the ordered walk's own asymmetry).
    hmax: f64,
    /// Elements over `½|x − y| + ε|avg|`.
    over: usize,
    /// Elements bitwise neither `x` nor `y`.
    neither: usize,
    /// Elements whose `x` and `y` differ in bits (the "which one" checks are
    /// vacuous when this is 0).
    observable: usize,
}

impl VsAvg {
    fn add(&mut self, new: f64, x: f64, y: f64) {
        let avg = 0.5 * (x + y);
        let d = (new - avg).abs();
        let half = 0.5 * (x - y).abs();
        self.dmax = self.dmax.max(d);
        self.hmax = self.hmax.max(half);
        if !(d <= half + f64::EPSILON * avg.abs()) {
            self.over += 1;
        }
        if !bits_eq(new, x) && !bits_eq(new, y) {
            self.neither += 1;
        }
        if !bits_eq(x, y) {
            self.observable += 1;
        }
    }

    fn cadd(&mut self, new: Complex64, x: Complex64, y: Complex64) {
        let avg = 0.5 * (x + y);
        let d = (new - avg).norm();
        let half = 0.5 * (x - y).norm();
        self.dmax = self.dmax.max(d);
        self.hmax = self.hmax.max(half);
        if !(d <= half + f64::EPSILON * avg.norm()) {
            self.over += 1;
        }
        if !cbits_eq(new, x) && !cbits_eq(new, y) {
            self.neither += 1;
        }
        if !cbits_eq(x, y) {
            self.observable += 1;
        }
    }
}

/// `(M-symmetry violations, item 4 against the ordered bins)` of s2 bins
/// `s2` vs the ordered `s1`, `x = s1[b][μν]`, `y = s1[M(b)][νμ]`.
fn bins_vs_ordered(
    s2: &[Array2<f64>],
    s1: &[Array2<f64>],
    n: usize,
    mod_l: [usize; 3],
    mod_t: [usize; 3],
) -> (usize, VsAvg) {
    let mut asym = 0usize;
    let mut v = VsAvg::default();
    for b in 0..s2.len() {
        let mb = kpair_mirror(mod_l, mod_t, b);
        for mu in 0..n {
            for nu in 0..n {
                let (r, rt) = (mu * n + nu, nu * n + mu);
                for p in 0..s2[b].ncols() {
                    let new = s2[b][(r, p)];
                    if !bits_eq(new, s2[mb][(rt, p)]) {
                        asym += 1;
                    }
                    v.add(new, s1[b][(r, p)], s1[mb][(rt, p)]);
                }
            }
        }
    }
    (asym, v)
}

fn live_bins(bins: &[Array2<f64>]) -> usize {
    bins.iter().filter(|m| m.iter().any(|x| *x != 0.0)).count()
}

fn kbins_runs(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: &PreparedBasis,
    mesh: &KPointMesh,
    cfg: &RsGdfConfig,
) -> Vec<[KSr3Parts; 2]> {
    THREADS
        .iter()
        .map(|&t| {
            in_pool(t, || {
                sr3_kbins_s2_and_s1(cell, prep, aux, mesh, cfg, None).expect("k SR3 bins")
            })
        })
        .collect()
}

/// Items 1-4 and 6 for one mesh; an unsplit build (`split = None`) adds the
/// exact prescription and the ordered-equivalent count.
fn check_kbins(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: &PreparedBasis,
    mesh: &KPointMesh,
    split: Option<RangeSplit>,
    tag: &str,
) {
    let n = prep.nbasis();
    let (mod_l, mod_t) = (mesh.residue_moduli(), mesh.n());
    let runs = kbins_runs(cell, prep, aux, mesh, &gdf_cfg(split));
    let [(s2, c2, o2), (s1, c1, o1)] = &runs[0];
    assert!(
        live_bins(s2) >= 2,
        "{tag}: only {} live bins, the residue index is vacuous",
        live_bins(s2)
    );
    // The library's bin map is the one derived here.
    for b in 0..s2.len() {
        assert_eq!(
            kpair_mirror_bin(mod_l, mod_t, b).unwrap(),
            kpair_mirror(mod_l, mod_t, b),
            "{tag}: M({b})"
        );
    }
    // 3. Across threads.
    for (&t, [(a2, d2, p2), (a1, d1, p1)]) in THREADS.iter().zip(&runs) {
        assert_eq!(
            bins_bit_diffs(a2, s2),
            0,
            "{tag}: s2 bins at {t} vs 1 thread"
        );
        assert_eq!(
            bins_bit_diffs(a1, s1),
            0,
            "{tag}: s1 bins at {t} vs 1 thread"
        );
        assert_eq!((d2, p2, d1, p1), (c2, o2, c1, o1), "{tag}: counts at {t}");
    }
    // 1, 2 and 4.
    let (asym, v) = bins_vs_ordered(s2, s1, n, mod_l, mod_t);
    eprintln!(
        "{tag} mod_l {mod_l:?} mod_t {mod_t:?}: {asym} M-asymmetric; {v:?}; triplets s2 {c2} \
         (ordered-equivalent {o2}) vs ordered {c1}"
    );
    assert_eq!(asym, 0, "{tag}: s2 bins not exactly M-symmetric");
    let map_bar = if split.is_some() {
        MAP_BAR_SPLIT
    } else {
        MAP_BAR
    };
    assert!(
        v.hmax <= map_bar,
        "{tag}: the ordered walk's two evaluations differ by {:.2e} under M",
        v.hmax
    );
    assert_eq!(v.neither, 0, "{tag}: s2 element outside {{x, y}}");
    assert_eq!(
        v.over, 0,
        "{tag}: s2 outside ½|x − y| + ε|avg| of the average"
    );
    assert!(
        v.observable > 0,
        "{tag}: x and y agree in every bit, so the 'which one' checks are vacuous \
         (redesign the fixture)"
    );
    assert_eq!(o1, c1, "{tag}: the ordered oracle's two counts");
    assert!(c2 < c1, "{tag}: s2 computed {c2} of the ordered {c1}");
    if split.is_none() {
        // 2 (exact prescription) and 6.
        let x = kpair_s2_expected(s1, n, &shell_of(prep), mod_l, mod_t);
        assert_eq!(
            bins_bit_diffs(s2, &x),
            0,
            "{tag}: s2 bins vs the prescribed ordered elements"
        );
        assert_eq!(o2, c1, "{tag}: ordered-equivalent vs the ordered count");
    }
}

#[test]
fn krsgdf_s2_bins_are_the_ordered_walk_under_the_minus_rl_map() {
    let (cell, prep, aux) = tri_sp();
    for (tag, mesh) in [
        ("Gamma 1x1x2 (TRIM)", KPointMesh::gamma_centred(&cell, TRIM)),
        ("Gamma 1x1x3", KPointMesh::gamma_centred(&cell, NON_TRIM)),
        // Pair-image moduli 2N = [1, 4, 4] differ from the aux-image N.
        ("MP 1x2x2", KPointMesh::monkhorst_pack(&cell, [1, 2, 2])),
    ] {
        let mesh = mesh.unwrap();
        check_kbins(&cell, &prep, &aux, &mesh, None, tag);
    }
}

#[test]
fn krsgdf_s2_split_bins_take_either_orientation() {
    let (cell, prep, aux) = tri_sp();
    let mesh = KPointMesh::gamma_centred(&cell, NON_TRIM).unwrap();
    check_kbins(
        &cell,
        &prep,
        &aux,
        &mesh,
        Some(RangeSplit::default()),
        "split Gamma 1x1x3",
    );
}

/// Every bins mutant leaves `{x, y}` by far more than round-off on the
/// non-TRIM mesh; on the TRIM mesh `WrongPairBin` is bitwise the production
/// walk and every other mutant is still caught.
#[test]
fn krsgdf_s2_bin_mutants_fail_and_wrong_pair_bin_needs_a_non_trim_mesh() {
    let (cell, prep, aux) = tri_sp();
    let n = prep.nbasis();
    let cfg = gdf_cfg(None);
    for (dims, trim) in [(NON_TRIM, false), (TRIM, true)] {
        let mesh = KPointMesh::gamma_centred(&cell, dims).unwrap();
        let (mod_l, mod_t) = (mesh.residue_moduli(), mesh.n());
        let [(prod, _, _), (s1, _, _)] =
            sr3_kbins_s2_and_s1(&cell, &prep, &aux, &mesh, &cfg, None).expect("production");
        let scale = prod
            .iter()
            .fold(0.0_f64, |a, m| m.iter().fold(a, |b, x| b.max(x.abs())));
        for m in KPAIR_MUTANTS {
            let [(bad, _, _), _] =
                sr3_kbins_s2_and_s1(&cell, &prep, &aux, &mesh, &cfg, Some(m)).expect("mutant");
            let (asym, v) = bins_vs_ordered(&bad, &s1, n, mod_l, mod_t);
            let same = bins_bit_diffs(&bad, &prod);
            eprintln!(
                "{dims:?} mutant {m:?}: max|J3| {scale:.2e}; {asym} M-asymmetric; \
                 {same} bits differ from production; {v:?}"
            );
            if trim && m == KPairSymMutant::WrongPairBin {
                assert_eq!(
                    same, 0,
                    "{dims:?}: WrongPairBin must be an identity on a TRIM mesh (−r ≡ r)"
                );
                continue;
            }
            assert!(
                v.over > 0 && v.dmax > MUTANT_ELEMENT_BAR,
                "{dims:?}: mutant {m:?} not caught (max|new − avg| {:.2e})",
                v.dmax
            );
        }
    }
}

// ---------------------------------------------------------------- hcore

/// Items 1-4 for one set of s2 matrices vs the serial ordered loop's.
fn herm_vs_ordered(new: &[Array2<Complex64>], old: &[Array2<Complex64>]) -> (usize, VsAvg) {
    let mut asym = 0usize;
    let mut v = VsAvg::default();
    for (a, o) in new.iter().zip(old) {
        let n = a.nrows();
        for i in 0..n {
            for j in 0..n {
                if i != j && !cbits_eq(a[(i, j)], a[(j, i)].conj()) {
                    asym += 1;
                }
                // `hermitize`'s expression: ½(m[i,j] + conj m[j,i]).
                v.cadd(a[(i, j)], o[(i, j)], o[(j, i)].conj());
            }
        }
    }
    (asym, v)
}

fn hcore_parts(cell: &Cell, prep: &PreparedBasis, mesh: &KPointMesh) -> Vec<HcoreKS2Parts> {
    THREADS
        .iter()
        .map(|&t| {
            in_pool(t, || {
                hcore_kpts_s2_mutant(cell, prep, mesh, &hcore_cfg(), None).expect("hcore s2")
            })
        })
        .collect()
}

#[test]
fn hcore_kpts_s2_is_the_ordered_loop_mirrored_and_conjugated() {
    let (cell, prep, _) = tri_sp();
    let cfg = hcore_cfg();
    for dims in [TRIM, NON_TRIM] {
        let mesh = KPointMesh::gamma_centred(&cell, dims).unwrap();
        let runs = hcore_parts(&cell, &prep, &mesh);
        let (s, t, v, nc, no) = &runs[0];
        for (&th, (s_, t_, v_, nc_, no_)) in THREADS.iter().zip(&runs) {
            for (a, b, what) in [(s_, s, "S"), (t_, t, "T"), (v_, v, "V_SR")] {
                assert_eq!(
                    cmats_bit_diffs(a, b),
                    0,
                    "{dims:?} {what} at {th} vs 1 thread"
                );
            }
            assert_eq!((nc_, no_), (nc, no), "{dims:?} counts at {th}");
        }
        let ([_, (s_old, t_old)], _) =
            overlap_kinetic_kpts_parallel_and_serial(&cell, &prep, &mesh, &cfg).expect("S/T");
        let [_, (v_old, n_old)] =
            sr_attraction_kpts_parallel_and_serial(&cell, &prep, &mesh, &cfg).expect("V_SR");
        for (new, old, what) in [(s, &s_old, "S"), (t, &t_old, "T"), (v, &v_old, "V_SR")] {
            let x: Vec<_> = old.iter().map(mirror_upper_herm).collect();
            assert_eq!(
                cmats_bit_diffs(new, &x),
                0,
                "{dims:?} {what}: s2 vs the ordered μ ≤ ν elements and conjugates"
            );
            let (asym, va) = herm_vs_ordered(new, old);
            eprintln!("{dims:?} {what}: {asym} non-Hermitian off-diagonal; {va:?}");
            assert_eq!(
                asym, 0,
                "{dims:?} {what}: not exactly Hermitian off the diagonal"
            );
            assert!(
                va.hmax <= MAP_BAR,
                "{dims:?} {what}: ordered x and conj y differ by {:.2e}",
                va.hmax
            );
            assert_eq!(va.neither, 0, "{dims:?} {what}: outside {{x, conj y}}");
            assert_eq!(va.over, 0, "{dims:?} {what}: outside the item-4 bound");
        }
        if dims == NON_TRIM {
            assert!(
                s.iter().any(|m| m.iter().any(|z| z.im != 0.0)),
                "{dims:?}: S(k) real everywhere, so the conjugation is vacuous"
            );
        }
        assert_eq!(
            *no, n_old,
            "{dims:?}: ordered-equivalent vs the ordered count"
        );
        assert!(nc < no, "{dims:?}: s2 computed {nc} of {no}");
    }
}

#[test]
fn hcore_kpts_s2_mutants_fail_and_no_conj_needs_a_non_trim_mesh() {
    let (cell, prep, _) = tri_sp();
    let cfg = hcore_cfg();
    for (dims, trim) in [(NON_TRIM, false), (TRIM, true)] {
        let mesh = KPointMesh::gamma_centred(&cell, dims).unwrap();
        let ([_, (s_old, _)], _) =
            overlap_kinetic_kpts_parallel_and_serial(&cell, &prep, &mesh, &cfg).expect("S/T");
        let [_, (v_old, _)] =
            sr_attraction_kpts_parallel_and_serial(&cell, &prep, &mesh, &cfg).expect("V_SR");
        let (s_prod, _, v_prod, _, _) =
            hcore_kpts_s2_mutant(&cell, &prep, &mesh, &cfg, None).expect("production");
        for m in HCORE_MUTANTS {
            let (s, _, v, _, _) =
                hcore_kpts_s2_mutant(&cell, &prep, &mesh, &cfg, Some(m)).expect("mutant");
            let (_, vs) = herm_vs_ordered(&s, &s_old);
            let (_, vv) = herm_vs_ordered(&v, &v_old);
            eprintln!("{dims:?} hcore mutant {m:?}: S {vs:?}; V_SR {vv:?}");
            if trim && m == KHcorePairMutant::NoConj {
                // Real phases: conj changes at most the sign of a zero.
                for (a, b, what) in [(&s, &s_prod, "S"), (&v, &v_prod, "V_SR")] {
                    let differ: usize = a
                        .iter()
                        .zip(b.iter())
                        .map(|(x, y)| x.iter().zip(y.iter()).filter(|(p, q)| p != q).count())
                        .sum();
                    assert_eq!(differ, 0, "{dims:?}: NoConj must be an identity ({what})");
                }
                continue;
            }
            for (va, what) in [(&vs, "S"), (&vv, "V_SR")] {
                assert!(
                    va.over > 0 && va.dmax > MUTANT_ELEMENT_BAR,
                    "{dims:?}: hcore mutant {m:?} not caught in {what} ({:.2e})",
                    va.dmax
                );
            }
        }
    }
}

// ---------------------------------------------------------------- energies

struct KSys {
    cell: Cell,
    prep: PreparedBasis,
    aux: PreparedBasis,
    mesh: KPointMesh,
}

fn ksys(cell: Cell, bs: &BasisSet, aux_bs: &BasisSet, n: [usize; 3]) -> KSys {
    let prep = prep_for(&cell, bs);
    let aux = PreparedBasis::new(cell.mol(), aux_bs).expect("aux prep");
    let mesh = KPointMesh::gamma_centred(&cell, n).expect("mesh");
    KSys {
        cell,
        prep,
        aux,
        mesh,
    }
}

/// The production (s2) or frozen pre-s2 `(h(k), KRsGdf)` of `ks`.
fn build(ks: &KSys, cfg: &KRsGdfConfig, s1: bool) -> (PeriodicHcoreK, KRsGdf) {
    let hk = if s1 {
        periodic_hcore_kpts_pair_s1_oracle(&ks.cell, &ks.prep, &ks.mesh, &hcore_cfg())
    } else {
        periodic_hcore_kpts(&ks.cell, &ks.prep, &ks.mesh, &hcore_cfg())
    }
    .expect("hcore(k)");
    let gdf = if s1 {
        KRsGdf::build_pair_s1_oracle(&ks.cell, &ks.prep, &ks.aux, &ks.mesh, &hk.s, cfg)
    } else {
        KRsGdf::build(&ks.cell, &ks.prep, &ks.aux, &ks.mesh, &hk.s, cfg)
    }
    .expect("KRsGdf");
    (hk, gdf)
}

fn inj<'a>(hk: &PeriodicHcoreK, jk: Box<dyn KPointJk + 'a>) -> KPointInjection<'a> {
    KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk,
    }
}

/// `nab = None`: k-RHF; `Some((na, nb))`: k-UHF. Returns the energy.
fn energy(ks: &KSys, hk: &PeriodicHcoreK, gdf: &KRsGdf, nab: Option<(usize, usize)>) -> f64 {
    let jk = Box::new(gdf.jk_builder());
    match nab {
        None => {
            let r =
                solve_krhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj(hk, jk)).expect("k-RHF");
            assert!(r.converged, "k-RHF not converged");
            r.energy
        }
        Some((na, nb)) => {
            let r = solve_kuhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj(hk, jk), na, nb)
                .expect("k-UHF");
            assert!(r.converged, "k-UHF not converged ({} it)", r.iterations);
            r.energy
        }
    }
}

/// Item 5 (and 6 for the unsplit builds) on one system.
fn energy_vs_pre_s2(ks: &KSys, split: Option<RangeSplit>, nab: Option<(usize, usize)>, tag: &str) {
    let cfg = kgdf_cfg(split, None);
    let (hk_new, new) = build(ks, &cfg, false);
    let (hk_old, old) = build(ks, &cfg, true);
    let (e_new, e_old) = (
        energy(ks, &hk_new, &new, nab),
        energy(ks, &hk_old, &old, nab),
    );
    let (sn, so) = (new.stats(), old.stats());
    eprintln!(
        "{tag}: E s2 {e_new:.15} vs pre-s2 {e_old:.15} (ΔE {:.2e}); SR3 {} computed / {} \
         ordered-equivalent vs {}; hcore SR {} / {} vs {}",
        e_new - e_old,
        sn.n_sr3_triplets,
        sn.n_sr3_triplets_ordered,
        so.n_sr3_triplets,
        hk_new.n_sr_triplets,
        hk_new.n_sr_triplets_ordered,
        hk_old.n_sr_triplets
    );
    assert!(
        (e_new - e_old).abs() <= E_BAR,
        "{tag}: ΔE {:.3e}",
        e_new - e_old
    );
    assert_eq!(
        so.n_sr3_triplets_ordered, so.n_sr3_triplets,
        "{tag}: oracle counts"
    );
    assert!(
        sn.n_sr3_triplets < so.n_sr3_triplets,
        "{tag}: s2 did less work"
    );
    if split.is_none() {
        assert_eq!(
            sn.n_sr3_triplets_ordered, so.n_sr3_triplets,
            "{tag}: ordered-equivalent vs the pre-s2 counter"
        );
    }
    assert_eq!(
        hk_new.n_sr_triplets_ordered, hk_old.n_sr_triplets,
        "{tag}: hcore counts"
    );
    assert_eq!(hk_old.n_sr_triplets_ordered, hk_old.n_sr_triplets);
}

#[test]
fn krhf_energy_matches_the_frozen_pre_s2_build_on_trim_and_non_trim_meshes() {
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    for dims in [TRIM, NON_TRIM] {
        let ks = ksys(h2_cell(4.0), &pyscf_sto3g_h(), &aux, dims);
        energy_vs_pre_s2(&ks, None, None, &format!("H2 KRHF {dims:?}"));
    }
}

#[test]
fn kuhf_energy_matches_the_frozen_pre_s2_build() {
    let ks = ksys(
        cell_at(&H3_ATOMS, cubic(H3_A), 2),
        &pyscf_sto3g_h(),
        &et_sp_aux(),
        NON_TRIM,
    );
    energy_vs_pre_s2(&ks, None, Some((2, 1)), "H3 KUHF 1x1x3");
}

#[test]
fn split_krhf_energy_matches_the_frozen_pre_s2_build() {
    let ks = ksys(h2_cell(4.0), &pyscf_sto3g_h(), &et_sp_aux(), NON_TRIM);
    energy_vs_pre_s2(&ks, Some(RangeSplit::new(1.0)), None, "H2 split KRHF 1x1x3");
}

/// The wrong-pair-bin mutant through the whole k-point RHF: the SAME energy
/// bit for bit at 1×1×2 (its bins are bitwise production's), off by far more
/// than `E_BAR` at 1×1×3.
#[test]
fn wrong_pair_bin_mutant_moves_the_krhf_energy_only_on_a_non_trim_mesh() {
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let mutant = Some(KRsGdfMutation::PairSym(KPairSymMutant::WrongPairBin));
    for (dims, trim) in [(TRIM, true), (NON_TRIM, false)] {
        let ks = ksys(h2_cell(4.0), &pyscf_sto3g_h(), &aux, dims);
        let (hk, good) = build(&ks, &kgdf_cfg(None, None), false);
        let bad = KRsGdf::build(
            &ks.cell,
            &ks.prep,
            &ks.aux,
            &ks.mesh,
            &hk.s,
            &kgdf_cfg(None, mutant),
        )
        .expect("mutant KRsGdf");
        let e_good = energy(&ks, &hk, &good, None);
        // The mutant's SCF is not required to converge (its J3 is wrong at
        // k' ≠ −k'); a failed or unconverged SCF also counts as caught.
        let bad_run = solve_krhf_injected(
            &ks.cell,
            &ks.mesh,
            &kscf_cfg(),
            inj(&hk, Box::new(bad.jk_builder())),
        )
        .map(|r| (r.energy, r.converged));
        eprintln!("H2 KRHF {dims:?}: WrongPairBin run {bad_run:?} vs E {e_good:.15}");
        match (trim, bad_run) {
            (true, Ok((e_bad, conv))) => assert!(
                conv && bits_eq(e_bad, e_good),
                "{dims:?}: WrongPairBin must be an identity on a TRIM mesh"
            ),
            (true, Err(e)) => panic!("{dims:?}: WrongPairBin broke a TRIM-mesh SCF: {e}"),
            (false, Ok((e_bad, conv))) => assert!(
                !conv || (e_bad - e_good).abs() > MUTANT_E_BAR,
                "{dims:?}: WrongPairBin not caught (ΔE {:.3e})",
                e_bad - e_good
            ),
            (false, Err(_)) => {}
        }
    }
}

/// The k-point builds do not implement the Gamma-only SR column rotation:
/// an EXPLICIT request is a typed refusal naming the option, never a
/// silently unrotated build; the default `Auto` (every config above) runs
/// unrotated without complaint.
#[test]
fn kpoint_builds_refuse_the_gamma_column_rotation() {
    use ferric_pbc::sr_rotation::SrColumnRotation;
    let ks = ksys(h2_cell(4.0), &pyscf_sto3g_h(), &et_sp_aux(), TRIM);
    let rot = SrColumnRotation::on();
    assert_eq!(hcore_cfg().sr_column_rotation, SrColumnRotation::Auto);
    let hcfg = hcore_cfg().with_sr_column_rotation(rot);
    let msg = periodic_hcore_kpts(&ks.cell, &ks.prep, &ks.mesh, &hcfg)
        .expect_err("k hcore with a column rotation must be refused")
        .to_string();
    assert!(
        msg.contains("sr_column_rotation") && msg.contains("explicitly"),
        "{msg}"
    );
    let hk = periodic_hcore_kpts(&ks.cell, &ks.prep, &ks.mesh, &hcore_cfg()).expect("hcore(k)");
    let mut cfg = kgdf_cfg(None, None);
    assert_eq!(cfg.gdf.sr_column_rotation, SrColumnRotation::Auto);
    KRsGdf::build(&ks.cell, &ks.prep, &ks.aux, &ks.mesh, &hk.s, &cfg)
        .expect("KRsGdf with the default Auto runs unrotated");
    cfg.gdf.sr_column_rotation = rot;
    let msg = KRsGdf::build(&ks.cell, &ks.prep, &ks.aux, &ks.mesh, &hk.s, &cfg)
        .expect_err("KRsGdf with a column rotation must be refused")
        .to_string();
    assert!(
        msg.contains("sr_column_rotation") && msg.contains("explicitly"),
        "{msg}"
    );
}
