//! Shared harness of `pbc_kpair_symmetry.rs` (per-commit) and
//! `validation_pbc_kpair_symmetry.rs` (weekly): fixtures, the item 1-7
//! checks of the module doc of `pbc_kpair_symmetry.rs` and the bins-mutant
//! body, parameterised by fixture and thread counts.
#![allow(dead_code)]

use crate::common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::hcore::kpoint::{
    hcore_kpts_s2_mutant, periodic_hcore_kpts, periodic_hcore_kpts_pair_s1_oracle, HcoreKS2Parts,
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

pub const THREADS: [usize; 3] = [1, 2, 6];
/// Nuclear-attraction ω and precision (as `pbc_krsgdf_split.rs`).
pub const HCORE_OMEGA: f64 = 0.8;
pub const HCORE_PRECISION: f64 = 1e-14;
/// RS-GDF ω (λ = 1 moves orbital a ≤ 0.5, aux α ≤ 1).
pub const GDF_OMEGA: f64 = 1.0;
pub const AMPLE: usize = 1 << 31;
/// Item 5 of the module doc.
pub const E_BAR: f64 = 1e-11;
/// A caught mutant is off by O(|J3|) ~ 1e-2 (artifact hypothesis); this is
/// 5+ decades below that and 7+ above the round-off scale of item 4.
pub const MUTANT_ELEMENT_BAR: f64 = 1e-8;
/// Item 7: max ½|x − y| of the ordered walk's two evaluations under the map.
/// They differ by summation order (~1e-16 relative) and by a triplet the
/// screen keeps in one orientation only, exactly at its radius, which
/// contributes at most ~the screen precision (1e-13 RS-GDF, 1e-14 hcore,
/// absolute). A WRONG map pairs different lattice sums: O(|J3|) ~ 1e-2. The
/// bar sits 3 decades above the first and 8 below the second.
pub const MAP_BAR: f64 = 1e-10;
/// Item 7 with a range split: a compact × split pair's two orientations run
/// DIFFERENT kept calls (1 vs 2) with their own piece screens, so `x` and `y`
/// agree only to the accumulated screen tolerance (many sub-threshold
/// triplets), not to round-off. Still 6 decades below a wrong map.
pub const MAP_BAR_SPLIT: f64 = 1e-8;
/// A caught energy mutant: 3 decades above `E_BAR`.
pub const MUTANT_E_BAR: f64 = 1e-8;

pub const TRIM: [usize; 3] = [1, 1, 2];
pub const NON_TRIM: [usize; 3] = [1, 1, 3];

pub const KPAIR_MUTANTS: [KPairSymMutant; 5] = [
    KPairSymMutant::WrongPairBin,
    KPairSymMutant::NoAuxShift,
    KPairSymMutant::DropTransposed,
    KPairSymMutant::SkipDiagonal,
    KPairSymMutant::DiagonalNoMirror,
];
pub const HCORE_MUTANTS: [KHcorePairMutant; 3] = [
    KHcorePairMutant::NoConj,
    KHcorePairMutant::DropTransposed,
    KHcorePairMutant::SkipDiagonal,
];

/// H3 (Bohr), a = 4.5, STO-3G, doublet (2, 1) per cell (`pbc_krsgdf_split.rs`).
pub const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
pub const H3_A: f64 = 4.5;

/// Runs `f` inside a fresh rayon pool of `n` threads and returns its result.
pub fn in_pool<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("rayon pool")
        .install(f)
}

pub fn hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(HCORE_OMEGA)
    }
}

/// k-point SCF config with `energy_conv = 1e-13`, `grad_conv = 1e-10` and
/// `max_iter = 400`; the other fields are the defaults.
pub fn kscf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    }
}

pub fn gdf_cfg(split: Option<RangeSplit>) -> RsGdfConfig {
    RsGdfConfig {
        omega: GDF_OMEGA,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        range_split: split,
        ..Default::default()
    }
}

/// k-point RS-GDF config wrapping `gdf_cfg(split)` with the test-only
/// `mutation`.
pub fn kgdf_cfg(split: Option<RangeSplit>, mutation: Option<KRsGdfMutation>) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: gdf_cfg(split),
        mutation,
    }
}

/// The prototype's ET-sp aux on H (`pbc_krsgdf_split.rs`): s(4.7, 1.9, 0.75,
/// 0.3) + p(1.25, 0.5), one unit-normalised primitive per shell. Small, so
/// the bins tests stay cheap; λ = 1 moves s 0.75, 0.3 and p 0.5.
pub fn et_sp_aux() -> BasisSet {
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
pub fn tri_sp() -> (Cell, PreparedBasis, PreparedBasis) {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    let aux = PreparedBasis::new(cell.mol(), &et_sp_aux()).expect("aux");
    (cell, prep, aux)
}

/// A neutral hydrogen cell with atoms at `pos` (Bohr), lattice rows `lattice`
/// (Bohr) and spin multiplicity `mult`.
pub fn cell_at(pos: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

pub fn bits_eq(x: f64, y: f64) -> bool {
    x.to_bits() == y.to_bits()
}

pub fn cbits_eq(x: Complex64, y: Complex64) -> bool {
    bits_eq(x.re, y.re) && bits_eq(x.im, y.im)
}

pub fn bins_bit_diffs(a: &[Array2<f64>], b: &[Array2<f64>]) -> usize {
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

pub fn cmats_bit_diffs(a: &[Array2<Complex64>], b: &[Array2<Complex64>]) -> usize {
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
pub struct VsAvg {
    /// max |new − avg|.
    pub dmax: f64,
    /// max ½|x − y| (the ordered walk's own asymmetry).
    pub hmax: f64,
    /// Elements over `½|x − y| + ε|avg|`.
    pub over: usize,
    /// Elements bitwise neither `x` nor `y`.
    pub neither: usize,
    /// Elements whose `x` and `y` differ in bits (the "which one" checks are
    /// vacuous when this is 0).
    pub observable: usize,
}

impl VsAvg {
    pub fn add(&mut self, new: f64, x: f64, y: f64) {
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

    pub fn cadd(&mut self, new: Complex64, x: Complex64, y: Complex64) {
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
pub fn bins_vs_ordered(
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

pub fn live_bins(bins: &[Array2<f64>]) -> usize {
    bins.iter().filter(|m| m.iter().any(|x| *x != 0.0)).count()
}

pub fn kbins_runs(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: &PreparedBasis,
    mesh: &KPointMesh,
    cfg: &RsGdfConfig,
    threads: &[usize],
) -> Vec<[KSr3Parts; 2]> {
    threads
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
pub fn check_kbins(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: &PreparedBasis,
    mesh: &KPointMesh,
    split: Option<RangeSplit>,
    tag: &str,
    threads: &[usize],
) {
    let n = prep.nbasis();
    let (mod_l, mod_t) = (mesh.residue_moduli(), mesh.n());
    let runs = kbins_runs(cell, prep, aux, mesh, &gdf_cfg(split), threads);
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
    for (&t, [(a2, d2, p2), (a1, d1, p1)]) in threads.iter().zip(&runs) {
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

// ---------------------------------------------------------------- hcore

/// Items 1-4 for one set of s2 matrices vs the serial ordered loop's.
pub fn herm_vs_ordered(new: &[Array2<Complex64>], old: &[Array2<Complex64>]) -> (usize, VsAvg) {
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

pub fn hcore_parts(cell: &Cell, prep: &PreparedBasis, mesh: &KPointMesh) -> Vec<HcoreKS2Parts> {
    THREADS
        .iter()
        .map(|&t| {
            in_pool(t, || {
                hcore_kpts_s2_mutant(cell, prep, mesh, &hcore_cfg(), None).expect("hcore s2")
            })
        })
        .collect()
}

// ---------------------------------------------------------------- energies

pub struct KSys {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub aux: PreparedBasis,
    pub mesh: KPointMesh,
}

pub fn ksys(cell: Cell, bs: &BasisSet, aux_bs: &BasisSet, n: [usize; 3]) -> KSys {
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
pub fn build(ks: &KSys, cfg: &KRsGdfConfig, s1: bool) -> (PeriodicHcoreK, KRsGdf) {
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

pub fn inj<'a>(hk: &PeriodicHcoreK, jk: Box<dyn KPointJk + 'a>) -> KPointInjection<'a> {
    KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk,
    }
}

/// `nab = None`: k-RHF; `Some((na, nb))`: k-UHF. Returns the energy.
pub fn energy(ks: &KSys, hk: &PeriodicHcoreK, gdf: &KRsGdf, nab: Option<(usize, usize)>) -> f64 {
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
pub fn energy_vs_pre_s2(
    ks: &KSys,
    split: Option<RangeSplit>,
    nab: Option<(usize, usize)>,
    tag: &str,
) {
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

/// Two H of the triclinic cell (STO-3G s + one Cartesian p, 4 shells), the
/// ET s+p aux: the canaries' smaller twin of [`tri_sp`] with the same
/// 1×1, 1×3, 3×3 blocks (within-shell `μ ≠ ν` pairs, so the diagonal orbit
/// rule stays live) and compact × split pairs at ω = 1, λ = 1.
pub fn tri2_sp() -> (Cell, PreparedBasis, PreparedBasis) {
    let cell = cell_at(&TRI_ATOMS[..2], TRI_A, 1);
    let prep = prep_for(&cell, &sp_basis_h());
    let aux = PreparedBasis::new(cell.mol(), &et_sp_aux()).expect("aux");
    (cell, prep, aux)
}

/// Every bins mutant leaves `{x, y}` by far more than round-off on each
/// non-TRIM mesh of `meshes` (`(dims, trim)`); on a TRIM mesh
/// `WrongPairBin` is bitwise the production walk and every other mutant is
/// still caught.
pub fn kbins_mutants_case(
    cell: &Cell,
    prep: &PreparedBasis,
    aux: &PreparedBasis,
    meshes: &[([usize; 3], bool)],
) {
    let n = prep.nbasis();
    let cfg = gdf_cfg(None);
    for &(dims, trim) in meshes {
        let mesh = KPointMesh::gamma_centred(cell, dims).unwrap();
        let (mod_l, mod_t) = (mesh.residue_moduli(), mesh.n());
        let [(prod, _, _), (s1, _, _)] =
            sr3_kbins_s2_and_s1(cell, prep, aux, &mesh, &cfg, None).expect("production");
        let scale = prod
            .iter()
            .fold(0.0_f64, |a, m| m.iter().fold(a, |b, x| b.max(x.abs())));
        for m in KPAIR_MUTANTS {
            let [(bad, _, _), _] =
                sr3_kbins_s2_and_s1(cell, prep, aux, &mesh, &cfg, Some(m)).expect("mutant");
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
