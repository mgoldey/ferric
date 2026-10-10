//! Column rotation of the k-point SR walks (`ferric_pbc::sr_rotation`;
//! `rsgdf::kpoint` and `hcore::kpoint` module docs "Column rotation at k";
//! issue #232, the k-point sibling of the Gamma `pbc_sr_rotation.rs`).
//!
//! Construction under test: the k-point SR 3-centre walk (RS-GDF, unsplit and
//! range split) and the k-point hcore SR attraction run on the column-rotated
//! orbital shells; the finished residue bins `J3[r_L, r_T][μν]` (a general,
//! NON-symmetric pass: the two orientations of a pair live in different bins
//! `b` and `M(b)`) and the Bloch-summed `V_SR(k)` are transformed back with the
//! same real, k-independent `T` before the phase contraction, the LR and the
//! G = 0 terms.
//!
//! What each test pins (the exactness anchor was written and run first):
//!
//! (a) TRIVIAL-LIMIT ANCHOR: a basis with nothing to rotate gives BITWISE
//!     today's build (B(k, k') for every pair, h(k), V(k), counters) with the
//!     flag on and under EVERY mutant (they cannot act), on a TRIM and on a
//!     non-TRIM mesh, unsplit and split.
//! (b) COVARIANCE ANCHOR at the bin level (an independent construction: the
//!     parent walk vs rotated walk + back-transform): at `P_TIGHT`,
//!     `max|bin_rot − bin| ≤ TOL_COV · max|bin|` for EVERY bin, and the
//!     rotated bins keep the pair symmetry `bin[b][μν] = bin[M(b)][νμ]` to
//!     round-off (`SYM_BAR`). Both bars are DERIVED by measuring correct and
//!     mutant values (printed; constants below).
//! (c) The fitted tensors B(k, k') and the hcore matrices rotated vs
//!     unrotated, at the default screen.
//! (d) MUTANTS: the Gamma set (wrong sign of `T_ks`, no libint
//!     renormalisation, one AO index only) plus `T` dropped, `T` where `Tᵀ`
//!     belongs, and `T` applied AFTER the LR and G = 0 terms
//!     (`KRsGdfMutation::RotationLate`; applying it after the phase
//!     contraction but BEFORE the LR is not a defect and is covered by (b)'s
//!     bin-level and (c)'s B-level agreement). Each must miss by orders of
//!     magnitude on the fixture where production passes. A mutant on the
//!     pair symmetry (`WrongPairBin`) must still be caught WITH rotation on.
//! (e) 1x1x1 mesh ≡ the Gamma rotated build; k-mesh ≡ Gamma supercell with
//!     the rotation on.
//! (f) The rotated builds are BITWISE identical across 1/2/6 threads.
//!
//! Artifact hypothesis, stated before measuring. Correct: rotated vs parent
//! bins at the ~1e-15 relative level (same as Gamma), shrinking with
//! precision, pair-symmetry violation at round-off. Back-transform mutants:
//! O(|T_ks|) ≈ 0.5 relative on the rotated columns, independent of the
//! precision. `RotationLate` transforms the (parent-basis) LR and G = 0 parts
//! too, which is O(|T_ks|) times their size. If the correct values were
//! `O(|T_ks|)` or the mutants round-off the experiment could not tell a broken
//! back-transform from a correct one.

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kgrad::{kpoint_rhf_gradient, KGradConfig, KGradJk, KRsGdfGradSource};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KScfConfig, KScfResult};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{
    sr3_kbins_rotated_and_parent, KPairSymMutant, KRsGdf, KRsGdfConfig, KRsGdfMutation,
};
use ferric_pbc::rsgdf::{RangeSplit, RsGdf, RsGdfConfig};
use ferric_pbc::sr_rotation::{ColumnRotation, ColumnRotationMutant, SrColumnRotation};
use ndarray::Array2;
use num_complex::Complex64;

const THREADS: [usize; 3] = [1, 2, 6];
const HCORE_OMEGA: f64 = 0.8;
/// (b): the tight truncation of the covariance anchor (the Gamma file's).
const P_TIGHT: f64 = 1e-16;
/// (d): truncation of the mutant sweeps (they miss by O(|T_ks|), far above
/// any truncation; a looser screen keeps the sweep cheap).
const P_MUTANT: f64 = 1e-11;
/// (b): relative covariance bar (the Gamma file's derivation: truncation
/// ≲ 1e3 · P_TIGHT absolute, round-off ≲ 1e-14 · (1 + Σ|T_ks|)).
const TOL_COV: f64 = 1e-13;
/// (b): rotated bins' pair-symmetry violation, relative to `max|bin|`
/// (measured below; the two halves of a pair are summed in a different order).
const SYM_BAR: f64 = 1e-14;
/// (d): smallest relative miss a mutant must show (the Gamma file's).
const MUTANT_FLOOR: f64 = 1e-6;
/// (c)/(e): rotated vs unrotated at the DEFAULT screen, B(k, k') relative.
const B_BAR: f64 = 1e-9;
/// (c)/(e): RHF energy bar (Ha/cell), the Gamma file's.
const E_BAR: f64 = 1e-10;
/// (h): force after a rotated energy vs the all-unrotated force (Ha/Bohr).
const F_BAR: f64 = 1e-9;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

const OFF: SrColumnRotation = SrColumnRotation::Off;

fn in_pool<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("rayon pool")
        .install(f)
}

fn rotation(m: ColumnRotationMutant) -> SrColumnRotation {
    SrColumnRotation::On(ColumnRotation { mutant: m })
}

fn production() -> SrColumnRotation {
    SrColumnRotation::on()
}

fn bset(name: &str) -> BasisSet {
    basis::bundled(name).expect("bundled basis")
}

fn bundled(cell: &Cell, name: &str) -> PreparedBasis {
    prep_for(cell, &bset(name))
}

fn water_cell() -> Cell {
    Cell::new(Molecule::parse_xyz(WATER, 0, 1).expect("xyz"), cubic(8.0)).expect("cell")
}

/// H2 / cc-pVDZ: the cheapest cell with something to rotate (one s column
/// per H).
fn h2_ccpvdz() -> (Cell, BasisSet, BasisSet) {
    (h2_cell(5.0), bset("cc-pvdz"), bset("cc-pvdz-ri"))
}

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

fn kcfg(
    split: Option<RangeSplit>,
    rot: SrColumnRotation,
    precision: f64,
    mutation: Option<KRsGdfMutation>,
) -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: gdf_cfg(split, rot, precision),
        mutation,
    }
}

fn hcore_cfg(rot: SrColumnRotation, precision: f64) -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision,
        sr_column_rotation: rot,
        ..PeriodicHcoreConfig::with_omega(HCORE_OMEGA)
    }
}

fn kscf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-12,
        grad_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    }
}

struct KSys {
    cell: Cell,
    prep: PreparedBasis,
    aux: PreparedBasis,
    mesh: KPointMesh,
}

fn ksys(cell: Cell, bs: &BasisSet, aux_bs: &BasisSet, n: [usize; 3]) -> KSys {
    let prep = prep_for(&cell, bs);
    let aux = prep_for(&cell, aux_bs);
    let mesh = KPointMesh::gamma_centred(&cell, n).expect("mesh");
    KSys {
        cell,
        prep,
        aux,
        mesh,
    }
}

fn hk_of(ks: &KSys, cfg: &PeriodicHcoreConfig) -> PeriodicHcoreK {
    periodic_hcore_kpts(&ks.cell, &ks.prep, &ks.mesh, cfg).expect("hcore(k)")
}

fn kgdf(ks: &KSys, hk: &PeriodicHcoreK, cfg: &KRsGdfConfig) -> KRsGdf {
    KRsGdf::build(&ks.cell, &ks.prep, &ks.aux, &ks.mesh, &hk.s, cfg).expect("KRsGdf")
}

fn krhf_energy(ks: &KSys, hk: &PeriodicHcoreK, gdf: &KRsGdf) -> f64 {
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk: Box::new(gdf.jk_builder()),
    };
    let r = solve_krhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj).expect("k-RHF");
    assert!(r.converged, "k-RHF not converged ({} it)", r.iterations);
    r.energy
}

fn cmax(a: &Array2<Complex64>) -> f64 {
    a.iter().fold(0.0_f64, |m, z| m.max(z.norm()))
}

fn cdiff(a: &Array2<Complex64>, b: &Array2<Complex64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).norm()))
}

fn c_bit_diffs(a: &Array2<Complex64>, b: &Array2<Complex64>) -> usize {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.re.to_bits() != y.re.to_bits() || x.im.to_bits() != y.im.to_bits())
        .count()
}

/// Largest relative `max|B_a(k,k') − B_b(k,k')| / max|B_b|` over every pair.
fn b_rel(a: &KRsGdf, b: &KRsGdf) -> f64 {
    let nk = a.nk();
    let mut worst = 0.0_f64;
    for k in 0..nk {
        for kp in 0..nk {
            let (x, y) = (a.block(k, kp), b.block(k, kp));
            assert_eq!(x.dim(), y.dim(), "kept aux counts differ at ({k}, {kp})");
            worst = worst.max(cdiff(&x, &y) / cmax(&y));
        }
    }
    worst
}

/// Number of differing bit patterns in B(k, k') over every pair.
fn b_bit_diffs(a: &KRsGdf, b: &KRsGdf) -> usize {
    let nk = a.nk();
    (0..nk)
        .flat_map(|k| (0..nk).map(move |kp| (k, kp)))
        .map(|(k, kp)| c_bit_diffs(&a.block(k, kp), &b.block(k, kp)))
        .sum()
}

/// Largest relative miss of the hcore `h(k)`/`V(k)` against `reference`.
fn hk_rel(a: &PeriodicHcoreK, reference: &PeriodicHcoreK) -> f64 {
    a.v.iter()
        .zip(&reference.v)
        .chain(a.h.iter().zip(&reference.h))
        .fold(0.0_f64, |m, (x, y)| m.max(cdiff(x, y) / cmax(y)))
}

fn hk_bit_diffs(a: &PeriodicHcoreK, b: &PeriodicHcoreK) -> usize {
    a.s.iter()
        .zip(&b.s)
        .chain(a.t.iter().zip(&b.t))
        .chain(a.v.iter().zip(&b.v))
        .chain(a.h.iter().zip(&b.h))
        .map(|(x, y)| c_bit_diffs(x, y))
        .sum()
}

/// Largest `|bin[b][μν] − bin[M(b)][νμ]|` (the k-point pair symmetry).
fn pair_asymmetry(bins: &[Array2<f64>], n: usize, mod_l: [usize; 3], mod_t: [usize; 3]) -> f64 {
    let mut d = 0.0_f64;
    for (b, bin) in bins.iter().enumerate() {
        let mirror = &bins[kpair_mirror(mod_l, mod_t, b)];
        for mu in 0..n {
            for nu in 0..n {
                for p in 0..bin.ncols() {
                    d = d.max((bin[(mu * n + nu, p)] - mirror[(nu * n + mu, p)]).abs());
                }
            }
        }
    }
    d
}

const MUTANTS: [ColumnRotationMutant; 5] = [
    ColumnRotationMutant::FlipSign,
    ColumnRotationMutant::NoRenormalization,
    ColumnRotationMutant::OneSidedBackTransform,
    ColumnRotationMutant::NoBackTransform,
    ColumnRotationMutant::TransposedBackTransform,
];

// ------------------------------------------------------------------ (a)

/// STO-3G s + one Cartesian p: nothing rotates, so every k-point build with
/// the flag on - and under every mutant, `RotationLate` included - is
/// BITWISE the unrotated build, on a TRIM (1x1x2) and a non-TRIM (1x1x3)
/// mesh, unsplit and split.
#[test]
fn identity_rotation_k_builds_are_bitwise_the_unrotated_build() {
    let (obs, aux_bs) = (sp_basis_h(), bset("cc-pvdz-ri"));
    for n in [[1, 1, 2], [1, 1, 3]] {
        let ks = ksys(h2_cell(5.0), &obs, &aux_bs, n);
        let hk0 = hk_of(&ks, &hcore_cfg(OFF, 1e-14));
        assert_eq!(hk0.sr_rotated_columns, 0);
        let mut hk_muts = vec![production()];
        hk_muts.extend(MUTANTS.iter().map(|&m| rotation(m)));
        for rot in hk_muts {
            let hk1 = hk_of(&ks, &hcore_cfg(rot, 1e-14));
            assert_eq!(hk_bit_diffs(&hk1, &hk0), 0, "{n:?} {rot:?}: hcore(k)");
            assert_eq!(hk1.n_sr_triplets, hk0.n_sr_triplets, "{n:?} {rot:?}");
            assert_eq!(hk1.sr_rotated_columns, 0, "{n:?} {rot:?}");
        }
        for split in [None, Some(RangeSplit::default())] {
            let g0 = kgdf(&ks, &hk0, &kcfg(split, OFF, 1e-13, None));
            assert_eq!(g0.stats().sr_rotated_columns, 0);
            let mut cases: Vec<(SrColumnRotation, Option<KRsGdfMutation>)> =
                vec![(production(), None)];
            cases.push((production(), Some(KRsGdfMutation::RotationLate)));
            if split.is_none() {
                cases.extend(MUTANTS.iter().map(|&m| (rotation(m), None)));
            }
            for (rot, mu) in cases {
                let g1 = kgdf(&ks, &hk0, &kcfg(split, rot, 1e-13, mu));
                let tag = format!("{n:?} split {:?} {rot:?} {mu:?}", split.map(|s| s.lambda));
                assert_eq!(b_bit_diffs(&g1, &g0), 0, "{tag}: B");
                let (s1, s0) = (g1.stats(), g0.stats());
                assert_eq!(
                    (s1.n_sr3_triplets, s1.n_sr3_triplets_ordered, s1.n_sr2_pairs),
                    (s0.n_sr3_triplets, s0.n_sr3_triplets_ordered, s0.n_sr2_pairs),
                    "{tag}: counters"
                );
                assert_eq!(s1.sr_rotated_columns, 0, "{tag}: the flag was on");
            }
        }
    }
}

// ------------------------------------------------------------------ (b)

/// Bin-level covariance and pair-symmetry anchor on H2O / cc-pVDZ (5 rotated
/// columns: O 1s, 2s, p and one s per H), unsplit and split, on a non-TRIM
/// (1x1x3: `mod_l = 3`, `−1 ≡ 2`) and a TRIM (1x1x2) mesh.
#[test]
fn rotated_k_bins_are_the_parent_bins_and_keep_the_pair_symmetry() {
    let cell = water_cell();
    let prep = bundled(&cell, "cc-pvdz");
    let aux = bundled(&cell, "cc-pvdz-ri");
    let n = prep.nbasis();
    for mesh_n in [[1, 1, 3], [1, 1, 2]] {
        let mesh = KPointMesh::gamma_centred(&cell, mesh_n).expect("mesh");
        let (mod_l, mod_t) = (mesh.residue_moduli(), mesh.n());
        for split in [None, Some(RangeSplit::default())] {
            let tag = format!("{mesh_n:?} split {:?}", split.map(|s| s.lambda));
            let cfg = gdf_cfg(split, production(), P_TIGHT);
            let ([rot, par], ncol) =
                sr3_kbins_rotated_and_parent(&cell, &prep, &aux, &mesh, &cfg).expect("bins");
            assert_eq!(ncol, 5, "{tag}");
            assert!(rot.1 < par.1, "{tag}: rotated walk {} vs {}", rot.1, par.1);
            let scale = par.0.iter().fold(0.0_f64, |m, b| {
                m.max(b.iter().fold(0.0_f64, |a, x| a.max(x.abs())))
            });
            let d_cov = rot.0.iter().zip(&par.0).fold(0.0_f64, |m, (x, y)| {
                x.iter()
                    .zip(y.iter())
                    .fold(m, |a, (u, v)| a.max((u - v).abs()))
            });
            let d_sym = pair_asymmetry(&rot.0, n, mod_l, mod_t);
            eprintln!(
                "{tag}: max|bin| {scale:.3e}, rel cov {:.2e}, rel pair-symmetry {:.2e}; \
                 triplets rotated {} vs parent {}",
                d_cov / scale,
                d_sym / scale,
                rot.1,
                par.1
            );
            assert!(d_cov > 0.0, "{tag}: rotated bins bitwise the parent's");
            assert!(d_cov / scale <= TOL_COV, "{tag}: cov {:.3e}", d_cov / scale);
            assert!(d_sym / scale <= SYM_BAR, "{tag}: sym {:.3e}", d_sym / scale);
        }
    }
}

// ------------------------------------------------------------- (b), mutants

/// The back-transform mutants fail the bin-level covariance anchor by
/// orders of magnitude on the fixture where production passes (same fixture
/// as the anchor, non-TRIM mesh, unsplit and split).
#[test]
fn back_transform_mutants_fail_the_bin_level_anchor() {
    let cell = water_cell();
    let prep = bundled(&cell, "cc-pvdz");
    let aux = bundled(&cell, "cc-pvdz-ri");
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).expect("mesh");
    for split in [None, Some(RangeSplit::default())] {
        let (mod_l, mod_t, n) = (mesh.residue_moduli(), mesh.n(), prep.nbasis());
        // (relative miss against the parent bins, relative pair-symmetry violation)
        let rel_of = |rot: SrColumnRotation, prec: f64| -> (f64, f64) {
            let cfg = gdf_cfg(split, rot, prec);
            let ([r, p], _) =
                sr3_kbins_rotated_and_parent(&cell, &prep, &aux, &mesh, &cfg).expect("bins");
            let scale = p.0.iter().fold(0.0_f64, |m, b| {
                m.max(b.iter().fold(0.0_f64, |a, x| a.max(x.abs())))
            });
            let d = r.0.iter().zip(&p.0).fold(0.0_f64, |m, (x, y)| {
                x.iter()
                    .zip(y.iter())
                    .fold(m, |a, (u, v)| a.max((u - v).abs()))
            });
            (d / scale, pair_asymmetry(&r.0, n, mod_l, mod_t) / scale)
        };
        let (good, good_sym) = rel_of(production(), P_TIGHT);
        assert!(good <= TOL_COV, "production {good:.3e}");
        assert!(good_sym <= SYM_BAR, "production symmetry {good_sym:.3e}");
        // The symmetry diagnostic must itself be able to fail: a one-sided
        // back-transform breaks `bin[b][μν] = bin[M(b)][νμ]`.
        let (_, one_sided_sym) = rel_of(
            rotation(ColumnRotationMutant::OneSidedBackTransform),
            P_MUTANT,
        );
        eprintln!(
            "one-sided mutant: pair-symmetry violation {one_sided_sym:.3e} (bar {SYM_BAR:e})"
        );
        assert!(one_sided_sym >= MUTANT_FLOOR, "{one_sided_sym:.3e}");
        for m in MUTANTS {
            let (bad, _) = rel_of(rotation(m), P_MUTANT);
            eprintln!(
                "split {:?} {m:?}: rel miss {bad:.3e} (production {good:.2e})",
                split.map(|s| s.lambda)
            );
            assert!(bad >= MUTANT_FLOOR, "{m:?} only misses by {bad:.3e}");
        }
    }
}

// ------------------------------------------------------------------ (c)

/// The fitted B(k, k') and the hcore h(k), V(k) rotated vs unrotated at the
/// default screen on H2O / cc-pVDZ (5 rotated columns), non-TRIM mesh,
/// unsplit and split.
#[test]
fn rotated_k_builds_match_unrotated() {
    let ks = ksys(
        water_cell(),
        &bset("cc-pvdz"),
        &bset("cc-pvdz-ri"),
        [1, 1, 3],
    );
    let hk_off = hk_of(&ks, &hcore_cfg(OFF, 1e-14));
    let hk_on = hk_of(&ks, &hcore_cfg(production(), 1e-14));
    assert_eq!(hk_on.sr_rotated_columns, 5);
    assert!(
        hk_bit_diffs(&hk_on, &hk_off) > 0,
        "hcore bitwise the unrotated one: not rotated"
    );
    let h_good = hk_rel(&hk_on, &hk_off);
    eprintln!(
        "hcore(k): rel {h_good:.2e}; SR triplets {} vs {}",
        hk_on.n_sr_triplets, hk_off.n_sr_triplets
    );
    assert!(h_good <= TOL_COV, "hcore {h_good:.3e}");
    assert!(hk_on.n_sr_triplets < hk_off.n_sr_triplets);
    // S(k), T(k) are outside the rotated walk: bitwise.
    for (a, b) in hk_on
        .s
        .iter()
        .zip(&hk_off.s)
        .chain(hk_on.t.iter().zip(&hk_off.t))
    {
        assert_eq!(c_bit_diffs(a, b), 0, "S/T(k) outside the rotated walk");
    }
    for split in [None, Some(RangeSplit::default())] {
        let tag = format!("split {:?}", split.map(|s| s.lambda));
        let g_off = kgdf(&ks, &hk_off, &kcfg(split, OFF, 1e-13, None));
        let g_on = kgdf(&ks, &hk_off, &kcfg(split, production(), 1e-13, None));
        let (s_on, s_off) = (g_on.stats(), g_off.stats());
        assert_eq!(s_on.sr_rotated_columns, 5, "{tag}");
        assert!(s_on.n_sr3_triplets < s_off.n_sr3_triplets, "{tag}");
        assert_eq!(s_on.n_sr2_pairs, s_off.n_sr2_pairs, "{tag}");
        let good = b_rel(&g_on, &g_off);
        eprintln!(
            "{tag}: B rel {good:.2e}; SR3 triplets {} (ordered {}) vs {} (ordered {})",
            s_on.n_sr3_triplets,
            s_on.n_sr3_triplets_ordered,
            s_off.n_sr3_triplets,
            s_off.n_sr3_triplets_ordered
        );
        assert!(good > 0.0, "{tag}: B bitwise the unrotated one");
        assert!(good <= B_BAR, "{tag}: B rel {good:.3e}");
    }
}

/// MUTANTS on the fitted B(k, k') and h(k): the five back-transform defects
/// and `RotationLate` (and the hcore ones) miss by orders of magnitude on
/// H2 / cc-pVDZ (the cheapest cell with a rotated column) where production
/// matches the unrotated build to `B_BAR`; a pair-symmetry defect
/// (`WrongPairBin`) is still caught WITH the rotation on, on a non-TRIM mesh.
#[test]
fn rotated_k_mutants_miss_the_fitted_tensors() {
    let (cell, obs, aux_bs) = h2_ccpvdz();
    let ks = ksys(cell, &obs, &aux_bs, [1, 1, 3]);
    let hk_off = hk_of(&ks, &hcore_cfg(OFF, 1e-14));
    let hk_on = hk_of(&ks, &hcore_cfg(production(), 1e-14));
    let h_good = hk_rel(&hk_on, &hk_off);
    assert!(h_good <= TOL_COV, "hcore production {h_good:.3e}");
    for m in MUTANTS {
        let hk_m = hk_of(&ks, &hcore_cfg(rotation(m), 1e-14));
        let bad = hk_rel(&hk_m, &hk_off);
        eprintln!("hcore mutant {m:?}: rel miss {bad:.3e} (production {h_good:.2e})");
        assert!(bad >= MUTANT_FLOOR, "hcore {m:?} only misses by {bad:.3e}");
    }
    for split in [None, Some(RangeSplit::default())] {
        let tag = format!("split {:?}", split.map(|s| s.lambda));
        let g_off = kgdf(&ks, &hk_off, &kcfg(split, OFF, 1e-13, None));
        let g_on = kgdf(&ks, &hk_off, &kcfg(split, production(), 1e-13, None));
        let good = b_rel(&g_on, &g_off);
        assert!(good > 0.0 && good <= B_BAR, "{tag}: production {good:.3e}");
        let mut bad_cases: Vec<(String, KRsGdfConfig)> = MUTANTS
            .iter()
            .map(|&m| (format!("{m:?}"), kcfg(split, rotation(m), 1e-13, None)))
            .collect();
        bad_cases.push((
            "RotationLate".into(),
            kcfg(
                split,
                production(),
                1e-13,
                Some(KRsGdfMutation::RotationLate),
            ),
        ));
        bad_cases.push((
            "WrongPairBin + rotation".into(),
            kcfg(
                split,
                production(),
                1e-13,
                Some(KRsGdfMutation::PairSym(KPairSymMutant::WrongPairBin)),
            ),
        ));
        for (name, cfg) in bad_cases {
            let g = kgdf(&ks, &hk_off, &cfg);
            let bad = b_rel(&g, &g_off);
            eprintln!("{tag} {name}: B rel miss {bad:.3e} (production {good:.2e})");
            assert!(
                bad >= MUTANT_FLOOR,
                "{tag} {name}: only misses by {bad:.3e}"
            );
        }
    }
}

/// k-point RHF energy rotated (hcore AND fit) vs unrotated: H2 / cc-pVDZ on
/// a non-TRIM (1x1x3) and a TRIM (1x1x2) mesh, unsplit and split.
#[test]
fn rotated_k_rhf_energy_matches_the_unrotated_build() {
    let (cell, obs, aux_bs) = h2_ccpvdz();
    for n in [[1, 1, 3], [1, 1, 2]] {
        let ks = ksys(cell.clone(), &obs, &aux_bs, n);
        let hk_off = hk_of(&ks, &hcore_cfg(OFF, 1e-14));
        let hk_on = hk_of(&ks, &hcore_cfg(production(), 1e-14));
        assert_eq!(hk_on.sr_rotated_columns, 2, "one s column per H");
        for split in [None, Some(RangeSplit::default())] {
            let g_off = kgdf(&ks, &hk_off, &kcfg(split, OFF, 1e-13, None));
            let g_on = kgdf(&ks, &hk_on, &kcfg(split, production(), 1e-13, None));
            let e0 = krhf_energy(&ks, &hk_off, &g_off);
            let e1 = krhf_energy(&ks, &hk_on, &g_on);
            eprintln!(
                "{n:?} split {:?}: E off {e0:.12} on {e1:.12} dE {:.2e}",
                split.map(|s| s.lambda),
                e1 - e0
            );
            assert!((e1 - e0).abs() <= E_BAR, "{n:?}: dE {:.3e}", e1 - e0);
        }
    }
}

// ------------------------------------------------------------------ (e)

/// A 1x1x1 mesh with the rotation on is the Gamma rotated build: the fitted
/// kernels, h, and the RHF energy.
#[test]
fn one_point_mesh_is_the_gamma_rotated_build() {
    let (cell, obs, aux_bs) = h2_ccpvdz();
    let prep = prep_for(&cell, &obs);
    let aux = prep_for(&cell, &aux_bs);
    for split in [None, Some(RangeSplit::default())] {
        let tag = format!("split {:?}", split.map(|s| s.lambda));
        let hc = periodic_hcore(&cell, &prep, &hcore_cfg(production(), 1e-14)).expect("hcore");
        assert_eq!(hc.sr_rotated_columns, 2);
        let cfg = gdf_cfg(split, production(), 1e-13);
        let gamma = RsGdf::build(&cell, &prep, &aux, &hc.s, &cfg).expect("rsgdf");
        assert_eq!(
            gamma.timings().counter("rsgdf SR3 rotated columns"),
            Some(2)
        );
        let ks = ksys(cell.clone(), &obs, &aux_bs, [1, 1, 1]);
        let hk = hk_of(&ks, &hcore_cfg(production(), 1e-14));
        assert_eq!(hk.sr_rotated_columns, 2);
        let kg = kgdf(&ks, &hk, &kcfg(split, production(), 1e-13, None));
        assert_eq!(kg.stats().sr_rotated_columns, 2);
        let eri = gamma
            .fitted_eri(ferric_pbc::rsgdf::DEFAULT_FITTED_ERI_MAX_BYTES)
            .expect("eri");
        let (jf, kf) = kg
            .fitted_kernels(ferric_pbc::rsgdf::kpoint::DEFAULT_K_FITTED_KERNELS_MAX_BYTES)
            .expect("kernels");
        let mut d = 0.0_f64;
        for (x, y) in jf[0]
            .iter()
            .zip(eri.iter())
            .chain(kf[0].iter().zip(eri.iter()))
        {
            d = d.max((x - Complex64::new(*y, 0.0)).norm());
        }
        let dh = cdiff(&hk.h[0], &hc.h.mapv(|x| Complex64::new(x, 0.0)));
        let eg = gamma_rhf_jk(
            &cell,
            &prep,
            &hc,
            Box::new(gamma.j_builder()),
            Box::new(gamma.k_builder()),
        )
        .energy;
        let ek = krhf_energy(&ks, &hk, &kg);
        eprintln!(
            "{tag}: 1x1x1 vs Gamma rotated: kernels {d:.2e}, h {dh:.2e}, dE {:.2e}",
            ek - eg
        );
        assert!(d < 1e-12, "{tag}: kernels {d:.3e}");
        assert!(dh < 1e-12, "{tag}: h {dh:.3e}");
        assert!((ek - eg).abs() < 1e-12, "{tag}: dE {:.3e}", ek - eg);
    }
}

/// Explicit diag(n) supercell (copy of `pbc_krsgdf.rs::supercell`).
fn supercell(cell: &Cell, n: [usize; 3]) -> Cell {
    let a = *cell.lattice();
    let mut atoms = Vec::new();
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                let t: Vec<f64> = (0..3)
                    .map(|d| m0 as f64 * a[0][d] + m1 as f64 * a[1][d] + m2 as f64 * a[2][d])
                    .collect();
                for p in cell.positions() {
                    atoms.push([p[0] + t[0], p[1] + t[1], p[2] + t[2]]);
                }
            }
        }
    }
    let mut lat = a;
    for (i, row) in lat.iter_mut().enumerate() {
        for v in row.iter_mut() {
            *v *= n[i] as f64;
        }
    }
    Cell::new(hydrogens(&atoms), lat).expect("supercell")
}

/// The k-mesh ≡ supercell anchor still holds with the rotation on: the
/// 1x1x3 rotated k energy per cell equals the Gamma energy of the explicit
/// 3-cell supercell per cell, rotated and unrotated (each side against the
/// independent construction).
#[test]
fn rotated_k_mesh_is_the_gamma_supercell() {
    let (cell, obs, aux_bs) = h2_ccpvdz();
    let n = [1, 1, 3];
    let sc = supercell(&cell, n);
    let prep_sc = bundled(&sc, "cc-pvdz");
    let aux_sc = bundled(&sc, "cc-pvdz-ri");
    let esc = |rot: SrColumnRotation| -> f64 {
        let hc = periodic_hcore(&sc, &prep_sc, &hcore_cfg(rot, 1e-14)).expect("hcore");
        let g =
            RsGdf::build(&sc, &prep_sc, &aux_sc, &hc.s, &gdf_cfg(None, rot, 1e-13)).expect("gdf");
        let r = gamma_rhf_jk(
            &sc,
            &prep_sc,
            &hc,
            Box::new(g.j_builder()),
            Box::new(g.k_builder()),
        );
        r.energy / 3.0
    };
    let ks = ksys(cell, &obs, &aux_bs, n);
    let ek = |rot: SrColumnRotation| -> f64 {
        let hk = hk_of(&ks, &hcore_cfg(rot, 1e-14));
        let g = kgdf(&ks, &hk, &kcfg(None, rot, 1e-13, None));
        krhf_energy(&ks, &hk, &g)
    };
    let (sc_on, sc_off) = (esc(production()), esc(OFF));
    let (k_on, k_off) = (ek(production()), ek(OFF));
    eprintln!(
        "E_k rot {k_on:.12} off {k_off:.12}; E_sc/3 rot {sc_on:.12} off {sc_off:.12}; \
         k-sc rotated {:.2e}, unrotated {:.2e}",
        k_on - sc_on,
        k_off - sc_off
    );
    assert!(
        (k_on - sc_on).abs() <= E_BAR,
        "rotated: {:.3e}",
        k_on - sc_on
    );
    assert!(
        (k_off - sc_off).abs() <= E_BAR,
        "unrotated: {:.3e}",
        k_off - sc_off
    );
    assert!(
        (k_on - sc_off).abs() <= E_BAR,
        "cross: {:.3e}",
        k_on - sc_off
    );
}

// ------------------------------------------------------------------ (f)

/// The rotated k builds are BITWISE identical across 1/2/6 threads.
#[test]
fn rotated_k_builds_are_bitwise_across_thread_counts() {
    let ks = ksys(
        water_cell(),
        &bset("cc-pvdz"),
        &bset("cc-pvdz-ri"),
        [1, 1, 3],
    );
    let cfg = kcfg(Some(RangeSplit::default()), production(), 1e-13, None);
    let runs: Vec<(PeriodicHcoreK, KRsGdf)> = THREADS
        .iter()
        .map(|&t| {
            in_pool(t, || {
                let hk = hk_of(&ks, &hcore_cfg(production(), 1e-14));
                let g = kgdf(&ks, &hk, &cfg);
                (hk, g)
            })
        })
        .collect();
    let (hk_ref, g_ref) = &runs[0];
    for (&t, (hk, g)) in THREADS.iter().zip(&runs) {
        assert_eq!(hk_bit_diffs(hk, hk_ref), 0, "hcore(k) at {t} threads");
        assert_eq!(b_bit_diffs(g, g_ref), 0, "B at {t} threads");
        assert_eq!(hk.n_sr_triplets, hk_ref.n_sr_triplets, "{t} threads");
        assert_eq!(
            g.stats().n_sr3_triplets,
            g_ref.stats().n_sr3_triplets,
            "{t} threads"
        );
    }
}

// ------------------------------------------------------------------ (g)

/// Defaults and refusals: `Auto` rotates the k-point ENERGY builds (bitwise
/// the explicit `On`); the frozen s1 oracles run unrotated under `Auto` and
/// refuse an explicit `On`; the `RotateAux` mutant is refused by the hcore.
#[test]
fn k_default_rotates_and_the_oracles_refuse() {
    let (cell, obs, aux_bs) = h2_ccpvdz();
    let ks = ksys(cell, &obs, &aux_bs, [1, 1, 3]);
    let hdef = PeriodicHcoreConfig::with_omega(HCORE_OMEGA);
    assert_eq!(hdef.sr_column_rotation, SrColumnRotation::Auto);
    let hk_def = hk_of(&ks, &hdef);
    let hk_on = hk_of(&ks, &hdef.with_sr_column_rotation(production()));
    assert_eq!(hk_def.sr_rotated_columns, 2);
    assert_eq!(hk_bit_diffs(&hk_def, &hk_on), 0, "hcore Auto vs On");
    let kdef = KRsGdfConfig {
        gdf: RsGdfConfig {
            exxdiv: ExxDiv::None,
            budget_bytes: Some(1 << 30),
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(kdef.gdf.sr_column_rotation, SrColumnRotation::Auto);
    let g_def = kgdf(&ks, &hk_def, &kdef);
    let mut kon = kdef;
    kon.gdf.sr_column_rotation = production();
    let g_on = kgdf(&ks, &hk_def, &kon);
    assert_eq!(g_def.stats().sr_rotated_columns, 2);
    assert_eq!(b_bit_diffs(&g_def, &g_on), 0, "KRsGdf Auto vs On");
    // Frozen oracles: Auto resolves off (bitwise explicit Off), On refused.
    let oracle = |cfg: &KRsGdfConfig| {
        KRsGdf::build_pair_s1_oracle(&ks.cell, &ks.prep, &ks.aux, &ks.mesh, &hk_def.s, cfg)
    };
    let mut koff = kon;
    koff.gdf.sr_column_rotation = OFF;
    let mut kauto = kon;
    kauto.gdf.sr_column_rotation = SrColumnRotation::Auto;
    let (o_auto, o_off) = (oracle(&kauto).expect("auto"), oracle(&koff).expect("off"));
    assert_eq!(b_bit_diffs(&o_auto, &o_off), 0, "s1 oracle Auto vs Off");
    assert_eq!(o_auto.stats().sr_rotated_columns, 0);
    let err = oracle(&kon).expect_err("explicit On on the s1 oracle");
    assert!(err.to_string().contains("sr_column_rotation"), "{err}");
    let err = ferric_pbc::hcore::kpoint::periodic_hcore_kpts_pair_s1_oracle(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(production(), 1e-14),
    )
    .expect_err("explicit On on the hcore s1 oracle");
    assert!(err.to_string().contains("sr_column_rotation"), "{err}");
    let err = periodic_hcore_kpts(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg(rotation(ColumnRotationMutant::RotateAux), 1e-14),
    )
    .expect_err("RotateAux on the hcore");
    assert!(err.to_string().contains("RotateAux"), "{err}");
}

// ------------------------------------------------------------------ (h)

fn krhf_scf(ks: &KSys, hk: &PeriodicHcoreK, gdf: &KRsGdf) -> KScfResult {
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk: Box::new(gdf.jk_builder()),
    };
    let r = solve_krhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj).expect("k-RHF");
    assert!(r.converged, "k-RHF not converged ({} it)", r.iterations);
    r
}

/// The k-point FORCE builds stay unrotated: a rotated energy build (hcore AND
/// fit, `Auto`) is differentiated by the unrotated derivative walks (the
/// force path rebuilds the fit and checks it against the given one), and the
/// force equals the all-unrotated force to the screening precision. An
/// explicit `On` on either force input is refused by name, never ignored.
#[test]
fn k_forces_follow_the_rotated_energy_and_refuse_an_explicit_rotation() {
    let (cell, obs, aux_bs) = h2_ccpvdz();
    let ks = ksys(cell, &obs, &aux_bs, [1, 1, 2]);
    // Energy builds at `energy_rot` (built and converged once); the force
    // call's hcore / fit configs at `h_rot` / `g_rot`.
    let energy = |rot: SrColumnRotation| {
        let hk = hk_of(&ks, &hcore_cfg(rot, 1e-11));
        let g = kgdf(&ks, &hk, &kcfg(None, rot, 1e-13, None));
        let scf = krhf_scf(&ks, &hk, &g);
        (hk, g, scf)
    };
    let (hk_a, g_a, scf_a) = energy(SrColumnRotation::Auto);
    let (hk_o, g_o, scf_o) = energy(OFF);
    let force = |which: &(PeriodicHcoreK, KRsGdf, KScfResult),
                 h_rot: SrColumnRotation,
                 g_rot: SrColumnRotation| {
        let fit_cfg = kcfg(None, g_rot, 1e-13, None);
        kpoint_rhf_gradient(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg(h_rot, 1e-11),
            &which.0,
            KGradJk::RsGdf(KRsGdfGradSource {
                gdf: &which.1,
                cfg: &fit_cfg,
                aux: &ks.aux,
            }),
            &which.2,
            ExxDiv::None,
            &KGradConfig::default(),
        )
    };
    let (rot_run, off_run) = ((hk_a, g_a, scf_a), (hk_o, g_o, scf_o));
    let auto = SrColumnRotation::Auto;
    let f_rot = force(&rot_run, auto, auto).expect("force after a rotated energy");
    let f_off = force(&off_run, OFF, OFF).expect("unrotated force");
    let d = f_rot
        .grad
        .iter()
        .zip(f_off.grad.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    eprintln!("k force, rotated energy vs unrotated: max|dF| {d:.2e}");
    assert!(d <= F_BAR, "force dF {d:.3e}");
    for (name, h_rot, g_rot) in [
        ("hcore config", production(), auto),
        ("fit config", auto, production()),
    ] {
        let err = force(&rot_run, h_rot, g_rot)
            .err()
            .unwrap_or_else(|| panic!("{name}: explicit On not refused"));
        let msg = err.to_string();
        assert!(
            msg.contains("sr_column_rotation") && msg.contains("explicitly"),
            "{name}: {msg}"
        );
    }
}
