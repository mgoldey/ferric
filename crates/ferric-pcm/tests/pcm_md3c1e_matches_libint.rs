//! Exactness anchor for the batched-kernel PCM potentials and reaction-field
//! operator (`ferric_integrals::md3c1e`) against the libint2 path.
//!
//! The libint path (`*_libint`, one nuclear-attraction probe per tessera) is
//! an independent construction of the same integrals (Obara-Saika vs
//! McMurchie-Davidson, separate Boys functions and cart->sph tables), so
//! agreement is corroboration. The kernel's own bar against libint is
//! `1e-12 * max(1, max|A|)` per integral (`md3c1e_matches_cosx_a.rs`).
//!
//! Derived bars (stated before measuring):
//!   v_k  = nuc_k - sum_{mu nu} D_{mu nu} A^k_{mu nu}
//!     |dv_k| <= 1e-12 * A_scale * sum|D|              (sum over ALL entries)
//!   V_{mu nu} = - sum_k q_k A^k_{mu nu}
//!     |dV_{mu nu}| <= 1e-12 * A_scale * sum|q|
//! with `A_scale = max(1, max|A^k_{mu nu}|)` over the tesserae of the test,
//! taken from the kernel's own dense `a_matrices`.
//!
//! Artifact hypothesis: a dropped factor 2 on off-diagonal shell pairs moves
//! v by O(0.1..1) Ha and V by O(1e-2); a flipped kernel sign moves both by
//! 2x their value. Both are orders of magnitude outside the bars, and the
//! reachability asserts below fail if the test molecule has no off-diagonal
//! pair or no d/f shell.
//!
//! Pool independence: the same inputs through rayon pools of 1, 2 and 6
//! threads give identical bits.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::cosx_a::CosxScreen;
use ferric_integrals::md3c1e::Md3c1e;
use ferric_pcm::potential::{
    build_reaction_field_operator, build_reaction_field_operator_libint,
    solute_potential_at_tesserae, solute_potential_at_tesserae_libint,
};
use ferric_pcm::{build_cavity, CavityConfig, Tessera};
use ndarray::Array2;

const KERNEL_BAR: f64 = 1e-12;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }
}

fn water() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap()
}

fn symmetric_density(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = Lcg(seed);
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let v = rng.next() * 0.4 + if i == j { 0.6 } else { 0.0 };
            d[(i, j)] = v;
            d[(j, i)] = v;
        }
    }
    d
}

fn charges(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..n).map(|_| rng.next() * 0.1).collect()
}

fn a_scale(kernel: &Md3c1e, tess: &[Tessera]) -> f64 {
    let pts: Vec<[f64; 3]> = tess.iter().map(|t| t.position).collect();
    let mut scr = kernel.scratch();
    let batch = kernel
        .a_matrices(&pts, None, CosxScreen::none(), &mut scr)
        .unwrap();
    batch.a.iter().fold(1.0_f64, |m, x| m.max(x.abs()))
}

fn has_l(prep: &PreparedBasis, l: i32) -> bool {
    prep.located_shells().iter().any(|s| s.l == l)
}

fn check(basis_name: &str, need_l: &[i32]) {
    let mol = water();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    for &l in need_l {
        assert!(has_l(&prep, l), "{basis_name} has no l={l} shell: vacuous");
    }
    assert!(prep.nshells() > 1, "no off-diagonal shell pair");
    let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
    let n = prep.nbasis();
    let d = symmetric_density(n, 7);
    let q = charges(tess.len(), 11);
    let kernel = Md3c1e::new(&prep).unwrap();
    let scale = a_scale(&kernel, &tess);

    let v_md = solute_potential_at_tesserae(&mol, &prep, &d, &tess).unwrap();
    let v_lb = solute_potential_at_tesserae_libint(&mol, &prep, &d, &tess).unwrap();
    let sum_d: f64 = d.iter().map(|x| x.abs()).sum();
    let bar_v = KERNEL_BAR * scale * sum_d;
    let dv = v_md
        .iter()
        .zip(&v_lb)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    let vmax = v_lb.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(vmax > 1e-2, "potential is trivially small: {vmax}");
    eprintln!(
        "{basis_name}: max|dv| = {dv:.3e} (bar {bar_v:.3e}, max|v| {vmax:.3}, A_scale {scale:.2})"
    );
    assert!(
        dv <= bar_v,
        "{basis_name}: potential differs by {dv:e} > {bar_v:e}"
    );

    let op_md = build_reaction_field_operator(&prep, &tess, &q).unwrap();
    let op_lb = build_reaction_field_operator_libint(&prep, &tess, &q).unwrap();
    let sum_q: f64 = q.iter().map(|x| x.abs()).sum();
    let bar_op = KERNEL_BAR * scale * sum_q;
    let dop = (&op_md - &op_lb)
        .iter()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    let opmax = op_lb.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(opmax > 1e-3, "operator is trivially small: {opmax}");
    eprintln!("{basis_name}: max|dV| = {dop:.3e} (bar {bar_op:.3e}, max|V| {opmax:.3})");
    assert!(
        dop <= bar_op,
        "{basis_name}: operator differs by {dop:e} > {bar_op:e}"
    );
    let asym = (&op_md - &op_md.t())
        .iter()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    assert_eq!(asym, 0.0, "operator is not exactly symmetric");
}

#[test]
fn kernel_matches_libint_on_water_sto3g() {
    check("sto-3g", &[0, 1]);
}

#[test]
fn kernel_matches_libint_on_water_def2_tzvp_with_d() {
    check("def2-tzvp", &[0, 1, 2]);
}

/// cc-pVTZ on water: f on O, d on H.
#[test]
fn kernel_matches_libint_on_water_cc_pvtz_with_d_and_f() {
    check("cc-pvtz", &[2, 3]);
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn results_are_bit_identical_across_pool_sizes() {
    let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
    // More than one tessera chunk, or the chunking is not exercised.
    assert!(tess.len() > 256, "only {} tesserae", tess.len());
    let d = symmetric_density(prep.nbasis(), 3);
    let q = charges(tess.len(), 5);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let v = solute_potential_at_tesserae(&mol, &prep, &d, &tess).unwrap();
            let op = build_reaction_field_operator(&prep, &tess, &q).unwrap();
            (bits(&v), bits(op.as_slice().unwrap()))
        })
    };
    let r1 = run(1);
    assert!(r1 == run(2), "pool 2 differs from pool 1");
    assert!(r1 == run(6), "pool 6 differs from pool 1");
}

/// A pure p shell is refused by the kernel, so the dispatchers take the
/// libint route and return exactly its bits.
#[test]
fn refused_basis_falls_back_to_libint() {
    let mol = water();
    let mut bs = basis::bundled("sto-3g").unwrap();
    for shells in bs.shells.values_mut() {
        for sh in shells.iter_mut() {
            if sh.l == 1 {
                sh.pure = true;
            }
        }
    }
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    assert!(
        Md3c1e::new(&prep).is_err(),
        "kernel accepted a pure p shell"
    );
    let tess = build_cavity(&mol, &CavityConfig::default()).unwrap();
    let d = symmetric_density(prep.nbasis(), 9);
    let q = charges(tess.len(), 2);
    let v = solute_potential_at_tesserae(&mol, &prep, &d, &tess).unwrap();
    let v_lb = solute_potential_at_tesserae_libint(&mol, &prep, &d, &tess).unwrap();
    assert_eq!(bits(&v), bits(&v_lb));
    let op = build_reaction_field_operator(&prep, &tess, &q).unwrap();
    let op_lb = build_reaction_field_operator_libint(&prep, &tess, &q).unwrap();
    assert_eq!(
        bits(op.as_slice().unwrap()),
        bits(op_lb.as_slice().unwrap())
    );
}
