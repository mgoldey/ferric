//! Scheduling anchor for the COSX builder's block groups.
//!
//! `CosxK` runs the md3c1e sub-batches of several grid blocks in one parallel
//! region (`ferric_scf::cosx_k::block_group_len`) and folds the blocks into
//! `Ktilde` and `S_num` serially in grid order. That is a change of schedule
//! only: every sub-batch, GEMM and scatter-add is the same operation on the
//! same inputs as in the one-block-at-a-time loop, and the folds add the same
//! matrices in the same order. So `K` must be BITWISE equal for every group
//! size and every pool size. The reference is group size 1 on a pool of 1,
//! which is the old loop operation for operation.
//!
//! Artifact hypotheses: blocks folded out of grid order, or a sub-batch
//! written into the wrong block's `G`, change `K` in the last bits (fold
//! order) or by O(|K|) (wrong block); either fails the bitwise compare. A
//! comparison that cannot fail is ruled out by the reachability asserts
//! (several blocks per build, several blocks per group).

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_scf::cosx_k::{
    block_group_len, CosxConfig, CosxHalfTransform, CosxK, COSX_BLOCK_POINTS,
};
use ferric_scf::fock::KBuilder;
use ndarray::Array2;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

/// A deterministic symmetric positive semidefinite density `C C^T` with
/// decaying entries (an LCG, no RNG crate), and its factor `C`.
fn density(n: usize, nocc: usize) -> (Array2<f64>, Array2<f64>) {
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let c = Array2::from_shape_fn((n, nocc), |(mu, i)| {
        next() / (1.0 + (mu as f64 - 2.0 * i as f64).abs())
    });
    (c.dot(&c.t()), c)
}

fn grid() -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial: 25,
        n_angular: 50,
        ..Default::default()
    }
}

/// `K` from two builds (the first factors `S_num`, the second reuses it):
/// `build(D)`, then `build_from_occ(C)`, on a pool of `threads` with
/// `group` blocks per kernel region (`None` = the default).
fn two_builds(
    mol: &Molecule,
    prep: &PreparedBasis,
    half: CosxHalfTransform,
    threads: usize,
    group: Option<usize>,
    d: &Array2<f64>,
    c: &Array2<f64>,
) -> (Vec<u64>, usize, usize) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool");
    pool.install(|| {
        let ctx = ParallelContext::default();
        let cfg = CosxConfig {
            grid: grid(),
            half_transform: half,
            ..CosxConfig::flat_reference()
        };
        let mut kb = CosxK::new(&ctx, mol, prep, cfg, usize::MAX).expect("CosxK::new");
        if let Some(g) = group {
            kb.set_block_group(g);
        }
        let n = prep.nbasis();
        let mut k1 = Array2::zeros((n, n));
        kb.build(d, &mut k1).expect("build");
        let mut k2 = Array2::zeros((n, n));
        kb.build_from_occ(c, &mut k2).expect("build_from_occ");
        let bits = k1.iter().chain(k2.iter()).map(|v| v.to_bits()).collect();
        (bits, kb.npts(), kb.block_group())
    })
}

/// K is bitwise identical across pools 1/2/6 and group sizes 1/2/3/default,
/// on both half transforms, with the overlap fit on (so the `S_num` fold is
/// exercised on the first build and the cached factor on the second).
#[test]
fn cosx_k_is_bitwise_independent_of_block_grouping_and_pool_size() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).expect("water");
    let bs = bundled("cc-pvdz").expect("cc-pvdz");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let n = prep.nbasis();
    let (d, c) = density(n, 5);
    for half in [CosxHalfTransform::Dense, CosxHalfTransform::SPARSE_DEFAULT] {
        let (reference, npts, g1) = two_builds(&mol, &prep, half, 1, Some(1), &d, &c);
        assert_eq!(g1, 1);
        let nblocks = npts.div_ceil(COSX_BLOCK_POINTS);
        // Reachability: with fewer than 4 blocks the group sizes below would
        // not all split the grid differently.
        assert!(nblocks >= 4, "{half:?}: only {nblocks} blocks ({npts} pts)");
        assert!(reference.iter().any(|&b| f64::from_bits(b) != 0.0));
        for threads in [1, 2, 6] {
            for group in [Some(1), Some(2), Some(3), None] {
                let (k, _, used) = two_builds(&mol, &prep, half, threads, group, &d, &c);
                if group.is_none() {
                    assert_eq!(used, block_group_len(threads));
                    assert!(used >= 2, "default group {used} at {threads} threads");
                }
                let first_diff = reference.iter().zip(&k).position(|(a, b)| a != b);
                assert!(
                    first_diff.is_none(),
                    "{half:?}, pool {threads}, group {used}: K differs from pool 1 / group 1 at \
                     element {first_diff:?}: {} vs {}",
                    f64::from_bits(reference[first_diff.unwrap_or(0)]),
                    f64::from_bits(k[first_diff.unwrap_or(0)]),
                );
            }
        }
    }
}
