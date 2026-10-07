#![cfg(feature = "gpu")]
//! Pure planning arithmetic for the device DF-K path and a fit table (does the
//! resident tensor plus scratch fit a given pool ceiling), checked against the
//! REAL bundled bases (shell structure only, no integrals). No device is queried:
//! the two pool ceilings below are inputs of the table, not measurements.
use ferric_core::basis;
use ferric_core::gpu::device::GpuError;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_scf::df_k_gpu::{
    chunk_plan, k_error_factor, resident_bytes, ChunkPlan, SCRATCH_BYTES_DEFAULT,
};

/// The two example pool ceilings of the table: 0.8 × 8.1e9 B (a default pool on an
/// 8 GB card) and 4.0e9 B (a card that also drives a display).
const POOL_A: usize = 6_480_000_000;
const POOL_B: usize = 4_000_000_000;

#[test]
fn chunk_plan_is_a_pure_function_of_shape_and_budget() {
    // benzene/aTZ: 256 MiB holds every aux row (38.8 MB), one chunk
    let p = chunk_plan(558, 414, 21, SCRATCH_BYTES_DEFAULT).unwrap();
    assert_eq!(
        p,
        ChunkPlan {
            chunk: 558,
            nchunks: 1
        }
    );
    // a budget of exactly 5 rows' scratch: chunk 5, ceil(558/5) chunks
    let per_aux = 8 * 21 * 414;
    let p = chunk_plan(558, 414, 21, 5 * per_aux + per_aux - 1).unwrap();
    assert_eq!(
        p,
        ChunkPlan {
            chunk: 5,
            nchunks: 112
        }
    );
    // one byte short of a single row: a typed pool refusal naming the scratch
    match chunk_plan(558, 414, 21, per_aux - 1) {
        Err(GpuError::PoolFull { label, .. }) => assert!(label.contains("scratch"), "{label}"),
        o => panic!("{o:?}"),
    }
    // total on zero sizes: no work, no error
    for (b, n, o) in [(0usize, 414usize, 21usize), (558, 0, 21), (558, 414, 0)] {
        assert_eq!(
            chunk_plan(b, n, o, 0).unwrap(),
            ChunkPlan {
                chunk: 0,
                nchunks: 0
            }
        );
    }
    // overflow is a typed refusal, never a wrap
    assert!(matches!(
        chunk_plan(1, usize::MAX / 2, 3, 1),
        Err(GpuError::Layout(_))
    ));
    assert!(matches!(
        resident_bytes(usize::MAX / 8, 2),
        Err(GpuError::Layout(_))
    ));
}

#[test]
fn k_error_factor_matches_the_formula_and_grows_with_depth() {
    let u = 2f64.powi(-53);
    let g = |k: usize| k as f64 * u / (1.0 - k as f64 * u);
    let (n, kc, nch) = (414usize, 11_718usize, 1usize);
    let want = 2.0 * g(n) + g(n) * g(n) + g(kc + nch) * (1.0 + g(n)).powi(2);
    assert!((k_error_factor(n, kc, nch) - want).abs() <= 1e-15 * want);
    assert!(k_error_factor(n, kc, 10) > k_error_factor(n, kc, 1));
    assert!(k_error_factor(n, 2 * kc, 1) > k_error_factor(n, kc, 1));
}

/// (system, basis, nbf, naux, fits the first ceiling, fits the second).
/// nbf/naux are re-derived here from the real bases, so the table cannot drift.
#[test]
fn the_fit_table_matches_the_real_basis_sizes() {
    let cases: [(&str, &str, usize, usize, bool, bool); 8] = [
        ("water", "cc-pvdz", 24, 113, true, true),
        ("nh3", "cc-pvdz", 29, 131, true, true),
        ("benzene", "def2-svp", 114, 558, true, true),
        ("benzene", "aug-cc-pvtz", 414, 558, true, true),
        ("alkane_8", "def2-tzvp", 356, 924, true, true),
        ("alkane_20", "def2-svp", 490, 2256, true, false),
        ("alkane_16", "def2-tzvp", 700, 1812, false, false),
        ("alkane_20", "def2-tzvp", 872, 2256, false, false),
    ];
    for (mol_name, bs, nbf, naux, fits0, fits1) in cases {
        let mol = Molecule::load_xyz(&format!("../../testdata/molecules/{mol_name}.xyz")).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled(bs).unwrap()).unwrap();
        let aux =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        assert_eq!((obs.nbasis(), aux.nbasis()), (nbf, naux), "{mol_name}/{bs}");
        let need = resident_bytes(naux, nbf).unwrap() + SCRATCH_BYTES_DEFAULT + 8 * nbf * nbf;
        assert_eq!(
            need <= POOL_A,
            fits0,
            "{mol_name}/{bs}: {need} B vs the first ceiling"
        );
        assert_eq!(
            need <= POOL_B,
            fits1,
            "{mol_name}/{bs}: {need} B vs the second ceiling"
        );
    }
    // the brief's expectation: benzene/aTZ is 765 MB of B, nowhere near a card
    assert_eq!(8 * 558 * 414 * 414, 765_111_744);
}

/// What a pool ceiling does for a range-separated hybrid, which holds one resident
/// dressed tensor PER FITTER (`DfK` for erfc and another for erf).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum RshFit {
    /// `2·(resident + scratch) <= pool`: both fitters on the device.
    Both,
    /// `resident + scratch <= pool < 2·(resident + scratch)`: the second declines (`PoolFull`).
    One,
    /// Not even one fitter fits: both stay on the CPU.
    Cpu,
}

fn rsh_fit(naux: usize, nbf: usize, pool: usize) -> RshFit {
    let one = resident_bytes(naux, nbf).unwrap();
    if 2 * (one + SCRATCH_BYTES_DEFAULT) <= pool {
        RshFit::Both
    } else if one + SCRATCH_BYTES_DEFAULT <= pool {
        RshFit::One
    } else {
        RshFit::Cpu
    }
}

/// The two-fitter fit rule against the real bundled bases. The ceilings are the
/// table's policy inputs; every byte count is computed from the real shell
/// structure. `resident_bytes` includes the n^2 K accumulator, so it exceeds the
/// bare `8*naux*n^2` figure by `8*n^2` (asserted below), and the rule charges the
/// second fitter the same resident size as the first. Each `DfK` holds its own
/// scratch and `C_occ`, so the rule charges two scratches; an earlier form of the
/// rule charged one, which under-counted by up to `SCRATCH_BYTES_DEFAULT` (256 MiB).
/// No row of the table changes under the two-scratch rule.
#[test]
fn the_rsh_fit_table_charges_both_fitters() {
    let cases: [(&str, &str, [RshFit; 2]); 4] = [
        ("water", "cc-pvdz", [RshFit::Both, RshFit::Both]),
        ("benzene", "aug-cc-pvtz", [RshFit::Both, RshFit::Both]),
        ("alkane_20", "def2-svp", [RshFit::One, RshFit::Cpu]),
        ("alkane_16", "def2-tzvp", [RshFit::Cpu, RshFit::Cpu]),
    ];
    for (mol_name, bs, want) in cases {
        let mol = Molecule::load_xyz(&format!("../../testdata/molecules/{mol_name}.xyz")).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled(bs).unwrap()).unwrap();
        let aux =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let (n, naux) = (obs.nbasis(), aux.nbasis());
        let one = resident_bytes(naux, n).unwrap();
        assert_eq!(one, 8 * naux * n * n + 8 * n * n, "{mol_name}: B plus K");
        let got = [rsh_fit(naux, n, POOL_A), rsh_fit(naux, n, POOL_B)];
        assert_eq!(
            got, want,
            "{mol_name}/{bs}: n {n}, naux {naux}, one {one} B"
        );
        // the single-fitter verdict must agree with the one-fitter rule
        for (pool, g) in [POOL_A, POOL_B].into_iter().zip(got) {
            let single_fits = one + SCRATCH_BYTES_DEFAULT <= pool;
            assert_eq!(
                single_fits,
                g != RshFit::Cpu,
                "{mol_name}: one-fitter verdict"
            );
        }
    }
    // benzene/aTZ: the 8*naux*n^2 figure is 765_111_744 B and two fitters fit even
    // the 4 GB pool.
    let one = resident_bytes(558, 414).unwrap();
    assert_eq!(one - 8 * 414 * 414, 765_111_744);
    assert!(2 * (one + SCRATCH_BYTES_DEFAULT) <= POOL_B);
    // alkane_20/def2-SVP: one fitter is ~4.3 GB, so exactly one fits the first pool.
    let big = resident_bytes(2256, 490).unwrap();
    assert!((4.33e9..4.34e9).contains(&(big as f64)), "{big}");
    assert!(big + SCRATCH_BYTES_DEFAULT <= POOL_A && 2 * (big + SCRATCH_BYTES_DEFAULT) > POOL_A);
}
