//! The unblocked `solve_uhf` path is bit-identical to origin/main before the
//! point-group-blocking refactor (stage 3a of #294): energy bits and iteration
//! count of a doublet and a triplet, recorded from an origin/main build
//! (merge-base 236db4ed) with the same test body.
//!
//! The iteration count is pinned exactly and the energy to 1e-12 Ha: bitwise
//! equality can move by an ulp across machines (BLAS dispatch). On the
//! recording machine the bits of the refactored and the origin/main builds
//! were identical.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use std::path::PathBuf;

fn run(rel: &str, mult: usize) -> (u64, usize) {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/molecules")
        .join(rel);
    let mol = Molecule::load_xyz_with_charge(p.to_str().unwrap(), 0, mult).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let cfg = RhfConfig {
        density_conv: 1e-10,
        energy_conv: 1e-12,
        max_iter: 300,
        ..RhfConfig::default()
    };
    let r = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg).unwrap();
    eprintln!(
        "GOLDEN {rel} bits={:#x} iters={}",
        r.energy.to_bits(),
        r.iterations
    );
    (r.energy.to_bits(), r.iterations)
}

/// (file, multiplicity, energy bits, iterations) from origin/main.
const RECORDED: [(&str, usize, u64, usize); 2] = [
    ("validation/nh2.xyz", 2, 0xc04b6b2056e5972a, 12),
    ("o2.xyz", 3, 0xc06274497ab49afa, 8),
];

#[test]
fn unblocked_uhf_matches_the_origin_main_golden() {
    let got: Vec<_> = RECORDED.iter().map(|r| run(r.0, r.1)).collect();
    for ((rel, _, bits, iters), (b, n)) in RECORDED.iter().zip(got) {
        assert_eq!(n, *iters, "{rel}: iteration count");
        let (e, e0) = (f64::from_bits(b), f64::from_bits(*bits));
        assert!((e - e0).abs() < 1e-12, "{rel}: {e} vs {e0}");
    }
}
