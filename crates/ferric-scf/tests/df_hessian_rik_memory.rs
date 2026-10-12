//! Memory pre-flight of the RI-K Hessian: the tensors it keeps resident
//! (`L^x/Z^x`, `V^x`, `K^x`, see `ferric_scf::hessian_df_k`) are charged to the
//! global pool before allocation, and a pool that cannot hold them yields a
//! typed error, not an allocation failure.
//!
//! Own test binary: the pool is process-global, so installing a tiny one must
//! not share a process with the numerical tests.
//!
//! Hypotheses. Right: with a 1 kB pool the call fails with an error naming the
//! RI-K Hessian reservation; with a pool 4x the documented peak
//! ([`rik_peak_bytes`]) it succeeds and returns the same skeleton as with no
//! pool (the reservation changes nothing numerically). Artifact: the charge is
//! too small (a 4x-sized pool still fails) or the gate never fires (the 1 kB
//! pool succeeds).

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::hessian::rij_support::{rik_terms_with_aux, RikMutants};
use ferric_scf::hessian_df_k::rik_peak_bytes;
use ndarray::Array2;

const WATER: &str = "3\nH2O\nO 0.01 -0.02 0.12\nH 0.03 0.77 -0.46\nH -0.02 -0.74 -0.49\n";

#[test]
fn an_undersized_pool_gives_a_typed_error_and_a_right_sized_one_runs() {
    if ferric_integrals::engine::libint_max_deriv_order() < 1 {
        eprintln!("SKIP: libint2 without first derivatives");
        return;
    }
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
    let aux = || PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let n = prep.nbasis();
    let x = Array2::from_shape_fn((n, 2), |(i, c)| 0.3 + 0.1 * (i + c) as f64);
    let run = || {
        rik_terms_with_aux(
            3,
            &prep,
            aux(),
            Operator::coulomb(),
            usize::MAX,
            &x,
            &RikMutants::default(),
        )
    };
    clear_global();
    let reference = run().expect("no pool: inert");

    install_global(MemoryPool::with_capacity_bytes(1024));
    let err = run().expect_err("1 kB pool must refuse").to_string();
    eprintln!("undersized pool: {err}");
    assert!(err.contains("RI-K Hessian"), "{err}");

    let need = rik_peak_bytes(3, n, 2, aux().nbasis());
    install_global(MemoryPool::with_capacity_bytes(4 * need));
    let got = run().expect("4x the documented peak must fit");
    clear_global();
    let d = got
        .skeleton
        .iter()
        .zip(reference.skeleton.iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("peak {need} bytes; skeleton difference with/without pool {d:.1e}");
    assert_eq!(d, 0.0);
}
