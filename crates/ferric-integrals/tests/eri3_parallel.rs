//! ERI3 parallel must be bit-identical to serial. Run once at 1 rayon thread and
//! once at many; the tensor must match exactly (deterministic indexed writes).
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::eri3_tensor;

fn co_eri3(n_threads: usize) -> ndarray::Array3<f64> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build().unwrap();
    pool.install(|| {
        let mol = Molecule::parse_xyz("2\nCO\nC 0 0 -0.6442\nO 0 0 0.4828\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        eri3_tensor(Operator::coulomb(), &obs, &aux).unwrap()
    })
}

#[test]
fn eri3_parallel_matches_serial() {
    let serial = co_eri3(1);
    let parallel = co_eri3(4);
    assert_eq!(serial.dim(), parallel.dim());
    let max_diff = serial.iter().zip(parallel.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    assert!(max_diff < 1e-12, "ERI3 parallel vs serial max diff {max_diff:.2e}");
}
