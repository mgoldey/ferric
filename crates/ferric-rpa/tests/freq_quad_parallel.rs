//! The freq-quadrature region was the crash site (OpenBLAS-under-rayon stack
//! overflow at >1 BLAS thread). With the with_blas_threads(1) guard it must run
//! at full rayon threads without crashing AND give the same PDEP result as serial.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::PdepRpaConfig;
use ferric_rpa::run_pdep_rpa;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn co_rpa_energy(n_threads: usize) -> f64 {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build().unwrap();
    pool.install(|| {
        let ctx = ParallelContext::default();
        let mol = Molecule::parse_xyz("2\nCO\nC 0 0 -0.6442\nO 0 0 0.4828\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        run_pdep_rpa(&mol, &obs, &aux, op, &rhf, &PdepRpaConfig::default()).unwrap().e_rpa
    })
}

#[test]
fn freq_quad_parallel_matches_serial_and_no_crash() {
    let serial = co_rpa_energy(1);
    let parallel = co_rpa_energy(4); // would stack-overflow pre-fix if BLAS threaded
    assert!((serial - parallel).abs() < 1e-10,
        "RPA energy serial {serial} vs parallel {parallel}");
}
