//! The SCF's serial sections (Fock diagonalization, DIIS commutator) run at
//! the serial BLAS thread count: by default the rayon pool size, `1` when set
//! to 1, and 1 inside a rayon pool. Read through `ferric_scf::blas_probe`,
//! which records `openblas_get_num_threads()` from inside those sections.
//!
//! One test in its own binary: it changes process-global state (the rayon
//! global pool, the serial setting, the OpenBLAS count).

use ferric_core::basis;
use ferric_core::blas_threads::{current_blas_threads, set_serial_blas_threads};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::blas_probe::{DIIS_COMMUTATOR, FOCK_DIAG};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::sync::atomic::Ordering;

const POOL: usize = 4;

fn probes() -> (usize, usize) {
    (
        FOCK_DIAG.load(Ordering::Relaxed),
        DIIS_COMMUTATOR.load(Ordering::Relaxed),
    )
}

#[test]
fn scf_serial_sections_follow_the_serial_blas_setting() {
    std::env::remove_var("FERRIC_BLAS_THREADS");
    rayon::ThreadPoolBuilder::new()
        .num_threads(POOL)
        .build_global()
        .expect("this binary's first rayon use");
    assert_eq!(rayon::current_num_threads(), POOL);
    let baseline = current_blas_threads();

    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
        0,
        1,
    )
    .unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    };
    let run = || solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();

    // Default: the rayon pool size.
    set_serial_blas_threads(None);
    let threaded = run();
    assert_eq!(
        probes(),
        (POOL, POOL),
        "default serial BLAS threads must be the rayon pool size"
    );
    assert_eq!(current_blas_threads(), baseline, "baseline not restored");

    // Explicit 1 is honoured.
    set_serial_blas_threads(Some(1));
    let single = run();
    assert_eq!(probes(), (1, 1), "blas_threads = 1 must be honoured");
    set_serial_blas_threads(None);

    // Same SCF either way, up to reduction order.
    assert_eq!(threaded.iterations, single.iterations);
    let de = (threaded.energy - single.energy).abs();
    assert!(de <= 1e-10, "threaded vs single BLAS: dE = {de:e}");

    // The env knob, when nothing explicit is set.
    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let _ = run();
    std::env::remove_var("FERRIC_BLAS_THREADS");
    assert_eq!(probes(), (2, 2), "FERRIC_BLAS_THREADS must be honoured");

    // Inside a rayon pool the serial sections stay on one BLAS thread.
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| {
            let _ = run();
        });
    assert_eq!(
        probes(),
        (1, 1),
        "SCF inside a rayon pool must use 1 BLAS thread"
    );
    assert_eq!(current_blas_threads(), baseline, "baseline not restored");
}
