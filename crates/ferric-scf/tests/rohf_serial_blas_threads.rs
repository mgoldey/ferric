//! The ROHF/ROKS per-iteration linear algebra (Roothaan effective Fock, MO
//! Focks and DIIS error, overlap orthogonalizer) and the virtual-block level
//! shift of RHF, UHF and ROHF run at the serial BLAS thread count, like the
//! Fock diagonalization and the RHF DIIS commutator. Read through
//! `ferric_scf::blas_probe`, which records `openblas_get_num_threads()` from
//! inside each wrapped scope.
//!
//! Anchors:
//!  * the probe of every site equals the pool size by default and 1 when
//!    `blas_threads = 1` (a site that is not wrapped reads the baseline, 1,
//!    so the default-count assertion fails when one wrap is removed);
//!  * with `blas_threads = 1` the energy is a fixed bit pattern run to run
//!    (wrapping at 1 thread is the baseline count: no change), compared to
//!    the pre-change build in the commit message;
//!  * at the default count the energy and iteration count agree with the
//!    single-thread run to 1e-10 Ha. Derivation: each wrapped product is a
//!    GEMM of order n (n = 12 here) whose reduction order changes with the
//!    thread count, a relative error of at most n*u ~ 12 * 1.1e-16 per
//!    entry, times ||F|| ~ 1e2 Ha, ~ 1.5e-13 per product; the SCF energy is
//!    stationary in the orbitals, so first-order input noise enters the
//!    energy at second order and the bar of 1e-10 is far above it.
//!
//! One test in its own binary: it changes process-global state (the rayon
//! global pool, the serial setting, the OpenBLAS count).

use ferric_core::basis;
use ferric_core::blas_threads::set_serial_blas_threads;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::blas_probe as p;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use std::sync::atomic::{AtomicUsize, Ordering};

const POOL: usize = 4;

fn read(slot: &AtomicUsize) -> usize {
    slot.load(Ordering::Relaxed)
}

struct Setup {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
}

fn setup(xyz: &str, mult: usize) -> Setup {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Setup { mol, prep, bounds }
}

fn cfg(xc: Option<&str>) -> RhfConfig {
    RhfConfig {
        xc: xc.map(String::from),
        level_shift: 0.3,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    }
}

#[test]
fn rohf_and_level_shift_sites_follow_the_serial_blas_setting() {
    std::env::remove_var("FERRIC_BLAS_THREADS");
    rayon::ThreadPoolBuilder::new()
        .num_threads(POOL)
        .build_global()
        .expect("this binary's first rayon use");

    let water = setup(
        "3\nH2O\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
        1,
    );
    let oh = setup("2\n\nO 0 0 0\nH 0 0 0.97\n", 2);
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();

    let run_all = || {
        let rhf = solve_rhf(&ctx, &water.mol, &water.prep, op, &water.bounds, &cfg(None)).unwrap();
        let uhf = solve_uhf(&ctx, &oh.mol, &oh.prep, &oh.bounds, &cfg(None)).unwrap();
        let rohf = solve_rohf(&ctx, &oh.mol, &oh.prep, op, &oh.bounds, &cfg(None)).unwrap();
        let roks = solve_rohf(&ctx, &oh.mol, &oh.prep, op, &oh.bounds, &cfg(Some("pbe"))).unwrap();
        [rhf, uhf, rohf, roks]
    };
    let sites = || {
        [
            read(&p::ROOTHAAN_FOCK),
            read(&p::ROHF_DIIS_ERROR),
            read(&p::ROHF_ORTHOGONALIZER),
            read(&p::RHF_LEVEL_SHIFT),
            read(&p::UHF_LEVEL_SHIFT),
            read(&p::ROHF_LEVEL_SHIFT),
        ]
    };

    set_serial_blas_threads(None);
    let threaded = run_all();
    assert_eq!(
        sites(),
        [POOL; 6],
        "default serial BLAS threads must be the rayon pool size at every site \
         (order: roothaan, diis error, orthogonalizer, rhf/uhf/rohf level shift)"
    );

    set_serial_blas_threads(Some(1));
    let single = run_all();
    assert_eq!(sites(), [1; 6], "blas_threads = 1 must be honoured");
    let single_again = run_all();
    set_serial_blas_threads(None);

    for (i, ((t, s1), s2)) in threaded.iter().zip(&single).zip(&single_again).enumerate() {
        assert!(t.converged && s1.converged, "run {i} did not converge");
        assert_eq!(
            s1.energy.to_bits(),
            s2.energy.to_bits(),
            "run {i}: blas_threads = 1 is not reproducible"
        );
        assert_eq!(t.iterations, s1.iterations, "run {i}: iteration count");
        let de = (t.energy - s1.energy).abs();
        assert!(de <= 1e-10, "run {i}: threaded vs single BLAS dE = {de:e}");
    }
}
