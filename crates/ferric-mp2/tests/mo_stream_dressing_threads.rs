//! The RI-MP2 MO stream's V^-1/2 dressing (`stream_dressed_mo_band_budgeted`)
//! runs on the rayon pool through `ferric_core::linalg::par_gemm`, and the
//! RI-MP2 energy and dressed tensor are bit-identical across pool sizes.
//!
//! Own binary: it reads the process-global `par_gemm` probe.

use ferric_core::basis;
use ferric_core::linalg::par_gemm_probe;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::sync::atomic::Ordering;

/// Water/cc-pVDZ: nocc 5, nvir 19, so the (P|ia) width is 95 and the aux
/// stream is 84 functions. Pools of 1, 2 and 6 threads must give the same
/// bits for E_OS, E_SS and B[P,ia], and the dressing must have gone through
/// `par_gemm` (one column block per 256-wide aux chunk here).
///
/// FAILS if the dressing goes back to a direct `general_mat_mul` (no
/// `par_gemm` blocks recorded) or if any step of the energy path starts to
/// depend on the thread count.
#[test]
fn mo_stream_dressing_uses_the_pool_and_is_thread_count_independent() {
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
        0,
        1,
    )
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig::default(),
    )
    .unwrap();
    let config = RiMp2Config::default();

    let run_on = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        par_gemm_probe::reset();
        let (sc, b) = pool
            .install(|| ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &config))
            .unwrap();
        let blocks = par_gemm_probe::BLOCKS.load(Ordering::Relaxed);
        (sc, b, blocks)
    };

    let (sc1, b1, blocks1) = run_on(1);
    assert!(
        blocks1 > 0,
        "the MO-stream dressing did not go through par_gemm"
    );
    for threads in [2, 6] {
        let (sc, b, blocks) = run_on(threads);
        assert_eq!(blocks, blocks1, "{threads} threads: block count changed");
        assert_eq!(
            sc.e_os.to_bits(),
            sc1.e_os.to_bits(),
            "{threads} threads: E_OS {} vs {}",
            sc.e_os,
            sc1.e_os
        );
        assert_eq!(
            sc.e_ss.to_bits(),
            sc1.e_ss.to_bits(),
            "{threads} threads: E_SS {} vs {}",
            sc.e_ss,
            sc1.e_ss
        );
        assert_eq!(b, b1, "{threads} threads: B[P,ia] differs");
    }
}
