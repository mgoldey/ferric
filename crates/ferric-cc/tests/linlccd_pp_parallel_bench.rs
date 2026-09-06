//! Wall-clock check for the parallelized LinLCCD ladder matvec (the pp/hh
//! per-pair GEMMs, formerly a serial loop re-run every CG iteration).
//! #[ignore]d — RELEASE, quiet box:
//!   RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cc \
//!     --release --test linlccd_pp_parallel_bench -- --ignored --nocapture
//! then again without RAYON_NUM_THREADS=1 to see the scaling.
//! The energy is printed too, so a 1-thread vs N-thread pair also confirms
//! the parallel matvec is answer-identical (the anchors already pin that;
//! this is the wall-clock companion).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_cc::linlccd::LadderVariant;
use ferric_cc::linlccd_amplitude::{amplitude_linlccd_direct, AmplitudeLinLccdConfig};
use ferric_mp2::lmp2_direct::DirectConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

#[test]
#[ignore]
fn bench_linlccd_full_ladder_parallel() {
    let nthreads = rayon::current_num_threads();
    for xyz in ["alkane_4.xyz", "alkane_8.xyz"] {
        let mol = Molecule::load_xyz(&format!(
            "{}/../../testdata/molecules/{xyz}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let obs_bs = basis::bundled("6-31g").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol, &obs, op, &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() },
        )
        .unwrap();
        let nc = mol.atoms.iter().filter(|a| a.z == 6).count();
        let cfg = AmplitudeLinLccdConfig { eps: 1e-3, frozen_core: nc, ..Default::default() };
        let dcfg = DirectConfig {
            aux_radius_bohr: 10.0,
            virt_radius_bohr: Some(12.0),
            ao_tail: 1e-3,
            schwarz_skip: 1e-5,
            batch_merge: 4,
            ..Default::default()
        };
        let t0 = Instant::now();
        let (r, _st) = amplitude_linlccd_direct(
            &mol, &obs, &obs_bs, &dfbs, op, &rhf, &cfg, &dcfg, LadderVariant::Full,
        )
        .unwrap();
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "{xyz} Full/eps=1e-3 [{nthreads} rayon threads]: E_corr={:.10} \
             cg={} iters, wall={dt:.2} s",
            r.e_corr, r.cg_iterations
        );
        assert!(r.cg_converged);
    }
}
