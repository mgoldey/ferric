//! MPI rank stripe of the integral-direct LMP2 pair fits
//! (`DirectConfig::mpi`), checked under a real `mpirun`.
//!
//! Each rank runs the serial driver and the striped driver in the same
//! process and compares them, then the ranks compare with each other. The
//! reference SCF is solved serially on every rank (a rank-pinned context has
//! no MPI world), so the only distributed step is the one under test.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-mp2 --features mpi \
//!     --test mpi_lmp2_direct --no-run
//! mpirun -np 2 <bin> --nocapture --test-threads=1
//! ```
//!
//! ferric never calls `MPI_Finalize`, so `mpirun` exits non-zero even when
//! every rank's tests pass; read the `test result:` lines.
//!
//! What a run at `-np 1` shows: the striped path with one rank is the serial
//! path (the stripe is off at `size == 1`). The cross-rank and
//! serial-vs-striped checks only mean something at `-np >= 2`.

#![cfg(feature = "mpi")]

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::AmplitudeLmp2Config;
use ferric_mp2::lmp2_direct::{amplitude_lmp2_direct, DirectConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Largest and smallest `value` over the ranks.
fn rank_extremes(ctx: &ParallelContext, value: f64) -> (f64, f64) {
    use mpi::collective::SystemOperation;
    use mpi::traits::CommunicatorCollectives;
    let (mut hi, mut lo) = (value, value);
    if let Some(world) = ctx.world() {
        world.all_reduce_into(
            std::slice::from_ref(&value),
            std::slice::from_mut(&mut hi),
            SystemOperation::max(),
        );
        world.all_reduce_into(
            std::slice::from_ref(&value),
            std::slice::from_mut(&mut lo),
            SystemOperation::min(),
        );
    }
    (hi, lo)
}

/// Octane / 6-31G / cc-pVDZ-RI at eps = 1e-4 with the production maps: long
/// enough that the pair domains differ, so there are several domain groups
/// to stripe.
#[test]
fn mpi_lmp2_direct_striped_matches_serial_octane() {
    let ctx = ParallelContext::default();
    let xyz = format!(
        "{}/../../testdata/molecules/alkane_8.xyz",
        env!("CARGO_MANIFEST_DIR")
    );
    let mol = Molecule::load_xyz(&xyz).unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    // Serial SCF on every rank: `for_rank(0, 1)` has no MPI world.
    let serial_ctx = ParallelContext::for_rank(0, 1);
    let rhf = solve_rhf(&serial_ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    assert!(rhf.converged, "rank {}: RHF must converge", ctx.rank);

    let cfg = AmplitudeLmp2Config {
        eps: 1e-4,
        frozen_core: 8,
        ..Default::default()
    };
    let serial_maps = DirectConfig::production();
    let striped_maps = DirectConfig {
        mpi: Some(ctx),
        ..DirectConfig::production()
    };
    let (serial, st_serial) =
        amplitude_lmp2_direct(&mol, &obs, &obs_bs, &dfbs, op, &rhf, &cfg, &serial_maps).unwrap();
    let (striped, st_striped) =
        amplitude_lmp2_direct(&mol, &obs, &obs_bs, &dfbs, op, &rhf, &cfg, &striped_maps).unwrap();

    let diff = (striped.e_corr - serial.e_corr).abs();
    eprintln!(
        "[octane] rank {}/{}: serial e_corr = {:.15} ({} groups)  striped e_corr = {:.15} \
         ({} groups on this rank)  diff = {diff:.3e}",
        ctx.rank,
        ctx.size,
        serial.e_corr,
        st_serial.n_pair_groups,
        striped.e_corr,
        st_striped.n_pair_groups,
    );
    assert!(
        serial.e_corr < -0.1,
        "a real correlation energy, got {}",
        serial.e_corr
    );
    // The gathered blocks are the serial blocks (up to the sign of a zero),
    // and every rank then runs the same solve.
    assert!(
        diff <= 1e-12,
        "rank {}: striped LMP2 differs from serial by {diff:.3e} Ha",
        ctx.rank
    );
    if ctx.size > 1 {
        // The stripe was on: this rank fitted a strict share of the groups.
        assert!(
            st_striped.n_pair_groups < st_serial.n_pair_groups,
            "rank {}: fitted {} of {} groups, so nothing was striped",
            ctx.rank,
            st_striped.n_pair_groups,
            st_serial.n_pair_groups
        );
    } else {
        assert_eq!(st_striped.n_pair_groups, st_serial.n_pair_groups);
        assert_eq!(striped.e_corr.to_bits(), serial.e_corr.to_bits());
    }

    let (hi, lo) = rank_extremes(&ctx, striped.e_corr);
    if ctx.is_root() {
        eprintln!(
            "[octane] cross-rank spread over {} ranks: {:.3e}",
            ctx.size,
            hi - lo
        );
    }
    assert!(
        hi - lo <= 1e-12,
        "ranks disagree on the striped energy by {:.3e} Ha",
        hi - lo
    );
    // Every rank's groups together are the serial groups.
    let (mut total, mine) = (0.0f64, st_striped.n_pair_groups as f64);
    if let Some(world) = ctx.world() {
        use mpi::traits::CommunicatorCollectives;
        world.all_reduce_into(
            std::slice::from_ref(&mine),
            std::slice::from_mut(&mut total),
            mpi::collective::SystemOperation::sum(),
        );
        assert_eq!(total as usize, st_serial.n_pair_groups);
    }
}
