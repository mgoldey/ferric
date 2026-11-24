//! MPI rank stripe of the integral-direct LMP2 (`DirectConfig::mpi`: strips,
//! metric, pair fits, and the PCG solve over each rank's own pair blocks),
//! checked under a real `mpirun`.
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
//!
//! The ignored `..._memory_alkane` case prints each rank's strip bytes and
//! peak resident set for the striped and the serial run. Give each rank its
//! share of the cores, e.g. `RAYON_NUM_THREADS=3 mpirun -np 2 <bin>
//! --ignored --nocapture --test-threads=1`; `FERRIC_MPI_LMP2_ALKANE` picks
//! the chain length (default 20), `FERRIC_MPI_LMP2_SCRATCH_MIB` the stage-3
//! slab scratch cap, and `FERRIC_MEM_TRACE=1` prints RSS per stage.

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

/// A `/proc/self/status` field in bytes.
fn status_bytes(field: &str) -> usize {
    let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    ferric_core::memory::sys::parse_status_kb_field(&status, field).expect(field)
}

/// Peak resident set of this process since the last [`reset_hwm`], bytes.
fn vm_hwm() -> usize {
    status_bytes("VmHWM")
}

/// Resets the high-water mark to the current resident set and returns it.
/// The two runs share a process, so the second starts on whatever the
/// allocator kept from the first (measured: ~1.7 GB after a one-rank run
/// whose solve built the gather plan); its peak is only comparable as a
/// rise above this start.
fn reset_hwm() -> usize {
    std::fs::write("/proc/self/clear_refs", "5").expect("clear_refs");
    status_bytes("VmRSS")
}

/// What one rank saw: the two runs' counters and peak resident sets.
struct Case {
    ctx: ParallelContext,
    e_serial: f64,
    e_striped: f64,
    groups_serial: usize,
    groups_striped: usize,
    strips_serial: usize,
    strips_striped: usize,
    hwm_serial: usize,
    hwm_striped: usize,
    /// Resident set when each run started (the high-water mark's reset).
    start_serial: usize,
    start_striped: usize,
}

/// [`DirectConfig::production`], with the stage-3 slab scratch cap taken from
/// `FERRIC_MPI_LMP2_SCRATCH_MIB` when it is set (a measurement knob).
fn production_maps() -> DirectConfig {
    let base = DirectConfig::production();
    match std::env::var("FERRIC_MPI_LMP2_SCRATCH_MIB") {
        Ok(v) => DirectConfig {
            scratch_budget_bytes: v.parse::<usize>().expect("scratch MiB") << 20,
            ..base
        },
        Err(_) => base,
    }
}

/// `alkane_<n>` / 6-31G / cc-pVDZ-RI at eps = 1e-4 with the production maps,
/// striped over the world and then serial, on every rank.
fn run_alkane(n: usize) -> Case {
    let ctx = ParallelContext::default();
    let xyz = format!(
        "{}/../../testdata/molecules/alkane_{n}.xyz",
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

    // The production model (Fock screen on), as the CLI runs it.
    let cfg = AmplitudeLmp2Config::production(1e-4, n, None);
    let striped_maps = DirectConfig {
        mpi: Some(ctx),
        ..production_maps()
    };
    let start_striped = reset_hwm();
    let (striped, st_striped) =
        amplitude_lmp2_direct(&mol, &obs, &obs_bs, &dfbs, op, &rhf, &cfg, &striped_maps).unwrap();
    let hwm_striped = vm_hwm();
    let start_serial = reset_hwm();
    let serial_maps = production_maps();
    let (serial, st_serial) =
        amplitude_lmp2_direct(&mol, &obs, &obs_bs, &dfbs, op, &rhf, &cfg, &serial_maps).unwrap();
    let hwm_serial = vm_hwm();
    let case = Case {
        ctx,
        e_serial: serial.e_corr,
        e_striped: striped.e_corr,
        groups_serial: st_serial.n_pair_groups,
        groups_striped: st_striped.n_pair_groups,
        strips_serial: st_serial.strip_bytes_total,
        strips_striped: st_striped.strip_bytes_total,
        hwm_serial,
        hwm_striped,
        start_serial,
        start_striped,
    };
    let mb = |b: usize| b as f64 / (1 << 20) as f64;
    for (label, r) in [("striped", &striped), ("serial", &serial)] {
        eprintln!(
            "[alkane_{n}] rank {}/{} {label} solve: {} matvec, {} iters, {:.3} s, plan {:.1} MiB, \
             resident {:.1} MiB",
            ctx.rank,
            ctx.size,
            r.matvec_path.as_str(),
            r.cg_iterations,
            r.timings.t_solve_s,
            mb(r.solve_plan_bytes),
            mb(r.solve_resident_bytes),
        );
    }
    eprintln!(
        "[alkane_{n}] rank {}/{}: e_corr serial {:.15} striped {:.15} diff {:.3e} | groups \
         {}/{} | strips {:.1}/{:.1} MiB | peak RSS {:.0}/{:.0} MiB, rise above start \
         {:.0}/{:.0} MiB (striped/serial)",
        ctx.rank,
        ctx.size,
        case.e_serial,
        case.e_striped,
        (case.e_striped - case.e_serial).abs(),
        case.groups_striped,
        case.groups_serial,
        mb(case.strips_striped),
        mb(case.strips_serial),
        mb(case.hwm_striped),
        mb(case.hwm_serial),
        mb(case.hwm_striped - case.start_striped),
        mb(case.hwm_serial - case.start_serial),
    );
    case
}

/// The checks every case must pass: striped == serial to rounding on every
/// rank, the ranks agree, and their groups together are the serial groups.
fn check(case: &Case) {
    let Case { ctx, .. } = *case;
    assert!(
        case.e_serial < -0.1,
        "a real correlation energy, got {}",
        case.e_serial
    );
    // A strip over fewer partners goes through differently shaped GEMMs, so
    // the striped blocks are the serial ones to rounding, not bit for bit.
    let diff = (case.e_striped - case.e_serial).abs();
    assert!(
        diff <= 1e-10,
        "rank {}: striped LMP2 differs from serial by {diff:.3e} Ha",
        ctx.rank
    );
    if ctx.size > 1 {
        // The stripe was on: this rank fitted a strict share of the groups
        // and built no more strip than the serial run.
        assert!(
            case.groups_striped < case.groups_serial,
            "rank {}: fitted {} of {} groups, so nothing was striped",
            ctx.rank,
            case.groups_striped,
            case.groups_serial
        );
        assert!(
            case.strips_striped <= case.strips_serial,
            "rank {}: {} strip bytes, serial {}",
            ctx.rank,
            case.strips_striped,
            case.strips_serial
        );
    } else {
        assert_eq!(case.groups_striped, case.groups_serial);
        assert_eq!(case.strips_striped, case.strips_serial);
        assert_eq!(case.e_striped.to_bits(), case.e_serial.to_bits());
    }
    let (hi, lo) = ctx.all_reduce_extremes(case.e_striped);
    if ctx.is_root() {
        eprintln!("cross-rank spread over {} ranks: {:.3e}", ctx.size, hi - lo);
    }
    assert!(
        hi - lo <= 1e-12,
        "ranks disagree on the striped energy by {:.3e} Ha",
        hi - lo
    );
    // Every rank's groups together are the serial groups.
    let mut totals = [case.groups_striped as f64];
    ctx.all_reduce_sum(&mut totals);
    if ctx.world().is_some() {
        assert_eq!(totals[0] as usize, case.groups_serial);
    }
}

/// Octane: long enough that the pair domains differ, so there are several
/// domain groups to stripe.
#[test]
fn mpi_lmp2_direct_striped_matches_serial_octane() {
    check(&run_alkane(8));
}

/// A longer chain (default C20), for the per-rank memory printout: at more
/// than one rank some rank must hold fewer strip bytes than the serial run.
#[test]
#[ignore = "minutes; run under mpirun for the per-rank memory numbers"]
fn mpi_lmp2_direct_striped_memory_alkane() {
    let n = std::env::var("FERRIC_MPI_LMP2_ALKANE")
        .ok()
        .map_or(20, |v| {
            v.parse().expect("FERRIC_MPI_LMP2_ALKANE is a chain length")
        });
    let case = run_alkane(n);
    check(&case);
    let (hi, lo) = case.ctx.all_reduce_extremes(case.strips_striped as f64);
    if case.ctx.is_root() {
        eprintln!(
            "[alkane_{n}] strip bytes over {} ranks: largest {:.3}, smallest {:.3} of serial",
            case.ctx.size,
            hi / case.strips_serial as f64,
            lo / case.strips_serial as f64
        );
    }
    if case.ctx.size > 1 {
        assert!(lo < case.strips_serial as f64, "no rank's strips shrank");
    }
}
