//! Timing harness for the MPI integral-direct LMP2 (`DirectConfig::mpi`).
//!
//! One timed LMP2 call per rank, no serial comparison run inside the binary.
//! The reference RHF is solved serially on every rank
//! (`ParallelContext::for_rank(0, 1)`, no MPI world) before the clock starts
//! and is NOT part of any LMP2 number. Correctness of the stripe is the job of
//! `tests/mpi_lmp2_direct.rs`; this binary only checks that the ranks agree.
//!
//! ## Build
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo build --release -p ferric-mp2 \
//!     --features mpi --example mpi_lmp2_bench
//! BIN=target/release/examples/mpi_lmp2_bench
//! ```
//!
//! ## Run (always `OPENBLAS_NUM_THREADS=1`)
//!
//! Defaults: `testdata/molecules/alkane_20.xyz`, 6-31g, cc-pvdz-ri, eps 1e-4,
//! one frozen core per non-hydrogen atom, and the production model the CLI
//! runs: `AmplitudeLmp2Config::production` with `DirectConfig::production()`.
//! Flags (all optional): `--xyz PATH --basis NAME --aux NAME --eps X
//! --frozen-core N --aux-radius BOHR --virt-radius BOHR|none --ao-tail X
//! --schwarz-skip X --batch-merge N --kappa X|none --slab-mib N --scratch-mib N`
//! (the last two set `DirectConfig::slab_budget_bytes` and
//! `scratch_budget_bytes` in MiB).
//!
//! Each rank's pool has `RAYON_NUM_THREADS` threads if that is set, otherwise
//! `ParallelContext::rayon_threads()` (physical cores per local rank). Set
//! `RAYON_NUM_THREADS` explicitly when the node is not exclusively yours or
//! the launcher is not recognised, so ranks x threads never exceeds the cores.
//!
//! One node with k cores. `--bind-to none` is REQUIRED: Open MPI pins each
//! rank to one core by default, and every thread of the rank's pool then
//! shares that core (measured here: 6 threads ran at 199% CPU and the strip
//! build took 5x longer).
//!
//! ```text
//! # 1 rank x k threads (the serial path: the stripe is off at one rank)
//! RAYON_NUM_THREADS=k OPENBLAS_NUM_THREADS=1 mpirun --bind-to none -np 1 $BIN
//! # n ranks x k/n threads
//! RAYON_NUM_THREADS=$((k/n)) OPENBLAS_NUM_THREADS=1 mpirun --bind-to none -np n $BIN
//! ```
//!
//! Several nodes (one rank per node, k threads each; adapt the hostfile and
//! binding flags to the scheduler; `-x` exports the variables to the remote
//! ranks, and the binary and its `testdata` must be at the same path there):
//!
//! ```text
//! mpirun -np N --hostfile hosts --map-by ppr:1:node --bind-to none \
//!     -x RAYON_NUM_THREADS=k -x OPENBLAS_NUM_THREADS=1 $BIN
//! ```
//!
//! ferric never calls `MPI_Finalize`, so `mpirun` exits non-zero even when
//! every rank finished; judge the run by the output below, not by `$?`. The
//! binary itself exits non-zero (and says why) if the ranks' `e_corr` differ
//! by more than 1e-12 Ha.
//!
//! ## Output
//!
//! Per rank, one JSON line on stdout (everything else goes to stderr):
//! rank, size, rayon threads, `e_corr` (12 decimals), `lmp2_wall_s` (the whole
//! LMP2 call), the `DirectStats` stage timers (`t_maps_s`, `t_eri3_s` with its
//! thread-second parts, `t_metric_s`, `t_pairs_s`, `t_loc_s`), the
//! `StageTimings` (`t_spaces_s`, `t_loc_s`, `t_assembly_s`, `t_solve_s`),
//! the stage-3 slab count, largest slab and slab cap (`n_eri3_slabs`,
//! `eri3_slab_bytes_max`, `stage3_scratch_bytes`),
//! `n_pair_groups`, the stage-5 group schedule (`group_concurrency`,
//! `group_in_flight_peak`, `n_group_waves`), the resolved memory budget
//! (`memory_budget_bytes`) and `strip_bytes_total` on this rank, CG iterations, and
//! peak RSS in bytes (`VmHWM`, reset just before the call). Rank 0 also prints
//! a summary: slowest rank's wall, per-stage sums over ranks, and the
//! cross-rank energy spread.
//!
//! SEND BACK: the JSON lines of every rank (stdout), the launch line, the
//! node's core count and CPU model, and the MPI implementation and version.
//! `*_thread_s` timers are summed over a rank's workers, not wall time, and
//! the eval/scatter split is `null` (it is off in the production maps).

use std::time::Instant;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::AmplitudeLmp2Config;
use ferric_mp2::lmp2_direct::{amplitude_lmp2_direct, DirectConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use serde_json::json;

/// Largest `e_corr` disagreement between ranks accepted, Hartree.
const MAX_RANK_SPREAD: f64 = 1e-12;

/// Command-line options.
struct Args {
    xyz: String,
    basis: String,
    aux: String,
    eps: f64,
    frozen_core: Option<usize>,
    maps: DirectConfig,
}

fn parse_opt<T: std::str::FromStr>(flag: &str, v: &str) -> T {
    v.parse()
        .unwrap_or_else(|_| panic!("{flag}: cannot parse {v:?}"))
}

/// `none` is the trivial limit (`None`); anything else must parse.
fn parse_some<T: std::str::FromStr>(flag: &str, v: &str) -> Option<T> {
    (v != "none").then(|| parse_opt(flag, v))
}

fn parse_args() -> Args {
    let mut a = Args {
        xyz: format!(
            "{}/../../testdata/molecules/alkane_20.xyz",
            env!("CARGO_MANIFEST_DIR")
        ),
        basis: "6-31g".into(),
        aux: "cc-pvdz-ri".into(),
        eps: 1e-4,
        frozen_core: None,
        maps: DirectConfig::production(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let v = it
            .next()
            .unwrap_or_else(|| panic!("{flag} needs a value (see the module doc)"));
        match flag.as_str() {
            "--xyz" => a.xyz = v,
            "--basis" => a.basis = v,
            "--aux" => a.aux = v,
            "--eps" => a.eps = parse_opt(&flag, &v),
            "--frozen-core" => a.frozen_core = Some(parse_opt(&flag, &v)),
            "--aux-radius" => a.maps.aux_radius_bohr = parse_opt(&flag, &v),
            "--virt-radius" => a.maps.virt_radius_bohr = parse_some(&flag, &v),
            "--ao-tail" => a.maps.ao_tail = parse_opt(&flag, &v),
            "--schwarz-skip" => a.maps.schwarz_skip = parse_opt(&flag, &v),
            "--batch-merge" => a.maps.batch_merge = parse_opt(&flag, &v),
            "--kappa" => a.maps.virt_schwarz_kappa = parse_some(&flag, &v),
            "--slab-mib" => a.maps.slab_budget_bytes = parse_opt::<usize>(&flag, &v) << 20,
            "--scratch-mib" => a.maps.scratch_budget_bytes = parse_opt::<usize>(&flag, &v) << 20,
            _ => panic!("unknown flag {flag} (see the module doc)"),
        }
    }
    a
}

fn vm_hwm() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    ferric_core::memory::sys::parse_status_kb_field(&status, "VmHWM").expect("VmHWM")
}

/// Restart the peak resident-set counter at the current resident set.
fn reset_hwm() {
    std::fs::write("/proc/self/clear_refs", "5").expect("clear_refs");
}

/// Run `f` in this rank's pool: `RAYON_NUM_THREADS` if set, else the
/// rank-aware width of `ParallelContext::with_rank_pool`.
fn in_rank_pool<R: Send>(ctx: &ParallelContext, f: impl FnOnce() -> R + Send) -> R {
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("rayon pool")
            .install(f),
        None => ctx.with_rank_pool(f).expect("rank pool"),
    }
}

fn main() {
    // Stack size for threaded LAPACK and the OpenBLAS baseline of 1.
    ferric_core::blas_threads::init_threading();
    let args = parse_args();
    let ctx = ParallelContext::default();
    if !in_rank_pool(&ctx, || run(&ctx, &args)) {
        std::process::exit(1);
    }
}

/// Everything inside the rank pool. `false` if the ranks disagree.
fn run(ctx: &ParallelContext, args: &Args) -> bool {
    let mol = Molecule::load_xyz(&args.xyz).expect("xyz");
    let obs_bs = basis::bundled(&args.basis).expect("basis");
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("obs basis");
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(&args.aux).expect("aux")).expect("aux");
    let op = Operator::coulomb();
    let frozen_core = args
        .frozen_core
        .unwrap_or_else(|| mol.atoms.iter().filter(|at| at.z > 2).count());
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");

    // Serial SCF on every rank (`for_rank(0, 1)` has no MPI world): outside
    // the LMP2 timing, and the only distributed step left is the LMP2.
    let t_scf = Instant::now();
    let rhf = solve_rhf(
        &ParallelContext::for_rank(0, 1),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig::default(),
    )
    .expect("rhf");
    assert!(rhf.converged, "rank {}: RHF must converge", ctx.rank);
    let t_scf = t_scf.elapsed().as_secs_f64();

    // The production model, from its single source: what the CLI runs.
    let cfg = AmplitudeLmp2Config::production(args.eps, frozen_core, None);
    let maps = DirectConfig {
        mpi: Some(*ctx),
        ..args.maps.clone()
    };
    if ctx.is_root() {
        eprintln!(
            "mpi_lmp2_bench: {} {}/{} eps {:e} frozen_core {frozen_core} nbas {} naux {} ranks {}; \
             reference RHF solved serially on every rank ({t_scf:.1} s, excluded from LMP2 timings); \
             maps {:?}",
            args.xyz,
            args.basis,
            args.aux,
            args.eps,
            obs.nbasis(),
            dfbs.nbasis(),
            ctx.size,
            maps,
        );
    }

    // The budget the call resolves (auto: a fraction of available RAM),
    // read just before it.
    let budget = ferric_mp2::rimp2::eri3_budget_bytes(cfg.eri3_budget_bytes);
    reset_hwm();
    let t0 = Instant::now();
    let (res, st) =
        amplitude_lmp2_direct(&mol, &obs, &obs_bs, &dfbs, op, &rhf, &cfg, &maps).expect("lmp2");
    let wall = t0.elapsed().as_secs_f64();
    let hwm = vm_hwm();
    let tm = &res.timings;

    // NaN (an unsplit eri3 timer) is not JSON: serde_json writes null.
    let line = json!({
        "rank": ctx.rank,
        "size": ctx.size,
        "rayon_threads": rayon::current_num_threads(),
        "system": args.xyz,
        "basis": args.basis,
        "aux": args.aux,
        "eps": args.eps,
        "frozen_core": frozen_core,
        "e_corr": format!("{:.12}", res.e_corr),
        "lmp2_wall_s": wall,
        "direct": {
            "t_maps_s": st.t_maps_s,
            "t_eri3_s": st.t_eri3_s,
            "t_eri3_setup_s": st.t_eri3_setup_s,
            "n_eri3_slabs": st.n_eri3_slabs,
            "eri3_slab_bytes_max": st.eri3_slab_bytes_max,
            "stage3_scratch_bytes": st.memory.stage3_scratch_bytes,
            "t_eri3_fill_thread_s": st.t_eri3_fill_thread_s,
            "t_eri3_transform_thread_s": st.t_eri3_transform_thread_s,
            "t_eri3_eval_thread_s": st.t_eri3_eval_thread_s,
            "t_eri3_scatter_thread_s": st.t_eri3_scatter_thread_s,
            "t_metric_s": st.t_metric_s,
            "t_pairs_s": st.t_pairs_s,
            "t_loc_s": st.t_loc_s,
        },
        "stage": {
            "t_spaces_s": tm.t_spaces_s,
            "t_loc_s": tm.t_loc_s,
            "t_assembly_s": tm.t_assembly_s,
            "t_solve_s": tm.t_solve_s,
        },
        "n_pair_groups": st.n_pair_groups,
        "group_concurrency": st.group_concurrency,
        "group_in_flight_peak": st.group_in_flight_peak,
        "n_group_waves": st.n_group_waves,
        "memory_budget_bytes": budget,
        "strip_bytes_total": st.strip_bytes_total,
        "cg_iterations": res.cg_iterations,
        "peak_rss_bytes": hwm,
    });
    println!("{line}");

    // Sums over ranks of the stage walls, then the energy spread.
    let mut sums = [
        wall,
        tm.t_spaces_s,
        st.t_maps_s,
        st.t_eri3_s,
        st.t_metric_s,
        st.t_pairs_s,
        tm.t_solve_s,
    ];
    ctx.all_reduce_sum(&mut sums);
    let (wall_max, wall_min) = ctx.all_reduce_extremes(wall);
    let (e_hi, e_lo) = ctx.all_reduce_extremes(res.e_corr);
    let spread = e_hi - e_lo;
    if ctx.is_root() {
        eprintln!(
            "summary over {} ranks: slowest rank wall {wall_max:.2} s (fastest {wall_min:.2} s); \
             sum over ranks of wall: total {:.2}, spaces {:.2}, maps {:.2}, eri3 {:.2}, \
             metric {:.2}, pairs {:.2}, solve {:.2} s; e_corr {:.12}, cross-rank spread {spread:.3e}",
            ctx.size, sums[0], sums[1], sums[2], sums[3], sums[4], sums[5], sums[6], res.e_corr,
        );
    }
    if spread.is_nan() || spread > MAX_RANK_SPREAD {
        eprintln!(
            "rank {}: FAIL: ranks disagree on e_corr by {spread:.3e} Ha (limit {MAX_RANK_SPREAD:e})",
            ctx.rank
        );
        return false;
    }
    true
}
