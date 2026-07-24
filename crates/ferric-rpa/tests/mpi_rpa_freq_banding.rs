//! MPI-distributed RPA imaginary-frequency quadrature verification (T10, RPA half).
//!
//! Mirrors `ferric-mp2/tests/mpi_rimp2_banding.rs` (T9) and
//! `ferric-scf/tests/mpi_dfjk_banding.rs` (T8) exactly: this test is meant to
//! be launched under `mpirun`/`mpiexec`. It builds `ferric-rpa` with
//! `--features mpi`, runs PDEP-RPA on water (and benzene) through
//! `run_pdep_rpa_mpi` (which round-robins the imaginary-frequency quadrature
//! points across ranks and Allreduce-sums a zero-padded buffer back to the
//! full per-frequency arrays on every rank), and:
//!
//!   * prints each rank's full-precision RPA correlation energy + raw f64
//!     bits, so a `mpirun -np 1` run vs a `mpirun -np 2` run can be compared
//!     bit-for-bit off-line;
//!   * when built with `--features mpi` and run on >1 rank, actively verifies
//!     that ALL ranks agree on the RPA energy to <= 1e-12 Ha via an MPI
//!     all_reduce of (max - min) across the world — a failure here means the
//!     per-rank frequency-round-robin partials did not sum back to the
//!     serial result;
//!   * at `-np 1` (single-process, no active MPI world), compares
//!     `run_pdep_rpa_mpi` directly against the serial `run_pdep_rpa` path,
//!     which must match to the tolerance documented below.
//!
//! Build the test binary, then launch it:
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --features mpi \
//!       --test mpi_rpa_freq_banding --no-run
//!   mpirun -np 2 <path-to-test-binary> --nocapture --test-threads=1
//!
//! Without `--features mpi` this is an ordinary single-process test (size==1
//! -- `ParallelContext::default()` reports rank=0/size=1 and `run_pdep_rpa_mpi`
//! is simply not compiled), so `cargo test -p ferric-rpa` alone does not
//! build this file's `#[cfg(feature = "mpi")]`-gated bodies.

#![cfg(feature = "mpi")]

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::PdepRpaConfig;
use ferric_rpa::mpi_rpa::run_pdep_rpa_mpi;
use ferric_rpa::run_pdep_rpa;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Current resident set size (VmRSS) of THIS process, in MiB.
fn cur_rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

/// Peak resident set size (VmHWM) of THIS process, in MiB.
fn peak_rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
        0,
        1,
    )
    .unwrap()
}

/// Converge RHF via the DF-J/DF-K path with an hcore guess — same two
/// deliberate workarounds T9 documented for the pre-existing `ferric-scf`
/// MPI bugs (see `docs/superpowers/mpi.md`'s "Two pre-existing ferric-scf MPI
/// bugs" section and `mpi_rimp2_banding.rs::df_rhf_config`): the direct
/// (non-DF) J/K path N-folds under MPI instead of partitioning (#86, not
/// fixed), and the default SAD guess routes free-atom sub-solves through that
/// same broken direct path even under an active multi-rank MPI world.
fn df_rhf_config() -> RhfConfig {
    RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        use_sad_guess: false,
        ..Default::default()
    }
}

/// RPA config with `need_inv_dielectric_freq` set — exercises BOTH
/// MPI-distributed frequency evaluators (`eval_eigenvalues_at_frequencies_mpi`
/// and `eval_inv_dielectric_matrices_mpi`), not just the RPA-energy one, since
/// GW's Σ_c consumes the latter.
fn rpa_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        need_inv_dielectric_freq: true,
        need_eigenvalues_freq: true,
        ..Default::default()
    }
}

/// Assert every rank agrees on `value` to <= tol. Trivially satisfied at -np 1.
fn assert_all_ranks_agree(ctx: &ParallelContext, label: &str, value: f64, tol: f64) {
    use mpi::collective::SystemOperation;
    use mpi::traits::CommunicatorCollectives;
    if let Some(world) = ctx.world() {
        let mut v_max = 0.0f64;
        let mut v_min = 0.0f64;
        world.all_reduce_into(std::slice::from_ref(&value), std::slice::from_mut(&mut v_max), SystemOperation::max());
        world.all_reduce_into(std::slice::from_ref(&value), std::slice::from_mut(&mut v_min), SystemOperation::min());
        let spread = v_max - v_min;
        if ctx.is_root() {
            eprintln!(
                "[{label}] cross-rank spread over {} ranks: {spread:.3e} (max={v_max:.15}, min={v_min:.15})",
                ctx.size
            );
        }
        assert!(
            spread <= tol,
            "[{label}] ranks disagree by {spread:.3e} (> {tol:e}): frequency-round-robin RPA did not sum to the same result",
        );
    }
}

/// -np 1 (or feature-off-equivalent single rank): `run_pdep_rpa_mpi` must
/// reproduce the serial `run_pdep_rpa` path. At a single rank every
/// `k % 1 == 0`, so the frequency round-robin covers every quadrature point
/// on this one rank and the Allreduce is a no-op sum of one nonzero
/// contribution per row — the eigenvalues_freq/inv_dielectric_freq/e_rpa
/// fields should match at or near machine precision, not just RPA tolerance.
#[test]
fn mpi_rpa_np1_matches_serial_water() {
    let ctx = ParallelContext::default();
    let mol = water();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &df_rhf_config()).unwrap();
    assert!(rhf.converged, "rank {}/{}: water RHF must converge (energy={:.10})", ctx.rank, ctx.size, rhf.energy);
    eprintln!("[np1] rank {}/{}: RHF energy = {:.12}", ctx.rank, ctx.size, rhf.energy);

    let cfg = rpa_cfg();
    let serial = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let mpi_res = run_pdep_rpa_mpi(&ctx, &mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();

    eprintln!(
        "[np1] rank {}/{}: serial e_rpa={:.15}  mpi e_rpa={:.15}  diff={:.3e}",
        ctx.rank, ctx.size, serial.e_rpa, mpi_res.e_rpa, (serial.e_rpa - mpi_res.e_rpa).abs()
    );
    assert!(
        (serial.e_rpa - mpi_res.e_rpa).abs() < 1e-11,
        "np1 MPI RPA must match serial to ~machine precision: serial={:.15} mpi={:.15}",
        serial.e_rpa, mpi_res.e_rpa
    );

    // eigenvalues_freq must match row-for-row (same dielectric build/eigh per ω).
    assert_eq!(serial.eigenvalues_freq.dim(), mpi_res.eigenvalues_freq.dim());
    let max_eval_diff = serial
        .eigenvalues_freq
        .iter()
        .zip(mpi_res.eigenvalues_freq.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("[np1] max |eigenvalues_freq diff| = {max_eval_diff:.3e}");
    assert!(max_eval_diff < 1e-11, "eigenvalues_freq mismatch: max diff {max_eval_diff:.3e}");

    // inv_dielectric_freq (needed by GW) must also match.
    let serial_invd = serial.inv_dielectric_freq.as_ref().expect("need_inv_dielectric_freq=true");
    let mpi_invd = mpi_res.inv_dielectric_freq.as_ref().expect("need_inv_dielectric_freq=true");
    assert_eq!(serial_invd.len(), mpi_invd.len());
    let mut max_invd_diff = 0.0_f64;
    for (a, b) in serial_invd.iter().zip(mpi_invd.iter()) {
        for (x, y) in a.iter().zip(b.iter()) {
            max_invd_diff = max_invd_diff.max((x - y).abs());
        }
    }
    eprintln!("[np1] max |inv_dielectric_freq diff| = {max_invd_diff:.3e}");
    assert!(max_invd_diff < 1e-11, "inv_dielectric_freq mismatch: max diff {max_invd_diff:.3e}");

    assert_all_ranks_agree(&ctx, "water np1/np-N e_rpa", mpi_res.e_rpa, 1e-12);
}

/// Cross-rank correctness at any rank count: every rank's `run_pdep_rpa_mpi`
/// must produce the SAME RPA correlation energy (the real proof the
/// frequency-round-robin + zero-padded-Allreduce reproduces the full
/// trace-log sum).
#[test]
fn mpi_rpa_cross_rank_agreement_water() {
    let ctx = ParallelContext::default();
    let mol = water();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &df_rhf_config()).unwrap();
    eprintln!("[water] rank {}/{}: RHF energy = {:.12} converged={}", ctx.rank, ctx.size, rhf.energy, rhf.converged);

    let cfg = rpa_cfg();
    let mpi_res = run_pdep_rpa_mpi(&ctx, &mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();

    eprintln!(
        "[water] rank {}/{}: RPA corr = {:.15} Ha  bits=0x{:016x}",
        ctx.rank, ctx.size, mpi_res.e_rpa, mpi_res.e_rpa.to_bits()
    );
    assert_all_ranks_agree(&ctx, "water", mpi_res.e_rpa, 1e-12);

    // Loose sanity bound: dRPA@PBE-quality-RHF-DF-JK water correlation energy
    // is O(-0.3 Ha); this just guards against a gross unit/sign error, not a
    // tight external reference (the serial-vs-MPI comparison above is what
    // pins exact numerical agreement).
    assert!(
        mpi_res.e_rpa < -0.1 && mpi_res.e_rpa > -0.6,
        "RPA correlation energy sanity bound: got {:.6}", mpi_res.e_rpa
    );

    eprintln!("[mem] rank {}/{}: peak RSS (VmHWM) = {:.1} MiB", ctx.rank, ctx.size, peak_rss_mib());
}

/// Benzene: bigger aux basis / more quadrature-relevant frequency points,
/// same target system T8/T9 used for their memory probes. Cross-rank
/// agreement + bits printed for off-line -np1-vs-np2 comparison.
#[test]
fn mpi_rpa_cross_rank_agreement_benzene() {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/molecules/benzene.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &df_rhf_config()).unwrap();

    let cfg = rpa_cfg();
    let mpi_res = run_pdep_rpa_mpi(&ctx, &mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();

    eprintln!(
        "[benzene] rank {}/{}: RPA corr = {:.15} Ha  bits=0x{:016x}",
        ctx.rank, ctx.size, mpi_res.e_rpa, mpi_res.e_rpa.to_bits()
    );
    assert_all_ranks_agree(&ctx, "benzene", mpi_res.e_rpa, 1e-9);

    eprintln!("[mem] rank {}/{}: peak RSS (VmHWM) = {:.1} MiB", ctx.rank, ctx.size, peak_rss_mib());
}

/// Compute-scaling probe (mirrors `mpi_rimp2_band_memory_probe` /
/// `mpi_dfk_band_memory_probe` in spirit): reports each rank's frequency
/// share and wall time on benzene, the system with enough quadrature points
/// (default n_points=20) and a big enough dielectric eigh per point to show a
/// genuine 2-rank speedup on the frequency loop. Unlike T8/T9, this is a
/// COMPUTE probe, not a memory-banding probe (see `mpi_rpa.rs` module docs:
/// the frequency loop shares the full replicated `b_ov`/`eigenvectors` on
/// every rank by construction — there is no per-rank memory reduction to
/// measure here, the win is wall-clock).
///
/// `#[ignore]` — run explicitly under mpirun:
///   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --features mpi \
///       --test mpi_rpa_freq_banding --no-run
///   mpirun -np 1 <bin> mpi_rpa_freq_compute_probe --ignored --nocapture --test-threads=1
///   mpirun -np 2 <bin> mpi_rpa_freq_compute_probe --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn mpi_rpa_freq_compute_probe() {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz(concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/molecules/benzene.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &df_rhf_config()).unwrap();

    let cfg = rpa_cfg();
    let n_quad = cfg.quadrature.n_points;
    let my_freqs = (0..n_quad).filter(|k| k % ctx.size.max(1) == ctx.rank).count();

    let rss_before = cur_rss_mib();
    let t0 = std::time::Instant::now();
    let mpi_res = run_pdep_rpa_mpi(&ctx, &mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let elapsed = t0.elapsed();
    let rss_after = cur_rss_mib();

    eprintln!(
        "[freq-probe] rank {}/{}: n_quad={n_quad} this-rank-freqs={my_freqs}  \
         wall={:.3}s  RSS_before={rss_before:.1} RSS_after={rss_after:.1} delta={:.1} MiB  \
         e_rpa={:.10}",
        ctx.rank, ctx.size, elapsed.as_secs_f64(), rss_after - rss_before, mpi_res.e_rpa,
    );
    eprintln!("[mem] rank {}/{}: peak RSS (VmHWM) = {:.1} MiB", ctx.rank, ctx.size, peak_rss_mib());
}
