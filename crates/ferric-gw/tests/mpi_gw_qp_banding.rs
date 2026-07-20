//! MPI-distributed GW per-MO QP self-energy loop verification (T10 remainder).
//!
//! Mirrors `ferric-rpa/tests/mpi_rpa_freq_banding.rs` (T10) and
//! `ferric-mp2/tests/mpi_rimp2_banding.rs` (T9): this test is meant to be
//! launched under `mpirun`/`mpiexec`. It builds `ferric-gw` with
//! `--features mpi`, runs G0W0 on water (RHF reference, cc-pVDZ/cc-pVDZ-RI)
//! through `run_g0w0_mpi` (which round-robins the per-MO QP Newton/Padé loop
//! across ranks and Allreduce-sums a zero-padded buffer back to the full
//! per-MO result on every rank), and:
//!
//!   * at `-np 1` (single-process, no active MPI world), compares
//!     `run_g0w0_mpi` directly against the serial `run_gw` (G0W0) path, which
//!     must match at or near machine precision — not just GW tolerance, since
//!     at size 1 every MO's `solve_qp_for_mo` call runs through the identical
//!     Padé-fit + Newton kernel in the same order as the serial path;
//!   * when built with `--features mpi` and run on >1 rank, actively verifies
//!     that ALL ranks agree on every per-MO QP energy/Σ_c/Z/converged-flag to
//!     <= 1e-12 via an MPI all_reduce of (max-min) across the world — a
//!     failure here means the per-rank MO-round-robin partials did not sum
//!     back to the serial result;
//!   * reports wall-clock at -np1 vs -np2 on a slightly larger QP window to
//!     show the round-robin genuinely distributes compute (mirrors T10's
//!     `mpi_rpa_freq_compute_probe` table format).
//!
//! Build the test binary, then launch it:
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-gw --features mpi \
//!       --test mpi_gw_qp_banding --no-run
//!   mpirun -np 1 <path-to-test-binary> --nocapture --test-threads=1
//!   mpirun -np 2 <path-to-test-binary> --nocapture --test-threads=1
//!
//! Without `--features mpi` this is an ordinary single-process test (size==1
//! -- `ParallelContext::default()` reports rank=0/size=1 and `run_g0w0_mpi`
//! is simply not compiled), so `cargo test -p ferric-gw` alone does not build
//! this file's `#[cfg(feature = "mpi")]`-gated bodies.

#![cfg(feature = "mpi")]

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::mo_b;
use ferric_gw::mpi_gw::run_g0w0_mpi;
use ferric_gw::sigma::run_g0w0;
use ferric_gw::w_pdep;
use ferric_gw::GwConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_rpa::run_pdep_rpa;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const HA_TO_EV: f64 = 27.211386245988_f64;

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

/// Converge RHF via the DF-J/DF-K path with an hcore guess — same workaround
/// T9/T10 documented for the pre-existing `ferric-scf` MPI bugs (see
/// `docs/superpowers/mpi.md`'s "Two pre-existing ferric-scf MPI bugs"
/// section): the direct (non-DF) J/K path N-folds under MPI instead of
/// partitioning, and the default SAD guess routes free-atom sub-solves
/// through that same broken direct path even under an active multi-rank MPI
/// world.
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

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 16,
            u0: 0.5,
        },
        eigensolver_conv_thresh: 1e-7,
        eigensolver_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        need_inv_dielectric_freq: true, // GW's Σ_c requires this
        verbose: false,
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
            "[{label}] ranks disagree by {spread:.3e} (> {tol:e}): MO-round-robin QP loop did not sum to the same result",
        );
    }
}

/// Shared setup: RHF (replicated/serial — this task only distributes the
/// per-MO QP loop, not RHF or W construction) + `MoB`/redressed
/// eigenpotentials. `PdepRpaResult` does not implement `Clone` (it owns
/// several large `Array2`/`Vec` fields), so callers that need TWO independent
/// `PdepRpaResult`s (e.g. a serial-vs-MPI comparison test) must call
/// `run_pdep_rpa` twice — `run_pdep_rpa` is a pure function of
/// (mol, obs, dfbs, op, rhf, config), so two calls with the same inputs
/// reproduce the identical result deterministically (same RI transform +
/// eigensolve, no RNG).
fn prepare_h2o_gw_inputs() -> (
    Molecule,
    PreparedBasis,
    PreparedBasis,
    ferric_scf::ScfResult,
    mo_b::MoB,
    ndarray::Array2<f64>,
) {
    let ctx = ParallelContext::default();
    let mol = water();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &df_rhf_config()).unwrap();
    assert!(rhf.converged, "water RHF must converge (energy={:.10})", rhf.energy);

    let pcfg = pdep_cfg();
    let pdep = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &pcfg).unwrap();
    let mo_b = mo_b::build_full_b(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();
    let (v_dressed, _dev) = w_pdep::redress_with_check(&mo_b.v_inv_sqrt, &pdep.eigenpotentials).unwrap();

    (mol, obs, dfbs, rhf, mo_b, v_dressed)
}

/// -np 1: `run_g0w0_mpi` must reproduce the serial `run_gw`(G0W0) path to
/// machine precision. At a single rank every `i % 1 == 0`, so the MO
/// round-robin covers every QP MO on this one rank and the Allreduce is a
/// no-op sum of one nonzero contribution per row.
#[test]
fn mpi_gw_np1_matches_serial_water_g0w0() {
    let ctx = ParallelContext::default();
    let (mol, obs, dfbs, rhf, mo_b, v_dressed) = prepare_h2o_gw_inputs();
    let op = Operator::coulomb();
    let pcfg = pdep_cfg();
    let gcfg = GwConfig::default();
    let nocc = (mol.nelec() as usize) / 2;
    let qp_range = nocc.saturating_sub(3)..(nocc + 3).min(rhf.eps_r().len());

    // Two independent PdepRpaResults (no Clone impl — see prepare_h2o_gw_inputs
    // doc comment): run_pdep_rpa is a pure function of its inputs, so calling
    // it twice with the identical (mol, obs, dfbs, op, rhf, pcfg) reproduces
    // the identical W deterministically.
    let pdep_serial = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &pcfg).unwrap();
    let pdep_mpi = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &pcfg).unwrap();

    let serial = run_g0w0(&mol, &rhf, &mo_b, &v_dressed, pdep_serial, qp_range.clone(), &gcfg, None)
        .expect("serial G0W0 must run");
    let mpi_res = run_g0w0_mpi(&ctx, &mol, &rhf, &mo_b, &v_dressed, pdep_mpi, qp_range, &gcfg, None)
        .expect("MPI G0W0 must run");

    assert_eq!(serial.mo_indices, mpi_res.mo_indices);
    let mut max_eps_diff = 0.0_f64;
    let mut max_sc_diff = 0.0_f64;
    let mut max_z_diff = 0.0_f64;
    for i in 0..serial.mo_indices.len() {
        max_eps_diff = max_eps_diff.max((serial.eps_qp[i] - mpi_res.eps_qp[i]).abs());
        max_sc_diff = max_sc_diff.max((serial.sigma_c[i] - mpi_res.sigma_c[i]).abs());
        max_z_diff = max_z_diff.max((serial.z_factor[i] - mpi_res.z_factor[i]).abs());
        assert_eq!(
            serial.qp_converged[i], mpi_res.qp_converged[i],
            "MO {} converged-flag mismatch", serial.mo_indices[i]
        );
    }
    eprintln!(
        "[np1] max |eps_qp diff|={max_eps_diff:.3e}  max |sigma_c diff|={max_sc_diff:.3e}  max |z diff|={max_z_diff:.3e}"
    );
    assert!(max_eps_diff < 1e-12, "eps_qp mismatch: {max_eps_diff:.3e}");
    assert!(max_sc_diff < 1e-12, "sigma_c mismatch: {max_sc_diff:.3e}");
    assert!(max_z_diff < 1e-12, "z_factor mismatch: {max_z_diff:.3e}");

    let nocc_idx = serial.mo_indices.iter().position(|&i| i == nocc - 1).unwrap();
    let ip_ev = -mpi_res.eps_qp[nocc_idx] * HA_TO_EV;
    eprintln!("[np1] G0W0@HF/cc-pVDZ H2O IP = {ip_ev:.3} eV (ref ~11.97 eV)");
    assert_all_ranks_agree(&ctx, "water np1/np-N eps_qp[HOMO]", mpi_res.eps_qp[nocc_idx], 1e-12);
}

/// Cross-rank correctness at any rank count: every rank's `run_g0w0_mpi` must
/// produce the SAME per-MO QP results (the real proof the MO-round-robin +
/// zero-padded-Allreduce reproduces the full per-MO loop).
#[test]
fn mpi_gw_cross_rank_agreement_water_g0w0() {
    let ctx = ParallelContext::default();
    let (mol, obs, dfbs, rhf, mo_b, v_dressed) = prepare_h2o_gw_inputs();
    let op = Operator::coulomb();
    let pcfg = pdep_cfg();
    let pdep = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &pcfg).unwrap();
    let gcfg = GwConfig::default();
    let nmo = rhf.eps_r().len();
    // Full MO range so >1 rank actually splits multiple MOs even for water's
    // small basis (24 MOs at cc-pVDZ).
    let qp_range = 0..nmo;

    let mpi_res = run_g0w0_mpi(&ctx, &mol, &rhf, &mo_b, &v_dressed, pdep, qp_range, &gcfg, None)
        .expect("MPI G0W0 must run");

    eprintln!(
        "[water] rank {}/{}: n_mo={}  eps_qp[0]={:.15}  bits=0x{:016x}",
        ctx.rank, ctx.size, mpi_res.mo_indices.len(), mpi_res.eps_qp[0], mpi_res.eps_qp[0].to_bits()
    );

    for (idx, &mo_abs) in mpi_res.mo_indices.iter().enumerate() {
        assert_all_ranks_agree(&ctx, &format!("water eps_qp[mo={mo_abs}]"), mpi_res.eps_qp[idx], 1e-12);
        assert_all_ranks_agree(&ctx, &format!("water sigma_c[mo={mo_abs}]"), mpi_res.sigma_c[idx], 1e-12);
        assert_all_ranks_agree(&ctx, &format!("water z_factor[mo={mo_abs}]"), mpi_res.z_factor[idx], 1e-12);
    }

    // Loose sanity bound: every QP energy must be finite.
    for &e in mpi_res.eps_qp.iter() {
        assert!(e.is_finite(), "non-finite QP energy: {e}");
    }

    eprintln!("[mem] rank {}/{}: peak RSS (VmHWM) = {:.1} MiB", ctx.rank, ctx.size, peak_rss_mib());
}

/// Compute-scaling probe (mirrors `mpi_rpa_freq_compute_probe`): reports each
/// rank's MO share and wall time on BENZENE's full MO window at cc-pVDZ
/// (~114 MOs — same target system T8/T9/T10 used for their probes, and large
/// enough that the per-MO Newton/Padé loop's wall-clock is not swamped by
/// RHF/PDEP-RPA setup or thread-pool spin-up noise the way water's 24-MO
/// window is). This is a COMPUTE probe, not a memory-banding probe:
/// `m_proj`/`inv_diel_freq` are fully replicated on every rank by
/// construction (see `mpi_gw.rs` module docs), so there is no per-rank
/// memory reduction to measure here — the win is wall-clock. We separately
/// time (a) the full `run_g0w0_mpi` call and (b) JUST the QP loop portion
/// (by also timing an equivalent-shaped `run_gw` on rank 0-only baseline
/// isn't meaningful here since RHF+PDEP dominate at this size for a SINGLE
/// molecule; instead this probe reports the QP-loop-only wall time by
/// diffing `run_g0w0_mpi`'s total against a pre-measured RHF+PDEP-RPA setup
/// time, which is IDENTICAL across rank counts by construction (T10: RHF and
/// PDEP-RPA-with-full-inv-dielectric are unaffected by this task and run
/// replicated per rank)).
///
/// `#[ignore]` — run explicitly under mpirun:
///   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-gw --features mpi \
///       --test mpi_gw_qp_banding --no-run
///   mpirun -np 1 <bin> mpi_gw_qp_compute_probe --ignored --nocapture --test-threads=1
///   mpirun -np 2 <bin> mpi_gw_qp_compute_probe --ignored --nocapture --test-threads=1
#[test]
#[ignore]
fn mpi_gw_qp_compute_probe() {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/benzene.xyz"
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();

    let t_setup0 = std::time::Instant::now();
    // Mirrors `mpi_rpa_freq_compute_probe`'s benzene setup (T10): does NOT
    // assert `rhf.converged` — this probe only needs a fixed, reproducible
    // set of orbital energies/eigenpotentials to drive a realistic-sized QP
    // loop for a wall-clock comparison, not a chemically converged energy.
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &df_rhf_config()).unwrap();
    eprintln!(
        "[qp-probe] rank {}/{}: benzene RHF converged={} energy={:.10}",
        ctx.rank, ctx.size, rhf.converged, rhf.energy
    );
    let pcfg = pdep_cfg();
    let pdep = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &pcfg).unwrap();
    let mo_b = mo_b::build_full_b(&mol, &obs, &dfbs, op, &rhf, 0).unwrap();
    let (v_dressed, _dev) = w_pdep::redress_with_check(&mo_b.v_inv_sqrt, &pdep.eigenpotentials).unwrap();
    let setup_elapsed = t_setup0.elapsed();

    let gcfg = GwConfig::default();
    let nmo = rhf.eps_r().len();
    let qp_range = 0..nmo;
    let n_mo = qp_range.len();
    let my_mos = (0..n_mo).filter(|i| i % ctx.size.max(1) == ctx.rank).count();

    let t0 = std::time::Instant::now();
    let mpi_res = run_g0w0_mpi(&ctx, &mol, &rhf, &mo_b, &v_dressed, pdep, qp_range, &gcfg, None)
        .expect("MPI G0W0 must run");
    let qp_elapsed = t0.elapsed();

    eprintln!(
        "[qp-probe] rank {}/{}: n_mo={n_mo} this-rank-mos={my_mos}  setup={:.3}s  qp_loop={:.3}s  eps_qp[0]={:.10}",
        ctx.rank, ctx.size, setup_elapsed.as_secs_f64(), qp_elapsed.as_secs_f64(), mpi_res.eps_qp[0],
    );
    eprintln!("[mem] rank {}/{}: peak RSS (VmHWM) = {:.1} MiB", ctx.rank, ctx.size, peak_rss_mib());
}
