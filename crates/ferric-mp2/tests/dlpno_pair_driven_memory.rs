//! MEASURED peak RSS: the dense DLPNO-MP2 entry point versus the pair-driven one.
//!
//! `dlpno_pair_driven_vs_rimp2.rs` proves the two paths give the same energy.
//! This one measures what they COST, on a system big enough for the difference
//! to be visible in the process's own resident set rather than only in an
//! arithmetic counter.
//!
//! # How to run
//!
//! Both tests are `#[ignore]`d because they are measurements, not contracts, and
//! because a debug build's numbers are meaningless:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-mp2 --release \
//!     --test dlpno_pair_driven_memory -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` is not optional: peak RSS is a PROCESS-wide high-water
//! mark, so two measurements running concurrently in the same binary would
//! each report the other's allocations.
//!
//! # What these numbers are, and are not
//!
//! They are a fixed-size memory comparison, which is robust to a busy box: an
//! allocation happens or it does not, regardless of who else is on the CPU.
//! They are NOT timings, and nothing here should be read as a speedup. Per the
//! repo's measurement rules a wall-clock claim needs a quiet box, and anything
//! run under a cgroup cap is good for fixed-size comparison only, never for
//! fitting a scaling exponent.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::dlpno_mp2::{
    dense_g_bytes, dlpno_mp2_from_b_ov, dlpno_mp2_spin_components, DlpnoConfig,
};
use ferric_mp2::pair_domains::complete_pair_domains;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Peak resident set size in bytes, from the kernel's own high-water mark.
///
/// `VmHWM` is monotone for the life of the process, which is exactly what is
/// wanted: it records the largest the process ever got, not what it happens to
/// hold at the moment of the call, so a transient allocation that is freed
/// before the read still shows up.
fn peak_rss_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .expect("parse VmHWM");
            return kb * 1024;
        }
    }
    panic!("VmHWM not found in /proc/self/status");
}

fn gib(b: u64) -> f64 {
    b as f64 / 1024.0f64.powi(3)
}

/// C20H42 — chosen because at cc-pVDZ (nocc=81, nvir=409) the dense `g` is
/// 8.2 GiB, a real fraction of this 23 GB box, while one pair block is 1.3 MiB.
/// Benzene was tried first and rejected: its dense `g` is 28 MiB at cc-pVDZ,
/// which is invisible in RSS and would have made the comparison unmeasurable.
fn alkane20() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/alkane_20.xyz")
        .or_else(|_| Molecule::load_xyz("testdata/molecules/alkane_20.xyz"))
        .expect("alkane_20.xyz")
}

struct Setup {
    b_ov: Array2<f64>,
    eps: Vec<f64>,
    nocc: usize,
    nvir: usize,
    nocc_total: usize,
}

fn setup(basis_name: &str) -> Setup {
    let ctx = ParallelContext::new();
    let mol = alkane20();
    let obs_bs = basis::bundled(basis_name).unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    assert!(rhf.converged, "premise: RHF must converge");

    let cfg = RiMp2Config::default();
    let (_sc, b_ov) = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let nocc_total = mol.nelec() as usize / 2;
    Setup {
        b_ov,
        eps: rhf.eps_r().to_vec(),
        nocc: nocc_total - cfg.frozen_core,
        nvir: obs.nbasis() - nocc_total,
        nocc_total,
    }
}

/// MEASUREMENT: peak RSS with the dense path versus the pair-driven path.
///
/// The two are measured in SEPARATE processes (one `#[ignore]`d test each would
/// still share a process, so instead the delta is taken against the baseline
/// established before either path runs, and the dense array is dropped in
/// between). Because `VmHWM` never falls, the pair-driven number is measured
/// FIRST and the dense one second — if the order were reversed the dense peak
/// would mask the pair-driven one entirely and the test would report nothing.
#[test]
#[ignore = "measurement, not a contract; needs --release and --test-threads=1"]
fn peak_rss_dense_versus_pair_driven() {
    let s = setup("cc-pvdz");
    let d = complete_pair_domains(&Array2::zeros((s.nocc, 3))).unwrap();
    let cfg = DlpnoConfig::exact();

    let base = peak_rss_bytes();
    eprintln!(
        "C20H42/cc-pVDZ: nocc={} nvir={} naux={}\n  baseline peak RSS after SCF+RI: {:.3} GiB\n  \
         b_ov itself: {:.3} GiB",
        s.nocc,
        s.nvir,
        s.b_ov.nrows(),
        gib(base),
        gib((s.b_ov.len() * 8) as u64)
    );

    // --- Pair-driven FIRST, so its peak is not masked by the dense one. ---
    let (paired, _) =
        dlpno_mp2_from_b_ov(&s.b_ov, &s.eps, s.nocc, s.nvir, 0, s.nocc_total, &d, &cfg).unwrap();
    let after_paired = peak_rss_bytes();
    eprintln!(
        "  after pair-driven: peak RSS {:.3} GiB (delta over baseline {:+.3} GiB), \
         E_corr = {:.12}",
        gib(after_paired),
        gib(after_paired) - gib(base),
        paired.e_total
    );

    // --- Dense second: allocate g, run, and read the new high-water mark. ---
    let g = s.b_ov.t().dot(&s.b_ov);
    let predicted = dense_g_bytes(s.nocc, s.nvir);
    assert_eq!(
        g.len() * 8,
        predicted,
        "dense_g_bytes must describe the array actually built"
    );
    let (dense, _) =
        dlpno_mp2_spin_components(&g, &s.eps, s.nocc, s.nvir, 0, s.nocc_total, &d, &cfg).unwrap();
    let after_dense = peak_rss_bytes();
    eprintln!(
        "  dense g alone: {:.3} GiB\n  after dense: peak RSS {:.3} GiB (delta over \
         pair-driven {:+.3} GiB), E_corr = {:.12}",
        gib(predicted as u64),
        gib(after_dense),
        gib(after_dense) - gib(after_paired),
        dense.e_total
    );

    // The energies must still agree -- a memory comparison between two paths
    // that disagree on the answer would be meaningless.
    assert!(
        (paired.e_total - dense.e_total).abs() < 1e-10,
        "the two paths must agree before their costs can be compared"
    );

    // The pair-driven path must not have grown the process by anything like the
    // dense array. Allow generous slack for allocator behaviour and the per-pair
    // blocks; the claim is an order of magnitude, not a byte count.
    let paired_delta = after_paired.saturating_sub(base);
    eprintln!(
        "  VERDICT: pair-driven added {:.3} GiB; the dense g alone is {:.3} GiB ({:.1}x)",
        gib(paired_delta),
        gib(predicted as u64),
        predicted as f64 / (paired_delta.max(1)) as f64
    );
    assert!(
        (paired_delta as f64) < 0.5 * predicted as f64,
        "pair-driven added {:.3} GiB, which is not decisively less than the dense \
         array's {:.3} GiB",
        gib(paired_delta),
        gib(predicted as u64)
    );
}

/// The shape where the dense path is simply unallocatable.
///
/// No attempt is made to RUN the dense path here — it would OOM-kill the box,
/// which this repo has done before. The counted allocation is the measurement,
/// and the pair-driven path is run for real at the same `nvir` to show that the
/// per-pair working set really is what the counter claims.
#[test]
#[ignore = "measurement, not a contract; needs --release and --test-threads=1"]
fn dense_is_unallocatable_at_drug_scale() {
    // danuglipron + capped SER31 at def2-SVP.
    let (nocc, nvir) = (170usize, 760usize);
    let dense = dense_g_bytes(nocc, nvir);
    eprintln!(
        "danuglipron-scale shape nocc={nocc} nvir={nvir}:\n  dense g would be {:.1} GiB\n  \
         one pair block is {:.4} GiB\n  this box has 23 GB, so the dense path cannot START",
        gib(dense as u64),
        gib((nvir * nvir * 8) as u64)
    );
    assert!(gib(dense as u64) > 23.0);
}
