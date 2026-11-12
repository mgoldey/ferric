//! The CAS-CI memory pre-flight runs before the determinant strings are
//! enumerated, and charges only the RDM block buffers that can be live at
//! once.

use ferric_ci::integrals::ActiveSpaceIntegrals;
use ferric_ci::{run_cas_ci_on_integrals, CasCiConfig};
use ndarray::Array2;

/// Diagonal one-electron part and `(pp|rr) = 0.5`: a valid symmetric
/// Hamiltonian whose only role is to give the solver something to run on.
fn toy_integrals(n: usize) -> ActiveSpaceIntegrals {
    let h = Array2::from_shape_fn(
        (n, n),
        |(p, q)| {
            if p == q {
                -1.0 + 0.1 * p as f64
            } else {
                0.01
            }
        },
    );
    let mut eri = vec![0.0; n.pow(4)];
    for p in 0..n {
        for r in 0..n {
            eri[((p * n + p) * n + r) * n + r] = 0.5;
        }
    }
    ActiveSpaceIntegrals {
        n_active: n,
        h,
        eri,
        e_core: 0.0,
    }
}

/// Peak resident set of this process (`VmHWM`), in bytes.
#[cfg(target_os = "linux")]
fn peak_rss_bytes() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .expect("VmHWM in /proc/self/status");
    let kb: usize = line
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .expect("VmHWM value in kB");
    kb * 1024
}

/// CAS(30,30), (15,15): each string list holds C(30,15) = 155,117,520 `u64`s
/// (1.24 GB), so enumerating both before the budget check would raise the
/// peak RSS by about 2.5 GB. A 1 MB budget must be refused with the string
/// lists named in the plan, and the peak RSS may grow by at most 100 MB (the
/// 6.5 MB ERI array plus allocator slack; an order of magnitude below one
/// list).
#[test]
fn casci_refuses_an_oversized_space_before_enumerating_strings() {
    let ints = toy_integrals(30);
    let cfg = CasCiConfig {
        n_active: 30,
        n_elec_active: (15, 15),
        memory_budget_bytes: Some(1 << 20),
        ..Default::default()
    };
    #[cfg(target_os = "linux")]
    let before = peak_rss_bytes();
    let msg = match run_cas_ci_on_integrals(ints, &cfg) {
        Ok(_) => panic!("a 1 MB budget accepted CAS(30,30)"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("budget is"), "{msg}");
    assert!(msg.contains("N_det=24061445010950400"), "{msg}");
    // 2 x 155,117,520 x 8 B = 2.482 GB, on the plan's string-list row.
    assert!(
        msg.lines()
            .any(|l| l.contains("determinant string lists") && l.contains("2.482 GB")),
        "{msg}"
    );
    #[cfg(target_os = "linux")]
    {
        let grown = peak_rss_bytes().saturating_sub(before);
        assert!(
            grown < 100 << 20,
            "peak RSS grew by {grown} bytes before the budget refusal"
        );
    }
}

/// CAS(8,8), (4,4): 70 α and 70 β strings, `chunk_rows(70) = 4096 / 70 = 58`
/// α rows per RDM block, so ceil(70 / 58) = 2 blocks and at most two
/// `D` buffers of `64 × 58 × 70 × 8 B = 2.08 MB` are live, whatever the pool
/// size. On a 32-thread pool, charging 32 buffers would need 66.5 MB for
/// them alone.
///
/// The other reservations total about 4.6 MB (Davidson bases
/// `2 × 50 × 4900 × 8 B = 3.92 MB`, the work, sigma and excitation vectors
/// about 0.5 MB, RDM results and accumulators 0.13 MB), so the run needs
/// about 8.8 MB with two buffers. A 16 MB budget sits between that and the
/// 32-buffer total of about 71 MB.
#[test]
fn casci_rdm_scratch_is_charged_per_concurrent_block() {
    assert_eq!(ferric_ci::rdm::rdm_concurrent_blocks(70, 70, 32), 2);
    assert_eq!(ferric_ci::rdm::rdm_concurrent_blocks(70, 70, 1), 1);
    // C(16,8) = 12870 β strings: one α row per block, so the batch size
    // (64) caps the count on a wide pool.
    assert_eq!(ferric_ci::rdm::rdm_concurrent_blocks(12870, 12870, 32), 32);
    assert_eq!(ferric_ci::rdm::rdm_concurrent_blocks(12870, 12870, 128), 64);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(32)
        .build()
        .unwrap();
    let cfg = CasCiConfig {
        n_active: 8,
        n_elec_active: (4, 4),
        compute_rdms: true,
        memory_budget_bytes: Some(16 << 20),
        ..Default::default()
    };
    let res = pool.install(|| run_cas_ci_on_integrals(toy_integrals(8), &cfg));
    match res {
        Ok(r) => assert!(r.rdms.is_some()),
        Err(e) => panic!("CAS(8,8) RDMs refused under a 16 MB budget: {e}"),
    }
}
