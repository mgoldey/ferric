//! Peak-RSS probe isolating the PDEP eigensolve at aTZ-dimer aux scale.
//!
//! The full pipeline's peak RSS at small naux is dominated by the RI 3-index
//! transform, which masks the eigensolve term T12 targets. This probe builds a
//! realistic dielectric operator ε̃ = I + B diag(s²) Bᵀ (the exact structure of
//! `sternheimer::dielectric_apply`) at a chosen (naux, nov) and runs ONE
//! eigensolve, reporting the process peak RSS (`VmHWM`). Run each mode in its own
//! process so the peaks don't contaminate each other:
//!
//!   cargo run --release -p ferric-benchmarks --example lanczos_mem_probe -- old   [naux] [nov]
//!   cargo run --release -p ferric-benchmarks --example lanczos_mem_probe -- new   [naux] [nov]
//!
//! `old` = identity-seed block Lanczos (the pre-T12 production path).
//! `new` = paneled full-rank assembly (`run_lanczos_full_rank`).
//! Defaults: naux=1824, nov=13680 (benzene-dimer / aug-cc-pVTZ-RI class).

use ferric_rpa::{run_lanczos_full_rank, run_lanczos_seeded};
use ndarray::Array2;

fn vm_hwm_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("new");
    let naux: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1824);
    let nov: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(13680);

    // Deterministic pseudo-random B (naux × nov) and positive scale factors,
    // mirroring the ω=0 dielectric operator.
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / u32::MAX as f64) - 0.5
    };
    let mut b = Array2::<f64>::zeros((naux, nov));
    for v in b.iter_mut() {
        *v = next() * 0.05;
    }
    let s2: Vec<f64> = (0..nov).map(|k| 0.2 + (k as f64 % 11.0) * 0.05).collect();

    // matvec: out = V + B (diag(s²) (Bᵀ V)) — identical shape to dielectric_apply.
    let matvec = |v: &Array2<f64>| -> Array2<f64> {
        let mut y = b.t().dot(v); // (nov × m)
        for k in 0..y.nrows() {
            let sk = s2[k];
            let mut row = y.row_mut(k);
            row.mapv_inplace(|x| x * sk);
        }
        let mut out = v.to_owned();
        out = &out + &b.dot(&y);
        out
    };

    let hwm_before = vm_hwm_kb();
    let t0 = std::time::Instant::now();

    let (nev, first) = match mode {
        "old" => {
            // Pre-T12 production path: full naux-wide identity seed.
            let seed = Array2::<f64>::eye(naux);
            let max_iter = 3 * naux / naux.max(1) + 8;
            let res = run_lanczos_seeded(seed, &matvec, naux, max_iter, 1e-10).unwrap();
            (res.eigenvalues.len(), res.eigenvalues.first().copied().unwrap_or(0.0))
        }
        "new" => {
            let res = run_lanczos_full_rank(naux, nov, &matvec, naux).unwrap();
            (res.eigenvalues.len(), res.eigenvalues.first().copied().unwrap_or(0.0))
        }
        other => {
            eprintln!("unknown mode '{other}' (use 'old' or 'new')");
            std::process::exit(2);
        }
    };

    let dt = t0.elapsed().as_secs_f64();
    let hwm_after = vm_hwm_kb();
    let panel = std::env::var("FERRIC_LANCZOS_PANEL").unwrap_or_else(|_| "auto".into());
    println!(
        "mode={mode} naux={naux} nov={nov} panel={panel} nev={nev} lambda0={first:.6} \
         time={dt:.2}s VmHWM_before={:.2}GB VmHWM_after={:.2}GB",
        hwm_before as f64 / 1048576.0,
        hwm_after as f64 / 1048576.0,
    );
}
