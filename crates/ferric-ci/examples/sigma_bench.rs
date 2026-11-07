//! Timing harness: direct (string-driven) vs naive (pair-loop) CAS-CI sigma.
//!
//! N2 (r = 1.0977 Å, `testdata/molecules/n2.xyz`) / cc-pVDZ, valence CAS
//! with `active_start = (14 − n_el) / 2`.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-ci --example sigma_bench -- \
//!     <sigma-new|sigma-old|casci-new|casci-old> <n_el> <n_orb> [reps]
//! ```
//!
//! `sigma-*` times `reps` sigma calls on one random vector (the direct
//! operator's list build is timed separately). `casci-new` times
//! `run_cas_ci`; `casci-old` runs the same Davidson with `sigma_naive` as the
//! matvec. Each line reports the rayon thread count and the process VmHWM
//! (peak RSS, includes the RHF and the integral transform).

use ferric_ci::hamiltonian::{hamiltonian_diagonal, sigma_naive, DeterminantSpace};
use ferric_ci::integrals::build_active_space_integrals;
use ferric_ci::sigma::DirectSigma;
use ferric_ci::strings::enumerate_strings;
use ferric_ci::{davidson, run_cas_ci, CasCiConfig};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

fn vm_hwm_mb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<f64>().ok())
        .map_or(f64::NAN, |kb| kb / 1024.0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: sigma_bench <sigma-new|sigma-old|casci-new|casci-old> <n_el> <n_orb> [reps]"
        );
        std::process::exit(2);
    }
    let mode = args[1].as_str();
    let n_el: usize = args[2].parse().expect("n_el");
    let n_orb: usize = args[3].parse().expect("n_orb");
    let reps: usize = args.get(4).map_or(1, |r| r.parse().expect("reps"));
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/n2.xyz"
    );
    let mol = Molecule::load_xyz(path).expect("n2.xyz");
    let bs = ferric_core::basis::bundled("cc-pvdz").expect("cc-pvdz");
    let prep = PreparedBasis::new(&mol, &bs).expect("basis");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &prep,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-9,
            ..Default::default()
        },
    )
    .expect("rhf");
    let start = (14 - n_el) / 2;
    let (na, nb) = (n_el / 2, n_el - n_el / 2);
    let threads = rayon::current_num_threads();
    let ndet = enumerate_strings(n_orb, na).len() * enumerate_strings(n_orb, nb).len();
    let tag = format!("mode={mode} CAS({n_el},{n_orb}) N_det={ndet} rayon_threads={threads}");

    match mode {
        "sigma-new" | "sigma-old" => {
            let ints =
                build_active_space_integrals(&mol, &prep, &rhf, start, n_orb, None).expect("ints");
            let space = DeterminantSpace {
                alpha_strings: enumerate_strings(n_orb, na),
                beta_strings: enumerate_strings(n_orb, nb),
            };
            let mut x = 0x9E37_79B9_7F4A_7C15u64;
            let c: Vec<f64> = (0..ndet)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                })
                .collect();
            let t0 = Instant::now();
            let direct = (mode == "sigma-new").then(|| DirectSigma::new(&ints, &space));
            let build = t0.elapsed().as_secs_f64();
            let mut times = Vec::new();
            let mut checksum = 0.0;
            for _ in 0..reps {
                let t = Instant::now();
                let s = match &direct {
                    Some(d) => d.apply(&c),
                    None => sigma_naive(&ints, &space, &c),
                };
                times.push(t.elapsed().as_secs_f64());
                checksum = s.iter().zip(&c).map(|(a, b)| a * b).sum::<f64>();
            }
            println!(
                "{tag} list_build_s={build:.6} sigma_call_s={times:?} c.Hc={checksum:.12} vmhwm_mb={:.1}",
                vm_hwm_mb()
            );
        }
        "casci-new" => {
            let cfg = CasCiConfig {
                n_active: n_orb,
                n_elec_active: (na, nb),
                active_start: start,
                ..Default::default()
            };
            let t = Instant::now();
            let res = run_cas_ci(&mol, &prep, &rhf, cfg).expect("casci");
            println!(
                "{tag} casci_wall_s={:.4} e_total={:.12} iters={} converged={} vmhwm_mb={:.1}",
                t.elapsed().as_secs_f64(),
                res.e_total,
                res.iterations,
                res.converged,
                vm_hwm_mb()
            );
        }
        "casci-old" => {
            let t = Instant::now();
            let ints =
                build_active_space_integrals(&mol, &prep, &rhf, start, n_orb, None).expect("ints");
            let space = DeterminantSpace {
                alpha_strings: enumerate_strings(n_orb, na),
                beta_strings: enumerate_strings(n_orb, nb),
            };
            let diag = hamiltonian_diagonal(&ints, &space);
            let cfg = CasCiConfig::default();
            let dav = davidson::davidson_lowest(
                ndet,
                |c: &[f64]| sigma_naive(&ints, &space, c),
                &diag,
                cfg.conv_thresh,
                cfg.max_subspace,
                cfg.max_iter,
            )
            .expect("davidson");
            println!(
                "{tag} casci_wall_s={:.4} e_total={:.12} iters={} converged={} vmhwm_mb={:.1}",
                t.elapsed().as_secs_f64(),
                dav.eigenvalue,
                dav.iterations,
                dav.converged,
                vm_hwm_mb()
            );
        }
        other => panic!("unknown mode {other}"),
    }
}
