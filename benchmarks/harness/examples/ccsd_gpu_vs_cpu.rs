//! Within-code CCSD timing: closed-shell CCSD with the device on vs off, same
//! molecule, basis, SCF thresholds and binary, arms interleaved
//! (cpu, gpu, cpu, gpu, ...) so drift shows up instead of biasing one arm.
//! No ranking against other codes.
//!
//! One process cannot switch the device off after install, so each arm is a
//! child process of THIS binary (`FERRIC_ARM=cpu` runs with `FERRIC_GPU=off`,
//! `FERRIC_ARM=gpu` with `FERRIC_GPU=auto`); the parent interleaves them and
//! prints each child's measurement. SCF is outside the timed region. A GPU
//! child runs one untimed CCSD first (cuBLAS handle and pool allocation happen
//! lazily on the first GEMM), so the timed run excludes that one-off cost; the
//! CPU child does the same untimed run so the arms stay matched.
//! The run prints `NOT QUOTABLE: box contested` unless `/proc/pressure/cpu`
//! some avg10 is readable and <= 0.05 before, and the external-load sampler
//! (`ferric_benchmarks::quiet`, which excludes this process and its child arms)
//! finds no other process using CPU during the run.
//!
//! Matched settings are the caller's job and are printed: run with
//! `RAYON_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1`, on a quiet box
//! (`/proc/pressure/cpu` some avg10 <= 0.05). The GPU arm uses the default
//! `FERRIC_GPU_MIN_FLOPS` unless the caller sets it.
//!
//!   cargo run --release -p ferric-benchmarks --features gpu --example ccsd_gpu_vs_cpu
//!
//! Env: `FERRIC_MOLS` (comma list, default `c2h6,c4h10`), `FERRIC_OBS`
//! (default `cc-pvdz`), `FERRIC_REPEATS` (default 3).
#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("ccsd_gpu_vs_cpu needs --features gpu");
}

#[cfg(feature = "gpu")]
fn main() {
    if std::env::var("FERRIC_ARM").is_ok() {
        harness::child();
    } else {
        harness::parent();
    }
}

#[cfg(feature = "gpu")]
mod harness {
    use std::time::Instant;

    use ferric_cc::ccsd_closed_shell::ccsd_closed_shell;
    use ferric_cc::CcConfig;
    use ferric_core::basis;
    use ferric_core::gpu::{install, stats, GpuSettingsExplicit, GpuStatus};
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    /// utime + stime of this process in seconds (/proc/self/stat fields 14, 15).
    fn cpu_seconds() -> f64 {
        let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
        // Fields after the ")" that closes the comm name; utime is the 12th after it.
        let rest = s.rsplit(')').next().unwrap_or("");
        let f: Vec<&str> = rest.split_whitespace().collect();
        let tick = 100.0; // USER_HZ on Linux
        let g = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
        (g(11) + g(12)) / tick
    }

    fn psi(name: &str) -> String {
        std::fs::read_to_string(format!("/proc/pressure/{name}"))
            .unwrap_or_else(|_| "unavailable".into())
            .trim()
            .replace('\n', " | ")
    }

    struct Arm {
        wall: Vec<f64>,
        cpu: Vec<f64>,
    }

    fn summary(label: &str, a: &Arm) {
        let med = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(|x, y| x.total_cmp(y));
            s[s.len() / 2]
        };
        let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "  {label}: wall min {:.2}s median {:.2}s | cpu_s min {:.2} median {:.2}",
            min(&a.wall),
            med(&a.wall),
            min(&a.cpu),
            med(&a.cpu)
        );
    }

    fn load_and_scf(
        name: &str,
        obs_name: &str,
    ) -> (
        Molecule,
        PreparedBasis,
        PreparedBasis,
        ferric_scf::ScfResult,
    ) {
        let mut xyz = format!("testdata/molecules/validation/{name}.xyz");
        if !std::path::Path::new(&xyz).exists() {
            xyz = format!("testdata/molecules/{name}.xyz");
        }
        let mol =
            Molecule::load_xyz_with_charge(&xyz, 0, 1).unwrap_or_else(|e| panic!("{xyz}: {e:?}"));
        let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled(&format!("{obs_name}-ri")).unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig {
                max_iter: 200,
                energy_conv: 1e-11,
                density_conv: 1e-9,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(rhf.converged, "{name}: RHF must converge");
        (mol, obs, dfbs, rhf)
    }

    /// One timed CCSD in this process; prints `RESULT wall cpu_s offloaded h2d d2h E_corr`.
    pub fn child() {
        let obs_name = std::env::var("FERRIC_OBS").unwrap_or_else(|_| "cc-pvdz".into());
        let name = std::env::var("FERRIC_MOL").expect("FERRIC_MOL");
        install(GpuSettingsExplicit::default()).expect("install gpu settings");
        let arm = std::env::var("FERRIC_ARM").unwrap_or_default();
        let ready = matches!(ferric_core::gpu::status(), GpuStatus::Ready(_));
        if arm == "gpu" && !ready {
            eprintln!(
                "gpu arm but no usable device: {:?}",
                ferric_core::gpu::status()
            );
            std::process::exit(2);
        }
        let (mol, obs, dfbs, rhf) = load_and_scf(&name, &obs_name);
        let run = || {
            ccsd_closed_shell(
                &mol,
                &obs,
                &dfbs,
                Operator::coulomb(),
                &rhf,
                &CcConfig {
                    max_iter: 200,
                    energy_conv: 1e-11,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let _warm = run();
        let (s0, c0, t0) = (stats(), cpu_seconds(), Instant::now());
        let cc = run();
        let wall = t0.elapsed().as_secs_f64();
        let cpu = cpu_seconds() - c0;
        let s1 = stats();
        println!(
            "RESULT {wall} {cpu} {} {} {} {:.17e}",
            s1.gemm_offloaded - s0.gemm_offloaded,
            s1.bytes_h2d - s0.bytes_h2d,
            s1.bytes_d2h - s0.bytes_d2h,
            cc.correlation_energy
        );
    }

    pub fn parent() {
        let obs_name = std::env::var("FERRIC_OBS").unwrap_or_else(|_| "cc-pvdz".into());
        let mols = std::env::var("FERRIC_MOLS").unwrap_or_else(|_| "c2h6,c4h10".into());
        let repeats: usize = std::env::var("FERRIC_REPEATS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        println!("PSI cpu: {}", psi("cpu"));
        println!("PSI memory: {}", psi("memory"));
        let psi_before = ferric_benchmarks::quiet::psi_cpu_some_avg10();
        let sampler = ferric_benchmarks::quiet::Sampler::start();
        for k in [
            "RAYON_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "FERRIC_GPU_MIN_FLOPS",
            "CUDA_VISIBLE_DEVICES",
        ] {
            println!(
                "{k}={}",
                std::env::var(k).unwrap_or_else(|_| "unset".into())
            );
        }
        let exe = std::env::current_exe().expect("current_exe");
        for name in mols.split(',') {
            println!("\n{name}/{obs_name}");
            let mut arms = [
                Arm {
                    wall: vec![],
                    cpu: vec![],
                },
                Arm {
                    wall: vec![],
                    cpu: vec![],
                },
            ];
            let mut energies = [f64::NAN; 2];
            for _ in 0..repeats {
                for (i, label) in ["cpu", "gpu"].iter().enumerate() {
                    let out = std::process::Command::new(&exe)
                        .env("FERRIC_ARM", label)
                        .env("FERRIC_MOL", name)
                        .env("FERRIC_OBS", &obs_name)
                        .env("FERRIC_GPU", if i == 0 { "off" } else { "auto" })
                        .output()
                        .expect("spawn arm");
                    assert!(
                        out.status.success(),
                        "{label} arm failed: {}",
                        String::from_utf8_lossy(&out.stderr)
                    );
                    let text = String::from_utf8_lossy(&out.stdout);
                    let line = text
                        .lines()
                        .find(|l| l.starts_with("RESULT "))
                        .expect("RESULT line");
                    let f: Vec<&str> = line.split_whitespace().skip(1).collect();
                    let wall: f64 = f[0].parse().unwrap();
                    let cpu: f64 = f[1].parse().unwrap();
                    energies[i] = f[5].parse().unwrap();
                    println!(
                        "  {label}: wall_s {wall:.2} cpu_s {cpu:.2} cpu_s/wall_s {:.2} offloaded {} h2d {} B d2h {} B  E_corr {}",
                        cpu / wall,
                        f[2],
                        f[3],
                        f[4],
                        f[5]
                    );
                    arms[i].wall.push(wall);
                    arms[i].cpu.push(cpu);
                }
            }
            summary("cpu", &arms[0]);
            summary("gpu", &arms[1]);
            println!(
                "  |E_gpu - E_cpu| = {:.3e} Ha",
                (energies[1] - energies[0]).abs()
            );
        }
        let psi_after = ferric_benchmarks::quiet::psi_cpu_some_avg10();
        let summary = sampler.finish();
        if !ferric_benchmarks::quiet::print_report(psi_before, psi_after, &summary) {
            println!("NOT QUOTABLE: box contested");
        }
    }
}
