//! Isolated RI-J build timing: the host path and the device-resident path on the
//! SAME density and ONE `DfJ` (the `FORCE_HOST` seam of `ferric_scf::df_j_gpu`
//! toggles host/device per build). Gates nothing; a slower device arm is
//! reported in the same table.
//!
//! Sections:
//!  * COLD device build: tensor upload + first J, with the `stats()` deltas
//!    against the expected `8*naux*pair + 8*(pair+naux)` H2D bytes
//!    (`pair = n(n+1)/2`).
//!  * WARM builds interleaved `cpu, device` x `FERRIC_DFJ_REPS` (7): min and
//!    median per arm, `cpu_s/wall_s` (this process, `/proc/self/stat`), the
//!    counter deltas, bytes per warm device build vs `expected 8*(pair+naux)` each
//!    way, and `max|dJ|/max|J|` of the device J against the host J.
//!
//! The density is a fixed symmetric pattern: a J build's cost does not depend on
//! the density values. The storage tier is the packed in-core tier (budget midway
//! between the unpacked and packed footprints) unless `FERRIC_DFJ_BUDGET_GB` says
//! otherwise.
//!
//! Env: `FERRIC_MOL` (alkane_8), `FERRIC_OBS` (def2-svp), `FERRIC_AUX`
//! (def2-universal-jkfit), `FERRIC_GPU_MEM_GB` (pool, unset = default),
//! `FERRIC_DFJ_REPS` (7), `FERRIC_DFJ_SMOKE=1` (1 rep, no PSI wait: a wiring
//! check, not a measurement), `FERRIC_PSI_SETTLE_SECS` (180). Quotable only if
//! PSI before <= 0.05 and the external-load sampler (`ferric_benchmarks::quiet`,
//! covering the timed region) is clean; otherwise `NOT QUOTABLE: box contested`
//! is printed. Run nothing alongside.
//!
//!   flock /tmp/ferric-gpu.lock env CUDA_VISIBLE_DEVICES=0 FERRIC_GPU_MEM_GB=7 \
//!     FERRIC_MOL=alkane_20 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=6 \
//!     cargo run --release -p ferric-benchmarks --features gpu --example dfj_device_split
//!
//! Without `--features gpu` this builds to a stub that says so.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("dfj_device_split needs --features gpu");
}

#[cfg(feature = "gpu")]
mod run {
    use ferric_benchmarks::quiet;
    use ferric_core::basis;
    use ferric_core::gpu::{
        install, pool, stats, status, GpuMode, GpuSettingsExplicit, GpuStatsSnapshot, GpuStatus,
    };
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::df_j::DfJ;
    use ferric_scf::df_j_gpu::FORCE_HOST;
    use ferric_scf::fock::JBuilder;
    use ndarray::Array2;
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    fn env_or(name: &str, default: &str) -> String {
        std::env::var(name).unwrap_or_else(|_| default.into())
    }

    fn process_cpu_seconds() -> f64 {
        std::fs::read_to_string("/proc/self/stat")
            .ok()
            .and_then(|s| quiet::parse_stat(&s))
            .map(|p| p.ticks as f64 / quiet::CLK_TCK)
            .unwrap_or(f64::NAN)
    }

    fn min_median(v: &[f64]) -> (f64, f64) {
        let mut v = v.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        (v[0], v[v.len() / 2])
    }

    /// Poll PSI every 2 s until within the quotable ceiling or `max_secs` elapse.
    fn settle_psi(max_secs: u64) -> f64 {
        let t = Instant::now();
        loop {
            let psi = quiet::psi_cpu_some_avg10();
            if (psi.is_finite() && psi <= quiet::MAX_PSI_BEFORE)
                || t.elapsed().as_secs() >= max_secs
            {
                return psi;
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }

    #[derive(Default, Clone, Copy)]
    struct Delta {
        h2d: u64,
        d2h: u64,
        device_builds: u64,
        uploads: u64,
        declined: u64,
    }

    impl Delta {
        fn between(a: &GpuStatsSnapshot, b: &GpuStatsSnapshot) -> Self {
            Self {
                h2d: b.bytes_h2d - a.bytes_h2d,
                d2h: b.bytes_d2h - a.bytes_d2h,
                device_builds: b.dfj_device_builds - a.dfj_device_builds,
                uploads: b.resident_uploads - a.resident_uploads,
                declined: b.dfj_declined - a.dfj_declined,
            }
        }

        fn add(&mut self, o: Self) {
            self.h2d += o.h2d;
            self.d2h += o.d2h;
            self.device_builds += o.device_builds;
            self.uploads += o.uploads;
            self.declined += o.declined;
        }

        fn line(&self) -> String {
            format!(
                "dfj_device_builds={} resident_uploads={} h2d={} d2h={} dfj_declined={}",
                self.device_builds, self.uploads, self.h2d, self.d2h, self.declined
            )
        }
    }

    #[derive(Default)]
    struct Arm {
        wall: Vec<f64>,
        cpu_s: f64,
        delta: Delta,
        max_rel_vs_cpu: f64,
    }

    fn timed(f: impl FnOnce()) -> (f64, f64, Delta) {
        let (s0, c0, t) = (stats(), process_cpu_seconds(), Instant::now());
        f();
        let wall = t.elapsed().as_secs_f64();
        (
            wall,
            process_cpu_seconds() - c0,
            Delta::between(&s0, &stats()),
        )
    }

    impl Arm {
        fn record(&mut self, r: (f64, f64, Delta)) {
            self.wall.push(r.0);
            self.cpu_s += r.1;
            self.delta.add(r.2);
        }

        fn print(&self, name: &str) {
            if self.wall.is_empty() {
                return;
            }
            let (mn, md) = min_median(&self.wall);
            let wall: f64 = self.wall.iter().sum();
            println!(
                "  {name:<7} min {:9.2} ms  median {:9.2} ms  cpu_s/wall_s {:5.2}  max|dJ|/max|J| vs cpu {:.2e}",
                mn * 1e3,
                md * 1e3,
                self.cpu_s / wall,
                self.max_rel_vs_cpu
            );
            println!("          {}", self.delta.line());
        }
    }

    fn max_rel(j: &Array2<f64>, reference: &Array2<f64>) -> f64 {
        let scale = reference.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        let diff = (j - reference).iter().fold(0.0f64, |a, b| a.max(b.abs()));
        diff / scale
    }

    fn setup_gpu() -> bool {
        let memory_gb = std::env::var("FERRIC_GPU_MEM_GB")
            .ok()
            .map(|s| s.parse::<f64>().expect("FERRIC_GPU_MEM_GB"));
        install(GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            memory_gb,
            ..Default::default()
        })
        .expect("install GPU settings");
        let GpuStatus::Ready(info) = status() else {
            println!("no usable GPU: {:?}", status());
            return false;
        };
        println!(
            "device {} ({}); pool {:.2} GB; RAYON_NUM_THREADS={} OPENBLAS_NUM_THREADS={}",
            info.ordinal,
            info.name,
            pool().map_or(f64::NAN, |p| p.capacity_bytes() as f64 / 1e9),
            env_or("RAYON_NUM_THREADS", "(unset)"),
            env_or("OPENBLAS_NUM_THREADS", "(unset)"),
        );
        true
    }

    pub fn main() {
        let smoke = std::env::var_os("FERRIC_DFJ_SMOKE").is_some();
        let mol_name = env_or("FERRIC_MOL", "alkane_8");
        let obs_name = env_or("FERRIC_OBS", "def2-svp");
        let aux_name = env_or("FERRIC_AUX", "def2-universal-jkfit");
        let reps: usize = if smoke {
            1
        } else {
            env_or("FERRIC_DFJ_REPS", "7")
                .parse()
                .expect("FERRIC_DFJ_REPS")
        };
        if !setup_gpu() {
            return;
        }
        println!("{mol_name}/{obs_name} aux {aux_name}; reps {reps}");
        let mol = Molecule::load_xyz(&format!("testdata/molecules/{mol_name}.xyz")).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled(&obs_name).unwrap()).unwrap();
        let auxp = PreparedBasis::new(&mol, &basis::bundled(&aux_name).unwrap()).unwrap();
        let (n, naux) = (obs.nbasis(), auxp.nbasis());
        let pair = n * (n + 1) / 2;
        let budget = match std::env::var("FERRIC_DFJ_BUDGET_GB") {
            Ok(s) => (s.parse::<f64>().expect("FERRIC_DFJ_BUDGET_GB") * 1e9) as usize,
            Err(_) => (naux * n * n * 8 + naux * pair * 8) / 2,
        };
        println!(
            "n = {n} naux = {naux} pair = {pair}; packed B = {:.3} GB; host budget {:.3} GB",
            8.0 * (naux * pair) as f64 / 1e9,
            budget as f64 / 1e9
        );
        let d = Array2::from_shape_fn((n, n), |(i, j)| {
            let (a, b) = (i.min(j), i.max(j));
            0.01 * ((a * 7 + b * 3) % 11) as f64
        });
        let mut dfj = DfJ::new(Operator::coulomb(), &obs, &auxp, budget).expect("DfJ");
        let settle = if smoke {
            0
        } else {
            env_or("FERRIC_PSI_SETTLE_SECS", "180")
                .parse()
                .expect("settle")
        };
        let psi_before = settle_psi(settle);
        println!(
            "PSI cpu some avg10 before = {psi_before:.2} (must be <= 0.05 for a quotable run; waited up to {settle} s)"
        );
        let sampler = quiet::Sampler::start();

        let mut j = Array2::zeros((n, n));
        FORCE_HOST.store(false, Ordering::Relaxed);
        let (wall, _, cold) = timed(|| {
            dfj.build(&d, &mut j).expect("cold build");
        });
        let expect = 8 * (naux * pair + pair + naux);
        println!(
            "COLD device build (upload + first J): {:.2} ms\n  {}\n  expected H2D 8*naux*pair + 8*(pair+naux) = {expect}; measured {} ({})",
            wall * 1e3,
            cold.line(),
            cold.h2d,
            if cold.h2d as usize == expect {
                "match"
            } else {
                "MISMATCH"
            }
        );
        if cold.device_builds == 0 {
            println!("  DEVICE NOT USED on the cold build (declined); the device arm below is the CPU path");
        }

        let (mut cpu, mut dev) = (Arm::default(), Arm::default());
        let (mut j_cpu, mut j_dev) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        println!("warm builds, arm order per rep: cpu, device (interleaved), {reps} reps");
        for _ in 0..reps {
            FORCE_HOST.store(true, Ordering::Relaxed);
            cpu.record(timed(|| {
                dfj.build(&d, &mut j_cpu).expect("cpu build");
            }));
            FORCE_HOST.store(false, Ordering::Relaxed);
            dev.record(timed(|| {
                dfj.build(&d, &mut j_dev).expect("device build");
            }));
            dev.max_rel_vs_cpu = dev.max_rel_vs_cpu.max(max_rel(&j_dev, &j_cpu));
        }
        FORCE_HOST.store(false, Ordering::Relaxed);
        cpu.print("cpu");
        dev.print("device");
        if let (Some(h), Some(d2)) = (
            dev.delta.h2d.checked_div(dev.delta.device_builds),
            dev.delta.d2h.checked_div(dev.delta.device_builds),
        ) {
            let per = 8 * (pair + naux);
            println!(
                "bytes per warm device build: h2d {h}, d2h {d2}; expected 8*(pair+naux) = {per} each ({})",
                if h as usize == per && d2 as usize == per {
                    "match"
                } else {
                    "MISMATCH"
                }
            );
        }
        let psi_after = quiet::psi_cpu_some_avg10();
        let summary = sampler.finish();
        if !quiet::print_report(psi_before, psi_after, &summary) {
            println!("NOT QUOTABLE: box contested");
        }
    }
}

#[cfg(feature = "gpu")]
fn main() {
    run::main();
}
