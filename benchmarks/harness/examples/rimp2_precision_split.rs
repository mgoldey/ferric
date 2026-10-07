//! Three-arm, same-process, interleaved RI-MP2 energy-stage harness: the CPU
//! path (`spin_components_from_b_ov_kappa_cpu`, ambient 6-worker pool), the f64
//! device path and the mixed device path (`spin_components_on_device` with
//! `Precision::F64` / `Precision::Mixed`) on the SAME `B_ov` and orbital
//! energies (one RI-JK RHF, one `B_ov`), reps interleaved
//! with the arm order rotated each rep (printed per rep). Each arm gets one untimed warm-up first
//! (cuBLAS handle, pool and kernel first-use are one-off costs).
//!
//! Prints per arm: min and median wall over the reps, `cpu_s/wall_s` (this
//! process, from `/proc/self/stat`), the `stats()` deltas over the timed reps
//! (offloaded, mixed, panels, bytes H2D/D2H, fallbacks), `|dE|` against the CPU
//! arm of the same rep (OS and SS separately, Eh; the max over reps) and, for the
//! device arms, the MODELLED download time (`bytes_d2h / 6.5e9`) as a share of
//! the arm's median wall. The only comparison is between these arms; a mixed arm
//! slower than the f64 arm is reported in the same table.
//!
//! Quotable only if `/proc/pressure/cpu` `some avg10` <= 0.05 before and the
//! external-load sampler (`ferric_benchmarks::quiet`), which covers the timed
//! arms only (not the untimed SCF/`B_ov` setup), finds no other process using CPU; otherwise `NOT QUOTABLE: box contested` is printed.
//!
//! Same env knobs as `rimp2_stage_split`: `FERRIC_MOL` (default benzene),
//! `FERRIC_OBS` (aug-cc-pvtz), `FERRIC_AUX` (aug-cc-pvtz-rifit), `FERRIC_FROZEN`
//! (6); plus `FERRIC_PREC_REPS` (7). `FERRIC_PREC_SMOKE=1` runs one rep on water
//! (a wiring check, not a measurement); `FERRIC_PSI_SETTLE_SECS` (180) bounds the wait for PSI to decay after setup.
//! Arm order rotates per rep (cpu,f64,mixed / f64,mixed,cpu / mixed,cpu,f64) and is printed.
//! Any concurrent monitor (ps loop, htop) counts as external load.
//!
//!   CUDA_VISIBLE_DEVICES=0 FERRIC_MOL=benzene OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=6 \
//!     cargo run --release -p ferric-benchmarks --features gpu --example rimp2_precision_split
//!
//! Without `--features gpu` this builds to a stub that says so.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("rimp2_precision_split needs --features gpu");
}

#[cfg(feature = "gpu")]
mod run {
    use ferric_benchmarks::quiet;
    use ferric_core::basis;
    use ferric_core::gpu::device::device;
    use ferric_core::gpu::{
        install, pool, stats, GpuMode, GpuSettingsExplicit, GpuStatsSnapshot, MixedKernel,
        MixedKernelSet, Precision,
    };
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_integrals::three_index_source::ThreeIndexSource;
    use ferric_integrals::threeindex;
    use ferric_mp2::rimp2::{
        active_occ, eri3_budget_bytes, metric_inverse_sqrt, spin_components_from_b_ov_kappa_cpu,
        stream_dressed_mo_band, SpinComponents,
    };
    use ferric_mp2::rimp2_gpu::spin_components_on_device;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use ndarray::Array2;
    use std::time::Instant;

    /// Modelled PCIe download bandwidth for the share column (bytes/s).
    const D2H_BW: f64 = 6.5e9;

    struct Prepared {
        b_ov: Array2<f64>,
        eps: Vec<f64>,
        nocc: usize,
        nvir: usize,
        first_occ: usize,
        nocc_total: usize,
    }

    fn env_or(name: &str, default: &str) -> String {
        std::env::var(name).unwrap_or_else(|_| default.into())
    }

    fn env_or_unset(name: &str) -> String {
        std::env::var(name).unwrap_or_else(|_| "(unset)".into())
    }

    /// RI-JK RHF and `B_ov`, built once (the same orbitals feed every arm).
    fn prepare(smoke: bool) -> Prepared {
        let mol_name = env_or("FERRIC_MOL", if smoke { "water" } else { "benzene" });
        let obs_name = env_or("FERRIC_OBS", if smoke { "cc-pvdz" } else { "aug-cc-pvtz" });
        let aux_name = env_or(
            "FERRIC_AUX",
            if smoke {
                "cc-pvdz-ri"
            } else {
                "aug-cc-pvtz-rifit"
            },
        );
        let frozen: usize = env_or("FERRIC_FROZEN", if smoke { "0" } else { "6" })
            .parse()
            .expect("FERRIC_FROZEN");
        let mol = Molecule::load_xyz(&format!("testdata/molecules/{mol_name}.xyz")).unwrap();
        let obs_set = basis::bundled(&obs_name).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf_cfg = RhfConfig {
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("def2-universal-jkfit".into()),
            max_iter: 100,
            ..Default::default()
        };
        eprintln!(
            "[setup] {mol_name}/{obs_name}, aux={aux_name}, RI-JK SCF (def2-universal-jkfit), nbasis={}",
            obs.nbasis()
        );
        let rhf = solve_rhf(
            &ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &rhf_cfg,
        )
        .unwrap();
        eprintln!(
            "[setup] RHF converged={} iters={} E={:.10}",
            rhf.converged, rhf.iterations, rhf.energy
        );
        let dfbs_set = basis::bundled(&aux_name).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let nocc_total = mol.nelec() as usize / 2;
        let nocc = active_occ(nocc_total, frozen).unwrap();
        let nvir = obs.nbasis() - nocc_total;
        let c = rhf.mos_r();
        let c_occ = c.slice(ndarray::s![.., frozen..frozen + nocc]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc_total..]).to_owned();
        let v2c = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
        let v_inv_sqrt = metric_inverse_sqrt(&v2c, op).unwrap();
        let mut src = ThreeIndexSource::build(op, &obs, &dfbs, eri3_budget_bytes(None)).unwrap();
        let b_ov = stream_dressed_mo_band(&mut src, &v_inv_sqrt, &c_occ, &c_vir, None).unwrap();
        eprintln!(
            "[dims] naux={} nocc={nocc} nvir={nvir} (frozen={frozen}) B_ov {:?}",
            dfbs.nbasis(),
            b_ov.dim()
        );
        Prepared {
            b_ov,
            eps: rhf.eps_r().to_vec(),
            nocc,
            nvir,
            first_occ: frozen,
            nocc_total,
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Arm {
        Cpu,
        DevF64,
        DevMixed,
    }

    const ARMS: [(Arm, &str); 3] = [
        (Arm::Cpu, "cpu"),
        (Arm::DevF64, "dev_f64"),
        (Arm::DevMixed, "dev_mixed"),
    ];

    fn run_arm(p: &Prepared, arm: Arm) -> SpinComponents {
        let device_run = |precision| {
            let dev = device(0).expect("GPU 0");
            let pl = pool().expect("device pool");
            spin_components_on_device(
                &dev,
                &pl,
                &p.b_ov,
                &p.eps,
                p.nocc,
                p.nvir,
                p.first_occ,
                p.nocc_total,
                None,
                precision,
                true,
            )
            .expect("device arm")
        };
        match arm {
            Arm::Cpu => spin_components_from_b_ov_kappa_cpu(
                &p.b_ov,
                &p.eps,
                p.nocc,
                p.nvir,
                p.first_occ,
                p.nocc_total,
                None,
            ),
            Arm::DevF64 => device_run(Precision::F64),
            Arm::DevMixed => device_run(Precision::Mixed),
        }
    }

    /// Counter deltas summed over the timed reps of one arm.
    #[derive(Default, Clone, Copy)]
    struct Delta {
        offloaded: u64,
        mixed: u64,
        panels: u64,
        h2d: u64,
        d2h: u64,
        fallbacks: u64,
    }

    impl Delta {
        fn add(&mut self, a: &GpuStatsSnapshot, b: &GpuStatsSnapshot) {
            self.offloaded += b.gemm_offloaded - a.gemm_offloaded;
            self.mixed += b.gemm_mixed - a.gemm_mixed;
            self.panels += b.mixed_panels - a.mixed_panels;
            self.h2d += b.bytes_h2d - a.bytes_h2d;
            self.d2h += b.bytes_d2h - a.bytes_d2h;
            let fb = |s: &GpuStatsSnapshot| {
                s.mixed_fallback_f64
                    + s.gemm_cpu_pool_full
                    + s.gemm_cpu_layout
                    + s.gemm_cpu_cuda_error
            };
            self.fallbacks += fb(b) - fb(a);
        }
    }

    #[derive(Default)]
    struct Rows {
        wall: Vec<f64>,
        cpu: Vec<f64>,
        delta: Delta,
        d_os: f64,
        d_ss: f64,
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

    /// Poll PSI every 2 s until it is within the quotable ceiling or `max_secs`
    /// elapse; returns the last reading.
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

    /// Arm index order for rep `r`: the arms are rotated each rep (rep 0 cpu,
    /// dev_f64, dev_mixed; rep 1 dev_f64, dev_mixed, cpu; rep 2 dev_mixed,
    /// cpu, dev_f64; ...) so no arm always runs first or after the same arm.
    fn order(r: usize) -> [usize; 3] {
        [r % 3, (r + 1) % 3, (r + 2) % 3]
    }

    fn measure(p: &Prepared, reps: usize) -> [Rows; 3] {
        let mut rows: [Rows; 3] = Default::default();
        for (arm, _) in ARMS {
            run_arm(p, arm); // untimed warm-up
        }
        for r in 0..reps {
            let ord = order(r);
            let names: Vec<&str> = ord.iter().map(|&i| ARMS[i].1).collect();
            println!("rep {r} order: {}", names.join(", "));
            let mut res: [Option<SpinComponents>; 3] = [None, None, None];
            for &i in &ord {
                let s0 = stats();
                let c0 = process_cpu_seconds();
                let t = Instant::now();
                let sc = run_arm(p, ARMS[i].0);
                let wall = t.elapsed().as_secs_f64();
                let cpu = process_cpu_seconds() - c0;
                let row = &mut rows[i];
                row.wall.push(wall);
                row.cpu.push(cpu);
                row.delta.add(&s0, &stats());
                res[i] = Some(sc);
            }
            let c = res[0].as_ref().expect("cpu arm ran");
            for i in 1..3 {
                let sc = res[i].as_ref().expect("arm ran");
                rows[i].d_os = rows[i].d_os.max((sc.e_os - c.e_os).abs());
                rows[i].d_ss = rows[i].d_ss.max((sc.e_ss - c.e_ss).abs());
            }
        }
        rows
    }

    fn print_table(rows: &[Rows; 3], reps: usize) {
        println!(
            "\n{:<10} {:>9} {:>9} {:>8} | {:>11} {:>11} | {:>5} {:>5} {:>7} {:>10} {:>10} {:>4} | {:>11}",
            "arm", "min ms", "med ms", "cpu/wall", "|dE_OS| Eh", "|dE_SS| Eh", "offl", "mixed",
            "panels", "h2d B", "d2h B", "fb", "d2h share"
        );
        for ((arm, name), r) in ARMS.iter().zip(rows) {
            let (mn, md) = min_median(&r.wall);
            let (cmed, wmed) = (min_median(&r.cpu).1, md);
            let dev = *arm != Arm::Cpu;
            let share = if dev {
                format!(
                    "{:.1}%",
                    100.0 * (r.delta.d2h as f64 / reps as f64 / D2H_BW) / md
                )
            } else {
                "-".to_string()
            };
            let de = |d: f64| {
                if dev {
                    format!("{d:.3e}")
                } else {
                    "(ref)".to_string()
                }
            };
            println!(
                "{:<10} {:>9.1} {:>9.1} {:>8.2} | {:>11} {:>11} | {:>5} {:>5} {:>7} {:>10} {:>10} {:>4} | {:>11}",
                name,
                mn * 1e3,
                md * 1e3,
                cmed / wmed,
                de(r.d_os),
                de(r.d_ss),
                r.delta.offloaded,
                r.delta.mixed,
                r.delta.panels,
                r.delta.h2d,
                r.delta.d2h,
                r.delta.fallbacks,
                share
            );
        }
        println!(
            "stats columns are summed over the {reps} timed reps (warm-up excluded); d2h share is the \
             modelled download (bytes_d2h per rep / {D2H_BW:.1e} B/s) over the arm's median wall; \
             |dE| is the max over reps against the cpu arm of the same rep"
        );
    }

    pub fn main() {
        let smoke = std::env::var_os("FERRIC_PREC_SMOKE").is_some();
        let reps: usize = if smoke {
            1
        } else {
            env_or("FERRIC_PREC_REPS", "7").parse().expect("reps")
        };
        println!(
            "RAYON_NUM_THREADS={} OPENBLAS_NUM_THREADS={} reps={reps}{}",
            env_or_unset("RAYON_NUM_THREADS"),
            env_or_unset("OPENBLAS_NUM_THREADS"),
            if smoke {
                " SMOKE: wiring check only, not a measurement"
            } else {
                ""
            }
        );
        let p = prepare(smoke);
        // Device and mixed policy are installed after the SCF and B_ov build so
        // the shared orbitals are made by the same code for every arm.
        install(GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            memory_gb: Some(4.0),
            precision: Some(Precision::Mixed),
            mixed_kernels: Some(MixedKernelSet::EMPTY.with(MixedKernel::RiMp2Energy)),
            ..Default::default()
        })
        .expect("install GPU settings");
        // The SCF and B_ov build load this process's own cores, which raises PSI
        // for tens of seconds; wait for it to decay (no timing is taken here) so
        // the reading is the box's, then read PSI and start the sampler together.
        let settle = if smoke {
            0
        } else {
            env_or("FERRIC_PSI_SETTLE_SECS", "180")
                .parse()
                .expect("settle")
        };
        let psi_before = settle_psi(settle);
        println!(
            "PSI cpu some avg10 before = {psi_before:.2} (must be <= 0.05 for a quotable run; \
             read after setup, waiting up to {settle} s for the harness's own setup load to decay)"
        );
        let sampler = quiet::Sampler::start();
        let rows = measure(&p, reps);
        print_table(&rows, reps);
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
