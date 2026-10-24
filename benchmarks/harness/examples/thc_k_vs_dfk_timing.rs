//! THC-K vs DF-K wall clock, same aux basis, same thread count, same system.
//! Results and their interpretation: `reference/thc/thc-lane-2026-10.md`.
//!
//! Two modes, run in one invocation:
//!
//! - **full** (`THC_SIZES_FULL`, default `1,2,3,4,6,8`): a loosely converged
//!   DF-RHF supplies `C_occ`; `DfK::new` and `ThcK::new` are both timed (setup),
//!   then their steady-state `build_from_occ` (median of `THC_REPS`). Accuracy
//!   of THC-K against DF-K on the SAME aux: `max|ΔK|/max|K|` and the RHF
//!   exchange-energy error `ΔE_x = −tr(C Cᵀ ΔK)` in mHa.
//! - **build** (`THC_SIZES_BUILD`, default `4,8,12,16,20,24`): per-build cost
//!   only, at the prescribed `N_IP = ceil(c_isdf · naux)`, with SYNTHETIC
//!   factors (`ThcK::from_parts`) and a synthetic `C_occ`. Both builds are
//!   dense GEMM chains whose cost depends only on the shapes; the overlap rows
//!   (sizes present in both modes) are the control for that claim. No accuracy
//!   is measured in this mode.
//!
//! Run (matched single-thread settings, inside the memory cgroup):
//!   OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 scripts/ferric-limited -- \
//!     target/release/examples/thc_k_vs_dfk_timing
//! after `cargo build --release -p ferric-benchmarks --features thc --example thc_k_vs_dfk_timing`.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_k::DfK;
use ferric_scf::fock::KBuilder;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::thc_k::{ThcK, ThcKConfig};
use ndarray::{s, Array2};
use std::time::Instant;

const OBS: &str = "cc-pvdz";
const AUX: &str = "cc-pvdz-ri";
const DFK_BUDGET: usize = 10 << 30;

// The THC_* environment knobs below are EXAMPLE-ONLY measurement switches for
// this harness; they are deliberately not ConfigVar settings and nothing in
// the library reads them.
fn env_list(key: &str, default: &str) -> Vec<usize> {
    std::env::var(key)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.trim().parse().expect("integer list"))
        .collect()
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .map_or(default, |v| v.parse().expect("float"))
}

struct System {
    mol: Molecule,
    obs_bs: BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    nocc: usize,
}

fn system(n_c: usize) -> System {
    let mol = Molecule::load_xyz(&format!("testdata/molecules/alkane_{n_c}.xyz")).expect("xyz");
    let obs_bs = basis::bundled(OBS).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    let nocc = mol.nelec() as usize / 2;
    System {
        mol,
        obs_bs,
        obs,
        dfbs,
        nocc,
    }
}

/// Loosely converged DF-RHF occupied block (realistic orbitals for accuracy).
fn scf_c_occ(sys: &System) -> Array2<f64> {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &sys.obs).unwrap();
    let cfg = RhfConfig {
        df_j_aux: Some(AUX.into()),
        df_k_aux: Some(AUX.into()),
        energy_conv: 1e-6,
        density_conv: 1e-5,
        ..RhfConfig::default()
    };
    let r = solve_rhf(
        &ParallelContext::default(),
        &sys.mol,
        &sys.obs,
        op,
        &bounds,
        &cfg,
    )
    .expect("RHF");
    r.mos_alpha.slice(s![.., ..sys.nocc]).to_owned()
}

/// Deterministic dense filler; cost of the timed GEMMs does not depend on it.
fn filler(r: usize, c: usize, scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((r, c), |(i, j)| {
        scale * ((((i * 131 + j * 71) % 997) as f64) / 997.0 - 0.5)
    })
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Median wall (s) of `build_from_occ` after one warm-up; returns the last K.
fn time_build(b: &mut dyn KBuilder, c: &Array2<f64>, reps: usize) -> (f64, Array2<f64>) {
    let n = c.nrows();
    let mut k = Array2::zeros((n, n));
    b.build_from_occ(c, &mut k).unwrap();
    let samples = (0..reps)
        .map(|_| {
            let t = Instant::now();
            b.build_from_occ(c, &mut k).unwrap();
            t.elapsed().as_secs_f64()
        })
        .collect();
    (median(samples), k)
}

fn timed<T>(f: impl FnOnce() -> T) -> (f64, T) {
    let t = Instant::now();
    let v = f();
    (t.elapsed().as_secs_f64(), v)
}

fn vm_hwm_gb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with("VmHWM"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|kb| kb.parse::<f64>().ok())
        .map_or(f64::NAN, |kb| kb / 1048576.0)
}

fn dfk_setup_and_build(sys: &System, c: &Array2<f64>, reps: usize) -> (f64, f64, Array2<f64>) {
    let (t_setup, dfk) =
        timed(|| DfK::new(Operator::coulomb(), &sys.obs, &sys.dfbs, DFK_BUDGET).unwrap());
    let mut dfk = dfk;
    let (t_build, k) = time_build(&mut dfk, c, reps);
    (t_setup, t_build, k)
}

fn run_full(n_c: usize, cfg: &ThcKConfig, reps: usize) {
    let sys = system(n_c);
    let (t_scf, c) = timed(|| scf_c_occ(&sys));
    let (dfk_setup, dfk_build, k_df) = dfk_setup_and_build(&sys, &c, reps);
    let (thc_setup, thc) = timed(|| {
        ThcK::new(
            &sys.mol,
            &sys.obs_bs,
            &sys.obs,
            &sys.dfbs,
            Operator::coulomb(),
            cfg,
        )
        .unwrap()
    });
    let n_ip = thc.n_interpolation_points();
    let mut thc = thc;
    let (thc_build, k_thc) = time_build(&mut thc, &c, reps);
    let dk = &k_thc - &k_df;
    let rel = dk.iter().fold(0.0f64, |m, v| m.max(v.abs()))
        / k_df.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let de_x_mha = -1e3 * (c.dot(&c.t()) * &dk).sum();
    println!(
        "full  alkane_{n_c:<3} nbf {:>5} naux {:>5} nocc {:>4} n_ip {:>6} c_eff {:>5.2} | DFK setup {:>8.3} build {:>8.4} | THC setup {:>9.2} build {:>8.4} | build ratio THC/DFK {:>7.2} | rel|dK| {:>8.2e} dE_x {:>9.3} mHa | scf {:>7.1} s | HWM {:>5.2} GB",
        sys.obs.nbasis(), sys.dfbs.nbasis(), sys.nocc, n_ip, n_ip as f64 / sys.dfbs.nbasis() as f64,
        dfk_setup, dfk_build, thc_setup, thc_build, thc_build / dfk_build, rel, de_x_mha, t_scf, vm_hwm_gb()
    );
}

fn run_build(n_c: usize, c_isdf: f64, reps: usize) {
    let sys = system(n_c);
    let (nbf, naux) = (sys.obs.nbasis(), sys.dfbs.nbasis());
    let c = filler(nbf, sys.nocc, 0.2);
    let (dfk_setup, dfk_build, _) = dfk_setup_and_build(&sys, &c, reps);
    let n_ip = (c_isdf * naux as f64).ceil() as usize;
    let m = {
        let a = filler(n_ip, n_ip, 1e-3);
        (&a + &a.t()) * 0.5
    };
    let mut thc = ThcK::from_parts(filler(nbf, n_ip, 0.1), m).unwrap();
    let (thc_build, _) = time_build(&mut thc, &c, reps);
    println!(
        "build alkane_{n_c:<3} nbf {nbf:>5} naux {naux:>5} nocc {:>4} n_ip {n_ip:>6} | DFK setup {dfk_setup:>8.3} build {dfk_build:>8.4} | THC build {thc_build:>8.4} | build ratio THC/DFK {:>7.2} | HWM {:>5.2} GB",
        sys.nocc, thc_build / dfk_build, vm_hwm_gb()
    );
}

fn main() {
    let grid = env_list("THC_GRID", "30,50");
    assert_eq!(
        grid.len(),
        2,
        "THC_GRID must be \"n_radial,n_angular\", got {grid:?}"
    );
    let cfg = ThcKConfig {
        c_isdf: env_f64("THC_C_ISDF", ThcKConfig::default().c_isdf),
        n_radial: grid[0],
        n_angular: grid[1],
        ..ThcKConfig::default()
    };
    let reps = env_list("THC_REPS", "3")[0];
    println!("# THC-K vs DF-K: obs {OBS}, aux {AUX} (both builders), build_from_occ timed, median of {reps}");
    println!(
        "# c_isdf {} parent grid ({},{}) rank_tol {:e}; times in s",
        cfg.c_isdf, cfg.n_radial, cfg.n_angular, cfg.rank_tol
    );
    for n in env_list("THC_SIZES_FULL", "1,2,3,4,6,8") {
        run_full(n, &cfg, reps);
    }
    for n in env_list("THC_SIZES_BUILD", "4,8,12,16,20,24") {
        run_build(n, cfg.c_isdf, reps);
    }
}
