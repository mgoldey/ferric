#![cfg(all(feature = "gpu", feature = "test-seams"))]
//! The SCF precision ladder for the mixed J/K device kernels `dfj-pack` (RI-J)
//! and `dfk-occ` (DF-K occupied path): mixed builds only while the density
//! moves, and an f64 exit (`ferric_scf::jk_ladder`). Both kernels are unshipped,
//! so the tests allow them through the `MIXED_SETTINGS_OVERRIDE` seam of
//! `df_k_gpu` (the resolver's shipped set plus both names); mode, device and
//! pool come from the installed (auto, 3.5 GB) settings. Its own binary: the GPU
//! settings are installed once per process.
//!
//! THE BAR IS DERIVED, NOT FITTED (module docs of `jk_ladder`, "Error budget").
//! For each system the test computes, at the f64 reference density, the
//! summation magnitudes `S_J = |c|ᵀ|B||w|` (RI-J energy) and
//! `Σ|D|∘S`, `S = Σ_P (|B_P||C|)(|B_P||C|)ᵀ` (DF-K energy), and from them
//!   * the f64 floor `T_γ = (γ_pair+γ_naux)(u64)·S_J
//!       + ½(2γ_nbf+γ_{naux·nocc})(u64)·Σ|D|S + γ_{n_tr}(u64)·½Σ|D||H+F|`
//!     (either code path's f64 accumulation of J, K and the energy trace);
//!   * the bar `2·T_γ + 2·|dE_exit(f64 run)|`: two f64 SCFs, each within T_γ
//!     of exact arithmetic, stopped by the same ΔP gate, whose realized
//!     termination is the energy change of the f64 reference's exit iteration.
//!     (A laddered run's own exit dE is NOT used: after the exit veto its last
//!     step crosses from the mixed to the f64 fixed point, a first-order jump
//!     of the mutant's size, and would make the bar vacuous. Measured: the
//!     τ = 0 water run's exit dE equalled the mutant's error, 6.3e-7 Eh.)
//!   * the first-order mixed budget `M = (u32+γ_pair+γ_naux)·S_J + ½ε_K·Σ|D|S`
//!     that a converged energy keeps when its last build was mixed.
//!
//! Asserted per system: the f64 default never consults the ladder and is
//! bit-identical to the no-ladder seam; the production ladder (τ = 1e-5, both
//! kernels) and the stress ladder (τ = 0: every build after the first mixed
//! until the exit; `dfj-pack` only beyond water, see `Kernels::JOnly`) land
//! inside the bar, with counters showing mixed builds AND the f64 exit; the
//! mutant without the exit rebuild (τ = 0) lands OUTSIDE the bar and inside
//! `M + bar` (the error model, not just its size); the mutant whose exit
//! criterion never fires is caught by the counters (the τ test must end the
//! mixed builds by itself at a tight `density_conv`, without the exit veto).
use ferric_core::basis;
use ferric_core::gpu::config::{GpuSettings, GpuSettingsExplicit};
use ferric_core::gpu::mixed::effective_k_panel;
use ferric_core::gpu::mixed_host::{gamma, U32, U64};
use ferric_core::gpu::{
    install, probe, stats, GpuMode, GpuStatsSnapshot, GpuStatus, MixedKernel, MixedKernelSet,
    Precision,
};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::precision_ladder::seams::{NEVER_EXIT, NO_EXIT_REBUILD};
use ferric_core::precision_ladder::{counters, LadderCounters, LadderSite};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_scf::df_j::DfJ;
use ferric_scf::df_k::DfK;
use ferric_scf::df_k_gpu::{k_error_factor_mixed, MIXED_SETTINGS_OVERRIDE};
use ferric_scf::jk_ladder::seams::{JK_LADDER_OFF, JK_LADDER_THRESH_OVERRIDE};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::{Array1, Array2, ArrayView2};
use std::sync::atomic::Ordering;
use std::sync::Mutex;

static GPU: Mutex<()> = Mutex::new(());

const AUX: &str = "def2-universal-jkfit";

fn ready() -> bool {
    if !matches!(probe(0), GpuStatus::Ready(_)) {
        eprintln!("skipping: no CUDA device");
        assert!(
            std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"),
            "FERRIC_GPU_TESTS_REQUIRED=1 but no device"
        );
        return false;
    }
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        install(GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            memory_gb: Some(3.5),
            ..Default::default()
        })
        .expect("install");
    });
    true
}

/// Which mixed J/K kernels an arm allows.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kernels {
    /// `dfj-pack` and `dfk-occ`.
    Both,
    /// `dfj-pack` only. The τ = 0 stress needs it beyond water: `dfk-occ`
    /// re-rounds C_occ every build, and with EVERY build mixed that noise held
    /// ΔP above `density_conv` = 1e-9 for 200 iterations on C6H6/def2-SVP
    /// (measured), so the exit gate the veto acts on was never reached. The
    /// production ladder stops mixing at ΔD_rms ≤ 1e-5 and has no such stall.
    JOnly,
}

/// The build's shipped set plus the arm's J/K kernels.
fn jk_kernels(k: Kernels) -> MixedKernelSet {
    let s = MixedKernelSet::SHIPPED.with(MixedKernel::DfjPack);
    match k {
        Kernels::Both => s.with(MixedKernel::DfkOcc),
        Kernels::JOnly => s,
    }
}

fn mixed_settings(k: Kernels) -> GpuSettings {
    GpuSettings::resolve_with_default(
        GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            precision: Some(Precision::Mixed),
            mixed_kernels: Some(jk_kernels(k)),
            memory_gb: Some(3.5),
            ..Default::default()
        },
        |_| None,
        Precision::F64,
        jk_kernels(k),
    )
    .map(|(s, _)| s)
    .expect("resolve")
}

/// One SCF configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Arm {
    /// f64 (no mixed kernel allowed): the reference.
    F64,
    /// f64, with the no-ladder seam (must equal `F64` bit for bit).
    F64NoLadder,
    /// Mixed allowed, the production ladder (τ = `JK_LADDER_THRESH`).
    Ladder,
    /// Mixed allowed, τ = 0: every build after the first mixed until the exit.
    LadderTau0,
    /// Mutant: τ = 0 and no exit rebuild.
    NoExitRebuild,
    /// Mutant: production τ, but the exit criterion never fires.
    NeverExit,
}

/// Sets the seams for an arm; resets every seam on drop.
struct Seams;
impl Seams {
    fn arm(arm: Arm, k: Kernels) -> Self {
        let mixed = !matches!(arm, Arm::F64 | Arm::F64NoLadder);
        *MIXED_SETTINGS_OVERRIDE.lock().unwrap() = mixed.then(|| mixed_settings(k));
        JK_LADDER_OFF.store(arm == Arm::F64NoLadder, Ordering::SeqCst);
        *JK_LADDER_THRESH_OVERRIDE.lock().unwrap() =
            matches!(arm, Arm::LadderTau0 | Arm::NoExitRebuild).then_some(0.0);
        NO_EXIT_REBUILD.store(arm == Arm::NoExitRebuild, Ordering::SeqCst);
        NEVER_EXIT.store(arm == Arm::NeverExit, Ordering::SeqCst);
        Seams
    }
}
impl Drop for Seams {
    fn drop(&mut self) {
        *MIXED_SETTINGS_OVERRIDE
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        *JK_LADDER_THRESH_OVERRIDE
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        JK_LADDER_OFF.store(false, Ordering::SeqCst);
        NO_EXIT_REBUILD.store(false, Ordering::SeqCst);
        NEVER_EXIT.store(false, Ordering::SeqCst);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Rhf,
    Uhf,
    Rohf,
}

struct System {
    label: &'static str,
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    kind: Kind,
}

impl System {
    fn new(label: &'static str, mol: Molecule, obs: &str, kind: Kind) -> Self {
        let obs = PreparedBasis::new(&mol, &basis::bundled(obs).unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
        Self {
            label,
            mol,
            obs,
            dfbs,
            kind,
        }
    }
}

struct Run {
    res: ScfResult,
    ladder: LadderCounters,
    gpu: GpuStatsSnapshot,
}

fn gpu_delta(a: &GpuStatsSnapshot, b: &GpuStatsSnapshot) -> (u64, u64, u64) {
    (
        b.dfj_device_builds - a.dfj_device_builds,
        b.dfk_device_builds - a.dfk_device_builds,
        b.mixed_panels - a.mixed_panels,
    )
}

fn scf(sys: &System, arm: Arm, k: Kernels) -> Run {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &sys.obs).unwrap();
    let cfg = RhfConfig {
        // dP is the real gate; energy_conv is only a "not still descending" bound
        // (an RI energy never settles to 1e-10, see the noise-floor notes).
        energy_conv: 1e-6,
        density_conv: 1e-9,
        max_iter: 200,
        df_j_aux: Some(AUX.into()),
        df_k_aux: Some(AUX.into()),
        three_index_budget_bytes: 3 << 30,
        ..Default::default()
    };
    let _s = Seams::arm(arm, k);
    let (l0, g0) = (counters(LadderSite::Jk), stats());
    let ctx = ParallelContext::default();
    let res = match sys.kind {
        Kind::Rhf => solve_rhf(&ctx, &sys.mol, &sys.obs, op, &bounds, &cfg),
        Kind::Uhf => solve_uhf(&ctx, &sys.mol, &sys.obs, &bounds, &cfg),
        Kind::Rohf => solve_rohf(&ctx, &sys.mol, &sys.obs, op, &bounds, &cfg),
    }
    .unwrap_or_else(|e| panic!("{} {arm:?}: {e:?}", sys.label));
    assert!(res.converged, "{} {arm:?}: not converged", sys.label);
    Run {
        res,
        ladder: counters(LadderSite::Jk).since(&l0),
        gpu: {
            let g1 = stats();
            let mut d = g1;
            let (j, k, p) = gpu_delta(&g0, &g1);
            d.dfj_device_builds = j;
            d.dfk_device_builds = k;
            d.mixed_panels = p;
            d
        },
    }
}

/// The derived bar against the f64 reference (see the file docs).
fn bar_of(b: &Budget, f64_run: &ScfResult) -> f64 {
    2.0 * b.t_gamma + 2.0 * last_de(f64_run)
}

fn last_de(r: &ScfResult) -> f64 {
    r.convergence.as_ref().map(|c| c.de.abs()).unwrap_or(0.0)
}

// ---------------------------------------------------------------- the budget

/// The packed raw tensor `(naux, n(n+1)/2)` (streamed from a spilled source; every
/// tier stores the same numbers bit for bit).
fn packed_tensor(sys: &System) -> Array2<f64> {
    let (n, naux) = (sys.obs.nbasis(), sys.dfbs.nbasis());
    let mut src = ThreeIndexSource::build_band(
        Operator::coulomb(),
        &sys.obs,
        &sys.dfbs,
        n * n * 8 * 40,
        0,
        naux,
    )
    .unwrap();
    let mut out = Array2::<f64>::zeros((naux, n * (n + 1) / 2));
    src.for_each_packed_block(1, |blk| {
        let rows = blk.data.nrows();
        out.slice_mut(ndarray::s![blk.p0..blk.p0 + rows, ..])
            .assign(&blk.data);
        Ok(())
    })
    .unwrap();
    out
}

/// Packed weights `w_k` of `tr(D J) = Σ_k w_k J_k` (off-diagonals doubled).
fn weights(d: &Array2<f64>) -> Vec<f64> {
    let n = d.nrows();
    let mut w = Vec::with_capacity(n * (n + 1) / 2);
    for mu in 0..n {
        for nu in 0..mu {
            w.push(d[(mu, nu)] + d[(nu, mu)]);
        }
        w.push(d[(mu, mu)]);
    }
    w
}

/// `S = Σ_P (|B_P||C|)(|B_P||C|)ᵀ` over the dressed tensor.
fn s_matrix(flat: &ArrayView2<f64>, n: usize, c: &Array2<f64>) -> Array2<f64> {
    let abs_c = c.mapv(f64::abs);
    let mut s = Array2::<f64>::zeros((n, n));
    let all = flat.as_slice().expect("incore_flat is standard layout");
    for p in 0..flat.nrows() {
        let bp = ArrayView2::from_shape((n, n), &all[p * n * n..(p + 1) * n * n]).unwrap();
        let a = bp.mapv(f64::abs).dot(&abs_c);
        s += &a.dot(&a.t());
    }
    s
}

fn abs_dot(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x * y).abs()).sum()
}

/// The derived quantities of one system at the f64 reference solution.
struct Budget {
    /// f64 floor `T_γ` (Eh).
    t_gamma: f64,
    /// First-order mixed budget `M` (Eh).
    m_mixed: f64,
    s_j: f64,
    s_k: f64,
}

/// Spin blocks `(D_σ, C_occ,σ, F_σ)` whose K-energy is `−½ tr(D_σ K(C_σ))` and
/// trace term `½ tr(D_σ (H + F_σ))` (closed shell: `D` total, `K(D) = 2 K(C)`,
/// one block).
fn spin_blocks(sys: &System, r: &ScfResult) -> Vec<(Array2<f64>, Array2<f64>, Array2<f64>)> {
    let nelec = sys.mol.nelec() as usize;
    let unpaired = sys.mol.multiplicity - 1;
    let na = (nelec + unpaired) / 2;
    let nb = nelec - na;
    let occ = |c: &Array2<f64>, k: usize| c.slice(ndarray::s![.., ..k]).to_owned();
    match sys.kind {
        Kind::Rhf => vec![(
            r.density_total.clone(),
            occ(&r.mos_alpha, na),
            r.fock_alpha.clone(),
        )],
        Kind::Uhf | Kind::Rohf => {
            let (fa, fb) = match (&r.rohf_spin_focks, &r.fock_beta) {
                (Some((fa, fb)), _) => (fa.clone(), fb.clone()),
                (None, Some(fb)) => (r.fock_alpha.clone(), fb.clone()),
                (None, None) => panic!("{}: no spin Fock matrices", sys.label),
            };
            let cb = r.mos_beta.as_ref().unwrap_or(&r.mos_alpha);
            vec![
                (r.density_alpha.clone(), occ(&r.mos_alpha, na), fa),
                (
                    r.density_beta.clone().expect("open-shell beta density"),
                    occ(cb, nb),
                    fb,
                ),
            ]
        }
    }
}

fn budget(sys: &System, f64_run: &ScfResult) -> Budget {
    let (n, naux) = (sys.obs.nbasis(), sys.dfbs.nbasis());
    let pair = n * (n + 1) / 2;
    // J: S_J = |c|ᵀ|B||w| at the converged total density.
    let bp = packed_tensor(sys);
    let w = weights(&f64_run.density_total);
    let mut d_p = vec![0.0; naux];
    let mut s_d = vec![0.0; naux];
    for p in 0..naux {
        for (k, wk) in w.iter().enumerate() {
            let t = bp[(p, k)] * wk;
            d_p[p] += t;
            s_d[p] += t.abs();
        }
    }
    drop(bp);
    let dfj = DfJ::new(Operator::coulomb(), &sys.obs, &sys.dfbs, usize::MAX).unwrap();
    let c = dfj.solve_metric_for_test(&Array1::from(d_p)).unwrap();
    drop(dfj);
    let s_j: f64 = c.iter().zip(&s_d).map(|(c, s)| c.abs() * s).sum();
    // K: Σ_σ Σ|D_σ| S(C_σ), and the trace term Σ_σ ½Σ|D_σ||H+F_σ|.
    let dfk = DfK::new(Operator::coulomb(), &sys.obs, &sys.dfbs, usize::MAX).unwrap();
    let flat = dfk
        .dressed_incore_flat_for_test()
        .expect("in-core dressed B");
    let h = oneelectron::hcore(&sys.obs);
    let blocks = spin_blocks(sys, f64_run);
    let (mut s_k, mut s_tr, mut nocc_max) = (0.0, 0.0, 0usize);
    for (d, c, f) in &blocks {
        let s = s_matrix(&flat, n, c);
        // RHF: E_K = −¼ tr(D K(D)), K(D) = 2K(C)  ⇒ ½ Σ|D| S; UHF: −½ tr(D_σ K(C_σ)) ⇒ ½ Σ|D_σ| S_σ.
        s_k += 0.5 * abs_dot(d, &s);
        s_tr += 0.5 * abs_dot(d, &(&h + f));
        nocc_max = nocc_max.max(c.ncols());
    }
    let nblk = blocks.len();
    let g_j = gamma(pair, U64) + gamma(naux, U64);
    let g_k64 = 2.0 * gamma(n, U64) + gamma(naux * nocc_max + naux, U64);
    let t_gamma = g_j * s_j + g_k64 * s_k + gamma(nblk * n * n, U64) * s_tr;
    let eps_k = k_error_factor_mixed(n, effective_k_panel(), naux * nocc_max, naux);
    let m_mixed = (U32 + g_j) * s_j + eps_k * s_k;
    Budget {
        t_gamma,
        m_mixed,
        s_j,
        s_k,
    }
}

// ---------------------------------------------------------------- the sweep

fn water() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap()
}

fn sized_systems() -> Vec<System> {
    vec![
        System::new("H2O/cc-pVDZ", water(), "cc-pvdz", Kind::Rhf),
        System::new(
            "C6H6/def2-SVP",
            Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap(),
            "def2-svp",
            Kind::Rhf,
        ),
        System::new(
            "C8H18/def2-SVP",
            Molecule::load_xyz("../../testdata/molecules/alkane_8.xyz").unwrap(),
            "def2-svp",
            Kind::Rhf,
        ),
        System::new(
            "C10H22/def2-SVP",
            Molecule::load_xyz("../../testdata/molecules/alkane_10.xyz").unwrap(),
            "def2-svp",
            Kind::Rhf,
        ),
    ]
}

/// One sweep row: label, nbf, then `(arm, kernels, |dE|, bar)` per arm, and M.
type Row = (String, usize, Vec<(Arm, Kernels, f64, f64)>, f64);

/// One system through every arm (production ladder with both kernels, the τ = 0
/// stress and both mutants with `stress` kernels).
fn sweep(sys: &System, stress: &[Kernels]) -> Row {
    let label = sys.label;
    let f64_run = scf(sys, Arm::F64, Kernels::Both);
    // The f64 default never consults the ladder and runs the pre-ladder code.
    assert_eq!(
        f64_run.ladder,
        LadderCounters::default(),
        "{label}: an f64 SCF stepped the ladder"
    );
    let off = scf(sys, Arm::F64NoLadder, Kernels::Both);
    assert_eq!(
        f64_run.res.energy.to_bits(),
        off.res.energy.to_bits(),
        "{label}: the f64 default is not bit-identical to the no-ladder path"
    );
    assert_eq!(f64_run.res.iterations, off.res.iterations);
    assert!(
        f64_run
            .res
            .fock_alpha
            .iter()
            .zip(off.res.fock_alpha.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{label}: f64 Fock matrices differ"
    );
    let b = budget(sys, &f64_run.res);
    let mut arms = vec![(Arm::Ladder, Kernels::Both)];
    for &k in stress {
        arms.extend([
            (Arm::LadderTau0, k),
            (Arm::NoExitRebuild, k),
            (Arm::NeverExit, k),
        ]);
    }
    let mut out = Vec::new();
    eprintln!(
        "{label}: nbf {} naux {}  E_f64 {:.12} ({} it)  S_J {:.3e}  S_K {:.3e}  T_gamma {:.3e}  M {:.3e}",
        sys.obs.nbasis(),
        sys.dfbs.nbasis(),
        f64_run.res.energy,
        f64_run.res.iterations,
        b.s_j,
        b.s_k,
        b.t_gamma,
        b.m_mixed
    );
    for (arm, k) in arms {
        let r = scf(sys, arm, k);
        let de = (r.res.energy - f64_run.res.energy).abs();
        let bar = bar_of(&b, &f64_run.res);
        out.push((arm, k, de, bar));
        let l = r.ladder;
        eprintln!(
            "{label}: {arm:?} {k:?}: |dE| {de:.3e}  bar {bar:.3e}  ({} it; ladder builds {} mixed {} exit-f64 {}; device J {} K {} mixed panels {})",
            r.res.iterations, l.builds_total, l.builds_mixed, l.exit_rebuilt_f64,
            r.gpu.dfj_device_builds, r.gpu.dfk_device_builds, r.gpu.mixed_panels
        );
        // Every mixed arm really ran mixed device builds of BOTH kernels.
        assert!(l.builds_mixed >= 1, "{label} {arm:?}: no build ran mixed");
        assert!(
            r.gpu.dfj_device_builds >= 1,
            "{label} {arm:?}: RI-J never on the device"
        );
        if sys.kind == Kind::Rhf && k == Kernels::Both {
            assert!(
                r.gpu.mixed_panels > 0,
                "{label} {arm:?}: DF-K never ran mixed"
            );
        }
        // Consulted on every build of this SCF (the counters are process-wide, so
        // the SAD guess's free-atom SCFs add their own, all f64: they run inside
        // a rayon pool, where the device path never runs).
        assert!(
            l.builds_total as usize >= r.res.iterations,
            "{label} {arm:?}: the ladder skipped a build"
        );
        match arm {
            Arm::Ladder => {
                // The τ test ends the mixed builds by itself (no exit veto), and
                // at least two builds are f64: the first and the settled tail.
                assert_eq!(
                    l.exit_rebuilt_f64, 0,
                    "{label}: the ladder never exited by itself"
                );
                assert!(l.builds_total >= l.builds_mixed + 2, "{label}: no f64 tail");
            }
            Arm::LadderTau0 | Arm::NeverExit => {
                // Mixed up to the exit, which the veto turned into f64 iterations.
                assert_eq!(l.exit_rebuilt_f64, 1, "{label} {arm:?}: no f64 exit");
                assert!(l.builds_total >= l.builds_mixed + 2);
            }
            Arm::NoExitRebuild => assert_eq!(l.exit_rebuilt_f64, 0),
            Arm::F64 | Arm::F64NoLadder => unreachable!(),
        }
    }
    (label.to_string(), sys.obs.nbasis(), out, b.m_mixed)
}

/// The size sweep (H2O/cc-pVDZ, C6H6, C8H18, C10H22 at def2-SVP; RHF, RI-J and
/// DF-K from def2-universal-jkfit): see the file docs for what is asserted.
#[test]
fn ladder_energy_is_f64_quality_across_system_size_and_mutants_fail() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let mut rows = Vec::new();
    for (i, sys) in sized_systems().into_iter().enumerate() {
        // Water also runs the stress with both kernels (it converges there).
        let stress: &[Kernels] = if i == 0 {
            &[Kernels::JOnly, Kernels::Both]
        } else {
            &[Kernels::JOnly]
        };
        rows.push(sweep(&sys, stress));
    }
    eprintln!("system            nbf  arm             kernels  |dE|        bar         M");
    for (label, n, arms, m) in &rows {
        for (arm, k, de, bar) in arms {
            eprintln!(
                "{label:<17} {n:>4}  {:<15} {:<8} {de:.3e}   {bar:.3e}   {m:.3e}",
                format!("{arm:?}"),
                format!("{k:?}")
            );
        }
    }
    for (label, _, arms, m) in &rows {
        for &(arm, k, de, bar) in arms {
            if arm == Arm::NoExitRebuild {
                // The mutant without the exit rebuild keeps the first-order
                // mixed term: outside the bar, inside its derived budget.
                assert!(
                    de > bar,
                    "{label} {k:?}: no-exit mutant {de:e} inside the bar {bar:e}"
                );
                assert!(
                    de <= m + bar,
                    "{label} {k:?}: no-exit mutant {de:e} exceeds its first-order budget {:e}",
                    m + bar
                );
            } else {
                assert!(
                    de <= bar,
                    "{label} {arm:?} {k:?}: |dE| {de:e} above the bar {bar:e}"
                );
            }
        }
    }
}

/// Open shell: UHF (RI-J and both spin DF-K builds) and ROHF (RI-J only; its
/// DF-K uses the density path) on H2O⁺/cc-pVDZ at τ = 0, against the derived bar.
#[test]
fn open_shell_ladder_exits_in_f64() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let cation =
        || Molecule::load_xyz_with_charge("../../testdata/molecules/water.xyz", 1, 2).unwrap();
    for kind in [Kind::Uhf, Kind::Rohf] {
        let label = if kind == Kind::Uhf {
            "UHF H2O+"
        } else {
            "ROHF H2O+"
        };
        let sys = System::new(label, cation(), "cc-pvdz", kind);
        let f = scf(&sys, Arm::F64, Kernels::Both);
        assert_eq!(f.ladder, LadderCounters::default());
        let b = budget(&sys, &f.res);
        let t0 = scf(&sys, Arm::LadderTau0, Kernels::Both);
        let mu = scf(&sys, Arm::NoExitRebuild, Kernels::Both);
        let bar = bar_of(&b, &f.res);
        let bar_mu = bar;
        let (de, dm) = (
            (t0.res.energy - f.res.energy).abs(),
            (mu.res.energy - f.res.energy).abs(),
        );
        eprintln!(
            "{label}: |dE| tau=0 {de:.3e}  no-exit {dm:.3e}  bar {bar:.3e}  M {:.3e}  (ladder {:?})",
            b.m_mixed, t0.ladder
        );
        assert_eq!(t0.ladder.exit_rebuilt_f64, 1, "{label}: no f64 exit");
        assert!(t0.ladder.builds_mixed >= 1, "{label}: never mixed");
        assert!(de <= bar, "{label}: |dE| {de:e} above the bar {bar:e}");
        assert!(
            dm > bar_mu,
            "{label}: the no-exit mutant {dm:e} is inside its bar {bar_mu:e}"
        );
        assert!(dm <= b.m_mixed + bar_mu, "{label}: mutant above its budget");
    }
}
