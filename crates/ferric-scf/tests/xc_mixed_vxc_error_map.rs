#![cfg(feature = "test-seams")]
//! Error map of the mixed-precision `xc-vxc` kernel through a real SCF, on the
//! CPU: the `MIXED_VXC_HOST_TWIN` seam forms every ladder-approved V_xc batch
//! product with the host twin of the device kernel (`mixed_host`: f32
//! operands, f32 panels of 64 summed in f64), so the effect of the kernel on
//! converged energies and on SCF convergence is measured without a device.
//! The device kernel's own agreement with this twin is `ferric-dft`'s
//! `gpu_xc_mixed.rs`.
//!
//! The `a_converged_*_exit_returns_*` tests pin that a converged SCF returns
//! an f64 V_xc whatever the ladder did on its exit iteration, and
//! `start_scf_makes_the_next_build_a_first_build` that a reused evaluator's
//! ladder restarts with each SCF.
//!
//! `measured_error_map` prints the table on the validation page
//! (`cargo test -p ferric-scf --features test-seams --test
//! xc_mixed_vxc_error_map -- --ignored --nocapture`).
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::ks::{KsXc, KsXcUks, MIXED_VXC_BUILDS, MIXED_VXC_LADDER_THRESH};
use ferric_dft::xc_batch::{FORCE_HOST, MIXED_VXC_HOST_TWIN};
use ferric_dft::xc_trait::{UksXcContribution, XcContribution};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::direct_jk::DirectJK;
use ferric_scf::rhf::{ks_vxc_closed, solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;
use std::sync::atomic::Ordering;

static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Run {
    energy: f64,
    iterations: usize,
    converged: bool,
    mixed_builds: u64,
}

fn scf(xyz: &str, bas: &str, xc: &str, (charge, mult): (i32, usize), conv: f64, twin: bool) -> Run {
    FORCE_HOST.store(true, Ordering::SeqCst);
    MIXED_VXC_HOST_TWIN.store(twin, Ordering::SeqCst);
    let m0 = MIXED_VXC_BUILDS.load(Ordering::SeqCst);
    let mol = Molecule::load_xyz_with_charge(xyz, charge, mult).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled(bas).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some(xc.into()),
        density_conv: conv,
        energy_conv: 1e-8,
        ..Default::default()
    };
    let ctx = ParallelContext::default();
    let r = if mult == 1 {
        solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg)
    } else {
        solve_uhf(&ctx, &mol, &prep, &bounds, &cfg)
    };
    MIXED_VXC_HOST_TWIN.store(false, Ordering::SeqCst);
    FORCE_HOST.store(false, Ordering::SeqCst);
    let r = r.unwrap();
    Run {
        energy: r.energy,
        iterations: r.iterations,
        converged: r.converged,
        mixed_builds: MIXED_VXC_BUILDS.load(Ordering::SeqCst) - m0,
    }
}

const WATER: &str = "../../testdata/molecules/water.xyz";
const BENZENE: &str = "../../testdata/molecules/benzene.xyz";

/// Water PBE/cc-pVDZ at a tight `density_conv`: the twin SCF ran some builds
/// mixed, converged within 3 iterations of f64, and its energy agrees to far
/// below any convergence threshold (measured 1e-13 Eh).
#[test]
fn mixed_vxc_twin_scf_matches_f64_and_converges() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let f = scf(WATER, "cc-pvdz", "PBE", (0, 1), 1e-10, false);
    let t = scf(WATER, "cc-pvdz", "PBE", (0, 1), 1e-10, true);
    assert_eq!(f.mixed_builds, 0, "the f64 run used the mixed kernel");
    assert!(
        t.mixed_builds >= 2,
        "the ladder never chose mixed ({})",
        t.mixed_builds
    );
    assert!(
        t.converged && t.iterations <= f.iterations + 3,
        "{} vs {}",
        t.iterations,
        f.iterations
    );
    let de = (t.energy - f.energy).abs();
    assert!(de < 1e-10, "|dE| = {de:e}");
}

/// A meta-GGA never uses the kernel (meta-GGA τ is on the never-f32 list):
/// with the twin on, r2SCAN makes no mixed build and reproduces f64 exactly.
#[test]
fn a_meta_gga_never_runs_mixed() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let f = scf(WATER, "cc-pvdz", "R2SCAN", (0, 1), 1e-8, false);
    let t = scf(WATER, "cc-pvdz", "R2SCAN", (0, 1), 1e-8, true);
    assert_eq!(t.mixed_builds, 0);
    assert_eq!(t.energy.to_bits(), f.energy.to_bits());
}

/// Loose SCF settings whose exit build, on the twin, is one the ladder runs
/// mixed (the exit's ΔP is far above `MIXED_VXC_LADDER_THRESH`): exact J (so
/// the Fock matrix can be rebuilt independently) and PBE (no exchange term).
fn loose_cfg() -> RhfConfig {
    RhfConfig {
        xc: Some("PBE".into()),
        density_conv: LOOSE_CONV,
        energy_conv: 1e-2,
        df_j_aux: Some(String::new()),
        three_index_budget_bytes: 2 << 30,
        ..Default::default()
    }
}

const LOOSE_CONV: f64 = 1e-3;

/// Bar for "this Fock matrix carries an f64 V_xc": the two sides are the same
/// arithmetic up to the dense-vs-occupied density pass and the J rebuild,
/// O(1e-15)·|F| (see `ks_vxc_tests`); a mixed V_xc is off by ~1e-7.
const F64_FOCK_TOL: f64 = 1e-11;

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Catches: a converged RKS exit diagonalising (and returning) the Fock
/// matrix of a mixed build. With `density_conv` 1e-3 the exit iteration's
/// density moved by far more than the ladder's 1e-5, so without the f64
/// rebuild at the exit its V_xc is the twin's. The returned Fock matrix must
/// equal H + J[D] + V_xc^f64[D] rebuilt from the returned density.
#[test]
fn a_converged_rks_exit_returns_an_f64_fock_matrix() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mol = Molecule::load_xyz(WATER).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = loose_cfg();
    FORCE_HOST.store(true, Ordering::SeqCst);
    MIXED_VXC_HOST_TWIN.store(true, Ordering::SeqCst);
    let m0 = MIXED_VXC_BUILDS.load(Ordering::SeqCst);
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg);
    let mixed = MIXED_VXC_BUILDS.load(Ordering::SeqCst) - m0;
    MIXED_VXC_HOST_TWIN.store(false, Ordering::SeqCst);
    let r = r.unwrap();
    let (_, v64) = ks_vxc_closed(&mol, &bs, &cfg, &r.density_total).unwrap();
    FORCE_HOST.store(false, Ordering::SeqCst);
    assert!(
        r.converged && mixed >= 1,
        "converged {} mixed {mixed}",
        r.converged
    );
    let dp = r.convergence.as_ref().expect("report").dp_rms;
    assert!(
        dp > 10.0 * MIXED_VXC_LADDER_THRESH,
        "exit ΔP {dp:e}: the exit build was not a mixed one, the test is not live"
    );
    let h = oneelectron::hcore_ecp_with_external(&prep, &mol, &bs, None).unwrap();
    let n = h.nrows();
    let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    DirectJK::new(&ctx, &prep, &bounds, cfg.integral_thresh, 2 << 30)
        .build(&r.density_total, &mut j, &mut k)
        .unwrap();
    let res = max_abs(&(&r.fock_alpha - &h - &j - &v64));
    eprintln!("RKS exit ΔP {dp:.2e}, {mixed} mixed builds, max|F − F_f64| {res:.2e}");
    assert!(res < F64_FOCK_TOL, "returned Fock is not f64: {res:e}");
}

/// Catches: the same for UKS, where the ladder fires on ANY spin while the
/// exit gate reads the total density. PBE has no exchange and J is common to
/// both spins, so F_α − F_β must equal the f64 V_α − V_β of the returned
/// spin densities (a fresh evaluator: its first build is f64).
#[test]
fn a_converged_uks_exit_returns_f64_fock_matrices() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mol = Molecule::load_xyz_with_charge(WATER, 1, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let cfg = loose_cfg();
    FORCE_HOST.store(true, Ordering::SeqCst);
    MIXED_VXC_HOST_TWIN.store(true, Ordering::SeqCst);
    let m0 = MIXED_VXC_BUILDS.load(Ordering::SeqCst);
    let r = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg);
    let mixed = MIXED_VXC_BUILDS.load(Ordering::SeqCst) - m0;
    MIXED_VXC_HOST_TWIN.store(false, Ordering::SeqCst);
    let r = r.unwrap();
    let ks = KsXcUks::new_with_omega_budgeted(
        &mol,
        &bs,
        "PBE",
        &AtomicGridConfig::default(),
        &AtomicGridConfig::default_nlc(),
        None,
        Some(2 << 30),
    )
    .unwrap();
    let (d_a, d_b) = (&r.density_alpha, r.density_beta.as_ref().unwrap());
    let n = d_a.nrows();
    let (mut v_a, mut v_b) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    ks.add_xc_uks(d_a, d_b, &mut v_a, &mut v_b);
    FORCE_HOST.store(false, Ordering::SeqCst);
    assert!(
        r.converged && mixed >= 1,
        "converged {} mixed {mixed}",
        r.converged
    );
    let dp = r.convergence.as_ref().expect("report").dp_rms;
    // rms(Δα + Δβ) ≤ rms(Δα) + rms(Δβ): a total above 2τ moves some spin by > τ.
    assert!(
        dp > 20.0 * MIXED_VXC_LADDER_THRESH,
        "exit ΔP {dp:e}: the exit build was not a mixed one, the test is not live"
    );
    let f_diff = &r.fock_alpha - r.fock_beta.as_ref().unwrap();
    let res = max_abs(&(&f_diff - &(&v_a - &v_b)));
    eprintln!("UKS exit ΔP {dp:.2e}, {mixed} mixed builds, max|ΔF − ΔV_f64| {res:.2e}");
    assert!(res < F64_FOCK_TOL, "returned Fock is not f64: {res:e}");
}

/// Catches: the exit rebuild itself (shared by the UKS and ROKS exits, since
/// both evaluate through `KsXcUks`) not restoring the f64 V_xc. On a reused
/// UKS evaluator, a mixed build at D1 followed by `refine_last_to_f64_uks`
/// must equal a fresh evaluator's f64 build at D1 to rounding, report `true`
/// exactly once, and report `false` (touching nothing) after an f64 build.
/// The ROKS SCF exit is not pinned end to end here: ROHF returns the densities
/// of the NEXT orbitals, not those its exit Fock matrices were built from.
#[test]
fn the_exit_rebuild_restores_the_f64_spin_potentials() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mol = Molecule::load_xyz_with_charge(WATER, 1, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let grid = AtomicGridConfig {
        n_radial: 30,
        n_angular: 50,
        ..Default::default()
    };
    let nlc = AtomicGridConfig::default_nlc();
    let new_ks = || KsXcUks::new(&mol, &bs, "PBE", &grid, &nlc).unwrap();
    let nbf = new_ks().xc_screening_stats().nbf;
    let d = |s: f64| Array2::from_shape_fn((nbf, nbf), |(i, j)| s * (((i + j) % 7) as f64 - 3.0));
    let (d_b0, d_b1) = (d(0.004), d(0.0045));
    let build = |ks: &KsXcUks, da: &Array2<f64>, db: &Array2<f64>| {
        let (mut fa, mut fb) = (Array2::zeros((nbf, nbf)), Array2::zeros((nbf, nbf)));
        ks.add_xc_uks(da, db, &mut fa, &mut fb);
        (fa, fb)
    };
    FORCE_HOST.store(true, Ordering::SeqCst);
    MIXED_VXC_HOST_TWIN.store(true, Ordering::SeqCst);
    let ks = new_ks();
    build(&ks, &d(0.010), &d_b0);
    let m0 = MIXED_VXC_BUILDS.load(Ordering::SeqCst);
    let (mut fa, mut fb) = build(&ks, &d(0.011), &d_b1);
    let mixed = MIXED_VXC_BUILDS.load(Ordering::SeqCst) - m0;
    let mixed_dev = {
        MIXED_VXC_HOST_TWIN.store(false, Ordering::SeqCst);
        let (ra, rb) = build(&new_ks(), &d(0.011), &d_b1);
        MIXED_VXC_HOST_TWIN.store(true, Ordering::SeqCst);
        (max_abs(&(&fa - &ra)).max(max_abs(&(&fb - &rb))), ra, rb)
    };
    let refined = ks.refine_last_to_f64_uks(&mut fa, &mut fb);
    let again = ks.refine_last_to_f64_uks(&mut fa, &mut fb);
    build(&ks, &d(0.011), &d_b1);
    let (mut za, mut zb) = (Array2::zeros((nbf, nbf)), Array2::zeros((nbf, nbf)));
    let after_f64 = ks.refine_last_to_f64_uks(&mut za, &mut zb);
    MIXED_VXC_HOST_TWIN.store(false, Ordering::SeqCst);
    FORCE_HOST.store(false, Ordering::SeqCst);
    let (dev, ra, rb) = mixed_dev;
    let res = max_abs(&(&fa - &ra)).max(max_abs(&(&fb - &rb)));
    eprintln!("mixed build off f64 by {dev:.2e}; after the rebuild {res:.2e}");
    assert_eq!(mixed, 1, "the moving build is mixed (the test is live)");
    // Measured: the mixed build is off by 9.0e-9, the rebuilt one by 0.
    assert!(dev > 1e-10, "mixed V equals f64: {dev:e}");
    assert!(refined && !again, "refine must fire exactly once");
    assert!(res < 1e-3 * dev, "rebuilt V is not f64: {res:e}");
    assert!(!after_f64 && max_abs(&za) == 0.0 && max_abs(&zb) == 0.0);
}

/// Catches: the ladder carrying a previous SCF's density into the next one on
/// a reused evaluator, which would make the new SCF's first build mixed. After
/// `start_scf` a build is a first build: f64, bit-identical to a fresh
/// evaluator's, and not counted as mixed.
#[test]
fn start_scf_makes_the_next_build_a_first_build() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mol = Molecule::load_xyz(WATER).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let grid = AtomicGridConfig {
        n_radial: 30,
        n_angular: 50,
        ..Default::default()
    };
    let nlc = AtomicGridConfig::default_nlc();
    let new_ks = || KsXc::new(&mol, &bs, "PBE", &grid, &nlc).unwrap();
    let nbf = new_ks().xc_screening_stats().nbf;
    let d = |s: f64| Array2::from_shape_fn((nbf, nbf), |(i, j)| s * (((i + j) % 7) as f64 - 3.0));
    FORCE_HOST.store(true, Ordering::SeqCst);
    MIXED_VXC_HOST_TWIN.store(true, Ordering::SeqCst);
    let build = |ks: &KsXc, d: &Array2<f64>| {
        let mut f = Array2::<f64>::zeros((nbf, nbf));
        ks.add_xc(d, &mut f);
        f
    };
    let reused = new_ks();
    let m0 = MIXED_VXC_BUILDS.load(Ordering::SeqCst);
    build(&reused, &d(0.010));
    build(&reused, &d(0.011));
    let m1 = MIXED_VXC_BUILDS.load(Ordering::SeqCst);
    reused.start_scf();
    let after_reset = build(&reused, &d(0.012));
    let m2 = MIXED_VXC_BUILDS.load(Ordering::SeqCst);
    let fresh = build(&new_ks(), &d(0.012));
    MIXED_VXC_HOST_TWIN.store(false, Ordering::SeqCst);
    FORCE_HOST.store(false, Ordering::SeqCst);
    assert_eq!(
        m1 - m0,
        1,
        "the moving second build is mixed (the test is live)"
    );
    assert_eq!(m2, m1, "the first build after start_scf ran mixed");
    assert!(after_reset
        .iter()
        .zip(fresh.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
#[ignore = "measurement: prints the validation-page error map"]
fn measured_error_map() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let cases: &[(&str, &str, &str, &str, (i32, usize))] = &[
        ("H2O", WATER, "cc-pvdz", "PBE", (0, 1)),
        ("H2O", WATER, "aug-cc-pvtz", "PBE", (0, 1)),
        ("H2O", WATER, "cc-pvdz", "R2SCAN", (0, 1)),
        ("H2O+ (UKS)", WATER, "cc-pvdz", "PBE", (1, 2)),
        ("benzene", BENZENE, "cc-pvdz", "PBE", (0, 1)),
        ("benzene", BENZENE, "cc-pvdz", "B3LYP", (0, 1)),
        (
            "n-octane",
            "../../testdata/molecules/alkane_8.xyz",
            "cc-pvdz",
            "PBE",
            (0, 1),
        ),
    ];
    eprintln!("| System | basis | functional | density_conv | iterations f64 / mixed | mixed builds | ΔE (Eh) |");
    for &(name, xyz, bas, xc, cm) in cases {
        for conv in [1e-6, 1e-10] {
            let f = scf(xyz, bas, xc, cm, conv, false);
            let t = scf(xyz, bas, xc, cm, conv, true);
            assert!(f.converged && t.converged, "{name} {xc} {conv:e}");
            eprintln!(
                "| {name} | {bas} | {xc} | {conv:.0e} | {} / {} | {} | {:+.1e} |",
                f.iterations,
                t.iterations,
                t.mixed_builds,
                t.energy - f.energy
            );
        }
    }
}
