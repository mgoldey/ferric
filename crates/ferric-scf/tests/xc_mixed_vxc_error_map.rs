#![cfg(feature = "test-seams")]
//! Error map of the mixed-precision `xc-vxc` kernel through a real SCF, on the
//! CPU: the `MIXED_VXC_HOST_TWIN` seam forms every ladder-approved V_xc batch
//! product with the host twin of the device kernel (`mixed_host`: f32
//! operands, f32 panels of 64 summed in f64), so the effect of the kernel on
//! converged energies and on SCF convergence is measured without a device.
//! The device kernel's own agreement with this twin is `ferric-dft`'s
//! `gpu_xc_mixed.rs`.
//!
//! `measured_error_map` prints the table on the validation page
//! (`cargo test -p ferric-scf --features test-seams --test
//! xc_mixed_vxc_error_map -- --ignored --nocapture`).
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::ks::MIXED_VXC_BUILDS;
use ferric_dft::xc_batch::{FORCE_HOST, MIXED_VXC_HOST_TWIN};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
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
