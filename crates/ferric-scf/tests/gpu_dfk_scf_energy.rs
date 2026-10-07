#![cfg(feature = "gpu")]
//! SCF energy with the device DF-K vs the CPU DF-K, same binary, same config.
//!
//! Both arms run their own SCF, so the comparison is the converged energy; the
//! iteration counts and `converged` flags are printed (the flags of the host and
//! device arms are also asserted true), and so is the DIIS trajectory's outcome,
//! never its shape. The device and CPU K differ only by rounding (derived bound in
//! `gpu_dfk_resident.rs`), so any difference of the converged energies is
//! SCF-trajectory noise on the DF-JK floor (`dfk_occ_path.rs`: ~1.1e-8 at
//! water/cc-pVDZ), not algebra. A dropped RHF factor 2 or a swapped alpha/beta
//! channel would sit at mHa, far above both sides below.
//!
//! The tolerance is the geometric midpoint of two measured sides on the same
//! systems: (A) defect absent, |E_dev - E_host|; (B) defect present,
//! |E_dev(B rounded through f32) - E_host|, produced through the
//! `ROUND_B_TO_F32` seam. Pre-registered: B must exceed A by `MIN_SEPARATION`.
//! `ENERGY_TOL = sqrt(max_systems A * min_systems B)` over the three default
//! systems.
//!
//! The host arm forces BOTH the DF-K and the RI-J (`df_j_gpu`) onto the CPU, so side
//! A carries the device RI-J difference as well; re-measured with RI-J on the
//! device: A = 1.4e-14 (water), 0 (benzene, OH), within the table below.
//!
//! MEASURED SIDES (RHF/UHF, both cards identical; every arm converged in 12-13
//! iterations, iteration counts equal across arms):
//!   system                    A (clean)   B (f32-B)   B/A
//!   RHF water/cc-pVDZ         4.3e-14     4.6e-8      1.1e6
//!   RHF benzene/def2-SVP      0 (equal)   4.9e-7      infinite
//!   UHF OH/cc-pVDZ            5.7e-14     2.3e-7      4.0e6
//! max A = 5.7e-14 (about one ulp of the energy), min B = 4.6e-8, so
//! ENERGY_TOL = sqrt(5.7e-14 * 4.6e-8) = 5.1e-11, written as 5e-11; the lowest
//! separation is 8e5, against the pre-registered 10. The two `#[ignore]` systems,
//! not used to derive the tolerance, measured A 9.7e-13 / B 5.1e-7 (benzene/aug-cc-pVTZ)
//! and A 6.8e-13 / B 2.5e-8 (alkane_8/def2-TZVP), on both cards identically. The tolerance sits below the
//! ~1e-8 DF-JK floor quoted above because the two SCF runs here differ only by
//! K rounding at 1e-14 relative, and a converged energy is second-order in the
//! K error: this is a measured fact of these systems, not a bound.
//!
//! The mode-off bit-identity of the CPU path is NOT tested here: the CPU-vs-
//! forced-host equality is not an independent reference, so it rests on the
//! existing pins (`df_k::` bit-identity unit tests, `dfk_occ_path`,
//! `gpu_dfk_mode_off`).
use ferric_core::basis;
use ferric_core::gpu::{install, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_j_gpu::FORCE_HOST as FORCE_HOST_J;
use ferric_scf::df_k_gpu::{FORCE_HOST, ROUND_B_TO_F32};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, UhfConfig};
use std::sync::atomic::Ordering;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// sqrt(max_A * min_B) over the three default systems (module docs), one
/// significant figure.
const ENERGY_TOL: f64 = 5e-11;
/// Pre-registered: the defect side must exceed the clean side by at least this
/// factor; otherwise the SCF-level gate cannot discriminate and the K-level gate
/// (`gpu_dfk_resident.rs`) stands alone. The tolerance is never widened.
const MIN_SEPARATION: f64 = 10.0;

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

#[derive(Clone, Copy)]
enum Arm {
    Device,
    Host,
    DeviceRoundedB,
}

fn set(arm: Arm) {
    FORCE_HOST.store(matches!(arm, Arm::Host), Ordering::SeqCst);
    FORCE_HOST_J.store(matches!(arm, Arm::Host), Ordering::SeqCst);
    ROUND_B_TO_F32.store(matches!(arm, Arm::DeviceRoundedB), Ordering::SeqCst);
}

/// (energy, iterations, converged); a plateau on the DF-JK noise floor is
/// reported like a converged run (as `dfk_occ_path.rs` does), flag printed.
type Run = (f64, usize, bool);

fn rhf(mol: &Molecule, prep: &PreparedBasis, arm: Arm) -> Run {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, prep).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-3, // a loose "not still descending" bound; dP is the real gate
        density_conv: 1e-8,
        max_iter: 200,
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    set(arm);
    let r = solve_rhf(&ParallelContext::default(), mol, prep, op, &bounds, &cfg);
    set(Arm::Device);
    finish(r, cfg.max_iter)
}

fn uhf(mol: &Molecule, prep: &PreparedBasis, arm: Arm) -> Run {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, prep).unwrap();
    let cfg = UhfConfig {
        energy_conv: 1e-3,
        density_conv: 1e-8,
        max_iter: 300,
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    set(arm);
    let r = solve_uhf(&ParallelContext::default(), mol, prep, &bounds, &cfg);
    set(Arm::Device);
    finish(r, cfg.max_iter)
}

fn finish(r: Result<ferric_scf::ScfResult, FerricError>, max_iter: usize) -> Run {
    match r {
        Ok(res) => (res.energy, res.iterations, res.converged),
        Err(FerricError::ScfConvergence { last_energy, .. }) => (last_energy, max_iter, false),
        Err(e) => panic!("unexpected SCF error: {e:?}"),
    }
}

/// Run the three arms and apply the two-sided gate; returns (d_clean, d_defect).
fn gate(label: &str, run: impl Fn(Arm) -> Run) -> (f64, f64) {
    let s0 = stats();
    let (e_host, it_h, c_h) = run(Arm::Host);
    let s1 = stats();
    assert_eq!(
        s1.dfk_device_builds, s0.dfk_device_builds,
        "{label}: the forced-host arm ran the device DF-K"
    );
    let (e_dev, it_d, c_d) = run(Arm::Device);
    let s2 = stats();
    assert!(
        s2.dfk_device_builds > s1.dfk_device_builds,
        "{label}: the device arm never ran the device DF-K (declined: {}, pool-full fallbacks: {}); a vacuous pass is not a pass",
        s2.dfk_declined - s1.dfk_declined,
        s2.gemm_cpu_pool_full - s1.gemm_cpu_pool_full
    );
    assert!(
        s2.resident_uploads > s1.resident_uploads
            && s2.bytes_h2d > s1.bytes_h2d
            && s2.bytes_d2h > s1.bytes_d2h,
        "{label}: the device arm moved no bytes"
    );
    let (e_bad, it_b, c_b) = run(Arm::DeviceRoundedB);
    let s3 = stats();
    assert!(
        s3.dfk_device_builds > s2.dfk_device_builds,
        "{label}: the f32-B arm never ran the device DF-K"
    );
    let (d_clean, d_defect) = ((e_dev - e_host).abs(), (e_bad - e_host).abs());
    eprintln!(
        "{label}: host {e_host:.12} ({it_h} it, conv {c_h}); device {e_dev:.12} ({it_d} it, conv {c_d}); f32-B {e_bad:.12} ({it_b} it, conv {c_b})"
    );
    eprintln!(
        "{label}: device builds {}, resident uploads {}, h2d {} B, d2h {} B",
        s2.dfk_device_builds - s1.dfk_device_builds,
        s2.resident_uploads - s1.resident_uploads,
        s2.bytes_h2d - s1.bytes_h2d,
        s2.bytes_d2h - s1.bytes_d2h
    );
    eprintln!(
        "{label}: |dE| clean {d_clean:.3e}  defect {d_defect:.3e}  separation {:.3e}",
        d_defect / d_clean
    );
    assert!(
        c_h && c_d,
        "{label}: converged flags host {c_h} device {c_d}"
    );
    assert!(
        d_clean <= ENERGY_TOL,
        "{label}: device vs CPU SCF energy differ by {d_clean:e}"
    );
    assert!(
        d_defect > ENERGY_TOL,
        "{label}: the f32-B mutant ({d_defect:e}) does not exceed the tolerance: the gate has no teeth"
    );
    assert!(
        d_defect >= MIN_SEPARATION * d_clean,
        "{label}: separation {:.2} below the pre-registered {MIN_SEPARATION}",
        d_defect / d_clean
    );
    (d_clean, d_defect)
}

fn load(xyz: &str, basis_name: &str) -> (Molecule, PreparedBasis) {
    let mol = Molecule::load_xyz(&format!("../../testdata/molecules/{xyz}")).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    (mol, prep)
}

#[test]
fn rhf_water_ccpvdz() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mol, prep) = load("water.xyz", "cc-pvdz");
    gate("RHF water/cc-pVDZ", |a| rhf(&mol, &prep, a));
}

#[test]
fn rhf_benzene_def2svp() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mol, prep) = load("benzene.xyz", "def2-svp");
    gate("RHF benzene/def2-SVP", |a| rhf(&mol, &prep, a));
}

#[test]
fn uhf_oh_ccpvdz_two_channels() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    gate("UHF OH/cc-pVDZ (alpha 5, beta 4)", |a| uhf(&mol, &prep, a));
}

/// Heavier systems. `#[ignore]` states a PRECONDITION (serial run on a quiet box;
/// minutes of host integral work; 0.8 / 0.9 GB of device memory), not a skip.
#[test]
#[ignore = "precondition: run serially on a quiet box; minutes of host work"]
fn rhf_benzene_augccpvtz() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mol, prep) = load("benzene.xyz", "aug-cc-pvtz");
    gate("RHF benzene/aug-cc-pVTZ (B 765 MB)", |a| {
        rhf(&mol, &prep, a)
    });
}

#[test]
#[ignore = "precondition: run serially on a quiet box; minutes of host work"]
fn rhf_alkane8_def2tzvp() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (mol, prep) = load("alkane_8.xyz", "def2-tzvp");
    gate("RHF alkane_8/def2-TZVP (B 937 MB)", |a| rhf(&mol, &prep, a));
}
