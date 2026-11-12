#![cfg(feature = "gpu")]
//! `[gpu] mode = auto` takes the device RI-J and DF-K builds only above the
//! measured size gate (`ferric_core::gpu::gate`), and never takes RI-J for a
//! pure functional (whose J build overlaps the XC digest on the CPU).
//!
//! Anchors, in the order they are asserted:
//! * Below the gate, `auto` is the CPU run: no device build, and the host
//!   tensors keep the CPU default layout (f32 RI-J and RI-K), which is what
//!   `off` stores. A device-taking layout would be f64.
//! * Above the gate, a hybrid reaches the device DF-K; a pure functional keeps
//!   J on the CPU and the J || XC overlap runs.
use ferric_core::basis;
use ferric_core::gpu::{install, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_j::RiJPrecision;
use ferric_scf::df_k::RiKPrecision;
use ferric_scf::rhf::{solve_rhf, RhfConfig, JXC_OVERLAP_BUILDS};
use ferric_scf::screening::SchwarzBounds;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

static SERIAL: Mutex<()> = Mutex::new(());

const WATER: &str = "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n";

/// `n` waters on a cubic lattice, 3.2 A apart.
fn waters(n: usize) -> String {
    let mut s = format!("{}\ncluster\n", 3 * n);
    for i in 0..n {
        let (x, y, z) = (
            (i % 3) as f64 * 3.2,
            ((i / 3) % 3) as f64 * 3.2,
            (i / 9) as f64 * 3.2,
        );
        s += &format!("O {} {} {}\n", x, y, z);
        s += &format!("H {} {} {}\n", x, y + 0.7572, z + 0.5868);
        s += &format!("H {} {} {}\n", x, y - 0.7572, z + 0.5868);
    }
    s
}

fn setup() -> bool {
    if !matches!(probe(0), GpuStatus::Ready(_)) {
        eprintln!("skipping: no CUDA device");
        assert!(
            std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"),
            "FERRIC_GPU_TESTS_REQUIRED=1 but no device"
        );
        return false;
    }
    // `install` is process-wide; every test here wants auto.
    let _ = install(GpuSettingsExplicit {
        mode: Some(GpuMode::Auto),
        ..Default::default()
    });
    true
}

struct Run {
    d_dfj: u64,
    d_dfk: u64,
    d_overlap: usize,
    ri_j: RiJPrecision,
    ri_k: RiKPrecision,
    nbf: usize,
}

fn run(xyz: &str, basis_name: &str, xc: &str, k_aux: bool) -> Run {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: k_aux.then(|| "def2-universal-jkfit".into()),
        density_conv: 1e-6,
        max_iter: 3,
        ..Default::default()
    };
    let (a, o0) = (stats(), JXC_OVERLAP_BUILDS.load(Ordering::Relaxed));
    let r = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
    let (b, o1) = (stats(), JXC_OVERLAP_BUILDS.load(Ordering::Relaxed));
    let route = r.df_jk.expect("density-fitted run");
    Run {
        d_dfj: b.dfj_device_builds - a.dfj_device_builds,
        d_dfk: b.dfk_device_builds - a.dfk_device_builds,
        d_overlap: o1 - o0,
        ri_j: route.ri_j_precision,
        ri_k: route.ri_k_precision,
        nbf: prep.nbasis(),
    }
}

#[test]
fn below_the_gate_auto_is_the_cpu_run() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !setup() {
        return;
    }
    for (xc, k_aux) in [("PBE", false), ("B3LYP", true)] {
        let r = run(WATER, "def2-svp", xc, k_aux);
        assert!(r.nbf < ferric_core::gpu::gate::AUTO_MIN_NBF_DF_K);
        assert_eq!((r.d_dfj, r.d_dfk), (0, 0), "{xc}: a device build ran");
        assert_eq!(
            r.ri_j,
            RiJPrecision::F32,
            "{xc}: RI-J layout is not the CPU default"
        );
        if k_aux {
            assert_eq!(
                r.ri_k,
                RiKPrecision::DEFAULT,
                "{xc}: RI-K layout is not the CPU default"
            );
        }
    }
}

#[test]
fn above_the_gate_a_hybrid_takes_dfk_and_a_pure_functional_keeps_j_and_the_overlap() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !setup() {
        return;
    }
    let xyz = waters(9);
    let h = run(&xyz, "cc-pvtz", "B3LYP", true);
    assert!(
        h.nbf >= ferric_core::gpu::gate::AUTO_MIN_NBF_DF_K,
        "nbf {}",
        h.nbf
    );
    assert!(
        h.d_dfk > 0,
        "hybrid above the gate never reached the device DF-K"
    );
    let p = run(&xyz, "cc-pvtz", "PBE", false);
    assert_eq!(
        p.d_dfj, 0,
        "a pure functional's J went to the device under auto"
    );
    assert!(
        p.d_overlap > 0,
        "the J || XC overlap did not run under auto"
    );
    assert_eq!(p.ri_j, RiJPrecision::F32);
}
