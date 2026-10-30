//! Precision provenance on `ScfResult::precision` (plan rank 1, the first
//! certificate field). CPU only, no device and no installed GPU settings.
//!
//! * An all-f64 SCF (RHF, UHF, RKS) carries the explicit f64 record and no
//!   kernel entry.
//! * A CPU mixed run is recorded as one: COSX with the `f32_route` test seam
//!   (the CPU `md3c1e` f32 block, not user-reachable in this build: wiring it
//!   to `[precision]` is Task D6) reports `cosx-kern`, `mode = mixed`, on the
//!   CPU, with one mixed call per K build that computed f32 units, and
//!   `requested = f64` because no setting asked for it. The seam-off run of
//!   the same SCF reports f64 and leaves the energy bits unchanged by the
//!   record (the record observes, it never participates).
//!
//! The counters are process-wide, so every test here holds `SERIAL`.
//!
//! Artifact hypothesis: a record built from counters that are never written
//! (a missing hook) reads f64 for the seam-on run as well; a record that is
//! never attached (a missing probe in the solver) reads f64 everywhere. Both
//! fail `cosx_f32_route_is_recorded_as_cpu_mixed_cosx_kern`.
use std::sync::Mutex;

use ferric_core::basis::bundled;
use ferric_core::gpu::provenance::{ErrorBound, KernelDevice, PrecisionRecord};
use ferric_core::gpu::{MixedKernel, Precision};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::md3c1e::PrimPairSum;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::CosxConfig;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

static SERIAL: Mutex<()> = Mutex::new(());

const WATER: &str = "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
const OH: &str = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";

fn rhf(xyz: &str, basis: &str, cfg: &RhfConfig) -> ScfResult {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &bundled(basis).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    solve_rhf(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        cfg,
    )
    .expect("rhf")
}

#[test]
fn f64_scf_results_carry_the_explicit_f64_record() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let r = rhf(WATER, "sto-3g", &RhfConfig::default());
    assert!(r.converged);
    assert_eq!(r.precision, PrecisionRecord::f64());
    assert_eq!(
        r.precision.to_json(),
        serde_json::json!({"precision": "f64", "kernels": []})
    );

    let ks = rhf(
        WATER,
        "sto-3g",
        &RhfConfig {
            xc: Some("LDA".into()),
            ..Default::default()
        },
    );
    assert!(ks.converged);
    assert_eq!(ks.precision, PrecisionRecord::f64(), "RKS");

    let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
    let prep = PreparedBasis::new(&mol, &bundled("sto-3g").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let u = solve_uhf(
        &ParallelContext::default(),
        &mol,
        &prep,
        &bounds,
        &RhfConfig::default(),
    )
    .expect("uhf");
    assert_eq!(u.precision, PrecisionRecord::f64(), "UHF");
}

fn cosx_cfg(f32_route: Option<PrimPairSum>) -> RhfConfig {
    RhfConfig {
        k_builder: Some("cosx".into()),
        cosx: CosxConfig {
            grid: AtomicGridConfig {
                n_radial: 30,
                n_angular: 110,
                ..Default::default()
            },
            fp64_multiplier: 1e5,
            f32_route,
            ..CosxConfig::flat_reference()
        },
        ..Default::default()
    }
}

#[test]
fn cosx_f32_route_is_recorded_as_cpu_mixed_cosx_kern() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    // Seam off: the router classifies but every unit is f64, so nothing ran
    // mixed and no setting asked for it: the plain f64 record.
    let off = rhf(WATER, "def2-svp", &cosx_cfg(None));
    assert!(off.converged);
    assert_eq!(off.precision, PrecisionRecord::f64(), "seam off");

    let on = rhf(WATER, "def2-svp", &cosx_cfg(Some(PrimPairSum::F32)));
    assert!(on.converged);
    let p = &on.precision;
    assert_eq!(p.precision, Precision::Mixed, "{p:?}");
    assert_eq!(p.kernels.len(), 1, "{p:?}");
    let k = &p.kernels[0];
    assert_eq!(k.kernel, MixedKernel::CosxKern);
    assert_eq!(k.mode, Precision::Mixed);
    assert_eq!(
        k.requested,
        Precision::F64,
        "no setting asked for cosx-kern"
    );
    assert_eq!(k.device, KernelDevice::Cpu);
    // One K build per SCF iteration (plus at most a final rebuild); every one
    // computes f32 units on water/SVP at multiplier 1e5.
    assert!(
        k.calls_mixed >= on.iterations as u64 - 1 && k.calls_mixed <= on.iterations as u64 + 2,
        "calls_mixed {} vs {} iterations",
        k.calls_mixed,
        on.iterations
    );
    assert_eq!(
        k.calls_f64_fallback,
        [0, 0, 0],
        "no f32-range unit on water/SVP"
    );
    assert_eq!(k.ladder, None);
    assert!(matches!(k.error_bound, ErrorBound::NoneComputed { .. }));
    // The f32 path is live: the energy moved (by a little).
    assert_ne!(on.energy.to_bits(), off.energy.to_bits());
    assert!((on.energy - off.energy).abs() < 1e-6);
    // JSON round trip of every field.
    let j = p.to_json();
    assert_eq!(j["precision"], "mixed");
    let jk = &j["kernels"][0];
    assert_eq!(jk["kernel"], "cosx-kern");
    assert_eq!(jk["requested"], "f64");
    assert_eq!(jk["mode"], "mixed");
    assert_eq!(jk["device"], "cpu");
    assert_eq!(jk["calls_mixed"], k.calls_mixed);
    assert_eq!(
        jk["calls_f64_fallback"],
        serde_json::json!({"device_f64": 0, "f32_range": 0, "cpu_f64": 0})
    );
    assert!(jk["ladder"].is_null());
    assert_eq!(jk["error_bound"]["kind"], "none_computed");
    assert!(jk["error_bound"]["why"]
        .as_str()
        .is_some_and(|w| !w.is_empty()));
    let keys: Vec<&String> = jk.as_object().unwrap().keys().collect();
    assert_eq!(
        keys,
        [
            "calls_f64_fallback",
            "calls_mixed",
            "device",
            "error_bound",
            "kernel",
            "ladder",
            "mode",
            "requested"
        ],
        "the kernel record's schema"
    );

    // A later f64 run in the same process starts its own probe: f64 again.
    let again = rhf(WATER, "def2-svp", &cosx_cfg(None));
    assert_eq!(again.precision, PrecisionRecord::f64());
    assert_eq!(again.energy.to_bits(), off.energy.to_bits());
}
