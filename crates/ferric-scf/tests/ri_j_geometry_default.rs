//! Geometry and frequency drivers resolve an UNSET `ri_j_precision` to the f64
//! RI-J tensor (Matt 2026-11-08): the f32 tensor re-rounds every integral when
//! an atom moves, so its energy surface jitters by ~1e-7..1e-6 Ha between
//! nearby geometries (site/src/methods/scf.md, "RI-J tensor precision"). An
//! explicit value always wins, and a single-point SCF keeps the f32 default.
//!
//! Observable: the energy at the starting geometry. PBE takes RI-J by default,
//! and the f32 and f64 tensors give energies that differ by ~1e-6 Ha, so the
//! driver's first energy equals, bit for bit, exactly one of the two
//! single-point references. Artifact hypothesis: if a driver ignored the task
//! (or the rule were broken), its unset run would equal the f32 reference.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test ri_j_geometry_default
//! ```

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_j::{resolve_ri_j_precision, RiJPrecision, RiJPrecisionSource, RiJTask};
use ferric_scf::frequencies::{harmonic_frequencies, FrequencyConfig};
use ferric_scf::optimize::{optimize_geometry, OptimizeConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const WATER: &str = "3\n\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
const BASIS: &str = "sto-3g";

fn water() -> Molecule {
    Molecule::parse_xyz(WATER, 0, 1).unwrap()
}

fn config(p: Option<RiJPrecision>) -> RhfConfig {
    RhfConfig {
        xc: Some("PBE".into()),
        energy_conv: 1e-10,
        density_conv: 1e-9,
        ri_j_precision: p,
        ..Default::default()
    }
}

/// A plain single-point SCF at the starting geometry.
fn single_point(p: Option<RiJPrecision>) -> f64 {
    let mol = water();
    let prep = PreparedBasis::new(&mol, &basis::bundled(BASIS).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let r = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &prep,
        op,
        &bounds,
        &config(p),
    )
    .unwrap();
    assert!(r.converged);
    r.energy
}

fn first_optimize_energy(p: Option<RiJPrecision>) -> f64 {
    let opt = OptimizeConfig {
        max_steps: 1,
        ..Default::default()
    };
    let r = optimize_geometry(
        &ParallelContext::default(),
        &water(),
        BASIS,
        Operator::coulomb(),
        &config(p),
        &opt,
    )
    .unwrap();
    r.energy_trace[0]
}

#[test]
fn the_rule_explicit_wins_geometry_is_f64_energy_is_f32() {
    use RiJPrecision::{F32, F64};
    for task in [RiJTask::Energy, RiJTask::Geometry] {
        for p in [F32, F64] {
            let r = resolve_ri_j_precision(task, Some(p));
            assert_eq!((r.precision, r.source), (p, RiJPrecisionSource::Explicit));
        }
    }
    let g = resolve_ri_j_precision(RiJTask::Geometry, None);
    assert_eq!(
        (g.precision, g.source),
        (F64, RiJPrecisionSource::GeometryTaskDefault)
    );
    #[cfg(not(feature = "gpu"))]
    {
        let e = resolve_ri_j_precision(RiJTask::Energy, None);
        assert_eq!((e.precision, e.source), (F32, RiJPrecisionSource::Default));
    }
    assert_eq!(RhfConfig::default().ri_j_task, RiJTask::Energy);
    let g = config(None).for_geometry_task();
    assert_eq!(g.ri_j_task, RiJTask::Geometry);
    assert_eq!(g.resolved_ri_j_precision().precision, F64);
    let g = config(Some(F32)).for_geometry_task();
    assert_eq!(g.resolved_ri_j_precision().precision, F32);
}

#[test]
fn optimize_and_frequencies_default_to_f64_and_honour_an_explicit_f32() {
    let sp_f64 = single_point(Some(RiJPrecision::F64));
    let sp_f32 = single_point(Some(RiJPrecision::F32));
    assert_ne!(sp_f64.to_bits(), sp_f32.to_bits(), "observable is blind");
    // A single point with nothing set keeps the f32 default.
    #[cfg(not(feature = "gpu"))]
    assert_eq!(single_point(None).to_bits(), sp_f32.to_bits());

    let opt_default = first_optimize_energy(None);
    assert_eq!(opt_default.to_bits(), sp_f64.to_bits(), "optimize default");
    let opt_f32 = first_optimize_energy(Some(RiJPrecision::F32));
    assert_eq!(opt_f32.to_bits(), sp_f32.to_bits(), "optimize explicit f32");

    let freq = |p| {
        harmonic_frequencies(
            &ParallelContext::default(),
            &water(),
            BASIS,
            Operator::coulomb(),
            &config(p),
            &FrequencyConfig::default(),
        )
        .unwrap()
        .energy
    };
    assert_eq!(
        freq(None).to_bits(),
        sp_f64.to_bits(),
        "frequencies default"
    );
    assert_eq!(
        freq(Some(RiJPrecision::F32)).to_bits(),
        sp_f32.to_bits(),
        "frequencies explicit f32"
    );
}
