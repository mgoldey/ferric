//! The free-atom volume (the TS / MBD@rsSCS ratio denominator) is a property
//! of the ISOLATED atom: the molecule's environment must not reach it.
//!
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --test free_atom_volume_environment

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::live_free_atom_volume;
use ferric_scf::rhf::RhfConfig;

fn base() -> RhfConfig {
    RhfConfig {
        max_iter: 200,
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    }
}

/// A +1 point charge 3 Bohr from the origin (where the free atom is solved)
/// and a uniform field: an environment that strongly polarizes an atom there.
fn embedded() -> RhfConfig {
    RhfConfig {
        external_potential: Some(ExternalPotential {
            point_charges: vec![PointCharge {
                q: 1.0,
                x: 0.0,
                y: 0.0,
                z: 3.0,
            }],
            smeared_charges: Vec::new(),
            field: Some([0.0, 0.0, 0.02]),
        }),
        cosmo: Some(Default::default()),
        ..base()
    }
}

/// Catches: the environment leaking into the free-atom SCF (external
/// potential, COSMO). Bit-identical, because the stripped configuration IS
/// the vacuum one. O (open-shell triplet, UHF path) and Ne (closed-shell, RHF
/// path), in 6-31G: a minimal basis has no virtuals for Ne, so its density
/// cannot polarize and the test would pass vacuously (measured: 2e-16).
#[test]
fn free_atom_volume_ignores_the_molecular_environment() {
    let ctx = ParallelContext::default();
    let bs = basis::bundled("6-31g").expect("6-31g");
    let op = Operator::coulomb();
    for z in [8usize, 10] {
        let vac = live_free_atom_volume(&ctx, z, &bs, op, &base()).expect("vacuum");
        let emb = live_free_atom_volume(&ctx, z, &bs, op, &embedded()).expect("embedded");
        assert_eq!(
            vac.to_bits(),
            emb.to_bits(),
            "Z={z}: free volume {vac} (vacuum) vs {emb} (molecule embedded)"
        );
    }
}

/// Negative control: the environment above is strong enough that an atom
/// really solved inside it has a measurably different volume, so the test
/// above cannot pass by the environment being too weak to matter. Uses the
/// free-atom SCF directly, with the environment applied.
#[test]
fn the_environment_would_change_the_volume_if_applied() {
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_rpa::properties::atomic_effective_volumes_hirshfeld;
    use ferric_scf::rhf::solve_rhf;
    use ferric_scf::screening::SchwarzBounds;

    let ctx = ParallelContext::default();
    let bs = basis::bundled("6-31g").expect("6-31g");
    let op = Operator::coulomb();
    let mol = Molecule::parse_xyz("1\nNe\nNe 0 0 0\n", 0, 1).expect("Ne");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let bounds = SchwarzBounds::compute(op, &prep).expect("bounds");
    let vol = |cfg: &RhfConfig| {
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, cfg).expect("scf");
        assert!(r.converged);
        atomic_effective_volumes_hirshfeld(&mol, &bs, r.density_r(), None).expect("vol")[0]
    };
    let vac = vol(&base());
    let emb = vol(&embedded());
    let rel = (emb - vac).abs() / vac;
    println!("Ne/6-31G free volume: vacuum {vac:.10}, embedded {emb:.10}, rel {rel:.3e}");
    assert!(
        rel > 1e-6,
        "environment changed the volume by only {rel:.3e}"
    );
}
