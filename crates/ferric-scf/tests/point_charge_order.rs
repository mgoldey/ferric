//! The converged RHF energy does not depend on the ORDER of the external point-charge list beyond
//! roundoff (#444).
//!
//! Hypotheses. Physics: the list is a set, so any permutation is the same Hamiltonian; the only
//! legitimate difference is the summation-order roundoff of `V` (~1e-15 relative, see the
//! ferric-integrals test of the same name). Artifact: if some SCF stage (guess, screening, the
//! external-potential fold) read the charges positionally, a permutation would move the converged
//! energy beyond the SCF tolerance, or change convergence. Iteration COUNTS are deliberately not
//! asserted: on a hard field they depend on how roundoff-sized perturbations interact with the
//! convergence accelerators (see #444), which is not a property of the Hamiltonian.

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn field(n: usize) -> Vec<PointCharge> {
    let mut s = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u, v, w) = (next() * 2.0 - 1.0, next() * 2.0 - 1.0, next() * 2.0 - 1.0);
            let norm = (u * u + v * v + w * w).sqrt().max(1e-3);
            let r = 5.0 + 30.0 * next();
            PointCharge {
                q: 0.8 * (next() * 2.0 - 1.0),
                x: r * u / norm,
                y: r * v / norm,
                z: r * w / norm,
            }
        })
        .collect()
}

#[test]
fn converged_energy_is_independent_of_the_charge_order() {
    let mol = Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
        0,
        1,
    )
    .unwrap();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let run = |charges: Vec<PointCharge>| {
        let cfg = RhfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-8,
            max_iter: 100,
            external_potential: Some(ExternalPotential {
                point_charges: charges,
                smeared_charges: Vec::new(),
                field: None,
            }),
            ..Default::default()
        };
        solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap()
    };
    let base_charges = field(1500);
    let base = run(base_charges.clone());
    assert!(base.converged);
    let mut rev = base_charges.clone();
    rev.reverse();
    let mut rot = base_charges.clone();
    rot.rotate_left(401);
    for (name, p) in [("reversed", rev), ("rotated", rot)] {
        let r = run(p);
        eprintln!(
            "{name}: converged {} iters {} (base {}) dE {:.2e}",
            r.converged,
            r.iterations,
            base.iterations,
            (r.energy - base.energy).abs()
        );
        assert!(r.converged, "{name} order did not converge");
        assert!(
            (r.energy - base.energy).abs() < 1e-9,
            "{name}: energy moved by {:e}",
            r.energy - base.energy
        );
    }
    // Opposite control: one charge dropped moves the energy far above the bar.
    let mut fewer = base_charges;
    fewer.pop();
    let dropped = (run(fewer).energy - base.energy).abs();
    eprintln!("one charge dropped: dE {dropped:.2e}");
    assert!(
        dropped > 1e-6,
        "a dropped charge ({dropped:e}) must be visible against the 1e-9 bar"
    );
}
