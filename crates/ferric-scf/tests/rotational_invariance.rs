//! Rotational invariance of the total SCF energy.
//!
//! Physics: the electronic Hamiltonian depends only on internal (relative)
//! geometry — interatomic distances and angles — never on the molecule's
//! orientation in the lab frame. Rigidly rotating every atom about a common
//! origin must therefore leave the total energy unchanged. This is the
//! rotational analogue of the translational-invariance checks already in
//! this directory (`uks_b3lyp_h2o_ccpvdz_translational_invariance` in
//! `dft_grid_response.rs`, `qm_and_site_gradients_sum_to_zero_translational_invariance`
//! in `qmmm_polarizable.rs`), which shift atoms by a common vector instead of
//! rotating them. A bug that leaked lab-frame coordinates into the energy
//! (e.g. an angular-momentum-dependent integral evaluated in the wrong frame,
//! or a basis-function phase convention tied to a fixed axis) would show up
//! here as a nonzero energy drift under rotation but would not show up under
//! translation.
//!
//! Rotation choice: exact 90° about the z-axis, (x, y, z) -> (-y, x, z). This
//! is chosen deliberately over an arbitrary-angle rotation because it is a
//! pure coordinate permutation/sign-flip — every rotated coordinate is exactly
//! representable in f64 (no sine/cosine round-off is introduced by the
//! rotation itself), so any residual energy difference is attributable to the
//! SCF/integral code, not to the test's own arithmetic. The matrix
//! R = [[0,-1,0],[1,0,0],[0,0,1]] is exactly orthogonal (R^T R = I bit-for-bit
//! for this permutation form).
//!
//! Method choice: plain RHF (no XC functional, no DFT quadrature grid). A
//! DFT grid is built from atom-centered radial/angular shells and is only
//! invariant to within the quadrature's own numerical noise floor (see the
//! ~1e-10 residuals documented in `dft_grid_response.rs`), which would muddy
//! a rotation check. RHF's energy comes entirely from analytic one- and
//! two-electron integrals (overlap, kinetic, nuclear attraction, ERIs) plus
//! the nuclear repulsion sum, all of which are exact functions of interatomic
//! distances alone, so RHF is the cleanest possible probe of rotational
//! invariance.

use ferric_core::mol::{Atom, Molecule};
use ferric_core::basis;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
        0,
        1,
    )
    .unwrap()
}

/// Rigidly rotate every atom 90 degrees about the z-axis: (x, y, z) -> (-y, x, z).
fn rotate_90_about_z(mol: &Molecule) -> Molecule {
    let atoms = mol
        .atoms
        .iter()
        .map(|a| Atom {
            symbol: a.symbol.clone(),
            z: a.z,
            x: -a.y,
            y: a.x,
            zpos: a.zpos,
            ghost: a.ghost,
            n_core_ecp: a.n_core_ecp,
        })
        .collect();
    Molecule { atoms, charge: mol.charge, multiplicity: mol.multiplicity }
}

fn rhf_energy(mol: &Molecule) -> f64 {
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    };
    let res = solve_rhf(&ParallelContext::default(), mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(res.converged, "RHF must converge for the rotational-invariance check to be meaningful");
    res.energy
}

/// RHF/STO-3G total energy of water must be unchanged by a rigid 90-degree
/// rotation about the z-axis.
///
/// Tolerance: 1e-10 Ha. This test uses NO density-fitting (exact 4-index
/// ERIs) and NO DFT grid, so both orientations are computed through
/// bit-for-bit the same analytic integral code, only fed permuted/sign-flipped
/// input coordinates. With energy_conv=1e-11 / density_conv=1e-9 the SCF
/// itself resolves the energy far tighter than 1e-10, so the only source of
/// residual drift is genuine floating-point summation-order noise from the
/// permuted coordinates propagating through the integral and SCF code paths —
/// expected to sit many orders below 1e-10. A bug that mixed lab-frame axes
/// into the Hamiltonian (e.g. a hard-coded z-axis somewhere that should have
/// been rotation-covariant) would blow this bound by 6+ orders of magnitude,
/// just like the un-corrected translational-invariance cases documented
/// elsewhere in this directory (~1e-4 to ~1e-3 drift without the relevant fix).
#[test]
fn rhf_h2o_sto3g_rotational_invariance() {
    let mol = water();
    let rotated = rotate_90_about_z(&mol);

    // Sanity: the rotation must actually move the atoms (otherwise this test
    // would trivially pass no matter what).
    let moved = mol
        .atoms
        .iter()
        .zip(rotated.atoms.iter())
        .any(|(a, b)| (a.x - b.x).abs() > 1e-6 || (a.y - b.y).abs() > 1e-6);
    assert!(moved, "rotation must actually displace atomic coordinates");

    let e0 = rhf_energy(&mol);
    let e1 = rhf_energy(&rotated);
    let drift = (e0 - e1).abs();
    eprintln!("RHF/STO-3G H2O: E(unrotated) = {e0:.12}, E(rotated 90 deg about z) = {e1:.12}, drift = {drift:.3e}");

    assert!(
        drift < 1e-10,
        "RHF energy must be invariant under rigid rotation: drift = {drift:.3e} (want < 1e-10)"
    );
}
