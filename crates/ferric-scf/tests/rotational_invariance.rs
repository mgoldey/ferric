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

use ferric_core::basis;
use ferric_core::mol::{Atom, Molecule};
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
    Molecule {
        atoms,
        charge: mol.charge,
        multiplicity: mol.multiplicity,
    }
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
    assert!(
        res.converged,
        "RHF must converge for the rotational-invariance check to be meaningful"
    );
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

/// H2O⁺ doublet — an OPEN-SHELL system, on the DEFAULT guess.
fn water_cation() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O+\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
        1,
        2,
    )
    .unwrap()
}

/// UHF total energy on the DEFAULT config — i.e. `use_sad_guess` left alone.
///
/// Pinning the guess here would defeat the entire purpose: the defect this
/// test exists to catch lives IN the default (MINAO) guess.
fn uhf_energy_default_guess(mol: &Molecule) -> (f64, bool) {
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = ferric_scf::uhf::UhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    };
    let res =
        ferric_scf::uhf::solve_uhf(&ParallelContext::default(), mol, &prep, &bounds, &cfg).unwrap();
    (res.energy, res.converged)
}

/// **OPEN-SHELL rotational invariance, on the path real callers use.**
///
/// # Why this test exists (2026-09-17)
///
/// `rhf_h2o_sto3g_rotational_invariance` above covers the CLOSED-shell path
/// only, and closed-shell atoms are spherical, so it cannot see this defect.
///
/// `fix/scf-unconstrained-state-selection` made `use_sad_guess` take effect for
/// open-shell SCF (default MINAO). For a light atom, `guess::free_atom_density`
/// builds the MINAO block by running a live **UHF + MOM** solve on the free
/// atom. On a free O atom the hcore 2p shell is EXACTLY degenerate, so which 2p
/// orbital receives the β hole is decided by whatever the eigensolver returns
/// on a degenerate block — i.e. by rounding — and MOM then pins that choice.
/// The resulting atomic density block is non-spherical AND oriented in the LAB
/// FRAME, so the molecular guess, and the state the SCF converges to, depend on
/// how the molecule happens to be oriented.
///
/// MEASURED before the fix, H2O⁺/STO-3G UHF, a pure x↔y coordinate swap (the
/// same physical molecule):
///
/// ```text
///   xz plane:  E = -74.6581025896   (the ground doublet)
///   yz plane:  E = -74.5758683062   (0.0822 Ha = 2.24 eV HIGHER)
/// ```
///
/// Both converged. The second is the ²A₁ excited state: a converged, wrong
/// answer on the default path, selected by nothing but lab-frame orientation.
/// The same defect made CH3 quartet/STO-3G fail to converge at all in one
/// orientation while converging in three others.
///
/// The fix spherically averages the free atom's partially-occupied frontier
/// shell, exactly as `guess::atomic_hcore_density` already did for the
/// heavy-atom GWH path — whose own comment records the identical pathology
/// ("water/STO-3G PBE limit-cycled … when O's 2p SOMOs were placed on two of
/// the three 2p orbitals"). The light-atom path simply never received it.
///
/// # Why 90° about z is the right rotation here
///
/// It is a pure coordinate permutation with a sign flip, exactly representable
/// in f64, so it introduces no arithmetic of its own. For THIS defect it is
/// also maximally diagnostic: it exchanges the x and y axes, which is precisely
/// the degeneracy the free-atom 2p hole is choosing among.
#[test]
fn uhf_open_shell_rotational_invariance_on_the_default_guess() {
    let mol = water_cation();
    let rotated = rotate_90_about_z(&mol);

    let moved = mol
        .atoms
        .iter()
        .zip(rotated.atoms.iter())
        .any(|(a, b)| (a.x - b.x).abs() > 1e-6 || (a.y - b.y).abs() > 1e-6);
    assert!(moved, "rotation must actually displace atomic coordinates");

    let (e0, c0) = uhf_energy_default_guess(&mol);
    let (e1, c1) = uhf_energy_default_guess(&rotated);
    let drift = (e0 - e1).abs();
    eprintln!(
        "UHF/STO-3G H2O+ (default guess): E(unrotated) = {e0:.12} (conv {c0}), \
         E(rotated 90 deg about z) = {e1:.12} (conv {c1}), drift = {drift:.3e}"
    );

    // Non-convergence is its own failure mode for this defect (CH3 quartet),
    // so assert it explicitly rather than letting an unconverged energy be
    // compared as though it meant something.
    assert!(
        c0 && c1,
        "both orientations must converge; got conv({c0}, {c1}). A converged/\
         not-converged split under pure rotation is the SAME defect this test \
         targets, not a separate problem."
    );

    // 1e-10 Ha, same bar and same reasoning as the closed-shell case: exact
    // 4-index ERIs, no DFT grid, no density fitting, so only summation-order
    // noise should survive. The defect this guards against produced 8.2e-2 Ha
    // — eight orders of magnitude above this bound, not a tolerance question.
    assert!(
        drift < 1e-10,
        "open-shell UHF energy must be invariant under rigid rotation: drift = \
         {drift:.3e} (want < 1e-10). A drift of ~1e-1 Ha means the MINAO \
         free-atom block is lab-frame oriented and the two orientations \
         converged to DIFFERENT electronic states."
    );
}
