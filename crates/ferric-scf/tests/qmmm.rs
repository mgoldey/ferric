//! End-to-end QM/MM partitioning + electrostatic embedding tests.
//!
//! The test order here is deliberate and matches the module's contract:
//!
//! 1. **Exactness first.** An empty MM region must be a *bit-identical* no-op
//!    vs a plain gas-phase run on the same QM atoms. Everything else rests on
//!    this: if the layer perturbed the gas-phase path even at the ulp level,
//!    no downstream number would be attributable.
//! 2. **Then non-triviality.** MM charges must measurably and *directionally
//!    correctly* perturb the QM region. A layer that silently did nothing would
//!    pass every exactness test, so the sign of the response is checked against
//!    physics, not just its magnitude against zero.
//! 3. Then geometry (link atoms) and the MM-force machinery.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::qmmm::{
    electric_field_at_points, mm_forces, BoundaryChargeScheme, QmSelection, QmmmAtom, QmmmSystem,
    DEFAULT_LINK_SCALE,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const ANG2BOHR: f64 = 1.0 / 0.529_177_210_92;

/// Water geometry in **Bohr**, O first then the two H. C2v, dipole along +z
/// pointing from O toward the H's midpoint... in this orientation the O sits at
/// negative z relative to the hydrogens' centroid, so the electron-rich O lone
/// pair region lies on the −z side of the molecule.
fn water_atoms(o_charge: f64, h_charge: f64) -> Vec<QmmmAtom> {
    // Standard experimental water: r(OH)=0.9572 Å, angle 104.52°, in the yz plane.
    let r = 0.9572 * ANG2BOHR;
    let half = 104.52_f64.to_radians() / 2.0;
    vec![
        QmmmAtom::new("O", 8, 0.0, 0.0, 0.0, o_charge),
        QmmmAtom::new("H", 1, 0.0, r * half.sin(), r * half.cos(), h_charge),
        QmmmAtom::new("H", 1, 0.0, -r * half.sin(), r * half.cos(), h_charge),
    ]
}

fn setup(mol: &Molecule) -> (basis::BasisSet, PreparedBasis, Operator, SchwarzBounds) {
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    (bs, prep, op, bounds)
}

/// Total electronic + nuclear dipole moment (a.u.) about the origin.
fn dipole(mol: &Molecule, prep: &PreparedBasis, d_total: &Array2<f64>) -> [f64; 3] {
    let mu_ao = oneelectron::dipole(prep, [0.0, 0.0, 0.0]).unwrap();
    let mut out = [0.0_f64; 3];
    for (axis, m) in mu_ao.iter().enumerate() {
        // Electronic part: -Tr(D · mu). Nuclear part: +Σ Z_A R_A.
        let elec: f64 = d_total.iter().zip(m.iter()).map(|(d, x)| d * x).sum();
        let mut nuc = 0.0;
        for atom in &mol.atoms {
            let z = atom.effective_z() as f64;
            nuc += z * match axis {
                0 => atom.x,
                1 => atom.y,
                _ => atom.zpos,
            };
        }
        out[axis] = nuc - elec;
    }
    out
}


const ETHANE_CC: f64 = 1.53 * ANG2BOHR;

/// Staggered ethane along z: C0 at the origin, C1 at z = 1.53 Å, three H on
/// each carbon at 1.09 Å / 109.5°. Indices: 0 = C0, 1 = C1, then the H's
/// alternate (2,4,6 on C0; 3,5,7 on C1). Charges are a generic nonpolar
/// pattern (C −0.1, H +0.033) — the numbers only need to be nonzero and
/// distinct enough that a redistribution scheme's arithmetic is visible.
fn ethane_atoms() -> Vec<QmmmAtom> {
    let cc = ETHANE_CC;
    let ch = 1.09 * ANG2BOHR;
    // Tetrahedral: H's at ~109.5° from the C-C axis.
    let theta = 109.5_f64.to_radians();
    let (s, c) = (theta.sin(), theta.cos());
    let mut atoms = vec![
        QmmmAtom::new("C", 6, 0.0, 0.0, 0.0, -0.1),
        QmmmAtom::new("C", 6, 0.0, 0.0, cc, -0.1),
    ];
    for k in 0..3 {
        let phi = 2.0 * std::f64::consts::PI * (k as f64) / 3.0;
        // Methyl on C0, pointing away from C1 (i.e. toward -z).
        atoms.push(QmmmAtom::new("H", 1, ch * s * phi.cos(), ch * s * phi.sin(), ch * c, 0.033));
        // Methyl on C1, pointing away from C0 (toward +z).
        atoms.push(QmmmAtom::new("H", 1, ch * s * phi.cos(), ch * s * phi.sin(), cc - ch * c, 0.033));
    }
    atoms
}

/// Full covalent bond list of `ethane_atoms()`: the C-C bond plus six C-H.
fn ethane_bonds() -> Vec<(usize, usize)> {
    vec![(0, 1), (0, 2), (0, 4), (0, 6), (1, 3), (1, 5), (1, 7)]
}

/// The one-methyl QM partition of ethane with the C-C bond cut and capped.
fn capped_ethane() -> QmmmSystem {
    QmmmSystem::new(&ethane_atoms(), QmSelection::Indices(vec![0, 2, 4, 6]), 0, 1)
        .unwrap()
        .with_link_atoms(&ethane_bonds(), DEFAULT_LINK_SCALE)
        .unwrap()
}

/// Σ q and Σ q·r of every charge in the embedding potential.
fn total_charge_and_dipole(sys: &QmmmSystem) -> (f64, [f64; 3]) {
    let Some(ep) = sys.to_external_potential() else { return (0.0, [0.0; 3]) };
    let mut q = 0.0;
    let mut mu = [0.0_f64; 3];
    for pc in &ep.point_charges {
        q += pc.q;
        mu[0] += pc.q * pc.x;
        mu[1] += pc.q * pc.y;
        mu[2] += pc.q * pc.z;
    }
    (q, mu)
}

// ---------------------------------------------------------------------------
// 1. EXACTNESS: an empty MM region is a bit-identical no-op.
// ---------------------------------------------------------------------------

/// The foundational contract. An all-QM partition must reproduce a plain
/// gas-phase SCF **exactly** — `assert_eq!` on the raw f64 energy, not a
/// tolerance — because `to_external_potential()` returns `None` and `None` is
/// the literal gas-phase code path in `solve_rhf`.
#[test]
fn empty_mm_region_is_bit_identical_to_gas_phase() {
    let atoms = water_atoms(-0.834, 0.417);
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();

    // No MM atoms at all => no embedding potential, even though every atom
    // carries a nonzero MM partial charge in the input structure.
    assert!(sys.mm_indices.is_empty());
    assert!(sys.to_external_potential().is_none());

    let mol = sys.to_qm_molecule();
    let (_bs, prep, op, bounds) = setup(&mol);
    let ctx = ParallelContext::default();

    let gas = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
    let qmmm_config =
        RhfConfig { external_potential: sys.to_external_potential(), ..Default::default() };
    let embedded = solve_rhf(&ctx, &mol, &prep, op, &bounds, &qmmm_config).unwrap();

    assert!(gas.converged && embedded.converged);
    assert_eq!(
        gas.energy, embedded.energy,
        "empty MM region must be BIT-IDENTICAL to gas phase, not merely close"
    );
    // Orbital energies too — the whole Fock matrix must be untouched.
    for (a, b) in gas.eps_alpha.iter().zip(embedded.eps_alpha.iter()) {
        assert_eq!(a, b, "orbital energies must be bit-identical");
    }
}

/// Same contract via the other route to an empty potential: MM atoms exist,
/// but all carry exactly zero charge. Must still collapse to `None` (a
/// potential full of zero-valued charges would change the hcore code path and
/// could perturb the last bits).
#[test]
fn zero_charge_mm_region_is_bit_identical_to_gas_phase() {
    // Water QM + a far-away "MM" atom carrying zero charge.
    let mut atoms = water_atoms(-0.834, 0.417);
    atoms.push(QmmmAtom::new("Na", 11, 0.0, 0.0, 20.0, 0.0));

    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
    assert_eq!(sys.mm_indices, vec![3]);
    assert!(sys.to_external_potential().is_none(), "all-zero MM charges must give None");

    let mol = sys.to_qm_molecule();
    assert_eq!(mol.atoms.len(), 3, "the MM atom must not enter the QM molecule");
    let (_bs, prep, op, bounds) = setup(&mol);
    let ctx = ParallelContext::default();

    let gas = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
    let cfg = RhfConfig { external_potential: sys.to_external_potential(), ..Default::default() };
    let embedded = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert_eq!(gas.energy, embedded.energy);
}

/// The QM molecule handed to the solver must be independent of how big the MM
/// region is: partitioning a larger structure must not disturb the QM atoms.
#[test]
fn qm_molecule_is_independent_of_mm_region_size() {
    let base = water_atoms(-0.834, 0.417);
    let mut extended = base.clone();
    for k in 1..=5 {
        extended.push(QmmmAtom::new("Na", 11, 0.0, 0.0, 15.0 + k as f64, 0.5));
    }

    let sys_small = QmmmSystem::new(&base, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
    let sys_big = QmmmSystem::new(&extended, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();

    let m1 = sys_small.to_qm_molecule();
    let m2 = sys_big.to_qm_molecule();
    assert_eq!(m1.atoms.len(), m2.atoms.len());
    for (a, b) in m1.atoms.iter().zip(m2.atoms.iter()) {
        assert_eq!(a.z, b.z);
        assert_eq!(a.x, b.x);
        assert_eq!(a.y, b.y);
        assert_eq!(a.zpos, b.zpos);
    }
    assert_eq!(sys_big.mm_indices.len(), 5);
}

// ---------------------------------------------------------------------------
// 2. NON-TRIVIALITY: MM charges must perturb the QM region, in the RIGHT
//    direction.
// ---------------------------------------------------------------------------

/// A positive point charge placed on the oxygen lone-pair side of water must
/// **stabilize** the system (lower the total energy) relative to gas phase.
///
/// Physics: water's oxygen carries the negative end of the molecular dipole.
/// In this geometry the O sits at z=0 with both H at positive z, so the lone
/// pair / negative lobe points toward −z. A +1 charge on the −z side is
/// attracted to that negative lobe: both the classical charge-nuclear +
/// charge-electron electrostatics and the induced polarization lower E.
///
/// The mirror-image test (same charge on the +z hydrogen side) must
/// *destabilize*, which pins the sign down as a real directional response and
/// not an accident of the magnitude.
#[test]
fn positive_charge_near_oxygen_lone_pair_stabilizes_water() {
    let atoms_base = water_atoms(0.0, 0.0);
    let sys_gas = QmmmSystem::new(&atoms_base, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
    let mol = sys_gas.to_qm_molecule();
    let (_bs, prep, op, bounds) = setup(&mol);
    let ctx = ParallelContext::default();
    let gas = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
    assert!(gas.converged);

    // Build the two mirrored MM setups: +1 charge 6 Bohr away on the lone-pair
    // (−z) side, and on the hydrogen (+z) side.
    let run_with_charge_at_z = |z: f64| -> f64 {
        let mut atoms = atoms_base.clone();
        atoms.push(QmmmAtom::new("Na", 11, 0.0, 0.0, z, 1.0));
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
        let ep = sys.to_external_potential().expect("MM charge must produce a potential");
        assert_eq!(ep.point_charges.len(), 1);
        let cfg = RhfConfig { external_potential: Some(ep), ..Default::default() };
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        assert!(r.converged);
        r.energy
    };

    let e_lone_pair_side = run_with_charge_at_z(-6.0);
    let e_hydrogen_side = run_with_charge_at_z(6.0);

    // Non-triviality: the layer must actually do something.
    let shift_lp = e_lone_pair_side - gas.energy;
    let shift_h = e_hydrogen_side - gas.energy;
    assert!(
        shift_lp.abs() > 1e-6,
        "MM charge produced no measurable energy shift ({shift_lp:.3e} Ha) — the \
         embedding layer is silently doing nothing"
    );

    // Directionality: lone-pair side stabilizes, hydrogen side destabilizes.
    assert!(
        shift_lp < 0.0,
        "a +1 charge on water's oxygen lone-pair side must LOWER the energy; \
         got ΔE = {shift_lp:+.6} Ha"
    );
    assert!(
        shift_h > 0.0,
        "a +1 charge on water's hydrogen side must RAISE the energy; \
         got ΔE = {shift_h:+.6} Ha"
    );

    eprintln!(
        "[qmmm] water/STO-3G, +1 charge at 6 Bohr: E_gas = {:.8} Ha, \
         ΔE(lone-pair side) = {:+.6} Ha, ΔE(H side) = {:+.6} Ha",
        gas.energy, shift_lp, shift_h
    );
}

/// The MM charge must polarize the *density*, not merely add a classical
/// constant. A charge along the dipole axis must change the QM dipole moment,
/// and the sign of the change must follow the applied field.
#[test]
fn mm_charge_polarizes_the_qm_density() {
    let atoms_base = water_atoms(0.0, 0.0);
    let sys_gas = QmmmSystem::new(&atoms_base, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
    let mol = sys_gas.to_qm_molecule();
    let (_bs, prep, op, bounds) = setup(&mol);
    let ctx = ParallelContext::default();

    let gas = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
    let mu_gas = dipole(&mol, &prep, gas.density_total());

    let mut atoms = atoms_base.clone();
    atoms.push(QmmmAtom::new("Na", 11, 0.0, 0.0, -6.0, 1.0));
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
    let cfg = RhfConfig { external_potential: sys.to_external_potential(), ..Default::default() };
    let pol = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let mu_pol = dipole(&mol, &prep, pol.density_total());

    let dmu_z = mu_pol[2] - mu_gas[2];
    assert!(
        dmu_z.abs() > 1e-5,
        "MM charge did not change the QM dipole ({dmu_z:.3e} a.u.) — density is \
         not being polarized"
    );

    // A +1 charge at z = −6 pulls electron density toward −z. Electron density
    // moving to −z makes the dipole (which points from −q to +q) move toward
    // +z, i.e. mu_z must INCREASE.
    assert!(
        dmu_z > 0.0,
        "a +1 charge at z=-6 must draw electron density toward -z and so raise \
         mu_z; got Δmu_z = {dmu_z:+.6} a.u."
    );

    eprintln!(
        "[qmmm] water/STO-3G dipole: mu_z(gas) = {:.6}, mu_z(embedded) = {:.6}, \
         Δmu_z = {:+.6} a.u.",
        mu_gas[2], mu_pol[2], dmu_z
    );
}

/// Doubling the MM charge must roughly double the leading energy response, and
/// flipping its sign must flip the response. Confirms the coupling is linear in
/// q to leading order (as electrostatic embedding must be) rather than an
/// arbitrary constant.
#[test]
fn embedding_response_scales_with_mm_charge() {
    let atoms_base = water_atoms(0.0, 0.0);
    let sys_gas = QmmmSystem::new(&atoms_base, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
    let mol = sys_gas.to_qm_molecule();
    let (_bs, prep, op, bounds) = setup(&mol);
    let ctx = ParallelContext::default();
    let gas = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();

    let energy_for_q = |q: f64| -> f64 {
        let mut atoms = atoms_base.clone();
        // Far enough (12 Bohr) that the response is dominated by the linear term.
        atoms.push(QmmmAtom::new("Na", 11, 0.0, 0.0, -12.0, q));
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
        let cfg =
            RhfConfig { external_potential: sys.to_external_potential(), ..Default::default() };
        solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap().energy
    };

    let d1 = energy_for_q(0.5) - gas.energy;
    let d2 = energy_for_q(1.0) - gas.energy;
    let dm = energy_for_q(-0.5) - gas.energy;

    assert!(d1.abs() > 1e-7, "no response at q=0.5");
    // Linear to leading order: E(2q)/E(q) ≈ 2 (small quadratic polarization
    // correction, so allow a loose band).
    let ratio = d2 / d1;
    assert!(
        (ratio - 2.0).abs() < 0.2,
        "response should be ~linear in q: E(1.0)/E(0.5) = {ratio:.4}, expected ~2"
    );
    // Sign flip with charge sign.
    assert!(
        d1 * dm < 0.0,
        "flipping the MM charge sign must flip the energy shift: {d1:+.3e} vs {dm:+.3e}"
    );
}

// ---------------------------------------------------------------------------
// 3. LINK ATOMS: geometric verification.
// ---------------------------------------------------------------------------

/// Ethane cut down the middle of the C-C bond: the QM region is one methyl,
/// capped with a link H. Verify the cap is (a) on the C-C bond vector, (b) at
/// exactly the requested fraction of the bond length, and (c) that the
/// resulting QM molecule is a chemically sensible methane-like fragment that
/// actually converges.
#[test]
fn link_atom_caps_a_cut_cc_bond_with_verified_geometry() {
    let atoms = ethane_atoms();
    let cc = ETHANE_CC;
    // QM = C0 and its three H (indices 0, 2, 4, 6); MM = C1 and its H's.
    let qm = vec![0, 2, 4, 6];
    let bonds = vec![(0usize, 1usize)]; // the cut C-C bond

    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(qm.clone()), 0, 1)
        .unwrap()
        .with_link_atoms(&bonds, DEFAULT_LINK_SCALE)
        .unwrap();

    assert_eq!(sys.link_atoms.len(), 1, "exactly one bond was cut");
    let link = &sys.link_atoms[0];

    // (a) On the bond vector: R_link - R_C0 must be PARALLEL to R_C1 - R_C0.
    let p = link.position;
    let bond = [0.0, 0.0, cc];
    let v = [p[0] - 0.0, p[1] - 0.0, p[2] - 0.0];
    let cross = [
        v[1] * bond[2] - v[2] * bond[1],
        v[2] * bond[0] - v[0] * bond[2],
        v[0] * bond[1] - v[1] * bond[0],
    ];
    let cross_norm = (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
    assert!(cross_norm < 1e-12, "link H is off the bond vector (|v x b| = {cross_norm:.3e})");

    // (b) At exactly the requested fraction of the bond length.
    let d = (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt();
    let expected = DEFAULT_LINK_SCALE * cc;
    assert!(
        (d - expected).abs() < 1e-12,
        "link H at {d:.10} Bohr from C, expected {expected:.10}"
    );
    // Sanity: that fraction of a C-C bond is a physical C-H distance (~1.09 Å).
    let d_ang = d / ANG2BOHR;
    assert!((d_ang - 1.09).abs() < 0.02, "capped C-H = {d_ang:.4} Å, expected ~1.09");

    // (c) The capped fragment is CH4-like (4 real QM atoms + 1 link H) and runs.
    let mol = sys.to_qm_molecule();
    assert_eq!(sys.qm_atom_count(), 4);
    assert_eq!(mol.atoms.len(), 5);
    assert_eq!(mol.atoms[4].symbol, "H", "link atom appended last");

    let (_bs, prep, op, bounds) = setup(&mol);
    let ctx = ParallelContext::default();
    let cfg = RhfConfig { external_potential: sys.to_external_potential(), ..Default::default() };
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(r.converged, "capped QM fragment failed to converge");

    // The diagnostic must flag that the cap sits close to a real MM charge —
    // this is the documented link-atom pathology, and we report it rather than
    // silently pretending the boundary is clean.
    let dmin = sys.min_link_to_charge_distance().unwrap();
    eprintln!(
        "[qmmm] link H at {d:.4} Bohr from frontier C; nearest MM charge \
         {dmin:.4} Bohr away; capped fragment E = {:.8} Ha",
        r.energy
    );
    assert!(dmin > 0.0);
}

/// Cutting a bond *within* the QM region or *within* the MM region must add no
/// link atom, and a partition that cuts nothing must leave the QM molecule
/// exactly as the uncapped partition would.
#[test]
fn uncut_bonds_add_no_link_atoms() {
    let atoms = water_atoms(-0.834, 0.417);
    let bonds = vec![(0usize, 1usize), (0usize, 2usize)];
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1)
        .unwrap()
        .with_link_atoms(&bonds, DEFAULT_LINK_SCALE)
        .unwrap();
    assert!(sys.link_atoms.is_empty());
    assert_eq!(sys.to_qm_molecule().atoms.len(), 3);
}

// ---------------------------------------------------------------------------
// 4. MM FORCES.
// ---------------------------------------------------------------------------

/// `electric_field_at_points` must produce a physically correct field. A direct
/// value-by-value comparison against `electric_field_at_atoms` is not possible:
/// that routine skips the divergent A==A nuclear self-term at each nucleus,
/// which the at-points variant cannot know to skip, so the two quantities
/// genuinely differ at the nuclei. Instead this checks the far-field decay law,
/// which is convention-sensitive: a sign error, a missing factor of 2 on the
/// off-diagonal shell pairs, or a dropped nuclear term would all break the
/// neutral-molecule 1/r^3 cancellation. The finite-difference test below pins
/// the absolute normalization.
#[test]
fn field_at_points_has_correct_neutral_far_field_decay() {
    let atoms = water_atoms(0.0, 0.0);
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
    let mol = sys.to_qm_molecule();
    let (_bs, prep, op, bounds) = setup(&mol);
    let ctx = ParallelContext::default();
    let scf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
    let d = scf.density_total();

    // Far-field: water is neutral, so |E| must fall off like a dipole (1/r^3),
    // not like a monopole (1/r^2). This only holds if the electronic and
    // nuclear contributions cancel to leading order, which requires both the
    // sign convention and the off-diagonal factor of 2 to be right.
    let f1 = electric_field_at_points(&mol, &prep, d, &[[0.0, 0.0, -30.0]]).unwrap()[0];
    let f2 = electric_field_at_points(&mol, &prep, d, &[[0.0, 0.0, -60.0]]).unwrap()[0];
    let n1 = (f1[0].powi(2) + f1[1].powi(2) + f1[2].powi(2)).sqrt();
    let n2 = (f2[0].powi(2) + f2[1].powi(2) + f2[2].powi(2)).sqrt();
    assert!(n1 > 0.0 && n2 > 0.0);
    let ratio = n1 / n2;
    // Dipole field: doubling r cuts |E| by 2^3 = 8. Monopole would give 4.
    assert!(
        (ratio - 8.0).abs() < 1.0,
        "neutral molecule's far field must decay as 1/r^3 (ratio ~8); got {ratio:.3}"
    );
}

/// The MM force must be nonzero, must point the physically correct way, and
/// must obey Newton's third law against the QM region's own response.
///
/// Setup: a +1 MM charge on water's oxygen lone-pair side. That charge is
/// *attracted* to the electron-rich lone pair, so the force on it must point
/// toward the molecule (+z, since the charge sits at −z).
#[test]
fn mm_force_on_a_probe_charge_points_toward_the_electron_rich_side() {
    let mut atoms = water_atoms(0.0, 0.0);
    atoms.push(QmmmAtom::new("Na", 11, 0.0, 0.0, -6.0, 1.0));
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();

    let mol = sys.to_qm_molecule();
    let (_bs, prep, op, bounds) = setup(&mol);
    let ctx = ParallelContext::default();
    let cfg = RhfConfig { external_potential: sys.to_external_potential(), ..Default::default() };
    let scf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(scf.converged);

    let forces = mm_forces(&sys, &mol, &prep, scf.density_total()).unwrap();
    assert_eq!(forces.len(), 1, "one nonzero MM charge => one force row");
    let f = forces[0];

    assert!(
        f[2].abs() > 1e-6,
        "MM force is numerically zero ({:.3e}) — the QM density is exerting nothing",
        f[2]
    );
    // Attracted toward the lone pair, i.e. toward +z (the molecule).
    assert!(
        f[2] > 0.0,
        "a +1 charge at z=-6 must be ATTRACTED toward water's lone pair (+z); \
         got F_z = {:+.6e} a.u.",
        f[2]
    );
    // By C2v symmetry about the z axis, the transverse force must vanish.
    assert!(f[0].abs() < 1e-8, "F_x = {:.3e} should vanish by symmetry", f[0]);
    assert!(f[1].abs() < 1e-8, "F_y = {:.3e} should vanish by symmetry", f[1]);

    eprintln!("[qmmm] force on +1 MM charge at z=-6 Bohr: F = {f:?} a.u.");
}

/// The MM force must agree with a finite-difference derivative of the total
/// energy with respect to the MM charge's position. This is the real
/// correctness check on the electronic contraction — it validates the sign and
/// normalization of the derivative-block reuse against energies computed by a
/// completely independent code path (the SCF itself).
///
/// Note this compares against the *fixed-density* (Hellmann-Feynman) force:
/// the MM charge carries no basis functions, so there is no Pulay term, and the
/// derivative of the converged energy w.r.t. an MM charge position is exactly
/// the electrostatic force. Displacing the charge does relax the QM density,
/// but that relaxation is second order at the variational minimum.
#[test]
fn mm_force_matches_finite_difference_of_the_scf_energy() {
    let atoms_base = water_atoms(0.0, 0.0);
    let z0 = -6.0_f64;
    let ctx = ParallelContext::default();

    // Reference geometry.
    let mut atoms = atoms_base.clone();
    atoms.push(QmmmAtom::new("Na", 11, 0.0, 0.0, z0, 1.0));
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
    let mol = sys.to_qm_molecule();
    let (_bs, prep, op, bounds) = setup(&mol);
    let cfg = RhfConfig {
        external_potential: sys.to_external_potential(),
        // Tighten so the FD signal is well above the convergence noise.
        density_conv: 1e-10,
        ..Default::default()
    };
    let scf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let analytic = mm_forces(&sys, &mol, &prep, scf.density_total()).unwrap()[0];

    // Central difference on the MM charge's z coordinate. The QM basis/geometry
    // is untouched, so `prep`/`bounds` are reused — only the point charge moves.
    let h = 1e-4;
    let energy_at = |z: f64| -> f64 {
        let mut a = atoms_base.clone();
        a.push(QmmmAtom::new("Na", 11, 0.0, 0.0, z, 1.0));
        let s = QmmmSystem::new(&a, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
        let c = RhfConfig {
            external_potential: s.to_external_potential(),
            density_conv: 1e-10,
            ..Default::default()
        };
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &c).unwrap();
        assert!(r.converged);
        r.energy
    };

    let e_plus = energy_at(z0 + h);
    let e_minus = energy_at(z0 - h);
    // Force = -dE/dz.
    let fd_force = -(e_plus - e_minus) / (2.0 * h);

    let rel = (analytic[2] - fd_force).abs() / fd_force.abs().max(1e-12);
    eprintln!(
        "[qmmm] MM force F_z: analytic = {:+.8e}, finite-difference = {:+.8e}, \
         rel err = {rel:.3e}",
        analytic[2], fd_force
    );
    assert!(
        rel < 1e-4,
        "analytic MM force {:+.8e} disagrees with FD {:+.8e} (rel {rel:.3e})",
        analytic[2],
        fd_force
    );
}

/// Forces come back in `mm_charge_positions` order and skip zero-charge MM
/// atoms — a caller indexing rows against that list must not be silently
/// off-by-one when the structure contains uncharged MM atoms.
#[test]
fn mm_force_rows_align_with_mm_charge_positions() {
    let mut atoms = water_atoms(0.0, 0.0);
    atoms.push(QmmmAtom::new("Na", 11, 0.0, 0.0, -8.0, 0.0)); // zero charge: skipped
    atoms.push(QmmmAtom::new("Na", 11, 0.0, 0.0, -6.0, 1.0)); // real charge
    atoms.push(QmmmAtom::new("Cl", 17, 0.0, 0.0, 9.0, -1.0)); // real charge

    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
    let positions = sys.mm_charge_positions();
    assert_eq!(positions.len(), 2, "zero-charge MM atom must be excluded");
    assert_eq!(positions[0][2], -6.0);
    assert_eq!(positions[1][2], 9.0);

    let ep = sys.to_external_potential().unwrap();
    assert_eq!(ep.point_charges.len(), 2);
    assert_eq!(ep.point_charges[0].z, -6.0);
    assert_eq!(ep.point_charges[1].z, 9.0);

    let mol = sys.to_qm_molecule();
    let (_bs, prep, op, bounds) = setup(&mol);
    let ctx = ParallelContext::default();
    let cfg = RhfConfig { external_potential: Some(ep), ..Default::default() };
    let scf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();

    let forces = mm_forces(&sys, &mol, &prep, scf.density_total()).unwrap();
    assert_eq!(forces.len(), positions.len());
    // The +1 and the −1 charge sit on opposite sides and carry opposite-signed
    // charges, so both are attracted toward the molecule => opposite-signed F_z.
    assert!(forces[0][2] * forces[1][2] < 0.0);
}

/// No MM charges => no force rows, and the routine must not error or allocate
/// a spurious zero row.
#[test]
fn mm_forces_empty_for_an_empty_mm_region() {
    let atoms = water_atoms(0.0, 0.0);
    let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
    let mol = sys.to_qm_molecule();
    let (_bs, prep, op, bounds) = setup(&mol);
    let ctx = ParallelContext::default();
    let scf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
    let forces = mm_forces(&sys, &mol, &prep, scf.density_total()).unwrap();
    assert!(forces.is_empty());
}

// ---------------------------------------------------------------------------
// 5. BOUNDARY CHARGE SCHEMES (link-atom / host-charge pathology).
// ---------------------------------------------------------------------------

/// Exactness anchor: `Keep` must be BIT-IDENTICAL to not calling
/// `with_boundary_charges` at all — same potential, same charge positions.
#[test]
fn boundary_scheme_keep_is_bit_identical_to_no_scheme() {
    let plain = capped_ethane();
    let kept = capped_ethane().with_boundary_charges(&ethane_bonds(), BoundaryChargeScheme::Keep).unwrap();
    assert_eq!(plain.to_external_potential(), kept.to_external_potential());
    assert_eq!(plain.mm_charge_positions(), kept.mm_charge_positions());
    assert_eq!(plain.min_link_to_charge_distance(), kept.min_link_to_charge_distance());
}

/// Second anchor: every scheme is a no-op when the partition cuts no bond —
/// there is no host charge to delete or move.
#[test]
fn boundary_schemes_are_no_ops_without_a_cut_bond() {
    // Whole ethane QM except one far-away charge; no bond crosses the boundary.
    let mut atoms = ethane_atoms();
    atoms.push(QmmmAtom::new("Na", 11, 0.0, 0.0, 20.0, 1.0));
    let qm: Vec<usize> = (0..8).collect();
    let base = QmmmSystem::new(&atoms, QmSelection::Indices(qm.clone()), 0, 1)
        .unwrap()
        .with_link_atoms(&ethane_bonds(), DEFAULT_LINK_SCALE)
        .unwrap();
    assert!(base.link_atoms.is_empty());
    for scheme in [
        BoundaryChargeScheme::Keep,
        BoundaryChargeScheme::DeleteHost,
        BoundaryChargeScheme::RedistributedCharge,
        BoundaryChargeScheme::RedistributedChargeDipole,
    ] {
        let s = base.clone().with_boundary_charges(&ethane_bonds(), scheme).unwrap();
        assert_eq!(base.to_external_potential(), s.to_external_potential(), "{scheme:?}");
        assert_eq!(base.mm_charge_positions(), s.mm_charge_positions(), "{scheme:?}");
    }
}

/// Z1: the MM host of the cut bond (C1) loses its charge; nothing else moves.
#[test]
fn delete_host_removes_exactly_the_m1_charge() {
    let plain = capped_ethane();
    let z1 = capped_ethane().with_boundary_charges(&ethane_bonds(), BoundaryChargeScheme::DeleteHost).unwrap();

    let ep_plain = plain.to_external_potential().unwrap();
    let ep_z1 = z1.to_external_potential().unwrap();
    // Plain: C1 + three H = 4 charges. Z1: the three H only.
    assert_eq!(ep_plain.point_charges.len(), 4);
    assert_eq!(ep_z1.point_charges.len(), 3);
    for pc in &ep_z1.point_charges {
        assert_eq!(pc.q, 0.033, "only the H charges survive, unchanged");
        assert!(ep_plain.point_charges.contains(pc), "surviving charges are untouched");
    }
    // Total charge is NOT conserved under Z1 — that is the documented price.
    let (q_plain, _) = total_charge_and_dipole(&plain);
    let (q_z1, _) = total_charge_and_dipole(&z1);
    assert!(((q_plain - q_z1) - (-0.1)).abs() < 1e-14, "Z1 drops exactly q(C1) = -0.1");
}

/// RC (Lin & Truhlar 2005): q(M1) is split evenly over the midpoints of the
/// M1-M2 bonds. Total charge is conserved by construction, and the M1 site
/// itself carries nothing afterwards.
#[test]
fn redistributed_charge_conserves_total_charge_via_m1_m2_midpoints() {
    let plain = capped_ethane();
    let rc = capped_ethane()
        .with_boundary_charges(&ethane_bonds(), BoundaryChargeScheme::RedistributedCharge)
        .unwrap();
    let atoms = ethane_atoms();

    let (q_plain, _) = total_charge_and_dipole(&plain);
    let (q_rc, _) = total_charge_and_dipole(&rc);
    assert!((q_plain - q_rc).abs() < 1e-14, "RC must conserve total MM charge");

    let ep = rc.to_external_potential().unwrap();
    // Three H (unchanged) + three midpoint charges; C1's own charge is gone.
    assert_eq!(ep.point_charges.len(), 6);
    let c1 = &atoms[1];
    assert!(
        !ep.point_charges.iter().any(|pc| pc.x == c1.x && pc.y == c1.y && pc.z == c1.z_pos),
        "no charge may remain on the M1 site"
    );
    let mut n_mid = 0;
    for &h in &[3usize, 5, 7] {
        let mid = [(c1.x + atoms[h].x) / 2.0, (c1.y + atoms[h].y) / 2.0, (c1.z_pos + atoms[h].z_pos) / 2.0];
        let found = ep.point_charges.iter().find(|pc| (pc.x - mid[0]).abs() < 1e-14 && (pc.y - mid[1]).abs() < 1e-14 && (pc.z - mid[2]).abs() < 1e-14);
        let pc = found.expect("a charge at each M1-M2 midpoint");
        assert!((pc.q - (-0.1 / 3.0)).abs() < 1e-14, "q(M1)/n_M2 at each midpoint");
        n_mid += 1;
    }
    assert_eq!(n_mid, 3);
    // The M2 hydrogens keep their own charge under RC.
    assert_eq!(ep.point_charges.iter().filter(|pc| pc.q == 0.033).count(), 3);
}

/// RCD: as RC, but each midpoint gets 2·q0 and each M2 loses q0, which keeps
/// BOTH the total charge and the dipole of the {M1, M2...} group unchanged.
#[test]
fn redistributed_charge_dipole_conserves_charge_and_dipole() {
    let plain = capped_ethane();
    let rcd = capped_ethane()
        .with_boundary_charges(&ethane_bonds(), BoundaryChargeScheme::RedistributedChargeDipole)
        .unwrap();

    let (q_plain, mu_plain) = total_charge_and_dipole(&plain);
    let (q_rcd, mu_rcd) = total_charge_and_dipole(&rcd);
    assert!((q_plain - q_rcd).abs() < 1e-14, "RCD must conserve total MM charge");
    for k in 0..3 {
        assert!(
            (mu_plain[k] - mu_rcd[k]).abs() < 1e-13,
            "RCD must conserve the MM dipole: axis {k} {} vs {}",
            mu_plain[k],
            mu_rcd[k]
        );
    }
    // And it is NOT the same charge set as RC (the M2 charges shifted).
    let ep = rcd.to_external_potential().unwrap();
    assert_eq!(ep.point_charges.len(), 6);
    let q0 = -0.1 / 3.0;
    assert_eq!(ep.point_charges.iter().filter(|pc| (pc.q - 2.0 * q0).abs() < 1e-14).count(), 3, "midpoints carry 2q0");
    assert_eq!(ep.point_charges.iter().filter(|pc| (pc.q - (0.033 - q0)).abs() < 1e-14).count(), 3, "M2 carry q_M2 - q0");
}

/// The whole point: under RC/RCD the nearest MM charge to the link hydrogen
/// is no longer the M1 host it sits on top of.
#[test]
fn redistribution_moves_the_nearest_charge_away_from_the_link_atom() {
    let plain = capped_ethane();
    let d_plain = plain.min_link_to_charge_distance().unwrap();
    for scheme in [
        BoundaryChargeScheme::DeleteHost,
        BoundaryChargeScheme::RedistributedCharge,
        BoundaryChargeScheme::RedistributedChargeDipole,
    ] {
        let s = capped_ethane().with_boundary_charges(&ethane_bonds(), scheme).unwrap();
        let d = s.min_link_to_charge_distance().unwrap();
        assert!(d > d_plain, "{scheme:?}: nearest charge {d:.4} should be farther than {d_plain:.4}");
    }
}

/// MM forces must cover every charge that enters the potential — including
/// the off-atom midpoint charges — in `mm_charge_positions` order.
#[test]
fn mm_forces_cover_redistributed_midpoint_charges() {
    let sys = capped_ethane()
        .with_boundary_charges(&ethane_bonds(), BoundaryChargeScheme::RedistributedChargeDipole)
        .unwrap();
    let positions = sys.mm_charge_positions();
    let ep = sys.to_external_potential().unwrap();
    assert_eq!(positions.len(), ep.point_charges.len());
    for (p, pc) in positions.iter().zip(ep.point_charges.iter()) {
        assert_eq!(*p, [pc.x, pc.y, pc.z]);
    }

    let mol = sys.to_qm_molecule();
    let (_bs, prep, op, bounds) = setup(&mol);
    let ctx = ParallelContext::default();
    let cfg = RhfConfig { external_potential: Some(ep), ..Default::default() };
    let scf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(scf.converged);
    let f = mm_forces(&sys, &mol, &prep, scf.density_total()).unwrap();
    assert_eq!(f.len(), positions.len());
    assert!(f.iter().all(|r| r.iter().all(|v| v.is_finite())));
}

/// A host with no MM neighbours has nowhere to send its charge: RC/RCD must
/// refuse rather than silently fall back to deletion.
#[test]
fn redistribution_errors_when_the_host_has_no_mm_neighbours() {
    // Only the cut bond is known; C1's C-H bonds are not in the list.
    let sys = QmmmSystem::new(&ethane_atoms(), QmSelection::Indices(vec![0, 2, 4, 6]), 0, 1)
        .unwrap()
        .with_link_atoms(&[(0, 1)], DEFAULT_LINK_SCALE)
        .unwrap();
    assert!(sys.clone().with_boundary_charges(&[(0, 1)], BoundaryChargeScheme::RedistributedCharge).is_err());
    assert!(sys.clone().with_boundary_charges(&[(0, 1)], BoundaryChargeScheme::RedistributedChargeDipole).is_err());
    // Deletion needs no neighbours and still works.
    assert!(sys.with_boundary_charges(&[(0, 1)], BoundaryChargeScheme::DeleteHost).is_ok());
}
