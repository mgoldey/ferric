//! Size-extensivity of RI-MP2: E(two well-separated monomers) ≈ 2 × E(one
//! monomer).
//!
//! Physics: for any properly size-extensive electronic-structure method, the
//! total energy of two non-interacting fragments must equal the sum of the
//! fragment energies (Bartlett's classic extensivity criterion — MP2, like
//! full CI/CC, is diagrammatically size-extensive because every energy
//! contribution factorizes into a sum over connected diagrams on each
//! fragment once cross-fragment integrals vanish). A supersystem calculation
//! that instead scaled sub-linearly (a common symptom of a truncated/
//! non-size-consistent correlation treatment, or of an accidental
//! normalization by the total electron/orbital count) would show up here as
//! E(dimer) departing measurably from 2 × E(monomer). This gap is ABSENT
//! from the existing MP2 test suite (see `crates/ferric-mp2/tests/`), which
//! checks accuracy/gradients/spin-components but never additivity.
//!
//! Setup follows the RI-MP2 test idiom in
//! `crates/ferric-mp2/tests/rimp2_gradient_external.rs`: STO-3G orbital
//! basis (cheapest bundled AO basis) paired with the `cc-pvdz-ri` bundled
//! auxiliary/fitting basis, `solve_rhf` for the reference determinant, then
//! `ri_mp2` for the correlation energy. H2 is used instead of H2O/STO-3G
//! (only 2 electrons, 2 basis functions per monomer / 4 for the dimer) to
//! keep both SCF+MP2 solves well under a second.
//!
//! Separation: the two H2 monomers are placed 100 Bohr apart along x. At
//! this range:
//!   - Coulomb/exchange overlap between the fragments' STO-3G basis
//!     functions is astronomically small (Gaussian overlap decays as
//!     exp(-alpha * R^2); at R = 100 Bohr this is far below f64 precision
//!     for STO-3G-scale exponents), so SCF sees two isolated H2 molecules
//!     for all numerical purposes.
//!   - The residual physical interaction is the leading-order dispersion
//!     (London/van der Waals) tail, which for two H2 molecules falls off as
//!     C6 / R^6. With C6 for H2-H2 of order 1-10 Hartree*Bohr^6 (physical
//!     scale), R^6 = 1e12 at R = 100 Bohr puts the true physical
//!     non-additivity at ~1e-12 to 1e-11 Hartree or smaller — far below any
//!     tolerance we could resolve here, so this test is not fighting real
//!     physics, only floating-point/SCF-convergence noise.
//!   - No distance-based integral screening in ferric (Schwarz bounds are
//!     value-based, not a hard geometric cutoff — see
//!     `ferric_scf::screening::SchwarzBounds::compute`) kicks in at this
//!     range in a way that would zero out or distort the cross terms
//!     differently between the monomer and dimer runs.
//!
//! Tolerance: 1e-7 Hartree on `2 * E(monomer) - E(dimer)`. This is many
//! orders above the ~1e-11 estimated residual dispersion tail (so the test
//! is not sensitive to real long-range physics) but far tighter than the
//! ~1e-3 to ~1e-4 Hartree gap that a broken/non-extensive correlation
//! energy would produce (e.g. forgetting to double an intermediate, or a
//! bug that scales the correlation energy by 1/N of the supersystem instead
//! of being additive over fragments). It also comfortably clears the SCF's
//! own `energy_conv=1e-9` convergence gate used below.
use ferric_core::basis;
use ferric_core::mol::{Atom, Molecule};
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// A single H2 monomer at its STO-3G-appropriate bond length (0.74 Angstrom,
/// converted by the parser to Bohr), centered near the given x-origin.
fn h2_at(x_center: f64) -> Molecule {
    let xyz = format!(
        "2\nH2\nH {} 0.0 0.0\nH {} 0.0 0.0\n",
        x_center - 0.37,
        x_center + 0.37
    );
    Molecule::parse_xyz(&xyz, 0, 1).unwrap()
}

/// Two H2 monomers separated by 100 Bohr along x, as a single `Molecule`
/// (same charge/multiplicity semantics as one closed-shell supersystem: 4
/// electrons total, singlet).
fn h2_dimer_far() -> Molecule {
    let m1 = h2_at(-50.0);
    let m2 = h2_at(50.0);
    let mut atoms: Vec<Atom> = Vec::with_capacity(4);
    atoms.extend(m1.atoms);
    atoms.extend(m2.atoms);
    Molecule { atoms, charge: 0, multiplicity: 1 }
}

fn rimp2_total_energy(mol: &Molecule) -> f64 {
    let obs_bs = basis::bundled("sto-3g").unwrap();
    let obs = PreparedBasis::new(mol, &obs_bs).unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-9,
        density_conv: 1e-8,
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), mol, &obs, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged, "RHF must converge for the extensivity check to be meaningful");

    let mp2_cfg = RiMp2Config::default();
    let result = ri_mp2(mol, &obs, &dfbs, op, &rhf, &mp2_cfg).unwrap();
    result.total_energy
}

/// RI-MP2/STO-3G total energy of two H2 molecules separated by 100 Bohr must
/// equal 2x the single-H2 total energy to within 1e-7 Hartree — size
/// extensivity of a correlated (post-HF) method.
#[test]
fn rimp2_h2_dimer_far_is_extensive() {
    let e_monomer = rimp2_total_energy(&h2_at(0.0));
    let e_dimer = rimp2_total_energy(&h2_dimer_far());

    let non_additivity = (2.0 * e_monomer - e_dimer).abs();
    eprintln!(
        "RI-MP2/STO-3G H2: E(monomer) = {e_monomer:.12}, E(dimer @ 100 Bohr) = {e_dimer:.12}, \
         2*E(monomer) - E(dimer) = {non_additivity:.3e}"
    );

    assert!(
        non_additivity < 1e-7,
        "RI-MP2 total energy must be size-extensive: |2*E(monomer) - E(dimer)| = \
         {non_additivity:.3e} (want < 1e-7 Ha)"
    );
}
