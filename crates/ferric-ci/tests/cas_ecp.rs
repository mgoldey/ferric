//! CAS-CI and CASSCF carry the basis set's ECP in their one-electron
//! Hamiltonian, as the RHF reference does.
//!
//! System: HI, def2-SVP (28-electron ECP on I), closed shell. Anchor: the
//! zero-correlation limit. With one active orbital holding two electrons
//! (the RHF HOMO) the CAS wavefunction is the RHF determinant, so the CAS
//! energy at the RHF orbitals is the RHF energy, and the CASSCF orbital
//! gradient there is the RHF Brillouin residual.
//!
//! Bars. Energy 1e-9 Ha: the RHF energy is converged to `energy_conv` 1e-12
//! and is stationary, so a density error `δ` at `density_conv` 1e-10 moves it
//! by `O(δ²)`; the CAS energy is the same expression built by a different
//! path (dense AO ERIs, exact J/K or an MO transform), whose difference from
//! the SCF's screened direct J/K is bounded by the 1e-12 Schwarz threshold
//! times the `O(nbas²)` density sum, below 1e-9 here. Without V_ECP the
//! energies differ by `Tr[D V_ECP]`, Hartrees. Gradient 1e-6: the core-virtual
//! element is `4 F_ai`, which RHF drives to the `density_conv` scale (1e-10)
//! times the Fock norm (O(10) Ha for I); a missing V_ECP leaves `F_ai` of
//! the size of `V_ECP`'s occupied-virtual block, O(0.1) Ha.

use ferric_ci::casscf::{CasScfConfig, CasScfProblem};
use ferric_ci::{run_cas_ci, CasCiConfig};
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

#[test]
fn cas_zero_correlation_limit_reproduces_the_ecp_rhf_energy() {
    let bs = basis::bundled("def2-svp").unwrap();
    let mut mol = Molecule::parse_xyz("2\n\nH 0.0 0.0 0.0\nI 0.0 0.0 1.61\n", 0, 1).unwrap();
    mol.apply_ecp(&bs);
    assert!(mol.atoms[1].n_core_ecp > 0, "def2-SVP carries an ECP on I");
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        max_iter: 300,
        energy_conv: 1e-12,
        density_conv: 1e-10,
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(rhf.converged && rhf.df_jk.is_none());
    let nocc = mol.nelec() as usize / 2;
    let nmo = rhf.mos_r().ncols();

    let ci = run_cas_ci(
        &mol,
        &prep,
        &rhf,
        CasCiConfig {
            n_active: 1,
            n_elec_active: (1, 1),
            active_start: nocc - 1,
            ..Default::default()
        },
    )
    .unwrap();
    let d_ci = (ci.e_total - rhf.energy).abs();
    println!("CAS-CI(2,1) - RHF = {d_ci:.3e} Ha");
    assert!(d_ci < 1e-9, "CAS-CI(2,1) differs from RHF by {d_ci:e} Ha");

    let problem = CasScfProblem::new(
        &mol,
        &prep,
        nmo,
        &CasScfConfig {
            n_active: 1,
            n_elec_active: (1, 1),
            n_core: nocc - 1,
            ..Default::default()
        },
    )
    .unwrap();
    let pt = problem.evaluate(rhf.mos_r()).unwrap();
    let d_scf = (pt.e_total - rhf.energy).abs();
    let gmax = pt.gradient.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    println!("CASSCF(2,1) - RHF = {d_scf:.3e} Ha, max|g| = {gmax:.3e}");
    assert!(d_scf < 1e-9, "CASSCF(2,1) differs from RHF by {d_scf:e} Ha");
    assert!(gmax < 1e-6, "CASSCF(2,1) gradient at RHF orbitals {gmax:e}");
}
