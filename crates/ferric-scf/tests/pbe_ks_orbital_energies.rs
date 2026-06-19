//! PBE Kohn–Sham SCF — ferric vs PySCF on TOTAL ENERGY and ORBITAL ENERGIES.
//!
//! The existing dft_def2_tzvp test only checks total energy to 0.1 Ha. For a
//! trustworthy @PBE GW reference, GW consumes the orbital energies, so we need
//! tight agreement on ε (especially ε_HOMO). This pins ferric's self-consistent
//! PBE-KS (solve_rhf with xc=Some("pbe")) against PySCF dft.RKS xc=pbe on the
//! identical H2O/cc-pVDZ setup used in gw100_full.
//!
//! PySCF reference (grids.level=3):
//!   E      = -76.33348163 Ha
//!   ε_HOMO = -6.12123 eV
//!   ε_occ  = [-509.9147, -24.5743, -12.4260, -8.2887, -6.1212] eV

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const HA: f64 = 27.211386245988;

// PySCF dft.RKS xc=pbe, H2O/cc-pVDZ, grids.level=3.
const PYSCF_E: f64 = -76.33348163;
const PYSCF_HOMO_EV: f64 = -6.12123;

#[test]
fn pbe_ks_h2o_energy_and_homo_match_pyscf() {
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).expect("xyz");
    let obs_bs = basis::bundled("cc-pvdz").expect("cc-pvdz");
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("obs");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        xc: Some("pbe".into()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        ..Default::default()
    };
    let rks = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).expect("PBE-KS SCF");
    let nocc = (mol.nelec() as usize) / 2;
    let homo_ev = rks.eps_r()[nocc - 1] * HA;

    eprintln!(
        "ferric PBE-KS: E = {:.8} (Δ {:.2e} Ha), ε_HOMO = {:.5} eV (Δ {:.4} eV)",
        rks.energy,
        rks.energy - PYSCF_E,
        homo_ev,
        homo_ev - PYSCF_HOMO_EV,
    );

    // Total energy: tight (grid + functional agreement).
    assert!(
        (rks.energy - PYSCF_E).abs() < 2e-3,
        "PBE-KS energy {:.8} vs PySCF {:.8} (Δ {:.2e} Ha)",
        rks.energy, PYSCF_E, rks.energy - PYSCF_E
    );
    // HOMO orbital energy: the quantity GW consumes. Looser than energy because
    // it is grid-sensitive, but must be well within chemical relevance.
    assert!(
        (homo_ev - PYSCF_HOMO_EV).abs() < 0.05,
        "PBE-KS ε_HOMO {:.5} eV vs PySCF {:.5} eV (Δ {:.4} eV)",
        homo_ev, PYSCF_HOMO_EV, homo_ev - PYSCF_HOMO_EV
    );
}
