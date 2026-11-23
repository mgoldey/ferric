//! The OP-MP3 drivers on a `PreparedSystem`: they screen with the system's
//! kind, and they refuse a fitting basis prepared on another system.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_mp2::opmp3_driver::{opmp3_on, opmp3_scf, Opmp3Config, Opmp3Integrals, Opmp3RefKind};
use ferric_scf::screening::ScreeningKind;
use ferric_scf::{EcpMolecule, PrepareOptions, PreparedSystem};

const WATER: &str = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n";

fn water(screening: ScreeningKind) -> PreparedSystem {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let options = PrepareOptions {
        screening,
        ..Default::default()
    };
    EcpMolecule::new(mol, &bs).prepare(options).unwrap()
}

/// `opmp3_scf` builds its own config; on a CSB system it must screen with
/// CSB (the system's solve methods refuse any other kind).
#[test]
fn opmp3_scf_runs_on_a_csb_system() {
    let sys = water(ScreeningKind::Csb);
    let scf = opmp3_scf(&sys, &Opmp3Config::new(Opmp3RefKind::Hf)).unwrap();
    assert!(scf.converged);
}

/// A fitting basis of one system next to another system is an error
/// (`PreparedAux::basis_for`), not a silently mixed run.
#[test]
fn opmp3_on_reference_refuses_an_aux_of_another_system() {
    let config = Opmp3Config::new(Opmp3RefKind::Hf);
    let a = water(ScreeningKind::Schwarz);
    let b = water(ScreeningKind::Schwarz);
    let scf = opmp3_scf(&a, &config).unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let b_aux = b.prepare_aux(&aux_bs).unwrap();
    let err = opmp3_on(&a, &scf, Opmp3Integrals::Ri(&b_aux), &config)
        .map(|_| ())
        .unwrap_err();
    assert!(err.to_string().contains("not prepared on"), "{err}");
    // The same aux prepared on `a` runs.
    let a_aux = a.prepare_aux(&aux_bs).unwrap();
    opmp3_on(&a, &scf, Opmp3Integrals::Ri(&a_aux), &config).unwrap();
}
