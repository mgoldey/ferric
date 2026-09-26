//! OO-RI-MP2 with an ECP basis: the orbital-optimization functional must use
//! the SAME core Hamiltonian as the SCF it starts from.
//!
//! The defect this pins: `oo_ri_mp2` / `u_oo_ri_mp2` (and the OO gradient's
//! semicanonical Fock) built hcore with `hcore_with_external`, which has no
//! V_ECP, while the molecule's nuclear charges (V_nuc and E_nn) were the
//! ECP-reduced ones and the starting orbitals came from an SCF solved WITH
//! V_ECP (`driver::prepare` uses `hcore_ecp_with_external`). The OO functional
//! was therefore a different Hamiltonian: at zero rotation its reference part
//! sat tr(D V_ECP) below the SCF energy (50.46 Ha for HI/def2-SVP at 1.75 Å,
//! measured with PySCF; see scripts/validation/gen_oo_rimp2_ecp.py).
//!
//! Exactness anchor: with `max_iter = 0` no rotation is taken, so the reported
//! `hf_energy` is the OO reference energy of the SCF orbitals and must equal
//! the SCF energy, and `mp2_corr` must equal plain RI-MP2's. With the
//! ECP-less hcore the first assertion misses by tr(D V_ECP) (tens of Ha for
//! HI/def2-SVP), so it cannot pass by accident; the `tr(D V_ECP)` resolving-
//! power assertion below checks that on every run.
//!
//! All-electron: `hcore_ecp_with_external` returns `hcore_with_external`'s
//! matrix unchanged when the basis has no ECP, so the fix is bit-identical
//! there; that equality is asserted exactly, and the zero-rotation anchor is
//! repeated on an all-electron molecule.
//!
//! MUTATION (run once): revert `oo_rimp2.rs` / `u_oo_rimp2.rs` to
//! `hcore_with_external(obs, ext)`; the two ECP anchors fail by > 1 Ha.

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_mp2::u_oo_rimp2::{u_oo_ri_mp2, UOoRiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

/// Zero-rotation OO reference energy vs the SCF energy. Both use exact J/K on
/// the same orbitals; the residual is the SCF's final-density convergence
/// (second order) and J/K screening.
const TOL_ANCHOR: f64 = 1e-10;
/// tr(D V_ECP) must exceed this, so the anchor above resolves an ECP-less
/// hcore by > 1e10 x its bar.
const MIN_TR_D_VECP: f64 = 1.0;

/// HI stretched to 1.75 A (testdata/molecules/validation/ecp/hi.xyz); def2-SVP
/// puts the 28-electron def2 ECP on I.
const HI_XYZ: &str = "2\nHI\nH 0.0 0.0 0.0\nI 0.0 0.0 1.75\n";

struct Sys {
    mol: Molecule,
    bs: basis::BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    op: Operator,
    bounds: SchwarzBounds,
}

fn system(xyz: &str, charge: i32, mult: usize, orb: &str, aux: &str) -> Sys {
    let mut mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let bs = basis::bundled(orb).unwrap();
    // apply_ecp BEFORE nelec() / nuclear_repulsion() / PreparedBasis.
    mol.apply_ecp(&bs);
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    Sys {
        mol,
        bs,
        obs,
        dfbs,
        op,
        bounds,
    }
}

fn scf_config() -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-12,
        density_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    }
}

fn assert_ecp_active(s: &Sys, ctx: &str) {
    assert!(
        s.mol.atoms.iter().any(|a| a.n_core_ecp > 0),
        "{ctx}: no atom carries an ECP — this test would not exercise V_ECP"
    );
}

/// tr(D V_ECP): what an ECP-less hcore would shift the zero-rotation energy by.
fn assert_resolving_power(s: &Sys, d_total: &ndarray::Array2<f64>, ctx: &str) {
    let vecp = oneelectron::ecp_potential(&s.mol, &s.bs)
        .unwrap_or_else(|| panic!("{ctx}: ecp_potential returned None for an ECP basis"));
    let tr: f64 = d_total.iter().zip(vecp.iter()).map(|(d, v)| d * v).sum();
    eprintln!("{ctx}: tr(D V_ECP) = {tr:.6} Ha");
    assert!(
        tr.abs() > MIN_TR_D_VECP,
        "{ctx}: tr(D V_ECP) = {tr:.3e} is too small for the anchor to detect a missing V_ECP"
    );
}

fn zero_rotation_closed(s: &Sys, ctx: &str) {
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &s.mol,
        &s.obs,
        s.op,
        &s.bounds,
        &scf_config(),
    )
    .unwrap();
    assert!(rhf.converged, "{ctx}: RHF not converged");
    let oo = oo_ri_mp2(
        &s.mol,
        &s.obs,
        &s.dfbs,
        s.op,
        &s.bounds,
        &rhf,
        &OoRiMp2Config {
            max_iter: 0,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(
        oo.iterations, 0,
        "{ctx}: max_iter = 0 must take no rotation"
    );
    let d = (oo.hf_energy - rhf.energy).abs();
    eprintln!(
        "{ctx}: OO reference at kappa=0 {:.12} vs RHF {:.12} |d| {d:.2e}",
        oo.hf_energy, rhf.energy
    );
    assert!(
        d < TOL_ANCHOR,
        "{ctx}: zero-rotation OO reference energy {:.12} != RHF {:.12} (|d| {d:.2e}) — \
         the OO functional's hcore is not the SCF's",
        oo.hf_energy,
        rhf.energy
    );
    let mp2 = ri_mp2(&s.mol, &s.obs, &s.dfbs, s.op, &rhf, &RiMp2Config::default()).unwrap();
    let dc = (oo.mp2_corr - mp2.mp2_corr).abs();
    eprintln!("{ctx}: OO E_corr at kappa=0 vs RI-MP2 |d| {dc:.2e}");
    assert!(
        dc < TOL_ANCHOR,
        "{ctx}: zero-rotation OO E_corr {:.12} != RI-MP2 {:.12} (|d| {dc:.2e})",
        oo.mp2_corr,
        mp2.mp2_corr
    );
    if s.mol.atoms.iter().any(|a| a.n_core_ecp > 0) {
        assert_resolving_power(s, rhf.density_r(), ctx);
    }
}

#[test]
fn oo_rimp2_zero_rotation_reference_equals_rhf_with_ecp() {
    let s = system(HI_XYZ, 0, 1, "def2-svp", "def2-svp-rifit");
    assert_ecp_active(&s, "HI/def2-SVP");
    zero_rotation_closed(&s, "HI/def2-SVP");
}

#[test]
fn oo_rimp2_zero_rotation_reference_equals_rhf_all_electron() {
    let xyz = "3\nH2O\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
    let s = system(xyz, 0, 1, "cc-pvdz", "cc-pvdz-ri");
    assert!(s.mol.atoms.iter().all(|a| a.n_core_ecp == 0));
    zero_rotation_closed(&s, "H2O/cc-pVDZ");
}

#[test]
fn u_oo_rimp2_zero_rotation_reference_equals_uhf_with_ecp() {
    let ctx = "HI+/def2-SVP";
    let s = system(HI_XYZ, 1, 2, "def2-svp", "def2-svp-rifit");
    assert_ecp_active(&s, ctx);
    let uhf = solve_uhf(
        &ParallelContext::default(),
        &s.mol,
        &s.obs,
        &s.bounds,
        &scf_config(),
    )
    .unwrap();
    assert!(uhf.converged, "{ctx}: UHF not converged");
    let oo = u_oo_ri_mp2(
        &s.mol,
        &s.obs,
        &s.dfbs,
        s.op,
        &s.bounds,
        &uhf,
        &UOoRiMp2Config {
            max_iter: 0,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(
        oo.iterations, 0,
        "{ctx}: max_iter = 0 must take no rotation"
    );
    let d = (oo.hf_energy - uhf.energy).abs();
    eprintln!(
        "{ctx}: U-OO reference at kappa=0 {:.12} vs UHF {:.12} |d| {d:.2e}",
        oo.hf_energy, uhf.energy
    );
    assert!(
        d < TOL_ANCHOR,
        "{ctx}: zero-rotation U-OO reference energy {:.12} != UHF {:.12} (|d| {d:.2e})",
        oo.hf_energy,
        uhf.energy
    );
    assert_resolving_power(&s, uhf.density_total(), ctx);
}

/// For an all-electron basis the ECP-aware hcore the OO code now calls is the
/// old matrix, bit for bit (with and without an external potential), so every
/// all-electron OO result is unchanged.
#[test]
fn all_electron_ecp_hcore_is_bit_identical_to_plain() {
    let xyz = "3\nH2O\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
    let s = system(xyz, 0, 1, "cc-pvdz", "cc-pvdz-ri");
    let ext = ExternalPotential {
        point_charges: vec![PointCharge {
            q: 0.5,
            x: 0.0,
            y: 0.0,
            z: -5.0,
        }],
        field: Some([0.0, 0.0, 1e-3]),
        smeared_charges: Vec::new(),
    };
    for e in [None, Some(&ext)] {
        let plain = oneelectron::hcore_with_external(&s.obs, e).unwrap();
        let with_ecp =
            oneelectron::hcore_ecp_with_external(&s.obs, &s.mol, s.obs.basis_set(), e).unwrap();
        assert!(
            plain
                .iter()
                .zip(with_ecp.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "all-electron hcore_ecp_with_external differs from hcore_with_external (ext {:?})",
            e.is_some()
        );
    }
}
