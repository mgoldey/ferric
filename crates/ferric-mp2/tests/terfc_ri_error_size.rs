//! SCRATCH diagnostic: how big is the RI (density-fitting) error on THIS
//! system/basis? Compare exact 4-index Coulomb MP2 against RI-MP2 with the
//! same aux basis. If the RI error is of the same order as the terfc
//! "overshoot" (1.5e-4 Ha), the overshoot is RI noise, not an integral bug.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::canonical::canonical_mp2;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

#[test]
fn probe_ri_error_magnitude_on_coulomb() {
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/alkane_4.xyz"
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let opc = Operator::coulomb();
    let bounds = SchwarzBounds::compute(opc, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        opc,
        &bounds,
        &RhfConfig { energy_conv: 1e-9, ..Default::default() },
    )
    .unwrap();

    let e_ri = ri_mp2_spin_components(&mol, &obs, &dfbs, opc, &rhf, &RiMp2Config::default())
        .unwrap()
        .0
        .e_total;
    let e_exact = canonical_mp2(&mol, &obs, opc, &rhf, 0).unwrap();
    eprintln!("alkane_4/cc-pVDZ  E_MP2 exact  = {e_exact:.10}");
    eprintln!("alkane_4/cc-pVDZ  E_MP2 RI     = {e_ri:.10}");
    eprintln!(
        "RI error = {:.3e} Ha  ({:.4}% of E)",
        e_ri - e_exact,
        (e_ri - e_exact) / e_exact * 100.0
    );
    eprintln!("terfc overshoot at r0=2.0A was 1.47e-4 Ha (0.0245% of E) for comparison");
}
