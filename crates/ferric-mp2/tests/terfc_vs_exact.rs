//! SCRATCH diagnostic: the terfc "overshoot" vs the EXACT (non-RI) Coulomb
//! MP2 rather than the RI Coulomb MP2. If terfc(r0) approaches the exact
//! Coulomb value from below, the operator physics is right and the failing
//! assertion is simply comparing against the wrong reference.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::canonical::canonical_mp2;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const A2B: f64 = 1.889_725_988_6;

#[test]
fn probe_terfc_vs_exact_coulomb() {
    if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
        eprintln!("skip: no tables");
        return;
    }
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
    let cfg = RiMp2Config::default();

    let e_ri_coul = ri_mp2_spin_components(&mol, &obs, &dfbs, opc, &rhf, &cfg)
        .unwrap()
        .0
        .e_total;
    let e_exact = canonical_mp2(&mol, &obs, opc, &rhf, 0).unwrap();
    eprintln!("E_coul  RI    = {e_ri_coul:.10}");
    eprintln!("E_coul  EXACT = {e_exact:.10}   (RI err {:+.3e})", e_ri_coul - e_exact);
    eprintln!();
    eprintln!("   r0      E_terfc(RI)      /E_RI_coul     /E_exact_coul");
    for &r0a in &[0.75_f64, 1.05, 1.5, 2.0, 3.0, 6.0, 12.0] {
        let et = ri_mp2_spin_components(&mol, &obs, &dfbs, Operator::terfc(r0a * A2B), &rhf, &cfg)
            .unwrap()
            .0
            .e_total;
        eprintln!(
            "{r0a:6.2}   {et:.10}   {:.8}   {:.8}",
            et / e_ri_coul,
            et / e_exact
        );
    }
}
