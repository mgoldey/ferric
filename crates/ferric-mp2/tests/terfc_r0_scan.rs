//! SCRATCH diagnostic: energy ratio |E_terfc|/|E_coul| over a wide r0 range,
//! and the SAME quantity with the metric forced to Coulomb (metric_op), to
//! separate "3-index tensor" error from "RI metric" error.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const A2B: f64 = 1.889_725_988_6;

#[test]
fn probe_terfc_r0_scan_wide() {
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
    let cfg_cmet = RiMp2Config { metric_op: Some(opc), ..RiMp2Config::default() };

    let ec = ri_mp2_spin_components(&mol, &obs, &dfbs, opc, &rhf, &cfg)
        .unwrap()
        .0
        .e_total;
    eprintln!("E_coul = {ec:.10}");
    eprintln!("   r0     ratio(terfc-metric)   ratio(coulomb-metric)");
    for &r0a in &[1.0_f64, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 50.0] {
        let op = Operator::terfc(r0a * A2B);
        let rt = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &cfg)
            .map(|r| r.0.e_total / ec);
        let rc = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &cfg_cmet)
            .map(|r| r.0.e_total / ec);
        match (rt, rc) {
            (Ok(a), Ok(b)) => eprintln!("{r0a:6.1}   {a:.8}   {b:.8}"),
            (a, b) => eprintln!("{r0a:6.1}   {a:?}  {b:?}"),
        }
    }
}
