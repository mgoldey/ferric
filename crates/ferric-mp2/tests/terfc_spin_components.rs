//! SCRATCH diagnostic: which spin component overshoots? E_OS is a pure sum of
//! squares over a positive-definite-ish kernel and should be monotone in the
//! attenuation; E_SS carries the exchange-like difference
//! (ia|w|jb)[(ia|w|jb) - (ib|w|ja)] which has no sign theorem.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const A2B: f64 = 1.889_725_988_6;

#[test]
fn probe_terfc_spin_components() {
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

    let sc = ri_mp2_spin_components(&mol, &obs, &dfbs, opc, &rhf, &cfg).unwrap().0;
    let (os_c, ss_c) = (sc.e_os, sc.e_ss);
    eprintln!("Coulomb: e_os={os_c:.10}  e_ss={ss_c:.10}  tot={:.10}", sc.e_total);
    eprintln!("   r0        e_os        os/os_c        e_ss        ss/ss_c");
    for &r0a in &[0.75_f64, 1.05, 1.5, 2.0, 3.0, 6.0, 12.0] {
        let s = ri_mp2_spin_components(&mol, &obs, &dfbs, Operator::terfc(r0a * A2B), &rhf, &cfg)
            .unwrap()
            .0;
        eprintln!(
            "{r0a:6.2}  {:.10}  {:.8}  {:.10}  {:.8}",
            s.e_os,
            s.e_os / os_c,
            s.e_ss,
            s.e_ss / ss_c
        );
    }
}
