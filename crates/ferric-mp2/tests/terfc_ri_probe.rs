//! SCRATCH diagnostic: is the terfc far-field energy overshoot an RI-basis
//! artefact or an integral error? Vary the aux basis at fixed orbital basis.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::screening::SchwarzBounds;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};

const A2B: f64 = 1.889_725_988_6;

#[test]
#[ignore = "benchmark: terfc RI probe; --release --ignored --nocapture"]
fn probe_terfc_overshoot_vs_aux_basis() {
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

    for auxname in ["cc-pvdz-ri", "def2-tzvpp-rifit", "aug-cc-pvtz-rifit"] {
        let Ok(auxbs) = basis::bundled(auxname) else {
            eprintln!("aux {auxname}: unavailable");
            continue;
        };
        let dfbs = match PreparedBasis::new(&mol, &auxbs) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("aux {auxname}: prep failed {e}");
                continue;
            }
        };
        let e = |op: Operator| {
            ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &cfg)
                .map(|r| r.0.e_total)
        };
        let ec = e(opc).unwrap();
        eprint!("aux={auxname:14} naux={:5} E_coul={ec:.8}", dfbs.nbasis());
        for &r0 in &[2.0_f64, 3.0] {
            match e(Operator::terfc(r0 * A2B)) {
                Ok(et) => eprint!("  r0={r0}: ratio={:.8}", et / ec),
                Err(err) => eprint!("  r0={r0}: ERR {err}"),
            }
        }
        eprintln!();
    }
}
