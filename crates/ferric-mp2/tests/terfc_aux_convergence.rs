//! SCRATCH diagnostic: absolute aux-basis convergence of BOTH E_coul and
//! E_terfc(r0), against the exact 4-index Coulomb MP2. Distinguishes
//!   (a) "terfc-RI converges to Coulomb-RI"  [expected, benign]
//! from
//!   (b) "terfc-RI converges somewhere else" [would be a real defect]
//! by printing absolute energies, not ratios.
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
fn probe_aux_convergence_absolute() {
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

    let e_exact = canonical_mp2(&mol, &obs, opc, &rhf, 0).unwrap();
    eprintln!("EXACT 4-index Coulomb MP2 = {e_exact:.10}");
    eprintln!();
    eprintln!("aux                 naux    E_coul(RI)     err_coul    E_terfc(12A)    err_terfc");
    for auxname in ["cc-pvdz-ri", "def2-tzvpp-rifit", "aug-cc-pvtz-rifit"] {
        let Ok(auxbs) = basis::bundled(auxname) else { continue };
        let Ok(dfbs) = PreparedBasis::new(&mol, &auxbs) else { continue };
        let ec = ri_mp2_spin_components(&mol, &obs, &dfbs, opc, &rhf, &cfg)
            .unwrap()
            .0
            .e_total;
        // r0 = 12 A: terfc is numerically indistinguishable from Coulomb as an
        // OPERATOR, so any residual gap here is pure RI-fit difference.
        let et = ri_mp2_spin_components(&mol, &obs, &dfbs, Operator::terfc(12.0 * A2B), &rhf, &cfg)
            .unwrap()
            .0
            .e_total;
        eprintln!(
            "{auxname:18} {:5}  {ec:.10}  {:+.3e}  {et:.10}  {:+.3e}",
            dfbs.nbasis(),
            ec - e_exact,
            et - e_exact
        );
    }
}
