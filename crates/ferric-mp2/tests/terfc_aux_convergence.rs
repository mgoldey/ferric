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
#[ignore = "benchmark: terfc aux-basis convergence probe; --release --ignored --nocapture"]
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
    // Report-and-continue: the terfc (P|Q) metric is known to be poorly
    // conditioned in Coulomb-optimized aux bases, so a Cholesky failure on ONE
    // aux basis is expected data, not a reason to abort the sweep. (An earlier
    // version unwrap()'d here and lost the third row to a
    // `Lapack("Cholesky on (P|Q)")` error.)
    for auxname in ["cc-pvdz-ri", "def2-tzvpp-rifit", "aug-cc-pvtz-rifit"] {
        let Ok(auxbs) = basis::bundled(auxname) else {
            eprintln!("{auxname:18}  SKIP (not a bundled basis)");
            continue;
        };
        let dfbs = match PreparedBasis::new(&mol, &auxbs) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{auxname:18}  SKIP (PreparedBasis: {e})");
                continue;
            }
        };
        let naux = dfbs.nbasis();
        let ec = match ri_mp2_spin_components(&mol, &obs, &dfbs, opc, &rhf, &cfg) {
            Ok(r) => r.0.e_total,
            Err(e) => {
                eprintln!("{auxname:18} {naux:5}  Coulomb FAILED: {e}");
                continue;
            }
        };
        // r0 = 12 A: terfc is numerically indistinguishable from Coulomb as an
        // OPERATOR, so any residual gap here is pure RI-fit difference.
        let et = match ri_mp2_spin_components(
            &mol,
            &obs,
            &dfbs,
            Operator::terfc(12.0 * A2B),
            &rhf,
            &cfg,
        ) {
            Ok(r) => r.0.e_total,
            Err(e) => {
                eprintln!(
                    "{auxname:18} {naux:5}  {ec:.10}  {:+.3e}  terfc FAILED: {e}",
                    ec - e_exact
                );
                continue;
            }
        };
        eprintln!(
            "{auxname:18} {naux:5}  {ec:.10}  {:+.3e}  {et:.10}  {:+.3e}   |terfc-coul| {:.2e}",
            ec - e_exact,
            et - e_exact,
            (et - ec).abs()
        );
    }
}
