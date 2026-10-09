//! The SCF two-electron integrals must be exact to double precision.
//!
//! `ERI_PRECISION` is libint2's primitive-screening threshold for every J/K
//! build. At the previous 1e-14 it put CS2/cc-pVDZ's Σ D·K off by 7.6e-10 and
//! E_J by 4.7e-11 against unscreened integrals (precision 0), and free
//! third-row atoms' E_J off by up to 6.0e-9 Ha. At 1e-20 both agree with
//! precision 0 to ≤6e-14 here. The bar sits between the two.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine_pool::{EnginePool, ERI_PRECISION};
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{build_jk_with_pool, solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

fn jk_energies(
    ctx: &ParallelContext,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    d: &Array2<f64>,
    precision: f64,
) -> (f64, f64) {
    let pool = EnginePool::new(Operator::coulomb(), prep, precision).unwrap();
    let mut j = Array2::<f64>::zeros(d.raw_dim());
    let mut k = Array2::<f64>::zeros(d.raw_dim());
    build_jk_with_pool(
        ctx,
        prep,
        bounds,
        RhfConfig::default().integral_thresh,
        d,
        &mut j,
        &mut k,
        &pool,
        ferric_scf::reduce::default_band_bytes(),
    )
    .unwrap();
    (0.5 * (d * &j).sum(), (d * &k).sum())
}

#[test]
fn scf_eri_precision_matches_unscreened_integrals() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/molecules");
    let mol = Molecule::load_xyz(&format!("{root}/cs2.xyz")).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        energy_conv: 1e-8,
        density_conv: 1e-6,
        ..Default::default()
    };
    let d = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg)
        .unwrap()
        .density_total
        .clone();

    let (ej0, ek0) = jk_energies(&ctx, &prep, &bounds, &d, 0.0);
    let (ej, ek) = jk_energies(&ctx, &prep, &bounds, &d, ERI_PRECISION);
    eprintln!(
        "CS2/cc-pVDZ at ERI_PRECISION={ERI_PRECISION:e}: E_J {:+.2e}, sum(D K) {:+.2e} vs precision 0",
        ej - ej0,
        ek - ek0
    );
    assert!(
        (ej - ej0).abs() < 1e-12,
        "E_J at ERI_PRECISION misses unscreened integrals by {:.2e}",
        ej - ej0
    );
    assert!(
        (ek - ek0).abs() < 1e-12,
        "sum(D K) at ERI_PRECISION misses unscreened integrals by {:.2e} (1e-14 gave 7.6e-10)",
        ek - ek0
    );
}
