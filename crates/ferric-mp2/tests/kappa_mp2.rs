//! Anchors for κ-regularized RI-MP2 (Lee & Head-Gordon, JCTC 14, 5203
//! (2018)): amplitude damping (1 − e^{−κΔ})². Protocol: both trivial
//! limits (κ→∞ ≡ plain MP2, κ→0 ≡ no correlation) are TESTED, the
//! interior is pinned by an ANALYTIC single-pair identity on H2/STO-3G
//! (independent scalar construction from the RHF gap + the plain-MP2 OS
//! energy), and monotonicity in κ is asserted on water.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
}

fn setup(xyz: &str, obs_name: &str, aux_name: &str) -> Setup {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig { energy_conv: 1e-10, ..Default::default() },
    )
    .unwrap();
    Setup { mol, obs, dfbs, rhf }
}

fn e_total(su: &Setup, kappa: Option<f64>) -> f64 {
    ri_mp2_spin_components(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &RiMp2Config { kappa, ..Default::default() },
    )
    .unwrap()
    .0
    .e_total
}

/// Both trivial limits, on water/6-31G: κ = 1e6 must reproduce plain MP2
/// exactly (the damping underflows to 1), κ = 1e-6 must kill correlation
/// (damping ~ (κΔ)²).
#[test]
fn kappa_limits_recover_mp2_and_zero() {
    let su = setup("water.xyz", "6-31g", "cc-pvdz-ri");
    let e_plain = e_total(&su, None);
    let e_inf = e_total(&su, Some(1e6));
    let e_zero = e_total(&su, Some(1e-6));
    eprintln!("plain={e_plain:.12} kappa=1e6 -> {e_inf:.12} kappa=1e-6 -> {e_zero:.3e}");
    assert!((e_inf - e_plain).abs() < 1e-12, "kappa->inf limit broken");
    assert!(e_zero.abs() < 1e-9, "kappa->0 limit broken: {e_zero:.3e}");
}

/// Interior pinned analytically: H2/STO-3G has ONE pair, so
/// E(κ) = −K² (1 − e^{−2κΔ})² / (2Δ) with Δ the RHF gap and K obtained
/// from the PLAIN MP2 OS energy (e_os = −K²/2Δ) — an independent scalar
/// path sharing only the plain-MP2 number.
#[test]
fn h2_single_pair_matches_the_analytic_damping() {
    let su = setup("h2.xyz", "sto-3g", "sto-3g");
    let eps = su.rhf.eps_r();
    let delta = eps[1] - eps[0];
    let e_os_plain = ri_mp2_spin_components(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &RiMp2Config::default(),
    )
    .unwrap()
    .0
    .e_os;
    let k2 = -e_os_plain * 2.0 * delta; // K^2
    for kappa in [0.5, 1.0, 1.45, 3.0] {
        let e = e_total(&su, Some(kappa));
        let d1 = 1.0 - (-2.0 * kappa * delta).exp();
        let e_pred = -k2 * d1 * d1 / (2.0 * delta);
        let dev = (e - e_pred).abs();
        eprintln!("kappa={kappa}: E={e:.12} analytic={e_pred:.12} dev={dev:.3e}");
        assert!(dev < 1e-12, "analytic single-pair identity broken at kappa={kappa}");
    }
}

/// |E(κ)| must grow monotonically with κ (the damping is pointwise
/// monotone), and invalid κ must be rejected, not propagated.
#[test]
fn kappa_monotone_and_validated() {
    let su = setup("water.xyz", "6-31g", "cc-pvdz-ri");
    let es: Vec<f64> = [0.5, 1.0, 1.45, 2.0].iter().map(|&k| e_total(&su, Some(k))).collect();
    eprintln!("E(kappa) sweep: {es:?}");
    for w in es.windows(2) {
        assert!(w[0].abs() < w[1].abs(), "not monotone: {w:?}");
    }
    for bad in [0.0, -1.0, f64::NAN] {
        let r = ri_mp2_spin_components(
            &su.mol,
            &su.obs,
            &su.dfbs,
            Operator::coulomb(),
            &su.rhf,
            &RiMp2Config { kappa: Some(bad), ..Default::default() },
        );
        assert!(r.is_err(), "kappa={bad} not rejected");
    }
}
