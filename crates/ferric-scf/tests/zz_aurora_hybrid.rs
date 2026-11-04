//! Does AURORA help on HYBRID functionals? The gap in the default decision.
//!
//! The head-to-head covered closed-shell RHF only; the PBE row silently ran
//! pure DIIS because a_x = 0.0 is below MIN_VALIDATED_EXCHANGE_FRACTION.
//! B3LYP (a_x = 0.20) and PBE0 (0.25) are the common production choices and
//! sit inside the validated regime.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::aurora::{target_jk_builds, AuroraConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::time::Instant;

fn run(xyz: &str, bas: &str, xc: &str, aurora: bool) -> (usize, usize, f64, f64, bool) {
    let mol = Molecule::load_xyz(xyz).unwrap();
    let bs = basis::bundled(bas).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        xc: Some(xc.into()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        aurora: AuroraConfig {
            enabled: aurora,
            ..Default::default()
        },
        ..Default::default()
    };
    let jk0 = target_jk_builds();
    let t0 = Instant::now();
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    (
        r.iterations,
        target_jk_builds() - jk0,
        t0.elapsed().as_secs_f64(),
        r.energy,
        r.converged,
    )
}

#[test]
#[ignore]
fn zz_aurora_on_hybrids() {
    println!(
        "\n{:<28}{:>7}{:>7}{:>8}{:>7}{:>7}{:>8}{:>10}",
        "system", "it_D", "JK_D", "t_D", "it_A", "JK_A", "t_A", "dE"
    );
    for (name, xyz, bas, xc) in [
        (
            "water/cc-pVDZ   B3LYP",
            "../../testdata/molecules/water.xyz",
            "cc-pvdz",
            "B3LYP",
        ),
        (
            "methane/cc-pVDZ B3LYP",
            "../../testdata/molecules/methane.xyz",
            "cc-pvdz",
            "B3LYP",
        ),
        (
            "benzene/STO-3G  B3LYP",
            "../../testdata/molecules/benzene.xyz",
            "sto-3g",
            "B3LYP",
        ),
    ] {
        let (id, jd, td, ed, cd) = run(xyz, bas, xc, false);
        let (ia, ja, ta, ea, ca) = run(xyz, bas, xc, true);
        println!(
            "{name:<28}{id:>7}{jd:>7}{td:>8.2}{ia:>7}{ja:>7}{ta:>8.2}{:>10.1e}",
            (ea - ed).abs()
        );
        assert!(cd && ca, "{name}: both arms must converge (D={cd} A={ca})");
        assert!((ea - ed).abs() < 1e-7, "{name}: AURORA moved the answer");
    }
}
