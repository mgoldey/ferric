//! Open-shell terf, formulation B: a smoke test and the large-r0 anchor.
//!
//! OH doublet, UHF, STO-3G / cc-pVDZ-RI, formulation B with terf at
//! r0 = 10 and 20 Bohr (omega = 1/(r0 sqrt 2) = 0.071 and 0.035 Bohr^-1). The
//! long-range kernel is K_lr = (Coulomb RI) - (terfc RI) with terfc -> Coulomb,
//! so K_lr -> 0 and E_B = E_MP2[C] + dE_dRPA[lr] must tend to U-RI-MP2: the
//! bracket dE_dRPA[lr] = E_dRPA[lr] - dMP2[lr] starts at third order in K_lr
//! (`rsrpa.forms.limits`).
//!
//! # What was measured (run on this test's commit, quiet box)
//!
//! | r0 (Bohr) | e_mp2_full - U-MP2 | dMP2[lr] | E_dRPA[lr] | B - U-MP2 |
//! |---|---|---|---|---|
//! | 10 | < 5e-16 | -5.1e-12 | -7.5e-12 | -2.4e-12 |
//! | 20 | < 5e-16 | -3.7e-15 | -2.7e-12 | -2.7e-12 |
//!
//! `e_mp2_full` is the Coulomb RI-MP2 energy of the same intermediates and
//! equals `u_ri_mp2` to rounding, so the whole difference is the bracket.
//! The second-order ring dMP2[lr] falls by 1.4e3 between r0 = 10 and 20, as
//! K_lr^2 must (K_lr falls at least 10-fold when r0 doubles: terfc -> Coulomb).
//! The bracket does NOT fall: it is -2.4e-12 and -2.7e-12. A third-order
//! quantity would be |K|^3/(4 D^2) = |dMP2| |K|/D, below dMP2 = 4e-15 at
//! r0 = 20, so what remains is a floor of the signed log-det that does not
//! depend on r0. Its origin is the cancellation the signed route performs:
//! its summand ln det(1 + Sigma P) - tr(Sigma P) subtracts Coulomb-sized blocks
//! B_C and B_s that are equal to within K_lr. The floor is a measurement here,
//! not a derived number; the closed-shell analogue on H2 / cc-pVDZ is 8.2e-13.
//!
//! # Bars
//!
//! * `|B - U-MP2| < 1e-10` Ha at both r0: 37 times the larger measured floor
//!   (2.7e-12), so BLAS or LU reordering that moves the floor by a few times
//!   does not fail it, and 1.5e9 times smaller than the signal of a wrong
//!   sign (1.5e-1 Ha, below).
//! * `|dMP2[lr](20)| < 1e-12` and `|e_lr_mp2(20)| < 1e-12`: the kernel is
//!   second-order small (measured 3.7e-15 and 1.8e-15).
//! * `|dMP2[lr](10)| > 100 |dMP2[lr](20)|`: K_lr falls at least 10-fold.
//!
//! The test cannot resolve the third-order bracket under the floor. It checks
//! that B reaches U-MP2 to 1e-10 and that the long-range part is the small
//! quantity the argument says; it catches a wrong sign or a lost block.
//!
//! # Mutation
//!
//! Flipping the signed pair block to `+` (`general_mat_mul(1.0, ..)` in
//! `ferric_mp2::rimp2::signed_pair_block`) gives, at r0 = 10, dMP2[lr] =
//! -1.5e-1 and B - U-MP2 = +1.5e-1 Ha; the first bar fails by assertion.
//!
//! Needs the terf interpolation tables (`FERRIC_TERF_TABLE_DIR`); skips with a
//! note without them.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::RiMp2Config;
use ferric_rpa::rs_mp2_rpa::{rs_mp2_lr_rpa, Attenuator, RsMp2RpaConfig, RsMp2RpaFormulation};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";

fn tables() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR")
        .map(|d| std::path::Path::new(&d).join("16_4_2.bin").exists())
        .unwrap_or(false)
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    scf: ScfResult,
}

fn oh_uhf() -> Sys {
    let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let scf = ferric_scf::uhf::solve_uhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        &bounds,
        &RhfConfig::default()
            .with_energy_conv(1e-11)
            .with_density_conv(1e-9)
            .with_max_iter(300),
    )
    .unwrap();
    assert!(scf.converged);
    Sys {
        mol,
        obs,
        dfbs,
        scf,
    }
}

fn terf_b(r0: f64) -> RsMp2RpaConfig {
    RsMp2RpaConfig {
        attenuator: Attenuator::Terf,
        formulation: RsMp2RpaFormulation::DeltaLr,
        r0,
        ..Default::default()
    }
}

/// Open-shell terf B tends to U-RI-MP2 as r0 grows (see the module header).
#[test]
fn open_shell_terf_b_tends_to_ump2_at_large_r0() {
    if !tables() {
        eprintln!("SKIP open_shell_terf_b_tends_to_ump2_at_large_r0: terf tables absent");
        return;
    }
    let sys = oh_uhf();
    let ump2 = ferric_mp2::u_rimp2::u_ri_mp2(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        &sys.scf,
        &RiMp2Config::default(),
    )
    .unwrap()
    .mp2_corr;
    let at = |r0: f64| {
        rs_mp2_lr_rpa(&sys.mol, &sys.obs, &sys.dfbs, &sys.scf, &terf_b(r0))
            .expect("open-shell terf B runs")
    };
    let (r10, r20) = (at(10.0), at(20.0));
    for (r0, r) in [(10.0, &r10), (20.0, &r20)] {
        let drpa = r.e_drpa_lr.expect("B computes the long-range dRPA");
        eprintln!(
            "OH terf B r0={r0}: U-MP2 {ump2:.12}  B {:.12}  |B-U-MP2| {:.3e}  \
             |e_mp2_full-U-MP2| {:.3e}  dmp2_lr {:.3e}  drpa_lr {drpa:.3e}  e_lr_mp2 {:.3e}",
            r.e_corr,
            (r.e_corr - ump2).abs(),
            (r.e_mp2_full - ump2).abs(),
            r.e_dmp2_lr,
            r.e_lr_mp2
        );
        // Smoke: finite correlation energy, total = reference + correlation.
        assert!(r.e_corr.is_finite() && r.e_corr < 0.0);
        assert!((r.total_energy - (sys.scf.energy + r.e_corr)).abs() < 1e-12);
        assert!(
            (r.e_mp2_full - ump2).abs() < 1e-13,
            "r0 = {r0}: Coulomb part"
        );
        assert!((r.e_corr - ump2).abs() < 1e-10, "r0 = {r0}: |B - U-MP2|");
    }
    assert!(r20.e_dmp2_lr.abs() < 1e-12 && r20.e_lr_mp2.abs() < 1e-12);
    assert!(
        r10.e_dmp2_lr.abs() > 100.0 * r20.e_dmp2_lr.abs(),
        "dMP2[lr] does not fall as K_lr^2: {:.3e} vs {:.3e}",
        r10.e_dmp2_lr,
        r20.e_dmp2_lr
    );
}
