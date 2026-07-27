//! What ROHF semi-canonicalization is worth to a real open-shell correlated method.
//!
//! Lives in ferric-cc rather than ferric-scf because ferric-mp2 depends on ferric-scf,
//! not the reverse, so the SCF crate's own tests cannot call U-RI-MP2.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::semicanonical::semicanonicalize;

/// OH radical, doublet.
fn oh_radical() -> Molecule {
    Molecule::parse_xyz("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap()
}

/// MEASURE what semi-canonicalization is worth to a real correlated method.
///
/// ferric's `u_ri_mp2` falls back to alpha orbital energies for BOTH spins on a ROHF
/// reference (`u_rimp2.rs:97`), because ROHF carries no `eps_beta`. Those are the
/// EFFECTIVE Roothaan Fock's eigenvalues, not either spin's, so the denominators are
/// wrong. Feeding it semi-canonical orbitals instead fixes that.
///
/// MEASURED (cc-pVDZ):
/// ```text
///        raw ROHF fallback   semi-canonical      shift      true UHF
///   OH      -0.1569792         -0.1522374      +4.74e-3    -0.1510030
///   CH3     -0.1348466         -0.1308388      +4.01e-3    -0.1290537
/// ```
///
/// ~4-5 mEh, i.e. ~3 kcal/mol -- chemically significant, and on the scale of the
/// "3 kcal/mol transition-metal chemical accuracy" the wB97X-L-V paper targets.
///
/// The DIRECTION is the evidence this is a fix rather than a perturbation: the
/// semi-canonical result moves TOWARD the true UHF value in both cases, because the
/// fallback's F_eff eigenvalues systematically understate the gap.
#[test]
fn semicanonicalization_measurably_corrects_open_shell_mp2() {
    use ferric_mp2::rimp2::RiMp2Config;
    use ferric_mp2::u_rimp2::u_ri_mp2;

    let ctx = ParallelContext::default();
    let mol = oh_radical();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let cfg = RhfConfig { density_conv: 1e-9, max_iter: 200, ..Default::default() };

    let rohf = solve_rohf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &cfg).unwrap();
    assert!(rohf.converged);
    let sc = semicanonicalize(&ctx, &mol, &obs, &bounds, &rohf, 1e-12, None).unwrap();
    let semi = sc.to_unrestricted_result(&rohf);

    let mp2 = |s: &ferric_scf::result::ScfResult| {
        u_ri_mp2(&mol, &obs, &dfbs, Operator::coulomb(), s, &RiMp2Config::default())
            .unwrap()
            .mp2_corr
    };
    let e_raw = mp2(&rohf);
    let e_semi = mp2(&semi);
    let e_uhf = ferric_scf::uhf::solve_uhf(&ctx, &mol, &obs, &bounds, &cfg).map(|u| mp2(&u));

    eprintln!("E_corr(raw ROHF fallback) = {e_raw:.10}");
    eprintln!("E_corr(semi-canonical)    = {e_semi:.10}   shift = {:+.3e}", e_semi - e_raw);
    if let Ok(u) = &e_uhf {
        eprintln!("E_corr(true UHF ref)      = {u:.10}");
    }

    // It must actually change the answer -- otherwise the routine is inert.
    assert!(
        (e_semi - e_raw).abs() > 1e-4,
        "semi-canonicalization changed U-MP2 by only {:.3e}; the correction is not \
         reaching the correlated method",
        e_semi - e_raw
    );

    // And it must move TOWARD the true UHF answer, not away.
    if let Ok(e_u) = e_uhf {
        let before = (e_raw - e_u).abs();
        let after = (e_semi - e_u).abs();
        eprintln!("|raw - UHF| = {before:.3e}   |semi - UHF| = {after:.3e}");
        assert!(
            after < before,
            "semi-canonical result ({e_semi:.10}) is FARTHER from the true UHF value \
             ({e_u:.10}) than the raw fallback ({e_raw:.10}) -- that is the wrong \
             direction for a correction"
        );
    }
}
