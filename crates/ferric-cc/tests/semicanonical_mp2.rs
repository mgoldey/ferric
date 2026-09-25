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

/// The pre-fix treatment of a ROHF reference, rebuilt explicitly: the ROHF MOs and
/// the Roothaan EFFECTIVE Fock's eigenvalues for BOTH spins, labelled Unrestricted so
/// the correlated method uses them as given.
fn legacy_effective_fock_view(rohf: &ferric_scf::result::ScfResult) -> ferric_scf::result::ScfResult {
    let mut v = rohf.clone();
    v.spin = ferric_scf::result::Spin::Unrestricted;
    v.mos_beta = Some(rohf.mos_alpha.clone());
    v.eps_beta = Some(rohf.eps_alpha.clone());
    v
}

/// MEASURE what semi-canonicalization is worth to a real correlated method, and that
/// `u_ri_mp2` applies it to a ROHF reference itself.
///
/// Before the fix `u_ri_mp2` fed a ROHF reference the α orbitals and the EFFECTIVE
/// Roothaan Fock's eigenvalues for BOTH spins (reconstructed here by
/// `legacy_effective_fock_view`). Those belong to neither spin's Fock operator.
///
/// MEASURED (cc-pVDZ):
/// ```text
///        effective-Fock      semi-canonical      shift      true UHF
///   OH      -0.1569792         -0.1522374      +4.74e-3    -0.1510030
///   CH3     -0.1348466         -0.1308388      +4.01e-3    -0.1290537
/// ```
///
/// ~4-5 mEh, i.e. ~3 kcal/mol. The semi-canonical result moves TOWARD the true UHF
/// value, because the effective-Fock eigenvalues systematically understate the gap.
///
/// Two constructions of the semi-canonical value are compared: `u_ri_mp2` on the raw
/// ROHF result (which semi-canonicalizes with the SCF's stored spin Focks) and
/// `u_ri_mp2` on `semicanonicalize(..)` (which rebuilds `F_σ` from J/K).
#[test]
fn semicanonicalization_measurably_corrects_open_shell_mp2() {
    use ferric_mp2::rimp2::RiMp2Config;
    use ferric_mp2::u_rimp2::u_ri_mp2;

    let ctx = ParallelContext::default();
    let mol = oh_radical();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let cfg = RhfConfig {
        density_conv: 1e-9,
        max_iter: 200,
        ..Default::default()
    };

    let rohf = solve_rohf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &cfg).unwrap();
    assert!(rohf.converged);
    let sc = semicanonicalize(&ctx, &mol, &obs, &bounds, &rohf, 1e-12, None).unwrap();
    let semi = sc.to_unrestricted_result(&rohf);

    let mp2 = |s: &ferric_scf::result::ScfResult| {
        u_ri_mp2(
            &mol,
            &obs,
            &dfbs,
            Operator::coulomb(),
            s,
            &RiMp2Config::default(),
        )
        .unwrap()
        .mp2_corr
    };
    let e_legacy = mp2(&legacy_effective_fock_view(&rohf));
    let e_direct = mp2(&rohf);
    let e_semi = mp2(&semi);
    let e_uhf = ferric_scf::uhf::solve_uhf(&ctx, &mol, &obs, &bounds, &cfg).map(|u| mp2(&u));

    eprintln!("E_corr(effective-Fock, legacy) = {e_legacy:.10}");
    eprintln!("E_corr(u_ri_mp2 on ROHF)       = {e_direct:.10}");
    eprintln!(
        "E_corr(semicanonicalize rebuilt) = {e_semi:.10}   shift vs legacy = {:+.3e}",
        e_semi - e_legacy
    );
    if let Ok(u) = &e_uhf {
        eprintln!("E_corr(true UHF ref)           = {u:.10}");
    }

    // u_ri_mp2 on a raw ROHF result IS the semi-canonical value (two independent
    // Fock constructions; they differ only at the SCF density-convergence level).
    assert!(
        (e_direct - e_semi).abs() < 1e-7,
        "u_ri_mp2 on ROHF ({e_direct:.10}) is not the semi-canonical value ({e_semi:.10})"
    );

    // Semi-canonicalization must actually change the answer vs the legacy treatment.
    assert!(
        (e_semi - e_legacy).abs() > 1e-4,
        "semi-canonicalization changed U-MP2 by only {:.3e}; the correction is not \
         reaching the correlated method",
        e_semi - e_legacy
    );

    // And it must move TOWARD the true UHF answer, not away.
    if let Ok(e_u) = e_uhf {
        let before = (e_legacy - e_u).abs();
        let after = (e_semi - e_u).abs();
        eprintln!("|legacy - UHF| = {before:.3e}   |semi - UHF| = {after:.3e}");
        assert!(
            after < before,
            "semi-canonical result ({e_semi:.10}) is FARTHER from the true UHF value \
             ({e_u:.10}) than the legacy effective-Fock one ({e_legacy:.10}) -- that is the \
             wrong direction for a correction"
        );
    }
}
