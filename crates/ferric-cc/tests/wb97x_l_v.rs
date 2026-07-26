//! ωB97X-L-V double-hybrid end-to-end tests.
//!
//! Ransford & Carter-Fenk, PCCP 2026, 28, 14428 (`papers/wb97xlv.pdf`).
//!   E = E_KS[ωB97X-L] + E_c,VV10 + λ·E_c,LinLCCD(hh)^{sr,ω}      (eqn 27)
//!
//! What these DO test: the assembly is correct and internally consistent -- the SCF
//! converges with the custom functional, the SR-attenuated correlation is finite and
//! sensibly signed, λ and ω enter where the paper says, and an unconverged reference
//! is refused rather than silently used.
//!
//! What these do NOT test: agreement with the paper's published GMTKN55 statistics.
//! Those require Def2-ma-TZVPP (not bundled) and a 1750-point benchmark set. No claim
//! of reproducing the functional's benchmark performance is made here.

use ferric_cc::double_hybrid::{
    run_wb97x_l_v, solve_wb97x_l_v, DoubleHybridConfig, WB97X_L_V_LAMBDA, WB97X_L_V_NAME,
    WB97X_L_V_OMEGA,
};
use ferric_cc::linlccd::LadderVariant;
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;

fn mol_path(name: &str) -> String {
    format!("{}/../../testdata/molecules/{}", env!("CARGO_MANIFEST_DIR"), name)
}

/// The full chain: name -> KS ladder -> converged density -> SR correlation -> total.
#[test]
fn water_end_to_end() {
    let mol = Molecule::load_xyz(&mol_path("water.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();

    let (dh, ks) = run_wb97x_l_v(
        &ParallelContext::default(),
        &mol,
        &obs,
        &dfbs,
        &bounds,
        &RhfConfig::default(),
        &DoubleHybridConfig::default(),
    )
    .expect("wB97X-L-V should run end to end on water");

    eprintln!("E_KS[wB97X-L-V]     = {:.10}", dh.e_ks);
    eprintln!("E_c,LinLCCD(hh) SR  = {:.10}", dh.e_c_wft);
    eprintln!("lambda * E_c        = {:.10}", dh.e_c_scaled);
    eprintln!("E_total             = {:.10}", dh.total_energy);
    eprintln!("SCF iterations      = {}", ks.iterations);

    assert!(ks.converged, "KS reference must be converged");
    assert!(dh.total_energy.is_finite());
    // Water/cc-pVDZ total energy sits near -76 Ha for any sane functional.
    assert!(
        (-77.5..-75.0).contains(&dh.total_energy),
        "total energy {:.6} is outside the physically plausible range for water/cc-pVDZ",
        dh.total_energy
    );
    // Correlation must be negative and a small fraction of the total.
    assert!(dh.e_c_wft < 0.0, "WFT correlation must be negative, got {:.10}", dh.e_c_wft);
    assert!(
        dh.e_c_wft.abs() < 1.0,
        "SR correlation {:.10} is implausibly large",
        dh.e_c_wft
    );
    // The reported decomposition must actually add up.
    assert!(
        (dh.total_energy - (dh.e_ks + dh.e_c_scaled)).abs() < 1e-12,
        "reported components do not sum to the reported total"
    );
    assert!((dh.e_c_scaled - dh.lambda * dh.e_c_wft).abs() < 1e-12);
    assert!((dh.lambda - WB97X_L_V_LAMBDA).abs() < 1e-12);
    assert!((dh.omega - WB97X_L_V_OMEGA).abs() < 1e-12);
}

/// ω must reach the operator, and the ω → 0 limit must recover full Coulomb.
///
/// NOTE ON DIRECTION -- attenuation does NOT monotonically shrink LinLCCD(hh)
/// correlation, which is worth stating because the naive expectation is wrong.
/// Measured on water/cc-pVDZ (E_c relative to the Coulomb value):
///
/// ```text
///   omega     MP2 (drivers only)   LinLCCD(hh)
///   1e-6            1.0000            1.0000
///   0.05            0.9997            1.0113
///   0.1             0.9976            1.0207
///   0.2             0.9807            1.0255   <- maximum
///   0.3             0.9372            0.9990
///   1.0             0.3194            0.3621
/// ```
///
/// MP2 falls monotonically, as attenuation should. LinLCCD(hh) RISES to ~1.026 near
/// omega = 0.2 before falling. The two differ only by the hh-ladder term, which
/// isolates the cause: attenuating the operator also weakens the screening integrals
/// v_ij^kl that widen the effective gap (eqn 15), and larger amplitudes from weaker
/// screening outweigh the lost long-range correlation at small omega. At the paper's
/// omega = 0.1 the SR correlation is ~1.02x the Coulomb value, NOT smaller.
///
/// So this test asserts the two things that must hold -- omega reaches the operator,
/// and omega -> 0 recovers Coulomb -- rather than a monotonicity that is false.
#[test]
fn omega_reaches_the_operator_and_zero_recovers_coulomb() {
    let mol = Molecule::load_xyz(&mol_path("water.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();

    let mut scf = RhfConfig::default();
    scf.xc = Some(WB97X_L_V_NAME.to_string());
    scf.df_j_aux = Some("def2-universal-jkfit".to_string());
    scf.df_k_aux = Some("def2-universal-jkfit".to_string());
    let ladder = ferric_scf::ladder::ksdft_ladder(&scf);
    let lr = ferric_scf::ladder::solve_rhf_ladder(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &ladder,
    )
    .unwrap();
    assert!(lr.converged);

    let at = |omega: f64| {
        solve_wb97x_l_v(
            &mol,
            &obs,
            &dfbs,
            &lr.result,
            &DoubleHybridConfig { omega, ..Default::default() },
        )
        .unwrap()
        .e_c_wft
    };

    // omega -> 0: erfc(omega*r)/r -> 1/r, so this must reproduce full Coulomb.
    let near_zero = at(1e-6);
    let coulomb = ferric_cc::linlccd::linlccd(
        &mol,
        &obs,
        &dfbs,
        Operator::coulomb(),
        &lr.result,
        &DoubleHybridConfig::default().cc,
        LadderVariant::Hh,
    )
    .unwrap()
    .correlation_energy;

    let paper = at(WB97X_L_V_OMEGA);
    let strong = at(1.0);

    eprintln!("E_c (true Coulomb)     = {coulomb:.10}");
    eprintln!("E_c (omega = 1e-6)     = {near_zero:.10}");
    eprintln!("E_c (omega = 0.1)      = {paper:.10}   ratio {:.4}", paper / coulomb);
    eprintln!("E_c (omega = 1.0)      = {strong:.10}   ratio {:.4}", strong / coulomb);

    assert!(
        (near_zero - coulomb).abs() < 1e-6,
        "omega -> 0 must recover full Coulomb: {near_zero:.10} vs {coulomb:.10}"
    );
    // Strong attenuation must remove most of the correlation -- this is the regime
    // where the short-range restriction unambiguously bites.
    assert!(
        strong.abs() < 0.5 * coulomb.abs(),
        "omega = 1.0 should strip most correlation, got {strong:.10} vs Coulomb {coulomb:.10}"
    );
    // And omega must actually reach the operator at the paper's value.
    assert!(
        (paper - coulomb).abs() > 1e-6,
        "omega = {WB97X_L_V_OMEGA} produced the Coulomb answer -- omega is being ignored"
    );
}

/// λ must scale the correlation contribution linearly, and λ = 0 must reduce the
/// double hybrid EXACTLY to its underlying range-separated hybrid.
///
/// This is the cleanest available check that the WFT half is added where the paper
/// says it is: at λ = 0 there is no wave-function correlation at all, so the total
/// must equal the bare KS energy.
#[test]
fn lambda_scales_correlation_and_zero_recovers_the_hybrid() {
    let mol = Molecule::load_xyz(&mol_path("water.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();

    let mut scf = RhfConfig::default();
    scf.xc = Some(WB97X_L_V_NAME.to_string());
    scf.df_j_aux = Some("def2-universal-jkfit".to_string());
    scf.df_k_aux = Some("def2-universal-jkfit".to_string());
    let ladder = ferric_scf::ladder::ksdft_ladder(&scf);
    let lr = ferric_scf::ladder::solve_rhf_ladder(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &ladder,
    )
    .unwrap();

    let at = |lambda: f64| {
        solve_wb97x_l_v(
            &mol,
            &obs,
            &dfbs,
            &lr.result,
            &DoubleHybridConfig { lambda, ..Default::default() },
        )
        .unwrap()
    };

    let zero = at(0.0);
    assert!(
        (zero.total_energy - lr.result.energy).abs() < 1e-12,
        "lambda = 0 must recover the bare KS energy exactly: {:.12} vs {:.12}",
        zero.total_energy,
        lr.result.energy
    );

    // Linear in lambda at fixed amplitudes: E(2c) - E(c) must be constant.
    let (a, b, c) = (at(0.2), at(0.4), at(0.6));
    let d1 = b.total_energy - a.total_energy;
    let d2 = c.total_energy - b.total_energy;
    eprintln!("E(0.2) = {:.10}  E(0.4) = {:.10}  E(0.6) = {:.10}", a.total_energy, b.total_energy, c.total_energy);
    assert!(
        (d1 - d2).abs() < 1e-12,
        "lambda scaling must be linear at fixed amplitudes: {d1:.3e} vs {d2:.3e}"
    );
    assert!(d1.abs() > 1e-6, "lambda had no effect -- correlation is not being added");
}

/// An unconverged reference must be REFUSED, not silently used.
///
/// This is the guard that matters most in practice. `solve_rhf` returns `Ok` with
/// `converged: false`, and RI-MP2 / CC / RPA / GW all consume an `ScfResult` without
/// checking it -- so a stuck SCF otherwise flows straight into a plausible-looking
/// correlation energy. Here we force non-convergence with a 1-iteration cap.
#[test]
fn unconverged_reference_is_refused() {
    let mol = Molecule::load_xyz(&mol_path("water.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();

    let scf = RhfConfig { max_iter: 1, density_conv: 1e-14, ..Default::default() };
    let stuck = ferric_scf::rhf::solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &scf,
    )
    .expect("solve_rhf returns Ok even when it does not converge");
    assert!(!stuck.converged, "test premise: this SCF must NOT have converged");

    let err = solve_wb97x_l_v(&mol, &obs, &dfbs, &stuck, &DoubleHybridConfig::default());
    assert!(
        err.is_err(),
        "an unconverged reference must be refused, but the driver returned a result"
    );
}

/// Nonsensical λ / ω are rejected up front rather than producing quiet garbage.
#[test]
fn invalid_parameters_are_rejected() {
    let mol = Molecule::load_xyz(&mol_path("water.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let rhf = ferric_scf::rhf::solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig { density_conv: 1e-8, ..Default::default() },
    )
    .unwrap();
    assert!(rhf.converged);

    let bad = |cfg: DoubleHybridConfig| solve_wb97x_l_v(&mol, &obs, &dfbs, &rhf, &cfg).is_err();

    assert!(bad(DoubleHybridConfig { lambda: -0.1, ..Default::default() }), "negative lambda");
    assert!(bad(DoubleHybridConfig { lambda: 1.5, ..Default::default() }), "lambda > 1");
    assert!(bad(DoubleHybridConfig { omega: 0.0, ..Default::default() }), "omega = 0");
    assert!(bad(DoubleHybridConfig { omega: -0.1, ..Default::default() }), "negative omega");
}

/// The `Full` LinLCCD variant is reachable through the double-hybrid config.
///
/// Not the published functional (which uses `Hh`), but it must not be silently
/// ignored -- a config knob that does nothing is worse than no knob.
#[test]
fn ladder_variant_is_honored() {
    let mol = Molecule::load_xyz(&mol_path("water.xyz")).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let rhf = ferric_scf::rhf::solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig { density_conv: 1e-8, ..Default::default() },
    )
    .unwrap();

    let cc = CcConfig { energy_conv: 1e-10, max_iter: 100, ..Default::default() };
    let run = |variant: LadderVariant| {
        solve_wb97x_l_v(
            &mol,
            &obs,
            &dfbs,
            &rhf,
            &DoubleHybridConfig { variant, cc: cc.clone(), ..Default::default() },
        )
        .unwrap()
        .e_c_wft
    };

    let hh = run(LadderVariant::Hh);
    let full = run(LadderVariant::Full);
    eprintln!("E_c SR: Hh = {hh:.10}   Full = {full:.10}");
    assert!(
        (hh - full).abs() > 1e-9,
        "Hh and Full gave the same energy ({hh:.10}) -- the variant knob is inert"
    );
}
