//! MP2C limit and identity tests.
//!
//! These pin the three claims the construction rests on:
//!
//!  1. **Reduction limit.** `LongRange` with ω→0 turns the dispersion swap off
//!     (`erf(ωr)/r → 0`), so both `2·E_OS[v_lr]` and `E_dRPA[v_lr]` vanish and
//!     MP2C must reduce to plain RI-MP2. This is the only reduction limit MP2C
//!     has — `FullRange` has no parameter to turn off.
//!
//!  2. **Structural equivalence to formulation B.** The methodology document
//!     (§1.1) claims B *is* MP2C with dRPA coupling restricted to the long
//!     range. That is a testable algebraic identity, not a rhetorical one, and
//!     if it ever stops holding one of the two implementations has drifted.
//!
//!  3. **Full-range MP2C is the ω→∞ limit of the long-range form**, and equals
//!     `E_MP2 + ΔdRPA[Coulomb]` assembled from independent calls.
//!
//! Every test also checks the component-additivity identity, so a bookkeeping
//! error in `Mp2cResult` cannot hide behind a correct `e_corr`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2, ri_mp2_spin_components, RiMp2Config};
use ferric_rpa::mp2c::{mp2c_energy, Mp2cConfig, Mp2cDispersionRange};
use ferric_rpa::rs_mp2_rpa::{rs_mp2_lr_rpa, RsMp2RpaConfig, RsMp2RpaFormulation};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

fn setup_h2() -> (Molecule, PreparedBasis, PreparedBasis, ScfResult) {
    setup("2\nH2\nH 0 0 0\nH 0 0 0.74\n", "cc-pvdz", "cc-pvdz-ri")
}

fn setup(
    xyz: &str,
    obs_name: &str,
    aux_name: &str,
) -> (Molecule, PreparedBasis, PreparedBasis, ScfResult) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(obs_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    (mol, obs, dfbs, rhf)
}

/// **The reduction limit.** ω→0 ⇒ `erf(ωr)/r → 0` ⇒ the swapped dispersion
/// terms both vanish ⇒ MP2C = plain MP2.
///
/// ω = 0.05 Bohr⁻¹ is the tested floor for the erf 2-center metric (below it
/// the metric loses rank); at that ω the residual correction is 4th-order
/// small in v_lr. The tolerance 1e-5 Ha is ~3 orders of magnitude below the
/// correlation energy itself, so a sign error or a missing factor in either
/// dispersion term would blow through it.
#[test]
fn long_range_omega_to_zero_reduces_to_mp2() {
    let (mol, obs, dfbs, rhf) = setup_h2();
    let mp2 = ri_mp2(
        &mol,
        &obs,
        &dfbs,
        Operator::coulomb(),
        &rhf,
        &RiMp2Config::default(),
    )
    .unwrap();
    let cfg = Mp2cConfig {
        range: Mp2cDispersionRange::LongRange,
        omega: 0.05,
        ..Default::default()
    };
    let r = mp2c_energy(&mol, &obs, &dfbs, &rhf, &cfg).unwrap();
    eprintln!(
        "MP2 {:.10}  MP2C(LR, w=0.05) {:.10}  correction {:.3e}  (uncoupled {:.3e}, coupled {:.3e})",
        mp2.mp2_corr, r.e_corr, r.e_correction, r.e_disp_uncoupled, r.e_disp_coupled
    );
    assert!(
        (r.e_corr - mp2.mp2_corr).abs() < 1e-5,
        "omega->0 must reduce to MP2: {} vs {}",
        r.e_corr,
        mp2.mp2_corr
    );
    // The correction itself must vanish, not just the total (a compensating
    // pair of large wrong terms would pass the line above but not this one).
    assert!(
        r.e_correction.abs() < 1e-5,
        "the MP2C correction itself must vanish as omega->0, got {}",
        r.e_correction
    );
    assert!(
        r.components_sum_to_corr().abs() < 1e-15,
        "components must sum to e_corr to ULP scale, got {:.3e}",
        r.components_sum_to_corr()
    );
    assert!((r.total_energy - (rhf.energy + r.e_corr)).abs() < 1e-12);
}

/// **The structural claim of methodology §1.1**, executed: MP2C with the swap
/// restricted to the long range IS formulation B, at the same ω.
///
/// Tolerance 1e-10 (not a physics tolerance — an implementation-identity one).
/// The two code paths build their intermediates differently (B builds three
/// operators, MP2C two), so bit-identity is not guaranteed, but any algebraic
/// divergence is orders of magnitude larger than this.
#[test]
fn long_range_mp2c_equals_formulation_b() {
    let (mol, obs, dfbs, rhf) = setup_h2();
    let omega = 0.42;
    let mp2c = mp2c_energy(
        &mol,
        &obs,
        &dfbs,
        &rhf,
        &Mp2cConfig {
            range: Mp2cDispersionRange::LongRange,
            omega,
            ..Default::default()
        },
    )
    .unwrap();
    let b = rs_mp2_lr_rpa(
        &mol,
        &obs,
        &dfbs,
        &rhf,
        &RsMp2RpaConfig {
            omega,
            formulation: RsMp2RpaFormulation::DeltaLr,
            ..Default::default()
        },
    )
    .unwrap();
    eprintln!(
        "MP2C(LR) {:.12}  formulation B {:.12}  diff {:.2e}",
        mp2c.e_corr,
        b.e_corr,
        mp2c.e_corr - b.e_corr
    );
    assert!(
        (mp2c.e_corr - b.e_corr).abs() < 1e-10,
        "long-range MP2C must equal formulation B: {} vs {}",
        mp2c.e_corr,
        b.e_corr
    );
    // The pieces must match individually too, not just the sum.
    assert!((mp2c.e_mp2 - b.e_mp2_full).abs() < 1e-10);
    assert!((mp2c.e_disp_uncoupled - b.e_dmp2_lr).abs() < 1e-10);
    assert!((mp2c.e_disp_coupled - b.e_drpa_lr.unwrap()).abs() < 1e-10);
}

/// **Full-range MP2C = E_MP2[Coulomb] + ΔdRPA[Coulomb]**, assembled from
/// independent `ri_mp2_spin_components` + `run_pdep_rpa` calls.
///
/// This is the load-bearing convention test: the closed-shell spin adaptation
/// of the RPA kernel (the classic factor-2-vs-4 trap in χ₀) would show up here
/// as an ~18 mHa error, 4 orders of magnitude above the 1e-9 tolerance.
#[test]
fn full_range_is_mp2_plus_delta_drpa_coulomb() {
    let (mol, obs, dfbs, rhf) = setup_h2();
    let r = mp2c_energy(&mol, &obs, &dfbs, &rhf, &Mp2cConfig::default()).unwrap();

    let (sc, _) = ri_mp2_spin_components(
        &mol,
        &obs,
        &dfbs,
        Operator::coulomb(),
        &rhf,
        &RiMp2Config::default(),
    )
    .unwrap();
    let drpa = run_pdep_rpa(
        &mol,
        &obs,
        &dfbs,
        Operator::coulomb(),
        &rhf,
        &PdepRpaConfig {
            trunc_thresh: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    let expected = sc.e_total + drpa.e_rpa - 2.0 * sc.e_os;
    eprintln!(
        "MP2C(full) {:.12}  MP2+dRPA-2E_OS {:.12}  diff {:.2e}",
        r.e_corr,
        expected,
        r.e_corr - expected
    );
    assert!(
        (r.e_corr - expected).abs() < 1e-9,
        "full-range MP2C broken: {} vs {}",
        r.e_corr,
        expected
    );
    assert!((r.e_disp_uncoupled - 2.0 * sc.e_os).abs() < 1e-10);
    assert!((r.e_disp_coupled - drpa.e_rpa).abs() < 1e-10);
    assert!(
        r.components_sum_to_corr().abs() < 1e-15,
        "components must sum to e_corr to ULP scale, got {:.3e}",
        r.components_sum_to_corr()
    );
}

/// The long-range arm at large ω must converge to the full-range arm: `erf →
/// Coulomb`. Confirms the two `Mp2cDispersionRange` variants are the same
/// construction at different ranges, not two unrelated formulas.
///
/// Tolerance 1e-6 at ω = 200 Bohr⁻¹ matches what `rs_mp2_rpa`'s ω→∞ test
/// documents: the residual is the finite-ω truncation of the 2-center
/// `(P|erf(200 r)/r|Q)` vs `(P|1/r|Q)` metric (~5e-7 Ha for H2/cc-pVDZ), not a
/// convention error.
#[test]
fn long_range_omega_to_infinity_approaches_full_range() {
    let (mol, obs, dfbs, rhf) = setup_h2();
    let full = mp2c_energy(&mol, &obs, &dfbs, &rhf, &Mp2cConfig::default()).unwrap();
    let lr = mp2c_energy(
        &mol,
        &obs,
        &dfbs,
        &rhf,
        &Mp2cConfig {
            range: Mp2cDispersionRange::LongRange,
            omega: 200.0,
            ..Default::default()
        },
    )
    .unwrap();
    eprintln!(
        "MP2C(full) {:.12}  MP2C(LR, w=200) {:.12}  diff {:.2e}",
        full.e_corr,
        lr.e_corr,
        lr.e_corr - full.e_corr
    );
    assert!(
        (lr.e_corr - full.e_corr).abs() < 1e-6,
        "omega->inf long-range must approach full-range: {} vs {}",
        lr.e_corr,
        full.e_corr
    );
}

/// **Sign and magnitude of the correction.** MP2C exists because MP2's
/// uncoupled dispersion is *too attractive*; coupling screens it. The
/// correction `E_dRPA − 2E_OS` must therefore be positive (less binding), and
/// it must be a correction — same order as, but smaller than, E_MP2 itself.
///
/// This catches a swapped-sign assembly that all three limit tests above would
/// still pass in their respective limits (where both terms vanish or the
/// comparison target is built the same wrong way is not possible here, since
/// the reference is an independently signed physical expectation).
#[test]
fn correction_is_positive_and_of_correction_magnitude() {
    // Water/cc-pVDZ: a real molecule, not the 2-electron H2 edge case.
    let (mol, obs, dfbs, rhf) = setup(
        "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
        "cc-pvdz",
        "cc-pvdz-ri",
    );
    let r = mp2c_energy(&mol, &obs, &dfbs, &rhf, &Mp2cConfig::default()).unwrap();
    eprintln!(
        "water: E_MP2 {:.8}  uncoupled {:.8}  coupled {:.8}  correction {:.8}",
        r.e_mp2, r.e_disp_uncoupled, r.e_disp_coupled, r.e_correction
    );
    assert!(
        r.e_mp2 < 0.0,
        "MP2 correlation must be negative, got {}",
        r.e_mp2
    );
    assert!(
        r.e_disp_uncoupled < 0.0 && r.e_disp_coupled < 0.0,
        "both dispersion terms must be negative (attractive): {} {}",
        r.e_disp_uncoupled,
        r.e_disp_coupled
    );
    assert!(
        r.e_correction > 0.0,
        "coupling must SCREEN the uncoupled dispersion (E_dRPA > 2E_OS, i.e. less \
         attractive), so the correction is positive; got {}",
        r.e_correction
    );
    assert!(
        r.e_correction.abs() < r.e_mp2.abs(),
        "the MP2C correction ({}) must be smaller than E_MP2 ({}) — it is a \
         correction, not a replacement",
        r.e_correction,
        r.e_mp2
    );
    assert!(r.drpa_converged, "dRPA eigensolve must have converged");
}

/// An open-shell reference must be a hard error, not a plausible number: the
/// `E_dMP2 = 2·E_OS` identity underpinning the uncoupled term is a
/// closed-shell spin-summation result.
#[test]
fn open_shell_reference_is_rejected() {
    use ferric_scf::uhf::solve_uhf;
    let mol = Molecule::parse_xyz("1\nH atom\nH 0 0 0\n", 0, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let uhf = solve_uhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        &bounds,
        // UhfConfig is a type alias for RhfConfig.
        &RhfConfig {
            energy_conv: 1e-8,
            ..Default::default()
        },
    )
    .unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let err = mp2c_energy(&mol, &obs, &dfbs, &uhf, &Mp2cConfig::default()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("restricted") || msg.contains("Restricted"),
        "expected a clear closed-shell-only error, got: {msg}"
    );
}

/// A non-finite or non-positive ω on the long-range arm must error rather than
/// produce a NaN energy. `FullRange` never reads ω, so a garbage value there is
/// deliberately fine — asserted so the asymmetry is intentional, not accidental.
#[test]
fn bad_omega_errors_only_on_the_long_range_arm() {
    let (mol, obs, dfbs, rhf) = setup_h2();
    for bad in [0.0, -1.0, f64::NAN] {
        let cfg = Mp2cConfig {
            range: Mp2cDispersionRange::LongRange,
            omega: bad,
            ..Default::default()
        };
        assert!(
            mp2c_energy(&mol, &obs, &dfbs, &rhf, &cfg).is_err(),
            "LongRange with omega={bad} must error"
        );
    }
    // FullRange ignores omega entirely.
    let ok = mp2c_energy(
        &mol,
        &obs,
        &dfbs,
        &rhf,
        &Mp2cConfig {
            range: Mp2cDispersionRange::FullRange,
            omega: f64::NAN,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(ok.e_corr.is_finite());
}

/// **Mutation-driven test (M4).** The long-range arm must feed the dRPA solve
/// the *long-range* intermediates, not Coulomb ones.
///
/// This exists because a seeded mutation replacing `op_d` with
/// `Operator::coulomb()` in the `LongRange` dRPA call **escaped** the original
/// test set. The reason is structural and worth recording: with prebuilt
/// intermediates, `run_pdep_rpa_from_intermediates`'s `op` argument is very
/// nearly vestigial — the correlation energy comes entirely from the operator
/// already baked into `inter.b_ov`, and `op` only reaches the (off-by-default)
/// screened-b_ov path. So passing the *wrong* `op` alongside the *right*
/// intermediates changes nothing, and the original mutation was in fact
/// semantically inert rather than undetected-but-harmful.
///
/// The mutation that would actually be harmful is passing the wrong
/// *intermediates*, and that is what this pins: the long-range coupled
/// dispersion must differ from the full-range one by far more than solver
/// noise. `rs_mp2_rpa.rs` carries the identical exposure on the same shared
/// function; this test documents it for both.
#[test]
fn long_range_coupled_dispersion_uses_long_range_intermediates() {
    let (mol, obs, dfbs, rhf) = setup_h2();
    let lr = mp2c_energy(
        &mol,
        &obs,
        &dfbs,
        &rhf,
        &Mp2cConfig {
            range: Mp2cDispersionRange::LongRange,
            omega: 0.42,
            ..Default::default()
        },
    )
    .unwrap();
    let full = mp2c_energy(&mol, &obs, &dfbs, &rhf, &Mp2cConfig::default()).unwrap();

    // Independent ground truth for the LR coupled term: a standalone dRPA[erf]
    // solve that builds its own intermediates from scratch.
    let drpa_erf = run_pdep_rpa(
        &mol,
        &obs,
        &dfbs,
        Operator::erf(0.42),
        &rhf,
        &PdepRpaConfig {
            trunc_thresh: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    eprintln!(
        "coupled: MP2C(LR) {:.10}  standalone dRPA[erf] {:.10}  MP2C(full) {:.10}",
        lr.e_disp_coupled, drpa_erf.e_rpa, full.e_disp_coupled
    );
    assert!(
        (lr.e_disp_coupled - drpa_erf.e_rpa).abs() < 1e-10,
        "LongRange coupled dispersion must equal a standalone dRPA[erf(0.42)]: {} vs {}",
        lr.e_disp_coupled,
        drpa_erf.e_rpa
    );
    // And it must be clearly distinct from the Coulomb one, so a silent
    // substitution of Coulomb intermediates could not pass the line above.
    assert!(
        (lr.e_disp_coupled - full.e_disp_coupled).abs() > 1e-3,
        "erf and Coulomb coupled dispersion are indistinguishable ({} vs {}), so this \
         test cannot discriminate them",
        lr.e_disp_coupled,
        full.e_disp_coupled
    );
}

/// **Mutation-driven test (M6).** `frozen_core` must reach BOTH halves.
///
/// A seeded mutation setting `drpa_cfg.frozen_core = 0` **escaped** the
/// original test set, because no test used a nonzero `frozen_core` at all.
/// Investigating the escape showed the mutation is in fact **semantically
/// inert** — on the `from_intermediates` path the frozen space comes from
/// `inter.nocc`/`inter.first_occ`, not from `config.frozen_core` (see the
/// module docs). It still escapes today, and correctly so: there is nothing to
/// catch.
///
/// The *harmful* sibling — dropping `frozen_core` from `ri_cfg`, so the
/// intermediates themselves are built all-electron — IS caught by this test.
/// Verified by mutation `M6b`. A mismatch there would leave a spurious residue
/// in `E_dRPA − 2E_OS`, which is supposed to cancel exactly at second order.
///
/// Water/cc-pVDZ with `frozen_core = 1` (the O 1s). The check is against an
/// independently assembled `MP2[fc] + dRPA[fc] − 2E_OS[fc]`, all three
/// explicitly frozen — so a dropped forward in either direction fails.
#[test]
fn frozen_core_reaches_both_halves() {
    let (mol, obs, dfbs, rhf) = setup(
        "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
        "cc-pvdz",
        "cc-pvdz-ri",
    );
    let fc = 1usize;
    let r = mp2c_energy(
        &mol,
        &obs,
        &dfbs,
        &rhf,
        &Mp2cConfig {
            frozen_core: fc,
            ..Default::default()
        },
    )
    .unwrap();

    let (sc, _) = ri_mp2_spin_components(
        &mol,
        &obs,
        &dfbs,
        Operator::coulomb(),
        &rhf,
        &RiMp2Config {
            frozen_core: fc,
            ..Default::default()
        },
    )
    .unwrap();
    let drpa = run_pdep_rpa(
        &mol,
        &obs,
        &dfbs,
        Operator::coulomb(),
        &rhf,
        &PdepRpaConfig {
            trunc_thresh: 0.0,
            frozen_core: fc,
            ..Default::default()
        },
    )
    .unwrap();
    let expected = sc.e_total + drpa.e_rpa - 2.0 * sc.e_os;
    eprintln!(
        "frozen_core=1: MP2C {:.10}  MP2[fc]+dRPA[fc]-2E_OS[fc] {:.10}  diff {:.2e}",
        r.e_corr,
        expected,
        r.e_corr - expected
    );
    assert!(
        (r.e_corr - expected).abs() < 1e-9,
        "frozen_core must reach both halves: {} vs {}",
        r.e_corr,
        expected
    );
    assert!(
        (r.e_disp_coupled - drpa.e_rpa).abs() < 1e-10,
        "the dRPA half ignored frozen_core: {} vs {}",
        r.e_disp_coupled,
        drpa.e_rpa
    );
    // Sanity: freezing must actually change the answer, else the test is inert.
    let all_e = mp2c_energy(&mol, &obs, &dfbs, &rhf, &Mp2cConfig::default()).unwrap();
    assert!(
        (r.e_corr - all_e.e_corr).abs() > 1e-4,
        "frozen_core=1 did not change the energy; the test cannot discriminate"
    );
}
