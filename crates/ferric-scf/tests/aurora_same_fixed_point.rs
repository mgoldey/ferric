//! AURORA must reach the SAME stationary point as DIIS.
//!
//! An accelerator changes the PATH to the fixed point, never the fixed point
//! itself. The target Hamiltonian supplies every energy, gradient and
//! convergence decision; the auxiliary STO-3G model touches only curvature. So
//! if AURORA and DIIS disagree on the converged energy by more than the SCF
//! convergence tolerance, that is a bug in the accelerator — not a new answer.
//!
//! The paper reports a largest absolute final-energy difference of
//! 1.60e-10 Eh over its 16 direct CPU RHF pairs; these tests assert agreement at
//! a comparable bar.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::aurora::AuroraConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Run one system both ways and return `(E_diis, iters_diis, E_aurora, iters_aurora)`.
fn pair(xyz: &str, basis_name: &str, xc: Option<&str>) -> (f64, usize, bool, f64, usize, bool) {
    let mol = Molecule::load_xyz(xyz).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        xc: xc.map(|s| s.to_string()),
        df_j_aux: xc.map(|_| "def2-universal-jkfit".to_string()),
        ..Default::default()
    };
    let cfg_aurora = RhfConfig {
        aurora: AuroraConfig {
            enabled: true,
            ..Default::default()
        },
        ..cfg_diis.clone()
    };

    let a = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();
    let b = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_aurora).unwrap();
    (
        a.energy,
        a.iterations,
        a.converged,
        b.energy,
        b.iterations,
        b.converged,
    )
}

fn check(label: &str, xyz: &str, basis_name: &str, xc: Option<&str>, tol: f64) {
    let (e_d, it_d, cv_d, e_a, it_a, cv_a) = pair(xyz, basis_name, xc);
    let de = (e_a - e_d).abs();
    eprintln!(
        "{label:<24} DIIS: E = {e_d:.12} ({it_d} it, conv={cv_d})  \
         AURORA: E = {e_a:.12} ({it_a} it, conv={cv_a})  |ΔE| = {de:.3e}"
    );
    assert!(cv_d, "{label}: DIIS baseline must converge");
    assert!(cv_a, "{label}: AURORA must converge");
    assert!(
        de < tol,
        "{label}: AURORA and DIIS must reach the same stationary point, \
         |ΔE| = {de:.3e} > {tol:.3e}. An accelerator changes the path, not the answer."
    );
}

#[test]
fn water_sto3g_rhf_same_fixed_point() {
    check(
        "water/STO-3G RHF",
        "../../testdata/molecules/water.xyz",
        "sto-3g",
        None,
        1e-9,
    );
}

#[test]
fn water_ccpvdz_rhf_same_fixed_point() {
    check(
        "water/cc-pVDZ RHF",
        "../../testdata/molecules/water.xyz",
        "cc-pvdz",
        None,
        1e-9,
    );
}

#[test]
fn methane_ccpvdz_rhf_same_fixed_point() {
    check(
        "methane/cc-pVDZ RHF",
        "../../testdata/molecules/methane.xyz",
        "cc-pvdz",
        None,
        1e-9,
    );
}

#[test]
fn benzene_sto3g_rhf_same_fixed_point() {
    check(
        "benzene/STO-3G RHF",
        "../../testdata/molecules/benzene.xyz",
        "sto-3g",
        None,
        1e-9,
    );
}

#[test]
fn benzene_ccpvdz_rhf_same_fixed_point() {
    check(
        "benzene/cc-pVDZ RHF",
        "../../testdata/molecules/benzene.xyz",
        "cc-pvdz",
        None,
        1e-9,
    );
}

/// A pure functional must be DECLINED, not silently accelerated.
///
/// PBE has no exact exchange, so the auxiliary model has no exchange curvature
/// and (lacking the paper's undefined `D_k^xc`) understates the true curvature.
/// The gate keeps such references on DIIS, which must therefore reproduce the
/// DIIS result exactly — same energy bits, same iteration count.
#[test]
fn pure_functional_is_declined_and_falls_back_to_diis_exactly() {
    let (e_d, it_d, cv_d, e_a, it_a, cv_a) =
        pair("../../testdata/molecules/water.xyz", "cc-pvdz", Some("PBE"));
    eprintln!(
        "water/cc-pVDZ PBE (declined)  DIIS: E = {e_d:.12} ({it_d} it)  \
         AURORA-requested: E = {e_a:.12} ({it_a} it)"
    );
    assert!(cv_d && cv_a);
    assert_eq!(
        e_a.to_bits(),
        e_d.to_bits(),
        "a declined reference must fall back to DIIS bit-for-bit"
    );
    assert_eq!(
        it_a, it_d,
        "a declined reference must take exactly the DIIS iteration count"
    );
}

/// The gate must be REACHABLE in both directions — otherwise the test above is
/// asserting arithmetic rather than measuring a decision.
///
/// Opting in with `allow_low_exchange_ks` must actually engage the accelerator
/// and produce a DIFFERENT path. This is the reachability check the repo's
/// protocol requires: a gate whose "on" branch cannot be reached would make the
/// "off" assertion vacuous.
#[test]
fn the_low_exchange_gate_is_reachable_in_both_directions() {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let base = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 300,
        xc: Some("PBE".to_string()),
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        ..Default::default()
    };
    let declined = solve_rhf(
        &ctx,
        &mol,
        &prep,
        op,
        &bounds,
        &RhfConfig {
            aurora: AuroraConfig {
                enabled: true,
                ..Default::default()
            },
            ..base.clone()
        },
    )
    .unwrap();
    // Opted in, with the shorter step the measurement showed this regime needs.
    let engaged = solve_rhf(
        &ctx,
        &mol,
        &prep,
        op,
        &bounds,
        &RhfConfig {
            aurora: AuroraConfig {
                enabled: true,
                allow_low_exchange_ks: true,
                trust_radius: 0.10,
                first_trust_radius: 0.10,
                ..Default::default()
            },
            ..base.clone()
        },
    )
    .unwrap();

    eprintln!(
        "PBE declined: {} it (E = {:.10});  opted in: {} it (E = {:.10})",
        declined.iterations, declined.energy, engaged.iterations, engaged.energy
    );
    assert!(engaged.converged, "the opted-in run must still converge");
    assert_ne!(
        engaged.iterations, declined.iterations,
        "opting in must actually change the path — otherwise the gate is inert"
    );
    // Same fixed point, different path: the whole point of an accelerator.
    let de = (engaged.energy - declined.energy).abs();
    assert!(
        de < 1e-7,
        "even in the un-validated regime the fixed point must not move: {de:.3e}"
    );
}

/// B3LYP (a_x = 0.20) clears the validated-exchange gate and must reach the same
/// stationary point.
///
/// It is deliberately NOT asserted to be faster: measured on water/cc-pVDZ it
/// reaches the right answer but takes more iterations than DIIS at the paper's
/// default trust radius (166 vs 57; 59 at radius 0.10). That measurement is
/// reported rather than hidden — see the accompanying report.
#[test]
fn water_ccpvdz_b3lyp_same_fixed_point() {
    check(
        "water/cc-pVDZ B3LYP",
        "../../testdata/molecules/water.xyz",
        "cc-pvdz",
        Some("B3LYP"),
        1e-8,
    );
}

/// The auxiliary basis is a CURVATURE model: changing it must not move the
/// answer, only the path.
///
/// This is the sharpest available test that nothing from the auxiliary model
/// leaks into the result. If STO-3G and 6-31G curvature gave different converged
/// energies, the auxiliary model would be contaminating the target problem —
/// precisely the failure mode the method's central claim rules out.
#[test]
fn changing_the_auxiliary_curvature_basis_does_not_move_the_answer() {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let base = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };

    let mut energies = Vec::new();
    for aux in ["sto-3g", "6-31g"] {
        let cfg = RhfConfig {
            aurora: AuroraConfig {
                enabled: true,
                auxbasis: aux.to_string(),
                ..Default::default()
            },
            ..base.clone()
        };
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        eprintln!(
            "aux = {aux:<8} E = {:.12}  iters = {}  converged = {}",
            r.energy, r.iterations, r.converged
        );
        assert!(r.converged, "AURORA with aux={aux} must converge");
        energies.push(r.energy);
    }
    let d = (energies[0] - energies[1]).abs();
    eprintln!("curvature-basis sensitivity of the ANSWER: |ΔE| = {d:.3e}");
    assert!(
        d < 1e-9,
        "the auxiliary basis is a curvature model only; it must not change the \
         converged energy: |ΔE| = {d:.3e}"
    );
}

/// AURORA must reach the same answer regardless of thread count.
///
/// The auxiliary build and the response contraction both run under rayon; a
/// reduction-order bug there would show up as a thread-count-dependent path and,
/// at a loose convergence threshold, potentially a different answer.
#[test]
fn aurora_answer_is_independent_of_rayon_thread_count() {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();

    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        aurora: AuroraConfig {
            enabled: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut seen: Vec<(usize, f64, usize)> = Vec::new();
    for nthreads in [1usize, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads)
            .build()
            .unwrap();
        let (e, it) = pool.install(|| {
            let ctx = ParallelContext::default();
            let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
            assert!(r.converged, "AURORA must converge at {nthreads} threads");
            (r.energy, r.iterations)
        });
        eprintln!("threads = {nthreads}: E = {e:.12}, iters = {it}");
        seen.push((nthreads, e, it));
    }
    let e0 = seen[0].1;
    for (n, e, _) in &seen {
        let d = (e - e0).abs();
        assert!(
            d < 1e-9,
            "AURORA energy must not depend on thread count: {n} threads gave \
             ΔE = {d:.3e} vs 1 thread"
        );
    }
}
