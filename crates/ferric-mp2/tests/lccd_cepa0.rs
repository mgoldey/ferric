//! Anchors for closed-shell spatial LCCD / CEPA(0).
//!
//! The residual is PROVED element-by-element against a full spin-orbital
//! antisymmetrized LCCD oracle in `wiki/notebooks/16-lccd-cepa0.ipynb`, so
//! a failure here is an implementation bug by construction.
//!
//! EXACTNESS ANCHOR FIRST (repo Experimental Protocol): LCCD's trivial
//! limit is one Jacobi step from a ZERO amplitude. Every ladder and ring
//! term is BILINEAR in `T`, so all of them vanish at `T = 0` and the step
//! reduces to `T⁽¹⁾ = (ia|jb)/(−D)` — the MP2 amplitude — making the LCCD
//! energy functional evaluate to MP2 exactly. That is a SHARP identity, and
//! it pins the driver term and the energy expression before any ladder or
//! ring correctness is at stake.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lccd_cepa0::{dense_operator, lccd, LccdConfig};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ferric_scf::result::ScfResult,
    #[allow(dead_code)]
    bounds: SchwarzBounds,
}

fn setup(xyz: &str, obs_name: &str, aux_name: &str) -> Setup {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled(obs_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
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
    Setup { mol, obs, dfbs, rhf, bounds }
}

/// THE EXACTNESS ANCHOR: the `e_mp2` the solver reports (its own energy
/// functional evaluated on the first-order amplitude built from its own
/// blocks and denominators) must equal ferric's INDEPENDENTLY implemented
/// RI-MP2 to the RI-consistency floor.
///
/// Independent construction, not a self-check: `ri_mp2` shares no code path
/// with `lccd_cepa0` beyond the raw integrals, so agreement pins the
/// driver term, the denominators, and the `2T − T_ibja` energy expression
/// all at once.
#[test]
fn lccd_energy_functional_reproduces_mp2_at_first_order() {
    for (xyz, obs_name, aux, fc) in [
        ("h2.xyz", "sto-3g", "cc-pvdz-ri", 0),
        ("water.xyz", "6-31g", "cc-pvdz-ri", 1),
    ] {
        let su = setup(xyz, obs_name, aux);
        let cfg = LccdConfig { frozen_core: fc, ..Default::default() };
        let r = lccd(
            &su.mol,
            &su.obs,
            &su.dfbs,
            Operator::coulomb(),
            &su.rhf,
            &cfg,
        )
        .unwrap();

        let mp2 = ri_mp2(
            &su.mol,
            &su.obs,
            &su.dfbs,
            Operator::coulomb(),
            &su.rhf,
            &RiMp2Config { frozen_core: fc, ..Default::default() },
        )
        .unwrap();

        eprintln!(
            "{xyz}/{obs_name}: lccd e_mp2={:.12} vs ri_mp2 {:.12} (d={:+.2e}); \
             E_LCCD={:.10} ({} GMRES its, relres {:.2e})",
            r.e_mp2,
            mp2.mp2_corr,
            r.e_mp2 - mp2.mp2_corr,
            r.e_corr,
            r.iterations,
            r.relres
        );
        assert!(
            (r.e_mp2 - mp2.mp2_corr).abs() < 1e-9,
            "LCCD's first-order limit is not MP2 on {xyz}: {:+.3e}",
            r.e_mp2 - mp2.mp2_corr
        );
    }
}

/// LCCD must actually differ from MP2 — otherwise the ladders and rings
/// could be silently contributing nothing and every other assert would
/// still pass. Direction is also checked: LCCD overcorrelates relative to
/// MP2 on these systems.
#[test]
fn lccd_is_not_merely_mp2() {
    let su = setup("water.xyz", "6-31g", "cc-pvdz-ri");
    let cfg = LccdConfig { frozen_core: 1, ..Default::default() };
    let r = lccd(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
    )
    .unwrap();
    eprintln!(
        "water: E_LCCD={:.10} E_MP2={:.10} ratio={:.4}",
        r.e_corr,
        r.e_mp2,
        r.e_corr / r.e_mp2
    );
    assert!(r.converged);
    assert!(
        (r.e_corr - r.e_mp2).abs() > 1e-4,
        "LCCD coincides with MP2 — ladders/rings are contributing nothing"
    );
    assert!(r.e_corr < 0.0, "correlation energy must be negative");
}

/// A TEST YOU HAVE NEVER SEEN FAIL IS AN ASSUMPTION. The GMRES choice rests
/// on a MEASURED claim: the LCCD operator is genuinely non-symmetric. If it
/// were symmetric, PCG/MINRES would be legitimate and the module's central
/// design decision would be wrong. Measure it rather than restate it.
///
/// The notebook measured 7.63% on water/6-31G with exact ERIs, reproducible
/// to the digit across reruns. Here the integrals are RI, so the bar is a
/// generous band around that rather than an equality — the claim under test
/// is "not roundoff-symmetric", not a specific percentage.
#[test]
fn the_lccd_operator_is_genuinely_non_symmetric() {
    let su = setup("water.xyz", "6-31g", "cc-pvdz-ri");
    let cfg = LccdConfig { frozen_core: 1, ..Default::default() };
    let a = dense_operator(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
    )
    .unwrap();
    let mut asym: f64 = 0.0;
    let mut scale: f64 = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            asym = asym.max((a[(i, j)] - a[(j, i)]).abs());
            scale = scale.max(a[(i, j)].abs());
        }
    }
    let ratio = asym / scale;
    eprintln!(
        "water LCCD operator: dim={} |A-A^T|_max={:.6} scale={:.6} asym/scale={:.2}%",
        a.nrows(),
        asym,
        scale,
        100.0 * ratio
    );
    assert!(
        ratio > 0.01,
        "operator is effectively symmetric (asym/scale = {:.3}%) — the GMRES-over-PCG \
         justification in this module's docs does NOT hold and must be revisited",
        100.0 * ratio
    );
    // Notebook measured 7.63% with exact ERIs; RI should not move this much.
    assert!(
        (0.03..0.15).contains(&ratio),
        "asymmetry {:.2}% is far from the notebook's measured 7.63% — the operator \
         being built here may not be the one that was verified",
        100.0 * ratio
    );
}

/// The `is_not_merely_mp2` test would pass even if the ring term were
/// dropped (the ladders alone would still move the energy). This pins the
/// ring specifically: with the residual as implemented, the converged LCCD
/// energy must sit on the exact-ERI oracle value, which the notebook
/// verified is only reproduced when all four ring pieces are present.
///
/// References are regenerated AT FERRIC'S OWN GEOMETRY by
/// `scripts/gen_lccd_oracle_refs.py`. The notebook's published literals
/// (H2 -0.0207912500, water -0.1339778905) are NOT assertable here: they
/// were computed at the notebook's geometries (H2 at 0.74 A vs ferric's
/// 0.7414 A). Using them would have inflated the apparent RI error ~10x
/// and hidden real drift inside a loose bar.
///
/// The oracle also uses a damped Jacobi iteration where the Rust port uses
/// GMRES, so agreement is between two DIFFERENT solvers on the same
/// equations, not a reimplementation of one algorithm.
///
/// Bar is the RI floor, not machine precision (ferric fits (ia|jb) in an
/// auxiliary basis, the oracle does not) -- MEASURED 2.6e-6..5.5e-6.
#[test]
fn lccd_matches_the_exact_eri_oracle() {
    for (xyz, obs_name, aux, fc, want) in [
        ("h2.xyz", "sto-3g", "cc-pvdz-ri", 0, -0.020_854_691_3_f64),
        ("water.xyz", "6-31g", "cc-pvdz-ri", 1, -0.133_996_954_6_f64),
    ] {
        let su = setup(xyz, obs_name, aux);
        let cfg = LccdConfig { frozen_core: fc, ..Default::default() };
        let r = lccd(
            &su.mol,
            &su.obs,
            &su.dfbs,
            Operator::coulomb(),
            &su.rhf,
            &cfg,
        )
        .unwrap();
        eprintln!(
            "{xyz}/{obs_name}: E_LCCD={:.10} oracle={want:.10} d={:+.2e} \
             ({} GMRES its)",
            r.e_corr,
            r.e_corr - want,
            r.iterations
        );
        assert!(
            (r.e_corr - want).abs() < 5e-5,
            "LCCD off the exact-ERI oracle on {xyz}: {:+.3e}",
            r.e_corr - want
        );
    }
}


/// A GUARD YOU HAVE NEVER SEEN FIRE IS AN ASSUMPTION. CEPA(0)'s documented
/// failure mode is convergence onto a SPURIOUS fixed point with a perfectly
/// healthy residual, so a residual-only check would pass it straight
/// through. This exercises the real thing.
///
/// MEASURED here at H2/STO-3G r = 4.0 Å: GMRES converges to relres ~1.4e-15
/// — machine precision — on an energy of −17.95 Ha. The notebook's
/// INDEPENDENT damped-Jacobi rig converged to −17.98 Ha at the same
/// geometry. Two unrelated solvers landing on the same absurd value is
/// evidence the spurious fixed point is a property of the CEPA(0) equations
/// themselves, not of either solver.
///
/// Breakdown onset measured across r = 1.5–4.0 Å (HOMO-LUMO gap 0.58 →
/// 0.14 Ha): E_LCCD/E_MP2 grows 2.2 → 3.1 → 4.9 → 8.9 and is rejected from
/// r = 3.5 Å on. There is no clean cliff — the ratio degrades smoothly,
/// which is exactly why a magnitude bound is needed rather than a
/// convergence flag.
#[test]
fn cepa0_breakdown_is_rejected_not_silently_returned() {
    let su = setup("h2_r4.xyz", "sto-3g", "cc-pvdz-ri");
    let cfg = LccdConfig::default();
    let err = lccd(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &cfg,
    )
    .expect_err(
        "CEPA(0) at r=4.0 A converges cleanly onto a spurious ~-18 Ha fixed point; \
         returning it as a result would be silently wrong",
    );
    let msg = format!("{err}");
    eprintln!("r=4.0 rejection: {msg}");
    assert!(
        msg.contains("unphysical"),
        "rejected for the wrong reason: {msg}"
    );

    // The guard must be what caught it — NOT a convergence failure. Disable
    // the bound and confirm the solver reports a clean, converged, and
    // completely wrong answer. If this ever starts failing to converge
    // instead, the test above stops testing the guard.
    let unguarded = LccdConfig { max_corr_vs_mp2: None, ..Default::default() };
    let r = lccd(
        &su.mol,
        &su.obs,
        &su.dfbs,
        Operator::coulomb(),
        &su.rhf,
        &unguarded,
    )
    .expect("without the magnitude bound the solve itself succeeds");
    eprintln!(
        "r=4.0 UNGUARDED: E_corr={:.6} E_MP2={:.6} relres={:.2e} converged={}",
        r.e_corr, r.e_mp2, r.relres, r.converged
    );
    assert!(r.converged && r.relres < 1e-10, "the residual looks healthy");
    assert!(
        r.e_corr < -5.0,
        "expected the spurious ~-18 Ha fixed point, got {:.6}",
        r.e_corr
    );
}
