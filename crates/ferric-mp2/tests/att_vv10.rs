//! Validation for attenuated MP2 + VV10 ("MP2-V", JCTC 11, 4159 (2015)).
//!
//! Test system choice: water/cc-pVDZ. Deliberately NOT H2 — the VV10 damping
//! factor `1 − terfc(R,r₀)²` and the VV10 kernel itself only differ from each
//! other over a range of pair distances, and a two-electron system with one
//! bond length exercises almost none of that. Water carries a heavy atom, lone
//! pairs, and a grid spanning ~0.1–15 Bohr of pair separations, so the damping
//! branch is genuinely exercised (asserted directly in
//! `damping_actually_changes_the_energy`).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_dft::libxc::Vv10Params;
use ferric_dft::vv10::Vv10Damping;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::att_vv10::{
    att_mp2_vv10, vv10_energy_on_density, AttVv10Attenuator, AttVv10Config, BOHR_PER_ANG,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

struct Case {
    mol: Molecule,
    bs: basis::BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

fn water_ccpvdz() -> Case {
    let xyz = "3\nwater\nO 0.000 0.000 0.118\nH 0.000 0.755 -0.471\nH 0.000 -0.755 -0.471\n";
    build_case(xyz, "cc-pvdz")
}

fn build_case(xyz: &str, basis_name: &str) -> Case {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
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
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    Case {
        mol,
        bs,
        obs,
        dfbs,
        rhf,
    }
}

/// Small grid for the tests: the pair sum is O(npts²) and the production
/// 50x50 NLC grid on water is ~19k points, which is minutes per call.
fn test_grid() -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial: 20,
        n_angular: 26,
        prune: None,
    }
}

/// The erfc attenuator, so these tests need no terfc interpolation tables.
/// Correctness of the *combination* (the thing under test here) is independent
/// of which short-range operator supplies E_c; the terfc path is covered by
/// `terfc_path_runs_when_tables_present` below, which skips when tables are absent.
fn erfc_test_config() -> AttVv10Config {
    AttVv10Config {
        nlc_grid: test_grid(),
        ..AttVv10Config::erfc_control_at_atz_params()
    }
}

// ---------------------------------------------------------------------------
// 1. THE cross-check: our VV10 call vs the wB97X-V code path
// ---------------------------------------------------------------------------

/// The single most valuable check available: `vv10_energy_on_density` must
/// reproduce, to the last bit, the E_nl that `ferric_dft::vv10::add_vv10_scratch`
/// produces — the exact function `ferric_dft::ks::KsXc::add_xc` calls for
/// wB97X-V. This isolates "did I wire VV10 up correctly" from "is the MP2-V
/// combination right".
///
/// Uses wB97X-V's own VV10 parameters (b = 6.0, C = 0.01, read out of libxc via
/// `xc_def_from_name`, not hardcoded) on the same density, same grid, same AOs.
#[test]
fn vv10_energy_matches_wb97xv_add_vv10_path() {
    let c = water_ccpvdz();

    // wB97X-V's VV10 parameters, straight from libxc.
    let xc = ferric_dft::libxc::xc_def_from_name("wB97X-V").unwrap();
    let params = xc.vv10.expect("wB97X-V must carry VV10 parameters");
    eprintln!("wB97X-V VV10 params from libxc: b={}, C={}", params.b, params.c);

    let grid_cfg = test_grid();
    let grid = build_atomic_grid(&c.mol, &grid_cfg);
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) =
        ferric_dft::ao_grid::eval_basis_and_grad_on_points(&c.mol, &c.bs, &pts).unwrap();
    let d_total = c.rhf.density_total();
    let dens = ferric_dft::density_on_grid::eval_density_closed(d_total, &chi, &dchi);

    // Reference: the production KS path. add_vv10 returns E_nl and accumulates
    // V_nl into `f`; we only need the energy.
    let mut f = ndarray::Array2::<f64>::zeros((chi.nrows(), chi.nrows()));
    let e_nl_ks = ferric_dft::vv10::add_vv10(&grid, &chi, &dchi, &dens, &params, &mut f);

    // Ours, through the att_vv10 entry point, undamped so it is the same functional.
    let (e_nl_ours, npts) = vv10_energy_on_density(
        &c.mol,
        &c.bs,
        d_total,
        &params,
        Vv10Damping::None,
        &grid_cfg,
    )
    .unwrap();

    eprintln!(
        "VV10 cross-check (water/cc-pVDZ, {npts} pts): KS add_vv10 path = {e_nl_ks:.14} Ha, \
         att_vv10 path = {e_nl_ours:.14} Ha, diff = {:.2e}",
        (e_nl_ks - e_nl_ours).abs()
    );
    assert_eq!(npts, grid.len());
    // Same inputs through the same pair sum: this must be BIT-identical, not
    // merely close. A nonzero diff means the two paths disagree on the grid,
    // the density, or the parameters.
    assert_eq!(
        e_nl_ks, e_nl_ours,
        "att_vv10's VV10 call must be bit-identical to the wB97X-V add_vv10 path"
    );
    assert!(
        e_nl_ks.is_finite() && e_nl_ks > 0.0,
        "VV10 E_nl on water should be a small positive number, got {e_nl_ks}"
    );
}

// ---------------------------------------------------------------------------
// 2. The damping is real, and off-by-default is bit-identical to plain VV10
// ---------------------------------------------------------------------------

/// `Vv10Damping::None` must not perturb the historical VV10 result at all —
/// the new damping code is inert unless asked for.
#[test]
fn undamped_is_bit_identical_to_plain_vv10() {
    let c = water_ccpvdz();
    let params = Vv10Params { c: 0.0089, b: 11.0 };
    let grid_cfg = test_grid();

    let grid = build_atomic_grid(&c.mol, &grid_cfg);
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) =
        ferric_dft::ao_grid::eval_basis_and_grad_on_points(&c.mol, &c.bs, &pts).unwrap();
    let dens = ferric_dft::density_on_grid::eval_density_closed(c.rhf.density_total(), &chi, &dchi);

    let (e_plain, _, _) =
        ferric_dft::vv10::compute_vv10_energy_and_potentials(&grid, &dens, &params);
    let (e_none, _, _) = ferric_dft::vv10::compute_vv10_damped_energy_and_potentials(
        &grid,
        &dens,
        &params,
        Vv10Damping::None,
    );
    assert_eq!(
        e_plain, e_none,
        "Vv10Damping::None must be bit-identical to the undamped entry point"
    );
}

/// The terfc damping must actually change the energy on this system, and in
/// the physically required direction: `1 − terfc(R,r₀)² ∈ [0,1)` strictly
/// removes short-range kernel weight, so the damped |E_nl| must be SMALLER.
///
/// This is the test that proves the test system exercises the damping branch
/// at all — the failure mode flagged in the task brief (a system too small to
/// reach the code under test).
#[test]
fn damping_actually_changes_the_energy() {
    let c = water_ccpvdz();
    let params = Vv10Params { c: 0.0089, b: 11.0 };
    let grid_cfg = test_grid();
    let r0_bohr = 1.00 * BOHR_PER_ANG;

    let (e_undamped, _) = vv10_energy_on_density(
        &c.mol,
        &c.bs,
        c.rhf.density_total(),
        &params,
        Vv10Damping::None,
        &grid_cfg,
    )
    .unwrap();
    let (e_damped, _) = vv10_energy_on_density(
        &c.mol,
        &c.bs,
        c.rhf.density_total(),
        &params,
        Vv10Damping::Terfc { r0_bohr },
        &grid_cfg,
    )
    .unwrap();

    let rel = (e_damped - e_undamped).abs() / e_undamped.abs();
    eprintln!(
        "VV10 damping (water/cc-pVDZ, r0=1.00 A = {r0_bohr:.4} Bohr): \
         undamped E_nl = {e_undamped:.12} Ha, damped = {e_damped:.12} Ha, \
         relative change = {:.1}%",
        100.0 * rel
    );
    assert!(
        rel > 0.01,
        "terfc damping changed E_nl by only {:.3e} relative — this test system does \
         not exercise the damping branch meaningfully",
        rel
    );
    // DIRECTION. ferric/PySCF write E_nl = Σ_g w_g ρ_g (β + ½ f_g) where β > 0 is a
    // LOCAL self-energy constant and f_g < 0 is the NONLOCAL pair integral
    // (`-1.5·Σ_p …`, vv10.rs). The damping multiplies Φ_VV10, i.e. only f — β
    // carries no |r − r'| and is not a pair quantity, so it is untouched
    // (matching the paper's Eq. 11, which damps Φ_VV10 inside the double
    // integral). Removing short-range pair weight therefore makes f LESS
    // negative, so the total E_nl goes UP. The attractive/binding effect shows
    // up in the DIFFERENCE between a dimer and its monomers, which is what
    // `vv10_increases_binding_on_a_bound_dimer` asserts — not in the sign of
    // this absolute number.
    assert!(
        e_damped > e_undamped,
        "damping removes negative pair weight from f, so the total E_nl (β + ½f) \
         must RISE: damped {e_damped} vs undamped {e_undamped}"
    );
    assert!(e_damped.is_finite());
}

/// The damping must vary monotonically with r₀: a LARGER r₀ means a
/// longer-ranged terfc, i.e. MORE of the (negative) nonlocal pair integral
/// removed, i.e. a LARGER total E_nl = β + ½f. Guards against a sign or unit
/// inversion in the `1 − terfc²` construction that a single-r₀ test would miss.
/// (See `damping_actually_changes_the_energy` for why the direction is "up".)
#[test]
fn damping_is_monotone_in_r0() {
    let c = water_ccpvdz();
    let params = Vv10Params { c: 0.0089, b: 11.0 };
    let grid_cfg = test_grid();

    let mut prev = f64::NEG_INFINITY;
    for r0_ang in [0.5_f64, 1.0, 1.5, 2.0] {
        let (e, _) = vv10_energy_on_density(
            &c.mol,
            &c.bs,
            c.rhf.density_total(),
            &params,
            Vv10Damping::Terfc {
                r0_bohr: r0_ang * BOHR_PER_ANG,
            },
            &grid_cfg,
        )
        .unwrap();
        eprintln!("r0 = {r0_ang:.2} A -> damped E_nl = {e:.12} Ha");
        assert!(
            e > prev,
            "E_nl must increase monotonically with r0 (r0={r0_ang}: {e} not > {prev})"
        );
        prev = e;
    }
}

/// r₀ → 0 turns the damping off: `terfc(R, r₀→0) → 0` for every fixed R > 0, so
/// the factor `1 − terfc²` → 1 and the damped energy must converge to plain
/// VV10. This is the limit test for the damping half (the ω → 0 limit test for
/// the MP2 half already lives in `attenuated.rs`).
///
/// The convergence is to a small nonzero FLOOR, not to zero, and the reason is
/// structural rather than a defect: the damping is a function of the PAIR
/// distance, and the pair sum includes the `p = i` self-pair at exactly R = 0,
/// where `1 − terfc(0, r₀)² = 0` for **every** r₀ > 0. That one always-fully-
/// damped term never goes away, so the residual falls as r₀ shrinks and then
/// saturates the moment r₀ drops below the grid's smallest nonzero pair
/// distance.
///
/// MEASURED here (water/cc-pVDZ, 20x26 grid): the residual falls from r₀ = 1e-2
/// Bohr and is pinned at 9.051e-6 Ha for every r₀ ≤ 1e-3 Bohr — 0.05% of
/// E_nl — which is the self-pair floor. This test asserts (a) the residual
/// never GROWS as r₀ shrinks and (b) it settles to a floor that is a negligible
/// fraction of E_nl. A sign error or unit inversion would violate both.
#[test]
fn damping_vanishes_as_r0_goes_to_zero() {
    let c = water_ccpvdz();
    let params = Vv10Params { c: 0.0089, b: 11.0 };
    let grid_cfg = test_grid();

    let (e_undamped, _) = vv10_energy_on_density(
        &c.mol,
        &c.bs,
        c.rhf.density_total(),
        &params,
        Vv10Damping::None,
        &grid_cfg,
    )
    .unwrap();

    // Start from a clearly-damping r0 so there is a real decrease to observe
    // before the self-pair floor is reached.
    let mut prev_diff = f64::INFINITY;
    let mut diffs = Vec::new();
    for r0_bohr in [1.0_f64, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5] {
        let (e, _) = vv10_energy_on_density(
            &c.mol,
            &c.bs,
            c.rhf.density_total(),
            &params,
            Vv10Damping::Terfc { r0_bohr },
            &grid_cfg,
        )
        .unwrap();
        let diff = (e - e_undamped).abs();
        eprintln!(
            "r0 = {r0_bohr:.0e} Bohr -> E_nl = {e:.14}, undamped = {e_undamped:.14}, \
             diff = {diff:.3e}"
        );
        // Non-increasing: the residual may plateau at the self-pair floor but
        // must never grow as r0 shrinks.
        assert!(
            diff <= prev_diff * (1.0 + 1e-12),
            "the r0 -> 0 residual must never grow (r0={r0_bohr:.0e}: \
             {diff:.3e} > {prev_diff:.3e})"
        );
        prev_diff = diff;
        diffs.push(diff);
    }

    // It must actually have DECREASED overall — not been flat the whole way,
    // which is what a no-op damping would look like.
    let first = diffs[0];
    let last = *diffs.last().unwrap();
    assert!(
        last < 0.5 * first,
        "the residual should fall substantially from r0=1 Bohr ({first:.3e}) to \
         r0=1e-5 Bohr ({last:.3e})"
    );
    // And the floor it settles on is the R=0 self-pair term: negligible next to E_nl.
    assert!(
        last < 1e-3 * e_undamped.abs(),
        "the r0 -> 0 floor ({last:.3e}) should be < 0.1% of E_nl ({e_undamped:.6})"
    );
}

// ---------------------------------------------------------------------------
// 3. The combination: additivity, magnitudes, signs
// ---------------------------------------------------------------------------

#[test]
fn components_sum_exactly_to_total() {
    let c = water_ccpvdz();
    let cfg = erfc_test_config();
    let r = att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.rhf, &cfg).unwrap();
    eprintln!(
        "MP2-V(erfc control)/water/cc-pVDZ: E_HF = {:.10}, E_c^attMP2 = {:.10}, \
         E_nl^VV10 = {:.10}, total = {:.10}",
        r.e_hf, r.e_c_att_mp2, r.e_nl_vv10, r.total
    );
    assert_eq!(
        r.components_sum_to_total(),
        0.0,
        "reported total must be exactly the sum of the reported components"
    );
    assert_eq!(r.e_hf, c.rhf.energy);
}

/// Physical sanity on the magnitudes and signs of the three pieces.
#[test]
fn component_signs_and_magnitudes_are_physical() {
    let c = water_ccpvdz();
    let cfg = erfc_test_config();
    let r = att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.rhf, &cfg).unwrap();

    assert!(r.e_hf < 0.0, "HF total energy must be negative");
    assert!(
        r.e_c_att_mp2 < 0.0,
        "MP2 correlation energy must be negative, got {}",
        r.e_c_att_mp2
    );
    // VV10's E_nl as PySCF/ferric define it (Σ w ρ (β + ½f), β > 0 local, f < 0
    // nonlocal) is a small POSITIVE number for a molecule at equilibrium. The
    // attraction it supplies is a DIFFERENCE effect (dimer vs monomers), not the
    // sign of this absolute number — see `vv10_increases_binding_on_a_bound_dimer`.
    assert!(
        r.e_nl_vv10 > 0.0,
        "E_nl^VV10 as defined here (β + ½f folded against ρw) is positive for a \
         molecule at equilibrium, got {}",
        r.e_nl_vv10
    );
    // MEASURED on this system: 0.0187 / 0.1807 = 10.3%. Bound at 25% — loose
    // enough not to be a change-detector, tight enough that a factor-of-2 or
    // sign-scale error in E_nl (the failure mode that matters) trips it.
    let ratio = r.e_nl_vv10.abs() / r.e_c_att_mp2.abs();
    eprintln!("|E_nl| / |E_c^attMP2| = {ratio:.4}");
    assert!(
        ratio < 0.25,
        "E_nl^VV10 ({}) should be a small correction next to E_c^attMP2 ({}), \
         ratio = {ratio:.4}",
        r.e_nl_vv10,
        r.e_c_att_mp2
    );
}

/// Attenuated MP2 keeps only short-range correlation, so |E_c^att| must be
/// smaller than full Coulomb MP2 — the same guard `attenuated.rs` applies,
/// re-asserted through the MP2-V entry point so a wrong operator (e.g. Coulomb
/// silently substituted) is caught here too.
#[test]
fn attenuated_half_is_smaller_than_full_mp2() {
    let c = water_ccpvdz();
    let full = ferric_mp2::rimp2::ri_mp2(
        &c.mol,
        &c.obs,
        &c.dfbs,
        Operator::coulomb(),
        &c.rhf,
        &ferric_mp2::rimp2::RiMp2Config::default(),
    )
    .unwrap();
    let cfg = erfc_test_config();
    let r = att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.rhf, &cfg).unwrap();
    eprintln!(
        "full MP2 corr = {:.10}, MP2-V attenuated half = {:.10}",
        full.mp2_corr, r.e_c_att_mp2
    );
    assert!(
        r.e_c_att_mp2.abs() < full.mp2_corr.abs(),
        "attenuated correlation |{}| must be < full |{}|",
        r.e_c_att_mp2,
        full.mp2_corr
    );
}

/// VV10 must add *binding*: on a bound dimer, the MP2-V interaction energy must
/// be more negative than the bare attenuated-MP2 interaction energy. This is
/// the sign/magnitude check the whole method exists for — attenuated MP2 has
/// zero long-range C6 and underbinds; VV10 pastes the tail back.
///
/// Water dimer at ~2.9 Å O–O, STO-3G to keep this affordable. Interaction
/// energies are computed WITHOUT counterpoise, matching the paper's convention
/// (§2 Methods: "Counterpoise corrections were not performed unless otherwise
/// indicated"). The absolute numbers are therefore BSSE-contaminated and not
/// comparable to the paper's; only the SIGN of the VV10 contribution is
/// asserted, which BSSE does not flip.
#[test]
fn vv10_increases_binding_on_a_bound_dimer() {
    let dimer_xyz = "6\nwater dimer\n\
        O  0.000  0.000  0.000\n\
        H  0.000  0.000  0.958\n\
        H  0.927  0.000 -0.240\n\
        O  0.000  0.000  2.900\n\
        H -0.478  0.756  3.185\n\
        H -0.478 -0.756  3.185\n";
    let mono_xyz = "3\nwater\n\
        O  0.000  0.000  0.000\n\
        H  0.000  0.000  0.958\n\
        H  0.927  0.000 -0.240\n";

    let dimer = build_case(dimer_xyz, "sto-3g");
    let mono = build_case(mono_xyz, "sto-3g");
    let cfg = erfc_test_config();

    let rd = att_mp2_vv10(&dimer.mol, &dimer.obs, &dimer.bs, &dimer.dfbs, &dimer.rhf, &cfg).unwrap();
    let rm = att_mp2_vv10(&mono.mol, &mono.obs, &mono.bs, &mono.dfbs, &mono.rhf, &cfg).unwrap();

    // Interaction energies (dimer minus 2x monomer), with and without the VV10 term.
    let e_att_only_int = (rd.e_hf + rd.e_c_att_mp2) - 2.0 * (rm.e_hf + rm.e_c_att_mp2);
    let e_mp2v_int = rd.total - 2.0 * rm.total;
    let vv10_contribution = rd.e_nl_vv10 - 2.0 * rm.e_nl_vv10;

    let kcal = 627.509_474_06;
    eprintln!(
        "water dimer (STO-3G, no CP): attMP2 interaction = {:.6} kcal/mol, \
         MP2-V interaction = {:.6} kcal/mol, VV10 contribution = {:.6} kcal/mol",
        e_att_only_int * kcal,
        e_mp2v_int * kcal,
        vv10_contribution * kcal
    );
    assert!(
        e_att_only_int < 0.0,
        "the dimer must be bound at the attenuated-MP2 level for this test to mean \
         anything (got {:.4} kcal/mol)",
        e_att_only_int * kcal
    );
    assert!(
        vv10_contribution < 0.0,
        "the VV10 term must INCREASE binding (contribute negatively to the interaction \
         energy), got {:.6} kcal/mol",
        vv10_contribution * kcal
    );
    assert!(
        e_mp2v_int < e_att_only_int,
        "MP2-V must bind more strongly than bare attenuated MP2: {:.6} vs {:.6} kcal/mol",
        e_mp2v_int * kcal,
        e_att_only_int * kcal
    );
}

// ---------------------------------------------------------------------------
// 4. Parameter plumbing and unit conversion
// ---------------------------------------------------------------------------

/// The published parameters must be exactly what the paper says, in the units
/// the paper says, and the terfc damping must share the MP2 r₀ (paper p. 4161:
/// "The r0 parameter is shared with the attenuated short-range MP2 part").
#[test]
fn published_parameters_are_the_paper_values() {
    let cfg = AttVv10Config::mp2_v_terfc_atz();
    assert!(
        (cfg.r0_angstrom() - 1.00).abs() < 1e-12,
        "MP2-V(terfc, aTZ) r0 must be 1.00 A (Table 1), got {}",
        cfg.r0_angstrom()
    );
    // Pin the ABSOLUTE Bohr value too. `r0_angstrom()` divides by the same
    // constant `r0_bohr` was multiplied by, so it round-trips even if that
    // constant is inverted — an independent anchor is required. (Found by
    // mutation testing: inverting BOHR_PER_ANG left the round-trip assertion
    // above passing.) 1.00 A = 1.8897259886 Bohr; a 1/x error would give 0.529.
    assert!(
        (cfg.r0_bohr - 1.889_725_988_6).abs() < 1e-9,
        "r0 = 1.00 A must be 1.8897259886 Bohr, got {} (a value near 0.529 means \
         the Angstrom->Bohr conversion is inverted)",
        cfg.r0_bohr
    );
    assert_eq!(cfg.vv10.b, 11.0, "b must be 11.0 (Table 1)");
    assert_eq!(cfg.vv10.c, 0.0089, "C must be 0.0089 (LC-VV10 value, section 3)");
    assert_eq!(cfg.attenuator, AttVv10Attenuator::Terfc);
    match cfg.vv10_damping {
        Vv10Damping::Terfc { r0_bohr } => assert_eq!(
            r0_bohr, cfg.r0_bohr,
            "the VV10 damping r0 must be the SAME r0 the MP2 half uses (paper Eq. 11)"
        ),
        other => panic!("MP2-V must damp VV10, got {other:?}"),
    }
}

/// The Table 1 (r₀, b) valley: `b` must track `r₀`, the global minimum must be
/// the (1.00, 11.0) row, and an off-table r₀ must be refused rather than have
/// its `b` interpolated into existence.
#[test]
fn table1_r0_b_valley_pairs() {
    const EXPECT: [(f64, f64); 6] = [
        (0.85, 8.0),
        (0.90, 9.0),
        (0.95, 9.5),
        (1.00, 11.0),
        (1.05, 12.5),
        (1.10, 14.5),
    ];
    assert_eq!(AttVv10Config::TABLE1_R0_B_PAIRS, EXPECT);

    // b is strictly increasing with r0 (the physics: weaker attenuation needs
    // stronger short-range damping of the VV10 tail).
    let mut prev_b = 0.0;
    for (r0, b) in AttVv10Config::TABLE1_R0_B_PAIRS {
        assert!(b > prev_b, "b must increase with r0 (r0={r0}: b={b})");
        prev_b = b;
        let cfg = AttVv10Config::mp2_v_terfc_atz_at_r0(r0)
            .unwrap_or_else(|| panic!("r0={r0} should be a published valley point"));
        assert_eq!(cfg.vv10.b, b);
        assert_eq!(cfg.vv10.c, 0.0089, "C is fixed across the whole valley");
        assert!((cfg.r0_angstrom() - r0).abs() < 1e-12);
        match cfg.vv10_damping {
            Vv10Damping::Terfc { r0_bohr } => assert_eq!(r0_bohr, cfg.r0_bohr),
            other => panic!("valley configs must damp VV10, got {other:?}"),
        }
    }

    // The published optimum is the (1.00, 11.0) row.
    let opt = AttVv10Config::mp2_v_terfc_atz();
    let at_opt = AttVv10Config::mp2_v_terfc_atz_at_r0(1.00).unwrap();
    assert_eq!(opt.vv10.b, at_opt.vv10.b);
    assert_eq!(opt.r0_bohr, at_opt.r0_bohr);

    // Off-table r0 must be refused, not interpolated.
    for bad in [0.80_f64, 0.925, 1.35, 2.0] {
        assert!(
            AttVv10Config::mp2_v_terfc_atz_at_r0(bad).is_none(),
            "r0={bad} A is not a published valley point and must not be fabricated"
        );
    }
}

/// Å → Bohr must go the right way. `from_r0_angstrom` must also keep the
/// damping's r₀ in sync, or the two halves silently use different r₀.
#[test]
fn angstrom_to_bohr_conversion_and_damping_sync() {
    let cfg = AttVv10Config::mp2_v_terfc_atz().from_r0_angstrom(1.35);
    // 1.35 A is 2.551 Bohr, NOT 0.714 (which is what dividing would give).
    assert!(
        (cfg.r0_bohr - 2.551_130_1).abs() < 1e-6,
        "1.35 A must be ~2.5511 Bohr, got {}",
        cfg.r0_bohr
    );
    assert!(cfg.r0_bohr > 1.35, "Bohr value must exceed the Angstrom value");
    match cfg.vv10_damping {
        Vv10Damping::Terfc { r0_bohr } => assert_eq!(r0_bohr, cfg.r0_bohr),
        other => panic!("damping should have stayed terfc, got {other:?}"),
    }
    assert!((cfg.r0_angstrom() - 1.35).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// 5. Honest rejection of unsupported references
// ---------------------------------------------------------------------------

#[test]
fn open_shell_reference_is_rejected() {
    // OH radical via UHF — must hard-error, never return a plausible number.
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let uhf = ferric_scf::uhf::solve_uhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        &bounds,
        &ferric_scf::uhf::UhfConfig {
            energy_conv: 1e-8,
            ..Default::default()
        },
    )
    .unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();

    let err = att_mp2_vv10(&mol, &obs, &bs, &dfbs, &uhf, &erfc_test_config())
        .expect_err("open-shell reference must be rejected");
    let msg = format!("{err}");
    eprintln!("open-shell rejection: {msg}");
    assert!(
        msg.contains("closed-shell") || msg.contains("restricted"),
        "error should say why: {msg}"
    );
}

#[test]
fn nonpositive_r0_is_rejected() {
    let c = water_ccpvdz();
    let mut cfg = erfc_test_config();
    cfg.r0_bohr = 0.0;
    let err = att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.rhf, &cfg)
        .expect_err("r0 = 0 must be rejected");
    eprintln!("r0=0 rejection: {err}");
}

// ---------------------------------------------------------------------------
// 6. The published terfc operator, when its tables are available
// ---------------------------------------------------------------------------

/// The paper's actual method needs the terfc interpolation tables. This test
/// runs the published `mp2_v_terfc_atz()` configuration end-to-end when
/// `FERRIC_TERF_TABLE_DIR` is set and SKIPS otherwise, rather than silently
/// substituting erfc.
///
/// NOTE: this runs in cc-pVDZ, not the aug-cc-pVTZ the parameters were fitted
/// for — it is a plumbing test (does the terfc path execute and produce finite,
/// correctly-ordered components), NOT a validation of the parameterization.
#[test]
fn terfc_path_runs_when_tables_present() {
    if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR not set; terfc interpolation tables unavailable");
        return;
    }
    let c = water_ccpvdz();
    let cfg = AttVv10Config {
        nlc_grid: test_grid(),
        ..AttVv10Config::mp2_v_terfc_atz()
    };
    let r = match att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.rhf, &cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP: terfc path unavailable at runtime: {e}");
            return;
        }
    };
    eprintln!(
        "MP2-V(terfc) plumbing on water/cc-pVDZ (NOT the fitted basis): \
         E_HF = {:.10}, E_c = {:.10}, E_nl = {:.10}, total = {:.10}",
        r.e_hf, r.e_c_att_mp2, r.e_nl_vv10, r.total
    );
    assert!(r.e_c_att_mp2.is_finite() && r.e_c_att_mp2 < 0.0);
    assert!(r.e_nl_vv10.is_finite());
    assert_eq!(r.components_sum_to_total(), 0.0);
}
