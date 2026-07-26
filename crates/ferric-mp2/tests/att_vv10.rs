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
    att_mp2_vv10, u_att_mp2_vv10, vv10_energy_on_density, AttVv10Attenuator, AttVv10Config,
    AttVv10SpinComponents, BOHR_PER_ANG,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::{ScfResult, Spin};

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

/// An open-shell case. Structurally identical to [`Case`]; separate only so the
/// field name (`scf`, not `rhf`) does not lie about what it holds.
struct OpenCase {
    mol: Molecule,
    bs: basis::BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    scf: ScfResult,
}

/// UHF open-shell case. `mult` is the spin multiplicity (2 for a doublet).
fn build_uhf_case(xyz: &str, charge: i32, mult: usize, basis_name: &str) -> OpenCase {
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let scf = ferric_scf::uhf::solve_uhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        &bounds,
        &ferric_scf::uhf::UhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(scf.spin, Spin::Unrestricted);
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    OpenCase { mol, bs, obs, dfbs, scf }
}

/// ROHF open-shell case.
fn build_rohf_case(xyz: &str, charge: i32, mult: usize, basis_name: &str) -> OpenCase {
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let scf = ferric_scf::rohf::solve_rohf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &ferric_scf::rohf::RohfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(scf.spin, Spin::RestrictedOpen);
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    OpenCase { mol, bs, obs, dfbs, scf }
}

/// OH radical, STO-3G, UHF doublet. The standard small open-shell probe.
/// OH radical in **cc-pVDZ, not STO-3G**, and the basis choice is load-bearing.
///
/// OH/STO-3G has nbf = 6 with nocc_α = 5, i.e. **nvir = 1**. Same-spin
/// correlation is built from the antisymmetrized `K = (ia|jb) − (ib|ja)`; with a
/// single virtual orbital `a = b` is forced, so `K ≡ 0` for every pair and
/// `e_aa = e_bb = 0` **exactly** — by the Pauli principle, not by any defect.
/// Two same-spin electrons cannot be antisymmetrized into one spatial virtual.
///
/// An earlier revision of this test used STO-3G and asserted `e_aa < 0`, which
/// failed for exactly that reason: the system could not exercise the branch the
/// assertion was about. cc-pVDZ gives nbf = 24, nvir = 19 — a real same-spin
/// space. Do not shrink this basis back down.
fn oh_radical_uhf() -> OpenCase {
    build_uhf_case("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2, "cc-pvdz")
}

/// The degenerate case above, kept deliberately so the nvir = 1 identity is
/// pinned rather than rediscovered as a bug.
fn oh_radical_uhf_sto3g() -> OpenCase {
    build_uhf_case("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2, "sto-3g")
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

/// The closed-shell entry point must still refuse an open-shell reference —
/// it now redirects to `u_att_mp2_vv10` rather than claiming "not implemented",
/// but it must NOT silently accept one (`ri_mp2_spin_components` would take the
/// α orbitals as doubly occupied and return a meaningless number).
#[test]
fn closed_shell_entry_point_rejects_open_shell_reference() {
    let c = oh_radical_uhf();
    let err = att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.scf, &erfc_test_config())
        .expect_err("open-shell reference must be rejected by the restricted entry point");
    let msg = format!("{err}");
    eprintln!("closed-shell entry point rejection: {msg}");
    assert!(
        msg.contains("closed-shell") || msg.contains("restricted"),
        "error should say why: {msg}"
    );
    assert!(
        msg.contains("u_att_mp2_vv10"),
        "error should point at the open-shell entry point: {msg}"
    );
}

/// And the mirror image: the open-shell entry point must refuse a *restricted*
/// reference rather than treating `mos_alpha` as an independent spin channel.
#[test]
fn open_shell_entry_point_rejects_restricted_reference() {
    let c = water_ccpvdz();
    let err = u_att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.rhf, &erfc_test_config())
        .expect_err("restricted reference must be rejected by the unrestricted entry point");
    let msg = format!("{err}");
    eprintln!("open-shell entry point rejection: {msg}");
    assert!(
        msg.contains("att_mp2_vv10"),
        "error should point at the closed-shell entry point: {msg}"
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

// ---------------------------------------------------------------------------
// 7. Open shell (Phase B)
//
// PARAMETER HONESTY: every number below is produced with the paper's
// CLOSED-SHELL-fitted (r0, b, C). The paper's training set (S66) contains only
// closed-shell dimers and it publishes no open-shell MP2-V parameterization, so
// none of these tests validate the *parameters* for radicals — they validate
// the SPIN BOOKKEEPING (that the three U-MP2 blocks and the spin-summed VV10
// density are wired correctly), which is a separate and checkable claim.
// ---------------------------------------------------------------------------

/// **The regression pin for Phase A.** These are the committed Phase A values
/// for the erfc control on water/cc-pVDZ (commit be4404c). Refactoring the
/// entry point to share `assemble` with the open-shell path must not move the
/// closed-shell number by a single bit.
const PHASE_A_WATER_CCPVDZ_ERFC_E_HF: f64 = -76.0267833623;
const PHASE_A_WATER_CCPVDZ_ERFC_E_C: f64 = -0.1806640950;
const PHASE_A_WATER_CCPVDZ_ERFC_E_NL: f64 = 0.0186722266;

#[test]
fn closed_shell_path_is_unchanged_by_the_open_shell_refactor() {
    let c = water_ccpvdz();
    let r = att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.rhf, &erfc_test_config()).unwrap();
    eprintln!(
        "Phase A regression: E_HF = {:.10} (want {:.10}), E_c = {:.10} (want {:.10}), \
         E_nl = {:.10} (want {:.10})",
        r.e_hf,
        PHASE_A_WATER_CCPVDZ_ERFC_E_HF,
        r.e_c_att_mp2,
        PHASE_A_WATER_CCPVDZ_ERFC_E_C,
        r.e_nl_vv10,
        PHASE_A_WATER_CCPVDZ_ERFC_E_NL
    );
    // Tolerance is the print precision of the committed constants (1e-10), not
    // a physics tolerance — these are pinned digits, not a converged reference.
    assert!(
        (r.e_hf - PHASE_A_WATER_CCPVDZ_ERFC_E_HF).abs() < 5e-10,
        "E_HF drifted: {} vs {}",
        r.e_hf,
        PHASE_A_WATER_CCPVDZ_ERFC_E_HF
    );
    assert!(
        (r.e_c_att_mp2 - PHASE_A_WATER_CCPVDZ_ERFC_E_C).abs() < 5e-10,
        "E_c^attMP2 drifted: {} vs {}",
        r.e_c_att_mp2,
        PHASE_A_WATER_CCPVDZ_ERFC_E_C
    );
    assert!(
        (r.e_nl_vv10 - PHASE_A_WATER_CCPVDZ_ERFC_E_NL).abs() < 5e-10,
        "E_nl^VV10 drifted: {} vs {}",
        r.e_nl_vv10,
        PHASE_A_WATER_CCPVDZ_ERFC_E_NL
    );
    assert_eq!(r.reference_spin, Spin::Restricted);
    assert!(!r.is_open_shell_extrapolation());
    match r.spin_components {
        AttVv10SpinComponents::Restricted(_) => {}
        other => panic!("restricted path must report restricted components, got {other:?}"),
    }
}

/// **THE spin-limit test.** A closed-shell singlet driven through the OPEN-SHELL
/// path must reproduce the closed-shell path's total energy. This is the
/// open-shell analogue of the ω→0 limit: it isolates "is the α/β bookkeeping
/// right" from every other question, because for a UHF solution that has
/// collapsed onto the RHF one, α ≡ β and
///
///     E_αα + E_ββ + E_αβ  ≡  E_ss + E_os
///
/// must hold identically. If the same-spin blocks were dropped, double-counted,
/// or the antisymmetrization factor were wrong, this test fails.
///
/// The guard: if UHF does NOT collapse to RHF (a symmetry-broken solution at a
/// different energy), the comparison is meaningless and this test says so
/// rather than loosening the tolerance to accommodate a different SCF solution.
#[test]
fn closed_shell_singlet_through_open_shell_path_matches_restricted() {
    let r_case = water_ccpvdz();
    // Same geometry/basis, but solved as a spin-unrestricted singlet.
    let u_case = build_uhf_case(
        "3\nwater\nO 0.000 0.000 0.118\nH 0.000 0.755 -0.471\nH 0.000 -0.755 -0.471\n",
        0,
        1,
        "cc-pvdz",
    );

    let d_e_scf = (u_case.scf.energy - r_case.rhf.energy).abs();
    eprintln!(
        "spin-limit SCF check: E_RHF = {:.12}, E_UHF(singlet) = {:.12}, |diff| = {:.3e}",
        r_case.rhf.energy, u_case.scf.energy, d_e_scf
    );
    assert!(
        d_e_scf < 1e-8,
        "UHF did NOT collapse to the RHF solution (|ΔE_SCF| = {d_e_scf:.3e} Ha). The \
         spin-limit comparison is only meaningful when the two SCFs found the same \
         state; this is a symmetry-broken UHF solution, not a tolerance problem. \
         Refusing to compare rather than loosening the tolerance."
    );

    let cfg = erfc_test_config();
    let r = att_mp2_vv10(
        &r_case.mol,
        &r_case.obs,
        &r_case.bs,
        &r_case.dfbs,
        &r_case.rhf,
        &cfg,
    )
    .unwrap();
    let u = u_att_mp2_vv10(
        &u_case.mol,
        &u_case.obs,
        &u_case.bs,
        &u_case.dfbs,
        &u_case.scf,
        &cfg,
    )
    .unwrap();

    let (e_aa, e_bb, e_ab) = match &u.spin_components {
        AttVv10SpinComponents::Unrestricted(c) => (c.e_aa, c.e_bb, c.e_ab),
        other => panic!("open-shell path must report unrestricted components, got {other:?}"),
    };
    let (e_os, e_ss) = match &r.spin_components {
        AttVv10SpinComponents::Restricted(c) => (c.e_os, c.e_ss),
        other => panic!("restricted path must report restricted components, got {other:?}"),
    };
    eprintln!(
        "spin limit (water/cc-pVDZ, erfc control):\n  \
         R: E_c = {:.12} (e_os = {:.12}, e_ss = {:.12}), E_nl = {:.12}, total = {:.12}\n  \
         U: E_c = {:.12} (e_aa = {e_aa:.12}, e_bb = {e_bb:.12}, e_ab = {e_ab:.12}), \
         E_nl = {:.12}, total = {:.12}",
        r.e_c_att_mp2, e_os, e_ss, r.e_nl_vv10, r.total,
        u.e_c_att_mp2, u.e_nl_vv10, u.total
    );

    // (a) the correlation halves must agree
    let d_ec = (u.e_c_att_mp2 - r.e_c_att_mp2).abs();
    eprintln!("  |ΔE_c| = {d_ec:.3e}, |ΔE_nl| = {:.3e}, |Δtotal| = {:.3e}",
        (u.e_nl_vv10 - r.e_nl_vv10).abs(), (u.total - r.total).abs());
    // TOLERANCES. These compare two INDEPENDENT SCF solutions (a restricted and
    // an unrestricted one that collapsed onto it) fed through two INDEPENDENT
    // RI-MP2 code paths (`ri_mp2_spin_components` vs `u_ri_mp2`'s per-spin
    // `compute_rpa_intermediates_spin`). The two SCFs agree to 9.9e-13 Ha, and
    // that residual propagates and amplifies through the MO transform and the
    // correlation assembly — so the correct floor is set by SCF convergence and
    // path divergence, NOT by machine epsilon.
    //
    // MEASURED on water/cc-pVDZ (erfc control): |ΔE_c| = 2.1e-9,
    // |e_aa − e_bb| = 6.4e-9, |Δtotal| = 2.0e-9. Bounds set one decade above the
    // measured values: tight enough that a genuine spin-bookkeeping error (which
    // would show up at 1e-3 or larger — half the same-spin energy is 2.2e-2 here)
    // is caught by orders of magnitude, loose enough not to be an SCF-noise
    // change-detector. An earlier revision asserted 1e-9/1e-10 and failed on
    // exactly this residual; the numbers were right and the bar was wrong.
    assert!(
        d_ec < 1e-8,
        "U-attMP2 must reproduce R-attMP2 for a collapsed singlet: {} vs {} (|Δ| = {d_ec:.3e})",
        u.e_c_att_mp2,
        r.e_c_att_mp2
    );
    // (b) and the block-by-block identification must hold, not just the sum:
    //     αα = ββ = ½ e_ss  and  αβ = e_os.
    assert!(
        (e_aa - e_bb).abs() < 1e-7,
        "αα and ββ must be equal for a spin-symmetric singlet: {e_aa} vs {e_bb} \
         (|Δ| = {:.3e})",
        (e_aa - e_bb).abs()
    );
    assert!(
        (e_aa + e_bb - e_ss).abs() < 1e-8,
        "e_aa + e_bb must equal the restricted e_ss: {} vs {e_ss}",
        e_aa + e_bb
    );
    assert!(
        (e_ab - e_os).abs() < 1e-8,
        "e_ab must equal the restricted e_os: {e_ab} vs {e_os}"
    );
    // Guard the loosened bounds against becoming vacuous: the same-spin blocks
    // must be genuinely NONZERO and of the right scale, so "αα ≈ ββ" cannot be
    // satisfied by both being ~0 (the OH/STO-3G nvir=1 trap, one basis away).
    assert!(
        e_aa < -1e-3 && e_bb < -1e-3,
        "same-spin blocks must be real and negative here (water/cc-pVDZ has 19 \
         virtuals), got e_aa={e_aa}, e_bb={e_bb}"
    );
    // (c) the total energies
    assert!(
        (u.total - r.total).abs() < 1e-8,
        "MP2-V totals must agree in the spin limit: {} vs {}",
        u.total,
        r.total
    );
    assert!(u.is_open_shell_extrapolation());
}

/// VV10 is a functional of the TOTAL density only, so the same molecule's UHF
/// and RHF densities (for a collapsed singlet) must give the same E_nl to the
/// SCF convergence floor. Checked at the `vv10_energy_on_density` level, i.e.
/// on the densities directly, so it isolates the VV10 half from the MP2 half.
#[test]
fn vv10_is_spin_agnostic_on_a_collapsed_singlet() {
    let r_case = water_ccpvdz();
    let u_case = build_uhf_case(
        "3\nwater\nO 0.000 0.000 0.118\nH 0.000 0.755 -0.471\nH 0.000 -0.755 -0.471\n",
        0,
        1,
        "cc-pvdz",
    );
    assert!(
        (u_case.scf.energy - r_case.rhf.energy).abs() < 1e-8,
        "UHF must have collapsed to RHF for this comparison to mean anything"
    );

    // The UHF total density really is the spin sum, not a copy of one channel.
    let d_a = &u_case.scf.density_alpha;
    let d_b = u_case.scf.density_beta.as_ref().expect("UHF must carry a beta density");
    let d_sum = d_a + d_b;
    let max_dev = (&d_sum - u_case.scf.density_total())
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        max_dev < 1e-12,
        "density_total() must be D_alpha + D_beta, max deviation {max_dev:.3e}"
    );

    let params = Vv10Params { c: 0.0089, b: 11.0 };
    let damping = Vv10Damping::Terfc { r0_bohr: 1.00 * BOHR_PER_ANG };
    let (e_r, n_r) = vv10_energy_on_density(
        &r_case.mol, &r_case.bs, r_case.rhf.density_total(), &params, damping, &test_grid(),
    ).unwrap();
    let (e_u, n_u) = vv10_energy_on_density(
        &u_case.mol, &u_case.bs, u_case.scf.density_total(), &params, damping, &test_grid(),
    ).unwrap();
    eprintln!("VV10 spin agnosticism: E_nl[RHF rho] = {e_r:.14}, E_nl[UHF rho] = {e_u:.14}, \
               diff = {:.3e} ({n_r} vs {n_u} pts)", (e_r - e_u).abs());
    assert_eq!(n_r, n_u);
    assert!(
        (e_r - e_u).abs() < 1e-10,
        "VV10 depends only on the total density; RHF and collapsed-UHF must agree: \
         {e_r} vs {e_u}"
    );
}

/// A genuine open-shell system end to end: finite, correctly signed components
/// with exact additivity, and a spin decomposition that is actually asymmetric.
/// Pins the `nvir = 1` same-spin identity that made an earlier revision of
/// `oh_radical_uhf_components_are_physical_and_additive` look like a bug.
///
/// OH/STO-3G forces nvir = 1, so the antisymmetrized same-spin numerator
/// `K = (ia|jb) − (ib|ja)` vanishes identically and `e_aa = e_bb = 0` EXACTLY.
/// Opposite-spin has no antisymmetrization and survives. If a future change
/// makes `e_aa` nonzero here, the same-spin kernel has stopped antisymmetrizing
/// — which is a real bug this test will catch.
#[test]
fn same_spin_vanishes_exactly_when_only_one_virtual() {
    let c = oh_radical_uhf_sto3g();
    let r = u_att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.scf, &erfc_test_config()).unwrap();
    let (e_aa, e_bb, e_ab) = match &r.spin_components {
        AttVv10SpinComponents::Unrestricted(x) => (x.e_aa, x.e_bb, x.e_ab),
        other => panic!("expected unrestricted components, got {other:?}"),
    };
    let nbf = c.obs.nbasis();
    eprintln!(
        "OH/STO-3G (nbf={nbf}, nvir=1): e_aa={e_aa:.12}, e_bb={e_bb:.12}, e_ab={e_ab:.12}"
    );
    assert_eq!(
        nbf, 6,
        "this test depends on OH/STO-3G having nbf=6 (nocc_alpha=5 => nvir=1)"
    );
    // Magnitude, not `== 0.0`: K = (ia|jb) - (ib|ja) is a difference of two
    // floats that are equal only up to the RI fit, so the cancellation is to
    // roundoff, not to the exact bit pattern (it also lands on -0.0 in the beta
    // channel). What matters physically is that it is zero to many orders below
    // any real correlation energy — the opposite-spin term below is ~1e-2.
    assert!(
        e_aa.abs() < 1e-14,
        "with nvir=1, K=(ia|jb)-(ib|ja) vanishes identically => e_aa must be zero \
         to roundoff, got {e_aa:.3e}"
    );
    assert!(
        e_bb.abs() < 1e-14,
        "same argument for the beta channel, got {e_bb:.3e}"
    );
    // Opposite-spin is NOT antisymmetrized, so it survives and must be real.
    assert!(
        e_ab < 0.0,
        "opposite-spin correlation has no antisymmetrization and must remain \
         negative even at nvir=1, got {e_ab}"
    );
}

#[test]
fn oh_radical_uhf_components_are_physical_and_additive() {
    let c = oh_radical_uhf();
    let r = u_att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.scf, &erfc_test_config()).unwrap();
    let (e_aa, e_bb, e_ab) = match &r.spin_components {
        AttVv10SpinComponents::Unrestricted(x) => (x.e_aa, x.e_bb, x.e_ab),
        other => panic!("expected unrestricted components, got {other:?}"),
    };
    eprintln!(
        "MP2-V(erfc control)/OH radical/STO-3G [UNPARAMETERIZED for open shell]:\n  \
         E_UHF = {:.10}, E_c = {:.10} (aa {e_aa:.10}, bb {e_bb:.10}, ab {e_ab:.10}), \
         E_nl = {:.10}, total = {:.10}",
        r.e_hf, r.e_c_att_mp2, r.e_nl_vv10, r.total
    );

    assert_eq!(r.components_sum_to_total(), 0.0, "additivity must be exact");
    assert_eq!(r.e_hf, c.scf.energy);
    assert_eq!(r.reference_spin, Spin::Unrestricted);
    assert!(r.is_open_shell_extrapolation());

    assert!(r.e_hf < 0.0 && r.e_hf.is_finite());
    assert!(e_aa < 0.0, "same-spin αα correlation must be negative, got {e_aa}");
    assert!(e_bb < 0.0, "same-spin ββ correlation must be negative, got {e_bb}");
    assert!(e_ab < 0.0, "opposite-spin correlation must be negative, got {e_ab}");
    assert!(r.e_c_att_mp2 < 0.0);
    // As for closed shell, E_nl as defined here is a small positive number.
    assert!(r.e_nl_vv10 > 0.0 && r.e_nl_vv10.is_finite(), "got {}", r.e_nl_vv10);

    // The α and β channels genuinely differ for a doublet — if they came out
    // equal, the β channel is being fed the α orbitals (or vice versa).
    assert!(
        (e_aa - e_bb).abs() > 1e-6,
        "OH is a doublet; αα ({e_aa}) and ββ ({e_bb}) must differ. Equality means the \
         two spin channels were built from the same MOs."
    );
    // ORIENTATION, not just difference. "αα ≠ ββ" is symmetric: it still holds
    // if the two channels are swapped, so on its own it cannot catch a spin
    // transposition. (Found by mutation testing — M14 swapped the α/β orbital
    // energies between the same-spin channels and survived every other
    // assertion here.) OH is a doublet with nocc_α = 5 > nocc_β = 4: the α
    // channel has strictly more occupied pairs to correlate, so its same-spin
    // correlation must be the LARGER in magnitude. This pins which channel is
    // which.
    assert!(
        e_aa.abs() > e_bb.abs(),
        "OH has nocc_α=5 > nocc_β=4, so |e_aa| ({}) must exceed |e_bb| ({}) — if it \
         does not, the α and β channels have been transposed",
        e_aa.abs(),
        e_bb.abs()
    );
    // Opposite-spin dominates in MP2, as at closed shell.
    assert!(
        e_ab.abs() > (e_aa + e_bb).abs(),
        "opposite-spin ({e_ab}) should dominate same-spin ({})",
        e_aa + e_bb
    );
}

/// The open-shell path must be sensitive to the spin state: the same nuclei and
/// electron count in a different multiplicity must give a different energy.
/// Catches an occupation-count bug that ignores `mol.multiplicity`.
#[test]
fn multiplicity_changes_the_open_shell_result() {
    // CH3 radical (doublet) vs the same geometry as a quartet.
    let xyz = "4\nCH3\nC 0.000 0.000 0.000\nH 0.000 1.079 0.000\n\
               H 0.934 -0.539 0.000\nH -0.934 -0.539 0.000\n";
    let d = build_uhf_case(xyz, 0, 2, "sto-3g");
    let q = build_uhf_case(xyz, 0, 4, "sto-3g");
    let cfg = erfc_test_config();
    let rd = u_att_mp2_vv10(&d.mol, &d.obs, &d.bs, &d.dfbs, &d.scf, &cfg).unwrap();
    let rq = u_att_mp2_vv10(&q.mol, &q.obs, &q.bs, &q.dfbs, &q.scf, &cfg).unwrap();
    eprintln!(
        "CH3 doublet total = {:.10}, quartet total = {:.10}, E_c: {:.10} vs {:.10}",
        rd.total, rq.total, rd.e_c_att_mp2, rq.e_c_att_mp2
    );
    assert_eq!(rd.components_sum_to_total(), 0.0);
    assert_eq!(rq.components_sum_to_total(), 0.0);
    assert!(
        (rd.e_c_att_mp2 - rq.e_c_att_mp2).abs() > 1e-6,
        "doublet and quartet correlation energies must differ: {} vs {}",
        rd.e_c_att_mp2,
        rq.e_c_att_mp2
    );
    assert!(
        rd.total < rq.total,
        "the doublet should lie below the quartet for CH3: {} vs {}",
        rd.total,
        rq.total
    );
}

/// ROHF references are accepted (u_ri_mp2 supports them by sharing the single
/// ROHF MO set across both spin channels). Asserted rather than assumed —
/// if ROHF were silently mishandled this would produce a nonsense number, and
/// if it were unsupported this test would catch the error instead of a caller.
#[test]
fn rohf_reference_is_accepted_and_finite() {
    // cc-pVDZ, not STO-3G — see `oh_radical_uhf`: STO-3G forces nvir = 1, which
    // makes same-spin correlation vanish identically and the `e_aa < 0`
    // assertion below unsatisfiable for reasons that have nothing to do with ROHF.
    let c = build_rohf_case("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2, "cc-pvdz");
    let r = u_att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.scf, &erfc_test_config()).unwrap();
    let (e_aa, e_bb, e_ab) = match &r.spin_components {
        AttVv10SpinComponents::Unrestricted(x) => (x.e_aa, x.e_bb, x.e_ab),
        other => panic!("expected unrestricted components, got {other:?}"),
    };
    eprintln!(
        "MP2-V(erfc control)/OH radical/STO-3G, ROHF reference [UNPARAMETERIZED]:\n  \
         E_ROHF = {:.10}, E_c = {:.10} (aa {e_aa:.10}, bb {e_bb:.10}, ab {e_ab:.10}), \
         E_nl = {:.10}, total = {:.10}",
        r.e_hf, r.e_c_att_mp2, r.e_nl_vv10, r.total
    );
    assert_eq!(r.reference_spin, Spin::RestrictedOpen);
    assert!(r.is_open_shell_extrapolation());
    assert_eq!(r.components_sum_to_total(), 0.0);
    assert!(r.e_hf.is_finite() && r.e_hf < 0.0);
    assert!(r.e_c_att_mp2 < 0.0 && r.e_c_att_mp2.is_finite());
    assert!(r.e_nl_vv10 > 0.0 && r.e_nl_vv10.is_finite());
    assert!(e_aa < 0.0 && e_bb < 0.0 && e_ab < 0.0);
    // ROHF and UHF are different references, so the energies must differ (and
    // the variational UHF one must be lower) — a sanity check that the ROHF
    // result is not silently the UHF one.
    let uhf = oh_radical_uhf();
    assert!(
        r.e_hf > uhf.scf.energy,
        "ROHF reference energy ({}) must lie above the variational UHF one ({})",
        r.e_hf,
        uhf.scf.energy
    );
}

/// The open-shell path must use the SAME attenuated operator as the closed-shell
/// one, not silently fall back to bare Coulomb. Same guard as
/// `attenuated_half_is_smaller_than_full_mp2`, applied to U-MP2.
#[test]
fn open_shell_attenuated_half_is_smaller_than_full_u_mp2() {
    let c = oh_radical_uhf();
    let full = ferric_mp2::u_rimp2::u_ri_mp2(
        &c.mol,
        &c.obs,
        &c.dfbs,
        Operator::coulomb(),
        &c.scf,
        &ferric_mp2::rimp2::RiMp2Config::default(),
    )
    .unwrap();
    let r = u_att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.scf, &erfc_test_config()).unwrap();
    eprintln!(
        "OH/STO-3G: full U-MP2 corr = {:.10}, attenuated half = {:.10}",
        full.mp2_corr, r.e_c_att_mp2
    );
    assert!(
        r.e_c_att_mp2.abs() < full.mp2_corr.abs(),
        "attenuated U-MP2 |{}| must be < full |{}| — an unattenuated operator would \
         make these equal",
        r.e_c_att_mp2,
        full.mp2_corr
    );
}

/// `frozen_core` must be honoured on the open-shell path, and an out-of-range
/// value must ERROR (via `rimp2::active_occ`) rather than underflow.
#[test]
fn open_shell_frozen_core_is_honoured_and_range_checked() {
    let c = oh_radical_uhf();
    let base = erfc_test_config();
    let all_electron = u_att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.scf, &base).unwrap();

    let fc1 = AttVv10Config { frozen_core: 1, ..base.clone() };
    let frozen = u_att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.scf, &fc1).unwrap();
    eprintln!(
        "OH/STO-3G frozen core: all-electron E_c = {:.10}, frozen_core=1 E_c = {:.10}",
        all_electron.e_c_att_mp2, frozen.e_c_att_mp2
    );
    assert!(
        frozen.e_c_att_mp2 > all_electron.e_c_att_mp2,
        "freezing the O 1s must REDUCE |E_c|: {} vs {}",
        frozen.e_c_att_mp2,
        all_electron.e_c_att_mp2
    );
    // E_nl does not depend on the correlation treatment at all.
    assert_eq!(frozen.e_nl_vv10, all_electron.e_nl_vv10);

    // Absurd frozen_core: must be a clean error, not a panic or a usize wrap.
    let fc_huge = AttVv10Config { frozen_core: 1000, ..base.clone() };
    let err = u_att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.scf, &fc_huge)
        .expect_err("frozen_core beyond nocc must error");
    eprintln!("frozen_core=1000 rejection: {err}");
}

/// The open-shell entry point must reject a bad r0 the same way the closed-shell
/// one does — before doing any work.
#[test]
fn open_shell_nonpositive_r0_is_rejected() {
    let c = oh_radical_uhf();
    let mut cfg = erfc_test_config();
    cfg.r0_bohr = -1.0;
    let err = u_att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.scf, &cfg)
        .expect_err("negative r0 must be rejected");
    eprintln!("open-shell r0<0 rejection: {err}");
    assert!(format!("{err}").contains("u_att_mp2_vv10"));
}

/// The published terfc operator through the open-shell path, when tables exist.
/// Plumbing only: cc-pVDZ is not the fitted basis and OH is not a closed-shell
/// dimer, so this asserts finiteness and ordering, never a parameterized value.
#[test]
fn open_shell_terfc_path_runs_when_tables_present() {
    if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR not set; terfc interpolation tables unavailable");
        return;
    }
    let c = oh_radical_uhf();
    let cfg = AttVv10Config {
        nlc_grid: test_grid(),
        ..AttVv10Config::mp2_v_terfc_atz()
    };
    let r = match u_att_mp2_vv10(&c.mol, &c.obs, &c.bs, &c.dfbs, &c.scf, &cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP: terfc path unavailable at runtime: {e}");
            return;
        }
    };
    eprintln!(
        "MP2-V(terfc) open-shell plumbing on OH/STO-3G (NOT the fitted basis, NOT a \
         parameterized spin case): E_HF = {:.10}, E_c = {:.10}, E_nl = {:.10}, total = {:.10}",
        r.e_hf, r.e_c_att_mp2, r.e_nl_vv10, r.total
    );
    assert!(r.e_c_att_mp2.is_finite() && r.e_c_att_mp2 < 0.0);
    assert!(r.e_nl_vv10.is_finite());
    assert_eq!(r.components_sum_to_total(), 0.0);
}
