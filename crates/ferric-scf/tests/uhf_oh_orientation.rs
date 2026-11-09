//! OH ²Π UKS-LDA/STO-3G: convergence that does not depend on how the molecule
//! sits on the DFT grid, nor on which orientation of the degenerate β π hole
//! the eigensolver's rounding happens to pick.
//!
//! The integer-occupation stationary state is non-aufbau (its occupied β π
//! lies 5.0e-3 Ha above the empty one, on every grid from 75x110 to 150x590),
//! so aufbau moves the hole π_x ↔ π_y every iteration. The two ends are
//! degenerate physically and split only by the grid. MEASURED before the
//! degenerate-flip pin (`uhf::FlipPin`), production grid, energy_conv 1e-10,
//! density_conv 1e-9, bond tilted off z by `a` in a plane 0.4 rad about z:
//!
//! | a (deg) | 0 | 5 | 13 | 30 | 45 | 60 | 71 | 90 |
//! |---|---|---|---|---|---|---|---|---|
//! | converged | yes* | no | no | no | no | no | no | no |
//!
//! (*) accepted as a degenerate swap with the "not a stationary point" warning
//! (the one-step model drop is the inverted gap, -5.0e-3 Ha). The flip
//! changed E by 2.6e-6..3.3e-6 Ha at 30° and 90° on the production grid,
//! 2e-9..6e-9 at 75x302 and 6e-11..4e-10 at 99x590: grid quadrature, not
//! physics. Hartree-Fock (no grid) is aufbau and orientation-invariant to
//! 1e-13 Ha.
//!
//! After (`uhf::FlipPin` + TRAH with the Fock's own DF-J response): all eight
//! converge in 8-12 iterations, no warning, on both this config and the CLI
//! defaults (RI-J). Energies across tilts still spread 7e-5 Ha on the
//! production grid — the molecular grid's own rotational variance (closed-shell
//! OH⁻ spreads 9.7e-5 Ha at the same tilts) — and 2.6e-9 Ha at 99x590.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_j::RiJPrecision;
use ferric_scf::result::non_stationary_warnings_on_this_thread;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{degenerate_flip_pins_on_this_thread, solve_uhf, solve_uhf_with_guess};
use ferric_scf::ScfResult;
use ndarray::{s, Array2};

/// α/β occupied counts of the OH doublet.
const NOCC: (usize, usize) = (5, 4);

/// OH at 0.98 Å with the bond tilted `tilt_deg` off z, in the plane rotated
/// 0.4 rad about z (the Becke/Lebedev grid is not rotation invariant).
fn oh_tilted(tilt_deg: f64) -> Molecule {
    let a = tilt_deg.to_radians();
    let r = 0.98;
    let (hx, hy, hz) = (
        r * a.sin() * 0.4f64.cos(),
        r * a.sin() * 0.4f64.sin(),
        r * a.cos(),
    );
    let xyz = format!("2\nOH\nO 0 0 0\nH {hx:.12} {hy:.12} {hz:.12}\n");
    Molecule::parse_xyz(&xyz, 0, 2).unwrap()
}

struct Sys {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
}

fn sys(tilt_deg: f64) -> Sys {
    let mol = oh_tilted(tilt_deg);
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Sys { mol, prep, bounds }
}

fn lda(energy_conv: f64, density_conv: f64) -> RhfConfig {
    RhfConfig {
        xc: Some("LDA".to_string()),
        energy_conv,
        density_conv,
        max_iter: 60,
        // The flip splits at the 1e-6..1e-9 Ha level, below f32 RI-J rounding.
        ri_j_precision: Some(RiJPrecision::F64),
        ..Default::default()
    }
}

/// max |C_occ C_occᵀ − D| over both spins.
fn mo_density_miss(r: &ScfResult) -> f64 {
    let dev = |c: &Array2<f64>, d: &Array2<f64>, n: usize| {
        let occ = c.slice(s![.., ..n]);
        (occ.dot(&occ.t()) - d)
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()))
    };
    dev(&r.mos_alpha, &r.density_alpha, NOCC.0).max(dev(
        r.mos_beta.as_ref().unwrap(),
        r.density_beta.as_ref().unwrap(),
        NOCC.1,
    ))
}

/// Tilts of the sweep: on and off the grid's C4 axis, both octants.
const TILTS: [f64; 8] = [0.0, 5.0, 13.0, 30.0, 45.0, 60.0, 71.0, 90.0];

/// Every orientation converges, at tight thresholds, to a stationary state:
/// the degenerate flip is pinned (reachability), the run ends within a few
/// iterations of the flip, no non-stationary warning fires, and the returned
/// MOs build the returned density.
///
/// MEASURED after the pin: 8-15 iterations; `err_max` 1.0e-10..1.03e-7 (the
/// largest at 13°: 2.5e-8 from the unturned canonical guess, 1.03e-7 from the
/// default one turned by `uhf::GUESS_HOLE_ANGLE`); every stability
/// `lambda_min` at or above -2.2e-7 (inside the 1e-6 noise floor). The 13°
/// residual is a stiff-mode one: the TRAH step that would remove the 2.5e-8
/// is 2.8e-9 long (curvature ~9 Ha; below the ΔP gate's 10·density_conv) and
/// worth 8e-18 Ha. Bar: a step just under the gate (10·density_conv) along
/// the stiffest rotation, whose curvature is at most the occupied-virtual
/// span (18.6 Ha MEASURED, O 1s to the top STO-3G virtual), leaves a gradient up to
/// ~1.9e-7, so 2e-7. The oscillation it replaces never converged (the
/// `converged` and iteration checks catch it) and sat at 1e-7..1e-6.
#[test]
fn oh_uks_lda_converges_stationary_at_every_orientation() {
    let cfg = lda(1e-10, 1e-9);
    let mut failures = Vec::new();
    for tilt in TILTS {
        let s = sys(tilt);
        let (pins0, warn0) = (
            degenerate_flip_pins_on_this_thread(),
            non_stationary_warnings_on_this_thread(),
        );
        let r = match solve_uhf(
            &ParallelContext::default(),
            &s.mol,
            &s.prep,
            &s.bounds,
            &cfg,
        ) {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("{tilt}°: {e}"));
                continue;
            }
        };
        let pins = degenerate_flip_pins_on_this_thread() - pins0;
        let warns = non_stationary_warnings_on_this_thread() - warn0;
        let err_max = r.convergence.as_ref().map(|c| c.err_max).unwrap();
        let miss = mo_density_miss(&r);
        println!(
            "{tilt:5.1}°: E {:.10} in {} iters, err_max {err_max:.2e}, pins {pins}, \
             warnings {warns}, |D(C_occ)-D| {miss:.1e}",
            r.energy, r.iterations
        );
        if !(r.converged && r.iterations <= 30 && pins == 1 && warns == 0 && err_max < 2e-7)
            || miss > 1e-6
        {
            failures.push(format!(
                "{tilt}°: converged {} iters {} pins {pins} warnings {warns} \
                 err_max {err_max:.2e} miss {miss:.1e}",
                r.converged, r.iterations
            ));
        }
    }
    assert!(failures.is_empty(), "orientations failed: {failures:#?}");
}

/// On a grid fine enough that the π_x/π_y split is ~1e-10 Ha, the converged
/// energy does not depend on the orientation.
///
/// MEASURED at 99x590 over all eight tilts: spread 2.6e-9 Ha
/// (−74.0755746796..−74.0755746822). The bar is 1e-8: the spread plus the
/// production-to-fine difference in the flip split (1e-10..4e-10) has room
/// below it, while a run stuck on the wrong side of the split or not
/// converged along the hole mode is not bounded by anything this small on the
/// production grid (there the same molecule spreads 7e-5 Ha across tilts,
/// almost all of it the rotational variance of the molecular grid itself:
/// OH⁻ spreads 9.7e-5 Ha at the same tilts).
#[test]
fn oh_uks_lda_energy_is_orientation_invariant_on_a_fine_grid() {
    let cfg = RhfConfig {
        dft_grid: Some(ferric_dft::grid::AtomicGridConfig::production_sized(
            99, 590,
        )),
        ..lda(1e-10, 1e-9)
    };
    let energies: Vec<f64> = [0.0, 30.0, 71.0]
        .iter()
        .map(|&tilt| {
            let s = sys(tilt);
            let r = solve_uhf(
                &ParallelContext::default(),
                &s.mol,
                &s.prep,
                &s.bounds,
                &cfg,
            )
            .unwrap_or_else(|e| panic!("{tilt}°: {e}"));
            assert!(r.converged, "{tilt}°: not converged");
            println!("{tilt:5.1}° (99x590): E {:.10}", r.energy);
            r.energy
        })
        .collect();
    let spread = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - energies.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(spread < 1e-8, "orientation spread {spread:.2e} Ha");
}

/// Rotate the β hole by `theta` inside the π pair of a converged result: the
/// β occupied π (highest occupied) and the empty π (lowest virtual).
fn rotate_beta_hole(r: &ScfResult, theta: f64) -> (Array2<f64>, Array2<f64>) {
    let mut cb = r.mos_beta.clone().unwrap();
    let (i, a) = (NOCC.1 - 1, NOCC.1);
    let (ci, ca) = (cb.column(i).to_owned(), cb.column(a).to_owned());
    cb.column_mut(i)
        .assign(&(theta.cos() * &ci + theta.sin() * &ca));
    cb.column_mut(a)
        .assign(&(-theta.sin() * &ci + theta.cos() * &ca));
    (r.mos_alpha.clone(), cb)
}

/// Started anywhere inside the basin of the hole-orientation minimum, the
/// run descends to that minimum (the pin hands the soft mode to TRAH); the
/// swap acceptance it replaces exited wherever the hole happened to be.
///
/// The bond is on z, so the grid has C4 symmetry about it and the hole energy
/// E(θ) is a periodic grid artefact. MEASURED (production grid, from the
/// converged default run): starts at θ0 = 0.1, 0.3, 0.5 rad all end at
/// −74.0755553219 (stability lambda_min +4.0e-7, a minimum); starts at 0.7
/// and π/4 end at −74.0755552216, a second local minimum (lambda_min
/// +1.1e-7) 1.0e-7 Ha higher. So the hole-orientation dependence is bounded by
/// the grid anisotropy (1.0e-7 Ha here), and is not removed by any local
/// optimiser; the default guess is made deterministic instead
/// (`uhf::canonical_frontier_basis`, turned by `uhf::GUESS_HOLE_ANGLE`; it
/// ends 7e-10 above the minimum, as the unturned canonical start did). Before
/// the pin the 0.1..0.5 starts spread 9.3e-8 Ha. Bar for the basin: 2e-9
/// (measured spread 0).
#[test]
fn oh_uks_lda_descends_to_the_minimum_from_inside_its_basin() {
    let s = sys(0.0);
    let cfg = lda(1e-10, 1e-9);
    let ctx = ParallelContext::default();
    let r0 = solve_uhf(&ctx, &s.mol, &s.prep, &s.bounds, &cfg).expect("reference solve");
    let mut energies = Vec::new();
    for theta in [0.1f64, 0.3, 0.5] {
        let (ga, gb) = rotate_beta_hole(&r0, theta);
        let r = solve_uhf_with_guess(&ctx, &s.mol, &s.prep, &s.bounds, &cfg, Some((&ga, &gb)))
            .unwrap_or_else(|e| panic!("θ0 = {theta}: {e}"));
        println!(
            "θ0 = {theta:.3}: E {:.10} in {} iters",
            r.energy, r.iterations
        );
        assert!(r.converged);
        energies.push(r.energy);
    }
    let spread = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - energies.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("spread over starting orientations: {spread:.2e} Ha");
    assert!(
        spread < 2e-9,
        "spread {spread:.2e} Ha over starts inside one basin"
    );
    // ... and that common energy is at or below the deterministic default
    // run's (the canonical axis start ends on the symmetric point, 8e-10 above
    // the minimum, inside the stability noise floor).
    assert!(energies[0] <= r0.energy + 1e-12);
}

/// Under RI-J (the CLI default) the held state converges too: the TRAH
/// matvec takes the Fock's own DF-J response
/// (`UhfNewtonInputs::j_response`).
///
/// MEASURED with the exact four-centre δJ in the matvec instead: the
/// DF-vs-exact kernel mismatch put ~2e-5 Ha/rad² of curvature on the
/// hole mode, every TRAH step realised twice its prediction (ρ = 2.00), and
/// the 13° run crept along the mode to `max_iter` (CLI defaults: 5° in 139
/// iterations, 13° unconverged at 200). With the DF-J response: 12 and 8.
#[test]
fn oh_uks_lda_ri_j_converges_at_tilted_orientations() {
    let cfg = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        ..lda(1e-10, 1e-9)
    };
    for tilt in [5.0, 13.0] {
        let s = sys(tilt);
        let r = solve_uhf(
            &ParallelContext::default(),
            &s.mol,
            &s.prep,
            &s.bounds,
            &cfg,
        )
        .unwrap_or_else(|e| panic!("{tilt}° RI-J: {e}"));
        println!(
            "{tilt:5.1}° RI-J: E {:.10} in {} iters",
            r.energy, r.iterations
        );
        assert!(
            r.converged && r.iterations <= 30,
            "{tilt}°: {} iterations",
            r.iterations
        );
    }
}

/// Hole orientation of a bond-on-z ²Π state, folded by the grid's C4v
/// symmetry about z into [0, π/4]: φ = ½ atan2(2Q_xy, Q_xx − Q_yy) of the
/// spin density's second moment about the origin; φ, −φ and φ ± π/2 are
/// grid-equivalent.
fn folded_hole_angle(prep: &PreparedBasis, r: &ScfResult) -> f64 {
    let m = ferric_integrals::oneelectron::second_moment(prep, [0.0; 3]).unwrap();
    let ds = &r.density_alpha - r.density_beta.as_ref().unwrap();
    let tr = |a: &Array2<f64>| (a * &ds).sum();
    let phi = 0.5 * (2.0 * tr(&m[1])).atan2(tr(&m[0]) - tr(&m[3]));
    let f = phi.abs() % std::f64::consts::FRAC_PI_2;
    f.min(std::f64::consts::FRAC_PI_2 - f)
}

/// On the (99,590) grid the OH ²Π UKS/PBE/cc-pVDZ state does not depend on
/// where in its π pair the β hole ends up: two starts that end at different
/// hole orientations agree in energy and in every window orbital energy to
/// within the validation_gw bars. That is what lets
/// `validation_gw` `u_g0w0_uks_pbe_oh_*` compare against PySCF on that grid
/// without caring which ripple state either code lands in (gen_gw.py
/// `UKS_GRID`).
///
/// The two starts: the default (MINAO) guess, whose hole is turned
/// `uhf::GUESS_HOLE_ANGLE` = π/8 off the pivot AO, and the core-Hamiltonian
/// guess (not turned, hole on the x axis). MEASURED (bond on z, 2026-11-09):
/// on (99,590) E −75.64489417899 / −75.64489417893 (|d| 6e-11), folded hole
/// angles 0.393 / 0.008 rad; over 10 starts the spread is 1.4e-10 Ha in E and
/// 1.9e-9 Ha in the orbital energies (scripts/queue/out/oh-grid-2026-11-09/).
/// On (75,110) the same two starts end 1.08e-7 Ha apart (the grid-ripple
/// minima the old reference had to pick between) — mutation check: this
/// test on (75,110) fails. Bars: `TOL_E_SCF` 1e-9 and `TOL_EPS_MF` 3e-8 of
/// validation_gw. The angle check keeps it from passing vacuously (both
/// starts ending on one orientation).
#[test]
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
fn oh_uks_pbe_hole_orientation_is_below_the_gw_bars_on_99x590() {
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.9697\n", 0, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some("pbe".into()),
        // Exact J/K and the flat grid, as validation_gw's OH UKS row.
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        dft_grid: Some(ferric_dft::grid::AtomicGridConfig {
            n_radial: 99,
            n_angular: 590,
            ..Default::default()
        }),
        max_iter: 200,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    };
    let hcore = RhfConfig {
        use_sad_guess: false,
        ..cfg.clone()
    };
    let ctx = ParallelContext::default();
    let runs: Vec<ScfResult> = [("default guess", &cfg), ("core-Hamiltonian guess", &hcore)]
        .into_iter()
        .map(|(name, c)| {
            let r =
                solve_uhf(&ctx, &mol, &prep, &bounds, c).unwrap_or_else(|e| panic!("{name}: {e}"));
            println!(
                "{name}: E {:.11} in {} iters, folded hole angle {:.3} rad",
                r.energy,
                r.iterations,
                folded_hole_angle(&prep, &r)
            );
            r
        })
        .collect();
    let (a, b) = (&runs[0], &runs[1]);
    let d_angle = (folded_hole_angle(&prep, a) - folded_hole_angle(&prep, b)).abs();
    assert!(
        d_angle > 0.1,
        "both starts ended at one hole orientation (folded angles differ by {d_angle:.3} rad): \
         the comparison below would not test the ripple"
    );
    let d_e = (a.energy - b.energy).abs();
    // Window of the GW row (orbitals 2..=7), both spins.
    let window = |r: &ScfResult| {
        let mut v = r.eps_alpha[2..8].to_vec();
        v.extend_from_slice(&r.eps_beta.as_ref().unwrap()[2..8]);
        v
    };
    let d_eps = window(a)
        .iter()
        .zip(window(b))
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    println!("|dE| {d_e:.2e} Ha, max |d eps| {d_eps:.2e} Ha");
    assert!(
        d_e < 1e-9,
        "hole orientations differ by {d_e:.2e} Ha on (99,590)"
    );
    assert!(
        d_eps < 3e-8,
        "orbital energies differ by {d_eps:.2e} Ha between hole orientations on (99,590)"
    );
}
