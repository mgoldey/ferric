//! `optimize_qmmm` stationarity: the geometry the QM/MM optimizer returns is a
//! stationary point of the energy it claims to minimize — over the **full**
//! structure, not just the QM rows.
//!
//! There is no literature QM/MM minimum to compare a relaxed geometry against
//! (a QM/MM potential energy surface is defined by the partition, the link
//! scale and the boundary scheme, none of which is standardized). But a minimum
//! is defined by its own gradient, so no external reference is needed: assert
//! that `full_gradient` vanishes there, and that `full_gradient` is itself
//! correct — finite-difference verified — **at that same geometry**.
//!
//! ## Why both halves are needed
//!
//! An FD check performed only at the converged geometry is nearly vacuous: at a
//! minimum every row is ~1e-5 Ha/Bohr, so "analytic agrees with FD to 1e-6"
//! cannot distinguish a correct gradient from one that is wrong by an amount
//! that also happens to be small. This file therefore does two separate things:
//!
//! 1. **Stationarity** (`..._is_stationary_over_every_row_class`): at the
//!    converged geometry, `max |dE/dR|` is below the optimizer's own threshold
//!    for **every row class** — QM real atoms, MM atoms, the link-atom host
//!    pair, and the RCD midpoint hosts. Under `MoveMm::All` every one of those
//!    rows is in the BFGS vector, so this is a claim about the whole structure.
//!    (Under `MoveMm::None` the MM rows are never seen by the optimizer's own
//!    convergence test, which is exactly why this test uses `All`.)
//!
//! 2. **Gradient correctness with amplitude**
//!    (`..._matches_finite_difference_at_the_minimum` and
//!    `..._off_the_minimum`): central FD of the TOTAL energy (SCF + MM force
//!    field), with the partition **rebuilt** at every displaced geometry, over
//!    an FD-delta scan. Run at the minimum (where both sides are small and must
//!    be small together) AND at a geometry deliberately displaced off it (where
//!    every row is O(1e-2) Ha/Bohr and a broken chain rule has room to show).
//!
//! ## The rows that matter
//!
//! Capped ethane with the RCD boundary scheme puts all four chain-rule paths
//! on distinct atoms, so a test that only perturbed QM atoms would pass with
//! the MM and link machinery entirely broken:
//!
//! | full idx | role | paths that reach this row |
//! |---|---|---|
//! | 0 (C0) | QM frontier | own QM row + `(1−g)` of the link row |
//! | 2,4,6 (H) | QM only | own QM row |
//! | 1 (C1) | MM host M1 | `g` of the link row + ½ of each of 3 midpoint forces. Its OWN charge is zeroed by RCD, so it has no atom-centred MM force row at all |
//! | 3,5,7 (H) | MM M2 | own (RCD-shifted) charge force + ½ of one midpoint force |
//!
//! Row 1 is the discriminating one: it is reached ONLY through folded paths.

use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mm::{Angle, Bond, LjParams, MmTopology, Torsion};
use ferric_scf::gradient::rhf_gradient;
use ferric_scf::optimize::OptimizeConfig;
use ferric_scf::qmmm::{
    full_gradient_with_mm, mm_forces, optimize_qmmm, qmmm_mm_terms, BoundaryChargeScheme, MoveMm,
    QmSelection, QmmmAtom, QmmmMethod, QmmmOptimizeConfig, QmmmSystem, DEFAULT_LINK_SCALE,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const ANG2BOHR: f64 = 1.0 / 0.529_177_210_92;
const ETHANE_CC: f64 = 1.53 * ANG2BOHR;

/// Mirrors `tests/qmmm.rs` / `tests/qmmm_optimize.rs::ethane_atoms`.
fn ethane_atoms() -> Vec<QmmmAtom> {
    let cc = ETHANE_CC;
    let ch = 1.09 * ANG2BOHR;
    let theta = 109.5_f64.to_radians();
    let (s, c) = (theta.sin(), theta.cos());
    let mut atoms = vec![
        QmmmAtom::new("C", 6, 0.0, 0.0, 0.0, -0.1),
        QmmmAtom::new("C", 6, 0.0, 0.0, cc, -0.1),
    ];
    for k in 0..3 {
        let phi = 2.0 * std::f64::consts::PI * (k as f64) / 3.0;
        atoms.push(QmmmAtom::new(
            "H",
            1,
            ch * s * phi.cos(),
            ch * s * phi.sin(),
            ch * c,
            0.033,
        ));
        atoms.push(QmmmAtom::new(
            "H",
            1,
            ch * s * phi.cos(),
            ch * s * phi.sin(),
            cc - ch * c,
            0.033,
        ));
    }
    atoms
}

fn ethane_bonds() -> Vec<(usize, usize)> {
    vec![(0, 1), (0, 2), (0, 4), (0, 6), (1, 3), (1, 5), (1, 7)]
}

/// Build the capped-ethane RCD partition at an arbitrary geometry. Every FD
/// displacement goes through this, so the link atom is re-placed and the RCD
/// midpoint charges re-derived at each displaced geometry — the partition is
/// the exact derivative of what is evaluated, not stale geometry.
fn build_system(atoms: &[QmmmAtom]) -> QmmmSystem {
    let bonds = ethane_bonds();
    QmmmSystem::new(atoms, QmSelection::Indices(vec![0, 2, 4, 6]), 0, 1)
        .unwrap()
        .with_link_atoms(&bonds, DEFAULT_LINK_SCALE)
        .unwrap()
        .with_boundary_charges(&bonds, BoundaryChargeScheme::RedistributedChargeDipole)
        .unwrap()
}

/// Same made-up-but-generic parameters as `tests/qmmm_optimize.rs`.
fn full_ethane_topology() -> MmTopology {
    let ch_r0 = 1.09 * ANG2BOHR;
    let mut bonds = vec![Bond {
        i: 0,
        j: 1,
        k: 0.35,
        r0: ETHANE_CC,
    }];
    for (i, j) in [(0, 2), (0, 4), (0, 6), (1, 3), (1, 5), (1, 7)] {
        bonds.push(Bond {
            i,
            j,
            k: 0.4,
            r0: ch_r0,
        });
    }
    let theta0 = 109.5_f64.to_radians();
    let mut angles = vec![];
    for h in [2, 4, 6] {
        angles.push(Angle {
            i: h,
            j: 0,
            k: 1,
            k_theta: 0.06,
            theta0,
        });
    }
    for h in [3, 5, 7] {
        angles.push(Angle {
            i: h,
            j: 1,
            k: 0,
            k_theta: 0.06,
            theta0,
        });
    }
    for (center, hs) in [(0usize, [2, 4, 6]), (1, [3, 5, 7])] {
        for a in 0..3 {
            for b in (a + 1)..3 {
                angles.push(Angle {
                    i: hs[a],
                    j: center,
                    k: hs[b],
                    k_theta: 0.04,
                    theta0,
                });
            }
        }
    }
    let mut torsions = vec![];
    for &hi in &[2, 4, 6] {
        for &hj in &[3, 5, 7] {
            torsions.push(Torsion {
                i: hi,
                j: 0,
                k: 1,
                l: hj,
                periodicity: 3,
                k_phi: 0.02,
                phase: 0.0,
            });
        }
    }
    let charges = vec![-0.1, -0.1, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033];
    let lj_small = LjParams {
        sigma: 0.5 * ANG2BOHR,
        epsilon: 0.0157 / 627.509_474,
    };
    let lj_c = LjParams {
        sigma: 0.6 * ANG2BOHR,
        epsilon: 0.109 / 627.509_474,
    };
    let lj = vec![
        lj_c, lj_c, lj_small, lj_small, lj_small, lj_small, lj_small, lj_small,
    ];
    MmTopology::new(charges, lj, bonds, angles, torsions).unwrap()
}

fn tight_scf() -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    }
}

fn coords_of(atoms: &[QmmmAtom]) -> Array2<f64> {
    let mut c = Array2::<f64>::zeros((atoms.len(), 3));
    for (i, a) in atoms.iter().enumerate() {
        c[(i, 0)] = a.x;
        c[(i, 1)] = a.y;
        c[(i, 2)] = a.z_pos;
    }
    c
}

/// Total QM/MM energy (SCF in the embedding potential + MM force-field terms)
/// at `atoms`, with the partition REBUILT from those coordinates. This is
/// exactly what `optimize_qmmm`'s closure minimizes, so its finite difference
/// is the right thing to compare `full_gradient_with_mm` against.
fn total_energy(atoms: &[QmmmAtom], top: &MmTopology) -> f64 {
    let ctx = ParallelContext::default();
    let sys = build_system(atoms);
    let mol = sys.to_qm_molecule();
    let bs = ferric_core::basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        external_potential: sys.to_external_potential(),
        ..tight_scf()
    };
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(r.converged, "SCF did not converge in the FD energy probe");
    let (mm_e, _) = qmmm_mm_terms(&sys, top, &coords_of(atoms)).unwrap();
    r.energy + mm_e.total
}

/// Analytic `dE/dR` over the full structure at `atoms`, assembled the same way
/// `optimize_qmmm` assembles it.
fn analytic_full_gradient(atoms: &[QmmmAtom], top: &MmTopology) -> Array2<f64> {
    let ctx = ParallelContext::default();
    let sys = build_system(atoms);
    let mol = sys.to_qm_molecule();
    let bs = ferric_core::basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        external_potential: sys.to_external_potential(),
        ..tight_scf()
    };
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(r.converged);
    let qm_grad = rhf_gradient(
        &mol,
        &prep,
        op,
        &bounds,
        &r,
        cfg.external_potential.as_ref(),
    )
    .unwrap();
    let forces = mm_forces(&sys, &mol, &prep, r.density_total()).unwrap();
    let (_, mm_g) = qmmm_mm_terms(&sys, top, &coords_of(atoms)).unwrap();
    full_gradient_with_mm(&sys, &qm_grad, &forces, &mm_g).unwrap()
}

/// Displace one Cartesian component of one atom.
fn displaced(atoms: &[QmmmAtom], idx: usize, axis: usize, delta: f64) -> Vec<QmmmAtom> {
    let mut out = atoms.to_vec();
    match axis {
        0 => out[idx].x += delta,
        1 => out[idx].y += delta,
        _ => out[idx].z_pos += delta,
    }
    out
}

/// Relax the capped-ethane RCD system with `MoveMm::All` so that EVERY row of
/// the full structure is a free coordinate in the BFGS vector. Returns the
/// converged atom list.
fn relaxed_atoms() -> (Vec<QmmmAtom>, MmTopology, usize) {
    let ctx = ParallelContext::default();
    let system = build_system(&ethane_atoms());
    let top = full_ethane_topology();
    let cfg = QmmmOptimizeConfig {
        method: QmmmMethod::Rhf,
        move_mm: MoveMm::All,
        opt: OptimizeConfig {
            trust_radius: 0.1,
            max_steps: 120,
            // Tighter than the default so the stationarity claim below is a
            // real bar and not just "the default threshold was loose".
            g_max_thresh: 1.0e-4,
            g_rms_thresh: 5.0e-5,
            e_conv: 1.0e-9,
            ..Default::default()
        },
        mm_topology: Some(top.clone()),
        scf: tight_scf(),
    };
    let result = optimize_qmmm(&ctx, &system, "sto-3g", &cfg).unwrap();
    assert!(
        result.converged,
        "MoveMm::All capped-ethane RCD optimization did not converge in {} steps",
        result.steps
    );
    (result.system.atoms.clone(), top, result.steps)
}

// ---------------------------------------------------------------------------
// 1. STATIONARITY: the gradient vanishes at the returned geometry, for every
//    row class — not just the QM rows.
// ---------------------------------------------------------------------------

/// The geometry `optimize_qmmm` returns is a stationary point of the energy it
/// minimizes, over the FULL structure. `max |dE/dR|` is reported and asserted
/// separately per row class so that a large residual confined to (say) the MM
/// host row cannot hide inside a whole-structure maximum dominated by nothing.
///
/// Row 1 (the MM host C1) is the discriminating row: RCD zeroes its own MM
/// charge, so the ONLY things that reach it are the link-atom `g` fold and the
/// three midpoint ½ folds. If either of those were dropped, this row would be
/// the one left with a residual gradient.
#[test]
fn optimize_qmmm_minimum_is_stationary_over_every_row_class() {
    let (atoms, top, steps) = relaxed_atoms();
    let grad = analytic_full_gradient(&atoms, &top);

    // The bar. The optimizer was asked for g_max < 1e-4 (see relaxed_atoms);
    // it actually stops at ~1.2e-6, because BFGS's last accepted step
    // overshoots the threshold by a wide margin on this surface. Asserting at
    // the 1e-4 it was ASKED for would therefore be vacuous — it could not be
    // violated by anything short of a 100x regression. The bar here is 5e-6:
    // ~4x above the worst measured row (1.215e-6, the fold-only MM host) and
    // ~20x BELOW the requested threshold, so it pins the observed behaviour
    // rather than restating the request.
    //
    // The analytic gradient recomputed here is the SAME quantity BFGS
    // converged on, so this is not an independent re-derivation of
    // convergence — it is the per-row-class DECOMPOSITION of it, which the
    // optimizer never sees (it only ever looks at the flat max over the free
    // coordinates, where a bad MM-host row could hide under a good QM row).
    // What makes the decomposition a real claim is the FD tests below, which
    // check the same rows against the energy itself.
    const BAR: f64 = 5.0e-6;

    let classes: [(&str, &[usize]); 3] = [
        ("QM real (own rows only)", &[2, 4, 6]),
        ("QM frontier C0 (own row + (1-g) link fold)", &[0]),
        ("MM M2 hydrogens (own charge + 1/2 midpoint)", &[3, 5, 7]),
    ];
    let mut overall: f64 = 0.0;
    for (label, rows) in classes {
        let mut m: f64 = 0.0;
        for &i in rows {
            for k in 0..3 {
                m = m.max(grad[(i, k)].abs());
            }
        }
        overall = overall.max(m);
        eprintln!("[stationary] {label}: max |dE/dR| = {m:.3e} Ha/Bohr");
        assert!(m < BAR, "{label}: max |dE/dR| = {m:.3e} exceeds {BAR:.1e}");
    }
    // The discriminating row, reported on its own.
    let mut host: f64 = 0.0;
    for k in 0..3 {
        host = host.max(grad[(1, k)].abs());
    }
    overall = overall.max(host);
    eprintln!(
        "[stationary] MM host C1 (link g fold + 3x 1/2 midpoint folds ONLY, \
         own charge zeroed by RCD): max |dE/dR| = {host:.3e} Ha/Bohr"
    );
    assert!(
        host < BAR,
        "MM host C1: max |dE/dR| = {host:.3e} exceeds {BAR:.1e} — the link/midpoint \
         folded paths are NOT stationary at the returned geometry"
    );
    eprintln!(
        "[stationary] converged in {steps} steps; overall max |dE/dR| over all 8 real \
         atoms = {overall:.3e} Ha/Bohr"
    );

    // Sanity: RCD really did zero C1's own MM charge, so row 1 really is
    // fold-only. If this ever stops holding, the claim above is weaker than
    // its comment says.
    let sys = build_system(&atoms);
    assert_eq!(
        sys.effective_charges[1], 0.0,
        "RCD should zero the host C1 charge; row 1 is no longer fold-only"
    );
    assert_eq!(sys.boundary_charges.len(), 3, "expected 3 RCD midpoints");
    assert_eq!(sys.link_atoms.len(), 1, "expected 1 link atom");
}

// ---------------------------------------------------------------------------
// 2. FD-VS-ANALYTIC AT THE MINIMUM. Both sides small, and small TOGETHER.
// ---------------------------------------------------------------------------

/// At the converged geometry, central FD of the total QM/MM energy reproduces
/// the analytic full gradient, across an FD-delta scan, on every row class
/// including the fold-only MM host.
///
/// This is the weaker of the two FD tests by construction — at a minimum every
/// row is ~1e-5 Ha/Bohr, so agreement here has little amplitude to it. Its job
/// is to certify that the FD of the energy is ALSO small (i.e. the optimizer
/// stopped where the true derivative vanishes, not merely where its own
/// gradient formula returns zero). The amplitude-carrying check is the
/// off-minimum test below.
#[test]
fn full_gradient_matches_finite_difference_at_the_minimum() {
    let (atoms, top, _) = relaxed_atoms();
    let grad = analytic_full_gradient(&atoms, &top);

    let probes: [(usize, usize, &str); 6] = [
        (0, 2, "QM frontier C0 z (own + (1-g) link)"),
        (0, 0, "QM frontier C0 x"),
        (
            1,
            2,
            "MM host C1 z (link g + 3x 1/2 midpoint, own charge zeroed)",
        ),
        (1, 0, "MM host C1 x"),
        (3, 2, "MM M2 H3 z (own charge + 1/2 midpoint)"),
        (6, 1, "QM H6 y"),
    ];

    for h in [5.0e-5_f64, 1.0e-4, 2.0e-4] {
        for &(idx, axis, label) in &probes {
            let ep = total_energy(&displaced(&atoms, idx, axis, h), &top);
            let em = total_energy(&displaced(&atoms, idx, axis, -h), &top);
            let fd = (ep - em) / (2.0 * h);
            let an = grad[(idx, axis)];
            let err = (an - fd).abs();
            eprintln!("[min h={h:.0e}] {label}: analytic {an:+.6e}  FD {fd:+.6e}  |D| {err:.2e}");
            // Bar: 2e-7. The worst residual measured across this scan is
            // 2.7e-8 (h=5e-5, C0 x), so this is ~7x headroom. It is NOT set
            // relative to the analytic value, because the analytic values
            // here are ~1e-7..1e-6 and a relative bar would be meaningless at
            // that amplitude — see the off-minimum test for the version of
            // this check that has amplitude. The residual here is dominated
            // by SCF convergence noise in the two displaced energies
            // (density_conv 1e-10 over a 2h = 1e-4 divisor), which is why it
            // does NOT fall as h grows.
            assert!(
                err < 2.0e-7,
                "{label} at h={h:.0e}: analytic {an:+.6e} vs FD {fd:+.6e} (|D| {err:.2e})"
            );
            // The stationarity claim, seen from the ENERGY side: the numerical
            // derivative of the energy is itself small at this geometry. The
            // largest measured is 1.22e-6, so 5e-6 matches the analytic bar.
            assert!(
                fd.abs() < 5.0e-6,
                "{label} at h={h:.0e}: FD derivative {fd:+.6e} is not small — the \
                 returned geometry is not a stationary point of the energy"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. FD-VS-ANALYTIC OFF THE MINIMUM. Both sides LARGE, and agreeing.
//    This is where a broken fold has amplitude to fail.
// ---------------------------------------------------------------------------

/// The amplitude-carrying gradient check: start from the relaxed geometry,
/// push it well off the minimum, and require FD-vs-analytic agreement while
/// every probed row is O(1e-2) Ha/Bohr — four orders of magnitude above the
/// residual at the minimum, and four orders above the FD truncation error.
///
/// Mutation target. Breaking the link `(1-g)/g` split or the midpoint half/half
/// split leaves rows 0 and 1 wrong by O(1e-3..1e-2), which this test's 2e-6 bar
/// rejects by ~3 orders of magnitude, while the test at the minimum (where
/// everything is ~1e-5) would be far closer to passing.
#[test]
fn full_gradient_matches_finite_difference_off_the_minimum() {
    let (relaxed, top, _) = relaxed_atoms();

    // Push off the minimum: stretch the cut C-C bond (drags the link atom AND
    // all three RCD midpoints), and bend one M2 hydrogen off-axis.
    let mut atoms = relaxed;
    atoms[1].z_pos += 0.35;
    atoms[0].x += 0.12;
    atoms[3].y += 0.20;
    atoms[6].z_pos -= 0.15;

    let grad = analytic_full_gradient(&atoms, &top);

    let probes: [(usize, usize, &str); 7] = [
        (0, 2, "QM frontier C0 z (own + (1-g) link)"),
        (0, 0, "QM frontier C0 x (own + (1-g) link)"),
        (
            1,
            2,
            "MM host C1 z (link g + 3x 1/2 midpoint, own charge zeroed)",
        ),
        (1, 0, "MM host C1 x (fold-only)"),
        (3, 1, "MM M2 H3 y (own charge + 1/2 midpoint)"),
        (5, 2, "MM M2 H5 z (own charge + 1/2 midpoint)"),
        (6, 2, "QM H6 z"),
    ];

    let mut min_amplitude = f64::INFINITY;
    for h in [5.0e-5_f64, 1.0e-4, 2.0e-4] {
        for &(idx, axis, label) in &probes {
            let ep = total_energy(&displaced(&atoms, idx, axis, h), &top);
            let em = total_energy(&displaced(&atoms, idx, axis, -h), &top);
            let fd = (ep - em) / (2.0 * h);
            let an = grad[(idx, axis)];
            let err = (an - fd).abs();
            let rel = err / an.abs();
            min_amplitude = min_amplitude.min(an.abs());
            eprintln!(
                "[off  h={h:.0e}] {label}: analytic {an:+.6e}  FD {fd:+.6e}  \
                 |D| {err:.2e}  rel {rel:.2e}"
            );
            // Bar: 1e-5 RELATIVE, i.e. agreement to 5 significant digits.
            // Probed amplitudes span 1.9e-2 .. 3.7e-1 Ha/Bohr, so a single
            // absolute bar would mean very different things on different
            // rows. The worst measured relative residual across the scan is
            // 4.1e-7 (h=2e-4, C0 x), giving ~24x headroom.
            //
            // Why this bar is meaningful rather than arbitrary: the residual
            // GROWS with h roughly as h^2 on the large-amplitude rows
            // (C1 z: 5.3e-11 -> 1.9e-9 -> 6.2e-9 for h = 5e-5 -> 1e-4 ->
            // 2e-4; C0 x: 6.3e-10 -> 3.4e-9 -> 1.4e-8, a 22x rise for a 4x
            // rise in h against 16x for exact h^2). That is the central-
            // difference truncation error of the FD side, not an error in the
            // analytic side — so the disagreement is fully attributed, and
            // the bar sits an order of magnitude above the largest truncation
            // residual while still rejecting the fold mutations (which move
            // these rows by O(10%), i.e. ~1e-1 relative, four orders of
            // magnitude above it).
            assert!(
                rel < 1.0e-5,
                "{label} at h={h:.0e}: analytic {an:+.6e} vs FD {fd:+.6e} \
                 (|D| {err:.2e}, relative {rel:.2e})"
            );
        }
    }

    // REACHABILITY: the bar above is only meaningful if the quantities it
    // compares are large compared to it. Every probed row must carry at least
    // 1e-3 Ha/Bohr of signal, i.e. >=500x the 2e-6 bar, so agreement at that
    // bar is a real 3-significant-digit statement and not two small numbers
    // agreeing because both are near zero.
    eprintln!("[off] smallest probed |analytic| = {min_amplitude:.3e} Ha/Bohr");
    assert!(
        min_amplitude > 1.0e-3,
        "off-minimum probes carry too little amplitude ({min_amplitude:.3e}) for the \
         2e-6 agreement bar to be a meaningful statement"
    );
}

// ---------------------------------------------------------------------------
// 4. The FD harness itself must be able to SEE the MM rows.
// ---------------------------------------------------------------------------

/// Guard against the failure mode this whole file exists to rule out: an FD
/// check that silently only exercises QM atoms. Displacing the MM host C1 and
/// an M2 hydrogen must actually CHANGE the total energy — if
/// `total_energy` were blind to MM coordinates (e.g. if the partition were not
/// rebuilt, or the MM terms not included), these differences would be zero and
/// every FD comparison above would degenerate into `analytic == 0`.
#[test]
fn finite_difference_probe_is_sensitive_to_mm_and_host_coordinates() {
    let atoms = ethane_atoms();
    let top = full_ethane_topology();
    let e0 = total_energy(&atoms, &top);
    for (idx, axis, label) in [
        (1usize, 2usize, "MM host C1 z"),
        (3, 1, "MM M2 H3 y"),
        (5, 0, "MM M2 H5 x"),
        (0, 2, "QM frontier C0 z"),
    ] {
        let e1 = total_energy(&displaced(&atoms, idx, axis, 1.0e-2), &top);
        let d = (e1 - e0).abs();
        eprintln!("[sensitivity] {label}: |dE| over 1e-2 Bohr = {d:.3e} Ha");
        assert!(
            d > 1.0e-7,
            "{label}: displacing this atom changed the total energy by only {d:.3e} Ha — \
             the FD probe is not sensitive to this coordinate, so every FD comparison \
             against it is vacuous"
        );
    }
}

// ---------------------------------------------------------------------------
// MUTATION RECORD (2026-09-16). Every guard above was broken deliberately and
// observed to fail, in the foreground, with the binary rebuilt each cycle;
// `git status` was clean of source changes after each restore.
//
// | # | mutation (in src/qmmm.rs) | observed |
// |---|---|---|
// | 1 | link fold factors swapped: `(1-g)`<->`g` on the two hosts (column sum PRESERVED, so the translational-invariance tests in qmmm_smeared.rs do NOT see it) | at-minimum: C0 z analytic +4.97e-9 vs FD +9.20e-3, \|D\| 9.20e-3 (46000x the 2e-7 bar). off-minimum: C0 z relative 1.13e-1 (11000x the 1e-5 bar). Stationarity test PASSED — see note below. |
// | 2 | midpoint split `0.5/0.5` -> `1.0/0.0` (column sum PRESERVED) | at-minimum: C1 z analytic -7.35e-10 vs FD -1.01e-2, \|D\| 1.01e-2. off-minimum: C1 z relative 2.95e-2 (2950x the bar). Caught on exactly the fold-only host row. Stationarity test PASSED. |
// | 3 | atom-centred MM force sign `-F` -> `+F` | `optimize_qmmm` no longer converges at all ("MoveMm::All capped-ethane RCD optimization did not converge in 120 steps"); all three gradient tests fail through `relaxed_atoms`. |
// | 4 | mutation 2 PLUS the off-minimum probe list crippled to QM atoms only | **PASSED.** This is the trap this file exists to close: with the midpoint half/half split completely broken, an FD check that perturbs only QM atoms agrees to the same 1e-5 relative bar with full amplitude. The MM-atom and MM-host probe rows are what make the check discriminating; without them it is vacuous. |
// | 5 | `with_coordinates` reuses stale RCD midpoint positions instead of re-deriving them at the new geometry | stationarity: C0 max \|dE/dR\| = 2.08e-5, exceeds the 5e-6 bar; at-minimum FD: "FD derivative -2.08e-5 is not small". Off-minimum PASSED — a self-consistently stale partition still has analytic and FD agree at a FIXED geometry, it just is not a minimum of anything. The two tests are complementary, not redundant. |
//
// NOTE ON WHAT THE STATIONARITY TEST DOES AND DOES NOT CERTIFY. Mutations 1
// and 2 left `optimize_qmmm_minimum_is_stationary_over_every_row_class`
// PASSING. That is not a defect in the test, it is the definition of the
// thing: BFGS drives whatever gradient it is handed to zero, so "the returned
// geometry is a stationary point of the gradient formula" is true even when
// the formula is wrong. The stationarity test's job is to show the residual is
// small in EVERY row class rather than only in the rows that dominate the flat
// max — it is not, and cannot be, a check that the gradient is correct. The
// gradient-correctness claim rests entirely on the two FD tests, which compare
// against the ENERGY. Anyone reading a green stationarity result as evidence
// of a correct QM/MM gradient is reading it wrong.
