//! LinLCCD(hh) validation ladder.
//!
//! Method: Carter-Fenk, J. Phys. Chem. A 2025, 129, 7251-7260 (papers/linccd.pdf).
//! Design + rationale for each rung: docs/superpowers/specs/2026-07-26-linlccd-hh-design.md
//!
//! Rung 1 (this file, `hh_ladder_off_reproduces_rimp2`) is the highest-value test: with
//! the hh-ladder term switched off, LinLCCD(hh) must reduce EXACTLY to RI-MP2. That pins
//! the driver terms, integral blocks, denominators, and energy expression against
//! already-validated code, leaving a sign/factor error on the one new contraction as the
//! only thing the later rungs can be detecting.

use ferric_cc::linlccd::{linlccd, LadderVariant};
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2_spin_components, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn mol_path(name: &str) -> String {
    format!("{}/../../testdata/molecules/{}", env!("CARGO_MANIFEST_DIR"), name)
}

/// Converge RHF and return everything the correlated methods need.
fn setup(
    xyz: &str,
    obs_name: &str,
    aux_name: &str,
) -> (Molecule, PreparedBasis, PreparedBasis, ferric_scf::result::ScfResult) {
    let mol = Molecule::load_xyz(&mol_path(xyz)).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig { energy_conv: 1e-9, ..Default::default() },
    )
    .unwrap();
    (mol, obs, dfbs, rhf)
}

/// RUNG 1 — structural equivalence.
///
/// LinLCCD(hh) with the hh ladder disabled keeps only the driver terms, which ARE the
/// MP2 amplitude equations (paper eq. 17-18). So it must reproduce RI-MP2 to numerical
/// noise. Both sides use the same RI metric and operator, so there is no RI error floor
/// between them -- this is a tight comparison, not a loose one.
#[test]
fn hh_ladder_off_reproduces_rimp2() {
    let (mol, obs, dfbs, rhf) = setup("water.xyz", "cc-pvdz", "cc-pvdz-ri");
    let op = Operator::coulomb();

    let e_mp2 = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default())
        .unwrap()
        .0
        .e_total;

    let e_lin = linlccd(
        &mol,
        &obs,
        &dfbs,
        op,
        &rhf,
        &CcConfig { energy_conv: 1e-11, max_iter: 100, ..Default::default() },
        LadderVariant::DriversOnly,
    )
    .unwrap()
    .correlation_energy;

    eprintln!("E(RI-MP2)            = {e_mp2:.12}");
    eprintln!("E(LinLCCD drivers)   = {e_lin:.12}");
    eprintln!("difference           = {:+.3e}", e_lin - e_mp2);
    assert!(
        (e_lin - e_mp2).abs() < 1e-9,
        "drivers-only LinLCCD must equal RI-MP2; got {e_lin:.12} vs {e_mp2:.12} \
         (diff {:+.3e})",
        e_lin - e_mp2
    );
}

/// RUNG 1b — the hh ladder actually does something.
///
/// Guards against the failure mode where the contraction is silently zero (wrong slice,
/// wrong axis labels) and rung 1 passes for the wrong reason. The hh ladder REDUCES the
/// magnitude of the correlation energy: it widens the effective gap (paper eq. 15), so
/// |E_hh| < |E_MP2|.
#[test]
fn hh_ladder_is_nonzero_and_reduces_correlation() {
    let (mol, obs, dfbs, rhf) = setup("water.xyz", "cc-pvdz", "cc-pvdz-ri");
    let op = Operator::coulomb();
    let cfg = CcConfig { energy_conv: 1e-11, max_iter: 100, ..Default::default() };

    let e_drivers = linlccd(&mol, &obs, &dfbs, op, &rhf, &cfg, LadderVariant::DriversOnly)
        .unwrap()
        .correlation_energy;
    let e_hh = linlccd(&mol, &obs, &dfbs, op, &rhf, &cfg, LadderVariant::Hh)
        .unwrap()
        .correlation_energy;

    eprintln!("E(drivers only) = {e_drivers:.12}");
    eprintln!("E(LinLCCD(hh))  = {e_hh:.12}");
    eprintln!("hh contribution = {:+.6e}", e_hh - e_drivers);

    assert!(
        (e_hh - e_drivers).abs() > 1e-6,
        "hh ladder contributed nothing ({:+.3e}) -- contraction is silently zero",
        e_hh - e_drivers
    );
    assert!(
        e_hh.abs() < e_drivers.abs(),
        "hh dressing widens the gap, so |E_hh| ({:.10}) must be < |E_MP2| ({:.10})",
        e_hh.abs(),
        e_drivers.abs()
    );
}

/// RUNG 3 — size consistency.
///
/// The paper proves E(A...B) = E(A) + E(B) analytically for LinLCCD(hh) (this is the
/// property LinCCD/CID LACK, and recovering it is a headline claim). An exact identity,
/// so the tolerance reflects only SCF/amplitude convergence, not method error.
/// A wrong factor on the hh term breaks this.
#[test]
fn size_consistent_on_separated_water_dimer() {
    let op = Operator::coulomb();
    let cfg = CcConfig { energy_conv: 1e-11, max_iter: 200, ..Default::default() };

    let monomer = Molecule::load_xyz(&mol_path("water.xyz")).unwrap();

    // Two copies displaced far enough that the interaction is numerically dead.
    let mut dimer = monomer.clone();
    const SEP: f64 = 1000.0; // Bohr
    for atom in &monomer.atoms {
        let mut far = atom.clone();
        far.x += SEP;
        dimer.atoms.push(far);
    }

    let run = |m: &Molecule| -> f64 {
        let obs = PreparedBasis::new(m, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(m, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            m,
            &obs,
            op,
            &bounds,
            &RhfConfig { energy_conv: 1e-10, ..Default::default() },
        )
        .unwrap();
        linlccd(m, &obs, &dfbs, op, &rhf, &cfg, LadderVariant::Hh).unwrap().correlation_energy
    };

    let e_mono = run(&monomer);
    let e_dimer = run(&dimer);
    let defect = e_dimer - 2.0 * e_mono;

    eprintln!("E(monomer)     = {e_mono:.12}");
    eprintln!("E(dimer, {SEP} a0) = {e_dimer:.12}");
    eprintln!("2*E(mono)      = {:.12}", 2.0 * e_mono);
    eprintln!("size-consistency defect = {defect:+.3e}");
    assert!(
        defect.abs() < 1e-7,
        "LinLCCD(hh) must be size-consistent; defect {defect:+.3e} is too large"
    );
}

/// RUNG 2 — H2/STO-3G regularity SMOKE TEST.
///
/// SCOPE: H2 has ONE occupied spatial orbital, so no ring/crossed-ring exchange pathway
/// exists. This exercises the DENOMINATOR-REGULARIZATION mechanism only -- as R grows the
/// HOMO-LUMO gap closes and linearized CC has no quadratic T2^2 term to counteract it;
/// the hh dressing keeps amplitudes finite. It does NOT test the paper's ring-exchange
/// diagnosis, which needs multiple occupied orbitals (see the H6 rung).
///
/// RANGE: capped at 11 A. Beyond ~12 A the RHF *reference* stops converging for stretched
/// H2 (measured: `converged=false` at the 200-iteration cap, E_scf jumping -0.570 ->
/// -0.203 with inverted orbital energies) -- the textbook RHF instability toward UHF at
/// bond breaking, not a LinLCCD failure. `solve_rhf` returns Ok with `converged: false`
/// rather than erroring, so the assertion below is load-bearing: without it an
/// unconverged reference silently propagates into the correlated method. The paper's
/// large-R curves use a UHF/ROHF-based reference, which ferric cannot yet feed to
/// LinLCCD (open-shell needs semi-canonicalization -- see the design doc).
#[test]
fn h2_dissociation_stays_regular() {
    let op = Operator::coulomb();
    let cfg = CcConfig { energy_conv: 1e-10, max_iter: 300, ..Default::default() };

    let mut prev = 0.0f64;
    for &r_ang in &[0.74_f64, 1.5, 3.0, 6.0, 9.0, 11.0] {
        let xyz = format!("2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 {r_ang}\n");
        let mol = Molecule::parse_xyz(&xyz, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig { energy_conv: 1e-10, max_iter: 200, ..Default::default() },
        )
        .unwrap();
        assert!(
            rhf.converged,
            "RHF reference did not converge at R = {r_ang} A (E_scf = {:.6}); the \
             correlated energy below would be meaningless",
            rhf.energy
        );

        let e = linlccd(&mol, &obs, &dfbs, op, &rhf, &cfg, LadderVariant::Hh)
            .unwrap()
            .correlation_energy;
        eprintln!("R = {r_ang:5.2} A   E_corr = {e:.10}");

        assert!(e.is_finite(), "LinLCCD(hh) diverged at R = {r_ang} A");
        assert!(e < 0.0, "correlation energy must be negative; got {e:.10} at R = {r_ang} A");
        assert!(
            e.abs() < 1.0,
            "correlation energy {e:.10} at R = {r_ang} A is unphysically large \
             (near-singular amplitudes)"
        );
        // Monotonic: correlation grows in magnitude as the gap closes.
        assert!(
            e <= prev,
            "E_corr not monotonic: {e:.10} at R = {r_ang} A vs {prev:.10} previously"
        );
        prev = e;
    }
}

/// RUNG 2b — the regularization actually bites.
///
/// The point of LinLCCD(hh) is that it stays bounded where MP2 does not. As the H2 gap
/// closes, MP2's correlation energy grows without bound (measured: -0.67 Ha at 6 A,
/// -1.09 Ha at 9 A -- already past the total SCF energy, i.e. nonsense). LinLCCD(hh)
/// must remain far smaller in magnitude. This is the quantitative content of the
/// "naturally regular" claim on a system this cheap.
#[test]
fn hh_dressing_bounds_amplitudes_where_mp2_blows_up() {
    let op = Operator::coulomb();
    let cfg = CcConfig { energy_conv: 1e-10, max_iter: 300, ..Default::default() };

    for &r_ang in &[6.0_f64, 9.0, 11.0] {
        let xyz = format!("2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 {r_ang}\n");
        let mol = Molecule::parse_xyz(&xyz, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig { energy_conv: 1e-10, max_iter: 200, ..Default::default() },
        )
        .unwrap();
        assert!(rhf.converged, "RHF reference did not converge at R = {r_ang} A");

        let e_mp2 = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default())
            .unwrap()
            .0
            .e_total;
        let e_hh = linlccd(&mol, &obs, &dfbs, op, &rhf, &cfg, LadderVariant::Hh)
            .unwrap()
            .correlation_energy;

        eprintln!(
            "R = {r_ang:5.2} A   E_MP2 = {e_mp2:12.6}   E_LinLCCD(hh) = {e_hh:12.6}   \
             ratio = {:.3}",
            e_hh / e_mp2
        );
        assert!(
            e_hh.abs() < e_mp2.abs(),
            "hh dressing must bound the amplitudes: |E_hh| {:.6} >= |E_MP2| {:.6} at R = {r_ang} A",
            e_hh.abs(),
            e_mp2.abs()
        );
    }
}

/// Regular hexagonal H6 ring; `r_ang` is both the circumradius and the
/// nearest-neighbour distance. The paper's strong-correlation probe (Fig. 2c).
fn h6_ring(r_ang: f64) -> String {
    let mut s = String::from("6\n\n");
    for k in 0..6 {
        let th = std::f64::consts::PI / 3.0 * (k as f64);
        s.push_str(&format!("H {:.10} {:.10} 0.0\n", r_ang * th.cos(), r_ang * th.sin()));
    }
    s
}

/// RUNG 5 — H6/cc-pVDZ, the LOAD-BEARING robustness test.
///
/// Unlike H2, H6 has multiple occupied orbitals, so the ring/crossed-ring exchange
/// pathway the paper identifies as the cause of LinCCD's divergence is actually active.
/// This is the system the paper calls "prototypical of strongly correlated systems in
/// chemistry ... reminiscent of the Hubbard model".
///
/// Measured here across R = 1-6 A (all with a converged RHF reference):
///   * MP2 runs away:      -0.111 -> -0.714 Ha as the gap closes (0.64 -> 0.074)
///   * CCD FAILS: returns a POSITIVE correlation energy (+0.104) at R = 3.0 A, then
///     stops converging entirely by R = 6.0 A
///   * LinLCCD(hh)/full:   smooth, bounded, monotonic over the whole range
///
/// The CCD failure is what makes this discriminating: it is a qualitative breakdown that
/// LinLCCD does not share, on the exact system the paper uses to make the claim.
#[test]
fn h6_ring_stays_robust_where_ccd_fails() {
    let op = Operator::coulomb();
    let cfg = CcConfig { energy_conv: 1e-10, max_iter: 200, ..Default::default() };

    let mut prev_hh = 0.0f64;
    let mut prev_mp2 = 0.0f64;
    for &r in &[1.0_f64, 2.0, 3.0, 4.0, 6.0] {
        let mol = Molecule::parse_xyz(&h6_ring(r), 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            op,
            &bounds,
            &RhfConfig { energy_conv: 1e-10, max_iter: 200, ..Default::default() },
        )
        .unwrap();
        assert!(rhf.converged, "RHF reference did not converge for H6 at R = {r} A");

        let e_mp2 = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default())
            .unwrap()
            .0
            .e_total;
        let e_hh = linlccd(&mol, &obs, &dfbs, op, &rhf, &cfg, LadderVariant::Hh)
            .unwrap()
            .correlation_energy;

        eprintln!("R = {r:4.1} A   E_MP2 = {e_mp2:12.8}   E_LinLCCD(hh) = {e_hh:12.8}");

        // Physical: correlation energy is negative and bounded.
        assert!(e_hh.is_finite(), "LinLCCD(hh) diverged on H6 at R = {r} A");
        assert!(
            e_hh < 0.0,
            "LinLCCD(hh) gave a POSITIVE correlation energy {e_hh:.8} at R = {r} A -- \
             this is the qualitative failure mode CCD exhibits here"
        );
        // Monotonic: correlation grows in magnitude as the ring expands.
        assert!(
            e_hh <= prev_hh,
            "LinLCCD(hh) not monotonic: {e_hh:.8} at R = {r} A vs {prev_hh:.8} previously"
        );
        // Regular: bounded well inside MP2's runaway.
        assert!(
            e_hh.abs() < e_mp2.abs(),
            "LinLCCD(hh) must stay bounded relative to MP2 at R = {r} A: \
             |{e_hh:.8}| >= |{e_mp2:.8}|"
        );
        prev_hh = e_hh;
        prev_mp2 = e_mp2;
    }

    // MP2 really does run away over this range -- documents WHY the bound above is
    // meaningful rather than vacuous.
    assert!(
        prev_mp2 < -0.6,
        "expected MP2 to run away on stretched H6 (got {prev_mp2:.6}); if it no longer \
         does, the boundedness assertions above have lost their teeth"
    );
}
