//! MP2C on counterpoise (ghost-atom) fragments.
//!
//! The A24/S22/S66 benchmarks are all CP-corrected, so every term MP2C is built
//! from has to behave correctly when the molecule carries basis-only `@X`
//! centers. The claim being tested is *not* "ghosts are handled somewhere" but
//! the two specific properties MP2C depends on:
//!
//!  1. **Ghosts add basis, not electrons.** `nocc` is unchanged, `nvir` grows by
//!     the ghost's basis size. Every orbital count in the MP2/dRPA path is
//!     derived from `mol.nelec()` and `obs.nbasis()`, so this is the whole of
//!     the requirement — but it is checked directly rather than assumed.
//!
//!  2. **A far-away ghost is a small perturbation, not a leak.** Correlation
//!     energy must shift by the (real, physical) basis-set superposition error
//!     and nothing more. A term that erroneously treated a ghost as a real
//!     center — e.g. by counting its nuclear charge, or by summing over
//!     `mol.atoms.len()` where it meant electrons — would show up as a large
//!     shift here.
//!
//!  3. **The CP triple gives a finite, sensible interaction energy.**
//!
//! Note on cost: these run in `sto-3g`/`cc-pvdz` on tiny systems deliberately.
//! The point is the ghost bookkeeping, which is basis-size-independent; running
//! aug-cc-pVTZ here would test nothing extra and cost 100x.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::mp2c::{mp2c_energy, mp2c_interaction_energy, Mp2cConfig, Mp2cDispersionRange};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

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

/// Property 1: a ghost center contributes basis functions and zero electrons.
///
/// This is the definition of counterpoise, and it is what makes every
/// downstream orbital count in the MP2/dRPA path correct without any
/// ghost-specific code. Checked at the source (`nelec`, `nbasis`) so a
/// regression in the parser or the basis bridge is caught here rather than as a
/// mysterious energy shift.
#[test]
fn ghost_adds_basis_functions_but_no_electrons() {
    let he = Molecule::parse_xyz("1\nHe\nHe 0 0 0\n", 0, 1).unwrap();
    let he_ghost = Molecule::parse_xyz("2\nHe + ghost He\nHe 0 0 0\n@He 0 0 8.0\n", 0, 1).unwrap();
    assert_eq!(he.nelec(), 2);
    assert_eq!(
        he_ghost.nelec(),
        2,
        "a ghost must contribute zero electrons"
    );
    assert_eq!(
        he.nuclear_repulsion(),
        he_ghost.nuclear_repulsion(),
        "a ghost must contribute zero nuclear charge"
    );

    let bs = basis::bundled("cc-pvdz").unwrap();
    let n_plain = PreparedBasis::new(&he, &bs).unwrap().nbasis();
    let n_ghost = PreparedBasis::new(&he_ghost, &bs).unwrap().nbasis();
    assert_eq!(
        n_ghost,
        2 * n_plain,
        "a ghost He must carry the full He basis: {n_plain} -> {n_ghost}"
    );

    // Consequence for MP2C: nocc unchanged, nvir grows by exactly the ghost basis.
    let nocc = he.nelec() as usize / 2;
    let nocc_g = he_ghost.nelec() as usize / 2;
    assert_eq!(nocc, nocc_g);
    assert_eq!(n_ghost - nocc_g, (n_plain - nocc) + n_plain);
}

/// Property 2: a ghost perturbs MP2C by the physical BSSE and nothing more.
///
/// The scales here are **externally anchored**, not guessed. Local PySCF
/// (`gto.M(atom='He 0 0 0; ghost-He 0 0 R', basis=ccpvdz)`, `mp.MP2`) gives for
/// the He/cc-pVDZ MP2 correlation energy:
///
/// ```text
///     R = 3 Å  :  BSSE = −6.90e-7 Ha
///     R = 8 Å  :  BSSE = +4.2e-17 Ha   (i.e. exactly zero to double precision)
/// ```
///
/// So the ghost distance matters enormously, and **8 Å is the wrong probe** —
/// there is no signal there to detect, and an earlier draft of this test
/// asserted a >1e-8 Ha shift at 8 Å and failed for that reason. The probe is
/// therefore at **3 Å**, where BSSE is real and resolvable.
///
/// The assertions bracket the shift from both sides:
///  * upper: treating the ghost as a *real* He would add a whole correlated
///    pair (~−26 mHa, since He/cc-pVDZ E_corr ≈ −0.0258 Ha), 4 orders of
///    magnitude above BSSE. That is the leak this catches.
///  * lower: the shift must be nonzero at the PySCF-anchored scale, so a ghost
///    whose basis never reaches the correlated virtual space also fails.
#[test]
fn ghost_shifts_mp2c_by_bsse_scale_only() {
    let (mol, obs, dfbs, rhf) = setup("1\nHe\nHe 0 0 0\n", "cc-pvdz", "cc-pvdz-ri");
    let plain = mp2c_energy(&mol, &obs, &dfbs, &rhf, &Mp2cConfig::default()).unwrap();

    let (molg, obsg, dfbsg, rhfg) = setup(
        "2\nHe + ghost He at 3A\nHe 0 0 0\n@He 0 0 3.0\n",
        "cc-pvdz",
        "cc-pvdz-ri",
    );
    let ghosted = mp2c_energy(&molg, &obsg, &dfbsg, &rhfg, &Mp2cConfig::default()).unwrap();

    let d_hf = rhfg.energy - rhf.energy;
    let d_corr = ghosted.e_corr - plain.e_corr;
    let d_mp2 = ghosted.e_mp2 - plain.e_mp2;
    eprintln!(
        "He vs He+@He(3A): dHF {:.3e}  dE_corr(MP2C) {:.3e}  dE_MP2 {:.3e} \
         [PySCF MP2 BSSE reference: -6.90e-7]  dCorrection {:.3e}",
        d_hf,
        d_corr,
        d_mp2,
        ghosted.e_correction - plain.e_correction
    );

    // The MP2 half is directly comparable to the PySCF-anchored number; require
    // the same order of magnitude, not just "small".
    assert!(
        (d_mp2 / -6.90e-7).abs() > 0.2 && (d_mp2 / -6.90e-7).abs() < 5.0,
        "MP2 BSSE at 3 A was {d_mp2:.3e} Ha; PySCF gives -6.90e-7, so this is off by \
         more than the RI/threshold slack that could explain it"
    );
    // Upper bound: a real extra He would add a correlated pair (~-26 mHa).
    assert!(
        d_corr.abs() < 1e-3,
        "ghost moved MP2C correlation by {d_corr:.3e} Ha — far above BSSE scale; \
         the ghost is likely being treated as a real center"
    );
    // Lower bound: the ghost basis must actually be reaching the virtual space.
    assert!(
        d_corr.abs() > 1e-8,
        "ghost moved MP2C correlation by only {d_corr:.3e} Ha — the ghost basis is \
         apparently not entering the correlated space at all"
    );
    // Every MP2C piece must stay finite.
    assert!(ghosted.e_corr.is_finite() && ghosted.e_correction.is_finite());
    assert!(ghosted.components_sum_to_corr().abs() < 1e-15);
}

/// Property 2, tightened: as the ghost is moved further away the BSSE must
/// *decrease monotonically*. A term that mishandled ghosts would generally not
/// have this distance dependence.
#[test]
fn bsse_decreases_with_ghost_distance() {
    let (mol, obs, dfbs, rhf) = setup("1\nHe\nHe 0 0 0\n", "cc-pvdz", "cc-pvdz-ri");
    let plain = mp2c_energy(&mol, &obs, &dfbs, &rhf, &Mp2cConfig::default()).unwrap();

    let mut prev = f64::INFINITY;
    for r_ang in [4.0_f64, 6.0, 10.0] {
        let xyz = format!("2\nHe + ghost\nHe 0 0 0\n@He 0 0 {r_ang}\n");
        let (m, o, d, s) = setup(&xyz, "cc-pvdz", "cc-pvdz-ri");
        let g = mp2c_energy(&m, &o, &d, &s, &Mp2cConfig::default()).unwrap();
        let bsse = (g.e_corr - plain.e_corr).abs();
        eprintln!("ghost at {r_ang} A: |BSSE(corr)| = {bsse:.4e} Ha");
        assert!(
            bsse < prev,
            "BSSE must shrink as the ghost recedes: {bsse:.3e} at {r_ang} A is not \
             below the previous {prev:.3e}"
        );
        prev = bsse;
    }
}

/// Property 3: a genuine CP triple (dimer + two ghosted monomers) gives a
/// finite, sensible interaction energy.
///
/// System: the water–ammonia dimer of A24-01, in a small basis. The reference
/// CCSD(T)/CBS binding is −6.524 kcal/mol; a small-basis MP2C number will not
/// reproduce that, so the assertions here are about *sanity and structure*, not
/// accuracy:
///
///  * every piece finite,
///  * the complex is bound (negative interaction energy) — A24-01 is a strong
///    hydrogen bond, so even a small basis must get the sign right,
///  * magnitude in a physically plausible window (0–30 kcal/mol),
///  * component additivity holds.
///
/// The accuracy question — how close MP2C lands to the CCSD(T)/CBS reference in
/// a converged basis — is a benchmark question, not a unit test, and is
/// deliberately not asserted here.
///
/// # What this test learned (an earlier assertion here was wrong)
///
/// This test originally also asserted that the MP2C correction is *repulsive*,
/// on the reasoning that "coupling screens dispersion, so MP2C binds less".
/// It failed: on A24-01 the measured correction is **−0.163 kcal/mol**
/// (attractive). The decomposition (`probe_a24_01_decomposition` below) shows
/// why, and it is physics, not a bug:
///
/// ```text
///                    dimer      mA_cp      mB_cp    Δ (kcal/mol)
///     uncoupled   -0.59911   -0.30593   -0.29337        +0.118
///     coupled     -0.45799   -0.23175   -0.22616        -0.045
///     correction  +0.14113   +0.07418   +0.06721        -0.163
/// ```
///
/// The *uncoupled* second-order ring term's own interaction energy is
/// **repulsive** for this hydrogen bond, so swapping it for a coupled term that
/// is attractive lowers the total. MP2 does not overbind hydrogen bonds, and
/// MP2C accordingly does not push back on them — the repulsive correction
/// MP2C is known for belongs to *dispersion-bound* complexes. That
/// class-dependent sign is asserted where it actually holds, in
/// `mp2c_a24.rs::a24_cp_triples_run_and_correction_sign_tracks_binding_class`,
/// which checks A24-19/20 (methane–methane, methane–Ar).
#[test]
fn cp_triple_gives_finite_sensible_interaction_energy() {
    // A24-01: water-ammonia. Geometry from testdata/molecules/a24/a24-01_dimer.xyz.
    const DIMER: &str = "7\na24-01 dimer\n\
        O 0.00000000 -0.05786571 -1.47979303\n\
        H 0.00000000 0.82293384 -1.85541474\n\
        H 0.00000000 0.07949567 -0.51934253\n\
        N 0.00000000 0.01436394 1.46454628\n\
        H 0.00000000 -0.98104857 1.65344779\n\
        H -0.81348351 0.39876776 1.92934049\n\
        H 0.81348351 0.39876776 1.92934049\n";
    const MA_CP: &str = "7\na24-01 mA_cp\n\
        O 0.00000000 -0.05786571 -1.47979303\n\
        H 0.00000000 0.82293384 -1.85541474\n\
        H 0.00000000 0.07949567 -0.51934253\n\
        @N 0.00000000 0.01436394 1.46454628\n\
        @H 0.00000000 -0.98104857 1.65344779\n\
        @H -0.81348351 0.39876776 1.92934049\n\
        @H 0.81348351 0.39876776 1.92934049\n";
    const MB_CP: &str = "7\na24-01 mB_cp\n\
        @O 0.00000000 -0.05786571 -1.47979303\n\
        @H 0.00000000 0.82293384 -1.85541474\n\
        @H 0.00000000 0.07949567 -0.51934253\n\
        N 0.00000000 0.01436394 1.46454628\n\
        H 0.00000000 -0.98104857 1.65344779\n\
        H -0.81348351 0.39876776 1.92934049\n\
        H 0.81348351 0.39876776 1.92934049\n";

    let cfg = Mp2cConfig::default();
    let run = |xyz: &str| {
        let (m, o, d, s) = setup(xyz, "cc-pvdz", "cc-pvdz-ri");
        // Every fragment MUST see the same basis dimension — that is what makes
        // it a counterpoise calculation at all. Asserted per-fragment below.
        let n = o.nbasis();
        let r = mp2c_energy(&m, &o, &d, &s, &cfg).unwrap();
        (n, r)
    };
    let (n_d, dimer) = run(DIMER);
    let (n_a, ma) = run(MA_CP);
    let (n_b, mb) = run(MB_CP);
    assert_eq!(
        (n_d, n_a, n_b),
        (n_d, n_d, n_d),
        "all three CP fragments must span the SAME basis — {n_d}/{n_a}/{n_b}"
    );

    let int = mp2c_interaction_energy(&dimer, &ma, &mb);
    let kcal = int.e_int_mp2c_kcal();
    eprintln!(
        "A24-01 (cc-pVDZ, CP): MP2C {:.4} kcal/mol | MP2 {:.4} | HF {:.4} | correction {:.4} \
         [CCSD(T)/CBS ref -6.524]",
        kcal,
        int.e_int_mp2_kcal(),
        int.e_int_hf * ferric_rpa::mp2c::Mp2cInteraction::HARTREE_TO_KCAL,
        int.e_int_correction * ferric_rpa::mp2c::Mp2cInteraction::HARTREE_TO_KCAL
    );

    assert!(kcal.is_finite(), "interaction energy must be finite");
    assert!(
        kcal < 0.0,
        "A24-01 is a hydrogen-bonded complex; MP2C must find it bound, got {kcal:.4} kcal/mol"
    );
    assert!(
        kcal > -30.0,
        "interaction energy {kcal:.4} kcal/mol is far outside the plausible range for a \
         neutral hydrogen-bonded dimer"
    );
    // The correction must be a CORRECTION: same order as, but well below, the
    // MP2 interaction energy itself. No sign is asserted here — see the doc
    // comment; the sign is class-dependent and is tested in mp2c_a24.rs.
    assert!(
        int.e_int_correction.abs() < 0.5 * int.e_int_mp2.abs(),
        "the MP2C correction ({:.4} kcal/mol) must be small against the MP2 interaction \
         energy ({:.4} kcal/mol)",
        int.e_int_correction * ferric_rpa::mp2c::Mp2cInteraction::HARTREE_TO_KCAL,
        int.e_int_mp2_kcal()
    );
    assert!(
        int.components_sum_to_total().abs() < 1e-12,
        "interaction-energy components must sum to the total (residual {:.2e} Ha)",
        int.components_sum_to_total()
    );
    assert!(dimer.drpa_converged && ma.drpa_converged && mb.drpa_converged);
}

/// The CP construction must work on the long-range arm too — the ghost basis
/// enters the erf-attenuated intermediates exactly the same way. Cheap check
/// that the second `Mp2cDispersionRange` variant is not silently broken on
/// ghosted systems (a separate code path builds a second set of intermediates).
#[test]
fn long_range_mp2c_also_runs_on_a_ghosted_fragment() {
    let (m, o, d, s) = setup(
        "2\nHe + ghost He\nHe 0 0 0\n@He 0 0 4.0\n",
        "cc-pvdz",
        "cc-pvdz-ri",
    );
    let r = mp2c_energy(
        &m,
        &o,
        &d,
        &s,
        &Mp2cConfig {
            range: Mp2cDispersionRange::LongRange,
            omega: 0.42,
            ..Default::default()
        },
    )
    .unwrap();
    eprintln!(
        "He+@He LongRange: e_corr {:.8}  correction {:.3e}",
        r.e_corr, r.e_correction
    );
    assert!(r.e_corr.is_finite() && r.e_correction.is_finite());
    assert!(r.components_sum_to_corr().abs() < 1e-15);
}

/// DIAGNOSTIC (not an assertion of correctness): print the per-fragment MP2C
/// decomposition for A24-01 so the sign of the interaction-energy correction
/// can be attributed to its parts.
#[test]
fn probe_a24_01_decomposition() {
    const DIMER: &str = "7\nd\nO 0.00000000 -0.05786571 -1.47979303\nH 0.00000000 0.82293384 -1.85541474\nH 0.00000000 0.07949567 -0.51934253\nN 0.00000000 0.01436394 1.46454628\nH 0.00000000 -0.98104857 1.65344779\nH -0.81348351 0.39876776 1.92934049\nH 0.81348351 0.39876776 1.92934049\n";
    const MA: &str = "7\na\nO 0.00000000 -0.05786571 -1.47979303\nH 0.00000000 0.82293384 -1.85541474\nH 0.00000000 0.07949567 -0.51934253\n@N 0.00000000 0.01436394 1.46454628\n@H 0.00000000 -0.98104857 1.65344779\n@H -0.81348351 0.39876776 1.92934049\n@H 0.81348351 0.39876776 1.92934049\n";
    const MB: &str = "7\nb\n@O 0.00000000 -0.05786571 -1.47979303\n@H 0.00000000 0.82293384 -1.85541474\n@H 0.00000000 0.07949567 -0.51934253\nN 0.00000000 0.01436394 1.46454628\nH 0.00000000 -0.98104857 1.65344779\nH -0.81348351 0.39876776 1.92934049\nH 0.81348351 0.39876776 1.92934049\n";
    let cfg = Mp2cConfig::default();
    let run = |x: &str| {
        let (m, o, d, s) = setup(x, "cc-pvdz", "cc-pvdz-ri");
        (s.energy, mp2c_energy(&m, &o, &d, &s, &cfg).unwrap())
    };
    let (hd, d) = run(DIMER);
    let (ha, a) = run(MA);
    let (hb, b) = run(MB);
    const K: f64 = 627.50947406;
    eprintln!(
        "            {:>14} {:>14} {:>14} {:>14}",
        "dimer", "mA_cp", "mB_cp", "Delta(kcal)"
    );
    eprintln!(
        "HF          {hd:14.8} {ha:14.8} {hb:14.8} {:14.5}",
        (hd - ha - hb) * K
    );
    eprintln!(
        "E_MP2       {:14.8} {:14.8} {:14.8} {:14.5}",
        d.e_mp2,
        a.e_mp2,
        b.e_mp2,
        (d.e_mp2 - a.e_mp2 - b.e_mp2) * K
    );
    eprintln!(
        "uncoupled   {:14.8} {:14.8} {:14.8} {:14.5}",
        d.e_disp_uncoupled,
        a.e_disp_uncoupled,
        b.e_disp_uncoupled,
        (d.e_disp_uncoupled - a.e_disp_uncoupled - b.e_disp_uncoupled) * K
    );
    eprintln!(
        "coupled     {:14.8} {:14.8} {:14.8} {:14.5}",
        d.e_disp_coupled,
        a.e_disp_coupled,
        b.e_disp_coupled,
        (d.e_disp_coupled - a.e_disp_coupled - b.e_disp_coupled) * K
    );
    eprintln!(
        "correction  {:14.8} {:14.8} {:14.8} {:14.5}",
        d.e_correction,
        a.e_correction,
        b.e_correction,
        (d.e_correction - a.e_correction - b.e_correction) * K
    );
}
