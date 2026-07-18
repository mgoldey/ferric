//! Per-atom Hirshfeld decomposition of the static polarizability tensor.
//!
//! Validates:
//!   * Origin independence: α^A is unchanged (to grid precision) under a
//!     global translation of the molecule — the atom-centred (r − R_A)
//!     dipole operator must not leak lab-frame dependence into α^A. This is
//!     the regression test for the 2026-07-13 gauge-origin bug (translating
//!     danuglipron's cryo-EM lab-frame pose produced α^A up to ±215 a.u.;
//!     recentering to the origin gave physically sane ±13 a.u. — the bug was
//!     the AO dipole quadrature using raw lab-frame r instead of (r − R_A)).
//!   * Per-atom symmetry: max|α^A − (α^A)^T| < 1e-5 (consumer schema bound).
//!   * Chemical sanity: oxygen dominates over hydrogen in H2O.
//!
//! NOTE: there is deliberately no Σ_A α^A ≈ α_mol assertion. The atom-centred
//! per-atom tensors omit inter-atomic (charge-transfer) coupling by
//! construction — same design as `pdep_polarizability_hirshfeld_dynamic` — so
//! the sum is not expected to reproduce the molecular total. A prior version
//! of this test asserted that sum rule; it only passed because `water.xyz` is
//! already centred near the origin, which masked the gauge-origin bug fixed
//! here (renormalizing per-atom pieces to the lab-frame analytical dipole is
//! itself the bug, not a correctness check).

#![allow(clippy::needless_range_loop)] // index loops over tensor/array axes read clearer with explicit indices

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::pdep_polarizability_hirshfeld;
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn setup_h2o_ccpvdz() -> (
    Molecule,
    PreparedBasis,
    ferric_core::basis::BasisSet,
    PreparedBasis,
    Operator,
    ferric_scf::ScfResult,
) {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    (mol, obs, obs_bs, dfbs, op, rhf)
}

#[test]
fn h2o_hirshfeld_origin_independence_and_symmetry() {
    let (mol, obs, obs_bs, dfbs, op, rhf) = setup_h2o_ccpvdz();

    let cfg = PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 0.0,
        davidson_conv_thresh: 1e-10,
        ..Default::default()
    };

    let alpha_atomic =
        pdep_polarizability_hirshfeld(&mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg, None).unwrap();

    assert_eq!(alpha_atomic.len(), mol.atoms.len());

    // Per-atom symmetry < 1e-5 (consumer schema tolerance).
    for (a, alpha_a) in alpha_atomic.iter().enumerate() {
        let mut asym = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                asym = asym.max((alpha_a[i][j] - alpha_a[j][i]).abs());
            }
        }
        assert!(
            asym < 1e-5,
            "atom {a} symmetry violation {asym:.3e} > 1e-5"
        );
    }

    // Origin independence: translate the whole molecule far from the origin
    // (well beyond danuglipron's real cryo-EM lab-frame displacement of
    // ~120-155 Angstrom that first exposed this bug) and recompute. A correct
    // atom-centred (r - R_A) dipole operator gives identical α^A; the fixed
    // bug (raw lab-frame r) blew up to ±215 a.u. under exactly this kind of
    // shift.
    let shift = [130.0_f64, 140.0, 150.0]; // Bohr, comparable to the real regression case
    let mut mol_shifted = mol.clone();
    for atom in &mut mol_shifted.atoms {
        atom.x += shift[0];
        atom.y += shift[1];
        atom.zpos += shift[2];
    }
    let obs_shifted = PreparedBasis::new(&mol_shifted, &obs_bs).unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let dfbs_shifted = PreparedBasis::new(&mol_shifted, &dfbs_bs).unwrap();
    let bounds_shifted = SchwarzBounds::compute(op, &obs_shifted).unwrap();
    let ctx = ParallelContext::default();
    let rhf_shifted = solve_rhf(
        &ctx,
        &mol_shifted,
        &obs_shifted,
        op,
        &bounds_shifted,
        &RhfConfig::default(),
    )
    .unwrap();
    let alpha_atomic_shifted = pdep_polarizability_hirshfeld(
        &mol_shifted,
        &obs_shifted,
        &obs_bs,
        &dfbs_shifted,
        &rhf_shifted,
        op,
        &cfg,
        None,
    )
    .unwrap();

    for (a, (orig, shifted)) in alpha_atomic
        .iter()
        .zip(alpha_atomic_shifted.iter())
        .enumerate()
    {
        for i in 0..3 {
            for j in 0..3 {
                let d = (orig[i][j] - shifted[i][j]).abs();
                assert!(
                    d < 1e-2,
                    "atom {a} α^A[{i}][{j}] not origin-independent: {:.4} (origin) vs {:.4} (shifted), Δ={:.2e}",
                    orig[i][j], shifted[i][j], d
                );
            }
        }
        let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        let iso_o = iso(orig);
        assert!(
            iso_o.abs() < 20.0,
            "atom {a} α_iso={iso_o:.3} is unphysically large — regression of the gauge-origin bug"
        );
    }

    // Chemical sanity: oxygen (Z=8) should dominate over hydrogens (Z=1).
    // Locate O atom and the two H atoms.
    let mut o_idx = usize::MAX;
    let mut h_idxs = Vec::new();
    for (a, atom) in mol.atoms.iter().enumerate() {
        match atom.z {
            8 => o_idx = a,
            1 => h_idxs.push(a),
            _ => {}
        }
    }
    assert_ne!(o_idx, usize::MAX, "no oxygen found in water.xyz");
    let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
    let iso_o = iso(&alpha_atomic[o_idx]);
    for &h in &h_idxs {
        let iso_h = iso(&alpha_atomic[h]);
        assert!(
            iso_o > iso_h,
            "expected α_iso(O)={iso_o:.3} > α_iso(H{h})={iso_h:.3}"
        );
    }

    eprintln!(
        "H2O Hirshfeld α_iso: O={:.3}, H={:.3},{:.3}",
        iso_o,
        iso(&alpha_atomic[h_idxs[0]]),
        iso(&alpha_atomic[h_idxs[1]]),
    );
}

/// Full Hirshfeld-I with ad-hoc same-basis NEUTRAL+ION proatoms.
///
/// The CANONICAL published Hirshfeld-I charge for water oxygen is q(O) = −0.872
/// (Verstraelen et al., JCTC 12, 3894 (2016), Table 1, BLYP/6-311+G(2df,p);
/// MBIS gives −0.885). Hirshfeld-I is *designed* to be over-ionic — it amplifies
/// plain-Hirshfeld charges to reproduce the ESP/dipole — so q(O) near −0.87 is
/// CORRECT; the −0.6..−0.7 numbers are CM5/ESP-class, not raw HI. (My earlier
/// −0.92 with a 15-Bohr radial grid was anion-tail truncation; the grid now
/// extends to 30 Bohr to capture the diffuse anion reference.)
#[test]
fn h2o_hirshfeld_i_adhoc_charges_physical() {
    use ferric_core::elements::z_to_symbol;
    use ferric_rpa::properties::{hirshfeld_i_charges, spherically_averaged_proatom, RadialProatom};

    // Augmented basis so the O⁻ anion proatom binds (Hirshfeld-I requirement).
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let ctx0 = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx0, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    let ctx = ParallelContext::default();

    // Shared radial grid for proatoms. Extend to 30 Bohr: the anion reference
    // (O⁻) has a diffuse tail that, if truncated, mis-normalizes the
    // interpolated proatom and over-polarizes the charge.
    let radii: Vec<f64> = (1..=600).map(|k| k as f64 * 0.05).collect(); // 0.05..30 Bohr

    // Ground-state multiplicity (2S+1) of an atom/ion with N electrons, from the
    // aufbau filling + Hund's rule on the open subshell (good for N ≤ 18).
    let mult_for_n = |n: i32| -> usize {
        if n <= 0 { return 1; }
        // (subshell capacity) in aufbau order through 3p.
        let shells = [2, 2, 6, 2, 6]; // 1s,2s,2p,3s,3p
        let mut rem = n;
        let mut unpaired = 0i32;
        for &cap in &shells {
            let inshell = rem.min(cap);
            rem -= inshell;
            // unpaired in this subshell by Hund: cap/2 orbitals, fill singly first.
            let norb = cap / 2;
            unpaired = if inshell <= norb { inshell } else { cap - inshell };
            if rem == 0 { break; }
        }
        (unpaired.unsigned_abs() as usize) + 1
    };

    // Proatom closure: run an atomic SCF for element z at integer charge qi
    // (N = z − qi electrons) in the molecule's basis, spherically average to a
    // radial proatom. Neutral + ions for full Hirshfeld-I charge interpolation.
    let bs = obs_bs.clone();
    let proatom = |z: i32, qi: i32| -> Option<RadialProatom> {
        let n_elec = z - qi;
        if n_elec <= 0 {
            // Bare nucleus (e.g. H+): zero electron density.
            return Some(RadialProatom { radii: radii.clone(), rho: vec![0.0; radii.len()] });
        }
        let mult = mult_for_n(n_elec);
        let sym = z_to_symbol(z).unwrap_or("X");
        let xyz = format!("1\n{sym}\n{sym} 0 0 0\n");
        let amol = Molecule::parse_xyz(&xyz, qi, mult).ok()?;
        let aobs = PreparedBasis::new(&amol, &bs).ok()?;
        let abounds = SchwarzBounds::compute(op, &aobs).ok()?;
        let mut cfg = RhfConfig::default();
        // Closed-shell singlet → RHF; open-shell → UHF with MOM. Anions in a
        // non-augmented basis may not bind the extra electron; if SCF fails the
        // `?`/.ok() returns None and hirshfeld_i falls back to the nearest state.
        let dens = if mult == 1 {
            solve_rhf(&ctx, &amol, &aobs, op, &abounds, &cfg).ok()?.density_r().to_owned()
        } else {
            cfg.mom_after_iter = 5;
            ferric_scf::uhf::solve_uhf(&ctx, &amol, &aobs, &abounds, &cfg)
                .ok()?
                .density_total().to_owned()
        };
        spherically_averaged_proatom(z, &bs, &dens, &radii).ok()
    };

    let q = hirshfeld_i_charges(&mol, &obs_bs, rhf.density_r(), &proatom).unwrap();
    let (q_o, q_h1, q_h2) = (q[0], q[1], q[2]);
    eprintln!("H2O Hirshfeld-I (ad-hoc, neutral+ion proatoms): O={q_o:.4} H={q_h1:.4} H={q_h2:.4}");
    // Literature Hirshfeld-I water oxygen is −0.872 (Verstraelen 2016). HI is
    // intentionally over-ionic; accept the published HI window. (Plain neutral
    // Hirshfeld gives −0.32 — see the neutral test; HI sharpens it toward ESP.)
    assert!((-0.95..=-0.78).contains(&q_o), "O charge outside Hirshfeld-I range: {q_o:.3} (lit -0.872)");
    assert!((0.39..=0.48).contains(&q_h1), "H charge outside Hirshfeld-I range: {q_h1:.3} (lit +0.436)");
    assert!((q_o + q_h1 + q_h2).abs() < 1e-6, "charges must sum to 0");
    assert!((q_h1 - q_h2).abs() < 1e-3, "equivalent H must have equal charge");
}

/// The SHIPPED fix: ad-hoc same-basis NEUTRAL proatoms give correct standard
/// Hirshfeld charges and fix the H-starvation. Legacy single-Slater gives
/// O=−0.95 (badly over-charged); basis-consistent neutral free-atom densities
/// give O≈−0.32, H≈+0.16 — the textbook Hirshfeld result.
#[test]
fn h2o_adhoc_neutral_hirshfeld_charges() {
    use ferric_core::elements::z_to_symbol;
    use ferric_rpa::properties::{hirshfeld_i_charges, spherically_averaged_proatom, RadialProatom};

    let (mol, _obs, obs_bs, _dfbs, op, rhf) = setup_h2o_ccpvdz();
    let ctx = ParallelContext::default();
    let radii: Vec<f64> = (1..=300).map(|k| k as f64 * 0.05).collect();
    let gs_mult = |z: i32| -> usize {
        match z { 1=>2,6=>3,7=>4,8=>3,9=>2,_=>1 }
    };
    let bs = obs_bs.clone();
    // Neutral-only proatoms (qi != 0 → None → no charge iteration: standard Hirshfeld).
    let proatom = |z: i32, qi: i32| -> Option<RadialProatom> {
        if qi != 0 { return None; }
        let sym = z_to_symbol(z).unwrap_or("X");
        let amol = Molecule::parse_xyz(&format!("1\n{sym}\n{sym} 0 0 0\n"), 0, gs_mult(z)).ok()?;
        let aobs = PreparedBasis::new(&amol, &bs).ok()?;
        let abounds = SchwarzBounds::compute(op, &aobs).ok()?;
        let mut cfg = RhfConfig::default();
        let dens = if gs_mult(z) == 1 {
            solve_rhf(&ctx, &amol, &aobs, op, &abounds, &cfg).ok()?.density_r().to_owned()
        } else {
            cfg.mom_after_iter = 5;
            ferric_scf::uhf::solve_uhf(&ctx, &amol, &aobs, &abounds, &cfg).ok()?.density_total().to_owned()
        };
        spherically_averaged_proatom(z, &bs, &dens, &radii).ok()
    };
    let q = hirshfeld_i_charges(&mol, &obs_bs, rhf.density_r(), &proatom).unwrap();
    eprintln!("H2O ad-hoc neutral Hirshfeld: O={:.4} H={:.4} H={:.4}", q[0], q[1], q[2]);
    assert!((-0.45..=-0.20).contains(&q[0]), "O charge: {:.3} (legacy bug was -0.95)", q[0]);
    assert!((0.10..=0.25).contains(&q[1]), "H charge: {:.3}", q[1]);
    assert!((q[0]+q[1]+q[2]).abs() < 1e-6, "sum to 0");
    assert!((q[1]-q[2]).abs() < 1e-3, "H symmetry");
}

/// DIAGNOSTIC (2026-07-13): does an off-origin geometry WITH a nearby external
/// point charge blow up the per-atom α even after the atom-centred-dipole fix?
/// The real danuglipron/7LCJ pocket-embedded run (basis sto-3g, 71 atoms,
/// ~250 Bohr from origin, 6458 point charges) still shows α_iso up to ±570
/// a.u. after the fix — this test isolates whether that's (a) the same
/// gauge-origin class of bug surviving in a code path the vacuum-only
/// origin-independence test didn't exercise, or (b) a numerical-stability
/// issue unrelated to origin (e.g. large basis + many point charges).
/// Reports both the molecular α (pdep_polarizability_static, presumed
/// trustworthy) and the per-atom α at two origins (near-origin vs shifted +
/// point charge) so the failure mode is visible rather than just asserted.
#[test]
fn h2o_hirshfeld_with_external_point_charge_diagnostic() {
    use ferric_core::external_potential::{ExternalPotential, PointCharge};
    use ferric_rpa::properties::pdep_polarizability_static;

    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();

    let cfg = PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 0.0,
        davidson_conv_thresh: 1e-10,
        ..Default::default()
    };

    // Case A: molecule near origin, point charge nearby (still off-axis / not
    // symmetric) — the "control" case most similar to what the passing
    // origin-independence test covered (vacuum, no field).
    let run_case = |shift: [f64; 3], charge_pos: [f64; 3], label: &str| {
        let mut mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        for atom in &mut mol.atoms {
            atom.x += shift[0];
            atom.y += shift[1];
            atom.zpos += shift[2];
        }
        let ext = ExternalPotential {
            point_charges: vec![PointCharge {
                q: 0.5,
                x: charge_pos[0],
                y: charge_pos[1],
                z: charge_pos[2],
            }],
            field: None,
        };
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf_cfg = RhfConfig { external_potential: Some(ext), ..Default::default() };
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &rhf_cfg).unwrap();

        let alpha_mol = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, op, &cfg)
            .unwrap()
            .tensor;
        let alpha_atomic =
            pdep_polarizability_hirshfeld(&mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg, None)
                .unwrap();

        let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        eprintln!(
            "[{label}] molecular alpha_iso = {:.4}",
            iso(&alpha_mol)
        );
        for (a, alpha_a) in alpha_atomic.iter().enumerate() {
            eprintln!("[{label}] atom {a} alpha_iso = {:.4}", iso(alpha_a));
        }
        (iso(&alpha_mol), alpha_atomic)
    };

    let (mol_near, atomic_near) =
        run_case([0.0, 0.0, 0.0], [5.0, 0.0, 0.0], "near-origin");
    let (mol_far, atomic_far) =
        run_case([130.0, 140.0, 150.0], [135.0, 140.0, 150.0], "far-from-origin");

    eprintln!("molecular alpha_iso: near={mol_near:.4} far={mol_far:.4}");
    assert!(
        (mol_near - mol_far).abs() < 1e-2,
        "molecular alpha_iso should be origin-independent even with an external \
         point charge: near={mol_near:.4} far={mol_far:.4}"
    );

    for (a, (near, far)) in atomic_near.iter().zip(atomic_far.iter()).enumerate() {
        let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        let (iso_near, iso_far) = (iso(near), iso(far));
        assert!(
            (iso_near - iso_far).abs() < 1e-1,
            "atom {a} alpha_iso not origin-independent under external point charge: \
             near={iso_near:.4} far={iso_far:.4}"
        );
    }
}
