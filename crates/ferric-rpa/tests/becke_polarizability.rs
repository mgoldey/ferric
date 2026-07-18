//! Tests for Becke-Lebedev per-atom static polarizability.
//!
//! `pdep_polarizability_becke` uses the ATOM-CENTRED (r − R_A) dipole
//! operator (fixed 2026-07-13 — see the gauge-origin regression note below),
//! so per-atom tensors omit inter-atomic charge-transfer coupling by
//! construction and are NOT expected to sum to the molecular α. Origin
//! independence is the correctness property that matters instead.
//!
//! Unlike the Slater-Hirshfeld scheme (which had ~50% magnitude error from
//! the bad proatom), Becke is geometry-only and accurate at production
//! grid sizes (75 radial × 110 angular).
//!
//! GAUGE-ORIGIN REGRESSION (2026-07-13): this function previously used raw
//! lab-frame grid coordinates `r` in the AO dipole quadrature, then
//! renormalized each atom's contribution to match the global lab-frame
//! analytical dipole. That renormalization is gauge-breaking for off-origin
//! geometries — the real danuglipron/7LCJ production run (71 atoms, sto-3g,
//! ~250 Bohr from origin, 6458 external point charges) showed α_iso up to
//! hundreds of a.u. A small water+point-charge test didn't catch it because
//! `pdep_polarizability_hirshfeld` (a sibling function with the identical bug
//! pattern) was fixed and tested first, but THIS function — the one actually
//! wired to the CLI's `alpha_atomic` NPZ export (`main.rs` compute_alpha_atomic
//! path) — still had the bug until this fix. Any future per-atom property
//! function copied from this file should use atom-centred (r − R_A), not
//! lab-frame r, and should NOT renormalize to a lab-frame analytical total.

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::pdep_polarizability_becke;
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn setup_h2o() -> (Molecule, PreparedBasis, ferric_core::basis::BasisSet, PreparedBasis,
                   Operator, ferric_scf::ScfResult) {
    let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
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
fn becke_h2o_origin_independent_with_external_point_charge() {
    // Regression test for the 2026-07-13 gauge-origin bug: translate the
    // whole molecule far from the origin (comparable to danuglipron's real
    // cryo-EM displacement) with a nearby external point charge, and confirm
    // both the molecular alpha (pdep_polarizability_static, trusted) and the
    // atom-centred per-atom alpha are unchanged.
    use ferric_rpa::properties::pdep_polarizability_static;

    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let cfg = PdepRpaConfig::default();

    let run_case = |shift: [f64; 3], charge_pos: [f64; 3]| {
        let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
        let mut mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        for atom in &mut mol.atoms {
            atom.x += shift[0];
            atom.y += shift[1];
            atom.zpos += shift[2];
        }
        let ext = ExternalPotential {
            point_charges: vec![PointCharge {
                q: 0.5, x: charge_pos[0], y: charge_pos[1], z: charge_pos[2],
            }],
            field: None,
        };
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf_cfg = RhfConfig { external_potential: Some(ext), ..Default::default() };
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &rhf_cfg).unwrap();

        let alpha_mol = pdep_polarizability_static(&mol, &obs, &dfbs, &rhf, op, &cfg).unwrap();
        let alphas_per_atom =
            pdep_polarizability_becke(&mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg).unwrap();
        (alpha_mol.iso, alphas_per_atom)
    };

    let (mol_iso_near, atomic_near) = run_case([0.0, 0.0, 0.0], [5.0, 0.0, 0.0]);
    let (mol_iso_far, atomic_far) = run_case([130.0, 140.0, 150.0], [135.0, 140.0, 150.0]);

    eprintln!("molecular alpha_iso: near={mol_iso_near:.4} far={mol_iso_far:.4}");
    assert!(
        (mol_iso_near - mol_iso_far).abs() < 1e-2,
        "molecular alpha_iso should be origin-independent: near={mol_iso_near:.4} far={mol_iso_far:.4}"
    );

    let iso = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
    for (a, (near, far)) in atomic_near.iter().zip(atomic_far.iter()).enumerate() {
        let (iso_near, iso_far) = (iso(near), iso(far));
        eprintln!("atom {a}: near={iso_near:.4} far={iso_far:.4}");
        assert!(
            (iso_near - iso_far).abs() < 1e-1,
            "atom {a} alpha_iso not origin-independent: near={iso_near:.4} far={iso_far:.4}"
        );
        assert!(
            iso_near.abs() < 20.0,
            "atom {a} alpha_iso={iso_near:.3} is unphysically large — regression of the gauge-origin bug"
        );
    }
}

#[test]
fn becke_h2o_bit_identical_across_thread_counts() {
    // Regression guard for the 2026-07-13 Rayon-parallelization of the
    // atom-centred dipole grid quadrature (accumulate_atom_centred_dipoles):
    // the fold is chunked and reduced in a fixed ascending order (thread-count-
    // independent, mirroring df_j_bit_identical_across_thread_counts), so the
    // result must be EXACTLY the same regardless of RAYON_NUM_THREADS. Water's
    // grid (~3 atoms x 75 radial x 110 angular ~ 24.75k points) splits into
    // ~990 chunks at TARGET_CHUNKS=1024, enough for fold order to matter.
    let (mol, obs, obs_bs, dfbs, op, rhf) = setup_h2o();
    let cfg = PdepRpaConfig::default();

    let run_at = |threads: usize| -> Vec<[[f64; 3]; 3]> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            pdep_polarizability_becke(&mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg).unwrap()
        })
    };

    let a1 = run_at(1);
    let a4 = run_at(4);
    assert_eq!(
        a1, a4,
        "pdep_polarizability_becke must be bit-identical across thread counts \
         (rayon reduction order leak in accumulate_atom_centred_dipoles)"
    );
}

#[test]
fn becke_h2o_oxygen_dominant() {
    // Physically reasonable: oxygen carries more of the polarizability
    // than the hydrogens. (O is bigger, more valence electrons, more
    // polarizable.) This is a *qualitative* test — the actual ratio
    // depends on partition scheme.
    let (mol, obs, obs_bs, dfbs, op, rhf) = setup_h2o();
    let cfg = PdepRpaConfig::default();
    let alphas = pdep_polarizability_becke(&mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg).unwrap();
    let isos: Vec<f64> = alphas.iter()
        .map(|t| (t[0][0] + t[1][1] + t[2][2]) / 3.0).collect();
    eprintln!("H2O Becke per-atom α_iso: O={:.4}, H={:.4}, H={:.4}", isos[0], isos[1], isos[2]);

    // O is index 0 (per setup_h2o ordering).
    assert!(isos[0] > isos[1], "O α should exceed H α: O={}, H1={}", isos[0], isos[1]);
    assert!(isos[0] > isos[2], "O α should exceed H α: O={}, H2={}", isos[0], isos[2]);
    // H atoms equivalent (within grid noise).
    assert!((isos[1] - isos[2]).abs() < 0.05,
        "H atoms should have equivalent α: H1={}, H2={}", isos[1], isos[2]);
}
