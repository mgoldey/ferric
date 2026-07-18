//! S9 spike: per-atom anisotropic C6 tensor validation scoping (open-work
//! triage item #9).
//!
//! Two diagnostics, both `--ignored` (preflight probes, not CI gates — they
//! print physics, they don't assert a pass/fail tolerance that could rot):
//!
//! 1. `per_atom_pair_sum_vs_molecular_c6_water`: does
//!    `c6_iso_pair.sum()` (per-atom-pair Casimir-Polder sum) reproduce the
//!    already-validated `c6_molecular_iso` (DOSD-comparable molecular total)
//!    for water? Per `dispersion.rs`'s own doc comments (per_atom field,
//!    `casimir_polder_c6`'s `c6_molecular_iso` derivation) these are NOT
//!    expected to agree by construction: `per_atom` uses the atom-centred
//!    (r-R_A) operator (charge-transfer/coupling excluded, intrinsic atomic
//!    polarizability), while `molecular` is the lab-frame global-origin
//!    response (includes inter-atomic coupling). This probe quantifies the
//!    gap for a real DOSD molecule instead of relying on the doc comment.
//!
//! 2. `partition_dependence_becke_vs_hirshfeld_water`: re-run the
//!    Becke-vs-Hirshfeld per-atom comparison from the `per-atom-c6-status`
//!    memory (which found up to ~10x disagreement, driven by Hirshfeld
//!    H-starvation) now that Hirshfeld-I same-basis proatoms are wired in
//!    (commits 72ec0b5..81df65a) — is the gap improved, same, or worse?
//!
//! Run: cargo test -p ferric-rpa --release --test s9_per_atom_c6_consistency \
//!        -- --ignored --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::{casimir_polder_c6, pdep_dynamic_polarizability, DispersionPartition};
use ferric_rpa::properties::{spherically_averaged_proatom, RadialProatom};
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn water_mol() -> Molecule {
    Molecule::parse_xyz(
        "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n",
        0,
        1,
    )
    .unwrap()
}

/// DOSD-anchor probe: PBE reference, aug-cc-pVDZ, matching the validated
/// dRPA@PBE methodology from docs/dosd-c6-rpa-vs-ts.md (water DOSD=45.3 a.u.,
/// aug-cc-pVDZ dRPA@PBE reported there is the aug-cc-pVTZ row; aDZ is cheaper
/// for a spike and the ~15-19% systematic-underbind story is basis-independent
/// in SIGN, which is all this probe needs).
#[test]
#[ignore = "spike probe: cargo test -p ferric-rpa --release --test s9_per_atom_c6_consistency -- --ignored --nocapture"]
fn per_atom_pair_sum_vs_molecular_c6_water() {
    let mol = water_mol();
    let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("aug-cc-pvdz-rifit").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let scf_cfg = RhfConfig {
        xc: Some("PBE".to_string()),
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        df_k_aux: Some("def2-universal-jkfit".to_string()),
        ..Default::default()
    };
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).unwrap();
    let cfg = PdepRpaConfig {
        trunc_thresh: 0.0,
        ..Default::default()
    };

    println!("\n=== S9: per-atom pair-sum vs molecular C6 (water, aug-cc-pVDZ, RPA@PBE) ===");
    println!("  DOSD reference (Meath/Toulouse): C6(H2O) = 45.3 a.u.\n");

    for partition in [DispersionPartition::Becke, DispersionPartition::Hirshfeld] {
        let dp = pdep_dynamic_polarizability(
            &mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg, partition, None,
        )
        .unwrap();
        let res = casimir_polder_c6(&dp);

        let pair_sum: f64 = res.c6_iso_pair.sum();
        let molecular = res.c6_molecular_iso;
        let gap_pct = 100.0 * (pair_sum - molecular) / molecular;

        println!("  partition={partition:?}");
        println!("    c6_molecular_iso (DOSD-comparable, validated)   = {molecular:.4} a.u.");
        println!("    c6_iso_pair.sum() (per-atom pair Casimir-Polder) = {pair_sum:.4} a.u.");
        println!("    gap = {gap_pct:+.1}%  (pair_sum vs molecular)");

        // Per-atom self-terms (diagonal) vs cross terms, to see where the
        // "coupling" mass actually sits.
        let n = res.c6_iso_pair.nrows();
        let diag: f64 = (0..n).map(|a| res.c6_iso_pair[(a, a)]).sum();
        let off_diag = pair_sum - diag;
        println!(
            "    diag (self A=B) = {diag:.4}  off-diag (A!=B pairs, double-counted in .sum()) = {off_diag:.4}"
        );
    }
    println!(
        "\n  Expectation per dispersion.rs doc comments: pair_sum != molecular by construction \
         (per_atom uses atom-centred r-R_A operator, excludes charge-transfer/coupling that the \
         lab-frame molecular tensor includes). This probe quantifies, not just asserts, the gap."
    );
}

/// Partition-dependence re-check: Becke vs Hirshfeld per-atom C6 self-terms
/// for water, now that Hirshfeld-I same-basis proatoms are wired
/// (72ec0b5..81df65a) into the dynamic per-atom path. Compare the per-atom
/// magnitude ratio to the per-atom-c6-status memory's CH4 finding
/// (Hirshfeld C=59.6/H=0.054 vs Becke C=4.74/H=0.641, ~10x-100x apart).
#[test]
#[ignore = "spike probe: cargo test -p ferric-rpa --release --test s9_per_atom_c6_consistency -- --ignored --nocapture"]
fn partition_dependence_becke_vs_hirshfeld_water() {
    let mol = water_mol();
    let obs_bs = basis::bundled("aug-cc-pvdz").unwrap();
    let dfbs_bs = basis::bundled("aug-cc-pvdz-rifit").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let scf_cfg = RhfConfig {
        xc: Some("PBE".to_string()),
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        df_k_aux: Some("def2-universal-jkfit".to_string()),
        ..Default::default()
    };
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg).unwrap();
    let cfg = PdepRpaConfig {
        trunc_thresh: 0.0,
        ..Default::default()
    };

    // Ad-hoc same-basis neutral proatom provider (the CLI's standard choice
    // for the Hirshfeld partition today — see per-atom-c6-status memory,
    // commit 81df65a "wire ad-hoc same-basis proatom into ALL Hirshfeld paths",
    // and ferric-cli/src/main.rs:408-465 for the reference closure this
    // mirrors: neutral-only, atomic SCF in the molecule's own basis,
    // spherically averaged).
    let proatom_radii: Vec<f64> = (1..=600).map(|k| k as f64 * 0.05).collect();
    let proatom_gs_mult = |z: i32| -> usize {
        match z {
            1 | 3 | 5 | 9 | 11 | 13 | 17 | 31 | 35 | 53 => 2,
            19 | 29 | 37 | 47 => 2,
            6 | 8 | 14 | 16 | 32 | 34 => 3,
            7 | 15 | 33 => 4,
            _ if z % 2 == 1 => 2,
            _ => 1,
        }
    };
    let proatom = |z: i32, qi: i32| -> Option<RadialProatom> {
        if qi != 0 || z - qi <= 0 {
            return None;
        }
        let sym = ferric_core::elements::z_to_symbol(z).unwrap_or("X");
        let axyz = format!("1\n{sym}\n{sym} 0 0 0\n");
        let amol = Molecule::parse_xyz(&axyz, 0, proatom_gs_mult(z)).ok()?;
        let aobs = PreparedBasis::new(&amol, &obs_bs).ok()?;
        let abounds = SchwarzBounds::compute(op, &aobs).ok()?;
        let mult = proatom_gs_mult(z);
        let adens = if mult == 1 {
            solve_rhf(&ctx, &amol, &aobs, op, &abounds, &scf_cfg)
                .ok()
                .map(|r| r.density_r().to_owned())
        } else {
            let mut acfg = scf_cfg.clone();
            acfg.mom_after_iter = 5;
            if acfg.xc.is_some() {
                acfg.fractional_occ = true;
            }
            ferric_scf::uhf::solve_uhf(&ctx, &amol, &aobs, &abounds, &acfg)
                .ok()
                .map(|r| r.density_total().to_owned())
        }?;
        spherically_averaged_proatom(z, &obs_bs, &adens, &proatom_radii).ok()
    };

    let dp_becke = pdep_dynamic_polarizability(
        &mol,
        &obs,
        &obs_bs,
        &dfbs,
        &rhf,
        op,
        &cfg,
        DispersionPartition::Becke,
        None,
    )
    .unwrap();
    let dp_hirshfeld = pdep_dynamic_polarizability(
        &mol,
        &obs,
        &obs_bs,
        &dfbs,
        &rhf,
        op,
        &cfg,
        DispersionPartition::Hirshfeld,
        Some(&proatom),
    )
    .unwrap();

    let res_becke = casimir_polder_c6(&dp_becke);
    let res_hirshfeld = casimir_polder_c6(&dp_hirshfeld);

    println!("\n=== S9: Becke vs Hirshfeld per-atom self-C6 (water, aug-cc-pVDZ, RPA@PBE) ===");
    println!("  per-atom-c6-status memory (2026-06-02, CH4): Hirshfeld C=59.6/H=0.054 vs Becke C=4.74/H=0.641");
    let labels = ["O", "H1", "H2"];
    let mut ratios = Vec::new();
    for (a, label) in labels.iter().enumerate() {
        let becke = res_becke.c6_iso_pair[(a, a)];
        let hirshfeld = res_hirshfeld.c6_iso_pair[(a, a)];
        let ratio = if becke.abs() > 1e-12 { hirshfeld / becke } else { f64::NAN };
        ratios.push(ratio);
        println!(
            "  atom {label}: Becke C6_self={becke:.4}  Hirshfeld C6_self={hirshfeld:.4}  ratio(H/B)={ratio:.3}"
        );
    }
    let max_ratio = ratios
        .iter()
        .filter(|r| r.is_finite() && **r > 0.0)
        .cloned()
        .fold(0.0_f64, f64::max);
    let min_ratio = ratios
        .iter()
        .filter(|r| r.is_finite() && **r > 0.0)
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let spread = if min_ratio > 0.0 { max_ratio / min_ratio } else { f64::NAN };
    println!(
        "  partition-dependence spread across atoms (max ratio / min ratio) = {spread:.2}x \
         (memory's CH4 finding was ~10x-100x on individual atoms)"
    );
}
