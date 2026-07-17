//! Verify `free_atom_ref::ts_free_atom`'s `vol_free` table (Z=1..=18)
//! against ferric's own free-atom UKS-PBE + Becke-volume pipeline.
//!
//! This reuses exactly the machinery `ferric-cli`'s TS-C6 free-atom fallback
//! already uses (see `crates/ferric-cli/src/main.rs` around the "Compute
//! free-atom vol_free" comment): a free neutral-atom SCF at the correct
//! ground-state multiplicity (UKS-PBE with `fractional_occ: true` for
//! open-shell atoms to avoid the degenerate-p-shell GGA oscillation, RHF-PBE
//! for closed shells), followed by `atomic_effective_volumes_becke` on the
//! resulting density. For a single isolated atom the Becke partition weight
//! is 1 everywhere (no neighbors), so this computes exactly
//! `v_free = ∫ ρ_atom(r) |r|³ dr` — the TS vol_free denominator.
//!
//! See docs/perf-tasks/G7-verify-vol-free-table.md (task) and
//! docs/vol-free-verification.md (result table + per-element verdicts +
//! root-cause dig: which table entries were ever actually sourced, and
//! whether the disagreement is a grid-quadrature artifact or real).
//!
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --test free_atom_volumes_pbe \
//!     --release -- --nocapture --ignored
//!
//! `#[ignore]`d by default: 18 free-atom UKS-PBE solves at aug-cc-pVDZ are
//! slow (open-shell p/d-degenerate atoms in particular) and this is a
//! one-time verification, not a regression gate (the regression gate is
//! `free_atom_ref::tests`, which pins the *current* table values).
//!
//! This file also has a second, smaller `#[ignore]`d test,
//! `diagnose_becke_vs_hirshfeld_grid_truncation`, which cross-checks the
//! primary test's `atomic_effective_volumes_becke` (unbounded Becke-Lebedev
//! quadrature) against `atomic_effective_volumes_hirshfeld` (the fixed
//! 6-Bohr-margin real-space cubic grid `ferric-cli`'s actual free-atom TS
//! fallback uses) on 6 representative elements, to separate "disagreement
//! is a grid/truncation artifact" from "disagreement is a real difference
//! vs the table" (see docs/vol-free-verification.md "Reading the pattern").

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::free_atom_ref::ts_free_atom;
use ferric_rpa::properties::{atomic_effective_volumes_becke, atomic_effective_volumes_hirshfeld};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

/// Neutral free-atom ground-state multiplicity for Z=1..=18. Mirrors
/// `ferric_cli::main::proatom_gs_mult` / `ferric_scf::guess::atom_ground_state_mult`
/// exactly (kept as a local literal copy since both are crate-private) —
/// see crates/ferric-cli/src/main.rs:413-431.
fn gs_mult(z: usize) -> usize {
    match z {
        1 | 3 | 5 | 9 | 11 | 13 | 17 => 2, // doublets: H, Li, B, F, Na, Al, Cl
        6 | 8 | 14 | 16 => 3,              // triplets (3P): C, O, Si, S
        7 | 15 => 4,                       // quartets (4S): N, P
        _ => 1,                            // singlets: He, Be, Ne, Mg, Ar
    }
}

fn symbol(z: usize) -> &'static str {
    ferric_core::elements::z_to_symbol(z as i32).unwrap_or("X")
}

/// Free-atom UKS/RKS-PBE volume for element `z` at basis `basis_name`.
/// Returns None if the SCF fails to converge (reported, not fabricated).
fn pbe_vol_free(z: usize, basis_name: &str) -> Option<f64> {
    let sym = symbol(z);
    let xyz = format!("1\n{sym}\n{sym} 0 0 0\n");
    let mult = gs_mult(z);
    let mol = Molecule::parse_xyz(&xyz, 0, mult).ok()?;
    let bs = basis::bundled(basis_name).ok()?;
    let obs = PreparedBasis::new(&mol, &bs).ok()?;
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).ok()?;

    let density = if mult > 1 {
        let cfg = RhfConfig {
            xc: Some("PBE".to_string()),
            fractional_occ: true,
            mom_after_iter: 5,
            max_iter: 200,
            ..Default::default()
        };
        solve_uhf(&ctx, &mol, &obs, &bounds, &cfg)
            .ok()
            .map(|r| r.density_total().to_owned())
    } else {
        let cfg = RhfConfig {
            xc: Some("PBE".to_string()),
            max_iter: 200,
            ..Default::default()
        };
        solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg)
            .ok()
            .map(|r| r.density_r().to_owned())
    }?;

    let vols = atomic_effective_volumes_becke(&mol, &obs, &bs, &density).ok()?;
    Some(vols[0])
}

#[test]
#[ignore] // slow: 18 free-atom UKS-PBE solves at aug-cc-pVDZ; run explicitly.
fn verify_vol_free_table_z1_18_pbe() {
    println!(
        "\nVerification of free_atom_ref::ts_free_atom vol_free (Z=1..=18) \
         against ferric's own free-atom UKS/RKS-PBE + Becke-volume pipeline."
    );
    println!(
        "Reuses the exact SCF convention ferric-cli/src/main.rs already uses for \
         its free-atom TS fallback (xc=PBE, fractional_occ for open shells, \
         mom_after_iter=5, Becke partition; single free atom => Becke w=1 \
         everywhere, so this is exactly v_free = integral rho(r) |r|^3 dr)."
    );
    println!();
    println!(
        "{:>4} {:>4} {:>6} {:>12} {:>14} {:>14} {:>8}",
        "Z", "Sym", "mult", "table", "aug-cc-pvdz", "aug-cc-pvtz", "%diff(pvtz)"
    );

    let mut rows = Vec::new();
    for z in 1..=18usize {
        let (_, _, table_vf) = ts_free_atom(z).expect("Z=1..=18 must be in table");
        let table_vf = table_vf.expect("Z=1..=18 vol_free must be Some");

        let v_dz = pbe_vol_free(z, "aug-cc-pvdz");
        let v_tz = pbe_vol_free(z, "aug-cc-pvtz");

        let pct = v_tz.map(|v| 100.0 * (v - table_vf) / table_vf);

        println!(
            "{:>4} {:>4} {:>6} {:>12.3} {:>14} {:>14} {:>8}",
            z,
            symbol(z),
            gs_mult(z),
            table_vf,
            v_dz.map(|v| format!("{v:.3}")).unwrap_or_else(|| "FAILED".into()),
            v_tz.map(|v| format!("{v:.3}")).unwrap_or_else(|| "FAILED".into()),
            pct.map(|p| format!("{p:+.1}%")).unwrap_or_else(|| "N/A".into()),
        );
        rows.push((z, table_vf, v_dz, v_tz, pct));
    }

    println!();
    println!("Agreement threshold: <10% (Bučko et al. JCTC 9, 4293 (2013) cross-validation precedent).");
    let mut agree = Vec::new();
    let mut disagree = Vec::new();
    let mut failed = Vec::new();
    for (z, table_vf, _v_dz, v_tz, pct) in &rows {
        match pct {
            Some(p) if p.abs() < 10.0 => agree.push((*z, *table_vf, v_tz.unwrap(), *p)),
            Some(p) => disagree.push((*z, *table_vf, v_tz.unwrap(), *p)),
            None => failed.push(*z),
        }
    }
    println!("AGREE (<10%): {:?}", agree.iter().map(|r| symbol(r.0)).collect::<Vec<_>>());
    println!("DISAGREE (>=10%): {:?}", disagree.iter().map(|r| symbol(r.0)).collect::<Vec<_>>());
    if !failed.is_empty() {
        println!("SCF FAILED (no computed value): {:?}", failed.iter().map(|z| symbol(*z)).collect::<Vec<_>>());
    }

    // This test is a verification report, not a regression gate — it always
    // "passes" (info only) unless every single SCF failed outright, which
    // would indicate the pipeline itself is broken (not a table disagreement).
    assert!(
        rows.iter().any(|(_, _, _, v_tz, _)| v_tz.is_some()),
        "every free-atom PBE SCF failed — pipeline is broken, not a table mismatch"
    );
}

/// Diagnostic (not the primary verification): compare the unbounded
/// Becke-Lebedev `atomic_effective_volumes_becke` quadrature used above
/// against `atomic_effective_volumes_hirshfeld` — the fixed
/// 6-Bohr-margin/0.2-Bohr-spacing real-space cubic grid that
/// `ferric-cli/src/main.rs`'s ACTUAL free-atom TS fallback uses (see
/// "Compute free-atom vol_free using Hirshfeld on isolated atoms" there).
/// Both reduce exactly for a single free atom (partition weight = 1
/// everywhere in both schemes), so any gap between them is pure grid/
/// truncation artifact, not a partition-scheme physics difference — this
/// tests whether the vol_free-table's true convention is the *bounded*
/// grid (which would explain the G7 disagreement trend as tail
/// truncation, worse for more diffuse atoms) rather than a real free-atom
/// integral. See docs/vol-free-verification.md.
#[test]
#[ignore] // slow: 6 free-atom UKS-PBE solves at aug-cc-pVTZ; diagnostic only.
fn diagnose_becke_vs_hirshfeld_grid_truncation() {
    println!(
        "\nDiagnostic: unbounded Becke-Lebedev vs 6-Bohr-margin Hirshfeld-grid \
         free-atom volumes (aug-cc-pVTZ PBE), representative Z."
    );
    println!(
        "{:>4} {:>4} {:>14} {:>14} {:>10} {:>12}",
        "Z", "Sym", "table", "Becke", "Hirshfeld", "Hirsh %diff"
    );
    for z in [1usize, 6, 10, 11, 14, 18] {
        let (_, _, table_vf) = ts_free_atom(z).unwrap();
        let table_vf = table_vf.unwrap();

        let sym = symbol(z);
        let xyz = format!("1\n{sym}\n{sym} 0 0 0\n");
        let mult = gs_mult(z);
        let mol = Molecule::parse_xyz(&xyz, 0, mult).unwrap();
        let bs = basis::bundled("aug-cc-pvtz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();

        let density = if mult > 1 {
            let cfg = RhfConfig {
                xc: Some("PBE".to_string()),
                fractional_occ: true,
                mom_after_iter: 5,
                max_iter: 200,
                ..Default::default()
            };
            solve_uhf(&ctx, &mol, &obs, &bounds, &cfg)
                .unwrap()
                .density_total()
                .to_owned()
        } else {
            let cfg = RhfConfig {
                xc: Some("PBE".to_string()),
                max_iter: 200,
                ..Default::default()
            };
            solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg)
                .unwrap()
                .density_r()
                .to_owned()
        };

        let v_becke = atomic_effective_volumes_becke(&mol, &obs, &bs, &density).unwrap()[0];
        let v_hirsh =
            atomic_effective_volumes_hirshfeld(&mol, &bs, &density, None).unwrap()[0];
        let pct_hirsh = 100.0 * (v_hirsh - table_vf) / table_vf;

        println!(
            "{:>4} {:>4} {:>14.3} {:>14.3} {:>10.3} {:>+11.1}%",
            z, sym, table_vf, v_becke, v_hirsh, pct_hirsh
        );
    }
}
