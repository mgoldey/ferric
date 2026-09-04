//! Free-atom UKS/RKS-PBE + Becke-volume pipeline check for Z=1..=18.
//!
//! Originally (G7, docs/perf-tasks/G7-verify-vol-free-table.md) this verified
//! `free_atom_ref::ts_free_atom`'s hardcoded `vol_free` table against a live
//! free-atom calc and found 12 of 18 entries were never sourced and disagreed
//! 11%–98%. G8 (docs/perf-tasks/G8-fix-vol-free-table.md) removed the CLI's
//! table fallback entirely and dropped the Z≤18 `vol_free` entries to `None`,
//! so the live free-atom SCF is now the ONLY source of the TS vol_free
//! denominator. This test therefore no longer compares against a table — its
//! lasting job is a **robustness gate**: for every Z=1..=18 at its ground-state
//! multiplicity, does the live free-atom SCF actually converge and yield a
//! finite volume? An element that fails here would have TS C6 silently skipped
//! for any molecule containing it (the honest new behavior), so a regression to
//! "flaky free-atom SCF" is worth catching.
//!
//! It reuses exactly the machinery `ferric-cli`'s TS-C6 branch uses (see
//! `crates/ferric-cli/src/lib.rs` around the "Compute free-atom vol_free"
//! comment): a free neutral-atom SCF at the correct ground-state multiplicity
//! (UKS-PBE with `fractional_occ: true` for open-shell atoms to avoid the
//! degenerate-p-shell GGA oscillation, RHF-PBE for closed shells), followed by
//! `atomic_effective_volumes_becke` on the resulting density. For a single
//! isolated atom the Becke partition weight is 1 everywhere (no neighbors), so
//! this computes exactly `v_free = ∫ ρ_atom(r) |r|³ dr` — the TS vol_free
//! denominator.
//!
//! See docs/vol-free-verification.md (the G7 result table + per-element
//! verdicts + root-cause dig + the G8 disposition update).
//!
//!   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-rpa --test free_atom_volumes_pbe \
//!     --release -- --nocapture --ignored
//!
//! `#[ignore]`d by default: 18 free-atom UKS-PBE solves at aug-cc-pVDZ are
//! slow (open-shell p/d-degenerate atoms in particular) and this is a
//! convergence-robustness probe, not a fast CI gate.
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
use ferric_rpa::properties::{atomic_effective_volumes_becke, atomic_effective_volumes_hirshfeld};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

/// Neutral free-atom ground-state multiplicity for Z=1..=18. Mirrors
/// `ferric_cli::run::proatom_gs_mult` / `ferric_scf::guess::atom_ground_state_mult`
/// exactly (kept as a local literal copy since both are crate-private) —
/// see crates/ferric-cli/src/lib.rs:500-517 (this reference was already
/// stale before ferric-cli's main.rs -> lib.rs move -- 413-431 pointed at
/// the SCF-ladder dispatch, not this closure -- corrected while fixing the
/// line-number shift the rename caused).
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
fn free_atom_scf_converges_z1_18_pbe() {
    println!(
        "\nRobustness gate: does the live free-atom UKS/RKS-PBE + Becke-volume \
         pipeline converge and yield a finite v_free for every Z=1..=18?"
    );
    println!(
        "The hardcoded vol_free table was removed (G8) — this live SCF is now \
         the ONLY source of the TS vol_free denominator, so an element that \
         fails here would have TS C6 silently skipped for any molecule \
         containing it. Reuses the exact SCF convention ferric-cli/src/lib.rs \
         uses (xc=PBE, fractional_occ for open shells, mom_after_iter=5, Becke \
         partition; single free atom => Becke w=1 everywhere, so this is exactly \
         v_free = integral rho(r) |r|^3 dr). Also prints the G7 aug-cc-pVTZ \
         reference volumes for the record (docs/vol-free-verification.md)."
    );
    println!();
    println!(
        "{:>4} {:>4} {:>6} {:>14} {:>14} {:>10}",
        "Z", "Sym", "mult", "aug-cc-pvdz", "aug-cc-pvtz", "DZ->TZ%"
    );

    let mut failed = Vec::new();
    for z in 1..=18usize {
        let v_dz = pbe_vol_free(z, "aug-cc-pvdz");
        let v_tz = pbe_vol_free(z, "aug-cc-pvtz");
        let dz_tz = match (v_dz, v_tz) {
            (Some(a), Some(b)) => Some(100.0 * (b - a) / a),
            _ => None,
        };

        println!(
            "{:>4} {:>4} {:>6} {:>14} {:>14} {:>10}",
            z,
            symbol(z),
            gs_mult(z),
            v_dz.map(|v| format!("{v:.3}")).unwrap_or_else(|| "FAILED".into()),
            v_tz.map(|v| format!("{v:.3}")).unwrap_or_else(|| "FAILED".into()),
            dz_tz.map(|p| format!("{p:+.1}%")).unwrap_or_else(|| "N/A".into()),
        );

        // The gate: a finite converged volume at BOTH bases. aug-cc-pVTZ is the
        // production-relevant basis (the C6 examples run at aug-cc-pVTZ); DZ is
        // the cheaper cross-check. Fail loud on any element that can't produce a
        // volume — that's the regression this test exists to catch.
        if v_dz.is_none() || v_tz.map(|v| !(v.is_finite() && v > 0.0)).unwrap_or(true) {
            failed.push(z);
        }
    }

    println!();
    if !failed.is_empty() {
        println!(
            "SCF FAILED / non-finite volume for: {:?}",
            failed.iter().map(|z| symbol(*z)).collect::<Vec<_>>()
        );
    } else {
        println!("All Z=1..=18 free-atom SCF converged to a finite v_free.");
    }

    assert!(
        failed.is_empty(),
        "live free-atom SCF failed to yield a finite volume for {:?} — TS C6 \
         would be silently skipped for molecules containing these elements; \
         fix the convergence issue (mirror the O/S/Si fractional_occ+MOM fix) \
         rather than accepting a gap",
        failed.iter().map(|z| symbol(*z)).collect::<Vec<_>>()
    );
}

/// Diagnostic (not the primary verification): compare the unbounded
/// Becke-Lebedev `atomic_effective_volumes_becke` quadrature used above
/// against `atomic_effective_volumes_hirshfeld` — the fixed
/// 6-Bohr-margin/0.2-Bohr-spacing real-space cubic grid that
/// `ferric-cli/src/lib.rs`'s ACTUAL free-atom TS fallback uses (see
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
        "{:>4} {:>4} {:>14} {:>10} {:>14}",
        "Z", "Sym", "Becke(unbdd)", "Hirshfeld", "Becke<->Hirsh%"
    );
    for z in [1usize, 6, 10, 11, 14, 18] {
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
        // Both partition weights are trivially 1 for a single free atom, so any
        // gap between the two is a pure grid/truncation artifact (the bounded
        // 6-Bohr Hirshfeld grid truncates diffuse tails, e.g. Na's 3s valence).
        let gap = 100.0 * (v_becke - v_hirsh) / v_hirsh;

        println!(
            "{:>4} {:>4} {:>14.3} {:>10.3} {:>+13.1}%",
            z, sym, v_becke, v_hirsh, gap
        );
    }
}
