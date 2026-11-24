//! Issue #498 (meta-GGA ROKS derivatives, from #492): the k-point ROKS
//! SCAN / r2SCAN force and stress (`kpoint_roks_gradient`,
//! `kpoint_roks_stress`, dense-AFT J/K) wired to the CLI (`task =
//! "optimize"`, `[cell] stress = true`). The binding change is the library
//! derivative gate (`ferric_pbc::kxcgrad::check_deriv_functional`, shared
//! with Python), which refused every ROKS meta-GGA derivative; the CLI's
//! force and stress dispatch already routed a ROKS SCF to these entries.
//! Shared harness: `open_rsh_cli/mod.rs`.
//!
//! EXACTNESS ANCHOR (CLI vs library, BIT FOR BIT): the CLI energy, gradient
//! (`task = "optimize"` with zero optimizer steps: the gradient of the input
//! geometry) and stress tensor are read from the run log's `run_end` record;
//! the library is called on the SAME cell with the CLI's configs restated.
//! Fixture: the H3 doublet in a 4 Bohr cubic cell, STO-3G, 1x1x2 mesh,
//! exxdiv ewald, XC grid 20x50 with a 14 Bohr cutoff.
//!
//! SIBLING (the mis-route a bit-equal pass could hide): the library k-point
//! UKS force / stress of the same functional at the same geometry. A CLI that
//! differentiated a UKS state, or the UKS assembly on the ROKS densities
//! without `W_RO`, would match it instead; it must differ by far more than
//! one ulp.
//!
//! INDEPENDENT CONSTRUCTION (in the same test so the CLI run is shared): the
//! CLI force and `sigma_zz` against a central finite difference of the CLI's
//! OWN energies. Artifact hypothesis: a wiring bug that differentiates the
//! wrong spin density or drops a tau / `W_RO` term misses the FD by the size
//! of the term (>= 5e-3 Ha/Bohr in the library's mutation ledger,
//! `validation_pbc_kroks_mgga.rs`), a correct one by the FD floor only.
//!
//! Measured (2026-11-19, test profile): every anchor bit-equal. UKS
//! sibling: SCAN force 2.07e-3 Ha/Bohr, r2SCAN stress 1.18e-5 Ha/Bohr^3.
//! FD (h = 1e-4): SCAN dE/dz(atom 1) 6.1e-10 (|g| 0.356, bar 1e-7); r2SCAN
//! sigma_zz 9.4e-10 (|sigma_zz| 1.04e-2, bar 1e-8). Runtime 60 s for both
//! (test profile, one thread). The printed `run_end` values are pinned in
//! the Python twin `crates/ferric-python/tests/test_pbc_kroks_mgga_derivs.py`.
//!
//! Fail-before: on origin/main both anchor tests fail at the CLI run (the
//! derivative gate refused "the k-point ROKS meta-GGA force and stress are
//! library-only").
//!
//! Mutation ledger (CLI source mutated, rebuilt, this file run, reverted),
//! KILLED: the k-point ROKS stress returned transposed fails the r2SCAN
//! anchor on the bits (the tensor is unsymmetrised; the zz FD alone is blind
//! to it). The gate mutant (the blanket ROKS meta-GGA refusal restored) is
//! the fail-before state above.

mod open_rsh_cli;
use open_rsh_cli::*;

const H: f64 = 1e-4;

/// k-point ROKS SCAN force: CLI == library bit for bit, differs from the
/// UKS SCAN force at the same geometry, and matches a central FD of the
/// CLI's own energy along atom 1 z (the largest component).
#[ignore = "tier slow (>=60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn kpoint_roks_scan_force_equals_the_library_and_matches_fd() {
    let r = Run::new("kroks-scan-f", Spin::Roks, "SCAN", "optimize");
    let (e, x, g) = force_anchor(&r);
    eprintln!("k ROKS SCAN run_end: E {e:?}\n x {x:?}\n g {g:?}");
    let uks = lib_force(
        &Run {
            spin: Spin::Uks,
            ..r
        },
        Some(&x),
    );
    let sib = max_diff(&g, &uks.1);
    eprintln!("k ROKS SCAN force vs UKS SCAN sibling: {sib:.2e}");
    assert!(sib > 1e-5, "ROKS force indistinguishable from UKS: {sib:e}");
    let e_at = |dz: f64, tag: &str| {
        let mut at = H3;
        at[1][2] += dz;
        cli_energy(&Run {
            atoms: Some(&at),
            ..Run::new(tag, Spin::Roks, "SCAN", "energy")
        })
    };
    let fd = (e_at(H, "kroks-scan-fd-p") - e_at(-H, "kroks-scan-fd-m")) / (2.0 * H);
    let miss = (fd - g[5]).abs();
    eprintln!(
        "k ROKS SCAN dE/dz(atom 1): CLI {:.12e} FD {fd:.12e} miss {miss:.2e}",
        g[5]
    );
    assert!(g[5].abs() > 1e-2, "trivial component");
    assert!(miss < 1e-7, "FD miss {miss:e}");
}

/// k-point ROKS r2SCAN stress: CLI == library bit for bit, differs from the
/// UKS r2SCAN stress, and `sigma_zz = (1/Omega) dE/d(eps_zz)` matches a
/// central FD of the CLI's own energy under a uniaxial strain (lattice c and
/// every atom z scaled by `1 + eps`).
#[ignore = "tier slow (>=60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn kpoint_roks_r2scan_stress_equals_the_library_and_matches_fd() {
    let r = Run::new("kroks-r2scan-s", Spin::Roks, "r2SCAN", "stress");
    let (e, s) = stress_anchor(&r);
    eprintln!("k ROKS r2SCAN run_end: E {e:?}\n sigma {s:?}");
    let uks = lib_stress(&Run {
        spin: Spin::Uks,
        ..r
    });
    let sib = max_diff(&s, &uks.1);
    eprintln!("k ROKS r2SCAN stress vs UKS r2SCAN sibling: {sib:.2e}");
    assert!(
        sib > 1e-6,
        "ROKS stress indistinguishable from UKS: {sib:e}"
    );
    let e_at = |eps: f64, tag: &str| {
        cli_energy(&Run {
            strain_zz: eps,
            ..Run::new(tag, Spin::Roks, "r2SCAN", "energy")
        })
    };
    let omega = A_BOHR.powi(3);
    let fd = (e_at(H, "kroks-r2-sfd-p") - e_at(-H, "kroks-r2-sfd-m")) / (2.0 * H) / omega;
    let miss = (fd - s[8]).abs();
    eprintln!(
        "k ROKS r2SCAN sigma_zz: CLI {:.12e} FD {fd:.12e} miss {miss:.2e}",
        s[8]
    );
    assert!(s[8].abs() > 1e-4, "trivial component");
    assert!(miss < 1e-8, "FD miss {miss:e}");
}

/// What stays refused, by name, before any integral: the ROKS meta-GGA on
/// the RS-GDF J/K (its SCF refuses it, so its derivatives do too) and TPSS
/// (closed-shell only).
#[test]
fn roks_meta_gga_derivatives_outside_the_envelope_are_refused() {
    let e = refused(&Run {
        rsgdf: true,
        ..Run::new("kroks-scan-gdf-f", Spin::Roks, "SCAN", "optimize")
    });
    assert!(
        e.contains("the k-point ROKS SCAN / r2SCAN energy is validated on the dense-AFT"),
        "{e}"
    );
    let e = refused(&Run::new("kroks-tpss-s", Spin::Roks, "TPSS", "stress"));
    assert!(
        e.contains("k-point ROKS supports only the pure SCAN / r2SCAN"),
        "{e}"
    );
}
