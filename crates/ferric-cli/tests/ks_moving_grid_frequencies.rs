//! `[frequencies] hessian = "analytic-moving-grid"` (issue #490): the opt-in
//! analytic closed-shell Kohn-Sham Hessian on the production (moving) grid,
//! `ferric_scf::hessian::rks_moving_grid_hessian`, through the CLI.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cli --release --test ks_moving_grid_frequencies
//! ```
//!
//! The Hessian itself is validated in the library
//! (`ferric-scf/tests/ks_moving_grid_hessian.rs`,
//! `validation_ks_moving_grid_hessian.rs`). What the CLI can get wrong is the
//! wiring, which these tests pin by running the real binary and reading the
//! full-precision `frequencies` record of its JSON run log:
//!
//! * `cli_moving_grid_equals_the_library_bit_for_bit`: the CLI's Cartesian
//!   Hessian, wavenumbers and energy equal `harmonic_frequencies` with
//!   `HessianMethod::AnalyticMovingGrid` called in-process with the same
//!   `RhfConfig`, bit for bit (both run the same deterministic code on the
//!   same rayon width).
//! * `cli_auto_is_still_finite_differences_for_ks`: the default is unchanged,
//!   `auto` gives the `fd` record bit for bit for a KS reference.
//! * `cli_refusals_are_named`: every refusal names what it refuses, and
//!   fires before the SCF (no SCF iteration in the run log).
//!
//! The bit-for-bit test cannot catch a wrong ROUTE (library and CLI both
//! carry it); the per-commit routing check is
//! `ferric-scf/tests/ks_moving_grid_frequencies.rs` (routed Hessian == direct
//! `rks_moving_grid_hessian` call). The moving-grid vs finite-difference
//! agreement through the CLI (LDA and PBE, H2O) is
//! `validation_ks_moving_grid_frequencies.rs`.
//!
//! # Mutation ledger (each measured to fail)
//!
//! * `AnalyticMovingGrid` routed to `HessianMethod::FiniteDifference` inside
//!   `harmonic_frequencies`: the bit-for-bit test fails on `hessian_source`
//!   and `n_gradient_evaluations` (library and CLI both mutated, so the
//!   numbers alone would agree).
//! * `Auto` routed to the moving-grid Hessian for a KS reference: the `auto`
//!   test fails (`analytic-moving-grid` vs `finite-difference`).
//! * the CLI ignoring `[frequencies] hessian` (always `auto`): the bit-for-bit
//!   test fails on `hessian_source`.
//! * the preflight (`rks_moving_grid_hessian_preflight`) not called: every
//!   `cli_refusals_are_named` case it covers fails (the SCF runs first; the
//!   Hessian's own post-SCF refusal repeats the message, so the iteration
//!   count is what catches it).

mod ks_moving_grid_cli;

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    harmonic_frequencies, FrequencyConfig, FrequencyReference, HessianMethod, HessianSource,
};
use ks_moving_grid_cli::*;

#[test]
fn cli_moving_grid_equals_the_library_bit_for_bit() {
    if !has_deriv2() {
        return;
    }
    let cli = run_cli_ok(
        "bit_lda",
        WATER,
        "LDA",
        "[frequencies]\nhessian = \"analytic-moving-grid\"\n",
        "",
    );
    assert_eq!(cli.source, "analytic-moving-grid");
    assert_eq!(cli.n_grad, 0);

    let root = workspace_root();
    let mol = Molecule::load_xyz(root.join(WATER).to_str().unwrap()).unwrap();
    let lib = with_cli_threads(|| {
        harmonic_frequencies(
            &ParallelContext::default(),
            &mol,
            "sto-3g",
            Operator::coulomb(),
            &library_config("LDA"),
            &FrequencyConfig {
                reference: FrequencyReference::Rhf,
                hessian: HessianMethod::AnalyticMovingGrid,
                ..Default::default()
            },
        )
        .expect("library moving-grid frequencies")
    });
    assert_eq!(lib.hessian_source, HessianSource::AnalyticMovingGrid);
    assert_eq!(lib.n_gradient_evaluations, 0);
    let lib_h: Vec<f64> = lib.cartesian_hessian.iter().copied().collect();
    assert_eq!(cli.energy.to_bits(), lib.energy.to_bits(), "energy");
    assert_bits_eq(&cli.hessian, &lib_h, "Cartesian Hessian");
    assert_bits_eq(&cli.frequencies, &lib.frequencies, "frequencies");
    eprintln!(
        "CLI == library bit for bit: {} Hessian entries, frequencies {:?} cm^-1",
        lib_h.len(),
        lib.frequencies
    );
}

#[test]
fn cli_auto_is_still_finite_differences_for_ks() {
    let auto = run_cli_ok("auto_pbe", H2, "PBE", "", "");
    let fd = run_cli_ok("fd_pbe", H2, "PBE", "[frequencies]\nhessian = \"fd\"\n", "");
    assert_eq!(auto.source, "finite-difference");
    assert_eq!(auto.n_grad, 12);
    assert_eq!(fd.source, "finite-difference");
    assert_eq!(auto.energy.to_bits(), fd.energy.to_bits(), "energy");
    assert_bits_eq(&auto.hessian, &fd.hessian, "auto vs fd Hessian");
    assert_bits_eq(&auto.frequencies, &fd.frequencies, "auto vs fd frequencies");
}

#[test]
fn cli_refusals_are_named() {
    let moving = "[frequencies]\nhessian = \"analytic-moving-grid\"\n";
    // (label, molecule, method kind, functional, extra TOML, basis, fragment)
    let cases: &[(&str, &str, &str, &str, &str, &str, &str)] = &[
        (
            "hf",
            WATER,
            "rhf",
            "",
            "",
            "sto-3g",
            "Hartree-Fock has no XC grid",
        ),
        (
            "uks",
            H2,
            "uhf",
            "LDA",
            "",
            "sto-3g",
            "open-shell (Uhf, i.e. UKS/ROKS) reference",
        ),
        (
            "roks",
            H2,
            "rohf",
            "LDA",
            "",
            "sto-3g",
            "open-shell (Rohf, i.e. UKS/ROKS) reference",
        ),
        (
            "mgga",
            WATER,
            "ksdft",
            "SCAN",
            "",
            "sto-3g",
            "a meta-GGA ('SCAN')",
        ),
        (
            "vv10",
            WATER,
            "ksdft",
            "wB97X-V",
            "",
            "sto-3g",
            "VV10 nonlocal correlation",
        ),
        (
            "rsh",
            WATER,
            "ksdft",
            "HSE06",
            "",
            "sto-3g",
            "a range-separated hybrid ('HSE06')",
        ),
        (
            "ecp",
            HI,
            "ksdft",
            "LDA",
            "",
            "def2-svp",
            "an ECP or an external potential",
        ),
        (
            "extpot",
            WATER,
            "ksdft",
            "LDA",
            "[[external_potential.point_charges]]\nq = 0.5\nx = 0.0\ny = 0.0\nz = 6.0\n",
            "sto-3g",
            "an ECP or an external potential",
        ),
        (
            "prune",
            WATER,
            "ksdft",
            "LDA",
            "",
            "sto-3g",
            "an angular-pruned XC grid",
        ),
        (
            "disp",
            WATER,
            "ksdft",
            "B3LYP",
            "",
            "sto-3g",
            "hessian = \"analytic-moving-grid\" is not available",
        ),
    ];
    let mut failures = Vec::new();
    for &(label, mol, kind, xc, extra, basis, fragment) in cases {
        let mut dft = String::new();
        if !xc.is_empty() {
            dft = format!("[dft]\nfunctional = \"{xc}\"\n");
            if label == "prune" {
                dft.push_str("grid_prune = \"nwchem\"\n");
            }
            if label == "disp" {
                dft.push_str("dispersion = \"d3bj\"\n");
            }
        }
        let toml = toml_body(mol, basis, kind, &dft, &format!("{moving}{extra}"));
        let (ok, _stdout, stderr) = run_cli_raw(&format!("refuse_{label}"), &toml);
        // Refused BEFORE the SCF: the run log holds no SCF iteration.
        let iters = scf_iterations_logged(&format!("refuse_{label}"));
        if ok || !stderr.contains(fragment) || iters > 0 {
            failures.push(format!(
                "{label}: success={ok}, {iters} SCF iterations logged, stderr lacks \
                 {fragment:?}: {}",
                !stderr.contains(fragment)
            ));
        } else {
            eprintln!("{label}: refused before the SCF ({fragment})");
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
