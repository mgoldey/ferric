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
//! * `cli_auto_is_the_moving_grid_hessian_for_ks`: `auto` (the default)
//!   gives the `analytic-moving-grid` record bit for bit for a closed-shell
//!   KS reference (water, LDA and PBE).
//! * `cli_auto_falls_back_to_fd_where_unsupported`: `auto` gives the `fd`
//!   record bit for bit for meta-GGA (SCAN), an open-shell UKS reference and
//!   `[dft] dispersion` (H2).
//! * `cli_fd_stays_selectable_for_ks`: `hessian = "fd"` still differences
//!   gradients for a supported KS reference.
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
//! * `Auto` routed back to finite differences for KS (the `Auto` arm of
//!   `analytic_route` deleted): `cli_auto_is_the_moving_grid_hessian_for_ks`
//!   fails (`finite-difference` vs `analytic-moving-grid`).
//! * `Auto` sent to the moving-grid route for every KS reference without the
//!   support predicate or the fallback: `cli_auto_falls_back_to_fd_where_unsupported`
//!   fails (the SCAN and UKS runs exit with the moving-grid refusal).
//! * the CLI ignoring `[frequencies] hessian` (always `auto`):
//!   `cli_fd_stays_selectable_for_ks` fails (`analytic-moving-grid` vs
//!   `finite-difference`); the moving-grid tests cannot see it now that
//!   `auto` is the moving-grid Hessian for KS.
//! * the predicate (`rks_moving_grid_hessian_available`) not called: every
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
fn cli_auto_is_the_moving_grid_hessian_for_ks() {
    if !has_deriv2() {
        return;
    }
    let moving = "[frequencies]\nhessian = \"analytic-moving-grid\"\n";
    for xc in ["LDA", "PBE"] {
        let auto = run_cli_ok(&format!("auto_{xc}"), WATER, xc, "", "");
        let explicit = run_cli_ok(&format!("moving_{xc}"), WATER, xc, moving, "");
        assert_eq!(auto.source, "analytic-moving-grid", "{xc}");
        assert_eq!(auto.n_grad, 0, "{xc}");
        assert_same_record(
            &auto,
            &explicit,
            &format!("{xc}: auto vs analytic-moving-grid"),
        );
        eprintln!(
            "{xc}: auto == analytic-moving-grid, {:?} cm^-1",
            auto.frequencies
        );
    }
}

#[test]
fn cli_auto_falls_back_to_fd_where_unsupported() {
    let fd = "[frequencies]\nhessian = \"fd\"\n";
    // (label, method kind, functional, extra [dft] lines)
    let cases = [
        ("mgga", "ksdft", "SCAN", ""),
        ("uks", "uhf", "LDA", ""),
        ("disp", "ksdft", "B3LYP", "dispersion = \"d3bj\"\n"),
    ];
    for (label, kind, xc, extra) in cases {
        let auto = run_cli_kind_ok(&format!("auto_fb_{label}"), H2, kind, xc, "", extra);
        let diff = run_cli_kind_ok(&format!("fd_fb_{label}"), H2, kind, xc, fd, extra);
        assert_eq!(auto.source, "finite-difference", "{label}");
        assert_eq!(auto.n_grad, 12, "{label}");
        assert_same_record(&auto, &diff, &format!("{label}: auto vs fd"));
        eprintln!("{label}: auto == fd, {:?} cm^-1", auto.frequencies);
    }
}

#[test]
fn cli_fd_stays_selectable_for_ks() {
    let fd = run_cli_ok("fd_lda", H2, "LDA", "[frequencies]\nhessian = \"fd\"\n", "");
    assert_eq!(fd.source, "finite-difference");
    assert_eq!(fd.n_grad, 12);
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
