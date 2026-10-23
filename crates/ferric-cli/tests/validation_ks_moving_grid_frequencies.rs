//! Validation tier: `[frequencies] hessian = "analytic-moving-grid"` against
//! `hessian = "fd"` through the CLI (issue #490), H2O/STO-3G, LDA and PBE,
//! default 75x110 grid, RI-J (the CLI's Kohn-Sham default), tight SCF.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cli --release \
//!   --test validation_ks_moving_grid_frequencies -- --ignored --nocapture
//! ```
//!
//! The finite-difference side is the CLI's own central difference of the
//! production KS gradient at `delta = h` and `h/2`, Richardson-extrapolated
//! per wavenumber (`(4 nu(h/2) - nu(h)) / 3`, which cancels the `h^2` term of
//! a smooth function of the Hessian). The analytic side is one SCF plus CPHF.
//!
//! # Artifact hypothesis next to the physics hypothesis
//!
//! * Right wiring: the moving-grid wavenumbers match the extrapolated FD ones
//!   at the FD floor (SCF noise over `h`, plus the `h^4` remainder).
//! * Wrong wiring that still runs an analytic KS Hessian (the fixed-grid one,
//!   which omits the Becke weight response): it misses by the grid's
//!   weight-response scale, 3-7 cm^-1 on these systems (#511), i.e. orders
//!   above the bar.
//!
//! # Measured (2026-10-23; h = 5e-4 and 2.5e-4, density_conv 1e-11)
//!
//! | functional | max \|d nu\| (cm^-1) | plain FD at h/2 | max \|d H\| (Ha/Bohr^2) | FD asymmetry |
//! |---|---|---|---|---|
//! | LDA | 7.3e-8 | 1.8e-4 | 2.7e-10 | 2.2e-8 |
//! | PBE | 2.1e-7 | 1.8e-4 | 1.2e-10 | 2.2e-8 |
//!
//! At h = 2e-3 and 1e-3 (density_conv 1e-10) the same comparison gave 5.5e-5
//! and 9.4e-5 cm^-1, i.e. the miss is the stencil's remainder, not the
//! analytic Hessian.
//!
//! # Mutation ledger
//!
//! * `analytic_frequencies_ks_moving_grid` routed to the FIXED-grid Hessian
//!   (`rks_fixed_grid_hessian_parts`): both cases fail, max |d nu| 4.63 (LDA)
//!   and 2.34 (PBE) cm^-1, max |d H| 2.4e-3 and 2.2e-3 Ha/Bohr^2 (measured
//!   2026-10-23), about 1e6 bars.

mod ks_moving_grid_cli;

use ks_moving_grid_cli::*;

/// FD step (Bohr) of the coarse difference; the fine one is half of it.
const H: f64 = 5e-4;

/// Bar on `max |nu_moving - nu_FD(Richardson)|`, cm^-1: 14x the larger
/// measured miss (2.1e-7, PBE). The fixed-grid mutant misses by 3-7 cm^-1.
const TOL_CM1: f64 = 3e-6;

/// Bar on `max |H_moving - H_FD(Richardson)|`, Ha/Bohr^2: 11x the larger
/// measured miss (2.7e-10, LDA).
const TOL_HESS: f64 = 3e-9;

fn case(xc: &str) {
    if !has_deriv2() {
        return;
    }
    let tag = xc.to_ascii_lowercase();
    let moving = run_cli_ok(
        &format!("v_{tag}_moving"),
        WATER,
        xc,
        "[frequencies]\nhessian = \"analytic-moving-grid\"\n",
        "",
    );
    let fd = |h: f64, label: &str| {
        run_cli_ok(
            &format!("v_{tag}_{label}"),
            WATER,
            xc,
            &format!("[frequencies]\nhessian = \"fd\"\ndelta = {h:e}\n"),
            "",
        )
    };
    let coarse = fd(H, "fd_h");
    let fine = fd(0.5 * H, "fd_h2");
    assert_eq!(moving.source, "analytic-moving-grid");
    assert_eq!(coarse.source, "finite-difference");
    let rich: Vec<f64> = fine
        .frequencies
        .iter()
        .zip(&coarse.frequencies)
        .map(|(f, c)| (4.0 * f - c) / 3.0)
        .collect();
    let rich_h: Vec<f64> = fine
        .hessian
        .iter()
        .zip(&coarse.hessian)
        .map(|(f, c)| (4.0 * f - c) / 3.0)
        .collect();
    let miss = max_abs_diff(&moving.frequencies, &rich);
    let miss_plain = max_abs_diff(&moving.frequencies, &fine.frequencies);
    let miss_h = max_abs_diff(&moving.hessian, &rich_h);
    eprintln!(
        "{xc} H2O/STO-3G: moving-grid {:?} cm^-1; FD(Richardson) {:?}; max |d nu| {miss:.3e} \
         cm^-1 (vs plain FD at h/2: {miss_plain:.3e}); max |d H| {miss_h:.3e} Ha/Bohr^2; FD \
         asymmetry {:.1e}",
        moving.frequencies, rich, fine.asymmetry
    );
    assert!(miss < TOL_CM1, "{xc}: {miss:.3e} cm^-1 >= {TOL_CM1}");
    assert!(
        miss_h < TOL_HESS,
        "{xc}: {miss_h:.3e} Ha/Bohr^2 >= {TOL_HESS}"
    );
}

#[test]
#[ignore = "validation: moving-grid KS Hessian"]
fn cli_moving_grid_frequencies_match_fd_lda() {
    case("LDA");
}

#[test]
#[ignore = "validation: moving-grid KS Hessian"]
fn cli_moving_grid_frequencies_match_fd_pbe() {
    case("PBE");
}
