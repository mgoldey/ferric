//! Issue #498 (open-shell RSH derivatives): the library routes wired to the
//! CLI by it, per-commit anchors. The rest of the routes (and the sibling
//! checks) are `validation_periodic_open_rsh_derivs.rs`; the shared harness
//! is `open_rsh_cli/mod.rs`.
//!
//! * G2, the k-point ROKS range-separated FORCE (`kpoint_roks_gradient`).
//! * G4, the k-point open-shell range-separated STRESS (`kpoint_uks_stress`).
//! * G3, the Gamma ROKS range-separated force (`gamma_roks_any_keep` +
//!   `gamma_roks_rsh_gradient`), dense AFT.
//!
//! The RS-GDF routes (G1: the k-point UKS / ROKS force and stress on
//! RS-GDF; G5: the Gamma RKS / UKS / ROKS force on RS-GDF) are anchored in
//! the validation twin only: each of those CLI + library pairs takes ~50 s.
//!
//! EXACTNESS ANCHOR (CLI vs library, BIT FOR BIT): the CLI energy, gradient
//! (`task = "optimize"` with zero optimizer steps: the gradient of the input
//! geometry) and stress tensor are read from the run log's `run_end` record;
//! the library is called on the SAME cell with the CLI's configs restated.
//!
//! INDEPENDENT CONSTRUCTION (per commit, for a NEW route): the CLI force and
//! one stress component against a central finite difference of the CLI's OWN
//! energies (k-point ROKS HSE06 force; k-point UKS HSE06 `sigma_zz` from a
//! uniaxial strain of the lattice and the atoms). Artifact hypothesis: a
//! wiring bug that differentiates the wrong density or spin (alpha density
//! passed as beta, the UKS force on a ROKS state, a dropped stress term)
//! misses the FD by the size of the spin polarization or of the term (>=
//! 1e-3 Ha/Bohr here; the HSE06 vs CAM-B3LYP force gap alone is 5e-3),
//! while a correct one misses only by the FD floor (SCF noise / h + h^2
//! remainder, measured below).
//!
//! Measured (2026-10-26): every anchor bit-equal. FD: k ROKS HSE06 force
//! atom 1 z 6.6e-10 (h = 1e-4, |g| 0.369; bar 1e-7); k UKS HSE06 sigma_zz
//! 7.5e-10 Hartree/Bohr^3 (h = 1e-4, |sigma_zz| 1.04e-2; bar 1e-8). Sibling
//! gaps (HSE06 vs CAM-B3LYP, RS-GDF vs dense) are in the validation twin.
//! The file runs in ~90 s on 3 threads (each test is one or three CLI runs
//! of ~10-20 s plus a library run).
//!
//! Mutation ledger (CLI source mutated, rebuilt, this file run, reverted),
//! all KILLED:
//! * `k_force_ro` hands the ROKS force a state whose beta density (and beta
//!   / effective Fock) is the alpha one: both ROKS force tests fail (the
//!   library's Fock guard refuses it, mismatch 1.27e-1).
//! * `k_force_ro` returns the force times (1 + 1e-6): the anchor fails on the
//!   bits and the FD misses by 3.7e-7 (bar 1e-7).
//! * `k_stress_uks` returns the transposed tensor: the stress anchor fails
//!   on the bits (the tensor is unsymmetrised); the zz FD is blind to it.
//! * `k_stress_uks` drops the stress (`sigma: None`): the anchor and the FD
//!   fail (no tensor in `run_end`).
//! * the Gamma ROKS route ignores the kept exchange (native
//!   `gamma_roks_gradient_rsgdf`, run against the RS-GDF anchor now in the
//!   validation twin): the anchor fails (the native force refuses a
//!   range-separated functional).
//! * the library gate accepts every range-separated hybrid (no validated
//!   check): the refusal test fails (wB97X runs).

mod open_rsh_cli;
use open_rsh_cli::*;

#[test]
fn kpoint_roks_hse06_force_equals_the_library_bit_for_bit() {
    let r = Run::new("kroks-hse-f", Spin::Roks, "HSE06", "optimize");
    let c = cli_force(&r);
    assert_force_bit_equal(r.tag, &c, lib_k_force(Spin::Roks, "HSE06", &c.1, false));
}

/// Central FD of the CLI's own k-point ROKS HSE06 energy along atom 1 z (the
/// largest force component) against the CLI force at the same geometry.
#[test]
fn kpoint_roks_hse06_force_matches_fd_of_the_cli_energy() {
    const H: f64 = 1e-4;
    let r = Run::new("kroks-hse-fd", Spin::Roks, "HSE06", "optimize");
    let (_, _, g) = cli_force(&r);
    let e_at = |dz: f64, tag: &str| {
        let mut at = H3;
        at[1][2] += dz;
        cli_energy(&Run {
            tag,
            atoms: Some(&at),
            ..Run::new(tag, Spin::Roks, "HSE06", "energy")
        })
    };
    let fd = (e_at(H, "kroks-fd-p") - e_at(-H, "kroks-fd-m")) / (2.0 * H);
    let miss = (fd - g[5]).abs();
    eprintln!(
        "k ROKS HSE06 dE/dz(atom 1): CLI {:.12e} FD {fd:.12e} miss {miss:.2e}",
        g[5]
    );
    assert!(g[5].abs() > 1e-2, "trivial component");
    assert!(miss < 1e-7, "FD miss {miss:e}");
}

#[test]
fn kpoint_uks_hse06_stress_equals_the_library_bit_for_bit() {
    let r = Run::new("kuks-hse-s", Spin::Uks, "HSE06", "stress");
    let c = cli_stress(&r);
    assert_stress_bit_equal(r.tag, &c, lib_k_stress(Spin::Uks, "HSE06", false));
}

/// `sigma_zz = (1/Omega) dE/d(eps_zz)`: central FD of the CLI's own energy
/// under a uniaxial strain (lattice c and every atom z scaled by `1 + eps`).
#[test]
fn kpoint_uks_hse06_stress_zz_matches_fd_of_the_cli_energy() {
    const H: f64 = 1e-4;
    let r = Run::new("kuks-hse-sfd", Spin::Uks, "HSE06", "stress");
    let (_, s) = cli_stress(&r);
    let e_at = |eps: f64, tag: &str| {
        cli_energy(&Run {
            strain_zz: eps,
            ..Run::new(tag, Spin::Uks, "HSE06", "energy")
        })
    };
    let omega = A_BOHR.powi(3);
    let fd = (e_at(H, "kuks-sfd-p") - e_at(-H, "kuks-sfd-m")) / (2.0 * H) / omega;
    let miss = (fd - s[8]).abs();
    eprintln!(
        "k UKS HSE06 sigma_zz: CLI {:.12e} FD {fd:.12e} miss {miss:.2e}",
        s[8]
    );
    assert!(s[8].abs() > 1e-4, "trivial component");
    assert!(miss < 1e-8, "FD miss {miss:e}");
}

#[test]
fn gamma_roks_cam_b3lyp_force_equals_the_library_bit_for_bit() {
    let r = Run {
        kpoints: false,
        ..Run::new("groks-cam-f", Spin::Roks, CAM, "optimize")
    };
    let c = cli_force(&r);
    assert_force_bit_equal(r.tag, &c, lib_gamma_force(Spin::Roks, CAM, &c.1, false));
}

/// What stays refused, by name, before any integral: the derivatives of an
/// unvalidated range-separated hybrid (wB97X) on the newly wired routes. The
/// CLI has no Gamma stress at all (`[cell] stress` needs `kmesh`).
#[test]
fn unvalidated_open_shell_rsh_derivatives_are_refused_by_name() {
    let wb = "HYB_GGA_XC_WB97X";
    let msg = "HYB_GGA_XC_WB97X: the range-separated hybrid force and stress are validated for \
               HSE06 / CAM-B3LYP (HYB_GGA_XC_CAM_B3LYP) only; HYB_GGA_XC_WB97X runs as an \
               energy";
    for (tag, spin, task, kpoints, rsgdf, prefix) in [
        (
            "wb-kroks-f",
            Spin::Roks,
            "optimize",
            true,
            false,
            "method.task = \"optimize\" with [cell] kmesh: ",
        ),
        (
            "wb-kuks-gdf-f",
            Spin::Uks,
            "optimize",
            true,
            true,
            "method.task = \"optimize\" with [cell] kmesh: ",
        ),
        (
            "wb-kroks-s",
            Spin::Roks,
            "stress",
            true,
            true,
            "[cell] stress = true: ",
        ),
        (
            "wb-groks-f",
            Spin::Roks,
            "optimize",
            false,
            true,
            "method.task = \"optimize\" with [cell]: ",
        ),
    ] {
        let e = refused(&Run {
            kpoints,
            rsgdf,
            ..Run::new(tag, spin, wb, task)
        });
        assert!(e.contains(prefix) && e.contains(msg), "{tag}: {e}");
    }
    let e = refused(&Run {
        kpoints: false,
        ..Run::new("g-stress", Spin::Uks, "HSE06", "stress")
    });
    assert!(
        e.contains(
            "[cell] stress = true needs [cell] kmesh: the CLI wires the k-point stress only"
        ),
        "{e}"
    );
}
