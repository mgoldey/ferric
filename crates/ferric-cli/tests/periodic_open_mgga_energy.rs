//! Issue #498: the open-shell periodic SCAN / r2SCAN ENERGIES wired to the
//! CLI by it.
//!
//! * k-point ROKS (`method.kind = "rohf"` with `[cell] kmesh`; library
//!   `solve_kroks`, #582), dense-AFT J/K only.
//! * Gamma-point UKS and ROKS (no kmesh; library `gamma_rsh::gamma_uks_any` /
//!   `gamma_roks_any` routing to `dft::gamma_uks` / `rohf::gamma_roks`,
//!   #581 / #582), dense-AFT J/K only.
//! * k-point UKS on the RS-GDF J/K (library `solve_kuks`, #579), unsplit and
//!   with `[cell] range_split = true`.
//!
//! EXACTNESS ANCHOR (CLI vs library, BIT FOR BIT): the CLI energy is read at
//! full precision from the run log's `run_end` record (serde_json parses
//! floats correctly rounded since #489); the library is called on the same
//! cell, mesh, grid and configs (the CLI's config builders restated field by
//! field in `open_rsh_cli`), and the two must be identical to the last bit.
//! Fixtures: an H3 doublet in a 4 Bohr cubic cell, STO-3G or STO-3G plus a p
//! shell on every H (`H_SP`), exxdiv ewald, XC grid 20x50, 14 Bohr cutoff,
//! k-mesh 1x1x2.
//!
//! Each anchor also checks the library value against siblings a mis-routed
//! CLI would produce (the GGA PBE for SCAN, SCAN for r2SCAN, UKS for ROKS,
//! dense for RS-GDF): each differs by far more than one ulp, so a bit-equal
//! pass cannot come from the wrong route. The derivatives of these routes are
//! refused by name (`periodic_kpoint_open_shell.rs`,
//! `periodic_rsh_open_mgga_derivs.rs`); the k-point ROKS meta-GGA force and
//! stress are library work in progress (#586), not wired here.

mod open_rsh_cli;

use open_rsh_cli::*;

/// `|a - b| > 1e-6`: a sibling that a bit-equal pass could not come from.
fn distinct(tag: &str, what: &str, a: f64, b: f64) {
    eprintln!("{tag}: vs {what} {:.3e}", (a - b).abs());
    assert!((a - b).abs() > 1e-6, "{tag}: {what} sibling too close");
}

#[test]
fn kpoint_roks_scan_energy_equals_the_library_bit_for_bit() {
    let r = Run {
        sp: true,
        ..Run::new("kroks-scan-sp", Spin::Roks, "SCAN", "energy")
    };
    let e = energy_anchor(&r);
    distinct(r.tag, "PBE", e, lib_energy(&r.with("PBE", false)));
    let uks = Run {
        spin: Spin::Uks,
        ..r
    };
    distinct(r.tag, "UKS", e, lib_energy(&uks));
}

#[test]
fn kpoint_roks_r2scan_energy_equals_the_library_bit_for_bit() {
    let r = Run::new("kroks-r2scan", Spin::Roks, "r2SCAN", "energy");
    let e = energy_anchor(&r);
    distinct(r.tag, "SCAN", e, lib_energy(&r.with("SCAN", false)));
}

#[test]
fn gamma_uks_scan_energy_equals_the_library_bit_for_bit() {
    let r = Run {
        kpoints: false,
        ..Run::new("guks-scan", Spin::Uks, "SCAN", "energy")
    };
    let e = energy_anchor(&r);
    distinct(r.tag, "PBE", e, lib_energy(&r.with("PBE", false)));
}

#[test]
fn gamma_roks_r2scan_energy_equals_the_library_bit_for_bit() {
    let r = Run {
        kpoints: false,
        sp: true,
        ..Run::new("groks-r2scan-sp", Spin::Roks, "r2SCAN", "energy")
    };
    let e = energy_anchor(&r);
    distinct(r.tag, "SCAN", e, lib_energy(&r.with("SCAN", false)));
    let uks = Run {
        spin: Spin::Uks,
        ..r
    };
    distinct(r.tag, "UKS", e, lib_energy(&uks));
}

#[test]
fn kpoint_uks_scan_on_rsgdf_energy_equals_the_library_bit_for_bit() {
    let r = Run {
        rsgdf: true,
        ..Run::new("kuks-scan-rsgdf", Spin::Uks, "SCAN", "energy")
    };
    let e = energy_anchor(&r);
    let dense = lib_energy(&r.with("SCAN", false));
    distinct(r.tag, "dense J/K", e, dense);
    assert!((e - dense).abs() < 1e-3, "{}: RS-GDF vs dense", r.tag);
}

/// The same RS-GDF run with `[cell] range_split = true` (the opt-in split of
/// the RS-GDF short-range sums at the default lambda): a J-only construction
/// change for a pure meta-GGA, reachable since the RS-GDF energy is wired.
/// The split repartitions the same sums (measured 1.5e-14 Ha), so split and
/// unsplit agree to rounding but differ in the bits: a CLI that dropped the
/// key would equal the unsplit value and fail the anchor (mutant: the CLI's
/// `range_split_lambda` forced to `None`, killed).
#[test]
fn kpoint_uks_scan_on_range_split_rsgdf_energy_equals_the_library_bit_for_bit() {
    let r = Run {
        rsgdf: true,
        range_split: true,
        ..Run::new("kuks-scan-rsgdf-split", Spin::Uks, "SCAN", "energy")
    };
    let e = energy_anchor(&r);
    let unsplit = lib_energy(&Run {
        range_split: false,
        ..r
    });
    eprintln!("{}: split - unsplit {:.3e}", r.tag, e - unsplit);
    assert!((e - unsplit).abs() < 1e-10, "{}: split vs unsplit", r.tag);
    assert_ne!(
        e.to_bits(),
        unsplit.to_bits(),
        "{}: split not applied",
        r.tag
    );
}
