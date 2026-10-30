//! Weekly half of `periodic_kpoint_open_shell.rs`: the Monkhorst-Pack
//! (half-shifted 1x1x2, no Gamma point) energy anchor over every open-shell
//! driver (UKS PBE, ROKS PBE, ROHF), with the Gamma-centring mutant. It took
//! ~123 s on CI (runs 40957025137 / 40953725119, mean), above the 120 s
//! per-commit threshold. Per-commit canaries:
//! `cli_open_shell_scan_and_hse06_equal_library_solve_kuks` (UKS on the MP
//! mesh, bit-tight, centring mutant) and
//! `printed_energy_equals_the_library_energy_on_a_monkhorst_pack_mesh_rohf_canary`.
//! Shared harness: `kpoint_open_shell_cli/mod.rs`.

mod kpoint_open_shell_cli;
use kpoint_open_shell_cli::*;

/// A Monkhorst-Pack mesh (half-shifted at an even size, so no Gamma point)
/// reaches every open-shell driver: the CLI energy is the library energy on
/// the same mesh, and differs from the Gamma-centred mesh of the same size.
#[test]
#[ignore = "validation: k-point open-shell Monkhorst-Pack energy CLI == library (UKS, ROKS, ROHF); CI ~123 s"]
fn printed_energy_equals_the_library_energy_on_a_monkhorst_pack_mesh() {
    for (tag, method, f) in [
        ("mp-uks", Method::Uks, "PBE"),
        ("mp-roks", Method::Roks, "PBE"),
        ("mp-rohf", Method::Rohf, ""),
    ] {
        mp_mesh_energy_anchor(tag, method, f);
    }
}
