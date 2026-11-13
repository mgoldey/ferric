//! VALIDATION tier — the `laplace-sos-mp2` row of the validation page.
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_sos_mp2 \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's `laplace_sos_mp2` (SOS-MP2 = `c_os * E_OS`, with `E_OS` evaluated by
//! a minimax-Laplace denominator, Jung et al. JCP 121, 9793 (2004)) on five
//! systems against PySCF's own `mp.dfmp2.DFMP2` opposite-spin energy
//! `e_corr_os` at the SAME auxiliary basis (`scripts/gen_pyscf_sos_mp2_refs.py`
//! -> `testdata/reference/validation/sos_mp2/<system>_<basis>.json`). PySCF is
//! fed ferric's bundled orbital and aux basis JSON and geometry; its RHF is
//! exact four-centre J/K, as is ferric's `RhfConfig::default()`.
//!
//! # Where the bar comes from
//!
//! The Laplace substitution `1/Δ ≈ Σ_k w_k e^{-t_k Δ}` makes ferric's `E_OS`
//! differ from the exact-denominator DF-MP2 `E_OS` by a QUADRATURE error that is
//! a property of `n_quad` and the range `R = ymax/ymin` alone. That error is the
//! only quantity allowed to be large, so the bar on `E_OS` is derived from it:
//!
//! 1. Measure `|E_OS(n_quad) - E_OS^{PySCF}|` at `n_quad` = 3, 5, 7 on every
//!    system (table below).
//! 2. [`TOL_VS_EXACT`] is `10x` the largest `n_quad = 7` error (round to one
//!    digit), the same rule the Laplace-MP2 row uses.
//! 3. The `n_quad = 3` error must EXCEED [`TOL_VS_EXACT`] on every system, and
//!    the error must fall strictly over 3 -> 5 -> 7. Otherwise the bar could not
//!    tell 3 nodes from 7 and passing it would be arithmetic.
//! 4. The default-config SOS correlation energy carries the factor
//!    `c_os = 1.3`, so its bar is `1.3 * TOL_VS_EXACT`.
//!
//! The implementation error is separated from the quadrature error by comparing
//! ferric with the Laplace-MP2 row's quadrature-isolated numpy reference
//! (`testdata/reference/validation/laplace_mp2`, the same dense sum with the
//! IDENTICAL minimax nodes): [`TOL_VS_QUADRATURE`].
//!
//! MEASURED (release build, `OPENBLAS_NUM_THREADS=1`), `|E_OS - PySCF|` in Ha:
//!
//! ```text
//! system              n_quad=3    n_quad=5    n_quad=7
//! ch4 / cc-pVDZ       1.52e-5     1.03e-6     3.35e-9
//! h2o / cc-pVDZ       3.72e-4     7.67e-6     4.94e-8
//! alkane_8 / cc-pVDZ  1.06e-3     1.02e-5     4.00e-8
//! butadiene / cc-pVDZ 1.85e-3     1.70e-6     8.16e-8
//! h2o_aug / aug-cc-pVDZ 2.56e-4   6.17e-6     1.35e-7
//! ```
//!
//! The largest `n_quad = 7` error is 1.35e-7 (h2o / aug-cc-pVDZ), so
//! [`TOL_VS_EXACT`] = 2e-6 (10x, rounded up). The smallest `n_quad = 3` error is
//! 1.52e-5 (ch4), 7.6x above the bar. The other measured maxima, each with its
//! bar: MO vs AO 1.6e-14 (1e-10); ferric vs the quadrature-isolated reference
//! 1.6e-11 (1e-9); default-config `sos_corr` vs `1.3 E_OS` 1.75e-7 (= 1.3 x the
//! bar); E_OS vs PySCF E_SS (swap) at least 0.10 Ha away.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * PHYSICS: the error vs PySCF is quadrature error. It tracks `R`, falls by
//!   orders of magnitude from 3 -> 5 -> 7 nodes, and equals the Laplace-MP2
//!   row's `quadrature_error_e_os` (same nodes, same denominator).
//! * ARTIFACT: a defect in the shared assembly (frozen core, a scaling, the
//!   wrong spin component) is flat in `n_quad`; it survives against the
//!   quadrature-isolated reference. The two are distinguishable because the
//!   quadrature-isolated comparison is blind to quadrature error by
//!   construction.
//! * Controls: E_OS compared with PySCF's E_SS (a component swap) must miss;
//!   MO and AO algebra must agree (the exactness anchor, asserted first); the
//!   default `c_os` must be 1.3, so changing it moves `sos_corr` by `0.3 E_OS`.
//!
//! A missing reference JSON is a HARD failure.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::laplace::{laplace_sos_mp2, SosFormulation, SosMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/sos_mp2";
const LAPLACE_DIR: &str = "testdata/reference/validation/laplace_mp2";
const MOL_DIR: &str = "testdata/molecules/validation";

/// Geometry check (PySCF E_nuc from ferric's Bohr geometry).
const TOL_ENUC: f64 = 1e-9;
/// E_RHF vs PySCF, exact J/K on both sides.
const TOL_E_RHF: f64 = 1e-9;
/// MO ≡ AO: two assembly paths over the same nodes and integrals.
const TOL_MO_VS_AO: f64 = 1e-10;
/// ferric vs the quadrature-isolated numpy reference (implementation error).
const TOL_VS_QUADRATURE: f64 = 1e-9;
/// ferric `E_OS` at `n_quad = 7` vs PySCF DF-MP2 `E_OS`: quadrature error
/// (see the header for the derivation).
const TOL_VS_EXACT: f64 = 2e-6;

const N_QUADS: [usize; 3] = [3, 5, 7];

fn workspace_root() -> PathBuf {
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("testdata").is_dir() && p.join("crates").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ferric-mp2 manifest dir should be <root>/crates/ferric-mp2")
        .to_path_buf()
}

fn read_json(dir: &str, system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(dir)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/gen_pyscf_sos_mp2_refs.py — a missing reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn num(v: &Value, ptr: &str, ctx: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not a number"))
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<34} ferric {got:+.13} ref {want:+.13} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.13} vs reference {want:.13} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_miss(ctx: &str, what: &str, got: f64, want: f64, bar: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<34} |d| {d:.2e} (must exceed {bar:.0e})");
    assert!(
        d > bar,
        "{ctx}: negative control {what}: |d| {d:.2e} <= {bar:.0e}: the bar cannot tell these apart"
    );
}

fn check_system(system: &str, basis_name: &str, aux_name: &str) {
    let r = read_json(ROW_DIR, system, basis_name);
    let lap = read_json(LAPLACE_DIR, system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    assert_eq!(r["basis"].as_str(), Some(basis_name), "{ctx}: basis");
    assert_eq!(r["aux_basis"].as_str(), Some(aux_name), "{ctx}: aux basis");

    let stem = r["geometry_stem"]
        .as_str()
        .unwrap_or_else(|| panic!("{ctx}: geometry_stem missing"));
    let xyz = workspace_root().join(MOL_DIR).join(format!("{stem}.xyz"));
    let mol = Molecule::load_xyz(xyz.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    check_close(
        &ctx,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    let obs = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    };
    assert!(
        cfg.df_j_aux.is_none() && cfg.df_k_aux.is_none(),
        "exact J/K expected"
    );
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
    assert!(rhf.converged, "{ctx}: RHF not converged");
    check_close(
        &ctx,
        "E_RHF",
        rhf.energy,
        num(&r, "/e_rhf", &ctx),
        TOL_E_RHF,
    );

    let e_os_ref = num(&r, "/dfmp2/e_os", &ctx);
    let e_ss_ref = num(&r, "/dfmp2/e_ss", &ctx);

    let sos = |n_quad: usize, formulation: SosFormulation| {
        let c = SosMp2Config {
            c_os: 1.0,
            n_quad,
            ..Default::default()
        };
        laplace_sos_mp2(&mol, &obs, &dfbs, op, &rhf, &c, formulation)
            .unwrap_or_else(|e| panic!("{ctx}: laplace_sos_mp2(n_quad={n_quad}) failed: {e:?}"))
    };

    let mut e_os_by_n = Vec::new();
    for &n_quad in &N_QUADS {
        let mo = sos(n_quad, SosFormulation::Mo);
        // EXACTNESS ANCHOR first: MO == AO over the same nodes.
        let ao = sos(n_quad, SosFormulation::Ao);
        check_close(
            &ctx,
            &format!("MO vs AO (n_quad={n_quad})"),
            mo.e_os,
            ao.e_os,
            TOL_MO_VS_AO,
        );
        // IMPLEMENTATION error: the identical-node dense numpy reference.
        let row = lap["quadrature"]
            .as_array()
            .and_then(|a| {
                a.iter()
                    .find(|q| q["n_quad"].as_u64() == Some(n_quad as u64))
            })
            .unwrap_or_else(|| panic!("{ctx}: no laplace_mp2 quadrature row n_quad={n_quad}"));
        check_close(
            &ctx,
            &format!("vs quadrature ref (n_quad={n_quad})"),
            mo.e_os,
            num(row, "/e_os", &ctx),
            TOL_VS_QUADRATURE,
        );
        e_os_by_n.push((n_quad, mo.e_os));
    }

    // QUADRATURE error vs PySCF DF-MP2 E_OS.
    let errs: Vec<(usize, f64)> = e_os_by_n
        .iter()
        .map(|&(n, e)| (n, (e - e_os_ref).abs()))
        .collect();
    eprintln!(
        "{ctx}: |E_OS - PySCF| by n_quad: {}",
        errs.iter()
            .map(|(n, d)| format!("{n}:{d:.3e}"))
            .collect::<Vec<_>>()
            .join("  ")
    );
    let e7 = e_os_by_n[2].1;
    check_close(
        &ctx,
        "E_OS vs PySCF DF-MP2 (n=7)",
        e7,
        e_os_ref,
        TOL_VS_EXACT,
    );
    check_miss(
        &ctx,
        "E_OS(n=3) vs PySCF DF-MP2",
        e_os_by_n[0].1,
        e_os_ref,
        TOL_VS_EXACT,
    );
    for w in errs.windows(2) {
        assert!(
            w[0].1 > w[1].1,
            "{ctx}: |error| did not decrease from n_quad={} ({:.3e}) to n_quad={} ({:.3e})",
            w[0].0,
            w[0].1,
            w[1].0,
            w[1].1
        );
    }
    // Component swap must miss.
    check_miss(
        &ctx,
        "E_OS vs PySCF E_SS (swap)",
        e7,
        e_ss_ref,
        TOL_VS_EXACT,
    );

    // End to end at the DEFAULT config (c_os = 1.3, n_quad = 7).
    let d = SosMp2Config::default();
    let res = laplace_sos_mp2(&mol, &obs, &dfbs, op, &rhf, &d, SosFormulation::Mo)
        .unwrap_or_else(|e| panic!("{ctx}: default laplace_sos_mp2 failed: {e:?}"));
    let c_ref = num(&r, "/sos/c_os_default", &ctx);
    assert_eq!(res.c_os, c_ref, "{ctx}: default c_os");
    check_close(
        &ctx,
        "default sos_corr vs c_os*E_OS",
        res.sos_corr,
        num(&r, "/sos/sos_corr_default", &ctx),
        c_ref * TOL_VS_EXACT,
    );
    check_close(
        &ctx,
        "default total vs E_RHF + c_os*E_OS",
        res.total_energy,
        num(&r, "/sos/sos_total_default", &ctx),
        c_ref * TOL_VS_EXACT + TOL_E_RHF,
    );
}

#[test]
#[ignore = "validation: Laplace SOS-MP2"]
fn sos_mp2_h2o_ccpvdz_vs_pyscf_dfmp2() {
    check_system("h2o", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: Laplace SOS-MP2"]
fn sos_mp2_h2o_augccpvdz_vs_pyscf_dfmp2() {
    check_system("h2o_aug", "aug-cc-pvdz", "aug-cc-pvdz-rifit");
}

#[test]
#[ignore = "validation: Laplace SOS-MP2"]
fn sos_mp2_ch4_ccpvdz_vs_pyscf_dfmp2() {
    check_system("ch4", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: Laplace SOS-MP2"]
fn sos_mp2_butadiene_ccpvdz_vs_pyscf_dfmp2() {
    check_system("butadiene", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: Laplace SOS-MP2"]
fn sos_mp2_octane_ccpvdz_vs_pyscf_dfmp2() {
    check_system("alkane_8", "cc-pvdz", "cc-pvdz-ri");
}
