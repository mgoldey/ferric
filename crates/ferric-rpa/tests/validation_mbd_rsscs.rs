//! VALIDATION tier — MBD@rsSCS dispersion energy and nuclear gradient
//! (`ferric_rpa::dispersion::mbd_rsscs`).
//!
//! ```text
//! cargo nextest run -p ferric-rpa --test validation_mbd_rsscs \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! References: `scripts/validation/gen_ts_mbd.py` → the `mbd/rsscs_targets`
//! block of `testdata/reference/validation/ts_mbd/<system>_<basis>.json` and
//! `r_vdw_table_comparison` in `anchors_model.json`.
//!
//! # Like-for-like
//!
//! Z, coordinates (Bohr) and volume ratios are READ from the reference.
//! α_free/C6_free/R_vdW^free are ferric's own tables; the generator parses
//! them from `free_atom_ref.rs` and refuses to write a reference whose pymbd
//! TS parameters differ for any element used (R_vdW: all Z = 1..54 equal).
//! The target is built twice — pymbd python (`screening` + `mbd_energy`) and
//! libMBD Fortran (`variant='rsscs'`) — which agree to ≤2.8e-12 (E),
//! ≤6.9e-16 (α₀^rsSCS), ≤1.5e-14 (C6^rsSCS) on every system and β.
//! β = 0.83 (PBE) and 0.85 (PBE0/HSE06), a = 6, libMBD's 15-node grid.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! If ferric is the stated model, the per-atom α₀/C6/R/ω^rsSCS agree to the
//! libMBD-vs-pymbd level (~1e-14) and E to the cancellation floor of
//! ½Σ√λ − (3/2)Σω (~eps·Σω/|E| ≈ 1e-12 rel). Artifacts and their predicted
//! size: a different frequency grid → E off by 7.5e-9..1.2e-7 rel (the
//! generator's 30/60-node sensitivity), the Abramowitz–Stegun erf → ~1e-7,
//! any construction difference (S_AB convention, which radii enter the
//! long-range Fermi function, SR = f instead of 1 − f) → percent level.
//!
//! # TOLERANCES (PROVISIONAL — set from the generator's measurements; tighten
//! to ~10x the first measured ferric-vs-reference maximum)
//!
//! | quantity | expected | bar |
//! |---|---:|---:|
//! | `mbd_freq_grid(15)` vs pymbd `freq_grid(15)` | ~1e-15 | `TOL_GRID` 1e-12 |
//! | front door (Z, ratios) vs reference TS α/C6/R | ~1e-16 | `TOL_TS` 1e-14 |
//! | α₀, C6, R_vdW, ω ^rsSCS vs pymbd and libMBD (rel) | ~1e-14 | `TOL_RSSCS_PARAMS` 1e-12 |
//! | E vs pymbd and libMBD (rel) | ~1e-12 | `TOL_RSSCS_E` 1e-10 |
//! | dE/dR (ratios fixed) vs libMBD `force=True` (abs, Ha/Bohr) | 6.2e-16 | `TOL_RSSCS_GRAD_ABS` 1e-14 |
//!
//! The gradient reference is libMBD's analytic gradient stored as dE/dR; the
//! generator decides libMBD's sign convention by a central FD of libMBD's own
//! energy and stores that check (`gradient_fd_check`). ferric's gradient is
//! the reverse-mode adjoint of its own forward pass (unit-tested against FD
//! in `mbd_rsscs.rs`), so this is an independent construction.
//!
//! # CONTROLS (always on)
//!
//! * β kick: β = 0.83 ↔ 0.85 must move E by > 5% (measured 10.6–13.9%).
//! * ratio kick ×(1 + 1e-4): E must move by > `MUST_MISS` × `TOL_RSSCS_E`.
//! * grid: `n_freq = 30` must reproduce the generator's 30-node energy to
//!   `TOL_RSSCS_E` AND miss the 15-node reference by > 10 × `TOL_RSSCS_E`
//!   (measured 7.5e-9..1.2e-7), so the 15-node match is a grid match, not luck.
//!
//! # MUTATIONS (run by hand; each must turn this file red)
//!
//! * A — `S_AB = β (R_A + R_B)` → `β R_A` in either Fermi call.
//! * B — `1.0 - fermi(..)` → `fermi(..)` in `rsscs_screen`.
//! * C — long-range Fermi with `params.r_vdw` instead of `r_vdw_rsscs`.
//! * D — `alpha_0_rsscs = a_dyn[0]` → `a_dyn[1]`.
//! * E — `erf_exact` → the A&S approximation (predicted ~1e-7 rel miss).
//! * F — `.cbrt()` → `.sqrt()` in `r_vdw_rsscs`.
//! * G — in `rsscs_backward`, drop the `+ g[(3 * b + j, 3 * a + i)]` (the
//!   (B,A) block) or the `- fermi` R-derivative in step 4 (gradient test).

use std::path::{Path, PathBuf};

use ferric_rpa::dispersion::free_atom_ref::ts_free_atom_r_vdw;
use ferric_rpa::dispersion::mbd_rsscs::{
    mbd_freq_grid, mbd_rsscs_energy, mbd_rsscs_energy_from_params, mbd_rsscs_gradient,
    MbdAtomParams, MbdRsscsConfig, MbdRsscsResult,
};
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/ts_mbd";
const SYSTEMS: &[(&str, &str)] = &[
    ("h2o", "cc-pvdz"),
    ("h2o", "def2-svp"),
    ("co", "cc-pvdz"),
    ("co", "def2-svp"),
    ("ch3oh", "cc-pvdz"),
    ("ch3oh", "def2-svp"),
    ("ch4", "cc-pvdz"),
    ("benzene", "cc-pvdz"),
];

const TOL_GRID: f64 = 1e-12;
const TOL_TS: f64 = 1e-14;
const TOL_RSSCS_PARAMS: f64 = 1e-12;
const TOL_RSSCS_E: f64 = 1e-10;
/// Absolute bar (Hartree/Bohr) on ferric dE/dR vs libMBD. Measured max
/// 6.2e-16 over 8 systems × 2 β; the negated-reference control misses by
/// ≥ 2.0e-5 (CH4).
const TOL_RSSCS_GRAD_ABS: f64 = 1e-14;
const MUST_MISS: f64 = 1000.0;
const RATIO_KICK: f64 = 1e-4;
const BETA_KICK_MIN: f64 = 0.05;

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
        .expect("ferric-rpa manifest dir should be <root>/crates/ferric-rpa")
        .to_path_buf()
}

fn load(name: &str) -> Value {
    let path = workspace_root().join(ROW_DIR).join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             `scripts/validation/gen_ts_mbd.py` — a missing reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn at<'a>(v: &'a Value, ptr: &str, ctx: &str) -> &'a Value {
    v.pointer(ptr)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing"))
}

fn num(v: &Value, ptr: &str, ctx: &str) -> f64 {
    at(v, ptr, ctx)
        .as_f64()
        .unwrap_or_else(|| panic!("{ctx}: {ptr} is not a number"))
}

fn arr<'a>(v: &'a Value, ptr: &str, ctx: &str) -> &'a Vec<Value> {
    at(v, ptr, ctx)
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: {ptr} is not an array"))
}

fn vec1(v: &Value, ptr: &str, ctx: &str) -> Vec<f64> {
    arr(v, ptr, ctx)
        .iter()
        .enumerate()
        .map(|(i, x)| {
            x.as_f64()
                .unwrap_or_else(|| panic!("{ctx}: {ptr}[{i}] is not a number"))
        })
        .collect()
}

/// max |a - b| / max |b|.
fn rel_max(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "rel_max: length mismatch");
    assert!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "rel_max: non-finite value"
    );
    let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let d = a
        .iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    d / scale
}

fn rel(a: f64, b: f64) -> f64 {
    assert!(a.is_finite() && b.is_finite(), "rel: non-finite value");
    (a - b).abs() / b.abs()
}

fn check(ctx: &str, what: &str, d: f64, tol: f64) {
    eprintln!("{ctx}: {what:<44} {d:.2e} (tol {tol:.0e})");
    assert!(d < tol, "{ctx}: {what}: {d:.2e} >= {tol:.0e}");
}

fn must_miss(ctx: &str, what: &str, d: f64, floor: f64) {
    eprintln!("{ctx}: CONTROL {what:<36} {d:.2e} (must exceed {floor:.0e})");
    assert!(d > floor, "{ctx}: control {what}: {d:.2e} <= {floor:.0e}");
}

struct Inputs {
    z: Vec<usize>,
    pos: Vec<[f64; 3]>,
    ratios: Vec<f64>,
}

fn inputs(r: &Value, ctx: &str) -> Inputs {
    let z: Vec<usize> = arr(r, "/inputs/z", ctx)
        .iter()
        .map(|x| x.as_u64().expect("z must be an integer") as usize)
        .collect();
    let pos: Vec<[f64; 3]> = arr(r, "/inputs/coords_bohr", ctx)
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let c = vec1(r, &format!("/inputs/coords_bohr/{i}"), ctx);
            [c[0], c[1], c[2]]
        })
        .collect();
    let ratios = vec1(r, "/inputs/volume_ratios", ctx);
    assert_eq!(z.len(), pos.len(), "{ctx}: z/coords length");
    assert_eq!(z.len(), ratios.len(), "{ctx}: z/ratios length");
    Inputs { z, pos, ratios }
}

fn run_ptr(r: &Value, beta: f64, ctx: &str) -> String {
    let runs = arr(r, "/mbd/rsscs_targets/runs", ctx);
    let i = runs
        .iter()
        .position(|x| x.get("beta").and_then(Value::as_f64) == Some(beta))
        .unwrap_or_else(|| panic!("{ctx}: no rsSCS reference run for beta={beta}"));
    format!("/mbd/rsscs_targets/runs/{i}")
}

fn compare_run(ctx: &str, r: &Value, p: &str, res: &MbdRsscsResult) {
    for (field, got) in [
        ("alpha_0_rsscs", &res.alpha_0_rsscs),
        ("c6_rsscs", &res.c6_rsscs),
        ("r_vdw_rsscs", &res.r_vdw_rsscs),
        ("omega_rsscs", &res.omega_rsscs),
    ] {
        check(
            ctx,
            &format!("{field} vs pymbd"),
            rel_max(got, &vec1(r, &format!("{p}/{field}"), ctx)),
            TOL_RSSCS_PARAMS,
        );
    }
    check(
        ctx,
        "alpha_0_rsscs vs libMBD",
        rel_max(
            &res.alpha_0_rsscs,
            &vec1(r, &format!("{p}/alpha_0_rsscs_libmbd"), ctx),
        ),
        TOL_RSSCS_PARAMS,
    );
    check(
        ctx,
        "c6_rsscs vs libMBD",
        rel_max(
            &res.c6_rsscs,
            &vec1(r, &format!("{p}/c6_rsscs_libmbd"), ctx),
        ),
        TOL_RSSCS_PARAMS,
    );
    check(
        ctx,
        "E vs pymbd",
        rel(res.energy, num(r, &format!("{p}/energy_pymbd"), ctx)),
        TOL_RSSCS_E,
    );
    check(
        ctx,
        "E vs libMBD",
        rel(res.energy, num(r, &format!("{p}/energy_libmbd"), ctx)),
        TOL_RSSCS_E,
    );
}

/// ferric's R_vdW table vs pymbd's `R_vdw(TS)` for Z = 1..54 (the generator
/// refuses to write a mismatch; this pins the read-back).
#[test]
#[ignore = "validation: MBD@rsSCS — R_vdW table read-back vs pymbd vdw_params"]
fn r_vdw_table_read_back() {
    let a = load("anchors_model");
    let ctx = "R_vdW table";
    let rows = arr(&a, "/r_vdw_table_comparison/rows", ctx);
    assert_eq!(rows.len(), 54, "{ctx}: expected Z=1..54");
    for i in 0..rows.len() {
        let p = format!("/r_vdw_table_comparison/rows/{i}");
        let z = num(&a, &format!("{p}/z"), ctx) as usize;
        let got = ts_free_atom_r_vdw(z).unwrap_or_else(|| panic!("Z={z}: no R_vdW"));
        assert_eq!(got, num(&a, &format!("{p}/pymbd_r_vdw_ts"), ctx), "Z={z}");
    }
}

/// `mbd_freq_grid(15)` reproduces libMBD/pymbd's grid node for node.
#[test]
#[ignore = "validation: MBD@rsSCS — frequency grid vs pymbd freq_grid(15)"]
fn freq_grid_matches_pymbd() {
    let r = load("h2o_cc-pvdz");
    let (f, w) = mbd_freq_grid(15);
    check(
        "grid",
        "nodes vs pymbd15",
        rel_max(&f, &vec1(&r, "/grids/pymbd15/freqs", "grid")),
        TOL_GRID,
    );
    check(
        "grid",
        "weights vs pymbd15",
        rel_max(&w, &vec1(&r, "/grids/pymbd15/weights", "grid")),
        TOL_GRID,
    );
}

/// ferric's MBD@rsSCS vs pymbd and libMBD with identical inputs, both β, plus
/// the β / ratio / grid controls.
#[test]
#[ignore = "validation: MBD@rsSCS — pymbd/libMBD with identical inputs"]
fn mbd_rsscs_matches_pymbd_and_libmbd() {
    for &(system, basis) in SYSTEMS {
        let r = load(&format!("{system}_{basis}"));
        let ctx0 = format!("{system}/{basis}");
        let inp = inputs(&r, &ctx0);
        let ref_params = MbdAtomParams {
            alpha_0: vec1(&r, "/ts_params/alpha_eff", &ctx0),
            c6: vec1(&r, "/ts_params/c6_eff", &ctx0),
            r_vdw: vec1(&r, "/mbd/rsscs_targets/r_vdw_ts", &ctx0),
        };
        let mut energies = Vec::new();
        for beta in [0.83, 0.85] {
            let ctx = format!("{ctx0} beta={beta}");
            let p = run_ptr(&r, beta, &ctx);
            let cfg = MbdRsscsConfig::with_beta(beta);
            // Front door: Z + volume ratios through ferric's own tables.
            let res = mbd_rsscs_energy(&inp.z, &inp.pos, &inp.ratios, &cfg).unwrap();
            check(
                &ctx,
                "TS alpha_0 (front door) vs reference",
                rel_max(&res.ts.alpha_0, &ref_params.alpha_0),
                TOL_TS,
            );
            check(
                &ctx,
                "TS C6 (front door) vs reference",
                rel_max(&res.ts.c6, &ref_params.c6),
                TOL_TS,
            );
            check(
                &ctx,
                "TS R_vdW (front door) vs reference",
                rel_max(&res.ts.r_vdw, &ref_params.r_vdw),
                TOL_TS,
            );
            compare_run(&ctx, &r, &p, &res);
            // Same model from the reference's own α/C6/R (no table involved).
            let res_p = mbd_rsscs_energy_from_params(&inp.pos, &ref_params, &cfg).unwrap();
            compare_run(&format!("{ctx} [from params]"), &r, &p, &res_p);
            energies.push(res.energy);

            if beta == 0.83 {
                // Ratio kick: the ratios are read and resolved.
                let kicked: Vec<f64> = inp.ratios.iter().map(|x| x * (1.0 + RATIO_KICK)).collect();
                let ek = mbd_rsscs_energy(&inp.z, &inp.pos, &kicked, &cfg)
                    .unwrap()
                    .energy;
                must_miss(
                    &ctx,
                    "ratio kick 1e-4",
                    rel(ek, num(&r, &format!("{p}/energy_pymbd"), &ctx)),
                    MUST_MISS * TOL_RSSCS_E,
                );
            }
            // Grid control: n_freq = 30 matches the generator's 30-node value
            // and misses the 15-node reference.
            let c30 = MbdRsscsConfig { n_freq: 30, ..cfg };
            let e30 = mbd_rsscs_energy(&inp.z, &inp.pos, &inp.ratios, &c30)
                .unwrap()
                .energy;
            check(
                &ctx,
                "E (n_freq=30) vs pymbd 30-node",
                rel(
                    e30,
                    num(&r, &format!("{p}/grid_sensitivity/30/energy"), &ctx),
                ),
                TOL_RSSCS_E,
            );
            must_miss(
                &ctx,
                "E (n_freq=30) vs 15-node reference",
                rel(e30, num(&r, &format!("{p}/energy_pymbd"), &ctx)),
                10.0 * TOL_RSSCS_E,
            );
        }
        must_miss(
            &ctx0,
            "beta 0.83 vs 0.85",
            rel(energies[0], energies[1]),
            BETA_KICK_MIN,
        );
    }
}

/// ferric's analytic MBD@rsSCS nuclear gradient (volume ratios fixed) vs
/// libMBD's (`force=True`, stored as dE/dR after the generator's FD
/// convention check), both β.
///
/// CONTROL: the same comparison against the NEGATED reference must miss by
/// far more than the bar — pins the dE/dR (not force) convention and proves
/// the reference is not ~0.
#[test]
#[ignore = "validation: MBD@rsSCS gradient — libMBD force=True with identical inputs"]
fn mbd_rsscs_gradient_matches_libmbd() {
    for &(system, basis) in SYSTEMS {
        let r = load(&format!("{system}_{basis}"));
        let ctx0 = format!("{system}/{basis}");
        let inp = inputs(&r, &ctx0);
        for beta in [0.83, 0.85] {
            let ctx = format!("{ctx0} beta={beta}");
            let p = run_ptr(&r, beta, &ctx);
            let kind = at(&r, &format!("{p}/gradient_fd_check/libmbd_array_is"), &ctx);
            eprintln!(
                "{ctx}: libMBD array convention {kind}, generator FD residual {:.2e}",
                num(
                    &r,
                    &format!("{p}/gradient_fd_check/max_abs_residual_vs_fd"),
                    &ctx
                )
            );
            let rows = arr(&r, &format!("{p}/gradient"), &ctx);
            assert_eq!(rows.len(), inp.z.len(), "{ctx}: gradient rows");
            let want: Vec<f64> = (0..rows.len())
                .flat_map(|i| {
                    let row = vec1(&r, &format!("{p}/gradient/{i}"), &ctx);
                    assert_eq!(row.len(), 3, "{ctx}: gradient row {i}");
                    row
                })
                .collect();
            let cfg = MbdRsscsConfig::with_beta(beta);
            let (g, _) = mbd_rsscs_gradient(&inp.z, &inp.pos, &inp.ratios, &cfg).unwrap();
            let got: Vec<f64> = g.d_positions.iter().flatten().copied().collect();
            let d = got
                .iter()
                .zip(&want)
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
            check(
                &ctx,
                "dE/dR vs libMBD (abs, Ha/Bohr)",
                d,
                TOL_RSSCS_GRAD_ABS,
            );
            let d_neg = got
                .iter()
                .zip(&want)
                .fold(0.0_f64, |m, (a, b)| m.max((a + b).abs()));
            must_miss(
                &ctx,
                "dE/dR vs NEGATED libMBD",
                d_neg,
                MUST_MISS * TOL_RSSCS_GRAD_ABS,
            );
            check(
                &ctx,
                "gradient's energy vs libMBD (rel)",
                rel(
                    g.result.energy,
                    num(&r, &format!("{p}/energy_libmbd"), &ctx),
                ),
                TOL_RSSCS_E,
            );
        }
    }
}
