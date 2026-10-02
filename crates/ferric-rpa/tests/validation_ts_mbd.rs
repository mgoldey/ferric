//! VALIDATION tier — VALIDATION.md rows "TS C6" (153) and "MBD@TS" (156).
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-rpa --test validation_ts_mbd \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! References: `scripts/validation/gen_ts_mbd.py`, files in
//! `testdata/reference/validation/ts_mbd/` (one per system/basis plus
//! `anchors_model.json`).
//!
//! # What ferric computes (read from the loops, not the doc comments)
//!
//! TS (`mbd::ts_atom_params`, `ts_dynamic_polarizability`, `casimir_polder_c6`):
//! α_A = r_A α_free, C6_A = r_A² C6_free, ω_A = (4/3) C6_A/α_A², single-pole
//! α_A(iu) = α_A/(1+(u/ω_A)²), and EVERY pair C6_AB = (3/π) Σ_k w_k α_A α_B by
//! QUADRATURE. The TS combination rule is the exact value of that integral, so
//! ferric equals it up to the frequency quadrature only. ferric has no TS
//! pairwise ENERGY; nothing here compares one.
//!
//! MBD (`mbd::mbd_screen`, `mbd_dynamic_polarizability`, `mbd::coupled_qho_energy_plain_gg`):
//! PLAIN SCS screening with the Gaussian-damped dipole tensor (no Fermi
//! damping, no β, no R_vdw), widths from the DYNAMIC α at each node, full 3×3
//! per-atom block row sums; energy = plain MBD with 'dip,gg' damping from the
//! UNSCREENED static TS α/ω. That is libmbd `variant='scs'` / `variant='plain',
//! damping='dip,gg'`, NOT MBD@rsSCS (`pymbd.mbd_energy`). The rsSCS numbers are
//! a SCOPE control that these functions must miss here; ferric's MBD@rsSCS
//! (`dispersion::mbd_rsscs`) is validated against them in
//! `validation_mbd_rsscs.rs`.
//!
//! # Like-for-like
//!
//! Z, coordinates (Bohr), volume ratios, frequency nodes and weights are READ
//! from the reference; ferric recomputes only the model. α_free/C6_free are
//! ferric's own table; the generator parses it from `free_atom_ref.rs` and
//! refuses to write a system whose elements' pymbd `vdw_params` (TS columns)
//! differ. The MBD target is built twice by the generator — numpy on pymbd's
//! `T_erf_coulomb` and libmbd (Fortran) — which agree to ≤1.5e-14.
//!
//! Systems: H2O, CO, CH3OH (ratios from the committed Hirshfeld row) × cc-pVDZ,
//! def2-SVP; CH4, benzene (ratios from gen_hirshfeld.py's functions) × cc-pVDZ.
//! Two grids each: ferric's default (GL n=20, u0=0.5) and pymbd's 15-node grid
//! (with a u=0 node). The basis only produced the ratios.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * TS. If ferric is the stated model: same-grid C6 agrees to ~1e-14 and its
//!   own default grid reproduces the closed form to ~3e-14 (measured in the
//!   generator). If ratios are applied with the wrong power, or ω is mis-defined,
//!   the miss is O(r−1) ≈ 1e-2..5e-1 — except at r = 1, where the ratio-1 anchor
//!   is BLIND to the power (anchor blind spot); the molecular rows (r = 0.57-0.98)
//!   are what see it.
//! * MBD. ferric's `erf` is Abramowitz–Stegun 7.1.26 (|err| < 1.5e-7). If the
//!   only difference from the reference is that erf, ferric misses the
//!   exact-erf numbers by the generator's PREDICTED floor (α_scs 1.2e-7..3.2e-7,
//!   C6 pair ≤4.6e-7, energy 1.7e-7..5.1e-7 rel.) AND matches the A&S-erf
//!   variants to ~1e-13. If there is any other construction difference (σ from
//!   static α, a T sign, a missing η term), the A&S-erf comparison fails too.
//!   So the row's "1e-8" bar is NOT reachable against exact erf without
//!   replacing ferric's erf — that is a finding, not a tolerance to tune.
//!
//! # TOLERANCES (measured 2026-10-01, worst case over all 8 systems and both grids)
//!
//! | quantity | measured max | bar |
//! |---|---:|---:|
//! | TS α_eff, ω vs reference (rel) | 4.7e-16 | `TOL_TS_PARAMS` 1e-14 |
//! | TS C6 pair and molecular, same grid (rel) | 7.5e-16 | `TOL_TS_C6_SAME_GRID` 1e-12 |
//! | TS C6 pair, ferric default grid vs combination rule | 3.0e-14 | `TOL_TS_C6_CLOSED` 1e-12 |
//! | ferric default nodes vs reference nodes (abs) | — | `TOL_GRID` 1e-12 |
//! | MBD α_scs / C6 / energy vs EXACT erf (rel) | 5.1e-7 | `TOL_MBD_EXACT_ERF` 1e-6 |
//! | MBD α_scs iso / C6 mol / energy vs A&S erf (rel) | 8.1e-15 | `TOL_MBD_AS_ERF` 1e-11 |
//! | dimer closed form, large-R energy (cancellation) | — | `TOL_LONDON_E` 1e-6 |
//!
//! The exact-erf bar is only 2x the worst miss, but that miss is the deterministic
//! A&S erf error (no run-to-run noise); the A&S-erf comparison at 8e-15 shows the
//! rest of the construction is exact.
//!
//! # NEGATIVE CONTROLS / MUTATIONS
//!
//! * ALWAYS ON — ratio kick: ratios × (1 + 1e-4) must move TS C6 by more than
//!   `MUST_MISS` × the same-grid bar (the ratios are read and resolved).
//! * ALWAYS ON — screening active: MBD C6 must miss the unscreened TS C6 by more
//!   than `MUST_MISS` × the MBD bar (measured 5e-2..3.4e-1).
//! * ALWAYS ON — scope: ferric's MBD energy must miss MBD@rsSCS (β = 0.83) by
//!   > 50% (measured 97-100%): ferric is not MBD@rsSCS.
//! * MUTATION A — `r * r * c6_free` → `r * c6_free` in `ts_atom_params`: the
//!   ratio-1 anchor still passes (blind), every molecular TS/MBD row fails by >2%.
//! * MUTATION B — `(4.0 / 3.0)` → `(3.0 / 4.0)` in `ts_atom_params`: everything
//!   fails, including the ratio-1 anchor.
//! * MUTATION C — `eta * extra` → `-eta * extra` in `dipole_coupling_tensor`:
//!   the dimer anchors and the MBD rows fail at both erf variants.
//! * MUTATION D — in `mbd_screen`, compute σ from `per_atom_alpha[a][0]`
//!   (frozen at the first node): fails at every other node.
//! * EXPECTED FLIP — replacing ferric's A&S `erf` with an exact one makes the
//!   `*_as_erf` assertions fail and the exact-erf misses drop to ~1e-13; at that
//!   point delete the A&S assertions and tighten `TOL_MBD_EXACT_ERF`.

use std::path::{Path, PathBuf};

use ferric_rpa::config::QuadratureConfig;
use ferric_rpa::dispersion::free_atom_ref::ts_free_atom;
use ferric_rpa::dispersion::mbd::{coupled_qho_energy_plain_gg, mbd_screen, ts_atom_params};
use ferric_rpa::dispersion::{
    casimir_polder_c6, mbd_dynamic_polarizability, ts_dynamic_polarizability,
};
use ferric_rpa::quadrature::build_quadrature;
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
const GRIDS: &[&str] = &["ferric_default", "pymbd15"];

// Bars: see the module doc's TOLERANCES table.
const TOL_TS_PARAMS: f64 = 1e-14;
const TOL_TS_C6_SAME_GRID: f64 = 1e-12;
const TOL_TS_C6_CLOSED: f64 = 1e-12;
const TOL_GRID: f64 = 1e-12;
const TOL_MBD_EXACT_ERF: f64 = 1e-6;
const TOL_MBD_AS_ERF: f64 = 1e-11;
const TOL_LONDON_E: f64 = 1e-6;
const TOL_SINGLE_ATOM: f64 = 1e-14;
const MUST_MISS: f64 = 1000.0;
const RATIO_KICK: f64 = 1e-4;

/// Workspace root, found by walking up from the CWD; `CARGO_MANIFEST_DIR` is
/// only a fallback (wrong inside a nextest archive).
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

fn vec2(v: &Value, ptr: &str, ctx: &str) -> Vec<Vec<f64>> {
    (0..arr(v, ptr, ctx).len())
        .map(|i| vec1(v, &format!("{ptr}/{i}"), ctx))
        .collect()
}

fn t33(v: &Value, ptr: &str, ctx: &str) -> [[f64; 3]; 3] {
    let m = vec2(v, ptr, ctx);
    assert_eq!(m.len(), 3, "{ctx}: {ptr} is not 3x3");
    std::array::from_fn(|i| std::array::from_fn(|j| m[i][j]))
}

/// max |a - b| / max |b| over flattened values.
fn rel_max(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "rel_max: length mismatch");
    // f64::max drops a NaN when the other side is finite, so check first.
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
    let pos: Vec<[f64; 3]> = vec2(r, "/inputs/coords_bohr", ctx)
        .into_iter()
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let ratios = vec1(r, "/inputs/volume_ratios", ctx);
    assert_eq!(z.len(), pos.len(), "{ctx}: z/coords length");
    assert_eq!(z.len(), ratios.len(), "{ctx}: z/ratios length");
    Inputs { z, pos, ratios }
}

fn grid(r: &Value, g: &str, ctx: &str) -> (Vec<f64>, Vec<f64>) {
    (
        vec1(r, &format!("/grids/{g}/freqs"), ctx),
        vec1(r, &format!("/grids/{g}/weights"), ctx),
    )
}

fn iso(t: &[[f64; 3]; 3]) -> f64 {
    (t[0][0] + t[1][1] + t[2][2]) / 3.0
}

fn flat_c6(c6: &ndarray::Array2<f64>) -> Vec<f64> {
    c6.iter().copied().collect()
}

/// Isotropic static tensors (the screening and the iso C6 read only Tr/3).
fn iso_static(alpha: &[f64]) -> Vec<[[f64; 3]; 3]> {
    alpha
        .iter()
        .map(|&a| [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
        .collect()
}

/// Anisotropic static tensors with Tr/3 = α: only the SHAPE enters TS, so the
/// isotropic pair C6 must not see it.
fn aniso_static(alpha: &[f64]) -> Vec<[[f64; 3]; 3]> {
    alpha
        .iter()
        .map(|&a| {
            [
                [0.5 * a, 0.1 * a, 0.0],
                [0.1 * a, a, 0.0],
                [0.0, 0.0, 1.5 * a],
            ]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Row 153 — TS C6
// ---------------------------------------------------------------------------

/// Table read-back: ferric's `ts_free_atom` against pymbd's `vdw_params` TS
/// columns (stored in `anchors_model.json`). Z ≤ 18 must agree EXACTLY (both
/// are TS PRL 2009 Table I). Z = 19..54 are DIFFERENT SOURCES (ferric: Gould &
/// Bučko 2016 Table 2; pymbd: its own vdw-params.csv) — pinned here as a known,
/// reported discrepancy so a silent change of either table is caught.
#[test]
#[ignore = "validation: TS C6 — free-atom table read-back vs pymbd vdw_params"]
fn ts_free_atom_table_read_back() {
    let a = load("anchors_model");
    let ctx = "table";
    let rows = arr(&a, "/free_atom_table_comparison/rows", ctx);
    let mut mismatched = Vec::new();
    for (i, _) in rows.iter().enumerate() {
        let p = format!("/free_atom_table_comparison/rows/{i}");
        let z = num(&a, &format!("{p}/z"), ctx) as usize;
        let (af, cf, vf) = ts_free_atom(z).unwrap_or_else(|| panic!("ts_free_atom({z}) is None"));
        assert!(
            vf.is_none(),
            "Z={z}: vol_free must stay None (no table fallback)"
        );
        // Read-back of ferric's own table as the generator parsed it.
        assert_eq!(
            af,
            num(&a, &format!("{p}/ferric_alpha"), ctx),
            "Z={z} alpha read-back"
        );
        assert_eq!(
            cf,
            num(&a, &format!("{p}/ferric_c6"), ctx),
            "Z={z} C6 read-back"
        );
        let ap = num(&a, &format!("{p}/pymbd_alpha_ts"), ctx);
        let cp = num(&a, &format!("{p}/pymbd_c6_ts"), ctx);
        if z <= 18 {
            assert_eq!(af, ap, "Z={z}: alpha_free differs from pymbd TS");
            assert_eq!(cf, cp, "Z={z}: C6_free differs from pymbd TS");
        } else if af != ap || cf != cp {
            mismatched.push(z);
            eprintln!(
                "{ctx}: Z={z:2} alpha {af} vs pymbd {ap} ({:+.1e}), C6 {cf} vs {cp} ({:+.1e})",
                (af - ap) / ap,
                (cf - cp) / cp
            );
        }
    }
    let recorded: Vec<usize> = arr(&a, "/free_atom_table_comparison/mismatched_z", ctx)
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(mismatched, recorded, "Z>18 mismatch set changed");
}

/// EXACTNESS ANCHOR (written first): volume ratio 1 → α_eff = α_free and
/// ω = 4C6_free/(3α_free²) exactly; the Casimir–Polder C6_AA on ferric's own
/// default grid reproduces C6_free to the quadrature error; heteronuclear pairs
/// reproduce the TS combination rule.
#[test]
#[ignore = "validation: TS C6 — ratio-1 exactness anchor"]
fn ts_ratio_one_reproduces_free_atom_c6() {
    let a = load("anchors_model");
    let (freqs, weights) = build_quadrature(&QuadratureConfig::default());
    let ref_nodes = {
        let mut v = vec1(&a, "/ferric_default_grid/freqs", "anchors");
        v.sort_by(f64::total_cmp);
        v
    };
    let mut own = freqs.clone();
    own.sort_by(f64::total_cmp);
    check(
        "anchors",
        "default grid nodes vs reference",
        rel_max(&own, &ref_nodes),
        TOL_GRID,
    );

    let elements = ["H", "C", "N", "O"];
    let mut zs = Vec::new();
    for el in elements {
        let ctx = format!("ratio1/{el}");
        let z = num(&a, &format!("/ratio1/{el}/z"), &ctx) as usize;
        zs.push(z);
        let st = iso_static(&[1.0]);
        let p = ts_atom_params(&[z], &[1.0], &st).unwrap();
        check(
            &ctx,
            "alpha_eff vs alpha_free",
            rel(p[0].0, num(&a, &format!("/ratio1/{el}/alpha_free"), &ctx)),
            TOL_TS_PARAMS,
        );
        check(
            &ctx,
            "omega vs 4C6/(3a^2)",
            rel(p[0].1, num(&a, &format!("/ratio1/{el}/omega"), &ctx)),
            TOL_TS_PARAMS,
        );
        let dp = ts_dynamic_polarizability(&[z], &[1.0], &st, &freqs, &weights).unwrap();
        let c6 = casimir_polder_c6(&dp);
        check(
            &ctx,
            "C6_AA (default grid) vs C6_free",
            rel(
                c6.c6_iso_pair[(0, 0)],
                num(&a, &format!("/ratio1/{el}/c6_free"), &ctx),
            ),
            TOL_TS_C6_CLOSED,
        );
    }
    // All four at ratio 1 in one call: every pair against the combination rule.
    let ones = vec![1.0; zs.len()];
    let dp = ts_dynamic_polarizability(&zs, &ones, &iso_static(&ones), &freqs, &weights).unwrap();
    let c6 = casimir_polder_c6(&dp);
    for (i, ea) in elements.iter().enumerate() {
        for (j, eb) in elements.iter().enumerate().skip(i) {
            let key = format!("{ea}-{eb}");
            let want = num(&a, &format!("/c6_closed_form_pairs_ratio1/{key}"), "pairs");
            check(
                "ratio1 pairs",
                &format!("C6 {key} vs combination rule"),
                rel(c6.c6_iso_pair[(i, j)], want),
                TOL_TS_C6_CLOSED,
            );
        }
    }
}

/// Same inputs (Z, ratios, grid nodes/weights from the reference), ferric's TS
/// model vs the pymbd-sourced reference: per-atom α/ω, the pair C6 on both
/// stored grids, the molecular C6, and ferric's own default grid against the
/// closed-form combination rule.
#[test]
#[ignore = "validation: TS C6 — pymbd TS C6 from ferric's volume ratios"]
fn ts_c6_matches_pymbd_same_inputs() {
    let (own_f, own_w) = build_quadrature(&QuadratureConfig::default());
    for &(system, basis) in SYSTEMS {
        let ctx = format!("{system}/{basis}");
        let r = load(&format!("{system}_{basis}"));
        let inp = inputs(&r, &ctx);
        let n = inp.z.len();
        let a_ref = vec1(&r, "/ts_params/alpha_eff", &ctx);
        let st = aniso_static(&a_ref);
        let p = ts_atom_params(&inp.z, &inp.ratios, &st).unwrap();
        let a_f: Vec<f64> = p.iter().map(|x| x.0).collect();
        let w_f: Vec<f64> = p.iter().map(|x| x.1).collect();
        check(&ctx, "alpha_eff", rel_max(&a_f, &a_ref), TOL_TS_PARAMS);
        check(
            &ctx,
            "omega",
            rel_max(&w_f, &vec1(&r, "/ts_params/omega", &ctx)),
            TOL_TS_PARAMS,
        );

        let closed: Vec<f64> = vec2(&r, "/ts/c6_pair_closed_form", &ctx).concat();
        for &g in GRIDS {
            let (f, w) = grid(&r, g, &ctx);
            let dp = ts_dynamic_polarizability(&inp.z, &inp.ratios, &st, &f, &w).unwrap();
            let c6 = casimir_polder_c6(&dp);
            let want: Vec<f64> = vec2(&r, &format!("/ts/grids/{g}/c6_pair_cp"), &ctx).concat();
            assert_eq!(want.len(), n * n);
            check(
                &ctx,
                &format!("C6 pair [{g}]"),
                rel_max(&flat_c6(&c6.c6_iso_pair), &want),
                TOL_TS_C6_SAME_GRID,
            );
            check(
                &ctx,
                &format!("C6 molecular [{g}]"),
                rel(
                    c6.c6_molecular_iso,
                    num(&r, &format!("/ts/grids/{g}/c6_molecular_iso"), &ctx),
                ),
                TOL_TS_C6_SAME_GRID,
            );

            if g == "ferric_default" {
                // Ratio kick control: the ratios are read and resolved.
                let kicked: Vec<f64> = inp.ratios.iter().map(|x| x * (1.0 + RATIO_KICK)).collect();
                let dk = ts_dynamic_polarizability(&inp.z, &kicked, &st, &f, &w).unwrap();
                let ck = casimir_polder_c6(&dk);
                must_miss(
                    &ctx,
                    "ratio kick 1e-4",
                    rel_max(&flat_c6(&ck.c6_iso_pair), &want),
                    MUST_MISS * TOL_TS_C6_SAME_GRID,
                );
            }
        }
        // Production path: ferric's own default grid vs the closed form.
        let dp = ts_dynamic_polarizability(&inp.z, &inp.ratios, &st, &own_f, &own_w).unwrap();
        let c6 = casimir_polder_c6(&dp);
        check(
            &ctx,
            "C6 pair, own grid vs combination rule",
            rel_max(&flat_c6(&c6.c6_iso_pair), &closed),
            TOL_TS_C6_CLOSED,
        );
    }
}

// ---------------------------------------------------------------------------
// Row 156 — MBD@TS
// ---------------------------------------------------------------------------

/// EXACTNESS ANCHOR: one atom has no partner — screening returns the bare α at
/// every node and the MBD energy is zero.
#[test]
#[ignore = "validation: MBD@TS — single-atom exactness anchor"]
fn mbd_single_atom_is_unscreened() {
    let (f, w) = build_quadrature(&QuadratureConfig::default());
    for z in [1usize, 6, 7, 8] {
        let ctx = format!("single atom Z={z}");
        let st = iso_static(&[1.0]);
        let ts = ts_dynamic_polarizability(&[z], &[0.8], &st, &f, &w).unwrap();
        let scr = mbd_screen(&ts.per_atom, &[[0.0; 3]], &[0.0], &f).unwrap();
        let a: Vec<f64> = scr[0].iter().flatten().flatten().copied().collect();
        let b: Vec<f64> = ts.per_atom[0].iter().flatten().flatten().copied().collect();
        check(&ctx, "alpha_scs vs bare", rel_max(&a, &b), TOL_SINGLE_ATOM);
        let p = ts_atom_params(&[z], &[0.8], &st).unwrap();
        let e = coupled_qho_energy_plain_gg(&[[0.0; 3]], &[p[0].0], &[p[0].1]);
        check(&ctx, "E_MBD / omega", e.abs() / p[0].1, TOL_SINGLE_ATOM);
    }
}

/// EXACTNESS ANCHOR: a dimer on the z axis decouples into three 2×2 problems
/// (x, y, z), so the screened α and the coupled-QHO energy have closed forms
/// (mpmath, 50 digits, in `anchors_model.json`, cross-checked there against
/// the numpy construction to ≤5e-16 / ≤5e-13). Both erf variants are stored.
#[test]
#[ignore = "validation: MBD@TS — dimer closed-form anchor"]
fn mbd_dimer_matches_closed_form() {
    let a = load("anchors_model");
    let f = vec1(&a, "/ferric_default_grid/freqs", "anchors");
    for (d, _) in arr(&a, "/dimers", "anchors").iter().enumerate() {
        let p = format!("/dimers/{d}");
        let zz = vec1(&a, &format!("{p}/z"), "dimer");
        let r = num(&a, &format!("{p}/r_bohr"), "dimer");
        let ctx = format!("dimer Z={}-{} R={r}", zz[0], zz[1]);
        let al = vec1(&a, &format!("{p}/alpha"), &ctx);
        let om = vec1(&a, &format!("{p}/omega"), &ctx);
        let pos = [[0.0, 0.0, 0.0], [0.0, 0.0, r]];
        let input: Vec<Vec<[[f64; 3]; 3]>> = (0..2)
            .map(|i| {
                f.iter()
                    .map(|&u| {
                        let x = al[i] / (1.0 + (u / om[i]).powi(2));
                        [[x, 0.0, 0.0], [0.0, x, 0.0], [0.0, 0.0, x]]
                    })
                    .collect()
            })
            .collect();
        let scr = mbd_screen(&input, &pos, &al, &f).unwrap();
        for (tag, tol) in [("exact", TOL_MBD_EXACT_ERF), ("as_erf", TOL_MBD_AS_ERF)] {
            let mut got = Vec::new();
            let mut want = Vec::new();
            for k in 0..f.len() {
                let q = format!("{p}/screened_per_node/{tag}/{k}");
                got.extend([
                    scr[0][k][0][0],
                    scr[0][k][1][1],
                    scr[0][k][2][2],
                    scr[1][k][0][0],
                    scr[1][k][1][1],
                    scr[1][k][2][2],
                ]);
                let (ap, az) = (
                    num(&a, &format!("{q}/a_perp"), &ctx),
                    num(&a, &format!("{q}/a_par"), &ctx),
                );
                let (bp, bz) = (
                    num(&a, &format!("{q}/b_perp"), &ctx),
                    num(&a, &format!("{q}/b_par"), &ctx),
                );
                want.extend([ap, ap, az, bp, bp, bz]);
            }
            check(
                &ctx,
                &format!("alpha_scs diag [{tag}]"),
                rel_max(&got, &want),
                tol,
            );
            let key = if tag == "exact" {
                "energy"
            } else {
                "energy_as_erf"
            };
            let e = coupled_qho_energy_plain_gg(&pos, &al, &om);
            check(
                &ctx,
                &format!("E_MBD [{tag}]"),
                rel(e, num(&a, &format!("{p}/{key}"), &ctx)),
                tol,
            );
        }
        // Off-diagonal blocks vanish for a dimer on an axis.
        let off = (0..f.len())
            .flat_map(|k| {
                [
                    scr[0][k][0][1],
                    scr[0][k][0][2],
                    scr[0][k][1][2],
                    scr[1][k][0][2],
                ]
            })
            .fold(0.0_f64, |m, v| {
                assert!(v.is_finite(), "{ctx}: non-finite off-diagonal alpha_scs");
                m.max(v.abs())
            });
        check(
            &ctx,
            "off-diagonal alpha_scs / alpha",
            off / al[0],
            TOL_SINGLE_ATOM,
        );
    }
}

/// EXACTNESS ANCHOR (the identity TS and MBD share): at large separation the
/// coupled-QHO energy tends to −C6_AB/R⁶ with C6_AB the TS combination rule.
/// The stored mpmath values show E R⁶/(−C6) − 1 = O(R⁻⁶) (1.2e-5 at 15 Bohr,
/// 1.9e-7 at 30 Bohr for C–C); ferric must reproduce E(R) and therefore the limit.
#[test]
#[ignore = "validation: MBD@TS — London-limit anchor (MBD -> TS pair C6)"]
fn mbd_energy_tends_to_ts_pair_c6() {
    let a = load("anchors_model");
    for (d, _) in arr(&a, "/london", "anchors").iter().enumerate() {
        let p = format!("/london/{d}");
        let zz = vec1(&a, &format!("{p}/z"), "london");
        let r = num(&a, &format!("{p}/r_bohr"), "london");
        let ctx = format!("london Z={}-{} R={r}", zz[0], zz[1]);
        let za = zz[0] as usize;
        let zb = zz[1] as usize;
        let p_ab = ts_atom_params(&[za, zb], &[1.0, 1.0], &iso_static(&[1.0, 1.0])).unwrap();
        let al = [p_ab[0].0, p_ab[1].0];
        let om = [p_ab[0].1, p_ab[1].1];
        let e = coupled_qho_energy_plain_gg(&[[0.0, 0.0, 0.0], [0.0, 0.0, r]], &al, &om);
        check(
            &ctx,
            "E_MBD vs closed form",
            rel(e, num(&a, &format!("{p}/energy"), &ctx)),
            TOL_LONDON_E,
        );
        let c6 = num(&a, &format!("{p}/c6_ts_closed_form"), &ctx);
        let ratio = -e * r.powi(6) / c6;
        let want = num(&a, &format!("{p}/energy_times_r6_over_minus_c6"), &ctx);
        eprintln!("{ctx}: E R^6/(-C6_TS) = {ratio:.10} (closed form {want:.10})");
        check(
            &ctx,
            "E R^6/(-C6_TS) vs closed form",
            (ratio - want).abs(),
            TOL_LONDON_E,
        );
    }
}

/// Same inputs, ferric's MBD vs the pymbd/libmbd reference: full screened 3×3
/// per-atom tensors at every node of both grids, the pair (iso + aniso) and
/// molecular C6, and the MBD energy — each against the EXACT-erf target and,
/// for the iso/molecular/energy quantities, the A&S-erf variant (ferric's own
/// erf), which isolates the erf as the whole residual.
#[test]
#[ignore = "validation: MBD@TS — pymbd with identical inputs"]
fn mbd_matches_pymbd_same_inputs() {
    for &(system, basis) in SYSTEMS {
        let ctx = format!("{system}/{basis}");
        let r = load(&format!("{system}_{basis}"));
        let inp = inputs(&r, &ctx);
        let n = inp.z.len();
        let alpha = vec1(&r, "/ts_params/alpha_eff", &ctx);
        let omega = vec1(&r, "/ts_params/omega", &ctx);
        let st = iso_static(&alpha);
        for &g in GRIDS {
            let (f, w) = grid(&r, g, &ctx);
            let gp = format!("/mbd/grids/{g}");
            let dp =
                mbd_dynamic_polarizability(&inp.pos, &inp.z, &inp.ratios, &st, &f, &w).unwrap();
            let mut got = Vec::new();
            let mut want = Vec::new();
            let mut got_iso = Vec::new();
            for a in 0..n {
                for k in 0..f.len() {
                    let t = t33(&r, &format!("{gp}/alpha_scs/{a}/{k}"), &ctx);
                    got.extend(dp.per_atom[a][k].iter().flatten());
                    want.extend(t.iter().flatten());
                    got_iso.push(iso(&dp.per_atom[a][k]));
                }
            }
            check(
                &ctx,
                &format!("alpha_scs tensors [{g}] exact erf"),
                rel_max(&got, &want),
                TOL_MBD_EXACT_ERF,
            );
            let want_iso_as: Vec<f64> =
                vec2(&r, &format!("{gp}/alpha_scs_iso_as_erf"), &ctx).concat();
            check(
                &ctx,
                &format!("alpha_scs iso [{g}] A&S erf"),
                rel_max(&got_iso, &want_iso_as),
                TOL_MBD_AS_ERF,
            );

            let c6 = casimir_polder_c6(&dp);
            let want_iso: Vec<f64> = vec2(&r, &format!("{gp}/c6_pair_iso"), &ctx).concat();
            check(
                &ctx,
                &format!("C6 pair iso [{g}]"),
                rel_max(&flat_c6(&c6.c6_iso_pair), &want_iso),
                TOL_MBD_EXACT_ERF,
            );
            let mut got_an = Vec::new();
            let mut want_an = Vec::new();
            for a in 0..n {
                for b in 0..n {
                    got_an.extend(c6.c6_aniso_pair[a][b].iter().flatten());
                    want_an.extend(
                        t33(&r, &format!("{gp}/c6_pair_aniso/{a}/{b}"), &ctx)
                            .iter()
                            .flatten(),
                    );
                }
            }
            check(
                &ctx,
                &format!("C6 pair aniso [{g}]"),
                rel_max(&got_an, &want_an),
                TOL_MBD_EXACT_ERF,
            );
            check(
                &ctx,
                &format!("C6 molecular [{g}] exact erf"),
                rel(
                    c6.c6_molecular_iso,
                    num(&r, &format!("{gp}/c6_molecular_iso"), &ctx),
                ),
                TOL_MBD_EXACT_ERF,
            );
            check(
                &ctx,
                &format!("C6 molecular [{g}] A&S erf"),
                rel(
                    c6.c6_molecular_iso,
                    num(&r, &format!("{gp}/c6_molecular_iso_as_erf"), &ctx),
                ),
                TOL_MBD_AS_ERF,
            );

            // Control: screening is active (MBD C6 is not the TS C6).
            let ts = ts_dynamic_polarizability(&inp.z, &inp.ratios, &st, &f, &w).unwrap();
            let c6_ts = casimir_polder_c6(&ts);
            must_miss(
                &ctx,
                &format!("MBD vs unscreened TS C6 [{g}]"),
                rel_max(&flat_c6(&c6.c6_iso_pair), &flat_c6(&c6_ts.c6_iso_pair)),
                MUST_MISS * TOL_MBD_EXACT_ERF,
            );

            if g == "pymbd15" {
                // Third construction: libmbd (Fortran) variant='scs' — static
                // screened α at the u=0 node and C6_AA on its own 15-node grid.
                assert_eq!(f[0], 0.0, "{ctx}: pymbd15 grid must start at u=0");
                let lib_a = vec1(&r, "/mbd/libmbd/scs_static_alpha_iso", &ctx);
                let got0: Vec<f64> = (0..n).map(|a| iso(&dp.per_atom[a][0])).collect();
                check(
                    &ctx,
                    "static alpha_scs vs libmbd scs",
                    rel_max(&got0, &lib_a),
                    TOL_MBD_EXACT_ERF,
                );
                let lib_c6 = vec1(&r, "/mbd/libmbd/scs_c6_diag_pymbd15", &ctx);
                let diag: Vec<f64> = (0..n).map(|a| c6.c6_iso_pair[(a, a)]).collect();
                check(
                    &ctx,
                    "C6_AA vs libmbd scs",
                    rel_max(&diag, &lib_c6),
                    TOL_MBD_EXACT_ERF,
                );
            }
        }

        let e = coupled_qho_energy_plain_gg(&inp.pos, &alpha, &omega);
        check(
            &ctx,
            "E_MBD exact erf",
            rel(e, num(&r, "/mbd/energy", &ctx)),
            TOL_MBD_EXACT_ERF,
        );
        check(
            &ctx,
            "E_MBD A&S erf",
            rel(e, num(&r, "/mbd/energy_as_erf", &ctx)),
            TOL_MBD_AS_ERF,
        );
        check(
            &ctx,
            "E_MBD vs libmbd plain dip,gg",
            rel(e, num(&r, "/mbd/libmbd/plain_dipgg_energy", &ctx)),
            TOL_MBD_EXACT_ERF,
        );
        // Scope control: ferric is NOT MBD@rsSCS.
        let e_rs = num(&r, "/mbd/scope_rsscs/energy_pymbd", &ctx);
        must_miss(&ctx, "E_MBD vs MBD@rsSCS (beta 0.83)", rel(e, e_rs), 0.5);
    }
}
