//! VALIDATION tier — row "SCS-MP2(2terfc)", integral half.
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-integrals --test validation_terfc_integrals \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's terf and terfc 3-index `(P|μν)` and 2-index `(P|Q)` integrals
//! (`threeindex::eri3_tensor`, `threeindex::coulomb_metric_2c` with
//! `Operator::terf(r0)` / `Operator::terfc(r0)`: the hand-rolled
//! McMurchie–Davidson driver plus the committed `terf-tables/*.bin`) against
//! a Fourier-space construction, `scripts/validation/gen_terfc_integrals.py`:
//! `(P|K|μν) = (2π)⁻³∫d³k K̂(k) Re[χ̂_P(k)* ρ̂_μν(k)]` with PySCF's analytic
//! Gaussian transforms (`ft_ao`, `ft_aopair`), K̂_terf = (4π/k²)·e^{−k²/4w²}·
//! cos(k r₀), Gauss–Legendre × Lebedev k-grid; terfc = PySCF analytic Coulomb
//! − that terf. Nothing of ferric's MD recursion, Hermite sign conventions,
//! tables, series or asymptotic branches is shared. H2O and NH3 / cc-pVDZ +
//! cc-pVDZ-RI, H2O / aug-cc-pVDZ + aug-cc-pVDZ-RIFIT (both aux sets carry f),
//! r₀ = 0.75 and 1.05 Å (× 1.8897259886, ferric's conversion).
//!
//! Before writing anything the generator requires (all recorded in the JSON
//! under `checks`): the radial inverse of K̂_terf reproduces the real-space
//! kernel at 20 radii (≤ 8.4e-15); the same k-grid with the plain erf kernel
//! reproduces PySCF `with_range_coulomb(w)` `int3c2e`/`int2c2e` to ≤ 1e-10 on
//! the production rung of a recorded grid ladder; terf moves ≤ 1e-10 between
//! the last two rungs; r₀ → 0 at fixed w reproduces erf; r₀ → ∞ on the
//! curvature constraint approaches C(r₀)·N_P·S_μν monotonically (measured
//! ~16× per doubling of r₀, i.e. O(1/r₀⁴) relative: the O(w²r²) term of the
//! kernel vanishes because w·r₀ = 1/√2 is exactly the zero-curvature point).
//!
//! Stored per tensor: the largest-|value| element of every shell-class (l_P,
//! l_μ ≥ l_ν for 3-index; l_P ≥ l_Q for 2-index), a seeded random subset of a
//! few hundred elements with full indices (ferric AO order), and the
//! full-tensor sum of squares and max |value|.
//!
//! # Physics hypothesis vs artifact hypothesis (Experimental Protocol)
//!
//! * If ferric is right, every class agrees at the table-interpolation floor,
//!   with no dependence on angular momentum.
//! * ARTIFACT HYPOTHESIS: a shared MD error in ferric — the ket-side
//!   (−1)^(t+u+v) Hermite sign, an l > 0 normalization, an index slip — shows
//!   up in the p/d/f classes and NOT in ss. That is why the existing checks
//!   cannot see it: `terfc_base_validation.rs` M2 compares (ss|ss) only, and
//!   `eri4_terfc_split.rs` (terf + terfc = Coulomb) is blind to anything both
//!   pieces share through `compute_cart_eri3`. The per-class element-wise
//!   comparison here is the test that can; a table-interpolation error would
//!   instead show in every class, ss included.
//! * If the AO order were wrong (harness, not ferric): the Coulomb samples,
//!   compared FIRST against ferric's libint Coulomb at the same indices, fail.
//!
//! # TOLERANCES
//!
//! Absolute, per element (terfc = Coulomb − terf nearly cancels at short
//! range, so relative error is the wrong measure). See each const for the
//! measured maximum it is set from.
//!
//! # NEGATIVE CONTROLS / MUTATIONS
//!
//! Always-on: ferric terfc at r₀ = 0.75 Å must MISS the 1.05 Å reference, and
//! ferric erfc at ω = 1/(r₀√2) must MISS terfc(r₀) (same curvature, different
//! tails). Each by ≥ [`MUST_MISS`], with the premise (the two references
//! differ) asserted first. The mutation ledger is at the end of this file.
//!
//! A missing JSON or a missing table directory is a HARD failure.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Once;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::{coulomb_metric_2c, eri3_tensor};
use ndarray::{Array2, Array3};
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/scs_mp2_2terfc";
const MOL_DIR: &str = "testdata/molecules/validation";
const CASES: &[(&str, &str, &str)] = &[
    ("h2o", "cc-pvdz", "cc-pvdz-ri"),
    ("nh3", "cc-pvdz", "cc-pvdz-ri"),
    ("h2o", "aug-cc-pvdz", "aug-cc-pvdz-rifit"),
];

/// ferric libint Coulomb vs PySCF Coulomb at the stored indices (the AO-order
/// anchor). Measured max 8.9e-14 (3-index), 4.7e-13 (2-index).
const TOL_COULOMB_3C: f64 = 1e-12;
const TOL_COULOMB_2C: f64 = 5e-12;
/// terf / terfc 3-index, element-wise vs the k-space reference. Measured max
/// 1.4e-13 (NH3/cc-pVDZ terfc 0.75 A), every class. Spec target 1e-9.
const TOL_INT_3C: f64 = 2e-12;
/// terf / terfc 2-index, element-wise. Measured max 2.6e-12 (H2O/aug, 0.75 A);
/// the reference's own erf-anchor floor on (P|Q) is 6.7e-13..3.4e-12 (JSON
/// `checks.kspace_grid`), so this measures the reference as much as ferric.
const TOL_INT_2C: f64 = 3e-11;
/// Full-tensor sum of squares, relative. Measured max 9.0e-14.
const TOL_SUMSQ_REL: f64 = 1e-12;
/// ferric's libint erfc vs PySCF erfc at the same indices (the erfc control's
/// own premise). Measured max 2.1e-12 (H2O/aug).
const TOL_ERFC_PREMISE: f64 = 2e-11;
const MUST_MISS: f64 = 1000.0 * TOL_INT_2C;

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
        .expect("manifest dir should be <root>/crates/ferric-integrals")
        .to_path_buf()
}

static TABLES: Once = Once::new();

/// The table engine reads `FERRIC_TERF_TABLE_DIR` only. Point it at the
/// committed `terf-tables/` when unset (the CI validation job does not set
/// it), and FAIL — never skip — when no table set can be found.
fn ensure_tables() {
    TABLES.call_once(|| {
        let probe = "16_4_2.bin";
        if let Ok(d) = std::env::var("FERRIC_TERF_TABLE_DIR") {
            assert!(
                Path::new(&d).join(probe).is_file(),
                "FERRIC_TERF_TABLE_DIR={d} has no {probe}; a validation row never skips"
            );
            return;
        }
        let repo = workspace_root().join("terf-tables");
        assert!(
            repo.join(probe).is_file(),
            "terfc tables not found: FERRIC_TERF_TABLE_DIR is unset and {} has no {probe}",
            repo.display()
        );
        std::env::set_var("FERRIC_TERF_TABLE_DIR", &repo);
    });
}

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_terfc_integrals.py — a missing reference is a failure",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn f64s(v: &Value, ctx: &str) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("{ctx}: not an array"))
        .iter()
        .map(|x| x.as_f64().unwrap_or_else(|| panic!("{ctx}: not a number")))
        .collect()
}

fn usizes(v: &Value, ctx: &str) -> Vec<usize> {
    v.as_array()
        .unwrap_or_else(|| panic!("{ctx}: not an array"))
        .iter()
        .map(|x| x.as_u64().unwrap_or_else(|| panic!("{ctx}: not an index")) as usize)
        .collect()
}

fn l_of_dim(d: usize) -> usize {
    match d {
        1 => 0,
        3 => 1,
        5 => 2,
        7 => 3,
        9 => 4,
        _ => panic!("unexpected (Cartesian?) shell dimension {d}"),
    }
}

/// Angular momentum of every AO in ferric's order, from ferric's own shells.
fn ao_l(b: &PreparedBasis) -> Vec<usize> {
    b.shell_dims()
        .iter()
        .flat_map(|&d| std::iter::repeat_n(l_of_dim(d), d))
        .collect()
}

const L: [char; 5] = ['s', 'p', 'd', 'f', 'g'];

struct Case {
    r: Value,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    idx3: Vec<[usize; 3]>,
    idx2: Vec<[usize; 2]>,
    class3: Vec<String>,
    class2: Vec<String>,
}

fn load(system: &str, basis_name: &str, auxbasis: &str) -> Case {
    ensure_tables();
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz(xyz.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let enuc = r["nuclear_repulsion"].as_f64().unwrap();
    assert!(
        (mol.nuclear_repulsion() - enuc).abs() < 1e-9,
        "{ctx}: E_nuc {} vs {enuc}",
        mol.nuclear_repulsion()
    );
    let obs = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(auxbasis).unwrap()).unwrap();
    assert_eq!(
        obs.nbasis(),
        r["nao"].as_u64().unwrap() as usize,
        "{ctx}: nao"
    );
    assert_eq!(
        dfbs.nbasis(),
        r["naux"].as_u64().unwrap() as usize,
        "{ctx}: naux"
    );
    let ints = &r["integrals"];
    // The class labels in the JSON come from the generator's reading of the
    // basis JSON; re-derive them from ferric's OWN shells and assert equality
    // (an index, not a word).
    let lobs = ao_l(&obs);
    let laux = ao_l(&dfbs);
    assert_eq!(
        lobs,
        usizes(&ints["obs_l_ferric_order"], &ctx),
        "{ctx}: orbital AO l"
    );
    assert_eq!(
        laux,
        usizes(&ints["aux_l_ferric_order"], &ctx),
        "{ctx}: aux AO l"
    );
    let idx3: Vec<[usize; 3]> = ints["eri3_indices_p_mu_nu"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            let v = usizes(t, &ctx);
            [v[0], v[1], v[2]]
        })
        .collect();
    let idx2: Vec<[usize; 2]> = ints["eri2_indices_p_q"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            let v = usizes(t, &ctx);
            [v[0], v[1]]
        })
        .collect();
    let class3 = idx3
        .iter()
        .map(|&[p, m, n]| {
            let (a, b) = if lobs[m] >= lobs[n] {
                (lobs[m], lobs[n])
            } else {
                (lobs[n], lobs[m])
            };
            format!("{}{}{}", L[laux[p]], L[a], L[b])
        })
        .collect::<Vec<_>>();
    let class2 = idx2
        .iter()
        .map(|&[p, q]| {
            let (a, b) = if laux[p] >= laux[q] {
                (laux[p], laux[q])
            } else {
                (laux[q], laux[p])
            };
            format!("{}{}", L[a], L[b])
        })
        .collect::<Vec<_>>();
    // Every class the generator found must be among the sampled classes, and
    // the d/f classes the artifact hypothesis needs must be present.
    for c in ints["eri3_classes"].as_array().unwrap() {
        assert!(
            class3.iter().any(|x| x == c.as_str().unwrap()),
            "{ctx}: class {c}"
        );
    }
    for need in ["sss", "spp", "pps", "dds", "fpp", "fdd"] {
        assert!(
            class3.iter().any(|x| x == need),
            "{ctx}: 3-index samples lack class {need}"
        );
    }
    Case {
        r,
        obs,
        dfbs,
        idx3,
        idx2,
        class3,
        class2,
    }
}

fn at3(t: &Array3<f64>, idx: &[[usize; 3]]) -> Vec<f64> {
    idx.iter().map(|&[p, m, n]| t[(p, m, n)]).collect()
}

fn at2(t: &Array2<f64>, idx: &[[usize; 2]]) -> Vec<f64> {
    idx.iter().map(|&[p, q]| t[(p, q)]).collect()
}

/// Max |got - want| per class; returns the overall max.
fn compare(ctx: &str, classes: &[String], got: &[f64], want: &[f64], tol: f64) -> f64 {
    assert_eq!(got.len(), want.len(), "{ctx}: sample count");
    let mut per: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    for ((c, g), w) in classes.iter().zip(got).zip(want) {
        let e = per.entry(c.as_str()).or_insert((0.0, 0.0));
        e.0 = e.0.max((g - w).abs());
        e.1 = e.1.max(w.abs());
    }
    let mut worst = 0.0_f64;
    let mut bad = Vec::new();
    for (c, (err, mag)) in &per {
        eprintln!("{ctx}: class {c:<4} max|d| {err:.2e}  (max|ref| {mag:.2e})");
        worst = worst.max(*err);
        if *err >= tol {
            bad.push(format!("{c}: {err:.2e}"));
        }
    }
    eprintln!("{ctx}: OVERALL max|d| {worst:.2e} (tol {tol:.0e})");
    assert!(
        bad.is_empty(),
        "{ctx}: classes over tolerance {tol:.0e}: {}",
        bad.join(", ")
    );
    worst
}

fn sum_sq_check(ctx: &str, got: f64, full: &Value) {
    let want = full["sum_sq"].as_f64().unwrap();
    let rel = (got - want).abs() / want;
    eprintln!("{ctx}: sum of squares ferric {got:.12e} ref {want:.12e} rel {rel:.2e}");
    assert!(rel < TOL_SUMSQ_REL, "{ctx}: sum of squares rel {rel:.2e}");
}

fn max_abs3(t: &Array3<f64>) -> f64 {
    t.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

fn check_case(system: &str, basis_name: &str, auxbasis: &str) {
    let c = load(system, basis_name, auxbasis);
    let ctx = format!("{system}/{basis_name}");
    let ints = &c.r["integrals"];

    // The reference's own s-type check (1-D real-space Gaussian-blob oracle,
    // no Fourier transform) is recorded by the generator, which refuses to
    // write above its bar. Re-assert it so a JSON from an older generator
    // without the oracle fails here instead of passing silently.
    let orc = &c.r["checks"]["s_type_oracle"];
    let bar = orc["bar"]
        .as_f64()
        .expect("checks.s_type_oracle.bar missing");
    let got = orc["this_system_max_abs"].as_f64().unwrap();
    let syn = orc["synthetic_primitives"]["max_abs"].as_f64().unwrap();
    eprintln!("{ctx}: k-space vs blob oracle: {got:.2e} (synthetic {syn:.2e}, bar {bar:.0e})");
    assert!(
        got < bar && syn < bar,
        "{ctx}: k-space reference misses the s-type oracle"
    );
    assert!(
        orc["synthetic_primitives"]["negative_control"]["n_over_bar"]
            .as_u64()
            .unwrap()
            > 0,
        "{ctx}: the oracle's negative control did not fail"
    );

    // AO-order anchor first: ferric libint Coulomb vs PySCF Coulomb.
    let c3 = eri3_tensor(Operator::coulomb(), &c.obs, &c.dfbs).unwrap();
    let c2 = coulomb_metric_2c(Operator::coulomb(), &c.dfbs).unwrap();
    compare(
        &format!("{ctx}/coulomb/3c"),
        &c.class3,
        &at3(&c3, &c.idx3),
        &f64s(&ints["coulomb"]["eri3"], &ctx),
        TOL_COULOMB_3C,
    );
    compare(
        &format!("{ctx}/coulomb/2c"),
        &c.class2,
        &at2(&c2, &c.idx2),
        &f64s(&ints["coulomb"]["eri2"], &ctx),
        TOL_COULOMB_2C,
    );

    for blk in ints["per_r0"].as_array().unwrap() {
        let r0 = blk["r0_bohr"].as_f64().unwrap();
        let r0a = blk["r0_angstrom"].as_f64().unwrap();
        for (name, op) in [("terf", Operator::terf(r0)), ("terfc", Operator::terfc(r0))] {
            let cx = format!("{ctx}/{name}/r0={r0a}A");
            let t3 = eri3_tensor(op, &c.obs, &c.dfbs).unwrap();
            let t2 = coulomb_metric_2c(op, &c.dfbs).unwrap();
            compare(
                &format!("{cx}/3c"),
                &c.class3,
                &at3(&t3, &c.idx3),
                &f64s(&blk[name]["eri3"], &cx),
                TOL_INT_3C,
            );
            compare(
                &format!("{cx}/2c"),
                &c.class2,
                &at2(&t2, &c.idx2),
                &f64s(&blk[name]["eri2"], &cx),
                TOL_INT_2C,
            );
            sum_sq_check(
                &format!("{cx}/3c"),
                t3.iter().map(|x| x * x).sum(),
                &blk[name]["eri3_full"],
            );
            sum_sq_check(
                &format!("{cx}/2c"),
                t2.iter().map(|x| x * x).sum(),
                &blk[name]["eri2_full"],
            );
            let m = max_abs3(&t3);
            let want = blk[name]["eri3_full"]["max_abs"].as_f64().unwrap();
            assert!(
                (m - want).abs() < TOL_INT_3C,
                "{cx}: max|(P|mu nu)| {m} vs {want}"
            );
        }
    }
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
#[ignore = "validation: SCS-MP2(2terfc)"]
fn terfc_integrals_h2o_cc_pvdz_vs_kspace() {
    check_case("h2o", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: SCS-MP2(2terfc)"]
fn terfc_integrals_nh3_cc_pvdz_vs_kspace() {
    check_case("nh3", "cc-pvdz", "cc-pvdz-ri");
}

#[test]
#[ignore = "validation: SCS-MP2(2terfc)"]
fn terfc_integrals_h2o_aug_cc_pvdz_vs_kspace() {
    check_case("h2o", "aug-cc-pvdz", "aug-cc-pvdz-rifit");
}

/// Resolving power: the comparison above must be able to tell r₀ = 0.75 Å
/// from 1.05 Å, and terfc from erfc at the same curvature.
#[test]
#[ignore = "validation: SCS-MP2(2terfc)"]
fn terfc_integrals_negative_controls() {
    for &(system, basis_name, auxbasis) in CASES {
        let c = load(system, basis_name, auxbasis);
        let ctx = format!("{system}/{basis_name}");
        let blocks = c.r["integrals"]["per_r0"].as_array().unwrap();
        let (b1, b2) = (&blocks[0], &blocks[1]);
        let r0_1 = b1["r0_bohr"].as_f64().unwrap();
        assert!(b2["r0_bohr"].as_f64().unwrap() > r0_1);
        let ref1 = f64s(&b1["terfc"]["eri3"], &ctx);
        let ref2 = f64s(&b2["terfc"]["eri3"], &ctx);
        let erfc_ref = f64s(&b1["erfc_same_w_info"]["eri3"], &ctx);
        // Premises: the references themselves differ.
        assert!(
            max_diff(&ref1, &ref2) >= MUST_MISS,
            "{ctx}: r0 refs coincide"
        );
        assert!(
            max_diff(&ref1, &erfc_ref) >= MUST_MISS,
            "{ctx}: erfc ref == terfc ref"
        );

        let got1 = at3(
            &eri3_tensor(Operator::terfc(r0_1), &c.obs, &c.dfbs).unwrap(),
            &c.idx3,
        );
        let d_r0 = max_diff(&got1, &ref2);
        eprintln!("{ctx}: terfc(r0_1) vs ref terfc(r0_2): max|d| {d_r0:.2e}");
        assert!(d_r0 >= MUST_MISS, "{ctx}: cannot tell r0_1 from r0_2");

        let w = 1.0 / (r0_1 * std::f64::consts::SQRT_2);
        let got_erfc = at3(
            &eri3_tensor(Operator::erfc(w), &c.obs, &c.dfbs).unwrap(),
            &c.idx3,
        );
        // ferric's erfc agrees with PySCF's erfc (the control's own premise) ...
        let d_e = max_diff(&got_erfc, &erfc_ref);
        eprintln!("{ctx}: ferric erfc(w) vs PySCF erfc(w): max|d| {d_e:.2e}");
        assert!(d_e < TOL_ERFC_PREMISE, "{ctx}: erfc premise {d_e:.2e}");
        // ... and misses terfc.
        let d_t = max_diff(&got_erfc, &ref1);
        eprintln!("{ctx}: ferric erfc(w) vs ref terfc(r0): max|d| {d_t:.2e}");
        assert!(d_t >= MUST_MISS, "{ctx}: erfc indistinguishable from terfc");
    }
}

// MUTATION LEDGER (2026-10-05, each mutant compiled, its test ran with
// `--ignored --exact`, and the harness restored the source with
// `git checkout`; test = terfc_integrals_h2o_cc_pvdz_vs_kspace):
//
// | mutant | where | result |
// |---|---|---|
// | M1 drop the r0 shift: `r02 = r0 * r0` -> `0.0` (3 sites; terf -> erf) | shim.cc compute_cart_eri3/eri2 | FAILED, 0 passed 1 failed; terf 3c misses in every class, 4e-4..1.4e-1 |
// | M4 ket-side `(-1)^(t+u+v)` -> `1.0` | shim.cc compute_cart_eri3 | FAILED, 0 passed 1 failed; ddp dds dpp dps fdp fds fpp fps pdp pds ppp pps sdp sds ... miss by 8e-5..1.2e-1, while the s-pair classes (sss pss dss fss) PASS |
// | M5 omega = 1/r0 instead of 1/(r0 sqrt2) | operator.rs Operator::terfc | FAILED, 0 passed 1 failed; terfc 3c misses in every class incl. sss (1.1e-1) |
//
// M4 is the artifact hypothesis made concrete: the broken sign touches only
// odd Hermite orders of the orbital pair, so every class whose pair is s-s
// passes at the floor and only the p/d pair classes fail. A one-centre pair
// with even l_mu + l_nu (O d-d in cc-pVDZ) also passes, because for A = B the
// Hermite expansion of x^(a+b) has a single parity and the dropped sign is the
// constant (-1)^(l_mu+l_nu) = +1. The literal issue mutant "erf(w(r+r0)) ->
// erf(w(r-r0))" is not expressible in this engine: r0 enters only through
// s = phi^2 r0^2 (terf is even in r0), so M1 removes the shift instead.
// Energy-level mutants (A->Bohr, Eq. 12 SS difference) are in
// crates/ferric-mp2/tests/validation_scs_mp2_2terfc.rs.
