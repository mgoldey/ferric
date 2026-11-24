//! ferric's own OP-MP3 on the paper's basis and geometries reproduces the
//! paper's HF-reference W matrices, `E_HF`, and per-reaction W4-17 TAE errors
//! (Bystrom, Berkelbach, Hait, arXiv:2609.25330; Zenodo
//! 10.5281/zenodo.23482348).
//!
//! # What is compared
//!
//! H, O, F (ROHF) and H2, HF, H2O (RHF) at aug-cc-pCVTZ (H: aug-cc-pVTZ;
//! `testdata/reference/opmp3/aug-cc-pcvtz_hbcnof.json`), all electrons, exact
//! integrals, the ACCDB geometries (`testdata/molecules/w4-17/`). Per
//! structure: `E_HF[ρ]`, `E_NBS` and the sorted eigenvalues of each of the
//! eight W channels against the paper's `.npy` (`paper_w4-17.json`). Per
//! reaction (TAE_W4-17_110 H2, _115 H2O, _125 HF): every model's error
//! against the paper's csv. The W eigenvalues, not the matrices, are compared
//! because the occupied orbitals of a degenerate shell (the atoms' p shell)
//! are fixed only up to a rotation that the eigenvalues do not see.
//!
//! # Artifact hypothesis vs physics hypothesis
//!
//! * Right: every number within the SCF-threshold bar below.
//! * ROHF semicanonicalized in the wrong Fock (e.g. the ROHF effective Fock
//!   instead of the spin Focks): the atoms' `E_NBS` and W eigenvalues move by
//!   ≥ 1e-4 Ha, and reactions 115/125 by ≥ 0.06 kcal/mol.
//! * OS prefactor ⅛ or the β OS block built without the axis swap: the OS
//!   eigenvalues of every molecule off by a factor of 2 or permuted.
//!
//! # TOLERANCES (derived)
//!
//! The paper's PySCF SCF stopped at `conv_tol_grad` 1e-5 and ferric's at
//! density change 1e-10. An orbital error κ moves `E_HF` to second order
//! (≤ 1e-9 Ha at κ ~ 1e-5 with orbital Hessian eigenvalues ≲ 10) and the
//! non-stationary `E_NBS`, `E_c` and the W eigenvalues to first order, by at
//! most `|E_c| · κ` times an O(10) response factor. Per structure:
//! `tol_E_HF = 1e-8` Ha and `tol_W = 10 · |E_c| · 1e-5` Ha (≈ 3e-5 Ha for
//! H2O). Per reaction: `Σ_s |count_s| · tol_W(s)` in kcal/mol (≈ 0.03).
//!
//! # MEASURED (2026-11-14, release, 38 s, 3.0 GB peak)
//!
//! `E_HF` ≤ 2.1e-11 Ha, `E_NBS` ≤ 4.3e-12 Ha, W eigenvalues ≤ 1.0e-11 Ha,
//! per-reaction errors ≤ 6.6e-7 kcal/mol (the csv's rounding), all models.
//!
//! # MUTATIONS (run 2026-11-15; failed on an assertion)
//!
//! * `opmp3.rs` `w_channels`: OS factor `0.25` → `0.125` (both spins).
//!
//! The reference-side mutations (β Fock from the α exchange, `E_NBS` sign)
//! are run against `opmp3_qchem.rs`, which is cheaper and fails on both.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_mp2::opmp3::{Opmp3Models, WChannels};
use ferric_mp2::opmp3_driver::{run_opmp3, Opmp3Config, Opmp3Integrals, Opmp3RefKind};
use ferric_scf::PreparedSystem;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};
use serde_json::Value;

const KCAL: f64 = 627.5094740630558;
const STRUCTURES: [&str; 6] = ["h", "o", "f", "h2", "hf", "h2o"];
const REACTIONS: [&str; 3] = ["TAE_W4-17_110", "TAE_W4-17_115", "TAE_W4-17_125"];

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

/// A symmetric W from its packed upper triangle (`{"n", "upper"}`,
/// row-major over `i <= j`, as `scripts/gen_opmp3_paper_fixtures.py` writes it).
fn matrix(v: &Value) -> Array2<f64> {
    let n = v["n"].as_u64().expect("matrix n") as usize;
    let upper = v["upper"].as_array().expect("matrix upper");
    assert_eq!(upper.len(), n * (n + 1) / 2, "packed upper triangle length");
    let mut m = Array2::zeros((n, n));
    let mut it = upper.iter();
    for i in 0..n {
        for j in i..n {
            let x = it.next().unwrap().as_f64().expect("number");
            m[[i, j]] = x;
            m[[j, i]] = x;
        }
    }
    m
}

fn eigenvalues(m: &Array2<f64>) -> Vec<f64> {
    if m.nrows() == 0 {
        return vec![];
    }
    let (l, _) = m.eigh(UPLO::Lower).expect("eigh");
    l.to_vec()
}

/// `(label, ours, paper)` per W channel, the paper's file labels.
fn channels<'a>(
    n: u8,
    ours: &'a WChannels,
    w: &'a Value,
) -> Vec<(String, Array2<f64>, Array2<f64>)> {
    let mut out = vec![];
    for (s, spin) in ["alpha", "beta"].iter().enumerate() {
        out.push((
            format!("W{n}{spin}_ss"),
            ours.ss[s].clone(),
            matrix(&w[format!("W{n}{spin}_ss")]),
        ));
        out.push((
            format!("W{n}{spin}_os"),
            ours.os[s].clone(),
            matrix(&w[format!("W{n}{spin}_os")]),
        ));
    }
    out
}

fn model_energies(m: &Opmp3Models) -> [(&'static str, f64); 6] {
    [
        ("MP2", m.mp2),
        ("MP3", m.mp3),
        ("OP-MP3", m.op_mp3),
        ("SOP-MP3", m.sop_mp3),
        ("SSD-OPMP3", m.ssd_opmp3),
        ("P-MP3", m.p_mp3),
    ]
}

#[test]
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
fn hf_reference_reproduces_the_paper_w_and_tae_errors() {
    let root = workspace_root();
    let data: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("testdata/reference/opmp3/paper_w4-17.json"))
            .expect("fixture"),
    )
    .expect("fixture JSON");
    let bs = basis::load_bse_json(
        root.join("testdata/reference/opmp3/aug-cc-pcvtz_hbcnof.json")
            .to_str()
            .unwrap(),
    )
    .expect("basis");
    let mut bad: Vec<String> = vec![];
    let mut totals = std::collections::BTreeMap::new();
    let mut tol_w = std::collections::BTreeMap::new();
    for name in STRUCTURES {
        let xyz =
            std::fs::read_to_string(root.join(format!("testdata/molecules/w4-17/{name}.xyz")))
                .expect("xyz");
        let cm: Vec<i64> = xyz
            .lines()
            .nth(1)
            .unwrap()
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        let mol = Molecule::parse_xyz(&xyz, cm[0] as i32, cm[1] as usize).unwrap();
        let sys = PreparedSystem::new(mol, &bs).unwrap();
        let cfg = Opmp3Config {
            scf_energy_conv: Some(1e-12),
            scf_density_conv: Some(1e-10),
            scf_max_iter: Some(400),
            ..Opmp3Config::new(Opmp3RefKind::Hf)
        };
        let r =
            run_opmp3(&sys, Opmp3Integrals::Exact, &cfg).unwrap_or_else(|e| panic!("{name}: {e}"));
        let p = &data["structures"]["HF"][name];
        let tw = 10.0 * r.models.mp3.abs() * 1e-5 + 1e-10;
        tol_w.insert(name, tw);
        let check = |bad: &mut Vec<String>, what: String, got: f64, want: f64, tol: f64| {
            let d = (got - want).abs();
            eprintln!("{name:<4} {what:<14} ferric {got:+.10} paper {want:+.10} |d| {d:.2e} (tol {tol:.1e})");
            if d >= tol {
                bad.push(format!("{name} {what}: |d| {d:.2e} >= {tol:.1e}"));
            }
        };
        check(
            &mut bad,
            "E_HF".into(),
            r.e_hf,
            p["e_hf"].as_f64().unwrap(),
            1e-8,
        );
        check(
            &mut bad,
            "E_NBS".into(),
            r.e_nbs,
            p["e_nbs"].as_f64().unwrap(),
            tw,
        );
        for (n, ours) in [(2u8, &r.w.w2), (3u8, &r.w.w3)] {
            for (label, mine, theirs) in channels(n, ours, &p["w"]) {
                let (a, b) = (eigenvalues(&mine), eigenvalues(&theirs));
                if a.len() != b.len() {
                    bad.push(format!(
                        "{name} {label}: {} vs {} eigenvalues",
                        a.len(),
                        b.len()
                    ));
                    continue;
                }
                let d = a
                    .iter()
                    .zip(&b)
                    .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
                check(&mut bad, format!("{label} eig"), d, 0.0, tw);
            }
        }
        let base = r.e_hf + r.e_nbs;
        totals.insert(name, model_energies(&r.models).map(|(k, e)| (k, base + e)));
    }
    for rxn in REACTIONS {
        let rx = &data["reactions"]["HF"][rxn];
        let refe = rx["reference_kcal"].as_f64().unwrap();
        let structs: Vec<&str> = rx["structs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let counts: Vec<f64> = rx["counts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let tol: f64 = structs
            .iter()
            .zip(&counts)
            .map(|(s, c)| c.abs() * tol_w[s])
            .sum::<f64>()
            * KCAL;
        for m in 0..6 {
            let label = totals[structs[0]][m].0;
            let pred: f64 = structs
                .iter()
                .zip(&counts)
                .map(|(s, c)| c * totals[s][m].1)
                .sum();
            let ours = KCAL * pred - refe;
            let theirs = rx["errors_kcal"][label].as_f64().unwrap();
            let d = (ours - theirs).abs();
            eprintln!("{rxn} {label:<10} ours {ours:+.6} paper {theirs:+.6} |d| {d:.2e} (tol {tol:.1e}) kcal/mol");
            if d >= tol {
                bad.push(format!("{rxn} {label}: |d| {d:.2e} >= {tol:.1e} kcal/mol"));
            }
        }
    }
    assert!(bad.is_empty(), "off the paper:\n{}", bad.join("\n"));
}
