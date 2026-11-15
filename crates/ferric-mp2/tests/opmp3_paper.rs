//! OP-MP3 resummation on the paper's own W matrices reproduces the paper's
//! per-reaction W4-17 TAE errors (Bystrom, Berkelbach, Hait,
//! arXiv:2609.25330; Zenodo 10.5281/zenodo.23482348).
//!
//! # What is compared
//!
//! The fixture `testdata/reference/opmp3/paper_w4-17.json`
//! (`scripts/gen_opmp3_paper_fixtures.py`) holds, for the HF, PBE0 and
//! κOOMP2 references, the eight W matrices, `E_HF[ρ]` and `E_NBS` of 18
//! structures and the paper's errors of the 12 W4-17 nonMR TAEs they form
//! (H2, H2O, HCN, HF, benzene, N2, NH3, BH3, C2H2, C2H4, CH4, CO). Each
//! structure's total `E_HF + E_NBS + E_c(model)` comes from
//! `ferric_mp2::opmp3` and every model's per-reaction error is compared with
//! the paper's csv: MP2, MP3, OP-MP3, SOP-MP3, SSD-OPMP3 (x_ss 0, 0.5,
//! 0.45) and the scalar Padé P-MP3.
//!
//! This tests the resummation and the model definitions (Padé form, which
//! matrix sits in which slot, where x enters, SS/OS damping) on inputs the
//! paper computed; `opmp3.rs` tests the W construction.
//!
//! # Artifact hypothesis vs physics hypothesis
//!
//! * Right: every per-reaction error agrees to the csv's printed 1e-6.
//! * Numerator and denominator swapped, or W2/W3 in each other's slot: errors
//!   of tens to hundreds of kcal/mol.
//! * Same-spin damping dropped (x_ss = 1) or put on the OS channel: only the
//!   SSD-OPMP3 column fails, by about 1-10 kcal/mol.
//! * SS/OS labels swapped: SOP-MP3 and SSD-OPMP3 fail, OP-MP3/MP2/MP3 pass.
//!
//! # TOLERANCE (derived)
//!
//! [`TOL_KCAL`] 1e-5 kcal/mol: the csv prints the errors to 1e-6 kcal/mol
//! (half-unit rounding 5e-7), the reference TAE is printed to 1e-6 and
//! E_HF/E_NBS are read at 1e-15 Ha; summing up to 13 structures at ε ≈ 1e-16
//! relative of ~230 Ha totals gives ≤ 1e-11 Ha = 6e-9 kcal/mol. 1e-5 is ten
//! times the combined rounding.
//!
//! # MUTATIONS (run 2026-11-15; each failed all three tests on an assertion)
//!
//! In `crates/ferric-mp2/src/opmp3.rs`:
//! * `pade`: `ak.dot(&v)` → `(&ak - &submatrix(b, &keep)).dot(&v)` (the
//!   numerator `(A − B)`): the OP-MP3/SOP-MP3/SSD-OPMP3 rows fail.
//! * `ssd_opmp3`: `sop_mp3(w, x_ss, 1.0)` → `sop_mp3(w, 1.0, 1.0)` (damping
//!   dropped): the SSD-OPMP3 rows fail.
//! * `w_channels` is not on this path (the W come from the fixture).

use std::path::{Path, PathBuf};

use ferric_mp2::opmp3::{paper_x_ss, Opmp3Models, WChannels, WMatrices};
use ndarray::Array2;
use serde_json::Value;

const TOL_KCAL: f64 = 1e-5;

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

fn fixture() -> Value {
    let p = workspace_root().join("testdata/reference/opmp3/paper_w4-17.json");
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; regenerate with scripts/gen_opmp3_paper_fixtures.py",
            p.display()
        )
    });
    serde_json::from_str(&text).expect("fixture JSON")
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

/// The W matrices of one structure from the fixture (paper's file labels).
pub fn paper_w(s: &Value) -> WMatrices {
    let w = &s["w"];
    let ch = |n: u8| WChannels {
        ss: [
            matrix(&w[format!("W{n}alpha_ss")]),
            matrix(&w[format!("W{n}beta_ss")]),
        ],
        os: [
            matrix(&w[format!("W{n}alpha_os")]),
            matrix(&w[format!("W{n}beta_os")]),
        ],
    };
    WMatrices {
        w2: ch(2),
        w3: ch(3),
    }
}

/// (model label in the fixture, correlation energy).
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

/// Worst |ours − paper| per model over the reactions of `reference`.
fn reproduce(data: &Value, reference: &str, x_ss: f64) -> Vec<(&'static str, f64)> {
    let kcal = data["provenance"]["kcal_per_hartree"].as_f64().unwrap();
    let structs = &data["structures"][reference];
    let total = |name: &str| -> Vec<(&'static str, f64)> {
        let s = &structs[name];
        let base = s["e_hf"].as_f64().unwrap() + s["e_nbs"].as_f64().unwrap();
        let m = Opmp3Models::evaluate(&paper_w(s), x_ss).expect("models");
        model_energies(&m)
            .into_iter()
            .map(|(k, e)| (k, base + e))
            .collect()
    };
    let rxns = data["reactions"][reference].as_object().expect("reactions");
    assert_eq!(rxns.len(), 12, "{reference}: 12 TAE reactions expected");
    let mut worst = vec![];
    for (name, r) in rxns {
        let refe = r["reference_kcal"].as_f64().unwrap();
        let mut pred = [0.0f64; 6];
        let mut labels = [""; 6];
        for (st, c) in r["structs"]
            .as_array()
            .unwrap()
            .iter()
            .zip(r["counts"].as_array().unwrap())
        {
            for (n, (k, e)) in total(st.as_str().unwrap()).into_iter().enumerate() {
                pred[n] += c.as_f64().unwrap() * e;
                labels[n] = k;
            }
        }
        for n in 0..6 {
            let ours = kcal * pred[n] - refe;
            let theirs = r["errors_kcal"][labels[n]].as_f64().unwrap();
            let d = (ours - theirs).abs();
            eprintln!(
                "{reference} {name:<14} {:<10} ours {ours:+.6} paper {theirs:+.6} |d| {d:.1e}",
                labels[n]
            );
            match worst.iter_mut().find(|(k, _)| *k == labels[n]) {
                Some((_, w)) => *w = f64::max(*w, d),
                None => worst.push((labels[n], d)),
            }
        }
    }
    worst
}

fn check_reference(reference: &str, x_ss: f64, label: &str) {
    let data = fixture();
    let any = data["reactions"][reference]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap();
    assert_eq!(
        any["errors_kcal"]["x_ss_label"].as_str(),
        Some(label),
        "the csv's SSD-OPMP3 column must be the x_ss used here"
    );
    let worst = reproduce(&data, reference, x_ss);
    for (k, w) in &worst {
        eprintln!("{reference} {k}: max per-reaction |ours - paper| {w:.2e} kcal/mol");
    }
    let bad: Vec<_> = worst.iter().filter(|(_, w)| *w >= TOL_KCAL).collect();
    assert!(
        bad.is_empty(),
        "{reference}: per-reaction errors off the paper's: {bad:?}"
    );
}

#[test]
fn hf_reference_reproduces_the_paper_per_reaction() {
    check_reference("HF", paper_x_ss("hf").unwrap(), "SSD-OPMP3(x_ss=0)");
}

#[test]
fn pbe0_reference_reproduces_the_paper_per_reaction() {
    check_reference("PBE0", paper_x_ss("pbe0").unwrap(), "SSD-OPMP3(x_ss=0.5)");
}

#[test]
fn koomp2_reference_reproduces_the_paper_per_reaction() {
    check_reference(
        "KOOMP2",
        paper_x_ss("koomp2").unwrap(),
        "SSD-OPMP3(x_ss=0.45)",
    );
}
