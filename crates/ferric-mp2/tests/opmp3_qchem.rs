//! The OP-MP3 reference path against Q-Chem: `E_HF[ρ]`, `E_NBS`,
//! `Tr W^(2)` (SS, OS) and `Tr W^(3)` of HF and PBE0 references at cc-pVDZ.
//!
//! # What is compared
//!
//! `testdata/reference/opmp3/qchem_cc-pvdz.json` holds the Q-Chem 6.4
//! values the paper's own `wmatrix_code` tests check against (Zenodo
//! 10.5281/zenodo.23482348, `scripts/gen_opmp3_paper_fixtures.py`):
//! restricted open-shell CH2 ³B1, CH3, CN, HCO; closed-shell H2O, He; the
//! one-electron H atom. ferric runs `run_opmp3` (RHF/ROHF or RKS/ROKS, the HF
//! Fock of the reference density, semicanonicalization, `E_NBS`, exact
//! integrals, all electrons) and compares all five numbers. This is the
//! independent-code anchor of the reference path: a wrong semicanonical
//! rotation, a KS orbital energy in a denominator or a missing `E_NBS` term
//! moves `E_NBS` and `Tr W^(2)` by 1e-4..1e-2 Ha.
//!
//! # Artifact hypothesis vs physics hypothesis
//!
//! * Right: HF rows at the printed-digit floor; PBE0 rows within the grid
//!   difference (ferric's pruned 75 × 110 grid vs Q-Chem SG-1).
//! * HF Fock replaced by the KS Fock (the ROKS spin Focks): every PBE0 E_NBS
//!   and Tr W2 row misses by ≥ 1e-3 Ha (KS vs HF orbital energies).
//! * E_NBS summed over one spin only or with the wrong sign: the ROHF/ROKS
//!   E_NBS rows miss by 1e-3.
//!
//! # TOLERANCES (derived)
//!
//! * HF: half a unit of the last printed digit (`0.5·10^−d`, `d` capped at
//!   10 because some JSON values are sums with float tails; a dropped
//!   trailing zero only makes the bar looser), plus the first-order
//!   orbital-error term `10·|E_c|·(1e-9 + 1e-10)` of the two SCF thresholds
//!   (Q-Chem `scf_convergence 9`, ferric `density_conv` 1e-10). A value
//!   printed as exactly zero must be below 1e-10.
//! * PBE0: the paper's own PySCF-vs-Q-Chem bar (`tests/test_reference.py`:
//!   1e-5 Ha for E_HF, E_NBS and Tr W2, 1e-4 Ha for Tr W3), the same
//!   cross-code, cross-grid situation.
//!
//! # MUTATIONS (run 2026-11-15; each failed `hf_reference_matches_qchem` on an assertion)
//!
//! `crates/ferric-mp2/src/opmp3_reference.rs`:
//! * `opmp3_reference`: `f_b = &h + &j - &k_b` → `- &k_a` (β exchange
//!   from the α density).
//! * `SemicanonicalSpin::singles_energy`: `f * f / (ε_i − ε_a)` → `/ (ε_a − ε_i)`.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_mp2::opmp3_driver::{run_opmp3, Opmp3Config, Opmp3Integrals, Opmp3RefKind};
use serde_json::Value;

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
    let p = workspace_root().join("testdata/reference/opmp3/qchem_cc-pvdz.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; regenerate with scripts/gen_opmp3_paper_fixtures.py",
            p.display()
        )
    }))
    .expect("fixture JSON")
}

/// The molecule of an xyz text whose line 2 is `charge multiplicity`.
fn molecule(xyz: &str) -> Molecule {
    let line2: Vec<i32> = xyz
        .lines()
        .nth(1)
        .unwrap()
        .split_whitespace()
        .take(2)
        .map(|t| t.parse().unwrap())
        .collect();
    Molecule::parse_xyz(xyz, line2[0], line2[1] as usize).unwrap()
}

/// (name, ferric value, Q-Chem key).
fn ferric_values(r: &ferric_mp2::opmp3_driver::Opmp3Result) -> [(&'static str, f64); 5] {
    let tr = |m: &[ndarray::Array2<f64>; 2]| m[0].diag().sum() + m[1].diag().sum();
    [
        ("e_hf", r.e_hf),
        ("e_nbs", r.e_nbs),
        ("tr_w2_ss", tr(&r.w.w2.ss)),
        ("tr_w2_os", tr(&r.w.w2.os)),
        ("tr_w3", r.w.w3.trace()),
    ]
}

fn run_reference(functional: &str) {
    let data = fixture();
    let mut bad = vec![];
    for (name, entry) in data["molecules"].as_object().unwrap() {
        let mol = molecule(entry["xyz"].as_str().unwrap());
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let kind = match functional {
            "hf" => Opmp3RefKind::Hf,
            f => Opmp3RefKind::Dft(f.to_string()),
        };
        let cfg = Opmp3Config {
            scf_energy_conv: Some(1e-12),
            scf_density_conv: Some(1e-10),
            scf_max_iter: Some(400),
            ..Opmp3Config::new(kind)
        };
        let r = run_opmp3(
            &ParallelContext::default(),
            &mol,
            &obs,
            Opmp3Integrals::Exact,
            &cfg,
        )
        .unwrap_or_else(|e| panic!("{name} {functional}: {e}"));
        let q = &entry[functional];
        let e_c = (r.models.mp3).abs();
        for (key, got) in ferric_values(&r) {
            let want = q[key].as_f64().unwrap();
            let tol = if functional == "hf" {
                if want == 0.0 {
                    1e-10
                } else {
                    let d = q["decimals"][key].as_i64().unwrap().min(10) as i32;
                    0.5 * 10f64.powi(-d) + 10.0 * e_c * (1e-9 + 1e-10)
                }
            } else if key == "tr_w3" {
                1e-4
            } else {
                1e-5
            };
            let d = (got - want).abs();
            eprintln!(
                "{name:<4} {functional:<5} {key:<9} ferric {got:+.10} qchem {want:+.10} |d| {d:.2e} \
                 (tol {tol:.1e})"
            );
            if d >= tol {
                bad.push(format!("{name} {key}: |d| {d:.2e} >= {tol:.1e}"));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{functional} rows off Q-Chem:\n{}",
        bad.join("\n")
    );
}

#[test]
fn hf_reference_matches_qchem() {
    run_reference("hf");
}

#[test]
fn pbe0_reference_matches_qchem() {
    run_reference("pbe0");
}
