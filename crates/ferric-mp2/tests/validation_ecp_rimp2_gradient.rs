//! VALIDATION tier — VALIDATION.md row "ECP gradients", RI-MP2 case.
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_ecp_rimp2_gradient \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's analytic RI-MP2 gradient (`rimp2_gradient_analytical`, exact-J/K
//! RHF reference, all electrons outside the ECP core correlated) on distorted
//! CH3I / def2-SVP with the def2 I ECP, def2-svp-rifit correlation aux. The ECP
//! enters the MP2 gradient as `Σ D_relaxed dV_ECP/dR`, added in
//! `mp2_relaxed_lagrangian_gradient` (crates/ferric-mp2/src/gradient.rs) right
//! after `oneelectron_gradient` with the same relaxed total density.
//!
//! References (`scripts/validation/gen_ecp_gradient.py`, case `ch3i_rimp2`,
//! `testdata/reference/validation/ecp_gradient/ch3i_rimp2_def2-svp.json`; same
//! like-for-like basis/ECP/aux injection as the RI-MP2 gradient and ECP rows):
//!
//! * PySCF 2.13 exact-J/K RHF + `mp.dfmp2.DFMP2`: energies, and a 5-point FD
//!   (h = 2e-3 Bohr) of the DF-MP2 total energy as the gradient (PySCF has no
//!   analytic DF-MP2 gradient);
//! * the MP2 ECP term `tr[D_relaxed dV_ECP/dR]` WITHOUT a relaxed-density
//!   code: the 5-point λ-derivative of the re-solved DF-MP2 energy with
//!   `hcore + λ dV_ECP/dR_{A,c}`. The same λ construction on the RHF energy
//!   alone is checked against PySCF's analytic RHF ECP term before writing;
//! * ORCA 6.1.1 `RI-MP2 NoRI NoFrozenCore ExtremeSCF EnGrad` (NewGTO + NewECP +
//!   NewAuxCGTO from ferric's JSONs), written only if its RHF and correlation
//!   energies match PySCF's.
//!
//! Checks: V_nn / N_core / counts; RHF and correlation energies; ferric's RHF
//! gradient vs PySCF analytic (anchor); RI-MP2 gradient vs PySCF FD and vs
//! ORCA; Σ_A g_A ≈ 0; ferric analytic vs a 5-point FD of ferric's OWN RI-MP2
//! energy on the I atom's coordinates plus the largest other component.
//!
//! # NEGATIVE CONTROLS / MUTATION
//!
//! * ferric's gradient must MISS `gradient_without_ecp` (the reference minus
//!   its MP2 ECP term) by > [`MUST_MISS`] — what the gradient was before the
//!   ECP line existed.
//! * ferric's MP2 gradient must MISS PySCF's RHF gradient (correlation tested).
//! * If the MP2 ECP term differs resolvably from the HF-density ECP term, the
//!   gradient must also miss "reference with the ECP term contracted with the
//!   HF density" — i.e. the RELAXED density is what is contracted.
//! * MUTATION: delete the line
//!   `grad += &ferric_scf::gradient::ecp_gradient(mol, obs, &dm1_total_ao)?;`
//!   in `mp2_relaxed_lagrangian_gradient`. The PySCF-FD, ORCA and own-FD
//!   gradient checks must fail (by the ECP term, ~1e-2 Ha/Bohr).
//!   Run 2026-09-25: the test fails.
//!
//! A missing reference JSON is a HARD failure.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::gradient::rimp2_gradient_analytical;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::gradient::rhf_gradient_exact_jk;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use serde_json::Value;

const REF: &str = "testdata/reference/validation/ecp_gradient/ch3i_rimp2_def2-svp.json";
const XYZ: &str = "testdata/molecules/validation/ecp/ch3i.xyz";
const BASIS: &str = "def2-svp";
const AUX: &str = "def2-svp-rifit";

const TOL_ENUC: f64 = 1e-9;
// Measured 2.4e-8 Ha.
const TOL_E_RHF: f64 = 2.5e-7;
// Measured 7.4e-10 Ha.
const TOL_E_CORR: f64 = 1e-8;
// Measured 1.1e-8 Ha/Bohr.
const TOL_G_RHF: f64 = 2e-7;
// vs PySCF 5-point FD of DF-MP2: measured 7.7e-9 Ha/Bohr.
const TOL_G_FD: f64 = 1e-7;
// vs ORCA RI-MP2 EnGrad: measured 4.3e-8 Ha/Bohr.
const TOL_G_ORCA: f64 = 5e-7;
// |Σ_A g_A|.
const TOL_SUM: f64 = 1e-8;
// ferric analytic vs ferric's own 5-point FD: measured 4.7e-8 Ha/Bohr.
const TOL_SELF_FD: f64 = 6e-7;
const FD_STEP: f64 = 2e-3;
/// A reference ferric must MISS is missed by at least this.
const MUST_MISS: f64 = 1e-4;

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

fn reference() -> Value {
    let path = workspace_root().join(REF);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_ecp_gradient.py ch3i_rimp2 — a missing reference is a \
             failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn num(v: &Value, ptr: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("reference field {ptr} missing or not a number"))
}

fn grad(v: &Value, ptr: &str, natm: usize) -> Array2<f64> {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("reference field {ptr} missing"));
    assert_eq!(rows.len(), natm, "{ptr}: rows");
    let mut g = Array2::zeros((natm, 3));
    for (a, row) in rows.iter().enumerate() {
        for c in 0..3 {
            g[(a, c)] = row[c].as_f64().expect("gradient entry");
        }
    }
    g
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    // f64::max drops a NaN operand: a non-finite component must fail instead.
    assert!(
        a.iter().chain(b.iter()).all(|v| v.is_finite()),
        "non-finite gradient component"
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn check_close(what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!(
        "ch3i/rimp2: {what:<24} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})"
    );
    assert!(
        d < tol,
        "{what}: {got:.12} vs {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_grad(what: &str, got: &Array2<f64>, want: &Array2<f64>, tol: f64) {
    let d = max_abs_diff(got, want);
    eprintln!("ch3i/rimp2: {what:<28} max|d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{what}: max |d| {d:.2e} >= {tol:.0e}\n ferric {got:?}\n ref {want:?}"
    );
}

fn check_miss(what: &str, got: &Array2<f64>, want: &Array2<f64>, floor: f64) {
    let d = max_abs_diff(got, want);
    eprintln!("ch3i/rimp2: {what:<28} max|d| {d:.2e} (must exceed {floor:.0e})");
    assert!(
        d > floor,
        "negative control {what}: max |d| {d:.2e} <= {floor:.0e} — the bar cannot tell these apart"
    );
}

fn rhf_config() -> RhfConfig {
    RhfConfig {
        density_conv: 1e-10,
        max_iter: 300,
        ..Default::default()
    }
}

/// RI-MP2 total energy at `mol` (exact-J/K RHF + RI-MP2).
fn rimp2_energy(mol: &Molecule) -> f64 {
    let obs_bs = basis::bundled(BASIS).unwrap();
    let aux_bs = basis::bundled(AUX).unwrap();
    let obs = PreparedBasis::new(mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(mol, &aux_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        mol,
        &obs,
        op,
        &bounds,
        &rhf_config(),
    )
    .expect("displaced RHF");
    assert!(rhf.converged, "displaced RHF not converged");
    ri_mp2(mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default())
        .expect("displaced RI-MP2")
        .total_energy
}

#[test]
#[ignore = "validation: ECP gradient"]
fn ecp_rimp2_gradient_ch3i_def2svp() {
    let r = reference();
    assert_eq!(r["method"].as_str(), Some("rimp2"));
    assert_eq!(r["basis"].as_str(), Some(BASIS));
    assert_eq!(r["aux_basis"].as_str(), Some(AUX));

    // ---- system build: ECP applied BEFORE nelec / V_nn ----
    let xyz = workspace_root().join(XYZ);
    let mut mol = Molecule::load_xyz(xyz.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let obs_bs = basis::bundled(BASIS).unwrap();
    let aux_bs = basis::bundled(AUX).unwrap();
    mol.apply_ecp(&obs_bs);
    let natm = mol.atoms.len();
    let core: Vec<i64> = mol.atoms.iter().map(|a| a.n_core_ecp as i64).collect();
    let core_ref: Vec<i64> = r["ecp_core_electrons"]
        .as_array()
        .expect("ecp_core_electrons")
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(core, core_ref, "per-atom ECP core electrons");
    let ecp_atom = mol
        .atoms
        .iter()
        .position(|a| a.n_core_ecp > 0)
        .expect("I must carry an ECP");
    assert_eq!(
        mol.nelec() as i64,
        r["nelectron"].as_i64().unwrap(),
        "nelec"
    );
    check_close(
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion"),
        TOL_ENUC,
    );
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    assert_eq!(obs.nbasis() as u64, r["nao"].as_u64().unwrap(), "AO count");
    assert_eq!(
        dfbs.nbasis() as u64,
        r["naux"].as_u64().unwrap(),
        "aux count"
    );

    // ---- RHF anchor ----
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = rhf_config();
    assert!(cfg.df_j_aux.is_none() && cfg.df_k_aux.is_none());
    let rhf =
        solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).expect("solve_rhf");
    assert!(rhf.converged, "RHF not converged");
    check_close(
        "E_RHF vs PySCF",
        rhf.energy,
        num(&r, "/rimp2/e_rhf"),
        TOL_E_RHF,
    );
    let g_rhf = rhf_gradient_exact_jk(&mol, &obs, op, &bounds, &rhf, None).unwrap();
    let g_rhf_ref = grad(&r, "/rimp2/rhf_gradient", natm);
    check_grad("RHF grad vs PySCF", &g_rhf, &g_rhf_ref, TOL_G_RHF);

    // ---- RI-MP2 energy ----
    let mp2_cfg = RiMp2Config::default();
    assert_eq!(
        mp2_cfg.frozen_core, 0,
        "all-electron (outside the ECP core)"
    );
    let mp2 = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &mp2_cfg).unwrap();
    check_close(
        "E_corr vs PySCF",
        mp2.mp2_corr,
        num(&r, "/rimp2/e_corr"),
        TOL_E_CORR,
    );

    // ---- RI-MP2 gradient ----
    let g = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &mp2_cfg, None)
        .expect("rimp2_gradient_analytical");
    let g_fd = grad(&r, "/rimp2/gradient", natm);
    let g_orca = grad(&r, "/rimp2/orca/gradient", natm);
    check_grad("MP2 grad vs PySCF FD", &g, &g_fd, TOL_G_FD);
    check_grad("MP2 grad vs ORCA", &g, &g_orca, TOL_G_ORCA);
    assert!(g.iter().all(|v| v.is_finite()), "non-finite MP2 gradient");
    let sum = (0..3)
        .map(|c| g.column(c).sum().abs())
        .fold(0.0f64, f64::max);
    eprintln!("ch3i/rimp2: |Σ_A g_A| = {sum:.2e} (tol {TOL_SUM:.0e})");
    assert!(sum < TOL_SUM, "Σ_A g_A = {sum:.2e}");

    // ---- negative controls ----
    let g_no_ecp = grad(&r, "/rimp2/gradient_without_ecp", natm);
    check_miss("MP2 vs ref without ECP term", &g, &g_no_ecp, MUST_MISS);
    check_miss("MP2 vs PySCF RHF grad", &g, &g_rhf_ref, MUST_MISS);
    let ecp_mp2 = grad(&r, "/rimp2/gradient_ecp_term", natm);
    let ecp_hf = grad(&r, "/rimp2/rhf_gradient_ecp_term", natm);
    let relax = max_abs_diff(&ecp_mp2, &ecp_hf);
    eprintln!("ch3i/rimp2: |ECP term(D_relaxed) - ECP term(D_HF)| = {relax:.2e}");
    if relax > 10.0 * TOL_G_FD {
        let g_hf_density = &g_fd - &ecp_mp2 + &ecp_hf;
        check_miss("MP2 vs ref, ECP on D_HF", &g, &g_hf_density, 3.0 * TOL_G_FD);
    }

    // ---- ferric's own 5-point FD ----
    let mut coords: Vec<(usize, usize)> = (0..3).map(|c| (ecp_atom, c)).collect();
    let other = (0..natm)
        .filter(|&a| a != ecp_atom)
        .flat_map(|a| (0..3).map(move |c| (a, c)))
        .max_by(|p, q| g[*p].abs().total_cmp(&g[*q].abs()))
        .expect("a second atom");
    coords.push(other);
    let mut worst = 0.0f64;
    for (a, c) in coords {
        let e = |k: f64| {
            let mut m = mol.clone();
            let dx = k * FD_STEP;
            match c {
                0 => m.atoms[a].x += dx,
                1 => m.atoms[a].y += dx,
                _ => m.atoms[a].zpos += dx,
            }
            rimp2_energy(&m)
        };
        let fd = (e(-2.0) - 8.0 * e(-1.0) + 8.0 * e(1.0) - e(2.0)) / (12.0 * FD_STEP);
        let d = (g[(a, c)] - fd).abs();
        eprintln!(
            "ch3i/rimp2: FD atom {a} coord {c}: analytic {:+.10} FD {fd:+.10} |d| {d:.2e}",
            g[(a, c)]
        );
        assert!(
            d.is_finite(),
            "ch3i/rimp2: FD atom {a} coord {c} is not finite"
        );
        worst = worst.max(d);
    }
    eprintln!("ch3i/rimp2: max |analytic - own FD| = {worst:.3e} (tol {TOL_SELF_FD:.0e})");
    assert!(worst < TOL_SELF_FD, "analytic misses own FD by {worst:.3e}");
}
