//! VALIDATION tier — VALIDATION.md row "External potential in MP2 gradients".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_qmmm_mp2_gradient \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's analytic RI-MP2 gradient in a field of fixed point charges
//! (`rimp2_gradient_analytical(.., Some(&ext))`, RHF with exact four-centre
//! J/K, the charges folded into hcore) against a finite difference of the
//! SAME model's total energy in PySCF 2.13: `qmmm.mm_charge`-wrapped exact
//! RHF + `mp.dfmp2.DFMP2(frozen=None)` with ferric's own `cc-pvdz-ri` aux
//! JSON. PySCF 2.13 has no analytic DF-MP2 gradient, so the reference is a
//! 5-point central FD (h = 2e-3 Bohr); the generator also runs it at 2h and
//! refuses to write unless the two agree to 5e-8 (measured 1.7e-9 and
//! 2.4e-9, so the FD truncation at h is ~1e-10). Two independent methods
//! (analytic Z-vector vs FD of the energy), two independent codes, the same
//! RI approximation — unlike the older `rimp2_gradient_vs_pyscf_qmmm.rs`,
//! which compares RI against canonical MP2 and so cannot see anything below
//! the ~5e-5 Ha/Bohr fitting difference.
//!
//! Systems (references in `testdata/reference/validation/qmmm/`, generator
//! `scripts/validation/gen_qmmm.py`):
//! * H2O / cc-pVDZ + 10 charges (two H-bonded TIP3P waters, ±1 ions, ±0.3);
//! * CH3OH / 6-31G + the 20 TIP3P shell sites nearest the QM atoms.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If the field is threaded through every term of the MP2 gradient (the
//!   Hellmann-Feynman charge-electron derivative contracted with the RELAXED
//!   density, and the charge-nuclear term): the analytic gradient matches the
//!   FD at the Z-vector/FD floor (~1e-9 Ha/Bohr).
//! * If the charge-electron term is contracted with the SCF density instead
//!   of the relaxed MP2 density, or the field is dropped from the Z-vector's
//!   Fock response: the energy still matches (it never sees the gradient
//!   code) but the gradient misses by roughly |P_relaxed − D_SCF| × the field
//!   gradient — far above the bar for charges ~2 Å from the QM atoms.
//! * If the field is dropped entirely: ferric returns the vacuum MP2
//!   gradient, which misses the reference by the embedding force (asserted
//!   FIRST below).
//! * If the harness is broken (geometry, basis, aux): the nuclear repulsion,
//!   AO count, aux count and RHF-gradient anchor fail before the MP2 check.
//!
//! # Exactness anchor
//!
//! Everything except the correlation part: ferric's RHF gradient IN THE
//! FIELD must equal PySCF's analytic RHF gradient in the field
//! (`rhf_gradient` in the JSON) at the HF bar. The MP2 reference must then
//! MISS that RHF gradient (measured 2.7e-2 / 3.1e-2 Ha/Bohr), so the MP2
//! comparison is not satisfied by the SCF part alone.
//!
//! # NEGATIVE CONTROLS (asserted inside every test)
//!
//! * vacuum: ferric's MP2 gradient with `ext = None` must MISS the reference;
//! * charges × 1.01: ferric's RI-MP2 energy must reproduce PySCF's ×1.01
//!   energy and MISS the ×1 reference (gap 1.4e-5 / 1.5e-4 Ha);
//! * SCF-only: ferric's in-field RHF gradient must MISS the MP2 reference.
//! MUTATION to run by hand: pass `None` instead of `Some(&ext)` to
//! `rimp2_gradient_analytical` only (keep the field in the SCF) — the
//! gradient assertion must fail while the energy assertions still pass.
//!
//! # TOLERANCES
//!
//! Each bar is set from the measured maximum recorded on its const.
//! after the first run.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::elements;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::{Atom, Molecule};
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::gradient::rimp2_gradient_analytical;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::gradient::rhf_gradient;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/qmmm";

/// RHF and total RI-MP2 energies (exact J/K, same aux). Measured max 8.4e-12.
const TOL_E: f64 = 1e-10;
/// RHF gradient in the field vs PySCF analytic (the anchor). Measured 1.2e-10.
const TOL_G_RHF: f64 = 1e-9;
/// RI-MP2 gradient vs PySCF central FD (h vs 2h agree to 2.4e-9). Measured
/// max 4.8e-9.
const TOL_G_MP2: f64 = 3e-8;
const TOL_ENUC: f64 = 1e-9;
/// A control must miss the reference by this multiple of its bar.
const MISS_FACTOR: f64 = 10.0;

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

fn reference(file_stem: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{file_stem}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_qmmm.py mp2 — a missing reference is a failure, never a skip",
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

fn xyz3(v: &Value) -> [f64; 3] {
    let a = v.as_array().expect("xyz array");
    [
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
    ]
}

fn rows3(v: &Value, key: &str) -> Vec<[f64; 3]> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("reference field {key} missing"))
        .iter()
        .map(xyz3)
        .collect()
}

fn molecule(r: &Value) -> Molecule {
    Molecule {
        atoms: r["atoms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                let symbol = a["symbol"].as_str().unwrap().to_string();
                let z = elements::symbol_to_z(&symbol).unwrap();
                let [x, y, zpos] = xyz3(&a["xyz_bohr"]);
                Atom {
                    symbol,
                    z,
                    x,
                    y,
                    zpos,
                    ghost: false,
                    n_core_ecp: 0,
                }
            })
            .collect(),
        charge: r["charge"].as_i64().unwrap() as i32,
        multiplicity: r["multiplicity"].as_u64().unwrap() as usize,
    }
}

fn external_potential(r: &Value, q_scale: f64) -> ExternalPotential {
    ExternalPotential {
        point_charges: r["mm_charges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let [x, y, z] = xyz3(&c["xyz_bohr"]);
                PointCharge {
                    q: c["q"].as_f64().unwrap() * q_scale,
                    x,
                    y,
                    z,
                }
            })
            .collect(),
        field: None,
        smeared_charges: Vec::new(),
    }
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<22} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn max_diff(g: &Array2<f64>, want: &[[f64; 3]]) -> (f64, usize, usize) {
    assert_eq!(g.nrows(), want.len());
    let mut worst = (0.0, 0, 0);
    for (a, w) in want.iter().enumerate() {
        for k in 0..3 {
            let d = (g[(a, k)] - w[k]).abs();
            if d > worst.0 {
                worst = (d, a, k);
            }
        }
    }
    worst
}

fn check_grad(ctx: &str, what: &str, g: &Array2<f64>, want: &[[f64; 3]], tol: f64) {
    let (d, a, k) = max_diff(g, want);
    eprintln!("{ctx}: {what:<22} max|d| {d:.2e} at [{a}][{k}] (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what}[{a}][{k}] ferric {} vs reference {} (|d| {d:.2e} >= {tol:.0e})",
        g[(a, k)],
        want[a][k]
    );
}

fn check_grad_misses(ctx: &str, what: &str, g: &Array2<f64>, want: &[[f64; 3]], tol: f64) {
    let (d, _, _) = max_diff(g, want);
    let bar = MISS_FACTOR * tol;
    eprintln!("{ctx}: CONTROL {what:<30} max|d| {d:.2e} (must exceed {bar:.0e})");
    assert!(
        d > bar,
        "{ctx}: negative control '{what}' did NOT miss the reference ({d:.2e} <= {bar:.0e})"
    );
}

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
}

fn setup(r: &Value, ctx: &str) -> Setup {
    let mol = molecule(r);
    let enuc = mol.nuclear_repulsion();
    let enuc_ref = num(r, "/nuclear_repulsion");
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{ctx}: nuclear repulsion {enuc:.12} vs reference {enuc_ref:.12} — geometry/unit mismatch"
    );
    let obs_bs = basis::bundled(r["basis"].as_str().unwrap()).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let nao = ferric_integrals::oneelectron::overlap(&obs).nrows();
    assert_eq!(nao as u64, r["nao"].as_u64().unwrap(), "{ctx}: AO count");
    let aux_bs = basis::bundled(r["aux_basis"].as_str().unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let naux = ferric_integrals::oneelectron::overlap(&dfbs).nrows();
    assert_eq!(naux as u64, r["naux"].as_u64().unwrap(), "{ctx}: aux count");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    Setup {
        mol,
        obs,
        dfbs,
        bounds,
    }
}

fn rhf(s: &Setup, ext: Option<&ExternalPotential>, ctx: &str) -> ScfResult {
    let cfg = RhfConfig {
        external_potential: ext.cloned(),
        energy_conv: 1e-11,
        density_conv: 1e-10,
        max_iter: 300,
        ..Default::default()
    };
    let res = solve_rhf(
        &ParallelContext::default(),
        &s.mol,
        &s.obs,
        Operator::coulomb(),
        &s.bounds,
        &cfg,
    )
    .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
    assert!(res.converged, "{ctx}: RHF not converged");
    res
}

fn mp2_gradient(s: &Setup, scf: &ScfResult, ext: Option<&ExternalPotential>) -> Array2<f64> {
    rimp2_gradient_analytical(
        &s.mol,
        &s.obs,
        &s.dfbs,
        Operator::coulomb(),
        &s.bounds,
        scf,
        &RiMp2Config::default(),
        ext,
    )
    .unwrap()
}

fn check(file_stem: &str) {
    let r = reference(file_stem);
    let ctx = file_stem;
    let s = setup(&r, ctx);
    let ext = external_potential(&r, 1.0);
    let g_ref = rows3(&r, "gradient_fd");
    let g_rhf_ref = rows3(&r, "rhf_gradient");
    let cfg = RiMp2Config::default();
    assert_eq!(cfg.frozen_core, 0, "reference correlates all electrons");

    // --- anti-vacuum control FIRST: the field must matter to the gradient --
    let vac = rhf(&s, None, &format!("{ctx}/vacuum"));
    let g_vac = mp2_gradient(&s, &vac, None);
    check_grad_misses(ctx, "vacuum MP2 gradient", &g_vac, &g_ref, TOL_G_MP2);
    let e_gas = ri_mp2(&s.mol, &s.obs, &s.dfbs, Operator::coulomb(), &vac, &cfg)
        .unwrap()
        .total_energy;
    check_close(
        ctx,
        "E_MP2 gas phase",
        e_gas,
        num(&r, "/energy_gas_phase"),
        TOL_E,
    );

    // --- the SCF part in the field (exactness anchor for all but E_corr) ----
    let scf = rhf(&s, Some(&ext), ctx);
    check_close(ctx, "E_RHF in field", scf.energy, num(&r, "/e_rhf"), TOL_E);
    let g_rhf = rhf_gradient(
        &s.mol,
        &s.obs,
        Operator::coulomb(),
        &s.bounds,
        &scf,
        Some(&ext),
    )
    .unwrap();
    check_grad(ctx, "RHF gradient in field", &g_rhf, &g_rhf_ref, TOL_G_RHF);
    check_grad_misses(
        ctx,
        "RHF-only gradient vs MP2 ref",
        &g_rhf,
        &g_ref,
        TOL_G_MP2,
    );

    // --- RI-MP2 energy and gradient in the field ----------------------------
    let mp2 = ri_mp2(&s.mol, &s.obs, &s.dfbs, Operator::coulomb(), &scf, &cfg).unwrap();
    check_close(
        ctx,
        "E_corr in field",
        mp2.mp2_corr,
        num(&r, "/e_corr"),
        TOL_E,
    );
    check_close(
        ctx,
        "E_MP2 in field",
        mp2.total_energy,
        num(&r, "/energy"),
        TOL_E,
    );
    check_close(
        ctx,
        "MP2 embedding shift",
        mp2.total_energy - e_gas,
        num(&r, "/shift"),
        TOL_E,
    );
    let g = mp2_gradient(&s, &scf, Some(&ext));
    check_grad(ctx, "RI-MP2 gradient", &g, &g_ref, TOL_G_MP2);

    // --- charges x 1.01 ------------------------------------------------------
    let scale = num(&r, "/control_scaled_101/scale");
    let ext_s = external_potential(&r, scale);
    let scf_s = rhf(&s, Some(&ext_s), &format!("{ctx}/x{scale}"));
    let e_s = ri_mp2(&s.mol, &s.obs, &s.dfbs, Operator::coulomb(), &scf_s, &cfg)
        .unwrap()
        .total_energy;
    check_close(
        ctx,
        "E_MP2 charges x1.01",
        e_s,
        num(&r, "/control_scaled_101/energy"),
        TOL_E,
    );
    let d = (e_s - num(&r, "/energy")).abs();
    eprintln!("{ctx}: CONTROL charges x1.01 vs x1 ref |d| {d:.2e}");
    assert!(
        d > MISS_FACTOR * TOL_E,
        "{ctx}: charges x1.01 did NOT miss the x1 reference ({d:.2e})"
    );
}

#[test]
#[ignore = "validation: External potential in MP2 gradients"]
fn rimp2_gradient_in_field_h2o_q10_ccpvdz() {
    check("h2o_q10_rimp2_cc-pvdz");
}

#[test]
#[ignore = "validation: External potential in MP2 gradients"]
fn rimp2_gradient_in_field_ch3oh_q20_631g() {
    check("ch3oh_q20_rimp2_6-31g");
}
