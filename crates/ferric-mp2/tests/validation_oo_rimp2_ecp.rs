//! VALIDATION tier — OO-RI-MP2 with an ECP (energy and nuclear gradient).
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_oo_rimp2_ecp \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! | system | ECP | basis | aux | checked |
//! |---|---|---|---|---|
//! | HI (r = 1.75 Å, off r_e ≈ 1.609 Å; testdata/molecules/validation/ecp/hi.xyz) | I, def2 28-core | def2-SVP | def2-svp-rifit | RHF, RI-MP2, OO-RI-MP2 energy vs numpy; OO gradient on I vs own 5-point FD |
//!
//! ENERGY: ferric's `oo_ri_mp2` against the independent numpy OO-RI-MP2 of
//! `scripts/validation/gen_oo_rimp2_ecp.py` — the same functional as the
//! all-electron OO row (`validation_oo_rimp2.rs`), with hcore = PySCF
//! `get_hcore()` = T + V_nuc(Z − N_core) + V_ECP, the ECP taken from ferric's
//! own def2-SVP JSON (like-for-like checked by the generator: per-atom N_core,
//! electron count, ECP term count). The numpy minimiser shares no code with
//! ferric; it is anchored at κ = 0 to PySCF RHF + DF-MP2 (same auxmol).
//!
//! GRADIENT: ferric's analytic `oo_ri_mp2_gradient` on the iodine atom along
//! the bond (z; the stretched bond makes it far from zero) against a 5-point
//! central FD (h = 2e-3 Bohr) of ferric's own re-converged RHF + OO-RI-MP2
//! energy. The perpendicular components (x, y) are zero by symmetry and are
//! asserted so on the analytic gradient.
//! Self-FD needs no external code; it tests that the analytic gradient
//! differentiates the energy ferric actually minimises, which with an ECP
//! includes the Σ γ dV_ECP/dR term (`ferric_scf::gradient::ecp_gradient`
//! with the relaxed total 1-PDM).
//!
//! Reference side (generator, 2026-09-25): numpy κ = 0 vs PySCF RHF + DF-MP2
//! −5.7e-14 Ha; numpy minimum −297.369197888 Ha (1.05e-3 Ha below RI-MP2)
//! at max |dE/dκ| 7.0e-10; tr(D V_ECP) = 50.464 Ha.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * Right: E_RHF and E_corr(RI-MP2) match PySCF to the multi-centre ECP
//!   integral floor the RHF+ECP row measured (0.8e-8..2.9e-8 Ha, libecpint vs
//!   PySCF), and the OO total matches the numpy minimum to about the same
//!   floor; the analytic gradient matches the self-FD to the FD floor.
//! * V_ECP missing from the OO hcore (the defect this row was built for): the
//!   OO energy is a different Hamiltonian; at κ = 0 alone it sits
//!   tr(D V_ECP) = 50.46 Ha below (JSON `ecp_effect`). The energy check
//!   misses by ≫ its bar.
//! * dV_ECP/dR missing from the OO gradient: the self-FD misses by the ECP
//!   term, which the always-on resolving-power check asserts is > 1000× the
//!   gradient bar on the I atom.
//! * Harness: a wrong basis/aux/geometry fails E_nuc, nao, naux or N_core
//!   before any OO number is read; a wrong SCF fails the RHF anchor.
//!
//! # TOLERANCES
//!
//! | quantity | measured | bar |
//! |---|---:|---:|
//! | RHF energy vs PySCF | 1.5e-8 Ha | [`TOL_E_SCF`] 2.5e-7 Ha |
//! | RI-MP2 E_corr vs PySCF DF-MP2 | 7.0e-10 Ha | [`TOL_E_CORR`] 1e-8 Ha |
//! | OO total vs numpy | 1.5e-8 Ha | [`TOL_E_OO_NUMPY`] 2.5e-7 Ha |
//! | OO reference / doubles split vs numpy | 1.5e-8 / 5.5e-11 Ha | [`TOL_E_OO_SPLIT`] 2.5e-7 Ha |
//! | OO gradient (I atom, z) vs own FD | 2.3e-9 Ha/Bohr | [`TOL_G_SELF_FD`] 5e-8 Ha/Bohr |
//!
//! The 1.5e-8 Ha offsets are the libecpint-vs-PySCF ECP integral difference
//! (the RHF energy carries the same amount), not an OO effect.
//!
//! # NEGATIVE CONTROLS / MUTATIONS
//!
//! * Always on: the stored κ = 0 energy of the ECP-less functional must miss
//!   the numpy OO total by more than [`MUST_MISS_E_ECP`]; plain RI-MP2 must
//!   miss it by more than [`MUST_MISS_E_OO`] (the bar distinguishes OO from
//!   non-OO).
//! * Always on: the ECP gradient term on I, Σ D dV_ECP/dR_I, must exceed
//!   [`ECP_GRAD_MUST_EXCEED`].
//! * MUTATION A: in `oo_rimp2.rs` revert `hcore_ecp_with_external(obs, mol,
//!   obs.basis_set(), ext)` to `hcore_with_external(obs, ext)` — the OO energy
//!   check fails.
//! * MUTATION B: in `oo_rimp2_gradient.rs` delete `grad +=
//!   &ecp_gradient(mol, obs, &dm1_total_ao)?;` — the energy passes, the
//!   self-FD fails.
//!
//! Run 2026-09-25: A fails the energy test here and the ECP zero-rotation anchor
//! in `oo_rimp2_ecp.rs`; B leaves the energy test green and fails the self-FD.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config, OoRiMp2Result};
use ferric_mp2::oo_rimp2_gradient::oo_ri_mp2_gradient;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::gradient::ecp_gradient;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/oo_rimp2_ecp";
const MOL_DIR: &str = "testdata/molecules/validation/ecp";
const SYSTEM: &str = "hi";
const BASIS: &str = "def2-svp";
const AUX: &str = "def2-svp-rifit";
/// Index of I in hi.xyz (H, I).
const IODINE: usize = 1;

const TOL_ENUC: f64 = 1e-9;
// Measured 1.5e-8 Ha.
const TOL_E_SCF: f64 = 2.5e-7;
// Measured 7.0e-10 Ha.
const TOL_E_CORR: f64 = 1e-8;
// Measured 1.5e-8 Ha: the libecpint-vs-PySCF ECP integral offset.
const TOL_E_OO_NUMPY: f64 = 2.5e-7;
// Measured 1.5e-8 (reference part) / 5.5e-11 Ha (doubles).
const TOL_E_OO_SPLIT: f64 = 2.5e-7;
/// Generator's own convergence bar on the numpy minimum.
const NUMPY_MAX_ORBITAL_GRAD: f64 = 1e-8;
// Measured 2.3e-9 Ha/Bohr.
const TOL_G_SELF_FD: f64 = 5e-8;
const H_FD: f64 = 2e-3;
/// The ECP-less functional must miss the OO reference by 1000× the energy bar.
const MUST_MISS_E_ECP: f64 = 1000.0 * TOL_E_OO_NUMPY;
/// Plain RI-MP2 must miss the OO minimum by this (all-electron row: 6e-4..9e-4).
const MUST_MISS_E_OO: f64 = 1e-6;
/// The ECP gradient term on I must exceed 1000× the gradient bar.
const ECP_GRAD_MUST_EXCEED: f64 = 1000.0 * TOL_G_SELF_FD;

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
        .expect("ferric-mp2 manifest dir should be <root>/crates/ferric-mp2")
        .to_path_buf()
}

fn reference() -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{SYSTEM}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_oo_rimp2_ecp.py — a missing reference is a failure, never a skip",
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
    eprintln!("{ctx}: {what:<28} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_miss(ctx: &str, what: &str, got: f64, want: f64, must: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<28} |d| {d:.2e} (must exceed {must:.0e})");
    assert!(
        d > must,
        "{ctx}: negative control {what}: |d| {d:.2e} <= {must:.0e}"
    );
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
    op: Operator,
}

/// `mol` must already carry the ECP (`apply_ecp`); clones of it keep it.
fn build(mol: Molecule, bs: &BasisSet) -> Sys {
    let obs = PreparedBasis::new(&mol, bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    Sys {
        mol,
        obs,
        dfbs,
        bounds,
        op,
    }
}

/// Load HI on the ECP path and check the harness anchors.
fn load(r: &Value, bs: &BasisSet) -> Sys {
    let ctx = format!("{SYSTEM}/{BASIS}");
    assert_eq!(r["basis"].as_str(), Some(BASIS), "{ctx}: basis");
    assert_eq!(r["aux_basis"].as_str(), Some(AUX), "{ctx}: aux basis");
    let xyz = workspace_root().join(MOL_DIR).join(format!("{SYSTEM}.xyz"));
    let mut mol = Molecule::load_xyz_with_charge(
        xyz.to_str().unwrap(),
        r["charge"].as_i64().unwrap() as i32,
        r["multiplicity"].as_u64().unwrap() as usize,
    )
    .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    mol.apply_ecp(bs);
    let core: Vec<i64> = mol.atoms.iter().map(|a| a.n_core_ecp as i64).collect();
    let core_ref: Vec<i64> = r["ecp_core_electrons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(core, core_ref, "{ctx}: per-atom ECP core electrons");
    assert!(core[IODINE] > 0, "{ctx}: I must carry the ECP");
    assert_eq!(
        mol.nelec() as i64,
        r["nelectron"].as_i64().unwrap(),
        "{ctx}: electron count"
    );
    check_close(
        &ctx,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    let s = build(mol, bs);
    assert_eq!(
        s.obs.nbasis() as u64,
        r["nao"].as_u64().unwrap(),
        "{ctx}: nao"
    );
    assert_eq!(
        s.dfbs.nbasis() as u64,
        r["naux"].as_u64().unwrap(),
        "{ctx}: naux"
    );
    s
}

fn solve_closed(s: &Sys, ctx: &str) -> ScfResult {
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    };
    assert!(
        cfg.df_j_aux.is_none() && cfg.df_k_aux.is_none(),
        "exact J/K"
    );
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &s.mol,
        &s.obs,
        s.op,
        &s.bounds,
        &cfg,
    )
    .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
    assert!(rhf.converged, "{ctx}: RHF not converged");
    rhf
}

fn solve_oo(s: &Sys, rhf: &ScfResult, ctx: &str) -> OoRiMp2Result {
    let cfg = OoRiMp2Config {
        grad_conv: 1e-8,
        energy_conv: 1e-11,
        max_iter: 200,
        ..Default::default()
    };
    assert_eq!(cfg.frozen_core, 0, "all non-ECP electrons correlated");
    let oo = oo_ri_mp2(&s.mol, &s.obs, &s.dfbs, s.op, &s.bounds, rhf, &cfg, None)
        .unwrap_or_else(|e| panic!("{ctx}: oo_ri_mp2 failed: {e:?}"));
    assert!(
        oo.converged,
        "{ctx}: OO-RI-MP2 not converged (|g| {:.2e})",
        oo.grad_norm
    );
    oo
}

#[test]
#[ignore = "validation: OO-RI-MP2 ECP"]
fn oo_rimp2_hi_def2svp_ecp_energy_vs_numpy() {
    let r = reference();
    let ctx = format!("{SYSTEM}/{BASIS}");
    let bs = basis::bundled(BASIS).unwrap();
    let s = load(&r, &bs);

    let gmax = num(&r, "/numpy_oo/max_orbital_gradient", &ctx);
    assert!(
        gmax <= NUMPY_MAX_ORBITAL_GRAD,
        "{ctx}: numpy reference not converged (max |dE/dκ| {gmax:.2e})"
    );

    // ---- exactness anchor: RHF and plain RI-MP2 (κ = 0) ----
    let rhf = solve_closed(&s, &ctx);
    check_close(
        &ctx,
        "E_RHF vs PySCF",
        rhf.energy,
        num(&r, "/pyscf/e_scf", &ctx),
        TOL_E_SCF,
    );
    let mp2_cfg = RiMp2Config::default();
    assert_eq!(mp2_cfg.frozen_core, 0);
    let mp2 = ri_mp2(&s.mol, &s.obs, &s.dfbs, s.op, &rhf, &mp2_cfg).unwrap();
    check_close(
        &ctx,
        "E_corr(RI) vs PySCF DF-MP2",
        mp2.mp2_corr,
        num(&r, "/pyscf/e_corr_dfmp2", &ctx),
        TOL_E_CORR,
    );

    // ---- OO-RI-MP2 ----
    let oo = solve_oo(&s, &rhf, &ctx);
    let e_oo_ref = num(&r, "/numpy_oo/e_total", &ctx);
    check_close(
        &ctx,
        "E_OO reference part",
        oo.hf_energy,
        num(&r, "/numpy_oo/e_reference", &ctx),
        TOL_E_OO_SPLIT,
    );
    check_close(
        &ctx,
        "E_OO doubles part",
        oo.mp2_corr,
        num(&r, "/numpy_oo/e_doubles", &ctx),
        TOL_E_OO_SPLIT,
    );
    check_close(
        &ctx,
        "E_OO total vs numpy",
        oo.total_energy,
        e_oo_ref,
        TOL_E_OO_NUMPY,
    );

    // ---- negative controls ----
    check_miss(
        &ctx,
        "ECP-less κ=0 vs numpy OO",
        num(&r, "/ecp_effect/kappa0_total_without_vecp", &ctx),
        e_oo_ref,
        MUST_MISS_E_ECP,
    );
    check_miss(
        &ctx,
        "RI-MP2 vs numpy OO",
        mp2.total_energy,
        e_oo_ref,
        MUST_MISS_E_OO,
    );
}

#[test]
#[ignore = "validation: OO-RI-MP2 ECP"]
fn oo_rimp2_hi_def2svp_ecp_gradient_self_fd() {
    let r = reference();
    let ctx = format!("{SYSTEM}/{BASIS} self-FD");
    let bs = basis::bundled(BASIS).unwrap();
    let s = load(&r, &bs);
    let rhf = solve_closed(&s, &ctx);
    let oo = solve_oo(&s, &rhf, &ctx);
    let g = oo_ri_mp2_gradient(&s.mol, &s.obs, &s.dfbs, s.op, &s.bounds, &oo, 0, None)
        .unwrap_or_else(|e| panic!("{ctx}: oo_ri_mp2_gradient failed: {e:?}"));

    // Resolving power: the ECP term alone on I, with the OO orbitals'
    // occupied density (the relaxed-density correction is small against it).
    let nocc = (s.mol.nelec() / 2) as usize;
    let c_occ = oo.mos.slice(ndarray::s![.., ..nocc]);
    let d_occ: Array2<f64> = c_occ.dot(&c_occ.t()) * 2.0;
    let g_ecp = ecp_gradient(&s.mol, &s.obs, &d_occ).unwrap();
    let ecp_i = (0..3).map(|k| g_ecp[(IODINE, k)].abs()).fold(0.0, f64::max);
    eprintln!("{ctx}: max |Σ D dV_ECP/dR_I| {ecp_i:.3e} (must exceed {ECP_GRAD_MUST_EXCEED:.0e})");
    assert!(
        ecp_i > ECP_GRAD_MUST_EXCEED,
        "{ctx}: ECP gradient term on I {ecp_i:.2e} too small for the self-FD to detect its loss"
    );

    // Linear molecule along z: the perpendicular components vanish by symmetry.
    for k in 0..2 {
        let gk = g[(IODINE, k)].abs();
        assert!(
            gk < TOL_G_SELF_FD,
            "{ctx}: I[{k}] analytic {gk:.2e} must vanish by symmetry (tol {TOL_G_SELF_FD:.0e})"
        );
    }
    let stencil = [
        (-2.0, 1.0 / 12.0),
        (-1.0, -8.0 / 12.0),
        (1.0, 8.0 / 12.0),
        (2.0, -1.0 / 12.0),
    ];
    let k = 2;
    let mut fd = 0.0;
    for (step, w) in stencil {
        let mut m = s.mol.clone();
        m.atoms[IODINE].zpos += step * H_FD;
        let sd = build(m, &bs);
        let rd = solve_closed(&sd, &ctx);
        fd += w * solve_oo(&sd, &rd, &ctx).total_energy;
    }
    fd /= H_FD;
    let worst = (g[(IODINE, k)] - fd).abs();
    eprintln!(
        "{ctx}: I[z] analytic {:+.10} FD {fd:+.10} |d| {worst:.2e} (tol {TOL_G_SELF_FD:.0e})",
        g[(IODINE, k)]
    );
    assert!(
        worst < TOL_G_SELF_FD,
        "{ctx}: OO analytic gradient on I vs own FD max |d| {worst:.2e} >= {TOL_G_SELF_FD:.0e}"
    );
}
