//! VALIDATION tier — VALIDATION.md row "COSMO".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_cosmo \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's conductor-like screening model (`ferric_scf::cosmo`, Gaussian-
//! smeared S matrix, SWIG cavity, f(ε) = (ε−1)/(ε+½)) on H2O, NH3, CH3OH and
//! acetate(−) × STO-3G and cc-pVDZ × ε ∈ {4.7, 78.4} (RHF), plus HO2 (²A'') UHF
//! at cc-pVDZ, ε = 78.4. (OH ²Π is not used: its π-hole orientation is a
//! degenerate mode that the non-axial Lebedev cavity only weakly splits, so
//! the solvated UHF reference drifts by 1e-7 Ha along a near-flat mode — see
//! the generator's docstring.) References come from `scripts/validation/gen_cosmo.py`
//! and live in `testdata/reference/validation/cosmo/<system>_<basis>.json`;
//! PySCF was fed ferric's bundled basis JSON and ferric's geometry in Bohr.
//!
//! ferric's model is PySCF `pcm.py` `method="COSMO"` in every piece except
//! one: ferric evaluates the solute potential at each segment, and the
//! reaction field, with POINT charges (1/r), while PySCF uses Gaussian charges
//! of exponent ξ_k² (erf(ξ_k r)/r). The generator therefore writes:
//!
//! * `ferric_model` — PySCF's own SCF + PCM driver with PySCF's
//!   `gen_surface`/`get_D_S` cavity and S matrix, ferric's radii (Bondi × 1.17,
//!   H = 1.20 Å), 110 Lebedev points per sphere, ferric's keep criterion, and
//!   point-charge v / V_reaction. This is an independent construction of
//!   exactly ferric's model, and the tight assertions are against it.
//! * `pyscf_cosmo_matched_radii` — stock PySCF COSMO with ferric's radii. The
//!   gap to it is the point-vs-Gaussian potential FORMULATION difference; it is
//!   asserted only against the looser [`TOL_FORMULATION_REL`].
//! * `ferric_model_h_radius_1p10` — `ferric_model` with only the H radius set
//!   to PySCF's modified-Bondi 1.10 Å. ferric's radii are NOT changed by this
//!   row; this block is the like-for-like radii resolving-power control.
//! * `pyscf_cosmo_modified_bondi` / `pyscf_cosmo_default` — stock PySCF with
//!   its own radii table (and, for `default`, scale 1.2 and 302 points).
//!   Printed only.
//!
//! Exactness anchor: the ε → 1 limit (f(ε) → 0) must reproduce the vacuum SCF
//! energy — [`cosmo_matches_vacuum_in_the_trivial_limit`].
//!
//! # Physics hypothesis vs artifact hypothesis (Experimental Protocol)
//!
//! * If ferric is right: the cavity segment count equals the reference's
//!   exactly, the total area agrees to round-off, and the solvated energy and
//!   E_cosmo agree with `ferric_model` at the SCF floor (~1e-9 Ha) on every
//!   system, both bases, both ε, and the anion.
//! * If the cavity is wrong (radius, scale, Lebedev set, switching function,
//!   keep criterion): the segment count / area assertion fails before any
//!   energy is compared. Changing the H radius 1.20 → 1.10 Å shifts ΔG by
//!   0.24–0.78 kcal/mol on the neutrals and 0.03–0.10 kcal/mol (5e-5–1.7e-4 Ha)
//!   on acetate (reference side, `ferric_model_h_radius_1p10`), above the bar.
//! * If f(ε) used the C-PCM form (ε−1)/ε: at ε = 4.7 f moves from 0.712 to
//!   0.787 (+11%), so the ε = 4.7 energies miss by ~0.3–9 mHa; the direct
//!   `f_epsilon` check fails first.
//! * If the operator ordering / sign of the solve or reaction field were wrong
//!   (the defect class found in IEF-PCM, PR #181): a mis-sign shows up as a
//!   wrong ΔG sign or magnitude on every system; a wrong but symmetric
//!   ordering cannot hide here because K = S is symmetric and R ∝ I, so there
//!   is no operator ordering to cancel against — the check that makes this
//!   meaningful is the ε = 4.7 vs 78.4 pair (f enters linearly) and the anion
//!   (Σq ≈ +f(ε)·|Q| by Gauss's law, asserted against the reference Σq).
//! * If the HARNESS is broken (geometry, basis, charge): nuclear repulsion,
//!   AO count and the vacuum energy are asserted first.
//!
//! # TOLERANCES
//!
//! Each bar is set from the measured maximum recorded on its const: solvated
//! energies 1.9e-11 Ha, E_cosmo 9.7e-11 Ha, Σq 3.7e-11.
//!
//! # NEGATIVE CONTROLS / MUTATIONS
//!
//! * Always-on: ferric's solvated energy must MISS the H = 1.10 Å
//!   `ferric_model` reference and the other-ε reference by ≥ [`MUST_MISS`] (the bar resolves
//!   a radius change and a dielectric change on every system).
//! * Always-on: ferric's solvated energy must MISS the vacuum reference by
//!   ≥ [`MUST_MISS`] (a COSMO term silently dropped cannot pass).
//! * MUTATION A — in `cosmo.rs::bondi_radius_angstrom` set H to 1.10: the
//!   segment-count/area assertion fails on every system.
//! * MUTATION B — in `CosmoConfig::f_epsilon` use `(eps - 1)/eps`: the
//!   `f_epsilon` assertion and every energy assertion fail.
//! * MUTATION C — flip the sign of `v_elec` in `cosmo_reaction_field`: every
//!   energy assertion fails by ≥ 1e-2 Ha.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosmo::{cosmo_reaction_field, CosmoCavity, CosmoConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/cosmo";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASES: [&str; 2] = ["sto-3g", "cc-pvdz"];
const HARTREE_TO_KCAL: f64 = 627.509_474_063_1;

/// Vacuum SCF energy vs PySCF. Measured max 8.0e-12.
const TOL_E_VAC: f64 = 1e-10;
/// Solvated energy vs `ferric_model` (PySCF's COSMO driver with ferric's
/// radii, grid, keep rule and point-charge potentials). Measured max 1.9e-11.
const TOL_E_SOLV: f64 = 1e-10;
/// E_cosmo (½ q·v at the converged density) vs `ferric_model.e_solvent`.
/// First order in the density error. Measured max 9.7e-11.
const TOL_E_COSMO: f64 = 1e-9;
/// Σq vs reference, first order. Measured max 3.7e-11.
const TOL_SUM_Q: f64 = 5e-10;
/// Total cavity area, relative. Same formula, same points; propose 1e-12.
const TOL_AREA_REL: f64 = 1e-12;
/// |ΔG_ferric − ΔG_pyscf_stock| / |ΔG_pyscf_stock| — point vs Gaussian
/// potential, same cavity. Reference side (`ferric_model` vs
/// `pyscf_cosmo_matched_radii`, 17 cases): 0.05%–0.51%, max NH3/cc-pVDZ/ε=78.4
/// (−4.5137 vs −4.4907 kcal/mol); acetate cc-pVDZ 0.17%. 2% = 4× the max.
const TOL_FORMULATION_REL: f64 = 0.02;
const TOL_ENUC: f64 = 1e-9;
/// A reference ferric's energy must MISS: 1000× the solvated bar.
const MUST_MISS: f64 = 1000.0 * TOL_E_SOLV;

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
        .expect("ferric-scf manifest dir should be <root>/crates/ferric-scf")
        .to_path_buf()
}

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_cosmo.py — a missing reference is a failure, never a skip",
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
    eprintln!("{ctx}: {what:<14} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_misses(ctx: &str, what: &str, got: f64, other: f64) {
    let d = (got - other).abs();
    eprintln!("{ctx}: resolving power vs {what}: |d| {d:.2e} (must be >= {MUST_MISS:.0e})");
    assert!(
        d >= MUST_MISS,
        "{ctx}: ferric's energy is within {d:.2e} of the {what} reference — the bar cannot \
         tell the two apart (or the reference file is mislabelled)"
    );
}

struct System {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn load_system(system: &str, basis_name: &str, r: &Value) -> System {
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let ctx = format!("{system}/{basis_name}");
    check_close(
        &ctx,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    assert_eq!(
        prep.nbasis(),
        r["nao"].as_u64().expect("nao") as usize,
        "{ctx}: AO count differs from the reference's"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    System {
        mol,
        prep,
        bounds,
        ctx: ParallelContext::default(),
    }
}

fn scf_config(cosmo: Option<CosmoConfig>) -> RhfConfig {
    RhfConfig {
        max_iter: 300,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        cosmo,
        ..Default::default()
    }
}

fn cosmo_config(r: &Value, eps: f64) -> CosmoConfig {
    let lebedev = r["lebedev_points"].as_u64().expect("lebedev_points") as usize;
    CosmoConfig {
        epsilon: eps,
        radius_scale: num(r, "/radius_scale", "cosmo_config"),
        lebedev_order: lebedev,
        ..Default::default()
    }
}

fn run(sys: &System, method: &str, cfg: &RhfConfig, ctx: &str) -> ScfResult {
    let res = match method {
        "rhf" => solve_rhf(
            &sys.ctx,
            &sys.mol,
            &sys.prep,
            Operator::coulomb(),
            &sys.bounds,
            cfg,
        ),
        "uhf" => solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, cfg),
        other => panic!("{ctx}: unknown method {other}"),
    }
    .unwrap_or_else(|e| panic!("{ctx}: SCF failed: {e:?}"));
    assert!(res.converged, "{ctx}: SCF not converged");
    res
}

/// ⟨S²⟩ of a UHF determinant from ferric's own orbitals.
fn s_squared(res: &ScfResult, s: &Array2<f64>, na: usize, nb: usize) -> f64 {
    let sz = 0.5 * (na as f64 - nb as f64);
    let ca = res.mos_alpha.slice(ndarray::s![.., ..na]);
    let cb = res.mos_beta.as_ref().unwrap().slice(ndarray::s![.., ..nb]);
    let ov = ca.t().dot(s).dot(&cb);
    sz * (sz + 1.0) + nb as f64 - ov.iter().map(|v| v * v).sum::<f64>()
}

/// Check one system × basis at every ε the reference carries.
fn check_system(system: &str, basis_name: &str) {
    let r = reference(system, basis_name);
    let method = r["method"].as_str().expect("method").to_string();
    let base_ctx = format!("{system}/{basis_name}/{method}");
    let sys = load_system(system, basis_name, &r);

    let vac = run(&sys, &method, &scf_config(None), &base_ctx);
    let e_vac_ref = num(&r, "/vacuum/energy", &base_ctx);
    check_close(&base_ctx, "E_vacuum", vac.energy, e_vac_ref, TOL_E_VAC);

    let solvated = r["solvated"].as_object().expect("solvated block");
    let all_fm: Vec<(f64, f64)> = solvated
        .values()
        .map(|b| {
            (
                b["epsilon"].as_f64().unwrap(),
                b["ferric_model"]["energy"].as_f64().unwrap(),
            )
        })
        .collect();
    assert!(!solvated.is_empty(), "{base_ctx}: no solvated blocks");

    for block in solvated.values() {
        let eps = block["epsilon"].as_f64().expect("epsilon");
        let ctx = format!("{base_ctx}/eps={eps}");
        let cfg = cosmo_config(&r, eps);

        // f(eps) convention: (eps-1)/(eps+1/2), as PySCF method="COSMO".
        check_close(
            &ctx,
            "f(eps)",
            cfg.f_epsilon(),
            num(block, "/ferric_model/f_epsilon", &ctx),
            1e-15,
        );

        // Cavity like-for-like BEFORE any energy.
        let cavity = CosmoCavity::build(&sys.mol, &cfg).unwrap();
        let nseg_ref = block["ferric_model"]["n_segments"].as_u64().unwrap() as usize;
        eprintln!(
            "{ctx}: cavity n_seg ferric {} ref {nseg_ref} (PySCF-criterion extra points: {})",
            cavity.n_segments(),
            block["ferric_model"]["n_dropped_by_ferric_keep_criterion"]
        );
        assert_eq!(
            cavity.n_segments(),
            nseg_ref,
            "{ctx}: cavity segment count differs from PySCF gen_surface with ferric's radii"
        );
        let area_ref = num(block, "/ferric_model/total_area_bohr2", &ctx);
        check_close(
            &ctx,
            "area/ref-1",
            cavity.total_area() / area_ref - 1.0,
            0.0,
            TOL_AREA_REL,
        );

        // Solvated SCF vs the like-for-like reference.
        let res = run(&sys, &method, &scf_config(Some(cfg.clone())), &ctx);
        let e_ref = num(block, "/ferric_model/energy", &ctx);
        check_close(&ctx, "E_solvated", res.energy, e_ref, TOL_E_SOLV);

        let cr =
            cosmo_reaction_field(&sys.mol, &sys.prep, &cavity, &cfg, &res.density_total).unwrap();
        check_close(
            &ctx,
            "E_cosmo",
            cr.e_cosmo,
            num(block, "/ferric_model/e_solvent", &ctx),
            TOL_E_COSMO,
        );
        check_close(
            &ctx,
            "sum_q",
            cr.charges.sum(),
            num(block, "/ferric_model/sum_q", &ctx),
            TOL_SUM_Q,
        );

        if method == "uhf" {
            let s2_ref = num(block, "/ferric_model/stability/s_squared", &ctx);
            let (na, nb) = {
                let nelec = sys.mol.nelec() as usize;
                let two_s = sys.mol.multiplicity - 1;
                ((nelec + two_s) / 2, (nelec - two_s) / 2)
            };
            let s = ferric_integrals::oneelectron::overlap(&sys.prep);
            // First-order quantity; measured 7.5e-10 (HO2).
            check_close(&ctx, "<S^2>", s_squared(&res, &s, na, nb), s2_ref, 1e-8);
        }

        // Report the gaps that are NOT the like-for-like comparison.
        let dg = res.energy - vac.energy;
        let dg_stock = num(block, "/pyscf_cosmo_matched_radii/energy", &ctx) - e_vac_ref;
        let dg_h110 = num(block, "/ferric_model_h_radius_1p10/energy", &ctx) - e_vac_ref;
        let dg_modb = num(block, "/pyscf_cosmo_modified_bondi/energy", &ctx) - e_vac_ref;
        let dg_def = num(block, "/pyscf_cosmo_default/energy", &ctx) - e_vac_ref;
        let rel_stock = (dg - dg_stock).abs() / dg_stock.abs();
        eprintln!(
            "{ctx}: dG kcal/mol ferric {:+.4} | PySCF stock COSMO, ferric radii {:+.4} \
             (rel {:.2e}) | ferric model with H 1.10 A {:+.4} | PySCF modified-Bondi x1.17 \
             {:+.4} | PySCF all-default {:+.4}",
            dg * HARTREE_TO_KCAL,
            dg_stock * HARTREE_TO_KCAL,
            rel_stock,
            dg_h110 * HARTREE_TO_KCAL,
            dg_modb * HARTREE_TO_KCAL,
            dg_def * HARTREE_TO_KCAL
        );
        assert!(
            dg < 0.0,
            "{ctx}: solvation energy must be stabilizing, got {dg:.3e}"
        );
        assert!(
            rel_stock < TOL_FORMULATION_REL,
            "{ctx}: point-charge-potential ferric vs stock (Gaussian-potential) PySCF COSMO \
             differ by {rel_stock:.3e} relative (> {TOL_FORMULATION_REL})"
        );

        // Negative controls (always on).
        check_misses(&ctx, "vacuum", res.energy, e_vac_ref);
        check_misses(
            &ctx,
            "H radius 1.10 A (modified Bondi)",
            res.energy,
            num(block, "/ferric_model_h_radius_1p10/energy", &ctx),
        );
        for &(eps_other, e_other) in &all_fm {
            if eps_other != eps {
                check_misses(&ctx, &format!("eps={eps_other}"), res.energy, e_other);
            }
        }
    }
}

/// Exactness anchor: f(ε) → 0 as ε → 1, so the COSMO SCF must reproduce the
/// vacuum SCF. f(1+1e-10) ≈ 6.7e-11, so the shift is ~1e-10 × |E_cosmo(78)|
/// ≈ 1e-12 Ha. Measured 5.8e-13; bar 1e-10 (SCF convergence floor).
#[test]
#[ignore = "validation: COSMO"]
fn cosmo_matches_vacuum_in_the_trivial_limit() {
    for basis_name in BASES {
        let r = reference("h2o", basis_name);
        let ctx = format!("h2o/{basis_name}/anchor");
        let sys = load_system("h2o", basis_name, &r);
        let vac = run(&sys, "rhf", &scf_config(None), &ctx);
        let cfg = cosmo_config(&r, 1.0 + 1e-10);
        let near_vac = run(&sys, "rhf", &scf_config(Some(cfg)), &ctx);
        check_close(&ctx, "E(eps->1)", near_vac.energy, vac.energy, 1e-10);
    }
}

#[test]
#[ignore = "validation: COSMO"]
fn cosmo_h2o_vs_pyscf() {
    for b in BASES {
        check_system("h2o", b);
    }
}

#[test]
#[ignore = "validation: COSMO"]
fn cosmo_nh3_vs_pyscf() {
    for b in BASES {
        check_system("nh3", b);
    }
}

#[test]
#[ignore = "validation: COSMO"]
fn cosmo_ch3oh_vs_pyscf() {
    for b in BASES {
        check_system("ch3oh", b);
    }
}

#[test]
#[ignore = "validation: COSMO"]
fn cosmo_acetate_anion_vs_pyscf() {
    for b in BASES {
        check_system("acetate", b);
    }
}

/// HO2 ²A'' UHF, cc-pVDZ, ε = 78.4: COSMO on the spin-summed density.
#[test]
#[ignore = "validation: COSMO"]
fn cosmo_ho2_uhf_vs_pyscf() {
    check_system("ho2", "cc-pvdz");
}
