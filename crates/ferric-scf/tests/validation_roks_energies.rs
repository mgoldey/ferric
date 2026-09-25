//! VALIDATION tier — VALIDATION.md row "KS-DFT, ROKS".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_roks_energies \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's ROKS — `ferric_scf::rohf::solve_rohf` with `RhfConfig::xc` set,
//! which is what the CLI runs for `method.kind = "rohf"` + `[dft] functional`
//! (crates/ferric-cli/src/lib.rs `run_rohf`) — against PySCF 2.13 `dft.ROKS`
//! fed ferric's basis JSON, aux JSON and Bohr geometry
//! (`scripts/validation/common.py`):
//!
//! NH2 (²B1), planar CH3 (²A2''), HO2 (²A'') and OH (²Π) × PBE, B3LYP ×
//! 6-31G and def2-SVP. OH is the DEGENERATE π-hole case that the F6 occupation
//! work (`crate::rohf_occupation`, `tests/rohf_degenerate_open_shell.rs`)
//! fixed; it is in the row on purpose.
//!
//! References: `scripts/validation/gen_roks_energies.py` →
//! `testdata/reference/validation/roks_energies/<system>_<basis>.json`.
//!
//! # The like-for-like recipe (read from the code)
//!
//! * Grid (75,110) unpruned Becke + Becke-1988 radii (ferric's default, the
//!   KS-DFT row's recipe).
//! * J/K: `solve_rohf` resolves `j_aux_eff = df_j_aux` (ω = 0) and
//!   `k_aux_eff = df_k_aux` when exact exchange is used, through the same
//!   `fock_assembly::build_df_jk` as `solve_uhf`; the CLI's
//!   `scf_xc_and_aux_defaults` puts def2-universal-jkfit in both for a
//!   functional. So PBE is RI-J, B3LYP is RI-J + RI-K; PySCF
//!   `ROKS(...).density_fit(aux)` is the same.
//! * Coupling: ferric's `roothaan_fock` is a port of PySCF
//!   `get_roothaan_fock` (Guest–Saunders). The converged ENERGY does not depend
//!   on the coupling choice, so it is the compared quantity; Roothaan orbital
//!   energies are coupling-dependent and are not compared.
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's ROKS is right: energies agree at the fitting/grid floor the
//!   UKS half of the KS-DFT row measured (≤7.7e-13 Ha).
//! * If the Roothaan assembly is wrong (a coupling block taking the wrong spin
//!   Fock): the SCF converges to a different stationary point or not at all —
//!   an mHa-scale miss or a convergence failure, never a 1e-10 one.
//! * If the XC is applied as closed-shell / to the wrong spin density: every
//!   system misses by mHa; the ROHF control (below) catches an ignored `xc`.
//! * If ferric lands on a different STATE (OH: the σ*-occupied state the F6
//!   defect converged to, 0.58 Ha high; or a hole orientation PySCF does not
//!   reach): ≥0.1 Ha for a state change; ~1e-6 Ha for a hole orientation (the
//!   Lebedev grid is 4-fold, not cylindrically, symmetric about the OH axis),
//!   which is why OH has its own bar ([`TOL_E_OH`]).
//! * If the HARNESS is broken (geometry, basis, a mislabelled file): the
//!   nuclear-repulsion and AO-count assertions fail first; the functional and
//!   basis discrimination checks catch a swapped block or file.
//!
//! # OH: PySCF itself is unstable here (MEASURED by the generator)
//!
//! PySCF's plain DIIS ROKS on OH does not reach conv_tol 1e-10 in 300 cycles
//! from any of minao/atom/huckel (6-31G, PBE, measured 2026-09-25); the
//! generator therefore polishes every start with `mf.newton()` and accepts a
//! start at max |orbital gradient| ≤ 1e-5. The accepted starts span
//! `accepted_spread` (recorded per block) — the hole-orientation grid
//! anisotropy. The reference is the LOWEST accepted point; ferric must land
//! within [`TOL_E_OH`] of it, which is set above that spread and five orders
//! below the nearest other state.
//!
//! # TOLERANCES
//!
//! | quantity | measured max \|d\| | bar |
//! |---|---:|---:|
//! | ROKS energy, NH2 / CH3 / HO2 (PBE, B3LYP) | 8.0e-13 (HO2/def2-SVP PBE) | [`TOL_E`] 1e-11 Ha |
//! | ROKS energy, OH (PBE, B3LYP) | 8.0e-7 (def2-SVP B3LYP; PySCF's own start spread there is 8.0e-7) | [`TOL_E_OH`] 5e-6 Ha |
//! | nuclear repulsion | 3.6e-15 | [`TOL_ENUC`] 1e-9 Ha |
//!
//! # NEGATIVE CONTROLS (asserted)
//!
//! * ROKS vs UKS: ROKS is a spin-constrained UKS, so ferric's ROKS energy must
//!   lie ABOVE the stability-followed PySCF UKS energy (same functional, basis
//!   and recipe) by more than [`MIN_ROKS_MINUS_UKS`]. The generator measured
//!   ROKS − UKS = +4.1e-4 (OH/6-31G PBE) to +1.7e-3 Ha (HO2/def2-SVP B3LYP)
//!   over all 16 blocks (spin contamination). A
//!   `solve_rohf` that silently ran unrestricted would sit ON the UKS energy.
//! * ROHF vs ROKS: ferric ROKS must miss the exact-integral ROHF control by
//!   more than [`CONTROL_GAP`] (the functional is applied).
//! * Functional: ferric's PBE energy must miss the B3LYP reference (and vice
//!   versa) by ≥ [`MUST_MISS_FACTOR`] × the bar.
//! * Basis: ferric's 6-31G energy must miss the def2-SVP reference (and vice
//!   versa) by ≥ [`MUST_MISS_FACTOR`] × the bar.
//!
//! # MUTATION (to run once, record the outcome here)
//!
//! In `crates/ferric-scf/src/rohf.rs` `roothaan_fock`, swap the spin Fock of
//! the open–closed coupling block:
//!
//! ```text
//! f = &f + &p_o_t.dot(f_b).dot(&p_c);   ->   f = &f + &p_o_t.dot(f_a).dot(&p_c);
//! ```
//!
//! (the line after `f = &f + &(0.5 * p_v_t.dot(&f_c).dot(&p_v));`). That makes
//! the diagonalized operator enforce f_α[o,c] = 0 instead of the true
//! stationarity condition f_β[o,c] = 0, while the DIIS error
//! (`rohf_gradient_mo`) still measures the true gradient — NOT an identity at
//! convergence where f_α − f_β ≠ 0 on the open–closed block.
//!
//! Outcome (2026-09-25): both HO2 tests fail; CH3, NH2 and OH pass. In those
//! three the SOMO's irrep has no doubly occupied partner (CH3 a2'', NH2 b1, OH
//! π against σ closed shells), so the open–closed block of every totally
//! symmetric operator is zero by symmetry and the mutant is an identity there.
//! HO2 (Cs, SOMO a'' with a'' closed shells) is the case that keeps the
//! coupling block honest; do not drop it from the set.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::Spin;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/roks_energies";
const MOL_DIR: &str = "testdata/molecules/validation";
const AUX: &str = "def2-universal-jkfit";
const BASES: [&str; 2] = ["6-31g", "def2-svp"];
const XCS: [&str; 2] = ["pbe", "b3lyp"];

/// ROKS energy, non-degenerate SOMO systems. Measured ≤ 8.0e-13 Ha.
const TOL_E: f64 = 1e-11;
/// ROKS energy, OH (degenerate π hole). Measured ≤ 8.0e-7 Ha; the bar sits above the
/// hole-orientation spread of PySCF's accepted starts (generator
/// `accepted_spread`: 1.26e-6 / 1.31e-6 Ha at 6-31G PBE / B3LYP, 1.4e-7 /
/// 8.0e-7 at def2-SVP) and ≥ 4 orders below the nearest other state.
/// (NH2/CH3/HO2 spreads are ≤ 6e-13 Ha.)
const TOL_E_OH: f64 = 5e-6;
const TOL_ENUC: f64 = 1e-9;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 100.0;
/// KS vs HF: the functional must move the energy by more than this.
const CONTROL_GAP: f64 = 1e-2;
/// ROKS − UKS must exceed this (1e-4 Ha); the generator
/// measured ROKS − UKS ≥ +4.06e-4 Ha (OH/6-31G PBE) over all 16 blocks.
const MIN_ROKS_MINUS_UKS: f64 = 1e-4;

/// Workspace root, found by walking up from the CWD (nextest sets the CWD to
/// the package dir); `CARGO_MANIFEST_DIR` is only a fallback because it is
/// baked in at compile time and is wrong inside a nextest archive.
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

/// Load a reference JSON. Missing or unparsable is a HARD failure.
fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_roks_energies.py — a missing reference is a failure, never a skip",
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

fn tol_e(system: &str) -> f64 {
    if system == "oh" {
        TOL_E_OH
    } else {
        TOL_E
    }
}

/// ferric's spelling of each functional (what the CLI / RhfConfig accept).
fn ferric_xc(xc: &str) -> &'static str {
    match xc {
        "pbe" => "PBE",
        "b3lyp" => "B3LYP",
        other => panic!("unknown functional {other}"),
    }
}

/// ROKS config, J/K matched to the generator (and to the CLI's defaults for
/// `rohf` + `[dft] functional`): RI-J always, RI-K when exchange is used.
fn roks_config(xc: &str) -> RhfConfig {
    let df_k_aux = match xc {
        "pbe" => None,
        "b3lyp" => Some(AUX.to_string()),
        other => panic!("unknown functional {other}"),
    };
    let cfg = RhfConfig {
        xc: Some(ferric_xc(xc).into()),
        df_j_aux: Some(AUX.to_string()),
        df_k_aux,
        max_iter: 500,
        energy_conv: 1e-10,
        // Same as the UKS half of the KS-DFT row (validation_ks_energies.rs).
        density_conv: 1e-7,
        ..Default::default()
    };
    // The F6 occupation guard is what makes OH converge to its ground state;
    // it is ON by default and must stay on here (the CLI sets it true).
    assert!(cfg.rohf_occupation_guard, "F6 occupation guard must be on");
    assert_eq!(cfg.mom_after_iter, 0, "MOM would disable the F6 guard");
    cfg
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = got - want;
    eprintln!("{ctx}: {what:<12} ferric {got:+.12} ref {want:+.12} d {d:+.2e} (tol {tol:.0e})");
    assert!(
        d.abs() < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {:.2e} >= {tol:.0e})",
        d.abs()
    );
}

/// Run ferric ROKS on one system × basis × functional, check it and its
/// per-file negative controls, and return the energy.
fn check_roks(system: &str, basis_name: &str, xc: &str) -> f64 {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}/ROKS-{xc}");
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    // Geometry like-for-like FIRST.
    check_close(
        &ctx,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    let prep = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    assert_eq!(
        ferric_integrals::oneelectron::overlap(&prep).nrows() as u64,
        r["nao"].as_u64().expect("nao"),
        "{ctx}: AO count differs from the reference's"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let block = format!("roks_{xc}");

    // The reference itself: the generator's acceptance, re-checked so a
    // hand-edited JSON cannot slip through.
    let spread = num(&r, &format!("/{block}/accepted_spread"), &ctx);
    eprintln!("{ctx}: PySCF accepted-start spread {spread:.2e} Ha");
    assert!(
        spread < 0.5 * tol_e(system),
        "{ctx}: PySCF's own accepted ROKS starts spread {spread:.2e} Ha, not resolvable at the \
         {:.0e} bar — the reference state is not well defined",
        tol_e(system)
    );

    let res = solve_rohf(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &roks_config(xc),
    )
    .unwrap_or_else(|e| panic!("{ctx}: solve_rohf failed: {e:?}"));
    assert!(res.converged, "{ctx}: not converged");
    assert!(
        matches!(res.spin, Spin::RestrictedOpen),
        "{ctx}: not a restricted open-shell result"
    );
    eprintln!("{ctx}: {} iterations", res.iterations);

    let e_ref = num(&r, &format!("/{block}/energy"), &ctx);
    check_close(&ctx, "energy", res.energy, e_ref, tol_e(system));

    // ---- negative controls ----
    // ROKS is a constrained UKS: strictly above the UKS minimum.
    let e_uks = num(&r, &format!("/uks_{xc}_control/energy"), &ctx);
    let gap = res.energy - e_uks;
    eprintln!("{ctx}: ROKS - UKS(ref) {gap:+.3e} Ha (must exceed {MIN_ROKS_MINUS_UKS:.0e})");
    assert!(
        gap > MIN_ROKS_MINUS_UKS,
        "{ctx}: ferric ROKS {:.10} is not above the UKS reference {e_uks:.10} by \
         {MIN_ROKS_MINUS_UKS:.0e} (ROKS - UKS = {gap:+.2e}) — is the solve unrestricted?",
        res.energy
    );
    // Functional applied at all.
    let e_rohf = num(&r, "/rohf_control/energy", &ctx);
    assert!(
        (res.energy - e_rohf).abs() > CONTROL_GAP,
        "{ctx}: ROKS {:.10} within {CONTROL_GAP:.0e} of the ROHF control {e_rohf:.10} — the \
         functional is not being applied",
        res.energy
    );
    // Functional discriminates.
    let must_miss = MUST_MISS_FACTOR * tol_e(system);
    for other in XCS.iter().filter(|o| **o != xc) {
        let e_other = num(&r, &format!("/roks_{other}/energy"), &ctx);
        assert!(
            (res.energy - e_other).abs() > must_miss,
            "{ctx}: ferric {xc} energy within {must_miss:.0e} of the {other} reference — the \
             comparison does not respond to the functional"
        );
    }
    res.energy
}

fn roks_row(system: &str, xc: &str) {
    let e = [
        check_roks(system, BASES[0], xc),
        check_roks(system, BASES[1], xc),
    ];
    let must_miss = MUST_MISS_FACTOR * tol_e(system);
    for (i, b_self) in BASES.iter().enumerate() {
        let b_other = BASES[1 - i];
        let other = num(
            &reference(system, b_other),
            &format!("/roks_{xc}/energy"),
            system,
        );
        assert!(
            (e[i] - other).abs() > must_miss,
            "{system}/{xc}: ferric {b_self} energy {:.10} within {must_miss:.0e} of the \
             {b_other} reference {other:.10} — the comparison does not respond to the basis",
            e[i]
        );
    }
}

#[test]
#[ignore = "validation: KS-DFT, ROKS"]
fn roks_nh2_pbe_vs_pyscf() {
    roks_row("nh2", "pbe");
}

#[test]
#[ignore = "validation: KS-DFT, ROKS"]
fn roks_nh2_b3lyp_vs_pyscf() {
    roks_row("nh2", "b3lyp");
}

#[test]
#[ignore = "validation: KS-DFT, ROKS"]
fn roks_ch3_pbe_vs_pyscf() {
    roks_row("ch3", "pbe");
}

#[test]
#[ignore = "validation: KS-DFT, ROKS"]
fn roks_ch3_b3lyp_vs_pyscf() {
    roks_row("ch3", "b3lyp");
}

#[test]
#[ignore = "validation: KS-DFT, ROKS"]
fn roks_ho2_pbe_vs_pyscf() {
    roks_row("ho2", "pbe");
}

#[test]
#[ignore = "validation: KS-DFT, ROKS"]
fn roks_ho2_b3lyp_vs_pyscf() {
    roks_row("ho2", "b3lyp");
}

/// OH ²Π: the degenerate π hole (F6). ferric must land on PySCF's ROKS
/// ground state; see the module doc for why the bar is [`TOL_E_OH`].
#[test]
#[ignore = "validation: KS-DFT, ROKS"]
fn roks_oh_pbe_vs_pyscf() {
    roks_row("oh", "pbe");
}

#[test]
#[ignore = "validation: KS-DFT, ROKS"]
fn roks_oh_b3lyp_vs_pyscf() {
    roks_row("oh", "b3lyp");
}
