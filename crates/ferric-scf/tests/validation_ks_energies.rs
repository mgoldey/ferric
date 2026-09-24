//! VALIDATION tier — VALIDATION.md row "KS-DFT".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_ks_energies \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! The older `dft_{lda,pbe,b3lyp,wb97xv}.rs` tests cover CLOSED-SHELL KS on
//! first-row molecules at cc-pVDZ and def2-SVP. This file widens the row on the
//! two axes it lacked, against PySCF 2.13 fed ferric's own basis JSON, aux JSON
//! and Bohr geometry (`scripts/validation/common.py`):
//!
//! 1. **Open-shell UKS** (`solve_uhf` with `xc`): NH2 (²B1), planar CH3
//!    (²A2''), HO2 (²A'') and O2 (³Σg⁻) × PBE, B3LYP, ωB97X-V × 6-31G and
//!    def2-SVP. Radicals with a DEGENERATE singly occupied shell (OH, NO, CH)
//!    are left out on purpose: their symmetric and symmetry-broken UKS
//!    solutions are near-degenerate, so a mismatch there is a state-selection
//!    question (the F6 ROKS lane), not a KS-energy one. ROKS is not tested here
//!    for the same reason.
//! 2. **Closed-shell second row, triple zeta** (the CLI's `solve_rhf_ladder`
//!    path, exactly as `dft_pbe.rs`/`dft_b3lyp.rs` drive it): H2S, HCl, SiH4 ×
//!    PBE, B3LYP × def2-SVP and def2-TZVP.
//!
//! References: `scripts/validation/gen_ks_energies.py` →
//! `testdata/reference/validation/ks_energies/<system>_<basis>.json`.
//!
//! # The like-for-like recipe (read from the code, not from a doc)
//!
//! Grid (75,110) unpruned Becke + Becke-1988 radii, VV10 on (50,50) — ferric's
//! defaults, the recipe the existing closed-shell tests already match. J/K is
//! matched PER CODE PATH, because ferric's open- and closed-shell solvers do
//! not fit the same things:
//!
//! | path | ferric (this file's config) | PySCF (generator) |
//! |---|---|---|
//! | RKS PBE | RI-J (`df_j_aux`) | `density_fit` |
//! | RKS B3LYP | RI-J + RI-K | `density_fit` |
//! | UKS PBE | RI-J (`df_j_aux`) | `density_fit` |
//! | UKS B3LYP | RI-J + RI-K | `density_fit` |
//! | UKS ωB97X-V | **exact J** (uhf.rs: `j_aux_eff` is `None` for ω > 0) + RI-K as c_SR K[erfc] + c_LR K[erf] | exact J + attenuated-metric DF-K, full-range K served as K_SR + K_LR |
//!
//! A `density_fit` reference for UKS ωB97X-V would have compared ferric's
//! EXACT J against PySCF's RI-J, i.e. measured the fitting error.
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's KS is right: energies agree at the grid/fit floor the
//!   closed-shell tests measured (PBE 2.1e-8, B3LYP 1.6e-8, ωB97X-V 3.1e-5 Ha),
//!   and ⟨S²⟩ / frontier orbitals / λ_min agree to first-order noise.
//! * If the spin-polarized XC path is wrong (α/β stride, ζ handling, a factor
//!   in the UKS f_xc or VV10 on a polarized density): the open-shell energies
//!   miss by mHa or more while the closed-shell half of this file still
//!   passes — the two halves are designed to separate those outcomes.
//! * If ferric lands on a different STATE: mHa-to-eV miss, plus a failed
//!   STABLE verdict (PBE/B3LYP) or a ⟨S²⟩ miss (ωB97X-V, see below).
//! * If the HARNESS is broken (geometry constant, basis, wrong/mislabelled
//!   file, a functional silently not applied): the nuclear-repulsion and AO
//!   count assertions fail first; the cross-basis, cross-functional and
//!   UKS-vs-UHF assertions below make a swapped file or an ignored `xc` fail
//!   loudly rather than pass by accident.
//!
//! # TOLERANCES (measured 2026-09-24, all systems x both bases)
//!
//! | quantity | measured max |d| | bar |
//! |---|---:|---:|
//! | UKS PBE / B3LYP energy | 6.3e-13 / 7.7e-13 Ha | 1e-10 Ha |
//! | RKS PBE / B3LYP energy (2nd row, def2-SVP, def2-TZVP) | 3.1e-12 / 2.7e-12 Ha | 1e-10 Ha |
//! | UKS ωB97X-V energy | 6.0e-6 Ha | 2e-5 Ha |
//! | ⟨S²⟩ UKS PBE / B3LYP | 4.0e-10 / 4.6e-9 | 5e-8 |
//! | ⟨S²⟩ UKS ωB97X-V | 1.3e-7 | 1e-6 |
//! | HOMO/LUMO, PBE and B3LYP (UKS and RKS) | 1.4e-7 Ha | 1e-6 Ha |
//! | HOMO/LUMO, ωB97X-V (UKS) | 6.1e-5 Ha (α LUMO) | 3e-4 Ha |
//! | λ_min (UKS PBE/B3LYP) | 9.0e-6 Ha | 5e-5 Ha |
//! | nuclear repulsion | 1.4e-14 Ha | 1e-9 Ha |
//!
//! The PBE/B3LYP energies agree to ~1e-12 because both codes use ferric's own
//! basis and aux JSON, the same (75,110) Becke grid and the same RI recipe.
//! ωB97X-V sits five orders higher; its construction differs in the VV10
//! pairwise sum on the (50,50) grid and the range-separated DF-K metrics
//! (see the generator's `_UksExactJDfK`), and its virtual orbitals are the
//! most sensitive (6e-5). λ_min differs by ~9e-6 because ferric's stability
//! Hessian uses exact-J/K response at a DF density while PySCF uses the
//! fitted response.
//!
//! # State checks
//!
//! * PBE / B3LYP UKS: ferric runs `check_stability` + `scf_stability_descent`
//!   and must end STABLE; λ_min is compared with PySCF's dense UKS Hessian.
//! * ωB97X-V UKS: ferric's stability analysis SKIPS range-separated
//!   functionals (`StabilitySkip::RangeSeparated` — the Hessian's exchange
//!   response is plain Coulomb). The state is then pinned by the energy and
//!   ⟨S²⟩ matching a stability-followed reference. If ferric ever returns a
//!   verdict here it must be STABLE, and this paragraph must be updated.
//! * O2: ⟨S²⟩ within 0.05 of 2. UKS at multiplicity 3 fixes N_α − N_β = 2, so
//!   it cannot become a singlet; the state within the triplet manifold is
//!   pinned by the energy and ⟨S²⟩ matching PySCF's stability-checked
//!   reference.
//!
//! # NEGATIVE CONTROLS / MUTATIONS (the test must be able to fail)
//!
//! * Always on — basis: ferric's energy in basis A must MISS basis B's
//!   reference by ≥ 100× the bar (`assert_basis_discriminates`).
//! * Always on — functional: ferric's energy with functional X must miss the
//!   reference of every other functional in the same file by ≥ 100× the bar.
//!   A dropped/ignored `xc` or a mislabelled reference block fails here.
//! * Always on — functional applied at all: UKS must miss the exact UHF
//!   control, RKS the exact RHF control, by > 1e-2 Ha.
//! * MUTATION A (to run once, record the outcome here) — swap the α/β
//!   densities passed to the UKS XC evaluation (or set the UKS XC to the
//!   closed-shell kernel on ρ_α+ρ_β): every open-shell energy must fail while
//!   every closed-shell case still passes.
//! * MUTATION B — set `df_j_aux: None` for the UKS PBE config (exact J against
//!   an RI-J reference): the PBE energies must fail by the RI-J error. If they
//!   do NOT, the energy bar is looser than the fitting error and cannot see a
//!   J/K recipe mismatch — tighten it.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ladder::{default_ladder_from, solve_rhf_ladder};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::{ScfResult, Spin};
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/ks_energies";
const MOL_DIR: &str = "testdata/molecules/validation";
const AUX: &str = "def2-universal-jkfit";
const OPEN_SHELL_BASES: [&str; 2] = ["6-31g", "def2-svp"];
const CLOSED_SHELL_BASES: [&str; 2] = ["def2-svp", "def2-tzvp"];
const OPEN_SHELL_XC: [&str; 3] = ["pbe", "b3lyp", "wb97x-v"];
const CLOSED_SHELL_XC: [&str; 2] = ["pbe", "b3lyp"];

// Bars from the measured maxima (module doc table), ~10x headroom.
const TOL_E_PBE: f64 = 1e-10;
const TOL_E_B3LYP: f64 = 1e-10;
const TOL_E_WB97XV: f64 = 2e-5;
const TOL_S2_GGA: f64 = 5e-8;
const TOL_S2_WB97XV: f64 = 1e-6;
const TOL_EPS_GGA: f64 = 1e-6;
const TOL_EPS_WB97XV: f64 = 3e-4;
const TOL_LAMBDA: f64 = 5e-5;
const TOL_ENUC: f64 = 1e-9;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 100.0;
/// KS vs HF: the functional must move the energy by more than this.
const CONTROL_GAP: f64 = 1e-2;

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
             scripts/validation/gen_ks_energies.py — a missing reference is a failure, never a skip",
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

fn tol_e(xc: &str) -> f64 {
    match xc {
        "pbe" => TOL_E_PBE,
        "b3lyp" => TOL_E_B3LYP,
        "wb97x-v" => TOL_E_WB97XV,
        other => panic!("no energy bar for functional {other}"),
    }
}

fn tol_s2(xc: &str) -> f64 {
    if xc == "wb97x-v" {
        TOL_S2_WB97XV
    } else {
        TOL_S2_GGA
    }
}

fn tol_eps(xc: &str) -> f64 {
    if xc == "wb97x-v" {
        TOL_EPS_WB97XV
    } else {
        TOL_EPS_GGA
    }
}

/// ferric's spelling of each functional (what the CLI / RhfConfig accept).
fn ferric_xc(xc: &str) -> &'static str {
    match xc {
        "pbe" => "PBE",
        "b3lyp" => "B3LYP",
        "wb97x-v" => "wB97X-V",
        other => panic!("unknown functional {other}"),
    }
}

struct System {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn load_system(system: &str, basis_name: &str, reference: &Value) -> System {
    let charge = reference["charge"].as_i64().expect("charge") as i32;
    let mult = reference["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let ctx = format!("{system}/{basis_name}");
    // Geometry like-for-like FIRST: a constant or unit slip fails here as a
    // geometry defect, not later as an unexplained energy offset.
    let enuc_ref = num(reference, "/nuclear_repulsion", &ctx);
    let enuc = mol.nuclear_repulsion();
    eprintln!(
        "{ctx}: E_nuc ferric {enuc:.12} ref {enuc_ref:.12} |d| {:.2e}",
        (enuc - enuc_ref).abs()
    );
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{ctx}: nuclear repulsion {enuc:.12} vs reference {enuc_ref:.12} — geometry/unit mismatch"
    );
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let nao_ref = reference["nao"].as_u64().expect("nao") as usize;
    let s = ferric_integrals::oneelectron::overlap(&prep);
    assert_eq!(
        s.nrows(),
        nao_ref,
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

fn nocc_ab(mol: &Molecule) -> (usize, usize) {
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    ((nelec + two_s) / 2, (nelec - two_s) / 2)
}

/// ⟨S²⟩ of the UKS determinant from ferric's own MOs and overlap (the KS
/// "⟨S²⟩" of the non-interacting reference — the same quantity PySCF's
/// `spin_square` reports for UKS).
fn s_squared(res: &ScfResult, s: &Array2<f64>, na: usize, nb: usize) -> f64 {
    let sz = 0.5 * (na as f64 - nb as f64);
    let ca = res.mos_alpha.slice(ndarray::s![.., ..na]);
    let cb = res.mos_beta.as_ref().unwrap().slice(ndarray::s![.., ..nb]);
    let ov = ca.t().dot(s).dot(&cb);
    sz * (sz + 1.0) + nb as f64 - ov.iter().map(|v| v * v).sum::<f64>()
}

/// UKS config, J/K matched to the generator per functional (module-doc table).
fn uks_config(xc: &str) -> RhfConfig {
    let (df_j_aux, df_k_aux) = match xc {
        // Pure GGA: RI-J; no exact exchange is consumed.
        "pbe" => (Some(AUX.to_string()), None),
        // Plain hybrid: RI-J + RI-K.
        "b3lyp" => (Some(AUX.to_string()), Some(AUX.to_string())),
        // RSH: the open-shell solver builds J exactly for ω > 0 whatever
        // df_j_aux says, so leave it unset to make that visible; K is fitted.
        "wb97x-v" => (None, Some(AUX.to_string())),
        other => panic!("unknown functional {other}"),
    };
    RhfConfig {
        xc: Some(ferric_xc(xc).into()),
        df_j_aux,
        df_k_aux,
        max_iter: 500,
        energy_conv: 1e-10,
        density_conv: 1e-7,
        check_stability: true,
        scf_stability_descent: true,
        ..Default::default()
    }
}

/// Closed-shell config: identical to dft_pbe.rs / dft_b3lyp.rs, driven through
/// the same `default_ladder_from` + `solve_rhf_ladder` path.
fn rks_config(xc: &str) -> RhfConfig {
    let df_k_aux = match xc {
        "pbe" => None,
        "b3lyp" => Some(AUX.to_string()),
        other => panic!("unknown closed-shell functional {other}"),
    };
    RhfConfig {
        xc: Some(ferric_xc(xc).into()),
        df_j_aux: Some(AUX.to_string()),
        df_k_aux,
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    }
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<12} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

/// ferric's energy with functional `xc` must MISS every OTHER functional's
/// reference in the same file: an ignored `xc` or a mislabelled block fails.
fn assert_functional_discriminates(
    ctx: &str,
    r: &Value,
    prefix: &str,
    xc: &str,
    all: &[&str],
    e: f64,
) {
    let must_miss = MUST_MISS_FACTOR * tol_e(xc);
    for other in all.iter().filter(|o| **o != xc) {
        let e_other = num(r, &format!("/{prefix}_{other}/energy"), ctx);
        assert!(
            (e - e_other).abs() > must_miss,
            "{ctx}: ferric {xc} energy {e:.10} is within {must_miss:.0e} of the {other} \
             reference {e_other:.10} — the comparison does not respond to the functional"
        );
    }
}

/// Two-input requirement: the result must RESPOND to the basis.
fn assert_basis_discriminates(
    system: &str,
    block: &str,
    xc: &str,
    bases: &[&str; 2],
    energies: &[f64; 2],
) {
    let must_miss = MUST_MISS_FACTOR * tol_e(xc);
    for (i, b_self) in bases.iter().enumerate() {
        let b_other = bases[1 - i];
        let other = num(
            &reference(system, b_other),
            &format!("/{block}/energy"),
            system,
        );
        assert!(
            (energies[i] - other).abs() > must_miss,
            "{system}/{block}: ferric {b_self} energy {:.10} is within {must_miss:.0e} of the \
             {b_other} reference {other:.10} — the comparison does not respond to the basis",
            energies[i]
        );
    }
}

/// Run ferric UKS on one system × basis × functional and check it.
fn check_uks(system: &str, basis_name: &str, xc: &str) -> f64 {
    let r = reference(system, basis_name);
    let block = format!("uks_{xc}");
    let ctx = format!("{system}/{basis_name}/UKS-{xc}");
    let sys = load_system(system, basis_name, &r);
    let res = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &uks_config(xc))
        .unwrap_or_else(|e| panic!("{ctx}: solve_uhf failed: {e:?}"));
    assert!(res.converged, "{ctx}: not converged");
    assert!(
        matches!(res.spin, Spin::Unrestricted),
        "{ctx}: not an unrestricted result"
    );

    // State: a verdict must exist and be STABLE for ω = 0 functionals; for the
    // RSH one ferric skips the analysis by design (module doc).
    match (res.stability.as_ref(), xc) {
        (Some(st), _) => {
            eprintln!("{ctx}: {}", st.summary());
            assert_eq!(
                st.verdict(),
                StabilityVerdict::Stable,
                "{ctx}: ferric's final UKS state is not STABLE: {}",
                st.summary()
            );
            check_close(
                &ctx,
                "lambda_min",
                st.lowest_eigenvalue,
                num(&r, &format!("/{block}/stability/lambda_min"), &ctx),
                TOL_LAMBDA,
            );
        }
        (None, "wb97x-v") => eprintln!(
            "{ctx}: ferric stability analysis skipped (range-separated); state pinned by \
             energy + <S^2> against the stability-followed reference"
        ),
        (None, _) => panic!("{ctx}: check_stability was set but no verdict came back"),
    }

    let e_ref = num(&r, &format!("/{block}/energy"), &ctx);
    check_close(&ctx, "energy", res.energy, e_ref, tol_e(xc));

    let (na, nb) = nocc_ab(&sys.mol);
    assert_eq!(
        na as i64,
        r[&block]["nelec_alpha"].as_i64().unwrap(),
        "{ctx}: n_alpha"
    );
    assert_eq!(
        nb as i64,
        r[&block]["nelec_beta"].as_i64().unwrap(),
        "{ctx}: n_beta"
    );
    let s = ferric_integrals::oneelectron::overlap(&sys.prep);
    let s2 = s_squared(&res, &s, na, nb);
    check_close(
        &ctx,
        "<S^2>",
        s2,
        num(&r, &format!("/{block}/s_squared"), &ctx),
        tol_s2(xc),
    );
    let eb = res.eps_beta.as_ref().unwrap();
    for (what, got, ptr) in [
        ("HOMO alpha", res.eps_alpha[na - 1], "homo_alpha"),
        ("LUMO alpha", res.eps_alpha[na], "lumo_alpha"),
        ("HOMO beta", eb[nb - 1], "homo_beta"),
        ("LUMO beta", eb[nb], "lumo_beta"),
    ] {
        check_close(
            &ctx,
            what,
            got,
            num(&r, &format!("/{block}/{ptr}"), &ctx),
            tol_eps(xc),
        );
    }

    // Negative controls.
    assert_functional_discriminates(&ctx, &r, "uks", xc, &OPEN_SHELL_XC, res.energy);
    let e_uhf = num(&r, "/uhf_control/energy", &ctx);
    assert!(
        (res.energy - e_uhf).abs() > CONTROL_GAP,
        "{ctx}: UKS {:.10} is within {CONTROL_GAP:.0e} of the exact UHF control {e_uhf:.10} — \
         the functional is not being applied",
        res.energy
    );
    if system == "o2" {
        assert!(
            (s2 - 2.0).abs() < 0.05,
            "{ctx}: <S^2> = {s2:.6} is not a triplet (expected ~2)"
        );
    }
    res.energy
}

/// Run ferric RKS (ladder) on one system × basis × functional and check it.
fn check_rks(system: &str, basis_name: &str, xc: &str) -> f64 {
    let r = reference(system, basis_name);
    let block = format!("rks_{xc}");
    let ctx = format!("{system}/{basis_name}/RKS-{xc}");
    let sys = load_system(system, basis_name, &r);
    let ladder = default_ladder_from(&rks_config(xc));
    let lr = solve_rhf_ladder(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        Operator::coulomb(),
        &sys.bounds,
        &ladder,
    )
    .unwrap_or_else(|e| panic!("{ctx}: solve_rhf_ladder failed: {e:?}"));
    // NOT gating on lr.converged, exactly as dft_pbe.rs / dft_b3lyp.rs: DF-JK
    // KS can end on a benign density-oscillation floor above density_conv
    // while the energy is converged far below the bar. The energy is the gate;
    // the flag and rung are printed for the record.
    eprintln!(
        "{ctx}: ladder_converged={} rung={}",
        lr.converged, lr.rung_reached
    );
    let res = lr.result;
    check_close(
        &ctx,
        "energy",
        res.energy,
        num(&r, &format!("/{block}/energy"), &ctx),
        tol_e(xc),
    );
    let nocc = (sys.mol.nelec() as usize) / 2;
    check_close(
        &ctx,
        "HOMO",
        res.eps_alpha[nocc - 1],
        num(&r, &format!("/{block}/homo"), &ctx),
        tol_eps(xc),
    );
    check_close(
        &ctx,
        "LUMO",
        res.eps_alpha[nocc],
        num(&r, &format!("/{block}/lumo"), &ctx),
        tol_eps(xc),
    );

    assert_functional_discriminates(&ctx, &r, "rks", xc, &CLOSED_SHELL_XC, res.energy);
    let e_rhf = num(&r, "/rhf_control/energy", &ctx);
    assert!(
        (res.energy - e_rhf).abs() > CONTROL_GAP,
        "{ctx}: RKS {:.10} is within {CONTROL_GAP:.0e} of the exact RHF control {e_rhf:.10} — \
         the functional is not being applied",
        res.energy
    );
    res.energy
}

fn uks_row(system: &str, xc: &str) {
    let e = [
        check_uks(system, OPEN_SHELL_BASES[0], xc),
        check_uks(system, OPEN_SHELL_BASES[1], xc),
    ];
    assert_basis_discriminates(system, &format!("uks_{xc}"), xc, &OPEN_SHELL_BASES, &e);
}

fn rks_row(system: &str, xc: &str) {
    let e = [
        check_rks(system, CLOSED_SHELL_BASES[0], xc),
        check_rks(system, CLOSED_SHELL_BASES[1], xc),
    ];
    assert_basis_discriminates(system, &format!("rks_{xc}"), xc, &CLOSED_SHELL_BASES, &e);
}

// ---------------------------------------------------------------- open shell

#[test]
#[ignore = "validation: KS-DFT energies"]
fn uks_nh2_pbe_vs_pyscf() {
    uks_row("nh2", "pbe");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn uks_nh2_b3lyp_vs_pyscf() {
    uks_row("nh2", "b3lyp");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn uks_nh2_wb97xv_vs_pyscf() {
    uks_row("nh2", "wb97x-v");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn uks_ch3_pbe_vs_pyscf() {
    uks_row("ch3", "pbe");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn uks_ch3_b3lyp_vs_pyscf() {
    uks_row("ch3", "b3lyp");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn uks_ch3_wb97xv_vs_pyscf() {
    uks_row("ch3", "wb97x-v");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn uks_ho2_pbe_vs_pyscf() {
    uks_row("ho2", "pbe");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn uks_ho2_b3lyp_vs_pyscf() {
    uks_row("ho2", "b3lyp");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn uks_ho2_wb97xv_vs_pyscf() {
    uks_row("ho2", "wb97x-v");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn uks_o2_triplet_pbe_vs_pyscf() {
    uks_row("o2", "pbe");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn uks_o2_triplet_b3lyp_vs_pyscf() {
    uks_row("o2", "b3lyp");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn uks_o2_triplet_wb97xv_vs_pyscf() {
    uks_row("o2", "wb97x-v");
}

// ----------------------------------------------- closed shell, second row, TZ

#[test]
#[ignore = "validation: KS-DFT energies"]
fn rks_h2s_pbe_vs_pyscf() {
    rks_row("h2s", "pbe");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn rks_h2s_b3lyp_vs_pyscf() {
    rks_row("h2s", "b3lyp");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn rks_hcl_pbe_vs_pyscf() {
    rks_row("hcl", "pbe");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn rks_hcl_b3lyp_vs_pyscf() {
    rks_row("hcl", "b3lyp");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn rks_sih4_pbe_vs_pyscf() {
    rks_row("sih4", "pbe");
}

#[test]
#[ignore = "validation: KS-DFT energies"]
fn rks_sih4_b3lyp_vs_pyscf() {
    rks_row("sih4", "b3lyp");
}
