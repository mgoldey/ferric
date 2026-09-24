//! VALIDATION tier — VALIDATION.md row "UHF / ROHF".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_open_shell_scf \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric UHF and ROHF against PySCF 2.13 on four radicals whose ground states
//! are stability-sensitive — HO2 (²A''), NO2 (²A1), CH2 (³B1) and the allyl
//! radical (²A2) — each at 6-31G AND def2-SVP. References come from
//! `scripts/validation/gen_uhf_rohf.py` and live in
//! `testdata/reference/validation/uhf_rohf/<system>_<basis>.json`. PySCF was fed
//! ferric's OWN bundled basis JSON and ferric's geometry in Bohr (see
//! `scripts/validation/common.py`), so the only differences left are the
//! integral engine (libint2 vs libcint), the SCF convergence on each side, and
//! — the thing this row actually tests — WHICH STATE each code lands on.
//!
//! Every reference state was followed to an internally stable solution by a
//! PySCF `stability()` loop from three guesses; the generator refuses to write
//! an unstable one. ferric is run with `check_stability` +
//! `scf_stability_descent` so it too must end on a stable UHF state.
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric is right: the energies agree to the integral/convergence floor
//!   (~1e-9, see TOLERANCES), ferric's UHF verdict is STABLE, and its λ_min,
//!   ⟨S²⟩ and frontier orbital energies match PySCF's.
//! * If ferric lands on a different state (the 2026-09-17 guess defect class):
//!   the energy misses by mHa-to-eV — five or more orders of magnitude above
//!   the bar, so no tolerance choice can blur the two outcomes.
//! * If the HARNESS is broken (geometry constant, basis mismatch, wrong file):
//!   the nuclear-repulsion assertion fails first (geometry), or every system
//!   misses by a smooth, basis-dependent amount (basis) — the cross-basis and
//!   UHF/ROHF discrimination assertions below exist to make a swapped or
//!   mislabelled reference file fail loudly instead of passing by accident.
//!
//! # TOLERANCES (measured 2026-09-24, all 8 systems x 2 references)
//!
//! | quantity | measured max |d| | bar | headroom |
//! |---|---:|---:|---:|
//! | total energy (UHF and ROHF) | 4.04e-12 Ha | 1e-10 Ha | 25x |
//! | <S^2> (UHF) | 3.13e-7 | 1e-6 | 3x |
//! | HOMO / LUMO (UHF) | 2.44e-8 Ha | 2e-7 Ha | 8x |
//! | lambda_min (UHF orbital Hessian) | 6.73e-8 Ha | 5e-7 Ha | 7x |
//! | nuclear repulsion | 4.26e-14 Ha | 1e-9 Ha | geometry check |
//!
//! The energy is second order in the density error, which is why it sits
//! four orders below the first-order quantities. Its bar keeps 25x (not the
//! protocol's 3-10x) because this tier also runs on CI hardware, where SCF
//! paths have been seen to differ from this box. The derived floor before
//! measurement was ~1e-9 (libint2 vs libcint at integral_thresh 1e-12); the
//! measured 4e-12 shows the two integral libraries agree far better here.
//!
//! # NEGATIVE CONTROLS / MUTATIONS (the test must be able to fail)
//!
//! * MUTATION A — revert the open-shell guess fix and the descent: set
//!   `use_sad_guess: false` and `scf_stability_descent: false` in
//!   [`uhf_config`]. On any system whose JSON records
//!   `uhf.stability.rounds > 0` (PySCF itself had to follow an instability),
//!   ferric returns the saddle and the energy assertion fails by the saddle
//!   gap; the STABLE-verdict assertion fails on every such system regardless.
//!   (Which systems need rounds is a MEASUREMENT recorded in the JSON, not
//!   assumed here; if none do, this mutation reaches only the verdict branch
//!   and that must be stated when the row is re-graded.)
//! * MUTATION B — point [`reference`] at the wrong basis (swap the two
//!   basis strings). The in-test cross-basis assertion
//!   (`assert_basis_discriminates`) fails, and so does the energy assertion:
//!   6-31G and def2-SVP differ by ~0.05-0.2 Ha on these systems.
//! * Always-on resolving-power check: ferric's UHF energy is asserted to MISS
//!   the ROHF reference by far more than the tolerance (UHF < ROHF strictly for
//!   a spin-contaminated radical), so the energy bar demonstrably separates two
//!   nearby, physically distinct states on every system.
//!
//! A missing reference JSON is a HARD failure (panic naming the path) — a
//! silent skip is exactly defect F3.

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::{ScfResult, Spin};
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/uhf_rohf";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASES: [&str; 2] = ["6-31g", "def2-svp"];

const TOL_E: f64 = 1e-10;
const TOL_S2: f64 = 1e-6;
const TOL_EPS: f64 = 2e-7;
const TOL_LAMBDA: f64 = 5e-7;
const TOL_ENUC: f64 = 1e-9;
/// A reference that ferric's energy must MISS: 1000× the energy bar.
const MUST_MISS: f64 = 1000.0 * TOL_E;

/// Workspace root, found by walking up from the CWD (nextest sets the CWD to
/// the package dir, and `--workspace-remap` makes that the checkout);
/// `CARGO_MANIFEST_DIR` is only a fallback because it is baked in at compile
/// time and is wrong inside a nextest archive.
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
             scripts/validation/gen_uhf_rohf.py — a missing reference is a failure, never a skip",
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

/// ⟨S²⟩ of a UHF determinant from its MOs (ferric does not expose it on
/// `ScfResult`, so it is computed here from ferric's own orbitals and overlap).
fn s_squared(res: &ScfResult, s: &Array2<f64>, na: usize, nb: usize) -> f64 {
    let sz = 0.5 * (na as f64 - nb as f64);
    let ca = res.mos_alpha.slice(ndarray::s![.., ..na]);
    let cb = res.mos_beta.as_ref().unwrap().slice(ndarray::s![.., ..nb]);
    let ov = ca.t().dot(s).dot(&cb);
    sz * (sz + 1.0) + nb as f64 - ov.iter().map(|v| v * v).sum::<f64>()
}

fn uhf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        check_stability: true,
        scf_stability_descent: true,
        ..Default::default()
    }
}

fn rohf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        // ferric's ROHF deliberately skips the stability check (the Roothaan
        // Hessian is a third operator). The REFERENCE is ROHF-stable, so a
        // ferric ROHF saddle shows up as an energy miss instead.
        check_stability: false,
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

/// Run ferric UHF on one system × basis and check it against the reference.
/// Returns ferric's energy for the cross-basis check.
fn check_uhf(system: &str, basis_name: &str) -> f64 {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}/UHF");
    let sys = load_system(system, basis_name, &r);
    let res = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &uhf_config())
        .unwrap_or_else(|e| panic!("{ctx}: solve_uhf failed: {e:?}"));
    assert!(res.converged, "{ctx}: not converged");
    assert!(
        matches!(res.spin, Spin::Unrestricted),
        "{ctx}: not a UHF result"
    );

    let st = res
        .stability
        .as_ref()
        .unwrap_or_else(|| panic!("{ctx}: check_stability was set but no verdict came back"));
    eprintln!("{ctx}: {}", st.summary());
    assert_eq!(
        st.verdict(),
        StabilityVerdict::Stable,
        "{ctx}: ferric's final UHF state is not STABLE: {}",
        st.summary()
    );

    check_close(
        &ctx,
        "energy",
        res.energy,
        num(&r, "/uhf/energy", &ctx),
        TOL_E,
    );
    check_close(
        &ctx,
        "lambda_min",
        st.lowest_eigenvalue,
        num(&r, "/uhf/stability/lambda_min", &ctx),
        TOL_LAMBDA,
    );

    let (na, nb) = nocc_ab(&sys.mol);
    assert_eq!(
        na as i64,
        r["uhf"]["nelec_alpha"].as_i64().unwrap(),
        "{ctx}: n_alpha"
    );
    assert_eq!(
        nb as i64,
        r["uhf"]["nelec_beta"].as_i64().unwrap(),
        "{ctx}: n_beta"
    );
    let s = ferric_integrals::oneelectron::overlap(&sys.prep);
    check_close(
        &ctx,
        "<S^2>",
        s_squared(&res, &s, na, nb),
        num(&r, "/uhf/s_squared", &ctx),
        TOL_S2,
    );
    let eb = res.eps_beta.as_ref().unwrap();
    check_close(
        &ctx,
        "HOMO alpha",
        res.eps_alpha[na - 1],
        num(&r, "/uhf/homo_alpha", &ctx),
        TOL_EPS,
    );
    check_close(
        &ctx,
        "LUMO alpha",
        res.eps_alpha[na],
        num(&r, "/uhf/lumo_alpha", &ctx),
        TOL_EPS,
    );
    check_close(
        &ctx,
        "HOMO beta",
        eb[nb - 1],
        num(&r, "/uhf/homo_beta", &ctx),
        TOL_EPS,
    );
    check_close(
        &ctx,
        "LUMO beta",
        eb[nb],
        num(&r, "/uhf/lumo_beta", &ctx),
        TOL_EPS,
    );

    // Resolving power: the UHF energy must MISS the ROHF reference by far more
    // than the bar. If this ever fails, the energy assertion above cannot tell
    // UHF from ROHF on this system and proves nothing about state selection.
    let e_rohf_ref = num(&r, "/rohf/energy", &ctx);
    assert!(
        e_rohf_ref - res.energy > MUST_MISS,
        "{ctx}: UHF {:.10} is not resolvably below the ROHF reference {e_rohf_ref:.10}",
        res.energy
    );
    res.energy
}

/// Run ferric ROHF on one system × basis and check it against the reference.
fn check_rohf(system: &str, basis_name: &str) -> f64 {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}/ROHF");
    let sys = load_system(system, basis_name, &r);
    let res = solve_rohf(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        Operator::coulomb(),
        &sys.bounds,
        &rohf_config(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: solve_rohf failed: {e:?}"));
    assert!(res.converged, "{ctx}: not converged");
    assert!(
        matches!(res.spin, Spin::RestrictedOpen),
        "{ctx}: not an ROHF result"
    );
    check_close(
        &ctx,
        "energy",
        res.energy,
        num(&r, "/rohf/energy", &ctx),
        TOL_E,
    );

    // Resolving power, mirrored: ROHF must miss the (lower) UHF reference.
    let e_uhf_ref = num(&r, "/uhf/energy", &ctx);
    assert!(
        res.energy - e_uhf_ref > MUST_MISS,
        "{ctx}: ROHF {:.10} is not resolvably above the UHF reference {e_uhf_ref:.10}",
        res.energy
    );
    res.energy
}

/// Two-input requirement: the result must RESPOND to the basis. ferric's
/// energy in basis A must miss basis B's reference by far more than the bar,
/// so a stub, a basis ignored somewhere in the chain, or a swapped/mislabelled
/// reference file cannot pass both bases.
fn assert_basis_discriminates(system: &str, method: &str, energies: &[f64; 2]) {
    for (i, b_self) in BASES.iter().enumerate() {
        let b_other = BASES[1 - i];
        let other = num(
            &reference(system, b_other),
            &format!("/{method}/energy"),
            system,
        );
        assert!(
            (energies[i] - other).abs() > MUST_MISS,
            "{system}/{method}: ferric {b_self} energy {:.10} is within {MUST_MISS:.0e} of the \
             {b_other} reference {other:.10} — the comparison does not respond to the basis",
            energies[i]
        );
    }
}

fn uhf_row(system: &str) {
    let e = [check_uhf(system, BASES[0]), check_uhf(system, BASES[1])];
    assert_basis_discriminates(system, "uhf", &e);
}

fn rohf_row(system: &str) {
    let e = [check_rohf(system, BASES[0]), check_rohf(system, BASES[1])];
    assert_basis_discriminates(system, "rohf", &e);
}

#[test]
#[ignore = "validation: UHF / ROHF"]
fn uhf_ho2_vs_pyscf() {
    uhf_row("ho2");
}

#[test]
#[ignore = "validation: UHF / ROHF"]
fn rohf_ho2_vs_pyscf() {
    rohf_row("ho2");
}

#[test]
#[ignore = "validation: UHF / ROHF"]
fn uhf_no2_vs_pyscf() {
    uhf_row("no2");
}

#[test]
#[ignore = "validation: UHF / ROHF"]
fn rohf_no2_vs_pyscf() {
    rohf_row("no2");
}

#[test]
#[ignore = "validation: UHF / ROHF"]
fn uhf_ch2_triplet_vs_pyscf() {
    uhf_row("ch2_triplet");
}

#[test]
#[ignore = "validation: UHF / ROHF"]
fn rohf_ch2_triplet_vs_pyscf() {
    rohf_row("ch2_triplet");
}

#[test]
#[ignore = "validation: UHF / ROHF"]
fn uhf_allyl_vs_pyscf() {
    uhf_row("allyl");
}

#[test]
#[ignore = "validation: UHF / ROHF"]
fn rohf_allyl_vs_pyscf() {
    rohf_row("allyl");
}
