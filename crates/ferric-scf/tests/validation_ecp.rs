//! VALIDATION tier — VALIDATION.md rows "RHF+ECP" and "ECP gradients".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_ecp \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric RHF energies AND analytic nuclear gradients on def2 ECP systems
//! against PySCF 2.13 (analytic `nuc_grad_method()`), and the RHF energies
//! against ORCA 6.1.1 as a third code:
//!
//! | system | ECP | bases | checks |
//! |---|---|---|---|
//! | HI (stretched) | I, def2 28-core | def2-SVP, def2-TZVP | E, ε_HOMO, gradient, ORCA E |
//! | CH3I (distorted) | I | def2-SVP, def2-TZVP | E, ε_HOMO, gradient (x, y, z all move), ORCA E |
//! | RbH (stretched) | Rb, def2 28-core, different channel shapes | def2-SVP, def2-TZVP | E, ε_HOMO, gradient, ORCA E |
//! | HBr (stretched) | NONE — Br is all-electron in def2 | def2-SVP, def2-TZVP | negative control + E, gradient, ORCA E |
//! | SnH4 (one bond stretched) | Sn, def2 28-core | def2-TZVP only | E, ε_HOMO, gradient, ORCA E |
//! | I atom, ²P | I | def2-SVP, def2-TZVP | UHF E, ⟨S²⟩, stability |
//!
//! References: `scripts/validation/gen_ecp.py` (PySCF) and
//! `scripts/validation/gen_ecp_orca.py` (ORCA, inputs committed under
//! `scripts/validation/orca/ecp/`), in `testdata/reference/validation/ecp/`.
//! Both codes were fed ferric's OWN bundled basis JSON, the ECP from the SAME
//! file (ferric's def2 files carry their ECPs inline; that block is what
//! `Molecule::apply_ecp` and libecpint use), and ferric's geometry in Bohr.
//! The generator asserts the per-atom core-electron counts, electron count and
//! ECP term count match ferric's before writing. What is left to differ is the
//! ECP integral code — libecpint (ferric) vs PySCF's own ECP integrals vs
//! ORCA's — plus the ERI engine and SCF convergence on each side.
//!
//! SnH4 is single-basis because ferric's bundled def2-SVP stops at Xe with only
//! Rb, I and Xe carrying ECPs (no Ag, no Sn); it adds a third ECP shape at
//! def2-TZVP and is NOT one of the two-basis inputs the grade rests on.
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's ECP path is right: energies agree to the integral /
//!   convergence floor (≲1e-7 Ha expected; the pre-existing `ecp_rhf.rs`
//!   measured 1e-6..2e-12 against PySCF's built-in def2 copy), and the
//!   analytic gradient agrees to ≲1e-7 Ha/Bohr.
//! * If an ECP channel is mis-mapped (local channel not the highest l, the
//!   r-power shifted, a semilocal term taken as U_l instead of U_l − U_L): the
//!   energy moves by tenths of a Hartree or more, since V_ECP is a large term.
//! * If the `dV_ECP/dR` term is missing or attributed to the wrong atom (the
//!   pre-2026-07-28 defect class): the gradient misses by the ECP term itself,
//!   which the always-on resolving-power check below asserts is > 1000× the
//!   gradient bar on every ECP system.
//! * If N_core or the geometry is wrong on either side: the nuclear-repulsion
//!   assertion fails FIRST (V_nn uses Z − N_core), and the per-atom N_core
//!   assertion names the atom.
//! * If the HARNESS mislabels a file: the cross-basis assertion
//!   (`assert_basis_discriminates`) fails — def2-SVP and def2-TZVP differ by
//!   ≫ the bar on every system.
//!
//! # TOLERANCES (measured 2026-09-24, all systems x both bases)
//!
//! | quantity | measured max \|d\| | bar |
//! |---|---:|---:|
//! | RHF/UHF energy vs PySCF | 2.86e-8 Ha | 2.5e-7 Ha |
//! | ε_HOMO vs PySCF | 2.25e-8 Ha | 1e-7 Ha |
//! | analytic gradient vs PySCF analytic | 2.25e-8 Ha/Bohr | 1e-7 Ha/Bohr |
//! | RHF energy vs ORCA, explicit NewECP | 2.86e-8 Ha | 2.5e-7 Ha |
//! | RHF energy vs ORCA, built-in def2-ECP fallback | — (not generated) | 1e-5 Ha |
//! | ⟨S²⟩ (I atom UHF) | 5.46e-12 | 1e-9 |
//! | nuclear repulsion vs PySCF | 2.22e-16 Ha | 1e-9 Ha |
//! | nuclear repulsion vs ORCA (8 printed decimals) | 3.23e-12 Ha | 1e-7 Ha |
//!
//! Where the residual comes from, measured: the all-electron HBr control
//! agrees with PySCF to 1e-11 (energy) / 3.4e-10 (gradient), and the
//! one-centre I atom (ECP and basis on the same nucleus) to 2.4e-12. Every
//! ECP MOLECULE carries a 0.8e-8..2.9e-8 offset in energy, orbital energies
//! and gradient against BOTH PySCF and ORCA, which agree with each other to
//! ~1e-10. So the offset is ferric's multi-centre ECP integrals (libecpint's
//! angular quadrature for basis functions off the ECP centre), not the SCF
//! or the reference codes. Energies keep 9x headroom (CI hardware), the
//! first-order quantities 4-5x.
//!
//! The ORCA fallback bar is looser because ORCA's library copy of the def2-ECP
//! digits is not ferric's JSON; it is used only if a reference file says
//! `"ecp_source": "orca-builtin def2-ECP"` (see gen_ecp_orca.py).
//!
//! # NEGATIVE CONTROLS / MUTATIONS (the test must be able to fail)
//!
//! * HBr: def2 Br is all-electron. The test asserts the bundled basis has NO
//!   ECP for Br, that no atom gets `n_core_ecp > 0`, that the electron count
//!   is the all-electron 36, and that the ECP gradient term is identically
//!   zero — i.e. the ECP path is demonstrably NOT taken, while on HI/CH3I/RbH/
//!   SnH4 it demonstrably IS (`n_core_ecp > 0` on the heavy atom, and a
//!   non-zero ECP gradient term).
//! * Always-on resolving power (gradient): on every ECP system the ECP
//!   gradient term alone, `Σ D dV_ECP/dR`, must exceed 1000× the gradient bar,
//!   so dropping it (MUTATION A: delete the `ecp_gradient` line in
//!   `rhf_gradient`) cannot pass. Do run this mutation once and record it.
//! * MUTATION B — point [`reference`] at the other basis. The cross-basis
//!   assertion fails, and so does the energy assertion.
//! * MUTATION C — in `common.pyscf_ecp` (the reference side) tag the local
//!   channel as its own l instead of −1 and regenerate one file: the energy
//!   assertion fails by ≫ the bar. (Reference-side mutation; do not commit.)
//!
//! A missing reference JSON (PySCF or ORCA) is a HARD failure naming the path.

use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{ecp_gradient, rhf_gradient};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::{ScfResult, Spin};
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/ecp";
const MOL_DIR: &str = "testdata/molecules/validation/ecp";
const BASES: [&str; 2] = ["def2-svp", "def2-tzvp"];

const TOL_E: f64 = 2.5e-7;
const TOL_EPS: f64 = 1e-7;
const TOL_G: f64 = 1e-7;
const TOL_E_ORCA: f64 = 2.5e-7;
const TOL_E_ORCA_BUILTIN: f64 = 1e-5;
const TOL_S2: f64 = 1e-9;
const TOL_ENUC: f64 = 1e-9;
const TOL_ENUC_ORCA: f64 = 1e-7;
/// A reference ferric's energy must MISS: 1000× the energy bar.
const MUST_MISS: f64 = 1000.0 * TOL_E;
/// The ECP gradient term alone must exceed this on every ECP system.
const ECP_GRAD_MUST_EXCEED: f64 = 1000.0 * TOL_G;
/// A reference gradient smaller than this cannot tell right from wrong.
const MIN_REF_GRAD: f64 = 1e-3;

/// Workspace root, found by walking up from the CWD (see
/// validation_open_shell_scf.rs for why CARGO_MANIFEST_DIR is only a fallback).
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

fn read_json(path: &Path, generator: &str) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with {generator} — a missing \
             reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

/// PySCF reference. Missing or unparsable is a HARD failure.
fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    read_json(&path, "scripts/validation/gen_ecp.py")
}

/// ORCA reference: the like-for-like (explicit NewECP) file, else the
/// built-in-ECP fallback. Returns the JSON and the energy bar that applies.
fn orca_reference(system: &str, basis_name: &str) -> (Value, f64) {
    let dir = workspace_root().join(ROW_DIR);
    let explicit = dir.join(format!("orca_{system}_{basis_name}.json"));
    let builtin = dir.join(format!("orca_{system}_builtin_{basis_name}.json"));
    let (v, tol) = if explicit.is_file() || !builtin.is_file() {
        (
            read_json(&explicit, "scripts/validation/gen_ecp_orca.py"),
            TOL_E_ORCA,
        )
    } else {
        (
            read_json(&builtin, "scripts/validation/gen_ecp_orca.py --builtin-ecp"),
            TOL_E_ORCA_BUILTIN,
        )
    };
    // The bar must follow what the file SAYS it is, not which name it has.
    let source = v["ecp_source"].as_str().unwrap_or("<missing>");
    let tol_by_source = match source {
        "ferric-json NewECP" => TOL_E_ORCA,
        "orca-builtin def2-ECP" => TOL_E_ORCA_BUILTIN,
        other => panic!("{system}/{basis_name}: unknown ORCA ecp_source {other:?}"),
    };
    assert_eq!(
        tol, tol_by_source,
        "{system}/{basis_name}: ORCA reference file name and ecp_source disagree"
    );
    (v, tol)
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

struct System {
    mol: Molecule,
    bs: BasisSet,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

/// Build the molecule on the ECP path and check everything that must match
/// BEFORE any SCF: N_core per atom, electron count, V_nn, AO count.
fn load_system(system: &str, basis_name: &str, r: &Value) -> System {
    let ctx = format!("{system}/{basis_name}");
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mut mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let bs = basis::bundled(basis_name).unwrap();
    // CLAUDE.md: apply_ecp BEFORE nelec() / nuclear_repulsion().
    mol.apply_ecp(&bs);

    let core_ref: Vec<i64> = r["ecp_core_electrons"]
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: ecp_core_electrons missing"))
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let core: Vec<i64> = mol.atoms.iter().map(|a| a.n_core_ecp as i64).collect();
    assert_eq!(
        core, core_ref,
        "{ctx}: per-atom ECP core electrons ferric {core:?} vs reference {core_ref:?}"
    );
    assert_eq!(
        mol.nelec() as i64,
        r["nelectron"].as_i64().unwrap(),
        "{ctx}: electron count after apply_ecp"
    );

    let enuc_ref = num(r, "/nuclear_repulsion", &ctx);
    check_close(&ctx, "E_nuc", mol.nuclear_repulsion(), enuc_ref, TOL_ENUC);

    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let s = ferric_integrals::oneelectron::overlap(&prep);
    assert_eq!(
        s.nrows(),
        r["nao"].as_u64().unwrap() as usize,
        "{ctx}: AO count differs from the reference's"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    System {
        mol,
        bs,
        prep,
        bounds,
        ctx: ParallelContext::default(),
    }
}

fn rhf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        // The gradient is FIRST order in the density error.
        density_conv: 1e-10,
        ..Default::default()
    }
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

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Whether this system is expected to take the ECP path.
#[derive(Clone, Copy, PartialEq)]
enum Ecp {
    Active,
    /// Negative control: the basis has no ECP for any atom here.
    AllElectron,
}

/// Run ferric RHF + analytic gradient on one system × basis and check it
/// against PySCF (energy, ε_HOMO, gradient) and ORCA (energy). Returns
/// ferric's energy for the cross-basis check.
fn check_rhf(system: &str, basis_name: &str, ecp: Ecp) -> f64 {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}/RHF");
    let sys = load_system(system, basis_name, &r);
    let op = Operator::coulomb();

    match ecp {
        Ecp::Active => assert!(
            sys.mol.atoms.iter().any(|a| a.n_core_ecp > 0),
            "{ctx}: no atom carries an ECP — this case would be an all-electron test"
        ),
        Ecp::AllElectron => {
            for a in &sys.mol.atoms {
                assert!(
                    sys.bs.ecp_for_element(a.z).is_none(),
                    "{ctx}: bundled {basis_name} unexpectedly carries an ECP for Z={}",
                    a.z
                );
                assert_eq!(a.n_core_ecp, 0, "{ctx}: Z={} took the ECP path", a.z);
            }
            let n_all: i32 = sys.mol.atoms.iter().map(|a| a.z).sum::<i32>() - sys.mol.charge;
            assert_eq!(sys.mol.nelec(), n_all, "{ctx}: all-electron count expected");
        }
    }

    let res = solve_rhf(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        op,
        &sys.bounds,
        &rhf_config(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
    assert!(res.converged, "{ctx}: not converged");
    assert!(
        matches!(res.spin, Spin::Restricted),
        "{ctx}: not an RHF result"
    );

    check_close(
        &ctx,
        "energy",
        res.energy,
        num(&r, "/rhf/energy", &ctx),
        TOL_E,
    );
    let nocc = (sys.mol.nelec() / 2) as usize;
    check_close(
        &ctx,
        "HOMO",
        res.eps_alpha[nocc - 1],
        num(&r, "/rhf/homo", &ctx),
        TOL_EPS,
    );

    // --- analytic gradient vs PySCF's analytic gradient ---
    let grad = rhf_gradient(&sys.mol, &sys.prep, op, &sys.bounds, &res, None)
        .unwrap_or_else(|e| panic!("{ctx}: rhf_gradient failed: {e:?}"));
    let gref = r["rhf"]["gradient"]
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: /rhf/gradient missing"));
    assert_eq!(gref.len(), sys.mol.atoms.len(), "{ctx}: gradient rows");
    let mut worst = 0.0f64;
    let mut ref_max = 0.0f64;
    for (a, row) in gref.iter().enumerate() {
        for c in 0..3 {
            let want = row[c].as_f64().expect("gradient entry");
            let got = grad[(a, c)];
            eprintln!(
                "{ctx}: grad atom {a} coord {c}: ferric {got:+.10} ref {want:+.10} d {:+.2e}",
                got - want
            );
            worst = worst.max((got - want).abs());
            ref_max = ref_max.max(want.abs());
        }
    }
    eprintln!("{ctx}: max |grad ferric - grad PySCF| = {worst:.3e} Ha/Bohr (tol {TOL_G:.0e})");
    assert!(
        ref_max > MIN_REF_GRAD,
        "{ctx}: reference gradient max {ref_max:.2e} — a near-zero gradient makes the \
         comparison vacuous"
    );
    assert!(
        worst < TOL_G,
        "{ctx}: analytic gradient misses PySCF by {worst:.3e} Ha/Bohr (tol {TOL_G:.0e})"
    );

    // --- resolving power of the gradient bar against a missing ECP term ---
    let g_ecp = ecp_gradient(&sys.mol, &sys.prep, res.density_r())
        .unwrap_or_else(|e| panic!("{ctx}: ecp_gradient failed: {e:?}"));
    let g_ecp_max = max_abs(&g_ecp);
    eprintln!("{ctx}: max |Σ D dV_ECP/dR| = {g_ecp_max:.3e} Ha/Bohr");
    match ecp {
        Ecp::Active => assert!(
            g_ecp_max > ECP_GRAD_MUST_EXCEED,
            "{ctx}: the ECP gradient term ({g_ecp_max:.2e}) is not resolvable by the \
             {TOL_G:.0e} bar, so this test could not detect it missing"
        ),
        Ecp::AllElectron => assert!(
            g_ecp_max == 0.0,
            "{ctx}: all-electron control produced a non-zero ECP gradient term {g_ecp_max:.2e}"
        ),
    }

    // --- third code: ORCA energy ---
    let (o, tol_orca) = orca_reference(system, basis_name);
    let octx = format!("{ctx}/ORCA");
    check_close(
        &octx,
        "E_nuc",
        sys.mol.nuclear_repulsion(),
        num(&o, "/nuclear_repulsion", &octx),
        TOL_ENUC_ORCA,
    );
    check_close(
        &octx,
        "energy",
        res.energy,
        num(&o, "/energy", &octx),
        tol_orca,
    );

    res.energy
}

/// ⟨S²⟩ of a UHF determinant from ferric's own MOs and overlap.
fn s_squared(res: &ScfResult, s: &Array2<f64>, na: usize, nb: usize) -> f64 {
    let sz = 0.5 * (na as f64 - nb as f64);
    let ca = res.mos_alpha.slice(ndarray::s![.., ..na]);
    let cb = res.mos_beta.as_ref().unwrap().slice(ndarray::s![.., ..nb]);
    let ov = ca.t().dot(s).dot(&cb);
    sz * (sz + 1.0) + nb as f64 - ov.iter().map(|v| v * v).sum::<f64>()
}

/// I atom ²P, UHF. Spatially degenerate: rotating the p hole is a zero mode
/// of the orbital Hessian, so ferric's verdict may honestly be MARGINAL. The
/// test rejects only a proven saddle (UNSTABLE) or an unconverged eigensolve,
/// and does not compare λ_min (it is a zero-mode noise value on both sides).
fn check_uhf_atom(system: &str, basis_name: &str) -> f64 {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}/UHF");
    let sys = load_system(system, basis_name, &r);
    assert!(
        sys.mol.atoms[0].n_core_ecp > 0,
        "{ctx}: I must carry an ECP"
    );
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
    assert!(
        matches!(
            st.verdict(),
            StabilityVerdict::Stable | StabilityVerdict::Marginal
        ),
        "{ctx}: ferric's final UHF state is not a minimum: {}",
        st.summary()
    );

    check_close(
        &ctx,
        "energy",
        res.energy,
        num(&r, "/uhf/energy", &ctx),
        TOL_E,
    );
    let nelec = sys.mol.nelec() as usize;
    let two_s = sys.mol.multiplicity - 1;
    let (na, nb) = ((nelec + two_s) / 2, (nelec - two_s) / 2);
    assert_eq!(
        (na as i64, nb as i64),
        (
            r["uhf"]["nelec_alpha"].as_i64().unwrap(),
            r["uhf"]["nelec_beta"].as_i64().unwrap()
        ),
        "{ctx}: (n_alpha, n_beta)"
    );
    let s = ferric_integrals::oneelectron::overlap(&sys.prep);
    check_close(
        &ctx,
        "<S^2>",
        s_squared(&res, &s, na, nb),
        num(&r, "/uhf/s_squared", &ctx),
        TOL_S2,
    );
    res.energy
}

/// Two-input requirement: ferric's energy in basis A must miss basis B's
/// reference by far more than the bar, so a basis ignored somewhere in the
/// chain, or a swapped/mislabelled reference file, cannot pass both bases.
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

fn rhf_row(system: &str, ecp: Ecp) {
    let e = [
        check_rhf(system, BASES[0], ecp),
        check_rhf(system, BASES[1], ecp),
    ];
    assert_basis_discriminates(system, "rhf", &e);
}

#[test]
#[ignore = "validation: RHF+ECP / ECP gradients"]
fn ecp_rhf_hi_vs_pyscf_and_orca() {
    rhf_row("hi", Ecp::Active);
}

#[test]
#[ignore = "validation: RHF+ECP / ECP gradients"]
fn ecp_rhf_ch3i_vs_pyscf_and_orca() {
    rhf_row("ch3i", Ecp::Active);
}

#[test]
#[ignore = "validation: RHF+ECP / ECP gradients"]
fn ecp_rhf_rbh_vs_pyscf_and_orca() {
    rhf_row("rbh", Ecp::Active);
}

/// Negative control: HBr must NOT take the ECP path (def2 Br is all-electron),
/// yet still match both references.
#[test]
#[ignore = "validation: RHF+ECP / ECP gradients"]
fn ecp_rhf_hbr_all_electron_control() {
    rhf_row("hbr", Ecp::AllElectron);
}

/// Single basis (ferric's def2-SVP has no Sn): a third ECP shape, not one of
/// the two-basis inputs the grade rests on.
#[test]
#[ignore = "validation: RHF+ECP / ECP gradients"]
fn ecp_rhf_snh4_tzvp_vs_pyscf_and_orca() {
    check_rhf("snh4", "def2-tzvp", Ecp::Active);
}

#[test]
#[ignore = "validation: RHF+ECP / ECP gradients"]
fn ecp_uhf_i_atom_vs_pyscf() {
    let e = [
        check_uhf_atom("i_atom", BASES[0]),
        check_uhf_atom("i_atom", BASES[1]),
    ];
    assert_basis_discriminates("i_atom", "uhf", &e);
}
