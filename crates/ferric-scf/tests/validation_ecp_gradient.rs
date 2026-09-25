//! VALIDATION tier — VALIDATION.md row "ECP gradients".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_ecp_gradient \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's analytic RHF and UHF nuclear gradients on def2 ECP systems, where
//! the ECP enters as `Σ_μν D_μν dV_ECP_μν/dR` ([`ecp_gradient`], libecpint
//! first derivatives, added in `rhf_gradient` / `uhf_gradient`):
//!
//! | system | ECP | bases | SCF |
//! |---|---|---|---|
//! | HI (stretched) | I, def2 28-core | def2-SVP, def2-TZVP | RHF |
//! | CH3I (distorted, x/y/z all move) | I | def2-SVP, def2-TZVP | RHF |
//! | SnH4 (one bond stretched) | Sn | def2-TZVP only | RHF |
//! | CH2I• (distorted doublet radical) | I | def2-SVP, def2-TZVP | UHF |
//! | HBr (stretched) | NONE — Br is all-electron in def2 | def2-SVP, def2-TZVP | RHF, control |
//!
//! SnH4 is single-basis because ferric's bundled def2-SVP has no Sn. The UHF
//! case is CH2I•, not HI⁺: HI⁺'s ²Π hole is spatially degenerate (a zero mode
//! of the orbital Hessian), while CH2I•'s distorted SOMO is not.
//!
//! Per case, against references from `scripts/validation/gen_ecp_gradient.py`
//! (`testdata/reference/validation/ecp_gradient/`, same like-for-like basis/ECP
//! injection as the energy row: ferric's own def2 JSON, ECP from its inline
//! block, geometry in Bohr):
//!
//! 1. V_nn, N_core per atom, electron and AO counts (before any SCF);
//! 2. SCF energy vs PySCF (and ⟨S²⟩ for UHF — the same state);
//! 3. analytic gradient vs PySCF's analytic `nuc_grad_method()`;
//! 4. ferric's ECP term ALONE vs PySCF's ECP term alone (the generator builds
//!    it from `ECPscalar_ipnuc`/`ECPscalar_iprinv` and checks it against a
//!    fixed-density FD of `tr[D V_ECP(R)]` before writing), and ferric's
//!    gradient MINUS its ECP term vs PySCF's gradient minus its ECP term — the
//!    two halves agree separately, not just in sum;
//! 5. translation invariance: `Σ_A g_A ≈ 0` for the full gradient AND for the
//!    ECP term alone;
//! 6. ferric's analytic gradient vs a 5-point central FD of FERRIC'S OWN SCF
//!    energy (independent of PySCF), on the ECP atom's three coordinates plus
//!    the largest-gradient coordinate of another atom;
//! 7. analytic gradient vs ORCA 6.1.1 `EnGrad` (third ECP-derivative code).
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * Right: every comparison lands at the ECP-integral floor the energy row
//!   measured (~1e-8 in energy and gradient vs PySCF/ORCA; libecpint's
//!   off-centre angular quadrature), and FD at its step/SCF-noise floor.
//! * dV_ECP/dR missing for a spin treatment (the defect this row found: until
//!   this row, only `rhf_gradient` added [`ecp_gradient`]; UHF/ROHF/KS dropped
//!   it while their SCF energies included V_ECP): the full gradient misses by
//!   the ECP term, ≥1e-3 Ha/Bohr here, AND ferric-minus-ECP-term then matches
//!   PySCF's `gradient_without_ecp` — the check below that asserts the term
//!   is resolvable turns "missing" into a failure, not a pass.
//! * ECP term attributed to the wrong atom / wrong sign: term-alone comparison
//!   (4) fails, FD (6) fails, and a sign flip also breaks nothing in (5) — so
//!   (5) is NOT relied on for that; see MUTATIONS.
//! * Energy right but the gradient differentiates a different energy (e.g.
//!   bare Z in V_nn'): FD (6) fails even if a reference shared the error.
//!
//! # TOLERANCES
//!
//! Measured maxima over all cases (2026-09-25) are noted beside each bar. The
//! gradient floor (1.3e-7 Ha/Bohr at def2-TZVP) is the libecpint-vs-PySCF ECP
//! quadrature difference: it lives entirely in the ECP term (the ECP-term check
//! carries it) and matches
//! the energy row's 1e-8 Ha ECP offsets. Removing or negating the ECP term
//! misses by 1e-3..4e-2 Ha/Bohr, four orders above the bars.
//!
//! # NEGATIVE CONTROLS / MUTATIONS
//!
//! * HBr: no ECP anywhere; ECP term identically zero; still matches PySCF,
//!   FD and ORCA — the non-ECP path agrees on its own.
//! * On every ECP case `max|ferric − ecp_term − PySCF_full| > MUST_MISS`:
//!   a gradient without the ECP derivative cannot pass.
//! * MUTATION A (the defect itself): delete the line
//!   `grad += &ecp_gradient(mol, prep, &d_total)?;` in `uhf_gradient`
//!   (crates/ferric-scf/src/gradient.rs, after the `fitted_hf_gradient` match).
//!   Run 2026-09-25: the CH2I UHF test fails; the RHF cases do not use that line.
//! * MUTATION B (a sign in the contraction): in `ecp_gradient`, change
//!   `grad[(a, c)] = d.iter().zip(dv.iter()).map(|(x, y)| x * y).sum::<f64>();`
//!   to `= -d.iter()...`. Run 2026-09-25: all four ECP cases fail; the
//!   all-electron HBr control passes. Check 5 cannot see it (the ECP term
//!   still sums to zero), which is why (5) is an anchor, not a guard.
//!
//! A missing reference JSON is a HARD failure naming the path.

use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{ecp_gradient, rhf_gradient, uhf_gradient};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::solve_uhf_with_guess;
use ferric_scf::{ScfResult, Spin};
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/ecp_gradient";
const MOL_DIR: &str = "testdata/molecules/validation/ecp";
const BASES: [&str; 2] = ["def2-svp", "def2-tzvp"];

// Energy vs PySCF: 2.9e-8 Ha.
const TOL_E: f64 = 2.5e-7;
// Gradient vs PySCF: 1.3e-7 Ha/Bohr (def2-TZVP; libecpint vs PySCF ECP quadrature).
const TOL_G: f64 = 5e-7;
// ECP term alone, ferric(D_ferric) vs PySCF(D_PySCF): 1.3e-7 Ha/Bohr.
const TOL_G_ECP_TERM: f64 = 5e-7;
// |Σ_A g_A|, full gradient and ECP term alone: ≤ 4e-10.
const TOL_SUM: f64 = 1e-9;
// 5-point FD (measured 1.9e-7), h = 2e-3 Bohr (h⁴ truncation ~1e-11,
// SCF noise ×1.5/h ~1e-9)
const TOL_FD: f64 = 6e-7;
// vs ORCA EnGrad: 2.5e-7 Ha/Bohr (ORCA's own ECP derivative integrals).
const TOL_G_ORCA: f64 = 1e-6;
const TOL_S2: f64 = 1e-6;
const TOL_ENUC: f64 = 1e-9;
const FD_STEP: f64 = 2e-3;
/// A gradient without the ECP term must miss PySCF by at least this.
const MUST_MISS: f64 = 1000.0 * TOL_G;
/// A reference gradient smaller than this cannot tell right from wrong.
const MIN_REF_GRAD: f64 = 1e-3;

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

fn read_json(name: &str) -> Value {
    let path = workspace_root().join(ROW_DIR).join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_ecp_gradient.py — a missing reference is a failure, \
             never a skip",
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

fn matrix(v: &Value, ptr: &str, natoms: usize, ctx: &str) -> Array2<f64> {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing"));
    assert_eq!(rows.len(), natoms, "{ctx}: {ptr} has {} rows", rows.len());
    let mut m = Array2::zeros((natoms, 3));
    for (a, row) in rows.iter().enumerate() {
        for c in 0..3 {
            m[(a, c)] = row[c]
                .as_f64()
                .unwrap_or_else(|| panic!("{ctx}: {ptr}[{a}][{c}] not a number"));
        }
    }
    m
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn max_col_sum(a: &Array2<f64>) -> f64 {
    (0..3)
        .map(|c| a.column(c).sum().abs())
        .fold(0.0f64, f64::max)
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<24} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_grad(ctx: &str, what: &str, got: &Array2<f64>, want: &Array2<f64>, tol: f64) -> f64 {
    let d = max_diff(got, want);
    for a in 0..got.nrows() {
        eprintln!(
            "{ctx}: {what} atom {a}: ferric [{:+.10} {:+.10} {:+.10}] ref [{:+.10} {:+.10} {:+.10}]",
            got[(a, 0)],
            got[(a, 1)],
            got[(a, 2)],
            want[(a, 0)],
            want[(a, 1)],
            want[(a, 2)]
        );
    }
    eprintln!("{ctx}: {what}: max |d| = {d:.3e} Ha/Bohr (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} misses by {d:.3e} Ha/Bohr (tol {tol:.0e})"
    );
    d
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Rhf,
    Uhf,
}

#[derive(Clone, Copy, PartialEq)]
enum Ecp {
    Active,
    /// Negative control: the basis has no ECP for any atom here.
    AllElectron,
}

fn rhf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-10,
        ..Default::default()
    }
}

fn uhf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-10,
        check_stability: true,
        scf_stability_descent: true,
        ..Default::default()
    }
}

/// SCF at `mol`'s geometry. For UHF, `seed` (the undisplaced MOs) keeps a
/// displaced run on the same state; stability is checked either way.
fn scf(
    kind: Kind,
    mol: &Molecule,
    bs: &BasisSet,
    seed: Option<&ScfResult>,
    ctx: &str,
) -> (ScfResult, PreparedBasis, SchwarzBounds) {
    let pctx = ParallelContext::default();
    let prep = PreparedBasis::new(mol, bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let res = match kind {
        Kind::Rhf => solve_rhf(&pctx, mol, &prep, op, &bounds, &rhf_config())
            .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}")),
        Kind::Uhf => {
            let guess = seed.map(|r| (&r.mos_alpha, r.mos_beta.as_ref().unwrap()));
            let r = solve_uhf_with_guess(&pctx, mol, &prep, &bounds, &uhf_config(), guess)
                .unwrap_or_else(|e| panic!("{ctx}: solve_uhf failed: {e:?}"));
            let st = r
                .stability
                .as_ref()
                .unwrap_or_else(|| panic!("{ctx}: no stability verdict"));
            assert!(
                matches!(
                    st.verdict(),
                    StabilityVerdict::Stable | StabilityVerdict::Marginal
                ),
                "{ctx}: UHF state is not a minimum: {}",
                st.summary()
            );
            r
        }
    };
    assert!(res.converged, "{ctx}: SCF not converged");
    (res, prep, bounds)
}

fn gradient(
    kind: Kind,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    res: &ScfResult,
) -> Array2<f64> {
    let op = Operator::coulomb();
    match kind {
        Kind::Rhf => rhf_gradient(mol, prep, op, bounds, res, None),
        Kind::Uhf => uhf_gradient(mol, prep, op, bounds, res, None),
    }
    .expect("analytic gradient")
}

fn total_density(res: &ScfResult) -> Array2<f64> {
    match res.spin {
        Spin::Restricted => res.density_r().clone(),
        _ => &res.density_alpha + res.density_beta.as_ref().unwrap(),
    }
}

fn s_squared(res: &ScfResult, s: &Array2<f64>, na: usize, nb: usize) -> f64 {
    let sz = 0.5 * (na as f64 - nb as f64);
    let ca = res.mos_alpha.slice(ndarray::s![.., ..na]);
    let cb = res.mos_beta.as_ref().unwrap().slice(ndarray::s![.., ..nb]);
    let ov = ca.t().dot(s).dot(&cb);
    sz * (sz + 1.0) + nb as f64 - ov.iter().map(|v| v * v).sum::<f64>()
}

fn displaced(mol: &Molecule, a: usize, c: usize, dx: f64) -> Molecule {
    let mut m = mol.clone();
    match c {
        0 => m.atoms[a].x += dx,
        1 => m.atoms[a].y += dx,
        _ => m.atoms[a].zpos += dx,
    }
    m
}

/// All checks for one system × basis. Returns ferric's energy.
fn check_case(system: &str, basis_name: &str, kind: Kind, ecp: Ecp) -> f64 {
    let r = read_json(&format!("{system}_{basis_name}.json"));
    let key = match kind {
        Kind::Rhf => "rhf",
        Kind::Uhf => "uhf",
    };
    let ctx = format!("{system}/{basis_name}/{key}");
    assert_eq!(r["method"].as_str(), Some(key), "{ctx}: reference method");

    // --- 1. system build, before any SCF ---
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mut mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let bs = basis::bundled(basis_name).unwrap();
    mol.apply_ecp(&bs);
    let natoms = mol.atoms.len();
    let core_ref: Vec<i64> = r["ecp_core_electrons"]
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: ecp_core_electrons missing"))
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let core: Vec<i64> = mol.atoms.iter().map(|a| a.n_core_ecp as i64).collect();
    assert_eq!(core, core_ref, "{ctx}: per-atom ECP core electrons");
    assert_eq!(
        mol.nelec() as i64,
        r["nelectron"].as_i64().unwrap(),
        "{ctx}: electron count after apply_ecp"
    );
    check_close(
        &ctx,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    let ecp_atom = match ecp {
        Ecp::Active => {
            assert_eq!(
                r["has_ecp"].as_bool(),
                Some(true),
                "{ctx}: reference has_ecp"
            );
            mol.atoms
                .iter()
                .position(|a| a.n_core_ecp > 0)
                .unwrap_or_else(|| panic!("{ctx}: no atom carries an ECP"))
        }
        Ecp::AllElectron => {
            assert_eq!(
                r["has_ecp"].as_bool(),
                Some(false),
                "{ctx}: reference has_ecp"
            );
            for a in &mol.atoms {
                assert!(
                    bs.ecp_for_element(a.z).is_none(),
                    "{ctx}: bundled {basis_name} unexpectedly carries an ECP for Z={}",
                    a.z
                );
                assert_eq!(a.n_core_ecp, 0, "{ctx}: Z={} took the ECP path", a.z);
            }
            // heaviest atom stands in for "the ECP atom" in the FD selection
            (0..natoms).max_by_key(|&i| mol.atoms[i].z).unwrap()
        }
    };

    // --- 2. SCF energy (and state) ---
    let (res, prep, bounds) = scf(kind, &mol, &bs, None, &ctx);
    assert_eq!(
        ferric_integrals::oneelectron::overlap(&prep).nrows(),
        r["nao"].as_u64().unwrap() as usize,
        "{ctx}: AO count"
    );
    check_close(
        &ctx,
        "energy",
        res.energy,
        num(&r, &format!("/{key}/energy"), &ctx),
        TOL_E,
    );
    if kind == Kind::Uhf {
        let nelec = mol.nelec() as usize;
        let (na, nb) = ((nelec + mult - 1) / 2, (nelec - (mult - 1)) / 2);
        let s = ferric_integrals::oneelectron::overlap(&prep);
        check_close(
            &ctx,
            "<S^2>",
            s_squared(&res, &s, na, nb),
            num(&r, "/uhf/s_squared", &ctx),
            TOL_S2,
        );
    }

    // --- 3. analytic gradient vs PySCF analytic ---
    let g = gradient(kind, &mol, &prep, &bounds, &res);
    let g_ref = matrix(&r, &format!("/{key}/gradient"), natoms, &ctx);
    assert!(
        max_abs(&g_ref) > MIN_REF_GRAD,
        "{ctx}: reference gradient max {:.2e} is vacuous",
        max_abs(&g_ref)
    );
    check_grad(&ctx, "gradient vs PySCF", &g, &g_ref, TOL_G);

    // --- 4. the ECP term alone, and the rest alone ---
    let g_ecp = ecp_gradient(&mol, &prep, &total_density(&res)).expect("ecp_gradient");
    let g_ecp_ref = matrix(&r, &format!("/{key}/gradient_ecp_term"), natoms, &ctx);
    let g_rest_ref = matrix(&r, &format!("/{key}/gradient_without_ecp"), natoms, &ctx);
    let g_rest = &g - &g_ecp;
    match ecp {
        Ecp::Active => {
            check_grad(
                &ctx,
                "ECP term vs PySCF",
                &g_ecp,
                &g_ecp_ref,
                TOL_G_ECP_TERM,
            );
            check_grad(&ctx, "gradient - ECP term", &g_rest, &g_rest_ref, TOL_G);
            // Negative control: a gradient that DROPS the ECP derivative.
            let miss = max_diff(&g_rest, &g_ref);
            eprintln!("{ctx}: gradient without ECP term misses PySCF by {miss:.3e}");
            assert!(
                miss > MUST_MISS,
                "{ctx}: dropping the ECP term moves the gradient by only {miss:.2e} \
                 (< {MUST_MISS:.0e}); this case could not detect it missing"
            );
        }
        Ecp::AllElectron => {
            assert_eq!(max_abs(&g_ecp), 0.0, "{ctx}: ECP term non-zero on control");
            assert_eq!(
                max_abs(&g_ecp_ref),
                0.0,
                "{ctx}: reference ECP term non-zero"
            );
        }
    }

    // --- 5. translation invariance ---
    let sum_full = max_col_sum(&g);
    let sum_ecp = max_col_sum(&g_ecp);
    eprintln!(
        "{ctx}: |Σ_A g_A| = {sum_full:.2e}, |Σ_A g_ECP,A| = {sum_ecp:.2e} (tol {TOL_SUM:.0e})"
    );
    assert!(sum_full < TOL_SUM, "{ctx}: Σ_A g_A = {sum_full:.2e}");
    assert!(sum_ecp < TOL_SUM, "{ctx}: Σ_A g_ECP,A = {sum_ecp:.2e}");

    // --- 6. 5-point FD of ferric's own energy ---
    let mut coords: Vec<(usize, usize)> = (0..3).map(|c| (ecp_atom, c)).collect();
    let other = (0..natoms)
        .filter(|&a| a != ecp_atom)
        .flat_map(|a| (0..3).map(move |c| (a, c)))
        .max_by(|p, q| g[*p].abs().total_cmp(&g[*q].abs()))
        .expect("a second atom");
    coords.push(other);
    let mut fd_worst = 0.0f64;
    for (a, c) in coords {
        let e = |k: f64| {
            let m = displaced(&mol, a, c, k * FD_STEP);
            scf(kind, &m, &bs, Some(&res), &format!("{ctx}/FD"))
                .0
                .energy
        };
        let fd = (e(-2.0) - 8.0 * e(-1.0) + 8.0 * e(1.0) - e(2.0)) / (12.0 * FD_STEP);
        let d = (g[(a, c)] - fd).abs();
        eprintln!(
            "{ctx}: FD atom {a} coord {c}: analytic {:+.10} FD {fd:+.10} |d| {d:.2e}",
            g[(a, c)]
        );
        fd_worst = fd_worst.max(d);
    }
    eprintln!("{ctx}: max |analytic - FD| = {fd_worst:.3e} (tol {TOL_FD:.0e})");
    assert!(
        fd_worst < TOL_FD,
        "{ctx}: analytic gradient misses its own 5-point FD by {fd_worst:.3e}"
    );

    // --- 7. ORCA EnGrad ---
    let o = read_json(&format!("orca_{system}_{basis_name}.json"));
    assert_eq!(
        o["ecp_source"].as_str(),
        Some("ferric-json NewECP"),
        "{ctx}: ORCA file must use ferric's ECP"
    );
    let g_orca = matrix(&o, "/gradient", natoms, &ctx);
    check_grad(&ctx, "gradient vs ORCA", &g, &g_orca, TOL_G_ORCA);

    res.energy
}

/// Two-input requirement: a mislabelled or swapped reference file cannot pass
/// both bases (def2-SVP and def2-TZVP energies differ by ≫ the bar).
fn assert_basis_discriminates(system: &str, key: &str, energies: &[f64; 2]) {
    for (i, b_self) in BASES.iter().enumerate() {
        let b_other = BASES[1 - i];
        let other = num(
            &read_json(&format!("{system}_{b_other}.json")),
            &format!("/{key}/energy"),
            system,
        );
        assert!(
            (energies[i] - other).abs() > 1000.0 * TOL_E,
            "{system}: ferric {b_self} energy {:.10} is within {:.0e} of the {b_other} \
             reference {other:.10}",
            energies[i],
            1000.0 * TOL_E
        );
    }
}

fn two_basis(system: &str, kind: Kind, ecp: Ecp) {
    let e = [
        check_case(system, BASES[0], kind, ecp),
        check_case(system, BASES[1], kind, ecp),
    ];
    let key = if kind == Kind::Rhf { "rhf" } else { "uhf" };
    assert_basis_discriminates(system, key, &e);
}

#[test]
#[ignore = "validation: ECP gradient"]
fn ecp_gradient_hi_rhf() {
    two_basis("hi", Kind::Rhf, Ecp::Active);
}

#[test]
#[ignore = "validation: ECP gradient"]
fn ecp_gradient_ch3i_rhf() {
    two_basis("ch3i", Kind::Rhf, Ecp::Active);
}

/// Single basis: ferric's def2-SVP has no Sn.
#[test]
#[ignore = "validation: ECP gradient"]
fn ecp_gradient_snh4_rhf_tzvp() {
    check_case("snh4", "def2-tzvp", Kind::Rhf, Ecp::Active);
}

/// Open shell: before this row `uhf_gradient` omitted the ECP term entirely.
#[test]
#[ignore = "validation: ECP gradient"]
fn ecp_gradient_ch2i_uhf() {
    two_basis("ch2i", Kind::Uhf, Ecp::Active);
}

/// Negative control: all-electron HBr takes no ECP path and still agrees.
#[test]
#[ignore = "validation: ECP gradient"]
fn ecp_gradient_hbr_all_electron_control() {
    two_basis("hbr", Kind::Rhf, Ecp::AllElectron);
}
