//! VALIDATION tier — VALIDATION.md row "Static α".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-rpa --test validation_static_alpha \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What ferric computes (read from the loop, not the name)
//!
//! `ferric_rpa::properties::pdep_polarizability_static` returns the
//! closed-shell DIRECT-RPA static polarizability — time-dependent Hartree, NO
//! exchange kernel — with a density-fitted Coulomb kernel and no frozen core:
//!
//! ```text
//!   α_xy = 4 μ_xᵀ (Δ + 4 K)⁻¹ μ_y,   K_ia,jb = (ia|P) V⁻¹ (P|jb),   μ_ia = ⟨i|r|a⟩
//! ```
//!
//! solved in the aux space by Sherman–Morrison–Woodbury on ε̃ = I + 4 B̃ Δ⁻¹ B̃ᵀ.
//! It is NOT coupled-perturbed HF (that has the −(ib|ja) − (ij|ab) exchange
//! kernel) and NOT an exact-integral dRPA. The reference
//! (`scripts/validation/gen_properties.py alpha`, files in
//! `testdata/reference/validation/static_alpha/`) solves the SAME definition
//! by a different route — a dense ov-space linear solve on PySCF's
//! `int3c2e`/`int2c2e` with the SAME aux basis read from ferric's JSON — and
//! also stores the CPHF and exact-ERI dRPA tensors for scope.
//!
//! Systems: H2O / aug-cc-pVDZ (aux aug-cc-pvdz-rifit) and CH3OH / cc-pVDZ (aux
//! cc-pvdz-ri) — two molecules, two orbital bases, two aux bases.
//!
//! Two comparisons per case:
//!
//! 1. SAME ORBITALS. PySCF's converged MOs (permuted to ferric's AO order) and
//!    orbital energies are injected into a ferric `ScfResult`; α is then a pure
//!    function of (C, ε, integrals). Expected at the integral floor.
//! 2. FULL CHAIN. ferric's own RHF → α.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric's α is the stated dRPA-RI α: same-orbital agreement ~1e-10
//!   relative (the SMW and ov routes are algebraically identical).
//! * If ferric silently used a different aux basis, a frozen core, or a missing
//!   spin factor: misses by 1e-4 to O(1) relative — see the controls.
//! * If ferric were actually computing CPHF (exchange in the kernel): misses
//!   the dRPA reference by ~10% and MATCHES `alpha_cphf` — the always-on scope
//!   control asserts the opposite.
//! * AO-order artifact: the injected MOs are checked orthonormal under
//!   FERRIC'S overlap (CᵀSC = I) and the reference overlap is compared
//!   elementwise first, so a wrong permutation fails before α is computed.
//!
//! # TOLERANCES — PLACEHOLDERS until the build agent measures them
//!
//! | quantity | measured max rel |d| | bar |
//! |---|---:|---:|
//! | overlap, PySCF (ferric order) vs ferric (abs) | 1.3e-15 | 1e-12 |
//! | CᵀSC − I (abs) | (asserted, not printed) | 1e-9 |
//! | α, same orbitals (rel to ‖α‖max) | 4.8e-14 | 5e-13 |
//! | RHF energy (abs, Ha) | 8.8e-13 | 1e-10 |
//! | α, full chain (rel) | 5.3e-10 | 5e-9 |
//!
//! # NEGATIVE CONTROLS / MUTATIONS
//!
//! * ALWAYS ON — aux swap: ferric α with `def2-universal-jkfit` in place of the
//!   matched aux, same orbitals, must MISS the reference by > 1000× the
//!   same-orbital bar (proves the bar resolves the aux choice, i.e. the aux
//!   basis is actually matched rather than irrelevant).
//! * ALWAYS ON — scope: ferric α must MISS `alpha_cphf` by > 1000× the
//!   full-chain bar (ferric does not compute CPHF α).
//! * ALWAYS ON — orbital-energy kick: ε_virt += 1e-3 Ha must move α by more
//!   than 1000× the same-orbital bar (the injected ε are read).
//! * MUTATION A — change `4.0 * bare - 16.0 * coupled` to
//!   `4.0 * bare - 8.0 * coupled` in `pdep_polarizability_static`: fails by
//!   O(10%).
//! * MUTATION B — set `frozen_core: 1` in its `RiMp2Config`: fails by ~1e-4
//!   relative (core excitations are small but not below the bar).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::pdep_polarizability_static;
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::Array2;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/static_alpha";
const MOL_DIR: &str = "testdata/molecules/validation";

// PLACEHOLDER bars — see the module doc's TOLERANCES table.
const TOL_S: f64 = 1e-12;
const TOL_ORTHO: f64 = 1e-9;
const TOL_ALPHA_SAME_C_REL: f64 = 5e-13;
const TOL_E_SCF: f64 = 1e-10;
const TOL_ALPHA_CHAIN_REL: f64 = 5e-9;
const TOL_ENUC: f64 = 1e-9;
const MUST_MISS: f64 = 1000.0;
const EPS_KICK: f64 = 1e-3;

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
        .expect("ferric-rpa manifest dir should be <root>/crates/ferric-rpa")
        .to_path_buf()
}

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             `scripts/validation/gen_properties.py alpha` — a missing reference is a failure, \
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

fn text<'a>(v: &'a Value, ptr: &str, ctx: &str) -> &'a str {
    v.pointer(ptr)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not a string"))
}

fn mat(v: &Value, ptr: &str, ctx: &str) -> Array2<f64> {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"));
    let n = rows.len();
    let m = rows
        .first()
        .and_then(Value::as_array)
        .map_or(0, |r| r.len());
    Array2::from_shape_fn((n, m), |(i, j)| {
        rows[i][j]
            .as_f64()
            .unwrap_or_else(|| panic!("{ctx}: {ptr}[{i}][{j}] is not a number"))
    })
}

fn tensor(a: &Array2<f64>) -> [[f64; 3]; 3] {
    assert_eq!(a.dim(), (3, 3), "alpha tensor must be 3x3");
    std::array::from_fn(|i| std::array::from_fn(|j| a[(i, j)]))
}

/// max_ij |a_ij - b_ij| / max_ij |b_ij|.
fn rel_diff(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    let scale = b.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
    let d = a
        .iter()
        .flatten()
        .zip(b.iter().flatten())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    d / scale
}

fn check(ctx: &str, what: &str, d: f64, tol: f64) {
    eprintln!("{ctx}: {what:<26} {d:.2e} (tol {tol:.0e})");
    assert!(d < tol, "{ctx}: {what}: {d:.2e} >= {tol:.0e}");
}

fn alpha(
    mol: &Molecule,
    obs: &PreparedBasis,
    aux_name: &str,
    scf: &ScfResult,
    ctx: &str,
) -> [[f64; 3]; 3] {
    let aux = basis::bundled(aux_name).unwrap_or_else(|e| panic!("{ctx}: aux {aux_name}: {e:?}"));
    let dfbs = PreparedBasis::new(mol, &aux).unwrap();
    pdep_polarizability_static(
        mol,
        obs,
        &dfbs,
        scf,
        Operator::coulomb(),
        &PdepRpaConfig::default(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: pdep_polarizability_static failed: {e:?}"))
    .tensor
}

fn alpha_row(system: &str, basis_name: &str) {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    let aux_name = text(&r, "/aux_basis", &ctx).to_string();
    let swap_aux = text(&r, "/swap_aux_negative_control", &ctx).to_string();

    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let enuc_ref = num(&r, "/nuclear_repulsion", &ctx);
    assert!(
        (mol.nuclear_repulsion() - enuc_ref).abs() < TOL_ENUC,
        "{ctx}: nuclear repulsion mismatch — geometry/unit slip"
    );
    let bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let s = ferric_integrals::oneelectron::overlap(&obs);
    let s_ref = mat(&r, "/overlap_ferric_order", &ctx);
    assert_eq!(s.dim(), s_ref.dim(), "{ctx}: AO count differs");
    check(
        &ctx,
        "overlap max|d|",
        (&s - &s_ref).iter().fold(0.0_f64, |m, v| m.max(v.abs())),
        TOL_S,
    );

    // ferric's own RHF: the full-chain input, and a valid ScfResult shell to
    // inject PySCF's orbitals into.
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    };
    let own = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
    assert!(own.converged, "{ctx}: ferric RHF not converged");
    check(
        &ctx,
        "RHF energy |d|",
        (own.energy - num(&r, "/scf/energy", &ctx)).abs(),
        TOL_E_SCF,
    );

    // --- 1. Same orbitals.
    let c = mat(&r, "/mo_coeff_ferric_order", &ctx);
    let eps: Vec<f64> = r["mo_energy"]
        .as_array()
        .expect("mo_energy")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let nocc = r["nocc"].as_u64().expect("nocc") as usize;
    assert_eq!(nocc, mol.nelec() as usize / 2, "{ctx}: nocc");
    assert_eq!(c.ncols(), own.mos_r().ncols(), "{ctx}: MO count differs");
    let ctsc = c.t().dot(&s).dot(&c);
    let ortho = ctsc.indexed_iter().fold(0.0_f64, |m, ((i, j), v)| {
        m.max((v - f64::from(u8::from(i == j))).abs())
    });
    check(&ctx, "C^T S C - I", ortho, TOL_ORTHO);
    let occ = c.slice(ndarray::s![.., ..nocc]);
    let mut injected = own.clone();
    injected.mos_alpha = c.clone();
    injected.eps_alpha = eps.clone();
    injected.density_total = occ.dot(&occ.t()) * 2.0;
    injected.density_alpha = occ.dot(&occ.t());

    let a_ref = tensor(&mat(&r, "/alpha_drpa_ri", &ctx));
    let a_cphf = tensor(&mat(&r, "/alpha_cphf", &ctx));
    let a_exact = tensor(&mat(&r, "/alpha_drpa_exact_eri", &ctx));
    eprintln!(
        "{ctx}: [scope] reference iso dRPA-RI {:.8} exact-ERI {:.8} CPHF {:.8}",
        (a_ref[0][0] + a_ref[1][1] + a_ref[2][2]) / 3.0,
        (a_exact[0][0] + a_exact[1][1] + a_exact[2][2]) / 3.0,
        (a_cphf[0][0] + a_cphf[1][1] + a_cphf[2][2]) / 3.0,
    );
    let a1 = alpha(&mol, &obs, &aux_name, &injected, &ctx);
    check(
        &ctx,
        "alpha same-orbitals (rel)",
        rel_diff(&a1, &a_ref),
        TOL_ALPHA_SAME_C_REL,
    );

    // Control: orbital energies are read.
    let mut kicked = injected.clone();
    for e in kicked.eps_alpha.iter_mut().skip(nocc) {
        *e += EPS_KICK;
    }
    let ak = alpha(&mol, &obs, &aux_name, &kicked, &ctx);
    let moved = rel_diff(&ak, &a1);
    eprintln!("{ctx}: control eps kick moves alpha by {moved:.2e} (rel)");
    assert!(
        moved > MUST_MISS * TOL_ALPHA_SAME_C_REL,
        "{ctx}: a {EPS_KICK:.0e} Ha virtual shift moved alpha by only {moved:.2e}"
    );

    // Control: the aux basis matters at this bar.
    let a_swap = alpha(&mol, &obs, &swap_aux, &injected, &ctx);
    let miss = rel_diff(&a_swap, &a_ref);
    eprintln!("{ctx}: control aux swap ({swap_aux}) misses by {miss:.2e} (rel)");
    assert!(
        miss > MUST_MISS * TOL_ALPHA_SAME_C_REL,
        "{ctx}: alpha with {swap_aux} is within {:.0e} of the {aux_name} reference — the bar \
         does not resolve the aux basis",
        MUST_MISS * TOL_ALPHA_SAME_C_REL
    );

    // --- 2. Full chain.
    let a2 = alpha(&mol, &obs, &aux_name, &own, &ctx);
    check(
        &ctx,
        "alpha full chain (rel)",
        rel_diff(&a2, &a_ref),
        TOL_ALPHA_CHAIN_REL,
    );

    // Scope: ferric's α is not CPHF.
    let cphf_miss = rel_diff(&a2, &a_cphf);
    eprintln!("{ctx}: [scope] ferric alpha vs CPHF alpha {cphf_miss:.2e} (rel)");
    assert!(
        cphf_miss > MUST_MISS * TOL_ALPHA_CHAIN_REL,
        "{ctx}: ferric's alpha is within {:.0e} of the CPHF (exchange-kernel) alpha — the row's \
         stated definition (direct RPA, no exchange) would be wrong",
        MUST_MISS * TOL_ALPHA_CHAIN_REL
    );
}

#[test]
#[ignore = "validation: Static α"]
fn static_alpha_h2o_aug_cc_pvdz_vs_numpy() {
    alpha_row("h2o", "aug-cc-pvdz");
}

#[test]
#[ignore = "validation: Static α"]
fn static_alpha_ch3oh_cc_pvdz_vs_numpy() {
    alpha_row("ch3oh", "cc-pvdz");
}
