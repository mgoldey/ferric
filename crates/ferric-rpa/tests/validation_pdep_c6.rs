//! VALIDATION tier — VALIDATION.md row "PDEP-RPA C6".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-rpa --test validation_pdep_c6 \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What ferric computes (read from the loops, not the names)
//!
//! `ferric_rpa::dispersion::pdep_dynamic_polarizability(.., Becke, None)`:
//!
//! * `freqs`/`weights`: `quadrature::build_quadrature(&cfg.quadrature)`; the
//!   default (MiniMax, n = 20) is Gauss-Legendre mapped by ω = u₀(1+x)/(1−x)
//!   with u₀ = 0.5.
//! * `molecular[k]` (`properties::molecular_dynamic_polarizability`): the
//!   closed-shell DIRECT-RPA (time-dependent Hartree, no exchange kernel)
//!   dipole polarizability with an RI Coulomb kernel, frozen_core = 0
//!   (hard-coded), full rank (this path never calls the PDEP eigensolver; the
//!   "PDEP" in the name is historical):
//!
//!   ```text
//!     α(iω) = 4 μᵀ [diag((ω²+Δ²)/Δ) + 4K]⁻¹ μ,  K = (ia|P)V⁻¹(P|jb),  μ_ia = ⟨i|r|a⟩
//!   ```
//!
//!   evaluated per node by SMW on ε̃(ω) = I + 4 B̃ diag(Δ/(ω²+Δ²)) B̃ᵀ.
//! * `per_atom[a][k]` (`properties::pdep_polarizability_becke_dynamic`):
//!   α^A_dj = sym 4 (m^A_d)ᵀ R(iω) M_j with m^A the Becke-weighted ATOM-CENTRED
//!   dipole (r − R_A) on the flat (75, 110) grid and M = Σ_A m^A (NOT the
//!   analytic lab-frame dipole). So Σ_A α^A = 4 Mᵀ R M exactly, and it is NOT
//!   the molecular tensor: the two differ by the charge-transfer pieces
//!   Σ_A R_A q^A of r = Σ_A w_A (r − R_A) + Σ_A w_A R_A (≈37% of α_iso for
//!   H2O, ≈28% for N2 at ω_0 — see the reference's `per_atom_becke` block).
//! * `casimir_polder_c6`: `c6_molecular_iso` = (3/π) Σ_k w_k ᾱ(iω_k)² from
//!   `molecular`; `c6_iso_pair`/`c6_aniso_pair` from `per_atom` (aniso is
//!   elementwise, (3/π) Σ_k w_k α^A_ij α^B_ij).
//!
//! The reference (`scripts/validation/gen_pdep_c6.py`, files in
//! `testdata/reference/validation/pdep_c6/`) is PySCF RHF + numpy: the
//! molecular α(iω) by SUM OVER STATES from one Casida diagonalisation
//! Ω² Z = Δ^½(Δ+4K)Δ^½ Z (a different algebraic route from the per-node SMW;
//! the generator cross-checks it against a dense per-node solve, 1.5e-14 /
//! 5.9e-14), the RI kernel from `int3c2e`/`int2c2e` with the SAME aux read
//! from ferric's JSON, ferric's quadrature rebuilt with `numpy.leggauss`, and
//! the per-atom pieces on a numpy replica of ferric's Becke grid (the replica
//! the Becke-volume row matched to 7e-16). It also stores the EXACT
//! Casimir–Polder integral of the SOS form (quadrature error), and for scope
//! the exact-ERI dRPA and TDHF (exchange kernel) C6.
//!
//! Systems: H2O and N2 / aug-cc-pVDZ, aux aug-cc-pvdz-rifit (two molecules:
//! bent polar vs linear non-polar, different anisotropy and atom types).
//!
//! # Exactness anchor
//!
//! ω = 0: `molecular_dynamic_polarizability(.., &[0.0])` must equal ferric's
//! own `pdep_polarizability_static` (independent code path: Δ⁻¹ instead of
//! Δ/(ω²+Δ²)) and the numpy static α — which is also the `static_alpha` row's
//! reference (H2O: the two generators agree to 6.7e-13).
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric's α(iω) is the stated dRPA-RI response: same-orbital agreement
//!   at every node to ~1e-12 relative, and C6 likewise.
//! * A wrong frequency map (ω vs ω², a missing Jacobian, wrong u₀) leaves the
//!   ω = 0 anchor intact and fails per-node and C6 — hence BOTH are checked,
//!   and the nodes/weights are compared directly first.
//! * A wrong spin factor / SMW coefficient fails ω = 0 and every node by O(10%).
//! * Exchange in the kernel (TDHF) would miss by ~35% on C6 — the scope
//!   control asserts ferric does not match the TDHF C6.
//! * Wrong aux basis: the aux-swap control proves the bar resolves it.
//!
//! # Definitions that disagree BY CONSTRUCTION (measured, not asserted equal)
//!
//! * Σ_A α^A_Becke ≠ molecular α (charge transfer, above). Tested: the GAP
//!   ferric reports equals the numpy gap.
//! * `pdep_polarizability_becke` (static per-atom) uses the analytic lab-frame
//!   dipole as the RIGHT operand; the dynamic per-atom path uses M = Σ_A m^A.
//!   Its doc says it matches the dynamic ω = 0 limit exactly; the reference
//!   shows the two definitions differ by up to 36% (H2O) / 20% (N2) per
//!   tensor element. Each ferric path is checked against its OWN definition;
//!   the gap is printed.
//! * Quadrature: ferric's 20-node C6 vs the exact Casimir–Polder integral is
//!   3.3e-8 (H2O) / −1.9e-8 (N2) relative — printed, not asserted.
//!
//! # TOLERANCES (measured 2026-09-30, release, H2O and N2 / aug-cc-pVDZ)
//!
//! The defects the controls inject move α(iω) by ≥5e-5 relative (orbital-energy kick)
//! and C6 by ≥1.4e-5 (aux swap), so every bar sits orders of magnitude between the
//! measured agreement and the smallest injected defect.
//!
//! | quantity | measured max | bar |
//! |---|---:|---:|
//! | quadrature nodes/weights (rel) | 5.7e-14 | 1e-12 |
//! | overlap, PySCF (ferric order) vs ferric (abs) | 1.3e-15 | 1e-12 |
//! | molecular α(iω), same orbitals, worst node | 4.9e-14 | 1e-11 |
//! | ω=0 dynamic vs ferric static (internal) | 1.9e-16 | 1e-12 |
//! | C6 iso / tensor, same orbitals | 6.2e-14 | 1e-11 |
//! | per-atom α^A(iω), same orbitals, worst node | 3.4e-14 | 1e-10 |
//! | RHF energy (abs, Ha) | 1.5e-12 | 1e-10 |
//! | molecular α(iω) and C6, full chain | 5.3e-10 | 5e-9 |
//!
//! # NEGATIVE CONTROLS / MUTATIONS
//!
//! * ALWAYS ON — orbital-energy kick (ε_virt += 1e-3 Ha) must move the
//!   molecular α(iω) by > MUST_MISS × the same-orbital bar.
//! * ALWAYS ON — aux swap (`def2-universal-jkfit`) must miss the reference
//!   C6 by > MUST_MISS × the same-orbital bar.
//! * ALWAYS ON — scope: ferric C6 must miss the TDHF C6 by > MUST_MISS × the
//!   full-chain bar.
//! * MUTATION A — in `molecular_dynamic_polarizability` (closed shell) set
//!   `let omega2 = omega;`: ω = 0 still passes, every other node and C6 fail.
//! * MUTATION B — `16.0 * coupled` → `8.0 * coupled` in the same function:
//!   ω = 0 anchor and every node fail by O(10%).
//! * MUTATION C — `let pref = 3.0 / PI;` → `6.0 / PI` in `casimir_polder_c6`:
//!   α passes, C6 fails by 100%.
//! * MUTATION D — in `pdep_polarizability_becke_dynamic` (closed shell) build
//!   `mu_flat` from the analytic dipole instead of Σ_A m^A: per-atom fails
//!   by ~20-36%.

use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::{
    casimir_polder_c6, pdep_dynamic_polarizability, DispersionPartition, DynamicPolarizability,
};
use ferric_rpa::properties::{
    molecular_dynamic_polarizability, pdep_polarizability_becke, pdep_polarizability_becke_dynamic,
    pdep_polarizability_static,
};
use ferric_rpa::quadrature::build_quadrature;
use ferric_rpa::PdepRpaConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use ndarray::Array2;
use serde_json::Value;

type T3 = [[f64; 3]; 3];

const ROW_DIR: &str = "testdata/reference/validation/pdep_c6";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS: &str = "aug-cc-pvdz";

// Bars: see the module doc's TOLERANCES table (measured values alongside).
const TOL_QUAD_REL: f64 = 1e-12;
const TOL_S: f64 = 1e-12;
const TOL_ORTHO: f64 = 1e-9;
const TOL_ALPHA_SAME_C_REL: f64 = 1e-11;
const TOL_W0_INTERNAL_REL: f64 = 1e-12;
const TOL_C6_SAME_C_REL: f64 = 1e-11;
const TOL_PER_ATOM_SAME_C_REL: f64 = 1e-10;
const TOL_E_SCF: f64 = 1e-10;
const TOL_CHAIN_REL: f64 = 5e-9;
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

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             `scripts/validation/gen_pdep_c6.py` — a missing reference is a failure, never a skip",
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

fn vec1(v: &Value, ptr: &str, ctx: &str) -> Vec<f64> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
        .iter()
        .map(|x| {
            x.as_f64()
                .unwrap_or_else(|| panic!("{ctx}: {ptr} has a non-number"))
        })
        .collect()
}

fn mat_of(rows: &Value, ptr: &str, ctx: &str) -> Array2<f64> {
    let rows = rows
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} is not an array"));
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

fn mat(v: &Value, ptr: &str, ctx: &str) -> Array2<f64> {
    let node = v
        .pointer(ptr)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing"));
    mat_of(node, ptr, ctx)
}

fn tensor(a: &Array2<f64>) -> T3 {
    assert_eq!(a.dim(), (3, 3), "tensor must be 3x3");
    std::array::from_fn(|i| std::array::from_fn(|j| a[(i, j)]))
}

fn t3(v: &Value, ptr: &str, ctx: &str) -> T3 {
    tensor(&mat(v, ptr, ctx))
}

/// A JSON array of 3x3 tensors.
fn t3_list(v: &Value, ptr: &str, ctx: &str) -> Vec<T3> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
        .iter()
        .map(|t| tensor(&mat_of(t, ptr, ctx)))
        .collect()
}

/// max_ij |a_ij - b_ij| / max_ij |b_ij|.
fn rel_diff(a: &T3, b: &T3) -> f64 {
    let scale = b.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
    let d = a
        .iter()
        .flatten()
        .zip(b.iter().flatten())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    d / scale
}

/// Worst per-node `rel_diff` over a frequency list (each node scaled by its
/// own largest element, so the ω ≈ 145 tail node is held to the same bar).
fn worst_rel(a: &[T3], b: &[T3], ctx: &str) -> f64 {
    assert_eq!(a.len(), b.len(), "{ctx}: node count differs");
    a.iter()
        .zip(b)
        .map(|(x, y)| rel_diff(x, y))
        .fold(0.0_f64, f64::max)
}

fn iso(t: &T3) -> f64 {
    (t[0][0] + t[1][1] + t[2][2]) / 3.0
}

fn check(ctx: &str, what: &str, d: f64, tol: f64) {
    eprintln!("{ctx}: {what:<40} {d:.2e} (tol {tol:.0e})");
    assert!(d < tol, "{ctx}: {what}: {d:.2e} >= {tol:.0e}");
}

struct Case {
    r: Value,
    ctx: String,
    mol: Molecule,
    bs: BasisSet,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    aux_name: String,
    swap_aux: String,
    own: ScfResult,
    injected: ScfResult,
    nocc: usize,
}

fn aux(mol: &Molecule, name: &str, ctx: &str) -> PreparedBasis {
    let a = basis::bundled(name).unwrap_or_else(|e| panic!("{ctx}: aux {name}: {e:?}"));
    PreparedBasis::new(mol, &a).unwrap()
}

/// Load the reference, check geometry/AO order, run ferric's RHF, and build
/// a ScfResult carrying PySCF's orbitals (the same-orbital input).
fn setup(system: &str) -> Case {
    let r = reference(system);
    let ctx = format!("{system}/{BASIS}");
    let aux_name = text(&r, "/aux_basis", &ctx).to_string();
    let swap_aux = text(&r, "/swap_aux_negative_control", &ctx).to_string();

    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    assert!(
        (mol.nuclear_repulsion() - num(&r, "/nuclear_repulsion", &ctx)).abs() < TOL_ENUC,
        "{ctx}: nuclear repulsion mismatch — geometry/unit slip"
    );
    let bs = basis::bundled(BASIS).unwrap();
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
    let dfbs = aux(&mol, &aux_name, &ctx);

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

    let c = mat(&r, "/mo_coeff_ferric_order", &ctx);
    let eps = vec1(&r, "/mo_energy", &ctx);
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
    injected.eps_alpha = eps;
    injected.density_total = occ.dot(&occ.t()) * 2.0;
    injected.density_alpha = occ.dot(&occ.t());

    Case {
        r,
        ctx,
        mol,
        bs,
        obs,
        dfbs,
        aux_name,
        swap_aux,
        own,
        injected,
        nocc,
    }
}

fn dyn_pol(c: &Case, scf: &ScfResult, dfbs: &PreparedBasis) -> DynamicPolarizability {
    pdep_dynamic_polarizability(
        &c.mol,
        &c.obs,
        &c.bs,
        dfbs,
        scf,
        Operator::coulomb(),
        &PdepRpaConfig::default(),
        DispersionPartition::Becke,
        None,
    )
    .unwrap_or_else(|e| panic!("{}: pdep_dynamic_polarizability failed: {e:?}", c.ctx))
}

fn molecular_at(c: &Case, scf: &ScfResult, dfbs: &PreparedBasis, freqs: &[f64]) -> Vec<T3> {
    molecular_dynamic_polarizability(
        &c.mol,
        &c.obs,
        dfbs,
        scf,
        Operator::coulomb(),
        &PdepRpaConfig::default(),
        freqs,
    )
    .unwrap_or_else(|e| panic!("{}: molecular_dynamic_polarizability failed: {e:?}", c.ctx))
}

/// Molecular α(iω) at every node, the ω = 0 anchor, and the Casimir–Polder C6.
fn molecular_row(system: &str) {
    let c = setup(system);
    let ctx = c.ctx.as_str();
    let r = &c.r;

    // --- Quadrature: ferric's nodes are the reference's (rebuilt with numpy).
    let (freqs, weights) = build_quadrature(&PdepRpaConfig::default().quadrature);
    let f_ref = vec1(r, "/quadrature/freqs", ctx);
    let w_ref = vec1(r, "/quadrature/weights", ctx);
    assert_eq!(freqs.len(), f_ref.len(), "{ctx}: node count differs");
    let qd = freqs
        .iter()
        .zip(&f_ref)
        .chain(weights.iter().zip(&w_ref))
        .map(|(a, b)| ((a - b) / b).abs())
        .fold(0.0_f64, f64::max);
    check(ctx, "quadrature nodes+weights (rel)", qd, TOL_QUAD_REL);

    // --- 1. Same orbitals, through the public dispersion entry point.
    let dp = dyn_pol(&c, &c.injected, &c.dfbs);
    assert_eq!(
        dp.freqs, freqs,
        "{ctx}: dispersion path used a different grid"
    );
    assert_eq!(
        dp.weights, weights,
        "{ctx}: dispersion path used different weights"
    );
    let a_ref = t3_list(r, "/molecular/alpha_iw", ctx);
    check(
        ctx,
        "molecular alpha(iw) same-C, worst node",
        worst_rel(&dp.molecular, &a_ref, ctx),
        TOL_ALPHA_SAME_C_REL,
    );

    // Exactness anchor: ω = 0 is the static α (two ferric code paths + numpy).
    let a0 = molecular_at(&c, &c.injected, &c.dfbs, &[0.0])[0];
    let st = pdep_polarizability_static(
        &c.mol,
        &c.obs,
        &c.dfbs,
        &c.injected,
        Operator::coulomb(),
        &PdepRpaConfig::default(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: pdep_polarizability_static failed: {e:?}"))
    .tensor;
    check(
        ctx,
        "w=0 dynamic vs ferric static (internal)",
        rel_diff(&a0, &st),
        TOL_W0_INTERNAL_REL,
    );
    check(
        ctx,
        "w=0 dynamic vs numpy static",
        rel_diff(&a0, &t3(r, "/molecular/alpha_static", ctx)),
        TOL_ALPHA_SAME_C_REL,
    );

    // C6: ferric's own contraction, isotropic and elementwise tensor. The
    // tensor contraction is only applied to `per_atom` in ferric, so the
    // molecular tensor is routed through it as a one-site system.
    let c6 = casimir_polder_c6(&dp);
    let c6_ref = num(r, "/molecular/c6_iso", ctx);
    check(
        ctx,
        "C6 molecular iso same-C (rel)",
        ((c6.c6_molecular_iso - c6_ref) / c6_ref).abs(),
        TOL_C6_SAME_C_REL,
    );
    let one_site = DynamicPolarizability {
        freqs: dp.freqs.clone(),
        weights: dp.weights.clone(),
        per_atom: vec![dp.molecular.clone()],
        molecular: dp.molecular.clone(),
    };
    let c6_one = casimir_polder_c6(&one_site);
    check(
        ctx,
        "C6 molecular tensor same-C (rel)",
        rel_diff(
            &c6_one.c6_aniso_pair[0][0],
            &t3(r, "/molecular/c6_tensor", ctx),
        ),
        TOL_C6_SAME_C_REL,
    );
    check(
        ctx,
        "C6 one-site iso pair == molecular iso",
        ((c6_one.c6_iso_pair[(0, 0)] - c6.c6_molecular_iso) / c6.c6_molecular_iso).abs(),
        1e-14,
    );
    let exact = num(r, "/molecular/c6_iso_exact_integral", ctx);
    eprintln!(
        "{ctx}: [measurement] C6 iso {:.8} ; exact Casimir-Polder integral {exact:.8} ; \
         ferric quadrature error {:.2e} (rel)",
        c6.c6_molecular_iso,
        (c6.c6_molecular_iso - exact) / exact
    );
    eprintln!(
        "{ctx}: [scope] C6 dRPA exact-ERI {:.8}  TDHF exact-ERI {:.8}",
        num(r, "/scope/c6_iso_drpa_exact_eri", ctx),
        num(r, "/scope/c6_iso_tdhf_exact_eri", ctx)
    );

    // Control: orbital energies are read at every node.
    let mut kicked = c.injected.clone();
    for e in kicked.eps_alpha.iter_mut().skip(c.nocc) {
        *e += EPS_KICK;
    }
    let ak = molecular_at(&c, &kicked, &c.dfbs, &freqs);
    let moved = ak
        .iter()
        .zip(&dp.molecular)
        .map(|(x, y)| rel_diff(x, y))
        .fold(f64::INFINITY, f64::min);
    eprintln!("{ctx}: control eps kick moves alpha(iw) by >= {moved:.2e} at every node");
    assert!(
        moved > MUST_MISS * TOL_ALPHA_SAME_C_REL,
        "{ctx}: a {EPS_KICK:.0e} Ha virtual shift moved some node by only {moved:.2e}"
    );

    // Control: the aux basis matters at this bar.
    let swap = aux(&c.mol, &c.swap_aux, ctx);
    let a_swap = molecular_at(&c, &c.injected, &swap, &freqs);
    let c6_swap = casimir_polder_c6(&DynamicPolarizability {
        freqs: freqs.clone(),
        weights: weights.clone(),
        per_atom: vec![a_swap.clone()],
        molecular: a_swap,
    })
    .c6_molecular_iso;
    let miss = ((c6_swap - c6_ref) / c6_ref).abs();
    eprintln!(
        "{ctx}: control aux swap ({}) C6 misses by {miss:.2e} (rel)",
        c.swap_aux
    );
    assert!(
        miss > MUST_MISS * TOL_C6_SAME_C_REL,
        "{ctx}: C6 with {} is within {:.0e} of the {} reference — the bar does not resolve \
         the aux basis",
        c.swap_aux,
        MUST_MISS * TOL_C6_SAME_C_REL,
        c.aux_name
    );

    // --- 2. Full chain: ferric's own RHF.
    let dp2 = dyn_pol(&c, &c.own, &c.dfbs);
    check(
        ctx,
        "molecular alpha(iw) full chain, worst node",
        worst_rel(&dp2.molecular, &a_ref, ctx),
        TOL_CHAIN_REL,
    );
    let c6_2 = casimir_polder_c6(&dp2).c6_molecular_iso;
    check(
        ctx,
        "C6 molecular iso full chain (rel)",
        ((c6_2 - c6_ref) / c6_ref).abs(),
        TOL_CHAIN_REL,
    );

    // Scope: ferric's C6 is the direct-RPA one, not TDHF.
    let tdhf = num(r, "/scope/c6_iso_tdhf_exact_eri", ctx);
    let tdhf_miss = ((c6_2 - tdhf) / tdhf).abs();
    eprintln!("{ctx}: [scope] ferric C6 vs TDHF C6 {tdhf_miss:.2e} (rel)");
    assert!(
        tdhf_miss > MUST_MISS * TOL_CHAIN_REL,
        "{ctx}: ferric's C6 is within {:.0e} of the TDHF (exchange-kernel) C6 — the row's \
         stated definition (direct RPA) would be wrong",
        MUST_MISS * TOL_CHAIN_REL
    );
}

/// Becke per-atom α^A(iω), its sum over atoms, the pair C6, and both ω = 0
/// per-atom definitions.
fn per_atom_row(system: &str) {
    let c = setup(system);
    let ctx = c.ctx.as_str();
    let r = &c.r;
    let natoms = c.mol.atoms.len();
    assert_eq!(
        natoms,
        r["natoms"].as_u64().expect("natoms") as usize,
        "{ctx}: atom count"
    );

    let dp = dyn_pol(&c, &c.injected, &c.dfbs);
    let nfreq = dp.freqs.len();
    for a in 0..natoms {
        let pa_ref = t3_list(r, &format!("/per_atom_becke/alpha_iw/{a}"), ctx);
        check(
            ctx,
            &format!("per-atom {a} alpha(iw) same-C, worst node"),
            worst_rel(&dp.per_atom[a], &pa_ref, ctx),
            TOL_PER_ATOM_SAME_C_REL,
        );
    }

    // The partition: Σ_A α^A, and its (charge-transfer) gap to the molecular α.
    let sum: Vec<T3> = (0..nfreq)
        .map(|k| {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| (0..natoms).map(|a| dp.per_atom[a][k][i][j]).sum::<f64>())
            })
        })
        .collect();
    let sum_ref = t3_list(r, "/per_atom_becke/sum_over_atoms_iw", ctx);
    check(
        ctx,
        "sum_A alpha^A(iw) same-C, worst node",
        worst_rel(&sum, &sum_ref, ctx),
        TOL_PER_ATOM_SAME_C_REL,
    );
    let mol_ref = t3_list(r, "/molecular/alpha_iw", ctx);
    let gap = |m: &[T3], s: &[T3]| -> Vec<T3> {
        m.iter()
            .zip(s)
            .map(|(x, y)| std::array::from_fn(|i| std::array::from_fn(|j| x[i][j] - y[i][j])))
            .collect()
    };
    let gap_ferric = gap(&dp.molecular, &sum);
    let gap_ref = gap(&mol_ref, &sum_ref);
    check(
        ctx,
        "molecular - sum_A alpha^A (CT gap), worst",
        worst_rel(&gap_ferric, &gap_ref, ctx),
        TOL_PER_ATOM_SAME_C_REL,
    );
    let lab_grid = t3_list(r, "/per_atom_becke/alpha_lab_grid_iw", ctx);
    eprintln!(
        "{ctx}: [measurement] w_0: sum_A alpha^A iso {:.6} vs molecular {:.6} \
         (CT share {:.1}%); grid lab-frame dipole reproduces molecular to {:.2e} (rel)",
        iso(&sum[0]),
        iso(&dp.molecular[0]),
        100.0 * (1.0 - iso(&sum[0]) / iso(&dp.molecular[0])),
        worst_rel(&lab_grid, &mol_ref, ctx)
    );

    // Pair C6 from the per-atom tensors.
    let c6 = casimir_polder_c6(&dp);
    let iso_ref = mat(r, "/per_atom_becke/c6_iso_pair", ctx);
    let scale = iso_ref.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let d_iso = (&c6.c6_iso_pair - &iso_ref)
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        / scale;
    check(
        ctx,
        "C6 iso pair matrix same-C (rel)",
        d_iso,
        TOL_PER_ATOM_SAME_C_REL,
    );
    let mut d_aniso = 0.0_f64;
    for a in 0..natoms {
        for b in 0..natoms {
            let t = t3(r, &format!("/per_atom_becke/c6_aniso_pair/{a}/{b}"), ctx);
            d_aniso = d_aniso.max(rel_diff(&c6.c6_aniso_pair[a][b], &t));
        }
    }
    check(
        ctx,
        "C6 aniso pair tensors same-C (rel)",
        d_aniso,
        TOL_PER_ATOM_SAME_C_REL,
    );
    eprintln!(
        "{ctx}: [measurement] C6 pair sum {:.6} vs molecular {:.6}",
        c6.c6_iso_pair.sum(),
        c6.c6_molecular_iso
    );

    // ω = 0, both per-atom definitions, each against its own reference.
    let dyn0 = pdep_polarizability_becke_dynamic(
        &c.mol,
        &c.obs,
        &c.bs,
        &c.dfbs,
        &c.injected,
        Operator::coulomb(),
        &PdepRpaConfig::default(),
        &[0.0],
    )
    .unwrap_or_else(|e| panic!("{ctx}: pdep_polarizability_becke_dynamic failed: {e:?}"));
    let st = pdep_polarizability_becke(
        &c.mol,
        &c.obs,
        &c.bs,
        &c.dfbs,
        &c.injected,
        Operator::coulomb(),
        &PdepRpaConfig::default(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: pdep_polarizability_becke failed: {e:?}"));
    let dyn0_ref = t3_list(r, "/per_atom_becke/alpha_dynamic_w0", ctx);
    let st_ref = t3_list(r, "/per_atom_becke/alpha_static_lab_right", ctx);
    let dyn0_k0: Vec<T3> = (0..natoms).map(|a| dyn0[a][0]).collect();
    check(
        ctx,
        "per-atom dynamic(w=0) vs its definition",
        worst_rel(&dyn0_k0, &dyn0_ref, ctx),
        TOL_PER_ATOM_SAME_C_REL,
    );
    check(
        ctx,
        "per-atom static vs its definition",
        worst_rel(&st, &st_ref, ctx),
        TOL_PER_ATOM_SAME_C_REL,
    );
    eprintln!(
        "{ctx}: [measurement] per-atom static (lab-frame right operand) vs dynamic w=0 \
         (sum-of-atom-centred right operand): worst atom {:.2e} (rel) — different \
         definitions, not quadrature",
        worst_rel(&st, &dyn0_k0, ctx)
    );
}

#[test]
#[ignore = "validation: PDEP-RPA C6"]
fn pdep_c6_molecular_h2o_aug_cc_pvdz_vs_numpy() {
    molecular_row("h2o");
}

#[test]
#[ignore = "validation: PDEP-RPA C6"]
fn pdep_c6_molecular_n2_aug_cc_pvdz_vs_numpy() {
    molecular_row("n2");
}

#[test]
#[ignore = "validation: PDEP-RPA C6"]
fn pdep_c6_per_atom_becke_h2o_aug_cc_pvdz_vs_numpy() {
    per_atom_row("h2o");
}

#[test]
#[ignore = "validation: PDEP-RPA C6"]
fn pdep_c6_per_atom_becke_n2_aug_cc_pvdz_vs_numpy() {
    per_atom_row("n2");
}
