//! VALIDATION tier — VALIDATION.md rows "ESP at nuclei", "Electric field at
//! nuclei" and "Becke volumes".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_density_properties \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! `ferric_scf::properties::{esp_at_atoms, electric_field_at_atoms,
//! atomic_effective_volumes_becke}` against PySCF 2.13 + numpy, from
//! `scripts/validation/gen_properties.py` (references in
//! `testdata/reference/validation/density_properties/` and
//! `.../free_atom_volumes/`). Systems: H2O and CH3OH (RHF) and the HO2 radical
//! (UHF, stability-checked) at cc-pVDZ AND def2-SVP; free H, C, N, O atoms
//! (UKS-PBE, C/O as the fractional-occupation ensemble) at aug-cc-pVDZ.
//!
//! Two comparisons per case, against the SAME reference numbers:
//!
//! 1. SAME DENSITY (integral level). The reference file carries PySCF's
//!    converged density permuted into ferric's AO order; ferric's property
//!    functions are fed THAT density. The SCF is out of the comparison, so the
//!    difference is the property code alone (libint2 vs libcint integrals;
//!    ferric's grid/AO evaluation vs a numpy rebuild of the same grid on
//!    PySCF's AO values).
//! 2. FULL CHAIN. ferric's own SCF density -> property. Adds the SCF
//!    convergence difference (first order for these one-electron properties).
//!
//! Before either, the AO-order convention is TESTED: the reference's PySCF
//! overlap matrix (in ferric order) must equal ferric's own overlap
//! elementwise. A wrong shell match, a swapped m-component or a solid-harmonic
//! sign convention difference fails there by O(0.1), not later as a mystery.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric's property code is right: same-density agreement at the
//!   integral floor (~1e-10 a.u. ESP/field; ~1e-10 relative volumes, the grid
//!   being the same points), full chain at the SCF floor (~1e-7).
//! * If the density is not actually READ (a property computed from a cached or
//!   default density): the perturbation control below fails — perturbing D by
//!   1e-3 I must move every property by >> the bar.
//! * If the AO order were silently wrong but "cancelled" in a symmetric test:
//!   impossible here, because the overlap check is elementwise and runs first.
//! * If the reference were for another basis (mislabelled file): the full-chain
//!   basis-swap control fails (cc-pVDZ vs def2-SVP ESPs differ by ~1e-3 a.u.).
//! * Scope of the Becke row: the reference rebuilds ferric's PARTITION
//!   (Becke 1988 Eq. A4 on ferric's Bragg-Slater radii) from its formula, so
//!   the same-density comparison checks quadrature, AO values and contraction
//!   — NOT the choice of partition, which is a definition. The reference also
//!   carries the same volumes on a dense (200, 590) grid; the test prints
//!   ferric's default-grid quadrature error but does not assert it.
//!
//! # TOLERANCES — PLACEHOLDERS until the build agent measures them
//!
//! | quantity | measured max |d| | bar |
//! |---|---:|---:|
//! | overlap, PySCF (ferric order) vs ferric | 1.3e-15 | 1e-12 |
//! | ESP, same density | 4.8e-14 a.u. | 5e-13 a.u. |
//! | field, same density | 1.5e-13 a.u. | 1e-12 a.u. |
//! | Becke volume, same density (relative) | 7.4e-16 | 1e-13 |
//! | SCF energy, ferric vs PySCF | 9.4e-12 Ha | 1e-10 Ha |
//! | ESP, full chain | 4.6e-9 a.u. | 5e-8 a.u. |
//! | field, full chain | 2.7e-9 a.u. | 3e-8 a.u. |
//! | Becke volume, full chain (relative) | 7.1e-9 | 5e-8 |
//! | free atom UKS energy | 4.3e-14 Ha | 1e-10 Ha |
//! | free atom volume, full chain (relative) | 4.8e-9 | 5e-8 |
//!
//! Set each bar to 3-10x the measured maximum (protocol §3.4) and record the
//! measured column here and on the site page.
//!
//! # NEGATIVE CONTROLS / MUTATIONS
//!
//! * ALWAYS ON — density perturbation: D -> D + 1e-3 I (every AO, so the
//!   diffuse functions that dominate the r^3 volume integrand are kicked too)
//!   must move the ESP, the field and the volumes by more than [`MUST_MISS`] x
//!   their bars. Proves the functions read the density they are given.
//! * ALWAYS ON — basis swap: ferric's full-chain ESP at basis A must MISS the
//!   basis-B reference by more than 100x the full-chain bar.
//! * ALWAYS ON — zero-density anchor: ESP(D = 0) is the pure nuclear sum and
//!   must equal reference ESP minus reference electronic ESP to 1e-12; the
//!   same for the field.
//! * MUTATION A — flip the sign of `e_elec[d] -= acc` in
//!   `electric_field_at_atoms` (ferric-scf/src/properties.rs): the same-density
//!   field fails on every non-symmetric component.
//! * MUTATION B — drop the `2.0 *` off-diagonal factor in `esp_at_atoms`: the
//!   same-density ESP fails by O(1) a.u.
//! * MUTATION C — change the Becke size-adjustment clip in
//!   `ferric-dft/src/becke.rs` from 0.5 to 0.45: the heteroatom volumes fail
//!   the same-density bar (a partition change, which is exactly what this row
//!   CAN see because the reference pins ferric's stated formula).
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::properties::{
    atomic_effective_volumes_becke, electric_field_at_atoms, esp_at_atoms,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;
use serde_json::Value;

const DENSITY_DIR: &str = "testdata/reference/validation/density_properties";
const ATOM_DIR: &str = "testdata/reference/validation/free_atom_volumes";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASES: [&str; 2] = ["cc-pvdz", "def2-svp"];
const ATOM_BASIS: &str = "aug-cc-pvdz";

// PLACEHOLDER bars — see the module doc's TOLERANCES table.
const TOL_S: f64 = 1e-12;
const TOL_ESP_SAME_D: f64 = 5e-13;
const TOL_FIELD_SAME_D: f64 = 1e-12;
const TOL_VOL_SAME_D_REL: f64 = 1e-13;
const TOL_E_SCF: f64 = 1e-10;
const TOL_ESP_CHAIN: f64 = 5e-8;
const TOL_FIELD_CHAIN: f64 = 3e-8;
const TOL_VOL_CHAIN_REL: f64 = 5e-8;
const TOL_E_ATOM: f64 = 1e-10;
const TOL_VOL_ATOM_CHAIN_REL: f64 = 5e-8;
const TOL_ENUC: f64 = 1e-9;
const TOL_ANCHOR: f64 = 1e-12;
/// A control must move a quantity by at least this multiple of its bar.
const MUST_MISS: f64 = 1000.0;
/// Size of the `D + kick * I` density perturbation (the always-on control).
const DENSITY_KICK: f64 = 1e-3;

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
fn reference(dir: &str, system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(dir)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_properties.py — a missing reference is a failure, never a skip",
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

fn vec1(v: &Value, ptr: &str, ctx: &str) -> Vec<f64> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
        .iter()
        .map(|x| {
            x.as_f64()
                .unwrap_or_else(|| panic!("{ctx}: {ptr}: non-number"))
        })
        .collect()
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

struct System {
    mol: Molecule,
    bs: basis::BasisSet,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

/// Geometry, AO count and AO-ORDER checks, in that order, before any property.
fn load_system(system: &str, basis_name: &str, r: &Value, ctx: &str) -> System {
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let enuc_ref = num(r, "/nuclear_repulsion", ctx);
    let enuc = mol.nuclear_repulsion();
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{ctx}: nuclear repulsion {enuc:.12} vs reference {enuc_ref:.12} — geometry/unit mismatch"
    );
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let s = ferric_integrals::oneelectron::overlap(&prep);
    let nao_ref = r["nao"].as_u64().expect("nao") as usize;
    assert_eq!(
        s.nrows(),
        nao_ref,
        "{ctx}: AO count differs from the reference's"
    );
    // AO-order convention check: elementwise, so it cannot pass by symmetry.
    let s_ref = mat(r, "/overlap_ferric_order", ctx);
    let ds = (&s - &s_ref).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    eprintln!("{ctx}: overlap max|d| {ds:.2e} (tol {TOL_S:.0e})");
    assert!(
        ds < TOL_S,
        "{ctx}: PySCF overlap in ferric AO order differs from ferric's by {ds:.2e} — the AO \
         permutation / solid-harmonic convention in gen_properties.py is wrong, so no \
         density-level comparison below would mean anything"
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

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

fn max_rel_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0_f64, |m, (x, y)| {
        m.max((x - y).abs() / y.abs().max(1e-300))
    })
}

fn flat3(v: &[[f64; 3]]) -> Vec<f64> {
    v.iter().flat_map(|r| r.iter().copied()).collect()
}

fn check(ctx: &str, what: &str, d: f64, tol: f64) {
    eprintln!("{ctx}: {what:<28} max|d| {d:.2e} (tol {tol:.0e})");
    assert!(d < tol, "{ctx}: {what}: {d:.2e} >= {tol:.0e}");
}

/// The three properties of one density, flattened for comparison.
struct Props {
    esp: Vec<f64>,
    field: Vec<f64>,
    vol: Vec<f64>,
}

fn props(sys: &System, d: &Array2<f64>) -> Props {
    Props {
        esp: esp_at_atoms(&sys.mol, &sys.prep, d).unwrap(),
        field: flat3(&electric_field_at_atoms(&sys.mol, &sys.prep, d).unwrap()),
        vol: atomic_effective_volumes_becke(&sys.mol, &sys.prep, &sys.bs, d).unwrap(),
    }
}

fn ref_field(r: &Value, key: &str, ctx: &str) -> Vec<f64> {
    mat(r, key, ctx).iter().copied().collect()
}

fn rhf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-9,
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

/// One molecule × basis: same-density, controls, full chain. Returns ferric's
/// full-chain ESP for the cross-basis control.
fn check_molecule(system: &str, basis_name: &str) -> Vec<f64> {
    let r = reference(DENSITY_DIR, system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    let sys = load_system(system, basis_name, &r, &ctx);

    let esp_ref = vec1(&r, "/esp_at_nuclei", &ctx);
    let esp_el_ref = vec1(&r, "/esp_electronic_at_nuclei", &ctx);
    let field_ref = ref_field(&r, "/electric_field_at_nuclei", &ctx);
    let field_el_ref = ref_field(&r, "/electric_field_electronic_at_nuclei", &ctx);
    let vol_ref = vec1(&r, "/becke/volumes", &ctx);
    let vol_dense = vec1(&r, "/becke/volumes_dense_grid", &ctx);

    // --- Anchor: D = 0 leaves only the nuclear sums.
    let nao = sys.prep.nbasis();
    let zero = Array2::<f64>::zeros((nao, nao));
    let p0 = props(&sys, &zero);
    let esp_nuc_ref: Vec<f64> = esp_ref
        .iter()
        .zip(&esp_el_ref)
        .map(|(t, e)| t - e)
        .collect();
    let field_nuc_ref: Vec<f64> = field_ref
        .iter()
        .zip(&field_el_ref)
        .map(|(t, e)| t - e)
        .collect();
    check(
        &ctx,
        "anchor ESP(D=0)",
        max_abs_diff(&p0.esp, &esp_nuc_ref),
        TOL_ANCHOR,
    );
    check(
        &ctx,
        "anchor field(D=0)",
        max_abs_diff(&p0.field, &field_nuc_ref),
        TOL_ANCHOR,
    );

    // --- 1. Same density.
    let d_ref = mat(&r, "/density_total_ferric_order", &ctx);
    let p1 = props(&sys, &d_ref);
    check(
        &ctx,
        "ESP same-D",
        max_abs_diff(&p1.esp, &esp_ref),
        TOL_ESP_SAME_D,
    );
    check(
        &ctx,
        "field same-D",
        max_abs_diff(&p1.field, &field_ref),
        TOL_FIELD_SAME_D,
    );
    check(
        &ctx,
        "Becke vol same-D (rel)",
        max_rel_diff(&p1.vol, &vol_ref),
        TOL_VOL_SAME_D_REL,
    );
    eprintln!(
        "{ctx}: [measurement] ferric-grid vs dense-grid volume max rel {:.2e}",
        max_rel_diff(&p1.vol, &vol_dense)
    );

    // --- Control: a perturbed density must move every property.
    let d_kick = &d_ref + &(Array2::<f64>::eye(nao) * DENSITY_KICK);
    let pk = props(&sys, &d_kick);
    for (what, moved, bar) in [
        ("ESP", max_abs_diff(&pk.esp, &p1.esp), TOL_ESP_SAME_D),
        (
            "field",
            max_abs_diff(&pk.field, &p1.field),
            TOL_FIELD_SAME_D,
        ),
        (
            "volume (rel)",
            max_rel_diff(&pk.vol, &p1.vol),
            TOL_VOL_SAME_D_REL,
        ),
    ] {
        eprintln!("{ctx}: control kick moves {what} by {moved:.2e}");
        assert!(
            moved > MUST_MISS * bar,
            "{ctx}: a {DENSITY_KICK:.0e} density kick moved {what} by only {moved:.2e} \
             (< {MUST_MISS}x the bar) — the property does not respond to the density it is given"
        );
    }

    // --- 2. Full chain: ferric's own SCF on the reference's state.
    let res = if r["reference"] == "uhf" {
        let res = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &uhf_config())
            .unwrap_or_else(|e| panic!("{ctx}: solve_uhf failed: {e:?}"));
        let st = res
            .stability
            .as_ref()
            .expect("check_stability set but no verdict");
        assert_eq!(
            st.verdict(),
            StabilityVerdict::Stable,
            "{ctx}: ferric UHF state not STABLE: {}",
            st.summary()
        );
        res
    } else {
        solve_rhf(
            &sys.ctx,
            &sys.mol,
            &sys.prep,
            Operator::coulomb(),
            &sys.bounds,
            &rhf_config(),
        )
        .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"))
    };
    assert!(res.converged, "{ctx}: ferric SCF not converged");
    check(
        &ctx,
        "SCF energy",
        (res.energy - num(&r, "/scf/energy", &ctx)).abs(),
        TOL_E_SCF,
    );
    let p2 = props(&sys, res.density_total());
    check(
        &ctx,
        "ESP full chain",
        max_abs_diff(&p2.esp, &esp_ref),
        TOL_ESP_CHAIN,
    );
    check(
        &ctx,
        "field full chain",
        max_abs_diff(&p2.field, &field_ref),
        TOL_FIELD_CHAIN,
    );
    check(
        &ctx,
        "Becke vol full chain (rel)",
        max_rel_diff(&p2.vol, &vol_ref),
        TOL_VOL_CHAIN_REL,
    );
    p2.esp
}

/// Two-input requirement: the full-chain ESP must RESPOND to the basis.
fn molecule_row(system: &str) {
    let esp = [
        check_molecule(system, BASES[0]),
        check_molecule(system, BASES[1]),
    ];
    for (i, b_self) in BASES.iter().enumerate() {
        let b_other = BASES[1 - i];
        let other = vec1(
            &reference(DENSITY_DIR, system, b_other),
            "/esp_at_nuclei",
            system,
        );
        let d = max_abs_diff(&esp[i], &other);
        eprintln!("{system}: ferric {b_self} ESP vs {b_other} reference max|d| {d:.2e}");
        assert!(
            d > 100.0 * TOL_ESP_CHAIN,
            "{system}: ferric {b_self} ESP is within {:.0e} of the {b_other} reference — \
             the comparison does not resolve the basis",
            100.0 * TOL_ESP_CHAIN
        );
    }
}

/// Free atom: same-density volume, then ferric's own UKS-PBE (the TS-C6
/// free-atom recipe: fractional_occ + MOM after iter 5 for mult > 1).
fn atom_row(symbol_lc: &str) {
    let system = format!("{symbol_lc}_atom");
    let r = reference(ATOM_DIR, &system, ATOM_BASIS);
    let ctx = format!("{system}/{ATOM_BASIS}");
    let sys = load_system(&system, ATOM_BASIS, &r, &ctx);
    let vol_ref = vec1(&r, "/becke/volumes", &ctx);

    let d_ref = mat(&r, "/density_total_ferric_order", &ctx);
    let v1 = atomic_effective_volumes_becke(&sys.mol, &sys.prep, &sys.bs, &d_ref).unwrap();
    check(
        &ctx,
        "volume same-D (rel)",
        max_rel_diff(&v1, &vol_ref),
        TOL_VOL_SAME_D_REL,
    );
    eprintln!(
        "{ctx}: [measurement] ferric-grid vs dense-grid volume rel {:.2e}",
        max_rel_diff(&v1, &vec1(&r, "/becke/volumes_dense_grid", &ctx))
    );

    let nao = sys.prep.nbasis();
    let d_kick = &d_ref + &(Array2::<f64>::eye(nao) * DENSITY_KICK);
    let vk = atomic_effective_volumes_becke(&sys.mol, &sys.prep, &sys.bs, &d_kick).unwrap();
    let moved = max_rel_diff(&vk, &v1);
    assert!(
        moved > MUST_MISS * TOL_VOL_SAME_D_REL,
        "{ctx}: density kick moved the volume by only {moved:.2e}"
    );

    let frac = r["uks"]["fractional_occupation"]
        .as_bool()
        .expect("uks.fractional_occupation");
    let cfg = RhfConfig {
        xc: Some("PBE".to_string()),
        fractional_occ: sys.mol.multiplicity > 1,
        mom_after_iter: if sys.mol.multiplicity > 1 { 5 } else { 0 },
        max_iter: 400,
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };
    let res = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &cfg)
        .unwrap_or_else(|e| panic!("{ctx}: solve_uhf (UKS-PBE) failed: {e:?}"));
    assert!(res.converged, "{ctx}: ferric UKS not converged");
    eprintln!("{ctx}: reference fractional_occupation = {frac}");
    check(
        &ctx,
        "UKS-PBE energy",
        (res.energy - num(&r, "/uks/energy", &ctx)).abs(),
        TOL_E_ATOM,
    );
    let v2 =
        atomic_effective_volumes_becke(&sys.mol, &sys.prep, &sys.bs, res.density_total()).unwrap();
    check(
        &ctx,
        "volume full chain (rel)",
        max_rel_diff(&v2, &vol_ref),
        TOL_VOL_ATOM_CHAIN_REL,
    );
}

#[test]
#[ignore = "validation: ESP at nuclei / Electric field at nuclei / Becke volumes"]
fn density_properties_h2o_vs_pyscf() {
    molecule_row("h2o");
}

#[test]
#[ignore = "validation: ESP at nuclei / Electric field at nuclei / Becke volumes"]
fn density_properties_ch3oh_vs_pyscf() {
    molecule_row("ch3oh");
}

#[test]
#[ignore = "validation: ESP at nuclei / Electric field at nuclei / Becke volumes"]
fn density_properties_ho2_uhf_vs_pyscf() {
    molecule_row("ho2");
}

#[test]
#[ignore = "validation: Becke volumes"]
fn free_atom_volume_h_vs_pyscf() {
    atom_row("h");
}

#[test]
#[ignore = "validation: Becke volumes"]
fn free_atom_volume_c_vs_pyscf() {
    atom_row("c");
}

#[test]
#[ignore = "validation: Becke volumes"]
fn free_atom_volume_n_vs_pyscf() {
    atom_row("n");
}

#[test]
#[ignore = "validation: Becke volumes"]
fn free_atom_volume_o_vs_pyscf() {
    atom_row("o");
}
