//! VALIDATION tier — VALIDATION.md row "Hirshfeld" (design §5.3, row 150),
//! plus the Hirshfeld effective volumes that ferric-cli's TS C6 divides.
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-rpa --test validation_hirshfeld \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What ferric computes (read from the loops, not the doc comments)
//!
//! `ferric_rpa::properties::hirshfeld_charges(mol, bs, D, proatom)`:
//!
//! ```text
//!   q_A = Z_A − s·n_A,   n_A = Σ_g w_g ρ(r_g) w_A(r_g),   s = N_e / Σ_B n_B
//!   w_A(r) = ρ⁰_A(|r − R_A|) / (Σ_B ρ⁰_B(|r − R_B|) + 1e-12)
//! ```
//!
//! on the default XC grid (75 TA-M4 × 110 Lebedev, unpruned, home-atom Becke
//! factor folded into w_g). The proatom ρ⁰ is either
//!
//! * `Some(provider)` — what ferric-cli and `ferric.hirshfeld_charges` pass
//!   (`ferric_scf::properties::scf_proatom_provider`): a `RadialProatom` table on
//!   r_k = 0.05·k Bohr (k = 1..600), interpolated by a cubic spline in ln ρ
//!   (even extension through r = 0, natural far end, linear-in-ln ρ tail),
//!   holding the spherical average of the free
//!   NEUTRAL atom's SCF density in the molecule's basis (RHF singlets, UHF with
//!   MOM after iteration 5 otherwise, for an HF molecular config); or
//! * `None` — `ferric.hirshfeld_charges(..., proatom="slater")`: a
//!   single normalized Slater exponential Z ξ³/π e^{−2ξr}, ξ = 1/R(Z) from
//!   `slater_xi_for_z`.
//!
//! `atomic_effective_volumes_hirshfeld` integrates Σ_g w_g ρ w_A |r − R_A|³
//! with the same weight on the SAME atom-centred Becke–Lebedev grid, with no
//! renormalization. `atomic_effective_volumes_hirshfeld_on_grid` is the same
//! integral on a uniform lattice (`GridSpec::bounding_box`, h = 0.20 Bohr,
//! 6 Bohr margin) — the quadrature MBD@rsSCS uses, and this row's negative
//! control for the one above.
//!
//! # The reference
//!
//! `scripts/validation/gen_hirshfeld.py` (files in
//! `testdata/reference/validation/hirshfeld/`): numpy on PySCF AO values, with
//! ferric's grid rebuilt from PySCF's TA-M4 and Lebedev rules and ferric's
//! Becke partition (gen_properties.py's construction), ferric's lattice rebuilt
//! point for point, and the proatom tables spherically averaged with Lebedev
//! 302 (ferric: 110; both exact for these atoms). Systems H2O, CO, CH3OH (RHF)
//! at cc-pVDZ and def2-SVP; free H, C, O (UHF) and He (RHF) atoms.
//!
//! Comparisons, each against the SAME reference numbers:
//!
//! 1. PROATOMS. (a) ferric's `spherically_averaged_proatom` on PySCF's
//!    free-atom density (ferric AO order): the averaging alone. (b) ferric's own
//!    free-atom SCF → table: adds the free-atom SCF (energy compared first, so a
//!    different UHF state fails as a state, not as a charge).
//! 2. SAME DENSITY, SAME PROATOMS. PySCF's molecular density and the reference
//!    tables are fed to ferric: the partition and quadrature code alone.
//! 3. FULL CHAIN. ferric's RHF and the production proatom provider
//!    `scf_proatom_provider` (ferric-cli's and the Python default), with this
//!    file's HF config.
//!
//! Stored in the reference and printed (the dense-grid charge is also bounded):
//! the charges on a dense (200, 590) grid (ferric's grid error), with the
//! proatom tabulated at 0.005 Bohr (the 0.05 Bohr table's error), and the
//! volumes on a dense Becke–Lebedev grid (the lattice's error). The dense-grid
//! charges agree with the same integral on PySCF's own level-9 grid (PySCF's
//! Becke partition and pruning) to ≤2.8e-7 e, so ferric's default-grid error
//! (≤2.0e-4 e, measured by numpy on the rebuilt grid) is ferric's, not the
//! reference's. The 0.05 Bohr table moves the dense-grid charges by ≤2.9e-7 e
//! against a 0.005 Bohr table.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric's code is the stated definition: same-density charges and
//!   volumes at the floating-point floor (~1e-13), full chain at the SCF floor.
//! * If the proatoms were assigned to the wrong atoms, or a different proatom
//!   model were used: the promolecule anchor fails by O(0.1) e (the Slater
//!   proatom on the same promolecule gives 0.104 e), and the proatom-swap
//!   control shows the size of such a change on the real molecules (0.23–0.72 e).
//! * If the density were not read: the density-kick control fails.
//! * If the AO order were wrong: the elementwise overlap check fails first, for
//!   the molecule and for every free atom.
//! * If ferric's free atom landed in a different UHF state than PySCF's stable
//!   one: the free-atom energy check fails before any table is compared.
//!
//! # EXACTNESS ANCHORS
//!
//! ## Volumes: one atom, where the partition is the identity
//!
//! For a single free atom the Hirshfeld weight is exactly 1 (`ρ⁰/(ρ⁰ +
//! 1e-12)`), so the volume collapses to the plain moment `∫ ρ r³` with no
//! partition left in it. Two independent forms, both asserted:
//!
//! * CLOSED FORM, no AOs and no SCF —
//!   `ferric_rpa::properties::tests::hirshfeld_volume_quadrature_matches_the_closed_form_slater_moment`
//!   (a fast unit test, NOT ignored) integrates the analytic Slater proatom
//!   `Z ξ³/π e^{−2ξr}` on the production grid against `7.5 Z/ξ³`: measured
//!   8.2e-7 relative at worst over Z ∈ {1, 6, 8}, bar 1e-5. This is the only
//!   check here that is independent of every other ferric quadrature.
//! * REFERENCE — `free_atom_volume_row` against `volume_dense_grid` in
//!   `{h,c,o}_atom_volume_*.json` (the same moment on a dense (200,590)
//!   Becke–Lebedev grid): see the TOLERANCES table.
//!
//! ## Charges: a promolecule has no charge
//!
//! A promolecule density (block-diagonal He and H free-atom densities, both
//! spherical) with its own proatoms has zero Hirshfeld charges up to the table
//! and the grid: measured 8.7e-6 e on ferric's grid with the 0.05 Bohr table
//! (the grid's error: 2.7e-9 e on the dense grid with the same table), 1.1e-9 e
//! on the dense grid with a 0.005 Bohr table. Asserted twice: on the
//! reference promolecule (same density) and on ferric's own free atoms.
//!
//! # TOLERANCES (measured 2026-10-01, worst over every system and basis)
//!
//! | quantity | measured max |d| | bar |
//! |---|---:|---:|
//! | overlap, PySCF (ferric order) vs ferric (molecules and atoms) | 8.9e-16 | 1e-12 |
//! | proatom table, same density (rel. to max ρ) | 2.4e-15 | 1e-12 |
//! | free-atom SCF energy (Ha) | 2.8e-14 | 1e-9 |
//! | proatom table, ferric's free atom (rel. to max ρ) | 1.1e-11 | 1e-7 |
//! | charges, same density + proatoms (SCF and Slater) (e) | 1.9e-13 | 1e-10 |
//! | Σq (e) | 5.1e-15 | 1e-10 |
//! | charges vs the dense-grid reference (e) | 2.0e-4 | 1e-3 |
//! | volumes vs the dense (200,590) grid, same density (rel) | 1.1e-7 | 1e-6 |
//! | ∫ρ − N_e on the volume grid (e) | 2.0e-7 | 2e-6 |
//! | CH3OH mirror-equivalent H volumes (rel) | 2.9e-14 | 1e-10 |
//! | free-atom weight-one moment vs the dense grid (rel) | 8.1e-13 | 1e-11 |
//! | free-atom 1e-12 weight-floor loss (rel) | 1.2e-3 (H) | 1e-2 |
//! | lattice volumes, same density (rel) | 3.2e-13 | 1e-11 |
//! | free-atom lattice volume, same density (rel) | 6.9e-14 | 1e-11 |
//! | RHF energy (Ha) | 9.3e-12 | 1e-10 |
//! | charges, full chain (e) | 2.6e-9 | 1e-6 |
//! | volumes, full chain vs same density (rel) | 4.2e-10 | 1e-6 |
//! | promolecule anchor |q| (e), both routes | 8.7e-6 | 5e-5 |
//!
//! # NEGATIVE CONTROLS / MUTATIONS
//!
//! * ALWAYS ON — density kick: D → D + 1e-3·I must move the charges by more
//!   than [`MUST_MISS`] × the same-density bar.
//! * ALWAYS ON — proatom swap: the Slater-proatom charges (`proatom="slater"`
//!   in Python) must MISS the SCF-proatom reference by more than [`SWAP_MIN`] e.
//! * ALWAYS ON — anchor control: the Slater proatom on the promolecule must give
//!   |q| > [`ANCHOR_SLATER_MIN`] e, so the anchor can fail.
//! * ALWAYS ON — CO sign: the SCF-proatom charges put C positive, O negative.
//! * ALWAYS ON — basis swap: ferric's full-chain charges at one basis must miss
//!   the other basis's reference by more than 100× the full-chain bar.
//! * ALWAYS ON — quadrature control (volumes): the bounding-box lattice,
//!   reached through `atomic_effective_volumes_hirshfeld_on_grid`, must MISS
//!   the volume bar (measured 1.2e-4 rel., 1100x the bar), the electron-count
//!   bar (1.7e-2 to 0.70 e, 8500x to 350,000x) and, on CH3OH, the mirror bar
//!   (2.6e-5, 2.6e5x). On the free atoms the same control is SET-level: the
//!   lattice's worst weight-one error over {H,C,O} × both bases is 8.6e-4,
//!   against the production grid's 8.1e-13, but oxygen's alone is 1.1e-5 —
//!   see `LATTICE_FREE_ATOM_MIN_WORST_REL`.
//! * MUTATION A — drop `+ eps_floor` in `hirshfeld_charges`
//!   (ferric-rpa/src/properties.rs): invisible where Σρ⁰ ≫ 1e-12, so this row
//!   CANNOT see it on these molecules; recorded so nobody reads the row as
//!   testing the floor.
//! * MUTATION B — use `rho_free[0]` for every atom in the weight: the
//!   same-density charges fail by 1.1–5.3 e and the anchor by 0.50 e (4 of 5
//!   tests; the one-atom free-volume test cannot see it, atom 0 being the only
//!   atom). Run 2026-09-30.
//! * MUTATION C — remove the `scale` renormalization: the same-density charges
//!   fail by the grid's electron-count error (7e-6 e for H2O, 1.7e-4 e for CO,
//!   7.5e-4 e for CH3OH on ferric's grid, from the reference files).
//! * MUTATION D — make `RadialProatom::at` (ferric-scf/src/properties.rs)
//!   drop the spline's moment terms (piecewise linear in ln ρ): the
//!   same-density charges fail by 3.5e-5–9.3e-5 e and the anchor's
//!   same-density charges by 9.9e-7 e (4 of 5 tests). Run 2026-10-01.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ferric_core::basis::{self, BasisSet};
use ferric_core::elements::z_to_symbol;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::{
    atomic_effective_volumes_hirshfeld, atomic_effective_volumes_hirshfeld_on_grid,
    hirshfeld_charges, hirshfeld_volume_grid, hirshfeld_volume_grid_config,
    spherically_averaged_proatom, ProatomProvider, RadialProatom,
};
use ferric_scf::properties::{proatom_ground_state_mult, scf_proatom_provider, scf_proatom_radii};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::{s, Array2};
use serde_json::Value;

const REF_DIR: &str = "testdata/reference/validation/hirshfeld";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASES: [&str; 2] = ["cc-pvdz", "def2-svp"];

// Bars — measured values in the module doc's TOLERANCES table.
const TOL_S: f64 = 1e-12;
const TOL_ENUC: f64 = 1e-9;
const TOL_TABLE_SAME: f64 = 1e-12;
const TOL_E_ATOM: f64 = 1e-9;
const TOL_TABLE_CHAIN: f64 = 1e-7;
const TOL_Q_SAME: f64 = 1e-10;
const TOL_SUM: f64 = 1e-10;
const TOL_Q_GRID: f64 = 1e-3;
const TOL_VOL_SAME_REL: f64 = 1e-11;
/// The production volume quadrature against the dense (200,590) Becke
/// reference (`/hirshfeld_volumes/scf_proatom_dense_grid`), same density and
/// proatoms. Measured max over H2O, CO, CH3OH x cc-pVDZ, def2-SVP: 1.1e-7
/// relative, and that residual is the 75-point radial floor, not the angular
/// one (see `properties::hirshfeld_volume_grid_config`'s convergence table).
/// Bar ~9x that.
///
/// On the bounding-box lattice the same figure is 2.4e-4 — 2200x the bar. The
/// negative control in `check_molecule` asserts the lattice MISSES this bar,
/// so a quadrature that silently reverted would fail here.
const TOL_VOL_VS_DENSE_REL: f64 = 1e-6;
/// `Σ_g w_g ρ(r_g)` on the volume quadrature against `mol.nelec()`. This is
/// the quadrature's own electron-count anchor: a grid that cannot resolve the
/// nuclear cusp misses it.
///
/// DERIVED FROM BOTH SIDES, not guessed. Measured max over H2O, CO, CH3OH x
/// cc-pVDZ, def2-SVP: 2.0e-7 e on the production volume grid and, on the
/// bounding-box lattice, a MINIMUM of 1.0e-2 e (H2O/def2-SVP) rising to
/// 0.70 e (CO/cc-pVDZ). Bar 10x the Becke side, five orders below the
/// lattice's best case. `check_molecule`'s negative control asserts the
/// lattice side, so a bar that drifted up toward it would be caught.
const TOL_NELEC_GRID: f64 = 2e-6;
/// Symmetry-equivalent atoms must integrate identically: CH3OH's two
/// mirror-image methyl hydrogens (indices 3 and 4, at z = ±0.890 A).
///
/// Measured 2.9e-14 relative on the production volume grid, against 2.6e-5
/// (cc-pVDZ) and 2.5e-5 (def2-SVP) on the bounding-box lattice. Bar 3400x the
/// measured value and 2.6e5x below the lattice's, which `check_molecule`'s
/// negative control asserts the lattice misses.
///
/// NOT exactly zero, and that is a property of the construction rather than a
/// defect: the Lebedev rules are O_h-invariant and this molecule's mirror
/// plane is not a grid plane, so the two atoms do not see point sets that are
/// exact mirror images — only sets that integrate the same function to within
/// the rule's accuracy. The reference's own dense (200,590) volumes differ by
/// 4.0e-14 / 4.4e-15 on the same pair for the same reason.
const TOL_MIRROR_REL: f64 = 1e-10;
/// THE EXACTNESS ANCHOR's bar: the one-atom WEIGHT-ONE moment `∫ ρ r³` on
/// ferric's production volume grid against the reference's
/// `volume_dense_grid`, which computes the same un-weighted moment on a
/// (200,590) grid. Measured max over {H, C, O} x {cc-pVDZ, def2-SVP}: 8.1e-13
/// relative — the floating-point floor, at 75 radial points, because the
/// integrand of a spherical free atom is exactly what a radial-times-Lebedev
/// rule integrates. Bar ~12x that.
const TOL_VOL_FREE_ATOM_WEIGHT_ONE_REL: f64 = 1e-11;
/// How much of a free atom's volume the 1e-12 Hirshfeld weight floor is
/// allowed to remove. Measured: 1.2e-3 (H, whose Slater ρ⁰ crosses 1e-12 at
/// ~6.6 Bohr, inside the region `r³` weights up), 3.4e-10 (C), 7.0e-11 (O).
/// Bar ~8x the H value. This is a DEFINITION difference from the reference,
/// which applies no floor at all; it is bounded here rather than asserted to
/// zero so that a floor that started eating a percent of the volume would
/// fail.
const TOL_VOL_FREE_ATOM_FLOOR_LOSS: f64 = 1e-2;
/// The bounding-box lattice's error on the SAME un-floored free-atom moment,
/// over the whole {H, C, O} x {cc-pVDZ, def2-SVP} set (from the reference's
/// `volume_weight_one_lattice` vs `volume_dense_grid`).
///
/// The control is a SET-level one — "the lattice misses the anchor SOMEWHERE
/// on the set" — not a per-atom one, because oxygen's lattice moment happens
/// to land 1.1e-5 relative from the dense value while hydrogen's is 8.6e-4
/// and carbon's 3.2e-4 / 4.1e-4. Stating it as a set bound keeps the control
/// reachable without pretending O fails as loudly as H. The bar is 1e-4,
/// below H's 8.6e-4 (8.6x margin) and above O's 1.1e-5, so the control is
/// reachable but not vacuous.
const LATTICE_FREE_ATOM_MIN_WORST_REL: f64 = 1e-4;
const TOL_E_SCF: f64 = 1e-10;
const TOL_Q_CHAIN: f64 = 1e-6;
const TOL_VOL_CHAIN_REL: f64 = 1e-6;
const TOL_ANCHOR: f64 = 5e-5;
/// A control must move a quantity by at least this multiple of its bar.
const MUST_MISS: f64 = 1000.0;
const DENSITY_KICK: f64 = 1e-3;
/// Slater vs SCF proatom on the real molecules: measured 0.23–0.72 e.
const SWAP_MIN: f64 = 0.1;
/// Slater proatom on the HeH promolecule: measured 0.104 e.
const ANCHOR_SLATER_MIN: f64 = 0.01;

/// The production proatom radii: 0.05·k Bohr, k = 1..600.
fn proatom_radii() -> Vec<f64> {
    scf_proatom_radii()
}

/// The production free-atom multiplicity, for the elements this row covers.
fn gs_mult(z: i32) -> usize {
    assert!(
        matches!(z, 1 | 2 | 6 | 8),
        "no reference free atom for Z={z} in this row"
    );
    proatom_ground_state_mult(z)
}

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

/// Load a reference JSON. Missing or unparsable is a HARD failure.
fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(REF_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_hirshfeld.py — a missing reference is a failure, never a skip",
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

/// Table difference relative to the table's largest value (the tail is
/// ~1e-30, where a pointwise relative difference measures nothing).
fn table_diff(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    max_abs_diff(a, b) / scale
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn check(ctx: &str, what: &str, d: f64, tol: f64) {
    eprintln!("{ctx}: {what:<34} max|d| {d:.2e} (tol {tol:.0e})");
    assert!(d < tol, "{ctx}: {what}: {d:.2e} >= {tol:.0e}");
}

fn scf_config() -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    }
}

/// Elementwise overlap check: the reference's PySCF overlap, permuted to
/// ferric's AO order, must equal ferric's own.
fn check_overlap(prep: &PreparedBasis, r: &Value, ptr: &str, ctx: &str) {
    let s = ferric_integrals::oneelectron::overlap(prep);
    let s_ref = mat(r, ptr, ctx);
    assert_eq!(
        s.dim(),
        s_ref.dim(),
        "{ctx}: AO count differs from the reference's"
    );
    let ds = (&s - &s_ref).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    check(ctx, "overlap (AO order)", ds, TOL_S);
}

/// Load `testdata/molecules/validation/<system>.xyz`, check geometry and AO order.
fn load(
    system: &str,
    basis_name: &str,
    r: &Value,
    ctx: &str,
) -> (Molecule, BasisSet, PreparedBasis) {
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    if let Some(enuc_ref) = r.pointer("/nuclear_repulsion").and_then(Value::as_f64) {
        let enuc = mol.nuclear_repulsion();
        assert!(
            (enuc - enuc_ref).abs() < TOL_ENUC,
            "{ctx}: nuclear repulsion {enuc:.12} vs reference {enuc_ref:.12} — geometry/unit mismatch"
        );
    }
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    check_overlap(&prep, r, "/overlap_ferric_order", ctx);
    (mol, bs, prep)
}

/// The free-atom SCF `scf_proatom_provider` runs (HF config): RHF for a
/// singlet, else UHF with MOM armed after iteration 5, rebuilt here because the
/// provider does not return the free-atom energy. Returns (energy, total
/// density).
fn ferric_free_atom(z: i32, bs: &BasisSet) -> (f64, Array2<f64>, PreparedBasis) {
    let sym = z_to_symbol(z).unwrap_or("X");
    let amol = Molecule::parse_xyz(&format!("1\n{sym}\n{sym} 0 0 0\n"), 0, gs_mult(z)).unwrap();
    let aprep = PreparedBasis::new(&amol, bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &aprep).unwrap();
    let ctx = ParallelContext::default();
    let mut cfg = scf_config();
    let res = if gs_mult(z) == 1 {
        solve_rhf(&ctx, &amol, &aprep, op, &bounds, &cfg)
    } else {
        cfg.mom_after_iter = 5;
        solve_uhf(&ctx, &amol, &aprep, &bounds, &cfg)
    }
    .unwrap_or_else(|e| panic!("free atom Z={z}: SCF failed: {e:?}"));
    assert!(res.converged, "free atom Z={z}: ferric SCF not converged");
    (res.energy, res.density_total().to_owned(), aprep)
}

/// Per element: same-density table, then ferric's own free atom → table.
/// Returns (reference tables, ferric's own tables, ferric's free-atom densities).
#[allow(clippy::type_complexity)]
fn proatoms(
    r: &Value,
    bs: &BasisSet,
    zs: &[i32],
    ctx: &str,
) -> (
    BTreeMap<i32, RadialProatom>,
    BTreeMap<i32, RadialProatom>,
    BTreeMap<i32, Array2<f64>>,
) {
    let radii = proatom_radii();
    let (mut t_ref, mut t_own, mut d_own) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    for &z in zs {
        let actx = format!("{ctx} free Z={z}");
        let base = format!("/free_atoms/{z}");
        let step = num(r, &format!("{base}/proatom_radii_step_bohr"), &actx);
        assert_eq!(
            step, 0.05,
            "{actx}: reference table step is not ferric-cli's 0.05 Bohr"
        );
        let rho_ref = vec1(r, &format!("{base}/proatom_rho"), &actx);
        assert_eq!(rho_ref.len(), radii.len(), "{actx}: table length");

        let (e_own, dens_own, aprep) = ferric_free_atom(z, bs);
        check_overlap(&aprep, r, &format!("{base}/overlap_ferric_order"), &actx);

        // (a) same density: the spherical averaging alone.
        let d_ref = mat(r, &format!("{base}/density_total_ferric_order"), &actx);
        let same = spherically_averaged_proatom(z, bs, &d_ref, &radii).unwrap();
        check(
            &actx,
            "proatom table same-D (rel max)",
            table_diff(same.rho(), &rho_ref),
            TOL_TABLE_SAME,
        );

        // (b) ferric's own free atom: state first, then the table.
        check(
            &actx,
            "free-atom SCF energy",
            (e_own - num(r, &format!("{base}/scf/energy"), &actx)).abs(),
            TOL_E_ATOM,
        );
        let own = spherically_averaged_proatom(z, bs, &dens_own, &radii).unwrap();
        check(
            &actx,
            "proatom table own SCF (rel max)",
            table_diff(own.rho(), &rho_ref),
            TOL_TABLE_CHAIN,
        );

        t_ref.insert(
            z,
            RadialProatom::new(radii.clone(), rho_ref).expect("reference proatom table"),
        );
        t_own.insert(z, own);
        d_own.insert(z, dens_own);
    }
    (t_ref, t_own, d_own)
}

fn distinct_z(mol: &Molecule) -> Vec<i32> {
    let mut zs: Vec<i32> = mol.atoms.iter().map(|a| a.z).collect();
    zs.sort_unstable();
    zs.dedup();
    zs
}

/// `Σ_g w_g ρ(r_g)` on the SAME production Becke–Lebedev grid
/// [`atomic_effective_volumes_hirshfeld`] integrates its volumes on: the
/// electron-count anchor of that quadrature. Written out here rather than read
/// from the reference because the point is to measure FERRIC's grid, and
/// because a quadrature that cannot integrate ρ to N_e cannot be trusted with
/// ρ r³ either.
fn n_electrons_on_volume_grid(mol: &Molecule, bs: &BasisSet, d: &Array2<f64>) -> f64 {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::grid::build_atomic_grid;
    let grid = build_atomic_grid(mol, &hirshfeld_volume_grid_config());
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(mol, bs, &points).expect("chi on the Becke grid");
    let d_chi = d.dot(&chi);
    let nbf = chi.nrows();
    let mut n = 0.0_f64;
    for (g, gp) in grid.iter().enumerate() {
        let mut rho = 0.0_f64;
        for mu in 0..nbf {
            rho += chi[(mu, g)] * d_chi[(mu, g)];
        }
        n += gp.weight * rho;
    }
    n
}

/// `Σ_g h³ ρ(r_g)` on the OLD bounding-box lattice — the negative control's
/// half of [`n_electrons_on_volume_grid`]. Rebuilt here point for point, same
/// as `scripts/validation/gen_hirshfeld.py` does, so the control measures the
/// lattice rather than quoting a stored number.
fn n_electrons_on_lattice(mol: &Molecule, bs: &BasisSet, d: &Array2<f64>) -> f64 {
    use ferric_integrals::ao_grid::eval_basis_on_grid;
    let grid = hirshfeld_volume_grid(mol);
    let dv = grid.step_x[0] * grid.step_y[1] * grid.step_z[2];
    let chi = eval_basis_on_grid(mol, bs, &grid).expect("chi on the lattice");
    let d_chi = d.dot(&chi);
    let nbf = chi.nrows();
    let npts = grid.n_x * grid.n_y * grid.n_z;
    let mut n = 0.0_f64;
    for g in 0..npts {
        let mut rho = 0.0_f64;
        for mu in 0..nbf {
            rho += chi[(mu, g)] * d_chi[(mu, g)];
        }
        n += dv * rho;
    }
    n
}

/// One molecule × basis. Returns ferric's full-chain charges.
fn check_molecule(system: &str, basis_name: &str) -> Vec<f64> {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    let (mol, bs, prep) = load(system, basis_name, &r, &ctx);
    let zs = distinct_z(&mol);
    let (t_ref, _, _) = proatoms(&r, &bs, &zs, &ctx);

    let ref_provider = |z: i32, q: i32| -> Option<RadialProatom> {
        if q != 0 {
            return None;
        }
        t_ref.get(&z).cloned()
    };
    let p_ref: &ProatomProvider = &ref_provider;
    // The production provider (ferric-cli's and the Python default).
    let prov_ctx = ParallelContext::default();
    let prov_cfg = scf_config();
    let production = scf_proatom_provider(&prov_ctx, &bs, Operator::coulomb(), &prov_cfg);
    let p_own: &ProatomProvider = &production;

    let q_ref = vec1(&r, "/charges/scf_proatom", &ctx);
    let q_ref_slater = vec1(&r, "/charges/slater", &ctx);
    let q_dense = vec1(&r, "/charges/scf_proatom_dense_grid", &ctx);
    let q_fine = vec1(&r, "/charges/scf_proatom_fine_table_dense_grid", &ctx);
    let v_ref = vec1(&r, "/hirshfeld_volumes/scf_proatom_lattice", &ctx);
    let v_dense = vec1(&r, "/hirshfeld_volumes/scf_proatom_dense_grid", &ctx);

    // --- Same density, same proatoms.
    let d_ref = mat(&r, "/density_total_ferric_order", &ctx);
    let q1 = hirshfeld_charges(&mol, &bs, &d_ref, Some(p_ref)).unwrap();
    check(
        &ctx,
        "charges same-D, SCF proatom",
        max_abs_diff(&q1, &q_ref),
        TOL_Q_SAME,
    );
    check(
        &ctx,
        "sum of charges (SCF proatom)",
        q1.iter().sum::<f64>().abs(),
        TOL_SUM,
    );
    let q1s = hirshfeld_charges(&mol, &bs, &d_ref, None).unwrap();
    check(
        &ctx,
        "charges same-D, Slater proatom",
        max_abs_diff(&q1s, &q_ref_slater),
        TOL_Q_SAME,
    );
    check(
        &ctx,
        "sum of charges (Slater)",
        q1s.iter().sum::<f64>().abs(),
        TOL_SUM,
    );
    check(
        &ctx,
        "charges vs dense-grid reference",
        max_abs_diff(&q1, &q_dense),
        TOL_Q_GRID,
    );
    eprintln!(
        "{ctx}: [measurement] 0.05 vs 0.005 Bohr proatom table (dense grid) max|d| {:.2e} e",
        max_abs_diff(&q_dense, &q_fine)
    );

    // --- Volumes, production quadrature (atom-centred Becke-Lebedev).
    //
    // ASSERTED against the DENSE Becke reference, not against the lattice
    // reference `v_ref`: the lattice is the less accurate quadrature (its
    // nodes land on nuclei), so pinning the new numbers to it would pin the
    // very error this row exists to bound. `v_ref` is still used below, as
    // the pin on the lattice entry point that MBD@rsSCS integrates on.
    let v1 = atomic_effective_volumes_hirshfeld(&mol, &bs, &d_ref, Some(p_ref)).unwrap();
    let v1_vs_dense = max_rel_diff(&v1, &v_dense);
    eprintln!(
        "{ctx}: [measurement] per-atom volume rel vs dense Becke: {:?}",
        v1.iter()
            .zip(&v_dense)
            .map(|(a, b)| format!("{:.2e}", (a - b) / b))
            .collect::<Vec<_>>()
    );
    check(
        &ctx,
        "volumes same-D vs dense Becke (rel)",
        v1_vs_dense,
        TOL_VOL_VS_DENSE_REL,
    );
    // The quadrature's own electron-count anchor: it must integrate ρ itself.
    let n_grid = n_electrons_on_volume_grid(&mol, &bs, &d_ref);
    check(
        &ctx,
        "electrons on the volume grid",
        (n_grid - mol.nelec() as f64).abs(),
        TOL_NELEC_GRID,
    );
    // Mirror symmetry: CH3OH's two out-of-plane methyl hydrogens (indices 3
    // and 4, z = ±0.890 A) are exact mirror images, so their volumes must be
    // equal to the floating-point floor. On a bounding-box lattice they are
    // not: the lattice is not invariant under the molecular point group.
    if system == "ch3oh" {
        let (a, b) = (v1[3], v1[4]);
        let rel = (a - b).abs() / b.abs();
        eprintln!("{ctx}: mirror H volumes {a:.12} vs {b:.12} (rel {rel:.2e})");
        assert!(
            rel < TOL_MIRROR_REL,
            "{ctx}: symmetry-equivalent H volumes differ by {rel:.2e} >= {TOL_MIRROR_REL:.0e}              relative — the volume quadrature does not carry the molecular point group"
        );
    }

    // --- The lattice entry point, still used by MBD@rsSCS: unchanged, so its
    // original same-density pin stays at the floating-point bar.
    let v1_lat = atomic_effective_volumes_hirshfeld_on_grid(
        &mol,
        &bs,
        &d_ref,
        Some(p_ref),
        &hirshfeld_volume_grid(&mol),
    )
    .unwrap();
    check(
        &ctx,
        "lattice volumes same-D (rel)",
        max_rel_diff(&v1_lat, &v_ref),
        TOL_VOL_SAME_REL,
    );

    // --- NEGATIVE CONTROL. The bounding-box lattice must MISS both anchors
    // the Becke grid passes, by a margin that leaves no doubt the assertions
    // above are measuring the quadrature and not a bar that any grid clears.
    let lat_vs_dense = max_rel_diff(&v1_lat, &v_dense);
    let n_lat = n_electrons_on_lattice(&mol, &bs, &d_ref);
    let dn_lat = (n_lat - mol.nelec() as f64).abs();
    eprintln!(
        "{ctx}: control lattice vs dense Becke {lat_vs_dense:.2e} rel (Becke {v1_vs_dense:.2e});          lattice electrons off by {dn_lat:.2e} e (Becke {:.2e} e)",
        (n_grid - mol.nelec() as f64).abs()
    );
    assert!(
        dn_lat > TOL_NELEC_GRID,
        "{ctx}: the bounding-box lattice integrates ρ to within {dn_lat:.2e} e of N_e, inside          the {TOL_NELEC_GRID:.0e} bar — the electron-count anchor cannot tell the two          quadratures apart, so it is not a measurement"
    );
    if system == "ch3oh" {
        let rel = (v1_lat[3] - v1_lat[4]).abs() / v1_lat[4].abs();
        eprintln!("{ctx}: control lattice mirror H volumes differ by {rel:.2e} rel");
        assert!(
            rel > TOL_MIRROR_REL,
            "{ctx}: the bounding-box lattice also gives mirror-equal H volumes              ({rel:.2e} < {TOL_MIRROR_REL:.0e}) — the symmetry assertion cannot fail"
        );
    }

    // --- Controls.
    let nao = prep.nbasis();
    let d_kick = &d_ref + &(Array2::<f64>::eye(nao) * DENSITY_KICK);
    let qk = hirshfeld_charges(&mol, &bs, &d_kick, Some(p_ref)).unwrap();
    let moved = max_abs_diff(&qk, &q1);
    eprintln!("{ctx}: control kick moves the charges by {moved:.2e}");
    assert!(
        moved > MUST_MISS * TOL_Q_SAME,
        "{ctx}: a {DENSITY_KICK:.0e} density kick moved the charges by only {moved:.2e} — \
         hirshfeld_charges does not respond to the density it is given"
    );
    let swap = max_abs_diff(&q1s, &q_ref);
    eprintln!("{ctx}: control Slater vs SCF proatom max|d| {swap:.2e} e");
    assert!(
        swap > SWAP_MIN,
        "{ctx}: the Slater-proatom charges are within {SWAP_MIN} e of the SCF-proatom \
         reference — the comparison does not resolve the proatom model"
    );
    if system == "co" {
        assert!(
            q1[0] > 0.0 && q1[1] < 0.0,
            "{ctx}: SCF-proatom Hirshfeld CO must be C(+) O(-); got C {:+.4} O {:+.4}",
            q1[0],
            q1[1]
        );
    }
    eprintln!(
        "{ctx}: [measurement] charges SCF proatom {:?} | Slater {:?}",
        q1.iter()
            .map(|x| (x * 1e4).round() / 1e4)
            .collect::<Vec<_>>(),
        q1s.iter()
            .map(|x| (x * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );

    // --- Full chain: ferric's RHF and ferric's own free atoms.
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let res = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &prep,
        op,
        &bounds,
        &scf_config(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
    assert!(res.converged, "{ctx}: ferric RHF not converged");
    check(
        &ctx,
        "RHF energy",
        (res.energy - num(&r, "/scf/energy", &ctx)).abs(),
        TOL_E_SCF,
    );
    let q2 = hirshfeld_charges(&mol, &bs, res.density_total(), Some(p_own)).unwrap();
    check(
        &ctx,
        "charges full chain",
        max_abs_diff(&q2, &q_ref),
        TOL_Q_CHAIN,
    );
    let v2 =
        atomic_effective_volumes_hirshfeld(&mol, &bs, res.density_total(), Some(p_own)).unwrap();
    // Full chain (ferric's own RHF and free atoms) against the dense Becke
    // reference: this bound carries the SCF and proatom chain on top of the
    // quadrature, so it is the quadrature bar, not the 1e-6 same-density one.
    check(
        &ctx,
        "volumes full chain vs dense Becke (rel)",
        max_rel_diff(&v2, &v_dense),
        TOL_VOL_VS_DENSE_REL,
    );
    // The chain's own contribution, isolated: same quadrature, ferric's SCF
    // and proatoms vs the reference density and tables.
    check(
        &ctx,
        "volumes full chain vs same-D (rel)",
        max_rel_diff(&v2, &v1),
        TOL_VOL_CHAIN_REL,
    );
    q2
}

/// Two-input requirement: the full-chain charges must RESPOND to the basis.
fn molecule_row(system: &str) {
    let q = [
        check_molecule(system, BASES[0]),
        check_molecule(system, BASES[1]),
    ];
    for (i, b_self) in BASES.iter().enumerate() {
        let b_other = BASES[1 - i];
        let other = vec1(&reference(system, b_other), "/charges/scf_proatom", system);
        let d = max_abs_diff(&q[i], &other);
        eprintln!("{system}: ferric {b_self} charges vs {b_other} reference max|d| {d:.2e}");
        assert!(
            d > 100.0 * TOL_Q_CHAIN,
            "{system}: ferric {b_self} charges are within {:.0e} of the {b_other} reference — \
             the comparison does not resolve the basis",
            100.0 * TOL_Q_CHAIN
        );
    }
}

/// Exactness anchor: a promolecule of spherical atoms has zero charges.
fn anchor_row(basis_name: &str) {
    let r = reference("heh", basis_name);
    let ctx = format!("heh-promolecule/{basis_name}");
    let (mol, bs, prep) = load("heh", basis_name, &r, &ctx);
    let zs = distinct_z(&mol);
    let (t_ref, t_own, d_own) = proatoms(&r, &bs, &zs, &ctx);
    let ref_provider = |z: i32, q: i32| if q == 0 { t_ref.get(&z).cloned() } else { None };
    let own_provider = |z: i32, q: i32| if q == 0 { t_own.get(&z).cloned() } else { None };
    let p_ref: &ProatomProvider = &ref_provider;
    let p_own: &ProatomProvider = &own_provider;

    // Same (reference) promolecule density and tables.
    let d_ref = mat(&r, "/density_total_ferric_order", &ctx);
    let q1 = hirshfeld_charges(&mol, &bs, &d_ref, Some(p_ref)).unwrap();
    check(
        &ctx,
        "anchor charges same-D vs numpy",
        max_abs_diff(&q1, &vec1(&r, "/charges/ferric_grid/charges", &ctx)),
        TOL_Q_SAME,
    );
    check(
        &ctx,
        "anchor |q| (reference atoms)",
        max_abs(&q1),
        TOL_ANCHOR,
    );

    // The anchor can fail: the Slater proatom on the same promolecule.
    let qs = hirshfeld_charges(&mol, &bs, &d_ref, None).unwrap();
    eprintln!(
        "{ctx}: control Slater proatom on the promolecule |q| {:.2e}",
        max_abs(&qs)
    );
    assert!(
        max_abs(&qs) > ANCHOR_SLATER_MIN,
        "{ctx}: the Slater proatom also gives |q| < {ANCHOR_SLATER_MIN} on the promolecule — \
         the anchor cannot distinguish proatom models"
    );

    // ferric's own free atoms: D = He block (+) H block, ferric's AO order is
    // atom-major, so the blocks stack in atom order.
    let nao = prep.nbasis();
    let mut d = Array2::<f64>::zeros((nao, nao));
    let mut off = 0usize;
    for atom in &mol.atoms {
        let blk = &d_own[&atom.z];
        let n = blk.nrows();
        d.slice_mut(s![off..off + n, off..off + n]).assign(blk);
        off += n;
    }
    assert_eq!(
        off, nao,
        "{ctx}: free-atom blocks do not tile the HeH AO space"
    );
    let q2 = hirshfeld_charges(&mol, &bs, &d, Some(p_own)).unwrap();
    check(
        &ctx,
        "anchor |q| (ferric's free atoms)",
        max_abs(&q2),
        TOL_ANCHOR,
    );
}

/// `Σ_g w_g ρ r³` about atom 0 on the production volume grid with NO
/// Hirshfeld weight — exactly what the reference's `volume_dense_grid`
/// computes (`gen_hirshfeld.py`: `np.sum(density_on(amol, adm, pts) * rr**3 *
/// wg)`, with no weight factor at all).
///
/// This is the like-for-like comparison for the free-atom row, and it is a
/// SECOND, INDEPENDENT construction of the same integral — written out here
/// in the test rather than reusing the library loop, so agreement is evidence
/// about the quadrature and not a tautology.
fn weight_one_volume_on_the_production_grid(mol: &Molecule, bs: &BasisSet, d: &Array2<f64>) -> f64 {
    use ferric_dft::ao_grid::eval_basis_on_points;
    use ferric_dft::grid::build_atomic_grid;
    let grid = build_atomic_grid(mol, &hirshfeld_volume_grid_config());
    let points: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(mol, bs, &points).expect("chi");
    let d_chi = d.dot(&chi);
    let nbf = chi.nrows();
    let pos = [mol.atoms[0].x, mol.atoms[0].y, mol.atoms[0].zpos];
    let mut acc = 0.0_f64;
    for (g, gp) in grid.iter().enumerate() {
        let mut rho = 0.0_f64;
        for mu in 0..nbf {
            rho += chi[(mu, g)] * d_chi[(mu, g)];
        }
        let dx = gp.xyz[0] - pos[0];
        let dy = gp.xyz[1] - pos[1];
        let dz = gp.xyz[2] - pos[2];
        acc += gp.weight * rho * (dx * dx + dy * dy + dz * dz).powf(1.5);
    }
    acc
}

/// EXACTNESS ANCHOR on a real density, plus the free-atom TS denominator
/// ferric-cli divides by (`atomic_effective_volumes_hirshfeld` with `None` on
/// a single atom).
///
/// # The anchor
///
/// One atom is the trivial limit of the Hirshfeld partition: `w^A =
/// ρ⁰_A/(ρ⁰_A + 1e-12)` is 1 wherever ρ⁰ is above the floor, so the volume
/// reduces to the plain moment `∫ ρ r³` with no partition left in it. The
/// reference's `volume_dense_grid` is that moment on a dense (200,590)
/// Becke-Lebedev grid WITH NO WEIGHT AT ALL, so the quantity that must equal
/// it is the weight-one integral, which it does to 1e-12 —
/// [`TOL_VOL_FREE_ATOM_WEIGHT_ONE_REL`].
///
/// # The 1e-12 weight floor is NOT quadrature error
///
/// The production `None` volume applies `ρ⁰/(ρ⁰ + 1e-12)`, and for hydrogen
/// the single-Slater ρ⁰ (ξ = 2.1167 Bohr⁻¹) drops below 1e-12 at about
/// 6.6 Bohr, where the `r³` weight still has real mass. The floor therefore
/// removes 1.2e-3 of H's volume — and that is a definition difference from
/// the reference, not an integration error. Measured here in BOTH places it
/// can be seen, so neither can be mistaken for the other:
///
/// * weight-one vs `volume_dense_grid`: 1e-12 (the quadrature), and
/// * floored vs weight-one: the floor loss, bounded by
///   [`TOL_VOL_FREE_ATOM_FLOOR_LOSS`]. The reference's own
///   `measurements/floor_loss_rel` is printed beside it but NOT asserted
///   against: it was measured on the lattice, which covers the region beyond
///   the ρ⁰ = 1e-12 shell differently, so the two are not the same number
///   (1.19e-3 here vs 8.48e-4 there for H).
///
/// Carbon and oxygen have no such loss (3.4e-10 and 7.0e-11): their ρ⁰ stays
/// above the floor far enough out that `r³` has nothing left to weight.
///
/// Returns `(weight_one_rel, floor_loss_rel, lattice_rel)` for the set-level
/// control in the caller.
fn free_atom_volume_row(symbol_lc: &str, basis_name: &str) -> (f64, f64, f64) {
    let system = format!("{symbol_lc}_atom");
    let r = reference(&format!("{system}_volume"), basis_name);
    let ctx = format!("{system}/{basis_name}");
    let (mol, bs, _prep) = load(&system, basis_name, &r, &ctx);
    let d_ref = mat(&r, "/density_total_ferric_order", &ctx);
    let dense = num(&r, "/volume_dense_grid", &ctx);

    // THE ANCHOR.
    let w1 = weight_one_volume_on_the_production_grid(&mol, &bs, &d_ref);
    let w1_rel = (w1 - dense).abs() / dense;
    check(
        &ctx,
        "free-atom weight-1 moment vs dense",
        w1_rel,
        TOL_VOL_FREE_ATOM_WEIGHT_ONE_REL,
    );

    // The production value, and the floor that separates it from the anchor.
    let v = atomic_effective_volumes_hirshfeld(&mol, &bs, &d_ref, None).unwrap();
    let floor_loss = (w1 - v[0]) / w1;
    assert!(
        (0.0..TOL_VOL_FREE_ATOM_FLOOR_LOSS).contains(&floor_loss),
        "{ctx}: the 1e-12 weight floor removes {floor_loss:.2e} of the free-atom volume, \
         outside [0, {TOL_VOL_FREE_ATOM_FLOOR_LOSS:.0e}) — a floor loss this large is no \
         longer a tail effect"
    );
    // MEASUREMENT, not an assertion: the reference measured the same floor on
    // the LATTICE. The two numbers are NOT expected to agree closely — the
    // floor cuts whatever each grid places beyond the ρ⁰ = 1e-12 shell
    // (~6.6 Bohr for H), and the lattice is a 12-Bohr cube while the Becke
    // radial rule reaches 13.6 Bohr, so they cover that region differently.
    // Measured H: 1.19e-3 here vs 8.48e-4 on the lattice. Printed so the
    // difference is on the record rather than asserted as a property it is
    // not.
    let ref_floor = num(&r, "/measurements/floor_loss_rel", &ctx);
    eprintln!(
        "{ctx}: weight-1 {w1:.9} vs dense {dense:.9} ({w1_rel:.2e} rel); floored {:.9}; \
         floor loss {floor_loss:.3e} ([measurement] reference, on the lattice: \
         {ref_floor:.3e})",
        v[0]
    );

    // The lattice entry point MBD@rsSCS still uses, on its own lattice: its
    // same-density pin is unchanged, at the floating-point bar.
    let v_lat = atomic_effective_volumes_hirshfeld_on_grid(
        &mol,
        &bs,
        &d_ref,
        None,
        &hirshfeld_volume_grid(&mol),
    )
    .unwrap();
    check(
        &ctx,
        "free-atom lattice volume same-D (rel)",
        max_rel_diff(&v_lat, &[num(&r, "/volume_ferric_none_lattice", &ctx)]),
        TOL_VOL_SAME_REL,
    );
    // The lattice's own error on the un-floored moment, from the reference's
    // two lattice numbers: this is what the quadratures must be separated by.
    let lattice_rel = (num(&r, "/volume_weight_one_lattice", &ctx) - dense).abs() / dense;

    (w1_rel, floor_loss, lattice_rel)
}

#[test]
#[ignore = "validation: Hirshfeld"]
fn hirshfeld_free_atom_volumes_vs_numpy() {
    let mut worst_w1 = 0.0_f64;
    let mut worst_floor = 0.0_f64;
    let mut worst_lattice = 0.0_f64;
    for b in BASES {
        for sym in ["h", "c", "o"] {
            let (w1, floor, lat) = free_atom_volume_row(sym, b);
            worst_w1 = worst_w1.max(w1);
            worst_floor = worst_floor.max(floor);
            worst_lattice = worst_lattice.max(lat);
        }
    }
    // NEGATIVE CONTROL, at SET level. On a one-atom molecule the nucleus sits
    // exactly on a bounding-box lattice node by construction, so the lattice
    // must miss the anchor the Becke grid clears — but only SOMEWHERE on the
    // set, not for every atom: oxygen's lattice volume lands 1.1e-5 relative
    // from the dense value, which is far outside the 1e-12 anchor bar but is
    // not itself a dramatic failure. Hydrogen is the one that fails loudly.
    // Both quantities are the UN-FLOORED moment, so the floor plays no part in
    // this comparison.
    eprintln!(
        "free-atom volumes: worst weight-1 vs dense {worst_w1:.2e} rel (bar \
         {TOL_VOL_FREE_ATOM_WEIGHT_ONE_REL:.0e}); worst lattice weight-1 vs dense \
         {worst_lattice:.2e} rel; worst 1e-12 floor loss {worst_floor:.2e}"
    );
    assert!(
        worst_lattice > LATTICE_FREE_ATOM_MIN_WORST_REL,
        "the bounding-box lattice's worst free-atom weight-1 error over the whole set is \
         {worst_lattice:.2e}, below {LATTICE_FREE_ATOM_MIN_WORST_REL:.0e} — the anchor above \
         cannot distinguish the two quadratures anywhere, so it is not a measurement"
    );
    assert!(
        worst_lattice > 1e6 * worst_w1,
        "the bounding-box lattice's worst free-atom error ({worst_lattice:.2e}) is within 1e6x \
         the production grid's ({worst_w1:.2e}) — the quadratures are not separated"
    );
}

#[test]
#[ignore = "validation: Hirshfeld"]
fn hirshfeld_h2o_vs_numpy() {
    molecule_row("h2o");
}

#[test]
#[ignore = "validation: Hirshfeld"]
fn hirshfeld_co_vs_numpy() {
    molecule_row("co");
}

#[test]
#[ignore = "validation: Hirshfeld"]
fn hirshfeld_ch3oh_vs_numpy() {
    molecule_row("ch3oh");
}

#[test]
#[ignore = "validation: Hirshfeld"]
fn hirshfeld_promolecule_anchor() {
    for b in BASES {
        anchor_row(b);
    }
}
