//! VALIDATION tier — VALIDATION.md row "cDFT-ET coupling (Wu–Van Voorhis)".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_cdft_et \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # The reference, and why it is not the same quantity as ferric's H_ab
//!
//! References: `scripts/validation/gen_cdft_et.py` ->
//! `testdata/reference/validation/cdft_et/he2p_r<R>_<basis>.json` (NWChem 7.2.2
//! `cdft ... pop becke` + `et`; inputs in `scripts/validation/nwchem/cdft_et/`).
//! System: He₂⁺ (charge +1, doublet) at R = 2.50 / 3.00 / 3.50 Å × def2-SVP and
//! aug-cc-pVDZ (ferric's bundled JSON fed to NWChem). Two diabats per geometry,
//! constrained UKS/PBE: A = Becke population of He1 equal to 1.0 e (hole on
//! atom 0), B = the mirror image (hole on atom 1). PBE because NWChem builds its
//! Becke weight operator only on an XC grid, so `xc hfexch` + cdft aborts (see
//! `gen_cdft.py`); that is also why this row is NOT ferric's existing UHF He₂⁺
//! test (`cdft_coupling.rs`) re-run.
//!
//! NWChem's `et` module (src/etrans/et_calc.F, et_fock.F) computes the
//! Farazdel–Dupuis "direct" coupling between the two UHF/UKS determinants:
//!
//! ```text
//! S_RP  = det(M_α) det(M_β),  M_σ = C_B,occᵀ S C_A,occ          (signed)
//! H(RP) = ⟨A|T + V_ne + 1/r12|B⟩   generalized Slater–Condon, HF-form J − K of
//!         the transition density; NO XC functional, NO constraint potential
//! H(RR), H(PP) = the SCF (here PBE) energies stored in the movecs, minus E_nuc
//! V(RP) = | H(RP) − S_RP·(H(RR)+H(PP))/2 | / (1 − S_RP²)
//! ```
//!
//! ferric's `coupling_hab` applies the SAME final orthogonalization, but its raw
//! element is the Wu–Van Voorhis approximation built from (E, λ, ⟨A|w|B⟩) with
//! no two-electron transition integrals. So:
//!
//! * the STATE INGREDIENTS (E_A, E_B, λ_A, λ_B, |S_AB|) are the same quantities
//!   in both codes and are compared directly;
//! * NWChem's V(RP) is reproduced EXACTLY by evaluating its formula with ferric's
//!   own integrals — first on NWChem's determinants (isolates the kernel), then
//!   on ferric's determinants (end to end);
//! * ferric's Wu–VV |H_ab| is compared to V(RP) only through the log-slope
//!   d ln|H|/dR between neighbouring R, which a constant prefactor cannot move.
//!   With PBE determinants V(RP) itself mixes a HF-form off-diagonal with PBE
//!   diagonals, so it is a well-defined function of the determinants, not a
//!   "more correct" coupling; a residual difference in the VALUE of |H| is by
//!   construction and is only reported.
//!
//! The generator reproduces every NWChem `et` number from the movecs files in
//! PySCF (explicit-inverse Slater–Condon) before writing, and stores the
//! full-precision signed S_RP (NWChem prints 3 significant figures) and the
//! occupied MO coefficients.
//!
//! # Exactness anchors and identities (NWChem-independent, run first)
//!
//! 1. TRANSITION-DENSITY KERNEL, trivial limit: ⟨A|H|A·Q⟩ for an occupied-space
//!    rotation Q equals det(Q)·E_HF[D_A] computed from the ordinary density.
//! 2. WU–VV, vacuous constraint: with λ_A = λ_B = 0 both the textbook Wu–VV
//!    element and ferric's reduce to ½(E_A+E_B)S_AB, so H_ab = 0 EXACTLY.
//! 3. BECKE PARTITION OF UNITY: ⟨A|w_He1|B⟩ + ⟨A|w_He2|B⟩ = N_e·S_AB (to grid
//!    quadrature of the AO overlap), and by mirror symmetry the two terms are
//!    equal. On He₂⁺ this makes the textbook Wu–VV coupling a closed form,
//!    H_ab = λ S (N − N_e/2)/(1 − S²) — no W matrix needed.
//! 4. CONSTRAINT-OFFSET INVARIANCE (`wu_vv_coupling_is_invariant_under_constraint_offset`):
//!    the constrained problem (w, N) is identical to (w − (N/N_e)·1̂, 0) — same
//!    states, same λ, same E — so a correct H_ab must not change. The textbook
//!    Wu–VV element ⟨A|H|B⟩ ≈ F_B S − λ_B⟨A|w_B|B⟩ with F_B = E_B + λ_B N_B is
//!    invariant; the E-only form E_B S − λ_B⟨A|w_B|B⟩ is not (it moves by
//!    ½(λ_A N_A + λ_B N_B) S/(1 − S²)).
//!
//! # The raw element: why ferric now uses F = E + λN
//!
//! `coupling_hab` originally used the E-only element
//! `h_raw = ½[(E_b S − λ_b⟨a|W_b|b⟩) + …]` with W the raw Becke population
//! operator. By identity 3 that form gives −λ S (N_e/2)/(1 − S²) on symmetric
//! He₂⁺, EXACTLY (N_e/2)/(N_e/2 − N) = 3× the textbook value, and it fails
//! identity 4. Measured with this file (kernel test, NWChem determinants, all 12
//! points): ferric/textbook = 3.000000, e.g. −2.006551924109e−3 vs
//! −6.688505865637e−4 Ha at aug-cc-pVDZ 3.50 Å. Against NWChem's direct
//! |V(RP)| the textbook form is within 1.4–6% at every point, the E-only form
//! about 3× high. `coupling_hab` now implements the textbook (F) form, with
//! N_X taken as the state's own ⟨X|w_X|X⟩; the E-only element is kept here
//! only as a negative control. The CP2K manual quotes the E-only form with E
//! "including constraint terms" and does not say whether its W carries the
//! target.
//!
//! BLIND SPOT OF THE LOG-SLOPE: on a symmetric dimer the E-only and textbook
//! forms differ by an R-INDEPENDENT factor of exactly 3, so they have IDENTICAL
//! d ln|H|/dR. The slope comparison therefore cannot see this defect (its
//! artifact and physics predictions coincide); only the value-level assertions
//! and identity 4 can. Measured from the NWChem ingredients (300x974 grid):
//!
//! | basis | R (Å) | slope NWChem V | slope Wu–VV (either form) | slope |S_AB| |
//! |---|---|---:|---:|---:|
//! | def2-SVP | 2.50→3.00 | −2.508 | −2.483 | −2.554 |
//! | def2-SVP | 3.00→3.50 | −2.803 | −2.738 | −2.781 |
//! | aug-cc-pVDZ | 2.50→3.00 | −2.257 | −2.235 | −2.218 |
//! | aug-cc-pVDZ | 3.00→3.50 | −2.223 | −2.222 | −2.219 |
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If ferric is right: NWChem's determinants are orthonormal under ferric's
//!   overlap (≤1e-12, harness); ferric's integrals reproduce NWChem's S_RP, H1,
//!   H2, V(RP) on those determinants to integral precision; ferric's own diabats
//!   reproduce NWChem's E − E_unc and λ to the W-quadrature floor already
//!   measured by the cDFT row (≤2e−7 Ha, ≤1.2e−6 in λ on LiH); |S_AB| and V(RP)
//!   then follow to the same relative floor; log-slopes agree.
//! * If the harness is broken (AO order/normalization, units, wrong file):
//!   orthonormality of NWChem's MOs under ferric's S fails first (a p-shell
//!   permutation breaks it along the bond axis), as do the E_nuc/AO-count checks.
//!   The stored MOs are in ferric/NWChem AO order (each atom's shells in BSE-file
//!   order: s,s,p,s,p for He aug-cc-pVDZ). PySCF regroups by l; the generator's
//!   recompute first ran WITHOUT the row permutation and missed H1(RP) by
//!   9e-3 Ha — the real size of an AO-order artifact on this system.
//! * If the KERNEL is wrong (missing exchange of the transition density, a
//!   sign/transpose error in P = B M⁻¹ Aᵀ, the S-scaling): H2 or V(RP) on
//!   NWChem's own determinants misses NWChem by the size of the dropped term
//!   (measured, see NEGATIVE CONTROLS), independent of any SCF difference.
//! * If the DIABATS differ (wrong state, symmetry-broken unconstrained ref,
//!   λ-tolerance stop): E_A ≠ E_B or λ_A ≠ λ_B (mirror identity), the population
//!   misses the target, or λ misses NWChem by ≫ the cDFT-row floor. S_AB is
//!   exponentially sensitive to the hole localization, so |S_AB| is the most
//!   sensitive end-to-end witness.
//!
//! # TOLERANCES
//!
//! Each bar sits between a measured good floor and a measured defect; the
//! measured values are on the consts. Bars marked "derived" follow from an
//! exact identity and are confirmed by the first full run (see each const).
//!
//! # Running (release; keep OPENBLAS at 1)
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-scf --test validation_cdft_et \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//! The two NWChem-determinant tests run no SCF (seconds). The two end-to-end
//! tests run 3 unconstrained + 3 constrained solves per basis (+1 for the
//! mirror check at def2-SVP 2.50 Å).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::ao_grid::eval_basis_on_points;
use ferric_dft::cdft::{build_weight_matrix, Constraint, SpinChannel};
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::oneelectron::{hcore, overlap};
use ferric_integrals::operator::Operator;
use ferric_scf::cdft_coupling::{
    biorth_pairing, coupling_hab, cross_one_body, DiabaticState, HabResult,
};
use ferric_scf::cdft_driver::{solve_cdft_uhf, CdftResult};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::{s, Array2};
use ndarray_linalg::{Determinant, Inverse};
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/cdft_et";
const MOL_DIR: &str = "testdata/molecules/validation";
const SEPARATIONS: [&str; 3] = ["2.50", "3.00", "3.50"];
/// End-to-end points ferric's cDFT solve does not converge from NWChem's
/// converged lambda (8 outer iterations, level shift 0.3, 100 inner
/// iterations). def2-SVP 3.50 A fails in the INNER SCF, not the outer loop:
/// the outer loop reaches lambda = 2.4570634438 with |N - 1| = 9.6e-10 (the
/// tolerance is 1e-10), where the converged inner solves take 77-99 of their
/// 100 iterations, and every inner solve between there and 1e-9 above it hits
/// the cap unconverged (N = 1.000018, 1.00013, 1.99936). The outer loop backs
/// off toward the last converged lambda down to a step of 5e-10 and runs out
/// of outer iterations. Asserted to still fail; the kernel tests cover this
/// point on NWChem's own determinants.
const KNOWN_NONCONVERGING: &[(&str, &str)] = &[("def2-svp", "3.50")];
const BASES: [&str; 2] = ["def2-svp", "aug-cc-pvdz"];
/// Electrons on the hole-bearing He in each diabat (the generator's TARGET_N).
const TARGET_N: f64 = 1.0;
/// He₂⁺: 3 electrons, doublet.
const N_ELEC: f64 = 3.0;
const NOCC: (usize, usize) = (2, 1);

// Bars. "meas" = measured 2026-10-01 by this file on main @ libint 1e-20
// (release, both bases for the kernel test; def2-SVP R = 2.50/3.00 for the
// end-to-end rows, the rest of that run was stopped). "derived" = not yet
// measured by this file; the bar follows from an exact identity or from the
// measured bars it composes, and the first full run should confirm it.

// ── harness ──
const TOL_ENUC: f64 = 1e-9; // geometry/unit check, same bar as the cDFT row
/// meas max 1.7e-14 (all R, bases, grids); a p-order swap gives O(1).
const TOL_ORTHO: f64 = 1e-12;

// ── identities (NWChem-independent) ──
/// meas max 1.1e-13 Ha on |E_HF| ≈ 5.7 Ha.
const TOL_ANCHOR_ROTATION: f64 = 1e-11;
/// meas max 6.9e-18 (the expression is exactly zero).
const TOL_ANCHOR_LAMBDA0: f64 = 1e-15;
/// meas: partition rel ≤ 4.9e-7, mirror rel ≤ 7.4e-7, closed form vs textbook
/// rel ≤ 1.6e-7 (99x302 quadrature of the AO overlap). A wrong fragment or a
/// missing weight is O(1).
const TOL_PARTITION_REL: f64 = 5e-6;
/// derived: the F-form element is exactly invariant (round-off on |H| ≲ 2e-2
/// is ~1e-17). The superseded E-only element moves by ½(λ_A+λ_B)·N·S/(1−S²),
/// ≥ 9.4e-4 Ha on every point here (asserted as the negative control).
const TOL_OFFSET_INVARIANCE: f64 = 1e-12;
/// derived: `coupling_hab` and this file's textbook rebuild are the same
/// formula through different code; the E-only element missed it by
/// 9.4e-4..1.3e-2 Ha (meas, exactly 3x on every point).
const TOL_WUVV_FORM: f64 = 1e-12;

// ── kernel on NWChem's determinants vs NWChem `et` ──
/// meas |d| ≤ 6.9e-18 abs on |S| ≥ 3.8e-4 (vs the generator's full-precision
/// PySCF S_RP), i.e. ≤ 2e-14 relative.
const TOL_KERNEL_S_REL: f64 = 1e-9;
/// meas |d| ≤ 5.9e-11 Ha vs the NWChem printout, whose F18.10 format resolves
/// 1e-10: the bar is 2 printout quanta. The no-exchange mutant misses |V| by
/// ≥ 6.1e-5 (meas), 3e5x the bar.
const TOL_KERNEL_H: f64 = 2e-10;
/// meas |d| ≤ 4.5e-11 (same printout floor as H).
const TOL_KERNEL_V: f64 = 2e-10;

// ── end to end: ferric's diabats vs NWChem ──
/// meas ≤ 2.8e-8 Ha (def2-SVP and aug-cc-pVDZ, R = 2.50/3.00/3.50; cDFT row 1.0e-8).
const TOL_E_UNC: f64 = 1e-7;
/// meas ≤ 4.0e-7 Ha (def2-SVP 2.50/3.00; cDFT row ≤ 2e-7). aug-cc-pVDZ not
/// yet measured by this file.
const TOL_DE: f64 = 2e-6;
/// meas ≤ 6.6e-6 (def2-SVP 2.50/3.00 vs NWChem 300x974; vs 99x302 ≤ 6.4e-6;
/// the Python binding gives 8.1e-7 on aug-cc-pVDZ 2.50). cDFT row ≤ 1.1e-6.
const TOL_LAMBDA: f64 = 3e-5;
/// meas 1.4e-11 Ha / 6.6e-11 between two INDEPENDENT solves (def2-SVP 3.00).
const TOL_MIRROR_E: f64 = 1e-9;
const TOL_MIRROR_LAMBDA: f64 = 1e-8;
/// derived: 1 − |⟨B_mirror|B_solved⟩| for the reflected vs the independently
/// solved hole-on-He2 state (same state ⇒ ~orbital-difference², ≪ 1e-8).
const TOL_MIRROR_STATE: f64 = 1e-6;
/// Population residual; the driver runs at lambda_tol 1e-10 (cDFT row bar 1e-9).
const TOL_POP: f64 = 1e-9;
/// meas rel ≤ 1.3e-4 (def2-SVP 2.50/3.00), 6.4e-4 (aug-cc-pVDZ 3.50). S is
/// exponentially sensitive to the hole localization and the two codes' W
/// quadratures differ; a wrong state moves it by orders of magnitude.
const TOL_S_REL: f64 = 2e-3;
/// meas rel ≤ 4.1e-6 (def2-SVP 2.50/3.00), 2.05e-5 (aug-cc-pVDZ 3.00). The
/// defect it guards (the E-only coupling form) is a factor of 3.
const TOL_V_REL: f64 = 1e-4;
/// derived: a relative V error δ at both ends of a 0.5 Å step moves the slope
/// by ≤ 2δ/0.5; with δ = TOL_V_REL that is 8e-5 Å⁻¹.
const TOL_SLOPE_V: f64 = 1e-4;
/// By construction (Wu–VV vs the direct element), measured on the NWChem
/// ingredients: |Δslope| ≤ 0.065 Å⁻¹ (table above). A |S|² or λ² scaling
/// error doubles the slope (≈ 2.5 Å⁻¹ off).
const TOL_SLOPE_WUVV: f64 = 0.1;

// ── end-to-end solver settings (see "End-to-end solver settings" above) ──
const E2E_LEVEL_SHIFT: f64 = 0.3;
const E2E_INNER_MAX_ITER: usize = 100;
const E2E_MAX_OUTER: usize = 8;

// ───────────────────────────── harness ─────────────────────────────

/// Workspace root, found by walking up from the CWD (nextest sets the CWD to
/// the package dir); `CARGO_MANIFEST_DIR` is only a fallback.
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
fn reference(r_tag: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("he2p_r{r_tag}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_cdft_et.py — a missing reference is a failure, never a skip",
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

/// (nao × nocc) matrix from the reference's row-major list of rows.
fn mat(v: &Value, ptr: &str, ctx: &str) -> Array2<f64> {
    let rows = v
        .pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing"));
    let nr = rows.len();
    let nc = rows[0].as_array().expect("row").len();
    Array2::from_shape_fn((nr, nc), |(i, j)| {
        rows[i].as_array().expect("row")[j]
            .as_f64()
            .expect("number")
    })
}

fn check(ctx: &str, what: &str, got: f64, want: f64, tol: f64) -> f64 {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<34} ferric {got:+.12e} ref {want:+.12e} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} ferric {got:.12e} vs reference {want:.12e}: |d| {d:.2e} >= {tol:.0e}"
    );
    d
}

fn check_rel(ctx: &str, what: &str, got: f64, want: f64, tol: f64) -> f64 {
    let d = ((got - want) / want).abs();
    eprintln!("{ctx}: {what:<34} ferric {got:+.12e} ref {want:+.12e} rel {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} ferric {got:.12e} vs reference {want:.12e}: rel {d:.2e} >= {tol:.0e}"
    );
    d
}

fn grid_cfg() -> AtomicGridConfig {
    AtomicGridConfig {
        n_radial: 99,
        n_angular: 302,
        ..Default::default()
    }
}

/// Unconstrained UKS/PBE; exact J (no `df_j_aux`), as NWChem `direct`.
fn base_config() -> RhfConfig {
    RhfConfig {
        xc: Some("PBE".into()),
        dft_grid: Some(grid_cfg()),
        max_iter: 500,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    }
}

struct System {
    tag: String,
    r_ang: f64,
    mol: Molecule,
    bs: basis::BasisSet,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
    s_ao: Array2<f64>,
    h_ao: Array2<f64>,
    /// Dense (μν|λσ), flat ((μ n + ν) n + λ) n + σ. nbf ≤ 18 here.
    eri: Vec<f64>,
    /// Becke weight operators of He1 and He2 on the driver's grid.
    w: [Array2<f64>; 2],
}

fn load_system(r_tag: &str, basis_name: &str, r: &Value) -> System {
    let tag = format!("He2+ R={r_tag} {basis_name}");
    let xyz = workspace_root()
        .join(MOL_DIR)
        .join(format!("he2p_r{r_tag}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 1, 2)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    check(
        &tag,
        "E_nuc (geometry/units)",
        mol.nuclear_repulsion(),
        num(r, "/nuclear_repulsion", &tag),
        TOL_ENUC,
    );
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    assert_eq!(
        prep.nbasis(),
        r["nao"].as_u64().expect("nao") as usize,
        "{tag}: AO count differs from the reference's"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let s_ao = overlap(&prep);
    let h_ao = hcore(&prep);
    let eri = dense_eri(&prep);
    let grid = build_atomic_grid(&mol, &grid_cfg());
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(&mol, &bs, &pts).unwrap();
    let w = [
        build_weight_matrix(&mol, &grid, &chi, &[0]),
        build_weight_matrix(&mol, &grid, &chi, &[1]),
    ];
    System {
        tag,
        r_ang: r_tag.parse().unwrap(),
        mol,
        bs,
        prep,
        bounds,
        ctx: ParallelContext::default(),
        s_ao,
        h_ao,
        eri,
        w,
    }
}

/// Dense AO ERIs by brute force over every shell quartet (no symmetry, no
/// screening threshold beyond libint's precision). He₂⁺ has ≤ 10 shells.
fn dense_eri(prep: &PreparedBasis) -> Vec<f64> {
    let n = prep.nbasis();
    let nsh = prep.nshells();
    let dims = prep.shell_dims().to_vec();
    let offs = prep.shell_offsets().to_vec();
    let mut eng = Engine::new_2e(Operator::coulomb(), prep, 1e-14).expect("2e engine");
    let mut eri = vec![0.0; n * n * n * n];
    for s1 in 0..nsh {
        for s2 in 0..nsh {
            for s3 in 0..nsh {
                for s4 in 0..nsh {
                    let Some(q) = eng.compute_quartet(prep, s1, s2, s3, s4) else {
                        continue;
                    };
                    let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                    for a in 0..n1 {
                        for b in 0..n2 {
                            for c in 0..n3 {
                                for d in 0..n4 {
                                    let (m, nu, l, sg) =
                                        (offs[s1] + a, offs[s2] + b, offs[s3] + c, offs[s4] + d);
                                    eri[((m * n + nu) * n + l) * n + sg] =
                                        q[((a * n2 + b) * n3 + c) * n4 + d];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    eri
}

/// Occupied blocks [α, β] of one determinant, (nao × nocc_σ).
type Det = [Array2<f64>; 2];

fn det_from_ref(r: &Value, grid: &str, state: &str, ctx: &str) -> Det {
    [
        mat(r, &format!("/results/{grid}/{state}/mo_occ_alpha"), ctx),
        mat(r, &format!("/results/{grid}/{state}/mo_occ_beta"), ctx),
    ]
}

fn det_from_cdft(res: &CdftResult) -> Det {
    let cb = res.scf.mos_beta.as_ref().expect("UKS beta MOs");
    [
        res.scf.mos_alpha.slice(s![.., ..NOCC.0]).to_owned(),
        cb.slice(s![.., ..NOCC.1]).to_owned(),
    ]
}

fn max_abs(m: &Array2<f64>) -> f64 {
    m.iter().fold(0.0_f64, |a, v| a.max(v.abs()))
}

/// Coulomb-like contraction Σ (μν|λσ) X_νμ Y_σλ.
fn coul(eri: &[f64], n: usize, x: &Array2<f64>, y: &Array2<f64>) -> f64 {
    let mut acc = 0.0;
    for m in 0..n {
        for nu in 0..n {
            let xnm = x[(nu, m)];
            if xnm == 0.0 {
                continue;
            }
            for l in 0..n {
                for sg in 0..n {
                    acc += eri[((m * n + nu) * n + l) * n + sg] * xnm * y[(sg, l)];
                }
            }
        }
    }
    acc
}

/// Exchange-like contraction Σ (μν|λσ) X_σμ X_νλ.
fn exch(eri: &[f64], n: usize, x: &Array2<f64>) -> f64 {
    let mut acc = 0.0;
    for m in 0..n {
        for nu in 0..n {
            for l in 0..n {
                let xnl = x[(nu, l)];
                for sg in 0..n {
                    acc += eri[((m * n + nu) * n + l) * n + sg] * x[(sg, m)] * xnl;
                }
            }
        }
    }
    acc
}

/// NWChem `et` quantities evaluated with ferric's integrals.
#[derive(Debug, Clone, Copy)]
struct Direct {
    s: f64,
    h1: f64,
    h2: f64,
    h: f64,
    /// Signed V; |V| is NWChem's printed V(RP).
    v: f64,
}

/// ⟨A|H|B⟩ by the explicit-inverse generalized Slater–Condon rule:
/// P_σ = B_σ (A_σᵀ S B_σ)⁻¹ A_σᵀ, S_AB = Π_σ det(A_σᵀ S B_σ),
/// H1 = S_AB Σ_σ tr(h P_σ), H2 = S_AB · ½[J(P_tot,P_tot) − Σ_σ K(P_σ)].
/// This is an INDEPENDENT construction from both NWChem's SVD/cofactor path and
/// ferric's `cross_one_body`; it needs every singular value > 0 (true here).
/// `with_exchange = false` is the negative-control mutant.
fn direct_coupling(
    sys: &System,
    a: &Det,
    b: &Det,
    e_a_elec: f64,
    e_b_elec: f64,
    with_exchange: bool,
) -> Direct {
    let n = sys.prep.nbasis();
    let mut s_ab = 1.0;
    let mut ps: Vec<Array2<f64>> = Vec::new();
    for sp in 0..2 {
        let m = a[sp].t().dot(&sys.s_ao).dot(&b[sp]);
        s_ab *= m.det().expect("det of MO overlap");
        let minv = m
            .inv()
            .expect("MO overlap is singular — kernel needs all s_i > 0");
        ps.push(b[sp].dot(&minv).dot(&a[sp].t()));
    }
    let one: f64 = ps.iter().map(|p| (&sys.h_ao * &p.t()).sum()).sum();
    let pt = &ps[0] + &ps[1];
    let j = coul(&sys.eri, n, &pt, &pt);
    let k: f64 = if with_exchange {
        ps.iter().map(|p| exch(&sys.eri, n, p)).sum()
    } else {
        0.0
    };
    let h1 = s_ab * one;
    let h2 = s_ab * 0.5 * (j - k);
    let h = h1 + h2;
    let v = (h - s_ab * 0.5 * (e_a_elec + e_b_elec)) / (1.0 - s_ab * s_ab);
    Direct {
        s: s_ab,
        h1,
        h2,
        h,
        v,
    }
}

/// Electronic HF energy of a determinant from its ordinary density.
fn e_hf_elec(sys: &System, a: &Det) -> f64 {
    let n = sys.prep.nbasis();
    let d: Vec<Array2<f64>> = a.iter().map(|c| c.dot(&c.t())).collect();
    let dt = &d[0] + &d[1];
    let one = (&sys.h_ao * &dt).sum();
    let k: f64 = d.iter().map(|x| exch(&sys.eri, n, x)).sum();
    one + 0.5 * (coul(&sys.eri, n, &dt, &dt) - k)
}

/// ⟨S²⟩ of a UHF/UKS determinant.
fn s_squared(sys: &System, a: &Det) -> f64 {
    let (na, nb) = (a[0].ncols() as f64, a[1].ncols() as f64);
    let sz = 0.5 * (na - nb);
    let ov = a[0].t().dot(&sys.s_ao).dot(&a[1]);
    sz * (sz + 1.0) + nb - ov.iter().map(|x| x * x).sum::<f64>()
}

/// Square (nao × nao) MO matrix whose first columns are the occupied block —
/// `DiabaticState` slices `..nocc` off full coefficient matrices.
fn pad(c: &Array2<f64>, nao: usize) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((nao, nao));
    out.slice_mut(s![.., ..c.ncols()]).assign(c);
    out
}

/// Wu–VV couplings for one pair of diabats, both forms, plus the W elements.
struct WuVv {
    /// ferric's `coupling_hab`.
    ferric: HabResult,
    /// Textbook Wu–VV rebuilt in this file: F_X = E_X + λ_X N_X, N_X the
    /// state's own population ⟨X|w_X|X⟩ computed here from its MOs.
    textbook: f64,
    /// The superseded E-only element (F_X → E_X), kept as the negative control.
    e_only: f64,
    /// ⟨A|w_He1|B⟩, ⟨A|w_He2|B⟩ (ferric's unsigned-S convention).
    w_elems: [f64; 2],
}

/// `w_a`, `w_b`: each state's OWN constraint operator (A constrains He1, B He2).
#[allow(clippy::too_many_arguments)]
fn wu_vv(
    sys: &System,
    a: &Det,
    b: &Det,
    e_a: f64,
    e_b: f64,
    lam_a: f64,
    lam_b: f64,
    w_a: &Array2<f64>,
    w_b: &Array2<f64>,
) -> WuVv {
    let nao = sys.prep.nbasis();
    let (aa, ab, ba, bb) = (
        pad(&a[0], nao),
        pad(&a[1], nao),
        pad(&b[0], nao),
        pad(&b[1], nao),
    );
    let st_a = DiabaticState {
        c_a: &aa,
        c_b: &ab,
        nocc_a: NOCC.0,
        nocc_b: NOCC.1,
        energy: e_a,
        lambda: lam_a,
        w: w_a,
    };
    let st_b = DiabaticState {
        c_a: &ba,
        c_b: &bb,
        nocc_a: NOCC.0,
        nocc_b: NOCC.1,
        energy: e_b,
        lambda: lam_b,
        w: w_b,
    };
    let ferric = coupling_hab(&st_a, &st_b, &sys.s_ao);
    // Textbook raw element, rebuilt from the same pairing so only the F-vs-E
    // term differs (independent of `coupling_hab`'s internals beyond the
    // shared pairing kernel).
    let pa = biorth_pairing(&a[0], &b[0], &sys.s_ao);
    let pb = biorth_pairing(&a[1], &b[1], &sys.s_ao);
    let s_ab = pa.det_m * pb.det_m;
    let w1 = cross_one_body(&sys.w[0], &pa, &pb, s_ab);
    let w2 = cross_one_body(&sys.w[1], &pa, &pb, s_ab);
    let wa = cross_one_body(w_a, &pa, &pb, s_ab);
    let wb = cross_one_body(w_b, &pa, &pb, s_ab);
    let own = |d: &Det, w: &Array2<f64>| -> f64 {
        d.iter().map(|c| c.t().dot(w).dot(c).diag().sum()).sum()
    };
    let f_a = e_a + lam_a * own(a, w_a);
    let f_b = e_b + lam_b * own(b, w_b);
    let h_raw = 0.5 * ((f_a + f_b) * s_ab - lam_a * wa - lam_b * wb);
    let textbook = (h_raw - 0.5 * (e_a + e_b) * s_ab) / (1.0 - s_ab * s_ab);
    let h_raw_e = 0.5 * ((e_a + e_b) * s_ab - lam_a * wa - lam_b * wb);
    let e_only = (h_raw_e - 0.5 * (e_a + e_b) * s_ab) / (1.0 - s_ab * s_ab);
    WuVv {
        ferric,
        textbook,
        e_only,
        w_elems: [w1, w2],
    }
}

fn slope(r1: f64, h1: f64, r2: f64, h2: f64) -> f64 {
    (h2.abs().ln() - h1.abs().ln()) / (r2 - r1)
}

// ──────────────────────────────── tests ────────────────────────────────

/// Identities 1–3 and the kernel on NWChem's determinants vs NWChem `et`.
/// No SCF is run here: every determinant, energy and λ is NWChem's, so a
/// mismatch is the KERNEL or the harness, never the cDFT driver.
#[test]
#[ignore = "validation: cDFT-ET coupling (Wu–Van Voorhis)"]
fn et_kernel_on_nwchem_determinants_matches_nwchem() {
    for basis_name in BASES {
        for r_tag in SEPARATIONS {
            let r = reference(r_tag, basis_name);
            let sys = load_system(r_tag, basis_name, &r);
            let enuc = sys.mol.nuclear_repulsion();
            for grid in ["matched", "converged"] {
                let ctx = format!("{} [{grid} NWChem dets]", sys.tag);
                let a = det_from_ref(&r, grid, "state_a", &ctx);
                let b = det_from_ref(&r, grid, "state_b", &ctx);

                // ── harness: NWChem MOs orthonormal under ferric's S ──
                for (nm, d) in [("A", &a), ("B", &b)] {
                    for (sp, c) in d.iter().enumerate() {
                        let e = &c.t().dot(&sys.s_ao).dot(c) - &Array2::<f64>::eye(c.ncols());
                        let err = max_abs(&e);
                        eprintln!("{ctx}: orthonormality {nm} spin {sp}: {err:.2e}");
                        assert!(err < TOL_ORTHO, "{ctx}: {nm}/{sp} not orthonormal under ferric S ({err:.2e}) — AO order/normalization mismatch");
                    }
                }

                // ── ANCHOR 1: occupied-space rotation ──
                if grid == "converged" {
                    let th = 0.7_f64;
                    let q = ndarray::array![[th.cos(), th.sin()], [th.sin(), -th.cos()]]; // det −1
                    let aq: Det = [a[0].dot(&q), a[1].clone()];
                    let e_ref = e_hf_elec(&sys, &a);
                    let dd = direct_coupling(&sys, &a, &aq, e_ref, e_ref, true);
                    check(
                        &ctx,
                        "ANCHOR <A|H|AQ> = det(Q) E_HF",
                        dd.h,
                        -e_ref,
                        TOL_ANCHOR_ROTATION,
                    );
                    check(
                        &ctx,
                        "ANCHOR S(A,AQ) = det(Q)",
                        dd.s,
                        -1.0,
                        TOL_ANCHOR_ROTATION,
                    );
                }

                // ── kernel vs NWChem et ──
                let e_a = num(&r, &format!("/results/{grid}/state_a/energy"), &ctx);
                let e_b = num(&r, &format!("/results/{grid}/state_b/energy"), &ctx);
                let lam_a = num(&r, &format!("/results/{grid}/state_a/lambda"), &ctx);
                let lam_b = num(&r, &format!("/results/{grid}/state_b/lambda"), &ctx);
                let et = format!("/results/{grid}/et_nwchem");
                let dc = direct_coupling(&sys, &a, &b, e_a - enuc, e_b - enuc, true);
                check_rel(
                    &ctx,
                    "S_RP (full precision)",
                    dc.s,
                    num(
                        &r,
                        &format!("/results/{grid}/et_pyscf_recompute/s_rp"),
                        &ctx,
                    ),
                    TOL_KERNEL_S_REL,
                );
                check(
                    &ctx,
                    "H1(RP)",
                    dc.h1,
                    num(&r, &format!("{et}/h1_rp"), &ctx),
                    TOL_KERNEL_H,
                );
                check(
                    &ctx,
                    "H2(RP)",
                    dc.h2,
                    num(&r, &format!("{et}/h2_rp"), &ctx),
                    TOL_KERNEL_H,
                );
                check(
                    &ctx,
                    "H(RP)",
                    dc.h,
                    num(&r, &format!("{et}/h_rp"), &ctx),
                    TOL_KERNEL_H,
                );
                let v_ref = num(&r, &format!("{et}/v_rp_abs"), &ctx);
                let dv = check(&ctx, "|V(RP)|", dc.v.abs(), v_ref, TOL_KERNEL_V);

                // NEGATIVE CONTROL: dropping the transition-density exchange.
                let nk = direct_coupling(&sys, &a, &b, e_a - enuc, e_b - enuc, false);
                let miss = (nk.v.abs() - v_ref).abs();
                eprintln!("{ctx}: NEG CONTROL no-exchange |V| {:.10e} misses by {miss:.2e} (good {dv:.2e})", nk.v.abs());
                assert!(miss > 100.0 * TOL_KERNEL_V, "{ctx}: the exchange-free mutant also matches NWChem ({miss:.2e}) — the V bar cannot see K");

                // ── ANCHOR 2: Wu–VV at λ = 0 is exactly zero (both forms) ──
                let z = wu_vv(&sys, &a, &b, e_a, e_b, 0.0, 0.0, &sys.w[0], &sys.w[1]);
                eprintln!(
                    "{ctx}: ANCHOR lambda=0: ferric {:.2e} textbook {:.2e}",
                    z.ferric.h_ab, z.textbook
                );
                assert!(
                    z.ferric.h_ab.abs() < TOL_ANCHOR_LAMBDA0
                        && z.textbook.abs() < TOL_ANCHOR_LAMBDA0,
                    "{ctx}: Wu–VV at lambda = 0 is not zero"
                );

                // ── IDENTITY 3: partition of unity + mirror symmetry ──
                let wv = wu_vv(&sys, &a, &b, e_a, e_b, lam_a, lam_b, &sys.w[0], &sys.w[1]);
                let s_u = wv.ferric.s_ab; // unsigned, ferric convention
                let part = ((wv.w_elems[0] + wv.w_elems[1] - N_ELEC * s_u) / s_u).abs();
                let mirror = ((wv.w_elems[0] - wv.w_elems[1]) / s_u).abs();
                eprintln!("{ctx}: <A|w1|B> {:+.10e} <A|w2|B> {:+.10e} N_e S {:+.10e}: partition rel {part:.2e}, mirror rel {mirror:.2e}",
                    wv.w_elems[0], wv.w_elems[1], N_ELEC * s_u);
                assert!(
                    part < TOL_PARTITION_REL,
                    "{ctx}: partition-of-unity identity fails ({part:.2e})"
                );
                // Closed form of the textbook element in the symmetric case.
                let lam = 0.5 * (lam_a + lam_b);
                let closed = lam * s_u * (TARGET_N - 0.5 * N_ELEC) / (1.0 - s_u * s_u);
                eprintln!(
                    "{ctx}: |H| textbook {:.10e} (closed form {:.10e}) | ferric coupling_hab {:.10e} | ratio ferric/textbook {:.6} | NWChem |V| {v_ref:.10e}",
                    wv.textbook.abs(), closed.abs(), wv.ferric.h_ab.abs(), wv.ferric.h_ab / wv.textbook
                );
                assert!(
                    ((wv.textbook - closed) / closed).abs() < TOL_PARTITION_REL,
                    "{ctx}: textbook Wu–VV != its symmetric closed form"
                );
                check(
                    &ctx,
                    "coupling_hab vs textbook Wu–VV",
                    wv.ferric.h_ab,
                    wv.textbook,
                    TOL_WUVV_FORM,
                );
                // NEGATIVE CONTROL: the superseded E-only element must miss.
                let miss_e = (wv.e_only - wv.textbook).abs();
                eprintln!("{ctx}: NEG CONTROL E-only element {:+.10e} misses by {miss_e:.2e} (ratio {:.6})", wv.e_only, wv.e_only / wv.textbook);
                assert!(
                    miss_e > 1e6 * TOL_WUVV_FORM,
                    "{ctx}: the E-only element is indistinguishable from the textbook one"
                );
            }
        }
    }
}

/// Identity 4: the constrained problem is unchanged by w → w − (N/N_e)·1̂
/// with target N → 0 (same states, λ, E), so H_ab must not move. In the AO
/// basis the number operator is the overlap matrix S. Run on NWChem's
/// determinants and ingredients (no SCF). Negative control: the superseded
/// E-only raw element moves by exactly ½(λ_A+λ_B)·N·S/(1−S²) (≥ 9.4e-4 Ha
/// here), which is asserted so the bar is known to resolve it.
#[test]
#[ignore = "validation: cDFT-ET coupling (Wu–Van Voorhis)"]
fn wu_vv_coupling_is_invariant_under_constraint_offset() {
    for basis_name in BASES {
        for r_tag in SEPARATIONS {
            let r = reference(r_tag, basis_name);
            let sys = load_system(r_tag, basis_name, &r);
            let ctx = format!("{} [converged NWChem dets]", sys.tag);
            let g = "converged";
            let a = det_from_ref(&r, g, "state_a", &ctx);
            let b = det_from_ref(&r, g, "state_b", &ctx);
            let e_a = num(&r, &format!("/results/{g}/state_a/energy"), &ctx);
            let e_b = num(&r, &format!("/results/{g}/state_b/energy"), &ctx);
            let lam_a = num(&r, &format!("/results/{g}/state_a/lambda"), &ctx);
            let lam_b = num(&r, &format!("/results/{g}/state_b/lambda"), &ctx);
            let shift = &sys.s_ao * (TARGET_N / N_ELEC);
            let w0p = &sys.w[0] - &shift;
            let w1p = &sys.w[1] - &shift;
            let base = wu_vv(&sys, &a, &b, e_a, e_b, lam_a, lam_b, &sys.w[0], &sys.w[1]);
            let off = wu_vv(&sys, &a, &b, e_a, e_b, lam_a, lam_b, &w0p, &w1p);
            let d_text = (off.textbook - base.textbook).abs();
            let d_eonly = (off.e_only - base.e_only).abs();
            let s_u = base.ferric.s_ab;
            let predicted = 0.5 * (lam_a + lam_b) * TARGET_N * s_u / (1.0 - s_u * s_u);
            eprintln!(
                "{ctx}: offset invariance: textbook |dH| {d_text:.2e}; coupling_hab {:+.10e} -> {:+.10e} (|dH| {:.2e}); E-only |dH| {d_eonly:.2e} (predicted {:.2e})",
                base.ferric.h_ab, off.ferric.h_ab, (off.ferric.h_ab - base.ferric.h_ab).abs(), predicted.abs()
            );
            assert!(
                d_text < TOL_OFFSET_INVARIANCE,
                "{ctx}: textbook Wu–VV not offset-invariant ({d_text:.2e})"
            );
            check(
                &ctx,
                "coupling_hab offset invariance",
                off.ferric.h_ab,
                base.ferric.h_ab,
                TOL_OFFSET_INVARIANCE,
            );
            // NEGATIVE CONTROL / reachability: the superseded E-only element is
            // NOT invariant, by the predicted amount — so the bar can see it.
            assert!(d_eonly > 1e6 * TOL_OFFSET_INVARIANCE, "{ctx}: the E-only element is offset-invariant too ({d_eonly:.2e}); this test has no resolving power");
            assert!(
                ((d_eonly - predicted.abs()) / predicted).abs() < 1e-6,
                "{ctx}: E-only shift {d_eonly:.6e} != predicted {:.6e}",
                predicted.abs()
            );
        }
    }
}

/// The mirror σ: z → R − z maps He1 ↔ He2. ferric's AO order is atom-major
/// with identical shells on both He, so AO μ on atom 0 maps to μ + nbf/2 on
/// atom 1; Cartesian p components are (x, y, z) and only p_z changes sign.
fn mirror_det(sys: &System, a: &Det) -> Det {
    let n = sys.prep.nbasis();
    assert_eq!(
        n % 2,
        0,
        "{}: odd AO count for a homonuclear dimer",
        sys.tag
    );
    let half = n / 2;
    let mut sign = Vec::with_capacity(half);
    for sh in sys.bs.for_element(2).expect("He shells") {
        assert!(
            sh.l <= 1 && !sh.pure,
            "mirror map assumes Cartesian s/p shells"
        );
        if sh.l == 0 {
            sign.push(1.0);
        } else {
            sign.extend_from_slice(&[1.0, 1.0, -1.0]);
        }
    }
    assert_eq!(
        sign.len(),
        half,
        "{}: He shell list does not cover nbf/2",
        sys.tag
    );
    let refl = |c: &Array2<f64>| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros(c.dim());
        for mu in 0..n {
            let (atom, local) = (mu / half, mu % half);
            let to = (1 - atom) * half + local;
            out.row_mut(to).assign(&(&c.row(mu) * sign[local]));
        }
        out
    };
    [refl(&a[0]), refl(&a[1])]
}

/// Constrained UKS/PBE solve of the diabat with the hole on `frag`.
/// The end-to-end cDFT config for one diabat (hole on fragment `frag`).
fn diabat_config(frag: usize, lambda_init: f64) -> RhfConfig {
    RhfConfig {
        constraints: vec![Constraint {
            fragment: vec![frag],
            spin: SpinChannel::Total,
            target: TARGET_N,
        }],
        cdft_lambda_tol: 1e-10,
        cdft_lambda_init: Some(vec![lambda_init]),
        cdft_max_outer: E2E_MAX_OUTER,
        // Skipped on KS references anyway (no f_xc response kernel); off so
        // the run cannot pay for an eigensolve it will not use.
        cdft_stability_descent: false,
        level_shift: E2E_LEVEL_SHIFT,
        max_iter: E2E_INNER_MAX_ITER,
        ..base_config()
    }
}

fn solve_diabat(sys: &System, frag: usize, lambda_init: f64) -> CdftResult {
    let cfg = diabat_config(frag, lambda_init);
    let tag = &sys.tag;
    let res = solve_cdft_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bs, &sys.bounds, &cfg)
        .unwrap_or_else(|e| {
            panic!("{tag}: cDFT hole-on-{frag} from lambda0 = {lambda_init}: {e:?}")
        });
    assert!(
        res.scf.converged,
        "{tag}: inner SCF not converged (frag {frag})"
    );
    let pop = res.populations[0];
    assert!(
        (pop - TARGET_N).abs() < TOL_POP,
        "{tag}: frag {frag} population {pop:.12} misses {TARGET_N}"
    );
    // The start is NWChem's λ, so the root must actually have been SOLVED for,
    // not accepted at iteration 1 (the two codes' roots differ by ~1e-6).
    assert!(
        res.outer_iters > 1,
        "{tag}: outer loop accepted the starting lambda unchanged"
    );
    let dw = max_abs(&(&res.weight_matrices[0] - &sys.w[frag]));
    assert!(
        dw < 1e-12,
        "{tag}: driver W differs from the test's W by {dw:.2e}"
    );
    res
}

/// End to end: ferric's own constrained UKS/PBE diabats vs NWChem, the direct
/// coupling on ferric's determinants vs NWChem's V(RP), and log-slopes across
/// R = 2.50/3.00/3.50 Å (the plan's convention-robust comparison).
///
/// # End-to-end solver settings (why, measured 2026-10-01 on He₂⁺/PBE)
///
/// Measured with an outer loop that had only the Newton step and the
/// sign-change bracket (no backtracking from unconverged inner solves, no
/// rejection of a probe in another basin; see `ScalarStepper` in
/// cdft_driver.rs). The λ = 0 start has not been re-measured with those.
/// From the default λ = 0 start that loop failed or crawled: R = 2.50
/// aug-cc-pVDZ hole-on-1 did not converge in 60 outer iterations, and def2-SVP
/// took up to 29 outer iterations of mostly 500-iteration unconverged inner
/// solves. Traced (`FERRIC_CDFT_TRACE=1`) causes:
/// * λ = 0 is the SYMMETRIC delocalized state (N = 1.5); a 1e-3 FD step flips
///   the inner SCF into a localized basin, so the "Jacobian" is ±490 and Newton
///   crawls or walks λ the wrong way (to −0.11 at def2-SVP 3.00 Å);
/// * N(λ) is then a flat plateau (dN/dλ ≈ −0.006) ending at an over-
///   localization cliff (N → 0.01–0.06, E → −3 to −2 Ha) with the root at its
///   edge; the ±1-clamped step from the plateau lands on the cliff, whose inner
///   solves do not converge and so never tighten the bracket, and the loop
///   limit-cycled (def2-SVP 3.50 Å: λ = 2.6498 → 2.8291 → 2.6485 → …). With
///   backtracking that point instead stops on an inner-SCF failure; see
///   `KNOWN_NONCONVERGING`.
/// With only a 0.3 level shift and a 150-iteration inner cap (no λ start),
/// 2 of 5 points converged (def2-SVP 2.50 in 36 s, 3.00 in 273 s) and 3 failed.
///
/// So each diabat starts at NWChem's converged multiplier
/// (`cdft_lambda_init`). The result is still ferric's OWN root of c(λ) = 0 to
/// 1e-10 (`outer_iters > 1` is asserted), so λ and E remain a real
/// comparison; what the start does is select the SAME localized branch as the
/// reference, which the like-for-like rule requires anyway. It does NOT show
/// that ferric's outer loop finds this state unaided from λ = 0, which is
/// not this row's subject. The 0.3 Ha level shift changes the path, not the fixed point
/// (it acts on the virtual block and commutes with D at convergence): def2-SVP
/// 2.50 Å gave E = −4.856787855242, λ = 2.3217366412 both with and without it.
/// Hard caps: 8 outer × (1 + 1 FD) inner solves × 100 inner iterations, so a
/// non-converging point fails in minutes, not the ~30 min the first run took.
///
/// State B (hole on He2) is the mirror image of A, built by `mirror_det`
/// (E_B = E_A, λ_B = λ_A by symmetry) instead of a second solve; at one point
/// (def2-SVP 2.50 Å) B is ALSO solved independently and the mirror identity and
/// the state identity |⟨B_mirror|B_solved⟩| = 1 are asserted.
fn run_basis(basis_name: &str) {
    let mut rows: Vec<(f64, f64, f64, f64, f64)> = Vec::new(); // (R, |V_nw|, |V_ferric dets|, |H textbook|, |H coupling_hab|)
    for r_tag in SEPARATIONS {
        let r = reference(r_tag, basis_name);
        let sys = load_system(r_tag, basis_name, &r);
        let tag = sys.tag.clone();
        let enuc = sys.mol.nuclear_repulsion();
        let cv = "/results/converged";

        let unc = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &base_config())
            .unwrap_or_else(|e| panic!("{tag}: unconstrained UKS: {e:?}"));
        assert!(unc.converged, "{tag}: unconstrained UKS not converged");
        let e_unc = unc.energy;
        check(
            &tag,
            "E_unc vs NWChem 99x302",
            e_unc,
            num(&r, "/results/matched/unconstrained/energy", &tag),
            TOL_E_UNC,
        );

        let lam_ref = num(&r, &format!("{cv}/state_a/lambda"), &tag);
        if KNOWN_NONCONVERGING.contains(&(basis_name, r_tag)) {
            // Asserted, not skipped: if the driver learns to converge this point
            // the assert fails and the case should move back into the row.
            // Only a NON-CONVERGENCE counts as the known gap: any other failure
            // (population, W mismatch, a regression) must not pass silently.
            let attempt = solve_cdft_uhf(
                &sys.ctx,
                &sys.mol,
                &sys.prep,
                &sys.bs,
                &sys.bounds,
                &diabat_config(0, lam_ref),
            );
            let not_converged = match &attempt {
                Err(ferric_core::FerricError::Convergence(_)) => true,
                Err(e) => panic!("{tag}: known non-converging point failed differently: {e:?}"),
                Ok(r) => !r.scf.converged,
            };
            assert!(
                not_converged,
                "{tag}: listed in KNOWN_NONCONVERGING but now converges -- restore it"
            );
            eprintln!("{tag}: KNOWN driver gap -- outer loop does not converge from NWChem's lambda; skipped");
            continue;
        }
        let ra = solve_diabat(&sys, 0, lam_ref);
        let (e_a, lam_a) = (ra.scf.energy, ra.lambdas[0]);
        let a = det_from_cdft(&ra);
        let b = mirror_det(&sys, &a);
        let (e_b, lam_b) = (e_a, lam_a);
        eprintln!(
            "{tag}: ferric E_a {e_a:.12} lam_a {lam_a:+.10} (start {lam_ref:+.10}) outer {} <S2> {:.6}",
            ra.outer_iters,
            s_squared(&sys, &a)
        );

        // ── the reflected state is a valid determinant of the mirrored constraint ──
        for (sp, c) in b.iter().enumerate() {
            let err = max_abs(&(&c.t().dot(&sys.s_ao).dot(c) - &Array2::<f64>::eye(c.ncols())));
            assert!(
                err < TOL_ORTHO,
                "{tag}: mirrored B spin {sp} not orthonormal ({err:.2e}) — mirror map wrong"
            );
        }
        let pop = |d: &Det, w: &Array2<f64>| -> f64 {
            d.iter().map(|c| c.t().dot(w).dot(c).diag().sum()).sum()
        };
        check(
            &tag,
            "MIRROR N_He2[B] vs N_He1[A]",
            pop(&b, &sys.w[1]),
            pop(&a, &sys.w[0]),
            TOL_PARTITION_REL,
        );

        // ── one independent B solve: mirror identity + state identity ──
        if basis_name == "def2-svp" && r_tag == "2.50" {
            let rb = solve_diabat(&sys, 1, lam_ref);
            let b_solved = det_from_cdft(&rb);
            check(
                &tag,
                "MIRROR E_a vs E_b (independent)",
                e_a,
                rb.scf.energy,
                TOL_MIRROR_E,
            );
            check(
                &tag,
                "MIRROR lambda_a vs lambda_b",
                lam_a,
                rb.lambdas[0],
                TOL_MIRROR_LAMBDA,
            );
            let mut ov = 1.0;
            for sp in 0..2 {
                ov *= b[sp]
                    .t()
                    .dot(&sys.s_ao)
                    .dot(&b_solved[sp])
                    .det()
                    .expect("det");
            }
            let d_state = 1.0 - ov.abs();
            eprintln!("{tag}: 1 - |<B_mirror|B_solved>| = {d_state:.2e}");
            assert!(
                d_state < TOL_MIRROR_STATE,
                "{tag}: reflected B is not the solved B (1-|S| = {d_state:.2e})"
            );
        }

        // ── state ingredients vs NWChem (grid limit, as in the cDFT row) ──
        let e_unc_nw = num(&r, &format!("{cv}/unconstrained/energy"), &tag);
        check(
            &tag,
            "E_A - E_unc (300x974)",
            e_a - e_unc,
            num(&r, &format!("{cv}/state_a/energy"), &tag) - e_unc_nw,
            TOL_DE,
        );
        check(&tag, "lambda_A (300x974)", lam_a, lam_ref, TOL_LAMBDA);
        eprintln!(
            "{tag}: vs NWChem 99x302 (info): dlambda {:+.2e} dE {:+.2e}",
            lam_a - num(&r, "/results/matched/state_a/lambda", &tag),
            e_a - num(&r, "/results/matched/state_a/energy", &tag)
        );

        // ── direct (NWChem-definition) coupling on ferric's determinants ──
        let dc = direct_coupling(&sys, &a, &b, e_a - enuc, e_b - enuc, true);
        let s_nw = num(&r, &format!("{cv}/et_pyscf_recompute/s_rp"), &tag);
        let v_nw = num(&r, &format!("{cv}/et_nwchem/v_rp_abs"), &tag);
        check_rel(&tag, "|S_AB| vs NWChem", dc.s.abs(), s_nw.abs(), TOL_S_REL);
        check_rel(
            &tag,
            "|V(RP)| ferric dets vs NWChem",
            dc.v.abs(),
            v_nw,
            TOL_V_REL,
        );

        // ── Wu–VV on ferric's determinants ──
        let wv = wu_vv(&sys, &a, &b, e_a, e_b, lam_a, lam_b, &sys.w[0], &sys.w[1]);
        check(
            &tag,
            "coupling_hab vs textbook Wu–VV",
            wv.ferric.h_ab,
            wv.textbook,
            TOL_WUVV_FORM,
        );
        eprintln!(
            "{tag}: |H_ab| Wu–VV {:.10e} (E-only {:.10e}) | direct |V| {:.10e} | NWChem |V| {v_nw:.10e} | S_ab {:.6e}",
            wv.ferric.h_ab.abs(), wv.e_only.abs(), dc.v.abs(), wv.ferric.s_ab
        );
        rows.push((
            sys.r_ang,
            v_nw,
            dc.v.abs(),
            wv.textbook.abs(),
            wv.ferric.h_ab.abs(),
        ));
    }

    // ── log-slopes d ln|H| / dR between neighbouring R (Å⁻¹) ──
    for w in rows.windows(2) {
        let (r1, nw1, fd1, tb1, fc1) = w[0];
        let (r2, nw2, fd2, tb2, fc2) = w[1];
        let ctx = format!("He2+ {basis_name} slope {r1:.2}->{r2:.2} A");
        let s_nw = slope(r1, nw1, r2, nw2);
        eprintln!(
            "{ctx}: NWChem V {s_nw:+.6} | ferric direct {:+.6} | Wu–VV textbook {:+.6} | coupling_hab {:+.6}",
            slope(r1, fd1, r2, fd2), slope(r1, tb1, r2, tb2), slope(r1, fc1, r2, fc2)
        );
        check(
            &ctx,
            "d ln|V|/dR ferric direct vs NWChem",
            slope(r1, fd1, r2, fd2),
            s_nw,
            TOL_SLOPE_V,
        );
        check(
            &ctx,
            "d ln|H|/dR Wu–VV vs NWChem V",
            slope(r1, fc1, r2, fc2),
            s_nw,
            TOL_SLOPE_WUVV,
        );
        assert!(
            nw2 < nw1 && fd2 < fd1 && tb2 < tb1 && fc2 < fc1,
            "{ctx}: a coupling does not decay with R"
        );
    }
}

#[test]
#[ignore = "validation: cDFT-ET coupling (Wu–Van Voorhis)"]
fn he2_plus_def2_svp_diabats_and_coupling_match_nwchem() {
    run_basis("def2-svp");
}

#[test]
#[ignore = "validation: cDFT-ET coupling (Wu–Van Voorhis)"]
fn he2_plus_aug_cc_pvdz_diabats_and_coupling_match_nwchem() {
    run_basis("aug-cc-pvdz");
}
