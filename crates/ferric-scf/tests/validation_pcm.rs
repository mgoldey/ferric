//! VALIDATION tier — VALIDATION.md row "IEF-PCM".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_pcm \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's IEF-PCM against PySCF 2.13 `scf.RHF(mol).PCM()` with
//! `method = "IEF-PCM"`, on water and NH3 at STO-3G and cc-pVDZ, ε = 78.4 and
//! 4.7. References come from `scripts/validation/gen_pcm.py` and live in
//! `testdata/reference/validation/pcm/<system>_<basis>.json`. Each file
//! carries PySCF's own SWIG cavity (points, normals, areas, sphere radii,
//! Gaussian exponents, switching values), so the SOLVER and the CAVITY are
//! tested separately:
//!
//! 1. **Solver** ([`solver_matches_pyscf_on_pyscf_cavity`]): PySCF's cavity
//!    goes through ferric's `build_s_d_kind` / `build_k_r` /
//!    `solve_pcm_charges` with PySCF's surface potential `v`. diag(S),
//!    diag(D), K·v, R·v, the symmetrized charges and `E_pcm = ½ q·v` must
//!    match PySCF. No basis, integral or SCF enters this comparison.
//! 2. **SCF on the shared cavity** (`scf_*_on_pyscf_cavity`): the cavity is
//!    injected through `PcmConfig::cavity` with `ProbeKind::GaussianSmeared`
//!    (PySCF's Gaussian-charge probes), and the self-consistent solvated
//!    total energy must match PySCF's.
//! 3. **ferric's own cavity** (`native_cavity_*`): the default `PcmConfig`
//!    (modified-Bondi radii, 110 points per sphere, point probes) against
//!    PySCF's solvation energy. With 1 and 2 passing, this difference is the
//!    cavity discretization (plus the ~1e-6 Ha probe convention), asserted
//!    to lie within ±2%.
//!
//! # Where ferric's own-cavity gap comes from
//!
//! Emulated in PySCF by swapping one ingredient at a time (water/STO-3G,
//! ε = 78.4, kcal/mol; PySCF −3.8228):
//!
//! | change from PySCF's setup | E_solv |
//! |---|---:|
//! | point probes instead of Gaussian | −3.8202 |
//! | + 110 points per sphere instead of 302 (= ferric's default; ferric measures −3.7995) | −3.7995 |
//! | + Bondi's original H radius, 1.20 Å | −3.3960 |
//!
//! The last line is why the cavity uses modified Bondi: the hydrogen radius
//! alone moves E_solv by ~11%.
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's solver is right: part 1 agrees to LAPACK precision (the S/D
//!   formulas are the same equations; only LU pivoting and summation order
//!   differ), and part 2 to the RHF integral/convergence floor.
//! * If the solver is wrong (operator ordering, a diagonal self-term, f(ε)):
//!   part 1 misses by 1e-8 Ha (a 1% D self-term error) to 3e-4 Ha (the
//!   transposed symmetrization), far above its bar.
//! * If the HARNESS is broken (tessera fields mis-mapped, a unit slip, a
//!   wrong file): diag(S)/diag(D) and K·v fail first, before any energy is
//!   compared; the nuclear-repulsion and vacuum-energy checks catch a
//!   geometry or basis mismatch before the PCM energy is blamed.
//!
//! # TOLERANCES
//!
//! Each const records its measured maximum. On PySCF's cavity the solver
//! agrees to 1e-17 and the SCF to 5e-12 Ha; ferric's own cavity is within
//! 0.07-0.61% of PySCF's.
//!
//! # NEGATIVE CONTROLS (asserted inside the tests)
//!
//! * ε swap: ferric's charges/energy at ε = 78.4 must MISS PySCF's ε = 4.7
//!   result (and vice versa) by ≫ the bar — solver level and SCF level.
//! * Mutated self-terms: diag(D) × 1.01 must move `E_pcm` by more than
//!   [`MUST_MISS_SOLVER`] (measured in numpy on these cavities: 5e-8 to
//!   7e-8 Ha), and the point-charge self-terms (`SdKind::PointCharge`) by
//!   far more (8e-5 to 9e-5 Ha).
//! * Probe convention (water/STO-3G): point probes on the SAME injected
//!   cavity must miss the SCF reference by more than the SCF bar (PySCF
//!   emulation: 4.0e-6 Ha at ε = 78.4), so part 2 resolves the probe choice.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_pcm::matrices::{build_k_r, build_s_d_kind, SdKind};
use ferric_pcm::solver::solve_pcm_charges;
use ferric_pcm::{PcmConfig, ProbeKind, Tessera};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array1, Array2};
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/pcm";
const SYSTEMS: [&str; 2] = ["water", "nh3"];
const BASES: [&str; 2] = ["sto-3g", "cc-pvdz"];
const EPS_KEYS: [&str; 2] = ["78.4", "4.7"];
const HARTREE_TO_KCAL: f64 = 627.5094740631;

/// diag(S)/diag(D), relative: closed-form per tessera. Measured 3.9e-16.
const TOL_DIAG_REL: f64 = 1e-12;
/// K·v and R·v, absolute: dense GEMVs (n ≤ 678). Measured max 7.3e-12
/// (NH3 / cc-pVDZ, K·v).
const TOL_KV: f64 = 1e-10;
/// Symmetrized charges from two LU solves. Measured max 1.3e-17.
const TOL_Q: f64 = 1e-11;
/// E_pcm = ½ q·v on the shared cavity. Measured max 3.5e-18 Ha; the 1%
/// diag(D) mutation (≥5.2e-8 Ha) stays 5000× above the bar.
const TOL_E_PCM: f64 = 1e-11;
/// A solver-level result that must NOT be reproduced: 1000x the energy bar
/// (1e-8 Ha, below the 5e-8 Ha diag(D)x1.01 shift).
const MUST_MISS_SOLVER: f64 = 1000.0 * TOL_E_PCM;

/// Vacuum RHF vs PySCF (harness check). Measured max 5.0e-12 Ha.
const TOL_E_VAC: f64 = 1e-10;
/// Solvated total energy on the shared cavity. Measured max 4.8e-12 Ha;
/// the bar still resolves the point-vs-Gaussian probe choice (4e-6 Ha).
const TOL_E_SCF: f64 = 1e-10;
/// An SCF-level result that must NOT be reproduced.
const MUST_MISS_SCF: f64 = 100.0 * TOL_E_SCF;

/// ferric's own cavity vs PySCF, relative error of E_solv (positive = ferric
/// weaker). With modified-Bondi radii on both sides the measured gaps are
/// 0.07% (NH3/cc-pVDZ) to 0.61% (water/STO-3G, both ε); what remains is
/// ferric's 110-point Lebedev spheres against PySCF's 302. A regression in
/// the cavity, radii or probe choice moves this by several percent.
const NATIVE_GAP_MIN: f64 = -0.02;
const NATIVE_GAP_MAX: f64 = 0.02;

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
             scripts/validation/gen_pcm.py — a missing reference is a failure, never a skip",
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

fn vec_f64(v: &Value, ptr: &str, ctx: &str) -> Vec<f64> {
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

fn vec3(v: &Value, ptr: &str, ctx: &str) -> Vec<[f64; 3]> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
        .iter()
        .map(|row| {
            let r = row.as_array().expect("3-vector");
            assert_eq!(r.len(), 3, "{ctx}: {ptr} row is not a 3-vector");
            [
                r[0].as_f64().unwrap(),
                r[1].as_f64().unwrap(),
                r[2].as_f64().unwrap(),
            ]
        })
        .collect()
}

/// PySCF's cavity as ferric tesserae. Field map (PySCF `gen_surface` →
/// `Tessera`): grid_coords → position, norm_vec → normal, area → area
/// (`w·R²·swf`, w summing to 4π), R_vdw → sphere_radius, charge_exp →
/// charge_exp, switch_fun → switch_fun, gslice_by_atom → atom_index.
fn pyscf_cavity(r: &Value, ctx: &str) -> Vec<Tessera> {
    let pos = vec3(r, "/cavity/position", ctx);
    let nrm = vec3(r, "/cavity/normal", ctx);
    let area = vec_f64(r, "/cavity/area", ctx);
    let rad = vec_f64(r, "/cavity/sphere_radius", ctx);
    let xi = vec_f64(r, "/cavity/charge_exp", ctx);
    let sw = vec_f64(r, "/cavity/switch_fun", ctx);
    let n = r["cavity"]["n_tesserae"].as_u64().expect("n_tesserae") as usize;
    for (name, len) in [
        ("position", pos.len()),
        ("normal", nrm.len()),
        ("area", area.len()),
        ("sphere_radius", rad.len()),
        ("charge_exp", xi.len()),
        ("switch_fun", sw.len()),
    ] {
        assert_eq!(
            len, n,
            "{ctx}: cavity/{name} has {len} entries, expected {n}"
        );
    }
    let mut atom_of = vec![usize::MAX; n];
    for (a, sl) in r["cavity"]["atom_slices"]
        .as_array()
        .expect("atom_slices")
        .iter()
        .enumerate()
    {
        let p0 = sl[0].as_u64().unwrap() as usize;
        let p1 = sl[1].as_u64().unwrap() as usize;
        for slot in &mut atom_of[p0..p1] {
            *slot = a;
        }
    }
    assert!(
        atom_of.iter().all(|&a| a != usize::MAX),
        "{ctx}: atom_slices do not cover every tessera"
    );
    (0..n)
        .map(|k| Tessera {
            position: pos[k],
            normal: nrm[k],
            area: area[k],
            sphere_radius: rad[k],
            atom_index: atom_of[k],
            charge_exp: xi[k],
            switch_fun: sw[k],
        })
        .collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<14} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_misses(ctx: &str, what: &str, got: f64, other: f64, must_miss: f64) {
    let d = (got - other).abs();
    eprintln!("{ctx}: control {what}: |d| {d:.2e} (must exceed {must_miss:.0e})");
    assert!(
        d > must_miss,
        "{ctx}: negative control '{what}' did not fail: |d| {d:.2e} <= {must_miss:.0e} — the \
         comparison cannot resolve this defect"
    );
}

/// ferric's `E_pcm` on the given S/D (after any mutation) at dielectric `eps`.
fn e_pcm_from(
    s: &Array2<f64>,
    d: &Array2<f64>,
    tess: &[Tessera],
    eps: f64,
    v: &Array1<f64>,
) -> (Array1<f64>, f64) {
    let (k, r, _f) = build_k_r(s, d, tess, eps).unwrap();
    let res = solve_pcm_charges(&k, &r, v).unwrap();
    (res.q, res.e_pcm)
}

/// Part 1 for one system × basis: both dielectrics, plus the controls.
fn solver_case(system: &str, basis_name: &str) {
    let r = reference(system, basis_name);
    let ctx0 = format!("{system}/{basis_name}/solver");
    let tess = pyscf_cavity(&r, &ctx0);
    let (s, d) = build_s_d_kind(&tess, SdKind::GaussianSmeared);

    for (i, key) in EPS_KEYS.iter().enumerate() {
        let ctx = format!("{ctx0}/eps={key}");
        let base = format!("/solvents/{key}");
        let eps = num(&r, &format!("{base}/epsilon"), &ctx);
        let v_ref = vec_f64(&r, &format!("{base}/v_grids"), &ctx);
        let v = Array1::from_vec(v_ref.clone());

        // Matrix fingerprints first: a mis-mapped tessera field fails here.
        let s_diag: Vec<f64> = (0..tess.len()).map(|k| s[(k, k)]).collect();
        let d_diag: Vec<f64> = (0..tess.len()).map(|k| d[(k, k)]).collect();
        let rel = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs() / y.abs().max(1e-300))
                .fold(0.0, f64::max)
        };
        let ds = rel(&s_diag, &vec_f64(&r, &format!("{base}/s_diag"), &ctx));
        let dd = rel(&d_diag, &vec_f64(&r, &format!("{base}/d_diag"), &ctx));
        eprintln!("{ctx}: diag(S) rel {ds:.2e}, diag(D) rel {dd:.2e} (tol {TOL_DIAG_REL:.0e})");
        assert!(
            ds < TOL_DIAG_REL,
            "{ctx}: diag(S) differs by {ds:.2e} relative"
        );
        assert!(
            dd < TOL_DIAG_REL,
            "{ctx}: diag(D) differs by {dd:.2e} relative"
        );

        let (k, rr, f_eps) = build_k_r(&s, &d, &tess, eps).unwrap();
        check_close(
            &ctx,
            "f(eps)",
            f_eps,
            num(&r, &format!("{base}/f_epsilon"), &ctx),
            1e-15,
        );
        let dkv = max_abs_diff(
            k.dot(&v).as_slice().unwrap(),
            &vec_f64(&r, &format!("{base}/k_dot_v"), &ctx),
        );
        let drv = max_abs_diff(
            rr.dot(&v).as_slice().unwrap(),
            &vec_f64(&r, &format!("{base}/r_dot_v"), &ctx),
        );
        eprintln!("{ctx}: K.v max|d| {dkv:.2e}, R.v max|d| {drv:.2e} (tol {TOL_KV:.0e})");
        assert!(dkv < TOL_KV, "{ctx}: K.v differs by {dkv:.2e}");
        assert!(drv < TOL_KV, "{ctx}: R.v differs by {drv:.2e}");

        let res = solve_pcm_charges(&k, &rr, &v).unwrap();
        let dq = max_abs_diff(
            res.q.as_slice().unwrap(),
            &vec_f64(&r, &format!("{base}/q_sym"), &ctx),
        );
        eprintln!("{ctx}: q_sym max|d| {dq:.2e} (tol {TOL_Q:.0e})");
        assert!(dq < TOL_Q, "{ctx}: symmetrized charges differ by {dq:.2e}");
        let e_ref = num(&r, &format!("{base}/e_pcm"), &ctx);
        check_close(&ctx, "E_pcm", res.e_pcm, e_ref, TOL_E_PCM);

        // CONTROL: the other dielectric's reference must be missed.
        let other = EPS_KEYS[1 - i];
        let e_other = num(&r, &format!("/solvents/{other}/e_pcm"), &ctx);
        check_misses(&ctx, "eps swap", res.e_pcm, e_other, MUST_MISS_SOLVER);

        // CONTROL: a 1% error in the D self-term must be resolved.
        let mut d_mut = d.clone();
        for kk in 0..tess.len() {
            d_mut[(kk, kk)] *= 1.01;
        }
        let (_q, e_mut) = e_pcm_from(&s, &d_mut, &tess, eps, &v);
        check_misses(&ctx, "diag(D) x 1.01", e_mut, e_ref, MUST_MISS_SOLVER);

        // CONTROL: point-charge self-terms (and off-diagonals) must be resolved.
        let (s_pc, d_pc) = build_s_d_kind(&tess, SdKind::PointCharge);
        let (_q, e_pc) = e_pcm_from(&s_pc, &d_pc, &tess, eps, &v);
        check_misses(&ctx, "SdKind::PointCharge", e_pc, e_ref, MUST_MISS_SOLVER);
    }
}

/// Part 1 across every system × basis (the cavity is basis-independent, but
/// `v` is not, so each file is its own input).
#[test]
#[ignore = "validation: IEF-PCM"]
fn solver_matches_pyscf_on_pyscf_cavity() {
    for system in SYSTEMS {
        for basis_name in BASES {
            solver_case(system, basis_name);
        }
    }
}

struct Loaded {
    r: Value,
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn load(system: &str, basis_name: &str) -> Loaded {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    let xyz = workspace_root()
        .join("testdata/molecules")
        .join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz(xyz.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let enuc = mol.nuclear_repulsion();
    check_close(
        &ctx,
        "E_nuc",
        enuc,
        num(&r, "/nuclear_repulsion", &ctx),
        1e-9,
    );
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    assert_eq!(
        prep.nbasis() as u64,
        r["nao"].as_u64().expect("nao"),
        "{ctx}: AO count differs from the reference's"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Loaded {
        r,
        mol,
        prep,
        bounds,
        ctx: ParallelContext::default(),
    }
}

fn scf_config(pcm: Option<PcmConfig>) -> RhfConfig {
    RhfConfig {
        max_iter: 200,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        pcm,
        ..Default::default()
    }
}

fn run(l: &Loaded, pcm: Option<PcmConfig>, ctx: &str) -> f64 {
    let res = solve_rhf(
        &l.ctx,
        &l.mol,
        &l.prep,
        Operator::coulomb(),
        &l.bounds,
        &scf_config(pcm),
    )
    .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
    assert!(res.converged, "{ctx}: SCF not converged");
    res.energy
}

fn injected(eps: f64, tess: &[Tessera], probe: ProbeKind) -> PcmConfig {
    PcmConfig {
        epsilon: eps,
        sd_kind: SdKind::GaussianSmeared,
        probe,
        cavity: Some(tess.to_vec()),
        ..PcmConfig::water()
    }
}

/// Part 2 for one system × basis: SCF with PySCF's cavity injected.
fn scf_case(system: &str, basis_name: &str, probe_control: bool) {
    let l = load(system, basis_name);
    let ctx0 = format!("{system}/{basis_name}/scf");
    let tess = pyscf_cavity(&l.r, &ctx0);

    let e_vac = run(&l, None, &ctx0);
    check_close(
        &ctx0,
        "E_vac",
        e_vac,
        num(&l.r, "/vacuum_energy", &ctx0),
        TOL_E_VAC,
    );

    for (i, key) in EPS_KEYS.iter().enumerate() {
        let ctx = format!("{ctx0}/eps={key}");
        let eps = num(&l.r, &format!("/solvents/{key}/epsilon"), &ctx);
        let e = run(
            &l,
            Some(injected(eps, &tess, ProbeKind::GaussianSmeared)),
            &ctx,
        );
        let e_ref = num(&l.r, &format!("/solvents/{key}/energy"), &ctx);
        check_close(&ctx, "E_total", e, e_ref, TOL_E_SCF);
        check_close(
            &ctx,
            "E_solv",
            e - e_vac,
            num(&l.r, &format!("/solvents/{key}/e_solv"), &ctx),
            TOL_E_SCF,
        );

        // CONTROL: the other dielectric's reference must be missed.
        let other = EPS_KEYS[1 - i];
        let e_other = num(&l.r, &format!("/solvents/{other}/energy"), &ctx);
        check_misses(&ctx, "eps swap", e, e_other, MUST_MISS_SCF);

        // CONTROL (one system): point probes on the same cavity must miss.
        if probe_control && i == 0 {
            let e_point = run(&l, Some(injected(eps, &tess, ProbeKind::Point)), &ctx);
            check_misses(&ctx, "ProbeKind::Point", e_point, e_ref, TOL_E_SCF);
        }
    }
}

#[test]
#[ignore = "validation: IEF-PCM"]
fn scf_water_sto3g_on_pyscf_cavity() {
    scf_case("water", "sto-3g", true);
}

#[test]
#[ignore = "validation: IEF-PCM"]
fn scf_water_ccpvdz_on_pyscf_cavity() {
    scf_case("water", "cc-pvdz", false);
}

#[test]
#[ignore = "validation: IEF-PCM"]
fn scf_nh3_sto3g_on_pyscf_cavity() {
    scf_case("nh3", "sto-3g", false);
}

#[test]
#[ignore = "validation: IEF-PCM"]
fn scf_nh3_ccpvdz_on_pyscf_cavity() {
    scf_case("nh3", "cc-pvdz", false);
}

/// Part 3 for one system × basis × ε: ferric's default cavity against
/// PySCF's solvation energy. Asserts the SIGN (ferric weaker: its larger H
/// spheres keep the dielectric farther from the solute) and the band.
fn native_case(system: &str, basis_name: &str, key: &str) {
    let l = load(system, basis_name);
    let ctx = format!("{system}/{basis_name}/native/eps={key}");
    let eps = num(&l.r, &format!("/solvents/{key}/epsilon"), &ctx);
    let e_vac = run(&l, None, &ctx);
    let e = run(
        &l,
        Some(PcmConfig {
            epsilon: eps,
            ..PcmConfig::water()
        }),
        &ctx,
    );
    let got = (e - e_vac) * HARTREE_TO_KCAL;
    let want = num(&l.r, &format!("/solvents/{key}/e_solv"), &ctx) * HARTREE_TO_KCAL;
    let gap = (got - want) / want.abs();
    eprintln!(
        "{ctx}: E_solv ferric-native {got:.4} kcal/mol, PySCF {want:.4} kcal/mol, \
         ferric weaker by {:.2}% (band [{:.0}%, {:.0}%])",
        gap * 100.0,
        NATIVE_GAP_MIN * 100.0,
        NATIVE_GAP_MAX * 100.0
    );
    assert!(got < 0.0, "{ctx}: solvation must stabilize; got {got:.4}");
    assert!(
        (NATIVE_GAP_MIN..=NATIVE_GAP_MAX).contains(&gap),
        "{ctx}: ferric-native E_solv {got:.4} vs PySCF {want:.4} kcal/mol: gap {:.2}% outside \
         [{:.0}%, {:.0}%]",
        gap * 100.0,
        NATIVE_GAP_MIN * 100.0,
        NATIVE_GAP_MAX * 100.0
    );
}

#[test]
#[ignore = "validation: IEF-PCM"]
fn native_cavity_water_gap_is_in_band() {
    native_case("water", "sto-3g", "78.4");
    native_case("water", "sto-3g", "4.7");
    native_case("water", "cc-pvdz", "78.4");
}

#[test]
#[ignore = "validation: IEF-PCM"]
fn native_cavity_nh3_gap_is_in_band() {
    native_case("nh3", "sto-3g", "78.4");
    native_case("nh3", "cc-pvdz", "78.4");
}
