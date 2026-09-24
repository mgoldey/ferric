//! VALIDATION tier — VALIDATION.md row "SCF stability".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-scf --test validation_scf_stability \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What ferric computes, and what it is compared with
//!
//! `ferric_scf::stability` implements two operators, both INTERNAL (real
//! rotations within one ansatz):
//!
//! | ferric | question | PySCF construction (dense, from its matvec) | factor |
//! |---|---|---|---:|
//! | `uhf_internal_stability`, HF | UHF → UHF | `newton_ah.gen_g_hop_uhf` | 1 |
//! | `uhf_internal_stability`, KS (`fxc` threaded by `solve_uhf`) | UKS → UKS | `gen_g_hop_uhf` on `dft.UKS` (f_xc via `gen_response`) | 1 |
//! | `rhf_internal_stability` | RHF → RHF (singlet) | `newton_ah.gen_g_hop_rhf` | 1/2 |
//! | `uhf_internal_stability` on UHF inputs built AT THE RHF POINT | RHF → UHF (triplet) ∪ RHF → RHF | `stability._gen_hop_rhf_external` `hop_rhf2uhf` ∪ `gen_g_hop_rhf`/2 | 1 |
//!
//! There is no dedicated RHF → UHF operator in ferric. The external check
//! here is the UHF Hessian evaluated at the RHF solution (C_α = C_β = C_RHF,
//! F_α = F_β = F_RHF), which block-diagonalizes EXACTLY into the singlet
//! (κ_β = κ_α) and triplet (κ_β = −κ_α) channels: in the triplet channel δJ
//! cancels and each spin keeps −K(δD_σ), which is PySCF's `hop_rhf2uhf`; in
//! the singlet channel it is `(A+B)`, ferric's RHF Hessian. Its λ_min is
//! therefore min(singlet, triplet), and the triplet spectrum is extracted
//! here by rotating the dense Hessian into the (κ, ±κ)/√2 basis.
//!
//! ## Normalization (read from both codes; proved numerically)
//!
//! * ferric `rhf_newton::hessian_matvec` = `(ε_a − ε_i)κ + [δJ − ½δK]_ai` on
//!   `δD = 2C(κ+κᵀ)Cᵀ`: the singlet `A+B`. PySCF `gen_g_hop_rhf` builds the
//!   same thing and returns it `* 2`, so **PySCF raw = 2 × ferric**. (PySCF's
//!   `rhf_internal` Davidson doubles it again, so its LOGGED eigenvalues are
//!   4 × ferric's.)
//! * ferric `uhf_newton::hessian_matvec` and PySCF `gen_g_hop_uhf` are the same
//!   expression, `(ε_a − ε_i)κ_σ + [J(δD_α+δD_β) − K(δD_σ)]_ai`, with no
//!   overall factor: **equal**.
//! * PySCF `hop_rhf2uhf` = `(ε_a − ε_i)κ − ½K(2C(κ+κᵀ)Cᵀ)`: **equal** to the
//!   triplet block defined above.
//!
//! The generator asserts spec(UHF Hessian at the RHF point) =
//! spec(`gen_g_hop_rhf`)/2 ∪ spec(`hop_rhf2uhf`) inside PySCF (measured
//! 4.6e-14 / 3.9e-14), and the references store every spectrum ALREADY in
//! ferric's convention (the raw PySCF singlet values are kept alongside, and
//! a test asserts ferric MISSES them, so a dropped factor of 2 fails).
//!
//! # Systems (6-31G, ferric's bundled JSON on both sides)
//!
//! * **N2⁺ UHF** — the known-hard state-selection case (`RhfConfig::
//!   scf_stability_descent` docs): from the default guess both codes reach a
//!   SADDLE (PySCF λ_min = −1.003e-1); following the instability reaches the
//!   stable minimum (λ_min = +1.0036e-1, a degenerate π pair). Both states are
//!   compared: the saddle is a negative control on a real open-shell state.
//! * **OH UHF (²Π)** — the β hole in one π orbital breaks the axial symmetry,
//!   so rotating it about the axis is an EXACT zero mode (Goldstone) of the
//!   Hessian: PySCF λ₀ = −7e-13. The correct verdict is MARGINAL, never
//!   STABLE; the first NONZERO eigenvalue (1.6348e-1) carries the information.
//! * **water RHF, r(OH) = 0.9572 Å** — internally and externally STABLE.
//! * **water RHF, r(OH) = 2.0 Å** — internally stable (singlet λ = +1.97e-2)
//!   but RHF → UHF UNSTABLE (triplet λ = −3.07e-1). This is the row's negative
//!   control: the RHF-internal verdict alone is STABLE here, so a checker that
//!   only asks the internal question passes a saddle.
//! * **NH2 UKS/PBE** — exact J (no density fitting) and the (75,110) unpruned
//!   Becke grid with Becke-1988 radii on both sides, so the KS Hessian is
//!   compared like-for-like (the KS-DFT row compares λ_min at RI-J, where the
//!   response/density fit mismatch floors it at ~9e-6).
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's Hessians are right: the lowest `N_COMPARE` eigenvalues of
//!   ferric's dense Hessian and ferric's Davidson λ_min agree with PySCF's
//!   dense spectrum at the SCF-convergence floor (first order in the density
//!   error), and every verdict matches.
//! * If a Hessian carries a wrong FACTOR (the 2 / 4 conventions above): every
//!   eigenvalue misses by 50-300%; the raw-PySCF-singlet MISS assertion is the
//!   explicit guard for the RHF factor.
//! * If the triplet channel is mis-built (δJ not cancelling, exchange on the
//!   wrong spin): the stretched-water external λ misses PySCF's −3.07e-1 and
//!   the singlet/triplet coupling block of ferric's UHF-at-RHF Hessian is not
//!   zero (ferric-internal identity, asserted).
//! * If the KS kernel is dropped (`fxc: None`, the HF Hessian at the KS
//!   density): NH2's λ_min lands near the UHF control (7.50e-2) instead of
//!   the UKS reference (7.87e-2); asserted as a MISS.
//! * If ferric lands on a different SCF STATE: the energy assertion (made
//!   FIRST) fails by mHa, so a state difference cannot masquerade as a
//!   Hessian difference.
//! * If the HARNESS is broken (geometry, basis, file): nuclear repulsion and
//!   AO count fail first.
//!
//! # TOLERANCES
//!
//! Each const records its measured maximum. The references are converged to
//! conv_tol_grad 1e-10: Hessian eigenvalues are first order in the orbital
//! error, and at conv_tol_grad 1e-8 the reference alone put ~1.2e-8 into
//! them.
//!
//! # NEGATIVE CONTROLS (asserted inside the tests)
//!
//! * stretched water: external verdict UNSTABLE with λ matching PySCF's
//!   negative triplet eigenvalue, while the internal verdict is STABLE.
//! * N2⁺ default guess: verdict UNSTABLE with PySCF's negative λ_min.
//! * RHF factor: ferric's singlet spectrum must MISS the raw (×2) PySCF one.
//! * channel identity: ferric's triplet λ must MISS the singlet reference.
//! * KS kernel: UKS λ_min must MISS the UHF control's λ_min.
//! * OH: verdict must be MARGINAL, not STABLE (a Goldstone mode is not a
//!   curvature).
//!
//! The UKS case compares the Davidson λ_min, not the dense UKS spectrum: the
//! f_xc response kernel (`FxcKernelStore`) is crate-private, so a test cannot
//! build the dense UKS Hessian. The UHF control is a separate SCF, so it is a
//! negative control only and does not isolate f_xc; dropping f_xc from the
//! stability Hessian (`fxc_ref = None` in `uhf.rs`) is the check that does,
//! and it fails this test. A defect that moves only higher UKS eigenvalues is
//! not covered.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::engine_pool::EnginePool;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rhf_newton::RhfNewtonInputs;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::{
    uhf_internal_stability, StabilityConfig, StabilityKind, StabilityResult, StabilityVerdict,
};
use ferric_scf::uhf::solve_uhf;
use ferric_scf::uhf_newton::UhfNewtonInputs;
use ferric_scf::ScfResult;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/scf_stability";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS: &str = "6-31g";

/// How many of the lowest dense-Hessian eigenvalues are compared.
const N_COMPARE: usize = 6;

/// Geometry check.
const TOL_ENUC: f64 = 1e-9;
/// HF energy vs PySCF. Measured max 7.1e-12.
const TOL_E: f64 = 1e-10;
/// UKS/PBE (exact J) energy vs PySCF. Measured 3.6e-14.
const TOL_E_KS: f64 = 1e-10;
/// Hessian eigenvalues (Davidson λ_min and the dense spectrum) vs PySCF.
/// Measured max 4.5e-10 (stretched water, UHF-at-RHF spectrum).
const TOL_LAMBDA: f64 = 1e-8;
/// UKS/PBE eigenvalues vs PySCF (f_xc on the same grid from the same
/// density). Measured 1.3e-11.
const TOL_LAMBDA_KS: f64 = 1e-8;
/// ferric-INTERNAL identities (singlet block of the UHF-at-RHF Hessian vs the
/// RHF Hessian; singlet/triplet coupling block). Same integrals, same thresh,
/// two different matvecs, integral_thresh 1e-12.
const TOL_IDENTITY: f64 = 1e-10;
/// Davidson λ_min vs ferric's own dense λ_min. Davidson conv_thresh 1e-6 on
/// the residual gives a λ error ~ r²/gap. Measured max 6.3e-15.
const TOL_DAVIDSON_VS_DENSE: f64 = 1e-9;
/// A quantity that must be MISSED is missed by at least this factor × its bar.
const MUST_MISS_FACTOR: f64 = 1000.0;

// ---------------------------------------------------------------------------
// Harness (same conventions as validation_open_shell_scf.rs)
// ---------------------------------------------------------------------------

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

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_scf_stability.py — a missing reference is a failure, never a skip",
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

fn nums(v: &Value, ptr: &str, ctx: &str) -> Vec<f64> {
    v.pointer(ptr)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an array"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

struct System {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn load_system(system: &str, r: &Value) -> System {
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let ctx = format!("{system}/{BASIS}");
    let enuc_ref = num(r, "/nuclear_repulsion", &ctx);
    let enuc = mol.nuclear_repulsion();
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{ctx}: nuclear repulsion {enuc:.12} vs reference {enuc_ref:.12} — geometry/unit mismatch"
    );
    let bs = basis::bundled(BASIS).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    assert_eq!(
        prep.nbasis(),
        r["nao"].as_u64().expect("nao") as usize,
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

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<22} ferric {got:+.12e} ref {want:+.12e} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12e} vs reference {want:.12e} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_spectrum(ctx: &str, what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert!(
        got.len() >= N_COMPARE && want.len() >= N_COMPARE,
        "{ctx}: {what}: need {N_COMPARE} eigenvalues (ferric {}, ref {})",
        got.len(),
        want.len()
    );
    for k in 0..N_COMPARE {
        check_close(ctx, &format!("{what}[{k}]"), got[k], want[k], tol);
    }
}

fn must_miss(ctx: &str, what: &str, got: f64, other: f64, bar: f64) {
    let d = (got - other).abs();
    eprintln!("{ctx}: MUST MISS {what}: ferric {got:+.10e} other {other:+.10e} |d| {d:.2e}");
    assert!(
        d > MUST_MISS_FACTOR * bar,
        "{ctx}: {what}: ferric {got:+.10e} is within {:.0e} of {other:+.10e} — the \
         comparison cannot tell the two apart",
        MUST_MISS_FACTOR * bar
    );
}

fn stability<'a>(res: &'a ScfResult, ctx: &str) -> &'a StabilityResult {
    let st = res
        .stability
        .as_ref()
        .unwrap_or_else(|| panic!("{ctx}: check_stability was set but no verdict came back"));
    eprintln!("{ctx}: {}", st.summary());
    assert!(
        st.converged,
        "{ctx}: Davidson not converged: {}",
        st.summary()
    );
    st
}

fn hf_config(descent: bool) -> RhfConfig {
    RhfConfig {
        max_iter: 500,
        energy_conv: 1e-11,
        // Tighter than the UHF/ROHF row (1e-9): the eigenvalues are FIRST
        // order in the density error, and this row asserts them at 1e-8.
        density_conv: 1e-10,
        check_stability: true,
        scf_stability_descent: descent,
        ..Default::default()
    }
}

fn uks_pbe_config() -> RhfConfig {
    RhfConfig {
        xc: Some("PBE".to_string()),
        // EXACT J, as the generator's dft.UKS without density_fit.
        df_j_aux: None,
        df_k_aux: None,
        max_iter: 500,
        energy_conv: 1e-10,
        density_conv: 1e-9,
        check_stability: true,
        scf_stability_descent: true,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Dense Hessians from ferric's matvecs
// ---------------------------------------------------------------------------

fn sym_eigh(h: &Array2<f64>, ctx: &str) -> Vec<f64> {
    let asym = (h - &h.t()).iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    eprintln!(
        "{ctx}: dense Hessian dim {} max asymmetry {asym:.2e}",
        h.nrows()
    );
    assert!(
        asym < 1e-8,
        "{ctx}: orbital Hessian not symmetric ({asym:.2e})"
    );
    let (ev, _) = (0.5 * (h + &h.t())).eigh(UPLO::Lower).unwrap();
    ev.to_vec()
}

fn dense_uhf(sys: &System, inp: &UhfNewtonInputs) -> Array2<f64> {
    let n = inp.c_a.nrows();
    let (na, nb) = (inp.nocc_a, inp.nocc_b);
    let (nva, nvb) = (n - na, n - nb);
    let (da, db) = (nva * na, nvb * nb);
    let pool = EnginePool::new(sys.bounds.op, &sys.prep, 1e-14).unwrap();
    let mut h = Array2::<f64>::zeros((da + db, da + db));
    for col in 0..da + db {
        let mut v = vec![0.0; da + db];
        v[col] = 1.0;
        let ka = Array2::from_shape_vec((nva, na), v[..da].to_vec()).unwrap();
        let kb = Array2::from_shape_vec((nvb, nb), v[da..].to_vec()).unwrap();
        let (ha, hb) =
            ferric_scf::uhf_newton::hessian_matvec(&sys.ctx, inp, &ka, &kb, &pool).unwrap();
        for (r, x) in ha.iter().chain(hb.iter()).enumerate() {
            h[(r, col)] = *x;
        }
    }
    h
}

fn dense_rhf(sys: &System, inp: &RhfNewtonInputs) -> Array2<f64> {
    let n = inp.c.nrows();
    let (no, nv) = (inp.nocc, n - inp.nocc);
    let pool = EnginePool::new(sys.bounds.op, &sys.prep, 1e-14).unwrap();
    let mut h = Array2::<f64>::zeros((nv * no, nv * no));
    for col in 0..nv * no {
        let mut v = vec![0.0; nv * no];
        v[col] = 1.0;
        let k = Array2::from_shape_vec((nv, no), v).unwrap();
        let hv = ferric_scf::rhf_newton::hessian_matvec(&sys.ctx, inp, &k, &pool).unwrap();
        for (r, x) in hv.iter().enumerate() {
            h[(r, col)] = *x;
        }
    }
    h
}

fn uhf_inputs<'a>(
    sys: &'a System,
    c_a: &'a Array2<f64>,
    c_b: &'a Array2<f64>,
    f_a_mo: &'a Array2<f64>,
    f_b_mo: &'a Array2<f64>,
    na: usize,
    nb: usize,
) -> UhfNewtonInputs<'a> {
    UhfNewtonInputs {
        prep: &sys.prep,
        bounds: &sys.bounds,
        c_a,
        c_b,
        f_a_mo,
        f_b_mo,
        nocc_a: na,
        nocc_b: nb,
        k_mix_sr: 1.0,
        fxc: None,
        thresh: RhfConfig::default().integral_thresh,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    }
}

/// Dense spectrum of ferric's HF UHF Hessian at a converged UHF result.
fn uhf_result_spectrum(sys: &System, res: &ScfResult, ctx: &str) -> Vec<f64> {
    let (na, nb) = nocc_ab(&sys.mol);
    let c_a = res.mos_alpha.clone();
    let c_b = res.mos_beta.clone().expect("UHF beta MOs");
    let f_a = c_a.t().dot(&res.fock_alpha).dot(&c_a);
    let f_b = c_b
        .t()
        .dot(res.fock_beta.as_ref().expect("UHF beta Fock"))
        .dot(&c_b);
    let inp = uhf_inputs(sys, &c_a, &c_b, &f_a, &f_b, na, nb);
    sym_eigh(&dense_uhf(sys, &inp), ctx)
}

// ---------------------------------------------------------------------------
// UHF internal: N2+ and OH
// ---------------------------------------------------------------------------

/// N2⁺ UHF: the stability-followed minimum AND the default-guess saddle.
#[test]
#[ignore = "validation: SCF stability"]
fn n2_plus_uhf_minimum_and_saddle_vs_pyscf() {
    let r = reference("n2_plus");
    let sys = load_system("n2_plus", &r);

    // (a) Stability-followed: must reach PySCF's stable minimum.
    let ctx = "n2_plus/6-31g/UHF minimum";
    let res = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &hf_config(true))
        .unwrap_or_else(|e| panic!("{ctx}: solve_uhf failed: {e:?}"));
    assert!(res.converged, "{ctx}: not converged");
    check_close(
        ctx,
        "energy",
        res.energy,
        num(&r, "/uhf/energy", ctx),
        TOL_E,
    );
    let st = stability(&res, ctx);
    assert_eq!(st.kind, StabilityKind::UhfInternal);
    assert_eq!(
        st.verdict(),
        StabilityVerdict::Stable,
        "{ctx}: {}",
        st.summary()
    );
    let want = nums(&r, "/uhf/hessian/lowest", ctx);
    check_close(
        ctx,
        "Davidson lambda_min",
        st.lowest_eigenvalue,
        want[0],
        TOL_LAMBDA,
    );
    let spec = uhf_result_spectrum(&sys, &res, ctx);
    check_close(
        ctx,
        "Davidson vs dense",
        st.lowest_eigenvalue,
        spec[0],
        TOL_DAVIDSON_VS_DENSE,
    );
    check_spectrum(ctx, "dense", &spec, &want, TOL_LAMBDA);

    // (b) NEGATIVE CONTROL: default guess, no descent. Both codes land on the
    // same saddle; the verdict must be UNSTABLE with PySCF's negative λ.
    let ctx = "n2_plus/6-31g/UHF default-guess saddle";
    let res = solve_uhf(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        &hf_config(false),
    )
    .unwrap_or_else(|e| panic!("{ctx}: solve_uhf failed: {e:?}"));
    assert!(res.converged, "{ctx}: not converged");
    check_close(
        ctx,
        "energy",
        res.energy,
        num(&r, "/uhf_default_guess_saddle/energy", ctx),
        TOL_E,
    );
    let st = stability(&res, ctx);
    assert_eq!(
        st.verdict(),
        StabilityVerdict::Unstable,
        "{ctx}: {}",
        st.summary()
    );
    let want = nums(&r, "/uhf_default_guess_saddle/hessian/lowest", ctx);
    assert!(
        want[0] < 0.0,
        "{ctx}: reference saddle must have a negative eigenvalue"
    );
    check_close(
        ctx,
        "Davidson lambda_min",
        st.lowest_eigenvalue,
        want[0],
        TOL_LAMBDA,
    );
    let spec = uhf_result_spectrum(&sys, &res, ctx);
    check_spectrum(ctx, "dense", &spec, &want, TOL_LAMBDA);
}

/// OH ²Π UHF: an exact zero mode, so the verdict must be MARGINAL.
#[test]
#[ignore = "validation: SCF stability"]
fn oh_uhf_goldstone_mode_vs_pyscf() {
    let r = reference("oh");
    let sys = load_system("oh", &r);
    let ctx = "oh/6-31g/UHF";
    let res = solve_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bounds, &hf_config(true))
        .unwrap_or_else(|e| panic!("{ctx}: solve_uhf failed: {e:?}"));
    assert!(res.converged, "{ctx}: not converged");
    check_close(
        ctx,
        "energy",
        res.energy,
        num(&r, "/uhf/energy", ctx),
        TOL_E,
    );

    let want = nums(&r, "/uhf/hessian/lowest", ctx);
    assert!(
        want[0].abs() < TOL_LAMBDA && want[1] > 0.1,
        "{ctx}: the reference must show one zero mode then a gap (got {:?})",
        &want[..2]
    );
    let st = stability(&res, ctx);
    // A Goldstone mode is not curvature: the analysis must say it cannot tell,
    // never STABLE. MARGINAL requires |λ_min| <= noise_floor (1e-6).
    assert_eq!(
        st.verdict(),
        StabilityVerdict::Marginal,
        "{ctx}: an exact zero mode must be reported MARGINAL: {}",
        st.summary()
    );
    check_close(
        ctx,
        "Davidson lambda_min",
        st.lowest_eigenvalue,
        want[0],
        TOL_LAMBDA,
    );
    let spec = uhf_result_spectrum(&sys, &res, ctx);
    // The zero mode itself and the first NONZERO eigenvalues.
    check_spectrum(ctx, "dense", &spec, &want, TOL_LAMBDA);
}

// ---------------------------------------------------------------------------
// RHF internal + RHF -> UHF external: water at two geometries
// ---------------------------------------------------------------------------

/// Split a UHF-at-RHF dense Hessian (blocks [[aa, ab], [ba, bb]], each of the
/// RHF dimension) into singlet (κ, κ)/√2 and triplet (κ, −κ)/√2 blocks plus
/// the coupling block between them.
fn singlet_triplet(h: &Array2<f64>) -> (Array2<f64>, Array2<f64>, f64) {
    let d = h.nrows() / 2;
    let aa = h.slice(ndarray::s![..d, ..d]);
    let ab = h.slice(ndarray::s![..d, d..]);
    let ba = h.slice(ndarray::s![d.., ..d]);
    let bb = h.slice(ndarray::s![d.., d..]);
    let s = 0.5 * ((&aa + &ab) + (&ba + &bb));
    let t = 0.5 * ((&aa - &ab) - (&ba - &bb));
    let c = 0.5 * ((&aa - &ab) + (&ba - &bb));
    let cmax = c.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    (s, t, cmax)
}

fn water_row(system: &str, expect_external: StabilityVerdict) {
    let r = reference(system);
    let sys = load_system(system, &r);
    let ctx = format!("{system}/6-31g/RHF");
    let ctx = ctx.as_str();

    let res = solve_rhf(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        Operator::coulomb(),
        &sys.bounds,
        &hf_config(false),
    )
    .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
    assert!(res.converged, "{ctx}: not converged");
    check_close(
        ctx,
        "energy",
        res.energy,
        num(&r, "/rhf/energy", ctx),
        TOL_E,
    );

    // --- RHF internal (singlet), through the SCF path --------------------
    let st_int = stability(&res, ctx);
    assert_eq!(st_int.kind, StabilityKind::RhfInternal);
    assert_eq!(
        st_int.verdict(),
        StabilityVerdict::Stable,
        "{ctx}: the RHF state is internally stable at both geometries: {}",
        st_int.summary()
    );
    let want_s = nums(&r, "/rhf/singlet_rhf_internal/lowest", ctx);
    let want_s_raw = nums(&r, "/rhf/singlet_rhf_internal/lowest_pyscf_raw", ctx);
    check_close(
        ctx,
        "internal lambda_min",
        st_int.lowest_eigenvalue,
        want_s[0],
        TOL_LAMBDA,
    );
    // Factor guard: PySCF's gen_g_hop_rhf is 2x ferric; the raw value must MISS.
    must_miss(
        ctx,
        "raw (x2) PySCF singlet lambda",
        st_int.lowest_eigenvalue,
        want_s_raw[0],
        TOL_LAMBDA,
    );

    let nocc = sys.mol.nelec() as usize / 2;
    let c = res.mos_alpha.clone();
    let f_mo = c.t().dot(&res.fock_alpha).dot(&c);
    let rinp = RhfNewtonInputs {
        prep: &sys.prep,
        bounds: &sys.bounds,
        c: &c,
        f_mo: &f_mo,
        nocc,
        k_mix_sr: 1.0,
        fxc: None,
        thresh: RhfConfig::default().integral_thresh,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let rhf_spec = sym_eigh(&dense_rhf(&sys, &rinp), &format!("{ctx} RHF"));
    check_spectrum(ctx, "singlet dense", &rhf_spec, &want_s, TOL_LAMBDA);

    // --- RHF -> UHF external: the UHF Hessian AT the RHF point -----------
    let uinp = uhf_inputs(&sys, &c, &c, &f_mo, &f_mo, nocc, nocc);
    let st_ext = uhf_internal_stability(&sys.ctx, &uinp, &StabilityConfig::default())
        .unwrap_or_else(|e| panic!("{ctx}: external analysis failed: {e:?}"));
    eprintln!(
        "{ctx}: external (UHF Hessian at the RHF point): {}",
        st_ext.summary()
    );
    assert!(st_ext.converged, "{ctx}: {}", st_ext.summary());
    let want_u = nums(&r, "/rhf/uhf_hessian_at_rhf_point/lowest", ctx);
    let want_t = nums(&r, "/rhf/triplet_rhf_to_uhf/lowest", ctx);
    check_close(
        ctx,
        "external lambda_min",
        st_ext.lowest_eigenvalue,
        want_u[0],
        TOL_LAMBDA,
    );
    assert_eq!(
        st_ext.verdict(),
        expect_external,
        "{ctx}: RHF->UHF verdict: {}",
        st_ext.summary()
    );
    let pyscf_ext_stable = r
        .pointer("/rhf/pyscf_verdict/external_stable")
        .and_then(Value::as_bool)
        .expect("pyscf_verdict.external_stable");
    assert_eq!(
        pyscf_ext_stable,
        expect_external == StabilityVerdict::Stable,
        "{ctx}: the reference's own external verdict disagrees with this test's expectation"
    );

    let h_u = dense_uhf(&sys, &uinp);
    let u_spec = sym_eigh(&h_u, &format!("{ctx} UHF@RHF"));
    check_close(
        ctx,
        "Davidson vs dense (ext)",
        st_ext.lowest_eigenvalue,
        u_spec[0],
        TOL_DAVIDSON_VS_DENSE,
    );
    check_spectrum(ctx, "UHF@RHF dense", &u_spec, &want_u, TOL_LAMBDA);

    // Channel split. Ferric-internal identities first (independent matvecs):
    // the singlet block IS the RHF Hessian, and nothing couples the channels.
    let (h_s, h_t, coupling) = singlet_triplet(&h_u);
    eprintln!("{ctx}: singlet/triplet coupling block max |.| {coupling:.2e}");
    assert!(
        coupling < TOL_IDENTITY,
        "{ctx}: singlet and triplet channels couple ({coupling:.2e}) — the UHF Hessian at an \
         RHF point must block-diagonalize"
    );
    let s_spec = sym_eigh(&h_s, &format!("{ctx} singlet block"));
    for k in 0..N_COMPARE {
        check_close(
            ctx,
            &format!("singlet block vs RHF [{k}]"),
            s_spec[k],
            rhf_spec[k],
            TOL_IDENTITY,
        );
    }
    let t_spec = sym_eigh(&h_t, &format!("{ctx} triplet block"));
    check_spectrum(ctx, "triplet (RHF->UHF)", &t_spec, &want_t, TOL_LAMBDA);
    // Channel guard: a triplet built as the singlet must fail.
    must_miss(
        ctx,
        "singlet reference as triplet",
        t_spec[0],
        want_s[0],
        TOL_LAMBDA,
    );

    match expect_external {
        StabilityVerdict::Unstable => {
            // THE NEGATIVE CONTROL of the row: internal says STABLE (asserted
            // above), external must find the negative triplet mode, and it is
            // the external λ_min that carries the instability.
            assert!(
                t_spec[0] < 0.0 && s_spec[0] > 0.0,
                "{ctx}: expected triplet<0<singlet"
            );
            check_close(
                ctx,
                "external = triplet min",
                st_ext.lowest_eigenvalue,
                t_spec[0],
                TOL_DAVIDSON_VS_DENSE,
            );
        }
        StabilityVerdict::Stable => {
            assert!(
                t_spec[0] > 0.0,
                "{ctx}: triplet must be positive at equilibrium"
            );
        }
        other => panic!("{ctx}: unexpected expectation {other:?}"),
    }
}

#[test]
#[ignore = "validation: SCF stability"]
fn water_equilibrium_rhf_internal_and_external_stable_vs_pyscf() {
    water_row("water_eq", StabilityVerdict::Stable);
}

#[test]
#[ignore = "validation: SCF stability"]
fn water_stretched_rhf_to_uhf_unstable_vs_pyscf() {
    water_row("water_stretched", StabilityVerdict::Unstable);
}

// ---------------------------------------------------------------------------
// UKS / PBE internal: NH2
// ---------------------------------------------------------------------------

#[test]
#[ignore = "validation: SCF stability"]
fn nh2_uks_pbe_vs_pyscf() {
    let r = reference("nh2");
    let sys = load_system("nh2", &r);
    let ctx = "nh2/6-31g/UKS-PBE";
    let res = solve_uhf(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        &uks_pbe_config(),
    )
    .unwrap_or_else(|e| panic!("{ctx}: solve_uhf failed: {e:?}"));
    assert!(res.converged, "{ctx}: not converged");
    check_close(
        ctx,
        "energy",
        res.energy,
        num(&r, "/uks_pbe/energy", ctx),
        TOL_E_KS,
    );
    must_miss(
        ctx,
        "UHF control energy",
        res.energy,
        num(&r, "/uhf_control/energy", ctx),
        TOL_E_KS,
    );

    let st = stability(&res, ctx);
    assert_eq!(st.kind, StabilityKind::UhfInternal);
    assert_eq!(
        st.verdict(),
        StabilityVerdict::Stable,
        "{ctx}: {}",
        st.summary()
    );
    let want = nums(&r, "/uks_pbe/hessian/lowest", ctx);
    check_close(
        ctx,
        "Davidson lambda_min",
        st.lowest_eigenvalue,
        want[0],
        TOL_LAMBDA_KS,
    );
    // KS kernel guard: the HF Hessian (the fxc-dropped trap) is ~3.6e-3 away.
    must_miss(
        ctx,
        "UHF control lambda_min",
        st.lowest_eigenvalue,
        num(&r, "/uhf_control/hessian/lowest/0", ctx),
        TOL_LAMBDA_KS,
    );
}
