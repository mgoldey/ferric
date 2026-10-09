//! VALIDATION tier — VALIDATION.md row "AO-Laplace MP2 (O(N))".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-mp2 --test validation_laplace_mp2 \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's minimax-Laplace RI-MP2 (`LaplaceMp2::compute_mo` and
//! `compute_ao`, Häser & Almlöf CPL 191, 299 (1992); Takatsuka, Ten-no,
//! Hackbusch JCP 129, 044112 (2008)) at n_quad ∈ {3, 5, 7} on five systems,
//! all electrons correlated, against TWO independent numpy references built on
//! PySCF density-fitted integrals (`scripts/validation/gen_laplace_mp2.py` →
//! `testdata/reference/validation/laplace_mp2/<system>_<basis>.json`):
//!
//! * `exact` — the dense (i,a,j,b) sum with the true `1/Δ` denominator, proved
//!   equal to PySCF's own `mp.dfmp2.DFMP2` to 1e-16 (JSON
//!   `exact.numpy_vs_pyscf_abs_diff`). This is the EXTERNAL reference the row
//!   previously lacked: its only checks were internal two-code-path agreement.
//! * `quadrature[n]` — the SAME dense sum with `1/Δ → Σ_k w_k e^{−t_k Δ}`,
//!   using the IDENTICAL minimax nodes, parsed out of
//!   `crates/ferric-quadrature/src/minimax.rs` (sha256 in provenance) and
//!   rescaled `t/ymin, w/ymin` exactly as `LaplaceQuadrature::new` does.
//!
//! PySCF was fed ferric's own orbital and aux basis JSON and ferric's geometry
//! in Bohr; its RHF is exact four-centre J/K, as is ferric's
//! `RhfConfig::default()`.
//!
//! # Why the second reference carries the weight
//!
//! It SPLITS the error a single DF-MP2 comparison conflates:
//!
//! * ferric vs `quadrature[n]` is IMPLEMENTATION error — same nodes, same
//!   fitted integrals, completely different assembly (τ-weighted DF amplitudes
//!   and a `naux × naux` Gram, versus an explicit four-index sum). It must sit
//!   at the DF/SCF floor, [`TOL_VS_QUADRATURE`].
//! * `quadrature[n]` vs `exact` is QUADRATURE error — the physics of the
//!   method, a property of `n_quad` and the range `R` alone, and the only part
//!   that is allowed to be large.
//!
//! Without the split, a 1e-7 disagreement with DF-MP2 is unattributable.
//!
//! # The Laplace range R, measured
//!
//! `LaplaceMp2::compute_mo` derives `ymin = 2(ε_LUMO − ε_HOMO)`,
//! `ymax = 2(ε_max − ε_0)` with ε_0 the lowest orbital even when frozen, and
//! `R = ymax/ymin`. `select_minimax_points` tabulates R ≤ 100 (k=3) and
//! R ≤ 1000 (k=5, 7) and hard-errors beyond that. Measured here:
//!
//! MEASURED 2026-10-03, release build, `OPENBLAS_NUM_THREADS=1`,
//! `RAYON_NUM_THREADS=2`, all five cases `5 passed; 0 failed`:
//!
//! | system             | R     | R_tab | MO ≡ AO | vs quad ref | vs exact DF-MP2 |
//! |--------------------|-------|-------|---------|-------------|-----------------|
//! | ch4/cc-pVDZ        | 19.04 |  20   | 8.3e-16 | 6.9e-14     | 8.01e-9         |
//! | alkane_8/cc-pVDZ   | 24.51 |  50   | 3.4e-13 | 3.1e-11     | 3.97e-8         |
//! | butadiene/cc-pVDZ  | 33.68 |  50   | 2.1e-14 | 4.2e-12     | 1.50e-7         |
//! | h2o/cc-pVDZ        | 36.40 |  50   | 1.3e-15 | 3.9e-12     | 8.31e-8         |
//! | h2o/aug-cc-pVDZ    | 45.67 |  50   | 7.6e-15 | 4.3e-12     | 2.22e-7         |
//!
//! The `vs exact DF-MP2` column reproduces each system's INDEPENDENTLY
//! computed quadrature error (JSON `quadrature[7].quadrature_error_e_corr`)
//! to its last printed digit. That is the split working: ferric's whole
//! disagreement with DF-MP2 is accounted for by the minimax nodes, with
//! nothing left over for the implementation.
//!
//! NO case approaches R_max for k=7, and none exceeds even the k=3 cap of 100.
//! Octane is NOT the widest range: ymax is set by the deepest core orbital, and
//! C 1s (−11.2 Eh) is far shallower than O 1s (−20.6 Eh), while octane's gap
//! (0.586 Eh) is wider than butadiene's (0.439 Eh). Water in a diffuse basis —
//! deep core AND small gap — is the widest case available at this scale.
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! Stated BEFORE the comparison was measured:
//!
//! * PHYSICS: the error vs exact DF-MP2 is quadrature error and therefore
//!   tracks R — it grows with R, is the same for the MO and AO paths (they
//!   share the nodes), falls by orders of magnitude from n_quad 3 → 5 → 7, and
//!   appears in BOTH spin components, because the Laplace substitution is made
//!   in the denominator that both share.
//! * ARTIFACT: an error that is FLAT across systems does not come from the
//!   quadrature (R varies by 2.4x here and the minimax error by ~30x), and an
//!   error confined to E_SS is an implementation suspect — E_SS is the only
//!   component with the index-swapped exchange Gram
//!   (`laplace_exchange_energy`), so a transposition or blocking defect there
//!   shows up in E_SS while leaving the J term, hence E_OS, correct.
//! * The two hypotheses are DISTINGUISHABLE here: ferric-vs-`quadrature[n]`
//!   is blind to quadrature error by construction, so an implementation defect
//!   fails that assertion at 1e-12 while quadrature error cannot.
//!
//! MEASURED VERDICT (recorded, provisional, 2026-10-03). Ordered by R, the
//! error vs exact DF-MP2 at n_quad = 7 is:
//!
//! ```text
//!   ch4      R 19.04   8.01e-9
//!   alkane_8 R 24.51   3.97e-8
//!   butadiene R 33.68  1.50e-7
//!   h2o      R 36.40   8.31e-8
//!   h2o/aug  R 45.67   2.22e-7
//! ```
//!
//! The error DOES track R: it spans 28x from the smallest R to the largest,
//! monotone at both ends (ch4 smallest, h2o/aug largest). The one inversion
//! is butadiene (R 33.7, 1.50e-7) above h2o (R 36.4, 8.31e-8), a factor 1.8
//! out of order. That scatter is what a real minimax error looks like — the
//! error depends on where the actual Δ spectrum SITS inside `[ymin, ymax]`,
//! not on the endpoint ratio alone — and per the Experimental Protocol's "too
//! clean is a stop condition", a perfectly monotone series across five
//! heterogeneous systems would have been the more suspicious outcome.
//!
//! Neither artifact signature appears: the error is NOT flat (28x spread) and
//! NOT confined to E_SS (OS and SS each carry a share, and on alkane_8 the SS
//! error is 2.6e-10 against an OS error of 4.0e-8 — the opposite imbalance to
//! the one an exchange-Gram defect would produce).
//!
//! NOTE on the issue's framing: it expected octane to be the stress case for
//! R. It is not — octane is the SECOND SMALLEST R of the set. R is driven by
//! the GAP and the core depth, not by system size: octane's gap (0.586 Eh) is
//! wider than butadiene's (0.439), and its C 1s core (−11.2 Eh) is far
//! shallower than water's O 1s (−20.6). Water in a diffuse basis — deep core
//! AND small gap — is the widest case available at this scale.
//!
//! # MUTATION LEDGER (2026-10-03)
//!
//! Each mutation was applied to `crates/ferric-mp2/src/laplace.rs`, built, and
//! run against `laplace_mp2_h2o_ccpvdz_vs_pyscf_dfmp2` with
//! `-- --ignored` (these tests are `#[ignore]`-gated, so a run without it
//! reports `0 passed; 5 ignored` and proves nothing). Every line below was
//! read off a run that executed exactly one test.
//!
//! | # | mutation | result | caught by | observed |
//! |---|----------|--------|-----------|----------|
//! | M1 | drop one quadrature weight (`weights.last = 0`) | `0 passed; 1 failed` | vs quadrature ref, n_quad=3 | \|d\| 1.30e-2 vs bar 1e-9 |
//! | M2 | shrink the range: `ymax` from ε_HOMO, not ε_0 (4 sites) | `0 passed; 1 failed` | vs quadrature ref, n_quad=3 | \|d\| 1.31e-3 vs bar 1e-9 |
//! | M3 | frozen core off by one, `active_occ` only (4 sites) | `0 passed; 1 failed` | library panic, laplace.rs:784 | `ShapeError/IncompatibleShape` |
//! | M3b | frozen core off by one, applied CONSISTENTLY (4 sites) | `0 passed; 1 failed` | vs quadrature ref, n_quad=3 | \|d\| 2.46e-3 vs bar 1e-9 |
//! | M4 | perturb one weight by 1e-4 relative | `1 passed; 3 failed` (unit tests) | the TIGHTENED `laplace.rs` bars | 2.66e-6, 3.53e-6, 2.65e-6 |
//!
//! Three results are worth keeping:
//!
//! * M3 was caught only by a SHAPE PANIC, not by a number — the naive
//!   off-by-one is internally inconsistent (`nocc` shrinks while `c_occ` is
//!   still sliced `frozen_core..nocc_total`). That is a weak observation, so
//!   M3b repeats it CONSISTENTLY, which is the version that would return a
//!   silently wrong energy. M3b is caught numerically at 2.46e-3.
//! * M3b passed the MO ≡ AO anchor at 2.8e-16 while being wrong by 2.5e-3.
//!   The exactness anchor cannot see what the two paths share; only the
//!   external reference can. Recorded because the previous grade for this row
//!   rested on exactly that internal agreement.
//! * M4 is the check that the TIGHTENING in `laplace.rs` bought something: a
//!   1e-4 weight perturbation produces errors of 2.6e-6 .. 3.5e-6, which the
//!   OLD 1e-3 bars all passed and the new ones all fail. The H2 convergence
//!   test still passes under M4 (a near-uniform weight scaling moves e3, e5
//!   and e7 together, leaving their DIFFERENCES intact) — a scope limit of
//!   that test, not a defect.
//!
//! # TOLERANCES
//!
//! Each bar is set from the measured maximum recorded on its const. The spec
//! target was 1e-6 vs DF-MP2; the measured maximum is 2.2e-7 (h2o/aug-cc-pVDZ),
//! so [`TOL_VS_EXACT`] is set from the measurement at 2e-6 rather than from
//! the target.
//!
//! # NEGATIVE CONTROLS (asserted in-test, always on)
//!
//! * n_quad = 3 must MISS the n_quad = 7 bar on every system. MEASURED
//!   misses: ch4 9.21e-6, alkane_8 7.86e-4, butadiene 2.34e-3, h2o 4.96e-4,
//!   h2o/aug 4.03e-4. The smallest, ch4's 9.21e-6, is 4.6x above
//!   [`TOL_VS_EXACT`] — the control passes on every system, but with the
//!   least headroom exactly where R is smallest, which is the expected place
//!   for it to be tightest.
//! * The OS/SS SWAP must miss: ferric's E_OS compared to the reference E_SS
//!   and vice versa, so a component transposition cannot pass.
//! * |quadrature error| must decrease over n_quad 3 → 5 → 7 for each system.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::laplace::LaplaceMp2;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/laplace_mp2";
const MOL_DIR: &str = "testdata/molecules/validation";

/// Geometry check (PySCF E_nuc from ferric's Bohr geometry).
const TOL_ENUC: f64 = 1e-9;
/// E_RHF vs PySCF, exact J/K on both sides.
// Measured max 2.69e-10 (alkane_8; 1.3e-13 on h2o). Bar ~4x above the max:
// octane's 313 Eh total means 2.7e-10 is 9e-13 relative.
const TOL_E_RHF: f64 = 1e-9;
/// MO ≡ AO: two independent assembly paths over the SAME nodes and integrals.
/// The exactness anchor, asserted before any external comparison.
//
// Measured max 3.4e-13 over five systems x n_quad {3,5,7} — and that single
// value is octane (E_corr -1.18 Eh, so 2.9e-13 relative); every other case is
// at or below 2.1e-14, i.e. machine precision. Bar ~300x above the max to
// leave room for
// summation-order drift across thread counts (the width/thread traps are
// pinned separately by laplace_energy_is_thread_independent.rs).
//
// SCOPE LIMIT, demonstrated by mutation M3b: this anchor is blind to any
// defect the two paths SHARE. A consistent frozen-core off-by-one changed
// E_corr by 2.5e-3 while MO and AO still agreed to 2.8e-16. Internal
// agreement is not corroboration; that is what the external reference is for.
const TOL_MO_VS_AO: f64 = 1e-10;
/// ferric vs the quadrature-isolated numpy reference: IMPLEMENTATION error,
/// with quadrature error removed by construction (identical nodes).
//
// Measured max 3.1e-11 (alkane_8, where E_corr is -1.18 Eh, so 2.6e-11
// relative); 6.9e-14 .. 4.3e-12 on the four smaller cases. This is the
// DF/SCF/summation-order floor with NO quadrature content, so it is the
// assertion that actually pins the implementation. Bar ~30x above the max.
//
// This is the assertion all three code mutations tripped (M1 1.3e-2,
// M2 1.3e-3, M3b 2.5e-3) — including M3b, which the MO=AO anchor could not
// see.
const TOL_VS_QUADRATURE: f64 = 1e-9;
/// ferric at n_quad = 7 vs EXACT DF-MP2: this is dominated by quadrature
/// error, not by implementation error.
//
// Measured max 2.22e-7 (h2o/aug-cc-pVDZ, R = 45.67 — the widest range of the
// set). The spec target was 1e-6; the bar is set ~10x above the MEASUREMENT,
// which lands at 2e-6, i.e. LOOSER than the spec target. Recording the
// measurement rather than forcing the target is deliberate: at n_quad = 7 the
// minimax error for R ~ 46 simply is 2e-7, and a 1e-6 bar would leave only
// 4.5x headroom on a quantity that moves with the SCF's converged spectrum.
const TOL_VS_EXACT: f64 = 2e-6;
/// The lowest quadrature size must MISS the n_quad = 7 bar — otherwise
/// [`TOL_VS_EXACT`] would be too loose to distinguish 3 nodes from 7, and
/// passing it would be arithmetic rather than measurement.
// Measured smallest n_quad=3 error: 9.21e-6 (ch4), 4.6x above TOL_VS_EXACT;
// largest 2.34e-3 (butadiene).
const MUST_MISS: f64 = TOL_VS_EXACT;

const N_QUADS: [usize; 3] = [3, 5, 7];
const N_QUAD_PROD: usize = 7;

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
        .expect("ferric-mp2 manifest dir should be <root>/crates/ferric-mp2")
        .to_path_buf()
}

fn reference(system: &str, basis_name: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{basis_name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_laplace_mp2.py — a missing reference is a failure, never a skip",
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

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<30} ferric {got:+.13} ref {want:+.13} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.13} vs reference {want:.13} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_miss(ctx: &str, what: &str, got: f64, want: f64, bar: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<30} |d| {d:.2e} (must exceed {bar:.0e})");
    assert!(
        d > bar,
        "{ctx}: negative control {what}: |d| {d:.2e} <= {bar:.0e} — the bar cannot \
         tell these apart"
    );
}

struct Setup {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

/// The quadrature row for `n_quad`, or a panic if the generator recorded that
/// the range was exceeded — in which case ferric must hard-error too, which is
/// checked separately so a range finding never reads as agreement.
fn quad_row<'a>(r: &'a Value, n_quad: usize, ctx: &str) -> &'a Value {
    r["quadrature"]
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: quadrature array missing"))
        .iter()
        .find(|row| row["n_quad"].as_u64() == Some(n_quad as u64))
        .unwrap_or_else(|| panic!("{ctx}: no quadrature row for n_quad={n_quad}"))
}

fn check_system(system: &str, basis_name: &str, aux_name: &str, run_ao: bool) {
    let r = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    assert_eq!(r["basis"].as_str(), Some(basis_name), "{ctx}: basis");
    assert_eq!(r["aux_basis"].as_str(), Some(aux_name), "{ctx}: aux basis");

    // ---- harness checks: geometry, basis, RHF ----
    let stem = r["geometry_stem"]
        .as_str()
        .unwrap_or_else(|| panic!("{ctx}: geometry_stem missing"));
    let xyz = workspace_root().join(MOL_DIR).join(format!("{stem}.xyz"));
    let mol = Molecule::load_xyz(xyz.to_str().unwrap())
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    check_close(
        &ctx,
        "E_nuc",
        mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &ctx),
        TOL_ENUC,
    );
    let obs = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    assert_eq!(
        obs.nbasis() as u64,
        r["nao"].as_u64().unwrap(),
        "{ctx}: nao"
    );
    assert_eq!(
        dfbs.nbasis() as u64,
        r["naux"].as_u64().unwrap(),
        "{ctx}: naux"
    );

    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    };
    assert!(
        cfg.df_j_aux.is_none() && cfg.df_k_aux.is_none(),
        "exact J/K expected"
    );
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{ctx}: solve_rhf failed: {e:?}"));
    assert!(rhf.converged, "{ctx}: RHF not converged");
    check_close(
        &ctx,
        "E_RHF",
        rhf.energy,
        num(&r, "/e_rhf", &ctx),
        TOL_E_RHF,
    );

    // ---- the Laplace range ferric will derive, asserted against the one the
    // reference's nodes were built for. A range mismatch means ferric and the
    // reference are not using the same quadrature at all, and must fail HERE
    // rather than as a mystery energy disagreement.
    let eps = rhf.eps_r();
    let nmo = eps.len();
    let nocc = mol.nelec() as usize / 2;
    let ymin = 2.0 * (eps[nocc] - eps[nocc - 1]);
    let ymax = 2.0 * (eps[nmo - 1] - eps[0]);
    let r_range = ymax / ymin;
    check_close(
        &ctx,
        "ymin",
        ymin,
        num(&r, "/laplace_range/ymin", &ctx),
        1e-8,
    );
    check_close(
        &ctx,
        "ymax",
        ymax,
        num(&r, "/laplace_range/ymax", &ctx),
        1e-7,
    );
    eprintln!(
        "{ctx}: R = {r_range:.4} (reference {:.4})",
        num(&r, "/laplace_range/r", &ctx)
    );

    let su = Setup {
        mol,
        obs,
        dfbs,
        rhf,
    };

    // ---- EXACTNESS ANCHOR, first: MO == AO over every n_quad ----
    //
    // Two independent assembly paths (MO τ-weighted Gram vs AO pseudo-density
    // J plus MO exchange) over the same nodes. This catches a construction
    // error in either path before any external number is consulted; it cannot
    // see a defect in what they SHARE, which is what the external comparison
    // below is for.
    let mut mo_by_n = Vec::new();
    for &n_quad in &N_QUADS {
        let mut lap = LaplaceMp2::new(n_quad);
        let e_mo = lap
            .compute_mo(&su.mol, &su.obs, &su.dfbs, op, &su.rhf, 0)
            .unwrap_or_else(|e| panic!("{ctx}: compute_mo(n_quad={n_quad}) failed: {e:?}"));
        if run_ao {
            let mut lap_ao = LaplaceMp2::new(n_quad);
            let (e_ao, _, _) = lap_ao
                .compute_ao(&su.mol, &su.obs, &su.dfbs, op, &su.rhf, 0, None)
                .unwrap_or_else(|e| panic!("{ctx}: compute_ao(n_quad={n_quad}) failed: {e:?}"));
            check_close(
                &ctx,
                &format!("MO vs AO (n_quad={n_quad})"),
                e_mo,
                e_ao,
                TOL_MO_VS_AO,
            );
        }
        mo_by_n.push((n_quad, e_mo));
    }

    // ---- ferric vs the quadrature-isolated reference: IMPLEMENTATION error ----
    for &(n_quad, e_mo) in &mo_by_n {
        let row = quad_row(&r, n_quad, &ctx);
        assert_eq!(
            row["range_exceeded"].as_bool(),
            Some(false),
            "{ctx}: the reference recorded the minimax range as EXCEEDED for \
             n_quad={n_quad}; ferric hard-errors there and this comparison does not apply"
        );
        check_close(
            &ctx,
            &format!("vs quadrature ref (n_quad={n_quad})"),
            e_mo,
            num(row, "/e_corr", &ctx),
            TOL_VS_QUADRATURE,
        );
    }

    // ---- ferric vs EXACT DF-MP2, spin-resolved, at production n_quad ----
    let e_os_ref = num(&r, "/exact/e_os", &ctx);
    let e_ss_ref = num(&r, "/exact/e_ss", &ctx);
    let e_corr_ref = num(&r, "/exact/e_corr", &ctx);
    // The reference's own anchor: its exact sum IS PySCF DF-MP2.
    check_close(
        &ctx,
        "ref exact vs PySCF DFMP2",
        e_corr_ref,
        num(&r, "/exact/pyscf_dfmp2_e_corr", &ctx),
        1e-10,
    );

    let mut lap = LaplaceMp2::new(N_QUAD_PROD);
    let (e_corr, e_os, e_ss) = lap
        .compute_ao(&su.mol, &su.obs, &su.dfbs, op, &su.rhf, 0, None)
        .unwrap_or_else(|e| panic!("{ctx}: compute_ao(n_quad=7) failed: {e:?}"));
    check_close(&ctx, "E_OS vs exact DF-MP2", e_os, e_os_ref, TOL_VS_EXACT);
    check_close(&ctx, "E_SS vs exact DF-MP2", e_ss, e_ss_ref, TOL_VS_EXACT);
    check_close(
        &ctx,
        "E_corr vs exact DF-MP2",
        e_corr,
        e_corr_ref,
        TOL_VS_EXACT,
    );

    // ---- NEGATIVE CONTROL: the OS/SS swap must miss ----
    //
    // E_OS and E_SS differ by ~0.06-0.4 Eh on these systems, so a transposed
    // component pair cannot pass the bar above. Without this, a test that only
    // ever sums the two would be blind to the swap.
    check_miss(
        &ctx,
        "E_OS vs exact E_SS (swap)",
        e_os,
        e_ss_ref,
        TOL_VS_EXACT,
    );
    check_miss(
        &ctx,
        "E_SS vs exact E_OS (swap)",
        e_ss,
        e_os_ref,
        TOL_VS_EXACT,
    );

    // ---- NEGATIVE CONTROL: n_quad = 3 must MISS the production bar ----
    let e3 = mo_by_n
        .iter()
        .find(|(n, _)| *n == 3)
        .map(|(_, e)| *e)
        .expect("n_quad=3 was computed above");
    check_miss(&ctx, "n_quad=3 vs exact DF-MP2", e3, e_corr_ref, MUST_MISS);

    // ---- convergence: |error vs exact| strictly decreasing in n_quad ----
    let errs: Vec<(usize, f64)> = mo_by_n
        .iter()
        .map(|&(n, e)| (n, (e - e_corr_ref).abs()))
        .collect();
    eprintln!(
        "{ctx}: |err vs exact| by n_quad: {}",
        errs.iter()
            .map(|(n, d)| format!("{n}:{d:.3e}"))
            .collect::<Vec<_>>()
            .join("  ")
    );
    for w in errs.windows(2) {
        assert!(
            w[0].1 > w[1].1,
            "{ctx}: |error| did not decrease from n_quad={} ({:.3e}) to n_quad={} ({:.3e})",
            w[0].0,
            w[0].1,
            w[1].0,
            w[1].1
        );
    }
}

#[test]
#[ignore = "validation: AO-Laplace MP2"]
fn laplace_mp2_h2o_ccpvdz_vs_pyscf_dfmp2() {
    check_system("h2o", "cc-pvdz", "cc-pvdz-ri", true);
}

#[test]
#[ignore = "validation: AO-Laplace MP2"]
fn laplace_mp2_h2o_augccpvdz_vs_pyscf_dfmp2() {
    check_system("h2o_aug", "aug-cc-pvdz", "aug-cc-pvdz-rifit", true);
}

#[test]
#[ignore = "validation: AO-Laplace MP2"]
fn laplace_mp2_ch4_ccpvdz_vs_pyscf_dfmp2() {
    check_system("ch4", "cc-pvdz", "cc-pvdz-ri", true);
}

/// Butadiene adds the smallest HOMO-LUMO gap of the set (0.439 Eh vs water's
/// 0.679), hence a wider Laplace range at a hydrocarbon core depth.
#[test]
#[ignore = "validation: AO-Laplace MP2"]
fn laplace_mp2_butadiene_ccpvdz_vs_pyscf_dfmp2() {
    check_system("butadiene", "cc-pvdz", "cc-pvdz-ri", true);
}

/// n-octane: the largest system of the set (nao 202, naux 700, 33 occupied),
/// which is where the AO path's blocked pseudo-density machinery is actually
/// exercised. Run under the full slot, not `--light`.
#[test]
#[ignore = "validation: AO-Laplace MP2"]
fn laplace_mp2_octane_ccpvdz_vs_pyscf_dfmp2() {
    check_system("alkane_8", "cc-pvdz", "cc-pvdz-ri", true);
}
