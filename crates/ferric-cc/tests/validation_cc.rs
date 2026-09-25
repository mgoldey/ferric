//! VALIDATION tier — VALIDATION.md rows "MP3", "RI-CCD", "RI-CCSD spin-orbital /
//! spin-adapted" and "CCSD(T) / spin-adapted (T)".
//!
//! Runs only in the `validation` CI job (and on demand):
//!
//! ```text
//! cargo nextest run -p ferric-cc --test validation_cc \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's density-fitted MP3 (`ferric_mp2::mp3::mp3_energy`), CCD
//! (`ferric_cc::ccd::ccd`), spin-orbital CCSD (`ccsd::ccsd`), spin-adapted CCSD
//! (`ccsd_closed_shell::ccsd_closed_shell`), spin-orbital (T)
//! (`ccsd_t::ccsd_t`) and spin-adapted (T)
//! (`ccsd_t_closed_shell::ccsd_t_closed_shell`) correlation energies against
//! references from `scripts/validation/gen_cc.py`, stored in
//! `testdata/reference/validation/cc/<system>_<basis>.json`.
//!
//! The references use the SAME approximation as ferric, not a nearby one:
//! an exact-integral RHF (ferric's `solve_rhf` default: no RI-J, no RI-K), then
//! every correlation integral from one DF factorization with ferric's own
//! bundled auxiliary basis (`cc-pvdz-ri`, `def2-svp-rifit`,
//! `aug-cc-pvdz-rifit`), `(pq|rs) = Σ_P B^P_pq B^P_rs`, and a diagonal
//! canonical Fock. PySCF's own `CCD`, `CCSD` and `ccsd_t` run on those MO
//! integrals; MP3 is two independent numpy constructions (spin-orbital
//! textbook and closed-shell via PySCF's linear doubles residual) that the
//! generator requires to agree to 1e-10. The generator also checks its DF
//! factor against PySCF's `density_fit` MP2 (agreement ~1e-16). So the only
//! differences left are the integral engine (libint2 vs libcint), SCF
//! convergence, and ferric's CC/MP3 algebra — the thing under test.
//!
//! Systems: H2O and NH3 at cc-pVDZ and def2-SVP (MP3, CCD, CCSD); HCN/cc-pVDZ
//! and H2O/aug-cc-pVDZ (CCSD, (T)); H2O/cc-pVDZ ((T)); C2H6/cc-pVDZ
//! (spin-adapted CCSD and (T) only — the spin-orbital driver holds two
//! (2·nv)⁴ = 98⁴ tensors there, ~1.5 GB, and is too slow for this tier).
//!
//! Closed shell only: ferric's CC and MP3 drivers take a restricted reference
//! (`ScfResult::eps_r`/`mos_r`), so there is no open-shell CC to validate.
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's CC algebra is right: every correlation energy agrees with the
//!   like-for-like reference to the integral/convergence floor (expected
//!   ~1e-9; the SCF sits 1e-12 apart on the UHF/ROHF row, and the CC
//!   convergence test is |ΔE| < 1e-11).
//! * If a residual term is wrong or missing (a dropped ring or ladder, a wrong
//!   permutation factor, a (T) multiplicity weight): the energy misses by
//!   1e-5 to 1e-2 Ha, which is 3+ orders above the bar. A term that is exactly
//!   zero for H2 (the old test system) is not zero here.
//! * If the HARNESS is broken instead: a geometry/unit slip fails the nuclear
//!   repulsion check; a basis slip fails the AO count or the RHF energy check;
//!   a wrong aux basis or an exact-vs-DF mix-up shows up as the DF error,
//!   7e-5 to 2.7e-4 Ha on these systems (recorded in each JSON as
//!   `conventional_exact_integrals`) — ferric is ASSERTED to miss that
//!   conventional value, so a bar loose enough to accept either approximation
//!   fails.
//!
//! # TOLERANCES
//!
//! Each bar is set from the measured maximum recorded on its const. The
//! spin-orbital CCSD bar was unreachable before the ccsd.rs fix on this branch
//! (it missed by 6.6e-6..8.9e-5; see the commit message).
//!
//! # NEGATIVE CONTROLS / MUTATIONS (asserted inside the tests)
//!
//! * Frozen core: ferric all-electron must MISS the frozen-core reference
//!   (and ferric frozen-core must MATCH it) — frozen-vs-all-electron differs
//!   by ~2e-3 Ha.
//! * CCD vs CCSD: ferric CCD must MISS the CCSD reference and ferric CCSD must
//!   MISS the CCD reference (they differ by ~7e-4 Ha).
//! * DF vs exact: ferric CCSD must MISS the exact-integral CCSD.
//! * Internal cross-check: spin-orbital and spin-adapted CCSD, and
//!   spin-orbital and spin-adapted (T), run as two independent ferric
//!   pipelines on the same system, must agree with each other.
//! * MUTATION (manual, when re-grading): set the ring-term coefficient in
//!   `ccd.rs` or the `e_ring` einsum in `mp3.rs` to zero; every system misses
//!   by the ring energy (~8e-2 Ha for MP3).
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_cc::ccd::ccd;
use ferric_cc::ccsd::ccsd;
use ferric_cc::ccsd_closed_shell::ccsd_closed_shell;
use ferric_cc::ccsd_t::ccsd_t;
use ferric_cc::ccsd_t_closed_shell::ccsd_t_closed_shell;
use ferric_cc::{CcConfig, CcResult};
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::mp3::mp3_energy;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/cc";
const MOL_DIR: &str = "testdata/molecules/validation";

/// Geometry check.
const TOL_ENUC: f64 = 1e-9;
/// RHF total energy vs PySCF exact RHF. Measured max 1.9e-11.
const TOL_RHF: f64 = 1e-9;
/// MP2 and MP3 correlation energies. Measured max 7.6e-12.
const TOL_MP: f64 = 1e-10;
/// CCD/CCSD correlation energies (spin-orbital and spin-adapted). Measured
/// max 4.7e-11 (HCN, spin-orbital CCSD).
const TOL_CC: f64 = 2e-10;
/// (T) energies (spin-orbital and spin-adapted). Measured max 6.9e-12.
const TOL_T: f64 = 1e-10;
/// ferric spin-orbital vs ferric spin-adapted on the same system (two
/// independent ferric pipelines, same integrals). Measured 8.9e-12.
const TOL_INTERNAL: f64 = 1e-10;
/// A reference ferric must MISS: the smallest control gap on these systems is
/// the DF error of CCSD (7e-5 Ha at def2-SVP), so 1e-5 separates cleanly and
/// is still ≥ 1000x the energy bars above.
const MUST_MISS: f64 = 1e-5;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

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
        .expect("ferric-cc manifest dir should be <root>/crates/ferric-cc")
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
             scripts/validation/gen_cc.py — a missing reference is a failure, never a skip",
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

fn uint(v: &Value, ptr: &str, ctx: &str) -> usize {
    v.pointer(ptr)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not an integer"))
        as usize
}

/// One system x basis: molecule, orbital + aux bases, converged exact RHF.
struct Case {
    ctx: String,
    reference: Value,
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
    n_frozen: usize,
}

fn load_case(system: &str, basis_name: &str, checks: &mut Checks) -> Case {
    let reference = reference(system, basis_name);
    let ctx = format!("{system}/{basis_name}");
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 1)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    // Geometry like-for-like FIRST: a unit/constant slip fails as a geometry
    // defect, not as an unexplained correlation-energy offset. Hard assert:
    // nothing downstream means anything if this is wrong.
    let enuc_ref = num(&reference, "/nuclear_repulsion", &ctx);
    let enuc = mol.nuclear_repulsion();
    assert!(
        (enuc - enuc_ref).abs() < TOL_ENUC,
        "{ctx}: nuclear repulsion {enuc:.12} vs reference {enuc_ref:.12} — geometry/unit mismatch"
    );

    let aux_name = reference["aux_basis"]
        .as_str()
        .unwrap_or_else(|| panic!("{ctx}: aux_basis missing"))
        .to_string();
    let obs = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(&aux_name).unwrap()).unwrap();
    assert_eq!(
        obs.nbasis(),
        uint(&reference, "/nao", &ctx),
        "{ctx}: AO count differs from the reference's"
    );
    assert_eq!(
        dfbs.nbasis(),
        uint(&reference, "/naux", &ctx),
        "{ctx}: aux ({aux_name}) function count differs from the reference's"
    );

    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        max_iter: 200,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg)
        .unwrap_or_else(|e| panic!("{ctx}: RHF failed: {e:?}"));
    assert!(rhf.converged, "{ctx}: RHF did not converge");
    assert!(
        rhf.df_jk.is_none(),
        "{ctx}: RHF used density fitting ({:?}); the reference RHF is exact",
        rhf.df_jk
    );
    checks.close(
        &ctx,
        "E_RHF",
        rhf.energy,
        num(&reference, "/rhf/energy", &ctx),
        TOL_RHF,
    );
    let n_frozen = uint(&reference, "/n_frozen_core_control", &ctx);
    Case {
        ctx,
        reference,
        mol,
        obs,
        dfbs,
        rhf,
        n_frozen,
    }
}

fn cc_cfg(frozen_core: usize) -> CcConfig {
    CcConfig {
        frozen_core,
        max_iter: 200,
        // The drivers stop on |ΔE| < min(energy_conv, 1e-9).
        energy_conv: 1e-11,
        ..Default::default()
    }
}

/// Collects every comparison, prints all of them, then fails once at the end,
/// so a single run MEASURES every system (the tolerances are set from that
/// output) instead of stopping at the first miss.
#[derive(Default)]
struct Checks {
    failures: Vec<String>,
}

impl Checks {
    fn close(&mut self, ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
        let d = (got - want).abs();
        eprintln!(
            "{ctx}: {what:<22} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})"
        );
        if d.is_nan() || d >= tol {
            self.failures.push(format!(
                "{ctx}: {what} {got:.12} vs {want:.12} (|d| {d:.2e} >= {tol:.0e})"
            ));
        }
    }

    /// Negative control: `got` must differ from `wrong` by more than MUST_MISS.
    fn miss(&mut self, ctx: &str, what: &str, got: f64, wrong: f64) {
        let d = (got - wrong).abs();
        eprintln!("{ctx}: MUST MISS {what:<30} |d| {d:.2e} (> {MUST_MISS:.0e})");
        if d.is_nan() || d <= MUST_MISS {
            self.failures.push(format!(
                "{ctx}: negative control '{what}' did not miss: {got:.12} vs {wrong:.12} (|d| {d:.2e})"
            ));
        }
    }

    fn finish(self, row: &str) {
        assert!(
            self.failures.is_empty(),
            "{row}: {} failure(s):\n  {}",
            self.failures.len(),
            self.failures.join("\n  ")
        );
    }
}

fn run_ccsd_so(c: &Case, frozen: usize) -> CcResult {
    ccsd(
        &c.mol,
        &c.obs,
        &c.dfbs,
        Operator::coulomb(),
        &c.rhf,
        &cc_cfg(frozen),
    )
    .unwrap_or_else(|e| panic!("{}: spin-orbital CCSD failed: {e:?}", c.ctx))
}

fn run_ccsd_sa(c: &Case, frozen: usize) -> CcResult {
    ccsd_closed_shell(
        &c.mol,
        &c.obs,
        &c.dfbs,
        Operator::coulomb(),
        &c.rhf,
        &cc_cfg(frozen),
    )
    .unwrap_or_else(|e| panic!("{}: spin-adapted CCSD failed: {e:?}", c.ctx))
}

/// Guards against a vacuous comparison: the reference energy must be a real,
/// negative correlation energy of plausible size.
fn assert_nonvacuous(ctx: &str, what: &str, e: f64) {
    assert!(
        e < -1e-4,
        "{ctx}: reference {what} = {e:.12} is not a plausible correlation energy"
    );
}

// ---------------------------------------------------------------------------
// MP3 (VALIDATION.md "MP3")
// ---------------------------------------------------------------------------

#[test]
#[ignore = "validation: MP3"]
fn mp3_matches_df_reference() {
    let mut ck = Checks::default();
    for (system, basis_name, check_frozen) in [
        ("h2o", "cc-pvdz", true),
        ("h2o", "def2-svp", false),
        ("nh3", "cc-pvdz", true),
        ("nh3", "def2-svp", false),
    ] {
        let c = load_case(system, basis_name, &mut ck);
        let ctx = c.ctx.as_str();
        let r = &c.reference;
        let e2_ref = num(r, "/all_electron/mp3/e_mp2", ctx);
        let e3_ref = num(r, "/all_electron/mp3/e_mp3", ctx);
        assert_nonvacuous(ctx, "E_MP2", e2_ref);
        assert!(
            e3_ref.abs() > 1e-4,
            "{ctx}: reference E(3) {e3_ref} is vacuous"
        );
        let m = mp3_energy(
            &c.mol,
            &c.obs,
            &c.dfbs,
            Operator::coulomb(),
            &c.rhf,
            0,
            None,
        )
        .unwrap_or_else(|e| panic!("{ctx}: MP3 failed: {e:?}"));
        ck.close(ctx, "E_MP2", m.e_mp2, e2_ref, TOL_MP);
        ck.close(ctx, "E(3)", m.e_mp3, e3_ref, TOL_MP);
        ck.close(
            ctx,
            "E_MP2+E(3)",
            m.e_corr,
            num(r, "/all_electron/mp3/e_corr", ctx),
            TOL_MP,
        );
        let fc_ref = num(r, "/frozen_core/mp3/e_corr", ctx);
        ck.miss(ctx, "all-electron MP3 vs frozen-core ref", m.e_corr, fc_ref);
        if check_frozen {
            let mf = mp3_energy(
                &c.mol,
                &c.obs,
                &c.dfbs,
                Operator::coulomb(),
                &c.rhf,
                c.n_frozen,
                None,
            )
            .unwrap_or_else(|e| panic!("{ctx}: frozen-core MP3 failed: {e:?}"));
            ck.close(
                ctx,
                "E(3) frozen core",
                mf.e_mp3,
                num(r, "/frozen_core/mp3/e_mp3", ctx),
                TOL_MP,
            );
            ck.close(ctx, "E_MP3 corr frozen core", mf.e_corr, fc_ref, TOL_MP);
        }
    }
    ck.finish("MP3");
}

// ---------------------------------------------------------------------------
// RI-CCD (VALIDATION.md "RI-CCD")
// ---------------------------------------------------------------------------

#[test]
#[ignore = "validation: RI-CCD"]
fn ccd_matches_df_reference() {
    let mut ck = Checks::default();
    for (system, basis_name, check_frozen) in [
        ("h2o", "cc-pvdz", true),
        ("h2o", "def2-svp", false),
        ("nh3", "cc-pvdz", false),
        ("nh3", "def2-svp", false),
    ] {
        let c = load_case(system, basis_name, &mut ck);
        let ctx = c.ctx.as_str();
        let r = &c.reference;
        let e_ref = num(r, "/all_electron/ccd/e_corr", ctx);
        assert_nonvacuous(ctx, "E_CCD", e_ref);
        let res = ccd(
            &c.mol,
            &c.obs,
            &c.dfbs,
            Operator::coulomb(),
            &c.rhf,
            &cc_cfg(0),
        )
        .unwrap_or_else(|e| panic!("{ctx}: CCD failed: {e:?}"));
        assert!(res.t1.is_none(), "{ctx}: CCD returned T1 amplitudes");
        ck.close(ctx, "E_CCD", res.correlation_energy, e_ref, TOL_CC);
        ck.miss(
            ctx,
            "CCD vs CCSD ref",
            res.correlation_energy,
            num(r, "/all_electron/ccsd/e_corr", ctx),
        );
        ck.miss(
            ctx,
            "all-electron CCD vs frozen-core ref",
            res.correlation_energy,
            num(r, "/frozen_core/ccd/e_corr", ctx),
        );
        if check_frozen {
            let rf = ccd(
                &c.mol,
                &c.obs,
                &c.dfbs,
                Operator::coulomb(),
                &c.rhf,
                &cc_cfg(c.n_frozen),
            )
            .unwrap_or_else(|e| panic!("{ctx}: frozen-core CCD failed: {e:?}"));
            ck.close(
                ctx,
                "E_CCD frozen core",
                rf.correlation_energy,
                num(r, "/frozen_core/ccd/e_corr", ctx),
                TOL_CC,
            );
        }
    }
    ck.finish("RI-CCD");
}

// ---------------------------------------------------------------------------
// RI-CCSD (VALIDATION.md "RI-CCSD spin-orbital" and "spin-adapted")
// ---------------------------------------------------------------------------

/// Shared CCSD comparisons and negative controls for one driver's result.
fn check_ccsd(ck: &mut Checks, c: &Case, label: &str, e: f64) {
    let ctx = c.ctx.as_str();
    let r = &c.reference;
    let e_ref = num(r, "/all_electron/ccsd/e_corr", ctx);
    assert_nonvacuous(ctx, "E_CCSD", e_ref);
    ck.close(ctx, &format!("E_CCSD {label}"), e, e_ref, TOL_CC);
    ck.miss(
        ctx,
        &format!("{label} CCSD vs frozen-core ref"),
        e,
        num(r, "/frozen_core/ccsd/e_corr", ctx),
    );
    ck.miss(
        ctx,
        &format!("{label} CCSD vs exact-integral CCSD"),
        e,
        num(r, "/conventional_exact_integrals/ccsd_e_corr", ctx),
    );
    if let Some(e_ccd) = r
        .pointer("/all_electron/ccd/e_corr")
        .and_then(Value::as_f64)
    {
        ck.miss(ctx, &format!("{label} CCSD vs CCD ref"), e, e_ccd);
    }
}

const CCSD_SYSTEMS: [(&str, &str); 6] = [
    ("h2o", "cc-pvdz"),
    ("h2o", "def2-svp"),
    ("nh3", "cc-pvdz"),
    ("nh3", "def2-svp"),
    ("hcn", "cc-pvdz"),
    ("h2o", "aug-cc-pvdz"),
];

#[test]
#[ignore = "validation: RI-CCSD spin-orbital"]
fn ccsd_spin_orbital_matches_df_reference() {
    let mut ck = Checks::default();
    for (system, basis_name) in CCSD_SYSTEMS {
        let c = load_case(system, basis_name, &mut ck);
        let res = run_ccsd_so(&c, 0);
        check_ccsd(&mut ck, &c, "spin-orbital", res.correlation_energy);
        if (system, basis_name) == ("h2o", "cc-pvdz") {
            let rf = run_ccsd_so(&c, c.n_frozen);
            ck.close(
                &c.ctx,
                "E_CCSD so frozen core",
                rf.correlation_energy,
                num(&c.reference, "/frozen_core/ccsd/e_corr", &c.ctx),
                TOL_CC,
            );
        }
    }
    ck.finish("RI-CCSD spin-orbital");
}

#[test]
#[ignore = "validation: RI-CCSD spin-adapted"]
fn ccsd_spin_adapted_matches_df_reference() {
    let mut ck = Checks::default();
    for (system, basis_name) in CCSD_SYSTEMS.iter().copied().chain([("c2h6", "cc-pvdz")]) {
        let c = load_case(system, basis_name, &mut ck);
        let res = run_ccsd_sa(&c, 0);
        check_ccsd(&mut ck, &c, "spin-adapted", res.correlation_energy);
        if (system, basis_name) == ("h2o", "cc-pvdz") {
            let rf = run_ccsd_sa(&c, c.n_frozen);
            ck.close(
                &c.ctx,
                "E_CCSD sa frozen core",
                rf.correlation_energy,
                num(&c.reference, "/frozen_core/ccsd/e_corr", &c.ctx),
                TOL_CC,
            );
        }
        // Internal cross-check: the spin-orbital driver on the same system,
        // integrals and aux — two independent ferric CCSD implementations.
        if matches!(
            (system, basis_name),
            ("h2o", "cc-pvdz") | ("hcn", "cc-pvdz")
        ) {
            let so = run_ccsd_so(&c, 0);
            ck.close(
                &c.ctx,
                "CCSD sa vs so (ferric)",
                res.correlation_energy,
                so.correlation_energy,
                TOL_INTERNAL,
            );
        }
    }
    ck.finish("RI-CCSD spin-adapted");
}

// ---------------------------------------------------------------------------
// CCSD(T) (VALIDATION.md "CCSD(T)" and "spin-adapted (T)")
// ---------------------------------------------------------------------------

fn check_t(ck: &mut Checks, c: &Case, label: &str, e_t: f64) {
    let ctx = c.ctx.as_str();
    let r = &c.reference;
    let t_ref = num(r, "/all_electron/ccsd_t/e_t", ctx);
    assert!(
        t_ref < -1e-4,
        "{ctx}: reference (T) = {t_ref:.12} is not a plausible triples energy"
    );
    ck.close(ctx, &format!("E_(T) {label}"), e_t, t_ref, TOL_T);
    // Frozen-core (T) differs from all-electron by only ~2e-5 Ha on
    // H2O/cc-pVDZ — above MUST_MISS but the closest control in this file.
    ck.miss(
        ctx,
        &format!("{label} (T) vs frozen-core (T) ref"),
        e_t,
        num(r, "/frozen_core/ccsd_t/e_t", ctx),
    );
}

#[test]
#[ignore = "validation: CCSD(T)"]
fn ccsd_t_spin_orbital_matches_df_reference() {
    let mut ck = Checks::default();
    for (system, basis_name) in [
        ("h2o", "cc-pvdz"),
        ("hcn", "cc-pvdz"),
        ("h2o", "aug-cc-pvdz"),
    ] {
        let c = load_case(system, basis_name, &mut ck);
        let cc = run_ccsd_so(&c, 0);
        check_ccsd(&mut ck, &c, "spin-orbital", cc.correlation_energy);
        let e_t = ccsd_t(
            &c.mol,
            &c.obs,
            &c.dfbs,
            Operator::coulomb(),
            &c.rhf,
            &cc,
            &cc_cfg(0),
        )
        .unwrap_or_else(|e| panic!("{}: spin-orbital (T) failed: {e:?}", c.ctx));
        check_t(&mut ck, &c, "spin-orbital", e_t);
        if (system, basis_name) == ("h2o", "cc-pvdz") {
            let ccf = run_ccsd_so(&c, c.n_frozen);
            let e_tf = ccsd_t(
                &c.mol,
                &c.obs,
                &c.dfbs,
                Operator::coulomb(),
                &c.rhf,
                &ccf,
                &cc_cfg(c.n_frozen),
            )
            .unwrap_or_else(|e| panic!("{}: frozen-core (T) failed: {e:?}", c.ctx));
            ck.close(
                &c.ctx,
                "E_(T) so frozen core",
                e_tf,
                num(&c.reference, "/frozen_core/ccsd_t/e_t", &c.ctx),
                TOL_T,
            );
        }
    }
    ck.finish("CCSD(T) spin-orbital");
}

#[test]
#[ignore = "validation: spin-adapted (T)"]
fn ccsd_t_spin_adapted_matches_df_reference() {
    let mut ck = Checks::default();
    for (system, basis_name) in [
        ("h2o", "cc-pvdz"),
        ("hcn", "cc-pvdz"),
        ("h2o", "aug-cc-pvdz"),
        ("c2h6", "cc-pvdz"),
    ] {
        let c = load_case(system, basis_name, &mut ck);
        let cc = run_ccsd_sa(&c, 0);
        check_ccsd(&mut ck, &c, "spin-adapted", cc.correlation_energy);
        let e_t = ccsd_t_closed_shell(
            &c.mol,
            &c.obs,
            &c.dfbs,
            Operator::coulomb(),
            &c.rhf,
            &cc,
            &cc_cfg(0),
        )
        .unwrap_or_else(|e| panic!("{}: spin-adapted (T) failed: {e:?}", c.ctx));
        check_t(&mut ck, &c, "spin-adapted", e_t);
        if (system, basis_name) == ("h2o", "cc-pvdz") {
            let ccf = run_ccsd_sa(&c, c.n_frozen);
            let e_tf = ccsd_t_closed_shell(
                &c.mol,
                &c.obs,
                &c.dfbs,
                Operator::coulomb(),
                &c.rhf,
                &ccf,
                &cc_cfg(c.n_frozen),
            )
            .unwrap_or_else(|e| panic!("{}: frozen-core spin-adapted (T) failed: {e:?}", c.ctx));
            ck.close(
                &c.ctx,
                "E_(T) sa frozen core",
                e_tf,
                num(&c.reference, "/frozen_core/ccsd_t/e_t", &c.ctx),
                TOL_T,
            );
        }
        // Internal cross-check: the fully independent spin-orbital pipeline
        // (its own CCSD amplitudes, its own (T) kernel) on the same system.
        if (system, basis_name) == ("h2o", "aug-cc-pvdz") {
            let so = run_ccsd_so(&c, 0);
            let e_t_so = ccsd_t(
                &c.mol,
                &c.obs,
                &c.dfbs,
                Operator::coulomb(),
                &c.rhf,
                &so,
                &cc_cfg(0),
            )
            .unwrap_or_else(|e| panic!("{}: spin-orbital (T) failed: {e:?}", c.ctx));
            ck.close(&c.ctx, "(T) sa vs so (ferric)", e_t, e_t_so, TOL_INTERNAL);
        }
    }
    ck.finish("spin-adapted (T)");
}
