//! VALIDATION tier — validation page row "UCCSD / UCCSD(T)".
//!
//! Runs only in the `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-cc --test validation_uccsd \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's spin-blocked UCCSD and UCCSD(T) as the CLI runs them for
//! `method.kind = "ccsd"` / `"ccsd(t)"` on a molecule with multiplicity > 1:
//! a UHF reference (`solve_uhf`, exact four-centre J/K), then
//! `ferric_cc::u_ccsd::u_ccsd` and `ferric_cc::u_ccsd_t::u_ccsd_t` with the
//! cc-pvdz-ri correlation aux and `frozen_core` 0 or 1 (per spin). Compared
//! against PySCF `cc.dfuccsd.UCCSD` (DF in every block, vvvv included, on
//! ferric's aux JSON) and its `ccsd_t()` on a stability-checked
//! exact-integral UHF, both fed ferric's basis JSON and Bohr geometry:
//!
//! | system | state | frozen core |
//! |---|---|---|
//! | OH | ²Π doublet | 0 and 1 |
//! | NH2 | ²B1 doublet | 0 and 1 |
//! | O2 | ³Σg⁻ triplet | 0 and 1 |
//!
//! Quantities: E_UHF, E_CCSD and its aa / ab / bb split, E_(T) and its four
//! occupied spin classes. PySCF reports only totals; the generator splits
//! them (CCSD from PySCF's amplitudes on numpy DF blocks, (T) by the
//! per-class formulas of `uccsd_t_slow`) and anchors each split to PySCF's
//! total at 1e-10 Ha (`numpy_anchor_*`, re-checked here).
//!
//! References: `scripts/validation/gen_uccsd.py` →
//! `testdata/reference/validation/uccsd/<system>_cc-pvdz.json`.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * Right algebra: every number agrees to the SCF/CC convergence floor.
//!   The remaining differences are the integral engine, the convergence
//!   of both codes and PySCF's off-diagonal MO Fock (`fock_offdiag_max`,
//!   ~1e-9 at these SCF thresholds, entering at second order).
//! * A wrong residual term, a spin label slip or a wrong (T) weight moves
//!   an energy by 1e-5 Ha or more on these systems, 100x above the bar.
//! * A harness slip (state, frozen core, aux): E_UHF is checked first, and
//!   the frozen-core controls below must miss.
//!
//! # TOLERANCES
//!
//! Set from the measured maxima recorded on each const.
//!
//! # NEGATIVE CONTROLS (asserted)
//!
//! * frozen core 1 must miss the frozen-core-0 reference and vice versa;
//! * E_aa must miss the reference E_bb (N_alpha != N_beta).

use std::path::{Path, PathBuf};

use ferric_cc::u_ccsd::{u_ccsd, UCcsdResult};
use ferric_cc::u_ccsd_t::{u_ccsd_t, UTriples};
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::{ScfResult, Spin};
use ferric_testkit::reference::num_ctx;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/uccsd";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS: &str = "cc-pvdz";
const AUX: &str = "cc-pvdz-ri";

const TOL_ENUC: f64 = 1e-9;
/// E_UHF vs the stability-followed PySCF UHF (exact J/K both sides).
const TOL_E_SCF: f64 = 3e-11;
/// Every CCSD energy and spin block. Measured max 3.4e-11 (NH2, E_CCSD);
/// PySCF stops at an amplitude norm of 1e-10, so the bar sits above that.
const TOL_E_CC: f64 = 2e-10;
/// E_(T) and each class. Measured max 2.2e-12 (NH2 fc1, E_(T)).
const TOL_E_T: f64 = 2e-11;
/// The generator's split-vs-total anchor bar.
const TOL_NUMPY_ANCHOR: f64 = 1e-10;
const MUST_MISS_FACTOR: f64 = 100.0;

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

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_uccsd.py — a missing reference is a failure, never a skip",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<10} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs reference {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn check_miss(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    let must = MUST_MISS_FACTOR * tol;
    eprintln!("{ctx}: control {what:<28} |d| {d:.2e} (must exceed {must:.0e})");
    assert!(
        d > must,
        "{ctx}: negative control '{what}' did not miss: |d| {d:.2e} <= {must:.0e}"
    );
}

struct Sys {
    label: String,
    r: Value,
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
}

fn load(system: &str) -> Sys {
    let r = reference(system);
    let label = format!("{system}/{BASIS}");
    assert_eq!(r["basis"].as_str(), Some(BASIS), "{label}: basis");
    assert_eq!(r["aux_basis"].as_str(), Some(AUX), "{label}: aux basis");
    let charge = r["charge"].as_i64().expect("charge") as i32;
    let mult = r["multiplicity"].as_u64().expect("multiplicity") as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), charge, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    check_close(
        &label,
        "E_nuc",
        mol.nuclear_repulsion(),
        num_ctx(&r, "/nuclear_repulsion", &label),
        TOL_ENUC,
    );
    let obs = PreparedBasis::new(&mol, &basis::bundled(BASIS).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    assert_eq!(
        obs.nbasis() as u64,
        r["nao"].as_u64().unwrap(),
        "{label}: AO count"
    );
    assert_eq!(
        dfbs.nbasis() as u64,
        r["naux"].as_u64().unwrap(),
        "{label}: aux count"
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    Sys {
        label,
        r,
        mol,
        obs,
        dfbs,
        bounds,
    }
}

/// The CLI's UHF (exact J/K) with the stability descent that pins it to
/// PySCF's stability-followed state (as validation_u_rimp2.rs).
fn uhf(sys: &Sys) -> ScfResult {
    let e_ref = num_ctx(&sys.r, "/uhf/energy", &sys.label);
    let base = RhfConfig::default()
        .with_max_iter(400)
        .with_energy_conv(1e-11)
        .with_density_conv(1e-9)
        .with_stability(ferric_scf::options::Stability::Descend);
    let steered = base.clone().with_level_shift(0.5).with_mom_after_iter(5);
    let mut tried = Vec::new();
    for (name, cfg) in [("stability-descent", base), ("ls0.5+MOM5", steered)] {
        let res = solve_uhf(
            &ParallelContext::default(),
            &sys.mol,
            &sys.obs,
            &sys.bounds,
            &cfg,
        )
        .unwrap_or_else(|e| panic!("{}: solve_uhf failed: {e:?}", sys.label));
        let d = (res.energy - e_ref).abs();
        tried.push(format!("{name}: E {:.10} |d| {d:.2e}", res.energy));
        if res.converged && d < TOL_E_SCF {
            assert!(matches!(res.spin, Spin::Unrestricted));
            let st = res.stability.as_ref().expect("stability analysis");
            assert!(
                matches!(
                    st.verdict(),
                    StabilityVerdict::Stable | StabilityVerdict::Marginal
                ),
                "{}: ferric UHF state not stable: {}",
                sys.label,
                st.summary()
            );
            check_close(&sys.label, "E_UHF", res.energy, e_ref, TOL_E_SCF);
            return res;
        }
    }
    panic!(
        "{}: ferric did not reach the reference UHF state {e_ref:.10}: {tried:?}",
        sys.label
    );
}

fn run(sys: &Sys, scf: &ScfResult, frozen_core: usize) -> (UCcsdResult, UTriples) {
    let cfg = CcConfig {
        frozen_core,
        energy_conv: 1e-11,
        max_iter: 300,
        ..Default::default()
    };
    let op = Operator::coulomb();
    let cc = u_ccsd(&sys.mol, &sys.obs, &sys.dfbs, op, scf, &cfg)
        .unwrap_or_else(|e| panic!("{}: u_ccsd(fc {frozen_core}): {e:?}", sys.label));
    let t = u_ccsd_t(&sys.mol, &sys.obs, &sys.dfbs, op, scf, &cc, &cfg)
        .unwrap_or_else(|e| panic!("{}: u_ccsd_t(fc {frozen_core}): {e:?}", sys.label));
    (cc, t)
}

fn check_block(sys: &Sys, frozen_core: usize, cc: &UCcsdResult, t: &UTriples) {
    let block = format!("uccsd_fc{frozen_core}");
    let ctx = format!("{} {block}", sys.label);
    let r = &sys.r;
    assert_eq!(r[&block]["frozen_core"].as_u64(), Some(frozen_core as u64));
    for key in ["numpy_anchor_ccsd_diff", "numpy_anchor_t_diff"] {
        let a = num_ctx(r, &format!("/{block}/{key}"), &ctx);
        assert!(a.abs() < TOL_NUMPY_ANCHOR, "{ctx}: generator {key} {a:.2e}");
    }
    let want = |k: &str| num_ctx(r, &format!("/{block}/{k}"), &ctx);
    check_close(
        &ctx,
        "E_CCSD",
        cc.correlation_energy,
        want("e_corr"),
        TOL_E_CC,
    );
    check_close(&ctx, "E_aa", cc.e_aa, want("e_aa"), TOL_E_CC);
    check_close(&ctx, "E_ab", cc.e_ab, want("e_ab"), TOL_E_CC);
    check_close(&ctx, "E_bb", cc.e_bb, want("e_bb"), TOL_E_CC);
    check_close(&ctx, "E_(T)", t.e_t, want("e_t"), TOL_E_T);
    check_close(&ctx, "(T) aaa", t.aaa, want("e_t_aaa"), TOL_E_T);
    check_close(&ctx, "(T) aab", t.aab, want("e_t_aab"), TOL_E_T);
    check_close(&ctx, "(T) abb", t.abb, want("e_t_abb"), TOL_E_T);
    check_close(&ctx, "(T) bbb", t.bbb, want("e_t_bbb"), TOL_E_T);
    check_miss(
        &ctx,
        "E_aa vs reference E_bb",
        cc.e_aa,
        want("e_bb"),
        TOL_E_CC,
    );
}

fn uccsd_case(system: &str) {
    let sys = load(system);
    let scf = uhf(&sys);
    let (cc0, t0) = run(&sys, &scf, 0);
    check_block(&sys, 0, &cc0, &t0);
    let (cc1, t1) = run(&sys, &scf, 1);
    check_block(&sys, 1, &cc1, &t1);
    let r = &sys.r;
    check_miss(
        &sys.label,
        "fc1 E_CCSD vs fc0 reference",
        cc1.correlation_energy,
        num_ctx(r, "/uccsd_fc0/e_corr", &sys.label),
        TOL_E_CC,
    );
    check_miss(
        &sys.label,
        "fc0 E_CCSD vs fc1 reference",
        cc0.correlation_energy,
        num_ctx(r, "/uccsd_fc1/e_corr", &sys.label),
        TOL_E_CC,
    );
}

#[test]
#[ignore = "validation: UCCSD / UCCSD(T)"]
fn uccsd_oh_ccpvdz_vs_pyscf_dfuccsd() {
    uccsd_case("oh");
}

#[test]
#[ignore = "validation: UCCSD / UCCSD(T)"]
fn uccsd_nh2_ccpvdz_vs_pyscf_dfuccsd() {
    uccsd_case("nh2");
}

#[test]
#[ignore = "validation: UCCSD / UCCSD(T)"]
fn uccsd_o2_triplet_ccpvdz_vs_pyscf_dfuccsd() {
    uccsd_case("o2");
}
