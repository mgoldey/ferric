//! VALIDATION tier — VALIDATION.md row "Unrestricted RI-MP2".
//!
//! Runs only in the weekly `validation` CI job (and on demand):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo nextest run -p ferric-mp2 --test validation_u_rimp2 \
//!     --run-ignored only -E 'binary(/^validation_/)'
//! ```
//!
//! # What is compared
//!
//! ferric's unrestricted RI-MP2 as the CLI runs it for `method.kind = "rimp2"`
//! on a molecule with multiplicity > 1: a UHF reference (`solve_uhf`, exact
//! four-centre J/K — crates/ferric-cli/src/lib.rs `solve_open_shell_reference`),
//! then `ferric_mp2::u_rimp2::u_ri_mp2` (`run_u_rimp2`) with the correlation aux
//! and `[mp2] frozen_core` (default 0). Compared against PySCF 2.13
//! `mp.dfump2.DFUMP2` on a stability-checked exact-integral UHF, both fed
//! ferric's basis JSON, aux JSON and Bohr geometry:
//!
//! | system | state | frozen core |
//! |---|---|---|
//! | OH | ²Π doublet | 0 |
//! | CH3 (planar) | ²A2'' doublet | 0 |
//! | NH2 | ²B1 doublet | 0 |
//! | O2 | ³Σg⁻ triplet | 0 |
//! | HO2 | ²A'' doublet | 0 and 2 |
//!
//! all at cc-pVDZ with the cc-pvdz-ri aux (the CLI's default correlation aux
//! for cc-pVDZ).
//!
//! Quantities: E_UHF, E_corr, E_total, and the three spin blocks E_αα, E_ββ,
//! E_αβ (`URiMp2Components`). DFUMP2 reports only E_ss = E_αα + E_ββ and
//! E_os; the generator splits E_ss with an independent numpy build (PySCF
//! int3c2e/int2c2e, Cholesky-metric B) anchored to DFUMP2's E_ss and E_os at
//! 1e-11 Ha (`numpy_anchor_*`, re-checked here).
//!
//! References: `scripts/validation/gen_u_rimp2.py` →
//! `testdata/reference/validation/u_rimp2/<system>_cc-pvdz.json`.
//!
//! # Exactness anchor (asserted first, per system)
//!
//! E_nuc (geometry/units), the AO and aux counts, and E_UHF (the reference
//! STATE: pinned by energy against PySCF's stability-followed UHF, and ferric's
//! own stability verdict must be Stable or Marginal — OH's π hole is an exact
//! zero mode, which ferric reports as MARGINAL).
//!
//! # Physics hypothesis vs artifact hypothesis (per the Experimental Protocol)
//!
//! * If ferric's U-RI-MP2 is right: every spin block agrees at the integral
//!   floor the closed-shell RI-MP2 rows measured (≤1.3e-10 Ha), because both
//!   codes use the same aux, the same Coulomb metric and the same canonical UHF.
//! * If a spin block is wrong (αα/ββ swapped channel, a dropped same-spin
//!   exchange, the ½ vs ¼ prefactor, α ε in the β denominator): that block
//!   misses by 1e-3–1e-2 Ha while the others pass — the blocks are compared
//!   separately precisely so the failure names the block.
//! * If frozen core is mis-counted (per spin vs total, occupied vs orbitals):
//!   the HO2 frozen-core block misses by the core contribution (4.1e-3 Ha).
//! * If the UHF lands on a different state: E_UHF fails before any MP2 number.
//!
//! # TOLERANCES
//!
//! | quantity | measured max \|d\| | bar |
//! |---|---:|---:|
//! | E_UHF | 2.8e-12 (HO2) | [`TOL_E_SCF`] 3e-11 Ha |
//! | E_corr, E_αα, E_ββ, E_αβ, E_total | 9.2e-10 (HO2 E_corr); ≤ 5.8e-11 elsewhere | [`TOL_E_MP2`] 1e-8 Ha |
//!
//! # NEGATIVE CONTROLS (asserted)
//!
//! * Spin channels: every system here has N_α ≠ N_β, so E_αα ≠ E_ββ. ferric's
//!   E_αα must MISS the reference E_ββ (and vice versa) by >
//!   [`MUST_MISS_FACTOR`] × the bar — a comparison that swapped the channels
//!   would fail.
//! * Frozen core on vs off (HO2): ferric at frozen core 2 must match the
//!   `ump2_fc2` block and MISS the all-electron block, and ferric all-electron
//!   must MISS the `ump2_fc2` block (measured frozen-core effect 4.1e-3 Ha).
//!
//! # MUTATION (to run once, record the outcome here)
//!
//! In `crates/ferric-mp2/src/u_rimp2.rs` `same_spin_pair_kernel`, energy-only
//! branch (the branch `u_ri_mp2` reaches: `same_spin_pair_energy` passes
//! `want_amplitudes = false`), drop the same-spin exchange integral:
//!
//! ```text
//! let k = g_ab - g_ba;   ->   let k = g_ab;
//! ```
//!
//! Expected: E_αα and E_ββ fail for every system (by ~|E_ss|, 1e-2 Ha), E_αβ
//! still passes (it has no exchange), naming the defective blocks.
//! Outcome (2026-09-25): all 5 tests fail.
//!
//! A missing reference JSON is a HARD failure (panic naming the path).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::u_rimp2::{u_ri_mp2, URiMp2Result};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::{ScfResult, Spin};
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/u_rimp2";
const MOL_DIR: &str = "testdata/molecules/validation";
const BASIS: &str = "cc-pvdz";
const AUX: &str = "cc-pvdz-ri";

const TOL_ENUC: f64 = 1e-9;
/// E_UHF vs the stability-followed PySCF UHF (exact J/K both sides).
/// Measured ≤ 2.8e-12 Ha (HO2).
const TOL_E_SCF: f64 = 3e-11;
/// Every MP2 energy (E_corr, each spin block, E_total).
/// Measured ≤ 9.2e-10 Ha (HO2, the largest correlation energy here; the
/// others ≤ 5.8e-11).
const TOL_E_MP2: f64 = 1e-8;
/// The generator's numpy-vs-DFUMP2 anchor bar.
const TOL_NUMPY_ANCHOR: f64 = 1e-11;
/// A reference ferric must MISS is missed by at least this multiple of the bar.
const MUST_MISS_FACTOR: f64 = 100.0;

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
        .expect("ferric-mp2 manifest dir should be <root>/crates/ferric-mp2")
        .to_path_buf()
}

fn reference(system: &str) -> Value {
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_u_rimp2.py — a missing reference is a failure, never a skip",
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
    eprintln!("{ctx}: {what:<14} ferric {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
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
        "{ctx}: negative control '{what}' did not miss: |d| {d:.2e} <= {must:.0e} — the \
         comparison does not respond to what the control changes"
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
        num(&r, "/nuclear_repulsion", &label),
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

/// The CLI's UHF (exact J/K: no SCF aux for `rimp2`) plus the stability
/// descent that pins it to PySCF's stability-followed state; a level-shift +
/// MOM fallback as in validation_gw.rs. Accepted only if the energy matches.
fn uhf(sys: &Sys) -> ScfResult {
    let e_ref = num(&sys.r, "/uhf/energy", &sys.label);
    let base = RhfConfig {
        max_iter: 400,
        energy_conv: 1e-11,
        density_conv: 1e-9,
        check_stability: true,
        scf_stability_descent: true,
        ..Default::default()
    };
    assert!(
        base.df_j_aux.is_none() && base.df_k_aux.is_none(),
        "exact J/K, as the CLI's rimp2 reference"
    );
    let steered = RhfConfig {
        level_shift: 0.5,
        mom_after_iter: 5,
        ..base.clone()
    };
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
            assert!(
                matches!(res.spin, Spin::Unrestricted),
                "{}: not a UHF result",
                sys.label
            );
            let st = res
                .stability
                .as_ref()
                .unwrap_or_else(|| panic!("{}: no stability analysis", sys.label));
            assert!(
                matches!(
                    st.verdict(),
                    StabilityVerdict::Stable | StabilityVerdict::Marginal
                ),
                "{}: ferric UHF state not stable: {}",
                sys.label,
                st.summary()
            );
            eprintln!("{}: UHF state reached with {name}", sys.label);
            check_close(&sys.label, "E_UHF", res.energy, e_ref, TOL_E_SCF);
            return res;
        }
    }
    panic!(
        "{}: ferric did not reach the reference UHF state {e_ref:.10}: {tried:?}",
        sys.label
    );
}

fn run_ump2(sys: &Sys, scf: &ScfResult, frozen_core: usize) -> URiMp2Result {
    let cfg = RiMp2Config {
        frozen_core,
        ..Default::default()
    };
    u_ri_mp2(
        &sys.mol,
        &sys.obs,
        &sys.dfbs,
        Operator::coulomb(),
        scf,
        &cfg,
    )
    .unwrap_or_else(|e| panic!("{}: u_ri_mp2(fc {frozen_core}) failed: {e:?}", sys.label))
}

/// Compare one U-RI-MP2 result against one reference block, spin block by
/// spin block, after re-checking the generator's numpy anchor.
fn check_block(sys: &Sys, block: &str, res: &URiMp2Result, frozen_core: usize) {
    let ctx = format!("{} {block}", sys.label);
    let r = &sys.r;
    assert_eq!(
        r[block]["frozen_core"].as_u64(),
        Some(frozen_core as u64),
        "{ctx}: reference block frozen core"
    );
    for key in ["numpy_anchor_ss_diff", "numpy_anchor_os_diff"] {
        let a = num(r, &format!("/{block}/{key}"), &ctx);
        assert!(
            a.abs() < TOL_NUMPY_ANCHOR,
            "{ctx}: generator {key} {a:.2e} exceeds {TOL_NUMPY_ANCHOR:.0e}"
        );
    }
    let c = &res.components;
    check_close(
        &ctx,
        "E_aa",
        c.e_aa,
        num(r, &format!("/{block}/e_aa"), &ctx),
        TOL_E_MP2,
    );
    check_close(
        &ctx,
        "E_bb",
        c.e_bb,
        num(r, &format!("/{block}/e_bb"), &ctx),
        TOL_E_MP2,
    );
    check_close(
        &ctx,
        "E_ab",
        c.e_ab,
        num(r, &format!("/{block}/e_ab"), &ctx),
        TOL_E_MP2,
    );
    check_close(
        &ctx,
        "E_ss (DFUMP2)",
        c.e_aa + c.e_bb,
        num(r, &format!("/{block}/e_corr_ss"), &ctx),
        TOL_E_MP2,
    );
    check_close(
        &ctx,
        "E_os (DFUMP2)",
        c.e_ab,
        num(r, &format!("/{block}/e_corr_os"), &ctx),
        TOL_E_MP2,
    );
    check_close(
        &ctx,
        "E_corr",
        res.mp2_corr,
        num(r, &format!("/{block}/e_corr"), &ctx),
        TOL_E_MP2,
    );
    check_close(
        &ctx,
        "E_total",
        res.total_energy,
        num(r, &format!("/{block}/e_total"), &ctx),
        TOL_E_MP2,
    );
    // Internal consistency: the blocks ARE the correlation energy.
    assert!(
        (c.e_total - res.mp2_corr).abs() < 1e-14 * res.mp2_corr.abs().max(1.0),
        "{ctx}: components do not sum to mp2_corr"
    );
    // Control: αα and ββ are distinguishable (N_α ≠ N_β for every system).
    check_miss(
        &ctx,
        "E_aa vs reference E_bb",
        c.e_aa,
        num(r, &format!("/{block}/e_bb"), &ctx),
        TOL_E_MP2,
    );
    check_miss(
        &ctx,
        "E_bb vs reference E_aa",
        c.e_bb,
        num(r, &format!("/{block}/e_aa"), &ctx),
        TOL_E_MP2,
    );
}

fn u_rimp2_case(system: &str) {
    let sys = load(system);
    let scf = uhf(&sys);
    let res = run_ump2(&sys, &scf, 0);
    check_block(&sys, "ump2", &res, 0);
}

#[test]
#[ignore = "validation: Unrestricted RI-MP2"]
fn u_rimp2_oh_ccpvdz_vs_pyscf_dfump2() {
    u_rimp2_case("oh");
}

#[test]
#[ignore = "validation: Unrestricted RI-MP2"]
fn u_rimp2_ch3_ccpvdz_vs_pyscf_dfump2() {
    u_rimp2_case("ch3");
}

#[test]
#[ignore = "validation: Unrestricted RI-MP2"]
fn u_rimp2_nh2_ccpvdz_vs_pyscf_dfump2() {
    u_rimp2_case("nh2");
}

#[test]
#[ignore = "validation: Unrestricted RI-MP2"]
fn u_rimp2_o2_triplet_ccpvdz_vs_pyscf_dfump2() {
    u_rimp2_case("o2");
}

/// HO2 all-electron and frozen core 2, with the frozen-core controls.
#[test]
#[ignore = "validation: Unrestricted RI-MP2"]
fn u_rimp2_ho2_ccpvdz_all_electron_and_frozen_core_vs_pyscf_dfump2() {
    let sys = load("ho2");
    let scf = uhf(&sys);
    let ae = run_ump2(&sys, &scf, 0);
    check_block(&sys, "ump2", &ae, 0);
    let fc = run_ump2(&sys, &scf, 2);
    check_block(&sys, "ump2_fc2", &fc, 2);
    // Controls: frozen core on vs off must be distinguishable in both
    // directions (measured frozen-core effect 4.1e-3 Ha).
    check_miss(
        &sys.label,
        "fc2 E_corr vs all-electron ref",
        fc.mp2_corr,
        num(&sys.r, "/ump2/e_corr", &sys.label),
        TOL_E_MP2,
    );
    check_miss(
        &sys.label,
        "all-electron E_corr vs fc2 ref",
        ae.mp2_corr,
        num(&sys.r, "/ump2_fc2/e_corr", &sys.label),
        TOL_E_MP2,
    );
}
