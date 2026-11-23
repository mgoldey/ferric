//! Unrestricted OO-RI-MP2 with a frozen core.
//!
//! The lowest `frozen_core` orbitals of each spin are left out of the MP2
//! sum and are not rotated: only active-occupied/virtual rotations are
//! optimized, as in the closed-shell `oo_ri_mp2`. The kernel returns the
//! active columns; `u_gradient_solver_frame` places them at MO columns
//! `first_occ..nocc_total` before adding the UHF and Fock-response terms.
//!
//! Checks:
//! 1. Closed-shell limit (exactness anchor). H2O / 6-31G: the RHF, relabelled
//!    as an unrestricted reference with `C_α = C_β`, run through
//!    `u_oo_ri_mp2` with frozen core 1 equals `oo_ri_mp2` with frozen core 1.
//!    Both start from the same orbitals, so the same core orbital is frozen.
//! 2. External reference. OH and NH2 doublets / 6-31G (cc-pvdz-ri): the total
//!    energy equals an independent numpy U-OO-RI-MP2 on PySCF integrals
//!    (`scripts/validation/gen_u_oo_rimp2_frozen_core.py`; no external code
//!    has this functional, see the generator header), at frozen core 1 and 0.
//!    The plain U-RI-MP2 energy at the UHF point is checked against PySCF
//!    DFUMP2(frozen = 1), which fixes the frozen-core definition.
//!
//! Tolerances.
//! * Convergence. Each optimizer stops at `grad_conv` = 1e-8 (gradient norm).
//!   At a minimum the energy error is at most ‖g‖²/(2λ_min). The lowest
//!   orbital-Hessian eigenvalue of these molecules is above 0.05 Ha, so the
//!   error is below 1e-15 Ha. The numpy reference reached max |dE/dκ| ≤ 1e-10.
//! * Frozen core orbital. The energy is NOT stationary in the frozen core
//!   orbital, so a difference δθ in the reference core orbital enters at
//!   first order, |∂E/∂θ_core| δθ. Check 1 starts both runs from one set of
//!   orbitals, δθ = 0. In check 2 ferric's and PySCF's UHFs are separate
//!   calculations; with density_conv 1e-10 and a core-valence gap above
//!   15 Ha, δθ ≲ 1e-11, and |∂E/∂θ_core| is below 1 Ha, giving ≲ 1e-11 Ha.
//! * Round-off. |E| ≈ 76 Ha; with f64 and sums over a few thousand terms the
//!   floor is ~1e-12 Ha.
//! Bars: check 1, 1e-10 Ha (100× round-off); check 2, 1e-10 Ha (10× the
//! frozen-orbital term). The frozen-core effect on the OO energy is above
//! 1e-4 Ha on every system (asserted; measured 9.6e-4 to 1.06e-3 Ha), so the
//! bars resolve it.
//!
//! The panics are caught and turned into assertion failures, so a mutation
//! that breaks the indexing fails on an assertion, not on a panic.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config};
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::u_oo_rimp2::{u_oo_ri_mp2, UOoRiMp2Config, UOoRiMp2Result};
use ferric_mp2::u_rimp2::u_ri_mp2;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::{ScfResult, Spin};
use serde_json::Value;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

const BASIS: &str = "6-31g";
const AUX: &str = "cc-pvdz-ri";
const ROW_DIR: &str = "testdata/reference/validation/u_oo_rimp2_frozen_core";
const MOL_DIR: &str = "testdata/molecules/validation";
/// Closed-shell limit, U-OO vs closed-shell OO (module doc).
const TOL_CLOSED_LIMIT: f64 = 1e-10;
/// U-OO total energy vs the numpy minimum (module doc).
const TOL_E_NUMPY: f64 = 1e-10;
/// Plain U-RI-MP2 E_corr vs PySCF DFUMP2 at the UHF point: the same
/// separate-UHF term as `TOL_E_NUMPY`.
const TOL_E_CORR_DFUMP2: f64 = 1e-10;
/// UHF energy vs PySCF (exact J/K on both sides).
const TOL_E_SCF: f64 = 1e-9;
/// The frozen-core effect the energy bar must resolve: measured 9.6e-4 (OH)
/// and 1.06e-3 Ha (NH2) on the numpy minimum.
const MIN_FROZEN_EFFECT: f64 = 1e-4;

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

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
}

fn load(system: &str, mult: usize) -> Sys {
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let obs = PreparedBasis::new(&mol, &basis::bundled(BASIS).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    Sys {
        mol,
        obs,
        dfbs,
        bounds,
    }
}

fn scf_config() -> RhfConfig {
    RhfConfig::default()
        .with_max_iter(400)
        .with_energy_conv(1e-12)
        .with_density_conv(1e-10)
}

fn u_oo_config(frozen_core: usize) -> UOoRiMp2Config {
    UOoRiMp2Config {
        frozen_core,
        grad_conv: 1e-8,
        energy_conv: 1e-13,
        max_iter: 300,
        ..UOoRiMp2Config::default()
    }
}

/// `u_oo_ri_mp2`, with a panic turned into an assertion failure.
fn run_u_oo(s: &Sys, reference: &ScfResult, frozen_core: usize, ctx: &str) -> UOoRiMp2Result {
    let cfg = u_oo_config(frozen_core);
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        u_oo_ri_mp2(
            &s.mol,
            &s.obs,
            &s.dfbs,
            Operator::coulomb(),
            &s.bounds,
            reference,
            &cfg,
            None,
        )
    }));
    assert!(
        outcome.is_ok(),
        "{ctx}: u_oo_ri_mp2 with frozen_core = {frozen_core} panicked"
    );
    let r = outcome
        .unwrap()
        .unwrap_or_else(|e| panic!("{ctx}: u_oo_ri_mp2 failed: {e}"));
    assert!(
        r.converged,
        "{ctx}: U-OO-RI-MP2 (frozen_core = {frozen_core}) not converged, |g| {:.2e}",
        r.grad_norm
    );
    r
}

fn check_close(ctx: &str, what: &str, got: f64, want: f64, tol: f64) {
    let d = (got - want).abs();
    eprintln!("{ctx}: {what:<34} {got:+.12} ref {want:+.12} |d| {d:.2e} (tol {tol:.0e})");
    assert!(
        d < tol,
        "{ctx}: {what} {got:.12} vs {want:.12} (|d| {d:.2e} >= {tol:.0e})"
    );
}

fn num(v: &Value, ptr: &str, ctx: &str) -> f64 {
    v.pointer(ptr)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("{ctx}: reference field {ptr} missing or not a number"))
}

/// The RHF as an unrestricted reference with `C_β = C_α`.
fn as_unrestricted(rhf: &ScfResult) -> ScfResult {
    let mut u = rhf.clone();
    u.spin = Spin::Unrestricted;
    u.mos_beta = Some(rhf.mos_alpha.clone());
    u.eps_beta = Some(rhf.eps_alpha.clone());
    u.density_alpha = 0.5 * &rhf.density_total;
    u.density_beta = Some(0.5 * &rhf.density_total);
    u.fock_beta = Some(rhf.fock_alpha.clone());
    u
}

#[test]
fn u_oo_rimp2_frozen_core_matches_closed_shell_in_the_closed_shell_limit() {
    let ctx = "h2o/6-31g";
    let s = load("h2o", 1);
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &s.mol,
        &s.obs,
        Operator::coulomb(),
        &s.bounds,
        &scf_config(),
    )
    .unwrap();
    assert!(rhf.converged, "{ctx}: RHF not converged");
    let oo_cfg = |frozen_core| OoRiMp2Config {
        frozen_core,
        grad_conv: 1e-8,
        energy_conv: 1e-13,
        max_iter: 300,
        ..OoRiMp2Config::default()
    };
    let closed = |fc| {
        let r = oo_ri_mp2(
            &s.mol,
            &s.obs,
            &s.dfbs,
            Operator::coulomb(),
            &s.bounds,
            &rhf,
            &oo_cfg(fc),
            None,
        )
        .unwrap();
        assert!(
            r.converged,
            "{ctx}: closed-shell OO (fc {fc}) not converged"
        );
        r.total_energy
    };
    let uref = as_unrestricted(&rhf);
    let e_closed_fc1 = closed(1);
    let e_closed_fc0 = closed(0);
    let u_fc1 = run_u_oo(&s, &uref, 1, ctx);
    let u_fc0 = run_u_oo(&s, &uref, 0, ctx);
    check_close(
        ctx,
        "U-OO fc1 vs closed-shell OO fc1",
        u_fc1.total_energy,
        e_closed_fc1,
        TOL_CLOSED_LIMIT,
    );
    check_close(
        ctx,
        "U-OO fc0 vs closed-shell OO fc0",
        u_fc0.total_energy,
        e_closed_fc0,
        TOL_CLOSED_LIMIT,
    );
    let effect = (e_closed_fc1 - e_closed_fc0).abs();
    assert!(
        effect > MIN_FROZEN_EFFECT,
        "{ctx}: frozen-core effect {effect:.2e} <= {MIN_FROZEN_EFFECT:.0e}"
    );
    // C_α = C_β is kept by the optimizer: the α and β same-spin energies agree.
    let c = &u_fc1.components;
    assert!(
        (c.e_aa - c.e_bb).abs() < TOL_CLOSED_LIMIT,
        "{ctx}: E_aa {:.12} != E_bb {:.12}",
        c.e_aa,
        c.e_bb
    );
}

fn check_vs_numpy(system: &str) {
    let ctx = format!("{system}/{BASIS}");
    let path = workspace_root()
        .join(ROW_DIR)
        .join(format!("{system}_{BASIS}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing reference {} ({e}); regenerate with \
             scripts/validation/gen_u_oo_rimp2_frozen_core.py",
            path.display()
        )
    });
    let r: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(r["basis"].as_str(), Some(BASIS), "{ctx}: basis");
    assert_eq!(r["aux_basis"].as_str(), Some(AUX), "{ctx}: aux");
    let mult = r["multiplicity"].as_u64().unwrap() as usize;
    let s = load(system, mult);
    check_close(
        &ctx,
        "E_nuc",
        s.mol.nuclear_repulsion(),
        num(&r, "/nuclear_repulsion", &ctx),
        1e-9,
    );
    let mut uhf_cfg = scf_config().with_check_stability(true);
    uhf_cfg.scf_stability_descent = true;
    let uhf = solve_uhf(
        &ParallelContext::default(),
        &s.mol,
        &s.obs,
        &s.bounds,
        &uhf_cfg,
    )
    .unwrap();
    assert!(uhf.converged, "{ctx}: UHF not converged");
    check_close(
        &ctx,
        "E_UHF vs PySCF",
        uhf.energy,
        num(&r, "/pyscf/e_scf", &ctx),
        TOL_E_SCF,
    );
    let mp2_fc1 = u_ri_mp2(
        &s.mol,
        &s.obs,
        &s.dfbs,
        Operator::coulomb(),
        &uhf,
        &RiMp2Config {
            frozen_core: 1,
            ..Default::default()
        },
    )
    .unwrap();
    check_close(
        &ctx,
        "U-RI-MP2 fc1 E_corr vs DFUMP2",
        mp2_fc1.mp2_corr,
        num(&r, "/numpy_oo/fc1/e_dfump2_corr_kappa0", &ctx),
        TOL_E_CORR_DFUMP2,
    );
    let mut e = [0.0; 2];
    for fc in [1usize, 0] {
        let key = format!("/numpy_oo/fc{fc}");
        let g_np = num(&r, &format!("{key}/max_orbital_gradient"), &ctx);
        assert!(g_np <= 1e-8, "{ctx}: numpy fc{fc} max|g| {g_np:.2e}");
        let anchor = num(&r, &format!("{key}/anchor_kappa0_diff"), &ctx);
        assert!(
            anchor.abs() <= 1e-10,
            "{ctx}: numpy fc{fc} anchor {anchor:.2e}"
        );
        let oo = run_u_oo(&s, &uhf, fc, &ctx);
        let want = num(&r, &format!("{key}/e_total"), &ctx);
        check_close(
            &ctx,
            &format!("U-OO fc{fc} total vs numpy"),
            oo.total_energy,
            want,
            TOL_E_NUMPY,
        );
        e[fc] = oo.total_energy;
    }
    let effect = (e[1] - e[0]).abs();
    assert!(
        effect > MIN_FROZEN_EFFECT,
        "{ctx}: frozen-core effect {effect:.2e} <= {MIN_FROZEN_EFFECT:.0e}"
    );
}

#[test]
fn u_oo_rimp2_frozen_core_oh_matches_numpy() {
    check_vs_numpy("oh");
}

#[test]
fn u_oo_rimp2_frozen_core_nh2_matches_numpy() {
    check_vs_numpy("nh2");
}
