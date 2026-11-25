//! VALIDATION tier: the analytic open-shell attenuated (erfc) RI-MP2 gradient
//! ([`ferric_mp2::att_rimp2_gradient::total_u_att_rimp2_gradient_with`])
//! against Richardson differences of PySCF's UHF + DF-UMP2 energy on an erfc
//! fit (`scripts/validation/gen_u_att_rimp2_gradient_pyscf.py`).
//!
//! ```text
//! cargo test --release -p ferric-mp2 --test validation_u_att_rimp2_gradient_pyscf -- --ignored
//! ```
//!
//! PySCF has no DF-UMP2 gradient and no attenuated MP2 gradient, so the
//! reference is a finite difference of PySCF's own energy: an independent
//! SCF, integral library, fit and correlation code. Systems: OH and NH2
//! doublets, cc-pVDZ / cc-pVDZ-RI, ω = 0.420 Å⁻¹, no frozen core, four-centre
//! UHF on both sides.
//!
//! # The bar (per coordinate)
//!
//! `R = (4 D(h) − D(2h)) / 3` from the reference's energies, h = 2e-3 Bohr.
//!
//! * truncation of `R`: `1e-2 · |D(2h) − D(h)| / 3`, the derivation of
//!   `u_rimp2_gradient_fd.rs`;
//! * PySCF's energy noise: `conv_tol_grad = 1e-11` bounds the orbital
//!   gradient, so the orbital error is `≲ 1e-11 / Δε_min (≈ 0.5) = 2e-11`
//!   and the correlation energy, linear in it, moves by `≲ ‖∂E_c/∂κ‖ · 2e-11
//!   ≲ 1e-10`; `R` carries `1.5 δE / h = 7.5e-8`;
//! * the two codes' energies differ by the integral libraries' accuracy,
//!   relative `≲ 1e-12` on `|E| ≈ 100` Ha, `≲ 1e-10` Ha, a smooth function of
//!   the geometry over `ℓ ≥ 0.1` Bohr, so its derivative is `≲ 1e-9`;
//! * ferric's analytic side: the production Z-vector residual, `2e-7`.
//!
//! `bar = 1e-2 · |D(2h) − D(h)| / 3 + 7.5e-8 + 1e-9 + 2e-7`.
//!
//! State check: ferric's total energy at the reference geometry must equal
//! PySCF's to `1e-8` Ha (the budget above gives `≲ 2e-10`; a different UHF
//! state is off by `≫ 1e-6`).

use std::path::{Path, PathBuf};

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_mp2::att_rimp2_gradient::total_u_att_rimp2_gradient_with;
use ferric_mp2::attenuated::{AttenuatedMp2Config, BOHR_INV_PER_ANG_INV};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::scf_jk::ScfJk;
use ferric_scf::warm_start::ScfWarmStart;
use ferric_testkit::reference::num;
use serde_json::Value;

const ROW_DIR: &str = "testdata/reference/validation/u_att_rimp2_gradient_pyscf";
const MOL_DIR: &str = "testdata/molecules/validation";
const NOISE: f64 = 7.5e-8;
const CODES: f64 = 1e-9;
const ANALYTIC: f64 = 2e-7;
const STATE_BAR: f64 = 1e-8;

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
        .join(format!("{system}_cc-pvdz.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing validation reference {} ({e}); regenerate with \
             scripts/validation/gen_u_att_rimp2_gradient_pyscf.py",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn check(system: &str) {
    let r = reference(system);
    let mult = r["multiplicity"].as_u64().unwrap() as usize;
    let xyz = workspace_root().join(MOL_DIR).join(format!("{system}.xyz"));
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, mult)
        .unwrap_or_else(|e| panic!("{}: {e:?}", xyz.display()));
    let omega = num(&r, "/omega_bohr_inv");
    assert!(
        (omega - 0.420 * BOHR_INV_PER_ANG_INV).abs() < 1e-12,
        "omega {omega}"
    );
    let config = AttenuatedMp2Config {
        omega,
        scaling: 1.0,
        frozen_core: 0,
        screen_thresh: None,
        memory_budget_bytes: None,
    };
    let (e, g) = total_u_att_rimp2_gradient_with(
        &mol,
        &basis::bundled("cc-pvdz").unwrap(),
        &basis::bundled("cc-pvdz-ri").unwrap(),
        &config,
        None,
        &ScfJk::exact(),
        &mut ScfWarmStart::new(false),
        RhfConfig::default()
            .with_density_conv(1e-12)
            .with_energy_conv(1e-13)
            .with_max_iter(300),
    )
    .unwrap();
    let e_ref = num(&r, "/energy_total");
    eprintln!(
        "{system}: E ferric {e:.12} PySCF {e_ref:.12} diff {:.3e}",
        e - e_ref
    );
    assert!(
        (e - e_ref).abs() < STATE_BAR,
        "{system}: energy {e:.12} vs {e_ref:.12}"
    );
    let mut worst = 0.0f64;
    for row in r["fd"].as_array().unwrap() {
        let atom = row["atom"].as_u64().unwrap() as usize;
        let c = row["coord"].as_u64().unwrap() as usize;
        let d1 = num(row, "/d_h");
        let d2 = num(row, "/d_2h");
        let rich = num(row, "/richardson");
        let bar = 1e-2 * (d2 - d1).abs() / 3.0 + NOISE + CODES + ANALYTIC;
        let err = (g[(atom, c)] - rich).abs();
        eprintln!(
            "{system} atom {atom} coord {c}: ferric {:+.10e} PySCF R {rich:+.10e} |diff| \
             {err:.3e} bar {bar:.3e}",
            g[(atom, c)]
        );
        assert!(
            err < bar,
            "{system} atom {atom} coord {c}: |g - R| = {err:.3e} >= bar {bar:.3e}"
        );
        worst = worst.max(err / bar);
    }
    eprintln!("{system}: worst |g - R| / bar = {worst:.3}");
}

#[test]
#[ignore = "validation: open-shell att-RI-MP2 gradient vs PySCF"]
fn u_att_rimp2_gradient_oh_vs_pyscf_fd() {
    check("oh");
}

#[test]
#[ignore = "validation: open-shell att-RI-MP2 gradient vs PySCF"]
fn u_att_rimp2_gradient_nh2_vs_pyscf_fd() {
    check("nh2");
}
