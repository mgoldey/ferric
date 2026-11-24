//! The terfc RI-MP2 energies and analytic gradients on the term-list
//! McMurchie-Davidson kernels (production) against the same quantities on
//! the reference kernels they replaced
//! (`Engine::debug_use_reference_md_kernels(true)`).
//!
//! The kernels compute the same finite sums in a different association
//! (`crates/ferric-integrals/tests/terfc_md_terms_vs_reference.rs`: at most
//! 5e-15 absolute per integral, ≤ 1.4e-15 of the tensor maximum). Nothing
//! else changes: the Coulomb RHF, the Z-vector and the four-centre term do
//! not touch the table engine, so both runs share the reference bit for bit
//! and the gradients differ only through the terf/terfc integrals.
//!
//! # Bar (derived)
//!
//! The gradient is linear in the 3c/2c integrals and their derivatives at a
//! fixed reference, with weights (amplitude densities, `Γ`) of order one.
//! Each integral moves by rounding, `≲ ε` of its size, with no preferred
//! sign, so the sum over `N` weighted integrals moves by `~ ε √N max|Γ dI|`.
//! For the largest case here (benzene / 6-31G + cc-pVDZ-RI,
//! `N ≈ naux · nbf² ≈ 2e6`) that is `2e-16 · 1.4e3 ≈ 3e-13` Ha/Bohr; the
//! Z-vector solve passes it on with the conditioning of the orbital
//! Hessian. The bar is `1e-12` Ha/Bohr per component. The energy is a sum
//! of the same kind over `B[P,ia]` and moves by less than one unit in the
//! last place of `|E| ≈ 1e2`; its bar is `1e-12` Ha. Measured: `|ΔE| = 0`
//! and `max|Δg|` 4.7e-15 (H2O), 7.5e-15 (CH3F), 2.0e-13 (benzene); both
//! paths are deterministic (a repeated run is bit-identical). A kernel error
//! (a wrong term, sign or offset) moves the integrals by `O(1)` of their
//! size and the gradient by `1e-4` or more.
//!
//! The switch is process-wide, so this file holds a single test.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_mp2::att_rimp2_gradient::{
    total_terfc_rimp2_gradient_on, total_u_terfc_rimp2_gradient_with,
};
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::scs::ScsMp2TerfcConfig;
use ferric_mp2::scs_2terfc_gradient::total_scs_mp2_2terfc_gradient_on;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::scf_jk::{ScfJk, ScfJkMode};
use ferric_scf::warm_start::ScfWarmStart;
use ndarray::Array2;
use std::path::Path;

const GRAD_BAR: f64 = 1e-12;
const ENERGY_BAR: f64 = 1e-12;
const R0: f64 = 1.05 / 0.529_177_210_92;

const WATER: &str = "3\nwater, off C2v\nO 0.02 -0.01 0.117790\nH 0.0 0.785453 -0.451161\nH 0.03 -0.735453 -0.491161\n";
const CH3F: &str = "5\nfluoromethane, distorted\nC 0.00 0.01 -0.63\nF 0.02 0.00 0.75\nH 1.03 0.02 -0.99\nH -0.52 0.90 -0.97\nH -0.50 -0.88 -1.01\n";
const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";

fn ensure_tables() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let probe = "16_4_2.bin";
        if let Ok(d) = std::env::var("FERRIC_TERF_TABLE_DIR") {
            assert!(
                Path::new(&d).join(probe).is_file(),
                "FERRIC_TERF_TABLE_DIR={d} has no {probe}"
            );
            return;
        }
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../terf-tables");
        assert!(
            repo.join(probe).is_file(),
            "terfc tables not found at {}",
            repo.display()
        );
        std::env::set_var("FERRIC_TERF_TABLE_DIR", &repo);
    });
}

fn exact() -> ScfJk {
    ScfJk {
        mode: ScfJkMode::Exact,
        aux: None,
    }
}

fn rijk() -> ScfJk {
    ScfJk {
        mode: ScfJkMode::RiJk,
        aux: Some("def2-universal-jkfit".into()),
    }
}

fn bs(name: &str) -> BasisSet {
    basis::bundled(name).unwrap()
}

fn closed(
    xyz: &'static str,
    obs: &'static str,
    fc: usize,
    jk: ScfJk,
) -> impl Fn() -> (f64, Array2<f64>) {
    move || {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let ri = RiMp2Config {
            frozen_core: fc,
            ..Default::default()
        };
        total_terfc_rimp2_gradient_on(
            &mol,
            &bs(obs),
            &bs("cc-pvdz-ri"),
            Operator::terfc(R0),
            &ri,
            None,
            &jk,
            &mut ScfWarmStart::new(false),
        )
        .unwrap()
    }
}

fn compare(label: &str, eval: &dyn Fn() -> (f64, Array2<f64>)) {
    Engine::debug_use_reference_md_kernels(true);
    let (e_ref, g_ref) = eval();
    Engine::debug_use_reference_md_kernels(false);
    let (e_new, g_new) = eval();
    let gmax = g_ref.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let dg = (&g_new - &g_ref)
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    let de = (e_new - e_ref).abs();
    eprintln!("{label}: |dE| {de:.2e} Ha, max|dg| {dg:.2e} Ha/Bohr (max|g| {gmax:.2e})");
    assert!(
        gmax > 1e-4,
        "{label}: gradient {gmax:.2e} too small to see a kernel error"
    );
    assert!(de <= ENERGY_BAR, "{label}: energy moved by {de:.3e} Ha");
    assert!(
        dg <= GRAD_BAR,
        "{label}: gradient moved by {dg:.3e} Ha/Bohr (bar {GRAD_BAR:.0e})"
    );
}

#[test]
fn terfc_gradients_on_term_list_kernels_match_the_reference_kernels() {
    ensure_tables();
    let benzene = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/molecules/benzene.xyz"),
    )
    .unwrap();
    let benzene: &'static str = Box::leak(benzene.into_boxed_str());
    compare(
        "H2O / cc-pVDZ, exact J/K",
        &closed(WATER, "cc-pvdz", 0, exact()),
    );
    compare(
        "CH3F / cc-pVDZ, frozen core 2",
        &closed(CH3F, "cc-pvdz", 2, exact()),
    );
    compare(
        "benzene / 6-31G, RI-JK, frozen core 6",
        &closed(benzene, "6-31g", 6, rijk()),
    );
    compare("OH doublet (UHF) / 6-31G", &|| {
        let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
        total_u_terfc_rimp2_gradient_with(
            &mol,
            &bs("6-31g"),
            &bs("cc-pvdz-ri"),
            Operator::terfc(R0),
            &RiMp2Config::default(),
            None,
            &exact(),
            &mut ScfWarmStart::new(false),
            RhfConfig::default().with_density_conv(1e-11),
        )
        .unwrap()
    });
    compare("SCS-MP2(2terfc) H2O / cc-pVDZ", &|| {
        let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
        total_scs_mp2_2terfc_gradient_on(
            &mol,
            &bs("cc-pvdz"),
            &bs("cc-pvdz-ri"),
            &ScsMp2TerfcConfig::default(),
            None,
            &exact(),
            &mut ScfWarmStart::new(false),
        )
        .unwrap()
    });
}
