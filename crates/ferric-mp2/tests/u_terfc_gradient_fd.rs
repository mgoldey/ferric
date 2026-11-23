//! Finite-difference anchors of the open-shell terfc gradients on the OH and
//! NH2 doublets (UHF, 6-31G, cc-pVDZ-RI):
//!
//! * terfc RI-MP2
//!   ([`ferric_mp2::att_rimp2_gradient::total_u_terfc_rimp2_gradient_with`]),
//!   the derivative of `E_UHF + E_c[terfc]` (`u_ri_mp2` on terfc);
//! * SCS-MP2(2terfc)
//!   ([`ferric_mp2::scs_2terfc_gradient::total_u_scs_mp2_2terfc_gradient_with`]),
//!   the derivative of `E_UHF + c_os E_αβ[τ1] + c_ss (E_SS[τ2] − E_SS[τ1])`
//!   (`scs_mp2_2terfc` on a UHF), at the published cutoffs and coefficients
//!   (r0 0.75 / 1.05 Å, c_os 1.27, c_ss 4.05).
//!
//! Both against Richardson differences of ferric's own energies. The terfc
//! energies are the ones matched to the independent Fourier-space reference
//! (`HYPOTHESES-terfc-kspace.md`, `terfc_vs_exact.rs`).
//!
//! # Hypotheses
//!
//! Physics: with terfc in the per-spin amplitudes, `x^σ`, `Imat^σ` and the
//! 3c/2c response (each SCS term on its own cutoff, with `c_ss` on the
//! same-spin products and `c_os` on the opposite-spin ones), and Coulomb in
//! the coupled Z-vector, the `veff^σ` and the two-electron derivative, the
//! gradient matches FD to the bar below. Artifact: a term on the wrong
//! cutoff, `c_ss` on the opposite-spin products, a dropped β Z-vector or
//! terfc on the reference moves the gradient by `1e-4`–`1e-2` Ha/Bohr, far
//! above the bar. The closed-shell-limit anchors
//! (`scs_2terfc_gradient.rs`) cannot see a spin-swapped weight on a
//! genuine open shell (α ≡ β there), which is why FD on doublets decides.
//!
//! # The bar (per coordinate)
//!
//! The derivation of `u_att_rimp2_gradient_fd.rs`, which does not depend on
//! the operator or on the spin scaling: `bar = 1e-2 · |D(2h) − D(h)| / 3 +
//! 1.5 δE / h + 2e-7` with `h = 2e-3`, `δE = 1e-11`, every displaced UHF at
//! `density_conv = 1e-13`, `energy_conv = 1e-14`, final DIIS commutator
//! below `1e-9`. The terfc table interpolation adds `≲ 1e-8` to the FD
//! (`terfc_rimp2_gradient_fd.rs`), inside the `2e-7` analytic term. The
//! SCS-MP2(2terfc) coefficients multiply the correlation energy by at most
//! `c_os + 2 c_ss ≈ 9.4`; the SCF-noise term `δE` is the reference's and does
//! not scale, and the correlation part of `δE` (`≲ 1e-12` at this
//! convergence) times 9.4 stays inside `1e-11`.
//!
//! # Translational invariance
//!
//! `Σ_A ∂E/∂R_A` is roundoff; bar `1e-9`.
//!
//! For OH only the bond-axis coordinates are differenced (the others are
//! zero by symmetry, asserted on the analytic side); for NH2 the six
//! in-plane ones.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::units::ANGSTROM_TO_BOHR;
use ferric_integrals::operator::Operator;
use ferric_mp2::att_rimp2_gradient::{
    total_u_terfc_rimp2_energy_with, total_u_terfc_rimp2_gradient_with,
};
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::scs::ScsMp2TerfcConfig;
use ferric_mp2::scs_2terfc_gradient::{
    total_u_scs_mp2_2terfc_energy_with, total_u_scs_mp2_2terfc_gradient_with,
};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::scf_jk::{ScfJk, ScfJkMode};
use ferric_scf::warm_start::ScfWarmStart;
use ferric_scf::ScfResult;
use ndarray::Array2;
use std::path::Path;

const H: f64 = 2e-3;
const DELTA_E: f64 = 1e-11;
const ANALYTIC: f64 = 2e-7;
const ERR_MAX: f64 = 1e-9;
const TRANSLATION_BAR: f64 = 1e-9;

static TABLES: std::sync::Once = std::sync::Once::new();

/// Point the terfc table engine at the committed tables when
/// `FERRIC_TERF_TABLE_DIR` is unset; fail (never skip) when absent.
fn ensure_tables() {
    TABLES.call_once(|| {
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

fn tight() -> RhfConfig {
    RhfConfig::default()
        .with_density_conv(1e-13)
        .with_energy_conv(1e-14)
        .with_max_iter(300)
}

fn route(fit: bool) -> ScfJk {
    if fit {
        ScfJk {
            mode: ScfJkMode::RiJk,
            aux: Some("def2-universal-jkfit".into()),
        }
    } else {
        ScfJk {
            mode: ScfJkMode::Exact,
            aux: None,
        }
    }
}

/// The energy and its gradient: terfc RI-MP2 on `op`, or SCS-MP2(2terfc).
#[derive(Clone)]
enum Method {
    Terfc(Operator),
    Scs2terfc,
}

struct Case {
    mol: Molecule,
    obs: BasisSet,
    aux: BasisSet,
    jk: ScfJk,
    method: Method,
    frozen_core: usize,
}

impl Case {
    fn new(xyz: &str, fit: bool, method: Method, frozen_core: usize) -> Self {
        ensure_tables();
        Self {
            mol: Molecule::parse_xyz(xyz, 0, 2).unwrap(),
            obs: basis::bundled("6-31g").unwrap(),
            aux: basis::bundled("cc-pvdz-ri").unwrap(),
            jk: route(fit),
            method,
            frozen_core,
        }
    }

    fn ri(&self) -> RiMp2Config {
        RiMp2Config {
            frozen_core: self.frozen_core,
            ..Default::default()
        }
    }

    fn scs(&self) -> ScsMp2TerfcConfig {
        ScsMp2TerfcConfig {
            frozen_core: self.frozen_core,
            ..Default::default()
        }
    }

    fn energy(&self, mol: &Molecule) -> f64 {
        let (e, uhf): (f64, ScfResult) = match self.method {
            Method::Terfc(op) => total_u_terfc_rimp2_energy_with(
                mol,
                &self.obs,
                &self.aux,
                op,
                &self.ri(),
                &self.jk,
                tight(),
            ),
            Method::Scs2terfc => total_u_scs_mp2_2terfc_energy_with(
                mol,
                &self.obs,
                &self.aux,
                &self.scs(),
                &self.jk,
                tight(),
            ),
        }
        .unwrap();
        let rep = uhf.convergence.expect("a convergence report");
        assert!(
            uhf.converged && rep.err_max < ERR_MAX,
            "displaced UHF not converged tightly: err_max = {:.3e}",
            rep.err_max
        );
        e
    }

    fn analytic(&self) -> Array2<f64> {
        let warm = &mut ScfWarmStart::new(false);
        let (e, g) = match self.method {
            Method::Terfc(op) => total_u_terfc_rimp2_gradient_with(
                &self.mol,
                &self.obs,
                &self.aux,
                op,
                &self.ri(),
                None,
                &self.jk,
                warm,
                tight(),
            ),
            Method::Scs2terfc => total_u_scs_mp2_2terfc_gradient_with(
                &self.mol,
                &self.obs,
                &self.aux,
                &self.scs(),
                None,
                &self.jk,
                warm,
                tight(),
            ),
        }
        .unwrap();
        let e_ref = self.energy(&self.mol);
        assert!(
            (e - e_ref).abs() < 1e-10,
            "driver energy {e:.12} vs energy {e_ref:.12}"
        );
        let sum = g.sum_axis(ndarray::Axis(0));
        for (c, v) in sum.iter().enumerate() {
            assert!(
                v.abs() < TRANSLATION_BAR,
                "translational sum, coord {c}: {v:.3e}"
            );
        }
        g
    }

    fn central(&self, atom: usize, c: usize, h: f64) -> f64 {
        let shifted = |s: f64| {
            let mut m = self.mol.clone();
            let a = &mut m.atoms[atom];
            match c {
                0 => a.x += s,
                1 => a.y += s,
                _ => a.zpos += s,
            }
            m
        };
        (self.energy(&shifted(h)) - self.energy(&shifted(-h))) / (2.0 * h)
    }

    fn check(&self, label: &str, g: &Array2<f64>, coords: &[(usize, usize)]) {
        let mut worst = 0.0f64;
        for &(atom, c) in coords {
            let d1 = self.central(atom, c, H);
            let d2 = self.central(atom, c, 2.0 * H);
            let rich = (4.0 * d1 - d2) / 3.0;
            let bar = 1e-2 * (d2 - d1).abs() / 3.0 + 1.5 * DELTA_E / H + ANALYTIC;
            let err = (g[(atom, c)] - rich).abs();
            worst = worst.max(err / bar);
            eprintln!(
                "{label} atom {atom} coord {c}: analytic {:+.10e} richardson {rich:+.10e} \
                 |diff| {err:.3e} bar {bar:.3e}",
                g[(atom, c)]
            );
            assert!(
                err < bar,
                "{label}: atom {atom} coord {c}: |g - R| = {err:.3e} >= bar {bar:.3e}"
            );
        }
        eprintln!("{label}: worst |g - R| / bar = {worst:.3}");
    }
}

const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";
const NH2: &str = "3\nNH2\nN 0.000 0.020 0.150\nH 0.000 0.810 -0.440\nH 0.000 -0.790 -0.470\n";

fn oh(case: &Case, label: &str) {
    let g = case.analytic();
    for a in 0..2 {
        for c in 0..2 {
            assert!(
                g[(a, c)].abs() < 1e-8,
                "perpendicular g[{a},{c}] = {:.3e}",
                g[(a, c)]
            );
        }
    }
    case.check(label, &g, &[(0, 2), (1, 2)]);
}

fn nh2(case: &Case, label: &str) {
    let g = case.analytic();
    for a in 0..3 {
        assert!(
            g[(a, 0)].abs() < 1e-8,
            "out-of-plane g[{a},0] = {:.3e}",
            g[(a, 0)]
        );
    }
    let coords: Vec<(usize, usize)> = (0..3).flat_map(|a| [(a, 1), (a, 2)]).collect();
    case.check(label, &g, &coords);
}

fn terfc() -> Method {
    Method::Terfc(Operator::terfc(1.05 * ANGSTROM_TO_BOHR))
}

#[test]
fn u_terfc_oh_doublet_matches_fd_four_centre() {
    oh(&Case::new(OH, false, terfc(), 0), "terfc OH exact");
}

#[test]
fn u_terfc_nh2_doublet_matches_fd_rijk_frozen_core() {
    nh2(&Case::new(NH2, true, terfc(), 1), "terfc NH2 rijk fc1");
}

#[test]
fn u_scs_2terfc_oh_doublet_matches_fd_frozen_core() {
    oh(
        &Case::new(OH, false, Method::Scs2terfc, 1),
        "2terfc OH exact fc1",
    );
}

#[test]
fn u_scs_2terfc_nh2_doublet_matches_fd_four_centre() {
    nh2(
        &Case::new(NH2, false, Method::Scs2terfc, 0),
        "2terfc NH2 exact",
    );
}

#[test]
fn u_scs_2terfc_nh2_doublet_matches_fd_rijk() {
    nh2(
        &Case::new(NH2, true, Method::Scs2terfc, 0),
        "2terfc NH2 rijk",
    );
}
