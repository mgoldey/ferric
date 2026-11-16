//! Finite-difference anchor of the analytic open-shell attenuated RI-MP2
//! gradient ([`ferric_mp2::att_rimp2_gradient::u_att_rimp2_gradient_analytical`])
//! on the OH and NH2 doublets in 6-31G with cc-pVDZ-RI: Richardson
//! differences of ferric's own `E_UHF + s · E_c[erfc]`
//! ([`ferric_mp2::attenuated::attenuated_ri_mp2`] on a UHF reference).
//!
//! The external check is in `validation_u_att_rimp2_gradient_pyscf.rs`
//! (finite differences of PySCF's UHF + DF-UMP2 energy on an erfc fit).
//!
//! # Hypotheses
//!
//! Physics: with erfc in the per-spin amplitudes, `x^σ`, `Imat^σ` and the
//! 3c/2c response, and Coulomb in the coupled Z-vector, the `veff^σ` of the
//! correlation densities and the two-electron derivative, the gradient
//! matches FD of the attenuated energy to the bar below. Artifact: a term on
//! the wrong operator, a dropped spin block of the relaxed density or a
//! dropped β Z-vector moves the gradient by `1e-4`-`1e-2` Ha/Bohr, far above
//! the bar. The ω → 0 and closed-shell anchors (in
//! `att_rimp2_gradient.rs`) cannot see an operator split (both operators
//! coincide there, and the closed shell has α ≡ β), which is why FD at a
//! finite ω on genuine open shells is the deciding test.
//!
//! # The bar (per coordinate)
//!
//! The derivation of `u_rimp2_gradient_fd.rs`, which does not depend on the
//! operator: `D(h) = (E(x+h) − E(x−h)) / 2h` at `h = 2e-3` and `2h`, every
//! displaced UHF converged with `density_conv = 1e-13`, `energy_conv =
//! 1e-14` and its final DIIS commutator asserted below `1e-9`; the analytic
//! gradient is compared with `R = (4 D(h) − D(2h)) / 3`, and
//! `bar = 1e-2 · |D(2h) − D(h)| / 3 + 1.5 δE / h + 2e-7` with `δE = 1e-11`
//! (truncation read off the data, SCF noise in the non-variational
//! correlation energy, the production Z-vector residual `1e-8` times
//! `‖∂_R Xvo‖ ‖A⁻¹‖ ≲ 20`).
//!
//! # Translational invariance
//!
//! Every block is a density contracted with integral derivatives whose
//! centre blocks sum to zero, so `Σ_A ∂E/∂R_A` is roundoff, `≲ 1e-11`; bar
//! `1e-9`.
//!
//! For OH only the bond-axis coordinates are differenced: the perpendicular
//! ones are zero by symmetry (asserted on the analytic side), and a ²Π UHF
//! displaced off-axis can land on either of the split states.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_mp2::att_rimp2_gradient::{
    total_u_att_rimp2_energy_with, total_u_att_rimp2_gradient_with,
};
use ferric_mp2::attenuated::{AttenuatedMp2Config, BOHR_INV_PER_ANG_INV};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::scf_jk::{ScfJk, ScfJkMode};
use ferric_scf::warm_start::ScfWarmStart;
use ndarray::Array2;

const H: f64 = 2e-3;
const DELTA_E: f64 = 1e-11;
const ANALYTIC: f64 = 2e-7;
const ERR_MAX: f64 = 1e-9;
const TRANSLATION_BAR: f64 = 1e-9;

fn tight() -> RhfConfig {
    RhfConfig {
        density_conv: 1e-13,
        energy_conv: 1e-14,
        max_iter: 300,
        ..Default::default()
    }
}

fn route(fit: bool) -> ScfJk {
    if fit {
        ScfJk {
            mode: ScfJkMode::RiJk,
            aux: Some("def2-universal-jkfit".into()),
            k_builder: None,
        }
    } else {
        ScfJk {
            mode: ScfJkMode::Exact,
            aux: None,
            k_builder: None,
        }
    }
}

fn att(omega_ang_inv: f64, scaling: f64, frozen_core: usize) -> AttenuatedMp2Config {
    AttenuatedMp2Config {
        omega: omega_ang_inv * BOHR_INV_PER_ANG_INV,
        scaling,
        frozen_core,
        screen_thresh: None,
        memory_budget_bytes: None,
    }
}

struct Case {
    mol: Molecule,
    obs: BasisSet,
    aux: BasisSet,
    jk: ScfJk,
    config: AttenuatedMp2Config,
}

impl Case {
    fn new(xyz: &str, fit: bool, config: AttenuatedMp2Config) -> Self {
        Self {
            mol: Molecule::parse_xyz(xyz, 0, 2).unwrap(),
            obs: basis::bundled("6-31g").unwrap(),
            aux: basis::bundled("cc-pvdz-ri").unwrap(),
            jk: route(fit),
            config,
        }
    }

    fn energy(&self, mol: &Molecule) -> f64 {
        let (e, uhf) = total_u_att_rimp2_energy_with(
            mol,
            &self.obs,
            &self.aux,
            &self.config,
            &self.jk,
            tight(),
        )
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
        let (e, g) = total_u_att_rimp2_gradient_with(
            &self.mol,
            &self.obs,
            &self.aux,
            &self.config,
            None,
            &self.jk,
            &mut ScfWarmStart::new(false),
            tight(),
        )
        .unwrap();
        // The driver's energy is the energy the FD differences.
        let e_ref = self.energy(&self.mol);
        assert!(
            (e - e_ref).abs() < 1e-10,
            "driver energy {e:.12} vs attenuated energy {e_ref:.12}"
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

    /// Compare `g` with the Richardson difference on `coords`.
    fn check(&self, label: &str, g: &Array2<f64>, coords: &[(usize, usize)]) {
        for &(atom, c) in coords {
            let d1 = self.central(atom, c, H);
            let d2 = self.central(atom, c, 2.0 * H);
            let rich = (4.0 * d1 - d2) / 3.0;
            let bar = 1e-2 * (d2 - d1).abs() / 3.0 + 1.5 * DELTA_E / H + ANALYTIC;
            let err = (g[(atom, c)] - rich).abs();
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

/// NH2 off C2v in the molecular (yz) plane: the out-of-plane components are
/// zero by the plane's symmetry; the six in-plane ones are differenced.
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

#[test]
fn oh_doublet_matches_fd_four_centre() {
    oh(&Case::new(OH, false, att(0.420, 1.0, 0)), "OH exact");
}

#[test]
fn oh_doublet_large_omega_scaled_frozen_core_matches_fd() {
    oh(
        &Case::new(OH, false, att(1.0, 1.2, 1)),
        "OH exact w=1.0 s=1.2 fc1",
    );
}

#[test]
fn oh_doublet_matches_fd_rijk() {
    oh(&Case::new(OH, true, att(0.420, 1.0, 0)), "OH rijk");
}

#[test]
fn nh2_doublet_matches_fd_four_centre() {
    nh2(&Case::new(NH2, false, att(0.420, 1.0, 0)), "NH2 exact");
}
