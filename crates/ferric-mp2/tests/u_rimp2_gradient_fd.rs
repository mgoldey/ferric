//! Finite-difference anchor of the analytic unrestricted RI-MP2 gradient
//! ([`ferric_mp2::u_rimp2_gradient`]) on open-shell molecules: OH and NH2
//! (doublets) and O2 (triplet) in 6-31G with cc-pVDZ-RI, on both the
//! four-centre and the RI-JK reference.
//!
//! PySCF has no density-fitted UMP2 gradient (`pyscf/grad/ump2.py` is the
//! conventional-integral one), so there is no external reference for the
//! fitted energy; the anchor is the derivative of ferric's own U-RI-MP2
//! energy, whose energy is itself pinned to PySCF DF-UMP2 elsewhere
//! (`u_rimp2` tests).
//!
//! # The bar (per coordinate)
//!
//! `D(h) = (E(x+h) − E(x−h)) / 2h` at `h = 2e-3` and `2h`, every displaced UHF
//! converged with `density_conv = 1e-13`, `energy_conv = 1e-14` and its final
//! DIIS commutator asserted below `1e-9` (a displaced SCF that stops on a
//! different iteration count than its partner otherwise injects a step of the
//! convergence error into the difference). The analytic gradient is compared
//! with the Richardson value `R = (4 D(h) − D(2h)) / 3`. Error sources:
//!
//! * truncation: `D(h) = g + c h² + d h⁴`, so `R = g − 4 d h⁴`. The `h²`
//!   coefficient is read off the data, `|c| h² = |D(2h) − D(h)| / 3`, and the
//!   `h⁴` term is that times `(2h/ℓ)²` with `ℓ ≥ 0.1` Bohr the length over
//!   which the curvature of a bonded PES changes, so `4|d| h⁴ ≤ 4 · 1.6e-3 ·
//!   |c| h²`; with margin `1e-2 · |D(2h) − D(h)| / 3`;
//! * energy noise: the MP2 energy is not variational in the orbitals, so an
//!   SCF density error `δD ≤ 10·density_conv` moves it linearly,
//!   `δE ≲ ‖∂E/∂D‖ δD ≈ 1e-12`; with margin `δE = 1e-11`. `R` carries
//!   `(4/3)·δE/h + (1/3)·δE/(2h) = 1.5 δE / h = 7.5e-9`;
//! * the analytic side: a Z-vector residual `r < 1e-8` (the production
//!   tolerance) perturbs the Lagrangian gradient by `≤ ‖∂_R Xvo‖ ‖A⁻¹‖ r`,
//!   `≲ 10 · 2 · 1e-8 = 2e-7`.
//!
//! `bar = 1e-2 · |D(2h) − D(h)| / 3 + 1.5 δE / h + 2e-7`. The plain central
//! difference `D(h)` is printed too: its own error is the truncation
//! `|D(2h) − D(h)| / 3` (up to `2e-6` Ha/Bohr for O2 at this step), which is
//! why it is not the compared value. The mutations of the opposite-spin
//! weight or of one spin's Z-vector move the gradient by `1e-3`-`1e-2`
//! Ha/Bohr.
//!
//! For the diatomics only the bond-axis coordinates are differenced: the
//! perpendicular ones are zero by symmetry (asserted on the analytic side),
//! and a ²Π UHF displaced off-axis can land on either of the split A'/A''
//! states, which is a property of the reference, not of the gradient.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::u_rimp2_gradient::{total_u_rimp2_energy_with, total_u_rimp2_gradient_with};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::scf_jk::{ScfJk, ScfJkMode};
use ferric_scf::warm_start::ScfWarmStart;
use ndarray::Array2;

const H: f64 = 2e-3;
const DELTA_E: f64 = 1e-11;
const ANALYTIC: f64 = 2e-7;
const ERR_MAX: f64 = 1e-9;

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

struct Case {
    mol: Molecule,
    obs: BasisSet,
    aux: BasisSet,
    jk: ScfJk,
    config: RiMp2Config,
}

impl Case {
    fn new(xyz: &str, mult: usize, fit: bool, frozen_core: usize) -> Self {
        Self {
            mol: Molecule::parse_xyz(xyz, 0, mult).unwrap(),
            obs: basis::bundled("6-31g").unwrap(),
            aux: basis::bundled("cc-pvdz-ri").unwrap(),
            jk: route(fit),
            config: RiMp2Config {
                frozen_core,
                ..Default::default()
            },
        }
    }

    fn energy(&self, mol: &Molecule) -> f64 {
        let (e, uhf) = total_u_rimp2_energy_with(
            mol,
            &self.obs,
            &self.aux,
            Operator::coulomb(),
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
        total_u_rimp2_gradient_with(
            &self.mol,
            &self.obs,
            &self.aux,
            Operator::coulomb(),
            &self.config,
            None,
            &self.jk,
            &mut ScfWarmStart::new(false),
            tight(),
        )
        .unwrap()
        .1
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

    /// Compare the analytic gradient with the Richardson difference on
    /// `coords`; returns the largest `|g − R| / bar`.
    fn check(&self, label: &str, coords: &[(usize, usize)]) -> f64 {
        let g = self.analytic();
        let mut worst = 0.0f64;
        for &(atom, c) in coords {
            let d1 = self.central(atom, c, H);
            let d2 = self.central(atom, c, 2.0 * H);
            let rich = (4.0 * d1 - d2) / 3.0;
            let bar = 1e-2 * (d2 - d1).abs() / 3.0 + 1.5 * DELTA_E / H + ANALYTIC;
            let err = (g[(atom, c)] - rich).abs();
            eprintln!(
                "{label} atom {atom} coord {c}: analytic {:+.10e} richardson {rich:+.10e} \
                 |diff| {err:.3e} bar {bar:.3e} (central D(h) {d1:+.10e}, |g - D(h)| {:.3e})",
                g[(atom, c)],
                (g[(atom, c)] - d1).abs()
            );
            assert!(
                err < bar,
                "{label}: atom {atom} coord {c}: |g - R| = {err:.3e} >= bar {bar:.3e}"
            );
            worst = worst.max(err / bar);
        }
        worst
    }

    /// The perpendicular components of a diatomic on the z axis vanish.
    fn assert_axial(&self, g: &Array2<f64>) {
        for a in 0..2 {
            for c in 0..2 {
                assert!(
                    g[(a, c)].abs() < 1e-8,
                    "perpendicular g[{a},{c}] = {:.3e}",
                    g[(a, c)]
                );
            }
        }
    }
}

const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";
const O2: &str = "2\nO2\nO 0 0 0\nO 0 0 1.21\n";
const NH2: &str = "3\nNH2\nN 0.000 0.000 0.150\nH 0.000 0.800 -0.450\nH 0.000 -0.800 -0.450\n";

fn axial(case: &Case, label: &str) {
    case.assert_axial(&case.analytic());
    case.check(label, &[(0, 2), (1, 2)]);
}

fn all(natoms: usize) -> Vec<(usize, usize)> {
    (0..natoms)
        .flat_map(|a| (0..3).map(move |c| (a, c)))
        .collect()
}

#[test]
fn oh_doublet_matches_fd_four_centre() {
    axial(&Case::new(OH, 2, false, 0), "OH exact");
}

#[test]
fn oh_doublet_matches_fd_rijk() {
    axial(&Case::new(OH, 2, true, 0), "OH rijk");
}

#[test]
fn oh_doublet_frozen_core_matches_fd() {
    axial(&Case::new(OH, 2, false, 1), "OH exact fc1");
}

#[test]
fn nh2_doublet_matches_fd_four_centre() {
    Case::new(NH2, 2, false, 0).check("NH2 exact", &all(3));
}

#[test]
fn nh2_doublet_matches_fd_rijk() {
    Case::new(NH2, 2, true, 0).check("NH2 rijk", &all(3));
}

#[test]
fn o2_triplet_matches_fd_four_centre() {
    axial(&Case::new(O2, 3, false, 0), "O2 exact");
}

#[test]
fn o2_triplet_matches_fd_rijk() {
    axial(&Case::new(O2, 3, true, 0), "O2 rijk");
}
