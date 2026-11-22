//! The analytic attenuated RI-MP2 gradient
//! ([`ferric_mp2::att_rimp2_gradient`]) against Richardson finite differences
//! of ferric's own attenuated RI-MP2 energy
//! ([`ferric_mp2::attenuated::attenuated_ri_mp2`], erfc(ωr)/r on the `(P|ia)`
//! integrals and the RI metric, Coulomb RHF reference).
//!
//! There is no independent attenuated RI-MP2 gradient code (PySCF has no
//! erfc-attenuated MP2 gradient), so the anchors are the derivative of the
//! energy and the ω → 0 limit, where erfc becomes Coulomb and the gradient
//! is the RI-MP2 gradient (itself pinned to PySCF elsewhere).
//!
//! Systems (ω = 0.420 Å⁻¹ unless stated, cc-pvdz-ri):
//! distorted H2O/cc-pVDZ; CH3F/cc-pVDZ (heavy atom F), all-electron and with
//! the two 1s cores frozen; acetaldehyde/6-31G* (7 atoms, the larger case);
//! H2O/cc-pVDZ at ω = 1.0 Å⁻¹ with scaling 1.2; H2O/cc-pVDZ on an RI-JK
//! reference (the CLI's default optimize route), whose FD energy is the
//! attenuated RI-MP2 energy of the fitted SCF.
//!
//! # Hypotheses
//!
//! Physics: if the operator split is right (erfc in the amplitudes, `b_full`,
//! `Imat` and the 3c/2c response; Coulomb in the Z-vector Hessian, the `veff`
//! of the correlation density and the four-centre derivative term), the
//! analytic gradient matches FD of the attenuated energy to the bar below.
//! Artifact: a term on the wrong operator (erfc where the HF Fock is
//! Coulomb, or Coulomb where the energy is erfc) moves the gradient by the
//! size of `E_c[1/r] − E_c[erfc]` derivatives, `1e-4`-`1e-2` Ha/Bohr at
//! ω = 0.420 Å⁻¹, far above the bar; the ω → 0 anchor cannot see such a split
//! (both operators coincide there), which is why FD at a finite ω is the
//! deciding test, and why the mutations below include each split.
//!
//! # The bar (per coordinate)
//!
//! `D(h) = (E(x+h) − E(x−h)) / 2h` at `h = 2e-3` and `2h`, every displaced
//! RHF converged with `density_conv = 1e-13`, `energy_conv = 1e-14` and its
//! final DIIS commutator asserted below `1e-9`. The analytic gradient is
//! compared with `R = (4 D(h) − D(2h)) / 3`. Error sources:
//!
//! * truncation: `D(h) = g + c h² + d h⁴`, so `R = g − 4 d h⁴`; `|c| h² =
//!   |D(2h) − D(h)| / 3` is read off the data, and with `ℓ ≥ 0.1` Bohr the
//!   length over which a bonded PES's curvature changes, `4|d| h⁴ ≤ 4 (2h/ℓ)²
//!   |c| h² = 6.4e-3 |c| h²`; with margin `1e-2 · |D(2h) − D(h)| / 3`;
//! * energy noise: an SCF density error `δD ≤ 10 · density_conv` moves the
//!   non-variational correlation energy linearly, `δE ≲ ‖∂E/∂D‖ δD ≈ 1e-12`;
//!   with margin `δE = 1e-11`, and `R` carries `(4/3) δE/h + (1/3) δE/(2h) =
//!   1.5 δE/h = 7.5e-9`;
//! * the analytic side: the Z-vector residual `max|r| < 1e-8` perturbs the
//!   Lagrangian gradient by `≲ ‖∂_R Xvo‖ ‖A⁻¹‖ r ≲ 10 · 2 · 1e-8 = 2e-7`.
//!
//! `bar = 1e-2 · |D(2h) − D(h)| / 3 + 1.5 δE/h + 2e-7`, the derivation of
//! `u_rimp2_gradient_fd.rs`, which does not depend on the operator. The
//! RI-JK case's analytic side runs the production driver (`density_conv =
//! 1e-9`): the orbitals' residual `≲ 1e-8` adds `≲ 1e-8 · O(1)`, inside the
//! `2e-7`.
//!
//! # Translational invariance
//!
//! Every block of the gradient is a density contracted with integral
//! derivatives whose centre blocks sum to zero, for ANY density (the
//! Z-vector residual does not break it). `Σ_A ∂E/∂R_A` is then roundoff:
//! `~1e-15 · Σ|terms| ≲ 1e-11`; the bar is `1e-9`.
//!
//! # ω → 0 anchor
//!
//! `erfc(ωr)/r = 1/r − 2ω/√π + O(ω³ r²)`. The constant shifts `(P|μν)` by
//! `−(2ω/√π) n_P S_μν`, which vanishes in `(P|ia)` (`S_ia = 0`), and the
//! metric by the rank-one `−(2ω/√π) n nᵀ`, which moves the fitted `(ia|jb)`
//! by `(2ω/√π) q_ia q_jb` with `q_ia = nᵀV⁻¹(P|ia)` the fitted charge of an
//! orthogonal pair (the fit error, `≲ 1e-2`). At `ω = 1e-7` both effects are
//! `≲ 1e-11`: the attenuated gradient must equal
//! [`rimp2_gradient_analytical`] (Coulomb) to `1e-10`.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::att_rimp2_gradient::{att_rimp2_gradient_analytical, total_att_rimp2_gradient_on};
use ferric_mp2::attenuated::{attenuated_ri_mp2, AttenuatedMp2Config, BOHR_INV_PER_ANG_INV};
use ferric_mp2::gradient::rimp2_gradient_analytical;
use ferric_mp2::rimp2::RiMp2Config;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::scf_jk::{ScfJk, ScfJkMode};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::warm_start::ScfWarmStart;
use ferric_scf::ScfResult;
use ndarray::Array2;

const H: f64 = 2e-3;
const DELTA_E: f64 = 1e-11;
const ANALYTIC: f64 = 2e-7;
const ERR_MAX: f64 = 1e-9;
const TRANSLATION_BAR: f64 = 1e-9;
const ANCHOR_BAR: f64 = 1e-10;
const AUX: &str = "cc-pvdz-ri";

const WATER: &str = "3\nwater, off C2v\nO 0.02 -0.01 0.117790\nH 0.0 0.785453 -0.451161\nH 0.03 -0.735453 -0.491161\n";
const CH3F: &str = "5\nfluoromethane, distorted\nC 0.00 0.01 -0.63\nF 0.02 0.00 0.75\nH 1.03 0.02 -0.99\nH -0.52 0.90 -0.97\nH -0.50 -0.88 -1.01\n";
const ACETALDEHYDE: &str = "7\nacetaldehyde\nC 0.000 0.000 0.000\nC 1.500 0.000 0.000\nO 2.190 1.020 0.010\nH 1.980 -1.000 -0.020\nH -0.380 1.030 0.020\nH -0.390 -0.520 0.880\nH -0.370 -0.540 -0.870\n";

fn att(omega_ang_inv: f64, scaling: f64, frozen_core: usize) -> AttenuatedMp2Config {
    AttenuatedMp2Config {
        omega: omega_ang_inv * BOHR_INV_PER_ANG_INV,
        scaling,
        frozen_core,
        screen_thresh: None,
        memory_budget_bytes: None,
    }
}

fn rijk() -> ScfJk {
    ScfJk {
        mode: ScfJkMode::RiJk,
        aux: Some("def2-universal-jkfit".to_string()),
    }
}

/// One geometry's prepared bases, Coulomb bounds and tight RHF on `jk`.
struct Point {
    prep: PreparedBasis,
    dfbs: PreparedBasis,
    bounds: SchwarzBounds,
    rhf: ScfResult,
}

fn point(mol: &Molecule, obs: &BasisSet, aux: &BasisSet, jk: Option<&ScfJk>) -> Point {
    let op = Operator::coulomb();
    let prep = PreparedBasis::new(mol, obs).unwrap();
    let dfbs = PreparedBasis::new(mol, aux).unwrap();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut cfg = RhfConfig {
        density_conv: 1e-13,
        energy_conv: 1e-14,
        max_iter: 300,
        ..Default::default()
    }
    .for_geometry_task();
    if let Some(jk) = jk {
        jk.apply(&mut cfg);
    }
    let ctx = ferric_core::parallel::ParallelContext::default();
    let rhf = solve_rhf(&ctx, mol, &prep, op, &bounds, &cfg).unwrap();
    let rep = rhf.convergence.as_ref().expect("a convergence report");
    assert!(
        rhf.converged && rep.err_max < ERR_MAX,
        "RHF not converged tightly: err_max = {:.3e}",
        rep.err_max
    );
    Point {
        prep,
        dfbs,
        bounds,
        rhf,
    }
}

fn displaced(mol: &Molecule, atom: usize, coord: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match coord {
        0 => m.atoms[atom].x += h,
        1 => m.atoms[atom].y += h,
        _ => m.atoms[atom].zpos += h,
    }
    m
}

struct Case {
    label: &'static str,
    mol: Molecule,
    obs: BasisSet,
    aux: BasisSet,
    jk: Option<ScfJk>,
    config: AttenuatedMp2Config,
}

impl Case {
    fn new(label: &'static str, xyz: &str, basis_name: &str, config: AttenuatedMp2Config) -> Self {
        Self {
            label,
            mol: Molecule::parse_xyz(xyz, 0, 1).unwrap(),
            obs: basis::bundled(basis_name).unwrap(),
            aux: basis::bundled(AUX).unwrap(),
            jk: None,
            config,
        }
    }

    fn energy(&self, mol: &Molecule) -> f64 {
        let p = point(mol, &self.obs, &self.aux, self.jk.as_ref());
        attenuated_ri_mp2(mol, &p.prep, &p.dfbs, &p.rhf, &self.config)
            .unwrap()
            .total_energy
    }

    fn analytic(&self) -> Array2<f64> {
        match &self.jk {
            None => {
                let p = point(&self.mol, &self.obs, &self.aux, None);
                att_rimp2_gradient_analytical(
                    &self.mol,
                    &p.prep,
                    &p.dfbs,
                    &p.bounds,
                    &p.rhf,
                    &self.config,
                    None,
                )
                .unwrap()
            }
            Some(jk) => {
                total_att_rimp2_gradient_on(
                    &self.mol,
                    &self.obs,
                    &self.aux,
                    &self.config,
                    None,
                    jk,
                    &mut ScfWarmStart::new(false),
                )
                .unwrap()
                .1
            }
        }
    }

    /// Richardson FD vs analytic on every coordinate, and the translational
    /// sum. Returns the worst `|analytic − R|`.
    fn check(&self) -> f64 {
        let g = self.analytic();
        let mut worst = 0.0f64;
        let mut worst_ratio = 0.0f64;
        for atom in 0..self.mol.atoms.len() {
            for coord in 0..3 {
                let d = |h: f64| {
                    let ep = self.energy(&displaced(&self.mol, atom, coord, h));
                    let em = self.energy(&displaced(&self.mol, atom, coord, -h));
                    (ep - em) / (2.0 * h)
                };
                let (d1, d2) = (d(H), d(2.0 * H));
                let r = (4.0 * d1 - d2) / 3.0;
                let bar = 1e-2 * (d2 - d1).abs() / 3.0 + 1.5 * DELTA_E / H + ANALYTIC;
                let err = (g[(atom, coord)] - r).abs();
                worst = worst.max(err);
                worst_ratio = worst_ratio.max(err / bar);
                assert!(
                    err < bar,
                    "{} atom {atom} coord {coord}: analytic {:.10e} vs Richardson {r:.10e}: \
                     |diff| {err:.3e} > bar {bar:.3e} (central D(h) {d1:.10e})",
                    self.label,
                    g[(atom, coord)]
                );
            }
        }
        let trans = (0..3)
            .map(|c| g.column(c).sum().abs())
            .fold(0.0f64, f64::max);
        eprintln!(
            "{}: max |analytic − Richardson| = {worst:.3e} ({worst_ratio:.2} of the bar); \
             max |Σ_A g| = {trans:.2e}; max |g| = {:.3e}",
            self.label,
            g.iter().fold(0.0f64, |m, v| m.max(v.abs()))
        );
        assert!(
            trans < TRANSLATION_BAR,
            "{}: translational sum {trans:.3e}",
            self.label
        );
        worst
    }
}

#[test]
fn att_gradient_matches_rimp2_in_the_trivial_limit() {
    // ω = 1e-7 Bohr⁻¹: erfc(ωr)/r = 1/r to the 1e-11 level (module doc).
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let obs = basis::bundled("cc-pvdz").unwrap();
    let aux = basis::bundled(AUX).unwrap();
    let p = point(&mol, &obs, &aux, None);
    for fc in [0usize, 1] {
        let cfg = AttenuatedMp2Config {
            omega: 1e-7,
            ..att(0.420, 1.0, fc)
        };
        let g_att =
            att_rimp2_gradient_analytical(&mol, &p.prep, &p.dfbs, &p.bounds, &p.rhf, &cfg, None)
                .unwrap();
        let ri = RiMp2Config {
            frozen_core: fc,
            ..Default::default()
        };
        let g_mp2 = rimp2_gradient_analytical(
            &mol,
            &p.prep,
            &p.dfbs,
            Operator::coulomb(),
            &p.bounds,
            &p.rhf,
            &ri,
            None,
        )
        .unwrap();
        let diff = (&g_att - &g_mp2).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        // Non-vacuity: at the production ω the two must differ by far more.
        let g_prod = att_rimp2_gradient_analytical(
            &mol,
            &p.prep,
            &p.dfbs,
            &p.bounds,
            &p.rhf,
            &att(0.420, 1.0, fc),
            None,
        )
        .unwrap();
        let gap = (&g_prod - &g_mp2)
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        eprintln!(
            "ω→0 anchor (fc {fc}): max |g_att(ω=1e-7) − g_RI-MP2| = {diff:.2e} (bar \
             {ANCHOR_BAR:.0e}); at ω=0.420 Å⁻¹ the gap is {gap:.2e}"
        );
        assert!(diff < ANCHOR_BAR, "ω→0 anchor (fc {fc}): {diff:.3e}");
        assert!(
            gap > 1e4 * ANCHOR_BAR,
            "ω=0.420 Å⁻¹ indistinguishable from Coulomb: {gap:.3e}"
        );
    }
}

#[test]
fn att_driver_energy_is_the_attenuated_energy() {
    let case = Case::new("H2O/cc-pVDZ", WATER, "cc-pvdz", att(0.420, 1.0, 0));
    let (e_drv, _) = total_att_rimp2_gradient_on(
        &case.mol,
        &case.obs,
        &case.aux,
        &case.config,
        None,
        &ferric_mp2::gradient::exact_scf_jk(),
        &mut ScfWarmStart::new(false),
    )
    .unwrap();
    let e = case.energy(&case.mol);
    // The driver's RHF stops at density_conv 1e-9: the energy differs from
    // the tight one by the (non-variational) correlation part, ≲ 1e-9.
    assert!(
        (e_drv - e).abs() < 1e-8,
        "driver {e_drv:.12} vs energy {e:.12}"
    );
}

#[test]
fn att_gradient_matches_fd_water() {
    Case::new("H2O/cc-pVDZ", WATER, "cc-pvdz", att(0.420, 1.0, 0)).check();
}

#[test]
fn att_gradient_matches_fd_water_large_omega_scaled() {
    Case::new(
        "H2O/cc-pVDZ ω=1.0 s=1.2",
        WATER,
        "cc-pvdz",
        att(1.0, 1.2, 0),
    )
    .check();
}

#[test]
fn att_gradient_matches_fd_ch3f() {
    Case::new("CH3F/cc-pVDZ", CH3F, "cc-pvdz", att(0.420, 1.0, 0)).check();
}

#[test]
fn att_gradient_matches_fd_ch3f_frozen_core() {
    Case::new("CH3F/cc-pVDZ fc=2", CH3F, "cc-pvdz", att(0.420, 1.0, 2)).check();
}

#[test]
fn att_gradient_matches_fd_acetaldehyde() {
    Case::new("CH3CHO/6-31G*", ACETALDEHYDE, "6-31g*", att(0.420, 1.0, 0)).check();
}

#[test]
fn att_gradient_matches_fd_water_rijk_reference() {
    let mut case = Case::new("H2O/cc-pVDZ RI-JK", WATER, "cc-pvdz", att(0.420, 1.0, 0));
    case.jk = Some(rijk());
    case.check();
}

#[test]
fn att_gradient_refuses_what_it_does_not_differentiate() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let obs = basis::bundled("sto-3g").unwrap();
    let aux = basis::bundled(AUX).unwrap();
    let p = point(&mol, &obs, &aux, None);
    let run = |cfg: &AttenuatedMp2Config, bounds: &SchwarzBounds| {
        att_rimp2_gradient_analytical(&mol, &p.prep, &p.dfbs, bounds, &p.rhf, cfg, None)
    };
    let screened = AttenuatedMp2Config {
        screen_thresh: Some(1e-10),
        ..att(0.420, 1.0, 0)
    };
    assert!(run(&screened, &p.bounds).is_err(), "screen_thresh accepted");
    for omega in [0.0, -0.1, f64::NAN] {
        let bad = AttenuatedMp2Config {
            omega,
            ..att(0.420, 1.0, 0)
        };
        assert!(run(&bad, &p.bounds).is_err(), "omega {omega} accepted");
    }
    let erf_bounds = SchwarzBounds::compute(Operator::erf(0.3), &p.prep).unwrap();
    assert!(
        run(&att(0.420, 1.0, 0), &erf_bounds).is_err(),
        "a non-Coulomb reference accepted"
    );
}
