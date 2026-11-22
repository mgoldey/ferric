//! The analytic terfc RI-MP2 gradient
//! ([`ferric_mp2::att_rimp2_gradient::terfc_rimp2_gradient_analytical`])
//! against Richardson finite differences of ferric's own terfc RI-MP2 energy
//! (`ri_mp2` with `Operator::terfc` on the `(P|ia)` integrals and the RI
//! metric, Coulomb RHF reference): the `att_operator = "terfc"` energy of the
//! CLI.
//!
//! No other code has a terfc-attenuated MP2 gradient, so the anchors are the
//! derivative of the energy and the `r0 → 0` limit at fixed `ω`, where
//! `terfc(r; 0, ω) = erfc(ωr)` exactly
//! (`book/src/15-correlated-gradients/02-rimp2-gradient.md#rimp2-gradient-attenuated-erfc-ri-mp2`)
//! and the gradient is the
//! erfc gradient (itself FD-validated in `att_rimp2_gradient_fd.rs`).
//!
//! Systems (r0 = 1.05 Å, `ω r0 = 1/√2` unless stated, cc-pvdz-ri):
//! distorted H2O/cc-pVDZ; H2O/cc-pVDZ at a decoupled `ω r0 = 4`; CH3F/cc-pVDZ
//! with the two 1s cores frozen; acetaldehyde/6-31G* (7 atoms); H2O/cc-pVDZ on
//! an RI-JK reference (the CLI's default optimize route), whose FD energy is
//! the terfc RI-MP2 energy of the fitted SCF.
//!
//! # Hypotheses
//!
//! Physics: if the operator split is right (terfc in the amplitudes, `b_full`,
//! `Imat` and the 3c/2c response; Coulomb in the Z-vector Hessian, the `veff`
//! of the correlation density and the four-centre derivative term) and the
//! terf/terfc derivative integrals are right, the analytic gradient matches FD
//! of the terfc energy to the bar below. Artifact: a term on the wrong
//! operator moves the gradient by the derivative of `E_c[1/r] − E_c[terfc]`,
//! `1e-4`–`1e-2` Ha/Bohr at r0 = 1.05 Å, far above the bar; the `r0 → 0`
//! anchor compares with erfc and cannot see a terfc-vs-Coulomb split (both
//! sides would carry it), which is why FD at a finite r0 decides. A table
//! path read at the wrong order of `G_m` passes the anchor at `s = 0` only if
//! it is also wrong for erfc's Boys path, which it does not share.
//!
//! # The bar (per coordinate)
//!
//! The derivation of `att_rimp2_gradient_fd.rs` (which does not depend on the
//! operator): `bar = 1e-2 · |D(2h) − D(h)| / 3 + 1.5 δE/h + 2e-7` with
//! `h = 2e-3`, `δE = 1e-11`. One term is new: the table engine interpolates
//! `G_m(S, s)` to `≤ 1e-12` relative (`terf_interp_accuracy.rs`), a smooth
//! error in R that the FD differentiates. Its derivative, the interpolant's
//! slope error over the node spacing, is checked at the integral level
//! (`terfc_deriv_integrals.rs`: FD of the values vs the analytic derivative
//! at the Coulomb FD floor, 2.6e-11 per integral). Contracted with densities
//! of `O(1)` over `≲ 1e5` significant integrals it stays `≲ 1e-8`, inside the
//! `2e-7` analytic term.
//!
//! # Translational invariance
//!
//! As in `att_rimp2_gradient_fd.rs`: the table derivative sets the third
//! centre to `−(dP + dA)`, so the sum is roundoff; bar `1e-9`.
//!
//! # r0 → 0 anchor
//!
//! `terf(r; r0, ω) = ½[erf(ω(r − r0)) + erf(ω(r + r0))]` is even in r0, so
//! `terfc(r; r0, ω) = erfc(ωr) + O(ω³ r0² r e^{−ω²r²})`. At `r0 = 1e-6` Bohr
//! the operator difference is `≲ 1e-12`; the table's `s = φ²r0² ≈ 1e-13` sits
//! on the first node, so what remains is the table's interpolation of `G_m(S,
//! 0) = F_m(S)` against libint2's Boys function, `≤ 1e-12` relative per
//! integral. The gradient is a sum of `O(1)`-weighted integral derivatives;
//! the bar is `1e-9`, and a non-vacuity check requires the gap at the
//! production r0 to exceed it by 1e4.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::att_rimp2_gradient::{
    att_rimp2_gradient_analytical, terfc_rimp2_gradient_analytical, total_terfc_rimp2_gradient_on,
};
use ferric_mp2::attenuated::AttenuatedMp2Config;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::scf_jk::{ScfJk, ScfJkMode};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::warm_start::ScfWarmStart;
use ferric_scf::ScfResult;
use ndarray::Array2;
use std::path::Path;

const H: f64 = 2e-3;
const DELTA_E: f64 = 1e-11;
const ANALYTIC: f64 = 2e-7;
const ERR_MAX: f64 = 1e-9;
const TRANSLATION_BAR: f64 = 1e-9;
const ANCHOR_BAR: f64 = 1e-9;
const AUX: &str = "cc-pvdz-ri";
/// r0 = 1.05 Å in Bohr.
const R0: f64 = 1.05 / 0.529_177_210_92;

const WATER: &str = "3\nwater, off C2v\nO 0.02 -0.01 0.117790\nH 0.0 0.785453 -0.451161\nH 0.03 -0.735453 -0.491161\n";
const CH3F: &str = "5\nfluoromethane, distorted\nC 0.00 0.01 -0.63\nF 0.02 0.00 0.75\nH 1.03 0.02 -0.99\nH -0.52 0.90 -0.97\nH -0.50 -0.88 -1.01\n";
const ACETALDEHYDE: &str = "7\nacetaldehyde\nC 0.000 0.000 0.000\nC 1.500 0.000 0.000\nO 2.190 1.020 0.010\nH 1.980 -1.000 -0.020\nH -0.380 1.030 0.020\nH -0.390 -0.520 0.880\nH -0.370 -0.540 -0.870\n";

static TABLES: std::sync::Once = std::sync::Once::new();

/// The terfc table engine reads `FERRIC_TERF_TABLE_DIR` only. Point it at the
/// committed `terf-tables/` when unset and FAIL (never skip) when absent.
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

fn ri(frozen_core: usize) -> RiMp2Config {
    RiMp2Config {
        frozen_core,
        ..Default::default()
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
    ensure_tables();
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

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

struct Case {
    label: &'static str,
    mol: Molecule,
    obs: BasisSet,
    aux: BasisSet,
    jk: Option<ScfJk>,
    op: Operator,
    ri: RiMp2Config,
}

impl Case {
    fn new(label: &'static str, xyz: &str, basis_name: &str, op: Operator, fc: usize) -> Self {
        Self {
            label,
            mol: Molecule::parse_xyz(xyz, 0, 1).unwrap(),
            obs: basis::bundled(basis_name).unwrap(),
            aux: basis::bundled(AUX).unwrap(),
            jk: None,
            op,
            ri: ri(fc),
        }
    }

    fn energy(&self, mol: &Molecule) -> f64 {
        let p = point(mol, &self.obs, &self.aux, self.jk.as_ref());
        ri_mp2(mol, &p.prep, &p.dfbs, self.op, &p.rhf, &self.ri)
            .unwrap()
            .total_energy
    }

    fn analytic(&self) -> Array2<f64> {
        match &self.jk {
            None => {
                let p = point(&self.mol, &self.obs, &self.aux, None);
                terfc_rimp2_gradient_analytical(
                    &self.mol, &p.prep, &p.dfbs, &p.bounds, &p.rhf, self.op, &self.ri, None,
                )
                .unwrap()
            }
            Some(jk) => {
                ensure_tables();
                total_terfc_rimp2_gradient_on(
                    &self.mol,
                    &self.obs,
                    &self.aux,
                    self.op,
                    &self.ri,
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
            max_abs(&g)
        );
        assert!(
            trans < TRANSLATION_BAR,
            "{}: translational sum {trans:.3e}",
            self.label
        );
        worst
    }
}

/// ANCHOR: at r0 → 0 and fixed ω the terfc gradient is the erfc gradient.
#[test]
fn terfc_gradient_matches_erfc_in_the_r0_to_zero_limit() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let obs = basis::bundled("cc-pvdz").unwrap();
    let aux = basis::bundled(AUX).unwrap();
    let p = point(&mol, &obs, &aux, None);
    let omega = 1.0 / (R0 * std::f64::consts::SQRT_2);
    for fc in [0usize, 1] {
        let g_terfc = terfc_rimp2_gradient_analytical(
            &mol,
            &p.prep,
            &p.dfbs,
            &p.bounds,
            &p.rhf,
            Operator::terfc_with_omega(1e-6, omega),
            &ri(fc),
            None,
        )
        .unwrap();
        let erfc = AttenuatedMp2Config {
            omega,
            scaling: 1.0,
            frozen_core: fc,
            screen_thresh: None,
            memory_budget_bytes: None,
        };
        let g_erfc =
            att_rimp2_gradient_analytical(&mol, &p.prep, &p.dfbs, &p.bounds, &p.rhf, &erfc, None)
                .unwrap();
        let diff = max_abs(&(&g_terfc - &g_erfc));
        let g_prod = terfc_rimp2_gradient_analytical(
            &mol,
            &p.prep,
            &p.dfbs,
            &p.bounds,
            &p.rhf,
            Operator::terfc(R0),
            &ri(fc),
            None,
        )
        .unwrap();
        let gap = max_abs(&(&g_prod - &g_erfc));
        eprintln!(
            "r0→0 anchor (fc {fc}): max |g_terfc(r0=1e-6) − g_erfc| = {diff:.2e} (bar \
             {ANCHOR_BAR:.0e}); at r0 = 1.05 Å the gap is {gap:.2e}"
        );
        assert!(diff < ANCHOR_BAR, "r0→0 anchor (fc {fc}): {diff:.3e}");
        assert!(
            gap > 1e4 * ANCHOR_BAR,
            "r0 = 1.05 Å indistinguishable from erfc: {gap:.3e}"
        );
    }
}

#[test]
fn terfc_driver_energy_is_the_terfc_energy() {
    let case = Case::new("H2O/cc-pVDZ", WATER, "cc-pvdz", Operator::terfc(R0), 0);
    ensure_tables();
    let (e_drv, _) = total_terfc_rimp2_gradient_on(
        &case.mol,
        &case.obs,
        &case.aux,
        case.op,
        &case.ri,
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
fn terfc_gradient_matches_fd_water() {
    Case::new("H2O/cc-pVDZ", WATER, "cc-pvdz", Operator::terfc(R0), 0).check();
}

#[test]
fn terfc_gradient_matches_fd_water_decoupled_omega() {
    Case::new(
        "H2O/cc-pVDZ ω r0 = 4",
        WATER,
        "cc-pvdz",
        Operator::terfc_with_omega(R0, 4.0 / R0),
        0,
    )
    .check();
}

#[test]
fn terfc_gradient_matches_fd_ch3f_frozen_core() {
    Case::new("CH3F/cc-pVDZ fc=2", CH3F, "cc-pvdz", Operator::terfc(R0), 2).check();
}

#[test]
fn terfc_gradient_matches_fd_acetaldehyde() {
    Case::new(
        "CH3CHO/6-31G*",
        ACETALDEHYDE,
        "6-31g*",
        Operator::terfc(R0),
        0,
    )
    .check();
}

#[test]
fn terfc_gradient_matches_fd_water_rijk_reference() {
    let mut case = Case::new(
        "H2O/cc-pVDZ RI-JK fc=1",
        WATER,
        "cc-pvdz",
        Operator::terfc(R0),
        1,
    );
    case.jk = Some(rijk());
    case.check();
}

#[test]
fn terfc_gradient_refuses_what_it_does_not_differentiate() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let obs = basis::bundled("sto-3g").unwrap();
    let aux = basis::bundled(AUX).unwrap();
    let p = point(&mol, &obs, &aux, None);
    let run = |op: Operator, cfg: &RiMp2Config, bounds: &SchwarzBounds| {
        terfc_rimp2_gradient_analytical(&mol, &p.prep, &p.dfbs, bounds, &p.rhf, op, cfg, None)
    };
    for op in [Operator::erfc(0.3), Operator::terf(R0), Operator::coulomb()] {
        assert!(
            run(op, &ri(0), &p.bounds).is_err(),
            "{:?} accepted",
            op.kind
        );
    }
    for (r0, w) in [(0.0, 1.0), (R0, 0.0), (f64::NAN, 1.0), (R0, f64::INFINITY)] {
        assert!(
            run(Operator::terfc_with_omega(r0, w), &ri(0), &p.bounds).is_err(),
            "r0 {r0} omega {w} accepted"
        );
    }
    let kappa = RiMp2Config {
        kappa: Some(1.0),
        ..ri(0)
    };
    assert!(
        run(Operator::terfc(R0), &kappa, &p.bounds).is_err(),
        "kappa accepted"
    );
    let erf_bounds = SchwarzBounds::compute(Operator::erf(0.3), &p.prep).unwrap();
    assert!(
        run(Operator::terfc(R0), &ri(0), &erf_bounds).is_err(),
        "a non-Coulomb reference accepted"
    );
}
