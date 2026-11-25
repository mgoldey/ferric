//! The analytic closed-shell SCS-MP2(2terfc) gradient
//! ([`ferric_mp2::scs_2terfc_gradient::scs_mp2_2terfc_gradient_analytical`])
//! against Richardson finite differences of ferric's own
//! [`ferric_mp2::scs::scs_mp2_2terfc`] energy,
//! `E_HF + c_os E_OS[τ1] + c_ss (E_SS[τ2] − E_SS[τ1])`, at the published
//! parameters (r0 0.75 / 1.05 Å, c_os 1.27, c_ss 4.05) with cc-pVDZ-RI. The
//! terfc energies are the ones matched to the independent Fourier-space
//! reference (`HYPOTHESES-terfc-kspace.md`).
//!
//! Systems: distorted H2O/cc-pVDZ on the four-centre reference; H2O/cc-pVDZ
//! on an RI-JK reference with the O 1s frozen (the CLI's default optimize
//! route); distorted CH3F/cc-pVDZ with both 1s cores frozen.
//!
//! # Hypotheses
//!
//! Physics: if each term runs on its own cutoff with its own spin scaling
//! and the two close one Coulomb Z-vector, the gradient matches FD to the
//! bar. Artifact: a term on the wrong cutoff (τ1 ↔ τ2), `c_ss` on the
//! opposite-spin part or terfc on the reference moves the gradient by
//! `1e-4`–`1e-2` Ha/Bohr (the term gradients are `3e-3` and `3e-2` on H2O,
//! `scs_2terfc_gradient.rs`), far above the bar. The in-crate anchors
//! (`r0(1) = r0(2)` → SOS; one Z-vector = sum of single-term gradients)
//! both read the same term definitions as the gradient, so they cannot see
//! a wrong cutoff or weight in `scs_2terfc_terms`; FD can.
//!
//! # The bar (per coordinate)
//!
//! As in `terfc_rimp2_gradient_fd.rs`: `bar = 1e-2 · |D(2h) − D(h)| / 3 +
//! 1.5 δE / h + 2e-7`, `h = 2e-3`, `δE = 1e-11`. The coefficients multiply
//! the correlation energy by at most `c_os + 2 c_ss ≈ 9.4`, whose SCF noise
//! at `density_conv = 1e-13` is `≲ 1e-12`, so `δE` holds.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::scs::{scs_mp2_2terfc, ScsMp2TerfcConfig};
use ferric_mp2::scs_2terfc_gradient::{
    scs_mp2_2terfc_gradient_analytical, total_scs_mp2_2terfc_gradient_on,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::scf_jk::ScfJk;
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
const AUX: &str = "cc-pvdz-ri";

const WATER: &str = "3\nwater, off C2v\nO 0.02 -0.01 0.117790\nH 0.0 0.785453 -0.451161\nH 0.03 -0.735453 -0.491161\n";
const CH3F: &str = "5\nfluoromethane, distorted\nC 0.00 0.01 -0.63\nF 0.02 0.00 0.75\nH 1.03 0.02 -0.99\nH -0.52 0.90 -0.97\nH -0.50 -0.88 -1.01\n";

static TABLES: std::sync::Once = std::sync::Once::new();

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

fn rijk() -> ScfJk {
    ScfJk::rijk("def2-universal-jkfit")
}

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
    let mut cfg = RhfConfig::default()
        .with_density_conv(1e-13)
        .with_energy_conv(1e-14)
        .with_max_iter(300)
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
    config: ScsMp2TerfcConfig,
}

impl Case {
    fn new(label: &'static str, xyz: &str, fc: usize, jk: Option<ScfJk>) -> Self {
        Self {
            label,
            mol: Molecule::parse_xyz(xyz, 0, 1).unwrap(),
            obs: basis::bundled("cc-pvdz").unwrap(),
            aux: basis::bundled(AUX).unwrap(),
            jk,
            config: ScsMp2TerfcConfig {
                frozen_core: fc,
                ..Default::default()
            },
        }
    }

    fn energy(&self, mol: &Molecule) -> f64 {
        let p = point(mol, &self.obs, &self.aux, self.jk.as_ref());
        scs_mp2_2terfc(mol, &p.prep, &p.dfbs, &p.rhf, &self.config)
            .unwrap()
            .total_energy
    }

    fn analytic(&self) -> Array2<f64> {
        match &self.jk {
            None => {
                let p = point(&self.mol, &self.obs, &self.aux, None);
                scs_mp2_2terfc_gradient_analytical(
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
                ensure_tables();
                total_scs_mp2_2terfc_gradient_on(
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

    fn check(&self) {
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
                     |diff| {err:.3e} > bar {bar:.3e}",
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
    }
}

#[test]
fn scs_2terfc_driver_energy_is_the_scs_2terfc_energy() {
    let case = Case::new("H2O/cc-pVDZ", WATER, 0, None);
    ensure_tables();
    let (e_drv, _) = total_scs_mp2_2terfc_gradient_on(
        &case.mol,
        &case.obs,
        &case.aux,
        &case.config,
        None,
        &ferric_scf::scf_jk::ScfJk::exact(),
        &mut ScfWarmStart::new(false),
    )
    .unwrap();
    let e = case.energy(&case.mol);
    // The driver's RHF stops at density_conv 1e-9: the energy differs from
    // the tight one by the (non-variational) correlation part, ≲ 9.4 · 1e-9.
    assert!(
        (e_drv - e).abs() < 1e-7,
        "driver {e_drv:.12} vs energy {e:.12}"
    );
}

#[test]
fn scs_2terfc_gradient_matches_fd_water() {
    Case::new("H2O/cc-pVDZ", WATER, 0, None).check();
}

#[test]
fn scs_2terfc_gradient_matches_fd_water_rijk_frozen_core() {
    Case::new("H2O/cc-pVDZ RI-JK fc=1", WATER, 1, Some(rijk())).check();
}

#[test]
fn scs_2terfc_gradient_matches_fd_ch3f_frozen_core() {
    Case::new("CH3F/cc-pVDZ fc=2", CH3F, 2, None).check();
}
