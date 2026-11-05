//! A converged SCF must be a fixed point of one plain Roothaan step.
//!
//! At exit, build `F[D]` from the returned density, diagonalize it once, fill
//! the aufbau occupation to get `D'`, and evaluate `E[D']`. At a stationary
//! minimum the orbital gradient vanishes and one Roothaan step cannot lower
//! the energy: `E[D'] >= E[D] - tol`. At a false exit (a stalled ADIIS/EDIIS
//! extrapolation, or a frozen DIIS history) the gradient is O(0.1) and the
//! step drops the energy by O(0.1) Ha. The check needs no reference value.
//!
//! Why not "E_final <= min of earlier iterates": every measured false exit
//! (SO2/CS2 SAD + ADIIS/EDIIS stopping at iteration 3, benzene in 2000 point
//! charges with EDIIS stopping at iteration 4-5) sits at the lowest
//! idempotent energy reached so far, so that history check passes on them.
//!
//! Tolerance. To second order in the occupied-virtual rotation `kappa` of the
//! Roothaan step (orthonormal basis, `F_ia = (e_a - e_i) kappa_ia`),
//! `dE ~ -sum_ia F_ia^2 / (e_a - e_i) = -sum_ia (e_a - e_i) kappa_ia^2`,
//! so `|dE| <= span(e) * |kappa|_F^2` with `span(e)` the eigenvalue range of
//! `F`. At a converged exit the step changes the density by at most the
//! gate's density residual, `|kappa|_F <= |dD|_F <= n * density_conv`
//! (`dp_rms < density_conv` over `n^2` elements). The energy gate adds
//! `energy_conv` as slack for the energy change one more iteration may still
//! carry. Hence
//!
//! `tol = energy_conv + span(e) * (n * density_conv)^2`.
//!
//! The evaluator is anchored: `E[D]` recomputed here must reproduce the SCF's
//! own energy (`ANCHOR`), so `E[D'] - E[D]` compares like with like.
//!
//! The solver reports the first-order part of the same step at every
//! converged exit (`ConvergenceReport::one_step_energy_drop`, no extra Fock
//! build); each case also asserts that it reaches the same verdict.
//!
//! Mutation (recorded in the commit message): with the #411 guess veto, the
//! #431 commutator veto and the orbital-gradient veto all disabled, SO2, CS2
//! and benzene/2000 report `converged` at a false exit and this check fails.

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::diis::DiisFlavor;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{build_jk, solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::{s, Array2};
use ndarray_linalg::{Eigh, UPLO};

/// The recomputed `E[D_final]` must equal the SCF's energy to this (Ha).
const ANCHOR: f64 = 1e-9;

const FLAVORS: [DiisFlavor; 3] = [DiisFlavor::Pulay, DiisFlavor::Adiis, DiisFlavor::Ediis];
const SHIFTS: [f64; 2] = [0.0, 0.5];

struct System {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ext: Option<ExternalPotential>,
}

impl System {
    fn new(mol: Molecule, ext: Option<ExternalPotential>) -> Self {
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        System {
            mol,
            prep,
            bounds,
            ext,
        }
    }

    fn load(name: &str) -> Self {
        Self::new(
            Molecule::load_xyz(&format!("../../testdata/molecules/{name}.xyz")).unwrap(),
            None,
        )
    }

    /// HF energy and spin Fock matrices at spin densities `(D_a, D_b)`:
    /// `F_s = h + J[D_a + D_b] - K[D_s]`,
    /// `E = Tr[D h] + 1/2 Tr[D J] - 1/2 sum_s Tr[D_s K[D_s]] + V_nn`.
    fn energy(&self, da: &Array2<f64>, db: &Array2<f64>) -> (f64, Array2<f64>, Array2<f64>) {
        let ctx = ParallelContext::default();
        let thresh = RhfConfig::default().integral_thresh;
        let h = oneelectron::hcore_with_external(&self.prep, self.ext.as_ref()).unwrap();
        let n = h.nrows();
        let jk = |d: &Array2<f64>| {
            let (mut j, mut k) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
            build_jk(&ctx, &self.prep, &self.bounds, thresh, d, &mut j, &mut k).unwrap();
            (j, k)
        };
        let dt = da + db;
        let (j, _) = jk(&dt);
        let (_, ka) = jk(da);
        let (_, kb) = jk(db);
        let vnn = self.mol.nuclear_repulsion()
            + self.ext.as_ref().map_or(0.0, |e| {
                e.charge_nuclear_energy(&self.mol) + e.field_nuclear_energy(&self.mol)
            });
        let e = (&dt * &h).sum() + 0.5 * (&dt * &j).sum()
            - 0.5 * ((da * &ka).sum() + (db * &kb).sum())
            + vnn;
        (e, &h + &j - &ka, &h + &j - &kb)
    }
}

/// Aufbau density of `nocc` orbitals of `F` in the `S` metric, and the
/// eigenvalue range of `F`.
fn roothaan_density(f: &Array2<f64>, s_mat: &Array2<f64>, nocc: usize) -> (Array2<f64>, f64) {
    let (w, v) = s_mat.eigh(UPLO::Lower).unwrap();
    let x = &v * &w.mapv(|x| 1.0 / x.sqrt());
    let (eps, cp) = x.t().dot(f).dot(&x).eigh(UPLO::Lower).unwrap();
    let c = x.dot(&cp);
    let co = c.slice(s![.., ..nocc]);
    (co.dot(&co.t()), eps[eps.len() - 1] - eps[0])
}

/// One Roothaan step from a converged result: `(E[D'] - E[D], tol)`.
fn one_step_drop(sys: &System, r: &ScfResult, cfg: &RhfConfig) -> (f64, f64) {
    let da = r.density_alpha.clone();
    let db = r
        .density_beta
        .clone()
        .unwrap_or_else(|| r.density_alpha.clone());
    let (e0, fa, fb) = sys.energy(&da, &db);
    assert!(
        (e0 - r.energy).abs() < ANCHOR,
        "evaluator does not reproduce the SCF energy: {e0} vs {}",
        r.energy
    );
    let s_mat = oneelectron::overlap(&sys.prep);
    let nelec = sys.mol.nelec() as usize;
    let nb = (nelec - (sys.mol.multiplicity - 1)) / 2;
    let na = nelec - nb;
    let (da1, span_a) = roothaan_density(&fa, &s_mat, na);
    let (db1, span_b) = roothaan_density(&fb, &s_mat, nb);
    let (e1, _, _) = sys.energy(&da1, &db1);
    let n = s_mat.nrows() as f64;
    let tol = cfg.energy_conv + span_a.max(span_b) * (n * cfg.density_conv).powi(2);
    (e1 - e0, tol)
}

fn cfg(flavor: DiisFlavor, level_shift: f64, sad: bool, tight: bool) -> RhfConfig {
    let mut c = RhfConfig {
        diis_flavor: flavor,
        level_shift,
        use_sad_guess: sad,
        max_iter: 200,
        ..Default::default()
    };
    if tight {
        c.energy_conv = 1e-10;
        c.density_conv = 1e-8;
    }
    c
}

/// Check one result; returns the violation, if any. Unconverged runs are
/// not claims and are skipped.
fn check(label: &str, sys: &System, r: &ScfResult, c: &RhfConfig) -> Option<String> {
    if !r.converged {
        eprintln!(
            "{label}: not converged ({} iterations), skipped",
            r.iterations
        );
        return None;
    }
    let (drop, tol) = one_step_drop(sys, r, c);
    // The solver's own first-order report (`one_step_energy_drop`) must
    // reach the same verdict as the exact one-step energy.
    let model = r
        .convergence
        .and_then(|c| c.one_step_energy_drop)
        .expect("a converged aufbau RHF/UHF exit reports one_step_energy_drop");
    eprintln!(
        "{label}: E={:.10} it={} one-step dE={drop:.3e} model={model:.3e} tol={tol:.3e}",
        r.energy, r.iterations
    );
    assert!(
        model <= 1e-12,
        "{label}: first-order model drop {model} > 0"
    );
    assert_eq!(
        model < -tol,
        drop < -tol,
        "{label}: report (model {model:.3e}) and exact one-step dE ({drop:.3e}) disagree at tol {tol:.1e}"
    );
    (drop < -tol).then(|| {
        format!(
            "{label}: converged at E={:.10} after {} iterations, but one Roothaan step \
             lowers it by {:.3e} Ha (tol {tol:.1e}): not a stationary point",
            r.energy, r.iterations, -drop
        )
    })
}

fn rhf_case(sys: &System, label: &str, c: &RhfConfig, failures: &mut Vec<String>) {
    let r = solve_rhf(
        &ParallelContext::default(),
        &sys.mol,
        &sys.prep,
        Operator::coulomb(),
        &sys.bounds,
        c,
    )
    .unwrap();
    failures.extend(check(label, sys, &r, c));
}

/// SO2 and CS2 / STO-3G, SAD guess: ADIIS and EDIIS (level shift 0 or 0.5)
/// once reported `converged` at iteration 3, 0.163 Ha (SO2) and 0.098 Ha (CS2)
/// above the RHF minimum. Every flavour x shift; water is the control.
#[test]
fn a_converged_rhf_is_a_fixed_point_of_one_roothaan_step() {
    let mut failures = Vec::new();
    for name in ["so2", "cs2", "water"] {
        let sys = System::load(name);
        for flavor in FLAVORS {
            for ls in SHIFTS {
                let label = format!("{name} RHF {flavor:?} level_shift={ls} SAD");
                rhf_case(&sys, &label, &cfg(flavor, ls, true, true), &mut failures);
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// 2000 deterministic +-0.8 e charges 9-23 Bohr from the centroid (>= 5 Bohr
/// from every atom), net -3 e (the #422 construction of
/// `adiis_stalled_extrapolation.rs`).
fn field(mol: &Molecule, n: usize, seed: u64, net: f64) -> ExternalPotential {
    let mut st = seed;
    let mut rnd = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let na = mol.atoms.len() as f64;
    let (cx, cy, cz) = mol.atoms.iter().fold((0.0, 0.0, 0.0), |a, at| {
        (a.0 + at.x / na, a.1 + at.y / na, a.2 + at.zpos / na)
    });
    let mut pcs = Vec::new();
    while pcs.len() < n {
        let (x, y, z) = (rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0);
        let r = (x * x + y * y + z * z).sqrt();
        if !(1e-3..=1.0).contains(&r) {
            continue;
        }
        let rad = 9.0 + rnd() * 14.0;
        let (px, py, pz) = (cx + x / r * rad, cy + y / r * rad, cz + z / r * rad);
        if mol
            .atoms
            .iter()
            .any(|a| ((a.x - px).powi(2) + (a.y - py).powi(2) + (a.zpos - pz).powi(2)).sqrt() < 5.0)
        {
            continue;
        }
        let q = if rnd() < 0.5 { 0.8 } else { -0.8 } + net / n as f64;
        pcs.push(PointCharge {
            q,
            x: px,
            y: py,
            z: pz,
        });
    }
    ExternalPotential {
        point_charges: pcs,
        ..Default::default()
    }
}

/// Benzene / STO-3G in 2000 point charges (seed 11), EDIIS at the default
/// thresholds: once reported `converged` at iteration 4-5 with err_max 0.45,
/// 0.164 Ha above the Pulay energy.
#[test]
fn a_converged_ediis_in_a_field_is_a_fixed_point_of_one_roothaan_step() {
    let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
    let ext = field(&mol, 2000, 11, -3.0);
    let sys = System::new(mol, Some(ext.clone()));
    let mut failures = Vec::new();
    for ls in SHIFTS {
        let c = RhfConfig {
            diis_switch_thresh: 1e-1,
            external_potential: Some(ext.clone()),
            ..cfg(DiisFlavor::Ediis, ls, true, false)
        };
        let label = format!("benzene/2000 RHF Ediis level_shift={ls}");
        rhf_case(&sys, &label, &c, &mut failures);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// UHF (always Pulay) carries the same check per spin: OH doublet and O2
/// triplet / STO-3G across level shift x guess.
#[test]
fn a_converged_uhf_is_a_fixed_point_of_one_roothaan_step() {
    let mut failures = Vec::new();
    for (name, mult) in [("oh", 2), ("o2", 3)] {
        let xyz = std::fs::read_to_string(format!("../../testdata/molecules/{name}.xyz")).unwrap();
        let sys = System::new(Molecule::parse_xyz(&xyz, 0, mult).unwrap(), None);
        for ls in SHIFTS {
            for sad in [true, false] {
                let c = cfg(DiisFlavor::Pulay, ls, sad, true);
                let r = solve_uhf(
                    &ParallelContext::default(),
                    &sys.mol,
                    &sys.prep,
                    &sys.bounds,
                    &c,
                )
                .unwrap();
                let label = format!("{name} UHF level_shift={ls} sad={sad}");
                failures.extend(check(&label, &sys, &r, &c));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
