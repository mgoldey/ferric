//! Opt-in nonbonded cutoff (`ferric_mm::cutoff`, theory.md section 8).
//!
//! - exactness anchors: `NoCutoff` is the all-pairs sum bit for bit; a cutoff
//!   beyond the system diameter with `eps_rf = 1` reproduces the all-pairs LJ
//!   energy and the whole gradient bit for bit and shifts Coulomb by exactly
//!   `-c_rf * sum(q_i q_j)`; at general `eps_rf` the difference is the
//!   independently summed reaction-field term;
//! - the cell list returns exactly the brute-force pair set;
//! - central finite differences on a solvated ethanol whose pairs straddle
//!   the cutoff and the switching region;
//! - OpenMM `NonbondedForce.CutoffNonPeriodic` (Reference platform) on the
//!   same system, references from `scripts/gen_openmm_mm_cutoff_refs.py`.

use ferric_mm::units::{ANGSTROM_TO_BOHR, KCAL_PER_MOL_TO_HARTREE};
use ferric_mm::{
    energy_with, gradient, gradient_with, neighbor_pairs, CutoffParams, LjParams, MmTopology,
    NonbondedMethod,
};
use ndarray::Array2;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    cutoff_angstrom: f64,
    reaction_field_dielectric: f64,
    switch_angstrom: Option<f64>,
    nonbonded_energy_kcal: f64,
    nonbonded_forces_kcal_per_angstrom: Vec<[f64; 3]>,
}

#[derive(Debug, Deserialize)]
struct Ref {
    n_atoms: usize,
    coordinates_angstrom: Vec<[f64; 3]>,
    charges: Vec<f64>,
    lj_sigma_angstrom: Vec<f64>,
    lj_epsilon_kcal: Vec<f64>,
    bonds: Vec<(usize, usize, f64, f64)>,
    angles: Vec<(usize, usize, usize, f64, f64)>,
    torsions: Vec<(usize, usize, usize, usize, u32, f64, f64)>,
    scale_lj_14: f64,
    scale_coul_14: f64,
    coulomb_constant_kj_nm: f64,
    cases: Vec<Case>,
}

fn load() -> Ref {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/mm_cutoff_ethanol_12w_openmm.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).unwrap()
}

/// Topology with every charge multiplied by `charge_scale`.
fn topology(r: &Ref, charge_scale: f64) -> MmTopology {
    MmTopology::from_amber_units(
        r.charges.iter().map(|q| q * charge_scale).collect(),
        r.lj_sigma_angstrom
            .iter()
            .copied()
            .zip(r.lj_epsilon_kcal.iter().copied())
            .collect(),
        r.bonds.clone(),
        r.angles.clone(),
        r.torsions.clone(),
    )
    .unwrap()
    .with_scales(r.scale_lj_14, r.scale_coul_14)
}

fn coords(r: &Ref) -> Array2<f64> {
    let mut c = Array2::<f64>::zeros((r.n_atoms, 3));
    for (i, xyz) in r.coordinates_angstrom.iter().enumerate() {
        for k in 0..3 {
            c[(i, k)] = xyz[k] * ANGSTROM_TO_BOHR;
        }
    }
    c
}

fn dist(c: &Array2<f64>, i: usize, j: usize) -> f64 {
    ((c[(i, 0)] - c[(j, 0)]).powi(2)
        + (c[(i, 1)] - c[(j, 1)]).powi(2)
        + (c[(i, 2)] - c[(j, 2)]).powi(2))
    .sqrt()
}

fn diameter(c: &Array2<f64>) -> f64 {
    let n = c.nrows();
    let mut d = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            d = d.max(dist(c, i, j));
        }
    }
    d
}

fn params(case: &Case) -> CutoffParams {
    let mut p = CutoffParams::new(case.cutoff_angstrom * ANGSTROM_TO_BOHR)
        .with_reaction_field_dielectric(case.reaction_field_dielectric);
    if let Some(sw) = case.switch_angstrom {
        p = p.with_switch_distance(sw * ANGSTROM_TO_BOHR);
    }
    p
}

/// Ordinary (neither excluded nor 1-4) pairs, lexicographic.
fn ordinary_pairs(top: &MmTopology) -> Vec<(usize, usize)> {
    let n = top.n_atoms();
    let mut v = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if !top.exclusions().contains(&(i, j)) && !top.pairs14().contains(&(i, j)) {
                v.push((i, j));
            }
        }
    }
    v
}

fn bits(g: &Array2<f64>) -> Vec<u64> {
    g.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn no_cutoff_is_the_all_pairs_sum_bit_for_bit() {
    let r = load();
    let (top, c) = (topology(&r, 1.0), coords(&r));
    let (e0, g0) = gradient(&top, &c).unwrap();
    let (e1, g1) = gradient_with(&top, &c, &NonbondedMethod::NoCutoff).unwrap();
    assert_eq!(e0, e1);
    assert_eq!(bits(&g0), bits(&g1));
    assert_eq!(NonbondedMethod::default(), NonbondedMethod::NoCutoff);
}

#[test]
fn cutoff_beyond_diameter_eps_rf_1_is_all_pairs_plus_constant_shift_bit_for_bit() {
    let r = load();
    let (top, c) = (topology(&r, 1.0), coords(&r));
    let rc = 2.0 * diameter(&c);
    // Switching starts beyond every pair as well: S = 1 everywhere.
    for p in [
        CutoffParams::new(rc).with_reaction_field_dielectric(1.0),
        CutoffParams::new(rc)
            .with_reaction_field_dielectric(1.0)
            .with_switch_distance(0.75 * rc),
    ] {
        let (k_rf, c_rf) = p.reaction_field_constants();
        assert_eq!(k_rf, 0.0);
        let (e_all, g_all) = gradient(&top, &c).unwrap();
        let (e_cut, g_cut) =
            gradient_with(&top, &c, &NonbondedMethod::CutoffNonPeriodic(p)).unwrap();
        assert_eq!(e_cut.bond, e_all.bond);
        assert_eq!(e_cut.angle, e_all.angle);
        assert_eq!(e_cut.torsion, e_all.torsion);
        assert_eq!(e_cut.lj.to_bits(), e_all.lj.to_bits());
        assert_eq!(bits(&g_cut), bits(&g_all), "gradient must be bit-identical");
        let mut sum_qq = 0.0_f64;
        for (i, j) in ordinary_pairs(&top) {
            let qq = top.charges[i] * top.charges[j];
            if qq != 0.0 {
                sum_qq += qq;
            }
        }
        assert!(sum_qq.abs() > 1e-3, "shift must be visible: {sum_qq}");
        assert_eq!(
            e_cut.coulomb.to_bits(),
            (e_all.coulomb - c_rf * sum_qq).to_bits(),
            "Coulomb = all-pairs - c_rf * sum(qq)"
        );
    }
}

#[test]
fn cutoff_beyond_diameter_differs_from_all_pairs_by_the_reaction_field_term() {
    let r = load();
    let (top, c) = (topology(&r, 1.0), coords(&r));
    let rc = 2.0 * diameter(&c);
    for eps in [2.0, 78.3, 1e6] {
        let p = CutoffParams::new(rc).with_reaction_field_dielectric(eps);
        let (k_rf, c_rf) = p.reaction_field_constants();
        let e_all = gradient(&top, &c).unwrap().0;
        let e_cut = energy_with(&top, &c, &NonbondedMethod::CutoffNonPeriodic(p)).unwrap();
        let mut expect = 0.0_f64;
        for (i, j) in ordinary_pairs(&top) {
            let d = dist(&c, i, j);
            expect += top.charges[i] * top.charges[j] * (k_rf * d * d - c_rf);
        }
        assert_eq!(e_cut.lj, e_all.lj);
        let got = e_cut.coulomb - e_all.coulomb;
        assert!(
            (got - expect).abs() < 1e-14,
            "eps_rf {eps}: shift {got:.17e} vs {expect:.17e}"
        );
        assert!(expect.abs() > 1e-4);
    }
}

#[test]
fn neighbor_pairs_matches_brute_force() {
    // Deterministic LCG coordinates: one dense cluster and a sparse halo.
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let n = 160;
    let mut c = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let span = if i < 120 { 12.0 } else { 60.0 };
        for k in 0..3 {
            c[(i, k)] = span * (next() - 0.5) - 3.0;
        }
    }
    for rc in [0.7, 2.5, 6.0, 13.3, 1e4] {
        let mut brute = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if dist(&c, i, j).powi(2) < rc * rc {
                    brute.push((i, j));
                }
            }
        }
        let got = neighbor_pairs(&c, rc);
        assert_eq!(got, brute, "cutoff {rc}");
        assert!(!brute.is_empty());
    }
    assert!(neighbor_pairs(&Array2::zeros((1, 3)), 5.0).is_empty());
}

#[test]
fn invalid_cutoff_params_are_errors() {
    let r = load();
    let (top, c) = (topology(&r, 1.0), coords(&r));
    for p in [
        CutoffParams::new(0.0),
        CutoffParams::new(-1.0),
        CutoffParams::new(f64::NAN),
        CutoffParams::new(f64::INFINITY),
        CutoffParams::new(9.0).with_reaction_field_dielectric(0.0),
        CutoffParams::new(9.0).with_switch_distance(9.0),
        CutoffParams::new(9.0).with_switch_distance(-0.1),
    ] {
        assert!(
            gradient_with(&top, &c, &NonbondedMethod::CutoffNonPeriodic(p)).is_err(),
            "{p:?} must be rejected"
        );
    }
}

/// Two-atom energy at separation `d` along x.
fn dimer_energy(p: CutoffParams, d: f64) -> (f64, f64) {
    let top = MmTopology::new(
        vec![0.4, -0.7],
        vec![
            LjParams {
                sigma: 6.0,
                epsilon: 3e-4,
            };
            2
        ],
        vec![],
        vec![],
        vec![],
    )
    .unwrap();
    let mut c = Array2::<f64>::zeros((2, 3));
    c[(1, 0)] = d;
    let e = energy_with(&top, &c, &NonbondedMethod::CutoffNonPeriodic(p)).unwrap();
    (e.lj, e.coulomb)
}

#[test]
fn reaction_field_and_switched_lj_vanish_continuously_at_the_cutoff() {
    let rc = 9.0;
    let d = 1e-9;
    let p = CutoffParams::new(rc).with_switch_distance(7.0);
    let (lj_in, coul_in) = dimer_energy(p, rc - d);
    let (lj_out, coul_out) = dimer_energy(p, rc + d);
    assert_eq!((lj_out, coul_out), (0.0, 0.0));
    assert!(coul_in.abs() < 1e-10, "RF Coulomb at r_c^-: {coul_in:e}");
    assert!(lj_in.abs() < 1e-18, "switched LJ at r_c^-: {lj_in:e}");
    // Inside the switching start the LJ is untouched; at the cutoff without
    // switching it is truncated, not shifted (OpenMM's convention).
    let (lj_plain_in, _) = dimer_energy(CutoffParams::new(rc), rc - d);
    let x = (6.0_f64 / (rc - d)).powi(6);
    assert!((lj_plain_in - 4.0 * 3e-4 * (x * x - x)).abs() < 1e-18);
    assert!(lj_plain_in.abs() > 1e-6);
}

fn fd_check(top: &MmTopology, c: &Array2<f64>, p: CutoffParams) -> f64 {
    let m = NonbondedMethod::CutoffNonPeriodic(p);
    let (_, g) = gradient_with(top, c, &m).unwrap();
    let h = 1e-5;
    let mut max_err = 0.0_f64;
    for i in 0..c.nrows() {
        for k in 0..3 {
            let mut cp = c.clone();
            cp[(i, k)] += h;
            let mut cm = c.clone();
            cm[(i, k)] -= h;
            let fd = (energy_with(top, &cp, &m).unwrap().total
                - energy_with(top, &cm, &m).unwrap().total)
                / (2.0 * h);
            max_err = max_err.max((fd - g[(i, k)]).abs());
        }
    }
    max_err
}

#[test]
fn gradient_matches_fd_with_pairs_straddling_cutoff_and_switch() {
    let r = load();
    let (top, c) = (topology(&r, 1.0), coords(&r));
    let ordinary = ordinary_pairs(&top);
    for case in &r.cases {
        let p = params(case);
        // No pair may sit within the FD stencil of an edge (truncated LJ and
        // the RF force jump there), and pairs must populate every region.
        let rsw = p.switch_distance.unwrap_or(p.cutoff);
        let (mut inner, mut switch, mut outer) = (0, 0, 0);
        for &(i, j) in &ordinary {
            let d = dist(&c, i, j);
            assert!((d - p.cutoff).abs() > 1e-3 && (d - rsw).abs() > 1e-3);
            if d < rsw {
                inner += 1;
            } else if d < p.cutoff {
                switch += 1;
            } else {
                outer += 1;
            }
        }
        assert!(
            inner > 0 && outer > 0,
            "{}: {inner} in, {outer} out",
            case.name
        );
        if p.switch_distance.is_some() {
            assert!(switch > 0, "{}: no pair in the switching region", case.name);
        }
        let err = fd_check(&top, &c, p);
        eprintln!(
            "{}: {inner} inner / {switch} switch / {outer} outer pairs, FD err {err:.2e}",
            case.name
        );
        assert!(err < 1e-10, "{}: max |FD - analytic| {err:.3e}", case.name);
    }
}

#[test]
fn matches_openmm_cutoff_nonperiodic() {
    let r = load();
    // OpenMM's Coulomb constant (CODATA 2018) is not ferric's Hartree*Bohr;
    // scaling every charge by sqrt(ratio) makes the Coulomb constants equal,
    // so the comparison tests the functional form only.
    let ferric_ke_kj_nm = 627.509_474 * 4.184 * (0.1 / ANGSTROM_TO_BOHR);
    let ratio = r.coulomb_constant_kj_nm / ferric_ke_kj_nm;
    let top = topology(&r, ratio.sqrt());
    let c = coords(&r);
    let to_ha_bohr = KCAL_PER_MOL_TO_HARTREE / ANGSTROM_TO_BOHR;
    let g_bonded = bonded_gradient(&top, &c);
    for case in &r.cases {
        let m = NonbondedMethod::CutoffNonPeriodic(params(case));
        let (e, g) = gradient_with(&top, &c, &m).unwrap();
        let e_ref = case.nonbonded_energy_kcal * KCAL_PER_MOL_TO_HARTREE;
        let rel_e = ((e.lj + e.coulomb) - e_ref).abs() / e_ref.abs();
        let mut max_err = 0.0_f64;
        let mut max_ref = 0.0_f64;
        for (i, f) in case.nonbonded_forces_kcal_per_angstrom.iter().enumerate() {
            for k in 0..3 {
                let g_nb = g[(i, k)] - g_bonded[(i, k)];
                max_err = max_err.max((g_nb + f[k] * to_ha_bohr).abs());
                max_ref = max_ref.max((f[k] * to_ha_bohr).abs());
            }
        }
        let rel_g = max_err / max_ref;
        eprintln!(
            "{}: energy rel {rel_e:.2e}, gradient rel {rel_g:.2e}",
            case.name
        );
        assert!(rel_e < 1e-12, "{}: energy rel err {rel_e:.2e}", case.name);
        assert!(rel_g < 1e-12, "{}: gradient rel err {rel_g:.2e}", case.name);
    }
}

/// Bonded-only gradient: the same topology with every charge and epsilon
/// zeroed (cutoff-independent).
fn bonded_gradient(top: &MmTopology, c: &Array2<f64>) -> Array2<f64> {
    let mut b = top.clone();
    b.charges.iter_mut().for_each(|q| *q = 0.0);
    b.lj.iter_mut().for_each(|p| p.epsilon = 0.0);
    gradient(&b, c).unwrap().1
}
