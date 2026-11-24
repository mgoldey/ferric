//! The term-list McMurchie-Davidson kernels (`md_terms` in `shim.cc`, behind
//! `scf_compute_terf{,c}_eri3` and `scf_compute_terf_table_eri3_deriv`)
//! against the reference kernels they replace (`compute_cart_eri3` and
//! `compute_cart_eri3_deriv`, reachable as `Engine::debug_reference_eri3{,_deriv}`).
//!
//! # What is checked
//!
//! Every shell triple of water / cc-pVTZ + cc-pVTZ-RI (orbital s..f, aux
//! s..g), for terf and terfc, values and the 9 derivative blocks.
//!
//! * terf values: **bit-identical**. md_terms keeps the reference's loop
//!   order, products and R recurrence; only the bookkeeping changed.
//! * terfc values and all derivatives: md_terms forms `Coulomb − terf` on the
//!   seeds of the R recurrence and contracts once, where the reference
//!   contracts twice and subtracts (values) or subtracts the two R tables
//!   (derivatives), and the derivative products associate differently. Both
//!   are exact rearrangements of one finite sum, so they differ by rounding.
//!
//! # Tolerance (derived)
//!
//! Two association orders of one sum of `N` products differ by at most
//! `~2 N ε Σ|terms|`. The terms of an element are bounded by the Coulomb
//! and terf pieces separately, each at most the largest element of the
//! tensor `M` (≈ 2–6 here, the tight O 1s primitives), and a primitive
//! triple contributes at most a few hundred terms. An element can be far
//! smaller than its terms (Coulomb − terf cancels, and so do contracted
//! primitives), so the scale is the tensor maximum, not the element:
//! `|new − ref| ≤ REL_TOL · M` with `REL_TOL = 1e-13` (about `500 ε`) for
//! every element of every class. That is the 1e-13 absolute bar asked of
//! the kernel, to within the factor `M`. A wrong coefficient, sign, offset or
//! dropped term moves an element by `O(1)` of its size and fails by ten
//! orders of magnitude.
//!
//! Measured (release): terf values bit-identical; terfc values 5.1e-15,
//! terf derivatives 1.6e-15, terfc derivatives 4.1e-15 (absolute, worst
//! element over the 80 classes). One class, (g|ff), is zero in every
//! derivative block: all three shells sit on O, and a one-centre integral of
//! odd total angular momentum vanishes by inversion.
//!
//! # Artifact hypothesis
//!
//! If the offset trick `R[t'+t,u'+u,v'+v] = R[off_P + off_K]` were wrong
//! (stride mismatch), classes with `t + u + v > 0` on both sides would fail
//! while (ss|ss) passed; if the combined seed had the wrong sign, every
//! terfc element would fail while terf passed. Both are separable here
//! because terf and terfc are checked independently per class.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use std::collections::BTreeMap;

const WATER: &str = "3\nwater, off C2v\nO 0.02 -0.01 0.117790\nH 0.0 0.785453 -0.451161\nH 0.03 -0.735453 -0.491161\n";

/// r0 = 1.05 Å in Bohr.
const R0_BOHR: f64 = 1.05 / 0.529_177_210_92;

/// Relative to the largest element of the shell class (derived in the module
/// doc).
const REL_TOL: f64 = 1e-13;

fn tables() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let probe = "16_4_2.bin";
        if let Ok(d) = std::env::var("FERRIC_TERF_TABLE_DIR") {
            assert!(
                std::path::Path::new(&d).join(probe).is_file(),
                "FERRIC_TERF_TABLE_DIR={d} has no {probe}"
            );
            return;
        }
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../terf-tables");
        assert!(
            repo.join(probe).is_file(),
            "terfc tables not found at {}",
            repo.display()
        );
        std::env::set_var("FERRIC_TERF_TABLE_DIR", &repo);
    });
}

fn bases(obs: &str, aux: &str) -> (PreparedBasis, PreparedBasis) {
    tables();
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let o = PreparedBasis::new(&mol, &basis::bundled(obs).unwrap()).unwrap();
    let d = PreparedBasis::new(&mol, &basis::bundled(aux).unwrap()).unwrap();
    (o, d)
}

fn ang(dim: usize) -> usize {
    (dim - 1) / 2
}

/// Per shell class `(lP, l1, l2)`: the largest `|new − ref|` and the
/// largest `|ref|` over every element of the class; and whether every
/// element was bit-identical.
type ClassStats = BTreeMap<(usize, usize, usize), (f64, f64)>;

fn compare(
    op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    derivs: bool,
) -> (ClassStats, bool) {
    let mut eng = if derivs {
        Engine::new_3center_deriv(op, obs, dfbs, 1e-15).unwrap()
    } else {
        Engine::new_3center(op, obs, dfbs, 1e-15).unwrap()
    };
    let (od, dd) = (obs.shell_dims(), dfbs.shell_dims());
    let mut stats = ClassStats::new();
    let mut identical = true;
    for sp in 0..dfbs.nshells() {
        for s1 in 0..obs.nshells() {
            for s2 in 0..obs.nshells() {
                let (new, reference) = if derivs {
                    let r = eng
                        .debug_reference_eri3_deriv(obs, dfbs, sp, s1, s2)
                        .unwrap();
                    (
                        eng.compute_eri3_deriv(obs, dfbs, sp, s1, s2)
                            .unwrap()
                            .to_vec(),
                        r,
                    )
                } else {
                    let r = eng.debug_reference_eri3(obs, dfbs, sp, s1, s2).unwrap();
                    (eng.compute_eri3(obs, dfbs, sp, s1, s2).unwrap().to_vec(), r)
                };
                assert_eq!(new.len(), reference.len());
                identical &= new
                    .iter()
                    .zip(&reference)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                let key = (ang(dd[sp]), ang(od[s1]), ang(od[s2]));
                let e = stats.entry(key).or_insert((0.0_f64, 0.0_f64));
                for (a, b) in new.iter().zip(&reference) {
                    e.0 = e.0.max((a - b).abs());
                    e.1 = e.1.max(b.abs());
                }
            }
        }
    }
    (stats, identical)
}

fn check(op: Operator, derivs: bool, want_identical: bool, label: &str) {
    let (obs, dfbs) = bases("cc-pvtz", "cc-pvtz-ri");
    let (stats, identical) = compare(op, &obs, &dfbs, derivs);
    assert!(
        stats.len() >= 30,
        "{label}: only {} shell classes",
        stats.len()
    );
    let tensor_max = stats.values().fold(0.0_f64, |m, s| m.max(s.1));
    let bar = REL_TOL * tensor_max;
    let mut worst = 0.0_f64;
    for (k, &(diff, _)) in &stats {
        worst = worst.max(diff);
        assert!(
            diff <= bar,
            "{label}: class {k:?} differs by {diff:.3e} (bar {bar:.3e} = {REL_TOL:.0e} x the \
             tensor maximum {tensor_max:.3e})"
        );
    }
    eprintln!(
        "{label}: {} classes, worst {worst:.2e} absolute ({:.2e} of the tensor maximum \
         {tensor_max:.2e}), bit-identical {identical}",
        stats.len(),
        worst / tensor_max
    );
    if want_identical {
        assert!(
            identical,
            "{label}: not bit-identical to the reference kernel"
        );
    }
}

#[test]
fn terf_values_are_bit_identical_to_the_reference_kernel() {
    check(Operator::terf(R0_BOHR), false, true, "terf values");
}

#[test]
fn terfc_values_match_the_reference_kernel() {
    check(Operator::terfc(R0_BOHR), false, false, "terfc values");
}

#[test]
fn terf_derivatives_match_the_reference_kernel() {
    check(Operator::terf(R0_BOHR), true, false, "terf derivatives");
}

#[test]
fn terfc_derivatives_match_the_reference_kernel() {
    check(Operator::terfc(R0_BOHR), true, false, "terfc derivatives");
}

/// Serial wall time of every (P|μν) triple of benzene / cc-pVDZ +
/// cc-pVDZ-RI, values and derivatives, term-list kernel against the
/// reference. A measurement, not a check: run with `--ignored --nocapture`.
#[test]
#[ignore = "timing measurement (~1 min); run with --ignored --nocapture"]
fn time_md_terms_against_reference_on_benzene() {
    tables();
    let xyz = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/molecules/benzene.xyz"),
    )
    .unwrap();
    let mol = Molecule::parse_xyz(&xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::terfc(1.55 / 0.529_177_210_92);
    for derivs in [false, true] {
        let mut eng = if derivs {
            Engine::new_3center_deriv(op, &obs, &dfbs, 1e-15).unwrap()
        } else {
            Engine::new_3center(op, &obs, &dfbs, 1e-15).unwrap()
        };
        let mut run = |reference: bool| {
            let t = std::time::Instant::now();
            let mut sum = 0.0;
            for sp in 0..dfbs.nshells() {
                for s1 in 0..obs.nshells() {
                    for s2 in 0..=s1 {
                        let v: Vec<f64> = match (derivs, reference) {
                            (false, false) => {
                                eng.compute_eri3(&obs, &dfbs, sp, s1, s2).unwrap().to_vec()
                            }
                            (false, true) => {
                                eng.debug_reference_eri3(&obs, &dfbs, sp, s1, s2).unwrap()
                            }
                            (true, false) => eng
                                .compute_eri3_deriv(&obs, &dfbs, sp, s1, s2)
                                .unwrap()
                                .to_vec(),
                            (true, true) => eng
                                .debug_reference_eri3_deriv(&obs, &dfbs, sp, s1, s2)
                                .unwrap(),
                        };
                        sum += v.iter().map(|x| x * x).sum::<f64>();
                    }
                }
            }
            (t.elapsed().as_secs_f64(), sum)
        };
        let (t_ref, s_ref) = run(true);
        let (t_new, s_new) = run(false);
        eprintln!(
            "{}: reference {t_ref:.2} s, term-list {t_new:.2} s, ratio {:.2}; sum2 {s_ref:.15e} vs {s_new:.15e}",
            if derivs { "derivatives" } else { "values" },
            t_ref / t_new
        );
    }
}
