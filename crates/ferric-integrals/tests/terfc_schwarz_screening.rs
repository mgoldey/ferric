//! Validity of the terfc Schwarz bound against TRUE integrals.
//!
//! The 3-index screen used by the integral-direct local MP2 skips a shell
//! triple when `Qaux[P] * Q[a,b] < tau`, with `Q[a,b] = sqrt(max (ab|ab))` and
//! `Qaux[P] = sqrt(max (P|P))`, both from the terfc table engine's own
//! diagonals. This file checks the one property that matters: the bound never
//! undershoots a true integral, `max |(P|ab)| <= Qaux[P] * Q[a,b]` for every
//! shell triple (and `|(ab|cd)| <= Q_ab Q_cd` for every quartet), across
//! operators, shell types (d, diffuse) and sharp seams.
//!
//! Mutation arms: a bound built with the wrong diagonal, or halved, MUST be
//! caught by the same check; otherwise the check could not fail.
//!
//! Operators (Bohr internally; the Angstrom values are the production ones):
//!   linked  r0 = 1.000 A, r0*omega = 1/sqrt2
//!   NCP     r0 = 0.991 A, omega = 1.3663 A^-1   (r0*omega = 1.354)
//!   sharp   r0 = 1.000 A, r0*omega = 2
//!   sharpest r0 = 1.408 A, r0*omega = 4
//! NCP and sharper have E_att/E_Coul > 1 in the LMP2 correlation energy; the
//! kernel is bounded by Coulomb pointwise in r, but that does not make the
//! Coulomb diagonal a bound for it (the Fourier transform 1 - cos(k r0)
//! e^{-k^2/4w^2} exceeds Coulomb's 1 where cos < 0); the Coulomb-diagonal
//! mutant below is therefore expected to be caught too.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::schwarz::{bound_inflation, schwarz, schwarz3_aux};
use ndarray::Array2;
use std::sync::Once;

const ANG: f64 = 1.0 / 0.529_177_210_903;

fn ensure_tables() -> bool {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if std::env::var("FERRIC_TERF_TABLE_DIR").is_err() {
            let d = concat!(env!("CARGO_MANIFEST_DIR"), "/../../terf-tables");
            if std::path::Path::new(d).is_dir() {
                std::env::set_var("FERRIC_TERF_TABLE_DIR", d);
            }
        }
    });
    std::env::var("FERRIC_TERF_TABLE_DIR").is_ok()
}

fn terfc_ang(r0_ang: f64, omega_inv_ang: f64) -> Operator {
    Operator::terfc_with_omega(r0_ang * ANG, omega_inv_ang / ANG)
}

fn operators() -> Vec<(&'static str, Operator)> {
    vec![
        ("linked", terfc_ang(1.0, 1.0 / std::f64::consts::SQRT_2)),
        ("NCP", terfc_ang(0.991, 1.3663)),
        ("sharp s=2", terfc_ang(1.0, 2.0)),
        ("sharpest s=4", terfc_ang(1.408, 4.0 / 1.408)),
    ]
}

fn prep(xyz: &str, orb: &str) -> PreparedBasis {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    PreparedBasis::new(&mol, &basis::bundled(orb).unwrap()).unwrap()
}

/// (violations, significant triples, min bound/true over significant).
struct Report {
    viol: usize,
    sig: usize,
    min_ratio: f64,
    median_ratio: f64,
}

/// Compare `qa[P] * q[a,b]` with the true max |(P|ab)| for every shell triple.
fn triple_report(
    op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    q: &Array2<f64>,
    qa: &[f64],
) -> Report {
    let mut eng = Engine::new_3center(op, obs, dfbs, 1e-14).unwrap();
    let (mut viol, mut sig) = (0usize, 0usize);
    let mut ratios = Vec::new();
    for sp in 0..dfbs.nshells() {
        for ua in 0..obs.nshells() {
            for ub in 0..=ua {
                let t = eng
                    .compute_eri3(obs, dfbs, sp, ua, ub)
                    .map(|b| b.iter().fold(0.0f64, |m, &v| m.max(v.abs())))
                    .unwrap_or(0.0);
                if t <= 1e-12 {
                    continue;
                }
                sig += 1;
                let bound = qa[sp] * q[(ua, ub)];
                let r = bound / t;
                ratios.push(r);
                if bound < t {
                    viol += 1;
                }
            }
        }
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Report {
        viol,
        sig,
        min_ratio: ratios.first().copied().unwrap_or(f64::INFINITY),
        median_ratio: ratios
            .get(ratios.len() / 2)
            .copied()
            .unwrap_or(f64::INFINITY),
    }
}

fn bound_tables(
    diag_op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
) -> (Array2<f64>, Vec<f64>) {
    (
        schwarz(diag_op, obs).unwrap(),
        schwarz3_aux(diag_op, dfbs).unwrap(),
    )
}

fn check_set(label: &str, obs: &PreparedBasis, dfbs: &PreparedBasis) {
    for (name, op) in operators() {
        let (q, qa) = bound_tables(op, obs, dfbs);
        let r = triple_report(op, obs, dfbs, &q, &qa);
        eprintln!(
            "{label} {name}: significant {} viol {} min bound/true {:.6} median {:.1}",
            r.sig, r.viol, r.min_ratio, r.median_ratio
        );
        assert!(
            r.sig > 100,
            "{label} {name}: vacuous set ({} significant)",
            r.sig
        );
        assert_eq!(
            r.viol, 0,
            "{label} {name}: Schwarz UNDERSHOT a true integral"
        );
        assert!(
            r.min_ratio >= 1.0,
            "{label} {name}: min ratio {}",
            r.min_ratio
        );
    }
}

#[test]
fn inflation_is_exactly_one_for_table_free_operators_and_small_for_terfc() {
    assert_eq!(bound_inflation(Operator::coulomb()), 1.0);
    assert_eq!(bound_inflation(Operator::erfc(0.4)), 1.0);
    let t = bound_inflation(Operator::terfc(1.0));
    assert!(t > 1.0 && t < 1.0 + 1e-4, "{t}");
}

/// d shells (cc-pVDZ methane + RI aux), all triples, all four operators.
#[test]
fn terfc_schwarz_never_undershoots_triples_with_d_shells() {
    if !ensure_tables() {
        eprintln!("SKIP: terf tables absent");
        return;
    }
    let obs = prep("methane.xyz", "cc-pvdz");
    let dfbs = prep("methane.xyz", "cc-pvdz-ri");
    check_set("methane/cc-pVDZ", &obs, &dfbs);
}

/// The larger set (ethane, 33k significant triples per operator): release,
/// 5.9 s on a loaded box (2026-10-11).
#[test]
#[ignore = "ethane/cc-pVDZ, ~6 s release"]
fn terfc_schwarz_never_undershoots_triples_ethane() {
    if !ensure_tables() {
        return;
    }
    let obs = prep("alkane_2.xyz", "cc-pvdz");
    let dfbs = prep("alkane_2.xyz", "cc-pvdz-ri");
    check_set("ethane/cc-pVDZ", &obs, &dfbs);
}

/// Diffuse shells (aug-cc-pVDZ water).
#[test]
fn terfc_schwarz_never_undershoots_triples_with_diffuse_shells() {
    if !ensure_tables() {
        eprintln!("SKIP: terf tables absent");
        return;
    }
    let obs = prep("water.xyz", "aug-cc-pvdz");
    let dfbs = prep("water.xyz", "cc-pvdz-ri");
    check_set("water/aug-cc-pVDZ", &obs, &dfbs);
}

/// Quartets: |(ab|cd)| <= Q_ab Q_cd, the property the QQR/direct-J users of
/// the same table would rely on. Water/cc-pVDZ, every shell quartet.
#[test]
fn terfc_schwarz_never_undershoots_quartets() {
    if !ensure_tables() {
        eprintln!("SKIP: terf tables absent");
        return;
    }
    let obs = prep("water.xyz", "cc-pvdz");
    // linked and sharpest only: ~7 s release for the pair (13.7 s for all four)
    let ops = operators();
    for (name, op) in [ops[0], ops[3]] {
        let q = schwarz(op, &obs).unwrap();
        let mut eng = Engine::new_2e(op, &obs, 0.0).unwrap();
        let n = obs.nshells();
        let (mut viol, mut sig) = (0usize, 0usize);
        for a in 0..n {
            for b in 0..=a {
                for c in 0..n {
                    for d in 0..=c {
                        let t = eng
                            .compute_quartet(&obs, a, b, c, d)
                            .map(|blk| blk.iter().fold(0.0f64, |m, &v| m.max(v.abs())))
                            .unwrap_or(0.0);
                        if t <= 1e-12 {
                            continue;
                        }
                        sig += 1;
                        if q[(a, b)] * q[(c, d)] < t {
                            viol += 1;
                        }
                    }
                }
            }
        }
        eprintln!("quartets {name}: significant {sig} viol {viol}");
        assert!(sig > 1000);
        assert_eq!(viol, 0, "{name}: quartet bound undershot");
    }
}

/// MUTATION ARMS. The check above must be able to fail.
///  * halved factors: bound x 0.25 must violate on every operator;
///  * the Coulomb diagonal is LOOSER (median) and, because terfc <= Coulomb holds
///    pointwise in r but not in Fourier space, not even rigorous: it must be
///    caught on at least one operator;
///  * the erfc diagonal at the same omega is the wrong operator for these
///    seams and must be caught where it undershoots.
#[test]
fn mutants_are_caught_by_the_validity_check() {
    if !ensure_tables() {
        eprintln!("SKIP: terf tables absent");
        return;
    }
    let obs = prep("methane.xyz", "cc-pvdz");
    let dfbs = prep("methane.xyz", "cc-pvdz-ri");
    let mut erfc_caught = 0;
    let mut coul_viol = 0usize;
    for (name, op) in operators() {
        let (q, qa) = bound_tables(op, &obs, &dfbs);
        let good = triple_report(op, &obs, &dfbs, &q, &qa);

        let qh = &q * 0.5;
        let qah: Vec<f64> = qa.iter().map(|v| v * 0.5).collect();
        let half = triple_report(op, &obs, &dfbs, &qh, &qah);
        eprintln!("mutant halved {name}: viol {} of {}", half.viol, half.sig);
        assert!(half.viol > 0, "{name}: halved bound not caught");

        let (qc, qac) = bound_tables(Operator::coulomb(), &obs, &dfbs);
        let coul = triple_report(op, &obs, &dfbs, &qc, &qac);
        eprintln!(
            "mutant coulomb-diag {name}: viol {} min ratio {:.4} median {:.1} (terfc diag median {:.1})",
            coul.viol, coul.min_ratio, coul.median_ratio, good.median_ratio
        );
        assert!(
            coul.median_ratio > good.median_ratio,
            "Coulomb diag should be looser"
        );
        coul_viol += coul.viol;

        let eop = Operator::erfc(op.omega);
        let (qe, qae) = bound_tables(eop, &obs, &dfbs);
        let er = triple_report(op, &obs, &dfbs, &qe, &qae);
        eprintln!("mutant erfc-diag {name}: viol {} of {}", er.viol, er.sig);
        if er.viol > 0 {
            erfc_caught += 1;
        }
    }
    // terfc <= Coulomb pointwise in r but NOT in Fourier space (1 - cos(k r0)
    // e^{..} exceeds 1 wherever cos < 0), so the Coulomb diagonal is not a
    // rigorous bound for terfc either; it is both looser and, for some
    // triples, wrong.
    assert!(coul_viol > 0, "Coulomb-diagonal mutant was never caught");
    assert!(
        erfc_caught > 0,
        "erfc-diagonal mutant was never caught on any operator"
    );
}

/// On a diagonal block `(ab|ab)` the bound is the integral itself, so the
/// stored `Q` must equal `sqrt(max diagonal) * inflation` to rounding: this
/// pins that the inflation is applied exactly once and nothing else rescales Q.
#[test]
fn bound_is_attained_to_rounding_on_diagonal_blocks() {
    if !ensure_tables() {
        eprintln!("SKIP: terf tables absent");
        return;
    }
    let obs = prep("water.xyz", "cc-pvdz");
    let op = terfc_ang(0.991, 1.3663);
    let q = schwarz(op, &obs).unwrap();
    let mut eng = Engine::new_2e(op, &obs, 0.0).unwrap();
    let dims = obs.shell_dims();
    for a in 0..obs.nshells() {
        for b in 0..=a {
            let blk = eng.compute_quartet(&obs, a, b, a, b).unwrap().to_vec();
            let (n1, n2) = (dims[a], dims[b]);
            let mut m = 0.0f64;
            for x in 0..n1 {
                for y in 0..n2 {
                    m = m.max(blk[((x * n2 + y) * n1 + x) * n2 + y].abs());
                }
            }
            let ratio = q[(a, b)] / m.sqrt();
            assert!(
                (ratio - bound_inflation(op)).abs() < 1e-12,
                "pair ({a},{b}) ratio {ratio}"
            );
        }
    }
}

/// The validity BUDGET: r0 in {0.5, 1.0, 1.5} A x r0*omega in {0.35, 0.7071,
/// 1.354, 2, 4, 8.9} (8.9 is the edge of the shipped tables, s <= 80).
fn budget_operators() -> Vec<(String, Operator)> {
    let mut v = Vec::new();
    for r0 in [0.5, 1.0, 1.5] {
        for s in [0.35, std::f64::consts::FRAC_1_SQRT_2, 1.354, 2.0, 4.0, 8.9] {
            v.push((format!("r0={r0} s={s:.3}"), terfc_ang(r0, s / r0)));
        }
    }
    v
}

fn check_budget(label: &str, obs: &PreparedBasis, dfbs: &PreparedBasis) {
    let mut worst = f64::INFINITY;
    for (name, op) in budget_operators() {
        let (q, qa) = bound_tables(op, obs, dfbs);
        let r = triple_report(op, obs, dfbs, &q, &qa);
        eprintln!(
            "{label} {name}: significant {} viol {} min bound/true {:.6}",
            r.sig, r.viol, r.min_ratio
        );
        assert_eq!(
            r.viol, 0,
            "{label} {name}: Schwarz UNDERSHOT a true integral"
        );
        worst = worst.min(r.min_ratio);
    }
    eprintln!("{label}: min bound/true over the whole grid {worst:.6}");
}

/// Budget on methane/cc-pVDZ (d shells), 18 operators.
#[test]
fn terfc_schwarz_budget_across_r0_and_seam_sharpness() {
    if !ensure_tables() {
        return;
    }
    let obs = prep("methane.xyz", "cc-pvdz");
    let dfbs = prep("methane.xyz", "cc-pvdz-ri");
    check_budget("methane/cc-pVDZ", &obs, &dfbs);
}

/// Budget on diffuse and high-L sets. Times (release, loaded box) are in the
/// eprintln of the first run recorded next to the measurement notes.
#[test]
#[ignore = "water aug-cc-pVDZ, aug-cc-pVTZ (f), def2-TZVP+jkfit (high-L aux): minutes"]
fn terfc_schwarz_budget_diffuse_and_high_l() {
    if !ensure_tables() {
        return;
    }
    for (orb, aux) in [
        ("aug-cc-pvdz", "cc-pvdz-ri"),
        ("aug-cc-pvtz", "cc-pvdz-ri"),
        ("def2-tzvp", "def2-universal-jkfit"),
    ] {
        let obs = prep("water.xyz", orb);
        let dfbs = prep("water.xyz", aux);
        check_budget(&format!("water/{orb}+{aux}"), &obs, &dfbs);
    }
}
