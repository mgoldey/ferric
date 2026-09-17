//! How much does a terfc Schwarz bound actually screen, and does it hold?
//!
//! The 4-center terfc engine now exists and `Engine::compute_quartet` dispatches
//! to it, so a Schwarz `Q` table for terfc is buildable. This file MEASURES what
//! that buys before anything is enabled in production.
//!
//! The question that decides the whole lane: terfc is a tempered SHORT-range
//! kernel, so `(ij|ij)` should fall off with the bra-ket separation and the
//! bound should prune far pairs. But `attenuated.rs` records the opposite
//! outcome for erfc -- there, `(P|P)_erfc` is a SELF-overlap with both electrons
//! pinned to one center, so r12 -> 0, the attenuation never engages, and Schwarz
//! drops ZERO additional triples. The 4-center `(ij|ij)` is a different animal:
//! i and j sit on DIFFERENT centers, so the pair itself has spatial extent and
//! the kernel can see it. Whether that is enough is an empirical question, which
//! is what this file answers rather than argues.
//!
//! Reported, never asserted as a target: the fraction of shell pairs whose Q
//! falls below a threshold, for terfc at several r0 against Coulomb on the same
//! basis. A screening claim needs the ratio to beat Coulomb's; if it does not,
//! the honest result is that terfc screening is a null and the lane closes.
//!
//! # MEASURED 2026-09-14: it is a NULL. The lane closes.
//!
//! decane (alkane_10) / cc-pVDZ, 126 shells, 64.0M quartet pairs:
//!
//! ```text
//!   thresh   coulomb   terfc(r0=1)  terfc(r0=2)  terfc(r0=4)
//!   1e-8      48.5%      51.1%        49.7%        49.0%
//!   1e-10     38.5%      40.9%        39.6%        39.0%
//!   1e-12     30.3%      32.1%        31.1%        30.7%
//! ```
//!
//! Best case is +2.6 percentage points over Coulomb, and it DECAYS toward
//! Coulomb as r0 grows -- the correct limiting behaviour, which is what marks
//! this as a real measurement rather than an artifact. water screens 0% for
//! every operator: a compact molecule has no far pairs, so the comparison is
//! vacuous there (kept in the sweep as the below-onset control).
//!
//! The hypothesis in the paragraph above -- that 4-center `(ij|ij)` differs
//! from erfc's 3-center `(P|P)` because i and j sit on different centers -- is
//! REFUTED. The pair's spatial extent is irrelevant: `(ij|ij)` puts the SAME
//! pair on both sides of the operator, so r12 -> 0 within the diagonal quartet
//! and terfc's attenuation never engages there, exactly as `attenuated.rs`
//! records for erfc's `(P|P)`. Schwarz is structurally blind to attenuation for
//! ANY operator, because its diagonal is always a self-interaction.
//!
//! That is why `qqr3.rs` exists: only an explicit bra-ket DISTANCE envelope
//! (`exp(-w^2 R^2)`) can see the attenuation. Do not re-propose Schwarz or CSB
//! as a way to exploit short-rangedness -- the defect is in what the diagonal
//! measures, not in how tight the bound is, so a tighter pair bound cannot fix
//! it.

use ferric_core::{basis, mol::Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;

fn tables_available() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR").is_ok()
}

fn load(name: &str, basis_name: &str) -> PreparedBasis {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{name}.xyz",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap()
}

/// Build the Schwarz Q table for `op` via the 4-center path, exactly as
/// `schwarz.rs` would: `Q_ij = sqrt(max |(ij|ij)|)` over the pair block.
///
/// Returns `None` entries where the engine refused (too-high total L).
fn q_table(op: Operator, prep: &PreparedBasis) -> Vec<Vec<Option<f64>>> {
    let mut eng = Engine::new_2e(op, prep, 0.0).expect("engine");
    let dims = prep.shell_dims();
    let nsh = dims.len();
    let mut q = vec![vec![None; nsh]; nsh];
    for i in 0..nsh {
        for j in i..nsh {
            let (n1, n2) = (dims[i], dims[j]);
            let val = eng.compute_quartet(prep, i, j, i, j).map(|block| {
                let mut m = 0.0f64;
                for a in 0..n1 {
                    for b in 0..n2 {
                        m = m.max(block[((a * n2 + b) * n1 + a) * n2 + b].abs());
                    }
                }
                m.sqrt()
            });
            q[i][j] = val;
            q[j][i] = val;
        }
    }
    q
}

/// Fraction of shell-pair QUARTETS (ij|kl) that a Q*Q product screens away at
/// `thresh` -- the quantity that actually governs work in a direct build.
fn screened_fraction(q: &[Vec<Option<f64>>], thresh: f64) -> (f64, usize) {
    let nsh = q.len();
    let mut total = 0usize;
    let mut dropped = 0usize;
    for i in 0..nsh {
        for j in i..nsh {
            let Some(qij) = q[i][j] else { continue };
            for k in 0..nsh {
                for l in k..nsh {
                    let Some(qkl) = q[k][l] else { continue };
                    total += 1;
                    if qij * qkl < thresh {
                        dropped += 1;
                    }
                }
            }
        }
    }
    if total == 0 {
        return (0.0, 0);
    }
    (dropped as f64 / total as f64, total)
}

/// The sweep that produced the table in the module header. It is O(nsh^4) over
/// four operators and takes ~3 minutes on decane, so it is `#[ignore]`d: the
/// numbers are recorded above and do not need re-deriving on every run. Run it
/// with `--ignored` to reproduce or to re-measure on a different system.
#[test]
#[ignore = "3-minute O(nsh^4) sweep; results recorded in the module header"]
fn terfc_schwarz_screening_vs_coulomb() {
    if !tables_available() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset");
        return;
    }
    // A chain long enough that short-rangedness has something to bite on: a
    // compact molecule has no far pairs, so ANY operator screens ~nothing and
    // the comparison would be vacuous (see the repo's "do not declare a
    // negative below the onset" rule).
    for mol_name in ["water", "alkane_10"] {
        let prep = load(mol_name, "cc-pvdz");
        let nsh = prep.shell_dims().len();

        let q_coul = q_table(Operator::coulomb(), &prep);
        eprintln!("\n=== {mol_name} / cc-pVDZ, {nsh} shells ===");

        for thresh in [1e-8_f64, 1e-10, 1e-12] {
            let (f_coul, total) = screened_fraction(&q_coul, thresh);
            let mut line = format!("thresh {thresh:.0e}  coulomb {:.1}%", 100.0 * f_coul);
            for r0 in [1.0_f64, 2.0, 4.0] {
                let q_t = q_table(Operator::terfc(r0), &prep);
                let (f_t, _) = screened_fraction(&q_t, thresh);
                line.push_str(&format!("  terfc(r0={r0}) {:.1}%", 100.0 * f_t));
            }
            eprintln!("{line}   [{total} quartet pairs]");
        }
    }
}

/// The bound must HOLD, on a real molecule, at the r0 values measured above.
/// This is the assertion; the sweep above is measurement.
#[test]
fn terfc_schwarz_bound_holds_on_a_chain() {
    if !tables_available() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset");
        return;
    }
    let prep = load("alkane_10", "cc-pvdz");
    let dims = prep.shell_dims();
    let nsh = dims.len();
    let r0 = 2.0_f64;
    let q = q_table(Operator::terfc(r0), &prep);
    let mut eng = Engine::new_2e(Operator::terfc(r0), &prep, 0.0).unwrap();

    let step = (nsh / 6).max(1);
    let mut worst = 0.0f64;
    let mut worst_at = [0usize; 4];
    let mut checked = 0usize;

    for a in (0..nsh).step_by(step) {
        for b in (0..nsh).step_by(step) {
            for c in (0..nsh).step_by(step) {
                for d in (0..nsh).step_by(step) {
                    let (Some(qab), Some(qcd)) = (q[a][b], q[c][d]) else {
                        continue;
                    };
                    let bound = qab * qcd;
                    if bound <= 0.0 {
                        continue;
                    }
                    let Some(block) = eng.compute_quartet(&prep, a, b, c, d) else {
                        continue;
                    };
                    let maxv = block.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    if maxv < 1e-14 {
                        continue;
                    }
                    let ratio = maxv / bound;
                    if ratio > worst {
                        worst = ratio;
                        worst_at = [a, b, c, d];
                    }
                    checked += 1;
                }
            }
        }
    }

    assert!(
        checked > 50,
        "only {checked} quartets -- too few to mean anything"
    );
    eprintln!("alkane_10 terfc(r0=2): {checked} quartets, worst ratio {worst:.9} at {worst_at:?}");
    assert!(
        worst <= 1.0 + 1e-9,
        "Schwarz bound VIOLATED for terfc on alkane_10: ratio {worst:.9} at {worst_at:?}"
    );
}
