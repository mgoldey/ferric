//! WHY the QQR benefit measured over the FULL quartet population does not
//! survive contact with LinK.
//!
//! PR #35 measured QQR screening 5.18% more quartets than Schwarz at
//! alkane_16/1e-8 — over the full `nsh^4`-style quartet population. Wiring QQR
//! into LinK's inner test delivers 0.009% instead, a ~500x shortfall. This file
//! measures the two populations side by side so the shortfall is explained
//! rather than merely reported.
//!
//! The cause is that LinK and QQR target the SAME quartets, so the second
//! screen finds almost nothing the first has not already removed:
//!
//! * LinK's pair lists keep a shell pair only if it is individually
//!   significant, then restrict the ket to `sp.partners(ish)` ∩
//!   `dp.partners(jsh)`. A distant bra/ket pairing survives that intersection
//!   only when BOTH pairs are locally significant AND the density couples them
//!   — i.e. the distance-based prune has largely already happened, by a
//!   different mechanism.
//! * QQR's envelope prunes bra/ket pairings by separation, which is the same
//!   axis LinK's pair-list intersection is already cutting on.
//!
//! Neither screen is wrong; they are redundant. Over the full population, where
//! LinK's structural prune is absent, QQR's few percent is real — and that is
//! exactly the population PR #35 measured. It is not the population LinK walks.
//!
//! Run:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 FERRIC_MEM_BUDGET_GB=2 \
//!   scripts/ferric-limited --max=4G --high=3600M -- \
//!   cargo test -p ferric-scf --test qqr_population_gap --release \
//!   -- --ignored --nocapture
//! ```

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::pairs::SignificantPairs;
use ferric_scf::qqr::QqrBounds;
use ferric_scf::screening::{Bound, SchwarzBounds};

/// Count, over a quartet population, how many survive Schwarz and how many
/// survive QQR at `thresh`.
fn counts(
    sb: &SchwarzBounds,
    qb: &QqrBounds,
    quartets: impl Iterator<Item = (usize, usize, usize, usize)>,
    thresh: f64,
) -> (usize, usize, usize) {
    let mut total = 0usize;
    let mut kept_s = 0usize;
    let mut kept_q = 0usize;
    for (a, b, c, d) in quartets {
        total += 1;
        if Bound::estimate(sb, a, b, c, d) >= thresh {
            kept_s += 1;
        }
        if Bound::estimate(qb, a, b, c, d) >= thresh {
            kept_q += 1;
        }
    }
    (total, kept_s, kept_q)
}

fn report(stem: &str, basis_name: &str) {
    let mol = Molecule::load_xyz(&format!("../../testdata/molecules/{stem}.xyz")).expect("mol");
    let bs = basis::bundled(basis_name).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let sb = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let qb = QqrBounds::new(sb.clone(), &mol, &bs, &prep, op);
    let nsh = prep.nshells();

    println!("\n=== {stem} / {basis_name}: {nsh} shells ===");

    for thresh in [1e-8, 1e-10, 1e-12] {
        // Population A: ALL unique shell quartets — what PR #35 measured over.
        let all = (0..nsh).flat_map(move |i| {
            (0..=i).flat_map(move |j| {
                (0..=i).flat_map(move |k| (0..=k).map(move |l| (i, j, k, l)))
            })
        });
        let (tot_a, s_a, q_a) = counts(&sb, &qb, all, thresh);
        let extra_a = 100.0 * (s_a as f64 - q_a as f64) / (s_a as f64).max(1.0);

        // Population B: only quartets LinK's pair lists actually admit —
        // both (i,j) and (k,l) drawn from the significant pair list.
        let sp = SignificantPairs::build(&sb, nsh, thresh);
        let mut link_pop: Vec<(usize, usize, usize, usize)> = Vec::new();
        for i in 0..nsh {
            for &j in sp.partners(i) {
                if j > i {
                    continue;
                }
                for k in 0..=i {
                    for &l in sp.partners(k) {
                        if l > k {
                            continue;
                        }
                        if (k, l) > (i, j) {
                            continue;
                        }
                        link_pop.push((i, j, k, l));
                    }
                }
            }
        }
        let (tot_b, s_b, q_b) = counts(&sb, &qb, link_pop.into_iter(), thresh);
        let extra_b = 100.0 * (s_b as f64 - q_b as f64) / (s_b as f64).max(1.0);

        println!("--- thresh {thresh:.0e} ---");
        println!(
            "  full population : {tot_a:>12} quartets, Schwarz keeps {s_a:>10}, QQR keeps {q_a:>10}  -> QQR screens {extra_a:.3}% more"
        );
        println!(
            "  LinK population : {tot_b:>12} quartets, Schwarz keeps {s_b:>10}, QQR keeps {q_b:>10}  -> QQR screens {extra_b:.3}% more"
        );
    }
}

#[test]
#[ignore = "measurement; run explicitly (see module docs)"]
fn population_gap_alkane_16() {
    report("alkane_16", "cc-pvdz");
}

/// SCOPE CONTROL for the alkane result: 3-D topology, and a diffuse basis.
///
/// The `full population` vs `LinK population` gap is the whole mechanism, and
/// it is cheap to measure (no SCF, no K build — just bound evaluations), so
/// there is no excuse for having checked it on one molecular shape. A 1-D
/// gapped chain maximizes how much of the far field LinK's density screen has
/// already taken; benzene is compact and 3-D, and aug-cc-pVDZ widens both pair
/// lists. If the LinK-population column stays near zero here too, the dilution
/// is a property of the composition rather than of alkanes.
#[test]
#[ignore = "measurement; run explicitly (see module docs)"]
fn population_gap_benzene_3d() {
    // Bare benzene FIRST, as a deliberate negative control on the control:
    // it is compact (~9.4 Bohr) and turns out to have almost no far field at
    // all — QQR screens only 0.030% of even the FULL population at cc-pVDZ.
    // That makes it useless for the question being asked here, and recording
    // it stops a future reader from mistaking "QQR does nothing on benzene"
    // for evidence about the LinK composition. A system must HAVE a far field
    // before "does LinK already screen the far field" is a meaningful test.
    report("benzene", "cc-pvdz");
    report("benzene", "aug-cc-pvdz");
}

/// The real 3-D scope control: an EXTENDED, non-linear system, in a plain and
/// a diffuse basis.
///
/// Bare benzene (above) fails as a control because it is too compact to have a
/// far field. The T-shaped benzene dimer keeps the 3-D, non-chain topology but
/// adds genuine separation between the two monomers — so `dp.partners` has real
/// long-range entries for LinK's screen and QQR's envelope to compete over.
/// aug-cc-pVDZ then widens both pair lists via diffuse functions.
///
/// Artifact hypothesis, stated before running: if the alkane dilution reflects
/// the LinK composition, the LinK-population column stays far below the
/// full-population column here too. If instead it approaches the full-population
/// figure, the alkane result was topology-specific and must not be generalized.
#[test]
#[ignore = "measurement; run explicitly (see module docs)"]
fn population_gap_benzene_dimer_extended() {
    report("s22/benzene_dimer_T", "cc-pvdz");
    report("s22/benzene_dimer_T", "aug-cc-pvdz");
}
