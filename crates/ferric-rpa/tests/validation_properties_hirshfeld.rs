//! Weekly half of `properties_hirshfeld.rs`: water's full Hirshfeld-I
//! charges with ad-hoc same-basis neutral + ion proatoms in aug-cc-pVDZ,
//! against the published Hirshfeld-I window (q(O) = -0.872, Verstraelen et
//! al., JCTC 12, 3894 (2016)). It took ~130 s on CI (runs 40957025137 /
//! 40953725119, mean), above the 120 s per-commit threshold. Per-commit
//! canary: `properties_hirshfeld.rs::h2o_adhoc_neutral_hirshfeld_charges`
//! (the same `hirshfeld_i_charges` driver, neutral-only proatoms); the
//! ion-proatom iteration runs here only. Driver: `hirshfeld_i_support/mod.rs`.

mod hirshfeld_i_support;
use hirshfeld_i_support::water_hirshfeld_i_charges;

/// Full Hirshfeld-I with ad-hoc same-basis NEUTRAL+ION proatoms.
///
/// The CANONICAL published Hirshfeld-I charge for water oxygen is q(O) = −0.872
/// (Verstraelen et al., JCTC 12, 3894 (2016), Table 1, BLYP/6-311+G(2df,p);
/// MBIS gives −0.885). Hirshfeld-I is *designed* to be over-ionic — it amplifies
/// plain-Hirshfeld charges to reproduce the ESP/dipole — so q(O) near −0.87 is
/// CORRECT; the −0.6..−0.7 numbers are CM5/ESP-class, not raw HI. (My earlier
/// −0.92 with a 15-Bohr radial grid was anion-tail truncation; the grid now
/// extends to 30 Bohr to capture the diffuse anion reference.)
#[test]
#[ignore = "validation: water Hirshfeld-I charges, neutral + ion proatoms, aug-cc-pVDZ, vs the published window; CI ~130 s"]
fn h2o_hirshfeld_i_adhoc_charges_physical() {
    // Augmented basis so the O⁻ anion proatom binds (Hirshfeld-I
    // requirement); proatom grid to 30 Bohr for the diffuse anion tail.
    let q = water_hirshfeld_i_charges("aug-cc-pvdz", 600);
    let (q_o, q_h1, q_h2) = (q[0], q[1], q[2]);
    eprintln!("H2O Hirshfeld-I (ad-hoc, neutral+ion proatoms): O={q_o:.4} H={q_h1:.4} H={q_h2:.4}");
    // Literature Hirshfeld-I water oxygen is −0.872 (Verstraelen 2016). HI is
    // intentionally over-ionic; accept the published HI window. (Plain neutral
    // Hirshfeld gives −0.32 — see the neutral test; HI sharpens it toward ESP.)
    assert!(
        (-0.95..=-0.78).contains(&q_o),
        "O charge outside Hirshfeld-I range: {q_o:.3} (lit -0.872)"
    );
    assert!(
        (0.39..=0.48).contains(&q_h1),
        "H charge outside Hirshfeld-I range: {q_h1:.3} (lit +0.436)"
    );
    assert!((q_o + q_h1 + q_h2).abs() < 1e-6, "charges must sum to 0");
    assert!(
        (q_h1 - q_h2).abs() < 1e-3,
        "equivalent H must have equal charge"
    );
}
