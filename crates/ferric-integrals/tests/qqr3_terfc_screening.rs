//! Does routing terfc through QQR-3 screen the 3-index build?
//!
//! `attenuated_ri_mp2` (erfc) already goes through `eri3_tensor_screened_qqr`
//! and drops ~48% of shell triples on decane. The terfc RI-MP2 path
//! (`ri_mp2` / `scs.rs`, the published MP2-V attenuator) builds its 3-index
//! tensor DENSE. This measures what QQR-3 would buy it.
//!
//! # What the QQR-3 bound actually is
//!
//! Read `estimate3` before predicting anything here. The live bound is
//!
//! ```text
//!   Q3[P] * Q(mu,nu) * min(1, 1.10 * ext_sum / r_eff)
//! ```
//!
//! The envelope is a COULOMB monopole term. There is NO `exp(-w^2 R^2)` factor
//! in the implementation -- `estimate3` never reads `op.omega`, and the
//! `debug_omega_tilde` helper is `#[doc(hidden)]` and unused in the live path.
//! The code says why: for an attenuated operator the decay is already carried
//! by the per-operator Schwarz seeds, and multiplying in a second attenuation
//! factor over-suppresses real triples and BREAKS validity (measured worst
//! ratio 1.7-5.3).
//!
//! So ALL operator dependence enters through the seeds `Q3[P] = sqrt|(P|P)|`
//! and `Q(mu,nu) = sqrt|(mu nu|mu nu)|` -- both SELF-interactions. That is
//! precisely the quantity `terfc_screening_measurement.rs` measured as blind to
//! attenuation (r12 -> 0 on the diagonal, so `terfc(0)/0 -> 1/0` = Coulomb).
//!
//! # Artifact hypothesis vs physics hypothesis
//!
//! If QQR-3 genuinely exploits terfc's short-rangedness, terfc should drop
//! MORE triples than Coulomb at the same threshold, and the margin should GROW
//! as r0 shrinks. If the seeds are blind (the prediction above), terfc and
//! Coulomb should track each other within a couple of points with no
//! systematic r0 trend. These differ, so the experiment can distinguish them.
//!
//! # MEASURED 2026-09-14 — the blind-seed prediction holds
//!
//! decane / cc-pVDZ + cc-pVDZ-RI, 126 obs shells, 312 aux shells,
//! 2 496 312 shell triples. Fraction KEPT (lower = more screening):
//!
//! ```text
//!   thresh   coulomb  terfc(r0=1)  terfc(r0=2)  terfc(r0=4)
//!   1e-8      79.1%     77.9%        78.5%        78.8%
//!   1e-10     85.3%     84.4%        84.9%        85.2%
//!   1e-12     89.4%     88.8%        89.2%        89.3%
//!
//!   erfc(w=0.222) 78.5%   erfc(w=0.5) 77.8%      (both at thresh 1e-8)
//! ```
//!
//! terfc beats Coulomb by 1.2 points at r0=1, decaying to 0.3 at r0=4. erfc --
//! the operator `attenuated.rs` screens in production -- beats Coulomb by 0.6.
//! water keeps 100% for every operator: below the onset, no far pairs.
//!
//! So routing terfc through QQR-3 is now POSSIBLE (it could not construct at
//! all before terfc was wired into `Engine::new_2e`) but buys ~1 point over
//! plain Coulomb screening. Screening on this path is GEOMETRIC, not
//! operator-specific.
//!
//! # This also refutes a claim in `attenuated.rs`
//!
//! That module's header says QQR-3 drops "about 48%" of decane's shell triples
//! for erfc, and that "operator-specific decay is what makes attenuated MP2
//! intrinsically more screenable". Both describe the PRE-`f28f3044` bound,
//! which carried an `erfc(w~R)` factor and was INVALID -- it under-estimated
//! real integrals and silently dropped them (worst |true|/bound 1.7-5.3). The
//! 48% was bought with that invalidity. Measured here: ~21% dropped, erfc
//! ahead of Coulomb by 0.6 points. The header has been annotated.

use ferric_core::{basis, mol::Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::qqr3::QqrBounds3;

fn tables_available() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR").is_ok()
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
}

fn load(name: &str) -> Sys {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{name}.xyz",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    Sys { mol, obs, dfbs }
}

/// Fraction of shell triples (P|s1 s2) that the QQR-3 bound drops at `thresh`.
fn kept_fraction(op: Operator, s: &Sys, thresh: f64) -> Result<(usize, usize), String> {
    let bounds = QqrBounds3::new(op, &s.mol, &s.obs, &s.dfbs).map_err(|e| format!("{e}"))?;
    let nsh_obs = s.obs.nshells();
    let nsh_aux = s.dfbs.nshells();
    let mut kept = 0usize;
    let mut total = 0usize;
    for p in 0..nsh_aux {
        for s1 in 0..nsh_obs {
            for s2 in 0..=s1 {
                total += 1;
                if bounds.estimate3(p, s1, s2) >= thresh {
                    kept += 1;
                }
            }
        }
    }
    Ok((kept, total))
}

#[test]
fn qqr3_accepts_terfc_now_that_new_2e_supports_it() {
    if !tables_available() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset");
        return;
    }
    let s = load("water");
    // The point of the assertion: QqrBounds3 seeds via Engine::new_2center and
    // Engine::new_2e. Before terfc was wired into new_2e this construction
    // failed outright, so terfc could not be screened at all.
    let b = QqrBounds3::new(Operator::terfc(2.0), &s.mol, &s.obs, &s.dfbs);
    assert!(
        b.is_ok(),
        "QQR-3 must construct for terfc now that new_2e dispatches it: {:?}",
        b.err()
    );
    // And it must produce a finite, positive bound somewhere -- a table of
    // zeros would "screen" everything and pass a fraction test vacuously.
    let bounds = b.unwrap();
    let mut any_positive = false;
    for p in 0..s.dfbs.nshells() {
        for s1 in 0..s.obs.nshells() {
            let e = bounds.estimate3(p, s1, s1);
            assert!(e.is_finite(), "non-finite terfc QQR-3 bound at ({p},{s1},{s1})");
            if e > 0.0 {
                any_positive = true;
            }
        }
    }
    assert!(any_positive, "every terfc QQR-3 bound was zero -- the table is inert");
}

/// The measurement. `#[ignore]`d because it builds several QQR-3 bound tables
/// on a 126-shell chain; results are recorded in the header comment below.
#[test]
#[ignore = "multi-minute sweep; run with --ignored to reproduce"]
fn qqr3_terfc_vs_coulomb_screening_sweep() {
    if !tables_available() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset");
        return;
    }
    // decane: long enough that distance screening has far pairs to bite on.
    // A compact molecule is below the onset and would show ~nothing for any
    // operator, which is a vacuous comparison rather than a negative result.
    for name in ["water", "alkane_10"] {
        let s = load(name);
        eprintln!(
            "\n=== {name} / cc-pVDZ + cc-pVDZ-RI, {} obs shells, {} aux shells ===",
            s.obs.nshells(),
            s.dfbs.nshells()
        );
        for thresh in [1e-8_f64, 1e-10, 1e-12] {
            let mut line = format!("thresh {thresh:.0e}");
            match kept_fraction(Operator::coulomb(), &s, thresh) {
                Ok((k, t)) => line.push_str(&format!(
                    "  coulomb {:.1}% kept ({k}/{t})",
                    100.0 * k as f64 / t as f64
                )),
                Err(e) => line.push_str(&format!("  coulomb ERR {e}")),
            }
            for r0 in [1.0_f64, 2.0, 4.0] {
                match kept_fraction(Operator::terfc(r0), &s, thresh) {
                    Ok((k, t)) => line.push_str(&format!(
                        "  terfc(r0={r0}) {:.1}%",
                        100.0 * k as f64 / t as f64
                    )),
                    Err(e) => line.push_str(&format!("  terfc(r0={r0}) ERR {e}")),
                }
            }
            eprintln!("{line}");
        }
        // erfc for reference: this is the operator attenuated.rs screens today.
        for thresh in [1e-8_f64] {
            for omega in [0.222234_f64, 0.5] {
                if let Ok((k, t)) = kept_fraction(Operator::erfc(omega), &s, thresh) {
                    eprintln!(
                        "thresh {thresh:.0e}  erfc(w={omega}) {:.1}% kept ({k}/{t})",
                        100.0 * k as f64 / t as f64
                    );
                }
            }
        }
    }
}
