//! GATE: which two-electron operators admit Schwarz/CSB screening, and why.
//!
//! Screening by `Q_ij Q_kl` IS the Cauchy-Schwarz inequality, which requires
//! the two-electron form `(f|g)` to be an inner product — i.e. the kernel
//! positive-definite (Bochner: Fourier transform > 0 for k > 0). This file
//! pins that property per operator with MEASUREMENTS rather than comments,
//! because a comment saying "terf is indefinite" decays and an assertion does
//! not.
//!
//! | operator | F(k)                                      | PD? | screening |
//! |----------|-------------------------------------------|-----|-----------|
//! | erf      | e^(-k²/4ω²) / (2π²k²)                     | yes | Schwarz/CSB/CSAM |
//! | erfc     | (1 - e^(-k²/4ω²)) / (2π²k²)               | yes | CSB (rigorous) |
//! | terfc    | (1 - cos(k·r₀)e^(-k²/4ω²)) / (2π²k²)      | yes | CSB-eligible; blocked, no 4-center engine |
//! | terf     | cos(k·r₀)e^(-k²/4ω²) / (2π²k²)            | NO  | none valid, permanently |
//!
//! terf's `cos(k·r₀)` is negative at `k·r₀ = π`. The cause is the FINITE SHELL
//! RADIUS r₀, not long-rangedness — plain erf is long-range and positive
//! definite, which is exactly why it is the control below.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;

fn have_tables() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR").is_ok()
}

fn water() -> PreparedBasis {
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/water.xyz"
    ))
    .unwrap();
    PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap()
}

/// Minimum 2-center diagonal element `(P|P)` over all shells.
fn min_diagonal(op: Operator, dfbs: &PreparedBasis) -> f64 {
    let mut eng = Engine::new_2center(op, dfbs, 0.0).expect("2-center engine");
    let mut lo = f64::INFINITY;
    for p in 0..dfbs.nshells() {
        for &x in eng.compute_eri2(dfbs, p, p).iter() {
            lo = lo.min(x);
        }
    }
    lo
}

/// The positive-definite operators must never produce a negative `(P|P)`.
///
/// This is the half that would fail if ferric's engines were broken, so it is
/// the control that makes the terf assertion below meaningful rather than a
/// statement about a buggy integral path.
#[test]
fn positive_definite_operators_have_nonnegative_diagonals() {
    let dfbs = water();
    // 1/(2*sqrt2), 1/sqrt2 and sqrt2 spelled from std consts rather than as
    // decimal literals: clippy::approx_constant flags 0.7071/1.4142, and the
    // exact values are what the curvature constraint r0*omega = 1/sqrt2
    // actually means here.
    for &omega in &[
        0.1_f64,
        std::f64::consts::FRAC_1_SQRT_2 / 2.0,
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::SQRT_2,
    ] {
        let erf = min_diagonal(Operator::erf(omega), &dfbs);
        assert!(
            erf >= 0.0,
            "erf(w={omega}) produced a NEGATIVE 2-center diagonal ({erf:.3e}). erf's transform \
             e^(-k^2/4w^2)/(2 pi^2 k^2) is strictly positive, so this is not a property of the \
             operator — suspect the engine."
        );
        let erfc = min_diagonal(Operator::erfc(omega), &dfbs);
        assert!(
            erfc >= 0.0,
            "erfc(w={omega}) produced a NEGATIVE 2-center diagonal ({erfc:.3e}); its transform \
             (1 - e^(-k^2/4w^2))/(2 pi^2 k^2) is strictly positive."
        );
    }
}

/// terf IS indefinite and terfc is NOT — asserted on the same molecule, basis
/// and table machinery, so the contrast cannot be an engine artifact.
///
/// Skips when the interpolation tables are absent (they are untracked data);
/// the `positive_definite_...` control above still runs, so a broken-engine
/// regression is caught even without tables.
#[test]
fn terf_is_indefinite_and_terfc_is_not() {
    if !have_tables() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset (terf tables are untracked data)");
        return;
    }
    let dfbs = water();

    // r0 = 0.5 is deliberately included as the NEAR-COULOMB end: the sign
    // change sits at k*r0 = pi, so a small r0 pushes it to large k where
    // Gaussian shell pairs carry little weight and no negative diagonal
    // appears. Asserting a negative only at the larger r0 keeps this honest
    // about WHERE the effect shows up rather than claiming it is universal.
    let mut saw_negative = false;
    for &r0 in &[0.5_f64, 1.0, 2.0, 4.0] {
        let terf = min_diagonal(Operator::terf(r0), &dfbs);
        let terfc = min_diagonal(Operator::terfc(r0), &dfbs);

        assert!(
            terfc >= 0.0,
            "terfc(r0={r0}) produced a NEGATIVE diagonal ({terfc:.3e}). terfc is PROVEN \
             positive-definite (F(k) = (1 - cos(k*r0)e^(-k^2/4w^2))/(2 pi^2 k^2) >= \
             (1 - e^(-k^2/4w^2))/(2 pi^2 k^2) > 0), so this contradicts the proof and means \
             either the proof or the interpolated engine is wrong. Do NOT relax this."
        );
        if terf < 0.0 {
            saw_negative = true;
        }
    }
    assert!(
        saw_negative,
        "terf never produced a negative 2-center diagonal across r0 in [0.5, 4]. Measured \
         values at the time this gate was written: r0=2 -> -7.543e-3, r0=4 -> -4.439e-4. If \
         this now passes cleanly, either the terf engine changed or the operator is no longer \
         what it was — because a positive-definite terf would contradict its Fourier transform \
         cos(k*r0)e^(-k^2/4w^2)/(2 pi^2 k^2), which is negative at k*r0 = pi. Investigate \
         rather than deleting this assertion: it is the entire justification for refusing \
         Schwarz and CSB on terf."
    );
}

/// `terf + terfc == Coulomb` to machine precision (shim.cc:990).
///
/// Guards the gate above: if the terf engine silently fell back to terfc or
/// Coulomb, the indefiniteness assertion would be measuring the wrong operator.
#[test]
fn terf_plus_terfc_reconstructs_coulomb() {
    if !have_tables() {
        eprintln!("SKIP: FERRIC_TERF_TABLE_DIR unset");
        return;
    }
    let dfbs = water();
    let nsh = dfbs.nshells();
    for &r0 in &[0.5_f64, 2.0, 4.0] {
        let mut e_terf = Engine::new_2center(Operator::terf(r0), &dfbs, 0.0).unwrap();
        let mut e_terfc = Engine::new_2center(Operator::terfc(r0), &dfbs, 0.0).unwrap();
        let mut e_coul = Engine::new_2center(Operator::coulomb(), &dfbs, 0.0).unwrap();
        let mut worst = 0.0_f64;
        for p in 0..nsh {
            for q in 0..=p {
                let t: Vec<f64> = e_terf.compute_eri2(&dfbs, p, q).to_vec();
                let tc: Vec<f64> = e_terfc.compute_eri2(&dfbs, p, q).to_vec();
                let c: Vec<f64> = e_coul.compute_eri2(&dfbs, p, q).to_vec();
                for i in 0..c.len() {
                    worst = worst.max((t[i] + tc[i] - c[i]).abs());
                }
            }
        }
        assert!(
            worst < 1e-11,
            "terf + terfc - coulomb = {worst:.3e} at r0={r0}, above the 1e-11 machine-precision \
             bar. The three engines are supposed to share identical MD machinery and differ \
             only in the final combine step (shim.cc:990), so a deviation here means one of \
             them is not computing what its name says — which would invalidate \
             terf_is_indefinite_and_terfc_is_not."
        );
    }
}

/// Schwarz and CSB must REFUSE terf, and the message must say INVALID rather
/// than unimplemented — the distinction tells a future contributor not to
/// "add terf support".
#[test]
fn schwarz_and_csb_refuse_terf_as_invalid_not_unimplemented() {
    let dfbs = water();
    let terf = Operator::terf(2.0);

    let e = ferric_integrals::schwarz::schwarz(terf, &dfbs)
        .expect_err("Schwarz must refuse terf");
    let msg = format!("{e:?}").to_lowercase();
    assert!(
        msg.contains("invalid"),
        "Schwarz's terf refusal must say INVALID (not 'not implemented'): a contributor reading \
         'unimplemented' will reasonably try to implement it, and no implementation can make \
         Cauchy-Schwarz hold for an indefinite form. Got: {e:?}"
    );

    let e2 = ferric_integrals::csb::csb_m_table(terf, &dfbs)
        .expect_err("CSB must refuse terf");
    let msg2 = format!("{e2:?}").to_lowercase();
    assert!(
        msg2.contains("invalid"),
        "CSB's terf refusal must say INVALID for the same reason. Got: {e2:?}"
    );
}

/// terfc's refusal must read as an ENGINE gap, not a validity failure — the
/// opposite of terf. Someone who reads terfc as "invalid" will not build the
/// 4-center kernel that would unblock it.
#[test]
fn terfc_refusal_reads_as_an_engine_gap_not_invalidity() {
    let dfbs = water();
    let terfc = Operator::terfc(2.0);
    let e = ferric_integrals::schwarz::schwarz(terfc, &dfbs)
        .expect_err("Schwarz has no terfc path yet");
    let msg = format!("{e:?}").to_lowercase();
    assert!(
        msg.contains("valid") && !msg.contains("invalid"),
        "terfc's refusal must state the kernel IS valid and the gap is the missing 4-center \
         engine. Got: {e:?}"
    );
}
