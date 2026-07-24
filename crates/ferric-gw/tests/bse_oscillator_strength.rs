//! Oscillator-strength validation for BSE/CIS-TDA (`tda_oscillator_strengths`
//! in `bse.rs`), H₂O / cc-pVDZ.
//!
//! Two gates:
//!
//! 1. `cis_tda_oscillator_strengths_match_pyscf_df_kernel` — the DECISIVE
//!    numerical cross-check. `run_cis_tda` uses an EXACT (non-DF) RHF
//!    reference (`RhfConfig::default()` has `df_j_aux`/`df_k_aux` both
//!    `None`) with a DF (RI, cc-pvdz-ri) Coulomb+exchange kernel for the
//!    CIS-TDA matrix itself; `scripts/pyscf_cis_osc_ref.py` independently
//!    builds the SAME exact-RHF + DF-kernel combination from scratch in
//!    PySCF/numpy (no `pyscf.tdscf`, no ferric code) and computes
//!    oscillator strengths with the same length-gauge formula. This
//!    isolates the oscillator-strength FORMULA (the thing this task adds)
//!    from any GW/screening physics already validated elsewhere
//!    (h2o_bse_tda.rs). Reference values are pasted from that script's
//!    printed output (provenance below) — tight tolerance since both sides
//!    use bit-for-bit the same RHF orbitals and DF integrals up to
//!    numpy-vs-ndarray-linalg eigensolver floating-point noise.
//!
//! 2. `bse_tda_oscillator_strengths_are_sane` — smoke gate on the full
//!    G0W0@HF-screened `run_bse_tda` path: oscillator strengths must be
//!    finite, non-negative, and the same length as `omega`. The absolute BSE
//!    excitation energies on this system are GW-gap-limited (see
//!    h2o_bse_tda.rs / docs/bse-tda-water-gap-investigation.md) so this gate
//!    does not re-litigate that; it only proves oscillator strengths flow
//!    correctly through the GW-screened path, not just the bare-HF CIS path.
//!
//! PySCF cross-check provenance (2026-07-19, pyscf 2.13.0, scipy 1.15.2):
//!
//! ```text
//! $ OMP_NUM_THREADS=2 python3 scripts/pyscf_cis_osc_ref.py
//! # exact RHF E = -76.0267679974
//! # nmo=24 nocc=5 nvir=19 n=95
//! # lowest 6 CIS-TDA (DF kernel, exact RHF) excitation energies (eV):
//! #   1  9.197811
//! #   2  10.983873
//! #   3  11.842606
//! #   4  13.634028
//! #   5  15.042594
//! #   6  18.341076
//! # oscillator strengths (length gauge, sqrt(2) convention):
//! #   1  E=9.197811 eV   f=2.844846e-02
//! #   2  E=10.983873 eV   f=4.957058e-28
//! #   3  E=11.842606 eV   f=1.081263e-01
//! #   4  E=13.634028 eV   f=9.469235e-02
//! #   5  E=15.042594 eV   f=3.129394e-01
//! #   6  E=18.341076 eV   f=1.588274e-01
//! # origin-shift cross-check (must match above f values):
//! #   ... (identical to 6 s.f. — transition dipole origin-independence
//! #        confirmed for the occ-virt block)
//! ```
//!
//! ferric's `run_cis_tda` reference RHF is EXACT (non-DF): `RhfConfig::default()`
//! leaves `df_j_aux`/`df_k_aux` as `None`, so an earlier draft of the PySCF
//! script that used `scf.RHF(mol).density_fit(...)` was mismatched by ~1.4e-2
//! eV — fixed by switching the PySCF RHF step to exact (non-DF) `scf.RHF(mol)`,
//! after which the two codebases' excitation energies agree to ~1e-5 eV.
//!
//! The manual PySCF/numpy formula in that script was itself cross-checked
//! against `pyscf.tdscf.rhf.TDA.oscillator_strength()` (same exact-4-index
//! kernel convention) and reproduces its oscillator strengths EXACTLY (not
//! approximately) once the eigenvector normalization convention
//! (`sum_ia X_ia^2 = 1`, LAPACK's `dsyev`/`dsyevd` convention) and the
//! `sqrt(2)` singlet spin-adaptation prefactor are matched — see the
//! derivation note in `tda_oscillator_strengths`'s doc comment in `bse.rs`.
//!
//! Run: cargo test -p ferric-gw --release --test bse_oscillator_strength -- --ignored --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::bse::{run_bse_tda, run_cis_tda};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const HA_TO_EV: f64 = 27.211386245988_f64;

fn prepare_h2o() -> (Molecule, PreparedBasis, PreparedBasis, ferric_scf::ScfResult) {
    let xyz = "3\nH2O\nO 0.0 0.0 0.117790\nH 0.0 0.755453 -0.471161\nH 0.0 -0.755453 -0.471161\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).expect("parse H2O");
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    (mol, obs, dfbs, rhf)
}

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig { scheme: QuadratureScheme::GaussLegendre, n_points: 16, u0: 0.5 },
        eigensolver_conv_thresh: 1e-7,
        eigensolver_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        need_inv_dielectric_freq: false,
        need_eigenvalues_freq: true,
        verbose: false,
    }
}

#[test]
#[ignore = "fast: RHF + CIS-TDA + oscillator strengths (no GW); run --release --ignored"]
fn cis_tda_oscillator_strengths_match_pyscf_df_kernel() {
    let (mol, obs, dfbs, rhf) = prepare_h2o();
    let res = run_cis_tda(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, 0).unwrap();

    assert_eq!(res.omega.len(), res.oscillator_strength.len());

    // PySCF DF-kernel reference (see module doc for the exact generating
    // script + provenance). (energy_eV, f) pairs for the lowest 6 states.
    let pyscf_ref: [(f64, f64); 6] = [
        (9.197811, 2.844846e-02),
        (10.983873, 4.957058e-28), // symmetry-forbidden: near-zero, loose abs tol below
        (11.842606, 1.081263e-01),
        (13.634028, 9.469235e-02),
        (15.042594, 3.129394e-01),
        (18.341076, 1.588274e-01),
    ];

    eprintln!("\nCIS-TDA (DF kernel) / cc-pVDZ H2O -- oscillator strengths vs PySCF");
    eprintln!("  {:>4} {:>12} {:>12}  {:>12} {:>12}", "n", "E ferric", "E pyscf", "f ferric", "f pyscf");
    for (n, ((&om, &f), &(e_ref, f_ref))) in
        res.omega.iter().zip(res.oscillator_strength.iter()).zip(pyscf_ref.iter()).enumerate()
    {
        let e_ev = om * HA_TO_EV;
        eprintln!("  {:>4} {:>12.6} {:>12.6}  {:>12.6e} {:>12.6e}", n + 1, e_ev, e_ref, f, f_ref);

        // Energies: same DF kernel, same exact-RHF reference -- expect
        // near-exact agreement (ferric uses ndarray-linalg/LAPACK, PySCF
        // uses numpy.linalg.eigh -- both LAPACK dsyevd under the hood on the
        // same matrix, so residual is numerical noise only). Measured
        // residual is ~1e-5 eV; tolerance kept at 1e-3 eV (2 orders of
        // margin) rather than pinned to the noise floor, since a future
        // integral-engine change (e.g. a different libint2 minor version)
        // could shift this slightly without indicating a real bug.
        assert!(
            (e_ev - e_ref).abs() < 1e-3,
            "state {}: excitation energy {e_ev:.6} eV vs PySCF DF-kernel ref {e_ref:.6} eV (diff {:.2e})",
            n + 1,
            (e_ev - e_ref).abs()
        );

        // Oscillator strengths: symmetry-forbidden state 2 has f ~ 1e-28 on
        // both sides (numerical zero, sign/magnitude not meaningful past
        // machine precision) -- use an absolute-or-relative tolerance that
        // treats "both effectively zero" as a pass without demanding
        // matching noise floors.
        if f_ref.max(f) < 1e-10 {
            assert!(f < 1e-8, "state {}: expected ~0 oscillator strength, got {f:.3e}", n + 1);
        } else {
            let rel_err = (f - f_ref).abs() / f_ref;
            assert!(
                rel_err < 0.01,
                "state {}: oscillator strength {f:.6e} vs PySCF DF-kernel ref {f_ref:.6e} \
                 (rel err {:.4}, tol 1%)",
                n + 1,
                rel_err
            );
        }
    }
}

#[test]
#[ignore = "slow: RHF + PDEP-RPA + G0W0 + BSE-TDA + oscillator strengths; run --release --ignored"]
fn bse_tda_oscillator_strengths_are_sane() {
    let (mol, obs, dfbs, rhf) = prepare_h2o();
    let res = run_bse_tda(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &pdep_cfg(), 0)
        .expect("BSE-TDA runs");

    assert_eq!(
        res.omega.len(),
        res.oscillator_strength.len(),
        "oscillator_strength must have one entry per excitation energy"
    );
    assert!(!res.oscillator_strength.is_empty());

    eprintln!("\nBSE-TDA@G0W0@HF / cc-pVDZ H2O -- lowest 5 states + oscillator strengths");
    for (n, (&om, &f)) in res.omega.iter().zip(res.oscillator_strength.iter()).take(5).enumerate() {
        eprintln!("  Omega_{:<2} = {:8.4} eV   f = {:10.6}", n + 1, om * HA_TO_EV, f);
        assert!(f.is_finite(), "state {}: oscillator strength must be finite, got {f}", n + 1);
        assert!(f >= -1e-8, "state {}: oscillator strength must be non-negative, got {f}", n + 1);
    }

    let f_lowest = res.lowest_oscillator_strength();
    eprintln!("  lowest singlet: {:.4} eV, f = {:.6}", res.lowest_ev(), f_lowest);
    assert!(f_lowest.is_finite() && f_lowest >= -1e-8);
}
