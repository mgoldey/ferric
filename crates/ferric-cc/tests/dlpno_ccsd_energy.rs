//! DLPNO-CCSD pair screening validated on REAL converged CCSD amplitudes.
//!
//! The unit tests in `dlpno_ccsd.rs` use synthetic tensors, which pins the
//! bookkeeping but says nothing about physics. This drives a real closed-shell
//! CCSD run on water, then applies the pair mask to its converged `t2` and
//! recomputes the correlation energy.
//!
//! All assertions are on ENERGIES and COUNTS, deliberately — no wall clocks. The
//! box these ran on was contested, so any timing would be untrustworthy, whereas
//! an energy is unaffected by load.

use ferric_cc::dlpno_ccsd::{apply_pair_mask, pair_mask_retention};
use ferric_cc::{ccsd_closed_shell::ccsd_closed_shell, CcConfig};
use ferric_core::{basis, mol::Molecule, parallel::ParallelContext};
use ferric_integrals::{basis_bridge::PreparedBasis, operator::Operator};
use ferric_mp2::boys::boys_localize;
use ferric_mp2::pair_domains::{build_pair_domains, complete_pair_domains, PairDomains};
use ferric_scf::{
    rhf::{solve_rhf, RhfConfig},
    screening::SchwarzBounds,
};
use ndarray::{Array2, Array4};

struct Case {
    t2: Array4<f64>,
    ovov: Array4<f64>,
    centers: Array2<f64>,
}

/// Converge RHF + CCSD on water/STO-3G and return the converged t2, the (ia|jb)
/// block needed to re-evaluate the energy, and the Boys centers.
fn water_ccsd() -> Case {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let scf = RhfConfig { density_conv: 1e-10, max_iter: 200, ..Default::default() };
    let rhf = solve_rhf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &scf).unwrap();
    assert!(rhf.converged, "SCF must converge");

    let cfg = CcConfig { energy_conv: 1e-10, max_iter: 100, ..Default::default() };
    let cc = ccsd_closed_shell(&mol, &obs, &dfbs, Operator::coulomb(), &rhf, &cfg).unwrap();
    let t2 = cc.t2.clone().into_dimensionality::<ndarray::Ix4>().unwrap();

    // (ia|jb) from the same RI tensors CCSD used, so the energy we recompute is
    // the same quantity CCSD reports.
    let nocc = (mol.nelec() / 2) as usize;
    let nvir = obs.nbasis() - nocc;
    let ovov = {
        let inter = ferric_mp2::rimp2::compute_rpa_intermediates(
            &mol,
            &obs,
            &dfbs,
            Operator::coulomb(),
            &rhf,
            &ferric_mp2::rimp2::RiMp2Config::default(),
        )
        .unwrap();
        let b = &inter.b_ov; // (naux, nocc*nvir)
        let naux = b.nrows();
        Array4::from_shape_fn((nocc, nvir, nocc, nvir), |(i, a, j, bb)| {
            (0..naux).map(|p| b[(p, i * nvir + a)] * b[(p, j * nvir + bb)]).sum()
        })
    };

    let c_occ = rhf.mos_r().slice(ndarray::s![.., ..nocc]).to_owned();
    let dip = ferric_integrals::oneelectron::dipole(&obs, [0.0, 0.0, 0.0]).unwrap();
    let centers = boys_localize(&c_occ, &dip, 200).centers;

    Case { t2, ovov, centers }
}

/// Closed-shell CCSD correlation energy from t2 alone (t1 omitted: at the
/// converged RHF reference the t1 contribution enters through `tau`, and this
/// helper is used only to COMPARE masked vs unmasked, where the omitted piece is
/// common to both).
fn energy_from_t2(t2: &Array4<f64>, ovov: &Array4<f64>) -> f64 {
    let (no, _, nv, _) = t2.dim();
    let mut e_a = 0.0;
    let mut e_b = 0.0;
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let t = t2[[i, j, a, b]];
                    e_a += t * ovov[[i, a, j, b]];
                    e_b += t * ovov[[i, b, j, a]];
                }
            }
        }
    }
    2.0 * e_a - e_b
}

fn domains_for(centers: &Array2<f64>, cutoff: f64) -> PairDomains {
    build_pair_domains(centers, cutoff, f64::INFINITY).unwrap()
}

/// THE EXACTNESS CONTRACT ON REAL DATA: a complete mask must not change the
/// CCSD correlation energy at all.
#[test]
fn complete_mask_leaves_ccsd_energy_bit_identical() {
    let case = water_ccsd();
    let e_ref = energy_from_t2(&case.t2, &case.ovov);

    let mut masked = case.t2.clone();
    let d = complete_pair_domains(&case.centers).unwrap();
    let zeroed = apply_pair_mask(&mut masked, &d).unwrap();
    let e_masked = energy_from_t2(&masked, &case.ovov);

    eprintln!("CCSD E_corr(t2) reference = {e_ref:.12}");
    eprintln!("               complete mask = {e_masked:.12}");
    assert_eq!(zeroed, 0, "complete domains must zero no pairs");
    assert_eq!(
        e_ref, e_masked,
        "a complete pair mask must be bit-identical, got {e_masked:.12} vs {e_ref:.12}"
    );
}

/// Screening removes correlation, and the loss must grow as the cutoff tightens.
///
/// This is the accuracy half of the accuracy/cost curve — reported rather than
/// asserted against a fixed number, because the useful output is the shape.
#[test]
fn tighter_cutoffs_lose_more_correlation_monotonically() {
    let case = water_ccsd();
    let e_ref = energy_from_t2(&case.t2, &case.ovov);
    eprintln!("water/STO-3G CCSD, E_corr(t2) = {e_ref:.10}");

    let mut last_err = -1.0_f64;
    let mut last_ret = 2.0_f64;
    for cutoff in [f64::INFINITY, 4.0, 2.0, 1.0, 0.5] {
        let d = domains_for(&case.centers, cutoff);
        let mut masked = case.t2.clone();
        apply_pair_mask(&mut masked, &d).unwrap();
        let e = energy_from_t2(&masked, &case.ovov);
        let err = (e - e_ref).abs();
        let ret = pair_mask_retention(&d);

        eprintln!(
            "  cutoff {cutoff:>5} Bohr: retention {ret:.3}  E = {e:.10}  |dE| = {err:.3e}"
        );
        assert!(
            ret <= last_ret + 1e-12,
            "retention rose when the cutoff tightened ({ret} > {last_ret})"
        );
        assert!(
            err >= last_err - 1e-12,
            "error FELL when the cutoff tightened ({err:.3e} < {last_err:.3e}) -- \
             screening should only ever lose correlation"
        );
        last_ret = ret;
        last_err = err;
    }
}

/// Screening must actually bite at some cutoff, otherwise the knob is inert and
/// none of the above means anything.
#[test]
fn screening_is_not_inert() {
    let case = water_ccsd();
    let e_ref = energy_from_t2(&case.t2, &case.ovov);

    let d = domains_for(&case.centers, 0.5);
    let mut masked = case.t2.clone();
    let zeroed = apply_pair_mask(&mut masked, &d).unwrap();
    let e = energy_from_t2(&masked, &case.ovov);

    eprintln!(
        "tight cutoff: zeroed {zeroed} (i,j) blocks, retention {:.3}, dE {:+.3e}",
        pair_mask_retention(&d),
        e - e_ref
    );
    assert!(zeroed > 0, "a 0.5 Bohr cutoff should screen something on water");
    assert!((e - e_ref).abs() > 1e-12, "screening changed no energy -- the mask is inert");
    // Dropping amplitude blocks removes (negative) correlation, so E_corr rises.
    assert!(
        e > e_ref,
        "masking should REDUCE |E_corr|: got {e:.10} vs reference {e_ref:.10}"
    );
}
