//! Stage 6: Gamma-point closed-shell MP2 (`ferric_pbc::mp2::gamma_mp2`),
//! ported from `reference/pbc/pbc_mp2.py` (FINDINGS "Iteration 3 (Python,
//! Gamma MP2)"). Every pinned number below is from that iteration's
//! measurements (`run_mp2_anchor.py`, `run_mp2_oracle.py`,
//! `run_mp2_box_limit.py`, `run_mp2_fit_error.py`).
//!
//! # Exactness anchor (written first)
//!
//! Triclinic 4H, ONE primitive s (α = 0.5) per H, aux = every periodic pair
//! product (10 atom pairs × 8 half-lattice classes = 80 s functions on ghost
//! sites): the periodic pair densities lie exactly in the aux span, so MP2
//! from the RS-GDF B must equal MP2 from the exact dense pure-AFT (ia|jb)
//! (prototype |ΔE| 2e-13; asserted 1e-11). The two sides share NO MO
//! transform (per-k `C_oᵀ B_k C_v` vs `Wᵀ I W`) and NO energy loop
//! (`spin_components_from_b_ov` vs `spin_components_from_g`).
//! The anchor is triclinic 4H single-s, not s+p: s+p pair products are not
//! single s Gaussians, so no finite s aux set is exact there. H2 alone is not
//! enough: with nocc = nvir = 1 a dropped exchange term is invisible
//! (prototype −7e-14); on 4H it moves E by 1.4e-6.
//!
//! # Artifact hypotheses (stated before measuring)
//!
//! * "The anchor is vacuous": an incomplete aux (7 of 8 classes) must fail
//!   it (prototype +1.1e-8), and the dropped-exchange mutant must be
//!   visible (|e_ss| > 1e-7; prototype 1.4e-6) — both asserted in-test.
//! * "Shifted converges because of a sign/convention slip": then the box
//!   residual would decay as 1/a (a flipped Madelung sign gives ≈ 2 c1/a);
//!   physics predicts a⁻³ with c3 from MOLECULAR moments (0.671094; prototype
//!   fit 0.671358). Unshifted must decay as 1/a with the prototype's values.
//! * Anchor blind spots (prototype, pinned so nobody over-reads the anchor):
//!   ov-MP2 cannot see the J3 G = 0 term (C_oᵀ S C_v = 0) and sees the J2
//!   term only through the fitted ov charge, exactly 0 in the anchor. Those
//!   stay guarded by the HF ERI anchor in `pbc_rsgdf.rs`.
//!
//! # Mutation plan (for the main agent; each must fail ≥ 1 test here)
//!
//! * `b_ov_from_ao_b`: use `c_vir` for both indices → anchor fails.
//! * `gamma_mp2` shift table: swap the sign of `-madelung` for the
//!   (None, MadelungShifted) row → pinned-PySCF + fit-error tests fail.
//! * `gamma_mp2`: apply `occ_shift` to the virtuals too (`take(nmo)`) → all
//!   shifted pins fail (denominators unchanged).
//! * NOT covered: a `first_occ` slicing slip in `gamma_mp2` hits both paths
//!   identically (the anchor cannot see it); only the ferric-mp2 molecular
//!   frozen-core tests and the prototype's formula test pin that logic.
//!
//! # Molecular path unchanged (task item 2)
//!
//! No ferric-mp2 code was edited: the seam already existed
//! (`spin_components_from_b_ov`, which `ri_mp2_spin_components` reaches via
//! `spin_components_from_b_ov_kappa(.., None)`). The regression to run anyway:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-mp2 --lib rimp2::tests
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-mp2 \
//!     --test mwe_mp2_pool_is_inert_without_a_pool --test kappa_mp2
//! ```
//!
//! plus `molecular_ri_mp2_goes_through_the_shared_b_ov_kernel` below
//! (bitwise).

mod common;

use common::*;
use ferric_core::basis;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ferric_mp2::canonical::{canonical_mp2, dense_ao_eri};
use ferric_mp2::rimp2::{ri_mp2, ri_mp2_spin_components, spin_components_from_b_ov, RiMp2Config};
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::mp2::{
    gamma_mp2, GammaMp2Config, GammaMp2Integrals, GammaMp2Result, Mp2Denominators,
};
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig, DEFAULT_FITTED_ERI_MAX_BYTES};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::f64::consts::PI;

use ferric_pbc::mp2::Mp2Denominators::{MadelungShifted as Shifted, Unshifted};

/// ω for the nuclear-attraction split (as in pbc_dense_aft_scf.rs; not 1).
const HCORE_OMEGA: f64 = 0.8;
const ANCHOR_ALPHA: f64 = 0.5;
const AMPLE: usize = 1 << 30;

// --- PySCF 2.13 pbc.mp.RMP2 on AFTDF (mesh 61³), FINDINGS Iteration 3
// "Oracle" table. Values are the prototype's pure-AFT MP2 ("ours"); PySCF
// differs by +4.2e-16/+2.9e-16 (H2) and −1.2e-10/+3.7e-11 (tri).
// RMP2(exxdiv=None) ≡ unshifted, RMP2(exxdiv='ewald') ≡ shifted, and PySCF
// RCCSD's eris-MP2 ≡ shifted under BOTH HF exxdiv (to 5.7e-11).
const H2_MP2_UNSHIFTED: f64 = -5.891222456423e-3;
const H2_MP2_SHIFTED: f64 = -3.881428851328e-3;
const H2_VM: f64 = 0.7093243699;
const TRI_MP2_UNSHIFTED: f64 = -6.080626811746e-2;
const TRI_MP2_SHIFTED: f64 = -4.009219916836e-2;
const TRI_VM: f64 = 0.6224368790;

// --- Box limit, H2/STO-3G (FINDINGS Iteration 3 "Box limit" table):
// residual E_corr(pbc) − E_corr(mol), 4 significant figures.
const MOL_MP2_H2_STO3G: f64 = -1.315787005264e-2; // pyscf.mp.MP2, cart
const BOX_EDGES: [f64; 3] = [16.0, 20.0, 24.0];
const BOX_SHIFTED: [f64; 3] = [1.646e-4, 8.419e-5, 4.868e-5];
const BOX_UNSHIFTED: [f64; 3] = [-1.980e-3, -1.589e-3, -1.321e-3];
/// Prototype tail fit (24, 32, 40) and the molecular moment prediction.
const PROTO_C3_FIT: f64 = 0.671358;
const PROTO_C3_PRED: f64 = 0.67109405;

fn hcore(cell: &Cell, prep: &PreparedBasis) -> PeriodicHcore {
    periodic_hcore(cell, prep, &PeriodicHcoreConfig::with_omega(HCORE_OMEGA)).expect("hcore")
}

fn dense_none(cell: &Cell, prep: &PreparedBasis, hc: &PeriodicHcore) -> DenseAftEri {
    DenseAftEri::build(
        cell,
        prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("dense AFT")
}

fn gdf_cfg(omega: f64) -> RsGdfConfig {
    RsGdfConfig {
        omega,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

fn mp2_cfg(reference_exxdiv: ExxDiv, denominators: Mp2Denominators, fc: usize) -> GammaMp2Config {
    GammaMp2Config {
        frozen_core: fc,
        reference_exxdiv,
        denominators,
        budget_bytes: Some(AMPLE),
    }
}

fn mp2(
    cell: &Cell,
    rhf: &ScfResult,
    ints: GammaMp2Integrals<'_>,
    exx: ExxDiv,
    den: Mp2Denominators,
) -> GammaMp2Result {
    gamma_mp2(cell, rhf, ints, &mp2_cfg(exx, den, 0)).expect("gamma MP2")
}

fn gdf_rhf(cell: &Cell, prep: &PreparedBasis, hc: &PeriodicHcore, gdf: &RsGdf) -> ScfResult {
    gamma_rhf_jk(
        cell,
        prep,
        hc,
        Box::new(gdf.j_builder()),
        Box::new(gdf.k_builder()),
    )
}

/// Anchor aux centres for EVERY atom pair i <= j (`run_mp2_anchor.anchor`):
/// `(R_i + R_j + h·A)/2`, `h ∈ {0,1}³` in np.ndindex order, restricted to
/// the classes `k ∈ classes`; exponent 2α.
fn anchor_sites(cell: &Cell, classes: &[usize]) -> Vec<[f64; 4]> {
    let r = cell.positions();
    let a = cell.lattice();
    let n = r.len();
    let mut out = Vec::new();
    for i in 0..n {
        for j in i..n {
            for k in 0..8usize {
                if !classes.contains(&k) {
                    continue;
                }
                let h = [(k >> 2) & 1, (k >> 1) & 1, k & 1];
                let mut c = [0.0; 3];
                for d in 0..3 {
                    let ha: f64 = (0..3).map(|x| h[x] as f64 * a[x][d]).sum();
                    c[d] = 0.5 * (r[i][d] + r[j][d] + ha);
                }
                out.push([c[0], c[1], c[2], 2.0 * ANCHOR_ALPHA]);
            }
        }
    }
    out
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    eprintln!(
        "  {what}: {got:.12e} (want {want:.12e}, diff {:.2e})",
        got - want
    );
    assert!(got.is_finite(), "{what}: non-finite");
    assert!(
        (got - want).abs() < tol,
        "{what}: {got:.12e} vs {want:.12e} (tol {tol:.1e})"
    );
}

// ===========================================================================
// (1) Exactness anchor.
// ===========================================================================

#[test]
fn mp2_from_rsgdf_matches_dense_aft_in_the_trivial_aux_limit() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    let hc = hcore(&cell, &prep);
    let eri = dense_none(&cell, &prep, &hc);
    let site = SiteBasis::new(&anchor_sites(&cell, &[0, 1, 2, 3, 4, 5, 6, 7]), 0).unwrap();
    assert_eq!(site.prep.nbasis(), 80);
    let gdf = RsGdf::build(&cell, &prep, &site.prep, &hc.s, &gdf_cfg(0.8)).unwrap();
    // Prerequisite: the aux span IS exact (HF-level anchor on this cell).
    let d_eri = max_abs_diff(
        &gdf.fitted_eri(DEFAULT_FITTED_ERI_MAX_BYTES).unwrap(),
        eri.eri(),
    );
    eprintln!(
        "tri 4H anchor: kept {}/{}, max|I_fit − I| = {d_eri:.2e}",
        gdf.stats().naux_kept,
        gdf.stats().naux
    );
    assert!(d_eri < 1e-10, "aux span not exact: {d_eri:.3e}");

    let rhf_dense = gamma_rhf(&cell, &prep, &hc, &eri);
    let rhf_gdf = gdf_rhf(&cell, &prep, &hc, &gdf);
    for den in [Shifted, Unshifted] {
        let exact = mp2(
            &cell,
            &rhf_dense,
            GammaMp2Integrals::DenseAft(&eri),
            ExxDiv::None,
            den,
        );
        let same_c = mp2(
            &cell,
            &rhf_dense,
            GammaMp2Integrals::RsGdf(&gdf),
            ExxDiv::None,
            den,
        );
        let own = mp2(
            &cell,
            &rhf_gdf,
            GammaMp2Integrals::RsGdf(&gdf),
            ExxDiv::None,
            den,
        );
        eprintln!(
            "{den:?}: exact {:.12e} (os {:.6e} ss {:.6e}); B same C {:+.2e}; B own SCF {:+.2e}",
            exact.mp2_corr,
            exact.components.e_os,
            exact.components.e_ss,
            same_c.mp2_corr - exact.mp2_corr,
            own.mp2_corr - exact.mp2_corr
        );
        assert_eq!((exact.nocc_active, exact.nvir), (2, 2));
        assert_eq!(same_c.naux, Some(gdf.stats().naux_kept));
        for (x, lab) in [(&same_c, "same C"), (&own, "own SCF")] {
            assert!((x.mp2_corr - exact.mp2_corr).abs() < 1e-11, "{den:?} {lab}");
            assert!(
                (x.components.e_os - exact.components.e_os).abs() < 1e-11,
                "{den:?} {lab} e_os"
            );
        }
        // Reachable failure: dropping the exchange term (E -> e_os) moves E
        // by |e_ss| (prototype 1.4e-6, shifted) — far above the tolerance.
        assert!(
            exact.components.e_ss.abs() > 1e-7,
            "{den:?}: anchor cannot see a dropped exchange term (|e_ss| = {:.2e})",
            exact.components.e_ss.abs()
        );

        // Frozen core through active_occ: B vs dense agree with fc = 1 too;
        // fc = nocc is an error, not an underflow.
        let fc = |ints| gamma_mp2(&cell, &rhf_dense, ints, &mp2_cfg(ExxDiv::None, den, 1));
        let (e1, e2) = (
            fc(GammaMp2Integrals::DenseAft(&eri)).unwrap(),
            fc(GammaMp2Integrals::RsGdf(&gdf)).unwrap(),
        );
        assert_eq!(e1.nocc_active, 1);
        assert!(
            (e1.mp2_corr - e2.mp2_corr).abs() < 1e-11,
            "{den:?} frozen core"
        );
        assert!(
            (e1.mp2_corr - exact.mp2_corr).abs() > 1e-9,
            "frozen core had no effect"
        );
        let err = gamma_mp2(
            &cell,
            &rhf_dense,
            GammaMp2Integrals::RsGdf(&gdf),
            &mp2_cfg(ExxDiv::None, den, 2),
        )
        .expect_err("frozen_core = nocc must be refused");
        assert!(err.to_string().contains("frozen_core"), "{err}");
    }

    // Negative control: an incomplete aux (7 of 8 classes) must fail the
    // anchor (prototype +1.1e-8).
    let seven = SiteBasis::new(&anchor_sites(&cell, &[0, 1, 2, 3, 4, 5, 6]), 0).unwrap();
    let g7 = RsGdf::build(&cell, &prep, &seven.prep, &hc.s, &gdf_cfg(0.8)).unwrap();
    let exact = mp2(
        &cell,
        &rhf_dense,
        GammaMp2Integrals::DenseAft(&eri),
        ExxDiv::None,
        Shifted,
    );
    let e7 = mp2(
        &cell,
        &rhf_dense,
        GammaMp2Integrals::RsGdf(&g7),
        ExxDiv::None,
        Shifted,
    );
    let d7 = e7.mp2_corr - exact.mp2_corr;
    eprintln!("aux 7/8 classes: dMP2 = {d7:+.3e} (prototype +1.1e-8)");
    assert!(d7.abs() > 1e-9, "incomplete aux not detected: {d7:.3e}");
}

// ===========================================================================
// (2) Molecular path: the PBC kernel IS the molecular kernel.
// ===========================================================================

#[test]
fn molecular_ri_mp2_goes_through_the_shared_b_ov_kernel() {
    let mol = hydrogens(&H2_ATOMS);
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            density_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(rhf.converged);
    let cfg = RiMp2Config::default();
    let (sc, b_flat) = ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let nocc = 1;
    let nvir = rhf.eps_r().len() - nocc;
    let again = spin_components_from_b_ov(&b_flat, rhf.eps_r(), nocc, nvir, 0, nocc);
    let e = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap().mp2_corr;
    eprintln!(
        "molecular H2/cc-pVDZ RI-MP2 {e:.15e}; shared kernel {:.15e}",
        again.e_total
    );
    assert_eq!(sc.e_total.to_bits(), again.e_total.to_bits());
    assert_eq!(e.to_bits(), again.e_total.to_bits());
}

// ===========================================================================
// (3) PySCF pbc.mp.RMP2 (AFTDF) pins, both conventions, all four
//     reference/denominator combinations.
// ===========================================================================

fn pinned_case(
    name: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    vm_ref: f64,
    unshifted: f64,
    shifted: f64,
    tol: f64,
) {
    let hc = hcore(cell, prep);
    let base = dense_none(cell, prep, &hc);
    let ew = base.clone().with_exxdiv(cell, ExxDiv::Ewald).unwrap();
    let rhf_none = gamma_rhf(cell, prep, &hc, &base);
    let rhf_ewald = gamma_rhf(cell, prep, &hc, &ew);
    let ints = GammaMp2Integrals::DenseAft(&base);
    eprintln!("{name}:");
    for (rhf, exx, den, want, shift) in [
        (&rhf_none, ExxDiv::None, Unshifted, unshifted, 0.0),
        (&rhf_none, ExxDiv::None, Shifted, shifted, -vm_ref),
        (&rhf_ewald, ExxDiv::Ewald, Shifted, shifted, 0.0),
        (&rhf_ewald, ExxDiv::Ewald, Unshifted, unshifted, vm_ref),
    ] {
        let r = mp2(cell, rhf, ints, exx, den);
        close(r.madelung, vm_ref, 1e-9, "v_M");
        assert!(
            (r.occ_shift - shift).abs() < 1e-9,
            "occ_shift {}",
            r.occ_shift
        );
        close(r.mp2_corr, want, tol, &format!("ref {exx:?} → {den:?}"));
    }
    // A MISLABELLED reference (ewald SCF declared as None) double-shifts:
    // the stated convention is load-bearing, not decorative.
    let wrong = mp2(cell, &rhf_ewald, ints, ExxDiv::None, Shifted);
    eprintln!("  mislabelled ewald-as-none: {:.6e}", wrong.mp2_corr);
    assert!((wrong.mp2_corr - shifted).abs() > 1e-4);
}

#[test]
fn h2_sto3g_a4_matches_pinned_pyscf_rmp2() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    // H2 minimal: C is fixed by symmetry, so only ε and the ERI matter.
    pinned_case(
        "H2/STO-3G a=4",
        &cell,
        &prep,
        H2_VM,
        H2_MP2_UNSHIFTED,
        H2_MP2_SHIFTED,
        1e-10,
    );
}

#[test]
fn triclinic_4h_sp_matches_pinned_pyscf_rmp2() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    // SCF density_conv 1e-10; PySCF-vs-prototype scatter 1.2e-10.
    pinned_case(
        "tri 4H s+p",
        &cell,
        &prep,
        TRI_VM,
        TRI_MP2_UNSHIFTED,
        TRI_MP2_SHIFTED,
        1e-9,
    );
}

// ===========================================================================
// (4) Box limit vs ferric's OWN molecular MP2 (exact: canonical_mp2 on
//     dense libint2 ERIs, no DF on either side — the periodic side is the
//     dense pure-AFT oracle, so the residual is finite-size only).
// ===========================================================================

#[test]
fn box_limit_shifted_is_a3_with_the_predicted_coefficient_unshifted_is_1_over_a() {
    let basis = pyscf_sto3g_h();
    let mol = hydrogens(&H2_ATOMS);
    let prep_mol = PreparedBasis::new(&mol, &basis).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep_mol).unwrap();
    let mr = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &prep_mol,
        op,
        &bounds,
        &RhfConfig {
            density_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(mr.converged);
    let e_mol = canonical_mp2(&mol, &prep_mol, op, &mr, 0).unwrap();
    close(
        e_mol,
        MOL_MP2_H2_STO3G,
        1e-9,
        "molecular MP2 (ferric canonical vs PySCF)",
    );

    // Moment-model prediction of c3 from MOLECULAR quantities only
    // (pbc_mp2.dipole_prediction_h2_minimal): one occupied i, one virtual a.
    let c = mr.mos_r();
    let (ci, ca) = (c.column(0).to_owned(), c.column(1).to_owned());
    let dip = oneelectron::dipole(&prep_mol, [0.0; 3]).unwrap();
    let d_ia: Vec<f64> = (0..3).map(|x| ci.dot(&dip[x].dot(&ca))).collect();
    let cen = [
        ci.dot(&dip[0].dot(&ci)),
        ci.dot(&dip[1].dot(&ci)),
        ci.dot(&dip[2].dot(&ci)),
    ];
    let sigma2 = ci.dot(&oneelectron::r2_moment(&prep_mol, cen).unwrap().dot(&ci));
    let ao = dense_ao_eri(&prep_mol, op).unwrap();
    let n = prep_mol.nbasis();
    let mut iaia = 0.0;
    for m in 0..n {
        for nu in 0..n {
            for l in 0..n {
                for sg in 0..n {
                    iaia += ci[m] * ca[nu] * ci[l] * ca[sg] * ao[((m * n + nu) * n + l) * n + sg];
                }
            }
        }
    }
    let eps = mr.eps_r();
    let dd = 2.0 * eps[0] - 2.0 * eps[1];
    let k = 4.0 * PI / 3.0;
    let d2: f64 = d_ia.iter().map(|x| x * x).sum();
    let c3_pred =
        2.0 * iaia * (-k * d2) / dd - iaia * iaia * (-2.0 * k * sigma2 - 2.0 * k * d2) / (dd * dd);
    eprintln!(
        "E_MP2(mol) {e_mol:.12e}; sigma2 {sigma2:.6}, |d_ia|^2 {d2:.6}, (ia|ia) {iaia:.6}, D {dd:.6} \
         => c3_pred {c3_pred:.8} (prototype {PROTO_C3_PRED})"
    );
    assert!(
        (c3_pred - PROTO_C3_PRED).abs() < 1e-6,
        "moment model {c3_pred}"
    );

    // Periodic boxes (dense pure AFT, exxdiv = ewald reference).
    let prec = 1e-11;
    let mut sh = Vec::new();
    let mut un = Vec::new();
    for (w, &a) in BOX_EDGES.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let cell = h2_cell(a);
        let prep = prep_for(&cell, &basis);
        let hcfg = PeriodicHcoreConfig {
            precision: prec,
            ..PeriodicHcoreConfig::with_omega(0.9)
        };
        let hc = periodic_hcore(&cell, &prep, &hcfg).unwrap();
        let eri = DenseAftEri::build(
            &cell,
            &prep,
            &hc.s,
            ExxDiv::Ewald,
            prec,
            DEFAULT_DENSE_AFT_MAX_BYTES,
        )
        .unwrap();
        let rhf = gamma_rhf(&cell, &prep, &hc, &eri);
        let ints = GammaMp2Integrals::DenseAft(&eri);
        let ds = mp2(&cell, &rhf, ints, ExxDiv::Ewald, Shifted).mp2_corr - e_mol;
        let du = mp2(&cell, &rhf, ints, ExxDiv::Ewald, Unshifted).mp2_corr - e_mol;
        eprintln!(
            "a = {a:4.1}: dE shifted {ds:+.6e} (proto {:+.3e}), unshifted {du:+.6e} (proto {:+.3e}), \
             c3/a^3 {:+.6e} ({:.1} s)",
            BOX_SHIFTED[w],
            BOX_UNSHIFTED[w],
            c3_pred / a.powi(3),
            t0.elapsed().as_secs_f64()
        );
        // Prototype table to its printed precision (half a unit in the 4th
        // significant figure).
        let half_unit = |x: f64| 0.5 * 10f64.powi(x.abs().log10().floor() as i32 - 3);
        close(
            ds,
            BOX_SHIFTED[w],
            half_unit(BOX_SHIFTED[w]),
            "shifted residual",
        );
        close(
            du,
            BOX_UNSHIFTED[w],
            half_unit(BOX_UNSHIFTED[w]),
            "unshifted residual",
        );
        sh.push(ds);
        un.push(du);
    }
    let local = |v: &[f64], i: usize| {
        -(v[i + 1] / v[i]).abs().ln() / (BOX_EDGES[i + 1] / BOX_EDGES[i]).ln()
    };
    for i in 0..2 {
        let (ps, pu) = (local(&sh, i), local(&un, i));
        eprintln!(
            "local exponent {}->{}: shifted {ps:.4}, unshifted {pu:.4}",
            BOX_EDGES[i],
            BOX_EDGES[i + 1]
        );
        assert!(ps > 2.8 && ps < 3.3, "shifted exponent {ps}");
        assert!(pu > 0.6 && pu < 1.4, "unshifted exponent {pu}");
    }
    assert!(sh.iter().all(|d| *d > 0.0) && un.iter().all(|d| *d < 0.0));
    assert!(un[2].abs() > 10.0 * sh[2].abs());

    // Least squares dE = c3 a^-3 + c5 a^-5 (3 points, 2 params). With the
    // prototype's own table at (16,20,24) this fit gives 0.67214.
    let (mut sxx, mut sxy, mut syy, mut sxd, mut syd) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&a, &d) in BOX_EDGES.iter().zip(&sh) {
        let (x, y) = (a.powi(-3), a.powi(-5));
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
        sxd += x * d;
        syd += y * d;
    }
    let det = sxx * syy - sxy * sxy;
    let c3 = (sxd * syy - syd * sxy) / det;
    let c5 = (sxx * syd - sxy * sxd) / det;
    let rel_pred = (c3 - c3_pred).abs() / c3_pred;
    let rel_fit = (c3 - PROTO_C3_FIT).abs() / PROTO_C3_FIT;
    eprintln!(
        "fit: c3 = {c3:.6}, c5 = {c5:.4}; predicted {c3_pred:.6} (rel {rel_pred:.2e}); \
         prototype tail fit {PROTO_C3_FIT} (rel {rel_fit:.2e})"
    );
    assert!(rel_pred < 1e-2, "c3 {c3} vs predicted {c3_pred}");
    assert!(rel_fit < 1e-2, "c3 {c3} vs prototype fit {PROTO_C3_FIT}");
}

// ===========================================================================
// (5) cc-pvdz-ri fitting error vs dense (FINDINGS Iteration 3 aux table).
// ===========================================================================

/// `(dMP2 shifted same C, shifted own SCF, unshifted same C, rel shifted
/// same C)`, as printed (3 / 3 / 3 / 2 significant figures).
struct FitRow {
    same_s: f64,
    own_s: f64,
    same_u: f64,
    rel: f64,
}

fn fit_error_case(cell: &Cell, obs: &PreparedBasis, naux: usize, row: &FitRow) {
    let hc = hcore(cell, obs);
    let eri = dense_none(cell, obs, &hc);
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    assert_eq!(
        aux.nbasis(),
        naux,
        "aux count differs from the prototype's spherical set"
    );
    let gdf = RsGdf::build(cell, obs, &aux, &hc.s, &gdf_cfg(1.0)).unwrap();
    let rhf0 = gamma_rhf(cell, obs, &hc, &eri);
    let rhf1 = gdf_rhf(cell, obs, &hc, &gdf);
    let dense = GammaMp2Integrals::DenseAft(&eri);
    let fit = GammaMp2Integrals::RsGdf(&gdf);
    let ex_s = mp2(cell, &rhf0, dense, ExxDiv::None, Shifted).mp2_corr;
    let ex_u = mp2(cell, &rhf0, dense, ExxDiv::None, Unshifted).mp2_corr;
    let same_s = mp2(cell, &rhf0, fit, ExxDiv::None, Shifted).mp2_corr - ex_s;
    let own_s = mp2(cell, &rhf1, fit, ExxDiv::None, Shifted).mp2_corr - ex_s;
    let same_u = mp2(cell, &rhf0, fit, ExxDiv::None, Unshifted).mp2_corr - ex_u;
    let rel = same_s / ex_s;
    let half = |x: f64, sig: i32| 0.5 * 10f64.powi(x.abs().log10().floor() as i32 - (sig - 1));
    close(
        same_s,
        row.same_s,
        half(row.same_s, 3),
        "dMP2 shifted, same C",
    );
    close(
        own_s,
        row.own_s,
        half(row.own_s, 3),
        "dMP2 shifted, own SCF",
    );
    close(
        same_u,
        row.same_u,
        half(row.same_u, 3),
        "dMP2 unshifted, same C",
    );
    close(rel, row.rel, half(row.rel, 2), "relative (shifted, same C)");
    assert!(
        same_s > 0.0 && own_s > 0.0 && same_u > 0.0,
        "fit errors must underbind"
    );
}

#[test]
fn h2_sto3g_a4_cc_pvdz_ri_fitting_error_matches_prototype() {
    let cell = h2_cell(4.0);
    let obs = prep_for(&cell, &pyscf_sto3g_h());
    fit_error_case(
        &cell,
        &obs,
        28,
        &FitRow {
            same_s: 6.31e-7,
            own_s: 7.02e-7,
            same_u: 9.58e-7,
            rel: -1.6e-4,
        },
    );
}

#[test]
#[ignore = "slow: triclinic 4H s+p RS-GDF with cc-pvdz-ri (17-21M SR triplets) plus the \
            dense-AFT oracle and two SCFs; run with --release -- --ignored, serially, on a quiet box"]
fn triclinic_4h_sp_cc_pvdz_ri_fitting_error_matches_prototype() {
    let cell = triclinic_cell();
    let obs = prep_for(&cell, &sp_basis_h());
    fit_error_case(
        &cell,
        &obs,
        56,
        &FitRow {
            same_s: 3.33e-5,
            own_s: 3.65e-5,
            same_u: 4.85e-5,
            rel: -8.3e-4,
        },
    );
}

#[test]
fn gamma_mp2_rejects_bad_references() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc = hcore(&cell, &prep);
    let eri = dense_none(&cell, &prep, &hc);
    let rhf = gamma_rhf(&cell, &prep, &hc, &eri);
    let ints = GammaMp2Integrals::DenseAft(&eri);
    // frozen_core = nocc (= 1): active_occ refuses.
    assert!(gamma_mp2(&cell, &rhf, ints, &mp2_cfg(ExxDiv::None, Shifted, 1)).is_err());
    // Unconverged reference.
    let mut bad = rhf.clone();
    bad.converged = false;
    assert!(gamma_mp2(&cell, &bad, ints, &mp2_cfg(ExxDiv::None, Shifted, 0)).is_err());
    // A tiny budget names the quantity.
    let msg = gamma_mp2(
        &cell,
        &rhf,
        ints,
        &GammaMp2Config {
            budget_bytes: Some(1),
            ..mp2_cfg(ExxDiv::None, Shifted, 0)
        },
    )
    .unwrap_err()
    .to_string();
    assert!(msg.contains("Gamma MP2 dense"), "{msg}");
}
