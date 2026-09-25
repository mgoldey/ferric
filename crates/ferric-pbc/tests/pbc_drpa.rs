//! Stage 7: Gamma-point closed-shell dRPA (`ferric_pbc::drpa::gamma_drpa`),
//! ported from `reference/pbc/pbc_rpa.py` (FINDINGS "Iteration 4 (Python,
//! Gamma dRPA)"). Every pinned number is from that iteration
//! (`run_rpa_anchor.py`, `run_rpa_oracle.py`, `run_rpa_box_limit.py`,
//! `run_rpa_fit_error.py`).
//!
//! # Exactness anchor (written first)
//!
//! Triclinic 4H, ONE primitive s (α = 0.5) per H, aux = every periodic pair
//! product (as in `pbc_mp2.rs`): dRPA from the RS-GDF B through ferric-rpa's
//! frequency integral (Lanczos eigensolve, log-det summands) must equal dRPA
//! from the exact dense pure-AFT `(ia|jb)` through the PLASMON formula
//! (prototype 1.7e-13 / 3.3e-13; asserted 1e-11). The two sides share no ERI
//! construction, no MO transform and no energy formula.
//!
//! # Artifact hypotheses (stated before measuring)
//!
//! * "The anchor is vacuous": an incomplete aux (7 of 8 classes) must fail it
//!   (prototype +2.1e-8); a lost spin factor (B/√2) and a Cv⊗Cv transform
//!   must move E by > 1e-3 (prototype 1.5e-2 / 6.3e-3). All asserted here.
//! * "The normalisation (factor 4, 1/2π on the half line) is right only
//!   because PySCF agrees": the O(V²) term is checked against DIRECT MP2 from
//!   `gamma_mp2` (a different kernel, `spin_components_from_b_ov`), both by
//!   an independent `tr Π²` quadrature and by the λ → 0 limit of the
//!   PRODUCTION path. A 1/π normalisation fails by 100 %, a factor 2 by 300 %.
//! * "Shifted converges because of a sign slip": physics predicts a⁻³ with
//!   c3 = 0.922741 from MOLECULAR moments (nov = 1 closed form); a flipped
//!   Madelung sign gives a 1/a residual; a normalisation slip gives a plateau.
//! * Anchor blind spots: like MP2, ov-only dRPA cannot see either G = 0 term
//!   of the fit (the three pbc_gdf G = 0 mutants moved dRPA by <= 1.7e-13 in
//!   the prototype). The HF ERI anchor in `pbc_rsgdf.rs` is their only guard.
//!
//! # Molecular box-limit oracle
//!
//! Test (4) compares against ferric's OWN molecular dRPA computed by the
//! DENSE PLASMON ORACLE on exact libint2 ERIs (`dense_ao_eri` →
//! `ovov_from_dense` → `drpa_plasmon`), NOT `run_pdep_rpa`: the latter needs
//! an aux basis, and its RI error (~2.4e-6 on H2/cc-pvdz-ri) is 4 % of the
//! a = 24 residual.
//!
//! # Mutation plan (for the main agent; each must fail ≥ 1 test here)
//!
//! * ferric-rpa `run_pdep_rpa_from_parts`: pass `eps_occ` for both energy
//!   slices → (1), (2), (6) fail.
//! * `gamma_drpa`: flip the sign of `occ_shift` (use `-occupied_shift`) →
//!   pins (3) and box limit (4) fail.
//! * `drpa_plasmon`: `4.0 * sd[p]` → `2.0 * sd[p]` → (1), (3), (4) fail.
//! * `drpa_second_order_from_b_ov`: `/ (2.0 * PI)` → `/ PI` → (2) fails.
//! * `gamma_drpa`: `metric_inv_sqrt` replaced by identity → no energy test can
//!   fail (eigenpotentials only) — documented blind spot; the shape check
//!   `w.ncols() != naux` is the only guard.
//!
//! # Molecular path unchanged (task item 6)
//!
//! ferric-rpa gained ONE additive function, `run_pdep_rpa_from_parts`; no
//! existing line of `run_pdep_rpa{,_from_intermediates,_eigensolve}` was
//! edited, so the molecular path is byte-identical by construction. Test (6)
//! below also shows the new entry reproduces `run_pdep_rpa_from_intermediates`
//! BITWISE on the same molecular intermediates. Regression to run anyway:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-rpa --lib
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-rpa \
//!     --test pdep_rpa --test lanczos --test freq_quad_parallel \
//!     --test mwe_rpa_pool_is_inert_without_a_pool --test p5_parallel_determinism
//! ```

mod common;

use common::*;
use ferric_core::basis;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ferric_mp2::canonical::dense_ao_eri;
use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config};
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::drpa::{
    drpa_from_b_ov, drpa_plasmon, drpa_second_order_from_b_ov, gamma_drpa, pdep_config,
    GammaDrpaConfig, GammaDrpaIntegrals, GammaDrpaResult, DEFAULT_GAMMA_DRPA_QUAD_POINTS,
};
use ferric_pbc::ewald::madelung_constant;
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::mp2::{
    b_ov_from_ao_b, gamma_mp2, ovov_from_dense, GammaMp2Config, GammaMp2Integrals, Mp2Denominators,
};
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig, DEFAULT_FITTED_ERI_MAX_BYTES};
use ferric_rpa::{run_pdep_rpa, run_pdep_rpa_from_intermediates, run_pdep_rpa_from_parts};
use ferric_rpa::{Chi0Sparsity, Eigensolver};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{s, Array2};
use std::f64::consts::PI;

use ferric_pbc::mp2::Mp2Denominators::{MadelungShifted as Shifted, Unshifted};

/// ω for the nuclear-attraction split (as in pbc_mp2.rs).
const HCORE_OMEGA: f64 = 0.8;
const ANCHOR_ALPHA: f64 = 0.5;
const AMPLE: usize = 1 << 30;
/// Anchor grid: 80 points (prototype quadrature error on these cells is
/// ~1e-15 at n >= 40; 80 keeps the anchor free of any grid question).
const ANCHOR_QUAD: usize = 80;

// --- FINDINGS Iteration 4 "PySCF oracles" table, column "ours exact"
// (pure-AFT (ia|jb) + plasmon). PySCF AFTDF (ia|jb) + plasmon differs by
// +6.7e-16 / +2.2e-16 (H2) and −1.5e-10 / +5.7e-11 (tri, SCF orbitals).
const H2_DRPA_UNSHIFTED: f64 = -1.000052013959e-2;
const H2_DRPA_SHIFTED: f64 = -6.938130614440e-3;
const H2_VM: f64 = 0.7093243699;
const TRI_DRPA_UNSHIFTED: f64 = -9.631964613214e-2;
const TRI_DRPA_SHIFTED: f64 = -6.846932539148e-2;
const TRI_VM: f64 = 0.6224368790;

// --- Box limit, H2/STO-3G (FINDINGS Iteration 4 "Box limit" table).
const MOL_DRPA_H2_STO3G: f64 = -2.065890717501e-2; // exact ERI, plasmon, cart
const BOX_EDGES: [f64; 3] = [16.0, 20.0, 24.0];
const BOX_SHIFTED: [f64; 3] = [2.267e-4, 1.159e-4, 6.697e-5];
const BOX_UNSHIFTED: [f64; 3] = [-2.377e-3, -1.927e-3, -1.612e-3];
/// (unshifted − shifted) / [E_mol(e_ia − v_M) − E_mol(e_ia)] at 16/20/24.
const BOX_SHIFT_FN_RATIO: [f64; 3] = [0.988, 0.994, 0.996];
/// Prototype a priori prediction (nov = 1 closed form == r2_kernel_c3) and
/// its (24, 32, 40) c3 + c5 tail fit. The prototype's own (16, 20, 24)
/// table gives c3 = 0.92422 under the same 2-parameter fit.
const PROTO_C3_PRED: f64 = 0.922741;
const PROTO_C3_FIT: f64 = 0.923006;

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

fn drpa_cfg(exx: ExxDiv, den: Mp2Denominators, quad_points: usize) -> GammaDrpaConfig {
    GammaDrpaConfig {
        frozen_core: 0,
        reference_exxdiv: exx,
        denominators: den,
        quad_points,
        budget_bytes: Some(AMPLE),
    }
}

fn drpa(
    cell: &Cell,
    rhf: &ScfResult,
    ints: GammaDrpaIntegrals<'_>,
    exx: ExxDiv,
    den: Mp2Denominators,
    quad_points: usize,
) -> GammaDrpaResult {
    gamma_drpa(cell, rhf, ints, &drpa_cfg(exx, den, quad_points)).expect("gamma dRPA")
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

/// Same construction as `pbc_mp2.rs::anchor_sites` (`run_mp2_anchor.anchor`).
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

/// Active (shifted per `occ_shift`) occupied and virtual energies of a
/// closed-shell reference with `nocc` occupied orbitals.
fn split_eps(rhf: &ScfResult, nocc: usize, occ_shift: f64) -> (Vec<f64>, Vec<f64>) {
    let e = rhf.eps_r();
    (
        e[..nocc].iter().map(|x| x + occ_shift).collect(),
        e[nocc..].to_vec(),
    )
}

/// The triclinic 4H trivial-aux anchor: (cell, prep, hcore, dense ERI, B,
/// RHF on dense, RHF on B).
struct Anchor {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
    eri: DenseAftEri,
    gdf: RsGdf,
    rhf_dense: ScfResult,
    rhf_gdf: ScfResult,
}

fn tri_anchor() -> Anchor {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    let hc = hcore(&cell, &prep);
    let eri = dense_none(&cell, &prep, &hc);
    let site = SiteBasis::new(&anchor_sites(&cell, &[0, 1, 2, 3, 4, 5, 6, 7]), 0).unwrap();
    assert_eq!(site.prep.nbasis(), 80);
    let gdf = RsGdf::build(&cell, &prep, &site.prep, &hc.s, &gdf_cfg(0.8)).unwrap();
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
    Anchor {
        cell,
        prep,
        hc,
        eri,
        gdf,
        rhf_dense,
        rhf_gdf,
    }
}

// ===========================================================================
// (1) Exactness anchor: RS-GDF + frequency integral ≡ dense AFT + plasmon.
// ===========================================================================

#[test]
fn drpa_from_rsgdf_matches_dense_plasmon_in_the_trivial_aux_limit() {
    let t = tri_anchor();
    let (cell, nao) = (&t.cell, t.prep.nbasis());
    let w = t.gdf.metric_inv_sqrt();
    assert_eq!(w.dim(), (t.gdf.stats().naux, t.gdf.stats().naux_kept));

    for den in [Shifted, Unshifted] {
        let exact = drpa(
            cell,
            &t.rhf_dense,
            GammaDrpaIntegrals::DenseAft(&t.eri),
            ExxDiv::None,
            den,
            ANCHOR_QUAD,
        );
        let same_c = drpa(
            cell,
            &t.rhf_dense,
            GammaDrpaIntegrals::RsGdf(&t.gdf),
            ExxDiv::None,
            den,
            ANCHOR_QUAD,
        );
        let own = drpa(
            cell,
            &t.rhf_gdf,
            GammaDrpaIntegrals::RsGdf(&t.gdf),
            ExxDiv::None,
            den,
            ANCHOR_QUAD,
        );
        let default_grid = drpa(
            cell,
            &t.rhf_dense,
            GammaDrpaIntegrals::RsGdf(&t.gdf),
            ExxDiv::None,
            den,
            DEFAULT_GAMMA_DRPA_QUAD_POINTS,
        );
        let mp2 = gamma_mp2(
            cell,
            &t.rhf_dense,
            GammaMp2Integrals::DenseAft(&t.eri),
            &GammaMp2Config {
                frozen_core: 0,
                reference_exxdiv: ExxDiv::None,
                denominators: den,
                budget_bytes: Some(AMPLE),
            },
        )
        .unwrap();
        let dmp2 = 2.0 * mp2.components.e_os;
        eprintln!(
            "{den:?}: exact (plasmon) {:.12e}; B same C {:+.2e}; B own SCF {:+.2e}; \
             n=40 {:+.2e}; dRPA − dMP2 {:+.3e}",
            exact.drpa_corr,
            same_c.drpa_corr - exact.drpa_corr,
            own.drpa_corr - exact.drpa_corr,
            default_grid.drpa_corr - exact.drpa_corr,
            exact.drpa_corr - dmp2
        );
        assert_eq!((exact.nocc_active, exact.nvir), (2, 2));
        assert_eq!(same_c.naux, Some(t.gdf.stats().naux_kept));
        assert_eq!(same_c.quad_points, Some(ANCHOR_QUAD));
        assert!(exact.naux.is_none() && exact.quad_points.is_none());
        for (x, lab) in [(&same_c, "same C"), (&own, "own SCF")] {
            assert!(
                (x.drpa_corr - exact.drpa_corr).abs() < 1e-11,
                "{den:?} {lab}: {:.3e}",
                x.drpa_corr - exact.drpa_corr
            );
        }
        assert!(
            (default_grid.drpa_corr - exact.drpa_corr).abs() < 1e-10,
            "{den:?}: default 40-point grid"
        );
        // Static dielectric ε̃(0) = 1 + Π(0) >= 1.
        let lam = same_c.eigenvalues_static.as_ref().unwrap();
        assert!(lam.iter().all(|&l| l >= 1.0 - 1e-12), "{lam:?}");
        // dRPA is not MP2 in disguise (prototype +3.8e-3 shifted).
        assert!((exact.drpa_corr - dmp2).abs() > 1e-4);
    }

    // Negative controls through the production energy path (shifted, exact C).
    let vm = madelung_constant(cell).unwrap();
    let (eo, ev) = split_eps(&t.rhf_dense, 2, -vm);
    let c = t.rhf_dense.mos_r();
    let (co, cv) = (c.slice(s![.., ..2]), c.slice(s![.., 2..]));
    let exact = drpa(
        cell,
        &t.rhf_dense,
        GammaDrpaIntegrals::DenseAft(&t.eri),
        ExxDiv::None,
        Shifted,
        ANCHOR_QUAD,
    )
    .drpa_corr;
    let b_ov = b_ov_from_ao_b(t.gdf.b(), nao, co, cv).unwrap();
    let run = |b: Array2<f64>| {
        drpa_from_b_ov(b, None, &eo, &ev, ANCHOR_QUAD, Some(AMPLE))
            .unwrap()
            .e_rpa
    };
    let base = run(b_ov.clone());
    assert!(
        (base - exact).abs() < 1e-11,
        "identity v_inv_sqrt changed E"
    );
    let spin = run(b_ov.mapv(|x| x / 2f64.sqrt())) - exact;
    let cvcv = run(b_ov_from_ao_b(t.gdf.b(), nao, cv, cv).unwrap()) - exact;
    let seven = SiteBasis::new(&anchor_sites(cell, &[0, 1, 2, 3, 4, 5, 6]), 0).unwrap();
    let g7 = RsGdf::build(cell, &t.prep, &seven.prep, &t.hc.s, &gdf_cfg(0.8)).unwrap();
    let d7 = run(b_ov_from_ao_b(g7.b(), nao, co, cv).unwrap()) - exact;
    eprintln!(
        "mutants: spin factor 4->2 {spin:+.2e} (proto +1.5e-2); Cv⊗Cv {cvcv:+.2e} \
         (proto -6.3e-3); aux 7/8 {d7:+.2e} (proto +2.1e-8)"
    );
    assert!(spin.abs() > 1e-3 && cvcv.abs() > 1e-3);
    assert!(d7.abs() > 1e-9, "incomplete aux not detected: {d7:.3e}");
}

// ===========================================================================
// (2) The O(V²) term is DIRECT MP2 from gamma_mp2 (independent kernel).
// ===========================================================================

#[test]
fn drpa_second_order_term_is_direct_mp2_from_gamma_mp2() {
    let t = tri_anchor();
    let (cell, nao) = (&t.cell, t.prep.nbasis());
    let vm = madelung_constant(cell).unwrap();
    let c = t.rhf_dense.mos_r();
    let (co, cv) = (c.slice(s![.., ..2]), c.slice(s![.., 2..]));
    let b_ov = b_ov_from_ao_b(t.gdf.b(), nao, co, cv).unwrap();
    for (den, shift) in [(Shifted, -vm), (Unshifted, 0.0)] {
        let (eo, ev) = split_eps(&t.rhf_dense, 2, shift);
        let mcfg = GammaMp2Config {
            frozen_core: 0,
            reference_exxdiv: ExxDiv::None,
            denominators: den,
            budget_bytes: Some(AMPLE),
        };
        let m_b = gamma_mp2(cell, &t.rhf_dense, GammaMp2Integrals::RsGdf(&t.gdf), &mcfg).unwrap();
        let m_d = gamma_mp2(
            cell,
            &t.rhf_dense,
            GammaMp2Integrals::DenseAft(&t.eri),
            &mcfg,
        )
        .unwrap();
        assert!((m_b.occ_shift - shift).abs() < 1e-12);
        let (dmp2_b, dmp2_d) = (2.0 * m_b.components.e_os, 2.0 * m_d.components.e_os);

        // (a) independent tr Π² quadrature (prototype n = 256: 1.9e-13).
        let e2 = drpa_second_order_from_b_ov(&b_ov, &eo, &ev, 256).unwrap();
        // (b) the PRODUCTION path in the weak-coupling limit: B → √λ B scales
        // the kernel by λ; E(λ)/λ² = dMP2 + c1 λ + c2 λ² + O(λ³). Two-level
        // Richardson on λ = h, h/2, h/4 (h = 0.04) removes c1 and c2:
        // (8 f(h/4) − 6 f(h/2) + f(h)) / 3. λ is kept >= 1e-2 so the log-det
        // summand's absolute roundoff (~1e-12, ulp of tr ε̃ ≈ naux) stays
        // ~1e-6 of E(λ).
        let f = |lam: f64| {
            drpa_from_b_ov(
                b_ov.mapv(|x| x * lam.sqrt()),
                None,
                &eo,
                &ev,
                ANCHOR_QUAD,
                Some(AMPLE),
            )
            .unwrap()
            .e_rpa
                / (lam * lam)
        };
        let h = 0.04;
        let (f1, f2, f4) = (f(h), f(h / 2.0), f(h / 4.0));
        let rich = (8.0 * f4 - 6.0 * f2 + f1) / 3.0;
        eprintln!(
            "{den:?}: dMP2 (B kernel) {dmp2_b:.12e}, (dense kernel) {:+.2e}; tr Π² term {:+.2e}; \
             E(λ)/λ² rel λ=0.04 {:+.2e}, 0.02 {:+.2e}, 0.01 {:+.2e}, Richardson {:+.2e}",
            dmp2_d - dmp2_b,
            e2 - dmp2_b,
            (f1 - dmp2_b) / dmp2_b,
            (f2 - dmp2_b) / dmp2_b,
            (f4 - dmp2_b) / dmp2_b,
            (rich - dmp2_b) / dmp2_b
        );
        assert!((dmp2_d - dmp2_b).abs() < 1e-11, "{den:?}: B vs dense dMP2");
        assert!((e2 - dmp2_b).abs() < 1e-11, "{den:?}: tr Π² term");
        assert!(
            ((rich - dmp2_b) / dmp2_b).abs() < 1e-4,
            "{den:?}: production path O(V²) limit"
        );
        // The limit is reached from the dRPA side, not by coincidence: the
        // uncorrected λ = 0.04 value still carries the O(λ) term.
        assert!(((f1 - dmp2_b) / dmp2_b).abs() > ((rich - dmp2_b) / dmp2_b).abs());
    }
}

// ===========================================================================
// (3) PySCF AFTDF pins, both conventions, all four reference/denominator
//     combinations (dense oracle).
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
    let ints = GammaDrpaIntegrals::DenseAft(&base);
    let q = DEFAULT_GAMMA_DRPA_QUAD_POINTS;
    eprintln!("{name}:");
    for (rhf, exx, den, want, shift) in [
        (&rhf_none, ExxDiv::None, Unshifted, unshifted, 0.0),
        (&rhf_none, ExxDiv::None, Shifted, shifted, -vm_ref),
        (&rhf_ewald, ExxDiv::Ewald, Shifted, shifted, 0.0),
        (&rhf_ewald, ExxDiv::Ewald, Unshifted, unshifted, vm_ref),
    ] {
        let r = drpa(cell, rhf, ints, exx, den, q);
        close(r.madelung, vm_ref, 1e-9, "v_M");
        assert!(
            (r.occ_shift - shift).abs() < 1e-9,
            "occ_shift {}",
            r.occ_shift
        );
        close(r.drpa_corr, want, tol, &format!("ref {exx:?} → {den:?}"));
    }
    let wrong = drpa(cell, &rhf_ewald, ints, ExxDiv::None, Shifted, q);
    eprintln!("  mislabelled ewald-as-none: {:.6e}", wrong.drpa_corr);
    assert!((wrong.drpa_corr - shifted).abs() > 1e-4);
}

#[test]
fn h2_sto3g_a4_matches_pinned_pyscf_aftdf_drpa() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    pinned_case(
        "H2/STO-3G a=4",
        &cell,
        &prep,
        H2_VM,
        H2_DRPA_UNSHIFTED,
        H2_DRPA_SHIFTED,
        1e-10,
    );
}

#[test]
fn triclinic_4h_sp_matches_pinned_pyscf_aftdf_drpa() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    // SCF density_conv 1e-10; prototype-vs-PySCF scatter 1.5e-10.
    pinned_case(
        "tri 4H s+p",
        &cell,
        &prep,
        TRI_VM,
        TRI_DRPA_UNSHIFTED,
        TRI_DRPA_SHIFTED,
        1e-9,
    );
}

// ===========================================================================
// (4) Box limit vs ferric's OWN molecular dRPA (dense plasmon oracle on
//     exact libint2 ERIs; the periodic side is the dense pure-AFT oracle, so
//     the residual is finite-size only).
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
    let n = prep_mol.nbasis();
    assert_eq!(n, 2);
    let ao = dense_ao_eri(&prep_mol, op).unwrap();
    let ao = Array2::from_shape_vec((n * n, n * n), ao).unwrap();
    let c = mr.mos_r();
    let ov_mol = ovov_from_dense(&ao, n, c.slice(s![.., ..1]), c.slice(s![.., 1..])).unwrap();
    let eps = mr.eps_r();
    let (eo, ev) = (vec![eps[0]], vec![eps[1]]);
    let e_mol = drpa_plasmon(&ov_mol, &eo, &ev).unwrap();
    close(
        e_mol,
        MOL_DRPA_H2_STO3G,
        1e-9,
        "molecular dRPA (ferric dense plasmon vs prototype)",
    );

    // nov = 1 closed-form moment model (pbc_rpa.drpa_moment_prediction_h2_minimal):
    // E = ½[Ω − D − 2K], Ω = √(D(D+4K)); dD = k(σ² + |d|²), dK = −k|d|², k = 4π/3.
    let (ci, ca) = (c.column(0).to_owned(), c.column(1).to_owned());
    let dip = oneelectron::dipole(&prep_mol, [0.0; 3]).unwrap();
    let d_ia: Vec<f64> = (0..3).map(|x| ci.dot(&dip[x].dot(&ca))).collect();
    let cen = [
        ci.dot(&dip[0].dot(&ci)),
        ci.dot(&dip[1].dot(&ci)),
        ci.dot(&dip[2].dot(&ci)),
    ];
    let sigma2 = ci.dot(&oneelectron::r2_moment(&prep_mol, cen).unwrap().dot(&ci));
    let kk = ov_mol[(0, 0)];
    let dd = eps[1] - eps[0];
    let om = (dd * (dd + 4.0 * kk)).sqrt();
    let de_dd = 0.5 * ((dd + 2.0 * kk) / om - 1.0);
    let de_dk = dd / om - 1.0;
    let k = 4.0 * PI / 3.0;
    let d2: f64 = d_ia.iter().map(|x| x * x).sum();
    let c3_pred = de_dd * k * (sigma2 + d2) + de_dk * (-k * d2);
    eprintln!(
        "E_dRPA(mol) {e_mol:.12e}; sigma2 {sigma2:.6}, |d_ia|^2 {d2:.6}, K {kk:.6}, D {dd:.6}, \
         dE/dD {de_dd:.6}, dE/dK {de_dk:.6} => c3_pred {c3_pred:.8} (prototype {PROTO_C3_PRED})"
    );
    assert!(
        (c3_pred - PROTO_C3_PRED).abs() < 1e-6,
        "moment model {c3_pred}"
    );

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
        let ints = GammaDrpaIntegrals::DenseAft(&eri);
        let q = DEFAULT_GAMMA_DRPA_QUAD_POINTS;
        let rs = drpa(&cell, &rhf, ints, ExxDiv::Ewald, Shifted, q);
        let ru = drpa(&cell, &rhf, ints, ExxDiv::Ewald, Unshifted, q);
        let (ds, du) = (rs.drpa_corr - e_mol, ru.drpa_corr - e_mol);
        // unshifted − shifted vs the molecular shift function (all e_ia − v_M).
        let vm = rs.madelung;
        let shift_fn = drpa_plasmon(&ov_mol, &[eps[0] + vm], &ev).unwrap() - e_mol;
        let ratio = (du - ds) / shift_fn;
        eprintln!(
            "a = {a:4.1}: dE shifted {ds:+.6e} (proto {:+.3e}), unshifted {du:+.6e} (proto {:+.3e}), \
             ratio {ratio:.4} (proto {}), c3/a^3 {:+.6e} ({:.1} s)",
            BOX_SHIFTED[w],
            BOX_UNSHIFTED[w],
            BOX_SHIFT_FN_RATIO[w],
            c3_pred / a.powi(3),
            t0.elapsed().as_secs_f64()
        );
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
        close(ratio, BOX_SHIFT_FN_RATIO[w], 1.5e-3, "shift-function ratio");
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

    // Least squares dE = c3 a^-3 + c5 a^-5 on (16, 20, 24).
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
// (5) cc-pvdz-ri fitting error vs dense (FINDINGS Iteration 4 aux table).
// ===========================================================================

/// `(dE shifted same C, shifted own SCF, unshifted same C, rel shifted same
/// C)`, as printed (3 / 3 / 3 / 2 significant figures).
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
    let dense = GammaDrpaIntegrals::DenseAft(&eri);
    let fit = GammaDrpaIntegrals::RsGdf(&gdf);
    let q = DEFAULT_GAMMA_DRPA_QUAD_POINTS;
    let ex_s = drpa(cell, &rhf0, dense, ExxDiv::None, Shifted, q).drpa_corr;
    let ex_u = drpa(cell, &rhf0, dense, ExxDiv::None, Unshifted, q).drpa_corr;
    let same_s = drpa(cell, &rhf0, fit, ExxDiv::None, Shifted, q).drpa_corr - ex_s;
    let own_s = drpa(cell, &rhf1, fit, ExxDiv::None, Shifted, q).drpa_corr - ex_s;
    let same_u = drpa(cell, &rhf0, fit, ExxDiv::None, Unshifted, q).drpa_corr - ex_u;
    let rel = same_s / ex_s;
    eprintln!(
        "naux kept {}/{}: exact shifted {ex_s:.10e}, unshifted {ex_u:.10e}",
        gdf.stats().naux_kept,
        gdf.stats().naux
    );
    let half = |x: f64, sig: i32| 0.5 * 10f64.powi(x.abs().log10().floor() as i32 - (sig - 1));
    close(
        same_s,
        row.same_s,
        half(row.same_s, 3),
        "dE shifted, same C",
    );
    close(own_s, row.own_s, half(row.own_s, 3), "dE shifted, own SCF");
    close(
        same_u,
        row.same_u,
        half(row.same_u, 3),
        "dE unshifted, same C",
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
            same_s: 1.07e-6,
            own_s: 1.18e-6,
            same_u: 1.51e-6,
            rel: -1.5e-4,
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
            same_s: 6.00e-5,
            own_s: 6.49e-5,
            same_u: 8.16e-5,
            rel: -8.8e-4,
        },
    );
}

// ===========================================================================
// (6) Molecular path: the new ferric-rpa entry IS the molecular pipeline.
// ===========================================================================

#[test]
fn run_pdep_rpa_from_parts_reproduces_the_molecular_path_bitwise() {
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
    // Exactly the config gamma_drpa hands to ferric-rpa.
    let cfg = pdep_config(DEFAULT_GAMMA_DRPA_QUAD_POINTS, AMPLE);
    let inter = compute_rpa_intermediates(
        &mol,
        &obs,
        &dfbs,
        op,
        &rhf,
        &RiMp2Config {
            frozen_core: cfg.frozen_core,
            memory_budget_bytes: cfg.memory_budget_bytes,
            ..Default::default()
        },
    )
    .unwrap();
    let (nocc, nvir) = (inter.nocc, inter.nvir);
    let eps = rhf.eps_r();
    let (eo, ev) = (&eps[..nocc], &eps[nocc..nocc + nvir]);
    let mol_r = run_pdep_rpa_from_intermediates(&inter, &mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let parts = run_pdep_rpa_from_parts(&inter, eo, ev, &cfg).unwrap();
    let full = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    eprintln!(
        "H2/cc-pVDZ dRPA: from_intermediates {:.15e}, from_parts {:.15e}, run_pdep_rpa {:+.2e}",
        mol_r.e_rpa,
        parts.e_rpa,
        full.e_rpa - mol_r.e_rpa
    );
    assert_eq!(parts.e_rpa.to_bits(), mol_r.e_rpa.to_bits());
    assert_eq!(parts.n_eigenpotentials, mol_r.n_eigenpotentials);
    assert_eq!(
        parts
            .eigenvalues_static
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        mol_r
            .eigenvalues_static
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    assert!((full.e_rpa - mol_r.e_rpa).abs() < 1e-12);

    // Config honesty: everything the parts path cannot honour is refused.
    let refuse = |c: &ferric_rpa::PdepRpaConfig, eo: &[f64], ev: &[f64], what: &str| {
        let e = run_pdep_rpa_from_parts(&inter, eo, ev, c)
            .expect_err(what)
            .to_string();
        eprintln!("  refused ({what}): {e}");
    };
    let mut c = cfg.clone();
    c.trunc_thresh = 1e-4;
    refuse(&c, eo, ev, "trunc_thresh > 0");
    let mut c = cfg.clone();
    c.chi0_sparsity = Chi0Sparsity::BoysScreened {
        thresh: 1e-6,
        dist_cutoff: 10.0,
    };
    refuse(&c, eo, ev, "Boys sparsity");
    let mut c = cfg.clone();
    c.eigensolver = Eigensolver::Davidson;
    refuse(&c, eo, ev, "Davidson");
    refuse(&cfg, &eps[..nocc + 1], ev, "wrong occupied count");
    let gapless = vec![ev[0]; nocc];
    refuse(&cfg, &gapless, ev, "zero gap");
}

#[test]
fn gamma_drpa_rejects_bad_inputs() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc = hcore(&cell, &prep);
    let eri = dense_none(&cell, &prep, &hc);
    let rhf = gamma_rhf(&cell, &prep, &hc, &eri);
    let ints = GammaDrpaIntegrals::DenseAft(&eri);
    let q = DEFAULT_GAMMA_DRPA_QUAD_POINTS;
    // frozen_core = nocc (= 1): active_occ refuses.
    let mut fc = drpa_cfg(ExxDiv::None, Shifted, q);
    fc.frozen_core = 1;
    assert!(gamma_drpa(&cell, &rhf, ints, &fc).is_err());
    // Grid outside [8, 1024] is refused, not clamped.
    for bad in [0, 7, 1025] {
        let msg = gamma_drpa(&cell, &rhf, ints, &drpa_cfg(ExxDiv::None, Shifted, bad))
            .unwrap_err()
            .to_string();
        assert!(msg.contains("quad_points"), "{msg}");
    }
    // Unconverged reference.
    let mut bad = rhf.clone();
    bad.converged = false;
    assert!(gamma_drpa(&cell, &bad, ints, &drpa_cfg(ExxDiv::None, Shifted, q)).is_err());
    // A tiny budget names the quantity.
    let msg = gamma_drpa(
        &cell,
        &rhf,
        ints,
        &GammaDrpaConfig {
            budget_bytes: Some(1),
            ..drpa_cfg(ExxDiv::None, Shifted, q)
        },
    )
    .unwrap_err()
    .to_string();
    assert!(msg.contains("Gamma dRPA dense"), "{msg}");
}
