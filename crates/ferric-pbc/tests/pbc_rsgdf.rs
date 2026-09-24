//! Stage-1 PBC step 9: Gamma-point RS-GDF (`ferric_pbc::rsgdf`) against the
//! dense pure-AFT oracle (`DenseAftEri`) and the Python prototype
//! (`reference/pbc/pbc_gdf.py`, FINDINGS "Iteration 2 (Python, RS-GDF)").
//!
//! # Exactness anchor (written first; everything else is measured after it)
//!
//! One primitive s per H (exponent α = 0.5), cubic a = 4. Every lattice pair
//! product `φ_m(r) φ_n(r − L)` is ONE s Gaussian (exponent 2α) at
//! `(A_m + A_n + L)/2`; modulo the lattice those centres fall in the 8
//! half-lattice classes `(A_m + A_n + h·a)/2`, `h ∈ {0,1}³`, per (m,n) type.
//! With those 24 aux s functions the periodic pair densities lie EXACTLY in
//! the aux span, so `BᵀB` must equal the pure-AFT ERI to truncation
//! (prototype: 1.6e-12 at ω = 0.8, 1.2e-11 at ω = 1.2; both exxdiv dE
//! ≤ 8e-12). The aux sit on ghost sites (`SiteBasis`), not on atoms — the
//! builder's aux FT and image enumeration must not assume atom centres.
//!
//! # Artifact hypotheses (stated before measuring)
//!
//! * "The anchor passes because G = 0 errors cancel by accident" vs "the
//!   convention is consistent": if consistent, the anchor is exact at EVERY ω
//!   AND the one-sided G = 0 mutants fail by O(1e-2..1)
//!   (`g0_mutants_fail_the_anchor`, prototype 5.3 / 6.1e-2).
//! * "The anchor is vacuous (would pass a broken build)": an INCOMPLETE aux
//!   (only the h = 0 class) must fail it (prototype 6.9e-4).
//! * "The fitting error is SR/LR truncation, not fitting": then dE would move
//!   with ω; it must not (`fitted_eri_is_omega_independent`, prototype < 1e-9).
//!
//! # Mutation plan (for the main agent)
//!
//! * One-sided G = 0 — covered IN-TEST by `G0Handling::{MetricOnlyMutant,
//!   NeitherMutant}` (the negative controls run every time). To mutate the
//!   production code instead: in `rsgdf::subtract_g0` delete the `do_three`
//!   branch → the anchor, H2 and ω-independence tests fail.
//! * aux FT phase `e^{+iG·A}` (flip the sign of `-mag * ph.sin()` in
//!   `aux_ft_shells`) → anchor fails (prototype 7.1e-2).
//! * SR aux images truncated to T = 0 (make `LatticeWalker::visit` call `f`
//!   only for `n = 0`) → anchor fails (prototype 2.9e-2).
//! * K contracted as J (`general_mat_mul(.., &tmp, &bk.t(), ..)` → use
//!   `bk.dot(&d)` scalar pattern) → every energy test fails.
//! * Madelung term dropped in `RsGdfK` → the ewald rows fail.
//! * lindep cut ignored (keep every eigenvalue) → `lindep_drops_are_reported`
//!   fails (a duplicated aux function makes the metric exactly singular:
//!   NaN/inf in B or a wrong drop count).

mod common;

use common::*;
use ferric_core::basis;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::site_basis::SiteBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::{G0Handling, RsGdf, RsGdfConfig, DEFAULT_FITTED_ERI_MAX_BYTES};

/// ω for the nuclear-attraction split (as in pbc_dense_aft_scf.rs).
const HCORE_OMEGA: f64 = 0.8;
const ANCHOR_ALPHA: f64 = 0.5;
const AMPLE: usize = 1 << 30;

/// Prototype fitting errors `E_GDF − E_exact` (Hartree), w = 1, spherical
/// aux read from ferric's bundled JSON (`pbc_gdf.ferric_basis`), PySCF-digit
/// STO-3G / SP_BASIS, measured 2026-09-23 with
/// `reference/pbc/pbc_gdf.py` (full precision of FINDINGS' table).
/// `(aux, dE_none, dE_ewald, naux)`.
const H2_PROTO_DE: [(&str, f64, f64, usize); 2] = [
    ("cc-pvdz-ri", -1.974228505452e-06, -1.974228505564e-06, 28),
    (
        "def2-universal-jkfit",
        -4.472349374840e-06,
        -4.472349374840e-06,
        36,
    ),
];
const TRI_PROTO_DE: [(&str, f64, f64, usize); 2] = [
    ("cc-pvdz-ri", -2.662132833464e-05, -2.662132833731e-05, 56),
    (
        "def2-universal-jkfit",
        -1.722310277907e-05,
        -1.722310277819e-05,
        72,
    ),
];
/// The prototype's own ω-scatter of dE (0.7/1.0/1.4) is ≤ 5e-11 (triclinic
/// jkfit); both codes' SCF energies agree with PySCF AFTDF to < 1e-9.
const PROTO_DE_TOL: f64 = 1e-9;

fn hcore(cell: &Cell, prep: &PreparedBasis) -> PeriodicHcore {
    periodic_hcore(cell, prep, &PeriodicHcoreConfig::with_omega(HCORE_OMEGA)).expect("hcore")
}

fn dense(cell: &Cell, prep: &PreparedBasis, hc: &PeriodicHcore) -> DenseAftEri {
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

fn cfg(omega: f64) -> RsGdfConfig {
    RsGdfConfig {
        omega,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

/// The 24 anchor aux centres (`test_prototype._anchor_cell_and_aux`),
/// restricted to the half-lattice classes `k ∈ classes` (np.ndindex order).
fn anchor_sites(cell: &Cell, classes: &[usize]) -> Vec<[f64; 4]> {
    let r = cell.positions();
    let a = cell.lattice();
    let mut out = Vec::new();
    for (i, j) in [(0usize, 0usize), (1, 1), (0, 1)] {
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
    out
}

struct Anchor {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
    eri: DenseAftEri,
}

fn anchor() -> Anchor {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    let hc = hcore(&cell, &prep);
    let eri = dense(&cell, &prep, &hc);
    Anchor {
        cell,
        prep,
        hc,
        eri,
    }
}

fn fitted_vs_dense(gdf: &RsGdf, eri: &DenseAftEri) -> f64 {
    max_abs_diff(
        &gdf.fitted_eri(DEFAULT_FITTED_ERI_MAX_BYTES).unwrap(),
        eri.eri(),
    )
}

/// `E_gdf − E_dense` for `[none, ewald]`.
fn de_both(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    gdf: &RsGdf,
) -> [f64; 2] {
    let mut out = [0.0; 2];
    for (slot, exx) in [ExxDiv::None, ExxDiv::Ewald].into_iter().enumerate() {
        let e_ref = eri.clone().with_exxdiv(cell, exx).unwrap();
        let g = gdf.clone().with_exxdiv(cell, exx).unwrap();
        let e0 = gamma_rhf(cell, prep, hc, &e_ref).energy;
        let e1 = gamma_rhf_jk(
            cell,
            prep,
            hc,
            Box::new(g.j_builder()),
            Box::new(g.k_builder()),
        )
        .energy;
        eprintln!(
            "  {exx:?}: E_dense = {e0:.12}, E_gdf = {e1:.12}, dE = {:.12e}",
            e1 - e0
        );
        out[slot] = e1 - e0;
    }
    out
}

#[test]
fn rsgdf_matches_dense_aft_in_the_trivial_limit() {
    let an = anchor();
    let site = SiteBasis::new(&anchor_sites(&an.cell, &[0, 1, 2, 3, 4, 5, 6, 7]), 0).unwrap();
    assert_eq!(site.prep.nbasis(), 24);
    // ω = 1 would hide a 1/ω vs 1/ω² slip in the G = 0 term.
    for omega in [0.8, 1.2] {
        let gdf = RsGdf::build(&an.cell, &an.prep, &site.prep, &an.hc.s, &cfg(omega)).unwrap();
        let st = gdf.stats();
        let d = fitted_vs_dense(&gdf, &an.eri);
        eprintln!(
            "anchor ω={omega}: max|I_fit − I| = {d:.2e}; kept {}/{} (eig {:.2e}..{:.2e}), \
             {} SR3 triplets, {} SR2 pairs, {} pair images, {} half-G, asym J2 {:.1e} J3 {:.1e}",
            st.naux_kept,
            st.naux,
            st.metric_eig_min,
            st.metric_eig_max,
            st.n_sr3_triplets,
            st.n_sr2_pairs,
            st.n_pair_images,
            st.n_g_half,
            st.asym_j2,
            st.asym_j3
        );
        assert!(d < 1e-10, "ω={omega}: fitted ERI off by {d:.3e}");
        let de = de_both(&an.cell, &an.prep, &an.hc, &an.eri, &gdf);
        for x in de {
            assert!(x.abs() < 1e-10, "ω={omega}: dE {x:.3e}");
        }
    }
}

#[test]
fn g0_mutants_and_incomplete_aux_fail_the_anchor() {
    let an = anchor();
    let full = SiteBasis::new(&anchor_sites(&an.cell, &[0, 1, 2, 3, 4, 5, 6, 7]), 0).unwrap();
    for (mode, floor) in [
        (G0Handling::MetricOnlyMutant, 1e-2), // prototype 5.3
        (G0Handling::NeitherMutant, 1e-3),    // prototype 6.1e-2
    ] {
        let c = RsGdfConfig {
            g0: mode,
            ..cfg(0.8)
        };
        let gdf = RsGdf::build(&an.cell, &an.prep, &full.prep, &an.hc.s, &c).unwrap();
        let d = fitted_vs_dense(&gdf, &an.eri);
        eprintln!("{mode:?}: max|ΔI| = {d:.3e}");
        assert!(d > floor, "{mode:?} is not detected by the anchor: {d:.3e}");
    }
    // Incomplete span: only the h = 0 class (3 aux).
    let h0 = SiteBasis::new(&anchor_sites(&an.cell, &[0]), 0).unwrap();
    let gdf = RsGdf::build(&an.cell, &an.prep, &h0.prep, &an.hc.s, &cfg(0.8)).unwrap();
    let d = fitted_vs_dense(&gdf, &an.eri);
    eprintln!("aux h=0 class only (3 aux): max|ΔI| = {d:.3e}");
    assert!(d > 1e-5, "incomplete aux not detected: {d:.3e}");
}

#[test]
fn lindep_drops_are_reported_and_harmless() {
    let an = anchor();
    let base_sites = anchor_sites(&an.cell, &[0, 1, 2, 3, 4, 5, 6, 7]);
    let base = SiteBasis::new(&base_sites, 0).unwrap();
    let g0 = RsGdf::build(&an.cell, &an.prep, &base.prep, &an.hc.s, &cfg(0.8)).unwrap();
    // Duplicate one aux function: the metric becomes exactly singular.
    let mut dup_sites = base_sites.clone();
    dup_sites.push(base_sites[5]);
    let dup = SiteBasis::new(&dup_sites, 0).unwrap();
    let g1 = RsGdf::build(&an.cell, &an.prep, &dup.prep, &an.hc.s, &cfg(0.8)).unwrap();
    let (s0, s1) = (g0.stats(), g1.stats());
    eprintln!(
        "lindep: base dropped {}/{} (min eig {:.2e}); duplicated dropped {}/{} (min eig {:.2e})",
        s0.n_dropped, s0.naux, s0.metric_eig_min, s1.n_dropped, s1.naux, s1.metric_eig_min
    );
    assert_eq!(s1.naux, s0.naux + 1);
    assert_eq!(
        s1.n_dropped,
        s0.n_dropped + 1,
        "the exact duplicate must be dropped and counted"
    );
    assert_eq!(s1.naux_kept + s1.n_dropped, s1.naux);
    assert_eq!(g1.b().nrows(), s1.naux_kept);
    assert!(g1.b().iter().all(|x| x.is_finite()));
    let d = fitted_vs_dense(&g1, &an.eri);
    assert!(d < 1e-10, "duplicated aux changed the exact fit: {d:.3e}");
}

#[test]
fn sr_screen_matches_the_global_image_radius() {
    let an = anchor();
    let site = SiteBasis::new(&anchor_sites(&an.cell, &[0, 1, 2, 3, 4, 5, 6, 7]), 0).unwrap();
    let screened = RsGdf::build(&an.cell, &an.prep, &site.prep, &an.hc.s, &cfg(1.0)).unwrap();
    let global = RsGdf::build(
        &an.cell,
        &an.prep,
        &site.prep,
        &an.hc.s,
        &RsGdfConfig {
            sr_screen: false,
            ..cfg(1.0)
        },
    )
    .unwrap();
    let a = screened.fitted_eri(DEFAULT_FITTED_ERI_MAX_BYTES).unwrap();
    let b = global.fitted_eri(DEFAULT_FITTED_ERI_MAX_BYTES).unwrap();
    let d = max_abs_diff(&a, &b);
    let (ns, ng) = (
        screened.stats().n_sr3_triplets,
        global.stats().n_sr3_triplets,
    );
    eprintln!("SR3 triplets screened {ns} vs global {ng}; max|ΔI| = {d:.2e}");
    assert!(ns < ng, "the per-triplet screen skipped nothing");
    // Skipped triplets are each < precision (1e-13); the metric solve can
    // amplify their sum, so hold it to a tenth of the anchor tolerance.
    assert!(d < 1e-11, "screen error {d:.3e}");
}

fn fitting_error_case(cell: &Cell, obs: &PreparedBasis, table: &[(&str, f64, f64, usize)]) {
    let hc = hcore(cell, obs);
    let eri = dense(cell, obs, &hc);
    for &(name, de_none, de_ewald, naux) in table {
        let aux = PreparedBasis::new(cell.mol(), &basis::bundled(name).unwrap()).unwrap();
        assert_eq!(
            aux.nbasis(),
            naux,
            "{name}: aux count differs from the prototype's spherical set"
        );
        let gdf = RsGdf::build(cell, obs, &aux, &hc.s, &cfg(1.0)).unwrap();
        let st = gdf.stats();
        eprintln!(
            "{name}: kept {}/{} (eig {:.2e}..{:.2e}), {} SR3 triplets, {} pair images, {} half-G",
            st.naux_kept,
            st.naux,
            st.metric_eig_min,
            st.metric_eig_max,
            st.n_sr3_triplets,
            st.n_pair_images,
            st.n_g_half
        );
        let de = de_both(cell, obs, &hc, &eri, &gdf);
        for (got, want, lab) in [(de[0], de_none, "none"), (de[1], de_ewald, "ewald")] {
            eprintln!(
                "  {name} {lab}: dE = {got:.12e} (prototype {want:.12e}, diff {:.2e})",
                got - want
            );
            assert!(
                (got - want).abs() < PROTO_DE_TOL,
                "{name} {lab}: dE {got:.6e} vs prototype {want:.6e}"
            );
        }
        // The Madelung term is v_M S D S: fit-independent at Gamma.
        assert!((de[0] - de[1]).abs() < 1e-10, "{name}: exxdiv changed dE");
    }
}

#[test]
fn h2_sto3g_fitting_error_matches_prototype() {
    let cell = h2_cell(4.0);
    let obs = prep_for(&cell, &pyscf_sto3g_h());
    fitting_error_case(&cell, &obs, &H2_PROTO_DE);
}

#[test]
#[ignore = "slow: triclinic 4H s+p RS-GDF with two aux sets plus the dense-AFT oracle and four \
            SCFs (estimated minutes in debug); run with --release -- --ignored, serially, on a quiet box"]
fn triclinic_4h_sp_fitting_error_matches_prototype() {
    let cell = triclinic_cell();
    let obs = prep_for(&cell, &sp_basis_h());
    fitting_error_case(&cell, &obs, &TRI_PROTO_DE);
}

#[test]
fn fitted_eri_is_omega_independent() {
    // Prototype test_gdf_is_independent_of_ewald_split_parameter: < 1e-9
    // (H2/STO-3G a=4, cc-pvdz-ri with p and d shells; prototype dE scatter
    // over ω = 0.7/1.0/1.4 was 1.6e-13).
    let cell = h2_cell(4.0);
    let obs = prep_for(&cell, &pyscf_sto3g_h());
    let hc = hcore(&cell, &obs);
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let i: Vec<_> = [0.7, 1.4]
        .iter()
        .map(|&w| {
            RsGdf::build(&cell, &obs, &aux, &hc.s, &cfg(w))
                .unwrap()
                .fitted_eri(DEFAULT_FITTED_ERI_MAX_BYTES)
                .unwrap()
        })
        .collect();
    let d = max_abs_diff(&i[0], &i[1]);
    eprintln!("cc-pvdz-ri: max|I(ω=0.7) − I(ω=1.4)| = {d:.2e}");
    assert!(d < 1e-9, "{d:.3e}");
}

#[test]
fn rsgdf_memory_gates_name_the_quantity_and_bytes() {
    let an = anchor();
    let site = SiteBasis::new(&anchor_sites(&an.cell, &[0, 1, 2, 3, 4, 5, 6, 7]), 0).unwrap();
    let (n2, naux) = (4usize, 24usize);
    let build = |b: usize| {
        RsGdf::build(
            &an.cell,
            &an.prep,
            &site.prep,
            &an.hc.s,
            &RsGdfConfig {
                budget_bytes: Some(b),
                ..cfg(0.8)
            },
        )
    };
    let msg = build(1).unwrap_err().to_string();
    assert!(
        msg.contains("RsGdf n×n matrices") && msg.contains(&format!("({} bytes", n2 * 32)),
        "{msg}"
    );
    // Enough for the matrices, not for J3 + B + temporary: the gates compose.
    let msg = build(n2 * 32 + 100).unwrap_err().to_string();
    assert!(
        msg.contains("RsGdf 3-index J3") && msg.contains(&format!("({} bytes", n2 * naux * 24)),
        "{msg}"
    );

    // Chunk invariance: a budget just above the resident set forces many LR
    // chunks; only the GEMM summation order changes.
    let ample = build(AMPLE).unwrap();
    let resident = ample.stats().resident_bytes;
    let tight = build(resident + (64 << 10)).unwrap();
    let (c1, c2) = (ample.stats().n_g_chunks, tight.stats().n_g_chunks);
    let d = max_abs_diff(
        &ample.fitted_eri(DEFAULT_FITTED_ERI_MAX_BYTES).unwrap(),
        &tight.fitted_eri(DEFAULT_FITTED_ERI_MAX_BYTES).unwrap(),
    );
    eprintln!("LR chunks {c1} (ample) vs {c2} (tight); max|ΔI| = {d:.2e}");
    assert_eq!(c1, 1);
    assert!(c2 > 1, "tight budget did not force chunking");
    // GEMM reassociation only (~1e-16 relative in J2/J3), amplified by the
    // metric's conditioning; a tenth of the anchor tolerance.
    assert!(d < 1e-11, "{d:.3e}");
}

#[test]
fn rsgdf_rejects_bad_inputs() {
    let an = anchor();
    let site = SiteBasis::new(&anchor_sites(&an.cell, &[0]), 0).unwrap();
    for bad in [
        RsGdfConfig {
            omega: 0.0,
            ..cfg(1.0)
        },
        RsGdfConfig {
            omega: f64::NAN,
            ..cfg(1.0)
        },
        RsGdfConfig {
            precision: 1.0,
            ..cfg(1.0)
        },
        RsGdfConfig {
            lindep: -1.0,
            ..cfg(1.0)
        },
    ] {
        assert!(
            RsGdf::build(&an.cell, &an.prep, &site.prep, &an.hc.s, &bad).is_err(),
            "{bad:?}"
        );
    }
    let wrong_s = ndarray::Array2::<f64>::zeros((3, 3));
    assert!(RsGdf::build(&an.cell, &an.prep, &site.prep, &wrong_s, &cfg(1.0)).is_err());
    // The orbital basis must sit on the cell's atoms.
    assert!(RsGdf::build(&an.cell, &site.prep, &site.prep, &an.hc.s, &cfg(1.0)).is_err());
}
