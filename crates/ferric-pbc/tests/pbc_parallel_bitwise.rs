//! The pair-parallel SR walks (FINDINGS "Performance plan (research) —
//! 2026-09-25" §3 Class A: RS-GDF SR 3-centre `J3` and the hcore SR nuclear
//! attraction `V_SR`) are BIT-IDENTICAL across rayon thread counts AND to the
//! serial loops they replaced.
//!
//! Construction under test: the shell-pair loop is outermost and runs in
//! parallel; each pair accumulates its own block from zero in exactly the
//! serial per-element order ("L ascending, then T / nucleus candidate in
//! walk order") and COPIES it into the output. What each test guards:
//!
//! * `rsgdf_sr_sums_are_bitwise_across_threads_and_vs_serial` — J3 (and J2)
//!   at 1/2/6 threads agree bit for bit, AND the parallel Gamma J3 equals the
//!   SERIAL `L → i1 → i2 → P → T` walk (the k-point residue-binned walk at one
//!   bin, which still runs the old serial nest) bit for bit. A pair-outer nest
//!   that reordered an element's addends (e.g. P-outer over L, or a per-thread
//!   partial sum folded at the end) fails the serial comparison even at one
//!   thread; a shared-accumulation race or a thread-count-dependent split
//!   fails the cross-thread one.
//! * `hcore_sr_attraction_is_bitwise_across_threads_and_vs_serial` — the same
//!   for `V_SR` against a FROZEN copy of the pre-parallel serial loop, both
//!   production-screened and with the predicted-skip tracking of the SR
//!   screening study.
//! * `rsgdf_rhf_energy_is_bitwise_across_threads` — `h`, B and the RS-GDF RHF
//!   energy at 1/2/6 threads agree bit for bit, and the energy keeps the
//!   pinned RS-GDF − dense-AFT fitting error of `pbc_rsgdf.rs`
//!   (−1.974228505452e-06 Ha, cc-pvdz-ri, exxdiv none).
//!
//! * `krsgdf_sr_bins_are_bitwise_across_threads_and_vs_serial` — the
//!   k-point residue-binned SR metric and 3-centre bins (the
//!   `(pair, r_L)`-parallel walk) at 1/2/6 threads agree bit for bit, and
//!   equal the FROZEN serial binned walk bit for bit, on a Monkhorst-Pack
//!   mesh whose pair-image moduli (`2N`) differ from the aux-image moduli
//!   (`N`), so a bin mix-up in either dimension fails.
//! * `hcore_kpts_sr_attraction_is_bitwise_across_threads_and_vs_serial` —
//!   `V_SR(k)` against a FROZEN copy of the serial k-point loop, and the full
//!   `h(k)` across thread counts.
//! * `krsgdf_blocks_and_krhf_energy_are_bitwise_across_threads` — every
//!   `B(k, k')` of `KRsGdf`, `h(k)` and the k-point RS-GDF RHF energy at
//!   1/2/6 threads agree bit for bit (1×1×2, all TRIM, and 1×1×3, with a
//!   mirrored q pair); the 1×1×2 energy keeps its PySCF-backed pin of
//!   `pbc_krsgdf.rs` (`TRI_112_JKFIT`, exxdiv none).
//!
//! Reachability: the J3/V systems have ≥ 64 ordered shell pairs (≥ 10 per
//! thread at 6 threads, asserted), so the parallel split really binds.
//!
//! The force / stress DERIVATIVE walks add every triplet into SHARED rows
//! (atom rows, aux rows, the 3×3 stress), so no element belongs to one pair
//! task; they are ordered-parallel instead (`ferric_pbc::ordered`: the
//! integrals and per-triplet from-zero subtotals in parallel over a window
//! of units, every `+=` into a shared row replayed serially in the serial
//! walk order). That is BIT-IDENTICAL to the serial walks, and these tests
//! assert it against the FROZEN pre-parallel walks (the `SerialDerivWalks`
//! / `SerialImages` test mutations), at 1/2/6 threads, with the serial
//! oracle itself run at 1 and 6 threads:
//!
//! * `gamma_rhf_rsgdf_force_…` — SR attraction force visitor, RS-GDF SR
//!   3-centre and SR metric force walks (triclinic 4 H s+p, cc-pvdz-ri).
//! * `gamma_uks_rsgdf_force_…` — the same on an open-shell hybrid (H3 UKS
//!   PBE0).
//! * `gamma_rhf_rsgdf_stress_…` — SR attraction strain visitor, RS-GDF SR
//!   3-centre and metric strain walks.
//! * `kpoint_rhf_rsgdf_force_…` — the L-weighted SR attraction visitor and
//!   the phase-folded k-point SR 3-centre / metric walks (H2 1×1×3).
//! * `ecp_force_term_…` — the periodic-ECP image loop (compact H–I,
//!   LANL2DZ), fixed density.
//!
//! A test of this kind only proves anything if the oracle and the parallel
//! path are different code: the oracles are verbatim copies of the pre-
//! parallel loops (`*_serial` functions), selected by the mutation.

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet, Shell};
use ferric_core::ecp::{EcpDef, EcpShell, EcpTerm};
use ferric_core::mol::{Atom, Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{gamma_uks, GammaUksConfig, PeriodicGridConfig};
use ferric_pbc::ecp::{
    periodic_ecp_gradient_with, EcpGradMutation, PeriodicEcpConfig, PeriodicEcpGradient,
};
use ferric_pbc::grad::{
    gamma_rhf_gradient_rsgdf, gamma_uks_gradient_rsgdf, GammaGradConfig, GammaGradient,
    GradMutation, RsGdfGradSource,
};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, sr_attraction_kpts_parallel_and_serial};
use ferric_pbc::hcore::{
    periodic_hcore, sr_attraction_parallel_and_serial, PeriodicHcore, PeriodicHcoreConfig,
    SrAttractionParts, SrBound, DEFAULT_HCORE_PRECISION,
};
use ferric_pbc::kgrad::{
    kpoint_rhf_gradient, KGradConfig, KGradJk, KGradMutation, KGradient, KRsGdfGradSource,
};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{
    solve_krhf, solve_krhf_injected, KJkKind, KPointInjection, KRhfConfig, KScfConfig,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::pair_ft::residues::{
    pair_ft_residues_chunked_serial_oracle, pair_ft_residues_chunked_timed,
};
use ferric_pbc::pair_ft::{
    pair_ft_bytes_per_g, pair_ft_chunked_serial_oracle, pair_ft_chunked_timed, pair_ft_with_thresh,
    CTR_LARGEST_GROUP, CTR_PARTIAL_WINDOW, CTR_SITES, CTR_SURVIVORS, DEFAULT_PAIR_FT_THRESH,
};
use ferric_pbc::rsgdf::kpoint::{
    sr_bins_parallel_and_serial, sr_sums_gamma_and_single_bin, KRsGdf, KRsGdfConfig, SrBinsParts,
};
use ferric_pbc::rsgdf::{lr_sums_parallel_and_serial, RangeSplit, RsGdf, RsGdfConfig};
use ferric_pbc::stress::{gamma_rhf_stress_rsgdf, GammaStress, GammaStressConfig, StressMutation};
use ferric_pbc::timing::PbcTimings;
use ferric_pbc::uhf::GammaUhfIntegrals;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::collections::HashMap;

const THREADS: [usize; 3] = [1, 2, 6];
const HCORE_OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 30;
/// `pbc_rsgdf.rs` H2_PROTO_DE (cc-pvdz-ri, exxdiv none).
const H2_RI_DE_NONE: f64 = -1.974228505452e-06;
/// `pbc_krsgdf.rs` TRI_112_JKFIT (exxdiv none) and its tolerance.
const TRI_112_JKFIT_NONE: f64 = -1.5876606977729906;
const PIN_TOL: f64 = 1e-9;

fn in_pool<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("rayon pool")
        .install(f)
}

fn gdf_cfg() -> RsGdfConfig {
    RsGdfConfig {
        omega: 1.0,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

/// Number of elements whose bits differ (dims asserted equal).
fn bit_diffs(a: &Array2<f64>, b: &Array2<f64>) -> usize {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn assert_bitwise(a: &Array2<f64>, b: &Array2<f64>, what: &str) {
    let d = bit_diffs(a, b);
    assert_eq!(d, 0, "{what}: {d} of {} elements differ in bits", a.len());
}

/// Number of complex elements whose bits (re or im) differ.
fn cbit_diffs(a: &Array2<Complex64>, b: &Array2<Complex64>) -> usize {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.re.to_bits() != y.re.to_bits() || x.im.to_bits() != y.im.to_bits())
        .count()
}

fn assert_cbitwise(a: &[Array2<Complex64>], b: &[Array2<Complex64>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: matrix counts");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        let d = cbit_diffs(x, y);
        assert_eq!(
            d,
            0,
            "{what}[{k}]: {d} of {} elements differ in bits",
            x.len()
        );
    }
}

fn assert_bins_bitwise(a: &[Array2<f64>], b: &[Array2<f64>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: bin counts");
    for (r, (x, y)) in a.iter().zip(b).enumerate() {
        assert_bitwise(x, y, &format!("{what} bin {r}"));
    }
}

/// Bins holding at least one non-zero element.
fn live_bins(bins: &[Array2<f64>]) -> usize {
    bins.iter().filter(|m| m.iter().any(|x| *x != 0.0)).count()
}

/// ≥ 10 ordered shell pairs per thread at the largest pool.
fn assert_split_binds(prep: &PreparedBasis) {
    let pairs = prep.nshells() * prep.nshells();
    let tmax = *THREADS.iter().max().unwrap();
    assert!(
        pairs >= 10 * tmax,
        "only {pairs} shell pairs: the {tmax}-thread split would not bind"
    );
}

#[test]
fn rsgdf_sr_sums_are_bitwise_across_threads_and_vs_serial() {
    // 4 H, STO-3G s + p: 8 shells, 64 ordered pairs with 1×1, 1×3, 3×3
    // blocks; jkfit aux carries s/p/d on H.
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    let aux = prep_for(&cell, &basis::bundled("def2-universal-jkfit").unwrap());
    assert_split_binds(&prep);
    let cfg = gdf_cfg();
    let runs: Vec<[(Array2<f64>, Array2<f64>); 2]> = THREADS
        .iter()
        .map(|&n| {
            in_pool(n, || {
                sr_sums_gamma_and_single_bin(&cell, &prep, &aux, &cfg).expect("SR sums")
            })
        })
        .collect();
    let [(j2_ref, j3_ref), _] = &runs[0];
    assert!(j3_ref.iter().any(|x| *x != 0.0), "J3: vacuous");
    for (&n, [(j2g, j3g), (j2b, j3b)]) in THREADS.iter().zip(&runs) {
        // Parallel Gamma J3 vs the serial L-outer walk, same thread count.
        assert_bitwise(
            j3g,
            j3b,
            &format!("J3 parallel vs serial walk ({n} threads)"),
        );
        assert_bitwise(j2g, j2b, &format!("J2 vs binned walk ({n} threads)"));
        // Across thread counts.
        assert_bitwise(j3g, j3_ref, &format!("J3 at {n} vs 1 thread"));
        assert_bitwise(j2g, j2_ref, &format!("J2 at {n} vs 1 thread"));
    }
}

#[test]
fn hcore_sr_attraction_is_bitwise_across_threads_and_vs_serial() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    assert_split_binds(&prep);
    // (cand, screen, track): production; screened at a looser threshold with
    // the predicted-skip tracking (the SR screening study path).
    let cases = [
        (DEFAULT_HCORE_PRECISION, DEFAULT_HCORE_PRECISION, false),
        (DEFAULT_HCORE_PRECISION, 1e-8, true),
    ];
    for (cand, screen, track) in cases {
        let runs: Vec<[SrAttractionParts; 2]> = THREADS
            .iter()
            .map(|&n| {
                in_pool(n, || {
                    sr_attraction_parallel_and_serial(
                        &cell,
                        &prep,
                        HCORE_OMEGA,
                        cand,
                        screen,
                        SrBound::Derived,
                        track,
                    )
                    .expect("SR attraction")
                })
            })
            .collect();
        let tag = format!("screen {screen:e}, track {track}");
        let (v_ref, nt_ref, ns_ref, p_ref) = &runs[0][0];
        assert!(v_ref.iter().any(|x| *x != 0.0), "{tag}: V_SR vacuous");
        assert!(*nt_ref > 0);
        if track {
            let p = p_ref.as_ref().expect("tracked");
            assert!(
                p.iter().any(|x| *x > 0.0),
                "{tag}: nothing skipped, so the tracking path is vacuous"
            );
        }
        for (&n, [par, ser]) in THREADS.iter().zip(&runs) {
            let (v, nt, ns, pred) = par;
            let (vs, nts, nss, preds) = ser;
            assert_bitwise(
                v,
                vs,
                &format!("{tag}: V_SR parallel vs serial ({n} threads)"),
            );
            assert_bitwise(v, v_ref, &format!("{tag}: V_SR at {n} vs 1 thread"));
            assert_eq!((nt, ns), (nts, nss), "{tag}: counters vs serial");
            assert_eq!((nt, ns), (nt_ref, ns_ref), "{tag}: counters vs 1 thread");
            match (pred, preds, p_ref) {
                (None, None, None) => assert!(!track),
                (Some(p), Some(ps), Some(p1)) => {
                    assert_bitwise(p, ps, &format!("{tag}: predicted vs serial ({n} threads)"));
                    assert_bitwise(p, p1, &format!("{tag}: predicted at {n} vs 1 thread"));
                }
                _ => panic!("{tag}: predicted presence differs"),
            }
        }
    }

    // The full Gamma hcore (V_SR symmetrised, V, h) across thread counts.
    let hcs: Vec<_> = THREADS
        .iter()
        .map(|&n| {
            in_pool(n, || {
                periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(HCORE_OMEGA))
                    .expect("hcore")
            })
        })
        .collect();
    for (&n, hc) in THREADS.iter().zip(&hcs) {
        assert_bitwise(
            &hc.v_sr,
            &hcs[0].v_sr,
            &format!("hcore V_SR at {n} vs 1 thread"),
        );
        assert_bitwise(&hc.h, &hcs[0].h, &format!("hcore h at {n} vs 1 thread"));
        assert_eq!(hc.n_sr_triplets, hcs[0].n_sr_triplets);
    }
}

#[test]
fn rsgdf_rhf_energy_is_bitwise_across_threads() {
    let cell = h2_cell(4.0);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = prep_for(&cell, &basis::bundled("cc-pvdz-ri").unwrap());
    let cfg = gdf_cfg();
    let runs: Vec<(Array2<f64>, Array2<f64>, f64)> = THREADS
        .iter()
        .map(|&n| {
            in_pool(n, || {
                let hc =
                    periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(HCORE_OMEGA))
                        .expect("hcore");
                let gdf = RsGdf::build(&cell, &prep, &aux, &hc.s, &cfg).expect("rsgdf");
                let e = gamma_rhf_jk(
                    &cell,
                    &prep,
                    &hc,
                    Box::new(gdf.j_builder()),
                    Box::new(gdf.k_builder()),
                )
                .energy;
                (hc.h.clone(), gdf.b().clone(), e)
            })
        })
        .collect();
    let (h1, b1, e1) = &runs[0];
    for (&n, (h, b, e)) in THREADS.iter().zip(&runs) {
        assert_bitwise(h, h1, &format!("h at {n} vs 1 thread"));
        assert_bitwise(b, b1, &format!("B at {n} vs 1 thread"));
        // h and B are bitwise equal here, so a difference below would come
        // from the SCF driver itself, not from the SR walks.
        assert_eq!(
            e.to_bits(),
            e1.to_bits(),
            "RS-GDF RHF energy at {n} threads {e:.17e} vs 1 thread {e1:.17e}"
        );
    }

    // Nothing moved against the pinned fitting error.
    let hc =
        periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(HCORE_OMEGA)).expect("hcore");
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("dense AFT");
    let e_dense = gamma_rhf(&cell, &prep, &hc, &eri).energy;
    let de = e1 - e_dense;
    assert!(
        (de - H2_RI_DE_NONE).abs() < 1e-9,
        "RS-GDF fitting error {de:e} vs pinned {H2_RI_DE_NONE:e}"
    );
}

#[test]
fn krsgdf_sr_bins_are_bitwise_across_threads_and_vs_serial() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    let aux = prep_for(&cell, &basis::bundled("def2-universal-jkfit").unwrap());
    assert_split_binds(&prep);
    // Monkhorst-Pack, even N: pair images binned mod 2N = [1, 4, 4]
    // (R_L = 16), aux images mod N = [1, 2, 2] (R_T = 4).
    let mesh = KPointMesh::monkhorst_pack(&cell, [1, 2, 2]).unwrap();
    assert_eq!(mesh.residue_moduli(), [1, 4, 4]);
    let cfg = gdf_cfg();
    let runs: Vec<[SrBinsParts; 2]> = THREADS
        .iter()
        .map(|&n| {
            in_pool(n, || {
                sr_bins_parallel_and_serial(&cell, &prep, &aux, &mesh, &cfg).expect("SR bins")
            })
        })
        .collect();
    let (j2_ref, j3_ref, n2_ref, n3_ref) = &runs[0][0];
    assert_eq!((j2_ref.len(), j3_ref.len()), (4, 64));
    // Several bins in BOTH dimensions carry sums, so the bin index is live.
    assert!(
        live_bins(j2_ref) >= 2,
        "J2: only {} live bins",
        live_bins(j2_ref)
    );
    assert!(
        live_bins(j3_ref) >= 8,
        "J3: only {} live bins",
        live_bins(j3_ref)
    );
    for (&n, [par, ser]) in THREADS.iter().zip(&runs) {
        let (j2p, j3p, n2p, n3p) = par;
        let (j2s, j3s, n2s, n3s) = ser;
        assert_bins_bitwise(j2p, j2s, &format!("J2 parallel vs serial ({n} threads)"));
        assert_bins_bitwise(j3p, j3s, &format!("J3 parallel vs serial ({n} threads)"));
        assert_bins_bitwise(j2p, j2_ref, &format!("J2 at {n} vs 1 thread"));
        assert_bins_bitwise(j3p, j3_ref, &format!("J3 at {n} vs 1 thread"));
        assert_eq!((n2p, n3p), (n2s, n3s), "counts vs serial ({n} threads)");
        assert_eq!((n2p, n3p), (n2_ref, n3_ref), "counts at {n} vs 1 thread");
    }
}

#[test]
fn hcore_kpts_sr_attraction_is_bitwise_across_threads_and_vs_serial() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    assert_split_binds(&prep);
    // Complex phases on two axes (MP, even N).
    let mesh = KPointMesh::monkhorst_pack(&cell, [1, 2, 2]).unwrap();
    let cfg = PeriodicHcoreConfig::with_omega(HCORE_OMEGA);
    type VsrRuns = [(Vec<Array2<Complex64>>, usize); 2];
    let runs: Vec<VsrRuns> = THREADS
        .iter()
        .map(|&n| {
            in_pool(n, || {
                sr_attraction_kpts_parallel_and_serial(&cell, &prep, &mesh, &cfg).expect("V_SR(k)")
            })
        })
        .collect();
    let (v_ref, nt_ref) = &runs[0][0];
    assert_eq!(v_ref.len(), 4);
    assert!(*nt_ref > 0);
    assert!(
        v_ref.iter().any(|m| m.iter().any(|z| z.im != 0.0)),
        "V_SR(k): no imaginary part, so the phases are vacuous"
    );
    for (&n, [(v, nt), (vs, nts)]) in THREADS.iter().zip(&runs) {
        assert_cbitwise(v, vs, &format!("V_SR(k) parallel vs serial ({n} threads)"));
        assert_cbitwise(v, v_ref, &format!("V_SR(k) at {n} vs 1 thread"));
        assert_eq!(nt, nts, "triplets vs serial ({n} threads)");
        assert_eq!(nt, nt_ref, "triplets at {n} vs 1 thread");
    }

    // The full h(k) across thread counts.
    let hks: Vec<_> = THREADS
        .iter()
        .map(|&n| {
            in_pool(n, || {
                periodic_hcore_kpts(&cell, &prep, &mesh, &cfg).expect("hcore k")
            })
        })
        .collect();
    for (&n, hk) in THREADS.iter().zip(&hks) {
        assert_cbitwise(&hk.h, &hks[0].h, &format!("h(k) at {n} vs 1 thread"));
        assert_eq!(hk.n_sr_triplets, hks[0].n_sr_triplets);
        assert_eq!(hk.n_sr_triplets, *nt_ref, "hcore vs oracle triplet count");
    }
}

#[test]
fn krsgdf_blocks_and_krhf_energy_are_bitwise_across_threads() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &sp_basis_h());
    let aux = prep_for(&cell, &basis::bundled("def2-universal-jkfit").unwrap());
    assert_split_binds(&prep);
    let hcfg = PeriodicHcoreConfig::with_omega(HCORE_OMEGA);
    let kcfg = KRsGdfConfig {
        gdf: gdf_cfg(),
        mutation: None,
    };
    for n_mesh in [[1usize, 1, 2], [1, 1, 3]] {
        let mesh = KPointMesh::gamma_centred(&cell, n_mesh).unwrap();
        let nk = mesh.nk();
        type Run = (Vec<Array2<Complex64>>, Vec<Array2<Complex64>>, f64);
        let runs: Vec<Run> = THREADS
            .iter()
            .map(|&n| {
                in_pool(n, || {
                    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcfg).expect("hcore k");
                    let gdf =
                        KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kcfg).expect("krsgdf");
                    let blocks: Vec<Array2<Complex64>> =
                        (0..nk * nk).map(|kk| gdf.block(kk / nk, kk % nk)).collect();
                    let mut cfg = KRhfConfig::for_cell(&cell, ExxDiv::None);
                    cfg.scf = KScfConfig {
                        energy_conv: 1e-13,
                        grad_conv: 1e-10,
                        ..Default::default()
                    };
                    cfg.hcore = hcfg;
                    cfg.jk = KJkKind::RsGdf;
                    cfg.rsgdf = kcfg;
                    let r = solve_krhf(&cell, &prep, Some(&aux), &mesh, &cfg).expect("krhf");
                    assert!(r.converged, "{n_mesh:?} at {n} threads did not converge");
                    (hk.h, blocks, r.energy)
                })
            })
            .collect();
        let (h1, b1, e1) = &runs[0];
        if n_mesh == [1, 1, 3] {
            // (At N = 2 every phase is ±1 and B is real by construction.)
            assert!(
                b1.iter().any(|b| b.iter().any(|z| z.im != 0.0)),
                "{n_mesh:?}: B(k, k') all real, so the q phases are vacuous"
            );
        }
        for (&n, (h, b, e)) in THREADS.iter().zip(&runs) {
            let tag = format!("{n_mesh:?} at {n} vs 1 thread");
            assert_cbitwise(h, h1, &format!("h(k) {tag}"));
            assert_cbitwise(b, b1, &format!("B(k, k') {tag}"));
            // h and B are bitwise equal here, so a difference below would
            // come from the SCF driver itself, not from the SR walks.
            assert_eq!(
                e.to_bits(),
                e1.to_bits(),
                "k-point RS-GDF RHF energy {tag}: {e:.17e} vs {e1:.17e}"
            );
        }
        if n_mesh == [1, 1, 2] {
            assert!(
                (e1 - TRI_112_JKFIT_NONE).abs() < PIN_TOL,
                "1x1x2 energy {e1:.16} vs pin {TRI_112_JKFIT_NONE:.16}"
            );
        }
    }
}

// ======================================================================
// Force / stress derivative walks (ordered-parallel, `ferric_pbc::ordered`)
// ======================================================================

/// `pbc_grad_rsgdf.rs` TRI_MOVED (Bohr): the triclinic s+p cell its RHF
/// force anchor converges on.
const TRI_MOVED: [[f64; 3]; 4] = [
    [0.13, 0.25, 0.31],
    [0.02, 0.27, 1.66],
    [2.47, 2.41, 2.25],
    [3.52, 2.98, 2.71],
];
const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
const GRAD_OMEGA: f64 = 0.8;
const GRAD_HCORE_PRECISION: f64 = 1e-14;

fn grad_hcore_cfg() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: GRAD_HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(GRAD_OMEGA)
    }
}

fn cell_of(pos: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

fn assert_mat3_bitwise(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3], what: &str) {
    for x in 0..3 {
        for z in 0..3 {
            assert_eq!(
                a[x][z].to_bits(),
                b[x][z].to_bits(),
                "{what}: element ({x}, {z}) {:.17e} vs {:.17e}",
                a[x][z],
                b[x][z]
            );
        }
    }
}

fn nonzero(a: &Array2<f64>) -> bool {
    a.iter().any(|x| *x != 0.0)
}

/// A Gamma RS-GDF system: hcore, `build_for_gradient` fit and its aux.
struct GdfSys {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
    aux: PreparedBasis,
    gdf: RsGdf,
}

fn gdf_sys(cell: Cell, bs: &BasisSet) -> GdfSys {
    let prep = prep_for(&cell, bs);
    let hc = periodic_hcore(&cell, &prep, &grad_hcore_cfg()).expect("hcore");
    let aux = prep_for(&cell, &basis::bundled("cc-pvdz-ri").unwrap());
    let gdf = RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, &gdf_cfg())
        .expect("RsGdf::build_for_gradient");
    GdfSys {
        cell,
        prep,
        hc,
        aux,
        gdf,
    }
}

impl GdfSys {
    fn src(&self) -> RsGdfGradSource<'_> {
        RsGdfGradSource {
            gdf: &self.gdf,
            aux: &self.aux,
            aux_jac: None,
        }
    }
}

fn ggcfg(m: Option<GradMutation>) -> GammaGradConfig {
    GammaGradConfig {
        budget_bytes: Some(AMPLE),
        mutation: m,
        ..Default::default()
    }
}

/// `[1, 2, 6 threads, serial oracle at 1 thread, serial oracle at 6
/// threads]` of `f(mutation)`.
fn thread_runs<R: Send>(f: impl Fn(bool) -> R + Sync) -> (Vec<R>, [R; 2]) {
    let par = THREADS.iter().map(|&n| in_pool(n, || f(false))).collect();
    let ser = [in_pool(1, || f(true)), in_pool(6, || f(true))];
    (par, ser)
}

/// Every piece of a Gamma force the ordered walks feed, bitwise.
fn assert_gamma_grad_bitwise(a: &GammaGradient, b: &GammaGradient, what: &str) {
    assert_bitwise(&a.grad, &b.grad, &format!("{what}: grad"));
    let (p, q) = (&a.parts, &b.parts);
    assert_bitwise(&p.vsr_basis, &q.vsr_basis, &format!("{what}: vsr_basis"));
    assert_bitwise(&p.vsr_nuc, &q.vsr_nuc, &format!("{what}: vsr_nuc"));
    assert_bitwise(&p.fit_orb_sr, &q.fit_orb_sr, &format!("{what}: fit_orb_sr"));
    assert_bitwise(&p.fit_aux_sr, &q.fit_aux_sr, &format!("{what}: fit_aux_sr"));
    assert_bitwise(
        &p.fit_metric_sr,
        &q.fit_metric_sr,
        &format!("{what}: fit_metric_sr"),
    );
    assert_bitwise(&p.ecp, &q.ecp, &format!("{what}: ecp"));
    assert_eq!(a.n_sr_triplets, b.n_sr_triplets, "{what}: SR triplets");
    match (&a.fit, &b.fit) {
        (Some(f), Some(g)) => {
            assert_eq!(
                (f.n_sr3_deriv, f.n_sr2_deriv),
                (g.n_sr3_deriv, g.n_sr2_deriv),
                "{what}: SR derivative counts"
            );
        }
        (None, None) => {}
        _ => panic!("{what}: fit diagnostics presence differs"),
    }
}

/// Ordered-parallel derivative walks: the Gamma RHF RS-GDF force (SR
/// attraction walk + fit SR 3-centre + SR metric) at 1/2/6 threads equals
/// the FROZEN serial walks (`GradMutation::SerialDerivWalks`) BIT FOR BIT,
/// and the serial oracle itself is thread-count independent. Every walk's
/// output is non-zero (no vacuous comparison).
#[test]
fn gamma_rhf_rsgdf_force_is_bitwise_across_threads_and_vs_serial_walks() {
    let sys = gdf_sys(cell_of(&TRI_MOVED, TRI_A, 1), &sp_basis_h());
    assert_split_binds(&sys.prep);
    let g = sys
        .gdf
        .clone()
        .with_exxdiv(&sys.cell, ExxDiv::None)
        .unwrap();
    let scf = gamma_rhf_jk(
        &sys.cell,
        &sys.prep,
        &sys.hc,
        Box::new(g.j_builder()),
        Box::new(g.k_builder()),
    );
    let run = |serial: bool| {
        let m = serial.then_some(GradMutation::SerialDerivWalks);
        gamma_rhf_gradient_rsgdf(
            &sys.cell,
            &sys.prep,
            &grad_hcore_cfg(),
            &sys.hc,
            &sys.src(),
            &scf,
            ExxDiv::None,
            &ggcfg(m),
        )
        .expect("gamma_rhf_gradient_rsgdf")
    };
    let (par, ser) = thread_runs(run);
    let r = &ser[0];
    assert!(nonzero(&r.parts.vsr_basis) && nonzero(&r.parts.vsr_nuc));
    let p = &r.parts;
    assert!(
        nonzero(&p.fit_orb_sr) && nonzero(&p.fit_aux_sr) && nonzero(&p.fit_metric_sr),
        "fit SR force vacuous"
    );
    let f = r.fit.as_ref().expect("fit diagnostics");
    assert!(f.n_sr3_deriv > 0 && f.n_sr2_deriv > 0);
    assert_gamma_grad_bitwise(&ser[1], r, "serial oracle at 6 vs 1 thread");
    for (&n, g) in THREADS.iter().zip(&par) {
        assert_gamma_grad_bitwise(g, r, &format!("RHF RS-GDF force at {n} threads vs serial"));
    }
}

/// The same for an open-shell hybrid: H3 doublet UKS PBE0 on RS-GDF.
#[test]
fn gamma_uks_rsgdf_force_is_bitwise_across_threads_and_vs_serial_walks() {
    let sys = gdf_sys(cell_of(&H3_ATOMS, cubic(4.5), 2), &pyscf_sto3g_h());
    let dft = GammaUksConfig {
        grid: PeriodicGridConfig {
            neighbour_cutoff: Some(8.0),
            ..PeriodicGridConfig::with_size(40, 50)
        },
        exxdiv: ExxDiv::None,
        ..GammaUksConfig::new("PBE0")
    };
    let r = gamma_uks(
        &sys.cell,
        &sys.prep,
        &sys.hc,
        GammaUhfIntegrals::RsGdf(&sys.gdf),
        &dft,
    )
    .expect("gamma_uks");
    assert!(r.scf.converged);
    let run = |serial: bool| {
        let m = serial.then_some(GradMutation::SerialDerivWalks);
        gamma_uks_gradient_rsgdf(
            &sys.cell,
            &sys.prep,
            &grad_hcore_cfg(),
            &sys.hc,
            &sys.src(),
            &r.scf,
            &dft,
            &ggcfg(m),
        )
        .expect("gamma_uks_gradient_rsgdf")
    };
    let (par, ser) = thread_runs(run);
    assert!(nonzero(&ser[0].parts.vsr_basis) && nonzero(&ser[0].parts.fit_orb_sr));
    assert_gamma_grad_bitwise(&ser[1], &ser[0], "serial oracle at 6 vs 1 thread");
    for (&n, g) in THREADS.iter().zip(&par) {
        assert_gamma_grad_bitwise(
            g,
            &ser[0],
            &format!("UKS PBE0 RS-GDF force at {n} threads vs serial"),
        );
    }
}

/// The RS-GDF stress: SR attraction strain + fit SR 3-centre and metric
/// strain walks, bitwise across threads and vs the serial walks.
#[test]
fn gamma_rhf_rsgdf_stress_is_bitwise_across_threads_and_vs_serial_walks() {
    let sys = gdf_sys(cell_of(&TRI_MOVED, TRI_A, 1), &sp_basis_h());
    let g = sys
        .gdf
        .clone()
        .with_exxdiv(&sys.cell, ExxDiv::None)
        .unwrap();
    let scf = gamma_rhf_jk(
        &sys.cell,
        &sys.prep,
        &sys.hc,
        Box::new(g.j_builder()),
        Box::new(g.k_builder()),
    );
    let run = |serial: bool| {
        let cfg = GammaStressConfig {
            budget_bytes: Some(AMPLE),
            mutation: serial.then_some(StressMutation::SerialDerivWalks),
            ..Default::default()
        };
        gamma_rhf_stress_rsgdf(
            &sys.cell,
            &sys.prep,
            &grad_hcore_cfg(),
            &sys.hc,
            &sys.src(),
            &scf,
            ExxDiv::None,
            &cfg,
        )
        .expect("gamma_rhf_stress_rsgdf")
    };
    let (par, ser) = thread_runs(run);
    let r = &ser[0];
    let nz = |m: &[[f64; 3]; 3]| m.iter().flatten().any(|x| *x != 0.0);
    assert!(nz(&r.parts.vsr) && nz(&r.parts.fit_j3_sr) && nz(&r.parts.fit_j2_sr));
    let check = |a: &GammaStress, what: &str| {
        assert_mat3_bitwise(&a.de_deps, &r.de_deps, &format!("{what}: dE/dε"));
        assert_mat3_bitwise(&a.parts.vsr, &r.parts.vsr, &format!("{what}: vsr"));
        assert_mat3_bitwise(
            &a.parts.fit_j3_sr,
            &r.parts.fit_j3_sr,
            &format!("{what}: j3_sr"),
        );
        assert_mat3_bitwise(
            &a.parts.fit_j2_sr,
            &r.parts.fit_j2_sr,
            &format!("{what}: j2_sr"),
        );
    };
    check(&ser[1], "serial oracle at 6 vs 1 thread");
    for (&n, s) in THREADS.iter().zip(&par) {
        check(s, &format!("RS-GDF stress at {n} threads vs serial"));
    }
}

/// `pbc_kgrad.rs`'s even-tempered s+p aux on H.
fn et_sp_aux() -> BasisSet {
    let mut shells = Vec::new();
    for a in [4.7, 1.9, 0.75, 0.3] {
        shells.push(Shell {
            l: 0,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    for a in [1.25, 0.5] {
        shells.push(Shell {
            l: 1,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "et-sp-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

/// k-point RS-GDF RHF forces (the L-weighted SR attraction visitor, the
/// phase-folded SR 3-centre and metric walks) on H2 1×1×3: bitwise across
/// threads and vs the serial walks (`KGradMutation::SerialDerivWalks`).
#[test]
fn kpoint_rhf_rsgdf_force_is_bitwise_across_threads_and_vs_serial_walks() {
    let cell = cell_of(&[[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]], cubic(4.0), 1);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = prep_for(&cell, &et_sp_aux());
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &grad_hcore_cfg()).expect("hcore k");
    let kcfg = KRsGdfConfig {
        gdf: gdf_cfg(),
        mutation: None,
    };
    let gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kcfg).expect("KRsGdf");
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk: Box::new(gdf.jk_builder_with_madelung(0.0)),
    };
    let scf_cfg = KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    };
    let scf = solve_krhf_injected(&cell, &mesh, &scf_cfg, inj).expect("k-RHF");
    assert!(scf.converged);
    let src = KRsGdfGradSource {
        gdf: &gdf,
        cfg: &kcfg,
        aux: &aux,
    };
    let run = |serial: bool| {
        let cfg = KGradConfig {
            budget_bytes: Some(AMPLE),
            mutation: serial.then_some(KGradMutation::SerialDerivWalks),
            ..Default::default()
        };
        kpoint_rhf_gradient(
            &cell,
            &prep,
            &mesh,
            &grad_hcore_cfg(),
            &hk,
            KGradJk::RsGdf(src),
            &scf,
            ExxDiv::None,
            &cfg,
        )
        .expect("kpoint_rhf_gradient")
    };
    let (par, ser) = thread_runs(run);
    let r = &ser[0];
    assert!(nonzero(&r.parts.vsr_basis) && nonzero(&r.parts.vsr_nuc));
    let check = |a: &KGradient, what: &str| {
        assert_bitwise(&a.grad, &r.grad, &format!("{what}: grad"));
        assert_bitwise(
            &a.parts.vsr_basis,
            &r.parts.vsr_basis,
            &format!("{what}: vsr_basis"),
        );
        assert_bitwise(
            &a.parts.vsr_nuc,
            &r.parts.vsr_nuc,
            &format!("{what}: vsr_nuc"),
        );
        assert_eq!(a.n_sr_triplets, r.n_sr_triplets, "{what}: SR triplets");
    };
    check(&ser[1], "serial oracle at 6 vs 1 thread");
    for (&n, g) in THREADS.iter().zip(&par) {
        check(g, &format!("k-point RS-GDF force at {n} threads vs serial"));
    }
}

// ------------------------------------------------ periodic ECP force term

/// `pbc_grad_ecp.rs` fixtures (compact H–I, LANL2DZ ECP on I).
fn norm_shell(l: i32, exps: &[f64], coefs: &[f64]) -> Shell {
    let lf = l as f64;
    let mut s = 0.0;
    for (a, ca) in exps.iter().zip(coefs) {
        for (b, cb) in exps.iter().zip(coefs) {
            s += ca * cb * (2.0 * (a * b).sqrt() / (a + b)).powf(lf + 1.5);
        }
    }
    Shell {
        l,
        pure: false,
        exponents: exps.to_vec(),
        coefficients: coefs.iter().map(|c| c / s.sqrt()).collect(),
    }
}

fn lanl2dz_i_ecp() -> EcpDef {
    let ch = |l: i32, t: &[(i32, f64, f64)]| EcpShell {
        angular_momentum: l,
        terms: t
            .iter()
            .map(|&(n, z, d)| EcpTerm {
                coef: d,
                r_exp: n,
                gexp: z,
            })
            .collect(),
    };
    EcpDef {
        n_core: 46,
        shells: vec![
            ch(
                3,
                &[
                    (0, 1.0715702, -0.0747621),
                    (1, 44.1936028, -30.0811224),
                    (2, 12.9367609, -75.3722721),
                    (2, 3.1956412, -22.0563758),
                    (2, 0.8589806, -1.6979585),
                ],
            ),
            ch(
                0,
                &[
                    (0, 127.9202670, 2.9380036),
                    (1, 78.6211465, 41.2471267),
                    (2, 36.5146237, 287.8680095),
                    (2, 9.9065681, 114.3758506),
                    (2, 1.9420086, 37.6547714),
                ],
            ),
            ch(
                1,
                &[
                    (0, 13.0035304, 2.2222630),
                    (1, 76.0331404, 39.4090831),
                    (2, 24.1961684, 177.4075002),
                    (2, 6.4053433, 77.9889462),
                    (2, 1.5851786, 25.7547641),
                ],
            ),
            ch(
                2,
                &[
                    (0, 40.4278108, 7.0524360),
                    (1, 28.9084375, 33.3041635),
                    (2, 15.6268936, 186.9453875),
                    (2, 4.1442856, 71.9688361),
                    (2, 0.9377235, 9.3630657),
                ],
            ),
        ],
    }
}

fn hi_compact_basis() -> BasisSet {
    let mut shells = HashMap::new();
    shells.insert(
        1,
        vec![norm_shell(
            0,
            &[3.42525091, 0.62391373, 0.1688554],
            &[0.15432897, 0.53532814, 0.44463454],
        )],
    );
    shells.insert(
        53,
        vec![
            norm_shell(0, &[0.4653], &[1.0]),
            norm_shell(1, &[0.32], &[1.0]),
        ],
    );
    let mut ecps = HashMap::new();
    ecps.insert(53, lanl2dz_i_ecp());
    BasisSet {
        name: "HI-compact".into(),
        shells,
        ecps,
    }
}

fn hi_cell(bs: &BasisSet) -> Cell {
    let atom = |symbol: &str, z: i32, r: [f64; 3]| Atom {
        symbol: symbol.into(),
        z,
        x: r[0],
        y: r[1],
        zpos: r[2],
        ghost: false,
        n_core_ecp: 0,
    };
    let mut mol = Molecule {
        atoms: vec![
            atom("H", 1, [0.3, 0.2, 0.4]),
            atom("I", 53, [0.9, -0.4, 3.3]),
        ],
        charge: 0,
        multiplicity: 1,
    };
    mol.apply_ecp(bs);
    let a = [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]];
    Cell::new(mol, a).expect("HI cell")
}

/// The periodic-ECP force term (one ECP derivative block per orbital image,
/// every addend replayed serially): bitwise across threads and vs the
/// frozen serial image loop (`EcpGradMutation::SerialImages`), on a fixed
/// symmetric density.
#[test]
fn ecp_force_term_is_bitwise_across_threads_and_vs_serial_loop() {
    let bs = hi_compact_basis();
    let cell = hi_cell(&bs);
    let prep = prep_for(&cell, &bs);
    let n = prep.nbasis();
    let d = Array2::from_shape_fn((n, n), |(i, j)| {
        let (a, b) = (i as f64, j as f64);
        0.3 * (1.1 * a + 0.7 * b).cos()
            + 0.3 * (1.1 * b + 0.7 * a).cos()
            + if i == j { 0.5 } else { 0.0 }
    });
    let ecfg = PeriodicEcpConfig::with_precision(GRAD_HCORE_PRECISION);
    let run = |serial: bool| {
        let m = serial.then_some(EcpGradMutation::SerialImages);
        periodic_ecp_gradient_with(&cell, &prep, &ecfg, &d, m)
            .expect("ECP gradient")
            .expect("cell has an ECP")
    };
    let (par, ser) = thread_runs(run);
    let r = &ser[0];
    assert!(
        r.n_calls > 1,
        "only {} image(s): order is vacuous",
        r.n_calls
    );
    assert!(nonzero(&r.bra) && nonzero(&r.ket) && nonzero(&r.centre));
    let check = |a: &PeriodicEcpGradient, what: &str| {
        assert_bitwise(&a.grad, &r.grad, &format!("{what}: grad"));
        assert_bitwise(&a.bra, &r.bra, &format!("{what}: bra"));
        assert_bitwise(&a.ket, &r.ket, &format!("{what}: ket"));
        assert_bitwise(&a.centre, &r.centre, &format!("{what}: centre"));
        assert_eq!(
            (a.n_triples, a.n_triples_evaluated, a.n_calls),
            (r.n_triples, r.n_triples_evaluated, r.n_calls),
            "{what}: counts"
        );
    };
    check(&ser[1], "serial oracle at 6 vs 1 thread");
    for (&nt, g) in THREADS.iter().zip(&par) {
        check(g, &format!("ECP force term at {nt} threads vs serial"));
    }
    // A mutant still flows through the parallel path (NoCentre drops it).
    let nc = in_pool(6, || {
        periodic_ecp_gradient_with(&cell, &prep, &ecfg, &d, Some(EcpGradMutation::NoCentre))
            .unwrap()
            .unwrap()
    });
    assert!(!nonzero(&nc.centre), "NoCentre: centre part not dropped");
    assert_bitwise(&nc.bra, &r.bra, "NoCentre leaves bra untouched");
}

// ---------------------------------------------------------------------------
// LR pair FT: survivor-cached, shell-pair-parallel kernel (FINDINGS
// "Performance plan (research) — 2026-09-25" §5, "Pair-FT re-walk").
//
// Construction under test: the primitive-pair screen is walked ONCE per
// call and cached per shell pair in walk order; every G chunk then runs in
// parallel over SHELL PAIRS, each task owning its pair's row segments of P,
// written into output buffers REUSED (not re-zeroed) across chunks. Every
// element keeps the serial addend sequence, so production must equal the FROZEN
// pre-parallel kernels (`*_serial_oracle`, which re-walk the screen per
// chunk) BIT FOR BIT, at every thread count and with SEVERAL G chunks. What
// each test guards:
//
// * `pair_ft_kernels_are_bitwise_vs_frozen_serial_kernels_across_threads` —
//   the Gamma and the residue-binned kernels, s + Cartesian p + PURE d
//   (the cart→sph path), an unsorted G list with G = 0 (the |G| sort and
//   the scatter back), ≥ 3 chunks. A survivor list out of walk order, a
//   window evaluated per chunk instead of over the whole set, a dropped
//   survivor that reached some G, a bucket mix-up, a pair-task race or a
//   missed write into a reused buffer (the chunks shrink from 7 G to 6 G, so
//   the last reuses a longer buffer's stale values) each break it. Its G
//   list ends (|G| <= 5.3) far inside every pair's window, so it never
//   takes the partial-window path (`0 < n_used < n_G`).
// * `pair_ft_kernels_are_bitwise_vs_frozen_serial_kernels_in_partial_windows`
//   — the same comparison on the same G directions scaled ×6 (|G| <= 31.6 /
//   35.4), where the p/d pairs' windows (|G| ≈ 17.5–20.8) and the s–p/d ones
//   (≈ 27.3–29.5) end INSIDE most chunks while the s–s ones (≈ 32–33.4)
//   stay full; asserts via `pair FT partial-window pair-chunks` that both
//   kernels really took that path. A nonzero written beyond a partial window
//   (the oracle has +0.0 there) breaks it. Both spd tests also assert that
//   every shell group is one shell (sites == survivors): they never share a
//   site.
// * `pair_ft_kernels_are_bitwise_vs_frozen_serial_kernels_with_general_contraction`
//   — the same comparison (G ×6, partial windows) on a GENERALLY CONTRACTED
//   basis: three s columns on one exponent set (one with zero coefficients,
//   as cc-pVDZ's third s) and two p columns on another, so the production
//   kernel groups shells, shares each site's trig/E/F values between the
//   member shell pairs (whose coefficients, screens and windows differ) and
//   runs tasks over multi-shell bra groups. Asserts survivors > sites and a
//   group of 3 shells for both kernels. A site value taken from the wrong
//   member, an entry accumulated past its OWN window (the site's is larger),
//   a member mapped to the wrong accumulator or AO rows/columns, or a wrong
//   exp-table slot each break it.
// * `rsgdf_lr_sums_are_bitwise_vs_frozen_serial_across_threads` — the whole
//   RS-GDF LR stage (pair FT + aux FT + reused-buffer packing + the
//   row-blocked parallel J3 GEMMs, both terms per block, beside the J2
//   GEMMs) against the FROZEN serial stage at the SAME G chunks, with and
//   without the range split (moved-aux and smooth-pair LR passes).
// * `rsgdf_b_and_energy_are_bitwise_across_threads_with_several_lr_chunks` —
//   `RsGdf::build` under a budget that forces ≥ 3 LR chunks: J2 (via the fit
//   parts), B and the RHF energy at 1/2/6 threads, with and without split.
// ---------------------------------------------------------------------------

/// H: STO-3G s, Cartesian p (0.8) and a PURE d (0.9) — 3 shells per atom.
fn spd_basis_h() -> BasisSet {
    let mut s = BasisSet {
        name: "pbc-bitwise-spd-H".into(),
        shells: HashMap::new(),
        ecps: HashMap::new(),
    };
    let mut d = norm_shell(2, &[0.9], &[1.0]);
    d.pure = true;
    s.shells.insert(
        1,
        vec![
            norm_shell(
                0,
                &[3.42525091, 0.62391373, 0.1688554],
                &[0.15432897, 0.53532814, 0.44463454],
            ),
            norm_shell(1, &[0.8], &[1.0]),
            d,
        ],
    );
    s
}

/// H, generally contracted: three s columns on the STO-3G exponents (the
/// STO-3G contraction, a sign-changing one, and the diffuse primitive alone
/// with zero coefficients elsewhere, like cc-pVDZ's third s), two Cartesian p
/// columns on {1.2, 0.35} (the second the diffuse primitive alone) and a PURE
/// d (0.9): 6 shells, 3 shell groups per atom.
fn gc_basis_h() -> BasisSet {
    let mut s = BasisSet {
        name: "pbc-bitwise-gc-H".into(),
        shells: HashMap::new(),
        ecps: HashMap::new(),
    };
    let se = [3.42525091, 0.62391373, 0.1688554];
    let pe = [1.2, 0.35];
    let mut d = norm_shell(2, &[0.9], &[1.0]);
    d.pure = true;
    s.shells.insert(
        1,
        vec![
            norm_shell(0, &se, &[0.15432897, 0.53532814, 0.44463454]),
            norm_shell(0, &se, &[-0.3, 0.2, 1.0]),
            norm_shell(0, &se, &[0.0, 0.0, 1.0]),
            norm_shell(1, &pe, &[0.6, 0.5]),
            norm_shell(1, &pe, &[0.0, 1.0]),
            d,
        ],
    );
    s
}

/// `Σ m_i b_i` for `m ∈ {−2..=2}³` plus `shift`, in a deterministic
/// NON-|G|-sorted order (index permutation `i → 37 i mod 125`), G = 0 kept.
fn scrambled_gvecs(cell: &Cell, shift: [f64; 3]) -> Vec<[f64; 3]> {
    let b = cell.reciprocal();
    let mut all = Vec::new();
    for i in -2i32..=2 {
        for j in -2i32..=2 {
            for k in -2i32..=2 {
                let (i, j, k) = (i as f64, j as f64, k as f64);
                all.push([
                    i * b[0][0] + j * b[1][0] + k * b[2][0] + shift[0],
                    i * b[0][1] + j * b[1][1] + k * b[2][1] + shift[1],
                    i * b[0][2] + j * b[1][2] + k * b[2][2] + shift[2],
                ]);
            }
        }
    }
    let n = all.len();
    (0..n).map(|i| all[(37 * i) % n]).collect()
}

fn c3bit_diffs(a: &Array3<Complex64>, b: &Array3<Complex64>) -> usize {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.re.to_bits() != y.re.to_bits() || x.im.to_bits() != y.im.to_bits())
        .count()
}

/// Chunks of one chunked call: `(g0, [P_r])` in sink order.
type Chunks = Vec<(usize, Vec<Array3<Complex64>>)>;

fn assert_chunks_bitwise(a: &Chunks, b: &Chunks, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: chunk counts");
    for ((g0a, pa), (g0b, pb)) in a.iter().zip(b) {
        assert_eq!(g0a, g0b, "{what}: chunk offsets");
        assert_eq!(pa.len(), pb.len(), "{what}: bucket counts");
        for (r, (x, y)) in pa.iter().zip(pb).enumerate() {
            let d = c3bit_diffs(x, y);
            assert_eq!(
                d,
                0,
                "{what}: chunk g0 = {g0a}, bucket {r}: {d} of {} elements differ in bits",
                x.len()
            );
        }
    }
}

/// `v` with every component multiplied by `s` (`s = 1` is exact).
fn scaled(v: Vec<[f64; 3]>, s: f64) -> Vec<[f64; 3]> {
    v.into_iter()
        .map(|g| [g[0] * s, g[1] * s, g[2] * s])
        .collect()
}

/// Production chunked pair FT (Gamma and residue kernels) vs the FROZEN
/// serial oracles, bit for bit at 1/2/6 threads, on `scrambled_gvecs` scaled
/// by `g_scale`, in basis `bs`. Returns the production runs' timings
/// (Gamma, residue) at 1 thread; their [`CTR_PARTIAL_WINDOW`] count is
/// asserted equal at every thread count.
fn pair_ft_bitwise_case(bs: &BasisSet, g_scale: f64) -> (PbcTimings, PbcTimings) {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, bs);
    assert_split_binds(&prep);
    let n = prep.nbasis();
    let thresh = DEFAULT_PAIR_FT_THRESH;
    // 7 G per chunk: 125 G -> 18 chunks.
    let budget = 7 * pair_ft_bytes_per_g(n, 2);

    // --- Gamma kernel.
    let gv = scaled(scrambled_gvecs(&cell, [0.0; 3]), g_scale);
    assert!(gv.contains(&[0.0; 3]), "G = 0 missing");
    let gamma = |oracle: bool| -> (Chunks, PbcTimings) {
        let mut out: Chunks = Vec::new();
        let mut t = PbcTimings::default();
        let sink = |g0: usize, _: &[[f64; 3]], p: &Array3<Complex64>| {
            out.push((g0, vec![p.clone()]));
            Ok(())
        };
        let nc = if oracle {
            pair_ft_chunked_serial_oracle(&cell, &prep, &gv, thresh, budget, 0, sink)
        } else {
            pair_ft_chunked_timed(&cell, &prep, &gv, thresh, budget, 0, &mut t, sink)
        }
        .expect("pair FT");
        assert_eq!(nc, out.len());
        (out, t)
    };
    let oracle = in_pool(1, || gamma(true)).0;
    assert!(oracle.len() >= 3, "only {} G chunk(s)", oracle.len());
    assert!(
        oracle
            .iter()
            .all(|(_, p)| p[0].iter().any(|z| z.re != 0.0 || z.im != 0.0)),
        "a vacuous chunk"
    );
    let mut partial_gamma = Vec::new();
    let mut t_gamma = None;
    for &nt in &THREADS {
        let (par, t) = in_pool(nt, || gamma(false));
        assert_chunks_bitwise(
            &par,
            &oracle,
            &format!("Gamma pair FT at {nt} threads vs serial"),
        );
        partial_gamma.push(t.counter(CTR_PARTIAL_WINDOW).expect("window counter"));
        t_gamma.get_or_insert(t);
    }
    // The unchunked entry point is the same kernel over one chunk.
    let whole = in_pool(6, || {
        pair_ft_with_thresh(&cell, &prep, &gv, thresh).unwrap()
    });
    let mut g = 0usize;
    for (g0, p) in &oracle {
        assert_eq!(*g0, g);
        let ng = p[0].dim().2;
        let slice = whole.slice(ndarray::s![.., .., g..g + ng]).to_owned();
        assert_eq!(c3bit_diffs(&slice, &p[0]), 0, "unchunked vs chunk at {g0}");
        g += ng;
    }

    // --- Residue-binned kernel at K = G + q (moduli differ per axis, so a
    // bucket mix-up in any dimension fails).
    let b = cell.reciprocal();
    let q = [
        0.3 * b[0][0] + 0.5 * b[2][0],
        0.3 * b[0][1] + 0.5 * b[2][1],
        0.3 * b[0][2] + 0.5 * b[2][2],
    ];
    let kv = scaled(scrambled_gvecs(&cell, q), g_scale);
    let moduli = [2, 1, 3];
    let kbudget = 7 * 6 * pair_ft_bytes_per_g(n, 2);
    let resid = |oracle: bool| -> (Chunks, PbcTimings) {
        let mut out: Chunks = Vec::new();
        let mut t = PbcTimings::default();
        let sink = |g0: usize, _: &[[f64; 3]], p: &[Array3<Complex64>]| {
            out.push((g0, p.to_vec()));
            Ok(())
        };
        let nc = if oracle {
            pair_ft_residues_chunked_serial_oracle(
                &cell, &prep, &kv, moduli, thresh, kbudget, 0, sink,
            )
        } else {
            pair_ft_residues_chunked_timed(
                &cell, &prep, &kv, moduli, thresh, kbudget, 0, &mut t, sink,
            )
        }
        .expect("residue pair FT");
        assert_eq!(nc, out.len());
        (out, t)
    };
    let roracle = in_pool(1, || resid(true)).0;
    assert!(roracle.len() >= 3, "only {} K chunk(s)", roracle.len());
    let live = roracle[0]
        .1
        .iter()
        .filter(|p| p.iter().any(|z| z.re != 0.0 || z.im != 0.0))
        .count();
    assert!(
        live >= 2,
        "only {live} live residue bucket(s): binning is vacuous"
    );
    let mut partial_resid = Vec::new();
    let mut t_resid = None;
    for &nt in &THREADS {
        let (par, t) = in_pool(nt, || resid(false));
        assert_chunks_bitwise(
            &par,
            &roracle,
            &format!("residue pair FT at {nt} threads vs serial"),
        );
        partial_resid.push(t.counter(CTR_PARTIAL_WINDOW).expect("window counter"));
        t_resid.get_or_insert(t);
    }
    // The window of a (pair, chunk) does not depend on the thread count.
    for (what, v) in [("Gamma", &partial_gamma), ("residue", &partial_resid)] {
        assert!(
            v.iter().all(|&x| x == v[0]),
            "{what}: partial-window counts differ across threads: {v:?}"
        );
    }
    (
        t_gamma.expect("a Gamma run"),
        t_resid.expect("a residue run"),
    )
}

/// `(survivors, sites, largest shell group)` of a production run.
fn sharing(t: &PbcTimings) -> (u64, u64, u64) {
    (
        t.counter(CTR_SURVIVORS).expect("survivor counter"),
        t.counter(CTR_SITES).expect("site counter"),
        t.counter(CTR_LARGEST_GROUP).expect("group counter"),
    )
}

/// The spd basis has one shell per group: no site is shared.
fn assert_no_sharing(t: &PbcTimings, what: &str) {
    let (surv, sites, largest) = sharing(t);
    assert!(
        surv > 0 && surv == sites && largest == 1,
        "{what}: {surv} survivors, {sites} sites, largest group {largest}"
    );
}

#[test]
fn pair_ft_kernels_are_bitwise_vs_frozen_serial_kernels_across_threads() {
    let (g, r) = pair_ft_bitwise_case(&spd_basis_h(), 1.0);
    assert_no_sharing(&g, "Gamma");
    assert_no_sharing(&r, "residue");
}

/// G scale of the partial-window test (see below).
const PARTIAL_WINDOW_G_SCALE: f64 = 6.0;

/// The same bitwise comparison where the per-pair G window ends INSIDE the
/// chunks, so production accumulates `0..n_used` and writes `+0.0` beyond it
/// (`plan.rs` module doc) — the path the unscaled test never takes.
///
/// Expectation from the screen (`g2max = 4p [ln(|cc| e^{−ab r²/p} / thresh)
/// + 10 + (la + lb) ln |G|max]`, largest at a pair's nearest image; thresh
/// 1e-15; `cc = c_a c_b (π/p)^{3/2}` with unit-normalised primitives):
///
/// * one-atom p–p (0.8 + 0.8): p = 1.6, cc = 2p = 3.2, ln(3.2e15) = 35.7;
///   at |G|max = 31.6, g2max = 6.4 (35.7 + 10 + 2 ln 31.6) ≈ 337, |G| ≈ 18.3.
/// * one-atom d–d (0.9 + 0.9): p = 1.8, cc = 4p²/3 = 4.32;
///   g2max = 7.2 (36.0 + 10 + 4 ln 31.6) ≈ 431, |G| ≈ 20.8.
/// * pairs with the contracted STO-3G s: the tight 3.425 primitive sets the
///   window, |G| ≈ 27.3–29.5 against p/d, ≈ 32–33.4 for s–s (p = 6.85).
///
/// Unscaled, |G|max = 5.3 (5.9 for K = G + q) and every window is full. Scaled
/// ×6 the G run to 31.6 (35.4), so every pair involving only p/d shells and
/// the s–p/d pairs end inside the range while the s–s pairs stay full; the
/// scrambled order (index `37 i mod 125`) mixes small and large |G| in each
/// 7-G chunk, so the windows fall INSIDE chunks (not only between them). A
/// replay of the screen over the 18 chunks × 144 shell pairs predicts 1816
/// (Gamma) and 1960 (residue) partial pair-chunks; the test only requires
/// > 0 for each kernel.
#[test]
fn pair_ft_kernels_are_bitwise_vs_frozen_serial_kernels_in_partial_windows() {
    let (g, r) = pair_ft_bitwise_case(&spd_basis_h(), PARTIAL_WINDOW_G_SCALE);
    assert_no_sharing(&g, "Gamma");
    assert_no_sharing(&r, "residue");
    let partial = |t: &PbcTimings| t.counter(CTR_PARTIAL_WINDOW).expect("window counter");
    let (gamma, resid) = (partial(&g), partial(&r));
    assert!(
        gamma > 0,
        "Gamma kernel: no (pair, chunk) with 0 < n_used < n_G; the partial-window path is unreached"
    );
    assert!(
        resid > 0,
        "residue kernel: no (pair, chunk) with 0 < n_used < n_G; the partial-window path is unreached"
    );
}

/// The bitwise comparison on the generally contracted [`gc_basis_h`] with
/// the partial-window G scale: the production kernel's shared-site path
/// (shell groups of 3 s and 2 p columns; entries of one site with different
/// coefficients and windows, some of them zero; multi-shell bra-group tasks),
/// which the spd tests never reach (they assert sites == survivors).
#[test]
fn pair_ft_kernels_are_bitwise_vs_frozen_serial_kernels_with_general_contraction() {
    let (g, r) = pair_ft_bitwise_case(&gc_basis_h(), PARTIAL_WINDOW_G_SCALE);
    for (what, t) in [("Gamma", &g), ("residue", &r)] {
        let (surv, sites, largest) = sharing(t);
        assert!(
            sites > 0 && surv > sites,
            "{what}: {surv} survivors in {sites} sites: no site is shared, the grouped path is unreached"
        );
        assert_eq!(largest, 3, "{what}: the three s columns are not one group");
        assert!(
            t.counter(CTR_PARTIAL_WINDOW).expect("window counter") > 0,
            "{what}: no partial window"
        );
    }
}

/// `RsGdfConfig` of the LR tests: ω = 1, exxdiv none, `split` optional.
fn lr_cfg(split: Option<RangeSplit>, budget: usize) -> RsGdfConfig {
    RsGdfConfig {
        range_split: split,
        budget_bytes: Some(budget),
        ..gdf_cfg()
    }
}

/// Upper bound on the per-G bytes of every LR pass (`pair_ft` scratch plus
/// the moved-aux sink's, the largest of the three sinks).
fn lr_per_g_bound(prep: &PreparedBasis, aux: &PreparedBasis, lmax: usize) -> usize {
    let n = prep.nbasis();
    pair_ft_bytes_per_g(n, lmax) + 16 * n * n + 112 * aux.nbasis() + 64
}

#[test]
fn rsgdf_lr_sums_are_bitwise_vs_frozen_serial_across_threads() {
    // The triclinic cell of `pbc_rsgdf_split.rs` (the split moves the
    // STO-3G 0.169 primitive and 8/24 aux shells there) with a pure d added:
    // nao = 36, so J3 has 1296 rows and the row-blocked LR GEMM
    // (`LR_GEMM_ROW_BLOCK` = 512) really splits.
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &spd_basis_h());
    assert!(prep.nbasis().pow(2) > 2 * 512, "J3 GEMM would not split");
    let aux = prep_for(&cell, &basis::bundled("cc-pvdz-ri").unwrap());
    assert_split_binds(&prep);
    let n_g = RsGdf::build(&cell, &prep, &aux, &Array2::eye(prep.nbasis()), &gdf_cfg())
        .expect("rsgdf")
        .stats()
        .n_g_half;
    // At most n_g / 8 G per chunk of the main pass.
    let budget = lr_per_g_bound(&prep, &aux, 2) * (n_g / 8).max(1);
    let mut j3_by_split = Vec::new();
    for split in [None, Some(RangeSplit::default())] {
        let tag = format!("split {}", split.is_some());
        let cfg = lr_cfg(split, AMPLE);
        let runs: Vec<[(Array2<f64>, Array2<f64>, usize); 2]> = THREADS
            .iter()
            .map(|&nt| {
                in_pool(nt, || {
                    lr_sums_parallel_and_serial(&cell, &prep, &aux, &cfg, budget).expect("LR")
                })
            })
            .collect();
        let [_, (j2s, j3s, cs)] = &runs[0];
        assert!(*cs >= 4, "{tag}: only {cs} LR chunk(s)");
        assert!(nonzero(j2s) && nonzero(j3s), "{tag}: vacuous LR sums");
        for (&nt, [(j2p, j3p, cp), (j2o, j3o, co)]) in THREADS.iter().zip(&runs) {
            assert_eq!((cp, co), (cs, cs), "{tag}: chunk counts at {nt} threads");
            assert_bitwise(
                j3p,
                j3o,
                &format!("{tag}: J3_LR parallel vs serial ({nt} threads)"),
            );
            assert_bitwise(
                j2p,
                j2o,
                &format!("{tag}: J2_LR parallel vs serial ({nt} threads)"),
            );
            assert_bitwise(
                j3o,
                j3s,
                &format!("{tag}: serial J3_LR at {nt} vs 1 thread"),
            );
            assert_bitwise(
                j2o,
                j2s,
                &format!("{tag}: serial J2_LR at {nt} vs 1 thread"),
            );
        }
        j3_by_split.push(j3s.clone());
    }
    // The split really took the other LR passes.
    assert!(
        bit_diffs(&j3_by_split[0], &j3_by_split[1]) > 0,
        "split and unsplit LR J3 are identical: the split LR passes did not run"
    );
}

#[test]
fn rsgdf_b_and_energy_are_bitwise_across_threads_with_several_lr_chunks() {
    let cell = triclinic_cell();
    let prep = prep_for(&cell, &spd_basis_h());
    let aux = prep_for(&cell, &basis::bundled("cc-pvdz-ri").unwrap());
    let hc =
        periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(HCORE_OMEGA)).expect("hcore");
    for split in [None, Some(RangeSplit::default())] {
        let tag = format!("split {}", split.is_some());
        // Reference build at an ample budget: the resident bytes before the
        // LR chunks and the half-G count fix a budget with ≥ 3 LR chunks.
        // (build_with_fit_parts, like the runs below: its retained parts are
        // reserved before the snapshot.)
        let (ample, _) =
            RsGdf::build_with_fit_parts(&cell, &prep, &aux, &hc.s, &lr_cfg(split, AMPLE))
                .expect("rsgdf");
        let st = ample.stats();
        let budget = st.resident_bytes + lr_per_g_bound(&prep, &aux, 2) * (st.n_g_half / 4).max(1);
        let runs: Vec<(Array2<f64>, Array2<f64>, f64, usize)> = THREADS
            .iter()
            .map(|&nt| {
                in_pool(nt, || {
                    let (gdf, parts) = RsGdf::build_with_fit_parts(
                        &cell,
                        &prep,
                        &aux,
                        &hc.s,
                        &lr_cfg(split, budget),
                    )
                    .expect("rsgdf (tight budget)");
                    let e = gamma_rhf_jk(
                        &cell,
                        &prep,
                        &hc,
                        Box::new(gdf.j_builder()),
                        Box::new(gdf.k_builder()),
                    )
                    .energy;
                    (parts.j2, gdf.b().clone(), e, gdf.stats().n_g_chunks)
                })
            })
            .collect();
        let (j2_1, b1, e1, c1) = &runs[0];
        assert!(
            *c1 >= 3,
            "{tag}: only {c1} LR chunk(s) under the tight budget"
        );
        for (&nt, (j2, b, e, c)) in THREADS.iter().zip(&runs) {
            assert_eq!(c, c1, "{tag}: chunk count at {nt} threads");
            assert_bitwise(j2, j2_1, &format!("{tag}: J2 at {nt} vs 1 thread"));
            assert_bitwise(b, b1, &format!("{tag}: B at {nt} vs 1 thread"));
            assert_eq!(
                e.to_bits(),
                e1.to_bits(),
                "{tag}: RHF energy at {nt} threads {e:.17e} vs 1 thread {e1:.17e}"
            );
        }
        // Chunking only re-blocks the LR GEMM k-sums: the energy moves at
        // round-off level, not more.
        let e_ample = gamma_rhf_jk(
            &cell,
            &prep,
            &hc,
            Box::new(ample.j_builder()),
            Box::new(ample.k_builder()),
        )
        .energy;
        assert!(
            (e1 - e_ample).abs() <= 1e-11,
            "{tag}: tight-budget energy {e1:.15} vs ample {e_ample:.15}"
        );
    }
}
