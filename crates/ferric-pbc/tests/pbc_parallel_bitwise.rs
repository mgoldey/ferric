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
//! Not covered (still serial by design): the force/stress derivative walks,
//! where every triplet adds into shared atom rows, so no output element
//! belongs to one pair task.

mod common;

use common::*;
use ferric_core::basis;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, sr_attraction_kpts_parallel_and_serial};
use ferric_pbc::hcore::{
    periodic_hcore, sr_attraction_parallel_and_serial, PeriodicHcoreConfig, SrAttractionParts,
    SrBound, DEFAULT_HCORE_PRECISION,
};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{solve_krhf, KJkKind, KRhfConfig, KScfConfig};
use ferric_pbc::rsgdf::kpoint::{
    sr_bins_parallel_and_serial, sr_sums_gamma_and_single_bin, KRsGdf, KRsGdfConfig, SrBinsParts,
};
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
use ndarray::Array2;
use num_complex::Complex64;

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
