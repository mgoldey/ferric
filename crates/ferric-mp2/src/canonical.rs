//! Canonical MP2 using full 4-center ERIs transformed to MO basis.
//! For cross-validation only -- O(N^5).
//!
//! ## Transform structure (rewritten 2026-07-25)
//!
//! The AO->MO step is the classic **four quarter transforms**, each a single
//! BLAS3 `dgemm` over a reshaped 2-index view:
//!
//! ```text
//!   (mu nu|la sg) --C_occ--> (i nu|la sg) --C_vir--> (i a|la sg)
//!                 --C_occ--> (i a|j sg)  --C_vir--> (i a|j b)
//! ```
//!
//! Cost is `O(nbas^4 * nmo)` -- i.e. O(N^5) -- and every step is a wide GEMM.
//!
//! The previous implementation fused the whole transform into the AO
//! shell-quartet loop, applying all four MO coefficients to each AO integral
//! individually. That is an outer product per AO element, so it cost
//! `O(nbas^4 * (nocc*nvir)^2)` -- **O(N^8)**, not O(N^5) -- in scalar code with
//! four levels of data-dependent branches and a strided accumulate into
//! `mo_eri`. At water/cc-pVDZ that is ~119x more arithmetic than the quarter
//! transform (2.99e9 vs 2.52e7 FMAs), and the gap grows as `N^3`.
//!
//! The AO integrals are built once into a dense `nbas^4` buffer using the
//! 8-fold permutational symmetry of `(mu nu|la sg)` (only the
//! `s1>=s2, s3>=s4, pair12>=pair34` triangle is evaluated, then scattered to
//! its images), parallel over canonical shell pairs via [`ferric_integrals::engine_pool::EnginePool`] -- one
//! libint2 engine per rayon worker, since constructing one per work-chunk is
//! the documented contention footgun (see `ferric_integrals::engine_pool`).
//!
//! ## Measured (water/cc-pVDZ, OPENBLAS=1 RAYON=1, best of 3)
//!
//! | stage | before | after |
//! |---|---|---|
//! | total | 4.48 s | 0.025 s |
//!
//! ## Where the remaining time goes (measured 2026-07-25, ethane/cc-pVDZ)
//!
//! | stage | time | share |
//! |---|---|---|
//! | libint2 `compute_quartet` (raw FFI loop) | 0.37 s | ~71% |
//! | 8-way scatter into the dense `nbas^4` buffer | ~0.06 s | ~12% |
//! | four quarter transforms + energy | ~0.09 s | ~17% |
//!
//! An earlier note here claimed libint2 was ~99% of wall time; that measurement
//! folded the scatter and the wrapper copy into the "ERI build" stage. libint2
//! itself is ~71%, and the rest is memory traffic over the dense buffer.
//!
//! ### The `compute_quartet` wrapper copy is NOT worth removing (measured)
//!
//! `Engine::compute_quartet` zero-fills its buffer and then runs a scalar
//! `buf[i] += coeff * scratch[i]` accumulate on every call, even for a single
//! unit-coefficient Coulomb handle where it reduces to a plain copy. Removing
//! both (shim writes straight into `buf`) was implemented and A/B'd against a
//! one-component composite control that computes the SAME integrals through the
//! general accumulate loop: **0.98x on water and ethane/cc-pVDZ** -- i.e. no
//! effect outside run-to-run noise. libint2's quartet evaluation so dominates
//! per-call cost that the fill+copy is invisible. Reverted rather than kept:
//! it added a second unsafe FFI write path for zero measured gain. Do not
//! re-propose without a profile showing the copy is material.
//!
//! ### Screening does NOT help at these sizes (measured, negative result)
//!
//! Schwarz bounds prune **0.0%** of the canonical quartet list for
//! water/methane/ethane at cc-pVDZ, all the way up to a 1e-8 threshold (ethane
//! loses 27 of 108345 quartets = 0.02% of AO-element cost at 1e-8, and nothing
//! at 1e-9 or tighter). These molecules are small and compact enough that every
//! shell pair overlaps every other, so no bound falls below threshold. Wiring
//! `SchwarzBounds` in here would add per-quartet bound lookups and prune
//! nothing. Screening only pays past the locality crossover (~naphthalene for
//! this repo's other screened paths). This lane is closed for reference-sized
//! systems.
//!
//! ### The PySCF comparison was warm-vs-cold, not a real gap
//!
//! `mp.MP2` appearing ~15x faster (0.033 s vs ferric 0.52 s on ethane) is an
//! artefact: PySCF's `scf.RHF` **caches the full AO ERI tensor in `mf._eri`**
//! during the SCF, and `mp.MP2` reuses it, so the timed region builds no
//! integrals at all. `mol.intor("int2e")` alone costs 0.146 s on ethane --
//! already 4x the "total MP2 time". Forcing PySCF to build its own integrals
//! (`mf._eri = None`) gives the apples-to-apples number:
//!
//! | system | ferric | PySCF (cold, builds own ERIs) | PySCF (warm, reuses SCF's) |
//! |---|---|---|---|
//! | methane/cc-pVDZ | 0.067 s | 0.101 s | 0.004 s |
//! | ethane/cc-pVDZ  | 0.522 s | 0.691 s | 0.033 s |
//!
//! So ferric is already **faster than PySCF** at the same task. The remaining
//! structural difference is storage, not kernel speed: PySCF keeps the
//! 8-fold-symmetry-packed `s8` form (1.46e6 doubles for ethane) while this path
//! expands to a dense `nbas^4` buffer (1.13e7 doubles, 7.7x more) -- the same
//! unique integrals, scattered wider. Adopting `s8` packing would cut the
//! scatter and buffer traffic (~12% of wall time here) at the cost of packed
//! indexing in the first quarter transform; it would not reduce libint2 work,
//! which is the ~71% floor. Not attempted -- see the git history for this note.

use crate::rimp2::active_occ;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine_pool::EnginePool;
use ferric_integrals::operator::Operator;
use ferric_scf::ScfResult;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Build the dense AO ERI tensor `(μν|λσ)` exactly, with no density fitting.
///
/// Returns a flat `nbas⁴` buffer indexed `((μ·n + ν)·n + λ)·n + σ`, in chemist
/// notation. Uses 8-fold permutational symmetry, so only ~1/8 of the shell quartets
/// are evaluated and each is scattered to its distinct images.
///
/// This is O(nbas⁴) memory and is intended for reference / cross-check work on small
/// systems — it is what lets an RI-based method be compared against exact integrals
/// to quantify its RI error floor. Callers must apply their own memory guard; this
/// function does not (the caller knows what else it holds co-resident).
///
/// `op` is honored, so attenuated operators (`Operator::erfc(ω)`) work here too.
pub fn dense_ao_eri(prep: &PreparedBasis, op: Operator) -> Result<Vec<f64>, FerricError> {
    let nbas = prep.nbasis();
    let nsh = prep.nshells();
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let nb2 = nbas * nbas;

    let pool = EnginePool::new(op, prep, 1e-14)?;
    let mut ao = vec![0.0f64; nb2 * nb2];

    let pairs: Vec<(usize, usize)> =
        (0..nsh).flat_map(|s1| (0..=s1).map(move |s2| (s1, s2))).collect();

    struct AoPtr(*mut f64);
    // SAFETY: tasks write disjoint element sets (each AO element is produced by
    // exactly one canonical quartet, owned by exactly one task); no aliasing.
    unsafe impl Send for AoPtr {}
    unsafe impl Sync for AoPtr {}
    let ao_ptr = AoPtr(ao.as_mut_ptr());

    let ao_ptr = &ao_ptr;
    let pair_list = &pairs;
    pair_list.par_iter().enumerate().for_each(move |(p12, &(s1, s2))| {
        let ao_base = ao_ptr.0;
        pool.with(|eng| {
            for (p34, &(s3, s4)) in pair_list.iter().enumerate() {
                if p34 > p12 {
                    break;
                }
                let Some(q) = eng.compute_quartet(prep, s1, s2, s3, s4) else {
                    continue;
                };
                let (n1, n2, n3, n4) = (dims[s1], dims[s2], dims[s3], dims[s4]);
                let (o1, o2, o3, o4) = (offs[s1], offs[s2], offs[s3], offs[s4]);
                for a in 0..n1 {
                    let mu = o1 + a;
                    for b in 0..n2 {
                        let nu = o2 + b;
                        for cc in 0..n3 {
                            let la = o3 + cc;
                            for dd in 0..n4 {
                                let sg = o4 + dd;
                                let val = q[((a * n2 + b) * n3 + cc) * n4 + dd];
                                let idx = [
                                    ((mu * nbas + nu) * nbas + la) * nbas + sg,
                                    ((nu * nbas + mu) * nbas + la) * nbas + sg,
                                    ((mu * nbas + nu) * nbas + sg) * nbas + la,
                                    ((nu * nbas + mu) * nbas + sg) * nbas + la,
                                    ((la * nbas + sg) * nbas + mu) * nbas + nu,
                                    ((sg * nbas + la) * nbas + mu) * nbas + nu,
                                    ((la * nbas + sg) * nbas + nu) * nbas + mu,
                                    ((sg * nbas + la) * nbas + nu) * nbas + mu,
                                ];
                                for &k in &idx {
                                    // SAFETY: k < nb2*nb2 by construction; this task's
                                    // write set is disjoint from every other task's.
                                    unsafe { *ao_base.add(k) = val };
                                }
                            }
                        }
                    }
                }
            }
        });
    });
    Ok(ao)
}

/// Compute the canonical MP2 correlation energy using full 4-center ERIs.
///
/// This is an O(N^5) reference implementation for cross-validating RI-MP2.
/// Not intended for production use on large molecules.
pub fn canonical_mp2(
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    frozen_core: usize,
) -> Result<f64, FerricError> {
    let nbas = prep.nbasis();
    let nelec = mol.nelec() as usize;
    let nocc_total = nelec / 2;
    let nocc = active_occ(nocc_total, frozen_core)?;
    let first_occ = frozen_core;
    let nvir = nbas - nocc_total;
    let eps = rhf.eps_r();
    let c = rhf.mos_r();

    let nov = nocc * nvir;
    let nb2 = nbas * nbas;

    // Fail-fast size guard. The dense AO buffer (nbas^4) now dominates: it is
    // the single largest allocation and, for any system where this reference
    // path is usable, is far bigger than the MO-side buffers. Count it plus the
    // largest transform intermediate (nbas^3 * nocc) and the final (nov)^2.
    let peak = nb2
        .saturating_mul(nb2)
        .saturating_add(nb2.saturating_mul(nbas).saturating_mul(nocc))
        .saturating_add(nov.saturating_mul(nov))
        .saturating_mul(8);
    ferric_core::memory::check_alloc(
        &format!(
            "canonical MP2 (nocc={nocc}, nvir={nvir}; dense AO ERI nbas^4 (nbas={nbas}), MO-ERI (nov={nov})²)"
        ),
        peak,
        ferric_core::memory::resolve_budget_bytes(None),
    )?;

    // ---- Step 1: dense AO ERI (mu nu|la sg), 8-fold permutational symmetry. ----
    // Shared with the exact-integral reference paths via `dense_ao_eri` so there is
    // exactly one implementation of the quartet loop + symmetry scatter.
    //
    // libint2 quartet evaluation is ~99% of this function's wall time (the four
    // quarter transforms are ~1%).
    let ao = dense_ao_eri(prep, op)?;

    // ---- Steps 2-5: four quarter transforms, one GEMM each. ----------------
    // C_occ is columns [first_occ, first_occ+nocc); C_vir is [nocc_total, nbas).
    let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., nocc_total..nbas]).to_owned();

    // (mu, nu la sg) -> (i, nu la sg):  C_occ^T * ao
    let ao_m = ArrayView2::from_shape((nbas, nb2 * nbas), &ao)
        .map_err(|e| FerricError::General(format!("canonical MP2 AO reshape: {e}")))?;
    let t1 = c_occ.t().dot(&ao_m); // (nocc, nu la sg)
    drop(ao);

    // (i nu, la sg) -> transform nu to a. Move nu to the front: for each i,
    // reshape row i as (nbas, nb2) and left-multiply by C_vir^T.
    let mut t2 = Array2::<f64>::zeros((nocc * nvir, nb2)); // (i a, la sg)
    for i in 0..nocc {
        let row = t1.row(i);
        let m = ArrayView2::from_shape((nbas, nb2), row.as_slice().ok_or_else(|| {
            FerricError::General("canonical MP2: t1 row not contiguous".into())
        })?)
        .map_err(|e| FerricError::General(format!("canonical MP2 t1 reshape: {e}")))?;
        let ia = c_vir.t().dot(&m); // (nvir, la sg)
        t2.slice_mut(ndarray::s![i * nvir..(i + 1) * nvir, ..]).assign(&ia);
    }
    drop(t1);

    // (ia, la sg) -> (ia, j sg):  for the trailing pair, contract la with C_occ.
    // t2 rows are (la, sg) matrices; (ia, la sg) * C_occ over la needs a
    // per-row GEMM, but we can instead do it as one GEMM by viewing the
    // trailing index: reshape to (ia*nbas, nbas) is (ia la, sg) -- contract sg
    // with C_vir first (one big GEMM), then la with C_occ.
    let t2_m = ArrayView2::from_shape((nov * nbas, nbas), t2.as_slice().ok_or_else(|| {
        FerricError::General("canonical MP2: t2 not contiguous".into())
    })?)
    .map_err(|e| FerricError::General(format!("canonical MP2 t2 reshape: {e}")))?;
    let t3 = t2_m.dot(&c_vir); // (ia la, b)
    drop(t2);

    // (ia la, b) -> (ia, j b): contract la with C_occ, per ia row.
    let mut mo = Array2::<f64>::zeros((nov, nocc * nvir)); // (ia, j b)
    let t3_s = t3.as_slice().ok_or_else(|| {
        FerricError::General("canonical MP2: t3 not contiguous".into())
    })?;
    for ia in 0..nov {
        let m = ArrayView2::from_shape((nbas, nvir), &t3_s[ia * nbas * nvir..(ia + 1) * nbas * nvir])
            .map_err(|e| FerricError::General(format!("canonical MP2 t3 reshape: {e}")))?;
        let jb = c_occ.t().dot(&m); // (nocc, nvir)
        mo.slice_mut(ndarray::s![ia, ..])
            .assign(&jb.into_shape_with_order(nocc * nvir).map_err(|e| {
                FerricError::General(format!("canonical MP2 jb reshape: {e}"))
            })?);
    }
    drop(t3);

    // ---- Energy: fused single pass over (i,a,j,b). -------------------------
    // E = sum_ijab (ia|jb) * [2(ia|jb) - (ib|ja)] / D_ijab.
    // The old code materialised five (nov)^2 temporaries (a reshaped copy, two
    // axis permutations, 2V, and t) purely to feed an elementwise `ijab,ijab->`
    // contraction that reduces to a scalar. That is pure memory traffic; fold
    // it into one pass with no intermediates.
    let mo_s = mo.as_slice().ok_or_else(|| {
        FerricError::General("canonical MP2: mo not contiguous".into())
    })?;
    // Per-i partials are collected in index order and summed serially, so the
    // result is bit-identical run to run regardless of how rayon schedules the
    // work (a bare `.sum()` on a parallel iterator reduces in completion order
    // -- see the repo's GEMM summation-order convention: pick an order, pin it).
    let partials: Vec<f64> = (0..nocc)
        .into_par_iter()
        .map(|i| {
            let ei = eps[first_occ + i];
            let mut acc = 0.0;
            for a in 0..nvir {
                let ea = eps[nocc_total + a];
                let row_ia = (i * nvir + a) * nov;
                for j in 0..nocc {
                    let ej = eps[first_occ + j];
                    for b in 0..nvir {
                        let eb = eps[nocc_total + b];
                        // (ia|jb) and the exchange partner (ib|ja)
                        let v_iajb = mo_s[row_ia + j * nvir + b];
                        let v_ibja = mo_s[(i * nvir + b) * nov + j * nvir + a];
                        let d = ei + ej - ea - eb;
                        acc += v_iajb * (2.0 * v_iajb - v_ibja) / d;
                    }
                }
            }
            acc
        })
        .collect();
    let e_mp2: f64 = partials.iter().sum();
    Ok(e_mp2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use std::sync::Mutex;

    // FERRIC_MEM_BUDGET_GB is process-global; serialize any test that sets it
    // (blas_threads.rs / memory.rs pattern).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // Baseline canonical MP2 energy for H2/cc-pVDZ from the pre-port scalar loop.
    const CANONICAL_MP2_H2_CCPVDZ: f64 = -0.026371557616130;

    #[test]
    fn canonical_mp2_fails_fast_under_tiny_env_budget() {
        use ferric_core::parallel::ParallelContext;
        use ferric_integrals::operator::Operator;
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::parse_xyz("2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n", 0, 1).unwrap();
        let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let rhf = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        // Tiny env budget → the guard (which resolves via None) must fire.
        std::env::set_var("FERRIC_MEM_BUDGET_GB", "0.000001");
        let res = canonical_mp2(&mol, &prep, op, &rhf, 0);
        std::env::remove_var("FERRIC_MEM_BUDGET_GB");
        let err = res.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("canonical MP2") && msg.contains("budget is"), "unexpected: {msg}");
    }

    #[test]
    fn canonical_mp2_energy_via_einsum_matches_scalar() {
        use ferric_core::parallel::ParallelContext;
        use ferric_integrals::operator::Operator;

        // canonical_mp2 reads the process-global FERRIC_MEM_BUDGET_GB
        // internally (via resolve_budget_bytes(None)); hold ENV_LOCK so this
        // can't observe canonical_mp2_fails_fast_under_tiny_env_budget's
        // temporary tiny-budget mutation under cargo test's default
        // parallelism (found 2026-07-18, same class of bug as gto_eval.rs).
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xyz = "2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let op = Operator::coulomb();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let ctx = ParallelContext::default();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default()).unwrap();
        let e = canonical_mp2(&mol, &prep, op, &rhf, 0).unwrap();
        assert!((e - CANONICAL_MP2_H2_CCPVDZ).abs() < 1e-10, "got {e:.15}");
    }

    #[test]
    fn test_canonical_mp2_h2_sto3g() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &prep,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        let e_corr = canonical_mp2(&mol, &prep, op, &rhf, 0).unwrap();
        eprintln!(
            "H2/STO-3G: RHF={:.10}, canonical MP2 corr={:.10}",
            rhf.energy, e_corr
        );
        // PySCF: -0.0131380736
        assert!(
            (e_corr - (-0.0131380736)).abs() < 1e-7,
            "H2/STO-3G MP2 corr: {e_corr:.10}"
        );
    }

    #[test]
    fn test_canonical_vs_ri_mp2_h2_ccpvdz() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &prep,
            op,
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        let e_canonical = canonical_mp2(&mol, &prep, op, &rhf, 0).unwrap();

        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let ri_result = crate::rimp2::ri_mp2(
            &mol,
            &prep,
            &dfbs,
            op,
            &rhf,
            &crate::rimp2::RiMp2Config::default(),
        )
        .unwrap();

        eprintln!(
            "H2/cc-pVDZ: canonical={:.10}, RI={:.10}, diff={:.2e}",
            e_canonical,
            ri_result.mp2_corr,
            (e_canonical - ri_result.mp2_corr).abs()
        );

        let diff = (e_canonical - ri_result.mp2_corr).abs();
        assert!(
            diff < 1e-4,
            "canonical={e_canonical:.10} ri={:.10} diff={diff:.2e}",
            ri_result.mp2_corr
        );
    }
}
