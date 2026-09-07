//! Schwarz upper-bound screening matrix for shell-pair integrals.

use crate::basis_bridge::PreparedBasis;
use crate::engine::Engine;
use crate::operator::{Operator, OperatorKind};
use ferric_core::FerricError;
use ndarray::Array2;

/// Below this many shells, run the single-engine serial FFI loop — avoids
/// rayon/engine-construction overhead for free-atom/tiny-basis jobs (same
/// rationale as [`PAR_AUX_SHELL_THRESHOLD`] below and `oneelectron.rs`).
const PAR_SCHWARZ_SHELL_THRESHOLD: usize = 64;

/// Engine precision used to build the SCREENING TABLE, as distinct from the
/// threshold the table is later used to enforce.
///
/// A Schwarz bound's whole contract is that it never UNDERestimates: `Q(ij) *
/// Q(kl) >= |(ij|kl)|` is what makes discarding a quartet safe. libint2 applies
/// its own internal prescreening at the precision it is constructed with, and
/// declines to compute a shell quartet whose result would fall below it —
/// correct, documented behaviour for an integral engine, but fatal for a table
/// builder, because [`Engine::compute_quartet`] then returns `None` and the
/// pair is recorded as `Q = 0.0`. A stored zero on a pair with a nonzero true
/// `(ij|ij)` underestimates by construction, and since `SignificantPairs::build`
/// compares with a strict `estimate(..) > threshold`, such a pair becomes
/// unreachable at EVERY threshold — including 0, which destroys the trivial
/// limit in which screening must do nothing at all.
///
/// MEASURED before this constant existed (alkane_8/cc-pVDZ, 102 shells):
/// 1049 of 5253 unique pairs stored `Q == 0.0` while only 121 are genuinely
/// zero, and the smallest nonzero stored Q was 4.632e-7 — i.e. exactly
/// `sqrt(~1e-13)`, the fingerprint of the engine's precision cliff rather than
/// of any property of the molecule. alkane_16 zeroed 9977 of 19701.
///
/// The remedy is what every production code does: build the table TIGHTER than
/// the threshold it enforces. PySCF builds `q_cond` at `direct_scf_tol**2` with
/// a `1e-100` floor; Molpro, NWChem, Q-Chem, ORCA and Psi4 (via its own
/// `eps * thresh`) do the equivalent. ferric previously used one hardcoded
/// `1e-14` for BOTH roles, which is the outlier.
///
/// `0.0` disables libint2's internal prescreening outright, which is the
/// tightest available table. MEASURED one-time cost, both arms run back to back
/// on an otherwise-idle box (release build, best of five,
/// OPENBLAS_NUM_THREADS=1) so the comparison is like for like:
///
/// ```text
///                    1e-14      0.0      ratio
/// alkane_8/cc-pVDZ   0.0262 s   0.0412 s  1.6x
/// alkane_16/cc-pVDZ  0.0452 s   0.1328 s  2.9x
/// ```
///
/// That is a setup cost paid once per Fock-builder construction, not per SCF
/// iteration, and it buys an alkane_8 RHF energy that moves from 1.46e-4 Ha off
/// PySCF to 6.96e-9 Ha off — tens of milliseconds against five orders of
/// magnitude of accuracy. The ratio grows with system size (the zeroed fraction
/// did too: 20% of pairs on alkane_8, 51% on alkane_16), so it is worth
/// re-measuring if it ever shows up in a profile at much larger N.
///
/// A cheaper `thresh * thresh` table (PySCF's actual policy) would also restore
/// the invariant and would scale better; `0.0` is chosen because the measured
/// cost is small enough that the extra knob is not worth its failure modes —
/// notably that it re-couples the table to a threshold the caller can change.
const SCHWARZ_TABLE_PRECISION: f64 = 0.0;

/// Floor applied to every stored Q so that no table entry is ever exactly zero.
///
/// Even with prescreening disabled a genuinely vanishing `(ij|ij)` (or one that
/// underflows to zero in double precision) would still be stored as `0.0` and
/// hit the same strict-`>` unreachability. Flooring costs nothing — `1e-100`
/// squared is `1e-200`, still finite and ~180 orders of magnitude below any
/// threshold anyone would enforce, so it never makes a negligible quartet
/// survive a real screen — but it makes `estimate(..) > 0.0` true everywhere,
/// which is precisely the trivial-limit guarantee. Same value and same
/// rationale as PySCF's `q_cond` floor.
const SCHWARZ_Q_FLOOR: f64 = 1e-100;

/// Q(i,j) = sqrt(max_{a,b} |(ab|ab)|) over the functions of shell pair (i,j),
/// from one computed (ij|ij) quartet block, floored at [`SCHWARZ_Q_FLOOR`] so
/// that the returned bound is never exactly zero.
///
/// The `None` arm (engine returned no block at all) is retained for safety but
/// is unreachable in normal use now that the table engines are built at
/// [`SCHWARZ_TABLE_PRECISION`]; it too returns the floor rather than 0.0, so a
/// bound that cannot be evaluated still does not silently underestimate.
fn schwarz_pair(eng: &mut Engine, prep: &PreparedBasis, i: usize, j: usize) -> f64 {
    let dims = prep.shell_dims();
    let (n1, n2) = (dims[i], dims[j]);
    let maxv = match eng.compute_quartet(prep, i, j, i, j) {
        Some(block) => {
            // The (ab|ab) magnitudes live on the generalized diagonal of the
            // (n1·n2)×(n1·n2) block: index ((a·n2+b)·n1+a)·n2+b.
            let mut m = 0.0f64;
            for a in 0..n1 {
                for b in 0..n2 {
                    let v = block[((a * n2 + b) * n1 + a) * n2 + b].abs();
                    if v > m {
                        m = v;
                    }
                }
            }
            m
        }
        None => 0.0,
    };
    maxv.sqrt().max(SCHWARZ_Q_FLOOR)
}

/// Compute the Schwarz screening matrix Q(i,j) = sqrt(|(ij|ij)|) for all shell pairs.
///
/// Q(i,j) * Q(k,l) provides an upper bound on |(ij|kl)|, enabling integral screening.
///
/// Parallelized over upper-triangle shell pairs once `nsh` clears
/// `PAR_SCHWARZ_SHELL_THRESHOLD` — the historical `scf_compute_schwarz` FFI
/// call ran the whole O(nsh²) diagonal-quartet loop serially on one engine,
/// which dominated setup on large direct jobs. Each rayon worker builds its own
/// [`Engine`] via `map_init` (construction is serialized behind a global ctor
/// mutex — see the ROW_BLOCK note in the body for why granularity matters).
/// Every pair is computed by the same
/// `scf_compute_eri_quartet` kernel with unit coefficient as the serial loop,
/// and each (i, j) is written exactly once, so the result is bit-identical to
/// the serial loop regardless of thread count or block size.
pub fn schwarz(op: Operator, prep: &PreparedBasis) -> Result<Array2<f64>, FerricError> {
    match op.kind {
        OperatorKind::Coulomb | OperatorKind::ErfCoulomb | OperatorKind::ErfcCoulomb => {}
        _ => {
            return Err(FerricError::Libint(format!(
                "operator {:?} not implemented",
                op.kind
            )))
        }
    }
    let nsh = prep.nshells();
    let mut qmat = Array2::zeros((nsh, nsh));

    if nsh < PAR_SCHWARZ_SHELL_THRESHOLD {
        let mut eng = Engine::new_2e(op, prep, SCHWARZ_TABLE_PRECISION)?;
        for i in 0..nsh {
            for j in 0..=i {
                let q = schwarz_pair(&mut eng, prep, i, j);
                qmat[(i, j)] = q;
                qmat[(j, i)] = q;
            }
        }
        return Ok(qmat);
    }

    use rayon::prelude::*;

    // Validate engine construction once up front so worker-side construction
    // can't fail (mirrors schwarz3_aux below).
    Engine::new_2e(op, prep, SCHWARZ_TABLE_PRECISION)?;

    // Parallelize over ROW BLOCKS, not individual shell pairs. libint2 engine
    // construction is expensive AND serialized behind a global ctor mutex in the
    // shim, and rayon invokes a `map_init` closure once per work-CHUNK, not once
    // per thread. The previous version made each of the nsh(nsh+1)/2 shell pairs
    // (31,878 at benzene/aug-cc-pVTZ) its own work item, so dozens of engine
    // constructions queued on that mutex: MEASURED 123 ms at RAYON=1 vs 3211 ms
    // at RAYON=12 — parallelism made this function 26x SLOWER. Row blocks keep
    // the item count at ceil(nsh/ROW_BLOCK) (8 here) so the init closure fires
    // about once per thread, while each item still carries real integral work.
    // Sweep at 252 shells / RAYON=12: block 8 -> 211 ms, 16 -> 144, 32 -> 75,
    // 64 -> 86; serial baseline 124 ms.
    //
    // NOT an `EnginePool` (engine_pool.rs), despite that module existing for
    // exactly this ctor-mutex pathology: it eagerly builds `nthreads + 1`
    // engines, which costs MORE than this whole loop when there are fewer
    // blocks than threads (measured: 125.7 ms of pool construction, 190 ms
    // total, vs 75 ms here). The pool wins where engines are reused across many
    // chunks and many SCF iterations (the direct Fock builders); a one-shot
    // setup loop with ~nthreads items is the opposite regime.
    const ROW_BLOCK: usize = 32;
    let row_blocks: Vec<(usize, usize)> = (0..nsh)
        .step_by(ROW_BLOCK)
        .map(|i0| (i0, (i0 + ROW_BLOCK).min(nsh)))
        .collect();
    // Each block owns its rows exclusively, so the (i, j) writes never overlap;
    // collect per-block triangles and scatter serially to keep `qmat` unshared.
    let blocks: Vec<Vec<(usize, usize, f64)>> = row_blocks
        .par_iter()
        .map_init(
            || Engine::new_2e(op, prep, SCHWARZ_TABLE_PRECISION).expect("2e engine (pre-validated)"),
            |eng, &(i0, i1)| {
                let mut out = Vec::with_capacity((i1 - i0) * (i1 + 1));
                for i in i0..i1 {
                    for j in 0..=i {
                        out.push((i, j, schwarz_pair(eng, prep, i, j)));
                    }
                }
                out
            },
        )
        .collect();
    for block in &blocks {
        for &(i, j, q) in block {
            qmat[(i, j)] = q;
            qmat[(j, i)] = q;
        }
    }
    Ok(qmat)
}

/// Below this many aux shells, run the loop serially — avoids
/// rayon/engine-construction overhead for free-atom/tiny-basis jobs.
const PAR_AUX_SHELL_THRESHOLD: usize = 64;

/// Per-shell Schwarz bound for an auxiliary (density-fitting) basis:
/// `Q3[P] = sqrt(max_a |(P_a | P_a)|)` over the functions a in aux shell P.
///
/// Combined with the orbital-pair matrix Q(μ,ν) = sqrt(|(μν|μν)|), this gives
/// the rigorous 3-index Cauchy–Schwarz bound
///   `|(P | μν)|  ≤  Q3[P] · Q(μ,ν)`
/// which lets `eri3_tensor_screened` skip shell triples whose contribution
/// is below threshold without computing them.
///
/// Built at [`SCHWARZ_TABLE_PRECISION`] and floored at [`SCHWARZ_Q_FLOOR`] for
/// the same reason as the orbital-pair table above: this is a screening bound,
/// so it must never underestimate. (Note `compute_eri2` returns an EMPTY slice
/// rather than an `Option` when libint2 prescreens a block away, so that case
/// would index out of bounds and PANIC rather than silently store a zero —
/// disabling prescreening removes that path too, though the floor is what
/// guarantees the invariant for a genuinely vanishing diagonal.)
///
/// In practice this arm was never the one biting — MEASURED on
/// alkane_8 and alkane_16 with cc-pVDZ-RI, the pre-fix aux table stored ZERO
/// zeros and its smallest nonzero entry was 4.475e-1, because a 2-centre
/// `(P|P)` diagonal over a normalized aux shell is O(1) and nowhere near the
/// engine's precision cliff. It is changed for invariant consistency rather
/// than to repair an observed failure, and the measured cost is 0.004 s.
///
/// Parallelized over `p` once `nsh` clears `PAR_AUX_SHELL_THRESHOLD`: each
/// rayon worker builds its own `Engine` via `for_each_init` (never per-item —
/// construction runs under a global ctor mutex). Each iteration writes only
/// `q3[p]` — a single distinct index per task, so the write set is trivially
/// disjoint across workers; `into_par_iter().map().collect()` is
/// order-preserving, giving a `Vec` bit-identical to the serial loop.
pub fn schwarz3_aux(op: Operator, dfbs: &PreparedBasis) -> Result<Vec<f64>, FerricError> {
    let nsh = dfbs.nshells();
    let dims = dfbs.shell_dims();

    if nsh < PAR_AUX_SHELL_THRESHOLD {
        let mut eng = Engine::new_2center(op, dfbs, SCHWARZ_TABLE_PRECISION)?;
        let mut q3 = vec![0.0f64; nsh];
        for p in 0..nsh {
            let block = eng.compute_eri2(dfbs, p, p);
            let np = dims[p];
            // (P_a | P_a) lives on the diagonal of the np×np block.
            let mut maxv = 0.0f64;
            for a in 0..np {
                let v = block[a * np + a].abs();
                if v > maxv {
                    maxv = v;
                }
            }
            q3[p] = maxv.sqrt().max(SCHWARZ_Q_FLOOR);
        }
        return Ok(q3);
    }

    use rayon::prelude::*;
    Engine::new_2center(op, dfbs, SCHWARZ_TABLE_PRECISION)?;
    let q3: Vec<f64> = (0..nsh)
        .into_par_iter()
        .map_init(
            || {
                Engine::new_2center(op, dfbs, SCHWARZ_TABLE_PRECISION)
                    .expect("2-center engine (pre-validated)")
            },
            |eng, p| {
                let block = eng.compute_eri2(dfbs, p, p);
                let np = dims[p];
                let mut maxv = 0.0f64;
                for a in 0..np {
                    let v = block[a * np + a].abs();
                    if v > maxv {
                        maxv = v;
                    }
                }
                maxv.sqrt().max(SCHWARZ_Q_FLOOR)
            },
        )
        .collect();
    Ok(q3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_bridge::PreparedBasis;
    use crate::operator::Operator;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    #[test]
    fn test_schwarz_erfc_bounded_by_coulomb() {
        // erfc(ωr)/r ≤ 1/r pointwise, so |(ij|ij)_erfc| ≤ |(ij|ij)_Coulomb|
        // and therefore Q_erfc(i,j) ≤ Q_Coulomb(i,j) for every shell pair.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let q_c = schwarz(Operator::coulomb(), &prep).unwrap();
        let q_e = schwarz(Operator::erfc(0.222), &prep).unwrap();
        let nsh = prep.nshells();
        for i in 0..nsh {
            for j in 0..nsh {
                assert!(q_e[(i, j)] >= 0.0, "Q_erfc[{i},{j}] < 0");
                assert!(
                    q_e[(i, j)] <= q_c[(i, j)] + 1e-12,
                    "Q_erfc[{i},{j}]={} exceeds Q_Coulomb={}",
                    q_e[(i, j)], q_c[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_schwarz_symmetric() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let q = schwarz(Operator::coulomb(), &prep).unwrap();
        let nsh = prep.nshells();
        for i in 0..nsh {
            for j in 0..nsh {
                assert!(
                    (q[(i, j)] - q[(j, i)]).abs() < 1e-12,
                    "Q not symmetric at ({i},{j})"
                );
                assert!(q[(i, j)] >= 0.0, "Q[{i},{j}] < 0");
            }
        }
    }

    /// Serial reference for `schwarz3_aux` (pre-parallelization implementation,
    /// kept verbatim).
    fn schwarz3_aux_serial(op: Operator, dfbs: &PreparedBasis) -> Vec<f64> {
        let nsh = dfbs.nshells();
        let dims = dfbs.shell_dims();
        let mut eng = Engine::new_2center(op, dfbs, SCHWARZ_TABLE_PRECISION).unwrap();
        let mut q3 = vec![0.0f64; nsh];
        for p in 0..nsh {
            let block = eng.compute_eri2(dfbs, p, p);
            let np = dims[p];
            let mut maxv = 0.0f64;
            for a in 0..np {
                let v = block[a * np + a].abs();
                if v > maxv {
                    maxv = v;
                }
            }
            q3[p] = maxv.sqrt().max(SCHWARZ_Q_FLOOR);
        }
        q3
    }

    /// Serial reference for `schwarz` (single engine, pair loop — the exact
    /// small-system path), used to prove the parallel path is bit-identical.
    fn schwarz_serial(op: Operator, prep: &PreparedBasis) -> Array2<f64> {
        let nsh = prep.nshells();
        let mut eng = Engine::new_2e(op, prep, SCHWARZ_TABLE_PRECISION).unwrap();
        let mut qmat = Array2::zeros((nsh, nsh));
        for i in 0..nsh {
            for j in 0..=i {
                let q = super::schwarz_pair(&mut eng, prep, i, j);
                qmat[(i, j)] = q;
                qmat[(j, i)] = q;
            }
        }
        qmat
    }

    /// INVARIANT: the Schwarz table never stores an exact zero.
    ///
    /// A Schwarz bound's contract is that it never underestimates
    /// (arXiv:2302.11307). `SignificantPairs::build` compares with a strict
    /// `estimate(..) > threshold`, so a stored `Q = 0.0` makes that shell pair
    /// unreachable at every threshold INCLUDING 0 — which is exactly what
    /// destroyed the trivial limit before [`SCHWARZ_TABLE_PRECISION`] and
    /// [`SCHWARZ_Q_FLOOR`] existed.
    ///
    /// alkane_8/cc-pVDZ is the system the defect was characterised on: it
    /// stored 1049 zeros out of 5253 unique pairs (of which only 121 are
    /// genuinely zero), and its smallest nonzero entry was 4.632e-7 — the
    /// sqrt of the engine's 1e-14 precision cliff rather than any property of
    /// the molecule. Water/CH4 show NOTHING here, which is why the older tests
    /// passed while proving nothing; the molecule has to be big enough for
    /// distant shell pairs to fall under the cliff.
    ///
    /// This asserts the invariant at its source rather than through a
    /// downstream symptom, and it is the standing replacement for the
    /// deleted `schwarz_table_stores_zero_for_nonzero_pairs_alkane_8` anchor.
    #[test]
    fn schwarz_table_never_stores_a_zero_alkane_8() {
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_8.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let nsh = prep.nshells();
        assert_eq!(
            nsh, 102,
            "alkane_8/cc-pVDZ should give 102 shells; the counts quoted here were measured on \
             that decomposition"
        );

        for op in [Operator::coulomb(), Operator::erfc(0.222)] {
            let q = schwarz(op, &prep).unwrap();
            let zeros = (0..nsh)
                .flat_map(|i| (0..=i).map(move |j| (i, j)))
                .filter(|&(i, j)| q[(i, j)] == 0.0)
                .count();
            assert_eq!(
                zeros, 0,
                "op={op:?}: {zeros} of {} unique shell pairs store Q == 0.0. Such an entry \
                 UNDERestimates a nonzero (ij|ij) and, because SignificantPairs uses a strict \
                 `> threshold`, makes the pair unreachable even at threshold 0 — the bound is \
                 no longer a bound. Check SCHWARZ_TABLE_PRECISION and SCHWARZ_Q_FLOOR.",
                nsh * (nsh + 1) / 2
            );
            // Every entry must also be finite and non-negative: a NaN would
            // compare false against every threshold and silently screen
            // everything away, which is the same failure wearing a disguise.
            for (idx, &v) in q.indexed_iter() {
                assert!(
                    v.is_finite() && v >= SCHWARZ_Q_FLOOR,
                    "op={op:?}: Q{idx:?} = {v} is not a valid bound (finite and >= the floor)"
                );
            }
        }
    }

    /// Same invariant for the auxiliary-basis bound used by the 3-index
    /// screened path. Measured pre-fix, this table stored NO zeros on
    /// alkane_8/cc-pVDZ-RI (smallest entry 4.475e-1) because a 2-centre
    /// `(P|P)` diagonal is O(1); the assertion guards the contract rather than
    /// repairing an observed failure, and would catch a future aux basis or
    /// operator whose diagonals do approach the cliff.
    #[test]
    fn schwarz3_aux_never_stores_a_zero_alkane_8() {
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_8.xyz").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        for op in [Operator::coulomb(), Operator::erfc(0.222)] {
            let q3 = schwarz3_aux(op, &dfbs).unwrap();
            for (p, &v) in q3.iter().enumerate() {
                assert!(
                    v.is_finite() && v >= SCHWARZ_Q_FLOOR,
                    "op={op:?}: Q3[{p}] = {v} is not a valid bound (finite and >= the floor)"
                );
            }
        }
    }

    #[test]
    fn test_schwarz_parallel_bitidentical_to_serial() {
        // alkane_6/cc-pVDZ clears PAR_SCHWARZ_SHELL_THRESHOLD (64 shells) so
        // schwarz() takes the parallel path; compare bitwise vs the serial loop.
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_6.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        assert!(
            prep.nshells() >= PAR_SCHWARZ_SHELL_THRESHOLD,
            "test basis too small to exercise the parallel path: {} shells",
            prep.nshells()
        );
        for op in [Operator::coulomb(), Operator::erfc(0.222)] {
            let par = schwarz(op, &prep).unwrap();
            let ser = schwarz_serial(op, &prep);
            let n_diff = par
                .iter()
                .zip(ser.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(n_diff, 0, "schwarz: {n_diff} elements differ bitwise (op={op:?})");
        }
    }

    #[test]
    fn test_schwarz3_aux_bitidentical_to_serial() {
        // alkane_6/cc-pVDZ-RI clears PAR_AUX_SHELL_THRESHOLD (64 aux shells).
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_6.xyz").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        assert!(dfbs.nshells() >= 64,
            "test aux basis too small to exercise the parallel path: {} shells", dfbs.nshells());
        for op in [Operator::coulomb(), Operator::erfc(0.222)] {
            let par = schwarz3_aux(op, &dfbs).unwrap();
            let ser = schwarz3_aux_serial(op, &dfbs);
            assert_eq!(par.len(), ser.len());
            let n_diff = par.iter().zip(ser.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(n_diff, 0, "schwarz3_aux: {n_diff} elements differ bitwise (op={op:?})");
        }
    }
}
