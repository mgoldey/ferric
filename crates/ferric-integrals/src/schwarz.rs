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

/// Q(i,j) = sqrt(max_{a,b} |(ab|ab)|) over the functions of shell pair (i,j),
/// from one computed (ij|ij) quartet block. `None` (engine screened the
/// quartet to zero) maps to 0.0, matching the shim's null-result branch.
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
    maxv.sqrt()
}

/// Compute the Schwarz screening matrix Q(i,j) = sqrt(|(ij|ij)|) for all shell pairs.
///
/// Q(i,j) * Q(k,l) provides an upper bound on |(ij|kl)|, enabling integral screening.
///
/// Parallelized over upper-triangle shell pairs once `nsh` clears
/// [`PAR_SCHWARZ_SHELL_THRESHOLD`] — the historical `scf_compute_schwarz` FFI
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
        let mut eng = Engine::new_2e(op, prep, 1e-14)?;
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
    Engine::new_2e(op, prep, 1e-14)?;

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
            || Engine::new_2e(op, prep, 1e-14).expect("2e engine (pre-validated)"),
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
/// Q3[P] = sqrt(max_a |(P_a | P_a)|) over the functions a in aux shell P.
///
/// Combined with the orbital-pair matrix Q(μ,ν) = sqrt(|(μν|μν)|), this gives
/// the rigorous 3-index Cauchy–Schwarz bound
///   |(P | μν)|  ≤  Q3[P] · Q(μ,ν)
/// which lets `eri3_tensor_screened` skip shell triples whose contribution
/// is below threshold without computing them.
///
/// Parallelized over `p` once `nsh` clears [`PAR_AUX_SHELL_THRESHOLD`]: each
/// rayon worker builds its own `Engine` via `for_each_init` (never per-item —
/// construction runs under a global ctor mutex). Each iteration writes only
/// `q3[p]` — a single distinct index per task, so the write set is trivially
/// disjoint across workers; `into_par_iter().map().collect()` is
/// order-preserving, giving a `Vec` bit-identical to the serial loop.
pub fn schwarz3_aux(op: Operator, dfbs: &PreparedBasis) -> Result<Vec<f64>, FerricError> {
    let nsh = dfbs.nshells();
    let dims = dfbs.shell_dims();

    if nsh < PAR_AUX_SHELL_THRESHOLD {
        let mut eng = Engine::new_2center(op, dfbs, 1e-14)?;
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
            q3[p] = maxv.sqrt();
        }
        return Ok(q3);
    }

    use rayon::prelude::*;
    Engine::new_2center(op, dfbs, 1e-14)?;
    let q3: Vec<f64> = (0..nsh)
        .into_par_iter()
        .map_init(
            || Engine::new_2center(op, dfbs, 1e-14).expect("2-center engine (pre-validated)"),
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
                maxv.sqrt()
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
        let mut eng = Engine::new_2center(op, dfbs, 1e-14).unwrap();
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
            q3[p] = maxv.sqrt();
        }
        q3
    }

    /// Serial reference for `schwarz` (single engine, pair loop — the exact
    /// small-system path), used to prove the parallel path is bit-identical.
    fn schwarz_serial(op: Operator, prep: &PreparedBasis) -> Array2<f64> {
        let nsh = prep.nshells();
        let mut eng = Engine::new_2e(op, prep, 1e-14).unwrap();
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
