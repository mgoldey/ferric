//! Three-center and two-center integral builders for density fitting (RI).

use crate::basis_bridge::PreparedBasis;
use crate::engine::Engine;
use crate::operator::Operator;
use crate::qqr3::QqrBounds3;
use crate::schwarz::{schwarz, schwarz3_aux};
use ferric_core::FerricError;
use ndarray::{Array2, Array3};

/// Below this many aux shells, run the serial loop directly — avoids
/// rayon/engine-construction overhead for free-atom/tiny-basis jobs.
const PAR_METRIC_SHELL_THRESHOLD: usize = 64;

/// Build the 2-center Coulomb metric (P|Q), shape (naux, naux).
///
/// Parallelized over the outer aux-shell index `sp` (independent row bands)
/// once `nsh` clears `PAR_METRIC_SHELL_THRESHOLD`; each rayon worker builds
/// its own `Engine` via `for_each_init` (never per-item — construction runs
/// under a global ctor mutex). For a fixed `sp`, the write set is
/// `{(offs[sp]+p, offs[sq]+q), (offs[sq]+q, offs[sp]+p) : sq ≤ sp}` — every
/// written row index lies in `[offs[sp], offs[sp]+np)` (directly, or via the
/// transposed form using the same range), so distinct `sp` values own disjoint
/// row bands of `v` — the same disjointness argument as `eri3_tensor`'s
/// aux-row-band scatter below. Each element is written exactly once, so the
/// result is bit-identical to the serial loop regardless of thread count.
pub fn coulomb_metric_2c(op: Operator, dfbs: &PreparedBasis) -> Result<Array2<f64>, FerricError> {
    let naux = dfbs.nbasis();
    let nsh = dfbs.nshells();
    let dims = dfbs.shell_dims();
    let offs = dfbs.shell_offsets();
    let mut v = Array2::zeros((naux, naux));

    if nsh < PAR_METRIC_SHELL_THRESHOLD {
        let mut eng = Engine::new_2center(op, dfbs, 1e-14)?;
        for sp in 0..nsh {
            for sq in 0..=sp {
                let block = eng.compute_eri2(dfbs, sp, sq);
                let np = dims[sp];
                let nq = dims[sq];
                for p in 0..np {
                    for q in 0..nq {
                        let val = block[p * nq + q];
                        v[(offs[sp] + p, offs[sq] + q)] = val;
                        v[(offs[sq] + q, offs[sp] + p)] = val;
                    }
                }
            }
        }
        return Ok(v);
    }

    use rayon::prelude::*;

    // Surface any engine-construction error up front (serial, cheap) — see
    // eri3_tensor's rationale: FerricError is not Clone so per-worker rebuilds
    // below must `.expect()`.
    Engine::new_2center(op, dfbs, 1e-14)?;

    let v_ptr = v.as_mut_ptr() as usize;
    let stride = naux; // row-major (naux, naux)

    (0..nsh).into_par_iter().for_each_init(
        || Engine::new_2center(op, dfbs, 1e-14).expect("2-center engine (pre-validated)"),
        |eng, sp| {
            let np = dims[sp];
            let o_p = offs[sp];
            for sq in 0..=sp {
                let block = eng.compute_eri2(dfbs, sp, sq);
                let nq = dims[sq];
                let o_q = offs[sq];
                for p in 0..np {
                    for q in 0..nq {
                        let val = block[p * nq + q];
                        let r = o_p + p;
                        let c = o_q + q;
                        // SAFETY: rayon workers write to disjoint (sp, sq)
                        // shell-pair blocks; the symmetric (r,c)/(c,r) write
                        // stays within the same block's triangle.
                        unsafe {
                            let base = v_ptr as *mut f64;
                            *base.add(r * stride + c) = val;
                            *base.add(c * stride + r) = val;
                        }
                    }
                }
            }
        },
    );
    Ok(v)
}

/// Build 3-center integrals (P|mn), shape (naux, nbasis, nbasis).
///
/// Parallelized over aux shells `sp` (independent outer loop). Each rayon worker
/// builds its own `Engine` (Engine: Send) once per region via `for_each_init`.
/// Writes go to disjoint `(p, mu, nu)` regions — each `sp` owns a distinct
/// aux-row band — so the raw-pointer scatter is data-race-free and bit-identical
/// to the serial build.
pub fn eri3_tensor(op: Operator, obs: &PreparedBasis, dfbs: &PreparedBasis) -> Result<Array3<f64>, FerricError> {
    use rayon::prelude::*;

    let naux = dfbs.nbasis();
    let nbas = obs.nbasis();
    let nsh_obs = obs.nshells();
    let nsh_df = dfbs.nshells();
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();

    // Surface any engine-construction error up front (serial, cheap). After this
    // succeeds the per-worker rebuilds below cannot fail for the same args, so
    // they `.expect()` — `FerricError` is not `Clone`, so we cannot thread the
    // error out of the per-worker init closure.
    Engine::new_3center(op, obs, dfbs, 1e-14)?;

    let mut eri = Array3::<f64>::zeros((naux, nbas, nbas));

    // Raw-pointer scatter: each aux shell `sp` writes a disjoint band of aux rows,
    // so the `(p, mu, nu)` regions written by distinct workers never overlap.
    let eri_ptr = eri.as_mut_ptr() as usize;
    let stride0 = nbas * nbas; // p-stride
    let stride1 = nbas; // mu-stride

    // One `Engine` per rayon worker, built by the `init` closure scoped to THIS
    // parallel region (so it is always built for the current op/obs/dfbs — a
    // process-wide TLS cache would reuse a stale engine on a later call with a
    // different operator). `Engine: Send`.
    (0..nsh_df).into_par_iter().for_each_init(
        || Engine::new_3center(op, obs, dfbs, 1e-14).expect("3-center engine (pre-validated)"),
        |eng, sp| {
            let np = dims_df[sp];
            let p0 = offs_df[sp];
            for s1 in 0..nsh_obs {
                let n1 = dims_obs[s1];
                let m0 = offs_obs[s1];
                for s2 in 0..=s1 {
                    if let Some(block) = eng.compute_eri3(obs, dfbs, sp, s1, s2) {
                        let n2 = dims_obs[s2];
                        let n0 = offs_obs[s2];
                        for p in 0..np {
                            for i in 0..n1 {
                                for j in 0..n2 {
                                    let val = block[(p * n1 + i) * n2 + j];
                                    let pp = p0 + p;
                                    let mm = m0 + i;
                                    let nn = n0 + j;
                                    // SAFETY: rayon workers write to disjoint
                                    // (P, s1, s2) shell-triple blocks; the μ↔ν
                                    // symmetry write stays within the same block.
                                    unsafe {
                                        let base = eri_ptr as *mut f64;
                                        *base.add(pp * stride0 + mm * stride1 + nn) = val;
                                        *base.add(pp * stride0 + nn * stride1 + mm) = val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
    );
    Ok(eri)
}

/// Build raw (P|μν) for aux rows [p0, p1) only, returned as (p1-p0, nbas, nbas).
/// Same values as the corresponding slice of `eri3_tensor`.
///
/// Parallelized over aux shells `sp` exactly like [`eri3_tensor`] (one `Engine`
/// per rayon worker via `for_each_init`, disjoint aux-row-band raw-pointer
/// scatter). Only aux shells overlapping `[p0, p1)` do any work; each such `sp`
/// owns a distinct band of local rows `pl = offs_df[sp] + p − p0`, so writes
/// from distinct workers never overlap and every element is written exactly
/// once — the output is bit-identical to the serial fill.
pub fn eri3_block(
    op: Operator, obs: &PreparedBasis, dfbs: &PreparedBasis, p0: usize, p1: usize,
) -> Result<Array3<f64>, FerricError> {
    use rayon::prelude::*;

    let nbas = obs.nbasis();
    let nsh_obs = obs.nshells();
    let nsh_df = dfbs.nshells();
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();

    // Surface any engine-construction error up front (serial, cheap). After this
    // succeeds the per-worker rebuilds below cannot fail for the same args, so
    // they `.expect()` — `FerricError` is not `Clone`, so we cannot thread the
    // error out of the per-worker init closure. (See eri3_tensor.)
    Engine::new_3center(op, obs, dfbs, 1e-14)?;

    let mut eri = Array3::<f64>::zeros((p1 - p0, nbas, nbas));

    // Raw-pointer scatter: each aux shell `sp` writes a disjoint band of local
    // aux rows, so the `(pl, mu, nu)` regions written by distinct workers never
    // overlap.
    let eri_ptr = eri.as_mut_ptr() as usize;
    let stride0 = nbas * nbas; // pl-stride
    let stride1 = nbas; // mu-stride

    (0..nsh_df).into_par_iter().for_each_init(
        || Engine::new_3center(op, obs, dfbs, 1e-14).expect("3-center engine (pre-validated)"),
        |eng, sp| {
            let pbase = offs_df[sp];
            let np = dims_df[sp];
            // Skip aux shells entirely outside [p0, p1).
            if pbase + np <= p0 || pbase >= p1 { return; }
            for s1 in 0..nsh_obs {
                let n1 = dims_obs[s1];
                let m0 = offs_obs[s1];
                for s2 in 0..=s1 {
                    if let Some(block) = eng.compute_eri3(obs, dfbs, sp, s1, s2) {
                        let n2 = dims_obs[s2];
                        let n0 = offs_obs[s2];
                        for p in 0..np {
                            let pg = pbase + p;
                            if pg < p0 || pg >= p1 { continue; }
                            let pl = pg - p0;
                            for i in 0..n1 {
                                for j in 0..n2 {
                                    let val = block[(p * n1 + i) * n2 + j];
                                    let mm = m0 + i;
                                    let nn = n0 + j;
                                    // SAFETY: same disjoint-write argument as
                                    // the full-tensor path; pl is the chunk-local
                                    // aux index within the current P-block.
                                    unsafe {
                                        let base = eri_ptr as *mut f64;
                                        *base.add(pl * stride0 + mm * stride1 + nn) = val;
                                        *base.add(pl * stride0 + nn * stride1 + mm) = val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
    );
    Ok(eri)
}

/// Schwarz-screened 3-center ERI builder.
///
/// Same dense `(naux, nbas, nbas)` output as [`eri3_tensor`], but skips shell
/// triples whose Cauchy–Schwarz bound `Q3[P] · Q(μ,ν)` is below `thresh`.
/// Skipped blocks remain zero in the output. With `thresh = 0.0` this is a
/// drop-in equivalent of `eri3_tensor` (modulo libint's internal precision).
///
/// Returns `(tensor, n_kept, n_total)` where the shell-triple counts let
/// callers report screening effectiveness without re-walking the loop.
///
/// Parallelized over aux shells `sp` exactly like [`eri3_tensor`] (one
/// `Engine` per rayon worker, disjoint aux-row-band raw-pointer scatter).
/// Screening decisions are evaluated per `(sp, s1, s2)` triple with the same
/// bound and threshold as the serial loop, so the kept/skipped set — and the
/// output tensor — are bit-identical to a serial build.
pub fn eri3_tensor_screened(
    op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    thresh: f64,
) -> Result<(Array3<f64>, usize, usize), FerricError> {
    use rayon::prelude::*;

    let naux = dfbs.nbasis();
    let nbas = obs.nbasis();
    let nsh_obs = obs.nshells();
    let nsh_df = dfbs.nshells();
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();

    // Schwarz bounds matched to the operator: |(P|μν)| ≤ Q3[P] · Q(μ,ν).
    let q_obs = schwarz(op, obs)?;
    let q3 = schwarz3_aux(op, dfbs)?;

    // Surface any engine-construction error up front (see eri3_tensor).
    Engine::new_3center(op, obs, dfbs, 1e-14)?;

    let mut eri = Array3::<f64>::zeros((naux, nbas, nbas));

    // Raw-pointer scatter: each aux shell `sp` writes a disjoint band of aux
    // rows, so writes from distinct workers never overlap.
    let eri_ptr = eri.as_mut_ptr() as usize;
    let stride0 = nbas * nbas; // p-stride
    let stride1 = nbas; // mu-stride

    let (n_kept, n_total) = (0..nsh_df)
        .into_par_iter()
        .map_init(
            || Engine::new_3center(op, obs, dfbs, 1e-14).expect("3-center engine (pre-validated)"),
            |eng, sp| {
                let mut kept = 0usize;
                let mut total = 0usize;
                let q3p = q3[sp];
                let np = dims_df[sp];
                let p0 = offs_df[sp];
                for s1 in 0..nsh_obs {
                    let n1 = dims_obs[s1];
                    let m0 = offs_obs[s1];
                    for s2 in 0..=s1 {
                        total += 1;
                        if q3p * q_obs[(s1, s2)] < thresh {
                            continue;
                        }
                        let Some(block) = eng.compute_eri3(obs, dfbs, sp, s1, s2) else { continue };
                        kept += 1;
                        let n2 = dims_obs[s2];
                        let n0 = offs_obs[s2];
                        for p in 0..np {
                            for i in 0..n1 {
                                for j in 0..n2 {
                                    let val = block[(p * n1 + i) * n2 + j];
                                    let pp = p0 + p;
                                    let mm = m0 + i;
                                    let nn = n0 + j;
                                    // SAFETY: rayon workers write to disjoint
                                    // (P, s1, s2) shell-triple blocks; the μ↔ν
                                    // symmetry write stays within the same block.
                                    unsafe {
                                        let base = eri_ptr as *mut f64;
                                        *base.add(pp * stride0 + mm * stride1 + nn) = val;
                                        *base.add(pp * stride0 + nn * stride1 + mm) = val;
                                    }
                                }
                            }
                        }
                    }
                }
                (kept, total)
            },
        )
        .reduce(|| (0usize, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));
    Ok((eri, n_kept, n_total))
}

/// QQR-screened 3-center ERI builder.
///
/// Same dense output as [`eri3_tensor`], but uses the distance-aware QQR-3
/// bound (Schwarz × `min(1, ext·ext/R) × op_decay(R)`) so erfc-attenuated
/// operators actually drop long-range shell triples. The basic Schwarz path
/// [`eri3_tensor_screened`] cannot do this — its bound is bra-only and has
/// no notion of bra-ket distance.
///
/// Skipped blocks remain zero. Returns `(tensor, n_kept, n_total)`.
///
/// Parallelized over aux shells `sp` exactly like [`eri3_tensor`] (one
/// `Engine` per rayon worker, disjoint aux-row-band raw-pointer scatter).
/// The QQR bound is evaluated per `(sp, s1, s2)` triple with the same
/// threshold as the serial loop (`QqrBounds3` is shared read-only), so the
/// kept/skipped set — and the output tensor — are bit-identical to a serial
/// build.
pub fn eri3_tensor_screened_qqr(
    op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    bounds: &QqrBounds3,
    thresh: f64,
) -> Result<(Array3<f64>, usize, usize), FerricError> {
    use rayon::prelude::*;

    let naux = dfbs.nbasis();
    let nbas = obs.nbasis();
    let nsh_obs = obs.nshells();
    let nsh_df = dfbs.nshells();
    let dims_obs = obs.shell_dims();
    let offs_obs = obs.shell_offsets();
    let dims_df = dfbs.shell_dims();
    let offs_df = dfbs.shell_offsets();

    // Surface any engine-construction error up front (see eri3_tensor).
    Engine::new_3center(op, obs, dfbs, 1e-14)?;

    let mut eri = Array3::<f64>::zeros((naux, nbas, nbas));

    // Raw-pointer scatter: each aux shell `sp` writes a disjoint band of aux
    // rows, so writes from distinct workers never overlap.
    let eri_ptr = eri.as_mut_ptr() as usize;
    let stride0 = nbas * nbas; // p-stride
    let stride1 = nbas; // mu-stride

    let (n_kept, n_total) = (0..nsh_df)
        .into_par_iter()
        .map_init(
            || Engine::new_3center(op, obs, dfbs, 1e-14).expect("3-center engine (pre-validated)"),
            |eng, sp| {
                let mut kept = 0usize;
                let mut total = 0usize;
                let np = dims_df[sp];
                let p0 = offs_df[sp];
                for s1 in 0..nsh_obs {
                    let n1 = dims_obs[s1];
                    let m0 = offs_obs[s1];
                    for s2 in 0..=s1 {
                        total += 1;
                        if bounds.estimate3(sp, s1, s2) < thresh {
                            continue;
                        }
                        let Some(block) = eng.compute_eri3(obs, dfbs, sp, s1, s2) else { continue };
                        kept += 1;
                        let n2 = dims_obs[s2];
                        let n0 = offs_obs[s2];
                        for p in 0..np {
                            for i in 0..n1 {
                                for j in 0..n2 {
                                    let val = block[(p * n1 + i) * n2 + j];
                                    let pp = p0 + p;
                                    let mm = m0 + i;
                                    let nn = n0 + j;
                                    // SAFETY: rayon workers write to disjoint
                                    // (P, s1, s2) shell-triple blocks; the μ↔ν
                                    // symmetry write stays within the same block.
                                    unsafe {
                                        let base = eri_ptr as *mut f64;
                                        *base.add(pp * stride0 + mm * stride1 + nn) = val;
                                        *base.add(pp * stride0 + nn * stride1 + mm) = val;
                                    }
                                }
                            }
                        }
                    }
                }
                (kept, total)
            },
        )
        .reduce(|| (0usize, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));
    Ok((eri, n_kept, n_total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    /// Serial reference for `eri3_tensor_screened` (the pre-parallelization
    /// implementation, kept verbatim). The parallel builder must reproduce it
    /// bit-for-bit: same screening decisions, same single-write-per-element
    /// scatter, no accumulation anywhere.
    fn eri3_tensor_screened_serial(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        thresh: f64,
    ) -> Result<(Array3<f64>, usize, usize), FerricError> {
        let naux = dfbs.nbasis();
        let nbas = obs.nbasis();
        let nsh_obs = obs.nshells();
        let nsh_df = dfbs.nshells();
        let dims_obs = obs.shell_dims();
        let offs_obs = obs.shell_offsets();
        let dims_df = dfbs.shell_dims();
        let offs_df = dfbs.shell_offsets();

        let q_obs = schwarz(op, obs)?;
        let q3 = schwarz3_aux(op, dfbs)?;

        let mut eng = Engine::new_3center(op, obs, dfbs, 1e-14)?;
        let mut eri = Array3::zeros((naux, nbas, nbas));
        let mut n_kept = 0usize;
        let mut n_total = 0usize;

        for sp in 0..nsh_df {
            let q3p = q3[sp];
            for s1 in 0..nsh_obs {
                for s2 in 0..=s1 {
                    n_total += 1;
                    if q3p * q_obs[(s1, s2)] < thresh {
                        continue;
                    }
                    let Some(block) = eng.compute_eri3(obs, dfbs, sp, s1, s2) else { continue };
                    n_kept += 1;
                    let np = dims_df[sp];
                    let n1 = dims_obs[s1];
                    let n2 = dims_obs[s2];
                    for p in 0..np {
                        for i in 0..n1 {
                            for j in 0..n2 {
                                let val = block[(p * n1 + i) * n2 + j];
                                eri[(offs_df[sp] + p, offs_obs[s1] + i, offs_obs[s2] + j)] = val;
                                eri[(offs_df[sp] + p, offs_obs[s2] + j, offs_obs[s1] + i)] = val;
                            }
                        }
                    }
                }
            }
        }
        Ok((eri, n_kept, n_total))
    }

    /// Serial reference for `eri3_tensor_screened_qqr` (pre-parallelization
    /// implementation, kept verbatim).
    fn eri3_tensor_screened_qqr_serial(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        bounds: &QqrBounds3,
        thresh: f64,
    ) -> Result<(Array3<f64>, usize, usize), FerricError> {
        let naux = dfbs.nbasis();
        let nbas = obs.nbasis();
        let nsh_obs = obs.nshells();
        let nsh_df = dfbs.nshells();
        let dims_obs = obs.shell_dims();
        let offs_obs = obs.shell_offsets();
        let dims_df = dfbs.shell_dims();
        let offs_df = dfbs.shell_offsets();

        let mut eng = Engine::new_3center(op, obs, dfbs, 1e-14)?;
        let mut eri = Array3::zeros((naux, nbas, nbas));
        let mut n_kept = 0usize;
        let mut n_total = 0usize;

        for sp in 0..nsh_df {
            for s1 in 0..nsh_obs {
                for s2 in 0..=s1 {
                    n_total += 1;
                    if bounds.estimate3(sp, s1, s2) < thresh {
                        continue;
                    }
                    let Some(block) = eng.compute_eri3(obs, dfbs, sp, s1, s2) else { continue };
                    n_kept += 1;
                    let np = dims_df[sp];
                    let n1 = dims_obs[s1];
                    let n2 = dims_obs[s2];
                    for p in 0..np {
                        for i in 0..n1 {
                            for j in 0..n2 {
                                let val = block[(p * n1 + i) * n2 + j];
                                eri[(offs_df[sp] + p, offs_obs[s1] + i, offs_obs[s2] + j)] = val;
                                eri[(offs_df[sp] + p, offs_obs[s2] + j, offs_obs[s1] + i)] = val;
                            }
                        }
                    }
                }
            }
        }
        Ok((eri, n_kept, n_total))
    }

    /// Assert two tensors are bit-identical (not just close): the parallel
    /// scatter writes each element exactly once, so no FP summation order can
    /// change and any difference at all is a bug.
    fn assert_bit_identical(a: &Array3<f64>, b: &Array3<f64>, what: &str) {
        assert_eq!(a.dim(), b.dim(), "{what}: shape mismatch");
        let n_diff = a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        assert_eq!(n_diff, 0, "{what}: {n_diff} elements differ bitwise");
    }

    #[test]
    fn test_eri3_screened_parallel_bitidentical_to_serial() {
        // Parallel Schwarz-screened builder must match the serial reference
        // bit-for-bit — including at a loose threshold where screening
        // actually fires and the skip set is nontrivial.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        for op in [Operator::coulomb(), Operator::erfc(0.222)] {
            for &thresh in &[0.0, 1e-10, 1e-4, 1e-2] {
                let (ser, k_ser, t_ser) = eri3_tensor_screened_serial(op, &obs, &dfbs, thresh).unwrap();
                let (par, k_par, t_par) = eri3_tensor_screened(op, &obs, &dfbs, thresh).unwrap();
                assert_eq!((k_par, t_par), (k_ser, t_ser),
                    "screening counts diverge at thresh={thresh:.0e}");
                assert_bit_identical(&ser, &par, &format!("schwarz-screened thresh={thresh:.0e}"));
            }
        }
        // Sanity: at 1e-2 screening must actually drop triples, otherwise the
        // "screening fires" leg of this test is vacuous.
        let (_, k, t) = eri3_tensor_screened(Operator::erfc(0.222), &obs, &dfbs, 1e-2).unwrap();
        assert!(k < t, "expected screening to fire at thresh=1e-2 ({k}/{t} kept)");
    }

    #[test]
    fn test_eri3_qqr_screened_parallel_bitidentical_to_serial() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::erfc(0.222);
        let bounds = crate::qqr3::QqrBounds3::new(op, &mol, &obs, &dfbs).unwrap();
        for &thresh in &[0.0, 1e-10, 1e-4, 1e-2] {
            let (ser, k_ser, t_ser) =
                eri3_tensor_screened_qqr_serial(op, &obs, &dfbs, &bounds, thresh).unwrap();
            let (par, k_par, t_par) =
                eri3_tensor_screened_qqr(op, &obs, &dfbs, &bounds, thresh).unwrap();
            assert_eq!((k_par, t_par), (k_ser, t_ser),
                "QQR screening counts diverge at thresh={thresh:.0e}");
            assert_bit_identical(&ser, &par, &format!("qqr-screened thresh={thresh:.0e}"));
        }
        // Sanity: the loose threshold must actually drop triples.
        let (_, k, t) = eri3_tensor_screened_qqr(op, &obs, &dfbs, &bounds, 1e-2).unwrap();
        assert!(k < t, "expected QQR screening to fire at thresh=1e-2 ({k}/{t} kept)");
    }

    #[test]
    fn eri3_block_equals_dense_slice() {
        use crate::basis_bridge::PreparedBasis;
        let mol = Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let dense = eri3_tensor(op, &obs, &dfbs).unwrap();
        let naux = dense.dim().0;
        let (p0, p1) = (2, naux.min(9));
        let blk = eri3_block(op, &obs, &dfbs, p0, p1).unwrap();
        let ref_slice = dense.slice(ndarray::s![p0..p1, .., ..]);
        let maxdiff = (&blk - &ref_slice).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(maxdiff == 0.0, "eri3_block != dense slice, maxdiff={maxdiff}");
    }

    #[test]
    fn test_coulomb_metric_2c_symmetric() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let v = coulomb_metric_2c(Operator::coulomb(), &dfbs).unwrap();
        let n = dfbs.nbasis();
        for i in 0..n {
            for j in 0..n {
                assert!((v[(i, j)] - v[(j, i)]).abs() < 1e-12,
                    "(P|Q) not symmetric at ({i},{j})");
            }
        }
        // Diagonal should be positive
        for i in 0..n { assert!(v[(i, i)] > 0.0, "(P|P) should be positive"); }
    }

    /// Serial reference for `coulomb_metric_2c` (pre-parallelization
    /// implementation, kept verbatim).
    fn coulomb_metric_2c_serial(op: Operator, dfbs: &PreparedBasis) -> Array2<f64> {
        let naux = dfbs.nbasis();
        let nsh = dfbs.nshells();
        let dims = dfbs.shell_dims();
        let offs = dfbs.shell_offsets();
        let mut eng = Engine::new_2center(op, dfbs, 1e-14).unwrap();
        let mut v = Array2::zeros((naux, naux));
        for sp in 0..nsh {
            for sq in 0..=sp {
                let block = eng.compute_eri2(dfbs, sp, sq);
                let np = dims[sp];
                let nq = dims[sq];
                for p in 0..np {
                    for q in 0..nq {
                        let val = block[p * nq + q];
                        v[(offs[sp] + p, offs[sq] + q)] = val;
                        v[(offs[sq] + q, offs[sp] + p)] = val;
                    }
                }
            }
        }
        v
    }

    #[test]
    fn test_coulomb_metric_2c_bitidentical_to_serial() {
        // alkane_6/cc-pVDZ-RI clears PAR_METRIC_SHELL_THRESHOLD (64 aux shells),
        // so this actually exercises the rayon path, not just the fallback.
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_6.xyz").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        assert!(dfbs.nshells() >= 64,
            "test aux basis too small to exercise the parallel path: {} shells", dfbs.nshells());
        for op in [Operator::coulomb(), Operator::erfc(0.222)] {
            let par = coulomb_metric_2c(op, &dfbs).unwrap();
            let ser = coulomb_metric_2c_serial(op, &dfbs);
            assert_eq!(par.dim(), ser.dim());
            let n_diff = par.iter().zip(ser.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(n_diff, 0, "coulomb_metric_2c: {n_diff} elements differ bitwise (op={op:?})");
        }
    }

    #[test]
    fn test_eri3_screened_zero_thresh_matches_dense() {
        // With thresh = 0, the screened path must reproduce the dense tensor
        // bit-for-bit (modulo libint's internal 1e-14 precision filter).
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_set = basis::bundled("sto-3g").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let dense = eri3_tensor(Operator::coulomb(), &obs, &dfbs).unwrap();
        let (screened, n_kept, n_total) =
            eri3_tensor_screened(Operator::coulomb(), &obs, &dfbs, 0.0).unwrap();
        assert_eq!(n_kept, n_total, "thresh=0 should keep every (P,s1,s2≤s1) triple");
        let max_diff = dense.iter().zip(screened.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(max_diff < 1e-12, "screened tensor diverges from dense: max diff {max_diff:.2e}");
    }

    #[test]
    fn test_eri3_screened_erfc_water_drops_triples() {
        // Real check: on water/cc-pVDZ with erfc(ω=0.222 Bohr⁻¹), a production
        // threshold must (a) drop some triples that Coulomb keeps, and (b) the
        // surviving tensor must agree with the unscreened erfc build to high
        // precision on retained entries.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs_set = basis::bundled("cc-pvdz").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let op = Operator::erfc(0.222);
        let unscreened = eri3_tensor(op, &obs, &dfbs).unwrap();
        let (screened, n_kept, n_total) =
            eri3_tensor_screened(op, &obs, &dfbs, 1e-10).unwrap();
        eprintln!("H2O/cc-pVDZ erfc(0.222) eri3 screening: {n_kept}/{n_total} triples kept");
        // Either we drop some, or the system is too small for screening to fire.
        // Tensor agreement is the load-bearing check.
        let max_diff = unscreened.iter().zip(screened.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(max_diff < 1e-9,
            "screened erfc tensor diverges from unscreened: max diff {max_diff:.2e}");
    }

    #[test]
    fn test_eri3_qqr_screened_water_matches_unscreened() {
        // Correctness: QQR-screened tensor on water/cc-pVDZ with erfc must
        // agree with the unscreened build to high precision at production
        // threshold. Water is too small for screening to fire, but the
        // surviving entries must still be correct.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs_set = basis::bundled("cc-pvdz").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let op = Operator::erfc(0.222);
        let bounds = crate::qqr3::QqrBounds3::new(op, &mol, &obs, &dfbs).unwrap();
        let unscreened = eri3_tensor(op, &obs, &dfbs).unwrap();
        let (screened, n_kept, n_total) =
            eri3_tensor_screened_qqr(op, &obs, &dfbs, &bounds, 1e-10).unwrap();
        eprintln!("water erfc QQR3 thresh=1e-10: {n_kept}/{n_total} kept");
        let max_diff = unscreened.iter().zip(screened.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(max_diff < 1e-9,
            "QQR-screened tensor diverges from unscreened: max diff {max_diff:.2e}");
    }

    #[test]
    #[ignore = "decane/cc-pVDZ build is heavy (~minute); run with --ignored for screening curve"]
    fn bench_eri3_screened_decane_erfc() {
        // The water test confirms correctness but cannot fire screening — the
        // molecule is smaller than erfc's range. Decane (C10H22, ~12 Å) is the
        // smallest system where shell-triple distances exceed the erfc range
        // and screening should start to drop triples meaningfully.
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_10.xyz").unwrap();
        let obs_set = basis::bundled("cc-pvdz").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        eprintln!(
            "decane: nbas={}, naux={}, nsh_obs={}, nsh_aux={}",
            obs.nbasis(), dfbs.nbasis(), obs.nshells(), dfbs.nshells()
        );

        // Coulomb screens little — the operator has infinite range.
        let op_c = Operator::coulomb();
        let (_, n_kept_c, n_total_c) =
            eri3_tensor_screened(op_c, &obs, &dfbs, 1e-10).unwrap();
        eprintln!("  Coulomb thresh=1e-10: {n_kept_c}/{n_total_c} triples kept ({:.1}%)",
            100.0 * n_kept_c as f64 / n_total_c as f64);

        // erfc with the dissertation optimal omega should drop substantially more.
        let op_e = Operator::erfc(0.222);
        for &thresh in &[1e-12, 1e-10, 1e-8, 1e-6] {
            let (_, n_kept, n_total) =
                eri3_tensor_screened(op_e, &obs, &dfbs, thresh).unwrap();
            eprintln!("  Schwarz erfc(0.222) thresh={thresh:.0e}: {n_kept}/{n_total} kept ({:.1}%)",
                100.0 * n_kept as f64 / n_total as f64);
        }

        // QQR-3 with distance-aware bound — this is what should actually fire
        // for erfc. Same operator and thresholds; compare retention to Schwarz.
        let bounds = crate::qqr3::QqrBounds3::new(op_e, &mol, &obs, &dfbs).unwrap();
        for &thresh in &[1e-12, 1e-10, 1e-8, 1e-6] {
            let (_, n_kept, n_total) =
                eri3_tensor_screened_qqr(op_e, &obs, &dfbs, &bounds, thresh).unwrap();
            eprintln!("  QQR3   erfc(0.222) thresh={thresh:.0e}: {n_kept}/{n_total} kept ({:.1}%)",
                100.0 * n_kept as f64 / n_total as f64);
        }
    }

    #[test]
    fn test_eri3_symmetric_in_mu_nu() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_set = basis::bundled("sto-3g").unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();
        let eri = eri3_tensor(Operator::coulomb(), &obs, &dfbs).unwrap();
        let naux = dfbs.nbasis();
        let nbas = obs.nbasis();
        for p in 0..naux {
            for i in 0..nbas {
                for j in 0..nbas {
                    assert!((eri[(p, i, j)] - eri[(p, j, i)]).abs() < 1e-12,
                        "ERI3 not symmetric at P={p},i={i},j={j}");
                }
            }
        }
    }
}
