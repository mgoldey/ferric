//! Per-orbital screened-tile Sternheimer kernel.
//!
//! Sparse equivalent of `sternheimer::dielectric_matrix` and
//! `sternheimer::dielectric_apply`. Consumes a `ScreenedBov` (per-orbital
//! tiles on Boys-localized occupied orbitals) instead of a dense
//! (naux × nocc·nvir) `b_ov`.
//!
//! # Subspace dielectric (ε̃ in the trial-vector basis)
//!
//! Mirrors `sternheimer::dielectric_matrix_with_scale`. For each i_loc:
//!   1. Gather rows of `v_mat` at `p_lists[i_loc]` → `v_gather` shape
//!      (m_i, msub) where msub = v_mat.ncols().
//!   2. Compute `rhs_i = v_gather.T @ tile_i` → shape (msub, nvir).
//!   3. Scale columns by `s_ia = sqrt(4·e_ia / (ω² + e_ia²))` with
//!      `e_ia = eps_vir[a] - eps_loc[i_loc]`.
//!   4. Accumulate `out += rhs_i @ rhs_i.T` (SYRK over msub-msub).
//!
//! Finally `out += I` to form `ε̃ = I + Π`.
//!
//! # Apply form (ε̃ · V)
//!
//! Used by block-Lanczos. For each i_loc:
//!   1. Gather rows as above.
//!   2. `y_i = v_gather.T @ tile_i` shape (msub, nvir).
//!   3. Scale columns by `s_ia^2`.
//!   4. Compute `contrib = tile_i @ y_i.T` shape (m_i, msub).
//!   5. Scatter rows of `contrib` back into `out` at `p_lists[i_loc]`.
//!
//! # Why this is correct
//!
//! Dense form: `Π_{PQ} = Σ_{ia} (4·e_ia/(ω²+e_ia²)) B^P_{ia} B^Q_{ia}`.
//! Split by occ index:
//!   `Π_{PQ} = Σ_i [ Σ_a (s_ia²/4 · 4) B^P_{ia} B^Q_{ia} ]`
//!          = `Σ_i (B_{i,:}^P diag(s_i²) B_{i,:}^Q,T)`
//! Per-i_loc contributions are independent and add. Screening drops aux rows
//! P with negligible `B^P_{i_loc, a}` for all a; the dropped contributions
//! are bounded by `thresh² · nvir`.
//!
//! # Parallelism — measured impact (sternheimer-sparse-parallelize)
//!
//! Both `dielectric_matrix_screened` and `dielectric_apply_screened` used to
//! run the `i_loc` loop fully serially (zero rayon usage). They are now
//! parallelized over `i_loc` (see each function's doc comment for the
//! reduction shape). Confirmed via a rayon thread-touch probe that the work
//! IS distributed across multiple OS threads (10 distinct workers touched on
//! a 12-core box for a 41-`i_loc` alkane_10/STO-3G case).
//!
//! However, measured end-to-end on alkane_10/STO-3G/`boys:1e-3` (32 atoms,
//! `n_occ_loc≈41`, `naux≈868`), the win is modest: the eigensolve stage
//! (which calls these functions repeatedly) dropped from ~3.30s to ~3.20s
//! mean over repeated alternating A/B runs (≈3% faster), not a dramatic
//! speedup. Root cause: at `thresh=1e-3` on this system the exact `|p_ii|`
//! screening metric retains ~800/868 aux rows per orbital (~92%) — matching
//! the project's other measured finding that Boys-screening's distance/exact
//! metrics often don't produce much real sparsity at small-to-medium
//! molecular scale (see `g6-sparse-rpa-distance-cutoff-deadend`). With tiles
//! nearly as wide as the dense tensor, the per-`i_loc` GEMMs are large enough
//! that OpenBLAS-1-thread GEMM cost dominates and 41-way rayon fan-out only
//! trims dispatch/reduction overhead, not FLOPs.
//!
//! More importantly: on this same system, this module's functions are NOT
//! the dominant cost. `screen::build_screened_bov_boys` (Boys localization +
//! screened 3-index integral construction, called once before either
//! eigensolver runs) is itself fully serial with zero rayon usage and costs
//! ~11s vs this module's ~3.2s — i.e. `screen.rs`'s build phase is now the
//! larger of the two costs and is the more consequential lever for closing
//! the boys-vs-dense gap on molecules where screening doesn't prune much.
//! That module was out of this change's scope (task named only this file's
//! two functions) and is a natural follow-up.

use crate::screen::ScreenedBov;
use ferric_integrals::blas_threads::with_blas_threads;
use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array1, Array2, Axis, Zip};

/// Build per-i_loc scale factors `s_ia = sqrt(4·e_ia/(ω²+e_ia²))`.
#[inline]
fn build_scale_for_iloc(eps_loc_i: f64, eps_vir: &[f64], omega: f64) -> Array1<f64> {
    let omega2 = omega * omega;
    let nvir = eps_vir.len();
    let mut s = Array1::<f64>::zeros(nvir);
    for (a, &eps_a) in eps_vir.iter().enumerate() {
        let e_ia = eps_a - eps_loc_i;
        s[a] = (4.0 * e_ia / (omega2 + e_ia * e_ia)).sqrt();
    }
    s
}

/// Subspace dielectric matrix `ε̃(iω)` evaluated through a screened B-tile
/// representation.
///
/// `v_mat` is (naux × msub). Returns an (msub × msub) symmetric matrix
/// `ε̃ = I + Π` matching `sternheimer::dielectric_matrix`.
///
/// # Parallelism
///
/// The `i_loc` loop is embarrassingly parallel: each iteration reads only
/// `bov.p_lists[i_loc]`/`bov.tiles[i_loc]`/`bov.eps_loc[i_loc]` and produces an
/// independent `(msub × msub)` partial that is *added* into the shared `out`
/// accumulator — same reduction shape as `DirectJK`'s per-quartet-group (J,K)
/// partials (`ferric_scf::direct_jk`/`reduce.rs`). We reuse
/// `ferric_scf::reduce::grouped_deterministic_sum` directly: it produces group
/// partials in parallel (`into_par_iter().map().collect()`, which preserves
/// ascending index order regardless of worker count) and folds them into `out`
/// in strict group order, so the result is bit-identical across
/// `RAYON_NUM_THREADS` — the same invariant the screened-vs-dense regression
/// tests in `tests/screened_rpa.rs` rely on (they assert the diff is
/// "deterministic, independent of BLAS thread count").
///
/// BLAS is pinned to 1 thread for the duration of the rayon region (mirrors
/// `energy.rs`'s frequency-parallel eigh path): the per-`i_loc` GEMMs must not
/// nest OpenBLAS threads under rayon workers (stack-overflow/oversubscription
/// hazard — see `openblas-rayon-dgetrf-crash` and
/// `ferric_integrals::blas_threads` doc comments). This is a no-op for the
/// default Davidson/Lanczos BLAS-threads=1 scope and only matters if a caller
/// opts into `FERRIC_BLAS_THREADS`/`FERRIC_LANCZOS_BLAS_THREADS` > 1 upstream.
pub fn dielectric_matrix_screened(
    v_mat: &Array2<f64>,
    bov: &ScreenedBov,
    eps_vir: &[f64],
    omega: f64,
) -> Array2<f64> {
    let msub = v_mat.ncols();
    let mut out = Array2::<f64>::zeros((msub, msub));

    with_blas_threads(1, || {
        let n_groups = bov.n_occ_loc;
        // One i_loc per "group" here: msub is typically small (subspace width),
        // so a per-i_loc (msub × msub) partial is cheap and the byte-budgeted
        // banding in `grouped_deterministic_sum` still applies if msub is ever
        // large.
        ferric_scf::reduce::grouped_deterministic_sum(&mut out, n_groups, msub, ferric_scf::reduce::default_band_bytes(), |i_loc| {
            Ok(dielectric_matrix_partial(v_mat, bov, eps_vir, omega, i_loc, msub))
        })
        .expect("dielectric_matrix_screened: partial builder is infallible");
    });

    // Symmetrize (defensive — GEMM-of-A·A^T is symmetric in exact arithmetic,
    // but floating-point drift gives O(ε) asymmetry that the eigensolver does
    // not like).
    let out_sym = 0.5 * (&out + &out.t());

    // ε̃ = I + Π.
    let mut eps_mat = out_sym;
    for alpha in 0..msub {
        eps_mat[(alpha, alpha)] += 1.0;
    }
    eps_mat
}

/// Single-`i_loc` partial for [`dielectric_matrix_screened`]:
/// `rhs_i @ rhs_i.T` (msub × msub), zero if `p_lists[i_loc]` is empty.
/// Factored out so the rayon-parallel group builder and any future serial
/// caller share the exact same per-iteration arithmetic.
fn dielectric_matrix_partial(
    v_mat: &Array2<f64>,
    bov: &ScreenedBov,
    eps_vir: &[f64],
    omega: f64,
    i_loc: usize,
    msub: usize,
) -> Array2<f64> {
    let p_list = &bov.p_lists[i_loc];
    let tile = &bov.tiles[i_loc];
    let m_i = p_list.len();
    let mut partial = Array2::<f64>::zeros((msub, msub));
    if m_i == 0 {
        return partial;
    }

    // Gather rows of v_mat at p_list → v_gather shape (m_i, msub).
    let mut v_gather = Array2::<f64>::zeros((m_i, msub));
    for (slot, &p) in p_list.iter().enumerate() {
        v_gather.slice_mut(s![slot, ..]).assign(&v_mat.slice(s![p, ..]));
    }

    // rhs_i = v_gather.T @ tile  →  shape (msub, nvir).
    let mut rhs_i: Array2<f64> = v_gather.t().dot(tile);

    // Scale columns by s_ia.
    let scale = build_scale_for_iloc(bov.eps_loc[i_loc], eps_vir, omega);
    let scale_row = scale.view().insert_axis(Axis(0));
    Zip::from(&mut rhs_i)
        .and_broadcast(scale_row)
        .for_each(|x, &s| *x *= s);

    // partial = rhs_i @ rhs_i.T (msub × msub).
    // Plain GEMM keeps the code path simple; SYRK upgrade is a follow-up if
    // profiling demands it.
    let rhs_t = rhs_i.t().to_owned();
    general_mat_mul(1.0, &rhs_i, &rhs_t, 0.0, &mut partial);
    partial
}

/// Apply form: returns `ε̃(iω) · V` in the naux × msub space.
///
/// Used by block-Lanczos which needs A·V rather than V^T·A·V.
///
/// # Parallelism
///
/// Each `i_loc` produces a `contrib` block that is *scattered* into
/// `p_lists[i_loc]`'s (generally non-contiguous) rows of the `(naux × msub)`
/// output. Different `i_loc` can and do share aux rows in `p_lists` (Boys
/// orbitals near each other couple to overlapping aux shells), so this is
/// **not** a disjoint-output-range reduction like `dispersion.rs`'s
/// `casimir_polder_c6` pair loop — naive concurrent `+=` writes from multiple
/// threads into the same row would race. Instead each `i_loc` builds a full
/// `(naux × msub)` sparse-row partial (zero everywhere except its own
/// `p_list` rows) and partials are summed via
/// `ferric_scf::reduce::grouped_deterministic_sum`, the same
/// group-parallel/serial-fold-in-ascending-order idiom `DirectJK` uses for its
/// per-quartet-group (J,K) accumulation — bit-identical across
/// `RAYON_NUM_THREADS`, matching the determinism the `tests/screened_rpa.rs`
/// screened-vs-dense regressions already assume.
///
/// BLAS pinned to 1 thread for the rayon region (see
/// [`dielectric_matrix_screened`]'s doc comment for the hazard this guards).
pub fn dielectric_apply_screened(
    v_mat: &Array2<f64>,
    bov: &ScreenedBov,
    eps_vir: &[f64],
    omega: f64,
) -> Array2<f64> {
    let naux = v_mat.nrows();
    let msub = v_mat.ncols();
    let mut out: Array2<f64> = v_mat.to_owned(); // identity contribution

    with_blas_threads(1, || {
        let n_groups = bov.n_occ_loc;
        // `grouped_deterministic_sum`'s `nbf` parameter is a square-matrix byte
        // estimator (`nbf² * 8` per partial, sized for DirectJK's square nbf×nbf
        // partials). Our partial is rectangular (naux × msub); pass the
        // square-equivalent side length so the band-width byte budget still
        // approximates the true per-partial size instead of over/under-counting
        // by the naux/msub aspect ratio.
        let sq_equiv = ((naux as f64) * (msub as f64)).sqrt().ceil() as usize;
        ferric_scf::reduce::grouped_deterministic_sum(&mut out, n_groups, sq_equiv, ferric_scf::reduce::default_band_bytes(), |i_loc| {
            Ok(dielectric_apply_partial(v_mat, bov, eps_vir, omega, i_loc, naux, msub))
        })
        .expect("dielectric_apply_screened: partial builder is infallible");
    });

    out
}

/// Single-`i_loc` scatter partial for [`dielectric_apply_screened`]: a full
/// `(naux × msub)` matrix, zero everywhere except rows in
/// `bov.p_lists[i_loc]`, which hold `contrib`'s rows. Building the full-width
/// zero-padded partial (rather than a compact `(m_i × msub)` block) lets the
/// group reduction add it directly onto the shared accumulator without a
/// second, non-disjoint scatter step.
fn dielectric_apply_partial(
    v_mat: &Array2<f64>,
    bov: &ScreenedBov,
    eps_vir: &[f64],
    omega: f64,
    i_loc: usize,
    naux: usize,
    msub: usize,
) -> Array2<f64> {
    let p_list = &bov.p_lists[i_loc];
    let tile = &bov.tiles[i_loc];
    let m_i = p_list.len();
    let mut partial = Array2::<f64>::zeros((naux, msub));
    if m_i == 0 {
        return partial;
    }

    // Gather rows.
    let mut v_gather = Array2::<f64>::zeros((m_i, msub));
    for (slot, &p) in p_list.iter().enumerate() {
        v_gather.slice_mut(s![slot, ..]).assign(&v_mat.slice(s![p, ..]));
    }

    // y_i = v_gather.T @ tile  →  (msub, nvir)
    let mut y_i: Array2<f64> = v_gather.t().dot(tile);

    // Scale columns by s_ia².
    let scale = build_scale_for_iloc(bov.eps_loc[i_loc], eps_vir, omega);
    let scale_row = scale.view().insert_axis(Axis(0));
    Zip::from(&mut y_i)
        .and_broadcast(scale_row)
        .for_each(|x, &s| *x *= s * s);

    // contrib = tile @ y_i.T  →  (m_i, msub)
    let y_t = y_i.t().to_owned();
    let mut contrib = Array2::<f64>::zeros((m_i, msub));
    general_mat_mul(1.0, tile, &y_t, 0.0, &mut contrib);

    // Scatter rows of contrib into their p_list positions of the full-width
    // zero-padded partial.
    for (slot, &p) in p_list.iter().enumerate() {
        let mut row = partial.slice_mut(s![p, ..]);
        let crow = contrib.slice(s![slot, ..]);
        for col in 0..msub {
            row[col] += crow[col];
        }
    }
    partial
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic synthetic `ScreenedBov` with OVERLAPPING `p_lists` across
    /// `i_loc` (the exact scenario that makes `dielectric_apply_screened`'s
    /// scatter non-disjoint and rules out a naive concurrent-write
    /// parallelization). `naux=6`, `nvir=4`, `n_occ_loc=5`; every p_list is a
    /// distinct but overlapping window of aux indices so at least one aux row
    /// receives contributions from 2+ different i_loc.
    fn synthetic_bov() -> ScreenedBov {
        let naux = 6;
        let nvir = 4;
        // Overlapping windows: e.g. p_lists[0] and p_lists[1] share aux row 1.
        let p_lists: Vec<Vec<usize>> = vec![
            vec![0, 1],
            vec![1, 2, 3],
            vec![2, 3, 4],
            vec![4, 5],
            vec![0, 3, 5],
        ];
        let mut tiles = Vec::new();
        for (i_loc, p_list) in p_lists.iter().enumerate() {
            let m_i = p_list.len();
            let mut tile = Array2::<f64>::zeros((m_i, nvir));
            for r in 0..m_i {
                for c in 0..nvir {
                    // Deterministic, irregular values (no special symmetry) so
                    // a wrong reduction order or race would show up numerically.
                    let v = ((i_loc * 7 + r * 3 + c * 5 + 1) % 11) as f64 * 0.137
                        - ((i_loc + r + c) as f64) * 0.021;
                    tile[(r, c)] = v;
                }
            }
            tiles.push(tile);
        }
        let eps_loc: Vec<f64> = (0..p_lists.len()).map(|i| -0.5 - 0.1 * i as f64).collect();
        let n_occ_loc = p_lists.len();
        ScreenedBov {
            n_occ_loc,
            nvir,
            naux,
            p_lists,
            tiles,
            centroids: vec![[0.0, 0.0, 0.0]; n_occ_loc],
            eps_loc,
            v_inv_sqrt: Array2::eye(naux),
            total_retained: 0,
        }
    }

    /// Naive fully-serial reference implementation of the apply form,
    /// mirroring the pre-parallelization code exactly (kept local to the test
    /// module so it can't silently drift with the production implementation).
    fn dielectric_apply_screened_serial_reference(
        v_mat: &Array2<f64>,
        bov: &ScreenedBov,
        eps_vir: &[f64],
        omega: f64,
    ) -> Array2<f64> {
        let msub = v_mat.ncols();
        let mut out: Array2<f64> = v_mat.to_owned();
        for i_loc in 0..bov.n_occ_loc {
            let p_list = &bov.p_lists[i_loc];
            let tile = &bov.tiles[i_loc];
            let m_i = p_list.len();
            if m_i == 0 {
                continue;
            }
            let mut v_gather = Array2::<f64>::zeros((m_i, msub));
            for (slot, &p) in p_list.iter().enumerate() {
                v_gather.slice_mut(s![slot, ..]).assign(&v_mat.slice(s![p, ..]));
            }
            let mut y_i: Array2<f64> = v_gather.t().dot(tile);
            let scale = build_scale_for_iloc(bov.eps_loc[i_loc], eps_vir, omega);
            let scale_row = scale.view().insert_axis(Axis(0));
            Zip::from(&mut y_i).and_broadcast(scale_row).for_each(|x, &s| *x *= s * s);
            let y_t = y_i.t().to_owned();
            let mut contrib = Array2::<f64>::zeros((m_i, msub));
            general_mat_mul(1.0, tile, &y_t, 0.0, &mut contrib);
            for (slot, &p) in p_list.iter().enumerate() {
                let mut row = out.slice_mut(s![p, ..]);
                let crow = contrib.slice(s![slot, ..]);
                for col in 0..msub {
                    row[col] += crow[col];
                }
            }
        }
        out
    }

    fn dielectric_matrix_screened_serial_reference(
        v_mat: &Array2<f64>,
        bov: &ScreenedBov,
        eps_vir: &[f64],
        omega: f64,
    ) -> Array2<f64> {
        let msub = v_mat.ncols();
        let mut out = Array2::<f64>::zeros((msub, msub));
        for i_loc in 0..bov.n_occ_loc {
            let p_list = &bov.p_lists[i_loc];
            let tile = &bov.tiles[i_loc];
            let m_i = p_list.len();
            if m_i == 0 {
                continue;
            }
            let mut v_gather = Array2::<f64>::zeros((m_i, msub));
            for (slot, &p) in p_list.iter().enumerate() {
                v_gather.slice_mut(s![slot, ..]).assign(&v_mat.slice(s![p, ..]));
            }
            let mut rhs_i: Array2<f64> = v_gather.t().dot(tile);
            let scale = build_scale_for_iloc(bov.eps_loc[i_loc], eps_vir, omega);
            let scale_row = scale.view().insert_axis(Axis(0));
            Zip::from(&mut rhs_i).and_broadcast(scale_row).for_each(|x, &s| *x *= s);
            let rhs_t = rhs_i.t().to_owned();
            general_mat_mul(1.0, &rhs_i, &rhs_t, 1.0, &mut out);
        }
        let out_sym = 0.5 * (&out + &out.t());
        let mut eps_mat = out_sym;
        for alpha in 0..msub {
            eps_mat[(alpha, alpha)] += 1.0;
        }
        eps_mat
    }

    #[test]
    fn dielectric_apply_screened_matches_serial_reference() {
        let bov = synthetic_bov();
        let msub = 3;
        // Deterministic v_mat, not orthonormal/special — exercises generic values.
        let mut v_mat = Array2::<f64>::zeros((bov.naux, msub));
        for r in 0..bov.naux {
            for c in 0..msub {
                v_mat[(r, c)] = ((r * 3 + c * 2 + 1) % 7) as f64 * 0.29 - 0.5;
            }
        }
        let eps_vir: Vec<f64> = (0..bov.nvir).map(|a| 0.3 + 0.15 * a as f64).collect();
        let omega = 0.4;

        let got = dielectric_apply_screened(&v_mat, &bov, &eps_vir, omega);
        let expected = dielectric_apply_screened_serial_reference(&v_mat, &bov, &eps_vir, omega);

        assert_eq!(got.dim(), expected.dim());
        let max_diff = (&got - &expected).iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        assert!(
            max_diff < 1e-12,
            "parallel apply diverges from serial reference by {max_diff:.3e}"
        );
    }

    #[test]
    fn dielectric_matrix_screened_matches_serial_reference() {
        let bov = synthetic_bov();
        let msub = 3;
        let mut v_mat = Array2::<f64>::zeros((bov.naux, msub));
        for r in 0..bov.naux {
            for c in 0..msub {
                v_mat[(r, c)] = ((r * 5 + c * 3 + 2) % 7) as f64 * 0.19 - 0.4;
            }
        }
        let eps_vir: Vec<f64> = (0..bov.nvir).map(|a| 0.3 + 0.15 * a as f64).collect();
        let omega = 0.4;

        let got = dielectric_matrix_screened(&v_mat, &bov, &eps_vir, omega);
        let expected = dielectric_matrix_screened_serial_reference(&v_mat, &bov, &eps_vir, omega);

        assert_eq!(got.dim(), expected.dim());
        let max_diff = (&got - &expected).iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        assert!(
            max_diff < 1e-12,
            "parallel matrix diverges from serial reference by {max_diff:.3e}"
        );
    }

    /// Thread-count independence: the deterministic grouped-sum reduction
    /// must give bit-identical results regardless of `RAYON_NUM_THREADS` —
    /// the same invariant `ferric_scf::reduce`'s own tests pin for DirectJK,
    /// and the property `tests/screened_rpa.rs`'s screened-vs-dense
    /// regressions rely on when comparing against a run on a different box.
    #[test]
    fn dielectric_apply_screened_is_thread_count_independent() {
        let bov = synthetic_bov();
        let msub = 3;
        let mut v_mat = Array2::<f64>::zeros((bov.naux, msub));
        for r in 0..bov.naux {
            for c in 0..msub {
                v_mat[(r, c)] = ((r * 3 + c * 2 + 1) % 7) as f64 * 0.29 - 0.5;
            }
        }
        let eps_vir: Vec<f64> = (0..bov.nvir).map(|a| 0.3 + 0.15 * a as f64).collect();
        let omega = 0.4;

        let run = |n_threads: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build().unwrap();
            pool.install(|| dielectric_apply_screened(&v_mat, &bov, &eps_vir, omega))
        };

        let r1 = run(1);
        let r2 = run(2);
        let r4 = run(4);
        assert_eq!(r1, r2, "apply result must be bit-identical at 1 vs 2 rayon threads");
        assert_eq!(r1, r4, "apply result must be bit-identical at 1 vs 4 rayon threads");
    }

    #[test]
    fn dielectric_matrix_screened_is_thread_count_independent() {
        let bov = synthetic_bov();
        let msub = 3;
        let mut v_mat = Array2::<f64>::zeros((bov.naux, msub));
        for r in 0..bov.naux {
            for c in 0..msub {
                v_mat[(r, c)] = ((r * 5 + c * 3 + 2) % 7) as f64 * 0.19 - 0.4;
            }
        }
        let eps_vir: Vec<f64> = (0..bov.nvir).map(|a| 0.3 + 0.15 * a as f64).collect();
        let omega = 0.4;

        let run = |n_threads: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build().unwrap();
            pool.install(|| dielectric_matrix_screened(&v_mat, &bov, &eps_vir, omega))
        };

        let r1 = run(1);
        let r2 = run(2);
        let r4 = run(4);
        assert_eq!(r1, r2, "matrix result must be bit-identical at 1 vs 2 rayon threads");
        assert_eq!(r1, r4, "matrix result must be bit-identical at 1 vs 4 rayon threads");
    }
}
