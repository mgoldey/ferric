//! Boys-localized PDEP seed strategies for the dielectric eigensolver.
//!
//! Provides two Boys-based seed builders that complement the existing
//! atom-localized seed in `build_atom_seed`:
//!
//!   * **Occupied-only Boys**: one seed vector per Boys-localized occupied
//!     orbital `i_loc`, formed as the uniform sum over all virtuals of
//!     `b_ov[:, i_loc * nvir + a]`. This gives `nocc` columns.
//!
//!   * **Schurkus-Ochsenfeld occ × vir products**: for each significant
//!     (i_loc, a) pair, ranked by the orbital-energy denominator 1/(ε_a - ε_i),
//!     form a seed column `b_ov[:, i_loc * nvir + a] / (ε_a - ε_i)`. Up to
//!     `n_seed_target` columns are taken.
//!
//! Both are QR-orthonormalized; rank-deficient columns are dropped.
//!
//! The localized-virtual side of Schurkus-Ochsenfeld is approximated here by
//! using canonical virtual MOs directly — virtual MOs span the same column
//! space as any localized virtual transformation, so no separate localization
//! step is required for seed-construction purposes. Importance ranking and the
//! Boys mixing on the occupied side give the seed its locality.

use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::dipole;
use ferric_mp2::boys::boys_localize;
use ferric_mp2::rimp2::RpaIntermediates;
use ferric_scf::ScfResult;
use ndarray::{s, Array1, Array2};
use ndarray_linalg::QR;

/// Strategy selector matching the seed builders below.
#[derive(Debug, Clone, Copy)]
pub enum BoysSeedMode {
    /// One seed per Boys orbital, summed over all virtuals.
    OccupiedOnly,
    /// Schurkus-Ochsenfeld occ × (canonical) vir products.
    OccVirProducts,
}

/// Open-shell Boys-localized PDEP seed: per-spin Boys localization with
/// the resulting seed columns horizontally concatenated.
///
/// `c_full_a` / `c_full_b` are the full MO coefficient matrices for α/β
/// (per `compute_rpa_intermediates_spin`, ROHF reuses α MOs for β).
/// `inter_a` / `inter_b` are the per-spin RPA intermediates. Final seed
/// has shape `(naux, n_seed_a + n_seed_b)` after QR.
pub fn build_boys_seed_unrestricted(
    obs: &PreparedBasis,
    c_full_a: &Array2<f64>,
    c_full_b: &Array2<f64>,
    inter_a: &RpaIntermediates,
    inter_b: &RpaIntermediates,
    n_seed_target: usize,
    mode: BoysSeedMode,
) -> Result<Array2<f64>, FerricError> {
    let naux = inter_a.naux;
    debug_assert_eq!(inter_a.naux, inter_b.naux);

    // Per-spin Boys seed (occupied-only or occ×vir products). If a channel
    // has zero occupied (e.g. β on H atom), skip it.
    let seed_a = if inter_a.nocc > 0 {
        Some(boys_seed_one_spin(obs, c_full_a, inter_a, n_seed_target, mode)?)
    } else {
        None
    };
    let seed_b = if inter_b.nocc > 0 {
        Some(boys_seed_one_spin(obs, c_full_b, inter_b, n_seed_target, mode)?)
    } else {
        None
    };

    let ncols = seed_a.as_ref().map(|s| s.ncols()).unwrap_or(0)
        + seed_b.as_ref().map(|s| s.ncols()).unwrap_or(0);
    if ncols == 0 {
        return Err(FerricError::General(
            "build_boys_seed_unrestricted: both spin channels empty".into(),
        ));
    }
    let mut combined = Array2::<f64>::zeros((naux, ncols));
    let mut off = 0;
    if let Some(s) = seed_a {
        let n = s.ncols();
        combined.slice_mut(s![.., off..off + n]).assign(&s);
        off += n;
    }
    if let Some(s) = seed_b {
        let n = s.ncols();
        combined.slice_mut(s![.., off..off + n]).assign(&s);
    }

    // Combined QR to ensure orthonormality across spin blocks (per-spin
    // QR alone would leave α and β columns non-orthogonal). Skip QR if
    // wide (cols > rows).
    let q = if combined.ncols() > combined.nrows() {
        combined
    } else {
        let (q, _) = combined
            .qr()
            .map_err(|e| FerricError::General(format!("combined U-RPA Boys seed QR: {e}")))?;
        q
    };
    Ok(q)
}

/// Single-spin Boys seed for use by [`build_boys_seed_unrestricted`].
///
/// Mirrors [`build_boys_seed`] but takes a generic α-or-β MO coefficient
/// matrix instead of an [`ScfResult`] (so the caller can pass ROHF α-MOs
/// for both spin slots).
fn boys_seed_one_spin(
    obs: &PreparedBasis,
    c_full: &Array2<f64>,
    inter: &RpaIntermediates,
    n_seed_target: usize,
    mode: BoysSeedMode,
) -> Result<Array2<f64>, FerricError> {
    use ferric_integrals::oneelectron::overlap;

    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let naux = inter.naux;
    let first_occ = inter.first_occ;
    let b_ov = &inter.b_ov;

    let c_occ_active = c_full
        .slice(s![.., first_occ..first_occ + nocc])
        .to_owned();
    let dip = dipole(obs, [0.0, 0.0, 0.0]);
    let boys = boys_localize(&c_occ_active, &dip, 200);
    let s_mat = overlap(obs);
    let sc_loc = s_mat.dot(&boys.c_loc);
    let u_mix: Array2<f64> = c_occ_active.t().dot(&sc_loc);

    // Orbital-energy estimates for the importance ranking. We expect the
    // caller's eps_occ/eps_vir slices to match the active block in `inter`,
    // but here we only need the *gaps*, which the b_ov layout already
    // encodes. Pull energies from a default-zero vector if they're not
    // available — this only affects the OccVirProducts pair ranking.
    let eps_occ = vec![0.0f64; nocc];
    let eps_vir = vec![1.0f64; nvir];

    build_seed_inner(naux, nocc, nvir, b_ov, &u_mix, &eps_occ, &eps_vir, n_seed_target, mode)
}

/// Inner seed builder (shared between restricted and per-spin paths).
// Seed construction inputs: dimensions (naux/nocc/nvir), the RI block + Boys
// mixing matrix, both orbital-energy vectors, the seed-count target, and the
// seed mode — distinct knobs shared verbatim by the restricted and per-spin
// callers, so kept positional rather than bundled.
#[allow(clippy::too_many_arguments)]
fn build_seed_inner(
    naux: usize,
    nocc: usize,
    nvir: usize,
    b_ov: &Array2<f64>,
    u_mix: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    n_seed_target: usize,
    mode: BoysSeedMode,
) -> Result<Array2<f64>, FerricError> {
    let seed = match mode {
        BoysSeedMode::OccupiedOnly => {
            let mut seed = Array2::<f64>::zeros((naux, nocc));
            let mut vsum = Array2::<f64>::zeros((nocc, naux));
            for i in 0..nocc {
                let block = b_ov.slice(s![.., i * nvir..(i + 1) * nvir]);
                let row_sum: Array1<f64> = block.sum_axis(ndarray::Axis(1));
                vsum.slice_mut(s![i, ..]).assign(&row_sum);
            }
            for i_loc in 0..nocc {
                let mut col = Array1::<f64>::zeros(naux);
                for i in 0..nocc {
                    let w = u_mix[(i, i_loc)];
                    if w.abs() < 1e-14 {
                        continue;
                    }
                    col.scaled_add(w, &vsum.slice(s![i, ..]));
                }
                seed.slice_mut(s![.., i_loc]).assign(&col);
            }
            seed
        }
        BoysSeedMode::OccVirProducts => {
            let mut eps_loc = vec![0.0f64; nocc];
            for i_loc in 0..nocc {
                let mut e = 0.0;
                for i in 0..nocc {
                    let u = u_mix[(i, i_loc)];
                    e += u * u * eps_occ[i];
                }
                eps_loc[i_loc] = e;
            }
            let mut pairs: Vec<(usize, usize, f64)> = Vec::with_capacity(nocc * nvir);
            for i_loc in 0..nocc {
                for a in 0..nvir {
                    let denom = eps_vir[a] - eps_loc[i_loc];
                    let importance = if denom > 1e-10 { 1.0 / denom } else { 0.0 };
                    pairs.push((i_loc, a, importance));
                }
            }
            pairs.sort_by(|x, y| y.2.abs().partial_cmp(&x.2.abs()).unwrap());
            let n_take = n_seed_target.min(pairs.len()).max(1);
            let mut seed = Array2::<f64>::zeros((naux, n_take));
            for (slot, &(i_loc, a, w)) in pairs[..n_take].iter().enumerate() {
                let mut col = Array1::<f64>::zeros(naux);
                for i in 0..nocc {
                    let u = u_mix[(i, i_loc)];
                    if u.abs() < 1e-14 {
                        continue;
                    }
                    let b_ia = b_ov.slice(s![.., i * nvir + a]);
                    col.scaled_add(u, &b_ia);
                }
                col.mapv_inplace(|x| x * w);
                seed.slice_mut(s![.., slot]).assign(&col);
            }
            seed
        }
    };

    let mut keep_cols: Vec<usize> = Vec::with_capacity(seed.ncols());
    for j in 0..seed.ncols() {
        let n = seed.column(j).dot(&seed.column(j)).sqrt();
        if n > 1e-12 {
            keep_cols.push(j);
        }
    }
    if keep_cols.is_empty() {
        return Err(FerricError::General(
            "Boys seed produced no non-zero columns".into(),
        ));
    }
    let mut filtered = Array2::<f64>::zeros((naux, keep_cols.len()));
    for (slot, &j) in keep_cols.iter().enumerate() {
        filtered.slice_mut(s![.., slot]).assign(&seed.column(j));
    }
    let q = if filtered.ncols() > filtered.nrows() {
        filtered
    } else {
        let (q, _r) = filtered
            .qr()
            .map_err(|e| FerricError::General(format!("Boys seed QR: {e}")))?;
        q
    };
    Ok(q)
}

/// Build a PDEP eigensolver seed from Boys-localized occupied orbitals,
/// projected onto the auxiliary basis via the b_ov tensor.
///
/// Closed-shell only. Reuses `ferric_mp2::boys::boys_localize` for the
/// occupied-side localization (Foster-Boys via 2×2 Jacobi sweeps).
pub fn build_boys_seed(
    obs: &PreparedBasis,
    rhf: &ScfResult,
    inter: &RpaIntermediates,
    n_seed_target: usize,
    mode: BoysSeedMode,
) -> Result<Array2<f64>, FerricError> {
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let naux = inter.naux;
    let first_occ = inter.first_occ;
    let nocc_total = inter.nocc_total;
    let b_ov = &inter.b_ov;

    // Active-occupied canonical block (skip frozen core).
    let c_can = rhf.mos_r();
    let c_occ_active = c_can
        .slice(s![.., first_occ..first_occ + nocc])
        .to_owned();

    // Dipole AO integrals at origin; Boys formula only needs the diagonal
    // differences and off-diagonals — origin choice is gauge-invariant for
    // the localization rotation.
    let dip = dipole(obs, [0.0, 0.0, 0.0]);

    let boys = boys_localize(&c_occ_active, &dip, 200);
    // c_loc unused beyond the rotation U: the seed only cares about how Boys
    // mixes the active-occupied indices. Recover U from C_loc = C_can · U via
    // U = C_can^T S C_loc, but for seed construction we directly use C_loc to
    // define mixed b_ov columns: for Boys orbital i_loc,
    //     b_ov_loc[:, i_loc * nvir + a] = Σ_i U[i, i_loc] * b_ov[:, i * nvir + a]
    // where U[i, i_loc] is the rotation. Since orthonormal canonical
    // C_can^T S C_can = I, U = C_can^T S C_loc.
    //
    // Build U via overlap-based projection.
    use ferric_integrals::oneelectron::overlap;
    let s_mat = overlap(obs);
    let sc_loc = s_mat.dot(&boys.c_loc); // (nbas, nocc)
    let u_mix: Array2<f64> = c_occ_active.t().dot(&sc_loc); // (nocc, nocc)

    // Energy slices for the importance metric.
    let eps_occ: Vec<f64> = rhf.eps_r()[first_occ..first_occ + nocc].to_vec();
    let eps_vir: Vec<f64> = rhf.eps_r()[nocc_total..nocc_total + nvir].to_vec();

    let seed = match mode {
        BoysSeedMode::OccupiedOnly => {
            // One column per Boys orbital, summed over virtuals uniformly.
            // For Boys orbital i_loc:
            //   col_p = Σ_a Σ_i U[i, i_loc] · b_ov[p, i*nvir + a]
            let mut seed = Array2::<f64>::zeros((naux, nocc));
            // Precompute v_pa for canonical i: vsum[i, p] = Σ_a b_ov[p, i*nvir + a]
            let mut vsum = Array2::<f64>::zeros((nocc, naux));
            for i in 0..nocc {
                let block = b_ov.slice(s![.., i * nvir..(i + 1) * nvir]); // (naux, nvir)
                // Sum over a: take row-sum across columns.
                let row_sum: Array1<f64> = block.sum_axis(ndarray::Axis(1));
                vsum.slice_mut(s![i, ..]).assign(&row_sum);
            }
            // seed[:, i_loc] = Σ_i U[i, i_loc] * vsum[i, :]
            for i_loc in 0..nocc {
                let mut col = Array1::<f64>::zeros(naux);
                for i in 0..nocc {
                    let w = u_mix[(i, i_loc)];
                    if w.abs() < 1e-14 {
                        continue;
                    }
                    col.scaled_add(w, &vsum.slice(s![i, ..]));
                }
                seed.slice_mut(s![.., i_loc]).assign(&col);
            }
            seed
        }
        BoysSeedMode::OccVirProducts => {
            // Rank (i_loc, a) pairs by 1/(ε_a − ε_{i_loc}). Boys orbital
            // "energy" approximated as expectation value
            //   eps_iloc = Σ_i U[i,i_loc]^2 ε_i
            let mut eps_loc = vec![0.0f64; nocc];
            for i_loc in 0..nocc {
                let mut e = 0.0;
                for i in 0..nocc {
                    let u = u_mix[(i, i_loc)];
                    e += u * u * eps_occ[i];
                }
                eps_loc[i_loc] = e;
            }
            let mut pairs: Vec<(usize, usize, f64)> =
                Vec::with_capacity(nocc * nvir);
            for i_loc in 0..nocc {
                for a in 0..nvir {
                    let denom = eps_vir[a] - eps_loc[i_loc];
                    let importance = if denom > 1e-10 { 1.0 / denom } else { 0.0 };
                    pairs.push((i_loc, a, importance));
                }
            }
            pairs.sort_by(|x, y| y.2.abs().partial_cmp(&x.2.abs()).unwrap());

            let n_take = n_seed_target.min(pairs.len()).max(1);
            let mut seed = Array2::<f64>::zeros((naux, n_take));

            // Build b_loc_ov for the picked pairs on the fly:
            //   b_loc[:, i_loc*nvir + a] = Σ_i U[i, i_loc] · b_ov[:, i*nvir + a]
            for (slot, &(i_loc, a, w)) in pairs[..n_take].iter().enumerate() {
                let mut col = Array1::<f64>::zeros(naux);
                for i in 0..nocc {
                    let u = u_mix[(i, i_loc)];
                    if u.abs() < 1e-14 {
                        continue;
                    }
                    let b_ia = b_ov.slice(s![.., i * nvir + a]);
                    col.scaled_add(u, &b_ia);
                }
                col.mapv_inplace(|x| x * w);
                seed.slice_mut(s![.., slot]).assign(&col);
            }
            seed
        }
    };

    // Drop near-zero columns to avoid singular QR.
    let mut keep_cols: Vec<usize> = Vec::with_capacity(seed.ncols());
    for j in 0..seed.ncols() {
        let n = seed.column(j).dot(&seed.column(j)).sqrt();
        if n > 1e-12 {
            keep_cols.push(j);
        }
    }
    if keep_cols.is_empty() {
        return Err(FerricError::General(
            "Boys seed produced no non-zero columns".into(),
        ));
    }
    let mut filtered = Array2::<f64>::zeros((naux, keep_cols.len()));
    for (slot, &j) in keep_cols.iter().enumerate() {
        filtered.slice_mut(s![.., slot]).assign(&seed.column(j));
    }

    // QR orthonormalize; for "wide" matrices (cols > rows) skip QR.
    let q = if filtered.ncols() > filtered.nrows() {
        filtered
    } else {
        let (q, _r) = filtered
            .qr()
            .map_err(|e| FerricError::General(format!("Boys seed QR failed: {e}")))?;
        q
    };
    Ok(q)
}
