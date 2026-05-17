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
use ferric_scf::rhf::RhfResult;
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

/// Build a PDEP eigensolver seed from Boys-localized occupied orbitals,
/// projected onto the auxiliary basis via the b_ov tensor.
///
/// Closed-shell only. Reuses `ferric_mp2::boys::boys_localize` for the
/// occupied-side localization (Foster-Boys via 2×2 Jacobi sweeps).
pub fn build_boys_seed(
    obs: &PreparedBasis,
    rhf: &RhfResult,
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
