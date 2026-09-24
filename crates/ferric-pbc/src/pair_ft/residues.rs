//! Stage 3: the lattice pair FT resolved by the residue of the image
//! translation modulo a k-mesh, so ONE pass serves every mesh k-point.
//!
//! ```text
//! Q_r[m, n, g] = Σ_{L : n(L) ≡ r (mod M)} ∫ φ_m(r) φ_n(r − L) e^{−i K_g · r} dr
//! P^{k}(K)     = Σ_L e^{ik·L} ∫ φ_m φ_n(· − L) e^{−iK·r} = Σ_r e^{ik·t_r} Q_r(K)
//! ```
//!
//! with `n(L)` the integer coordinates of `L = Σ n_i a_i`, `M` the mesh's
//! [`KPointMesh::residue_moduli`](crate::kpts::KPointMesh::residue_moduli)
//! and `e^{ik·t_r}` = [`KPointMesh::phase`](crate::kpts::KPointMesh::phase)
//! at the residue `r` (exact: the phase depends on `n mod M` only). `K` may
//! be any vectors — `K = G + q` for momentum transfer `q = k' − k` — so
//! `P^{k'}(G + k' − k)` is the FT of the Bloch pair `χ*_{mk} χ_{nk'}` over one
//! cell (FINDINGS "Iteration 9"; the phase rides on the SECOND index's k).
//!
//! Sum over the buckets reproduces [`super::pair_ft()`] (the Gamma pair FT, all
//! phases 1) up to the order in which images are added. The kernel is a
//! bucketed copy of `super::pair_ft_block` (same screening, same G window,
//! same Cartesian→AO transform), kept separate so the Gamma kernel stays
//! byte-identical.

use super::{
    basis_lmax, build_shells, max_gnorm, pair_ft_bytes_per_g, validate_inputs, WINDOW_MARGIN,
};
use crate::budget::{bytes_of, Ledger};
use crate::kpts::lattice_coords;
use crate::lattice::Cell;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::md3c1e::{cart_components, e_table, ferric_cart2sph};
use ndarray::Array3;
use num_complex::Complex64;

/// Row-major residue index of integer lattice coordinates `nl` modulo
/// `moduli` (`r = (r0 M1 + r1) M2 + r2`).
pub fn residue_index(nl: [i64; 3], moduli: [usize; 3]) -> usize {
    let r0 = nl[0].rem_euclid(moduli[0] as i64) as usize;
    let r1 = nl[1].rem_euclid(moduli[1] as i64) as usize;
    let r2 = nl[2].rem_euclid(moduli[2] as i64) as usize;
    (r0 * moduli[1] + r1) * moduli[2] + r2
}

/// Integer coordinates of residue `r` (inverse of [`residue_index`]).
pub fn residue_coords(r: usize, moduli: [usize; 3]) -> [i64; 3] {
    [
        (r / (moduli[1] * moduli[2])) as i64,
        ((r / moduli[2]) % moduli[1]) as i64,
        (r % moduli[2]) as i64,
    ]
}

/// Residue-resolved pair FT in K chunks: `sink(k0, ks, Q)` receives the
/// `R = Π moduli` arrays `Q[r]` (each `(nbf, nbf, ks.len())`) for
/// `ks = kvecs[k0..]`, in order. Each chunk (`R ×` [`pair_ft_bytes_per_g`]
/// plus `extra_bytes_per_g`, per K) fits in `chunk_budget_bytes`; if one K
/// does not, the call fails before allocating. Returns the chunk count.
/// The per-primitive-pair K window uses `max |K|` over ALL of `kvecs`, so
/// chunking does not change a bit (as [`super::pair_ft_chunked`]).
#[allow(clippy::too_many_arguments)]
pub fn pair_ft_residues_chunked<F>(
    cell: &Cell,
    prep: &PreparedBasis,
    kvecs: &[[f64; 3]],
    moduli: [usize; 3],
    thresh: f64,
    chunk_budget_bytes: usize,
    extra_bytes_per_g: usize,
    mut sink: F,
) -> Result<usize, FerricError>
where
    F: FnMut(usize, &[[f64; 3]], &[Array3<Complex64>]) -> Result<(), FerricError>,
{
    validate_inputs(kvecs, thresh)?;
    if moduli.contains(&0) {
        return Err(FerricError::General(format!(
            "pair_ft_residues: moduli must be >= 1, got {moduli:?}"
        )));
    }
    if kvecs.is_empty() {
        return Ok(0);
    }
    let nr = moduli[0] * moduli[1] * moduli[2];
    let nao = prep.nbasis();
    let lmax = basis_lmax(prep);
    // R output columns + R Cartesian accumulators per K (the scratch of
    // pair_ft_bytes_per_g covers one of each).
    let per_g =
        bytes_of(nr as u64, pair_ft_bytes_per_g(nao, lmax)).saturating_add(extra_bytes_per_g);
    Ledger::new(chunk_budget_bytes).check(
        &format!("pair_ft_residues chunk, one K vector (nao = {nao}, lmax = {lmax}, R = {nr})"),
        per_g,
    )?;
    let chunk = (chunk_budget_bytes / per_g).max(1);
    let gmax = max_gnorm(kvecs);
    let mut n_chunks = 0usize;
    for (c, ks) in kvecs.chunks(chunk).enumerate() {
        let q = residue_block(cell, prep, ks, moduli, thresh, gmax)?;
        sink(c * chunk, ks, &q)?;
        n_chunks += 1;
    }
    Ok(n_chunks)
}

/// Bucketed copy of `super::pair_ft_block`.
fn residue_block(
    cell: &Cell,
    prep: &PreparedBasis,
    gvecs: &[[f64; 3]],
    moduli: [usize; 3],
    thresh: f64,
    gmax_window: f64,
) -> Result<Vec<Array3<Complex64>>, FerricError> {
    let shells = build_shells(cell, prep)?;
    let nbf = prep.nbasis();
    let ng = gvecs.len();
    let nr = moduli[0] * moduli[1] * moduli[2];
    let mut out: Vec<Array3<Complex64>> = (0..nr)
        .map(|_| Array3::<Complex64>::zeros((nbf, nbf, ng)))
        .collect();
    if ng == 0 || shells.is_empty() {
        return Ok(out);
    }
    let amin = shells
        .iter()
        .flat_map(|s| s.exps.iter().copied())
        .fold(f64::INFINITY, f64::min);
    if !(amin > 0.0) {
        return Err(FerricError::Basis(format!(
            "pair_ft_residues: smallest exponent is {amin}; must be > 0"
        )));
    }
    let rpair = (2.0 * (1e3 / thresh).ln() / amin).sqrt() + 2.0;
    let images = cell.translations(rpair)?;
    let b = cell.reciprocal();
    let bucket: Vec<usize> = images
        .iter()
        .map(|l| residue_index(lattice_coords(&b, l), moduli))
        .collect();

    let norm2 = |g: &[f64; 3]| g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    let mut order: Vec<usize> = (0..ng).collect();
    order.sort_by(|&x, &y| norm2(&gvecs[x]).total_cmp(&norm2(&gvecs[y])));
    let gsorted: Vec<[f64; 3]> = order.iter().map(|&i| gvecs[i]).collect();
    let gvecs: &[[f64; 3]] = &gsorted;
    let gmax = gmax_window;

    let lmax = shells.iter().map(|s| s.l).max().unwrap_or(0);
    let nt = 2 * lmax + 1;
    let g2: Vec<f64> = gvecs.iter().map(norm2).collect();
    let pw: Vec<Vec<Complex64>> = (0..3)
        .map(|d| {
            let mut v = vec![Complex64::new(0.0, 0.0); nt * ng];
            for (g, gv) in gvecs.iter().enumerate() {
                let step = Complex64::new(0.0, -gv[d]);
                let mut acc = Complex64::new(1.0, 0.0);
                for t in 0..nt {
                    v[t * ng + g] = acc;
                    acc *= step;
                }
            }
            v
        })
        .collect();
    let comps: Vec<Vec<[u8; 3]>> = (0..=lmax).map(cart_components).collect();
    let c2s: Vec<Vec<f64>> = (0..=lmax).map(ferric_cart2sph).collect();

    let zero = Complex64::new(0.0, 0.0);
    let pi = std::f64::consts::PI;
    let mut common = vec![zero; ng];
    let mut ebuf: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut fbuf: [Vec<Complex64>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for sa in &shells {
        for sb in &shells {
            let (la, lb) = (sa.l, sb.l);
            let (nca, ncb) = (sa.ncart, sb.ncart);
            let st = la + lb + 1;
            let nij = (la + 1) * (lb + 1);
            for d in 0..3 {
                ebuf[d].resize(nij * st, 0.0);
                fbuf[d].resize(nij * ng, zero);
            }
            let mut cart: Vec<Vec<Complex64>> =
                (0..nr).map(|_| vec![zero; nca * ncb * ng]).collect();
            let ca_comps = &comps[la];
            let cb_comps = &comps[lb];

            for (il, l) in images.iter().enumerate() {
                let cart = &mut cart[bucket[il]];
                let bc = [
                    sb.center[0] + l[0],
                    sb.center[1] + l[1],
                    sb.center[2] + l[2],
                ];
                let ab = [
                    sa.center[0] - bc[0],
                    sa.center[1] - bc[1],
                    sa.center[2] - bc[2],
                ];
                let r2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
                for (&a, &ca) in sa.exps.iter().zip(&sa.coefs) {
                    for (&bb, &cb) in sb.exps.iter().zip(&sb.coefs) {
                        let p = a + bb;
                        let cc = ca * cb * (pi / p).powf(1.5);
                        let mag = (cc * (-a * bb / p * r2).exp()).abs();
                        if mag < thresh {
                            continue;
                        }
                        let g2max = 4.0
                            * p
                            * ((mag / thresh).ln().max(0.0)
                                + WINDOW_MARGIN
                                + (la + lb) as f64 * gmax.max(1.0).ln());
                        let ngp = g2.partition_point(|&x| x <= g2max);
                        if ngp == 0 {
                            continue;
                        }
                        let pc = [
                            (a * sa.center[0] + bb * bc[0]) / p,
                            (a * sa.center[1] + bb * bc[1]) / p,
                            (a * sa.center[2] + bb * bc[2]) / p,
                        ];
                        for (g, gv) in gvecs.iter().enumerate().take(ngp) {
                            let mag = cc * (-g2[g] / (4.0 * p)).exp();
                            let ph = gv[0] * pc[0] + gv[1] * pc[1] + gv[2] * pc[2];
                            common[g] = Complex64::new(mag * ph.cos(), -mag * ph.sin());
                        }
                        for d in 0..3 {
                            e_table(la, lb, a, bb, ab[d], &mut ebuf[d]);
                            let (e, f, w) = (&ebuf[d], &mut fbuf[d], &pw[d]);
                            for ij in 0..nij {
                                let frow = &mut f[ij * ng..ij * ng + ngp];
                                frow.fill(zero);
                                for t in 0..st {
                                    let et = e[ij * st + t];
                                    if et == 0.0 {
                                        continue;
                                    }
                                    let wrow = &w[t * ng..t * ng + ngp];
                                    for (x, wv) in frow.iter_mut().zip(wrow) {
                                        *x += *wv * et;
                                    }
                                }
                            }
                        }
                        let (fx, fy, fz) = (&fbuf[0], &fbuf[1], &fbuf[2]);
                        for (u, ac) in ca_comps.iter().enumerate() {
                            for (v, bcmp) in cb_comps.iter().enumerate() {
                                let ix = (ac[0] as usize * (lb + 1) + bcmp[0] as usize) * ng;
                                let iy = (ac[1] as usize * (lb + 1) + bcmp[1] as usize) * ng;
                                let iz = (ac[2] as usize * (lb + 1) + bcmp[2] as usize) * ng;
                                let dst = &mut cart[(u * ncb + v) * ng..(u * ncb + v) * ng + ngp];
                                for g in 0..ngp {
                                    dst[g] += common[g] * fx[ix + g] * fy[iy + g] * fz[iz + g];
                                }
                            }
                        }
                    }
                }
            }

            let (nfa, nfb) = (sa.nfun, sb.nfun);
            for (r, cart) in cart.into_iter().enumerate() {
                let right: Vec<Complex64> = if sb.pure {
                    let cbm = &c2s[lb];
                    let mut tmp = vec![zero; nca * nfb * ng];
                    for m in 0..nca {
                        for j in 0..nfb {
                            for n in 0..ncb {
                                let c = cbm[n * nfb + j];
                                if c == 0.0 {
                                    continue;
                                }
                                for g in 0..ng {
                                    tmp[(m * nfb + j) * ng + g] += cart[(m * ncb + n) * ng + g] * c;
                                }
                            }
                        }
                    }
                    tmp
                } else {
                    cart
                };
                let dst = &mut out[r];
                for i in 0..nfa {
                    for j in 0..nfb {
                        for g in 0..ng {
                            let val = if sa.pure {
                                let cam = &c2s[la];
                                let mut acc = zero;
                                for m in 0..nca {
                                    let c = cam[m * nfa + i];
                                    if c != 0.0 {
                                        acc += right[(m * nfb + j) * ng + g] * c;
                                    }
                                }
                                acc
                            } else {
                                right[(i * nfb + j) * ng + g]
                            };
                            dst[[sa.off + i, sb.off + j, order[g]]] = val;
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residue_index_round_trips() {
        let m = [2, 3, 4];
        for r in 0..24 {
            assert_eq!(residue_index(residue_coords(r, m), m), r);
        }
        assert_eq!(residue_index([-1, -1, -1], m), residue_index([1, 2, 3], m));
    }
}
