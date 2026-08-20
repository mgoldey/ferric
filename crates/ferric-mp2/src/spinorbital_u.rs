//! Unrestricted (per-spin) spin-orbital integral builders.
//!
//! The builders in [`crate::spinorbital`] map spin-orbital index `p` to spatial `p >> 1`,
//! i.e. they assume **one** spatial orbital set shared by both spins. That holds for an
//! RHF reference, but not for UHF or for semi-canonicalized ROHF, where α and β have
//! genuinely different spatial orbitals.
//!
//! These builders take the spatial chemist blocks **per spin combination** and assemble
//! the same antisymmetrized `<pq||rs>` tensors. Spin convention is unchanged: even index
//! = α, odd = β, so a consumer that already works with [`crate::spinorbital`] output
//! needs no other change.
//!
//! # Which spatial blocks are needed
//!
//! Chemist `(pq|rs)` vanishes unless `spin(p) == spin(q)` and `spin(r) == spin(s)`, so
//! each 4-index block needs three distinct spatial builds — `αα|αα`, `αα|ββ`, `ββ|ββ` —
//! with `ββ|αα` recovered as the transpose of `αα|ββ` by the bra–ket symmetry
//! `(pq|rs) = (rs|pq)`. Building it explicitly instead would double the transform cost
//! for no information.

use crate::spinorbital::{spat, spin};
use ndarray::{ArrayD, IxDyn};

/// The three distinct spin blocks of a spatial chemist integral tensor.
///
/// `aa`/`bb` are the same-spin blocks; `ab` is the mixed block with the FIRST pair α and
/// the SECOND pair β. The `βα` block is obtained by transposing `ab`'s pairs, exploiting
/// `(pq|rs) = (rs|pq)`.
#[derive(Debug)]
pub struct SpinBlocks<'a> {
    /// `(pq|rs)` with all four indices α.
    pub aa: &'a ArrayD<f64>,
    /// `(pq|rs)` with `p,q` α and `r,s` β.
    pub ab: &'a ArrayD<f64>,
    /// `(pq|rs)` with all four indices β.
    pub bb: &'a ArrayD<f64>,
}

impl SpinBlocks<'_> {
    /// Fetch `(pq|rs)` for arbitrary spins, or `None` when the integral vanishes.
    ///
    /// Chemist notation pairs `(pq|` and `|rs)`, so the integral is zero unless each
    /// pair is spin-diagonal. The dimensions of each returned array are those of its own
    /// spin block, which is why the caller must pass spatial indices already resolved
    /// for the right spin.
    #[inline]
    fn get(
        &self,
        sp: usize,
        sq: usize,
        sr: usize,
        ss: usize,
        p: usize,
        q: usize,
        r: usize,
        s: usize,
    ) -> f64 {
        if sp != sq || sr != ss {
            return 0.0;
        }
        match (sp, sr) {
            (0, 0) => self.aa[[p, q, r, s]],
            (0, 1) => self.ab[[p, q, r, s]],
            // (pq|rs) = (rs|pq): the beta-alpha block is the alpha-beta block with the
            // pairs swapped, so no separate transform is needed.
            (1, 0) => self.ab[[r, s, p, q]],
            _ => self.bb[[p, q, r, s]],
        }
    }
}

/// Unrestricted `<ij||ab>` (oovv) from per-spin chemist `(ia|jb)` blocks.
///
/// `blocks.aa[i,a,j,b] = (i_α a_α | j_α b_α)`, `blocks.ab[i,a,j,b] = (i_α a_α | j_β b_β)`,
/// `blocks.bb` likewise all-β. `no_a`/`no_b` and `nv_a`/`nv_b` are the per-spin occupied
/// and virtual counts.
///
/// Output shape `(no_a + no_b, no_a + no_b, nv_a + nv_b, nv_a + nv_b)` in the interleaved
/// spin-orbital ordering (even = α, odd = β) — identical in layout to
/// [`crate::spinorbital::asym_oovv`], so downstream code is unchanged.
///
/// # Unequal spin dimensions
///
/// α and β generally have different occupied counts (that is what makes the system open
/// shell). The interleaved ordering therefore only accommodates spin-orbital index `p`
/// when `spat(p)` is in range for `spin(p)`; out-of-range combinations contribute zero.
/// Callers should pad to `max` per axis and treat the surplus as unoccupied/absent —
/// [`interleaved_dims`] computes the padded extents.
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
pub fn u_asym_oovv(
    blocks: &SpinBlocks<'_>,
    no_a: usize,
    no_b: usize,
    nv_a: usize,
    nv_b: usize,
) -> ArrayD<f64> {
    let (no2, nv2) = interleaved_dims(no_a, no_b, nv_a, nv_b);
    let mut out = ArrayD::zeros(IxDyn(&[no2, no2, nv2, nv2]));
    let no = [no_a, no_b];
    let nv = [nv_a, nv_b];

    for i in 0..no2 {
        for j in 0..no2 {
            for a in 0..nv2 {
                for b in 0..nv2 {
                    let (si, sj, sa, sb) = (spin(i), spin(j), spin(a), spin(b));
                    let (pi, pj, pa, pb) = (spat(i), spat(j), spat(a), spat(b));
                    // Skip padded slots that do not correspond to a real orbital.
                    if pi >= no[si] || pj >= no[sj] || pa >= nv[sa] || pb >= nv[sb] {
                        continue;
                    }
                    // <ij|ab> = (ia|jb)
                    let dir = blocks.get(si, sa, sj, sb, pi, pa, pj, pb);
                    // <ij|ba> = (ib|ja)
                    let exc = blocks.get(si, sb, sj, sa, pi, pb, pj, pa);
                    out[[i, j, a, b]] = dir - exc;
                }
            }
        }
    }
    out
}

/// Unrestricted same-space `<pq||rs>` from per-spin chemist `(pq|rs)` blocks.
///
/// Used for the `oooo` (hole–hole ladder) and `vvvv` (particle–particle ladder) blocks.
/// `n_a`/`n_b` are the per-spin dimensions of that space.
///
/// `blocks.aa[p,r,q,s] = (p_α r_α | q_α s_α)` — note the chemist index order matches
/// [`crate::spinorbital::asym_same`]: `<pq||rs> = (pr|qs) − (ps|qr)`.
#[allow(clippy::needless_range_loop)]
pub fn u_asym_same(blocks: &SpinBlocks<'_>, n_a: usize, n_b: usize) -> ArrayD<f64> {
    let n2 = 2 * n_a.max(n_b);
    let mut out = ArrayD::zeros(IxDyn(&[n2, n2, n2, n2]));
    let n = [n_a, n_b];

    for p in 0..n2 {
        for q in 0..n2 {
            for r in 0..n2 {
                for s in 0..n2 {
                    let (sp, sq, sr, ss) = (spin(p), spin(q), spin(r), spin(s));
                    let (xp, xq, xr, xs) = (spat(p), spat(q), spat(r), spat(s));
                    if xp >= n[sp] || xq >= n[sq] || xr >= n[sr] || xs >= n[ss] {
                        continue;
                    }
                    // <pq|rs> = (pr|qs)
                    let dir = blocks.get(sp, sr, sq, ss, xp, xr, xq, xs);
                    // <pq|sr> = (ps|qr)
                    let exc = blocks.get(sp, ss, sq, sr, xp, xs, xq, xr);
                    out[[p, q, r, s]] = dir - exc;
                }
            }
        }
    }
    out
}

/// Interleaved spin-orbital extents for per-spin dimensions.
///
/// The interleaved layout (even = α, odd = β) needs `2·max(n_α, n_β)` slots per axis
/// when the two spins differ in size; the surplus slots are padding that the builders
/// leave at zero.
pub fn interleaved_dims(
    no_a: usize,
    no_b: usize,
    nv_a: usize,
    nv_b: usize,
) -> (usize, usize) {
    (2 * no_a.max(no_b), 2 * nv_a.max(nv_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spinorbital::{asym_oovv, asym_same};

    /// When both spins share the same spatial orbitals, the unrestricted builders must
    /// reproduce the restricted ones EXACTLY.
    ///
    /// This is the load-bearing test: it pins the new code against the already-validated
    /// restricted path, so any index or sign error shows up immediately.
    #[test]
    fn reduces_to_the_restricted_builders_when_spins_match() {
        let (no, nv) = (3usize, 4usize);
        // Deterministic pseudo-random spatial block with the right symmetry.
        let mut g = ArrayD::<f64>::zeros(IxDyn(&[no, nv, no, nv]));
        let mut seed = 1u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        for i in 0..no {
            for a in 0..nv {
                for j in 0..no {
                    for b in 0..nv {
                        g[[i, a, j, b]] = next();
                    }
                }
            }
        }
        // Enforce (ia|jb) = (jb|ia).
        for i in 0..no {
            for a in 0..nv {
                for j in 0..no {
                    for b in 0..nv {
                        let v = 0.5 * (g[[i, a, j, b]] + g[[j, b, i, a]]);
                        g[[i, a, j, b]] = v;
                        g[[j, b, i, a]] = v;
                    }
                }
            }
        }

        let want = asym_oovv(&g, no, nv);
        let blocks = SpinBlocks { aa: &g, ab: &g, bb: &g };
        let got = u_asym_oovv(&blocks, no, no, nv, nv);

        assert_eq!(want.shape(), got.shape());
        let max_dev =
            want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(
            max_dev < 1e-15,
            "u_asym_oovv disagrees with asym_oovv on a shared spatial set: {max_dev:.3e}"
        );
    }

    #[test]
    fn same_space_builder_reduces_to_the_restricted_one() {
        let n = 4usize;
        let mut g = ArrayD::<f64>::zeros(IxDyn(&[n, n, n, n]));
        let mut seed = 7u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        g[[p, q, r, s]] = next();
                    }
                }
            }
        }
        // Impose the FULL 8-fold chemist symmetry by averaging over the whole orbit
        // and writing every member. A partial average (four of the eight images,
        // written back to only those four slots) does NOT converge to a symmetric
        // tensor -- later writes clobber earlier ones and the result fails
        // (pq|rs) == (rs|pq), which is exactly what u_asym_same's beta-alpha
        // transpose relies on.
        let mut sym = ArrayD::<f64>::zeros(IxDyn(&[n, n, n, n]));
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        let orbit = [
                            [p, q, r, s], [q, p, r, s], [p, q, s, r], [q, p, s, r],
                            [r, s, p, q], [s, r, p, q], [r, s, q, p], [s, r, q, p],
                        ];
                        let avg: f64 =
                            orbit.iter().map(|ix| g[[ix[0], ix[1], ix[2], ix[3]]]).sum::<f64>() / 8.0;
                        sym[[p, q, r, s]] = avg;
                    }
                }
            }
        }
        let g = sym;

        let want = asym_same(&g, n);
        let blocks = SpinBlocks { aa: &g, ab: &g, bb: &g };
        let got = u_asym_same(&blocks, n, n);
        let max_dev =
            want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(
            max_dev < 1e-15,
            "u_asym_same disagrees with asym_same on a shared spatial set: {max_dev:.3e}"
        );
    }

    /// Antisymmetry `<ij||ab> = -<ji||ab> = -<ij||ba>` must hold for genuinely
    /// different alpha/beta spatial blocks, not just the degenerate shared case.
    #[test]
    fn antisymmetry_holds_with_distinct_spin_blocks() {
        let (no, nv) = (2usize, 3usize);
        let mk = |seed0: u64| {
            let mut g = ArrayD::<f64>::zeros(IxDyn(&[no, nv, no, nv]));
            let mut seed = seed0;
            for i in 0..no {
                for a in 0..nv {
                    for j in 0..no {
                        for b in 0..nv {
                            seed = seed
                                .wrapping_mul(6364136223846793005)
                                .wrapping_add(1442695040888963407);
                            g[[i, a, j, b]] = ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
                        }
                    }
                }
            }
            for i in 0..no {
                for a in 0..nv {
                    for j in 0..no {
                        for b in 0..nv {
                            let v = 0.5 * (g[[i, a, j, b]] + g[[j, b, i, a]]);
                            g[[i, a, j, b]] = v;
                            g[[j, b, i, a]] = v;
                        }
                    }
                }
            }
            g
        };
        let (ga, gab, gb) = (mk(11), mk(29), mk(53));
        let blocks = SpinBlocks { aa: &ga, ab: &gab, bb: &gb };
        let v = u_asym_oovv(&blocks, no, no, nv, nv);

        let (no2, nv2) = (2 * no, 2 * nv);
        let mut max_dev = 0.0f64;
        for i in 0..no2 {
            for j in 0..no2 {
                for a in 0..nv2 {
                    for b in 0..nv2 {
                        max_dev = max_dev.max((v[[i, j, a, b]] + v[[j, i, a, b]]).abs());
                        max_dev = max_dev.max((v[[i, j, a, b]] + v[[i, j, b, a]]).abs());
                    }
                }
            }
        }
        assert!(max_dev < 1e-15, "antisymmetry violated with distinct spin blocks: {max_dev:.3e}");
    }
}
