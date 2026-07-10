//! Compute ρ, ∇ρ, σ = |∇ρ|² on a grid from D and (χ, ∇χ).
//!
//! Closed-shell only this round. The input density matrix is expected to be
//! the **total** density (trace = N_e), as produced by ferric's `ScfResult::density_total`.

use ndarray::{Array1, Array2, Array3, Zip};
use rayon::prelude::*;

/// Below this many grid points, run the point loops serially — rayon
/// spawn/join/steal overhead dwarfs the work on tiny grids (the free-atom SAD
/// solve case: single atom, few points). Mirrors `ao_grid`'s and `vv10`'s
/// serial-fallback guards; chosen as a pure function of grid size only, never
/// of thread count, so the parallel/serial choice itself cannot perturb
/// results (both paths must produce identical output).
const PAR_MIN_PTS: usize = 512;

#[derive(Debug, Clone)]
pub struct DensityGrid {
    /// ρ(r_g), shape (npts,)
    pub rho: Array1<f64>,
    /// ∇ρ(r_g), shape (3, npts) with axis order [x, y, z]
    pub grad: Array2<f64>,
    /// σ(r_g) = |∇ρ(r_g)|², shape (npts,)
    pub sigma: Array1<f64>,
}

/// Closed-shell density on a grid, from the total density matrix.
///
/// Given a total D (tr D = N_e), χ_μ(r_g), and ∇χ_μ(r_g):
///
///   ρ(r)   = Σ_μν D_μν χ_μ(r) χ_ν(r)
///   ∇ρ(r)  = 2 Σ_μν D_μν χ_μ(r) ∇χ_ν(r)
///   σ(r)   = |∇ρ(r)|²
///
/// The ∇ρ expression uses μ↔ν symmetry of D: the full sum
/// Σ_μν D_μν (χ_μ ∇χ_ν + χ_ν ∇χ_μ) equals 2 Σ_μν D_μν χ_μ ∇χ_ν when D = Dᵀ.
///
/// The reductions over μ run row-wise (contiguous in the C-order (nbf, npts)
/// layout) with per-point accumulation order identical to the naive
/// point-major loops.
pub fn eval_density_closed(
    d: &Array2<f64>,
    chi: &Array2<f64>,        // (nbf, npts)
    dchi: &Array3<f64>,       // (3, nbf, npts)
) -> DensityGrid {
    let (nbf, npts) = chi.dim();
    debug_assert_eq!(d.dim(), (nbf, nbf));
    debug_assert_eq!(dchi.dim(), (3, nbf, npts));

    // Phi_{μg} = Σ_ν D_μν χ_νg  (one GEMM; stays outside any rayon region —
    // OpenBLAS is process-pinned to 1 thread by convention, see blas_threads.rs)
    let phi: Array2<f64> = d.dot(chi);

    // ρ_g = Σ_μ χ_μg · Φ_μg, and ∇ρ_ag = 2 Σ_μ Φ_μg · ∂_a χ_μg.
    //
    // Restructured point-outer / μ-inner (vs. the original μ-outer / point-inner
    // row sweep) so grid points — independent, disjoint output slots — can be
    // split across rayon workers while each point's own Σ_μ accumulation runs
    // in the same ascending-μ order as before. That makes the result bit-
    // identical to the old serial code and to itself across thread counts:
    // this only changes which loop nest visits (μ, g) pairs, not the order in
    // which terms are added into each output element.
    let mut rho = Array1::<f64>::zeros(npts);
    let mut grad = Array2::<f64>::zeros((3, npts));
    let chi_s = chi.as_standard_layout();
    let phi_s = phi.as_standard_layout();
    let dchi_s = dchi.as_standard_layout();
    let compute_point = |g: usize| -> (f64, [f64; 3]) {
        let mut r = 0.0_f64;
        let mut gx = 0.0_f64;
        let mut gy = 0.0_f64;
        let mut gz = 0.0_f64;
        for mu in 0..nbf {
            let c = chi_s[(mu, g)];
            let p = phi_s[(mu, g)];
            r += c * p;
            gx += p * dchi_s[(0, mu, g)];
            gy += p * dchi_s[(1, mu, g)];
            gz += p * dchi_s[(2, mu, g)];
        }
        (r, [2.0 * gx, 2.0 * gy, 2.0 * gz])
    };
    if npts >= PAR_MIN_PTS {
        let results: Vec<(f64, [f64; 3])> = (0..npts).into_par_iter().map(compute_point).collect();
        for (g, (r, gv)) in results.into_iter().enumerate() {
            rho[g] = r;
            grad[(0, g)] = gv[0];
            grad[(1, g)] = gv[1];
            grad[(2, g)] = gv[2];
        }
    } else {
        for g in 0..npts {
            let (r, gv) = compute_point(g);
            rho[g] = r;
            grad[(0, g)] = gv[0];
            grad[(1, g)] = gv[1];
            grad[(2, g)] = gv[2];
        }
    }

    let mut sigma = Array1::<f64>::zeros(npts);
    Zip::from(&mut sigma)
        .and(grad.row(0))
        .and(grad.row(1))
        .and(grad.row(2))
        .for_each(|s, &gx, &gy, &gz| *s = gx * gx + gy * gy + gz * gz);

    DensityGrid { rho, grad, sigma }
}

/// Open-shell density on a grid, from separate α/β density matrices.
///
/// Each `D_σ` should have `tr(D_σ) = N_σ`. Returns:
///   `rho_α`, `rho_β`  : per-spin densities (each integrating to N_σ)
///   `grad_α`, `grad_β`: per-spin density gradients, shape (3, npts)
///   `sigma`           : (3, npts) — rows are σ_αα, σ_αβ, σ_ββ
#[derive(Debug, Clone)]
pub struct UksDensityGrid {
    pub rho_a: Array1<f64>,
    pub rho_b: Array1<f64>,
    pub grad_a: Array2<f64>,
    pub grad_b: Array2<f64>,
    /// (3, npts) — sigma[(0, g)] = σ_αα, sigma[(1, g)] = σ_αβ, sigma[(2, g)] = σ_ββ.
    pub sigma: Array2<f64>,
}

/// Fused α/β evaluation: two GEMMs (one per spin — unavoidable, D differs),
/// then a single pass over χ / ∇χ accumulating both spins at once, and all
/// three σ channels (αα, αβ, ββ) computed together. χ and ∇χ are read once
/// instead of twice, and σ_αα/σ_ββ are not computed twice as they were when
/// this delegated to `eval_density_closed` per spin.
pub fn eval_density_uks(
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    chi: &Array2<f64>,
    dchi: &Array3<f64>,
) -> UksDensityGrid {
    let (nbf, npts) = chi.dim();
    debug_assert_eq!(d_a.dim(), (nbf, nbf));
    debug_assert_eq!(d_b.dim(), (nbf, nbf));
    debug_assert_eq!(dchi.dim(), (3, nbf, npts));

    let phi_a: Array2<f64> = d_a.dot(chi);
    let phi_b: Array2<f64> = d_b.dot(chi);

    // Point-outer / μ-inner restructuring — see the comment in
    // `eval_density_closed` for the determinism argument (disjoint per-point
    // output slots, same ascending-μ per-point accumulation order).
    let mut rho_a = Array1::<f64>::zeros(npts);
    let mut rho_b = Array1::<f64>::zeros(npts);
    let mut grad_a = Array2::<f64>::zeros((3, npts));
    let mut grad_b = Array2::<f64>::zeros((3, npts));
    let chi_s = chi.as_standard_layout();
    let phi_a_s = phi_a.as_standard_layout();
    let phi_b_s = phi_b.as_standard_layout();
    let dchi_s = dchi.as_standard_layout();
    let compute_point = |g: usize| -> (f64, f64, [f64; 3], [f64; 3]) {
        let mut ra = 0.0_f64;
        let mut rb = 0.0_f64;
        let mut gax = 0.0_f64;
        let mut gay = 0.0_f64;
        let mut gaz = 0.0_f64;
        let mut gbx = 0.0_f64;
        let mut gby = 0.0_f64;
        let mut gbz = 0.0_f64;
        for mu in 0..nbf {
            let c = chi_s[(mu, g)];
            let pa = phi_a_s[(mu, g)];
            let pb = phi_b_s[(mu, g)];
            ra += c * pa;
            rb += c * pb;
            gax += pa * dchi_s[(0, mu, g)];
            gay += pa * dchi_s[(1, mu, g)];
            gaz += pa * dchi_s[(2, mu, g)];
            gbx += pb * dchi_s[(0, mu, g)];
            gby += pb * dchi_s[(1, mu, g)];
            gbz += pb * dchi_s[(2, mu, g)];
        }
        (ra, rb, [2.0 * gax, 2.0 * gay, 2.0 * gaz], [2.0 * gbx, 2.0 * gby, 2.0 * gbz])
    };
    let fill = |g: usize,
                rho_a: &mut Array1<f64>,
                rho_b: &mut Array1<f64>,
                grad_a: &mut Array2<f64>,
                grad_b: &mut Array2<f64>| {
        let (ra, rb, ga, gb) = compute_point(g);
        rho_a[g] = ra;
        rho_b[g] = rb;
        for axis in 0..3 {
            grad_a[(axis, g)] = ga[axis];
            grad_b[(axis, g)] = gb[axis];
        }
    };
    if npts >= PAR_MIN_PTS {
        let results: Vec<(f64, f64, [f64; 3], [f64; 3])> =
            (0..npts).into_par_iter().map(compute_point).collect();
        for (g, (ra, rb, ga, gb)) in results.into_iter().enumerate() {
            rho_a[g] = ra;
            rho_b[g] = rb;
            for axis in 0..3 {
                grad_a[(axis, g)] = ga[axis];
                grad_b[(axis, g)] = gb[axis];
            }
        }
    } else {
        for g in 0..npts {
            fill(g, &mut rho_a, &mut rho_b, &mut grad_a, &mut grad_b);
        }
    }

    // O(npts) with ~6 flops/point (no Σ_μ inside) — left serial: rayon
    // spawn/collect overhead would dominate a loop this cheap per element (the
    // "trivially small, no compute" case the task brief calls out to skip).
    let mut sigma = Array2::<f64>::zeros((3, npts));
    for g in 0..npts {
        let (ax, ay, az) = (grad_a[(0, g)], grad_a[(1, g)], grad_a[(2, g)]);
        let (bx, by, bz) = (grad_b[(0, g)], grad_b[(1, g)], grad_b[(2, g)]);
        sigma[(0, g)] = ax * ax + ay * ay + az * az;
        sigma[(1, g)] = ax * bx + ay * by + az * bz;
        sigma[(2, g)] = bx * bx + by * by + bz * bz;
    }
    UksDensityGrid { rho_a, rho_b, grad_a, grad_b, sigma }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random fill (no rand dep) — value depends only on
    /// the flat index, so arrays are reproducible.
    fn synth(n: usize, seed: u64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let x = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(seed);
                // map to (-1, 1) with irregular mantissas
                ((x >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    /// Inputs sized ABOVE `PAR_MIN_PTS` so the rayon path is exercised.
    fn inputs(nbf: usize, npts: usize) -> (Array2<f64>, Array2<f64>, Array3<f64>) {
        assert!(npts >= PAR_MIN_PTS, "test must exercise the parallel path");
        let mut d = Array2::from_shape_vec((nbf, nbf), synth(nbf * nbf, 1)).unwrap();
        d = &d + &d.t(); // symmetric, as a density matrix is
        let chi = Array2::from_shape_vec((nbf, npts), synth(nbf * npts, 2)).unwrap();
        let dchi = Array3::from_shape_vec((3, nbf, npts), synth(3 * nbf * npts, 3)).unwrap();
        (d, chi, dchi)
    }

    /// Naive reference: the pre-parallelization μ-outer serial sweep — per
    /// point, terms are added in ascending μ, which is exactly the order the
    /// restructured point-outer loop uses. Bit-identity against this is the
    /// "identical to old serial code" guarantee.
    fn naive_closed(d: &Array2<f64>, chi: &Array2<f64>, dchi: &Array3<f64>) -> DensityGrid {
        let (nbf, npts) = chi.dim();
        let phi: Array2<f64> = d.dot(chi);
        let mut rho = Array1::<f64>::zeros(npts);
        for mu in 0..nbf {
            for g in 0..npts {
                rho[g] += chi[(mu, g)] * phi[(mu, g)];
            }
        }
        let mut grad = Array2::<f64>::zeros((3, npts));
        for axis in 0..3 {
            for mu in 0..nbf {
                for g in 0..npts {
                    grad[(axis, g)] += phi[(mu, g)] * dchi[(axis, mu, g)];
                }
            }
            for g in 0..npts {
                grad[(axis, g)] *= 2.0;
            }
        }
        let mut sigma = Array1::<f64>::zeros(npts);
        for g in 0..npts {
            sigma[g] = grad[(0, g)] * grad[(0, g)]
                + grad[(1, g)] * grad[(1, g)]
                + grad[(2, g)] * grad[(2, g)];
        }
        DensityGrid { rho, grad, sigma }
    }

    fn assert_bits_eq(a: &[f64], b: &[f64], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}[{i}]: {x:e} vs {y:e}");
        }
    }

    #[test]
    fn closed_density_bit_identical_across_thread_counts_and_vs_serial_reference() {
        let (d, chi, dchi) = inputs(14, 600);

        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| eval_density_closed(&d, &chi, &dchi))
        };
        let r1 = run(1);
        let r4 = run(4);

        assert_bits_eq(r1.rho.as_slice().unwrap(), r4.rho.as_slice().unwrap(), "rho 1v4");
        assert_bits_eq(
            r1.grad.as_slice().unwrap(),
            r4.grad.as_slice().unwrap(),
            "grad 1v4",
        );
        assert_bits_eq(
            r1.sigma.as_slice().unwrap(),
            r4.sigma.as_slice().unwrap(),
            "sigma 1v4",
        );

        // ...and identical to the old μ-outer serial algorithm (same per-point
        // ascending-μ addition order — only the loop nest changed).
        let refr = naive_closed(&d, &chi, &dchi);
        assert_bits_eq(r1.rho.as_slice().unwrap(), refr.rho.as_slice().unwrap(), "rho vs ref");
        assert_bits_eq(
            r1.grad.as_slice().unwrap(),
            refr.grad.as_slice().unwrap(),
            "grad vs ref",
        );
        assert_bits_eq(
            r1.sigma.as_slice().unwrap(),
            refr.sigma.as_slice().unwrap(),
            "sigma vs ref",
        );
    }

    #[test]
    fn uks_density_bit_identical_across_thread_counts() {
        let (d_a, chi, dchi) = inputs(14, 600);
        let mut d_b = Array2::from_shape_vec((14, 14), synth(14 * 14, 7)).unwrap();
        d_b = &d_b + &d_b.t();

        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| eval_density_uks(&d_a, &d_b, &chi, &dchi))
        };
        let r1 = run(1);
        let r4 = run(4);

        assert_bits_eq(r1.rho_a.as_slice().unwrap(), r4.rho_a.as_slice().unwrap(), "rho_a");
        assert_bits_eq(r1.rho_b.as_slice().unwrap(), r4.rho_b.as_slice().unwrap(), "rho_b");
        assert_bits_eq(
            r1.grad_a.as_slice().unwrap(),
            r4.grad_a.as_slice().unwrap(),
            "grad_a",
        );
        assert_bits_eq(
            r1.grad_b.as_slice().unwrap(),
            r4.grad_b.as_slice().unwrap(),
            "grad_b",
        );
        assert_bits_eq(
            r1.sigma.as_slice().unwrap(),
            r4.sigma.as_slice().unwrap(),
            "sigma",
        );

        // Cross-check the fused UKS path against two closed-shell evaluations
        // (same per-spin math; σ channels recomputed): α block bitwise.
        let ca = naive_closed(&d_a, &chi, &dchi);
        assert_bits_eq(r1.rho_a.as_slice().unwrap(), ca.rho.as_slice().unwrap(), "rho_a vs closed");
        assert_bits_eq(
            r1.grad_a.as_slice().unwrap(),
            ca.grad.as_slice().unwrap(),
            "grad_a vs closed",
        );
    }
}
