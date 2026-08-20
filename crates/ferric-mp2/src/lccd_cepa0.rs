//! LCCD / CEPA(0): closed-shell spatial linearized coupled-cluster doubles.
//!
//! Proved element-by-element against a full spin-orbital antisymmetrized
//! LCCD residual in `wiki/notebooks/16-lccd-cepa0.ipynb` (exact ERIs, via
//! an INDEPENDENT second solve rather than a residual check on borrowed
//! amplitudes). The spatial ring term below needed FOUR distinct pieces,
//! each traced to a specific spin sector of the `k,c` contraction — do not
//! "simplify" it, and note that the `P(ij)P(ab)` swaps are INDEPENDENT, not
//! a joint transposition.
//!
//! ```text
//! R_iajb = (ia|jb)
//!        + Σ_kl (ik|jl) T_kalb                       [hh ladder]
//!        + Σ_cd (ac|bd) T_icjd                       [pp ladder]
//!        + Ring_iajb
//!
//! Ring_iajb = t1 − t2 − t3 + t4
//!   t1_iajb = 2 Σ_kc (kc|jb) T_iakc − Σ_kc (kc|jb) T_icka
//!                                   − Σ_kc (kj|cb) T_iakc
//!   t2_iajb =   Σ_kc (ki|bc) T_jcka
//!   t3_iajb =   Σ_kc (kj|ac) T_ickb
//!   t4_iajb =   t1_jbia            (t1 with i↔j, a↔b)
//!
//! (−D) T = R,   D_iajb = ε_a + ε_b − ε_i − ε_j
//! E_LCCD = Σ_iajb (ia|jb) (2 T_iajb − T_ibja)
//! ```
//!
//! # This is NOT [`ferric_cc::linlccd`]
//!
//! That module implements Carter-Fenk's LinLCCD/LinLCCD(hh), which
//! DELIBERATELY REMOVES the ring and crossed-ring contractions (its whole
//! point: the paper diagnoses exchange-ring terms as the small-gap
//! divergence culprit) and works in spin orbitals. This module RETAINS the
//! rings and is closed-shell spatial. They are different methods with
//! different energies; do not cross-validate one against the other.
//!
//! # Why GMRES and not PCG or MINRES
//!
//! MEASURED in the notebook, not assumed: the closed-shell spatial LCCD
//! linear operator is **genuinely non-symmetric**, `|A−Aᵀ|/‖A‖ ≈ 7.6%` on
//! water/6-31G (reproducible to the digit) and 6.4–9.2% on CH₄/STO-3G
//! (the spread is CH₄'s degenerate-orbital SCF gauge freedom, not noise).
//! Its spectrum is all-real and all-negative, so it is non-normal but
//! spectrally stable.
//!
//! PCG and MINRES both REQUIRE `A = Aᵀ`. Applied to this operator they have
//! no convergence guarantee and can converge silently to a wrong answer
//! rather than failing loudly — scipy runs them here only because it does
//! not check symmetry. GMRES's convergence theory actually covers a
//! non-symmetric operator, so that is what this module uses.
//!
//! # CEPA(0) divergence must not be masked
//!
//! CEPA(0) diverges as the gap closes (notebook §4: H₂/STO-3G past
//! ~r = 3.5–4.0 Å). Critically, **damping can hide this by converging to a
//! spurious unphysical fixed point** — measured at damp=0.5, r=4.0 Å, where
//! the iteration "converged" to E = −17.98 Ha, a nonsense value with a
//! perfectly healthy-looking residual. A small relres is therefore NOT
//! sufficient evidence of a physical solution, and this module sanity-bounds
//! the converged energy rather than trusting the residual alone.

use ndarray::{Array2, Array4};

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;

use crate::mo_transform::{transform_3center_oo, transform_3center_ov, transform_3center_vv};
use crate::rimp2::{active_occ, cholesky_inverse_sqrt};

/// Configuration for linearized CCD (LCCD / CEPA(0)) via GMRES.
#[derive(Debug, Clone)]
pub struct LccdConfig {
    pub frozen_core: usize,
    /// Relative residual tolerance for the GMRES solve.
    pub rtol: f64,
    /// Maximum total GMRES iterations (across restarts).
    pub max_iter: usize,
    /// GMRES restart length. Larger keeps more Krylov history (better
    /// convergence on a non-normal operator) at O(restart) extra vectors.
    pub restart: usize,
    /// Reject a converged correlation energy whose magnitude exceeds this
    /// multiple of |E_MP2|. Guards the measured CEPA(0) failure mode where
    /// the iteration lands on a spurious fixed point with a small residual
    /// (notebook §4). Set `None` to disable (not recommended).
    pub max_corr_vs_mp2: Option<f64>,
}

impl Default for LccdConfig {
    fn default() -> Self {
        Self {
            frozen_core: 0,
            rtol: 1e-10,
            max_iter: 2000,
            restart: 60,
            // LCCD systematically overcorrelates vs CCD but only by tens of
            // percent (notebook: ~5.7% on water, ~74% on the 2-electron H2
            // edge case). A 10x blowup past MP2 is not a physical LCCD
            // energy; the measured spurious fixed point was ~100x.
            max_corr_vs_mp2: Some(10.0),
        }
    }
}

/// Result of an LCCD / CEPA(0) calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct LccdResult {
    pub e_corr: f64,
    pub e_total: f64,
    /// MP2 correlation energy from the same integrals/denominators — the
    /// scale the sanity bound is measured against, reported for context.
    pub e_mp2: f64,
    pub iterations: usize,
    pub relres: f64,
    pub converged: bool,
}

impl std::fmt::Display for LccdResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LCCD total: {:.10} Ha (corr: {:.10}, {} iters, converged: {})",
            self.e_total, self.e_corr, self.iterations, self.converged)
    }
}

/// The four MO integral blocks the residual needs, in chemist notation,
/// plus the canonical orbital energies.
struct LccdBlocks {
    /// `(ia|jb)`, `(no, nv, no, nv)`.
    j_ovov: Array4<f64>,
    /// `(ik|jl)`, `(no, no, no, no)`.
    g_oooo: Array4<f64>,
    /// `(ac|bd)`, `(nv, nv, nv, nv)`.
    g_vvvv: Array4<f64>,
    /// `(ij|ab)`, `(no, no, nv, nv)` — the generic occ-occ-vir-vir block the
    /// ring's `t2`/`t3`/`tE` pieces index into.
    g_oovv: Array4<f64>,
    e_occ: Vec<f64>,
    e_vir: Vec<f64>,
    no: usize,
    nv: usize,
}

/// Build the MO blocks from RI 3-index integrals over CANONICAL orbitals.
///
/// Canonical (not localized) on purpose: the notebook's verified spatial
/// residual assumes a diagonal Fock, so the off-diagonal Fock terms are
/// absent from the equations above. Feeding localized orbitals would
/// silently drop real terms.
fn build_blocks(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &LccdConfig,
) -> Result<LccdBlocks, FerricError> {
    use ndarray::s;

    let nocc_total = (mol.nelec() as usize) / 2;
    let no = active_occ(nocc_total, cfg.frozen_core)?;
    let first = cfg.frozen_core;
    let nbas = obs.nbasis();
    let nv = nbas - nocc_total;

    let c_occ = rhf.mos_r().slice(s![.., first..nocc_total]).to_owned();
    let c_vir = rhf.mos_r().slice(s![.., nocc_total..]).to_owned();
    let eps = rhf.eps_r();
    let e_occ: Vec<f64> = (first..nocc_total).map(|i| eps[i]).collect();
    let e_vir: Vec<f64> = (nocc_total..nbas).map(|a| eps[a]).collect();

    let v2c = ferric_integrals::threeindex::coulomb_metric_2c(op, dfbs)?;
    let v_inv_sqrt = cholesky_inverse_sqrt(&v2c)?;
    let eri3_ao = ferric_integrals::threeindex::eri3_tensor(op, obs, dfbs)?;

    // Dressed B^P_{pq} = Σ_Q V^{-1/2}_{PQ} (Q|pq); the 4-index blocks are
    // then B·B contractions, so every block is symmetric-fitting consistent.
    let dress = |m: ndarray::Array3<f64>| -> Array2<f64> {
        let (naux, d1, d2) = (m.shape()[0], m.shape()[1], m.shape()[2]);
        let flat = m.into_shape_with_order((naux, d1 * d2)).unwrap();
        v_inv_sqrt.dot(&flat)
    };
    let b_ov = dress(transform_3center_ov(&eri3_ao, &c_occ, &c_vir)); // (naux, no*nv)
    let b_oo = dress(transform_3center_oo(&eri3_ao, &c_occ)); // (naux, no*no)
    let b_vv = dress(transform_3center_vv(&eri3_ao, &c_vir)); // (naux, nv*nv)
    drop(eri3_ao);

    let four = |x: &Array2<f64>, y: &Array2<f64>, d: (usize, usize, usize, usize)| -> Array4<f64> {
        x.t()
            .dot(y)
            .into_shape_with_order(d)
            .expect("RI 4-index reshape")
    };
    let j_ovov = four(&b_ov, &b_ov, (no, nv, no, nv)); // (ia|jb)
    let g_oooo = four(&b_oo, &b_oo, (no, no, no, no)); // (ij|kl)
    let g_vvvv = four(&b_vv, &b_vv, (nv, nv, nv, nv)); // (ab|cd)
    let g_oovv = four(&b_oo, &b_vv, (no, no, nv, nv)); // (ij|ab)

    Ok(LccdBlocks {
        j_ovov,
        g_oooo,
        g_vvvv,
        g_oovv,
        e_occ,
        e_vir,
        no,
        nv,
    })
}

/// Positive MP2-style denominators `D_iajb = ε_a + ε_b − ε_i − ε_j`.
fn denominators(bl: &LccdBlocks) -> Result<Array4<f64>, FerricError> {
    let (no, nv) = (bl.no, bl.nv);
    let mut d = Array4::<f64>::zeros((no, nv, no, nv));
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    d[[i, a, j, b]] =
                        bl.e_vir[a] + bl.e_vir[b] - bl.e_occ[i] - bl.e_occ[j];
                }
            }
        }
    }
    if d.iter().any(|&x| x <= 0.0) {
        return Err(FerricError::General(
            "lccd_cepa0: non-positive denominator (not a gapped system?)".into(),
        ));
    }
    Ok(d)
}

/// The ring term `t1 − t2 − t3 + t4`, exactly as verified in notebook §2.
///
/// Each `t` piece is a distinct spin sector of the spin-orbital
/// `<kb||cj> t_ikac` contraction; the four pieces were regressed
/// element-by-element against thousands of random `(i,a,j,b)` samples until
/// a UNIQUE exact fit was confirmed by singular values. `t4` is `t1` with
/// `i↔j, a↔b` — the `P(ij)P(ab)` swaps are INDEPENDENT of each other.
fn ring(t: &Array4<f64>, bl: &LccdBlocks) -> Array4<f64> {
    let (no, nv) = (bl.no, bl.nv);
    let j = &bl.j_ovov;
    let v = &bl.g_oovv;

    let mut t1 = Array4::<f64>::zeros((no, nv, no, nv));
    for i in 0..no {
        for a in 0..nv {
            for jj in 0..no {
                for b in 0..nv {
                    let mut acc = 0.0;
                    for k in 0..no {
                        for c in 0..nv {
                            // 2 Σ (kc|jb) T_iakc − Σ (kc|jb) T_icka
                            let kcjb = j[[k, c, jj, b]];
                            acc += 2.0 * kcjb * t[[i, a, k, c]];
                            acc -= kcjb * t[[i, c, k, a]];
                            // − Σ (kj|cb) T_iakc, with (kj|cb) from the
                            // oovv block as g_oovv[k, j, c, b]
                            acc -= v[[k, jj, c, b]] * t[[i, a, k, c]];
                        }
                    }
                    t1[[i, a, jj, b]] = acc;
                }
            }
        }
    }

    let mut t2 = Array4::<f64>::zeros((no, nv, no, nv));
    let mut t3 = Array4::<f64>::zeros((no, nv, no, nv));
    for i in 0..no {
        for a in 0..nv {
            for jj in 0..no {
                for b in 0..nv {
                    let mut acc2 = 0.0;
                    let mut acc3 = 0.0;
                    for k in 0..no {
                        for c in 0..nv {
                            // t2 = Σ (ki|bc) T_jcka
                            acc2 += v[[k, i, b, c]] * t[[jj, c, k, a]];
                            // t3 = Σ (kj|ac) T_ickb
                            acc3 += v[[k, jj, a, c]] * t[[i, c, k, b]];
                        }
                    }
                    t2[[i, a, jj, b]] = acc2;
                    t3[[i, a, jj, b]] = acc3;
                }
            }
        }
    }

    let mut out = Array4::<f64>::zeros((no, nv, no, nv));
    for i in 0..no {
        for a in 0..nv {
            for jj in 0..no {
                for b in 0..nv {
                    // t4 = t1 with i<->j, a<->b
                    out[[i, a, jj, b]] = t1[[i, a, jj, b]] - t2[[i, a, jj, b]]
                        - t3[[i, a, jj, b]]
                        + t1[[jj, b, i, a]];
                }
            }
        }
    }
    out
}

/// The hh and pp ladders: `Σ_kl (ik|jl) T_kalb + Σ_cd (ac|bd) T_icjd`.
fn ladders(t: &Array4<f64>, bl: &LccdBlocks) -> Array4<f64> {
    let (no, nv) = (bl.no, bl.nv);
    let mut out = Array4::<f64>::zeros((no, nv, no, nv));
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    let mut acc = 0.0;
                    for k in 0..no {
                        for l in 0..no {
                            // (ik|jl) from the oooo block
                            acc += bl.g_oooo[[i, k, j, l]] * t[[k, a, l, b]];
                        }
                    }
                    for c in 0..nv {
                        for d in 0..nv {
                            // (ac|bd) from the vvvv block
                            acc += bl.g_vvvv[[a, c, b, d]] * t[[i, c, j, d]];
                        }
                    }
                    out[[i, a, j, b]] = acc;
                }
            }
        }
    }
    out
}

/// The LCCD linear operator `A T = (−D) T − ladders(T) − ring(T)`, so that
/// the amplitude equation is `A T = (ia|jb)`.
///
/// NON-SYMMETRIC by construction (see module docs) — this is the operator
/// whose ~7.6% asymmetry rules out PCG/MINRES.
fn apply_operator(t: &Array4<f64>, bl: &LccdBlocks, d: &Array4<f64>) -> Array4<f64> {
    let lad = ladders(t, bl);
    let rng = ring(t, bl);
    let mut out = Array4::<f64>::zeros(t.raw_dim());
    for (o, (((tt, dd), l), r)) in out
        .iter_mut()
        .zip(t.iter().zip(d.iter()).zip(lad.iter()).zip(rng.iter()))
    {
        *o = -dd * tt - l - r;
    }
    out
}

fn dot(a: &Array4<f64>, b: &Array4<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &Array4<f64>) -> f64 {
    dot(a, a).sqrt()
}

/// Restarted GMRES(m) with a Jacobi (diagonal `−D`) right preconditioner.
///
/// GMRES specifically, NOT PCG/MINRES: the operator is non-symmetric
/// (module docs), so the symmetric methods have no convergence theory here
/// and can converge silently to a wrong answer.
///
/// Returns `(x, total_iterations, relres, converged)`.
fn gmres(
    bl: &LccdBlocks,
    d: &Array4<f64>,
    rhs: &Array4<f64>,
    cfg: &LccdConfig,
) -> (Array4<f64>, usize, f64, bool) {
    let bnorm = norm(rhs);
    if bnorm == 0.0 {
        return (Array4::zeros(rhs.raw_dim()), 0, 0.0, true);
    }
    // Right preconditioner M^{-1} y = y / (−D): the diagonal of A, which is
    // exactly the MP2 denominator. Cheap and it is the dominant part of A.
    let precond = |y: &Array4<f64>| -> Array4<f64> {
        let mut z = y.clone();
        for (zz, dd) in z.iter_mut().zip(d.iter()) {
            *zz /= -dd;
        }
        z
    };

    let m = cfg.restart.max(1);
    let mut x = Array4::<f64>::zeros(rhs.raw_dim());
    let mut total_it = 0usize;
    // Always assigned from the true residual at the top of the loop before
    // any read; declared here only to outlive the loop body.
    let mut relres;

    while total_it < cfg.max_iter {
        let ax = apply_operator(&x, bl, d);
        let mut r = rhs.clone();
        r -= &ax;
        let beta = norm(&r);
        relres = beta / bnorm;
        if relres < cfg.rtol {
            return (x, total_it, relres, true);
        }

        let mut v: Vec<Array4<f64>> = Vec::with_capacity(m + 1);
        v.push(r.mapv(|z| z / beta));
        let mut h = vec![vec![0.0f64; m]; m + 1];
        let mut cs = vec![0.0f64; m];
        let mut sn = vec![0.0f64; m];
        let mut g = vec![0.0f64; m + 1];
        g[0] = beta;
        let mut k_used = 0usize;

        for k in 0..m {
            if total_it >= cfg.max_iter {
                break;
            }
            total_it += 1;
            k_used = k + 1;
            // Arnoldi step on the RIGHT-preconditioned operator A M^{-1}.
            let w0 = precond(&v[k]);
            let mut w = apply_operator(&w0, bl, d);
            for (i, vi) in v.iter().enumerate().take(k + 1) {
                h[i][k] = dot(&w, vi);
                let hik = h[i][k];
                w.zip_mut_with(vi, |ww, vv| *ww -= hik * vv);
            }
            h[k + 1][k] = norm(&w);
            let breakdown = h[k + 1][k] <= 1e-14 * beta.max(1.0);
            if !breakdown {
                v.push(w.mapv(|z| z / h[k + 1][k]));
            }

            // Apply previous Givens rotations, then a new one.
            for i in 0..k {
                let t = cs[i] * h[i][k] + sn[i] * h[i + 1][k];
                h[i + 1][k] = -sn[i] * h[i][k] + cs[i] * h[i + 1][k];
                h[i][k] = t;
            }
            let denom = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
            if denom > 0.0 {
                cs[k] = h[k][k] / denom;
                sn[k] = h[k + 1][k] / denom;
                h[k][k] = denom;
                h[k + 1][k] = 0.0;
                let t = cs[k] * g[k];
                g[k + 1] = -sn[k] * g[k];
                g[k] = t;
            }
            relres = g[k + 1].abs() / bnorm;
            if relres < cfg.rtol || breakdown {
                break;
            }
        }

        // Back-substitute for y, then x += M^{-1} V y.
        let mut y = vec![0.0f64; k_used];
        for i in (0..k_used).rev() {
            let mut s = g[i];
            for j in (i + 1)..k_used {
                s -= h[i][j] * y[j];
            }
            y[i] = if h[i][i].abs() > 0.0 { s / h[i][i] } else { 0.0 };
        }
        let mut dx = Array4::<f64>::zeros(rhs.raw_dim());
        for (i, yi) in y.iter().enumerate().take(v.len()) {
            dx.zip_mut_with(&v[i], |a, b| *a += yi * b);
        }
        x += &precond(&dx);
    }

    // Final true residual — the Givens estimate can drift from it.
    let ax = apply_operator(&x, bl, d);
    let mut r = rhs.clone();
    r -= &ax;
    relres = norm(&r) / bnorm;
    let converged = relres < cfg.rtol;
    (x, total_it, relres, converged)
}

/// `E = Σ_iajb (ia|jb) (2 T_iajb − T_ibja)` — the standard closed-shell
/// singlet energy expression, cross-checked against the spin-orbital
/// energy in notebook §2.
fn lccd_energy(t: &Array4<f64>, bl: &LccdBlocks) -> f64 {
    let (no, nv) = (bl.no, bl.nv);
    let mut e = 0.0;
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    e += bl.j_ovov[[i, a, j, b]]
                        * (2.0 * t[[i, a, j, b]] - t[[i, b, j, a]]);
                }
            }
        }
    }
    e
}

/// LCCD / CEPA(0) correlation energy.
pub fn lccd(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &LccdConfig,
) -> Result<LccdResult, FerricError> {
    let bl = build_blocks(mol, obs, dfbs, op, rhf, cfg)?;
    let d = denominators(&bl)?;

    // MP2 amplitude/energy from the SAME blocks — both the GMRES start and
    // the scale the sanity bound is measured against.
    let t_mp2 = {
        let mut t = bl.j_ovov.clone();
        for (tt, dd) in t.iter_mut().zip(d.iter()) {
            *tt /= -dd;
        }
        t
    };
    let e_mp2 = lccd_energy(&t_mp2, &bl);

    let (t, iterations, relres, converged) = gmres(&bl, &d, &bl.j_ovov, cfg);
    if !converged {
        return Err(FerricError::General(format!(
            "lccd_cepa0: GMRES failed to converge (relres {relres:.2e} after {iterations} \
             iterations); CEPA(0) is expected to diverge as the gap closes"
        )));
    }
    let e_corr = lccd_energy(&t, &bl);

    // A SMALL RESIDUAL IS NOT PROOF OF A PHYSICAL SOLUTION (notebook §4):
    // the measured CEPA(0) failure mode converges cleanly onto a spurious
    // fixed point with a nonsense energy. Bound the energy explicitly.
    if let Some(factor) = cfg.max_corr_vs_mp2 {
        if e_mp2.abs() > 0.0 && e_corr.abs() > factor * e_mp2.abs() {
            return Err(FerricError::General(format!(
                "lccd_cepa0: converged (relres {relres:.2e}) onto an unphysical energy \
                 E_corr = {e_corr:.6}, |E_corr| > {factor}x |E_MP2| = {:.6}. CEPA(0) \
                 diverges as the gap closes and can converge to a SPURIOUS fixed point \
                 with a healthy-looking residual — this is that failure mode, not a \
                 solver bug",
                e_mp2.abs()
            )));
        }
    }
    // A positive correlation energy is unphysical for LCCD on a gapped
    // closed-shell reference regardless of magnitude.
    if e_corr > 0.0 {
        return Err(FerricError::General(format!(
            "lccd_cepa0: converged onto a POSITIVE correlation energy {e_corr:.6} \
             (relres {relres:.2e}) — unphysical for a gapped closed-shell reference"
        )));
    }

    Ok(LccdResult {
        e_corr,
        e_total: rhf.energy + e_corr,
        e_mp2,
        iterations,
        relres,
        converged,
    })
}

/// Dense `(no·nv)²` matrix of the LCCD linear operator, built by probing
/// with unit vectors. O(N⁴) probes × one operator apply each — SPIKE-SCALE
/// ONLY, for the symmetry/spectrum diagnostics that justify the GMRES
/// choice. Never call this on a production-sized system.
pub fn dense_operator(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    rhf: &ScfResult,
    cfg: &LccdConfig,
) -> Result<Array2<f64>, FerricError> {
    let bl = build_blocks(mol, obs, dfbs, op, rhf, cfg)?;
    let d = denominators(&bl)?;
    let dim = bl.no * bl.nv * bl.no * bl.nv;
    let mut a = Array2::<f64>::zeros((dim, dim));
    for col in 0..dim {
        let mut e = Array4::<f64>::zeros((bl.no, bl.nv, bl.no, bl.nv));
        e.as_slice_mut().expect("contiguous probe vector")[col] = 1.0;
        let col_vals = apply_operator(&e, &bl, &d);
        let slice = col_vals.as_slice().expect("contiguous operator image");
        for (row, &v) in slice.iter().enumerate() {
            a[(row, col)] = v;
        }
    }
    Ok(a)
}
