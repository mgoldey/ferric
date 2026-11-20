//! Hylleraas mask certificate of an amplitude-threshold local MP2 result:
//! a rigorous two-sided interval on the error the amplitude mask makes, from
//! quantities available after the masked solve (book chapter 13,
//! §lmp2-mask-certificate; sympy proof `scripts/lmp2_mask_certificate_sympy.py`).
//!
//! # What is bounded
//!
//! The closed-shell Hylleraas functional `E[t] = 2⟨Mt, g⟩ + ⟨Mt, At⟩`
//! (`M = 2 − S`, `S` the a↔b swap within a pair, `A` the Fock
//! superoperator, `g = (ia|jb)`) is minimized over the kept set `P` by the
//! masked solve. With the full residual `r = g + A t_P`,
//!
//! `E_P − E* = ⟨M r, A⁻¹ r⟩ ≥ 0`
//!
//! exactly, where `E*` is the minimum over the WHOLE amplitude space with the
//! SAME `g` and `A`. The certificate bounds this one number. It does not see
//! any approximation inside `g`: on this (in-core, global-metric) path `g` is
//! `B̃ᵀB̃`, the canonical RI integrals, so `E*` is the canonical RI-MP2
//! energy; the integral-direct path's maps and the pair-local fit change `g`
//! and are refused here (`crate::lmp2_amplitude::amplitude_lmp2_with_virtuals`).
//!
//! # The cheap route
//!
//! The residual on the dropped set `Q` is split into a NEAR set
//! `N = {q ∈ Q : |(A t_P)_q| > τ or |(A t_P)_{Sq}| > τ}` (swap-closed) and
//! the FAR rest. `g` is evaluated on `N` only; on the far set only its norm
//! enters, from the RI Gram identity
//! `‖g_far‖² = ‖B̃B̃ᵀ‖_F² − ‖g_P‖² − ‖g_N‖²` (naux × naux). Then
//!
//! - upper: `U = (√U_near + √U_far)²` with `U_near` the one-step
//!   Gauss–Radau bound on `⟨M v, A⁻¹ v⟩`, `v = r` on `P ∪ N` (the PCG
//!   residual on `P` is included, so `U` also covers the solver error), node
//!   `λ_min = 2(ε_LUMO − ε_HOMO)` from the eigenvalues of the localized Fock
//!   blocks (rotation-invariant; the local Fock diagonal is NOT a bound), and
//!   `U_far = (√3‖g_far‖ + ‖h_far‖_M)²/λ_min`;
//! - lower: `L = x²/y`, the energy decrease of an exact line search from
//!   `t_P` along `d = D⁻¹ r_N` (`x = ⟨M d, r_N⟩`, `y = ⟨M d, A d⟩`, `D` the
//!   local Fock diagonal), which is a Hylleraas value and so rigorous;
//! - estimate: `x` itself (not a bound).
//!
//! Cost: two streaming passes over the `no² · nv²` amplitude space (no
//! amplitude-sized object beyond the kept and near sets is stored), the
//! integrals on `N`, and one naux × naux Gram.

use std::time::Instant;

use ndarray::{Array2, Axis};
use rayon::prelude::*;

use ferric_core::error::FerricError;

use crate::ragged::Ragged;

/// Relative allowance added to the Gram far-set norm for the rounding of the
/// three sums it is the difference of (an upper bound must not go below the
/// exact value through cancellation).
const GRAM_ROUNDING_REL: f64 = 1e-13;

/// The mask certificate of one local MP2 run. Energies in Hartree; the error
/// is `E_P − E* ≥ 0`, the local correlation energy minus the full-space
/// Hylleraas minimum with the same integrals.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskCertificate {
    /// Point estimate of the mask error, `x = ⟨M d, r_N⟩` (NOT a bound).
    pub error_estimate: f64,
    /// Rigorous lower bound on the error (line search, `x²/y`).
    pub lower_bound: f64,
    /// Rigorous upper bound on the error (`(√U_near + √U_far)²`); it also
    /// covers the PCG residual on the kept set.
    pub upper_bound: f64,
    /// Certified interval on `E*`: `[e_corr − upper_bound, e_corr − lower_bound]`.
    pub e_corr_interval: [f64; 2],
    /// `e_corr − error_estimate`: the optionally corrected correlation energy.
    pub e_corr_corrected: f64,
    /// Gauss–Radau bound on the kept + near residual's contribution.
    pub near_upper: f64,
    /// Bound on the far residual's contribution.
    pub far_upper: f64,
    /// `‖g_far‖²` from the RI Gram identity (rounding allowance included).
    pub g_far_sq: f64,
    /// `‖h_far‖²_M`, `h = A t_P` on the far set.
    pub h_far_m_sq: f64,
    /// The Radau node `2(ε_LUMO − ε_HOMO)` (Ha).
    pub lambda_min: f64,
    /// Near-set threshold τ (= ε).
    pub tau: f64,
    /// Kept amplitudes `|P|`.
    pub n_kept: u64,
    /// Near-set size `|N|`.
    pub n_near: u64,
    /// Far-set size `|Q| − |N|`.
    pub n_far: u64,
    /// Wall seconds of the certificate (both passes, the near integrals and
    /// the Gram).
    pub t_s: f64,
}

impl MaskCertificate {
    /// Serialized entries, shared by the CLI run-log JSON and the Python
    /// `result.local["mask_certificate"]` dict.
    pub fn fields(&self) -> Vec<(&'static str, crate::lmp2_certificate::CertValue)> {
        use crate::lmp2_certificate::CertValue as V;
        let u = |n: u64| V::Usize(usize::try_from(n).unwrap_or(usize::MAX));
        vec![
            ("bounds", V::Str(SCOPE.to_string())),
            ("error_estimate", V::F64(self.error_estimate)),
            ("lower_bound", V::F64(self.lower_bound)),
            ("upper_bound", V::F64(self.upper_bound)),
            ("e_corr_interval_lo", V::F64(self.e_corr_interval[0])),
            ("e_corr_interval_hi", V::F64(self.e_corr_interval[1])),
            ("e_corr_corrected", V::F64(self.e_corr_corrected)),
            ("near_upper", V::F64(self.near_upper)),
            ("far_upper", V::F64(self.far_upper)),
            ("g_far_sq", V::F64(self.g_far_sq)),
            ("h_far_m_sq", V::F64(self.h_far_m_sq)),
            ("lambda_min", V::F64(self.lambda_min)),
            ("tau", V::F64(self.tau)),
            ("n_kept", u(self.n_kept)),
            ("n_near", u(self.n_near)),
            ("n_far", u(self.n_far)),
            ("t_s", V::F64(self.t_s)),
        ]
    }
}

/// What the certificate bounds, in one sentence, for every printout.
pub const SCOPE: &str = "the amplitude-mask error E_P - E* for the integrals actually solved \
     (in-core global-metric RI: E* is the canonical RI-MP2 energy); it does not see the \
     RI fit error against exact integrals";

/// The inputs of [`mask_certificate`].
pub struct MaskCertificateInputs<'a> {
    /// The ragged system of the masked solve (kept set `P`, `g` on `P`).
    pub rg: &'a Ragged,
    /// The masked amplitudes `t_P`, one block per pair of `rg`.
    pub t: &'a [Array2<f64>],
    /// Localized active-occupied Fock block (no × no).
    pub f_oo: &'a Array2<f64>,
    /// Localized virtual Fock block (nv × nv).
    pub f_vv: &'a Array2<f64>,
    /// Whitened RI factor `B̃` (naux × no·nv, column `i·nv + a`), with
    /// `g_iajb = B̃_iaᵀ B̃_jb` everywhere (the integrals `rg` holds on `P`).
    pub btilde: &'a Array2<f64>,
    /// Near-set threshold τ.
    pub tau: f64,
    /// `2(ε_LUMO − ε_HOMO)` ([`crate::lmp2_amplitude::operator_lambda_min_lower_bound`]).
    pub lambda_min: f64,
    /// The reported (Hylleraas) correlation energy of the masked solve.
    pub e_corr: f64,
}

/// One retained entry of the near vector `v` (r on `P ∪ N`) of a pair: the
/// global virtual indices, `r`, and the line-search direction `d = r/D`
/// (0 on `P`).
#[derive(Clone, Copy)]
struct Entry {
    a: u32,
    b: u32,
    v: f64,
    d: f64,
}

/// Pass-1 sums of one output pair.
#[derive(Default, Clone, Copy)]
struct Sums {
    g_p2: f64,
    g_n2: f64,
    h_far_m2: f64,
    x_n: f64,
    mu0: f64,
    n_kept: u64,
    n_near: u64,
    n_far: u64,
}

impl Sums {
    fn add(mut self, o: Sums) -> Sums {
        self.g_p2 += o.g_p2;
        self.g_n2 += o.g_n2;
        self.h_far_m2 += o.h_far_m2;
        self.x_n += o.x_n;
        self.mu0 += o.mu0;
        self.n_kept += o.n_kept;
        self.n_near += o.n_near;
        self.n_far += o.n_far;
        self
    }
}

/// Element class of the dense per-pair buffer.
const FAR: u8 = 0;
const KEPT: u8 = 1;
const NEAR: u8 = 2;

/// `pair_index[i·no + j]` = position of block (i, j) in `rg.pairs`, or
/// `usize::MAX`. Errors on a duplicated block.
fn pair_index(rg: &Ragged, no: usize) -> Result<Vec<usize>, FerricError> {
    let mut idx = vec![usize::MAX; no * no];
    for (p, pb) in rg.pairs.iter().enumerate() {
        let slot = &mut idx[pb.i * no + pb.j];
        if *slot != usize::MAX {
            return Err(FerricError::General(format!(
                "mask_certificate: pair ({}, {}) carries two blocks",
                pb.i, pb.j
            )));
        }
        *slot = p;
    }
    Ok(idx)
}

/// `h_ij = (A t_P)_ij` on the whole nv × nv square of output pair (i, j),
/// accumulated into `h` (zeroed by the caller): the virtual Fock products of
/// block (i, j) and the occupied couplings from blocks (k, j) and (i, k).
/// Same sign convention as [`crate::ragged::matvec_indexed`].
fn full_fock_product(
    inp: &MaskCertificateInputs<'_>,
    idx: &[usize],
    no: usize,
    i: usize,
    j: usize,
    h: &mut Array2<f64>,
) {
    let (rg, t, f_oo, f_vv) = (inp.rg, inp.t, inp.f_oo, inp.f_vv);
    let p = idx[i * no + j];
    if p != usize::MAX {
        let pb = &rg.pairs[p];
        let tp = &t[p];
        // (F T)[c, b] for every c, and (T F)[a, c] for every c
        let x = f_vv.select(Axis(1), &pb.da).dot(tp);
        for (cb, &b) in pb.db.iter().enumerate() {
            for c in 0..h.nrows() {
                h[(c, b)] += x[(c, cb)];
            }
        }
        let y = tp.dot(&f_vv.select(Axis(0), &pb.db));
        for (ra, &a) in pb.da.iter().enumerate() {
            for c in 0..h.ncols() {
                h[(a, c)] += y[(ra, c)];
            }
        }
    }
    for k in 0..no {
        for (q, f) in [
            (idx[k * no + j], f_oo[(i, k)]),
            (idx[i * no + k], f_oo[(k, j)]),
        ] {
            if q == usize::MAX || f == 0.0 {
                continue;
            }
            let qb = &rg.pairs[q];
            let tq = &t[q];
            for (r, &a) in qb.da.iter().enumerate() {
                for (c, &b) in qb.db.iter().enumerate() {
                    h[(a, b)] -= f * tq[(r, c)];
                }
            }
        }
    }
}

/// Pass 1 over output pair (i, j): classify every element (kept / near /
/// far), evaluate `g` on the near set, and return the pair's near-vector
/// entries with its sums.
fn pass_one(
    inp: &MaskCertificateInputs<'_>,
    idx: &[usize],
    no: usize,
    nv: usize,
    i: usize,
    j: usize,
) -> (Vec<Entry>, Sums) {
    let rg = inp.rg;
    let mut h = Array2::<f64>::zeros((nv, nv));
    full_fock_product(inp, idx, no, i, j, &mut h);
    let mut cls = vec![FAR; nv * nv];
    let mut rbuf = Array2::<f64>::zeros((nv, nv));
    let mut s = Sums::default();
    let p = idx[i * no + j];
    if p != usize::MAX {
        let pb = &rg.pairs[p];
        let nb = pb.db.len();
        for (r, &a) in pb.da.iter().enumerate() {
            for (c, &b) in pb.db.iter().enumerate() {
                if pb.pat[r * nb + c] {
                    let g = pb.j_blk[(r, c)];
                    cls[a * nv + b] = KEPT;
                    rbuf[(a, b)] = g + h[(a, b)];
                    s.g_p2 += g * g;
                    s.n_kept += 1;
                }
            }
        }
    }
    // near set, swap-closed: |h_ab| > τ or |h_ba| > τ
    let mut near: Vec<(usize, usize)> = Vec::new();
    for a in 0..nv {
        for b in 0..nv {
            if cls[a * nv + b] != KEPT && (h[(a, b)].abs() > inp.tau || h[(b, a)].abs() > inp.tau) {
                near.push((a, b));
            }
        }
    }
    for &(a, b) in &near {
        cls[a * nv + b] = NEAR;
    }
    if !near.is_empty() {
        let mut rows: Vec<usize> = near.iter().map(|x| x.0).collect();
        rows.sort_unstable();
        rows.dedup();
        let mut cols: Vec<usize> = near.iter().map(|x| x.1).collect();
        cols.sort_unstable();
        cols.dedup();
        let ci: Vec<usize> = rows.iter().map(|&a| i * nv + a).collect();
        let cj: Vec<usize> = cols.iter().map(|&b| j * nv + b).collect();
        let gsub = inp
            .btilde
            .select(Axis(1), &ci)
            .t()
            .dot(&inp.btilde.select(Axis(1), &cj));
        let mut pos_r = vec![usize::MAX; nv];
        for (k, &a) in rows.iter().enumerate() {
            pos_r[a] = k;
        }
        let mut pos_c = vec![usize::MAX; nv];
        for (k, &b) in cols.iter().enumerate() {
            pos_c[b] = k;
        }
        for &(a, b) in &near {
            let g = gsub[(pos_r[a], pos_c[b])];
            rbuf[(a, b)] = g + h[(a, b)];
            s.g_n2 += g * g;
        }
        s.n_near = near.len() as u64;
    }
    let fo = |k: usize| inp.f_oo[(k, k)];
    let fv = |a: usize| inp.f_vv[(a, a)];
    let mut entries = Vec::with_capacity(s.n_kept as usize + near.len());
    for a in 0..nv {
        for b in 0..nv {
            match cls[a * nv + b] {
                FAR => {
                    let hab = h[(a, b)];
                    s.h_far_m2 += (2.0 * hab - h[(b, a)]) * hab;
                    s.n_far += 1;
                }
                c => {
                    let v = rbuf[(a, b)];
                    let mv = 2.0 * v - rbuf[(b, a)];
                    s.mu0 += mv * v;
                    let d = if c == NEAR {
                        let den = fv(a) + fv(b) - fo(i) - fo(j);
                        s.x_n += mv * v / den;
                        v / den
                    } else {
                        0.0
                    };
                    entries.push(Entry {
                        a: a as u32,
                        b: b as u32,
                        v,
                        d,
                    });
                }
            }
        }
    }
    (entries, s)
}

/// Pass-2 sums: `⟨Mv, Av⟩`, `‖Av‖²_M`, `⟨Md, Ad⟩`.
#[derive(Default, Clone, Copy)]
struct Sums2 {
    alpha: f64,
    nu: f64,
    y: f64,
}

/// Pass 2 over output pair (i, j): `A v` and `A d` on the whole square of
/// the pair from the sparse entries of every coupled pair.
fn pass_two(
    f_oo: &Array2<f64>,
    f_vv: &Array2<f64>,
    ent: &[Vec<Entry>],
    no: usize,
    nv: usize,
    i: usize,
    j: usize,
) -> Sums2 {
    let own = &ent[i * no + j];
    let coupled = (0..no).any(|k| !ent[k * no + j].is_empty() || !ent[i * no + k].is_empty());
    if own.is_empty() && !coupled {
        return Sums2::default();
    }
    let mut w1 = Array2::<f64>::zeros((nv, nv));
    let mut w2 = Array2::<f64>::zeros((nv, nv));
    for e in own {
        let (a, b) = (e.a as usize, e.b as usize);
        for c in 0..nv {
            w1[(c, b)] += e.v * f_vv[(a, c)];
            w1[(a, c)] += e.v * f_vv[(b, c)];
        }
        if e.d != 0.0 {
            for c in 0..nv {
                w2[(c, b)] += e.d * f_vv[(a, c)];
                w2[(a, c)] += e.d * f_vv[(b, c)];
            }
        }
    }
    for k in 0..no {
        for (src, f) in [
            (&ent[k * no + j], f_oo[(i, k)]),
            (&ent[i * no + k], f_oo[(k, j)]),
        ] {
            if f == 0.0 {
                continue;
            }
            for e in src {
                let (a, b) = (e.a as usize, e.b as usize);
                w1[(a, b)] -= f * e.v;
                w2[(a, b)] -= f * e.d;
            }
        }
    }
    let mut s = Sums2::default();
    if !own.is_empty() {
        let mut vd = Array2::<f64>::zeros((nv, nv));
        let mut dd = Array2::<f64>::zeros((nv, nv));
        for e in own {
            vd[(e.a as usize, e.b as usize)] = e.v;
            dd[(e.a as usize, e.b as usize)] = e.d;
        }
        for e in own {
            let (a, b) = (e.a as usize, e.b as usize);
            s.alpha += (2.0 * e.v - vd[(b, a)]) * w1[(a, b)];
            s.y += (2.0 * e.d - dd[(b, a)]) * w2[(a, b)];
        }
    }
    for a in 0..nv {
        for b in 0..nv {
            s.nu += (2.0 * w1[(a, b)] - w1[(b, a)]) * w1[(a, b)];
        }
    }
    s
}

/// One-step Gauss–Radau upper bound on `⟨M v, A⁻¹ v⟩` in the M inner
/// product, node `a ≤ λ_min(A)`, from `μ₀ = ⟨Mv, v⟩`, `α' = ⟨Mv, Av⟩` and
/// `ν = ‖Av‖²_M`; never above the zero-step bound `μ₀/a`.
pub fn radau_one_step(mu0: f64, alpha_raw: f64, nu: f64, a: f64) -> f64 {
    if mu0 <= 0.0 {
        return 0.0;
    }
    let crude = mu0 / a;
    let alpha = alpha_raw / mu0;
    let beta2 = (nu / mu0 - alpha * alpha).max(0.0);
    if alpha - a <= 1e-14 * alpha.abs() {
        return crude;
    }
    let omega = a + beta2 / (alpha - a);
    let det = alpha * omega - beta2;
    if det <= 0.0 {
        return crude;
    }
    (mu0 * omega / det).min(crude)
}

/// The mask certificate of a masked solve ([`MaskCertificate`]).
pub fn mask_certificate(inp: &MaskCertificateInputs<'_>) -> Result<MaskCertificate, FerricError> {
    let t0 = Instant::now();
    let (no, nv) = (inp.f_oo.nrows(), inp.f_vv.nrows());
    if inp.btilde.ncols() != no * nv {
        return Err(FerricError::General(format!(
            "mask_certificate: B has {} columns, expected no·nv = {}",
            inp.btilde.ncols(),
            no * nv
        )));
    }
    if !(inp.lambda_min.is_finite() && inp.lambda_min > 0.0) {
        return Err(FerricError::General(format!(
            "mask_certificate: the Radau node 2(e_LUMO - e_HOMO) = {} is not positive; the \
             bounds are undefined",
            inp.lambda_min
        )));
    }
    let idx = pair_index(inp.rg, no)?;
    let pass1: Vec<(Vec<Entry>, Sums)> = (0..no * no)
        .into_par_iter()
        .map(|ij| pass_one(inp, &idx, no, nv, ij / no, ij % no))
        .collect();
    let s = pass1
        .iter()
        .fold(Sums::default(), |acc, (_, s)| acc.add(*s));
    let ent: Vec<Vec<Entry>> = pass1.into_iter().map(|(e, _)| e).collect();
    let s2 = (0..no * no)
        .into_par_iter()
        .map(|ij| pass_two(inp.f_oo, inp.f_vv, &ent, no, nv, ij / no, ij % no))
        .reduce(Sums2::default, |a, b| Sums2 {
            alpha: a.alpha + b.alpha,
            nu: a.nu + b.nu,
            y: a.y + b.y,
        });
    let lmin = inp.lambda_min;
    let near_upper = radau_one_step(s.mu0, s2.alpha, s2.nu, lmin);
    let (g_far_sq, far_upper) = if s.n_far == 0 {
        (0.0, 0.0)
    } else {
        let gram = inp.btilde.dot(&inp.btilde.t());
        let g_tot: f64 = gram.iter().map(|x| x * x).sum();
        let g_far_sq = (g_tot - s.g_p2 - s.g_n2).max(0.0) + GRAM_ROUNDING_REL * g_tot;
        let h_far = s.h_far_m2.max(0.0).sqrt();
        let far = (3f64.sqrt() * g_far_sq.sqrt() + h_far).powi(2) / lmin;
        (g_far_sq, far)
    };
    let upper = (near_upper.sqrt() + far_upper.sqrt()).powi(2);
    let lower = if s.x_n > 0.0 && s2.y > 0.0 {
        s.x_n * s.x_n / s2.y
    } else {
        0.0
    };
    Ok(MaskCertificate {
        error_estimate: s.x_n,
        lower_bound: lower,
        upper_bound: upper,
        e_corr_interval: [inp.e_corr - upper, inp.e_corr - lower],
        e_corr_corrected: inp.e_corr - s.x_n,
        near_upper,
        far_upper,
        g_far_sq,
        h_far_m_sq: s.h_far_m2,
        lambda_min: lmin,
        tau: inp.tau,
        n_kept: s.n_kept,
        n_near: s.n_near,
        n_far: s.n_far,
        t_s: t0.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::radau_one_step;

    /// On a diagonal operator the one-step Radau bound is exact for a
    /// two-level vector with its node on the lower eigenvalue, and above the
    /// exact value (never below) for any node under λ_min.
    #[test]
    fn radau_one_step_is_exact_on_two_levels_and_an_upper_bound_below() {
        // A = diag(2, 5), v = (1, 1), M = 1 (plain inner product)
        let (l1, l2) = (2.0f64, 5.0f64);
        let exact = 1.0 / l1 + 1.0 / l2;
        let mu0 = 2.0;
        let alpha = l1 + l2;
        let nu = l1 * l1 + l2 * l2;
        let at_node = radau_one_step(mu0, alpha, nu, l1);
        assert!((at_node - exact).abs() < 1e-14, "{at_node} vs {exact}");
        for a in [0.5, 1.0, 1.9] {
            let u = radau_one_step(mu0, alpha, nu, a);
            assert!(u >= exact && u <= mu0 / a, "node {a}: {u} vs {exact}");
        }
        assert_eq!(radau_one_step(0.0, 0.0, 0.0, 1.0), 0.0);
    }
}
