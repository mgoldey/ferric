//! String-driven direct sigma build `σ = H c` (Knowles–Handy / Olsen).
//!
//! The coefficient vector is the matrix `C[Iα, Iβ]` (row-major, `d = ia *
//! n_beta + ib`). The Hamiltonian is written with the spin-adapted one-body
//! operators `E^σ_pq = a†_{pσ} a_{qσ}`:
//!
//! ```text
//!   H = Σ_pq g_pq (E^α_pq + E^β_pq)
//!     + ½ Σ_pqrs (pq|rs) (E^α_pq E^α_rs + E^β_pq E^β_rs)
//!     +   Σ_pqrs (pq|rs)  E^α_pq E^β_rs
//!     + E_core,
//!   g_pq = h_pq − ½ Σ_r (pr|rq),
//! ```
//!
//! which splits σ into three string contractions:
//!
//! * σ1 (β-β): `σ(Iα, Iβ) += Σ_Jβ ⟨Iβ|H_β|Jβ⟩ C(Iα, Jβ)`,
//! * σ2 (α-α): `σ(Iα, Iβ) += Σ_Jα ⟨Iα|H_α|Jα⟩ C(Jα, Iβ)`,
//! * σ3 (α-β): `σ(Iα, Iβ) += Σ_pqrs (pq|rs) Σ_{Jα,Jβ} ⟨Iα|E_pq|Jα⟩⟨Iβ|E_rs|Jβ⟩ C(Jα, Jβ)`,
//!
//! where `H_σ` is the one-spin part of the first two lines. All three are built
//! from per-string single-excitation lists `E_pq|S⟩ = ε|T⟩`, with
//! `ε = pb(S, q) · pb(S∖q, p)` (`pb` = [`crate::strings::phase_below`]): the
//! same single-channel phase as `general_element`'s `ε₁` (see `theory.md`). A
//! β operator's extra `(−1)^{N_α}` in the α-before-β order appears in pairs
//! and cancels, exactly as in the Slater–Condon rules, so `⟨Iα,Iβ|E^α E^β|Jα,Jβ⟩`
//! factorizes into the two single-channel phases.
//!
//! Cost per call: σ1 + σ2 are `O(N_det · n_exc)` plus the same-spin double
//! fan-out; σ3 is `O(N_det · n_exc_α · n_exc_β)`, with `n_exc = k(n−k) + k`
//! single replacements per string. Never `O(N_det²)`.
//!
//! Exactness anchor: `tests/direct_sigma_vs_naive.rs` (against
//! [`crate::hamiltonian::sigma_naive`] on random and real integrals) and
//! `tests/slater_condon_brute_force.rs` (against an independent bit-loop
//! second-quantized H).

use crate::hamiltonian::DeterminantSpace;
use crate::integrals::ActiveSpaceIntegrals;
use crate::strings::phase_below;
use rayon::prelude::*;
use std::collections::HashMap;

/// One single replacement `E_pq |S⟩ = sign |T⟩` (including `p == q`).
#[derive(Debug, Clone, Copy)]
struct Excitation {
    /// Flat pair index `p * n + q`.
    pq: u32,
    /// Index of `T` in the same string list.
    target: u32,
    /// `±1`.
    sign: f64,
}

/// For one fixed pair `rs`: every β string `Jβ` with `E_rs|Jβ⟩ = sign|Iβ⟩`.
#[derive(Debug, Clone, Copy)]
struct PairEntry {
    source: u32,
    target: u32,
    sign: f64,
}

/// Single-replacement lists for every string of one spin channel.
fn excitation_lists(strings: &[u64], n: usize) -> Vec<Vec<Excitation>> {
    let index: HashMap<u64, u32> = strings
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i as u32))
        .collect();
    strings
        .iter()
        .map(|&s| {
            let mut out = Vec::new();
            for q in crate::strings::occupied_orbitals(s) {
                let s_minus = s & !(1u64 << q);
                let pb_q = phase_below(s, q);
                for p in 0..n {
                    if p != q && s & (1u64 << p) != 0 {
                        continue;
                    }
                    let t = s_minus | (1u64 << p);
                    out.push(Excitation {
                        pq: (p * n + q) as u32,
                        target: index[&t],
                        sign: pb_q * phase_below(s_minus, p),
                    });
                }
            }
            out
        })
        .collect()
}

/// Direct sigma operator for one active space and one set of integrals.
///
/// Build once per CAS-CI solve ([`DirectSigma::new`]) and apply per Davidson
/// iteration ([`DirectSigma::apply`]). Holds the integrals by reference plus
/// the string excitation lists (`O(n_str · n_exc)` entries per channel).
#[derive(Debug)]
pub struct DirectSigma<'a> {
    ints: &'a ActiveSpaceIntegrals,
    n_alpha_str: usize,
    n_beta_str: usize,
    alpha: Vec<Vec<Excitation>>,
    beta: Vec<Vec<Excitation>>,
    /// β replacements grouped by pair `rs` (length `n²`), for σ3.
    beta_by_pair: Vec<Vec<PairEntry>>,
    /// `g_pq = h_pq − ½ Σ_r (pr|rq)`, flat `p * n + q`.
    g: Vec<f64>,
}

impl<'a> DirectSigma<'a> {
    /// Build the excitation lists for `space` over `ints.n_active` orbitals.
    pub fn new(ints: &'a ActiveSpaceIntegrals, space: &DeterminantSpace) -> Self {
        let n = ints.n_active;
        let alpha = excitation_lists(&space.alpha_strings, n);
        let beta = excitation_lists(&space.beta_strings, n);
        let mut beta_by_pair = vec![Vec::new(); n * n];
        for (jb, list) in beta.iter().enumerate() {
            for e in list {
                beta_by_pair[e.pq as usize].push(PairEntry {
                    source: jb as u32,
                    target: e.target,
                    sign: e.sign,
                });
            }
        }
        let mut g = vec![0.0f64; n * n];
        for p in 0..n {
            for q in 0..n {
                let mut v = ints.h[(p, q)];
                for r in 0..n {
                    v -= 0.5 * ints.g(p, r, r, q);
                }
                g[p * n + q] = v;
            }
        }
        DirectSigma {
            ints,
            n_alpha_str: space.alpha_strings.len(),
            n_beta_str: space.beta_strings.len(),
            alpha,
            beta,
            beta_by_pair,
            g,
        }
    }

    /// Number of `f64`-sized elements the excitation lists occupy (each
    /// entry is 16 bytes = 2 `f64`), for memory planning.
    pub fn list_elems(&self) -> usize {
        let a: usize = self.alpha.iter().map(Vec::len).sum();
        let b: usize = self.beta.iter().map(Vec::len).sum();
        2 * (a + 2 * b)
    }

    /// `σ = H c` including `E_core` on the diagonal; `c` indexed
    /// `d = ia * n_beta + ib`.
    pub fn apply(&self, c: &[f64]) -> Vec<f64> {
        let na = self.n_alpha_str;
        let nb = self.n_beta_str;
        assert_eq!(c.len(), na * nb, "sigma: c has the wrong length");
        if c.is_empty() {
            return Vec::new();
        }
        let e_core = self.ints.e_core;
        let mut sig: Vec<f64> = c.iter().map(|&x| e_core * x).collect();

        // σ2 (α-α): rows of C are β-vectors indexed by Iα.
        self.same_spin(&self.alpha, c, &mut sig, nb);

        // σ1 (β-β): the same contraction on Cᵀ.
        let ct = transpose(c, na, nb);
        let mut st = vec![0.0f64; na * nb];
        self.same_spin(&self.beta, &ct, &mut st, na);
        drop(ct);
        sig.par_chunks_mut(nb).enumerate().for_each(|(ia, row)| {
            for (ib, s) in row.iter_mut().enumerate() {
                *s += st[ib * na + ia];
            }
        });
        drop(st);

        // σ3 (α-β).
        self.alpha_beta(c, &mut sig);
        sig
    }

    /// `out[I, :] += Σ_J ⟨J|H_σ|I⟩ x[J, :]` for one spin channel whose
    /// strings index the rows (`ncol` columns). `⟨J|H_σ|I⟩` is assembled per
    /// row `I` by applying `E_kl` then `E_ij` to `I`; real symmetric integrals
    /// make it equal to `⟨I|H_σ|J⟩`.
    fn same_spin(&self, lists: &[Vec<Excitation>], x: &[f64], out: &mut [f64], ncol: usize) {
        let nrow = lists.len();
        let n2 = self.ints.n_active * self.ints.n_active;
        let eri = &self.ints.eri;
        let g = &self.g;
        out.par_chunks_mut(ncol).enumerate().for_each_init(
            || (vec![0.0f64; nrow], Vec::<u32>::new()),
            |(f, touched), (i, row)| {
                for e1 in &lists[i] {
                    let k = e1.target as usize;
                    let kl = e1.pq as usize;
                    if f[k] == 0.0 {
                        touched.push(e1.target);
                    }
                    f[k] += e1.sign * g[kl];
                    for e2 in &lists[k] {
                        let j = e2.target as usize;
                        if f[j] == 0.0 {
                            touched.push(e2.target);
                        }
                        f[j] += 0.5 * e1.sign * e2.sign * eri[e2.pq as usize * n2 + kl];
                    }
                }
                // `touched` may list an index twice (a partial sum can
                // return to exactly 0.0); the coefficient is zeroed on first
                // use so the repeat adds nothing.
                for &j in touched.iter() {
                    let j = j as usize;
                    let w = f[j];
                    if w != 0.0 {
                        let xr = &x[j * ncol..(j + 1) * ncol];
                        for (o, &xv) in row.iter_mut().zip(xr) {
                            *o += w * xv;
                        }
                    }
                    f[j] = 0.0;
                }
                touched.clear();
            },
        );
    }

    /// σ3: for each β pair `rs`, gather `C'(Jα, m) = ε_m C(Jα, Jβ_m)` over the
    /// β strings it connects, then contract the α replacements of every row
    /// `Iα` with `(pq|rs)` and scatter into `σ(Iα, Iβ_m)`.
    fn alpha_beta(&self, c: &[f64], sig: &mut [f64]) {
        let na = self.n_alpha_str;
        let nb = self.n_beta_str;
        let n2 = self.ints.n_active * self.ints.n_active;
        let eri = &self.ints.eri;
        let mut cp: Vec<f64> = Vec::new();
        for (rs, entries) in self.beta_by_pair.iter().enumerate() {
            let len = entries.len();
            if len == 0 {
                continue;
            }
            cp.clear();
            cp.resize(na * len, 0.0);
            cp.par_chunks_mut(len).enumerate().for_each(|(ja, row)| {
                let crow = &c[ja * nb..(ja + 1) * nb];
                for (o, e) in row.iter_mut().zip(entries) {
                    *o = e.sign * crow[e.source as usize];
                }
            });
            let cp = &cp;
            sig.par_chunks_mut(nb).enumerate().for_each_init(
                || Vec::<f64>::new(),
                |v, (ia, srow)| {
                    v.clear();
                    v.resize(len, 0.0);
                    let mut any = false;
                    for e in &self.alpha[ia] {
                        // ⟨Iα|E_qp|Jα⟩ = ε for E_pq|Iα⟩ = ε|Jα⟩, and
                        // (qp|rs) = (pq|rs).
                        let w = e.sign * eri[e.pq as usize * n2 + rs];
                        if w == 0.0 {
                            continue;
                        }
                        any = true;
                        let ja = e.target as usize;
                        for (vm, &cv) in v.iter_mut().zip(&cp[ja * len..(ja + 1) * len]) {
                            *vm += w * cv;
                        }
                    }
                    if any {
                        for (vm, e) in v.iter().zip(entries) {
                            srow[e.target as usize] += vm;
                        }
                    }
                },
            );
        }
    }
}

fn transpose(c: &[f64], nrow: usize, ncol: usize) -> Vec<f64> {
    let mut t = vec![0.0f64; nrow * ncol];
    t.par_chunks_mut(nrow).enumerate().for_each(|(j, trow)| {
        for (i, tv) in trow.iter_mut().enumerate() {
            *tv = c[i * ncol + j];
        }
    });
    t
}
