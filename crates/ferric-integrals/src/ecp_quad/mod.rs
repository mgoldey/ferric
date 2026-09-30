//! ferric-owned ECP integrals: analytic angular projection + windowed radial
//! quadrature (FINDINGS "Iteration 25", pure Rust, no FFI). The
//! [`crate::ecp::EcpBackend::Quadrature`] backend of the four public ECP
//! functions in [`crate::ecp`].
//!
//! Output is exactly the libecpint shim's: bare-Cartesian coefficients in,
//! Cartesian blocks in CCA order out (row-major), so `ecp.rs`'s
//! Cartesian→spherical code is reused unchanged.
//!
//! * Semi-local channels (`type2`): exp-scaled modified spherical Bessel
//!   functions ([`bessel`](crate::ecp_quad::bessel)) — stable at every distance and exact on-centre —
//!   with the r-independent angular tensor `T` hoisted out of every
//!   primitive/term/node loop and ONE radial tensor per (shell pair, centre,
//!   channel).
//! * Local channel (`type1`): the product Gaussian's plane-wave expansion in
//!   the same `k̃_λ`.
//! * Radial ([`radial`](crate::ecp_quad::radial)): per-(primitive pair, ECP term) windows; Gauss–Hermite
//!   away from r = 0, adaptive Gauss–Legendre (16–40 nodes) otherwise.
//! * Derivatives: raised/lowered shells (`∂χ/∂A_x = 2α χ(i+1) − i χ(i−1)`);
//!   centre `= −(bra + ket)` per triple (exact translation invariance).
//!
//! Every function is pure (the only statics are lazily-built immutable tables),
//! so all entry points run shell pairs in parallel with rayon.

pub mod bessel;
pub mod harmonics;
pub mod radial;
mod type1;
mod type2;

use crate::ecp::{EcpCenter, EcpGaussianShell};
use ferric_core::FerricError;
use harmonics::{cart_index, cart_list, ncart};
use rayon::prelude::*;
use std::sync::OnceLock;

pub use radial::RadialRule;

const FOURPI: f64 = 4.0 * std::f64::consts::PI;
/// A (primitive pair, term) window whose bound is below this is skipped.
const SCREEN: f64 = 1e-16;
/// Largest shell `l` the public API accepts (the cart2sph tables).
pub const MAX_SHELL_L: usize = 4;
/// Largest SEMI-LOCAL ECP channel `l` (the local channel is unrestricted).
pub const MAX_ECP_SEMI_L: usize = harmonics::Q_LMAX;

/// Deliberate constructions that MUST fail the accuracy tests (negative
/// controls). Production is [`Mutation::None`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutation {
    /// Production.
    None,
    /// Unscaled `i_n` with the unfactored Gaussians (`e^{z}` × `e^{−α(r²+A²)}`):
    /// overflow × underflow at long range.
    UnscaledBessel,
    /// Drop the `λ > l` terms of the plane-wave expansion (an "on-centre"
    /// approximation of the semi-local projection).
    TruncateLambda,
    /// `U = d r^{n−1} e^{−ζr²}` instead of `r^{n−2}` (every channel).
    RadialPowerShift,
}

/// Accuracy knobs of one call. [`QuadKnobs::default`] is production; the
/// others exist for the screening and negative-control tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuadKnobs {
    /// Skip (primitive pair, term) windows whose bound is < 1e-16.
    pub screen: bool,
    /// Radial node policy.
    pub rule: RadialRule,
    /// Deliberately broken construction (tests only).
    pub mutation: Mutation,
}

impl Default for QuadKnobs {
    fn default() -> Self {
        Self {
            screen: true,
            rule: RadialRule::default(),
            mutation: Mutation::None,
        }
    }
}

/// A contracted shell (bare-Cartesian coefficients).
#[derive(Debug, Clone)]
pub(crate) struct QShell {
    pub l: usize,
    pub center: [f64; 3],
    pub exps: Vec<f64>,
    pub coefs: Vec<f64>,
}

/// One ECP term `d r^{n−2} e^{−ζ r²}`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Term {
    pub n: i32,
    pub zeta: f64,
    pub d: f64,
}

/// An ECP centre split into its semi-local channels (every `l` except the
/// maximum) and the local channel (the maximum `l`, libecpint convention).
#[derive(Debug, Clone)]
pub(crate) struct SplitEcp {
    pub center: [f64; 3],
    pub semi: Vec<(usize, Vec<Term>)>,
    pub local: Vec<Term>,
}

fn split_ecp(e: &EcpCenter, knobs: &QuadKnobs) -> SplitEcp {
    let lmax = e.ams.iter().copied().max().unwrap_or(0);
    let shift = i32::from(knobs.mutation == Mutation::RadialPowerShift);
    let mut semi: Vec<(usize, Vec<Term>)> = Vec::new();
    let mut local = Vec::new();
    for k in 0..e.ams.len() {
        let t = Term {
            n: e.ns[k] + shift,
            zeta: e.exponents[k],
            d: e.coefficients[k],
        };
        if e.ams[k] == lmax {
            local.push(t);
        } else {
            let l = e.ams[k] as usize;
            match semi.iter_mut().find(|(ll, _)| *ll == l) {
                Some((_, v)) => v.push(t),
                None => semi.push((l, vec![t])),
            }
        }
    }
    semi.sort_by_key(|(l, _)| *l);
    SplitEcp {
        center: e.center,
        semi,
        local,
    }
}

fn qshell(s: &EcpGaussianShell) -> QShell {
    QShell {
        l: s.l as usize,
        center: s.center,
        exps: s.exponents.clone(),
        coefs: s.coefficients.clone(),
    }
}

/// The raised (`+1`: coefficients `2αc`) or lowered (`−1`: coefficients `c`)
/// companion of a shell; `None` for the lowered companion of an s shell.
fn companion(s: &QShell, raise: bool) -> Option<QShell> {
    if raise {
        Some(QShell {
            l: s.l + 1,
            center: s.center,
            exps: s.exps.clone(),
            coefs: s
                .exps
                .iter()
                .zip(&s.coefs)
                .map(|(a, c)| 2.0 * a * c)
                .collect(),
        })
    } else if s.l > 0 {
        Some(QShell {
            l: s.l - 1,
            ..s.clone()
        })
    } else {
        None
    }
}

/// Quadrature-specific input limits (ecp.rs checks the shells' 0..=4 range,
/// list lengths and the mask).
pub(crate) fn validate(
    shells: &[&EcpGaussianShell],
    ecps: &[EcpCenter],
) -> Result<(), FerricError> {
    for s in shells {
        if !(0..=MAX_SHELL_L as i32).contains(&s.l) {
            return Err(FerricError::General(format!(
                "ECP quadrature: shell l = {} outside 0..={MAX_SHELL_L}",
                s.l
            )));
        }
        if s.exponents.iter().any(|&a| !(a.is_finite() && a > 0.0))
            || s.coefficients.iter().any(|c| !c.is_finite())
            || s.center.iter().any(|c| !c.is_finite())
        {
            return Err(FerricError::General(
                "ECP quadrature: non-finite or non-positive shell data".into(),
            ));
        }
    }
    for e in ecps {
        let n = e.ams.len();
        if n == 0 || e.ns.len() != n || e.exponents.len() != n || e.coefficients.len() != n {
            return Err(FerricError::General(
                "ECP quadrature: ragged or empty ECP term lists".into(),
            ));
        }
        let lmax = *e.ams.iter().max().unwrap_or(&0);
        for k in 0..n {
            let (l, nn, z, d) = (e.ams[k], e.ns[k], e.exponents[k], e.coefficients[k]);
            if l < 0 || (l != lmax && l as usize > MAX_ECP_SEMI_L) {
                return Err(FerricError::General(format!(
                    "ECP quadrature: channel l = {l} (semi-local l must lie in 0..={MAX_ECP_SEMI_L})"
                )));
            }
            if !(0..=12).contains(&nn) || !(z.is_finite() && z > 0.0) || !d.is_finite() {
                return Err(FerricError::General(format!(
                    "ECP quadrature: bad term (n = {nn}, zeta = {z}, d = {d})"
                )));
            }
        }
        if e.center.iter().any(|c| !c.is_finite()) {
            return Err(FerricError::General(
                "ECP quadrature: non-finite ECP centre".into(),
            ));
        }
    }
    Ok(())
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// One side (bra or ket) of a call: the shells, their raised/lowered
/// companions, and a lazily filled, call-local cache of projection tensors
/// keyed by (shell, ECP centre, variant) — only the triples actually
/// evaluated (after screening) build a tensor.
struct Side {
    /// `var[s] = [lowered, base, raised]`.
    var: Vec<[Option<QShell>; 3]>,
    necp: usize,
    /// Cache slots per (shell, ECP): 3 with derivatives, else 1 (base only —
    /// the periodic kernel calls with many ket/ECP images).
    nvar: usize,
    cache: Vec<OnceLock<Vec<Vec<f64>>>>,
}

impl Side {
    fn new(shells: &[EcpGaussianShell], necp: usize, derivs: bool) -> Self {
        let var: Vec<[Option<QShell>; 3]> = shells
            .iter()
            .map(|s| {
                let b = qshell(s);
                if derivs {
                    [companion(&b, false), Some(b.clone()), companion(&b, true)]
                } else {
                    [None, Some(b), None]
                }
            })
            .collect();
        let nvar = if derivs { 3 } else { 1 };
        let cache = (0..var.len() * necp * nvar)
            .map(|_| OnceLock::new())
            .collect();
        Self {
            var,
            necp,
            nvar,
            cache,
        }
    }

    fn shell(&self, s: usize, v: usize) -> Option<&QShell> {
        self.var[s][v].as_ref()
    }

    /// Projection tensors (one per semi-local channel of `ecps[u]`) of
    /// variant `v` of shell `s`.
    fn proj(&self, s: usize, v: usize, u: usize, e: &SplitEcp, knobs: &QuadKnobs) -> &[Vec<f64>] {
        let slot = if self.nvar == 3 { v } else { 0 };
        self.cache[(s * self.necp + u) * self.nvar + slot].get_or_init(|| {
            let sh = self.var[s][v].as_ref().expect("variant exists");
            let a = sub(sh.center, e.center);
            e.semi
                .iter()
                .map(|(l, _)| {
                    type2::proj_tensor(sh.l, *l, a, knobs.mutation == Mutation::TruncateLambda)
                })
                .collect()
        })
    }
}

/// `⟨sa|U_e|sb⟩` added to `out` (Cartesian `ncart(la) × ncart(lb)`).
/// `ta`/`tb` fetch the projection tensors only if some channel survives
/// screening.
fn triple_cart<'c>(
    sa: &QShell,
    ta: &dyn Fn() -> &'c [Vec<f64>],
    sb: &QShell,
    tb: &dyn Fn() -> &'c [Vec<f64>],
    e: &SplitEcp,
    knobs: &QuadKnobs,
    out: &mut [f64],
) {
    let a = sub(sa.center, e.center);
    let b = sub(sb.center, e.center);
    let (an, bn) = (norm(a), norm(b));
    for (ch, (l, terms)) in e.semi.iter().enumerate() {
        if let Some(rt) = type2::radial_tensor(sa, sb, an, bn, *l, terms, knobs) {
            type2::contract(sa.l, sb.l, *l, &ta()[ch], &tb()[ch], &rt, out);
        }
    }
    if !e.local.is_empty() {
        type1::type1_cart(sa, sb, a, b, &e.local, knobs, out);
    }
}

/// Everything one call shares across its rayon workers.
struct Ctx<'a> {
    bra: Side,
    ket: Side,
    ecps: Vec<SplitEcp>,
    knobs: &'a QuadKnobs,
}

impl<'a> Ctx<'a> {
    fn new(
        bra: &[EcpGaussianShell],
        ket: &[EcpGaussianShell],
        ecps: &[EcpCenter],
        knobs: &'a QuadKnobs,
        derivs: bool,
    ) -> Self {
        Self {
            bra: Side::new(bra, ecps.len(), derivs),
            ket: Side::new(ket, ecps.len(), derivs),
            ecps: ecps.iter().map(|e| split_ecp(e, knobs)).collect(),
            knobs,
        }
    }

    /// Cartesian value block of (bra `a` variant `va`, ket `b` variant `vb`,
    /// ECP `u`), freshly allocated.
    fn value(&self, a: usize, va: usize, b: usize, vb: usize, u: usize) -> Vec<f64> {
        let sa = self.bra.shell(a, va).expect("bra variant");
        let sb = self.ket.shell(b, vb).expect("ket variant");
        let e = &self.ecps[u];
        let mut out = vec![0.0; ncart(sa.l) * ncart(sb.l)];
        let ta = || self.bra.proj(a, va, u, e, self.knobs);
        let tb = || self.ket.proj(b, vb, u, e, self.knobs);
        triple_cart(sa, &ta, sb, &tb, e, self.knobs, &mut out);
        out
    }

    /// `(∂/∂A, ∂/∂B)` of the Cartesian block of triple `(a, b, u)`, each
    /// `[x] → ncart(la) × ncart(lb)`.
    fn deriv(&self, a: usize, b: usize, u: usize) -> ([Vec<f64>; 3], [Vec<f64>; 3]) {
        let la = self.bra.shell(a, 1).expect("bra").l;
        let lb = self.ket.shell(b, 1).expect("ket").l;
        let (nca, ncb) = (ncart(la), ncart(lb));
        let mut da: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; nca * ncb]);
        let mut db: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; nca * ncb]);
        // bra: rows from the raised (2αc) / lowered (c) bra shells
        let vup = self.value(a, 2, b, 1, u);
        let vdn = (la > 0).then(|| self.value(a, 0, b, 1, u));
        // ket: columns from the raised / lowered ket shells
        let wup = self.value(a, 1, b, 2, u);
        let wdn = (lb > 0).then(|| self.value(a, 1, b, 0, u));
        let ncbu = ncart(lb + 1);
        let ncbd = if lb > 0 { ncart(lb - 1) } else { 0 };
        let ca = cart_list(la);
        let cb = cart_list(lb);
        for x in 0..3 {
            for (ic, ijk) in ca.iter().enumerate() {
                let mut up = *ijk;
                up[x] += 1;
                let iu = cart_index(up[1], up[2]);
                let dn = (ijk[x] > 0).then(|| {
                    let mut d = *ijk;
                    d[x] -= 1;
                    cart_index(d[1], d[2])
                });
                for jc in 0..ncb {
                    let mut v = vup[iu * ncb + jc];
                    if let (Some(id), Some(vd)) = (dn, &vdn) {
                        v -= ijk[x] as f64 * vd[id * ncb + jc];
                    }
                    da[x][ic * ncb + jc] = v;
                }
            }
            for (jc, ijk) in cb.iter().enumerate() {
                let mut up = *ijk;
                up[x] += 1;
                let ju = cart_index(up[1], up[2]);
                let dn = (ijk[x] > 0).then(|| {
                    let mut d = *ijk;
                    d[x] -= 1;
                    cart_index(d[1], d[2])
                });
                for ic in 0..nca {
                    let mut v = wup[ic * ncbu + ju];
                    if let (Some(jd), Some(wd)) = (dn, &wdn) {
                        v -= ijk[x] as f64 * wd[ic * ncbd + jd];
                    }
                    db[x][ic * ncb + jc] = v;
                }
            }
        }
        (da, db)
    }
}

fn offsets(sh: &[EcpGaussianShell]) -> (Vec<usize>, usize) {
    let mut off = Vec::with_capacity(sh.len());
    let mut c = 0;
    for s in sh {
        off.push(c);
        c += ncart(s.l as usize);
    }
    (off, c)
}

#[inline]
fn enabled(mask: Option<&[u8]>, a: usize, b: usize, u: usize, nket: usize, necp: usize) -> bool {
    mask.map_or(true, |m| m[(a * nket + b) * necp + u] != 0)
}

/// Rectangular Cartesian block `Σ_{(a,b,u) enabled} ⟨bra_a|U_u|ket_b⟩`,
/// `ncart(bra) × ncart(ket)` row-major — the shim's `ferric_ecp_block`
/// layout. Mask index `(a · nket + b) · necp + u` (`None` = all).
pub(crate) fn block_cart(
    bra: &[EcpGaussianShell],
    ket: &[EcpGaussianShell],
    ecps: &[EcpCenter],
    mask: Option<&[u8]>,
    knobs: &QuadKnobs,
) -> Vec<f64> {
    let ctx = Ctx::new(bra, ket, ecps, knobs, false);
    let (nk, ne) = (ket.len(), ecps.len());
    let (offa, nra) = offsets(bra);
    let (offb, ncb_tot) = offsets(ket);
    let blocks: Vec<Vec<f64>> = (0..bra.len() * nk)
        .into_par_iter()
        .map(|pair| {
            let (a, b) = (pair / nk, pair % nk);
            let n = ncart(bra[a].l as usize) * ncart(ket[b].l as usize);
            let mut acc = vec![0.0; n];
            for u in (0..ne).filter(|&u| enabled(mask, a, b, u, nk, ne)) {
                for (d, s) in acc.iter_mut().zip(ctx.value(a, 1, b, 1, u)) {
                    *d += s;
                }
            }
            acc
        })
        .collect();
    let mut v = vec![0.0; nra * ncb_tot];
    for (pair, blk) in blocks.iter().enumerate() {
        let (a, b) = (pair / nk, pair % nk);
        let ncb = ncart(ket[b].l as usize);
        for (i, row) in blk.chunks(ncb).enumerate() {
            let o = (offa[a] + i) * ncb_tot + offb[b];
            v[o..o + ncb].copy_from_slice(row);
        }
    }
    v
}

/// Square symmetric Cartesian matrix `Σ_u ⟨s_i|U_u|s_j⟩` (the shim's
/// `ferric_ecp_matrix` layout): upper-triangle shell pairs, mirrored.
pub(crate) fn matrix_cart(
    shells: &[EcpGaussianShell],
    ecps: &[EcpCenter],
    knobs: &QuadKnobs,
) -> Vec<f64> {
    let ctx = Ctx::new(shells, shells, ecps, knobs, false);
    let ns = shells.len();
    let (off, nc) = offsets(shells);
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|i| (i..ns).map(move |j| (i, j))).collect();
    let blocks: Vec<Vec<f64>> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut acc = vec![0.0; ncart(shells[i].l as usize) * ncart(shells[j].l as usize)];
            for u in 0..ecps.len() {
                for (d, s) in acc.iter_mut().zip(ctx.value(i, 1, j, 1, u)) {
                    *d += s;
                }
            }
            acc
        })
        .collect();
    let mut v = vec![0.0; nc * nc];
    for (&(i, j), blk) in pairs.iter().zip(&blocks) {
        let ncj = ncart(shells[j].l as usize);
        for (p, row) in blk.chunks(ncj).enumerate() {
            for (q, &x) in row.iter().enumerate() {
                v[(off[i] + p) * nc + off[j] + q] = x;
                v[(off[j] + q) * nc + off[i] + p] = x;
            }
        }
    }
    v
}

/// Cartesian derivative blocks of [`block_cart`]: `(bra, ket, centre)` with
/// `bra`/`ket` `3 · blk` long (`[x][row][col]`) and `centre`
/// `ngroup · 3 · blk` (`[g][x][row][col]`), `blk = ncart(bra) · ncart(ket)` —
/// the shim's `ferric_ecp_block_deriv` layout. `centre = −(bra + ket)` per
/// triple, accumulated under `centre_group[u]`.
pub(crate) fn block_deriv_cart(
    bra: &[EcpGaussianShell],
    ket: &[EcpGaussianShell],
    ecps: &[EcpCenter],
    mask: Option<&[u8]>,
    centre_group: &[usize],
    ngroup: usize,
    knobs: &QuadKnobs,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    type PairDeriv = ([Vec<f64>; 3], [Vec<f64>; 3], Vec<(usize, [Vec<f64>; 3])>);
    let ctx = Ctx::new(bra, ket, ecps, knobs, true);
    let (nk, ne) = (ket.len(), ecps.len());
    let (offa, nra) = offsets(bra);
    let (offb, ncb_tot) = offsets(ket);
    let blocks: Vec<PairDeriv> = (0..bra.len() * nk)
        .into_par_iter()
        .map(|pair| {
            let (a, b) = (pair / nk, pair % nk);
            let n = ncart(bra[a].l as usize) * ncart(ket[b].l as usize);
            let mut sa: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; n]);
            let mut sb: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; n]);
            let mut cen: Vec<(usize, [Vec<f64>; 3])> = Vec::new();
            for u in (0..ne).filter(|&u| enabled(mask, a, b, u, nk, ne)) {
                let (da, db) = ctx.deriv(a, b, u);
                let g = centre_group[u];
                let k = match cen.iter().position(|(gg, _)| *gg == g) {
                    Some(k) => k,
                    None => {
                        cen.push((g, std::array::from_fn(|_| vec![0.0; n])));
                        cen.len() - 1
                    }
                };
                for x in 0..3 {
                    for i in 0..n {
                        sa[x][i] += da[x][i];
                        sb[x][i] += db[x][i];
                        cen[k].1[x][i] -= da[x][i] + db[x][i];
                    }
                }
            }
            (sa, sb, cen)
        })
        .collect();
    let blk = nra * ncb_tot;
    let mut d_bra = vec![0.0; 3 * blk];
    let mut d_ket = vec![0.0; 3 * blk];
    let mut d_cen = vec![0.0; ngroup * 3 * blk];
    for (pair, (sa, sb, cen)) in blocks.iter().enumerate() {
        let (a, b) = (pair / nk, pair % nk);
        let ncb = ncart(ket[b].l as usize);
        let scatter = |dst: &mut [f64], src: &[f64]| {
            for (i, row) in src.chunks(ncb).enumerate() {
                let o = (offa[a] + i) * ncb_tot + offb[b];
                for (d, s) in dst[o..o + ncb].iter_mut().zip(row) {
                    *d += s;
                }
            }
        };
        for x in 0..3 {
            scatter(&mut d_bra[x * blk..(x + 1) * blk], &sa[x][..]);
            scatter(&mut d_ket[x * blk..(x + 1) * blk], &sb[x][..]);
            for (g, c) in cen {
                let o = (g * 3 + x) * blk;
                scatter(&mut d_cen[o..o + blk], &c[x][..]);
            }
        }
    }
    (d_bra, d_ket, d_cen)
}

/// Molecular `dV_ECP/dR` (Cartesian): `3 · natoms` matrices `ncart² `
/// (`[3·atom + x][row][col]`, the shim's `ferric_ecp_matrix_deriv` layout).
/// The per-atom fold of the block derivative: for every shell pair `i ≤ j`
/// and ECP `u`, `∂/∂A` goes to `atom_of_shell[i]`, `∂/∂B` to
/// `atom_of_shell[j]`, `−(∂/∂A + ∂/∂B)` to `atom_of_ecp[u]`; the `(j, i)`
/// block is the transpose (dV/dR is symmetric).
pub(crate) fn matrix_deriv_cart(
    shells: &[EcpGaussianShell],
    ecps: &[EcpCenter],
    atom_of_shell: &[usize],
    atom_of_ecp: &[usize],
    natoms: usize,
    knobs: &QuadKnobs,
) -> Vec<f64> {
    type PairFold = Vec<(usize, [Vec<f64>; 3])>;
    let ctx = Ctx::new(shells, shells, ecps, knobs, true);
    let ns = shells.len();
    let (off, nc) = offsets(shells);
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|i| (i..ns).map(move |j| (i, j))).collect();
    let folds: Vec<PairFold> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let n = ncart(shells[i].l as usize) * ncart(shells[j].l as usize);
            let mut acc: PairFold = Vec::new();
            let mut add = |atom: usize, x: usize, src: &[f64], sign: f64| {
                let k = match acc.iter().position(|(a, _)| *a == atom) {
                    Some(k) => k,
                    None => {
                        acc.push((atom, std::array::from_fn(|_| vec![0.0; n])));
                        acc.len() - 1
                    }
                };
                for (d, s) in acc[k].1[x].iter_mut().zip(src) {
                    *d += sign * s;
                }
            };
            for u in 0..ecps.len() {
                let (da, db) = ctx.deriv(i, j, u);
                for x in 0..3 {
                    add(atom_of_shell[i], x, &da[x][..], 1.0);
                    add(atom_of_shell[j], x, &db[x][..], 1.0);
                    add(atom_of_ecp[u], x, &da[x][..], -1.0);
                    add(atom_of_ecp[u], x, &db[x][..], -1.0);
                }
            }
            acc
        })
        .collect();
    let block = nc * nc;
    let mut d = vec![0.0; 3 * natoms * block];
    for (&(i, j), fold) in pairs.iter().zip(&folds) {
        let ncj = ncart(shells[j].l as usize);
        for (atom, per_x) in fold {
            for (x, blk) in per_x.iter().enumerate() {
                let base = (3 * atom + x) * block;
                for (p, row) in blk.chunks(ncj).enumerate() {
                    for (q, &v) in row.iter().enumerate() {
                        d[base + (off[i] + p) * nc + off[j] + q] += v;
                        if i != j {
                            d[base + (off[j] + q) * nc + off[i] + p] += v;
                        }
                    }
                }
            }
        }
    }
    d
}

/// Debug/test helper: the SEMI-LOCAL part of `⟨sa|U_e|sb⟩` (Cartesian) by the
/// production radial-tensor form (`direct = false`) or by forming
/// `F^A_lm(r)`, `F^B_lm(r)` explicitly at every node (`direct = true`) — an
/// independent contraction of the same projections.
pub fn semi_local_cart(
    sa: &EcpGaussianShell,
    sb: &EcpGaussianShell,
    e: &EcpCenter,
    direct: bool,
) -> Result<Vec<f64>, FerricError> {
    validate(&[sa, sb], std::slice::from_ref(e))?;
    let knobs = QuadKnobs {
        screen: false,
        ..QuadKnobs::default()
    };
    let (qa, qb) = (qshell(sa), qshell(sb));
    let se = split_ecp(e, &knobs);
    let a = sub(qa.center, se.center);
    let b = sub(qb.center, se.center);
    let mut out = vec![0.0; ncart(qa.l) * ncart(qb.l)];
    for (l, terms) in &se.semi {
        if direct {
            type2::direct(&qa, &qb, a, b, *l, terms, &knobs, &mut out);
        } else if let Some(rt) = type2::radial_tensor(&qa, &qb, norm(a), norm(b), *l, terms, &knobs)
        {
            let ta = type2::proj_tensor(qa.l, *l, a, false);
            let tb = type2::proj_tensor(qb.l, *l, b, false);
            type2::contract(qa.l, qb.l, *l, &ta, &tb, &rt, &mut out);
        }
    }
    Ok(out)
}
