//! Second derivatives of the density-fitting integrals `(P|μν)` and `(P|Q)`
//! for the analytic RI Hessian (issue #290).
//!
//! The RI-J energy `E_J = ½ dᵀ V⁻¹ d` (`d_P = Σ D_μν (P|μν)`, `V_PQ = (P|Q)`)
//! needs, besides first derivatives, the second derivatives of the
//! three-index integrals `(P|μν)` and of the metric `(P|Q)`, always CONTRACTED
//! against a density-like weight. This module returns those contractions
//! (and the shell-block second derivatives they are built from).
//!
//! # How the second derivatives are obtained
//!
//! The wheel's libint2 (2.7.2 small export) has first derivatives of the
//! three- and two-centre integrals only. A fourth-order central stencil of the
//! FIRST-derivative blocks gives the second derivative: displacing atom `b`
//! along `y` by `o·h` for `o ∈ {−2, −1, 1, 2}` and weighting the first
//! derivative blocks `(1, −8, 8, −1)/(12h)` yields
//! `∂/∂R_{b,y} ∂/∂R_{k,x}` for every centre `k` of the shell triple.
//!
//! * The displacement is RIGID PER ATOM: every shell of the triple that sits
//!   on `b` moves, the auxiliary shell included. Moving one shell of a
//!   same-atom pair is not an atomic displacement and is off by ~1e-3.
//! * Each shell centre's first-derivative block is FOLDED onto its atom
//!   (bra/ket/aux partial to the atom of that shell), so the result is a
//!   block over the DISTINCT atoms of the triple, `AtomBlock`.
//! * A triple whose centres all sit on one atom is translated as a whole
//!   under the displacement and has no second derivative; a triple that does
//!   not include `b` is untouched. Both are skipped (the mask).
//! * Two-centre case: the shifted engine translates only the ket. The metric
//!   depends only on the relative position, so displacing the atom of `Q` by
//!   `d` is a ket shift `+d` and displacing the atom of `P` is a ket shift
//!   `−d` (the first-derivative blocks with respect to both centres are
//!   functions of `P − Q` alone). Because the stencil weights are odd, the
//!   two columns of the 2×2 atom block then satisfy translational invariance
//!   BY CONSTRUCTION; the genuine check is against value differences of
//!   really displaced geometries (see the accuracy tests).
//!
//! # Measured accuracy
//!
//! At the default `STEP` (2e-3 Bohr), against sixth-order Richardson mixed
//! differences of VALUES on really displaced geometries (the independent
//! oracle of `tests/df_hessian_stencil.rs`): water/6-31G/cc-pVDZ-RI misses by
//! 9.7e-11 (3c) and 1.8e-10 (2c) at second derivatives of size up to 2.4 and
//! 3.5, which is the oracle's own roundoff floor; the error of the stencil
//! itself falls as `h^4` (step-difference ratios 15.96, 15.99). Orbital f with
//! aux g (CO/def2-TZVP/def2-universal-jkfit, `(g|ff)` on different atoms) and
//! diffuse sets (aug-cc-pVDZ with its RIFIT aux) stay below 4e-9 relative,
//! again the oracle floor (the stencil step does not move it). Against
//! libint 2.13.1's ANALYTIC three-centre second derivatives (67 shell triples
//! up to `l = 3`, same-atom pairs and three distinct atoms included,
//! `scripts/probes/eri3_deriv2_vs_stencil.cc`) the stencil agrees to 2.7e-11.
//! Exercised up to orbital f with aux g. The first-derivative engines are
//! configured for `l <= 7` in the 2.13.1 source build and `l <= 6` in the
//! wheel's export (`config.h`, `ERI3_MAX_AM_LIST`), against `l <= 3` for
//! libint's three-centre second derivatives; shells above g were not run.
//!
//! Cost: each (shell triple, displaced atom, axis) needs 4 first-derivative
//! block evaluations, so a full Hessian pass makes ~30x the engine calls of
//! the DF gradient's three-centre pass (alkane_4, 6-31G/cc-pVDZ-RI, one
//! thread: 4.4 s against 0.11 s; the metric 0.12 s against 0.004 s); per
//! coordinate that is ~1x the whole gradient pass, i.e. ~4x the gradient
//! integrals of the triples that involve the displaced atom.
//!
//! # Mutants
//!
//! `Variant` selects deliberately broken constructions for the mutation
//! tests (`tests/df_hessian_stencil.rs`); `Variant::Production` is the only
//! one a caller may use. They are wired through the production code path so a
//! test that passes against the mutant cannot be passing vacuously.

use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ndarray::Array2;
use rayon::prelude::*;

/// Displacement step (Bohr) of the fourth-order stencil.
pub const STEP: f64 = 2e-3;

/// `(offset in steps, weight)` of the fourth-order first-derivative stencil
/// (÷ `12·h`).
const D1: [(f64, f64); 4] = [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)];
/// `(offset, weight)` of the second-order stencil (÷ `2·h`); mutant only.
const D1_ORDER2: [(f64, f64); 2] = [(-1.0, -1.0), (1.0, 1.0)];
/// Largest number of aux shells per ordered-reduction chunk.
const MAX_CHUNK: usize = 32;
/// Memory (bytes) one chunk of per-shell `3N × 3N` partials may occupy.
const CHUNK_BYTES: usize = 64 << 20;

/// Aux shells per ordered-reduction chunk of the contracted drivers: a
/// function of the atom count only (never of the thread count), so the
/// reduction order, and with it every bit of the result, is the same for
/// every pool size.
fn chunk_len(natoms: usize) -> usize {
    let partial = 9 * natoms * natoms * std::mem::size_of::<f64>();
    (CHUNK_BYTES / partial.max(1)).clamp(8, MAX_CHUNK)
}

/// Construction of the stencil. Everything but `Variant::Production` is a
/// deliberately wrong mutant for the mutation tests.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Variant {
    /// Rigid per-atom displacement, fourth order, atom-folded.
    #[default]
    Production,
    /// Displace only the FIRST shell on the displaced atom.
    SingleShell,
    /// Wrong sign on the auxiliary displacement.
    AuxSignFlip,
    /// Step multiplied by 10.
    Step10,
    /// Second-order (two-point) weights.
    SecondOrder,
    /// Drop the mixed-atom (`a ≠ b`) elements.
    DropCross,
    /// Fold the bra and ket partials onto each other's atoms.
    SwapBraKet,
}

/// Stencil settings. `Default` is the production construction.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct StencilConfig {
    /// Step in Bohr.
    pub step: f64,
    /// Construction (mutants for tests; production is the default).
    pub variant: Variant,
    /// libint2 precision of the engines the contracted drivers build (the
    /// DF gradient uses 1e-14).
    pub precision: f64,
}

impl Default for StencilConfig {
    fn default() -> Self {
        StencilConfig {
            step: STEP,
            variant: Variant::Production,
            precision: 1e-14,
        }
    }
}

impl StencilConfig {
    fn eff_step(&self) -> f64 {
        if self.variant == Variant::Step10 {
            10.0 * self.step
        } else {
            self.step
        }
    }

    /// `(points, denominator multiple of h)`.
    fn weights(&self) -> (&'static [(f64, f64)], f64) {
        if self.variant == Variant::SecondOrder {
            (&D1_ORDER2, 2.0)
        } else {
            (&D1, 12.0)
        }
    }
}

/// Second derivatives of one shell block over the distinct atoms of its
/// centres: element `(ia, x, ib, y)` is `∂²/∂R_{atoms[ia],x} ∂R_{atoms[ib],y}`
/// of every integral of the block (`n` values, row-major in the block's
/// function indices).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct AtomBlock {
    /// Distinct atoms of the block's centres, in order of first appearance.
    pub atoms: Vec<usize>,
    /// Integrals per element.
    pub n: usize,
    data: Vec<f64>,
}

impl AtomBlock {
    fn zeros(atoms: Vec<usize>, n: usize) -> Self {
        let na = atoms.len();
        AtomBlock {
            data: vec![0.0; na * 3 * na * 3 * n],
            atoms,
            n,
        }
    }

    fn start(&self, ia: usize, x: usize, ib: usize, y: usize) -> usize {
        let na = self.atoms.len();
        (((ia * 3 + x) * na + ib) * 3 + y) * self.n
    }

    /// The `n` second derivatives `∂²/∂R_{atoms[ia],x} ∂R_{atoms[ib],y}`.
    pub fn get(&self, ia: usize, x: usize, ib: usize, y: usize) -> &[f64] {
        let s = self.start(ia, x, ib, y);
        &self.data[s..s + self.n]
    }

    fn get_mut(&mut self, ia: usize, x: usize, ib: usize, y: usize) -> &mut [f64] {
        let s = self.start(ia, x, ib, y);
        let n = self.n;
        &mut self.data[s..s + n]
    }

    fn drop_cross(&mut self) {
        let na = self.atoms.len();
        for ia in 0..na {
            for ib in 0..na {
                if ia != ib {
                    for x in 0..3 {
                        for y in 0..3 {
                            self.get_mut(ia, x, ib, y).fill(0.0);
                        }
                    }
                }
            }
        }
    }
}

fn distinct(atoms: &[usize]) -> Vec<usize> {
    let mut d: Vec<usize> = Vec::with_capacity(atoms.len());
    for &a in atoms {
        if !d.contains(&a) {
            d.push(a);
        }
    }
    d
}

/// Shifts `[aux, bra, ket]` displacing atom `b` along `axis` by `d`.
fn shifts3(atoms: [usize; 3], b: usize, axis: usize, d: f64, cfg: &StencilConfig) -> [[f64; 3]; 3] {
    let mut sh = [[0.0; 3]; 3];
    for (k, &a) in atoms.iter().enumerate() {
        if a == b {
            let sgn = if k == 0 && cfg.variant == Variant::AuxSignFlip {
                -1.0
            } else {
                1.0
            };
            sh[k][axis] += sgn * d;
            if cfg.variant == Variant::SingleShell {
                break;
            }
        }
    }
    sh
}

/// Second derivatives of the shell triple `(P|μν)` over the distinct atoms of
/// `(P, μ, ν)`. `None` when the triple has only one distinct atom (rigidly
/// translated: exactly zero) or every point was screened. `eng` must be a
/// [`Engine::new_3center_deriv`] engine.
#[doc(hidden)]
pub fn eri3_deriv2_block(
    eng: &mut Engine,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    shells: (usize, usize, usize),
    cfg: &StencilConfig,
) -> Result<Option<AtomBlock>, FerricError> {
    let (sh_p, s1, s2) = shells;
    let atoms = [
        dfbs.shell_to_atom()[sh_p],
        obs.shell_to_atom()[s1],
        obs.shell_to_atom()[s2],
    ];
    let dist = distinct(&atoms);
    if dist.len() == 1 {
        return Ok(None);
    }
    let n = dfbs.shell_dims()[sh_p] * obs.shell_dims()[s1] * obs.shell_dims()[s2];
    let fold: [usize; 3] = if cfg.variant == Variant::SwapBraKet {
        [atoms[0], atoms[2], atoms[1]]
    } else {
        atoms
    };
    let (pts, den) = cfg.weights();
    let h = cfg.eff_step();
    let mut out = AtomBlock::zeros(dist.clone(), n);
    let mut any = false;
    for (ib, &b) in dist.iter().enumerate() {
        for y in 0..3 {
            for &(off, wt) in pts {
                let sh = shifts3(atoms, b, y, off * h, cfg);
                let Some(blk) = eng.compute_eri3_deriv_shifted(obs, dfbs, sh_p, s1, s2, sh)? else {
                    continue;
                };
                any = true;
                let f = wt / (den * h);
                for (k, &fa) in fold.iter().enumerate() {
                    let ia = dist.iter().position(|&a| a == fa).expect("fold atom");
                    for x in 0..3 {
                        let src = &blk[(3 * k + x) * n..(3 * k + x + 1) * n];
                        for (o, v) in out.get_mut(ia, x, ib, y).iter_mut().zip(src) {
                            *o += f * v;
                        }
                    }
                }
            }
        }
    }
    if !any {
        return Ok(None);
    }
    if cfg.variant == Variant::DropCross {
        out.drop_cross();
    }
    Ok(Some(out))
}

/// Second derivatives of the shell pair `(P|Q)` over the atoms `[A(P), A(Q)]`.
/// `None` when both shells sit on one atom (the metric block is rigidly
/// translated: exactly zero) or the pair is screened. `eng` must be a
/// [`Engine::new_2center_deriv`] engine.
#[doc(hidden)]
pub fn eri2_deriv2_block(
    eng: &mut Engine,
    dfbs: &PreparedBasis,
    shells: (usize, usize),
    cfg: &StencilConfig,
) -> Result<Option<AtomBlock>, FerricError> {
    let (sp, sq) = shells;
    let (ap, aq) = (dfbs.shell_to_atom()[sp], dfbs.shell_to_atom()[sq]);
    if ap == aq {
        return Ok(None);
    }
    let dims = dfbs.shell_dims();
    let n = dims[sp] * dims[sq];
    let (pts, den) = cfg.weights();
    let h = cfg.eff_step();
    let mut out = AtomBlock::zeros(vec![ap, aq], n);
    let mut any = false;
    // Displacing the atom of Q by +d = ket shift +d; the atom of P by +d = ket
    // shift -d (translation invariance of the pair).
    let flip = if cfg.variant == Variant::AuxSignFlip {
        -1.0
    } else {
        1.0
    };
    for (ib, sgn) in [(0usize, -1.0), (1usize, 1.0)] {
        for y in 0..3 {
            for &(off, wt) in pts {
                let mut shift = [0.0; 3];
                shift[y] = flip * sgn * off * h;
                let Some(blk) = eng.compute_eri2_deriv_shifted(dfbs, sp, sq, shift)? else {
                    continue;
                };
                any = true;
                let f = wt / (den * h);
                for k in 0..2 {
                    let ia = if cfg.variant == Variant::SwapBraKet {
                        1 - k
                    } else {
                        k
                    };
                    for x in 0..3 {
                        let src = &blk[(3 * k + x) * n..(3 * k + x + 1) * n];
                        for (o, v) in out.get_mut(ia, x, ib, y).iter_mut().zip(src) {
                            *o += f * v;
                        }
                    }
                }
            }
        }
    }
    if !any {
        return Ok(None);
    }
    if cfg.variant == Variant::DropCross {
        out.drop_cross();
    }
    Ok(Some(out))
}

/// Adds `Σ_i w_i ∂²X_i/∂R_{a,x}∂R_{b,y}` of one block to the `3N × 3N` matrix.
fn accumulate(h: &mut Array2<f64>, blk: &AtomBlock, w: &[f64]) {
    for (ia, &a) in blk.atoms.iter().enumerate() {
        for (ib, &b) in blk.atoms.iter().enumerate() {
            for x in 0..3 {
                for y in 0..3 {
                    let v: f64 = blk
                        .get(ia, x, ib, y)
                        .iter()
                        .zip(w)
                        .map(|(g, wi)| g * wi)
                        .sum();
                    h[(3 * a + x, 3 * b + y)] += v;
                }
            }
        }
    }
}

/// `Σ_{P,μν} W_{Pμν} ∂²(P|μν)/∂R_{a,x}∂R_{b,y}` (3N × 3N, N = atoms of the
/// aux basis) for a weight supplied per canonical shell triple `s2 ≤ s1`.
///
/// `weight(sh_p, s1, s2, w)` fills `w` (length `nP·n1·n2`, row-major
/// `(P, μ, ν)`) with the weight of that block — for `s1 ≠ s2` it must already
/// include both orders (`W_{Pμν} + W_{Pνμ}`) — and returns `false` if the
/// block's weight is identically zero. Reduction is chunked over aux shells
/// and summed in shell order, so the result is bit-identical for every thread
/// count.
#[doc(hidden)]
pub fn eri3_deriv2_contract<W>(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    cfg: &StencilConfig,
    weight: W,
) -> Result<Array2<f64>, FerricError>
where
    W: Fn(usize, usize, usize, &mut Vec<f64>) -> bool + Sync,
{
    let natoms = dfbs.atoms().len();
    let mut h = Array2::<f64>::zeros((3 * natoms, 3 * natoms));
    Engine::new_3center_deriv(op, obs, dfbs, cfg.precision)?;
    let shells: Vec<usize> = (0..dfbs.nshells()).collect();
    for chunk in shells.chunks(chunk_len(natoms)) {
        let partials: Vec<Result<Array2<f64>, FerricError>> = chunk
            .par_iter()
            .map_init(
                || {
                    Engine::new_3center_deriv(op, obs, dfbs, cfg.precision)
                        .expect("pre-validated engine")
                },
                |eng, &sp| {
                    let mut local = Array2::<f64>::zeros((3 * natoms, 3 * natoms));
                    let mut w = Vec::new();
                    for s1 in 0..obs.nshells() {
                        for s2 in 0..=s1 {
                            w.clear();
                            if !weight(sp, s1, s2, &mut w) {
                                continue;
                            }
                            if let Some(blk) = eri3_deriv2_block(eng, obs, dfbs, (sp, s1, s2), cfg)?
                            {
                                accumulate(&mut local, &blk, &w);
                            }
                        }
                    }
                    Ok(local)
                },
            )
            .collect();
        for p in partials {
            h += &p?;
        }
    }
    Ok(h)
}

/// `Σ_PQ W_PQ ∂²(P|Q)/∂R_{a,x}∂R_{b,y}` for a weight per canonical aux shell
/// pair `sq ≤ sp`, with the same conventions as [`eri3_deriv2_contract`]
/// (`weight(sp, sq, w)` includes both orders when `sp ≠ sq`).
#[doc(hidden)]
pub fn eri2_deriv2_contract<W>(
    dfbs: &PreparedBasis,
    op: Operator,
    cfg: &StencilConfig,
    weight: W,
) -> Result<Array2<f64>, FerricError>
where
    W: Fn(usize, usize, &mut Vec<f64>) -> bool + Sync,
{
    let natoms = dfbs.atoms().len();
    let mut h = Array2::<f64>::zeros((3 * natoms, 3 * natoms));
    Engine::new_2center_deriv(op, dfbs, cfg.precision)?;
    let shells: Vec<usize> = (0..dfbs.nshells()).collect();
    for chunk in shells.chunks(chunk_len(natoms)) {
        let partials: Vec<Result<Array2<f64>, FerricError>> = chunk
            .par_iter()
            .map_init(
                || {
                    Engine::new_2center_deriv(op, dfbs, cfg.precision)
                        .expect("pre-validated engine")
                },
                |eng, &sp| {
                    let mut local = Array2::<f64>::zeros((3 * natoms, 3 * natoms));
                    let mut w = Vec::new();
                    for sq in 0..=sp {
                        w.clear();
                        if !weight(sp, sq, &mut w) {
                            continue;
                        }
                        if let Some(blk) = eri2_deriv2_block(eng, dfbs, (sp, sq), cfg)? {
                            accumulate(&mut local, &blk, &w);
                        }
                    }
                    Ok(local)
                },
            )
            .collect();
        for p in partials {
            h += &p?;
        }
    }
    Ok(h)
}

/// Block weights `c_P · D_μν` (both orders for `s1 ≠ s2`) of the RI-J
/// skeleton term `Σ c_P D_μν ∂²(P|μν)`; `c` is indexed by aux function.
#[doc(hidden)]
pub fn weight_c_times_d(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    c: &[f64],
    d: &Array2<f64>,
    shells: (usize, usize, usize),
    w: &mut Vec<f64>,
) -> bool {
    let (sp, s1, s2) = shells;
    let (od, dd) = (obs.shell_dims(), dfbs.shell_dims());
    let (oo, do_) = (obs.shell_offsets(), dfbs.shell_offsets());
    for p in 0..dd[sp] {
        let cp = c[do_[sp] + p];
        for a in 0..od[s1] {
            for b in 0..od[s2] {
                let (mu, nu) = (oo[s1] + a, oo[s2] + b);
                let dv = if s1 != s2 {
                    d[(mu, nu)] + d[(nu, mu)]
                } else {
                    d[(mu, nu)]
                };
                w.push(cp * dv);
            }
        }
    }
    true
}

/// Block weights `m_PQ` (both orders for `sp ≠ sq`) for the metric
/// contraction `Σ m_PQ ∂²(P|Q)`.
#[doc(hidden)]
pub fn weight_matrix(
    dfbs: &PreparedBasis,
    m: &Array2<f64>,
    shells: (usize, usize),
    w: &mut Vec<f64>,
) -> bool {
    let (sp, sq) = shells;
    let (dd, off) = (dfbs.shell_dims(), dfbs.shell_offsets());
    for p in 0..dd[sp] {
        for q in 0..dd[sq] {
            let (i, j) = (off[sp] + p, off[sq] + q);
            w.push(if sp != sq {
                m[(i, j)] + m[(j, i)]
            } else {
                m[(i, j)]
            });
        }
    }
    true
}
