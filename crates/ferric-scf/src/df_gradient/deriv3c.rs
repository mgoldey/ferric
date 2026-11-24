//! The three-centre derivative pass of the DF-JK gradient (`dfg:3c-deriv-pass`):
//! `Σ_{P,μν} G_P,μν ∂(P|μν)` with `G_P = c_P D` (RI-J) `+ Σ_s 2 α_s X_s Γ^s_P
//! X_sᵀ` (RI-K), one auxiliary shell per rayon task.
//!
//! ## Contraction layout
//!
//! libint returns, per `(aux shell P, AO shell s1 ≥ s2)`, nine contiguous
//! components `[∂_P xyz, ∂_1 xyz, ∂_2 xyz]` of `np·n1·n2` values each, `(p, i,
//! j)` row-major. The weight of an element is `G[p,μ,ν] + G[p,ν,μ]` when `s1 ≠
//! s2` (the integral is symmetric in `μν`, only `s1 ≥ s2` is computed) and
//! `G[p,μ,ν]` when `s1 = s2` (libint returns the full square block).
//!
//! Each block's weights are formed ONCE into a contiguous buffer in libint's
//! `(p, i, j)` order (the same `G[p,μ,ν] + G[p,ν,μ]` sum, operands in the same
//! order, so every weight is bit-identical to the one the old loop formed;
//! the transposed operand is read from `n2` short row segments that stay in
//! L1 across the block), and the nine components are then nine contiguous dot
//! products. The old loop re-formed each weight inside the element loop and
//! scattered 27 bounds-checked updates per element: 1.5 ns per multiply-add,
//! ~25x below the FMA peak (P2 profile, n-dodecane/def2-TZVP). A first
//! version symmetrised the whole `(np, nao, nao)` slab in place instead; that
//! full extra pass over the slab (most of whose pairs no non-empty block
//! reads) cost about 40 % of the saving (MEASURED n-dodecane/def2-TZVP B3LYP,
//! `dfg:3c-deriv-pass` medians per evaluation: old 4.00 s, in-place
//! symmetrisation 3.77 s, per-block weights 3.62 s).
//!
//! ## What changes numerically
//!
//! Only the summation ORDER of the gradient entries. Every product
//! `fl(w · ∂)` is the same floating-point number as before (same weight, same
//! integral, no FMA contraction), and terms with `w = 0` that the old loop
//! skipped contribute an exact zero. For a gradient entry of one aux shell
//! fed by `n` terms `t_k`, any summation order computes `Σ t_k` with error at
//! most `γ_{n−1} Σ |t_k|` (`γ_m = m u / (1 − m u)`, Higham, *Accuracy and
//! Stability*, §4.2, any binary tree has depth ≤ `n − 1`), so the old and new
//! entries differ by at most `2 γ_n Σ |t_k|`. The per-shell partials are still
//! folded serially in shell order, so the result stays independent of the
//! thread count. `contraction_matches_reference_*` in the tests below check
//! that bound against the old loop (kept verbatim under `cfg(test)`).

use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ndarray::{Array1, Array2, Array3, Axis};
use rayon::prelude::*;

/// The weights entering the slab `G_P`: the RI-J coefficients `c = V⁻¹ d`
/// with the J density, and per exchange factor `(X_s, α_s)` its `Γ^s`
/// `(naux, m_s²)`.
pub(super) struct SlabWeights<'a> {
    /// `(c, D)` for RI-J, `None` without J.
    pub(super) j: Option<(&'a Array1<f64>, &'a Array2<f64>)>,
    /// `(X_s, α_s)` per exchange channel.
    pub(super) factors: &'a [(Array2<f64>, f64)],
    /// `Γ^s`, one per entry of `factors`.
    pub(super) gammas: &'a [Array2<f64>],
}

/// `Σ_{P,μν} G_P,μν ∂(P|μν)` over every auxiliary shell, `(natoms, 3)`.
///
/// Deterministic: one task per aux shell, collected in shell order and folded
/// serially.
pub(super) fn three_center_deriv_pass(
    natoms: usize,
    (obs, dfbs, op): (&PreparedBasis, &PreparedBasis, Operator),
    w: &SlabWeights<'_>,
) -> Result<Array2<f64>, FerricError> {
    let n = obs.nbasis();
    let nsh_df = dfbs.nshells();
    let dims_df = dfbs.shell_dims();
    let max_np = dims_df.iter().copied().max().unwrap_or(0);
    let max_no = obs.shell_dims().iter().copied().max().unwrap_or(0);
    let workers = rayon::current_num_threads().max(1);
    let _slab_charge = ferric_core::memory::pool::reserve_global(
        "DF-JK gradient per-worker 3c weight slabs",
        // Per worker: the (np, nao, nao) slab and one block's gathered weights.
        workers
            .saturating_mul(max_np)
            .saturating_mul(n.saturating_mul(n).saturating_add(max_no * max_no))
            .saturating_mul(8),
    )?;
    #[cfg(test)]
    let check = reference::capturing();

    // Surface an engine-construction error serially before the fan-out.
    Engine::new_3center_deriv(op, obs, dfbs, 1e-14)?;
    let partials: Vec<ShellOut> = (0..nsh_df)
        .into_par_iter()
        .map_init(
            || {
                Engine::new_3center_deriv(op, obs, dfbs, 1e-14)
                    .expect("3-center deriv engine (pre-validated)")
            },
            |eng, sp| {
                let slab = weight_slab(dfbs, sp, n, w);
                let grad = contract_shell(eng, (obs, dfbs), sp, natoms, &slab);
                ShellOut {
                    #[cfg(test)]
                    check: check.then(|| {
                        reference::check_shell(eng, (obs, dfbs), sp, natoms, &slab, &grad)
                    }),
                    grad,
                }
            },
        )
        .collect();
    let mut grad = Array2::<f64>::zeros((natoms, 3));
    for p in &partials {
        grad += &p.grad;
    }
    #[cfg(test)]
    if check {
        reference::record(op, natoms, n, dfbs.nbasis(), &grad, &partials);
    }
    Ok(grad)
}

/// One aux shell's contribution (and, in tests, its check against the old loop).
struct ShellOut {
    grad: Array2<f64>,
    #[cfg(test)]
    check: Option<reference::ShellCheck>,
}

/// `G[p, μ, ν] = ∂E/∂(P|μν)` for the functions `p` of aux shell `sp`.
fn weight_slab(dfbs: &PreparedBasis, sp: usize, n: usize, w: &SlabWeights<'_>) -> Array3<f64> {
    let np = dfbs.shell_dims()[sp];
    let pf0 = dfbs.shell_offsets()[sp];
    let mut slab = Array3::<f64>::zeros((np, n, n));
    for p in 0..np {
        let pf = pf0 + p;
        let mut g = slab.index_axis_mut(Axis(0), p);
        if let Some((c, dj)) = w.j {
            g.scaled_add(c[pf], dj);
        }
        for ((x, alpha), gam) in w.factors.iter().zip(w.gammas.iter()) {
            let m = x.ncols();
            let gp = Array2::from_shape_fn((m, m), |(i, j)| gam[(pf, i * m + j)]);
            let xg = x.dot(&gp);
            let contrib = xg.dot(&x.t());
            g.scaled_add(2.0 * alpha, &contrib);
        }
    }
    slab
}

/// Shape and atoms of one `(P, s1, s2)` libint block.
#[derive(Clone, Copy)]
struct DerivBlock {
    np: usize,
    off1: usize,
    n1: usize,
    off2: usize,
    n2: usize,
    /// Atoms of the aux shell, `s1` and `s2`: the owners of components 0-2,
    /// 3-5 and 6-8.
    atoms: [usize; 3],
}

impl DerivBlock {
    fn new(obs: &PreparedBasis, dfbs: &PreparedBasis, sp: usize, s1: usize, s2: usize) -> Self {
        Self {
            np: dfbs.shell_dims()[sp],
            off1: obs.shell_offsets()[s1],
            n1: obs.shell_dims()[s1],
            off2: obs.shell_offsets()[s2],
            n2: obs.shell_dims()[s2],
            atoms: [
                dfbs.shell_to_atom()[sp],
                obs.shell_to_atom()[s1],
                obs.shell_to_atom()[s2],
            ],
        }
    }

    fn size(&self) -> usize {
        self.np * self.n1 * self.n2
    }
}

/// Contract every `(s1 ≥ s2)` block of aux shell `sp` with the slab `g`;
/// the shell's `(natoms, 3)` partial.
fn contract_shell(
    eng: &mut Engine,
    (obs, dfbs): (&PreparedBasis, &PreparedBasis),
    sp: usize,
    natoms: usize,
    g: &Array3<f64>,
) -> Array2<f64> {
    let n = obs.nbasis();
    let g = g.as_slice().expect("weight slab is standard layout");
    let mut local = Array2::<f64>::zeros((natoms, 3));
    let mut wbuf: Vec<f64> = Vec::new();
    for s1 in 0..obs.nshells() {
        for s2 in 0..=s1 {
            let Some(deriv) = eng.compute_eri3_deriv(obs, dfbs, sp, s1, s2) else {
                continue;
            };
            let b = DerivBlock::new(obs, dfbs, sp, s1, s2);
            block_weights(g, n, &b, s1 != s2, &mut wbuf);
            contract_block(&wbuf, &b, deriv, &mut local);
        }
    }
    local
}

/// Block `b`'s weights in libint's `(p, i, j)` order from the slab `g`
/// (`(np, n, n)` flat): `G[p,μ,ν] + G[p,ν,μ]` when `sym` (`s1 ≠ s2`), else
/// `G[p,μ,ν]` — the old loop's `gval`, operands in the same order.
fn block_weights(g: &[f64], n: usize, b: &DerivBlock, sym: bool, wbuf: &mut Vec<f64>) {
    wbuf.clear();
    for p in 0..b.np {
        let gp = &g[p * n * n..(p + 1) * n * n];
        for i in 0..b.n1 {
            let mu = b.off1 + i;
            let row = &gp[mu * n + b.off2..mu * n + b.off2 + b.n2];
            if sym {
                // The transposed operand: column μ of rows ν ∈ s2.
                wbuf.extend(
                    row.iter()
                        .enumerate()
                        .map(|(j, &a)| a + gp[(b.off2 + j) * n + mu]),
                );
            } else {
                wbuf.extend_from_slice(row);
            }
        }
    }
}

/// Add block `b`'s nine component dot products with its weights `w` to
/// their atoms (components `[∂_P xyz, ∂_1 xyz, ∂_2 xyz]`).
fn contract_block(w: &[f64], b: &DerivBlock, deriv: &[f64], local: &mut Array2<f64>) {
    let bs = b.size();
    for (k, comp) in deriv[..9 * bs].chunks_exact(bs).enumerate() {
        local[(b.atoms[k / 3], k % 3)] += dot(w, comp);
    }
}

/// `Σ a_k b_k` with four independent partial sums (vectorisable, fixed order).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let (ca, ra) = a.as_chunks::<4>();
    let (cb, rb) = b.as_chunks::<4>();
    let mut acc = [0.0f64; 4];
    for (x, y) in ca.iter().zip(cb) {
        for l in 0..4 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

#[cfg(test)]
pub(crate) mod reference {
    //! The pre-2026-11-11 contraction, verbatim, and the check of the new
    //! one against it at the derived bound (module doc).

    use super::*;
    use std::cell::RefCell;

    /// One `dfg:3c-deriv-pass` checked against the old loop.
    #[derive(Clone, Debug)]
    pub(crate) struct PassCheck {
        pub(crate) op: Operator,
        pub(crate) natoms: usize,
        pub(crate) nao: usize,
        pub(crate) naux: usize,
        /// max over shells and entries of `|new − old| / bound` (`bound = 2
        /// γ_n Σ|t|`; an entry with bound 0 counts as ∞ unless it is equal).
        pub(crate) max_ratio: f64,
        /// max over shells and entries of `|new − old|`.
        pub(crate) max_diff: f64,
        /// max over shells and entries of the bound.
        pub(crate) max_bound: f64,
        /// `max |Σ_shells new − Σ_shells old|` (serial folds of both).
        pub(crate) total_diff: f64,
        /// `max |Σ_shells new|`.
        pub(crate) total_max: f64,
        /// non-empty libint blocks contracted.
        pub(crate) blocks: usize,
    }

    /// One aux shell's comparison, with the old loop's partial.
    pub(crate) struct ShellCheck {
        ratio: f64,
        diff: f64,
        bound: f64,
        blocks: usize,
        old: Array2<f64>,
    }

    thread_local! {
        /// Capture of THIS thread's passes. Thread-local, so a concurrently
        /// running lib test's DF gradient (another thread) is never checked
        /// or recorded here. The flag is read and the record written on the
        /// thread that calls [`three_center_deriv_pass`]; the per-shell checks
        /// on rayon workers receive the flag by value. A gradient that ran the
        /// pass on another thread would record nothing, which the tests'
        /// "no 3c-deriv pass ran" assert turns into a failure, never a pass.
        static CAPTURE: RefCell<Option<Vec<PassCheck>>> = const { RefCell::new(None) };
    }

    pub(super) fn capturing() -> bool {
        CAPTURE.with(|c| c.borrow().is_some())
    }

    /// Run `f` with capture on for this thread; return every pass it checked.
    pub(crate) fn capture<R>(f: impl FnOnce() -> R) -> (R, Vec<PassCheck>) {
        CAPTURE.with(|c| *c.borrow_mut() = Some(Vec::new()));
        let r = f();
        let got = CAPTURE.with(|c| c.borrow_mut().take().unwrap_or_default());
        (r, got)
    }

    pub(super) fn record(
        op: Operator,
        natoms: usize,
        nao: usize,
        naux: usize,
        grad: &Array2<f64>,
        partials: &[ShellOut],
    ) {
        let checks: Vec<&ShellCheck> = partials.iter().filter_map(|p| p.check.as_ref()).collect();
        assert_eq!(checks.len(), partials.len(), "every shell is checked");
        let mut old_total = Array2::<f64>::zeros((natoms, 3));
        for c in &checks {
            old_total += &c.old;
        }
        let pc = PassCheck {
            op,
            natoms,
            nao,
            naux,
            max_ratio: checks.iter().map(|c| c.ratio).fold(0.0, f64::max),
            max_diff: checks.iter().map(|c| c.diff).fold(0.0, f64::max),
            max_bound: checks.iter().map(|c| c.bound).fold(0.0, f64::max),
            total_diff: (grad - &old_total)
                .iter()
                .fold(0.0f64, |m, v| m.max(v.abs())),
            total_max: grad.iter().fold(0.0f64, |m, v| m.max(v.abs())),
            blocks: checks.iter().map(|c| c.blocks).sum(),
        };
        CAPTURE.with(|c| {
            if let Some(v) = c.borrow_mut().as_mut() {
                v.push(pc);
            }
        });
    }

    /// The old per-block loop, VERBATIM (2026-11-11, main 37880cd15) apart
    /// from taking its loop-invariant locals as arguments.
    #[allow(clippy::too_many_arguments)]
    fn contract_block_old(
        slab: &Array3<f64>,
        deriv: &[f64],
        (np, n1, n2): (usize, usize, usize),
        (offs1, offs2): (usize, usize),
        sym12: bool,
        (atom_p, atom_1, atom_2): (usize, usize, usize),
        local: &mut Array2<f64>,
    ) {
        let block_sz = np * n1 * n2;
        for p in 0..np {
            for i in 0..n1 {
                let mu = offs1 + i;
                for j in 0..n2 {
                    let nu = offs2 + j;
                    let idx = (p * n1 + i) * n2 + j;
                    let gval = if sym12 {
                        slab[(p, mu, nu)] + slab[(p, nu, mu)]
                    } else {
                        slab[(p, mu, nu)]
                    };
                    if gval == 0.0 {
                        continue;
                    }
                    // 9 blocks: [dP_xyz, d1_xyz, d2_xyz].
                    for coord in 0..3 {
                        local[(atom_p, coord)] += gval * deriv[coord * block_sz + idx];
                        local[(atom_1, coord)] += gval * deriv[(3 + coord) * block_sz + idx];
                        local[(atom_2, coord)] += gval * deriv[(6 + coord) * block_sz + idx];
                    }
                }
            }
        }
    }

    /// `Σ|t|` and the term count per gradient entry for one block (every
    /// element counted, zero weights included: a superset of both orders'
    /// leaves).
    #[allow(clippy::too_many_arguments)]
    fn abs_terms_block(
        slab: &Array3<f64>,
        deriv: &[f64],
        (np, n1, n2): (usize, usize, usize),
        (offs1, offs2): (usize, usize),
        sym12: bool,
        atoms: [usize; 3],
        abs: &mut Array2<f64>,
        cnt: &mut Array2<f64>,
    ) {
        let bs = np * n1 * n2;
        for p in 0..np {
            for i in 0..n1 {
                let mu = offs1 + i;
                for j in 0..n2 {
                    let nu = offs2 + j;
                    let idx = (p * n1 + i) * n2 + j;
                    let gval = if sym12 {
                        slab[(p, mu, nu)] + slab[(p, nu, mu)]
                    } else {
                        slab[(p, mu, nu)]
                    };
                    for k in 0..9 {
                        let e = (atoms[k / 3], k % 3);
                        abs[e] += (gval * deriv[k * bs + idx]).abs();
                        cnt[e] += 1.0;
                    }
                }
            }
        }
    }

    /// Recompute aux shell `sp` with the old loop on the UNSYMMETRISED slab
    /// and compare with `new` at `2 γ_n Σ|t|` per entry.
    pub(super) fn check_shell(
        eng: &mut Engine,
        (obs, dfbs): (&PreparedBasis, &PreparedBasis),
        sp: usize,
        natoms: usize,
        slab: &Array3<f64>,
        new: &Array2<f64>,
    ) -> ShellCheck {
        let mut old = Array2::<f64>::zeros((natoms, 3));
        let mut abs = Array2::<f64>::zeros((natoms, 3));
        let mut cnt = Array2::<f64>::zeros((natoms, 3));
        let mut blocks = 0;
        // Block geometry and atoms straight from the bases, as the old loop
        // read them: independent of the new path's `DerivBlock`.
        let (dims_obs, offs_obs) = (obs.shell_dims(), obs.shell_offsets());
        let (sh2at_obs, sh2at_df) = (obs.shell_to_atom(), dfbs.shell_to_atom());
        let np = dfbs.shell_dims()[sp];
        let atom_p = sh2at_df[sp];
        for s1 in 0..obs.nshells() {
            for s2 in 0..=s1 {
                let Some(deriv) = eng.compute_eri3_deriv(obs, dfbs, sp, s1, s2) else {
                    continue;
                };
                blocks += 1;
                let shape = (np, dims_obs[s1], dims_obs[s2]);
                let offs = (offs_obs[s1], offs_obs[s2]);
                let (atom_1, atom_2) = (sh2at_obs[s1], sh2at_obs[s2]);
                let sym12 = s1 != s2;
                contract_block_old(
                    slab,
                    deriv,
                    shape,
                    offs,
                    sym12,
                    (atom_p, atom_1, atom_2),
                    &mut old,
                );
                let atoms = [atom_p, atom_1, atom_2];
                abs_terms_block(slab, deriv, shape, offs, sym12, atoms, &mut abs, &mut cnt);
            }
        }
        let u = f64::EPSILON / 2.0;
        let (mut ratio, mut diff, mut bound) = (0.0f64, 0.0f64, 0.0f64);
        for ((e, &o), (&a, &c)) in old.indexed_iter().zip(abs.iter().zip(cnt.iter())) {
            let d = (new[e] - o).abs();
            let nu = c * u;
            let b = 2.0 * nu / (1.0 - nu) * a;
            let r = if d == 0.0 {
                0.0
            } else if b > 0.0 {
                d / b
            } else {
                f64::INFINITY
            };
            ratio = ratio.max(r);
            diff = diff.max(d);
            bound = bound.max(b);
        }
        ShellCheck {
            ratio,
            diff,
            bound,
            blocks,
            old,
        }
    }
}

#[cfg(test)]
mod tests {
    //! The contiguous contraction against the old strided loop, per aux shell,
    //! at the derived bound `2 γ_n Σ|t|` (module doc), on real converged
    //! ferric densities through the real gradient entry points (so the slab
    //! carries exactly the J / K / RSH / truncated-metric weights each route
    //! builds).
    //!
    //! MUTATIONS (2026-11-11, each applied alone, run, reverted):
    //! * `block_weights` with `sym` forced false (unsymmetrised `G`, current
    //!   version): all 6 tests FAIL, max |new − old| 0.13-0.70, |Δ|/bound
    //!   2e12-4e13.
    //! * component 4 (`∂_1 y`) credited to `∂_1 x` in `contract_block`:
    //!   all 6 tests FAIL, max |new − old| 0.13-1.2, |Δ|/bound 1e13-7e14.
    //! * atoms of `s1` and `s2` swapped in `DerivBlock::new` (the reference
    //!   reads them from the bases itself): all 6 tests FAIL, max |new − old|
    //!   0.91-2.3, |Δ|/bound 1e13-1.3e14.
    //!
    //! Unmutated (release, 2026-11-11): max |new − old| 7e-18..6.1e-14,
    //! |Δ|/bound ≤ 0.048 on every pass.
    use super::reference::{capture, PassCheck};
    use crate::df_gradient::DFK_GRADIENT_NOISE_FLOOR;
    use crate::gradient::rhf_gradient;
    use crate::ks_gradient::{ks_gradient_closed, ks_gradient_roks, ks_gradient_uks};
    use crate::result::ScfResult;
    use crate::rhf::{solve_rhf, RhfConfig};
    use crate::rohf::solve_rohf;
    use crate::screening::SchwarzBounds;
    use crate::uhf::solve_uhf;
    use ferric_core::basis::{self, BasisSet};
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;

    const JKFIT: &str = "def2-universal-jkfit";
    /// Distorted water (Å): no symmetry-zero gradient components.
    const WATER: &str = "3\nH2O\nO 0 0 0\nH 0.1 0.76 0.59\nH -0.05 -0.74 0.62\n";
    /// OH radical (Å), off-axis.
    const OH: &str = "2\nOH\nO 0 0 0\nH 0.05 0.1 0.97\n";
    /// NH2 radical (Å, ²B1), slightly distorted.
    const NH2: &str = "3\nNH2\nN 0 0 0\nH 0 0.80 0.62\nH 0.05 -0.81 0.60\n";

    #[derive(Clone, Copy, Debug)]
    enum Kind {
        Rhf,
        Uks,
        Roks,
    }

    struct Case<'a> {
        label: &'a str,
        mol: Molecule,
        basis: &'a str,
        kind: Kind,
        xc: Option<&'a str>,
        jk_aux: Option<&'a str>,
    }

    /// Converge ferric's own SCF, run the case's gradient with capture on and
    /// return every checked `dfg:3c-deriv-pass`.
    fn run(case: &Case<'_>) -> (ScfResult, Vec<PassCheck>) {
        let bs: BasisSet = basis::bundled(case.basis).unwrap();
        let mol = &case.mol;
        let prep = PreparedBasis::new(mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let cfg = RhfConfig {
            ri_j_precision: Some(crate::df_j::RiJPrecision::F64),
            xc: case.xc.map(Into::into),
            df_j_aux: case.jk_aux.map(Into::into),
            df_k_aux: case.jk_aux.map(Into::into),
            energy_conv: 1e-8,
            density_conv: 1e-8,
            max_iter: 300,
            ..Default::default()
        };
        let res = match case.kind {
            Kind::Rhf => solve_rhf(&ctx, mol, &prep, op, &bounds, &cfg),
            Kind::Uks => solve_uhf(&ctx, mol, &prep, &bounds, &cfg),
            Kind::Roks => solve_rohf(&ctx, mol, &prep, op, &bounds, &cfg),
        }
        .unwrap();
        assert!(res.converged, "{}: SCF did not converge", case.label);
        assert!(
            res.df_jk.is_some(),
            "{}: SCF recorded no DF route",
            case.label
        );
        let (g, checks) = capture(|| match (case.kind, case.xc) {
            (Kind::Rhf, None) => rhf_gradient(mol, &prep, op, &bounds, &res, None),
            (Kind::Rhf, Some(x)) => ks_gradient_closed(mol, &prep, &bs, op, &bounds, x, &res, None),
            (Kind::Uks, Some(x)) => ks_gradient_uks(mol, &prep, &bs, op, &bounds, x, &res, None),
            (Kind::Roks, Some(x)) => ks_gradient_roks(mol, &prep, &bs, op, &bounds, x, &res, None),
            (k, None) => panic!("{k:?} without a functional is not a case here"),
        });
        g.unwrap();
        let nao = prep.nbasis();
        let mine: Vec<PassCheck> = checks
            .into_iter()
            .filter(|c| c.nao == nao && c.natoms == mol.atoms.len())
            .collect();
        assert!(!mine.is_empty(), "{}: no 3c-deriv pass ran", case.label);
        for c in &mine {
            eprintln!(
                "{}: op {:?}(ω={}) naux {} blocks {}: max|new−old| {:.3e}, max bound {:.3e}, \
                 max ratio {:.3e}; pass total |Δ| {:.3e} (|g| ≤ {:.3e})",
                case.label,
                c.op.kind,
                c.op.omega,
                c.naux,
                c.blocks,
                c.max_diff,
                c.max_bound,
                c.max_ratio,
                c.total_diff,
                c.total_max
            );
            assert!(c.blocks > 0, "{}: no libint block contracted", case.label);
            assert!(
                c.max_ratio <= 1.0,
                "{}: op {:?}: new contraction differs from the old loop by {:.3e} \
                 (max |Δ|/bound {:.3e} > 1)",
                case.label,
                c.op.kind,
                c.max_diff,
                c.max_ratio
            );
        }
        (res, mine)
    }

    fn mol(xyz: &str, charge: i32, mult: usize) -> Molecule {
        Molecule::parse_xyz(xyz, charge, mult).unwrap()
    }

    fn has_op(checks: &[PassCheck], pred: impl Fn(&Operator) -> bool) -> bool {
        checks.iter().any(|c| pred(&c.op))
    }

    /// FAILS under either mutation in the module doc.
    #[test]
    fn contraction_matches_reference_water_rhf_rijk() {
        let (_, c) = run(&Case {
            label: "water/cc-pVDZ RHF RI-JK",
            mol: mol(WATER, 0, 1),
            basis: "cc-pvdz",
            kind: Kind::Rhf,
            xc: None,
            jk_aux: Some(JKFIT),
        });
        assert_eq!(c.len(), 1, "RI-J and RI-K share one pass");
    }

    // ~5 s in release; tiered because CI runs a debug build (B3LYP SCF on
    // 26 atoms plus the doubled libint pass of the check).
    #[test]
    #[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
    fn contraction_matches_reference_octane_b3lyp() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/molecules/alkane_8.xyz"
        );
        let _ = run(&Case {
            label: "n-octane/def2-SVP B3LYP",
            mol: Molecule::load_xyz(path).unwrap(),
            basis: "def2-svp",
            kind: Kind::Rhf,
            xc: Some("B3LYP"),
            jk_aux: Some(JKFIT),
        });
    }

    #[test]
    fn contraction_matches_reference_oh_uks_b3lyp() {
        let _ = run(&Case {
            label: "OH/cc-pVDZ UKS B3LYP",
            mol: mol(OH, 0, 2),
            basis: "cc-pvdz",
            kind: Kind::Uks,
            xc: Some("B3LYP"),
            jk_aux: Some(JKFIT),
        });
    }

    #[test]
    fn contraction_matches_reference_nh2_roks() {
        let _ = run(&Case {
            label: "NH2/cc-pVDZ ROKS B3LYP",
            mol: mol(NH2, 0, 2),
            basis: "cc-pvdz",
            kind: Kind::Roks,
            xc: Some("B3LYP"),
            jk_aux: Some(JKFIT),
        });
    }

    /// CAM-B3LYP: exchange through the erfc/erf DfK pair.
    #[test]
    fn contraction_matches_reference_rsh_pair() {
        let (_, c) = run(&Case {
            label: "water/cc-pVDZ CAM-B3LYP",
            mol: mol(WATER, 0, 1),
            basis: "cc-pvdz",
            kind: Kind::Rhf,
            xc: Some("HYB_GGA_XC_CAM_B3LYP"),
            jk_aux: None,
        });
        use ferric_integrals::operator::OperatorKind;
        let erf = |o: &Operator| matches!(o.kind, OperatorKind::ErfCoulomb);
        let erfc = |o: &Operator| matches!(o.kind, OperatorKind::ErfcCoulomb);
        assert!(
            has_op(&c, erf) && has_op(&c, erfc),
            "erf + erfc passes expected"
        );
    }

    /// wB97X (ω = 0.3) with the JK-fit basis: the erf metric has modes below
    /// `DFK_GRADIENT_NOISE_FLOOR`, so Γ comes from the truncated
    /// (Daleckii–Krein) path. Asserted, not assumed.
    #[test]
    fn contraction_matches_reference_truncated_erf_metric() {
        let (res, c) = run(&Case {
            label: "water/6-31G wB97X (truncated erf metric)",
            mol: mol(WATER, 0, 1),
            basis: "6-31g",
            kind: Kind::Rhf,
            xc: Some("HYB_GGA_XC_WB97X"),
            jk_aux: Some(JKFIT),
        });
        let (aux, omega) = res.df_jk.unwrap().rsh_k.expect("RSH route");
        let dfbs = crate::df_gradient::aux_basis(&mol(WATER, 0, 1), &aux).unwrap();
        let v =
            ferric_integrals::threeindex::coulomb_metric_2c(Operator::erf(omega), &dfbs).unwrap();
        let (lam, _) = ferric_core::linalg::eigh_dc(&v, ferric_core::linalg::Uplo::Upper).unwrap();
        let cut = lam
            .iter()
            .filter(|&&l| l < DFK_GRADIENT_NOISE_FLOOR)
            .count();
        eprintln!(
            "erf(ω={omega}) metric on {aux}: {cut} of {} modes cut",
            lam.len()
        );
        assert!(cut > 0, "the erf metric must have cut modes for this case");
        assert!(
            c.iter().any(|p| p.naux == lam.len()),
            "erf pass on that aux basis"
        );
    }
}
