//! Opt-in RS-GDF RANGE SPLIT (Gamma energy path): primitive-level
//! partition of the SR real-space sums into a kept part and a part moved to
//! G space. Rust port of `reference/pbc/pbc_gdf_split.py` (FINDINGS
//! "Iteration 23 (Python, RS-GDF range split)", grouping A, `twocall`).
//!
//! # Criterion (per primitive, λ = [`RangeSplit::lambda`])
//!
//! Every shell is split exactly by primitive exponent into a compact and a
//! smooth piece, `χ = χ^c + χ^s` (orbital primitive smooth iff
//! `a ≤ λω²/2`) and `X = X^c + X^s` (aux primitive smooth iff `α ≤ λω²`).
//! A pair splits as `χχ = [cc + cs + sc] + ss`.
//!
//! * **Moved to G space:** `(any pair | X^s)` and `(ss pair | X^c)`. At
//!   λ ≤ 1 every moved primitive combination has `1/p + 1/α ≥ 1/ω²`, so its
//!   FT decays at least as fast as `e^{−G²/4ω²}`, the factor that already
//!   sets the LR sphere `gcut = 2ω√ln(1/prec)`: NO new G vectors.
//! * **Kept in real space (SR erfc):** `(χ_i χ_j − χ_i^s χ_j^s | X^c)`,
//!   evaluated as TWO ordinary contracted libint calls per shell pair
//!   (`twocall`): `(χ_i, χ_j^c | X^c) + (χ_i^c, χ_j^s | X^c)`. No libint
//!   change; the prototype's `trim` (one call with the ss primitive pairs
//!   erased from the ket ShellPair) would visit 0.147× instead of 0.182× of
//!   today's triplets on diamond cc-pVDZ/cc-pvdz-ri, but needs a shim change.
//!   Each call has its own `pair_bound` over its own primitive pairs and the
//!   aux bound over the compact aux primitives.
//! * **Metric:** only `(P^c|Q^c)` stays in real space; the rest,
//!   `v_SR [X Xᴴ − X_c X_cᴴ]`, moves to G space.
//!
//! # G space (the half sphere, weight 2, G ≠ 0)
//!
//! ```text
//! J3 += (2/Ω) Σ_G Re[ P̄ (v_LR X + v_SR X_s) ] + (2/Ω) Σ_G v_SR Re[ P̄_ss X_c ]
//! J2 += (2/Ω) Σ_G Re[ (v_LR X + v_SR X_s)ᴴ X ] + (2/Ω) Σ_G v_SR Re[ X_cᴴ X_s ]
//! ```
//!
//! (`v_LR = 4π/G² e^{−G²/4ω²}`, `v_SR = 4π/G² − v_LR`; the J2 form is
//! `v_LR XXᴴ + v_SR (XXᴴ − X_c X_cᴴ)` without the cancellation.) When no
//! aux primitive is smooth the unchanged [`Stage::lr_accumulate`] runs
//! (weights on the pair side, bit for bit today's J2/J3).
//!
//! # G = 0 (grouping A, PySCF `gen_j3c_loader` vbar)
//!
//! The moved blocks take the full kernel at G ≠ 0 and NO G = 0 term; the
//! ONE subtract ([`subtract_g0`]) gets the KEPT inputs:
//! `J3 −= c0 (S − S_ss) q_cᵀ`, `J2 −= c0 q_c q_cᵀ`, with `S_ss` the lattice
//! overlap of the smooth orbital pieces from libint's 1e overlap (NOT the pair
//! FT at G = 0) and `q_c` the charges of the compact aux pieces. Nothing
//! moved means `S_ss = 0` and `q_c = q`, bit-identical to today.
//!
//! # libint normalisation of piece shells
//!
//! libint2 renormalises every contraction it is given to unit self-overlap,
//! so a piece handed to it is `χ^c / ‖χ^c‖`. The piece of the PARENT is
//! `f χ^c_libint` with `f = √(s(piece)/s(parent))`,
//! `s(X) = Σ_{pq∈X} c_p c_q (2√(a_p a_q)/(a_p + a_q))^{l+3/2}` over the raw
//! (unit-primitive) coefficients; every piece block is scaled by its shells'
//! `f`. A shell that is not split keeps its parent index and `f = 1` exactly.
//! The Rust-side FTs (`aux_ft_shells`, `pair_ft`) use the raw coefficients
//! directly and need no factor.
//!
//! # Metric guard
//!
//! A G = 0 bookkeeping mistake in J2 shows up as one LARGE negative metric
//! eigenvalue (prototype: −18.3 at ω = 1, −12.7 at ω = 1.2 on the exact-span
//! anchor) that the absolute `lindep` cut would otherwise silently drop. A
//! split build whose smallest metric eigenvalue is below
//! `−`[`RANGE_SPLIT_NEG_EIG_GUARD`] is an ERROR.
//!
//! # Scope
//!
//! Gamma energy, forces and stress (the derivative walks follow the same
//! partition: the child module `deriv`, FINDINGS "Iteration 26"), and the
//! k-point energy and forces ([`super::kpoint::KRsGdf`] and its `kderiv`;
//! the moved blocks at `K = G + q` with `v_SR(|G + q|)`: the child module
//! `ksplit`). No k-point stress. A derivative of a build made with a
//! [`RangeSplitMutant`] other than `Production` is refused (the mutants
//! exist for the Gamma energy anchors), and the k-point build takes
//! `Production` only.

use super::{
    aux_ft_shells, copy_pair_rows_mirrored, dot3, lr_gemms, lr_pair_ft_chunked, pack_pair_ft,
    pair_bound, segment_distance, subtract_g0, sum_counts_in_pair_order, sum_s2_counts,
    unordered_pairs, G0Handling, GShell, LrKernel, PackBufs, RsGdfConfig, SrBinning, Stage,
    ENGINE_PRECISION, LR_GEMM_ROW_BLOCK, SUB_AUX_FT, SUB_P_PACK, SUB_XY_PACK,
};
use crate::budget::{bytes_of, Ledger};
use crate::hcore::ONE_E_ENGINE_PRECISION;
use crate::lattice::Cell;
use crate::pair_ft::DEFAULT_PAIR_FT_THRESH;
use crate::timing::{PbcTimings, StageClock};
use ferric_core::basis::{BasisSet, Shell};
use ferric_core::mol::{Atom, Molecule};
use ferric_core::FerricError;
use ferric_integrals::ao_grid::LocatedShell;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::engine_pool::EnginePool;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Mutex;

mod deriv;
mod ksplit;
pub(crate) use deriv::{split_g0, SplitG0};
pub(super) use deriv::{LrForce, LrStrain};
pub(super) use ksplit::check_metric_guard;

/// The rigorous criterion (FINDINGS "Iteration 23": each of the two
/// conditions alone suffices at λ ≤ 1; do not spend the measured slack).
pub const DEFAULT_RANGE_SPLIT_LAMBDA: f64 = 1.0;

/// Smallest metric eigenvalue a range-split build accepts is
/// `−RANGE_SPLIT_NEG_EIG_GUARD` (absolute, like `lindep`).
///
/// Chosen from the prototype's numbers (FINDINGS "Iteration 23"): a correct
/// split's J2 differs from the unsplit one by ≤ 3.2e-11 at λ = 1 (1.0e-8 at
/// λ = 2, 1.0e-5 at the deliberately non-convergent λ = 4), and the unsplit
/// metric's own error is 1e-12..1e-10; a wrong G = 0 in the metric gave
/// −18.3 / −12.7. 1e-6 sits ≥ 4 decades above every legitimate λ ≤ 2
/// perturbation and 7 below the defect.
pub const RANGE_SPLIT_NEG_EIG_GUARD: f64 = 1e-6;

/// Production partition or a deliberately BROKEN variant (negative
/// controls for the split's anchors; an anchor that cannot tell them from
/// [`RangeSplitMutant::Production`] certifies nothing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeSplitMutant {
    /// The partition of the module doc.
    Production,
    /// MUTATION: J3's G = 0 subtract uses the FULL `S` and `q` (the moved
    /// blocks lose their G = 0 term; the metric stays correct).
    ThreeIndexFullG0,
    /// MUTATION: J2's G = 0 subtract uses the FULL `q` (J3 correct). Makes
    /// the metric strongly indefinite; the guard must fire.
    MetricFullG0,
    /// MUTATION: the SR walks keep EVERY block, so the moved blocks are
    /// counted in real space AND in G space.
    DoubleCount,
    /// MUTATION: every ORBITAL primitive is smooth (aux still at λ): moves
    /// compact pairs whose FT does not converge in the sphere.
    AllOrbitalSmooth,
}

/// Opt-in range split of an RS-GDF build ([`RsGdfConfig::range_split`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeSplit {
    /// Criterion scale: orbital primitive smooth iff `a ≤ λω²/2`, aux iff
    /// `α ≤ λω²`. `λ = 0` (or any λ below the smallest exponent) moves
    /// nothing; λ > 1 moves blocks whose FT need not converge in the sphere.
    pub lambda: f64,
    /// [`RangeSplitMutant::Production`] except in mutation tests.
    pub mutant: RangeSplitMutant,
}

impl RangeSplit {
    /// Production split at `lambda`.
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            mutant: RangeSplitMutant::Production,
        }
    }

    /// Orbital smooth threshold `λω²/2` (∞ for the all-orbital mutant).
    pub fn orbital_threshold(&self, omega: f64) -> f64 {
        match self.mutant {
            RangeSplitMutant::AllOrbitalSmooth => f64::INFINITY,
            _ => 0.5 * self.lambda * omega * omega,
        }
    }

    /// Aux smooth threshold `λω²`.
    pub fn aux_threshold(&self, omega: f64) -> f64 {
        self.lambda * omega * omega
    }

    /// Rejects a negative or non-finite `lambda` with [`FerricError::General`].
    fn validate(&self) -> Result<(), FerricError> {
        if !(self.lambda >= 0.0) || !self.lambda.is_finite() {
            return Err(FerricError::General(format!(
                "RsGdf range split: lambda must be finite and >= 0, got {}",
                self.lambda
            )));
        }
        Ok(())
    }
}

impl Default for RangeSplit {
    /// Production split at [`DEFAULT_RANGE_SPLIT_LAMBDA`].
    fn default() -> Self {
        Self::new(DEFAULT_RANGE_SPLIT_LAMBDA)
    }
}

/// SR walk counts of one RS-GDF configuration ([`sr_walk_counts`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SrWalkCounts {
    /// Shifted 3-centre calls the Gamma build makes (s2: each unordered shell
    /// pair once; = `RsGdfStats::n_sr3_triplets` of the build).
    pub n_sr3_triplets: usize,
    /// `n_sr3_triplets` in ordered-pair units (off-diagonal pairs × 2;
    /// = `RsGdfStats::n_sr3_triplets_ordered`).
    pub n_sr3_triplets_ordered: usize,
    /// The ORDERED walk's calls (every ordered pair in its own orientation):
    /// the pre-s2 `n_sr3_triplets`, and the unit of the Python prototype
    /// counters (`count_sr3_ferric`, `gc_count.py`). Unsplit it equals
    /// `n_sr3_triplets_ordered` whenever the screen decides `(i1, i2, L, T)`
    /// and `(i2, i1, −L, T − L)` alike (round-off at the radius aside); with
    /// a range split it also counts the costlier orientation.
    pub n_sr3_triplets_s1: usize,
    /// Shifted 2-centre calls (= `RsGdfStats::n_sr2_pairs`).
    pub n_sr2_pairs: usize,
}

// ---------------------------------------------------------------------------
// Pieces
// ---------------------------------------------------------------------------

/// `s(X)` of the module doc (self-overlap of a contraction over
/// unit-normalised primitives).
fn contraction_norm2(l: i32, exps: &[f64], coefs: &[f64]) -> f64 {
    let e = l as f64 + 1.5;
    let mut s = 0.0;
    for (a, ca) in exps.iter().zip(coefs) {
        for (b, cb) in exps.iter().zip(coefs) {
            s += ca * cb * (2.0 * (a * b).sqrt() / (a + b)).powf(e);
        }
    }
    s
}

/// The primitive subset `keep` of a raw shell: the libint input and its
/// normalisation factor `f` (module doc). `None` if the subset is empty or
/// carries only zero coefficients (it contributes exactly 0).
fn raw_piece(
    l: i32,
    pure: bool,
    exps: &[f64],
    coefs: &[f64],
    keep: &[bool],
) -> Option<(Shell, f64)> {
    let (pe, pc): (Vec<f64>, Vec<f64>) = exps
        .iter()
        .zip(coefs)
        .zip(keep)
        .filter(|(_, &k)| k)
        .map(|((&a, &c), _)| (a, c))
        .unzip();
    if pe.is_empty() {
        return None;
    }
    let s_pc = contraction_norm2(l, &pe, &pc);
    let s_par = contraction_norm2(l, exps, coefs);
    if !(s_pc > 0.0) || !(s_par > 0.0) {
        return None;
    }
    let shell = Shell {
        l,
        pure,
        exponents: pe,
        coefficients: pc,
    };
    Some((shell, (s_pc / s_par).sqrt()))
}

/// The Rust shell data of `parent` restricted to the primitives `keep`
/// (same centre, AO offset and function count; bounds recomputed exactly as
/// `gshells` computes them, so an all-true subset is bitwise the parent).
fn gshell_subset(parent: &GShell, keep: &[bool]) -> GShell {
    let (exps, coefs): (Vec<f64>, Vec<f64>) = parent
        .exps
        .iter()
        .zip(&parent.coefs)
        .zip(keep)
        .filter(|(_, &k)| k)
        .map(|((&a, &c), _)| (a, c))
        .unzip();
    let l = parent.l;
    let qbound = exps
        .iter()
        .zip(&coefs)
        .map(|(&a, &c)| c.abs() * (PI / a).powf(1.5) * (1.0 + a.powf(-0.5)).powi(l as i32))
        .fold(0.0_f64, f64::max);
    GShell {
        l,
        pure: parent.pure,
        center: parent.center,
        amin: exps.iter().copied().fold(f64::INFINITY, f64::min),
        amax: exps.iter().copied().fold(0.0_f64, f64::max),
        exps,
        coefs,
        nfun: parent.nfun,
        off: parent.off,
        qbound,
    }
}

/// One basis (orbital or aux) partitioned by primitive exponent, with the
/// COMBINED libint basis the SR walks call into: the parent shells first
/// (index = parent index, `f = 1`), then one extra shell per compact /
/// smooth piece of every MIXED parent shell. Each combined shell sits on
/// its own ghost site at the parent's centre.
struct Side {
    /// Combined libint basis (parents, then pieces).
    x: PreparedBasis,
    /// Rust data of every combined shell (parent AO offsets).
    xsh: Vec<GShell>,
    /// libint normalisation factor of every combined shell (1 for parents).
    scale: Vec<f64>,
    /// Parent shell → its compact piece in `x` (`None`: all smooth).
    compact: Vec<Option<usize>>,
    /// Parent shell → its smooth piece in `x` (`None`: all compact).
    smooth: Vec<Option<usize>>,
    /// Rust data of every compact / smooth piece (parent order; for FTs).
    c_sh: Vec<GShell>,
    s_sh: Vec<GShell>,
    /// Primitive counts (all, smooth).
    n_prims: usize,
    n_smooth_prims: usize,
}

/// A libint basis of `shells`, one ghost site per shell at `centers[k]`.
fn combined_basis(shells: Vec<Shell>, centers: &[[f64; 3]]) -> Result<PreparedBasis, FerricError> {
    let mut map = HashMap::new();
    let mut atoms = Vec::with_capacity(shells.len());
    for (k, (sh, c)) in shells.into_iter().zip(centers).enumerate() {
        let z = i32::try_from(k + 1)
            .map_err(|_| FerricError::General("RsGdf range split: too many piece shells".into()))?;
        atoms.push(Atom {
            symbol: "X".into(),
            z,
            x: c[0],
            y: c[1],
            zpos: c[2],
            ghost: true,
            n_core_ecp: 0,
        });
        map.insert(z, vec![sh]);
    }
    let mol = Molecule {
        atoms,
        charge: 0,
        multiplicity: 1,
    };
    let bs = BasisSet {
        name: "rsgdf range-split pieces".into(),
        shells: map,
        ecps: HashMap::new(),
    };
    PreparedBasis::new(&mol, &bs)
}

/// Owned copy of a located shell's `l`, purity, exponents and coefficients.
fn raw_shell(s: &LocatedShell<'_>) -> Shell {
    Shell {
        l: s.l,
        pure: s.pure,
        exponents: s.exponents.to_vec(),
        coefficients: s.coefficients.to_vec(),
    }
}

impl Side {
    /// Partition `prep` (Rust data `parents`, from `gshells`) at the
    /// smooth threshold `thresh` (primitive smooth iff exponent ≤ thresh).
    fn new(prep: &PreparedBasis, parents: &[GShell], thresh: f64) -> Result<Self, FerricError> {
        let located = prep.located_shells();
        let nsh = parents.len();
        let mut shells: Vec<Shell> = located.iter().map(raw_shell).collect();
        let mut centers: Vec<[f64; 3]> = located.iter().map(|s| s.center).collect();
        let mut xsh: Vec<GShell> = parents.to_vec();
        let mut scale = vec![1.0; nsh];
        let mut compact = vec![None; nsh];
        let mut smooth = vec![None; nsh];
        let (mut c_sh, mut s_sh) = (Vec::new(), Vec::new());
        let (mut n_prims, mut n_smooth_prims) = (0usize, 0usize);
        for (i, (ls, g)) in located.iter().zip(parents).enumerate() {
            let flags: Vec<bool> = ls.exponents.iter().map(|&a| a <= thresh).collect();
            let ns = flags.iter().filter(|&&f| f).count();
            n_prims += flags.len();
            n_smooth_prims += ns;
            if ns == 0 {
                compact[i] = Some(i);
                c_sh.push(g.clone());
                continue;
            }
            if ns == flags.len() {
                smooth[i] = Some(i);
                s_sh.push(g.clone());
                continue;
            }
            for want_smooth in [false, true] {
                let keep: Vec<bool> = flags.iter().map(|&f| f == want_smooth).collect();
                let Some((sh, f)) = raw_piece(ls.l, ls.pure, ls.exponents, ls.coefficients, &keep)
                else {
                    continue;
                };
                let piece = gshell_subset(g, &keep);
                let x = shells.len();
                shells.push(sh);
                centers.push(ls.center);
                scale.push(f);
                xsh.push(piece.clone());
                if want_smooth {
                    smooth[i] = Some(x);
                    s_sh.push(piece);
                } else {
                    compact[i] = Some(x);
                    c_sh.push(piece);
                }
            }
        }
        let x = combined_basis(shells, &centers)?;
        if x.nshells() != xsh.len() {
            return Err(FerricError::General(format!(
                "RsGdf range split: combined basis has {} shells, expected {}",
                x.nshells(),
                xsh.len()
            )));
        }
        for (k, g) in xsh.iter().enumerate() {
            if x.shell_dims()[k] != g.nfun {
                return Err(FerricError::General(format!(
                    "RsGdf range split: piece shell {k} has {} functions in libint, {} expected",
                    x.shell_dims()[k],
                    g.nfun
                )));
            }
        }
        Ok(Self {
            x,
            xsh,
            scale,
            compact,
            smooth,
            c_sh,
            s_sh,
            n_prims,
            n_smooth_prims,
        })
    }
}

/// The smooth ORBITAL pieces as a basis on the cell's atoms (for the
/// lattice pair FT `P_ss` and the overlap `S_ss`), and where its AOs sit in
/// the parent AO order.
struct SmoothObs {
    prep: PreparedBasis,
    /// Smooth AO → parent AO.
    ao_map: Vec<usize>,
    /// libint normalisation factor per smooth shell.
    scale: Vec<f64>,
}

/// Smooth piece of a raw element-template shell at `thresh`.
fn smooth_template(sh: &Shell, thresh: f64) -> Option<(Shell, f64)> {
    let keep: Vec<bool> = sh.exponents.iter().map(|&a| a <= thresh).collect();
    raw_piece(sh.l, sh.pure, &sh.exponents, &sh.coefficients, &keep)
}

/// [`SmoothObs`] of `obs` (built from `cell.mol()` and `obs.basis_set()`,
/// cross-checked shell by shell); `None` when no orbital primitive is smooth.
fn smooth_obs(
    cell: &Cell,
    obs: &PreparedBasis,
    thresh: f64,
) -> Result<Option<SmoothObs>, FerricError> {
    let bs = obs.basis_set();
    let parents = obs.located_shells();
    let (offs, dims) = (obs.shell_offsets(), obs.shell_dims());
    let mut ao_map = Vec::new();
    let mut scale = Vec::new();
    let mut k = 0usize;
    for atom in &cell.mol().atoms {
        let tmpls = bs.for_element(atom.z).ok_or_else(|| {
            FerricError::Basis(format!(
                "RsGdf range split: no orbital shells for Z = {} in {:?}",
                atom.z, bs.name
            ))
        })?;
        for sh in tmpls {
            let same = parents
                .get(k)
                .is_some_and(|p| p.l == sh.l && p.exponents == sh.exponents.as_slice());
            if !same {
                return Err(FerricError::General(format!(
                    "RsGdf range split: orbital shell {k} does not match cell.mol() + its BasisSet \
                     (build the orbital PreparedBasis from cell.mol())"
                )));
            }
            if let Some((_, f)) = smooth_template(sh, thresh) {
                ao_map.extend(offs[k]..offs[k] + dims[k]);
                scale.push(f);
            }
            k += 1;
        }
    }
    if k != obs.nshells() {
        return Err(FerricError::General(format!(
            "RsGdf range split: cell.mol() + BasisSet give {k} orbital shells, the basis has {}",
            obs.nshells()
        )));
    }
    if scale.is_empty() {
        return Ok(None);
    }
    let shells: HashMap<i32, Vec<Shell>> = bs
        .shells
        .iter()
        .map(|(z, v)| {
            let s: Vec<Shell> = v
                .iter()
                .filter_map(|sh| smooth_template(sh, thresh).map(|(p, _)| p))
                .collect();
            (*z, s)
        })
        .collect();
    let sbs = BasisSet {
        name: format!("{} (smooth pieces)", bs.name),
        shells,
        ecps: bs.ecps.clone(),
    };
    let prep = PreparedBasis::new(cell.mol(), &sbs)?;
    if prep.nbasis() != ao_map.len() || prep.nshells() != scale.len() {
        return Err(FerricError::General(format!(
            "RsGdf range split: smooth orbital basis has {} AOs / {} shells, expected {} / {}",
            prep.nbasis(),
            prep.nshells(),
            ao_map.len(),
            scale.len()
        )));
    }
    Ok(Some(SmoothObs {
        prep,
        ao_map,
        scale,
    }))
}

// ---------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------

/// Everything a range-split build needs beyond the [`Stage`].
pub(super) struct SplitPlan {
    rs: RangeSplit,
    obs: Side,
    aux: Side,
    smooth_obs: Option<SmoothObs>,
    /// `S_ss` (parent AO order) over the build's pair images; `None` when
    /// no orbital primitive is smooth (then `S_ss = 0`).
    s_ss: Option<Array2<f64>>,
}

/// The loop invariants of the parallel split SR 3-centre sum.
struct Sr3Ctx<'a> {
    pool: &'a EnginePool,
    images: &'a [[f64; 3]],
    /// `r_L` of each image (same order as `images`).
    l_bin: &'a [usize],
    /// Reciprocal lattice (residue of `T`).
    recip: [[f64; 3]; 3],
    bins: SrBinning,
    global: f64,
    /// `R_L · R_T` zeroed `(nao², naux)` bins every task copies into.
    out: &'a Mutex<Vec<Array2<f64>>>,
}

/// The loop invariants of the parallel split SR metric.
struct Sr2Ctx<'a> {
    pool: &'a EnginePool,
    recip: [[f64; 3]; 3],
    mod_t: [usize; 3],
    global: f64,
    /// `R_T` zeroed `(naux, naux)` bins every task copies into.
    out: &'a Mutex<Vec<Array2<f64>>>,
}

impl SplitPlan {
    /// The plan of `cfg.range_split` (`None` without one), with its
    /// resident buffers (`S_ss`, `S − S_ss`, the `(ss pair | X_c)`
    /// accumulator) reserved on `ledger` and `S_ss` formed over the pair
    /// `images`.
    pub(super) fn maybe(
        st: &Stage<'_>,
        cfg: &RsGdfConfig,
        images: &[[f64; 3]],
        ledger: &mut Ledger,
    ) -> Result<Option<Self>, FerricError> {
        match cfg.range_split {
            None => Ok(None),
            Some(rs) => Self::new(st, rs, images, ledger).map(Some),
        }
    }

    /// The plan a DERIVATIVE of `gdf` walks (forces, stress): the build's
    /// own [`RangeSplit`] on the build's stage and pair `images`, so every
    /// piece, factor and screen is the energy's. `None` for an unsplit
    /// build; an error for a build made with a [`RangeSplitMutant`] (its
    /// energy is deliberately wrong and has no derivative here).
    pub(super) fn for_derivatives(
        st: &Stage<'_>,
        gdf: &super::RsGdf,
        images: &[[f64; 3]],
        ledger: &mut Ledger,
    ) -> Result<Option<Self>, FerricError> {
        let Some(rs) = gdf.range_split() else {
            return Ok(None);
        };
        if rs.mutant != RangeSplitMutant::Production {
            return Err(FerricError::General(format!(
                "RS-GDF derivatives: the RsGdf was built with the range-split mutant {:?}; \
                 forces/stress exist only for RangeSplitMutant::Production",
                rs.mutant
            )));
        }
        Self::new(st, rs, images, ledger).map(Some)
    }

    /// The plan of `rs` for the SR 3-centre walk ALONE on `st` — the
    /// column-rotated stage (`super` module doc "Column rotation"): the
    /// orbital and aux piece bases, but no smooth-pair basis and no `S_ss`
    /// (those serve the LR and G = 0 terms, which stay in the parent basis
    /// on the build's own plan). Only the SR 3-centre sum and its counts
    /// ([`SplitPlan::sr_three_index_s2`], [`SplitPlan::counts`]) may use it.
    /// Its pieces are split from the ROTATED shells, whose libint
    /// normalisation the piece factors `f` refer to.
    pub(super) fn sr3_only(st: &Stage<'_>, rs: RangeSplit) -> Result<Self, FerricError> {
        rs.validate()?;
        Ok(Self {
            rs,
            obs: Side::new(st.obs, &st.obs_sh, rs.orbital_threshold(st.omega))?,
            aux: Side::new(st.aux, &st.aux_sh, rs.aux_threshold(st.omega))?,
            smooth_obs: None,
            s_ss: None,
        })
    }

    /// This plan's [`RangeSplit`] for the SR 3-centre derivative walk on the
    /// column-rotated stage `st` ([`SplitPlan::sr3_only`]; the forces and
    /// stress of a rotated build, `super::deriv`).
    pub(super) fn sr3_plan_on(&self, st: &Stage<'_>) -> Result<Self, FerricError> {
        Self::sr3_only(st, self.rs)
    }

    /// The plan of `rs` with its resident buffers reserved on `ledger` and
    /// `S_ss` formed over the pair `images`.
    fn new(
        st: &Stage<'_>,
        rs: RangeSplit,
        images: &[[f64; 3]],
        ledger: &mut Ledger,
    ) -> Result<Self, FerricError> {
        rs.validate()?;
        let obs = Side::new(st.obs, &st.obs_sh, rs.orbital_threshold(st.omega))?;
        let aux = Side::new(st.aux, &st.aux_sh, rs.aux_threshold(st.omega))?;
        let smooth_obs = smooth_obs(st.cell, st.obs, rs.orbital_threshold(st.omega))?;
        let n = st.obs.nbasis() as u64;
        ledger.reserve(
            &format!("RsGdf range split: S_ss and S − S_ss (nao = {n})"),
            bytes_of(n.saturating_mul(n), 16),
        )?;
        if let Some(sm) = &smooth_obs {
            let ns = sm.prep.nbasis() as u64;
            ledger.reserve(
                &format!(
                    "RsGdf range split: (ss pair | X_c) accumulator (n_smooth = {ns}, naux = {})",
                    st.aux.nbasis()
                ),
                bytes_of(
                    ns.saturating_mul(ns).saturating_mul(st.aux.nbasis() as u64),
                    8,
                ),
            )?;
        }
        let s_ss = match &smooth_obs {
            None => None,
            Some(sm) => Some(Self::smooth_overlap(sm, images, st.obs.nbasis())?),
        };
        Ok(Self {
            rs,
            obs,
            aux,
            smooth_obs,
            s_ss,
        })
    }

    /// `q_c`: the charges of the compact aux pieces (`X_c` at G = 0), exactly
    /// the vector the build's G = 0 subtract uses.
    pub(in crate::rsgdf) fn compact_charges(&self, naux: usize) -> Vec<f64> {
        aux_ft_shells(&self.aux.c_sh, naux, &[[0.0; 3]])
            .column(0)
            .iter()
            .map(|z| z.re)
            .collect()
    }

    /// The (at most two) kept SR calls of parent pair `(i1, i2)`, as
    /// combined-basis shell pairs: `(χ_i, χ_j^c)` and `(χ_i^c, χ_j^s)`.
    fn calls(&self, i1: usize, i2: usize) -> [Option<(usize, usize)>; 2] {
        let a = self.obs.compact[i2].map(|c2| (i1, c2));
        let b = match (self.obs.compact[i1], self.obs.smooth[i2]) {
            (Some(c1), Some(s2)) => Some((c1, s2)),
            _ => None,
        };
        [a, b]
    }

    /// `(parent aux shell, compact piece)` pairs, parent order.
    fn compact_aux(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.aux
            .compact
            .iter()
            .enumerate()
            .filter_map(|(ip, x)| x.map(|x| (ip, x)))
    }

    /// `sr_screen = false`: the global 3-centre radius over the pieces.
    fn sr3_global_radius(&self, st: &Stage<'_>) -> f64 {
        if st.sr_screen {
            return 0.0;
        }
        let mut r = 0.0_f64;
        for a in &self.obs.xsh {
            for b in &self.obs.xsh {
                let (qab, pmin, pmax) = pair_bound(a, b, 0.0);
                for (_, xp) in self.compact_aux() {
                    let p = &self.aux.xsh[xp];
                    if let Some(x) = st.radius(qab, p.qbound, (pmin, pmax), (p.amin, p.amax)) {
                        r = r.max(x);
                    }
                }
            }
        }
        r
    }

    /// `sr_screen = false`: the global metric radius over compact pieces.
    fn sr2_global_radius(&self, st: &Stage<'_>) -> f64 {
        if st.sr_screen {
            return 0.0;
        }
        let mut r = 0.0_f64;
        for (_, xp) in self.compact_aux() {
            for (_, xq) in self.compact_aux() {
                let (p, q) = (&self.aux.xsh[xp], &self.aux.xsh[xq]);
                if let Some(x) = st.radius(p.qbound, q.qbound, (p.amin, p.amax), (q.amin, q.amax)) {
                    r = r.max(x);
                }
            }
        }
        r
    }

    /// One kept call `(x1, x2)` at pair image `l`: every compact aux piece
    /// in parent order, then every kept `T` in walker order —
    /// `visit(parent aux shell, compact piece, T)`. The same geometry and
    /// screen as [`Stage::sr3_pair_image`] on the piece data, so an unsplit
    /// shell pair visits exactly today's triplets in today's order.
    fn sr3_call<F>(
        &self,
        st: &Stage<'_>,
        (x1, x2): (usize, usize),
        l: &[f64; 3],
        global: f64,
        count: &mut usize,
        visit: &mut F,
    ) -> Result<(), FerricError>
    where
        F: FnMut(usize, usize, [f64; 3]) -> Result<(), FerricError>,
    {
        let a = &self.obs.xsh[x1];
        let b = &self.obs.xsh[x2];
        let bc = [b.center[0] + l[0], b.center[1] + l[1], b.center[2] + l[2]];
        let ab = [
            a.center[0] - bc[0],
            a.center[1] - bc[1],
            a.center[2] - bc[2],
        ];
        let r2 = dot3(&ab, &ab);
        let (qab, pmin, pmax) = pair_bound(a, b, r2);
        let mid = [
            0.5 * (a.center[0] + bc[0]),
            0.5 * (a.center[1] + bc[1]),
            0.5 * (a.center[2] + bc[2]),
        ];
        let half = 0.5 * r2.sqrt();
        for (ip, xp) in self.compact_aux() {
            let p = &self.aux.xsh[xp];
            let rad = if st.sr_screen {
                match st.radius(qab, p.qbound, (pmin, pmax), (p.amin, p.amax)) {
                    Some(r) => r,
                    None => continue,
                }
            } else {
                global
            };
            let x0 = [
                mid[0] - p.center[0],
                mid[1] - p.center[1],
                mid[2] - p.center[2],
            ];
            st.walker.visit(x0, rad + half, |t| {
                let x = [p.center[0] + t[0], p.center[1] + t[1], p.center[2] + t[2]];
                if segment_distance(x, a.center, bc) > rad {
                    return Ok(());
                }
                *count += 1;
                visit(ip, xp, t)
            })?;
        }
        Ok(())
    }

    /// Kept SR 3-index sum `Σ_{L,T} (χ_i χ_j − χ_i^s χ_j^s | X^c_T)_erfc`,
    /// `(nao², naux)` unsymmetrised, and the triplet count: the single bin
    /// of [`SplitPlan::sr_three_index_binned`].
    fn sr_three_index(
        &self,
        st: &Stage<'_>,
        images: &[[f64; 3]],
    ) -> Result<(Array2<f64>, usize), FerricError> {
        let (mut bins, count) = self.sr_three_index_binned(st, images, SrBinning::GAMMA)?;
        Ok((bins.swap_remove(0), count))
    }

    /// The kept SR 3-index sum binned by `(L mod mod_l, T mod mod_t)` (the
    /// k-point build; [`Stage::sr_three_index_binned`]'s layout and bins),
    /// and the triplet count.
    ///
    /// PARALLEL over `(ordered parent shell pair, r_L)` tasks, BIT-IDENTICAL
    /// across thread counts by the construction of
    /// [`Stage::sr_three_index_binned`]: row `μν` of every bin receives
    /// addends only from its parent pair's tasks, in the fixed order
    /// `L ≡ r_L (images order) → call → P → T`, into a zeroed scratch that is
    /// then COPIED. A split that moves nothing visits exactly the unsplit
    /// walk's triplets in its order with every factor exactly 1, so its bins
    /// are bitwise the unsplit ones.
    pub(in crate::rsgdf) fn sr_three_index_binned(
        &self,
        st: &Stage<'_>,
        images: &[[f64; 3]],
        bins: SrBinning,
    ) -> Result<(Vec<Array2<f64>>, usize), FerricError> {
        let n = st.obs.nbasis();
        let nsh = st.obs_sh.len();
        let rl = bins.n_l();
        let recip = st.cell.reciprocal();
        let l_bin: Vec<usize> = images
            .iter()
            .map(|l| SrBinning::residue(&recip, l, bins.mod_l))
            .collect();
        let pool = EnginePool::from_fn(|| {
            Engine::new_3center(
                Operator::erfc(st.omega),
                &self.obs.x,
                &self.aux.x,
                ENGINE_PRECISION,
            )
        })?;
        let out = Mutex::new(
            (0..rl * bins.n_t())
                .map(|_| Array2::<f64>::zeros((n * n, st.aux.nbasis())))
                .collect::<Vec<_>>(),
        );
        let ctx = Sr3Ctx {
            pool: &pool,
            images,
            l_bin: &l_bin,
            recip,
            bins,
            global: self.sr3_global_radius(st),
            out: &out,
        };
        let counts: Vec<Result<usize, FerricError>> = (0..nsh * nsh * rl)
            .into_par_iter()
            .map(|task| {
                let (pair, r_l) = (task / rl, task % rl);
                self.sr3_task(st, &ctx, r_l, (pair / nsh, pair % nsh))
            })
            .collect();
        let count = sum_counts_in_pair_order(counts)?;
        Ok((out.into_inner().unwrap_or_else(|e| e.into_inner()), count))
    }

    /// One `(parent pair, r_L)` task of [`SplitPlan::sr_three_index_binned`].
    fn sr3_task(
        &self,
        st: &Stage<'_>,
        ctx: &Sr3Ctx<'_>,
        r_l: usize,
        (i1, i2): (usize, usize),
    ) -> Result<usize, FerricError> {
        let (acc, count) = self.sr3_acc(st, ctx, r_l, (i1, i2))?;
        if count > 0 {
            let (a, b) = (&st.obs_sh[i1], &st.obs_sh[i2]);
            let (n, naux) = (st.obs.nbasis(), st.aux.nbasis());
            let (na, nb) = (a.nfun, b.nfun);
            let bl = na * nb * naux;
            let rt = ctx.bins.n_t();
            let mut out = ctx.out.lock().unwrap_or_else(|e| e.into_inner());
            for r_t in 0..rt {
                let j3 = &mut out[r_l * rt + r_t];
                for i in 0..na {
                    for j in 0..nb {
                        let row = (a.off + i) * n + b.off + j;
                        let src = r_t * bl + (i * nb + j) * naux;
                        j3.row_mut(row)
                            .iter_mut()
                            .zip(&acc[src..src + naux])
                            .for_each(|(d, &s)| *d = s);
                    }
                }
            }
        }
        Ok(count)
    }

    /// The accumulation of one [`SplitPlan::sr3_task`] (shared with the Gamma
    /// s2 walk [`SplitPlan::sr_three_index_s2`]): parent pair `(i1, i2)`'s
    /// kept calls over every image `L ≡ r_L` into a zeroed `(R_T, nμ nν,
    /// naux)` scratch. Returns `(scratch, triplet count)`.
    fn sr3_acc(
        &self,
        st: &Stage<'_>,
        ctx: &Sr3Ctx<'_>,
        r_l: usize,
        (i1, i2): (usize, usize),
    ) -> Result<(Vec<f64>, usize), FerricError> {
        let (a, b) = (&st.obs_sh[i1], &st.obs_sh[i2]);
        let (na, nb) = (a.nfun, b.nfun);
        let naux = st.aux.nbasis();
        let bl = na * nb * naux;
        // acc[r_T bl + (i nb + j) naux + P]
        //   == bins[r_L R_T + r_T][(a.off+i) n + b.off + j, P]
        let mut acc = vec![0.0_f64; ctx.bins.n_t() * bl];
        let mut count = 0usize;
        let calls = self.calls(i1, i2);
        ctx.pool.with(|eng| -> Result<(), FerricError> {
            let images = ctx.images.iter().zip(ctx.l_bin).filter(|(_, r)| **r == r_l);
            for (l, _) in images {
                for &(x1, x2) in calls.iter().flatten() {
                    let s12 = self.obs.scale[x1] * self.obs.scale[x2];
                    let mut visit =
                        |ip: usize, xp: usize, t: [f64; 3]| -> Result<(), FerricError> {
                            if let Some(blk) = eng.compute_eri3_shifted(
                                &self.obs.x,
                                &self.aux.x,
                                xp,
                                x1,
                                x2,
                                [t, [0.0; 3], *l],
                            )? {
                                let sc = s12 * self.aux.scale[xp];
                                let r_t = SrBinning::residue(&ctx.recip, &t, ctx.bins.mod_t);
                                let dst = &mut acc[r_t * bl..(r_t + 1) * bl];
                                let p = &st.aux_sh[ip];
                                for pp in 0..p.nfun {
                                    for i in 0..na {
                                        let src = (pp * na + i) * nb;
                                        for j in 0..nb {
                                            dst[(i * nb + j) * naux + p.off + pp] +=
                                                sc * blk[src + j];
                                        }
                                    }
                                }
                            }
                            Ok(())
                        };
                    self.sr3_call(st, (x1, x2), l, ctx.global, &mut count, &mut visit)?;
                }
            }
            Ok(())
        })?;
        Ok((acc, count))
    }

    /// The orientation the Gamma s2 walk evaluates unordered parent pair
    /// `lo <= hi` in: `(hi, lo)` iff it has FEWER kept calls ([`Self::calls`];
    /// a compact-only × split pair needs 1 call as `(split, compact)` and 2 as
    /// `(compact, split)`), else `(lo, hi)`. The kept part is symmetric per
    /// unordered pair (module doc), so either orientation gives both rows.
    /// Ties (every pair of a split that moves nothing) keep index order, so
    /// such a split stays bitwise the unsplit s2 walk.
    fn orient(&self, lo: usize, hi: usize) -> (usize, usize) {
        let n_calls = |i1: usize, i2: usize| self.calls(i1, i2).iter().flatten().count();
        if n_calls(hi, lo) < n_calls(lo, hi) {
            (hi, lo)
        } else {
            (lo, hi)
        }
    }

    /// Gamma kept SR 3-index sum over UNORDERED parent pairs (s2, `super`
    /// module doc "Orbital-pair symmetry"): `(J3_SR, computed triplets,
    /// ordered-equivalent triplets)`, EXACTLY symmetric in `μ ↔ ν`. Each
    /// pair `lo <= hi` runs the ordered walk's task body
    /// ([`SplitPlan::sr3_acc`], single Gamma bin) in the orientation
    /// [`SplitPlan::orient`] picks, and COPIES the block into both rows, so
    /// every element is bitwise the ordered walk's value of that
    /// orientation's row and is written by exactly one task (bitwise across
    /// thread counts).
    fn sr_three_index_s2(
        &self,
        st: &Stage<'_>,
        images: &[[f64; 3]],
    ) -> Result<(Array2<f64>, usize, usize), FerricError> {
        let n = st.obs.nbasis();
        let pool = EnginePool::from_fn(|| {
            Engine::new_3center(
                Operator::erfc(st.omega),
                &self.obs.x,
                &self.aux.x,
                ENGINE_PRECISION,
            )
        })?;
        let out = Mutex::new(vec![Array2::<f64>::zeros((n * n, st.aux.nbasis()))]);
        // Gamma: every image is residue 0 (as the single-bin walk has it).
        let l_bin = vec![0usize; images.len()];
        let ctx = Sr3Ctx {
            pool: &pool,
            images,
            l_bin: &l_bin,
            recip: st.cell.reciprocal(),
            bins: SrBinning::GAMMA,
            global: self.sr3_global_radius(st),
            out: &out,
        };
        let pairs = unordered_pairs(st.obs_sh.len());
        let counts: Vec<Result<usize, FerricError>> = pairs
            .par_iter()
            .map(|&(lo, hi)| {
                let (p, q) = self.orient(lo, hi);
                let (acc, count) = self.sr3_acc(st, &ctx, 0, (p, q))?;
                if count > 0 {
                    let mut bins = ctx.out.lock().unwrap_or_else(|e| e.into_inner());
                    copy_pair_rows_mirrored(
                        &mut bins[0],
                        n,
                        &acc,
                        &st.obs_sh[p],
                        &st.obs_sh[q],
                        lo == hi,
                    );
                }
                Ok(count)
            })
            .collect();
        let (count, ordered) = sum_s2_counts(&pairs, counts)?;
        let mut bins = out.into_inner().unwrap_or_else(|e| e.into_inner());
        Ok((bins.swap_remove(0), count, ordered))
    }

    /// Kept SR metric `Σ_T (P^c_0 | Q^c_T)_erfc` (unsymmetrised) and the pair
    /// count: the single bin of [`SplitPlan::sr_metric_binned`].
    fn sr_metric(&self, st: &Stage<'_>) -> Result<(Array2<f64>, usize), FerricError> {
        let (mut bins, count) = self.sr_metric_binned(st, SrBinning::GAMMA.mod_t)?;
        Ok((bins.swap_remove(0), count))
    }

    /// The kept SR metric binned by the residue of `T` modulo `mod_t` (the
    /// k-point build; [`Stage::sr_metric_binned`]'s layout), and the pair
    /// count; parallel over parent aux pairs, bit-identical across thread
    /// counts (as [`Stage::sr_metric_binned`]).
    pub(in crate::rsgdf) fn sr_metric_binned(
        &self,
        st: &Stage<'_>,
        mod_t: [usize; 3],
    ) -> Result<(Vec<Array2<f64>>, usize), FerricError> {
        let naux = st.aux.nbasis();
        let nsh = st.aux_sh.len();
        let rt: usize = mod_t.iter().product();
        let pool = EnginePool::from_fn(|| {
            Engine::new_2center(Operator::erfc(st.omega), &self.aux.x, ENGINE_PRECISION)
        })?;
        let out = Mutex::new(
            (0..rt)
                .map(|_| Array2::<f64>::zeros((naux, naux)))
                .collect::<Vec<_>>(),
        );
        let ctx = Sr2Ctx {
            pool: &pool,
            recip: st.cell.reciprocal(),
            mod_t,
            global: self.sr2_global_radius(st),
            out: &out,
        };
        let counts: Vec<Result<usize, FerricError>> = (0..nsh * nsh)
            .into_par_iter()
            .map(|pair| self.sr2_task(st, Some(&ctx), (pair / nsh, pair % nsh)))
            .collect();
        let count = sum_counts_in_pair_order(counts)?;
        Ok((out.into_inner().unwrap_or_else(|e| e.into_inner()), count))
    }

    /// One parent aux pair `(ip, iq)` of the kept metric: computed into
    /// `ctx`'s residue bins, or only COUNTED when `ctx` is `None`.
    fn sr2_task(
        &self,
        st: &Stage<'_>,
        ctx: Option<&Sr2Ctx<'_>>,
        (ip, iq): (usize, usize),
    ) -> Result<usize, FerricError> {
        let (Some(xp), Some(xq)) = (self.aux.compact[ip], self.aux.compact[iq]) else {
            return Ok(0);
        };
        let (p, q) = (&self.aux.xsh[xp], &self.aux.xsh[xq]);
        let global = ctx.map_or_else(|| self.sr2_global_radius(st), |c| c.global);
        let rad = if st.sr_screen {
            match st.radius(p.qbound, q.qbound, (p.amin, p.amax), (q.amin, q.amax)) {
                Some(r) => r,
                None => return Ok(0),
            }
        } else {
            global
        };
        let x0 = [
            p.center[0] - q.center[0],
            p.center[1] - q.center[1],
            p.center[2] - q.center[2],
        ];
        let mut count = 0usize;
        let Some(ctx) = ctx else {
            st.walker.visit(x0, rad, |_| {
                count += 1;
                Ok(())
            })?;
            return Ok(count);
        };
        let bl = p.nfun * q.nfun;
        let rt: usize = ctx.mod_t.iter().product();
        let sc = self.aux.scale[xp] * self.aux.scale[xq];
        let mut acc = vec![0.0_f64; rt * bl];
        ctx.pool.with(|eng| {
            st.walker.visit(x0, rad, |t| {
                count += 1;
                let blk = eng.compute_eri2_shifted(&self.aux.x, xp, xq, t)?;
                let r = SrBinning::residue(&ctx.recip, &t, ctx.mod_t);
                for (d, &s) in acc[r * bl..(r + 1) * bl].iter_mut().zip(&blk[..bl]) {
                    *d += sc * s;
                }
                Ok(())
            })
        })?;
        if count > 0 {
            let mut out = ctx.out.lock().unwrap_or_else(|e| e.into_inner());
            for (r, j2) in out.iter_mut().enumerate() {
                for i in 0..p.nfun {
                    for j in 0..q.nfun {
                        j2[(p.off + i, q.off + j)] = acc[r * bl + i * q.nfun + j];
                    }
                }
            }
        }
        Ok(count)
    }

    /// The split walks' counts without computing an integral.
    fn counts(&self, st: &Stage<'_>, images: &[[f64; 3]]) -> Result<SrWalkCounts, FerricError> {
        let nsh = st.obs_sh.len();
        let global = self.sr3_global_radius(st);
        let n3: Vec<Result<usize, FerricError>> = (0..nsh * nsh)
            .into_par_iter()
            .map(|pair| {
                let mut c = 0usize;
                let calls = self.calls(pair / nsh, pair % nsh);
                for l in images {
                    for &call in calls.iter().flatten() {
                        self.sr3_call(st, call, l, global, &mut c, &mut |_, _, _| Ok(()))?;
                    }
                }
                Ok(c)
            })
            .collect();
        let na = st.aux_sh.len();
        let n2: Vec<Result<usize, FerricError>> = (0..na * na)
            .into_par_iter()
            .map(|pair| self.sr2_task(st, None, (pair / na, pair % na)))
            .collect();
        let per_pair = collect_pair_counts(n3)?;
        let (n_sr3_triplets, n_sr3_triplets_ordered) =
            s2_counts_from_ordered(&per_pair, nsh, |lo, hi| self.orient(lo, hi));
        Ok(SrWalkCounts {
            n_sr3_triplets,
            n_sr3_triplets_ordered,
            n_sr3_triplets_s1: per_pair.iter().sum(),
            n_sr2_pairs: sum_counts_in_pair_order(n2)?,
        })
    }

    // -----------------------------------------------------------------------
    // G space
    // -----------------------------------------------------------------------

    /// Moved-aux LR: `J3 += Re[P̄ Y]`, `J2 += Re[Ȳ X] + v_SR Re[X̄_c X_s]`
    /// with `Y = v_LR X + v_SR X_s` (weights on the AUX side; module doc).
    #[allow(clippy::too_many_arguments)]
    fn lr_moved_aux(
        &self,
        st: &Stage<'_>,
        gv: &[[f64; 3]],
        j2: &mut Array2<f64>,
        j3: &mut Array2<f64>,
        chunk_budget: usize,
        kernel: LrKernel,
        sub: &mut PbcTimings,
    ) -> Result<(usize, f64, Option<f64>), FerricError> {
        let n = st.obs.nbasis();
        let n2 = n * n;
        let naux = st.aux.nbasis();
        let (vol, omega) = (st.cell.volume(), st.omega);
        // pr, pi (n² reals each) + X, X_s, X_c (naux complex each) + 8 real
        // (naux) rows.
        let extra_per_g = n2
            .saturating_mul(16)
            .saturating_add(naux.saturating_mul(112))
            .saturating_add(64);
        let pair_ft_thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
        let (mut sink_wall, mut sink_cpu) = (0.0_f64, None::<f64>);
        let mut sink_t = PbcTimings::default();
        let mut bufs = PackBufs::default();
        let sink =
            |_g0: usize, gs: &[[f64; 3]], pft: &Array3<Complex64>| -> Result<(), FerricError> {
                let clock = StageClock::start();
                let ng = gs.len();
                let c = StageClock::start();
                let x = aux_ft_shells(&st.aux_sh, naux, gs);
                let xs = aux_ft_shells(&self.aux.s_sh, naux, gs);
                let xc = aux_ft_shells(&self.aux.c_sh, naux, gs);
                sink_t.stop_sub(SUB_AUX_FT, &c);
                let c = StageClock::start();
                let (pr, pim) = pack_pair_ft(pft, None, kernel, &mut bufs);
                sink_t.stop_sub(SUB_P_PACK, &c);
                let c = StageClock::start();
                let z = || Array2::<f64>::zeros((naux, ng));
                let (mut xr, mut xi, mut yr, mut yi) = (z(), z(), z(), z());
                let (mut xsr, mut xsi, mut xcwr, mut xcwi) = (z(), z(), z(), z());
                for (g, gvec) in gs.iter().enumerate() {
                    let (w_lr, w_sr) = kernel_weights(gvec, vol, omega);
                    for p in 0..naux {
                        let (v, vs, vc) = (x[(p, g)], xs[(p, g)], xc[(p, g)]);
                        xr[(p, g)] = v.re;
                        xi[(p, g)] = v.im;
                        yr[(p, g)] = w_lr * v.re + w_sr * vs.re;
                        yi[(p, g)] = w_lr * v.im + w_sr * vs.im;
                        xsr[(p, g)] = vs.re;
                        xsi[(p, g)] = vs.im;
                        xcwr[(p, g)] = w_sr * vc.re;
                        xcwi[(p, g)] = w_sr * vc.im;
                    }
                }
                sink_t.stop_sub(SUB_XY_PACK, &c);
                // Re[conj(A) B] = A.re B.re + A.im B.im
                let j2_terms: [(&Array2<f64>, &Array2<f64>); 4] =
                    [(&yr, &xr), (&yi, &xi), (&xcwr, &xsr), (&xcwi, &xsi)];
                lr_gemms(
                    j3,
                    &[(pr, &yr), (pim, &yi)],
                    Some((&mut *j2, &j2_terms[..])),
                    kernel,
                    &mut sink_t,
                );
                add_clock(&mut sink_wall, &mut sink_cpu, clock.elapsed());
                Ok(())
            };
        let n_chunks = lr_pair_ft_chunked(
            kernel,
            st.cell,
            st.obs,
            gv,
            pair_ft_thresh,
            chunk_budget,
            extra_per_g,
            sub,
            sink,
        )?;
        sub.accumulate(&sink_t);
        Ok((n_chunks, sink_wall, sink_cpu))
    }

    /// The moved `(ss pair | X^c)` block: `J3 += (2/Ω) Σ_G v_SR Re[P̄_ss X_c]`
    /// with the lattice pair FT of the smooth orbital pieces, accumulated in
    /// the smooth AO basis and scattered into J3's parent rows at the end.
    #[allow(clippy::too_many_arguments)]
    fn lr_smooth_pairs(
        &self,
        st: &Stage<'_>,
        sm: &SmoothObs,
        gv: &[[f64; 3]],
        j3: &mut Array2<f64>,
        chunk_budget: usize,
        kernel: LrKernel,
        sub: &mut PbcTimings,
    ) -> Result<(usize, f64, Option<f64>), FerricError> {
        let n = st.obs.nbasis();
        let ns = sm.prep.nbasis();
        let ns2 = ns * ns;
        let naux = st.aux.nbasis();
        let (vol, omega) = (st.cell.volume(), st.omega);
        let extra_per_g = ns2
            .saturating_mul(16)
            .saturating_add(naux.saturating_mul(32))
            .saturating_add(64);
        let pair_ft_thresh = (0.01 * st.thresh).min(DEFAULT_PAIR_FT_THRESH);
        let mut acc = Array2::<f64>::zeros((ns2, naux));
        let (mut sink_wall, mut sink_cpu) = (0.0_f64, None::<f64>);
        let mut sink_t = PbcTimings::default();
        let mut bufs = PackBufs::default();
        let sink =
            |_g0: usize, gs: &[[f64; 3]], pft: &Array3<Complex64>| -> Result<(), FerricError> {
                let clock = StageClock::start();
                let ng = gs.len();
                let c = StageClock::start();
                let xc = aux_ft_shells(&self.aux.c_sh, naux, gs);
                sink_t.stop_sub(SUB_AUX_FT, &c);
                let c = StageClock::start();
                let (pr, pim) = pack_pair_ft(pft, None, kernel, &mut bufs);
                sink_t.stop_sub(SUB_P_PACK, &c);
                let c = StageClock::start();
                let mut xcwr = Array2::<f64>::zeros((naux, ng));
                let mut xcwi = Array2::<f64>::zeros((naux, ng));
                for (g, gvec) in gs.iter().enumerate() {
                    let (_, w_sr) = kernel_weights(gvec, vol, omega);
                    for p in 0..naux {
                        let v = xc[(p, g)];
                        xcwr[(p, g)] = w_sr * v.re;
                        xcwi[(p, g)] = w_sr * v.im;
                    }
                }
                sink_t.stop_sub(SUB_XY_PACK, &c);
                lr_gemms(
                    &mut acc,
                    &[(pr, &xcwr), (pim, &xcwi)],
                    None,
                    kernel,
                    &mut sink_t,
                );
                add_clock(&mut sink_wall, &mut sink_cpu, clock.elapsed());
                Ok(())
            };
        let n_chunks = lr_pair_ft_chunked(
            kernel,
            st.cell,
            &sm.prep,
            gv,
            pair_ft_thresh,
            chunk_budget,
            extra_per_g,
            sub,
            sink,
        )?;
        sub.accumulate(&sink_t);
        for a in 0..ns {
            for b in 0..ns {
                let row = sm.ao_map[a] * n + sm.ao_map[b];
                let mut dst = j3.row_mut(row);
                dst += &acc.row(a * ns + b);
            }
        }
        Ok((n_chunks, sink_wall, sink_cpu))
    }

    /// `S_ss = Σ_L ⟨χ^s_μ | χ^s_ν(· − L)⟩` over the pair `images` (libint 1e
    /// overlap on the smooth pieces, times their normalisation factors),
    /// symmetrised, in the parent AO order.
    fn smooth_overlap(
        sm: &SmoothObs,
        images: &[[f64; 3]],
        n: usize,
    ) -> Result<Array2<f64>, FerricError> {
        let prep = &sm.prep;
        let (offs, dims) = (prep.shell_offsets(), prep.shell_dims());
        let mut eng = Engine::new_1e(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION)?;
        let mut s = Array2::<f64>::zeros((n, n));
        let nsh = prep.nshells();
        for l in images {
            for i1 in 0..nsh {
                for i2 in 0..nsh {
                    let blk = eng.compute_1e_block_shifted(prep, i1, i2, *l)?;
                    let f = sm.scale[i1] * sm.scale[i2];
                    let (d1, d2) = (dims[i1], dims[i2]);
                    for a in 0..d1 {
                        for b in 0..d2 {
                            let (r, c) = (sm.ao_map[offs[i1] + a], sm.ao_map[offs[i2] + b]);
                            s[(r, c)] += f * blk[a * d2 + b];
                        }
                    }
                }
            }
        }
        Ok(0.5 * (&s + &s.t()))
    }
}

/// `(2/Ω) v_LR(G)` and `(2/Ω) v_SR(G)` for the half sphere (weight 2).
fn kernel_weights(g: &[f64; 3], vol: f64, omega: f64) -> (f64, f64) {
    let g2 = dot3(g, g);
    let full = 2.0 / vol * 4.0 * PI / g2;
    let x = g2 / (4.0 * omega * omega);
    (full * (-x).exp(), -full * (-x).exp_m1())
}

/// Accumulates a `(wall, cpu)` clock reading: `wall` always adds; `cpu` becomes `Some` once any
/// reading carries a CPU time.
fn add_clock(wall: &mut f64, cpu: &mut Option<f64>, (w, c): (f64, Option<f64>)) {
    *wall += w;
    if let Some(c) = c {
        *cpu = Some(cpu.unwrap_or(0.0) + c);
    }
}

/// `J3[μν, P] −= c0 s[μν] q[P]` (the three-index half of [`subtract_g0`];
/// the split mutants only).
fn subtract_three(j3: &mut Array2<f64>, s_flat: &[f64], q: &[f64], c0: f64) {
    for (mn, smn) in s_flat.iter().enumerate() {
        for (p, qp) in q.iter().enumerate() {
            j3[(mn, p)] -= c0 * smn * qp;
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch from `RsGdf::build_impl` (None = today's construction, verbatim)
// ---------------------------------------------------------------------------

/// Whether the SR walks follow the partition (every plan but the
/// double-count mutant).
fn split_walks(plan: Option<&SplitPlan>) -> Option<&SplitPlan> {
    plan.filter(|p| p.rs.mutant != RangeSplitMutant::DoubleCount)
}

/// SR metric of the build: [`Stage::sr_metric`] or the kept `(P^c|Q^c)`.
pub(super) fn sr_metric(
    st: &Stage<'_>,
    plan: Option<&SplitPlan>,
) -> Result<(Array2<f64>, usize), FerricError> {
    match split_walks(plan) {
        None => st.sr_metric(),
        Some(p) => p.sr_metric(st),
    }
}

/// Gamma SR 3-index sum of the build over unordered shell pairs (s2,
/// `super` module doc "Orbital-pair symmetry"): [`Stage::sr_three_index_s2`]
/// or the kept part's [`SplitPlan::sr_three_index_s2`]. Returns `(J3_SR,
/// computed triplets, ordered-equivalent triplets)`.
pub(super) fn sr_three_index(
    st: &Stage<'_>,
    plan: Option<&SplitPlan>,
    images: &[[f64; 3]],
) -> Result<(Array2<f64>, usize, usize), FerricError> {
    match split_walks(plan) {
        None => st.sr_three_index_s2(images),
        Some(p) => p.sr_three_index_s2(st, images),
    }
}

/// FROZEN pre-s2 Gamma SR 3-index sum (ordered pairs; the frozen oracle
/// [`super::RsGdf::build_pair_s1_oracle`]): [`Stage::sr_three_index`] or the
/// kept part, as `(J3_SR, count, count)`.
pub(super) fn sr_three_index_s1(
    st: &Stage<'_>,
    plan: Option<&SplitPlan>,
    images: &[[f64; 3]],
) -> Result<(Array2<f64>, usize, usize), FerricError> {
    let (j3, count) = match split_walks(plan) {
        None => st.sr_three_index(images)?,
        Some(p) => p.sr_three_index(st, images)?,
    };
    Ok((j3, count, count))
}

/// LR (G ≠ 0) terms of the build: [`Stage::lr_accumulate`] when no aux
/// primitive moved, else the aux-side-weighted form; plus the moved
/// `(ss | X^c)` block when orbital primitives moved. Returns
/// `(chunks, sink wall, sink CPU)` as [`Stage::lr_accumulate`] does;
/// sub-stages and counters into `sub`.
#[allow(clippy::too_many_arguments)]
pub(super) fn lr_accumulate(
    st: &Stage<'_>,
    plan: Option<&SplitPlan>,
    gv: &[[f64; 3]],
    j2: &mut Array2<f64>,
    j3: &mut Array2<f64>,
    chunk_budget: usize,
    kernel: LrKernel,
    sub: &mut PbcTimings,
) -> Result<(usize, f64, Option<f64>), FerricError> {
    sub.max_counter(
        "LR J3 GEMM row blocks per call",
        j3.nrows().div_ceil(LR_GEMM_ROW_BLOCK) as u64,
    );
    let Some(p) = plan else {
        return st.lr_accumulate(gv, j2, j3, chunk_budget, kernel, sub);
    };
    let (mut chunks, mut wall, mut cpu) = if p.aux.s_sh.is_empty() {
        st.lr_accumulate(gv, j2, j3, chunk_budget, kernel, sub)?
    } else {
        p.lr_moved_aux(st, gv, j2, j3, chunk_budget, kernel, sub)?
    };
    if let Some(sm) = p.smooth_obs.as_ref().filter(|_| !p.aux.c_sh.is_empty()) {
        let (c, w, u) = p.lr_smooth_pairs(st, sm, gv, j3, chunk_budget, kernel, sub)?;
        chunks += c;
        add_clock(&mut wall, &mut cpu, (w, u));
    }
    Ok((chunks, wall, cpu))
}

/// The G = 0 subtract: [`subtract_g0`] with `(S, q)` without a plan, with
/// the kept inputs `(S − S_ss, q_c)` with one (grouping A; module doc).
pub(super) fn subtract_g0_build(
    plan: Option<&SplitPlan>,
    j2: &mut Array2<f64>,
    j3: &mut Array2<f64>,
    s_flat: &[f64],
    q: &[f64],
    c0: f64,
    mode: G0Handling,
) {
    let Some(p) = plan else {
        subtract_g0(j2, j3, s_flat, q, c0, mode);
        return;
    };
    // Nothing smooth: S − 0 is S exactly (a copy, no arithmetic).
    let s_eff: Vec<f64> = match &p.s_ss {
        None => s_flat.to_vec(),
        Some(s_ss) => s_flat.iter().zip(s_ss.iter()).map(|(a, b)| a - b).collect(),
    };
    let q_c = p.compact_charges(q.len());
    match p.rs.mutant {
        RangeSplitMutant::ThreeIndexFullG0 => {
            subtract_g0(j2, j3, &s_eff, &q_c, c0, G0Handling::MetricOnlyMutant);
            subtract_three(j3, s_flat, q, c0);
        }
        RangeSplitMutant::MetricFullG0 => {
            subtract_g0(j2, j3, s_flat, q, c0, G0Handling::MetricOnlyMutant);
            subtract_three(j3, &s_eff, &q_c, c0);
        }
        _ => subtract_g0(j2, j3, &s_eff, &q_c, c0, mode),
    }
}

/// After the metric solve of a range-split build (a no-op without one):
/// the metric guard (module doc), then the partition's counters on the
/// build timings.
pub(super) fn finish(
    plan: Option<&SplitPlan>,
    eig_min: f64,
    t: &mut PbcTimings,
) -> Result<(), FerricError> {
    let Some(p) = plan else {
        return Ok(());
    };
    if !(eig_min >= -RANGE_SPLIT_NEG_EIG_GUARD) {
        return Err(FerricError::General(format!(
            "RsGdf range split: smallest metric eigenvalue {eig_min:.3e} < \
             -{RANGE_SPLIT_NEG_EIG_GUARD:e}. A G = 0 bookkeeping error in the split metric \
             produces exactly this (one large negative eigenvalue along q), and the lindep cut \
             would silently drop it; refusing the fit"
        )));
    }
    for (name, v) in p.counters() {
        t.set_counter(name, v as u64);
    }
    Ok(())
}

/// TEST/DIAGNOSTIC: the SR walk counts an [`super::RsGdf::build`] at `cfg`
/// would report (`n_sr3_triplets`, `n_sr3_triplets_ordered`,
/// `n_sr2_pairs`), from the SAME walks with every integral skipped — the
/// cost statement of the range split without its integrals (FINDINGS
/// "Iteration 23", `count_sr3_ferric`) — plus the pre-s2 ordered-walk count
/// `n_sr3_triplets_s1` (the walk visits every ordered pair once and derives
/// the s2 counts from those per-pair counts). With
/// [`RsGdfConfig::sr_column_rotation`] resolved on (`Auto`, the default, or
/// `On`) the counts are those of the build's ROTATED walk (`super` module doc
/// "Column rotation"); pass `SrColumnRotation::Off` for the unrotated walk.
pub fn sr_walk_counts(
    cell: &Cell,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    cfg: &RsGdfConfig,
) -> Result<SrWalkCounts, FerricError> {
    let (st, images) = super::kpoint::diagnostic_stage(cell, obs, aux, cfg)?;
    // Column rotation (`super` module doc): the build's rotated walk on the
    // parent pair images.
    if let Some(rot) = super::sr3_rotation(cell, obs, aux, cfg)? {
        let st_rot = st.with_bases(&rot.obs.prep, rot.aux_prep(aux))?;
        let plan = match cfg.range_split {
            None => None,
            Some(rs) => Some(SplitPlan::sr3_only(&st_rot, rs)?),
        };
        return walk_counts(&st_rot, plan.as_ref(), &images);
    }
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    let plan = SplitPlan::maybe(&st, cfg, &images, &mut ledger)?;
    walk_counts(&st, plan.as_ref(), &images)
}

/// An ESTIMATE of [`SrWalkCounts::n_sr3_triplets`] (the s2 count the build
/// reports) that walks at most about `budget` triplets: the cells
/// `(unordered shell pair, pair image)` are visited in a fixed
/// full-cycle pseudo-random order (stride permutation, no RNG) in waves, and
/// the count of the visited cells is scaled by `cells / visited` once the
/// budget is spent. A problem with at most `budget` triplets is counted
/// EXACTLY (every cell visited; equal to `sr_walk_counts(..).n_sr3_triplets`).
/// Deterministic and independent of the thread count (wave sums are taken
/// in cell order). Used by the RS-GDF ω chooser (`auto_omega`), where an
/// exact walk would cost a large fraction of the build it is choosing for.
pub fn sr_triplet_estimate(
    cell: &Cell,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    cfg: &RsGdfConfig,
    budget: u64,
) -> Result<f64, FerricError> {
    let (st, images) = super::kpoint::diagnostic_stage(cell, obs, aux, cfg)?;
    if let Some(rot) = super::sr3_rotation(cell, obs, aux, cfg)? {
        let st_rot = st.with_bases(&rot.obs.prep, rot.aux_prep(aux))?;
        let plan = match cfg.range_split {
            None => None,
            Some(rs) => Some(SplitPlan::sr3_only(&st_rot, rs)?),
        };
        return estimate_walk(&st_rot, plan.as_ref(), &images, budget);
    }
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    let plan = SplitPlan::maybe(&st, cfg, &images, &mut ledger)?;
    estimate_walk(&st, plan.as_ref(), &images, budget)
}

/// Cells per wave of [`sample_total`]: a multiple of every plausible thread
/// count, small enough that a wave is a few million triplets at most.
const ESTIMATE_WAVE: usize = 4096;

/// [`sr_triplet_estimate`] of the stage `st` under `plan`.
fn estimate_walk(
    st: &Stage<'_>,
    plan: Option<&SplitPlan>,
    images: &[[f64; 3]],
    budget: u64,
) -> Result<f64, FerricError> {
    let pairs = unordered_pairs(st.obs_sh.len());
    let n_img = images.len();
    if pairs.is_empty() || n_img == 0 {
        return Ok(0.0);
    }
    if let Some(p) = split_walks(plan) {
        let global = p.sr3_global_radius(st);
        return sample_total(pairs.len() * n_img, budget, |c| {
            let (lo, hi) = pairs[c / n_img];
            let (i1, i2) = p.orient(lo, hi);
            let mut n = 0usize;
            for &call in p.calls(i1, i2).iter().flatten() {
                p.sr3_call(
                    st,
                    call,
                    &images[c % n_img],
                    global,
                    &mut n,
                    &mut |_, _, _| Ok(()),
                )?;
            }
            Ok(n)
        });
    }
    let global = st.sr3_global_radius();
    sample_total(pairs.len() * n_img, budget, |c| {
        let (lo, hi) = pairs[c / n_img];
        let mut n = 0usize;
        st.sr3_pair_image(
            lo,
            hi,
            &images[c % n_img],
            global,
            &mut n,
            &mut |_, _, _, _, _| Ok(()),
        )?;
        Ok(n)
    })
}

/// A stride coprime to `n` near `0.618 n`: `i -> i * stride mod n` is a
/// permutation of `0..n` that scatters consecutive indices.
fn scatter_stride(n: usize) -> usize {
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    let mut s = ((n as f64) * 0.618_033_988_749_895) as usize;
    s = s.max(1);
    while gcd(s, n) != 1 {
        s += 1;
    }
    s
}

/// `Σ_c count(c)` over `n_cells` cells, estimated from the cells visited
/// before the running total reaches `budget` (module doc of
/// [`sr_triplet_estimate`]); exact when the budget is never reached.
fn sample_total<F>(n_cells: usize, budget: u64, count: F) -> Result<f64, FerricError>
where
    F: Fn(usize) -> Result<usize, FerricError> + Sync,
{
    let stride = scatter_stride(n_cells);
    let (mut visited, mut total) = (0usize, 0u64);
    while visited < n_cells && total < budget {
        let end = (visited + ESTIMATE_WAVE).min(n_cells);
        let wave: Vec<Result<usize, FerricError>> = (visited..end)
            .into_par_iter()
            .map(|i| count(((i as u128 * stride as u128) % n_cells as u128) as usize))
            .collect();
        for x in wave {
            total += x? as u64;
        }
        visited = end;
    }
    Ok(total as f64 * n_cells as f64 / visited as f64)
}

/// [`sr_walk_counts`] of the stage `st` under `plan`.
fn walk_counts(
    st: &Stage<'_>,
    plan: Option<&SplitPlan>,
    images: &[[f64; 3]],
) -> Result<SrWalkCounts, FerricError> {
    if let Some(p) = split_walks(plan) {
        return p.counts(st, images);
    }
    let nsh = st.obs_sh.len();
    let global = st.sr3_global_radius();
    let n3: Vec<Result<usize, FerricError>> = (0..nsh * nsh)
        .into_par_iter()
        .map(|pair| {
            let mut c = 0usize;
            for l in images {
                st.sr3_pair_image(
                    pair / nsh,
                    pair % nsh,
                    l,
                    global,
                    &mut c,
                    &mut |_, _, _, _, _| Ok::<(), FerricError>(()),
                )?;
            }
            Ok(c)
        })
        .collect();
    let n_sr2_pairs = st.sr_metric_walk(|_, _, _| Ok(()))?;
    let per_pair = collect_pair_counts(n3)?;
    let (n_sr3_triplets, n_sr3_triplets_ordered) =
        s2_counts_from_ordered(&per_pair, nsh, |lo, hi| (lo, hi));
    Ok(SrWalkCounts {
        n_sr3_triplets,
        n_sr3_triplets_ordered,
        n_sr3_triplets_s1: per_pair.iter().sum(),
        n_sr2_pairs,
    })
}

/// Per-ordered-pair counts `c[i1 nsh + i2]`; the first error in pair order.
fn collect_pair_counts(counts: Vec<Result<usize, FerricError>>) -> Result<Vec<usize>, FerricError> {
    counts.into_iter().collect()
}

/// The Gamma s2 walk's `(computed, ordered-equivalent)` counts from the
/// ordered walk's per-pair counts `c` (`nsh²`, row-major): unordered pair
/// `lo <= hi` costs `c` of the orientation `orient(lo, hi)` the s2 walk
/// evaluates (the same body on the same inputs, so the same count).
fn s2_counts_from_ordered<O>(c: &[usize], nsh: usize, orient: O) -> (usize, usize)
where
    O: Fn(usize, usize) -> (usize, usize),
{
    let (mut n, mut n_ordered) = (0usize, 0usize);
    for (lo, hi) in unordered_pairs(nsh) {
        let (p, q) = orient(lo, hi);
        let x = c[p * nsh + q];
        n += x;
        n_ordered += if lo == hi { x } else { 2 * x };
    }
    (n, n_ordered)
}
