//! Secondary-basis ("BASIS2") SCF initial guess, after Q-Chem's `BASIS2` /
//! `BASISPROJTYPE`.
//!
//! The SCF is first converged in a small secondary basis, with the same
//! method (HF or the same functional), charge, multiplicity, operator and
//! settings as the target SCF. The result is then projected into the target
//! basis and used as the target SCF's starting point.
//!
//! # The two projections
//!
//! With `S_bb` the target overlap, `S_bs` the target/secondary cross overlap
//! and `P = S_bb⁺ S_bs` (the least-squares map from secondary to target
//! coefficients; `S_bb⁺ = X Xᵀ` from the SCF's canonical orthogonalizer,
//! so linear dependencies are filtered rather than inverted):
//!
//! * [`GuessProjection::Fock`] (Q-Chem `FOPPROJECTION`, the Q-Chem default
//!   for a guess): per spin, `D_σ = P D_σ,small Pᵀ`, and the target Fock built
//!   at that (generally non-idempotent) density is diagonalised for the
//!   starting orbitals. Closed shell: the projected density is handed to the
//!   SCF as its starting density, so the Fock build and diagonalisation are
//!   the SCF's own first iteration, with the full target Fock (XC included).
//!   Open shell (UHF/UKS, ROHF/ROKS): the guess Fock is built before the loop
//!   as `F_σ = h + J[D_α + D_β] − K[D_σ]` (Roothaan-combined for ROHF), the
//!   same form ferric's MINAO open-shell guess uses; it carries no XC term.
//!   Q-Chem builds the Fock from the small-basis density with mixed-basis
//!   two-electron integrals; projecting the density first and using the
//!   target-basis integrals is the same thing whenever the secondary basis
//!   lies in the span of the target basis, and a guess-level approximation
//!   otherwise.
//! * [`GuessProjection::Overlap`] (Q-Chem `OVPROJECTION`): the occupied MOs
//!   are mapped, `C = P C_small,occ`, and re-orthonormalised in the `S_bb`
//!   metric (Löwdin). The SCF starts from those orbitals. For ROHF the doubly
//!   occupied block is orthonormalised first and the singly occupied block is
//!   made `S_bb`-orthogonal to it before its own Löwdin step, so the β
//!   density spans exactly the projected doubly occupied space.
//!
//! The small-basis occupied orbitals are taken as the natural orbitals of the
//! converged small-basis spin densities (occupation 1), not as "the first
//! `nocc` columns", so a MOM- or occupation-guard-reordered small-basis
//! result still projects the occupied space it actually converged.
//!
//! # Policy
//!
//! * A secondary SCF that does not converge never aborts the target SCF or a
//!   convergence ladder. Its last iterate is NOT projected: measured on CN
//!   UHF/aug-cc-pVDZ from STO-3G (the STO-3G UHF oscillates for 200
//!   iterations), the target started from that iterate also failed to
//!   converge in 200 iterations, for both projections, stuck 0.039 Ha above
//!   the state the default guess reaches in 22. Instead the target starts
//!   from its default guess (`use_sad_guess`), a warning is printed, and the
//!   report carries `converged: false`, the exit reason and
//!   `used_as_guess: false`. (A converged guess is not neutral either: it can
//!   select WHICH stationary state the target converges to, in either
//!   direction. See `site/src/methods/scf.md`.)
//! * Genuinely unusable guesses are errors: an element missing from the
//!   secondary basis, an ECP core count that differs between the two bases
//!   (the electron counts would differ), a non-finite secondary density, and
//!   a projected occupied space that is near-singular in the target basis.
//! * A secondary basis with more functions than the target is allowed, with
//!   a warning.
//! * An explicit starting point (`init_guess_density`, or caller-supplied
//!   initial MOs) takes precedence over `guess_basis`, exactly as it already
//!   takes precedence over `use_sad_guess`.
//! * `use_sad_guess` (`guess = "minao" | "sad" | "hcore"`) is inherited by
//!   the secondary SCF, so with `guess_basis` set it selects the SECONDARY
//!   solve's starting point only; the target starts from the projection.
//! * Every solve that reaches a molecular entry point without an explicit
//!   starting point re-runs the secondary SCF: each geometry of an
//!   optimisation or a finite-difference Hessian, each restart rung of a
//!   ladder. A carried-forward density (ladder rungs after the first, the
//!   exact stage after a DF pre-stage, a stability-descent step, the cDFT
//!   λ loop) outranks it, and the first stage's report is carried onto the
//!   final result.
//! * The dual-basis energy correction (Q-Chem `DUAL_BASIS_ENERGY`) is not
//!   implemented: this module only produces a guess.

use crate::result::ScfResult;
use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{s, Array2, Axis};
use ndarray_linalg::Eigh;

/// Smallest eigenvalue of `Cᵀ S_bb C` accepted for a projected occupied
/// block before its Löwdin orthonormalisation. Below this the secondary
/// occupied space is not representable in the target basis.
pub const MIN_PROJECTED_OCC_EIGENVALUE: f64 = 1e-8;

/// How the secondary-basis solution is carried into the target basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuessProjection {
    /// Q-Chem `FOPPROJECTION`: project the density, build the target Fock at
    /// it and diagonalise.
    Fock,
    /// Q-Chem `OVPROJECTION`: project the occupied MOs through the overlap
    /// and re-orthonormalise them in the target metric.
    Overlap,
}

impl GuessProjection {
    /// Accepted config spellings, in the order error messages list them.
    pub const VALID: &'static [&'static str] = &["fock", "overlap"];

    /// Strict, ASCII-case-insensitive parser shared by the CLI
    /// (`[scf] guess_projection`) and the Python bindings
    /// (`guess_projection=`).
    pub fn parse_config_str(s: &str) -> Result<Self, FerricError> {
        match s.to_ascii_lowercase().as_str() {
            "fock" => Ok(Self::Fock),
            "overlap" => Ok(Self::Overlap),
            _ => Err(FerricError::General(format!(
                "unknown guess_projection '{s}': valid options are 'fock' (default; \
                 Q-Chem FOPPROJECTION) and 'overlap' (Q-Chem OVPROJECTION)"
            ))),
        }
    }

    /// The canonical config spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fock => "fock",
            Self::Overlap => "overlap",
        }
    }
}

/// A secondary basis to converge the SCF in before the target solve.
#[derive(Debug, Clone, PartialEq)]
pub struct GuessBasis {
    /// The secondary (usually smaller) orbital basis.
    pub basis: BasisSet,
    /// How its solution is projected into the target basis.
    pub projection: GuessProjection,
}

impl GuessBasis {
    /// A bundled basis by name with a projection spelling, both parsed
    /// strictly (an unknown basis or projection is an error).
    pub fn bundled(name: &str, projection: &str) -> Result<Self, FerricError> {
        let projection = GuessProjection::parse_config_str(projection)?;
        let basis = ferric_core::basis::bundled(name)
            .map_err(|e| FerricError::Basis(format!("guess_basis '{name}': {e}")))?;
        Ok(Self { basis, projection })
    }
}

/// What the secondary-basis stage did, echoed on [`ScfResult::guess_basis`].
#[derive(Debug, Clone, PartialEq)]
pub struct GuessBasisReport {
    /// Secondary basis name.
    pub basis: String,
    /// Projection used.
    pub projection: GuessProjection,
    /// Secondary-basis function count.
    pub nbasis_guess: usize,
    /// Target-basis function count.
    pub nbasis_target: usize,
    /// Iterations the secondary SCF took.
    pub iterations: usize,
    /// Secondary-basis total energy (Ha) of the projected state: the
    /// converged energy, or the last iterate's when `converged` is false.
    pub energy: f64,
    /// Whether the secondary SCF converged. An unconverged one is still used
    /// as the guess (its last iterate); see the module policy.
    pub converged: bool,
    /// Why the secondary SCF stopped when it did not converge (its exit
    /// reason and iteration cap); `None` when it converged.
    pub unconverged_reason: Option<String>,
    /// Whether the projected secondary solution seeded the target SCF.
    /// `false` exactly when the secondary SCF did not converge: the target
    /// then starts from its default guess (see the module policy).
    pub used_as_guess: bool,
}

thread_local! {
    /// Secondary SCFs run on this thread (test instrumentation).
    static SECONDARY_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How many secondary-basis SCFs have run on the calling thread. The
/// secondary solve is driven from the thread that called the target solver
/// (never a rayon worker), so a test can difference this around one call to
/// assert how often the secondary stage ran.
#[doc(hidden)]
pub fn secondary_runs_on_this_thread() -> usize {
    SECONDARY_RUNS.with(|c| c.get())
}

/// Which SCF the guess is for; the secondary solve runs the same one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScfKind {
    Rhf,
    Uhf,
    Rohf,
}

/// The projected guess, in the target basis.
#[derive(Debug, Clone)]
pub(crate) struct ProjectedGuess {
    pub projection: GuessProjection,
    /// `P D_α,small Pᵀ` (Fock projection input).
    pub d_a: Array2<f64>,
    /// `P D_β,small Pᵀ`.
    pub d_b: Array2<f64>,
    /// Overlap-projected α MOs: orthonormal occupied block first, then an
    /// orthonormal completion of the target space (n × n). For ROHF the one
    /// MO set: doubly occupied, singly occupied, completion.
    pub c_a: Array2<f64>,
    /// Overlap-projected β MOs (UHF); equal to `c_a` otherwise.
    pub c_b: Array2<f64>,
    pub report: GuessBasisReport,
}

/// What [`build`] produced: a projected guess, or (secondary SCF
/// unconverged) only the report, with the target left to its default guess.
pub(crate) enum GuessOutcome {
    Projected(Box<ProjectedGuess>),
    Declined(GuessBasisReport),
}

impl ProjectedGuess {
    /// The closed-shell starting density for the requested projection.
    pub(crate) fn rhf_density(&self, nocc: usize) -> Array2<f64> {
        match self.projection {
            GuessProjection::Fock => &self.d_a + &self.d_b,
            GuessProjection::Overlap => {
                let c = self.c_a.slice(s![.., ..nocc]);
                c.dot(&c.t()) * 2.0
            }
        }
    }
}

/// `true` when `guess_basis` should drive the guess: it is set and no
/// explicit starting point outranks it.
pub(crate) fn applies(config: &RhfConfig, has_initial_mos: bool) -> bool {
    config.guess_basis.is_some() && config.init_guess_density.is_none() && !has_initial_mos
}

/// The secondary solve's config: the caller's, minus everything that would
/// recurse, outrank the guess or only matters for the target solution.
fn secondary_config(config: &RhfConfig) -> RhfConfig {
    let mut c = config.clone();
    c.guess_basis = None;
    c.init_guess_density = None;
    c.check_stability = false;
    c.scf_stability_descent = false;
    c
}

/// Per-atom ECP core counts of `bs` on `mol`'s real (non-ghost) atoms.
fn ecp_cores(mol: &Molecule, bs: &BasisSet) -> Vec<i32> {
    mol.atoms
        .iter()
        .filter(|a| !a.ghost)
        .map(|a| bs.ecp_for_element(a.z).map_or(0, |d| d.n_core))
        .collect()
}

/// `P = S_bb⁺ S_bs` (target × secondary) and the target overlap `S_bb`.
pub fn projector(
    mol: &Molecule,
    target: &PreparedBasis,
    secondary: &BasisSet,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let cross = ferric_integrals::cabs::cross_overlap(mol, target.basis_set(), secondary)?;
    if cross.nobs != target.nbasis() {
        return Err(FerricError::General(format!(
            "guess_basis: cross-overlap row count {} != target nbasis {}",
            cross.nobs,
            target.nbasis()
        )));
    }
    let s_bb = ferric_integrals::oneelectron::overlap(target);
    let x = crate::rhf::canonical_orthogonalizer(&s_bb)?;
    let p = x.dot(&x.t().dot(&cross.s_or));
    Ok((p, s_bb))
}

/// The `k` highest-occupation natural orbitals of the spin density `d`
/// (metric `s`): an S-orthonormal basis of the occupied space of an
/// idempotent `d`, whatever order the solver stored its MOs in.
fn occupied_natural_orbitals(
    d: &Array2<f64>,
    s: &Array2<f64>,
    k: usize,
) -> Result<Array2<f64>, FerricError> {
    let x = crate::rhf::canonical_orthogonalizer(s)?;
    let sx = s.dot(&x);
    let m = sx.t().dot(d).dot(&sx);
    let (_, v) = m
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("guess_basis natural orbitals: {e}")))?;
    let nv = v.ncols();
    if k > nv {
        return Err(FerricError::General(format!(
            "guess_basis: {k} occupied orbitals exceed the secondary basis dimension {nv}"
        )));
    }
    let top = v.slice(s![.., nv - k..]);
    let mut c = x.dot(&top);
    c.invert_axis(Axis(1));
    Ok(c)
}

/// Löwdin-orthonormalise the columns of `c` in metric `s`:
/// `C (Cᵀ S C)^{-1/2}`. Errors when `Cᵀ S C` is near-singular.
pub fn lowdin_orthonormalize(c: &Array2<f64>, s: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    if c.ncols() == 0 {
        return Ok(c.clone());
    }
    let m = c.t().dot(s).dot(c);
    let (w, v) = m
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("guess_basis Löwdin: {e}")))?;
    let wmin = w.iter().cloned().fold(f64::INFINITY, f64::min);
    if wmin < MIN_PROJECTED_OCC_EIGENVALUE {
        return Err(FerricError::General(format!(
            "guess_basis: the projected occupied space is near-singular in the target basis \
             (smallest eigenvalue of C^T S C = {wmin:.3e} < {MIN_PROJECTED_OCC_EIGENVALUE:e}); \
             the secondary basis is not representable in the target basis"
        )));
    }
    let mut vs = v.clone();
    for (j, &wj) in w.iter().enumerate() {
        vs.column_mut(j).mapv_inplace(|x| x / wj.sqrt());
    }
    Ok(c.dot(&vs.dot(&v.t())))
}

/// Complete an S-orthonormal occupied block `c_occ` (n × k) to an n × n MO
/// set: the S-orthonormal complement within the lindep-filtered target space
/// follows, and any filtered directions are zero columns (as the solvers'
/// own rectangular diagonaliser pads).
fn complete_mo_set(c_occ: &Array2<f64>, s: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let n = s.nrows();
    let k = c_occ.ncols();
    let x = crate::rhf::canonical_orthogonalizer(s)?;
    let m = x.ncols();
    let y = x.t().dot(s).dot(c_occ);
    let q = Array2::<f64>::eye(m) - y.dot(&y.t());
    let (_, v) = q
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("guess_basis completion: {e}")))?;
    let virt = x.dot(&v.slice(s![.., k..]));
    let mut c = Array2::<f64>::zeros((n, n));
    c.slice_mut(s![.., ..k]).assign(c_occ);
    c.slice_mut(s![.., k..m]).assign(&virt);
    Ok(c)
}

/// Overlap projection of one S_small-orthonormal occupied block.
pub fn project_occupied(
    p: &Array2<f64>,
    s_bb: &Array2<f64>,
    c_small_occ: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    lowdin_orthonormalize(&p.dot(c_small_occ), s_bb)
}

/// `P D Pᵀ`.
pub fn project_density(p: &Array2<f64>, d: &Array2<f64>) -> Array2<f64> {
    p.dot(d).dot(&p.t())
}

/// `(nocc_α, nocc_β)` for `mol`'s electron count and multiplicity.
fn occupations(mol: &Molecule) -> Result<(usize, usize), FerricError> {
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    if two_s < 0 || nelec < two_s || (nelec - two_s) % 2 != 0 {
        return Err(FerricError::General(format!(
            "guess_basis: incompatible nelec={nelec} and multiplicity={}",
            mol.multiplicity
        )));
    }
    Ok((
        ((nelec + two_s) / 2) as usize,
        ((nelec - two_s) / 2) as usize,
    ))
}

fn run_secondary(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep_s: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    kind: ScfKind,
) -> Result<ScfResult, FerricError> {
    let op = bounds.op;
    let bounds_s = SchwarzBounds::compute(op, prep_s)?;
    let cfg = secondary_config(config);
    SECONDARY_RUNS.with(|c| c.set(c.get() + 1));
    // The best-effort entry points: an unconverged secondary SCF returns its
    // last iterate rather than an error, because a guess need not converge.
    let r = match kind {
        ScfKind::Rhf => crate::rhf::solve_rhf(ctx, mol, prep_s, op, &bounds_s, &cfg),
        ScfKind::Uhf => crate::uhf::solve_uhf_best_effort(ctx, mol, prep_s, &bounds_s, &cfg, None),
        ScfKind::Rohf => crate::rohf::solve_rohf_best_effort(ctx, mol, prep_s, op, &bounds_s, &cfg),
    }?;
    let finite = |d: &Array2<f64>| d.iter().all(|x| x.is_finite());
    if !finite(&r.density_alpha) || !r.density_beta.as_ref().map_or(true, finite) {
        return Err(FerricError::General(format!(
            "the secondary SCF produced a non-finite density (exit {:?} after {} iterations)",
            r.exit, r.iterations
        )));
    }
    Ok(r)
}

/// Every real atom's element must have shells in the secondary basis.
fn check_elements(mol: &Molecule, bs: &BasisSet) -> Result<(), FerricError> {
    let mut missing: Vec<String> = Vec::new();
    for a in mol.atoms.iter().filter(|a| !a.ghost) {
        let tag = format!("{} (Z={})", a.symbol, a.z);
        if bs.for_element(a.z).map_or(true, <[_]>::is_empty) && !missing.contains(&tag) {
            missing.push(tag);
        }
    }
    if missing.is_empty() {
        return Ok(());
    }
    Err(FerricError::Basis(format!(
        "guess_basis '{}' has no functions for {}; pick a secondary basis that covers \
         every element",
        bs.name,
        missing.join(", ")
    )))
}

/// Converge the secondary-basis SCF of `kind` and project it into `prep`;
/// [`GuessOutcome::Declined`] when the secondary SCF did not converge.
pub(crate) fn build(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    kind: ScfKind,
) -> Result<GuessOutcome, FerricError> {
    let gb = config.guess_basis.as_ref().ok_or_else(|| {
        FerricError::General("guess_basis::build called without guess_basis".into())
    })?;
    let name = gb.basis.name.clone();
    let ctx_err = |what: &str, e: FerricError| {
        FerricError::General(format!("guess_basis '{name}': {what}: {e}"))
    };
    check_elements(mol, &gb.basis)?;
    if ecp_cores(mol, prep.basis_set()) != ecp_cores(mol, &gb.basis) {
        return Err(FerricError::General(format!(
            "guess_basis '{name}': its ECP core electron counts differ from the target basis \
             '{}' on this molecule, so the two SCFs would have different electron counts",
            prep.basis_set().name
        )));
    }
    let prep_s = PreparedBasis::new(mol, &gb.basis).map_err(|e| ctx_err("cannot build it", e))?;
    let (nb, ns) = (prep.nbasis(), prep_s.nbasis());
    if ns > nb && ctx.is_root() {
        eprintln!(
            "WARNING: guess_basis '{name}' has {ns} functions, more than the target basis \
             '{}' ({nb}); the secondary SCF will cost more than it saves",
            prep.basis_set().name
        );
    }
    let small = run_secondary(ctx, mol, &prep_s, bounds, config, kind)
        .map_err(|e| ctx_err("the secondary SCF failed", e))?;
    let report = GuessBasisReport {
        basis: name,
        projection: gb.projection,
        nbasis_guess: ns,
        nbasis_target: nb,
        iterations: small.iterations,
        energy: small.energy,
        converged: small.converged,
        unconverged_reason: (!small.converged).then(|| {
            format!(
                "exit {:?} after {} iterations (max_iter {})",
                small.exit, small.iterations, config.max_iter
            )
        }),
        used_as_guess: small.converged,
    };
    log_report(ctx, &report);
    if !small.converged {
        return Ok(GuessOutcome::Declined(report));
    }
    let (p, s_bb) = projector(mol, prep, &gb.basis)?;
    let s_ss = ferric_integrals::oneelectron::overlap(&prep_s);
    let d_a_s = small.density_alpha.clone();
    let d_b_s = small.density_beta.clone().unwrap_or_else(|| d_a_s.clone());
    let (nocc_a, nocc_b) = occupations(mol)?;

    let (c_a, c_b) = match kind {
        ScfKind::Rhf | ScfKind::Uhf => {
            let ca = project_occupied(
                &p,
                &s_bb,
                &occupied_natural_orbitals(&d_a_s, &s_ss, nocc_a)?,
            )?;
            let cb = project_occupied(
                &p,
                &s_bb,
                &occupied_natural_orbitals(&d_b_s, &s_ss, nocc_b)?,
            )?;
            (complete_mo_set(&ca, &s_bb)?, complete_mo_set(&cb, &s_bb)?)
        }
        ScfKind::Rohf => {
            let c = rohf_overlap_mos(&p, &s_bb, &s_ss, &d_a_s, &d_b_s, nocc_a, nocc_b)?;
            (c.clone(), c)
        }
    };
    Ok(GuessOutcome::Projected(Box::new(ProjectedGuess {
        projection: gb.projection,
        d_a: project_density(&p, &d_a_s),
        d_b: project_density(&p, &d_b_s),
        c_a,
        c_b,
        report,
    })))
}

/// One stderr line per secondary stage (root rank only): the converged
/// summary, or a WARNING that the target falls back to its default guess.
fn log_report(ctx: &ParallelContext, r: &GuessBasisReport) {
    if !ctx.is_root() {
        return;
    }
    match r.unconverged_reason.as_deref() {
        None => eprintln!(
            "[ferric] guess_basis {}: secondary SCF ({} functions) converged in {} \
             iterations, E = {:.10}; projection = {}",
            r.basis,
            r.nbasis_guess,
            r.iterations,
            r.energy,
            r.projection.as_str()
        ),
        Some(why) => eprintln!(
            "WARNING: guess_basis {}: the secondary SCF ({} functions) did NOT converge \
             ({why}, last E = {:.10}). Its last iterate is not used; the target SCF starts \
             from its default guess instead. The report records converged = false.",
            r.basis, r.nbasis_guess, r.energy
        ),
    }
}

/// ROHF overlap projection: doubly occupied block, then the singly occupied
/// block made S-orthogonal to it, then the completion.
fn rohf_overlap_mos(
    p: &Array2<f64>,
    s_bb: &Array2<f64>,
    s_ss: &Array2<f64>,
    d_a_s: &Array2<f64>,
    d_b_s: &Array2<f64>,
    nocc_a: usize,
    nocc_b: usize,
) -> Result<Array2<f64>, FerricError> {
    let docc_s = occupied_natural_orbitals(d_b_s, s_ss, nocc_b)?;
    let socc_s = occupied_natural_orbitals(&(d_a_s - d_b_s), s_ss, nocc_a - nocc_b)?;
    let cd = project_occupied(p, s_bb, &docc_s)?;
    let mut co = p.dot(&socc_s);
    co = &co - &cd.dot(&cd.t().dot(s_bb).dot(&co));
    let co = lowdin_orthonormalize(&co, s_bb)?;
    let mut occ = Array2::<f64>::zeros((s_bb.nrows(), nocc_a));
    occ.slice_mut(s![.., ..nocc_b]).assign(&cd);
    occ.slice_mut(s![.., nocc_b..]).assign(&co);
    complete_mo_set(&occ, s_bb)
}

/// `r` with `report` attached when `r` carries none of its own.
///
/// A stage that replaces the first result (a stability-descent re-solve, a
/// later ladder rung, the exact stage after a DF pre-stage, the cDFT λ loop)
/// starts from a carried density or MOs, which outrank `guess_basis`, so it
/// runs no secondary SCF and reports none. The secondary stage that started
/// the chain is still the one that ran, so its report is carried forward.
pub(crate) fn carry_report(mut r: ScfResult, report: Option<GuessBasisReport>) -> ScfResult {
    if r.guess_basis.is_none() {
        r.guess_basis = report;
    }
    r
}

/// The open-shell guess for a solve that will start many inner SCFs from the
/// same point (cDFT): built once, or `None` when `seeded` (explicit MOs
/// outrank it) or when `guess_basis` is unset.
pub(crate) fn open_shell_guess_once(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    seeded: bool,
) -> Result<Option<OpenShellGuess>, FerricError> {
    if !applies(config, seeded) {
        return Ok(None);
    }
    open_shell_guess(ctx, mol, prep, bounds, config, ScfKind::Uhf).map(Some)
}

/// Open-shell starting MOs from the secondary-basis guess, with the config
/// the target solve should run with (`guess_basis` cleared).
pub(crate) struct OpenShellGuess {
    /// (α, β) MOs (n × n each; for ROHF both are the one MO set), or `None`
    /// when the secondary SCF did not converge and the target should start
    /// from its default guess.
    pub mos: Option<(Array2<f64>, Array2<f64>)>,
    pub report: GuessBasisReport,
    pub config: RhfConfig,
}

impl OpenShellGuess {
    /// The starting MOs as the solvers' `initial_mos` argument.
    pub(crate) fn initial_mos(&self) -> Option<(&Array2<f64>, &Array2<f64>)> {
        self.mos.as_ref().map(|(a, b)| (a, b))
    }
}

/// Converge the secondary SCF of `kind` (UHF or ROHF) and turn it into
/// starting MOs. Overlap: the projected, re-orthonormalised MOs. Fock: the
/// eigenvectors of `F_σ = h + J[D_α + D_β] − K[D_σ]` at the projected
/// densities (Roothaan-combined for ROHF), with `h` the solve's own core
/// Hamiltonian (ECP and external potential included).
pub(crate) fn open_shell_guess(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    kind: ScfKind,
) -> Result<OpenShellGuess, FerricError> {
    let mut cfg = config.clone();
    cfg.guess_basis = None;
    let g = match build(ctx, mol, prep, bounds, config, kind)? {
        GuessOutcome::Projected(g) => *g,
        GuessOutcome::Declined(report) => {
            return Ok(OpenShellGuess {
                mos: None,
                report,
                config: cfg,
            })
        }
    };
    let mos = match g.projection {
        GuessProjection::Overlap => (g.c_a, g.c_b),
        GuessProjection::Fock => {
            fock_guess_mos(ctx, mol, prep, bounds, config, kind, &g.d_a, &g.d_b)?
        }
    };
    Ok(OpenShellGuess {
        mos: Some(mos),
        report: g.report,
        config: cfg,
    })
}

#[allow(clippy::too_many_arguments)]
fn fock_guess_mos(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    kind: ScfKind,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let n = prep.nbasis();
    let s = ferric_integrals::oneelectron::overlap(prep);
    let h = ferric_integrals::oneelectron::hcore_ecp_with_external(
        prep,
        mol,
        prep.basis_set(),
        config.external_potential.as_ref(),
    )?;
    let x = crate::rhf::canonical_orthogonalizer(&s)?;
    let (mut j_a, mut k_a) = (Array2::<f64>::zeros((n, n)), Array2::<f64>::zeros((n, n)));
    let (mut j_b, mut k_b) = (Array2::<f64>::zeros((n, n)), Array2::<f64>::zeros((n, n)));
    let t = config.integral_thresh;
    crate::rhf::build_jk(ctx, prep, bounds, t, d_a, &mut j_a, &mut k_a)?;
    crate::rhf::build_jk(ctx, prep, bounds, t, d_b, &mut j_b, &mut k_b)?;
    let hj = &h + &(&j_a + &j_b);
    let (f_a, f_b) = (&hj - &k_a, &hj - &k_b);
    if kind == ScfKind::Rohf {
        let f = crate::rohf::roothaan_fock(&f_a, &f_b, d_a, d_b, &s);
        let c = crate::driver::diagonalize_rect(&f, &x)?.1;
        return Ok((c.clone(), c));
    }
    let c_a = crate::driver::diagonalize_rect(&f_a, &x)?.1;
    let c_b = crate::driver::diagonalize_rect(&f_b, &x)?.1;
    Ok((c_a, c_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_integrals::operator::Operator;

    /// The overlap projection's full MO set is S_bb-orthonormal, virtuals
    /// included: the completion spans exactly the S-orthogonal complement of
    /// the projected occupied space. The virtual block is load-bearing: UHF
    /// with n_α = n_β mixes the β HOMO with the β LUMO column
    /// (`uhf::mix_beta_homo_lumo`) to break spin symmetry, and the solvers'
    /// rectangular orthogonalisers assume an orthonormal starting set. A
    /// completion that took the wrong eigenvector block (e.g. the occupied
    /// directions, `..m - k`) passes every occupied-only check.
    #[test]
    fn overlap_completion_is_the_orthonormal_complement() {
        let mol = Molecule::parse_xyz(
            "3\n\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n",
            0,
            1,
        )
        .unwrap();
        let prep =
            PreparedBasis::new(&mol, &ferric_core::basis::bundled("cc-pvdz").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let config = RhfConfig {
            guess_basis: Some(GuessBasis::bundled("sto-3g", "overlap").unwrap()),
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let GuessOutcome::Projected(g) =
            build(&ctx, &mol, &prep, &bounds, &config, ScfKind::Uhf).unwrap()
        else {
            panic!("the secondary SCF must converge here");
        };
        let s_bb = ferric_integrals::oneelectron::overlap(&prep);
        let n = prep.nbasis();
        for c in [&g.c_a, &g.c_b] {
            let m = c.t().dot(&s_bb).dot(c);
            let dev = (&m - &Array2::<f64>::eye(n))
                .iter()
                .fold(0.0f64, |a, x| a.max(x.abs()));
            assert!(
                dev < 1e-10,
                "full MO set deviates from S-orthonormal by {dev:e}"
            );
        }
    }

    /// ROHF overlap projection: the whole occupied block (doubly then singly
    /// occupied) is S_bb-orthonormal, not just each block on its own. A
    /// same-basis anchor cannot see a missing docc/socc orthogonalisation
    /// (the projected blocks are already orthogonal there), and neither can a
    /// symmetric molecule whose singly occupied orbitals share no irrep with
    /// the doubly occupied ones (O2 and NH2: the cross overlap is zero by
    /// symmetry). HCO's sigma radical shares a' with the doubly occupied
    /// orbitals, and 6-31G is projected DOWN onto STO-3G, where the
    /// projection is far from an isometry.
    #[test]
    fn rohf_overlap_occupied_block_is_orthonormal() {
        let mol = Molecule::parse_xyz("3\n\nC 0 0 0\nO 1.18 0 0\nH -0.55 0.95 0\n", 0, 2).unwrap();
        let prep =
            PreparedBasis::new(&mol, &ferric_core::basis::bundled("sto-3g").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let config = RhfConfig {
            guess_basis: Some(GuessBasis::bundled("6-31g", "overlap").unwrap()),
            ..Default::default()
        };
        let ctx = ParallelContext::default();
        let GuessOutcome::Projected(g) =
            build(&ctx, &mol, &prep, &bounds, &config, ScfKind::Rohf).unwrap()
        else {
            panic!("the secondary SCF must converge here");
        };
        let s_bb = ferric_integrals::oneelectron::overlap(&prep);
        let occ = g.c_a.slice(s![.., ..8]);
        let m = occ.t().dot(&s_bb).dot(&occ);
        let dev = (&m - &Array2::<f64>::eye(8))
            .iter()
            .fold(0.0f64, |a, x| a.max(x.abs()));
        assert!(
            dev < 1e-12,
            "occupied block deviates from S-orthonormal by {dev:e}"
        );
    }
}
