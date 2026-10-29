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
//! * The secondary SCF must converge; otherwise the target solve returns an
//!   error naming the secondary basis. There is no fallback to MINAO.
//! * An element missing from the secondary basis is an error, as is an ECP
//!   core count that differs between the two bases (the electron counts would
//!   differ).
//! * A secondary basis with more functions than the target is allowed, with
//!   a warning.
//! * An explicit starting point (`init_guess_density`, or caller-supplied
//!   initial MOs) takes precedence over `guess_basis`, exactly as it already
//!   takes precedence over `use_sad_guess`.
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
    /// Converged secondary-basis total energy (Ha).
    pub energy: f64,
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
    let r = match kind {
        ScfKind::Rhf => crate::rhf::solve_rhf(ctx, mol, prep_s, op, &bounds_s, &cfg),
        ScfKind::Uhf => crate::uhf::solve_uhf(ctx, mol, prep_s, &bounds_s, &cfg),
        ScfKind::Rohf => crate::rohf::solve_rohf(ctx, mol, prep_s, op, &bounds_s, &cfg),
    };
    match r {
        Ok(r) if r.converged => Ok(r),
        Ok(r) => Err(FerricError::ScfConvergence {
            iterations: r.iterations,
            last_energy: r.energy,
        }),
        Err(e) => Err(e),
    }
}

/// Converge the secondary-basis SCF of `kind` and project it into `prep`.
pub(crate) fn build(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    kind: ScfKind,
) -> Result<ProjectedGuess, FerricError> {
    let gb = config.guess_basis.as_ref().ok_or_else(|| {
        FerricError::General("guess_basis::build called without guess_basis".into())
    })?;
    let name = gb.basis.name.clone();
    let ctx_err = |what: &str, e: FerricError| {
        FerricError::General(format!("guess_basis '{name}': {what}: {e}"))
    };
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
        .map_err(|e| ctx_err("the secondary SCF did not converge", e))?;
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
    let report = GuessBasisReport {
        basis: name,
        projection: gb.projection,
        nbasis_guess: ns,
        nbasis_target: nb,
        iterations: small.iterations,
        energy: small.energy,
    };
    if ctx.is_root() {
        eprintln!(
            "[ferric] guess_basis {}: secondary SCF ({} functions) converged in {} iterations, \
             E = {:.10}; projection = {}",
            report.basis,
            ns,
            report.iterations,
            report.energy,
            report.projection.as_str()
        );
    }
    Ok(ProjectedGuess {
        projection: gb.projection,
        d_a: project_density(&p, &d_a_s),
        d_b: project_density(&p, &d_b_s),
        c_a,
        c_b,
        report,
    })
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

/// Open-shell starting MOs from the secondary-basis guess, with the config
/// the target solve should run with (`guess_basis` cleared).
pub(crate) struct OpenShellGuess {
    /// α MOs (n × n); for ROHF the one MO set.
    pub c_a: Array2<f64>,
    /// β MOs (UHF); equal to `c_a` for ROHF.
    pub c_b: Array2<f64>,
    pub report: GuessBasisReport,
    pub config: RhfConfig,
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
    let g = build(ctx, mol, prep, bounds, config, kind)?;
    let (c_a, c_b) = match g.projection {
        GuessProjection::Overlap => (g.c_a, g.c_b),
        GuessProjection::Fock => {
            fock_guess_mos(ctx, mol, prep, bounds, config, kind, &g.d_a, &g.d_b)?
        }
    };
    let mut cfg = config.clone();
    cfg.guess_basis = None;
    Ok(OpenShellGuess {
        c_a,
        c_b,
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
