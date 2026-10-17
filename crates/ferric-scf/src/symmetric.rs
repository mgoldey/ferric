//! Point-group-blocked closed-shell RHF (stage 2 of issue #294).
//!
//! [`solve_rhf_symmetric`] (with `RhfConfig::symmetry = true`) detects the
//! highest Abelian subgroup of the molecule (`ferric_core::symmetry`), builds
//! every integral from the reoriented, symmetrized geometry, and replaces the
//! one global Fock diagonalization by one per irrep. [`solve_rhf`](crate::rhf::solve_rhf)
//! itself refuses the symmetry fields:
//!
//! ```text
//! F_h = U_hᵀ F U_h        U_h = AoRepresentation::so_basis(irrep h)
//! X_h = canonical_orthogonalizer(U_hᵀ S U_h)     (own lindep filter per block)
//! F'_h = (U_h X_h)ᵀ F (U_h X_h) = V_h ε_h V_hᵀ,   C_h = U_h X_h V_h
//! ```
//!
//! The blocks' eigenpairs are merged into one list sorted by orbital energy,
//! so aufbau across irreps, `nocc`, the density build, DIIS and the energy are
//! the unchanged global code. Energies within 1e-10 Ha are ordered by irrep
//! (totally symmetric first) so exact degeneracies between irreps (H2 at long
//! range, sigma_g and sigma_u) resolve deterministically. The MO matrix keeps
//! the historical `(nao, nao)` shape: columns past the independent set are
//! zero with sentinel energy 1e6, labelled `"-"`.
//!
//! # Occupation control
//!
//! `RhfConfig::irrep_nelec` fixes the electron count of the listed irreps
//! (each even; the lowest `n/2` orbitals of the block are occupied, whatever
//! their energy relative to other blocks). Unlisted irreps share the remaining
//! electrons by aufbau. Occupied columns come first in the merged list, so the
//! level shift's virtual projector and the density build see the chosen state.
//!
//! # Entry point and frame
//!
//! The only entry point is [`solve_rhf_symmetric`], which returns a
//! [`SymmetricRhf`], not a bare [`ScfResult`]. Above C1 the SCF runs on the
//! reoriented, symmetrized molecule (`ferric_core::symmetry::detect`), so every
//! AO matrix of the result (MOs, density, Fock) lives in the AO basis of THAT
//! geometry, not of the input. Energies and orbital energies are invariant
//! under the rigid rotation and translation, but they belong to the
//! SYMMETRIZED geometry, which differs from the input by
//! [`SymmetricRhf::max_symmetrization_shift`] (never more than the detection
//! tolerance, 1e-4 Bohr; a geometry that is symmetric to rounding moves by
//! about 1e-12). A distortion just under the tolerance is removed, not
//! computed: water/6-31G with one O-H stretched by 9e-5 Bohr moves atoms by
//! 4.6e-5 Bohr and the energy differs from the input geometry's by 1.2e-9 Ha.
//! AO matrices are not frame-invariant at all.
//!
//! Why a separate type instead of a flag on `ScfResult`. About sixty public
//! functions take `&ScfResult` (MP2, CC, RPA, GW, CI, TDDFT, properties,
//! gradients, Hessians) together with the caller's molecule and basis. A
//! frame-rotated result passed to any of them with the input-frame molecule is
//! a silent wrong answer (RI-MP2 on a blocked NH3 gave -7.89 Ha of correlation
//! instead of -0.189), and a guard would have to be hand-added to each one and
//! to every future one. With a distinct type none of them can receive it by
//! accident: obtaining the plain result takes the explicit
//! [`SymmetricRhf::scf_in_frame`], whose documentation names the molecule to
//! pair it with ([`SymmetricRhf::molecule`]), and `solve_rhf` itself refuses
//! `RhfConfig::symmetry`, so no internal re-solve (Hessian, optimizer, ladder)
//! can return a frame result either.
//!
//! Rotating the result back to the input frame was rejected. Rotating back
//! would be exact for the symmetrized geometry, but that geometry differs from
//! the input by `max_symmetrization_shift`, so the rotated matrices would still
//! not belong to the input geometry's integrals, only look as if they did (a
//! quiet error of that order, unlike a loud refusal); and ferric has no
//! rotated-AO (Wigner-D) machinery for real solid harmonics under an arbitrary
//! rotation. Post-HF methods run on the frame molecule instead.
//!
//! C1 runs the original solver on the original inputs (bit-identical to
//! `solve_rhf`), and its [`SymmetricRhf::molecule`] is the input molecule.
//!
//! # Degenerate orbitals across irreps
//!
//! Orbital energies closer than 1e-10 Ha form a cluster ordered by irrep
//! (totally symmetric first), so the order is deterministic. Aufbau stops
//! inside a cluster only if the shell is partially filled (for example an E
//! pair split into two 1-dimensional irreps of an Abelian subgroup, with one
//! electron pair): the pair goes to the first irrep of the cluster and the
//! density is then not the symmetric average of the degenerate shell. That is
//! a symmetry-broken-in-labelling state, the same choice other Abelian-subgroup
//! codes make; use `irrep_nelec` to choose explicitly.
//!
//! # Refused (typed error, never silently ignored)
//!
//! MOM, smearing, fractional occupations, Newton / AH / TRAH / AURORA,
//! stability analysis and stability descent, constraints (cDFT), a user
//! initial density (it would be in the input frame), Kohn-Sham functionals
//! (the Becke-Lebedev grid is not symmetric, a maintainer decision pending),
//! external potentials, COSMO, PCM, polarizable embedding, the periodic
//! injected path, and UHF / ROHF. Compatible: DIIS, level shift, damping,
//! direct J/K, LinK, density-fitted J/K and COSX (whose grid breaks symmetry
//! only at grid-error level; off-block elements of F are dropped by the
//! blocking).
//!
//! The input molecule must already carry its ECP assignment
//! (`Molecule::apply_ecp`) when detection runs: species identity includes the
//! ECP core count.

use crate::result::ScfResult;
use crate::rhf::{canonical_orthogonalizer, RhfConfig};
use crate::screening::{SchwarzBounds, ScreeningKind};
use ferric_core::parallel::ParallelContext;
use ferric_core::symmetry::{
    self, irreps, require_no_external_potential, AbelianGroup, AoRepresentation, Irrep,
    DEFAULT_SYMMETRY_TOL,
};
use ferric_core::{mol::Molecule, FerricError};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::operator::Operator;
use ndarray::Array2;
use ndarray_linalg::Eigh;
use std::cell::RefCell;

/// Orbital energies closer than this are one degenerate cluster: ordered by
/// irrep instead of by roundoff.
const DEGENERACY_TOL: f64 = 1e-10;
/// Label of a padded (linearly dependent) MO column.
const PAD_IRREP: usize = usize::MAX;

/// Symmetry data one SCF run needs: the AO representation, the irreps, the
/// occupation request, and a slot where the solver records the irrep of each
/// MO column of its latest diagonalization.
pub(crate) struct SymmetrySpec {
    rep: AoRepresentation,
    irreps: Vec<Irrep>,
    /// Requested electrons per irrep (index-aligned with `irreps`).
    nelec: Vec<Option<usize>>,
    last_labels: RefCell<Vec<usize>>,
}

impl SymmetrySpec {
    pub(crate) fn new(
        rep: AoRepresentation,
        ops: &[usize],
        irrep_nelec: Option<&[(String, usize)]>,
        total_nelec: usize,
    ) -> Result<Self, FerricError> {
        let irreps = irreps(ops)?;
        let nelec = resolve_irrep_nelec(&rep, &irreps, irrep_nelec, total_nelec)?;
        Ok(Self {
            rep,
            irreps,
            nelec,
            last_labels: RefCell::new(Vec::new()),
        })
    }

    /// Irrep label of each MO column of the latest diagonalization.
    pub(crate) fn column_labels(&self) -> Vec<String> {
        self.last_labels
            .borrow()
            .iter()
            .map(|&h| match self.irreps.get(h) {
                Some(ir) => ir.label.to_string(),
                None => "-".to_string(),
            })
            .collect()
    }
}

fn resolve_irrep_nelec(
    rep: &AoRepresentation,
    irreps: &[Irrep],
    request: Option<&[(String, usize)]>,
    total: usize,
) -> Result<Vec<Option<usize>>, FerricError> {
    let mut out = vec![None; irreps.len()];
    let Some(req) = request else { return Ok(out) };
    let dims = rep.irrep_dimensions()?;
    for (label, n) in req {
        let h = irreps
            .iter()
            .position(|ir| ir.label == label)
            .ok_or_else(|| {
                let known: Vec<_> = irreps.iter().map(|ir| ir.label).collect();
                FerricError::General(format!(
                    "irrep_nelec: unknown irrep {label:?}; this point group has {known:?}"
                ))
            })?;
        if out[h].is_some() {
            return Err(FerricError::General(format!(
                "irrep_nelec: irrep {label:?} listed twice"
            )));
        }
        if n % 2 != 0 {
            return Err(FerricError::General(format!(
                "irrep_nelec: {label} has {n} electrons; closed-shell RHF needs an even count"
            )));
        }
        if n / 2 > dims[h].1 {
            return Err(FerricError::General(format!(
                "irrep_nelec: {label} holds {n} electrons but has only {} basis functions",
                dims[h].1
            )));
        }
        out[h] = Some(*n);
    }
    let fixed: usize = out.iter().flatten().sum();
    let all_listed = out.iter().all(Option::is_some);
    if fixed > total || (all_listed && fixed != total) {
        return Err(FerricError::General(format!(
            "irrep_nelec sums to {fixed} electrons but the molecule has {total}"
        )));
    }
    Ok(out)
}

/// The SCF's Fock diagonalizer: the historical global canonical-orthogonal
/// diagonalization, or the per-irrep blocked one.
pub(crate) struct FockDiag<'a> {
    x: &'a Array2<f64>,
    blocked: Option<BlockedDiag<'a>>,
}

impl<'a> FockDiag<'a> {
    pub(crate) fn new(
        sym: Option<&'a SymmetrySpec>,
        s: &Array2<f64>,
        x: &'a Array2<f64>,
        nocc: usize,
    ) -> Result<Self, FerricError> {
        let blocked = sym.map(|sp| BlockedDiag::new(sp, s, nocc)).transpose()?;
        Ok(Self { x, blocked })
    }

    pub(crate) fn run(&self, f: &Array2<f64>) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
        match &self.blocked {
            Some(b) => b.diagonalize(f),
            None => crate::driver::diagonalize_rect(f, self.x),
        }
    }
}

/// Per-iteration blocked diagonalizer: the per-irrep `U_h X_h` transforms
/// built once from the overlap matrix.
pub(crate) struct BlockedDiag<'a> {
    spec: &'a SymmetrySpec,
    /// `U_h X_h`, shape `(nao, m_h)`; `m_h` may be 0.
    xu: Vec<Array2<f64>>,
    nocc: usize,
}

impl<'a> BlockedDiag<'a> {
    pub(crate) fn new(
        spec: &'a SymmetrySpec,
        s: &Array2<f64>,
        nocc: usize,
    ) -> Result<Self, FerricError> {
        let mut xu = Vec::with_capacity(spec.irreps.len());
        for ir in &spec.irreps {
            let u = spec.rep.so_basis(ir)?;
            if u.ncols() == 0 {
                xu.push(Array2::zeros((s.nrows(), 0)));
                continue;
            }
            let s_h = u.t().dot(s).dot(&u);
            let x_h = canonical_orthogonalizer(&s_h)?;
            xu.push(u.dot(&x_h));
        }
        Ok(Self { spec, xu, nocc })
    }

    /// Same contract as `rhf::diagonalize`: `(eps, C)` with `C` of shape
    /// `(nao, nao)` padded by zero columns of energy 1e6, columns ordered by
    /// the occupation rule. Records each column's irrep on the spec.
    pub(crate) fn diagonalize(
        &self,
        f: &Array2<f64>,
    ) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
        let nao = f.nrows();
        let mut eps_all: Vec<f64> = Vec::new();
        let mut irrep_all: Vec<usize> = Vec::new();
        let mut cols: Vec<Array2<f64>> = Vec::new();
        let mut origin: Vec<(usize, usize)> = Vec::new();
        for (h, xu) in self.xu.iter().enumerate() {
            if xu.ncols() == 0 {
                continue;
            }
            let (e, v) = with_blas_threads(opt_in_blas_threads(), || {
                xu.t().dot(f).dot(xu).eigh(ndarray_linalg::UPLO::Upper)
            })
            .map_err(|e| FerricError::Lapack(format!("F diag (irrep block {h}): {e}")))?;
            cols.push(xu.dot(&v));
            for (k, &ek) in e.iter().enumerate() {
                eps_all.push(ek);
                irrep_all.push(h);
                origin.push((cols.len() - 1, k));
            }
        }
        let order = occupation_order(&eps_all, &irrep_all, &self.spec.nelec, self.nocc)?;
        let mut eps = vec![1e6; nao];
        let mut c = Array2::<f64>::zeros((nao, nao));
        let mut labels = vec![PAD_IRREP; nao];
        for (col, &i) in order.iter().enumerate() {
            let (b, k) = origin[i];
            eps[col] = eps_all[i];
            labels[col] = irrep_all[i];
            c.column_mut(col).assign(&cols[b].column(k));
        }
        self.spec.last_labels.replace(labels);
        Ok((eps, c))
    }
}

/// Indices into the merged eigenpair list in MO-column order: the occupied
/// orbitals (per the irrep request, else lowest `nocc` overall) in energy
/// order, then the virtuals in energy order. `eps` ascends within each irrep.
pub(crate) fn occupation_order(
    eps: &[f64],
    irrep: &[usize],
    request: &[Option<usize>],
    nocc: usize,
) -> Result<Vec<usize>, FerricError> {
    let all: Vec<usize> = (0..eps.len()).collect();
    if request.iter().all(Option::is_none) {
        return Ok(sorted_clusters(eps, irrep, all));
    }
    // Each listed irrep must keep at least n/2 eigenpairs after its own
    // linear-dependence filter; otherwise the requested state does not exist.
    for (h, req) in request.iter().enumerate() {
        let have = irrep.iter().filter(|&&g| g == h).count();
        if let Some(n) = req {
            if n / 2 > have {
                return Err(FerricError::General(format!(
                    "irrep_nelec asks for {n} electrons in irrep index {h}, but only {have} \
                     linearly independent functions survive in that block"
                )));
            }
        }
    }
    let mut rank_in_irrep = vec![0usize; request.len()];
    let mut occupied = vec![false; eps.len()];
    let mut pool = Vec::new();
    let mut n_forced = 0;
    for i in 0..eps.len() {
        let h = irrep[i];
        let r = rank_in_irrep[h];
        rank_in_irrep[h] += 1;
        match request[h] {
            Some(n) if r < n / 2 => {
                occupied[i] = true;
                n_forced += 1;
            }
            Some(_) => {}
            None => pool.push(i),
        }
    }
    let n_free = nocc.checked_sub(n_forced).ok_or_else(|| {
        FerricError::General("irrep_nelec requests more electrons than the molecule has".into())
    })?;
    let pool = sorted_clusters(eps, irrep, pool);
    if n_free > pool.len() {
        return Err(FerricError::General(format!(
            "irrep_nelec leaves {n_free} orbitals to fill by aufbau but the unlisted irreps \
             have only {} functions",
            pool.len()
        )));
    }
    for &i in &pool[..n_free] {
        occupied[i] = true;
    }
    let (occ, virt): (Vec<usize>, Vec<usize>) = all.into_iter().partition(|&i| occupied[i]);
    let mut out = sorted_clusters(eps, irrep, occ);
    out.extend(sorted_clusters(eps, irrep, virt));
    Ok(out)
}

/// Sort by energy, then reorder each chain of energies closer than
/// [`DEGENERACY_TOL`] by (irrep, energy). Sorting a chain by a key that is not
/// a strict weak order on the whole list is avoided: the energy sort is exact.
fn sorted_clusters(eps: &[f64], irrep: &[usize], mut ids: Vec<usize>) -> Vec<usize> {
    ids.sort_by(|&a, &b| eps[a].total_cmp(&eps[b]));
    let mut start = 0;
    while start < ids.len() {
        let mut end = start + 1;
        while end < ids.len() && eps[ids[end]] - eps[ids[end - 1]] < DEGENERACY_TOL {
            end += 1;
        }
        ids[start..end].sort_by(|&a, &b| irrep[a].cmp(&irrep[b]).then(eps[a].total_cmp(&eps[b])));
        start = end;
    }
    ids
}

/// Typed refusal of every feature the blocked solver does not support.
pub(crate) fn validate_config(config: &RhfConfig) -> Result<(), FerricError> {
    let no = |what: &str, why: &str| -> Result<(), FerricError> {
        Err(FerricError::General(format!(
            "RhfConfig.symmetry cannot be combined with {what}: {why}"
        )))
    };
    if config.mom_after_iter > 0 {
        return no("MOM", "it pins orbitals by overlap, not by irrep");
    }
    if config.smearing_sigma.is_some() || config.fractional_occ {
        return no("fractional occupation", "occupations are integer per irrep");
    }
    if config.newton_trigger > 0.0 || config.ah_trigger > 0.0 || config.trah_trigger.is_some() {
        return no(
            "Newton/AH/TRAH steps",
            "second-order steps rotate orbitals across irreps",
        );
    }
    if config.aurora.enabled {
        return no("AURORA", "it feeds the second-order step");
    }
    if config.check_stability || config.scf_stability_descent {
        return no(
            "stability analysis or descent",
            "the orbital Hessian is unblocked",
        );
    }
    if !config.constraints.is_empty() {
        return no(
            "constraints (cDFT)",
            "constraint weights are not symmetrized",
        );
    }
    if config.init_guess_density.is_some() {
        return no(
            "init_guess_density",
            "the density would be in the input frame, the SCF runs in the symmetry frame",
        );
    }
    if config.xc.is_some() {
        return no(
            "a Kohn-Sham functional",
            "the Becke-Lebedev grid is not symmetric (RKS blocking is a later stage)",
        );
    }
    if config.cosmo.is_some() || config.pcm.is_some() || config.polarizable.is_some() {
        return no(
            "COSMO/PCM/polarizable embedding",
            "cavities and sites are not part of the symmetry detection",
        );
    }
    require_no_external_potential(config.external_potential.as_ref())
}

/// A symmetry-blocked RHF solution and the frame its AO matrices live in.
#[derive(Debug, Clone)]
pub struct SymmetricRhf {
    scf: ScfResult,
    group: String,
    orbital_irreps: Vec<String>,
    molecule: Molecule,
    rotation: [[f64; 3]; 3],
    origin: [f64; 3],
    max_symmetrization_shift: f64,
}

impl SymmetricRhf {
    /// Largest atom displacement (Bohr) made by symmetrizing the input
    /// geometry; 0 for C1, at most the detection tolerance otherwise.
    pub fn max_symmetrization_shift(&self) -> f64 {
        self.max_symmetrization_shift
    }

    /// Total energy of the SYMMETRIZED geometry ([`molecule`](Self::molecule)),
    /// which differs from the input geometry's energy by the effect of
    /// [`max_symmetrization_shift`](Self::max_symmetrization_shift).
    pub fn energy(&self) -> f64 {
        self.scf.energy
    }

    /// Schoenflies symbol of the Abelian group used for blocking.
    pub fn group(&self) -> &str {
        &self.group
    }

    /// Irrep label of each MO column (`"-"` for padded, linearly dependent
    /// columns).
    pub fn orbital_irreps(&self) -> &[String] {
        &self.orbital_irreps
    }

    /// Electrons per irrep in the converged state, as `(label, n)` in order of
    /// first appearance among the MO columns, listing only occupied irreps
    /// plus those with virtuals (count 0). Occupied columns come first, so
    /// this counts two electrons for each of the first `nelec / 2` columns.
    pub fn irrep_occupations(&self) -> Vec<(String, usize)> {
        let nocc = self.molecule.nelec().max(0) as usize / 2;
        let mut out: Vec<(String, usize)> = Vec::new();
        for (k, lab) in self.orbital_irreps.iter().enumerate() {
            if lab == "-" {
                continue;
            }
            let add = if k < nocc { 2 } else { 0 };
            match out.iter_mut().find(|(l, _)| l == lab) {
                Some(e) => e.1 += add,
                None => out.push((lab.clone(), add)),
            }
        }
        out
    }

    /// The molecule the integrals were built from: the reoriented, symmetrized
    /// geometry (the input molecule itself for C1).
    pub fn molecule(&self) -> &Molecule {
        &self.molecule
    }

    /// `r_sym = rotation * (r_in - origin)`; identity and zero for C1.
    pub fn frame_transform(&self) -> ([[f64; 3]; 3], [f64; 3]) {
        (self.rotation, self.origin)
    }

    /// The plain SCF result, whose AO matrices are expressed in the basis of
    /// [`molecule`](Self::molecule). Pass that molecule (and a `PreparedBasis`
    /// built from it) to every consumer; the input geometry will give wrong
    /// answers.
    pub fn scf_in_frame(&self) -> &ScfResult {
        &self.scf
    }

    /// Consume into the plain result; see [`scf_in_frame`](Self::scf_in_frame).
    pub fn into_scf_in_frame(self) -> ScfResult {
        self.scf
    }
}

/// Refuse `RhfConfig.symmetry` in solvers that do not implement it.
pub(crate) fn require_symmetry_off(config: &RhfConfig, solver: &str) -> Result<(), FerricError> {
    if config.symmetry || config.irrep_nelec.is_some() {
        return Err(FerricError::General(format!(
            "{solver}: point-group symmetry (RhfConfig.symmetry / irrep_nelec) is only available \
             through ferric_scf::symmetric::solve_rhf_symmetric (closed-shell RHF); its result \
             lives in the symmetry frame and has a distinct type"
        )));
    }
    Ok(())
}

/// Closed-shell RHF with point-group blocking (`config.symmetry` must be set).
/// `mol` must carry its ECP assignment. See the module docs for the frame.
pub fn solve_rhf_symmetric(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
) -> Result<SymmetricRhf, FerricError> {
    if !config.symmetry {
        return Err(FerricError::General(
            "solve_rhf_symmetric needs RhfConfig.symmetry = true (irrep_nelec requires it)".into(),
        ));
    }
    validate_config(config)?;
    let det = symmetry::detect(mol, DEFAULT_SYMMETRY_TOL)?;
    if det.group == AbelianGroup::C1 {
        let scf = solve_c1(ctx, mol, prep, op, bounds, config)?;
        let labels = scf
            .eps_alpha
            .iter()
            .map(|&e| if e >= 1e5 { "-" } else { "A" }.to_string())
            .collect();
        return Ok(SymmetricRhf {
            scf,
            group: "C1".into(),
            orbital_irreps: labels,
            molecule: mol.clone(),
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            origin: [0.0; 3],
            max_symmetrization_shift: 0.0,
        });
    }
    let sym_mol = det.molecule.clone();
    let prep_sym = PreparedBasis::new(&sym_mol, prep.basis_set())?;
    let bounds_sym = frame_bounds(op, &prep_sym, bounds)?;
    let rep = AoRepresentation::new(&det, &sym_mol, prep.basis_set())?;
    let spec = SymmetrySpec::new(
        rep,
        &det.ops,
        config.irrep_nelec.as_deref(),
        sym_mol.nelec().max(0) as usize,
    )?;
    let scf =
        crate::rhf::solve_rhf_blocked(ctx, &sym_mol, &prep_sym, op, &bounds_sym, config, &spec)?;
    Ok(SymmetricRhf {
        scf,
        group: det.group.name().to_string(),
        orbital_irreps: spec.column_labels(),
        molecule: sym_mol,
        rotation: det.rotation,
        origin: det.origin,
        max_symmetrization_shift: det.max_symmetrization_shift,
    })
}

/// The screening flavour the caller's bounds carry (the two tables are
/// mutually exclusive, see [`SchwarzBounds::compute_for_screening`]).
pub(crate) fn screening_kind_of(bounds: &SchwarzBounds) -> ScreeningKind {
    if bounds.csb_m.is_some() {
        ScreeningKind::Csb
    } else if bounds.csam_x.is_some() {
        ScreeningKind::Csam
    } else {
        ScreeningKind::Schwarz
    }
}

/// Bounds for the symmetry-frame basis with the SAME screening flavour as the
/// caller's. Rebuilding plain Schwarz would not change the energy (screening
/// only drops quartets below the threshold) but would silently discard a
/// requested CSB/CSAM screen.
pub(crate) fn frame_bounds(
    op: Operator,
    prep_sym: &PreparedBasis,
    input: &SchwarzBounds,
) -> Result<SchwarzBounds, FerricError> {
    SchwarzBounds::compute_for_screening(op, prep_sym, screening_kind_of(input))
}

/// C1: the unmodified solver on the unmodified inputs.
fn solve_c1(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
) -> Result<ScfResult, FerricError> {
    if let Some(req) = config.irrep_nelec.as_deref() {
        let n = mol.nelec().max(0) as usize;
        if req.len() != 1 || req[0].0 != "A" || req[0].1 != n {
            return Err(FerricError::General(format!(
                "irrep_nelec {req:?} does not fit the detected point group C1 (irrep \"A\", {n} electrons)"
            )));
        }
    }
    crate::rhf::solve_rhf_once(ctx, mol, prep, op, bounds, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_bounds_keep_the_callers_screening_kind() {
        let mol = ferric_core::mol::Molecule::parse_xyz(
            "3\nw\nO 0 0 0.1\nH 0 0.75 -0.47\nH 0 -0.75 -0.47\n",
            0,
            1,
        )
        .unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        for kind in [
            ScreeningKind::Schwarz,
            ScreeningKind::Csb,
            ScreeningKind::Csam,
        ] {
            let input = SchwarzBounds::compute_for_screening(op, &prep, kind).unwrap();
            assert_eq!(screening_kind_of(&input), kind);
            let out = frame_bounds(op, &prep, &input).unwrap();
            assert_eq!(screening_kind_of(&out), kind, "{kind:?} was dropped");
        }
    }

    #[test]
    fn plain_aufbau_is_a_global_energy_sort() {
        let eps = [0.3, -1.0, 0.1, -0.5];
        let irrep = [0, 1, 0, 1];
        assert_eq!(
            occupation_order(&eps, &irrep, &[None, None], 2).unwrap(),
            vec![1, 3, 2, 0]
        );
    }

    #[test]
    fn degenerate_energies_order_by_irrep_not_by_roundoff() {
        // Index 0 is in irrep 1 and 5e-11 BELOW index 1 (irrep 0): one cluster,
        // so irrep 0 comes first.
        let eps = [0.5, 0.5 + 5e-11, 2.0];
        let irrep = [1, 0, 0];
        assert_eq!(
            occupation_order(&eps, &irrep, &[None, None], 1).unwrap(),
            vec![1, 0, 2]
        );
        // Beyond the tolerance the energy decides.
        let eps = [0.5, 0.5 + 5e-9, 2.0];
        assert_eq!(
            occupation_order(&eps, &irrep, &[None, None], 1).unwrap(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn occupation_is_by_energy_across_irreps_not_by_irrep_order() {
        // Irrep 0 has the lowest orbital but irrep 1 holds two below irrep 0's second.
        let eps = [-2.0, 4.0, -1.0, -0.5];
        let irrep = [0, 0, 1, 1];
        let order = occupation_order(&eps, &irrep, &[None, None], 3).unwrap();
        assert_eq!(&order[..3], &[0, 2, 3]);
    }

    #[test]
    fn irrep_request_overrides_aufbau_and_puts_occupied_first() {
        // Irrep 0 would win by aufbau; asking for zero electrons there forces
        // the occupation into irrep 1 and the (low) irrep-0 orbitals to virtual.
        let eps = [-3.0, -2.0, 0.0, 1.0];
        let irrep = [0, 0, 1, 1];
        let order = occupation_order(&eps, &irrep, &[Some(0), None], 2).unwrap();
        assert_eq!(order, vec![2, 3, 0, 1]);
        // A forced pair in irrep 1 plus aufbau in the unlisted irrep 0.
        let order = occupation_order(&eps, &irrep, &[None, Some(2)], 2).unwrap();
        assert_eq!(&order[..2], &[0, 2]);
    }

    #[test]
    fn request_exceeding_surviving_block_dimension_is_an_error() {
        // Irrep 1 lost functions to linear dependence: 1 eigenpair, 4 electrons asked.
        let eps = [-3.0, -2.0, 0.0];
        let irrep = [0, 0, 1];
        let err = occupation_order(&eps, &irrep, &[None, Some(4)], 3).unwrap_err();
        assert!(err.to_string().contains("linearly independent"), "{err}");
        // The same request within the surviving dimension is fine.
        assert!(occupation_order(&eps, &irrep, &[None, Some(2)], 2).is_ok());
    }
}
