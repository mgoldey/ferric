//! Point-group-blocked closed-shell RHF and RKS (issue #294).
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
//! # Kohn-Sham (RKS)
//!
//! With `RhfConfig::xc` set the blocked solve is RKS, with NO Fock or density
//! symmetrisation. The KS grid is built from the reoriented molecule, and in
//! that frame it is exactly invariant under every detected operation: each
//! Lebedev rule is an Oh orbit placed at the atoms without rotation, the
//! canonical-frame operations `diag(±1, ±1, ±1)` lie in Oh, and the radial
//! grids and Becke weights depend only on the element and are symmetric under
//! the atom permutation. So `V_xc` of a symmetric density commutes with the
//! operations to roundoff (measured about 3e-15 relative), and blocking
//! discards nothing but that roundoff and the batch screen's 1e-13-level
//! asymmetry. The grid is NOT rotation invariant, so a rotated input gives an
//! energy that differs from the unblocked input-frame energy at the
//! grid-error level (1e-6 to 1e-3 Ha measured at 75x110); against the
//! unblocked solve of the symmetry-frame molecule it agrees to roundoff.
//! RI-J (the KS default) and exact J both block exactly.
//!
//! C3 and C6 are not Lebedev symmetries, so degenerate partners of the full
//! (non-Abelian) group can split on the grid (NH3 e by 3e-6 to 4e-5 Ha, benzene
//! e1g by 6e-5 Ha at 75x110), far above the 1e-10 clustering tolerance. A
//! partially filled degenerate shell would then be occupied by grid noise.
//! Under KS the solve is therefore refused when, among the irreps that
//! `irrep_nelec` does not list (all of them without a request), the highest
//! occupied and lowest unoccupied orbitals of the FIRST blocked
//! diagonalization (the Fock matrix of the guess, symmetric under the full
//! point group) lie in different irreps within `KS_SHELL_SPLIT_TOL` (1e-3 Ha).
//! Stretched H2 (sigma_g and sigma_u degenerate) and singlet O2 (one pair in
//! pi_g, also with only Ag listed) are such cases; list one of the shell's
//! irreps in `irrep_nelec`. Closed degenerate shells are unaffected. The
//! check looks at the starting Fock matrix because a partially filled shell
//! that the SCF itself splits (O2's pi_g ends 0.05 Ha apart) would pass a
//! check of the converged state; at convergence the same test only warns.
//! The tolerance is meaningful only on a fine enough grid, so blocked KS
//! refuses a main grid below 110 angular points (`MIN_KS_ANGULAR`, with the
//! measured splittings).
//!
//! # Refused (typed error, never silently ignored)
//!
//! MOM, smearing, fractional occupations, Newton / AH / TRAH / AURORA,
//! stability analysis and stability descent, constraints (cDFT), a user
//! initial density (it would be in the input frame), COSX exchange (its
//! seminumerical grid and final-grid pass are not wired to the symmetry
//! frame), external potentials, COSMO, PCM, polarizable embedding, the periodic
//! injected path. Open shells have their own entry points
//! ([`crate::symmetric_open`]), which share this module's frame, blocking and
//! refusals and also refuse Kohn-Sham. Compatible: DIIS, level shift, damping,
//! direct J/K, LinK, density-fitted J/K, and Kohn-Sham functionals (RKS, see
//! above).
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
use std::cell::{Cell, RefCell};

/// Orbital energies closer than this are one degenerate cluster: ordered by
/// irrep instead of by roundoff.
const DEGENERACY_TOL: f64 = 1e-10;
/// Label of a padded (linearly dependent) MO column.
const PAD_IRREP: usize = usize::MAX;
/// Under Kohn-Sham, a HOMO and LUMO of different irreps closer than this
/// (Hartree) are taken as a split degenerate shell. Only sound together with
/// `MIN_KS_ANGULAR`; see there for the measured splittings.
const KS_SHELL_SPLIT_TOL: f64 = 1e-3;
/// Smallest main-grid Lebedev order blocked KS accepts. The grid splits
/// degenerate partners of non-Lebedev rotations (C3, C6), and on a coarse
/// grid the split exceeds `KS_SHELL_SPLIT_TOL`, so the split-shell refusal
/// would no longer see the shell. MEASURED, largest split of the 12
/// degenerate pairs of exact-D6h benzene PBE/STO-3G (75 radial unless
/// stated): 590: 1.6e-7, 302: 8.5e-7, 194: 6.6e-6, 110: 7.9e-5 (also with
/// NWChem pruning), 50: 7.0e-4, 40x26: 7.4e-3, 14: 2.6e-2, 20x6: 0.17 Ha
/// (the rounded `testdata` benzene adds a geometric split up to 3e-4).
/// 110 keeps a factor 12 below the tolerance; 50 does not.
const MIN_KS_ANGULAR: usize = 110;

/// Symmetry data one SCF run needs: the AO representation, the irreps, the
/// occupation request, and a slot where the solver records the irrep of each
/// MO column of its latest diagonalization.
///
/// Occupied orbitals are organized in ORDERED GROUPS whose sizes are `totals`:
/// closed-shell RHF and each UHF spin have one group (the occupied set), ROHF
/// has two (core, then open). A request fixes, per irrep, how many orbitals of
/// each group live there.
pub(crate) struct SymmetrySpec {
    pub(crate) rep: AoRepresentation,
    pub(crate) irreps: Vec<Irrep>,
    /// Orbitals per occupied group.
    totals: Vec<usize>,
    /// Requested orbitals per group, per irrep (index-aligned with `irreps`).
    request: Vec<Option<Vec<usize>>>,
    last_labels: RefCell<Vec<usize>>,
    /// Pending Kohn-Sham split-shell check of the FIRST blocked
    /// diagonalization (see `refuse_split_ks_shell`); cleared once run.
    split_shell_check: Cell<bool>,
}

impl SymmetrySpec {
    /// Closed-shell RHF: one group of `total_nelec / 2` orbitals; the request
    /// counts ELECTRONS (even) per irrep.
    pub(crate) fn new(
        rep: AoRepresentation,
        ops: &[usize],
        irrep_nelec: Option<&[(String, usize)]>,
        total_nelec: usize,
    ) -> Result<Self, FerricError> {
        let irreps = irreps(ops)?;
        let nelec = resolve_irrep_nelec(&rep, &irreps, irrep_nelec, total_nelec)?;
        let request = nelec.iter().map(|n| n.map(|n| vec![n / 2])).collect();
        Ok(Self::open(rep, irreps, vec![total_nelec / 2], request))
    }

    /// General constructor: ordered occupied groups and an already-validated
    /// per-irrep request (orbital counts per group).
    pub(crate) fn open(
        rep: AoRepresentation,
        irreps: Vec<Irrep>,
        totals: Vec<usize>,
        request: Vec<Option<Vec<usize>>>,
    ) -> Self {
        Self {
            rep,
            irreps,
            totals,
            request,
            last_labels: RefCell::new(Vec::new()),
            split_shell_check: Cell::new(false),
        }
    }

    /// Arm the Kohn-Sham split-shell refusal for the first diagonalization.
    fn arm_split_shell_check(&self) {
        self.split_shell_check.set(true);
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

/// How the unblocked diagonalization is done (kept bit-identical to each
/// solver's historical code).
#[derive(Clone, Copy)]
pub(crate) enum PlainDiag<'a> {
    /// Canonical orthogonalizer `X` (`n x m`): RHF and UHF.
    Rect(&'a Array2<f64>),
    /// Symmetric `S^{-1/2}`: ROHF.
    InvSqrt(&'a Array2<f64>),
}

/// The SCF's Fock diagonalizer: the historical global canonical-orthogonal
/// diagonalization, or the per-irrep blocked one.
pub(crate) struct FockDiag<'a> {
    plain: PlainDiag<'a>,
    blocked: Option<BlockedDiag<'a>>,
}

impl<'a> FockDiag<'a> {
    pub(crate) fn new(
        sym: Option<&'a SymmetrySpec>,
        s: &Array2<f64>,
        plain: PlainDiag<'a>,
    ) -> Result<Self, FerricError> {
        let blocked = sym.map(|sp| BlockedDiag::new(sp, s)).transpose()?;
        Ok(Self { plain, blocked })
    }

    /// Whether this is the per-irrep blocked diagonaliser, which records each
    /// returned column's irrep on its spec.
    pub(crate) fn is_blocked(&self) -> bool {
        self.blocked.is_some()
    }

    pub(crate) fn run(&self, f: &Array2<f64>) -> Result<(Vec<f64>, Array2<f64>), FerricError> {
        match (&self.blocked, self.plain) {
            (Some(b), _) => b.diagonalize(f),
            (None, PlainDiag::Rect(x)) => crate::driver::diagonalize_rect(f, x),
            (None, PlainDiag::InvSqrt(x)) => crate::rohf::diagonalize(f, x),
        }
    }
}

/// Per-iteration blocked diagonalizer: the per-irrep `U_h X_h` transforms
/// built once from the overlap matrix.
pub(crate) struct BlockedDiag<'a> {
    spec: &'a SymmetrySpec,
    /// `U_h X_h`, shape `(nao, m_h)`; `m_h` may be 0.
    xu: Vec<Array2<f64>>,
}

impl<'a> BlockedDiag<'a> {
    pub(crate) fn new(spec: &'a SymmetrySpec, s: &Array2<f64>) -> Result<Self, FerricError> {
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
        Ok(Self { spec, xu })
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
        let order = grouped_order(&eps_all, &irrep_all, &self.spec.request, &self.spec.totals)?;
        if self.spec.split_shell_check.replace(false) {
            let nocc = self.spec.totals.iter().sum();
            let sorted_eps: Vec<f64> = order.iter().map(|&i| eps_all[i]).collect();
            let sorted_irrep: Vec<usize> = order.iter().map(|&i| irrep_all[i]).collect();
            if let Some(msg) = split_ks_shell(
                &self.spec.irreps,
                &self.spec.request,
                &sorted_eps,
                &sorted_irrep,
                nocc,
            ) {
                return Err(FerricError::General(format!(
                    "RhfConfig.symmetry with a Kohn-Sham functional: {msg} of the starting Fock \
                     matrix. The KS grid can split a degenerate shell (C3/C6 partners by up to \
                     about 8e-5 Ha at 110 angular points), so aufbau would choose the occupation \
                     by grid noise; pass irrep_nelec listing at least one of these irreps to \
                     choose it explicitly"
                )));
            }
        }
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
/// groups in order (each in energy order), then the virtuals in energy order.
/// `eps` ascends within each irrep.
///
/// Without a request this is the global energy sort, whose first `totals[0]`
/// entries are group 0, the next `totals[1]` group 1, and so on. A request
/// lists, per irrep, how many orbitals of each group it holds (the lowest
/// ones of that irrep, whatever their energy relative to other irreps);
/// unlisted irreps share the remaining slots of each group by aufbau.
pub(crate) fn grouped_order(
    eps: &[f64],
    irrep: &[usize],
    request: &[Option<Vec<usize>>],
    totals: &[usize],
) -> Result<Vec<usize>, FerricError> {
    let all: Vec<usize> = (0..eps.len()).collect();
    if request.iter().all(Option::is_none) {
        return Ok(sorted_clusters(eps, irrep, all));
    }
    let ngroups = totals.len();
    // Each listed irrep must keep enough eigenpairs after its own
    // linear-dependence filter; otherwise the requested state does not exist.
    for (h, req) in request.iter().enumerate() {
        let have = irrep.iter().filter(|&&g| g == h).count();
        if let Some(counts) = req {
            let want: usize = counts.iter().sum();
            if want > have {
                return Err(FerricError::General(format!(
                    "irrep_nelec asks for {want} orbitals in irrep index {h}, but only {have} \
                     linearly independent functions survive in that block"
                )));
            }
        }
    }
    let mut rank_in_irrep = vec![0usize; request.len()];
    // Some(g): forced into group g; None: forced virtual or left to the pool.
    let mut group_of: Vec<Option<usize>> = vec![None; eps.len()];
    let mut forced = vec![0usize; ngroups];
    let mut pool = Vec::new();
    for i in 0..eps.len() {
        let h = irrep[i];
        let r = rank_in_irrep[h];
        rank_in_irrep[h] += 1;
        let Some(counts) = &request[h] else {
            pool.push(i);
            continue;
        };
        let mut upto = 0;
        for (g, &c) in counts.iter().enumerate() {
            upto += c;
            if r < upto {
                group_of[i] = Some(g);
                forced[g] += 1;
                break;
            }
        }
    }
    let mut free = Vec::with_capacity(ngroups);
    for g in 0..ngroups {
        free.push(totals[g].checked_sub(forced[g]).ok_or_else(|| {
            FerricError::General("irrep_nelec requests more electrons than the molecule has".into())
        })?);
    }
    let pool = sorted_clusters(eps, irrep, pool);
    let n_free: usize = free.iter().sum();
    if n_free > pool.len() {
        return Err(FerricError::General(format!(
            "irrep_nelec leaves {n_free} orbitals to fill by aufbau but the unlisted irreps \
             have only {} functions",
            pool.len()
        )));
    }
    let mut next = 0;
    for (g, &f) in free.iter().enumerate() {
        for &i in &pool[next..next + f] {
            group_of[i] = Some(g);
        }
        next += f;
    }
    let mut out = Vec::with_capacity(eps.len());
    for g in 0..ngroups {
        let members: Vec<usize> = all
            .iter()
            .copied()
            .filter(|&i| group_of[i] == Some(g))
            .collect();
        out.extend(sorted_clusters(eps, irrep, members));
    }
    let virt: Vec<usize> = all.into_iter().filter(|&i| group_of[i].is_none()).collect();
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

/// Message prefixes of every `FerricError::General` that `solve_rhf_symmetric`
/// raises for a bad symmetry request (unsupported feature combination or an
/// invalid `irrep_nelec`), as opposed to a numerical failure. Bindings use
/// [`is_symmetry_refusal`] to classify them as argument errors; every
/// refusal constructed in this module must start with one of these.
pub const SYMMETRY_REFUSAL_PREFIXES: [&str; 3] = [
    "RhfConfig.symmetry",
    "irrep_nelec",
    "solve_rhf_symmetric needs",
];

/// Whether `e` is a symmetry-request refusal (see [`SYMMETRY_REFUSAL_PREFIXES`]).
pub fn is_symmetry_refusal(e: &FerricError) -> bool {
    matches!(e, FerricError::General(m)
        if SYMMETRY_REFUSAL_PREFIXES.iter().any(|p| m.starts_with(p)))
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
    if config.k_builder.as_deref() == Some("cosx") {
        return no(
            "COSX exchange",
            "the seminumerical exchange grid and its final-grid pass are not wired to the symmetry frame",
        );
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
    if config.cosmo.is_some() || config.pcm.is_some() || config.polarizable.is_some() {
        return no(
            "COSMO/PCM/polarizable embedding",
            "cavities and sites are not part of the symmetry detection",
        );
    }
    refuse_coarse_ks_grid(config)?;
    require_no_external_potential(config.external_potential.as_ref())
        .map_err(|e| FerricError::General(format!("RhfConfig.symmetry: {e}")))
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
             through ferric_scf::symmetric::solve_rhf_symmetric (closed-shell RHF), \
             ferric_scf::symmetric_open::solve_uhf_symmetric (UHF) and \
             ferric_scf::symmetric_open::solve_rohf_symmetric (ROHF); their results live in the \
             symmetry frame and have distinct types"
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
    if config.xc.is_some() {
        spec.arm_split_shell_check();
    }
    let scf =
        crate::rhf::solve_rhf_blocked(ctx, &sym_mol, &prep_sym, op, &bounds_sym, config, &spec)?;
    if config.xc.is_some() {
        warn_split_ks_shell_at_convergence(
            &spec,
            &scf.eps_alpha,
            sym_mol.nelec().max(0) as usize / 2,
        );
    }
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

/// The Kohn-Sham split-shell test: among the orbitals of irreps NOT listed in
/// `irrep_nelec` (all irreps without a request), the highest occupied and the
/// lowest unoccupied lie in different irreps within `KS_SHELL_SPLIT_TOL`, so
/// their relative occupation was decided by aufbau over a near-degeneracy
/// the grid may have caused. Returns a description, or `None`. A listed irrep
/// takes its orbitals by request, not by energy, so it cannot be the culprit.
///
/// Run on the FIRST blocked diagonalization (a Fock matrix of the symmetric
/// guess, symmetric under the full point group) as a refusal, and at
/// convergence as a warning. Degenerate partners that fall into the SAME
/// Abelian irrep (an Oh e_g pair maps to Ag + Ag in D2h) pass, because only a
/// cross-irrep pair can be split by the blocking; aufbau within one irrep
/// block is an ordinary eigen-decomposition. `eps` and `irrep` are in
/// MO-column order (occupied first); `nocc` is the occupied count.
fn split_ks_shell(
    irreps: &[Irrep],
    request: &[Option<Vec<usize>>],
    eps: &[f64],
    irrep: &[usize],
    nocc: usize,
) -> Option<String> {
    let unlisted = |k: &usize| request.get(irrep[*k]).is_some_and(Option::is_none);
    let homo = (0..nocc.min(eps.len()))
        .filter(unlisted)
        .max_by(|&a, &b| eps[a].total_cmp(&eps[b]))?;
    let lumo = (nocc.min(eps.len())..eps.len())
        .filter(unlisted)
        .min_by(|&a, &b| eps[a].total_cmp(&eps[b]))?;
    let (h, l) = (irrep[homo], irrep[lumo]);
    let gap = (eps[lumo] - eps[homo]).abs();
    if h == l || gap >= KS_SHELL_SPLIT_TOL {
        return None;
    }
    Some(format!(
        "the near-degenerate highest occupied ({}, {:.8} Ha) and lowest unoccupied ({}, {:.8} \
         Ha) orbitals of the irreps irrep_nelec does not list lie in different irreps within \
         {KS_SHELL_SPLIT_TOL:e} Ha",
        irreps[h].label, eps[homo], irreps[l].label, eps[lumo]
    ))
}

/// The converged-state counterpart of the first-iteration refusal: a
/// warning only, because a shell the SCF itself split is a legitimate state.
fn warn_split_ks_shell_at_convergence(spec: &SymmetrySpec, eps: &[f64], nocc: usize) {
    let labels = spec.last_labels.borrow();
    let n = labels.iter().take_while(|&&h| h != PAD_IRREP).count();
    if let Some(msg) = split_ks_shell(&spec.irreps, &spec.request, &eps[..n], &labels[..n], nocc) {
        eprintln!("warning: blocked KS at convergence: {msg}; check irrep_nelec");
    }
}

/// Refuse a Kohn-Sham main grid coarser than `MIN_KS_ANGULAR` (see there).
fn refuse_coarse_ks_grid(config: &RhfConfig) -> Result<(), FerricError> {
    let n = config.dft_grid.as_ref().map_or(110, |g| g.n_angular);
    if config.xc.is_none() || n >= MIN_KS_ANGULAR {
        return Ok(());
    }
    Err(FerricError::General(format!(
        "RhfConfig.symmetry cannot be combined with a Kohn-Sham grid of {n} angular points: \
         below {MIN_KS_ANGULAR} the grid splits degenerate partners by more than the \
         {KS_SHELL_SPLIT_TOL:e} Ha split-shell tolerance (measured on benzene: 7e-4 Ha at 50, \
         7e-3 Ha at 26), so a partially filled shell could pass unseen; use at least \
         {MIN_KS_ANGULAR} angular points"
    )))
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
    fn refusals_are_classified_and_numerical_errors_are_not() {
        let cfg = RhfConfig {
            symmetry: true,
            k_builder: Some("cosx".into()),
            ..Default::default()
        };
        let e = validate_config(&cfg).unwrap_err();
        assert!(is_symmetry_refusal(&e), "{e}");
        assert!(is_symmetry_refusal(&FerricError::General(
            "irrep_nelec: unknown irrep".into()
        )));
        assert!(!is_symmetry_refusal(&FerricError::General(
            "SCF blew up".into()
        )));
        assert!(!is_symmetry_refusal(&FerricError::Lapack(
            "irrep_nelec".into()
        )));
    }

    #[test]
    fn plain_aufbau_is_a_global_energy_sort() {
        let eps = [0.3, -1.0, 0.1, -0.5];
        let irrep = [0, 1, 0, 1];
        assert_eq!(
            grouped_order(&eps, &irrep, &[None, None], &[2]).unwrap(),
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
            grouped_order(&eps, &irrep, &[None, None], &[1]).unwrap(),
            vec![1, 0, 2]
        );
        // Beyond the tolerance the energy decides.
        let eps = [0.5, 0.5 + 5e-9, 2.0];
        assert_eq!(
            grouped_order(&eps, &irrep, &[None, None], &[1]).unwrap(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn occupation_is_by_energy_across_irreps_not_by_irrep_order() {
        // Irrep 0 has the lowest orbital but irrep 1 holds two below irrep 0's second.
        let eps = [-2.0, 4.0, -1.0, -0.5];
        let irrep = [0, 0, 1, 1];
        let order = grouped_order(&eps, &irrep, &[None, None], &[3]).unwrap();
        assert_eq!(&order[..3], &[0, 2, 3]);
    }

    #[test]
    fn irrep_request_overrides_aufbau_and_puts_occupied_first() {
        // Irrep 0 would win by aufbau; asking for zero electrons there forces
        // the occupation into irrep 1 and the (low) irrep-0 orbitals to virtual.
        let eps = [-3.0, -2.0, 0.0, 1.0];
        let irrep = [0, 0, 1, 1];
        let order = grouped_order(&eps, &irrep, &[Some(vec![0]), None], &[2]).unwrap();
        assert_eq!(order, vec![2, 3, 0, 1]);
        // A forced pair in irrep 1 plus aufbau in the unlisted irrep 0.
        let order = grouped_order(&eps, &irrep, &[None, Some(vec![1])], &[2]).unwrap();
        assert_eq!(&order[..2], &[0, 2]);
    }

    #[test]
    fn request_exceeding_surviving_block_dimension_is_an_error() {
        // Irrep 1 lost functions to linear dependence: 1 eigenpair, 4 electrons asked.
        let eps = [-3.0, -2.0, 0.0];
        let irrep = [0, 0, 1];
        let err = grouped_order(&eps, &irrep, &[None, Some(vec![2])], &[3]).unwrap_err();
        assert!(err.to_string().contains("linearly independent"), "{err}");
        // The same request within the surviving dimension is fine.
        assert!(grouped_order(&eps, &irrep, &[None, Some(vec![1])], &[2]).is_ok());
    }
}
