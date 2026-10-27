//! Point-group-blocked open-shell UHF and ROHF (stage 3a of issue #294).
//!
//! [`solve_uhf_symmetric`] and [`solve_rohf_symmetric`] are the open-shell
//! counterparts of [`solve_rhf_symmetric`](crate::symmetric::solve_rhf_symmetric)
//! and share its design and frame contract (read the [`crate::symmetric`]
//! module docs): the SCF runs on the reoriented, symmetrized molecule, the
//! result is a distinct type ([`SymmetricUhf`], [`SymmetricRohf`]) and never a
//! bare `ScfResult`, and `solve_uhf` / `solve_rohf` themselves refuse
//! `RhfConfig::symmetry`.
//!
//! # What is blocked
//!
//! UHF diagonalizes `F_alpha` and `F_beta` separately, each per irrep
//! (`F_h = U_hᵀ F U_h`, the same per-block lindep-filtered orthogonalizer as
//! RHF), and occupies the lowest `n_alpha` / `n_beta` orbitals of each spin by
//! aufbau across the irreps. ROHF diagonalizes its ONE Roothaan effective
//! Fock per irrep; the lowest `n_beta` orbitals of the merged spectrum are the
//! core set, the next `n_alpha - n_beta` the open set. The DIIS, density and
//! energy code is the unchanged global code: the columns of the MO matrix are
//! merged and ordered exactly as the unblocked solver expects them (core,
//! open, virtual; occupied first per spin).
//!
//! # Occupation control
//!
//! `irrep_nelec: Option<&[(String, (usize, usize))]>` maps an irrep label to
//! `(n_alpha, n_beta)`, the number of electrons of each spin in that irrep
//! (one orbital per electron per spin, so these are orbital counts too). The
//! `RhfConfig::irrep_nelec` field (a single even electron count, closed shell)
//! is refused here: it cannot describe an open shell. Listed irreps take the
//! lowest orbitals of that irrep, whatever their energy relative to other
//! irreps; unlisted irreps share the remaining electrons by aufbau. For ROHF
//! the pair means `n_beta` core orbitals and `n_alpha - n_beta` open orbitals
//! in that irrep (`n_alpha >= n_beta`, else an error). A request that is
//! impossible (unknown irrep, listed twice, more electrons than the irrep has
//! linearly independent functions, a total that is not `n_alpha`/`n_beta`) is a
//! typed error.
//!
//! # Refusals beyond the closed-shell list
//!
//! All stage 2 refusals apply (MOM, smearing, Newton/AH/TRAH/AURORA, stability,
//! cDFT constraints, user initial density, external potentials, solvation,
//! polarizable embedding). Kohn-Sham functionals are refused here (UKS/ROKS
//! blocking is not validated yet), although closed-shell RKS is blocked by
//! [`solve_rhf_symmetric`](crate::symmetric::solve_rhf_symmetric). ROHF additionally refuses
//! `rohf_occupation_guard = true`: its swap witness and hole-swap lock move
//! electrons between orbitals of different irreps, which the blocking fixes.
//! Set it to `false` (the pre-F6 loop). UHF's usual `HOMO/LUMO` mix of the
//! beta orbitals on a closed-shell occupation is skipped. The reason is not
//! that the mix would cross irreps (HOMO and LUMO can share one), but that a
//! blocked closed-shell UHF can never leave `alpha = beta`: its two spin
//! blocks are identical, so it returns the restricted determinant. Spin- and
//! spatial-symmetry-broken UHF solutions lie by construction outside the
//! blocked space (the unblocked solver can find them, see the stretched-H2
//! negative control in the tests).
//!
//! An empty `irrep_nelec` request is an error everywhere (pass `None` for
//! plain aufbau), C1 included.

use crate::result::ScfResult;
use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;
use crate::symmetric::{frame_bounds, validate_config, SymmetrySpec};
use ferric_core::parallel::ParallelContext;
use ferric_core::symmetry::{
    self, irreps, AbelianGroup, AoRepresentation, Irrep, SymmetryDetection, DEFAULT_SYMMETRY_TOL,
};
use ferric_core::{mol::Molecule, FerricError};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;

/// `(irrep label, (n_alpha, n_beta))`.
pub type IrrepSpinCount = (String, (usize, usize));

/// The frame an open-shell blocked result lives in (see
/// [`crate::symmetric`]).
#[derive(Debug, Clone)]
struct Frame {
    group: String,
    molecule: Molecule,
    rotation: [[f64; 3]; 3],
    origin: [f64; 3],
    max_symmetrization_shift: f64,
}

impl Frame {
    fn c1(mol: &Molecule) -> Self {
        Self {
            group: "C1".into(),
            molecule: mol.clone(),
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            origin: [0.0; 3],
            max_symmetrization_shift: 0.0,
        }
    }

    fn of(det: &SymmetryDetection) -> Self {
        Self {
            group: det.group.name().to_string(),
            molecule: det.molecule.clone(),
            rotation: det.rotation,
            origin: det.origin,
            max_symmetrization_shift: det.max_symmetrization_shift,
        }
    }
}

macro_rules! frame_accessors {
    () => {
        /// Largest atom displacement (Bohr) made by symmetrizing the input
        /// geometry; 0 for C1, at most the detection tolerance otherwise.
        pub fn max_symmetrization_shift(&self) -> f64 {
            self.frame.max_symmetrization_shift
        }

        /// Total energy of the SYMMETRIZED geometry
        /// ([`molecule`](Self::molecule)).
        pub fn energy(&self) -> f64 {
            self.scf.energy
        }

        /// Schoenflies symbol of the Abelian group used for blocking.
        pub fn group(&self) -> &str {
            &self.frame.group
        }

        /// The molecule the integrals were built from: the reoriented,
        /// symmetrized geometry (the input molecule itself for C1).
        pub fn molecule(&self) -> &Molecule {
            &self.frame.molecule
        }

        /// `r_sym = rotation * (r_in - origin)`; identity and zero for C1.
        pub fn frame_transform(&self) -> ([[f64; 3]; 3], [f64; 3]) {
            (self.frame.rotation, self.frame.origin)
        }

        /// The plain SCF result, whose AO matrices are expressed in the basis
        /// of [`molecule`](Self::molecule); pair it with that molecule, never
        /// the input one.
        pub fn scf_in_frame(&self) -> &ScfResult {
            &self.scf
        }

        /// Consume into the plain result; see
        /// [`scf_in_frame`](Self::scf_in_frame).
        pub fn into_scf_in_frame(self) -> ScfResult {
            self.scf
        }
    };
}

/// A symmetry-blocked UHF solution and the frame its AO matrices live in.
#[derive(Debug, Clone)]
pub struct SymmetricUhf {
    scf: ScfResult,
    frame: Frame,
    alpha_irreps: Vec<String>,
    beta_irreps: Vec<String>,
    nocc: (usize, usize),
}

impl SymmetricUhf {
    frame_accessors!();

    /// Irrep label of each alpha MO column (`"-"` for padded columns).
    pub fn orbital_irreps_alpha(&self) -> &[String] {
        &self.alpha_irreps
    }

    /// Irrep label of each beta MO column (`"-"` for padded columns).
    pub fn orbital_irreps_beta(&self) -> &[String] {
        &self.beta_irreps
    }

    /// `(n_alpha, n_beta)` occupied orbitals per irrep label, irreps with no
    /// electrons omitted.
    pub fn irrep_occupations(&self) -> Vec<IrrepSpinCount> {
        let (na, nb) = self.nocc;
        merge_counts(&self.alpha_irreps[..na], &self.beta_irreps[..nb])
    }
}

/// A symmetry-blocked ROHF solution and the frame its AO matrices live in.
#[derive(Debug, Clone)]
pub struct SymmetricRohf {
    scf: ScfResult,
    frame: Frame,
    irreps: Vec<String>,
    /// (core, open) orbital counts.
    nocc: (usize, usize),
}

impl SymmetricRohf {
    frame_accessors!();

    /// Irrep label of each MO column of the single ROHF orbital set (`"-"` for
    /// padded columns). Columns are ordered core, open, virtual.
    pub fn orbital_irreps(&self) -> &[String] {
        &self.irreps
    }

    /// `(n_alpha, n_beta)` per irrep label (core orbitals count for both
    /// spins, open orbitals for alpha), irreps with no electrons omitted.
    pub fn irrep_occupations(&self) -> Vec<IrrepSpinCount> {
        let (core, open) = self.nocc;
        merge_counts(&self.irreps[..core + open], &self.irreps[..core])
    }
}

fn merge_counts(alpha: &[String], beta: &[String]) -> Vec<IrrepSpinCount> {
    let mut out: Vec<IrrepSpinCount> = Vec::new();
    let mut bump = |label: &String, spin: usize| {
        let at = out.iter().position(|(l, _)| l == label).unwrap_or_else(|| {
            out.push((label.clone(), (0, 0)));
            out.len() - 1
        });
        if spin == 0 {
            out[at].1 .0 += 1;
        } else {
            out[at].1 .1 += 1;
        }
    };
    alpha.iter().for_each(|l| bump(l, 0));
    beta.iter().for_each(|l| bump(l, 1));
    out
}

/// Electron counts of the molecule: `(n_alpha, n_beta)`.
fn spin_counts(mol: &Molecule) -> Result<(usize, usize), FerricError> {
    let nelec = mol.nelec() as i64;
    let two_s = mol.multiplicity as i64 - 1;
    if two_s < 0 || nelec < two_s || (nelec - two_s) % 2 != 0 {
        return Err(FerricError::General(format!(
            "symmetric open-shell SCF: incompatible nelec={nelec} and multiplicity={}",
            mol.multiplicity
        )));
    }
    Ok((
        ((nelec + two_s) / 2) as usize,
        ((nelec - two_s) / 2) as usize,
    ))
}

/// Everything the open-shell entry points refuse beyond the RHF list.
fn validate_open(config: &RhfConfig, who: &str, rohf: bool) -> Result<(), FerricError> {
    if !config.symmetry {
        return Err(FerricError::General(format!(
            "{who} needs RhfConfig.symmetry = true"
        )));
    }
    if config.irrep_nelec.is_some() {
        return Err(FerricError::General(format!(
            "{who}: RhfConfig.irrep_nelec holds closed-shell electron counts; pass the \
             open-shell (n_alpha, n_beta) per irrep through the irrep_nelec argument"
        )));
    }
    validate_config(config)?;
    if config.xc.is_some() {
        return Err(FerricError::General(format!(
            "RhfConfig.symmetry cannot be combined with a Kohn-Sham functional in {who}: UKS/ROKS \
             blocking is not validated yet (closed-shell RKS blocking is \
             ferric_scf::symmetric::solve_rhf_symmetric)"
        )));
    }
    if rohf && config.rohf_occupation_guard {
        return Err(FerricError::General(format!(
            "RhfConfig.symmetry cannot be combined with rohf_occupation_guard: its swap witness \
             and hole-swap lock move electrons between irreps; set rohf_occupation_guard = false \
             for {who}"
        )));
    }
    Ok(())
}

/// Validate a `(n_alpha, n_beta)` request against the irreps and the molecule:
/// the per-irrep counts, `None` for unlisted irreps.
fn resolve_request(
    rep: &AoRepresentation,
    irreps: &[Irrep],
    request: Option<&[IrrepSpinCount]>,
    (na, nb): (usize, usize),
    rohf: bool,
) -> Result<Vec<Option<(usize, usize)>>, FerricError> {
    let mut out = vec![None; irreps.len()];
    let Some(req) = request else { return Ok(out) };
    let dims = rep.irrep_dimensions()?;
    for (label, (a, b)) in req {
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
        if (*a).max(*b) > dims[h].1 {
            return Err(FerricError::General(format!(
                "irrep_nelec: {label} holds ({a}, {b}) electrons but has only {} basis functions",
                dims[h].1
            )));
        }
        if rohf && a < b {
            return Err(FerricError::General(format!(
                "irrep_nelec: {label} has n_alpha={a} < n_beta={b}; ROHF core orbitals are \
                 doubly occupied, so n_alpha >= n_beta in every irrep"
            )));
        }
        out[h] = Some((*a, *b));
    }
    let (fa, fb) = out
        .iter()
        .flatten()
        .fold((0, 0), |(x, y), (a, b)| (x + a, y + b));
    let all_listed = out.iter().all(Option::is_some);
    if fa > na || fb > nb || (all_listed && (fa, fb) != (na, nb)) {
        return Err(FerricError::General(format!(
            "irrep_nelec sums to (alpha, beta) = ({fa}, {fb}) electrons but the molecule has \
             ({na}, {nb})"
        )));
    }
    Ok(out)
}

fn spec_from_detection(
    det: &SymmetryDetection,
    sym_mol: &Molecule,
    prep_sym: &PreparedBasis,
) -> Result<(AoRepresentation, Vec<Irrep>), FerricError> {
    let rep = AoRepresentation::new(det, sym_mol, prep_sym.basis_set())?;
    Ok((rep, irreps(&det.ops)?))
}

/// An empty request is ambiguous (no constraint, or zero electrons everywhere);
/// refuse it uniformly, whatever the point group.
fn require_nonempty(request: Option<&[IrrepSpinCount]>, who: &str) -> Result<(), FerricError> {
    match request {
        Some([]) => Err(FerricError::General(format!(
            "{who}: irrep_nelec is empty; pass None for plain aufbau"
        ))),
        _ => Ok(()),
    }
}

fn c1_request_fits(
    request: Option<&[IrrepSpinCount]>,
    counts: (usize, usize),
) -> Result<(), FerricError> {
    match request {
        Some(r) if r.len() != 1 || r[0].0 != "A" || r[0].1 != counts => {
            Err(FerricError::General(format!(
                "irrep_nelec {r:?} does not fit the detected point group C1 (irrep \"A\", \
                 {counts:?} electrons)"
            )))
        }
        _ => Ok(()),
    }
}

fn c1_labels(eps: &[f64]) -> Vec<String> {
    eps.iter()
        .map(|&e| if e >= 1e5 { "-" } else { "A" }.to_string())
        .collect()
}

/// UHF with point-group blocking (`config.symmetry` must be set). `mol` must
/// carry its ECP assignment. See the module docs for the frame and occupation
/// control.
pub fn solve_uhf_symmetric(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    irrep_nelec: Option<&[IrrepSpinCount]>,
) -> Result<SymmetricUhf, FerricError> {
    validate_open(config, "solve_uhf_symmetric", false)?;
    require_nonempty(irrep_nelec, "solve_uhf_symmetric")?;
    let det = symmetry::detect(mol, DEFAULT_SYMMETRY_TOL)?;
    if det.group == AbelianGroup::C1 {
        c1_request_fits(irrep_nelec, spin_counts(mol)?)?;
        let plain = RhfConfig {
            symmetry: false,
            ..config.clone()
        };
        let scf = crate::uhf::solve_uhf(ctx, mol, prep, bounds, &plain)?;
        let nocc = spin_counts(mol)?;
        return Ok(SymmetricUhf {
            alpha_irreps: c1_labels(&scf.eps_alpha),
            beta_irreps: c1_labels(scf.eps_beta.as_deref().unwrap_or(&scf.eps_alpha)),
            scf,
            frame: Frame::c1(mol),
            nocc,
        });
    }
    let sym_mol = det.molecule.clone();
    let prep_sym = PreparedBasis::new(&sym_mol, prep.basis_set())?;
    let bounds_sym = frame_bounds(bounds.op, &prep_sym, bounds)?;
    let (rep, ir) = spec_from_detection(&det, &sym_mol, &prep_sym)?;
    let nocc = spin_counts(&sym_mol)?;
    let want = resolve_request(&rep, &ir, irrep_nelec, nocc, false)?;
    let per_spin = |pick: fn(&(usize, usize)) -> usize| -> Vec<Option<Vec<usize>>> {
        want.iter()
            .map(|w| w.as_ref().map(|c| vec![pick(c)]))
            .collect()
    };
    let spec_a = SymmetrySpec::open(rep.clone(), ir.clone(), vec![nocc.0], per_spin(|c| c.0));
    let spec_b = SymmetrySpec::open(rep, ir, vec![nocc.1], per_spin(|c| c.1));
    let scf = crate::uhf::solve_uhf_blocked(
        ctx,
        &sym_mol,
        &prep_sym,
        &bounds_sym,
        config,
        (&spec_a, &spec_b),
    )?;
    Ok(SymmetricUhf {
        scf,
        frame: Frame::of(&det),
        alpha_irreps: spec_a.column_labels(),
        beta_irreps: spec_b.column_labels(),
        nocc,
    })
}

/// ROHF with point-group blocking (`config.symmetry` must be set, and
/// `rohf_occupation_guard` cleared). `mol` must carry its ECP assignment. See
/// the module docs for the frame and occupation control.
pub fn solve_rohf_symmetric(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    irrep_nelec: Option<&[IrrepSpinCount]>,
) -> Result<SymmetricRohf, FerricError> {
    validate_open(config, "solve_rohf_symmetric", true)?;
    require_nonempty(irrep_nelec, "solve_rohf_symmetric")?;
    let det = symmetry::detect(mol, DEFAULT_SYMMETRY_TOL)?;
    if det.group == AbelianGroup::C1 {
        let (na, nb) = spin_counts(mol)?;
        c1_request_fits(irrep_nelec, (na, nb))?;
        let plain = RhfConfig {
            symmetry: false,
            ..config.clone()
        };
        let scf = crate::rohf::solve_rohf(ctx, mol, prep, op, bounds, &plain)?;
        return Ok(SymmetricRohf {
            irreps: c1_labels(&scf.eps_alpha),
            scf,
            frame: Frame::c1(mol),
            nocc: (nb, na - nb),
        });
    }
    let sym_mol = det.molecule.clone();
    let prep_sym = PreparedBasis::new(&sym_mol, prep.basis_set())?;
    let bounds_sym = frame_bounds(op, &prep_sym, bounds)?;
    let (rep, ir) = spec_from_detection(&det, &sym_mol, &prep_sym)?;
    let (na, nb) = spin_counts(&sym_mol)?;
    let want = resolve_request(&rep, &ir, irrep_nelec, (na, nb), true)?;
    let request = want
        .iter()
        .map(|w| w.as_ref().map(|&(a, b)| vec![b, a - b]))
        .collect();
    let spec = SymmetrySpec::open(rep, ir, vec![nb, na - nb], request);
    let scf =
        crate::rohf::solve_rohf_blocked(ctx, &sym_mol, &prep_sym, &bounds_sym, config, &spec)?;
    Ok(SymmetricRohf {
        scf,
        frame: Frame::of(&det),
        irreps: spec.column_labels(),
        nocc: (nb, na - nb),
    })
}
