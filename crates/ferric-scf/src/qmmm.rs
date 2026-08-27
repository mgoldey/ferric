//! QM/MM partitioning and electrostatic embedding for ligand-in-pocket
//! calculations.
//!
//! # What this module is (and is not)
//!
//! This is a **thin partitioning/bookkeeping layer** on top of machinery that
//! already exists and is already validated. It deliberately implements no new
//! physics:
//!
//! - The embedding operator itself is
//!   [`ferric_core::external_potential::ExternalPotential`], which is already
//!   threaded through **every** SCF variant (RHF/UHF/ROHF/RKS/UKS/ROKS) via
//!   `RhfConfig::external_potential` and folded into `hcore` once before the
//!   SCF loop.
//! - The MM-region force reuses the same nuclear-attraction derivative blocks
//!   that [`crate::properties::electric_field_at_atoms`] contracts (see
//!   [`electric_field_at_points`] below).
//!
//! What this module adds is the **partition**: given one big structure (protein
//! + ligand, say) and a rule for which atoms are quantum, produce (a) the QM
//! [`Molecule`] the SCF solver should see, and (b) the [`ExternalPotential`] of
//! MM point charges it should be embedded in, plus optional link-atom capping
//! when the partition cuts a covalent bond.
//!
//! # Level of theory: mechanical/electrostatic embedding
//!
//! The MM charges polarize the QM density (one-way coupling: QM sees MM, MM does
//! not see QM back). This is standard **electrostatic embedding**. There is no
//! MM force field here — no MM-MM bonded/nonbonded energy, no Lennard-Jones
//! QM-MM term, no MM polarizability. The energy this layer contributes to is
//! `E_QM(in the field of the MM charges)`, and the MM charges are fixed
//! external parameters, not dynamical variables. A complete QM/MM total energy
//! would add `E_MM` and a QM-MM van der Waals term from a force field; that is
//! the caller's job and out of scope here.
//!
//! # Known approximations (read before trusting a boundary number)
//!
//! - **Link atoms are an approximation with well-documented artifacts.** Capping
//!   a cut bond with a hydrogen introduces a chemically fictitious atom whose
//!   electrons are not in the real system, and it sits very close to the first
//!   MM atom. Properties evaluated *near the boundary* (charges, ESP, forces on
//!   the frontier atoms) are the least trustworthy quantities this module can
//!   produce. Cut only nonpolar single bonds (the classic guidance: C-C, never
//!   a C=O, an amide C-N, or anything conjugated/charged), and keep the cut at
//!   least two or three bonds away from anything chemically interesting.
//! - **Boundary charge schemes** for the "link atom sits on top of an MM point
//!   charge" problem are opt-in via [`QmmmSystem::with_boundary_charges`]:
//!   [`BoundaryChargeScheme::DeleteHost`] (Z1), and the Lin–Truhlar
//!   [`BoundaryChargeScheme::RedistributedCharge`] (RC) /
//!   [`BoundaryChargeScheme::RedistributedChargeDipole`] (RCD) schemes that
//!   move the host charge onto the midpoints of its bonds to the next MM shell
//!   (RC conserves charge; RCD conserves charge and dipole). The default,
//!   [`BoundaryChargeScheme::Keep`], is the untreated boundary. Gaussian
//!   smearing and double link atoms are not implemented. [`QmmmSystem`] always
//!   **excludes MM charges on atoms that are themselves in the QM region**
//!   (they would be double-counted). If your link atom lands within a Bohr or
//!   so of a nonzero MM charge under `Keep`, the result is unphysical and this
//!   module will not warn you beyond [`QmmmSystem::min_link_to_charge_distance`].
//! - **Force projection** onto real atoms is done by [`full_gradient`]: the
//!   link-atom row of the QM gradient is chain-ruled onto its two hosts, and
//!   the [`mm_forces`] on every embedding charge — including the off-atom
//!   midpoint charges of RC/RCD, listed after the atom-centred ones in
//!   [`QmmmSystem::mm_charge_positions`] order — are mapped onto the atoms
//!   that carry them. The result is `dE/dR` over the full structure with no
//!   MM force-field terms (none exist here).
//! - Everything is in **Bohr / atomic units**, consistent with
//!   [`ExternalPotential`]. Note [`Molecule::parse_xyz`] converts Ångström to
//!   Bohr on load, so coordinates taken off a [`Molecule`] are already Bohr and
//!   need no conversion to build a [`PointCharge`].
//!
//! # Exactness contract
//!
//! **An empty MM region is a bit-identical no-op.** A [`QmmmSystem`] with no MM
//! atoms produces `to_external_potential() == None`, so the SCF call is
//! byte-for-byte the same code path as a gas-phase calculation on the same QM
//! atoms — not "numerically close", the same. This is the contract every other
//! optional term in this codebase follows (`external_potential`, `cosmo`,
//! `pcm`, `verbose`) and it is regression-tested first, in
//! `crates/ferric-scf/tests/qmmm.rs`.

use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::{Atom, Molecule};
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mm::{MmEnergy, MmTopology};
use ndarray::Array2;

use crate::rhf::RhfConfig;
use crate::screening::SchwarzBounds;

/// One atom of the full (QM + MM) structure, with the MM partial charge it
/// carries if it ends up in the MM region.
///
/// `charge` is ignored for atoms selected into the QM region — those atoms are
/// described by the wavefunction, and including their MM point charge as well
/// would double-count them. Coordinates are in **Bohr**.
#[derive(Debug, Clone)]
pub struct QmmmAtom {
    pub symbol: String,
    pub z: i32,
    pub x: f64,
    pub y: f64,
    pub z_pos: f64,
    /// MM partial charge in units of e. Typically fractional (a force-field
    /// charge). Only used if this atom lands in the MM region.
    pub charge: f64,
}

impl QmmmAtom {
    /// Create a QM/MM atom with element, nuclear charge, Bohr coordinates, and MM partial charge.
    pub fn new(symbol: impl Into<String>, z: i32, x: f64, y: f64, z_pos: f64, charge: f64) -> Self {
        Self { symbol: symbol.into(), z, x, y, z_pos, charge }
    }

    #[inline]
    fn distance2(&self, other: &QmmmAtom) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z_pos - other.z_pos;
        dx * dx + dy * dy + dz * dz
    }
}

/// A covalent bond cut by the QM/MM partition, capped with a link hydrogen.
///
/// The link atom is placed along the original bond vector at a fixed fraction
/// of the original bond length:
///
/// ```text
///   R_link = R_qm + g · (R_mm − R_qm),    0 < g < 1
/// ```
///
/// where `R_qm` is the QM frontier atom and `R_mm` is the MM atom whose bond to
/// it was severed. This is the standard **scaled-position link atom**: the cap
/// stays on the real bond vector (so the local geometry is not distorted) and
/// `g` is chosen so the resulting X-H distance is a sensible one, typically
/// `g = d(X-H) / d(X-Y)` — for a cut C-C bond that is roughly `1.09/1.53 ≈ 0.71`
/// (see [`DEFAULT_LINK_SCALE`]).
#[derive(Debug, Clone)]
pub struct LinkAtomSpec {
    /// Index into the QM region (i.e. into [`QmmmSystem::qm_indices`] order) of
    /// the frontier QM atom the cap is bonded to.
    pub qm_atom: usize,
    /// Index into the **full** atom list of the MM atom whose bond was cut.
    /// Recorded so a caller can implement its own charge-redistribution scheme
    /// (this module does not — see the module docs).
    pub mm_atom_full_index: usize,
    /// Scale factor `g` along the bond vector.
    pub scale: f64,
    /// Placed link-hydrogen position in Bohr.
    pub position: [f64; 3],
}

/// Default scaled-position factor for a cut C-C single bond: `1.09 Å / 1.53 Å`.
///
/// This is the conventional choice (place the cap at a typical C-H bond length
/// along a typical C-C bond). It is *not* right for every bond type — cutting
/// anything other than a nonpolar single bond is discouraged in the first place
/// (see the module docs), but if you do, set `scale` explicitly.
pub const DEFAULT_LINK_SCALE: f64 = 1.09 / 1.53;

/// What to do with the MM charge on the host atom (M1) of a cut bond.
///
/// M1 is the MM atom whose bond to the QM frontier atom was severed; M2 are
/// the MM atoms bonded to M1 (the next shell out). Under the scaled-position
/// link atom, the cap sits ~0.5 Å from M1, so M1's point charge over-polarizes
/// the QM density unless it is removed or moved. Nomenclature follows
/// Lin & Truhlar, J. Phys. Chem. A 109, 3991 (2005).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryChargeScheme {
    /// Leave every MM charge where it is (the untreated boundary; default).
    Keep,
    /// Z1: zero the M1 charge. Simple, but the MM region's total charge
    /// changes by −q(M1).
    DeleteHost,
    /// RC: zero the M1 charge and place q(M1)/n on the midpoint of each of
    /// the n M1–M2 bonds. Conserves total charge.
    RedistributedCharge,
    /// RCD: as RC but each midpoint carries 2·q(M1)/n and each M2 charge is
    /// reduced by q(M1)/n. Conserves total charge AND the dipole of the
    /// {M1, M2…} group, so the electrostatic potential seen by the QM region
    /// is unchanged to dipole order.
    RedistributedChargeDipole,
}

impl BoundaryChargeScheme {
    /// Strict config-string parser (`"keep"`, `"delete-host"`, `"rc"`,
    /// `"rcd"`). Unknown values — including different casing — are an error,
    /// never a silent default, per the repo's config-honesty convention.
    pub fn parse_config_str(s: &str) -> Result<Self, FerricError> {
        match s {
            "keep" => Ok(Self::Keep),
            "delete-host" => Ok(Self::DeleteHost),
            "rc" => Ok(Self::RedistributedCharge),
            "rcd" => Ok(Self::RedistributedChargeDipole),
            other => Err(FerricError::General(format!(
                "unknown boundary charge scheme {other:?}; expected one of \
                 \"keep\", \"delete-host\", \"rc\", \"rcd\""
            ))),
        }
    }

    /// The config string [`Self::parse_config_str`] accepts for this scheme.
    pub fn config_str(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::DeleteHost => "delete-host",
            Self::RedistributedCharge => "rc",
            Self::RedistributedChargeDipole => "rcd",
        }
    }
}

/// An off-atom point charge introduced by a redistribution scheme, sitting on
/// the midpoint of an M1–M2 bond. Both host atom indices (into the full atom
/// list) are recorded so a force on this charge can be projected back onto
/// the two real atoms (half each, since the midpoint is their mean).
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryCharge {
    pub q: f64,
    /// Position in Bohr.
    pub position: [f64; 3],
    /// (M1, M2) indices into the full atom list.
    pub hosts: (usize, usize),
}

/// How the QM region is selected out of the full structure.
#[derive(Debug, Clone)]
pub enum QmSelection {
    /// Explicit indices into the full atom list.
    Indices(Vec<usize>),
    /// Every atom within `radius` Bohr of any of the `seeds` atoms, plus the
    /// seeds themselves. The common "ligand plus everything within R of it"
    /// pocket case: pass the ligand atom indices as seeds.
    ///
    /// Selection is **by atom**, with no residue/molecule completion — if a
    /// sidechain straddles the cutoff it will be cut mid-residue, which is
    /// exactly the situation link atoms exist for. A production pocket setup
    /// would normally expand the selection to whole residues first; that is
    /// the caller's job (build the index list and use [`QmSelection::Indices`]),
    /// or use [`QmSelection::WithinRadiusWholeResidues`].
    WithinRadius { seeds: Vec<usize>, radius: f64 },
    /// Like [`QmSelection::WithinRadius`], but a residue joins the QM region
    /// **in full** if any of its atoms lands within `radius` of any seed —
    /// no sidechain is cut mid-residue.
    ///
    /// `residue_ids[i]` is the residue index of atom `i` (`residue_ids.len()
    /// == atoms.len()`); atoms with the same id are treated as one residue,
    /// regardless of numeric value or contiguity. Passing `residue_ids = 0
    /// .. natoms` (every atom its own residue) makes this variant an exact
    /// no-op on top of [`QmSelection::WithinRadius`] — the same seeds/radius
    /// select the same QM/MM split, byte for byte.
    WithinRadiusWholeResidues { seeds: Vec<usize>, radius: f64, residue_ids: Vec<usize> },
}

/// A partition of a full structure into a QM region (wavefunction) and an MM
/// region (fixed point charges).
///
/// Build with [`QmmmSystem::new`], then feed an existing solver:
///
/// ```ignore
/// let sys = QmmmSystem::new(&atoms, QmSelection::WithinRadius { seeds, radius }, 0, 1)?;
/// let mol = sys.to_qm_molecule();
/// let prep = PreparedBasis::new(&mol, &basis)?;
/// let config = RhfConfig {
///     external_potential: sys.to_external_potential(),
///     ..Default::default()
/// };
/// let scf = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config)?;
/// ```
///
/// The solver is untouched — that is the whole point. `to_external_potential`
/// returns `None` for an empty MM region, so the gas-phase path is entered
/// literally, not approximately.
#[derive(Debug, Clone)]
pub struct QmmmSystem {
    /// Indices into the full atom list that are treated quantum-mechanically,
    /// ascending. Does **not** include link atoms (those are appended after
    /// these, in `link_atoms` order, by [`QmmmSystem::to_qm_molecule`]).
    pub qm_indices: Vec<usize>,
    /// Indices into the full atom list that are MM point charges, ascending.
    /// Disjoint from `qm_indices` by construction.
    pub mm_indices: Vec<usize>,
    /// Link hydrogens capping cut bonds, in the order they are appended to the
    /// QM molecule. Empty if the partition cut no bonds (or if link atoms were
    /// not requested).
    pub link_atoms: Vec<LinkAtomSpec>,
    /// The full structure this partition refers to.
    pub atoms: Vec<QmmmAtom>,
    /// Total charge of the QM region (passed through to the [`Molecule`]).
    pub qm_charge: i32,
    /// Spin multiplicity (2S+1) of the QM region.
    pub qm_multiplicity: usize,
    /// The MM charge each atom of the full structure actually contributes,
    /// after [`QmmmSystem::with_boundary_charges`]. Equals `atoms[i].charge`
    /// until a scheme other than [`BoundaryChargeScheme::Keep`] is applied
    /// (and for every atom a scheme does not touch). Only entries at
    /// `mm_indices` are ever read.
    pub effective_charges: Vec<f64>,
    /// Off-atom charges introduced by RC/RCD redistribution. Empty otherwise.
    pub boundary_charges: Vec<BoundaryCharge>,
    /// The scheme currently applied.
    pub boundary_scheme: BoundaryChargeScheme,
    /// The `residue_ids` passed to [`QmSelection::WithinRadiusWholeResidues`],
    /// kept for diagnostics/downstream residue-aware logic. `None` for every
    /// other selection variant.
    pub residue_ids: Option<Vec<usize>>,
    /// The bond list and scale factor last passed to
    /// [`QmmmSystem::with_link_atoms`], kept so [`QmmmSystem::with_coordinates`]
    /// can re-run link placement at a new geometry. `None` if link atoms were
    /// never requested.
    link_atom_setup: Option<(Vec<(usize, usize)>, f64)>,
    /// The bond list and scheme last passed to
    /// [`QmmmSystem::with_boundary_charges`], kept so
    /// [`QmmmSystem::with_coordinates`] can re-derive boundary charges at a
    /// new geometry. `None` if no scheme was ever applied (equivalent to
    /// `Keep` with an empty bond list, but recorded separately so
    /// `with_coordinates` skips re-deriving it when it was never asked for).
    boundary_charge_setup: Option<(Vec<(usize, usize)>, BoundaryChargeScheme)>,
}

impl QmmmSystem {
    /// Partition `atoms` into QM and MM regions. No link atoms are added — use
    /// [`QmmmSystem::with_link_atoms`] to cap cut bonds afterwards.
    ///
    /// MM charges of atoms selected into the QM region are dropped (they would
    /// double-count atoms the wavefunction already describes). MM atoms whose
    /// charge is exactly zero are still recorded in `mm_indices` but contribute
    /// nothing to the embedding potential.
    pub fn new(
        atoms: &[QmmmAtom],
        selection: QmSelection,
        qm_charge: i32,
        qm_multiplicity: usize,
    ) -> Result<Self, FerricError> {
        if atoms.is_empty() {
            return Err(FerricError::General(
                "QmmmSystem::new: empty atom list".to_string(),
            ));
        }
        if qm_multiplicity == 0 {
            return Err(FerricError::General(
                "QmmmSystem::new: multiplicity must be >= 1 (2S+1)".to_string(),
            ));
        }

        let natoms = atoms.len();
        let mut is_qm = vec![false; natoms];
        let mut residue_ids_out: Option<Vec<usize>> = None;

        match selection {
            QmSelection::Indices(idx) => {
                for i in idx {
                    if i >= natoms {
                        return Err(FerricError::General(format!(
                            "QmmmSystem::new: QM index {i} out of range (structure has \
                             {natoms} atoms)"
                        )));
                    }
                    is_qm[i] = true;
                }
            }
            QmSelection::WithinRadius { seeds, radius } => {
                Self::validate_radius_selection(&seeds, radius, natoms)?;
                Self::apply_within_radius(atoms, &seeds, radius, &mut is_qm);
            }
            QmSelection::WithinRadiusWholeResidues { seeds, radius, residue_ids } => {
                Self::validate_radius_selection(&seeds, radius, natoms)?;
                if residue_ids.len() != natoms {
                    return Err(FerricError::General(format!(
                        "QmmmSystem::new: residue_ids has {} entries, expected {natoms} \
                         (one per atom)",
                        residue_ids.len()
                    )));
                }
                Self::apply_within_radius(atoms, &seeds, radius, &mut is_qm);
                // Second pass: any residue with at least one QM atom joins
                // whole. `hit` collects the residue ids touched by the
                // by-atom pass above.
                let hit: std::collections::HashSet<usize> = (0..natoms)
                    .filter(|&i| is_qm[i])
                    .map(|i| residue_ids[i])
                    .collect();
                for i in 0..natoms {
                    if hit.contains(&residue_ids[i]) {
                        is_qm[i] = true;
                    }
                }
                residue_ids_out = Some(residue_ids);
            }
        }

        let qm_indices: Vec<usize> = (0..natoms).filter(|&i| is_qm[i]).collect();
        let mm_indices: Vec<usize> = (0..natoms).filter(|&i| !is_qm[i]).collect();

        if qm_indices.is_empty() {
            return Err(FerricError::General(
                "QmmmSystem::new: QM region is empty — nothing to solve".to_string(),
            ));
        }

        Ok(Self {
            qm_indices,
            mm_indices,
            link_atoms: Vec::new(),
            effective_charges: atoms.iter().map(|a| a.charge).collect(),
            boundary_charges: Vec::new(),
            boundary_scheme: BoundaryChargeScheme::Keep,
            residue_ids: residue_ids_out,
            atoms: atoms.to_vec(),
            qm_charge,
            qm_multiplicity,
            link_atom_setup: None,
            boundary_charge_setup: None,
        })
    }

    /// Shared seed/radius validation for [`QmSelection::WithinRadius`] and
    /// [`QmSelection::WithinRadiusWholeResidues`]: nonempty seeds, seeds in
    /// range, and a finite non-negative radius.
    fn validate_radius_selection(
        seeds: &[usize],
        radius: f64,
        natoms: usize,
    ) -> Result<(), FerricError> {
        if seeds.is_empty() {
            return Err(FerricError::General(
                "QmmmSystem::new: WithinRadius selection needs at least one seed atom"
                    .to_string(),
            ));
        }
        // `is_finite` first, then a plain `<` — this rejects NaN via the
        // finiteness check rather than via a negated comparison
        // (`!(radius >= 0.0)` would also catch NaN, but reads as a typo; see
        // clippy::neg_cmp_op_on_partial_ord).
        if !radius.is_finite() || radius < 0.0 {
            return Err(FerricError::General(format!(
                "QmmmSystem::new: radius must be finite and >= 0, got {radius}"
            )));
        }
        for &s in seeds {
            if s >= natoms {
                return Err(FerricError::General(format!(
                    "QmmmSystem::new: seed index {s} out of range (structure has {natoms} atoms)"
                )));
            }
        }
        Ok(())
    }

    /// By-atom radius selection: `is_qm[i] = true` for every seed and every
    /// atom within `radius` of any seed. Shared by [`QmSelection::WithinRadius`]
    /// and the by-atom first pass of [`QmSelection::WithinRadiusWholeResidues`].
    fn apply_within_radius(atoms: &[QmmmAtom], seeds: &[usize], radius: f64, is_qm: &mut [bool]) {
        for &s in seeds {
            is_qm[s] = true;
        }
        let r2 = radius * radius;
        let natoms = atoms.len();
        for i in 0..natoms {
            if is_qm[i] {
                continue;
            }
            for &s in seeds {
                if atoms[i].distance2(&atoms[s]) <= r2 {
                    is_qm[i] = true;
                    break;
                }
            }
        }
    }

    /// Cap covalently cut bonds with scaled-position link hydrogens.
    ///
    /// `bonds` lists the covalent bonds of the **full** structure as index
    /// pairs into the full atom list. Any bond with exactly one endpoint in the
    /// QM region is a cut bond and gets a link hydrogen placed along it at
    /// `scale` of the original bond length, measured from the QM frontier atom.
    /// Bonds fully inside one region are ignored.
    ///
    /// Pass `scale = `[`DEFAULT_LINK_SCALE`] for the usual C-C case.
    ///
    /// **This is an approximation.** See the module docs: link atoms distort
    /// the boundary region, and the MM charge sitting on the severed partner is
    /// *not* deleted or redistributed here. Check
    /// [`QmmmSystem::min_link_to_charge_distance`] before trusting anything
    /// near the cut.
    pub fn with_link_atoms(
        mut self,
        bonds: &[(usize, usize)],
        scale: f64,
    ) -> Result<Self, FerricError> {
        if !(scale > 0.0 && scale < 1.0) {
            return Err(FerricError::General(format!(
                "with_link_atoms: scale must be in (0,1) — it is a fraction of the \
                 original bond length measured from the QM frontier atom; got {scale}"
            )));
        }
        let natoms = self.atoms.len();
        let mut is_qm = vec![false; natoms];
        for &i in &self.qm_indices {
            is_qm[i] = true;
        }

        let mut links = Vec::new();
        for &(a, b) in bonds {
            if a >= natoms || b >= natoms {
                return Err(FerricError::General(format!(
                    "with_link_atoms: bond ({a},{b}) out of range (structure has \
                     {natoms} atoms)"
                )));
            }
            // Exactly one endpoint quantum => the partition cut this bond.
            let (qm_full, mm_full) = match (is_qm[a], is_qm[b]) {
                (true, false) => (a, b),
                (false, true) => (b, a),
                _ => continue,
            };

            let qm_atom_pos = self.qm_indices.iter().position(|&i| i == qm_full).ok_or_else(
                || {
                    FerricError::General(format!(
                        "with_link_atoms: internal inconsistency — QM atom {qm_full} not \
                         found in qm_indices"
                    ))
                },
            )?;

            let p = &self.atoms[qm_full];
            let q = &self.atoms[mm_full];
            let dx = q.x - p.x;
            let dy = q.y - p.y;
            let dz = q.z_pos - p.z_pos;
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < 1e-20 {
                return Err(FerricError::General(format!(
                    "with_link_atoms: bonded atoms {qm_full} and {mm_full} coincide — \
                     cannot define a bond vector"
                )));
            }

            links.push(LinkAtomSpec {
                qm_atom: qm_atom_pos,
                mm_atom_full_index: mm_full,
                scale,
                // R_link = R_qm + g*(R_mm - R_qm): on the bond vector, at
                // fraction g of the original bond length.
                position: [p.x + scale * dx, p.y + scale * dy, p.z_pos + scale * dz],
            });
        }

        self.link_atoms = links;
        self.link_atom_setup = Some((bonds.to_vec(), scale));
        Ok(self)
    }

    /// Apply a [`BoundaryChargeScheme`] to the MM host atoms of every cut bond.
    ///
    /// `bonds` is the covalent bond list of the **full** structure (the same
    /// list [`QmmmSystem::with_link_atoms`] takes); it is needed here both to
    /// find the cut bonds (hence the M1 hosts) and, for RC/RCD, the M2 shell
    /// bonded to each M1. Link atoms are not required — the scheme acts on the
    /// cut bonds themselves — but the two are meant to be used together.
    ///
    /// Calling this again replaces the previous scheme rather than stacking
    /// on it. With no cut bond every scheme is an exact no-op.
    ///
    /// Errors for RC/RCD if some M1 has no MM neighbour to receive its charge
    /// (silently falling back to deletion would change the total charge
    /// behind the caller's back), or if an M2 is itself the host of another
    /// cut bond (the two redistributions would interfere; cut further from
    /// each other).
    pub fn with_boundary_charges(
        mut self,
        bonds: &[(usize, usize)],
        scheme: BoundaryChargeScheme,
    ) -> Result<Self, FerricError> {
        // Start from the raw charges so repeated calls are idempotent.
        self.effective_charges = self.atoms.iter().map(|a| a.charge).collect();
        self.boundary_charges.clear();
        self.boundary_scheme = scheme;

        let natoms = self.atoms.len();
        let mut is_qm = vec![false; natoms];
        for &i in &self.qm_indices {
            is_qm[i] = true;
        }
        for &(a, b) in bonds {
            if a >= natoms || b >= natoms {
                return Err(FerricError::General(format!(
                    "with_boundary_charges: bond ({a},{b}) out of range (structure has \
                     {natoms} atoms)"
                )));
            }
        }
        self.boundary_charge_setup = Some((bonds.to_vec(), scheme));

        // M1 hosts: the MM endpoint of every cut bond, deduplicated.
        let mut is_m1 = vec![false; natoms];
        let mut hosts: Vec<usize> = Vec::new();
        for &(a, b) in bonds {
            let m1 = match (is_qm[a], is_qm[b]) {
                (true, false) => b,
                (false, true) => a,
                _ => continue,
            };
            if !is_m1[m1] {
                is_m1[m1] = true;
                hosts.push(m1);
            }
        }
        if hosts.is_empty() || scheme == BoundaryChargeScheme::Keep {
            return Ok(self);
        }

        for &m1 in &hosts {
            let q_m1 = self.atoms[m1].charge;
            match scheme {
                BoundaryChargeScheme::Keep => unreachable!(),
                BoundaryChargeScheme::DeleteHost => {
                    self.effective_charges[m1] = 0.0;
                }
                BoundaryChargeScheme::RedistributedCharge
                | BoundaryChargeScheme::RedistributedChargeDipole => {
                    // M2 shell: MM neighbours of M1 that are not hosts themselves.
                    let mut m2s: Vec<usize> = Vec::new();
                    for &(a, b) in bonds {
                        let other = if a == m1 {
                            b
                        } else if b == m1 {
                            a
                        } else {
                            continue;
                        };
                        if is_qm[other] {
                            continue;
                        }
                        if is_m1[other] {
                            return Err(FerricError::General(format!(
                                "with_boundary_charges: MM atoms {m1} and {other} are both \
                                 hosts of cut bonds and bonded to each other — the \
                                 redistributions would interfere; move one cut further out"
                            )));
                        }
                        if !m2s.contains(&other) {
                            m2s.push(other);
                        }
                    }
                    if m2s.is_empty() {
                        return Err(FerricError::General(format!(
                            "with_boundary_charges: host MM atom {m1} has no MM neighbour in \
                             `bonds` to receive its charge ({scheme:?}); pass the full bond \
                             list, or use DeleteHost if losing q = {q_m1} is acceptable"
                        )));
                    }
                    let q0 = q_m1 / m2s.len() as f64;
                    self.effective_charges[m1] = 0.0;
                    let p1 = &self.atoms[m1];
                    for &m2 in &m2s {
                        let p2 = &self.atoms[m2];
                        let position = [
                            0.5 * (p1.x + p2.x),
                            0.5 * (p1.y + p2.y),
                            0.5 * (p1.z_pos + p2.z_pos),
                        ];
                        let q = if scheme == BoundaryChargeScheme::RedistributedChargeDipole {
                            // The M2 charge moves toward M1 by q0 to keep the
                            // M1–M2 bond dipole: 2q0 at the midpoint, −q0 on M2.
                            self.effective_charges[m2] -= q0;
                            2.0 * q0
                        } else {
                            q0
                        };
                        self.boundary_charges.push(BoundaryCharge { q, position, hosts: (m1, m2) });
                    }
                }
            }
        }
        Ok(self)
    }

    /// Every charge that enters the embedding potential, as `(q, position)`:
    /// the MM atoms with a nonzero effective charge in ascending index order,
    /// then the off-atom boundary charges. This single ordering is what
    /// [`QmmmSystem::to_external_potential`], [`QmmmSystem::mm_charge_positions`]
    /// and [`mm_forces`] all follow.
    fn active_charges(&self) -> Vec<(f64, [f64; 3])> {
        let mut out: Vec<(f64, [f64; 3])> = self
            .mm_indices
            .iter()
            .filter(|&&i| self.effective_charges[i] != 0.0)
            .map(|&i| {
                let a = &self.atoms[i];
                (self.effective_charges[i], [a.x, a.y, a.z_pos])
            })
            .collect();
        out.extend(self.boundary_charges.iter().filter(|b| b.q != 0.0).map(|b| (b.q, b.position)));
        out
    }

    /// The QM region as a [`Molecule`] ready for `PreparedBasis::new` and any
    /// `solve_*` call. Coordinates are Bohr (already, no conversion).
    ///
    /// Atom order is: the QM atoms in ascending full-structure index order,
    /// then the link hydrogens in `link_atoms` order. Use
    /// [`QmmmSystem::qm_atom_count`] to find where the link atoms start.
    pub fn to_qm_molecule(&self) -> Molecule {
        let mut mol_atoms: Vec<Atom> = self
            .qm_indices
            .iter()
            .map(|&i| {
                let a = &self.atoms[i];
                Atom {
                    symbol: a.symbol.clone(),
                    z: a.z,
                    x: a.x,
                    y: a.y,
                    zpos: a.z_pos,
                    ghost: false,
                    n_core_ecp: 0,
                }
            })
            .collect();

        for link in &self.link_atoms {
            mol_atoms.push(Atom {
                symbol: "H".to_string(),
                z: 1,
                x: link.position[0],
                y: link.position[1],
                zpos: link.position[2],
                ghost: false,
                n_core_ecp: 0,
            });
        }

        Molecule {
            atoms: mol_atoms,
            charge: self.qm_charge,
            multiplicity: self.qm_multiplicity,
        }
    }

    /// The MM region as an [`ExternalPotential`] of point charges, ready for
    /// `RhfConfig::external_potential`.
    ///
    /// Returns `None` when the MM region contributes nothing — either there are
    /// no MM atoms at all, or every MM charge is exactly zero. `None` is the
    /// literal gas-phase code path in every SCF variant, which is what makes
    /// the empty-MM no-op exact rather than merely small.
    pub fn to_external_potential(&self) -> Option<ExternalPotential> {
        let point_charges: Vec<PointCharge> = self
            .active_charges()
            .into_iter()
            .map(|(q, [x, y, z])| PointCharge { q, x, y, z })
            .collect();

        if point_charges.is_empty() {
            return None;
        }
        Some(ExternalPotential { point_charges, field: None })
    }

    /// Number of real QM atoms, excluding link hydrogens. Link atoms occupy
    /// `qm_atom_count() .. qm_atom_count() + link_atoms.len()` in the molecule
    /// returned by [`QmmmSystem::to_qm_molecule`].
    pub fn qm_atom_count(&self) -> usize {
        self.qm_indices.len()
    }

    /// Positions (Bohr) of the MM point charges that actually enter the
    /// embedding potential, in the same order as
    /// [`QmmmSystem::to_external_potential`]'s `point_charges` and as the rows
    /// [`mm_forces`] returns: atom-centred charges first (zero-charge MM atoms
    /// excluded, matching `to_external_potential`), then any off-atom
    /// boundary charges from RC/RCD.
    pub fn mm_charge_positions(&self) -> Vec<[f64; 3]> {
        self.active_charges().into_iter().map(|(_, p)| p).collect()
    }

    /// Rebuild this partition at a new geometry: same QM/MM selection, same
    /// link-atom bond list/scale and boundary-charge scheme (if any were
    /// configured), but atom positions taken from `coords_full` and link
    /// atoms / boundary charges re-derived from those new positions.
    ///
    /// `coords_full` is `(natoms, 3)` in Bohr, in the SAME atom ordering as
    /// [`QmmmSystem::atoms`] (i.e. the ordering the system was originally
    /// built from — this does not re-run [`QmSelection`], so it cannot
    /// change which atoms are QM vs MM).
    ///
    /// This exists so a geometry optimizer can re-evaluate the QM/MM energy
    /// and gradient at a new geometry with the partition — link positions,
    /// boundary midpoint charges — kept the EXACT derivative of what gets
    /// evaluated, rather than reusing stale link/boundary geometry from the
    /// old coordinates (which would make the analytic gradient wrong at any
    /// step that actually moved a frontier or boundary atom).
    pub fn with_coordinates(&self, coords_full: &Array2<f64>) -> Result<QmmmSystem, FerricError> {
        let natoms = self.atoms.len();
        if coords_full.dim() != (natoms, 3) {
            return Err(FerricError::General(format!(
                "with_coordinates: coords_full shape {:?} != ({natoms}, 3)",
                coords_full.dim()
            )));
        }

        let mut new_atoms = self.atoms.clone();
        for (i, a) in new_atoms.iter_mut().enumerate() {
            a.x = coords_full[(i, 0)];
            a.y = coords_full[(i, 1)];
            a.z_pos = coords_full[(i, 2)];
        }

        // Rebuild from scratch with the SAME selection outcome (qm_indices/
        // mm_indices/residue_ids are geometry-independent — QmSelection was
        // already resolved once at construction — so we reconstruct the
        // struct directly rather than re-running QmSelection, which would
        // need the original seeds/radius we don't keep around).
        let mut out = QmmmSystem {
            qm_indices: self.qm_indices.clone(),
            mm_indices: self.mm_indices.clone(),
            link_atoms: Vec::new(),
            atoms: new_atoms,
            qm_charge: self.qm_charge,
            qm_multiplicity: self.qm_multiplicity,
            effective_charges: Vec::new(), // set below
            boundary_charges: Vec::new(),
            boundary_scheme: BoundaryChargeScheme::Keep,
            residue_ids: self.residue_ids.clone(),
            link_atom_setup: self.link_atom_setup.clone(),
            boundary_charge_setup: self.boundary_charge_setup.clone(),
        };
        // effective_charges defaults to the raw per-atom charges until
        // with_boundary_charges (if configured) recomputes them below —
        // mirrors QmmmSystem::new's initialization.
        out.effective_charges = out.atoms.iter().map(|a| a.charge).collect();

        if let Some((bonds, scale)) = self.link_atom_setup.clone() {
            out = out.with_link_atoms(&bonds, scale)?;
        }
        if let Some((bonds, scheme)) = self.boundary_charge_setup.clone() {
            out = out.with_boundary_charges(&bonds, scheme)?;
        }

        Ok(out)
    }

    /// Shortest distance (Bohr) from any link hydrogen to any nonzero MM point
    /// charge, or `None` if there are no link atoms or no MM charges.
    ///
    /// A **diagnostic, not a guard** — nothing in this module acts on it. The
    /// classic link-atom pathology is a cap sitting on top of an MM charge,
    /// which over-polarizes the QM density catastrophically. If this comes back
    /// under roughly 1-2 Bohr you need a charge-redistribution scheme, and this
    /// module does not implement one (see the module docs).
    pub fn min_link_to_charge_distance(&self) -> Option<f64> {
        let charges = self.mm_charge_positions();
        if self.link_atoms.is_empty() || charges.is_empty() {
            return None;
        }
        let mut best = f64::INFINITY;
        for link in &self.link_atoms {
            for c in &charges {
                let dx = link.position[0] - c[0];
                let dy = link.position[1] - c[1];
                let dz = link.position[2] - c[2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                if d < best {
                    best = d;
                }
            }
        }
        Some(best)
    }
}

/// Electric field **E**(r) at arbitrary points from a QM density plus its
/// nuclei, in atomic units.
///
/// This is [`crate::properties::electric_field_at_atoms`] generalized off the
/// nuclei: same nuclear-attraction derivative-block contraction, same sign
/// convention, but the probe sits at a caller-supplied point and the nuclear
/// sum runs over *all* QM atoms (there is no self-term to skip, since the probe
/// is not one of them).
///
/// The field is what makes MM forces nearly free: **F** on a point charge `q` at
/// `r` is just `q·`**E**`(r)`, which is why [`mm_forces`] is a thin wrapper
/// over this.
///
/// `density` must be the **total** (spin-summed) density: `D_total` for a
/// restricted result, `D_α + D_β` for unrestricted/restricted-open. The
/// electric field is a one-electron property of the total electron density.
///
/// Errors if any point coincides with a QM nucleus (the nuclear sum diverges).
pub fn electric_field_at_points(
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
    points: &[[f64; 3]],
) -> Result<Vec<[f64; 3]>, FerricError> {
    use ferric_integrals::blas_threads::with_blas_threads;
    use ferric_integrals::engine::Engine;
    use ferric_integrals::ffi::{self, CAtom};
    use rayon::prelude::*;
    use std::os::raw::c_int;

    let nbas = prep.nbasis();
    if density.shape() != [nbas, nbas] {
        return Err(FerricError::General(format!(
            "electric_field_at_points: density shape {:?} != ({nbas},{nbas})",
            density.shape()
        )));
    }
    if points.is_empty() {
        return Ok(Vec::new());
    }

    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let nsh = prep.nshells();
    let max_fn = dims.iter().copied().max().unwrap_or(1);

    // With a single point charge libint returns 6 shell-center + 3 charge-center
    // derivative blocks. Hand-sized per caller, per the 1e-deriv sizing
    // reliability convention (never trust a shared/engine-side estimate).
    let nderiv = 6 + 3;
    let max_block = max_fn * max_fn;

    with_blas_threads(1, || {
        points
            .par_iter()
            .map_init(
                || {
                    let eng = Engine::new_1e_deriv(ffi::OP_NUCLEAR, prep, 1e-14);
                    let buf = vec![0.0_f64; nderiv * max_block];
                    (eng, buf)
                },
                |(eng, buf), r| -> Result<[f64; 3], FerricError> {
                    let eng = eng.as_mut().map_err(|e| {
                        FerricError::General(format!(
                            "electric_field_at_points: engine init failed: {e}"
                        ))
                    })?;

                    // Unit positive probe charge at r.
                    let probe = [CAtom { atomic_number: 1.0, x: r[0], y: r[1], z: r[2] }];
                    // SAFETY: probe is a stack-local CAtom slice; handle_mut() is the live engine
                    // pointer; probe.len() fits in c_int. Shim catches C++ exceptions → negative rc.
                    let rc = unsafe {
                        ffi::scf_engine_set_point_charges(
                            eng.handle_mut(),
                            probe.as_ptr(),
                            probe.len() as c_int,
                        )
                    };
                    if rc < 0 {
                        return Err(FerricError::General(format!(
                            "electric_field_at_points: set_point_charges failed (rc={rc})"
                        )));
                    }

                    // Electronic part: contract D with the charge-center
                    // derivative blocks (indices 6,7,8). Sign convention is
                    // identical to electric_field_at_atoms:
                    //   dM/dR_probe_d = -<μ|(r-R)_d/|r-R|³|ν>
                    //   E^elec_d      = -Σ_μν D_μν · dM/dR_probe_d
                    let mut e_elec = [0.0_f64; 3];
                    for s1 in 0..nsh {
                        for s2 in 0..=s1 {
                            let n1 = dims[s1];
                            let n2 = dims[s2];
                            let block_sz = n1 * n2;
                            let total = nderiv * block_sz;
                            if buf.len() < total {
                                buf.resize(total, 0.0);
                            }
                            // SAFETY: buf is pre-sized to nderiv * block_sz; handle_mut()/handle()
                            // are live pointers; shell indices are in range. Shim returns written >= 0.
                            let written = unsafe {
                                ffi::scf_compute_1e_deriv_block(
                                    eng.handle_mut(),
                                    prep.handle(),
                                    s1 as c_int,
                                    s2 as c_int,
                                    buf.as_mut_ptr(),
                                )
                            };
                            assert!(
                                written >= 0,
                                "libint2 internal error in nuclear deriv block \
                                 ({s1},{s2}): status {written}"
                            );
                            if written == 0 {
                                continue;
                            }
                            let o1 = offs[s1];
                            let o2 = offs[s2];
                            for d in 0..3 {
                                let blk_off = (6 + d) * block_sz;
                                let mut acc = 0.0_f64;
                                if s1 == s2 {
                                    for i in 0..n1 {
                                        for j in 0..n2 {
                                            acc += density[(o1 + i, o2 + j)]
                                                * buf[blk_off + i * n2 + j];
                                        }
                                    }
                                } else {
                                    // Off-diagonal shell pair: D and the operator
                                    // are both symmetric in (μ,ν), so the (s2,s1)
                                    // partner contributes equally => factor 2.
                                    for i in 0..n1 {
                                        for j in 0..n2 {
                                            acc += 2.0
                                                * density[(o1 + i, o2 + j)]
                                                * buf[blk_off + i * n2 + j];
                                        }
                                    }
                                }
                                e_elec[d] -= acc;
                            }
                        }
                    }

                    // Nuclear part: Σ_A Z_A (r − R_A)/|r − R_A|³ over all QM
                    // nuclei (no self-term to skip — the probe is not a nucleus).
                    let mut e_nuc = [0.0_f64; 3];
                    for atom in &mol.atoms {
                        let za = atom.effective_z() as f64;
                        if za == 0.0 {
                            continue;
                        }
                        let dx = r[0] - atom.x;
                        let dy = r[1] - atom.y;
                        let dz = r[2] - atom.zpos;
                        let r2 = dx * dx + dy * dy + dz * dz;
                        let rr = r2.sqrt();
                        if rr < 1e-12 {
                            return Err(FerricError::General(
                                "electric_field_at_points: probe point coincides with a \
                                 QM nucleus (field diverges)"
                                    .to_string(),
                            ));
                        }
                        let inv_r3 = 1.0 / (r2 * rr);
                        e_nuc[0] += za * dx * inv_r3;
                        e_nuc[1] += za * dy * inv_r3;
                        e_nuc[2] += za * dz * inv_r3;
                    }

                    Ok([
                        e_elec[0] + e_nuc[0],
                        e_elec[1] + e_nuc[1],
                        e_elec[2] + e_nuc[2],
                    ])
                },
            )
            .collect::<Result<Vec<[f64; 3]>, FerricError>>()
    })
}

/// Force (a.u., **not** dE/dR) exerted on each MM point charge by the converged
/// QM region — electrons and nuclei both.
///
/// `F_i = q_i · E(r_i)`, where `E` is the total QM field from
/// [`electric_field_at_points`]. Rows are in the same order as
/// [`QmmmSystem::mm_charge_positions`] and as `to_external_potential`'s
/// `point_charges` — i.e. zero-charge MM atoms are excluded, not zero-filled.
///
/// Note the sign convention differs from the rest of this codebase's gradient
/// routines, which return `dE/dR`: this returns the **force** `-dE/dR`, because
/// "the force on the MM charges" is the physically meaningful quantity a caller
/// hands to an MM integrator. Negate it for a gradient.
///
/// `mol`/`prep`/`density` must describe the QM region as actually solved
/// (including any link hydrogens). `density` is the total spin-summed density.
///
/// Only the electrostatic QM→MM force is included. There is no QM-MM van der
/// Waals term and no MM-MM force here (no force field in this module), and the
/// link-atom chain-rule projection onto host atoms is not applied.
pub fn mm_forces(
    system: &QmmmSystem,
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
) -> Result<Vec<[f64; 3]>, FerricError> {
    let positions = system.mm_charge_positions();
    if positions.is_empty() {
        return Ok(Vec::new());
    }
    let charges: Vec<f64> = system
        .to_external_potential()
        .map(|ep| ep.point_charges.iter().map(|pc| pc.q).collect())
        .unwrap_or_default();

    let field = electric_field_at_points(mol, prep, density, &positions)?;
    Ok(field
        .iter()
        .zip(charges.iter())
        .map(|(e, &q)| [q * e[0], q * e[1], q * e[2]])
        .collect())
}

/// Gradient `dE/dR` of the QM/MM energy with respect to every **real** atom
/// of the full structure, `(natoms_full, 3)`, assembled from the two pieces
/// the solvers already produce:
///
/// - `qm_gradient`: the gradient on the QM molecule as actually solved
///   (real QM atoms then link hydrogens, i.e. `rhf_gradient`/`uhf_gradient`/
///   `ks_gradient_*` on [`QmmmSystem::to_qm_molecule`] with the embedding
///   potential passed as `ext`);
/// - `mm_forces`: the **force** on each embedding charge from [`mm_forces`],
///   in [`QmmmSystem::mm_charge_positions`] order.
///
/// Three chain rules are applied:
///
/// 1. **Link atoms.** `R_L = (1−g)·R_Q + g·R_M`, so the link row projects
///    as `(1−g)·∂E/∂R_L` onto the QM frontier atom and `g·∂E/∂R_L` onto the
///    MM host. The column sums of the result equal those of the inputs (a
///    rigid translation moves the cap rigidly).
/// 2. **Atom-centred MM charges.** `∂E/∂R = −F` on that atom.
/// 3. **Boundary (midpoint) charges** from RC/RCD sit at `(R_M1 + R_M2)/2`,
///    so `−F` is split half onto each host.
///
/// What this does NOT contain: any MM–MM or QM–MM van der Waals term (no
/// force field here), and any dependence of the MM *charges themselves* on
/// geometry (fixed by construction). Rows of MM atoms with zero effective
/// charge and no boundary role are exactly zero.
///
/// Verified against central finite differences of the total SCF energy with
/// the partition rebuilt at every displaced geometry, on the RCD-treated
/// ethane cut (`tests/qmmm.rs`).
pub fn full_gradient(
    system: &QmmmSystem,
    qm_gradient: &Array2<f64>,
    mm_forces: &[[f64; 3]],
) -> Result<Array2<f64>, FerricError> {
    let n_qm = system.qm_atom_count();
    let n_link = system.link_atoms.len();
    if qm_gradient.dim() != (n_qm + n_link, 3) {
        return Err(FerricError::General(format!(
            "full_gradient: qm_gradient has shape {:?}, expected ({}, 3) = {n_qm} QM atoms + \
             {n_link} link atoms",
            qm_gradient.dim(),
            n_qm + n_link
        )));
    }
    let charges = system.active_charges();
    if mm_forces.len() != charges.len() {
        return Err(FerricError::General(format!(
            "full_gradient: {} MM force rows for {} embedding charges (use mm_forces() on \
             the same QmmmSystem)",
            mm_forces.len(),
            charges.len()
        )));
    }

    let natoms = system.atoms.len();
    let mut full = Array2::<f64>::zeros((natoms, 3));

    // 1. Real QM rows straight through; link rows onto their two hosts.
    for (row, &full_idx) in system.qm_indices.iter().enumerate() {
        for k in 0..3 {
            full[(full_idx, k)] += qm_gradient[(row, k)];
        }
    }
    for (l, link) in system.link_atoms.iter().enumerate() {
        let row = n_qm + l;
        let q_full = system.qm_indices[link.qm_atom];
        let g = link.scale;
        for k in 0..3 {
            let d = qm_gradient[(row, k)];
            full[(q_full, k)] += (1.0 - g) * d;
            full[(link.mm_atom_full_index, k)] += g * d;
        }
    }

    // 2./3. MM forces: atom-centred rows first, then boundary charges — the
    // same order active_charges() produces.
    let n_atom_centred = system
        .mm_indices
        .iter()
        .filter(|&&i| system.effective_charges[i] != 0.0)
        .count();
    let atom_rows = system.mm_indices.iter().copied().filter(|&i| system.effective_charges[i] != 0.0);
    for (i, f) in atom_rows.zip(mm_forces.iter()) {
        for k in 0..3 {
            full[(i, k)] -= f[k];
        }
    }
    let boundary_rows = system.boundary_charges.iter().filter(|b| b.q != 0.0);
    for (b, f) in boundary_rows.zip(mm_forces[n_atom_centred..].iter()) {
        let (m1, m2) = b.hosts;
        for k in 0..3 {
            full[(m1, k)] -= 0.5 * f[k];
            full[(m2, k)] -= 0.5 * f[k];
        }
    }

    Ok(full)
}

/// The MM-force-field contribution to a QM/MM total energy and its gradient
/// over the **full** structure, under the additive-embedding convention
/// (spec §6):
///
/// - **MM-MM bonded terms** (bond/angle/torsion) are kept if the term
///   involves **at least one** MM atom, and dropped if every atom in the
///   term is QM (the QM density/wavefunction already describes that
///   interaction; keeping it too would double-count). This is the standard
///   "additive" QM/MM scheme: a bond, angle, or torsion straddling the
///   QM/MM boundary is treated classically in full.
/// - **MM-MM nonbonded** (LJ + Coulomb) is summed only between pairs of MM
///   atoms, using [`MmTopology`]'s own derived exclusions/1-4 scaling.
/// - **QM-MM Lennard-Jones**: every QM real atom (link atoms and boundary
///   midpoint charges carry no LJ) paired with every MM atom, using each
///   atom's own `top.lj` entry (Lorentz-Berthelot mixed), no cutoff, no
///   exclusions (a QM/MM boundary link atom sitting between them is not an
///   MM-topology bond, so there is nothing to exclude by construction —
///   the two regions were never bonded IN THE TOPOLOGY, only in reality,
///   which is exactly why this is an approximation like any additive
///   scheme).
/// - **No QM-MM Coulomb term**: the electrostatic embedding already lets
///   the QM density see every MM point charge (`to_external_potential`);
///   adding a classical Coulomb term between QM atoms' `top.charges` entry
///   and MM charges would double count that interaction. `top.charges` for
///   QM atoms is therefore only consulted for possible QM-QM terms, which
///   this function never computes (those are also already inside the QM
///   Hamiltonian).
/// - Link atoms and any RC/RCD boundary midpoint charges carry no MM terms
///   at all (they are not entries in `top`, which is indexed by full
///   structure atoms only).
///
/// `top.n_atoms()` must equal the number of atoms in the full structure
/// (`system.atoms.len()`), and `coords_full` must be `(n_atoms, 3)` in
/// Bohr and in the SAME ordering as `system.atoms`.
///
/// **Exactness anchor**: an empty topology (see [`MmTopology::new`] with
/// empty charges/lj/bonds/angles/torsions and `n_atoms() == 0`... actually
/// use one with `n_atoms() == system.atoms.len()` but zero LJ epsilon/zero
/// charges/no bonded terms) gives exactly zero energy and gradient — see
/// `qmmm_mm_terms_with_empty_topology_is_zero` in `tests/qmmm_mm.rs`.
pub fn qmmm_mm_terms(
    system: &QmmmSystem,
    top: &MmTopology,
    coords_full: &Array2<f64>,
) -> Result<(MmEnergy, Array2<f64>), FerricError> {
    let natoms = system.atoms.len();
    if top.n_atoms() != natoms {
        return Err(FerricError::General(format!(
            "qmmm_mm_terms: topology has {} atoms but the QM/MM system has {natoms}",
            top.n_atoms()
        )));
    }
    if coords_full.dim() != (natoms, 3) {
        return Err(FerricError::General(format!(
            "qmmm_mm_terms: coords_full shape {:?} != ({natoms}, 3)",
            coords_full.dim()
        )));
    }

    let mut is_qm = vec![false; natoms];
    for &i in &system.qm_indices {
        is_qm[i] = true;
    }

    // 1. Bonded terms: keep any term with >=1 MM atom, drop all-QM terms.
    //    Build a filtered sub-topology (bonds/angles/torsions only; charges
    //    and lj pass through unchanged since the MM-MM nonbonded and
    //    QM-MM LJ passes below need them for every atom).
    let bonds: Vec<_> = top.bonds.iter().filter(|b| !is_qm[b.i] || !is_qm[b.j]).cloned().collect();
    let angles: Vec<_> =
        top.angles.iter().filter(|a| !is_qm[a.i] || !is_qm[a.j] || !is_qm[a.k]).cloned().collect();
    let torsions: Vec<_> = top
        .torsions
        .iter()
        .filter(|t| !is_qm[t.i] || !is_qm[t.j] || !is_qm[t.k] || !is_qm[t.l])
        .cloned()
        .collect();

    // Bonded energy/gradient: a topology with the surviving (>=1 MM atom)
    // bonds/angles/torsions but ZERO charges/LJ, so ferric_mm::gradient's
    // own nonbonded pass (which would otherwise run over every pair not
    // excluded by THIS filtered bond graph — including QM-QM pairs no
    // longer excluded once their QM-QM bonds were dropped) contributes
    // nothing. Nonbonded terms are computed separately below, scoped
    // correctly (MM-MM only, and QM-MM LJ only).
    let zero_lj: Vec<ferric_mm::LjParams> =
        vec![ferric_mm::LjParams { sigma: 0.0, epsilon: 0.0 }; natoms];
    let bonded_only_top = MmTopology::new(vec![0.0; natoms], zero_lj, bonds, angles, torsions)
        .map_err(|e| FerricError::General(format!("qmmm_mm_terms: {e}")))?;
    let (e_bonded, mut g) = ferric_mm::gradient(&bonded_only_top, coords_full)
        .map_err(|e| FerricError::General(format!("qmmm_mm_terms: {e}")))?;

    // 2. MM-MM nonbonded: charges/lj zeroed on QM atoms so only MM-MM pairs
    //    contribute, using the ORIGINAL (unfiltered) bond graph for
    //    exclusions/1-4 (an MM-MM pair that is 1-2/1-3/1-4 through a path
    //    that crosses the QM region is still a real bonded relationship in
    //    the topology and must still be excluded/scaled the same way).
    let mm_only_charges: Vec<f64> =
        (0..natoms).map(|i| if is_qm[i] { 0.0 } else { top.charges[i] }).collect();
    let mm_only_lj: Vec<ferric_mm::LjParams> = (0..natoms)
        .map(|i| if is_qm[i] { ferric_mm::LjParams { sigma: 0.0, epsilon: 0.0 } } else { top.lj[i] })
        .collect();
    // The bond list here feeds ONLY MmTopology::new's exclusion/1-4 BFS
    // (over the ORIGINAL, unfiltered graph — see the doc comment above); it
    // must NOT also contribute a real harmonic bond energy/gradient a
    // second time (that would double the bond term the first pass already
    // counted, and at a geometry sitting almost exactly at r0 it would
    // instead surface as a machine-epsilon-scale bond GRADIENT residual
    // from `r - r0` not being exactly representable — found and fixed
    // during TDD via an all-QM anchor test that expected an exact 0.0
    // gradient and got ~3e-16 instead). Force k=0 on every bond so the
    // BFS-derived graph shape is preserved but the harmonic term itself is
    // identically zero (0.0 * anything = 0.0 exactly in IEEE754,
    // regardless of r0 or floating-point noise in r).
    let graph_only_bonds: Vec<ferric_mm::Bond> =
        top.bonds.iter().map(|b| ferric_mm::Bond { k: 0.0, ..*b }).collect();
    let mm_only_top = MmTopology::new(mm_only_charges, mm_only_lj, graph_only_bonds, vec![], vec![])
        .map_err(|e| FerricError::General(format!("qmmm_mm_terms: {e}")))?
        .with_scales(top.scale_lj_14, top.scale_coul_14);
    let (e_mm_nb, g_mm_nb) = ferric_mm::gradient(&mm_only_top, coords_full)
        .map_err(|e| FerricError::General(format!("qmmm_mm_terms: {e}")))?;

    let mut e = MmEnergy {
        bond: e_bonded.bond,
        angle: e_bonded.angle,
        torsion: e_bonded.torsion,
        lj: e_mm_nb.lj,
        coulomb: e_mm_nb.coulomb,
        total: 0.0,
    };
    for i in 0..natoms {
        for c in 0..3 {
            g[(i, c)] += g_mm_nb[(i, c)];
        }
    }

    // 3. QM-MM Lennard-Jones: every real QM atom (link atoms/boundary
    //    charges excluded by construction — they are not `system.atoms`
    //    indices) paired with every MM atom, using top.lj on both sides.
    let qm_lj: Vec<ferric_mm::LjParams> = system.qm_indices.iter().map(|&i| top.lj[i]).collect();
    let mm_lj: Vec<ferric_mm::LjParams> = system.mm_indices.iter().map(|&i| top.lj[i]).collect();
    if !qm_lj.is_empty() && !mm_lj.is_empty() {
        let mut coords_qm = Array2::<f64>::zeros((system.qm_indices.len(), 3));
        for (row, &i) in system.qm_indices.iter().enumerate() {
            for c in 0..3 {
                coords_qm[(row, c)] = coords_full[(i, c)];
            }
        }
        let mut coords_mm = Array2::<f64>::zeros((system.mm_indices.len(), 3));
        for (row, &i) in system.mm_indices.iter().enumerate() {
            for c in 0..3 {
                coords_mm[(row, c)] = coords_full[(i, c)];
            }
        }
        let (e_qm_mm_lj, g_qm, g_mm) =
            ferric_mm::qm_mm_lj_energy_gradient(&qm_lj, &coords_qm, &mm_lj, &coords_mm);
        e.lj += e_qm_mm_lj;
        for (row, &i) in system.qm_indices.iter().enumerate() {
            for c in 0..3 {
                g[(i, c)] += g_qm[(row, c)];
            }
        }
        for (row, &i) in system.mm_indices.iter().enumerate() {
            for c in 0..3 {
                g[(i, c)] += g_mm[(row, c)];
            }
        }
    }

    e.total = e.bond + e.angle + e.torsion + e.lj + e.coulomb;
    Ok((e, g))
}

/// [`full_gradient`] plus the MM force-field contribution from
/// [`qmmm_mm_terms`], summed row-wise over the full structure. `mm_gradient`
/// is `qmmm_mm_terms(...).1` (dE/dR, NOT a force) evaluated at the same
/// `coords_full` the [`QmmmSystem`] was built from.
pub fn full_gradient_with_mm(
    system: &QmmmSystem,
    qm_gradient: &Array2<f64>,
    mm_forces_vec: &[[f64; 3]],
    mm_gradient: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let mut full = full_gradient(system, qm_gradient, mm_forces_vec)?;
    let natoms = system.atoms.len();
    if mm_gradient.dim() != (natoms, 3) {
        return Err(FerricError::General(format!(
            "full_gradient_with_mm: mm_gradient shape {:?} != ({natoms}, 3)",
            mm_gradient.dim()
        )));
    }
    for i in 0..natoms {
        for c in 0..3 {
            full[(i, c)] += mm_gradient[(i, c)];
        }
    }
    Ok(full)
}

// ── F2-2: optimize_qmmm ──

/// Which MM atoms, in addition to the QM real atoms (always free), the
/// optimizer is allowed to move.
///
/// A [`QmmmOptimizeConfig::mm_topology`] is required whenever this is
/// anything other than `None`: moving MM atoms with no MM–MM force field to
/// resist that motion (bond/angle/torsion/LJ/Coulomb) is meaningless — they
/// would be free-falling into the QM region's electrostatic embedding with
/// nothing holding their own geometry together.
#[derive(Debug, Clone, PartialEq)]
pub enum MoveMm {
    /// Only QM real atoms move; every MM atom (and any boundary/link
    /// machinery) stays fixed at its input geometry. The only variant valid
    /// without `mm_topology`.
    None,
    /// MM atoms within `r` Bohr of any QM real atom, measured **once, at the
    /// starting geometry** (not re-evaluated as atoms move — a moving
    /// "free" set would make the coordinate vector's length change
    /// mid-optimization).
    WithinRadius(f64),
    /// MM atoms belonging to any of these residue ids. Requires
    /// `system.residue_ids` to be `Some(...)` (i.e. the system was built
    /// with [`QmSelection::WithinRadiusWholeResidues`]) — a typed error
    /// otherwise.
    Residues(Vec<usize>),
    /// Every MM atom moves.
    All,
}

/// Which SCF reference to optimize with. Mirrors `RhfConfig.xc`: `Rhf`/`Uhf`
/// run plain HF; `Rks`/`Uks` set `RhfConfig.xc` to the given functional name
/// (df_j_aux/df_k_aux auto-set to `def2-universal-jkfit`, matching
/// `run_qmmm`'s KS setup).
#[derive(Debug, Clone, PartialEq)]
pub enum QmmmMethod {
    Rhf,
    Uhf,
    Rks(String),
    Uks(String),
}

/// Configuration for [`optimize_qmmm`].
#[derive(Debug, Clone)]
pub struct QmmmOptimizeConfig {
    pub method: QmmmMethod,
    pub move_mm: MoveMm,
    pub opt: crate::optimize::OptimizeConfig,
    /// The MM force field. `None` means no MM energy/gradient terms at all —
    /// the same "all-zero" contract [`qmmm_mm_terms`] documents. Required
    /// (an error otherwise) whenever `move_mm != MoveMm::None`.
    pub mm_topology: Option<MmTopology>,
    /// SCF configuration. `external_potential` is OVERWRITTEN with the
    /// system's embedding potential at every step (any value set here is
    /// ignored, since it must track the moving partition); every other
    /// field (xc aux basis choices excepted — those are set by `method`)
    /// passes through unchanged.
    pub scf: RhfConfig,
}

/// Result of [`optimize_qmmm`].
#[derive(Debug, Clone)]
pub struct QmmmOptimizeResult {
    /// The partition at the final geometry (same QM/MM selection as the
    /// input `system`, atoms moved to the optimized positions).
    pub system: QmmmSystem,
    /// Total energy (SCF + MM force-field terms) at the final geometry.
    pub energy: f64,
    pub converged: bool,
    pub steps: usize,
    /// Total energy at every step, in order (length `steps + 1`).
    pub energies: Vec<f64>,
}

/// Which atoms (by full-structure index) are free to move, given `move_mm`.
/// QM real atoms are always free; link atoms and boundary midpoint charges
/// are never independent coordinates (they are derived, via
/// [`QmmmSystem::with_coordinates`], from the real atoms that host them).
fn free_atom_indices(system: &QmmmSystem, move_mm: &MoveMm) -> Result<Vec<usize>, FerricError> {
    let mut free: Vec<usize> = system.qm_indices.clone();
    match move_mm {
        MoveMm::None => {}
        MoveMm::All => {
            free.extend(system.mm_indices.iter().copied());
        }
        MoveMm::WithinRadius(r) => {
            if !(r.is_finite() && *r >= 0.0) {
                return Err(FerricError::General(format!(
                    "optimize_qmmm: MoveMm::WithinRadius radius must be finite and >= 0, got {r}"
                )));
            }
            let r2 = r * r;
            for &m in &system.mm_indices {
                let hit = system
                    .qm_indices
                    .iter()
                    .any(|&q| system.atoms[m].distance2(&system.atoms[q]) <= r2);
                if hit {
                    free.push(m);
                }
            }
        }
        MoveMm::Residues(ids) => {
            let Some(residue_ids) = &system.residue_ids else {
                return Err(FerricError::General(
                    "optimize_qmmm: MoveMm::Residues requires system.residue_ids (build the \
                     QmmmSystem with QmSelection::WithinRadiusWholeResidues)"
                        .to_string(),
                ));
            };
            let wanted: std::collections::HashSet<usize> = ids.iter().copied().collect();
            for &m in &system.mm_indices {
                if wanted.contains(&residue_ids[m]) {
                    free.push(m);
                }
            }
        }
    }
    free.sort_unstable();
    Ok(free)
}

/// Optimize a QM/MM system's geometry: real QM atoms always move, MM atoms
/// move per `cfg.move_mm`. Every energy/gradient evaluation rebuilds the
/// partition at the current geometry via [`QmmmSystem::with_coordinates`] (so
/// link atoms and boundary charges are the exact derivative of what gets
/// evaluated), runs the configured SCF in the resulting embedding potential,
/// adds any [`qmmm_mm_terms`] contribution, and projects the gradient rows of
/// every FROZEN atom to zero before the BFGS line search sees them (so it
/// never tries to move an atom `move_mm` excluded).
///
/// Requires `cfg.mm_topology` whenever `cfg.move_mm != MoveMm::None`
/// (typed error otherwise — moving MM atoms with no force field holding
/// their own geometry together is meaningless). `MoveMm::Residues` requires
/// `system.residue_ids` (typed error otherwise).
///
/// **Exactness anchor**: `move_mm = MoveMm::None` with no MM atoms in
/// `system` reproduces [`crate::optimize::optimize_geometry`]'s per-step
/// energies exactly (same BFGS core, same gradient) — see
/// `optimize_qmmm_with_no_mm_matches_optimize_geometry` in
/// `tests/qmmm_optimize.rs`.
pub fn optimize_qmmm(
    ctx: &ParallelContext,
    system: &QmmmSystem,
    basis_name: &str,
    cfg: &QmmmOptimizeConfig,
) -> Result<QmmmOptimizeResult, FerricError> {
    if cfg.move_mm != MoveMm::None && cfg.mm_topology.is_none() {
        return Err(FerricError::General(
            "optimize_qmmm: move_mm != MoveMm::None requires mm_topology (moving MM atoms \
             with no MM force field to hold their own geometry together is meaningless)"
                .to_string(),
        ));
    }

    let free = free_atom_indices(system, &cfg.move_mm)?;
    let natoms = system.atoms.len();

    let coords0 = {
        let mut c = Array2::<f64>::zeros((natoms, 3));
        for (i, a) in system.atoms.iter().enumerate() {
            c[(i, 0)] = a.x;
            c[(i, 1)] = a.y;
            c[(i, 2)] = a.z_pos;
        }
        c
    };

    // Flatten only the FREE atoms' coordinates into the BFGS vector.
    let flatten_free = |coords: &Array2<f64>| -> Vec<f64> {
        let mut x = Vec::with_capacity(free.len() * 3);
        for &i in &free {
            x.push(coords[(i, 0)]);
            x.push(coords[(i, 1)]);
            x.push(coords[(i, 2)]);
        }
        x
    };
    let unflatten_free = |x: &[f64], coords: &mut Array2<f64>| {
        for (k, &i) in free.iter().enumerate() {
            coords[(i, 0)] = x[3 * k];
            coords[(i, 1)] = x[3 * k + 1];
            coords[(i, 2)] = x[3 * k + 2];
        }
    };

    let x0 = flatten_free(&coords0);
    let bs = ferric_core::basis::bundled(basis_name)?;
    let op = Operator::coulomb();

    let mut energies: Vec<f64> = Vec::new();

    let (x_final, _energy_final, steps, converged) =
        crate::optimize::optimize_coordinates(&x0, &cfg.opt, |x| {
            let mut coords = coords0.clone();
            unflatten_free(x, &mut coords);
            let step_sys = system.with_coordinates(&coords)?;

            let mol = step_sys.to_qm_molecule();
            let prep = PreparedBasis::new(&mol, &bs)?;
            let bounds = SchwarzBounds::compute(op, &prep)?;
            let mut scf_cfg = cfg.scf.clone();
            scf_cfg.external_potential = step_sys.to_external_potential();
            let ext = scf_cfg.external_potential.clone();

            let (scf_energy, qm_grad, density_total) = match &cfg.method {
                QmmmMethod::Rhf => {
                    let r = crate::rhf::solve_rhf(ctx, &mol, &prep, op, &bounds, &scf_cfg)?;
                    let g = crate::gradient::rhf_gradient(&mol, &prep, op, &bounds, &r, ext.as_ref())?;
                    (r.energy, g, r.density_total().clone())
                }
                QmmmMethod::Uhf => {
                    let r = crate::uhf::solve_uhf(ctx, &mol, &prep, &bounds, &scf_cfg)?;
                    let g = crate::gradient::uhf_gradient(&mol, &prep, op, &bounds, &r, ext.as_ref())?;
                    (r.energy, g, r.density_total().clone())
                }
                QmmmMethod::Rks(xc) => {
                    scf_cfg.xc = Some(xc.clone());
                    scf_cfg.df_j_aux = Some("def2-universal-jkfit".to_string());
                    scf_cfg.df_k_aux = Some("def2-universal-jkfit".to_string());
                    let r = crate::rhf::solve_rhf(ctx, &mol, &prep, op, &bounds, &scf_cfg)?;
                    let g = crate::ks_gradient::ks_gradient_closed(
                        &mol, &prep, &bs, op, &bounds, xc, &r, ext.as_ref(),
                    )?;
                    (r.energy, g, r.density_total().clone())
                }
                QmmmMethod::Uks(xc) => {
                    scf_cfg.xc = Some(xc.clone());
                    scf_cfg.df_j_aux = Some("def2-universal-jkfit".to_string());
                    scf_cfg.df_k_aux = Some("def2-universal-jkfit".to_string());
                    let r = crate::uhf::solve_uhf(ctx, &mol, &prep, &bounds, &scf_cfg)?;
                    let g = crate::ks_gradient::ks_gradient_uks(
                        &mol, &prep, &bs, op, &bounds, xc, &r, ext.as_ref(),
                    )?;
                    (r.energy, g, r.density_total().clone())
                }
            };

            let forces = mm_forces(&step_sys, &mol, &prep, &density_total)?;

            let (total_energy, full_grad) = match &cfg.mm_topology {
                Some(top) => {
                    let (mm_e, mm_g) = qmmm_mm_terms(&step_sys, top, &coords)?;
                    let full = full_gradient_with_mm(&step_sys, &qm_grad, &forces, &mm_g)?;
                    (scf_energy + mm_e.total, full)
                }
                None => {
                    let full = full_gradient(&step_sys, &qm_grad, &forces)?;
                    (scf_energy, full)
                }
            };

            energies.push(total_energy);

            // Project: zero the gradient rows of every FROZEN atom before
            // BFGS sees them, then extract only the free-atom rows into the
            // flat vector the optimizer core operates on.
            let mut grad_free = Vec::with_capacity(free.len() * 3);
            for &i in &free {
                for c in 0..3 {
                    grad_free.push(full_grad[(i, c)]);
                }
            }
            Ok((total_energy, grad_free))
        })?;

    let mut coords_final = coords0;
    unflatten_free(&x_final, &mut coords_final);
    let final_system = system.with_coordinates(&coords_final)?;
    let final_energy = *energies.last().expect("at least one energy evaluation always occurs");

    Ok(QmmmOptimizeResult {
        system: final_system,
        energy: final_energy,
        converged,
        steps,
        energies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two atoms 2 Bohr apart on z, plus a distant third.
    fn three_atoms() -> Vec<QmmmAtom> {
        vec![
            QmmmAtom::new("C", 6, 0.0, 0.0, 0.0, -0.1),
            QmmmAtom::new("C", 6, 0.0, 0.0, 2.0, 0.1),
            QmmmAtom::new("O", 8, 0.0, 0.0, 10.0, -0.5),
        ]
    }

    #[test]
    fn boundary_scheme_parse_is_strict() {
        use BoundaryChargeScheme::*;
        assert_eq!(BoundaryChargeScheme::parse_config_str("keep").unwrap(), Keep);
        assert_eq!(BoundaryChargeScheme::parse_config_str("delete-host").unwrap(), DeleteHost);
        assert_eq!(BoundaryChargeScheme::parse_config_str("rc").unwrap(), RedistributedCharge);
        assert_eq!(BoundaryChargeScheme::parse_config_str("rcd").unwrap(), RedistributedChargeDipole);
        // Config honesty: unknown or differently-cased values error, never
        // silently default.
        for bad in ["Keep", "z1", "RCD", "", "redistributed"] {
            assert!(BoundaryChargeScheme::parse_config_str(bad).is_err(), "{bad:?}");
        }
        for s in [Keep, DeleteHost, RedistributedCharge, RedistributedChargeDipole] {
            assert_eq!(BoundaryChargeScheme::parse_config_str(s.config_str()).unwrap(), s);
        }
    }

    #[test]
    fn indices_selection_partitions_disjointly() {
        let atoms = three_atoms();
        let sys =
            QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1]), 0, 1).unwrap();
        assert_eq!(sys.qm_indices, vec![0, 1]);
        assert_eq!(sys.mm_indices, vec![2]);
        assert_eq!(sys.qm_atom_count(), 2);
    }

    #[test]
    fn within_radius_picks_up_neighbours_and_excludes_far_atoms() {
        let atoms = three_atoms();
        // Seed on atom 0, radius 3 Bohr: catches atom 1 (2 Bohr) not atom 2 (10).
        let sys = QmmmSystem::new(
            &atoms,
            QmSelection::WithinRadius { seeds: vec![0], radius: 3.0 },
            0,
            1,
        )
        .unwrap();
        assert_eq!(sys.qm_indices, vec![0, 1]);
        assert_eq!(sys.mm_indices, vec![2]);
    }

    #[test]
    fn within_radius_zero_selects_only_seeds() {
        let atoms = three_atoms();
        let sys = QmmmSystem::new(
            &atoms,
            QmSelection::WithinRadius { seeds: vec![1], radius: 0.0 },
            0,
            1,
        )
        .unwrap();
        assert_eq!(sys.qm_indices, vec![1]);
        assert_eq!(sys.mm_indices, vec![0, 2]);
    }

    #[test]
    fn empty_mm_region_yields_no_external_potential() {
        // THE exactness contract: all-QM => None => literal gas-phase path.
        let atoms = three_atoms();
        let sys =
            QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1, 2]), 0, 1).unwrap();
        assert!(sys.mm_indices.is_empty());
        assert!(sys.to_external_potential().is_none());
    }

    #[test]
    fn all_zero_mm_charges_yield_no_external_potential() {
        // An MM region that is present but electrostatically inert must also
        // collapse to the exact gas-phase path, not to a potential full of zeros.
        let mut atoms = three_atoms();
        for a in atoms.iter_mut() {
            a.charge = 0.0;
        }
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0]), 0, 1).unwrap();
        assert_eq!(sys.mm_indices, vec![1, 2]);
        assert!(sys.to_external_potential().is_none());
    }

    #[test]
    fn qm_atom_mm_charges_are_not_double_counted() {
        let atoms = three_atoms();
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1]), 0, 1).unwrap();
        let ep = sys.to_external_potential().unwrap();
        // Only atom 2 is MM; atoms 0/1 carry charges but are quantum.
        assert_eq!(ep.point_charges.len(), 1);
        assert_eq!(ep.point_charges[0].q, -0.5);
        assert_eq!(ep.point_charges[0].z, 10.0);
    }

    #[test]
    fn to_qm_molecule_preserves_coordinates_and_charge_state() {
        let atoms = three_atoms();
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 2]), -1, 2).unwrap();
        let mol = sys.to_qm_molecule();
        assert_eq!(mol.atoms.len(), 2);
        assert_eq!(mol.charge, -1);
        assert_eq!(mol.multiplicity, 2);
        assert_eq!(mol.atoms[0].z, 6);
        assert_eq!(mol.atoms[1].z, 8);
        assert_eq!(mol.atoms[1].zpos, 10.0);
        assert!(!mol.atoms[0].ghost);
    }

    #[test]
    fn link_atom_lies_on_the_bond_vector_at_the_requested_fraction() {
        // QM = atom 0 at origin; MM = atom 1 at z=2. Cut the 0-1 bond.
        let atoms = three_atoms();
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0]), 0, 1)
            .unwrap()
            .with_link_atoms(&[(0, 1)], 0.5)
            .unwrap();

        assert_eq!(sys.link_atoms.len(), 1);
        let link = &sys.link_atoms[0];
        assert_eq!(link.qm_atom, 0);
        assert_eq!(link.mm_atom_full_index, 1);
        // Bond vector is +z of length 2; g=0.5 puts H at z=1.0 exactly.
        assert!((link.position[0]).abs() < 1e-14);
        assert!((link.position[1]).abs() < 1e-14);
        assert!((link.position[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn link_atom_default_scale_gives_the_expected_bond_length() {
        let atoms = three_atoms();
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0]), 0, 1)
            .unwrap()
            .with_link_atoms(&[(0, 1)], DEFAULT_LINK_SCALE)
            .unwrap();
        let link = &sys.link_atoms[0];
        // |R_link - R_qm| must be exactly scale * |bond|.
        let d = (link.position[0].powi(2)
            + link.position[1].powi(2)
            + link.position[2].powi(2))
        .sqrt();
        assert!((d - DEFAULT_LINK_SCALE * 2.0).abs() < 1e-14, "got {d}");
    }

    #[test]
    fn link_atom_orientation_is_correct_for_an_off_axis_bond() {
        // Guard against a sign flip / wrong-endpoint bug that an on-axis test
        // would not catch: place the MM partner off all three axes.
        let atoms = vec![
            QmmmAtom::new("C", 6, 1.0, 2.0, 3.0, 0.0),
            QmmmAtom::new("C", 6, 4.0, 6.0, 3.0, 0.2), // 5 Bohr away in xy
        ];
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0]), 0, 1)
            .unwrap()
            .with_link_atoms(&[(0, 1)], 0.4)
            .unwrap();
        let p = sys.link_atoms[0].position;
        // R_qm + 0.4*(R_mm - R_qm) = (1,2,3) + 0.4*(3,4,0) = (2.2, 3.6, 3.0)
        assert!((p[0] - 2.2).abs() < 1e-14, "x = {}", p[0]);
        assert!((p[1] - 3.6).abs() < 1e-14, "y = {}", p[1]);
        assert!((p[2] - 3.0).abs() < 1e-14, "z = {}", p[2]);
        // And it must be *between* the two hosts, not beyond either.
        let d_from_qm =
            ((p[0] - 1.0).powi(2) + (p[1] - 2.0).powi(2) + (p[2] - 3.0).powi(2)).sqrt();
        assert!((d_from_qm - 0.4 * 5.0).abs() < 1e-14);
    }

    #[test]
    fn link_atom_appended_after_qm_atoms_in_molecule() {
        let atoms = three_atoms();
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0]), 0, 1)
            .unwrap()
            .with_link_atoms(&[(0, 1)], 0.5)
            .unwrap();
        let mol = sys.to_qm_molecule();
        assert_eq!(mol.atoms.len(), 2); // 1 real QM + 1 link H
        assert_eq!(sys.qm_atom_count(), 1);
        assert_eq!(mol.atoms[1].symbol, "H");
        assert_eq!(mol.atoms[1].z, 1);
        assert_eq!(mol.atoms[1].zpos, 1.0);
    }

    #[test]
    fn bonds_inside_one_region_are_not_cut() {
        let atoms = three_atoms();
        // Both endpoints QM => no link. Both endpoints MM => no link.
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1]), 0, 1)
            .unwrap()
            .with_link_atoms(&[(0, 1)], 0.5)
            .unwrap();
        assert!(sys.link_atoms.is_empty());

        let sys2 = QmmmSystem::new(&atoms, QmSelection::Indices(vec![2]), 0, 1)
            .unwrap()
            .with_link_atoms(&[(0, 1)], 0.5)
            .unwrap();
        assert!(sys2.link_atoms.is_empty());
    }

    #[test]
    fn min_link_to_charge_distance_reports_the_close_contact() {
        let atoms = three_atoms();
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0]), 0, 1)
            .unwrap()
            .with_link_atoms(&[(0, 1)], 0.5)
            .unwrap();
        // Link H at z=1.0; MM charges at z=2.0 (q=0.1) and z=10.0 (q=-0.5).
        let d = sys.min_link_to_charge_distance().unwrap();
        assert!((d - 1.0).abs() < 1e-14, "got {d}");
    }

    #[test]
    fn min_link_to_charge_distance_none_without_links_or_charges() {
        let atoms = three_atoms();
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1]), 0, 1).unwrap();
        assert!(sys.min_link_to_charge_distance().is_none());
    }

    #[test]
    fn whole_residue_selection_with_one_atom_per_residue_equals_by_atom() {
        // Exactness anchor: residue_ids = 0..n makes the new variant identical to WithinRadius.
        let atoms = three_atoms();
        let by_atom = QmmmSystem::new(&atoms, QmSelection::WithinRadius { seeds: vec![0], radius: 3.0 }, 0, 1).unwrap();
        let whole = QmmmSystem::new(&atoms, QmSelection::WithinRadiusWholeResidues { seeds: vec![0], radius: 3.0, residue_ids: vec![0, 1, 2] }, 0, 1).unwrap();
        assert_eq!(by_atom.qm_indices, whole.qm_indices);
        assert_eq!(by_atom.mm_indices, whole.mm_indices);
        assert_eq!(whole.residue_ids, Some(vec![0, 1, 2]));
        assert_eq!(by_atom.residue_ids, None);
    }

    #[test]
    fn whole_residue_selection_pulls_the_entire_residue_in() {
        // atoms 0,1 = residue 0 (2 Bohr apart); atom 2 = residue 1 at z=10, atom 3 = residue 1 at z=2.5.
        let mut atoms = three_atoms();
        atoms.push(QmmmAtom::new("N", 7, 0.0, 0.0, 2.5, 0.2));
        let sys = QmmmSystem::new(&atoms, QmSelection::WithinRadiusWholeResidues { seeds: vec![0], radius: 3.0, residue_ids: vec![0, 0, 1, 1] }, 0, 1).unwrap();
        // atom 3 is within 3 Bohr of the seed, so ALL of residue 1 (atoms 2 and 3) joins.
        assert_eq!(sys.qm_indices, vec![0, 1, 2, 3]);
        assert!(sys.mm_indices.is_empty());
    }

    #[test]
    fn whole_residue_selection_rejects_bad_residue_ids() {
        let atoms = three_atoms();
        assert!(QmmmSystem::new(&atoms, QmSelection::WithinRadiusWholeResidues { seeds: vec![0], radius: 1.0, residue_ids: vec![0, 0] }, 0, 1).is_err()); // length mismatch
        assert!(QmmmSystem::new(&atoms, QmSelection::WithinRadiusWholeResidues { seeds: vec![], radius: 1.0, residue_ids: vec![0, 0, 0] }, 0, 1).is_err());
        assert!(QmmmSystem::new(&atoms, QmSelection::WithinRadiusWholeResidues { seeds: vec![0], radius: f64::NAN, residue_ids: vec![0, 0, 0] }, 0, 1).is_err());
    }

    #[test]
    fn invalid_inputs_error_rather_than_panic() {
        let atoms = three_atoms();
        // Out-of-range QM index.
        assert!(QmmmSystem::new(&atoms, QmSelection::Indices(vec![7]), 0, 1).is_err());
        // Empty QM region.
        assert!(QmmmSystem::new(&atoms, QmSelection::Indices(vec![]), 0, 1).is_err());
        // Empty structure.
        assert!(QmmmSystem::new(&[], QmSelection::Indices(vec![0]), 0, 1).is_err());
        // Zero multiplicity.
        assert!(QmmmSystem::new(&atoms, QmSelection::Indices(vec![0]), 0, 0).is_err());
        // Bad seed index / negative radius.
        assert!(QmmmSystem::new(
            &atoms,
            QmSelection::WithinRadius { seeds: vec![9], radius: 1.0 },
            0,
            1
        )
        .is_err());
        assert!(QmmmSystem::new(
            &atoms,
            QmSelection::WithinRadius { seeds: vec![0], radius: -1.0 },
            0,
            1
        )
        .is_err());
        assert!(QmmmSystem::new(
            &atoms,
            QmSelection::WithinRadius { seeds: vec![], radius: 1.0 },
            0,
            1
        )
        .is_err());
        // NaN/inf radius must be rejected, not silently treated as "match all"
        // (a NaN comparison is always false, so an unguarded `d2 <= r2` would
        // quietly select NOTHING and produce an empty-QM error far from the
        // real cause).
        for bad in [f64::NAN, f64::INFINITY] {
            assert!(
                QmmmSystem::new(
                    &atoms,
                    QmSelection::WithinRadius { seeds: vec![0], radius: bad },
                    0,
                    1
                )
                .is_err(),
                "radius {bad} must be rejected"
            );
        }
        // Link scale outside (0,1), and an out-of-range bond.
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0]), 0, 1).unwrap();
        assert!(sys.clone().with_link_atoms(&[(0, 1)], 0.0).is_err());
        assert!(sys.clone().with_link_atoms(&[(0, 1)], 1.0).is_err());
        assert!(sys.clone().with_link_atoms(&[(0, 9)], 0.5).is_err());
    }

    // ── F2-1: QmmmSystem::with_coordinates ──

    const ANG2BOHR_TEST: f64 = 1.0 / 0.529_177_210_92;
    const ETHANE_CC_TEST: f64 = 1.53 * ANG2BOHR_TEST;

    /// Mirrors `crates/ferric-scf/tests/qmmm.rs::ethane_atoms` /
    /// `tests/qmmm_mm.rs::ethane_atoms` exactly.
    fn ethane_atoms_test() -> Vec<QmmmAtom> {
        let cc = ETHANE_CC_TEST;
        let ch = 1.09 * ANG2BOHR_TEST;
        let theta = 109.5_f64.to_radians();
        let (s, c) = (theta.sin(), theta.cos());
        let mut atoms = vec![
            QmmmAtom::new("C", 6, 0.0, 0.0, 0.0, -0.1),
            QmmmAtom::new("C", 6, 0.0, 0.0, cc, -0.1),
        ];
        for k in 0..3 {
            let phi = 2.0 * std::f64::consts::PI * (k as f64) / 3.0;
            atoms.push(QmmmAtom::new("H", 1, ch * s * phi.cos(), ch * s * phi.sin(), ch * c, 0.033));
            atoms.push(QmmmAtom::new("H", 1, ch * s * phi.cos(), ch * s * phi.sin(), cc - ch * c, 0.033));
        }
        atoms
    }

    fn ethane_bonds_test() -> Vec<(usize, usize)> {
        vec![(0, 1), (0, 2), (0, 4), (0, 6), (1, 3), (1, 5), (1, 7)]
    }

    fn atoms_to_coords(atoms: &[QmmmAtom]) -> Array2<f64> {
        let mut c = Array2::<f64>::zeros((atoms.len(), 3));
        for (i, a) in atoms.iter().enumerate() {
            c[(i, 0)] = a.x;
            c[(i, 1)] = a.y;
            c[(i, 2)] = a.z_pos;
        }
        c
    }

    fn capped_rcd_ethane_test() -> QmmmSystem {
        let bonds = ethane_bonds_test();
        QmmmSystem::new(&ethane_atoms_test(), QmSelection::Indices(vec![0, 2, 4, 6]), 0, 1)
            .unwrap()
            .with_link_atoms(&bonds, DEFAULT_LINK_SCALE)
            .unwrap()
            .with_boundary_charges(&bonds, BoundaryChargeScheme::RedistributedChargeDipole)
            .unwrap()
    }

    /// EXACTNESS ANCHOR: `with_coordinates` given the SAME coordinates the
    /// system was already built from must reproduce it bit-for-bit — same
    /// external potential (charges + positions), same link positions, same
    /// boundary charges.
    #[test]
    fn with_coordinates_at_the_same_geometry_is_bit_identical() {
        let sys = capped_rcd_ethane_test();
        let coords = atoms_to_coords(&ethane_atoms_test());
        let sys2 = sys.with_coordinates(&coords).unwrap();

        assert_eq!(sys.qm_indices, sys2.qm_indices);
        assert_eq!(sys.mm_indices, sys2.mm_indices);
        assert_eq!(sys.to_external_potential(), sys2.to_external_potential());
        assert_eq!(sys.link_atoms.len(), sys2.link_atoms.len());
        for (a, b) in sys.link_atoms.iter().zip(sys2.link_atoms.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.qm_atom, b.qm_atom);
            assert_eq!(a.mm_atom_full_index, b.mm_atom_full_index);
        }
        assert_eq!(sys.boundary_charges, sys2.boundary_charges);
        assert_eq!(sys.boundary_scheme, sys2.boundary_scheme);
    }

    /// Moving the QM frontier atom (C0, full index 0) must move the link
    /// atom along the (recomputed) bond vector by the SAME scale factor `g`
    /// used to build the system — i.e. `with_coordinates` genuinely re-runs
    /// link placement rather than just copying stale positions forward.
    #[test]
    fn with_coordinates_moves_the_link_atom_with_its_qm_host() {
        let sys = capped_rcd_ethane_test();
        let mut atoms = ethane_atoms_test();
        // Displace C0 (full index 0) by a Bohr along x.
        atoms[0].x += 1.0;
        let coords = atoms_to_coords(&atoms);
        let sys2 = sys.with_coordinates(&coords).unwrap();

        assert_eq!(sys2.link_atoms.len(), 1);
        let link = &sys2.link_atoms[0];
        let g = link.scale;
        let qm = &atoms[0];
        let mm = &atoms[1]; // C1, the MM host of the cut C0-C1 bond
        let expect = [
            qm.x + g * (mm.x - qm.x),
            qm.y + g * (mm.y - qm.y),
            qm.z_pos + g * (mm.z_pos - qm.z_pos),
        ];
        for k in 0..3 {
            assert!(
                (link.position[k] - expect[k]).abs() < 1e-12,
                "axis {k}: got {}, expected {}",
                link.position[k],
                expect[k]
            );
        }
        // And it must actually have moved from the original position.
        let orig = &capped_rcd_ethane_test().link_atoms[0];
        assert!((link.position[0] - orig.position[0]).abs() > 1e-3);
    }

    /// `with_coordinates` on a system with no link atoms / no boundary
    /// scheme (the common case) is a plain coordinate swap: same partition,
    /// new atom positions, no link/boundary machinery re-run.
    #[test]
    fn with_coordinates_without_link_atoms_or_boundary_scheme_just_moves_atoms() {
        let atoms = three_atoms();
        let sys = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1]), 0, 1).unwrap();
        let mut moved = atoms.clone();
        moved[2].z_pos += 5.0;
        let coords = atoms_to_coords(&moved);
        let sys2 = sys.with_coordinates(&coords).unwrap();
        assert_eq!(sys2.qm_indices, sys.qm_indices);
        assert_eq!(sys2.mm_indices, sys.mm_indices);
        assert_eq!(sys2.atoms[2].z_pos, moved[2].z_pos);
        assert!(sys2.link_atoms.is_empty());
    }

    #[test]
    fn with_coordinates_rejects_wrong_shape() {
        let sys = capped_rcd_ethane_test();
        let bad = Array2::<f64>::zeros((3, 3));
        assert!(sys.with_coordinates(&bad).is_err());
    }
}
