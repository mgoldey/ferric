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
//! - **Charge redistribution / deletion schemes are NOT implemented.** The
//!   standard fixes for the "link atom sits on top of an MM point charge"
//!   problem — deleting the host MM charge, redistributing it over the
//!   neighbouring MM sites (RCD/RC), smearing it onto a Gaussian, or using a
//!   double link atom — are all absent. [`QmmmSystem`] does automatically
//!   **exclude MM charges on atoms that are themselves in the QM region** (they
//!   would be double-counted), and [`LinkAtomSpec`] records which MM atom was
//!   the bond partner so a caller can implement redistribution itself, but this
//!   module applies none. If your link atom lands within a Bohr or so of a
//!   nonzero MM charge, the result is unphysical and this module will not warn
//!   you beyond [`QmmmSystem::min_link_to_charge_distance`].
//! - **No boundary-atom force projection.** The chain-rule force that a link
//!   atom's gradient should project back onto its two real host atoms is not
//!   applied. [`mm_forces`] returns forces on the *MM point charges* only.
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
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;

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
    /// the caller's job (build the index list and use [`QmSelection::Indices`]).
    WithinRadius { seeds: Vec<usize>, radius: f64 },
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
                if seeds.is_empty() {
                    return Err(FerricError::General(
                        "QmmmSystem::new: WithinRadius selection needs at least one seed atom"
                            .to_string(),
                    ));
                }
                // `is_finite` first, then a plain `<` — this rejects NaN via
                // the finiteness check rather than via a negated comparison
                // (`!(radius >= 0.0)` would also catch NaN, but reads as a
                // typo; see clippy::neg_cmp_op_on_partial_ord).
                if !radius.is_finite() || radius < 0.0 {
                    return Err(FerricError::General(format!(
                        "QmmmSystem::new: radius must be finite and >= 0, got {radius}"
                    )));
                }
                for &s in &seeds {
                    if s >= natoms {
                        return Err(FerricError::General(format!(
                            "QmmmSystem::new: seed index {s} out of range (structure has \
                             {natoms} atoms)"
                        )));
                    }
                    is_qm[s] = true;
                }
                let r2 = radius * radius;
                for i in 0..natoms {
                    if is_qm[i] {
                        continue;
                    }
                    for &s in &seeds {
                        if atoms[i].distance2(&atoms[s]) <= r2 {
                            is_qm[i] = true;
                            break;
                        }
                    }
                }
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
            atoms: atoms.to_vec(),
            qm_charge,
            qm_multiplicity,
        })
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
        Ok(self)
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
            .mm_indices
            .iter()
            .filter(|&&i| self.atoms[i].charge != 0.0)
            .map(|&i| {
                let a = &self.atoms[i];
                PointCharge { q: a.charge, x: a.x, y: a.y, z: a.z_pos }
            })
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
    /// [`mm_forces`] returns. Zero-charge MM atoms are excluded, matching
    /// `to_external_potential`.
    pub fn mm_charge_positions(&self) -> Vec<[f64; 3]> {
        self.mm_indices
            .iter()
            .filter(|&&i| self.atoms[i].charge != 0.0)
            .map(|&i| {
                let a = &self.atoms[i];
                [a.x, a.y, a.z_pos]
            })
            .collect()
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
        .mm_indices
        .iter()
        .filter(|&&i| system.atoms[i].charge != 0.0)
        .map(|&i| system.atoms[i].charge)
        .collect();

    let field = electric_field_at_points(mol, prep, density, &positions)?;
    Ok(field
        .iter()
        .zip(charges.iter())
        .map(|(e, &q)| [q * e[0], q * e[1], q * e[2]])
        .collect())
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
}
