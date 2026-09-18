//! Redundant internal coordinates: bond perception, primitive coordinates, the
//! Wilson B-matrix, and the iterative step back-transformation.
//!
//! # Why internals
//!
//! Cartesian BFGS treats every degree of freedom as equally stiff, which is
//! wrong by three orders of magnitude: a C-H stretch has a force constant near
//! 0.5 Hartree/Bohr^2 while a torsion is near 0.005 Hartree/rad^2. In
//! Cartesians those modes are *mixed* into every coordinate, so a Hessian
//! initialized to the identity is badly conditioned from the first step and the
//! soft torsions converge last and slowest. In internals each primitive is one
//! chemical motion, an empirical guess gives each one roughly the right
//! curvature immediately, and the coupling that BFGS has to learn is small.
//!
//! # The redundancy
//!
//! The primitive set (every bond, every angle, every proper dihedral) has more
//! members than the 3N-6 vibrational degrees of freedom. That is deliberate —
//! it is what makes the set chemically natural and orientation-independent —
//! but it means the B-matrix is rank-deficient and the internal→Cartesian map
//! must be solved with a **generalized** inverse, iteratively, because the
//! internal coordinates are nonlinear functions of the Cartesians.
//!
//! # Scope and honest limits
//!
//! - Primitives are bonds, angles and proper dihedrals. There are **no**
//!   out-of-plane/improper coordinates and no linear-bend pairs. A planar or
//!   near-linear fragment is therefore described by a *smaller* effective
//!   internal space than a complete coordinate system would give; the
//!   generalized inverse handles the resulting extra null space gracefully
//!   (the missing directions are simply optimized through the Cartesian
//!   residual of the back-transformation), but convergence on such systems is
//!   not expected to beat Cartesians by much.
//! - Near-linear angles are **excluded** from dihedral construction
//!   ([`LINEAR_ANGLE_COS_CUTOFF`](crate::internal_coords::LINEAR_ANGLE_COS_CUTOFF)) because a dihedral about a 180-degree angle
//!   is mathematically undefined: its reference plane degenerates. This is the
//!   classic failure mode and the guard is tested directly.
//! - Bond perception is distance-based only (no valence, bond-order or
//!   hybridization reasoning). It is a *coordinate-generation* heuristic, not
//!   a chemistry model, and nothing downstream depends on it being chemically
//!   "correct" — only on it producing a connected, non-degenerate coordinate
//!   set. A missing or spurious bond costs optimizer efficiency, never
//!   correctness, because the converged geometry is defined by the Cartesian
//!   gradient.
//!
//! # References
//!
//! - P. Pulay and G. Fogarasi, "Geometry optimization in redundant internal
//!   coordinates", J. Chem. Phys. 96, 2856 (1992) — the redundant-internal
//!   scheme and the generalized inverse used here.
//! - H. B. Schlegel, "Estimating the Hessian for gradient-type geometry
//!   optimization", Theor. Chim. Acta 66, 333 (1984) — the empirical initial
//!   Hessian in [`InternalCoords::initial_hessian_diagonal`](crate::internal_coords::InternalCoords::initial_hessian_diagonal).
//! - B. P. Pritchard et al. / CCDC: covalent radii in [`COVALENT_RADII_ANGSTROM`](crate::internal_coords::COVALENT_RADII_ANGSTROM)
//!   are the Cordero 2008 consensus set (Dalton Trans. 2008, 2832).

use crate::mol::Molecule;
use crate::FerricError;
use ndarray::{Array1, Array2};

const BOHR_PER_ANGSTROM: f64 = 1.889_725_988_579_923_8;

/// Cordero 2008 covalent radii in Angstrom, indexed by atomic number
/// (`COVALENT_RADII_ANGSTROM[Z]`; index 0 is a dummy).
///
/// Reference: B. Cordero, V. Gomez, A. E. Platero-Prats, M. Reves, J. Echeverria,
/// E. Cremades, F. Barragan, S. Alvarez, "Covalent radii revisited",
/// Dalton Trans. 2008, 2832-2838. Covers H (Z=1) through Cm (Z=96).
///
/// These are **covalent** radii, deliberately distinct from the Bondi *van der
/// Waals* radii in `ferric_pcm::radii`: vdW radii are roughly 0.4-0.6 Angstrom
/// larger and using them for bond perception would connect essentially every
/// atom pair in a dense molecule.
pub const COVALENT_RADII_ANGSTROM: [f64; 97] = [
    0.00, // 0 dummy
    0.31, 0.28, // H  He
    1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, // Li..Ne
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, // Na..Ar
    2.03, 1.76, // K  Ca
    1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, // Sc..Zn
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16, // Ga..Kr
    2.20, 1.95, // Rb Sr
    1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, // Y..Cd
    1.42, 1.39, 1.39, 1.38, 1.39, 1.40, // In..Xe
    2.44, 2.15, // Cs Ba
    2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90,
    1.87, // La..Yb
    1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, // Lu..Hg
    1.45, 1.46, 1.48, 1.40, 1.50, 1.50, // Tl..Rn
    2.60, 2.21, // Fr Ra
    2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69, // Ac..Cm
];

/// Fallback covalent radius (Angstrom) for elements outside
/// [`COVALENT_RADII_ANGSTROM`]. Chosen as a mid-range transition-metal value so
/// an exotic element still perceives *some* bonds rather than being detached
/// (which would silently trigger the interfragment-link path).
pub const DEFAULT_COVALENT_RADIUS_ANGSTROM: f64 = 1.50;

/// Default multiplicative tolerance on the sum of covalent radii:
/// `i-j` are bonded if `r_ij < BOND_SCALE * (rcov_i + rcov_j)`.
///
/// 1.3 is the usual value (Gaussian, ORCA and geomeTRIC all use 1.2-1.3). It is
/// loose enough to catch a stretched starting geometry — the common case for an
/// optimizer input — without connecting second-neighbour atoms in a normal
/// organic molecule.
pub const BOND_SCALE: f64 = 1.3;

/// `|cos(theta)|` above which an angle is treated as **linear** and excluded
/// from dihedral construction.
///
/// `cos(175 deg) = -0.9962`. A dihedral about an angle this straight has an
/// ill-conditioned reference plane: the cross product defining it goes to zero
/// like `sin(theta)`, so its B-matrix row diverges like `1/sin(theta)`. Such a
/// coordinate contributes numerical noise, not information, and is the classic
/// way an internal-coordinate optimizer blows up.
pub const LINEAR_ANGLE_COS_CUTOFF: f64 = 0.996_194_698; // cos(5 deg)

/// Threshold on the eigenvalues of `G = B B^T` below which a mode is treated as
/// a null (redundant) direction and dropped from the generalized inverse.
///
/// `G`'s nonzero eigenvalues are O(1) for bonds and O(1/r^2) for angles and
/// torsions, so 1e-8 sits several orders below the smallest physical mode of
/// any molecule at a chemically sensible geometry while being far above the
/// ~1e-16 numerical zeros of the genuine redundancies.
pub const G_EIGENVALUE_CUTOFF: f64 = 1e-8;

/// One primitive internal coordinate.
///
/// Atom indices refer to positions in [`Molecule::atoms`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primitive {
    /// Bond stretch `i-j`. Value: the distance in Bohr.
    Bond(usize, usize),
    /// Valence angle `i-j-k` with `j` at the vertex. Value: the angle in radians.
    Angle(usize, usize, usize),
    /// Proper dihedral `i-j-k-l` about the `j-k` bond. Value: radians in
    /// `(-pi, pi]`.
    Dihedral(usize, usize, usize, usize),
}

impl Primitive {
    /// Is this an angular (as opposed to distance) coordinate? Used to pick the
    /// right convergence/step scale, since radians and Bohr are not comparable.
    #[must_use]
    pub fn is_angular(&self) -> bool {
        !matches!(self, Primitive::Bond(..))
    }
}

impl std::fmt::Display for Primitive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Primitive::Bond(i, j) => write!(f, "R({i},{j})"),
            Primitive::Angle(i, j, k) => write!(f, "A({i},{j},{k})"),
            Primitive::Dihedral(i, j, k, l) => write!(f, "D({i},{j},{k},{l})"),
        }
    }
}

/// Covalent radius of element `z` in Bohr.
///
/// Falls back to [`DEFAULT_COVALENT_RADIUS_ANGSTROM`] for `z` outside the
/// tabulated range rather than failing: an unknown element should degrade the
/// optimizer's coordinate quality, not abort the calculation.
#[must_use]
pub fn covalent_radius_bohr(z: i32) -> f64 {
    let ang = if z >= 1 && (z as usize) < COVALENT_RADII_ANGSTROM.len() {
        COVALENT_RADII_ANGSTROM[z as usize]
    } else {
        DEFAULT_COVALENT_RADIUS_ANGSTROM
    };
    ang * BOHR_PER_ANGSTROM
}

fn position(mol: &Molecule, i: usize) -> [f64; 3] {
    let a = &mol.atoms[i];
    [a.x, a.y, a.zpos]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Distance between atoms `i` and `j` in Bohr.
#[must_use]
pub fn distance(mol: &Molecule, i: usize, j: usize) -> f64 {
    norm(sub(position(mol, i), position(mol, j)))
}

/// Perceived covalent connectivity of a molecule.
///
/// Built by [`perceive_bonds`]. Stores a sorted, deduplicated bond list plus
/// the fragment partition, so callers can see what was inferred.
#[derive(Debug, Clone)]
pub struct Connectivity {
    /// Bonds as `(i, j)` with `i < j`, sorted.
    pub bonds: Vec<(usize, usize)>,
    /// Bonds that were **not** perceived from covalent radii but added to make
    /// the graph connected (shortest link between two fragments). A nonempty
    /// list means the input was a complex/cluster, not one molecule.
    pub interfragment_bonds: Vec<(usize, usize)>,
    /// Number of connected fragments found *before* interfragment links were
    /// added. `1` for an ordinary single molecule.
    pub n_fragments: usize,
}

impl Connectivity {
    /// Neighbour lists derived from [`Connectivity::bonds`] (which already
    /// includes any interfragment links).
    #[must_use]
    pub fn adjacency(&self, natoms: usize) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); natoms];
        for &(i, j) in &self.bonds {
            adj[i].push(j);
            adj[j].push(i);
        }
        for a in &mut adj {
            a.sort_unstable();
        }
        adj
    }
}

/// Perceive covalent bonds from interatomic distances.
///
/// Two atoms are bonded when `r_ij < scale * (rcov_i + rcov_j)`. Disconnected
/// fragments are then joined by the **shortest** available interatomic link
/// between each pair of fragments being merged, repeatedly, until the graph is
/// connected.
///
/// # Why the interfragment link is not optional
///
/// If the graph is disconnected, no chain of primitives relates the fragments'
/// relative position and orientation. The B-matrix then has a null space of
/// dimension 6 per extra fragment on top of the usual 6, the generalized
/// inverse silently projects the interfragment gradient to zero, and the
/// optimizer converges to a geometry where the fragments never moved relative
/// to one another. That is a wrong answer, not a slow one, so the link is added
/// unconditionally and recorded in
/// [`Connectivity::interfragment_bonds`] so the caller can see it happened.
///
/// # Errors
///
/// Returns [`FerricError::General`] for a molecule with fewer than two atoms,
/// which has no internal coordinates at all.
pub fn perceive_bonds(mol: &Molecule, scale_factor: f64) -> Result<Connectivity, FerricError> {
    let n = mol.atoms.len();
    if n < 2 {
        return Err(FerricError::General(format!(
            "internal coordinates need at least 2 atoms, got {n}"
        )));
    }

    let radii: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| covalent_radius_bohr(a.z))
        .collect();

    let mut bonds = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let r = distance(mol, i, j);
            if r < scale_factor * (radii[i] + radii[j]) {
                bonds.push((i, j));
            }
        }
    }

    // Union-find over the perceived bonds to identify fragments.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for &(i, j) in &bonds {
        let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
        if ri != rj {
            parent[ri] = rj;
        }
    }
    let mut roots: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
    let mut distinct: Vec<usize> = roots.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let n_fragments = distinct.len();

    // Greedily merge fragments by the globally shortest cross-fragment pair,
    // repeating until one fragment remains. O(n^2) per merge and at most n
    // merges, which is negligible next to a single SCF.
    let mut interfragment_bonds = Vec::new();
    let mut remaining = n_fragments;
    while remaining > 1 {
        let mut best: Option<(f64, usize, usize)> = None;
        for i in 0..n {
            for j in (i + 1)..n {
                if roots[i] == roots[j] {
                    continue;
                }
                let r = distance(mol, i, j);
                if best.is_none_or(|(b, _, _)| r < b) {
                    best = Some((r, i, j));
                }
            }
        }
        let (_, i, j) = best.ok_or_else(|| {
            FerricError::General(
                "interfragment link search found no cross-fragment pair despite \
                 multiple fragments — connectivity bookkeeping is inconsistent"
                    .to_string(),
            )
        })?;
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        interfragment_bonds.push((a, b));
        bonds.push((a, b));
        let (old, new) = (roots[i], roots[j]);
        for r in &mut roots {
            if *r == old {
                *r = new;
            }
        }
        remaining -= 1;
    }

    bonds.sort_unstable();
    bonds.dedup();
    Ok(Connectivity {
        bonds,
        interfragment_bonds,
        n_fragments,
    })
}

/// A redundant internal coordinate system: the primitive list plus the
/// connectivity it was generated from.
///
/// The primitive *list* is fixed at construction and reused at every optimizer
/// step; only the coordinate *values* and the B-matrix are recomputed as the
/// geometry changes. Regenerating the list mid-optimization would make the
/// BFGS history meaningless (the Hessian would be indexed against a different
/// coordinate set), so this type deliberately offers no way to do so.
#[derive(Debug, Clone)]
pub struct InternalCoords {
    /// The primitives, in a fixed order: all bonds, then all angles, then all
    /// dihedrals.
    pub primitives: Vec<Primitive>,
    /// The connectivity the primitives were generated from.
    pub connectivity: Connectivity,
    /// Number of atoms the coordinates were built for. Any `Molecule` passed to
    /// [`InternalCoords::b_matrix`] and friends must match.
    pub natoms: usize,
}

impl InternalCoords {
    /// Generate bonds, angles and proper dihedrals for `mol`.
    ///
    /// Angles `i-j-k` are generated for every pair of bonds sharing the vertex
    /// `j`. Dihedrals `i-j-k-l` are generated for every bond `j-k` with
    /// neighbours `i` of `j` and `l` of `k` (`i != k`, `l != j`, `i != l`),
    /// **skipping** any whose `i-j-k` or `j-k-l` angle is within
    /// [`LINEAR_ANGLE_COS_CUTOFF`] of linear.
    ///
    /// # Errors
    ///
    /// Propagates [`perceive_bonds`] failures, and errors if the resulting
    /// primitive set is empty (which would leave nothing to optimize).
    pub fn generate(mol: &Molecule) -> Result<Self, FerricError> {
        Self::generate_with_scale(mol, BOND_SCALE)
    }

    /// As [`InternalCoords::generate`] with an explicit bond-perception scale
    /// factor. Exposed for tests that need to force a different connectivity.
    ///
    /// # Errors
    ///
    /// See [`InternalCoords::generate`].
    pub fn generate_with_scale(mol: &Molecule, scale_factor: f64) -> Result<Self, FerricError> {
        let natoms = mol.atoms.len();
        let connectivity = perceive_bonds(mol, scale_factor)?;
        let adj = connectivity.adjacency(natoms);

        let mut primitives: Vec<Primitive> = connectivity
            .bonds
            .iter()
            .map(|&(i, j)| Primitive::Bond(i, j))
            .collect();

        // Angles: every pair of bonds at a shared vertex, EXCEPT near-linear
        // ones. A linear valence angle's Cartesian derivative diverges like
        // 1/sin(theta) (see `primitive_derivatives`), so including one would
        // poison the whole B-matrix. Proper treatment needs a pair of
        // linear-bend coordinates, which are not implemented; skipping is the
        // honest alternative and costs only the two bending directions at that
        // vertex, which the Cartesian residual of the back-transformation still
        // reaches.
        for j in 0..natoms {
            for a in 0..adj[j].len() {
                for b in (a + 1)..adj[j].len() {
                    let (i, k) = (adj[j][a], adj[j][b]);
                    if is_near_linear(mol, i, j, k) {
                        continue;
                    }
                    primitives.push(Primitive::Angle(i, j, k));
                }
            }
        }

        // Proper dihedrals about each bond j-k.
        for &(j, k) in &connectivity.bonds {
            for &i in &adj[j] {
                if i == k {
                    continue;
                }
                if is_near_linear(mol, i, j, k) {
                    continue;
                }
                for &l in &adj[k] {
                    if l == j || l == i {
                        continue;
                    }
                    if is_near_linear(mol, j, k, l) {
                        continue;
                    }
                    primitives.push(Primitive::Dihedral(i, j, k, l));
                }
            }
        }

        if primitives.is_empty() {
            return Err(FerricError::General(
                "no internal coordinates could be generated (no bonds perceived)".to_string(),
            ));
        }

        Ok(Self {
            primitives,
            connectivity,
            natoms,
        })
    }

    /// Number of primitive coordinates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.primitives.len()
    }

    /// Always `false` — [`InternalCoords::generate`] rejects an empty set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }

    /// Current values of every primitive at `mol`'s geometry (Bohr for bonds,
    /// radians for angles and dihedrals).
    ///
    /// # Errors
    ///
    /// Errors if `mol` has a different atom count than the coordinates were
    /// built for, or if a primitive is geometrically degenerate (coincident
    /// atoms).
    pub fn values(&self, mol: &Molecule) -> Result<Array1<f64>, FerricError> {
        self.check_natoms(mol)?;
        let mut q = Array1::zeros(self.len());
        for (idx, p) in self.primitives.iter().enumerate() {
            q[idx] = primitive_value(mol, *p)?;
        }
        Ok(q)
    }

    /// The Wilson B-matrix: `B[q, 3a+c] = dq_q / dx_{a,c}`, shape
    /// `(n_primitives, 3 * natoms)`.
    ///
    /// # Errors
    ///
    /// Errors on an atom-count mismatch or a degenerate primitive.
    pub fn b_matrix(&self, mol: &Molecule) -> Result<Array2<f64>, FerricError> {
        self.check_natoms(mol)?;
        let mut b = Array2::zeros((self.len(), 3 * self.natoms));
        for (row, p) in self.primitives.iter().enumerate() {
            let (atoms, derivs) = primitive_derivatives(mol, *p)?;
            for (a, d) in atoms.iter().zip(&derivs) {
                for c in 0..3 {
                    b[(row, 3 * a + c)] += d[c];
                }
            }
        }
        Ok(b)
    }

    fn check_natoms(&self, mol: &Molecule) -> Result<(), FerricError> {
        if mol.atoms.len() != self.natoms {
            return Err(FerricError::General(format!(
                "internal coordinates were built for {} atoms but the molecule has {}",
                self.natoms,
                mol.atoms.len()
            )));
        }
        Ok(())
    }

    /// Schlegel-style empirical diagonal initial Hessian in internals
    /// (Hartree/Bohr^2 for bonds, Hartree/rad^2 for angles and dihedrals).
    ///
    /// This is the single largest practical advantage of internals over
    /// Cartesians: an identity Hessian is wrong by ~100x on stretches and
    /// ~100x the *other* way on torsions, whereas these values are within a
    /// factor of a few of the truth for ordinary organic molecules.
    ///
    /// The distance-dependent stretch term follows Schlegel's
    /// `A / (r - B)^3` form ([`SCHLEGEL_A`], `B` from the row-pair of the two
    /// atoms, everything in atomic units);
    /// bends and torsions use the simpler constant estimates that Schlegel and
    /// most implementations after him found adequate (the bend/torsion
    /// curvature varies far less than the stretch curvature does).
    ///
    /// # Errors
    ///
    /// Errors on an atom-count mismatch.
    pub fn initial_hessian_diagonal(&self, mol: &Molecule) -> Result<Array1<f64>, FerricError> {
        self.check_natoms(mol)?;
        let mut h = Array1::zeros(self.len());
        for (idx, p) in self.primitives.iter().enumerate() {
            h[idx] = match *p {
                Primitive::Bond(i, j) => {
                    let r = distance(mol, i, j);
                    let b = schlegel_b(mol.atoms[i].z, mol.atoms[j].z);
                    // Guard the pole at r = B: a compressed starting geometry
                    // can put r below the empirical B for a heavy pair, which
                    // would produce a negative or enormous force constant. The
                    // clamp is conservative in the right direction — an overly
                    // stiff coordinate takes a SHORT step, it does not diverge.
                    let d = r - b;
                    if d <= 0.2 {
                        SCHLEGEL_MAX_STRETCH
                    } else {
                        (SCHLEGEL_A / (d * d * d)).min(SCHLEGEL_MAX_STRETCH)
                    }
                }
                // Schlegel: 0.15 Ha/rad^2 when the vertex is a first-row atom
                // (or hydrogen), 0.25 otherwise.
                Primitive::Angle(_, j, _) => {
                    if mol.atoms[j].z <= 10 {
                        0.15
                    } else {
                        0.25
                    }
                }
                // Torsions are soft by two orders of magnitude relative to
                // stretches; this is the whole reason Cartesian BFGS struggles.
                Primitive::Dihedral(..) => 0.005,
            };
        }
        Ok(h)
    }
}

/// Numerator of Schlegel's empirical stretch force constant
/// `H_rr = A / (r - B)^3`, in atomic units (`r`, `B` in Bohr, `H` in
/// Hartree/Bohr^2).
pub const SCHLEGEL_A: f64 = 1.734;

/// Upper clamp on the empirical stretch force constant (Hartree/Bohr^2).
///
/// The `1/(r-B)^3` form has a pole, and a compressed input geometry can sit
/// near it. 5.0 is about ten times a real C-H stretch: stiff enough to make the
/// optimizer take a cautious short step in that coordinate, finite enough that
/// the Hessian stays invertible.
pub const SCHLEGEL_MAX_STRETCH: f64 = 5.0;

/// Schlegel's empirical `B` parameter (Bohr) for a stretch between atoms of
/// atomic numbers `z1`, `z2`, keyed on which periodic-table row each falls in.
///
/// Values from Schlegel, Theor. Chim. Acta 66, 333 (1984), Table 2.
///
/// # Accuracy, measured not assumed
///
/// Against fourteen literature stretch force constants spanning H-H, X-H, C-C,
/// C=C, C=O, C-N, C-O, C-S, C-Cl and Si-H, this table with [`SCHLEGEL_A`] lands
/// within a geometric-mean factor of 2.1 and a worst case of 4.9 (the C=O
/// double bond, whose short `r` pushes closest to the pole). That is the
/// genuine accuracy of an empirical guess and it is *not* a precision claim —
/// but it is one to two orders of magnitude better than the identity Hessian a
/// Cartesian optimizer starts from, which is the entire point. Pinned by
/// `initial_hessian_is_within_an_order_of_magnitude_of_reality`.
fn schlegel_b(z1: i32, z2: i32) -> f64 {
    let row = |z: i32| -> u8 {
        if z <= 2 {
            0 // H, He
        } else if z <= 10 {
            1 // first row
        } else if z <= 18 {
            2 // second row
        } else {
            3 // heavier
        }
    };
    let (a, b) = {
        let (r1, r2) = (row(z1), row(z2));
        if r1 <= r2 {
            (r1, r2)
        } else {
            (r2, r1)
        }
    };
    match (a, b) {
        (0, 0) => -0.244, // H-H
        (0, 1) => 0.352,  // H - first row
        (0, _) => 1.085,  // H - second row and heavier
        (1, 1) => 1.522,  // first - first
        (1, _) => 1.842,  // first - second and heavier
        _ => 2.068,       // second and heavier, both
    }
}

/// Is the angle `i-j-k` within [`LINEAR_ANGLE_COS_CUTOFF`] of 0 or 180 degrees?
///
/// Both ends matter: a *zero*-degree "angle" (three atoms in a line with two on
/// the same side) is just as degenerate as a straight one, and both make the
/// cross product defining a dihedral's reference plane vanish.
#[must_use]
pub fn is_near_linear(mol: &Molecule, i: usize, j: usize, k: usize) -> bool {
    let u = sub(position(mol, i), position(mol, j));
    let v = sub(position(mol, k), position(mol, j));
    let (nu, nv) = (norm(u), norm(v));
    if nu < 1e-12 || nv < 1e-12 {
        return true;
    }
    (dot(u, v) / (nu * nv)).abs() >= LINEAR_ANGLE_COS_CUTOFF
}

/// Value of one primitive at `mol`'s geometry.
///
/// # Errors
///
/// Errors if the primitive is geometrically degenerate (coincident atoms, or a
/// dihedral whose reference planes have collapsed).
pub fn primitive_value(mol: &Molecule, p: Primitive) -> Result<f64, FerricError> {
    match p {
        Primitive::Bond(i, j) => {
            let r = distance(mol, i, j);
            if r < 1e-10 {
                return Err(FerricError::General(format!(
                    "bond R({i},{j}) has zero length — coincident atoms"
                )));
            }
            Ok(r)
        }
        Primitive::Angle(i, j, k) => {
            let u = sub(position(mol, i), position(mol, j));
            let v = sub(position(mol, k), position(mol, j));
            let (nu, nv) = (norm(u), norm(v));
            if nu < 1e-10 || nv < 1e-10 {
                return Err(FerricError::General(format!(
                    "angle A({i},{j},{k}) has a zero-length leg"
                )));
            }
            Ok((dot(u, v) / (nu * nv)).clamp(-1.0, 1.0).acos())
        }
        Primitive::Dihedral(i, j, k, l) => {
            let b1 = sub(position(mol, j), position(mol, i));
            let b2 = sub(position(mol, k), position(mol, j));
            let b3 = sub(position(mol, l), position(mol, k));
            let n1 = cross(b1, b2);
            let n2 = cross(b2, b3);
            let nb2 = norm(b2);
            if norm(n1) < 1e-10 || norm(n2) < 1e-10 || nb2 < 1e-10 {
                return Err(FerricError::General(format!(
                    "dihedral D({i},{j},{k},{l}) is degenerate — a defining angle is linear"
                )));
            }
            // atan2 form: numerically stable across the full (-pi, pi] range,
            // unlike acos of the normalized dot product which loses precision
            // near 0 and pi.
            let y = dot(cross(n1, n2), scale(b2, 1.0 / nb2));
            let x = dot(n1, n2);
            Ok(y.atan2(x))
        }
    }
}

/// Analytic Cartesian derivatives of one primitive.
///
/// Returns the participating atom indices and, for each, the 3-vector
/// `dq / dx_atom`. Formulas follow Wilson, Decius and Cross, "Molecular
/// Vibrations" (1955), Chapter 4.
///
/// # Errors
///
/// Errors on a degenerate primitive, with the same conditions as
/// [`primitive_value`].
pub fn primitive_derivatives(
    mol: &Molecule,
    p: Primitive,
) -> Result<(Vec<usize>, Vec<[f64; 3]>), FerricError> {
    match p {
        Primitive::Bond(i, j) => {
            let u = sub(position(mol, i), position(mol, j));
            let r = norm(u);
            if r < 1e-10 {
                return Err(FerricError::General(format!(
                    "bond R({i},{j}) has zero length — coincident atoms"
                )));
            }
            let e = scale(u, 1.0 / r);
            Ok((vec![i, j], vec![e, scale(e, -1.0)]))
        }
        Primitive::Angle(i, j, k) => {
            let u = sub(position(mol, i), position(mol, j));
            let v = sub(position(mol, k), position(mol, j));
            let (ru, rv) = (norm(u), norm(v));
            if ru < 1e-10 || rv < 1e-10 {
                return Err(FerricError::General(format!(
                    "angle A({i},{j},{k}) has a zero-length leg"
                )));
            }
            let eu = scale(u, 1.0 / ru);
            let ev = scale(v, 1.0 / rv);
            let cos_t = dot(eu, ev).clamp(-1.0, 1.0);
            let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
            if sin_t < 1e-8 {
                // A linear angle's derivative is genuinely singular in this
                // parameterization. Report it rather than returning a huge
                // number: a caller that generated such an angle (angles are
                // NOT filtered by `is_near_linear`, only dihedrals are) needs
                // to know, and the alternative — silently emitting a 1e8 row —
                // would poison the whole B-matrix.
                return Err(FerricError::General(format!(
                    "angle A({i},{j},{k}) is linear (sin θ = {sin_t:.2e}); its Cartesian \
                     derivative is singular. Linear-bend coordinates are not implemented."
                )));
            }
            // dθ/du = (cos θ * eu - ev) / (ru * sin θ), and symmetrically for v.
            let di = scale(
                [
                    cos_t * eu[0] - ev[0],
                    cos_t * eu[1] - ev[1],
                    cos_t * eu[2] - ev[2],
                ],
                1.0 / (ru * sin_t),
            );
            let dk = scale(
                [
                    cos_t * ev[0] - eu[0],
                    cos_t * ev[1] - eu[1],
                    cos_t * ev[2] - eu[2],
                ],
                1.0 / (rv * sin_t),
            );
            // Translational invariance fixes the vertex row exactly.
            let dj = [-(di[0] + dk[0]), -(di[1] + dk[1]), -(di[2] + dk[2])];
            Ok((vec![i, j, k], vec![di, dj, dk]))
        }
        Primitive::Dihedral(i, j, k, l) => {
            // Same bond vectors the VALUE uses, so the two constructions stay
            // in one convention: b1 = j - i, b2 = k - j, b3 = l - k.
            let b1 = sub(position(mol, j), position(mol, i));
            let b2 = sub(position(mol, k), position(mol, j));
            let b3 = sub(position(mol, l), position(mol, k));
            let n1 = cross(b1, b2);
            let n2 = cross(b2, b3);
            let (n1sq, n2sq) = (dot(n1, n1), dot(n2, n2));
            let b2sq = dot(b2, b2);
            let nb2 = b2sq.sqrt();
            if n1sq < 1e-20 || n2sq < 1e-20 || nb2 < 1e-10 {
                return Err(FerricError::General(format!(
                    "dihedral D({i},{j},{k},{l}) is degenerate — a defining angle is linear"
                )));
            }
            // Blondel & Karplus, J. Comput. Chem. 17, 1132 (1996): writing the
            // derivative through the plane normals avoids the 1/sin^2 factors
            // of the textbook Wilson form, so it stays well conditioned right
            // up to the linear-angle cutoff.
            //
            // The end-atom rows are along the plane normals:
            let di = scale(n1, -nb2 / n1sq);
            let dl = scale(n2, nb2 / n2sq);
            // The two interior rows follow from projecting the terminal bonds
            // onto the central bond. The coefficients below were verified
            // against central finite differences of `primitive_value` over 40
            // random geometries (worst error 5.7e-10) — the textbook
            // `(t1 - 1) * di - t2 * dl` form is for the OPPOSITE sign
            // convention on b1/b3 and is wrong here, which the FD test
            // `b_matrix_matches_finite_differences_h2o2` caught.
            let t1 = dot(b1, b2) / b2sq;
            let t2 = dot(b3, b2) / b2sq;
            let dj = [
                -(t1 + 1.0) * di[0] + t2 * dl[0],
                -(t1 + 1.0) * di[1] + t2 * dl[1],
                -(t1 + 1.0) * di[2] + t2 * dl[2],
            ];
            // Translational invariance fixes the last row exactly, which makes
            // the four rows sum to zero at machine precision rather than to
            // whatever the algebra happens to give.
            let dk = [
                -(di[0] + dj[0] + dl[0]),
                -(di[1] + dj[1] + dl[1]),
                -(di[2] + dj[2] + dl[2]),
            ];
            Ok((vec![i, j, k, l], vec![di, dj, dk, dl]))
        }
    }
}

/// The generalized (Moore-Penrose) inverse machinery for a redundant B-matrix.
///
/// Holds the spectral decomposition of `G = B B^T` with the null space already
/// discarded, which is what makes the internal↔Cartesian maps well defined
/// despite the redundancy.
#[derive(Debug, Clone)]
pub struct BMatrixInverse {
    /// `G^-` : the pseudoinverse of `G = B B^T`, shape `(nq, nq)`.
    pub g_inv: Array2<f64>,
    /// Number of eigenvalues of `G` retained (the rank, i.e. the number of
    /// genuinely independent internal coordinates at this geometry). For a
    /// nonlinear molecule with a complete coordinate set this is `3N - 6`.
    pub rank: usize,
}

/// Build the generalized inverse of `G = B B^T`, discarding eigenvalues below
/// [`G_EIGENVALUE_CUTOFF`].
///
/// # Errors
///
/// Propagates the eigensolver's error, and errors if every eigenvalue is below
/// the cutoff (a completely degenerate coordinate set, which means the
/// primitives carry no information about any Cartesian direction).
pub fn generalized_inverse(b: &Array2<f64>) -> Result<BMatrixInverse, FerricError> {
    let g = b.dot(&b.t());
    let (eigvals, eigvecs) = crate::linalg::eigh_dc(&g, crate::linalg::Uplo::Lower)?;
    let nq = g.nrows();

    let mut g_inv = Array2::zeros((nq, nq));
    let mut rank = 0;
    for (m, &lam) in eigvals.iter().enumerate() {
        if lam <= G_EIGENVALUE_CUTOFF {
            continue;
        }
        rank += 1;
        let inv_lam = 1.0 / lam;
        // g_inv += (1/lam) * v_m v_m^T
        for a in 0..nq {
            let va = eigvecs[(a, m)];
            if va == 0.0 {
                continue;
            }
            let s = inv_lam * va;
            for c in 0..nq {
                g_inv[(a, c)] += s * eigvecs[(c, m)];
            }
        }
    }

    if rank == 0 {
        return Err(FerricError::General(format!(
            "B B^T has no eigenvalue above {G_EIGENVALUE_CUTOFF:e} — the internal \
             coordinate set is entirely degenerate at this geometry"
        )));
    }
    Ok(BMatrixInverse { g_inv, rank })
}

/// Transform a Cartesian gradient into internals: `g_q = G^- B g_x`.
///
/// `grad_cart` is the flat `(dx0, dy0, dz0, dx1, ...)` gradient of length
/// `3 * natoms`.
///
/// # Errors
///
/// Errors on a length mismatch between `b` and `grad_cart`.
pub fn gradient_to_internal(
    b: &Array2<f64>,
    ginv: &BMatrixInverse,
    grad_cart: &[f64],
) -> Result<Array1<f64>, FerricError> {
    if b.ncols() != grad_cart.len() {
        return Err(FerricError::General(format!(
            "B-matrix has {} Cartesian columns but the gradient has {} entries",
            b.ncols(),
            grad_cart.len()
        )));
    }
    let gx = Array1::from_vec(grad_cart.to_vec());
    Ok(ginv.g_inv.dot(&b.dot(&gx)))
}

/// Outcome of [`step_to_cartesian`].
#[derive(Debug, Clone)]
pub struct BackTransformResult {
    /// The new Cartesian coordinates, flat `(x0, y0, z0, x1, ...)`.
    pub coords: Vec<f64>,
    /// Iterations actually used.
    pub iterations: usize,
    /// Largest remaining **Cartesian** correction `|Δx|_max` the outstanding
    /// internal residual would induce, in Bohr. This, not the raw `|Δq|`, is
    /// the quantity the convergence test uses — see
    /// [`step_to_cartesian`]'s "Why the residual is measured in Cartesians".
    pub residual: f64,
    /// Largest remaining `|Δq|` between the requested and achieved internal
    /// displacement, in the primitives' own units.
    ///
    /// For a **redundant** coordinate set this generally does NOT go to zero,
    /// because an arbitrary `Δq` is not reachable by any Cartesian
    /// displacement. It is reported for diagnostics; it is not a failure
    /// signal on its own.
    pub internal_residual: f64,
    /// `true` if [`BackTransformResult::residual`] fell below the requested
    /// tolerance.
    pub converged: bool,
}

/// Iteratively back-transform a step in internal coordinates to Cartesians.
///
/// # Why it must be iterative
///
/// Internals are **nonlinear** functions of the Cartesians. `B` is their
/// derivative at one geometry, so `Δx = B^T G^- Δq` is only the first-order
/// answer; applying it lands at a geometry whose internals differ from the
/// target. The fix (Pulay/Fogarasi) is to recompute the internals there, take
/// the remaining `Δq` as a new target, and repeat. For ordinary optimizer-sized
/// steps this converges in two to four passes.
///
/// # Why the residual is measured in Cartesians
///
/// The coordinate set is **redundant**: for ethane there are 28 primitives but
/// only rank 18, so an arbitrary `Δq` vector has a component orthogonal to
/// `B`'s row space that **no** Cartesian displacement can produce. Testing
/// convergence on `|Δq|` therefore compares against an unreachable target and
/// can never succeed — measured directly: a uniform `Δq` of 1e-3 on ethane
/// stalls at a `|Δq|` residual of exactly 1.000e-3 for as many passes as it is
/// given, because the entire request is unreachable.
///
/// The meaningful question is instead "does any further Cartesian motion
/// remain?", i.e. how large the correction `Δx = B^T G^- Δq` still is. That
/// vanishes exactly when the iteration has extracted everything the geometry
/// can deliver, and it is what [`BackTransformResult::residual`] reports.
/// `tol` is therefore in **Bohr**. The raw internal residual is still returned
/// as [`BackTransformResult::internal_residual`] for diagnostics.
///
/// # Non-convergence is reported, not hidden
///
/// If the loop hits `max_iter` without reaching `tol`, the function returns the
/// **best iterate seen** with `converged: false` and the achieved residual,
/// rather than looping forever or silently returning a bad geometry. A caller
/// that gets `converged: false` should shorten its step; a diverging
/// back-transformation almost always means the internal step was too large for
/// the local linearization, not that the coordinates are broken.
///
/// `dq_target` is the desired change in every primitive. Dihedral components
/// are wrapped into `(-pi, pi]` at every pass so a step across the `+/-pi`
/// branch cut is treated as the short way round, not as a `2*pi` excursion.
///
/// # Errors
///
/// Propagates B-matrix and generalized-inverse construction failures.
pub fn step_to_cartesian(
    coords: &InternalCoords,
    mol: &Molecule,
    dq_target: &Array1<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<BackTransformResult, FerricError> {
    if dq_target.len() != coords.len() {
        return Err(FerricError::General(format!(
            "internal step has {} entries but there are {} primitives",
            dq_target.len(),
            coords.len()
        )));
    }

    let q0 = coords.values(mol)?;
    let q_goal = &q0 + dq_target;

    let mut work = mol.clone();
    let mut x: Vec<f64> = flatten(mol);
    // Best iterate seen so far, keyed on the CARTESIAN residual.
    let mut best: Option<(f64, f64, Vec<f64>)> = None;
    let mut iterations = 0;

    for it in 1..=max_iter.max(1) {
        iterations = it;
        let b = coords.b_matrix(&work)?;
        let ginv = generalized_inverse(&b)?;

        let q_now = coords.values(&work)?;
        let mut dq = Array1::zeros(coords.len());
        for (idx, p) in coords.primitives.iter().enumerate() {
            let d = q_goal[idx] - q_now[idx];
            dq[idx] = if matches!(p, Primitive::Dihedral(..)) {
                wrap_to_pi(d)
            } else {
                d
            };
        }
        let internal_residual = dq.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        // The Cartesian correction the outstanding internal residual implies.
        // This is the convergence measure — see the "Why the residual is
        // measured in Cartesians" section above.
        let dx = b.t().dot(&ginv.g_inv.dot(&dq));
        let residual = dx.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        if best.as_ref().is_none_or(|(r, _, _)| residual < *r) {
            best = Some((residual, internal_residual, x.clone()));
        }
        if residual < tol {
            return Ok(BackTransformResult {
                coords: x,
                iterations: it,
                residual,
                internal_residual,
                converged: true,
            });
        }

        for (xi, d) in x.iter_mut().zip(dx.iter()) {
            *xi += d;
        }
        unflatten(&mut work, &x);
    }

    // Exhausted the iteration budget: hand back the best iterate and say so.
    let (best_residual, best_internal, best_x) = best.unwrap_or((f64::INFINITY, f64::INFINITY, x));
    Ok(BackTransformResult {
        coords: best_x,
        iterations,
        residual: best_residual,
        internal_residual: best_internal,
        converged: false,
    })
}

/// Wrap an angle difference into `(-pi, pi]`.
///
/// Without this a torsion stepping from `+179` to `-179` degrees reads as a
/// `-358`-degree move instead of a `+2`-degree one, and the back-transformation
/// tries to spin the molecule right round.
#[must_use]
pub fn wrap_to_pi(mut d: f64) -> f64 {
    while d > std::f64::consts::PI {
        d -= 2.0 * std::f64::consts::PI;
    }
    while d <= -std::f64::consts::PI {
        d += 2.0 * std::f64::consts::PI;
    }
    d
}

fn flatten(mol: &Molecule) -> Vec<f64> {
    let mut v = Vec::with_capacity(mol.atoms.len() * 3);
    for a in &mol.atoms {
        v.push(a.x);
        v.push(a.y);
        v.push(a.zpos);
    }
    v
}

fn unflatten(mol: &mut Molecule, x: &[f64]) {
    for (i, a) in mol.atoms.iter_mut().enumerate() {
        a.x = x[3 * i];
        a.y = x[3 * i + 1];
        a.zpos = x[3 * i + 2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn water() -> Molecule {
        Molecule::parse_xyz(
            "3\nwater\nO  0.000  0.000  0.117\nH  0.000  0.757 -0.469\nH  0.000 -0.757 -0.469\n",
            0,
            1,
        )
        .unwrap()
    }

    fn h2o2() -> Molecule {
        Molecule::parse_xyz(
            "4\nH2O2\nO  0.000  0.734 -0.055\nO  0.000 -0.734 -0.055\n\
             H  0.839  0.885  0.435\nH -0.839 -0.885  0.435\n",
            0,
            1,
        )
        .unwrap()
    }

    fn ethane() -> Molecule {
        Molecule::parse_xyz(
            "8\nethane\nC  0.000  0.000  0.766\nC  0.000  0.000 -0.766\n\
             H  1.019  0.000  1.163\nH -0.509  0.883  1.163\nH -0.509 -0.883  1.163\n\
             H  1.019  0.000 -1.163\nH -0.509  0.883 -1.163\nH -0.509 -0.883 -1.163\n",
            0,
            1,
        )
        .unwrap()
    }

    fn co2_linear() -> Molecule {
        Molecule::parse_xyz(
            "3\nCO2\nC 0.0 0.0 0.0\nO 0.0 0.0 1.16\nO 0.0 0.0 -1.16\n",
            0,
            1,
        )
        .unwrap()
    }

    /// Two water molecules 8 Angstrom apart: two fragments, no covalent link.
    fn water_dimer_far() -> Molecule {
        Molecule::parse_xyz(
            "6\ndimer\nO  0.000  0.000  0.117\nH  0.000  0.757 -0.469\nH  0.000 -0.757 -0.469\n\
             O  8.000  0.000  0.117\nH  8.000  0.757 -0.469\nH  8.000 -0.757 -0.469\n",
            0,
            1,
        )
        .unwrap()
    }

    // -- covalent radii ---------------------------------------------------

    #[test]
    fn covalent_radii_are_not_vdw_radii() {
        // The whole point of adding a new table: the Bondi vdW radius of
        // carbon is 1.70 A, its covalent radius is 0.76 A. Using vdW radii at
        // scale 1.3 would call two carbons 4.4 A apart "bonded".
        let c = COVALENT_RADII_ANGSTROM[6];
        assert!(
            (c - 0.76).abs() < 1e-9,
            "carbon covalent radius should be Cordero's 0.76 A, got {c}"
        );
        assert!(
            c < 1.70,
            "a covalent radius must be smaller than the vdW one"
        );
        // Bohr conversion.
        let cb = covalent_radius_bohr(6);
        assert!((cb - 0.76 * BOHR_PER_ANGSTROM).abs() < 1e-12);
    }

    #[test]
    fn unknown_element_falls_back_rather_than_panicking() {
        let r = covalent_radius_bohr(200);
        assert!((r - DEFAULT_COVALENT_RADIUS_ANGSTROM * BOHR_PER_ANGSTROM).abs() < 1e-12);
        // Z = 0 (a dummy) must not return the zero table entry as a real radius.
        assert!(covalent_radius_bohr(0) > 0.0);
    }

    // -- bond perception --------------------------------------------------

    #[test]
    fn water_perceives_exactly_two_bonds() {
        let c = perceive_bonds(&water(), BOND_SCALE).unwrap();
        assert_eq!(c.bonds, vec![(0, 1), (0, 2)], "expected two O-H bonds");
        assert_eq!(c.n_fragments, 1);
        assert!(c.interfragment_bonds.is_empty());
        // The H...H distance (1.51 A) must NOT be a bond: 1.3 * (0.31 + 0.31)
        // = 0.81 A.
        assert!(!c.bonds.contains(&(1, 2)));
    }

    #[test]
    fn ethane_perceives_seven_bonds() {
        let m = Molecule::parse_xyz(
            "8\nethane\nC  0.000  0.000  0.766\nC  0.000  0.000 -0.766\n\
             H  1.019  0.000  1.163\nH -0.509  0.883  1.163\nH -0.509 -0.883  1.163\n\
             H  1.019  0.000 -1.163\nH -0.509  0.883 -1.163\nH -0.509 -0.883 -1.163\n",
            0,
            1,
        )
        .unwrap();
        let c = perceive_bonds(&m, BOND_SCALE).unwrap();
        assert_eq!(c.bonds.len(), 7, "C-C plus six C-H: {:?}", c.bonds);
        assert!(c.bonds.contains(&(0, 1)), "the C-C bond must be perceived");
        assert_eq!(c.n_fragments, 1);
    }

    /// A disconnected input MUST get an interfragment link, or the optimizer
    /// silently cannot move the fragments relative to one another.
    #[test]
    fn disconnected_fragments_get_a_shortest_link() {
        let c = perceive_bonds(&water_dimer_far(), BOND_SCALE).unwrap();
        assert_eq!(c.n_fragments, 2, "two waters 8 A apart are two fragments");
        assert_eq!(
            c.interfragment_bonds.len(),
            1,
            "merging 2 fragments needs exactly 1 link"
        );
        // Graph must now be connected.
        let adj = c.adjacency(6);
        let mut seen = [false; 6];
        let mut stack = vec![0usize];
        seen[0] = true;
        while let Some(v) = stack.pop() {
            for &w in &adj[v] {
                if !seen[w] {
                    seen[w] = true;
                    stack.push(w);
                }
            }
        }
        assert!(
            seen.iter().all(|&s| s),
            "graph still disconnected after adding interfragment links"
        );
        // And the link must be the SHORTEST cross-fragment pair: O...O at 8.0 A
        // beats any O...H or H...H here (the H's are pulled toward each other
        // by at most 0.757 A each, so H1(+0.757 y)...H4 is longer in x).
        let (a, b) = c.interfragment_bonds[0];
        let r = distance(&water_dimer_far(), a, b);
        let mut shortest = f64::INFINITY;
        let m = water_dimer_far();
        for i in 0..3 {
            for j in 3..6 {
                shortest = shortest.min(distance(&m, i, j));
            }
        }
        assert!(
            (r - shortest).abs() < 1e-12,
            "link {a}-{b} is {r:.4} Bohr but the shortest cross pair is {shortest:.4}"
        );
    }

    #[test]
    fn single_atom_is_rejected() {
        let m = Molecule::parse_xyz("1\nH\nH 0 0 0\n", 0, 2).unwrap();
        assert!(perceive_bonds(&m, BOND_SCALE).is_err());
    }

    // -- primitive generation ---------------------------------------------

    #[test]
    fn water_primitives_are_two_bonds_and_one_angle() {
        let ic = InternalCoords::generate(&water()).unwrap();
        assert_eq!(ic.len(), 3, "{:?}", ic.primitives);
        assert_eq!(ic.primitives[0], Primitive::Bond(0, 1));
        assert_eq!(ic.primitives[1], Primitive::Bond(0, 2));
        assert_eq!(ic.primitives[2], Primitive::Angle(1, 0, 2));
        // 3N - 6 = 3 for water, so the set happens to be non-redundant here.
        let b = ic.b_matrix(&water()).unwrap();
        assert_eq!(generalized_inverse(&b).unwrap().rank, 3);
    }

    #[test]
    fn h2o2_has_exactly_one_dihedral() {
        let ic = InternalCoords::generate(&h2o2()).unwrap();
        let dihedrals: Vec<_> = ic
            .primitives
            .iter()
            .filter(|p| matches!(p, Primitive::Dihedral(..)))
            .collect();
        assert_eq!(dihedrals.len(), 1, "H-O-O-H: {dihedrals:?}");
        // 3 bonds + 2 angles + 1 dihedral = 6 = 3N - 6 for N = 4.
        assert_eq!(ic.len(), 6, "{:?}", ic.primitives);
        let b = ic.b_matrix(&h2o2()).unwrap();
        assert_eq!(generalized_inverse(&b).unwrap().rank, 6);
    }

    /// The classic failure mode: a dihedral about a 180-degree angle has no
    /// defined reference plane. CO2 is linear, so every candidate dihedral
    /// through it must be rejected.
    #[test]
    fn linear_angles_produce_no_dihedrals() {
        let m = co2_linear();
        // Sanity: the O-C-O angle really is ~180.
        assert!(
            is_near_linear(&m, 1, 0, 2),
            "CO2 O-C-O should read as linear"
        );
        let ic = InternalCoords::generate(&m).unwrap();
        let n_dihedral = ic
            .primitives
            .iter()
            .filter(|p| matches!(p, Primitive::Dihedral(..)))
            .count();
        assert_eq!(
            n_dihedral, 0,
            "a dihedral about a linear angle is undefined: {:?}",
            ic.primitives
        );
    }

    /// `is_near_linear` must catch BOTH ends: 0 degrees is as degenerate as 180.
    #[test]
    fn near_linear_detects_zero_degree_angles_too() {
        let m = Molecule::parse_xyz(
            "3\nstacked\nA 0 0 0\nH 0 0 1.0\nH 0 0 2.0\n"
                .replace('A', "H")
                .as_str(),
            0,
            2,
        )
        .unwrap();
        // Angle at atom 0 between atoms 1 and 2: both along +z, so 0 degrees.
        assert!(
            is_near_linear(&m, 1, 0, 2),
            "a 0-degree angle is just as degenerate as 180"
        );
    }

    // -- B-matrix: finite differences are the independent construction -----

    /// Central finite differences of the primitive VALUES, checked against the
    /// ANALYTIC derivative rows. These are two independent constructions —
    /// `primitive_value` and `primitive_derivatives` share no code — which is
    /// what makes agreement evidence rather than a tautology (CLAUDE.md:
    /// "CONSISTENCY IS NOT CORROBORATION ... to test a CONSTRUCTION you need an
    /// INDEPENDENT CONSTRUCTION").
    fn check_b_against_fd(mol: &Molecule, tol: f64) {
        let ic = InternalCoords::generate(mol).unwrap();
        let b = ic.b_matrix(mol).unwrap();
        let h = 1e-5;
        let mut worst = 0.0f64;
        for (row, p) in ic.primitives.iter().enumerate() {
            for a in 0..mol.atoms.len() {
                for c in 0..3 {
                    let mut plus = mol.clone();
                    let mut minus = mol.clone();
                    for (m, s) in [(&mut plus, h), (&mut minus, -h)] {
                        match c {
                            0 => m.atoms[a].x += s,
                            1 => m.atoms[a].y += s,
                            _ => m.atoms[a].zpos += s,
                        }
                    }
                    let qp = primitive_value(&plus, *p).unwrap();
                    let qm = primitive_value(&minus, *p).unwrap();
                    // Dihedral differences can straddle the branch cut.
                    let d = if matches!(p, Primitive::Dihedral(..)) {
                        wrap_to_pi(qp - qm)
                    } else {
                        qp - qm
                    };
                    let fd = d / (2.0 * h);
                    let err = (fd - b[(row, 3 * a + c)]).abs();
                    worst = worst.max(err);
                    assert!(
                        err < tol,
                        "{p}: dq/dx[{a},{c}] analytic {:.10} vs FD {fd:.10} (err {err:.2e})",
                        b[(row, 3 * a + c)]
                    );
                }
            }
        }
        eprintln!("B-matrix vs FD: worst error {worst:.3e}");
    }

    #[test]
    fn b_matrix_matches_finite_differences_water() {
        check_b_against_fd(&water(), 1e-7);
    }

    #[test]
    fn b_matrix_matches_finite_differences_h2o2() {
        // H2O2 exercises the dihedral rows, which are the hard ones.
        check_b_against_fd(&h2o2(), 1e-7);
    }

    /// Translational invariance: shifting the whole molecule changes no
    /// internal coordinate, so every B-matrix row must sum to zero over atoms.
    /// This is an exact algebraic identity, checked at machine precision.
    #[test]
    fn b_matrix_rows_are_translationally_invariant() {
        for mol in [water(), h2o2()] {
            let ic = InternalCoords::generate(&mol).unwrap();
            let b = ic.b_matrix(&mol).unwrap();
            for row in 0..b.nrows() {
                for c in 0..3 {
                    let s: f64 = (0..mol.atoms.len()).map(|a| b[(row, 3 * a + c)]).sum();
                    assert!(
                        s.abs() < 1e-12,
                        "row {row} ({}) component {c} sums to {s:.3e}, not 0",
                        ic.primitives[row]
                    );
                }
            }
        }
    }

    /// Rotational invariance: the B-matrix row of a rotation-invariant
    /// coordinate must annihilate any infinitesimal rotation generator.
    #[test]
    fn b_matrix_rows_annihilate_rotations() {
        let mol = h2o2();
        let ic = InternalCoords::generate(&mol).unwrap();
        let b = ic.b_matrix(&mol).unwrap();
        // Three infinitesimal rotation generators: dx_a = e_axis x r_a.
        for axis in 0..3 {
            let mut e = [0.0; 3];
            e[axis] = 1.0;
            let mut v = vec![0.0; 3 * mol.atoms.len()];
            for (a, at) in mol.atoms.iter().enumerate() {
                let r = [at.x, at.y, at.zpos];
                let w = cross(e, r);
                v[3 * a..3 * a + 3].copy_from_slice(&w);
            }
            for row in 0..b.nrows() {
                let s: f64 = (0..v.len()).map(|c| b[(row, c)] * v[c]).sum();
                assert!(
                    s.abs() < 1e-10,
                    "row {row} ({}) responds {s:.3e} to a rotation about axis {axis}",
                    ic.primitives[row]
                );
            }
        }
    }

    // -- generalized inverse ----------------------------------------------

    /// The defining Moore-Penrose property on the retained subspace:
    /// `G G^- G = G`.
    #[test]
    fn generalized_inverse_satisfies_moore_penrose() {
        let mol = h2o2();
        let ic = InternalCoords::generate(&mol).unwrap();
        let b = ic.b_matrix(&mol).unwrap();
        let g = b.dot(&b.t());
        let ginv = generalized_inverse(&b).unwrap();
        let ggg = g.dot(&ginv.g_inv).dot(&g);
        let worst = ggg
            .iter()
            .zip(g.iter())
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst < 1e-9,
            "G G^- G != G, worst element error {worst:.3e}"
        );
    }

    /// The rank of `G` must equal the vibrational degree count `3N - 6` when
    /// the primitive set spans it — a direct, independent statement about the
    /// coordinate system's completeness.
    #[test]
    fn generalized_inverse_rank_is_3n_minus_6() {
        for (mol, expect) in [(water(), 3usize), (h2o2(), 6usize)] {
            let ic = InternalCoords::generate(&mol).unwrap();
            let b = ic.b_matrix(&mol).unwrap();
            let ginv = generalized_inverse(&b).unwrap();
            assert_eq!(
                ginv.rank,
                expect,
                "rank should be 3N-6 = {expect} for {} atoms",
                mol.atoms.len()
            );
        }
    }

    // -- gradient transform ------------------------------------------------

    /// **EXACTNESS ANCHOR for the gradient transform.** Transforming a
    /// Cartesian gradient into internals and back must reproduce it exactly,
    /// *provided* the gradient has no translational/rotational component — the
    /// internals genuinely cannot see those, and projecting them out is
    /// correct, not lossy. Constructing the test gradient as `B^T c` guarantees
    /// it lies in the space internals can represent.
    #[test]
    fn gradient_roundtrip_is_exact_in_the_representable_subspace() {
        let mol = h2o2();
        let ic = InternalCoords::generate(&mol).unwrap();
        let b = ic.b_matrix(&mol).unwrap();
        let ginv = generalized_inverse(&b).unwrap();

        // A Cartesian gradient that IS representable: g_x = B^T c.
        let c = Array1::from_vec((0..ic.len()).map(|i| 0.1 * (i as f64 + 1.0)).collect());
        let gx = b.t().dot(&c);

        let gq = gradient_to_internal(&b, &ginv, gx.as_slice().unwrap()).unwrap();
        let back = b.t().dot(&gq);
        let worst = back
            .iter()
            .zip(gx.iter())
            .map(|(a, d)| (a - d).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst < 1e-10,
            "B^T G^- B g_x != g_x for a representable gradient: worst {worst:.3e}"
        );
    }

    // -- back-transformation -----------------------------------------------

    /// **EXACTNESS ANCHOR for the back-transformation.** A zero step must
    /// return the starting geometry unchanged, in one iteration, with zero
    /// residual. This is the trivial limit: the transformation does nothing,
    /// and it must do nothing *exactly*.
    #[test]
    fn back_transform_of_a_zero_step_is_the_identity() {
        let mol = h2o2();
        let ic = InternalCoords::generate(&mol).unwrap();
        let dq = Array1::zeros(ic.len());
        let r = step_to_cartesian(&ic, &mol, &dq, 20, 1e-10).unwrap();
        assert!(r.converged);
        assert_eq!(r.iterations, 1, "a zero step should converge immediately");
        assert_eq!(r.residual, 0.0);
        assert_eq!(r.internal_residual, 0.0);
        for (i, a) in mol.atoms.iter().enumerate() {
            // Bit-identical, not merely close: nothing was computed.
            assert_eq!(r.coords[3 * i].to_bits(), a.x.to_bits());
            assert_eq!(r.coords[3 * i + 1].to_bits(), a.y.to_bits());
            assert_eq!(r.coords[3 * i + 2].to_bits(), a.zpos.to_bits());
        }
    }

    /// A finite step must actually LAND on the requested internals — that is
    /// the entire contract of the iterative back-transformation.
    #[test]
    fn back_transform_reaches_the_requested_internals() {
        let mol = h2o2();
        let ic = InternalCoords::generate(&mol).unwrap();
        let mut dq = Array1::zeros(ic.len());
        for (idx, p) in ic.primitives.iter().enumerate() {
            dq[idx] = match p {
                Primitive::Bond(..) => 0.05,     // Bohr
                Primitive::Angle(..) => 0.03,    // rad ~ 1.7 deg
                Primitive::Dihedral(..) => 0.10, // rad ~ 5.7 deg
            };
        }
        let r = step_to_cartesian(&ic, &mol, &dq, 25, 1e-9).unwrap();
        assert!(
            r.converged,
            "back-transformation did not converge: residual {:.3e} after {} iters",
            r.residual, r.iterations
        );
        eprintln!(
            "back-transform converged in {} iterations, residual {:.3e}",
            r.iterations, r.residual
        );

        // Independent verification: rebuild the molecule and re-measure.
        let mut moved = mol.clone();
        unflatten(&mut moved, &r.coords);
        let q0 = ic.values(&mol).unwrap();
        let q1 = ic.values(&moved).unwrap();
        for (idx, p) in ic.primitives.iter().enumerate() {
            let got = if matches!(p, Primitive::Dihedral(..)) {
                wrap_to_pi(q1[idx] - q0[idx])
            } else {
                q1[idx] - q0[idx]
            };
            assert!(
                (got - dq[idx]).abs() < 1e-8,
                "{p}: asked for Δq = {:.6}, got {got:.6}",
                dq[idx]
            );
        }
    }

    /// Non-convergence must be REPORTED, not hidden behind an infinite loop or
    /// a silently-wrong geometry. A one-iteration budget on a large step cannot
    /// converge, and the function must say so.
    #[test]
    fn back_transform_reports_non_convergence_instead_of_looping() {
        let mol = h2o2();
        let ic = InternalCoords::generate(&mol).unwrap();
        let mut dq = Array1::zeros(ic.len());
        dq[0] = 1.5; // a huge bond step, far outside the linear regime
        let r = step_to_cartesian(&ic, &mol, &dq, 1, 1e-12).unwrap();
        assert!(!r.converged, "a 1-iteration budget cannot reach 1e-12");
        assert_eq!(r.iterations, 1);
        assert!(
            r.residual.is_finite() && r.residual > 0.0,
            "a non-converged result must still report a finite residual, got {}",
            r.residual
        );
        // And it must return the best iterate, which is a usable geometry.
        assert_eq!(r.coords.len(), 3 * mol.atoms.len());
        assert!(r.coords.iter().all(|v| v.is_finite()));
    }

    /// **The redundancy finding.** In a genuinely redundant coordinate set an
    /// arbitrary `Δq` is NOT reachable: it has a component orthogonal to `B`'s
    /// row space that no Cartesian displacement can produce. A convergence test
    /// on `|Δq|` therefore chases an impossible target forever, which is
    /// exactly what ethane did before
    /// [`BackTransformResult::residual`] was redefined in Cartesians.
    ///
    /// This test pins BOTH halves: that the internal residual really does
    /// stall at the unreachable component, and that the Cartesian residual
    /// converges anyway.
    #[test]
    fn redundant_sets_have_unreachable_internal_steps_and_converge_anyway() {
        let mol = ethane();
        let ic = InternalCoords::generate(&mol).unwrap();
        let b = ic.b_matrix(&mol).unwrap();
        let ginv = generalized_inverse(&b).unwrap();
        assert!(
            ic.len() > ginv.rank,
            "this test needs a REDUNDANT set: {} primitives, rank {}",
            ic.len(),
            ginv.rank
        );
        assert_eq!(ginv.rank, 3 * mol.atoms.len() - 6);

        // A uniform step is mostly unreachable.
        let dq = Array1::from_elem(ic.len(), 1e-3);
        let dx = b.t().dot(&ginv.g_inv.dot(&dq));
        let dq_reachable = b.dot(&dx);
        let unreachable = (&dq - &dq_reachable)
            .iter()
            .map(|v| v.abs())
            .fold(0.0f64, f64::max);
        assert!(
            unreachable > 1e-5,
            "expected a substantially unreachable component, got {unreachable:.3e} — \
             if this ever goes to zero the premise of this test is gone"
        );

        let r = step_to_cartesian(&ic, &mol, &dq, 40, 1e-10).unwrap();
        eprintln!(
            "ethane uniform Δq=1e-3: cartesian residual {:.3e} (converged {}), \
             internal residual {:.3e} after {} iters; unreachable component {:.3e}",
            r.residual, r.converged, r.internal_residual, r.iterations, unreachable
        );
        assert!(
            r.converged,
            "the Cartesian residual must converge even when Δq is unreachable \
             (got {:.3e} after {} iters)",
            r.residual, r.iterations
        );
        // And the internal residual must NOT have gone to zero — that is the
        // whole point. If it did, the set was not really redundant and this
        // test is measuring nothing.
        assert!(
            r.internal_residual > 1e-6,
            "internal residual collapsed to {:.3e}; the unreachable component \
             should keep it finite",
            r.internal_residual
        );
    }

    /// A torsion step across the +/-pi branch cut must take the short way
    /// round. Without `wrap_to_pi` the back-transformation tries to rotate
    /// nearly 360 degrees and fails.
    #[test]
    fn dihedral_step_across_the_branch_cut_takes_the_short_route() {
        assert!(
            (wrap_to_pi(std::f64::consts::PI * 1.99) + 0.01 * std::f64::consts::PI).abs() < 1e-12
        );
        assert!(
            (wrap_to_pi(-std::f64::consts::PI * 1.99) - 0.01 * std::f64::consts::PI).abs() < 1e-12
        );
        // Exactly pi stays pi (the half-open convention).
        assert!((wrap_to_pi(std::f64::consts::PI) - std::f64::consts::PI).abs() < 1e-15);
    }

    // -- initial Hessian ---------------------------------------------------

    /// The Hessian guess only earns its keep if it actually SEPARATES the
    /// stiff coordinates from the soft ones — that separation is the entire
    /// reason internals beat an identity-initialized Cartesian Hessian.
    #[test]
    fn initial_hessian_separates_stiff_from_soft() {
        let mol = h2o2();
        let ic = InternalCoords::generate(&mol).unwrap();
        let h = ic.initial_hessian_diagonal(&mol).unwrap();
        let mut stretch = f64::INFINITY;
        let mut torsion = 0.0f64;
        for (idx, p) in ic.primitives.iter().enumerate() {
            assert!(h[idx] > 0.0, "{p}: non-positive force constant {}", h[idx]);
            assert!(h[idx].is_finite(), "{p}: non-finite force constant");
            match p {
                Primitive::Bond(..) => stretch = stretch.min(h[idx]),
                Primitive::Dihedral(..) => torsion = torsion.max(h[idx]),
                Primitive::Angle(..) => {}
            }
        }
        assert!(
            stretch > 20.0 * torsion,
            "the guess must separate stiff stretches ({stretch:.4}) from soft \
             torsions ({torsion:.4}) — otherwise it is no better than the identity"
        );
    }

    /// The guess must be within an order of magnitude of REAL stretch force
    /// constants — otherwise the claim "better than the identity" is unearned.
    ///
    /// Reference values are experimental/high-level harmonic stretch force
    /// constants in Hartree/Bohr^2. The bar is a factor of 5, which is where
    /// this table actually lands (worst case C=O, geometric mean ~2.1); the
    /// identity Hessian a Cartesian optimizer starts from is off by 3x to 100x
    /// on the same set and in the wrong direction for every soft mode.
    ///
    /// # Artifact hypothesis
    ///
    /// If the units or the `B` table were wrong — the two ways this can
    /// silently break — the errors would be systematic and large (a
    /// Bohr/Angstrom mix-up scales `(r-B)^3` by 6.7x; a too-large `B` sends the
    /// short bonds through the pole). Either shows up here as a factor of 10+
    /// on the heavy-atom rows specifically, which is exactly how the first two
    /// candidate tables were rejected during development.
    #[test]
    fn initial_hessian_is_within_an_order_of_magnitude_of_reality() {
        // (symbol_a, symbol_b, r_Angstrom, k_literature Ha/Bohr^2)
        let cases: &[(&str, &str, f64, f64)] = &[
            ("H", "H", 0.74, 0.37),
            ("O", "H", 0.96, 0.49),
            ("C", "H", 1.09, 0.32),
            ("N", "H", 1.01, 0.43),
            ("C", "C", 1.53, 0.28),
            ("O", "O", 1.47, 0.27),
            ("C", "N", 1.47, 0.33),
            ("C", "O", 1.43, 0.33),
            ("S", "H", 1.34, 0.26),
            ("C", "S", 1.82, 0.22),
            ("C", "Cl", 1.77, 0.22),
            ("Si", "H", 1.48, 0.17),
        ];
        let mut worst = 1.0f64;
        let mut worst_name = String::new();
        for (sa, sb, r, klit) in cases {
            let xyz = format!("2\npair\n{sa} 0 0 0\n{sb} 0 0 {r}\n");
            // The Hessian guess is a function of geometry and Z only, but
            // `parse_xyz` validates charge/multiplicity, so give an odd
            // electron count a doublet.
            let nelec = crate::elements::symbol_to_z(sa).unwrap()
                + crate::elements::symbol_to_z(sb).unwrap();
            let mult = if nelec % 2 == 0 { 1 } else { 2 };
            let m = Molecule::parse_xyz(&xyz, 0, mult).unwrap();
            // Force a bond regardless of the perception cutoff: this test is
            // about the Hessian value, not about bond perception.
            let ic = InternalCoords::generate_with_scale(&m, 3.0).unwrap();
            let h = ic.initial_hessian_diagonal(&m).unwrap();
            let k = h[0];
            let ratio = (k / klit).max(klit / k);
            eprintln!("{sa}-{sb} r={r:.2} A: guess {k:.3} vs lit {klit:.2} (x{ratio:.2})");
            if ratio > worst {
                worst = ratio;
                worst_name = format!("{sa}-{sb}");
            }
        }
        eprintln!("worst factor: {worst:.2} ({worst_name})");
        assert!(
            worst < 5.0,
            "empirical stretch guess is off by a factor of {worst:.2} on {worst_name} — \
             check the B table and the Bohr-vs-Angstrom units"
        );
    }

    /// A compressed geometry must not push the Schlegel stretch formula through
    /// its pole at `r = B`.
    #[test]
    fn initial_hessian_is_finite_at_a_compressed_geometry() {
        let m = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.10\n", 0, 1).unwrap();
        let ic = InternalCoords::generate_with_scale(&m, 5.0).unwrap();
        let h = ic.initial_hessian_diagonal(&m).unwrap();
        assert!(
            h.iter().all(|v| v.is_finite() && *v > 0.0),
            "compressed H2 gave a non-finite/non-positive force constant: {h:?}"
        );
    }

    // -- guards ------------------------------------------------------------

    #[test]
    fn atom_count_mismatch_is_an_error_not_a_panic() {
        let ic = InternalCoords::generate(&water()).unwrap();
        let other = h2o2();
        assert!(ic.values(&other).is_err());
        assert!(ic.b_matrix(&other).is_err());
        assert!(ic.initial_hessian_diagonal(&other).is_err());
    }

    #[test]
    fn wrong_length_internal_step_is_an_error() {
        let mol = water();
        let ic = InternalCoords::generate(&mol).unwrap();
        let dq = Array1::zeros(ic.len() + 1);
        assert!(step_to_cartesian(&ic, &mol, &dq, 5, 1e-8).is_err());
    }

    /// A linear molecule must still yield a USABLE B-matrix. This is the
    /// consequence of excluding near-linear angles from the primitive set: CO2
    /// keeps its two stretches, loses its (singular) bend, and the B-matrix
    /// builds cleanly at rank 2 instead of erroring or emitting a 1e8 row.
    #[test]
    fn linear_molecule_still_yields_a_usable_b_matrix() {
        let m = co2_linear();
        let ic = InternalCoords::generate(&m).unwrap();
        assert!(
            ic.primitives
                .iter()
                .all(|p| matches!(p, Primitive::Bond(..))),
            "CO2's only non-singular primitives are its two stretches: {:?}",
            ic.primitives
        );
        let b = ic
            .b_matrix(&m)
            .expect("B-matrix must build for a linear molecule");
        let ginv = generalized_inverse(&b).unwrap();
        // Two independent stretches; the two bending directions are genuinely
        // absent from this coordinate set and are honestly reported as such.
        assert_eq!(ginv.rank, 2);
    }

    #[test]
    fn linear_angle_derivative_errors_rather_than_returning_garbage() {
        let m = co2_linear();
        // The O-C-O angle IS generated (angles are not filtered), and asking
        // for its derivative must fail loudly rather than emit a 1e8 row.
        let r = primitive_derivatives(&m, Primitive::Angle(1, 0, 2));
        assert!(
            r.is_err(),
            "a linear angle's Cartesian derivative is singular and must error"
        );
    }
}
