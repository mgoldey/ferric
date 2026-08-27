//! MM topology: explicit AMBER-form bonded and nonbonded parameters.
//!
//! Every parameter here is **explicit, caller-supplied data** — this crate
//! assigns no force field of its own (see [`crate`] docs). [`MmTopology::new`]
//! only does bookkeeping: it validates indices and derives, from the bond
//! list, which atom pairs are excluded from nonbonded interactions (1-2, 1-3)
//! and which are scaled 1-4 pairs. AMBER convention: a pair that is
//! simultaneously 1-3 (via one path) and 1-4 (via another, e.g. in a small
//! ring) is **excluded**, not scaled — exclusion always wins.

use ferric_core::FerricError;
use std::collections::{HashMap, HashSet};

/// Lennard-Jones parameters for one atom. Atomic units (`sigma` in Bohr,
/// `epsilon` in Hartree).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LjParams {
    pub sigma: f64,
    pub epsilon: f64,
}

/// Harmonic bond stretch: `E = k (r - r0)^2` — the AMBER convention, with
/// **no** leading 1/2 (OpenMM's `HarmonicBondForce` uses `1/2 k (r-r0)^2`, so
/// `k_openmm = 2 k_amber`; see `scripts/gen_openmm_mm_refs.py`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bond {
    pub i: usize,
    pub j: usize,
    pub k: f64,
    pub r0: f64,
}

/// Harmonic angle bend: `E = k (theta - theta0)^2`, `theta0` in radians, same
/// no-1/2 AMBER convention as [`Bond`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Angle {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub k_theta: f64,
    pub theta0: f64,
}

/// Periodic torsion (proper or improper — impropers are just torsions with a
/// caller-chosen atom order): `E = k (1 + cos(n*phi - delta))`, `phase` in
/// radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Torsion {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub l: usize,
    pub periodicity: u32,
    pub k_phi: f64,
    pub phase: f64,
}

/// AMBER default 1-4 Lennard-Jones scale factor.
pub const DEFAULT_SCALE_LJ_14: f64 = 0.5;
/// AMBER default 1-4 Coulomb scale factor (`1/1.2`).
pub const DEFAULT_SCALE_COUL_14: f64 = 1.0 / 1.2;

/// A complete MM topology: charges, LJ parameters, bonded terms, and the
/// exclusion/1-4 bookkeeping [`MmTopology::new`] derives from the bond graph.
///
/// Atomic units throughout (Hartree, Bohr, radians) — use
/// [`MmTopology::from_amber_units`] to convert from the kcal/mol/Å/degree
/// convention force fields are usually published in.
#[derive(Debug, Clone)]
pub struct MmTopology {
    pub charges: Vec<f64>,
    pub lj: Vec<LjParams>,
    pub bonds: Vec<Bond>,
    pub angles: Vec<Angle>,
    pub torsions: Vec<Torsion>,
    /// LJ scale factor applied to 1-4 pairs (AMBER default 0.5).
    pub scale_lj_14: f64,
    /// Coulomb scale factor applied to 1-4 pairs (AMBER default 1/1.2).
    pub scale_coul_14: f64,
    /// 1-2 and 1-3 pairs (through any bond path of length 1 or 2), excluded
    /// entirely from nonbonded interactions. `(i, j)` with `i < j`, deduped.
    exclusions: HashSet<(usize, usize)>,
    /// 1-4 pairs (bond path of length exactly 3, i.e. `i-a-b-j`), scaled by
    /// `scale_lj_14`/`scale_coul_14`. `(i, j)` with `i < j`, deduped, and
    /// **never** overlapping `exclusions` — a pair reachable by both a
    /// length-<=2 path and a length-3 path is excluded, not scaled (see
    /// module docs).
    pairs14: HashSet<(usize, usize)>,
}

#[inline]
fn ordered(i: usize, j: usize) -> (usize, usize) {
    if i <= j {
        (i, j)
    } else {
        (j, i)
    }
}

impl MmTopology {
    /// Number of atoms (the length of `charges`/`lj`).
    pub fn n_atoms(&self) -> usize {
        self.charges.len()
    }

    /// Read-only view of the derived 1-2/1-3 exclusion set.
    pub fn exclusions(&self) -> &HashSet<(usize, usize)> {
        &self.exclusions
    }

    /// Read-only view of the derived 1-4 pair set.
    pub fn pairs14(&self) -> &HashSet<(usize, usize)> {
        &self.pairs14
    }

    /// Build a topology in atomic units, validating every index and deriving
    /// 1-2/1-3 exclusions and 1-4 pairs from `bonds` via a BFS over the bond
    /// graph. Uses the AMBER default scale factors
    /// ([`DEFAULT_SCALE_LJ_14`]/[`DEFAULT_SCALE_COUL_14`]); use
    /// [`MmTopology::with_scales`] afterwards to override them.
    pub fn new(
        charges: Vec<f64>,
        lj: Vec<LjParams>,
        bonds: Vec<Bond>,
        angles: Vec<Angle>,
        torsions: Vec<Torsion>,
    ) -> Result<Self, FerricError> {
        let n = charges.len();
        if lj.len() != n {
            return Err(FerricError::General(format!(
                "MmTopology::new: {n} charges but {} LJ params",
                lj.len()
            )));
        }
        let check_idx = |label: &str, idx: usize| -> Result<(), FerricError> {
            if idx >= n {
                Err(FerricError::General(format!(
                    "MmTopology::new: {label} index {idx} out of range ({n} atoms)"
                )))
            } else {
                Ok(())
            }
        };
        for b in &bonds {
            check_idx("bond", b.i)?;
            check_idx("bond", b.j)?;
        }
        for a in &angles {
            check_idx("angle", a.i)?;
            check_idx("angle", a.j)?;
            check_idx("angle", a.k)?;
        }
        for t in &torsions {
            check_idx("torsion", t.i)?;
            check_idx("torsion", t.j)?;
            check_idx("torsion", t.k)?;
            check_idx("torsion", t.l)?;
            if t.periodicity == 0 {
                return Err(FerricError::General(
                    "MmTopology::new: torsion periodicity must be >= 1".to_string(),
                ));
            }
        }

        // Adjacency list from the bond graph.
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for b in &bonds {
            adj.entry(b.i).or_default().push(b.j);
            adj.entry(b.j).or_default().push(b.i);
        }

        // For each atom, BFS out to depth 3 to classify every reachable atom
        // by shortest bond-path length. 1-2/1-3 (depth 1 or 2) => exclusion;
        // depth == 3 with no shorter path => 1-4 pair.
        let mut exclusions: HashSet<(usize, usize)> = HashSet::new();
        let mut pairs14: HashSet<(usize, usize)> = HashSet::new();
        for start in 0..n {
            let mut depth: HashMap<usize, usize> = HashMap::new();
            depth.insert(start, 0);
            let mut frontier = vec![start];
            for d in 1..=3 {
                let mut next = Vec::new();
                for &node in &frontier {
                    if let Some(neighbors) = adj.get(&node) {
                        for &nb in neighbors {
                            if let std::collections::hash_map::Entry::Vacant(e) = depth.entry(nb) {
                                e.insert(d);
                                next.push(nb);
                            }
                        }
                    }
                }
                frontier = next;
            }
            for (&other, &d) in &depth {
                if other == start {
                    continue;
                }
                let pair = ordered(start, other);
                match d {
                    1 | 2 => {
                        exclusions.insert(pair);
                    }
                    3 => {
                        pairs14.insert(pair);
                    }
                    _ => {}
                }
            }
        }
        // Exclusion wins over 1-4 for any pair reachable both ways (e.g. a
        // small ring where a 1-4 path coexists with a 1-3 path).
        for pair in &exclusions {
            pairs14.remove(pair);
        }

        Ok(Self {
            charges,
            lj,
            bonds,
            angles,
            torsions,
            scale_lj_14: DEFAULT_SCALE_LJ_14,
            scale_coul_14: DEFAULT_SCALE_COUL_14,
            exclusions,
            pairs14,
        })
    }

    /// Override the 1-4 scale factors (default AMBER 0.5 / 1/1.2).
    pub fn with_scales(mut self, scale_lj_14: f64, scale_coul_14: f64) -> Self {
        self.scale_lj_14 = scale_lj_14;
        self.scale_coul_14 = scale_coul_14;
        self
    }

    /// Build a topology from AMBER-convention units: `charges` in `e`,
    /// `sigma`/`r0` in Å, `epsilon`/`k`(bond)/`k_theta`(angle)/`k_phi`(torsion)
    /// in kcal/mol (bond `k` in kcal/mol/Å^2, angle `k_theta` in
    /// kcal/mol/rad^2 — `theta0`/`phase` in **degrees**, converted here), and
    /// converts once to Hartree/Bohr/radians.
    #[allow(clippy::too_many_arguments)]
    pub fn from_amber_units(
        charges: Vec<f64>,
        lj_angstrom_kcal: Vec<(f64, f64)>,
        bonds_amber: Vec<(usize, usize, f64, f64)>,
        angles_amber: Vec<(usize, usize, usize, f64, f64)>,
        torsions_amber: Vec<(usize, usize, usize, usize, u32, f64, f64)>,
    ) -> Result<Self, FerricError> {
        use crate::units::{deg_to_rad, ANGSTROM_TO_BOHR, KCAL_PER_MOL_TO_HARTREE};

        let lj = lj_angstrom_kcal
            .into_iter()
            .map(|(sigma, epsilon)| LjParams {
                sigma: sigma * ANGSTROM_TO_BOHR,
                epsilon: epsilon * KCAL_PER_MOL_TO_HARTREE,
            })
            .collect();
        let bonds = bonds_amber
            .into_iter()
            .map(|(i, j, k, r0)| Bond {
                i,
                j,
                // k: kcal/mol/Å^2 -> Hartree/Bohr^2
                k: k * KCAL_PER_MOL_TO_HARTREE / (ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR),
                r0: r0 * ANGSTROM_TO_BOHR,
            })
            .collect();
        let angles = angles_amber
            .into_iter()
            .map(|(i, j, k, k_theta, theta0_deg)| Angle {
                i,
                j,
                k,
                k_theta: k_theta * KCAL_PER_MOL_TO_HARTREE,
                theta0: deg_to_rad(theta0_deg),
            })
            .collect();
        let torsions = torsions_amber
            .into_iter()
            .map(|(i, j, k, l, periodicity, k_phi, phase_deg)| Torsion {
                i,
                j,
                k,
                l,
                periodicity,
                k_phi: k_phi * KCAL_PER_MOL_TO_HARTREE,
                phase: deg_to_rad(phase_deg),
            })
            .collect();

        Self::new(charges, lj, bonds, angles, torsions)
    }
}
