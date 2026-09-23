//! Periodic cell: a molecule (the atoms of the reference cell) plus lattice
//! vectors, with the real- and reciprocal-space lattice enumerations every
//! periodic sum needs.
//!
//! Conventions (same as PySCF `pbc.gto.Cell` and `reference/pbc/pbc_gamma.py`):
//! * `lattice[i]` is the i-th lattice vector `a_i` (a ROW), Bohr.
//! * Reciprocal vectors `b_i` are the rows of `2π (A⁻¹)ᵀ`, so `a_i · b_j = 2π δ_ij`.
//! * A lattice translation is `L = Σ_i n_i a_i` with integer `n_i`.
//! * Atom positions are taken from the molecule as given (they need not be
//!   wrapped into the cell).

use ferric_core::mol::Molecule;
use ferric_core::FerricError;

/// Hard cap on the number of integer triples a single lattice enumeration may
/// visit. A cutoff that would exceed it is a caller error (or a nearly
/// degenerate cell), reported instead of silently allocating gigabytes.
const MAX_ENUMERATED: u64 = 200_000_000;

/// A periodic cell. Construct with [`Cell::new`], which validates the lattice.
#[derive(Debug, Clone)]
pub struct Cell {
    mol: Molecule,
    lattice: [[f64; 3]; 3],
    /// `A⁻¹` where `A` has the lattice vectors as rows.
    inv: [[f64; 3]; 3],
    volume: f64,
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: &[f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

impl Cell {
    /// Build a cell from a molecule (the atoms of the reference cell, Bohr) and
    /// three lattice vectors given as rows (Bohr).
    ///
    /// Errors if any lattice component is non-finite, the molecule has no
    /// atoms, or the lattice is singular / numerically degenerate: the test is
    /// `|det A| <= 1e-10 · |a_1||a_2||a_3|`, i.e. the (scale-free) sine of the
    /// cell's "solid angle" is below 1e-10.
    pub fn new(mol: Molecule, lattice: [[f64; 3]; 3]) -> Result<Self, FerricError> {
        if lattice.iter().flatten().any(|v| !v.is_finite()) {
            return Err(FerricError::General(format!(
                "Cell: lattice has a non-finite component: {lattice:?}"
            )));
        }
        if mol.atoms.is_empty() {
            return Err(FerricError::General(
                "Cell: the molecule has no atoms".into(),
            ));
        }
        let [a1, a2, a3] = lattice;
        let det = dot(&a1, &cross(&a2, &a3));
        let scale = norm(&a1) * norm(&a2) * norm(&a3);
        if !(det.abs() > 1e-10 * scale) || scale == 0.0 {
            return Err(FerricError::General(format!(
                "Cell: singular or degenerate lattice (det = {det:.3e}, |a1||a2||a3| = {scale:.3e}); \
                 the three lattice vectors must be linearly independent"
            )));
        }
        // A⁻¹ = adj(A)/det. With A's rows a1,a2,a3, the COLUMNS of A⁻¹ are
        // (a2×a3, a3×a1, a1×a2)/det.
        let c0 = cross(&a2, &a3);
        let c1 = cross(&a3, &a1);
        let c2 = cross(&a1, &a2);
        let mut inv = [[0.0; 3]; 3];
        for r in 0..3 {
            inv[r][0] = c0[r] / det;
            inv[r][1] = c1[r] / det;
            inv[r][2] = c2[r] / det;
        }
        Ok(Self {
            mol,
            lattice,
            inv,
            volume: det.abs(),
        })
    }

    /// The reference-cell molecule (unchanged from construction).
    pub fn mol(&self) -> &Molecule {
        &self.mol
    }

    /// Lattice vectors as rows, Bohr.
    pub fn lattice(&self) -> &[[f64; 3]; 3] {
        &self.lattice
    }

    /// Cell volume `|det A|`, Bohr³.
    pub fn volume(&self) -> f64 {
        self.volume
    }

    /// Reciprocal lattice vectors as rows: `b_i = 2π (A⁻¹)ᵀ[i]`, so that
    /// `a_i · b_j = 2π δ_ij`. Bohr⁻¹.
    pub fn reciprocal(&self) -> [[f64; 3]; 3] {
        let tp = 2.0 * std::f64::consts::PI;
        let mut b = [[0.0; 3]; 3];
        for i in 0..3 {
            for k in 0..3 {
                b[i][k] = tp * self.inv[k][i];
            }
        }
        b
    }

    /// Atom positions (Bohr), in molecule order.
    pub fn positions(&self) -> Vec<[f64; 3]> {
        self.mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect()
    }

    /// Nuclear charges seen by the electrons, `Atom::effective_z` (0 for a
    /// ghost, `Z − n_core` for an ECP atom), in molecule order.
    pub fn nuclear_charges(&self) -> Vec<f64> {
        self.mol
            .atoms
            .iter()
            .map(|a| a.effective_z() as f64)
            .collect()
    }

    /// Distance between adjacent lattice planes normal to `b_j`:
    /// `d_j = 2π/|b_j| = 1/|column j of A⁻¹|`.
    fn plane_spacings(&self) -> [f64; 3] {
        let mut d = [0.0; 3];
        for j in 0..3 {
            let col = [self.inv[0][j], self.inv[1][j], self.inv[2][j]];
            d[j] = 1.0 / norm(&col);
        }
        d
    }

    /// Lattice translations `L` such that the minimum atom–atom distance
    /// between the reference cell and the image shifted by `L` is `<= rcut`
    /// (Bohr). Sorted by `|L|` ascending; the first element is the zero
    /// vector (every atom is at distance 0 from itself). Port of
    /// `pbc_gamma.Cell.translations`.
    pub fn translations(&self, rcut: f64) -> Result<Vec<[f64; 3]>, FerricError> {
        self.translations_for(&self.positions(), rcut)
    }

    /// [`Cell::translations`] for an arbitrary set of reference-cell points
    /// (used by the Ewald sum for point charges that are not the molecule's
    /// atoms, e.g. the Madelung probe charge).
    pub(crate) fn translations_for(
        &self,
        pos: &[[f64; 3]],
        rcut: f64,
    ) -> Result<Vec<[f64; 3]>, FerricError> {
        if !(rcut >= 0.0) || !rcut.is_finite() {
            return Err(FerricError::General(format!(
                "Cell::translations: rcut must be finite and >= 0, got {rcut}"
            )));
        }
        if pos.is_empty() {
            return Err(FerricError::General(
                "Cell::translations: no reference points".into(),
            ));
        }
        // Largest intra-cell separation: any contributing L has
        // |L| <= rcut + ext, and L's projection on b̂_j is n_j d_j, so
        // |n_j| <= (rcut + ext)/d_j bounds the enumeration exactly.
        let mut ext = 0.0_f64;
        for p in pos {
            for q in pos {
                ext = ext.max(norm(&[p[0] - q[0], p[1] - q[1], p[2] - q[2]]));
            }
        }
        let d = self.plane_spacings();
        let mut nmax = [0i64; 3];
        let mut count: u64 = 1;
        for j in 0..3 {
            let n = ((rcut + ext) / d[j]).ceil() + 1.0;
            if !n.is_finite() || n > 1e6 {
                return Err(FerricError::General(format!(
                    "Cell::translations: rcut {rcut} needs |n_{j}| up to {n}; too many images"
                )));
            }
            nmax[j] = n as i64;
            count = count.saturating_mul((2 * nmax[j] + 1) as u64);
        }
        if count > MAX_ENUMERATED {
            return Err(FerricError::General(format!(
                "Cell::translations: rcut {rcut} would enumerate {count} lattice triples (cap {MAX_ENUMERATED})"
            )));
        }
        let a = &self.lattice;
        let mut out: Vec<[f64; 3]> = Vec::new();
        for n0 in -nmax[0]..=nmax[0] {
            for n1 in -nmax[1]..=nmax[1] {
                for n2 in -nmax[2]..=nmax[2] {
                    let (f0, f1, f2) = (n0 as f64, n1 as f64, n2 as f64);
                    let l = [
                        f0 * a[0][0] + f1 * a[1][0] + f2 * a[2][0],
                        f0 * a[0][1] + f1 * a[1][1] + f2 * a[2][1],
                        f0 * a[0][2] + f1 * a[1][2] + f2 * a[2][2],
                    ];
                    let zero = n0 == 0 && n1 == 0 && n2 == 0;
                    let mut dmin = f64::INFINITY;
                    for p in pos {
                        for q in pos {
                            let dv = [p[0] - q[0] - l[0], p[1] - q[1] - l[1], p[2] - q[2] - l[2]];
                            dmin = dmin.min(norm(&dv));
                        }
                    }
                    if zero {
                        // Exactly zero vector first after the sort below.
                        out.push([0.0; 3]);
                    } else if dmin <= rcut {
                        out.push(l);
                    }
                }
            }
        }
        // Stable sort by |L|; the exact zero vector is the unique minimum.
        out.sort_by(|x, y| norm(x).total_cmp(&norm(y)));
        Ok(out)
    }

    /// Reciprocal-lattice vectors `G = Σ_i n_i b_i` with `|G| <= gcut`
    /// (Bohr⁻¹), INCLUDING `G = 0` (callers filter it). Sorted by `|G|`
    /// ascending, so `G = 0` is first. Port of `pbc_gamma.Cell.gvectors`:
    /// `G · a_i = 2π n_i` gives the exact bound `|n_i| <= gcut |a_i| / 2π`.
    pub fn gvectors(&self, gcut: f64) -> Result<Vec<[f64; 3]>, FerricError> {
        if !(gcut >= 0.0) || !gcut.is_finite() {
            return Err(FerricError::General(format!(
                "Cell::gvectors: gcut must be finite and >= 0, got {gcut}"
            )));
        }
        let b = self.reciprocal();
        let tp = 2.0 * std::f64::consts::PI;
        let mut nmax = [0i64; 3];
        let mut count: u64 = 1;
        for i in 0..3 {
            let n = (gcut * norm(&self.lattice[i]) / tp).ceil();
            if n > 1e6 {
                return Err(FerricError::General(format!(
                    "Cell::gvectors: gcut {gcut} needs |n_{i}| up to {n}; too many G vectors"
                )));
            }
            nmax[i] = n as i64;
            count = count.saturating_mul((2 * nmax[i] + 1) as u64);
        }
        if count > MAX_ENUMERATED {
            return Err(FerricError::General(format!(
                "Cell::gvectors: gcut {gcut} would enumerate {count} lattice triples (cap {MAX_ENUMERATED})"
            )));
        }
        let g2max = gcut * gcut;
        let mut out: Vec<[f64; 3]> = Vec::new();
        for n0 in -nmax[0]..=nmax[0] {
            for n1 in -nmax[1]..=nmax[1] {
                for n2 in -nmax[2]..=nmax[2] {
                    let (f0, f1, f2) = (n0 as f64, n1 as f64, n2 as f64);
                    let g = [
                        f0 * b[0][0] + f1 * b[1][0] + f2 * b[2][0],
                        f0 * b[0][1] + f1 * b[1][1] + f2 * b[2][1],
                        f0 * b[0][2] + f1 * b[1][2] + f2 * b[2][2],
                    ];
                    if n0 == 0 && n1 == 0 && n2 == 0 {
                        out.push([0.0; 3]);
                    } else if dot(&g, &g) <= g2max {
                        out.push(g);
                    }
                }
            }
        }
        out.sort_by(|x, y| dot(x, x).total_cmp(&dot(y, y)));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::mol::Atom;

    fn h_atom(x: f64, y: f64, z: f64) -> Atom {
        Atom {
            symbol: "H".into(),
            z: 1,
            x,
            y,
            zpos: z,
            ghost: false,
            n_core_ecp: 0,
        }
    }

    fn cell() -> Cell {
        let mol = Molecule {
            atoms: vec![h_atom(0.1, 0.2, 0.3), h_atom(0.4, 0.5, 1.6)],
            charge: 0,
            multiplicity: 1,
        };
        Cell::new(mol, [[4.0, 0.0, 0.0], [0.8, 4.2, 0.0], [0.5, 0.6, 4.5]]).unwrap()
    }

    #[test]
    fn reciprocal_is_dual_basis() {
        let c = cell();
        let b = c.reciprocal();
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j {
                    2.0 * std::f64::consts::PI
                } else {
                    0.0
                };
                assert!((dot(&c.lattice()[i], &b[j]) - want).abs() < 1e-13);
            }
        }
        // 4.0 * 4.2 * 4.5 for this lower-triangular lattice.
        assert!((c.volume() - 75.6).abs() < 1e-12);
    }

    #[test]
    fn translations_start_at_zero_sorted_and_are_complete() {
        let c = cell();
        let rcut = 11.0;
        let ls = c.translations(rcut).unwrap();
        assert_eq!(ls[0], [0.0; 3]);
        for w in ls.windows(2) {
            assert!(norm(&w[0]) <= norm(&w[1]) + 1e-15);
        }
        // Brute force over a much larger integer box must find nothing extra.
        let pos = c.positions();
        let a = c.lattice();
        let mut brute = 0usize;
        for n0 in -8i32..=8 {
            for n1 in -8i32..=8 {
                for n2 in -8i32..=8 {
                    let l: Vec<f64> = (0..3)
                        .map(|k| n0 as f64 * a[0][k] + n1 as f64 * a[1][k] + n2 as f64 * a[2][k])
                        .collect();
                    let mut dmin = f64::INFINITY;
                    for p in &pos {
                        for q in &pos {
                            let dv = [p[0] - q[0] - l[0], p[1] - q[1] - l[1], p[2] - q[2] - l[2]];
                            dmin = dmin.min(norm(&dv));
                        }
                    }
                    if dmin <= rcut {
                        brute += 1;
                    }
                }
            }
        }
        assert_eq!(ls.len(), brute);
    }

    #[test]
    fn gvectors_include_zero_first_and_respect_cutoff() {
        let c = cell();
        let gcut = 5.0;
        let gs = c.gvectors(gcut).unwrap();
        assert_eq!(gs[0], [0.0; 3]);
        assert!(gs.iter().all(|g| dot(g, g) <= gcut * gcut + 1e-12));
        // Inversion symmetry of the set: G in set <=> -G in set.
        for g in &gs {
            let m = [-g[0], -g[1], -g[2]];
            assert!(gs
                .iter()
                .any(|h| (0..3).all(|k| (h[k] - m[k]).abs() < 1e-12)));
        }
    }
}
