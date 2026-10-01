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
//!
//! # Frozen index sets (strained cells, the Gamma stress)
//!
//! Every truncated lattice set — the G sphere `|G| <= gcut` and every image
//! list `translations(rcut)` — is chosen by distance. Under a homogeneous
//! strain `r → (1 + ε) r` of the lattice rows AND the atoms, re-selecting the
//! sets at each strain makes `E(ε)` piecewise (a G vector crossing the sphere
//! is a jump; FINDINGS "Iteration 19" (g): 3e-6..7.6e-6 Ha at an unconverged
//! gcut, i.e. an FD "stress" error of jump/2h). [`Cell::strained`] returns a
//! cell that carries its REFERENCE cell: every enumeration then selects the
//! integer triples (Miller indices `n` with `G = Σ n_i b_i`, lattice indices
//! with `L = Σ n_i a_i`) exactly as the reference cell would — distances
//! measured in the reference frame, with the query points mapped to the
//! reference by their fractional coordinates — and returns them built from
//! the STRAINED `a`/`b`. The energy of a strained cell is then a smooth
//! function of ε, and [`crate::stress`] is its exact derivative. An unstrained
//! cell (`Cell::new`) selects in its own frame, bit for bit as before.

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use std::sync::Arc;

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
    /// The reference cell whose integer index sets every enumeration reuses
    /// (module doc, "Frozen index sets"); `None` for an ordinary cell. Never
    /// itself frozen (a strain of a strained cell keeps the ORIGINAL
    /// reference).
    frozen: Option<Arc<Cell>>,
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
            frozen: None,
        })
    }

    /// The homogeneously strained cell `r → (1 + ε) r`: lattice rows
    /// `a_i → (1 + ε) a_i` and every atom `R_A → (1 + ε) R_A` (fixed
    /// fractional coordinates), with every G sphere and image list FROZEN as
    /// integer indices at the reference cell (module doc). `eps[i][j]` is
    /// `ε_ij` (`F = 1 + ε`, `x'_i = Σ_j F_ij x_j`). The reference is `self`,
    /// or `self`'s own reference if `self` is already strained, so strains
    /// compose without re-selecting anything.
    ///
    /// Errors when the strained lattice is degenerate (as [`Cell::new`]).
    pub fn strained(&self, eps: &[[f64; 3]; 3]) -> Result<Cell, FerricError> {
        if eps.iter().flatten().any(|v| !v.is_finite()) {
            return Err(FerricError::General(format!(
                "Cell::strained: non-finite strain {eps:?}"
            )));
        }
        let apply = |v: [f64; 3]| -> [f64; 3] {
            let mut out = [0.0; 3];
            for (i, o) in out.iter_mut().enumerate() {
                *o = v[i] + eps[i][0] * v[0] + eps[i][1] * v[1] + eps[i][2] * v[2];
            }
            out
        };
        let lattice = [
            apply(self.lattice[0]),
            apply(self.lattice[1]),
            apply(self.lattice[2]),
        ];
        let mut mol = self.mol.clone();
        for a in &mut mol.atoms {
            let r = apply([a.x, a.y, a.zpos]);
            a.x = r[0];
            a.y = r[1];
            a.zpos = r[2];
        }
        let mut c = Cell::new(mol, lattice)?;
        c.frozen = Some(match &self.frozen {
            Some(r) => Arc::clone(r),
            None => Arc::new(self.clone()),
        });
        Ok(c)
    }

    /// The reference cell whose index sets this (strained) cell reuses;
    /// `None` for a cell built with [`Cell::new`].
    pub fn index_reference(&self) -> Option<&Cell> {
        self.frozen.as_deref()
    }

    /// `x` (Bohr) mapped to the index-selection frame: the point with the
    /// same fractional coordinates in the reference lattice (identity for an
    /// ordinary cell).
    pub fn to_index_frame(&self, x: [f64; 3]) -> [f64; 3] {
        match &self.frozen {
            None => x,
            Some(r) => {
                let f = self.fractional(x);
                frac_to_cart(&r.lattice, f)
            }
        }
    }

    /// Fractional coordinates `f` of `x` (`x = Σ_i f_i a_i`).
    pub fn fractional(&self, x: [f64; 3]) -> [f64; 3] {
        let mut f = [0.0; 3];
        for (c, fc) in f.iter_mut().enumerate() {
            *fc = x[0] * self.inv[0][c] + x[1] * self.inv[1][c] + x[2] * self.inv[2][c];
        }
        f
    }

    /// The lattice translation `L = Σ_i n_i a_i` of an integer index triple.
    pub fn translation_from_index(&self, n: [i64; 3]) -> [f64; 3] {
        index_combination(&self.lattice, n)
    }

    /// The reciprocal-lattice vector `G = Σ_i n_i b_i` of a Miller index.
    pub fn gvector_from_index(&self, n: [i64; 3]) -> [f64; 3] {
        index_combination(&self.reciprocal(), n)
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

    /// Integer indices `n` of [`Cell::translations`] (`L = Σ n_i a_i`), in
    /// the same order. For a strained cell they are the REFERENCE cell's
    /// selection (module doc).
    pub fn translation_indices(&self, rcut: f64) -> Result<Vec<[i64; 3]>, FerricError> {
        self.translation_indices_for(&self.positions(), rcut)
    }

    /// Miller indices `n` of [`Cell::gvectors`] (`G = Σ n_i b_i`), in the same
    /// order. For a strained cell they are the REFERENCE cell's selection.
    pub fn gvector_indices(&self, gcut: f64) -> Result<Vec<[i64; 3]>, FerricError> {
        match &self.frozen {
            None => self.select_gvector_indices(gcut),
            Some(r) => r.select_gvector_indices(gcut),
        }
    }

    /// Upper bound on `self.translations(rcut).len()`: the number of integer
    /// triples the enumeration visits (the returned list is the subset within
    /// `rcut`). Memory gates size the translation list with it BEFORE the
    /// enumeration allocates. Errors exactly when `translations` would.
    pub fn translation_count_bound(&self, rcut: f64) -> Result<u64, FerricError> {
        if !(rcut >= 0.0) || !rcut.is_finite() {
            return Err(FerricError::General(format!(
                "Cell::translations: rcut must be finite and >= 0, got {rcut}"
            )));
        }
        match &self.frozen {
            None => Ok(self.translation_box(&self.positions(), rcut)?.1),
            Some(r) => {
                let pos: Vec<[f64; 3]> = self
                    .positions()
                    .iter()
                    .map(|p| self.to_index_frame(*p))
                    .collect();
                Ok(r.translation_box(&pos, rcut)?.1)
            }
        }
    }

    /// Enumeration box `|n_j| <= nmax[j]` of [`Cell::translations_for`] and
    /// its triple count (capped at [`MAX_ENUMERATED`]).
    fn translation_box(&self, pos: &[[f64; 3]], rcut: f64) -> Result<([i64; 3], u64), FerricError> {
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
        Ok((nmax, count))
    }

    /// [`Cell::translations`] for an arbitrary set of reference-cell points
    /// (used by the Ewald sum for point charges that are not the molecule's
    /// atoms, e.g. the Madelung probe charge).
    pub(crate) fn translations_for(
        &self,
        pos: &[[f64; 3]],
        rcut: f64,
    ) -> Result<Vec<[f64; 3]>, FerricError> {
        Ok(self
            .translation_indices_for(pos, rcut)?
            .into_iter()
            .map(|n| self.translation_from_index(n))
            .collect())
    }

    /// Indices of [`Cell::translations_for`]: selected in this cell's own
    /// frame, or (strained cell) in the reference frame with `pos` mapped by
    /// fractional coordinates.
    pub(crate) fn translation_indices_for(
        &self,
        pos: &[[f64; 3]],
        rcut: f64,
    ) -> Result<Vec<[i64; 3]>, FerricError> {
        match &self.frozen {
            None => self.select_translation_indices(pos, rcut),
            Some(r) => {
                let pr: Vec<[f64; 3]> = pos.iter().map(|p| self.to_index_frame(*p)).collect();
                r.select_translation_indices(&pr, rcut)
            }
        }
    }

    /// The distance selection itself, in THIS cell's frame (never the
    /// reference's): every `n` in the enumeration box whose image is within
    /// `rcut` (minimum point–point distance), sorted by `|L|` ascending with
    /// the exact zero triple first (stable sort over the box order).
    fn select_translation_indices(
        &self,
        pos: &[[f64; 3]],
        rcut: f64,
    ) -> Result<Vec<[i64; 3]>, FerricError> {
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
        let (nmax, _) = self.translation_box(pos, rcut)?;
        let mut out: Vec<([i64; 3], f64)> = Vec::new();
        for n0 in -nmax[0]..=nmax[0] {
            for n1 in -nmax[1]..=nmax[1] {
                for n2 in -nmax[2]..=nmax[2] {
                    let n = [n0, n1, n2];
                    let l = self.translation_from_index(n);
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
                        out.push((n, 0.0));
                    } else if dmin <= rcut {
                        out.push((n, norm(&l)));
                    }
                }
            }
        }
        // Stable sort by |L|; the exact zero vector is the unique minimum.
        out.sort_by(|x, y| x.1.total_cmp(&y.1));
        Ok(out.into_iter().map(|(n, _)| n).collect())
    }

    /// Upper bound on `self.gvectors(gcut).len()` (the enumeration box's
    /// triple count), for sizing the G list BEFORE it is allocated. Errors
    /// exactly when `gvectors` would.
    pub fn gvector_count_bound(&self, gcut: f64) -> Result<u64, FerricError> {
        if !(gcut >= 0.0) || !gcut.is_finite() {
            return Err(FerricError::General(format!(
                "Cell::gvectors: gcut must be finite and >= 0, got {gcut}"
            )));
        }
        match &self.frozen {
            None => Ok(self.gvector_box(gcut)?.1),
            Some(r) => Ok(r.gvector_box(gcut)?.1),
        }
    }

    /// Enumeration box `|n_i| <= nmax[i]` of [`Cell::gvectors`] and its
    /// triple count (capped at [`MAX_ENUMERATED`]).
    fn gvector_box(&self, gcut: f64) -> Result<([i64; 3], u64), FerricError> {
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
        Ok((nmax, count))
    }

    /// Reciprocal-lattice vectors `G = Σ_i n_i b_i` with `|G| <= gcut`
    /// (Bohr⁻¹), INCLUDING `G = 0` (callers filter it). Sorted by `|G|`
    /// ascending, so `G = 0` is first. Port of `pbc_gamma.Cell.gvectors`:
    /// `G · a_i = 2π n_i` gives the exact bound `|n_i| <= gcut |a_i| / 2π`.
    /// A strained cell returns the reference cell's Miller indices built from
    /// its own `b` (module doc).
    pub fn gvectors(&self, gcut: f64) -> Result<Vec<[f64; 3]>, FerricError> {
        let b = self.reciprocal();
        Ok(self
            .gvector_indices(gcut)?
            .into_iter()
            .map(|n| index_combination(&b, n))
            .collect())
    }

    /// The `|G| <= gcut` selection in THIS cell's frame (never the
    /// reference's), sorted by `|G|²` (stable over the box order), zero first.
    fn select_gvector_indices(&self, gcut: f64) -> Result<Vec<[i64; 3]>, FerricError> {
        if !(gcut >= 0.0) || !gcut.is_finite() {
            return Err(FerricError::General(format!(
                "Cell::gvectors: gcut must be finite and >= 0, got {gcut}"
            )));
        }
        let b = self.reciprocal();
        let (nmax, _) = self.gvector_box(gcut)?;
        let g2max = gcut * gcut;
        let mut out: Vec<([i64; 3], f64)> = Vec::new();
        for n0 in -nmax[0]..=nmax[0] {
            for n1 in -nmax[1]..=nmax[1] {
                for n2 in -nmax[2]..=nmax[2] {
                    let n = [n0, n1, n2];
                    let g = index_combination(&b, n);
                    if n0 == 0 && n1 == 0 && n2 == 0 {
                        out.push((n, 0.0));
                    } else if dot(&g, &g) <= g2max {
                        out.push((n, dot(&g, &g)));
                    }
                }
            }
        }
        out.sort_by(|x, y| x.1.total_cmp(&y.1));
        Ok(out.into_iter().map(|(n, _)| n).collect())
    }
}

/// `Σ_i n_i rows[i]`, evaluated in exactly the expression order the
/// enumerations always used (so an unstrained cell's vectors are bitwise
/// unchanged, and `(−n)` gives the exact negation).
fn index_combination(rows: &[[f64; 3]; 3], n: [i64; 3]) -> [f64; 3] {
    if n == [0, 0, 0] {
        // The exact +0.0 zero vector the enumerations always returned.
        return [0.0; 3];
    }
    let (f0, f1, f2) = (n[0] as f64, n[1] as f64, n[2] as f64);
    [
        f0 * rows[0][0] + f1 * rows[1][0] + f2 * rows[2][0],
        f0 * rows[0][1] + f1 * rows[1][1] + f2 * rows[2][1],
        f0 * rows[0][2] + f1 * rows[1][2] + f2 * rows[2][2],
    ]
}

/// `Σ_i f_i rows[i]` for real `f`.
fn frac_to_cart(rows: &[[f64; 3]; 3], f: [f64; 3]) -> [f64; 3] {
    [
        f[0] * rows[0][0] + f[1] * rows[1][0] + f[2] * rows[2][0],
        f[0] * rows[0][1] + f[1] * rows[1][1] + f[2] * rows[2][1],
        f[0] * rows[0][2] + f[1] * rows[1][2] + f[2] * rows[2][2],
    ]
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

    #[test]
    fn strained_cells_reuse_the_reference_index_sets() {
        let c = cell();
        let (gcut, rcut) = (5.0, 11.0);
        // Zero strain: bitwise the reference's vectors.
        let z = c.strained(&[[0.0; 3]; 3]).unwrap();
        assert!(z.index_reference().is_some());
        assert_eq!(z.gvectors(gcut).unwrap(), c.gvectors(gcut).unwrap());
        assert_eq!(z.translations(rcut).unwrap(), c.translations(rcut).unwrap());
        // A strain large enough to re-select both sets if they were chosen by
        // distance in the strained frame.
        let eps = [[0.04, 0.03, -0.02], [0.01, -0.05, 0.02], [0.0, 0.02, 0.06]];
        let st = c.strained(&eps).unwrap();
        let n_ref = c.gvector_indices(gcut).unwrap();
        assert_eq!(st.gvector_indices(gcut).unwrap(), n_ref);
        let b = st.reciprocal();
        for (n, g) in n_ref.iter().zip(st.gvectors(gcut).unwrap()) {
            let want = index_combination(&b, *n);
            assert_eq!(g, want);
        }
        assert_eq!(
            st.translation_indices(rcut).unwrap(),
            c.translation_indices(rcut).unwrap()
        );
        // G(ε) = (1 + ε)^{-T} G_0: G(ε)·(1 + ε) a_i = G_0·a_i = 2π n_i.
        for (n, g) in n_ref.iter().zip(st.gvectors(gcut).unwrap()) {
            for i in 0..3 {
                let want = 2.0 * std::f64::consts::PI * n[i] as f64;
                assert!((dot(&g, &st.lattice()[i]) - want).abs() < 1e-10);
            }
        }
        // Strains compose onto the ORIGINAL reference.
        let st2 = st
            .strained(&[[0.01, 0.0, 0.0], [0.0; 3], [0.0; 3]])
            .unwrap();
        assert_eq!(st2.gvector_indices(gcut).unwrap(), n_ref);
        // The re-selected (plain) strained cell really does differ: the frozen
        // sets are doing something.
        let plain = Cell::new(st.mol().clone(), *st.lattice()).unwrap();
        assert_ne!(plain.gvector_indices(gcut).unwrap(), n_ref);
        // Fractional coordinates are strain-invariant.
        let f0 = c.fractional(c.positions()[1]);
        let f1 = st.fractional(st.positions()[1]);
        for k in 0..3 {
            assert!((f0[k] - f1[k]).abs() < 1e-13);
        }
    }
}
