//! Stage 3: k-point meshes (Monkhorst-Pack, Gamma-centred or shifted) with
//! EXACT Bloch phases and time-reversal pairing.
//!
//! Conventions (PySCF's, measured equal in `reference/pbc/FINDINGS.md`
//! "Iteration 9"):
//!
//! ```text
//! k      = Σ_i f_i b_i,       f_i = num_i / (2 N_i)            (fractional, integer numerator)
//! χ_μk(r) = Σ_L e^{ik·L} φ_μ(r − L),    L = Σ_i n_i a_i          (Bloch AO)
//! e^{ik·L} = exp(2πi Σ_i num_i n_i / (2 N_i))
//! ```
//!
//! * Gamma-centred (`MeshCentring::Gamma`, PySCF `cell.make_kpts(n)`):
//!   `f_i = m_i / N_i`, `m_i = 0..N_i−1`, i.e. `num_i = 2 m_i`.
//! * Monkhorst-Pack (`MeshCentring::MonkhorstPack`, PySCF
//!   `make_kpts(n, with_gamma_point=False)`): `f_i = (m_i + ½)/N_i − ½`,
//!   i.e. `num_i = 2 m_i + 1 − N_i` (odd `N_i`: the Gamma-centred set in
//!   another order/representative; even `N_i`: shifted by half a step).
//!
//! Order: `m_0` outer, `m_2` inner (PySCF's `cartesian_prod`, the prototype's
//! `mesh_index`).
//!
//! # Exact phases
//!
//! Every phase is computed from the INTEGER `Σ_i num_i n_i D/(2N_i) mod D`
//! (`D = Π 2N_i`), so quarter turns are exactly `±1, ±i`: at Gamma and at
//! every time-reversal-invariant k (TRIM, `2k ∈ G`) `S(k)`, `h(k)` are
//! exactly real sums, and `phase(−k) = conj(phase(k))` bit for bit (the
//! upper half-turn is evaluated as the conjugate of the lower one), so
//! `S(−k) = S(k)*` holds exactly, not to roundoff.
//!
//! # Time reversal
//!
//! For real AOs `X(−k) = X(k)*` for every one-particle matrix, and a
//! time-reversal-symmetric density stays so. The SCF diagonalises only the
//! representatives ([`KPointMesh::tr_representatives`]: `k <= minus(k)`) and
//! sets `C(−k) = C(k)*`, `ε(−k) = ε(k)`.
//!
//! # Supercell equivalence
//!
//! For a Gamma-centred mesh, `{G + q : q ∈ mesh}` IS the reciprocal lattice
//! of the `diag(N)` supercell, so a k-mesh RHF is term by term the Gamma RHF
//! of that supercell, per cell; the exchange-divergence constant is the
//! supercell's Madelung constant ([`KPointMesh::madelung`], = PySCF
//! `tools.pbc.madelung(cell, kpts)`). A SHIFTED mesh (even `N_i`, MP) is a
//! twisted-boundary supercell instead and has no such anchor.

use crate::ewald::madelung_constant;
use crate::lattice::Cell;
use ferric_core::FerricError;
use num_complex::Complex64;

/// Mesh centring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshCentring {
    /// `f_i = m_i/N_i` (contains Gamma; PySCF `make_kpts(n)`).
    Gamma,
    /// `f_i = (m_i + ½)/N_i − ½` (PySCF `make_kpts(n, with_gamma_point=False)`).
    MonkhorstPack,
}

/// Hard cap on the mesh size (dense k-point code: `N_k²` kernels).
pub const MAX_KPOINTS: usize = 4096;

/// A k-point mesh on a cell's reciprocal lattice.
#[derive(Debug, Clone)]
pub struct KPointMesh {
    n: [usize; 3],
    centring: MeshCentring,
    /// Numerators over `2 N_i`, per k-point.
    nums: Vec<[i64; 3]>,
    kpts: Vec<[f64; 3]>,
    minus: Vec<usize>,
    lattice: [[f64; 3]; 3],
}

/// `exp(2πi t/d)`, exact at quarter turns; `unit_root(d − t) =
/// conj(unit_root(t))` bit for bit.
pub(crate) fn unit_root(t: i64, d: i64) -> Complex64 {
    debug_assert!(d > 0);
    let t = t.rem_euclid(d);
    if 2 * t > d {
        return unit_root(d - t, d).conj();
    }
    if t == 0 {
        Complex64::new(1.0, 0.0)
    } else if 4 * t == d {
        Complex64::new(0.0, 1.0)
    } else if 2 * t == d {
        Complex64::new(-1.0, 0.0)
    } else {
        let th = 2.0 * std::f64::consts::PI * (t as f64) / (d as f64);
        Complex64::new(th.cos(), th.sin())
    }
}

/// Integer lattice coordinates `n_i` of a translation `L = Σ n_i a_i`
/// (`n_i = round(L·b_i/2π)`, exact for the lattice vectors `Cell`
/// enumerates).
pub(crate) fn lattice_coords(b: &[[f64; 3]; 3], l: &[f64; 3]) -> [i64; 3] {
    let tp = 2.0 * std::f64::consts::PI;
    let mut n = [0i64; 3];
    for i in 0..3 {
        n[i] = ((l[0] * b[i][0] + l[1] * b[i][1] + l[2] * b[i][2]) / tp).round() as i64;
    }
    n
}

impl KPointMesh {
    /// Build an `n[0] × n[1] × n[2]` mesh of the given centring on `cell`'s
    /// reciprocal lattice. Errors on a zero axis or more than
    /// [`MAX_KPOINTS`] points.
    pub fn new(cell: &Cell, n: [usize; 3], centring: MeshCentring) -> Result<Self, FerricError> {
        if n.contains(&0) {
            return Err(FerricError::General(format!(
                "KPointMesh: every axis needs >= 1 point, got {n:?}"
            )));
        }
        let nk = n[0].saturating_mul(n[1]).saturating_mul(n[2]);
        if nk > MAX_KPOINTS {
            return Err(FerricError::General(format!(
                "KPointMesh: {n:?} has {nk} points (cap {MAX_KPOINTS})"
            )));
        }
        let num_of = |m: usize, ni: usize| -> i64 {
            match centring {
                MeshCentring::Gamma => 2 * m as i64,
                MeshCentring::MonkhorstPack => 2 * m as i64 + 1 - ni as i64,
            }
        };
        let b = cell.reciprocal();
        let mut nums = Vec::with_capacity(nk);
        let mut kpts = Vec::with_capacity(nk);
        for m0 in 0..n[0] {
            for m1 in 0..n[1] {
                for m2 in 0..n[2] {
                    let nm = [num_of(m0, n[0]), num_of(m1, n[1]), num_of(m2, n[2])];
                    let f = [
                        nm[0] as f64 / (2 * n[0]) as f64,
                        nm[1] as f64 / (2 * n[1]) as f64,
                        nm[2] as f64 / (2 * n[2]) as f64,
                    ];
                    let mut k = [0.0; 3];
                    for (i, fi) in f.iter().enumerate() {
                        for d in 0..3 {
                            k[d] += fi * b[i][d];
                        }
                    }
                    nums.push(nm);
                    kpts.push(k);
                }
            }
        }
        let lattice = *cell.lattice();
        let mut mesh = Self {
            n,
            centring,
            nums,
            kpts,
            minus: Vec::new(),
            lattice,
        };
        let mut minus = Vec::with_capacity(nk);
        for k in 0..nk {
            let m = mesh.nums[k];
            let idx = mesh.index_of_nums([-m[0], -m[1], -m[2]]).ok_or_else(|| {
                FerricError::General(format!(
                    "KPointMesh: −k of point {k} is not on the mesh (internal error)"
                ))
            })?;
            minus.push(idx);
        }
        mesh.minus = minus;
        Ok(mesh)
    }

    /// Gamma-centred `n` mesh (PySCF `cell.make_kpts(n)`).
    pub fn gamma_centred(cell: &Cell, n: [usize; 3]) -> Result<Self, FerricError> {
        Self::new(cell, n, MeshCentring::Gamma)
    }

    /// Monkhorst-Pack `n` mesh (PySCF `make_kpts(n, with_gamma_point=False)`).
    pub fn monkhorst_pack(cell: &Cell, n: [usize; 3]) -> Result<Self, FerricError> {
        Self::new(cell, n, MeshCentring::MonkhorstPack)
    }

    /// Points per axis.
    pub fn n(&self) -> [usize; 3] {
        self.n
    }

    /// Centring.
    pub fn centring(&self) -> MeshCentring {
        self.centring
    }

    /// Number of k-points.
    pub fn nk(&self) -> usize {
        self.kpts.len()
    }

    /// Cartesian k-points (Bohr⁻¹), mesh order.
    pub fn kpts(&self) -> &[[f64; 3]] {
        &self.kpts
    }

    /// Fractional coordinates of point `k` (in units of the `b_i`).
    pub fn frac(&self, k: usize) -> [f64; 3] {
        let m = self.nums[k];
        [
            m[0] as f64 / (2 * self.n[0]) as f64,
            m[1] as f64 / (2 * self.n[1]) as f64,
            m[2] as f64 / (2 * self.n[2]) as f64,
        ]
    }

    /// Integer numerators (over `2 N_i`) of point `k`.
    pub fn numerators(&self, k: usize) -> [i64; 3] {
        self.nums[k]
    }

    /// Index of the point equivalent (mod G) to fractional numerators
    /// `x_i / (2 N_i)`, if it is on the mesh.
    pub fn index_of_nums(&self, x: [i64; 3]) -> Option<usize> {
        let mut m = [0usize; 3];
        for i in 0..3 {
            let ni = self.n[i] as i64;
            let off = self.nums_axis0(i);
            let d = x[i] - off;
            if d.rem_euclid(2) != 0 {
                return None;
            }
            m[i] = (d / 2).rem_euclid(ni) as usize;
        }
        Some((m[0] * self.n[1] + m[1]) * self.n[2] + m[2])
    }

    /// Numerator of `m_i = 0` on axis `i`.
    fn nums_axis0(&self, i: usize) -> i64 {
        match self.centring {
            MeshCentring::Gamma => 0,
            MeshCentring::MonkhorstPack => 1 - self.n[i] as i64,
        }
    }

    /// Index of `−k` (time-reversal partner).
    pub fn minus(&self, k: usize) -> usize {
        self.minus[k]
    }

    /// `2k ∈ G` (time-reversal invariant: `S(k)`, `h(k)` real).
    pub fn is_trim(&self, k: usize) -> bool {
        self.minus[k] == k
    }

    /// One point of each `{k, −k}` pair (the lower index), ascending.
    pub fn tr_representatives(&self) -> Vec<usize> {
        (0..self.nk()).filter(|&k| k <= self.minus[k]).collect()
    }

    /// `e^{ik·L}` for a lattice translation with integer coordinates `nl`,
    /// exact at quarter turns (module doc).
    pub fn phase(&self, k: usize, nl: [i64; 3]) -> Complex64 {
        let d: i64 = (0..3).map(|i| 2 * self.n[i] as i64).product();
        let mut t: i64 = 0;
        for i in 0..3 {
            let di = 2 * self.n[i] as i64;
            // (num · n mod 2N) keeps the product small before scaling.
            let r = (self.nums[k][i] * nl[i].rem_euclid(di)).rem_euclid(di);
            t += r * (d / di);
        }
        unit_root(t, d)
    }

    /// Per-axis residue moduli `M_i` such that `e^{ik·L}` depends only on
    /// `n_i mod M_i` for EVERY mesh point: `N_i` when all numerators on the
    /// axis are even (Gamma-centred, or MP with odd `N_i`), else `2 N_i`.
    pub fn residue_moduli(&self) -> [usize; 3] {
        let mut m = [0usize; 3];
        for i in 0..3 {
            let even = self.nums_axis0(i).rem_euclid(2) == 0;
            m[i] = if even { self.n[i] } else { 2 * self.n[i] };
        }
        m
    }

    /// Momentum-transfer classes: the Gamma-centred `N` mesh of
    /// `q = k' − k`. Returns, for class `iq` with integer `m_q`
    /// (`q = Σ m_q,i/N_i b_i`), the Cartesian `q` (representative in
    /// `[0, 1)` fractional) and `m_q`.
    pub fn q_class(&self, iq: usize) -> ([f64; 3], [i64; 3]) {
        let m = [
            (iq / (self.n[1] * self.n[2])) as i64,
            ((iq / self.n[2]) % self.n[1]) as i64,
            (iq % self.n[2]) as i64,
        ];
        let b = reciprocal_of(&self.lattice);
        let mut q = [0.0; 3];
        for i in 0..3 {
            let f = m[i] as f64 / self.n[i] as f64;
            for d in 0..3 {
                q[d] += f * b[i][d];
            }
        }
        (q, m)
    }

    /// Class index of `−q` for class `iq`.
    pub fn q_minus(&self, iq: usize) -> usize {
        let (_, m) = self.q_class(iq);
        let r = [
            (-m[0]).rem_euclid(self.n[0] as i64) as usize,
            (-m[1]).rem_euclid(self.n[1] as i64) as usize,
            (-m[2]).rem_euclid(self.n[2] as i64) as usize,
        ];
        (r[0] * self.n[1] + r[1]) * self.n[2] + r[2]
    }

    /// The point `k` with `k' − k = q` (class `m_q`), i.e. `k = k' − q`.
    pub fn k_minus_q(&self, kp: usize, mq: [i64; 3]) -> usize {
        let x = self.nums[kp];
        self.index_of_nums([x[0] - 2 * mq[0], x[1] - 2 * mq[1], x[2] - 2 * mq[2]])
            .expect("k' − q is on the mesh (q classes are Gamma-centred)")
    }

    /// Lattice of the `diag(N)` supercell: rows `N_i a_i`.
    pub fn supercell_lattice(&self) -> [[f64; 3]; 3] {
        let mut s = self.lattice;
        for (i, row) in s.iter_mut().enumerate() {
            for v in row.iter_mut() {
                *v *= self.n[i] as f64;
            }
        }
        s
    }

    /// The exchange-divergence constant of this mesh: the Gamma Madelung
    /// constant of the `diag(N)` supercell (= PySCF
    /// `tools.pbc.madelung(cell, kpts)`; measured equal to the k-mesh
    /// constant in the prototype). `cell` supplies the molecule only to
    /// satisfy [`Cell::new`]; the constant depends on the lattice alone.
    pub fn madelung(&self, cell: &Cell) -> Result<f64, FerricError> {
        let sc = Cell::new(cell.mol().clone(), self.supercell_lattice())?;
        madelung_constant(&sc)
    }
}

fn reciprocal_of(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let cross = |x: &[f64; 3], y: &[f64; 3]| {
        [
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ]
    };
    let c = [
        cross(&a[1], &a[2]),
        cross(&a[2], &a[0]),
        cross(&a[0], &a[1]),
    ];
    let det = a[0][0] * c[0][0] + a[0][1] * c[0][1] + a[0][2] * c[0][2];
    let tp = 2.0 * std::f64::consts::PI;
    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for d in 0..3 {
            b[i][d] = tp * c[i][d] / det;
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_root_is_exact_at_quarter_turns_and_conjugate_symmetric() {
        assert_eq!(unit_root(0, 6), Complex64::new(1.0, 0.0));
        assert_eq!(unit_root(3, 6), Complex64::new(-1.0, 0.0));
        assert_eq!(unit_root(2, 8), Complex64::new(0.0, 1.0));
        assert_eq!(unit_root(6, 8), Complex64::new(0.0, -1.0));
        assert_eq!(unit_root(-3, 6), Complex64::new(-1.0, 0.0));
        for d in 1..13 {
            for t in -20..20 {
                assert_eq!(unit_root(-t, d), unit_root(t, d).conj(), "t={t} d={d}");
                let th = 2.0 * std::f64::consts::PI * t as f64 / d as f64;
                let z = unit_root(t, d);
                // The REFERENCE cos/sin(2*pi*t/d) evaluates |theta| up to ~126 rad,
                // whose own argument rounding is ~1.4e-14; unit_root reduces
                // exactly, so bound by the reference's error, not 1e-15.
                assert!((z.re - th.cos()).abs() < 4e-14 && (z.im - th.sin()).abs() < 4e-14);
            }
        }
    }
}
