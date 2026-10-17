//! Detection of the highest Abelian point group, reorientation, symmetrization.

use super::{AbelianGroup, OP_SIGNS};
use crate::mol::Molecule;
use crate::FerricError;

type V3 = [f64; 3];
type M3 = [V3; 3];

/// Candidate-generating atoms are capped at this many (rarest species/shell
/// classes first). A molecule with at most this many atoms is searched
/// exhaustively; beyond it, axes and planes are proposed only from the
/// rarest atom classes (every real symmetry element still permutes those).
const CANDIDATE_ATOM_CAP: usize = 60;
/// Two proposed directions closer than this (1 - |cos|) are one direction.
const DIRECTION_MERGE: f64 = 5e-7;
/// Two directions with |cos| below this are treated as orthogonal.
const ORTHOGONAL_COS: f64 = 1e-3;

/// Result of [`detect`].
#[derive(Debug, Clone)]
pub struct SymmetryDetection {
    /// The detected Abelian group.
    pub group: AbelianGroup,
    /// Ascending indices (into [`super::OP_SIGNS`]) of the operations in the group.
    pub ops: Vec<usize>,
    /// The input molecule translated to its nuclear-charge centroid, rotated
    /// into the symmetry frame and symmetrized. Atom order is the input's.
    pub molecule: Molecule,
    /// Rows are the symmetry-frame x, y, z axes in the input frame:
    /// `r_new = rotation * (r_old - origin)`.
    pub rotation: [[f64; 3]; 3],
    /// Nuclear-charge centroid of the input geometry (Bohr, input frame).
    pub origin: [f64; 3],
    /// For each operation in `ops`, `atom_perms[k][i]` is the atom that atom
    /// `i` is carried onto.
    pub atom_perms: Vec<Vec<usize>>,
    /// Largest atom displacement (Bohr) made by symmetrizing the geometry.
    pub max_symmetrization_shift: f64,
}

type AtomKey = (i32, bool, i32);

struct Cloud {
    pts: Vec<V3>,
    keys: Vec<AtomKey>,
    /// Match order: atoms of the rarest classes first, so a wrong operation
    /// fails after few comparisons.
    order: Vec<usize>,
    classes: Vec<Vec<usize>>,
}

/// Detect the highest Abelian subgroup of `mol`'s point group.
///
/// `tol` is the atom-matching distance in Bohr (see the module docs); use
/// [`super::DEFAULT_SYMMETRY_TOL`] unless there is a reason not to.
pub fn detect(mol: &Molecule, tol: f64) -> Result<SymmetryDetection, FerricError> {
    if mol.atoms.is_empty() {
        return Err(FerricError::General("symmetry detection: no atoms".into()));
    }
    if !(tol > 0.0 && tol.is_finite()) {
        return Err(FerricError::General(format!(
            "symmetry detection: tolerance must be positive and finite, got {tol}"
        )));
    }
    let origin = charge_centroid(mol);
    let cloud = Cloud::new(mol, origin, tol);
    let dirs = symmetry_directions(&cloud, tol);
    let best = best_frame(&cloud, &dirs, tol);
    let group = AbelianGroup::from_ops(&best.ops)?;
    let (molecule, shift) = symmetrized_molecule(mol, &cloud, &best);
    Ok(SymmetryDetection {
        group,
        ops: best.ops,
        molecule,
        rotation: best.axes,
        origin,
        atom_perms: best.perms,
        max_symmetrization_shift: shift,
    })
}

fn charge_centroid(mol: &Molecule) -> V3 {
    let mut c = [0.0; 3];
    let mut w = 0.0;
    for a in &mol.atoms {
        let z = a.z as f64;
        c[0] += z * a.x;
        c[1] += z * a.y;
        c[2] += z * a.zpos;
        w += z;
    }
    [c[0] / w, c[1] / w, c[2] / w]
}

impl Cloud {
    fn new(mol: &Molecule, origin: V3, tol: f64) -> Self {
        let pts: Vec<V3> = mol
            .atoms
            .iter()
            .map(|a| [a.x - origin[0], a.y - origin[1], a.zpos - origin[2]])
            .collect();
        let keys: Vec<AtomKey> = mol
            .atoms
            .iter()
            .map(|a| (a.z, a.ghost, a.n_core_ecp))
            .collect();
        let mut classes: Vec<Vec<usize>> = Vec::new();
        for i in 0..pts.len() {
            let r = norm(pts[i]);
            let home = classes.iter_mut().find(|c| {
                let j = c[0];
                keys[j] == keys[i] && (norm(pts[j]) - r).abs() < 4.0 * tol
            });
            match home {
                Some(c) => c.push(i),
                None => classes.push(vec![i]),
            }
        }
        classes.sort_by_key(|c| c.len());
        let order = classes.iter().flatten().copied().collect();
        Cloud {
            pts,
            keys,
            order,
            classes,
        }
    }

    /// Atom permutation induced by the linear map `m`, if it is a symmetry.
    fn find_perm(&self, m: &M3, tol: f64) -> Option<Vec<usize>> {
        let n = self.pts.len();
        let mut perm = vec![usize::MAX; n];
        let mut used = vec![false; n];
        for &i in &self.order {
            let q = matvec(m, self.pts[i]);
            let (d, j) = (0..n)
                .filter(|&j| self.keys[j] == self.keys[i])
                .map(|j| (dist(q, self.pts[j]), j))
                .fold((f64::MAX, usize::MAX), |a, b| if b.0 < a.0 { b } else { a });
            if d > tol || used[j] {
                return None;
            }
            used[j] = true;
            perm[i] = j;
        }
        Some(perm)
    }
}

fn is_symmetry_direction(cloud: &Cloud, v: V3, tol: f64) -> bool {
    cloud.find_perm(&c2_matrix(v), tol).is_some()
        || cloud.find_perm(&mirror_matrix(v), tol).is_some()
}

/// Directions that carry a C2 axis or a mirror normal of the molecule.
fn symmetry_directions(cloud: &Cloud, tol: f64) -> Vec<V3> {
    let mut accepted: Vec<V3> = Vec::new();
    for v in proposed_directions(cloud, tol) {
        if accepted
            .iter()
            .any(|u| 1.0 - dot(*u, v).abs() < DIRECTION_MERGE)
        {
            continue;
        }
        if is_symmetry_direction(cloud, v, tol) {
            accepted.push(v);
        }
    }
    accepted
}

/// Unit directions that any C2 axis or mirror normal must be among: inertia
/// axes, atom vectors, and for the rarest atoms pair sums, differences and
/// cross products.
fn proposed_directions(cloud: &Cloud, tol: f64) -> Vec<V3> {
    let mut raw: Vec<V3> = inertia_axes(&cloud.pts, &cloud.keys).to_vec();
    raw.extend(cloud.pts.iter().copied());
    let mut sel: Vec<usize> = Vec::new();
    for c in &cloud.classes {
        if sel.len() >= CANDIDATE_ATOM_CAP {
            break;
        }
        sel.extend(c.iter().copied());
    }
    for (a, &i) in sel.iter().enumerate() {
        for &j in &sel[a + 1..] {
            let (p, q) = (cloud.pts[i], cloud.pts[j]);
            raw.push(add(p, q));
            raw.push(sub(p, q));
            raw.push(cross(p, q));
        }
    }
    raw.into_iter()
        .filter(|v| norm(*v) > tol)
        .map(|v| scale(v, 1.0 / norm(v)))
        .collect()
}

/// A frame (rows = x, y, z) and the operations that survive in it.
struct FrameResult {
    axes: M3,
    ops: Vec<usize>,
    perms: Vec<Vec<usize>>,
}

impl FrameResult {
    /// Ranking: more operations, then C2(z) present, then sigma(xy) present.
    fn score(&self) -> (usize, bool, bool) {
        (self.ops.len(), self.ops.contains(&1), self.ops.contains(&5))
    }
}

fn evaluate_frame(cloud: &Cloud, axes: M3, tol: f64) -> FrameResult {
    let mut ops = Vec::new();
    let mut perms = Vec::new();
    for (k, s) in OP_SIGNS.iter().enumerate() {
        if let Some(p) = cloud.find_perm(&op_matrix(&axes, s), tol) {
            ops.push(k);
            perms.push(p);
        }
    }
    FrameResult { axes, ops, perms }
}

/// Frame (rows x, y, z) with z along `z` and x the part of `x` orthogonal to it.
fn frame_from(z: V3, x: V3) -> M3 {
    let xo = sub(x, scale(z, dot(x, z)));
    let xo = scale(xo, 1.0 / norm(xo));
    [xo, cross(z, xo), z]
}

fn arbitrary_perpendicular(z: V3) -> V3 {
    let k = (0..3)
        .min_by(|&a, &b| z[a].abs().total_cmp(&z[b].abs()))
        .unwrap();
    let mut e = [0.0; 3];
    e[k] = 1.0;
    e
}

/// Highest-order frame built from the symmetry directions (first wins ties).
fn best_frame(cloud: &Cloud, dirs: &[V3], tol: f64) -> FrameResult {
    let z_choices: Vec<V3> = if dirs.is_empty() {
        vec![[0.0, 0.0, 1.0]]
    } else {
        dirs.to_vec()
    };
    let mut best: Option<FrameResult> = None;
    for &z in &z_choices {
        let mut xs: Vec<V3> = dirs
            .iter()
            .copied()
            .filter(|u| dot(*u, z).abs() < ORTHOGONAL_COS)
            .collect();
        xs.push(arbitrary_perpendicular(z));
        for x in xs {
            let cand = evaluate_frame(cloud, frame_from(z, x), tol);
            if best.as_ref().map_or(true, |b| cand.score() > b.score()) {
                best = Some(cand);
            }
        }
        if best.as_ref().is_some_and(|b| b.ops.len() == 8) {
            break;
        }
    }
    best.expect("at least one frame is evaluated")
}

/// Replace each atom by the average of the images that land on it, in the
/// symmetry frame; return the molecule and the largest displacement.
fn symmetrized_molecule(mol: &Molecule, cloud: &Cloud, best: &FrameResult) -> (Molecule, f64) {
    let n = mol.atoms.len();
    let framed: Vec<V3> = cloud.pts.iter().map(|p| matvec(&best.axes, *p)).collect();
    let mut acc = vec![[0.0; 3]; n];
    for (&g, perm) in best.ops.iter().zip(&best.perms) {
        for i in 0..n {
            for ax in 0..3 {
                acc[perm[i]][ax] += OP_SIGNS[g][ax] as f64 * framed[i][ax];
            }
        }
    }
    let mut out = mol.clone();
    let mut shift = 0.0f64;
    let order = best.ops.len() as f64;
    for i in 0..n {
        let p = scale(acc[i], 1.0 / order);
        shift = shift.max(dist(p, framed[i]));
        let a = &mut out.atoms[i];
        a.x = p[0];
        a.y = p[1];
        a.zpos = p[2];
    }
    (out, shift)
}

fn op_matrix(axes: &M3, s: &[i8; 3]) -> M3 {
    // M = A^T diag(s) A
    let mut m = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] = (0..3).map(|k| axes[k][r] * s[k] as f64 * axes[k][c]).sum();
        }
    }
    m
}

fn c2_matrix(v: V3) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] = 2.0 * v[r] * v[c] - if r == c { 1.0 } else { 0.0 };
        }
    }
    m
}

fn mirror_matrix(v: V3) -> M3 {
    let mut m = c2_matrix(v);
    for row in m.iter_mut() {
        for e in row.iter_mut() {
            *e = -*e;
        }
    }
    m
}

/// Principal axes of the charge-weighted inertia tensor (eigenvectors).
fn inertia_axes(pts: &[V3], keys: &[AtomKey]) -> [V3; 3] {
    let mut t = [[0.0; 3]; 3];
    for (p, k) in pts.iter().zip(keys) {
        let w = k.0 as f64;
        let r2 = dot(*p, *p);
        for a in 0..3 {
            for b in 0..3 {
                t[a][b] += w * (if a == b { r2 } else { 0.0 } - p[a] * p[b]);
            }
        }
    }
    jacobi_eigenvectors(t)
}

/// Eigenvectors of a symmetric 3x3 matrix by cyclic Jacobi rotations.
fn jacobi_eigenvectors(mut a: M3) -> [V3; 3] {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-15 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for row in v.iter_mut() {
                let (vkp, vkq) = (row[p], row[q]);
                row[p] = c * vkp - s * vkq;
                row[q] = s * vkp + c * vkq;
            }
        }
    }
    [
        [v[0][0], v[1][0], v[2][0]],
        [v[0][1], v[1][1], v[2][1]],
        [v[0][2], v[1][2], v[2][2]],
    ]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn dist(a: V3, b: V3) -> f64 {
    norm(sub(a, b))
}
fn matvec(m: &M3, v: V3) -> V3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}
