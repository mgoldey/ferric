//! AO layout shared by ferric (libint) and cuEST, and the map between their
//! AO orders.
//!
//! Both sides see the same shells in the same order: atoms in molecule order,
//! each atom's shells in `BasisSet::for_element` order (the order
//! `PreparedBasis::new` hands to libint). They can differ *inside* a shell:
//!
//! * ferric/libint never marks s or p shells pure, while cuEST refuses a basis
//!   that mixes pure and Cartesian shells. A spherical basis (pure d and up)
//!   is therefore sent to cuEST all-pure, so its p shells come back in
//!   cuEST's solid-harmonic order, not ferric's x, y, z.
//! * cuEST's order and phase for pure components need not match libint's.
//!
//! [`AoMap`] holds, for each cuEST AO, the ferric AO it equals and a sign.
//! [`AoMap::hypothesis`] is our guess; [`infer_map`] recovers the real map
//! from data (cuEST matrices against ferric references), so one GPU run is
//! enough to confirm or correct the guess.

use crate::{CuestError, Result};
use ferric_core::basis::{num_functions, BasisSet};
use ferric_core::mol::Molecule;
use ndarray::Array2;
use serde::Serialize;

/// One shell as both libraries see it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellInfo {
    pub atom: usize,
    pub l: i32,
    /// Representation ferric uses for this shell.
    pub ferric_pure: bool,
    /// Representation sent to cuEST (uniform over the basis).
    pub cuest_pure: bool,
    /// First AO of this shell (same offset on both sides: the counts agree).
    pub offset: usize,
}

impl ShellInfo {
    pub fn nfunc(&self) -> usize {
        num_functions(self.l, self.ferric_pure)
    }
}

#[derive(Debug, Clone)]
pub struct AoLayout {
    pub shells: Vec<ShellInfo>,
    pub nao: usize,
    /// The single pure flag every shell is sent to cuEST with.
    pub cuest_pure: bool,
}

impl AoLayout {
    /// Build the layout, refusing what cuEST cannot represent.
    pub fn new(mol: &Molecule, bs: &BasisSet) -> Result<Self> {
        let mut high_pure: Option<bool> = None;
        for atom in &mol.atoms {
            let shells = bs.for_element(atom.z).ok_or_else(|| {
                CuestError::Unsupported(format!("basis {:?} has no Z={}", bs.name, atom.z))
            })?;
            for sh in shells.iter().filter(|s| s.l >= 2) {
                match high_pure {
                    None => high_pure = Some(sh.pure),
                    Some(p) if p != sh.pure => {
                        return Err(CuestError::Unsupported(format!(
                            "basis {:?} mixes spherical and Cartesian l>=2 shells on this molecule",
                            bs.name
                        )))
                    }
                    _ => {}
                }
            }
        }
        // No l>=2 shell: s/p only, Cartesian == ferric's convention.
        let cuest_pure = high_pure.unwrap_or(false);
        let mut out = Vec::new();
        let mut offset = 0;
        for (ai, atom) in mol.atoms.iter().enumerate() {
            for sh in bs.for_element(atom.z).unwrap_or_default() {
                // The pure flag only changes the function count for l>=2, where
                // it agrees with ferric's by construction, so offsets match.
                let info = ShellInfo {
                    atom: ai,
                    l: sh.l,
                    ferric_pure: sh.pure,
                    cuest_pure,
                    offset,
                };
                offset += info.nfunc();
                out.push(info);
            }
        }
        Ok(Self {
            shells: out,
            nao: offset,
            cuest_pure,
        })
    }

    /// Shells per atom, in atom order (what `cuestAOBasisCreate` takes).
    pub fn shells_per_atom(&self, natoms: usize) -> Vec<u64> {
        let mut n = vec![0u64; natoms];
        for sh in &self.shells {
            n[sh.atom] += 1;
        }
        n
    }
}

/// Within-shell component map for one class of shell: entry `k` is the
/// (ferric component index, sign) equal to cuEST component `k`.
pub type ComponentMap = Vec<(usize, f64)>;

/// Our guess at cuEST's component order, per shell class, from the cuEST 0.3
/// basis documentation (docs.nvidia.com/cuda/cuest/0.3.0/gaussian_basis.html).
///
/// * l = 0, or Cartesian on both sides: identity. Both codes order Cartesian
///   components lexically (xx, xy, xz, yy, yz, zz).
/// * pure on both sides (l >= 2): cuEST orders m as 0, +1, -1, +2, -2, ...;
///   libint (STANDARD ordering) as -l..+l, so cuEST component k is ferric
///   component l + m(k). Both use real solid harmonics with Racah
///   normalization and no Condon-Shortley phase (cuEST's l = 2 table and
///   ferric's `ao_grid` formulas agree sign for sign), so every sign is +1.
/// * cuEST pure p vs ferric Cartesian p (ferric never sends pure p): cuEST's
///   pure p is z, x, y, so cuEST component k maps to ferric `[2, 0, 1][k]`.
///
/// Still UNVERIFIED on a device; [`infer_map`] reports the real order.
pub fn hypothesis_components(l: i32, cuest_pure: bool, ferric_pure: bool) -> ComponentMap {
    if l >= 1 && cuest_pure {
        // m(k) = 0, +1, -1, +2, -2, ...
        let m = |k: i32| if k % 2 == 1 { (k + 1) / 2 } else { -(k / 2) };
        if ferric_pure {
            return (0..2 * l + 1).map(|k| ((l + m(k)) as usize, 1.0)).collect();
        }
        if l == 1 {
            return vec![(2, 1.0), (0, 1.0), (1, 1.0)];
        }
    }
    (0..num_functions(l, ferric_pure))
        .map(|k| (k, 1.0))
        .collect()
}

/// Full map: `entries[i]` = (ferric AO, sign) for cuEST AO `i`.
#[derive(Debug, Clone, PartialEq)]
pub struct AoMap {
    pub entries: Vec<(usize, f64)>,
}

impl AoMap {
    pub fn hypothesis(layout: &AoLayout) -> Self {
        Self::from_components(layout, |sh| {
            hypothesis_components(sh.l, sh.cuest_pure, sh.ferric_pure)
        })
    }

    pub fn from_components(layout: &AoLayout, f: impl Fn(&ShellInfo) -> ComponentMap) -> Self {
        let mut entries = Vec::with_capacity(layout.nao);
        for sh in &layout.shells {
            for (j, s) in f(sh) {
                entries.push((sh.offset + j, s));
            }
        }
        Self { entries }
    }

    /// Reorder a cuEST-order matrix into ferric order: M_f[f(i), f(j)] = s_i s_j M_c[i, j].
    pub fn to_ferric(&self, m_cuest: &Array2<f64>) -> Array2<f64> {
        let n = self.entries.len();
        assert_eq!(m_cuest.dim(), (n, n), "matrix/map size mismatch");
        let mut out = Array2::<f64>::zeros((n, n));
        for (i, &(fi, si)) in self.entries.iter().enumerate() {
            for (j, &(fj, sj)) in self.entries.iter().enumerate() {
                out[[fi, fj]] = si * sj * m_cuest[[i, j]];
            }
        }
        out
    }

    /// Inverse of [`to_ferric`](Self::to_ferric) (used to build test fixtures).
    pub fn to_cuest(&self, m_ferric: &Array2<f64>) -> Array2<f64> {
        let n = self.entries.len();
        let mut out = Array2::<f64>::zeros((n, n));
        for (i, &(fi, si)) in self.entries.iter().enumerate() {
            for (j, &(fj, sj)) in self.entries.iter().enumerate() {
                out[[i, j]] = si * sj * m_ferric[[fi, fj]];
            }
        }
        out
    }
}

/// What [`infer_map`] found for one cuEST AO.
#[derive(Debug, Clone, Serialize)]
pub struct InferredAo {
    pub cuest_ao: usize,
    pub shell: usize,
    pub l: i32,
    /// Best-matching ferric component (index within the shell).
    pub ferric_component: usize,
    /// Least-squares scale c ≈ scale · f. ±1 is a pure permutation/phase;
    /// anything else is a normalization disagreement.
    pub scale: f64,
    /// Relative residual of the best match.
    pub residual: f64,
    /// Relative residual of the second-best candidate (ambiguity check).
    pub runner_up_residual: f64,
}

/// Recover the cuEST→ferric AO map from matrices computed by both codes.
///
/// For each cuEST AO, its "fingerprint" is its row of each matrix restricted
/// to the columns of s shells (l = 0 has one component, so those columns are
/// order-independent on both sides). The fingerprint is compared against every
/// ferric component of the same shell; the best least-squares match gives the
/// component and scale. Use a molecule in a generic orientation with several
/// atoms so no fingerprint is identically zero.
pub fn infer_map(
    layout: &AoLayout,
    cuest: &[&Array2<f64>],
    ferric: &[&Array2<f64>],
) -> Vec<InferredAo> {
    assert_eq!(cuest.len(), ferric.len());
    let s_cols: Vec<usize> = layout
        .shells
        .iter()
        .filter(|sh| sh.l == 0)
        .map(|sh| sh.offset)
        .collect();
    let fingerprint = |mats: &[&Array2<f64>], row: usize| -> Vec<f64> {
        mats.iter()
            .flat_map(|m| s_cols.iter().map(move |&c| m[[row, c]]))
            .collect()
    };
    let mut out = Vec::new();
    for (si, sh) in layout.shells.iter().enumerate() {
        let n = sh.nfunc();
        let f_prints: Vec<Vec<f64>> = (0..n).map(|j| fingerprint(ferric, sh.offset + j)).collect();
        for k in 0..n {
            let c = fingerprint(cuest, sh.offset + k);
            let cc: f64 = c.iter().map(|x| x * x).sum();
            let mut ranked: Vec<(usize, f64, f64)> = f_prints
                .iter()
                .enumerate()
                .map(|(j, f)| {
                    let ff: f64 = f.iter().map(|x| x * x).sum();
                    let cf: f64 = c.iter().zip(f).map(|(a, b)| a * b).sum();
                    let scale = if ff > 0.0 { cf / ff } else { 0.0 };
                    let r2: f64 = c.iter().zip(f).map(|(a, b)| (a - scale * b).powi(2)).sum();
                    let rel = if cc > 0.0 { (r2 / cc).sqrt() } else { f64::NAN };
                    (j, scale, rel)
                })
                .collect();
            ranked.sort_by(|a, b| a.2.total_cmp(&b.2));
            out.push(InferredAo {
                cuest_ao: sh.offset + k,
                shell: si,
                l: sh.l,
                ferric_component: ranked[0].0,
                scale: ranked[0].1,
                residual: ranked[0].2,
                runner_up_residual: ranked.get(1).map_or(f64::INFINITY, |r| r.2),
            });
        }
    }
    out
}

/// Per shell class, the component map the data supports, if every shell of
/// that class agrees and every match is clean (residual < `tol`, scale ±1
/// within `tol`, runner-up clearly worse).
#[derive(Debug, Clone, Serialize)]
pub struct ClassVerdict {
    pub l: i32,
    pub cuest_pure: bool,
    pub ferric_pure: bool,
    /// The agreed map, or `None` with the reason in `problem`.
    pub components: Option<Vec<(usize, f64)>>,
    pub hypothesis: Vec<(usize, f64)>,
    pub matches_hypothesis: bool,
    pub problem: Option<String>,
}

pub fn classify(layout: &AoLayout, inferred: &[InferredAo], tol: f64) -> Vec<ClassVerdict> {
    let mut classes: Vec<(i32, bool, bool)> = layout
        .shells
        .iter()
        .map(|s| (s.l, s.cuest_pure, s.ferric_pure))
        .collect();
    classes.sort();
    classes.dedup();
    classes
        .into_iter()
        .map(|(l, cp, fp)| {
            let hypothesis = hypothesis_components(l, cp, fp);
            let mut agreed: Option<Vec<(usize, f64)>> = None;
            let mut problem = None;
            for (si, sh) in layout.shells.iter().enumerate() {
                if (sh.l, sh.cuest_pure, sh.ferric_pure) != (l, cp, fp) {
                    continue;
                }
                let rows: Vec<&InferredAo> = inferred.iter().filter(|r| r.shell == si).collect();
                if let Some(bad) = rows.iter().find(|r| {
                    !(r.residual < tol && (r.scale.abs() - 1.0).abs() < tol && r.runner_up_residual > 1e3 * tol.max(r.residual))
                }) {
                    problem.get_or_insert(format!(
                        "shell {si} (atom {}, l={l}) AO {}: best ferric component {} with scale {:.6e}, \
                         residual {:.3e}, runner-up {:.3e}",
                        sh.atom, bad.cuest_ao, bad.ferric_component, bad.scale, bad.residual, bad.runner_up_residual
                    ));
                    continue;
                }
                let comp: Vec<(usize, f64)> = rows.iter().map(|r| (r.ferric_component, r.scale.signum())).collect();
                if let Some(why) = not_a_permutation(&comp, sh.nfunc()) {
                    problem.get_or_insert(format!("shell {si} (atom {}, l={l}): {why}: {comp:?}", sh.atom));
                    continue;
                }
                match &agreed {
                    None => agreed = Some(comp),
                    Some(a) if *a != comp => {
                        problem.get_or_insert(format!("shells of class l={l} disagree: {a:?} vs {comp:?}"));
                    }
                    _ => {}
                }
            }
            let components = if problem.is_none() { agreed } else { None };
            let matches_hypothesis = components.as_ref() == Some(&hypothesis);
            ClassVerdict {
                l,
                cuest_pure: cp,
                ferric_pure: fp,
                components,
                hypothesis,
                matches_hypothesis,
                problem,
            }
        })
        .collect()
}

/// `None` if `comp` maps the `n` components of a shell one-to-one onto
/// `0..n`, otherwise why not. Two cuEST components can each match the same
/// ferric component cleanly (a duplicated component); `to_ferric` would then
/// overwrite one and leave another zero.
fn not_a_permutation(comp: &[(usize, f64)], n: usize) -> Option<String> {
    if comp.len() != n {
        return Some(format!("{} components for a shell of {n}", comp.len()));
    }
    let mut seen = vec![false; n];
    for &(j, _) in comp {
        if j >= n {
            return Some(format!("component {j} out of range for a shell of {n}"));
        }
        if std::mem::replace(&mut seen[j], true) {
            return Some(format!("ferric component {j} matched twice"));
        }
    }
    None
}

/// Largest |a - b| over all elements; +infinity if any difference is not
/// finite (`f64::max` would otherwise drop a NaN and report agreement).
pub fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter().zip(b.iter()).fold(0.0, |acc, (x, y)| {
        let d = (x - y).abs();
        if d.is_finite() {
            acc.max(d)
        } else {
            f64::INFINITY
        }
    })
}
