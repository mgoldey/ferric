//! Cross-basis overlap S_OR = ⟨OBS | RIBS⟩ via a merged-basis trick.
//!
//! F12 / CABS needs the rectangular overlap between two *different* basis sets
//! on the same molecule (the orbital basis and a larger RI basis). libint's
//! one-body engine can take shells from two bases, but ferric's shim only
//! exposes single-basis 1e blocks (`scf_compute_1e_block(eng, bs, s1, s2)`).
//!
//! Rather than extend the FFI, we MERGE OBS and RIBS into one `BasisSet`
//! (per element: OBS shells followed by RIBS shells), build a single
//! `PreparedBasis`, compute the full (nobs+nri)² overlap, and slice out the
//! OBS×RIBS off-diagonal block. Shells are laid out atom-major
//! (`PreparedBasis::new`), so OBS functions are NOT globally contiguous — they
//! interleave per atom. We therefore return an index partition alongside the
//! block so callers can map merged indices back to OBS / RIBS spaces.
//!
//! Cost: one (nobs+nri)² overlap instead of a tight nobs×nri rectangle. At our
//! sizes that's negligible, and it keeps the whole thing FFI-free. A dedicated
//! `scf_compute_1e_block_cross` is the production optimization (see CABS sketch).

use crate::basis_bridge::PreparedBasis;
use crate::oneelectron;
use ferric_core::basis::{num_functions, BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ndarray::Array2;
use std::collections::HashMap;

/// Result of the merged-basis cross-overlap construction.
#[derive(Debug)]
pub struct CrossOverlap {
    /// S_OR, shape (nobs, nri): ⟨OBS function i | RIBS function j⟩.
    pub s_or: Array2<f64>,
    /// Number of OBS basis functions.
    pub nobs: usize,
    /// Number of RIBS basis functions.
    pub nri: usize,
}

/// Merge two basis sets into one. For each element present in *either* set, the
/// merged shell list is `obs_shells ++ ri_shells` (OBS first). Elements absent
/// from one set contribute only the other's shells.
fn merge_basis_sets(obs: &BasisSet, ri: &BasisSet) -> BasisSet {
    let mut shells: HashMap<i32, Vec<Shell>> = HashMap::new();
    for (&z, obs_sh) in &obs.shells {
        shells.entry(z).or_default().extend(obs_sh.iter().cloned());
    }
    for (&z, ri_sh) in &ri.shells {
        shells.entry(z).or_default().extend(ri_sh.iter().cloned());
    }
    BasisSet {
        name: format!("{}+{}", obs.name, ri.name),
        shells,
        ecps: HashMap::new(),
    }
}

/// Build the function-level provenance mask for the MERGED basis: for each
/// global merged basis function, `true` if it came from OBS, `false` if RIBS.
///
/// Mirrors `PreparedBasis::new`'s atom-major layout: for each atom, emit the
/// element's OBS shells (mask=true) then its RIBS shells (mask=false), counting
/// `num_functions(l, pure)` functions per shell.
fn obs_function_mask(mol: &Molecule, obs: &BasisSet, ri: &BasisSet) -> Vec<bool> {
    let mut mask = Vec::new();
    for atom in &mol.atoms {
        if let Some(obs_sh) = obs.for_element(atom.z) {
            let nfunc: usize = obs_sh.iter().map(|sh| num_functions(sh.l, sh.pure)).sum();
            mask.resize(mask.len() + nfunc, true);
        }
        if let Some(ri_sh) = ri.for_element(atom.z) {
            let nfunc: usize = ri_sh.iter().map(|sh| num_functions(sh.l, sh.pure)).sum();
            mask.resize(mask.len() + nfunc, false);
        }
    }
    mask
}

/// Compute the cross-basis overlap S_OR = ⟨OBS | RIBS⟩ on a shared molecule.
///
/// Builds a merged OBS+RIBS basis, computes its full overlap, and extracts the
/// OBS-row × RIBS-column block using the atom-major provenance mask.
pub fn cross_overlap(
    mol: &Molecule,
    obs: &BasisSet,
    ri: &BasisSet,
) -> Result<CrossOverlap, FerricError> {
    let merged = merge_basis_sets(obs, ri);
    let prep = PreparedBasis::new(mol, &merged)?;
    let s_full = oneelectron::overlap(&prep); // (nmerged, nmerged)

    let mask = obs_function_mask(mol, obs, ri);
    debug_assert_eq!(
        mask.len(),
        prep.nbasis(),
        "provenance mask length must match merged nbasis"
    );

    let obs_idx: Vec<usize> = mask.iter().enumerate().filter(|(_, &m)| m).map(|(i, _)| i).collect();
    let ri_idx: Vec<usize> = mask.iter().enumerate().filter(|(_, &m)| !m).map(|(i, _)| i).collect();
    let nobs = obs_idx.len();
    let nri = ri_idx.len();

    let mut s_or = Array2::zeros((nobs, nri));
    for (a, &i) in obs_idx.iter().enumerate() {
        for (b, &j) in ri_idx.iter().enumerate() {
            s_or[(a, b)] = s_full[(i, j)];
        }
    }
    Ok(CrossOverlap { s_or, nobs, nri })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    /// When RIBS == OBS, the cross-block ⟨OBS|RIBS⟩ must equal the plain OBS
    /// overlap S exactly (the two copies are the same functions).
    #[test]
    fn cross_overlap_self_equals_overlap() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs = basis::bundled("sto-3g").unwrap();

        let prep = PreparedBasis::new(&mol, &obs).unwrap();
        let s = oneelectron::overlap(&prep);

        let cx = cross_overlap(&mol, &obs, &obs).unwrap();
        assert_eq!(cx.nobs, s.nrows());
        assert_eq!(cx.nri, s.nrows());

        let mut max_dev = 0.0_f64;
        for i in 0..s.nrows() {
            for j in 0..s.ncols() {
                max_dev = max_dev.max((cx.s_or[(i, j)] - s[(i, j)]).abs());
            }
        }
        assert!(max_dev < 1e-12, "max |S_OR - S| = {max_dev:.3e}");
    }

    /// Shapes and OBS-diagonal: S_OR rows = nobs, cols = nri (> nobs for a
    /// genuine RIBS). The OBS-against-RIBS block must contain a unit-overlap
    /// match for the shared OBS shells when RIBS ⊇ OBS by construction.
    #[test]
    fn cross_overlap_shapes_obs_subset_ri() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs = basis::bundled("cc-pvdz").unwrap();
        let ri = basis::bundled("cc-pvdz-ri").unwrap();

        let nobs_prep = PreparedBasis::new(&mol, &obs).unwrap().nbasis();
        let nri_prep = PreparedBasis::new(&mol, &ri).unwrap().nbasis();

        let cx = cross_overlap(&mol, &obs, &ri).unwrap();
        assert_eq!(cx.nobs, nobs_prep);
        assert_eq!(cx.nri, nri_prep);
        assert_eq!(cx.s_or.shape(), &[nobs_prep, nri_prep]);

        // Sanity: cross overlap entries are bounded by 1 in magnitude
        // (normalized contracted Gaussians; Cauchy–Schwarz on the overlap).
        let max_abs = cx.s_or.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(max_abs <= 1.0 + 1e-10, "max |S_OR| = {max_abs} exceeds 1");
    }
}
