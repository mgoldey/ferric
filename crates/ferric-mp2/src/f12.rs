//! F12 / CABS construction.
//!
//! Builds the Complementary Auxiliary Basis Set (CABS): the part of a large RI
//! basis (RIBS) that is *orthogonal* to the orbital basis (OBS). This is the
//! one-particle space in which F12's geminal-generated pair functions are
//! resolved without double-counting what OBS already spans — the device that
//! lets F12 recover the Coulomb cusp instead of just re-expressing OBS.
//!
//! Standard Valeev (2004) recipe:
//!   1. S_OO = ⟨OBS|OBS⟩,  S_OR = ⟨OBS|RIBS⟩,  S_RR = ⟨RIBS|RIBS⟩
//!   2. Projector onto OBS, in the RIBS metric:  P = S_OR^T S_OO^{-1} S_OR
//!   3. Complement:  Q = S_RR − P   ( = ⟨RIBS|(1−P_OBS)|RIBS⟩ )
//!   4. Canonical-orthogonalize Q, dropping eigenvalues < δ (the OBS-spanned,
//!      now-null directions). Survivors, scaled by λ^{-1/2}, are the CABS
//!      vectors expressed in the RIBS AO basis:  C_cabs (nri × ncabs).
//!
//! The RIBS coefficients of a CABS vector k are C_cabs[:,k]; ⟨CABS_k|CABS_l⟩ in
//! the RIBS metric S_RR is δ_kl, and ⟨CABS_k|OBS_p⟩ = 0 by construction.

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};

/// Eigenvalues of the complement metric below this are treated as OBS-spanned
/// (redundant) directions and dropped from CABS.
const CABS_NULL_THRESH: f64 = 1e-8;

/// The complementary auxiliary basis, expressed in the RIBS AO basis.
#[derive(Debug, Clone)]
pub struct Cabs {
    /// C_cabs, shape (nri, ncabs): RIBS-AO coefficients of each CABS function.
    pub coeffs: Array2<f64>,
    /// RIBS metric S_RR, shape (nri, nri) — kept for downstream contractions.
    pub s_rr: Array2<f64>,
    /// Cross overlap S_OR, shape (nobs, nri).
    pub s_or: Array2<f64>,
    pub nobs: usize,
    pub nri: usize,
    /// Number of retained CABS functions (≈ nri − nobs).
    pub ncabs: usize,
}

/// Form the union basis OBS ∪ aux: per element, OBS shells followed by aux
/// shells. Used to build the OBS-inclusive F12 RIBS.
fn union_basis_sets(obs: &BasisSet, aux: &BasisSet) -> BasisSet {
    use std::collections::HashMap;
    let mut shells: HashMap<i32, Vec<_>> = HashMap::new();
    for (&z, obs_sh) in &obs.shells {
        shells.entry(z).or_default().extend(obs_sh.iter().cloned());
    }
    for (&z, aux_sh) in &aux.shells {
        shells.entry(z).or_default().extend(aux_sh.iter().cloned());
    }
    BasisSet {
        name: format!("{}∪{}", obs.name, aux.name),
        shells,
        ecps: HashMap::new(),
    }
}

/// Symmetric inverse via eigendecomposition (S_OO is small, SPD).
fn sym_inverse(a: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let (evals, evecs) = a
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::General(format!("S_OO eigh failed: {e}")))?;
    let n = evals.len();
    let mut scaled = evecs.clone();
    for i in 0..n {
        if evals[i] <= 0.0 {
            return Err(FerricError::General(format!(
                "S_OO not positive-definite (eval {i} = {})",
                evals[i]
            )));
        }
        let inv = 1.0 / evals[i];
        for mu in 0..n {
            scaled[(mu, i)] *= inv;
        }
    }
    Ok(scaled.dot(&evecs.t()))
}

/// Construct CABS from an orbital basis (OBS) and an auxiliary set (`aux`) on a
/// shared molecule.
///
/// The F12 RI basis (RIBS) is formed INTERNALLY as the union RIBS = OBS ∪ aux,
/// so OBS ⊆ RIBS by construction — the F12 convention. Passing a bare DF/RI-fit
/// set as `aux` is fine: it does not need to contain OBS itself (the union
/// guarantees inclusion). The complement Q = S_RR − P_OBS then has the OBS span
/// projected out; its near-null eigenvalues (the OBS-spanned directions PLUS any
/// linear dependencies introduced by the union) are dropped at
/// `CABS_NULL_THRESH`, leaving ncabs ≈ n(aux) genuinely-complementary functions.
pub fn build_cabs(
    mol: &Molecule,
    obs: &BasisSet,
    aux: &BasisSet,
) -> Result<Cabs, FerricError> {
    // RIBS = OBS ∪ aux (union, OBS-inclusive). Functions are atom-major:
    // per atom, OBS shells then aux shells — so OBS/aux functions interleave
    // and are NOT contiguous. We partition by provenance mask, then reorder
    // the union overlap into [OBS-block | aux-block] so we can Schur-complement
    // the aux block exactly as Valeev (PySCF mp2f12_slow.find_cabs).
    let ribs = union_basis_sets(obs, aux);
    let ri_prep = PreparedBasis::new(mol, &ribs)?;
    let s_union = oneelectron::overlap(&ri_prep); // (nri, nri), union AO order

    let mask = union_obs_mask(mol, obs, aux); // true = OBS-origin function
    debug_assert_eq!(mask.len(), ri_prep.nbasis());
    let obs_idx: Vec<usize> = mask.iter().enumerate().filter(|(_, &m)| m).map(|(i, _)| i).collect();
    let aux_idx: Vec<usize> = mask.iter().enumerate().filter(|(_, &m)| !m).map(|(i, _)| i).collect();
    let nobs = obs_idx.len();
    let naux = aux_idx.len();
    let nri = nobs + naux;

    // Slice the reordered overlap blocks (RIBS AO order = [obs_idx, aux_idx]).
    let s_oo = take_block(&s_union, &obs_idx, &obs_idx); // (nobs, nobs)
    let s_oa = take_block(&s_union, &obs_idx, &aux_idx); // (nobs, naux)
    let s_aa = take_block(&s_union, &aux_idx, &aux_idx); // (naux, naux)

    // ls12 = S_OO^{-1} S_Oa  (nobs, naux).
    let s_oo_inv = sym_inverse(&s_oo)?;
    let ls12 = s_oo_inv.dot(&s_oa);

    // Schur complement on the aux block: Q = S_aa − S_aO S_OO^{-1} S_Oa.
    // This is the overlap of the aux functions PROJECTED orthogonal to OBS.
    let q = &s_aa - &s_oa.t().dot(&ls12); // (naux, naux)

    let (q_evals, q_evecs) = q
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::General(format!("CABS Schur-complement eigh failed: {e}")))?;

    // Keep eigenvalues above lindep; these directions are genuinely outside OBS.
    let keep: Vec<usize> = (0..naux).filter(|&i| q_evals[i] > CABS_NULL_THRESH).collect();
    let ncabs = keep.len();

    // c2 = aux-block coeffs = v / sqrt(w); c1 = OBS-block coeffs = ls12 · c2.
    // CABS vector in RIBS AO order is [-c1 ; c2] (OBS rows then aux rows), then
    // scattered back to the interleaved union AO ordering via obs_idx / aux_idx.
    let mut c2 = Array2::zeros((naux, ncabs));
    for (col, &i) in keep.iter().enumerate() {
        let inv_sqrt = 1.0 / q_evals[i].sqrt();
        for mu in 0..naux {
            c2[(mu, col)] = q_evecs[(mu, i)] * inv_sqrt;
        }
    }
    let c1 = ls12.dot(&c2); // (nobs, ncabs)

    let mut coeffs = Array2::zeros((nri, ncabs)); // union AO order
    for col in 0..ncabs {
        for (a, &gi) in obs_idx.iter().enumerate() {
            coeffs[(gi, col)] = -c1[(a, col)];
        }
        for (b, &gi) in aux_idx.iter().enumerate() {
            coeffs[(gi, col)] = c2[(b, col)];
        }
    }

    // S_OR = ⟨OBS | RIBS⟩ for downstream use: OBS rows of the union overlap.
    let s_or = take_block(&s_union, &obs_idx, &(0..nri).collect::<Vec<_>>());

    Ok(Cabs { coeffs, s_rr: s_union, s_or, nobs, nri, ncabs })
}

/// Extract A[rows, cols] into a fresh (|rows| × |cols|) matrix.
fn take_block(a: &Array2<f64>, rows: &[usize], cols: &[usize]) -> Array2<f64> {
    let mut out = Array2::zeros((rows.len(), cols.len()));
    for (r, &i) in rows.iter().enumerate() {
        for (c, &j) in cols.iter().enumerate() {
            out[(r, c)] = a[(i, j)];
        }
    }
    out
}

/// Provenance mask over the union RIBS (OBS∪aux): true if the global union
/// basis function originates from OBS, false if from aux. Atom-major: per atom,
/// OBS shells (true) then aux shells (false).
fn union_obs_mask(mol: &Molecule, obs: &BasisSet, aux: &BasisSet) -> Vec<bool> {
    use ferric_core::basis::num_functions;
    let mut mask = Vec::new();
    for atom in &mol.atoms {
        if let Some(sh) = obs.for_element(atom.z) {
            let n: usize = sh.iter().map(|s| num_functions(s.l, s.pure)).sum();
            mask.resize(mask.len() + n, true);
        }
        if let Some(sh) = aux.for_element(atom.z) {
            let n: usize = sh.iter().map(|s| num_functions(s.l, s.pure)).sum();
            mask.resize(mask.len() + n, false);
        }
    }
    mask
}

impl Cabs {
    /// ⟨CABS|CABS⟩ in the RIBS metric: C^T S_RR C, should be I_{ncabs}.
    pub fn gram_cabs(&self) -> Array2<f64> {
        self.coeffs.t().dot(&self.s_rr).dot(&self.coeffs)
    }

    /// ⟨OBS|CABS⟩ = S_OR · C_cabs, should be the (nobs × ncabs) zero matrix.
    pub fn overlap_obs_cabs(&self) -> Array2<f64> {
        self.s_or.dot(&self.coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    fn max_abs(a: &Array2<f64>) -> f64 {
        a.iter().fold(0.0_f64, |m, &v| m.max(v.abs()))
    }

    fn off_unit_dev(a: &Array2<f64>) -> f64 {
        // max |A - I| for a square matrix
        let mut d = 0.0_f64;
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                let target = if i == j { 1.0 } else { 0.0 };
                d = d.max((a[(i, j)] - target).abs());
            }
        }
        d
    }

    #[test]
    fn cabs_is_orthonormal_and_complementary() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs = basis::bundled("cc-pvdz").unwrap();
        let ri = basis::bundled("cc-pvdz-ri").unwrap();

        let cabs = build_cabs(&mol, &obs, &ri).unwrap();

        // (1) CABS orthonormal in the RIBS metric: C^T S_RR C = I.
        let gram = cabs.gram_cabs();
        assert_eq!(gram.shape(), &[cabs.ncabs, cabs.ncabs]);
        let gdev = off_unit_dev(&gram);
        assert!(gdev < 1e-8, "‖C^T S_RR C − I‖∞ = {gdev:.3e}");

        // (2) CABS orthogonal to OBS: S_OR · C = 0.
        let cross = cabs.overlap_obs_cabs();
        assert_eq!(cross.shape(), &[cabs.nobs, cabs.ncabs]);
        let cdev = max_abs(&cross);
        assert!(cdev < 1e-8, "‖⟨OBS|CABS⟩‖∞ = {cdev:.3e}");
    }

    #[test]
    fn cabs_dimension_is_complement() {
        // RIBS = OBS ∪ aux is built internally, so OBS ⊆ RIBS and nri > nobs.
        // The complement projects OBS out, so ncabs = nri − rank(OBS-in-RIBS).
        // With OBS fully contained, ncabs ≈ nri − nobs. We assert the complement
        // is strictly smaller than RIBS (OBS span was removed) and that roughly
        // nobs directions were dropped.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs = basis::bundled("cc-pvdz").unwrap();
        let aux = basis::bundled("cc-pvdz-ri").unwrap();

        let nobs = PreparedBasis::new(&mol, &obs).unwrap().nbasis();

        let cabs = build_cabs(&mol, &obs, &aux).unwrap();
        assert!(cabs.ncabs > 0, "CABS empty");
        assert!(
            cabs.ncabs < cabs.nri,
            "complement dropped no directions (ncabs {} == nri {})",
            cabs.ncabs,
            cabs.nri
        );
        // OBS is fully contained in RIBS, so exactly ~nobs directions drop.
        let dropped = cabs.nri - cabs.ncabs;
        assert!(
            dropped >= nobs,
            "expected ≥ nobs={nobs} directions dropped, got {dropped} (nri={}, ncabs={})",
            cabs.nri,
            cabs.ncabs
        );
    }

    #[test]
    fn cabs_self_complement_is_empty() {
        // RIBS = OBS: the complement of OBS within OBS is empty (every direction
        // is OBS-spanned, so every Q eigenvalue is ~0 and gets dropped).
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs = basis::bundled("sto-3g").unwrap();

        let cabs = build_cabs(&mol, &obs, &obs).unwrap();
        assert_eq!(
            cabs.ncabs, 0,
            "self-complement should be empty, got ncabs = {}",
            cabs.ncabs
        );
    }
}
