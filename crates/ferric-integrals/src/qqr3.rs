//! QQR-style distance-aware screening bounds for 3-index integrals (P|μν).
//!
//! Refines the 3-index Schwarz bound  `|(P|μν)| ≤ Q3[P] · Q(μ,ν)`  with the
//! same distance/extent envelope used by [`crate::schwarz`] + [`ferric-scf`]'s
//! 4-center [`QqrBounds`]:
//!
//! ```text
//!   |(P|μν)|  ≤  Q3[P] · Q(μ,ν) · min(1, ext_aux[P]·ext_obs(μν)/R) · op_decay(R)
//! ```
//!
//! where R is the distance from the obs-pair center to the aux-shell center,
//! and `op_decay(R) = exp(-ω²R²)` for `ErfcCoulomb` (same form as the existing
//! 4-center QQR). This is the bound that lets erfc-attenuated MP2 actually
//! see speedup from attenuation: the basic 3-index Schwarz bound is bra-only
//! and has no notion of bra-ket distance, so it cannot capture erfc decay
//! (verified empirically on decane — Schwarz alone drops 0 additional triples
//! at production omega).

use crate::basis_bridge::PreparedBasis;
use crate::operator::{Operator, OperatorKind};
use crate::schwarz::{schwarz, schwarz3_aux};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ndarray::Array2;

/// Distance-aware screening for 3-index ERIs.
///
/// Holds per-aux-shell self-Schwarz and pair-center/extent, plus per-obs-pair
/// Schwarz and pair-center/extent. `estimate3(P, s1, s2)` returns the QQR bound.
pub struct QqrBounds3 {
    /// Per-aux-shell Schwarz: Q3[P] = sqrt(max_a |(P_a|P_a)|).
    q3: Vec<f64>,
    /// Aux-shell self-pair extents:  ext_aux[P] = 1/sqrt(2·α_min(P)).
    aux_extents: Vec<f64>,
    /// Aux-shell centers (= atom centers of the aux shell).
    aux_centers: Vec<[f64; 3]>,

    /// Obs-pair Schwarz matrix, Q(s1,s2) = sqrt(|(s1,s2|s1,s2)|).
    q_obs: Array2<f64>,
    /// Obs-pair extents and centers, indexed `i*nsh_obs + j`.
    obs_pair_extents: Vec<f64>,
    obs_pair_centers: Vec<[f64; 3]>,

    op: Operator,
    nsh_obs: usize,
    nsh_aux: usize,
}

impl QqrBounds3 {
    /// Build the QQR-3 bounds for an obs/aux basis pair under operator `op`.
    pub fn new(
        op: Operator,
        mol: &Molecule,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
    ) -> Result<Self, FerricError> {
        let q3 = schwarz3_aux(op, aux)?;
        let q_obs = schwarz(op, obs)?;

        // Per-shell minimum exponent and atom center for each basis, mirroring
        // the construction in ferric_scf::qqr::QqrBounds::new. Basis sets are
        // recovered from the PreparedBasis (which retains a reference to them).
        let (obs_min, obs_origin) = collect_min_exponents_and_origins(mol, obs.basis_set());
        let (aux_min, aux_origin) = collect_min_exponents_and_origins(mol, aux.basis_set());
        assert_eq!(obs_min.len(), obs.nshells());
        assert_eq!(aux_min.len(), aux.nshells());
        let nsh_obs = obs.nshells();
        let nsh_aux = aux.nshells();

        // Obs pair centers/extents: same definition as 4-center QQR.
        let mut obs_pair_centers = vec![[0.0; 3]; nsh_obs * nsh_obs];
        let mut obs_pair_extents = vec![0.0; nsh_obs * nsh_obs];
        for i in 0..nsh_obs {
            let ai = obs_min[i];
            let ri = obs_origin[i];
            for j in 0..nsh_obs {
                let aj = obs_min[j];
                let rj = obs_origin[j];
                let sum = ai + aj;
                let idx = i * nsh_obs + j;
                obs_pair_centers[idx] = [
                    (ai * ri[0] + aj * rj[0]) / sum,
                    (ai * ri[1] + aj * rj[1]) / sum,
                    (ai * ri[2] + aj * rj[2]) / sum,
                ];
                obs_pair_extents[idx] = 1.0 / sum.sqrt();
            }
        }

        // Aux self-pair extents/centers (P with itself).
        let mut aux_extents = vec![0.0; nsh_aux];
        let mut aux_centers = vec![[0.0; 3]; nsh_aux];
        for p in 0..nsh_aux {
            let ap = aux_min[p];
            aux_centers[p] = aux_origin[p];
            aux_extents[p] = 1.0 / (2.0 * ap).sqrt();
        }

        Ok(QqrBounds3 {
            q3,
            aux_extents,
            aux_centers,
            q_obs,
            obs_pair_extents,
            obs_pair_centers,
            op,
            nsh_obs,
            nsh_aux,
        })
    }

    pub fn nsh_obs(&self) -> usize { self.nsh_obs }
    pub fn nsh_aux(&self) -> usize { self.nsh_aux }
    pub fn op(&self) -> Operator { self.op }

    /// Upper bound estimate for |(P | s1 s2)|.
    pub fn estimate3(&self, p: usize, s1: usize, s2: usize) -> f64 {
        let schwarz_est = self.q3[p] * self.q_obs[(s1, s2)];
        if schwarz_est == 0.0 {
            return 0.0;
        }
        let idx_obs = s1 * self.nsh_obs + s2;
        let c_obs = self.obs_pair_centers[idx_obs];
        let c_aux = self.aux_centers[p];
        let dx = c_obs[0] - c_aux[0];
        let dy = c_obs[1] - c_aux[1];
        let dz = c_obs[2] - c_aux[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r < 1e-14 {
            return schwarz_est;
        }
        let ext_prod = self.obs_pair_extents[idx_obs] * self.aux_extents[p];
        let mut decay = (ext_prod / r).min(1.0);
        if let OperatorKind::ErfcCoulomb = self.op.kind {
            let omega = self.op.omega;
            decay *= (-omega * omega * r * r).exp();
        }
        schwarz_est * decay
    }
}

fn collect_min_exponents_and_origins(
    mol: &Molecule,
    bs: &BasisSet,
) -> (Vec<f64>, Vec<[f64; 3]>) {
    let mut min_exps = Vec::new();
    let mut origins = Vec::new();
    for atom in &mol.atoms {
        let shells = bs.for_element(atom.z).unwrap();
        for sh in shells {
            let alpha_min = sh.exponents.iter().cloned().fold(f64::INFINITY, f64::min);
            min_exps.push(alpha_min);
            origins.push([atom.x, atom.y, atom.zpos]);
        }
    }
    (min_exps, origins)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    #[test]
    fn test_qqr3_le_schwarz3() {
        // QQR-3 estimate must be ≤ the Schwarz-3 estimate (Q3·Q_obs) for every
        // shell triple — the distance/operator factor is always ≤ 1.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::erfc(0.222);
        let qqr3 = QqrBounds3::new(op, &mol, &obs, &aux).unwrap();
        for p in 0..qqr3.nsh_aux() {
            for s1 in 0..qqr3.nsh_obs() {
                for s2 in 0..qqr3.nsh_obs() {
                    let schwarz3 = qqr3.q3[p] * qqr3.q_obs[(s1, s2)];
                    let q = qqr3.estimate3(p, s1, s2);
                    assert!(
                        q <= schwarz3 + 1e-14,
                        "QQR3({p},{s1},{s2}) = {q} > Schwarz3 = {schwarz3}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_qqr3_erfc_tighter_than_qqr3_coulomb() {
        // Same Schwarz factors, but ErfcCoulomb adds the exp(-ω²R²) decay,
        // so erfc-QQR3 must be ≤ Coulomb-QQR3 at every triple, strictly
        // smaller for some distant triple.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let qqr3_c = QqrBounds3::new(
            Operator::coulomb(), &mol, &obs, &aux).unwrap();
        let qqr3_e = QqrBounds3::new(
            Operator::erfc(0.5), &mol, &obs, &aux).unwrap();
        let mut found_strictly_smaller = false;
        for p in 0..qqr3_c.nsh_aux() {
            for s1 in 0..qqr3_c.nsh_obs() {
                for s2 in 0..qqr3_c.nsh_obs() {
                    let c = qqr3_c.estimate3(p, s1, s2);
                    let e = qqr3_e.estimate3(p, s1, s2);
                    // Note: Q3 and Q_obs differ per operator (libint computes
                    // erfc Schwarz factors that are themselves smaller), so we
                    // only check the distance envelope on a normalized ratio
                    // when both are nonzero. The strict-smaller condition is
                    // about the *combined* bound being tighter.
                    assert!(e <= c + 1e-14,
                        "erfc QQR3 not ≤ Coulomb QQR3 at ({p},{s1},{s2}): {e} vs {c}");
                    if c > 1e-10 && (e / c) < 0.99 {
                        found_strictly_smaller = true;
                    }
                }
            }
        }
        assert!(found_strictly_smaller,
            "erfc QQR3 should be strictly smaller than Coulomb QQR3 somewhere");
    }
}
