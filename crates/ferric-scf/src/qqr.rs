//! QQR distance-dependent integral screening bounds.
//!
//! Refines Schwarz estimates with inverse-distance decay for well-separated
//! shell pairs. Reference: Maurer, Lambrecht, Ochsenfeld, JCP 136, 144107 (2012).

use crate::screening::{Bound, SchwarzBounds};
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::{Operator, OperatorKind};

/// QQR (distance-dependent) integral screening bounds.
///
/// For each shell pair (i,j), stores:
/// - **pair center**: weighted average of shell origins, using the most diffuse exponent
/// - **pair extent**: spatial width `1/sqrt(alpha_i_min + alpha_j_min)`
///
/// The bound is: `schwarz(i,j,k,l) * min(1, extent_ij * extent_kl / R_ij_kl) * op_decay(R)`
/// where `R_ij_kl` is the distance between pair centers and `op_decay` provides
/// operator-specific exponential decay (e.g., `exp(-omega^2 * R^2)` for ErfcCoulomb).
#[derive(Debug, Clone)]
pub struct QqrBounds {
    schwarz: SchwarzBounds,
    /// Pair centers, indexed as `pair_centers[i * nshells + j]`.
    pair_centers: Vec<[f64; 3]>,
    /// Pair extents, indexed as `pair_extents[i * nshells + j]`.
    pair_extents: Vec<f64>,
    /// The two-electron operator, used for operator-aware decay.
    op: Operator,
    nshells: usize,
}

impl QqrBounds {
    /// Compute QQR bounds from an existing Schwarz bound, molecule, basis, and prepared basis.
    ///
    /// Needs the raw `BasisSet` to access per-shell exponents (the minimum exponent of
    /// each shell determines the pair center and extent). The `Molecule` and `PreparedBasis`
    /// provide atom coordinates and the shell-to-atom mapping.
    /// Infallible wrapper over [`Self::try_new`], for call sites that have
    /// already established the system fits (the tests in this crate). Panics
    /// with the budget breakdown if the dense pair tables do not fit — prefer
    /// [`Self::try_new`] anywhere the size is not known in advance.
    pub fn new(
        schwarz: SchwarzBounds,
        mol: &Molecule,
        bs: &BasisSet,
        prep: &PreparedBasis,
        op: Operator,
    ) -> Self {
        Self::try_new(schwarz, mol, bs, prep, op)
            .unwrap_or_else(|e| panic!("QqrBounds::new: {e}"))
    }

    /// Fallible constructor: refuses up front when the dense `nshells²` pair
    /// tables would not fit the memory budget (defect E).
    ///
    /// The tables are DENSE over ordered shell pairs by design — the screening
    /// predicate's value comes from an O(1) `i * nshells + j` lookup, and that
    /// layout is deliberately NOT changed here. What was missing is any check:
    /// `vec![[0.0; 3]; nsh * nsh]` and `vec![0.0; nsh * nsh]` were allocated
    /// with no reference to the budget, so an oversized system met the OOM
    /// killer instead of an error naming the term.
    ///
    /// Per ordered pair: `[f64; 3]` center (24 B) + `f64` extent (8 B) = 32 B.
    pub fn try_new(
        schwarz: SchwarzBounds,
        mol: &Molecule,
        bs: &BasisSet,
        prep: &PreparedBasis,
        op: Operator,
    ) -> Result<Self, ferric_core::FerricError> {
        let nsh = prep.nshells();
        let pair_bytes = nsh
            .saturating_mul(nsh)
            .saturating_mul(std::mem::size_of::<[f64; 3]>() + std::mem::size_of::<f64>());
        ferric_core::memory::check_alloc(
            &format!("QQR pair tables (nshells={nsh} ordered pairs, dense by design)"),
            pair_bytes,
            ferric_core::memory::resolve_budget_bytes(None),
        )?;

        // Collect the minimum exponent per shell and the atom coordinates per shell.
        // We iterate atoms in order, collecting shells per atom, mirroring PreparedBasis::new.
        let mut min_exponents: Vec<f64> = Vec::with_capacity(nsh);
        let mut shell_origins: Vec<[f64; 3]> = Vec::with_capacity(nsh);

        for atom in &mol.atoms {
            let shells = bs.for_element(atom.z).unwrap();
            for sh in shells {
                let alpha_min = sh.exponents.iter().cloned().fold(f64::INFINITY, f64::min);
                min_exponents.push(alpha_min);
                shell_origins.push([atom.x, atom.y, atom.zpos]);
            }
        }
        assert_eq!(min_exponents.len(), nsh, "shell count mismatch");

        // Build pair centers and extents.
        let mut pair_centers = vec![[0.0; 3]; nsh * nsh];
        let mut pair_extents = vec![0.0; nsh * nsh];

        for i in 0..nsh {
            let ai = min_exponents[i];
            let ri = shell_origins[i];
            for j in 0..nsh {
                let aj = min_exponents[j];
                let rj = shell_origins[j];
                let sum_alpha = ai + aj;
                let idx = i * nsh + j;

                // Weighted pair center: R_ij = (alpha_i * R_i + alpha_j * R_j) / (alpha_i + alpha_j)
                pair_centers[idx] = [
                    (ai * ri[0] + aj * rj[0]) / sum_alpha,
                    (ai * ri[1] + aj * rj[1]) / sum_alpha,
                    (ai * ri[2] + aj * rj[2]) / sum_alpha,
                ];
                // Pair extent: epsilon_ij = 1 / sqrt(alpha_i + alpha_j)
                pair_extents[idx] = 1.0 / sum_alpha.sqrt();
            }
        }

        Ok(QqrBounds {
            schwarz,
            pair_centers,
            pair_extents,
            op,
            nshells: nsh,
        })
    }

    /// Access the underlying Schwarz bounds.
    pub fn schwarz(&self) -> &SchwarzBounds {
        &self.schwarz
    }

    /// The operator associated with these bounds.
    pub fn op(&self) -> Operator {
        self.op
    }

    /// Number of shells.
    pub fn nshells(&self) -> usize {
        self.nshells
    }

    /// Pair center for shell pair (i,j).
    pub fn pair_center(&self, i: usize, j: usize) -> [f64; 3] {
        self.pair_centers[i * self.nshells + j]
    }

    /// Pair extent for shell pair (i,j).
    pub fn pair_extent(&self, i: usize, j: usize) -> f64 {
        self.pair_extents[i * self.nshells + j]
    }
}

impl Bound for QqrBounds {
    fn estimate(&self, sh1: usize, sh2: usize, sh3: usize, sh4: usize) -> f64 {
        let schwarz_est = self.schwarz.estimate(sh1, sh2, sh3, sh4);
        let nsh = self.nshells;
        let idx_bra = sh1 * nsh + sh2;
        let idx_ket = sh3 * nsh + sh4;
        let c_bra = &self.pair_centers[idx_bra];
        let c_ket = &self.pair_centers[idx_ket];

        let dx = c_bra[0] - c_ket[0];
        let dy = c_bra[1] - c_ket[1];
        let dz = c_bra[2] - c_ket[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();

        if r < 1e-14 {
            // Overlapping pair centers: QQR reduces to Schwarz.
            return schwarz_est;
        }

        let extent_product = self.pair_extents[idx_bra] * self.pair_extents[idx_ket];
        let mut decay = (extent_product / r).min(1.0);

        // Operator-specific decay: ErfcCoulomb provides exponential decay at long range.
        // Coulomb / ErfCoulomb add no extra decay.
        if self.op.kind == OperatorKind::ErfcCoulomb {
            let omega = self.op.omega;
            decay *= (-omega * omega * r * r).exp();
        }

        schwarz_est * decay
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screening::{Bound, SchwarzBounds};
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;

    fn water_qqr() -> (QqrBounds, usize) {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let schwarz = SchwarzBounds::compute(op, &prep).unwrap();
        let nsh = prep.nshells();
        let qqr = QqrBounds::new(schwarz, &mol, &bs, &prep, op);
        (qqr, nsh)
    }

    #[test]
    fn test_qqr_le_schwarz() {
        // QQR estimate must be <= Schwarz estimate for all quartets.
        let (qqr, nsh) = water_qqr();
        for i in 0..nsh {
            for j in 0..nsh {
                for k in 0..nsh {
                    for l in 0..nsh {
                        let s = qqr.schwarz().estimate(i, j, k, l);
                        let q = qqr.estimate(i, j, k, l);
                        assert!(
                            q <= s + 1e-15,
                            "QQR({i},{j},{k},{l}) = {q} > Schwarz = {s}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_qqr_tighter_for_distant_pairs() {
        // For water STO-3G, oxygen shells (0,1,2) are on atom 0 and
        // hydrogen shells (3,4) are on atoms 1,2. Pairs crossing atoms
        // should have QQR strictly less than Schwarz.
        let (qqr, nsh) = water_qqr();
        let mut found_tighter = false;
        for i in 0..nsh {
            for j in 0..nsh {
                for k in 0..nsh {
                    for l in 0..nsh {
                        let s = qqr.schwarz().estimate(i, j, k, l);
                        let q = qqr.estimate(i, j, k, l);
                        if s > 1e-10 && (q / s) < 0.99 {
                            found_tighter = true;
                        }
                    }
                }
            }
        }
        assert!(found_tighter, "QQR should be strictly tighter than Schwarz for some distant pairs");
    }

    #[test]
    fn test_qqr_overlapping_equals_schwarz() {
        // Same shell on same atom: pair centers overlap, so QQR == Schwarz.
        let (qqr, _nsh) = water_qqr();
        let s = qqr.schwarz().estimate(0, 0, 0, 0);
        let q = qqr.estimate(0, 0, 0, 0);
        assert!(
            (q - s).abs() < 1e-15,
            "overlapping pairs: QQR={q} != Schwarz={s}"
        );
    }

    #[test]
    fn test_pair_center_self_pair() {
        // Self-pair center should be the atom center.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let schwarz = SchwarzBounds::compute(op, &prep).unwrap();
        let qqr = QqrBounds::new(schwarz, &mol, &bs, &prep, op);
        // Shell 0 is on atom 0 (oxygen)
        let c = qqr.pair_center(0, 0);
        let ox = mol.atoms[0].x;
        let oy = mol.atoms[0].y;
        let oz = mol.atoms[0].zpos;
        assert!((c[0] - ox).abs() < 1e-12);
        assert!((c[1] - oy).abs() < 1e-12);
        assert!((c[2] - oz).abs() < 1e-12);
    }

    #[test]
    fn test_erfccoulomb_qqr_tighter_than_coulomb_qqr() {
        // ErfcCoulomb QQR bounds should be strictly tighter than Coulomb QQR bounds
        // for distant shell pairs, because of the additional exp(-omega^2 * R^2) decay.
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();

        let op_coulomb = Operator::coulomb();
        let schwarz_c = SchwarzBounds::compute(op_coulomb, &prep).unwrap();
        let qqr_coulomb = QqrBounds::new(schwarz_c, &mol, &bs, &prep, op_coulomb);

        // ErfcCoulomb with omega=0.5 (moderate attenuation)
        let op_erfc = Operator::erfc(0.5);
        // Reuse the same Schwarz data for a fair comparison of just the decay factor.
        let schwarz_e = SchwarzBounds::compute(op_coulomb, &prep).unwrap();
        let qqr_erfc = QqrBounds::new(schwarz_e, &mol, &bs, &prep, op_erfc);

        let nsh = prep.nshells();
        let mut found_tighter = false;
        for i in 0..nsh {
            for j in 0..nsh {
                for k in 0..nsh {
                    for l in 0..nsh {
                        let c_est = qqr_coulomb.estimate(i, j, k, l);
                        let e_est = qqr_erfc.estimate(i, j, k, l);
                        // ErfcCoulomb should always be <= Coulomb (same Schwarz,
                        // extra multiplicative factor <= 1).
                        assert!(
                            e_est <= c_est + 1e-15,
                            "ErfcCoulomb QQR({i},{j},{k},{l}) = {e_est} > Coulomb QQR = {c_est}"
                        );
                        if c_est > 1e-10 && (e_est / c_est) < 0.99 {
                            found_tighter = true;
                        }
                    }
                }
            }
        }
        assert!(
            found_tighter,
            "ErfcCoulomb QQR should be strictly tighter than Coulomb QQR for some distant pairs"
        );
    }
}
