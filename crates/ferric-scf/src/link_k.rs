//! LinK (Linear-scaling K) exchange matrix builder.
//!
//! Uses significant pair lists (geometry-based) and density pair lists
//! (density-dependent) to restrict the shell quartet loop, achieving
//! linear scaling for large systems with sparse density matrices.
//!
//! Reference: Ochsenfeld, White, Head-Gordon, JCP 109, 1663 (1998).

use crate::fock::KBuilder;
use crate::pairs::{intersect_sorted, DensityPairs, SignificantPairs};
use crate::qqr::QqrBounds;
use crate::screening::Bound;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ndarray::Array2;
use std::collections::HashSet;

/// LinK exchange matrix builder.
///
/// Computes K using pair-list-driven loops instead of the canonical O(N^4) shell
/// quartet iteration. For each (ish, jsh) significant pair, the ket shells are
/// restricted to the intersection of geometry-significant and density-significant
/// partners, dramatically reducing the number of computed quartets for spatially
/// extended systems.
pub struct LinkK {
    prep: PreparedBasis,
    bound: QqrBounds,
    sp: SignificantPairs,
    dp: Option<DensityPairs>,
    op: Operator,
    thresh: f64,
}

impl LinkK {
    /// Create a new LinK exchange builder.
    ///
    /// Requires ownership of the `PreparedBasis` (since `Engine` needs a reference
    /// and `PreparedBasis` is not `Clone`). The significant pairs are built from the
    /// QQR bounds at construction time. Density pairs are built lazily on first
    /// `build()` call or via `update_density()`.
    pub fn new(
        prep: PreparedBasis,
        bound: QqrBounds,
        op: Operator,
        thresh: f64,
    ) -> Self {
        let nsh = prep.nshells();
        let sp = SignificantPairs::build(&bound, nsh, thresh);
        LinkK {
            prep,
            bound,
            sp,
            dp: None,
            op,
            thresh,
        }
    }
}

impl KBuilder for LinkK {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<(), FerricError> {
        // Ensure density pairs are built.
        if self.dp.is_none() {
            self.dp = Some(DensityPairs::build(d, &self.bound, &self.prep, self.thresh));
        }
        let dp = self.dp.as_ref().unwrap();

        let nsh = self.prep.nshells();
        let dims = self.prep.shell_dims();
        let offs = self.prep.shell_offsets();

        let mut engine = Engine::new_2e(self.op, &self.prep, 1e-14)?;

        // Track computed canonical quartets to avoid double-counting.
        let mut computed: HashSet<(usize, usize, usize, usize)> = HashSet::new();

        // Find max |D| for screening.
        let max_d = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        // LinK loop: iterate via pair lists.
        for ish in 0..nsh {
            for &jsh in self.sp.partners(ish) {
                let ksh_candidates = intersect_sorted(self.sp.partners(ish), dp.partners(jsh));
                for ksh in ksh_candidates {
                    for &lsh in self.sp.partners(ksh) {
                        // Canonicalize: s1>=s2, s3>=s4, (s1,s2)>=(s3,s4)
                        let (cs1, cs2) = if ish >= jsh { (ish, jsh) } else { (jsh, ish) };
                        let (cs3, cs4) = if ksh >= lsh { (ksh, lsh) } else { (lsh, ksh) };
                        let (cs1, cs2, cs3, cs4) = if (cs1, cs2) >= (cs3, cs4) {
                            (cs1, cs2, cs3, cs4)
                        } else {
                            (cs3, cs4, cs1, cs2)
                        };

                        // Skip if already computed or if screened.
                        if !computed.insert((cs1, cs2, cs3, cs4)) {
                            continue;
                        }
                        if self.bound.estimate(cs1, cs2, cs3, cs4) * max_d < self.thresh {
                            continue;
                        }

                        // Compute the shell quartet.
                        let quartet = engine.compute_quartet(&self.prep, cs1, cs2, cs3, cs4);
                        if let Some(q) = quartet {
                            let (n1, n2, n3, n4) = (dims[cs1], dims[cs2], dims[cs3], dims[cs4]);
                            let (o1, o2, o3, o4) = (offs[cs1], offs[cs2], offs[cs3], offs[cs4]);
                            let sym12 = cs1 != cs2;
                            let sym34 = cs3 != cs4;
                            let sym1234 = (cs1, cs2) != (cs3, cs4);

                            // 8-fold symmetry accumulation, same as build_jk in rhf.rs.
                            for a in 0..n1 {
                                for b in 0..n2 {
                                    for c in 0..n3 {
                                        for dd in 0..n4 {
                                            let v = q[((a * n2 + b) * n3 + c) * n4 + dd];
                                            let mu = o1 + a;
                                            let nu = o2 + b;
                                            let la = o3 + c;
                                            let sg = o4 + dd;

                                            // (mu nu | la sg)
                                            k[(mu, la)] += d[(nu, sg)] * v;

                                            if sym12 {
                                                // (nu mu | la sg)
                                                k[(nu, la)] += d[(mu, sg)] * v;
                                            }

                                            if sym34 {
                                                // (mu nu | sg la)
                                                k[(mu, sg)] += d[(nu, la)] * v;
                                            }

                                            if sym12 && sym34 {
                                                // (nu mu | sg la)
                                                k[(nu, sg)] += d[(mu, la)] * v;
                                            }

                                            if sym1234 {
                                                // (la sg | mu nu)
                                                k[(la, mu)] += d[(sg, nu)] * v;

                                                if sym34 {
                                                    // (sg la | mu nu)
                                                    k[(sg, mu)] += d[(la, nu)] * v;
                                                }

                                                if sym12 {
                                                    // (la sg | nu mu)
                                                    k[(la, nu)] += d[(sg, mu)] * v;
                                                }

                                                if sym12 && sym34 {
                                                    // (sg la | nu mu)
                                                    k[(sg, nu)] += d[(la, mu)] * v;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn update_density(&mut self, d: &Array2<f64>) {
        self.dp = Some(DensityPairs::build(d, &self.bound, &self.prep, self.thresh));
    }

    fn reset(&mut self) {
        self.dp = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fock::KBuilder;
    use crate::rhf::{build_jk, solve_rhf, RhfConfig};
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    /// Run RHF to convergence and return the density and molecule.
    fn converged_density(xyz: &str, basis_name: &str) -> (Array2<f64>, Molecule) {
        let mol = Molecule::parse_xyz(xyz).unwrap();
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let config = RhfConfig {
            energy_conv: 1e-12,
            density_conv: 1e-10,
            integral_thresh: 1e-14,
            ..Default::default()
        };
        let result = solve_rhf(&mol, &prep, op, &bounds, &config).unwrap();
        assert!(result.converged, "RHF did not converge");
        (result.density, mol)
    }

    /// Build K using the direct (canonical) method from rhf.rs.
    fn direct_k(mol: &Molecule, basis_name: &str, d: &Array2<f64>) -> Array2<f64> {
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();
        let mut j = Array2::zeros((n, n));
        let mut k = Array2::zeros((n, n));
        build_jk(&prep, &bounds, 1e-14, d, &mut j, &mut k).unwrap();
        k
    }

    /// Build K using LinK.
    fn link_k(mol: &Molecule, basis_name: &str, d: &Array2<f64>) -> Array2<f64> {
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(mol, &bs).unwrap();
        let op = Operator::coulomb();
        let schwarz = SchwarzBounds::compute(op, &prep).unwrap();
        let qqr = QqrBounds::new(schwarz, mol, &bs, &prep, op);
        let n = prep.nbasis();
        let thresh = 1e-14;
        let mut link = LinkK::new(prep, qqr, op, thresh);
        link.update_density(d);
        let mut k = Array2::zeros((n, n));
        link.build(d, &mut k).unwrap();
        k
    }

    #[test]
    fn test_link_k_matches_direct_k() {
        let water_xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let (d, mol) = converged_density(water_xyz, "sto-3g");

        let k_direct = direct_k(&mol, "sto-3g", &d);
        let k_link = link_k(&mol, "sto-3g", &d);

        let n = k_direct.nrows();
        let mut max_diff = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let diff = (k_direct[(i, j)] - k_link[(i, j)]).abs();
                max_diff = max_diff.max(diff);
            }
        }
        assert!(
            max_diff < 1e-10,
            "LinK K vs direct K max diff = {max_diff:.2e} (water/STO-3G)"
        );
    }

    #[test]
    fn test_link_k_matches_direct_k_ccpvdz() {
        let water_xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let (d, mol) = converged_density(water_xyz, "cc-pvdz");

        let k_direct = direct_k(&mol, "cc-pvdz", &d);
        let k_link = link_k(&mol, "cc-pvdz", &d);

        let n = k_direct.nrows();
        let mut max_diff = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let diff = (k_direct[(i, j)] - k_link[(i, j)]).abs();
                max_diff = max_diff.max(diff);
            }
        }
        assert!(
            max_diff < 1e-10,
            "LinK K vs direct K max diff = {max_diff:.2e} (water/cc-pVDZ)"
        );
    }
}
