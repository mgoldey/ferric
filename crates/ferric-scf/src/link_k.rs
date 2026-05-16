//! LinK (Linear-scaling K) exchange matrix builder.
//!
//! Uses significant pair lists (geometry-based) and density pair lists
//! (density-dependent) to restrict the shell quartet loop, achieving
//! linear scaling for large systems with sparse density matrices.
//!
//! Reference: Ochsenfeld, White, Head-Gordon, JCP 109, 1663 (1998).

use crate::fock::KBuilder;
use crate::pairs::{DensityPairs, SignificantPairs};

use crate::screening::Bound;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ndarray::Array2;

/// LinK exchange matrix builder.
///
/// Computes K using pair-list-driven loops instead of the canonical O(N^4) shell
/// quartet iteration. For each (ish, jsh) significant pair, the ket shells are
/// restricted to the intersection of geometry-significant and density-significant
/// partners, dramatically reducing the number of computed quartets for spatially
/// extended systems.
pub struct LinkK<'a, B: Bound> {
    ctx: &'a ParallelContext,
    prep: &'a PreparedBasis,
    bound: &'a B,
    sp: SignificantPairs,
    dp: Option<DensityPairs>,
    op: Operator,
    thresh: f64,
}

impl<'a, B: Bound> LinkK<'a, B> {
    /// Create a new LinK exchange builder.
    ///
    /// Requires a `ParallelContext`, `PreparedBasis`, and a `Bound`. The significant pairs
    /// are built from the bounds at construction time. Density pairs are built lazily on
    /// first `build()` call or via `update_density()`.
    pub fn new(
        ctx: &'a ParallelContext,
        prep: &'a PreparedBasis,
        bound: &'a B,
        op: Operator,
        thresh: f64,
    ) -> Self {
        let nsh = prep.nshells();
        let sp = SignificantPairs::build(bound, nsh, thresh);
        LinkK {
            ctx,
            prep,
            bound,
            sp,
            dp: None,
            op,
            thresh,
        }
    }
}

use rayon::prelude::*;

impl<'a, B: Bound + Sync> KBuilder for LinkK<'a, B> {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        self.ctx.check_interrupted()?;

        // Ensure density pairs are built.
        if self.dp.is_none() {
            self.dp = Some(DensityPairs::build(d, self.bound, &self.prep, self.thresh));
        }
        let dp = self.dp.as_ref().unwrap();

        let nsh = self.prep.nshells();
        let dims = self.prep.shell_dims();
        let offs = self.prep.shell_offsets();
        let thresh = self.thresh;
        let op = self.op;
        let rank = self.ctx.rank;
        let size = self.ctx.size;

        // Find max |D| for screening.
        let max_d = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        // Enumerate all significant (ish, jsh) pairs as parallel work units.
        // This gives O(N) tasks with roughly equal work each (each handles ksh/lsh loops),
        // vs O(1) tasks per ish where work grows as O(N²) causing severe imbalance.
        // MPI: distribute by pair index; Rayon: all pairs on this rank run in parallel.
        let ij_pairs: Vec<(usize, usize)> = (0..nsh)
            .flat_map(|ish| self.sp.partners(ish).iter().map(move |&jsh| (ish, jsh)))
            .enumerate()
            .filter(|(idx, _)| idx % size == rank)
            .map(|(_, p)| p)
            .collect();

        let k_final = ij_pairs
            .into_par_iter()
            .fold(
                || {
                    let k_local = Array2::zeros(k.raw_dim());
                    let engine = Engine::new_2e(op, &self.prep, 1e-14).unwrap();
                    // Bitvec dedup for (cs3, cs4) — size nsh² bits; ish and jsh are fixed per task.
                    let seen: Vec<u64> = vec![0u64; (nsh * nsh + 63) / 64];
                    Ok((k_local, engine, seen, Vec::<usize>::new(), 0usize))
                },
                |acc: Result<_, FerricError>, (ish, jsh)| {
                    let (mut k_local, mut engine, mut seen, mut dirty, mut count) = acc?;
                    for &w in &dirty { seen[w] = 0; }
                    dirty.clear();

                    // Inline merge: sp.partners(ish) ∩ dp.partners(jsh) — no Vec allocation.
                    let sp_ish = self.sp.partners(ish);
                    let dp_jsh = dp.partners(jsh);
                    let mut ai = 0;
                    let mut bi = 0;
                    while ai < sp_ish.len() && bi < dp_jsh.len() {
                        let ksh = match sp_ish[ai].cmp(&dp_jsh[bi]) {
                            std::cmp::Ordering::Equal => { ai += 1; bi += 1; sp_ish[ai - 1] }
                            std::cmp::Ordering::Less => { ai += 1; continue; }
                            std::cmp::Ordering::Greater => { bi += 1; continue; }
                        };

                        for &lsh in self.sp.partners(ksh) {
                            let (cs1, cs2) = if ish >= jsh { (ish, jsh) } else { (jsh, ish) };
                            let (cs3, cs4) = if ksh >= lsh { (ksh, lsh) } else { (lsh, ksh) };
                            let (cs1, cs2, cs3, cs4) = if (cs1, cs2) >= (cs3, cs4) {
                                (cs1, cs2, cs3, cs4)
                            } else {
                                (cs3, cs4, cs1, cs2)
                            };

                            // Canonical ownership: only the (ish,jsh) pair where ish==cs1 and
                            // jsh==cs2 computes this quartet, avoiding double-counting across tasks.
                            if cs1 != ish || cs2 != jsh { continue; }

                            // Bitvec dedup over (cs3, cs4) — ish==cs1 and jsh==cs2 are fixed.
                            let bit = cs3 * nsh + cs4;
                            let word = bit / 64;
                            let mask = 1u64 << (bit % 64);
                            if seen[word] & mask != 0 { continue; }
                            if seen[word] == 0 { dirty.push(word); }
                            seen[word] |= mask;

                            if self.bound.estimate(cs1, cs2, cs3, cs4) * max_d < thresh {
                                continue;
                            }

                            if let Some(q) = engine.compute_quartet(&self.prep, cs1, cs2, cs3, cs4) {
                                count += 1;
                                let (n1, n2, n3, n4) = (dims[cs1], dims[cs2], dims[cs3], dims[cs4]);
                                let (o1, o2, o3, o4) = (offs[cs1], offs[cs2], offs[cs3], offs[cs4]);
                                let sym12 = cs1 != cs2;
                                let sym34 = cs3 != cs4;
                                let sym1234 = (cs1, cs2) != (cs3, cs4);

                                for a in 0..n1 {
                                    for b in 0..n2 {
                                        for c in 0..n3 {
                                            for dd in 0..n4 {
                                                let v = q[((a * n2 + b) * n3 + c) * n4 + dd];
                                                let mu = o1 + a;
                                                let nu = o2 + b;
                                                let la = o3 + c;
                                                let sg = o4 + dd;

                                                k_local[(mu, la)] += d[(nu, sg)] * v;
                                                if sym12 { k_local[(nu, la)] += d[(mu, sg)] * v; }
                                                if sym34 { k_local[(mu, sg)] += d[(nu, la)] * v; }
                                                if sym12 && sym34 { k_local[(nu, sg)] += d[(mu, la)] * v; }

                                                if sym1234 {
                                                    k_local[(la, mu)] += d[(sg, nu)] * v;
                                                    if sym34 { k_local[(sg, mu)] += d[(la, nu)] * v; }
                                                    if sym12 { k_local[(la, nu)] += d[(sg, mu)] * v; }
                                                    if sym12 && sym34 { k_local[(sg, nu)] += d[(la, mu)] * v; }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Ok((k_local, engine, seen, dirty, count))
                },
            )
            .map(|res| res.map(|(k_local, _, _, _, count)| (k_local, count)))
            .reduce(
                || Ok((Array2::zeros(k.raw_dim()), 0usize)),
                |a, b| {
                    let (mut k_a, count_a) = a?;
                    let (k_b, count_b) = b?;
                    k_a += &k_b;
                    Ok((k_a, count_a + count_b))
                },
            )?;

        k.assign(&k_final.0);

        #[cfg(feature = "mpi")]
        if let Some(world) = &self.ctx.world {
            let mut k_global = Array2::zeros(k.dim());
            world.all_reduce_into(
                k.as_slice().unwrap(),
                k_global.as_slice_mut().unwrap(),
                mpi::collective::SystemOperation::sum(),
            );
            *k = k_global;
        }

        Ok(k_final.1)
    }

    fn update_density(&mut self, d: &Array2<f64>) {
        self.dp = Some(DensityPairs::build(d, self.bound, self.prep, self.thresh));
    }

    fn reset(&mut self) {
        self.dp = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qqr::QqrBounds;
    use crate::fock::KBuilder;
    use crate::rhf::{build_jk, solve_rhf, RhfConfig};
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    /// Run RHF to convergence and return the density and molecule.
    fn converged_density(xyz: &str, basis_name: &str) -> (Array2<f64>, Molecule) {
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
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
        let result = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &prep, op, &bounds, &config).unwrap();
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
        build_jk(&ferric_core::parallel::ParallelContext::default(), &prep, &bounds, 1e-14, d, &mut j, &mut k).unwrap();
        k
    }

    /// Build K using LinK with the given ParallelContext.
    fn link_k_with_ctx(mol: &Molecule, basis_name: &str, d: &Array2<f64>, ctx: &ferric_core::parallel::ParallelContext) -> Array2<f64> {
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(mol, &bs).unwrap();
        let op = Operator::coulomb();
        let schwarz = SchwarzBounds::compute(op, &prep).unwrap();
        let qqr = QqrBounds::new(schwarz, mol, &bs, &prep, op);
        let n = prep.nbasis();
        let thresh = 1e-14;
        let mut link = LinkK::new(ctx, &prep, &qqr, op, thresh);
        link.update_density(d);
        let mut k = Array2::zeros((n, n));
        link.build(d, &mut k).unwrap();
        k
    }

    /// Build K using LinK.
    fn link_k(mol: &Molecule, basis_name: &str, d: &Array2<f64>) -> Array2<f64> {
        link_k_with_ctx(mol, basis_name, d, &ferric_core::parallel::ParallelContext::default())
    }

    /// Simulate MPI sharding: build K using rank r of size n, return raw (un-reduced) partial K.
    fn link_k_rank(mol: &Molecule, basis_name: &str, d: &Array2<f64>, rank: usize, size: usize) -> Array2<f64> {
        let ctx = ferric_core::parallel::ParallelContext {
            rank,
            size,
            #[cfg(feature = "mpi")]
            world: None,
        };
        link_k_with_ctx(mol, basis_name, d, &ctx)
    }

    #[test]
    fn test_link_k_rank_sharding_sums_to_full() {
        // Verify that LinK with rank-based sharding produces partial K matrices
        // that sum to the full K matrix (simulating what MPI all-reduce would do).
        let water_xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let (d, mol) = converged_density(water_xyz, "sto-3g");

        let k_full = link_k(&mol, "sto-3g", &d);

        let k0 = link_k_rank(&mol, "sto-3g", &d, 0, 2);
        let k1 = link_k_rank(&mol, "sto-3g", &d, 1, 2);
        let k_sum = &k0 + &k1;

        let n = k_full.nrows();
        let mut max_diff = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                max_diff = max_diff.max((k_full[(i, j)] - k_sum[(i, j)]).abs());
            }
        }
        assert!(
            max_diff < 1e-10,
            "LinK rank0+rank1 sum vs full K max diff = {max_diff:.2e} (water/STO-3G)"
        );
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

    #[test]
    fn test_link_k_parallel_consistency() {
        let water_xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let (d, mol) = converged_density(water_xyz, "sto-3g");

        let k1 = link_k(&mol, "sto-3g", &d);
        let k2 = link_k(&mol, "sto-3g", &d);

        let n = k1.nrows();
        for i in 0..n {
            for j in 0..n {
                assert!((k1[(i, j)] - k2[(i, j)]).abs() < 1e-15);
            }
        }
    }
}
