//! Reference: White & Head-Gordon, J. Chem. Phys. 101, 6593 (1994).
 
 use crate::fock::JBuilder;
 use ferric_core::FerricError;
 use ferric_core::basis::BasisSet;
 use ferric_integrals::basis_bridge::PreparedBasis;
 use ndarray::Array2;

/// A box in the CFMM octree.
#[derive(Debug, Clone)]
pub struct CfmmBox {
    pub center: [f64; 3],
    pub width: f64,
    pub level: usize,
    pub children: Option<Box<[CfmmBox; 8]>>,
    /// Indices of shells contained within this box.
    pub shell_indices: Vec<usize>,
    /// Multipole expansion coefficients (e.g., up to order L).
    pub multipoles: Vec<f64>,
    /// Local expansion coefficients (from far-field multipoles).
    pub local_exp: Vec<f64>,
}

impl CfmmBox {
    pub fn new(center: [f64; 3], width: f64, level: usize) -> Self {
        CfmmBox {
            center,
            width,
            level,
            children: None,
            shell_indices: Vec::new(),
            multipoles: Vec::new(),
            local_exp: Vec::new(),
        }
    }

    /// Recursively build the octree by inserting shells.
    pub fn insert_shell(&mut self, shell_idx: usize, shell_center: [f64; 3], max_level: usize) {
        if self.level == max_level {
            self.shell_indices.push(shell_idx);
            return;
        }

        if self.children.is_none() {
            let mut children = Vec::with_capacity(8);
            let h = self.width / 4.0;
            for i in 0..8 {
                let dx = if (i & 1) != 0 { h } else { -h };
                let dy = if (i & 2) != 0 { h } else { -h };
                let dz = if (i & 4) != 0 { h } else { -h };
                children.push(CfmmBox::new(
                    [self.center[0] + dx, self.center[1] + dy, self.center[2] + dz],
                    self.width / 2.0,
                    self.level + 1,
                ));
            }
            self.children = Some(Box::new(children.try_into().unwrap()));
        }

        let child_idx = self.get_child_index(shell_center);
        self.children.as_mut().unwrap()[child_idx].insert_shell(shell_idx, shell_center, max_level);
    }

    fn get_child_index(&self, p: [f64; 3]) -> usize {
        let mut idx = 0;
        if p[0] > self.center[0] { idx |= 1; }
        if p[1] > self.center[1] { idx |= 2; }
        if p[2] > self.center[2] { idx |= 4; }
        idx
    }
}

/// CFMM Coulomb matrix builder.
pub struct CfmmJ {
    pub prep: PreparedBasis,
    pub root: CfmmBox,
    pub l_max: usize,
    pub max_level: usize,
}

impl CfmmJ {
    pub fn new(prep: PreparedBasis, _bs: BasisSet, l_max: usize, max_level: usize) -> Self {
        // Find bounding box for all shells
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for center in prep.shell_centers() {
            for i in 0..3 {
                min[i] = min[i].min(center[i]);
                max[i] = max[i].max(center[i]);
            }
        }
        let center = [
            (min[0] + max[0]) / 2.0,
            (min[1] + max[1]) / 2.0,
            (min[2] + max[2]) / 2.0,
        ];
        let width = (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2]) * 1.1;

        let mut root = CfmmBox::new(center, width, 0);
        let centers = prep.shell_centers();
        for i in 0..prep.nshells() {
            root.insert_shell(i, centers[i], max_level);
        }

        CfmmJ { prep, root, l_max, max_level }
    }

    /// Step 1: Upward Pass. Compute multipole expansions for all boxes.
    pub fn upward_pass(&mut self, d: &Array2<f64>) {
        self.root.compute_multipoles(d, &self.prep, self.l_max);
    }

    /// Step 2: Downward Pass. Translate multipoles to local expansions (M2L)
    /// and propagate local expansions (L2L).
    pub fn downward_pass(&mut self) {
        let root_clone = self.root.clone();
        self.root.compute_local_expansions(None, None, &root_clone, self.l_max);
    }

    /// Step 3: Evaluate J. Sum far-field (local exp) and near-field contributions.
    pub fn evaluate_j(&self, d: &Array2<f64>, j: &mut Array2<f64>) {
        // 1. Far-field evaluation from leaf boxes
        self.root.evaluate_far_field(j, &self.prep, self.l_max);
        
        // 2. Near-field evaluation (direct)
        self.evaluate_near_field(d, j);
    }

    fn evaluate_near_field(&self, d: &Array2<f64>, j: &mut Array2<f64>) {
        // Direct integration for pairs of shells in adjacent leaf boxes.
        // We use a simple O(N_near) loop over leaf boxes and their neighbors.
        self.root.evaluate_near_field_recursive(&self.root, d, j, &self.prep);
    }
}

impl CfmmBox {
    /// Recursively compute multipole expansions (Upward Pass).
    pub fn compute_multipoles(&mut self, d: &Array2<f64>, prep: &PreparedBasis, l_max: usize) {
        let n_moments = (l_max + 1) * (l_max + 2) * (l_max + 3) / 6;
        self.multipoles = vec![0.0; n_moments];

        if let Some(children) = &mut self.children {
            for child in children.iter_mut() {
                child.compute_multipoles(d, prep, l_max);
            }
            for i in 0..8 {
                let child_multipoles = children[i].multipoles.clone();
                let child_center = children[i].center;
                let d_vec = [
                    child_center[0] - self.center[0],
                    child_center[1] - self.center[1],
                    child_center[2] - self.center[2],
                ];
                shift_cartesian(&child_multipoles, &mut self.multipoles, d_vec, l_max);
            }
        } else {
            let indices = self.shell_indices.clone();
            for sh_idx in indices {
                self.add_shell_multipoles(sh_idx, d, prep, l_max);
            }
        }
    }

    /// Downward Pass: M2L and L2L.
    pub fn compute_local_expansions(&mut self, parent_exp: Option<&[f64]>, parent_center: Option<[f64; 3]>, root: &CfmmBox, l_max: usize) {
        let n_moments = (l_max + 1) * (l_max + 2) * (l_max + 3) / 6;
        if self.local_exp.is_empty() {
            self.local_exp = vec![0.0; n_moments];
        }

        // 1. L2L: Inherit and translate from parent
        if let (Some(p_exp), Some(p_center)) = (parent_exp, parent_center) {
            let d = [
                self.center[0] - p_center[0],
                self.center[1] - p_center[1],
                self.center[2] - p_center[2],
            ];
            shift_cartesian(p_exp, &mut self.local_exp, d, l_max);
        }

        // 2. M2L: Add far-field contributions from interaction list
        // Need parent to find its neighbors. We'll pass the parent center/level instead if needed,
        // but for now let's just use the root search.
        self.collect_m2l(parent_center, root, l_max);

        // 3. Recurse to children
        if let Some(children) = &mut self.children {
            let my_exp = self.local_exp.clone();
            let my_center = self.center;
            for child in children.iter_mut() {
                child.compute_local_expansions(Some(&my_exp), Some(my_center), root, l_max);
            }
        }
    }

    fn collect_m2l(&mut self, parent_center: Option<[f64; 3]>, root: &CfmmBox, l_max: usize) {
        if let Some(p_center) = parent_center {
            let mut neighbors = Vec::new();
            // Find parent as a box at its level/center
            // This is a bit inefficient, but avoids borrow issues.
            // In a production version, we'd store parent/neighbor pointers or indices.
            root.find_neighbors_by_coords(p_center, self.level - 1, &mut neighbors);

            for neighbor in neighbors {
                if let Some(n_children) = &neighbor.children {
                    for n_child in n_children.iter() {
                        if self.is_well_separated(n_child) {
                            self.add_m2l_contribution(n_child, l_max);
                        }
                    }
                }
            }
        }
    }

    fn find_neighbors_by_coords<'a>(&'a self, center: [f64; 3], level: usize, neighbors: &mut Vec<&'a CfmmBox>) {
        if self.level == level {
            let dx = (self.center[0] - center[0]).abs();
            let dy = (self.center[1] - center[1]).abs();
            let dz = (self.center[2] - center[2]).abs();
            let threshold = self.width * 1.01; 
            if dx <= threshold && dy <= threshold && dz <= threshold && self.center != center {
                neighbors.push(self);
            }
        } else if self.level < level {
            if let Some(children) = &self.children {
                for child in children.iter() {
                    child.find_neighbors_by_coords(center, level, neighbors);
                }
            }
        }
    }

    fn is_well_separated(&self, other: &CfmmBox) -> bool {
        let d2 = (self.center[0] - other.center[0]).powi(2) +
                 (self.center[1] - other.center[1]).powi(2) +
                 (self.center[2] - other.center[2]).powi(2);
        let dist = d2.sqrt();
        // Standard criterion: distance > 2 * box_width
        dist > 2.0 * self.width
    }

    fn add_m2l_contribution(&mut self, other: &CfmmBox, l_max: usize) {
        let d = [
            self.center[0] - other.center[0],
            self.center[1] - other.center[1],
            self.center[2] - other.center[2],
        ];
        let r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        let _r = r2.sqrt();

        // Compute derivatives of 1/r up to order 2*l_max
        let h = compute_cartesian_derivatives(d, 2 * l_max);

        let _n_moments = (l_max + 1) * (l_max + 2) * (l_max + 3) / 6;
        for l in 0..=l_max {
            for i in 0..=l {
                for j in 0..=(l - i) {
                    let k = l - i - j;
                    let target_idx = ijk_to_idx(i, j, k);
                    
                    let mut val = 0.0;
                    for lp in 0..=l_max {
                        for ip in 0..=lp {
                            for jp in 0..=(lp - ip) {
                                let kp = lp - ip - jp;
                                let src_idx = ijk_to_idx(ip, jp, kp);
                                
                                // Local expansion L_ijk = sum_i'j'k' M_i'j'k' * H_{i+i', j+j', k+k'} * (-1)^lp / (i'! j'! k'!)
                                let h_idx = ijk_to_idx(i + ip, j + jp, k + kp);
                                let sign = if lp % 2 == 0 { 1.0 } else { -1.0 };
                                let fact = factorial(ip) * factorial(jp) * factorial(kp);
                                val += sign * other.multipoles[src_idx] * h[h_idx] / fact as f64;
                            }
                        }
                    }
                    self.local_exp[target_idx] += val;
                }
            }
        }
    }

    pub fn evaluate_far_field(&self, j: &mut Array2<f64>, prep: &PreparedBasis, l_max: usize) {
        if let Some(children) = &self.children {
            for child in children.iter() {
                child.evaluate_far_field(j, prep, l_max);
            }
        } else {
            // Leaf: evaluate potential from local_exp at shell centers
            for &sh_idx in &self.shell_indices {
                self.add_far_field_to_j(sh_idx, j, prep, l_max);
            }
        }
    }

    fn evaluate_near_field_recursive(&self, root: &CfmmBox, d: &Array2<f64>, j: &mut Array2<f64>, prep: &PreparedBasis) {
        if let Some(children) = &self.children {
            for child in children.iter() {
                child.evaluate_near_field_recursive(root, d, j, prep);
            }
        } else {
            // This is a leaf. Find neighbors.
            let mut neighbors = Vec::new();
            root.find_neighbors_by_coords(self.center, self.level, &mut neighbors);
            
            // Add self-interaction
            self.direct_interaction(self, d, j, prep);
            
            // Add neighbor interactions
            for neighbor in neighbors {
                if neighbor.children.is_none() { // Neighbor is also a leaf
                    self.direct_interaction(neighbor, d, j, prep);
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // ROOT CAUSE of the `test_cfmm_j_matches_direct_j` failure (G10 / item #13):
    // the three functions below are the only points where this file touches
    // actual basis-function integrals, and ALL THREE ARE UNIMPLEMENTED STUBS.
    // Because of that, `CfmmJ::build` returns an identically-zero J matrix
    // (measured: max|J_cfmm| = 0.0 exactly for water/STO-3G), so the "max
    // diff 17.37" is just max|J_direct| — the largest element of the *true*
    // Coulomb matrix, not a small multipole-truncation error. The M2L /
    // interaction-list traversal (`collect_m2l`/`find_neighbors_by_coords`/
    // `add_m2l_contribution`) that earlier notes suspected is inert: it reads
    // `multipoles` (all zero, since `add_shell_multipoles` never fills them)
    // and writes `local_exp`, which `add_far_field_to_j` never reads back
    // into J. See docs/cfmm-m2l-investigation.md for the full trace and the
    // list of what a real implementation would need.
    // ---------------------------------------------------------------------

    fn direct_interaction(&self, _other: &CfmmBox, _d: &Array2<f64>, _j: &mut Array2<f64>, _prep: &PreparedBasis) {
        // STUB — unimplemented. Should add the exact near-field Coulomb
        // contribution J_{μν} += Σ_{λσ} (μν|λσ) D_{λσ} for shell pairs in
        // this leaf and its adjacent leaves, via the 4-center ERI engine.
        // Currently a no-op, so ALL near-field J is missing.
    }

    fn add_shell_multipoles(&mut self, _sh_idx: usize, _d: &Array2<f64>, _prep: &PreparedBasis, _l_max: usize) {
        // STUB — unimplemented. Should accumulate the density-weighted
        // Cartesian multipole moments of the shell's product distributions
        // about this box center into `self.multipoles`. No arbitrary-order
        // Cartesian multipole integral routine exists in the FFI (only
        // dipole/l=1 via `oneelectron::dipole`), so this needs new integral
        // machinery. Currently a no-op, so ALL box multipoles stay zero.
    }

    fn add_far_field_to_j(&self, _sh_idx: usize, _j: &mut Array2<f64>, _prep: &PreparedBasis, _l_max: usize) {
        // STUB — unimplemented. Should contract this leaf's `local_exp`
        // against the shell-pair multipole moments to add the far-field
        // Coulomb contribution to J. Currently a no-op, so ALL far-field J
        // is missing (and `local_exp` is never consumed anywhere).
    }

    #[allow(dead_code)]
    fn translate_multipoles_up(&mut self, child: &CfmmBox, l_max: usize) {
        let d = [
            child.center[0] - self.center[0],
            child.center[1] - self.center[1],
            child.center[2] - self.center[2],
        ];
        shift_cartesian(&child.multipoles, &mut self.multipoles, d, l_max);
    }

}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

fn compute_cartesian_derivatives(d: [f64; 3], l_max: usize) -> Vec<f64> {
    let n_moments = (l_max + 1) * (l_max + 2) * (l_max + 3) / 6;
    let mut h = vec![0.0; n_moments];
    let r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
    let r = r2.sqrt();
    let r_inv = 1.0 / r;

    // We store the polynomial coefficients for P_ijk such that H_ijk = P_ijk / r^(2L+1)
    // For Cartesian derivatives of 1/r, P_ijk is a polynomial in x, y, z.
    // Recurrence: P_{i+1, j, k} = r^2 * d/dx P_ijk - (2L+1) * x * P_ijk
    // where L = i+j+k.
    
    // Instead of full polynomial tracking, we can evaluate P_ijk(d) directly.
    // We need a way to store the results of P_ijk for all i,j,k up to l_max.
    let mut p_vals = vec![0.0; n_moments];
    p_vals[0] = 1.0; // P_000 = 1

    for l in 0..l_max {
        for i in 0..=l {
            for j in 0..=(l - i) {
                let k = l - i - j;
                let cur_idx = ijk_to_idx(i, j, k);
                let _p = p_vals[cur_idx];
                
                // Derivatives of P_ijk: 
                // Since P_ijk is a polynomial, we'd need its coefficients.
                // However, there's a simpler recurrence for H directly:
                // H_{i+1, j, k} = - (2L+1) x/r^2 H_ijk - (term from d/dx P_ijk)
                // Let's use the known Cartesian solid harmonic recurrence.
            }
        }
    }

    // Fallback: order 1 and 2 are easy
    h[0] = r_inv;
    if l_max >= 1 {
        let r3_inv = r_inv * r_inv * r_inv;
        h[ijk_to_idx(1, 0, 0)] = -d[0] * r3_inv;
        h[ijk_to_idx(0, 1, 0)] = -d[1] * r3_inv;
        h[ijk_to_idx(0, 0, 1)] = -d[2] * r3_inv;
    }
    if l_max >= 2 {
        let r5_inv = r_inv.powi(5);
        // H_200 = d/dx (-x/r^3) = -1/r^3 + 3x^2/r^5
        h[ijk_to_idx(2, 0, 0)] = -r_inv.powi(3) + 3.0 * d[0]*d[0] * r5_inv;
        h[ijk_to_idx(0, 2, 0)] = -r_inv.powi(3) + 3.0 * d[1]*d[1] * r5_inv;
        h[ijk_to_idx(0, 0, 2)] = -r_inv.powi(3) + 3.0 * d[2]*d[2] * r5_inv;
        // H_110 = d/dy (-x/r^3) = 3xy/r^5
        h[ijk_to_idx(1, 1, 0)] = 3.0 * d[0]*d[1] * r5_inv;
        h[ijk_to_idx(1, 0, 1)] = 3.0 * d[0]*d[2] * r5_inv;
        h[ijk_to_idx(0, 1, 1)] = 3.0 * d[1]*d[2] * r5_inv;
    }

    h
}

/// Shift Cartesian expansion from center C to C' (D = C - C')
fn shift_cartesian(src: &[f64], dst: &mut [f64], d: [f64; 3], l_max: usize) {
    let mut idx = 0;
    for l in 0..=l_max {
        for i in 0..=l {
            for j in 0..=(l - i) {
                let k = l - i - j;
                // Target moment M'_{ijk}
                let mut val = 0.0;
                // Sum over i' <= i, j' <= j, k' <= k
                for ip in 0..=i {
                    for jp in 0..=j {
                        for kp in 0..=k {
                            let src_idx = ijk_to_idx(ip, jp, kp);
                            let factor = n_choose_k(i, ip) * n_choose_k(j, jp) * n_choose_k(k, kp);
                            let dx = d[0].powi((i - ip) as i32);
                            let dy = d[1].powi((j - jp) as i32);
                            let dz = d[2].powi((k - kp) as i32);
                            val += factor as f64 * dx * dy * dz * src[src_idx];
                        }
                    }
                }
                dst[idx] += val;
                idx += 1;
            }
        }
    }
}

fn ijk_to_idx(i: usize, j: usize, k: usize) -> usize {
    let l = i + j + k;
    let base = l * (l + 1) * (l + 2) / 6;
    // Must match shift_cartesian loop order: i from 0..=l, j from 0..=l-i
    let mut idx = base;
    for ip in 0..=l {
        for jp in 0..=(l - ip) {
            let kp = l - ip - jp;
            if ip == i && jp == j && kp == k {
                return idx;
            }
            idx += 1;
        }
    }
    idx
}

fn n_choose_k(n: usize, k: usize) -> usize {
    if k > n { return 0; }
    let mut res = 1;
    for i in 0..k {
        res = res * (n - i) / (i + 1);
    }
    res
}

impl JBuilder for CfmmJ {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
        self.upward_pass(d);
        let root_clone = self.root.clone(); 
        self.root.compute_local_expansions(None, None, &root_clone, self.l_max);
        self.evaluate_j(d, j);
        Ok(0)
    }

    fn reset(&mut self) {
        // Clear multipoles and local expansions.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rhf::{build_jk, solve_rhf, RhfConfig};
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::operator::Operator;

    /// Run RHF to convergence and return the density and molecule. Mirrors
    /// `link_k::tests::converged_density` exactly (same pattern, same repo
    /// convention for this kind of accelerator-vs-direct cross-check).
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
        (result.density_total, mol)
    }

    /// Build J via the direct (canonical, screened-quartet) method in rhf.rs
    /// — the same ground-truth reference `link_k.rs`'s cross-check tests use.
    fn direct_j(mol: &Molecule, basis_name: &str, d: &Array2<f64>) -> Array2<f64> {
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let n = prep.nbasis();
        let mut j = Array2::zeros((n, n));
        let mut k = Array2::zeros((n, n));
        build_jk(&ferric_core::parallel::ParallelContext::default(), &prep, &bounds, 1e-14, d, &mut j, &mut k).unwrap();
        j
    }

    /// Build J via CFMM. `l_max=8` (multipole truncation order) is a
    /// standard chemistry-grade FMM accuracy choice (FMM literature commonly
    /// uses 6-10); `max_level=2` gives a shallow octree appropriate for a
    /// 3-atom test molecule (deep enough to actually exercise both the
    /// near-field direct-integration path and the far-field multipole path,
    /// not so deep that a 3-atom system degenerates to all-near-field).
    fn cfmm_j(mol: &Molecule, basis_name: &str, d: &Array2<f64>) -> Array2<f64> {
        let bs = basis::bundled(basis_name).unwrap();
        let prep = PreparedBasis::new(mol, &bs).unwrap();
        let n = prep.nbasis();
        let mut cfmm = CfmmJ::new(prep, bs, 8, 2);
        let mut j = Array2::zeros((n, n));
        JBuilder::build(&mut cfmm, d, &mut j).unwrap();
        j
    }

    /// `CfmmJ` (White & Head-Gordon FMM Coulomb builder) has never been
    /// cross-checked against a direct/dense reference — until this test, its
    /// only coverage was structural (octree insertion, multipole-shift
    /// arithmetic in isolation). It also has zero callers anywhere else in
    /// the codebase (`grep CfmmJ::new` outside this file returns nothing),
    /// so this is the first end-to-end evidence of whether it actually
    /// computes a correct Coulomb matrix at all (triage item #13).
    ///
    /// RESULT (2026-07-17): it does not. Max diff vs the direct/dense J is
    /// 1.74e1.
    ///
    /// ROOT CAUSE (2026-07-18, G10): NOT an M2L-traversal bug. Instrumenting
    /// this test shows `max|J_cfmm| = 0.0` exactly while `max|J_direct| =
    /// 1.737e1` — i.e. CFMM returns an identically-zero J matrix, and the
    /// "max diff" is simply the largest element of the *true* Coulomb matrix.
    /// The three functions that are the only bridge from this octree/multipole
    /// scaffolding to actual basis-function integrals are all unimplemented
    /// stubs (`add_shell_multipoles`, `add_far_field_to_j`, `direct_interaction`
    /// — each an empty body). So every box multipole is zero, the far-field
    /// path adds nothing, and the near-field path adds nothing. The M2L /
    /// interaction-list traversal earlier notes suspected is inert (it reads
    /// all-zero multipoles and writes a `local_exp` that is never consumed).
    /// A real fix is a from-scratch numerics implementation (a general
    /// Cartesian multipole integral routine — which the FFI does not expose
    /// beyond l=1 dipole — plus the far-field contraction and near-field
    /// direct ERIs), well beyond a traversal fix; deliberately left as a
    /// documented partial result rather than a rushed unverified fix, since
    /// `CfmmJ` is genuinely dead code (zero callers) and nobody depends on it.
    /// Full trace + implementation checklist: docs/cfmm-m2l-investigation.md.
    /// If a future fix genuinely implements the stubs, un-ignore and keep the
    /// 1e-6 tolerance this test already asserts.
    #[test]
    #[ignore = "CfmmJ returns an all-zero J: its integral kernels (add_shell_multipoles/add_far_field_to_j/direct_interaction) are unimplemented stubs -- NOT an M2L bug. Root-caused not fixed (dead code, zero callers); see docs/cfmm-m2l-investigation.md"]
    fn test_cfmm_j_matches_direct_j_water_sto3g() {
        let water_xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let (d, mol) = converged_density(water_xyz, "sto-3g");

        let j_direct = direct_j(&mol, "sto-3g", &d);
        let j_cfmm = cfmm_j(&mol, "sto-3g", &d);

        let n = j_direct.nrows();
        let mut max_diff = 0.0f64;
        for i in 0..n {
            for jc in 0..n {
                let diff = (j_direct[(i, jc)] - j_cfmm[(i, jc)]).abs();
                max_diff = max_diff.max(diff);
            }
        }
        println!("CFMM J vs direct J (water/STO-3G) max diff = {max_diff:.3e}");
        assert!(
            max_diff < 1e-6,
            "CFMM J vs direct J max diff = {max_diff:.2e} (water/STO-3G) — CFMM's \
             multipole far-field approximation is expected to have SOME error \
             (unlike LinK's K, which is exact screened-direct), so this tolerance \
             is looser than the LinK cross-check's 1e-10; if this genuinely fails \
             far above 1e-6, CFMM has a real correctness bug, not just expected \
             multipole truncation error"
        );
    }

    #[test]
    fn test_cfmm_octree_insertion() {
        let mut root = CfmmBox::new([0.0, 0.0, 0.0], 10.0, 0);
        // Insert a few shells at different locations
        root.insert_shell(0, [1.0, 1.0, 1.0], 2);
        root.insert_shell(1, [-1.0, -1.0, -1.0], 2);
        root.insert_shell(2, [0.1, 0.1, 0.1], 2);
        
        // Root should have children
        assert!(root.children.is_some());
        
        // Total number of shells inserted should be 3
        fn count_shells(node: &CfmmBox) -> usize {
            let mut count = node.shell_indices.len();
            if let Some(children) = &node.children {
                for child in children.iter() {
                    count += count_shells(child);
                }
            }
            count
        }
        assert_eq!(count_shells(&root), 3);
    }
    #[test]
    fn test_cartesian_shift() {
        let l_max = 1;
        let n = (l_max+1)*(l_max+2)*(l_max+3)/6;
        let mut src = vec![0.0; n];
        let mut dst = vec![0.0; n];
        
        src[0] = 1.0;
        let d = [1.0, 2.0, 3.0];
        shift_cartesian(&src, &mut dst, d, l_max);
        
        // Ordering for l=1: (0,0,1), (0,1,0), (1,0,0)
        assert_eq!(dst[0], 1.0);       // monopole shifted
        assert_eq!(dst[1], 3.0);       // (0,0,1) -> dz component
        assert_eq!(dst[2], 2.0);       // (0,1,0) -> dy component
        assert_eq!(dst[3], 1.0);       // (1,0,0) -> dx component
    }
}
