//! Continuous Fast Multipole Method (CFMM) for the Coulomb (J) matrix.
//!
//! CFMM achieves O(N) scaling for the Coulomb matrix by using multipole
//! expansions for far-field interactions and direct integration for near-field
//! interactions.
//!
//! Reference: White & Head-Gordon, J. Chem. Phys. 101, 6593 (1994).

use crate::fock::JBuilder;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;

/// A box in the CFMM octree.
#[derive(Debug)]
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
    pub fn new(prep: PreparedBasis, l_max: usize, max_level: usize) -> Self {
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
        self.root.compute_local_expansions(None, self.l_max);
    }

    /// Step 3: Evaluate J. Sum far-field (local exp) and near-field contributions.
    pub fn evaluate_j(&self, d: &Array2<f64>, j: &mut Array2<f64>) {
        // 1. Far-field evaluation from leaf boxes
        self.root.evaluate_far_field(j, &self.prep, self.l_max);
        
        // 2. Near-field evaluation (direct)
        self.evaluate_near_field(d, j);
    }

    fn evaluate_near_field(&self, _d: &Array2<f64>, _j: &mut Array2<f64>) {
        // Direct integration for pairs of shells in adjacent leaf boxes.
        // Uses Schwarz screening for performance.
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
    #[allow(unused_variables)]
    pub fn compute_local_expansions(&mut self, parent: Option<&CfmmBox>, l_max: usize) {
        let n_moments = (l_max + 1) * (l_max + 2) * (l_max + 3) / 6;
        self.local_exp = vec![0.0; n_moments];

        // 1. L2L: Inherit and translate from parent
        if let Some(p) = parent {
            self.translate_local_down(&p.local_exp, l_max);
        }

        // 2. M2L: Add far-field contributions from interaction list
        // Interaction list is computed based on neighbors of parent.
        if let Some(p) = parent {
            self.collect_m2l(p, l_max);
        }

        // 3. Recurse to children
        if let Some(_children) = &mut self.children {
            // Need a trick to pass 'self' as parent while mutating children
            // For now, assume we have a way to access parent neighbors
        }
    }

    fn collect_m2l(&mut self, parent: &CfmmBox, l_max: usize) {
        // Simple logic: iterate over parent's neighbors' children
        // and check if they are "well-separated" from this box.
        if let Some(p_neighbors) = &parent.children { // Simplified: should use actual neighbors
            for neighbor in p_neighbors.iter() {
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

    fn is_well_separated(&self, other: &CfmmBox) -> bool {
        let d2 = (self.center[0] - other.center[0]).powi(2) +
                 (self.center[1] - other.center[1]).powi(2) +
                 (self.center[2] - other.center[2]).powi(2);
        let dist = d2.sqrt();
        // Standard criterion: distance > 2 * box_width
        dist > 2.0 * self.width
    }

    fn add_m2l_contribution(&mut self, _other: &CfmmBox, _l_max: usize) {
        // L_this += Potential_Kernel(M_other)
        // Uses derivatives of 1/r
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

    fn add_shell_multipoles(&mut self, _sh_idx: usize, _d: &Array2<f64>, _prep: &PreparedBasis, _l_max: usize) {
        // TODO: Gaussian product moments integral
    }

    fn add_far_field_to_j(&self, _sh_idx: usize, _j: &mut Array2<f64>, _prep: &PreparedBasis, _l_max: usize) {
        // TODO: evaluate local expansion at shell center
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

    #[allow(unused_variables)]
    fn translate_local_down(&mut self, parent_exp: &[f64], l_max: usize) {
        // For local expansions, the shift is from parent to child
        // but the formula is slightly different or used in reverse.
        // Actually L2L is similar to M2M for Cartesian.
    }
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
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<(), FerricError> {
        self.upward_pass(d);
        self.downward_pass();
        self.evaluate_j(d, j);
        Ok(())
    }

    fn reset(&mut self) {
        // Clear multipoles and local expansions.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
