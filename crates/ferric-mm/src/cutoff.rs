//! Opt-in nonbonded cutoff: OpenMM `NonbondedForce.CutoffNonPeriodic`.
//!
//! [`crate::gradient`] sums every non-excluded pair with no cutoff and stays
//! the default. [`gradient_with`] with [`NonbondedMethod::CutoffNonPeriodic`]
//! instead evaluates, as OpenMM's `CutoffNonPeriodic` does (theory.md 8):
//!
//! - ordinary pairs (neither excluded nor 1-4) only when `r < r_c`, found by
//!   a cell list ([`neighbor_pairs`]), with reaction-field Coulomb
//!   `q_i q_j (1/r + k_rf r^2 - c_rf)` and Lennard-Jones truncated at `r_c`
//!   (optionally multiplied by OpenMM's quintic switching function from
//!   `switch_distance` to `r_c`);
//! - 1-4 pairs at every distance, with the plain scaled LJ + Coulomb of the
//!   all-pairs sum (OpenMM computes exceptions with no cutoff and no reaction
//!   field);
//! - 1-2/1-3 pairs not at all (no reaction-field exclusion correction).
//!
//! No periodic images and no PME / Ewald: those are not implemented. Bonded
//! terms are unchanged. Validated against OpenMM's Reference platform in
//! `tests/cutoff.rs`.

use crate::energy::{add_row, bonded_terms, check_coords, diff, mix, norm, plain_pair, MmEnergy};
use crate::topology::MmTopology;
use ferric_core::FerricError;
use ndarray::Array2;
use std::collections::HashMap;

/// OpenMM's default `NonbondedForce` reaction-field dielectric.
pub const OPENMM_DEFAULT_REACTION_FIELD_DIELECTRIC: f64 = 78.3;

/// Parameters of the non-periodic reaction-field cutoff (atomic units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutoffParams {
    /// Cutoff distance `r_c` (Bohr), `> 0`.
    pub cutoff: f64,
    /// Reaction-field dielectric `eps_rf`, `> 0` (OpenMM default 78.3;
    /// `1.0` makes `k_rf = 0`, a plain Coulomb shifted by `-q_i q_j / r_c`).
    pub reaction_field_dielectric: f64,
    /// LJ switching-function start `r_sw` (Bohr), `0 <= r_sw < r_c`; `None`
    /// truncates LJ at `r_c` with no switching.
    pub switch_distance: Option<f64>,
}

impl CutoffParams {
    /// Cutoff `r_c` (Bohr) with OpenMM's defaults: `eps_rf = 78.3`, no LJ
    /// switching function.
    pub fn new(cutoff: f64) -> Self {
        Self {
            cutoff,
            reaction_field_dielectric: OPENMM_DEFAULT_REACTION_FIELD_DIELECTRIC,
            switch_distance: None,
        }
    }

    /// Set the reaction-field dielectric.
    pub fn with_reaction_field_dielectric(mut self, eps_rf: f64) -> Self {
        self.reaction_field_dielectric = eps_rf;
        self
    }

    /// Switch LJ smoothly to zero between `switch_distance` and the cutoff.
    pub fn with_switch_distance(mut self, switch_distance: f64) -> Self {
        self.switch_distance = Some(switch_distance);
        self
    }

    /// Reaction-field constants `(k_rf, c_rf)` (theory.md 8.1):
    /// `k_rf = (eps-1) / ((2 eps + 1) r_c^3)`, `c_rf = 3 eps / ((2 eps + 1) r_c)`.
    pub fn reaction_field_constants(&self) -> (f64, f64) {
        let eps = self.reaction_field_dielectric;
        let rc = self.cutoff;
        let k_rf = (eps - 1.0) / ((2.0 * eps + 1.0) * rc * rc * rc);
        let c_rf = 3.0 * eps / ((2.0 * eps + 1.0) * rc);
        (k_rf, c_rf)
    }

    fn validate(&self) -> Result<(), FerricError> {
        if !(self.cutoff.is_finite() && self.cutoff > 0.0) {
            return Err(FerricError::General(format!(
                "CutoffParams: cutoff must be finite and > 0, got {}",
                self.cutoff
            )));
        }
        let eps = self.reaction_field_dielectric;
        if !(eps.is_finite() && eps > 0.0) {
            return Err(FerricError::General(format!(
                "CutoffParams: reaction_field_dielectric must be finite and > 0, got {eps}"
            )));
        }
        if let Some(sw) = self.switch_distance {
            if !(sw.is_finite() && sw >= 0.0 && sw < self.cutoff) {
                return Err(FerricError::General(format!(
                    "CutoffParams: switch_distance must satisfy 0 <= r_sw < cutoff ({}), got {sw}",
                    self.cutoff
                )));
            }
        }
        Ok(())
    }
}

/// How the nonbonded (LJ + Coulomb) sum is evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum NonbondedMethod {
    /// Every non-excluded pair, no cutoff: exactly [`crate::gradient`].
    #[default]
    NoCutoff,
    /// OpenMM `CutoffNonPeriodic`: reaction-field Coulomb and truncated
    /// (optionally switched) LJ within the cutoff; see the module docs.
    CutoffNonPeriodic(CutoffParams),
}

/// Total MM energy under `method`.
pub fn energy_with(
    top: &MmTopology,
    coords: &Array2<f64>,
    method: &NonbondedMethod,
) -> Result<MmEnergy, FerricError> {
    Ok(gradient_with(top, coords, method)?.0)
}

/// Energy and analytic gradient (`dE/dR`, Hartree/Bohr) under `method`.
/// `NonbondedMethod::NoCutoff` returns [`crate::gradient`] unchanged.
pub fn gradient_with(
    top: &MmTopology,
    coords: &Array2<f64>,
    method: &NonbondedMethod,
) -> Result<(MmEnergy, Array2<f64>), FerricError> {
    let params = match method {
        NonbondedMethod::NoCutoff => return crate::energy::gradient(top, coords),
        NonbondedMethod::CutoffNonPeriodic(p) => p,
    };
    params.validate()?;
    check_coords(top, coords)?;
    let (mut e, mut g) = bonded_terms(top, coords);
    nonbonded_cutoff(top, coords, params, &mut e, &mut g);
    Ok((e.sum(), g))
}

/// Every pair `(i, j)`, `i < j`, with `|R_i - R_j| < cutoff`, in
/// lexicographic order, from a cell list with cell edge `cutoff` (each atom
/// is compared only with atoms in its own and the 26 adjacent cells).
pub fn neighbor_pairs(coords: &Array2<f64>, cutoff: f64) -> Vec<(usize, usize)> {
    let n = coords.nrows();
    let mut pairs = Vec::new();
    if n < 2 || !(cutoff > 0.0) {
        return pairs;
    }
    let mut lo = [f64::INFINITY; 3];
    for i in 0..n {
        for c in 0..3 {
            lo[c] = lo[c].min(coords[(i, c)]);
        }
    }
    let cell_of = |i: usize| -> [i64; 3] {
        let mut k = [0i64; 3];
        for c in 0..3 {
            k[c] = ((coords[(i, c)] - lo[c]) / cutoff).floor() as i64;
        }
        k
    };
    let mut cells: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    let mut atom_cell = Vec::with_capacity(n);
    for i in 0..n {
        let k = cell_of(i);
        cells.entry(k).or_default().push(i);
        atom_cell.push(k);
    }
    let rc2 = cutoff * cutoff;
    for i in 0..n {
        let k = atom_cell[i];
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(members) = cells.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) else {
                        continue;
                    };
                    for &j in members {
                        if j <= i {
                            continue;
                        }
                        let d = diff(coords, i, j);
                        if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < rc2 {
                            pairs.push((i, j));
                        }
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs
}

/// Cutoff nonbonded sum (theory.md 8). Pairs are visited in lexicographic
/// `(i, j)` order, the order of the all-pairs loop, so with `r_c` beyond the
/// system diameter, `eps_rf = 1` and no switching the LJ energy and the whole
/// gradient are bit-identical to [`crate::gradient`] and the Coulomb energy
/// differs only by `-c_rf * sum(q_i q_j)` over ordinary pairs
/// (`tests/cutoff.rs`).
fn nonbonded_cutoff(
    top: &MmTopology,
    coords: &Array2<f64>,
    p: &CutoffParams,
    e: &mut MmEnergy,
    g: &mut Array2<f64>,
) {
    let rc = p.cutoff;
    let (k_rf, c_rf) = p.reaction_field_constants();

    // Ordinary candidates within r_c, merged with every 1-4 pair (no cutoff).
    let mut pairs = neighbor_pairs(coords, rc);
    pairs.extend(top.pairs14().iter().copied());
    pairs.sort_unstable();
    pairs.dedup();

    let mut qq_shift = 0.0_f64;
    for (i, j) in pairs {
        let pair = (i, j);
        if top.exclusions().contains(&pair) {
            continue;
        }
        if top.pairs14().contains(&pair) {
            plain_pair(top, coords, i, j, top.scale_lj_14, top.scale_coul_14, e, g);
            continue;
        }

        let rij = diff(coords, i, j);
        let r = norm(rij);
        let unit = [rij[0] / r, rij[1] / r, rij[2] / r];

        let mixed = mix(top.lj[i], top.lj[j]);
        if mixed.epsilon != 0.0 {
            let sr6 = (mixed.sigma / r).powi(6);
            let sr12 = sr6 * sr6;
            let mut e_lj = 4.0 * mixed.epsilon * (sr12 - sr6);
            let mut dedr = 4.0 * mixed.epsilon * (-12.0 * sr12 + 6.0 * sr6) / r;
            if let Some(rsw) = p.switch_distance {
                if r > rsw {
                    let w = rc - rsw;
                    let x = (r - rsw) / w;
                    let x2 = x * x;
                    let s = 1.0 + x2 * x * (-10.0 + x * (15.0 - 6.0 * x));
                    let ds = x2 * (-30.0 + x * (60.0 - 30.0 * x)) / w;
                    dedr = dedr * s + e_lj * ds;
                    e_lj *= s;
                }
            }
            e.lj += e_lj;
            add_row(g, i, [dedr * unit[0], dedr * unit[1], dedr * unit[2]]);
            add_row(g, j, [-dedr * unit[0], -dedr * unit[1], -dedr * unit[2]]);
        }

        let qq = top.charges[i] * top.charges[j];
        if qq != 0.0 {
            e.coulomb += qq / r + qq * k_rf * r * r;
            qq_shift += qq;
            let dedr = -qq / (r * r) + 2.0 * qq * k_rf * r;
            add_row(g, i, [dedr * unit[0], dedr * unit[1], dedr * unit[2]]);
            add_row(g, j, [-dedr * unit[0], -dedr * unit[1], -dedr * unit[2]]);
        }
    }
    e.coulomb -= c_rf * qq_shift;
}
