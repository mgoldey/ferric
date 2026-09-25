//! Becke fuzzy atomic weights for atomic partition of space.
//!
//! Becke 1988 (J. Chem. Phys. 88, 2547): each atom A gets a smooth weight
//! function `w^A(r) ∈ [0, 1]` of position, with `Σ_A w^A(r) = 1` everywhere.
//! The weight depends only on **geometry** (atom positions and atomic-size
//! radii) — no electron density required, unlike Hirshfeld.
//!
//! Construction:
//!
//! 1. For each pair (A, B): hyperbolic coordinate
//!    `μ_AB(r) = (r_A − r_B) / R_AB` where `r_X = |r − R_X|`.
//! 2. Bragg-Slater size correction: rescale μ by `a_AB` to bias the
//!    boundary toward the smaller atom.
//! 3. Apply Becke smoothing polynomial three times: `f(f(f(μ)))`,
//!    `f(x) = (3 x − x³) / 2`.
//! 4. Cell function `s_AB = (1 − f(f(f(μ_AB)))) / 2`.
//! 5. Cell function `P_A = Π_{B ≠ A} s_AB`.
//! 6. Normalized weight: `w^A = P_A / Σ_B P_B`.

use ferric_core::mol::Molecule;

/// Bragg-Slater atomic radii in Bohr (Z=1..18). Becke 1988 specifically
/// recommends a slight modification (Becke radii) but Bragg-Slater is the
/// common default and equivalent at the chemical-accuracy level needed
/// for atomic partitioning.
pub(crate) fn bragg_slater_bohr(z: i32) -> f64 {
    let r_a: f64 = match z {
        1 => 0.35,
        2 => 0.30,
        3 => 1.45,
        4 => 1.05,
        5 => 0.85,
        6 => 0.70,
        7 => 0.65,
        8 => 0.60,
        9 => 0.50,
        10 => 0.45,
        11 => 1.80,
        12 => 1.50,
        13 => 1.25,
        14 => 1.10,
        15 => 1.00,
        16 => 1.00,
        17 => 1.00,
        18 => 0.71,
        _ => 1.00,
    };
    r_a * 1.8897259886
}

/// Becke smoothing polynomial f(x) = (3x - x³) / 2, applied n times.
/// n = 3 is the standard choice (Becke 1988).
fn becke_smoothing(mu: f64, n_iter: usize) -> f64 {
    let mut x = mu;
    for _ in 0..n_iter {
        x = 0.5 * x * (3.0 - x * x);
    }
    x
}

/// Becke fuzzy weight `w^A(r)` for atom `a_idx` evaluated at position `r`.
///
/// Linear in atom count (Σ_B P_B normalization), quadratic in atom count
/// per atom (Π_{B≠A} s_AB).
pub fn becke_weight(mol: &Molecule, a_idx: usize, r: [f64; 3]) -> f64 {
    let natoms = mol.atoms.len();
    if natoms == 0 {
        return 0.0;
    }
    if natoms == 1 {
        return 1.0;
    }
    // Distances r_X = |r - R_X|.
    let mut r_dists = vec![0.0_f64; natoms];
    for x in 0..natoms {
        let dx = r[0] - mol.atoms[x].x;
        let dy = r[1] - mol.atoms[x].y;
        let dz = r[2] - mol.atoms[x].zpos;
        r_dists[x] = (dx * dx + dy * dy + dz * dz).sqrt();
    }

    // Cell functions P_A = Π_{B ≠ A} s_AB(μ_AB).
    let mut p_cell = vec![1.0_f64; natoms];
    for a in 0..natoms {
        let ra_z = mol.atoms[a].z;
        let r_a_bs = bragg_slater_bohr(ra_z);
        for b in 0..natoms {
            if a == b {
                continue;
            }
            let rb_z = mol.atoms[b].z;
            let r_b_bs = bragg_slater_bohr(rb_z);
            let dx = mol.atoms[a].x - mol.atoms[b].x;
            let dy = mol.atoms[a].y - mol.atoms[b].y;
            let dz = mol.atoms[a].zpos - mol.atoms[b].zpos;
            let r_ab = (dx * dx + dy * dy + dz * dz).sqrt();
            if r_ab < 1e-12 {
                continue; // degenerate; skip
            }
            // Hyperbolic coordinate.
            let mu = (r_dists[a] - r_dists[b]) / r_ab;
            // Bragg-Slater size correction (Becke Eq. A4):
            //   χ = R_A / R_B
            //   u = (χ - 1) / (χ + 1)
            //   a = u / (u² - 1)
            //   |a| ≤ 0.5 clipped
            //   ν = μ + a (1 - μ²)
            let chi = r_a_bs / r_b_bs;
            let u = (chi - 1.0) / (chi + 1.0);
            let a_corr = (u / (u * u - 1.0)).clamp(-0.5, 0.5);
            let nu = mu + a_corr * (1.0 - mu * mu);
            // Apply Becke smoothing 3 times.
            let smoothed = becke_smoothing(nu, 3);
            // Cell function s_AB.
            let s_ab = 0.5 * (1.0 - smoothed);
            p_cell[a] *= s_ab;
        }
    }
    let total: f64 = p_cell.iter().sum();
    if total < 1e-30 {
        return 0.0;
    }
    p_cell[a_idx] / total
}

/// All atom weights at one point (more efficient than calling becke_weight
/// N times, since the cell-function loop is shared).
pub fn becke_weights_all(mol: &Molecule, r: [f64; 3]) -> Vec<f64> {
    let natoms = mol.atoms.len();
    if natoms == 0 {
        return vec![];
    }
    if natoms == 1 {
        return vec![1.0];
    }
    let mut r_dists = vec![0.0_f64; natoms];
    for x in 0..natoms {
        let dx = r[0] - mol.atoms[x].x;
        let dy = r[1] - mol.atoms[x].y;
        let dz = r[2] - mol.atoms[x].zpos;
        r_dists[x] = (dx * dx + dy * dy + dz * dz).sqrt();
    }
    let mut p_cell = vec![1.0_f64; natoms];
    for a in 0..natoms {
        let r_a_bs = bragg_slater_bohr(mol.atoms[a].z);
        for b in 0..natoms {
            if a == b {
                continue;
            }
            let r_b_bs = bragg_slater_bohr(mol.atoms[b].z);
            let dx = mol.atoms[a].x - mol.atoms[b].x;
            let dy = mol.atoms[a].y - mol.atoms[b].y;
            let dz = mol.atoms[a].zpos - mol.atoms[b].zpos;
            let r_ab = (dx * dx + dy * dy + dz * dz).sqrt();
            if r_ab < 1e-12 {
                continue;
            }
            let mu = (r_dists[a] - r_dists[b]) / r_ab;
            let chi = r_a_bs / r_b_bs;
            let u = (chi - 1.0) / (chi + 1.0);
            let a_corr = (u / (u * u - 1.0)).clamp(-0.5, 0.5);
            let nu = mu + a_corr * (1.0 - mu * mu);
            let smoothed = becke_smoothing(nu, 3);
            let s_ab = 0.5 * (1.0 - smoothed);
            p_cell[a] *= s_ab;
        }
    }
    let total: f64 = p_cell.iter().sum();
    if total < 1e-30 {
        return vec![0.0; natoms];
    }
    p_cell.iter().map(|p| p / total).collect()
}

/// Becke fuzzy weights and their **lab-fixed-r** nuclear gradients.
///
/// Returns `(weights, dw)` where:
/// - `weights[a]` is `w^a(r)` (matches `becke_weights_all`)
/// - `dw[a][b][α]` is `∂w^a(r)/∂R_b^α` with `r` held fixed in the lab frame.
///
/// Used by the XC nuclear-gradient grid-response correction (P2.1). The
/// full PySCF "weight1" convention (which includes the home-translation
/// `∇_r w` chain-rule piece) is built on top in
/// `crate::grid::build_atomic_grid_with_response` via the
/// translational-invariance identity `Σ_c ∂w/∂R_c|_{r fixed} + ∇_r w = 0`.
///
/// Cost is O(natoms² + natoms³) per call (the pair caches + the response
/// accumulation); dominated by the AO derivative work in practice.
///
/// Derivation. With `μ_AB = (r_A − r_B) / R_AB`, `r_X = |r − R_X|`,
/// `R_AB = |R_A − R_B|`, the size-corrected coordinate is
/// `ν_AB = μ_AB + a_AB (1 − μ_AB²)` and the smoothed step is
/// `s_AB = (1 − f³(ν_AB)) / 2`. The cell function `P_A = Π_{B≠A} s_AB`,
/// and `w^A = P_A / T` with `T = Σ_C P_C`. Then
///
/// ```text
///   ∂μ_AB/∂R_C^α =
///     +δ_{CA} · [ −(r−R_A)^α / (r_A · R_AB)
///                 − (μ_AB / R_AB²) · (R_A − R_B)^α ]
///     +δ_{CB} · [ +(r−R_B)^α / (r_B · R_AB)
///                 + (μ_AB / R_AB²) · (R_A − R_B)^α ]
///   ∂ν/∂μ = 1 − 2 a_AB μ_AB
///   ∂f³/∂ν = f₂'·f₁'·f₀'  with fₖ' = 1.5·(1 − fₖ²) at the kth iterate.
///   ∂s_AB/∂R_C = −0.5 · (∂f³/∂ν) · (∂ν/∂μ) · ∂μ/∂R_C
/// ```
pub fn becke_weights_and_grad(mol: &Molecule, r: [f64; 3]) -> (Vec<f64>, Vec<Vec<[f64; 3]>>) {
    let natoms = mol.atoms.len();
    if natoms <= 1 {
        let w = if natoms == 1 { vec![1.0] } else { vec![] };
        let dw = vec![vec![[0.0; 3]; natoms]; natoms];
        return (w, dw);
    }

    let mut r_dists = vec![0.0_f64; natoms];
    let mut r_unit = vec![[0.0_f64; 3]; natoms];
    for x in 0..natoms {
        let dx = r[0] - mol.atoms[x].x;
        let dy = r[1] - mol.atoms[x].y;
        let dz = r[2] - mol.atoms[x].zpos;
        let rx = (dx * dx + dy * dy + dz * dz).sqrt();
        r_dists[x] = rx;
        let inv = if rx > 1e-30 { 1.0 / rx } else { 0.0 };
        r_unit[x] = [dx * inv, dy * inv, dz * inv];
    }

    let mut s_pair = vec![1.0_f64; natoms * natoms];
    let mut ds_dra = vec![[0.0_f64; 3]; natoms * natoms];
    let mut ds_drb = vec![[0.0_f64; 3]; natoms * natoms];
    let idx = |a: usize, b: usize| -> usize { a * natoms + b };

    for a in 0..natoms {
        let r_a_bs = bragg_slater_bohr(mol.atoms[a].z);
        for b in 0..natoms {
            if a == b {
                continue;
            }
            let r_b_bs = bragg_slater_bohr(mol.atoms[b].z);
            let dxab = mol.atoms[a].x - mol.atoms[b].x;
            let dyab = mol.atoms[a].y - mol.atoms[b].y;
            let dzab = mol.atoms[a].zpos - mol.atoms[b].zpos;
            let r_ab = (dxab * dxab + dyab * dyab + dzab * dzab).sqrt();
            if r_ab < 1e-12 {
                continue;
            }
            let inv_r_ab = 1.0 / r_ab;
            let r_ab_vec = [dxab, dyab, dzab];

            let mu = (r_dists[a] - r_dists[b]) * inv_r_ab;

            let chi = r_a_bs / r_b_bs;
            let u = (chi - 1.0) / (chi + 1.0);
            let a_corr = (u / (u * u - 1.0)).clamp(-0.5, 0.5);
            let nu = mu + a_corr * (1.0 - mu * mu);
            let dnu_dmu = 1.0 - 2.0 * a_corr * mu;

            let f0 = nu;
            let f0p = 1.5 * (1.0 - f0 * f0);
            let f1 = 0.5 * f0 * (3.0 - f0 * f0);
            let f1p = 1.5 * (1.0 - f1 * f1);
            let f2 = 0.5 * f1 * (3.0 - f1 * f1);
            let f2p = 1.5 * (1.0 - f2 * f2);
            let f3 = 0.5 * f2 * (3.0 - f2 * f2);
            let df3_dnu = f2p * f1p * f0p;

            let s_ab = 0.5 * (1.0 - f3);
            let ds_dmu = -0.5 * df3_dnu * dnu_dmu;

            s_pair[idx(a, b)] = s_ab;

            let mut dmu_dra = [0.0_f64; 3];
            let mut dmu_drb = [0.0_f64; 3];
            for k in 0..3 {
                let ab_term = mu * inv_r_ab * inv_r_ab * r_ab_vec[k];
                dmu_dra[k] = -r_unit[a][k] * inv_r_ab - ab_term;
                dmu_drb[k] = r_unit[b][k] * inv_r_ab + ab_term;
            }
            for k in 0..3 {
                ds_dra[idx(a, b)][k] = ds_dmu * dmu_dra[k];
                ds_drb[idx(a, b)][k] = ds_dmu * dmu_drb[k];
            }
        }
    }

    let mut p_cell = vec![1.0_f64; natoms];
    for a in 0..natoms {
        for b in 0..natoms {
            if a == b {
                continue;
            }
            p_cell[a] *= s_pair[idx(a, b)];
        }
    }
    let t: f64 = p_cell.iter().sum();
    if t < 1e-30 {
        return (vec![0.0; natoms], vec![vec![[0.0; 3]; natoms]; natoms]);
    }
    let inv_t = 1.0 / t;
    let weights: Vec<f64> = p_cell.iter().map(|p| p * inv_t).collect();

    let mut dp = vec![vec![[0.0_f64; 3]; natoms]; natoms];
    for a in 0..natoms {
        for b in 0..natoms {
            if a == b {
                continue;
            }
            let s = s_pair[idx(a, b)];
            if s.abs() < 1e-30 {
                continue;
            }
            let ratio = p_cell[a] / s;
            for k in 0..3 {
                dp[a][a][k] += ratio * ds_dra[idx(a, b)][k];
                dp[a][b][k] += ratio * ds_drb[idx(a, b)][k];
            }
        }
    }

    let mut dt = vec![[0.0_f64; 3]; natoms];
    for a in 0..natoms {
        for c in 0..natoms {
            for k in 0..3 {
                dt[c][k] += dp[a][c][k];
            }
        }
    }

    let mut dw = vec![vec![[0.0_f64; 3]; natoms]; natoms];
    for a in 0..natoms {
        for c in 0..natoms {
            for k in 0..3 {
                dw[a][c][k] = inv_t * dp[a][c][k] - p_cell[a] * inv_t * inv_t * dt[c][k];
            }
        }
    }

    (weights, dw)
}

// ────────────────────────────────────────────────────────────────────────────
// Explicit-neighbour-list partition (periodic grids, Stage 2 PBC)
// ────────────────────────────────────────────────────────────────────────────
//
// Additive: the molecular `becke_weight*` functions above are untouched. The
// periodic grid (`ferric_pbc::dft`) needs the fuzzy-cell weight of a HOME
// atom evaluated over an explicit list of IMAGE atoms (every image within a
// cutoff D of the point), not over `mol.atoms`; this is that kernel. With the
// list = the molecule's atoms and `PartitionScheme::Becke` it is the molecular
// `becke_weights_all` bit for bit (pinned by
// `partition_over_molecule_atoms_is_becke_weights_all`).

/// Smoothing of the cell step function `s(ν) = ½(1 − g(ν))`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PartitionScheme {
    /// Becke 1988: `g = f(f(f(ν)))`, `f(x) = (3x − x³)/2` (ferric's molecular
    /// partition). Tails are polynomial: every atom in the list contributes.
    Becke,
    /// Stratmann–Scuseria–Frisch (CPL 257, 213 (1996)), PySCF `stratmann`:
    /// `g = (35m − 35m³ + 21m⁵ − 5m⁷)/16`, `m = ν/0.64`, and `g = ±1` for
    /// `|ν| ≥ 0.64`. COMPACT support, so a finite image list is exact: the
    /// default for periodic grids (FINDINGS "Iteration 8": periodic SSF ≡
    /// molecular SSF to 1e-16 in a 20+ Bohr box, Becke only algebraically).
    #[default]
    Ssf,
}

impl PartitionScheme {
    /// Strict parse: `"becke"` or `"ssf"` (case-insensitive); anything else
    /// is an error (config honesty — never a silent default).
    pub fn parse_config_str(s: &str) -> Result<Self, ferric_core::error::FerricError> {
        match s.to_ascii_lowercase().as_str() {
            "becke" => Ok(Self::Becke),
            "ssf" | "stratmann" => Ok(Self::Ssf),
            other => Err(ferric_core::error::FerricError::General(format!(
                "partition scheme must be \"becke\" or \"ssf\", got {other:?}"
            ))),
        }
    }
}

/// SSF compact-support half-width `a` (|ν| ≥ a ⇒ s ∈ {0, 1} exactly).
pub const SSF_A: f64 = 0.64;

/// SSF smoothing `g(ν)` (PySCF `gen_grid.stratmann`): odd, `g(±a) = ±1`,
/// `g'(±a) = 0`, and exactly `±1` outside `(−a, a)`.
pub fn ssf_smoothing(nu: f64) -> f64 {
    if nu <= -SSF_A {
        return -1.0;
    }
    if nu >= SSF_A {
        return 1.0;
    }
    let m = nu / SSF_A;
    let m2 = m * m;
    m * (35.0 + m2 * (-35.0 + m2 * (21.0 - 5.0 * m2))) / 16.0
}

/// One atom of an explicit neighbour list (Bohr; `z` selects the Bragg-Slater
/// radius of the size adjustment).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeighbourAtom {
    pub xyz: [f64; 3],
    pub z: i32,
}

/// Fuzzy-cell weight of `atoms[home]` at `r` over the explicit list `atoms`,
/// with the same Becke/Bragg-Slater size adjustment as [`becke_weights_all`]
/// (`ν = μ + a(1 − μ²)`, `|a| ≤ ½`):
///
/// ```text
/// P_B(r) = Π_{C ≠ B} ½(1 − g(ν_BC(r))),    w = P_home / Σ_B P_B
/// ```
///
/// The CALLER chooses the list; for a periodic grid it is every image atom
/// within a cutoff D of `r` (the same list for every home at a given `r`, so
/// the weights stay an exact partition of unity). Size-adjustment caveat for
/// SSF: with `|a| = ½` (e.g. Li–H) `ν → −1 + 4 r_B/R_BC`, so an atom `C` at
/// distance `R_BC` still enters the SSF support while `r_B > 0.09 R_BC` —
/// the exact zone is ~0.09 of the separation, not the 0.18 an unadjusted
/// `ν = μ` would give (measured, FINDINGS "Iteration 8").
///
/// Returns 0 when `home` is out of range or every cell function vanishes.
pub fn partition_weight_over(
    scheme: PartitionScheme,
    atoms: &[NeighbourAtom],
    home: usize,
    r: [f64; 3],
) -> f64 {
    let n = atoms.len();
    if home >= n {
        return 0.0;
    }
    if n == 1 {
        return 1.0;
    }
    let dist = |p: &[f64; 3], q: &[f64; 3]| {
        let dx = p[0] - q[0];
        let dy = p[1] - q[1];
        let dz = p[2] - q[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    let r_d: Vec<f64> = atoms.iter().map(|a| dist(&r, &a.xyz)).collect();
    let radii: Vec<f64> = atoms.iter().map(|a| bragg_slater_bohr(a.z)).collect();
    let mut p_cell = vec![1.0_f64; n];
    for b in 0..n {
        let mut pb = 1.0_f64;
        for c in 0..n {
            if b == c {
                continue;
            }
            let r_bc = dist(&atoms[b].xyz, &atoms[c].xyz);
            if r_bc < 1e-12 {
                continue; // degenerate; skip (as becke_weights_all)
            }
            let mu = (r_d[b] - r_d[c]) / r_bc;
            // Same operation order as becke_weights_all (bit-identity).
            let chi = radii[b] / radii[c];
            let u = (chi - 1.0) / (chi + 1.0);
            let a_corr = (u / (u * u - 1.0)).clamp(-0.5, 0.5);
            let nu = mu + a_corr * (1.0 - mu * mu);
            let g = match scheme {
                PartitionScheme::Becke => becke_smoothing(nu, 3),
                PartitionScheme::Ssf => ssf_smoothing(nu),
            };
            pb *= 0.5 * (1.0 - g);
            if pb == 0.0 {
                break; // every further factor is finite: P_B stays 0
            }
        }
        p_cell[b] = pb;
    }
    let total: f64 = p_cell.iter().sum();
    if total < 1e-30 {
        return 0.0;
    }
    p_cell[home] / total
}

/// `(g(ν), dg/dν)` of the cell-function smoothing (`s = ½(1 − g)`).
fn smoothing_and_slope(scheme: PartitionScheme, nu: f64) -> (f64, f64) {
    match scheme {
        PartitionScheme::Ssf => {
            if nu <= -SSF_A || nu >= SSF_A {
                return (ssf_smoothing(nu), 0.0);
            }
            let m = nu / SSF_A;
            let one = 1.0 - m * m;
            (ssf_smoothing(nu), 35.0 * one * one * one / (16.0 * SSF_A))
        }
        PartitionScheme::Becke => {
            // Chain rule through the three iterates, f'(x) = 1.5 (1 − x²).
            let mut x = nu;
            let mut slope = 1.0;
            for _ in 0..3 {
                slope *= 1.5 * (1.0 - x * x);
                x = 0.5 * x * (3.0 - x * x);
            }
            (x, slope)
        }
    }
}

/// [`partition_weight_over`] and its derivative with respect to the
/// position of EVERY listed atom at FIXED `r`: `(w, dw)` with
/// `dw[k][α] = ∂w/∂X_{k,α}` (lab-fixed point; the caller adds the point's own
/// motion, `∇_r w = −Σ_k dw[k]` by translation invariance, and folds image
/// atoms onto their cell atoms). `w` is bit-identical to
/// [`partition_weight_over`] (same products in the same order).
///
/// With `μ_BC = (d_B − d_C)/R_BC`, `ν = μ + a_BC(1 − μ²)`,
/// `s_BC = ½(1 − g(ν))`, `u_B = (r − X_B)/d_B`, `e_BC = (X_B − X_C)/R_BC`:
///
/// ```text
/// ∂μ_BC/∂X_B = −(u_B + μ_BC e_BC)/R_BC,   ∂μ_BC/∂X_C = (u_C + μ_BC e_BC)/R_BC
/// ∂s_BC/∂μ   = −½ g'(ν)(1 − 2 a_BC μ)
/// ∂P_B/∂X    = Σ_C (∂s_BC/∂μ) (Π_{C'≠C} s_BC') ∂μ_BC/∂X
/// ∂w/∂X      = (∂P_home/∂X − w Σ_B ∂P_B/∂X) / Σ_B P_B
/// ```
///
/// The excluded products `Π_{C'≠C}` are prefix × suffix products, never
/// `P_B / s_BC` — SSF cell functions have EXACT zeros. The `a_BC` size
/// adjustment depends only on the atomic numbers, so it carries no
/// derivative. Changes of the neighbour list itself (the caller's hard
/// distance cutoff) are not differentiable and are ignored.
///
/// Returns `(0, zeros)` when `home` is out of range or every cell function
/// vanishes, and `(1, zeros)` for a one-atom list.
pub fn partition_weight_over_and_grad(
    scheme: PartitionScheme,
    atoms: &[NeighbourAtom],
    home: usize,
    r: [f64; 3],
) -> (f64, Vec<[f64; 3]>) {
    let n = atoms.len();
    let mut dw = vec![[0.0_f64; 3]; n];
    if home >= n {
        return (0.0, dw);
    }
    if n == 1 {
        return (1.0, dw);
    }
    let dist = |p: &[f64; 3], q: &[f64; 3]| {
        let dx = p[0] - q[0];
        let dy = p[1] - q[1];
        let dz = p[2] - q[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    let r_d: Vec<f64> = atoms.iter().map(|a| dist(&r, &a.xyz)).collect();
    let u: Vec<[f64; 3]> = atoms
        .iter()
        .zip(&r_d)
        .map(|(a, &d)| {
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            [
                (r[0] - a.xyz[0]) * inv,
                (r[1] - a.xyz[1]) * inv,
                (r[2] - a.xyz[2]) * inv,
            ]
        })
        .collect();
    let radii: Vec<f64> = atoms.iter().map(|a| bragg_slater_bohr(a.z)).collect();
    let mut p_cell = vec![1.0_f64; n];
    // Σ_B ∂P_B/∂X_D and ∂P_home/∂X_D.
    let mut dsum = vec![[0.0_f64; 3]; n];
    let mut dhome = vec![[0.0_f64; 3]; n];
    // Per-B scratch over C.
    let mut s = vec![1.0_f64; n];
    let mut q = vec![0.0_f64; n];
    let mut mu = vec![0.0_f64; n];
    let mut rbc = vec![1.0_f64; n];
    let mut pre = vec![1.0_f64; n];
    for b in 0..n {
        let mut pb = 1.0_f64;
        let mut n_zero = 0usize;
        for c in 0..n {
            s[c] = 1.0;
            q[c] = 0.0;
            mu[c] = 0.0;
            rbc[c] = 1.0;
            if b == c {
                continue;
            }
            let r_bc = dist(&atoms[b].xyz, &atoms[c].xyz);
            if r_bc < 1e-12 {
                continue; // degenerate; skip (as partition_weight_over)
            }
            let m = (r_d[b] - r_d[c]) / r_bc;
            // Same operation order as partition_weight_over (bit-identity of w).
            let chi = radii[b] / radii[c];
            let uu = (chi - 1.0) / (chi + 1.0);
            let a_corr = (uu / (uu * uu - 1.0)).clamp(-0.5, 0.5);
            let nu = m + a_corr * (1.0 - m * m);
            let (g, gp) = smoothing_and_slope(scheme, nu);
            let sbc = 0.5 * (1.0 - g);
            pb *= sbc;
            s[c] = sbc;
            q[c] = -0.5 * gp * (1.0 - 2.0 * a_corr * m);
            mu[c] = m;
            rbc[c] = r_bc;
            if sbc == 0.0 {
                n_zero += 1;
            }
        }
        p_cell[b] = pb;
        // Two or more exact zeros: every excluded product vanishes.
        if n_zero >= 2 {
            continue;
        }
        // Prefix products; the suffix is accumulated in the backward sweep.
        let mut acc = 1.0_f64;
        for c in 0..n {
            pre[c] = acc;
            acc *= s[c];
        }
        let mut suf = 1.0_f64;
        for c in (0..n).rev() {
            let excl = pre[c] * suf;
            suf *= s[c];
            if q[c] == 0.0 || excl == 0.0 {
                continue;
            }
            let coef = q[c] * excl / rbc[c];
            let inv = 1.0 / rbc[c];
            for k in 0..3 {
                let e = (atoms[b].xyz[k] - atoms[c].xyz[k]) * inv;
                let me = mu[c] * e;
                let d_b = -coef * (u[b][k] + me);
                let d_c = coef * (u[c][k] + me);
                dsum[b][k] += d_b;
                dsum[c][k] += d_c;
                if b == home {
                    dhome[b][k] += d_b;
                    dhome[c][k] += d_c;
                }
            }
        }
    }
    let total: f64 = p_cell.iter().sum();
    if total < 1e-30 {
        return (0.0, dw);
    }
    let w = p_cell[home] / total;
    for d in 0..n {
        for k in 0..3 {
            dw[d][k] = (dhome[d][k] - w * dsum[d][k]) / total;
        }
    }
    (w, dw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::mol::{Atom, Molecule};

    fn h2_at(d: f64) -> Molecule {
        Molecule {
            atoms: vec![
                Atom {
                    symbol: "H".into(),
                    z: 1,
                    x: -d / 2.0,
                    y: 0.0,
                    zpos: 0.0,
                    ghost: false,
                    n_core_ecp: 0,
                },
                Atom {
                    symbol: "H".into(),
                    z: 1,
                    x: d / 2.0,
                    y: 0.0,
                    zpos: 0.0,
                    ghost: false,
                    n_core_ecp: 0,
                },
            ],
            charge: 0,
            multiplicity: 1,
        }
    }

    #[test]
    fn becke_weights_sum_to_one_h2() {
        let mol = h2_at(1.4);
        // Sample at several positions; weights must sum to 1.
        for r in &[
            [0.0, 0.0, 0.0],
            [-0.7, 0.0, 0.0],
            [0.7, 0.0, 0.0],
            [0.0, 1.0, 0.5],
            [2.0, 2.0, 2.0],
        ] {
            let w = becke_weights_all(&mol, *r);
            let sum: f64 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "Becke weights at {:?}: sum {sum}, weights {:?}",
                r,
                w
            );
        }
    }

    #[test]
    fn becke_weight_centered_at_atom_is_one() {
        let mol = h2_at(1.4);
        // At atom A, w_A → 1 (smoothing function saturates at boundary).
        let w0 = becke_weights_all(&mol, [-0.7, 0.0, 0.0]);
        assert!(
            (w0[0] - 1.0).abs() < 1e-6,
            "Becke w_A at R_A = {} (expect 1)",
            w0[0]
        );
        let w1 = becke_weights_all(&mol, [0.7, 0.0, 0.0]);
        assert!(
            (w1[1] - 1.0).abs() < 1e-6,
            "Becke w_B at R_B = {} (expect 1)",
            w1[1]
        );
    }

    #[test]
    fn becke_weight_midpoint_h2_is_half() {
        let mol = h2_at(1.4);
        // At midpoint of homonuclear H2, both weights are 1/2.
        let w = becke_weights_all(&mol, [0.0, 0.0, 0.0]);
        assert!((w[0] - 0.5).abs() < 1e-12, "Becke w midpoint H2: {:?}", w);
        assert!((w[1] - 0.5).abs() < 1e-12, "Becke w midpoint H2: {:?}", w);
    }

    #[test]
    fn becke_weights_and_grad_match_value() {
        let mol = h2_at(1.4);
        let r = [0.3, 0.4, 0.5];
        let w_val = becke_weights_all(&mol, r);
        let (w_g, _) = becke_weights_and_grad(&mol, r);
        for a in 0..2 {
            assert!((w_val[a] - w_g[a]).abs() < 1e-14);
        }
    }

    #[test]
    fn becke_weights_grad_sums_to_zero_h2() {
        let mol = h2_at(1.4);
        for r in &[[0.3, 0.4, 0.5_f64], [-0.2, 0.6, 0.0], [0.7, -0.1, 0.4]] {
            let (_, dw) = becke_weights_and_grad(&mol, *r);
            for c in 0..2 {
                for k in 0..3 {
                    let s: f64 = (0..2).map(|a| dw[a][c][k]).sum();
                    assert!(s.abs() < 1e-12, "Σ_A dw^A/dR_{c}^{k} = {s:.3e} at r={r:?}");
                }
            }
        }
    }

    #[test]
    fn becke_weights_grad_finite_difference_ch() {
        let r = [0.4_f64, 0.3, 0.2];
        let h = 1e-5;

        let build = |atoms: Vec<Atom>| Molecule {
            atoms,
            charge: 0,
            multiplicity: 1,
        };

        let base = vec![
            Atom {
                symbol: "C".into(),
                z: 6,
                x: 0.0,
                y: 0.0,
                zpos: 0.0,
                ghost: false,
                n_core_ecp: 0,
            },
            Atom {
                symbol: "H".into(),
                z: 1,
                x: 2.0,
                y: 0.0,
                zpos: 0.0,
                ghost: false,
                n_core_ecp: 0,
            },
        ];
        let mol = build(base.clone());
        let (_, dw_ana) = becke_weights_and_grad(&mol, r);

        let mut max_err: f64 = 0.0;
        for c in 0..2 {
            for k in 0..3 {
                let mut atoms_plus = base.clone();
                let mut atoms_minus = base.clone();
                match k {
                    0 => {
                        atoms_plus[c].x += h;
                        atoms_minus[c].x -= h;
                    }
                    1 => {
                        atoms_plus[c].y += h;
                        atoms_minus[c].y -= h;
                    }
                    _ => {
                        atoms_plus[c].zpos += h;
                        atoms_minus[c].zpos -= h;
                    }
                }
                let w_plus = becke_weights_all(&build(atoms_plus), r);
                let w_minus = becke_weights_all(&build(atoms_minus), r);
                for a in 0..2 {
                    let fd = (w_plus[a] - w_minus[a]) / (2.0 * h);
                    let ana = dw_ana[a][c][k];
                    let diff = (fd - ana).abs();
                    if diff > max_err {
                        max_err = diff;
                    }
                }
            }
        }
        eprintln!("Becke grad max |ana − FD| = {max_err:.3e}");
        assert!(max_err < 1e-7, "Becke grad FD mismatch = {max_err:.3e}");
    }

    #[test]
    fn becke_size_correction_biases_toward_smaller_atom() {
        // CH₄-like atom pair: C and H. The boundary should be shifted
        // toward the smaller H (R_BS: C=0.70 Å, H=0.35 Å).
        let mol = Molecule {
            atoms: vec![
                Atom {
                    symbol: "C".into(),
                    z: 6,
                    x: 0.0,
                    y: 0.0,
                    zpos: 0.0,
                    ghost: false,
                    n_core_ecp: 0,
                },
                Atom {
                    symbol: "H".into(),
                    z: 1,
                    x: 2.0,
                    y: 0.0,
                    zpos: 0.0,
                    ghost: false,
                    n_core_ecp: 0,
                },
            ],
            charge: 0,
            multiplicity: 1,
        };
        // Midpoint: x=1.0. Without size correction this would give w_C = w_H = 0.5.
        // With size correction toward smaller H, w_C should exceed 0.5.
        let w = becke_weights_all(&mol, [1.0, 0.0, 0.0]);
        assert!(
            w[0] > 0.5,
            "C-H midpoint Becke: w_C should exceed 0.5 from size correction, got w_C={}",
            w[0]
        );
    }
}

#[cfg(test)]
mod partition_over_tests {
    use super::*;
    use ferric_core::mol::{Atom, Molecule};

    fn atom(z: i32, sym: &str, p: [f64; 3]) -> Atom {
        Atom {
            symbol: sym.into(),
            z,
            x: p[0],
            y: p[1],
            zpos: p[2],
            ghost: false,
            n_core_ecp: 0,
        }
    }

    fn lih_ch() -> Molecule {
        Molecule {
            atoms: vec![
                atom(3, "Li", [0.3, 0.2, 0.1]),
                atom(1, "H", [0.3, 0.2, 3.1]),
                atom(6, "C", [1.9, -0.4, 1.2]),
            ],
            charge: 0,
            multiplicity: 1,
        }
    }

    fn list(mol: &Molecule) -> Vec<NeighbourAtom> {
        mol.atoms
            .iter()
            .map(|a| NeighbourAtom {
                xyz: [a.x, a.y, a.zpos],
                z: a.z,
            })
            .collect()
    }

    /// The list kernel with the molecule's atoms and Becke smoothing IS the
    /// molecular `becke_weights_all` (heteronuclear: size adjustment active).
    /// Mutation: a different ν/size-adjust formula in the list kernel fails.
    #[test]
    fn partition_over_molecule_atoms_is_becke_weights_all() {
        let mol = lih_ch();
        let nb = list(&mol);
        for r in [
            [0.0, 0.0, 0.0],
            [0.3, 0.2, 1.6],
            [1.0, -0.2, 2.2],
            [-2.0, 1.5, 0.4],
            [0.31, 0.19, 0.12],
        ] {
            let w = becke_weights_all(&mol, r);
            for (a, wa) in w.iter().enumerate() {
                let wl = partition_weight_over(PartitionScheme::Becke, &nb, a, r);
                assert!((wl - wa).abs() <= 1e-15, "atom {a} at {r:?}: {wl} vs {wa}");
            }
        }
    }

    /// SSF polynomial: odd, ±1 with zero slope at ±a, exactly ±1 outside,
    /// and the closed form at m = ½: g = (35/2 − 35/8 + 21/32 − 5/128)/16
    /// = 0.85888671875 (exact in binary).
    #[test]
    fn ssf_polynomial_shape() {
        assert_eq!(ssf_smoothing(0.0), 0.0);
        assert_eq!(ssf_smoothing(SSF_A), 1.0);
        assert_eq!(ssf_smoothing(-SSF_A), -1.0);
        assert_eq!(ssf_smoothing(0.9), 1.0);
        assert_eq!(ssf_smoothing(-0.7), -1.0);
        // Continuity and zero slope at the support edge.
        let e = 1e-6;
        assert!((ssf_smoothing(SSF_A - e) - 1.0).abs() < 1e-9);
        let m = 0.5_f64;
        let want = (35.0 * m - 35.0 * m.powi(3) + 21.0 * m.powi(5) - 5.0 * m.powi(7)) / 16.0;
        assert!((ssf_smoothing(0.5 * SSF_A) - want).abs() < 1e-15);
        assert!((want - 0.858_886_718_75).abs() < 1e-15, "{want}");
        for x in [0.1, 0.33, 0.6] {
            assert!((ssf_smoothing(x) + ssf_smoothing(-x)).abs() < 1e-15);
        }
    }

    /// Both schemes are a partition of unity over any list, and SSF is
    /// exactly 1 at a nucleus whose neighbours are far (compact support).
    #[test]
    fn list_weights_sum_to_one_and_ssf_is_exact_near_a_nucleus() {
        let nb = list(&lih_ch());
        for scheme in [PartitionScheme::Becke, PartitionScheme::Ssf] {
            for r in [[0.0, 0.0, 0.0], [1.0, -0.2, 2.2], [5.0, 4.0, -3.0]] {
                let sum: f64 = (0..nb.len())
                    .map(|a| partition_weight_over(scheme, &nb, a, r))
                    .sum();
                assert!((sum - 1.0).abs() < 1e-14, "{scheme:?} {r:?}: {sum}");
            }
        }
        let w = partition_weight_over(PartitionScheme::Ssf, &nb, 1, [0.3, 0.2, 3.15]);
        assert_eq!(w, 1.0);
        assert_eq!(
            partition_weight_over(PartitionScheme::Ssf, &nb, 7, [0.0; 3]),
            0.0
        );
    }

    /// `partition_weight_over_and_grad`: `w` bit-identical to
    /// `partition_weight_over`; `dw` vs central FD of the atom positions
    /// (both schemes, every home, points inside the SSF switching zone), and
    /// `−Σ_k dw[k]` vs FD of the point position (translation invariance).
    /// Mutation: dropping the `μ e_BC` term or the `(1 − 2aμ)` factor fails
    /// the FD bound by orders (heteronuclear list: `a ≠ 0`).
    #[test]
    fn partition_weight_derivative_matches_fd() {
        let nb = list(&lih_ch());
        let h = 1e-5;
        let mut worst = 0.0_f64;
        let mut live = 0.0_f64;
        for scheme in [PartitionScheme::Becke, PartitionScheme::Ssf] {
            for r in [[0.9, 0.0, 1.4], [0.6, -0.1, 2.0], [1.2, -0.3, 1.0]] {
                for home in 0..nb.len() {
                    let (w, dw) = partition_weight_over_and_grad(scheme, &nb, home, r);
                    assert_eq!(w, partition_weight_over(scheme, &nb, home, r));
                    let mut dr = [0.0_f64; 3];
                    for (k, dk) in dw.iter().enumerate() {
                        for x in 0..3 {
                            dr[x] -= dk[x];
                            let (mut p, mut m) = (nb.clone(), nb.clone());
                            p[k].xyz[x] += h;
                            m[k].xyz[x] -= h;
                            let fd = (partition_weight_over(scheme, &p, home, r)
                                - partition_weight_over(scheme, &m, home, r))
                                / (2.0 * h);
                            worst = worst.max((fd - dk[x]).abs());
                            live = live.max(fd.abs());
                        }
                    }
                    for x in 0..3 {
                        let (mut rp, mut rm) = (r, r);
                        rp[x] += h;
                        rm[x] -= h;
                        let fd = (partition_weight_over(scheme, &nb, home, rp)
                            - partition_weight_over(scheme, &nb, home, rm))
                            / (2.0 * h);
                        worst = worst.max((fd - dr[x]).abs());
                    }
                }
            }
        }
        eprintln!("partition weight derivative vs FD: {worst:.2e} (max |dw| {live:.2e})");
        assert!(live > 1e-2, "the probe points must lie in a switching zone");
        assert!(worst < 1e-8, "{worst:e}");
    }

    #[test]
    fn partition_scheme_parse_is_strict() {
        assert_eq!(
            PartitionScheme::parse_config_str("SSF").unwrap(),
            PartitionScheme::Ssf
        );
        assert_eq!(
            PartitionScheme::parse_config_str("becke").unwrap(),
            PartitionScheme::Becke
        );
        assert!(PartitionScheme::parse_config_str("hirshfeld").is_err());
    }
}
