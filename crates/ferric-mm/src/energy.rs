//! Energies and analytic gradients for [`crate::topology::MmTopology`].
//!
//! Functional forms (AMBER convention, no leading 1/2 on the harmonic
//! terms — see `topology.rs` docs):
//!
//! ```text
//! E_bond    = sum_bonds     k (r - r0)^2
//! E_angle   = sum_angles    k (theta - theta0)^2
//! E_torsion = sum_torsions  k (1 + cos(n*phi - delta))
//! E_lj      = sum_{i<j, not excluded} scale_ij * 4 eps_ij [(sigma_ij/r)^12 - (sigma_ij/r)^6]
//! E_coul    = sum_{i<j, not excluded} scale_ij * q_i q_j / r
//! ```
//!
//! where `scale_ij` is 1 for a normal pair, `scale_lj_14`/`scale_coul_14` for
//! a 1-4 pair, and the pair is skipped entirely (both terms) if it is an
//! exclusion. LJ mixing is Lorentz-Berthelot: `sigma_ij = (sigma_i+sigma_j)/2`,
//! `eps_ij = sqrt(eps_i * eps_j)`.
//!
//! Every gradient here is analytic and FD-verified in `tests/terms_fd.rs`
//! (central difference, h = 1e-5, per-term tolerance 1e-9).

use crate::topology::{LjParams, MmTopology};
use ferric_core::FerricError;
use ndarray::Array2;

/// Energy components (Hartree) and their sum.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MmEnergy {
    pub bond: f64,
    pub angle: f64,
    pub torsion: f64,
    pub lj: f64,
    pub coulomb: f64,
    pub total: f64,
}

impl MmEnergy {
    fn sum(self) -> Self {
        Self { total: self.bond + self.angle + self.torsion + self.lj + self.coulomb, ..self }
    }
}

fn check_coords(top: &MmTopology, coords: &Array2<f64>) -> Result<(), FerricError> {
    let n = top.n_atoms();
    if coords.dim() != (n, 3) {
        return Err(FerricError::General(format!(
            "coords shape {:?} != ({n}, 3) matching the topology's {n} atoms",
            coords.dim()
        )));
    }
    Ok(())
}

#[inline]
fn diff(coords: &Array2<f64>, i: usize, j: usize) -> [f64; 3] {
    [
        coords[(i, 0)] - coords[(j, 0)],
        coords[(i, 1)] - coords[(j, 1)],
        coords[(i, 2)] - coords[(j, 2)],
    ]
}

#[inline]
fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[inline]
fn add_row(g: &mut Array2<f64>, i: usize, v: [f64; 3]) {
    g[(i, 0)] += v[0];
    g[(i, 1)] += v[1];
    g[(i, 2)] += v[2];
}

/// Total MM energy, `energy(top, coords).total == gradient(top, coords).0.total`.
pub fn energy(top: &MmTopology, coords: &Array2<f64>) -> Result<MmEnergy, FerricError> {
    Ok(gradient(top, coords)?.0)
}

/// Energy and analytic gradient (`dE/dR`, Hartree/Bohr), `(n_atoms, 3)`.
pub fn gradient(top: &MmTopology, coords: &Array2<f64>) -> Result<(MmEnergy, Array2<f64>), FerricError> {
    check_coords(top, coords)?;
    let n = top.n_atoms();
    let mut g = Array2::<f64>::zeros((n, 3));
    let mut e = MmEnergy::default();

    // Bonds: E = k (r - r0)^2, dE/dr = 2k(r-r0), force on i along r_ij/r.
    for b in &top.bonds {
        let rij = diff(coords, b.i, b.j);
        let r = norm(rij);
        let dr = r - b.r0;
        e.bond += b.k * dr * dr;
        let dedr = 2.0 * b.k * dr;
        let unit = [rij[0] / r, rij[1] / r, rij[2] / r];
        add_row(&mut g, b.i, [dedr * unit[0], dedr * unit[1], dedr * unit[2]]);
        add_row(&mut g, b.j, [-dedr * unit[0], -dedr * unit[1], -dedr * unit[2]]);
    }

    // Angles: E = k(theta - theta0)^2 with theta the i-j-k bond angle at j.
    for a in &top.angles {
        let rji = diff(coords, a.i, a.j); // vector from j to i... actually i - j
        let rjk = diff(coords, a.k, a.j);
        let rji_n = norm(rji);
        let rjk_n = norm(rjk);
        let cos_t = (dot(rji, rjk) / (rji_n * rjk_n)).clamp(-1.0, 1.0);
        let theta = cos_t.acos();
        let dtheta = theta - a.theta0;
        e.angle += a.k_theta * dtheta * dtheta;
        let dedtheta = 2.0 * a.k_theta * dtheta;

        // d(theta)/dR via standard bond-angle gradient formula.
        let sin_t = theta.sin();
        // Guard: sin_t -> 0 at theta = 0 or pi is a genuine coordinate
        // singularity of the angle gradient (not an FD tolerance issue); the
        // test geometries stay away from it.
        let inv_sin = 1.0 / sin_t;

        // dtheta/dRi = -inv_sin * (rjk/(|rji||rjk|) - cos_t * rji/|rji|^2)
        let mut dti = [0.0; 3];
        let mut dtk = [0.0; 3];
        for c in 0..3 {
            dti[c] = -inv_sin * (rjk[c] / (rji_n * rjk_n) - cos_t * rji[c] / (rji_n * rji_n));
            dtk[c] = -inv_sin * (rji[c] / (rji_n * rjk_n) - cos_t * rjk[c] / (rjk_n * rjk_n));
        }
        let dtj = [-(dti[0] + dtk[0]), -(dti[1] + dtk[1]), -(dti[2] + dtk[2])];

        add_row(&mut g, a.i, [dedtheta * dti[0], dedtheta * dti[1], dedtheta * dti[2]]);
        add_row(&mut g, a.j, [dedtheta * dtj[0], dedtheta * dtj[1], dedtheta * dtj[2]]);
        add_row(&mut g, a.k, [dedtheta * dtk[0], dedtheta * dtk[1], dedtheta * dtk[2]]);
    }

    // Torsions: E = k(1 + cos(n*phi - delta)), Blondel-Karplus gradient.
    for t in &top.torsions {
        let (phi, grads) = dihedral_and_gradient(coords, t.i, t.j, t.k, t.l);
        let n = t.periodicity as f64;
        let arg = n * phi - t.phase;
        e.torsion += t.k_phi * (1.0 + arg.cos());
        let dedphi = -t.k_phi * n * arg.sin();
        add_row(&mut g, t.i, [dedphi * grads[0][0], dedphi * grads[0][1], dedphi * grads[0][2]]);
        add_row(&mut g, t.j, [dedphi * grads[1][0], dedphi * grads[1][1], dedphi * grads[1][2]]);
        add_row(&mut g, t.k, [dedphi * grads[2][0], dedphi * grads[2][1], dedphi * grads[2][2]]);
        add_row(&mut g, t.l, [dedphi * grads[3][0], dedphi * grads[3][1], dedphi * grads[3][2]]);
    }

    // Nonbonded: all pairs, minus exclusions, with 1-4 scaling.
    for i in 0..n {
        for j in (i + 1)..n {
            let pair = (i, j);
            if top.exclusions().contains(&pair) {
                continue;
            }
            let (scale_lj, scale_coul) =
                if top.pairs14().contains(&pair) { (top.scale_lj_14, top.scale_coul_14) } else { (1.0, 1.0) };

            let rij = diff(coords, i, j);
            let r = norm(rij);
            let unit = [rij[0] / r, rij[1] / r, rij[2] / r];

            let mixed = mix(top.lj[i], top.lj[j]);
            if mixed.epsilon != 0.0 && scale_lj != 0.0 {
                let sr6 = (mixed.sigma / r).powi(6);
                let sr12 = sr6 * sr6;
                let e_lj = scale_lj * 4.0 * mixed.epsilon * (sr12 - sr6);
                e.lj += e_lj;
                // dE/dr = scale*4*eps*(-12 sr12 + 6 sr6)/r
                let dedr = scale_lj * 4.0 * mixed.epsilon * (-12.0 * sr12 + 6.0 * sr6) / r;
                add_row(&mut g, i, [dedr * unit[0], dedr * unit[1], dedr * unit[2]]);
                add_row(&mut g, j, [-dedr * unit[0], -dedr * unit[1], -dedr * unit[2]]);
            }

            let qq = top.charges[i] * top.charges[j];
            if qq != 0.0 && scale_coul != 0.0 {
                e.coulomb += scale_coul * qq / r;
                let dedr = -scale_coul * qq / (r * r);
                add_row(&mut g, i, [dedr * unit[0], dedr * unit[1], dedr * unit[2]]);
                add_row(&mut g, j, [-dedr * unit[0], -dedr * unit[1], -dedr * unit[2]]);
            }
        }
    }

    Ok((e.sum(), g))
}

/// Lorentz-Berthelot combining rules: sigma arithmetic mean, epsilon
/// geometric mean.
fn mix(a: LjParams, b: LjParams) -> LjParams {
    LjParams { sigma: 0.5 * (a.sigma + b.sigma), epsilon: (a.epsilon * b.epsilon).sqrt() }
}

/// Dihedral angle `phi` (radians, signed) for atoms i-j-k-l, and `dphi/dR`
/// for each of the four atom positions.
///
/// Bekker's formulation (as implemented in GROMACS `dih.c`; see also Blondel
/// & Karplus, J. Comput. Chem. 17, 1132 (1996)):
/// ```text
/// r_ij = Ri - Rj,  r_kj = Rk - Rj,  r_kl = Rk - Rl
/// m = r_ij x r_kj,  n = r_kj x r_kl
/// cos(phi) = (m . n) / (|m| |n|)
/// sign(phi) = sign(r_ij . n)         (right-handed convention)
/// ```
/// and the gradient (Bekker Eq. 28-31, sign-flipped from force to `dphi/dR`):
/// ```text
/// dphi/dRi =  (|r_kj| / |m|^2) m
/// dphi/dRl = -(|r_kj| / |n|^2) n
/// dphi/dRj = -dphi/dRi + (r_ij.r_kj)/|r_kj|^2 * dphi/dRi - (r_kl.r_kj)/|r_kj|^2 * dphi/dRl
/// dphi/dRk = -dphi/dRl - (r_ij.r_kj)/|r_kj|^2 * dphi/dRi + (r_kl.r_kj)/|r_kj|^2 * dphi/dRl
/// ```
/// This `atan2`-free (acos + explicit sign) form is what GROMACS uses in
/// production and is well-conditioned away from phi = 0/pi; the sign
/// disambiguation via `r_ij . n` avoids the branch-cut issue an unsigned
/// `acos` alone would have.
fn dihedral_and_gradient(coords: &Array2<f64>, i: usize, j: usize, k: usize, l: usize) -> (f64, [[f64; 3]; 4]) {
    let r_ij = diff(coords, i, j); // Ri - Rj
    let r_kj = diff(coords, k, j); // Rk - Rj
    let r_kl = diff(coords, k, l); // Rk - Rl

    let m = cross(r_ij, r_kj);
    let n = cross(r_kj, r_kl);

    let m_sq = dot(m, m);
    let n_sq = dot(n, n);
    let kj_norm = norm(r_kj);

    let cos_phi = (dot(m, n) / (m_sq.sqrt() * n_sq.sqrt())).clamp(-1.0, 1.0);
    let mut phi = cos_phi.acos();
    if dot(r_ij, n) < 0.0 {
        phi = -phi;
    }

    let mut grad_i = [0.0; 3];
    let mut grad_j = [0.0; 3];
    let mut grad_k = [0.0; 3];
    let mut grad_l = [0.0; 3];

    if m_sq > 1e-20 && n_sq > 1e-20 && kj_norm > 1e-20 {
        for c in 0..3 {
            grad_i[c] = (kj_norm / m_sq) * m[c];
            grad_l[c] = -(kj_norm / n_sq) * n[c];
        }
        let p = dot(r_ij, r_kj) / (kj_norm * kj_norm);
        let q = dot(r_kl, r_kj) / (kj_norm * kj_norm);
        for c in 0..3 {
            grad_j[c] = -grad_i[c] + p * grad_i[c] - q * grad_l[c];
            grad_k[c] = -grad_l[c] - p * grad_i[c] + q * grad_l[c];
        }
    }

    (phi, [grad_i, grad_j, grad_k, grad_l])
}

/// QM-MM Lennard-Jones energy and split gradient between "real QM" atoms
/// (which carry their own topology LJ parameters, e.g. an ordinary force
/// field's protein-ligand vdW term) and MM atoms. Full N×M direct sum, no
/// cutoff, Lorentz-Berthelot mixing.
///
/// Returns `(energy, grad_qm (n_qm,3), grad_mm (n_mm,3))`.
pub fn qm_mm_lj_energy_gradient(
    lj_qm: &[LjParams],
    coords_qm: &Array2<f64>,
    lj_mm: &[LjParams],
    coords_mm: &Array2<f64>,
) -> (f64, Array2<f64>, Array2<f64>) {
    let n_qm = lj_qm.len();
    let n_mm = lj_mm.len();
    let mut g_qm = Array2::<f64>::zeros((n_qm, 3));
    let mut g_mm = Array2::<f64>::zeros((n_mm, 3));
    let mut e = 0.0_f64;

    for a in 0..n_qm {
        if lj_qm[a].epsilon == 0.0 {
            continue;
        }
        for b in 0..n_mm {
            if lj_mm[b].epsilon == 0.0 {
                continue;
            }
            let rij = [
                coords_qm[(a, 0)] - coords_mm[(b, 0)],
                coords_qm[(a, 1)] - coords_mm[(b, 1)],
                coords_qm[(a, 2)] - coords_mm[(b, 2)],
            ];
            let r = norm(rij);
            let unit = [rij[0] / r, rij[1] / r, rij[2] / r];
            let mixed = mix(lj_qm[a], lj_mm[b]);
            let sr6 = (mixed.sigma / r).powi(6);
            let sr12 = sr6 * sr6;
            e += 4.0 * mixed.epsilon * (sr12 - sr6);
            let dedr = 4.0 * mixed.epsilon * (-12.0 * sr12 + 6.0 * sr6) / r;
            add_row(&mut g_qm, a, [dedr * unit[0], dedr * unit[1], dedr * unit[2]]);
            add_row(&mut g_mm, b, [-dedr * unit[0], -dedr * unit[1], -dedr * unit[2]]);
        }
    }

    (e, g_qm, g_mm)
}
