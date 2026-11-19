//! Shared model potentials for the ferric-md integration tests.
//!
//! All in atomic units: Bohr, Hartree, Hartree/Bohr.
#![allow(dead_code)]

use ferric_core::error::FerricError;
use ferric_md::{FnProvider, MdSystem, AMU_TO_ELECTRON_MASS, AU_TIME_FS};
use ndarray::Array2;

/// Closure type of every model provider here.
pub type EnergyGradient = Box<dyn FnMut(&Array2<f64>) -> Result<(f64, Array2<f64>), FerricError>>;

/// Independent isotropic harmonic wells `E = sum_i k_i |r_i|^2 / 2`.
pub fn harmonic_wells(k: Vec<f64>) -> FnProvider<EnergyGradient> {
    let n = k.len();
    FnProvider {
        n_atoms: n,
        label: "harmonic".into(),
        f: Box::new(move |x: &Array2<f64>| {
            let mut g = x.clone();
            let mut e = 0.0;
            for (mut row, &ki) in g.rows_mut().into_iter().zip(&k) {
                e += 0.5 * ki * row.dot(&row);
                row *= ki;
            }
            Ok((e, g))
        }),
    }
}

/// Sum of a pair potential `u(r) -> (u, du/dr)` over all atom pairs.
pub fn pair_provider(
    n: usize,
    label: &str,
    pair: fn(f64) -> (f64, f64),
) -> FnProvider<EnergyGradient> {
    FnProvider {
        n_atoms: n,
        label: label.into(),
        f: Box::new(move |x: &Array2<f64>| {
            let mut g = Array2::zeros(x.dim());
            let mut e = 0.0;
            for i in 0..x.nrows() {
                for j in (i + 1)..x.nrows() {
                    let d = &x.row(i) - &x.row(j);
                    let r = d.dot(&d).sqrt();
                    let (u, du) = pair(r);
                    e += u;
                    let f = &d * (du / r);
                    g.row_mut(i).scaled_add(1.0, &f);
                    g.row_mut(j).scaled_add(-1.0, &f);
                }
            }
            Ok((e, g))
        }),
    }
}

/// H2-like Morse potential: D = 0.17 Eh, a = 1.04 / Bohr, r_e = 1.4 Bohr.
pub fn morse(r: f64) -> (f64, f64) {
    let (d, a, re) = (0.17, 1.04, 1.4);
    let y = (-a * (r - re)).exp();
    (d * (1.0 - y).powi(2), 2.0 * d * a * y * (1.0 - y))
}

/// Argon-like Lennard-Jones: eps = 3.8e-4 Eh, sigma = 6.43 Bohr.
pub fn lennard_jones(r: f64) -> (f64, f64) {
    let (eps, sigma) = (3.8e-4, 6.43);
    let s6 = (sigma / r).powi(6);
    (
        4.0 * eps * (s6 * s6 - s6),
        4.0 * eps * (-12.0 * s6 * s6 + 6.0 * s6) / r,
    )
}

/// Morse diatomic at a slightly stretched bond, masses of H.
pub fn morse_dimer() -> (MdSystem, FnProvider<EnergyGradient>) {
    let coords = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 0.1, 0.2, 1.5]).unwrap();
    let sys = MdSystem::new(vec!["H".into(), "H".into()], vec![1.008; 2], coords).unwrap();
    (sys, pair_provider(2, "morse", morse))
}

/// Slightly distorted Lennard-Jones argon trimer.
pub fn lj_trimer() -> (MdSystem, FnProvider<EnergyGradient>) {
    let coords =
        Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 7.3, 0.0, 0.2, 3.5, 6.2, -0.1]).unwrap();
    let sys = MdSystem::new(vec!["Ar".into(); 3], vec![39.948; 3], coords).unwrap();
    (sys, pair_provider(3, "lj", lennard_jones))
}

/// One atom per harmonic well, mass 1 amu, placed off-centre.
pub fn harmonic_system(n: usize) -> MdSystem {
    let mut coords = Array2::zeros((n, 3));
    for (i, x) in coords.iter_mut().enumerate() {
        *x = 0.05 * ((i % 7) as f64 - 3.0);
    }
    MdSystem::new(vec!["X".into(); n], vec![1.0; n], coords).unwrap()
}

/// Mass of a 1 amu particle in electron masses (as the driver computes it).
pub fn unit_mass_me() -> f64 {
    AMU_TO_ELECTRON_MASS
}

/// `dt` in a.u. time exactly as the driver computes it.
pub fn dt_au(dt_fs: f64) -> f64 {
    dt_fs / AU_TIME_FS
}

/// Total linear momentum `sum m v` (electron masses x Bohr / a.u. time).
pub fn momentum(masses_amu: &[f64], v: &Array2<f64>) -> [f64; 3] {
    let mut p = [0.0; 3];
    for (row, &m) in v.rows().into_iter().zip(masses_amu) {
        for k in 0..3 {
            p[k] += m * AMU_TO_ELECTRON_MASS * row[k];
        }
    }
    p
}

/// Total angular momentum `sum m r x v` about the origin.
pub fn angular_momentum(masses_amu: &[f64], x: &Array2<f64>, v: &Array2<f64>) -> [f64; 3] {
    let mut l = [0.0; 3];
    for ((r, u), &m) in x.rows().into_iter().zip(v.rows()).zip(masses_amu) {
        let m = m * AMU_TO_ELECTRON_MASS;
        l[0] += m * (r[1] * u[2] - r[2] * u[1]);
        l[1] += m * (r[2] * u[0] - r[0] * u[2]);
        l[2] += m * (r[0] * u[1] - r[1] * u[0]);
    }
    l
}
