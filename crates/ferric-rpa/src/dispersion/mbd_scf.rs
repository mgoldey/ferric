//! MBD@rsSCS evaluated on an SCF density: Hirshfeld volume ratios from the
//! molecular density, the MBD@rsSCS energy, and its nuclear gradient.
//!
//! The volume ratio of atom A is `r_A = v_A / v_A^free`, with
//! `v_A = ∫ w_A ρ |r − R_A|³` from
//! [`atomic_effective_volumes_hirshfeld_on_grid`] on
//! [`hirshfeld_volume_grid`] and `v_A^free` the same integral for the
//! isolated neutral atom ([`live_free_atom_volume`]). Both use free-atom SCFs
//! that [`MbdFreeAtomCache`] solves once per element.
//!
//! The gradient returned by [`mbd_rsscs_for_density`] is
//! ```text
//!   dE/dR_B = ∂E/∂R_B |_{ratios fixed}
//!           + Σ_A c_A ∂v_A/∂R_B |_{D fixed, lattice fixed}
//!           − ½ Tr[V D Sˣ D],     c_A = (∂E/∂r_A) / v_A^free,
//!                                 V = ∂(Σ_A c_A v_A)/∂D
//! ```
//! (the second term from [`hirshfeld_volume_gradient`], the third, which keeps
//! the occupied orbitals orthonormal as the basis moves, from
//! [`crate::properties::hirshfeld_volume_density_derivative`] and the overlap
//! derivative). Together they are the derivative with the occupied orbitals
//! held fixed. NOT included: the orbital relaxation (the CPKS response of D to
//! the displacement) and the motion of the integration lattice with the
//! molecule. Measured against FD of the full pipeline (6-31G, PBE): the
//! relaxation is 1.0e-5 Hartree/Bohr for H2O and 1.6e-6 for NH3, the rest
//! agrees to 3e-8 (`tests/mbd_scf_gradient.rs`).

use std::collections::BTreeMap;

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::properties::{proatom_ground_state_mult, scf_proatom_provider, RadialProatom};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

use crate::dispersion::mbd_rsscs::{
    mbd_rsscs_energy, mbd_rsscs_gradient, MbdRsscsConfig, MbdRsscsResult,
};
use crate::properties::{
    atomic_effective_volumes_hirshfeld, atomic_effective_volumes_hirshfeld_on_grid,
    hirshfeld_volume_gradient, hirshfeld_volume_grid, ProatomProvider,
};

/// Free-atom TS volume `v_free` of neutral element `z` in basis `bs` — the
/// denominator of the Hirshfeld volume ratio, computed the way ferric-cli's
/// TS C6 path computes it:
///
/// * the isolated atom at the origin, multiplicity
///   [`proatom_ground_state_mult`], solved from a clone of `rhf_config` with
///   `mom_after_iter = 5` for an open shell (0 otherwise), `max_iter` raised
///   to at least 200, and `fractional_occ` for an open shell when `xc` is set;
/// * UHF/UKS for an open shell, RHF/RKS for a singlet, on a one-thread rayon
///   pool; if that solve returns an error, it is retried once with `xc = None`
///   and `fractional_occ = false` (HF/UHF);
/// * `v_free = atomic_effective_volumes_hirshfeld(atom, bs, D_atom, None)[0]`
///   — the single-atom Hirshfeld integral with the Slater proatom weight
///   (1 wherever the Slater ρ⁰ is above the 1e-12 floor).
///
/// As in the CLI, the solve's `converged` flag is not checked (an `Ok` result
/// at `max_iter` is used), and the remaining `rhf_config` fields — including
/// any external potential, solvent or constraints — are passed to the atom
/// unchanged.
///
/// # Errors
///
/// Instead of skipping the element: an unknown element, a basis without it,
/// a failed Schwarz bound, both SCF attempts failing, or a failed volume
/// integral.
pub fn live_free_atom_volume(
    ctx: &ParallelContext,
    z: usize,
    bs: &BasisSet,
    op: Operator,
    rhf_config: &RhfConfig,
) -> Result<f64, FerricError> {
    let zi = z as i32;
    let sym = ferric_core::elements::z_to_symbol(zi).ok_or_else(|| {
        FerricError::General(format!("live_free_atom_volume: unknown element Z={z}"))
    })?;
    let free_xyz = format!("1\n{sym}\n{sym} 0 0 0\n");
    let mult = proatom_ground_state_mult(zi);
    let free_mol = Molecule::parse_xyz(&free_xyz, 0, mult)?;
    let free_obs = PreparedBasis::new(&free_mol, bs)?;
    let free_bounds = SchwarzBounds::compute(op, &free_obs)?;
    let mut free_cfg = rhf_config.clone();
    free_cfg.mom_after_iter = if mult > 1 { 5 } else { 0 };
    free_cfg.max_iter = free_cfg.max_iter.max(200);
    if mult > 1 && free_cfg.xc.is_some() {
        free_cfg.fractional_occ = true;
    }
    let solve_free = |cfg: &RhfConfig| -> Result<Array2<f64>, FerricError> {
        if mult > 1 {
            solve_uhf(ctx, &free_mol, &free_obs, &free_bounds, cfg)
                .map(|r| r.density_total().to_owned())
        } else {
            solve_rhf(ctx, &free_mol, &free_obs, op, &free_bounds, cfg)
                .map(|r| r.density_r().to_owned())
        }
    };
    let solve_with_fallback = || -> Result<Array2<f64>, FerricError> {
        solve_free(&free_cfg).or_else(|first| {
            let mut hf_cfg = free_cfg.clone();
            hf_cfg.xc = None;
            hf_cfg.fractional_occ = false;
            solve_free(&hf_cfg).map_err(|second| {
                FerricError::General(format!(
                    "live_free_atom_volume: free-atom SCF for {sym} (Z={z}, mult={mult}) \
                     failed: {first}; HF/UHF retry failed: {second}"
                ))
            })
        })
    };
    // One-thread pool (inline if it cannot be built), as ferric-cli's
    // `run_serial`: rayon coordination dwarfs a one-atom Fock build.
    let density = match rayon::ThreadPoolBuilder::new().num_threads(1).build() {
        Ok(pool) => pool.install(solve_with_fallback),
        Err(_) => solve_with_fallback(),
    }?;
    let v = atomic_effective_volumes_hirshfeld(&free_mol, bs, &density, None)?;
    let v0 = v.first().copied().ok_or_else(|| {
        FerricError::General(format!(
            "live_free_atom_volume: empty volume vector for Z={z}"
        ))
    })?;
    if !(v0.is_finite() && v0 > 0.0) {
        return Err(FerricError::General(format!(
            "live_free_atom_volume: free-atom volume of Z={z} is {v0}, not finite and > 0"
        )));
    }
    Ok(v0)
}

/// Per-element free-atom data for MBD@rsSCS on SCF densities, built once and
/// reused across geometries: the neutral free-atom proatom from
/// [`scf_proatom_provider`] (the Hirshfeld weight) and the free-atom volume
/// from [`live_free_atom_volume`] (the ratio denominator).
#[derive(Debug, Clone)]
pub struct MbdFreeAtomCache {
    proatoms: BTreeMap<i32, RadialProatom>,
    free_volumes: BTreeMap<usize, f64>,
}

impl MbdFreeAtomCache {
    /// Solve each distinct element of `mol` once (two free-atom SCFs per
    /// element: the proatom provider's and [`live_free_atom_volume`]'s).
    ///
    /// # Errors
    ///
    /// Any element whose proatom provider returns `None` (free-atom SCF
    /// failed or did not converge) or whose free volume fails. Nothing falls
    /// back to a Slater proatom or a table volume here.
    pub fn build(
        ctx: &ParallelContext,
        mol: &Molecule,
        bs: &BasisSet,
        op: Operator,
        rhf_config: &RhfConfig,
    ) -> Result<Self, FerricError> {
        let provider = scf_proatom_provider(ctx, bs, op, rhf_config);
        let mut proatoms = BTreeMap::new();
        let mut free_volumes = BTreeMap::new();
        for atom in &mol.atoms {
            let z = atom.z;
            if proatoms.contains_key(&z) {
                continue;
            }
            let pa = provider(z, 0).ok_or_else(|| {
                FerricError::General(format!(
                    "MbdFreeAtomCache: no free-atom proatom for {} (Z={z}): its free-atom \
                     SCF failed or did not converge",
                    atom.symbol
                ))
            })?;
            let zu = usize::try_from(z)
                .map_err(|_| FerricError::General(format!("MbdFreeAtomCache: invalid Z={z}")))?;
            let vf = live_free_atom_volume(ctx, zu, bs, op, rhf_config)?;
            proatoms.insert(z, pa);
            free_volumes.insert(zu, vf);
        }
        Ok(Self {
            proatoms,
            free_volumes,
        })
    }

    /// Proatom provider over the cached elements: neutral (`q == 0`) only;
    /// `None` for any other charge state or an element not in the cache. The
    /// Hirshfeld routines substitute the Slater proatom for `None`, so for a
    /// molecule whose elements were all cached by [`build`](Self::build) that
    /// fallback never fires.
    pub fn proatom(&self) -> impl Fn(i32, i32) -> Option<RadialProatom> + '_ {
        move |z: i32, q: i32| {
            if q != 0 {
                return None;
            }
            self.proatoms.get(&z).cloned()
        }
    }

    /// Cached free-atom volume of element `z` (Bohr³).
    pub fn free_volume(&self, z: usize) -> Option<f64> {
        self.free_volumes.get(&z).copied()
    }
}

/// MBD@rsSCS on an SCF density (see the module doc).
#[derive(Debug, Clone)]
pub struct MbdScfResult {
    /// MBD@rsSCS dispersion energy (Hartree).
    pub energy: f64,
    /// Hirshfeld effective volumes v_A (Bohr³).
    pub volumes: Vec<f64>,
    /// Free-atom volumes v_A^free (Bohr³), per atom.
    pub free_volumes: Vec<f64>,
    /// r_A = v_A / v_A^free.
    pub volume_ratios: Vec<f64>,
    /// The full MBD@rsSCS result.
    pub rsscs: MbdRsscsResult,
    /// Full analytic gradient (term 1 + term 2), (natoms,3), Hartree/Bohr.
    pub gradient: Option<Array2<f64>>,
    /// Term 1 alone (ratios fixed), for diagnostics/tests.
    pub gradient_fixed_ratios: Option<Array2<f64>>,
    /// Volume term at fixed AO density matrix (AOs and proatoms follow the
    /// atoms), for diagnostics/tests.
    pub gradient_volume_fixed_d: Option<Array2<f64>>,
    /// Orbital-orthonormality term −½ Tr[V D S^x D], V = ∂(Σ c_A v_A)/∂D,
    /// for diagnostics/tests.
    pub gradient_orthonormality: Option<Array2<f64>>,
}

/// MBD@rsSCS energy (and, with `want_gradient`, nuclear gradient) for the
/// total AO density `density_total` of `mol` in `bs`.
///
/// Volumes: [`atomic_effective_volumes_hirshfeld_on_grid`] on
/// [`hirshfeld_volume_grid`]`(mol)` with `cache`'s proatoms; ratios against
/// `cache`'s free volumes. Gradient: `∂E/∂R|_ratios` from
/// [`mbd_rsscs_gradient`] plus [`hirshfeld_volume_gradient`] contracted with
/// `de_dv_A = (∂E/∂r_A) / v_A^free` on the same lattice, plus the
/// orthonormality term −½ Tr[V D Sˣ D]. Not included: the
/// orbital relaxation of D and the lattice following the molecule (see the
/// module doc). The gradient requires a closed-shell density (checked:
/// D S D = 2D); `want_gradient` on an open-shell density is an error.
///
/// # Errors
///
/// An element of `mol` missing from `cache`, a non-positive volume, and any
/// error of the volume, MBD or gradient routines.
pub fn mbd_rsscs_for_density(
    cache: &MbdFreeAtomCache,
    mol: &Molecule,
    bs: &BasisSet,
    density_total: &Array2<f64>,
    config: &MbdRsscsConfig,
    want_gradient: bool,
) -> Result<MbdScfResult, FerricError> {
    let natoms = mol.atoms.len();
    let mut z = Vec::with_capacity(natoms);
    let mut free_volumes = Vec::with_capacity(natoms);
    for (i, atom) in mol.atoms.iter().enumerate() {
        let zu = usize::try_from(atom.z).map_err(|_| {
            FerricError::General(format!("mbd_rsscs_for_density: invalid Z={}", atom.z))
        })?;
        if !cache.proatoms.contains_key(&atom.z) {
            return Err(FerricError::General(format!(
                "mbd_rsscs_for_density: atom {i} ({}, Z={zu}) has no cached proatom; \
                 build the MbdFreeAtomCache from a molecule containing every element",
                atom.symbol
            )));
        }
        let vf = cache.free_volume(zu).ok_or_else(|| {
            FerricError::General(format!(
                "mbd_rsscs_for_density: atom {i} ({}, Z={zu}) has no cached free-atom volume",
                atom.symbol
            ))
        })?;
        z.push(zu);
        free_volumes.push(vf);
    }
    let positions: Vec<[f64; 3]> = mol.atoms.iter().map(|a| [a.x, a.y, a.zpos]).collect();

    let grid = hirshfeld_volume_grid(mol);
    let provider = cache.proatom();
    let provider_ref: &ProatomProvider = &provider;
    let volumes = atomic_effective_volumes_hirshfeld_on_grid(
        mol,
        bs,
        density_total,
        Some(provider_ref),
        &grid,
    )?;
    let volume_ratios: Vec<f64> = volumes
        .iter()
        .zip(&free_volumes)
        .map(|(v, vf)| v / vf)
        .collect();

    if !want_gradient {
        let rsscs = mbd_rsscs_energy(&z, &positions, &volume_ratios, config)?;
        return Ok(MbdScfResult {
            energy: rsscs.energy,
            volumes,
            free_volumes,
            volume_ratios,
            rsscs,
            gradient: None,
            gradient_fixed_ratios: None,
            gradient_volume_fixed_d: None,
            gradient_orthonormality: None,
        });
    }

    let (g, de_dr) = mbd_rsscs_gradient(&z, &positions, &volume_ratios, config)?;
    if g.d_positions.len() != natoms || de_dr.len() != natoms {
        return Err(FerricError::General(format!(
            "mbd_rsscs_for_density: MBD gradient returned {} positions / {} ratio \
             derivatives for {natoms} atoms",
            g.d_positions.len(),
            de_dr.len()
        )));
    }
    let mut fixed = Array2::<f64>::zeros((natoms, 3));
    for (a, row) in g.d_positions.iter().enumerate() {
        for k in 0..3 {
            fixed[(a, k)] = row[k];
        }
    }
    let de_dv: Vec<f64> = de_dr
        .iter()
        .zip(&free_volumes)
        .map(|(d, vf)| d / vf)
        .collect();
    let term2 =
        hirshfeld_volume_gradient(mol, bs, density_total, Some(provider_ref), &grid, &de_dv)?;
    let orth = orthonormality_term(mol, bs, density_total, provider_ref, &grid, &de_dv)?;
    let full = &fixed + &term2 + &orth;
    Ok(MbdScfResult {
        energy: g.result.energy,
        volumes,
        free_volumes,
        volume_ratios,
        rsscs: g.result,
        gradient: Some(full),
        gradient_fixed_ratios: Some(fixed),
        gradient_volume_fixed_d: Some(term2),
        gradient_orthonormality: Some(orth),
    })
}

/// −½ Tr[V D S^x D]: the change of Σ_A c_A v_A from keeping the occupied
/// orbitals orthonormal as the basis moves (closed shell, D = 2 C_occ C_occᵀ,
/// dD = −½ D S^x D at fixed orbital rotation).
fn orthonormality_term(
    mol: &Molecule,
    bs: &BasisSet,
    density_total: &Array2<f64>,
    provider: &ProatomProvider,
    grid: &ferric_integrals::ao_grid::GridSpec,
    de_dv: &[f64],
) -> Result<Array2<f64>, FerricError> {
    let prep = PreparedBasis::new(mol, bs)?;
    // Closed-shell check: D S D = 2 D. An open-shell total density fails it,
    // and the term above is then wrong (it needs the spin densities).
    let s = ferric_integrals::oneelectron::overlap(&prep);
    let dsd = density_total.dot(&s).dot(density_total);
    let scale = density_total
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(1.0);
    let resid = (&dsd - &(density_total * 2.0))
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    if resid > 1e-6 * scale {
        return Err(FerricError::General(format!(
            "mbd_rsscs_for_density: the MBD nuclear gradient needs a closed-shell \
             (RKS) density; max|D S D - 2D| = {resid:.3e}"
        )));
    }
    let v = crate::properties::hirshfeld_volume_density_derivative(
        mol,
        bs,
        Some(provider),
        grid,
        de_dv,
    )?;
    let dvd = density_total.dot(&v).dot(density_total);
    let g = ferric_scf::gradient::overlap_deriv_contract(&prep, &dvd)?;
    Ok(g * -0.5)
}
