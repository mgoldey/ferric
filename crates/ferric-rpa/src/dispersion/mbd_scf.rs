//! MBD@rsSCS evaluated on an SCF density: Hirshfeld volume ratios from the
//! molecular density, the MBD@rsSCS energy, and its nuclear gradient.
//!
//! The volume ratio of atom A is `r_A = v_A / v_A^free`, with
//! `v_A = ∫ w_A ρ |r − R_A|³` from
//! [`atomic_effective_volumes_hirshfeld_on_grid`] on
//! [`mbd_volume_grid`] and `v_A^free` the same integral on the same lattice for
//! the isolated neutral atom ([`live_free_atom_volume_with`] with
//! [`FreeAtomVolumeQuadrature::Lattice`], so the ratio is on one integration
//! scale). Both use free-atom SCFs that [`MbdFreeAtomCache`] solves once per
//! element.
//!
//! MBD stays on the lattice while the TS C6 path moved to the atom-centred
//! Becke–Lebedev grid: `mbd_volume_grid`'s points move by exactly 1/N of every
//! atom's displacement, which is what makes the lattice-response term below a
//! single translation-invariance identity. A Becke grid moves WITH the atoms
//! and would need its own weight-derivative (grid-response) term in
//! [`hirshfeld_volume_gradient`], validated by FD — a separate piece of work.
//! The BOUNDING-BOX lattice's measured error against the dense Becke reference
//! is 5.0e-5 to 2.4e-4 relative on the molecular volumes and 1.1e-5 to 1.7e-3
//! on the free-atom ones (`tests/validation_hirshfeld.rs`); the centroid
//! lattice this path uses has not been measured against that reference,
//! because no reference was generated on it. What is measured is that the two
//! lattices agree with each other to 8.7e-8 to 1.8e-4 on the free atoms
//! (`tests/measure_free_atom_quadratures.rs`), so the centroid lattice is of
//! the same accuracy class. Whatever that error is, it is now the SAME error
//! in numerator and denominator and so largely cancels in the ratio — which
//! it did not before this change, when `v_free` came from the bounding-box
//! lattice while the numerators came from the centroid one.
//!
//! The exact nuclear gradient, returned by [`mbd_rsscs_for_scf`], is
//! ```text
//!   dE/dR_B = ∂E/∂R_B |_{ratios fixed}                      (mbd_rsscs_gradient)
//!           + Σ_A c_A ∂v_A/∂R_B |_{D fixed, lattice fixed}  (hirshfeld_volume_gradient)
//!           − ½ Tr[V D Sˣ D]   (UKS: − Σ_σ Tr[V D_σ Sˣ D_σ];   (orthonormality)
//!                               ROKS: − Tr[Sˣ W_Q], see below)
//!           − (1/N) Σ_C Σ_A c_A ∂v_A/∂R_C |_{D, lattice}     (lattice response)
//!           + Σ_ai Z_ai ∂F_ai/∂R_B                           (orbital relaxation)
//!   c_A = (∂E/∂r_A) / v_A^free,   V = ∂(Σ_A c_A v_A)/∂D
//! ```
//! The first four are the derivative with the occupied orbitals held fixed
//! (`gradient_unrelaxed`, also returned by [`mbd_rsscs_for_density`]); the
//! orthonormality term keeps them orthonormal as the basis moves; the lattice
//! term follows [`mbd_volume_grid`], whose points move by 1/N of every atom's
//! displacement (translation invariance at fixed D turns the lattice shift
//! into minus the sum of the fixed-lattice term). The relaxation term is the
//! KS Z-vector of [`ferric_scf::zvector_ks`] (closed-shell for RKS, coupled
//! α/β for UKS, three-block closed/open/virtual for ROKS) with V as its
//! right-hand side; MBD depends on the total density, so both spins see the
//! same V. For ROKS the closed and open orbitals are re-orthonormalized as one
//! set, so W_Q = D_α V D_α + P_c V P_c + ½ (P_c V P_o + P_o V P_c) carries a
//! closed–open cross term.
//!
//! Measured against central FD (h = 1e-3 Bohr) of the full SCF + MBD pipeline
//! at 6-31G (`tests/mbd_scf_gradient.rs`): ≤ 3.5e-9 Hartree/Bohr for H2O with
//! PBE, PBE0, HSE06 and PBE + RI-J; 2.0e-10 for NH3/PBE. The relaxation term
//! alone is 1.0e-5 (H2O) and 1.6e-6 (NH3) and matches FD to 3.5e-9. The
//! proatom interpolant is C2 in r (see `RadialProatom`), so the volumes and
//! the energy are smooth in the nuclear coordinates and the FD step needs no
//! special choice.
//!
//! UKS (`tests/mbd_scf_gradient_uks.rs`, h = 3e-5 Bohr): ≤ 1.9e-9 for NH2,
//! OH and O2 at 6-31G with PBE, PBE0 and HSE06, 6.0e-9 / 5.0e-9 for OH with
//! PBE + RI-J / PBE0 + RI-JK (the Z-vector Hessian uses exact J/K), against a
//! relaxation term of 2.7e-6 to 1.0e-5; an RKS result run through the UKS
//! path reproduces the closed-shell gradient to 3e-14.
//!
//! ROKS (`tests/mbd_scf_gradient_roks.rs`, h = 1e-3 Bohr): ≤ 2.1e-11 for HCO,
//! NH2 (doublets), CH2 and O2 (triplets) at 6-31G with PBE and PBE0, ≤ 4.5e-10
//! with HSE06 and 7.5e-9 for HCO with PBE + RI-J (the Z-vector Hessian uses
//! exact J/K), against a relaxation term of 2.7e-6 to 9.8e-6; an RKS result
//! run through the ROKS path reproduces the closed-shell gradient to 2.7e-14.

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
    hirshfeld_volume_gradient, mbd_volume_grid, ProatomProvider,
};

/// The quadrature a free-atom volume is integrated on.
///
/// The ratio `v_A / v_A^free` is only meaningful when numerator and
/// denominator are on the SAME integration scale (CLAUDE.md's TS/MBD honesty
/// note: a mismatched `vol_free` is what inflated Si's TS C6 by ~2x). Since
/// the TS path and MBD@rsSCS integrate their MOLECULAR volumes differently,
/// the denominator has to be selectable rather than fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreeAtomVolumeQuadrature {
    /// The atom-centred Becke–Lebedev grid of
    /// [`crate::properties::atomic_effective_volumes_hirshfeld`] — what the TS
    /// C6 path integrates its molecular volumes on.
    Becke,
    /// The uniform Cartesian lattice of [`mbd_volume_grid`] — what MBD@rsSCS
    /// integrates its molecular volumes on, because its gradient's
    /// lattice-response term is derived for a lattice.
    ///
    /// On a one-atom molecule the centroid IS the nucleus, so the nucleus sits
    /// exactly on a node, where ρ has its cusp. MEASURED against the
    /// [`Becke`](FreeAtomVolumeQuadrature::Becke) denominator
    /// (`tests/measure_free_atom_quadratures.rs`, cc-pVDZ / def2-SVP):
    /// this lattice puts the free-atom volume 3.33e-4 (H), 2.25e-4 (C) and
    /// 1.06e-5 (O) relative LOW at cc-pVDZ, and 3.34e-4 / 2.93e-4 / 9.96e-6 at
    /// def2-SVP. (The bounding-box lattice, which is what this denominator came
    /// from before the TS path moved, is lower still: 5.12e-4 / 3.24e-4 /
    /// 1.07e-5.) The free-atom Becke value does not depend on the grid size at
    /// all — it is identical from Lebedev 110 to 590 — so these gaps are the
    /// lattice's.
    ///
    /// MBD keeps the lattice anyway, so that error cancels against the same
    /// error in its molecular volumes rather than being added to them, which
    /// is the whole reason this variant exists.
    Lattice,
}

/// Free-atom TS volume `v_free` of neutral element `z` in basis `bs` — the
/// denominator of the Hirshfeld volume ratio, on the [`Becke`
/// quadrature](FreeAtomVolumeQuadrature::Becke) that ferric-cli's TS C6 path
/// uses for its molecular volumes. [`live_free_atom_volume_with`] selects
/// another.
///
/// Computed the way ferric-cli's TS C6 path computes it:
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
/// The free atom is ISOLATED: the molecule's environment (point charges and
/// field, COSMO/PCM solvent, polarizable sites, cDFT constraints) is not
/// applied to it, as for the proatoms of [`scf_proatom_provider`]. Otherwise
/// a QM/MM or solvated run would divide by the volume of an atom polarized by
/// an environment centred on someone else's coordinates. An SCF that returns
/// unconverged counts as failed (it is retried with HF/UHF, then an error).
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
    live_free_atom_volume_with(ctx, z, bs, op, rhf_config, FreeAtomVolumeQuadrature::Becke)
}

/// [`live_free_atom_volume`] with an explicit quadrature for the volume
/// integral, so a caller can match the denominator to the scale its molecular
/// volumes are on. Everything else — the free-atom SCF, the environment
/// stripping, the HF/UHF retry, the one-thread pool — is identical.
pub fn live_free_atom_volume_with(
    ctx: &ParallelContext,
    z: usize,
    bs: &BasisSet,
    op: Operator,
    rhf_config: &RhfConfig,
    quadrature: FreeAtomVolumeQuadrature,
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
    free_cfg.external_potential = None;
    free_cfg.cosmo = None;
    free_cfg.pcm = None;
    free_cfg.polarizable = None;
    free_cfg.constraints.clear();
    free_cfg.mom_after_iter = if mult > 1 { 5 } else { 0 };
    free_cfg.max_iter = free_cfg.max_iter.max(200);
    if mult > 1 && free_cfg.xc.is_some() {
        free_cfg.fractional_occ = true;
    }
    let solve_free = |cfg: &RhfConfig| -> Result<Array2<f64>, FerricError> {
        let r = if mult > 1 {
            solve_uhf(ctx, &free_mol, &free_obs, &free_bounds, cfg)?
        } else {
            solve_rhf(ctx, &free_mol, &free_obs, op, &free_bounds, cfg)?
        };
        if !r.converged {
            return Err(FerricError::Convergence(format!(
                "free-atom SCF for {sym} did not converge in {} iterations",
                cfg.max_iter
            )));
        }
        Ok(r.density_total().to_owned())
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
    let v = match quadrature {
        FreeAtomVolumeQuadrature::Becke => {
            atomic_effective_volumes_hirshfeld(&free_mol, bs, &density, None)?
        }
        FreeAtomVolumeQuadrature::Lattice => atomic_effective_volumes_hirshfeld_on_grid(
            &free_mol,
            bs,
            &density,
            None,
            &mbd_volume_grid(&free_mol),
        )?,
    };
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
            // Lattice, NOT Becke: these free volumes are the denominator of
            // ratios whose numerators `mbd_rsscs_impl` integrates on
            // `mbd_volume_grid`. A Becke denominator against a lattice
            // numerator would make the ratio carry the DIFFERENCE between the
            // two quadratures (~2e-4 relative, per
            // `FreeAtomVolumeQuadrature::Lattice`'s doc) instead of cancelling
            // it.
            let vf = live_free_atom_volume_with(
                ctx,
                zu,
                bs,
                op,
                rhf_config,
                FreeAtomVolumeQuadrature::Lattice,
            )?;
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
    /// The exact analytic nuclear gradient, (natoms,3), Hartree/Bohr:
    /// `gradient_unrelaxed + gradient_relaxation`. Set only by
    /// [`mbd_rsscs_for_scf`], which has the SCF needed for the relaxation term.
    pub gradient: Option<Array2<f64>>,
    /// The gradient with the occupied orbitals held fixed (no orbital
    /// relaxation): fixed-ratio term + fixed-D volume term + orthonormality
    /// term. Set when a gradient was requested.
    pub gradient_unrelaxed: Option<Array2<f64>>,
    /// The orbital-relaxation term Σ Z_ai ∂F_ai/∂R (Z-vector, see
    /// [`ferric_scf::zvector_ks`]). Set only by [`mbd_rsscs_for_scf`].
    pub gradient_relaxation: Option<Array2<f64>>,
    /// Term 1 alone (ratios fixed), for diagnostics/tests.
    pub gradient_fixed_ratios: Option<Array2<f64>>,
    /// Volume term at fixed AO density matrix (AOs and proatoms follow the
    /// atoms), for diagnostics/tests.
    pub gradient_volume_fixed_d: Option<Array2<f64>>,
    /// Orbital-orthonormality term −½ Tr[V D S^x D], V = ∂(Σ c_A v_A)/∂D,
    /// for diagnostics/tests.
    pub gradient_orthonormality: Option<Array2<f64>>,
    /// V = ∂E_MBD/∂D (AO, symmetric) at fixed geometry: the right-hand side
    /// of the Z-vector equation. Set when a gradient was requested.
    pub density_derivative: Option<Array2<f64>>,
}

/// MBD@rsSCS energy (and, with `want_gradient`, nuclear gradient) for the
/// total AO density `density_total` of `mol` in `bs`.
///
/// Volumes: [`atomic_effective_volumes_hirshfeld_on_grid`] on
/// [`mbd_volume_grid`]`(mol)` with `cache`'s proatoms; ratios against
/// `cache`'s free volumes. Gradient: `∂E/∂R|_ratios` from
/// [`mbd_rsscs_gradient`] plus [`hirshfeld_volume_gradient`] contracted with
/// `de_dv_A = (∂E/∂r_A) / v_A^free` on the same lattice, plus the
/// orthonormality and lattice-response terms: `gradient_unrelaxed`, the
/// derivative with the occupied orbitals held fixed (see the module doc).
/// The orbital relaxation needs the SCF: [`mbd_rsscs_for_scf`]. The gradient
/// requires a closed-shell density (checked:
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
    mbd_rsscs_impl(
        cache,
        mol,
        bs,
        density_total,
        OccupiedDensities::Closed,
        config,
        want_gradient,
    )
}

/// [`mbd_rsscs_for_density`] for an open-shell (UKS) reference given its spin
/// densities `d_alpha`, `d_beta` (each D_σ = C_σ,occ C_σ,occᵀ). The volumes,
/// energy and every gradient term but one use the total density
/// D_α + D_β; the orthonormality term keeps each spin's occupied orbitals
/// orthonormal, −Σ_σ Tr[V D_σ Sˣ D_σ] (checked: D_σ S D_σ = D_σ).
///
/// # Errors
///
/// Those of [`mbd_rsscs_for_density`]; with `want_gradient`, spin densities
/// that are not idempotent in the S metric.
pub fn mbd_rsscs_for_spin_densities(
    cache: &MbdFreeAtomCache,
    mol: &Molecule,
    bs: &BasisSet,
    d_alpha: &Array2<f64>,
    d_beta: &Array2<f64>,
    config: &MbdRsscsConfig,
    want_gradient: bool,
) -> Result<MbdScfResult, FerricError> {
    let total = d_alpha + d_beta;
    mbd_rsscs_impl(
        cache,
        mol,
        bs,
        &total,
        OccupiedDensities::Open {
            alpha: d_alpha,
            beta: d_beta,
        },
        config,
        want_gradient,
    )
}

/// [`mbd_rsscs_for_density`] for a restricted open-shell (ROKS) reference
/// given its spin densities `d_alpha` = P_c + P_o and `d_beta` = P_c (one set
/// of spatial orbitals: closed, open). As for UKS, everything but the
/// orthonormality term uses the total density. The closed and open orbitals
/// are re-orthonormalized as ONE set, so the term carries a closed–open cross
/// piece: −Tr\[Sˣ W_Q\], W_Q = D_α V D_α + P_c V P_c + ½ (P_c V P_o + P_o V P_c)
/// (see [`ferric_scf::zvector_ks`], "Restricted open-shell (ROKS)
/// references"). Checked: D_σ S D_σ = D_σ and D_α S D_β = D_β.
///
/// # Errors
///
/// Those of [`mbd_rsscs_for_density`]; with `want_gradient`, spin densities
/// that are not idempotent in the S metric or a β space not inside the α one.
pub fn mbd_rsscs_for_restricted_open_densities(
    cache: &MbdFreeAtomCache,
    mol: &Molecule,
    bs: &BasisSet,
    d_alpha: &Array2<f64>,
    d_beta: &Array2<f64>,
    config: &MbdRsscsConfig,
    want_gradient: bool,
) -> Result<MbdScfResult, FerricError> {
    let total = d_alpha + d_beta;
    mbd_rsscs_impl(
        cache,
        mol,
        bs,
        &total,
        OccupiedDensities::RestrictedOpen {
            alpha: d_alpha,
            beta: d_beta,
        },
        config,
        want_gradient,
    )
}

/// How the occupied orbitals behind the total density are held orthonormal.
#[derive(Clone, Copy)]
enum OccupiedDensities<'a> {
    /// Closed shell: D = 2 C_occ C_occᵀ, the total density.
    Closed,
    /// Open shell: D_σ = C_σ,occ C_σ,occᵀ per spin.
    Open {
        alpha: &'a Array2<f64>,
        beta: &'a Array2<f64>,
    },
    /// Restricted open shell: shared spatial orbitals, D_α = P_c + P_o,
    /// D_β = P_c, closed and open orthonormalized together.
    RestrictedOpen {
        alpha: &'a Array2<f64>,
        beta: &'a Array2<f64>,
    },
}

fn mbd_rsscs_impl(
    cache: &MbdFreeAtomCache,
    mol: &Molecule,
    bs: &BasisSet,
    density_total: &Array2<f64>,
    occupied: OccupiedDensities<'_>,
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

    let grid = mbd_volume_grid(mol);
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
            gradient_unrelaxed: None,
            gradient_relaxation: None,
            gradient_fixed_ratios: None,
            gradient_volume_fixed_d: None,
            gradient_orthonormality: None,
            density_derivative: None,
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
    let (orth, v) = orthonormality_term(
        mol,
        bs,
        density_total,
        occupied,
        provider_ref,
        &grid,
        &de_dv,
    )?;
    // Lattice response: every point of `mbd_volume_grid` moves by 1/N of each
    // atom's displacement, and at fixed D, ∂v/∂(lattice shift) = −Σ_B ∂v/∂R_B.
    let mut lattice = Array2::<f64>::zeros((natoms, 3));
    for k in 0..3 {
        let shift = -term2.column(k).sum() / natoms as f64;
        lattice.column_mut(k).fill(shift);
    }
    let full = &fixed + &term2 + &orth + &lattice;
    Ok(MbdScfResult {
        energy: g.result.energy,
        volumes,
        free_volumes,
        volume_ratios,
        rsscs: g.result,
        gradient: None,
        gradient_unrelaxed: Some(full),
        gradient_relaxation: None,
        gradient_fixed_ratios: Some(fixed),
        gradient_volume_fixed_d: Some(term2),
        gradient_orthonormality: Some(orth),
        density_derivative: Some(v),
    })
}

/// The change of Σ_A c_A v_A from keeping the occupied orbitals orthonormal
/// as the basis moves (dC_occ = −½ C_occ S^x_oo at fixed orbital rotation):
/// closed shell, D = 2 C_occ C_occᵀ, dD = −½ D S^x D, term −½ Tr[V D S^x D];
/// open shell, D_σ = C_σ,occ C_σ,occᵀ, dD_σ = −D_σ S^x D_σ, term
/// −Σ_σ Tr[V D_σ S^x D_σ]; restricted open shell (closed + open
/// orthonormalized as one set), term −Tr[S^x W_Q] with
/// W_Q = D_α V D_α + P_c V P_c + ½ (P_c V P_o + P_o V P_c). Returns the term
/// and V.
fn orthonormality_term(
    mol: &Molecule,
    bs: &BasisSet,
    density_total: &Array2<f64>,
    occupied: OccupiedDensities<'_>,
    provider: &ProatomProvider,
    grid: &ferric_integrals::ao_grid::GridSpec,
    de_dv: &[f64],
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let prep = PreparedBasis::new(mol, bs)?;
    let s = ferric_integrals::oneelectron::overlap(&prep);
    // Projector check: D S D = occ·D (occ = 2 closed shell, 1 per spin). A
    // density that fails it is not what the term above assumes.
    let check = |d: &Array2<f64>, occ: f64, what: &str| -> Result<(), FerricError> {
        let dsd = d.dot(&s).dot(d);
        let scale = d.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
        let resid = (&dsd - &(d * occ))
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        if resid > 1e-6 * scale {
            return Err(FerricError::General(format!(
                "mbd_rsscs_for_density: the MBD nuclear gradient needs {what}; \
                 max|D S D - {occ}D| = {resid:.3e}"
            )));
        }
        Ok(())
    };
    match occupied {
        OccupiedDensities::Closed => check(
            density_total,
            2.0,
            "a closed-shell (RKS) density (use mbd_rsscs_for_spin_densities for UKS)",
        )?,
        OccupiedDensities::Open { alpha, beta } => {
            check(alpha, 1.0, "an idempotent alpha spin density")?;
            check(beta, 1.0, "an idempotent beta spin density")?;
        }
        OccupiedDensities::RestrictedOpen { alpha, beta } => {
            check(alpha, 1.0, "an idempotent alpha spin density")?;
            check(beta, 1.0, "an idempotent beta spin density")?;
            // The closed space inside the occupied one: D_α S D_β = D_β.
            let scale = beta.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
            let resid = (&alpha.dot(&s).dot(beta) - beta)
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs()));
            if resid > 1e-6 * scale {
                return Err(FerricError::General(format!(
                    "mbd_rsscs_for_density: the ROKS MBD nuclear gradient needs the beta \
                     (closed) space inside the alpha one; max|D_a S D_b - D_b| = {resid:.3e}"
                )));
            }
        }
    }
    let v = crate::properties::hirshfeld_volume_density_derivative(
        mol,
        bs,
        Some(provider),
        grid,
        de_dv,
    )?;
    let g = match occupied {
        OccupiedDensities::Closed => {
            let dvd = density_total.dot(&v).dot(density_total);
            ferric_scf::gradient::overlap_deriv_contract(&prep, &dvd)? * -0.5
        }
        OccupiedDensities::Open { alpha, beta } => {
            let dvd = alpha.dot(&v).dot(alpha) + beta.dot(&v).dot(beta);
            -ferric_scf::gradient::overlap_deriv_contract(&prep, &dvd)?
        }
        OccupiedDensities::RestrictedOpen { alpha, beta } => {
            let p_o = alpha - beta;
            let cross = beta.dot(&v).dot(&p_o);
            let w_q =
                alpha.dot(&v).dot(alpha) + beta.dot(&v).dot(beta) + 0.5 * (&cross + &cross.t());
            -ferric_scf::gradient::overlap_deriv_contract(&prep, &w_q)?
        }
    };
    Ok((g, v))
}

/// MBD@rsSCS at the converged KS `result` of `config` with its exact nuclear
/// gradient, dispatching on `result.spin`:
///
/// * `Restricted` (RKS): [`mbd_rsscs_for_density`] plus the orbital-relaxation
///   term from [`ferric_scf::zvector_ks::relaxation_gradient_closed`];
/// * `Unrestricted` (UKS): [`mbd_rsscs_for_spin_densities`] plus the term from
///   [`ferric_scf::zvector_ks::relaxation_gradient_unrestricted`];
/// * `RestrictedOpen` (ROKS): [`mbd_rsscs_for_restricted_open_densities`]
///   plus the term from [`ferric_scf::zvector_ks::relaxation_gradient_roks`].
///
/// `gradient` is then the exact derivative of the energy the SCF + MBD
/// pipeline reports (see the module doc for the terms and their validation).
///
/// # Errors
///
/// Those of the density routines and every reference the Z-vector does not
/// support ([`ferric_scf::zvector_ks::unsupported_reason`],
/// [`ferric_scf::zvector_ks::unsupported_reason_unrestricted`],
/// [`ferric_scf::zvector_ks::unsupported_reason_roks`]): never an unrelaxed
/// gradient presented as the exact one.
#[allow(clippy::too_many_arguments)]
pub fn mbd_rsscs_for_scf(
    ctx: &ParallelContext,
    cache: &MbdFreeAtomCache,
    mol: &Molecule,
    bs: &BasisSet,
    op: Operator,
    rhf_config: &RhfConfig,
    result: &ferric_scf::ScfResult,
    config: &MbdRsscsConfig,
) -> Result<MbdScfResult, FerricError> {
    let unsupported = match result.spin {
        ferric_scf::Spin::Restricted => ferric_scf::zvector_ks::unsupported_reason(rhf_config),
        ferric_scf::Spin::Unrestricted => {
            ferric_scf::zvector_ks::unsupported_reason_unrestricted(rhf_config)
        }
        ferric_scf::Spin::RestrictedOpen => {
            ferric_scf::zvector_ks::unsupported_reason_roks(rhf_config)
        }
    };
    if let Some(r) = unsupported {
        return Err(FerricError::General(format!(
            "MBD@rsSCS nuclear gradient: the orbital-relaxation (Z-vector) term is not \
             available: {r}"
        )));
    }
    let beta = || {
        result.density_beta.as_ref().ok_or_else(|| {
            FerricError::General("mbd_rsscs_for_scf: open-shell result has no beta density".into())
        })
    };
    let mut out = match result.spin {
        ferric_scf::Spin::Unrestricted => mbd_rsscs_for_spin_densities(
            cache,
            mol,
            bs,
            &result.density_alpha,
            beta()?,
            config,
            true,
        )?,
        ferric_scf::Spin::RestrictedOpen => mbd_rsscs_for_restricted_open_densities(
            cache,
            mol,
            bs,
            &result.density_alpha,
            beta()?,
            config,
            true,
        )?,
        ferric_scf::Spin::Restricted => {
            mbd_rsscs_for_density(cache, mol, bs, result.density_r(), config, true)?
        }
    };
    let v = out.density_derivative.as_ref().ok_or_else(|| {
        FerricError::General("mbd_rsscs_for_scf: no density derivative was formed".into())
    })?;
    let prep = PreparedBasis::new(mol, bs)?;
    let bounds = SchwarzBounds::compute_for_screening(op, &prep, rhf_config.screening)?;
    let relax = match result.spin {
        ferric_scf::Spin::Unrestricted => {
            ferric_scf::zvector_ks::relaxation_gradient_unrestricted(
                ctx, mol, &prep, bs, op, &bounds, rhf_config, result, v,
            )?
            .gradient
        }
        ferric_scf::Spin::RestrictedOpen => {
            ferric_scf::zvector_ks::relaxation_gradient_roks(
                ctx, mol, &prep, bs, op, &bounds, rhf_config, result, v,
            )?
            .gradient
        }
        ferric_scf::Spin::Restricted => {
            ferric_scf::zvector_ks::relaxation_gradient_closed(
                ctx, mol, &prep, bs, op, &bounds, rhf_config, result, v,
            )?
            .gradient
        }
    };
    let unrelaxed = out.gradient_unrelaxed.as_ref().ok_or_else(|| {
        FerricError::General("mbd_rsscs_for_scf: no unrelaxed gradient was formed".into())
    })?;
    out.gradient = Some(unrelaxed + &relax);
    out.gradient_relaxation = Some(relax);
    Ok(out)
}
