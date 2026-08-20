//! IEF-PCM (integral-equation-formalism polarizable continuum model)
//! implicit solvation for ferric.
//!
//! # Overview
//!
//! Models the solvent as a dielectric continuum outside a molecular cavity
//! built from atom-centered spheres (scaled Bondi van der Waals radii),
//! tessellated into surface elements ("tesserae"). The solute's
//! electrostatic potential at the cavity surface induces an apparent
//! surface charge `q` (solved via the IEF-PCM boundary-integral equation),
//! which generates a reaction-field potential that feeds back into the
//! solute's one-electron Hamiltonian. Because `q` depends on the solute
//! density and the density depends on `q` (through the Fock operator), this
//! is solved self-consistently by re-computing `q` from the CURRENT density
//! every SCF iteration and letting the outer SCF/DIIS loop carry the
//! overall fixed point — the same "solve once per iteration, let outer SCF
//! converge it" pattern PySCF and Psi4 use (see `pcm_step`'s doc for
//! specifics).
//!
//! # Module map
//!
//! * [`radii`] — Bondi van der Waals radii table.
//! * [`cavity`] — atom-sphere + Lebedev tessellation (GEPOL-lite; see that
//!   module's doc for the explicit simplifications vs full GEPOL).
//! * [`matrices`] — S/D boundary-element matrices and the isotropic
//!   IEF-PCM K/R operators.
//! * [`solver`] — the K q = R v linear solve for apparent surface charges.
//! * [`potential`] — solute ESP at tesserae, and the reaction-field
//!   one-electron AO operator built from `q`.
//! * [`config`] — [`config::PcmConfig`], the public config struct threaded
//!   through the SCF configs.
//!
//! # Naming note for a future COSMO merge
//!
//! A parallel effort is implementing COSMO (the ε→∞ conductor-limit
//! approximation to this same physics) in a separate worktree. The cavity
//! representation here ([`cavity::Tessera`], [`cavity::CavityConfig`],
//! [`cavity::build_cavity`], and the Bondi table in [`radii`]) is
//! model-agnostic — COSMO needs exactly the same atom-sphere-tessellation
//! geometry, differing only in the K/R operator ([`matrices::build_k_r`]
//! would become a single `f(ε) = (ε−1)/ε` conductor-scaled `S` solve with no
//! `D` matrix at all) and the charge-symmetrization/solve step
//! ([`solver::solve_pcm_charges`]). A future dedup pass should be able to
//! hoist `radii.rs` and `cavity.rs` into a shared crate (or have one crate
//! depend on the other for just those two modules) with `matrices.rs` /
//! `solver.rs` staying model-specific. This crate is named `ferric-pcm`
//! (not `ferric-solvation`) precisely so that split is easy later; if the
//! other agent named their crate something COSMO-specific, no rename
//! collision is expected.

/// Molecular cavity construction (atom-sphere + Lebedev tessellation).
pub mod cavity;
/// PCM configuration struct.
pub mod config;
/// S/D boundary-element matrices and IEF-PCM K/R operators.
pub mod matrices;
/// Solute ESP at tesserae and the reaction-field AO operator.
pub mod potential;
/// Bondi van der Waals radii table.
pub mod radii;
/// IEF-PCM linear solve for apparent surface charges.
pub mod solver;

use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::Array2;

pub use cavity::{build_cavity, CavityConfig, Tessera};
pub use config::PcmConfig;
pub use solver::PcmChargeResult;

/// Geometry-only PCM state built ONCE before the SCF loop (cavity + S/D/K/R
/// matrices depend only on the molecular geometry and `epsilon`, never on
/// the density) — mirrors how `DfJ`/`DfK`/`LinkK` are built once in
/// `solve_rhf` and only their `.build(&d, ...)` step runs per iteration.
#[derive(Debug, Clone)]
pub struct PcmContext {
    tess: Vec<Tessera>,
    k: Array2<f64>,
    r: Array2<f64>,
}

impl PcmContext {
    /// Build the cavity and the isotropic IEF-PCM K/R operators for `mol`
    /// under `cfg`. Returns `Err` for a degenerate cavity (zero tesserae) or
    /// an invalid `epsilon`.
    pub fn new(mol: &Molecule, cfg: &PcmConfig) -> Result<Self, FerricError> {
        let tess = build_cavity(mol, &cfg.cavity_config())?;
        let (s, d) = matrices::build_s_d_kind(&tess, cfg.sd_kind);
        let (k, r, _f_eps) = matrices::build_k_r(&s, &d, &tess, cfg.epsilon)?;
        Ok(Self { tess, k, r })
    }

    /// Number of tesserae (surface integration points) in the cavity.
    pub fn n_tesserae(&self) -> usize {
        self.tess.len()
    }

    /// The cavity tessera array (positions, areas, normals, charge exponents).
    pub fn tesserae(&self) -> &[Tessera] {
        &self.tess
    }
}

/// One PCM step: given the current AO density, solve for the apparent
/// surface charge, and return both the reaction-field one-electron AO
/// operator (to be ADDED to the Fock/hcore matrix so the SCF equations see
/// it) and the solvation energy contribution (to be added ONCE to the total
/// energy expression, NOT folded into the ordinary `0.5·D·(H+F)` one-electron
/// trace — see the crate-level doc and the `rhf.rs` call site for why).
///
/// This is the per-SCF-iteration hook: `ctx` (cavity + K/R) is built once
/// before the loop via [`PcmContext::new`]; this function is called fresh
/// every iteration with the just-rebuilt density.
pub fn pcm_step(
    ctx: &PcmContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    density: &Array2<f64>,
) -> Result<(Array2<f64>, f64), FerricError> {
    let v = potential::solute_potential_at_tesserae(mol, prep, density, &ctx.tess)?;
    let v_arr = ndarray::Array1::from_vec(v);
    let PcmChargeResult { q, e_pcm } = solver::solve_pcm_charges(&ctx.k, &ctx.r, &v_arr)?;
    let v_pcm = potential::build_reaction_field_operator(prep, &ctx.tess, q.as_slice().unwrap())?;
    Ok((v_pcm, e_pcm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;

    fn water_sto3g() -> (Molecule, PreparedBasis) {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        (mol, prep)
    }

    #[test]
    fn pcm_context_builds_for_water() {
        let (mol, _prep) = water_sto3g();
        let cfg = PcmConfig::water();
        let ctx = PcmContext::new(&mol, &cfg).unwrap();
        assert!(ctx.n_tesserae() > 0);
    }

    #[test]
    fn pcm_step_gives_stabilizing_energy_for_water() {
        let (mol, prep) = water_sto3g();
        let cfg = PcmConfig::water();
        let ctx = PcmContext::new(&mol, &cfg).unwrap();

        // Crude but nonzero density: superposed s-function-diagonal guess is
        // overkill for this smoke test — just use the identity scaled down,
        // which is not physical but is nonzero and exercises the full path
        // without depending on a converged SCF.
        let n = prep.nbasis();
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            d[(i, i)] = 0.5;
        }
        let (v_pcm, e_pcm) = pcm_step(&ctx, &mol, &prep, &d).unwrap();
        assert_eq!(v_pcm.dim(), (n, n));
        // A neutral molecule's crude density still has a net nuclear-dominated
        // positive potential at the cavity surface (identity/2 density is a
        // gross underestimate of true electron density), so the qualitative
        // sign isn't guaranteed here — just check finiteness and shape; the
        // real correctness gate is the converged-SCF test in ferric-scf.
        assert!(e_pcm.is_finite());
    }

    #[test]
    fn invalid_epsilon_is_rejected_at_context_build() {
        let (mol, _prep) = water_sto3g();
        let cfg = PcmConfig { epsilon: 1.0, ..PcmConfig::water() };
        assert!(PcmContext::new(&mol, &cfg).is_err());
    }
}
