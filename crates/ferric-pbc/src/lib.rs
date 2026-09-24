//! Periodic-boundary-condition primitives for ferric — Stages 0-1 (library only).
//!
//! * [`lattice`] — [`Cell`]: a [`Molecule`](ferric_core::mol::Molecule)
//!   plus three lattice vectors, with volume, reciprocal vectors, real-space
//!   image translations and reciprocal-space G vectors.
//! * [`ewald`] — Ewald nuclear repulsion (neutralising-background convention,
//!   matching PySCF `Cell.energy_nuc()`) and the Gamma-point Madelung constant
//!   (matching `pyscf.pbc.tools.madelung`).
//! * [`mod@pair_ft`] — analytic Fourier transform of lattice-summed Gaussian pair
//!   densities, `P_mn(G) = Σ_L ∫ φ_m(r) φ_n(r−L) e^{−iG·r} dr`, in ferric's AO
//!   basis (the `PreparedBasis` order and normalization libint2 uses).
//! * [`hcore`] — Stage 1: Gamma-point lattice-summed `S`, `T`, Ewald-split
//!   nuclear attraction (Gaussian nuclei through the shifted 3-centre engine
//!   + analytic pair FT) and `E_nn` ([`PeriodicHcore`]).
//! * [`budget`] — memory gating: every large buffer is reserved against
//!   ferric's unified budget before allocation (Stage 1 step 10).
//! * [`dense_aft`] — TEST/ORACLE ONLY: dense pure-AFT `nao⁴` ERI tensor with
//!   `JBuilder`/`KBuilder` impls (Madelung `exxdiv`) for
//!   `ferric_scf::rhf::solve_rhf_injected`; hard size cap.
//! * [`rsgdf`] — Stage 1 step 9: Gamma-point range-separated Gaussian density
//!   fitting (PySCF-RSGDF G = 0 convention, eig-with-lindep metric solve) and
//!   `JBuilder`/`KBuilder` impls on the fitted B tensor (Madelung `exxdiv`).
//! * [`mp2`] — Stage 6: Gamma-point closed-shell MP2 on the RS-GDF B (the
//!   SCF's own B → `B[k,ia]` → ferric-mp2's `spin_components_from_b_ov`) or
//!   the dense-AFT oracle, with explicit Madelung-shifted denominators.
//! * [`drpa`] — Stage 7: Gamma-point closed-shell dRPA on the same B through
//!   ferric-rpa's full-rank pipeline (`run_pdep_rpa_from_parts`, no molecular
//!   basis objects), or the dense-AFT plasmon oracle; same denominator
//!   conventions as MP2.
//! * [`lmp2`] — Stage 8b: Gamma-point amplitude-threshold local MP2
//!   (`gamma_lmp2`): Berghold/Resta periodic localisation, periodic VV-HV
//!   virtuals, minimum-image distances, per-pair domain fits in the periodic
//!   metric, shifted denominators; ferric-mp2's ragged solver. KNOWN
//!   LIMITATION: at Gamma the default (raw-integral) eps gate keeps every
//!   pair below a volume onset; the opt-in `GammaEpsGate::UniformHeadRestored`
//!   ("A-drop": restored q = 0 head, needle supercells only) removes it, with
//!   a different eps = 0 target (see the module doc).
//! * [`mod@uhf`] — Stage 4: Gamma-point open-shell UHF (`gamma_uhf`) over
//!   `ferric_scf::uhf::solve_uhf_injected`, with the per-spin Madelung term in
//!   the K builders, a per-spin gap check against `v_M`, and the staged
//!   (`exxdiv` none → ewald) start that avoids the Gamma Ewald trap.
//!
//! Units: Bohr and Hartree throughout; G vectors in Bohr⁻¹.
//!
//! Not wired into the CLI or the Python bindings yet — see
//! `reference/pbc/stage1-design.md` for the staging plan.

pub mod budget;
pub mod dense_aft;
pub mod drpa;
pub mod ewald;
pub mod hcore;
pub mod lattice;
pub mod lmp2;
pub mod mp2;
pub mod pair_ft;
pub mod rsgdf;
pub mod uhf;

pub use dense_aft::{DenseAftEri, ExxDiv};
pub use drpa::{gamma_drpa, GammaDrpaConfig, GammaDrpaIntegrals, GammaDrpaResult};
pub use ewald::{ewald_nuclear_repulsion, madelung_constant};
pub use hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
pub use lattice::Cell;
pub use lmp2::{
    gamma_lmp2, gamma_lmp2_with_spaces, gamma_localized_spaces, mp2_closed_form_local, needle_axis,
    uniform_head, GammaEpsGate, GammaLmp2Config, GammaLmp2Inputs, GammaLmp2Result,
    GammaLocalSpaces, PeriodicDistance, UniformHead,
};
pub use mp2::{gamma_mp2, GammaMp2Config, GammaMp2Integrals, GammaMp2Result, Mp2Denominators};
pub use pair_ft::pair_ft;
pub use rsgdf::{PeriodicFitParts, RsGdf, RsGdfConfig};
pub use uhf::{gamma_uhf, EwaldStart, GammaUhfConfig, GammaUhfIntegrals, GammaUhfResult};
