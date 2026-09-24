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
//! * [`ucorr`] — Stage 8: Gamma-point open-shell UMP2 (`gamma_ump2`) and
//!   URPA (`gamma_urpa`) on the Gamma UHF: per-spin B from ONE periodic B →
//!   ferric-mp2's `u_ri_mp2_from_parts` / ferric-rpa's
//!   `run_u_pdep_rpa_from_parts`, or the dense-AFT oracle (independent UMP2
//!   loop, joint-spin plasmon); the same `v_M` on each spin's occupied levels.
//! * [`mod@uhf`] — Stage 4: Gamma-point open-shell UHF (`gamma_uhf`) over
//!   `ferric_scf::uhf::solve_uhf_injected`, with the per-spin Madelung term in
//!   the K builders, a per-spin gap check against `v_M`, and the staged
//!   (`exxdiv` none → ewald) start that avoids the Gamma Ewald trap.
//! * [`dft`] — Stage 2: Gamma-point closed-shell KS-DFT (`gamma_rks`; LDA,
//!   GGA, global hybrids): periodic atom-centred grid with SSF/Becke weights
//!   over image atoms (`PeriodicGrid`), lattice-summed AOs, and `PeriodicXc`
//!   injected into `solve_rhf_injected` as its `XcBuilder`. Stage 5: Gamma
//!   UKS (`gamma_uks`) — the same `PeriodicXc` evaluated spin-polarized and
//!   injected into `solve_uhf_injected` (`F_σ = h + J − a·K_inj(D_σ) + V_σ`),
//!   staged ewald start for hybrids, occupation-aware gap check against
//!   `a·v_M`.
//! * [`rohf`] — Stage 5b: Gamma ROHF / ROKS (`gamma_rohf`, `gamma_roks`) via
//!   `ferric_scf::rohf::solve_rohf_injected`: the UHF/UKS per-spin Fock
//!   (`F_σ = h + J − a·K_inj(D_σ) + V_σ`) Roothaan-combined on one MO set,
//!   staged ewald start for `a > 0`, per-spin gaps from the actual
//!   occupations against `a·v_M`.
//! * [`kpts`], [`kscf`], [`kdense_aft`] — Stage 3: k-point RHF
//!   (`solve_krhf`): Monkhorst-Pack / Gamma-centred meshes with exact
//!   Bloch phases and time-reversal pairing, per-k complex `S(k)`, `h(k)`
//!   (`hcore::kpoint`), the residue-resolved pair FT
//!   (`pair_ft::residues`), a DENSE pure-AFT k-point J/K oracle (supercell
//!   Madelung for `exxdiv = ewald`) and k-point RS-GDF per momentum transfer
//!   q ([`rsgdf::kpoint`], `KRhfConfig::jk = rsgdf`).
//! * [`kuscf`] — Stage 9: k-point open-shell UHF (`solve_kuhf`): per-spin
//!   `D_σ(k)`, `J[D_α + D_β]`, `K[D_σ]` from the same `KPointJk` (Madelung
//!   linear, no per-spin ½), GLOBAL per-spin aufbau over the mesh, staged
//!   (none → ewald) start by default, per-spin gaps from the actual
//!   occupations, giant-determinant `⟨S²⟩`.
//! * [`kcorr`] — Stage 9: k-point closed-shell MP2 (`kpoint_mp2`) and
//!   direct RPA (`kpoint_drpa`) on the complex k-point RS-GDF blocks (or the
//!   dense-AFT pair oracle `KDenseAftPairs`): explicit `Bvo`, momentum
//!   conservation `kb = ki + kj − ka`, per-q complex Hermitian `Π(q)`
//!   realified into ferric-rpa's real log-det pipeline; per-cell energies,
//!   Madelung-shifted denominators as at Gamma.
//!
//! * [`lindep`] — per-k canonical-cut diagnostics (`LindepReport` on
//!   `KScfResult`/`KUScfResult`: kept counts, smallest / largest-dropped
//!   eigenvalue, noise-floor flag) and the OPT-IN `exp_to_discard` basis
//!   filter (`prepare_cell_basis`); FINDINGS "Iteration 15".
//!
//! Units: Bohr and Hartree throughout; G vectors in Bohr⁻¹.
//!
//! Not wired into the CLI or the Python bindings yet — see
//! `reference/pbc/stage1-design.md` for the staging plan.

pub mod budget;
pub mod dense_aft;
pub mod dft;
pub mod drpa;
pub mod ewald;
pub mod hcore;
pub mod kcorr;
pub mod kdense_aft;
pub mod kpts;
pub mod kscf;
pub mod kuscf;
pub mod lattice;
pub mod lindep;
pub mod lmp2;
pub mod mp2;
pub mod pair_ft;
pub mod rohf;
pub mod rsgdf;
pub mod ucorr;
pub mod uhf;

pub use dense_aft::{DenseAftEri, ExxDiv};
pub use dft::{
    covering_radius_bound, gamma_rks, gamma_uks, gamma_uks_with_xc, GammaRksConfig, GammaRksResult,
    GammaUksConfig, GammaUksGridInfo, GammaUksResult, PeriodicDftError, PeriodicGrid,
    PeriodicGridConfig, PeriodicXc, PeriodicXcConfig,
};
pub use drpa::{gamma_drpa, GammaDrpaConfig, GammaDrpaIntegrals, GammaDrpaResult};
pub use ewald::{ewald_nuclear_repulsion, madelung_constant};
pub use hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
pub use hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
pub use kcorr::{
    kpoint_drpa, kpoint_mp2, KCorrIntegrals, KDenseAftPairs, KDrpaConfig, KDrpaEnergy, KDrpaResult,
    KMp2Config, KMp2Result,
};
pub use kdense_aft::{KDenseAftConfig, KDenseAftEri, KDenseAftJk};
pub use kpts::{KPointMesh, MeshCentring};
pub use kscf::{
    complex_canonical_orthogonalizer, complex_canonical_orthogonalizer_with_stats, solve_krhf,
    solve_krhf_injected, KJkKind, KPointInjection, KPointJk, KRhfConfig, KScfConfig, KScfResult,
};
pub use kuscf::{
    kuhf_gap_report, solve_kuhf, solve_kuhf_injected, solve_kuhf_injected_with_guess, KUScfResult,
    KUhfConfig, KUhfResult,
};
pub use lattice::Cell;
pub use lindep::{
    exp_to_discard, prepare_cell_basis, DiscardedShell, ExpToDiscardError, ExpToDiscardReport,
    KLindep, LindepReport,
};
pub use lmp2::{
    gamma_lmp2, gamma_lmp2_with_spaces, gamma_localized_spaces, mp2_closed_form_local, needle_axis,
    uniform_head, GammaEpsGate, GammaLmp2Config, GammaLmp2Inputs, GammaLmp2Result,
    GammaLocalSpaces, PeriodicDistance, UniformHead,
};
pub use mp2::{gamma_mp2, GammaMp2Config, GammaMp2Integrals, GammaMp2Result, Mp2Denominators};
pub use pair_ft::pair_ft;
pub use rohf::{
    gamma_rohf, gamma_roks, gamma_roks_with_xc, rohf_occupation_gaps, GammaRohfConfig,
    GammaRohfResult, GammaRoksConfig, GammaRoksResult,
};
pub use rsgdf::kpoint::{KRsGdf, KRsGdfConfig, KRsGdfJk, KRsGdfQStats, KRsGdfStats};
pub use rsgdf::{PeriodicFitParts, RsGdf, RsGdfConfig};
pub use ucorr::{gamma_ump2, gamma_urpa, GammaUmp2Result, GammaUrpaResult};
pub use uhf::{
    gamma_uhf, occupation_gaps, EwaldStart, GammaUhfConfig, GammaUhfIntegrals, GammaUhfResult,
    SpinGapReport,
};
