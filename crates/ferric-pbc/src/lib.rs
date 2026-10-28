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
//! * [`rohf`] — Stage 5b: Gamma ROHF / ROKS (`gamma_rohf`, `gamma_roks`,
//!   `gamma_roks_rsh` for range-separated hybrids; its dense-AFT force is
//!   the library-only `gamma_rsh::gamma_roks_rsh_gradient`) via
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
//! * [`ecp`] — periodic ECPs: the `apply_ecp` guard ([`check_ecp_applied`])
//!   and the Bloch sum `V_ECP(k) = Σ_L e^{ik·L} V_L` over libecpint's
//!   per-shell-pair kernel (`ferric_ecp_block`), added into `h` by
//!   `periodic_hcore` / `periodic_hcore_kpts`; FINDINGS "Iteration 14".
//!   The Gamma force term (`periodic_ecp_gradient`, the three moving centres
//!   bra / ket image / ECP image over the energy's own triples) is added to
//!   every Gamma force by [`grad`]; FINDINGS "Iteration 22".
//! * [`grad`] — analytic nuclear gradients (forces) of the Gamma-point RHF,
//!   UHF, RKS and UKS on the dense-AFT J/K (`gamma_rhf_gradient`,
//!   `gamma_uhf_gradient`, `gamma_rks_gradient`, `gamma_uks_gradient`):
//!   shifted 1e derivative blocks, erfc Gaussian-nucleus 3-centre
//!   derivatives, the bra-centre pair-FT derivative for `V_LR` and J/K, the
//!   Ewald `E_nn` gradient, the Madelung/G = 0 terms folded into the
//!   overlap-weighted matrix (per spin), and for KS the XC AO term plus the
//!   full periodic grid response (points riding on their home atom, analytic
//!   SSF/Becke weight derivatives over image atoms); FINDINGS "Iterations 16,
//!   17". The same four with RS-GDF J/K (`gamma_*_gradient_rsgdf`, B from
//!   `RsGdf::build_for_gradient`): fitted 3-index density against the SR/LR
//!   3-centre derivatives (orbital and aux centre), Loewner-form metric
//!   weight against the SR/LR metric derivative, J3's G = 0 term through
//!   `dS`; FINDINGS "Iteration 18".
//! * [`kgrad`] — analytic nuclear forces of the k-point RHF / UHF
//!   (`kpoint_rhf_gradient`, `kpoint_uhf_gradient`) on the dense-AFT or
//!   k-point RS-GDF J/K: the Gamma force terms with Bloch phases on every
//!   image-resolved derivative block, the residue-resolved pair-FT
//!   derivative (ket from `−iK p − Q_bra`), the mesh Madelung constant in the
//!   overlap weight; equals the Gamma force on ONE copy of each atom in the
//!   diag(N) supercell. FINDINGS "Iteration 21".
//! * [`kxcgrad`] — analytic nuclear forces of the k-point closed-shell RKS
//!   energy for LDA/GGA, meta-GGA (SCAN, r2SCAN, TPSS) and global-hybrid
//!   functionals (`kpoint_rks_gradient`)
//!   on the dense-AFT or RS-GDF (unsplit or range-split) J/K: the k-point HF
//!   force assembly with the exact-exchange fraction `a` plus the XC
//!   AO-derivative, grid-point-motion and weight-response terms on Bloch
//!   AOs, with the tau term on the AO Hessian for a meta-GGA; range-separated
//!   hybrids on the dense-AFT or RS-GDF J/K (HSE06, CAM-B3LYP force-tested; wB97X
//!   accepted, not exercised by a force test; the
//!   exchange walks weighted by the attenuated kernel, the mixed Madelung
//!   constant; the Gamma point through `gamma_rsh::gamma_rks_rsh_gradient` on
//!   the 1x1x1 mesh, dense AFT only); VV10 is refused.
//!   The Gamma-point meta-GGA force and stress route through the same 1x1x1
//!   code ([`gamma_mgga`], dense-AFT J).
//! * [`kxcstress`] — the stress of the same energy (`kpoint_rks_stress`): the
//!   k HF stress with the exact-exchange fraction `a` (dense K walks, the
//!   RS-GDF fit's exchange pieces and the Madelung term scaled by `a`) plus
//!   the XC strain terms on Bloch AOs (AO strain with point motion, the
//!   partition-weight response on an atom grid or the volume term on a
//!   uniform grid). LDA, GGA, meta-GGA (SCAN, r2SCAN, TPSS; dense-AFT or RS-GDF J),
//!   global hybrids (a hybrid needs `KRksStressConfig::exxdiv`) and
//!   range-separated (CAM) GGA hybrids on dense-AFT or RS-GDF J/K (the attenuated
//!   kernel's or fit's strain and the mixed Madelung strain; Gamma through the
//!   1×1×1 mesh, `gamma_rsh::gamma_rks_rsh_stress`, dense AFT only); VV10 is
//!   refused (open shell: see [`kuxcstress`]). The stress is
//!   returned unsymmetrised (the Gamma convention).
//! * [`kuxcstress`] — the open-shell twin (`kpoint_uks_stress`, k-point UKS
//!   from [`kuks`]): per-spin Pulay/exchange/Madelung assembly and the
//!   spin-resolved XC strain terms; LDA, GGA, global and range-separated
//!   (CAM) GGA hybrids (dense-AFT or RS-GDF J/K; Gamma UKS through the 1×1×1
//!   mesh, `gamma_rsh::gamma_uks_rsh_stress`), with the Fock guard and the
//!   stationarity gate. `kpoint_roks_stress` /
//!   `kpoint_rohf_stress` (the restricted-open-shell energies of [`kroks`]):
//!   the same assembly on the ROHF spin densities with the orthonormality
//!   Lagrangian `W_RO` as the overlap weight and the ROHF orbital-gradient
//!   gate in place of the commutator gate; dense-AFT or RS-GDF J/K
//!   (unsplit; anchored by FD in `tests/pbc_kuxcstress_ro_rsgdf.rs`). The gate is loose: at its tolerance (1e-5) the stress
//!   error can reach about 4e-5 to 5e-5 Ha per cell (~18 MPa on the 64 Bohr³
//!   test cell; measured on H3 and H5 STO-3G at 1x1x1 on the DENSE-AFT J/K only, not on RS-GDF and not a general bound),
//!   so it refuses an unconverged state without certifying the stress to the
//!   anchors' accuracy. The k gate applies the tolerance to the spectral norm of
//!   the blocks, which is at least the largest element the Gamma entry points
//!   use: it can refuse a state the Gamma entry point accepts.
//! * [`kuxcgrad`] — the open-shell twin (`kpoint_uks_gradient`, k-point UKS
//!   from [`kuks`]; `kpoint_roks_gradient` / `kpoint_rohf_gradient` for the
//!   restricted-open-shell energies of [`kroks`] with the ROHF orbital-gradient
//!   gate, a spectral-norm measure that can refuse a state the Gamma entry
//!   points accept; force error about 0.1 to 0.74 times the reported g on
//!   H3/STO-3G at 1x1x1, not a general bound): the same assembly with two spin channels, spin-resolved
//!   `v_xc` (`σ_αα`, `σ_αβ`, `σ_ββ` contractions), per-spin exchange and a
//!   per-spin Fock guard; closed-shell densities reproduce [`kxcgrad`] per
//!   part. UKS and ROKS take range-separated hybrids (`K_σ = K[D_σ]` on the
//!   attenuated kernel, the mixed Madelung constant; the Gamma point through
//!   `gamma_rsh::gamma_uks_rsh_gradient` / `gamma_roks_rsh_gradient`, dense
//!   AFT only; at k on the dense-AFT or RS-GDF J/K) and UKS also SCAN / r2SCAN
//!   (per-spin tau terms, dense-AFT or RS-GDF J/K);
//!   ROKS refuses meta-GGA (the k-point ROKS SCAN / r2SCAN energy has no
//!   derivative yet), and VV10 is refused.
//! * [`kstress`] — the k-point RHF / UHF stress tensor
//!   (`kpoint_rhf_stress`, `kpoint_uhf_stress`) on the dense-AFT and the
//!   unsplit RS-GDF J/K:
//!   the Gamma stress terms with the force's phase-folded weights times the
//!   pair vector (k at fixed fractional coordinates, so no k-strain term),
//!   the residue-resolved pair-FT strain kernel for `V_LR` and the per-q
//!   J/K, and the Madelung strain of the diag(N) supercell lattice; equals
//!   the Gamma stress of the explicit supercell per cell. Supports the
//!   range-split build, a scalar periodic ECP and the column-rotated SR walks
//!   (the config must agree with the build); refuses range-separated kernels and any SCF result whose Fock it
//!   cannot reproduce (KS, other `exxdiv`, an SCF run without the ECP).
//! * [`stress`] — the Gamma-point stress tensor `σ = (1/Ω) dE/dε` of the
//!   same eight energies (`gamma_{rhf,uhf,rks,uks}_stress[_rsgdf]`): the
//!   force derivative blocks contracted with image-resolved pair vectors,
//!   the pair-FT / aux-FT G-shape strain at fixed Miller index, the Ω and
//!   `|G|` dependence of every reciprocal kernel, Ewald / Madelung / c0 volume
//!   terms, the per-image grid-weight strain and AO first moments for KS, and
//!   the RS-GDF metric G = 0 strain term; strained cells with FROZEN index
//!   sets ([`Cell::strained`]) make the energy it differentiates smooth.
//!   FINDINGS "Iteration 19".
//! * [`lindep`] — per-k canonical-cut diagnostics (`LindepReport` on
//!   `KScfResult`/`KUScfResult`: kept counts, smallest / largest-dropped
//!   eigenvalue, noise-floor flag) and the OPT-IN `exp_to_discard` basis
//!   filter (`prepare_cell_basis`); FINDINGS "Iteration 15".
//!
//! * [`sr_rotation`] — column rotation of generally contracted orbital
//!   shells inside the SR 3-centre walk (RS-GDF) and the hcore SR
//!   attraction ([`SrColumnRotation`] on `RsGdfConfig` /
//!   `PeriodicHcoreConfig`, default `Auto` = on in the Gamma energy and
//!   gradient builds and the k-point energy builds, off in the k-point
//!   force and stress builds and the frozen oracles): subtract each
//!   shell group's single-primitive columns from the others, compute the SR
//!   blocks on the rotated shells, back-transform exactly (design
//!   `reference/pbc/sr-general-contraction-design.md`); the Gamma forces and
//!   stress differentiate the same rotated walks (weights `Tᵀ W T`).
//!
//! * [`timing`] — stage timers (wall + process CPU) and counters on
//!   `PeriodicHcore`, `RsGdf`, `DenseAftEri`, the KS results and the k-point
//!   results, plus per-SCF-iteration J/K/XC call clocks ([`PbcTimings`]);
//!   observation only, energies bit-identical (FINDINGS "Performance plan",
//!   item 0).
//!
//! Units: Bohr and Hartree throughout; G vectors in Bohr⁻¹.
//!
//! Bound by the Python bindings (`ferric-python`, `src/pbc.rs`) and the CLI
//! (`ferric-cli`, the `[cell]` path); `reference/pbc/stage1-design.md` holds
//! the staging plan.

pub mod budget;
pub mod dense_aft;
pub mod dft;
pub mod drpa;
pub mod ecp;
pub mod ewald;
pub mod gamma_mgga;
pub mod gamma_rsh;
pub mod grad;
pub mod hcore;
pub mod kcorr;
pub mod kdense_aft;
pub mod kdft;
pub mod kgrad;
pub mod kpts;
pub mod kroks;
pub mod kscf;
pub mod kstress;
pub mod kuks;
pub mod kuscf;
pub mod kuxcgrad;
pub mod kuxcstress;
pub mod kxcgrad;
pub mod kxcstress;
pub mod lattice;
pub mod lindep;
pub mod lmp2;
pub mod mp2;
mod ordered;
pub mod pair_ft;
pub mod rohf;
pub mod rsgdf;
pub mod rsh;
pub mod sr_rotation;
pub mod stress;
pub mod timing;
pub mod ucorr;
pub mod uhf;

pub use dense_aft::{DenseAftEri, ExxDiv};
pub use dft::{
    covering_radius_bound, gamma_rks, gamma_rks_rsh, gamma_rks_rsh_on_grid, gamma_uks,
    gamma_uks_rsh, gamma_uks_rsh_on_grid, gamma_uks_with_xc, GammaRksConfig, GammaRksResult,
    GammaUksConfig, GammaUksGridInfo, GammaUksResult, PeriodicDftError, PeriodicGrid,
    PeriodicGridConfig, PeriodicXc, PeriodicXcConfig,
};
pub use drpa::{gamma_drpa, GammaDrpaConfig, GammaDrpaIntegrals, GammaDrpaResult};
pub use ecp::{
    check_ecp_applied, periodic_ecp_gradient, periodic_ecp_images, EcpMutation, PeriodicEcpConfig,
    PeriodicEcpError, PeriodicEcpGradient, PeriodicEcpImages,
};
pub use ewald::{ewald_nuclear_gradient, ewald_nuclear_repulsion, madelung_constant};
pub use grad::{
    gamma_rhf_gradient, gamma_rhf_gradient_rsgdf, gamma_rhf_gradient_with, gamma_rks_gradient,
    gamma_rks_gradient_rsgdf, gamma_rks_gradient_with, gamma_uhf_gradient,
    gamma_uhf_gradient_rsgdf, gamma_uhf_gradient_with, gamma_uks_gradient,
    gamma_uks_gradient_rsgdf, gamma_uks_gradient_with, GammaGradConfig, GammaGradParts,
    GammaGradient, GammaRhfGradient, RsGdfGradSource,
};
pub use grad::{
    gamma_rohf_gradient, gamma_rohf_gradient_rsgdf, gamma_rohf_gradient_with, gamma_roks_gradient,
    gamma_roks_gradient_rsgdf, gamma_roks_gradient_with, rohf_lagrangian_w, rohf_orbital_gradient,
    RohfOrbitalGradient, RO_ORBITAL_GRADIENT_TOL,
};
pub use hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
pub use hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
pub use kcorr::{
    kpoint_drpa, kpoint_mp2, KCorrIntegrals, KDenseAftPairs, KDrpaConfig, KDrpaEnergy, KDrpaResult,
    KMp2Config, KMp2Result,
};
pub use kdense_aft::{KDenseAftConfig, KDenseAftEri, KDenseAftJk};
pub use kdft::{
    solve_krks, solve_krks_keep, solve_krks_keep_on_grid, solve_krks_on_grid, KPeriodicXc,
    KRksConfig, KRksResult,
};
pub use kgrad::{
    kpoint_rhf_gradient, kpoint_uhf_gradient, KGradConfig, KGradJk, KGradParts, KGradient,
    KRsGdfGradSource,
};
pub use kpts::{KPointMesh, MeshCentring};
pub use kroks::{solve_krohf, solve_krohf_keep, solve_kroks, solve_kroks_keep, KRoksResult};
pub use kscf::{
    complex_canonical_orthogonalizer, complex_canonical_orthogonalizer_with_stats, solve_krhf,
    solve_krhf_injected, solve_krhf_keep, solve_krks_injected, KJkKind, KPointInjection, KPointJk,
    KPointXc, KRetained, KRetainedJk, KRhfConfig, KScfConfig, KScfResult,
};
pub use kuks::{solve_kuks, solve_kuks_keep, solve_kuks_on_grid, KUksConfig, KUksResult};
pub use kuscf::{
    kuhf_gap_report, solve_kuhf, solve_kuhf_injected, solve_kuhf_injected_with_guess,
    solve_kuhf_keep, KUScfResult, KUhfConfig, KUhfResult,
};
pub use kuxcgrad::{
    kpoint_rohf_gradient, kpoint_roks_gradient, kpoint_uks_gradient, KUksGradConfig, KUksGradient,
};
pub use kuxcstress::{
    kpoint_rohf_stress, kpoint_roks_stress, kpoint_uks_stress, KUksStress, KUksStressConfig,
};
pub use kxcgrad::{kpoint_rks_gradient, KRksGradConfig, KRksGradient, KXcGrid};
pub use kxcstress::{kpoint_rks_stress, KRksStress, KRksStressConfig};
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
    gamma_rohf, gamma_roks, gamma_roks_rsh, gamma_roks_rsh_on_grid, gamma_roks_with_xc,
    rohf_occupation_gaps, GammaRohfConfig, GammaRohfResult, GammaRoksConfig, GammaRoksResult,
    RohfSecondOrderInfo, RohfSolver, RohfTrahConfig,
};
pub use rsgdf::kpoint::{KRsGdf, KRsGdfConfig, KRsGdfJk, KRsGdfQStats, KRsGdfStats};
pub use rsgdf::{PeriodicFitParts, RsGdf, RsGdfConfig, RsGdfFitDiagnostics};
pub use rsgdf::{RangeSplit, RangeSplitMutant};
pub use sr_rotation::{ColumnRotation, ColumnRotationMutant, SrColumnRotation};
pub use stress::{
    gamma_rhf_stress, gamma_rhf_stress_rsgdf, gamma_rks_stress, gamma_rks_stress_rsgdf,
    gamma_uhf_stress, gamma_uhf_stress_rsgdf, gamma_uks_stress, gamma_uks_stress_rsgdf,
    GammaStress, GammaStressConfig, GammaStressParts,
};
pub use stress::{gamma_rohf_stress, gamma_roks_stress};
pub use timing::{CallClock, PbcTimings, StageClock, StageTiming};
pub use ucorr::{gamma_ump2, gamma_urpa, GammaUmp2Result, GammaUrpaResult};
pub use uhf::{
    gamma_uhf, occupation_gaps, EwaldStart, GammaUhfConfig, GammaUhfIntegrals, GammaUhfResult,
    SpinGapReport,
};
