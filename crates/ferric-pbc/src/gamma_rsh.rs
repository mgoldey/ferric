//! Range-separated exact exchange at the Gamma point: the RKS, UKS and ROKS
//! energies ([`crate::dft::gamma_rks_rsh`], [`crate::dft::gamma_uks_rsh`],
//! [`crate::rohf::gamma_roks_rsh`]; mgoldey/ferric#354) and the nuclear
//! forces and stresses routed through the 1x1x1 mesh
//! ([`gamma_rks_rsh_gradient`], [`gamma_uks_rsh_gradient`],
//! [`gamma_roks_rsh_gradient`]; [`gamma_rks_rsh_stress`],
//! [`gamma_uks_rsh_stress`]), on the dense-AFT or the RS-GDF exchange
//! (mgoldey/ferric#492): the route hands the 1x1x1 derivative the source the
//! SCF's exchange ran on, the dense `k = 0` kernels or the `k = 0`
//! `KRsGdf::build_rsh` fit with its config and aux.
//!
//! # Route
//!
//! The attenuated K exists only as the k-point code ([`crate::rsh`],
//! [`crate::kdense_aft`] for the dense-AFT oracle, `KRsGdf::build_rsh` for
//! the RS-GDF fit). The Gamma point IS the 1x1x1 Gamma-centred mesh: its
//! supercell is the cell, `D(k=0)` is the real Gamma density, and the
//! complex K at `k = 0` is real. This module wraps that machinery as a REAL
//! [`KBuilder`]; J, the one-electron terms, the semilocal XC and the SCF are
//! the unchanged Gamma code ([`crate::dft::gamma_rks`]).
//!
//! Because the K kernel is shared with the k-point path, "Gamma equals the
//! 1x1x1 mesh" is only a check of the glue plus the independent Gamma J/XC/
//! SCF; the INDEPENDENT oracle for the attenuated exchange is PySCF's Gamma
//! `pbc.dft.RKS` + AFTDF (`tests/pbc_grsh.rs`).
//!
//! # Cost
//!
//! With `jk = rsgdf` the full-Coulomb fit is built TWICE: once for the Gamma
//! J (`RsGdf`) and once inside the k = 0 exchange (`KRsGdf::build_rsh`, plus
//! its attenuated G-space fit). The two objects have different layouts, so
//! they are not shared. Measured (H2/STO-3G, a = 4 Bohr, cc-pVDZ-RI, release-
//! like test profile, 6-core box, 2 runs): Gamma J fit 0.91 / 1.01 s, k = 0 K
//! fit (full + attenuated) 0.85 / 0.91 s, i.e. the RSH build is about 1.9x the
//! plain Gamma build; the SCF itself is unchanged.
//!
//! The exchange fractions and the mixed Madelung constant are INSIDE the K
//! this builder returns, so the SCF scales it by 1 ([`crate::dft`]).

use crate::dense_aft::ExxDiv;
use crate::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use crate::kpts::KPointMesh;
use crate::kscf::KPointJk;
use crate::lattice::Cell;
use crate::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use crate::rsh::{is_range_separated, RshParams};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_scf::fock::KBuilder;
use ndarray::Array2;
use num_complex::Complex64 as C64;

/// Largest imaginary part tolerated in the `k = 0` exchange of a real
/// density (the AOs are real, so it is rounding only).
const IMAG_TOL: f64 = 1e-9;

/// TEST ONLY: a deliberately wrong per-spin exchange of the open-shell
/// [`crate::dft::gamma_uks_rsh`] and [`crate::rohf::gamma_roks_rsh`] (the SCF
/// calls K once per spin, alpha first). Each moves a polarized energy.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GammaSpinMutant {
    /// The beta call returns the alpha result.
    AlphaOnBeta,
    /// The beta call returns zero.
    DropBeta,
    /// The Madelung term applied per spin is halved (RHF's D/2 bookkeeping
    /// wrongly carried over to a spin density).
    MadelungHalf,
}

enum Source {
    Dense(KDenseAftEri),
    RsGdf(Box<RsGdfSource>),
}

/// The RS-GDF exchange: the fit, the config it was built with and its aux
/// basis (re-prepared from the caller's on the cell's atoms); the RS-GDF
/// derivative passes rebuild the fit per q from these and check it against
/// `gdf`. A caller's aux prepared on other centres than the cell's atoms
/// therefore no longer matches `gdf` and is refused by that rebuilt-B check
/// (relative ‖B‖ mismatch 1.95e-4 for H2 with the aux centres moved 0.1
/// Bohr; bar 1e-8), not silently differentiated.
struct RsGdfSource {
    gdf: KRsGdf,
    cfg: KRsGdfConfig,
    aux: PreparedBasis,
}

/// The attenuated exchange of a Gamma-point range-separated hybrid.
pub struct GammaRshExchange {
    src: Source,
    rsh: RshParams,
    exxdiv: ExxDiv,
    madelung: f64,
    /// TEST ONLY: skip the functional / exxdiv consistency checks so a test
    /// can inject a deliberately wrong exchange (a mutant).
    unchecked: bool,
    /// TEST ONLY: see [`GammaSpinMutant`].
    spin_mutant: Option<GammaSpinMutant>,
}

impl GammaRshExchange {
    /// Dense-AFT exchange (the test oracle: `nao^4` kernels, capped by
    /// `cfg.max_bytes`). `s` is the Gamma overlap.
    pub fn build_dense(
        cell: &Cell,
        prep: &PreparedBasis,
        s: &Array2<f64>,
        exxdiv: ExxDiv,
        cfg: &KDenseAftConfig,
        rsh: RshParams,
    ) -> Result<Self, FerricError> {
        let mesh = KPointMesh::gamma_centred(cell, [1, 1, 1])?;
        let mut cfg = *cfg;
        cfg.rsh = Some(rsh);
        let eri = KDenseAftEri::build(cell, prep, &mesh, &[to_complex(s)], exxdiv, &cfg)?;
        let madelung = eri.madelung_ewald();
        Ok(Self {
            src: Source::Dense(eri),
            rsh,
            exxdiv,
            madelung,
            unchecked: false,
            spin_mutant: None,
        })
    }

    /// RS-GDF exchange: the full-Coulomb fit plus the G-space fit of the
    /// erf-attenuated kernel at the single `k = 0`.
    pub fn build_rsgdf(
        cell: &Cell,
        prep: &PreparedBasis,
        aux: &PreparedBasis,
        s: &Array2<f64>,
        exxdiv: ExxDiv,
        cfg: &KRsGdfConfig,
        rsh: RshParams,
    ) -> Result<Self, FerricError> {
        let mesh = KPointMesh::gamma_centred(cell, [1, 1, 1])?;
        let gdf = KRsGdf::build_rsh(cell, prep, aux, &mesh, &[to_complex(s)], cfg, rsh)?
            .with_exxdiv(exxdiv);
        let madelung = gdf.madelung_ewald();
        let aux = PreparedBasis::new(cell.mol(), aux.basis_set())?;
        Ok(Self {
            src: Source::RsGdf(Box::new(RsGdfSource {
                gdf,
                cfg: *cfg,
                aux,
            })),
            rsh,
            exxdiv,
            madelung,
            unchecked: false,
            spin_mutant: None,
        })
    }

    /// TEST ONLY: disable the consistency checks of `gamma_rks_rsh` so a
    /// mutant exchange (wrong `ω`, swapped fractions, wrong exxdiv) reaches the
    /// SCF and the anchors can be shown to catch it.
    #[doc(hidden)]
    pub fn unchecked_for_mutation_tests(mut self) -> Self {
        self.unchecked = true;
        self
    }

    /// TEST ONLY: make the open-shell K wrong in the way `m` says (implies
    /// the unchecked mode is NOT needed: the functional / exxdiv still match).
    #[doc(hidden)]
    pub fn with_spin_mutant_for_mutation_tests(mut self, m: GammaSpinMutant) -> Self {
        self.spin_mutant = Some(m);
        self
    }

    /// The range-separation parameters.
    pub fn params(&self) -> RshParams {
        self.rsh
    }

    /// The mixed exchange-divergence constant `c_sr v_M^{full} + (c_lr − c_sr)
    /// v_M^{LR}(ω)` of the cell, WHETHER OR NOT `exxdiv` applies it (the K
    /// builder applies it only under `exxdiv = ewald`).
    pub fn mixed_madelung(&self) -> f64 {
        self.madelung
    }

    /// Checks `gamma_rks_rsh` needs: the functional is range-separated with
    /// THESE parameters and the SCF's exxdiv is the one K was built with.
    pub(crate) fn check_for_rks(
        &self,
        functional: &str,
        exxdiv: ExxDiv,
    ) -> Result<(), FerricError> {
        self.check_for("gamma_rks_rsh", functional, exxdiv)
    }

    /// [`Self::check_for_rks`] naming the calling entry point `who`.
    pub(crate) fn check_for(
        &self,
        who: &str,
        functional: &str,
        exxdiv: ExxDiv,
    ) -> Result<(), FerricError> {
        if self.unchecked {
            return Ok(());
        }
        if !is_range_separated(functional) {
            return Err(FerricError::General(format!(
                "{who}: functional {functional:?} is not a range-separated hybrid \
                 (use the global-hybrid entry point)"
            )));
        }
        let want = crate::rsh::resolve_periodic_functional_rsh(functional)?
            .2
            .ok_or_else(|| FerricError::General(format!("{who}: no CAM parameters")))?;
        if want != self.rsh {
            return Err(FerricError::General(format!(
                "{who}: the exchange was built for {:?} but {functional:?} has {want:?}",
                self.rsh
            )));
        }
        if exxdiv != self.exxdiv {
            return Err(FerricError::General(format!(
                "{who}: exxdiv {exxdiv:?} differs from the one the exchange was built \
                 with ({:?})",
                self.exxdiv
            )));
        }
        Ok(())
    }

    /// `K[d]` (fractions and Madelung term included) for a real density `d`.
    /// Exposed so tests can compare two exchange sources as MATRICES.
    pub fn k_matrix(&self, d: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
        let mut k = Array2::<f64>::zeros(d.dim());
        self.k_builder().build(d, &mut k)?;
        Ok(k)
    }

    /// The real K builder (fractions and Madelung term included).
    pub fn k_builder(&self) -> GammaRshK<'_> {
        let jk: Box<dyn KPointJk + '_> = match &self.src {
            Source::Dense(e) => Box::new(e.jk_builder()),
            Source::RsGdf(r) => Box::new(r.gdf.jk_builder()),
        };
        GammaRshK::new(jk, None)
    }

    /// [`Self::k_builder`] with an explicit Madelung term `madelung` (0 =
    /// `exxdiv = none`), for the staged ewald start of an open-shell run.
    pub fn k_builder_with_madelung(&self, madelung: f64) -> GammaRshK<'_> {
        let madelung = if self.spin_mutant == Some(GammaSpinMutant::MadelungHalf) {
            0.5 * madelung
        } else {
            madelung
        };
        let jk: Box<dyn KPointJk + '_> = match &self.src {
            Source::Dense(e) => Box::new(e.jk_builder_with_madelung(madelung)),
            Source::RsGdf(r) => Box::new(r.gdf.jk_builder_with_madelung(madelung)),
        };
        GammaRshK::new(jk, self.spin_mutant)
    }
}

impl GammaRshExchange {
    /// The k = 0 J/K source a routed Gamma force differentiates (shared by the
    /// RKS, UKS and ROKS Gamma routes): the dense kernels, or the RS-GDF fit
    /// (`KRsGdf::build_rsh` at the single `k = 0`) with the config and aux it
    /// was built with. The 1x1x1 force rebuilds `h + J + V_xc − K` from this
    /// source and its Fock guard compares it with the SCF's stored Fock, so a
    /// Gamma J that is not the same fit (dense J under an RS-GDF exchange, or
    /// another split or aux) is refused there, not silently differentiated.
    fn force_jk(&self) -> crate::kgrad::KGradJk<'_> {
        match &self.src {
            Source::Dense(eri) => crate::kgrad::KGradJk::Dense(eri),
            Source::RsGdf(r) => crate::kgrad::KGradJk::RsGdf(crate::kgrad::KRsGdfGradSource {
                gdf: &r.gdf,
                cfg: &r.cfg,
                aux: &r.aux,
            }),
        }
    }

    /// [`Self::force_jk`] as the stress entry points' source.
    fn stress_jk(&self) -> crate::kstress::KStressJk<'_> {
        match self.force_jk() {
            crate::kgrad::KGradJk::Dense(e) => crate::kstress::KStressJk::Dense(e),
            crate::kgrad::KGradJk::RsGdf(s) => crate::kstress::KStressJk::RsGdf(s),
        }
    }
}

fn to_complex(m: &Array2<f64>) -> Array2<C64> {
    m.mapv(|x| C64::new(x, 0.0))
}

/// Real [`KBuilder`] over the `k = 0` complex exchange.
pub struct GammaRshK<'a> {
    jk: Box<dyn KPointJk + 'a>,
    mutant: Option<GammaSpinMutant>,
    /// Calls so far / the last result (only the spin mutants read them).
    calls: usize,
    last: Option<Array2<f64>>,
}

impl<'a> GammaRshK<'a> {
    fn new(jk: Box<dyn KPointJk + 'a>, mutant: Option<GammaSpinMutant>) -> Self {
        Self {
            jk,
            mutant,
            calls: 0,
            last: None,
        }
    }

    /// The beta-call (odd-call) spin mutants of [`GammaSpinMutant`].
    fn apply_spin_mutant(&mut self, k: &mut Array2<f64>) {
        let beta_call = self.calls % 2 == 1;
        match self.mutant {
            Some(GammaSpinMutant::AlphaOnBeta) if beta_call => {
                if let Some(prev) = &self.last {
                    k.assign(prev);
                }
            }
            Some(GammaSpinMutant::DropBeta) if beta_call => k.fill(0.0),
            _ => {}
        }
        if self.mutant == Some(GammaSpinMutant::AlphaOnBeta) && !beta_call {
            self.last = Some(k.clone());
        }
        self.calls += 1;
    }
}

impl KBuilder for GammaRshK<'_> {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        let n = d.nrows();
        let dm = [to_complex(d)];
        let mut j = [Array2::<C64>::zeros((n, n))];
        let mut kc = [Array2::<C64>::zeros((n, n))];
        self.jk.build(&dm, &mut j, &mut kc)?;
        let imag = kc[0].iter().fold(0.0f64, |m, z| m.max(z.im.abs()));
        if imag > IMAG_TOL {
            return Err(FerricError::General(format!(
                "gamma_rks_rsh: the k = 0 exchange of a real density has imaginary part {imag:.2e}"
            )));
        }
        *k = kc[0].mapv(|z| z.re);
        if self.mutant.is_some() {
            self.apply_spin_mutant(k);
        }
        Ok(0)
    }

    fn update_density(&mut self, _d: &Array2<f64>) {}

    fn reset(&mut self) {}
}

/// Closed-shell Gamma RKS nuclear gradient of a range-separated hybrid
/// (`scf` from [`crate::dft::gamma_rks_rsh`] with the exchange `rsh_k`: the
/// dense-AFT exchange with the dense-AFT Gamma J, or the RS-GDF exchange with
/// the RS-GDF Gamma J of the same aux and config), routed through the 1x1x1
/// k-point force ([`crate::kxcgrad::kpoint_rks_gradient`]): the Gamma point
/// IS the 1x1x1 mesh, so the Gamma density, Fock and MOs are the `k = 0`
/// ones, and the k-point force differentiates `rsh_k`'s source (the dense
/// kernels, whose J is the same G sphere as the Gamma J at equal precision;
/// or the `k = 0` fit, whose plain fit is the Gamma RS-GDF J).
///
/// * `hcore_cfg` — the hcore config of the energy (the 1x1x1 hcore is
///   rebuilt from it).
/// * `dft` — the SAME config the SCF ran with: functional, `grid` (the
///   atom-centred grid is rebuilt from it; a run on a caller-supplied grid,
///   `gamma_rks_rsh_on_grid`, is not covered), `xc.ao_threshold`, `exxdiv`.
/// * `cfg` — the HF-part settings (budget).
///
/// The k-point Fock guard compares the rebuilt `h + J + V_xc − K` with the
/// SCF's stored Fock, so a Gamma J from another source than `rsh_k` (dense J
/// under an RS-GDF exchange or the reverse) or a different grid is refused,
/// not silently differentiated. Refuses an open-shell result (the UKS twin is
/// [`gamma_uks_rsh_gradient`]).
pub fn gamma_rks_rsh_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &crate::hcore::PeriodicHcoreConfig,
    scf: &ferric_scf::result::ScfResult,
    dft: &crate::dft::GammaRksConfig,
    rsh_k: &GammaRshExchange,
    cfg: &crate::kgrad::KGradConfig,
) -> Result<crate::kxcgrad::KRksGradient, FerricError> {
    let who = "gamma_rks_rsh_gradient";
    let (mesh, hk, kscf) = gamma_route(who, cell, prep, hcore_cfg, scf, dft, rsh_k)?;
    let kcfg = crate::kxcgrad::KRksGradConfig {
        grad: *cfg,
        grid: crate::kxcgrad::KXcGrid::Atom(dft.grid.clone()),
        ao_threshold: dft.xc.ao_threshold,
        exxdiv: Some(dft.exxdiv),
        ..Default::default()
    };
    crate::kxcgrad::kpoint_rks_gradient(
        cell,
        prep,
        &mesh,
        hcore_cfg,
        &hk,
        rsh_k.force_jk(),
        &kscf,
        &dft.functional,
        &kcfg,
    )
}

/// Closed-shell Gamma RKS STRESS of a range-separated hybrid, routed through
/// the 1x1x1 k-point stress ([`crate::kxcstress::kpoint_rks_stress`]) exactly
/// as [`gamma_rks_rsh_gradient`] routes the force: same arguments, same
/// refusals (open-shell result, a config or Gamma J that is not the SCF's,
/// caught by the routed Fock guard); dense-AFT or RS-GDF `rsh_k`. `σ` is
/// returned unsymmetrised (the atom-centred grid is not rotation invariant;
/// see [`crate::kxcstress`]).
pub fn gamma_rks_rsh_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &crate::hcore::PeriodicHcoreConfig,
    scf: &ferric_scf::result::ScfResult,
    dft: &crate::dft::GammaRksConfig,
    rsh_k: &GammaRshExchange,
    cfg: &crate::kstress::KStressConfig,
) -> Result<crate::kxcstress::KRksStress, FerricError> {
    let who = "gamma_rks_rsh_stress";
    let (mesh, hk, kscf) = gamma_route(who, cell, prep, hcore_cfg, scf, dft, rsh_k)?;
    let kcfg = crate::kxcstress::KRksStressConfig {
        stress: *cfg,
        grid: crate::kxcgrad::KXcGrid::Atom(dft.grid.clone()),
        ao_threshold: dft.xc.ao_threshold,
        exxdiv: Some(dft.exxdiv),
        ..Default::default()
    };
    crate::kxcstress::kpoint_rks_stress(
        cell,
        prep,
        &mesh,
        hcore_cfg,
        &hk,
        rsh_k.stress_jk(),
        &kscf,
        &dft.functional,
        &kcfg,
    )
}

/// The open-shell tail of the routed RSH derivatives' restricted-only refusal.
const RSH_OPEN_SHELL_NOTE: &str = "an unrestricted result takes \
     gamma_uks_rsh_gradient / gamma_uks_rsh_stress, a restricted-open-shell one \
     gamma_roks_rsh_gradient (the Gamma ROKS range-separated stress has no routed entry; \
     kpoint_roks_stress on a 1x1x1 mesh differentiates it)";

/// The shared checks and 1x1x1 inputs of the routed Gamma RSH derivatives:
/// the dense kernels, the mesh, its hcore and the Gamma result as the 1x1x1
/// k-point result.
fn gamma_route(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &crate::hcore::PeriodicHcoreConfig,
    scf: &ferric_scf::result::ScfResult,
    dft: &crate::dft::GammaRksConfig,
    rsh_k: &GammaRshExchange,
) -> Result<
    (
        KPointMesh,
        crate::hcore::kpoint::PeriodicHcoreK,
        crate::kscf::KScfResult,
    ),
    FerricError,
> {
    rsh_k.check_for(who, &dft.functional, dft.exxdiv)?;
    one_point_inputs(who, RSH_OPEN_SHELL_NOTE, cell, prep, hcore_cfg, scf)
}

/// The 1x1x1 inputs of a routed closed-shell Gamma derivative: the mesh,
/// its hcore (rebuilt from `hcore_cfg`) and the restricted Gamma `scf` as the
/// 1x1x1 k-point result. Refuses an open-shell result (`open_shell_note`
/// ends the message: where an open-shell derivative lives, if anywhere).
pub(crate) fn one_point_inputs(
    who: &str,
    open_shell_note: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &crate::hcore::PeriodicHcoreConfig,
    scf: &ferric_scf::result::ScfResult,
) -> Result<
    (
        KPointMesh,
        crate::hcore::kpoint::PeriodicHcoreK,
        crate::kscf::KScfResult,
    ),
    FerricError,
> {
    if scf.spin != ferric_scf::result::Spin::Restricted {
        return Err(FerricError::General(format!(
            "{who}: closed-shell (restricted) results only; {open_shell_note}"
        )));
    }
    let mesh = KPointMesh::gamma_centred(cell, [1, 1, 1])?;
    let hk = crate::hcore::kpoint::periodic_hcore_kpts(cell, prep, &mesh, hcore_cfg)?;
    let kscf = gamma_as_one_point(scf, &mesh, hk.enn, cell.mol().nelec().max(0) as usize / 2);
    Ok((mesh, hk, kscf))
}

/// Open-shell Gamma UKS nuclear gradient of a range-separated hybrid (`scf`
/// from [`crate::dft::gamma_uks_rsh`] with the exchange `rsh_k` and the
/// Gamma J of the same source, dense AFT or RS-GDF), routed through the
/// 1x1x1 k-point UKS force ([`crate::kuxcgrad::kpoint_uks_gradient`]) exactly
/// as [`gamma_rks_rsh_gradient`] routes the closed shell: the Gamma spin
/// densities and spin Focks are the `k = 0` ones, `K_σ = K[D_σ]` with the
/// kernel and mixed Madelung constant of `rsh_k`.
///
/// Arguments as [`gamma_rks_rsh_gradient`] with `dft` the
/// [`crate::dft::GammaUksConfig`] of the SCF (functional, `grid` rebuilt as
/// an atom-centred grid, `xc.ao_threshold`, `exxdiv`; a run on a
/// caller-supplied grid is not covered). Both spin Focks pass the k-point
/// Fock guard and the UKS stationarity gate. Refuses a
/// restricted or restricted-open-shell result (the ROKS twin is
/// [`gamma_roks_rsh_gradient`]).
pub fn gamma_uks_rsh_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &crate::hcore::PeriodicHcoreConfig,
    scf: &ferric_scf::result::ScfResult,
    dft: &crate::dft::GammaUksConfig,
    rsh_k: &GammaRshExchange,
    cfg: &crate::kgrad::KGradConfig,
) -> Result<crate::kuxcgrad::KUksGradient, FerricError> {
    let who = "gamma_uks_rsh_gradient";
    let (mesh, hk, kscf) = uks_route(who, cell, prep, hcore_cfg, scf, dft, rsh_k)?;
    let kcfg = open_shell_kcfg(&dft.grid, dft.xc.ao_threshold, dft.exxdiv, cfg);
    crate::kuxcgrad::kpoint_uks_gradient(
        cell,
        prep,
        &mesh,
        hcore_cfg,
        &hk,
        rsh_k.force_jk(),
        &kscf,
        &dft.functional,
        &kcfg,
    )
}

/// Open-shell Gamma UKS STRESS of a range-separated hybrid, routed through
/// the 1x1x1 k-point UKS stress ([`crate::kuxcstress::kpoint_uks_stress`])
/// exactly as [`gamma_uks_rsh_gradient`] routes the force: same arguments
/// (`cfg` the HF-part stress settings; dense-AFT or RS-GDF `rsh_k`), same
/// refusals (a restricted or restricted-open-shell result, a config or Gamma
/// J that is not the SCF's, caught by the routed per-spin Fock guard). `σ`
/// is returned unsymmetrised (see [`crate::kxcstress`]).
pub fn gamma_uks_rsh_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &crate::hcore::PeriodicHcoreConfig,
    scf: &ferric_scf::result::ScfResult,
    dft: &crate::dft::GammaUksConfig,
    rsh_k: &GammaRshExchange,
    cfg: &crate::kstress::KStressConfig,
) -> Result<crate::kuxcstress::KUksStress, FerricError> {
    let who = "gamma_uks_rsh_stress";
    let (mesh, hk, kscf) = uks_route(who, cell, prep, hcore_cfg, scf, dft, rsh_k)?;
    let kcfg = crate::kuxcstress::KUksStressConfig {
        base: crate::kxcstress::KRksStressConfig {
            stress: *cfg,
            grid: crate::kxcgrad::KXcGrid::Atom(dft.grid.clone()),
            ao_threshold: dft.xc.ao_threshold,
            exxdiv: Some(dft.exxdiv),
            ..Default::default()
        },
        spin_mutation: None,
    };
    crate::kuxcstress::kpoint_uks_stress(
        cell,
        prep,
        &mesh,
        hcore_cfg,
        &hk,
        rsh_k.stress_jk(),
        &kscf,
        &dft.functional,
        &kcfg,
    )
}

/// The shared checks and 1x1x1 inputs of the routed Gamma UKS RSH
/// derivatives: the mesh, its hcore and the unrestricted Gamma result as the
/// 1x1x1 k-point result (spin densities, Focks, MOs); the J/K source is
/// `rsh_k`'s own ([`GammaRshExchange::force_jk`] / `stress_jk`).
fn uks_route(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &crate::hcore::PeriodicHcoreConfig,
    scf: &ferric_scf::result::ScfResult,
    dft: &crate::dft::GammaUksConfig,
    rsh_k: &GammaRshExchange,
) -> Result<
    (
        KPointMesh,
        crate::hcore::kpoint::PeriodicHcoreK,
        crate::kuscf::KUScfResult,
    ),
    FerricError,
> {
    rsh_k.check_for(who, &dft.functional, dft.exxdiv)?;
    if scf.spin != ferric_scf::result::Spin::Unrestricted {
        return Err(FerricError::General(format!(
            "{who}: unrestricted (UKS) results only, got {:?} (restricted: \
             gamma_rks_rsh_gradient / gamma_rks_rsh_stress; restricted-open-shell: \
             gamma_roks_rsh_gradient)",
            scf.spin
        )));
    }
    let (Some(db), Some(fb), Some(cb), Some(eb)) = (
        &scf.density_beta,
        &scf.fock_beta,
        &scf.mos_beta,
        &scf.eps_beta,
    ) else {
        return Err(FerricError::General(format!(
            "{who}: the unrestricted result lacks its beta density, Fock, MOs or energies"
        )));
    };
    let (na, nb) = crate::uhf::nocc_ab(cell.mol())?;
    let mesh = KPointMesh::gamma_centred(cell, [1, 1, 1])?;
    let hk = crate::hcore::kpoint::periodic_hcore_kpts(cell, prep, &mesh, hcore_cfg)?;
    let occ = |eps: &[f64], n: usize| -> Vec<f64> {
        (0..eps.len())
            .map(|i| if i < n { 1.0 } else { 0.0 })
            .collect()
    };
    let kscf = crate::kuscf::KUScfResult {
        energy: scf.energy,
        e_nuc: hk.enn,
        nalpha: na,
        nbeta: nb,
        eps_alpha: vec![scf.eps_alpha.clone()],
        eps_beta: vec![eb.clone()],
        mos_alpha: vec![to_complex(&scf.mos_alpha)],
        mos_beta: vec![to_complex(cb)],
        density_alpha: vec![to_complex(&scf.density_alpha)],
        density_beta: vec![to_complex(db)],
        fock_alpha: vec![to_complex(&scf.fock_alpha)],
        fock_beta: vec![to_complex(fb)],
        occ_alpha: vec![occ(&scf.eps_alpha, na)],
        occ_beta: vec![occ(eb, nb)],
        nocc_per_k_alpha: vec![na],
        nocc_per_k_beta: vec![nb],
        gap_alpha: None,
        gap_beta: None,
        s2: 0.0,
        converged: scf.converged,
        iterations: scf.iterations,
        max_error: 0.0,
        kpts: mesh.kpts().to_vec(),
        lindep: Default::default(),
    };
    Ok((mesh, hk, kscf))
}

/// The 1x1x1 open-shell force settings of a routed Gamma force: the SCF's
/// grid (rebuilt as an atom-centred grid), AO threshold and `exxdiv`.
fn open_shell_kcfg(
    grid: &crate::dft::PeriodicGridConfig,
    ao_threshold: f64,
    exxdiv: ExxDiv,
    cfg: &crate::kgrad::KGradConfig,
) -> crate::kuxcgrad::KUksGradConfig {
    crate::kuxcgrad::KUksGradConfig {
        base: crate::kxcgrad::KRksGradConfig {
            grad: *cfg,
            grid: crate::kxcgrad::KXcGrid::Atom(grid.clone()),
            ao_threshold,
            exxdiv: Some(exxdiv),
            ..Default::default()
        },
        spin_mutation: None,
    }
}

/// Restricted-open-shell Gamma ROKS nuclear gradient of a range-separated
/// hybrid (`scf` from [`crate::rohf::gamma_roks_rsh`] with the exchange
/// `rsh_k` and the Gamma J of the same source, dense AFT or RS-GDF), routed
/// through the 1x1x1 k-point ROKS force
/// ([`crate::kuxcgrad::kpoint_roks_gradient`]) as
/// [`gamma_uks_rsh_gradient`] routes UKS: the Gamma ROHF spin densities and
/// the one MO set are the `k = 0` ones, the spin Focks are rebuilt from them
/// (`K_σ = K[D_σ]` with the kernel and mixed Madelung constant of `rsh_k`),
/// `W = W_RO`, and the ROHF orbital-gradient gate applies.
///
/// The routed Fock guard compares the rebuilt spin Focks with the SCF's
/// (`ScfResult::rohf_spin_focks`, [`crate::kgrad::FOCK_CONSISTENCY_BAR`]), as
/// the UKS route does: a different grid, J source, functional or `exxdiv` is
/// refused. A result without spin Focks (hand-built) falls back to comparing
/// the Roothaan `F_eff` of the rebuilt spin Focks with `fock_alpha` (the
/// stored `F_eff`), which is blind to equal-and-opposite spin changes on its
/// diagonal blocks.
///
/// Arguments as [`gamma_uks_rsh_gradient`] with `dft` the
/// [`crate::rohf::GammaRoksConfig`] of the SCF (functional, `grid` rebuilt as
/// an atom-centred grid, `xc.ao_threshold`, `exxdiv`; a run on a
/// caller-supplied grid, `gamma_roks_rsh_on_grid`, is not covered). Refuses
/// a restricted or unrestricted result.
pub fn gamma_roks_rsh_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &crate::hcore::PeriodicHcoreConfig,
    scf: &ferric_scf::result::ScfResult,
    dft: &crate::rohf::GammaRoksConfig,
    rsh_k: &GammaRshExchange,
    cfg: &crate::kgrad::KGradConfig,
) -> Result<crate::kuxcgrad::KUksGradient, FerricError> {
    let who = "gamma_roks_rsh_gradient";
    rsh_k.check_for(who, &dft.functional, dft.exxdiv)?;
    let db = match (&scf.spin, &scf.density_beta) {
        (ferric_scf::result::Spin::RestrictedOpen, Some(db)) => db,
        _ => {
            return Err(FerricError::General(format!(
                "{who}: restricted-open-shell (ROKS) results with a beta density only, got {:?} \
                 (restricted: gamma_rks_rsh_gradient; unrestricted: gamma_uks_rsh_gradient)",
                scf.spin
            )))
        }
    };
    let mesh = KPointMesh::gamma_centred(cell, [1, 1, 1])?;
    let hk = crate::hcore::kpoint::periodic_hcore_kpts(cell, prep, &mesh, hcore_cfg)?;
    let (da, db) = ([to_complex(&scf.density_alpha)], [to_complex(db)]);
    let mos = [to_complex(&scf.mos_alpha)];
    let feff = [to_complex(&scf.fock_alpha)];
    let spin = scf
        .rohf_spin_focks
        .as_ref()
        .map(|(fa, fb)| ([to_complex(fa)], [to_complex(fb)]));
    let fock = match &spin {
        Some((fa, fb)) => crate::kuxcgrad::StoredFock::Spin(fa, fb),
        None => crate::kuxcgrad::StoredFock::Roothaan(&feff),
    };
    let kpts = mesh.kpts().to_vec();
    let inp =
        crate::kuxcgrad::SpinInput::restricted_open((&da, &db), fock, &mos, scf.converged, &kpts);
    let parts = crate::kuxcgrad::RoParts {
        cell,
        prep,
        mesh: &mesh,
        hcore_cfg,
        hk: &hk,
    };
    parts.roks_gradient(
        who,
        rsh_k.force_jk(),
        &inp,
        &dft.functional,
        &open_shell_kcfg(&dft.grid, dft.xc.ao_threshold, dft.exxdiv, cfg),
    )
}

/// A restricted Gamma result as the 1x1x1 k-point result (`k = 0`, complex
/// copies of the real matrices; `nocc` doubly occupied levels). The force
/// reads the density, the Fock, the convergence flag and the k-point.
fn gamma_as_one_point(
    scf: &ferric_scf::result::ScfResult,
    mesh: &KPointMesh,
    e_nuc: f64,
    nocc: usize,
) -> crate::kscf::KScfResult {
    let eps = scf.eps_alpha.clone();
    let occ: Vec<f64> = (0..eps.len())
        .map(|i| if i < nocc { 2.0 } else { 0.0 })
        .collect();
    let homo = nocc.checked_sub(1).map_or(f64::NEG_INFINITY, |i| eps[i]);
    let lumo = eps.get(nocc).copied().unwrap_or(f64::INFINITY);
    crate::kscf::KScfResult {
        energy: scf.energy,
        e_nuc,
        eps: vec![eps],
        mos: vec![to_complex(&scf.mos_alpha)],
        densities: vec![to_complex(&scf.density_total)],
        fock: vec![to_complex(&scf.fock_alpha)],
        occupations: vec![occ],
        nocc_per_k: vec![nocc],
        homo,
        lumo,
        converged: scf.converged,
        iterations: scf.iterations,
        max_error: 0.0,
        kpts: mesh.kpts().to_vec(),
        lindep: Default::default(),
        timings: Default::default(),
    }
}

/// How the bindings (Python, CLI) size the Gamma RSH exchange: the same
/// knobs their Gamma J/K already uses.
#[derive(Debug, Clone, Copy)]
pub struct GammaRshSizing {
    /// RS-GDF Ewald split (Bohr⁻¹) of the fit.
    pub rsgdf_omega: f64,
    /// RS-GDF memory budget (`None` = unified).
    pub budget_bytes: Option<usize>,
    /// Dense-AFT hard cap in bytes.
    pub max_eri_bytes: usize,
    /// The caller asked for the opt-in RS-GDF `range_split` on the Gamma J.
    /// The attenuated K is never split, so [`gamma_rks_any`] REFUSES this
    /// together with a range-separated functional instead of half-applying it.
    pub range_split: bool,
}

impl GammaRshExchange {
    /// Build the exchange for a binding: dense AFT when `aux` is `None`,
    /// otherwise the RS-GDF fit on `aux`.
    pub fn build_for_binding(
        cell: &Cell,
        prep: &PreparedBasis,
        aux: Option<&PreparedBasis>,
        s: &Array2<f64>,
        exxdiv: ExxDiv,
        sizing: GammaRshSizing,
        rsh: RshParams,
    ) -> Result<Self, FerricError> {
        match aux {
            None => {
                check_dense_budget(prep.nbasis(), sizing.max_eri_bytes)?;
                let cfg = KDenseAftConfig {
                    max_bytes: sizing.max_eri_bytes,
                    budget_bytes: sizing.budget_bytes,
                    ..Default::default()
                };
                Self::build_dense(cell, prep, s, exxdiv, &cfg, rsh)
            }
            Some(aux) => {
                let cfg = KRsGdfConfig {
                    gdf: crate::rsgdf::RsGdfConfig {
                        omega: sizing.rsgdf_omega,
                        exxdiv,
                        budget_bytes: sizing.budget_bytes,
                        ..Default::default()
                    },
                    mutation: None,
                };
                Self::build_rsgdf(cell, prep, aux, s, exxdiv, &cfg, rsh)
            }
        }
    }
}

/// Closed-shell Gamma RKS for ANY supported functional: a range-separated
/// hybrid takes the attenuated K of [`GammaRshExchange::build_for_binding`]
/// (built on `aux`, or dense AFT when `None`), everything else [`crate::dft::gamma_rks`].
/// The one entry the Python and CLI bindings share.
pub fn gamma_rks_any(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &crate::hcore::PeriodicHcore,
    ints: crate::uhf::GammaUhfIntegrals<'_>,
    aux: Option<&PreparedBasis>,
    sizing: GammaRshSizing,
    cfg: &crate::dft::GammaRksConfig,
) -> Result<crate::dft::GammaRksResult, FerricError> {
    Ok(gamma_rks_any_keep(cell, prep, hc, ints, aux, sizing, cfg)?.0)
}

/// [`gamma_rks_any`] that also returns the range-separated exchange the SCF
/// ran on (`None` for any other functional), so a binding can hand it to
/// [`gamma_rks_rsh_gradient`] / [`gamma_rks_rsh_stress`] without rebuilding
/// it.
pub fn gamma_rks_any_keep(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &crate::hcore::PeriodicHcore,
    ints: crate::uhf::GammaUhfIntegrals<'_>,
    aux: Option<&PreparedBasis>,
    sizing: GammaRshSizing,
    cfg: &crate::dft::GammaRksConfig,
) -> Result<(crate::dft::GammaRksResult, Option<GammaRshExchange>), FerricError> {
    check_range_split(&cfg.functional, sizing.range_split)?;
    match crate::rsh::resolve_periodic_functional_rsh(&cfg.functional)?.2 {
        Some(rsh) => {
            let ex = GammaRshExchange::build_for_binding(
                cell, prep, aux, &hc.s, cfg.exxdiv, sizing, rsh,
            )?;
            let r = crate::dft::gamma_rks_rsh(cell, prep, hc, ints, &ex, cfg)?;
            Ok((r, Some(ex)))
        }
        None => Ok((crate::dft::gamma_rks(cell, prep, hc, ints, cfg)?, None)),
    }
}

/// Open-shell Gamma UKS for ANY supported functional: the [`gamma_rks_any`]
/// routing for [`crate::dft::gamma_uks_rsh`] (range-separated hybrid) or
/// [`crate::dft::gamma_uks`] (everything else). The one entry the Python and
/// CLI bindings share. A pure SCAN / r2SCAN routes to [`crate::dft::gamma_uks`],
/// whose library energy is validated on the dense-AFT J/K (#492); the bindings
/// refuse it earlier ([`crate::dft::check_gamma_uks_request`], not yet wired,
/// #498), and the k-point `solve_kuks` on a 1x1x1 mesh is their route.
pub fn gamma_uks_any(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &crate::hcore::PeriodicHcore,
    ints: crate::uhf::GammaUhfIntegrals<'_>,
    aux: Option<&PreparedBasis>,
    sizing: GammaRshSizing,
    cfg: &crate::dft::GammaUksConfig,
) -> Result<crate::dft::GammaUksResult, FerricError> {
    Ok(gamma_uks_any_keep(cell, prep, hc, ints, aux, sizing, cfg)?.0)
}

/// [`gamma_uks_any`] that also returns the range-separated exchange the SCF
/// ran on (`None` for any other functional), so a binding can hand it to
/// [`gamma_uks_rsh_gradient`] without rebuilding it (the [`gamma_rks_any_keep`]
/// twin). The exchange stays resident through the force: on the dense AFT
/// its two complex k = 0 kernels, `32 nao^4` bytes ([`gamma_dense_bytes`]).
pub fn gamma_uks_any_keep(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &crate::hcore::PeriodicHcore,
    ints: crate::uhf::GammaUhfIntegrals<'_>,
    aux: Option<&PreparedBasis>,
    sizing: GammaRshSizing,
    cfg: &crate::dft::GammaUksConfig,
) -> Result<(crate::dft::GammaUksResult, Option<GammaRshExchange>), FerricError> {
    check_range_split(&cfg.functional, sizing.range_split)?;
    match crate::rsh::resolve_periodic_functional_rsh(&cfg.functional)?.2 {
        Some(rsh) => {
            let ex = GammaRshExchange::build_for_binding(
                cell, prep, aux, &hc.s, cfg.exxdiv, sizing, rsh,
            )?;
            let r = crate::dft::gamma_uks_rsh(cell, prep, hc, ints, &ex, cfg)?;
            Ok((r, Some(ex)))
        }
        None => Ok((crate::dft::gamma_uks(cell, prep, hc, ints, cfg)?, None)),
    }
}

/// Open-shell Gamma ROKS for ANY supported functional: the [`gamma_uks_any`]
/// routing for [`crate::rohf::gamma_roks_rsh`] (range-separated hybrid) or
/// [`crate::rohf::gamma_roks`] (LDA, GGA, global hybrids, and the pure SCAN /
/// r2SCAN energy on the dense-AFT J/K with the DIIS solver, #492). The one
/// entry the Python and CLI bindings share; they refuse a meta-GGA earlier
/// ([`crate::dft::check_gamma_roks_request`], not yet wired, #498).
pub fn gamma_roks_any(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &crate::hcore::PeriodicHcore,
    ints: crate::uhf::GammaUhfIntegrals<'_>,
    aux: Option<&PreparedBasis>,
    sizing: GammaRshSizing,
    cfg: &crate::rohf::GammaRoksConfig,
) -> Result<crate::rohf::GammaRoksResult, FerricError> {
    Ok(gamma_roks_any_keep(cell, prep, hc, ints, aux, sizing, cfg)?.0)
}

/// [`gamma_roks_any`] that also returns the range-separated exchange the SCF
/// ran on (`None` for any other functional), so a binding can hand it to
/// [`gamma_roks_rsh_gradient`] without rebuilding it (the
/// [`gamma_uks_any_keep`] twin; the same residency note applies).
pub fn gamma_roks_any_keep(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &crate::hcore::PeriodicHcore,
    ints: crate::uhf::GammaUhfIntegrals<'_>,
    aux: Option<&PreparedBasis>,
    sizing: GammaRshSizing,
    cfg: &crate::rohf::GammaRoksConfig,
) -> Result<(crate::rohf::GammaRoksResult, Option<GammaRshExchange>), FerricError> {
    check_range_split(&cfg.functional, sizing.range_split)?;
    match crate::rsh::resolve_periodic_functional_rsh(&cfg.functional)?.2 {
        Some(rsh) => {
            let ex = GammaRshExchange::build_for_binding(
                cell, prep, aux, &hc.s, cfg.exxdiv, sizing, rsh,
            )?;
            let r = crate::rohf::gamma_roks_rsh(cell, prep, hc, ints, &ex, cfg)?;
            Ok((r, Some(ex)))
        }
        None => Ok((crate::rohf::gamma_roks(cell, prep, hc, ints, cfg)?, None)),
    }
}

/// Refuse `range_split` with a range-separated functional at Gamma: the split
/// shapes only the Gamma J, never the attenuated K, so honouring it would be a
/// silent partial behaviour. Called at plan / preflight time by the bindings
/// and again by [`gamma_rks_any`]. Any other functional, or no split, is fine.
pub fn check_range_split(functional: &str, range_split: bool) -> Result<(), FerricError> {
    if range_split && is_range_separated(functional) {
        return Err(crate::dft::PeriodicDftError::Unsupported {
            feature: "range_split with a range-separated hybrid",
            reason: format!(
                "{functional}: range_split would shape only the Gamma J, never the attenuated \
                 K; drop range_split for a range-separated functional"
            ),
        }
        .into());
    }
    Ok(())
}

/// Bytes of dense-AFT tensors a Gamma run holds: the `8 nao^4` Gamma ERI, plus
/// for a range-separated hybrid the k = 0 exchange's two complex kernels
/// (`2 x 16 nao^4`, [`crate::kdense_aft`]).
pub fn gamma_dense_bytes(nao: usize, range_separated: bool) -> u128 {
    let n4 = (nao as u128).pow(4);
    n4 * if range_separated { 8 + 32 } else { 8 }
}

/// Refuse a dense Gamma RSH run whose TOTAL tensors ([`gamma_dense_bytes`])
/// exceed `cap_bytes`, before any is built.
pub fn check_dense_budget(nao: usize, cap_bytes: usize) -> Result<(), FerricError> {
    let need = gamma_dense_bytes(nao, true);
    if need > cap_bytes as u128 {
        return Err(FerricError::General(format!(
            "the dense AFT tensors of a range-separated hybrid need {need} bytes (nao = {nao}: \
             8 nao^4 for the Gamma ERI + 32 nao^4 for the attenuated exchange) > max_eri_gb cap \
             {cap_bytes} bytes. This path is a toy-scale oracle; use jk=\"rsgdf\" with an \
             auxbasis for larger cells."
        )));
    }
    Ok(())
}
