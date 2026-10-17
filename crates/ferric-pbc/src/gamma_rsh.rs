//! Range-separated exact exchange for the closed-shell Gamma-point RKS
//! ([`crate::dft::gamma_rks_rsh`]; mgoldey/ferric#354).
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

enum Source {
    Dense(KDenseAftEri),
    RsGdf(KRsGdf),
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
        Ok(Self {
            src: Source::RsGdf(gdf),
            rsh,
            exxdiv,
            madelung,
            unchecked: false,
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
        if self.unchecked {
            return Ok(());
        }
        if !is_range_separated(functional) {
            return Err(FerricError::General(format!(
                "gamma_rks_rsh: functional {functional:?} is not a range-separated hybrid \
                 (use gamma_rks)"
            )));
        }
        let want = crate::rsh::resolve_periodic_functional_rsh(functional)?
            .2
            .ok_or_else(|| FerricError::General("gamma_rks_rsh: no CAM parameters".into()))?;
        if want != self.rsh {
            return Err(FerricError::General(format!(
                "gamma_rks_rsh: the exchange was built for {:?} but {functional:?} has {want:?}",
                self.rsh
            )));
        }
        if exxdiv != self.exxdiv {
            return Err(FerricError::General(format!(
                "gamma_rks_rsh: exxdiv {exxdiv:?} differs from the one the exchange was built \
                 with ({:?})",
                self.exxdiv
            )));
        }
        Ok(())
    }

    /// The real K builder (fractions and Madelung term included).
    pub fn k_builder(&self) -> GammaRshK<'_> {
        let jk: Box<dyn KPointJk + '_> = match &self.src {
            Source::Dense(e) => Box::new(e.jk_builder()),
            Source::RsGdf(g) => Box::new(g.jk_builder()),
        };
        GammaRshK { jk }
    }
}

fn to_complex(m: &Array2<f64>) -> Array2<C64> {
    m.mapv(|x| C64::new(x, 0.0))
}

/// Real [`KBuilder`] over the `k = 0` complex exchange.
pub struct GammaRshK<'a> {
    jk: Box<dyn KPointJk + 'a>,
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
        Ok(0)
    }

    fn update_density(&mut self, _d: &Array2<f64>) {}

    fn reset(&mut self) {}
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
