//! Analytic stress tensor of the k-point open-shell KS-DFT energy
//! ([`crate::kuks::solve_kuks`]) for LDA, GGA and GLOBAL hybrid functionals
//! (exact-exchange fraction `a`), dense-AFT and RS-GDF J/K: the spin-resolved
//! twin of [`crate::kxcstress`] and the strain twin of [`crate::kuxcgrad`].
//!
//! # Convention and energy
//!
//! That of [`crate::kxcstress`] (`σ_ab = (1/Ω) dE/dε_ab`, `a_i → (1 + ε) a_i`,
//! `R_A → (1 + ε) R_A`, Miller indices and fractional `k` FIXED: no k-strain
//! term) and of [`crate::kuxcgrad`] for the energy:
//!
//! ```text
//! F_σ(k) = h(k) + J[D_α + D_β](k) − a K[D_σ](k) + V_xc,σ(k)
//! E = (1/N_k) Σ_k Σ_σ ½ Re tr[(h + F_σ^HF) D_σ] + E_xc[ρ_α, ρ_β] + E_nn
//! ```
//!
//! `D_σ` has unit occupation; `K[D_σ]` carries the Madelung shift and is scaled
//! as a whole by `a`.
//!
//! # Terms
//!
//! 1. **HF part with exchange fraction `a`**: the k HF stress
//!    ([`crate::kstress`]'s `assemble(KSpins::Unrestricted, .., exch_scale = a)`):
//!    per-spin Pulay weight `W = Σ_σ D_σ F_σ D_σ` with the spin Focks above, the
//!    per-spin dense K walks / fitted exchange pieces and the Madelung M-term
//!    `−a v_M Σ_σ D_σ S D_σ` with its strain, all scaled by `a`; J, the fit's
//!    Coulomb pieces and the nuclear terms are not.
//! 2. **Spin-resolved XC at fixed `D_α, D_β`.** The closed-shell strain of
//!    [`crate::kxcstress`] (AO strain with the grid point attached to its anchor
//!    `O_g`, `m1^k[a,b]_μ = Σ_L e^{ik·L} ∂_aχ_μ(r_g − X_L)(O_g − X_L)_b`, the
//!    Hessian moment `m2`, and the weight response) applied to each spin
//!    density with its own kernel. With `p^{σ,k}_μ = Σ_ν D_σ(k)_μν φ^k_ν*`,
//!    `md^{σ,k,i}_μ = Σ_ν D_σ(k)_μν (∂_iφ^k_ν)*` and the libxc partials
//!    `v_ρσ = ∂(ρε)/∂ρ_σ`, `v_σσσ`, `v_σαβ`:
//!
//!    ```text
//!    c^α_i = 2 v_σαα ∇_iρ_α + v_σαβ ∇_iρ_β ,   c^β_i = 2 v_σββ ∇_iρ_β + v_σαβ ∇_iρ_α
//!    xc_ao[a,b]     = (2/N_k) Σ_g w_g Σ_σ Σ_k Re Σ_μ [ m1^k[a,b]_μ ( v_ρσ p^{σ,k}_μ + Σ_i c^σ_i md^{σ,k,i}_μ )
//!                                                   + Σ_i c^σ_i m2^k[3i+a,b]_μ p^{σ,k}_μ ]
//!    xc_weight[a,b] = Σ_g (ρ_α + ρ_β) ε dw_g/dε_ab
//!    ```
//!
//!    (the `v_σαβ` cross term feeds the OTHER spin's gradient into `c^σ`). Gates
//!    are the energy's ([`crate::kuks::KPeriodicXcPolarized`]): spin `σ`'s
//!    `v_ρσ`, `c^σ` are zeroed where `ρ_σ ≤ DENSITY_FLOOR`; `ε_g` is ungated. In
//!    the closed-shell limit the two spins sum to the closed-shell term
//!    ([`crate::kxcstress`]) to roundoff.
//!
//! The atom grid is not rotation invariant, so `sigma` is returned
//! UNSYMMETRISED like the RKS entry ([`crate::kxcstress`] module doc);
//! [`KRksStress::symmetric`] and [`KRksStress::antisymmetry`] are the
//! diagnostics. A uniform grid gives a symmetric stress.
//!
//! # Fock guard and stationarity
//!
//! `F_α`, `F_β` are rebuilt (Madelung shift from [`KRksStressConfig::exxdiv`],
//! required for a hybrid) and compared with the SCF's stored spin Focks at
//! [`crate::kgrad::FOCK_CONSISTENCY_BAR`]. The stress is first order in the
//! departure from self-consistency; a state with
//! `max_{σ,k} |F_σ D_σ S − S D_σ F_σ|` above [`UKS_COMMUTATOR_TOL`] is refused.
//!
//! ## Calibration of the gate (stress side)
//!
//! The tolerance is the one derived for the FORCE ([`crate::kuxcgrad`]) and is
//! reused unchanged. Measured here (`tests/pbc_kuxcstress.rs`, H3 doublet, PBE,
//! 1x1x1, loose SCF): commutator 1.57e-4 gives a `dE/dε` error of 6.6e-5 against
//! the tight state, a slope of about 0.42 (one measured point; first order in the
//! commutator). At the tolerance 1e-5 that admits roughly 4e-6 in `dE/dε`
//! (about 140x the 3e-8 FD bar of the anchors, about 2 MPa on the 64 Bohr³ test
//! cell). So the gate is looser than the precision of the anchors: it refuses a
//! state that is plainly unconverged, it does not certify the stress to the
//! anchors' accuracy.
//!
//! UNVERIFIED RISK: the commutator is formed in the AO basis, while
//! [`crate::kuscf::solve_kuhf`] converges in the lindep-filtered orthogonal
//! basis. Whether a state converged there can show a larger AO commutator (or
//! the reverse) on a cell with discarded near-dependent functions has not been
//! tested; every fixture here is well conditioned.
//!
//! # Memory
//!
//! As [`crate::kxcstress`]: the Bloch AO cache of the energy is never retained;
//! a live chunk holds `40·nbf·N_k·chunk` complex AO planes shared by both spins
//! plus eight `nbf·chunk` work planes and the partial `V_xc,α`, `V_xc,β`
//! (`4 N_k nbf²`), pre-flighted on the ledger. The Fock-guard stage holds
//! `D_α, D_β, J_α, J_β, K_α, K_β, F_α, F_β, V_α, V_β` (`10 N_k n²` complex).
//!
//! # ROHF / ROKS ([`kpoint_roks_stress`], [`kpoint_rohf_stress`])
//!
//! The k-point ROHF/ROKS energy ([`crate::kroks`]) is the UHF/UKS functional
//! `E_U[D_α, D_β]` restricted to ONE orthonormal complex orbital set per k,
//! `C(k) = [C_c | C_o | C_v]` (`D_α = C_c C_c† + C_o C_o†`, `D_β = C_c C_c†`),
//! and the strain of that energy at fixed `C†SC = 1` is the open-shell
//! assembly of this module on the ROHF spin densities and the spin Focks
//! rebuilt from them (ROKS: the polarized `V_σ` of the XC strain above; ROHF:
//! zero XC, `a = 1`), with the orthonormality Lagrangian of the force
//! ([`crate::kuxcgrad`]) as the Pulay/overlap weight in place of `Σ_σ D_σ F_σ D_σ`:
//!
//! ```text
//! W_RO(k) = sym(D_α F_α D_α + D_α F_β D_β)        (sym = Hermitian part)
//! ```
//!
//! (`crate::kgrad::KSpins::RestrictedOpen`; it multiplies `dS/dε` in the Pulay,
//! `c0` and Madelung overlap virials exactly as it multiplies `dS/dR` in the
//! force; nothing else of the assembly changes). The Roothaan effective Fock
//! must never be used for `W`.
//!
//! `W_RO` exists exactly when the three ROHF orbital-gradient blocks vanish at
//! every k (closed-open of `F_β`, open-virtual of `F_α`, closed-virtual of
//! `F_α + F_β`). The entry points measure them from the DENSITIES (projectors
//! `P_c = D_β`, `P_o = D_α − D_β`, `P_v = 1 − D_α` in the `S`-orthonormal MO
//! basis; the largest spectral norm over k; see [`crate::kuxcgrad`] "The gate
//! is basis-independent") on the REBUILT spin Focks and refuse a state above
//! [`crate::grad::RO_ORBITAL_GRADIENT_TOL`] (the stress is first order in it).
//! At a converged state the unrestricted
//! form `Σ_σ D_σ F_σ D_σ` differs from `W_RO` only by ½ the closed-open β
//! gradient, so using it ([`KUStressMutation::RoWUhfForm`]) is an identity
//! there, and this gate is the only protection off stationarity (it is what
//! the Gamma entry points rely on too). The per-spin commutator gate of the
//! UKS entry does not apply (the per-spin commutators do not vanish at a ROHF
//! state); `hf.commutator` of the result holds the orbital gradient instead.
//! The gate's per-k transients ([`crate::kuxcgrad::RO_GATE_N2_ARRAYS`] `n²`
//! complex arrays, counted from the gate's code) are checked against the
//! ledger. The MOs must be one set per k-point (a short set is a typed error).
//!
//! ## Calibration of the ROHF gate (stress side)
//!
//! The tolerance is that of the force, reused unchanged, applied to the
//! basis-independent orbital gradient ([`crate::kuxcgrad`]). Measured here
//! (`tests/pbc_kuxcstress_ro.rs`, `ro_stress_gate_probe`, H3 doublet ROKS PBE,
//! 1x1x1, SCF stopped early, gate off, against the tight state): orbital
//! gradient `g` 6.9e-3 -> `dE/dε` error 2.4e-2, `g` 2.8e-4 -> 1.9e-4, `g` 9.3e-7
//! -> 3.7e-6, `g` 7.5e-9 -> 2.6e-8: first order (the last two states are 125x
//! apart in `g` and 139x in error) with a slope of 0.67 to 3.9. A second system
//! with a rank-2 closed-virtual block (H5 doublet, STO-3G, 1x1x1, ROKS PBE,
//! `ro_stress_gate_probe_rank2`) gives `g` 4.9e-4 -> 1.2e-3, 7.8e-5 -> 3.8e-4,
//! 2.7e-6 -> 6.5e-6 (slopes 2.4, 4.8, 2.4). MEASURED ON THESE TWO TINY CELLS
//! ONLY: for larger cells the per-k maximum is not the sum over orbitals or
//! atoms, and for blocks of rank > 1 the spectral norm is below the Frobenius
//! norm by up to `sqrt(rank)`; these numbers are not general bounds. On them,
//! at the tolerance 1e-5 the gate admits an error of up to about 4e-5 to 5e-5 in
//! `dE/dε` (about 1.5e3 times the 3e-8 FD bar of the anchors, about 18 MPa on
//! the 64 Bohr³ test cell): it refuses a plainly unconverged state, it does not
//! certify the stress to the anchors' accuracy. The unrestricted-form `W` is no
//! remedy off stationarity (comparable error at the same state).
//!
//! The ROHF/ROKS stress takes the dense-AFT or the (unsplit) RS-GDF J/K: the
//! fit pass of [`crate::kstress`] is generic over `KSpins` (`Unrestricted` and
//! `RestrictedOpen` pair `D_alpha`, `D_beta` identically) and `W_RO` enters only
//! as the overlap weight (including the fit's `M_g0 dS` term). Anchors
//! (`tests/pbc_kuxcstress_ro_rsgdf.rs`): closed-shell limit vs the RHF RS-GDF
//! stress, FD of the strained RS-GDF energy on all nine components (ROHF,
//! ROKS PBE0), and the RS-GDF / `W_RO` mutants.
//!
//! # Scope
//!
//! LDA, GGA and global hybrids. Range-separated hybrids, meta-GGA, VV10 and
//! double hybrids are refused with the typed errors of [`crate::kxcstress`]
//! (for ROKS the same; the ROHF entry has no functional). A ROKS state cannot
//! be handed to [`kpoint_uks_stress`] (its SCF type is
//! [`crate::kroks::KRoScfResult`], not [`KUScfResult`]).

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::dft::{BlochAoStrain, BlochPhase, LatticeAoHess};
use crate::grad::ao_atoms;
use crate::grad::RO_ORBITAL_GRADIENT_TOL;
use crate::hcore::kpoint::PeriodicHcoreK;
use crate::hcore::PeriodicHcoreConfig;
use crate::kgrad::{check_fock, check_inputs, jk_build, KGradJk, KSpins};
use crate::kpts::KPointMesh;
use crate::kroks::KRoScfResult;
use crate::kstress::{assemble, grad_jk, open_ledger, refuse_unsupported, KStressJk};
use crate::kuscf::KUScfResult;
pub use crate::kuxcgrad::UKS_COMMUTATOR_TOL;
use crate::kuxcgrad::{ro_gate_scratch_bytes, ro_orbital_gradient_k};
use crate::kxcgrad::{chunk_density, chunk_vxc, conj, hermitize};
use crate::kxcstress::{
    finish, resolve_global, strain_points_for, KRksStress, KRksStressConfig, KXcStressMutation,
    XcStressOut, AO_PLANES, CHUNK_TARGET_BYTES, KS_REMEDY, MAX_CHUNK, MIN_CHUNK,
};
use crate::lattice::Cell;
use crate::stress::{add_into, Mat3, ZERO3};
use ferric_core::FerricError;
use ferric_dft::density_on_grid::UksDensityGrid;
use ferric_dft::grid::GridPoint;
use ferric_dft::libxc::{xc_def_from_name_nspin, FunctionalFamily, XcDef};
use ferric_dft::vxc::{polarized_kernel, PolarizedKernel, DENSITY_FLOOR};
use ferric_integrals::basis_bridge::PreparedBasis;
use ndarray::{Array2, Axis};
use num_complex::Complex64 as C64;
use rayon::prelude::*;

/// Result of [`kpoint_uks_stress`] (the closed-shell result type).
pub type KUksStress = KRksStress;

/// Spin-specific deliberate defects for the mutation tests
/// (`tests/pbc_kuxcstress.rs`). Never set in production.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KUStressMutation {
    /// The XC strain term uses `v_ρβ`, `c^β` for the α density and vice versa
    /// (the potentials `V_σ` keep their own spin).
    XcSpinSwap,
    /// The β spin is dropped from the XC strain term.
    XcDropBeta,
    /// The `v_σαβ ∇ρ_other` cross term is dropped from `c^σ`.
    XcCrossDropped,
    /// The cross term uses the spin's OWN gradient (`v_σαβ ∇ρ_σ`).
    XcCrossOwn,
    /// The XC strain uses the total-density closed-shell form
    /// (`v_ρ = (v_ρα + v_ρβ)/2` on `D_α + D_β`).
    XcSpinAveraged,
    /// The Pulay `W` from `D_α F_β D_α + D_β F_α D_β`. Fock guard skipped.
    PulayFockSwap,
    /// `F_σ` rebuilt with the OTHER spin's exchange. Fock guard skipped.
    PulayKSwap,
    /// `F_σ` rebuilt with the Coulomb matrix of `D_α` only. Guard skipped.
    PulayJOneSpin,
    /// `F_σ` rebuilt without `V_xc,β`. Guard skipped.
    PulayNoVxcBeta,
    /// The stationarity gate skipped.
    StationarityGateOff,
    /// ROHF/ROKS: the energy-weighted matrix in the UNRESTRICTED form
    /// `Σ_σ D_σ F_σ D_σ`. An IDENTITY at a converged ROHF state (it differs
    /// from `W_RO` by ½ the closed-open β orbital gradient): the tests assert
    /// that, and that the orbital-gradient gate is what protects off
    /// stationarity.
    RoWUhfForm,
    /// ROHF/ROKS: `W = D_α F_α D_α` (the `D_α F_β D_β` cross term dropped).
    RoWDropCross,
    /// ROHF/ROKS: the orbital-gradient gate skipped.
    RoGateOff,
    /// ROHF/ROKS: [`Self::RoWUhfForm`] with the gate skipped (one mutation
    /// slot): what the unrestricted form gives off stationarity.
    RoWUhfFormGateOff,
}

/// Settings of [`kpoint_uks_stress`].
#[derive(Debug, Clone, Default)]
pub struct KUksStressConfig {
    /// The closed-shell settings (HF-part budget/mutation, grid, AO threshold,
    /// chunk cap, exxdiv, XC mutation).
    pub base: KRksStressConfig,
    /// TEST ONLY: a deliberate spin-specific defect.
    #[doc(hidden)]
    pub spin_mutation: Option<KUStressMutation>,
}

/// What a polarized XC stress response returns.
pub(crate) struct XcOutPolStress {
    e_xc: f64,
    electrons: f64,
    va: Vec<Array2<C64>>,
    vb: Vec<Array2<C64>>,
    ao: Mat3,
    weight: Mat3,
    npts: usize,
    chunk_points: usize,
    live_chunks: usize,
}

pub(crate) trait KUXcStressResponse {
    fn evaluate(
        &self,
        da: &[Array2<C64>],
        db: &[Array2<C64>],
        ledger: &Ledger,
    ) -> Result<XcOutPolStress, FerricError>;
}

/// Zero XC (exactness anchor against the exchange-bearing HF machinery).
struct NullXcPol;

impl KUXcStressResponse for NullXcPol {
    fn evaluate(
        &self,
        da: &[Array2<C64>],
        _db: &[Array2<C64>],
        _ledger: &Ledger,
    ) -> Result<XcOutPolStress, FerricError> {
        let n = da[0].nrows();
        let z = || da.iter().map(|_| Array2::zeros((n, n))).collect();
        Ok(XcOutPolStress {
            e_xc: 0.0,
            electrons: 0.0,
            va: z(),
            vb: z(),
            ao: ZERO3,
            weight: ZERO3,
            npts: 0,
            chunk_points: 0,
            live_chunks: 0,
        })
    }
}

struct GridXcPol<'a> {
    cell: &'a Cell,
    prep: &'a PreparedBasis,
    xc: XcDef,
    gga: bool,
    kpts: Vec<[f64; 3]>,
    cfg: &'a KUksStressConfig,
}

/// One spin's gated kernel vectors on a chunk: `v_ρσ` and `c^σ_i`.
struct SpinKernel {
    vr: Vec<f64>,
    c: Array2<f64>,
}

/// Which gradient feeds the `v_σαβ` cross term of `c^σ`.
#[derive(Clone, Copy)]
enum Cross {
    Other,
    Dropped,
    Own,
}

/// Spin `s`'s gated `v_ρσ` and `c^σ` (the energy's gates: each spin on its own
/// density).
fn spin_kernel(
    s: usize,
    cross: Cross,
    dens: &UksDensityGrid,
    kern: &PolarizedKernel,
    gga: bool,
) -> SpinKernel {
    let np = dens.rho_a.len();
    let (rho, vrho, vss, gself, gother) = if s == 0 {
        (
            &dens.rho_a,
            &kern.vrho_a,
            &kern.vsigma_aa,
            &dens.grad_a,
            &dens.grad_b,
        )
    } else {
        (
            &dens.rho_b,
            &kern.vrho_b,
            &kern.vsigma_bb,
            &dens.grad_b,
            &dens.grad_a,
        )
    };
    let vr = (0..np)
        .map(|g| if rho[g] > DENSITY_FLOOR { vrho[g] } else { 0.0 })
        .collect();
    let c = Array2::from_shape_fn((3, np), |(i, g)| {
        if gga && rho[g] > DENSITY_FLOOR {
            let x = match cross {
                Cross::Other => kern.vsigma_ab[g] * gother[(i, g)],
                Cross::Dropped => 0.0,
                Cross::Own => kern.vsigma_ab[g] * gself[(i, g)],
            };
            2.0 * vss[g] * gself[(i, g)] + x
        } else {
            0.0
        }
    });
    SpinKernel { vr, c }
}

/// One chunk's contributions.
struct ChunkPol {
    e: f64,
    ne: f64,
    va: Vec<Array2<C64>>,
    vb: Vec<Array2<C64>>,
    ao: Mat3,
    weight: Mat3,
}

/// Switches of the strain contraction (production values unless a mutant).
struct StrainOpts {
    use_gga: bool,
    pref: f64,
    kscale: f64,
    mutation: Option<KXcStressMutation>,
}

/// `Σ_g w_g Σ_k Re Σ_μ [..]` of one spin pass (module doc, item 2), added to `ao`.
fn ao_strain_pass(
    dm: &[Array2<C64>],
    sk: &SpinKernel,
    fb: &BlochAoStrain,
    w: &[f64],
    o: &StrainOpts,
    ao: &mut Mat3,
) {
    let nbf = dm[0].nrows();
    let np = w.len();
    for (k, dk) in dm.iter().enumerate() {
        let p = dk.dot(&conj(&fb.chi[k]));
        let md: Option<[Array2<C64>; 3]> = o.use_gga.then(|| {
            std::array::from_fn(|i| dk.dot(&conj(&fb.dchi[k].index_axis(Axis(0), i).to_owned())))
        });
        for g in 0..np {
            let wg = o.pref * w[g] * o.kscale;
            // q_μ = v_ρ p_μ + Σ_i c_i md^i_μ  (contracts with m1).
            let q: Vec<C64> = (0..nbf)
                .map(|mu| {
                    let mut t = p[(mu, g)] * sk.vr[g];
                    if let Some(md) = md.as_ref() {
                        if o.mutation != Some(KXcStressMutation::NoStrainMoment) {
                            for i in 0..3 {
                                t += md[i][(mu, g)] * sk.c[(i, g)];
                            }
                        }
                    }
                    t
                })
                .collect();
            for a in 0..3 {
                for b in 0..3 {
                    let mut t = C64::new(0.0, 0.0);
                    for mu in 0..nbf {
                        t += fb.m1[k][(a, b, mu, g)] * q[mu];
                    }
                    if o.use_gga && o.mutation != Some(KXcStressMutation::NoHessian) {
                        for i in 0..3 {
                            let mut h = C64::new(0.0, 0.0);
                            for mu in 0..nbf {
                                h += fb.m2[k][(3 * i + a, b, mu, g)] * p[(mu, g)];
                            }
                            t += h * sk.c[(i, g)];
                        }
                    }
                    ao[a][b] += wg * t.re;
                }
            }
        }
    }
}

impl GridXcPol<'_> {
    /// `V_xc,σ(k)` of one spin on a chunk from its TRUE kernel vectors.
    fn potential(&self, bl: &BlochAoStrain, wts: &[f64], sk: &SpinKernel) -> Vec<Array2<C64>> {
        chunk_vxc(&bl.chi, &bl.dchi, wts, &sk.vr, &sk.c, self.gga)
    }

    /// The (density, kernel) pairs of the strain passes (possibly mutated).
    fn passes<'b>(
        &self,
        da: &'b [Array2<C64>],
        db: &'b [Array2<C64>],
        dsum: Option<&'b [Array2<C64>]>,
        dens: &UksDensityGrid,
        kern: &PolarizedKernel,
    ) -> Vec<(&'b [Array2<C64>], SpinKernel)> {
        let cross = match self.cfg.spin_mutation {
            Some(KUStressMutation::XcCrossDropped) => Cross::Dropped,
            Some(KUStressMutation::XcCrossOwn) => Cross::Own,
            _ => Cross::Other,
        };
        let fka = spin_kernel(0, cross, dens, kern, self.gga);
        let fkb = spin_kernel(1, cross, dens, kern, self.gga);
        match self.cfg.spin_mutation {
            Some(KUStressMutation::XcSpinSwap) => vec![(da, fkb), (db, fka)],
            Some(KUStressMutation::XcDropBeta) => vec![(da, fka)],
            Some(KUStressMutation::XcSpinAveraged) => {
                let np = fka.vr.len();
                let avg = SpinKernel {
                    vr: (0..np).map(|g| 0.5 * (fka.vr[g] + fkb.vr[g])).collect(),
                    c: Array2::from_shape_fn((3, np), |(i, g)| {
                        0.5 * (fka.c[(i, g)] + fkb.c[(i, g)])
                    }),
                };
                vec![(dsum.expect("dsum for XcSpinAveraged"), avg)]
            }
            _ => vec![(da, fka), (db, fkb)],
        }
    }
}

impl KUXcStressResponse for GridXcPol<'_> {
    fn evaluate(
        &self,
        da: &[Array2<C64>],
        db: &[Array2<C64>],
        ledger: &Ledger,
    ) -> Result<XcOutPolStress, FerricError> {
        let mut led = ledger.clone();
        let nbf = self.prep.nbasis();
        let nk = da.len();
        let base = &self.cfg.base;
        let mutation = base.mutation;
        let sp = strain_points_for(self.cell, &base.grid, &mut led)?;
        let points = &sp.points;
        let ev = LatticeAoHess::new(
            self.cell,
            self.prep.basis_set(),
            points,
            base.ao_threshold,
            &led,
        )?;
        if ev.nbf() != nbf || ao_atoms(self.prep).len() != nbf {
            return Err(FerricError::General(format!(
                "k-point UKS XC stress: lattice AO count {} != prepared basis {nbf}",
                ev.nbf()
            )));
        }
        // Chunking and the memory pre-flight (two spins' work planes, two
        // partial potentials).
        let per_point = |np: usize| -> usize {
            bytes_of((nk * nbf * np) as u64, 16 * AO_PLANES)
                .saturating_add(bytes_of((nbf * np) as u64, 16 * 8))
        };
        let fixed = bytes_of((nk * nbf * nbf) as u64, 16 * 4);
        let mut chunk_points = MAX_CHUNK;
        while chunk_points > MIN_CHUNK && per_point(chunk_points) > CHUNK_TARGET_BYTES {
            chunk_points /= 2;
        }
        let per_chunk = per_point(chunk_points).saturating_add(fixed);
        led.check(
            &format!(
                "k-point UKS XC stress AO moment chunk ({chunk_points} points, nbf = {nbf}, N_k = {nk})"
            ),
            per_chunk,
        )?;
        let par = rayon::current_num_threads()
            .max(1)
            .min(base.max_live_chunks.unwrap_or(usize::MAX).max(1))
            .min((led.remaining() / per_chunk.max(1)).max(1));
        led.reserve(
            &format!("k-point UKS XC stress chunks ({par} x {chunk_points} points)"),
            per_chunk.saturating_mul(par),
        )?;
        let chunks = LatticeAoHess::chunks(points, chunk_points);
        let opts = StrainOpts {
            use_gga: self.gga && mutation != Some(KXcStressMutation::NoGga),
            pref: if mutation == Some(KXcStressMutation::WrongFactor) {
                1.0
            } else {
                2.0
            },
            kscale: if mutation == Some(KXcStressMutation::NoInvNk) {
                1.0
            } else {
                1.0 / nk as f64
            },
            mutation,
        };
        let alt_phase = match mutation {
            Some(KXcStressMutation::NoPhase) => Some(BlochPhase::None),
            Some(KXcStressMutation::ConjPhase) => Some(BlochPhase::Conj),
            _ => None,
        };
        let dsum: Option<Vec<Array2<C64>>> = (self.cfg.spin_mutation
            == Some(KUStressMutation::XcSpinAveraged))
        .then(|| da.iter().zip(db).map(|(a, b)| a + b).collect());

        let run = |idx: &Vec<usize>| -> Result<ChunkPol, FerricError> {
            let pts: Vec<GridPoint> = idx.iter().map(|&i| points[i]).collect();
            let xyz: Vec<[f64; 3]> = pts.iter().map(|g| g.xyz).collect();
            let anchors: Vec<[f64; 3]> = match mutation {
                Some(KXcStressMutation::NoPointMotion) => vec![[0.0; 3]; idx.len()],
                Some(KXcStressMutation::PointSelf) => xyz.clone(),
                _ => idx.iter().map(|&i| sp.anchors[i]).collect(),
            };
            let np = pts.len();
            let bl: BlochAoStrain =
                ev.eval_bloch_strain(&xyz, &anchors, &self.kpts, BlochPhase::True, self.gga)?;
            let dena = chunk_density(da, &bl.chi, &bl.dchi, self.gga);
            let denb = chunk_density(db, &bl.chi, &bl.dchi, self.gga);
            let mut sigma = Array2::<f64>::zeros((3, np));
            for g in 0..np {
                let dot = |x: &Array2<f64>, y: &Array2<f64>| -> f64 {
                    (0..3).map(|ax| x[(ax, g)] * y[(ax, g)]).sum()
                };
                sigma[(0, g)] = dot(&dena.grad, &dena.grad);
                sigma[(1, g)] = dot(&dena.grad, &denb.grad);
                sigma[(2, g)] = dot(&denb.grad, &denb.grad);
            }
            let dens = UksDensityGrid {
                rho_a: dena.rho,
                rho_b: denb.rho,
                grad_a: dena.grad,
                grad_b: denb.grad,
                sigma,
            };
            let kern = polarized_kernel(&dens, None, &self.xc);
            let mut e = 0.0;
            let mut ne = 0.0;
            let mut weight = ZERO3;
            for (g, &gi) in idx.iter().enumerate() {
                let rt = dens.rho_a[g] + dens.rho_b[g];
                e += pts[g].weight * rt * kern.exc[g];
                ne += pts[g].weight * rt;
                add_into(&mut weight, &sp.dweight[gi], rt * kern.exc[g]);
            }
            let wts: Vec<f64> = pts.iter().map(|p| p.weight).collect();
            // V_xc,σ(k) with the TRUE kernel vectors.
            let va = self.potential(
                &bl,
                &wts,
                &spin_kernel(0, Cross::Other, &dens, &kern, self.gga),
            );
            let vb = self.potential(
                &bl,
                &wts,
                &spin_kernel(1, Cross::Other, &dens, &kern, self.gga),
            );
            // Strain side. The mutant phases evaluate their own moments; the
            // density above keeps the true phase.
            let alt: Option<BlochAoStrain> = alt_phase
                .map(|ph| ev.eval_bloch_strain(&xyz, &anchors, &self.kpts, ph, self.gga))
                .transpose()?;
            let fb: &BlochAoStrain = alt.as_ref().unwrap_or(&bl);
            let mut ao = ZERO3;
            if mutation != Some(KXcStressMutation::NoAo) {
                for (dm, sk) in self.passes(da, db, dsum.as_deref(), &dens, &kern) {
                    ao_strain_pass(dm, &sk, fb, &wts, &opts, &mut ao);
                }
            }
            Ok(ChunkPol {
                e,
                ne,
                va,
                vb,
                ao,
                weight,
            })
        };

        // Concurrency-sized batches reduced in chunk order: memory is bounded by
        // `par` chunks and the sums do not depend on the thread count.
        let zeros = || -> Vec<Array2<C64>> { (0..nk).map(|_| Array2::zeros((nbf, nbf))).collect() };
        let mut out = XcOutPolStress {
            e_xc: 0.0,
            electrons: 0.0,
            va: zeros(),
            vb: zeros(),
            ao: ZERO3,
            weight: ZERO3,
            npts: points.len(),
            chunk_points,
            live_chunks: par,
        };
        for batch in chunks.chunks(par) {
            let outs: Vec<Result<ChunkPol, FerricError>> = batch.par_iter().map(&run).collect();
            for c in outs {
                let c = c?;
                out.e_xc += c.e;
                out.electrons += c.ne;
                for (acc, x) in out.va.iter_mut().zip(&c.va) {
                    *acc += x;
                }
                for (acc, x) in out.vb.iter_mut().zip(&c.vb) {
                    *acc += x;
                }
                add_into(&mut out.ao, &c.ao, 1.0);
                add_into(&mut out.weight, &c.weight, 1.0);
            }
        }
        if !out.e_xc.is_finite()
            || out
                .va
                .iter()
                .chain(out.vb.iter())
                .any(|m| m.iter().any(|z| !z.re.is_finite()))
            || out
                .ao
                .iter()
                .chain(out.weight.iter())
                .flatten()
                .any(|x| !x.is_finite())
        {
            return Err(FerricError::General(
                "k-point UKS XC stress: non-finite E_xc/V_xc/stress".into(),
            ));
        }
        for m in out.va.iter_mut().chain(out.vb.iter_mut()) {
            *m = hermitize(m);
        }
        if mutation == Some(KXcStressMutation::NoWeightDeriv) {
            out.weight = ZERO3;
        }
        Ok(out)
    }
}

fn grid_response<'a>(
    cell: &'a Cell,
    prep: &'a PreparedBasis,
    mesh: &KPointMesh,
    functional: &str,
    cfg: &'a KUksStressConfig,
) -> Result<(GridXcPol<'a>, f64), FerricError> {
    let (_xc1, exch) = resolve_global(functional)?;
    let xc = xc_def_from_name_nspin(functional, 2)
        .map_err(|e| FerricError::General(format!("{functional}: spin-polarized: {e:?}")))?;
    let gga = xc
        .funcs
        .iter()
        .any(|f| !matches!(f.family(), FunctionalFamily::Lda));
    Ok((
        GridXcPol {
            cell,
            prep,
            xc,
            gga,
            kpts: mesh.kpts().to_vec(),
            cfg,
        },
        exch,
    ))
}

/// k-point open-shell UKS stress (module doc) of a converged
/// [`crate::kuks::solve_kuks`] result (`KUksResult::scf`) for an LDA, GGA or
/// global-hybrid `functional` (a hybrid also needs
/// [`KRksStressConfig::exxdiv`], here `cfg.base.exxdiv`). Arguments as
/// [`crate::kxcstress::kpoint_rks_stress`] with the SCF a [`KUScfResult`].
///
/// Refuses (typed errors) range-separated hybrids, meta-GGA, VV10/double
/// hybrids, a hybrid without `cfg.base.exxdiv`, an SCF whose spin Focks the
/// rebuilt `h + J − a K_σ + V_xc,σ` do not reproduce, an unconverged SCF and a
/// non-stationary state (`max |F_σ D_σ S − S D_σ F_σ|` above
/// [`UKS_COMMUTATOR_TOL`]).
#[allow(clippy::too_many_arguments)]
pub fn kpoint_uks_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KStressJk<'_>,
    scf: &KUScfResult,
    functional: &str,
    cfg: &KUksStressConfig,
) -> Result<KUksStress, FerricError> {
    let (resp, exch) = grid_response(cell, prep, mesh, functional, cfg)?;
    uks_stress_with(
        "kpoint_uks_stress",
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        jk,
        &uks_input(scf),
        &resp,
        exch,
        cfg,
    )
}

/// k-point ROKS stress of a converged [`crate::kroks::solve_kroks`] result
/// (`KRoksResult::scf`): the open-shell assembly of [`kpoint_uks_stress`] on
/// the ROHF spin densities and spin Focks with the orthonormality Lagrangian
/// `W_RO = sym(D_α F_α D_α + D_α F_β D_β)` at each k (module doc "ROHF /
/// ROKS"). Arguments and refusals as [`kpoint_uks_stress`], except that the
/// stationarity gate is the ROHF orbital gradient (a state whose closed-open
/// `f_β`, open-virtual `f_α` or closed-virtual `f_α + f_β` block, at any k,
/// exceeds [`crate::grad::RO_ORBITAL_GRADIENT_TOL`] is refused), the MO set
/// must be one per k-point.
///
/// GATE LOOSENESS (module doc "Calibration of the ROHF gate"): the orbital
/// gradient is refused above [`crate::grad::RO_ORBITAL_GRADIENT_TOL`] = 1e-5,
/// and the stress error is first order in it with a measured slope of 0.67 to
/// 3.9, so a state at the tolerance can carry a `dE/dε` error of about 4e-5
/// (~18 MPa on the 64 Bohr³ test cell, ~1e3 times the FD bar of the anchors;
/// measured on H3 and H5 STO-3G at 1x1x1 only, not a general bound).
/// The gate refuses a plainly unconverged state; it does not certify the stress
/// to the anchors' accuracy, so converge the SCF well below the tolerance.
#[allow(clippy::too_many_arguments)]
pub fn kpoint_roks_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KStressJk<'_>,
    scf: &KRoScfResult,
    functional: &str,
    cfg: &KUksStressConfig,
) -> Result<KUksStress, FerricError> {
    let (resp, exch) = grid_response(cell, prep, mesh, functional, cfg)?;
    uks_stress_with(
        "kpoint_roks_stress",
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        jk,
        &ro_input(scf),
        &resp,
        exch,
        cfg,
    )
}

/// k-point ROHF stress of a converged [`crate::kroks::solve_krohf`] result:
/// [`kpoint_roks_stress`] with a zero XC and `a = 1` (`exxdiv` is the SCF's,
/// replacing `cfg.base.exxdiv`). The gate looseness of [`kpoint_roks_stress`]
/// applies: a state at the tolerance can carry a `dE/dε` error of about 4e-5.
#[allow(clippy::too_many_arguments)]
pub fn kpoint_rohf_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KStressJk<'_>,
    scf: &KRoScfResult,
    exxdiv: ExxDiv,
    cfg: &KUksStressConfig,
) -> Result<KUksStress, FerricError> {
    let mut c = cfg.clone();
    c.base.exxdiv = Some(exxdiv);
    uks_stress_with(
        "kpoint_rohf_stress",
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        jk,
        &ro_input(scf),
        &NullXcPol,
        1.0,
        &c,
    )
}

/// TEST/DIAGNOSTIC: the restricted-open-shell machinery with a zero XC and
/// exact-exchange fraction `exch`, on an SCF run by `solve_kroks_injected` with
/// no XC and the same `exch` (a hybrid, `exch != 0`, needs `cfg.base.exxdiv`).
/// With `exch = 1` it is [`kpoint_rohf_stress`] reached through the ROKS
/// route's own SCF driver.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn kpoint_null_xc_roks_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KStressJk<'_>,
    scf: &KRoScfResult,
    exch: f64,
    cfg: &KUksStressConfig,
) -> Result<KUksStress, FerricError> {
    uks_stress_with(
        "kpoint_null_xc_roks_stress",
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        jk,
        &ro_input(scf),
        &NullXcPol,
        exch,
        cfg,
    )
}

/// The converged spin state a stress differentiates (from a [`KUScfResult`]
/// or a [`KRoScfResult`]).
struct SpinInput<'a> {
    da: &'a [Array2<C64>],
    db: &'a [Array2<C64>],
    fa: &'a [Array2<C64>],
    fb: &'a [Array2<C64>],
    converged: bool,
    kpts: &'a [[f64; 3]],
    /// ROHF/ROKS: the one orthonormal MO set per k.
    mos: Option<&'a [Array2<C64>]>,
}

fn uks_input(scf: &KUScfResult) -> SpinInput<'_> {
    SpinInput {
        da: &scf.density_alpha,
        db: &scf.density_beta,
        fa: &scf.fock_alpha,
        fb: &scf.fock_beta,
        converged: scf.converged,
        kpts: &scf.kpts,
        mos: None,
    }
}

fn ro_input(scf: &KRoScfResult) -> SpinInput<'_> {
    SpinInput {
        da: &scf.density_alpha,
        db: &scf.density_beta,
        fa: &scf.fock_alpha,
        fb: &scf.fock_beta,
        converged: scf.converged,
        kpts: &scf.kpts,
        mos: Some(&scf.mos),
    }
}

/// TEST/DIAGNOSTIC: the open-shell KS machinery with a zero XC and
/// exact-exchange fraction `exch`, on an SCF run by `solve_kuks_injected` with a
/// zero polarized XC builder and the same `exch`. With `exch = 1` it is the
/// k-point UHF stress ([`crate::kstress::kpoint_uhf_stress`]). A hybrid
/// (`exch != 0`) needs `cfg.base.exxdiv`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn kpoint_null_xc_uks_stress(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KStressJk<'_>,
    scf: &KUScfResult,
    exch: f64,
    cfg: &KUksStressConfig,
) -> Result<KUksStress, FerricError> {
    uks_stress_with(
        "kpoint_null_xc_uks_stress",
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        jk,
        &uks_input(scf),
        &NullXcPol,
        exch,
        cfg,
    )
}

/// Bytes of the Fock-guard matrices `D_α, D_β, J_α, J_β, K_α, K_β, F_α, F_β,
/// V_α, V_β`, each `N_k n²` complex.
fn guard_bytes(nk: usize, n: usize) -> usize {
    bytes_of((nk * n * n) as u64, 16 * 10)
}

/// The Madelung shift of the SCF's K (0 for a pure functional: no K).
fn madelung_shift(
    who: &str,
    cell: &Cell,
    mesh: &KPointMesh,
    exch: f64,
    cfg: &KUksStressConfig,
) -> Result<f64, FerricError> {
    if exch == 0.0 {
        return Ok(0.0);
    }
    let exxdiv = cfg.base.exxdiv.ok_or_else(|| {
        FerricError::General(format!(
            "{who}: a hybrid functional (exact exchange {exch}) needs \
             KRksStressConfig::exxdiv, the treatment the SCF ran (its K carries the \
             Madelung shift)"
        ))
    })?;
    Ok(match exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => mesh.madelung(cell)?,
    })
}

/// Densities, Coulomb and exchange matrices of the Fock rebuild.
struct SpinJk {
    ja: Vec<Array2<C64>>,
    jb: Vec<Array2<C64>>,
    ka: Vec<Array2<C64>>,
    kb: Vec<Array2<C64>>,
}

/// `F_σ = h + J − a K_σ + V_xc,σ` per k, with the spin mutants applied.
fn rebuild_fock(
    s: usize,
    hk: &PeriodicHcoreK,
    jk: &SpinJk,
    xo: &XcOutPolStress,
    exch: f64,
    cfg: &KUksStressConfig,
) -> Vec<Array2<C64>> {
    let smut = cfg.spin_mutation;
    (0..hk.h.len())
        .map(|q| {
            let jtot = if smut == Some(KUStressMutation::PulayJOneSpin) {
                jk.ja[q].clone()
            } else {
                &jk.ja[q] + &jk.jb[q]
            };
            let mut base = &hk.h[q] + &jtot;
            if exch != 0.0 {
                let swap = smut == Some(KUStressMutation::PulayKSwap);
                let kk = if (s == 0) != swap {
                    &jk.ka[q]
                } else {
                    &jk.kb[q]
                };
                base.scaled_add(C64::new(-exch, 0.0), kk);
            }
            let base = hermitize(&base);
            let vxc = if s == 0 { &xo.va[q] } else { &xo.vb[q] };
            let drop_v = cfg.base.mutation == Some(KXcStressMutation::PulayNoVxc)
                || (s == 1 && smut == Some(KUStressMutation::PulayNoVxcBeta));
            if drop_v {
                base
            } else {
                hermitize(&(&base + vxc))
            }
        })
        .collect()
}

/// True when no Fock-tampering mutant is active, so the Fock guard and the
/// stationarity gate apply.
fn guard_applies(cfg: &KUksStressConfig) -> bool {
    let spin_ok = cfg.spin_mutation.map_or(true, |m| {
        !matches!(
            m,
            KUStressMutation::PulayFockSwap
                | KUStressMutation::PulayKSwap
                | KUStressMutation::PulayJOneSpin
                | KUStressMutation::PulayNoVxcBeta
        )
    });
    spin_ok
        && !matches!(
            cfg.base.mutation,
            Some(KXcStressMutation::PulayNoVxc) | Some(KXcStressMutation::KWrongDensity)
        )
}

/// The shared body: XC response, per-spin Fock guard, HF assembly at exchange
/// fraction `a`, stationarity gate.
#[allow(clippy::too_many_arguments)]
fn uks_stress_with(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KStressJk<'_>,
    inp: &SpinInput<'_>,
    xc: &dyn KUXcStressResponse,
    exch: f64,
    cfg: &KUksStressConfig,
) -> Result<KUksStress, FerricError> {
    if !(0.0..=1.0).contains(&exch) {
        return Err(FerricError::General(format!(
            "{who}: exact-exchange fraction {exch} outside [0, 1]"
        )));
    }
    refuse_unsupported(who, &jk)?;
    let gj: KGradJk<'_> = grad_jk(&jk);
    check_inputs(
        who,
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &gj,
        &[inp.da, inp.db],
        inp.converged,
        inp.kpts,
    )?;
    let ro = inp.mos.is_some();
    let nk = mesh.nk();
    let vm = madelung_shift(who, cell, mesh, exch, cfg)?;
    let mut ledger = open_ledger(prep, mesh, &cfg.base.stress, true)?;
    let da = inp.da.to_vec();
    let db = inp.db.to_vec();
    let xo = xc.evaluate(&da, &db, &ledger)?;
    ledger.reserve(
        &format!(
            "k UKS stress Fock-guard matrices D_a, D_b, J_a, J_b, K_a, K_b, F_a, F_b, V_a, V_b \
             (n = {}, N_k = {nk})",
            prep.nbasis()
        ),
        guard_bytes(nk, prep.nbasis()),
    )?;
    let (ja, mut ka) = jk_build(&gj, vm, &da)?;
    let (jb, mut kb) = jk_build(&gj, vm, &db)?;
    if cfg.base.mutation == Some(KXcStressMutation::KWrongDensity) && nk > 1 {
        let shift = |d: &[Array2<C64>]| -> Vec<Array2<C64>> {
            (0..nk).map(|q| d[(q + 1) % nk].clone()).collect()
        };
        ka = jk_build(&gj, vm, &shift(&da))?.1;
        kb = jk_build(&gj, vm, &shift(&db))?.1;
    }
    let sj = SpinJk { ja, jb, ka, kb };
    let fa = rebuild_fock(0, hk, &sj, &xo, exch, cfg);
    let fb = rebuild_fock(1, hk, &sj, &xo, exch, cfg);
    drop(sj);
    let guard = guard_applies(cfg);
    let mismatch = if guard {
        let ma = check_fock(who, "fock_alpha", &fa, inp.fa, KS_REMEDY)?;
        let mb = check_fock(who, "fock_beta", &fb, inp.fb, KS_REMEDY)?;
        ma.max(mb)
    } else {
        0.0
    };
    let (fa, fb) = if cfg.spin_mutation == Some(KUStressMutation::PulayFockSwap) {
        (fb, fa)
    } else {
        (fa, fb)
    };
    // ROHF/ROKS: the orbital-gradient gate on the REBUILT spin Focks (the
    // stress is first order in it), before the expensive assembly.
    let og = match inp.mos {
        Some(mos) => {
            ledger.check(
                &format!(
                    "k RO stress orbital-gradient gate scratch (n = {}, per-k transients)",
                    prep.nbasis()
                ),
                ro_gate_scratch_bytes(prep.nbasis()),
            )?;
            let g = ro_orbital_gradient_k(who, mos, &hk.s, &da, &db, &fa, &fb)?;
            let gate_on = !matches!(
                cfg.spin_mutation,
                Some(KUStressMutation::RoGateOff) | Some(KUStressMutation::RoWUhfFormGateOff)
            );
            if gate_on && (g.is_nan() || g > RO_ORBITAL_GRADIENT_TOL) {
                return Err(FerricError::General(format!(
                    "{who}: ROHF orbital gradient {g:.3e} exceeds {RO_ORBITAL_GRADIENT_TOL:.0e}: \
                     not a stationary ROHF state (the stress is first order in it); converge the \
                     SCF tighter"
                )));
            }
            Some(g)
        }
        None => None,
    };
    // What the HF assembly is told (the exchange-scale defects change only this).
    let a_asm = match cfg.base.mutation {
        Some(KXcStressMutation::ExchUnscaled) if exch != 0.0 => 1.0,
        Some(KXcStressMutation::ExchScaledTwice) => exch * exch,
        _ => exch,
    };
    let spmut = cfg.spin_mutation;
    let spins = if !ro
        || matches!(
            spmut,
            Some(KUStressMutation::RoWUhfForm) | Some(KUStressMutation::RoWUhfFormGateOff)
        ) {
        KSpins::Unrestricted { da, db, fa, fb }
    } else if spmut == Some(KUStressMutation::RoWDropCross) {
        let zb = fb.iter().map(|m| Array2::zeros(m.dim())).collect();
        KSpins::RestrictedOpen { da, db, fa, fb: zb }
    } else {
        KSpins::RestrictedOpen { da, db, fa, fb }
    };
    let mut hf = assemble(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &gj,
        spins,
        vm,
        a_asm,
        &cfg.base.stress,
        &mut ledger,
    )?;
    hf.fock_mismatch = mismatch;
    if let Some(g) = og {
        // The per-spin commutators do not vanish at a ROHF state; report the
        // gated quantity instead.
        hf.commutator = g;
    }
    if !ro
        && guard
        && cfg.spin_mutation != Some(KUStressMutation::StationarityGateOff)
        && (hf.commutator.is_nan() || hf.commutator > UKS_COMMUTATOR_TOL)
    {
        return Err(FerricError::General(format!(
            "{who}: the spin Fock/density commutator max|F D S - S D F| = {:.3e} exceeds \
             {UKS_COMMUTATOR_TOL:.0e}: not a stationary state (the stress is first order in \
             it); converge the SCF tighter",
            hf.commutator
        )));
    }
    finish(
        hf,
        XcStressOut {
            e_xc: xo.e_xc,
            electrons: xo.electrons,
            v: Vec::new(),
            ao: xo.ao,
            weight: xo.weight,
            npts: xo.npts,
            chunk_points: xo.chunk_points,
            live_chunks: xo.live_chunks,
        },
    )
}
