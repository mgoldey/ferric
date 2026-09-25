//! Analytic nuclear forces of the k-point RHF and UHF (Gamma-centred or
//! shifted meshes) — the Rust port of `reference/pbc/pbc_kgrad.py` and
//! `reference/pbc/pbc_kgrad_gdf.py` (FINDINGS "Iteration 21 (Python,
//! k-point RHF/UHF forces)").
//!
//! # Energy (per cell; [`crate::kscf`] / [`crate::kuscf`] conventions)
//!
//! ```text
//! E = (1/N_k) Σ_k tr[h(k) D(k)] + (1/2Ω) Σ_{G≠0} v |ρ(G)|²
//!   − (1/(2Ω N_k²)) Σ_s Σ_{k,k'} Σ_{K∈G+q, K≠0} v(K) tr[P^{k'}(K) D_s(k') P^{k'}(K)^H D_s(k)]
//!   − (v_M/2)(1/N_k) Σ_s Σ_k tr[D_s S D_s S] + E_nn,          q = k' − k
//! h(k) = T(k) + V_SR(k) + V_LR(k) + c0 Z_tot S(k)              (hcore::kpoint, c0 = π/(ω²Ω))
//! ```
//!
//! `v_M` is the MESH (diag(N) supercell) Madelung constant for
//! `exxdiv = ewald`, 0 for `none`. RHF: `D` total, `D_s = D/2`.
//!
//! # Derivative (derivation: the `pbc_kgrad.py` docstring)
//!
//! Every k enters through a Bloch phase on an IMAGE-RESOLVED molecular
//! quantity (`X(k) = Σ_L e^{ik·L} X(L)`, `P^{k'} = Σ_r e^{ik'·t_r} p_r`); the
//! phases do not depend on the atoms, so `d/dR_A` acts on `X(L)` and `p_r`
//! only, and every term is the Gamma term ([`crate::grad`]) with the image /
//! residue index kept and a phase-folded weight:
//!
//! ```text
//! 1e / overlap:  Σ_L Σ_mn dX_mn(L) Re Mt_nm(L),   Mt(L) = (1/N_k) Σ_k e^{ik·L} M(k)
//!                M = D for T;  M(k) = −W(k) + c0 Z_tot D(k) − v_M Σ_s D_s S D_s,
//!                W(k) = Σ_s D_s(k) F_s(k) D_s(k)        (RHF: ½ D F D, −(v_M/2) D S D)
//! V_SR:          the Gamma erfc 3-centre derivative walk, each image L weighted
//!                by Re Δ_nm(L), Δ(L) = (1/N_k) Σ_k e^{ik·L} D(k); nucleus by
//!                translation invariance (−(bra + ket)); nuclei carry no phase.
//! V_LR, J (q=0): dρ(G) = Σ_r Σ_mn dp_r,mn Δ_r,nm;
//!                dE = (1/Ω) Σ_G Re[(v conj ρ − v_ω conj S) dρ] − (1/Ω) Σ v_ω Re[ρ conj dS/dR]
//! K (every q):   dE = −(1/(Ω N_k²)) Σ_K v Re Σ_r tr[dp_r Yt_r],
//!                Yt_r = Σ_{k'} e^{ik'·t_r} Y^{k'},  Y^{k'} = Σ_s D_s(k') P^{k'}(K)^H D_s(k'−q)
//! pair FT:       bra Qb = ∂p_r/∂A_m (raised-bra kernel, binned by residue);
//!                ket −iK p_r − Qb (translation identity at ANY K — the Gamma
//!                `Q_mn = Q_nm` shortcut is wrong at K = G + q)
//! ```
//!
//! `c0 Z_tot D` in `M`: `c0 = π/(ω²Ω)` and `Z_tot` do not depend on the
//! atoms, so `h`'s G = 0 bookkeeping differentiates only through `S(k)`,
//! exactly as at Gamma (the prototype's pure-AFT `h` had none; this term and
//! the SR erfc part are anchored by 1×1×1 ≡ [`crate::grad`] and the supercell
//! anchor, both in `tests/pbc_kgrad.rs`).
//!
//! With RS-GDF J/K ([`KRsGdf`]) the two-electron lines are replaced by the
//! fitted derivative of `crate::rsgdf::kpoint::kderiv` (SR bins with the
//! energy's phases, complex Loewner metric weight per q, LR per q, `J3`'s
//! G = 0 term `M_g0(k)` through the same phase-weighted `dS` pass).
//!
//! # Supercell mapping (FINDINGS, measured)
//!
//! `E_cell = E_sc/N` and moving `A` moves all N copies, so the k-mesh force
//! IS the Gamma force on ONE copy of `A` in the diag(N) supercell (every copy
//! is equal by translation symmetry), NOT the sum over copies (that is N ×).
//!
//! # Anchor blind spots
//!
//! On meshes with every `n_i <= 2` (TRIM-only) `D(k)` is real and `q ≡ −q`:
//! a conjugation slip on `D(k)` ([`KGradMutation::ConjD`]) and a `k' − k`
//! sign slip in the exchange derivative ([`KGradMutation::WrongQ`]) are
//! algebraic IDENTITIES there. A mesh with some `n_i >= 3` is required to
//! test the phases. `ΣF` is blind to every mutant here except (sometimes)
//! `NoPhase`.
//!
//! # Scope
//!
//! HF only (no KS-DFT / ROHF at k), no ECPs, RS-GDF aux on the cell's atoms
//! (no ghost sites). The full per-q Loewner metric weight is used, so an
//! active lindep cut at any q is differentiated consistently (reported as
//! [`KGradient::fit_dropped_max`]).

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::ewald::{
    default_ewald_omega, ewald_nuclear_gradient_parts, madelung_constant, DEFAULT_EWALD_PRECISION,
};
use crate::grad::{ao_atoms, GRAD_NUCLEUS_EXPONENT};
use crate::hcore::kpoint::PeriodicHcoreK;
use crate::hcore::{
    gvector_list_bytes, hcore_pair_images, lr_gcut, sr_attraction_deriv_visit, PeriodicHcoreConfig,
    G_CHUNK_BYTES, ONE_E_ENGINE_PRECISION,
};
use crate::kdense_aft::KDenseAftEri;
use crate::kpts::{lattice_coords, KPointMesh};
use crate::kscf::KScfResult;
use crate::kuscf::KUScfResult;
use crate::lattice::Cell;
use crate::pair_ft::residues::{pair_ft_deriv_residues_chunked, residue_coords, residue_index};
use crate::rsgdf::deriv::{check_aux_map, fold_aux};
use crate::rsgdf::kpoint::kderiv::{kpoint_fit_gradient, KFitMutation};
use crate::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ndarray::{Array2, Array3};
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// Deliberate defects for the mutation tests (`tests/pbc_kgrad.rs`; the
/// prototype's `_MUTANT`). Never set in production.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KGradMutation {
    /// Bloch phase dropped on every derivative weight (Mt, Δ, Yt; the
    /// energy-side `ρ` and `P^{k'}` keep theirs).
    NoPhase,
    /// `D_s(k)`, `F_s(k)` conjugated inside the gradient (= `D(−k)` at k).
    /// IDENTITY on TRIM-only meshes.
    ConjD,
    /// Exchange derivative pairs `k'` with `k = k' + q` instead of `k' − q`
    /// (dense-AFT exchange only; a no-op on the RS-GDF path). IDENTITY on
    /// TRIM-only meshes.
    WrongQ,
    /// Primitive-cell Gamma `v_M` in the M-term instead of the mesh `v_M`
    /// (blind for `exxdiv = none` and at 1×1×1).
    GammaMadelung,
    /// Missing `1/N_k` in the overlap / kinetic `Mt(L)` (blind at 1×1×1).
    NoInvNk,
    /// RS-GDF: `e^{+iq·T}` on the SR aux images of `dJ3`.
    FitAuxPhase,
    /// RS-GDF: drop the metric derivative `Σ Wm dJ2`.
    FitNoMetric,
    /// RS-GDF: drop `J3`'s G = 0 term `M_g0`.
    FitNoG0,
}

/// Settings for [`kpoint_rhf_gradient`] / [`kpoint_uhf_gradient`].
#[derive(Debug, Clone, Copy, Default)]
pub struct KGradConfig {
    /// Memory budget (bytes); `None` = ferric's unified budget.
    pub budget_bytes: Option<usize>,
    /// TEST ONLY: a deliberate defect ([`KGradMutation`]).
    #[doc(hidden)]
    pub mutation: Option<KGradMutation>,
    /// Gaussian-nucleus exponent for the SR attraction DERIVATIVE only
    /// (`None` = [`GRAD_NUCLEUS_EXPONENT`], never tighter than the energy's;
    /// see [`crate::grad::GammaGradConfig::nucleus_exponent`]).
    pub nucleus_exponent: Option<f64>,
}

/// The k-point RS-GDF J/K a force differentiates.
#[derive(Clone, Copy)]
pub struct KRsGdfGradSource<'a> {
    /// The build the SCF ran on.
    pub gdf: &'a KRsGdf,
    /// The config it was built with (the tensors are rebuilt per q; the
    /// rebuilt fit is checked against `gdf`).
    pub cfg: &'a KRsGdfConfig,
    /// Its aux basis (on the cell's atoms).
    pub aux: &'a PreparedBasis,
}

/// Where the k-point J/K come from.
#[derive(Clone, Copy)]
pub enum KGradJk<'a> {
    /// Dense pure-AFT oracle.
    Dense(&'a KDenseAftEri),
    /// k-point RS-GDF.
    RsGdf(KRsGdfGradSource<'a>),
}

/// Per-term breakdown (each `natoms × 3`, Hartree/Bohr per cell).
#[derive(Debug, Clone)]
pub struct KGradParts {
    /// `Σ Mt dS` (Pulay, `h`'s G = 0 bookkeeping, Madelung).
    pub overlap: Array2<f64>,
    /// `Σ Mt_D dT`.
    pub kinetic: Array2<f64>,
    /// SR attraction, basis-centre motion.
    pub vsr_basis: Array2<f64>,
    /// SR attraction, nucleus motion.
    pub vsr_nuc: Array2<f64>,
    /// LR attraction, basis-centre motion (pair-FT derivative).
    pub vlr_basis: Array2<f64>,
    /// LR attraction, nucleus motion (structure factor).
    pub vlr_nuc: Array2<f64>,
    /// Dense AFT Coulomb (zero on the RS-GDF path).
    pub coulomb: Array2<f64>,
    /// Dense AFT exchange (zero on the RS-GDF path).
    pub exchange: Array2<f64>,
    /// Ewald `E_nn`, real space.
    pub nn_sr: Array2<f64>,
    /// Ewald `E_nn`, reciprocal space.
    pub nn_lr: Array2<f64>,
    /// RS-GDF `dJ3`, orbital centres, SR / LR.
    pub fit_orb_sr: Array2<f64>,
    pub fit_orb_lr: Array2<f64>,
    /// RS-GDF `dJ3`, aux centres, SR / LR.
    pub fit_aux_sr: Array2<f64>,
    pub fit_aux_lr: Array2<f64>,
    /// RS-GDF metric `Σ Wm dJ2`, SR / LR.
    pub fit_metric_sr: Array2<f64>,
    pub fit_metric_lr: Array2<f64>,
    /// RS-GDF `J3` G = 0 term through `dS`.
    pub fit_g0: Array2<f64>,
}

/// Output of the k-point force entry points.
#[derive(Debug, Clone)]
pub struct KGradient {
    /// `dE/dR_A` per cell atom, `natoms × 3` (Hartree/Bohr, per cell) — the
    /// force on ONE copy of each atom in the diag(N) supercell.
    pub grad: Array2<f64>,
    /// Per-term breakdown (sums to `grad`).
    pub parts: KGradParts,
    /// `max_{σ,k} max |F_σ D_σ S − S D_σ F_σ|` of the Focks rebuilt from the
    /// SCF densities (a gradient is only meaningful at a stationary state).
    pub commutator: f64,
    /// `v_M` used in K and M (0 for `exxdiv = none`).
    pub madelung: f64,
    /// `max_x |Σ_A dE/dR_{A,x}|` (sanity only; blind to most mutants).
    pub net_force: f64,
    /// Pair images summed for `dS`/`dT`.
    pub n_images: usize,
    /// SR attraction derivative triplets.
    pub n_sr_triplets: usize,
    /// Full-sphere G vectors in the `V_LR` derivative.
    pub n_g_lr: usize,
    /// K vectors (all q classes) in the two-electron derivative.
    pub n_k_eri: usize,
    /// Residue pair-FT derivative chunks (all passes).
    pub n_chunks: usize,
    /// Resolved memory budget (bytes).
    pub budget_bytes: usize,
    /// RS-GDF: largest number of metric eigenvalues dropped at any q
    /// (`None` on the dense path).
    pub fit_dropped_max: Option<usize>,
    /// RS-GDF: rebuild consistency and derivative counters (`None` on the
    /// dense path).
    pub fit_checks: Option<KFitChecks>,
}

/// RS-GDF k-force diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct KFitChecks {
    /// Largest relative `‖B‖_F` mismatch per `(k, k')` between the force's
    /// rebuilt 3-index tensor and the energy build's (kept counts must match).
    pub b_norm_mismatch: f64,
    /// SR 3-centre derivative triplets.
    pub n_sr3: usize,
    /// SR metric derivative pairs.
    pub n_sr2: usize,
}

/// Per-spin `D_s(k)`, `F_s(k)`.
enum KSpins {
    /// `D` total (occupation 2), `F`.
    Restricted {
        d: Vec<Array2<C64>>,
        f: Vec<Array2<C64>>,
    },
    Unrestricted {
        da: Vec<Array2<C64>>,
        db: Vec<Array2<C64>>,
        fa: Vec<Array2<C64>>,
        fb: Vec<Array2<C64>>,
    },
}

fn herm(m: &Array2<C64>) -> Array2<C64> {
    m.t().mapv(|z| z.conj())
}

fn cr(x: f64) -> C64 {
    C64::new(x, 0.0)
}

impl KSpins {
    fn nk(&self) -> usize {
        match self {
            Self::Restricted { d, .. } => d.len(),
            Self::Unrestricted { da, .. } => da.len(),
        }
    }

    fn total(&self, k: usize) -> Array2<C64> {
        match self {
            Self::Restricted { d, .. } => d[k].clone(),
            Self::Unrestricted { da, db, .. } => &da[k] + &db[k],
        }
    }

    /// `Σ_s D_s X D_s` at k (restricted `½ D X D`).
    fn sandwich(&self, k: usize, x: &Array2<C64>) -> Array2<C64> {
        match self {
            Self::Restricted { d, .. } => d[k].dot(x).dot(&d[k]).mapv(|z| z * 0.5),
            Self::Unrestricted { da, db, .. } => {
                &da[k].dot(x).dot(&da[k]) + &db[k].dot(x).dot(&db[k])
            }
        }
    }

    /// `W(k) = Σ_s D_s F_s D_s` (restricted `½ D F D`).
    fn energy_weighted(&self, k: usize) -> Array2<C64> {
        match self {
            Self::Restricted { d, f } => d[k].dot(&f[k]).dot(&d[k]).mapv(|z| z * 0.5),
            Self::Unrestricted { da, db, fa, fb } => {
                &da[k].dot(&fa[k]).dot(&da[k]) + &db[k].dot(&fb[k]).dot(&db[k])
            }
        }
    }

    /// `Σ_s D_s X D_s = Σ (c, D) c·D X D` terms.
    fn exch_terms(&self) -> Vec<(f64, &[Array2<C64>])> {
        match self {
            Self::Restricted { d, .. } => vec![(0.5, &d[..])],
            Self::Unrestricted { da, db, .. } => vec![(1.0, &da[..]), (1.0, &db[..])],
        }
    }

    /// `max_{s,k} |F D S − S D F|`.
    fn commutator(&self, s: &[Array2<C64>]) -> f64 {
        let one = |f: &Array2<C64>, d: &Array2<C64>, sk: &Array2<C64>| {
            let fds = f.dot(d).dot(sk);
            let sdf = sk.dot(d).dot(f);
            fds.iter()
                .zip(sdf.iter())
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).norm()))
        };
        let mut w = 0.0_f64;
        for k in 0..self.nk() {
            match self {
                Self::Restricted { d, f } => w = w.max(one(&f[k], &d[k], &s[k])),
                Self::Unrestricted { da, db, fa, fb } => {
                    w = w
                        .max(one(&fa[k], &da[k], &s[k]))
                        .max(one(&fb[k], &db[k], &s[k]));
                }
            }
        }
        w
    }

    /// MUTANT ConjD: every `D_s(k)`, `F_s(k)` conjugated.
    fn conjugate(&mut self) {
        let cj = |v: &mut Vec<Array2<C64>>| {
            for m in v.iter_mut() {
                m.mapv_inplace(|z| z.conj());
            }
        };
        match self {
            Self::Restricted { d, f } => {
                cj(d);
                cj(f);
            }
            Self::Unrestricted { da, db, fa, fb } => {
                cj(da);
                cj(db);
                cj(fa);
                cj(fb);
            }
        }
    }
}

/// k-point RHF nuclear gradient (module doc) of a converged
/// [`crate::kscf::solve_krhf_injected`] / [`crate::kscf::solve_krhf`]
/// result.
///
/// * `hcore_cfg`, `hk` — the config and output of
///   [`crate::hcore::kpoint::periodic_hcore_kpts`] the SCF used.
/// * `jk` — the J/K source the SCF used (its own Madelung setting is
///   ignored in favour of `exxdiv`).
/// * `scf` — the converged result; `F(k) = h + J − ½K` is rebuilt from
///   `D(k)`.
/// * `exxdiv` — the exchange-divergence treatment of the energy (the force
///   does not depend on it: the mesh-`v_M` M-term cancels the shift).
#[allow(clippy::too_many_arguments)]
pub fn kpoint_rhf_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KScfResult,
    exxdiv: ExxDiv,
    cfg: &KGradConfig,
) -> Result<KGradient, FerricError> {
    let who = "kpoint_rhf_gradient";
    check_inputs(
        who,
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        &[&scf.densities[..]],
        scf.converged,
        &scf.kpts,
    )?;
    let mut ledger = open_ledger(cell, prep, mesh, cfg, false)?;
    let vm = madelung_for(cell, mesh, exxdiv)?;
    let d = scf.densities.clone();
    let (j, k) = jk_build(&jk, vm, &d)?;
    let f: Vec<Array2<C64>> = (0..mesh.nk())
        .map(|q| &(&hk.h[q] + &j[q]) - &k[q].mapv(|z| z * 0.5))
        .collect();
    assemble(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        KSpins::Restricted { d, f },
        vm,
        cfg,
        &mut ledger,
    )
}

/// k-point UHF nuclear gradient (module doc) of a converged
/// [`crate::kuscf::solve_kuhf`] stage (`KUhfResult::scf`) or
/// [`crate::kuscf::solve_kuhf_injected`] result. `F_σ(k) = h + J[D_α + D_β]
/// − K[D_σ]` (K with the mesh `v_M` for ewald) is rebuilt from `D_σ(k)`.
/// Other arguments as [`kpoint_rhf_gradient`].
#[allow(clippy::too_many_arguments)]
pub fn kpoint_uhf_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: KGradJk<'_>,
    scf: &KUScfResult,
    exxdiv: ExxDiv,
    cfg: &KGradConfig,
) -> Result<KGradient, FerricError> {
    let who = "kpoint_uhf_gradient";
    check_inputs(
        who,
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        &[&scf.density_alpha[..], &scf.density_beta[..]],
        scf.converged,
        &scf.kpts,
    )?;
    let mut ledger = open_ledger(cell, prep, mesh, cfg, true)?;
    let vm = madelung_for(cell, mesh, exxdiv)?;
    let da = scf.density_alpha.clone();
    let db = scf.density_beta.clone();
    let (ja, ka) = jk_build(&jk, vm, &da)?;
    let (jb, kb) = jk_build(&jk, vm, &db)?;
    let nk = mesh.nk();
    let base: Vec<Array2<C64>> = (0..nk).map(|q| &(&hk.h[q] + &ja[q]) + &jb[q]).collect();
    let fa: Vec<Array2<C64>> = (0..nk).map(|q| &base[q] - &ka[q]).collect();
    let fb: Vec<Array2<C64>> = (0..nk).map(|q| &base[q] - &kb[q]).collect();
    assemble(
        cell,
        prep,
        mesh,
        hcore_cfg,
        hk,
        &jk,
        KSpins::Unrestricted { da, db, fa, fb },
        vm,
        cfg,
        &mut ledger,
    )
}

#[allow(clippy::too_many_arguments)]
fn check_inputs(
    who: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: &KGradJk<'_>,
    dens: &[&[Array2<C64>]],
    converged: bool,
    kpts: &[[f64; 3]],
) -> Result<(), FerricError> {
    let n = prep.nbasis();
    let nk = mesh.nk();
    if !converged {
        return Err(FerricError::General(format!(
            "{who}: the SCF did not converge; a gradient needs a stationary density"
        )));
    }
    if hk.omega != hcore_cfg.omega {
        return Err(FerricError::General(format!(
            "{who}: hcore was built at omega = {} but hcore_cfg.omega = {}",
            hk.omega, hcore_cfg.omega
        )));
    }
    if hk.v_ecp.is_some() {
        return Err(FerricError::General(format!(
            "{who}: periodic ECP gradients are not implemented"
        )));
    }
    if hk.s.len() != nk
        || hk.h.len() != nk
        || hk.s.iter().chain(&hk.h).any(|x| x.dim() != (n, n))
        || dens
            .iter()
            .any(|d| d.len() != nk || d.iter().any(|x| x.dim() != (n, n)))
    {
        return Err(FerricError::General(format!(
            "{who}: shape mismatch (nbasis {n}, N_k {nk}: S(k) {}, D(k) {:?})",
            hk.s.len(),
            dens.iter().map(|d| d.len()).collect::<Vec<_>>()
        )));
    }
    let dk = kpts
        .iter()
        .zip(mesh.kpts())
        .map(|(a, b)| (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0_f64, f64::max))
        .fold(0.0_f64, f64::max);
    if kpts.len() != nk || dk > 1e-12 {
        return Err(FerricError::General(format!(
            "{who}: the SCF's k-points are not this mesh's (max |Δk| {dk:.2e})"
        )));
    }
    match jk {
        KGradJk::Dense(e) => {
            if e.nk() != nk {
                return Err(FerricError::General(format!(
                    "{who}: dense kernels have {} k-points, the mesh {nk}",
                    e.nk()
                )));
            }
        }
        KGradJk::RsGdf(s) => {
            if s.gdf.nk() != nk {
                return Err(FerricError::General(format!(
                    "{who}: KRsGdf has {} k-points, the mesh {nk}",
                    s.gdf.nk()
                )));
            }
            check_aux_map(cell, s.aux, None)?;
        }
    }
    Ok(())
}

fn open_ledger(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    cfg: &KGradConfig,
    unrestricted: bool,
) -> Result<Ledger, FerricError> {
    let n = prep.nbasis();
    let nk = mesh.nk();
    let m = mesh.residue_moduli();
    let nr = m[0] * m[1] * m[2];
    let natoms = cell.positions().len();
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    // Per k: D_s, F_s, J_s, K_s, D, W, M, sandwiches (restricted ~10,
    // unrestricted ~16 complex n²).
    let per_k = if unrestricted { 16 } else { 10 };
    ledger.reserve(
        &format!("k gradient per-k complex n×n matrices (n = {n}, N_k = {nk})"),
        bytes_of((nk * n * n) as u64, 16 * per_k),
    )?;
    // Per residue: Δ, Δ_true (complex), and four real weight sets.
    ledger.reserve(
        &format!("k gradient residue-folded weights (R = {nr}, n = {n})"),
        bytes_of((nr * n * n) as u64, 16 * 2 + 8 * 5),
    )?;
    ledger.reserve(
        &format!("k gradient per-term arrays (natoms = {natoms}, nao = {n})"),
        bytes_of(((natoms + n) * 3) as u64, 8 * 24),
    )?;
    Ok(ledger)
}

fn madelung_for(cell: &Cell, mesh: &KPointMesh, exxdiv: ExxDiv) -> Result<f64, FerricError> {
    Ok(match exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => mesh.madelung(cell)?,
    })
}

type KJk = (Vec<Array2<C64>>, Vec<Array2<C64>>);

/// `J(k)`, `K(k)` (K with `v_M S D S`) of `dm` from the source.
fn jk_build(jk: &KGradJk<'_>, vm: f64, dm: &[Array2<C64>]) -> Result<KJk, FerricError> {
    let nk = dm.len();
    let n = dm[0].nrows();
    let mut j: Vec<Array2<C64>> = (0..nk).map(|_| Array2::zeros((n, n))).collect();
    let mut k: Vec<Array2<C64>> = (0..nk).map(|_| Array2::zeros((n, n))).collect();
    match jk {
        KGradJk::Dense(e) => e.contract(dm, vm, &mut j, &mut k)?,
        KGradJk::RsGdf(s) => s.gdf.contract(dm, vm, &mut j, &mut k)?,
    }
    Ok((j, k))
}

/// Per residue `r`: `w[r][m, n] = Re(scale Σ_k ph[k][r] X(k)[n, m])` — the
/// real weight of the derivative block `(m_0, n_L)`, `L ≡ r`.
fn fold_real(mats: &[Array2<C64>], ph: &[Vec<C64>], scale: f64) -> Vec<Array2<f64>> {
    let nr = ph[0].len();
    let n = mats[0].nrows();
    (0..nr)
        .map(|r| {
            let mut acc = Array2::<C64>::zeros((n, n));
            for (k, x) in mats.iter().enumerate() {
                acc.scaled_add(ph[k][r], x);
            }
            Array2::from_shape_fn((n, n), |(m, nn)| scale * acc[(nn, m)].re)
        })
        .collect()
}

/// Per residue `r`: `(1/N_k) Σ_k ph[k][r] X(k)` (complex, `[n, m]` as `X`).
fn fold_complex(mats: &[Array2<C64>], ph: &[Vec<C64>], scale: f64) -> Vec<Array2<C64>> {
    let nr = ph[0].len();
    let n = mats[0].nrows();
    (0..nr)
        .map(|r| {
            let mut acc = Array2::<C64>::zeros((n, n));
            for (k, x) in mats.iter().enumerate() {
                acc.scaled_add(ph[k][r] * scale, x);
            }
            acc
        })
        .collect()
}

/// `ρ = Σ_r Σ_mn P[r]_mn(g) w[r]_nm`.
fn residue_trace(p: &[Array3<C64>], gi: usize, w: &[Array2<C64>]) -> C64 {
    let n = w[0].nrows();
    let mut acc = C64::new(0.0, 0.0);
    for (pr, wr) in p.iter().zip(w) {
        for m in 0..n {
            for nn in 0..n {
                acc += pr[[m, nn, gi]] * wr[(nn, m)];
            }
        }
    }
    acc
}

/// `g_ao[m] += Re Σ_r Σ_n Qb[r]_mn (s w[r])_nm`,
/// `g_ao[n] += Re Σ_r Σ_m (−iK P[r] − Qb[r])_mn (s w[r])_nm` (bra and ket
/// motion of the residue-resolved pair FT at `K = kvec`).
fn contract_pair_deriv(
    g_ao: &mut [[f64; 3]],
    kvec: &[f64; 3],
    gi: usize,
    p: &[Array3<C64>],
    q: &[[Array3<C64>; 3]],
    w: &[Array2<C64>],
    scale: C64,
) {
    let n = g_ao.len();
    for ((pr, qr), wr) in p.iter().zip(q).zip(w) {
        for m in 0..n {
            for nn in 0..n {
                let wv = scale * wr[(nn, m)];
                if wv.re == 0.0 && wv.im == 0.0 {
                    continue;
                }
                let pv = pr[[m, nn, gi]];
                for x in 0..3 {
                    let qb = qr[x][[m, nn, gi]];
                    g_ao[m][x] += (qb * wv).re;
                    let qk = C64::new(0.0, -kvec[x]) * pv - qb;
                    g_ao[nn][x] += (qk * wv).re;
                }
            }
        }
    }
}

/// `g[atom(s1)] += Σ w_μν ∂_bra`, `g[atom(s2)] += Σ w_μν ∂_ket` (the Gamma
/// `add_pair_deriv` with a per-image weight).
#[allow(clippy::too_many_arguments)]
fn add_pair_deriv(
    g: &mut Array2<f64>,
    blk: &[f64],
    w: &Array2<f64>,
    dims: &[usize],
    offs: &[usize],
    sh2at: &[usize],
    s1: usize,
    s2: usize,
) {
    let (n1, n2) = (dims[s1], dims[s2]);
    let bs = n1 * n2;
    let (a1, a2) = (sh2at[s1], sh2at[s2]);
    for i in 0..n1 {
        for j in 0..n2 {
            let wv = w[(offs[s1] + i, offs[s2] + j)];
            if wv == 0.0 {
                continue;
            }
            let idx = i * n2 + j;
            for c in 0..3 {
                g[(a1, c)] += wv * blk[c * bs + idx];
                g[(a2, c)] += wv * blk[(3 + c) * bs + idx];
            }
        }
    }
}

fn fold_ao(g_ao: &[[f64; 3]], aoat: &[usize], natoms: usize) -> Array2<f64> {
    let mut g = Array2::<f64>::zeros((natoms, 3));
    for (mu, v) in g_ao.iter().enumerate() {
        for x in 0..3 {
            g[(aoat[mu], x)] += v[x];
        }
    }
    g
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    hcore_cfg: &PeriodicHcoreConfig,
    hk: &PeriodicHcoreK,
    jk: &KGradJk<'_>,
    spins: KSpins,
    vm: f64,
    cfg: &KGradConfig,
    ledger: &mut Ledger,
) -> Result<KGradient, FerricError> {
    let n = prep.nbasis();
    let nk = mesh.nk();
    let natoms = cell.positions().len();
    let mutation = cfg.mutation;
    let s_k = &hk.s;
    let commutator = spins.commutator(s_k);
    let mut spins = spins;
    if mutation == Some(KGradMutation::ConjD) {
        spins.conjugate();
    }

    // --- Phases per (k, residue): energy-side (true) and derivative-side.
    let moduli = mesh.residue_moduli();
    let nr = moduli[0] * moduli[1] * moduli[2];
    let ph_true: Vec<Vec<C64>> = (0..nk)
        .map(|k| {
            (0..nr)
                .map(|r| mesh.phase(k, residue_coords(r, moduli)))
                .collect()
        })
        .collect();
    let ph_d: Vec<Vec<C64>> = if mutation == Some(KGradMutation::NoPhase) {
        vec![vec![cr(1.0); nr]; nk]
    } else {
        ph_true.clone()
    };
    let inv_nk = 1.0 / nk as f64;

    // --- D(k), M(k).
    let dtot: Vec<Array2<C64>> = (0..nk).map(|k| spins.total(k)).collect();
    let zs = cell.nuclear_charges();
    let ztot: f64 = zs.iter().sum();
    let omega = hcore_cfg.omega;
    let vol = cell.volume();
    let c0 = PI / (omega * omega * vol);
    let vm_m = if mutation == Some(KGradMutation::GammaMadelung) && vm != 0.0 {
        madelung_constant(cell)?
    } else {
        vm
    };
    let mk: Vec<Array2<C64>> = (0..nk)
        .map(|k| {
            let mut m = spins.energy_weighted(k).mapv(|z| -z);
            m.scaled_add(cr(c0 * ztot), &dtot[k]);
            if vm_m != 0.0 {
                m.scaled_add(cr(-vm_m), &spins.sandwich(k, &s_k[k]));
            }
            m
        })
        .collect();
    let nrm1 = if mutation == Some(KGradMutation::NoInvNk) {
        1.0
    } else {
        inv_nk
    };
    let w_s = fold_real(&mk, &ph_d, nrm1);
    let w_t = fold_real(&dtot, &ph_d, nrm1);
    let w_d = fold_real(&dtot, &ph_d, inv_nk);
    let delta = fold_complex(&dtot, &ph_d, inv_nk);
    let delta_true = fold_complex(&dtot, &ph_true, inv_nk);
    drop(mk);

    // --- RS-GDF fitted two-electron pieces first (M_g0 enters the dS pass).
    let fit = match jk {
        KGradJk::Dense(_) => None,
        KGradJk::RsGdf(src) => {
            let exch = spins.exch_terms();
            let fm = KFitMutation {
                aux_phase: mutation == Some(KGradMutation::FitAuxPhase),
                no_metric: mutation == Some(KGradMutation::FitNoMetric),
                no_g0: mutation == Some(KGradMutation::FitNoG0),
            };
            Some(kpoint_fit_gradient(
                src.gdf, src.cfg, cell, prep, src.aux, mesh, s_k, &dtot, &exch, fm, ledger,
            )?)
        }
    };
    let w_g0 = fit.as_ref().map(|f| fold_real(&f.mg0, &ph_d, 1.0));

    // --- dS, dT (and the fit's G = 0 dS) over the energy's pair images.
    let images = hcore_pair_images(cell, prep, hcore_cfg.precision, ledger)?;
    let b = cell.reciprocal();
    let sh2at = prep.shell_to_atom().to_vec();
    let dims = prep.shell_dims().to_vec();
    let offs = prep.shell_offsets().to_vec();
    let nsh = prep.nshells();
    let mut g_s = Array2::<f64>::zeros((natoms, 3));
    let mut g_t = Array2::<f64>::zeros((natoms, 3));
    let mut g_fit_g0 = Array2::<f64>::zeros((natoms, 3));
    {
        let mut eng_s = Engine::new_1e_deriv(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION)?;
        let mut eng_t = Engine::new_1e_deriv(ffi::OP_KINETIC, prep, ONE_E_ENGINE_PRECISION)?;
        for l in &images {
            let r = residue_index(lattice_coords(&b, l), moduli);
            for s1 in 0..nsh {
                for s2 in 0..nsh {
                    if let Some(blk) = eng_s.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        add_pair_deriv(&mut g_s, blk, &w_s[r], &dims, &offs, &sh2at, s1, s2);
                        if let Some(wg) = &w_g0 {
                            add_pair_deriv(
                                &mut g_fit_g0,
                                blk,
                                &wg[r],
                                &dims,
                                &offs,
                                &sh2at,
                                s1,
                                s2,
                            );
                        }
                    }
                    if let Some(blk) = eng_t.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        add_pair_deriv(&mut g_t, blk, &w_t[r], &dims, &offs, &sh2at, s1, s2);
                    }
                }
            }
        }
    }

    // --- V_SR: the Gamma walk, each image weighted by Re Δ(L).
    let sr_cfg = PeriodicHcoreConfig {
        nucleus_exponent: cfg
            .nucleus_exponent
            .unwrap_or(GRAD_NUCLEUS_EXPONENT)
            .min(hcore_cfg.nucleus_exponent),
        ..*hcore_cfg
    };
    let mut g_vsr_basis = Array2::<f64>::zeros((natoms, 3));
    let mut g_vsr_nuc = Array2::<f64>::zeros((natoms, 3));
    let n_sr_triplets = sr_attraction_deriv_visit(cell, prep, &sr_cfg, ledger, |t| {
        let wd = &w_d[residue_index(lattice_coords(&b, &t.l), moduli)];
        for i in 0..t.dim1 {
            for j in 0..t.dim2 {
                let coeff = t.f * wd[(t.off1 + i, t.off2 + j)];
                if coeff == 0.0 {
                    continue;
                }
                let idx = i * t.dim2 + j;
                let idx_swapped = j * t.dim1 + i;
                for c in 0..3 {
                    let d1 = coeff * t.ket_a[c * t.bs + idx_swapped];
                    let d2 = coeff * t.ket_b[c * t.bs + idx];
                    g_vsr_basis[(t.at1, c)] += d1;
                    g_vsr_basis[(t.at2, c)] += d2;
                    g_vsr_nuc[(t.atc, c)] -= d1 + d2;
                }
            }
        }
    })?;

    // --- V_LR: full G sphere of periodic_hcore_kpts, residue pair-FT
    // derivative at q = 0.
    let aoat = ao_atoms(prep);
    let pos = cell.positions();
    let gcut_lr = lr_gcut(prep, omega, hcore_cfg.precision);
    ledger.reserve(
        &format!("k gradient V_LR G list (|G| <= {gcut_lr:.3})"),
        gvector_list_bytes(cell, gcut_lr)?,
    )?;
    let gv_lr: Vec<[f64; 3]> = cell
        .gvectors(gcut_lr)?
        .into_iter()
        .filter(|g| g[0] * g[0] + g[1] * g[1] + g[2] * g[2] > 0.0)
        .collect();
    let mut vlr_ao = vec![[0.0_f64; 3]; n];
    let mut g_vlr_nuc = Array2::<f64>::zeros((natoms, 3));
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let mut n_chunks = pair_ft_deriv_residues_chunked(
        cell,
        prep,
        &gv_lr,
        moduli,
        0.1 * hcore_cfg.precision,
        chunk_budget,
        0,
        |_g0, gs, p, q| {
            for (gi, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let v = 4.0 * PI / g2 * (-g2 / (4.0 * omega * omega)).exp();
                let mut sg = C64::new(0.0, 0.0);
                for (z, r) in zs.iter().zip(&pos) {
                    let a = gvec[0] * r[0] + gvec[1] * r[1] + gvec[2] * r[2];
                    sg += C64::new(z * a.cos(), -z * a.sin());
                }
                let rho = residue_trace(p, gi, &delta_true);
                // basis: −(1/Ω) v Re[conj(S) dρ]
                contract_pair_deriv(&mut vlr_ao, gvec, gi, p, q, &delta, sg.conj() * (-v / vol));
                // nucleus: −(1/Ω) v Re[ρ conj(dS/dR_A)], dS/dR_A = −iG Z_A e^{−iG·R_A}
                for (a, r) in pos.iter().enumerate() {
                    if zs[a] == 0.0 {
                        continue;
                    }
                    let ph = gvec[0] * r[0] + gvec[1] * r[1] + gvec[2] * r[2];
                    let t = v / vol * zs[a] * (rho.re * ph.sin() + rho.im * ph.cos());
                    for c in 0..3 {
                        g_vlr_nuc[(a, c)] += t * gvec[c];
                    }
                }
            }
            Ok(())
        },
    )?;
    let g_vlr_basis = fold_ao(&vlr_ao, &aoat, natoms);
    drop(vlr_ao);

    // --- Two-electron.
    let zeros = || Array2::<f64>::zeros((natoms, 3));
    let (mut g_j, mut g_k) = (zeros(), zeros());
    let n_k_eri: usize;
    let mut fit_parts = [zeros(), zeros(), zeros(), zeros(), zeros(), zeros()];
    let mut fit_dropped_max = None;
    let mut fit_checks = None;
    match jk {
        KGradJk::Dense(eri) => {
            let (jao, kao, ch, nkv) = dense_two_electron(
                cell,
                prep,
                mesh,
                eri,
                &spins,
                &delta,
                &delta_true,
                &ph_true,
                &ph_d,
                mutation,
                ledger,
            )?;
            g_j = fold_ao(&jao, &aoat, natoms);
            g_k = fold_ao(&kao, &aoat, natoms);
            n_chunks += ch;
            n_k_eri = nkv;
        }
        KGradJk::RsGdf(src) => {
            let f = fit.ok_or_else(|| {
                FerricError::General("k gradient: RS-GDF pieces missing (internal)".into())
            })?;
            let fold = |x: &Array2<f64>| fold_aux(x, src.aux, None, natoms);
            fit_parts = [
                f.orb_sr.clone(),
                f.orb_lr.clone(),
                fold(&f.aux_sr),
                fold(&f.aux_lr),
                fold(&f.metric_sr),
                fold(&f.metric_lr),
            ];
            n_chunks += f.n_chunks;
            n_k_eri = f.n_k_lr;
            fit_dropped_max = Some(f.n_dropped_max);
            fit_checks = Some(KFitChecks {
                b_norm_mismatch: f.b_norm_mismatch,
                n_sr3: f.n_sr3,
                n_sr2: f.n_sr2,
            });
        }
    }

    // --- Ewald E_nn.
    let (nn_sr, nn_lr) =
        ewald_nuclear_gradient_parts(cell, default_ewald_omega(cell), DEFAULT_EWALD_PRECISION)?;
    let to_arr = |v: &[[f64; 3]]| Array2::from_shape_fn((natoms, 3), |(a, c)| v[a][c]);
    let g_nn_sr = to_arr(&nn_sr);
    let g_nn_lr = to_arr(&nn_lr);

    let [f_orb_sr, f_orb_lr, f_aux_sr, f_aux_lr, f_met_sr, f_met_lr] = fit_parts;
    let grad = &g_s
        + &g_t
        + &g_vsr_basis
        + &g_vsr_nuc
        + &g_vlr_basis
        + &g_vlr_nuc
        + &g_j
        + &g_k
        + &g_nn_sr
        + &g_nn_lr
        + &f_orb_sr
        + &f_orb_lr
        + &f_aux_sr
        + &f_aux_lr
        + &f_met_sr
        + &f_met_lr
        + &g_fit_g0;
    if grad.iter().any(|v| !v.is_finite()) {
        return Err(FerricError::General("k gradient: non-finite force".into()));
    }
    let net_force = (0..3)
        .map(|c| grad.column(c).sum().abs())
        .fold(0.0_f64, f64::max);
    ferric_core::memory::warn_if_rss_over("ferric-pbc k gradient", ledger.budget(), 1.1);
    Ok(KGradient {
        grad,
        parts: KGradParts {
            overlap: g_s,
            kinetic: g_t,
            vsr_basis: g_vsr_basis,
            vsr_nuc: g_vsr_nuc,
            vlr_basis: g_vlr_basis,
            vlr_nuc: g_vlr_nuc,
            coulomb: g_j,
            exchange: g_k,
            nn_sr: g_nn_sr,
            nn_lr: g_nn_lr,
            fit_orb_sr: f_orb_sr,
            fit_orb_lr: f_orb_lr,
            fit_aux_sr: f_aux_sr,
            fit_aux_lr: f_aux_lr,
            fit_metric_sr: f_met_sr,
            fit_metric_lr: f_met_lr,
            fit_g0: g_fit_g0,
        },
        commutator,
        madelung: vm,
        net_force,
        n_images: images.len(),
        n_sr_triplets,
        n_g_lr: gv_lr.len(),
        n_k_eri,
        n_chunks,
        budget_bytes: ledger.budget(),
        fit_dropped_max,
        fit_checks,
    })
}

/// AO-resolved Coulomb and exchange forces of the dense pure-AFT kernels,
/// every q class (module doc). Returns `(J per AO, K per AO, chunks, K count)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn dense_two_electron(
    cell: &Cell,
    prep: &PreparedBasis,
    mesh: &KPointMesh,
    eri: &KDenseAftEri,
    spins: &KSpins,
    delta: &[Array2<C64>],
    delta_true: &[Array2<C64>],
    ph_true: &[Vec<C64>],
    ph_d: &[Vec<C64>],
    mutation: Option<KGradMutation>,
    ledger: &mut Ledger,
) -> Result<(Vec<[f64; 3]>, Vec<[f64; 3]>, usize, usize), FerricError> {
    let n = prep.nbasis();
    let nk = mesh.nk();
    let moduli = mesh.residue_moduli();
    let nr = moduli[0] * moduli[1] * moduli[2];
    let vol = cell.volume();
    let gcut = eri.gcut();
    let g2cut = gcut * gcut;
    let qmax = (0..nk)
        .map(|iq| {
            let q = mesh.q_class(iq).0;
            (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt()
        })
        .fold(0.0_f64, f64::max);
    ledger.reserve(
        &format!("k gradient ERI K list (|K| <= {gcut:.3}, |q| <= {qmax:.3})"),
        gvector_list_bytes(cell, gcut + qmax)?.saturating_mul(2),
    )?;
    let exch = spins.exch_terms();
    let cx_base = -1.0 / (vol * (nk * nk) as f64);
    let mut jao = vec![[0.0_f64; 3]; n];
    let mut kao = vec![[0.0_f64; 3]; n];
    let mut n_chunks = 0usize;
    let mut n_kv = 0usize;
    // Per K: Yt (R n²), P^{k'} and its adjoint, Y (4 n²), complex.
    let extra_per_g = bytes_of(((nr + 4) * n * n) as u64, 16);
    for iq in 0..nk {
        let (q, mq) = mesh.q_class(iq);
        let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
        let ks: Vec<[f64; 3]> = cell
            .gvectors(gcut + qn)?
            .into_iter()
            .map(|g| [g[0] + q[0], g[1] + q[1], g[2] + q[2]])
            .filter(|k| {
                let k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                k2 > 0.0 && k2 <= g2cut
            })
            .collect();
        n_kv += ks.len();
        let mq_used = if mutation == Some(KGradMutation::WrongQ) {
            [-mq[0], -mq[1], -mq[2]]
        } else {
            mq
        };
        let kof: Vec<usize> = (0..nk).map(|j| mesh.k_minus_q(j, mq_used)).collect();
        let is_q0 = mq == [0, 0, 0];
        let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
        n_chunks += pair_ft_deriv_residues_chunked(
            cell,
            prep,
            &ks,
            moduli,
            eri.pair_thresh(),
            chunk_budget,
            extra_per_g,
            |_k0, kv, p, qd| {
                for (gi, kvec) in kv.iter().enumerate() {
                    let k2 = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
                    let v = 4.0 * PI / k2;
                    if is_q0 {
                        let rho = residue_trace(p, gi, delta_true);
                        contract_pair_deriv(
                            &mut jao,
                            kvec,
                            gi,
                            p,
                            qd,
                            delta,
                            rho.conj() * (v / vol),
                        );
                    }
                    // Exchange: Yt_r = Σ_k' e^{ik'·t_r} Σ_s D_s(k') P^{k'H} D_s(k'−q).
                    let mut yt: Vec<Array2<C64>> = (0..nr).map(|_| Array2::zeros((n, n))).collect();
                    for j in 0..nk {
                        let mut pj = Array2::<C64>::zeros((n, n));
                        for (pr, ph) in p.iter().zip(&ph_true[j]) {
                            for m in 0..n {
                                for l in 0..n {
                                    pj[(m, l)] += *ph * pr[[m, l, gi]];
                                }
                            }
                        }
                        let pjh = herm(&pj);
                        let mut y = Array2::<C64>::zeros((n, n));
                        for (c, ds) in &exch {
                            y.scaled_add(cr(*c), &ds[j].dot(&pjh).dot(&ds[kof[j]]));
                        }
                        for (r, ytr) in yt.iter_mut().enumerate() {
                            ytr.scaled_add(ph_d[j][r], &y);
                        }
                    }
                    contract_pair_deriv(&mut kao, kvec, gi, p, qd, &yt, cr(cx_base * v));
                }
                Ok(())
            },
        )?;
    }
    Ok((jao, kao, n_chunks, n_kv))
}
