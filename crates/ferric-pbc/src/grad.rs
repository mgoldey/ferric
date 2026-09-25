//! Analytic nuclear gradients (forces) of the Gamma-point periodic RHF —
//! the Rust port of `reference/pbc/pbc_grad.py` (FINDINGS "Iteration 16").
//!
//! Energy (Gamma, G = 0 dropped everywhere; `v_M` = Madelung shift of
//! `exxdiv = ewald`, 0 for `none`):
//!
//! ```text
//! E = Σ D h + Σ Γ_μνλσ I_μνλσ − (v_M/4) tr(DSDS) + E_nn,
//! Γ = ½ D_μν D_λσ − ¼ D_μλ D_νσ,
//! h = T + V_SR(ω) + V_LR(ω) + c0 Z_tot S,   c0 = π/(ω²Ω)   (PeriodicHcore)
//! I = (1/Ω) Σ_{G≠0} (4π/G²) Re[P*_μν(G) P_λσ(G)]            (DenseAftEri, pure AFT)
//! ```
//!
//! Moving atom `A` moves its nucleus AND every basis function centred on it,
//! in every lattice image. Differentiating at fixed (converged) `D`:
//!
//! ```text
//! dE/dR_A = Σ D dT/dR_A                          lattice-summed shifted 1e derivative blocks
//!         + Σ D dV_SR/dR_A   (basis + nucleus)   erfc 3-centre derivative, Gaussian nuclei,
//!                                                nucleus = −(bra + ket) by translation invariance
//!         + Σ D dV_LR/dR_A   basis:   −(2/Ω) Σ_{G∈half} v_ω Re[ρ'_A* S(G)],  ρ'_A = 2 Σ_{μ∈A,ν} D_μν Q_μν
//!                            nucleus: −(2/Ω) Σ_{G∈half} v_ω Re[ρ* (−iG) Z_A e^{−iG·R_A}]
//!         + Σ Γ dI/dR_A    = (8/Ω) Σ_{G∈half} (4π/G²) Re Σ_{μ∈A,ν} Q*_μν Z_μν,
//!                            Z(G) = ½ D ρ(G) − ¼ D P(G) D
//!         + Σ M dS/dR_A,     M = −W + c0 Z_tot D − (v_M/2) D S D,   W = ½ D F D
//!         + dE_nn/dR_A       Ewald: SR erfc + LR structure factor
//! ```
//!
//! with `Q_μν = ∂P_μν/∂A_μ` the bra-centre pair-FT derivative
//! ([`crate::pair_ft::pair_ft_deriv_chunked`]; the ket derivative is `Q_νμ`
//! because `P_μν = P_νμ` at reciprocal-lattice G), streamed per G chunk
//! together with `P` and contracted immediately — `Q` (3 × the pair FT) is
//! never held for more than one chunk. `v_ω = 4π/G² e^{−G²/4ω²}`.
//!
//! # The overlap-coupled matrix `M`
//!
//! * `−W` is the usual Pulay term (orthonormality constraint).
//! * `c0 Z_tot D`: `h`'s G = 0 bookkeeping `V_G0 = c0 Z_tot S` enters ONLY
//!   through `S`. (The prototype also carried `−c0 N D + (c0/2) DSD` from
//!   the Ewald-split ERI's own G = 0 term; the dense-AFT ERI here is pure
//!   reciprocal space and has none.)
//! * `−(v_M/2) DSD` is the explicit `S` dependence of `E_M = −(v_M/4) tr(DSDS)`.
//!   With it, forces for `exxdiv = ewald` and `none` agree to roundoff
//!   (`E_M = −v_M N/2` is geometry-independent, and this term cancels the
//!   `−v_M` shift of the occupied levels inside `W`); without it they differ
//!   by `v_M tr(D dS/dR)`, a 6e-2 Ha/Bohr error on H₂ that the sum of forces
//!   CANNOT see (prototype, measured).
//!
//! # Consistency with the energy
//!
//! Every lattice sum uses exactly the energy's truncation: the `S`/`T`/`V_SR`
//! pair images, SR nucleus candidates and per-triplet screen come from the
//! same `hcore` helpers at `hcore_cfg.precision`; the `V_LR` G sphere is
//! `hcore::lr_gcut`; the ERI G sphere is `DenseAftEri::gcut`. The G spheres
//! do not depend on the geometry, so the gradient is the derivative of the
//! truncated energy up to the primitive screens (`≤ precision`).
//!
//! libint2's `erf_nuclear`/`erfc_nuclear` derivative operators are NOT used
//! (the libint 2.7.2 bug of FINDINGS "Iteration 2" affects them as it does
//! the energy); the SR attraction goes through the ordinary erfc 3-centre
//! derivative engine on Gaussian nuclei.
//!
//! # Out of scope (documented, not implemented)
//!
//! * **RS-GDF J/K gradient** — the scalable route. Needs the standard DF
//!   gradient `Σ Γ^P_μν d(μν|P)' − ½ Σ Γ^{PQ} d(P|Q)'` with every primed
//!   integral in the RS split (SR erfc 3c/2c derivative integrals over pair
//!   and aux images; LR pair-FT derivative `Q` and the one-centre aux FT
//!   derivative; G = 0 only through `dS`), plus a check that the lindep
//!   eigen-cut count does not change between displaced geometries
//!   (FINDINGS "Iteration 16", "RS-GDF gradient route"). Only the dense-AFT
//!   oracle is differentiated here.
//! * **Stress** — needs the lattice-vector derivative of every piece
//!   (per-image `L ⊗ Q_L` virials, `∂v(G)/∂ε`, `∂v_M/∂ε`, `∂Ω/∂ε`); FINDINGS
//!   "Iteration 16", "Stress".
//! * ECPs (`v_ecp`), open shells, k-points, KS-DFT: rejected or not provided.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::{DenseAftEri, ExxDiv};
use crate::ewald::{
    default_ewald_omega, ewald_nuclear_gradient_parts, madelung_constant, DEFAULT_EWALD_PRECISION,
};
use crate::hcore::{
    gvector_list_bytes, half_gvectors, hcore_pair_images, lr_gcut, sr_attraction_gradient,
    PeriodicHcore, PeriodicHcoreConfig, G_CHUNK_BYTES, ONE_E_ENGINE_PRECISION,
};
use crate::lattice::Cell;
use crate::pair_ft::pair_ft_deriv_chunked;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_scf::fock::{JBuilder, KBuilder};
use ferric_scf::result::{ScfResult, Spin};
use ndarray::{s, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Deliberate defects for the mutation tests (`tests/pbc_grad.rs`): each
/// must be caught by the finite-difference anchor, and (measured in the
/// prototype) NONE is caught by the sum of forces. Never set in production.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradMutation {
    /// Drop the basis-centre motion in `V_ne` (SR and LR): only the nucleus
    /// moves.
    VneNoBasis,
    /// `+W` instead of `−W` in `M`.
    WSign,
    /// Drop the reciprocal-space part of the Ewald `E_nn` gradient.
    NoEwaldLr,
    /// Drop the `−(v_M/2) DSD` Madelung term from `M`.
    NoMadelungS,
}

/// Settings for [`gamma_rhf_gradient_with`].
#[derive(Debug, Clone, Copy, Default)]
pub struct GammaGradConfig {
    /// Memory budget (bytes); `None` = ferric's unified budget
    /// ([`crate::budget::resolve`]).
    pub budget_bytes: Option<usize>,
    /// TEST ONLY: a deliberate defect ([`GradMutation`]).
    #[doc(hidden)]
    pub mutation: Option<GradMutation>,
    /// Gaussian-nucleus exponent used ONLY for the short-range attraction
    /// DERIVATIVE (`None` = [`GRAD_NUCLEUS_EXPONENT`], never tighter than the
    /// energy's `hcore_cfg.nucleus_exponent`). libint2's 3-centre derivative
    /// loses precision for very tight Gaussians: measured on the triclinic
    /// 4H s+p cell, the FD error was 8.3e-2 at 1e16, 4.3e-6 at 1e12 and 1.9e-7
    /// at 1e10, while the smeared-vs-point potential error is ~3.6e-7 at 1e10.
    pub nucleus_exponent: Option<f64>,
}

/// Default exponent for the SR attraction derivative (see
/// [`GammaGradConfig::nucleus_exponent`]).
pub const GRAD_NUCLEUS_EXPONENT: f64 = 1e10;

/// Per-term breakdown of the gradient (each `natoms × 3`, Hartree/Bohr).
#[derive(Debug, Clone)]
pub struct GammaGradParts {
    /// `Σ M dS` (Pulay, G = 0 bookkeeping, Madelung).
    pub overlap: Array2<f64>,
    /// `Σ D dT`.
    pub kinetic: Array2<f64>,
    /// `Σ D dV_SR`, basis-centre motion.
    pub vsr_basis: Array2<f64>,
    /// `Σ D dV_SR`, nucleus motion.
    pub vsr_nuc: Array2<f64>,
    /// `Σ D dV_LR`, basis-centre motion (pair-FT derivative).
    pub vlr_basis: Array2<f64>,
    /// `Σ D dV_LR`, nucleus motion (structure factor).
    pub vlr_nuc: Array2<f64>,
    /// `Σ Γ dI` (dense AFT J and K, pair-FT derivative).
    pub eri: Array2<f64>,
    /// Ewald `E_nn`, real-space part.
    pub nn_sr: Array2<f64>,
    /// Ewald `E_nn`, reciprocal-space part.
    pub nn_lr: Array2<f64>,
}

/// Output of [`gamma_rhf_gradient_with`].
#[derive(Debug, Clone)]
pub struct GammaRhfGradient {
    /// `dE/dR_A` per cell atom, `natoms × 3` (Hartree/Bohr, per cell).
    pub grad: Array2<f64>,
    /// Per-term breakdown (sums to `grad`).
    pub parts: GammaGradParts,
    /// `max |F D S − S D F|` of the Fock matrix rebuilt from the SCF density
    /// (a gradient is only meaningful at a stationary `D`).
    pub commutator: f64,
    /// `v_M` used in `K` and `M` (0 for `exxdiv = none`).
    pub madelung: f64,
    /// `max_x |Σ_A dE/dR_{A,x}|`. SANITY ONLY: translation invariance is
    /// blind to every mutation in [`GradMutation`] (prototype, measured).
    pub net_force: f64,
    /// Shifted 3-centre derivative calls in the SR attraction.
    pub n_sr_triplets: usize,
    /// Pair images summed for `dS`/`dT`.
    pub n_images: usize,
    /// Half-sphere G vectors in the `V_LR` derivative.
    pub n_g_lr: usize,
    /// Half-sphere G vectors in the ERI derivative.
    pub n_g_eri: usize,
    /// `pair_ft_deriv` chunks (`V_LR` + ERI).
    pub n_chunks: usize,
    /// Resolved memory budget (bytes).
    pub budget_bytes: usize,
}

/// Gamma-point RHF nuclear gradient `dE/dR` (`natoms × 3`, per cell) with
/// default settings; see [`gamma_rhf_gradient_with`].
pub fn gamma_rhf_gradient(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    exxdiv: ExxDiv,
) -> Result<Array2<f64>, FerricError> {
    Ok(gamma_rhf_gradient_with(
        cell,
        prep,
        hcore_cfg,
        hc,
        eri,
        scf,
        exxdiv,
        &GammaGradConfig::default(),
    )?
    .grad)
}

/// Gamma-point RHF nuclear gradient with its per-term breakdown (module doc).
///
/// * `hcore_cfg`, `hc` — the config and output of
///   [`crate::hcore::periodic_hcore`] the SCF used (`hc.omega` must equal
///   `hcore_cfg.omega`; the SR/LR truncation is rebuilt from `hcore_cfg`).
/// * `eri` — the dense pure-AFT tensor the SCF used (its G sphere is reused;
///   its own `madelung` is ignored in favour of `exxdiv`).
/// * `scf` — the converged restricted result (`density_total`); `F` is
///   rebuilt from it, `W = ½ D F D`.
/// * `exxdiv` — the exchange-divergence treatment of the energy being
///   differentiated. At Gamma RHF it does not change `D`, and (with the
///   Madelung `S` term) not the force either.
#[allow(clippy::too_many_arguments)]
pub fn gamma_rhf_gradient_with(
    cell: &Cell,
    prep: &PreparedBasis,
    hcore_cfg: &PeriodicHcoreConfig,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    scf: &ScfResult,
    exxdiv: ExxDiv,
    cfg: &GammaGradConfig,
) -> Result<GammaRhfGradient, FerricError> {
    let n = prep.nbasis();
    let natoms = cell.positions().len();
    let mutation = cfg.mutation;
    if hc.omega != hcore_cfg.omega {
        return Err(FerricError::General(format!(
            "gamma_rhf_gradient: hcore was built at omega = {} but hcore_cfg.omega = {}",
            hc.omega, hcore_cfg.omega
        )));
    }
    if hc.v_ecp.is_some() {
        return Err(FerricError::General(
            "gamma_rhf_gradient: periodic ECP gradients are not implemented".into(),
        ));
    }
    if scf.spin != Spin::Restricted {
        return Err(FerricError::General(format!(
            "gamma_rhf_gradient: needs a restricted closed-shell result, got {:?}",
            scf.spin
        )));
    }
    if !scf.converged {
        return Err(FerricError::General(
            "gamma_rhf_gradient: the SCF did not converge; a gradient needs a stationary density"
                .into(),
        ));
    }
    if hc.s.dim() != (n, n)
        || scf.density_total.dim() != (n, n)
        || eri.eri().dim() != (n * n, n * n)
    {
        return Err(FerricError::General(format!(
            "gamma_rhf_gradient: shape mismatch (nbasis {n}: S {:?}, D {:?}, ERI {:?})",
            hc.s.dim(),
            scf.density_total.dim(),
            eri.eri().dim()
        )));
    }

    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    // D, J, K, F, W, M, DSD, SD + per-G complex scratch (ρ-weighted Z,
    // D P D in re/im, P in re/im: ~8 real n²).
    ledger.reserve(
        &format!("gamma gradient n×n matrices + per-G scratch (n = {n})"),
        bytes_of((n * n) as u64, 8 * 16),
    )?;
    ledger.reserve(
        &format!("gamma gradient per-term arrays (natoms = {natoms})"),
        bytes_of((natoms * 3) as u64, 8 * 12),
    )?;

    // --- Density, rebuilt Fock, W, M.
    let d = scf.density_total.clone();
    let vm = match exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => madelung_constant(cell)?,
    };
    let mut jm = Array2::<f64>::zeros((n, n));
    let mut km = Array2::<f64>::zeros((n, n));
    eri.j_builder().build(&d, &mut jm)?;
    eri.k_builder_with_madelung(vm).build(&d, &mut km)?;
    let f = &(&hc.h + &jm) - &(0.5 * &km);
    let s_mat = &hc.s;
    let fds = f.dot(&d).dot(s_mat);
    let commutator = fds
        .iter()
        .zip(fds.t().iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    let w = 0.5 * d.dot(&f).dot(&d);
    let dsd = d.dot(s_mat).dot(&d);
    let zs = cell.nuclear_charges();
    let ztot: f64 = zs.iter().sum();
    let omega = hcore_cfg.omega;
    let vol = cell.volume();
    let c0 = PI / (omega * omega * vol);
    let mut m = if mutation == Some(GradMutation::WSign) {
        w.clone()
    } else {
        -&w
    };
    m.scaled_add(c0 * ztot, &d);
    if mutation != Some(GradMutation::NoMadelungS) {
        m.scaled_add(-0.5 * vm, &dsd);
    }

    // --- dS, dT over the energy's pair images (bra and ket blocks).
    let images = hcore_pair_images(cell, prep, hcore_cfg.precision, &mut ledger)?;
    let sh2at = prep.shell_to_atom().to_vec();
    let dims = prep.shell_dims().to_vec();
    let offs = prep.shell_offsets().to_vec();
    let nsh = prep.nshells();
    let mut g_s = Array2::<f64>::zeros((natoms, 3));
    let mut g_t = Array2::<f64>::zeros((natoms, 3));
    {
        let mut eng_s = Engine::new_1e_deriv(ffi::OP_OVERLAP, prep, ONE_E_ENGINE_PRECISION)?;
        let mut eng_t = Engine::new_1e_deriv(ffi::OP_KINETIC, prep, ONE_E_ENGINE_PRECISION)?;
        for l in &images {
            for s1 in 0..nsh {
                for s2 in 0..nsh {
                    if let Some(blk) = eng_s.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        add_pair_deriv(&mut g_s, blk, &m, &dims, &offs, &sh2at, s1, s2);
                    }
                    if let Some(blk) = eng_t.compute_1e_deriv_block_shifted(prep, s1, s2, *l)? {
                        add_pair_deriv(&mut g_t, blk, &d, &dims, &offs, &sh2at, s1, s2);
                    }
                }
            }
        }
    }

    // --- V_SR: Gaussian nuclei, erfc(ω), the energy's image/screen sets.
    let sr_cfg = PeriodicHcoreConfig {
        nucleus_exponent: cfg
            .nucleus_exponent
            .unwrap_or(GRAD_NUCLEUS_EXPONENT)
            .min(hcore_cfg.nucleus_exponent),
        ..*hcore_cfg
    };
    let (mut g_vsr_basis, g_vsr_nuc, n_sr_triplets) =
        sr_attraction_gradient(cell, prep, &sr_cfg, &d, &mut ledger)?;

    // AO -> atom.
    let mut aoat = vec![0usize; n];
    for sh in 0..nsh {
        for k in 0..dims[sh] {
            aoat[offs[sh] + k] = sh2at[sh];
        }
    }
    let pos = cell.positions();

    // --- V_LR: pair-FT derivative (basis) and structure factor (nucleus).
    let gcut_lr = lr_gcut(prep, omega, hcore_cfg.precision);
    ledger.reserve(
        &format!("gamma gradient V_LR G list (|G| <= {gcut_lr:.3})"),
        gvector_list_bytes(cell, gcut_lr)?,
    )?;
    let gv_lr = half_gvectors(cell, gcut_lr)?;
    let mut g_vlr_basis = Array2::<f64>::zeros((natoms, 3));
    let mut g_vlr_nuc = Array2::<f64>::zeros((natoms, 3));
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    let mut n_chunks = pair_ft_deriv_chunked(
        cell,
        prep,
        &gv_lr,
        0.1 * hcore_cfg.precision,
        chunk_budget,
        0,
        |_g0, gs, p, q| {
            let mut rq = vec![Complex64::new(0.0, 0.0); natoms * 3];
            for (g, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let v = 4.0 * PI / g2 * (-g2 / (4.0 * omega * omega)).exp();
                let fac = -2.0 / vol * v;
                // S(G) = Σ_C Z_C e^{−iG·R_C}
                let (mut sre, mut sim) = (0.0_f64, 0.0_f64);
                for (z, r) in zs.iter().zip(&pos) {
                    let ph = gvec[0] * r[0] + gvec[1] * r[1] + gvec[2] * r[2];
                    sre += z * ph.cos();
                    sim -= z * ph.sin();
                }
                let mut rho = Complex64::new(0.0, 0.0);
                rq.fill(Complex64::new(0.0, 0.0));
                for mu in 0..n {
                    let a = aoat[mu];
                    for nu in 0..n {
                        let dmn = d[(mu, nu)];
                        if dmn == 0.0 {
                            continue;
                        }
                        rho += p[[mu, nu, g]] * dmn;
                        for c in 0..3 {
                            rq[a * 3 + c] += q[c][[mu, nu, g]] * (2.0 * dmn);
                        }
                    }
                }
                for a in 0..natoms {
                    for c in 0..3 {
                        let z = rq[a * 3 + c];
                        // Re[ρ'* S] = ρ'.re S.re + ρ'.im S.im
                        g_vlr_basis[(a, c)] += fac * (z.re * sre + z.im * sim);
                    }
                    if zs[a] == 0.0 {
                        continue;
                    }
                    let ph = gvec[0] * pos[a][0] + gvec[1] * pos[a][1] + gvec[2] * pos[a][2];
                    // t = (−i) Z_A e^{−iφ} = Z_A (−sin φ − i cos φ)
                    let (tre, tim) = (-zs[a] * ph.sin(), -zs[a] * ph.cos());
                    let re = rho.re * tre + rho.im * tim;
                    for c in 0..3 {
                        g_vlr_nuc[(a, c)] += fac * re * gvec[c];
                    }
                }
            }
            Ok(())
        },
    )?;
    if mutation == Some(GradMutation::VneNoBasis) {
        g_vsr_basis.fill(0.0);
        g_vlr_basis.fill(0.0);
    }

    // --- Two-electron (dense pure AFT J and K) through the pair-FT derivative.
    let gcut_eri = eri.gcut();
    ledger.reserve(
        &format!("gamma gradient ERI G list (|G| <= {gcut_eri:.3})"),
        gvector_list_bytes(cell, gcut_eri)?,
    )?;
    let gv_eri = half_gvectors(cell, gcut_eri)?;
    let mut g_eri = Array2::<f64>::zeros((natoms, 3));
    let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
    n_chunks += pair_ft_deriv_chunked(
        cell,
        prep,
        &gv_eri,
        eri.pair_thresh(),
        chunk_budget,
        0,
        |_g0, gs, p, q| {
            for (g, gvec) in gs.iter().enumerate() {
                let g2 = gvec[0] * gvec[0] + gvec[1] * gvec[1] + gvec[2] * gvec[2];
                let fac = 8.0 / vol * (4.0 * PI / g2);
                let pg = p.slice(s![.., .., g]);
                let pre = pg.mapv(|z| z.re);
                let pim = pg.mapv(|z| z.im);
                let rho_re = (&d * &pre).sum();
                let rho_im = (&d * &pim).sum();
                // Z = ½ D ρ − ¼ D P D
                let mut zr = d.dot(&pre).dot(&d) * (-0.25);
                zr.scaled_add(0.5 * rho_re, &d);
                let mut zi = d.dot(&pim).dot(&d) * (-0.25);
                zi.scaled_add(0.5 * rho_im, &d);
                for mu in 0..n {
                    let a = aoat[mu];
                    let mut acc = [0.0_f64; 3];
                    for nu in 0..n {
                        let (zre, zim) = (zr[(mu, nu)], zi[(mu, nu)]);
                        for (c, qc) in q.iter().enumerate() {
                            let qz = qc[[mu, nu, g]];
                            // Re[Q* Z]
                            acc[c] += qz.re * zre + qz.im * zim;
                        }
                    }
                    for c in 0..3 {
                        g_eri[(a, c)] += fac * acc[c];
                    }
                }
            }
            Ok(())
        },
    )?;

    // --- Ewald E_nn (the energy's ω; the result is ω-independent).
    let (nn_sr, nn_lr) =
        ewald_nuclear_gradient_parts(cell, default_ewald_omega(cell), DEFAULT_EWALD_PRECISION)?;
    let to_arr = |v: &[[f64; 3]]| Array2::from_shape_fn((natoms, 3), |(a, c)| v[a][c]);
    let g_nn_sr = to_arr(&nn_sr);
    let mut g_nn_lr = to_arr(&nn_lr);
    if mutation == Some(GradMutation::NoEwaldLr) {
        g_nn_lr.fill(0.0);
    }

    let grad = &g_s
        + &g_t
        + &g_vsr_basis
        + &g_vsr_nuc
        + &g_vlr_basis
        + &g_vlr_nuc
        + &g_eri
        + &g_nn_sr
        + &g_nn_lr;
    let net_force = (0..3)
        .map(|c| grad.column(c).sum().abs())
        .fold(0.0_f64, f64::max);
    ferric_core::memory::warn_if_rss_over("ferric-pbc gamma_rhf_gradient", ledger.budget(), 1.1);
    Ok(GammaRhfGradient {
        grad,
        parts: GammaGradParts {
            overlap: g_s,
            kinetic: g_t,
            vsr_basis: g_vsr_basis,
            vsr_nuc: g_vsr_nuc,
            vlr_basis: g_vlr_basis,
            vlr_nuc: g_vlr_nuc,
            eri: g_eri,
            nn_sr: g_nn_sr,
            nn_lr: g_nn_lr,
        },
        commutator,
        madelung: vm,
        net_force,
        n_sr_triplets,
        n_images: images.len(),
        n_g_lr: gv_lr.len(),
        n_g_eri: gv_eri.len(),
        n_chunks,
        budget_bytes: ledger.budget(),
    })
}

/// `g[atom(s1)] += Σ w_μν ∂_bra`, `g[atom(s2)] += Σ w_μν ∂_ket` for one
/// shifted 1e derivative block (`[bra xyz, ket xyz]`, each `n1 × n2`).
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
