//! Unrestricted orbital-optimized RI-MP2 (U-OO-RI-MP2).
//!
//! Open-shell counterpart to `oo_rimp2`. Minimizes E_UHF + E_U-MP2 jointly
//! over independent α and β orbital rotations using a per-spin level-shifted
//! diagonal-Hessian Newton step with optional DIIS extrapolation.
//!
//! The U-MP2 orbital gradient (αα and αβ blocks for `g^α`; ββ and αβ for
//! `g^β`) at FIXED orbital energies is FD-validated to ~1e-10 in `u_rimp2`.
//! The UHF term `+2·F^σ_{ai}` and the Fock (orbital-energy) response of the
//! U-MP2 energy are added here (see `u_gradient_solver_frame`). The functional
//! is the full-Fock one, evaluated in each spin's semicanonical frame, the
//! same construction as closed-shell `oo_rimp2` (defect F2, 2026-09-24).
//!
//! Reference: Bozkaya, JCP 139, 154105 (2013).

use crate::oo_rimp2::{
    compute_b_full_mo_with, lowdin_orthonormalize, semicanonical_rotation, OoRiMp2AoTensors,
};
use crate::orbital_rotation::cayley_rotation;
use crate::rimp2::active_occ;
use crate::u_rimp2::{
    build_u_mp2_density, compute_u_mp2_amplitudes, compute_u_mp2_orbital_gradient, UMp2Amplitudes,
    URiMp2Components,
};
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::diis::Diis;
use ferric_scf::direct_j::DirectJ;
use ferric_scf::direct_k::DirectK;
use ferric_scf::fock::{JBuilder, KBuilder};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::{ScfResult, Spin};
use ndarray::Array2;

/// Configuration for U-OO-RI-MP2.
#[derive(Debug, Clone)]
pub struct UOoRiMp2Config {
    pub max_iter: usize,
    pub grad_conv: f64,
    pub energy_conv: f64,
    pub frozen_core: usize,
    /// Level shift on the approximate diagonal Hessian (Ha).
    pub level_shift: f64,
    /// DIIS subspace size for orbital rotations (per spin).
    pub diis_size: usize,
    pub use_diis: bool,
    /// Cap on |κ| per element (radians) to keep steps in the trust region.
    pub max_kappa: f64,
    /// Print one line per orbital-optimization iteration to stdout while the
    /// job runs (HF/MP2/total energy, per-spin gradient norms) — live progress
    /// for a long-running job, opt-in and additive. Default `false`
    /// (unchanged, silent-until-done output). Mirrors
    /// `OoRiMp2Config::verbose` (the closed-shell counterpart).
    pub verbose: bool,
    /// Optional resident-bytes ceiling for the AO-side 3-index tensor AND the
    /// per-iteration fail-fast guards on the t_aa/t_bb/t_ab amplitude trio and
    /// the b_full_a/b_full_b full-MO B tensors (all co-resident during the
    /// gradient step — see the "MEMORY NOTE" on [`u_oo_ri_mp2`]). `None` →
    /// resolved via [`ferric_core::memory::resolve_budget_bytes`]. Same
    /// field name/doc convention as `OoRiMp2Config::memory_budget_bytes` and
    /// `RiMp2Config::memory_budget_bytes` (ferric-rpa's `PdepRpaConfig` also
    /// follows this convention).
    pub memory_budget_bytes: Option<usize>,
}

impl Default for UOoRiMp2Config {
    fn default() -> Self {
        Self {
            max_iter: 100,
            grad_conv: 1e-4,
            energy_conv: 1e-8,
            frozen_core: 0,
            level_shift: 0.1,
            diis_size: 6,
            use_diis: true,
            max_kappa: 0.3,
            verbose: false,
            memory_budget_bytes: None,
        }
    }
}

/// Result of an unrestricted OO-RI-MP2 calculation.
#[derive(Debug, Clone)]
#[must_use]
pub struct UOoRiMp2Result {
    pub total_energy: f64,
    pub hf_energy: f64,
    pub mp2_corr: f64,
    pub components: URiMp2Components,
    pub converged: bool,
    pub iterations: usize,
    pub grad_norm: f64,
    pub mos_alpha: Array2<f64>,
    pub mos_beta: Array2<f64>,
    pub eps_alpha: Vec<f64>,
    pub eps_beta: Vec<f64>,
}

impl std::fmt::Display for UOoRiMp2Result {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "U-OO-RI-MP2 total: {:.10} Ha (corr: {:.10}, {} iters, converged: {})",
            self.total_energy, self.mp2_corr, self.iterations, self.converged
        )
    }
}

/// Compute UHF energy and α/β Fock matrices from MO coefficients.
///
/// `budget_bytes` is the caller-resolved memory ceiling for the DirectJ/DirectK
/// builds below — threaded from [`u_oo_ri_mp2`]'s single per-call
/// `resolve_budget_bytes(config.memory_budget_bytes)` rather than re-resolved
/// here (this function is called up to 3× per orbital-optimization iteration,
/// including backtracks — see the M-budget-plumbing-sweep note on
/// [`u_oo_ri_mp2`]).
// Spin-resolved UHF energy: alpha/beta coefficients and occupations are
// irreducibly distinct quantities with no natural sub-bundle to group.
// `vnn` is the FULL classical nuclear-repulsion-like constant: plain
/// `mol.nuclear_repulsion()` in vacuum, or (when an external potential is
/// present) that PLUS `ext.charge_nuclear_energy(mol) +
/// ext.field_nuclear_energy(mol)` -- the same convention
/// `ferric_scf::driver::prepare_scf_env` uses and the same fix applied to
/// closed-shell `oo_rimp2::compute_hf_energy`. Must be computed ONCE by the
/// caller and threaded into every `compute_uhf_energy` call for a given
/// `u_oo_ri_mp2` run -- never recomputed from `mol` alone once `ext` is
/// `Some` (an uncompensated one-electron attraction with no offsetting
/// classical repulsion lets the unconstrained Cayley rotation collapse to a
/// spuriously low, unphysical stationary point; see `oo_rimp2.rs`'s
/// `compute_hf_energy` doc comment for the measured ~1.6 Ha closed-shell
/// counterpart of this same bug).
#[allow(clippy::too_many_arguments)]
fn compute_uhf_energy(
    ctx: &ParallelContext,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    nocc_a: usize,
    nocc_b: usize,
    h: &Array2<f64>,
    vnn: f64,
    budget_bytes: usize,
) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
    let n = prep.nbasis();
    // Densities D_σ = C_occ_σ · C_occ_σ^T (unit occupancy, not 2× for spin).
    let mut d_a = Array2::zeros((n, n));
    let mut d_b = Array2::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            let mut sa = 0.0;
            let mut sb = 0.0;
            for i in 0..nocc_a {
                sa += c_a[(mu, i)] * c_a[(nu, i)];
            }
            for i in 0..nocc_b {
                sb += c_b[(mu, i)] * c_b[(nu, i)];
            }
            d_a[(mu, nu)] = sa;
            d_b[(mu, nu)] = sb;
        }
    }
    let d_tot = &d_a + &d_b;

    // J from D_tot; K per spin from D_σ. Mirror the UHF pattern.
    let mut j_tot = Array2::zeros((n, n));
    let mut k_a = Array2::zeros((n, n));
    let mut k_b = Array2::zeros((n, n));
    {
        let mut dj = DirectJ::new(ctx, prep, bounds, 1e-12, budget_bytes);
        dj.build(&d_tot, &mut j_tot)?;
    }
    {
        let mut dk = DirectK::new(ctx, prep, bounds, 1e-12, budget_bytes);
        <DirectK as KBuilder>::build(&mut dk, &d_a, &mut k_a)?;
    }
    if nocc_b > 0 {
        let mut dk = DirectK::new(ctx, prep, bounds, 1e-12, budget_bytes);
        <DirectK as KBuilder>::build(&mut dk, &d_b, &mut k_b)?;
    }

    let f_a = h + &j_tot - &k_a;
    let f_b = h + &j_tot - &k_b;

    // E_elec = ½ Σ_{μν} [(H + F_α)_μν D_α_μν + (H + F_β)_μν D_β_μν]
    let mut e_elec = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            e_elec += 0.5 * (h[(mu, nu)] + f_a[(mu, nu)]) * d_a[(mu, nu)];
            e_elec += 0.5 * (h[(mu, nu)] + f_b[(mu, nu)]) * d_b[(mu, nu)];
        }
    }
    let e_hf = e_elec + vnn;
    Ok((e_hf, f_a, f_b))
}

/// Build a temporary ScfResult from C_a, C_b, eps_a, eps_b for passing into
/// the U-MP2 amplitude/gradient routines. The Fock and density fields aren't
/// used downstream but must be populated.
// Assembles a ScfResult from its spin-resolved pieces; each arg maps to a
// distinct alpha/beta field of the result, so there is no bundle to extract.
#[allow(clippy::too_many_arguments)]
fn make_scf_view(
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    f_a: &Array2<f64>,
    f_b: &Array2<f64>,
    eps_a: Vec<f64>,
    eps_b: Vec<f64>,
    nocc_a: usize,
    nocc_b: usize,
    e_hf: f64,
) -> ScfResult {
    let c_a_occ = c_a.slice(ndarray::s![.., ..nocc_a]).to_owned();
    let c_b_occ = c_b.slice(ndarray::s![.., ..nocc_b]).to_owned();
    let d_a = c_a_occ.dot(&c_a_occ.t());
    let d_b = c_b_occ.dot(&c_b_occ.t());
    let d_tot = &d_a + &d_b;
    ScfResult {
        spin: Spin::Unrestricted,
        energy: e_hf,
        density_total: d_tot,
        density_alpha: d_a,
        density_beta: Some(d_b),
        mos_alpha: c_a.clone(),
        mos_beta: Some(c_b.clone()),
        eps_alpha: eps_a,
        eps_beta: Some(eps_b),
        fock_alpha: f_a.clone(),
        fock_beta: Some(f_b.clone()),
        converged: true,
        exit: ferric_scf::result::ScfExit::Converged,
        iterations: 0,
        computed_quartets: 0,
        induced_dipoles: None,
        stability: None,
        df_jk: None,
        rohf_spin_focks: None,
    }
}

/// MO-basis orbital energies from diagonal of F_mo = C^T F C.
fn orbital_energies_mo(c: &Array2<f64>, f: &Array2<f64>) -> Vec<f64> {
    let f_mo = c.t().dot(f).dot(c);
    (0..f_mo.nrows()).map(|i| f_mo[(i, i)]).collect()
}

/// Fail-fast pre-flight guard for the amplitude trio `t_aa`/`t_bb`/`t_ab` that
/// [`compute_u_mp2_amplitudes`] builds and [`u_oo_ri_mp2`] keeps resident
/// across the whole orbital-optimization loop (`amps`, see the "MEMORY NOTE"
/// on that function) — tripled relative to the closed-shell `oo_rimp2.rs`
/// guard since all three spin channels are co-resident simultaneously, not
/// just one `nov²` buffer. Sizes:
///   t_aa: nocc_a²·nvir_a², t_bb: nocc_b²·nvir_b², t_ab: nocc_a·nocc_b·nvir_a·nvir_b
fn check_u_amplitude_alloc(
    label: &str,
    nocc_a: usize,
    nvir_a: usize,
    nocc_b: usize,
    nvir_b: usize,
    budget_bytes: usize,
) -> Result<(), FerricError> {
    let t_aa_elems = nocc_a
        .saturating_mul(nocc_a)
        .saturating_mul(nvir_a)
        .saturating_mul(nvir_a);
    let t_bb_elems = nocc_b
        .saturating_mul(nocc_b)
        .saturating_mul(nvir_b)
        .saturating_mul(nvir_b);
    let t_ab_elems = nocc_a
        .saturating_mul(nocc_b)
        .saturating_mul(nvir_a)
        .saturating_mul(nvir_b);
    let peak = t_aa_elems
        .saturating_add(t_bb_elems)
        .saturating_add(t_ab_elems)
        .saturating_mul(8);
    ferric_core::memory::check_alloc(
        &format!(
            "{label} (nocc_a={nocc_a}, nvir_a={nvir_a}, nocc_b={nocc_b}, nvir_b={nvir_b}; \
             co-resident t_aa+t_bb+t_ab amplitude trio)"
        ),
        peak,
        budget_bytes,
    )
}

/// Fail-fast pre-flight guard for the `b_full_a`/`b_full_b` full-MO B tensors
/// held simultaneously every gradient step in [`u_oo_ri_mp2`]'s main loop
/// (each `(naux, nmo, nmo)` f64 — see the "MEMORY NOTE" on that function).
fn check_u_bfull_alloc(
    label: &str,
    naux: usize,
    nmo: usize,
    budget_bytes: usize,
) -> Result<(), FerricError> {
    let per_tensor = naux.saturating_mul(nmo).saturating_mul(nmo);
    let peak = per_tensor.saturating_mul(2).saturating_mul(8);
    ferric_core::memory::check_alloc(
        &format!("{label} (naux={naux}, nmo={nmo}; co-resident b_full_a+b_full_b)"),
        peak,
        budget_bytes,
    )
}

/// Apply Brillouin HF gradient term `+2·F^σ_{ai}` (MO basis) to the existing
/// MP2 gradient block. Returns `g_total = g_mp2 + 2·F^σ_{a+nocc, i}`.
///
/// Sign convention: U-OO-MP2 uses Cayley `U = (I − κ/2)^{−1}(I + κ/2)`, so
/// `g = +∂E/∂κ` (positive gradient). The Newton step is `κ = −g/(gap+μ)`,
/// driving κ down the gradient.  At HF stationarity F_ai = 0, so g_HF = 0
/// there. As MP2 perturbs orbitals, F_ai becomes nonzero and `+2·F_ai` is
/// the restoring force.
fn add_hf_gradient(g_mp2: &Array2<f64>, f_mo: &Array2<f64>, nocc: usize) -> Array2<f64> {
    let (nvir, nocc_check) = g_mp2.dim();
    assert_eq!(nocc, nocc_check);
    let mut g = g_mp2.clone();
    for a in 0..nvir {
        for i in 0..nocc {
            g[(a, i)] += 2.0 * f_mo[(nocc + a, i)];
        }
    }
    g
}

/// Orbital-space sizes of one U-OO-RI-MP2 run (per spin).
#[derive(Debug, Clone, Copy)]
struct USpaces {
    nocc_total_a: usize,
    nocc_total_b: usize,
    nocc_a: usize,
    nocc_b: usize,
    nvir_a: usize,
    nvir_b: usize,
    first_occ: usize,
}

/// Everything the solver needs at one orbital pair `(c_a, c_b)` (the solver's
/// own, continuous frame), evaluated once. Energies and amplitudes are in the
/// per-spin SEMICANONICAL frame `c_σ·u_σ` (see
/// [`crate::oo_rimp2::semicanonical_rotation`]), which makes the U-MP2 part
/// the full-Fock (non-canonical) Hylleraas functional, invariant to occ-occ
/// and vir-vir rotations within each spin.
struct UPoint {
    e_hf: f64,
    f_a: Array2<f64>,
    f_b: Array2<f64>,
    u_a: Array2<f64>,
    u_b: Array2<f64>,
    c_a_sc: Array2<f64>,
    c_b_sc: Array2<f64>,
    eps_a_sc: Vec<f64>,
    eps_b_sc: Vec<f64>,
    amps: UMp2Amplitudes,
}

impl UPoint {
    fn e_mp2(&self) -> f64 {
        self.amps.components.e_total
    }
    fn total(&self) -> f64 {
        self.e_hf + self.e_mp2()
    }
}

/// Evaluate UHF + U-RI-MP2 (semicanonical frame) at `(c_a, c_b)`.
#[allow(clippy::too_many_arguments)]
fn u_evaluate_point(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    sp: &USpaces,
    h: &Array2<f64>,
    vnn: f64,
    budget_bytes: usize,
    mp2_cfg: &crate::rimp2::RiMp2Config,
    label: &str,
) -> Result<UPoint, FerricError> {
    let (e_hf, f_a, f_b) = compute_uhf_energy(
        ctx,
        obs,
        bounds,
        c_a,
        c_b,
        sp.nocc_total_a,
        sp.nocc_total_b,
        h,
        vnn,
        budget_bytes,
    )?;
    let (u_a, eps_a_sc) =
        semicanonical_rotation(&c_a.t().dot(&f_a).dot(c_a), sp.first_occ, sp.nocc_total_a)?;
    let (u_b, eps_b_sc) =
        semicanonical_rotation(&c_b.t().dot(&f_b).dot(c_b), sp.first_occ, sp.nocc_total_b)?;
    let c_a_sc = c_a.dot(&u_a);
    let c_b_sc = c_b.dot(&u_b);
    // Fail-fast guard before building the amplitude trio — see
    // check_u_amplitude_alloc's doc comment for the co-residency this bounds
    // (t_aa+t_bb+t_ab, all three spin channels at once).
    check_u_amplitude_alloc(
        label,
        sp.nocc_a,
        sp.nvir_a,
        sp.nocc_b,
        sp.nvir_b,
        budget_bytes,
    )?;
    let scf_view = make_scf_view(
        &c_a_sc,
        &c_b_sc,
        &f_a,
        &f_b,
        eps_a_sc.clone(),
        eps_b_sc.clone(),
        sp.nocc_total_a,
        sp.nocc_total_b,
        e_hf,
    );
    let amps = compute_u_mp2_amplitudes(mol, obs, dfbs, op, &scf_view, mp2_cfg)?;
    Ok(UPoint {
        e_hf,
        f_a,
        f_b,
        u_a,
        u_b,
        c_a_sc,
        c_b_sc,
        eps_a_sc,
        eps_b_sc,
        amps,
    })
}

/// Embed the active occ-occ and vir-vir U-MP2 density blocks into an
/// `(nmo, nmo)` matrix `W^σ = dE_MP2/dF^σ` (symmetrised).
fn embed_u_density(
    p_oo: &Array2<f64>,
    p_vv: &Array2<f64>,
    first_occ: usize,
    nocc_total: usize,
    nmo: usize,
) -> Array2<f64> {
    let mut w = Array2::<f64>::zeros((nmo, nmo));
    let no = p_oo.nrows();
    let nv = p_vv.nrows();
    for i in 0..no {
        for j in 0..no {
            w[(first_occ + i, first_occ + j)] = 0.5 * (p_oo[(i, j)] + p_oo[(j, i)]);
        }
    }
    for a in 0..nv {
        for b in 0..nv {
            w[(nocc_total + a, nocc_total + b)] = 0.5 * (p_vv[(a, b)] + p_vv[(b, a)]);
        }
    }
    w
}

/// Full U-OO-MP2 orbital gradient `(g_a, g_b)` (active columns only) in the
/// SOLVER's frame.
///
/// In the semicanonical frame, for a rotation `kappa^τ_ck` of spin τ
/// (`dC_k = C_c`, `dC_c = −C_k`, so `dD^τ = C_c C_kᵀ + C_k C_cᵀ`):
///
/// ```text
///   g^τ_ck = 2 F^τ_ck                                         (UHF)
///          + [U-MP2 integral response at fixed denominators]  (compute_u_mp2_orbital_gradient)
///          + 2·[C^τᵀ (J[P^α + P^β] − K[P^τ]) C^τ]_ck             (Fock response, density part)
///          + 2·[(F^τ W^τ) − (W^τ F^τ)]_ck                       (Fock response, rotation part)
/// ```
///
/// with `W^σ = dE_MP2/dF^σ` the unrelaxed U-MP2 density (`build_u_mp2_density`,
/// occ-occ and vir-vir blocks) and `P^σ = C W^σ Cᵀ`. The rotated gradient is
/// then carried back by `g^σ = U_v^σ g^σ_sc U_o^σᵀ`.
///
/// DEFECT F2 (2026-09-24): before this function existed the U solver used only
/// the first two lines. It had NO orbital-energy response at all, which is the
/// term the closed-shell code lacked before 2026-07-20. Measured against
/// 4-point FD of its own functional
/// (`scripts/oo_mp2_stationarity_proto.py open`), the shipped gradient missed
/// by 1.7e-3 (OH/STO-3G), 5.6e-3 (OH/6-31G) and 4.3e-3 (NH2/6-31G) already at
/// the UHF point. Its "converged" points had FD gradients of 1.6e-3..5.4e-3,
/// with energies 1.3e-5..8.1e-5 Ha above the textbook stationary energy
/// (OH/STO-3G, NH2/6-31G, OH/6-31G). This formula: 1e-11..3e-11 at
/// ‖κ‖ = 0, 0.05 and 0.2.
#[allow(clippy::too_many_arguments)]
fn u_gradient_solver_frame(
    ctx: &ParallelContext,
    obs: &PreparedBasis,
    bounds: &SchwarzBounds,
    ao: &OoRiMp2AoTensors,
    p: &UPoint,
    sp: &USpaces,
    budget_bytes: usize,
    label: &str,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let nmo = p.c_a_sc.ncols();
    let n = p.c_a_sc.nrows();
    let USpaces {
        nocc_total_a,
        nocc_total_b,
        nocc_a,
        nocc_b,
        nvir_b,
        first_occ,
        ..
    } = *sp;
    // Fail-fast guard: b_full_a and b_full_b (each (naux, nmo, nmo)) are both
    // resident simultaneously for the gradient contraction just below.
    check_u_bfull_alloc(label, ao.naux(), nmo, budget_bytes)?;
    let (g_mp2_a, g_mp2_b) = {
        let b_full_a = compute_b_full_mo_with(ao, &p.c_a_sc)?;
        let b_full_b = compute_b_full_mo_with(ao, &p.c_b_sc)?;
        compute_u_mp2_orbital_gradient(&p.amps, &b_full_a, &b_full_b, budget_bytes)
    };
    let f_mo_a = p.c_a_sc.t().dot(&p.f_a).dot(&p.c_a_sc);
    let f_mo_b = p.c_b_sc.t().dot(&p.f_b).dot(&p.c_b_sc);
    let mut g_a = add_hf_gradient(&g_mp2_a, &f_mo_a, nocc_total_a);
    let mut g_b = if nocc_b == 0 {
        // No β occupied orbitals — zero β gradient (β is closed-shell vacuum).
        Array2::<f64>::zeros((nvir_b, nocc_total_b))
    } else {
        add_hf_gradient(&g_mp2_b, &f_mo_b, nocc_total_b)
    };

    // Fock response of the U-MP2 energy (both spins' Fock matrices depend on
    // both spins' densities through J; each on its own through K).
    let dens = build_u_mp2_density(&p.amps);
    let w_a = embed_u_density(&dens.p_oo_a, &dens.p_vv_a, first_occ, nocc_total_a, nmo);
    let w_b = embed_u_density(&dens.p_oo_b, &dens.p_vv_b, first_occ, nocc_total_b, nmo);
    let pa_ao = p.c_a_sc.dot(&w_a).dot(&p.c_a_sc.t());
    let pb_ao = p.c_b_sc.dot(&w_b).dot(&p.c_b_sc.t());
    let mut j_tot = Array2::<f64>::zeros((n, n));
    let mut k_a = Array2::<f64>::zeros((n, n));
    let mut k_b = Array2::<f64>::zeros((n, n));
    {
        let mut dj = DirectJ::new(ctx, obs, bounds, 1e-12, budget_bytes);
        dj.build(&(&pa_ao + &pb_ao), &mut j_tot)?;
    }
    {
        let mut dk = DirectK::new(ctx, obs, bounds, 1e-12, budget_bytes);
        <DirectK as KBuilder>::build(&mut dk, &pa_ao, &mut k_a)?;
    }
    if nocc_b > 0 {
        let mut dk = DirectK::new(ctx, obs, bounds, 1e-12, budget_bytes);
        <DirectK as KBuilder>::build(&mut dk, &pb_ao, &mut k_b)?;
    }
    let add_response = |g: &mut Array2<f64>,
                        c: &Array2<f64>,
                        k: &Array2<f64>,
                        w: &Array2<f64>,
                        f_mo: &Array2<f64>,
                        nocc_total: usize| {
        let g_mo = c.t().dot(&(&j_tot - k)).dot(c);
        let fw = f_mo.dot(w);
        let wf = w.dot(f_mo);
        let (nvir, nocc_cols) = g.dim();
        for a in 0..nvir {
            let a_mo = nocc_total + a;
            for i in 0..nocc_cols {
                g[(a, i)] += 2.0 * g_mo[(a_mo, i)] + 2.0 * (fw[(a_mo, i)] - wf[(a_mo, i)]);
            }
        }
    };
    add_response(&mut g_a, &p.c_a_sc, &k_a, &w_a, &f_mo_a, nocc_total_a);
    if nocc_b > 0 {
        add_response(&mut g_b, &p.c_b_sc, &k_b, &w_b, &f_mo_b, nocc_total_b);
    }

    // Active columns, rotated back to the solver's frame.
    let back = |g: &Array2<f64>, u: &Array2<f64>, nocc: usize, nocc_total: usize| {
        let g_act = g
            .slice(ndarray::s![.., first_occ..first_occ + nocc])
            .to_owned();
        let u_o = u.slice(ndarray::s![
            first_occ..first_occ + nocc,
            first_occ..first_occ + nocc
        ]);
        let u_v = u.slice(ndarray::s![nocc_total.., nocc_total..]);
        u_v.dot(&g_act).dot(&u_o.t())
    };
    let g_a_act = back(&g_a, &p.u_a, nocc_a, nocc_total_a);
    let g_b_act = if nocc_b == 0 {
        Array2::<f64>::zeros((nvir_b, 0))
    } else {
        back(&g_b, &p.u_b, nocc_b, nocc_total_b)
    };
    Ok((g_a_act, g_b_act))
}

/// Drive U-OO-RI-MP2 from a converged UHF or ROHF reference.
///
/// The returned `mos_alpha`/`mos_beta` are in the per-spin SEMICANONICAL frame
/// and `eps_alpha`/`eps_beta` are the MP2 denominators actually used.
pub fn u_oo_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    uhf: &ScfResult,
    config: &UOoRiMp2Config,
    ext: Option<&ExternalPotential>,
) -> Result<UOoRiMp2Result, FerricError> {
    if matches!(uhf.spin, Spin::Restricted) {
        return Err(FerricError::General(
            "u_oo_ri_mp2: requires UHF or ROHF reference".into(),
        ));
    }
    let ctx = ParallelContext::default();
    let nbas = obs.nbasis();
    let nelec_total = mol.nelec();
    let two_s = mol.multiplicity as i32 - 1;
    let nocc_total_a = ((nelec_total + two_s) / 2) as usize;
    let nocc_total_b = ((nelec_total - two_s) / 2) as usize;
    let nocc_a = active_occ(nocc_total_a, config.frozen_core)?;
    let nocc_b = active_occ(nocc_total_b, config.frozen_core)?;
    let first_occ = config.frozen_core;
    let nvir_a = nbas - nocc_total_a;
    let nvir_b = nbas - nocc_total_b;
    let sp = USpaces {
        nocc_total_a,
        nocc_total_b,
        nocc_a,
        nocc_b,
        nvir_a,
        nvir_b,
        first_occ,
    };

    // Initial MOs: copy from reference. For ROHF, β shares α MOs but with
    // different occupation (SOMO unoccupied in β). `c_a`/`c_b` are the
    // solver's own frame, kept continuous across iterations (DIIS mixes
    // successive iterates); energies and gradients are evaluated in the
    // semicanonical frame of each (see `u_evaluate_point`).
    let mut c_a = uhf.mos_a().clone();
    let mut c_b = match uhf.spin {
        Spin::Unrestricted => uhf.mos_b().clone(),
        Spin::RestrictedOpen => uhf.mos_a().clone(),
        Spin::Restricted => unreachable!(),
    };

    // `ext = None` is byte-for-byte identical to the pre-fix bare
    // `oneelectron::hcore(obs)` call -- this was the SAME hcore bug fixed in
    // closed-shell `oo_rimp2::oo_ri_mp2`, present here too (open-shell was
    // named in spec section 3 but not fixed in the first pass).
    let h = oneelectron::hcore_with_external(obs, ext)?;
    // Classical constant threaded into every compute_uhf_energy call below --
    // see that function's doc comment for why this must be computed ONCE
    // here rather than recomputed from `mol` alone (the second, independent
    // bug from the closed-shell fix: a missing classical charge-nuclear/
    // field-nuclear term lets the unconstrained orbital rotation collapse).
    let vnn = mol.nuclear_repulsion()
        + ext
            .map(|e| e.charge_nuclear_energy(mol) + e.field_nuclear_energy(mol))
            .unwrap_or(0.0);
    // AO overlap, for re-orthonormalising DIIS-extrapolated orbitals.
    let s_ao = oneelectron::overlap(obs);

    // Resolved once per call (not per iteration) — gates the AO-tensor build,
    // the RiMp2Config threaded into every compute_u_mp2_amplitudes call, and
    // the amps/b_full guards below, all off the same configured ceiling.
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(config.memory_budget_bytes);
    let mp2_cfg = crate::rimp2::RiMp2Config {
        frozen_core: config.frozen_core,
        memory_budget_bytes: config.memory_budget_bytes,
        ..Default::default()
    };

    // AO-side invariants for the full-MO B tensors: built once, reused every iter.
    // Served through the budgeted ThreeIndexSource (thread the config budget,
    // M1 resolver, rather than the env-only default `build` used), so the raw
    // (naux, nao, nao) AO tensor is no longer held resident across the whole
    // orbital-optimization loop when it exceeds the budget.
    //
    // MEMORY NOTE (deliberately not restructured in the M3 lane): `amps` keeps
    // the t_aa/t_bb/t_ab amplitude trio (nocc²·nvir² each) resident across
    // iterations, and the gradient step holds b_full_a AND b_full_b
    // (naux·nmo² each) simultaneously — the remaining O(N⁴) residents here.
    // Both are guarded by `check_u_amplitude_alloc`/`check_u_bfull_alloc`
    // rather than left as an unchecked O(N⁴) allocation.
    let ao = OoRiMp2AoTensors::build_with_budget(obs, dfbs, op, budget_bytes)?;

    let eval = |ca: &Array2<f64>, cb: &Array2<f64>, label: &str| {
        u_evaluate_point(
            &ctx,
            mol,
            obs,
            dfbs,
            op,
            bounds,
            ca,
            cb,
            &sp,
            &h,
            vnn,
            budget_bytes,
            &mp2_cfg,
            label,
        )
    };

    let mut point = eval(&c_a, &c_b, "U-OO-RI-MP2 initial amplitude trio")?;
    let mut grad_norm = f64::MAX;

    let finish = |p: UPoint, converged: bool, iterations: usize, grad_norm: f64| {
        let total_energy = p.total();
        let mp2_corr = p.e_mp2();
        UOoRiMp2Result {
            total_energy,
            hf_energy: p.e_hf,
            mp2_corr,
            components: p.amps.components.clone(),
            converged,
            iterations,
            grad_norm,
            mos_alpha: p.c_a_sc,
            mos_beta: p.c_b_sc,
            eps_alpha: p.eps_a_sc,
            eps_beta: p.eps_b_sc,
        }
    };

    let mut diis_a = if config.use_diis {
        Some(Diis::new(config.diis_size))
    } else {
        None
    };
    let mut diis_b = if config.use_diis {
        Some(Diis::new(config.diis_size))
    } else {
        None
    };

    let mut mu = config.level_shift;
    let mut stuck_count: usize = 0;
    const STUCK_LIMIT: usize = 3;
    const MU_MAX: f64 = 5.0;

    for iter in 1..=config.max_iter {
        let (g_a_act, g_b_act) = u_gradient_solver_frame(
            &ctx,
            obs,
            bounds,
            &ao,
            &point,
            &sp,
            budget_bytes,
            &format!("U-OO-RI-MP2 b_full_a/b_full_b (iter {iter})"),
        )?;

        let gn2_a: f64 = g_a_act.iter().map(|x| x * x).sum();
        let gn2_b: f64 = g_b_act.iter().map(|x| x * x).sum();
        grad_norm = (gn2_a + gn2_b).sqrt();

        // Live per-iteration progress; STDOUT, opt-in via `config.verbose`.
        // See `OoRiMp2Config::verbose` / `RhfConfig::verbose` for the
        // convention this mirrors.
        if config.verbose {
            println!(
                "U-OO-RI-MP2 iter {:3}: E_HF={:.10} E_MP2={:.10} E_tot={:.10} |g|={:.2e} (|g_a|={:.2e} |g_b|={:.2e})",
                iter, point.e_hf, point.e_mp2(), point.total(), grad_norm, gn2_a.sqrt(), gn2_b.sqrt()
            );
        }

        if grad_norm < config.grad_conv {
            return Ok(finish(point, true, iter, grad_norm));
        }

        // Newton step per spin: κ^σ_{ai} = -g^σ_{ai} / (ε^σ_a - ε^σ_i + μ),
        // with the diagonal Fock elements of the solver's own frame (the frame
        // `g` lives in) as the preconditioner.
        let eps_a = orbital_energies_mo(&c_a, &point.f_a);
        let eps_b = orbital_energies_mo(&c_b, &point.f_b);
        let mut kappa_a = Array2::<f64>::zeros((nvir_a, nocc_a));
        for a in 0..nvir_a {
            for i in 0..nocc_a {
                let gap = eps_a[nocc_total_a + a] - eps_a[first_occ + i];
                kappa_a[(a, i)] = -g_a_act[(a, i)] / (gap + mu);
            }
        }
        let mut kappa_b = Array2::<f64>::zeros((nvir_b, nocc_b));
        for a in 0..nvir_b {
            for i in 0..nocc_b {
                let gap = eps_b[nocc_total_b + a] - eps_b[first_occ + i];
                kappa_b[(a, i)] = -g_b_act[(a, i)] / (gap + mu);
            }
        }
        // Cap rotations
        let ka_max = kappa_a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if ka_max > config.max_kappa {
            kappa_a *= config.max_kappa / ka_max;
        }
        let kb_max = kappa_b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if kb_max > config.max_kappa {
            kappa_b *= config.max_kappa / kb_max;
        }

        // Build full (nmo × nmo) antisymmetric κ matrices
        let nmo = nbas;
        let mut k_a_full = Array2::<f64>::zeros((nmo, nmo));
        for a in 0..nvir_a {
            for i in 0..nocc_a {
                let a_mo = nocc_total_a + a;
                let i_mo = first_occ + i;
                k_a_full[(a_mo, i_mo)] = kappa_a[(a, i)];
                k_a_full[(i_mo, a_mo)] = -kappa_a[(a, i)];
            }
        }
        let mut k_b_full = Array2::<f64>::zeros((nmo, nmo));
        for a in 0..nvir_b {
            for i in 0..nocc_b {
                let a_mo = nocc_total_b + a;
                let i_mo = first_occ + i;
                k_b_full[(a_mo, i_mo)] = kappa_b[(a, i)];
                k_b_full[(i_mo, a_mo)] = -kappa_b[(a, i)];
            }
        }
        let u_a = cayley_rotation(&k_a_full)?;
        let u_b = cayley_rotation(&k_b_full)?;
        let mut c_a_new = c_a.dot(&u_a);
        let mut c_b_new = c_b.dot(&u_b);

        // DIIS per spin: error vector = g_antisym (full MO) projected to AO via
        // C. The extrapolated C is a linear combination of orbital sets and is
        // Löwdin re-orthonormalised (see `oo_rimp2::lowdin_orthonormalize`).
        if let Some(ref mut d) = diis_a {
            let mut g_anti = Array2::<f64>::zeros((nmo, nmo));
            for a in 0..nvir_a {
                for i in 0..nocc_a {
                    let a_mo = nocc_total_a + a;
                    let i_mo = first_occ + i;
                    g_anti[(a_mo, i_mo)] = g_a_act[(a, i)];
                    g_anti[(i_mo, a_mo)] = -g_a_act[(a, i)];
                }
            }
            let err = c_a_new.dot(&g_anti).dot(&c_a_new.t());
            c_a_new = lowdin_orthonormalize(&d.step(&c_a_new, &err), &s_ao)?;
        }
        if let Some(ref mut d) = diis_b {
            if nocc_b > 0 {
                let mut g_anti = Array2::<f64>::zeros((nmo, nmo));
                for a in 0..nvir_b {
                    for i in 0..nocc_b {
                        let a_mo = nocc_total_b + a;
                        let i_mo = first_occ + i;
                        g_anti[(a_mo, i_mo)] = g_b_act[(a, i)];
                        g_anti[(i_mo, a_mo)] = -g_b_act[(a, i)];
                    }
                }
                let err = c_b_new.dot(&g_anti).dot(&c_b_new.t());
                c_b_new = lowdin_orthonormalize(&d.step(&c_b_new, &err), &s_ao)?;
            }
        }

        // Evaluate at new orbitals
        let trial = eval(
            &c_a_new,
            &c_b_new,
            format!("U-OO-RI-MP2 amplitude trio (iter {iter})").as_str(),
        )?;
        let total_new = trial.total();
        let de = (total_new - point.total()).abs();

        // Backtracking if energy increased noticeably (DIIS can produce small uphill).
        if total_new > point.total() + 1e-4 {
            // The rejected trial is never used on this branch; free its
            // amplitude trio (and its pool reservation) before backtracking,
            // so at most `point` plus one backtrack trio are resident -- the
            // one-trio-per-point budget `check_u_amplitude_alloc` assumes.
            drop(trial);
            // Try damped pure-Newton step (no DIIS), halving until accepted.
            let mut accepted = false;
            let mut ka = k_a_full.clone();
            let mut kb = k_b_full.clone();
            let mut bt: Option<(Array2<f64>, Array2<f64>, UPoint)> = None;
            for bt_i in 0..10 {
                ka *= 0.5;
                kb *= 0.5;
                let ua2 = cayley_rotation(&ka)?;
                let ub2 = cayley_rotation(&kb)?;
                let bt_c_a = c_a.dot(&ua2);
                let bt_c_b = c_b.dot(&ub2);
                let bt_point = eval(
                    &bt_c_a,
                    &bt_c_b,
                    format!("U-OO-RI-MP2 amplitude trio (iter {iter}, backtrack {bt_i})").as_str(),
                )?;
                // Keep a backtrack point only once it is accepted; a rejected
                // one is dropped here, before the next evaluation allocates.
                if bt_point.total() <= point.total() + 1e-12 {
                    bt = Some((bt_c_a, bt_c_b, bt_point));
                    accepted = true;
                    break;
                }
            }
            if accepted {
                let (bc_a, bc_b, bp) = bt.expect("accepted implies a backtrack point");
                c_a = bc_a;
                c_b = bc_b;
                point = bp;
                if let Some(ref mut d) = diis_a {
                    d.reset();
                }
                if let Some(ref mut d) = diis_b {
                    d.reset();
                }
                stuck_count = 0;
            } else {
                stuck_count += 1;
                let mu_new = (mu * 2.0).min(MU_MAX);
                eprintln!(
                    "  backtracking failed at iter {iter} (stuck {stuck_count}/{STUCK_LIMIT}); μ {mu:.3}→{mu_new:.3}"
                );
                mu = mu_new;
                if let Some(ref mut d) = diis_a {
                    d.reset();
                }
                if let Some(ref mut d) = diis_b {
                    d.reset();
                }
                if stuck_count >= STUCK_LIMIT {
                    eprintln!(
                        "  U-OO-RI-MP2: bailing after {STUCK_LIMIT} stuck iters; returning current (non-converged) state"
                    );
                    return Ok(finish(point, false, iter, grad_norm));
                }
                // Don't accept the tiny step; keep old c_a/c_b/etc, retry next iter with larger μ.
            }
        } else {
            c_a = c_a_new;
            c_b = c_b_new;
            point = trial;
            stuck_count = 0;
        }

        if de < config.energy_conv && iter > 1 && grad_norm < config.grad_conv * 10.0 {
            return Ok(finish(point, true, iter, grad_norm));
        }
    }

    Ok(finish(point, false, config.max_iter, grad_norm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess, UhfConfig};

    /// M4 memory-guard regression: `check_u_amplitude_alloc` must fire (Err)
    /// at a realistic large open-shell scale (comparable per-spin-channel
    /// magnitude to the restricted-code incident: nocc≈60, nvir≈900 per spin)
    /// against a small configured budget, with both the estimate and the
    /// budget named in the error message.
    #[test]
    fn check_u_amplitude_alloc_rejects_realistic_large_scale() {
        let nocc_a = 60;
        let nvir_a = 900;
        let nocc_b = 58;
        let nvir_b = 902;
        let budget_bytes = ferric_core::memory::gib_to_bytes(1.0);
        let err = check_u_amplitude_alloc(
            "test large U-OO-MP2",
            nocc_a,
            nvir_a,
            nocc_b,
            nvir_b,
            budget_bytes,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("GB"),
            "expected a GB-shaped estimate in the error message, got: {msg}"
        );
        assert!(
            msg.contains("budget is"),
            "expected the configured budget named in the error message, got: {msg}"
        );
        // Sanity: this really is a huge co-resident trio (~ nocc²nvir² * 2 spins
        // + cross term, all *8 bytes -- comparable/larger than the restricted
        // t2 incident), so the rejection is not a trivially-tiny-input fluke.
        let t_aa = nocc_a * nocc_a * nvir_a * nvir_a;
        let t_bb = nocc_b * nocc_b * nvir_b * nvir_b;
        let t_ab = nocc_a * nocc_b * nvir_a * nvir_b;
        let total_gb = (t_aa + t_bb + t_ab) as f64 * 8.0 / 1e9;
        assert!(
            total_gb > 40.0,
            "test fixture too small to exercise the large-scale guard: {total_gb:.1} GB"
        );
    }

    /// Companion acceptance test: a small/typical open-shell system (OH/cc-pVDZ
    /// scale, well within the existing U-OO-MP2 test fixtures below) must NOT
    /// be rejected under a generous or auto-resolved budget.
    #[test]
    fn check_u_amplitude_alloc_accepts_small_system() {
        // OH/cc-pVDZ-ish scale: nocc_a=5, nocc_b=4, nvir_a=nvir_b=19 (see
        // u_oo_rimp2_lowers_energy_on_oh below for the real system this mirrors).
        let nocc_a = 5;
        let nocc_b = 4;
        let nvir_a = 19;
        let nvir_b = 19;
        let budget_bytes = ferric_core::memory::gib_to_bytes(1.0);
        assert!(check_u_amplitude_alloc(
            "test small U-OO-MP2",
            nocc_a,
            nvir_a,
            nocc_b,
            nvir_b,
            budget_bytes,
        )
        .is_ok());
        let auto_budget = ferric_core::memory::resolve_budget_bytes(None);
        assert!(check_u_amplitude_alloc(
            "test small U-OO-MP2 (auto budget)",
            nocc_a,
            nvir_a,
            nocc_b,
            nvir_b,
            auto_budget,
        )
        .is_ok());
    }

    /// `check_u_bfull_alloc` must likewise fire at a large (naux, nmo) scale
    /// and pass at small scale, with both estimate and budget named.
    #[test]
    fn check_u_bfull_alloc_rejects_large_and_accepts_small_scale() {
        // Large scale: naux~3000, nmo~1000 -> per tensor 3e9 elems * 8 B = 24 GB,
        // *2 (a+b) = 48 GB -- clearly over a 1 GiB budget.
        let budget_bytes = ferric_core::memory::gib_to_bytes(1.0);
        let err = check_u_bfull_alloc("test large b_full", 3000, 1000, budget_bytes).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("GB"),
            "expected a GB-shaped estimate, got: {msg}"
        );
        assert!(
            msg.contains("budget is"),
            "expected the configured budget named, got: {msg}"
        );

        // Small scale (water/cc-pVDZ-ish: naux~116, nmo~24) must pass.
        assert!(check_u_bfull_alloc("test small b_full", 116, 24, budget_bytes).is_ok());
        let auto_budget = ferric_core::memory::resolve_budget_bytes(None);
        assert!(
            check_u_bfull_alloc("test small b_full (auto budget)", 116, 24, auto_budget).is_ok()
        );
    }

    /// On a closed-shell singlet (H2 in cc-pVDZ), U-OO-RI-MP2 from a UHF
    /// reference must match closed-shell OO-RI-MP2 from an RHF reference
    /// to numerical noise.
    #[test]
    fn u_oo_rimp2_matches_closed_shell_on_h2() {
        let ctx = ParallelContext::default();
        let xyz = "2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();

        // Closed-shell OO-RI-MP2 reference
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let oo_cfg = crate::oo_rimp2::OoRiMp2Config {
            grad_conv: 1e-7,
            energy_conv: 1e-10,
            max_iter: 50,
            ..Default::default()
        };
        let cs_oo = crate::oo_rimp2::oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &rhf, &oo_cfg, None)
            .unwrap();

        // UHF reference seeded from RHF MOs to land at the same singlet solution
        let c_seed = rhf.mos_r().clone();
        let uhf_cfg = UhfConfig {
            max_iter: 200,
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        };
        let uhf = solve_uhf_with_guess(
            &ctx,
            &mol,
            &obs,
            &bounds,
            &uhf_cfg,
            Some((&c_seed, &c_seed)),
        )
        .unwrap();
        let uoo_cfg = UOoRiMp2Config {
            grad_conv: 1e-7,
            energy_conv: 1e-10,
            max_iter: 50,
            ..Default::default()
        };
        let us_oo = u_oo_ri_mp2(&mol, &obs, &dfbs, op, &bounds, &uhf, &uoo_cfg, None).unwrap();

        let de = (us_oo.total_energy - cs_oo.total_energy).abs();
        println!("CS OO-MP2 E_tot = {:.10}", cs_oo.total_energy);
        println!("US OO-MP2 E_tot = {:.10}", us_oo.total_energy);
        println!(
            "diff = {:.3e}, U converged in {} iters",
            de, us_oo.iterations
        );
        assert!(us_oo.converged, "U-OO-MP2 didn't converge on H2");
        // Both solvers now optimise the SAME invariant (full-Fock) functional
        // and start α = β, so they must reach the same energy. The former
        // 5e-4 allowance ("flat gauge direction" along degenerate π_g
        // virtuals) described the non-invariant diag-Fock functional, whose
        // converged point depended on the path. Remaining floor: different J/K
        // builders (DirectJ/K vs build_jk_with_pool, both 1e-12 screening),
        // ~1e-11, plus O(|g|²) at grad_conv 1e-7.
        assert!(de < 1e-8, "U-OO-MP2 vs CS OO-MP2 on H2: diff {de:.3e}");
    }

    /// U-OO-RI-MP2 on OH/cc-pVDZ should:
    /// (a) converge in reasonable iterations
    /// (b) lower the total energy below the UHF+U-MP2 starting point
    #[test]
    fn u_oo_rimp2_lowers_energy_on_oh() {
        let ctx = ParallelContext::default();
        let xyz = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";
        let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let uhf_cfg = UhfConfig {
            max_iter: 200,
            ..Default::default()
        };
        let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &uhf_cfg).unwrap();
        let umpc = crate::u_rimp2::u_ri_mp2(
            &mol,
            &obs,
            &dfbs,
            op,
            &uhf,
            &crate::rimp2::RiMp2Config::default(),
        )
        .unwrap();
        let e_start = uhf.energy + umpc.mp2_corr;
        println!("Starting UHF+UMP2 = {:.10}", e_start);

        let oo = u_oo_ri_mp2(
            &mol,
            &obs,
            &dfbs,
            op,
            &bounds,
            &uhf,
            &UOoRiMp2Config::default(),
            None,
        )
        .unwrap();
        println!(
            "U-OO-MP2: E_tot = {:.10}, iters = {}, |g|={:.2e}, converged={}",
            oo.total_energy, oo.iterations, oo.grad_norm, oo.converged,
        );
        assert!(
            oo.total_energy <= e_start + 1e-8,
            "U-OO-MP2 did not lower E vs UHF+UMP2"
        );
    }

    // ------------------------------------------------------------------
    // Orbital-gradient / stationarity harness (defect F2, 2026-09-24).
    // Same bar derivation as oo_rimp2.rs's harness: 4-point stencil, step
    // 1e-3, FD floor ~1e-9, corrected formula measured 1e-11..3e-11 in
    // `scripts/oo_mp2_stationarity_proto.py open`. The shipped (fixed-eps)
    // gradient missed by 1.7e-3..5.6e-3 already at the UHF point.
    // ------------------------------------------------------------------

    const U_FD_STEP: f64 = 1e-3;
    const U_ORB_GRAD_BAR: f64 = 1e-7;

    struct UOrbFd {
        ctx: ParallelContext,
        mol: Molecule,
        obs: PreparedBasis,
        dfbs: PreparedBasis,
        op: Operator,
        bounds: SchwarzBounds,
        uhf: ScfResult,
        ao: OoRiMp2AoTensors,
        sp: USpaces,
        h: Array2<f64>,
        mp2_cfg: crate::rimp2::RiMp2Config,
    }

    impl UOrbFd {
        fn new(xyz: &str, basis_name: &str, mult: usize) -> Self {
            let ctx = ParallelContext::default();
            let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
            let obs = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
            let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
            let op = Operator::coulomb();
            let bounds = SchwarzBounds::compute(op, &obs).unwrap();
            let uhf = solve_uhf(
                &ctx,
                &mol,
                &obs,
                &bounds,
                &UhfConfig {
                    max_iter: 200,
                    energy_conv: 1e-11,
                    density_conv: 1e-9,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(uhf.converged);
            let ao = OoRiMp2AoTensors::build_with_budget(&obs, &dfbs, op, usize::MAX).unwrap();
            let nbas = obs.nbasis();
            let two_s = mol.multiplicity as i32 - 1;
            let na = ((mol.nelec() + two_s) / 2) as usize;
            let nb = ((mol.nelec() - two_s) / 2) as usize;
            let sp = USpaces {
                nocc_total_a: na,
                nocc_total_b: nb,
                nocc_a: na,
                nocc_b: nb,
                nvir_a: nbas - na,
                nvir_b: nbas - nb,
                first_occ: 0,
            };
            let h = oneelectron::hcore(&obs);
            UOrbFd {
                ctx,
                mol,
                obs,
                dfbs,
                op,
                bounds,
                uhf,
                ao,
                sp,
                h,
                mp2_cfg: crate::rimp2::RiMp2Config::default(),
            }
        }

        fn point(&self, ca: &Array2<f64>, cb: &Array2<f64>) -> UPoint {
            u_evaluate_point(
                &self.ctx,
                &self.mol,
                &self.obs,
                &self.dfbs,
                self.op,
                &self.bounds,
                ca,
                cb,
                &self.sp,
                &self.h,
                self.mol.nuclear_repulsion(),
                usize::MAX,
                &self.mp2_cfg,
                "test",
            )
            .unwrap()
        }

        fn analytic(&self, ca: &Array2<f64>, cb: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
            u_gradient_solver_frame(
                &self.ctx,
                &self.obs,
                &self.bounds,
                &self.ao,
                &self.point(ca, cb),
                &self.sp,
                usize::MAX,
                "test",
            )
            .unwrap()
        }

        fn fd(&self, ca: &Array2<f64>, cb: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
            let n = ca.ncols();
            let one_spin = |alpha: bool, nocc: usize| {
                let mut g = Array2::zeros((n - nocc, nocc));
                for a in nocc..n {
                    for i in 0..nocc {
                        let e_at = |s: f64| {
                            let mut k = Array2::zeros((n, n));
                            k[(a, i)] = s * U_FD_STEP;
                            k[(i, a)] = -s * U_FD_STEP;
                            let u = cayley_rotation(&k).unwrap();
                            if alpha {
                                self.point(&ca.dot(&u), cb).total()
                            } else {
                                self.point(ca, &cb.dot(&u)).total()
                            }
                        };
                        g[(a - nocc, i)] = (-e_at(2.0) + 8.0 * e_at(1.0) - 8.0 * e_at(-1.0)
                            + e_at(-2.0))
                            / (12.0 * U_FD_STEP);
                    }
                }
                g
            };
            (
                one_spin(true, self.sp.nocc_total_a),
                one_spin(false, self.sp.nocc_total_b),
            )
        }
    }

    fn u_max_err(a: &(Array2<f64>, Array2<f64>), b: &(Array2<f64>, Array2<f64>)) -> f64 {
        let m =
            |x: &Array2<f64>, y: &Array2<f64>| (x - y).iter().map(|v| v.abs()).fold(0.0, f64::max);
        m(&a.0, &b.0).max(m(&a.1, &b.1))
    }

    /// Deterministic dense antisymmetric kappa (all blocks), Frobenius norm `norm`.
    fn u_pseudo_random_kappa(n: usize, norm: f64, seed: u64) -> Array2<f64> {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let mut k = Array2::<f64>::zeros((n, n));
        for p in 0..n {
            for q in 0..p {
                let v = next();
                k[(p, q)] = v;
                k[(q, p)] = -v;
            }
        }
        let fro = k.iter().map(|v| v * v).sum::<f64>().sqrt();
        k * (norm / fro)
    }

    const OH_XYZ: &str = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";

    /// U-OO analytic vs FD at the UHF point and at `C_UHF·U(kappa)` per spin.
    ///
    /// MUTATION NOTE (prototype, OH/6-31G): the shipped fixed-eps gradient
    /// missed FD by 5.6e-3 at κ=0, 5.8e-3 at ‖κ‖=0.05 and 7.9e-3 at 0.2.
    #[test]
    fn u_oo_gradient_matches_fd_on_and_off_reference_oh_631g() {
        let t = UOrbFd::new(OH_XYZ, "6-31g", 2);
        let ca0 = t.uhf.mos_a().clone();
        let cb0 = t.uhf.mos_b().clone();
        let n = ca0.ncols();
        for norm in [0.0, 0.05, 0.2] {
            let (ca, cb) = if norm == 0.0 {
                (ca0.clone(), cb0.clone())
            } else {
                (
                    ca0.dot(&cayley_rotation(&u_pseudo_random_kappa(n, norm, 3)).unwrap()),
                    cb0.dot(&cayley_rotation(&u_pseudo_random_kappa(n, norm, 5)).unwrap()),
                )
            };
            let an = t.analytic(&ca, &cb);
            let fd = t.fd(&ca, &cb);
            let err = u_max_err(&an, &fd);
            eprintln!("OH/6-31G U-OO ‖κ‖={norm}: max|analytic − FD| = {err:.3e}");
            assert!(
                err < U_ORB_GRAD_BAR,
                "‖κ‖={norm}: {err:.3e} ≥ {U_ORB_GRAD_BAR:e}"
            );
        }
    }

    /// The U solver's converged point must be stationary by independent FD,
    /// with orthonormal, per-spin semicanonical orbitals.
    ///
    /// MUTATION NOTE (prototype): the shipped U solver's converged OH/6-31G
    /// point had max|g_FD| = 5.4e-3 (of its own functional), with an energy
    /// 8.1e-5 Ha above the textbook stationary value.
    #[test]
    fn u_oo_converged_point_is_stationary_oh_631g() {
        let t = UOrbFd::new(OH_XYZ, "6-31g", 2);
        let cfg = UOoRiMp2Config {
            grad_conv: 1e-8,
            energy_conv: 1e-11,
            max_iter: 200,
            ..Default::default()
        };
        let oo = u_oo_ri_mp2(&t.mol, &t.obs, &t.dfbs, t.op, &t.bounds, &t.uhf, &cfg, None).unwrap();
        assert!(
            oo.converged,
            "U-OO did not converge: |g|={:.2e}",
            oo.grad_norm
        );
        let s = oneelectron::overlap(&t.obs);
        let n = oo.mos_alpha.ncols();
        let ortho = |c: &Array2<f64>| {
            (&c.t().dot(&s).dot(c) - &Array2::<f64>::eye(n))
                .iter()
                .map(|v| v.abs())
                .fold(0.0, f64::max)
        };
        let (oa, ob) = (ortho(&oo.mos_alpha), ortho(&oo.mos_beta));
        let fd = t.fd(&oo.mos_alpha, &oo.mos_beta);
        let fd_max =
            fd.0.iter()
                .chain(fd.1.iter())
                .map(|v| v.abs())
                .fold(0.0, f64::max);
        let p = t.point(&oo.mos_alpha, &oo.mos_beta);
        eprintln!(
            "OH/6-31G U-OO: iters={} E={:.12} |CᵀSC−1| α {oa:.2e} β {ob:.2e} max|g_FD|={fd_max:.2e}",
            oo.iterations, oo.total_energy
        );
        assert!((p.total() - oo.total_energy).abs() < 1e-10);
        assert!(
            oa < 1e-10 && ob < 1e-10,
            "U-OO MOs not orthonormal: {oa:.2e} {ob:.2e}"
        );
        assert!(
            fd_max < 1e-6,
            "U-OO FD gradient at convergence {fd_max:.3e} ≥ 1e-6"
        );
    }
}
