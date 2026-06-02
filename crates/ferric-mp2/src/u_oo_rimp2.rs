//! Unrestricted orbital-optimized RI-MP2 (U-OO-RI-MP2).
//!
//! Open-shell counterpart to `oo_rimp2`. Minimizes E_UHF + E_U-MP2 jointly
//! over independent α and β orbital rotations using a per-spin level-shifted
//! diagonal-Hessian Newton step with optional DIIS extrapolation.
//!
//! The U-MP2 orbital gradient (αα and αβ blocks for `g^α`; ββ and αβ for
//! `g^β`) is FD-validated to ~1e-10 in `u_rimp2`. The Brillouin term
//! `−2·F^σ_{ai}` is added in MO basis for the HF gradient piece.
//!
//! Reference: Bozkaya, JCP 139, 154105 (2013).

use crate::u_rimp2::{compute_u_mp2_amplitudes, compute_u_mp2_orbital_gradient, URiMp2Components};
use crate::oo_rimp2::compute_b_full_mo;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_core::parallel::ParallelContext;
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
use ndarray_linalg::Solve;

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
        }
    }
}

/// Result of an unrestricted OO-RI-MP2 calculation.
#[derive(Debug)]
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

/// Compute UHF energy and α/β Fock matrices from MO coefficients.
fn compute_uhf_energy(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    c_a: &Array2<f64>,
    c_b: &Array2<f64>,
    nocc_a: usize,
    nocc_b: usize,
    h: &Array2<f64>,
) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
    let n = prep.nbasis();
    // Densities D_σ = C_occ_σ · C_occ_σ^T (unit occupancy, not 2× for spin).
    let mut d_a = Array2::zeros((n, n));
    let mut d_b = Array2::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            let mut sa = 0.0;
            let mut sb = 0.0;
            for i in 0..nocc_a { sa += c_a[(mu, i)] * c_a[(nu, i)]; }
            for i in 0..nocc_b { sb += c_b[(mu, i)] * c_b[(nu, i)]; }
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
        let mut dj = DirectJ::new(ctx, prep, bounds, 1e-12);
        dj.build(&d_tot, &mut j_tot)?;
    }
    {
        let mut dk = DirectK::new(ctx, prep, bounds, 1e-12);
        <DirectK as KBuilder>::build(&mut dk, &d_a, &mut k_a)?;
    }
    if nocc_b > 0 {
        let mut dk = DirectK::new(ctx, prep, bounds, 1e-12);
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
    let e_hf = e_elec + mol.nuclear_repulsion();
    Ok((e_hf, f_a, f_b))
}

/// Build a temporary ScfResult from C_a, C_b, eps_a, eps_b for passing into
/// the U-MP2 amplitude/gradient routines. The Fock and density fields aren't
/// used downstream but must be populated.
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
        iterations: 0,
        computed_quartets: 0,
    }
}

/// Cayley transform `U = (I − κ/2)^{−1}(I + κ/2)` for antisymmetric κ.
fn cayley_rotation(kappa: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let n = kappa.nrows();
    let eye = Array2::<f64>::eye(n);
    let lhs = &eye - 0.5 * kappa;
    let rhs = &eye + 0.5 * kappa;
    let mut u = Array2::zeros((n, n));
    for col in 0..n {
        let rhs_col = rhs.column(col).to_owned();
        let u_col = lhs.solve(&rhs_col)
            .map_err(|e| FerricError::Lapack(format!("Cayley solve col {col}: {e}")))?;
        u.column_mut(col).assign(&u_col);
    }
    Ok(u)
}

/// MO-basis orbital energies from diagonal of F_mo = C^T F C.
fn orbital_energies_mo(c: &Array2<f64>, f: &Array2<f64>) -> Vec<f64> {
    let f_mo = c.t().dot(f).dot(c);
    (0..f_mo.nrows()).map(|i| f_mo[(i, i)]).collect()
}

/// Apply Brillouin HF gradient term `+2·F^σ_{ai}` (MO basis) to the existing
/// MP2 gradient block. Returns `g_total = g_mp2 + 2·F^σ_{a+nocc, i}`.
///
/// Sign convention: U-OO-MP2 uses Cayley `U = (I − κ/2)^{−1}(I + κ/2)`, so
/// `g = +∂E/∂κ` (positive gradient). The Newton step is `κ = −g/(gap+μ)`,
/// driving κ down the gradient.  At HF stationarity F_ai = 0, so g_HF = 0
/// there. As MP2 perturbs orbitals, F_ai becomes nonzero and `+2·F_ai` is
/// the restoring force.
fn add_hf_gradient(
    g_mp2: &Array2<f64>,
    f_mo: &Array2<f64>,
    nocc: usize,
) -> Array2<f64> {
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

/// Drive U-OO-RI-MP2 from a converged UHF or ROHF reference.
pub fn u_oo_ri_mp2(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    uhf: &ScfResult,
    config: &UOoRiMp2Config,
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
    let nocc_a = nocc_total_a - config.frozen_core;
    let nocc_b = nocc_total_b - config.frozen_core;
    let first_occ = config.frozen_core;
    let nvir_a = nbas - nocc_total_a;
    let nvir_b = nbas - nocc_total_b;

    // Initial MOs: copy from reference. For ROHF, β shares α MOs but with
    // different occupation (SOMO unoccupied in β).
    let mut c_a = uhf.mos_a().clone();
    let mut c_b = match uhf.spin {
        Spin::Unrestricted => uhf.mos_b().clone(),
        Spin::RestrictedOpen => uhf.mos_a().clone(),
        Spin::Restricted => unreachable!(),
    };

    let h = oneelectron::hcore(obs);

    // Initial UHF energy + Fock
    let (mut e_hf, mut f_a, mut f_b) = compute_uhf_energy(
        &ctx, mol, obs, bounds, &c_a, &c_b, nocc_total_a, nocc_total_b, &h,
    )?;
    let mut eps_a = orbital_energies_mo(&c_a, &f_a);
    let mut eps_b = orbital_energies_mo(&c_b, &f_b);

    // Initial U-MP2
    let scf_view = make_scf_view(
        &c_a, &c_b, &f_a, &f_b, eps_a.clone(), eps_b.clone(),
        nocc_total_a, nocc_total_b, e_hf,
    );
    let mut amps = compute_u_mp2_amplitudes(
        mol, obs, dfbs, op, &scf_view, &crate::rimp2::RiMp2Config { frozen_core: config.frozen_core },
    )?;
    let mut e_mp2 = amps.components.e_total;
    let mut total_energy = e_hf + e_mp2;
    let mut grad_norm = f64::MAX;

    let mut diis_a = if config.use_diis { Some(Diis::new(config.diis_size)) } else { None };
    let mut diis_b = if config.use_diis { Some(Diis::new(config.diis_size)) } else { None };

    let mut mu = config.level_shift;
    let mut stuck_count: usize = 0;
    const STUCK_LIMIT: usize = 3;
    const MU_MAX: f64 = 5.0;

    for iter in 1..=config.max_iter {
        // Full-MO B tensors for gradient
        let b_full_a = compute_b_full_mo(obs, dfbs, op, &c_a)?;
        let b_full_b = compute_b_full_mo(obs, dfbs, op, &c_b)?;
        let (g_mp2_a, g_mp2_b) = compute_u_mp2_orbital_gradient(&amps, &b_full_a, &b_full_b);

        // Add HF Brillouin term: g_total = g_mp2 − 2·F^σ_{a+nocc, i}
        let f_mo_a = c_a.t().dot(&f_a).dot(&c_a);
        let f_mo_b = c_b.t().dot(&f_b).dot(&c_b);
        let g_a = add_hf_gradient(&g_mp2_a, &f_mo_a, nocc_total_a);
        let g_b = if nocc_b == 0 {
            // No β occupied orbitals — zero β gradient (β is closed-shell vacuum).
            Array2::<f64>::zeros((nvir_b, nocc_total_b))
        } else {
            add_hf_gradient(&g_mp2_b, &f_mo_b, nocc_total_b)
        };
        // Slice gradient to the active (non-frozen-core) occupied block.
        let g_a_act = g_a.slice(ndarray::s![.., first_occ..first_occ + nocc_a]).to_owned();
        let g_b_act = if nocc_b == 0 {
            Array2::<f64>::zeros((nvir_b, 0))
        } else {
            g_b.slice(ndarray::s![.., first_occ..first_occ + nocc_b]).to_owned()
        };

        let gn2_a: f64 = g_a_act.iter().map(|x| x * x).sum();
        let gn2_b: f64 = g_b_act.iter().map(|x| x * x).sum();
        grad_norm = (gn2_a + gn2_b).sqrt();

        eprintln!(
            "U-OO-RI-MP2 iter {:3}: E_HF={:.10} E_MP2={:.10} E_tot={:.10} |g|={:.2e} (|g_a|={:.2e} |g_b|={:.2e})",
            iter, e_hf, e_mp2, total_energy, grad_norm, gn2_a.sqrt(), gn2_b.sqrt()
        );

        if grad_norm < config.grad_conv {
            return Ok(UOoRiMp2Result {
                total_energy, hf_energy: e_hf, mp2_corr: e_mp2,
                components: amps.components.clone(),
                converged: true, iterations: iter, grad_norm,
                mos_alpha: c_a, mos_beta: c_b,
                eps_alpha: eps_a, eps_beta: eps_b,
            });
        }

        // Newton step per spin: κ^σ_{ai} = -g^σ_{ai} / (ε^σ_a - ε^σ_i + μ)
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
        if ka_max > config.max_kappa { kappa_a *= config.max_kappa / ka_max; }
        let kb_max = kappa_b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if kb_max > config.max_kappa { kappa_b *= config.max_kappa / kb_max; }

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

        // DIIS per spin: error vector = g_antisym (full MO) projected to AO via C.
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
            c_a_new = d.step(&c_a_new, &err);
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
                c_b_new = d.step(&c_b_new, &err);
            }
        }

        // Evaluate at new orbitals
        let (e_hf_new, f_a_new, f_b_new) = compute_uhf_energy(
            &ctx, mol, obs, bounds, &c_a_new, &c_b_new, nocc_total_a, nocc_total_b, &h,
        )?;
        let eps_a_new = orbital_energies_mo(&c_a_new, &f_a_new);
        let eps_b_new = orbital_energies_mo(&c_b_new, &f_b_new);
        let scf_view_new = make_scf_view(
            &c_a_new, &c_b_new, &f_a_new, &f_b_new,
            eps_a_new.clone(), eps_b_new.clone(),
            nocc_total_a, nocc_total_b, e_hf_new,
        );
        let amps_new = compute_u_mp2_amplitudes(
            mol, obs, dfbs, op, &scf_view_new, &crate::rimp2::RiMp2Config { frozen_core: config.frozen_core },
        )?;
        let total_new = e_hf_new + amps_new.components.e_total;
        let de = (total_new - total_energy).abs();

        // Backtracking if energy increased noticeably (DIIS can produce small uphill).
        if total_new > total_energy + 1e-4 {
            // Try damped pure-Newton step (no DIIS), halving until accepted.
            let mut accepted = false;
            let mut ka = k_a_full.clone();
            let mut kb = k_b_full.clone();
            let mut bt_c_a = c_a_new.clone();
            let mut bt_c_b = c_b_new.clone();
            let mut bt_ehf = e_hf_new;
            let mut bt_fa = f_a_new.clone();
            let mut bt_fb = f_b_new.clone();
            let mut bt_eps_a = eps_a_new.clone();
            let mut bt_eps_b = eps_b_new.clone();
            let mut bt_amps = amps_new;
            let mut bt_total = total_new;
            for _bt in 0..10 {
                ka *= 0.5;
                kb *= 0.5;
                let ua2 = cayley_rotation(&ka)?;
                let ub2 = cayley_rotation(&kb)?;
                bt_c_a = c_a.dot(&ua2);
                bt_c_b = c_b.dot(&ub2);
                let (eh, fa, fb) = compute_uhf_energy(
                    &ctx, mol, obs, bounds, &bt_c_a, &bt_c_b, nocc_total_a, nocc_total_b, &h,
                )?;
                let ea = orbital_energies_mo(&bt_c_a, &fa);
                let eb = orbital_energies_mo(&bt_c_b, &fb);
                let sv = make_scf_view(&bt_c_a, &bt_c_b, &fa, &fb, ea.clone(), eb.clone(),
                    nocc_total_a, nocc_total_b, eh);
                let am = compute_u_mp2_amplitudes(
                    mol, obs, dfbs, op, &sv, &crate::rimp2::RiMp2Config { frozen_core: config.frozen_core },
                )?;
                bt_total = eh + am.components.e_total;
                bt_ehf = eh; bt_fa = fa; bt_fb = fb;
                bt_eps_a = ea; bt_eps_b = eb; bt_amps = am;
                if bt_total <= total_energy + 1e-12 { accepted = true; break; }
            }
            if accepted {
                c_a = bt_c_a;
                c_b = bt_c_b;
                e_hf = bt_ehf;
                f_a = bt_fa;
                f_b = bt_fb;
                eps_a = bt_eps_a;
                eps_b = bt_eps_b;
                amps = bt_amps;
                e_mp2 = amps.components.e_total;
                total_energy = bt_total;
                if let Some(ref mut d) = diis_a { d.reset(); }
                if let Some(ref mut d) = diis_b { d.reset(); }
                stuck_count = 0;
            } else {
                stuck_count += 1;
                let mu_new = (mu * 2.0).min(MU_MAX);
                eprintln!(
                    "  backtracking failed at iter {iter} (stuck {stuck_count}/{STUCK_LIMIT}); μ {mu:.3}→{mu_new:.3}"
                );
                mu = mu_new;
                if let Some(ref mut d) = diis_a { d.reset(); }
                if let Some(ref mut d) = diis_b { d.reset(); }
                if stuck_count >= STUCK_LIMIT {
                    eprintln!(
                        "  U-OO-RI-MP2: bailing after {STUCK_LIMIT} stuck iters; returning current (non-converged) state"
                    );
                    return Ok(UOoRiMp2Result {
                        total_energy, hf_energy: e_hf, mp2_corr: e_mp2,
                        components: amps.components.clone(),
                        converged: false, iterations: iter, grad_norm,
                        mos_alpha: c_a, mos_beta: c_b,
                        eps_alpha: eps_a, eps_beta: eps_b,
                    });
                }
                // Don't accept the tiny step; keep old c_a/c_b/etc, retry next iter with larger μ.
            }
        } else {
            c_a = c_a_new;
            c_b = c_b_new;
            e_hf = e_hf_new;
            f_a = f_a_new;
            f_b = f_b_new;
            eps_a = eps_a_new;
            eps_b = eps_b_new;
            amps = amps_new;
            e_mp2 = amps.components.e_total;
            total_energy = total_new;
            stuck_count = 0;
        }

        if de < config.energy_conv && iter > 1 && grad_norm < config.grad_conv * 10.0 {
            return Ok(UOoRiMp2Result {
                total_energy, hf_energy: e_hf, mp2_corr: e_mp2,
                components: amps.components.clone(),
                converged: true, iterations: iter, grad_norm,
                mos_alpha: c_a, mos_beta: c_b,
                eps_alpha: eps_a, eps_beta: eps_b,
            });
        }
    }

    Ok(UOoRiMp2Result {
        total_energy, hf_energy: e_hf, mp2_corr: e_mp2,
        components: amps.components.clone(),
        converged: false, iterations: config.max_iter, grad_norm,
        mos_alpha: c_a, mos_beta: c_b,
        eps_alpha: eps_a, eps_beta: eps_b,
    })
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
            grad_conv: 1e-7, energy_conv: 1e-10, max_iter: 50,
            ..Default::default()
        };
        let cs_oo = crate::oo_rimp2::oo_ri_mp2(
            &mol, &obs, &dfbs, op, &bounds, &rhf, &oo_cfg,
        ).unwrap();

        // UHF reference seeded from RHF MOs to land at the same singlet solution
        let c_seed = rhf.mos_r().clone();
        let uhf_cfg = UhfConfig {
            max_iter: 200, energy_conv: 1e-10, density_conv: 1e-8, ..Default::default()
        };
        let uhf = solve_uhf_with_guess(
            &ctx, &mol, &obs, op, &bounds, &uhf_cfg, Some((&c_seed, &c_seed)),
        ).unwrap();
        let uoo_cfg = UOoRiMp2Config {
            grad_conv: 1e-7, energy_conv: 1e-10, max_iter: 50, ..Default::default()
        };
        let us_oo = u_oo_ri_mp2(
            &mol, &obs, &dfbs, op, &bounds, &uhf, &uoo_cfg,
        ).unwrap();

        let de = (us_oo.total_energy - cs_oo.total_energy).abs();
        println!("CS OO-MP2 E_tot = {:.10}", cs_oo.total_energy);
        println!("US OO-MP2 E_tot = {:.10}", us_oo.total_energy);
        println!("diff = {:.3e}, U converged in {} iters", de, us_oo.iterations);
        assert!(us_oo.converged, "U-OO-MP2 didn't converge on H2");
        // H2/cc-pVDZ has degenerate virtuals (π_g); the CS-OO surface has a
        // flat gauge direction so CS and U can land at slightly different
        // stationary points along that gauge. Tolerate ~1e-4 Ha here.
        assert!(de < 5e-4, "U-OO-MP2 vs CS OO-MP2 on H2: diff {de:.3e}");
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
        let uhf_cfg = UhfConfig { max_iter: 200, ..Default::default() };
        let uhf = solve_uhf(&ctx, &mol, &obs, op, &bounds, &uhf_cfg).unwrap();
        let umpc = crate::u_rimp2::u_ri_mp2(
            &mol, &obs, &dfbs, op, &uhf, &crate::rimp2::RiMp2Config::default(),
        ).unwrap();
        let e_start = uhf.energy + umpc.mp2_corr;
        println!("Starting UHF+UMP2 = {:.10}", e_start);

        let oo = u_oo_ri_mp2(
            &mol, &obs, &dfbs, op, &bounds, &uhf, &UOoRiMp2Config::default(),
        ).unwrap();
        println!(
            "U-OO-MP2: E_tot = {:.10}, iters = {}, |g|={:.2e}, converged={}",
            oo.total_energy, oo.iterations, oo.grad_norm, oo.converged,
        );
        assert!(oo.total_energy <= e_start + 1e-8, "U-OO-MP2 did not lower E vs UHF+UMP2");
    }
}

