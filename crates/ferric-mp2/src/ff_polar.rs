//! Finite-field MP2 static polarizability.
//!
//! The dRPA polarizability in `ferric-rpa` is a *closed-form* linear response
//! (`α = 4 μᵀ(A+B)⁻¹μ`), which works only because dRPA's response is an exactly
//! resummable geometric series over the RI factors. MP2 has no such closed-form
//! polarizability, so we obtain the MP2 α by **finite field**:
//!
//! 1. Perturb the core Hamiltonian by `−Σ_d F_d r_d` (a uniform field couples to
//!    the dipole operator; the sign makes `α = −∂μ/∂F` come out positive).
//! 2. Re-converge RHF in that field, run RI-MP2, build the **relaxed** MP2
//!    1-PDM (the same Z-vector relaxed density the analytic gradient uses).
//! 3. The MP2 dipole is `μ_d(F) = −Tr[P_relax(F) · r_d] + Σ_A Z_A R_{A,d}`.
//! 4. Central-difference: `α_ij = −(μ_i(+F_j) − μ_i(−F_j)) / (2|F|)`.
//!
//! This is exact MP2 α (orbital-relaxed, matches PySCF's relaxed default) and
//! fully general, at the cost of `2·3 = 6` field-perturbed RHF+MP2 solves plus
//! one unperturbed solve. Closed-shell only.
//!
//! Compared to the RPA static-α path it adds correlation **beyond** RPA — the
//! lever for testing whether MP2 correlation lifts α toward reference where
//! attenuated RPA could not (see the attenuated_alpha_c6_probe experiment).

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::diis::Diis;
use ferric_scf::rhf::{build_jk, RhfConfig};
use ferric_scf::result::{ScfResult, Spin};
use ferric_scf::screening::SchwarzBounds;

/// `FERRIC_FF_TRACE` descriptor: finite-field α driver trace (env-only debug toggle).
static FF_TRACE: ferric_core::config::ConfigVar<bool> = ferric_core::config::ConfigVar {
    env_name: "FERRIC_FF_TRACE",
    default: false,
    parse: ferric_core::config::parse_toggle,
    validate: ferric_core::config::accept_any,
};
fn ff_trace() -> bool {
    FF_TRACE.toggle()
}

/// `FERRIC_FF_UNRELAXED` descriptor: use the unrelaxed (no orbital-response)
/// MP2 density instead of the relaxed one — a dev-only diagnostic for isolating
/// the orbital-relaxation contribution to α in this (experimental) finite-field
/// path. Routed through the shared toggle so `FERRIC_FF_UNRELAXED=0` means off
/// (the old `== Some("1")` read already treated 0 as off, but any other value
/// too — this canonicalizes it).
static FF_UNRELAXED: ferric_core::config::ConfigVar<bool> = ferric_core::config::ConfigVar {
    env_name: "FERRIC_FF_UNRELAXED",
    default: false,
    parse: ferric_core::config::parse_toggle,
    validate: ferric_core::config::accept_any,
};
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};

use crate::rimp2::{compute_mp2_intermediates, RiMp2Config};
use crate::zvector::{build_lagrangian, build_relaxed_density_ao};

/// Which MP2 1-PDM to difference for the finite-field polarizability.
///
/// * `Relaxed` — full orbital-relaxed density (SCF core + P_oo/P_vv + Z-vector).
///   Matches PySCF's default relaxed α, BUT the Z-vector CPHF response is
///   near-singular for symmetric molecules under a symmetry-axis field
///   (nh3 C3v, co2 D∞h, n2): the smallest response eigenvalue λ_min ∝ F, so
///   z ~ 1/F and the differenced dipole diverges. Robust only for low-symmetry
///   / large-dipole systems (water, ch4). See attenuated-mp2-alpha-experiment.
/// * `Unrelaxed` — z=0 (no orbital relaxation). No near-singular mode, so μ(F)
///   is smooth and α is stable for ALL molecules. Differs from relaxed α by a
///   few % (a known MP2 density distinction, not an error).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DensityMode {
    #[default]
    Relaxed,
    Unrelaxed,
}

/// Static (ω=0) MP2 polarizability tensor in atomic units.
#[derive(Debug, Clone)]
pub struct Mp2Polarizability {
    /// Cartesian α_ij(0) tensor, i,j ∈ {x,y,z}, a.u. (e²·a₀²/E_h).
    pub tensor: [[f64; 3]; 3],
    /// Isotropic average (1/3) Tr α.
    pub iso: f64,
    /// Principal values (eigenvalues of the symmetrized tensor), ascending.
    pub principal: [f64; 3],
}

/// Closed-shell RHF re-converged in a uniform external dipole field.
///
/// `external_v` is added to the core Hamiltonian (already includes the field
/// coupling `−Σ_d F_d r_d` built by the caller). Pure-HF only: no DFT, RSH, or
/// cDFT branches — finite-field α perturbs the HF reference and reads the MP2
/// relaxed density on top, so the SCF here is the plain Roothaan-Hall loop.
///
/// `pub(crate)` so `cpks_polar`'s ∂t2/∂F finite-difference ORACLE can build a
/// field-perturbed reference (the analytic ∂t2 is validated against the central
/// difference of t2 rebuilt from this).
pub(crate) fn solve_rhf_with_external(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    config: &RhfConfig,
    external_v: &Array2<f64>,
) -> Result<ScfResult, FerricError> {
    let s = oneelectron::overlap(prep);
    let h = &oneelectron::hcore(prep) + external_v;
    let n = prep.nbasis();
    let nelec = mol.nelec();
    if nelec % 2 != 0 {
        return Err(FerricError::ScfConvergence { iterations: 0, last_energy: 0.0 });
    }
    let nocc = (nelec / 2) as usize;
    let vnn = mol.nuclear_repulsion();

    // S^{-1/2} for the symmetric (Löwdin) orthogonalization.
    let (s_evals, s_evecs) = s
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("S diag: {e}")))?;
    let mut u_scaled = s_evecs.clone();
    for i in 0..n {
        let scale = 1.0 / s_evals[i].sqrt();
        for mu in 0..n {
            u_scaled[(mu, i)] *= scale;
        }
    }
    let s_inv_sqrt = u_scaled.dot(&s_evecs.t());

    let diag = |f: &Array2<f64>| -> Result<(Vec<f64>, Array2<f64>), FerricError> {
        // F' = S^{-1/2} F S^{-1/2}; eigvecs back-transformed by S^{-1/2}.
        let f_orth = s_inv_sqrt.dot(f).dot(&s_inv_sqrt);
        let (e, c_orth) = f_orth
            .eigh(UPLO::Upper)
            .map_err(|err| FerricError::Lapack(format!("F diag: {err}")))?;
        Ok((e.to_vec(), s_inv_sqrt.dot(&c_orth)))
    };

    // Core guess.
    let (_, c0) = diag(&h)?;
    let c_occ0 = c0.slice(ndarray::s![.., ..nocc]);
    let mut d = 2.0 * c_occ0.dot(&c_occ0.t());

    let mut j_buf = Array2::<f64>::zeros((n, n));
    let mut k_buf = Array2::<f64>::zeros((n, n));
    let mut diis = Diis::new(config.diis_size);
    let mut prev_e = 0.0;
    let mut total_quartets = 0;

    for iter in 1..=config.max_iter {
        ctx.check_interrupted()?;
        j_buf.fill(0.0);
        k_buf.fill(0.0);
        total_quartets +=
            build_jk(ctx, prep, bounds, config.integral_thresh, &d, &mut j_buf, &mut k_buf)?;

        // F = H(+field) + J − ½K
        let f = &h + &j_buf - &(0.5 * &k_buf);

        let e_elec: f64 = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| 0.5 * d[(i, j)] * (h[(i, j)] + f[(i, j)]))
            .sum();
        let energy = e_elec + vnn;

        let fds = f.dot(&d).dot(&s);
        let sdf = s.dot(&d).dot(&f);
        let err = &fds - &sdf;
        let de = (energy - prev_e).abs();
        let err_max = err.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        if iter > 1 && de < config.energy_conv && err_max < config.density_conv {
            let (orb_e, c) = diag(&f)?;
            let density_alpha = 0.5 * &d;
            return Ok(ScfResult {
                spin: Spin::Restricted,
                energy,
                density_total: d,
                density_alpha,
                density_beta: None,
                mos_alpha: c,
                mos_beta: None,
                eps_alpha: orb_e,
                eps_beta: None,
                fock_alpha: f,
                fock_beta: None,
                converged: true,
                exit: ferric_scf::result::ScfExit::Converged,
                iterations: iter,
                computed_quartets: total_quartets,
            });
        }
        prev_e = energy;

        let f_new = diis.step(&f, &err);
        let (_, c) = diag(&f_new)?;
        let c_occ = c.slice(ndarray::s![.., ..nocc]);
        d = 2.0 * c_occ.dot(&c_occ.t());
    }
    Err(FerricError::ScfConvergence { iterations: config.max_iter, last_energy: prev_e })
}

/// Build the MP2 relaxed AO 1-PDM for an (already converged) RHF reference.
///
/// This is exactly the relaxed density the analytic RI-MP2 gradient uses:
/// SCF core (2δ_ij) + MP2 P_oo/P_vv + Z-vector occ-vir orbital relaxation.
fn mp2_relaxed_density_ao(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    config: &RiMp2Config,
    density_mode: DensityMode,
) -> Result<Array2<f64>, FerricError> {
    let inter = compute_mp2_intermediates(mol, obs, dfbs, op, rhf, config)?;
    let orb = inter.orbital_space();
    // Relaxed: build the CPHF Lagrangian RHS L via build_lagrangian, then re-solve
    // (Δε+A)z=L by CG (its Jacobi-DIIS analog in solve_zvector plateaus/diverges
    // here). Unrelaxed: z=0 (no orbital response → no near-singular 1/F mode for
    // symmetric molecules). NOTE: this uses the finite-field-α Lagrangian
    // (build_lagrangian), which is the CPHF RHS for the FF-α path specifically;
    // the production analytic gradient's solve_zvector uses the PySCF Xvo RHS and
    // returns the RI-MP2 Lagrangian matrix Imat instead — a different second
    // return value, hence the direct build_lagrangian call here.
    let z = match density_mode {
        DensityMode::Unrelaxed => Array2::<f64>::zeros((orb.nvir, orb.nocc)),
        DensityMode::Relaxed => {
            let c = rhf.mos_r();
            let f_mo = c.t().dot(rhf.fock_r()).dot(c);
            let b_full = crate::oo_rimp2::compute_b_full_mo(obs, dfbs, op, c)?;
            let l = build_lagrangian(&f_mo, &inter.t2, &inter.p_oo, &inter.p_vv, &orb, &b_full);
            solve_zvector_cg(c, &l, obs, bounds, &orb, rhf.eps_r())?
        }
    };
    Ok(build_relaxed_density_ao(
        rhf.mos_r(),
        &inter.p_oo,
        &inter.p_vv,
        &z,
        &orb,
    ))
}

/// DIAGNOSTIC: the MP2 relaxed μ_z for a single field F applied along z.
/// Returns the z-component of the dipole. Used to inspect whether μ(F) is smooth
/// in the field (real derivative) or non-analytic (state crossing). `field` may
/// be signed; it applies V = −field·z to Hcore.
#[allow(clippy::too_many_arguments)]
pub fn debug_perturbed_dipole_z(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    scf_config: &RhfConfig,
    mp2_config: &RiMp2Config,
    field: f64,
) -> Result<f64, FerricError> {
    let n = obs.nbasis();
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let mut v = Array2::<f64>::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            v[(mu, nu)] = -field * dip_ao[2][(mu, nu)]; // axis 2 = z
        }
    }
    let rhf = solve_rhf_with_external(ctx, mol, obs, bounds, scf_config, &v)?;
    let mode = if FF_UNRELAXED.toggle() {
        DensityMode::Unrelaxed
    } else {
        DensityMode::Relaxed
    };
    let p = mp2_relaxed_density_ao(mol, obs, dfbs, op, bounds, &rhf, mp2_config, mode)?;

    if ff_trace() {
        // Electron count of the relaxed density and of the SCF reference, plus the
        // SCF-only μ_z vs the full MP2-relaxed μ_z. Localizes a 1/F blow-up to the
        // SCF reference vs the MP2 correction.
        let s = oneelectron::overlap(obs);
        let n_mp2 = (&p * &s).sum();
        let d_scf = rhf.density_total();
        let n_scf = (d_scf * &s).sum();
        let mu_scf = mp2_dipole(mol, &dip_ao, d_scf)[2];
        let mu_mp2 = mp2_dipole(mol, &dip_ao, &p)[2];
        eprintln!(
            "    [ff] F={field:+.0e}  N_scf={n_scf:.6} N_mp2relax={n_mp2:.6}  μz_scf={mu_scf:+.6} μz_mp2={mu_mp2:+.6}"
        );
    }

    Ok(mp2_dipole(mol, &dip_ao, &p)[2])
}

/// SCF-only μ_z for a single z-field F (no MP2). HF-level finite-field reference
/// used to cross-check the analytic CPHF α (no Z-vector involved, so stable on
/// all molecules — the HF reference dipole is smooth in the field).
pub fn debug_scf_dipole_z(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    bounds: &SchwarzBounds,
    scf_config: &RhfConfig,
    field: f64,
) -> Result<f64, FerricError> {
    let n = obs.nbasis();
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let mut v = Array2::<f64>::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            v[(mu, nu)] = -field * dip_ao[2][(mu, nu)];
        }
    }
    let rhf = solve_rhf_with_external(ctx, mol, obs, bounds, scf_config, &v)?;
    Ok(mp2_dipole(mol, &dip_ao, rhf.density_total())[2])
}

/// SCF-only full dipole vector for a field of strength `field` along `axis`
/// (0=x,1=y,2=z). HF-level finite-field reference for the full α tensor.
pub fn debug_scf_dipole_axis(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    bounds: &SchwarzBounds,
    scf_config: &RhfConfig,
    axis: usize,
    field: f64,
) -> Result<[f64; 3], FerricError> {
    let n = obs.nbasis();
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;
    let mut v = Array2::<f64>::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            v[(mu, nu)] = -field * dip_ao[axis][(mu, nu)];
        }
    }
    let rhf = solve_rhf_with_external(ctx, mol, obs, bounds, scf_config, &v)?;
    Ok(mp2_dipole(mol, &dip_ao, rhf.density_total()))
}

/// Solve the Z-vector / CPHF response equation (Δε + A) z = L by conjugate
/// gradients. The orbital-Hessian response operator is symmetric
/// positive-definite for a stable closed-shell HF reference, so CG converges
/// monotonically — unlike the Jacobi-DIIS fixed-point in `solve_zvector`, which
/// plateaus or diverges for some systems (nh3, co2). Reuses the production
/// matvec `compute_az_product` (the A·z piece); the diagonal Δε·z is added here.
///
/// Returns z of shape (nvir, nocc), matching `solve_zvector`.
fn solve_zvector_cg(
    c: &Array2<f64>,
    l: &Array2<f64>,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    orb: &ferric_core::orbitals::OrbitalSpace,
    eps: &[f64],
) -> Result<Array2<f64>, FerricError> {
    use crate::zvector::compute_az_product;
    let ferric_core::orbitals::OrbitalSpace { nocc, nvir, nocc_total, first_occ } = *orb;

    // EnginePool is geometry/basis-only (density-independent) — build ONCE
    // here and reuse across every compute_az_product call in the CG loop
    // below (up to max_iter=100 calls), instead of build_jk constructing a
    // fresh pool per call. Reduction order is unchanged, so results stay
    // bit-identical across thread counts.
    let pool = ferric_scf::engine_pool::EnginePool::new(bounds.op, prep, 1e-14)?;

    // No config reachable on this finite-field path (no RiMp2Config in scope)
    // — hoist ONE resolve here (not per CG iteration) rather than
    // re-resolving inside the `apply` closure on every call.
    let budget_bytes = ferric_core::memory::resolve_budget_bytes(None);

    // Δε_{ai} table (the SPD diagonal; also the Jacobi preconditioner).
    let de = |a: usize, i: usize| eps[nocc_total + a] - eps[first_occ + i];

    // Full operator M·z = Δε⊙z + A·z.
    let apply = |z: &Array2<f64>| -> Result<Array2<f64>, FerricError> {
        let mut mz = compute_az_product(c, z, prep, bounds, orb, &pool, budget_bytes)?;
        for a in 0..nvir {
            for i in 0..nocc {
                mz[(a, i)] += de(a, i) * z[(a, i)];
            }
        }
        Ok(mz)
    };

    let dot = |x: &Array2<f64>, y: &Array2<f64>| -> f64 {
        let mut s = 0.0;
        for a in 0..nvir { for i in 0..nocc { s += x[(a, i)] * y[(a, i)]; } }
        s
    };

    // Jacobi-preconditioned CG. x0 = L/Δε (the uncoupled guess).
    let mut x = Array2::<f64>::zeros((nvir, nocc));
    for a in 0..nvir { for i in 0..nocc {
        let d = de(a, i);
        if d.abs() > 1e-12 { x[(a, i)] = l[(a, i)] / d; }
    }}

    // r = L − M·x ; z_pc = M_diag^{-1} r ; p = z_pc
    let mut r = l - &apply(&x)?;
    let precond = |r: &Array2<f64>| -> Array2<f64> {
        let mut z = Array2::<f64>::zeros((nvir, nocc));
        for a in 0..nvir { for i in 0..nocc {
            let d = de(a, i);
            if d.abs() > 1e-12 { z[(a, i)] = r[(a, i)] / d; }
        }}
        z
    };
    let mut z_pc = precond(&r);
    let mut p = z_pc.clone();
    let mut rz_old = dot(&r, &z_pc);

    let max_iter = 100;
    let tol = 1e-10; // tighter than the 1e-8 the FF dipole difference needs
    let trace = crate::zvector::zvec_trace();
    for it in 0..max_iter {
        let resid_max = r.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if trace { eprintln!("  [zvec-cg] iter={it:3}  max_resid={resid_max:.3e}"); }
        if resid_max < tol { break; }
        let mp = apply(&p)?;
        let denom = dot(&p, &mp);
        if denom.abs() < 1e-30 { break; }
        let alpha = rz_old / denom;
        for a in 0..nvir { for i in 0..nocc {
            x[(a, i)] += alpha * p[(a, i)];
            r[(a, i)] -= alpha * mp[(a, i)];
        }}
        z_pc = precond(&r);
        let rz_new = dot(&r, &z_pc);
        let beta = rz_new / rz_old;
        for a in 0..nvir { for i in 0..nocc {
            p[(a, i)] = z_pc[(a, i)] + beta * p[(a, i)];
        }}
        rz_old = rz_new;
    }
    Ok(x)
}

/// MP2 dipole `μ_d = −Tr[P_relax · r_d] + Σ_A Z_A R_{A,d}` (a.u.), electronic
/// term from the relaxed AO density, nuclear term from the geometry.
fn mp2_dipole(mol: &Molecule, dip_ao: &[Array2<f64>; 3], p_relax_ao: &Array2<f64>) -> [f64; 3] {
    let mut mu = [0.0_f64; 3];
    for d in 0..3 {
        // Electronic: −Tr[P r_d]. P_relax is the total electron density
        // (electrons carry negative charge → electronic dipole is −⟨r⟩).
        let elec = (p_relax_ao * &dip_ao[d]).sum();
        // Nuclear: +Σ_A Z_A R_{A,d}.
        let nuc: f64 = mol
            .atoms
            .iter()
            .map(|a| a.z as f64 * [a.x, a.y, a.zpos][d])
            .sum();
        mu[d] = nuc - elec;
    }
    mu
}

/// Finite-field MP2 static polarizability (closed-shell, orbital-relaxed).
///
/// `field_strength` is the dipole-field magnitude in a.u. (default 1e-3 is a
/// good balance of truncation vs SCF/round-off error). The base `rhf` is used
/// only for its config-independent geometry/basis; the field-perturbed
/// references are re-converged here.
///
/// `op` selects the two-electron operator for the MP2 correlation (pass
/// `Operator::coulomb()` for standard MP2, or an attenuated `erfc(ω)` operator
/// to test attenuated-MP2 α).
#[allow(clippy::too_many_arguments)]
pub fn mp2_polarizability_static(
    ctx: &ParallelContext,
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    scf_config: &RhfConfig,
    mp2_config: &RiMp2Config,
    field_strength: f64,
    density_mode: DensityMode,
) -> Result<Mp2Polarizability, FerricError> {
    let n = obs.nbasis();
    // AO dipole integrals ⟨μ|r_d|ν⟩ about the origin.
    let dip_ao = oneelectron::dipole(obs, [0.0, 0.0, 0.0])?;

    let h = field_strength;
    let mut tensor = [[0.0_f64; 3]; 3];

    // For each field direction j, central-difference μ_i.
    for j in 0..3 {
        // External one-electron potential V = −F_j · r_j added to Hcore.
        // The +F and −F runs use ±h along axis j.
        let mut v_plus = Array2::<f64>::zeros((n, n));
        let mut v_minus = Array2::<f64>::zeros((n, n));
        for mu in 0..n {
            for nu in 0..n {
                v_plus[(mu, nu)] = -h * dip_ao[j][(mu, nu)];
                v_minus[(mu, nu)] = h * dip_ao[j][(mu, nu)];
            }
        }

        let rhf_p = solve_rhf_with_external(ctx, mol, obs, bounds, scf_config, &v_plus)?;
        let rhf_m = solve_rhf_with_external(ctx, mol, obs, bounds, scf_config, &v_minus)?;

        let p_p = mp2_relaxed_density_ao(mol, obs, dfbs, op, bounds, &rhf_p, mp2_config, density_mode)?;
        let p_m = mp2_relaxed_density_ao(mol, obs, dfbs, op, bounds, &rhf_m, mp2_config, density_mode)?;

        let mu_p = mp2_dipole(mol, &dip_ao, &p_p);
        let mu_m = mp2_dipole(mol, &dip_ao, &p_m);

        // α_ij = −∂μ_i/∂F_j = −(μ_i(+h) − μ_i(−h)) / (2h)
        for i in 0..3 {
            tensor[i][j] = -(mu_p[i] - mu_m[i]) / (2.0 * h);
        }
    }

    // Symmetrize (finite-field asymmetry is O(h²) + round-off).
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg = 0.5 * (tensor[i][j] + tensor[j][i]);
            tensor[i][j] = avg;
            tensor[j][i] = avg;
        }
    }
    let iso = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0;
    let principal = eig3_sym(tensor);

    Ok(Mp2Polarizability { tensor, iso, principal })
}

/// Crate-visible alias so `cpks_polar` reuses the same 3×3 symmetric eig.
pub(crate) fn eig3_sym_pub(m: [[f64; 3]; 3]) -> [f64; 3] {
    eig3_sym(m)
}

/// Eigenvalues of a symmetric 3×3 matrix, ascending. Closed-form
/// (Smith / trigonometric) — no LAPACK dependency for a 3×3.
fn eig3_sym(m: [[f64; 3]; 3]) -> [f64; 3] {
    let a = Array2::from_shape_vec(
        (3, 3),
        vec![
            m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
        ],
    )
    .unwrap();
    let (evals, _) = a.eigh(UPLO::Upper).expect("3x3 symmetric eig");
    let mut out = [evals[0], evals[1], evals[2]];
    out.sort_by(|x, y| x.partial_cmp(y).unwrap());
    out
}
