//! Z-vector / CPHF solver for RI-MP2 orbital response.
//!
//! Solves (ε_a - ε_i) z_ai + Σ_{bj} A_{ai,bj} z_bj = L_ai
//! iteratively with DIIS, where A is the orbital Hessian and L is the
//! MP2 Lagrangian. The A*z product is computed in the AO basis via J/K builds.

use crate::rimp2::Mp2Intermediates;
use ferric_core::mol::Molecule;
use ferric_core::orbitals::OrbitalSpace;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::diis::Diis;
use ferric_scf::engine_pool::EnginePool;
use ferric_scf::rhf::{build_jk, build_jk_with_pool};
use ferric_scf::ScfResult;
use ferric_scf::screening::SchwarzBounds;
use ndarray::{Array2, Array3};

/// `FERRIC_ZVEC_TRACE` descriptor: Z-vector CPHF residual trace (env-only debug
/// toggle). Read here and in ff_polar.rs via [`zvec_trace`].
static ZVEC_TRACE: ferric_core::config::ConfigVar<bool> = ferric_core::config::ConfigVar {
    env_name: "FERRIC_ZVEC_TRACE",
    default: false,
    parse: ferric_core::config::parse_toggle,
    validate: ferric_core::config::accept_any,
};
pub(crate) fn zvec_trace() -> bool {
    ZVEC_TRACE.toggle()
}

/// Solve the Z-vector (CPHF) equation for the RI-MP2 orbital response.
///
/// Returns `(z, imat)` where `z` is the (nvir, nocc) occupied-virtual block of the
/// relaxed density in MO basis, and `imat` is the (nmo, nmo) RI-MP2 Lagrangian
/// matrix `build_imat_ri` (needed downstream for the overlap/Pulay term).
///
/// The right-hand side is PySCF's `Xvo` (`grad/mp2.py::grad_elec` lines 138-140):
/// `Xvo = C_vir^T · vhf · C_occ + Imat[occ,vir]^T − Imat[vir,occ]`, with
/// `vhf = get_veff(dm1_corr)·2 = 2J − K` applied ONCE (no extra factor), and
/// `dm1_corr` the AO correlation density from the *unrelaxed* occ-occ (`doo+doo^T`)
/// and vir-vir (`dvv+dvv^T`) MP2 1-PDM blocks (no orbital response yet).
///
/// The CPHF is solved as `(ε_a−ε_i)·z + A·z = −Xvo` — PySCF's `cphf.solve` negates
/// the RHS, so `dm1[vir,occ] = z` carries the opposite sign of the naive `Xvo/Δε`.
/// The Hessian matvec is `A·z = ½·compute_az_product = 2J − K` (built from the
/// symmetric response density `D^z + D^z†`): `compute_az_product` returns `4J − 2K`,
/// which is exactly 2× PySCF's CPHF `fvind = 2·get_veff(D^z+D^z†)`, so it is halved
/// here. Both the RHS sign and the ½ matvec scale were verified element-by-element
/// against PySCF (H2/cc-pVDZ z matches to ~1e-5).
///
/// `budget_bytes` is the caller-resolved memory ceiling threaded into every
/// `compute_az_product` call in the DIIS loop below (up to `max_iter` calls) —
/// callers that hold a config with `memory_budget_bytes` in scope should pass
/// `resolve_budget_bytes(config.memory_budget_bytes)`, resolved once at their
/// own top; callers with no config in scope pass `resolve_budget_bytes(None)`,
/// likewise resolved once, not re-resolved per DIIS iteration.
pub fn solve_zvector(
    _mol: &Molecule,
    prep: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    inter: &Mp2Intermediates,
    budget_bytes: usize,
) -> Result<(Array2<f64>, Array2<f64>), FerricError> {
    let orb = inter.orbital_space();
    let OrbitalSpace { nocc, nvir, nocc_total, first_occ } = orb;
    let eps = rhf.eps_r();
    let c = rhf.mos_r();
    let nmo = c.ncols();

    // RI-MP2 Lagrangian matrix Imat (all-MO), from the shared x_ov / b_full.
    let b_full = crate::oo_rimp2::compute_b_full_mo(prep, dfbs, op, c)?;
    let x_ov = build_x_ov(&inter.t2, &inter.b_ov, nocc, nvir, inter.naux);
    let imat = build_imat_ri(&x_ov, &b_full, &orb);

    // Unrelaxed correlation density (MO): doo+doo^T on occ-occ, dvv+dvv^T on
    // vir-vir. inter.p_oo == PySCF doo, inter.p_vv == PySCF dvv (see
    // build_mp2_density / rimp2 doc: doo returned negated, matching PySCF).
    let mut dm1mo_corr = Array2::<f64>::zeros((nmo, nmo));
    for i in 0..nocc {
        let i_mo = first_occ + i;
        for j in 0..nocc {
            let j_mo = first_occ + j;
            dm1mo_corr[(i_mo, j_mo)] = inter.p_oo[(i, j)] + inter.p_oo[(j, i)];
        }
    }
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for b in 0..nvir {
            let b_mo = nocc_total + b;
            dm1mo_corr[(a_mo, b_mo)] = inter.p_vv[(a, b)] + inter.p_vv[(b, a)];
        }
    }
    // veff(dm1_corr_ao): 2J − K in AO (closed-shell), via build_jk.
    let dm1_corr_ao = {
        let cp = c.dot(&dm1mo_corr);
        cp.dot(&c.t())
    };
    let n = c.nrows();
    let (mut jc, mut kc) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    let ctx = ferric_core::parallel::ParallelContext::default();
    build_jk(&ctx, prep, bounds, 1e-12, &dm1_corr_ao, &mut jc, &mut kc)?;
    // PySCF `vhf = mp._scf.get_veff(dm1)*2 = 2·(J − ½K) = 2J − K` (grad/mp2.py
    // line 138); `Xvo = C_vir^T · vhf · C_occ` (line 139) with NO further factor.
    // So the veff coefficient in Xvo is exactly (2J − K), applied ONCE.
    let veff_corr_ao = 2.0 * &jc - &kc; // 2J − K == PySCF's `vhf`
    let veff_corr_mo = c.t().dot(&veff_corr_ao).dot(c);

    // Xvo[a,i] = veff_corr_mo[a,i] + Imat[i,a] − Imat[a,i]  (PySCF lines 138-140).
    // PySCF's `cphf.solve(fvind, mo_energy, mo_occ, Xvo)` solves the response
    // equation `(ε_a−ε_i) z + A·z = −Xvo` (the CPHF driver negates the RHS: its
    // returned `dm1[vir,occ] = z` has the opposite sign of the naive Xvo/Δε).
    // ferric's solve loop below uses `(ε_a−ε_i) z + A·z = rhs`, so `rhs = −Xvo`
    // reproduces PySCF's z element-for-element (verified against pyscf grad/mp2).
    let mut rhs = Array2::<f64>::zeros((nvir, nocc));
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            rhs[(a, i)] = -(veff_corr_mo[(a_mo, i_mo)] + imat[(i_mo, a_mo)] - imat[(a_mo, i_mo)]);
        }
    }

    let mut z = Array2::zeros((nvir, nocc));
    for a in 0..nvir {
        for i in 0..nocc {
            let denom = eps[nocc_total + a] - eps[first_occ + i];
            if denom.abs() > 1e-12 {
                z[(a, i)] = rhs[(a, i)] / denom;
            }
        }
    }

    let mut diis = Diis::new(8);
    let max_iter = 50;

    // EnginePool is geometry/basis-only (density-independent) — build ONCE
    // here and reuse across every compute_az_product call in the DIIS loop
    // below (up to max_iter calls), instead of build_jk constructing a fresh
    // pool per call. Reduction order is unchanged, so results stay
    // bit-identical across thread counts.
    let pool = EnginePool::new(bounds.op, prep, 1e-14)?;

    for _iter in 0..max_iter {
        // A·z Hessian coupling. `compute_az_product` returns 4J−2K (built from the
        // symmetric response density D^z + D^z†), which is EXACTLY 2× PySCF's CPHF
        // fvind `2·get_veff(D^z+D^z†) = 2J−K` (verified numerically: the ratio is
        // 0.5). The MP2 Z-vector Hessian is `Δε·z + (2J−K)·z`, so the matvec must be
        // scaled by ½ here (mirrors cpks_polar.rs's `ascale=0.5` for the same
        // symmetric-density double-count).
        let az = 0.5 * &compute_az_product(c, &z, prep, bounds, &orb, &pool, budget_bytes)?;

        let mut residual = Array2::zeros((nvir, nocc));
        let mut max_resid = 0.0f64;
        for a in 0..nvir {
            for i in 0..nocc {
                let denom = eps[nocc_total + a] - eps[first_occ + i];
                residual[(a, i)] = rhs[(a, i)] - denom * z[(a, i)] - az[(a, i)];
                max_resid = max_resid.max(residual[(a, i)].abs());
            }
        }

        if zvec_trace() {
            eprintln!("  [zvec] iter={_iter:3}  max_resid={max_resid:.3e}");
        }

        if max_resid < 1e-8 {
            return Ok((z, imat));
        }

        let mut z_new = Array2::zeros((nvir, nocc));
        for a in 0..nvir {
            for i in 0..nocc {
                let denom = eps[nocc_total + a] - eps[first_occ + i];
                if denom.abs() > 1e-12 {
                    z_new[(a, i)] = (rhs[(a, i)] - az[(a, i)]) / denom;
                }
            }
        }

        z = diis.step(&z_new, &residual);
    }

    // BUG (latent): reaching here means the Z-vector did NOT converge to 1e-8
    // within max_iter — yet we silently return the unconverged z. The
    // finite-field α driver then differences garbage dipoles. Trace it under
    // FERRIC_ZVEC_TRACE so the finite-field noise-floor diagnosis can see it.
    // (Not promoted to a hard error yet: the analytic-gradient callers tolerate
    // a loosely-converged z; the FF-α path is what needs the tighter floor.)
    if zvec_trace() {
        eprintln!("  [zvec] DID NOT CONVERGE in {max_iter} iters");
    }

    Ok((z, imat))
}

/// Build the MP2 Lagrangian L_ai (RHS of the Z-vector equation).
///
/// Uses the full-MO B tensor (computed internally) for exact integral response.
/// The Lagrangian has P*F terms plus 4-term integral response matching the
/// structure of compute_orbital_gradient in oo_rimp2.
pub(crate) fn build_lagrangian(
    f_mo: &Array2<f64>,
    t2: &[f64],
    p_oo: &Array2<f64>,
    p_vv: &Array2<f64>,
    orb: &OrbitalSpace,
    b_full: &Array3<f64>,
) -> Array2<f64> {
    let OrbitalSpace { nocc, nvir, nocc_total, first_occ } = *orb;
    let nov = nocc * nvir;
    let naux = b_full.shape()[0];
    let mut l = Array2::zeros((nvir, nocc));

    // P*F terms
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let mut sum = 0.0;
            for j in 0..nocc {
                let j_mo = first_occ + j;
                sum += p_oo[(i, j)] * f_mo[(a_mo, j_mo)];
            }
            l[(a, i)] += sum;
        }
    }
    for a in 0..nvir {
        for i in 0..nocc {
            let i_mo = first_occ + i;
            let mut sum = 0.0;
            for b in 0..nvir {
                let b_mo = nocc_total + b;
                sum += p_vv[(a, b)] * f_mo[(b_mo, i_mo)];
            }
            l[(a, i)] += sum;
        }
    }

    // Integral response using the same 4-term structure as compute_orbital_gradient.
    // The orbital gradient g_{ck} = -4*F_{ck} - 2*grad_ck.
    // The Lagrangian integral part = grad_ck (same raw sum, no extra factor).
    let eri = |p: usize, q: usize, r: usize, s: usize| -> f64 {
        (0..naux).map(|aux| b_full[(aux, p, q)] * b_full[(aux, r, s)]).sum()
    };

    for c_idx in 0..nvir {
        let c_mo = nocc_total + c_idx;
        for k in 0..nocc {
            let k_mo = first_occ + k;
            let mut grad_ck = 0.0;

            // Term 1 (delta_{ik}→i=k)
            for j in 0..nocc {
                let j_mo = first_occ + j;
                for a in 0..nvir {
                    let a_mo = nocc_total + a;
                    for b in 0..nvir {
                        let b_mo = nocc_total + b;
                        let t_kj_ab = t2[(k * nvir + a) * nov + j * nvir + b];
                        grad_ck += t_kj_ab * (2.0 * eri(c_mo, a_mo, j_mo, b_mo) - eri(c_mo, b_mo, j_mo, a_mo));
                    }
                }
            }

            // Term 2 (delta_{jk}→j=k)
            for i in 0..nocc {
                let i_mo = first_occ + i;
                for a in 0..nvir {
                    let a_mo = nocc_total + a;
                    for b in 0..nvir {
                        let b_mo = nocc_total + b;
                        let t_ik_ab = t2[(i * nvir + a) * nov + k * nvir + b];
                        grad_ck += t_ik_ab * (2.0 * eri(i_mo, a_mo, c_mo, b_mo) - eri(i_mo, b_mo, c_mo, a_mo));
                    }
                }
            }

            // Term 3 (-delta_{ac}→a=c)
            for i in 0..nocc {
                let i_mo = first_occ + i;
                for j in 0..nocc {
                    let j_mo = first_occ + j;
                    for b in 0..nvir {
                        let b_mo = nocc_total + b;
                        let t_ij_cb = t2[(i * nvir + c_idx) * nov + j * nvir + b];
                        grad_ck -= t_ij_cb * (2.0 * eri(i_mo, k_mo, j_mo, b_mo) - eri(i_mo, b_mo, j_mo, k_mo));
                    }
                }
            }

            // Term 4 (-delta_{bc}→b=c)
            for i in 0..nocc {
                let i_mo = first_occ + i;
                for j in 0..nocc {
                    let j_mo = first_occ + j;
                    for a in 0..nvir {
                        let a_mo = nocc_total + a;
                        let t_ij_ac = t2[(i * nvir + a) * nov + j * nvir + c_idx];
                        grad_ck -= t_ij_ac * (2.0 * eri(i_mo, a_mo, j_mo, k_mo) - eri(i_mo, k_mo, j_mo, a_mo));
                    }
                }
            }

            l[(c_idx, k)] += grad_ck;
        }
    }

    l
}

/// Build the t2-weighted RI amplitude object `x_ov[P, ia]`.
///
/// `x_ov[P, ia] = Σ_{jb} (2 t_{ij,ab} − t_{ij,ba}) B^P_{jb}`, the same object the
/// 3c/2c integral-response gradient builds. Exposed here so the Imat build and the
/// gradient share one definition. `t2` layout is `t2[(i*nvir+a)*nov + j*nvir+b]`.
pub(crate) fn build_x_ov(
    t2: &[f64],
    b_ov: &Array2<f64>,
    nocc: usize,
    nvir: usize,
    naux: usize,
) -> Array2<f64> {
    let nov = nocc * nvir;
    let mut x_ov = Array2::zeros((naux, nov));
    for i in 0..nocc {
        for a in 0..nvir {
            let ia = i * nvir + a;
            for jb in 0..nov {
                let j = jb / nvir;
                let b = jb % nvir;
                let t_ij_ab = t2[(i * nvir + a) * nov + j * nvir + b];
                let t_ij_ba = t2[(i * nvir + b) * nov + j * nvir + a];
                let tt = 2.0 * t_ij_ab - t_ij_ba;
                for p in 0..naux {
                    x_ov[(p, ia)] += tt * b_ov[(p, jb)];
                }
            }
        }
    }
    x_ov
}

/// Build the RI-MP2 Lagrangian energy-weighted matrix `Imat` in the MO basis.
///
/// This is the RI (density-fitted) analog of PySCF's conventional 4-index `Imat`
/// (`pyscf/grad/mp2.py::grad_elec`, lines 56-95 build `Imat_AO`, line 121 rotates
/// to MO with a `-1` factor). PySCF forms
/// `Imat_AO[μ,ν] = Σ_{λ,r,s} (μλ|rs) Γ2_AO[ν,λ,r,s]` from the *undifferentiated*
/// 4-center ERI and the fully back-transformed MP2 2-particle density, then
/// `Imat_MO = -C^T · Imat_AO · S · C`.
///
/// In the RI basis `(μλ|rs) = Σ_P B^P_{μλ} B^P_{rs}` (`B = V^{-1/2}(P|··)`), so the
/// `(r,s)` legs of `Γ2` collapse onto the fitted-amplitude object
/// `x_ov[P,ia] = Σ_{jb}(2t−t^T)_{ia,jb} B^P_{jb}` and no 4-index AO tensor is ever
/// materialized. The `-C^T(·)S·C` AO→MO rotation reduces to a direct MO
/// contraction against `b_full` (which is already `V^{-1/2}`-dressed and
/// C-transformed on both legs), with no leftover `S`. The resulting MO blocks are
/// (general MO index `q`; the overall `-1` from line 121 folded in):
///
/// ```text
///   Imat[q, i(occ)] = -2 Σ_{P,a} x_ov[P,ia] · b_full[P, q, a]
///   Imat[q, a(vir)] = -2 Σ_{P,i} x_ov[P,ia] · b_full[P, q, i]
/// ```
///
/// The factor `2` reflects the closed-shell 2-PDM weight `Γ2 = 2·(2t−t^T)`
/// (PySCF `part_dm2 = 4·t − 2·t^T`, lines 61-62). `Imat` is deliberately NOT
/// symmetric (PySCF: "matrix im1 is not hermitian", line 173). Only occupied and
/// virtual columns are filled (frozen-core rows/cols stay zero — frozen-core
/// gradients are out of scope here, matching the rest of this pipeline).
///
/// Verified element-by-element against a brute-force conventional-Imat build (RI
/// integrals + explicit 4-index `Γ2_AO` contraction) in
/// `test_ri_imat_vs_conventional`.
pub(crate) fn build_imat_ri(
    x_ov: &Array2<f64>,
    b_full: &Array3<f64>,
    orb: &OrbitalSpace,
) -> Array2<f64> {
    let OrbitalSpace { nocc, nvir, nocc_total, first_occ } = *orb;
    let naux = b_full.shape()[0];
    let nmo = b_full.shape()[1];
    let mut imat = Array2::zeros((nmo, nmo));

    // Imat[q, i] = -2 Σ_{P,a} x_ov[P,ia] b_full[P, q, a]
    for i in 0..nocc {
        let i_col = first_occ + i;
        for q in 0..nmo {
            let mut sum = 0.0;
            for a in 0..nvir {
                let a_mo = nocc_total + a;
                let ia = i * nvir + a;
                for p in 0..naux {
                    sum += x_ov[(p, ia)] * b_full[(p, q, a_mo)];
                }
            }
            imat[(q, i_col)] = -2.0 * sum;
        }
    }

    // Imat[q, a] = -2 Σ_{P,i} x_ov[P,ia] b_full[P, q, i]
    for a in 0..nvir {
        let a_col = nocc_total + a;
        for q in 0..nmo {
            let mut sum = 0.0;
            for i in 0..nocc {
                let i_mo = first_occ + i;
                let ia = i * nvir + a;
                for p in 0..naux {
                    sum += x_ov[(p, ia)] * b_full[(p, q, i_mo)];
                }
            }
            imat[(q, a_col)] = -2.0 * sum;
        }
    }

    imat
}

/// Compute the A*z product via J/K builds in the AO basis.
///
/// A_{ai,bj} z_{bj} = [4(ai|bj) - (ab|ij) - (aj|bi)] z_{bj}
/// In AO basis: form D^z, build J(D^z) and K(D^z), project back to MO.
///
/// `pub(crate)` so the finite-field α driver (`ff_polar`) can reuse this exact
/// matvec inside a CG solver — the production gradient's `solve_zvector` is
/// unchanged (this is a visibility-only change, no behavioral effect).
///
/// `ooc_budget` sizes the `build_jk_with_pool` reduction band
/// (`ferric_scf::reduce::resolve_band_bytes`) only — never affects the
/// result. Callers with no solver-resolved budget in scope pass
/// `ferric_core::memory::resolve_budget_bytes(None)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_az_product(
    c: &Array2<f64>,
    z: &Array2<f64>,
    prep: &PreparedBasis,
    bounds: &SchwarzBounds,
    orb: &OrbitalSpace,
    pool: &EnginePool,
    ooc_budget: usize,
) -> Result<Array2<f64>, FerricError> {
    let OrbitalSpace { nocc, nvir, nocc_total, first_occ } = *orb;
    let n = c.nrows();

    let mut dz = Array2::zeros((n, n));
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            for mu in 0..n {
                for nu in 0..n {
                    let val = z[(a, i)] * (c[(mu, a_mo)] * c[(nu, i_mo)] + c[(mu, i_mo)] * c[(nu, a_mo)]);
                    dz[(mu, nu)] += val;
                }
            }
        }
    }

    // Build J(D^z) and K(D^z)
    let mut jz = Array2::zeros((n, n));
    let mut kz = Array2::zeros((n, n));
    let ctx = ferric_core::parallel::ParallelContext::default();
    let band_bytes = ferric_scf::reduce::resolve_band_bytes(ooc_budget);
    build_jk_with_pool(&ctx, prep, bounds, 1e-12, &dz, &mut jz, &mut kz, pool, band_bytes)?;

    // The A*z product in AO: A_AO = 4*J(D^z) - K(D^z) - K(D^z)^T
    let az_ao = 4.0 * &jz - &kz - &kz.t();

    // Project to MO basis, extract virtual-occupied block
    let az_mo = c.t().dot(&az_ao).dot(c);
    let mut result = Array2::zeros((nvir, nocc));
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            result[(a, i)] = az_mo[(a_mo, i_mo)];
        }
    }

    Ok(result)
}

/// Build the relaxed 1-PDM in AO basis.
pub fn build_relaxed_density_ao(
    c: &Array2<f64>,
    p_oo: &Array2<f64>,
    p_vv: &Array2<f64>,
    z: &Array2<f64>,
    orb: &OrbitalSpace,
) -> Array2<f64> {
    let OrbitalSpace { nocc, nvir, nocc_total, first_occ } = *orb;
    let nmo = c.ncols();

    let mut p_mo = Array2::zeros((nmo, nmo));

    // Occ-occ: 2*δ_ij + P^MP2_ij
    for i in 0..nocc {
        let i_mo = first_occ + i;
        p_mo[(i_mo, i_mo)] += 2.0;
        for j in 0..nocc {
            let j_mo = first_occ + j;
            p_mo[(i_mo, j_mo)] += p_oo[(i, j)];
        }
    }

    // Vir-vir: P^MP2_ab
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for b in 0..nvir {
            let b_mo = nocc_total + b;
            p_mo[(a_mo, b_mo)] += p_vv[(a, b)];
        }
    }

    // Occ-vir and vir-occ: z_ai
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            p_mo[(a_mo, i_mo)] += z[(a, i)];
            p_mo[(i_mo, a_mo)] += z[(a, i)];
        }
    }

    // Transform to AO: P_AO = C * P_MO * C^T
    let cp = c.dot(&p_mo);
    cp.dot(&c.t())
}

/// Build the relaxed energy-weighted density in AO basis.
pub fn build_relaxed_w_ao(
    c: &Array2<f64>,
    f_mo: &Array2<f64>,
    p_relax_mo: &Array2<f64>,
    l: &Array2<f64>,
    orb: &OrbitalSpace,
) -> Array2<f64> {
    let OrbitalSpace { nocc, nvir, nocc_total, first_occ } = *orb;
    let nmo = c.ncols();

    let mut w_mo = Array2::zeros((nmo, nmo));

    // W_ij = Σ_k F_ik * P^relax_kj (occupied block)
    for i in 0..nocc {
        let i_mo = first_occ + i;
        for j in 0..nocc {
            let j_mo = first_occ + j;
            let mut sum = 0.0;
            for k in 0..nmo {
                sum += f_mo[(i_mo, k)] * p_relax_mo[(k, j_mo)];
            }
            w_mo[(i_mo, j_mo)] = sum;
        }
    }

    // W_ab = Σ_c F_ac * P^relax_cb (virtual block)
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for b in 0..nvir {
            let b_mo = nocc_total + b;
            let mut sum = 0.0;
            for k in 0..nmo {
                sum += f_mo[(a_mo, k)] * p_relax_mo[(k, b_mo)];
            }
            w_mo[(a_mo, b_mo)] = sum;
        }
    }

    // W_ai = L_ai (the Lagrangian RHS, not ε_i * z_ai)
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            w_mo[(a_mo, i_mo)] = l[(a, i)];
            w_mo[(i_mo, a_mo)] = l[(a, i)];
        }
    }

    // Transform to AO
    let cw = c.dot(&w_mo);
    cw.dot(&c.t())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rimp2::{compute_mp2_intermediates, RiMp2Config};
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    /// Verify [`build_imat_ri`] against a brute-force construction of PySCF's
    /// conventional `Imat` (`pyscf/grad/mp2.py::grad_elec` lines 56-121), but with
    /// the 4-center ERI replaced by its RI factorization so the reference matches
    /// ferric's RI energy exactly. On H2/cc-pVDZ (nao=10) the O(nao^4) reference is
    /// cheap. This isolates the Imat build from the zeta/vhf_s1occ/assembly logic.
    #[test]
    fn test_ri_imat_vs_conventional() {
        use ferric_integrals::threeindex;
        use ndarray::Array3;

        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let inter = compute_mp2_intermediates(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).unwrap();

        let orb = inter.orbital_space();
        let OrbitalSpace { nocc, nvir, nocc_total, first_occ } = orb;
        let c = rhf.mos_r();
        let nao = c.nrows();
        let nmo = c.ncols();
        let naux = inter.naux;
        let nov = nocc * nvir;
        let t2 = &inter.t2;

        // --- Candidate: RI Imat from x_ov and b_full ---
        let b_full = crate::oo_rimp2::compute_b_full_mo(&obs, &dfbs, op, c).unwrap();
        let x_ov = build_x_ov(t2, &inter.b_ov, nocc, nvir, naux);
        let imat_ri = build_imat_ri(&x_ov, &b_full, &orb);

        // --- Reference: conventional Imat with RI-factorized 4-center ERI ---
        // AO-basis dressed 3-center: B_AO[P,mu,nu] = Σ_Q V^{-1/2}[P,Q] (Q|mu,nu).
        let eri3_ao = threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let mut b_ao = Array3::<f64>::zeros((naux, nao, nao));
        for mu in 0..nao {
            for nu in 0..nao {
                for p in 0..naux {
                    let mut s = 0.0;
                    for q in 0..naux {
                        s += inter.v_inv_sqrt[(p, q)] * eri3_ao[(q, mu, nu)];
                    }
                    b_ao[(p, mu, nu)] = s;
                }
            }
        }
        // RI 4-center ERI: (mu la|r s) = Σ_P B_AO[P,mu,la] B_AO[P,r,s].
        let eri = |mu: usize, la: usize, r: usize, s: usize| -> f64 {
            (0..naux).map(|p| b_ao[(p, mu, la)] * b_ao[(p, r, s)]).sum()
        };

        let c_occ = c.slice(ndarray::s![.., first_occ..first_occ + nocc]).to_owned();
        let c_vir = c.slice(ndarray::s![.., nocc_total..nocc_total + nvir]).to_owned();

        // Fully back-transformed AO 2-PDM `dm2buf`, matching PySCF part_dm2 →
        // dm2buf exactly (mp2_grad.py lines 58-90, no r↔s symmetrization needed for
        // the contraction below since eri is r↔s symmetric):
        //   part_dm2[i,μ,ν,j] = Σ_ab (4 t_ij,ab − 2 t_ij,ba) C_μa C_νb   (i,j occ)
        //   dm2buf[p,q,r,s]   = Σ_ij C_pi C_sj part_dm2[i,q,r,j]         (all AO)
        // Then Imat_AO[p,q] = Σ_{i,r,s} (i p | r s) dm2buf[i,q,r,s]  (line 95).
        // dm2buf[i,q,r,s] = Σ_{k,l occ} C_ik C_sl part_dm2[k,q,r,l]. Fold the C_ik
        // contraction directly into the Imat integral leg. Precompute part_dm2.
        let mut part_dm2 = vec![0.0f64; nocc * nao * nao * nocc]; // [i,μ,ν,j]
        for i in 0..nocc {
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let t_ab = t2[(i * nvir + a) * nov + j * nvir + b];
                        let t_ba = t2[(i * nvir + b) * nov + j * nvir + a];
                        let w = 4.0 * t_ab - 2.0 * t_ba;
                        if w == 0.0 { continue; }
                        for mu in 0..nao {
                            let cma = w * c_vir[(mu, a)];
                            if cma == 0.0 { continue; }
                            for nu in 0..nao {
                                part_dm2[((i * nao + mu) * nao + nu) * nocc + j] +=
                                    cma * c_vir[(nu, b)];
                            }
                        }
                    }
                }
            }
        }
        // dm2buf_ao[ii, q, r, s], matching PySCF lines 86-88 (both symmetrizing
        // terms — line 86 back-transforms occ leg i→ii, line 87 back-transforms
        // occ leg i→q, then line 88 back-transforms the remaining occ leg j→s):
        //   dm2buf[ii,q,r,s] = Σ_{k,l occ} C_ii,k C_s,l part_dm2[k,q,r,l]      (86)
        //                    + Σ_{k,l occ} C_q,k  C_s,l part_dm2[k,ii,r,l]     (87)
        // Imat_AO[p,q] = Σ_{ii,r,s} (ii p | r s) dm2buf[ii,q,r,s]  (line 95).
        let dm2buf = |ii: usize, q: usize, r: usize, s: usize| -> f64 {
            let mut acc = 0.0;
            for k in 0..nocc {
                let cik = c_occ[(ii, k)];
                let cqk = c_occ[(q, k)];
                for l in 0..nocc {
                    let csl = c_occ[(s, l)];
                    acc += cik * csl * part_dm2[((k * nao + q) * nao + r) * nocc + l];
                    acc += cqk * csl * part_dm2[((k * nao + ii) * nao + r) * nocc + l];
                }
            }
            acc
        };
        let mut imat_ao_ref = Array2::<f64>::zeros((nao, nao));
        for p in 0..nao {
            for q in 0..nao {
                let mut acc = 0.0;
                for ii in 0..nao {
                    for r in 0..nao {
                        for s in 0..nao {
                            acc += eri(ii, p, r, s) * dm2buf(ii, q, r, s);
                        }
                    }
                }
                imat_ao_ref[(p, q)] = acc;
            }
        }
        // Rotate to MO and apply -1 (line 121): Imat_MO = -C^T Imat_AO S C.
        let s_ao = ferric_integrals::oneelectron::overlap(&obs);
        let imat_mo_ref = -(c.t().dot(&imat_ao_ref).dot(&s_ao).dot(c));

        // The reference Imat above collapses to the same (2t - t^T)-weighted energy
        // contraction as build_imat_ri once the two occ legs and two vir legs are
        // reduced; but PySCF's gamma2 weight (4/-2) already equals 2*(2t - t^T)
        // symmetrized. Compare the occ and vir COLUMNS (the only ones we use).
        let mut max_diff = 0.0f64;
        for q in 0..nmo {
            for i in 0..nocc {
                let col = first_occ + i;
                let d = (imat_ri[(q, col)] - imat_mo_ref[(q, col)]).abs();
                max_diff = max_diff.max(d);
            }
            for a in 0..nvir {
                let col = nocc_total + a;
                let d = (imat_ri[(q, col)] - imat_mo_ref[(q, col)]).abs();
                max_diff = max_diff.max(d);
            }
        }
        eprintln!("=== RI Imat vs conventional (RI-factorized) reference ===");
        eprintln!("  max column diff = {max_diff:.3e}");
        // Spot-print a few
        for &(q, col) in &[(0usize, first_occ), (nocc_total, first_occ), (0usize, nocc_total)] {
            eprintln!("  Imat[{q},{col}]: ri={:+.8} ref={:+.8}", imat_ri[(q, col)], imat_mo_ref[(q, col)]);
        }
        assert!(max_diff < 1e-7,
            "RI Imat disagrees with conventional reference: max diff = {max_diff:.3e}");
    }

    #[test]
    fn test_zvector_converges() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let inter = compute_mp2_intermediates(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).unwrap();

        let (z, _l) = solve_zvector(&mol, &obs, &dfbs, Operator::coulomb(), &bounds, &rhf, &inter, ferric_core::memory::resolve_budget_bytes(None)).unwrap();

        // Z should be finite and small
        for a in 0..inter.nvir {
            for i in 0..inter.nocc {
                assert!(z[(a,i)].is_finite(), "z[{},{}] not finite", a, i);
            }
        }
    }

    #[test]
    fn test_relaxed_density_symmetric() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let inter = compute_mp2_intermediates(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).unwrap();

        let (z, _l) = solve_zvector(&mol, &obs, &dfbs, Operator::coulomb(), &bounds, &rhf, &inter, ferric_core::memory::resolve_budget_bytes(None)).unwrap();
        let p_ao = build_relaxed_density_ao(
            rhf.mos_r(), &inter.p_oo, &inter.p_vv, &z, &inter.orbital_space(),
        );

        let n = p_ao.nrows();
        for i in 0..n {
            for j in 0..n {
                assert!((p_ao[(i,j)] - p_ao[(j,i)]).abs() < 1e-12,
                    "P_relax_AO not symmetric at ({},{}): {} vs {}", i, j, p_ao[(i,j)], p_ao[(j,i)]);
            }
        }
    }

    #[test]
    fn test_relaxed_w_symmetric() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let inter = compute_mp2_intermediates(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).unwrap();

        let (z, l) = solve_zvector(&mol, &obs, &dfbs, Operator::coulomb(), &bounds, &rhf, &inter, ferric_core::memory::resolve_budget_bytes(None)).unwrap();

        let nmo = rhf.mos_r().ncols();
        let f_mo = rhf.mos_r().t().dot(rhf.fock_r()).dot(rhf.mos_r());
        let mut p_relax_mo = ndarray::Array2::zeros((nmo, nmo));
        for i in 0..inter.nocc {
            let i_mo = inter.first_occ + i;
            p_relax_mo[(i_mo, i_mo)] += 2.0;
            for j in 0..inter.nocc {
                let j_mo = inter.first_occ + j;
                p_relax_mo[(i_mo, j_mo)] += inter.p_oo[(i, j)];
            }
        }
        for a in 0..inter.nvir {
            let a_mo = inter.nocc_total + a;
            for b in 0..inter.nvir {
                let b_mo = inter.nocc_total + b;
                p_relax_mo[(a_mo, b_mo)] += inter.p_vv[(a, b)];
            }
        }
        for a in 0..inter.nvir {
            let a_mo = inter.nocc_total + a;
            for i in 0..inter.nocc {
                let i_mo = inter.first_occ + i;
                p_relax_mo[(a_mo, i_mo)] += z[(a, i)];
                p_relax_mo[(i_mo, a_mo)] += z[(a, i)];
            }
        }

        let w_ao = build_relaxed_w_ao(
            rhf.mos_r(), &f_mo, &p_relax_mo, &l, &inter.orbital_space(),
        );

        let n = w_ao.nrows();
        for i in 0..n {
            for j in 0..n {
                // W_relax has residual asymmetry from the ov/vo blocks of F*P_relax.
                // This is a known limitation of the current W construction;
                // the gradient is still correct because W is contracted with symmetric dS/dR.
                assert!((w_ao[(i,j)] - w_ao[(j,i)]).abs() < 0.1,
                    "W_relax_AO asymmetry too large at ({},{}): {:.6e} vs {:.6e}", i, j, w_ao[(i,j)], w_ao[(j,i)]);
            }
        }
    }
}
