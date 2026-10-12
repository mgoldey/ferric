//! The UHF Hessian assembly shared by the exact-integral and the
//! density-fitted (RI-J / RI-K) paths: [`uhf_hessian_parts_with`] builds the
//! five terms with RI-J on the TOTAL density, one RI-K channel per spin and the
//! SCF's own `DfJ` / `DfK` in the coupled CPHF where the SCF fitted them (the
//! derivation is in `crate::hessian_df_k`, "UHF: one channel per spin"). With
//! no route recorded it is the exact four-centre path, unchanged.

use super::*;

/// [`uhf_hessian_parts`] with deliberately broken density-fitting constructions
/// (mutation tests); the `Default`s are the production path.
#[allow(clippy::too_many_arguments)]
pub(super) fn uhf_hessian_parts_with(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    uhf: &ScfResult,
    config: &RhfConfig,
    rij_mutants: &RijMutants,
    rik_mutants: &RikMutants,
    uhf_mutants: &UhfDfMutants,
) -> Result<UhfHessianParts, FerricError> {
    check_supported(Reference::Uhf, mol, prep, op, bounds, uhf, config)?;
    let natoms = mol.atoms.len();
    let nocc = uhf_occupations(mol);
    let (da, db) = uhf_spin_densities(uhf)?;
    let d = da + db;
    let w = build_energy_weighted_density_uhf(uhf, nocc[0], nocc[1]);
    let route = crate::gradient::active_df_route(uhf);
    let route_j = route.filter(|r| r.j_aux.is_some());
    let route_k = route.filter(|r| r.k_aux.is_some());

    let extra = extra_charges(config);
    let nuclear = nuclear_term(mol, config);
    let one_electron =
        skeleton_hess_1e(prep, natoms, extra, &d)? + extra_skeleton(mol, prep, config, &d)?;
    let overlap = skeleton_hess_overlap(prep, natoms, &w)?;

    // RI-J: the Coulomb energy ½ dᵀV⁻¹d depends on the TOTAL density only, so
    // the closed-shell RI-J terms are used with D = D_α + D_β unchanged.
    let d_for_j = if uhf_mutants.j_from_alpha {
        da.clone()
    } else {
        d.clone()
    };
    let rij = route_j
        .map(|r| crate::hessian_df::rij_terms(mol, prep, r, &d_for_j, rij_mutants))
        .transpose()?;
    // RI-K: one channel per spin, E_K = −½ Σ_σ L^σ · M L^σ with X_σ = C_σ,occ.
    let xs = match route_k {
        Some(_) => Some(uhf_occupied_factors(uhf, nocc, [da, db])?),
        None => None,
    };
    let rik = match (route_k, &xs) {
        (Some(r), Some(xs)) => {
            let (scale, alpha) = if uhf_mutants.closed_shell_channel {
                (std::f64::consts::SQRT_2, crate::hessian_df_k::RIK_ALPHA)
            } else {
                (1.0, crate::hessian_df_k::RIK_ALPHA_SPIN)
            };
            let scaled: Vec<Array2<f64>> = xs.iter().map(|x| x * scale).collect();
            let n_channels = if uhf_mutants.drop_beta_channel { 1 } else { 2 };
            let channels: Vec<crate::hessian_df_k::RikChannel<'_>> = scaled
                .iter()
                .take(n_channels)
                .filter(|x| x.ncols() > 0)
                .map(|x| crate::hessian_df_k::RikChannel { x, alpha })
                .collect();
            Some(crate::hessian_df_k::rik_channel_terms(
                mol,
                prep,
                r,
                &channels,
                rik_mutants,
            )?)
        }
        _ => None,
    };
    // The RI-K terms released their reservation on return but their outputs
    // stay alive through the first-order Fock matrices and the CPHF: charge
    // that phase (a typed error when it does not fit the pool).
    let _cphf_charge = match route_k {
        Some(r) => {
            let name = r.k_aux.as_deref().unwrap_or_default();
            let naux = crate::df_gradient::aux_basis(mol, name)?.nbasis();
            let noccs: Vec<usize> = nocc.iter().copied().filter(|&m| m > 0).collect();
            Some(crate::hessian_df_k::reserve_cphf_phase(
                natoms,
                prep.nbasis(),
                &noccs,
                naux,
            )?)
        }
        None => None,
    };
    // A spin with no electrons has no channel; its K^x is zero.
    let channel_of_spin = |sp: usize| -> Option<usize> {
        if sp == 1 && uhf_mutants.drop_beta_channel {
            return None;
        }
        if nocc[sp] == 0 {
            return None;
        }
        Some(if sp == 1 && nocc[0] > 0 { 1 } else { 0 })
    };
    // Four-centre skeleton: only the parts of Γ that stayed exact.
    let max_d = max_abs(&d);
    let mut two_electron = match (&rij, &rik) {
        (None, None) => {
            let gam = |mu, nu, la, sg| gamma_uhf(&d, da, db, mu, nu, la, sg);
            skeleton_hess_2e(prep, op, bounds, natoms, max_d, &gam)?
        }
        (Some(_), None) => {
            let gam_k =
                |mu, nu, la, sg| -0.5 * (da[(mu, la)] * da[(nu, sg)] + db[(mu, la)] * db[(nu, sg)]);
            skeleton_hess_2e(prep, op, bounds, natoms, max_d, &gam_k)?
        }
        (None, Some(_)) => {
            let gam_j = |mu, nu, la, sg| 0.5 * d[(mu, nu)] * d[(la, sg)];
            skeleton_hess_2e(prep, op, bounds, natoms, max_d, &gam_j)?
        }
        (Some(_), Some(_)) => Array2::<f64>::zeros((3 * natoms, 3 * natoms)),
    };
    if let Some(rij) = &rij {
        two_electron += &rij.skeleton;
    }
    if let Some(rik) = &rik {
        two_electron += &rik.skeleton;
    }

    let no_density = Array2::<f64>::zeros(d.dim());
    let specs = [da, db].map(|ds| FockSpec {
        coulomb: if rij.is_some() { &no_density } else { &d },
        exchange: if rik.is_some() { &no_density } else { ds },
        k_scale: 1.0,
    });
    let mut first = first_order_ao(prep, op, bounds, natoms, extra, &specs)?;
    if let Some(rij) = &rij {
        for f in first.f1.iter_mut() {
            for (f, j) in f.iter_mut().zip(&rij.first_order_j) {
                *f += j;
            }
        }
    }
    if let Some(rik) = &rik {
        for sp in 0..2 {
            let src_spin = if uhf_mutants.swap_k_channels {
                1 - sp
            } else {
                sp
            };
            if let Some(ci) = channel_of_spin(src_spin) {
                for (f, k) in first.f1[sp]
                    .iter_mut()
                    .zip(&rik.channels[ci].first_order_fock)
                {
                    *f += k;
                }
            }
        }
    }
    add_extra_first_order(&mut first, mol, prep, config)?;
    let df_j = route_j.map(|r| df_j_builder(mol, prep, r)).transpose()?;
    let df_k = route_k
        .filter(|_| !uhf_mutants.cphf_exact_k)
        .map(|r| df_k_builder(mol, prep, r))
        .transpose()?;
    let resp = ucphf_response(
        ctx,
        prep,
        bounds,
        uhf,
        nocc,
        &first,
        (df_j, df_k),
        uhf_mutants,
    )?;
    Ok(parts_from_terms(
        [nuclear, one_electron, overlap, two_electron],
        resp,
    ))
}

/// The occupied factors `X_σ = C_σ,occ` of the two spin channels of the RI-K
/// Hessian, each checked to reproduce its spin density.
fn uhf_occupied_factors(
    uhf: &ScfResult,
    nocc: [usize; 2],
    dens: [&Array2<f64>; 2],
) -> Result<[Array2<f64>; 2], FerricError> {
    let cs = [uhf.mos_a(), uhf.mos_b()];
    let mut out = [Array2::<f64>::zeros((0, 0)), Array2::<f64>::zeros((0, 0))];
    for sp in 0..2 {
        let x = cs[sp].slice(s![.., ..nocc[sp]]).to_owned();
        if max_abs(&(x.dot(&x.t()) - dens[sp])) > OCC_CONSISTENCY_TOL {
            return Err(refusal(
                Reference::Uhf,
                "a UHF result whose spin density is not built from its first nocc MOs (RI-K \
                 needs the occupied factor)",
            ));
        }
        out[sp] = x;
    }
    Ok(out)
}
