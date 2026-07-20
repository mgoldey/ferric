//! RI-MP2 nuclear gradients: analytical and finite-difference reference.

use crate::rimp2::{ri_mp2, compute_mp2_intermediates_ov_only, RiMp2Config};
use crate::zvector::solve_zvector;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::ScfResult;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Compute the total RI-MP2 energy (E_HF + E_MP2) for a given geometry.
fn total_energy(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
) -> Result<f64, FerricError> {
    let obs = PreparedBasis::new(mol, obs_basis)?;
    let bounds = SchwarzBounds::compute(op, &obs)?;
    // Tight SCF so finite-difference gradients are not noise-limited. Under the
    // ΔP convergence gate the *reachable* tight signal is density_conv (the
    // density drains cleanly to ~1e-9); energy_conv is only a loose
    // "not-descending" bound and floors above 1e-10 under DF noise, so setting it
    // tight would just hang the SCF at MaxIter (see rhf::scf_converged).
    let rhf_config = RhfConfig {
        density_conv: 1e-9,
        ..Default::default()
    };
    let ctx = ferric_core::parallel::ParallelContext::default();
    let rhf = solve_rhf(&ctx, mol, &obs, op, &bounds, &rhf_config)?;
    if !rhf.converged {
        return Err(FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy });
    }
    let dfbs = PreparedBasis::new(mol, aux_basis)?;
    let mp2 = ri_mp2(mol, &obs, &dfbs, op, &rhf, mp2_config)?;
    Ok(mp2.total_energy)
}

/// Compute RI-MP2 nuclear gradient via central finite differences.
///
/// `delta` is in Bohr (molecular coordinates are in Bohr).
/// Returns a `(natoms, 3)` array of dE/dR per atom per Cartesian direction.
pub fn rimp2_gradient_fd(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
    delta: f64,
) -> Result<Array2<f64>, FerricError> {
    let natoms = mol.atoms.len();
    let mut grad = Array2::zeros((natoms, 3));
    for atom in 0..natoms {
        for coord in 0..3 {
            let mut mol_p = mol.clone();
            let mut mol_m = mol.clone();
            match coord {
                0 => {
                    mol_p.atoms[atom].x += delta;
                    mol_m.atoms[atom].x -= delta;
                }
                1 => {
                    mol_p.atoms[atom].y += delta;
                    mol_m.atoms[atom].y -= delta;
                }
                _ => {
                    mol_p.atoms[atom].zpos += delta;
                    mol_m.atoms[atom].zpos -= delta;
                }
            }
            let e_p = total_energy(&mol_p, obs_basis, aux_basis, op, mp2_config)?;
            let e_m = total_energy(&mol_m, obs_basis, aux_basis, op, mp2_config)?;
            grad[(atom, coord)] = (e_p - e_m) / (2.0 * delta);
        }
    }
    Ok(grad)
}

/// Compute the analytical RI-MP2 nuclear gradient via the real multi-block
/// Lagrangian, following PySCF's verified `grad/mp2.py::grad_elec` (line numbers
/// below reference the fetched upstream source), RI-adapted.
///
/// The gradient is the sum of FOUR structurally distinct energy-weighted-density
/// contributions plus the hcore-deriv and RI 3c/2c integral-response terms — NOT
/// a single `F·P_relax` object (the old, ~50%-wrong construction):
///
/// 1. **Imat / overlap** (`+Σ dS·(im1+im1^T)`): the RI-MP2 Lagrangian matrix
///    [`crate::zvector::build_imat_ri`] rotated to AO (`im1 = C·Imat·C^T`, with the
///    vir-occ block set to the occ-vir transpose per PySCF line 145). This is the
///    RI analog of the 2-particle-density-derived Lagrangian block — built from the
///    same `x_ov`/`b_full` RI intermediates, no 4-index AO tensor. PySCF lines 121,
///    145-146, 174-175.
/// 2. **hcore-deriv** (`+Σ dH·dm1_total`): standard, `dm1_total = dm1_corr + hf_dm1`
///    (full relaxed correlation density plus the HF density). PySCF line 178. Also
///    carries the nuclear-repulsion gradient (once) via `oneelectron_gradient`.
/// 3. **zeta / overlap** (`−Σ dS·(zeta+zeta^T)`): the orbital-energy-weighted
///    relaxed density `zeta_mo = ζ ⊙ dm1mo` (ζ = 0.5(ε_i+ε_j) on oo/vv, ε_i on
///    ov/vo) PLUS the plain-HF energy-weighted density (`Σ_i 2ε_i C_i C_i^T`, reused
///    from `ferric_scf::gradient::build_energy_weighted_density`). This is
///    fundamentally `(ε-weight)⊙P_relax`, NOT `F·P_relax`. PySCF lines 156-159, 169,
///    180-181.
/// 4. **vhf_s1occ / overlap** (`−2Σ dS·vhf_s1occ`): PySCF's `get_veff = J − ½K`
///    potential of the full relaxed correlation density (`get_veff(dm1+dm1†)`, so
///    `J[2·dm1_corr] − ½K[2·dm1_corr]` from `build_jk` — NOT the SCF `2J − K`
///    convention on the already-doubled density), projected into the
///    occupied-occupied AO subspace by the HF-occupied projector on both sides.
///    PySCF lines 161-163, 183.
/// 5. **2e-integral-deriv** (`−Σ Γ(hf_dm1, hf_dm1+2·dm1_corr)·d(μν|λσ)`): the
///    BILINEAR two-electron gradient (`twoelectron_gradient_bilinear`), NOT
///    `Γ(P_relax,P_relax)`. Verified equal to PySCF's `vhf1·dm1p` element-by-element.
///    PySCF lines 103-109, 167, 184.
/// 6. **RI 3c/2c integral response** [`integral_response_gradient_3c2c`]: the RI
///    analog of PySCF's `part_dm2·int2e_ip1` (the separable 2-PDM contracted with
///    the differentiated integrals). Carries the closed-shell 2-PDM factor
///    Γ2 = 2·(2t−t̄): 3-center uses `2·y_ov = 2·V^{-1/2}·x_ov`; 2-center uses
///    `−2·(V^{-1}b_ov)·y_ov^T·dV` (the extra 2 vs 3-center from the metric
///    contracting both fitting legs).
///
/// Cross-checked block-by-block against `pyscf/grad/mp2.py::grad_elec`: blocks 1-5
/// match to ≤1e-4, block 6 to PySCF's `part_dm2·int2e_ip1`, for H2/cc-pVDZ AND
/// H2O/STO-3G. Both H2 (nocc=1) and H2O (nocc>1) are exact overall (~1e-9 vs FD).
pub fn rimp2_gradient_analytical(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    config: &RiMp2Config,
) -> Result<Array2<f64>, FerricError> {
    // ov-only intermediates: the gradient/zvector pipeline never reads
    // b_oo/b_vv, so the (naux, nvir²) block is never materialized here.
    let inter = compute_mp2_intermediates_ov_only(mol, obs, dfbs, op, rhf, config)?;
    let (z, imat) = solve_zvector(mol, obs, dfbs, op, bounds, rhf, &inter)?;
    let mut grad = mp2_relaxed_lagrangian_gradient(mol, obs, op, bounds, rhf, &inter, &z, &imat)?;
    grad += &integral_response_gradient_3c2c(mol, obs, dfbs, op, &inter, rhf.mos_r())?;
    Ok(grad)
}

/// Assemble the non-RI part of the MP2 analytical gradient (the four
/// overlap/hcore/2e-deriv Lagrangian blocks) from the solved z-vector and the RI
/// Lagrangian matrix `imat`. Shared by the RI-MP2 and SCS-MP2 gradient paths. The
/// caller adds the RI 3c/2c integral-response term separately.
///
/// See [`rimp2_gradient_analytical`] for the block-by-block derivation and PySCF
/// line citations.
#[allow(clippy::too_many_arguments)]
fn mp2_relaxed_lagrangian_gradient(
    mol: &Molecule,
    obs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    inter: &crate::rimp2::Mp2Intermediates,
    z: &Array2<f64>,
    imat: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    use ferric_scf::gradient::{
        build_energy_weighted_density, oneelectron_gradient, overlap_deriv_contract,
        twoelectron_gradient_bilinear,
    };
    let c = rhf.mos_r();
    let eps = rhf.eps_r();
    let nmo = c.ncols();
    let crate::rimp2::Mp2Intermediates { nocc, nvir, nocc_total, first_occ, .. } = *inter;

    // --- dm1mo: full relaxed correlation 1-PDM (MO), no HF 2·I part. ---
    // occ-occ: doo+doo^T; vir-vir: dvv+dvv^T; occ-vir/vir-occ: z (PySCF lines
    // 134-135 + _response_dm1). inter.p_oo==doo, inter.p_vv==dvv.
    let mut dm1mo = Array2::<f64>::zeros((nmo, nmo));
    for i in 0..nocc {
        let i_mo = first_occ + i;
        for j in 0..nocc {
            let j_mo = first_occ + j;
            dm1mo[(i_mo, j_mo)] = inter.p_oo[(i, j)] + inter.p_oo[(j, i)];
        }
    }
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for b in 0..nvir {
            let b_mo = nocc_total + b;
            dm1mo[(a_mo, b_mo)] = inter.p_vv[(a, b)] + inter.p_vv[(b, a)];
        }
    }
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            dm1mo[(a_mo, i_mo)] = z[(a, i)];
            dm1mo[(i_mo, a_mo)] = z[(a, i)];
        }
    }

    // Correlation density in AO, and the total (correlation + HF) density.
    let dm1_corr_ao = {
        let cp = c.dot(&dm1mo);
        cp.dot(&c.t())
    };
    let hf_dm1 = rhf.density_r();
    let dm1_total_ao = &dm1_corr_ao + hf_dm1;

    // --- im1 = C · Imat_pulay · C^T (Imat with vir-occ := occ-vir^T, PySCF 145) ---
    let mut imat_pulay = imat.clone();
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            imat_pulay[(a_mo, i_mo)] = imat[(i_mo, a_mo)];
        }
    }
    let im1 = {
        let cp = c.dot(&imat_pulay);
        cp.dot(&c.t())
    };

    // --- zeta_ao = C·(ζ ⊙ dm1mo)·C^T  +  make_rdm1e (PySCF lines 156-159, 169) ---
    let mut zeta_mo = Array2::<f64>::zeros((nmo, nmo));
    // ζ[i,j] = 0.5(ε_i+ε_j) on occ-occ AND vir-vir; ζ[vir,occ]=ε_i (occ energy);
    // ζ[occ,vir]=ε_i. Multiply element-wise by dm1mo.
    for p in 0..nmo {
        for q in 0..nmo {
            zeta_mo[(p, q)] = 0.5 * (eps[p] + eps[q]) * dm1mo[(p, q)];
        }
    }
    for a in 0..nvir {
        let a_mo = nocc_total + a;
        for i in 0..nocc {
            let i_mo = first_occ + i;
            zeta_mo[(a_mo, i_mo)] = eps[i_mo] * dm1mo[(a_mo, i_mo)];
            zeta_mo[(i_mo, a_mo)] = eps[i_mo] * dm1mo[(i_mo, a_mo)];
        }
    }
    let mut zeta_ao = {
        let cz = c.dot(&zeta_mo);
        cz.dot(&c.t())
    };
    // + plain-HF energy-weighted density Σ_i 2 ε_i C_i C_i^T.
    let nocc_hf = (mol.nelec() / 2) as usize;
    zeta_ao += &build_energy_weighted_density(rhf, nocc_hf);

    // --- vhf_s1occ = P_occ · get_veff(dm1_corr + dm1_corr^T) · P_occ (PySCF 161-163) ---
    // dm1_corr symmetric ⇒ dm1_corr+dm1_corr^T = 2·dm1_corr. PySCF's `get_veff(D)`
    // is `J[D] − ½K[D]` (the density D carries its own occupancy factor); here D is
    // already the doubled 2·dm1_corr, so veff = J[2dm] − ½K[2dm]. build_jk returns
    // raw J[D]/K[D], so this is `jv − 0.5·kv` — NOT the SCF `2J − K` convention
    // (which expects the un-doubled electron-count density).
    let n = c.nrows();
    let (mut jv, mut kv) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    let ctx = ferric_core::parallel::ParallelContext::default();
    let two_dm_corr = 2.0 * &dm1_corr_ao;
    ferric_scf::rhf::build_jk(&ctx, obs, bounds, 1e-12, &two_dm_corr, &mut jv, &mut kv)?;
    let veff_full = &jv - &(0.5 * &kv); // J[2dm] − ½K[2dm] == PySCF get_veff(dm1+dm1^T)
    // P_occ = C_occ C_occ^T (HF-occupied AO projector).
    let c_occ_hf = c.slice(ndarray::s![.., ..nocc_hf]);
    let p_occ = c_occ_hf.dot(&c_occ_hf.t());
    let vhf_s1occ = p_occ.dot(&veff_full).dot(&p_occ);

    // --- Assemble with PySCF's explicit signs (grad/mp2.py lines 174-184). ---
    // hcore-deriv (+ nuclear-repulsion, once) with dm1_total; pass W=0 so the
    // overlap/Pulay terms are added explicitly below with the correct per-block
    // signs (im1/zeta/vhf are asymmetric or differently-signed).
    let zero_w = Array2::<f64>::zeros((c.nrows(), c.nrows()));
    let mut grad = oneelectron_gradient(mol, obs, &dm1_total_ao, &zero_w, None)?;

    // Overlap (Pulay) contributions:
    //   + s1·im1 + s1^T·im1^T       →  + overlap_deriv_contract(im1)        (174-175)
    //   − s1·zeta − s1^T·zeta^T     →  − overlap_deriv_contract(zeta_ao)    (180-181)
    //   − 2·s1·vhf_s1occ            →  − overlap_deriv_contract(vhf_s1occ)  (183)
    // overlap_deriv_contract(M) = Σ_μν dS_μν (M+M^T)_μν, so it reproduces PySCF's
    // paired (s1 + s1^T) / ×2 conventions in one call per matrix (verified against
    // the RHF EWD term). Combine into one weight matrix.
    let w_overlap = &im1 - &zeta_ao - &vhf_s1occ;
    grad += &overlap_deriv_contract(obs, &w_overlap)?;

    // 2e-integral-derivative: bilinear Γ(hf_dm1, hf_dm1 + 2·dm1_corr) (PySCF 184,
    // dm1p = hf_dm1 + 2·dm1). ferric's twoelectron_gradient(X) = −∂veff(X)·X for
    // the HF case; the bilinear form generalizes it to −∂veff(hf_dm1)·dm1p.
    let dm1p = hf_dm1 + &two_dm_corr;
    grad += &twoelectron_gradient_bilinear(obs, op, bounds, hf_dm1, &dm1p)?;

    Ok(grad)
}

/// 3-center + 2-center RI integral-response gradient terms.
///
/// Factored out of [`rimp2_gradient_analytical`] so the P7-parallelized region
/// (x_ov par-i build, aux-shell 3c-derivative assembly, aux-pair 2c-derivative
/// assembly) can be exercised in isolation — e.g. by the thread-count
/// bit-identity test — with fixed precomputed intermediates, independent of the
/// z-vector/JK pipeline upstream. Returns just the 3c+2c contribution as a
/// `(natoms, 3)` array; the caller adds it onto the relaxed-density HF gradient.
pub(crate) fn integral_response_gradient_3c2c(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    inter: &crate::rimp2::Mp2Intermediates,
    c: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let mut grad = Array2::<f64>::zeros((mol.atoms.len(), 3));

    // 3-center derivative: Σ_{P,μ,ν} G3c_{P,μ,ν} * d(P|μν)/dR
    // G3c = "effective density" contracted with t2-weighted amplitudes
    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let naux = inter.naux;
    let nov = nocc * nvir;
    let t2 = &inter.t2;

    // Build X^P_{ia} = Σ_{jb} (2*t_{ij,ab} - t_{ij,ba}) * B^P_{jb} via a wide GEMM
    // per i: X_i[P, a] = B_ov · TT_i^T where TT_i[a, jb] = 2 t_{ij,ab} - t_{ij,ba}.
    // Replaces the per-element (P,i,a) scalar loop over (j,b): same FLOPs, BLAS3
    // throughput. t2 layout is t2[(i*nvir+a)*nov + j*nvir+b] = t_{ij,ab}.
    //
    // Each i owns a disjoint (naux, nvir) column band of x_ov and reads only its
    // own TT_i slice of t2, so the i-loop is embarrassingly parallel: fan it over
    // rayon via axis_chunks_iter_mut on the column-chunked view (order-preserving,
    // no shared accumulator — bit-identical to the serial loop regardless of
    // thread count). BLAS stays serial inside each closure (OPENBLAS_NUM_THREADS=1).
    let mut x_ov = Array2::zeros((naux, nocc * nvir));
    {
        use ndarray::Axis;
        use rayon::prelude::*;
        // AxisChunksIterMut is an IndexedParallelIterator: chunk i is exactly
        // columns [i*nvir, (i+1)*nvir), so `enumerate()` recovers i without any
        // extra bookkeeping, and the disjoint-column writes make this
        // order-independent/bit-identical to the serial loop.
        x_ov.axis_chunks_iter_mut(Axis(1), nvir)
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut x_i_slot)| {
                // TT_i[a, jb] = 2 t_{ij,ab} - t_{ij,ba}
                let mut tt_i = Array2::<f64>::zeros((nvir, nov));
                for a in 0..nvir {
                    for j in 0..nocc {
                        for b in 0..nvir {
                            let jb = j * nvir + b;
                            let t_ij_ab = t2[(i * nvir + a) * nov + j * nvir + b];
                            let t_ij_ba = t2[(i * nvir + b) * nov + j * nvir + a];
                            tt_i[(a, jb)] = 2.0 * t_ij_ab - t_ij_ba;
                        }
                    }
                }
                // X_i[P, a] = Σ_{jb} B_ov[P, jb] · TT_i[a, jb] = B_ov · TT_i^T  (naux, nvir)
                let x_i = inter.b_ov.dot(&tt_i.t());
                x_i_slot.assign(&x_i);
            });
    }

    // The 3-center DERIVATIVE integrals d(P|μν)/dR are RAW (undressed) 3-center
    // integrals, but the fitted amplitudes `x_ov` are dressed with V^{-1/2} on the
    // aux leg (b_ov = V^{-1/2}(P|ov)). The energy `E = ½ Σ Γ B^P_ia B^P_jb` with
    // `B = V^{-1/2}(Q|ia)` gives `dE/d(Q|ia)_raw = Σ_P V^{-1/2}_QP X^P_ia = y_ov`,
    // so the 3-center contraction must use `y_ov = V^{-1/2}·x_ov`, NOT `x_ov`
    // (verified element-by-element in test_3c_density_numerical: diff(y)≈8e-13,
    // diff(x)≈1.4e-2). V^{-1/2} is symmetric so `v_inv_sqrt.t()` == `v_inv_sqrt`.
    let y_ov = with_blas_threads(opt_in_blas_threads(), || inter.v_inv_sqrt.t().dot(&x_ov));

    // Back-transform X to AO and contract with 3-center derivative integrals,
    // blocked over the aux SHELL index. For each aux shell sp we build only its
    // g3c slab G3c_{p,μ,ν} = Σ_{ia} X^p_{ia} C_{μi} C_{νa} (symmetrized), then
    // contract it with that shell's derivative block. Peak g3c footprint is one
    // aux-shell slab (np, nbf, nbf) instead of the full (naux, nbf, nbf) tensor.
    let nbas = obs.nbasis();
    let c_occ = c.slice(ndarray::s![.., inter.first_occ..inter.first_occ + nocc]).to_owned();
    let c_vir = c.slice(ndarray::s![.., inter.nocc_total..]).to_owned();
    {
        use ferric_integrals::engine::Engine;
        use rayon::prelude::*;
        let nsh_obs = obs.nshells();
        let nsh_df = dfbs.nshells();
        let dims_obs = obs.shell_dims();
        let offs_obs = obs.shell_offsets();
        let dims_df = dfbs.shell_dims();
        let offs_df = dfbs.shell_offsets();
        let sh2at_obs = obs.shell_to_atom();
        let sh2at_df = dfbs.shell_to_atom();
        let natoms = mol.atoms.len();

        // Surface any engine-construction error up front (serial, cheap) so the
        // per-worker rebuilds inside for_each_init cannot fail for the same args
        // (mirrors ferric_integrals::threeindex::eri3_tensor).
        Engine::new_3center_deriv(op, obs, dfbs, 1e-14)?;

        // Axis: aux shells `sp`. Each sp is independent (own g3c slab, own deriv
        // engine call) and produces a (natoms, 3) partial. Reduction must not use
        // a rayon fold/reduce tree (thread-count-dependent FP association) or a
        // shared accumulator (data race) — instead: par-map sp -> partial via
        // for_each_init (one Engine per rayon worker), `collect` into an
        // sp-ordered Vec (IndexedParallelIterator preserves order regardless of
        // worker count), then fold serially in ascending sp order. This mirrors
        // ferric_scf::reduce::grouped_deterministic_sum's group-order-fold
        // discipline without depending on ferric-scf (mp2 does not link it);
        // natoms is small so no banding is needed — the live set is one
        // (natoms,3) partial per aux shell, negligible memory.
        let partials: Vec<Array2<f64>> = (0..nsh_df)
            .into_par_iter()
            .map_init(
                || Engine::new_3center_deriv(op, obs, dfbs, 1e-14).expect("3-center deriv engine (pre-validated)"),
                |eng3d, sp| {
                    let np = dims_df[sp];
                    let pf0 = offs_df[sp];
                    let mut local_grad = Array2::<f64>::zeros((natoms, 3));

                    // g3c for just this aux shell's rows: (np, nbf, nbf), symmetrized.
                    // For each aux fn p: X_p is (nocc, nvir); G_raw = C_occ · X_p · C_vir^T,
                    // then G3c_p = G_raw + G_raw^T (matches the old full-tensor symmetrize,
                    // including the ×2 on the diagonal via mu==nu).
                    let mut g3c_sp = ndarray::Array3::<f64>::zeros((np, nbas, nbas));
                    for p in 0..np {
                        let pf = pf0 + p;
                        let x_p = y_ov
                            .slice(ndarray::s![pf, ..])
                            .into_shape_with_order((nocc, nvir))
                            .unwrap();
                        // G_raw_{μν} = Σ_{ia} X^p_{ia} C_{μi} C_{νa} = C_occ · X_p · C_vir^T
                        // Symmetrized: dE/d(P|μν)_raw = g_raw[μν] + g_raw[νμ] (the aux fn P
                        // couples ia symmetrically; a raw 3c integral (P|μν) and its μ↔ν
                        // transpose share the derivative). g_raw + g_raw^T doubles the
                        // diagonal correctly (matches the FD single-element probe).
                        // The 3-center weight carries the closed-shell 2-PDM factor
                        // Γ2 = 2·(2t−t̄) (PySCF `part_dm2 = 4t − 2t^T`, grad/mp2.py
                        // lines 61-62). `x_ov`/`y_ov` are built with the ENERGY weight
                        // (2t−t̄) (shared with the Imat/energy path), so the separable
                        // 2-PDM gradient needs an explicit ×2 here. Verified: the term
                        // then equals PySCF's `dm2buf·int2e_ip1` contribution and drives
                        // both H2 (nocc=1) and H2O (nocc=5) analytic−FD to ~1e-9.
                        let g_raw = c_occ.dot(&x_p).dot(&c_vir.t());
                        let g_sym = 2.0 * (&g_raw + &g_raw.t());
                        g3c_sp.slice_mut(ndarray::s![p, .., ..]).assign(&g_sym);
                    }

                    for s1 in 0..nsh_obs {
                        for s2 in 0..=s1 {
                            if let Some(deriv) = eng3d.compute_eri3_deriv(obs, dfbs, sp, s1, s2) {
                                let n1 = dims_obs[s1];
                                let n2 = dims_obs[s2];
                                let block_sz = np * n1 * n2;
                                let sym12 = s1 != s2;

                                for p in 0..np {
                                    for i in 0..n1 {
                                        for j in 0..n2 {
                                            let mu = offs_obs[s1] + i;
                                            let nu = offs_obs[s2] + j;
                                            let idx = (p * n1 + i) * n2 + j;

                                            let gval = if sym12 {
                                                g3c_sp[(p, mu, nu)] + g3c_sp[(p, nu, mu)]
                                            } else {
                                                g3c_sp[(p, mu, nu)]
                                            };

                                            // 9 derivative blocks: [dP_x, dP_y, dP_z, d1_x, d1_y, d1_z, d2_x, d2_y, d2_z]
                                            let atom_p = sh2at_df[sp];
                                            let atom_1 = sh2at_obs[s1];
                                            let atom_2 = sh2at_obs[s2];

                                            for coord in 0..3 {
                                                let dp = deriv[coord * block_sz + idx];
                                                let d1 = deriv[(3 + coord) * block_sz + idx];
                                                let d2 = deriv[(6 + coord) * block_sz + idx];

                                                local_grad[(atom_p, coord)] += gval * dp;
                                                local_grad[(atom_1, coord)] += gval * d1;
                                                local_grad[(atom_2, coord)] += gval * d2;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    local_grad
                },
            )
            .collect();

        // Serial fold in ascending sp order — the determinism anchor (same
        // discipline as grouped_deterministic_sum, just without banding since
        // the partials are tiny).
        for p in &partials {
            grad += p;
        }
    }

    // 2-center metric derivative: Σ_{PQ} Γ^2c_{PQ} * d(P|Q)/dR. This GEMM sits
    // between two rayon regions (the 3c-derivative fan-out above has already
    // collected/returned; the 2c-derivative fan-out below hasn't started) —
    // outside any rayon region itself. Opt-in BLAS raise via
    // FERRIC_BLAS_THREADS (default 1, unchanged behavior).
    // 2-center metric-derivative weight Γ2c for V^{-1/2} fitting. Deriving the RI
    // energy `E = Σ Γ_ia,jb (ia|P) V^{-1}_PQ (Q|jb)` w.r.t. R gives the 2-center
    // term `Σ_PQ Γ2c_PQ dV_PQ` with base weight `−½·C·y_ov^T`, where
    // `C = V^{-1}(P|ov) = V^{-1/2}·b_ov` (the Coulomb-fitted 3-index integrals) and
    // `y_ov = V^{-1/2}·x_ov` (the same V^{-1/2}-dressed amplitudes the 3-center term
    // uses). The pair loop below symmetrizes (Γ2c_PQ + Γ2c_QP), so the asymmetric
    // C·y^T is contracted correctly. (The prior `−½ x_ov·x_ov^T` form was wrong — it
    // omitted the C factor and used x_ov instead of the fitted C/y pair.)
    //
    // The base weight is scaled by 4: the SAME closed-shell 2-PDM factor 2 the
    // 3-center term carries (Γ2 = 2·(2t−t̄)), PLUS a second factor 2 because the
    // metric appears on BOTH fitting legs — differentiating V^{-1} = −V^{-1}·dV·V^{-1}
    // contracts symmetrically against the P and Q legs. Verified against PySCF's
    // `part_dm2·int2e_ip1` DF-metric response and the frozen-amplitude FD (H2 + H2O
    // analytic−FD ~1e-9; the pre-fix ×1 weight left H2O ~1e-2 off).
    let c_fit = with_blas_threads(opt_in_blas_threads(), || inter.v_inv_sqrt.t().dot(&inter.b_ov));
    let gamma_2c = with_blas_threads(opt_in_blas_threads(), || -2.0 * c_fit.dot(&y_ov.t()));

    {
        use ferric_integrals::engine::Engine;
        use rayon::prelude::*;
        let nsh_df = dfbs.nshells();
        let dims_df = dfbs.shell_dims();
        let offs_df = dfbs.shell_offsets();
        let sh2at_df = dfbs.shell_to_atom();
        let natoms = mol.atoms.len();

        Engine::new_2center_deriv(op, dfbs, 1e-14)?;

        // Axis: aux shell `sp` (outer loop of the sp>=sq triangle). Same
        // par-map + sp-ordered collect + serial fold discipline as the 3c block
        // above — each sp owns the full sq<=sp inner sweep as its unit of work,
        // so partials are independent and the fold order is sp-ascending only.
        let partials: Vec<Array2<f64>> = (0..nsh_df)
            .into_par_iter()
            .map_init(
                || Engine::new_2center_deriv(op, dfbs, 1e-14).expect("2-center deriv engine (pre-validated)"),
                |eng2d, sp| {
                    let mut local_grad = Array2::<f64>::zeros((natoms, 3));
                    for sq in 0..=sp {
                        if let Some(deriv) = eng2d.compute_eri2_deriv(dfbs, sp, sq) {
                            let np = dims_df[sp];
                            let nq = dims_df[sq];
                            let block_sz = np * nq;
                            let sym_pq = sp != sq;

                            for p in 0..np {
                                for q in 0..nq {
                                    let pf = offs_df[sp] + p;
                                    let qf = offs_df[sq] + q;
                                    let idx = p * nq + q;

                                    let gval = if sym_pq {
                                        gamma_2c[(pf, qf)] + gamma_2c[(qf, pf)]
                                    } else {
                                        gamma_2c[(pf, qf)]
                                    };

                                    let atom_p = sh2at_df[sp];
                                    let atom_q = sh2at_df[sq];

                                    for coord in 0..3 {
                                        let dp = deriv[coord * block_sz + idx];
                                        let dq = deriv[(3 + coord) * block_sz + idx];

                                        local_grad[(atom_p, coord)] += gval * dp;
                                        local_grad[(atom_q, coord)] += gval * dq;
                                    }
                                }
                            }
                        }
                    }
                    local_grad
                },
            )
            .collect();

        // Serial fold in ascending sp order.
        for p in &partials {
            grad += p;
        }
    }

    Ok(grad)
}

/// Compute the analytical SCS-MP2 nuclear gradient.
pub fn scs_mp2_gradient_analytical(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    rhf: &ScfResult,
    config: &crate::scs::ScsMp2Config,
) -> Result<Array2<f64>, FerricError> {
    let mp2_config = RiMp2Config { frozen_core: config.frozen_core, memory_budget_bytes: config.memory_budget_bytes };
    let inter = compute_mp2_intermediates_ov_only(mol, obs, dfbs, op, rhf, &mp2_config)?;
    let (z, imat) = solve_zvector(mol, obs, dfbs, op, bounds, rhf, &inter)?;
    // Full (unscaled) RI-MP2 relaxed-Lagrangian gradient + RI 3c/2c response.
    // NOTE: external_potential is not threaded into correlated gradients (see the
    // external-potentials design doc's non-goals).
    let mut grad = mp2_relaxed_lagrangian_gradient(mol, obs, op, bounds, rhf, &inter, &z, &imat)?;
    grad += &integral_response_gradient_3c2c(mol, obs, dfbs, op, &inter, rhf.mos_r())?;
    // Approximate scaling: multiply MP2 part by average SCS scaling
    let scale = (config.c_os + config.c_ss) / 2.0;
    let rhf_grad = ferric_scf::gradient::rhf_gradient(mol, obs, op, bounds, rhf, None)?;
    for i in 0..mol.atoms.len() {
        for c in 0..3 {
            let mp2_part = grad[(i, c)] - rhf_grad[(i, c)];
            grad[(i, c)] = rhf_grad[(i, c)] + scale * mp2_part;
        }
    }
    Ok(grad)
}

/// Compute the total RI-MP2 energy and its analytical nuclear gradient at a
/// single geometry: solve RHF, then run `rimp2_gradient_analytical`.
///
/// Mirrors `ferric_rpa::gradient::total_rpa_gradient`'s energy+gradient
/// pairing so `ferric_mp2::optimize::optimize_geometry_rimp2` can drive a
/// BFGS loop the same way `ferric_rpa::optimize::optimize_geometry_rpa`
/// does for RPA. `E_total = E_HF + E_MP2`; the returned gradient is
/// dE_total/dR (analytical, not finite-difference).
pub fn total_rimp2_gradient(
    mol: &Molecule,
    obs_basis: &BasisSet,
    aux_basis: &BasisSet,
    op: Operator,
    mp2_config: &RiMp2Config,
) -> Result<(f64, Array2<f64>), FerricError> {
    let ctx = ferric_core::parallel::ParallelContext::default();
    let obs = PreparedBasis::new(mol, obs_basis)?;
    let dfbs = PreparedBasis::new(mol, aux_basis)?;
    let bounds = SchwarzBounds::compute(op, &obs)?;
    // Tight SCF via density_conv (reachable under the ΔP gate); energy_conv left
    // at the loose default — a tight 1e-10 would hang at MaxIter. See total_energy above.
    let rhf_cfg = RhfConfig {
        density_conv: 1e-9,
        ..Default::default()
    };
    let rhf = solve_rhf(&ctx, mol, &obs, op, &bounds, &rhf_cfg)?;
    if !rhf.converged {
        return Err(FerricError::ScfConvergence { iterations: rhf.iterations, last_energy: rhf.energy });
    }
    let mp2 = ri_mp2(mol, &obs, &dfbs, op, &rhf, mp2_config)?;
    let grad = rimp2_gradient_analytical(mol, &obs, &dfbs, op, &bounds, &rhf, mp2_config)?;
    Ok((mp2.total_energy, grad))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_scf::gradient::{hf_gradient_with_density, oneelectron_gradient, twoelectron_gradient};

    #[test]
    fn test_rimp2_gradient_fd_h2_symmetry() {
        // H2 along z-axis: gradient should be equal and opposite on the two atoms
        // and zero in x,y directions.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let config = RiMp2Config::default();
        let grad = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 1e-4).unwrap();

        eprintln!("H2 RI-MP2 gradient (FD, delta=1e-4):");
        for atom in 0..2 {
            eprintln!(
                "  atom {}: [{:+.10}, {:+.10}, {:+.10}]",
                atom,
                grad[(atom, 0)],
                grad[(atom, 1)],
                grad[(atom, 2)]
            );
        }

        // x and y gradients should be ~0
        for atom in 0..2 {
            for c in 0..2 {
                assert!(
                    grad[(atom, c)].abs() < 1e-8,
                    "atom={atom} coord={c}: {:.2e} should be ~0",
                    grad[(atom, c)]
                );
            }
        }

        // z gradients should be equal and opposite (translational invariance)
        assert!(
            (grad[(0, 2)] + grad[(1, 2)]).abs() < 1e-8,
            "z gradients not equal/opposite: {} vs {}",
            grad[(0, 2)],
            grad[(1, 2)]
        );

        // Should be nonzero (H2 at 0.74 A is near but not at equilibrium for MP2/cc-pVDZ)
        assert!(
            grad[(0, 2)].abs() > 1e-4,
            "z gradient too small: {}",
            grad[(0, 2)]
        );
    }

    #[test]
    fn test_rimp2_gradient_fd_consistency() {
        // Check that two different deltas give consistent results (FD convergence).
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let config = RiMp2Config::default();

        let g1 = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 1e-4).unwrap();
        let g2 = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 5e-5).unwrap();

        eprintln!("FD consistency check (H2 z-component):");
        eprintln!("  delta=1e-4: {:.10}", g1[(0, 2)]);
        eprintln!("  delta=5e-5: {:.10}", g2[(0, 2)]);
        eprintln!("  diff:       {:.2e}", (g1[(0, 2)] - g2[(0, 2)]).abs());

        // Central FD has O(delta^2) error, so halving delta should reduce error by ~4x.
        // Two independent runs should agree to ~1e-5 or better.
        assert!(
            (g1[(0, 2)] - g2[(0, 2)]).abs() < 1e-5,
            "FD inconsistent: delta=1e-4 gives {:.10}, delta=5e-5 gives {:.10}",
            g1[(0, 2)],
            g2[(0, 2)]
        );
    }

    #[test]
    fn test_rimp2_gradient_fd_h2o_translational_invariance() {
        // For any geometry the sum of forces over all atoms should be zero
        // (translational invariance / Newton's third law).
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("sto-3g").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let config = RiMp2Config::default();

        let grad = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 1e-4).unwrap();

        eprintln!("H2O/STO-3G RI-MP2 gradient (FD):");
        for atom in 0..3 {
            eprintln!(
                "  atom {}: [{:+.10}, {:+.10}, {:+.10}]",
                atom,
                grad[(atom, 0)],
                grad[(atom, 1)],
                grad[(atom, 2)]
            );
        }

        // Sum of gradients over all atoms should vanish for each coordinate.
        for c in 0..3 {
            let sum: f64 = (0..3).map(|a| grad[(a, c)]).sum();
            assert!(
                sum.abs() < 1e-6,
                "translational invariance violated: coord={c}, sum={:.2e}",
                sum
            );
        }
    }

    #[test]
    fn test_analytical_vs_fd_h2() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();

        let analytical = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();
        let fd = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 1e-4).unwrap();

        eprintln!("=== H2/cc-pVDZ Analytical vs FD RI-MP2 gradient ===");
        let mut max_diff = 0.0f64;
        for atom in 0..2 {
            for c in 0..3 {
                let diff = (analytical[(atom, c)] - fd[(atom, c)]).abs();
                max_diff = max_diff.max(diff);
                eprintln!(
                    "  atom={} coord={}: analytical={:+.8} fd={:+.8} diff={:.2e}",
                    atom, c, analytical[(atom, c)], fd[(atom, c)], diff
                );
            }
        }
        eprintln!("  max diff = {:.2e}", max_diff);
        // Full multi-block Lagrangian (Imat/zeta/vhf_s1occ) + RI 3c/2c response.
        // Analytic == FD to ~1e-9 after the 3c/2c 2-PDM-factor fix; 1e-6 leaves
        // headroom over the delta=1e-4 central-difference reference's truncation floor.
        assert!(max_diff < 1e-6,
            "analytical vs FD max diff = {:.2e} (expected < 1e-6)", max_diff);
    }

    #[test]
    fn test_analytical_gradient_translational_invariance() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();

        let grad = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();

        for c in 0..3 {
            let sum: f64 = (0..2).map(|a| grad[(a, c)]).sum();
            assert!(sum.abs() < 1e-8,
                "translational invariance: coord={} sum={:.2e}", c, sum);
        }
    }

    #[test]
    fn test_3c_density_numerical() {
        // Numerically verify dE/d(P|mu,nu) for a single element.
        // Perturb one raw 3-center integral and see how E_corr changes.
        use crate::rimp2::{compute_mp2_intermediates, RiMp2Config, cholesky_inverse_sqrt};
        use ferric_integrals::threeindex;

        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();
        let inter = compute_mp2_intermediates(&mol, &obs, &dfbs, op, &rhf, &config).unwrap();

        let nocc = inter.nocc;
        let nvir = inter.nvir;
        let nov = nocc * nvir;
        let naux = inter.naux;
        let c = rhf.mos_r();
        let t2 = &inter.t2;

        // Get raw 3-center integrals and metric
        let eri3_ao = threeindex::eri3_tensor(op, &obs, &dfbs).unwrap();
        let v2c = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();

        let c_occ = c.slice(ndarray::s![.., inter.first_occ..inter.first_occ + nocc]).to_owned();
        let c_vir = c.slice(ndarray::s![.., inter.nocc_total..]).to_owned();

        let compute_ecorr_from_3c = |eri3: &ndarray::Array3<f64>| -> f64 {
            let v_inv_sqrt = cholesky_inverse_sqrt(&v2c).unwrap();
            let eri3_ov = crate::mo_transform::transform_3center_ov(eri3, &c_occ, &c_vir);
            let b_ov = v_inv_sqrt.dot(&eri3_ov.into_shape_with_order((naux, nov)).unwrap());
            let mut e = 0.0;
            for i in 0..nocc {
                for a in 0..nvir {
                    let ia = i * nvir + a;
                    for j in 0..nocc {
                        for b in 0..nvir {
                            let jb = j * nvir + b;
                            let t_ij_ab = t2[ia * nov + jb];
                            let t_ij_ba = t2[(i*nvir+b) * nov + (j*nvir+a)];
                            let tt = 2.0 * t_ij_ab - t_ij_ba;
                            let eri: f64 = (0..naux).map(|p| b_ov[(p, ia)] * b_ov[(p, jb)]).sum();
                            e += tt * eri;
                        }
                    }
                }
            }
            e * 0.5
        };

        // Build x_ov and analytical densities
        let mut x_ov = Array2::zeros((naux, nov));
        for i in 0..nocc {
            for a in 0..nvir {
                let ia = i * nvir + a;
                for p_aux in 0..naux {
                    let mut sum = 0.0;
                    for j in 0..nocc {
                        for b in 0..nvir {
                            let jb = j * nvir + b;
                            let t_ij_ab = t2[(i * nvir + a) * nov + j * nvir + b];
                            let t_ij_ba = t2[(i * nvir + b) * nov + j * nvir + a];
                            let tt = 2.0 * t_ij_ab - t_ij_ba;
                            sum += tt * inter.b_ov[(p_aux, jb)];
                        }
                    }
                    x_ov[(p_aux, ia)] = sum;
                }
            }
        }
        let y_ov = inter.v_inv_sqrt.t().dot(&x_ov);

        // Test a specific 3c element: perturb eri3[P=0, mu=0, nu=0]
        let test_p = 0;
        let test_mu = 0;
        let test_nu = 1;
        let delta = 1e-6;

        let mut eri3_plus = eri3_ao.clone();
        let mut eri3_minus = eri3_ao.clone();
        eri3_plus[(test_p, test_mu, test_nu)] += delta;
        eri3_plus[(test_p, test_nu, test_mu)] += delta;  // symmetrize
        eri3_minus[(test_p, test_mu, test_nu)] -= delta;
        eri3_minus[(test_p, test_nu, test_mu)] -= delta;

        let e_plus = compute_ecorr_from_3c(&eri3_plus);
        let e_minus = compute_ecorr_from_3c(&eri3_minus);
        let fd_deriv = (e_plus - e_minus) / (2.0 * delta);

        // Analytical: dE/d(P|mu,nu) using x_ov (original code's formula)
        let g3c_x_munu = (0..nocc).flat_map(|i| (0..nvir).map(move |a| (i, a)))
            .map(|(i, a)| x_ov[(test_p, i * nvir + a)] * c_occ[(test_mu, i)] * c_vir[(test_nu, a)]
                        + x_ov[(test_p, i * nvir + a)] * c_occ[(test_nu, i)] * c_vir[(test_mu, a)])
            .sum::<f64>();

        // Analytical: dE/d(P|mu,nu) using y_ov (proposed fix)
        let g3c_y_munu = (0..nocc).flat_map(|i| (0..nvir).map(move |a| (i, a)))
            .map(|(i, a)| y_ov[(test_p, i * nvir + a)] * c_occ[(test_mu, i)] * c_vir[(test_nu, a)]
                        + y_ov[(test_p, i * nvir + a)] * c_occ[(test_nu, i)] * c_vir[(test_mu, a)])
            .sum::<f64>();

        eprintln!("=== dE/d(P={},mu={},nu={}) ===", test_p, test_mu, test_nu);
        eprintln!("FD:        {:.12}", fd_deriv);
        eprintln!("x_ov:      {:.12}  (code's formula, no extra V^{{-1/2}})", g3c_x_munu);
        eprintln!("y_ov:      {:.12}  (with extra V^{{-1/2}})", g3c_y_munu);
        eprintln!("2*y_ov:    {:.12}  (with factor 2)", 2.0 * g3c_y_munu);
        eprintln!("diff(x):   {:.6e}", g3c_x_munu - fd_deriv);
        eprintln!("diff(y):   {:.6e}", g3c_y_munu - fd_deriv);
        eprintln!("diff(2y):  {:.6e}", 2.0 * g3c_y_munu - fd_deriv);
    }


    #[test]
    fn test_analytical_vs_fd_h2o() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("sto-3g").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();

        let analytical = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();
        let fd = rimp2_gradient_fd(&mol, &obs_bs, &aux_bs, op, &config, 1e-4).unwrap();

        eprintln!("=== H2O/STO-3G Analytical vs FD RI-MP2 gradient ===");
        let mut max_diff = 0.0f64;
        for atom in 0..3 {
            for c in 0..3 {
                let diff = (analytical[(atom, c)] - fd[(atom, c)]).abs();
                max_diff = max_diff.max(diff);
                eprintln!("  atom={} coord={}: analytical={:+.8} fd={:+.8} diff={:.2e}",
                    atom, c, analytical[(atom, c)], fd[(atom, c)], diff);
            }
        }
        eprintln!("  max diff = {:.2e}", max_diff);
        // Both H2 (nocc=1) and H2O (nocc=5) now agree with FD to ~1e-9 after the
        // RI 3c/2c integral-response 2-PDM-factor fix (see
        // `integral_response_gradient_3c2c`). Previously H2O sat at ~1.0e-2: the
        // 3-center weight was missing the closed-shell 2-PDM factor 2 (Γ2 = 2·(2t−t̄))
        // and the 2-center metric weight was missing 4 (that same 2, times a second 2
        // from the metric contracting both fitting legs). At nocc=1 the two channel
        // errors cancelled in the total (H2 was accidentally tight at 2.8e-5); at
        // nocc>1 they did not. Term-by-term cross-check vs PySCF conventional MP2
        // (hcore/im1/zeta/vhf_s1occ/bilinear-2e all already matched to ≤1e-4; only the
        // RI 2e-response `part_dm2·int2e_ip1` term was off) pinned it precisely.
        assert!(max_diff < 1e-6,
            "H2O analytical vs FD max diff = {:.2e} (expected < 1e-6)", max_diff);
    }

    #[test]
    fn test_3c2c_assembly_bit_identical_across_thread_counts() {
        // Scope: ONLY the P7-parallelized region (x_ov par-i build, aux-shell
        // 3c-derivative assembly, aux-pair 2c-derivative assembly), fed the
        // SAME precomputed intermediates in rayon pools pinned to 1 and 4
        // workers, compared via f64::to_bits. The whole-pipeline gate this
        // test used to carry is RESOLVED: P14 migrated build_jk (used inside
        // solve_zvector, upstream of this region) off its thread-count-
        // dependent fold/reduce tree onto the grouped deterministic sum, and
        // the full solve_rhf → rhf_gradient pipeline is now covered by
        // `whole_pipeline_rhf_gradient_bit_identical_across_thread_counts`
        // in ferric-scf/src/rhf.rs.
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();

        // Precompute intermediates ONCE, outside any pinned pool, so both runs
        // see bit-identical inputs and only the P7 region is under test.
        let inter = compute_mp2_intermediates_ov_only(&mol, &obs, &dfbs, op, &rhf, &config).unwrap();

        let run_with_threads = |n: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
            pool.install(|| {
                integral_response_gradient_3c2c(&mol, &obs, &dfbs, op, &inter, rhf.mos_r()).unwrap()
            })
        };

        let g1 = run_with_threads(1);
        let g4 = run_with_threads(4);

        for atom in 0..2 {
            for c in 0..3 {
                assert_eq!(
                    g1[(atom, c)].to_bits(),
                    g4[(atom, c)].to_bits(),
                    "3c+2c assembly not bit-identical across thread counts at atom={atom} coord={c}: \
                     1-thread={:.17e} (0x{:016x}), 4-thread={:.17e} (0x{:016x})",
                    g1[(atom, c)], g1[(atom, c)].to_bits(),
                    g4[(atom, c)], g4[(atom, c)].to_bits(),
                );
            }
        }
    }

    #[test]
    fn test_analytical_h2_symmetry() {
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();

        let grad = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();

        // x,y components should be zero for H2 along z-axis
        for atom in 0..2 {
            for c in 0..2 {
                assert!(grad[(atom, c)].abs() < 1e-12,
                    "atom={} coord={}: {:.2e} should be ~0", atom, c, grad[(atom, c)]);
            }
        }
        // z components equal and opposite
        assert!((grad[(0, 2)] + grad[(1, 2)]).abs() < 1e-10,
            "z not equal/opposite: {} vs {}", grad[(0, 2)], grad[(1, 2)]);
        // z should be nonzero
        assert!(grad[(0, 2)].abs() > 1e-4, "z gradient too small: {}", grad[(0, 2)]);
    }

    #[test]
    fn test_analytical_h2o_translational_invariance() {
        let xyz = "3\nwater\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("sto-3g").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let config = RiMp2Config::default();

        let grad = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();

        for c in 0..3 {
            let sum: f64 = (0..3).map(|a| grad[(a, c)]).sum();
            assert!(sum.abs() < 1e-8,
                "H2O translational invariance: coord={} sum={:.2e}", c, sum);
        }
    }

    #[test]
    fn test_gradient_split_consistency() {
        // oneelectron_gradient + twoelectron_gradient should equal hf_gradient_with_density
        let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let obs = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ferric_core::parallel::ParallelContext::default(), &mol, &obs, op, &bounds, &RhfConfig { energy_conv: 1e-10, ..Default::default() }).unwrap();
        let nocc = (mol.nelec() / 2) as usize;
        let w = ferric_scf::gradient::build_energy_weighted_density(&rhf, nocc);

        let combined = hf_gradient_with_density(&mol, &obs, op, &bounds, rhf.density_r(), &w, None).unwrap();
        let oe = oneelectron_gradient(&mol, &obs, rhf.density_r(), &w, None).unwrap();
        let te = twoelectron_gradient(&obs, op, &bounds, rhf.density_r()).unwrap();
        let split = &oe + &te;

        for atom in 0..2 {
            for c in 0..3 {
                let diff = (combined[(atom, c)] - split[(atom, c)]).abs();
                assert!(diff < 1e-12,
                    "split mismatch: atom={} coord={} combined={:.10} split={:.10} diff={:.2e}",
                    atom, c, combined[(atom, c)], split[(atom, c)], diff);
            }
        }
    }
}
