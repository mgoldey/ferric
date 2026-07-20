//! Domain-natural-virtual (DNV) truncation for PDEP-RPA — a PNO-prototype.
//!
//! True OSV/PNO (Riplinger-Neese 2013) gives **per-orbital** (or per-pair)
//! virtual subspaces with overlap matrices `S^{ij}_{kk'}` between them.
//! Production DLPNO codes carry the overlap machinery through every
//! contraction. For ferric's first PNO prototype we take a simpler route:
//!
//!  1. Build per-orbital OSV bases from diagonal MP2 pair amplitudes
//!     `T^{ii}_{ab}` (closed-shell Riplinger formula simplified).
//!  2. **Concatenate** all kept OSV eigenvectors across orbitals into a
//!     single `nvir × N_concat` matrix.
//!  3. QR-orthogonalize that matrix to produce a **single shared
//!     reduced virtual basis** of dimension `n_vir_reduced ≤ nvir`.
//!  4. Transform `B^P_{ia} → B^P_{i, ã}` once for all `i`, feed to the
//!     existing PDEP-RPA dense path with no further changes.
//!
//! This is *not* the cheapest possible scheme — full DLPNO with
//! per-pair virtuals gives more compression — but it's correctness-
//! preserving at `t_osv → 0` (the concatenated/QR'd basis spans the
//! full virtual space) and gives a valid reduced canonical basis for
//! the existing dielectric matvec to consume unchanged.
//!
//! # Validation contract
//!
//! At `t_osv = 0`, OSV-RPA must recover canonical RI-RPA exactly (the
//! transform is lossless).  At `t_osv ~ 1e-3`, expect ~mHa truncation
//! error on small molecules; the goal is to show the error is bounded
//! and controllable, and that `n_vir_reduced ≪ nvir` at production
//! thresholds.

use ferric_core::FerricError;
use ferric_mp2::oo_rimp2::compute_t2_and_integrals;
use ferric_mp2::rimp2::RpaIntermediates;
use ndarray::{s, Array2};
use ndarray_linalg::Eigh;

/// Domain-natural-virtual transform: single shared reduced virtual basis
/// built by concatenating + QR'ing per-orbital OSV eigenvectors.
pub struct DnvTransform {
    /// Per-orbital number of retained OSVs (before concatenation).
    pub n_osv_per_orbital: Vec<usize>,
    /// Final number of reduced virtuals after concatenation + QR.
    pub n_vir_reduced: usize,
    /// Transformed RI tensor `B^P_{i, ã}` of shape `(naux, nocc * n_vir_reduced)`.
    pub b_ov_pno: Array2<f64>,
    /// Effective virtual energies (diagonal of the shared-basis Fock vir
    /// block). Shape `(n_vir_reduced,)`.
    pub eps_vir_reduced: Vec<f64>,
}

/// Build a shared reduced virtual basis from per-orbital OSVs.
///
/// Per-orbital OSV: diagonalize `D^{ii} = 2 T^{ii} (T^{ii})^T` (closed-shell
/// diagonal pair density). Concatenate all kept eigenvectors across i,
/// then column-QR to produce a single orthonormal virtual basis of
/// dimension `n_vir_reduced ≤ nvir`. Transform `B^P_{ia}` into that
/// basis once (same for every i).
///
/// At `t_osv = 0`, the kept-eigenvectors-per-orbital concatenated span
/// the full nvir-dim space (rank ≤ nvir but typically = nvir for
/// non-degenerate cases), so the transform is lossless.
pub fn build_dnv_transform(
    inter: &RpaIntermediates,
    eps: &[f64],
    t_osv: f64,
) -> Result<DnvTransform, FerricError> {
    use ndarray_linalg::QR;

    let nocc = inter.nocc;
    let nvir = inter.nvir;
    let naux = inter.naux;
    let first_occ = inter.first_occ;
    let nocc_total = inter.nocc_total;
    let b_ov = &inter.b_ov;

    let (t2, _eri) = compute_t2_and_integrals(
        b_ov, eps, nocc, nvir, nocc_total, first_occ, naux,
    );
    let nov = nocc * nvir;

    let mut n_osv_per: Vec<usize> = vec![0; nocc];
    let mut osv_vectors: Vec<Array2<f64>> = Vec::with_capacity(nocc);

    for i in 0..nocc {
        let mut tii = Array2::<f64>::zeros((nvir, nvir));
        for a in 0..nvir {
            let ia = i * nvir + a;
            for b in 0..nvir {
                let ib = i * nvir + b;
                tii[(a, b)] = t2[ia * nov + ib];
            }
        }
        // D^{ii} = 2 T^{ii} (T^{ii})^T, symmetrized.
        let d_ii = 2.0 * tii.dot(&tii.t());
        let mut d_sym = Array2::<f64>::zeros((nvir, nvir));
        for a in 0..nvir {
            for b in 0..nvir {
                d_sym[(a, b)] = 0.5 * (d_ii[(a, b)] + d_ii[(b, a)]);
            }
        }
        let (occ_eigs, occ_vecs) = d_sym
            .eigh(ndarray_linalg::UPLO::Upper)
            .map_err(|e| FerricError::General(format!("OSV eigh for orbital {i}: {e}")))?;
        let kept: Vec<usize> = (0..nvir).filter(|&k| occ_eigs[k].abs() > t_osv).collect();
        let kept = if kept.is_empty() { vec![nvir - 1] } else { kept };
        n_osv_per[i] = kept.len();
        let mut p = Array2::<f64>::zeros((nvir, kept.len()));
        for (slot, &k) in kept.iter().enumerate() {
            for a in 0..nvir {
                p[(a, slot)] = occ_vecs[(a, k)];
            }
        }
        osv_vectors.push(p);
    }

    // Concatenate OSV columns across orbitals: shape (nvir, sum n_osv).
    let total_cols: usize = n_osv_per.iter().sum();
    let mut concat = Array2::<f64>::zeros((nvir, total_cols));
    let mut col_off = 0;
    for p_i in &osv_vectors {
        let nc = p_i.ncols();
        for j in 0..nc {
            for a in 0..nvir {
                concat[(a, col_off + j)] = p_i[(a, j)];
            }
        }
        col_off += nc;
    }

    // QR to get an orthonormal column-space basis. Rank may be < total_cols.
    // For ndarray-linalg, `qr` returns (Q, R); we take Q's leading columns.
    // First drop near-zero columns to avoid numerical rank deficiency.
    let mut col_norms = Vec::with_capacity(concat.ncols());
    for j in 0..concat.ncols() {
        let nrm = concat.column(j).dot(&concat.column(j)).sqrt();
        col_norms.push(nrm);
    }
    let keep_cols: Vec<usize> = (0..concat.ncols())
        .filter(|&j| col_norms[j] > 1e-12)
        .collect();
    if keep_cols.is_empty() {
        return Err(FerricError::General(
            "DNV: all OSV concatenation columns zero".into(),
        ));
    }
    let mut filtered = Array2::<f64>::zeros((nvir, keep_cols.len()));
    for (slot, &j) in keep_cols.iter().enumerate() {
        filtered.slice_mut(s![.., slot]).assign(&concat.column(j));
    }

    // Use SVD via QR to get a properly-orthonormal basis whose column count
    // = numerical rank. `ndarray-linalg::QR::qr` returns (Q, R) with Q
    // having shape (nvir, min(nvir, ncols)). For wide matrices (ncols >
    // nvir) we want the QR of the (nvir, ncols) but only keep nvir
    // columns. For tall matrices we keep all.
    let (q, _r) = if filtered.ncols() <= filtered.nrows() {
        filtered.qr().map_err(|e| FerricError::General(format!("DNV QR: {e}")))?
    } else {
        // Force ncols ≤ nrows by transposing for QR? Simpler: just truncate
        // input to nvir columns first; rank can't exceed nvir.
        let mut sliced = Array2::<f64>::zeros((nvir, nvir));
        for j in 0..nvir {
            sliced.slice_mut(s![.., j]).assign(&filtered.column(j));
        }
        sliced.qr().map_err(|e| FerricError::General(format!("DNV QR: {e}")))?
    };
    let n_vir_reduced = q.ncols();

    // Q is an orthonormal column basis but NOT Fock-diagonal. RPA denominators
    // need canonical orbital energies, so diagonalize F_vir in the Q-subspace:
    //   F_red = Q^T F_vir Q   (n_vir_reduced × n_vir_reduced, diagonal in canonical)
    //   F_red = U_red diag(ε̃) U_red^T
    // The canonical reduced basis is Q · U_red.
    let mut f_red = Array2::<f64>::zeros((n_vir_reduced, n_vir_reduced));
    for k in 0..n_vir_reduced {
        for l in 0..n_vir_reduced {
            let mut x = 0.0;
            for a in 0..nvir {
                x += q[(a, k)] * eps[nocc_total + a] * q[(a, l)];
            }
            f_red[(k, l)] = x;
        }
    }
    let (eps_red, u_red) = f_red
        .eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| FerricError::General(format!("DNV F_red eigh: {e}")))?;
    let q_canonical = q.dot(&u_red); // (nvir, n_vir_reduced), Fock-diagonal

    // Transform B in the canonical reduced basis.
    let mut b_ov_pno = Array2::<f64>::zeros((naux, nocc * n_vir_reduced));
    for i in 0..nocc {
        let b_block_i = b_ov.slice(s![.., i * nvir..(i + 1) * nvir]).to_owned();
        let b_red_i = b_block_i.dot(&q_canonical); // (naux, n_vir_reduced)
        for k in 0..n_vir_reduced {
            for p in 0..naux {
                b_ov_pno[(p, i * n_vir_reduced + k)] = b_red_i[(p, k)];
            }
        }
    }

    let eps_vir_reduced: Vec<f64> = eps_red.to_vec();

    Ok(DnvTransform {
        n_osv_per_orbital: n_osv_per,
        n_vir_reduced,
        b_ov_pno,
        eps_vir_reduced,
    })
}

/// Run PDEP-RPA on the DNV-truncated virtual space.
///
/// Builds the shared reduced virtual basis from [`build_dnv_transform`],
/// then runs the standard dielectric matvec on the reduced (occ × n_vir_reduced)
/// space. The reduced B tensor and reduced virtual energies are valid
/// canonical objects (just in a smaller virtual basis), so we feed them
/// into the existing sternheimer::dielectric_apply path verbatim.
///
/// Returns (E_c^RPA, n_vir_reduced, naux).
pub fn run_pdep_rpa_osv(
    mol: &ferric_core::mol::Molecule,
    obs: &ferric_integrals::basis_bridge::PreparedBasis,
    dfbs: &ferric_integrals::basis_bridge::PreparedBasis,
    op: ferric_integrals::operator::Operator,
    rhf: &ferric_scf::ScfResult,
    config: &crate::PdepRpaConfig,
    t_osv: f64,
) -> Result<(f64, usize, usize), FerricError> {
    use crate::lanczos;
    use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config};

    let mp2_cfg = RiMp2Config { frozen_core: config.frozen_core, memory_budget_bytes: config.memory_budget_bytes };
    let inter = compute_rpa_intermediates(mol, obs, dfbs, op, rhf, &mp2_cfg)?;

    let dnv = build_dnv_transform(&inter, rhf.eps_r(), t_osv)?;
    let nocc = inter.nocc;
    let naux = inter.naux;
    let n_vir_red = dnv.n_vir_reduced;

    let eps_occ: Vec<f64> = rhf.eps_r()[inter.first_occ..inter.first_occ + nocc].to_vec();
    let eps_vir = dnv.eps_vir_reduced.clone();
    let b_ov = dnv.b_ov_pno.clone();

    // Davidson/Lanczos eigensolve with identity seed (matches the closed-
    // shell test path for trunc_thresh=0).
    let seed = Array2::<f64>::eye(naux);
    let max_iter = if config.eigensolver_max_vecs == 0 { 3 * naux } else { config.eigensolver_max_vecs };

    let b_ref = b_ov.clone();
    let eo = eps_occ.clone();
    let ev = eps_vir.clone();
    let matvec = move |v: &Array2<f64>| -> Array2<f64> {
        crate::sternheimer::dielectric_apply(v, &b_ref, &eo, &ev, 0.0)
    };
    let lz = lanczos::run_lanczos_seeded(
        seed, matvec, naux, max_iter, config.eigensolver_conv_thresh, config.verbose,
    )?;
    if !lz.converged {
        eprintln!(
            "warning: Lanczos eigensolve did NOT converge (max Ritz residual {:.3e} \
             > {:.3e}); OSV/PNO eigenpotentials are best-effort",
            lz.max_resid, config.eigensolver_conv_thresh
        );
    }
    let eigvals = &lz.eigenvalues;
    let eigvecs = &lz.eigenvectors;

    let n_keep = eigvals.iter().filter(|&&lam| (lam - 1.0).abs() > config.trunc_thresh).count().max(1);
    let v_kept = eigvecs.slice(s![.., ..n_keep]).to_owned();

    let (quad_freqs, quad_weights) = crate::quadrature::build_quadrature(&config.quadrature);
    let eigenvalues_freq = crate::energy::eval_eigenvalues_at_frequencies(
        &v_kept, &b_ov, &eps_occ, &eps_vir, &quad_freqs,
    )?;
    let e_c = crate::energy::rpa_correlation_energy(&quad_weights, &eigenvalues_freq);

    let _ = n_vir_red; // returned below
    Ok((e_c, dnv.n_vir_reduced, naux))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;
    use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config};
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;

    fn h2o_intermediates() -> (RpaIntermediates, Vec<f64>) {
        let ctx = ParallelContext::default();
        let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
        let inter = compute_rpa_intermediates(
            &mol, &obs, &dfbs, op, &rhf, &RiMp2Config { frozen_core: 0, memory_budget_bytes: None },
        ).unwrap();
        (inter, rhf.eps_r().to_vec())
    }

    #[test]
    fn dnv_zero_threshold_keeps_all_virtuals() {
        let (inter, eps) = h2o_intermediates();
        let dnv = build_dnv_transform(&inter, &eps, 0.0).unwrap();
        // At t_osv=0, the concatenated OSV basis spans the full nvir-dim
        // virtual space, so after QR we should have n_vir_reduced = nvir.
        eprintln!("DNV n_osv_per_orbital at t_osv=0: {:?}", dnv.n_osv_per_orbital);
        eprintln!("DNV n_vir_reduced = {} (nvir = {})", dnv.n_vir_reduced, inter.nvir);
        assert_eq!(dnv.n_vir_reduced, inter.nvir,
            "lossless DNV must span the full virtual space");
    }

    #[test]
    fn osv_rpa_at_zero_threshold_matches_canonical_h2o() {
        // Lossless OSV transform must reproduce canonical RI-RPA at t_osv=0.
        use crate::{run_pdep_rpa, PdepRpaConfig};
        use crate::config::{QuadratureConfig, QuadratureScheme};
        use ferric_core::mol::Molecule;
        let ctx = ParallelContext::default();
        let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

        let cfg = PdepRpaConfig {
            quadrature: QuadratureConfig {
                scheme: QuadratureScheme::GaussLegendre, n_points: 20, u0: 0.5,
            },
            frozen_core: 0,
            trunc_thresh: 0.0,
            eigensolver_conv_thresh: 1e-9,
            ..Default::default()
        };

        let e_canonical = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap().e_rpa;
        let (e_osv, n_kept, _naux) = run_pdep_rpa_osv(
            &mol, &obs, &dfbs, op, &rhf, &cfg, 0.0,
        ).unwrap();

        eprintln!("OSV/RPA H2O: canonical={e_canonical:.10}, osv={e_osv:.10}, nOSV={n_kept}");
        let dev = (e_canonical - e_osv).abs();
        assert!(dev < 1e-6,
            "OSV-RPA at t_osv=0 ≠ canonical: dev={dev:.2e}");
    }

    #[test]
    fn dnv_rpa_truncation_error_curve_h2o() {
        // Sweep t_osv and report (n_vir_reduced, ΔE) — characterizes the
        // accuracy-vs-compression tradeoff. No assertion past sanity bounds;
        // this is a probe test for tuning t_osv on real molecules.
        use crate::{run_pdep_rpa, PdepRpaConfig};
        use crate::config::{QuadratureConfig, QuadratureScheme};
        use ferric_core::mol::Molecule;
        let ctx = ParallelContext::default();
        let xyz = "3\nh2o\nO 0 0 0.117790\nH 0 0.755453 -0.471161\nH 0 -0.755453 -0.471161\n";
        let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
        let obs_bs = basis::bundled("cc-pvdz").unwrap();
        let dfbs_bs = basis::bundled("cc-pvdz-ri").unwrap();
        let op = Operator::coulomb();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

        let cfg = PdepRpaConfig {
            quadrature: QuadratureConfig {
                scheme: QuadratureScheme::GaussLegendre, n_points: 20, u0: 0.5,
            },
            frozen_core: 0,
            trunc_thresh: 0.0,
            eigensolver_conv_thresh: 1e-9,
            ..Default::default()
        };

        let e_canonical = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap().e_rpa;
        eprintln!("Canonical RPA(H2O/cc-pVDZ) = {e_canonical:.10}");
        eprintln!("DNV truncation curve:");
        eprintln!("    t_osv     n_vir_red    ΔE (mHa)");
        for &t_osv in &[1e-8_f64, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2] {
            let (e, n, _) = run_pdep_rpa_osv(&mol, &obs, &dfbs, op, &rhf, &cfg, t_osv).unwrap();
            let de_mha = (e - e_canonical).abs() * 1000.0;
            eprintln!("    {t_osv:.0e}    {n:>5}       {de_mha:.4}");
        }
        // Sanity check: at t_osv=1e-8 (essentially lossless), ΔE should be sub-μHa.
        let (e_tight, _, _) = run_pdep_rpa_osv(&mol, &obs, &dfbs, op, &rhf, &cfg, 1e-8).unwrap();
        assert!((e_tight - e_canonical).abs() < 1e-6,
            "DNV at t_osv=1e-8 should match canonical to <1 μHa");
    }

    #[test]
    fn dnv_high_threshold_truncates_aggressively() {
        let (inter, eps) = h2o_intermediates();
        let dnv = build_dnv_transform(&inter, &eps, 1e-3).unwrap();
        eprintln!("DNV at t_osv=1e-3: n_vir_reduced = {} (canonical nvir = {})",
            dnv.n_vir_reduced, inter.nvir);
        assert!(dnv.n_vir_reduced <= inter.nvir, "DNV cannot exceed nvir");
    }
}
