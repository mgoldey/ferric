//! Domain-Based Local RI Inverse Spike & Scaling Benchmark.
//!
//! Evaluates the structural performance and numerical accuracy of replacing
//! global inverse metric V^{-1} (which suffers from Green's function matrix fill-in)
//! with domain-local inverses V_{P_i}^{-1} constructed around Boys-localized
//! occupied orbital centers R_i = <i|r|i>.
//!
//! Key Metrics Tracked:
//!   - Average local domain size |P_i| vs molecule size (Alkane series C1..C8)
//!   - FLOP ratio for dressing: Local Domain Dressing vs Global Dense Dressing
//!   - Fit-coefficient norm error ||c_local - c_global|| / ||c_global||
//!   - Assembled-integral error ||G_x - G_exact||_F / ||G_exact||_F for the
//!     (ia|jb) matrix, two assemblies from the SAME truncated coefficients:
//!     naive one-sided G_1 = A^T c_local (FIRST order in dc) and robust
//!     (Dunlap) G_rob = G_1 + G_1^T - c^T V c (SECOND order in dc)
//!
//! Artifact hypothesis (stated before measuring): if the robust assembly is
//! implemented correctly, g_err_robust should sit at ~(g_err_naive)^2 scale
//! and both must be exactly 0 at 100% domain retention; a broken construction
//! would make robust track naive (still first order) or exceed it.
//!
//! MP2 ENERGY error (all-electron, closed-shell): the domain fit lives in the
//! Boys-localized occupied basis, but localization is a unitary rotation
//! within the occupied space, so the fitted coefficients back-transform
//! exactly to the canonical basis (U = C_can^T S C_loc) where the standard
//! denominator applies. Energy hypothesis (pre-registered): E is linear in
//! the integral error at leading order, so dE_naive should scale like c_err
//! and dE_robust like c_err^2; dE_robust tracking c_err FIRST order would
//! mean the tensor-level second-order property does not transfer to E, which
//! would itself be a reportable (negative) finding. Anchor: both dE must be
//! ~0 at 100% retention, and E_exact must match ferric's ri_mp2.
//!
//! Run:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-benchmarks \
//!     --example local_ri_scaling_bench

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::{dipole, overlap};
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_mp2::boys::boys_localize;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{s, Array2};
use ndarray_linalg::Inverse;

const OBS_NAME: &str = "cc-pvdz";
const AUX_NAME: &str = "cc-pvdz-ri";

/// Distance cutoffs (in Bohr) tested for local auxiliary domain construction.
const R_CUTOFFS_BOHR: &[f64] = &[6.0, 8.0, 10.0, 12.0, 15.0, 20.0];

/// Above this nov = nocc*nvir the dense (ia|jb) matrices are skipped (nov^2
/// doubles each — ~3.7 GB at C16) and only the trace-based errors are
/// reported. Below it, BOTH are computed and cross-checked, so the trace
/// construction is validated by an independent construction on every system
/// where the dense path is affordable (through octane, nov = 5577).
const DENSE_XCHECK_MAX_NOV: usize = 6000;

/// Frobenius norm. ndarray's .iter() walks logical order regardless of memory
/// layout, so these are safe on mixed-layout GEMM outputs (see the ndarray
/// dot-layout convention in CLAUDE.md).
fn frob(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// tr(a . b) = sum_PQ a[P,Q] b[Q,P] without forming the product matrix.
fn tr_prod(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a * &b.t()).sum()
}

fn frob_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Rotate the occupied index of a (naux, nocc*nvir) fitted-coefficient array
/// from the localized to the canonical basis: Y[P,i,a] = sum_i' U[i,i'] X[P,i',a].
fn occ_rotate(
    x: &Array2<f64>,
    u_rot: &Array2<f64>,
    naux: usize,
    nocc: usize,
    nvir: usize,
) -> Array2<f64> {
    let x3 = x.to_shape((naux, nocc, nvir)).unwrap();
    // (i', P, a) standard layout, then one (nocc x nocc) x (nocc x naux*nvir) GEMM.
    let xp = x3
        .permuted_axes([1, 0, 2])
        .as_standard_layout()
        .to_owned()
        .to_shape((nocc, naux * nvir))
        .unwrap()
        .to_owned();
    let y = u_rot.dot(&xp);
    y.to_shape((nocc, naux, nvir))
        .unwrap()
        .permuted_axes([1, 0, 2])
        .as_standard_layout()
        .to_owned()
        .to_shape((naux, nocc * nvir))
        .unwrap()
        .to_owned()
}

/// Closed-shell MP2 energy from a fitted (ia|jb), assembled per occupied i:
/// G_i = A_i^T c_fit (one-sided). When `z_robust = Some(Z)` with
/// Z = A - V c_fit, ALSO returns the robust-Dunlap energy from
/// G_i^rob = G_i + c_i^T Z. Exact reference: c_fit = V^{-1} A, z None.
/// rayon over i; GEMMs are matrixmultiply (no OpenBLAS), safe under rayon.
fn mp2_energies(
    a_can: &Array2<f64>,
    c_fit: &Array2<f64>,
    z_robust: Option<&Array2<f64>>,
    eps: &[f64],
    nocc: usize,
    nvir: usize,
) -> (f64, Option<f64>) {
    use rayon::prelude::*;
    let sums: Vec<(f64, f64)> = (0..nocc)
        .into_par_iter()
        .map(|i| {
            let a_i = a_can.slice(s![.., i * nvir..(i + 1) * nvir]);
            let g_base = a_i.t().dot(c_fit); // (nvir, nov)
            let g_rob = z_robust.map(|z| {
                let c_i = c_fit.slice(s![.., i * nvir..(i + 1) * nvir]);
                &g_base + &c_i.t().dot(z)
            });
            let mut e_b = 0.0;
            let mut e_r = 0.0;
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let d = eps[i] + eps[j] - eps[nocc + a] - eps[nocc + b];
                        let dir = g_base[(a, j * nvir + b)];
                        let exc = g_base[(b, j * nvir + a)];
                        e_b += dir * (2.0 * dir - exc) / d;
                        if let Some(g) = &g_rob {
                            let dirr = g[(a, j * nvir + b)];
                            let excr = g[(b, j * nvir + a)];
                            e_r += dirr * (2.0 * dirr - excr) / d;
                        }
                    }
                }
            }
            (e_b, e_r)
        })
        .collect();
    let e_base: f64 = sums.iter().map(|s| s.0).sum();
    let e_rob: f64 = sums.iter().map(|s| s.1).sum();
    (e_base, z_robust.map(|_| e_rob))
}

/// Represents an auxiliary domain for localized occupied orbital `i`.
struct LocalDomainSpec {
    occ_idx: usize,
    aux_fn_indices: Vec<usize>,
    v_inv_local: Array2<f64>,
}

fn main() {
    println!("# ==========================================================================");
    println!("# Local RI Inverse (Domain-Based Density Fitting) Scaling Benchmark");
    println!("# Basis: {OBS_NAME} / Aux: {AUX_NAME}");
    println!("# ==========================================================================\n");

    for n_c in [1usize, 2, 3, 4, 6, 8, 12, 16, 20] {
        let path = format!("testdata/molecules/alkane_{n_c}.xyz");
        let Ok(mol) = Molecule::load_xyz(&path) else {
            println!("alkane_{n_c}: SKIPPED (file missing)\n");
            continue;
        };

        let obs_bs = basis::bundled(OBS_NAME).unwrap();
        let aux_bs = basis::bundled(AUX_NAME).unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();

        let nbas = obs.nbasis();
        let naux = dfbs.nbasis();
        let nocc = mol.nelec() as usize / 2;
        let nvir = nbas - nocc;

        // 1. Run RHF to obtain canonical MOs
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let ctx = ParallelContext::default();
        let scf_cfg = RhfConfig {
            density_conv: 1e-8,
            ..Default::default()
        };
        let rhf = match solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg) {
            Ok(res) => res,
            Err(e) => {
                println!("alkane_{n_c}: SCF failed: {e}\n");
                continue;
            }
        };

        let c_occ_can = rhf.mos_r().slice(s![.., ..nocc]).to_owned();
        let c_vir_can = rhf.mos_r().slice(s![.., nocc..]).to_owned();

        // 2. Perform Boys localization on occupied orbitals
        let dip = dipole(&obs, [0.0, 0.0, 0.0]).unwrap();
        let boys = boys_localize(&c_occ_can, &dip, 200);
        if !boys.converged {
            println!("alkane_{n_c}: Boys localization did not converge\n");
            continue;
        }
        let c_occ_loc = boys.c_loc;
        let centers = boys.centers; // (nocc x 3)

        // Occ-space rotation c_loc = C_can U  =>  U = C_can^T S c_loc. Guard
        // orthogonality so a wrong-direction U fails loudly, not silently.
        let s_ao = overlap(&obs);
        let u_rot = c_occ_can.t().dot(&s_ao).dot(&c_occ_loc);
        let ortho_dev = frob_diff(&u_rot.t().dot(&u_rot), &Array2::eye(nocc));
        if ortho_dev > 1e-8 {
            println!("alkane_{n_c}: U not orthogonal (dev {ortho_dev:.3e}) — skipping\n");
            continue;
        }
        let eps: Vec<f64> = rhf.eps_r().to_vec();

        // 3. Compute Global 2-center metric V_global and its inverse V_global^{-1}
        let v_global = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
        let v_global_inv = v_global.inv().unwrap();

        // Transform 3c integrals to (i_loc, a_can) for the domain fits AND to
        // (i_can, a_can) for the MP2 energy. eri3-limit fix: blocked AO->MO
        // transforms (bit-identical to the dense path; never materialize the
        // naux*nbas^2 AO tensor, ~13 GB at C32/cc-pVDZ).
        let eri3_loc =
            ferric_mp2::rimp2::eri3_mo_ov_blocked(op, &obs, &dfbs, &c_occ_loc, &c_vir_can, 2 << 30).unwrap();
        let eri3_can =
            ferric_mp2::rimp2::eri3_mo_ov_blocked(op, &obs, &dfbs, &c_occ_can, &c_vir_can, 2 << 30).unwrap();

        // Reshape (P, i, a) -> (P, ia) once; every fit below is a GEMM on this.
        let nov = nocc * nvir;
        let a2 = eri3_loc.to_shape((naux, nov)).unwrap().to_owned();
        let a2_can = eri3_can.to_shape((naux, nov)).unwrap().to_owned();
        drop(eri3_loc);
        drop(eri3_can);

        // Canonical exact RI-MP2 energy (all-electron) — the reference every
        // truncated-fit energy below is measured against.
        let c_glob_can = v_global_inv.dot(&a2_can);
        let (e_exact, _) = mp2_energies(&a2_can, &c_glob_can, None, &eps, nocc, nvir);

        // Global fit coefficients c_global = V^{-1} A.
        let c_global = v_global_inv.dot(&a2);
        let c_global_norm = frob(&c_global);

        // Every Frobenius error below reduces EXACTLY to naux x naux traces
        // (A = V c_global, S_X = X X^T):
        //   G_exact = c^T V c            =>  |G_exact|^2      = tr(S_A S_c)
        //   G_1   - G_exact = A^T dc     =>  |G_1 - G_ex|^2   = tr(S_A S_dc)
        //   G_rob - G_exact = -dc^T V dc =>  |G_rob - G_ex|^2 = tr((V S_dc)^2)
        // so the nov^2 integral matrices are never materialized. The dense-G
        // path is kept below DENSE_XCHECK_MAX_NOV as an independent
        // construction cross-check of these identities.
        let s_a = a2.dot(&a2.t());
        let s_c = c_global.dot(&c_global.t());
        let g_exact_norm = tr_prod(&s_a, &s_c).max(0.0).sqrt();

        let g_exact_dense = if nov <= DENSE_XCHECK_MAX_NOV {
            Some(a2.t().dot(&c_global))
        } else {
            None
        };
        let mut xcheck_max_rel_dev = 0.0f64;

        let global_gemm_flops = 2.0 * (naux as f64) * (naux as f64) * nov as f64;

        println!("### alkane_{n_c}  (C_{n_c}H_{})  nocc={nocc}  nvir={nvir}  naux={naux}  |G_exact|={g_exact_norm:.4}  E_corr={e_exact:.9}", 2 * n_c + 2);
        println!(
            "{:>8}  {:>10}  {:>10}  {:>14}  {:>10}  {:>10}  {:>12}  {:>12}  {:>12}  {:>12}",
            "R_cut(B)", "avg|P_i|", "retention", "Local FLOPs", "FLOP ratio", "c_err", "g_err_naive", "g_err_robust", "dE_naive(Ha)", "dE_rob(Ha)"
        );

        let aux_shell_centers = dfbs.shell_centers();
        let aux_shell_offsets = dfbs.shell_offsets();
        let aux_shell_dims = dfbs.shell_dims();

        for &r_cut in R_CUTOFFS_BOHR {
            let mut local_domains = Vec::with_capacity(nocc);
            let mut total_aux_fns_kept = 0usize;

            for i in 0..nocc {
                let center_i = centers.row(i);
                let mut aux_fns = Vec::new();

                for s_aux in 0..dfbs.nshells() {
                    let shell_c = aux_shell_centers[s_aux];
                    let dist = ((center_i[0] - shell_c[0]).powi(2)
                        + (center_i[1] - shell_c[1]).powi(2)
                        + (center_i[2] - shell_c[2]).powi(2))
                    .sqrt();

                    if dist <= r_cut {
                        let f_start = aux_shell_offsets[s_aux];
                        let f_count = aux_shell_dims[s_aux];
                        for f in f_start..(f_start + f_count) {
                            aux_fns.push(f);
                        }
                    }
                }

                let m_i = aux_fns.len();
                total_aux_fns_kept += m_i;

                // Extract sub-block matrix V_domain (m_i x m_i)
                let mut v_domain = Array2::<f64>::zeros((m_i, m_i));
                for (p_loc, &p_glob) in aux_fns.iter().enumerate() {
                    for (q_loc, &q_glob) in aux_fns.iter().enumerate() {
                        v_domain[(p_loc, q_loc)] = v_global[(p_glob, q_glob)];
                    }
                }

                let v_inv_local = match v_domain.inv() {
                    Ok(inv) => inv,
                    Err(_) => Array2::eye(m_i),
                };

                local_domains.push(LocalDomainSpec {
                    occ_idx: i,
                    aux_fn_indices: aux_fns,
                    v_inv_local,
                });
            }

            let avg_domain_size = total_aux_fns_kept as f64 / nocc as f64;
            let aux_retention_pct = 100.0 * avg_domain_size / naux as f64;

            // Domain-local fit coefficients: zero outside each orbital's domain.
            let mut c_local = Array2::<f64>::zeros((naux, nov));
            let mut local_flops = 0.0f64;

            for dom in &local_domains {
                let i = dom.occ_idx;
                let m_i = dom.aux_fn_indices.len();
                if m_i == 0 {
                    continue;
                }

                local_flops += 2.0 * (m_i as f64) * (m_i as f64) * (nvir as f64);

                for a in 0..nvir {
                    let col = i * nvir + a;
                    let mut vec_3c_local = ndarray::Array1::<f64>::zeros(m_i);
                    for (p_loc, &p_glob) in dom.aux_fn_indices.iter().enumerate() {
                        vec_3c_local[p_loc] = a2[(p_glob, col)];
                    }

                    // Local inverse solve: c_vec = V_domain^{-1} * vec_3c_local
                    let c_vec = dom.v_inv_local.dot(&vec_3c_local);

                    for (p_loc, &p_glob) in dom.aux_fn_indices.iter().enumerate() {
                        c_local[(p_glob, col)] = c_vec[p_loc];
                    }
                }
            }

            let dc = &c_local - &c_global;
            let c_err = frob(&dc) / c_global_norm;

            // Assembled-(ia|jb) errors via the aux-space trace identities:
            //   naive one-sided G_1 = A^T c_local          (first order in dc)
            //   robust (Dunlap) G_rob = G_1 + G_1^T - c^T V c  (second order)
            let s_dc = dc.dot(&dc.t());
            let g_err_naive = tr_prod(&s_a, &s_dc).max(0.0).sqrt() / g_exact_norm;
            let m = v_global.dot(&s_dc);
            let g_err_robust = tr_prod(&m, &m).max(0.0).sqrt() / g_exact_norm;

            // Independent-construction cross-check: dense G matrices.
            if let Some(g_exact) = &g_exact_dense {
                let g_1 = a2.t().dot(&c_local);
                let w = v_global.dot(&c_local);
                let g_sym = c_local.t().dot(&w);
                let g_rob = &g_1 + &g_1.t() - &g_sym;
                let dense_naive = frob_diff(&g_1, g_exact) / g_exact_norm;
                let dense_robust = frob_diff(&g_rob, g_exact) / g_exact_norm;
                for (trace_v, dense_v) in [(g_err_naive, dense_naive), (g_err_robust, dense_robust)] {
                    if dense_v > 1e-8 {
                        xcheck_max_rel_dev = xcheck_max_rel_dev.max((trace_v - dense_v).abs() / dense_v);
                    }
                }
            }

            let flop_ratio = global_gemm_flops / local_flops.max(1.0);

            // MP2 energies from the truncated fit, back-rotated to canonical.
            let c_local_can = occ_rotate(&c_local, &u_rot, naux, nocc, nvir);
            let z_rob = &a2_can - &v_global.dot(&c_local_can);
            let (e_naive, e_robust) =
                mp2_energies(&a2_can, &c_local_can, Some(&z_rob), &eps, nocc, nvir);
            let de_naive = e_naive - e_exact;
            let de_robust = e_robust.unwrap() - e_exact;

            println!(
                "{r_cut:>8.1}  {avg_domain_size:>10.1}  {aux_retention_pct:>9.1}%  {local_flops:>14.3e}  {flop_ratio:>9.2}x  {c_err:>10.3e}  {g_err_naive:>12.3e}  {g_err_robust:>12.3e}  {de_naive:>12.3e}  {de_robust:>12.3e}"
            );
        }
        if g_exact_dense.is_some() {
            println!("# xcheck trace-vs-dense (rows with dense err > 1e-8): max rel dev {xcheck_max_rel_dev:.3e}");
        } else {
            println!("# dense xcheck skipped (nov = {nov} > {DENSE_XCHECK_MAX_NOV}); trace-path errors only");
        }
        println!();
    }
}
