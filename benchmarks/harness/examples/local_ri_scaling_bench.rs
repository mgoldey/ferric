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
//! Run:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-benchmarks \
//!     --example local_ri_scaling_bench

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::dipole;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_mp2::boys::boys_localize;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{s, Array2, Array3};
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

    for n_c in [1usize, 2, 3, 4, 6, 8, 12, 16] {
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

        // 3. Compute Global 2-center metric V_global and its inverse V_global^{-1}
        let v_global = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
        let v_global_inv = v_global.inv().unwrap();

        // Compute 3-center integrals in localized occupied basis: (P | i_loc a_can)
        // (naux, nocc, nvir) ERI tensor
        let g_3c = match threeindex::eri3_tensor(op, &obs, &dfbs) {
            Ok(tensor) => tensor,
            Err(e) => {
                println!("alkane_{n_c}: 3c integral failed: {e}\n");
                continue;
            }
        };

        // Transform 3c integral bra from AO to (i_loc, a_can)
        let mut eri3_loc = Array3::<f64>::zeros((naux, nocc, nvir));
        for p in 0..naux {
            let mat_ao = g_3c.slice(s![p, .., ..]).to_owned();
            let mat_loc = c_occ_loc.t().dot(&mat_ao).dot(&c_vir_can);
            for i in 0..nocc {
                for a in 0..nvir {
                    eri3_loc[(p, i, a)] = mat_loc[(i, a)];
                }
            }
        }

        // Reshape (P, i, a) -> (P, ia) once; every fit below is a GEMM on this.
        let nov = nocc * nvir;
        let a2 = eri3_loc.to_shape((naux, nov)).unwrap().to_owned();

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

        println!("### alkane_{n_c}  (C_{n_c}H_{})  nocc={nocc}  nvir={nvir}  naux={naux}  |G_exact|={g_exact_norm:.4}", 2 * n_c + 2);
        println!(
            "{:>8}  {:>10}  {:>10}  {:>14}  {:>10}  {:>10}  {:>12}  {:>12}",
            "R_cut(B)", "avg|P_i|", "retention", "Local FLOPs", "FLOP ratio", "c_err", "g_err_naive", "g_err_robust"
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

            println!(
                "{r_cut:>8.1}  {avg_domain_size:>10.1}  {aux_retention_pct:>9.1}%  {local_flops:>14.3e}  {flop_ratio:>9.2}x  {c_err:>10.3e}  {g_err_naive:>12.3e}  {g_err_robust:>12.3e}"
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
