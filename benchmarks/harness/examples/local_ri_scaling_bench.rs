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
//!   - Domain-dressing tensor norm error ||B_local - B_global|| / ||B_global||
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

    for n_c in [1usize, 2, 3, 4, 6, 8] {
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

        // Global dense dressing: B_global(P, i, a) = sum_Q [V^{-1/2}]_{PQ} (Q | i a)
        // For baseline comparison, compute using V_global^{-1}:
        let mut b_global = Array3::<f64>::zeros((naux, nocc, nvir));
        for i in 0..nocc {
            for a in 0..nvir {
                let col_3c = eri3_loc.slice(s![.., i, a]);
                let col_b = v_global_inv.dot(&col_3c);
                for p in 0..naux {
                    b_global[(p, i, a)] = col_b[p];
                }
            }
        }
        let b_global_norm = b_global.iter().map(|v| v * v).sum::<f64>().sqrt();

        let global_gemm_flops = 2.0 * (naux as f64) * (naux as f64) * (nocc * nvir) as f64;

        println!("### alkane_{n_c}  (C_{n_c}H_{})  nocc={nocc}  nvir={nvir}  naux={naux}  global_norm={b_global_norm:.4}", 2 * n_c + 2);
        println!(
            "{:>8}  {:>10}  {:>10}  {:>14}  {:>10}  {:>12}",
            "R_cut(B)", "avg|P_i|", "retention", "Local FLOPs", "FLOP ratio", "b_tensor_err"
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

            // Perform Local Domain Dressing and compute local B_local tensor
            let mut b_local = Array3::<f64>::zeros((naux, nocc, nvir));
            let mut local_flops = 0.0f64;

            for dom in &local_domains {
                let i = dom.occ_idx;
                let m_i = dom.aux_fn_indices.len();
                if m_i == 0 {
                    continue;
                }

                local_flops += 2.0 * (m_i as f64) * (m_i as f64) * (nvir as f64);

                for a in 0..nvir {
                    // Extract local 3c vector
                    let mut vec_3c_local = ndarray::Array1::<f64>::zeros(m_i);
                    for (p_loc, &p_glob) in dom.aux_fn_indices.iter().enumerate() {
                        vec_3c_local[p_loc] = eri3_loc[(p_glob, i, a)];
                    }

                    // Local inverse solve: b_vec_local = V_local^{-1} * vec_3c_local
                    let b_vec_local = dom.v_inv_local.dot(&vec_3c_local);

                    for (p_loc, &p_glob) in dom.aux_fn_indices.iter().enumerate() {
                        b_local[(p_glob, i, a)] = b_vec_local[p_loc];
                    }
                }
            }

            // Calculate tensor error ||B_local - B_global|| / ||B_global||
            let mut diff_sq = 0.0f64;
            for p in 0..naux {
                for i in 0..nocc {
                    for a in 0..nvir {
                        let d = b_local[(p, i, a)] - b_global[(p, i, a)];
                        diff_sq += d * d;
                    }
                }
            }
            let b_err = diff_sq.sqrt() / b_global_norm;
            let flop_ratio = global_gemm_flops / local_flops.max(1.0);

            println!(
                "{r_cut:>8.1}  {avg_domain_size:>10.1}  {aux_retention_pct:>9.1}%  {local_flops:>14.3e}  {flop_ratio:>9.2}x  {b_err:>12.3e}"
            );
        }
        println!();
    }
}
