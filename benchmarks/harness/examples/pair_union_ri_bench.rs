//! Pair-union RI fitting-domain probe — the gating measurement identified by
//! the 2026-08-12 adversarial review of wiki/local-ri-robust-domain-fitting.md
//! §7: do PAIR-union aux domains P_ij = P_i ∪ P_j inherit the single-orbital
//! robust-fit behavior as the pair separates?
//!
//! For every occupied pair (i<=j) of Boys-localized orbitals, BOTH densities
//! |ia) and |jb) are fitted on the shared union domain (sub-block V_dom^{-1},
//! so the contraction axis is common — addressing the "B_i^T B_j structurally
//! ill-formed" objection), and the (nvir x nvir) pair block is assembled three
//! ways: exact global fit, naive one-sided, robust Dunlap. Block errors are
//! binned by pair separation R_ij = |<i|r|i> - <j|r|j>|. For systems small
//! enough to hold a (nov x nov) matrix, the full fitted G is assembled in the
//! localized basis, both occupied axes are rotated to canonical
//! (U = C_can^T S C_loc), and the MP2 energy error is computed.
//!
//! Artifact hypotheses (pre-registered):
//!   - real: |P_ij| <= |P_i| + |P_j| bounded; ABSOLUTE robust block error
//!     decays with R_ij (both fitted densities are local); dE stays
//!     quadratic-small like the single-orbital probe.
//!   - broken union construction (index misalignment): errors independent of
//!     separation, or exact only at 100% retention.
//!   - drop-too-aggressive candidate (flagged for independent triage): the
//!     union of two balls EXCLUDES the corridor between distant centers; if
//!     corridor aux functions matter for the fitted (ia|jb) interaction,
//!     distant-pair errors will NOT decay with R_ij. Also: many small distant
//!     pairs could accumulate — the energy measurement (sum over all pairs)
//!     is the guard against per-pair stats hiding an O(nocc^2) tail.
//!
//! Anchors: r_cut = 50 Bohr on butane reproduces the global fit to machine
//! eps in every column; the exact-path E matches production ri_mp2 (checked
//! for the same transform machinery in local_ri_scaling_bench, reused here).
//!
//! Run:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-benchmarks \
//!     --example pair_union_ri_bench

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
use ndarray::{s, Array2, Array3};
use ndarray_linalg::Inverse;

const OBS_NAME: &str = "cc-pvdz";
const AUX_NAME: &str = "cc-pvdz-ri";

/// Pair-separation bin edges in Bohr. First bin is diagonal pairs (R = 0).
const BIN_EDGES: &[f64] = &[1e-6, 2.0, 4.0, 6.0, 9.0, 12.0, 18.0, 30.0];

/// Above this nov = nocc*nvir the (nov x nov) energy assembly is skipped
/// (block-error statistics only). C12 (nov = 12201) is the largest included.
const ENERGY_MAX_NOV: usize = 13000;

fn frob(m: &Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn frob_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Gather the domain rows of `full` (naux x nvir slice) into an (m x nvir)
/// matrix, rows ordered as `idx`.
fn gather_rows(full: &ndarray::ArrayView2<f64>, idx: &[usize]) -> Array2<f64> {
    let nvir = full.ncols();
    let mut out = Array2::<f64>::zeros((idx.len(), nvir));
    for (r, &p) in idx.iter().enumerate() {
        out.row_mut(r).assign(&full.row(p));
    }
    out
}

/// Rotate ONE occupied axis of a (nov x nov) matrix stored as
/// ((nocc,nvir),(nocc,nvir)) from localized to canonical:
/// axis = 0 rotates the row occ index, axis = 1 the column occ index.
/// Y[i,...] = sum_i' U[i,i'] X[i',...]. rayon over column chunks
/// (matrixmultiply GEMMs — no OpenBLAS under rayon).
fn rotate_occ_axis(g: &Array2<f64>, u_rot: &Array2<f64>, nocc: usize, nvir: usize, axis: usize) -> Array2<f64> {
    use rayon::prelude::*;
    let nov = nocc * nvir;
    assert_eq!(g.dim(), (nov, nov));
    let x = if axis == 0 {
        g.clone()
    } else {
        g.t().as_standard_layout().to_owned()
    };
    // x rows are (occ, vir) flattened; view as (nocc, nvir*nov) row-rotation.
    let x2 = x.to_shape((nocc, nvir * nov)).unwrap().to_owned();
    let ncols = nvir * nov;
    let chunk = ncols.div_ceil(24).max(1);
    let mut y2 = Array2::<f64>::zeros((nocc, ncols));
    let col_ranges: Vec<(usize, usize)> = (0..ncols)
        .step_by(chunk)
        .map(|c0| (c0, (c0 + chunk).min(ncols)))
        .collect();
    let pieces: Vec<(usize, Array2<f64>)> = col_ranges
        .par_iter()
        .map(|&(c0, c1)| (c0, u_rot.dot(&x2.slice(s![.., c0..c1]))))
        .collect();
    for (c0, piece) in pieces {
        let c1 = c0 + piece.ncols();
        y2.slice_mut(s![.., c0..c1]).assign(&piece);
    }
    let y = y2.to_shape((nov, nov)).unwrap().to_owned();
    if axis == 0 {
        y
    } else {
        y.t().as_standard_layout().to_owned()
    }
}

/// Closed-shell MP2 energy from a full (nov x nov) canonical (ia|jb) matrix.
fn mp2_energy_from_g(g: &Array2<f64>, eps: &[f64], nocc: usize, nvir: usize) -> f64 {
    let mut e = 0.0;
    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let d = eps[i] + eps[j] - eps[nocc + a] - eps[nocc + b];
                    let dir = g[(i * nvir + a, j * nvir + b)];
                    let exc = g[(i * nvir + b, j * nvir + a)];
                    e += dir * (2.0 * dir - exc) / d;
                }
            }
        }
    }
    e
}

/// Exact canonical MP2 energy from global-fit coefficients (per-i GEMMs,
/// rayon over i; matrixmultiply — no OpenBLAS under rayon).
fn mp2_energy_exact(a_can: &Array2<f64>, c_glob_can: &Array2<f64>, eps: &[f64], nocc: usize, nvir: usize) -> f64 {
    use rayon::prelude::*;
    (0..nocc)
        .into_par_iter()
        .map(|i| {
            let a_i = a_can.slice(s![.., i * nvir..(i + 1) * nvir]);
            let g_i = a_i.t().dot(c_glob_can); // (nvir, nov)
            let mut e = 0.0;
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let d = eps[i] + eps[j] - eps[nocc + a] - eps[nocc + b];
                        let dir = g_i[(a, j * nvir + b)];
                        let exc = g_i[(b, j * nvir + a)];
                        e += dir * (2.0 * dir - exc) / d;
                    }
                }
            }
            e
        })
        .sum()
}

struct BinStat {
    n: usize,
    sum_m: f64,
    sum_gnorm: f64,
    max_abs_naive: f64,
    max_abs_rob: f64,
    sum_abs_rob: f64,
    max_rel_rob: f64,
}

impl BinStat {
    fn new() -> Self {
        BinStat { n: 0, sum_m: 0.0, sum_gnorm: 0.0, max_abs_naive: 0.0, max_abs_rob: 0.0, sum_abs_rob: 0.0, max_rel_rob: 0.0 }
    }
}

fn bin_index(r: f64) -> usize {
    for (k, &edge) in BIN_EDGES.iter().enumerate() {
        if r < edge {
            return k;
        }
    }
    BIN_EDGES.len()
}

fn bin_label(k: usize) -> String {
    if k == 0 {
        "diag".to_string()
    } else if k < BIN_EDGES.len() {
        let lo = if k == 1 { 0.0 } else { BIN_EDGES[k - 1] };
        format!("{:.0}-{:.0}", lo, BIN_EDGES[k])
    } else {
        format!(">{:.0}", BIN_EDGES[BIN_EDGES.len() - 1])
    }
}

fn main() {
    println!("# ==========================================================================");
    println!("# Pair-union RI fitting-domain probe (P_ij = P_i U P_j, robust Dunlap)");
    println!("# Basis: {OBS_NAME} / Aux: {AUX_NAME}");
    println!("# ==========================================================================\n");

    for n_c in [4usize, 8, 12, 16] {
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
        let nov = nocc * nvir;

        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let ctx = ParallelContext::default();
        let scf_cfg = RhfConfig { density_conv: 1e-8, ..Default::default() };
        let rhf = match solve_rhf(&ctx, &mol, &obs, op, &bounds, &scf_cfg) {
            Ok(res) => res,
            Err(e) => {
                println!("alkane_{n_c}: SCF failed: {e}\n");
                continue;
            }
        };

        let c_occ_can = rhf.mos_r().slice(s![.., ..nocc]).to_owned();
        let c_vir_can = rhf.mos_r().slice(s![.., nocc..]).to_owned();
        let eps: Vec<f64> = rhf.eps_r().to_vec();

        let dip = dipole(&obs, [0.0, 0.0, 0.0]).unwrap();
        let boys = boys_localize(&c_occ_can, &dip, 200);
        if !boys.converged {
            println!("alkane_{n_c}: Boys localization did not converge\n");
            continue;
        }
        let c_occ_loc = boys.c_loc;
        let centers = boys.centers;

        let s_ao = overlap(&obs);
        let u_rot = c_occ_can.t().dot(&s_ao).dot(&c_occ_loc);
        let ortho_dev = frob_diff(&u_rot.t().dot(&u_rot), &Array2::eye(nocc));
        assert!(ortho_dev < 1e-8, "U not orthogonal: {ortho_dev:.3e}");

        let v_global = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
        let v_global_inv = v_global.inv().unwrap();

        let g_3c = match threeindex::eri3_tensor(op, &obs, &dfbs) {
            Ok(tensor) => tensor,
            Err(e) => {
                println!("alkane_{n_c}: 3c integral failed: {e}\n");
                continue;
            }
        };
        let mut eri3_loc = Array3::<f64>::zeros((naux, nocc, nvir));
        let mut eri3_can = Array3::<f64>::zeros((naux, nocc, nvir));
        for p in 0..naux {
            let mat_v = g_3c.slice(s![p, .., ..]).dot(&c_vir_can);
            eri3_loc.slice_mut(s![p, .., ..]).assign(&c_occ_loc.t().dot(&mat_v));
            eri3_can.slice_mut(s![p, .., ..]).assign(&c_occ_can.t().dot(&mat_v));
        }
        drop(g_3c);
        let a2 = eri3_loc.to_shape((naux, nov)).unwrap().to_owned();
        let a2_can = eri3_can.to_shape((naux, nov)).unwrap().to_owned();
        drop(eri3_loc);
        drop(eri3_can);

        let c_glob_loc = v_global_inv.dot(&a2);
        let c_glob_can = v_global_inv.dot(&a2_can);
        let e_exact = mp2_energy_exact(&a2_can, &c_glob_can, &eps, nocc, nvir);

        let aux_shell_centers = dfbs.shell_centers();
        let aux_shell_offsets = dfbs.shell_offsets();
        let aux_shell_dims = dfbs.shell_dims();

        let energy_on = nov <= ENERGY_MAX_NOV;
        println!(
            "### alkane_{n_c}  nocc={nocc}  nvir={nvir}  naux={naux}  E_corr={e_exact:.9}  energy_assembly={}",
            if energy_on { "yes" } else { "no (nov too large)" }
        );

        let radii: &[f64] = if n_c == 4 { &[6.0, 10.0, 50.0] } else { &[6.0, 10.0] };

        for &r_cut in radii {
            // Per-orbital aux-function domains.
            let mut orb_domains: Vec<Vec<usize>> = Vec::with_capacity(nocc);
            for i in 0..nocc {
                let ci = centers.row(i);
                let mut fns = Vec::new();
                for s_aux in 0..dfbs.nshells() {
                    let c = aux_shell_centers[s_aux];
                    let d2 = (ci[0] - c[0]).powi(2) + (ci[1] - c[1]).powi(2) + (ci[2] - c[2]).powi(2);
                    if d2.sqrt() <= r_cut {
                        let f0 = aux_shell_offsets[s_aux];
                        fns.extend(f0..f0 + aux_shell_dims[s_aux]);
                    }
                }
                orb_domains.push(fns);
            }

            let mut bins: Vec<BinStat> = (0..=BIN_EDGES.len()).map(|_| BinStat::new()).collect();
            let mut worst: Vec<(f64, f64, f64, usize, usize)> = Vec::new(); // (abs_rob, r, rel_rob, i, j)
            let mut sum_en2 = 0.0f64;
            let mut sum_er2 = 0.0f64;
            let mut sum_gex2 = 0.0f64;
            let mut n_singular = 0usize;

            let mut g_naive_loc = if energy_on { Some(Array2::<f64>::zeros((nov, nov))) } else { None };
            let mut g_rob_loc = if energy_on { Some(Array2::<f64>::zeros((nov, nov))) } else { None };

            for i in 0..nocc {
                for j in i..nocc {
                    // Union domain, sorted + deduped.
                    let mut idx: Vec<usize> = orb_domains[i].iter().chain(orb_domains[j].iter()).copied().collect();
                    idx.sort_unstable();
                    idx.dedup();
                    let m = idx.len();
                    if m == 0 {
                        continue;
                    }

                    let mut v_dom = Array2::<f64>::zeros((m, m));
                    for (r, &p) in idx.iter().enumerate() {
                        for (c, &q) in idx.iter().enumerate() {
                            v_dom[(r, c)] = v_global[(p, q)];
                        }
                    }
                    let v_dom_inv = match v_dom.inv() {
                        Ok(inv) => inv,
                        Err(_) => {
                            n_singular += 1;
                            continue;
                        }
                    };

                    let a_i_dom = gather_rows(&a2.slice(s![.., i * nvir..(i + 1) * nvir]), &idx);
                    let a_j_dom = gather_rows(&a2.slice(s![.., j * nvir..(j + 1) * nvir]), &idx);
                    let ct_i = v_dom_inv.dot(&a_i_dom); // (m, nvir)
                    let ct_j = v_dom_inv.dot(&a_j_dom);

                    // Exact block from the GLOBAL fit.
                    let g_ex = a2
                        .slice(s![.., i * nvir..(i + 1) * nvir])
                        .t()
                        .dot(&c_glob_loc.slice(s![.., j * nvir..(j + 1) * nvir]));

                    let naive_ij = a_i_dom.t().dot(&ct_j);
                    let cross_ij = ct_i.t().dot(&a_j_dom);
                    let sym_ij = ct_i.t().dot(&v_dom.dot(&ct_j));
                    let rob_ij = &naive_ij + &cross_ij - &sym_ij;

                    let en = frob_diff(&naive_ij, &g_ex);
                    let er = frob_diff(&rob_ij, &g_ex);
                    let gnorm = frob(&g_ex);
                    let dx = centers.row(i)[0] - centers.row(j)[0];
                    let dy = centers.row(i)[1] - centers.row(j)[1];
                    let dz = centers.row(i)[2] - centers.row(j)[2];
                    let r_ij = (dx * dx + dy * dy + dz * dz).sqrt();

                    // Off-diagonal pairs appear twice in the full ij sum.
                    let w = if i == j { 1.0 } else { 2.0 };
                    sum_en2 += w * en * en;
                    sum_er2 += w * er * er;
                    sum_gex2 += w * gnorm * gnorm;

                    let k = bin_index(r_ij);
                    let b = &mut bins[k];
                    b.n += 1;
                    b.sum_m += m as f64;
                    b.sum_gnorm += gnorm;
                    b.max_abs_naive = b.max_abs_naive.max(en);
                    b.max_abs_rob = b.max_abs_rob.max(er);
                    b.sum_abs_rob += er;
                    let rel = if gnorm > 1e-14 { er / gnorm } else { 0.0 };
                    b.max_rel_rob = b.max_rel_rob.max(rel);
                    worst.push((er, r_ij, rel, i, j));

                    if let (Some(gn), Some(gr)) = (g_naive_loc.as_mut(), g_rob_loc.as_mut()) {
                        gn.slice_mut(s![i * nvir..(i + 1) * nvir, j * nvir..(j + 1) * nvir]).assign(&naive_ij);
                        gr.slice_mut(s![i * nvir..(i + 1) * nvir, j * nvir..(j + 1) * nvir]).assign(&rob_ij);
                        if i != j {
                            // Robust block is symmetric under (ia)<->(jb) by
                            // construction; the one-sided naive (j,i) block is
                            // its OWN one-sided fit, not the transpose.
                            let naive_ji = a_j_dom.t().dot(&ct_i);
                            gn.slice_mut(s![j * nvir..(j + 1) * nvir, i * nvir..(i + 1) * nvir]).assign(&naive_ji);
                            gr.slice_mut(s![j * nvir..(j + 1) * nvir, i * nvir..(i + 1) * nvir]).assign(&rob_ij.t());
                        }
                    }
                }
            }

            let tot_naive = sum_en2.sqrt() / sum_gex2.sqrt();
            let tot_rob = sum_er2.sqrt() / sum_gex2.sqrt();
            println!("\n-- r_cut = {r_cut} Bohr   total ||dG||_F/||G||_F: naive {tot_naive:.3e}  robust {tot_rob:.3e}   singular domains skipped: {n_singular}");
            println!(
                "{:>6} {:>7} {:>9} {:>11} {:>13} {:>13} {:>13} {:>12}",
                "R bin", "npairs", "avg m", "avg|G_ex|", "max abs naive", "max abs rob", "mean abs rob", "max rel rob"
            );
            for (k, b) in bins.iter().enumerate() {
                if b.n == 0 {
                    continue;
                }
                println!(
                    "{:>6} {:>7} {:>9.1} {:>11.3e} {:>13.3e} {:>13.3e} {:>13.3e} {:>12.3e}",
                    bin_label(k),
                    b.n,
                    b.sum_m / b.n as f64,
                    b.sum_gnorm / b.n as f64,
                    b.max_abs_naive,
                    b.max_abs_rob,
                    b.sum_abs_rob / b.n as f64,
                    b.max_rel_rob
                );
            }
            worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for (er, r, rel, i, j) in worst.iter().take(3) {
                println!("   worst: pair ({i},{j}) R={r:.1} abs_rob={er:.3e} rel_rob={rel:.3e}");
            }

            if let (Some(gn), Some(gr)) = (g_naive_loc, g_rob_loc) {
                for (label, g_loc) in [("naive", gn), ("robust", gr)] {
                    let g1 = rotate_occ_axis(&g_loc, &u_rot, nocc, nvir, 0);
                    drop(g_loc);
                    let g_can = rotate_occ_axis(&g1, &u_rot, nocc, nvir, 1);
                    drop(g1);
                    let e_fit = mp2_energy_from_g(&g_can, &eps, nocc, nvir);
                    println!("   dE_{label} = {:.3e} Ha  (E_fit {:.9})", e_fit - e_exact, e_fit);
                }
            }
        }
        println!();
    }
}
