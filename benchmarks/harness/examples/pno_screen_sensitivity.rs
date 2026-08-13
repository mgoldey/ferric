//! PNO-screening sensitivity of union-domain-fitted pair blocks — follow-up
//! to wiki/local-ri-robust-domain-fitting.md §9's open question: does the
//! few-percent RELATIVE error on distant (R^-3-small) pair blocks change what
//! magnitude-based consumers keep?
//!
//! Two consumers modeled, per localized pair (i<=j), with IDENTICAL
//! semicanonical denominators on both sides so the block error is the only
//! variable:
//!   - Pair prescreening (DLPNO T_CutPairs): semicanonical pair energy
//!     e_ij = w * sum_ab K_ab (2K_ab - K_ba) / (F_ii + F_jj - e_a - e_b),
//!     F_ii = localized-occ diagonal Fock (U^T diag(eps) U), w = 1/2 for
//!     diag/offdiag. Keep iff |e_ij| > T for T in {1e-4, 1e-5, 1e-6} Ha.
//!   - PNO truncation (T_CutPNO in {1e-7, 1e-8}): MP1 pair density
//!     D = (Tt^T T + Tt T^T)/(1+delta_ij), Tt = 4T - 2T^T, T = K/D_ab;
//!     retained-PNO count = #{eigenvalue > T_CutPNO}.
//!
//! K_exact = global-fit pair block; K_fit = union-domain Galerkin block
//! (P_ij = P_i U P_j, sub-block V^{-1} — the robust form collapses to this
//! same-metric projection, see pair_union_ri_bench.rs).
//!
//! Artifact hypotheses (pre-registered):
//!   - harmless: keep/drop flips confined to pairs whose |e_exact| lies
//!     within +-10% of the threshold (boundary noise inherent to ANY
//!     estimator); PNO-count shifts at most +-1 on distant pairs; screened-out
//!     energy discrepancy << 1% of E_corr.
//!   - harmful: flips on pairs FAR from thresholds, or PNO-count shifts on
//!     kept (near) pairs.
//!   - broken implementation: flips/count shifts on NEAR pairs, where the
//!     relative block error is ~1e-4 and can never move a decision.
//! Anchor: C4 at r_cut = 50 (full retention) must give zero flips, zero PNO
//! count differences, and pair-energy errors at machine eps.
//!
//! Run:
//!   OPENBLAS_NUM_THREADS=1 cargo run --release -p ferric-benchmarks \
//!     --example pno_screen_sensitivity

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
use ndarray_linalg::{Eigh, Inverse, UPLO};

const OBS_NAME: &str = "cc-pvdz";
const AUX_NAME: &str = "cc-pvdz-ri";
const T_CUT_PAIRS: &[f64] = &[1e-4, 1e-5, 1e-6];
const T_CUT_PNO: &[f64] = &[1e-7, 1e-8];
const BIN_EDGES: &[f64] = &[1e-6, 4.0, 8.0, 12.0, 18.0, 30.0];

fn frob_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

fn gather_rows(full: &ndarray::ArrayView2<f64>, idx: &[usize]) -> Array2<f64> {
    let nvir = full.ncols();
    let mut out = Array2::<f64>::zeros((idx.len(), nvir));
    for (r, &p) in idx.iter().enumerate() {
        out.row_mut(r).assign(&full.row(p));
    }
    out
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

/// Semicanonical pair energy and PNO occupation spectrum from a pair block.
fn pair_quantities(
    k_blk: &Array2<f64>,
    f_ii: f64,
    f_jj: f64,
    eps_vir: &[f64],
    diag_pair: bool,
) -> (f64, Vec<f64>) {
    let nvir = eps_vir.len();
    let mut t = Array2::<f64>::zeros((nvir, nvir));
    let mut e = 0.0;
    for a in 0..nvir {
        for b in 0..nvir {
            let d = f_ii + f_jj - eps_vir[a] - eps_vir[b];
            let kab = k_blk[(a, b)];
            let kba = k_blk[(b, a)];
            e += kab * (2.0 * kab - kba) / d;
            t[(a, b)] = kab / d;
        }
    }
    let w = if diag_pair { 1.0 } else { 2.0 };
    let tt = 4.0 * &t - 2.0 * &t.t();
    let dens = (&tt.t().dot(&t) + &tt.dot(&t.t())) / if diag_pair { 2.0 } else { 1.0 };
    let (occs, _) = dens.eigh(UPLO::Lower).unwrap();
    (w * e, occs.to_vec())
}

struct BinAgg {
    n: usize,
    max_rel_e: f64,
    max_abs_e: f64,
    sum_abs_e_ex: f64,
    max_dn_pno: [usize; 2],
}

fn main() {
    println!("# ==========================================================================");
    println!("# PNO / prescreening sensitivity of union-domain-fitted pair blocks");
    println!("# Basis: {OBS_NAME} / Aux: {AUX_NAME}  T_CutPairs={T_CUT_PAIRS:?}  T_CutPNO={T_CUT_PNO:?}");
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
            Ok(r) => r,
            Err(e) => {
                println!("alkane_{n_c}: SCF failed: {e}\n");
                continue;
            }
        };
        let c_occ_can = rhf.mos_r().slice(s![.., ..nocc]).to_owned();
        let c_vir_can = rhf.mos_r().slice(s![.., nocc..]).to_owned();
        let eps: Vec<f64> = rhf.eps_r().to_vec();
        let eps_vir: Vec<f64> = eps[nocc..].to_vec();

        let dip = dipole(&obs, [0.0, 0.0, 0.0]).unwrap();
        let boys = boys_localize(&c_occ_can, &dip, 200);
        if !boys.converged {
            println!("alkane_{n_c}: Boys not converged\n");
            continue;
        }
        let c_occ_loc = boys.c_loc;
        let centers = boys.centers;
        let s_ao = overlap(&obs);
        let u_rot = c_occ_can.t().dot(&s_ao).dot(&c_occ_loc);
        let ortho = frob_diff(&u_rot.t().dot(&u_rot), &Array2::eye(nocc));
        assert!(ortho < 1e-8, "U not orthogonal: {ortho:.3e}");
        // Localized-occ diagonal Fock (semicanonical): F_ii = sum_k U[k,i]^2 eps_k.
        let f_loc: Vec<f64> = (0..nocc)
            .map(|i| (0..nocc).map(|k| u_rot[(k, i)] * u_rot[(k, i)] * eps[k]).sum())
            .collect();

        let v_global = threeindex::coulomb_metric_2c(op, &dfbs).unwrap();
        let v_global_inv = v_global.inv().unwrap();
        // eri3-limit fix: blocked AO->MO transform (never materializes the
        // naux*nbas^2 AO tensor — ~13 GB at C32/cc-pVDZ); bit-identical to
        // the dense path, which small systems still take under the budget.
        let eri3_loc =
            ferric_mp2::rimp2::eri3_mo_ov_blocked(op, &obs, &dfbs, &c_occ_loc, &c_vir_can, 2 << 30).unwrap();
        let a2 = eri3_loc.to_shape((naux, nov)).unwrap().to_owned();
        drop(eri3_loc);
        let c_glob = v_global_inv.dot(&a2);

        let aux_shell_centers = dfbs.shell_centers();
        let aux_shell_offsets = dfbs.shell_offsets();
        let aux_shell_dims = dfbs.shell_dims();

        println!("### alkane_{n_c}  nocc={nocc}  nvir={nvir}  naux={naux}");
        let radii: &[f64] = if n_c == 4 { &[6.0, 10.0, 50.0] } else { &[6.0, 10.0] };

        for &r_cut in radii {
            let mut orb_domains: Vec<Vec<usize>> = Vec::with_capacity(nocc);
            for i in 0..nocc {
                let ci = centers.row(i);
                let mut fns = Vec::new();
                for sh in 0..dfbs.nshells() {
                    let c = aux_shell_centers[sh];
                    let d2 = (ci[0] - c[0]).powi(2) + (ci[1] - c[1]).powi(2) + (ci[2] - c[2]).powi(2);
                    if d2.sqrt() <= r_cut {
                        let f0 = aux_shell_offsets[sh];
                        fns.extend(f0..f0 + aux_shell_dims[sh]);
                    }
                }
                orb_domains.push(fns);
            }

            // Per-pair: (e_ex, e_fit, R, n_pno_ex[2], n_pno_fit[2])
            let mut rows: Vec<(f64, f64, f64, [usize; 2], [usize; 2])> = Vec::new();
            let mut e_sum_ex = 0.0f64;

            for i in 0..nocc {
                for j in i..nocc {
                    let mut idx: Vec<usize> = orb_domains[i].iter().chain(orb_domains[j].iter()).copied().collect();
                    idx.sort_unstable();
                    idx.dedup();
                    let m = idx.len();
                    assert!(m > 0, "empty union domain for pair ({i},{j})");

                    let mut v_dom = Array2::<f64>::zeros((m, m));
                    for (r, &p) in idx.iter().enumerate() {
                        for (c, &q) in idx.iter().enumerate() {
                            v_dom[(r, c)] = v_global[(p, q)];
                        }
                    }
                    let v_dom_inv = v_dom.inv().expect("singular union domain");

                    let a_i_dom = gather_rows(&a2.slice(s![.., i * nvir..(i + 1) * nvir]), &idx);
                    let a_j_dom = gather_rows(&a2.slice(s![.., j * nvir..(j + 1) * nvir]), &idx);
                    let k_fit = a_i_dom.t().dot(&v_dom_inv.dot(&a_j_dom));
                    let k_ex = a2
                        .slice(s![.., i * nvir..(i + 1) * nvir])
                        .t()
                        .dot(&c_glob.slice(s![.., j * nvir..(j + 1) * nvir]));

                    let diag_pair = i == j;
                    let (e_ex, occ_ex) = pair_quantities(&k_ex, f_loc[i], f_loc[j], &eps_vir, diag_pair);
                    let (e_fit, occ_fit) = pair_quantities(&k_fit, f_loc[i], f_loc[j], &eps_vir, diag_pair);
                    e_sum_ex += e_ex;

                    let count = |occs: &[f64], t: f64| occs.iter().filter(|&&o| o > t).count();
                    let n_ex = [count(&occ_ex, T_CUT_PNO[0]), count(&occ_ex, T_CUT_PNO[1])];
                    let n_fit = [count(&occ_fit, T_CUT_PNO[0]), count(&occ_fit, T_CUT_PNO[1])];

                    let dx = centers.row(i)[0] - centers.row(j)[0];
                    let dy = centers.row(i)[1] - centers.row(j)[1];
                    let dz = centers.row(i)[2] - centers.row(j)[2];
                    let r_ij = (dx * dx + dy * dy + dz * dz).sqrt();
                    rows.push((e_ex, e_fit, r_ij, n_ex, n_fit));
                }
            }

            println!("\n-- r_cut = {r_cut} Bohr   n_pairs = {}   sum e_ij(semicanonical, exact) = {e_sum_ex:.6} Ha", rows.len());

            // Prescreening decisions.
            for &t in T_CUT_PAIRS {
                let kept_ex = rows.iter().filter(|r| r.0.abs() > t).count();
                let kept_fit = rows.iter().filter(|r| r.1.abs() > t).count();
                let wrongly_dropped: Vec<&(f64, f64, f64, [usize; 2], [usize; 2])> =
                    rows.iter().filter(|r| r.0.abs() > t && r.1.abs() <= t).collect();
                let wrongly_kept = rows.iter().filter(|r| r.0.abs() <= t && r.1.abs() > t).count();
                let boundary = rows.iter().filter(|r| r.0.abs() > 0.9 * t && r.0.abs() < 1.1 * t).count();
                let max_wd = wrongly_dropped.iter().map(|r| r.0.abs()).fold(0.0f64, f64::max);
                let dropped_e_ex: f64 = rows.iter().filter(|r| r.0.abs() <= t).map(|r| r.0).sum();
                let dropped_e_fit_decision: f64 = rows.iter().filter(|r| r.1.abs() <= t).map(|r| r.0).sum();
                println!(
                    "   T={t:.0e}: kept ex/fit {kept_ex}/{kept_fit}  flips drop/keep {}/{}  (pairs within +-10% of T: {boundary})  max|e| wrongly dropped {max_wd:.2e}  screened-out e: ex {dropped_e_ex:.3e} vs fit-decision {dropped_e_fit_decision:.3e}",
                    wrongly_dropped.len(),
                    wrongly_kept
                );
            }

            // Pair-energy error and PNO count shifts by separation bin.
            let mut bins: Vec<BinAgg> = (0..=BIN_EDGES.len())
                .map(|_| BinAgg { n: 0, max_rel_e: 0.0, max_abs_e: 0.0, sum_abs_e_ex: 0.0, max_dn_pno: [0, 0] })
                .collect();
            for (e_ex, e_fit, r, n_ex, n_fit) in &rows {
                let b = &mut bins[bin_index(*r)];
                b.n += 1;
                let ae = (e_fit - e_ex).abs();
                b.max_abs_e = b.max_abs_e.max(ae);
                if e_ex.abs() > 1e-12 {
                    b.max_rel_e = b.max_rel_e.max(ae / e_ex.abs());
                }
                b.sum_abs_e_ex += e_ex.abs();
                for k in 0..2 {
                    b.max_dn_pno[k] = b.max_dn_pno[k].max(n_ex[k].abs_diff(n_fit[k]));
                }
            }
            println!(
                "   {:>6} {:>7} {:>13} {:>12} {:>12} {:>10} {:>10}",
                "R bin", "npairs", "sum|e_ex|", "max abs dE", "max rel dE", "maxdN@1e-7", "maxdN@1e-8"
            );
            for (k, b) in bins.iter().enumerate() {
                if b.n == 0 {
                    continue;
                }
                println!(
                    "   {:>6} {:>7} {:>13.3e} {:>12.3e} {:>12.3e} {:>10} {:>10}",
                    bin_label(k),
                    b.n,
                    b.sum_abs_e_ex,
                    b.max_abs_e,
                    b.max_rel_e,
                    b.max_dn_pno[0],
                    b.max_dn_pno[1]
                );
            }
        }
        println!();
    }
}
