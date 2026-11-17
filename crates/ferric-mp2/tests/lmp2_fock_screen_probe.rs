//! Measurement probe for the tight-ε floor of the amplitude-threshold LMP2
//! mask (Fock-propagated screen lane, 2026-11-16). `#[ignore]`d: run
//! explicitly in a RELEASE build under `box-lease exec`:
//!
//! ```text
//! PROBE_XYZ=alkane_8.xyz PROBE_BASIS=6-31g PROBE_FC=8 \
//!   cargo test --release -p ferric-mp2 --test lmp2_fock_screen_probe -- --ignored --nocapture
//! ```
//!
//! Everything is built from ONE dense localized (ia|jb) (the global-fit test
//! path, `assemble_localized`), so the RI fit is identical between the exact
//! (ε = 0) and the masked solves and cancels from every error printed here.
//! Every solve is the production ragged PCG (`solve_under_rtol_policy`) on a
//! pattern built by the production block builder
//! (`pair_block_from_g_cand_gated`) from a caller mask.
//!
//! Pre-registered (written before the first run):
//! * H-Fock (physics of the mask): error at ε = 1e-6/1e-7 stays near the
//!   prototype's 1e-6 Ha; insensitive to CG rtol (1e-6 vs 1e-11); dropped
//!   exact amplitudes ≫ the diagonal estimate −J/D; Fock-ranked additions
//!   beat random additions at equal count by orders of magnitude.
//! * Artifact (Rust path): error moves with the CG rtol (solver), or the
//!   probe's Eq-8 mask disagrees with production `amplitude_lmp2` at the same
//!   ε (construction), or no floor at all (the prototype's mask differs).
//!   Ranked ≈ random would refute H-Fock.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{
    amplitude_lmp2, assemble_localized, build_vvhv, solve_under_rtol_policy, AmplitudeLmp2Config,
    CgRtolPolicy,
};
use ferric_mp2::ragged::{pair_block_from_g_cand_gated, PairBlock, Ragged};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{s, Array2};
use std::collections::HashMap;

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

/// Dense helper: amplitudes / masks as (no·nv, no·nv), row i·nv+a, col j·nv+b.
struct Dense {
    no: usize,
    nv: usize,
    j: Array2<f64>,
    foo: Array2<f64>,
    fvv: Array2<f64>,
    d: Array2<f64>,
}

impl Dense {
    /// The off-diagonal Fock part of A: (Fvv' t + t Fvv' − Foo' t − t Foo').
    /// `virt`/`occ` select which halves are applied.
    fn a_off(&self, t: &Array2<f64>, virt: bool, occ: bool) -> Array2<f64> {
        let (no, nv) = (self.no, self.nv);
        let mut fv = self.fvv.clone();
        for a in 0..nv {
            fv[(a, a)] = 0.0;
        }
        let mut fo = self.foo.clone();
        for i in 0..no {
            fo[(i, i)] = 0.0;
        }
        let mut out = Array2::<f64>::zeros(t.dim());
        if virt {
            for i in 0..no {
                for j in 0..no {
                    let tij = t.slice(s![i * nv..(i + 1) * nv, j * nv..(j + 1) * nv]);
                    let r = fv.dot(&tij) + tij.dot(&fv);
                    let mut o = out.slice_mut(s![i * nv..(i + 1) * nv, j * nv..(j + 1) * nv]);
                    o += &r;
                }
            }
        }
        if occ {
            // −Σ_k F_ik t_{ka,jb}
            let t3 = t.view().into_shape_with_order((no, nv * no * nv)).unwrap();
            let r = fo.dot(&t3);
            let r = r.into_shape_with_order((no * nv, no * nv)).unwrap();
            out -= &r;
            // −Σ_k t_{ia,kb} F_kj: for each row, (no, nv) block → Foᵀ·R
            for row in 0..no * nv {
                let rv = t
                    .row(row)
                    .to_owned()
                    .into_shape_with_order((no, nv))
                    .unwrap();
                let rr = fo.t().dot(&rv);
                let mut orow = out.row_mut(row);
                for (x, y) in orow.iter_mut().zip(rr.iter()) {
                    *x -= y;
                }
            }
        }
        out
    }
}

/// Close a mask under (ia)↔(jb) and the a↔b swap within a pair.
fn close(m: &mut [bool], no: usize, nv: usize) {
    let n = no * nv;
    for ia in 0..n {
        for jb in 0..n {
            if m[ia * n + jb] {
                let (i, a, j, b) = (ia / nv, ia % nv, jb / nv, jb % nv);
                let ib = i * nv + b;
                let ja = j * nv + a;
                m[jb * n + ia] = true;
                m[ib * n + ja] = true;
                m[ja * n + ib] = true;
            }
        }
    }
}

/// Production blocks from a dense mask (gate = indicator, threshold 0.5).
fn ragged_from_mask(dn: &Dense, m: &[bool]) -> Ragged {
    let (no, nv) = (dn.no, dn.nv);
    let n = no * nv;
    let fo: Vec<f64> = (0..no).map(|i| dn.foo[(i, i)]).collect();
    let fv: Vec<f64> = (0..nv).map(|a| dn.fvv[(a, a)]).collect();
    let cand: Vec<usize> = (0..nv).collect();
    let mut pairs: Vec<PairBlock> = Vec::new();
    for i in 0..no {
        for j in 0..no {
            let g =
                dn.j.slice(s![i * nv..(i + 1) * nv, j * nv..(j + 1) * nv])
                    .to_owned();
            let mut gate = Array2::<f64>::zeros((nv, nv));
            for a in 0..nv {
                for b in 0..nv {
                    if m[(i * nv + a) * n + j * nv + b] {
                        gate[(a, b)] = 1.0;
                    }
                }
            }
            if let Some(pb) =
                pair_block_from_g_cand_gated(i, j, &g, &gate, &cand, nv, &dn.fvv, &fo, &fv, 0.5)
            {
                pairs.push(pb);
            }
        }
    }
    let mut by_i: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut by_j: HashMap<usize, Vec<usize>> = HashMap::new();
    for (p, pb) in pairs.iter().enumerate() {
        by_i.entry(pb.i).or_default().push(p);
        by_j.entry(pb.j).or_default().push(p);
    }
    Ragged { pairs, by_i, by_j }
}

/// Solve on a mask: (E_hylleraas, dense amplitudes, kept fraction, iters).
fn solve(dn: &Dense, m: &[bool], rtol: f64) -> (f64, Array2<f64>, f64, usize) {
    let (no, nv) = (dn.no, dn.nv);
    let n = no * nv;
    let rg = ragged_from_mask(dn, m);
    let ps = solve_under_rtol_policy(
        &rg,
        &dn.foo,
        &dn.fvv,
        CgRtolPolicy::Explicit(rtol),
        2000,
        usize::MAX / 4,
        0.0,
    )
    .unwrap();
    let mut t = Array2::<f64>::zeros((n, n));
    let mut kept = 0usize;
    for (p, pb) in rg.pairs.iter().enumerate() {
        let nb = pb.db.len();
        for (r, &a) in pb.da.iter().enumerate() {
            for (c, &b) in pb.db.iter().enumerate() {
                if pb.pat[r * nb + c] {
                    kept += 1;
                    t[(pb.i * nv + a, pb.j * nv + b)] = ps.budgeted.solve.t[p][(r, c)];
                }
            }
        }
    }
    (
        ps.e_hylleraas,
        t,
        kept as f64 / (n * n) as f64,
        ps.budgeted.solve.iterations,
    )
}

/// t_{ib,ja} for every (ia, jb).
fn swap(dn: &Dense, t: &Array2<f64>) -> Array2<f64> {
    let (no, nv) = (dn.no, dn.nv);
    let mut out = Array2::<f64>::zeros(t.dim());
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    out[(i * nv + a, j * nv + b)] = t[(i * nv + b, j * nv + a)];
                }
            }
        }
    }
    out
}

/// Full-space Hylleraas functional Σ (2t − t_swap)(2J + A t): an upper
/// bound on the exact energy for ANY t (A positive definite).
fn e_h_full(dn: &Dense, t: &Array2<f64>) -> f64 {
    let at = &dn.d * t + dn.a_off(t, true, true);
    let tt = 2.0 * t - swap(dn, t);
    (&tt * &(2.0 * &dn.j + &at)).sum()
}

/// Kept-space solve with a caller RHS (dense, read on the pattern only).
fn solve_rhs(dn: &Dense, m: &[bool], rhs: &Array2<f64>, rtol: f64) -> Array2<f64> {
    let (no, nv) = (dn.no, dn.nv);
    let n = no * nv;
    let mut rg = ragged_from_mask(dn, m);
    for pb in rg.pairs.iter_mut() {
        let nb = pb.db.len();
        for (r, &a) in pb.da.iter().enumerate() {
            for (c, &b) in pb.db.iter().enumerate() {
                if pb.pat[r * nb + c] {
                    pb.j_blk[(r, c)] = rhs[(pb.i * nv + a, pb.j * nv + b)];
                }
            }
        }
    }
    let ps = solve_under_rtol_policy(
        &rg,
        &dn.foo,
        &dn.fvv,
        CgRtolPolicy::Explicit(rtol),
        2000,
        usize::MAX / 4,
        0.0,
    )
    .unwrap();
    let mut t = Array2::<f64>::zeros((n, n));
    for (p, pb) in rg.pairs.iter().enumerate() {
        let nb = pb.db.len();
        for (r, &a) in pb.da.iter().enumerate() {
            for (c, &b) in pb.db.iter().enumerate() {
                if pb.pat[r * nb + c] {
                    t[(pb.i * nv + a, pb.j * nv + b)] = ps.budgeted.solve.t[p][(r, c)];
                }
            }
        }
    }
    t
}

/// Linear (non-variational) energy Σ (2t − t_swap) J over the full space.
fn e_lin_full(dn: &Dense, t: &Array2<f64>) -> f64 {
    let tt = 2.0 * t - swap(dn, t);
    (&tt * &dn.j).sum()
}

/// WLLH25 Eq. 43-45: per pair, F_occ diagonal, Δ^ij τ^ij = −J^ij solved as
/// a Sylvester equation in the eigenbasis of F_vv (whole virtual space).
/// Returns (τ dense, kept-mask as 0/1 dense).
fn wllh_block_kapuy(dn: &Dense, m: &[bool]) -> (Array2<f64>, Array2<f64>) {
    use ndarray_linalg::{Eigh, UPLO};
    let (no, nv) = (dn.no, dn.nv);
    let n = no * nv;
    let (e, u) = dn.fvv.eigh(UPLO::Lower).unwrap();
    let mut tau = Array2::<f64>::zeros((n, n));
    for i in 0..no {
        for j in 0..no {
            let jb = dn.j.slice(s![i * nv..(i + 1) * nv, j * nv..(j + 1) * nv]);
            let mut x = u.t().dot(&jb).dot(&u);
            for a in 0..nv {
                for b in 0..nv {
                    x[(a, b)] /= -(e[a] + e[b] - dn.foo[(i, i)] - dn.foo[(j, j)]);
                }
            }
            let tb = u.dot(&x).dot(&u.t());
            tau.slice_mut(s![i * nv..(i + 1) * nv, j * nv..(j + 1) * nv])
                .assign(&tb);
        }
    }
    let mf = Array2::from_shape_vec((n, n), m.iter().map(|&k| f64::from(u8::from(k))).collect())
        .unwrap();
    (tau, mf)
}

fn eq8(dn: &Dense, eps: f64) -> Vec<bool> {
    let n = dn.no * dn.nv;
    let mut m: Vec<bool> = dn.j.iter().map(|x| x.abs() > eps).collect();
    close(&mut m, dn.no, dn.nv);
    assert_eq!(m.len(), n * n);
    m
}

/// xorshift64* — deterministic random control without a new dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

#[test]
#[ignore = "measurement probe: release build, box-lease exec, see module doc"]
fn probe_fock_floor() {
    let xyz = env_or("PROBE_XYZ", "alkane_8.xyz");
    let bname = env_or("PROBE_BASIS", "6-31g");
    let fc: usize = env_or("PROBE_FC", "8").parse().unwrap();
    let eps_list: Vec<f64> = env_or("PROBE_EPS", "1e-4,1e-5,1e-6,1e-7")
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled(&bname).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    let vvhv = build_vvhv(&mol, &obs, &obs_bs, &rhf).unwrap();
    let cfg0 = AmplitudeLmp2Config {
        eps: 0.0,
        frozen_core: fc,
        ..Default::default()
    };
    let lp = assemble_localized(&mol, &obs, &dfbs, op, &rhf, &cfg0, &vvhv).unwrap();
    let (no, nv) = (lp.no, lp.nv);
    let n = no * nv;
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    d[(i * nv + a, j * nv + b)] =
                        lp.f_vv[(a, a)] + lp.f_vv[(b, b)] - lp.f_oo[(i, i)] - lp.f_oo[(j, j)];
                }
            }
        }
    }
    let dn = Dense {
        no,
        nv,
        j: lp.j_dense.clone(),
        foo: lp.f_oo.clone(),
        fvv: lp.f_vv.clone(),
        d,
    };
    let nvv = vvhv.n_valence;
    println!(
        "PROBE {xyz}/{bname} fc={fc} no={no} nv={nv} n_vv={nvv} n_hv={}",
        vvhv.n_hard
    );
    let full = vec![true; n * n];
    let (e_ex, t_ex, _, it_ex) = solve(&dn, &full, 1e-11);
    println!("EXACT e_corr={e_ex:.12} cg={it_ex}");
    // Fock off-diagonal magnitudes by virtual block
    let (mut fvv_vv, mut fvv_hh, mut fvv_vh) = (0.0f64, 0.0f64, 0.0f64);
    for a in 0..nv {
        for b in 0..nv {
            if a == b {
                continue;
            }
            let x = lp.f_vv[(a, b)].abs();
            match (a < nvv, b < nvv) {
                (true, true) => fvv_vv = fvv_vv.max(x),
                (false, false) => fvv_hh = fvv_hh.max(x),
                _ => fvv_vh = fvv_vh.max(x),
            }
        }
    }
    let mut foo_off = 0.0f64;
    for i in 0..no {
        for j in 0..no {
            if i != j {
                foo_off = foo_off.max(lp.f_oo[(i, j)].abs());
            }
        }
    }
    println!(
        "FOCK max|Fvv off| vv-vv={fvv_vv:.3} hv-hv={fvv_hh:.3} vv-hv={fvv_vh:.3} max|Foo off|={foo_off:.3}"
    );

    for &eps in &eps_list {
        let m = eq8(&dn, eps);
        // production cross-check (same eps, global fit, no gate)
        let prod = amplitude_lmp2(
            &mol,
            &obs,
            &obs_bs,
            &dfbs,
            op,
            &rhf,
            &AmplitudeLmp2Config {
                eps,
                frozen_core: fc,
                cg_rtol: Some(1e-11),
                cg_max_iter: 2000,
                ..Default::default()
            },
        )
        .unwrap();
        let (e_m, t_m, keep, it) = solve(&dn, &m, 1e-11);
        let (e_m6, _, _, _) = solve(&dn, &m, 1e-6);
        println!(
            "FLOOR eps={eps:.0e} keep={keep:.4} err={:+.3e} err(rtol1e-6)={:+.3e} prod_err={:+.3e} prod_keep={:.4} probe-prod={:+.1e} cg={it}",
            e_m - e_ex,
            e_m6 - e_ex,
            prod.e_corr - e_ex,
            prod.keep_fraction,
            e_m - prod.e_corr
        );
        // dropped exact amplitudes vs the diagonal estimate
        let mut nd2 = 0.0;
        let mut ndiag2 = 0.0;
        let mut blk = [0.0f64; 4]; // vv-vv, vv-hv, hv-vv, hv-hv
        for ia in 0..n {
            for jb in 0..n {
                let k = ia * n + jb;
                if !m[k] {
                    let te = t_ex[(ia, jb)];
                    nd2 += te * te;
                    let td = dn.j[(ia, jb)] / dn.d[(ia, jb)];
                    ndiag2 += td * td;
                    let (a, b) = (ia % nv, jb % nv);
                    let bi = 2 * usize::from(a >= nvv) + usize::from(b >= nvv);
                    blk[bi] += te * te;
                }
            }
        }
        let tot: f64 = blk.iter().sum();
        println!(
            "DROPPED eps={eps:.0e} |t_ex|={:.3e} |J/D|={:.3e} ratio={:.0} share vv-vv={:.4} vv-hv={:.4} hv-vv={:.4} hv-hv={:.4}",
            nd2.sqrt(),
            ndiag2.sqrt(),
            nd2.sqrt() / ndiag2.sqrt().max(1e-300),
            blk[0] / tot,
            blk[1] / tot,
            blk[2] / tot,
            blk[3] / tot
        );
        // ---- embedding variants (WLLH25-style, our reading; abstract only) ----
        {
            let mf: Array2<f64> =
                Array2::from_shape_vec((n, n), m.iter().map(|&k| f64::from(u8::from(k))).collect())
                    .unwrap();
            let df = 1.0 - &mf;
            println!(
                "EMB eps={eps:.0e} check: e_h_full(t_M)-e_m={:+.1e} e_h_full(t_ex)-e_ex={:+.1e}",
                e_h_full(&dn, &t_m) - e_m,
                e_h_full(&dn, &t_ex) - e_ex
            );
            // V1: dropped first-order t_D0 = -J_D/D_D, coupled into kept RHS
            let t_d0 = &df * &(-&dn.j / &dn.d);
            let rhs = &dn.j + &dn.a_off(&t_d0, true, true);
            let t_k = solve_rhs(&dn, &m, &rhs, 1e-11);
            let t_v1 = &t_k + &t_d0;
            // V2: + one Jacobi (Kapuy-like) update of the dropped block
            let at = &dn.d * &t_v1 + dn.a_off(&t_v1, true, true);
            let t_d1 = &df * &(&t_d0 - &((&dn.j + &at) / &dn.d));
            let t_v2 = &t_k + &t_d1;
            // V3: V2's t_D1 fed back into the kept RHS once more, then re-update
            let rhs3 = &dn.j + &dn.a_off(&t_d1, true, true);
            let t_k3 = solve_rhs(&dn, &m, &rhs3, 1e-11);
            let t_v3a = &t_k3 + &t_d1;
            let at3 = &dn.d * &t_v3a + dn.a_off(&t_v3a, true, true);
            let t_d3 = &df * &(&t_d1 - &((&dn.j + &at3) / &dn.d));
            let t_v3 = &t_k3 + &t_d3;
            println!(
                "EMB eps={eps:.0e} masked={:+.3e} V1(rhs t_D0)={:+.3e} V1_keptonly={:+.3e} V2(+1 Jacobi t_D)={:+.3e} V3(2 macro)={:+.3e}",
                e_m - e_ex,
                e_h_full(&dn, &t_v1) - e_ex,
                e_h_full(&dn, &t_k) - e_ex,
                e_h_full(&dn, &t_v2) - e_ex,
                e_h_full(&dn, &t_v3) - e_ex
            );
            // ---- WLLH25 Eqs. 43-49 (preprint numbering): block Kapuy with
            // F_occ diagonal per pair, Sylvester solve over the pair's whole
            // virtual space (every integral exists here, so this is the
            // method's best case: no map limits), then the RHS embedding
            // m = Δ_ls τ_s on the kept equations solved with the FULL A.
            let (tau, mf_dense) = wllh_block_kapuy(&dn, &m);
            let tau_s = &df * &tau;
            let m_l = &mf_dense * &(dn.a_off(&tau_s, true, false));
            let t_l = solve_rhs(&dn, &m, &(&dn.j + &m_l), 1e-11);
            let t_w = &t_l + &tau_s;
            let t_bk = &t_m + &tau_s; // block Kapuy energy only, no RHS
            println!(
                "WLLH eps={eps:.0e} BKonly: lin={:+.3e} H={:+.3e} | BK+RHS: lin={:+.3e} H={:+.3e} | tau_s vs t_ex on D: rel={:.3e}",
                e_lin_full(&dn, &t_bk) - e_ex,
                e_h_full(&dn, &t_bk) - e_ex,
                e_lin_full(&dn, &t_w) - e_ex,
                e_h_full(&dn, &t_w) - e_ex,
                (&tau_s - &(&df * &t_ex)).mapv(|x| x * x).sum().sqrt()
                    / (&df * &t_ex).mapv(|x| x * x).sum().sqrt().max(1e-300)
            );
            // Gap (2) of WLLH25: the cross-pair F_occ coupling. Same probe on a
            // MODEL problem with F_occ made diagonal: if WLLH's residual error
            // vanishes there but not above, the F_occ gap is what is left.
            if std::env::var("PROBE_FOO_DIAG").is_ok() {
                let mut foo_d = dn.foo.clone();
                for i in 0..no {
                    for k in 0..no {
                        if i != k {
                            foo_d[(i, k)] = 0.0;
                        }
                    }
                }
                let dd = Dense {
                    no,
                    nv,
                    j: dn.j.clone(),
                    foo: foo_d,
                    fvv: dn.fvv.clone(),
                    d: dn.d.clone(),
                };
                let (e_ex_d, _, _, _) = solve(&dd, &full, 1e-11);
                let (e_m_d, t_m_d, _, _) = solve(&dd, &m, 1e-11);
                let (tau_d, _) = wllh_block_kapuy(&dd, &m);
                let tau_sd = &df * &tau_d;
                let m_ld = &mf * &(dd.a_off(&tau_sd, true, false));
                let t_ld = solve_rhs(&dd, &m, &(&dd.j + &m_ld), 1e-11);
                let t_wd = &t_ld + &tau_sd;
                let _ = t_m_d;
                println!(
                    "WLLH-FOODIAG eps={eps:.0e} masked={:+.3e} BK+RHS: lin={:+.3e} H={:+.3e}",
                    e_m_d - e_ex_d,
                    e_lin_full(&dd, &t_wd) - e_ex_d,
                    e_h_full(&dd, &t_wd) - e_ex_d
                );
            }
        }
        // ---- the shipped screen, dense (no candidate limit): keep where
        // |g~| = |J + F' t0 + t0 F'| > κ ε (t0 = −J/Δ, virtual Fock only) —
        // vs the production global path at the same (ε, κ), which evaluates it
        // only on each pair's Schwarz candidate square. Plus screen + WLLH.
        {
            let t0: Array2<f64> = -&dn.j / &dn.d;
            let gt = &dn.j + &dn.a_off(&t0, true, false);
            for kappa in [1.0, 0.3, 0.1] {
                let mut ms = m.clone();
                for (k, x) in gt.iter().enumerate() {
                    if x.abs() > kappa * eps {
                        ms[k] = true;
                    }
                }
                close(&mut ms, no, nv);
                let (e_s, _, keep_s, _) = solve(&dn, &ms, 1e-11);
                let prod = amplitude_lmp2(
                    &mol,
                    &obs,
                    &obs_bs,
                    &dfbs,
                    op,
                    &rhf,
                    &AmplitudeLmp2Config {
                        eps,
                        frozen_core: fc,
                        cg_rtol: Some(1e-11),
                        cg_max_iter: 2000,
                        fock_screen: Some(kappa),
                        ..Default::default()
                    },
                )
                .unwrap();
                let (tau, mfs) = wllh_block_kapuy(&dn, &ms);
                let dfs = 1.0 - &mfs;
                let tau_s = &dfs * &tau;
                let m_l = &mfs * &(dn.a_off(&tau_s, true, false));
                let t_l = solve_rhs(&dn, &ms, &(&dn.j + &m_l), 1e-11);
                let t_w = &t_l + &tau_s;
                println!(
                    "SCREEN eps={eps:.0e} kappa={kappa} dense: keep={keep_s:.4} err={:+.3e} | prod(cand): keep={:.4} err={:+.3e} | dense+WLLH: lin={:+.3e} H={:+.3e}",
                    e_s - e_ex,
                    prod.keep_fraction,
                    prod.e_corr - e_ex,
                    e_lin_full(&dn, &t_w) - e_ex,
                    e_h_full(&dn, &t_w) - e_ex
                );
            }
        }
        if std::env::var("PROBE_NO_RANK").is_ok()
            || (eps > 1e-6 && std::env::var("PROBE_RANK_ALL").is_err())
        {
            continue;
        }
        // estimates: (name, t_est on dropped)
        let t0: Array2<f64> = -&dn.j / &dn.d;
        let ests: Vec<(&str, Array2<f64>)> = vec![
            (
                "proto(tM,full)",
                -(&dn.j + &dn.a_off(&t_m, true, true)) / &dn.d,
            ),
            ("tM,virt", -(&dn.j + &dn.a_off(&t_m, true, false)) / &dn.d),
            ("tM,occ", -(&dn.j + &dn.a_off(&t_m, false, true)) / &dn.d),
            ("t0,virt", -(&dn.j + &dn.a_off(&t0, true, false)) / &dn.d),
            ("t0,full", -(&dn.j + &dn.a_off(&t0, true, true)) / &dn.d),
        ];
        let dropped: Vec<usize> = (0..n * n).filter(|&k| !m[k]).collect();
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for frac in [0.01, 0.05, 0.2] {
            let n_add = (frac * dropped.len() as f64) as usize;
            // random control: partial Fisher-Yates
            let mut pool = dropped.clone();
            for x in 0..n_add {
                let y = x + (rng.next() as usize) % (pool.len() - x);
                pool.swap(x, y);
            }
            let mut mr = m.clone();
            for &k in &pool[..n_add] {
                mr[k] = true;
            }
            close(&mut mr, no, nv);
            let (e_r, _, keep_r, _) = solve(&dn, &mr, 1e-11);
            print!(
                "RANK eps={eps:.0e} frac={frac:.2} random keep={keep_r:.4} err={:+.3e}",
                e_r - e_ex
            );
            for (name, te) in &ests {
                let mut ord = dropped.clone();
                let tf = te.as_slice().unwrap();
                ord.sort_by(|&x, &y| tf[y].abs().partial_cmp(&tf[x].abs()).unwrap());
                let mut mt = m.clone();
                for &k in &ord[..n_add] {
                    mt[k] = true;
                }
                close(&mut mt, no, nv);
                let (e_t, _, keep_t, _) = solve(&dn, &mt, 1e-11);
                print!(" | {name} keep={keep_t:.4} err={:+.3e}", e_t - e_ex);
            }
            println!();
        }
    }
}

fn vm_hwm_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|x| x.parse().ok()))
        })
        .unwrap_or(0)
}

/// One production run per process (so VmHWM is that run's peak):
/// PROBE_XYZ, PROBE_BASIS, PROBE_FC, PROBE_EPS (single), PROBE_KAPPA
/// ("none" or κ), PROBE_PATH ("direct" = CLI integral-direct production maps,
/// "global" = the global-fit path, "ref" = canonical ri_mp2). Prints one
/// `SWEEP {json}` line; RHF time is not in t_method.
#[test]
#[ignore = "measurement probe: release build, box-lease exec, one config per process"]
fn probe_sweep_one() {
    use ferric_mp2::lmp2_direct::{amplitude_lmp2_direct, DirectConfig};
    use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
    let xyz = env_or("PROBE_XYZ", "alkane_8.xyz");
    let bname = env_or("PROBE_BASIS", "6-31g");
    let fc: usize = env_or("PROBE_FC", "8").parse().unwrap();
    let eps: f64 = env_or("PROBE_EPS", "1e-6").parse().unwrap();
    let kappa: Option<f64> = match env_or("PROBE_KAPPA", "none").as_str() {
        "none" => None,
        k => Some(k.parse().unwrap()),
    };
    let path = env_or("PROBE_PATH", "direct");
    let xyz_path = if xyz.starts_with('/') {
        xyz.clone()
    } else {
        format!(
            "{}/../../testdata/molecules/{xyz}",
            env!("CARGO_MANIFEST_DIR")
        )
    };
    let mol = Molecule::load_xyz(&xyz_path).unwrap();
    let obs_bs = basis::bundled(&bname).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let aux = if bname.starts_with("cc-pv") {
        format!("{bname}-ri")
    } else {
        "cc-pvdz-ri".to_string()
    };
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(&aux).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let t_rhf = std::time::Instant::now();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    let t_rhf = t_rhf.elapsed().as_secs_f64();
    let hwm_rhf = vm_hwm_kb();
    let cfg = AmplitudeLmp2Config {
        eps,
        frozen_core: fc,
        fock_screen: kappa,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let (e_corr, keep, t_asm, t_solve, cg) = match path.as_str() {
        "ref" => {
            let r = ri_mp2(
                &mol,
                &obs,
                &dfbs,
                op,
                &rhf,
                &RiMp2Config {
                    frozen_core: fc,
                    ..Default::default()
                },
            )
            .unwrap();
            (r.mp2_corr, 1.0, 0.0, 0.0, 0)
        }
        "global" => {
            let r = amplitude_lmp2(&mol, &obs, &obs_bs, &dfbs, op, &rhf, &cfg).unwrap();
            (
                r.e_corr,
                r.keep_fraction,
                r.timings.t_assembly_s,
                r.timings.t_solve_s,
                r.cg_iterations,
            )
        }
        "direct" => {
            let dcfg = DirectConfig {
                aux_radius_bohr: 10.0,
                virt_radius_bohr: Some(12.0),
                ao_tail: 1e-3,
                schwarz_skip: 1e-5,
                batch_merge: 4,
                ..Default::default()
            };
            let (r, _) =
                amplitude_lmp2_direct(&mol, &obs, &obs_bs, &dfbs, op, &rhf, &cfg, &dcfg).unwrap();
            (
                r.e_corr,
                r.keep_fraction,
                r.timings.t_assembly_s,
                r.timings.t_solve_s,
                r.cg_iterations,
            )
        }
        p => panic!("PROBE_PATH {p}"),
    };
    let t_method = t0.elapsed().as_secs_f64();
    println!(
        "SWEEP {{\"xyz\":\"{xyz}\",\"basis\":\"{bname}\",\"aux\":\"{aux}\",\"fc\":{fc},\"path\":\"{path}\",\"eps\":{eps:e},\"kappa\":{},\"e_corr\":{e_corr:.12},\"keep\":{keep:.6e},\"t_method\":{t_method:.3},\"t_asm\":{t_asm:.3},\"t_solve\":{t_solve:.3},\"cg\":{cg},\"t_rhf\":{t_rhf:.2},\"hwm_rhf_kb\":{hwm_rhf},\"hwm_kb\":{}}}",
        kappa.map_or("null".to_string(), |k| format!("{k:e}")),
        vm_hwm_kb()
    );
}
