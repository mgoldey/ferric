//! THROWAWAY PROBE (research/terfc-screening-bound). Ignored; env-driven.
//!
//! MODE=unit      : erfc / gd_ssss / majorant unit checks
//! MODE=validity  : bound >= true integral on ALL shell triples (and quartets if QUARTETS=1)
//! MODE=count     : survivor counts (triples; quartets if QUARTETS=1; metric pairs)
//! Env: MOL (xyz file in testdata/molecules), OBS, AUX, OPS ("coul;terfc:1.0:0.7071;erfc:0.42"),
//!      DELTAS ("1e-2,1e-4,1e-6"), TAUS ("1e-6,1e-8"), B2_N (B2 sample size), ORACLE (0/1).
//!
//!   RAYON_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 MODE=validity MOL=alkane_4.xyz \
//!   cargo test -p ferric-mp2 --release --test terfc_screen_probe -- --ignored --nocapture

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::units::ANGSTROM_TO_BOHR as ANG;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex;
use ferric_mp2::probe_terfc_bound::*;
use rayon::prelude::*;

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}
fn flist(s: &str) -> Vec<f64> {
    s.split(',').map(|x| x.parse().unwrap()).collect()
}

fn parse_op(s: &str) -> (Operator, Kern) {
    let p: Vec<&str> = s.split(':').collect();
    match p[0] {
        "coul" => (
            Operator::coulomb(),
            Kern {
                kind: KKind::Coul,
                r0: 0.0,
                w: 0.0,
            },
        ),
        "terfc" => {
            let r0 = p[1].parse::<f64>().unwrap() * ANG;
            let s: f64 = p[2].parse().unwrap();
            (
                Operator::terfc_with_omega(r0, s / r0),
                Kern {
                    kind: KKind::Terfc,
                    r0,
                    w: s / r0,
                },
            )
        }
        "erfc" => {
            let w = p[1].parse::<f64>().unwrap() / ANG;
            (
                Operator::erfc(w),
                Kern {
                    kind: KKind::Erfc,
                    r0: 0.0,
                    w,
                },
            )
        }
        o => panic!("op {o}"),
    }
}

struct Ctx {
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    omaj: Vec<ShellMaj>,
    dmaj: Vec<ShellMaj>,
    pairs: Vec<(usize, usize)>,
    pm: Vec<PairMaj>,
    pp: Vec<PairMaj>,
}

fn setup() -> Ctx {
    let molf = env("MOL", "alkane_4.xyz");
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{molf}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(&env("OBS", "6-31g")).unwrap()).unwrap();
    let dfbs =
        PreparedBasis::new(&mol, &basis::bundled(&env("AUX", "cc-pvdz-ri")).unwrap()).unwrap();
    let f: f64 = env("FMAJ", "0.9").parse().unwrap();
    let omaj: Vec<ShellMaj> = obs
        .located_shells()
        .iter()
        .map(|s| shell_maj(s, f))
        .collect();
    let dmaj: Vec<ShellMaj> = dfbs
        .located_shells()
        .iter()
        .map(|s| shell_maj(s, f))
        .collect();
    let n = obs.nshells();
    let mut pairs = vec![];
    for a in 0..n {
        for b in 0..=a {
            pairs.push((a, b));
        }
    }
    let pm: Vec<PairMaj> = pairs
        .par_iter()
        .map(|&(a, b)| pair_maj(&omaj[a], &omaj[b]))
        .collect();
    let pp: Vec<PairMaj> = dmaj.iter().map(single_maj).collect();
    eprintln!(
        "SETUP {molf} nsh_obs={n} nbf={} nsh_aux={} naux={} pairs={} f={f}",
        obs.nbasis(),
        dfbs.nshells(),
        dfbs.nbasis(),
        pairs.len()
    );
    Ctx {
        obs,
        dfbs,
        omaj,
        dmaj,
        pairs,
        pm,
        pp,
    }
}

fn qpair(c: &Ctx, op: Operator) -> Vec<f64> {
    let dims = c.obs.shell_dims();
    c.pairs
        .par_iter()
        .map_init(
            || Engine::new_2e(op, &c.obs, 1e-14).unwrap(),
            |eng, &(a, b)| {
                let (na, nb) = (dims[a], dims[b]);
                let blk = eng.compute_quartet(&c.obs, a, b, a, b);
                let mut m = 0.0f64;
                if let Some(v) = blk {
                    for fa in 0..na {
                        for fb in 0..nb {
                            m = m.max(v[((fa * nb + fb) * na + fa) * nb + fb].abs());
                        }
                    }
                }
                m.sqrt()
            },
        )
        .collect()
}

fn qaux(c: &Ctx, op: Operator) -> Vec<f64> {
    let dims = c.dfbs.shell_dims();
    let mut eng = Engine::new_2center(op, &c.dfbs, 1e-14).unwrap();
    (0..c.dfbs.nshells())
        .map(|p| {
            let n = dims[p];
            let v = eng.compute_eri2(&c.dfbs, p, p);
            (0..n)
                .map(|i| v[i * n + i].abs())
                .fold(0.0, f64::max)
                .sqrt()
        })
        .collect()
}

/// true max|(P|ab)| per (P, pair): index p*npairs + k
fn true_triples(c: &Ctx, op: Operator) -> Vec<f64> {
    let np = c.pairs.len();
    let dims = c.obs.shell_dims();
    let ddims = c.dfbs.shell_dims();
    let rows: Vec<Vec<f64>> = (0..c.dfbs.nshells())
        .into_par_iter()
        .map_init(
            || Engine::new_3center(op, &c.obs, &c.dfbs, 1e-14).unwrap(),
            |eng, p| {
                let _ = (dims, ddims);
                c.pairs
                    .iter()
                    .map(|&(a, b)| {
                        eng.compute_eri3(&c.obs, &c.dfbs, p, a, b)
                            .map(|v| v.iter().fold(0.0f64, |m, x| m.max(x.abs())))
                            .unwrap_or(0.0)
                    })
                    .collect()
            },
        )
        .collect();
    let mut out = Vec::with_capacity(rows.len() * np);
    for r in rows {
        out.extend(r);
    }
    out
}

fn true_quartets(c: &Ctx, op: Operator) -> Vec<f64> {
    // pair-index lower triangle: (i,j), j<=i  -> idx i*(i+1)/2 + j
    let np = c.pairs.len();
    let rows: Vec<Vec<f64>> = (0..np)
        .into_par_iter()
        .map_init(
            || Engine::new_2e(op, &c.obs, 1e-14).unwrap(),
            |eng, i| {
                let (a, b) = c.pairs[i];
                (0..=i)
                    .map(|j| {
                        let (cc, d) = c.pairs[j];
                        eng.compute_quartet(&c.obs, a, b, cc, d)
                            .map(|v| v.iter().fold(0.0f64, |m, x| m.max(x.abs())))
                            .unwrap_or(0.0)
                    })
                    .collect()
            },
        )
        .collect();
    rows.into_iter().flatten().collect()
}

#[derive(Default)]
struct Stat {
    n: usize,
    viol: usize,
    vac: usize,
    zero: usize,
    ratios: Vec<f64>,
}
impl Stat {
    fn add(&mut self, bound: f64, tru: f64) {
        self.n += 1;
        if !bound.is_finite() {
            self.vac += 1;
            return;
        }
        if bound == 0.0 {
            // libint2 returned None (screened to zero) for the bra/ket diagonal => Q = 0
            self.zero += 1;
            return;
        }
        if bound < tru * (1.0 - 1e-9) {
            self.viol += 1;
        }
        self.ratios.push(bound / tru);
    }
    fn line(&mut self, tag: &str) {
        self.ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f64| {
            if self.ratios.is_empty() {
                f64::NAN
            } else {
                self.ratios[((self.ratios.len() - 1) as f64 * f) as usize]
            }
        };
        println!(
            "VALID {tag} n={} violations={} zeroQ={} vacuous(inf)={} minratio={:.4e} q10={:.3e} q50={:.3e} q90={:.3e}",
            self.n, self.viol, self.zero, self.vac, q(0.0), q(0.1), q(0.5), q(0.9)
        );
    }
}

#[test]
#[ignore]
fn terfc_screen_probe() {
    match env("MODE", "validity").as_str() {
        "unit" => unit(),
        "validity" => validity(),
        "count" => count(),
        "lmp2" => lmp2(),
        m => panic!("MODE {m}"),
    }
}

fn unit() {
    for (x, r) in [
        (0.5, 0.4795001221869535),
        (1.0, 0.15729920705028513),
        (2.0, 4.677734981047266e-3),
        (3.0, 2.209049699858544e-05),
        (5.0, 1.5374597944280349e-12),
        (-1.0, 1.8427007929497148),
    ] {
        println!(
            "erfc({x}) = {:.16e} ref {:.16e} rel {:.2e}",
            erfc(x),
            r,
            (erfc(x) / r - 1.0).abs()
        );
    }
    // gd_ssss with kappa=0 and p,q: Coulomb of two unnormalised Gaussians:
    // int int e^{-p r1^2} e^{-q r2^2}/r12 = pi^3 /(pq)^{3/2} * (2/sqrt(pi))*sqrt(a)*F0(a R^2)*... use ratio test vs
    // closed form: (ss|ss) = 2 pi^{5/2}/(p q sqrt(p+q)) F0(pq/(p+q) R^2)
    let (p, q, r2) = (0.7f64, 1.3f64, 2.0f64);
    let a = p * q / (p + q);
    let x = a * r2;
    let f0 = 0.5 * (std::f64::consts::PI.sqrt() / x.sqrt()) * erf_via(x.sqrt());
    let exact = 2.0 * std::f64::consts::PI.powf(2.5) / (p * q * (p + q).sqrt()) * f0;
    let got = gd_ssss(p, q, r2, 0.0);
    println!(
        "gd_ssss coulomb check got {got:.12e} exact {exact:.12e} rel {:.2e}",
        (got / exact - 1.0).abs()
    );
}
fn erf_via(x: f64) -> f64 {
    1.0 - erfc(x)
}

fn validity() {
    let c = setup();
    let ops = env("OPS", "terfc:1.0:0.70710678");
    let deltas = flist(&env("DELTAS", "1e-2,1e-4,1e-6,1e-8"));
    let b2n: usize = env("B2_N", "20000").parse().unwrap();
    let quart = env("QUARTETS", "0") == "1";
    let np = c.pairs.len();
    for ops_s in ops.split(';') {
        let (op, kern) = parse_op(ops_s);
        let qp = qpair(&c, op);
        let qa = qaux(&c, op);
        let t3 = true_triples(&c, op);
        let n_sig = t3.iter().filter(|&&x| x > 1e-12).count();
        println!("OP {ops_s} triples={} sig(>1e-12)={n_sig}", t3.len());
        // Schwarz
        let mut st = Stat::default();
        for p in 0..c.dfbs.nshells() {
            for k in 0..np {
                let t = t3[p * np + k];
                if t > 1e-12 {
                    st.add(qa[p] * qp[k], t);
                }
            }
        }
        st.line(&format!("{ops_s} triple schwarz"));
        for &delta in &deltas {
            let ext_p: Vec<f64> = c.pp.iter().map(|x| x.extent(delta)).collect();
            let ext_k: Vec<f64> = c.pm.iter().map(|x| x.extent(delta)).collect();
            let mut s_b1 = Stat::default();
            let mut s_cmb = Stat::default();
            let mut s_mut = Stat::default();
            for p in 0..c.dfbs.nshells() {
                for k in 0..np {
                    let t = t3[p * np + k];
                    if t > 1e-12 {
                        let b = b1(&c.pp[p], ext_p[p], &c.pm[k], ext_k[k], &kern);
                        s_b1.add(b, t);
                        s_cmb.add(b.min(qa[p] * qp[k]), t);
                        s_mut.add(
                            b1(&c.pp[p], 0.5 * ext_p[p], &c.pm[k], 0.5 * ext_k[k], &kern),
                            t,
                        );
                    }
                }
            }
            s_b1.line(&format!("{ops_s} triple B1 delta={delta:e}"));
            s_cmb.line(&format!("{ops_s} triple min(S,B1) delta={delta:e}"));
            s_mut.line(&format!(
                "{ops_s} triple MUTANT B1 extents*0.5 delta={delta:e}"
            ));
        }
        // B2 on a stride sample of significant triples
        let sig: Vec<usize> = (0..t3.len()).filter(|&i| t3[i] > 1e-12).collect();
        let stride = (sig.len() / b2n).max(1);
        let samp: Vec<usize> = sig.iter().copied().step_by(stride).collect();
        let vals: Vec<(f64, f64)> = samp
            .par_iter()
            .map(|&i| {
                let (p, k) = (i / np, i % np);
                (b2(&c.pp[p], &c.pm[k], &kern), t3[i])
            })
            .collect();
        let (mut s2, mut s2m) = (Stat::default(), Stat::default());
        for &(b, t) in &vals {
            s2.add(b, t);
            s2m.add(0.5 * b, t);
        }
        s2.line(&format!("{ops_s} triple B2 (sample {})", samp.len()));
        s2m.line(&format!("{ops_s} triple MUTANT B2*0.5"));
        if quart {
            let t4 = true_quartets(&c, op);
            let idx = |i: usize, j: usize| i * (i + 1) / 2 + j;
            let mut st = Stat::default();
            for i in 0..np {
                for j in 0..=i {
                    let t = t4[idx(i, j)];
                    if t > 1e-12 {
                        st.add(qp[i] * qp[j], t);
                    }
                }
            }
            st.line(&format!("{ops_s} quartet schwarz"));
            for &delta in &deltas {
                let ext: Vec<f64> = c.pm.iter().map(|x| x.extent(delta)).collect();
                let (mut s1, mut sc, mut sm) = (Stat::default(), Stat::default(), Stat::default());
                for i in 0..np {
                    for j in 0..=i {
                        let t = t4[idx(i, j)];
                        if t > 1e-12 {
                            let b = b1(&c.pm[i], ext[i], &c.pm[j], ext[j], &kern);
                            s1.add(b, t);
                            sc.add(b.min(qp[i] * qp[j]), t);
                            sm.add(b1(&c.pm[i], 0.5 * ext[i], &c.pm[j], 0.5 * ext[j], &kern), t);
                        }
                    }
                }
                s1.line(&format!("{ops_s} quartet B1 delta={delta:e}"));
                sc.line(&format!("{ops_s} quartet min(S,B1) delta={delta:e}"));
                sm.line(&format!(
                    "{ops_s} quartet MUTANT B1 extents*0.5 delta={delta:e}"
                ));
            }
            let sig: Vec<(usize, usize)> = (0..np)
                .flat_map(|i| (0..=i).map(move |j| (i, j)))
                .filter(|&(i, j)| t4[idx(i, j)] > 1e-12)
                .collect();
            let stride = (sig.len() / b2n).max(1);
            let samp: Vec<(usize, usize)> = sig.iter().copied().step_by(stride).collect();
            let vals: Vec<(f64, f64)> = samp
                .par_iter()
                .map(|&(i, j)| (b2(&c.pm[i], &c.pm[j], &kern), t4[idx(i, j)]))
                .collect();
            let (mut s2, mut s2m) = (Stat::default(), Stat::default());
            for &(b, t) in &vals {
                s2.add(b, t);
                s2m.add(0.5 * b, t);
            }
            s2.line(&format!("{ops_s} quartet B2 (sample {})", samp.len()));
            s2m.line(&format!("{ops_s} quartet MUTANT B2*0.5"));
        }
    }
    let _ = (&c.omaj, &c.dmaj);
}

fn count() {
    let c = setup();
    let ops = env("OPS", "terfc:1.0:0.70710678");
    let deltas = flist(&env("DELTAS", "1e-4,1e-8"));
    let taus = flist(&env("TAUS", "1e-6,1e-8"));
    let oracle = env("ORACLE", "1") == "1";
    let np = c.pairs.len();
    let nsa = c.dfbs.nshells();
    for ops_s in ops.split(';') {
        let (op, kern) = parse_op(ops_s);
        let qp = qpair(&c, op);
        let qa = qaux(&c, op);
        let t3 = if oracle { true_triples(&c, op) } else { vec![] };
        // cost weight per triple: nprim(P)*nprim(pair)*nfn(P)*nfn(pair)
        let w =
            |p: usize, k: usize| (c.pp[p].nprim * c.pm[k].nprim * c.pp[p].nfn * c.pm[k].nfn) as f64;
        let wtot: f64 = (0..nsa)
            .map(|p| (0..np).map(|k| w(p, k)).sum::<f64>())
            .sum();
        let ntot = (nsa * np) as f64;
        for &delta in &deltas {
            let ext_p: Vec<f64> = c.pp.iter().map(|x| x.extent(delta)).collect();
            let ext_k: Vec<f64> = c.pm.iter().map(|x| x.extent(delta)).collect();
            for &tau in &taus {
                // counts: [oracle, schwarz, b1, comb]; unweighted n and weighted
                let mut n = [0f64; 4];
                let mut wn = [0f64; 4];
                let mut miss = [0usize; 3];
                let mut fp = [0usize; 3]; // survivors with true < tau
                for p in 0..nsa {
                    for k in 0..np {
                        let s = qa[p] * qp[k] >= tau;
                        let b = b1(&c.pp[p], ext_p[p], &c.pm[k], ext_k[k], &kern);
                        let b1s = b >= tau;
                        let cm = s && b1s;
                        let flags = [oracle && t3[p * np + k] >= tau, s, b1s, cm];
                        let ww = w(p, k);
                        for (i, &f) in flags.iter().enumerate() {
                            if f {
                                n[i] += 1.0;
                                wn[i] += ww;
                            }
                        }
                        if oracle {
                            let tr = flags[0];
                            for (i, &f) in flags[1..].iter().enumerate() {
                                if tr && !f {
                                    miss[i] += 1;
                                }
                                if f && !tr {
                                    fp[i] += 1;
                                }
                            }
                        }
                    }
                }
                println!(
                    "COUNT3 {ops_s} delta={delta:e} tau={tau:e} total={ntot:.0} \
                     frac[oracle,schwarz,B1,min(S,B1)]=[{:.4},{:.4},{:.4},{:.4}] \
                     wfrac=[{:.4},{:.4},{:.4},{:.4}] miss[S,B1,cmb]={:?} falsepos[S,B1,cmb]={:?}",
                    n[0] / ntot,
                    n[1] / ntot,
                    n[2] / ntot,
                    n[3] / ntot,
                    wn[0] / wtot,
                    wn[1] / wtot,
                    wn[2] / wtot,
                    wn[3] / wtot,
                    miss,
                    fp
                );
            }
        }
        // B2 (rigorous Gaussian-majorant) on a stride sample of ALL triples
        {
            let b2n: usize = env("B2_N", "20000").parse().unwrap();
            let stride = ((nsa * np) / b2n).max(1);
            let samp: Vec<usize> = (0..nsa * np).step_by(stride).collect();
            let vals: Vec<f64> = samp
                .par_iter()
                .map(|&i| b2(&c.pp[i / np], &c.pm[i % np], &kern))
                .collect();
            for &tau in &taus {
                let (mut n, mut wn, mut wt) = ([0f64; 4], [0f64; 4], 0f64);
                let mut miss = 0usize;
                for (si, &i) in samp.iter().enumerate() {
                    let (p, k) = (i / np, i % np);
                    let ww = w(p, k);
                    wt += ww;
                    let s = qa[p] * qp[k] >= tau;
                    let v = vals[si] >= tau;
                    let fl = [oracle && t3[i] >= tau, s, v, s && v];
                    for (m, &f) in fl.iter().enumerate() {
                        if f {
                            n[m] += 1.0;
                            wn[m] += ww;
                        }
                    }
                    if fl[0] && !v {
                        miss += 1;
                    }
                }
                let ns = samp.len() as f64;
                println!(
                    "COUNT3B2 {ops_s} tau={tau:e} sample={} frac[oracle,schwarz,B2,min(S,B2)]=[{:.4},{:.4},{:.4},{:.4}] wfrac=[{:.4},{:.4},{:.4},{:.4}] B2miss={miss}",
                    samp.len(), n[0]/ns, n[1]/ns, n[2]/ns, n[3]/ns, wn[0]/wt, wn[1]/wt, wn[2]/wt, wn[3]/wt
                );
            }
        }
        // PRIMITIVE-level screening inside shell triples (the terfc 3-index MD driver
        // has NO primitive screening: every primitive triple pays full cost).
        {
            let pn: usize = env("PRIM_N", "3000").parse().unwrap();
            let stride = ((nsa * np) / pn).max(1);
            let samp: Vec<usize> = (0..nsa * np).step_by(stride + 1).collect();
            let geps = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
            for &tau in &taus {
                // per sample: (total prim triples*nfn, surv_coul*nfn, surv_kern*nfn, S flag)
                let rows: Vec<(f64, f64, f64, bool, f64, f64)> = samp
                    .par_iter()
                    .map(|&i| {
                        let (p, k) = (i / np, i % np);
                        let nt = (c.pp[p].nprim * c.pm[k].nprim) as f64;
                        let nfn = (c.pp[p].nfn * c.pm[k].nfn) as f64;
                        let tp = tau / nt;
                        let (mut sc, mut sk) = (0f64, 0f64);
                        for u in &c.pm[k].prims {
                            for v in &c.pp[p].prims {
                                if prim_b_coul(u, v) >= tp {
                                    sc += 1.0;
                                    if prim_b_kern(u, v, &kern, &geps) >= tp {
                                        sk += 1.0;
                                    }
                                }
                            }
                        }
                        (nt * nfn, sc * nfn, sk * nfn, qa[p] * qp[k] >= tau, nt, nfn)
                    })
                    .collect();
                let tot: f64 = rows.iter().map(|r| r.0).sum();
                let sc: f64 = rows.iter().map(|r| r.1).sum();
                let sk: f64 = rows.iter().map(|r| r.2).sum();
                let tot_s: f64 = rows.iter().filter(|r| r.3).map(|r| r.0).sum();
                let sc_s: f64 = rows.iter().filter(|r| r.3).map(|r| r.1).sum();
                let sk_s: f64 = rows.iter().filter(|r| r.3).map(|r| r.2).sum();
                println!(
                    "PRIM3 {ops_s} tau={tau:e} sample={} work(prim triples x nfn) surviving: coul-prim-bound={:.4} kernel-prim-bound={:.4} | after shell-Schwarz: shell-frac={:.4} +coul-prim={:.4} +kernel-prim={:.4} (fractions of ALL triples' primitive work)",
                    samp.len(), sc / tot, sk / tot, tot_s / tot, sc_s / tot, sk_s / tot
                );
            }
        }
        // metric (P|Q) pairs
        let ndat = nsa;
        let mut tm = vec![0.0; ndat * ndat];
        if oracle {
            let dims = c.dfbs.shell_dims();
            let rows: Vec<Vec<f64>> = (0..ndat)
                .into_par_iter()
                .map_init(
                    || Engine::new_2center(op, &c.dfbs, 1e-14).unwrap(),
                    |eng, p| {
                        (0..ndat)
                            .map(|q| {
                                let _ = dims;
                                eng.compute_eri2(&c.dfbs, p, q)
                                    .iter()
                                    .fold(0.0f64, |m, x| m.max(x.abs()))
                            })
                            .collect()
                    },
                )
                .collect();
            for (p, r) in rows.into_iter().enumerate() {
                for (q, v) in r.into_iter().enumerate() {
                    tm[p * ndat + q] = v;
                }
            }
        }
        for &delta in &deltas {
            let ext_p: Vec<f64> = c.pp.iter().map(|x| x.extent(delta)).collect();
            for &tau in &taus {
                let (mut n, mut tot) = ([0usize; 4], 0usize);
                for p in 0..ndat {
                    for q in 0..=p {
                        tot += 1;
                        let s = qa[p] * qa[q] >= tau;
                        let b = b1(&c.pp[p], ext_p[p], &c.pp[q], ext_p[q], &kern) >= tau;
                        let fl = [oracle && tm[p * ndat + q] >= tau, s, b, s && b];
                        for i in 0..4 {
                            if fl[i] {
                                n[i] += 1;
                            }
                        }
                    }
                }
                println!(
                    "COUNT2 {ops_s} delta={delta:e} tau={tau:e} total={tot} frac[oracle,schwarz,B1,cmb]=[{:.4},{:.4},{:.4},{:.4}]",
                    n[0] as f64 / tot as f64, n[1] as f64 / tot as f64, n[2] as f64 / tot as f64, n[3] as f64 / tot as f64
                );
            }
        }
        if env("QUARTETS", "0") == "1" {
            let t4 = if oracle {
                true_quartets(&c, op)
            } else {
                vec![]
            };
            let idx = |i: usize, j: usize| i * (i + 1) / 2 + j;
            let wq = |i: usize, j: usize| {
                (c.pm[i].nprim * c.pm[j].nprim * c.pm[i].nfn * c.pm[j].nfn) as f64
            };
            for &delta in &deltas {
                let ext: Vec<f64> = c.pm.iter().map(|x| x.extent(delta)).collect();
                for &tau in &taus {
                    let mut n = [0f64; 4];
                    let mut wn = [0f64; 4];
                    let (mut ntot, mut wtot) = (0f64, 0f64);
                    let mut miss = [0usize; 3];
                    for i in 0..np {
                        for j in 0..=i {
                            let ww = wq(i, j);
                            ntot += 1.0;
                            wtot += ww;
                            let s = qp[i] * qp[j] >= tau;
                            let b = b1(&c.pm[i], ext[i], &c.pm[j], ext[j], &kern) >= tau;
                            let fl = [oracle && t4[idx(i, j)] >= tau, s, b, s && b];
                            for (m, &f) in fl.iter().enumerate() {
                                if f {
                                    n[m] += 1.0;
                                    wn[m] += ww;
                                }
                            }
                            if fl[0] {
                                for (m, &f) in fl[1..].iter().enumerate() {
                                    if !f {
                                        miss[m] += 1;
                                    }
                                }
                            }
                        }
                    }
                    println!(
                        "COUNT4 {ops_s} delta={delta:e} tau={tau:e} total={ntot:.0} frac[oracle,S,B1,cmb]=[{:.4},{:.4},{:.4},{:.4}] wfrac=[{:.4},{:.4},{:.4},{:.4}] miss={:?}",
                        n[0]/ntot, n[1]/ntot, n[2]/ntot, n[3]/ntot, wn[0]/wtot, wn[1]/wtot, wn[2]/wtot, wn[3]/wtot, miss
                    );
                }
            }
        }
    }
    let _ = (&c.omaj, &c.dmaj);
    let _ = threeindex::eri3_tensor;
}

/// MODE=lmp2: production-map direct amplitude LMP2 (aux 10 Bohr, virt 12, ao_tail 1e-3,
/// batch_merge 4) with the probe hook: COUNT how many of the map-surviving triples the
/// bound would skip, and (APPLY=1) actually skip them and report the energy change.
fn lmp2() {
    use ferric_mp2::lmp2_amplitude::AmplitudeLmp2Config;
    use ferric_mp2::lmp2_direct::{amplitude_lmp2_direct, DirectConfig};
    use ferric_scf::rhf::{solve_rhf, RhfConfig};
    use ferric_scf::screening::SchwarzBounds;
    use std::sync::atomic::Ordering;
    let c = setup();
    let molf = env("MOL", "alkane_8.xyz");
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/{molf}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled(&env("OBS", "6-31g")).unwrap();
    let fc: usize = env("FC", "8").parse().unwrap();
    let eps: f64 = env("EPS", "5e-5").parse().unwrap();
    let delta: f64 = env("DELTA", "1e-6").parse().unwrap();
    let taus = flist(&env("TAUS", "1e-6,1e-8"));
    let apply = env("APPLY", "0") == "1";
    let rule: u8 = env("RULE", "2").parse().unwrap();
    let coul = Operator::coulomb();
    let bounds = SchwarzBounds::compute(coul, &c.obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &c.obs,
        coul,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    let mp = match env("MAPS", "prod").as_str() {
        "trivial" => DirectConfig {
            aux_radius_bohr: 1e6,
            virt_radius_bohr: None,
            ao_tail: 0.0,
            schwarz_skip: 0.0,
            ..Default::default()
        },
        _ => DirectConfig {
            aux_radius_bohr: env("AUXR", "10").parse().unwrap(),
            virt_radius_bohr: Some(env("VIRTR", "12").parse().unwrap()),
            ao_tail: env("AOTAIL", "1e-3").parse().unwrap(),
            schwarz_skip: 0.0,
            batch_merge: 4,
            ..Default::default()
        },
    };
    let cfg = AmplitudeLmp2Config {
        eps,
        frozen_core: fc,
        compute_reference: false,
        ..Default::default()
    };
    for ops_s in env("OPS", "terfc:1.0:0.70710678").split(';') {
        let (op, kern) = parse_op(ops_s);
        let qp = qpair(&c, op);
        let qa = qaux(&c, op);
        let mut e_base = f64::NAN;
        for (ti, &tau) in std::iter::once(&0.0).chain(taus.iter()).enumerate() {
            if ti == 0 && env("SKIPBASE", "0") == "1" {
                continue;
            }
            {
                let mut g = PROBE.write().unwrap();
                *g = if ti == 0 {
                    None
                } else {
                    Some(ProbeSkip {
                        tau,
                        apply,
                        rule,
                        kern,
                        ext_p: c.pp.iter().map(|x| x.extent(delta)).collect(),
                        ext_k: c.pm.iter().map(|x| x.extent(delta)).collect(),
                        pm: c.pm.clone(),
                        pp: c.pp.clone(),
                        qa: qa.clone(),
                        qp: qp.clone(),
                        b2_stride: env("B2_STRIDE", "0").parse().unwrap(),
                        no_eval: env("NOEVAL", "0") == "1",
                        seen: new_bitset(c.pp.len() * c.pm.len()),
                        sk_s: new_bitset(c.pp.len() * c.pm.len()),
                        sk_cmb: new_bitset(c.pp.len() * c.pm.len()),
                    })
                };
            }
            for x in CNT.iter() {
                x.store(0, Ordering::Relaxed);
            }
            let t = std::time::Instant::now();
            let res = amplitude_lmp2_direct(&mol, &c.obs, &obs_bs, &c.dfbs, op, &rhf, &cfg, &mp);
            let wall = t.elapsed().as_secs_f64();
            let (r, st) = match res {
                Ok(x) => x,
                Err(e) => {
                    let n: Vec<f64> = CNT
                        .iter()
                        .map(|x| x.load(Ordering::Relaxed) as f64)
                        .collect();
                    let (u, us, uc) = uniq_counts();
                    println!(
                        "UNIQ {molf} {ops_s} tau={tau:e} global_triples={} unique_map_survivors={u} ({:.4} of global) uniq S-skippable={:.4} uniq S|B1-skippable={:.4}",
                        c.pp.len() * c.pm.len(), u as f64 / (c.pp.len() * c.pm.len()) as f64, us as f64 / (u.max(1)) as f64, uc as f64 / (u.max(1)) as f64
                    );
                    println!(
                        "LMP2ERR {molf} {ops_s} tau={tau:e} ({e}) | map-survivor triples seen={:.0} skippable frac S={:.4} B1={:.4} cmb={:.4} (weighted S={:.4} B1={:.4} cmb={:.4}) | B2 sample n={:.0}: skippable S={:.4} B2={:.4} S|B2={:.4}",
                        n[0], n[2] / n[0].max(1.0), n[3] / n[0].max(1.0), n[4] / n[0].max(1.0),
                        n[5] / n[1].max(1.0), n[6] / n[1].max(1.0), n[7] / n[1].max(1.0),
                        n[8], n[9] / n[8].max(1.0), n[10] / n[8].max(1.0), n[11] / n[8].max(1.0)
                    );
                    continue;
                }
            };
            if ti == 0 {
                e_base = r.e_corr;
            }
            let n: Vec<f64> = CNT
                .iter()
                .map(|x| x.load(Ordering::Relaxed) as f64)
                .collect();
            let (u, us, uc) = uniq_counts();
            println!(
                "UNIQ {molf} {ops_s} tau={tau:e} global_triples={} unique_map_survivors={u} ({:.4} of global) uniq S-skippable={:.4} uniq S|B1-skippable={:.4}",
                c.pp.len() * c.pm.len(), u as f64 / (c.pp.len() * c.pm.len()) as f64, us as f64 / (u.max(1)) as f64, uc as f64 / (u.max(1)) as f64
            );
            println!(
                "LMP2 {molf} {ops_s} maps={} eps={eps:e} delta={delta:e} tau={tau:e} apply={apply} rule={rule} e_corr={:.10} dE_vs_base={:.3e} rel={:.3e} triples_eval={} skipped={} wall={wall:.1} t_eri3={:.1} t_metric={:.1} t_pairs={:.1} | map-survivor triples seen={:.0} skippable frac S={:.4} B1={:.4} cmb={:.4} (weighted S={:.4} B1={:.4} cmb={:.4}) | B2 sample n={:.0}: skippable S={:.4} B2={:.4} S|B2={:.4}",
                env("MAPS", "prod"), r.e_corr, r.e_corr - e_base, (r.e_corr - e_base).abs() / e_base.abs(),
                st.n_eri3_shell_triples, st.n_eri3_skipped, st.t_eri3_s, st.t_metric_s, st.t_pairs_s,
                n[0], n[2] / n[0].max(1.0), n[3] / n[0].max(1.0), n[4] / n[0].max(1.0),
                n[5] / n[1].max(1.0), n[6] / n[1].max(1.0), n[7] / n[1].max(1.0),
                n[8], n[9] / n[8].max(1.0), n[10] / n[8].max(1.0), n[11] / n[8].max(1.0)
            );
        }
    }
}
