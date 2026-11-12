//! NEVPT RDMs (`ferric_ci::rdm3`): the 3-RDM, the `(aa|aa)`-contracted 4-RDM
//! intermediates `f3ca`/`f3ac`, the hole RDMs, and the dense eigensolve
//! fallback of the CAS-CI driver.
//!
//! Inputs are ferric's own CI vectors and active integrals, read from
//! `testdata/reference/casci_<system>_rdm3.json` (written by
//! `scripts/gen_casci_rdm3_refs.py` from `examples/casci_rdm_dump.rs`):
//! H2O / 6-31G CAS(4,4) (36 determinants) and N2 / cc-pVDZ CAS(6,6) at 1.10 Å
//! and 2.20 Å (400), the windows of `tests/casci_rdm_common`. No SCF runs
//! here, so every comparison is between two constructions on the same vector.
//!
//! # Exactness anchors (asserted first)
//!
//! * Partial traces: `Σ_x dm3[p,q,r,s,x,x] = (N−2) dm2[p,q,r,s]`,
//!   `Σ_x dm2[p,q,x,x] = (N−1) dm1[p,q]`, and for the holes
//!   `Σ_x hdm3[p,q,x,a,b,x] = (2n−N−2) hdm2[p,q,a,b]`,
//!   `Σ_x hdm2[p,x,a,x] = (2n−N−1) hdm1[p,a]`.
//! * Single determinant: every RDM is the antisymmetrized product of the
//!   spin-resolved 1-RDM (particles) or of `η = 1 − γᵀ` (holes); the fully
//!   occupied space has every hole RDM zero.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! Both anchors are blind to a wrong sign or index order in the cumulant: a
//! single determinant has no cumulant, and a trace sums it away. So:
//!
//! * if the construction is right, the anchors hold to rounding AND every
//!   tensor equals PySCF on the same CI vector within the bar below, on the
//!   stretched N2 (largest cumulant) as on the others;
//! * if a phase, an index of the reordering, or a spin block is wrong, the
//!   anchors may still pass, but the PySCF comparison fails, and it fails
//!   most clearly at 2.20 Å.
//!
//! The independent dense construction (Fock-space operators over `2n` spin
//! orbitals, H2O only) checks the same definitions without PySCF, including
//! `f3ca`/`f3ac` as explicit contractions of a stored E-product 4-RDM.
//!
//! # Bars
//!
//! Element-wise `100 · N_det · ε · max|ref|` per tensor (the bar of
//! `scripts/prototypes/nevpt2_derivation.md`). The traces sum `n` elements,
//! so their bar is `n` times the element bar of the summed tensor. The dense
//! construction sums over the Fock space, so `N_det` there is replaced by
//! `max(N_det, 2^{2n})`.
//!
//! The references store `dm3`/`hdm3` as symmetry-unique elements (expanded
//! here and compared element-wise) and `f3ca`/`f3ac` in full for H2O only;
//! for every system they also store 32 projections `⟨P_k, f3⟩` onto `±1`
//! tensors, compared within `n⁶` element bars (`|⟨P, δ⟩| ≤ Σ_i |δ_i|`).
//!
//! # Measured (2026-11-12)
//!
//! See the `eprintln!` lines; the `tests/casci_rdm3.rs` row of the Checks
//! table in `crates/ferric-ci/theory.md` records them and the mutations
//! that fail this file.

mod casci_rdm_common;

use casci_rdm_common::workspace_root;
use ferric_ci::hamiltonian::{dense_hamiltonian, DeterminantSpace};
use ferric_ci::integrals::ActiveSpaceIntegrals;
use ferric_ci::rdm3::{hole_rdms, nevpt_rdms, HoleRdms, NevptRdms};
use ferric_ci::strings::enumerate_strings;
use ferric_ci::{run_cas_ci_on_integrals, CasCiConfig, DENSE_SOLVE_MAX_DETS};
use ndarray::Array2;
use serde_json::Value;

const EPS: f64 = f64::EPSILON;

struct Ref {
    v: Value,
    ints: ActiveSpaceIntegrals,
    n_elec: (usize, usize),
    c: Vec<f64>,
    ndet: usize,
}

fn f64s(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("reference field {key} missing"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn load(name: &'static str) -> Ref {
    let path = workspace_root()
        .join("testdata/reference")
        .join(format!("casci_{name}_rdm3.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing reference {} ({e}); regenerate with scripts/gen_casci_rdm3_refs.py",
            path.display()
        )
    });
    let v: Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()));
    let n = v["n_active"].as_u64().expect("n_active") as usize;
    let ne = v["n_elec_active"].as_array().expect("n_elec_active");
    let n_elec = (
        ne[0].as_u64().unwrap() as usize,
        ne[1].as_u64().unwrap() as usize,
    );
    // The reference CI vector is in ferric's string order.
    let strs = |key: &str| -> Vec<u64> {
        v[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect()
    };
    assert_eq!(strs("alpha_strings"), enumerate_strings(n, n_elec.0));
    assert_eq!(strs("beta_strings"), enumerate_strings(n, n_elec.1));
    let ints = ActiveSpaceIntegrals {
        n_active: n,
        h: Array2::from_shape_vec((n, n), f64s(&v, "h")).unwrap(),
        eri: f64s(&v, "eri"),
        e_core: v["e_core"].as_f64().unwrap(),
    };
    let c = f64s(&v, "ci_vector");
    let ndet = c.len();
    Ref {
        v,
        ints,
        n_elec,
        c,
        ndet,
    }
}

fn max_abs(x: impl Iterator<Item = f64>) -> f64 {
    x.fold(0.0f64, |m, v| m.max(v.abs()))
}

fn max_diff<'a>(a: impl Iterator<Item = &'a f64>, b: &[f64]) -> f64 {
    let mut m = 0.0f64;
    let mut count = 0;
    for (x, y) in a.zip(b) {
        m = m.max((x - y).abs());
        count += 1;
    }
    assert_eq!(count, b.len(), "tensor sizes differ");
    m
}

fn rdms(r: &Ref) -> (NevptRdms, HoleRdms) {
    let d = nevpt_rdms(&r.ints, r.n_elec, &r.c, None).expect("nevpt_rdms");
    let h = hole_rdms(&d.dm1, &d.dm2, &d.dm3, None).expect("hole_rdms");
    (d, h)
}

fn traces(name: &'static str) {
    let r = load(name);
    let (d, h) = rdms(&r);
    let n = r.ints.n_active;
    let ne = (r.n_elec.0 + r.n_elec.1) as f64;
    let nh = 2.0 * n as f64 - ne;
    let bar = |t: f64| n as f64 * 100.0 * r.ndet as f64 * EPS * t.max(1.0);
    let b3 = bar(max_abs(d.dm3.iter().copied()));
    let b2 = bar(max_abs(d.dm2.iter().copied()));
    let bh3 = bar(max_abs(h.hdm3.iter().copied()));
    let bh2 = bar(max_abs(h.hdm2.iter().copied()));
    let (mut t3, mut t2, mut th3, mut th2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for p in 0..n {
        for q in 0..n {
            let s2: f64 = (0..n).map(|x| d.dm2[[p, q, x, x]]).sum();
            t2 = t2.max((s2 - (ne - 1.0) * d.dm1[[p, q]]).abs());
            let sh2: f64 = (0..n).map(|x| h.hdm2[[p, x, q, x]]).sum();
            th2 = th2.max((sh2 - (nh - 1.0) * h.hdm1[[p, q]]).abs());
            for a in 0..n {
                for b in 0..n {
                    let s3: f64 = (0..n).map(|x| d.dm3[[p, q, a, b, x, x]]).sum();
                    t3 = t3.max((s3 - (ne - 2.0) * d.dm2[[p, q, a, b]]).abs());
                    let sh3: f64 = (0..n).map(|x| h.hdm3[[p, q, x, a, b, x]]).sum();
                    th3 = th3.max((sh3 - (nh - 2.0) * h.hdm2[[p, q, a, b]]).abs());
                }
            }
        }
    }
    eprintln!(
        "{name}: traces dm3->dm2 {t3:.2e} (bar {b3:.2e}) dm2->dm1 {t2:.2e} (bar {b2:.2e}) \
         hdm3->hdm2 {th3:.2e} (bar {bh3:.2e}) hdm2->hdm1 {th2:.2e} (bar {bh2:.2e})"
    );
    assert!(t3 <= b3, "{name}: dm3 partial trace off by {t3:.2e}");
    assert!(t2 <= b2, "{name}: dm2 partial trace off by {t2:.2e}");
    assert!(th3 <= bh3, "{name}: hdm3 partial trace off by {th3:.2e}");
    assert!(th2 <= bh2, "{name}: hdm2 partial trace off by {th2:.2e}");
}

/// The 12 images of a rank-6 index under a permutation of the three index
/// pairs `pairs` and the simultaneous swap within every pair (the storage
/// symmetry of `scripts/gen_casci_rdm3_refs.py`).
fn images(idx: [usize; 6], pairs: [(usize, usize); 3]) -> Vec<[usize; 6]> {
    const PERMS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut out = Vec::with_capacity(12);
    for perm in PERMS {
        for swap in [false, true] {
            let mut new = [0usize; 6];
            for (dst, &k) in pairs.iter().zip(&perm) {
                let src = pairs[k];
                let (mut a, mut b) = (idx[src.0], idx[src.1]);
                if swap {
                    std::mem::swap(&mut a, &mut b);
                }
                new[dst.0] = a;
                new[dst.1] = b;
            }
            out.push(new);
        }
    }
    out
}

const DM3_PAIRS: [(usize, usize); 3] = [(0, 1), (2, 3), (4, 5)];
const HDM3_PAIRS: [(usize, usize); 3] = [(0, 3), (1, 4), (2, 5)];

/// Full row-major rank-6 tensor from its symmetry-unique elements.
fn expand_unique(vals: &[f64], n: usize, pairs: [(usize, usize); 3]) -> Vec<f64> {
    let unflat = |k: usize| -> [usize; 6] {
        let mut idx = [0usize; 6];
        let mut r = k;
        for slot in idx.iter_mut().rev() {
            *slot = r % n;
            r /= n;
        }
        idx
    };
    let mut map = std::collections::HashMap::new();
    let mut it = vals.iter();
    let total = n.pow(6);
    let mut canon = Vec::with_capacity(total);
    for k in 0..total {
        let idx = unflat(k);
        let c = *images(idx, pairs).iter().min().unwrap();
        if c == idx {
            map.insert(c, *it.next().expect("too few unique elements"));
        }
        canon.push(c);
    }
    assert!(it.next().is_none(), "too many unique elements");
    canon.iter().map(|c| map[c]).collect()
}

/// `±1` from the splitmix64 finalizer, as in the generator.
fn proj_sign(k: u64, i: u64) -> f64 {
    let mut z = k
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(i.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(1);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    if z >> 63 == 0 {
        1.0
    } else {
        -1.0
    }
}

fn vs_pyscf(name: &'static str) {
    let r = load(name);
    let (d, h) = rdms(&r);
    let n = r.ints.n_active;
    let elem_bar = |scale: f64| 100.0 * r.ndet as f64 * EPS * scale;
    let mut failures = Vec::new();
    let mut worst = 0.0f64;
    let mut check = |key: &str, got: Vec<f64>, want: Vec<f64>| {
        let scale = max_abs(want.iter().copied());
        let bar = elem_bar(scale);
        let diff = max_diff(got.iter(), &want);
        worst = worst.max(diff / bar);
        eprintln!("{name}: {key} vs PySCF {diff:.2e} (bar {bar:.2e}, max|ref| {scale:.3})");
        if diff > bar {
            failures.push(format!("{key}: {diff:.2e} > {bar:.2e}"));
        }
    };
    let flat = |a: ndarray::ArrayViewD<'_, f64>| a.iter().copied().collect::<Vec<f64>>();
    check("dm1", flat(d.dm1.view().into_dyn()), f64s(&r.v, "dm1"));
    check("dm2", flat(d.dm2.view().into_dyn()), f64s(&r.v, "dm2"));
    check(
        "dm3",
        flat(d.dm3.view().into_dyn()),
        expand_unique(&f64s(&r.v, "dm3_unique"), n, DM3_PAIRS),
    );
    check("hdm1", flat(h.hdm1.view().into_dyn()), f64s(&r.v, "hdm1"));
    check("hdm2", flat(h.hdm2.view().into_dyn()), f64s(&r.v, "hdm2"));
    check(
        "hdm3",
        flat(h.hdm3.view().into_dyn()),
        expand_unique(&f64s(&r.v, "hdm3_unique"), n, HDM3_PAIRS),
    );
    if r.v.get("f3ca").is_some() {
        check("f3ca", flat(d.f3ca.view().into_dyn()), f64s(&r.v, "f3ca"));
        check("f3ac", flat(d.f3ac.view().into_dyn()), f64s(&r.v, "f3ac"));
    }
    // Projections: |<P, T - T_ref>| <= Σ|P_i| |δ_i| <= N_elem · element bar.
    for (key, t) in [("f3ca", &d.f3ca), ("f3ac", &d.f3ac)] {
        let want = f64s(&r.v, &format!("{key}_projections"));
        let scale = r.v[format!("{key}_max_abs")].as_f64().unwrap();
        let bar = t.len() as f64 * elem_bar(scale);
        let mut diff = 0.0f64;
        for (k, w) in want.iter().enumerate() {
            let got: f64 = t
                .iter()
                .enumerate()
                .map(|(i, &x)| proj_sign(k as u64, i as u64) * x)
                .sum();
            diff = diff.max((got - w).abs());
        }
        worst = worst.max(diff / bar);
        eprintln!(
            "{name}: {key} {} projections vs PySCF {diff:.2e} (bar {bar:.2e})",
            want.len()
        );
        if diff > bar {
            failures.push(format!("{key} projections: {diff:.2e} > {bar:.2e}"));
        }
    }
    eprintln!("{name}: worst diff / bar {worst:.3}");
    assert!(failures.is_empty(), "{name} vs PySCF: {failures:?}");
}

#[test]
fn partial_traces_h2o_cas44() {
    traces("h2o");
}

#[test]
fn partial_traces_n2_r1p10_cas66() {
    traces("n2_r1.10");
}

#[test]
fn partial_traces_n2_r2p20_cas66() {
    traces("n2_r2.20");
}

/// Spin-orbital Wick product: `Σ_spins Σ_π sgn π Π_i δ(s_i, s_π(i)) m^{s_i}[x_i, y_π(i)]`
/// over `k` index pairs, with `m[s]` the spin-resolved one-body matrix.
fn wick(k: usize, x: &[usize], y: &[usize], m: &[Array2<f64>; 2]) -> f64 {
    let perms: Vec<(Vec<usize>, f64)> = match k {
        1 => vec![(vec![0], 1.0)],
        2 => vec![(vec![0, 1], 1.0), (vec![1, 0], -1.0)],
        3 => vec![
            (vec![0, 1, 2], 1.0),
            (vec![1, 2, 0], 1.0),
            (vec![2, 0, 1], 1.0),
            (vec![1, 0, 2], -1.0),
            (vec![0, 2, 1], -1.0),
            (vec![2, 1, 0], -1.0),
        ],
        _ => unreachable!(),
    };
    let mut total = 0.0;
    for spins in 0..(1usize << k) {
        let s = |i: usize| (spins >> i) & 1;
        for (pi, sign) in &perms {
            let mut prod = *sign;
            for i in 0..k {
                if s(i) != s(pi[i]) {
                    prod = 0.0;
                    break;
                }
                prod *= m[s(i)][[x[i], y[pi[i]]]];
            }
            total += prod;
        }
    }
    total
}

fn single_determinant(n: usize, alpha: u64, beta: u64) {
    let (na, nb) = (alpha.count_ones() as usize, beta.count_ones() as usize);
    let sa = enumerate_strings(n, na);
    let sb = enumerate_strings(n, nb);
    let ia = sa.iter().position(|&s| s == alpha).unwrap();
    let ib = sb.iter().position(|&s| s == beta).unwrap();
    let mut c = vec![0.0; sa.len() * sb.len()];
    c[ia * sb.len() + ib] = 1.0;
    // Integrals do not enter dm3 or the hole RDMs; any (aa|aa) of the right
    // size will do.
    let ints = ActiveSpaceIntegrals {
        n_active: n,
        h: Array2::zeros((n, n)),
        eri: (0..n.pow(4)).map(|i| (i as f64 * 0.37).sin()).collect(),
        e_core: 0.0,
    };
    let d = nevpt_rdms(&ints, (na, nb), &c, None).unwrap();
    let h = hole_rdms(&d.dm1, &d.dm2, &d.dm3, None).unwrap();
    let occ = |s: u64| {
        Array2::from_shape_fn(
            (n, n),
            |(p, q)| {
                if p == q && s >> p & 1 == 1 {
                    1.0
                } else {
                    0.0
                }
            },
        )
    };
    let gamma = [occ(alpha), occ(beta)];
    let eta = [
        Array2::from_shape_fn((n, n), |(p, q)| {
            f64::from(u8::from(p == q)) - gamma[0][[q, p]]
        }),
        Array2::from_shape_fn((n, n), |(p, q)| {
            f64::from(u8::from(p == q)) - gamma[1][[q, p]]
        }),
    ];
    let bar = 100.0 * 8.0 * EPS;
    let mut e3 = 0.0f64;
    for ((p1, q1, p2, q2, p3, q3), &v) in d.dm3.indexed_iter() {
        e3 = e3.max((v - wick(3, &[p1, p2, p3], &[q1, q2, q3], &gamma)).abs());
    }
    let mut e2 = 0.0f64;
    for ((p1, q1, p2, q2), &v) in d.dm2.indexed_iter() {
        e2 = e2.max((v - wick(2, &[p1, p2], &[q1, q2], &gamma)).abs());
    }
    let mut h3 = 0.0f64;
    for ((p, q, r, a, b, cc), &v) in h.hdm3.indexed_iter() {
        h3 = h3.max((v - wick(3, &[p, q, r], &[a, b, cc], &eta)).abs());
    }
    let mut h2 = 0.0f64;
    for ((p, q, a, b), &v) in h.hdm2.indexed_iter() {
        h2 = h2.max((v - wick(2, &[p, q], &[a, b], &eta)).abs());
    }
    let mut h1 = 0.0f64;
    for ((p, a), &v) in h.hdm1.indexed_iter() {
        h1 = h1.max((v - wick(1, &[p], &[a], &eta)).abs());
    }
    eprintln!(
        "single det n={n} a={alpha:#b} b={beta:#b}: dm2 {e2:.1e} dm3 {e3:.1e} hdm1 {h1:.1e} \
         hdm2 {h2:.1e} hdm3 {h3:.1e} (bar {bar:.1e})"
    );
    for (what, e) in [
        ("dm2", e2),
        ("dm3", e3),
        ("hdm1", h1),
        ("hdm2", h2),
        ("hdm3", h3),
    ] {
        assert!(e <= bar, "single determinant {what} off by {e:.2e}");
    }
}

/// Open-shell-like occupation (α {0,2}, β {1,2}) in four orbitals, a
/// closed-shell one in five, the fully occupied space (every hole RDM zero)
/// and a single orbital.
#[test]
fn single_determinant_is_the_antisymmetrized_product() {
    single_determinant(4, 0b0101, 0b0110);
    single_determinant(5, 0b00111, 0b00111);
    single_determinant(3, 0b111, 0b111);
    // One orbital: the n² = 1 products of the build.
    single_determinant(1, 0b1, 0b0);
}

#[test]
fn rdm3_and_f3_match_pyscf_h2o_cas44() {
    vs_pyscf("h2o");
}

#[test]
fn rdm3_and_f3_match_pyscf_n2_r1p10_cas66() {
    vs_pyscf("n2_r1.10");
}

/// The discriminating comparison: N2 at 2.20 Å has the largest cumulant.
#[test]
fn rdm3_and_f3_match_pyscf_n2_r2p20_cas66() {
    vs_pyscf("n2_r2.20");
}

// ---- Dense Fock-space construction (independent of the string lists) ----

/// `a_m v`, modes `0..n` α and `n..2n` β, Jordan–Wigner order.
fn ann(m: usize, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (s, &x) in v.iter().enumerate() {
        if x != 0.0 && s >> m & 1 == 1 {
            let sign = if (s & ((1 << m) - 1)).count_ones().is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            out[s ^ (1 << m)] += sign * x;
        }
    }
    out
}

/// `a†_m v`.
fn cre(m: usize, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (s, &x) in v.iter().enumerate() {
        if x != 0.0 && s >> m & 1 == 0 {
            let sign = if (s & ((1 << m) - 1)).count_ones().is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            out[s | (1 << m)] += sign * x;
        }
    }
    out
}

fn epq(n: usize, p: usize, q: usize, v: &[f64]) -> Vec<f64> {
    let mut out = cre(p, &ann(q, v));
    for (o, x) in out.iter_mut().zip(cre(p + n, &ann(q + n, v))) {
        *o += x;
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Every NEVPT RDM of the H2O reference vector rebuilt from dense Fock-space
/// operators: `dm3` from annihilation chains, the hole RDMs from creation
/// chains, and `f3ca`/`f3ac` as explicit contractions of the stored E-product
/// 4-RDM `R4[a,b,c,d,e,f,g,h] = ⟨E_ab E_cd E_ef E_gh⟩` (`n⁸` = 65 536).
#[test]
fn dense_fock_space_construction_h2o_cas44() {
    let r = load("h2o");
    let n = r.ints.n_active;
    let (d, h) = rdms(&r);
    let sa = enumerate_strings(n, r.n_elec.0);
    let sb = enumerate_strings(n, r.n_elec.1);
    let dim = 1usize << (2 * n);
    let mut psi = vec![0.0; dim];
    for (ia, &a) in sa.iter().enumerate() {
        for (ib, &b) in sb.iter().enumerate() {
            psi[(a | (b << n)) as usize] = r.c[ia * sb.len() + ib];
        }
    }
    let bar_of = |t: f64| 100.0 * r.ndet.max(dim) as f64 * EPS * t;

    // dm3 and hdm3 from three-operator chains, spins summed.
    let chain = |ops: [usize; 3], create: bool| -> Vec<f64> {
        let mut v = psi.clone();
        for &m in &ops {
            v = if create { cre(m, &v) } else { ann(m, &v) };
        }
        v
    };
    let idx: Vec<[usize; 3]> = (0..n * n * n)
        .map(|i| [i / (n * n), (i / n) % n, i % n])
        .collect();
    // Element-wise sums over the 8 spin patterns (n⁶ = 4096 elements).
    let mut dm3 = vec![0.0; n.pow(6)];
    let mut hdm3 = vec![0.0; n.pow(6)];
    for spins in 0..8usize {
        let sp = |i: usize, o: usize| o + n * ((spins >> i) & 1);
        let annv: Vec<Vec<f64>> = idx
            .iter()
            .map(|x| chain([sp(0, x[0]), sp(1, x[1]), sp(2, x[2])], false))
            .collect();
        let crev: Vec<Vec<f64>> = idx
            .iter()
            .map(|x| chain([sp(0, x[0]), sp(1, x[1]), sp(2, x[2])], true))
            .collect();
        for (i, p) in idx.iter().enumerate() {
            for (j, q) in idx.iter().enumerate() {
                // dm3[p1,q1,p2,q2,p3,q3] = ⟨a_{p3}a_{p2}a_{p1}ψ | a_{q3}a_{q2}a_{q1}ψ⟩.
                let k = ((((p[0] * n + q[0]) * n + p[1]) * n + q[1]) * n + p[2]) * n + q[2];
                dm3[k] += dot(&annv[i], &annv[j]);
                // hdm3[p1,p2,p3,q1,q2,q3] = ⟨a†_{p3}a†_{p2}a†_{p1}ψ | a†_{q3}a†_{q2}a†_{q1}ψ⟩
                //   = ⟨a_{p1} a_{p2} a_{p3} a†_{q3} a†_{q2} a†_{q1}⟩.
                let kh = ((((p[0] * n + p[1]) * n + p[2]) * n + q[0]) * n + q[1]) * n + q[2];
                hdm3[kh] += dot(&crev[i], &crev[j]);
            }
        }
    }
    let e_dm3 = max_diff(d.dm3.iter(), &dm3);
    let e_hdm3 = max_diff(h.hdm3.iter(), &hdm3);

    // R4 from U[(x,y),(z,w)] = E_xy E_zw ψ: R4[a..h] = U[(d,c),(b,a)] · U[(e,f),(g,h)].
    let n2 = n * n;
    let u1: Vec<Vec<f64>> = (0..n2).map(|zw| epq(n, zw / n, zw % n, &psi)).collect();
    let u2: Vec<Vec<f64>> = (0..n2 * n2)
        .map(|k| {
            let (xy, zw) = (k / n2, k % n2);
            epq(n, xy / n, xy % n, &u1[zw])
        })
        .collect();
    let mut r4 = vec![0.0; n2 * n2 * n2 * n2];
    for (a, b, c, dd) in itertools4(n) {
        let bra = &u2[(dd * n + c) * n2 + b * n + a];
        for k in 0..n2 * n2 {
            r4[((((a * n + b) * n + c) * n + dd) * n2 * n2) + k] = dot(bra, &u2[k]);
        }
    }
    let r4i = |i: [usize; 8]| {
        let mut k = 0;
        for x in i {
            k = k * n + x;
        }
        r4[k]
    };
    let g = |p: usize, q: usize, rr: usize, s: usize| r.ints.g(p, q, rr, s);
    let mut e_ca = 0.0f64;
    let mut e_ac = 0.0f64;
    let mut scale = 0.0f64;
    for ((a, b, c, dd, e, f), &got_ca) in d.f3ca.indexed_iter() {
        let mut ca = 0.0;
        let mut ac = 0.0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    ca += g(k, i, f, j) * r4i([a, b, c, dd, e, j, k, i]);
                    ac += g(i, k, j, e) * r4i([a, b, c, dd, j, f, i, k]);
                }
            }
        }
        scale = scale.max(ca.abs()).max(ac.abs());
        e_ca = e_ca.max((got_ca - ca).abs());
        e_ac = e_ac.max((d.f3ac[[a, b, c, dd, e, f]] - ac).abs());
    }
    let b3 = bar_of(max_abs(dm3.iter().copied()));
    let bh = bar_of(max_abs(hdm3.iter().copied()));
    // n³ products are summed per element.
    let bf = bar_of(scale) * (n * n * n) as f64;
    eprintln!(
        "h2o dense: dm3 {e_dm3:.2e} (bar {b3:.2e}) hdm3 {e_hdm3:.2e} (bar {bh:.2e}) \
         f3ca {e_ca:.2e} f3ac {e_ac:.2e} (bar {bf:.2e}, max {scale:.3})"
    );
    assert!(e_dm3 <= b3, "dm3 vs dense {e_dm3:.2e}");
    assert!(e_hdm3 <= bh, "hdm3 vs dense {e_hdm3:.2e}");
    assert!(e_ca <= bf, "f3ca vs explicit R4 contraction {e_ca:.2e}");
    assert!(e_ac <= bf, "f3ac vs explicit R4 contraction {e_ac:.2e}");
}

fn itertools4(n: usize) -> impl Iterator<Item = (usize, usize, usize, usize)> {
    (0..n.pow(4)).map(move |k| (k / (n * n * n), (k / (n * n)) % n, (k / n) % n, k % n))
}

/// The scratch is reserved before any allocation: a 1-byte budget is refused
/// with the method named; the default budget runs (every test above).
#[test]
fn nevpt_rdms_refuse_a_starved_budget() {
    let r = load("h2o");
    let msg = match nevpt_rdms(&r.ints, r.n_elec, &r.c, Some(1)) {
        Ok(_) => panic!("a 1-byte budget must be refused"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("NEVPT RDMs"), "{msg}");
    let (d, _) = rdms(&r);
    let msg = match hole_rdms(&d.dm1, &d.dm2, &d.dm3, Some(1)) {
        Ok(_) => panic!("a 1-byte budget must be refused"),
        Err(e) => e.to_string(),
    };
    assert!(msg.contains("hole RDMs"), "{msg}");
}

// ---- Dense eigensolve fallback ------------------------------------------

fn cas_config(r: &Ref, conv_thresh: f64) -> CasCiConfig {
    CasCiConfig {
        n_active: r.ints.n_active,
        n_elec_active: r.n_elec,
        conv_thresh,
        ..Default::default()
    }
}

/// Davidson stopped after one iteration (residual far above the threshold)
/// on a small space: the dense eigensolve still meets a tight threshold.
#[test]
fn dense_fallback_meets_a_threshold_davidson_did_not_reach_h2o_cas44() {
    let r = load("h2o");
    assert!(r.ndet <= DENSE_SOLVE_MAX_DETS);
    let thresh = 1e-13;
    let cfg = CasCiConfig {
        max_iter: 1,
        ..cas_config(&r, thresh)
    };
    let res = run_cas_ci_on_integrals(r.ints.clone(), &cfg).unwrap();
    eprintln!(
        "h2o CAS(4,4) max_iter 1 conv_thresh {thresh:e}: converged {} residual {:.2e} iterations {}",
        res.converged, res.residual_norm, res.iterations
    );
    assert!(
        res.converged,
        "not converged: residual {:e}",
        res.residual_norm
    );
    assert!(res.residual_norm < thresh);
}

/// Anchored against Davidson: where Davidson converges, the dense root has
/// the same energy within `‖r‖ + 8 ε |E| √N_det` and the same vector within
/// `1 − |⟨x_d|x_D⟩| ≤ (‖r‖/Δ)² + N_det ε` (Δ the gap to the second root).
#[test]
fn dense_root_equals_davidson_root_n2_r1p10_cas66() {
    let r = load("n2_r1.10");
    let thresh = 1e-9;
    let dav = run_cas_ci_on_integrals(r.ints.clone(), &cas_config(&r, thresh)).unwrap();
    assert!(dav.converged && dav.residual_norm < thresh);
    let n = r.ints.n_active;
    let space = DeterminantSpace {
        alpha_strings: enumerate_strings(n, r.n_elec.0),
        beta_strings: enumerate_strings(n, r.n_elec.1),
    };
    let dense = ferric_ci::dense_lowest(&r.ints, &space, 0, None).unwrap();
    let hmat = dense_hamiltonian(&r.ints, &space, None).unwrap();
    let (w, _) = ndarray_linalg::Eigh::eigh(&hmat, ndarray_linalg::UPLO::Lower).unwrap();
    let gap = w[1] - w[0];
    let de = (dense.eigenvalue - dav.e_total).abs();
    let bar_e = dav.residual_norm + 8.0 * EPS * dav.e_total.abs() * (r.ndet as f64).sqrt();
    let ov = dot(&dense.eigenvector, &dav.ci_vector).abs();
    let bar_v = (dav.residual_norm / gap).powi(2) + r.ndet as f64 * EPS;
    eprintln!(
        "n2_r1.10: |E_dense - E_dav| {de:.2e} (bar {bar_e:.2e}) 1-|<x|x>| {:.2e} (bar {bar_v:.2e}) \
         gap {gap:.4}",
        1.0 - ov
    );
    assert!(de <= bar_e);
    assert!(1.0 - ov <= bar_v);
}
