//! Is `rohf_newton::hessian_matvec` an EXACT orbital Hessian?  (prerequisite
//! PR-1 of #291, analytic ROHF Hessian.)
//!
//! The matvec returns, for a rotation kappa with blocks (vc, vo, oc), the three
//! blocks of d g / d kappa, where g = `gradient_blocks_from` (g_vc = Fa+Fb[v,c],
//! g_vo = Fa[v,o], g_oc = Fb[o,c]; the TRUE energy gradient is 2 g).
//!
//! Method: central FD of g(C Cayley(eps kappa)) with the full per-spin Focks
//! rebuilt at every rotated C; compared with the analytic matvec.
//!
//! Hypotheses written BEFORE measuring.
//!   Physics (matvec exact): matvec/FD = 1 in every (input block, output block)
//!   cell, off-diagonal cells (vc->vo, oc->vo, ...) included.
//!   Artifact A (wrong constant): a clean ratio 2 or 1/2 in every cell.
//!   Artifact B (missing coupling): a cell that is 0 while its neighbours are 1
//!   (e.g. the oc column), which a single random direction would hide inside a
//!   total norm; hence the per-cell ratios and the single-block directions.
//!   Artifact C (diagonal-Fock approximation): the matvec applies only the
//!   DIAGONAL Fock gap `F[p,p]-F[q,q]` (a semicanonical form). The exact
//!   MO-rotation response is the full commutator `F kappa - kappa F`, which has
//!   off-diagonal terms such as `F_a[v,c] k_oc` and `F_a[v,v'] k_vo`. Predicted:
//!   an error that is NOT a constant ratio, is zero iff the Focks are diagonal
//!   in the rotated sub-blocks, and equals the omitted commutator exactly. The
//!   test computes that omitted piece independently and checks the residual
//!   against it.
//!
//! Pass bars are derived from measured floors (see `fd_floor`): the FD itself is
//! characterised by an h-sweep and Richardson extrapolation, and the bar is a
//! multiple of the worst measured self-disagreement of the FD, not a guess.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::engine_pool::EnginePool;
use ferric_scf::rhf::build_jk;
use ferric_scf::rohf::{solve_rohf, RohfConfig};
use ferric_scf::rohf_newton::{hessian_matvec, RohfNewtonInputs};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;
use ndarray_linalg::Solve;
use std::ops::Range;

type Blocks = [Array2<f64>; 3]; // (vc, vo, oc)

struct Xorshift64(u64);
impl Xorshift64 {
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

struct Sys {
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
    h: Array2<f64>,
    c0: Array2<f64>,
    n: usize,
    nc: usize,
    no: usize,
}

const THRESH: f64 = 1e-12;

fn setup(xyz: &str, mult: usize, basis_name: &str) -> Sys {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RohfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 300,
        ..Default::default()
    };
    let res = solve_rohf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &cfg).unwrap();
    assert!(res.converged, "ROHF reference must converge");
    let no = mult - 1;
    let nc = (mol.nelec() as usize - no) / 2;
    let h = oneelectron::hcore(&prep);
    let n = prep.nbasis();
    Sys {
        prep,
        bounds,
        ctx,
        h,
        c0: res.mos_alpha,
        n,
        nc,
        no,
    }
}

/// Per-spin MO Focks and the electronic energy (no E_nn: only differences used).
fn focks(s: &Sys, c: &Array2<f64>) -> (Array2<f64>, Array2<f64>, f64) {
    let n = s.n;
    let (nc, nocc_a) = (s.nc, s.nc + s.no);
    let ca = c.slice(ndarray::s![.., ..nocc_a]);
    let cb = c.slice(ndarray::s![.., ..nc]);
    let da = ca.dot(&ca.t());
    let db = cb.dot(&cb.t());
    let dt = &da + &db;
    let z = || Array2::<f64>::zeros((n, n));
    let (mut j, mut kd, mut ka, mut kb, mut jd) = (z(), z(), z(), z(), z());
    build_jk(&s.ctx, &s.prep, &s.bounds, THRESH, &dt, &mut j, &mut kd).unwrap();
    build_jk(&s.ctx, &s.prep, &s.bounds, THRESH, &da, &mut jd, &mut ka).unwrap();
    jd.fill(0.0);
    build_jk(&s.ctx, &s.prep, &s.bounds, THRESH, &db, &mut jd, &mut kb).unwrap();
    let fa = &s.h + &j - &ka;
    let fb = &s.h + &j - &kb;
    let e = 0.5 * ((&da * &(&s.h + &fa)).sum() + (&db * &(&s.h + &fb)).sum());
    (c.t().dot(&fa).dot(c), c.t().dot(&fb).dot(c), e)
}

fn spec(s: &Sys) -> [(Range<usize>, Range<usize>); 3] {
    let (n, nc, nocc_a) = (s.n, s.nc, s.nc + s.no);
    [
        (nocc_a..n, 0..nc),
        (nocc_a..n, nc..nocc_a),
        (nc..nocc_a, 0..nc),
    ]
}

/// g blocks, same pairing as `gradient_blocks_from` (that fn is pub(crate)).
fn grad(s: &Sys, fa: &Array2<f64>, fb: &Array2<f64>) -> Blocks {
    let sp = spec(s);
    let blk = |m: &Array2<f64>, i: usize| {
        m.slice(ndarray::s![sp[i].0.clone(), sp[i].1.clone()])
            .to_owned()
    };
    [&blk(fa, 0) + &blk(fb, 0), blk(fa, 1), blk(fb, 2)]
}

fn kappa_full(s: &Sys, k: &Blocks) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((s.n, s.n));
    for (bi, (rows, cols)) in spec(s).into_iter().enumerate() {
        for (i, p) in rows.enumerate() {
            for (j, q) in cols.clone().enumerate() {
                m[(p, q)] = k[bi][(i, j)];
                m[(q, p)] = -k[bi][(i, j)];
            }
        }
    }
    m
}

fn rotate(s: &Sys, c: &Array2<f64>, k: &Blocks, eps: f64) -> Array2<f64> {
    let kap = eps * kappa_full(s, k);
    let eye = Array2::<f64>::eye(s.n);
    let (a, b) = (&eye - &(0.5 * &kap), &eye + &(0.5 * &kap));
    let mut u = Array2::<f64>::zeros((s.n, s.n));
    for col in 0..s.n {
        let sol = a.solve(&b.column(col).to_owned()).unwrap();
        for row in 0..s.n {
            u[(row, col)] = sol[row];
        }
    }
    c.dot(&u)
}

fn rand_blocks(s: &Sys, rng: &mut Xorshift64, mask: [bool; 3]) -> Blocks {
    let (nv, nc, no) = (s.n - s.nc - s.no, s.nc, s.no);
    let mut mk = |r, c, on: bool| {
        Array2::<f64>::from_shape_fn((r, c), |_| if on { rng.next() } else { 0.0 })
    };
    [
        mk(nv, nc, mask[0]),
        mk(nv, no, mask[1]),
        mk(no, nc, mask[2]),
    ]
}

fn max_abs(b: &Blocks) -> f64 {
    b.iter()
        .flat_map(|a| a.iter())
        .fold(0.0, |m, &x| m.max(x.abs()))
}
fn sub(a: &Blocks, b: &Blocks) -> Blocks {
    [&a[0] - &b[0], &a[1] - &b[1], &a[2] - &b[2]]
}
fn add(a: &Blocks, b: &Blocks) -> Blocks {
    [&a[0] + &b[0], &a[1] + &b[1], &a[2] + &b[2]]
}
fn scaled(a: &Blocks, f: f64) -> Blocks {
    [
        a[0].mapv(|x| x * f),
        a[1].mapv(|x| x * f),
        a[2].mapv(|x| x * f),
    ]
}
fn dot(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Central FD of g along kappa (kappa has max|k| <= 1, so eps = h).
fn fd_hk(s: &Sys, c: &Array2<f64>, k: &Blocks, h: f64) -> Blocks {
    let at = |eps: f64| {
        let cr = rotate(s, c, k, eps);
        let (fa, fb, _) = focks(s, &cr);
        grad(s, &fa, &fb)
    };
    scaled(&sub(&at(h), &at(-h)), 0.5 / h)
}

/// 4th-order Richardson of the central FD at (h, 2h).
fn fd_rich(s: &Sys, c: &Array2<f64>, k: &Blocks, h: f64) -> Blocks {
    let (d1, d2) = (fd_hk(s, c, k, h), fd_hk(s, c, k, 2.0 * h));
    scaled(&sub(&scaled(&d1, 4.0), &d2), 1.0 / 3.0)
}

fn analytic(s: &Sys, c: &Array2<f64>, fa: &Array2<f64>, fb: &Array2<f64>, k: &Blocks) -> Blocks {
    let pool = EnginePool::new(s.bounds.op, &s.prep, 1e-14).unwrap();
    let inp = RohfNewtonInputs {
        prep: &s.prep,
        bounds: &s.bounds,
        c,
        f_a_mo: fa,
        f_b_mo: fb,
        nocc_double: s.nc,
        nocc_open: s.no,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: THRESH,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let (a, b, d) = hessian_matvec(&s.ctx, &inp, &k[0], &k[1], &k[2], &pool).unwrap();
    [a, b, d]
}

/// The full commutator `(F k - k F)` packed per block, and the diagonal-gap term
/// the matvec adds in its place. Returns (commutator, diag).
fn comm_and_diag(s: &Sys, fa: &Array2<f64>, fb: &Array2<f64>, k: &Blocks) -> (Blocks, Blocks) {
    let kap = kappa_full(s, k);
    let ca = fa.dot(&kap) - kap.dot(fa);
    let cb = fb.dot(&kap) - kap.dot(fb);
    let comm = grad(s, &ca, &cb);
    let mut diag = k.clone();
    for (bi, (rows, cols)) in spec(s).into_iter().enumerate() {
        for (i, p) in rows.enumerate() {
            for (j, q) in cols.clone().enumerate() {
                let gap = match bi {
                    0 => (fa[(p, p)] + fb[(p, p)]) - (fa[(q, q)] + fb[(q, q)]),
                    1 => fa[(p, p)] - fa[(q, q)],
                    _ => fb[(p, p)] - fb[(q, q)],
                };
                diag[bi][(i, j)] = gap * k[bi][(i, j)];
            }
        }
    }
    (comm, diag)
}

/// Measured FD floor on one direction: worst disagreement between Richardson
/// values taken at different h (4e-3, 2e-3, 1e-3), and the finest as reference.
fn fd_floor(s: &Sys, c: &Array2<f64>, k: &Blocks) -> (Blocks, f64) {
    let hs = [4e-3, 2e-3, 1e-3];
    let rich: Vec<Blocks> = hs.iter().map(|&h| fd_rich(s, c, k, h)).collect();
    let mut floor = 0.0f64;
    for i in 0..rich.len() {
        for j in i + 1..rich.len() {
            floor = floor.max(max_abs(&sub(&rich[i], &rich[j])));
        }
    }
    (rich[2].clone(), floor)
}

struct Cell {
    ratio: f64,
    dev: f64,
}
/// ratio = <mv,fd>/<fd,fd> and max |mv - fd| for one output block.
fn cell(mv: &Array2<f64>, fd: &Array2<f64>) -> Cell {
    let den = dot(fd, fd);
    Cell {
        ratio: if den > 1e-24 {
            dot(mv, fd) / den
        } else {
            f64::NAN
        },
        dev: (mv - fd).iter().fold(0.0, |m, &x| m.max(x.abs())),
    }
}

struct Report {
    floor: f64,
    /// worst |matvec - FD|
    raw_dev: f64,
    /// worst |(matvec - diag + commutator) - FD|
    comm_dev: f64,
    /// max |commutator - diag| (size of the omitted physics)
    omitted: f64,
}

fn analyse(s: &Sys, c: &Array2<f64>, tag: &str) -> Report {
    let (fa, fb, _) = focks(s, c);
    let mut rng = Xorshift64(0x9E37_79B9_7F4A_7C15);
    let mut dirs: Vec<(String, Blocks)> = vec![
        ("vc".into(), rand_blocks(s, &mut rng, [true, false, false])),
        ("vo".into(), rand_blocks(s, &mut rng, [false, true, false])),
        ("oc".into(), rand_blocks(s, &mut rng, [false, false, true])),
    ];
    for i in 0..2 {
        dirs.push((
            format!("all{i}"),
            rand_blocks(s, &mut rng, [true, true, true]),
        ));
    }
    let mut rep = Report {
        floor: 0.0,
        raw_dev: 0.0,
        comm_dev: 0.0,
        omitted: 0.0,
    };
    for (name, k) in &dirs {
        let (fd, floor) = fd_floor(s, c, k);
        let mv = analytic(s, c, &fa, &fb, k);
        let (comm, diag) = comm_and_diag(s, &fa, &fb, k);
        let pred = add(&sub(&mv, &diag), &comm);
        let cells = [
            cell(&mv[0], &fd[0]),
            cell(&mv[1], &fd[1]),
            cell(&mv[2], &fd[2]),
        ];
        eprintln!(
            "[{tag}] dir {name:5} floor {floor:.2e} |FD| {:.3e} | raw dev {:.3e} | +comm dev {:.3e} | omitted {:.3e}",
            max_abs(&fd),
            max_abs(&sub(&mv, &fd)),
            max_abs(&sub(&pred, &fd)),
            max_abs(&sub(&comm, &diag)),
        );
        for (o, cl) in ["vc", "vo", "oc"].iter().zip(cells.iter()) {
            eprintln!(
                "    out {o}: ratio matvec/FD = {:+.6}  max|dev| = {:.3e}",
                cl.ratio, cl.dev
            );
        }
        rep.floor = rep.floor.max(floor);
        rep.raw_dev = rep.raw_dev.max(max_abs(&sub(&mv, &fd)));
        rep.comm_dev = rep.comm_dev.max(max_abs(&sub(&pred, &fd)));
        rep.omitted = rep.omitted.max(max_abs(&sub(&comm, &diag)));
    }
    rep
}

/// Normalisation: the TRUE energy gradient is 2 g (FD of E, any point), and at
/// the converged point (g = 0) E'' = 2 kappa . (d g / d kappa) (FD of g).
fn check_normalisation(s: &Sys, c: &Array2<f64>, converged: bool) {
    let mut rng = Xorshift64(0xD1B5_4A32_D192_ED03);
    let k = rand_blocks(s, &mut rng, [true, true, true]);
    let (fa, fb, e0) = focks(s, c);
    let g = grad(s, &fa, &fb);
    let gk: f64 = (0..3).map(|i| dot(&g[i], &k[i])).sum();
    let e = |eps: f64| focks(s, &rotate(s, c, &k, eps)).2;
    if !converged {
        // Richardson (h, 2h): plain central FD of E leaves ~2e-5 relative
        // truncation error for a unit-max-norm kappa (measured, NH2/cc-pVDZ).
        let h = 1e-3;
        let cd = |h: f64| (e(h) - e(-h)) / (2.0 * h);
        let de = (4.0 * cd(h) - cd(2.0 * h)) / 3.0;
        eprintln!(
            "normalisation: dE/deps = {de:.8e}, 2 g.k = {:.8e}, ratio {:.8}",
            2.0 * gk,
            de / (2.0 * gk)
        );
        assert!(
            (de / (2.0 * gk) - 1.0).abs() < 1e-5,
            "true gradient is not 2 g"
        );
    } else {
        let hh = 5e-3;
        let d2 = (e(hh) - 2.0 * e0 + e(-hh)) / (hh * hh);
        let fd = fd_hk(s, c, &k, 1e-3);
        let kfd: f64 = (0..3).map(|i| dot(&fd[i], &k[i])).sum();
        eprintln!(
            "normalisation: E'' = {d2:.8e}, 2 k.FD(g) = {:.8e}, ratio {:.8}",
            2.0 * kfd,
            d2 / (2.0 * kfd)
        );
        assert!(
            (d2 / (2.0 * kfd) - 1.0).abs() < 1e-3,
            "E'' is not 2 k.(dg/dk)"
        );
    }
}

/// Run the whole study at the converged point and at a rotated, non-stationary
/// point; assert the findings and the mutants.
fn study(s: &Sys, tag: &str, off_converged: f64) {
    let mut rng = Xorshift64(0x1234_5678_9ABC_DEF1);
    let k0 = scaled(&rand_blocks(s, &mut rng, [true, true, true]), off_converged);
    let c_off = rotate(s, &s.c0, &k0, 1.0);
    for (point, c, converged) in [("converged", &s.c0, true), ("rotated", &c_off, false)] {
        let t = format!("{tag}/{point}");
        let rep = analyse(s, c, &t);
        check_normalisation(s, c, converged);
        let gmax = {
            let (fa, fb, _) = focks(s, c);
            max_abs(&grad(s, &fa, &fb))
        };
        eprintln!(
            "[{t}] |g|max {gmax:.2e}  FD floor {:.2e}  raw dev {:.3e}  comm-corrected dev {:.3e}  omitted-term size {:.3e}",
            rep.floor, rep.raw_dev, rep.comm_dev, rep.omitted
        );
        // Bar derived from the measured FD floor: 10x the worst FD
        // self-disagreement, never below 1e-7.
        let bar = (10.0 * rep.floor).max(1e-7);
        eprintln!(
            "[{t}] bar {bar:.2e}; omitted/bar = {:.1}",
            rep.omitted / bar
        );
        assert!(
            rep.comm_dev < bar,
            "{t}: matvec + omitted commutator must equal FD: {:.3e} >= {bar:.3e}",
            rep.comm_dev
        );
        // Non-vacuity of the explanation: the omitted term must be resolvable,
        // otherwise "matvec + commutator == FD" would also hold for the raw
        // matvec and the check above would say nothing about the commutator.
        assert!(
            rep.omitted > 10.0 * bar,
            "{t}: omitted commutator ({:.3e}) is not resolvable above the bar ({bar:.2e})",
            rep.omitted
        );
        mutants(s, c, bar, &t);
    }
}

fn mutants(s: &Sys, c: &Array2<f64>, bar: f64, tag: &str) {
    let (fa, fb, _) = focks(s, c);
    let mut rng = Xorshift64(0xA5A5_5A5A_1357_9BDF);
    let k = rand_blocks(s, &mut rng, [true, true, true]);
    let (fd, _) = fd_floor(s, c, &k);
    let mv = analytic(s, c, &fa, &fb, &k);
    let (comm, diag) = comm_and_diag(s, &fa, &fb, &k);
    let good = add(&sub(&mv, &diag), &comm);
    let dev = |p: &Blocks| max_abs(&sub(p, &fd));
    let base = dev(&good);
    eprintln!("[{tag}] mutation: unmutated dev {base:.3e} (bar {bar:.2e})");
    assert!(base < bar);
    let check = |name: &str, p: Blocks| {
        let d = dev(&p);
        eprintln!(
            "[{tag}] mutant {name:24} dev {d:.3e}  ({:.1}x bar)",
            d / bar
        );
        assert!(
            d > 10.0 * bar,
            "mutant {name} NOT detected: dev {d:.3e} vs bar {bar:.2e}"
        );
    };
    // (1) wrong constant on one block.
    check(
        "scale vc output x2",
        [good[0].mapv(|x| 2.0 * x), good[1].clone(), good[2].clone()],
    );
    // (2) halve everything (gradient/Hessian normalisation slip).
    check("all x0.5", scaled(&good, 0.5));
    // (3) drop the oc input (no closed-open coupling in).
    let mut k_no_oc = k.clone();
    k_no_oc[2].fill(0.0);
    let mv_no_oc = analytic(s, c, &fa, &fb, &k_no_oc);
    let (cm, dg) = comm_and_diag(s, &fa, &fb, &k_no_oc);
    check("drop oc input", add(&sub(&mv_no_oc, &dg), &cm));
    // (4) drop the oc output block.
    check(
        "zero oc output",
        [
            good[0].clone(),
            good[1].clone(),
            Array2::zeros(good[2].raw_dim()),
        ],
    );
    // (5) swap alpha/beta Focks in the inputs.
    let mv_sw = analytic(s, c, &fb, &fa, &k);
    let (cs, ds) = comm_and_diag(s, &fb, &fa, &k);
    check("swap Fa/Fb inputs", add(&sub(&mv_sw, &ds), &cs));
    // (6) sign flip on the vo output.
    check(
        "flip vo sign",
        [good[0].clone(), good[1].mapv(|x| -x), good[2].clone()],
    );
    // (7) the raw matvec (diag-gap only, commutator omitted): reported, and
    //     asserted resolvable in `study`'s caller when the omitted term is.
    let raw_dev = dev(&mv);
    eprintln!(
        "[{tag}] raw matvec (diag-gap only) dev {raw_dev:.3e} ({:.1}x bar)",
        raw_dev / bar
    );
}

#[test]
fn nh2_sto3g_matvec_vs_fd() {
    let s = setup(
        "3\nNH2\nN 0 0 0.12\nH 0 0.78 -0.45\nH 0 -0.78 -0.45\n",
        2,
        "sto-3g",
    );
    study(&s, "NH2/STO-3G", 0.08);
}

#[test]
fn ch2_triplet_sto3g_matvec_vs_fd() {
    let s = setup(
        "3\nCH2\nC 0 0 0.10\nH 0 0.86 -0.55\nH 0 -0.86 -0.55\n",
        3,
        "sto-3g",
    );
    study(&s, "CH2t/STO-3G", 0.08);
}

/// Larger basis. Ignored only for runtime: each case does many full J/K builds
/// (5 directions x 3 h x 2 Richardson points x 2 signs, twice) on a 24-function
/// system.
#[test]
#[ignore = "runtime: ~10 s each in release but minutes in debug (hundreds of J/K builds at cc-pVDZ); run once, release, quiet box"]
fn nh2_ccpvdz_matvec_vs_fd() {
    let s = setup(
        "3\nNH2\nN 0 0 0.12\nH 0 0.78 -0.45\nH 0 -0.78 -0.45\n",
        2,
        "cc-pvdz",
    );
    study(&s, "NH2/cc-pVDZ", 0.05);
}

#[test]
#[ignore = "runtime: ~10 s each in release but minutes in debug (hundreds of J/K builds at cc-pVDZ); run once, release, quiet box"]
fn ch2_triplet_ccpvdz_matvec_vs_fd() {
    let s = setup(
        "3\nCH2\nC 0 0 0.10\nH 0 0.86 -0.55\nH 0 -0.86 -0.55\n",
        3,
        "cc-pvdz",
    );
    study(&s, "CH2t/cc-pVDZ", 0.05);
}
