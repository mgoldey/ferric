//! Shared core of the COSX mixed-precision error law: used by
//! `tests/cosx_mixed_error_law.rs` and (through `#[path]`) by the harness
//! `benchmarks/harness/examples/cosx_mixed_error_map.rs`. Counts and errors
//! only; no timing.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::md3c1e::{Md3c1e, PrimPairSum};
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::{CosxConfig, CosxHalfTransform, CosxK};
use ferric_scf::fock::KBuilder;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

pub const JK: &str = "def2-universal-jkfit";
/// Hartree to kcal/mol (CODATA 2018 value used across ferric's docs).
pub const KCAL_PER_EH: f64 = 627.509_474_063_1;
/// Rayon workers used for every build (the box is shared).
pub const WORKERS: usize = 2;

pub struct System {
    pub label: String,
    pub basis: String,
    pub mol: Molecule,
    pub prep: PreparedBasis,
    pub d: Array2<f64>,
    pub natoms: usize,
}

pub fn pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(WORKERS)
        .build()
        .expect("pool")
}

fn rhf_cfg(cosx: CosxConfig, cosx_k: bool) -> RhfConfig {
    RhfConfig {
        k_builder: cosx_k.then(|| "cosx".to_string()),
        cosx,
        df_j_aux: Some(JK.to_string()),
        df_k_aux: Some(if cosx_k {
            String::new()
        } else {
            JK.to_string()
        }),
        energy_conv: 1e-9,
        density_conv: 1e-8,
        ..Default::default()
    }
}

/// Load `testdata/molecules/<xyz>` and converge an RI-JK RHF density.
pub fn load(label: &str, xyz: &str, basis: &str) -> System {
    let path = format!(
        "{}/../../testdata/molecules/{xyz}",
        env!("CARGO_MANIFEST_DIR")
    );
    let mol = Molecule::load_xyz(&path).expect("xyz");
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).expect("schwarz");
    let mut cfg = rhf_cfg(CosxConfig::default(), false);
    cfg.energy_conv = 1e-10;
    let r = pool().install(|| {
        solve_rhf(
            &ParallelContext::default(),
            &mol,
            &prep,
            Operator::coulomb(),
            &bounds,
            &cfg,
        )
        .expect("rhf")
    });
    assert!(r.converged, "{label}: RI-JK RHF did not converge");
    let natoms = mol.atoms.len();
    System {
        label: label.to_string(),
        basis: basis.to_string(),
        mol,
        prep,
        d: r.density_total,
        natoms,
    }
}

/// K-level configuration: production default grid, DENSE half transform (so
/// the a-priori bound is exact), no final pass (a K build never runs one).
fn k_cfg(mult: f64, route: Option<PrimPairSum>) -> CosxConfig {
    CosxConfig {
        half_transform: CosxHalfTransform::Dense,
        final_grid: None,
        fp64_multiplier: mult,
        f32_route: route,
        ..CosxConfig::default()
    }
}

#[derive(Clone, Debug)]
pub struct Row {
    pub system: String,
    pub basis: String,
    pub natoms: usize,
    pub nbf: usize,
    pub mult: f64,
    pub sum: PrimPairSum,
    pub flop_share: f64,
    pub f32_blocks: usize,
    pub f32_fallbacks: usize,
    pub max_dk: f64,
    /// max over elements of |dK| / bound (<= 1 required).
    pub max_ratio: f64,
    /// bound at the element of max |dK|, and the largest bound entry.
    pub bound_at_max: f64,
    pub max_bound: f64,
    pub de_x: f64,
    pub de_x_bound: f64,
    pub kappa_p99: f64,
    /// (|dE_SCF| Eh, iterations CPU, iterations mixed) when the SCF was run.
    pub scf: Option<(f64, usize, usize)>,
}

/// p99 of block-level `sum|terms| / |A|` over the elements above 1e-3 of the
/// block maximum, on 6 evenly spaced sub-batches of the production grid.
fn kappa_p99(sys: &System, points: &[[f64; 3]]) -> f64 {
    let kern = Md3c1e::new(&sys.prep).expect("kernel");
    let mut scr = kern.scratch();
    let nsub = points.len() / 256;
    let mut ks = Vec::new();
    for q in 0..6 {
        let c = (q * nsub) / 6;
        let pts = &points[c * 256..(c * 256 + 256).min(points.len())];
        for s1 in 0..kern.nshells() {
            for s2 in 0..=s1 {
                let need = kern.shell_dim(s1) * kern.shell_dim(s2) * pts.len();
                let (mut a, mut b, mut sab) = (vec![0.0; need], vec![0.0; need], vec![0.0; need]);
                kern.pair_block(s1, s2, pts, &mut scr, &mut a).unwrap();
                kern.pair_block_f32_bound(
                    s1,
                    s2,
                    pts,
                    &mut scr,
                    PrimPairSum::F64,
                    &mut b,
                    &mut sab,
                )
                .unwrap();
                let amax = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                ks.extend(
                    a.iter()
                        .zip(&sab)
                        .filter(|(a, _)| a.abs() > 1e-3 * amax)
                        .map(|(a, s)| s / a.abs()),
                );
            }
        }
    }
    ks.sort_by(|x, y| x.partial_cmp(y).unwrap());
    ks[((ks.len() as f64) * 0.99) as usize - 1]
}

fn scf_energy(sys: &System, cosx: CosxConfig) -> (f64, usize) {
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &sys.prep).expect("schwarz");
    let cfg = rhf_cfg(cosx, true);
    let r = pool().install(|| {
        solve_rhf(
            &ParallelContext::default(),
            &sys.mol,
            &sys.prep,
            Operator::coulomb(),
            &bounds,
            &cfg,
        )
        .expect("rhf cosx")
    });
    assert!(r.converged);
    (r.energy, r.iterations)
}

/// All rows of one system: every multiplier x every PrimPairSum; the SCF
/// comparison for the multipliers in `scf_mults` (CPU energy computed once).
pub fn run_system(sys: &System, mults: &[f64], scf_mults: &[f64]) -> Vec<Row> {
    let n = sys.prep.nbasis();
    let ctx = ParallelContext::default();
    let (k_ref, points) = pool().install(|| {
        let mut kb = CosxK::new(&ctx, &sys.mol, &sys.prep, k_cfg(0.0, None), usize::MAX).unwrap();
        let mut k = Array2::zeros((n, n));
        kb.build(&sys.d, &mut k).unwrap();
        (k, kb.grid_points().to_vec())
    });
    let kappa = pool().install(|| kappa_p99(sys, &points));
    let cpu = scf_mults
        .iter()
        .next()
        .map(|_| scf_energy(sys, CosxConfig::default()));
    let mut rows = Vec::new();
    for &m in mults {
        for sum in PrimPairSum::ALL {
            let (k, bound, t) = pool().install(|| {
                let mut kb =
                    CosxK::new(&ctx, &sys.mol, &sys.prep, k_cfg(m, Some(sum)), usize::MAX).unwrap();
                let mut k = Array2::zeros((n, n));
                kb.build(&sys.d, &mut k).unwrap();
                let bound = kb.f32_k_bound(&sys.d, sum).unwrap();
                (k, bound, *kb.last_timings())
            });
            let dk = &k - &k_ref;
            let mut max_dk = 0.0_f64;
            let (mut max_ratio, mut bound_at_max) = (0.0_f64, 0.0_f64);
            for (&e, &b) in dk.iter().zip(bound.iter()) {
                if e.abs() > max_dk {
                    max_dk = e.abs();
                    bound_at_max = b;
                }
                if b > 0.0 {
                    max_ratio = max_ratio.max(e.abs() / b);
                } else if e != 0.0 {
                    max_ratio = f64::INFINITY;
                }
            }
            let de_x = -0.25 * (&sys.d * &dk).sum();
            let de_x_bound = 0.25 * (sys.d.mapv(f64::abs) * &bound).sum();
            let scf = if scf_mults.contains(&m) {
                let (e_cpu, it_cpu) = cpu.expect("cpu energy");
                let (e_mix, it_mix) = scf_energy(
                    sys,
                    CosxConfig {
                        fp64_multiplier: m,
                        f32_route: Some(sum),
                        ..CosxConfig::default()
                    },
                );
                Some(((e_mix - e_cpu).abs(), it_cpu, it_mix))
            } else {
                None
            };
            rows.push(Row {
                system: sys.label.clone(),
                basis: sys.basis.clone(),
                natoms: sys.natoms,
                nbf: n,
                mult: m,
                sum,
                flop_share: t.route_f32_flop_share,
                f32_blocks: t.f32_blocks,
                f32_fallbacks: t.f32_fallbacks,
                max_dk,
                max_ratio,
                bound_at_max,
                max_bound: bound.iter().fold(0.0_f64, |a, &b| a.max(b)),
                de_x,
                de_x_bound,
                kappa_p99: kappa,
                scf,
            });
        }
    }
    rows
}

/// OLS slope of `y` on `x` with its standard error and degrees of freedom
/// (`n - 2`); `se = NaN` when `df == 0`.
pub fn fit_slope(x: &[f64], y: &[f64]) -> (f64, f64, usize) {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxx: f64 = x.iter().map(|v| (v - mx) * (v - mx)).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let slope = sxy / sxx;
    let df = x.len().saturating_sub(2);
    if df == 0 {
        return (slope, f64::NAN, 0);
    }
    let ssr: f64 = x
        .iter()
        .zip(y)
        .map(|(a, b)| (b - my - slope * (a - mx)).powi(2))
        .sum();
    (slope, (ssr / df as f64 / sxx).sqrt(), df)
}

/// One slope line per (multiplier, variant) over the rows of one basis:
/// `ln max|dK|`, `ln |dE_x|` (and `ln |dE_SCF|` when every row has it) vs
/// `ln N` (N = atoms).
pub struct Slopes {
    pub mult: f64,
    pub sum: PrimPairSum,
    pub dk: (f64, f64, usize),
    pub de_x: (f64, f64, usize),
    pub de_scf: Option<(f64, f64, usize)>,
}

pub fn slopes(rows: &[Row]) -> Vec<Slopes> {
    let mut out = Vec::new();
    let mut mults: Vec<f64> = rows.iter().map(|r| r.mult).collect();
    mults.dedup();
    mults.sort_by(|a, b| a.partial_cmp(b).unwrap());
    mults.dedup();
    for m in mults {
        for sum in PrimPairSum::ALL {
            let sel: Vec<&Row> = rows
                .iter()
                .filter(|r| r.mult == m && r.sum == sum)
                .collect();
            if sel.len() < 2 {
                continue;
            }
            let x: Vec<f64> = sel.iter().map(|r| (r.natoms as f64).ln()).collect();
            let col =
                |g: &dyn Fn(&Row) -> f64| -> Vec<f64> { sel.iter().map(|r| g(r).ln()).collect() };
            let de_scf = sel
                .iter()
                .all(|r| r.scf.is_some_and(|s| s.0 > 0.0))
                .then(|| fit_slope(&x, &col(&|r| r.scf.unwrap().0)));
            out.push(Slopes {
                mult: m,
                sum,
                dk: fit_slope(&x, &col(&|r| r.max_dk)),
                de_x: fit_slope(&x, &col(&|r| r.de_x.abs())),
                de_scf,
            });
        }
    }
    out
}

/// The pre-registered device-variant rule: variant `a` is NOT the device
/// variant when its `ln max|dK|` slope is more than 2 sigma (the two standard
/// errors in quadrature) above `b`'s at the same multiplier.
pub fn not_device_variant(a: &Slopes, b: &Slopes) -> bool {
    let se = (a.dk.1.powi(2) + b.dk.1.powi(2)).sqrt();
    se.is_finite() && a.dk.0 - b.dk.0 > 2.0 * se
}

pub fn print_rows(rows: &[Row]) {
    println!(
        "system basis atoms nbf mult sum flop_share f32_blocks fallbacks max|dK| bound@max max_bound max|dK|/bound dE_x dE_x_bound kappa_p99 |dE_SCF|(Eh) |dE_SCF|(kcal/mol) |dE_SCF|/atom(Eh) iters(cpu,mixed)"
    );
    for r in rows {
        let (de, kc, per, it) = match r.scf {
            Some((d, a, b)) => (
                format!("{d:.3e}"),
                format!("{:.3e}", d * KCAL_PER_EH),
                format!("{:.3e}", d / r.natoms as f64),
                format!("{a},{b}"),
            ),
            None => ("-".into(), "-".into(), "-".into(), "-".into()),
        };
        println!(
            "{} {} {} {} {:.0e} {} {:.4} {} {} {:.3e} {:.3e} {:.3e} {:.3e} {:+.3e} {:.3e} {:.1} {de} {kc} {per} {it}",
            r.system,
            r.basis,
            r.natoms,
            r.nbf,
            r.mult,
            r.sum.as_str(),
            r.flop_share,
            r.f32_blocks,
            r.f32_fallbacks,
            r.max_dk,
            r.bound_at_max,
            r.max_bound,
            r.max_ratio,
            r.de_x,
            r.de_x_bound,
            r.kappa_p99
        );
    }
}

pub fn print_slopes(sl: &[Slopes]) {
    println!("slopes vs ln(atoms): slope (se, df); df = points - 2, so with 3 points df = 1 and the se is nearly uninformative");
    for s in sl {
        let f = |t: (f64, f64, usize)| format!("{:+.2} ({:.2}, df {})", t.0, t.1, t.2);
        println!(
            "mult {:.0e} {:>14}: ln max|dK| {}  ln|dE_x| {}  ln|dE_SCF| {}",
            s.mult,
            s.sum.as_str(),
            f(s.dk),
            f(s.de_x),
            s.de_scf.map_or("-".to_string(), f)
        );
    }
}

/// Print the pre-registered selection rule between plain `F32` and
/// `CompensatedF32` (and `F64`) at every multiplier.
pub fn print_device_rule(sl: &[Slopes]) {
    let mut mults: Vec<f64> = sl.iter().map(|s| s.mult).collect();
    mults.sort_by(|a, b| a.partial_cmp(b).unwrap());
    mults.dedup();
    for m in mults {
        let at = |sum| sl.iter().find(|s| s.mult == m && s.sum == sum);
        let (Some(f32s), Some(comp), Some(f64s)) = (
            at(PrimPairSum::F32),
            at(PrimPairSum::CompensatedF32),
            at(PrimPairSum::F64),
        ) else {
            continue;
        };
        println!(
            "mult {m:.0e}: slope more than 2 sigma above CompensatedF32 -> NOT the device variant: F32 {}, F64 {}",
            not_device_variant(f32s, comp),
            not_device_variant(f64s, comp)
        );
    }
}
