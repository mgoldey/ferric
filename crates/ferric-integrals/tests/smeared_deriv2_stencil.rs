//! Sizing probe for analytic Hessians with Gaussian-smeared external charges
//! (benchmarks/hessian-sizing/REPORT.md): the second derivative of a smeared-
//! charge attraction block `(site | μ ν)` with respect to the two basis
//! centres, obtained by a fourth-order central stencil of the FIRST-derivative
//! 3-centre blocks (`compute_eri3_deriv_shifted`, available in every libint2
//! build ferric links, including the wheel's), against an INDEPENDENT
//! construction: a fourth-order (Richardson) mixed second difference of the
//! plain VALUE blocks (`compute_eri3_shifted`).
//!
//! Run: `OPENBLAS_NUM_THREADS=1 cargo test -p ferric-integrals --test
//! smeared_deriv2_stencil -- --nocapture`

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

const ZERO: [f64; 3] = [0.0; 3];

/// Shifts `[site, sh1, sh2]` that displace ATOM `b` along `axis` by `d`: every
/// shell sitting on `b` moves (a same-atom shell pair moves rigidly, which is
/// what an atomic Hessian element needs and keeps the integral smooth on the
/// scale of the site distance instead of the tightest primitive).
fn atom_shift(atoms: [usize; 2], b: usize, axis: usize, d: f64) -> [[f64; 3]; 3] {
    shift_of(atoms, b, axis, d, false)
}

/// `atom_shift`, or (`single_shell`) the WRONG construction that moves only the
/// first shell on `b`: a same-atom pair then no longer moves rigidly.
fn shift_of(atoms: [usize; 2], b: usize, axis: usize, d: f64, single_shell: bool) -> [[f64; 3]; 3] {
    let mut sh = [ZERO; 3];
    for (k, &a) in atoms.iter().enumerate() {
        if a == b {
            sh[1 + k][axis] += d;
            if single_shell {
                break;
            }
        }
    }
    sh
}

fn add_shifts(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = a;
    for k in 0..3 {
        for c in 0..3 {
            o[k][c] += b[k][c];
        }
    }
    o
}

/// `f(sh1 shifted by a·e_ax1, sh2 shifted by b·e_ax2)` value block.
fn value(
    eng: &mut Engine,
    obs: &PreparedBasis,
    site: &PreparedBasis,
    p: usize,
    s1: usize,
    s2: usize,
    sh: [[f64; 3]; 3],
) -> Vec<f64> {
    eng.compute_eri3_shifted(obs, site, p, s1, s2, sh)
        .unwrap()
        .map(|b| b.to_vec())
        .unwrap_or_default()
}

#[test]
fn first_derivative_stencil_matches_value_second_difference() {
    // width 1.2 Bohr, 0.14 Bohr, 0.01 Bohr (a near-point charge).
    for zeta in [1.0 / (1.2f64 * 1.2), 1.0 / (0.14f64 * 0.14), 1.0e4] {
        let (worst, scale) = stencil_vs_values(zeta, false);
        assert!(scale > 1e-4, "probe is vacuous: all second derivatives ~0");
        assert!(
            worst < 1e-7,
            "stencil disagrees with value second difference: {worst:.3e}"
        );
    }
}

/// Negative control: displacing a single shell of a same-atom pair instead of
/// the whole atom misses the independent reference by orders of magnitude more
/// than the correct construction (measured 2.0e-1 for this variant vs 7e-10 at width 1.2).
#[test]
fn single_shell_displacement_is_wrong() {
    let (worst, _) = stencil_vs_values(1.0 / (1.2f64 * 1.2), true);
    assert!(
        worst > 1e-4,
        "single-shell mutant unexpectedly passes: {worst:.3e}"
    );
}

fn stencil_vs_values(zeta: f64, single_shell: bool) -> (f64, f64) {
    let mol = Molecule::parse_xyz(WATER, 0, 1).expect("water");
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    // A smeared MM site 2.1 Bohr off the molecule (zeta = 1/w^2).
    let sb = SiteBasis::new(&[[1.3, -0.4, 2.1, zeta]], 0).unwrap();
    let p = sb.site_shell[0];
    let mut e_val = Engine::new_3center(Operator::coulomb(), &obs, &sb.prep, 1e-14).unwrap();
    let mut e_der = Engine::new_3center_deriv(Operator::coulomb(), &obs, &sb.prep, 1e-14).unwrap();
    let dims = obs.shell_dims();
    let sh2at = obs.shell_to_atom();

    // Hessian elements d2/dR_{a,x} dR_{b,y} (atoms of water: 0 = O, 1/2 = H).
    let elems = [
        (0usize, 0usize, 1usize, 1usize),
        (1, 2, 1, 0),
        (0, 1, 0, 1),
        (2, 0, 1, 2),
        (1, 1, 1, 1),
    ];
    let (h1, h2) = (2e-3, 1e-3);
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    let mut nblocks = 0usize;
    for s1 in 0..obs.nshells() {
        for s2 in 0..obs.nshells() {
            let n = dims[s1] * dims[s2];
            let atoms = [sh2at[s1], sh2at[s2]];
            for &(a, x, b, y) in &elems {
                if !atoms.contains(&a) && !atoms.contains(&b) {
                    continue; // no dependence on either atom: zero by construction
                }
                // (a) fourth-order stencil, over the displacement of atom b along y, of
                // the analytic first derivative of the block with respect to atom a, x.
                let w = [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)];
                let mut st = vec![0.0; n];
                for (off, wt) in w {
                    let sh = shift_of(atoms, b, y, off * h1, single_shell);
                    let Some(blk) = e_der
                        .compute_eri3_deriv_shifted(&obs, &sb.prep, p, s1, s2, sh)
                        .unwrap()
                    else {
                        continue;
                    };
                    for (k, &at) in atoms.iter().enumerate() {
                        if at == a {
                            let c = (3 + 3 * k + x) * n;
                            for (o, v) in st.iter_mut().zip(&blk[c..c + n]) {
                                *o += wt * v / (12.0 * h1);
                            }
                        }
                    }
                }
                // (b) independent construction: value blocks only. Mixed second
                // difference with atom a displaced along x and atom b along y,
                // M(h) = [f(++) - f(+-) - f(-+) + f(--)] / (4 h^2), Richardson to O(h^4).
                let mixed = |h: f64, eng: &mut Engine| -> Vec<f64> {
                    let mut m = vec![0.0; n];
                    for (sa, sb_) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                        let sh = add_shifts(
                            atom_shift(atoms, a, x, sa * h),
                            atom_shift(atoms, b, y, sb_ * h),
                        );
                        let v = value(eng, &obs, &sb.prep, p, s1, s2, sh);
                        if v.is_empty() {
                            continue;
                        }
                        for (o, t) in m.iter_mut().zip(&v) {
                            *o += sa * sb_ * t / (4.0 * h * h);
                        }
                    }
                    m
                };
                let m1 = mixed(h2, &mut e_val);
                let m2 = mixed(2.0 * h2, &mut e_val);
                let bb: Vec<f64> = m1
                    .iter()
                    .zip(&m2)
                    .map(|(u, v)| (4.0 * u - v) / 3.0)
                    .collect();
                for (u, v) in st.iter().zip(&bb) {
                    worst = worst.max((u - v).abs());
                    scale = scale.max(u.abs());
                }
                nblocks += 1;
            }
        }
    }
    println!(
        "zeta {zeta:.3e} single_shell={single_shell}: blocks {nblocks}, max |stencil(d1) - richardson(values)| = {worst:.3e}, max |d2| = {scale:.3e}"
    );
    (worst, scale)
}
