//! First nuclear derivatives of the terf / terfc table integrals
//! (`Engine::new_3center_deriv` / `new_2center_deriv` with
//! `Operator::terf{,c}`; the shim's `compute_cart_eri{3,2}_deriv`).
//!
//! # What is checked, and what each check can see
//!
//! The derivative kernel is the McMurchie-Davidson kernel of the values with
//! every Hermite E table replaced by its centre derivative,
//! `dE^{i,j}_t/dA = 2a E^{i+1,j}_t − i E^{i−1,j}_t`, and the R table built one
//! order higher (`crates/ferric-integrals/theory.md`, "Derivative integrals of
//! terf and terfc"). That machinery is the same for every kernel, so:
//!
//! 1. **Coulomb through the same kernel vs libint2** (anchor, no tables):
//!    `scf_debug_md_coulomb_eri{3,2}_deriv` against libint2's own derivative
//!    engine, every shell class of water / cc-pVTZ + cc-pVTZ-RI (obs s..f,
//!    aux s..g). libint2 shares no code with the MD kernel, so a wrong
//!    `2a`/`−i` term, a sign of the ket Hermite index, a block-order slip or
//!    the translational-invariance third centre fails here.
//! 2. **terf and terfc vs central differences of the engine's own values**,
//!    per atom, every shell class of water / cc-pVDZ + cc-pVDZ-RI (obs s..d,
//!    aux s..f, the classes the integral VALUES are validated for against
//!    the Fourier-space reference). This sees the table path: the derivative
//!    reads `G_{m}` one order higher than the value.
//! 3. **terfc = Coulomb − terf** for the derivatives, with the Coulomb piece
//!    from libint2 (a second implementation) and terf from the table engine.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If the derivative is right, (2) agrees with the FD at the FD floor for
//!   every class, the same for terf as for terfc, and the same order as the
//!   Coulomb control run through the same FD (no tables).
//! * If the Hermite derivative is wrong, (1) fails for Coulomb in the p/d/f
//!   classes (ss is blind to the `−i` term: it is zero there).
//! * If the table path is wrong at `m = L + 1` (e.g. the derivative read
//!   `G_m` at the value's depth), (1) passes and (2) fails for terf and terfc
//!   only. That is why both checks exist.
//!
//! # Tolerances (derived)
//!
//! (1) compares two exact evaluations of the same integrals: the libint2
//! Boys function and the shim's `boys_upto` each carry ~1e-14 relative
//! error, accumulated over at most a few hundred primitive terms, against
//! derivative values up to ~1e2 for the tight cc-pVTZ core primitives, so
//! `TOL_MD_VS_LIBINT = 1e-10` absolute is ~1e-12 relative to the largest
//! element; measured below.
//!
//! (2) Five-point central difference with step `h`:
//! `f'(x) = [8(f(h) − f(−h)) − (f(2h) − f(−2h))]/(12h) + h⁴f⁽⁵⁾/30 + O(ε|f|/h)`.
//! At `h = 2e-3` Bohr the truncation term is `h⁴/30·|f⁽⁵⁾| ≈ 5e-13·|f⁽⁵⁾|`;
//! `|f⁽⁵⁾|` of an integral over primitives of exponent ≤ ~1.2e4 (O 1s) is
//! bounded by the Coulomb control, which is FD-only (no tables). The
//! rounding term is `(18/12)·ε·max|f|/h ≈ 1.7e-13·max|f|` with
//! `max|f| ≲ 30`. The table term: the derivative reads the interpolated
//! `G_{L+1}` and the FD differentiates the interpolant of `G_L`; their
//! difference is the derivative of the interpolation error, bounded by the
//! table accuracy (≤ 1e-12 relative, `terf_interp_accuracy.rs`) over the
//! Lagrange node spacing. The bar is ten times the Coulomb control's FD
//! floor, measured in the same run.
//!
//! Measured (2026-11-16, release): (1) worst 3.8e-14 over all 3c and 2c
//! classes up to (g|ff); (2) Coulomb control floor 3.9e-11, terfc 2.6e-11,
//! terf 1.5e-11 (bar 3.9e-10); (3) 2.1e-14. Mutations: dropping the
//! `−i E^{i−1,j}` term fails all three; `terfc = C + terf` fails (2) and (3).
//! The three tests take ~20 s in release.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::operator::Operator;
use std::collections::BTreeMap;
use std::os::raw::c_int;

const WATER: &str = "3\nwater, off C2v\nO 0.02 -0.01 0.117790\nH 0.0 0.785453 -0.451161\nH 0.03 -0.735453 -0.491161\n";

/// r0 = 1.05 Å (the aug-cc-pVDZ default) in Bohr.
const R0_BOHR: f64 = 1.05 / 0.529_177_210_92;

const TOL_MD_VS_LIBINT: f64 = 1e-10;
/// The FD step (Bohr) of the five-point central differences.
const FD_STEP: f64 = 2e-3;

/// Points `FERRIC_TERF_TABLE_DIR` at the repository's `terf-tables/` unless
/// it is set already; a missing table set is a failure, never a skip.
fn tables() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let probe = "16_4_2.bin";
        if let Ok(d) = std::env::var("FERRIC_TERF_TABLE_DIR") {
            assert!(
                std::path::Path::new(&d).join(probe).is_file(),
                "FERRIC_TERF_TABLE_DIR={d} has no {probe}"
            );
            return;
        }
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../terf-tables");
        assert!(
            repo.join(probe).is_file(),
            "terfc tables not found at {}",
            repo.display()
        );
        // Set once, before any engine reads the variable.
        std::env::set_var("FERRIC_TERF_TABLE_DIR", &repo);
    });
}

fn water(basis: &str) -> (Molecule, PreparedBasis) {
    tables();
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let p = PreparedBasis::new(&mol, &basis::bundled(basis).unwrap()).unwrap();
    (mol, p)
}

fn ang(dim: usize) -> usize {
    (dim - 1) / 2
}

/// Full `(naux, nbf, nbf)` value tensor of `op`, flattened.
fn values3(op: Operator, obs: &PreparedBasis, dfbs: &PreparedBasis) -> Vec<f64> {
    let (nbf, naux) = (obs.nbasis(), dfbs.nbasis());
    let mut out = vec![0.0; naux * nbf * nbf];
    let mut eng = Engine::new_3center(op, obs, dfbs, 1e-15).unwrap();
    let (od, oo, dd, dof) = (
        obs.shell_dims(),
        obs.shell_offsets(),
        dfbs.shell_dims(),
        dfbs.shell_offsets(),
    );
    for sp in 0..dfbs.nshells() {
        for s1 in 0..obs.nshells() {
            for s2 in 0..obs.nshells() {
                let Some(v) = eng.compute_eri3(obs, dfbs, sp, s1, s2) else {
                    continue;
                };
                for p in 0..dd[sp] {
                    for i in 0..od[s1] {
                        for j in 0..od[s2] {
                            out[((dof[sp] + p) * nbf + oo[s1] + i) * nbf + oo[s2] + j] =
                                v[(p * od[s1] + i) * od[s2] + j];
                        }
                    }
                }
            }
        }
    }
    out
}

/// Full `(naux, naux)` value matrix of `op`, flattened.
fn values2(op: Operator, dfbs: &PreparedBasis) -> Vec<f64> {
    let naux = dfbs.nbasis();
    let mut out = vec![0.0; naux * naux];
    let mut eng = Engine::new_2center(op, dfbs, 1e-15).unwrap();
    let (dd, dof) = (dfbs.shell_dims(), dfbs.shell_offsets());
    for sp in 0..dfbs.nshells() {
        for sq in 0..dfbs.nshells() {
            let v = eng.compute_eri2(dfbs, sp, sq);
            for p in 0..dd[sp] {
                for q in 0..dd[sq] {
                    out[(dof[sp] + p) * naux + dof[sq] + q] = v[p * dd[sq] + q];
                }
            }
        }
    }
    out
}

/// A 3-centre derivative block source: `Some(9 blocks)` for (sp|s1 s2).
type Deriv3<'a> = dyn FnMut(usize, usize, usize) -> Option<Vec<f64>> + 'a;
type Deriv2<'a> = dyn FnMut(usize, usize) -> Option<Vec<f64>> + 'a;

/// Per-atom analytic derivative tensors `[atom*3 + xyz][(P, μ, ν)]` from the
/// 9-block derivative of every shell triple.
fn analytic3(
    mol: &Molecule,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    f: &mut Deriv3<'_>,
) -> Vec<Vec<f64>> {
    let (nbf, naux) = (obs.nbasis(), dfbs.nbasis());
    let mut out = vec![vec![0.0; naux * nbf * nbf]; 3 * mol.atoms.len()];
    let (od, oo, dd, dof) = (
        obs.shell_dims(),
        obs.shell_offsets(),
        dfbs.shell_dims(),
        dfbs.shell_offsets(),
    );
    let (oa, da) = (obs.shell_to_atom(), dfbs.shell_to_atom());
    for sp in 0..dfbs.nshells() {
        for s1 in 0..obs.nshells() {
            for s2 in 0..obs.nshells() {
                let Some(d) = f(sp, s1, s2) else { continue };
                let blk = dd[sp] * od[s1] * od[s2];
                assert_eq!(d.len(), 9 * blk);
                let atoms = [da[sp], oa[s1], oa[s2]];
                for (c, &atom) in atoms.iter().enumerate() {
                    for x in 0..3 {
                        let src = &d[(3 * c + x) * blk..(3 * c + x + 1) * blk];
                        let dst = &mut out[3 * atom + x];
                        for p in 0..dd[sp] {
                            for i in 0..od[s1] {
                                for j in 0..od[s2] {
                                    dst[((dof[sp] + p) * nbf + oo[s1] + i) * nbf + oo[s2] + j] +=
                                        src[(p * od[s1] + i) * od[s2] + j];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn analytic2(mol: &Molecule, dfbs: &PreparedBasis, f: &mut Deriv2<'_>) -> Vec<Vec<f64>> {
    let naux = dfbs.nbasis();
    let mut out = vec![vec![0.0; naux * naux]; 3 * mol.atoms.len()];
    let (dd, dof, da) = (
        dfbs.shell_dims(),
        dfbs.shell_offsets(),
        dfbs.shell_to_atom(),
    );
    for sp in 0..dfbs.nshells() {
        for sq in 0..dfbs.nshells() {
            let Some(d) = f(sp, sq) else { continue };
            let blk = dd[sp] * dd[sq];
            assert_eq!(d.len(), 6 * blk);
            for (c, &atom) in [da[sp], da[sq]].iter().enumerate() {
                for x in 0..3 {
                    let src = &d[(3 * c + x) * blk..(3 * c + x + 1) * blk];
                    for p in 0..dd[sp] {
                        for q in 0..dd[sq] {
                            out[3 * atom + x][(dof[sp] + p) * naux + dof[sq] + q] +=
                                src[p * dd[sq] + q];
                        }
                    }
                }
            }
        }
    }
    out
}

fn displaced(mol: &Molecule, atom: usize, x: usize, d: f64) -> Molecule {
    let mut m = mol.clone();
    match x {
        0 => m.atoms[atom].x += d,
        1 => m.atoms[atom].y += d,
        _ => m.atoms[atom].zpos += d,
    }
    m
}

/// Five-point central difference of `value(mol)` along every atom and axis.
fn fd(mol: &Molecule, h: f64, value: &dyn Fn(&Molecule) -> Vec<f64>) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for atom in 0..mol.atoms.len() {
        for x in 0..3 {
            let f = |k: f64| value(&displaced(mol, atom, x, k * h));
            let (p1, m1, p2, m2) = (f(1.0), f(-1.0), f(2.0), f(-2.0));
            out.push(
                (0..p1.len())
                    .map(|i| (8.0 * (p1[i] - m1[i]) - (p2[i] - m2[i])) / (12.0 * h))
                    .collect(),
            );
        }
    }
    out
}

/// Max |a − b| per 3-index shell class `(l_P, l_μ, l_ν)` over all atoms/axes.
fn by_class3(
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    a: &[Vec<f64>],
    b: &[Vec<f64>],
) -> BTreeMap<(usize, usize, usize), f64> {
    let nbf = obs.nbasis();
    let lo: Vec<usize> = (0..obs.nshells())
        .flat_map(|s| vec![ang(obs.shell_dims()[s]); obs.shell_dims()[s]])
        .collect();
    let la: Vec<usize> = (0..dfbs.nshells())
        .flat_map(|s| vec![ang(dfbs.shell_dims()[s]); dfbs.shell_dims()[s]])
        .collect();
    let mut m = BTreeMap::new();
    for (va, vb) in a.iter().zip(b) {
        for (k, (x, y)) in va.iter().zip(vb).enumerate() {
            let (p, mu, nu) = (k / (nbf * nbf), (k / nbf) % nbf, k % nbf);
            let key = (la[p], lo[mu].max(lo[nu]), lo[mu].min(lo[nu]));
            let e = m.entry(key).or_insert(0.0f64);
            *e = e.max((x - y).abs());
        }
    }
    m
}

fn by_class2(
    dfbs: &PreparedBasis,
    a: &[Vec<f64>],
    b: &[Vec<f64>],
) -> BTreeMap<(usize, usize), f64> {
    let naux = dfbs.nbasis();
    let la: Vec<usize> = (0..dfbs.nshells())
        .flat_map(|s| vec![ang(dfbs.shell_dims()[s]); dfbs.shell_dims()[s]])
        .collect();
    let mut m = BTreeMap::new();
    for (va, vb) in a.iter().zip(b) {
        for (k, (x, y)) in va.iter().zip(vb).enumerate() {
            let key = (
                la[k / naux].max(la[k % naux]),
                la[k / naux].min(la[k % naux]),
            );
            let e = m.entry(key).or_insert(0.0f64);
            *e = e.max((x - y).abs());
        }
    }
    m
}

fn worst<K>(m: &BTreeMap<K, f64>) -> f64 {
    m.values().fold(0.0, |a, &b| a.max(b))
}

fn max_abs(v: &[Vec<f64>]) -> f64 {
    v.iter().flatten().fold(0.0, |a, &b| a.max(b.abs()))
}

fn table_d3<'a>(
    op: Operator,
    obs: &'a PreparedBasis,
    dfbs: &'a PreparedBasis,
) -> impl FnMut(usize, usize, usize) -> Option<Vec<f64>> + 'a {
    let mut e = Engine::new_3center_deriv(op, obs, dfbs, 1e-15).unwrap();
    move |sp, s1, s2| {
        e.compute_eri3_deriv(obs, dfbs, sp, s1, s2)
            .map(<[f64]>::to_vec)
    }
}

fn table_d2(
    op: Operator,
    dfbs: &PreparedBasis,
) -> impl FnMut(usize, usize) -> Option<Vec<f64>> + '_ {
    let mut e = Engine::new_2center_deriv(op, dfbs, 1e-15).unwrap();
    move |sp, sq| e.compute_eri2_deriv(dfbs, sp, sq).map(<[f64]>::to_vec)
}

fn md_coulomb_d3<'a>(
    obs: &'a PreparedBasis,
    dfbs: &'a PreparedBasis,
) -> impl FnMut(usize, usize, usize) -> Option<Vec<f64>> + 'a {
    move |sp, s1, s2| {
        let n = 9 * dfbs.shell_dims()[sp] * obs.shell_dims()[s1] * obs.shell_dims()[s2];
        let mut out = vec![0.0; n];
        // SAFETY: live basis handles, in-range shells, `out` holds `n` doubles.
        let w = unsafe {
            ffi::scf_debug_md_coulomb_eri3_deriv(
                obs.handle(),
                dfbs.handle(),
                sp as c_int,
                s1 as c_int,
                s2 as c_int,
                out.as_mut_ptr(),
                n as c_int,
            )
        };
        assert_eq!(w as usize, n, "status {w}");
        Some(out)
    }
}

fn md_coulomb_d2(dfbs: &PreparedBasis) -> impl FnMut(usize, usize) -> Option<Vec<f64>> + '_ {
    move |sp, sq| {
        let n = 6 * dfbs.shell_dims()[sp] * dfbs.shell_dims()[sq];
        let mut out = vec![0.0; n];
        // SAFETY: live basis handle, in-range shells, `out` holds `n` doubles.
        let w = unsafe {
            ffi::scf_debug_md_coulomb_eri2_deriv(
                dfbs.handle(),
                sp as c_int,
                sq as c_int,
                out.as_mut_ptr(),
                n as c_int,
            )
        };
        assert_eq!(w as usize, n, "status {w}");
        Some(out)
    }
}

/// ANCHOR (1): the derivative kernel with the Coulomb kernel equals
/// libint2's derivative engine, every class of water / cc-pVTZ + cc-pVTZ-RI.
#[test]
fn md_coulomb_derivative_kernel_matches_libint2_every_class() {
    let (mol, obs) = water("cc-pvtz");
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvtz-ri").unwrap()).unwrap();
    assert_eq!(obs.max_l(), 3, "obs carries f");
    assert_eq!(dfbs.max_l(), 4, "aux carries g");
    let coul = Operator::coulomb();
    let mut li = Engine::new_3center_deriv(coul, &obs, &dfbs, 1e-15).unwrap();
    let a = analytic3(&mol, &obs, &dfbs, &mut |sp, s1, s2| {
        li.compute_eri3_deriv(&obs, &dfbs, sp, s1, s2)
            .map(<[f64]>::to_vec)
    });
    let b = analytic3(&mol, &obs, &dfbs, &mut md_coulomb_d3(&obs, &dfbs));
    let m3 = by_class3(&obs, &dfbs, &a, &b);
    eprintln!("3c MD-vs-libint2 (max |d| = {:.3e}): {m3:?}", max_abs(&a));
    let mut li2 = Engine::new_2center_deriv(coul, &dfbs, 1e-15).unwrap();
    let a2 = analytic2(&mol, &dfbs, &mut |sp, sq| {
        li2.compute_eri2_deriv(&dfbs, sp, sq).map(<[f64]>::to_vec)
    });
    let b2 = analytic2(&mol, &dfbs, &mut md_coulomb_d2(&dfbs));
    let m2 = by_class2(&dfbs, &a2, &b2);
    eprintln!("2c MD-vs-libint2 (max |d| = {:.3e}): {m2:?}", max_abs(&a2));
    for key in [(0, 0, 0), (4, 3, 3), (4, 3, 0), (0, 3, 3)] {
        assert!(m3.contains_key(&key), "class {key:?} not exercised");
    }
    assert!(m2.contains_key(&(4, 4)), "2c class (g|g) not exercised");
    assert!(worst(&m3) < TOL_MD_VS_LIBINT, "3c: {m3:?}");
    assert!(worst(&m2) < TOL_MD_VS_LIBINT, "2c: {m2:?}");
}

/// The FD of `op`'s values against `op`'s analytic derivative, water /
/// cc-pVDZ + cc-pVDZ-RI. Returns (3c by class, 2c by class, max |deriv|).
#[allow(clippy::type_complexity)]
fn fd_check(
    op: Operator,
) -> (
    BTreeMap<(usize, usize, usize), f64>,
    BTreeMap<(usize, usize), f64>,
    f64,
) {
    let (mol, obs) = water("cc-pvdz");
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let a3 = if op.kind == ferric_integrals::operator::OperatorKind::Coulomb {
        analytic3(&mol, &obs, &dfbs, &mut md_coulomb_d3(&obs, &dfbs))
    } else {
        analytic3(&mol, &obs, &dfbs, &mut table_d3(op, &obs, &dfbs))
    };
    let a2 = if op.kind == ferric_integrals::operator::OperatorKind::Coulomb {
        analytic2(&mol, &dfbs, &mut md_coulomb_d2(&dfbs))
    } else {
        analytic2(&mol, &dfbs, &mut table_d2(op, &dfbs))
    };
    let f3 = fd(&mol, FD_STEP, &|m| {
        let o = PreparedBasis::new(m, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let d = PreparedBasis::new(m, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        values3(op, &o, &d)
    });
    let f2 = fd(&mol, FD_STEP, &|m| {
        let d = PreparedBasis::new(m, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        values2(op, &d)
    });
    (
        by_class3(&obs, &dfbs, &a3, &f3),
        by_class2(&dfbs, &a2, &f2),
        max_abs(&a3).max(max_abs(&a2)),
    )
}

/// CHECK (2): terf and terfc derivatives against five-point differences of
/// the engine's own values, every class of water / cc-pVDZ + cc-pVDZ-RI, at
/// the linked sharpness (`w r0 = 1/sqrt 2`) and a decoupled one (`w r0 = 4`).
/// The bar is ten times the Coulomb control's FD floor, measured in the same
/// run through the same FD on the same geometry (no tables in the control).
#[test]
fn terf_table_derivatives_match_finite_differences() {
    let (c3, c2, cm) = fd_check(Operator::coulomb());
    let floor = worst(&c3).max(worst(&c2));
    eprintln!(
        "Coulomb control (MD, no tables) max|d|={cm:.3e} 3c {:.3e} 2c {:.3e}",
        worst(&c3),
        worst(&c2)
    );
    assert!(
        floor > 0.0 && floor < 1e-9,
        "Coulomb FD floor {floor:.3e} out of the derived range"
    );
    let bar = 10.0 * floor;
    for op in [
        Operator::terfc(R0_BOHR),
        Operator::terf(R0_BOHR),
        Operator::terfc_with_omega(R0_BOHR, 4.0 / R0_BOHR),
    ] {
        let (m3, m2, mx) = fd_check(op);
        eprintln!(
            "{:?} r0={:.3} w={:.3}: max|d|={mx:.3e} 3c {:.3e} 2c {:.3e} (bar {bar:.1e})\n  3c {m3:?}\n  2c {m2:?}",
            op.kind,
            op.distance,
            op.omega,
            worst(&m3),
            worst(&m2)
        );
        assert!(
            mx > 0.1,
            "{:?}: derivatives vanish, the check is vacuous",
            op.kind
        );
        for key in [(3, 2, 2), (3, 0, 0), (0, 2, 2)] {
            assert!(m3.contains_key(&key), "class {key:?} not exercised");
        }
        assert!(worst(&m3) < bar, "{:?} 3c: {m3:?}", op.kind);
        assert!(worst(&m2) < bar, "{:?} 2c: {m2:?}", op.kind);
    }
}

/// CHECK (3): the terfc derivative is the Coulomb derivative (libint2, a
/// second implementation) minus the terf derivative (table engine), every
/// class of water / cc-pVDZ + cc-pVDZ-RI. Same tolerance as (1): two exact
/// evaluations of the same Coulomb piece.
#[test]
fn terfc_derivative_is_coulomb_minus_terf() {
    let (mol, obs) = water("cc-pvdz");
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let coul = Operator::coulomb();
    let mut li = Engine::new_3center_deriv(coul, &obs, &dfbs, 1e-15).unwrap();
    let c3 = analytic3(&mol, &obs, &dfbs, &mut |sp, s1, s2| {
        li.compute_eri3_deriv(&obs, &dfbs, sp, s1, s2)
            .map(<[f64]>::to_vec)
    });
    let mut li2 = Engine::new_2center_deriv(coul, &dfbs, 1e-15).unwrap();
    let c2 = analytic2(&mol, &dfbs, &mut |sp, sq| {
        li2.compute_eri2_deriv(&dfbs, sp, sq).map(<[f64]>::to_vec)
    });
    for omega in [1.0 / (R0_BOHR * std::f64::consts::SQRT_2), 4.0 / R0_BOHR] {
        let terf = Operator::terf_with_omega(R0_BOHR, omega);
        let terfc = Operator::terfc_with_omega(R0_BOHR, omega);
        let t3 = analytic3(&mol, &obs, &dfbs, &mut table_d3(terf, &obs, &dfbs));
        let tc3 = analytic3(&mol, &obs, &dfbs, &mut table_d3(terfc, &obs, &dfbs));
        let t2 = analytic2(&mol, &dfbs, &mut table_d2(terf, &dfbs));
        let tc2 = analytic2(&mol, &dfbs, &mut table_d2(terfc, &dfbs));
        let diff = |c: &[Vec<f64>], t: &[Vec<f64>]| -> Vec<Vec<f64>> {
            c.iter()
                .zip(t)
                .map(|(cr, tr)| cr.iter().zip(tr).map(|(x, y)| x - y).collect())
                .collect()
        };
        let m3 = by_class3(&obs, &dfbs, &diff(&c3, &t3), &tc3);
        let m2 = by_class2(&dfbs, &diff(&c2, &t2), &tc2);
        eprintln!(
            "w={omega:.3}: |C - terf - terfc| 3c {:.3e} 2c {:.3e}; max|terf'| {:.3e}",
            worst(&m3),
            worst(&m2),
            max_abs(&t3)
        );
        assert!(
            max_abs(&t3) > 0.1 && max_abs(&t2) > 0.1,
            "terf derivatives vanish, the check is vacuous"
        );
        assert!(worst(&m3) < TOL_MD_VS_LIBINT, "3c: {m3:?}");
        assert!(worst(&m2) < TOL_MD_VS_LIBINT, "2c: {m2:?}");
    }
}
