// LANL2DZ exponents such as 0.318 are basis data, not approximations of 1/pi.
#![allow(clippy::approx_constant)]
//! Exactness anchors for the periodic-ECP derivative kernel
//! (`ferric_ecp_block_deriv` / [`ecp_block_deriv_spherical`]), checked in
//! isolation before any lattice sum or SCF (FINDINGS "Iteration 22", "For the
//! Rust port").
//!
//! * (a) Trivial limit: zero shift, bra = ket = the molecule's shells, one
//!   group per atom — the per-atom fold (bra rows + ket columns + centre
//!   group) IS the molecular `ecp_potential_deriv` (libecpint's
//!   `compute_first_derivs`), element-wise.
//! * (b) Per-TRIPLE slot checks (bra, ket, centre) vs Richardson FD
//!   (h = 1e-3) of the VALUE of that triple, ASSERTED only where both shells
//!   are >= 1 Bohr from the ECP centre, at libecpint's value accuracy (bar
//!   1e-5 relative, measured 3.6e-6); nearer bands are printed only.
//!   libecpint's VALUES are ~1e-6 off an independent quadrature and jitter
//!   with geometry (d projector), so an FD of them cannot referee the
//!   derivative more tightly — see the test doc and FINDINGS "ECP derivative
//!   clean-band discrepancy — 2026-09-27".
//! * (c) ON-CENTRE shells (a shell on its own atom's ECP, the libecpint
//!   `compute_shell_pair_derivative` quirk the shim sidesteps) have no FD
//!   referee (an FD through the centre crosses libecpint's non-smooth
//!   region; one that keeps the shell on the centre sees only −∂_ket).
//!   Their split cancels inside that atom's force, so they are covered by
//!   (a) (per-atom fold), (e) (bra + ket + centre = 0) and the SCF-FD
//!   anchor in `ferric-pbc/tests/pbc_grad_ecp.rs`. The former continuity-
//!   extrapolation test was DROPPED (its off-centre points at 0.05..0.2
//!   Bohr are themselves in the non-smooth band: 1.3e-1 vs FD at 0.2).
//! * (d) The off-centre element of FINDINGS Iteration 22 (2b):
//!   `∂/∂B_z ⟨H1s|U_I|H1s_L⟩`, bra H STO-3G at (0.3, 0.2, 0.4), ket the
//!   same H1s at L = (0, 0, 7), ONE ECP centre (I LANL2DZ, M = 0) at
//!   (0.9, −0.4, 3.3) — identified as the M = 0 configuration by
//!   reproducing its "FD of PySCF values" number −6.658296248e-3
//!   (PySCF 2.13, h = 1e-4). Primary: analytic vs FD of ferric's own values
//!   (measured 7.4e-10). Secondary: value accuracy vs the quadrature target
//!   −6.658296236e-3 (measured 1.8e-8, libecpint's radial quadrature), bar
//!   5e-8, which still REJECTS PySCF's `ECPscalar_ipnuc` (−6.658184541e-3,
//!   1.1e-7 off). Do NOT pin to PySCF ipnuc.
//! * (e) Translation invariance per element: bra + ket + Σ_g centre = 0.
//!   By construction (centre := −(bra + ket) in the shim), so it guards the
//!   accumulation/grouping bookkeeping, not the physics — (b)'s centre FD is
//!   the physics check of the centre slot.
//! * (f) Argument checks (bad group, ragged mask) error, never write.
//!
//! FD bars are RELATIVE to a block's max and are set by libecpint's VALUE
//! accuracy, not by the derivative (see (b)); `ecp_matrix_deriv.rs` saw the
//! same effect molecularly (an h-independent 1.4e-7 / 2.2e-7 relative
//! plateau on I2/def2-SVP).
//!
//! Every libecpint check above runs on `EcpBackend::Libecpint` explicitly
//! with its bars unchanged. The ferric-owned quadrature backend
//! (`EcpBackend::Quadrature`, FINDINGS "Iteration 25") has smooth, exact
//! on-centre values, so its versions of (b), (d), (e) and (f) are TIGHTER:
//! (b) asserts EVERY band including on-centre (relative 1e-8 on blocks with
//! max |dV| > 1e-6, absolute 1e-11 below; prototype 9.7e-10), (d) asserts
//! the quadrature target to 5e-12 (prototype 4.4e-13). (a) runs on the
//! process backend (`FERRIC_ECP_BACKEND`), the one `ecp_potential_deriv`
//! uses, so it checks "fold ≡ molecular derivative" for whichever is active.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::ecp::{
    ecp_backend, ecp_block_deriv_spherical_with_backend, ecp_block_spherical_with_backend,
    gto_norm, EcpBackend, EcpBlockDeriv, EcpCenter, EcpGaussianShell,
};
use ferric_integrals::oneelectron::ecp_potential_deriv;

const H0: [f64; 3] = [0.3, 0.2, 0.4];
const I0: [f64; 3] = [0.9, -0.4, 3.3];

// ------------------------------------------------------------ fixtures

/// Contracted shell over unit-normalised primitives scaled to unit
/// self-overlap, then `gto_norm` folded (the bare-Cartesian convention the
/// kernel takes; identical to `oneelectron::build_ecp_inputs`).
fn gshell(l: i32, center: [f64; 3], exps: &[f64], coefs: &[f64]) -> EcpGaussianShell {
    let lf = l as f64;
    let mut s = 0.0;
    for (a, ca) in exps.iter().zip(coefs) {
        for (b, cb) in exps.iter().zip(coefs) {
            s += ca * cb * (2.0 * (a * b).sqrt() / (a + b)).powf(lf + 1.5);
        }
    }
    EcpGaussianShell {
        l,
        center,
        exponents: exps.to_vec(),
        coefficients: exps
            .iter()
            .zip(coefs)
            .map(|(&a, &c)| c / s.sqrt() * gto_norm(l, a))
            .collect(),
    }
}

fn h_sto3g(c: [f64; 3]) -> Vec<EcpGaussianShell> {
    vec![gshell(
        0,
        c,
        &[3.42525091, 0.62391373, 0.1688554],
        &[0.15432897, 0.53532814, 0.44463454],
    )]
}

fn i_lanl2dz(c: [f64; 3]) -> Vec<EcpGaussianShell> {
    vec![
        gshell(0, c, &[0.7242, 0.4653], &[-2.9731048, 3.4827643]),
        gshell(0, c, &[0.1336], &[1.0]),
        gshell(1, c, &[1.29, 0.318], &[-0.2092377, 1.1035347]),
        gshell(1, c, &[0.1053], &[1.0]),
    ]
}

/// LANL2DZ ECP for I (PySCF / libecpint digits; `(n, ζ, d)`, r^{n−2}), the
/// same table as `ferric-pbc/tests/pbc_ecp.rs`.
fn i_lanl2dz_ecp(c: [f64; 3]) -> EcpCenter {
    let ch: [(i32, &[(i32, f64, f64)]); 4] = [
        (
            3,
            &[
                (0, 1.0715702, -0.0747621),
                (1, 44.1936028, -30.0811224),
                (2, 12.9367609, -75.3722721),
                (2, 3.1956412, -22.0563758),
                (2, 0.8589806, -1.6979585),
            ],
        ),
        (
            0,
            &[
                (0, 127.9202670, 2.9380036),
                (1, 78.6211465, 41.2471267),
                (2, 36.5146237, 287.8680095),
                (2, 9.9065681, 114.3758506),
                (2, 1.9420086, 37.6547714),
            ],
        ),
        (
            1,
            &[
                (0, 13.0035304, 2.2222630),
                (1, 76.0331404, 39.4090831),
                (2, 24.1961684, 177.4075002),
                (2, 6.4053433, 77.9889462),
                (2, 1.5851786, 25.7547641),
            ],
        ),
        (
            2,
            &[
                (0, 40.4278108, 7.0524360),
                (1, 28.9084375, 33.3041635),
                (2, 15.6268936, 186.9453875),
                (2, 4.1442856, 71.9688361),
                (2, 0.9377235, 9.3630657),
            ],
        ),
    ];
    let mut e = EcpCenter {
        center: c,
        ams: vec![],
        ns: vec![],
        exponents: vec![],
        coefficients: vec![],
    };
    for (l, terms) in ch {
        for &(n, z, d) in terms {
            e.ams.push(l);
            e.ns.push(n);
            e.exponents.push(z);
            e.coefficients.push(d);
        }
    }
    e
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn nsph(sh: &[EcpGaussianShell]) -> Vec<usize> {
    sh.iter().map(|s| (2 * s.l + 1) as usize).collect()
}

/// Spherical function index ranges of the shells in `sel`.
fn fn_mask(sh: &[EcpGaussianShell], sel: &[usize]) -> Vec<bool> {
    let mut out = Vec::new();
    for (s, n) in nsph(sh).into_iter().enumerate() {
        out.extend(std::iter::repeat_n(sel.contains(&s), n));
    }
    out
}

fn moved(sh: &[EcpGaussianShell], sel: &[usize], x: usize, h: f64) -> Vec<EcpGaussianShell> {
    sh.iter()
        .enumerate()
        .map(|(s, g)| {
            let mut g = g.clone();
            if sel.contains(&s) {
                g.center[x] += h;
            }
            g
        })
        .collect()
}

fn amax(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// Richardson-extrapolated central FD `(4 FD(h/2) − FD(h))/3` of a value
/// block as a function of one displacement.
fn richardson(f: &dyn Fn(f64) -> Vec<f64>, h: f64) -> Vec<f64> {
    let fd = |hh: f64| -> Vec<f64> {
        let p = f(hh);
        let m = f(-hh);
        p.iter()
            .zip(&m)
            .map(|(a, b)| (a - b) / (2.0 * hh))
            .collect()
    };
    let f1 = fd(h);
    let f2 = fd(0.5 * h);
    f1.iter()
        .zip(&f2)
        .map(|(a, b)| (4.0 * b - a) / 3.0)
        .collect()
}

/// `(max |analytic − FD|, max |analytic|)`.
fn cmp(an: &[f64], fd: &[f64]) -> (f64, f64) {
    assert_eq!(an.len(), fd.len());
    let d = an
        .iter()
        .zip(fd)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    (d, amax(an))
}

// ============================================================ (a) trivial limit

/// Zero shift: the per-atom fold equals libecpint's molecular
/// `compute_first_derivs` (via `ecp_potential_deriv`), element-wise. HI with
/// def2-SVP (d shells on I; its shells sit ON the ECP centre, so the
/// on-centre handling is inside this fold too — only its per-atom TOTAL is
/// checked here).
#[test]
fn zero_shift_block_deriv_folds_to_the_molecular_matrix_deriv() {
    let bs = basis::bundled("def2-svp").unwrap();
    let mut mol = Molecule::parse_xyz("2\n\nH 0.1 0.2 0.0\nI -0.1 0.05 1.61\n", 0, 1).unwrap();
    mol.apply_ecp(&bs);
    let reference = ecp_potential_deriv(&mol, &bs)
        .expect("ecp_potential_deriv")
        .expect("def2-SVP carries an ECP for I");

    let mut shells = Vec::new();
    let mut shell_atom = Vec::new();
    let mut centres = Vec::new();
    let mut groups = Vec::new();
    for (ia, a) in mol.atoms.iter().enumerate() {
        let c = [a.x, a.y, a.zpos];
        for sh in bs.for_element(a.z).unwrap() {
            shells.push(EcpGaussianShell {
                l: sh.l,
                center: c,
                exponents: sh.exponents.clone(),
                coefficients: sh
                    .exponents
                    .iter()
                    .zip(&sh.coefficients)
                    .map(|(&x, &cc)| cc * gto_norm(sh.l, x))
                    .collect(),
            });
            shell_atom.push(ia);
        }
        if let Some(def) = bs.ecp_for_element(a.z) {
            let mut e = EcpCenter {
                center: c,
                ams: vec![],
                ns: vec![],
                exponents: vec![],
                coefficients: vec![],
            };
            for ch in &def.shells {
                for t in &ch.terms {
                    e.ams.push(ch.angular_momentum);
                    e.ns.push(t.r_exp);
                    e.exponents.push(t.gexp);
                    e.coefficients.push(t.coef);
                }
            }
            centres.push(e);
            groups.push(ia);
        }
    }
    let natoms = mol.atoms.len();
    // The backend `ecp_potential_deriv` used (FERRIC_ECP_BACKEND).
    let backend = ecp_backend().unwrap();
    let d = ecp_block_deriv_spherical_with_backend(
        backend, &shells, &shells, &centres, None, &groups, natoms,
    )
    .unwrap();
    let n = d.nrow;
    assert_eq!(d.ncol, n);
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for (ia, per_atom) in reference.iter().enumerate() {
        let sel: Vec<usize> = (0..shells.len()).filter(|&s| shell_atom[s] == ia).collect();
        let on = fn_mask(&shells, &sel);
        for x in 0..3 {
            let fold: Vec<f64> = (0..n * n)
                .map(|k| {
                    let (i, j) = (k / n, k % n);
                    let mut v = d.centre[ia][x][k];
                    if on[i] {
                        v += d.bra[x][k];
                    }
                    if on[j] {
                        v += d.ket[x][k];
                    }
                    v
                })
                .collect();
            let r = per_atom[x].as_standard_layout();
            let r = r.as_slice().unwrap();
            let (dd, s) = cmp(&fold, r);
            worst = worst.max(dd);
            scale = scale.max(s);
        }
    }
    eprintln!(
        "[{backend}] zero shift fold vs molecular dV_ECP/dR: max |Δ| {worst:.2e} (max |dV| {scale:.3})"
    );
    assert!(scale > 0.1, "vacuous: dV_ECP ~ 0");
    assert!(worst < 1e-10 * scale.max(1.0), "{worst:e}");
}

// ============================================================ (b) shifted vs FD

/// Below this distance (Bohr) from the ECP centre a triple is in the
/// nearest (reported-only) band.
const FD_MIN_DIST: f64 = 0.5;
/// The ASSERTED band: both shells at least this far from the centre.
const CLEAN_DIST: f64 = 1.0;
/// FD step of the per-triple referee. 1e-3, not 1e-4: libecpint's VALUES
/// jitter with geometry (see the test doc), and the FD amplifies that by
/// 1/h; Richardson removes the h² truncation at this step.
const TRIPLE_FD_H: f64 = 1e-3;
/// Bar (relative to the band's largest |dV|) for the asserted band. It is
/// libecpint's own VALUE accuracy, not the derivative's: measured
/// 2026-09-27 the clean band sits at 3.6e-6 relative at h = 1e-3 (the same
/// with the d projector removed, so h-independent: a systematic value error,
/// not FD noise). See FINDINGS "ECP derivative clean-band discrepancy".
const TRIPLE_FD_REL_BAR: f64 = 1e-5;

/// `(d_min, |analytic − FD|, max |analytic|, label)` of one slot check.
type Row = (f64, f64, f64, String);

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
}

/// Per-triple slot rows `(d_min, |an − FD|, max |an|, label)` of the HI
/// fixture below, for one backend.
fn per_triple_rows(backend: EcpBackend) -> Vec<Row> {
    let l_vec = [0.8, -0.6, 1.1];
    let mut bra = h_sto3g(H0);
    bra.extend(i_lanl2dz(I0));
    let bra_name = ["H s", "I s(2)", "I s(1)", "I p(2)", "I p(1)"];
    let ket: Vec<EcpGaussianShell> = bra
        .iter()
        .map(|s| EcpGaussianShell {
            center: add(s.center, l_vec),
            ..s.clone()
        })
        .collect();
    let cs = [
        I0,
        add(add(I0, l_vec), [0.3, 0.2, -0.25]),
        add(H0, [-0.5, 0.4, 0.6]),
        add(I0, [1.5, -1.2, 0.8]),
    ];
    let ecps: Vec<EcpCenter> = cs.iter().map(|c| i_lanl2dz_ecp(*c)).collect();
    let h = TRIPLE_FD_H;
    let mut rows: Vec<Row> = Vec::new();
    for (ia, sa) in bra.iter().enumerate() {
        for (ib, sb) in ket.iter().enumerate() {
            for (iu, eu) in ecps.iter().enumerate() {
                let (ba, kb, uc) = (
                    std::slice::from_ref(sa),
                    std::slice::from_ref(sb),
                    std::slice::from_ref(eu),
                );
                let an = ecp_block_deriv_spherical_with_backend(backend, ba, kb, uc, None, &[0], 1)
                    .unwrap();
                let (dac, dbc) = (dist(sa.center, eu.center), dist(sb.center, eu.center));
                let dmin = dac.min(dbc);
                for x in 0..3 {
                    let fb = |d: f64| {
                        ecp_block_spherical_with_backend(
                            backend,
                            &moved(ba, &[0], x, d),
                            kb,
                            uc,
                            None,
                        )
                        .unwrap()
                    };
                    let fk = |d: f64| {
                        ecp_block_spherical_with_backend(
                            backend,
                            ba,
                            &moved(kb, &[0], x, d),
                            uc,
                            None,
                        )
                        .unwrap()
                    };
                    let fc = |d: f64| {
                        let mut e = eu.clone();
                        e.center[x] += d;
                        ecp_block_spherical_with_backend(backend, ba, kb, &[e], None).unwrap()
                    };
                    for (slot, f, a) in [
                        ("bra", &fb as &dyn Fn(f64) -> Vec<f64>, &an.bra[x]),
                        ("ket", &fk, &an.ket[x]),
                        ("centre", &fc, &an.centre[0][x]),
                    ] {
                        let (dd, sc) = cmp(a, &richardson(f, h));
                        rows.push((
                            dmin,
                            dd,
                            sc,
                            format!(
                                "{slot:6} x{x} bra {} ket {} U{iu} (|A−C| {dac:.2}, |B−C| {dbc:.2})",
                                bra_name[ia], bra_name[ib]
                            ),
                        ));
                    }
                }
            }
        }
    }
    rows
}

/// Per-TRIPLE slot checks: for every (bra shell a, ket shell b, ECP centre
/// u) the kernel is called on that triple ALONE, and each of its three slots
/// (bra, ket, centre) is compared with Richardson FD (h = 1e-3) of the value
/// of that one triple, moving only the centre in question. HI/LANL2DZ, ket =
/// bra shifted by a short vector, four ECP centres: one ON the bra I shells,
/// one ~0.4 Bohr from the ket I shells, two further out, so every distance
/// band is populated. Bands by `d = min(|A − C|, |B − C|)`.
///
/// WHAT THE FD REFEREE CAN AND CANNOT SEE (FINDINGS "ECP derivative
/// clean-band discrepancy — 2026-09-27", measured with the built shim via
/// ctypes against PySCF values and an independent radial × Lebedev
/// quadrature, which agree with each other to 1e-12..1e-15):
/// * libecpint's VALUE integrals carry a ~1e-7..1e-6 absolute error
///   (2.2e-7..7.0e-7 on the worst elements here, 2.3e-6 on an s-projector
///   element of magnitude 0.99). Part of it is systematic (s/p projectors,
///   h-independent), part is NON-SMOOTH in the geometry (the d projector's
///   type-2 radial integrals: cubic-fit residual 6.9e-7 along a 1e-3 Bohr
///   scan, PySCF 1.7e-16). The engine knobs (thresh 1e-12..1e-17, radial
///   grids 256/1024..1024/4096) do not change it.
/// * The FD of those values amplifies the non-smooth part by 1/h: at
///   h = 1e-4 the 0.5..1 Bohr band missed by 1.25e-2 and the >= 1 Bohr band
///   by 1.89e-5, while the ANALYTIC derivative matched the quadrature FD
///   to 2e-7..1e-6 on the same elements (its error is libecpint's value
///   accuracy, not amplified).
///
/// So this test can only assert analytic ≈ FD at libecpint's value accuracy,
/// in the >= 1 Bohr band, at h = 1e-3 (bar 1e-5 relative, measured 3.6e-6).
/// The 0.5..1 Bohr and < 0.5 Bohr bands are REPORTED ONLY (d-projector
/// jitter: 5.4e-4 relative at 0.5..1 Bohr even at h = 1e-3). The derivative
/// itself is refereed by the quadrature numbers in FINDINGS, by (a) (per-atom
/// fold ≡ libecpint's molecular derivative) and by the SCF-FD anchor of
/// `ferric-pbc/tests/pbc_grad_ecp.rs`. On-centre shells: module doc (c).
#[test]
fn shifted_block_deriv_matches_richardson_fd_per_triple_away_from_centres() {
    let h = TRIPLE_FD_H;
    let rows = per_triple_rows(EcpBackend::Libecpint);
    let band =
        |lo: f64, hi: f64| -> Vec<&Row> { rows.iter().filter(|r| r.0 >= lo && r.0 < hi).collect() };
    let summarize = |name: &str, rs: &[&Row]| -> (f64, f64) {
        let scale = rs.iter().fold(0.0_f64, |m, r| m.max(r.2));
        let worst = rs.iter().fold(0.0_f64, |m, r| m.max(r.1));
        eprintln!(
            "{name}: {} checks, worst |an − FD| {worst:.2e}, scale {scale:.3e} (relative {:.2e})",
            rs.len(),
            worst / scale.max(1e-300)
        );
        let mut top: Vec<&&Row> = rs.iter().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        for r in top.iter().take(6) {
            eprintln!("    {:.2e} (max {:.2e})  {}", r.1, r.2, r.3);
        }
        (worst, scale)
    };
    let near = band(0.0, FD_MIN_DIST);
    let mid = band(FD_MIN_DIST, CLEAN_DIST);
    let clean = band(CLEAN_DIST, f64::INFINITY);
    eprintln!("per-triple slot FD (Richardson h = {h:.0e}), by d = min(|A−C|, |B−C|):");
    summarize("  d < 0.5 (REPORTED ONLY: FD through/near a centre)", &near);
    summarize(
        "  0.5 <= d < 1.0 (REPORTED ONLY: d-projector value jitter)",
        &mid,
    );
    let (wc, sc) = summarize("  ASSERTED d >= 1.0", &clean);
    // Non-vacuity: every band is populated and the asserted one carries
    // real slopes.
    assert!(
        near.len() >= 9 && mid.len() >= 9 && clean.len() >= 30,
        "fixture bands too thin: near {} mid {} clean {}",
        near.len(),
        mid.len(),
        clean.len()
    );
    assert!(sc > 1e-2, "vacuous: max |dV| over the asserted band {sc:e}");
    assert!(
        wc < TRIPLE_FD_REL_BAR * sc,
        "{wc:e} (scale {sc:e}; relative {:.2e})",
        wc / sc
    );
}

/// (b) for the ferric-owned quadrature backend: the SAME fixture and FD
/// referee, asserted in EVERY band including the on-centre and < 0.5 Bohr
/// ones libecpint cannot pass (its values are non-smooth there): blocks with
/// max |dV| > 1e-6 relative ≤ 1e-8 (prototype 9.7e-10 worst, the FD floor),
/// smaller blocks absolute ≤ 1e-11.
#[test]
fn shifted_block_deriv_matches_richardson_fd_in_every_band_quadrature() {
    let rows = per_triple_rows(EcpBackend::Quadrature);
    let mut worst = [(0.0_f64, 0.0_f64, 0usize); 3];
    for r in &rows {
        let band = if r.0 < FD_MIN_DIST {
            0
        } else if r.0 < CLEAN_DIST {
            1
        } else {
            2
        };
        let w = &mut worst[band];
        w.2 += 1;
        if r.2 > 1e-6 {
            w.0 = w.0.max(r.1 / r.2);
        } else {
            w.1 = w.1.max(r.1);
        }
    }
    for (name, w) in ["d < 0.5", "0.5 <= d < 1", "d >= 1"].iter().zip(&worst) {
        eprintln!(
            "[quadrature] {name:13}: {:4} slot-blocks, worst rel {:.2e}, worst abs (small blocks) {:.2e}",
            w.2, w.0, w.1
        );
    }
    let on_centre = rows.iter().filter(|r| r.0 < 1e-9).count();
    assert!(on_centre >= 9, "no on-centre triples: {on_centre}");
    assert!(worst.iter().all(|w| w.2 >= 9), "{worst:?}");
    for w in &worst {
        assert!(w.0 < 1e-8, "{worst:?}");
        assert!(w.1 < 1e-11, "{worst:?}");
    }
}

// ============================================================ (d) quadrature target

/// FINDINGS Iteration 22 (2b): ∂/∂B_z ⟨H1s|U_I|H1s_L⟩ at L = (0, 0, 7), one
/// ECP centre at M = 0.
///
/// Two separate claims, two bars (measured 2026-09-27, first run):
/// * DERIVATIVE CONSISTENCY (primary): analytic −6.658314365644e-3 vs
///   Richardson FD of ferric's own values −6.658313625934e-3, |Δ| 7.4e-10.
///   Bar 1e-8.
/// * VALUE ACCURACY vs the independent radial × Lebedev quadrature target
///   −6.658296236e-3: |an − target| 1.8e-8 — libecpint's VALUE accuracy
///   (its radial quadrature), not a derivative error (the FD of its own
///   values carries the same offset). PySCF's `ECPscalar_ipnuc` is
///   −6.658184541e-3 (1.1e-7 off). Bar 5e-8: holds the measured 1.8e-8 and
///   still rejects ipnuc by > 2x.
#[test]
fn off_centre_element_matches_the_quadrature_target() {
    const TARGET: f64 = -6.658296236e-3;
    const PYSCF_IPNUC: f64 = -6.658184541e-3;
    const CONSISTENCY_BAR: f64 = 1e-8;
    const ACCURACY_BAR: f64 = 5e-8;
    let (dz, fd) = target_element(EcpBackend::Libecpint);
    eprintln!(
        "dB_z <H1s|U_I|H1s_L>: analytic {dz:.12e}, FD(own values) {fd:.12e} (|Δ| {:.1e}), \
         target {TARGET:.12e} (|an − target| {:.1e}, |FD − target| {:.1e}, \
         |ipnuc − target| {:.1e})",
        (dz - fd).abs(),
        (dz - TARGET).abs(),
        (fd - TARGET).abs(),
        (PYSCF_IPNUC - TARGET).abs()
    );
    assert!(
        (dz - fd).abs() < CONSISTENCY_BAR,
        "analytic {dz:e} vs own FD {fd:e}"
    );
    assert!(
        (PYSCF_IPNUC - TARGET).abs() > 2.0 * ACCURACY_BAR,
        "the accuracy bar cannot reject ipnuc"
    );
    assert!(
        (dz - TARGET).abs() < ACCURACY_BAR,
        "analytic {dz:e} vs target {TARGET:e}"
    );
}

/// `(analytic ∂/∂B_z, Richardson FD (h = 1e-4) of the backend's own values)`
/// of the (d) element.
fn target_element(backend: EcpBackend) -> (f64, f64) {
    let bra = h_sto3g(H0);
    let ket = h_sto3g(add(H0, [0.0, 0.0, 7.0]));
    let ecps = vec![i_lanl2dz_ecp(I0)];
    let an =
        ecp_block_deriv_spherical_with_backend(backend, &bra, &ket, &ecps, None, &[0], 1).unwrap();
    assert_eq!((an.nrow, an.ncol), (1, 1));
    let f = |d: f64| {
        ecp_block_spherical_with_backend(backend, &bra, &moved(&ket, &[0], 2, d), &ecps, None)
            .unwrap()
    };
    (an.ket[2][0], richardson(&f, 1e-4)[0])
}

/// (d) for the quadrature backend: the target to 5e-12 (prototype 4.4e-13 —
/// libecpint's 1.8e-8 is its radial quadrature) and its own FD to 1e-10.
#[test]
fn off_centre_element_matches_the_quadrature_target_quadrature() {
    const TARGET: f64 = -6.658296236e-3;
    const CONSISTENCY_BAR: f64 = 1e-10;
    const ACCURACY_BAR: f64 = 5e-12;
    let (dz, fd) = target_element(EcpBackend::Quadrature);
    eprintln!(
        "[quadrature] dB_z <H1s|U_I|H1s_L>: analytic {dz:.12e}, FD {fd:.12e} (|Δ| {:.1e}), \
         |an − target| {:.1e}",
        (dz - fd).abs(),
        (dz - TARGET).abs()
    );
    assert!(
        (dz - fd).abs() < CONSISTENCY_BAR,
        "analytic {dz:e} vs own FD {fd:e}"
    );
    assert!(
        (dz - TARGET).abs() < ACCURACY_BAR,
        "analytic {dz:e} vs target {TARGET:e}"
    );
}

// ============================================================ (e) invariance

/// bra + ket + Σ_g centre = 0 element-wise (per-triple translation
/// invariance), over a multi-group, masked call — both backends.
#[test]
fn bra_ket_centre_sum_to_zero_per_element() {
    for backend in [EcpBackend::Libecpint, EcpBackend::Quadrature] {
        bra_ket_centre_sum_to_zero(backend);
    }
}

fn bra_ket_centre_sum_to_zero(backend: EcpBackend) {
    let mut bra = h_sto3g(H0);
    bra.extend(i_lanl2dz(I0));
    let ket: Vec<EcpGaussianShell> = bra
        .iter()
        .map(|s| EcpGaussianShell {
            center: add(s.center, [6.0, 0.0, 0.0]),
            ..s.clone()
        })
        .collect();
    let ecps: Vec<EcpCenter> = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, -6.0, 7.0]]
        .iter()
        .map(|m| i_lanl2dz_ecp(add(I0, *m)))
        .collect();
    // A mask that drops every third triple, and two ECPs sharing a group.
    let nt = bra.len() * ket.len() * ecps.len();
    let mask: Vec<u8> = (0..nt).map(|k| u8::from(k % 3 != 0)).collect();
    let d: EcpBlockDeriv = ecp_block_deriv_spherical_with_backend(
        backend,
        &bra,
        &ket,
        &ecps,
        Some(mask.as_slice()),
        &[0, 1, 0],
        2,
    )
    .unwrap();
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for x in 0..3 {
        for k in 0..d.nrow * d.ncol {
            let s = d.bra[x][k] + d.ket[x][k] + d.centre[0][x][k] + d.centre[1][x][k];
            worst = worst.max(s.abs());
            scale = scale.max(d.bra[x][k].abs());
        }
    }
    eprintln!("[{backend}] bra + ket + centre: max {worst:.1e} (max |bra| {scale:.2e})");
    assert!(scale > 1e-3, "vacuous");
    assert!(worst < 1e-13 * scale.max(1.0), "{worst:e}");
    // The mask is honoured by the derivative kernel as by the value kernel.
    let zero = vec![0u8; nt];
    let z = ecp_block_deriv_spherical_with_backend(
        backend,
        &bra,
        &ket,
        &ecps,
        Some(zero.as_slice()),
        &[0, 1, 0],
        2,
    )
    .unwrap();
    assert!(z
        .bra
        .iter()
        .chain(&z.ket)
        .all(|v| v.iter().all(|&x| x == 0.0)));
    assert!(z
        .centre
        .iter()
        .all(|g| g.iter().all(|v| v.iter().all(|&x| x == 0.0))));
}

// ============================================================ (f) argument checks

#[test]
fn block_deriv_rejects_bad_groups_and_masks() {
    let bra = h_sto3g(H0);
    let ket = h_sto3g(add(H0, [0.0, 0.0, 7.0]));
    let ecps = vec![i_lanl2dz_ecp(I0)];
    for b in [EcpBackend::Libecpint, EcpBackend::Quadrature] {
        let f = ecp_block_deriv_spherical_with_backend;
        assert!(f(b, &bra, &ket, &ecps, None, &[1], 1).is_err());
        assert!(f(b, &bra, &ket, &ecps, None, &[0], 0).is_err());
        assert!(f(b, &bra, &ket, &ecps, None, &[0, 0], 1).is_err());
        assert!(f(b, &bra, &ket, &ecps, Some(&[1u8, 1][..]), &[0], 1).is_err());
        assert!(f(b, &bra, &ket, &[], None, &[], 1).is_err());
    }
}
