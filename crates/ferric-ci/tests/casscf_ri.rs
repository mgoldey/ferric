//! The RI integral route of CASSCF and SC-NEVPT2.
//!
//! # Exactness anchor
//!
//! The RI route is the exact CASSCF of the density-fitted Hamiltonian
//! `(mn|ls) = Σ_P B[P, mn] B[P, ls]`. Feeding the dense route that
//! reconstructed block (`RiTensor::reconstruct_ao_eri`) must reproduce the RI
//! route to rounding: only the order of the sums differs (the dense route sums
//! over `P` first and the RI route last). At one fixed orbital set (the RHF
//! orbitals rotated by a random `κ` of order 0.05, so every gradient class is
//! nonzero) the two give `F^I`, `F^A`, `(pu|vw)`, the energy, the orbital
//! gradient and a Hessian-vector product, on water CAS(4,4), N2 CAS(6,6) and
//! the O2 triplet CAS(8,6), 6-31G, def2-universal-jkfit. The same holds for
//! SC-NEVPT2 on the converged RI CASSCF reference.
//!
//! Artifact hypothesis: a wrong half transform, a dropped exchange factor or
//! a swapped `(pq|uv)`/`(pu|qv)` in the RI products moves these quantities by
//! integral-sized amounts (1e-3 and up), far above the rounding bars below;
//! a correct RI route sits at the rounding floor. Mutations checked: half the
//! exchange in the RI active Fock, and `(pu|qv)` returned in place of
//! `(pq|uv)` by the RI Hessian products.
//!
//! # Bars (a priori, from `casscf_direct.rs` with the RI chain lengths)
//!
//! With `e_abs = max_{mn,ls} Σ_P |B[P,mn]| |B[P,ls]|` bounding every term of
//! either summation order: a J/K element is a chain of `nbas² + naux` sums on
//! both routes, `δG <= 2 · 1.5 (nbas² + naux) ε e_abs Σ|D|`; `(pu|vw)` is
//! `4 nbas + naux`, `δP <= 2 (4 nbas + naux) ε e_abs c1 ca1³`; the Hessian
//! integrals `δQ <= 2 (4 nbas + naux) ε e_abs c1² co1²`. Energy, gradient and
//! the Hessian product follow from them exactly as in `casscf_direct.rs`.
//!
//! # PySCF
//!
//! `testdata/reference/casscf_ri_pyscf.json` (`scripts/gen_casscf_ri_refs.py`):
//! PySCF `density_fit()` CASSCF (`scf.RHF(mol).density_fit(aux)` then
//! `mcscf.CASSCF`) from its own orbitals, and PySCF DF CAS-CI + `mrpt.NEVPT`
//! on the orbitals ferric's RI CASSCF converged to (stored in the JSON).
//! CASSCF energy bar: `(‖g‖² + ‖g_ref‖²) / (2 λ_min) + conv_tol + δ_int`,
//! `λ_min` the lowest CI-relaxed Hessian eigenvalue by central differences at
//! ferric's point (as `casscf_vs_pyscf.rs`), `δ_int` the integral bar below.
//! Fixed-orbital CAS-CI and NEVPT2 class bars: `δ_int = 4 nbas⁴ δ_eri +
//! 8 nbas² δ_h` with `δ_eri`, `δ_h` the generator's measured agreement of the
//! two codes' exact ERIs and core Hamiltonians (the RI 3-index integrals come
//! from the same integral engines), each AO element entering the energy with
//! density weights at most 2 twice, plus the CI threshold. The NEVPT2 class
//! bars add the spread of the PySCF value over the generator's repeated
//! fixed-orbital runs (`nevpt2_class_spread`): on N2 PySCF's `Srs` moves by
//! up to ~1e-7 Ha between repeats on identical orbitals and identical CAS-CI
//! energies (labels whose norm is at rounding level enter `N / (Δ + K/N)`
//! with a noise-dominated `K/N`); every other class repeats to ~1e-16.
//! Both codes use the stored canonical orbitals as given (no re-diagonalized
//! degenerate pairs).

mod casscf_common;

use casscf_common::{hessian_min_eig, maxabs, molecule_text, num, setup_text, workspace_root, EPS};
use ferric_ci::casscf::{CasScfConfig, CasScfIntegrals, CasScfOptimizer, CasScfProblem};
use ferric_ci::nevpt2::{sc_nevpt2, sc_nevpt2_with_ao_eri, Nevpt2Classes, Nevpt2Config};
use ferric_ci::ri::RiTensor;
use ferric_ci::{run_casscf, Nevpt2Integrals, Nevpt2Reference};
use ndarray::Array2;
use serde_json::Value;

const AUX: &str = "def2-universal-jkfit";
const RI: CasScfIntegrals = CasScfIntegrals::Ri { aux: AUX };
const BUDGET: usize = 1 << 32;

fn normals(n: usize, seed: u64) -> Vec<f64> {
    let mut st = seed;
    let mut uni = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (uni(), uni());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect()
}

fn vdiff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    maxabs((a - b).iter())
}

/// Largest column 1-norm of `c`.
fn col1(c: ndarray::ArrayView2<'_, f64>) -> f64 {
    c.columns()
        .into_iter()
        .map(|col| col.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// `max_{mn,ls} Σ_P |B[P,mn]| |B[P,ls]|`.
fn abs_bound(ri: &RiTensor) -> f64 {
    let a = ri.tensor().mapv(f64::abs);
    let g = a.t().dot(&a);
    maxabs(g.iter())
}

fn ci_gap(pt: &ferric_ci::casscf::CasScfPoint, nel: (usize, usize)) -> f64 {
    use ndarray_linalg::{EigValsh, UPLO};
    let n = pt.ci.integrals.n_active;
    let space = ferric_ci::hamiltonian::DeterminantSpace {
        alpha_strings: ferric_ci::strings::enumerate_strings(n, nel.0),
        beta_strings: ferric_ci::strings::enumerate_strings(n, nel.1),
    };
    let h = ferric_ci::hamiltonian::dense_hamiltonian(&pt.ci.integrals, &space, None).unwrap();
    let w = h.eigvalsh(UPLO::Lower).unwrap();
    w[1] - w[0]
}

struct Pair<'a> {
    ri: CasScfProblem<'a>,
    dense: CasScfProblem<'a>,
    e_abs: f64,
    naux: usize,
}

fn pair<'a>(
    mol: &ferric_core::mol::Molecule,
    prep: &'a ferric_integrals::basis_bridge::PreparedBasis,
    nmo: usize,
    cfg: impl Fn(CasScfIntegrals) -> CasScfConfig,
) -> Pair<'a> {
    let t = RiTensor::build(mol, prep, AUX, BUDGET).unwrap();
    let e_abs = abs_bound(&t);
    let naux = t.naux();
    let ri = CasScfProblem::new(mol, prep, nmo, &cfg(RI)).unwrap();
    assert_eq!(ri.integrals(), RI);
    let dense = CasScfProblem::new_with_ao_eri(
        mol,
        prep,
        nmo,
        &cfg(CasScfIntegrals::Dense),
        t.reconstruct_ao_eri(),
    )
    .unwrap();
    Pair {
        ri,
        dense,
        e_abs,
        naux,
    }
}

fn check(file: &str, n_core: usize, n_active: usize, nel: (usize, usize)) {
    let (mol, prep, rhf) = setup_text(&molecule_text(file), "6-31g");
    let nmo = rhf.mos_r().ncols();
    let p = pair(&mol, &prep, nmo, |integrals| CasScfConfig {
        n_active,
        n_elec_active: nel,
        n_core,
        ci_conv_thresh: 1e-12,
        integrals,
        ..Default::default()
    });
    let kappa: Vec<f64> = normals(p.dense.pairs().len(), 11)
        .iter()
        .map(|x| 0.05 * x)
        .collect();
    let c = p.dense.rotate(rhf.mos_r(), &kappa).unwrap();
    let n = prep.nbasis() as f64;
    let naux = p.naux as f64;
    let na = n_active as f64;
    let ne = (nel.0 + nel.1) as f64;
    let c1 = col1(c.view());
    let ca1 = col1(c.slice(ndarray::s![.., n_core..n_core + n_active]));

    let paaa_d = p.dense.paaa(&c).unwrap();
    let paaa_r = p.ri.paaa(&c).unwrap();
    let dp = vdiff(&paaa_r, &paaa_d);
    let bar_p = 2.0 * (4.0 * n + naux) * EPS * p.e_abs * c1 * ca1.powi(3);
    assert!(dp <= bar_p, "δP = {dp:e} > {bar_p:e}");

    let d = p.dense.evaluate(&c).unwrap();
    let r = p.ri.evaluate(&c).unwrap();
    let sum_d = (2.0 * (n_core as f64) * c1 * c1).max(2.0 * na * ca1 * ca1);
    let bar_f = c1 * c1 * 2.0 * 1.5 * (n * n + naux) * EPS * p.e_abs * sum_d;
    let df = diff(&r.f_inactive, &d.f_inactive).max(diff(&r.f_active, &d.f_active));
    let gap = ci_gap(&d, nel);
    let bar_e = (n_core as f64) * df
        + 2.0 * na * na * df
        + na.powi(4) * dp
        + 4.0 * n * n * EPS * d.e_total.abs();
    let dh_ci = na * na * ne * df + 0.5 * na.powi(4) * ne * ne * dp;
    let dc = dh_ci / gap;
    let (dgam, dgam2) = (2.0 * ne * dc, 2.0 * ne * ne * dc);
    let fmax = maxabs(d.f_inactive.iter());
    let pmax = maxabs(paaa_d.iter());
    let bar_g = 4.0
        * (4.0 * df
            + 2.0 * na * df
            + 2.0 * na.powi(3) * dp
            + na * fmax * dgam
            + na.powi(3) * pmax * dgam2);
    let de = (r.e_total - d.e_total).abs();
    let de2 = (p.ri.energy(&c).unwrap() - p.dense.energy(&c).unwrap()).abs();
    let dg = vdiff(&r.gradient, &d.gradient);
    eprintln!(
        "{file} CAS({},{n_active}) naux {}: δF {df:.2e} (bar {bar_f:.1e}) δP {dp:.2e} \
         (bar {bar_p:.1e}) |ΔE| {de:.2e} energy() {de2:.2e} (bar {bar_e:.1e}) |Δg| {dg:.2e} \
         (bar {bar_g:.1e})",
        nel.0 + nel.1,
        p.naux
    );
    assert!(maxabs(d.gradient.iter()) > 1e-2, "gradient not exercised");
    assert!(df <= bar_f, "δF = {df:e} > {bar_f:e}");
    assert!(de <= bar_e, "|ΔE| = {de:e} > {bar_e:e}");
    assert!(de2 <= bar_e, "|ΔE| (energy()) = {de2:e} > {bar_e:e}");
    assert!(dg <= bar_g, "|Δg| = {dg:e} > {bar_g:e}");
}

#[test]
fn ri_matches_dense_on_the_reconstructed_block_h2o_cas44() {
    check("water.xyz", 3, 4, (2, 2));
}

#[test]
fn ri_matches_dense_on_the_reconstructed_block_n2_cas66() {
    check("n2.xyz", 4, 6, (3, 3));
}

#[test]
fn ri_matches_dense_on_the_reconstructed_block_o2_triplet_cas86() {
    check("o2.xyz", 4, 6, (5, 3));
}

fn hessian_check(file: &str, n_core: usize, n_active: usize, nel: (usize, usize)) {
    let (mol, prep, rhf) = setup_text(&molecule_text(file), "6-31g");
    let nmo = rhf.mos_r().ncols();
    let p = pair(&mol, &prep, nmo, |integrals| CasScfConfig {
        n_active,
        n_elec_active: nel,
        n_core,
        ci_conv_thresh: 1e-12,
        integrals,
        optimizer: CasScfOptimizer::AugmentedHessian,
        ..Default::default()
    });
    let npair = p.dense.pairs().len();
    let kappa: Vec<f64> = normals(npair, 11).iter().map(|x| 0.05 * x).collect();
    let c = p.dense.rotate(rhf.mos_r(), &kappa).unwrap();
    let pt = p.dense.evaluate(&c).unwrap();
    let hd = p.dense.orbital_hessian(&c, &pt).unwrap();
    let hr = p.ri.orbital_hessian(&c, &pt).unwrap();
    let mut x = normals(npair, 17);
    let xn = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    x.iter_mut().for_each(|v| *v /= xn);
    let a = p.dense.hessian_vector(&hd, &x).unwrap();
    let b = p.ri.hessian_vector(&hr, &x).unwrap();
    // The AO J/K oracle of the products on the RI route (general-D K).
    let hr_ao =
        p.ri.orbital_hessian_with(&c, &pt, ferric_ci::casscf_newton::HessianProducts::AoJk)
            .unwrap();
    let b_ao = p.ri.hessian_vector(&hr_ao, &x).unwrap();

    let n = prep.nbasis() as f64;
    let naux = p.naux as f64;
    let (nc, na) = (n_core as f64, n_active as f64);
    let c1 = col1(c.view());
    let co1 = col1(c.slice(ndarray::s![.., ..n_core + n_active]));
    let mut xm = Array2::<f64>::zeros((nmo, nmo));
    for (&(p_, q), &v) in p.dense.pairs().iter().zip(&x) {
        xm[(p_, q)] = v;
        xm[(q, p_)] = -v;
    }
    let xs = col1(xm.view());
    let x1: f64 = x.iter().map(|v| v.abs()).sum();
    let dq = 2.0 * (4.0 * n + naux) * EPS * p.e_abs * (c1 * co1).powi(2);
    let dg_c = 3.0 * dq * 2.0 * nc * xs;
    let dg_a = 3.0 * dq * 2.0 * na * na * xs;
    let bar = 4.0
        * (2.0 * (dg_c + dg_a)
            + 2.0 * na * dg_c
            + 12.0 * na * na * dq * x1
            + 4.0 * na.powi(3) * dq * xs);
    let d = vdiff(&a, &b);
    let d_ao = vdiff(&a, &b_ao);
    eprintln!(
        "{file}: max|Hx| {:.3}, |δ(Hx)| kernel {d:.2e}, AO J/K {d_ao:.2e} (bar {bar:.1e})",
        maxabs(a.iter())
    );
    assert!(maxabs(a.iter()) > 1e-2, "the product is not exercised");
    assert!(d <= bar, "|δ(Hx)| = {d:e} > {bar:e}");
    assert!(d_ao <= bar, "|δ(Hx)| (AO J/K) = {d_ao:e} > {bar:e}");
}

#[test]
fn ri_hessian_product_matches_dense_on_the_reconstructed_block_h2o_cas44() {
    hessian_check("water.xyz", 3, 4, (2, 2));
}

#[test]
fn ri_hessian_product_matches_dense_on_the_reconstructed_block_n2_cas66() {
    hessian_check("n2.xyz", 4, 6, (3, 3));
}

#[test]
fn ri_hessian_product_matches_dense_on_the_reconstructed_block_o2_triplet_cas86() {
    hessian_check("o2.xyz", 4, 6, (5, 3));
}

/// SC-NEVPT2 on the converged RI CASSCF: the RI route against the dense
/// route fed the reconstructed block, same reference and same (canonical,
/// not re-diagonalized) orbitals. Every MO integral the classes contract
/// moves by at most `δQ = 2 (4 nbas + naux) ε e_abs c1⁴`. A class energy is
/// `Σ_l N_l / Δ_eff,l` with `N_l`, `K_l` quadratic in the label's integrals
/// `v` (weights: RDM and active-Hamiltonian elements, bounded by `w = 4 n_a²`
/// per pair) and `Δ_eff >= gap`, the smaller of the lowest virtual and minus
/// the highest core orbital energy. To first order `|δE| <= Σ 2 |v| w δQ /
/// gap`, and by Cauchy–Schwarz over at most `A = nmo⁴` integrals with
/// `Σ v² w / gap ~ |E|`: `|δE| <= 2 δQ sqrt(A |E| w / gap)`, plus `4 ε |E|`
/// for the final sums. A first-order estimate, not a rigorous bound; an
/// integral-sized error (1e-3 and up) exceeds it by orders of magnitude.
fn nevpt2_check(file: &str, n_core: usize, n_active: usize, nel: (usize, usize)) {
    let (mol, prep, rhf) = setup_text(&molecule_text(file), "6-31g");
    let cfg = CasScfConfig {
        n_active,
        n_elec_active: nel,
        n_core,
        canonicalize: true,
        integrals: RI,
        ..Default::default()
    };
    let cas = run_casscf(&mol, &prep, &rhf, &cfg).unwrap();
    assert!(cas.converged);
    assert_eq!(cas.integrals, RI);
    let reference = Nevpt2Reference::from_casscf(&cas, &cfg).unwrap();
    // The CASSCF orbitals are already canonical; both routes use them as
    // given, so neither re-diagonalizes a degenerate block (N2's π pairs)
    // into a different rotation, to which SC-NEVPT2 is not invariant.
    let fixed = Nevpt2Config {
        canonicalize: false,
        ..Default::default()
    };
    let ri_cfg = Nevpt2Config {
        integrals: Nevpt2Integrals::Ri { aux: AUX },
        ..fixed
    };
    let r = sc_nevpt2(&mol, &prep, &reference, &ri_cfg).unwrap();
    let t = RiTensor::build(&mol, &prep, AUX, BUDGET).unwrap();
    let e_abs = abs_bound(&t);
    let d = sc_nevpt2_with_ao_eri(&mol, &prep, &reference, &fixed, t.reconstruct_ao_eri()).unwrap();
    let n = prep.nbasis() as f64;
    let nmo = cas.mo_coeff.ncols();
    let c1 = col1(cas.mo_coeff.view());
    let dq = 2.0 * (4.0 * n + t.naux() as f64) * EPS * e_abs * c1.powi(4);
    let eps = cas.mo_energy.as_ref().expect("canonical CASSCF");
    let gap = eps[n_core + n_active].min(-eps[n_core - 1]);
    assert!(gap > 0.0);
    let amplitudes = (nmo as f64).powi(4);
    let weight = 4.0 * (n_active as f64).powi(2);
    for (name, (a, b)) in Nevpt2Classes::NAMES
        .iter()
        .zip(r.energies.to_array().iter().zip(d.energies.to_array()))
    {
        let bar = 2.0 * dq * (amplitudes * b.abs() * weight / gap).sqrt() + 4.0 * EPS * b.abs();
        let dd = (a - b).abs();
        eprintln!("{file} {name:6} RI {a:.14} dense(BᵀB) {b:.14} diff {dd:.1e} bar {bar:.1e}");
        assert!(dd <= bar, "{file} {name}: |Δ| {dd:e} > {bar:e}");
    }
    assert!(r.e_corr.abs() > 1e-3, "NEVPT2 not exercised");
}

#[test]
fn ri_nevpt2_matches_dense_on_the_reconstructed_block_h2o_cas44() {
    nevpt2_check("water.xyz", 3, 4, (2, 2));
}

#[test]
fn ri_nevpt2_matches_dense_on_the_reconstructed_block_n2_cas66() {
    nevpt2_check("n2.xyz", 4, 6, (3, 3));
}

fn refs() -> Value {
    let path = workspace_root().join("testdata/reference/casscf_ri_pyscf.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

/// Ferric's RI CASSCF against PySCF `density_fit()` CASSCF (independently
/// converged), and on ferric's stored RI orbitals the fixed-orbital DF CAS-CI
/// energy and the SC-NEVPT2 class energies against PySCF's.
fn vs_pyscf(name: &str) {
    let all = refs();
    let s = all["systems"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["name"] == name)
        .unwrap_or_else(|| panic!("no reference {name}"));
    assert_eq!(s["auxbasis"].as_str(), Some(AUX));
    let xyz = s["xyz"].as_str().unwrap();
    let (mol, prep, rhf) = setup_text(xyz, s["basis"].as_str().unwrap());
    let nel = s["n_elec_active"].as_array().unwrap();
    let nel = (
        nel[0].as_u64().unwrap() as usize,
        nel[1].as_u64().unwrap() as usize,
    );
    let cfg = CasScfConfig {
        n_core: s["n_core"].as_u64().unwrap() as usize,
        n_active: s["n_active"].as_u64().unwrap() as usize,
        n_elec_active: nel,
        canonicalize: true,
        integrals: RI,
        ..Default::default()
    };
    let n = prep.nbasis() as f64;
    let d_int = 4.0 * n.powi(4) * num(s, "delta_eri") + 8.0 * n * n * num(s, "delta_hcore");

    // 1. Independently converged CASSCF energies.
    let res = run_casscf(&mol, &prep, &rhf, &cfg).unwrap();
    assert!(res.converged);
    let problem = CasScfProblem::new(&mol, &prep, res.mo_coeff.ncols(), &cfg).unwrap();
    let lam = hessian_min_eig(&problem, &res.mo_coeff, 1e-4);
    assert!(lam > 1e-3, "{name}: not a minimum (λ_min {lam:e})");
    let p = &s["pyscf_casscf"];
    let g_ref = 2.0 * num(p, "grad_norm");
    let bar = (res.grad_norm.powi(2) + g_ref.powi(2)) / (2.0 * lam) + num(p, "conv_tol") + d_int;
    let de = res.e_total - num(p, "e_total");
    eprintln!(
        "{name}: RI CASSCF {:.12} PySCF DF {:.12} dE {de:.2e} (bar {bar:.1e}); exact-integral \
         PySCF CASSCF {:.12} (RI error {:.2e})",
        res.e_total,
        num(p, "e_total"),
        num(s, "e_exact_casscf"),
        res.e_total - num(s, "e_exact_casscf")
    );
    assert!(de.abs() <= bar, "{name}: |dE| {:e} > {bar:e}", de.abs());

    // 2. Fixed orbitals: ferric's stored RI CASSCF orbitals.
    let f = &s["ferric_ri"];
    let shape = f["mo_coeff_shape"].as_array().unwrap();
    let (nb, nm) = (
        shape[0].as_u64().unwrap() as usize,
        shape[1].as_u64().unwrap() as usize,
    );
    let c = Array2::from_shape_vec((nb, nm), casscf_common::f64s(f, "mo_coeff")).unwrap();
    let pt = problem.evaluate(&c).unwrap();
    let q = &s["pyscf_fixed_orbitals"];
    let bar_ci = d_int + 1e-10;
    let de = pt.e_total - num(q, "e_casci");
    eprintln!("{name}: fixed-orbital CAS-CI ferric RI {:.12} PySCF DF {:.12} dE {de:.1e} (bar {bar_ci:.1e})", pt.e_total, num(q, "e_casci"));
    assert!(
        de.abs() <= bar_ci,
        "{name}: fixed-orbital CAS-CI |dE| {:e} > {bar_ci:e}",
        de.abs()
    );
    if nel.0 != nel.1 {
        return;
    }
    let reference = Nevpt2Reference {
        mo_coeff: &c,
        n_core: cfg.n_core,
        n_active: cfg.n_active,
        n_elec_active: nel,
        ci_vector: &pt.ci.ci_vector,
    };
    let nev = sc_nevpt2(
        &mol,
        &prep,
        &reference,
        // The stored orbitals are canonical; both codes use them as given
        // (SC-NEVPT2 is not invariant to rotations within N2's degenerate π
        // pairs, which a re-diagonalization picks arbitrarily).
        &Nevpt2Config {
            integrals: Nevpt2Integrals::Ri { aux: AUX },
            canonicalize: false,
            ..Default::default()
        },
    )
    .unwrap();
    for (cls, got) in Nevpt2Classes::NAMES.iter().zip(nev.energies.to_array()) {
        let want = num(&q["nevpt2_classes"], cls);
        let bar = bar_ci + num(&q["nevpt2_class_spread"], cls);
        let d = got - want;
        eprintln!(
            "{name} {cls:6} ferric RI {got:.12} PySCF DF {want:.12} diff {d:.1e} bar {bar:.1e}"
        );
        assert!(d.abs() <= bar, "{name} {cls}: |Δ| {:e} > {bar:e}", d.abs());
    }
}

#[test]
fn ri_casscf_and_nevpt2_match_pyscf_density_fit_h2o_cas44() {
    vs_pyscf("h2o_cas44_631g");
}

#[test]
fn ri_casscf_and_nevpt2_match_pyscf_density_fit_n2_cas66() {
    vs_pyscf("n2_cas66_631g");
}

#[test]
fn ri_casscf_matches_pyscf_density_fit_o2_triplet_cas86() {
    vs_pyscf("o2_triplet_cas86_631g");
}

#[test]
fn the_ri_route_needs_a_bundled_auxbasis_and_reads_no_threshold() {
    for (route, t, aux, want) in [
        ("ri", None, None, "needs auxbasis"),
        ("ri", Some(1e-12), Some(AUX), "integral_thresh is read by"),
        ("dense", None, Some(AUX), "auxbasis is read by"),
        ("ri", None, Some("no-such-aux"), "not a bundled basis set"),
    ] {
        let e = CasScfIntegrals::parse_config_str(route, t, aux).unwrap_err();
        assert!(e.contains(want), "{route} {t:?} {aux:?}: {e}");
    }
    assert_eq!(
        CasScfIntegrals::parse_config_str("ri", None, Some("DEF2/JK")).unwrap(),
        RI
    );
}
