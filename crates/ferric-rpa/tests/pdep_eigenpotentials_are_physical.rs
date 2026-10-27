//! `PdepRpaResult::eigenpotentials` are PHYSICAL aux-basis coefficients.
//!
//! Exactness anchor, independent of any external reference: with ferric's own
//! aux metric V = (P|op|Q) (`threeindex::coulomb_metric_2c`) and its own
//! physical-aux static polarizability Π = R diag(4/Δε) Rᵀ (R = raw (Q|ia)
//! from `eri3_mo_ov_blocked`, no metric factor anywhere), the exported
//! columns c_α must satisfy
//!
//! ```text
//!   cᵀ V c       = I
//!   cᵀ (V + Π) c = diag(λ_α(0))      (λ = eigenvalues_static)
//! ```
//!
//! i.e. c are the generalized eigenvectors of (V + Π) c = λ V c, which is the
//! basis-independent definition of the static dielectric eigenpotentials. The
//! checks never touch `v_inv_sqrt`, so they cannot be satisfied by a
//! factor-dependent storage convention.
//!
//! Both factorizations `metric_inverse_sqrt` can return are covered:
//! * Coulomb → lower-triangular Cholesky L⁻¹ (NOT symmetric);
//! * erf(ω)  → symmetric eigh V^{-1/2} with near-null modes dropped (for the
//!   kept modes, λ > 1, the eigenvector lies in the factor's range so the
//!   identities still hold exactly).
//! and the unrestricted path (`run_u_pdep_rpa`, per-spin prefactor 2).
//!
//! Artifact hypothesis: if the export were c = F·u (the dressed vector pushed
//! through the factor rather than its transpose) the Coulomb case gives
//! cᵀVc − I of order 1e3 (measured on water/cc-pVDZ from the CLI NPZ before
//! the fix), while the erf case would still pass (F symmetric), so the erf
//! case pins the factorization generality, not the transpose.
//!
//! MUTATION (documented, run by hand): reverting `.t()` in
//! `run_pdep_rpa_eigensolve` (`inter.v_inv_sqrt.t().dot(..)` →
//! `inter.v_inv_sqrt.dot(..)`) fails `coulomb_cholesky_*`; reverting it at the
//! `run_u_pdep_rpa` site fails `unrestricted_*`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ferric_mp2::rimp2::{eri3_budget_bytes, eri3_mo_ov_blocked, RpaIntermediates};
use ferric_rpa::{run_pdep_rpa, run_u_pdep_rpa, PdepRpaConfig, PdepRpaResult};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::{Array2, Axis};

const TOL: f64 = 1e-9;

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        eigensolver_conv_thresh: 1e-10,
        need_eigenvalues_freq: false,
        ..Default::default()
    }
}

/// Raw (un-dressed) (Q|ia) for one spin channel, flattened to (naux, nocc*nvir)
/// in ferric's i*nvir+a order, plus the matching Δε_ia.
fn raw_ov(
    op: Operator,
    obs: &PreparedBasis,
    dfbs: &PreparedBasis,
    mos: &Array2<f64>,
    eps: &[f64],
    nocc: usize,
) -> (Array2<f64>, Vec<f64>) {
    let nmo = mos.ncols();
    let nvir = nmo - nocc;
    let c_occ = mos.slice(ndarray::s![.., ..nocc]).to_owned();
    let c_vir = mos.slice(ndarray::s![.., nocc..]).to_owned();
    let r = eri3_mo_ov_blocked(op, obs, dfbs, &c_occ, &c_vir, eri3_budget_bytes(None)).unwrap();
    let naux = r.shape()[0];
    let r = r.into_shape_with_order((naux, nocc * nvir)).unwrap();
    let mut de = Vec::with_capacity(nocc * nvir);
    for i in 0..nocc {
        for a in 0..nvir {
            de.push(eps[nocc + a] - eps[i]);
        }
    }
    (r, de)
}

/// Π += R diag(pref/Δε) Rᵀ.
fn add_pi(pi: &mut Array2<f64>, r: &Array2<f64>, de: &[f64], pref: f64) {
    let mut rs = r.clone();
    for (ia, mut col) in rs.axis_iter_mut(Axis(1)).enumerate() {
        col.mapv_inplace(|x| x * pref / de[ia]);
    }
    *pi += &rs.dot(&r.t());
}

/// max|cᵀVc − I| and max|cᵀ(V+Π)c − diag(λ)| (the latter relative to max λ).
fn check(label: &str, res: &PdepRpaResult, v: &Array2<f64>, pi: &Array2<f64>) {
    let c = &res.eigenpotentials;
    let m = res.n_eigenpotentials;
    assert_eq!(
        c.nrows(),
        v.nrows(),
        "{label}: eigenpotentials rows != naux"
    );
    assert_eq!(
        c.ncols(),
        m,
        "{label}: eigenpotentials cols != n_eigenpotentials"
    );
    assert!(
        m >= 2,
        "{label}: need >= 2 modes for an off-diagonal check, got {m}"
    );
    let gram = c.t().dot(&v.dot(c));
    let vp = v + pi;
    let h = c.t().dot(&vp.dot(c));
    let lam = &res.eigenvalues_static;
    let lam_max = lam.iter().fold(1.0_f64, |a, &b| a.max(b.abs()));
    let mut err_g = 0.0_f64;
    let mut err_h = 0.0_f64;
    for a in 0..m {
        for b in 0..m {
            let id = if a == b { 1.0 } else { 0.0 };
            err_g = err_g.max((gram[(a, b)] - id).abs());
            err_h = err_h.max((h[(a, b)] - id * lam[a]).abs());
        }
    }
    eprintln!(
        "{label}: M={m} naux={} max|cᵀVc-I|={err_g:.3e} max|cᵀ(V+Π)c-diag(λ)|={err_h:.3e} \
         λ[0..3]={:?}",
        v.nrows(),
        &lam[..3.min(m)]
    );
    assert!(
        err_g <= TOL,
        "{label}: cᵀVc deviates from I by {err_g:.3e} (> {TOL:.0e})"
    );
    assert!(
        err_h <= TOL * lam_max,
        "{label}: cᵀ(V+Π)c deviates from diag(λ) by {err_h:.3e} (> {:.1e})",
        TOL * lam_max
    );
}

fn water() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap()
}

fn closed_shell_case(label: &str, obs_name: &str, op: Operator) {
    let ctx = ParallelContext::default();
    let mol = water();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    // Full-Coulomb RHF reference; `op` only enters the RI/RPA kernel.
    let coul = Operator::coulomb();
    let bounds = SchwarzBounds::compute(coul, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, coul, &bounds, &RhfConfig::default()).unwrap();

    let res = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &pdep_cfg()).unwrap();
    assert!(
        res.eigensolver_converged,
        "{label}: eigensolver not converged"
    );

    let v = coulomb_metric_2c(op, &dfbs).unwrap();
    let nocc = mol.nelec() as usize / 2;
    let (r, de) = raw_ov(op, &obs, &dfbs, rhf.mos_r(), rhf.eps_r(), nocc);
    let mut pi = Array2::<f64>::zeros(v.raw_dim());
    add_pi(&mut pi, &r, &de, 4.0);
    check(label, &res, &v, &pi);
}

#[test]
fn coulomb_cholesky_eigenpotentials_are_v_orthonormal_and_diagonalize_v_plus_pi() {
    closed_shell_case("H2O/cc-pVDZ Coulomb", "cc-pvdz", Operator::coulomb());
}

#[test]
fn erf_symmetric_eigenpotentials_are_v_orthonormal_and_diagonalize_v_plus_pi() {
    // ω = 0.2 Bohr⁻¹: the eigh factor drops 28 of 84 near-null metric modes
    // here (PySCF numpy mirror of the same construction: identities hold to
    // 4e-12), so this also exercises the dropped-mode branch.
    closed_shell_case("H2O/STO-3G erf(0.2)", "sto-3g", Operator::erf(0.2));
}

#[test]
fn unrestricted_eigenpotentials_are_v_orthonormal_and_diagonalize_v_plus_pi() {
    let ctx = ParallelContext::default();
    // OH radical (doublet), Bohr-free xyz in Angstrom.
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &RhfConfig::default()).unwrap();

    let res = run_u_pdep_rpa(&mol, &obs, &dfbs, op, &uhf, &pdep_cfg()).unwrap();
    assert!(res.eigensolver_converged, "U: eigensolver not converged");

    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    let nocc_a = (nelec + two_s) / 2;
    let nocc_b = (nelec - two_s) / 2;
    let v = coulomb_metric_2c(op, &dfbs).unwrap();
    let mut pi = Array2::<f64>::zeros(v.raw_dim());
    let (ra, dea) = raw_ov(op, &obs, &dfbs, &uhf.mos_alpha, &uhf.eps_alpha, nocc_a);
    add_pi(&mut pi, &ra, &dea, 2.0);
    let mos_b = uhf.mos_beta.as_ref().expect("UHF has beta MOs");
    let eps_b = uhf.eps_beta.as_ref().expect("UHF has beta energies");
    let (rb, deb) = raw_ov(op, &obs, &dfbs, mos_b, eps_b, nocc_b);
    add_pi(&mut pi, &rb, &deb, 2.0);
    check("OH/STO-3G UHF Coulomb", &res, &v, &pi);
}

// ===========================================================================
// The `_from_parts` entries (#573): ONE `v_inv_sqrt` convention, any factor.
// ===========================================================================
//
// Synthetic, molecule-free fixtures: a random SPD metric V (naux_src x
// naux_src), random physical (Q|ia) columns J, and a metric factor F with
// F V Fᵀ = I that is NON-symmetric (a random rotation of the kept
// eigenvectors) and, after dropping the `drop` smallest metric modes,
// RECTANGULAR (naux_src - drop, naux_src) — the periodic RS-GDF shape
// `F = Wᵀ`. b_ov = F·J. The identities of `check` must hold through both
// parts entries.
//
// Artifact hypothesis: a back-transform c = F·u (the closed-shell parts
// entry before #573) fails cᵀVc = I by O(1) on the square rotated factor and
// cannot even be formed on the rectangular one; the old shape guard
// (`ncols == naux`) refused every rectangular F in the molecular-shaped
// convention, so the rectangular cases fail on the old code as well.
//
// MUTATIONS (run by hand, each fails a test below):
// * `run_pdep_rpa_from_parts` back-transform `.t().dot` → `.dot`:
//   `parts_entries_eigenpotentials_*` fails `check` (cᵀVc = I) on the
//   square rotated factor; ferric-pbc's pbc_drpa (6) fails too (max|Δc| 4.4).
// * `u_pdep_rpa_core` back-transform `.t().dot` → `.dot`: the U case fails
//   (and ferric-pbc's lindep-drop gamma_urpa test panics in the GEMM).
// * `check_metric_factor` reverted to `ncols() != naux`: both tests fail
//   (rectangular F refused; a raw W is no longer refused and panics).

/// Deterministic uniform(-1, 1) stream (no rand dev-dependency).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn mat(&mut self, r: usize, c: usize) -> Array2<f64> {
        Array2::from_shape_fn((r, c), |_| self.next())
    }
}

struct Synth {
    v: Array2<f64>,
    /// F, `(naux_src - drop, naux_src)`, F V Fᵀ = I, non-symmetric.
    f: Array2<f64>,
}

fn synth_metric(rng: &mut Lcg, n: usize, drop: usize) -> Synth {
    use ndarray_linalg::{Eigh, UPLO};
    let x = rng.mat(n, n);
    let v = x.dot(&x.t()) + Array2::<f64>::eye(n) * 0.5;
    let (w, u) = v.eigh(UPLO::Lower).unwrap();
    let k = n - drop;
    // W = U_kept s_kept^{-1/2} Q with Q a random rotation: still Wᵀ V W = I,
    // but F = Wᵀ is neither symmetric nor triangular.
    let y = rng.mat(k, k);
    let (_, q) = (&y + &y.t()).eigh(UPLO::Lower).unwrap();
    let mut wk = u.slice(ndarray::s![.., drop..]).to_owned();
    for (j, mut col) in wk.axis_iter_mut(Axis(1)).enumerate() {
        col.mapv_inplace(|z| z / w[drop + j].sqrt());
    }
    let f = wk.dot(&q).t().to_owned();
    Synth { v, f }
}

/// e_ia > 0: occupied in [-1.2, -0.6], virtual in [0.2, 1.4].
fn synth_eps(rng: &mut Lcg, nocc: usize, nvir: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Ascending, as a Fock-diagonal basis has them (sternheimer refuses
    // heavily unsorted energies).
    let sorted = |mut x: Vec<f64>| {
        x.sort_by(f64::total_cmp);
        x
    };
    let eo = sorted((0..nocc).map(|_| -0.9 + 0.3 * rng.next()).collect());
    let ev = sorted((0..nvir).map(|_| 0.8 + 0.6 * rng.next()).collect());
    let de = eo
        .iter()
        .flat_map(|&ei| ev.iter().map(move |&ea| ea - ei))
        .collect();
    (eo, ev, de)
}

fn parts_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        trunc_thresh: 0.0,
        ..pdep_cfg()
    }
}

fn synth_inter(f: &Array2<f64>, j: &Array2<f64>, nocc: usize, nvir: usize) -> RpaIntermediates {
    RpaIntermediates {
        b_ov: f.dot(j),
        v_inv_sqrt: f.clone(),
        nocc,
        nvir,
        nocc_total: nocc,
        first_occ: 0,
        naux: f.nrows(),
    }
}

#[test]
fn parts_entries_eigenpotentials_are_physical_for_a_rectangular_nonsymmetric_factor() {
    let (n, nocc, nvir) = (12, 3, 4);
    for drop in [0, 3] {
        let mut rng = Lcg(0x5eed + drop as u64);
        let s = synth_metric(&mut rng, n, drop);
        let fvf = s.f.dot(&s.v).dot(&s.f.t()) - Array2::<f64>::eye(n - drop);
        let asym = if drop == 0 {
            (&s.f - &s.f.t())
                .iter()
                .fold(0.0_f64, |m, x| m.max(x.abs()))
        } else {
            f64::NAN
        };
        eprintln!(
            "F {:?}: |F V Fᵀ - I| = {:.1e}, |F - Fᵀ| = {asym:.2e}",
            s.f.dim(),
            fvf.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
        );

        // Closed shell (χ₀ prefactor 4).
        let j = rng.mat(n, nocc * nvir);
        let (eo, ev, de) = synth_eps(&mut rng, nocc, nvir);
        let inter = synth_inter(&s.f, &j, nocc, nvir);
        let res = ferric_rpa::run_pdep_rpa_from_parts(&inter, &eo, &ev, &parts_cfg()).unwrap();
        let mut pi = Array2::<f64>::zeros((n, n));
        add_pi(&mut pi, &j, &de, 4.0);
        check(&format!("parts closed drop={drop}"), &res, &s.v, &pi);

        // Open shell (per-spin prefactor 2), α (3,4) and β (2,5), ONE F.
        let (na, va, nb, vb) = (3, 4, 2, 5);
        let ja = rng.mat(n, na * va);
        let jb = rng.mat(n, nb * vb);
        let (eoa, eva, dea) = synth_eps(&mut rng, na, va);
        let (eob, evb, deb) = synth_eps(&mut rng, nb, vb);
        let ia = synth_inter(&s.f, &ja, na, va);
        let ib = synth_inter(&s.f, &jb, nb, vb);
        let ures =
            ferric_rpa::run_u_pdep_rpa_from_parts(&ia, &ib, &eoa, &eva, &eob, &evb, &parts_cfg())
                .unwrap();
        let mut upi = Array2::<f64>::zeros((n, n));
        add_pi(&mut upi, &ja, &dea, 2.0);
        add_pi(&mut upi, &jb, &deb, 2.0);
        check(&format!("parts open drop={drop}"), &ures, &s.v, &upi);
    }
}

#[test]
fn parts_entries_refuse_an_untransposed_map_and_mismatched_spin_factors() {
    let (n, drop, nocc, nvir) = (10, 2, 2, 3);
    let mut rng = Lcg(7);
    let s = synth_metric(&mut rng, n, drop);
    let j = rng.mat(n, nocc * nvir);
    let (eo, ev, _) = synth_eps(&mut rng, nocc, nvir);
    // The old periodic mistake: W = Fᵀ `(naux_src, naux)` stored as-is.
    let mut bad = synth_inter(&s.f, &j, nocc, nvir);
    bad.v_inv_sqrt = s.f.t().to_owned();
    let msg = ferric_rpa::run_pdep_rpa_from_parts(&bad, &eo, &ev, &parts_cfg())
        .expect_err("closed-shell parts must refuse W")
        .to_string();
    eprintln!("refused (closed, W): {msg}");
    assert!(msg.contains("TRANSPOSE"), "{msg}");
    let msg = ferric_rpa::run_u_pdep_rpa_from_parts(&bad, &bad, &eo, &ev, &eo, &ev, &parts_cfg())
        .expect_err("open-shell parts must refuse W")
        .to_string();
    eprintln!("refused (open, W): {msg}");
    assert!(msg.contains("TRANSPOSE"), "{msg}");
    // α and β with different factors: the back-transform reads α's only.
    let good = synth_inter(&s.f, &j, nocc, nvir);
    let mut other = good.clone();
    other.v_inv_sqrt[(0, 0)] += 1e-3;
    let msg =
        ferric_rpa::run_u_pdep_rpa_from_parts(&good, &other, &eo, &ev, &eo, &ev, &parts_cfg())
            .expect_err("different spin factors must be refused")
            .to_string();
    eprintln!("refused (spin factors differ): {msg}");
    assert!(msg.contains("different metric factors"), "{msg}");
}
