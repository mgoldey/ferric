//! Converged two-step CASSCF energies against `pyscf.mcscf.CASSCF`.
//!
//! References: `testdata/reference/casscf_<system>_pyscf.json`, written by
//! `scripts/gen_casscf_refs.py`. PySCF runs on ferric's AO integrals
//! (hcore, overlap, ERIs, nuclear repulsion from `ferric.AoIntegrals`) and
//! converges from its own RHF orbitals on those integrals. Ferric starts here
//! from its own exact-J/K RHF; no orbital passes between the codes. The
//! generator settings (PySCF `conv_tol` 1e-12, `conv_tol_grad` 1e-6, FCI
//! `conv_tol` 1e-14) are recorded in each JSON, and the generator refuses
//! to write a reference PySCF did not converge.
//!
//! Systems, 6-31G, closed-shell RHF start:
//!
//! * H2O CAS(4,4) above 3 core orbitals (`water.xyz`);
//! * N2 CAS(6,6) above 4 (`n2.xyz`, r = 1.0977 Å);
//! * N2 CAS(6,6) at r = 1.6 Å (geometry in the JSON), natural occupations
//!   1.90 / 1.72 / 0.28 / 0.10, a system the prototype never ran;
//! * O2 CAS(8,6) above 4 with `(n_alpha, n_beta) = (5, 3)` (`o2.xyz`): the
//!   lowest `M_s = 1` state, the triplet ground state (PySCF `<S²>` = 2).
//!
//! # Bar
//!
//! Near a minimum `E − E_min ≤ ‖g‖² / (2 λ_min)` for each code, so
//! `|ΔE| ≤ (‖g‖² + ‖g_ref‖²) / (2 λ_min) + conv_tol + 2 · 4 ε |E| √N_det`.
//! `λ_min` is the lowest eigenvalue of the CI-relaxed orbital Hessian by
//! central differences of ferric's analytic gradient (`h = 1e-4`) at ferric's
//! converged orbitals. `‖g_ref‖` is twice PySCF's packed gradient norm: its
//! packing omits this crate's factor 2 on some entries, so `2 ‖g_PySCF‖`
//! bounds the norm in this crate's convention.
//!
//! Physics vs artifact: if both codes reach the same minimum, `ΔE` sits inside
//! the bar and the natural occupations and canonical orbital energies agree to
//! first order in the orbital error. A different stationary point shows as
//! `ΔE` far outside the bar (1e-3 Ha scale) with different natural
//! occupations; it is recorded as a different solution, the bar is not
//! widened. The occupations and the eigenvalues of the core and virtual blocks
//! of `F^I + F^A` (ordering-free, sorted) are printed, not gated.
//!
//! The integral fingerprint in the JSON (sum of `h`, `E_nuc`) is re-checked to
//! rounding, so a drift in ferric's integrals fails here rather than as a
//! CASSCF difference.
//!
//! # Measured (2026-11-11)
//!
//! Tables in `crates/ferric-ci/theory.md` ("Two-step CASSCF").

mod casscf_common;

use casscf_common::{
    f64s, hessian_min_eig, maxabs, molecule_text, num, pyscf_reference, setup_text, EPS,
};
use ferric_ci::casscf::{CasScfConfig, CasScfProblem};
use ferric_ci::run_casscf;
use ferric_integrals::oneelectron::hcore;

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(f64::total_cmp);
    v
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn case(name: &str) {
    let r = pyscf_reference(name);
    assert!(
        r["converged"].as_bool() == Some(true),
        "{name}: reference not converged"
    );
    let xyz = r["xyz"].as_str().unwrap().to_string();
    if let Some(src) = r["xyz_source"].as_str().filter(|s| *s != "inline") {
        assert_eq!(xyz, molecule_text(src), "{name}: {src} changed");
    }
    let basis_name = r["basis"].as_str().unwrap();
    let (mol, prep, rhf) = setup_text(&xyz, basis_name);

    let fp = &r["integral_fingerprint"];
    let h = hcore(&prep);
    let hsum: f64 = h.sum();
    let habs: f64 = h.iter().map(|x| x.abs()).sum();
    let d_h = (hsum - num(fp, "hcore_sum")).abs();
    let d_nuc = (mol.nuclear_repulsion() - num(fp, "e_nuc")).abs();
    assert!(
        d_h <= 8.0 * EPS * habs && d_nuc <= 8.0 * EPS * mol.nuclear_repulsion(),
        "{name}: integrals moved since the reference (sum h {d_h:e}, E_nuc {d_nuc:e})"
    );

    let nel = r["n_elec_active"].as_array().unwrap();
    let cfg = CasScfConfig {
        n_core: r["n_core"].as_u64().unwrap() as usize,
        n_active: r["n_active"].as_u64().unwrap() as usize,
        n_elec_active: (
            nel[0].as_u64().unwrap() as usize,
            nel[1].as_u64().unwrap() as usize,
        ),
        canonicalize: true,
        ..Default::default()
    };
    let res = run_casscf(&mol, &prep, &rhf, &cfg).unwrap();
    assert!(
        res.converged,
        "{name}: not converged in {} iterations",
        res.iterations
    );
    let problem = CasScfProblem::new(&mol, &prep, res.mo_coeff.ncols(), &cfg).unwrap();
    let lam = hessian_min_eig(&problem, &res.mo_coeff, 1e-4);
    assert!(
        lam > 0.0,
        "{name}: lowest orbital Hessian eigenvalue {lam:e}"
    );

    let e_ref = num(&r, "e_total");
    let g_ref = 2.0 * num(&r, "grad_norm_pyscf");
    let conv_tol = num(&r["settings"], "mc_conv_tol");
    let round = 4.0 * EPS * res.e_total.abs() * (res.n_determinants as f64).sqrt();
    let bar = (res.grad_norm.powi(2) + g_ref.powi(2)) / (2.0 * lam) + conv_tol + 2.0 * round;
    let de = res.e_total - e_ref;

    let (nc, na) = (cfg.n_core, cfg.n_active);
    let occ = sorted(res.rdms.dm1.diag().to_vec());
    let occ_ref = sorted(f64s(&r, "natural_occupations"));
    let eps = res.mo_energy.as_ref().unwrap();
    let core = sorted(eps[..nc].to_vec());
    let virt = sorted(eps[nc + na..].to_vec());
    let d_occ = max_diff(&occ, &occ_ref);
    let d_core = max_diff(&core, &sorted(f64s(&r, "core_fock_eigenvalues")));
    let d_virt = max_diff(&virt, &sorted(f64s(&r, "virtual_fock_eigenvalues")));
    let offdiag = {
        let d = &res.rdms.dm1;
        let mut m = 0.0f64;
        for t in 0..na {
            for u in 0..na {
                if t != u {
                    m = m.max(d[(t, u)].abs());
                }
            }
        }
        m
    };
    eprintln!(
        "{name}: E {:.13} vs PySCF {e_ref:.13}: dE {de:.2e} (bar {bar:.2e}); |g| {:.1e} \
         (PySCF x2 {g_ref:.1e}); lambda_min {lam:.4}; iterations {}; CI residual {:.1e}\n  \
         natural occupations {:?} (max diff {d_occ:.1e}, max |dm1 off-diag| {:.1e})\n  \
         max diff core Fock eigenvalues {d_core:.1e}, virtual {d_virt:.1e}",
        res.e_total,
        res.grad_norm,
        res.iterations,
        res.ci_residual_norm,
        occ.iter()
            .rev()
            .map(|x| (x * 1e6).round() / 1e6)
            .collect::<Vec<_>>(),
        offdiag
    );
    assert!(maxabs(&res.rdms.dm1) > 0.0);
    assert!(de.abs() <= bar, "{name}: |dE| {:e} > bar {bar:e}", de.abs());
}

#[test]
fn h2o_cas44_631g_matches_pyscf() {
    case("h2o_cas44_631g");
}

#[test]
fn n2_cas66_631g_matches_pyscf() {
    case("n2_cas66_631g");
}

#[test]
fn n2_stretched_cas66_631g_matches_pyscf() {
    case("n2_r1.6_cas66_631g");
}

#[test]
fn o2_triplet_cas86_631g_matches_pyscf() {
    case("o2_triplet_cas86_631g");
}
