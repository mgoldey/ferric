//! Spin-summed CAS-CI 1- and 2-RDMs (`ferric_ci::rdm`).
//!
//! Systems (`casci_rdm_common::SYSTEMS`, the windows of
//! `validation_casci.rs`): N2 r = 1.10 Å / cc-pVDZ CAS(6,6) above 4 inactive
//! orbitals (400 determinants) and H2O / 6-31G CAS(4,4) above 3 (36), on
//! ferric's exact-J/K RHF orbitals, Davidson residual below
//! [`CONV_THRESH`] = 1e-10.
//!
//! # Exactness anchor (asserted first)
//!
//! `E_core + Σ h dm1 + ½ Σ (pq|rs) dm2` is algebraically `⟨c|H|c⟩` for the
//! normalized CI vector `c`, so it must equal
//!
//! * the Rayleigh quotient `c·(Hc)` from the direct sigma, to rounding:
//!   `|d| ≤ 4 ε A √(N_det + n⁴)`, with `A` the sum of absolute energy terms
//!   ([`ferric_ci::rdm::RdmEnergy::abs_terms`]) — a random-walk bound on the
//!   `N_det`-long and `n⁴`-long sums;
//! * the Davidson eigenvalue θ, to that plus `‖r‖ < CONV_THRESH`: the Ritz
//!   vector `x` and its residual `r = Hx − θx` give `xᵀHx − θ = xᵀr`, and
//!   `|xᵀr| ≤ ‖r‖` for `‖x‖ = 1`.
//!
//! # Physics hypothesis vs artifact hypothesis
//!
//! * If the RDMs are right: the anchor holds at ~1e-12, the identities below
//!   hold to rounding, and the RDMs equal PySCF `make_rdm12` of PySCF's own
//!   ground state on the same integrals to the Davidson error bound.
//! * A wrong phase in one channel changes off-diagonal elements: the anchor
//!   and the PySCF comparison fail; traces may still pass.
//! * A missing α-β term: `Σ dm2[p,p,r,r]` becomes `Nα(Nα−1) + Nβ(Nβ−1)`
//!   instead of `N(N−1)`, and the anchor misses by the α-β Coulomb energy.
//! * A factor of two on a diagonal: the traces fail.
//! * A transposed index convention (`dm2[p,r,q,s]`): the anchor fails (the
//!   exchange and Coulomb integrals swap); a `p↔q` transposition alone is
//!   invisible to the anchor for real orbitals and is pinned only by the
//!   PySCF comparison, which uses PySCF's index convention.
//!
//! # PySCF comparison
//!
//! `tests/data/casci_rdm/<system>.json` is written by
//! `scripts/gen_casci_rdm_refs.py` from `examples/casci_rdm_dump.rs`: PySCF
//! diagonalizes the dense Hamiltonian over all determinants
//! (`fci.direct_spin1.pspace`) on ferric's active integrals and calls
//! `fci.direct_spin1.make_rdm12` on that eigenvector. PySCF never supplies
//! orbitals. The generator also records `make_rdm12` of ferric's own CI
//! vector against ferric's RDMs (`construction_max_diff`, a check of the
//! construction alone at rounding level), re-asserted here.
//!
//! The reference RDMs are in the orbitals of the dump run. Orbital signs and
//! rotations among degenerate orbitals (the N2 π pairs) can differ between
//! runs, so the reference is rotated into this run's orbitals with
//! `U = C_refᵀ S C_now` (`n × n`, orthogonal when the active spaces agree).
//! The CAS-CI state is invariant under active-active rotations, so the
//! rotated reference is the reference in this run's orbitals.
//!
//! Tolerance: `‖c − c*‖ ≤ ‖r‖ / Δ` to first order (`Δ` = gap to the next
//! root of the same Ms, recorded by the generator), and an expectation value
//! of `O` moves by at most `2 ‖O‖ ‖c − c*‖`, with `‖E_pq‖ ≤ N` and
//! `‖E_pq E_rs‖ ≤ N²`. Bars: `2 N CONV_THRESH / Δ` (dm1) and
//! `2 N² CONV_THRESH / Δ` (dm2), plus `N² max|UᵀU − I|` for orbitals that
//! differ between the runs.
//!
//! A missing reference JSON is a hard failure (panic naming the path).
//!
//! # Measured (2026-11-11)
//!
//! | quantity | N2 CAS(6,6) | H2O CAS(4,4) |
//! |---|---:|---:|
//! | \|E_rdm − c·Hc\| | 1.3e-13 (bar 4.6e-12) | 1.4e-14 (bar 1.3e-12) |
//! | \|E_rdm − θ\| | 1.4e-14 (bar 1.05e-10) | 2.8e-14 (bar 1.01e-10) |
//! | max \|dm1 − PySCF\| | 4.6e-13 (bar 4.0e-9) | 4.9e-15 (bar 2.4e-9) |
//! | max \|dm2 − PySCF\| | 9.3e-12 (bar 2.4e-8) | 1.0e-14 (bar 9.7e-9) |
//! | generator construction check dm1 / dm2 | 6.7e-16 / 1.3e-15 | 2.2e-19 / 2.3e-16 |
//!
//! # Mutations (applied to `src/rdm.rs` one at a time, 2026-11-11)
//!
//! | mutation | anchor | identities | PySCF |
//! |---|---|---|---|
//! | β phase dropped (`ε C(Iα,T)` → `C(Iα,T)`) | fail | fail (dm1 symmetry) | fail |
//! | α-β cross terms dropped (`D^α D^αᵀ + D^β D^βᵀ`) | fail | fail (Σ dm2[p,p,r,r]) | fail |
//! | dm1 diagonal doubled | fail | fail (tr dm1) | fail |
//! | dm2[p,p,r,r] doubled | fail | fail (Σ dm2[p,p,r,r]) | fail |
//!
//! Each fails for both systems; `rdms_are_opt_in` passes under all four.

mod casci_rdm_common;

use casci_rdm_common::{run_system, system, workspace_root, Run, System, CONV_THRESH};
use ferric_ci::rdm::{rdm_energy, CasCiRdms};
use ferric_ci::sigma::DirectSigma;
use ndarray::{Array2, Array4};
use serde_json::Value;

const EPS: f64 = f64::EPSILON;

fn rdms(run: &Run) -> &CasCiRdms {
    run.res.rdms.as_ref().expect("compute_rdms was set")
}

fn n_elec(sys: &System) -> f64 {
    (sys.n_elec.0 + sys.n_elec.1) as f64
}

/// Rounding bar of an identity between RDM sums of size `scale`.
fn identity_tol(run: &Run, scale: f64) -> f64 {
    16.0 * EPS * scale.max(1.0) * (run.res.n_determinants as f64).sqrt()
}

fn anchor(name: &str) {
    let sys = system(name);
    let run = run_system(&sys);
    let r = rdms(&run);
    let e = rdm_energy(&run.res.integrals, r);
    let direct = DirectSigma::new(&run.res.integrals, &run.space);
    let c = &run.res.ci_vector;
    let hc = direct.apply(c);
    let e_rq: f64 = c.iter().zip(&hc).map(|(a, b)| a * b).sum();
    let n4 = (sys.n_active as f64).powi(4);
    let tol_round = 4.0 * EPS * e.abs_terms * (run.res.n_determinants as f64 + n4).sqrt();
    let d_rq = (e.e_total - e_rq).abs();
    let d_theta = (e.e_total - run.res.e_total).abs();
    eprintln!(
        "{name}: E_rdm {:.12} (one {:.10}, two {:.10}) c.Hc {e_rq:.12} theta {:.12} \
         |E_rdm-c.Hc| {d_rq:.2e} (bar {tol_round:.2e}) |E_rdm-theta| {d_theta:.2e} \
         (bar {:.2e})",
        e.e_total,
        e.e_one,
        e.e_two,
        run.res.e_total,
        tol_round + CONV_THRESH
    );
    assert!(
        d_rq <= tol_round,
        "{name}: RDM energy {} vs c.Hc {e_rq} (|d| {d_rq:.2e} > {tol_round:.2e})",
        e.e_total
    );
    assert!(
        d_theta <= tol_round + CONV_THRESH,
        "{name}: RDM energy {} vs Davidson eigenvalue {} (|d| {d_theta:.2e})",
        e.e_total,
        run.res.e_total
    );
}

fn identities(name: &str) {
    let sys = system(name);
    let run = run_system(&sys);
    let r = rdms(&run);
    let n = sys.n_active;
    let ne = n_elec(&sys);
    let tol = identity_tol(&run, ne * ne);

    let tr1: f64 = (0..n).map(|p| r.dm1[[p, p]]).sum();
    let tr2: f64 = (0..n)
        .flat_map(|p| (0..n).map(move |q| (p, q)))
        .map(|(p, q)| r.dm2[[p, p, q, q]])
        .sum();
    eprintln!(
        "{name}: tr dm1 {tr1:.14} (N {ne}) sum dm2[pprr] {tr2:.14} (N(N-1) {}) bar {tol:.1e}",
        ne * (ne - 1.0)
    );
    assert!((tr1 - ne).abs() <= tol, "{name}: tr dm1 = {tr1}, want {ne}");
    assert!(
        (tr2 - ne * (ne - 1.0)).abs() <= tol,
        "{name}: sum dm2[p,p,r,r] = {tr2}, want N(N-1) = {}",
        ne * (ne - 1.0)
    );

    let mut d_partial = 0.0f64;
    let mut d_sym1 = 0.0f64;
    let mut d_swap = 0.0f64;
    let mut d_herm = 0.0f64;
    for p in 0..n {
        for q in 0..n {
            d_sym1 = d_sym1.max((r.dm1[[p, q]] - r.dm1[[q, p]]).abs());
            let partial: f64 = (0..n).map(|s| r.dm2[[p, q, s, s]]).sum();
            d_partial = d_partial.max((partial - (ne - 1.0) * r.dm1[[p, q]]).abs());
            for s in 0..n {
                for t in 0..n {
                    let v = r.dm2[[p, q, s, t]];
                    d_swap = d_swap.max((v - r.dm2[[s, t, p, q]]).abs());
                    d_herm = d_herm.max((v - r.dm2[[q, p, t, s]]).abs());
                }
            }
        }
    }
    eprintln!(
        "{name}: max |sum_r dm2[p,q,r,r] - (N-1) dm1| {d_partial:.2e}, |dm1 - dm1^T| {d_sym1:.2e}, \
         |dm2[pqrs]-dm2[rspq]| {d_swap:.2e}, |dm2[pqrs]-dm2[qpsr]| {d_herm:.2e}"
    );
    assert!(d_partial <= tol, "{name}: partial trace off by {d_partial:.2e}");
    assert!(d_sym1 <= tol, "{name}: dm1 not symmetric ({d_sym1:.2e})");
    assert!(d_swap <= tol, "{name}: dm2 pair-swap symmetry off ({d_swap:.2e})");
    assert!(d_herm <= tol, "{name}: dm2 hermiticity off ({d_herm:.2e})");

    // Natural occupations lie in [0, 2].
    use ndarray_linalg::{Eigh, UPLO};
    let (occ, _) = r.dm1.eigh(UPLO::Upper).expect("eigh dm1");
    eprintln!("{name}: natural occupations {occ:.6}");
    for &o in occ.iter() {
        assert!(
            (-tol..=2.0 + tol).contains(&o),
            "{name}: natural occupation {o} outside [0, 2]"
        );
    }
}

fn reference(name: &str) -> Value {
    let path = workspace_root()
        .join("crates/ferric-ci/tests/data/casci_rdm")
        .join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing RDM reference {} ({e}); regenerate with scripts/gen_casci_rdm_refs.py",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: bad JSON: {e}", path.display()))
}

fn f64s(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("reference field {key} missing"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn num(v: &Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("reference field {key} missing"))
}

/// `out[a,b,..] = Σ U_ia U_jb .. in[i,j,..]`, one index at a time.
fn rotate2(m: &Array2<f64>, u: &Array2<f64>) -> Array2<f64> {
    u.t().dot(m).dot(u)
}

fn rotate4(m: &Array4<f64>, u: &Array2<f64>) -> Array4<f64> {
    let n = u.nrows();
    let mut cur = m.clone();
    for axis in 0..4 {
        let mut next = Array4::<f64>::zeros((n, n, n, n));
        for (idx, v) in next.indexed_iter_mut() {
            let mut acc = 0.0;
            for i in 0..n {
                let mut src = [idx.0, idx.1, idx.2, idx.3];
                let a = src[axis];
                src[axis] = i;
                acc += u[[i, a]] * cur[src];
            }
            *v = acc;
        }
        cur = next;
    }
    cur
}

fn vs_pyscf(name: &str) {
    let sys = system(name);
    let refj = reference(name);
    let run = run_system(&sys);
    let r = rdms(&run);
    let n = sys.n_active;
    let ne = n_elec(&sys);
    assert_eq!(refj["n_active"].as_u64(), Some(n as u64), "{name}: n_active");
    assert_eq!(
        refj["active_start"].as_u64(),
        Some(sys.active_start as u64),
        "{name}: active_start"
    );

    // Construction check recorded by the generator: PySCF make_rdm12 of
    // ferric's CI vector equals ferric's RDMs of that vector.
    let c1 = num(&refj, "construction_max_diff_dm1");
    let c2 = num(&refj, "construction_max_diff_dm2");
    eprintln!("{name}: generator construction check dm1 {c1:.2e} dm2 {c2:.2e}");
    assert!(c1 < 1e-12 && c2 < 1e-12, "{name}: construction check {c1:.2e}/{c2:.2e}");

    // Align the reference orbitals to this run's.
    let nbas = run.prep.nbasis();
    let c_ref = Array2::from_shape_vec((nbas, n), f64s(&refj, "c_act")).expect("c_act shape");
    let mos = run.rhf.mos_r();
    let c_now = mos.slice(ndarray::s![.., sys.active_start..sys.active_start + n]);
    let s = ferric_integrals::oneelectron::overlap(&run.prep);
    let u = c_ref.t().dot(&s).dot(&c_now);
    let utu = u.t().dot(&u);
    let d_orth = utu
        .indexed_iter()
        .map(|((i, j), v)| (v - if i == j { 1.0 } else { 0.0 }).abs())
        .fold(0.0f64, f64::max);
    eprintln!("{name}: max |U^T U - I| {d_orth:.2e}");
    assert!(
        d_orth < 1e-8,
        "{name}: the reference active space differs from this run's ({d_orth:.2e})"
    );

    let dm1_ref = Array2::from_shape_vec((n, n), f64s(&refj, "dm1")).expect("dm1");
    let dm2_ref = Array4::from_shape_vec((n, n, n, n), f64s(&refj, "dm2")).expect("dm2");
    let dm1_ref = rotate2(&dm1_ref, &u);
    let dm2_ref = rotate4(&dm2_ref, &u);

    let gap = num(&refj, "gap");
    let tol1 = 2.0 * ne * CONV_THRESH / gap + ne * ne * d_orth;
    let tol2 = 2.0 * ne * ne * CONV_THRESH / gap + ne * ne * d_orth;
    let d1 = (&r.dm1 - &dm1_ref).iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let d2 = (&r.dm2 - &dm2_ref).iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let e_ref = num(&refj, "e_total");
    eprintln!(
        "{name}: vs PySCF max|d dm1| {d1:.2e} (bar {tol1:.2e}) max|d dm2| {d2:.2e} (bar {tol2:.2e}) \
         gap {gap:.4} |dE| {:.2e}",
        (run.res.e_total - e_ref).abs()
    );
    assert!(d1 <= tol1, "{name}: dm1 vs PySCF {d1:.2e} > {tol1:.2e}");
    assert!(d2 <= tol2, "{name}: dm2 vs PySCF {d2:.2e} > {tol2:.2e}");
}

#[test]
fn rdm_energy_equals_davidson_eigenvalue_n2_cas66() {
    anchor("n2_r1.10");
}

#[test]
fn rdm_energy_equals_davidson_eigenvalue_h2o_cas44() {
    anchor("h2o");
}

#[test]
fn rdm_trace_and_symmetry_identities_n2_cas66() {
    identities("n2_r1.10");
}

#[test]
fn rdm_trace_and_symmetry_identities_h2o_cas44() {
    identities("h2o");
}

#[test]
fn rdms_match_pyscf_make_rdm12_n2_cas66() {
    vs_pyscf("n2_r1.10");
}

#[test]
fn rdms_match_pyscf_make_rdm12_h2o_cas44() {
    vs_pyscf("h2o");
}

/// RDMs are off by default and cost nothing then.
#[test]
fn rdms_are_opt_in() {
    let sys = system("h2o");
    let run = run_system(&sys);
    assert!(run.res.rdms.is_some());
    let cfg = ferric_ci::CasCiConfig {
        n_active: sys.n_active,
        n_elec_active: sys.n_elec,
        active_start: sys.active_start,
        ..Default::default()
    };
    let res = ferric_ci::run_cas_ci(&run.mol, &run.prep, &run.rhf, cfg).unwrap();
    assert!(res.rdms.is_none());
}
