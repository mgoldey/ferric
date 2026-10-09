//! Stage-1 PBC step 6: ferric-vs-ferric molecular limit (independent of
//! PySCF pbc — the check the adversarial review asked for, FINDINGS
//! "Iteration 1").
//!
//! H2/STO-3G in cubic boxes a = 16, 20, 24 Bohr with `exxdiv = ewald`
//! (Gamma-point `solve_rhf_injected`: `PeriodicHcore` + pure-AFT dense J/K)
//! minus ferric's own MOLECULAR `solve_rhf` on the same geometry and basis.
//! With the Madelung correction the residual is the exchange Makov–Payne
//! term: `E_box − E_mol = c3/a³ + c5/a⁵ + …`, `c3 = −(4π/3) σ²`, σ² the second
//! central moment of the occupied orbital density (a unit charge).
//! Prototype measurement: fitted −10.0282 vs predicted −10.028245 (6e-6
//! relative); asserted here at 1%.
//!
//! Artifact hypotheses: a G = 0 / Madelung convention error adds an
//! O(1/a) term (exxdiv=None's residual is v_M ≈ 2.837/a, i.e. +0.12 at
//! a = 24, ~1e4 × the a⁻³ term), which a c3/c5 fit cannot absorb and which
//! destroys the 1% agreement; an hcore/ERI truncation error would appear as
//! an a-independent offset, which likewise breaks the fitted c3. The
//! predicted coefficient comes from a molecular property (σ²) that shares
//! no code with the periodic path.
//!
//! Mutation plan: drop the Madelung term (residual ≈ +v_M, fit fails);
//! drop `v_g0` in `PeriodicHcore` (offset −π Z_tot S/(ω²Ω) ∝ a⁻³·const,
//! changes c3 by O(1)); use the prototype-unconverged a ≤ 12 boxes instead
//! (pre-onset, exponent 3.4–5: fit fails — do not "fix" by shrinking boxes).
//!
//! Cost (estimate, not measured): the pure-AFT G sphere grows as a³
//! (~2e6 half-sphere G at a = 24, precision 1e-11) but only the L = 0 pair
//! image survives at a ≥ 16, so each box is a few seconds at the test
//! profile's opt-level 2.

mod common;

use common::*;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::f64::consts::PI;

/// Truncation target for the big boxes: energies need ≪ 1% of the smallest
/// residual (7e-4 Ha at a = 24), so 1e-11 per term is ample.
const PRECISION: f64 = 1e-11;
const OMEGA: f64 = 0.9;

#[test]
fn box_residual_is_the_makov_payne_exchange_term() {
    let basis = pyscf_sto3g_h();

    // --- ferric molecular reference + σ² of the occupied orbital.
    let mol = hydrogens(&H2_ATOMS);
    let prep_mol = ferric_integrals::basis_bridge::PreparedBasis::new(&mol, &basis).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep_mol).unwrap();
    let cfg = RhfConfig {
        density_conv: 1e-10,
        ..Default::default()
    };
    let mr = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &prep_mol,
        op,
        &bounds,
        &cfg,
    )
    .unwrap();
    assert!(mr.converged);
    let e_mol = mr.energy;
    let c = mr.mos_r().column(0).to_owned();
    let dip = oneelectron::dipole(&prep_mol, [0.0; 3]).unwrap();
    let cen = [
        c.dot(&dip[0].dot(&c)),
        c.dot(&dip[1].dot(&c)),
        c.dot(&dip[2].dot(&c)),
    ];
    let r2 = oneelectron::r2_moment(&prep_mol, cen).unwrap();
    let sigma2 = c.dot(&r2.dot(&c));
    let c3_pred = -4.0 * PI / 3.0 * sigma2;
    eprintln!(
        "E_mol = {e_mol:.12}, centroid {cen:?}, sigma^2 = {sigma2:.10}, c3_pred = {c3_pred:.8}"
    );
    assert!(
        sigma2 > 1.0 && sigma2 < 5.0,
        "sigma^2 {sigma2} (prototype 2.394)"
    );

    // --- periodic boxes.
    let edges = [16.0, 20.0, 24.0];
    let mut de = Vec::new();
    for &a in &edges {
        let t0 = std::time::Instant::now();
        let cell = h2_cell(a);
        let prep = prep_for(&cell, &basis);
        let hcfg = PeriodicHcoreConfig {
            precision: PRECISION,
            ..PeriodicHcoreConfig::with_omega(OMEGA)
        };
        let hc = periodic_hcore(&cell, &prep, &hcfg).unwrap();
        let eri = DenseAftEri::build(
            &cell,
            &prep,
            &hc.s,
            ExxDiv::Ewald,
            PRECISION,
            DEFAULT_DENSE_AFT_MAX_BYTES,
        )
        .unwrap();
        let r = gamma_rhf(&cell, &prep, &hc, &eri);
        let d = r.energy - e_mol;
        eprintln!(
            "a = {a:4.1}: E = {:.12}, E - E_mol = {d:+.10e}, (4pi/3)sigma^2/a^3 = {:+.10e}, \
             v_M = {:.10}, half-G {} ({:.1} s)",
            r.energy,
            c3_pred / (a * a * a),
            eri.madelung(),
            eri.n_g_half(),
            t0.elapsed().as_secs_f64()
        );
        de.push(d);
    }
    for w in 0..2 {
        let p = -(de[w + 1] / de[w]).abs().ln() / (edges[w + 1] / edges[w]).ln();
        eprintln!("local exponent {}->{}: {p:.4}", edges[w], edges[w + 1]);
    }

    // Least squares dE = c3 x + c5 y, x = a^-3, y = a^-5 (3 points, 2 params).
    let (mut sxx, mut sxy, mut syy, mut sxd, mut syd) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&a, &d) in edges.iter().zip(&de) {
        let (x, y) = (a.powi(-3), a.powi(-5));
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
        sxd += x * d;
        syd += y * d;
    }
    let det = sxx * syy - sxy * sxy;
    let c3 = (sxd * syy - syd * sxy) / det;
    let c5 = (sxx * syd - sxy * sxd) / det;
    let rel = (c3 - c3_pred).abs() / c3_pred.abs();
    eprintln!("fit: c3 = {c3:.6}, c5 = {c5:.4}; predicted c3 = {c3_pred:.6} (rel {rel:.2e})");
    // All residuals negative and shrinking (ewald converges from below).
    assert!(de.iter().all(|d| *d < 0.0), "{de:?}");
    assert!(rel < 0.01, "c3 {c3} vs predicted {c3_pred} (rel {rel:.3e})");
}
