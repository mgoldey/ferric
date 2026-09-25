//! Exactness anchor for `pdep_dynamic_polarizability_truncated`.
//!
//! The truncated path projects ε̃(ω) = I + B̃ diag(4g(ω)) B̃ᵀ onto the PDEP
//! dressed eigenvectors U. When span(U) ⊇ range(B̃) (`trunc_thresh = 0`, and
//! every mode with λ(0) ≠ 1 kept) the projection is exact, so its per-atom
//! α^A(iω) must equal the full naux×naux SMW path (`pdep_dynamic_polarizability`,
//! same Becke partition, same quadrature) to roundoff at EVERY frequency.
//!
//! Artifact hypothesis: building the projected dielectric as
//! I + 4·(UᵀB̃ diag(g))(UᵀB̃ diag(g))ᵀ = I + UᵀB̃ diag(4g²) B̃ᵀU (the pre-fix
//! code) instead of diag(4g) is off by a factor g = Δε/(ω²+Δε²) per
//! transition, i.e. O(1) at ω = 0 — this test fails on it. The old spike
//! test (pdep_c6_truncation.rs, #[ignore]) only bounded C6 to 20%.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::dispersion::{
    pdep_dynamic_polarizability, pdep_dynamic_polarizability_truncated, DispersionPartition,
};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

#[test]
fn truncated_pdep_polarizability_equals_full_smw_at_full_rank() {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let obs_bs = basis::bundled("sto-3g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let cfg = PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 0.0,
        eigensolver_conv_thresh: 1e-10,
        ..Default::default()
    };
    let rpa = run_pdep_rpa(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap();
    let part = DispersionPartition::Becke;
    let trunc = pdep_dynamic_polarizability_truncated(
        &rpa, &mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg, part,
    )
    .unwrap();
    let full = pdep_dynamic_polarizability(&mol, &obs, &obs_bs, &dfbs, &rhf, op, &cfg, part, None)
        .unwrap();

    assert_eq!(trunc.freqs.len(), full.freqs.len());
    for (a, b) in trunc.freqs.iter().zip(full.freqs.iter()) {
        assert!(
            (a - b).abs() <= 1e-14 * b.abs().max(1.0),
            "quadrature grids differ"
        );
    }
    let natoms = mol.atoms.len();
    let mut max_rel = 0.0_f64;
    let mut scale = 0.0_f64;
    for at in 0..natoms {
        for k in 0..full.freqs.len() {
            for i in 0..3 {
                for j in 0..3 {
                    scale = scale.max(full.per_atom[at][k][i][j].abs());
                }
            }
        }
    }
    for at in 0..natoms {
        for k in 0..full.freqs.len() {
            for i in 0..3 {
                for j in 0..3 {
                    let d = (trunc.per_atom[at][k][i][j] - full.per_atom[at][k][i][j]).abs();
                    max_rel = max_rel.max(d / scale);
                }
            }
        }
    }
    eprintln!(
        "M={} of naux={}: max|α_trunc − α_full| / max|α_full| = {max_rel:.3e} (scale {scale:.4})",
        rpa.n_eigenpotentials,
        dfbs.nbasis()
    );
    assert!(
        max_rel <= 1e-8,
        "truncated ≠ full SMW at full rank: rel {max_rel:.3e}"
    );
}
