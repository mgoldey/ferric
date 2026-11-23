//! RI-MP2 on a spin-free X2C-1e reference (`RhfConfig::relativistic`).
//!
//! RI-MP2 reads the reference's orbitals and orbital energies and never
//! rebuilds the core Hamiltonian, so on an X2C reference it is the MP2 of the
//! X2C Hamiltonian. Checked against PySCF `DFMP2(sfx2c1e RHF)` with the same
//! cc-pVDZ / cc-pVDZ-RI basis sets, all electrons correlated
//! (`testdata/reference/x2c_scf_sfx2c1e.json`,
//! `scripts/gen_pyscf_x2c_scf_refs.py`). Bar: the nonrelativistic
//! disagreement of the same two codes on the same system (measured here; it
//! carries the shared fitting, integral and orbital-convergence floors) plus
//! the two SCF energy convergences. The X2C change of the correlation energy
//! is asserted to exceed that bar, so the comparison can see a reference that
//! is not the X2C one. The total is `E_SCF + E_corr` of the X2C SCF.
//!
//! The paths that rebuild `h` from the reference (orbital-optimized MP2) or
//! differentiate it (the RI-MP2 gradient) refuse an X2C reference.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::libint_max_deriv_order;
use ferric_integrals::operator::Operator;
use ferric_mp2::gradient::rimp2_gradient_analytical;
use ferric_mp2::oo_rimp2::{oo_ri_mp2, OoRiMp2Config};
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::x2c::Relativistic;

const ENERGY_CONV: f64 = 1e-10;

fn reference() -> serde_json::Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/reference/x2c_scf_sfx2c1e.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn rimp2_on_an_x2c_reference_matches_pyscf_dfmp2() {
    if libint_max_deriv_order() < 2 {
        eprintln!("SKIP: libint2 generated without second derivatives (no pVp)");
        return;
    }
    let refs = reference();
    let pyscf_conv = refs["conv_tol"].as_f64().unwrap();
    let r = &refs["systems"]["HBr"];
    let p = &r["bases"]["cc-pvdz"]["mp2"];
    let xyz = r["xyz_angstrom"].as_str().unwrap();
    let n = xyz.lines().count();
    let mol = Molecule::parse_xyz(&format!("{n}\n\n{xyz}\n"), 0, 1).unwrap();
    let obs_bs = basis::bundled("cc-pvdz").unwrap();
    let aux_bs = basis::bundled("cc-pvdz-ri").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &aux_bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let nr_cfg = RhfConfig::default()
        .with_energy_conv(ENERGY_CONV)
        .with_density_conv(1e-8);
    let mut x2c_cfg = nr_cfg.clone();
    x2c_cfg.relativistic = Some(Relativistic::X2c1e);
    let mp2 = RiMp2Config::default();
    let nr = solve_rhf(&ctx, &mol, &obs, op, &bounds, &nr_cfg).unwrap();
    let rel = solve_rhf(&ctx, &mol, &obs, op, &bounds, &x2c_cfg).unwrap();
    assert!(nr.converged && rel.converged);
    let c_nr = ri_mp2(&mol, &obs, &dfbs, op, &nr, &mp2).unwrap();
    let c_rel = ri_mp2(&mol, &obs, &dfbs, op, &rel, &mp2).unwrap();
    let d_nr = (c_nr.mp2_corr - p["e_corr_nr"].as_f64().unwrap()).abs();
    let d_x2c = (c_rel.mp2_corr - p["e_corr_x2c"].as_f64().unwrap()).abs();
    let bar = d_nr + 2.0 * (ENERGY_CONV + pyscf_conv);
    let shift = (c_rel.mp2_corr - c_nr.mp2_corr).abs();
    eprintln!(
        "HBr/cc-pVDZ RI-MP2: corr nr {:.10} x2c {:.10} |d_nr| {d_nr:.2e} |d_x2c| {d_x2c:.2e} \
         bar {bar:.2e} X2C change {shift:.2e}",
        c_nr.mp2_corr, c_rel.mp2_corr
    );
    assert!(d_x2c <= bar, "|E_corr(X2C) - PySCF| {d_x2c:e} > {bar:e}");
    assert!(
        shift > 10.0 * bar,
        "X2C change of E_corr {shift:e} is within the bar"
    );
    assert_eq!(
        c_rel.total_energy.to_bits(),
        (rel.energy + c_rel.mp2_corr).to_bits(),
        "total is E_SCF(X2C) + E_corr"
    );

    // Refusals on the X2C reference; the nonrelativistic one is accepted.
    let msg = "relativistic = \"x2c1e\" has no";
    let e = rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rel, &mp2, None)
        .unwrap_err()
        .to_string();
    assert!(e.contains(msg), "{e}");
    let e = oo_ri_mp2(
        &mol,
        &obs,
        &dfbs,
        op,
        &bounds,
        &rel,
        &OoRiMp2Config::default(),
        None,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains(msg), "{e}");
}
