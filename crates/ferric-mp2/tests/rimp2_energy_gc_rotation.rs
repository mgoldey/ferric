//! The RI-MP2 energy's raw `(P|μν)` build on column-rotated shells
//! (`rimp2::rotated_three_index_basis`, `RiMp2Config::gc_rotation`).
//!
//! On one converged RHF (water / cc-pVDZ, cc-pVDZ-RI):
//!   * `Auto` and `Rot1` agree with `Off` within 1e-10 Eh in the correlation
//!     energy (the rotation is exact; what is left is integral round-off);
//!   * the rotation really ran: `rotated_three_index_basis` returns a basis
//!     for cc-pVDZ, and the two energies are not bit-identical;
//!   * a segmented basis (def2-SVP) has nothing to rotate: `None`, and the
//!     `Auto` energy is bit-identical to `Off`.
//!
//! A wrong fold (`C` instead of `TᵀC`, or `T` where `Tᵀ` belongs) moves the
//! energy by O(|T_offdiag|) ~ 1e-2 Eh, far outside the 1e-10 bar.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2, rotated_three_index_basis, RiMp2Config};
use ferric_scf::direct_jk::GcRotate;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const WATER: &str = "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n";

/// E_corr of each `modes` entry on one shared RHF, plus whether the basis rotates.
fn corr(basis_name: &str, modes: &[GcRotate]) -> (Vec<f64>, bool) {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(basis_name).unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let c = RhfConfig::default()
        .with_energy_conv(1e-11)
        .with_density_conv(1e-9);
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &c).unwrap();
    assert!(rhf.converged);
    let probe = Array2::<f64>::eye(obs.nbasis());
    let rotates = rotated_three_index_basis(&obs, Some(GcRotate::Auto), &[&probe])
        .unwrap()
        .is_some();
    let e = modes
        .iter()
        .map(|&m| {
            let cfg = RiMp2Config {
                frozen_core: 1,
                gc_rotation: Some(m),
                ..Default::default()
            };
            ri_mp2(&mol, &obs, &dfbs, op, &rhf, &cfg).unwrap().mp2_corr
        })
        .collect();
    (e, rotates)
}

#[test]
fn rotated_rimp2_energy_matches_unrotated_cc_pvdz() {
    let (e, rotates) = corr("cc-pvdz", &[GcRotate::Off, GcRotate::Auto, GcRotate::Rot1]);
    assert!(
        rotates,
        "cc-pVDZ must rotate (generally contracted s shells)"
    );
    let (off, auto, rot1) = (e[0], e[1], e[2]);
    println!(
        "E_corr off {off:.15} auto-off {:.3e} rot1-off {:.3e}",
        auto - off,
        rot1 - off
    );
    assert!((auto - off).abs() < 1e-10, "auto {auto} vs off {off}");
    assert!((rot1 - off).abs() < 1e-10, "rot1 {rot1} vs off {off}");
    assert_ne!(
        auto.to_bits(),
        off.to_bits(),
        "the rotated build did not run"
    );
}

#[test]
fn segmented_basis_is_untouched() {
    let (e, rotates) = corr("def2-svp", &[GcRotate::Off, GcRotate::Auto]);
    assert!(!rotates, "def2-SVP has nothing to rotate");
    assert_eq!(e[0].to_bits(), e[1].to_bits());
}
