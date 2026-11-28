//! Exact-bits refactor pin for the amplitude-threshold local family
//! (LinLCCD global + direct, Full tier; dRPA global + direct; LMP2 global):
//! water/6-31G, cc-pVDZ-RI, ε = 1e-4. Each correlation energy must equal,
//! `to_bits`, the value recorded on bff7369c5 before the shared-seams
//! refactor (ragged rhs seam, Fock injection, `ferric_mp2::ladders`,
//! `whitened_oo_vv`, the plain-Boys config constructor). A refactor that
//! claims "bit-identical" runs this before and after.
//!
//! Ignored by default: the recorded bits are those of ONE machine and BLAS
//! build (this box, OpenBLAS, single-threaded); another CPU or OpenBLAS
//! kernel may differ in the last bits with no defect. Run with
//! `OPENBLAS_NUM_THREADS=1 cargo test -p ferric-cc --test local_family_bits -- --ignored --nocapture`
//! and, on a new machine, re-record from the base commit first.
use ferric_cc::linlccd::LadderVariant;
use ferric_cc::linlccd_amplitude::{
    amplitude_linlccd, amplitude_linlccd_direct, AmplitudeLinLccdConfig,
};
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::drpa_amplitude::{amplitude_drpa, amplitude_drpa_direct, AmplitudeDrpaConfig};
use ferric_mp2::lmp2_amplitude::{amplitude_lmp2, AmplitudeLmp2Config, CgEnergyTarget};
use ferric_mp2::lmp2_direct::DirectConfig;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// (name, recorded `f64::to_bits` of e_corr) on bff7369c5.
const RECORDED: &[(&str, u64)] = &[
    ("linlccd_full_global", 0xbfb7a654d8e216d1),
    ("linlccd_full_direct", 0xbfb7a654d8e217d2),
    ("drpa_global", 0xbfc1b883d8a37379),
    ("drpa_direct", 0xbfc1b883d8a37381),
    ("lmp2_global", 0xbfc07edf9fc536e2),
];

#[test]
#[ignore = "machine-specific bits: refactor pin, run with --ignored (see module doc)"]
fn local_family_energies_are_bit_identical_to_the_recorded_base() {
    let mol = Molecule::load_xyz(&format!(
        "{}/../../testdata/molecules/water.xyz",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let obs_bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ferric_core::parallel::ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig::default().with_energy_conv(1e-10),
    )
    .unwrap();
    let lcfg = AmplitudeLinLccdConfig::default();
    let dcfg = DirectConfig::default();
    let full = LadderVariant::Full;
    let got: Vec<(&str, f64)> = vec![
        (
            "linlccd_full_global",
            amplitude_linlccd(&mol, &obs, &obs_bs, &dfbs, op, &rhf, &lcfg, full)
                .unwrap()
                .e_corr,
        ),
        (
            "linlccd_full_direct",
            amplitude_linlccd_direct(&mol, &obs, &obs_bs, &dfbs, op, &rhf, &lcfg, &dcfg, full)
                .unwrap()
                .0
                .e_corr,
        ),
        (
            "drpa_global",
            amplitude_drpa(
                &mol,
                &obs,
                &obs_bs,
                &dfbs,
                op,
                &rhf,
                &AmplitudeDrpaConfig::default(),
            )
            .unwrap()
            .e_corr,
        ),
        (
            "drpa_direct",
            amplitude_drpa_direct(
                &mol,
                &obs,
                &obs_bs,
                &dfbs,
                op,
                &rhf,
                &AmplitudeDrpaConfig::default(),
                &dcfg,
            )
            .unwrap()
            .0
            .e_corr,
        ),
        (
            "lmp2_global",
            amplitude_lmp2(
                &mol,
                &obs,
                &obs_bs,
                &dfbs,
                op,
                &rhf,
                // the stop of the recorded run; the default is now the
                // 1e-7 Ha energy target, which moves e_corr by < 1e-7 Ha
                &AmplitudeLmp2Config {
                    cg_energy_target: CgEnergyTarget::Strict,
                    ..AmplitudeLmp2Config::default()
                },
            )
            .unwrap()
            .e_corr,
        ),
    ];
    let mut bad = Vec::new();
    for ((name, e), (rname, bits)) in got.iter().zip(RECORDED) {
        assert_eq!(name, rname);
        println!("BITS {name} {:#018x} {e:.15}", e.to_bits());
        if e.to_bits() != *bits {
            bad.push(format!(
                "{name}: {:#018x} != recorded {bits:#018x}",
                e.to_bits()
            ));
        }
    }
    assert!(bad.is_empty(), "bits changed:\n{}", bad.join("\n"));
}
