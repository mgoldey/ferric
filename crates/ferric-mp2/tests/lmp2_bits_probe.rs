//! Bit-identity probe for the amplitude-threshold LMP2 drivers: prints the
//! correlation energy BITS plus the solver/pattern/integral counters of a
//! fixed matrix of configurations, one `BITS` line each. It asserts only
//! that every run converges; it is the EVIDENCE harness for refactors that
//! claim "energies byte-identical" (PR #601 Stages A/B): run it on the base
//! and on the change and `diff` the `BITS` lines.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-mp2 --release --test lmp2_bits_probe \
//!     -- --ignored --nocapture 2>&1 | grep '^BITS' > bits.txt
//! ```
//!
//! Matrix: the original 20 (C4/C8 × Coulomb/erfc(ω=1) × ε 1e-4/1e-3 ×
//! batch_merge 1/4 (k=4 also sets virt_schwarz_kappa = 1) on the direct path,
//! gate on, plus the C4 global path), extended with ε = 0 and ε = 1e-5, the
//! pair gate OFF (the user default), and virt_schwarz_kappa on its own.
//! Ignored by default: C8 × 30-odd solves is minutes, not a unit test.
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::lmp2_amplitude::{amplitude_lmp2, AmplitudeLmp2Config};
use ferric_mp2::lmp2_direct::{amplitude_lmp2_direct, DirectConfig, Eri3Pass};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// FERRIC_LMP2_PROBE_PASS: unset or `batches` is the shipped pass, `aux-major`
/// runs every direct row on the aux-major pass (which forces k = 1; the row keeps
/// its label so the lines diff against a batch-pass run).
fn probe_pass() -> Eri3Pass {
    match std::env::var("FERRIC_LMP2_PROBE_PASS").as_deref() {
        Err(_) | Ok("batches") => Eri3Pass::Batches,
        Ok("aux-major") => Eri3Pass::AuxMajor,
        Ok(other) => panic!("FERRIC_LMP2_PROBE_PASS: unknown pass {other:?}"),
    }
}

fn direct(k: usize, kappa: Option<f64>) -> DirectConfig {
    let eri3_pass = probe_pass();
    let k = if eri3_pass == Eri3Pass::AuxMajor {
        1
    } else {
        k
    };
    DirectConfig {
        eri3_pass,
        aux_radius_bohr: 10.0,
        virt_radius_bohr: Some(12.0),
        ao_tail: 1e-3,
        schwarz_skip: 1e-5,
        batch_merge: k,
        virt_schwarz_kappa: kappa,
        ..Default::default()
    }
}

/// (eps, gate on?, batch_merge, virt_schwarz_kappa, also run the global path?)
type Row = (f64, bool, usize, Option<f64>, bool);

fn rows(c4: bool) -> Vec<Row> {
    let mut r = Vec::new();
    // the original 20: gate on, k=1 plain / k=4 with kappa, global at C4
    for eps in [1e-4, 1e-3] {
        r.push((eps, true, 1, None, c4));
        r.push((eps, true, 4, Some(1.0), false));
    }
    // extensions: exact limit and the tight end, gate OFF (user default)
    for eps in [0.0, 1e-5] {
        r.push((eps, false, 1, None, c4));
    }
    // gate off at a production eps, and virt_schwarz_kappa without merging
    r.push((1e-4, false, 1, None, c4));
    r.push((1e-4, false, 1, Some(1.0), false));
    r
}

#[test]
#[ignore = "evidence harness (minutes); run with --ignored --nocapture"]
fn lmp2_bits_probe() {
    for (xyz, fc) in [("alkane_4.xyz", 4usize), ("alkane_8.xyz", 8)] {
        let c4 = xyz == "alkane_4.xyz";
        let mol = Molecule::load_xyz(&format!(
            "{}/../../testdata/molecules/{xyz}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let obs_bs = basis::bundled("6-31g").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
        let rhf = solve_rhf(
            &ferric_core::parallel::ParallelContext::default(),
            &mol,
            &obs,
            Operator::coulomb(),
            &bounds,
            &RhfConfig {
                energy_conv: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        for (opn, op, cal) in [
            ("coul", Operator::coulomb(), 0.7),
            ("erfc", Operator::erfc(1.0), 0.02),
        ] {
            for (eps, gate, k, kappa, global) in rows(c4) {
                let cfg = AmplitudeLmp2Config {
                    eps,
                    frozen_core: fc,
                    pair_gate_cal: gate.then_some(cal),
                    // the historical tolerance, so a run stays comparable to
                    // bases older than the adaptive default
                    cg_rtol: Some(1e-11),
                    ..Default::default()
                };
                let tag = format!(
                    "{xyz} {opn} eps={eps:e} gate={} k={k} kappa={kappa:?}",
                    u8::from(gate)
                );
                let (r, st) = amplitude_lmp2_direct(
                    &mol,
                    &obs,
                    &obs_bs,
                    &dfbs,
                    op,
                    &rhf,
                    &cfg,
                    &direct(k, kappa),
                )
                .unwrap();
                assert!(r.cg_converged, "{tag}");
                println!(
                    "BITS {tag} direct E={:016x} e={:.15e} it={} kept={:016x} trip={} skip={} metric={}",
                    r.e_corr.to_bits(),
                    r.e_corr,
                    r.cg_iterations,
                    r.keep_fraction.to_bits(),
                    st.n_eri3_shell_triples,
                    st.n_eri3_skipped,
                    st.n_metric_shell_pairs
                );
                if global {
                    let g = amplitude_lmp2(&mol, &obs, &obs_bs, &dfbs, op, &rhf, &cfg).unwrap();
                    assert!(g.cg_converged, "{tag} global");
                    println!(
                        "BITS {tag} global E={:016x} e={:.15e} it={} kept={:016x}",
                        g.e_corr.to_bits(),
                        g.e_corr,
                        g.cg_iterations,
                        g.keep_fraction.to_bits()
                    );
                }
            }
        }
    }
}
