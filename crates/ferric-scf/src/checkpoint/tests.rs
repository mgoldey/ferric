//! Claims this format must hold. The mutants these were seen to fail against are
//! listed in the PR that introduced them. Kills cannot test the fsyncs: a killed
//! process's writes survive in the page cache, so only power loss would differ.

use super::format::{self, Members, WriteStage};
use super::identity::ProcessSettings;
use super::*;
use crate::diis::DiisFlavor;
use crate::result::{ConvergenceReport, DfJkRoute, DiisFallback, GradientVetoes, ScfExit, Spin};
use crate::rhf::RhfConfig;
use ferric_core::basis::{bundled, BasisSet};
use ferric_core::external_potential::{ExternalPotential, PointCharge, SmearedCharge};
use ferric_core::gpu::provenance::PrecisionRecord;
use ferric_core::mol::Molecule;
use ferric_integrals::operator::{Operator, OperatorKind};
use ndarray::{Array1, Array2, Axis, ShapeBuilder};
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

const WATER: &str = "3\nwater\nO 0 0 0.117\nH 0 0.757 -0.470\nH 0 -0.757 -0.470\n";

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

/// A private directory per test, removed at the end.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let n = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("ferric-ckpt-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn file(&self) -> PathBuf {
        self.0.join("state.npz")
    }
    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.0)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn clean_build() -> BuildStamp {
    BuildStamp {
        version: "0.0.0.dev1".to_string(),
        commit: Some("a".repeat(40)),
        dirty: Some(false),
    }
}

/// Everything an identity is computed from.
#[derive(Clone)]
struct Scenario {
    mol: Molecule,
    basis: BasisSet,
    spin: Spin,
    op: Operator,
    config: RhfConfig,
    process: ProcessSettings,
}

/// Fixed, so these tests do not depend on the environment they run in.
fn fixed_process() -> ProcessSettings {
    ProcessSettings {
        eri_precision: 1e-20,
        ecp_backend: "quadrature".to_string(),
        lindep_thresh: 1e-6,
        precision: "F64 MixedKernelSet(0)".to_string(),
        gpu_mode: "Off".to_string(),
        libint_version: "2.7.2".to_string(),
        cargo_features: "mpi=false gpu=false cfmm-incomplete=false".to_string(),
        env: vec![None; 8],
    }
}

impl Scenario {
    fn water(spin: Spin) -> Self {
        Self {
            mol: Molecule::parse_xyz(WATER, 0, 1).unwrap(),
            basis: bundled("sto-3g").unwrap(),
            spin,
            op: Operator::coulomb(),
            config: RhfConfig::default(),
            process: fixed_process(),
        }
    }
    fn identity(&self) -> ScfIdentity {
        ScfIdentity::with_process(
            &self.mol,
            &self.basis,
            self.spin,
            &self.op,
            &self.config,
            &self.process,
        )
        .unwrap()
    }
}

/// `tag + sqrt(i + 1) / (j + 2)`: every value distinct, none a short decimal, and
/// correctly rounded in IEEE arithmetic, so the Python test recomputes it exactly.
fn value(tag: f64, i: usize, j: usize) -> f64 {
    tag + ((i + 1) as f64).sqrt() / ((j + 2) as f64)
}

fn c_order(tag: f64, n: usize, m: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, m), |(i, j)| value(tag, i, j))
}

fn f_order(tag: f64, n: usize, m: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, m).f(), |(i, j)| value(tag, i, j))
}

fn eps(tag: f64, n: usize) -> Vec<f64> {
    (0..n).map(|k| tag - ((k + 1) as f64) / 3.0).collect()
}

const NMO: usize = 6;

fn orbitals(spin: Spin, nao: usize) -> OrbitalState {
    let open = spin != Spin::Restricted;
    OrbitalState {
        spin,
        density_total: c_order(1.0, nao, nao),
        density_alpha: f_order(2.0, nao, nao),
        density_beta: open.then(|| c_order(3.0, nao, nao)),
        // nmo < nao, as in a linearly dependent basis.
        mos_alpha: f_order(4.0, nao, NMO),
        mos_beta: open.then(|| f_order(5.0, nao, NMO)),
        eps_alpha: eps(6.0, NMO),
        eps_beta: open.then(|| eps(7.0, NMO)),
    }
}

fn route() -> DfJkRoute {
    DfJkRoute {
        j_aux: Some("def2-universal-jkfit".to_string()),
        k_aux: None,
        rsh_k: Some(("cc-pvdz-jkfit".to_string(), 0.3)),
        op: Operator::erfc(0.4),
        budget_bytes: 123_456_789,
    }
}

fn report() -> ConvergenceReport {
    ConvergenceReport {
        iteration: 12,
        de: 1e-9,
        // The report legitimately holds infinities, which JSON numbers cannot.
        dp_rms: f64::INFINITY,
        dp_max: 3e-7,
        err_max: 4e-6,
        energy_conv: 1e-3,
        density_conv: 1e-6,
        de_noise_floor: Some(2e-5),
        diis_fallback: Some(DiisFallback {
            iteration: 3,
            err_max: 0.5,
            bound: 0.25,
            flavor: "ADIIS",
        }),
        gradient_vetoes: Some(GradientVetoes {
            count: 2,
            last_iteration: 9,
            last_orbital_gradient: 1e-3,
            bound: 1e-4,
        }),
    }
}

fn final_state(spin: Spin, nao: usize) -> FinalState {
    let open = spin != Spin::Restricted;
    FinalState {
        energy: -74.962_991_23,
        one_electron_energy: -122.5,
        nuclear_repulsion: 9.25,
        fock_alpha: c_order(8.0, nao, nao),
        fock_beta: open.then(|| f_order(9.0, nao, nao)),
        rohf_spin_focks: (spin == Spin::RestrictedOpen)
            .then(|| (c_order(10.0, nao, nao), f_order(11.0, nao, nao))),
        df_jk: Some(route()),
        exit: ScfExit::Converged,
        converged: true,
        iterations: 13,
        computed_quartets: 4242,
        convergence: Some(report()),
        precision_summary: "f64".to_string(),
        precision_detail: format!("{:?}", PrecisionRecord::f64()),
        not_stored: Vec::new(),
    }
}

fn final_ck(spin: Spin) -> Checkpoint {
    let identity = Scenario::water(spin).identity();
    let nao = identity.nao;
    Checkpoint {
        build: clean_build(),
        identity,
        orbitals: orbitals(spin, nao),
        final_state: Some(final_state(spin, nao)),
        iteration: 13,
        file_hash: None,
    }
}

const ALL_SPINS: [Spin; 3] = [Spin::Restricted, Spin::Unrestricted, Spin::RestrictedOpen];

/// Values by bit pattern, and the layout in memory.
fn same_array(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.shape() == b.shape()
        && a.is_standard_layout() == b.is_standard_layout()
        && a.as_slice_memory_order().map(same_slice_bits_owned)
            == b.as_slice_memory_order().map(same_slice_bits_owned)
}

fn same_slice_bits_owned(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn same_opt(a: &Option<Array2<f64>>, b: &Option<Array2<f64>>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => same_array(x, y),
        (None, None) => true,
        _ => false,
    }
}

fn same_vec(a: &[f64], b: &[f64]) -> bool {
    same_slice_bits_owned(a) == same_slice_bits_owned(b)
}

fn same_final(a: &FinalState, b: &FinalState) -> bool {
    let same_spin_focks = match (&a.rohf_spin_focks, &b.rohf_spin_focks) {
        (Some((x, y)), Some((u, v))) => same_array(x, u) && same_array(y, v),
        (None, None) => true,
        _ => false,
    };
    a.energy.to_bits() == b.energy.to_bits()
        && a.one_electron_energy.to_bits() == b.one_electron_energy.to_bits()
        && a.nuclear_repulsion.to_bits() == b.nuclear_repulsion.to_bits()
        && same_array(&a.fock_alpha, &b.fock_alpha)
        && same_opt(&a.fock_beta, &b.fock_beta)
        && same_spin_focks
        && a.df_jk == b.df_jk
        && a.exit == b.exit
        && a.converged == b.converged
        && a.iterations == b.iterations
        && a.computed_quartets == b.computed_quartets
        && same_report(&a.convergence, &b.convergence)
        && a.precision_summary == b.precision_summary
        && a.precision_detail == b.precision_detail
        && a.not_stored == b.not_stored
}

/// `ConvergenceReport` by `PartialEq` would pass `-0.0 == 0.0`; its `Debug` does not.
fn same_report(a: &Option<ConvergenceReport>, b: &Option<ConvergenceReport>) -> bool {
    format!("{a:?}") == format!("{b:?}")
}

fn same_state(a: &Checkpoint, b: &Checkpoint) -> bool {
    let (x, y) = (&a.orbitals, &b.orbitals);
    let finals = match (&a.final_state, &b.final_state) {
        (Some(s), Some(t)) => same_final(s, t),
        (None, None) => true,
        _ => false,
    };
    x.spin == y.spin
        && same_array(&x.density_total, &y.density_total)
        && same_array(&x.density_alpha, &y.density_alpha)
        && same_opt(&x.density_beta, &y.density_beta)
        && same_array(&x.mos_alpha, &y.mos_alpha)
        && same_opt(&x.mos_beta, &y.mos_beta)
        && same_vec(&x.eps_alpha, &y.eps_alpha)
        && x.eps_beta.as_deref().map(same_slice_bits_owned)
            == y.eps_beta.as_deref().map(same_slice_bits_owned)
        && a.iteration == b.iteration
        && finals
}

fn same_checkpoint(a: &Checkpoint, b: &Checkpoint) -> bool {
    same_state(a, b) && a.identity == b.identity && a.build == b.build
}

fn round_trip(ck: &Checkpoint) -> Checkpoint {
    let dir = Scratch::new();
    let hash = ck.write(&dir.file()).unwrap();
    let back = Checkpoint::read(&dir.file()).unwrap();
    assert_eq!(back.file_hash, Some(hash));
    back
}

// ---------------------------------------------------------------- round trip

#[test]
fn final_and_mid_run_files_round_trip_bitwise_for_every_spin_kind() {
    for spin in ALL_SPINS {
        let ck = final_ck(spin);
        assert!(same_checkpoint(&ck, &round_trip(&ck)), "final {spin:?}");
        let mut mid = final_ck(spin);
        mid.final_state = None;
        mid.iteration = 5;
        let back = round_trip(&mid);
        assert!(same_checkpoint(&mid, &back), "mid-run {spin:?}");
        assert!(back.final_state.is_none());
    }
}

#[test]
fn memory_layout_survives_in_both_orders() {
    let mut ck = final_ck(Spin::Restricted);
    assert!(ck.orbitals.density_total.is_standard_layout());
    assert!(!ck.orbitals.density_alpha.is_standard_layout());
    ck.orbitals.mos_alpha = f_order(4.0, 7, NMO);
    let back = round_trip(&ck);
    assert!(back.orbitals.density_total.is_standard_layout());
    assert!(!back.orbitals.density_alpha.is_standard_layout());
    assert!(!back.orbitals.mos_alpha.is_standard_layout());
    // Flat indexing, which is what layout decides, sees the same numbers.
    assert_eq!(
        same_slice_bits_owned(back.orbitals.density_alpha.as_slice_memory_order().unwrap()),
        same_slice_bits_owned(ck.orbitals.density_alpha.as_slice_memory_order().unwrap()),
    );
}

#[test]
fn an_eigensolver_output_round_trips_in_the_layout_it_came_in() {
    use ndarray_linalg::{Eigh, UPLO};
    let symmetric = Array2::from_shape_fn((7, 7), |(i, j)| value(0.0, i.min(j), i.max(j)));
    let (w, v) = symmetric.eigh(UPLO::Lower).unwrap();
    let mut ck = final_ck(Spin::Restricted);
    ck.orbitals.eps_alpha = w.to_vec();
    ck.orbitals.mos_alpha = v;
    let back = round_trip(&ck);
    assert!(same_array(&back.orbitals.mos_alpha, &ck.orbitals.mos_alpha));
}

#[test]
fn a_non_contiguous_layout_keeps_its_values_and_verifies_on_read() {
    let mut ck = final_ck(Spin::Restricted);
    let mut reversed = c_order(1.0, 7, 7);
    reversed.invert_axis(Axis(0));
    let expected = reversed.clone();
    ck.orbitals.density_total = reversed;
    let back = round_trip(&ck);
    for (a, b) in back.orbitals.density_total.iter().zip(expected.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert!(back.orbitals.density_total.is_standard_layout());
}

#[test]
fn extreme_finite_values_round_trip_including_through_json() {
    let mut ck = final_ck(Spin::Restricted);
    let extremes = [
        -0.0,
        5e-324,
        f64::MAX,
        f64::MIN_POSITIVE,
        1.0 + f64::EPSILON,
        -1e-300,
    ];
    ck.orbitals.eps_alpha = extremes.to_vec();
    let state = ck.final_state.as_mut().unwrap();
    state.energy = -0.0;
    state.one_electron_energy = 5e-324;
    state.nuclear_repulsion = f64::MAX;
    ck.identity.coords_bohr[0] = [-0.0, 5e-324, 1.0 + f64::EPSILON];
    let back = round_trip(&ck);
    assert!(same_checkpoint(&ck, &back));
}

#[test]
fn df_jk_route_is_never_confused_with_exact() {
    let mut exact = final_ck(Spin::Restricted);
    exact.final_state.as_mut().unwrap().df_jk = None;
    assert_eq!(round_trip(&exact).final_state.unwrap().df_jk, None);
    for op in [
        Operator::coulomb(),
        Operator::erf(0.33),
        Operator::erfc(0.42),
        Operator::terfc(1.05),
        Operator::terf_with_omega(1.05, 0.7),
        Operator::yukawa(0.5),
        Operator::slater_geminal(1.5),
        Operator::stg(1.2, OperatorKind::CgtgCoulomb),
        Operator::stg(1.2, OperatorKind::Delcgtg2),
        Operator::terfc_fit(1.05),
    ] {
        let mut ck = final_ck(Spin::Restricted);
        let wanted = DfJkRoute { op, ..route() };
        ck.final_state.as_mut().unwrap().df_jk = Some(wanted.clone());
        let back = round_trip(&ck).final_state.unwrap().df_jk.unwrap();
        assert_eq!(back, wanted);
        assert_eq!(back.op.is_composite(), op.is_composite());
    }
}

#[test]
fn every_exit_state_round_trips() {
    for exit in [
        ScfExit::Converged,
        ScfExit::Plateau,
        ScfExit::Stalled,
        ScfExit::Diverged,
        ScfExit::MaxIter,
        ScfExit::NotCertified,
    ] {
        let mut ck = final_ck(Spin::Unrestricted);
        ck.final_state.as_mut().unwrap().exit = exit;
        assert_eq!(round_trip(&ck).final_state.unwrap().exit, exit);
    }
}

fn scf_result(spin: Spin) -> crate::result::ScfResult {
    let o = orbitals(spin, 7);
    let s = final_state(spin, 7);
    crate::result::ScfResult {
        spin,
        energy: s.energy,
        density_total: o.density_total,
        density_alpha: o.density_alpha,
        density_beta: o.density_beta,
        mos_alpha: o.mos_alpha,
        mos_beta: o.mos_beta,
        eps_alpha: o.eps_alpha,
        eps_beta: o.eps_beta,
        fock_alpha: s.fock_alpha,
        fock_beta: s.fock_beta,
        converged: s.converged,
        exit: s.exit,
        iterations: s.iterations,
        computed_quartets: s.computed_quartets,
        induced_dipoles: None,
        stability: None,
        stability_external: None,
        df_jk: s.df_jk,
        rohf_spin_focks: s.rohf_spin_focks,
        cosx_final: None,
        cosx_schedule: None,
        convergence: s.convergence,
        guess_basis: None,
        precision: PrecisionRecord::f64(),
    }
}

#[test]
fn from_result_carries_every_stored_field_to_disk() {
    for spin in ALL_SPINS {
        let result = scf_result(spin);
        let identity = Scenario::water(spin).identity();
        let ck = Checkpoint::from_result(&result, -122.5, 9.25, identity, clean_build()).unwrap();
        let back = round_trip(&ck);
        assert!(same_state(&final_ck(spin), &back), "{spin:?}");
        let state = back.final_state.unwrap();
        assert_eq!(state.df_jk, result.df_jk);
        assert_eq!(state.precision_summary, "f64");
    }
}

#[test]
fn from_result_records_what_it_could_not_store() {
    let mut result = scf_result(Spin::Restricted);
    result.induced_dipoles = Some(Array2::zeros((1, 3)));
    let identity = Scenario::water(Spin::Restricted).identity();
    let ck = Checkpoint::from_result(&result, 0.0, 0.0, identity, clean_build()).unwrap();
    assert_eq!(
        ck.final_state.unwrap().not_stored,
        vec!["induced_dipoles".to_string()]
    );
}

// --------------------------------------------------------- unusable files

fn is_format(e: &CheckpointError) -> bool {
    matches!(e, CheckpointError::Format(_))
}

#[test]
fn a_missing_file_is_an_error_with_no_way_to_read_it_as_absent() {
    let dir = Scratch::new();
    let err = Checkpoint::read(&dir.file()).unwrap_err();
    assert!(matches!(err, CheckpointError::Missing { .. }), "{err}");
    let err = Checkpoint::read(&dir.0).unwrap_err();
    assert!(matches!(err, CheckpointError::Io { .. }), "{err}");
}

#[test]
fn not_an_npz_is_a_format_error() {
    let dir = Scratch::new();
    for bytes in [&b""[..], &b"not a zip file"[..]] {
        std::fs::write(dir.file(), bytes).unwrap();
        let err = Checkpoint::read(&dir.file()).unwrap_err();
        assert!(is_format(&err), "{err}");
    }
}

/// Every single-byte corruption is either rejected or changes nothing that is
/// read. A few bytes (zip timestamps, say) carry no content, so "always rejected"
/// would be false; "never silently different" is the property that matters.
#[test]
fn no_single_byte_flip_produces_a_different_checkpoint() {
    let dir = Scratch::new();
    let ck = final_ck(Spin::Unrestricted);
    ck.write(&dir.file()).unwrap();
    let good = std::fs::read(dir.file()).unwrap();
    let (mut rejected, mut harmless) = (0_usize, 0_usize);
    for i in 0..good.len() {
        let mut bad = good.clone();
        bad[i] ^= 0xFF;
        std::fs::write(dir.file(), &bad).unwrap();
        match Checkpoint::read(&dir.file()) {
            Err(_) => rejected += 1,
            Ok(back) => {
                assert!(
                    same_checkpoint(&ck, &back),
                    "flip at byte {i} changed content"
                );
                harmless += 1;
            }
        }
    }
    eprintln!(
        "{} bytes: {rejected} rejected, {harmless} harmless",
        good.len()
    );
    // Reachability: the check is not vacuous, and the payload alone is far more
    // than the harmless count.
    assert!(
        rejected > good.len() * 9 / 10,
        "{rejected} of {}",
        good.len()
    );
}

#[test]
fn every_truncation_is_an_error() {
    let dir = Scratch::new();
    final_ck(Spin::Unrestricted).write(&dir.file()).unwrap();
    let good = std::fs::read(dir.file()).unwrap();
    for len in 0..good.len() {
        std::fs::write(dir.file(), &good[..len]).unwrap();
        assert!(Checkpoint::read(&dir.file()).is_err(), "kept {len} bytes");
    }
}

/// Encode `ck`, let `edit` change the members, and sign the result: what a reader
/// sees if the writer was wrong rather than the disk.
fn forged(ck: &Checkpoint, path: &Path, edit: impl FnOnce(&mut Members)) {
    let meta = format::encode_meta(ck).unwrap();
    let arrays = format::encode_arrays(ck);
    let mut members = Members {
        version: FORMAT_VERSION,
        checksum: 0,
        meta,
        arrays,
    };
    edit(&mut members);
    members.checksum = format::checksum(members.version, &members.meta, &members.arrays);
    format::write_atomic(path, &members, &mut |_| {}).unwrap();
}

#[test]
fn a_clean_forgery_reads_so_the_forgery_tests_below_mean_something() {
    let dir = Scratch::new();
    let ck = final_ck(Spin::Unrestricted);
    forged(&ck, &dir.file(), |_| {});
    assert!(same_checkpoint(
        &ck,
        &Checkpoint::read(&dir.file()).unwrap()
    ));
}

#[test]
fn wrong_version_is_named_with_or_without_a_good_checksum() {
    let dir = Scratch::new();
    let ck = final_ck(Spin::Restricted);
    for version in [0, 2, u64::MAX] {
        forged(&ck, &dir.file(), |m| m.version = version);
        let err = Checkpoint::read(&dir.file()).unwrap_err();
        assert_eq!(err, CheckpointError::Version { found: version });
    }
}

#[test]
fn wrong_checksum_is_an_error() {
    let dir = Scratch::new();
    let ck = final_ck(Spin::Restricted);
    let meta = format::encode_meta(&ck).unwrap();
    let arrays = format::encode_arrays(&ck);
    let right = format::checksum(FORMAT_VERSION, &meta, &arrays);
    let members = Members {
        version: FORMAT_VERSION,
        checksum: right ^ 1,
        meta,
        arrays,
    };
    format::write_atomic(&dir.file(), &members, &mut |_| {}).unwrap();
    let err = Checkpoint::read(&dir.file()).unwrap_err();
    assert_eq!(
        err,
        CheckpointError::Checksum {
            stored: right ^ 1,
            computed: right
        }
    );
}

#[test]
fn non_finite_values_are_refused_on_write_and_on_read() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let dir = Scratch::new();
        let mut ck = final_ck(Spin::Restricted);
        ck.orbitals.density_total[[2, 3]] = bad;
        let err = ck.write(&dir.file()).unwrap_err();
        assert!(matches!(err, CheckpointError::NonFinite { .. }), "{err}");
        assert!(
            !dir.file().exists(),
            "a refused write must not create a file"
        );

        let good = final_ck(Spin::Restricted);
        forged(&good, &dir.file(), |m| {
            m.arrays.get_mut("fock_alpha").unwrap()[[1, 1]] = bad;
        });
        let err = Checkpoint::read(&dir.file()).unwrap_err();
        assert_eq!(
            err,
            CheckpointError::NonFinite {
                array: "fock_alpha".to_string()
            }
        );
    }
    let mut ck = final_ck(Spin::Restricted);
    ck.final_state.as_mut().unwrap().energy = f64::NAN;
    assert!(ck.validate().is_err());
    let mut ck = final_ck(Spin::Restricted);
    ck.identity.coords_bohr[1][2] = f64::NAN;
    assert!(ck.validate().is_err());
}

#[test]
fn a_wrong_shape_is_named_for_each_array() {
    let dir = Scratch::new();
    let ck = final_ck(Spin::RestrictedOpen);
    for name in [
        "density_total",
        "density_alpha",
        "density_beta",
        "mos_alpha",
        "mos_beta",
        "fock_alpha",
        "fock_beta",
        "rohf_spin_fock_alpha",
        "rohf_spin_fock_beta",
    ] {
        forged(&ck, &dir.file(), |m| {
            let wrong = c_order(1.0, 3, 3).into_dyn();
            m.arrays.insert(name.to_string(), wrong);
        });
        let err = Checkpoint::read(&dir.file()).unwrap_err();
        assert!(
            matches!(&err, CheckpointError::Shape { array, .. } if array == name),
            "{name}: {err}"
        );
    }
    // eps must match the MO count, not merely be a vector.
    forged(&ck, &dir.file(), |m| {
        m.arrays.insert(
            "eps_alpha".to_string(),
            Array1::from(vec![1.0; NMO + 1]).into_dyn(),
        );
    });
    let err = Checkpoint::read(&dir.file()).unwrap_err();
    assert!(matches!(&err, CheckpointError::Shape { array, .. } if array == "eps_alpha"));
    // A 1-D array where a matrix belongs.
    forged(&ck, &dir.file(), |m| {
        m.arrays.insert(
            "mos_alpha".to_string(),
            Array1::from(vec![1.0; 7]).into_dyn(),
        );
    });
    let err = Checkpoint::read(&dir.file()).unwrap_err();
    assert!(matches!(&err, CheckpointError::Shape { array, .. } if array == "mos_alpha"));
}

#[test]
fn missing_and_unexpected_members_are_errors() {
    let dir = Scratch::new();
    let ck = final_ck(Spin::Unrestricted);
    for name in [
        "density_total",
        "density_alpha",
        "mos_alpha",
        "eps_alpha",
        "fock_alpha",
    ] {
        forged(&ck, &dir.file(), |m| {
            m.arrays.remove(name);
        });
        let err = Checkpoint::read(&dir.file()).unwrap_err();
        assert!(is_format(&err) && err.to_string().contains(name), "{err}");
    }
    forged(&ck, &dir.file(), |m| {
        m.arrays
            .insert("surprise".to_string(), c_order(1.0, 2, 2).into_dyn());
    });
    let err = Checkpoint::read(&dir.file()).unwrap_err();
    assert!(
        is_format(&err) && err.to_string().contains("surprise"),
        "{err}"
    );
    // Beta data on a restricted state, and beta missing on an open-shell one.
    let restricted = final_ck(Spin::Restricted);
    forged(&restricted, &dir.file(), |m| {
        m.arrays
            .insert("density_beta".to_string(), c_order(1.0, 7, 7).into_dyn());
    });
    assert!(is_format(&Checkpoint::read(&dir.file()).unwrap_err()));
    forged(&ck, &dir.file(), |m| {
        m.arrays.remove("density_beta");
    });
    assert!(is_format(&Checkpoint::read(&dir.file()).unwrap_err()));
}

#[test]
fn bad_metadata_is_an_error() {
    let dir = Scratch::new();
    let ck = final_ck(Spin::Restricted);
    let edits: [fn(&str) -> String; 5] = [
        |_| "not json".to_string(),
        |t| t.replacen("\"spin\":", "\"surprise\":1,\"spin\":", 1),
        |t| t.replacen("\"energy\":", "\"energy\":null,\"x\":", 1),
        |t| t.replacen("\"exit\":\"Converged\"", "\"exit\":\"Done\"", 1),
        |t| t.replacen("\"class\":\"Structure\"", "\"class\":\"Other\"", 1),
    ];
    for edit in edits {
        forged(&ck, &dir.file(), |m| {
            let text = String::from_utf8(m.meta.clone()).unwrap();
            let changed = edit(&text);
            assert_ne!(changed, text, "the edit must apply");
            m.meta = changed.into_bytes();
        });
        let err = Checkpoint::read(&dir.file()).unwrap_err();
        assert!(is_format(&err), "{err}");
    }
}

// ------------------------------------------------------------------ identity

const CLASSES_SKIP: ReadMode = ReadMode::Skip {
    allow_other_build: false,
};

type Mutator = fn(&mut Scenario);

/// One row per compared field: changing it must produce exactly that field's
/// mismatch. `every_identity_field_has_a_row` keeps this complete.
fn identity_table() -> Vec<(&'static str, FieldClass, Mutator)> {
    use FieldClass::{Environment, Method, Route, StateSelecting, Structure};
    vec![
        ("atoms", Structure, |s| s.mol.atoms[1].ghost = true),
        ("charge", Structure, |s| s.mol.charge = 1),
        ("multiplicity", Structure, |s| s.mol.multiplicity = 3),
        ("spin", Structure, |s| s.spin = Spin::Unrestricted),
        ("basis[Z=1]", Structure, |s| {
            s.basis.shells.get_mut(&1).unwrap()[0].exponents[0] *= 1.0 + f64::EPSILON;
        }),
        ("basis[Z=8]", Structure, |s| {
            s.basis.shells.get_mut(&8).unwrap()[0].coefficients[0] *= 1.0 + f64::EPSILON;
        }),
        ("operator", Method, |s| s.op = Operator::erfc(0.4)),
        ("precision", Route, |s| {
            s.process.precision = "Mixed".to_string();
        }),
        ("eri_precision", Route, |s| s.process.eri_precision = 1e-14),
        ("ecp_backend", Route, |s| {
            s.process.ecp_backend = "libecpint".to_string();
        }),
        ("lindep_thresh", Method, |s| s.process.lindep_thresh = 3e-3),
        ("gpu_mode", Route, |s| {
            s.process.gpu_mode = "Auto".to_string()
        }),
        ("libint_version", Route, |s| {
            s.process.libint_version = "2.13.1".to_string();
        }),
        ("cargo_features", Route, |s| {
            s.process.cargo_features = "mpi=true".to_string();
        }),
        ("FERRIC_SCF_INCREMENTAL", Route, |s| {
            s.process.env[0] = Some("0".to_string());
        }),
        ("FERRIC_SCF_UHF_INCREMENTAL", Route, |s| {
            s.process.env[1] = Some("1".to_string());
        }),
        ("FERRIC_SCF_COMBINED_JK", Route, |s| {
            s.process.env[2] = Some("0".to_string());
        }),
        ("FERRIC_SHELLPAIR_CACHE", Route, |s| {
            s.process.env[3] = Some("0".to_string());
        }),
        ("FERRIC_DFK_FORCE_DENSITY", Route, |s| {
            s.process.env[4] = Some("1".to_string());
        }),
        ("FERRIC_JK_STORAGE", Route, |s| {
            s.process.env[5] = Some("disk".to_string());
        }),
        ("FERRIC_TERF_TABLE_DIR", Route, |s| {
            s.process.env[6] = Some("/elsewhere".to_string());
        }),
        ("FERRIC_DF_GUESS_TIGHT", StateSelecting, |s| {
            s.process.env[7] = Some("0".to_string());
        }),
        ("use_sad_guess", StateSelecting, |s| {
            s.config.use_sad_guess = false;
        }),
        ("init_guess_density", StateSelecting, |s| {
            s.config.init_guess_density = Some(Array2::zeros((7, 7)));
        }),
        ("guess_basis", StateSelecting, |s| {
            s.config.guess_basis = Some(crate::basis2::GuessBasis {
                basis: bundled("sto-3g").unwrap(),
                projection: crate::basis2::GuessProjection::Fock,
            });
        }),
        ("trah_trigger", StateSelecting, |s| {
            s.config.trah_trigger = Some(0.1);
        }),
        ("trah", StateSelecting, |s| s.config.trah.radius0 = 0.11),
        ("newton_trigger", StateSelecting, |s| {
            s.config.newton_trigger = 1e-2;
        }),
        ("ah_trigger", StateSelecting, |s| s.config.ah_trigger = 1e-3),
        ("aurora", StateSelecting, |s| s.config.aurora.enabled = true),
        ("diis_flavor", StateSelecting, |s| {
            s.config.diis_flavor = DiisFlavor::Ediis;
        }),
        ("diis_size", StateSelecting, |s| s.config.diis_size = 3),
        ("diis_switch_thresh", StateSelecting, |s| {
            s.config.diis_switch_thresh = 0.7;
        }),
        ("level_shift", StateSelecting, |s| {
            s.config.level_shift = 0.2
        }),
        ("external_potential", Environment, |s| {
            s.config.external_potential = Some(ExternalPotential {
                field: Some([0.0, 0.0, 1e-3]),
                ..Default::default()
            });
        }),
        ("cosmo", Environment, |s| {
            s.config.cosmo = Some(Default::default())
        }),
        ("pcm", Environment, |s| {
            s.config.pcm = Some(Default::default())
        }),
        ("polarizable", Environment, |s| {
            s.config.polarizable = Some(Default::default());
        }),
        ("xc", Method, |s| s.config.xc = Some("PBE".to_string())),
        ("xc_omega", Method, |s| s.config.xc_omega = Some(0.3)),
        ("dft_grid", Method, |s| {
            s.config.dft_grid = Some(Default::default())
        }),
        ("nlc_grid", Method, |s| {
            s.config.nlc_grid = Some(Default::default())
        }),
        ("smearing_sigma", Method, |s| {
            s.config.smearing_sigma = Some(1e-3)
        }),
        ("symmetry", Method, |s| s.config.symmetry = true),
        ("irrep_nelec", Method, |s| {
            s.config.irrep_nelec = Some(vec![("A1".to_string(), 8)]);
        }),
        ("fractional_occ", Method, |s| s.config.fractional_occ = true),
        ("scf_stability_descent", Method, |s| {
            s.config.scf_stability_descent = true
        }),
        ("rohf_occupation_guard", Method, |s| {
            s.config.rohf_occupation_guard = false
        }),
        ("mom_after_iter", Method, |s| s.config.mom_after_iter = 5),
        ("constraints", Method, |s| {
            s.config.constraints = vec![ferric_dft::cdft::Constraint {
                fragment: vec![0],
                spin: ferric_dft::cdft::SpinChannel::Total,
                target: 1.0,
            }];
        }),
        ("cdft_lambda_tol", Method, |s| {
            s.config.cdft_lambda_tol = 1e-7
        }),
        ("cdft_lambda_init", Method, |s| {
            s.config.cdft_lambda_init = Some(vec![0.5])
        }),
        ("cdft_stability_descent", Method, |s| {
            s.config.cdft_stability_descent = false
        }),
        ("k_builder", Route, |s| {
            s.config.k_builder = Some("link".to_string())
        }),
        ("cosx", Route, |s| {
            s.config.cosx.overlap_fit = !s.config.cosx.overlap_fit
        }),
        ("df_j_aux", Route, |s| {
            s.config.df_j_aux = Some("cc-pvdz-jkfit".to_string())
        }),
        ("df_k_aux", Route, |s| {
            s.config.df_k_aux = Some("cc-pvdz-jkfit".to_string())
        }),
        ("integral_thresh", Route, |s| {
            s.config.integral_thresh = 1e-11
        }),
        ("screening", Route, |s| {
            s.config.screening = crate::screening::ScreeningKind::Csb;
        }),
    ]
}

/// The table above only moves `None` to `Some` for these, which would pass even
/// if the value were dropped and only its presence compared.
#[test]
fn a_change_between_two_values_is_seen_not_only_presence() {
    let mut base = Scenario::water(Spin::Restricted);
    base.config.xc = Some("PBE".to_string());
    base.config.k_builder = Some("link".to_string());
    base.config.df_j_aux = Some("def2-universal-jkfit".to_string());
    base.config.df_k_aux = Some("def2-universal-jkfit".to_string());
    base.config.xc_omega = Some(0.3);
    base.config.smearing_sigma = Some(1e-3);
    let table: [(&str, Mutator); 6] = [
        ("xc", |s| s.config.xc = Some("B3LYP".to_string())),
        ("k_builder", |s| {
            s.config.k_builder = Some("cosx".to_string())
        }),
        ("df_j_aux", |s| {
            s.config.df_j_aux = Some("cc-pvdz-jkfit".to_string())
        }),
        ("df_k_aux", |s| {
            s.config.df_k_aux = Some("cc-pvdz-jkfit".to_string())
        }),
        ("xc_omega", |s| {
            s.config.xc_omega = Some(0.30000000000000004)
        }),
        ("smearing_sigma", |s| s.config.smearing_sigma = Some(2e-3)),
    ];
    let mut file = final_ck(Spin::Restricted);
    file.identity = base.identity();
    for (name, mutate) in table {
        let mut changed = base.clone();
        mutate(&mut changed);
        let err = file
            .check(&changed.identity(), &clean_build(), CLASSES_SKIP)
            .unwrap_err();
        assert!(
            matches!(&err, CheckpointError::Identity(v) if v[0].name == name),
            "{err}"
        );
    }
}

fn is_checksum(e: &CheckpointError) -> bool {
    matches!(e, CheckpointError::Checksum { .. })
}

/// Change content but keep the original checksum, in a well-formed zip whose own
/// CRCs are right. The byte-flip test cannot show this: the zip CRC catches a
/// disk flip first, so only a file like this needs the checksum itself.
fn stale_checksum(ck: &Checkpoint, path: &Path, edit: impl FnOnce(&mut Members)) {
    let meta = format::encode_meta(ck).unwrap();
    let arrays = format::encode_arrays(ck);
    let checksum = format::checksum(FORMAT_VERSION, &meta, &arrays);
    let mut members = Members {
        version: FORMAT_VERSION,
        checksum,
        meta,
        arrays,
    };
    edit(&mut members);
    format::write_atomic(path, &members, &mut |_| {}).unwrap();
}

#[test]
fn the_checksum_covers_values_metadata_layout_and_member_names() {
    let dir = Scratch::new();
    let ck = final_ck(Spin::Unrestricted);

    stale_checksum(&ck, &dir.file(), |m| {
        let a = m.arrays.get_mut("mos_alpha").unwrap();
        a[[1, 1]] = f64::from_bits(a[[1, 1]].to_bits() + 1);
    });
    let err = Checkpoint::read(&dir.file()).unwrap_err();
    assert!(is_checksum(&err), "one ulp of a value: {err}");

    stale_checksum(&ck, &dir.file(), |m| {
        let text = String::from_utf8(m.meta.clone()).unwrap();
        let changed = text.replacen("\"iteration\":13", "\"iteration\":14", 1);
        assert_ne!(changed, text);
        m.meta = changed.into_bytes();
    });
    let err = Checkpoint::read(&dir.file()).unwrap_err();
    assert!(is_checksum(&err), "metadata: {err}");

    // The same logical matrix in the other memory order.
    stale_checksum(&ck, &dir.file(), |m| {
        let a = m.arrays.remove("density_total").unwrap();
        let (shape, values) = (
            a.shape().to_vec(),
            a.t().iter().copied().collect::<Vec<_>>(),
        );
        let f = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape).f(), values).unwrap();
        assert_eq!(f, a, "the swap must keep the logical matrix");
        m.arrays.insert("density_total".to_string(), f);
    });
    let err = Checkpoint::read(&dir.file()).unwrap_err();
    assert!(is_checksum(&err), "layout: {err}");

    // The same bytes under the other order flag: the transpose. Only the flag in
    // the checksum can see this, because the values hash identically.
    stale_checksum(&ck, &dir.file(), |m| {
        let a = m.arrays.remove("fock_alpha").unwrap();
        let values = a.as_slice_memory_order().unwrap().to_vec();
        let shape = a.shape().to_vec();
        let f = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape).f(), values).unwrap();
        assert_ne!(f, a, "the flag must change the logical matrix");
        m.arrays.insert("fock_alpha".to_string(), f);
    });
    let err = Checkpoint::read(&dir.file()).unwrap_err();
    assert!(is_checksum(&err), "layout flag alone: {err}");

    // Two same-shape arrays exchanged.
    stale_checksum(&ck, &dir.file(), |m| {
        let a = m.arrays.remove("density_alpha").unwrap();
        let b = m.arrays.remove("density_beta").unwrap();
        m.arrays.insert("density_alpha".to_string(), b);
        m.arrays.insert("density_beta".to_string(), a);
    });
    let err = Checkpoint::read(&dir.file()).unwrap_err();
    assert!(is_checksum(&err), "swapped members: {err}");
}

fn expect_guess(class: FieldClass, verdict: &ReadVerdict, name: &str) {
    let names = |v: &[Mismatch]| v.iter().map(|m| m.name.clone()).collect::<Vec<_>>();
    let want = vec![name.to_string()];
    match class {
        FieldClass::Environment
        | FieldClass::Method
        | FieldClass::Coordinates
        | FieldClass::StateSelecting => {
            assert_eq!(names(&verdict.warnings), want);
            assert!(verdict.logged.is_empty(), "{name}");
        }
        FieldClass::Route => {
            assert_eq!(names(&verdict.logged), want);
            assert!(verdict.warnings.is_empty(), "{name}");
        }
        FieldClass::Structure => unreachable!("a structure mismatch is an error under guess"),
    }
}

#[test]
fn each_identity_mismatch_is_one_named_error_or_the_documented_guess_outcome() {
    let base = Scenario::water(Spin::Restricted);
    let file = {
        let mut ck = final_ck(Spin::Restricted);
        ck.identity = base.identity();
        ck
    };
    let build = clean_build();
    for (name, class, mutate) in identity_table() {
        let mut changed = base.clone();
        mutate(&mut changed);
        let current = changed.identity();

        let err = file.check(&current, &build, CLASSES_SKIP).unwrap_err();
        let CheckpointError::Identity(found) = &err else {
            panic!("{name}: {err}");
        };
        assert_eq!(found.len(), 1, "{name}: {err}");
        assert_eq!(found[0].name, name);
        assert_eq!(found[0].class, class, "{name}");
        assert!(err.to_string().contains(name), "{err}");

        let guess = file.check(&current, &build, ReadMode::Guess);
        if class == FieldClass::Structure {
            let err = guess.unwrap_err();
            assert!(
                matches!(&err, CheckpointError::Identity(v) if v[0].name == name),
                "{err}"
            );
        } else {
            expect_guess(class, &guess.unwrap(), name);
        }
    }
    // The control: nothing changed, nothing reported, under both modes.
    let quiet = file.check(&base.identity(), &build, CLASSES_SKIP).unwrap();
    assert!(quiet.warnings.is_empty() && quiet.logged.is_empty());
    let quiet = file
        .check(&base.identity(), &build, ReadMode::Guess)
        .unwrap();
    assert!(quiet.warnings.is_empty() && quiet.logged.is_empty());
}

#[test]
fn every_identity_field_has_a_row() {
    let fields: BTreeSet<String> = Scenario::water(Spin::Restricted)
        .identity()
        .fields
        .iter()
        .map(|f| f.name.clone())
        .collect();
    let mut rows: BTreeSet<String> = identity_table()
        .iter()
        .map(|(name, _, _)| (*name).to_string())
        .collect();
    // Covered by its own test: it moves with the basis in any real change.
    rows.insert("nao".to_string());
    assert_eq!(fields, rows);
}

#[test]
fn nao_is_a_structure_field_and_follows_the_shell_type() {
    let mut s = Scenario::water(Spin::Restricted);
    s.basis = bundled("cc-pvdz").unwrap();
    let spherical = s.identity();
    for shells in s.basis.shells.values_mut() {
        for shell in shells.iter_mut() {
            shell.pure = false;
        }
    }
    let cartesian = s.identity();
    assert_eq!(spherical.nao, 24);
    assert_eq!(cartesian.nao, 25);
    let mut ck = final_ck(Spin::Restricted);
    ck.identity = spherical;
    let err = ck
        .check(&cartesian, &clean_build(), ReadMode::Guess)
        .unwrap_err();
    assert!(err.to_string().contains("nao"), "{err}");
    assert!(err.to_string().contains("basis[Z=8]"), "{err}");
}

#[test]
fn basis_identity_is_the_data_of_this_molecules_elements() {
    let base = Scenario::water(Spin::Restricted);
    let file = base.identity();

    // A name alone is not a difference.
    let mut renamed = base.clone();
    renamed.basis.name = "an alias".to_string();
    assert_eq!(renamed.identity().fields, file.fields);

    // Elements this molecule does not have cannot change it.
    let mut other_element = base.clone();
    other_element.basis.shells.get_mut(&6).unwrap()[0].exponents[0] *= 2.0;
    other_element.basis.shells.remove(&7);
    assert_eq!(other_element.identity().fields, file.fields);

    // Shell order is AO order.
    let mut reordered = base.clone();
    reordered.basis.shells.get_mut(&8).unwrap().swap(0, 1);
    assert_ne!(reordered.identity().fields, file.fields);

    // `pure` is data, even where the shell's function count does not show it.
    let mut pure = base.clone();
    let p_shell = pure.basis.shells.get_mut(&8).unwrap();
    let i = p_shell.iter().position(|s| s.l == 1).unwrap();
    p_shell[i].pure = !p_shell[i].pure;
    assert_ne!(pure.identity().fields, file.fields);
    assert_eq!(pure.identity().nao, file.nao);

    let mut no_oxygen = base.clone();
    no_oxygen.basis.shells.remove(&8);
    assert!(ScfIdentity::with_process(
        &no_oxygen.mol,
        &no_oxygen.basis,
        no_oxygen.spin,
        &no_oxygen.op,
        &no_oxygen.config,
        &no_oxygen.process,
    )
    .is_err());
}

fn iodomethane() -> Scenario {
    let xyz =
        "5\nCH3I\nC 0 0 0\nI 0 0 2.14\nH 1.03 0 -0.36\nH -0.51 0.89 -0.36\nH -0.51 -0.89 -0.36\n";
    let basis = bundled("def2-svp").unwrap();
    let mut mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    mol.apply_ecp(&basis);
    Scenario {
        mol,
        basis,
        ..Scenario::water(Spin::Restricted)
    }
}

#[test]
fn ecp_data_is_identity() {
    let base = iodomethane();
    let file = base.identity();
    assert!(file.fields.iter().any(|f| f.name == "ecp[Z=53]"));
    let mut changed = base.clone();
    changed.basis.ecps.get_mut(&53).unwrap().shells[0].terms[0].coef *= 1.0 + f64::EPSILON;
    let names: Vec<String> = changed
        .identity()
        .fields
        .iter()
        .zip(&file.fields)
        .filter(|(a, b)| a != b)
        .map(|(a, _)| a.name.clone())
        .collect();
    assert_eq!(names, ["ecp[Z=53]"]);
    // An ECP the molecule's atoms were not given changes the valence count.
    let mut not_applied = base.clone();
    not_applied.mol.atoms[1].n_core_ecp = 0;
    assert_ne!(not_applied.identity().fields, file.fields);
}

#[test]
fn an_empty_external_potential_is_no_potential() {
    let mut a = Scenario::water(Spin::Restricted);
    let none = a.identity();
    a.config.external_potential = Some(ExternalPotential::default());
    assert_eq!(a.identity().fields, none.fields);
}

#[test]
fn unchecked_settings_do_not_touch_the_identity() {
    let base = Scenario::water(Spin::Restricted);
    let mut s = base.clone();
    let c = &mut s.config;
    c.max_iter = 7;
    c.stall_window = Some(4);
    c.divergence_tol = Some(1.0);
    c.cdft_max_outer = 9;
    c.three_index_budget_bytes = 1;
    c.verbose = true;
    c.check_stability = true;
    assert_eq!(s.identity(), base.identity());
}

/// The table moves these from `None`, which would pass if only presence counted.
#[test]
fn a_state_selecting_field_is_compared_by_its_contents() {
    let mut base = Scenario::water(Spin::Restricted);
    base.config.guess_basis = Some(crate::basis2::GuessBasis {
        basis: bundled("sto-3g").unwrap(),
        projection: crate::basis2::GuessProjection::Fock,
    });
    base.config.init_guess_density = Some(c_order(1.0, 7, 7));
    let file = base.identity();
    let differs = |s: &Scenario| -> Vec<String> {
        s.identity()
            .fields
            .iter()
            .zip(&file.fields)
            .filter(|(a, b)| a != b)
            .map(|(a, _)| a.name.clone())
            .collect()
    };
    let mut exponent = base.clone();
    let guess = exponent.config.guess_basis.as_mut().unwrap();
    guess.basis.shells.get_mut(&1).unwrap()[0].exponents[0] *= 1.0 + f64::EPSILON;
    assert_eq!(differs(&exponent), ["guess_basis"]);
    let mut projection = base.clone();
    projection.config.guess_basis.as_mut().unwrap().projection =
        crate::basis2::GuessProjection::Overlap;
    assert_eq!(differs(&projection), ["guess_basis"]);
    let mut density = base.clone();
    density.config.init_guess_density = Some(c_order(1.5, 7, 7));
    assert_eq!(differs(&density), ["init_guess_density"]);
    // The same matrix in the other memory order is the same guess.
    let mut layout = base.clone();
    layout.config.init_guess_density = Some(f_order(1.0, 7, 7));
    assert!(differs(&layout).is_empty());
}

#[test]
fn skip_refuses_a_converged_exit_whose_flag_says_otherwise() {
    let current = Scenario::water(Spin::Restricted).identity();
    let mut ck = final_ck(Spin::Restricted);
    ck.final_state.as_mut().unwrap().converged = false;
    for allow_other_build in [false, true] {
        let mode = ReadMode::Skip { allow_other_build };
        let err = ck.check(&current, &clean_build(), mode).unwrap_err();
        assert!(matches!(err, CheckpointError::ExitRefused { .. }), "{err}");
    }
    assert!(ck.check(&current, &clean_build(), ReadMode::Guess).is_ok());
}

#[test]
fn orbitals_of_another_spin_kind_than_the_identity_are_refused() {
    let dir = Scratch::new();
    let mut ck = final_ck(Spin::Restricted);
    ck.identity = Scenario::water(Spin::Unrestricted).identity();
    let err = ck.write(&dir.file()).unwrap_err();
    assert!(
        is_format(&err) && err.to_string().contains("Restricted"),
        "{err}"
    );
    forged(&final_ck(Spin::Restricted), &dir.file(), |m| {
        let text = String::from_utf8(m.meta.clone()).unwrap();
        let changed = text.replacen(
            "\"name\":\"spin\",\"class\":\"Structure\",\"value\":\"Restricted\"",
            "\"name\":\"spin\",\"class\":\"Structure\",\"value\":\"Unrestricted\"",
            1,
        );
        assert_ne!(changed, text, "the edit must apply");
        m.meta = changed.into_bytes();
    });
    assert!(is_format(&Checkpoint::read(&dir.file()).unwrap_err()));
}

// ------------------------------------------------------- hostile member headers

/// An NPY 1.0 member with this header text and data, unpadded.
fn npy(header: &str, data: &[u8]) -> Vec<u8> {
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

/// Write these members as a stored zip, with no checksum or validation at all.
fn raw_npz(path: &Path, members: &[(&str, Vec<u8>)]) {
    use std::io::Write as _;
    let mut zip = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, bytes) in members {
        zip.start_file(format!("{name}.npy"), options).unwrap();
        zip.write_all(bytes).unwrap();
    }
    zip.finish().unwrap();
}

fn good_version() -> (&'static str, Vec<u8>) {
    let header = "{'descr': '<u8', 'fortran_order': False, 'shape': (1,)}\n";
    let data = FORMAT_VERSION.to_le_bytes();
    ("format_version", npy(header, &data))
}

/// A header that asks for far more than the file holds reached `vec![0; len]` and
/// aborted the process. It must come back through `Checkpoint::read` as `Err`.
#[test]
fn a_huge_shape_in_a_header_is_an_error_not_an_abort() {
    let dir = Scratch::new();
    let f8 = "{'descr': '<f8', 'fortran_order': False, 'shape': (1152921504606846975,)}\n";
    let u8_ = "{'descr': '<u8', 'fortran_order': False, 'shape': (1152921504606846975,)}\n";
    let u1 = "{'descr': '|u1', 'fortran_order': False, 'shape': (9223372036854775807,)}\n";
    let cases: Vec<Vec<(&str, Vec<u8>)>> = vec![
        vec![("format_version", npy(u8_, &[0; 8]))],
        vec![good_version(), ("checksum", npy(u8_, &[0; 8]))],
        vec![good_version(), ("meta", npy(u1, &[0; 8]))],
        vec![good_version(), ("density_total", npy(f8, &[0; 8]))],
    ];
    for members in cases {
        raw_npz(&dir.file(), &members);
        let err = Checkpoint::read(&dir.file()).unwrap_err();
        assert!(
            is_format(&err) && err.to_string().contains("shape"),
            "{err}"
        );
    }
}

#[test]
fn a_shape_that_overflows_or_disagrees_with_the_member_size_is_an_error() {
    let dir = Scratch::new();
    let overflow = "{'descr': '<f8', 'fortran_order': False, 'shape': (4294967296, 4294967296)}\n";
    let too_long = "{'descr': '<f8', 'fortran_order': False, 'shape': (2,)}\n";
    let too_short = "{'descr': '<f8', 'fortran_order': False, 'shape': (1,)}\n";
    // (2^61 + 1) * 8 wraps to exactly the 8 bytes present, so only a checked
    // multiplication catches it: a wrapping one sees a consistent member.
    let wraps_to_fit =
        "{'descr': '<f8', 'fortran_order': False, 'shape': (2305843009213693953,)}\n";
    let cases = [
        npy(overflow, &[0; 8]),
        npy(too_long, &[0; 8]),
        npy(too_short, &[0; 16]),
        npy(wraps_to_fit, &[0; 8]),
    ];
    for bytes in cases {
        raw_npz(&dir.file(), &[good_version(), ("density_total", bytes)]);
        let err = Checkpoint::read(&dir.file()).unwrap_err();
        // Named: without `density_total` this could be the missing checksum.
        assert!(
            is_format(&err) && err.to_string().contains("density_total"),
            "{err}"
        );
    }
}

/// Spellings the real parser may accept must not get past a check that cannot read
/// them. The escaped key here spells `shape`, hiding it from a substring search.
#[test]
fn a_header_this_format_does_not_write_is_refused() {
    let dir = Scratch::new();
    let hidden_shape =
        "{'descr': '<f8', 'fortran_order': False, 'sh\\x61pe': (9,), 'shape': (1,)}\n";
    // Consistent if read as NPY 1.0, so only the version byte can refuse it. NPY 2.0
    // has a longer length field, so reading it as 1.0 would misplace the header.
    let mut version_2 = npy(
        "{'descr': '<f8', 'fortran_order': False, 'shape': (1,)}\n",
        &[0; 8],
    );
    version_2[6] = 2;
    let cases = [
        npy(
            "{'descr':'<f8','fortran_order':False,'shape':(1,),}\n",
            &[0; 8],
        ),
        npy(
            // One element: a mutant treating every dtype as 8 bytes finds this
            // consistent, so the dtype itself has to refuse it.
            "{'descr': '<f4', 'fortran_order': False, 'shape': (1,)}\n",
            &[0; 8],
        ),
        npy(
            "{'descr': '<f8', 'fortran_order': False, 'shape': (0x1,)}\n",
            &[0; 8],
        ),
        npy(
            "{'descr': '<f8', 'fortran_order': False, 'shape': (1,)} x\n",
            &[0; 8],
        ),
        npy(hidden_shape, &[0; 8]),
        b"not an npy file at all".to_vec(),
        version_2,
    ];
    for bytes in cases {
        raw_npz(&dir.file(), &[good_version(), ("density_total", bytes)]);
        let err = Checkpoint::read(&dir.file()).unwrap_err();
        assert!(
            is_format(&err) && err.to_string().contains("density_total"),
            "{err}"
        );
    }
}

/// The zip's own size field is a claim too. Here it agrees with a large header
/// shape, so only the bound against the real file length stops the allocation.
#[test]
fn a_member_size_larger_than_the_file_is_refused() {
    let dir = Scratch::new();
    let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (100000000,)}\n";
    let bytes = npy(header, &[0; 8]);
    let claimed = (10 + header.len() + 800_000_000) as u32;
    raw_npz(&dir.file(), &[("density_total", bytes)]);
    let mut file = std::fs::read(dir.file()).unwrap();
    let central = file
        .windows(4)
        .position(|w| w == b"PK\x01\x02")
        .expect("a central directory entry");
    // Uncompressed size in the central directory, then in the local header.
    file[central + 24..central + 28].copy_from_slice(&claimed.to_le_bytes());
    file[22..26].copy_from_slice(&claimed.to_le_bytes());
    std::fs::write(dir.file(), &file).unwrap();
    let err = Checkpoint::read(&dir.file()).unwrap_err();
    assert!(
        is_format(&err) && err.to_string().contains("larger than the file"),
        "{err}"
    );
}

/// A deflated member's stored size says nothing about what it expands to.
#[test]
fn a_compressed_member_is_refused() {
    let dir = Scratch::new();
    let file = std::fs::File::create(dir.file()).unwrap();
    let mut npz = ndarray_npy::NpzWriter::new_compressed(file);
    npz.add_array("format_version", &ndarray::arr1(&[FORMAT_VERSION]))
        .unwrap();
    npz.finish().unwrap();
    let err = Checkpoint::read(&dir.file()).unwrap_err();
    assert!(
        is_format(&err) && err.to_string().contains("compressed"),
        "{err}"
    );
}

// ---------------------------------------------------------------- temp files

/// `count` names this process's next writes could pick, as a restarted job with a
/// reused PID would find them: its sequence restarts while the old files remain.
fn planted_names(dir: &Scratch, count: u64) -> Vec<PathBuf> {
    let start = format::TEMP_SEQ.load(Ordering::Relaxed);
    (start..start + count)
        .map(|seq| {
            let name = format!(".state.npz.tmp.{}.{seq}", std::process::id());
            dir.0.join(name)
        })
        .collect()
}

#[test]
fn a_leftover_temp_file_from_a_reused_pid_is_removed_and_does_not_block_the_write() {
    let dir = Scratch::new();
    for path in planted_names(&dir, 64) {
        std::fs::write(path, b"from a job that was killed").unwrap();
    }
    let ck = final_ck(Spin::Restricted);
    ck.write(&dir.file()).unwrap();
    assert!(same_checkpoint(
        &ck,
        &Checkpoint::read(&dir.file()).unwrap()
    ));
    assert_eq!(dir.names(), ["state.npz"]);
}

/// A name taken by a locked file belongs to a live writer, here standing in for one
/// in another PID namespace: the write steps past it and leaves it alone.
#[test]
fn a_temp_file_a_live_writer_holds_survives_and_its_name_is_stepped_past() {
    let dir = Scratch::new();
    let planted = planted_names(&dir, 64);
    let mut held = Vec::new();
    for path in &planted {
        let file = std::fs::File::create(path).unwrap();
        file.try_lock().unwrap();
        held.push(file);
    }
    let ck = final_ck(Spin::Restricted);
    ck.write(&dir.file()).unwrap();
    assert!(same_checkpoint(
        &ck,
        &Checkpoint::read(&dir.file()).unwrap()
    ));
    for path in &planted {
        assert!(path.exists(), "{path:?} was removed under a live writer");
    }
    assert_eq!(dir.names().len(), planted.len() + 1);
}

#[test]
fn cleanup_touches_only_this_paths_temp_files() {
    let dir = Scratch::new();
    std::fs::write(dir.0.join(".other.npz.tmp.1.1"), b"keep").unwrap();
    std::fs::write(dir.0.join("state.npz.bak"), b"keep").unwrap();
    std::fs::create_dir(dir.0.join(".state.npz.tmp.dir")).unwrap();
    final_ck(Spin::Restricted).write(&dir.file()).unwrap();
    assert_eq!(
        dir.names(),
        [
            ".other.npz.tmp.1.1",
            ".state.npz.tmp.dir",
            "state.npz",
            "state.npz.bak"
        ]
    );
}

/// The real thing rather than a planted file: a writer stopped mid-write, alive.
#[test]
fn a_write_does_not_remove_the_temp_file_of_a_writer_still_running() {
    // The last stage matters most: the lock must outlast the sync, up to the rename.
    for stage in BEFORE_RENAME_STAGES {
        let dir = Scratch::new();
        let mut running = writer_stopped_at(&dir.file(), stage);
        final_ck(Spin::Restricted).write(&dir.file()).unwrap();
        let names = dir.names();
        running.kill().unwrap();
        running.wait().unwrap();
        assert_eq!(names.len(), 2, "{stage}: {names:?}");
        assert!(
            names.iter().any(|n| n.contains(".tmp.")),
            "{stage}: {names:?}"
        );
    }
}

#[test]
fn a_symlink_at_the_path_is_refused_and_left_as_it_was() {
    let dir = Scratch::new();
    let target = dir.0.join("target.npz");
    let old = final_ck(Spin::Restricted);
    old.write(&target).unwrap();
    let before = std::fs::read(&target).unwrap();
    std::os::unix::fs::symlink(&target, dir.file()).unwrap();
    let err = newer_ck().write(&dir.file()).unwrap_err();
    assert!(
        matches!(&err, CheckpointError::Io { detail, .. } if detail.contains("symlink")),
        "{err}"
    );
    assert!(std::fs::symlink_metadata(dir.file())
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(std::fs::read(&target).unwrap(), before);
    assert_eq!(dir.names(), ["state.npz", "target.npz"]);
    // Reading through a link is harmless and still works.
    assert!(same_checkpoint(
        &old,
        &Checkpoint::read(&dir.file()).unwrap()
    ));
    // A dangling link is refused too, instead of being replaced.
    std::fs::remove_file(&target).unwrap();
    assert!(newer_ck().write(&dir.file()).is_err());
}

// -------------------------------------------------- process-wide settings, live

/// Set one variable on a real child and see which identity fields move. Children
/// because `set_var` would race every other test in this process.
#[test]
fn each_process_setting_reaches_the_identity_from_the_environment() {
    let test = "checkpoint::tests::child_prints_identity";
    let key = |envs: &[(&str, &str)]| -> ScfIdentity {
        let lines = child(test, envs);
        let line = lines.iter().find(|l| l.starts_with("KEY ")).unwrap();
        serde_json::from_str(&line["KEY ".len()..]).unwrap()
    };
    let base = key(&[]);
    for (variable, value, name) in [
        ("FERRIC_ERI_PRECISION", "1e-14", "eri_precision"),
        ("FERRIC_ECP_BACKEND", "libecpint", "ecp_backend"),
        ("FERRIC_LINDEP_THRESH", "0.003", "lindep_thresh"),
        ("FERRIC_GPU", "auto", "gpu_mode"),
        ("FERRIC_SCF_INCREMENTAL", "0", "FERRIC_SCF_INCREMENTAL"),
        (
            "FERRIC_SCF_UHF_INCREMENTAL",
            "1",
            "FERRIC_SCF_UHF_INCREMENTAL",
        ),
        ("FERRIC_SCF_COMBINED_JK", "0", "FERRIC_SCF_COMBINED_JK"),
        ("FERRIC_SHELLPAIR_CACHE", "0", "FERRIC_SHELLPAIR_CACHE"),
        ("FERRIC_DFK_FORCE_DENSITY", "1", "FERRIC_DFK_FORCE_DENSITY"),
        ("FERRIC_JK_STORAGE", "disk", "FERRIC_JK_STORAGE"),
        (
            "FERRIC_TERF_TABLE_DIR",
            "/elsewhere",
            "FERRIC_TERF_TABLE_DIR",
        ),
        ("FERRIC_DF_GUESS_TIGHT", "0", "FERRIC_DF_GUESS_TIGHT"),
    ] {
        let changed = key(&[(variable, value)]);
        let moved: Vec<&str> = changed
            .fields
            .iter()
            .zip(&base.fields)
            .filter(|(a, b)| a != b)
            .map(|(a, _)| a.name.as_str())
            .collect();
        assert_eq!(moved, [name], "{variable}={value}");
    }
    let field = |name: &str| base.fields.iter().find(|f| f.name == name).unwrap();
    assert_eq!(
        field("libint_version").value,
        ferric_integrals::libint_version()
    );
    assert!(field("cargo_features").value.contains("mpi="));
}

// ------------------------------------------------------------ identity, continued

#[test]
fn coordinates_warn_with_the_shift_under_guess_and_fail_bitwise_under_skip() {
    let base = Scenario::water(Spin::Restricted);
    let mut ck = final_ck(Spin::Restricted);
    ck.identity = base.identity();
    let mut moved = base.clone();
    moved.mol.atoms[2].y += 0.125;
    let current = moved.identity();
    let verdict = ck.check(&current, &clean_build(), ReadMode::Guess).unwrap();
    assert_eq!(verdict.warnings.len(), 1);
    assert_eq!(verdict.warnings[0].class, FieldClass::Coordinates);
    assert!((verdict.max_coord_shift_bohr.unwrap() - 0.125).abs() < 1e-12);
    let err = ck
        .check(&current, &clean_build(), CLASSES_SKIP)
        .unwrap_err();
    assert!(err.to_string().contains("coordinates"), "{err}");

    // One ulp, and the sign of zero, are both differences under "bitwise".
    let mut ulp = base.clone();
    ulp.mol.atoms[0].zpos = f64::from_bits(ulp.mol.atoms[0].zpos.to_bits() + 1);
    assert!(ck
        .check(&ulp.identity(), &clean_build(), CLASSES_SKIP)
        .is_err());
    let mut negative_zero = base.clone();
    negative_zero.mol.atoms[0].x = -0.0;
    assert!(ck
        .check(&negative_zero.identity(), &clean_build(), CLASSES_SKIP)
        .is_err());
}

#[test]
fn skip_needs_a_file_at_least_as_tight_and_guess_ignores_it() {
    let base = Scenario::water(Spin::Restricted);
    let mut ck = final_ck(Spin::Restricted);
    ck.identity = base.identity();
    let (e, d) = (base.config.energy_conv, base.config.density_conv);
    let cases = [
        // (requested energy_conv, requested density_conv, Some(field that fails))
        (e, d, None),
        (e * 10.0, d, None),
        (e, d * 10.0, None),
        (e / 10.0, d, Some("energy_conv")),
        (e, d / 10.0, Some("density_conv")),
    ];
    for (req_e, req_d, fails) in cases {
        let mut s = base.clone();
        s.config.energy_conv = req_e;
        s.config.density_conv = req_d;
        let current = s.identity();
        match (ck.check(&current, &clean_build(), CLASSES_SKIP), fails) {
            (Ok(_), None) => {}
            (Err(CheckpointError::Looser { field, .. }), Some(name)) => assert_eq!(field, name),
            (other, _) => panic!("{req_e} {req_d}: {other:?}"),
        }
        assert!(ck.check(&current, &clean_build(), ReadMode::Guess).is_ok());
    }
}

#[test]
fn skip_accepts_converged_and_echoes_plateau_and_refuses_every_other_exit() {
    let current = Scenario::water(Spin::Restricted).identity();
    for (exit, accepted) in [
        (ScfExit::Converged, true),
        (ScfExit::Plateau, true),
        (ScfExit::Stalled, false),
        (ScfExit::Diverged, false),
        (ScfExit::MaxIter, false),
        (ScfExit::NotCertified, false),
    ] {
        let mut ck = final_ck(Spin::Restricted);
        ck.final_state.as_mut().unwrap().exit = exit;
        let skip = ck.check(&current, &clean_build(), CLASSES_SKIP);
        if accepted {
            assert_eq!(skip.unwrap().file_exit, Some(exit));
        } else {
            let err = skip.unwrap_err();
            assert!(matches!(err, CheckpointError::ExitRefused { .. }), "{err}");
        }
        // The override is about builds; it cannot buy an uncertified state.
        let forced = ReadMode::Skip {
            allow_other_build: true,
        };
        assert_eq!(ck.check(&current, &clean_build(), forced).is_ok(), accepted);
        // A guess still iterates, so it takes any state and echoes it.
        let guess = ck.check(&current, &clean_build(), ReadMode::Guess).unwrap();
        assert_eq!(guess.file_exit, Some(exit));
    }
}

#[test]
fn a_mid_run_file_can_seed_but_not_replace_an_scf() {
    let current = Scenario::water(Spin::Restricted).identity();
    let mut ck = final_ck(Spin::Restricted);
    ck.final_state = None;
    let err = ck
        .check(&current, &clean_build(), CLASSES_SKIP)
        .unwrap_err();
    assert_eq!(err, CheckpointError::MidRun);
    assert_eq!(
        ck.check(&current, &clean_build(), ReadMode::Guess)
            .unwrap()
            .file_exit,
        None
    );
}

#[test]
fn a_state_that_had_unstored_physics_cannot_replace_an_scf() {
    let current = Scenario::water(Spin::Restricted).identity();
    for (name, blocks) in [
        ("induced_dipoles", true),
        ("cosx_final", true),
        ("stability", false),
        ("stability_external", false),
        ("cosx_schedule", false),
        ("guess_basis", false),
    ] {
        let mut ck = final_ck(Spin::Restricted);
        ck.final_state.as_mut().unwrap().not_stored = vec![name.to_string()];
        let skip = ck.check(&current, &clean_build(), CLASSES_SKIP);
        assert_eq!(skip.is_err(), blocks, "{name}");
        assert!(ck.check(&current, &clean_build(), ReadMode::Guess).is_ok());
    }
}

#[test]
fn skip_across_builds_is_an_error_unless_overridden_and_the_override_is_recorded() {
    let current = Scenario::water(Spin::Restricted).identity();
    let stamp = |commit: Option<&str>, dirty: Option<bool>| BuildStamp {
        version: "v".to_string(),
        commit: commit.map(str::to_string),
        dirty,
    };
    let clean_a = stamp(Some("a"), Some(false));
    let cases = [
        // (file, current, same build?)
        (clean_a.clone(), clean_a.clone(), true),
        (clean_a.clone(), stamp(Some("b"), Some(false)), false),
        // Same commit, but a dirty tree is not provably the same code.
        (stamp(Some("a"), Some(true)), clean_a.clone(), false),
        (clean_a.clone(), stamp(Some("a"), Some(true)), false),
        (
            stamp(Some("a"), Some(true)),
            stamp(Some("a"), Some(true)),
            false,
        ),
        // Unknown cannot be shown equal, even to itself.
        (stamp(None, Some(false)), stamp(None, Some(false)), false),
        (stamp(Some("a"), None), stamp(Some("a"), None), false),
    ];
    for (file_build, current_build, same) in cases {
        let mut ck = final_ck(Spin::Restricted);
        ck.build = file_build.clone();
        let strict = ck.check(&current, &current_build, CLASSES_SKIP);
        let forced = ck
            .check(
                &current,
                &current_build,
                ReadMode::Skip {
                    allow_other_build: true,
                },
            )
            .unwrap();
        assert_eq!(forced.file_build, file_build);
        let guess = ck.check(&current, &current_build, ReadMode::Guess).unwrap();
        assert!(!guess.build_override_used);
        if same {
            assert!(!strict.unwrap().build_override_used);
            assert!(
                !forced.build_override_used,
                "an unneeded override is not used"
            );
            assert!(guess.logged.is_empty());
        } else {
            let err = strict.unwrap_err();
            assert!(matches!(err, CheckpointError::OtherBuild { .. }), "{err}");
            assert!(
                forced.build_override_used,
                "{file_build:?} {current_build:?}"
            );
            assert_eq!(guess.logged[0].name, "build");
        }
    }
}

#[test]
fn the_override_does_not_excuse_an_identity_mismatch() {
    let mut changed = Scenario::water(Spin::Restricted);
    changed.config.xc = Some("PBE".to_string());
    let ck = final_ck(Spin::Restricted);
    let err = ck
        .check(
            &changed.identity(),
            &clean_build(),
            ReadMode::Skip {
                allow_other_build: true,
            },
        )
        .unwrap_err();
    assert!(matches!(err, CheckpointError::Identity(_)), "{err}");
}

#[test]
fn a_field_only_one_side_has_is_a_mismatch_in_that_fields_class() {
    let base = Scenario::water(Spin::Restricted);
    let mut ck = final_ck(Spin::Restricted);
    ck.identity = base.identity();
    let mut current = base.identity();
    current.fields.push(IdentityField {
        name: "new_method_knob".to_string(),
        class: FieldClass::Method,
        value: "1".to_string(),
    });
    let verdict = ck.check(&current, &clean_build(), ReadMode::Guess).unwrap();
    assert_eq!(verdict.warnings[0].name, "new_method_knob");
    assert_eq!(verdict.warnings[0].file, None);
    assert!(ck.check(&current, &clean_build(), CLASSES_SKIP).is_err());
    ck.identity.fields.retain(|f| f.name != "xc");
    assert!(ck
        .check(&base.identity(), &clean_build(), CLASSES_SKIP)
        .is_err());
}

#[test]
fn state_that_does_not_fit_the_identity_is_refused_at_construction() {
    let identity = Scenario::water(Spin::Restricted).identity();
    let wrong = orbitals(Spin::Restricted, 5);
    let err = Checkpoint::mid_run(wrong, 1, identity.clone(), clean_build()).unwrap_err();
    assert!(matches!(err, CheckpointError::Shape { .. }), "{err}");
    let right = orbitals(Spin::Restricted, 7);
    assert!(Checkpoint::mid_run(right, 1, identity, clean_build()).is_ok());
}

// ------------------------------------------------- across separate processes

const CHILD_ENV: &str = "FERRIC_CKPT_TEST_CHILD";

/// Run one test of this binary in a new process and return its stdout lines.
fn child(test: &str, envs: &[(&str, &str)]) -> Vec<String> {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", test, "--nocapture", "--test-threads=1"]);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.env(CHILD_ENV, "1").output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

/// Every map-free nested config populated, so that a future `HashMap` in any of
/// them is seen by the test below.
fn full_config() -> RhfConfig {
    RhfConfig {
        xc: Some("wB97X-V".to_string()),
        xc_omega: Some(0.3),
        dft_grid: Some(Default::default()),
        nlc_grid: Some(Default::default()),
        smearing_sigma: Some(1e-3),
        irrep_nelec: Some(vec![("A1".to_string(), 8), ("B2".to_string(), 2)]),
        k_builder: Some("cosx".to_string()),
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        df_k_aux: Some("def2-universal-jkfit".to_string()),
        external_potential: Some(ExternalPotential {
            point_charges: vec![PointCharge {
                q: 0.5,
                x: 1.0,
                y: 2.0,
                z: 3.0,
            }],
            smeared_charges: vec![SmearedCharge {
                q: -0.5,
                x: 1.0,
                y: 2.0,
                z: 4.0,
                width: 0.7,
            }],
            field: Some([0.0, 0.0, 1e-3]),
        }),
        cosmo: Some(Default::default()),
        pcm: Some(Default::default()),
        polarizable: Some(crate::polarizable::PolarizableSites {
            sites: vec![crate::polarizable::PolarizableSite {
                x: 1.0,
                y: 0.0,
                z: 0.0,
                alpha: 2.0,
            }],
            exclusions: vec![(0, 1)],
            ..Default::default()
        }),
        constraints: vec![ferric_dft::cdft::Constraint {
            fragment: vec![0, 1],
            spin: ferric_dft::cdft::SpinChannel::SpinDiff,
            target: 1.0,
        }],
        cdft_lambda_init: Some(vec![0.5]),
        // Holds a `BasisSet`, which is the one with maps in it.
        guess_basis: Some(crate::basis2::GuessBasis {
            basis: bundled("def2-svp").unwrap(),
            projection: crate::basis2::GuessProjection::Overlap,
        }),
        init_guess_density: Some(c_order(1.0, 3, 3)),
        use_sad_guess: false,
        trah_trigger: Some(0.1),
        newton_trigger: 1e-2,
        ah_trigger: 1e-3,
        diis_flavor: DiisFlavor::Ediis,
        level_shift: 0.2,
        ..Default::default()
    }
}

#[test]
fn child_prints_identity() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    let mut s = iodomethane();
    s.config = full_config();
    s.op = Operator::terfc_fit(1.05);
    let identity = ScfIdentity::new(&s.mol, &s.basis, s.spin, &s.op, &s.config).unwrap();
    // libtest has already printed "test name ... " on this line.
    println!();
    println!("KEY {}", serde_json::to_string(&identity).unwrap());
    println!("DEBUG {:?}", s.basis);
}

/// `BasisSet`'s `Debug` text must differ between these processes, or the test has
/// no power: this is the exact trap a key built from `{:?}` falls into.
#[test]
fn identity_is_the_same_bytes_across_processes_though_debug_order_is_not() {
    let test = "checkpoint::tests::child_prints_identity";
    let runs: Vec<Vec<String>> = (0..4).map(|_| child(test, &[])).collect();
    let pick = |lines: &[String], tag: &str| -> String {
        lines
            .iter()
            .find(|l| l.starts_with(tag))
            .unwrap_or_else(|| panic!("no {tag} line in {lines:?}"))
            .clone()
    };
    let keys: BTreeSet<String> = runs.iter().map(|r| pick(r, "KEY ")).collect();
    let debug: BTreeSet<String> = runs.iter().map(|r| pick(r, "DEBUG ")).collect();
    assert_eq!(keys.len(), 1, "the key changed between processes");
    assert!(
        debug.len() > 1,
        "the witness did not vary, so this proved nothing"
    );
}

// ---------------------------------------------------------- atomic write, kill

const KILL_PATH_ENV: &str = "FERRIC_CKPT_TEST_KILL_PATH";
const KILL_STAGE_ENV: &str = "FERRIC_CKPT_TEST_KILL_STAGE";
const READY: &str = "CHILD-AT-STAGE";

/// The new contents a killed writer was writing: distinct from `final_ck`.
fn newer_ck() -> Checkpoint {
    let mut ck = final_ck(Spin::Unrestricted);
    ck.iteration = 99;
    ck.orbitals.density_total[[0, 0]] = 123.5;
    ck
}

#[test]
fn child_stops_in_a_write() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    let path = PathBuf::from(std::env::var(KILL_PATH_ENV).unwrap());
    let stage = std::env::var(KILL_STAGE_ENV).unwrap();
    let ck = newer_ck();
    format::write(&ck, &path, &mut |s: WriteStage| {
        if format!("{s:?}") == stage {
            println!("{READY}");
            std::thread::sleep(std::time::Duration::from_secs(600));
        }
    })
    .unwrap();
    panic!("the write finished without reaching {stage}");
}

/// Start a real writer and wait until it is inside `stage`, where it sleeps.
fn writer_stopped_at(path: &Path, stage: &str) -> std::process::Child {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "checkpoint::tests::child_stops_in_a_write",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .env(KILL_PATH_ENV, path)
        .env(KILL_STAGE_ENV, stage)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let reached = BufReader::new(stdout)
        .lines()
        .any(|line| line.is_ok_and(|l| l.contains(READY)));
    assert!(reached, "the writer never reached {stage}");
    child
}

/// Start a real writer, wait until it is inside `stage`, SIGKILL it.
fn kill_writer_at(path: &Path, stage: &str) {
    let mut child = writer_stopped_at(path, stage);
    child.kill().unwrap();
    let status = child.wait().unwrap();
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(status.signal(), Some(9), "{stage}: {status:?}");
}

const BEFORE_RENAME_STAGES: [&str; 3] = ["PartWritten", "TempWritten", "BeforeRename"];

#[test]
fn a_kill_before_the_rename_leaves_the_old_file_byte_identical() {
    for stage in BEFORE_RENAME_STAGES {
        let dir = Scratch::new();
        let old = final_ck(Spin::Unrestricted);
        old.write(&dir.file()).unwrap();
        let old_bytes = std::fs::read(dir.file()).unwrap();
        kill_writer_at(&dir.file(), stage);
        assert_eq!(std::fs::read(dir.file()).unwrap(), old_bytes, "{stage}");
        let read = Checkpoint::read(&dir.file()).unwrap();
        assert!(same_checkpoint(&old, &read), "{stage}");
        // The seam is real: the killed writer left its temp file behind.
        let leftovers = dir.names();
        assert_eq!(leftovers.len(), 2, "{stage}: {leftovers:?}");
        assert!(
            leftovers.iter().any(|n| n.contains(".tmp.")),
            "{leftovers:?}"
        );
        // The next writer is not obstructed, and supersedes the old file.
        let newer = newer_ck();
        newer.write(&dir.file()).unwrap();
        assert!(same_checkpoint(
            &newer,
            &Checkpoint::read(&dir.file()).unwrap()
        ));
        // A dead writer holds no lock, so its leftover goes.
        assert_eq!(dir.names(), ["state.npz"], "{stage}");
    }
}

#[test]
fn a_kill_part_way_through_really_is_part_way_through() {
    let dir = Scratch::new();
    kill_writer_at(&dir.file(), "PartWritten");
    let names = dir.names();
    assert_eq!(names.len(), 1, "{names:?}");
    let partial = std::fs::metadata(dir.0.join(&names[0])).unwrap().len();
    let whole_path = dir.0.join("whole.npz");
    newer_ck().write(&whole_path).unwrap();
    let whole = std::fs::metadata(whole_path).unwrap().len();
    assert!(0 < partial && partial < whole, "{partial} of {whole}");
}

#[test]
fn a_kill_with_no_old_file_leaves_no_readable_file_at_the_path() {
    for stage in BEFORE_RENAME_STAGES {
        let dir = Scratch::new();
        kill_writer_at(&dir.file(), stage);
        let err = Checkpoint::read(&dir.file()).unwrap_err();
        assert!(
            matches!(err, CheckpointError::Missing { .. }),
            "{stage}: {err}"
        );
    }
}

#[test]
fn a_kill_after_the_rename_leaves_the_complete_new_file() {
    let dir = Scratch::new();
    final_ck(Spin::Unrestricted).write(&dir.file()).unwrap();
    kill_writer_at(&dir.file(), "Renamed");
    let read = Checkpoint::read(&dir.file()).unwrap();
    assert!(same_checkpoint(&newer_ck(), &read));
    assert_eq!(dir.names(), ["state.npz"]);
}

#[test]
fn a_successful_write_leaves_only_the_file_and_observes_the_order() {
    let dir = Scratch::new();
    let mut stages = Vec::new();
    format::write(&final_ck(Spin::Unrestricted), &dir.file(), &mut |s| {
        // The destination must not appear before the rename.
        assert_eq!(dir.file().exists(), s == WriteStage::Renamed, "{s:?}");
        stages.push(s);
    })
    .unwrap();
    assert_eq!(
        stages,
        [
            WriteStage::PartWritten,
            WriteStage::TempWritten,
            WriteStage::BeforeRename,
            WriteStage::Renamed
        ]
    );
    assert_eq!(dir.names(), ["state.npz"]);
}

#[test]
fn failed_writes_are_errors_and_clean_up() {
    let dir = Scratch::new();
    let ck = final_ck(Spin::Restricted);
    // The destination's directory does not exist.
    let err = ck
        .write(&dir.0.join("no-such-dir").join("x.npz"))
        .unwrap_err();
    assert!(matches!(err, CheckpointError::Io { .. }), "{err}");
    // The rename cannot replace a directory.
    let occupied = dir.0.join("occupied");
    std::fs::create_dir(&occupied).unwrap();
    let err = ck.write(&occupied).unwrap_err();
    assert!(matches!(err, CheckpointError::Io { .. }), "{err}");
    assert_eq!(dir.names(), ["occupied"]);
    // A path with no file name at all.
    let err = ck.write(Path::new("/")).unwrap_err();
    assert!(matches!(err, CheckpointError::Io { .. }), "{err}");
}

// -------------------------------------------------- the frozen file for numpy

fn golden_path() -> PathBuf {
    // At run time: a compile-time path is wrong for a test archive run elsewhere,
    // and cargo and nextest both set this for the process, remapped.
    Path::new(&std::env::var("CARGO_MANIFEST_DIR").expect("run this through cargo or nextest"))
        .join("../../testdata/checkpoint/synthetic_rohf_v1.npz")
}

fn golden_ck() -> Checkpoint {
    final_ck(Spin::RestrictedOpen)
}

/// A file a v1 reader must keep reading, and which
/// `crates/ferric-python/tests/test_scf_checkpoint_format.py` reads with numpy.
/// Regenerate only with a [`FORMAT_VERSION`] bump:
/// `FERRIC_REGEN_CKPT_GOLDEN=1 cargo test -p ferric-scf --lib checkpoint::tests::the_frozen`.
#[test]
fn the_frozen_v1_file_still_reads_to_the_same_state() {
    let path = golden_path();
    if std::env::var_os("FERRIC_REGEN_CKPT_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        golden_ck().write(&path).unwrap();
    }
    let read = Checkpoint::read(&path).unwrap();
    let want = golden_ck();
    assert!(same_state(&want, &read));
    assert_eq!(read.build, want.build);
    // Only structure is frozen: a new RhfConfig field legitimately adds a row
    // that this file predates. A canonicalization change would show up here.
    let structure = |id: &ScfIdentity| -> Vec<IdentityField> {
        id.fields
            .iter()
            .filter(|f| f.class == FieldClass::Structure)
            .cloned()
            .collect()
    };
    assert_eq!(structure(&read.identity), structure(&want.identity));
}

/// The chain from a fresh file to numpy: the Python test reads the frozen one, and
/// this holds the current writer to that file's members, dtypes and layouts.
#[test]
fn a_fresh_file_has_the_frozen_files_member_structure() {
    let dir = Scratch::new();
    golden_ck().write(&dir.file()).unwrap();
    let members = |path: &Path| -> Vec<(String, Vec<usize>, bool)> {
        let bytes = std::fs::read(path).unwrap();
        let mut npz = ndarray_npy::NpzReader::new(std::io::Cursor::new(bytes)).unwrap();
        let mut out = Vec::new();
        for name in npz.names().unwrap() {
            if let Ok(a) = npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::IxDyn>(&name) {
                out.push((name, a.shape().to_vec(), a.is_standard_layout()));
            } else {
                out.push((name, Vec::new(), true));
            }
        }
        out.sort();
        out
    };
    assert_eq!(members(&dir.file()), members(&golden_path()));
}
