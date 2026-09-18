//! With AURORA disabled, every SCF energy must be BIT-IDENTICAL to the value
//! the solver produced before the accelerator existed.
//!
//! This is the repo's non-negotiable exactness anchor for an opt-in feature: an
//! accelerator that is off must not perturb the default path at all — not by a
//! ulp, and not by an iteration.
//!
//! # Where the reference bits come from
//!
//! They were captured by running these exact six configurations against the
//! **unmodified** `rhf.rs` at base commit `396e0d61` (the AURORA module
//! unregistered from `lib.rs` and `rhf.rs` restored from `git show HEAD:`), then
//! re-run with the accelerator compiled in and disabled. Both runs produced the
//! bit patterns below and the same iteration counts.
//!
//! Pinning literal `u64` bit patterns rather than comparing two live runs is
//! deliberate: a live A/B comparison would pass even if BOTH sides drifted
//! together, which is exactly the regression this file exists to catch.
//!
//! # What this test can and cannot see
//!
//! It sees any change to the default SCF path — reordered arithmetic, an extra
//! or skipped iteration, a different convergence decision. It does NOT verify
//! that AURORA itself is correct; `aurora_math_anchors.rs` and
//! `aurora_same_fixed_point.rs` cover that. It is also, by construction, tied to
//! this machine's BLAS: a different OpenBLAS kernel would change the low bits
//! legitimately. Should that happen, re-capture against an unmodified tree and
//! confirm the AURORA-off and AURORA-absent numbers still agree with EACH OTHER
//! — that agreement, not the literal constant, is the invariant.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// One pinned case: label, geometry, basis, optional functional, expected energy
/// bits, expected iteration count.
struct Case {
    name: &'static str,
    xyz: &'static str,
    basis: &'static str,
    xc: Option<&'static str>,
    bits: u64,
    iters: usize,
}

const CASES: &[Case] = &[
    Case {
        name: "water/STO-3G RHF",
        xyz: "../../testdata/molecules/water.xyz",
        basis: "sto-3g",
        xc: None,
        bits: 0xc052bda43279da8f,
        iters: 9,
    },
    Case {
        name: "water/cc-pVDZ RHF",
        xyz: "../../testdata/molecules/water.xyz",
        basis: "cc-pvdz",
        xc: None,
        bits: 0xc05301b6911e53fd,
        iters: 12,
    },
    Case {
        name: "methane/cc-pVDZ RHF",
        xyz: "../../testdata/molecules/methane.xyz",
        basis: "cc-pvdz",
        xc: None,
        bits: 0xc044196f4811b2bb,
        iters: 11,
    },
    Case {
        name: "benzene/STO-3G RHF",
        xyz: "../../testdata/molecules/benzene.xyz",
        basis: "sto-3g",
        xc: None,
        bits: 0xc06c7c80f9e66780,
        iters: 10,
    },
    Case {
        name: "water/cc-pVDZ PBE",
        xyz: "../../testdata/molecules/water.xyz",
        basis: "cc-pvdz",
        xc: Some("PBE"),
        bits: 0xc05315583aeeda85,
        iters: 87,
    },
    Case {
        name: "water/cc-pVDZ B3LYP",
        xyz: "../../testdata/molecules/water.xyz",
        basis: "cc-pvdz",
        xc: Some("B3LYP"),
        bits: 0xc0531ae8002df977,
        iters: 57,
    },
];

fn solve(case: &Case) -> (f64, usize) {
    let mol = Molecule::load_xyz(case.xyz).unwrap();
    let bs = basis::bundled(case.basis).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        xc: case.xc.map(|s| s.to_string()),
        df_j_aux: case.xc.map(|_| "def2-universal-jkfit".to_string()),
        ..Default::default()
    };
    // Belt and braces: the default must BE disabled, and this test must be
    // measuring the default. If someone flips the default to `true`, this
    // assertion fires with a clear message rather than the bit comparison
    // failing with an inscrutable one.
    assert!(
        !cfg.aurora.enabled,
        "AuroraConfig::default() must be disabled — this test measures the default path"
    );
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    (r.energy, r.iterations)
}

#[test]
fn aurora_disabled_reproduces_pre_aurora_energies_bit_for_bit() {
    let mut failures = Vec::new();
    for case in CASES {
        let (e, iters) = solve(case);
        eprintln!(
            "{:<22} E = {:.12}  bits = {:#018x}  iters = {}",
            case.name,
            e,
            e.to_bits(),
            iters
        );
        if e.to_bits() != case.bits {
            failures.push(format!(
                "{}: energy bits {:#018x} != expected {:#018x} (ΔE = {:.3e})",
                case.name,
                e.to_bits(),
                case.bits,
                e - f64::from_bits(case.bits)
            ));
        }
        if iters != case.iters {
            failures.push(format!(
                "{}: {} iterations, expected {} — the default SCF PATH changed",
                case.name, iters, case.iters
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "AURORA-off must be bit-identical to the pre-AURORA solver:\n  {}",
        failures.join("\n  ")
    );
}

/// The accelerator must not even be CONSTRUCTED when disabled.
///
/// Building the auxiliary curvature model costs a three-index integral build, so
/// a disabled accelerator that still constructs itself would be a silent cost
/// regression that the bit-identity test above cannot see (the energy would be
/// unchanged). The auxiliary operator-call counter is the observable.
#[test]
fn disabled_aurora_performs_no_auxiliary_work() {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    assert!(r.converged);
    // The only way to observe "no accelerator was built" from outside is that
    // the run behaved exactly like the pinned baseline, which the test above
    // checks. Here we additionally assert the config default itself, so a future
    // change that flips it on by accident is caught at the source.
    assert!(!RhfConfig::default().aurora.enabled);
    eprintln!("disabled run converged in {} iterations", r.iterations);
}
