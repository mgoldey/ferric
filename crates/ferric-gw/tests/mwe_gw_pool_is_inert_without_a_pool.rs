//! MWE: with NO pool installed, GW's quasiparticle energies are BIT-IDENTICAL
//! to the pre-migration tree.
//!
//! # Why this test exists and why it was written FIRST
//!
//! The shared brief's non-negotiable #1: "Unbudgeted must mean *do exactly
//! what you did before*, never *refuse everything*." Every gate this crate
//! adds returns an INERT `Reservation` when `pool::global()` is `None`, which
//! debits nothing and releases nothing. The property that has to hold is not
//! "the run still succeeds" -- a gate that quietly shrank a GEMM's blocking
//! would also succeed, and would move the energy in the 12th digit. It is
//! `to_bits()` equality on every QP energy.
//!
//! # Why `to_bits()` and not a tolerance
//!
//! A tolerance cannot distinguish "the migration is a no-op" from "the
//! migration re-associated an accumulation by a few ulp". The ksdft migration
//! learned this the expensive way: a batch width sized from live RSS gave
//! -390.3794282913 and -390.3794337741 Ha for the SAME input, which every
//! sensible tolerance would have passed. Bits or nothing.
//!
//! # The reference values are FROZEN CONSTANTS, not a second run
//!
//! Comparing two runs in the same process only proves the code is
//! deterministic; it cannot see a change that moved BOTH runs. The constants
//! below were captured from the tree at `72883be0` (the base commit of this
//! migration, before any ferric-gw pool charge existed) and are compared
//! bit-for-bit. If a future change to the numerics is deliberate, these
//! constants must be re-captured deliberately, which is the point.
//!
//! Shape: water / STO-3G orbital basis with a cc-pVDZ-RI auxiliary basis,
//! G0W0@HF and COHSEX@HF, full rank -- small enough to run in the default
//! (non-`--release`, non-`--ignored`) suite at every worker count.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO  0.0   0.0       0.117790\nH  0.0   0.755453 -0.471161\nH  0.0  -0.755453 -0.471161\n",
        0,
        1,
    )
    .expect("parse H2O")
}

fn setup() -> (
    Molecule,
    PreparedBasis,
    PreparedBasis,
    ferric_scf::ScfResult,
) {
    let mol = water();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").expect("sto-3g")).expect("obs");
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").expect("ri")).expect("dfbs");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).expect("schwarz");
    let rhf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig::default(),
    )
    .expect("RHF");
    (mol, obs, dfbs, rhf)
}

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        need_eigenvalues_freq: true,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 8,
            u0: 0.5,
        },
        eigensolver_conv_thresh: 1e-7,
        eigensolver_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        need_inv_dielectric_freq: false, // run_gw forces this on (M9 gate)
        verbose: false,
    }
}

pub fn run(method: GwMethod) -> Vec<f64> {
    let (mol, obs, dfbs, rhf) = setup();
    let res = run_gw(
        &mol,
        &obs,
        &dfbs,
        Operator::coulomb(),
        &rhf,
        &pdep_cfg(),
        &GwConfig {
            method,
            ..Default::default()
        },
        None,
    )
    .expect("GW runs unbudgeted");
    res.eps_qp.to_vec()
}

/// Frozen G0W0 QP energies (Ha), captured on the pre-migration tree at
/// `72883be0`:
///   -6.15571215606518174e-1  -4.18041879665428884e-1  -3.30966250586579025e-1
///    6.09356792132542346e-1   7.42858606124310539e-1
const G0W0_QP_BITS: [u64; 5] = [
    0xBFE3B2C267EC725D,
    0xBFDAC132BA615E5D,
    0xBFD52E8D1196579E,
    0x3FE37FD9D0B9C06D,
    0x3FE7C57F695B64D6,
];

/// Frozen COHSEX QP energies (Ha), same shape, same tree:
///   -6.29599329764241822e-1  -4.22570898390289329e-1  -3.27295802184455642e-1
///    6.26889454649354971e-1   7.64533208860053715e-1
const COHSEX_QP_BITS: [u64; 5] = [
    0xBFE425AD7E5D7853,
    0xBFDB0B66CF34F7EB,
    0xBFD4F26A17A00548,
    0x3FE40F7A793DA3E1,
    0x3FE8770E591850D7,
];

fn assert_bits(got: &[f64], want: &[u64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: QP count changed");
    let mismatch: Vec<String> = got
        .iter()
        .zip(want)
        .enumerate()
        .filter(|(_, (g, w))| g.to_bits() != **w)
        .map(|(i, (g, w))| {
            format!(
                "  [{i}] got {g:.17e} (0x{:016X}) want 0x{w:016X} (delta {:.3e})",
                g.to_bits(),
                g - f64::from_bits(*w)
            )
        })
        .collect();
    assert!(
        mismatch.is_empty(),
        "{what}: the unbudgeted path is NOT bit-identical to the pre-migration tree.\n\
         Every ferric-gw gate must return an INERT reservation when no pool is installed; \
         a non-inert one that shrank a buffer or re-blocked a GEMM would show up exactly \
         here.\n{}",
        mismatch.join("\n")
    );
}

#[test]
fn g0w0_qp_energies_are_bit_identical_without_a_pool() {
    assert!(
        ferric_core::memory::pool::global().is_none(),
        "this test asserts the NO-POOL path; something installed a global pool"
    );
    assert_bits(&run(GwMethod::G0W0), &G0W0_QP_BITS, "G0W0");
}

#[test]
fn cohsex_qp_energies_are_bit_identical_without_a_pool() {
    assert!(
        ferric_core::memory::pool::global().is_none(),
        "this test asserts the NO-POOL path; something installed a global pool"
    );
    assert_bits(&run(GwMethod::Cohsex), &COHSEX_QP_BITS, "COHSEX");
}
