//! cc-pVDZ second-row elements (Z = 11..18), from the Basis Set Exchange.
//!
//! # Why these were added
//!
//! ferric's bundled cc-pVDZ covered only Z = 1..10 — anomalous, since STO-3G,
//! 6-31G and def2-SVP all reach Z = 18 and aug-cc-pVDZ reaches Z = 36. The gap
//! surfaced when a DLPNO validation sweep could run H2S in STO-3G but not
//! cc-pVDZ.
//!
//! Fetched from `basissetexchange.org/api/basis/cc-pvdz/format/json` rather than
//! derived or hand-entered. (An aug-cc-pVDZ-minus-diffuse derivation was tried
//! first and turned out to be byte-identical to BSE's data — but "verified
//! against the authoritative source" is a better provenance than "happened to
//! agree with my reconstruction".)
//!
//! # ROOT CAUSE FOUND (2026-07-27): the shipped data is NOT PySCF's basis
//!
//! All eight elements differ from PySCF, and the four that appeared to
//! "validate" were a COINCIDENCE at the energy level, not agreement.
//!
//! The difference is in the one-electron integrals, so it has nothing to do with
//! SCF, screening or convergence. NaH kinetic-energy diagonal, ferric vs PySCF:
//!
//! ```text
//!   AO      PySCF      ferric
//!   1s    56.2724     56.2719     ~
//!   3s     0.6472      0.2698     2.4x OFF
//!   3p     0.4783      0.1411     3.4x OFF
//!   4s     0.03461     0.03461    exact
//! ```
//!
//! H2S shows the same signature (3s 7.735 vs 2.470, 3p 4.530 vs 1.812) even
//! though its ENERGY matched PySCF to 10 digits — which is why the energy check
//! alone was not sufficient evidence.
//!
//! **What differs.** BSE's canonical cc-pVDZ for Z >= 11 fuses the most diffuse
//! s/p primitive into the contracted shell AND repeats it as a standalone
//! function: contractions 0-2 span all 12 primitives (coefficient on the 12th is
//! 2.4e-4 / -5.7e-3 / 4.3e-1 for Na — not negligible), plus a 4th contraction
//! that is just that primitive. PySCF instead contracts only the first 11 and
//! carries the 12th as a genuinely separate uncontracted shell.
//!
//! These span the same FUNCTION SPACE, which is why the total energies can land
//! close, but they are different non-orthogonal basis sets, so individual
//! integrals and any energy sensitive to the contraction differ. Na and Mg show
//! it worst because their diffuse exponent is the smallest (0.02307 for Na vs
//! 0.157 for S), so the fused-vs-separate choice matters most there.
//!
//! **Consequence.** ferric's Z = 11..18 cc-pVDZ is the BSE-canonical form. It is
//! internally consistent and a legitimate cc-pVDZ, but it is NOT bit-comparable
//! to PySCF's, and no element here should be quoted as PySCF-cross-checked. The
//! test below therefore pins the MEASURED deviations rather than asserting an
//! agreement that does not exist.
//!
//! Anyone needing PySCF-comparable second-row cc-pVDZ must split the diffuse
//! primitive out of the contractions to match PySCF's convention — a real change
//! to the shipped data, not done here because it is not obviously the right
//! convention to prefer.

use ferric_core::{basis, mol::Molecule, parallel::ParallelContext};
use ferric_integrals::{basis_bridge::PreparedBasis, operator::Operator};
use ferric_scf::{
    rhf::{solve_rhf, RhfConfig},
    screening::SchwarzBounds,
};

struct Case {
    label: &'static str,
    xyz: &'static str,
    nbasis: usize,
    /// PySCF cc-pVDZ RHF reference (pyscf 2.13.0) at this exact geometry.
    pyscf: f64,
}

/// Elements whose TOTAL energy happens to land within 1e-8 of PySCF despite the
/// different contraction convention (see the module doc — their integrals do NOT
/// agree). Kept as a stability pin, NOT as a cross-validation.
const ENERGY_COINCIDES: &[Case] = &[
    Case {
        label: "SiH4 (Z=14)",
        xyz: "5\n\nSi 0.0 0.0 0.0\nH 0.0 0.0 1.480\nH 1.395 0.0 -0.493\n\
              H -0.698 1.209 -0.493\nH -0.698 -1.209 -0.493\n",
        nbasis: 38,
        pyscf: -291.2428547526,
    },
    Case {
        label: "H2S (Z=16)",
        xyz: "3\n\nS 0.0 0.0 0.0\nH 0.0 0.0 1.336\nH 1.296 0.0 -0.323\n",
        nbasis: 28,
        pyscf: -398.6916925393,
    },
    Case {
        label: "HCl (Z=17)",
        xyz: "2\n\nCl 0.0 0.0 0.0\nH 0.0 0.0 1.275\n",
        nbasis: 23,
        pyscf: -460.0894465109,
    },
    Case {
        label: "Ar (Z=18)",
        xyz: "1\n\nAr 0.0 0.0 0.0\n",
        nbasis: 18,
        pyscf: -526.7998653097,
    },
];

/// Elements that do NOT agree, pinned so the discrepancy cannot silently grow
/// or silently disappear unnoticed.
const KNOWN_DIVERGENT: &[(Case, f64)] = &[
    (
        Case {
            label: "NaH (Z=11)",
            xyz: "2\n\nNa 0.0 0.0 0.0\nH 0.0 0.0 1.887\n",
            nbasis: 23,
            pyscf: -162.3839468805,
        },
        6.8e-6,
    ),
    (
        Case {
            label: "MgH2 (Z=12)",
            xyz: "3\n\nMg 0.0 0.0 0.0\nH 0.0 0.0 1.73\nH 0.0 0.0 -1.73\n",
            nbasis: 28,
            pyscf: -200.7293003289,
        },
        6.2e-5,
    ),
];

fn run(case: &Case) -> f64 {
    let ctx = ParallelContext::default();
    let bs = basis::bundled("cc-pvdz").expect("cc-pvdz must load");
    let mol = Molecule::parse_xyz(case.xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap_or_else(|e| panic!("{}: {e:?}", case.label));
    assert_eq!(
        obs.nbasis(),
        case.nbasis,
        "{}: nbasis {} != PySCF's {} — the contraction STRUCTURE differs, not \
         just the numbers",
        case.label,
        obs.nbasis(),
        case.nbasis
    );
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let cfg = RhfConfig { density_conv: 1e-10, max_iter: 200, ..Default::default() };
    let rhf = solve_rhf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &cfg).unwrap();
    assert!(rhf.converged, "{}: SCF must converge", case.label);
    rhf.energy
}

/// Si, S, Cl, Ar land within 1e-8 of PySCF's total energy.
///
/// This is a STABILITY pin, not a validation: their one-electron integrals
/// demonstrably differ from PySCF's (module doc), so the energy agreement is a
/// coincidence of the two contraction conventions spanning the same space. If
/// this test breaks, the shipped data changed — investigate, but do not read a
/// pass here as "ferric matches PySCF for these elements".
#[test]
fn energy_coincidence_is_stable() {
    for case in ENERGY_COINCIDES {
        let e = run(case);
        let d = (e - case.pyscf).abs();
        eprintln!("{:12} ferric {e:.10}  PySCF {:.10}  |dE| = {d:.2e}", case.label, case.pyscf);
        assert!(
            d < 1e-8,
            "{}: ferric {e:.10} vs PySCF {:.10}, |dE| = {d:.3e}",
            case.label,
            case.pyscf
        );
    }
}

/// Na and Mg diverge from PySCF. Pinned at their MEASURED size.
///
/// This asserts the discrepancy stays in its known band rather than asserting it
/// away: a tightening (someone fixes it) or a widening (someone makes it worse)
/// both fail the test and demand a look. Deliberately not `#[ignore]`d — an
/// ignored test is one nobody reads.
#[test]
fn known_divergent_second_row_stays_in_band() {
    for (case, expected) in KNOWN_DIVERGENT {
        let e = run(case);
        let d = (e - case.pyscf).abs();
        eprintln!(
            "{:12} ferric {e:.10}  PySCF {:.10}  |dE| = {d:.2e}  (known ~{expected:.1e})",
            case.label, case.pyscf
        );
        assert!(
            d > 1e-8,
            "{} now AGREES with PySCF (|dE| = {d:.3e}) — either the shipped data \
             was changed to PySCF's split-primitive convention, or PySCF changed. \
             Record which before updating this test.",
            case.label
        );
        assert!(
            d < expected * 3.0,
            "{}: |dE| = {d:.3e} has grown well beyond the known {expected:.1e} — \
             something regressed",
            case.label
        );
    }
}

/// First-row elements must be untouched by the addition.
///
/// Guards against the install having rewritten Z = 1..10 while adding Z = 11..18.
#[test]
fn first_row_ccpvdz_is_unchanged() {
    let ctx = ParallelContext::default();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let mol = Molecule::load_xyz(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/water.xyz"
    ))
    .unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    assert_eq!(obs.nbasis(), 24, "water/cc-pVDZ must still be 24 basis functions");

    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let cfg = RhfConfig { density_conv: 1e-10, max_iter: 200, ..Default::default() };
    let rhf = solve_rhf(&ctx, &mol, &obs, Operator::coulomb(), &bounds, &cfg).unwrap();

    let expected = -76.0267679973766_f64;
    eprintln!("water/cc-pVDZ ferric {:.10}  ref {expected:.10}", rhf.energy);
    assert!(
        (rhf.energy - expected).abs() < 1e-6,
        "water/cc-pVDZ moved to {:.10} (ref {expected:.10}) — adding Z = 11..18 \
         perturbed the existing first-row data",
        rhf.energy
    );
}
