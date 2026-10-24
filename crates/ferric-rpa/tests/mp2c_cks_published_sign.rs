//! KNOWN OPEN DEFECT: the CKS MP2C correction's sign vs the published table.
//!
//! Every test here is `#[ignore]`d with a `known open:` reason and asserts the
//! PUBLISHED sign, so it FAILS today. It is a standing reproduction of the
//! defect, not a regression guard; run it explicitly with
//! `cargo test -p ferric-rpa --test mp2c_cks_published_sign -- --ignored --nocapture`.
//! When the defect is fixed these become passing tests and should move into the
//! weekly validation tier.
//!
//! Published MP2C corrections (Huang, PhD dissertation, UC Riverside 2015,
//! Table 2.1; the work behind Huang, Goldey, Head-Gordon & Beran, JCTC 10, 2054
//! (2014)), kcal/mol:
//!
//! ```text
//!   system            published corr   ferric (aug-cc-pVDZ, ALDAX, n_freq 16)
//!   methane dimer         −0.060              +0.1852
//!   ethene dimer          −0.040              −0.3348   (sign agrees; 8x too large)
//!   A24-12 / S22-16       +0.060              +0.3908
//! ```
//!
//! Only the correction is like-for-like: the published totals are MP2/CBS plus
//! a finite-basis correction built on asymptotically corrected KS orbitals,
//! while these are single-basis with RHF monomer orbitals. Those protocol
//! differences are candidate causes; they are not established as the cause.
//!
//! On every system one monomer's static Dyson matrix has a negative eigenvalue
//! (methane −0.038, ethene −0.401, A24-12 −1.975), so the coupled integrand
//! crosses a pole and the correction is not a defined quantity; the ethene
//! value flipped sign between two builds with a 1e-6 Å geometry difference.
//! Every test therefore also asserts a stable response (`M(0) > 0`), which
//! fails on all three today. Each run also prints the
//! smallest eigenvalue of the symmetrized Dyson matrix `I + Π⁰^{1/2} K Π⁰^{1/2}`
//! at ω = 0 and every quadrature node, for both monomers
//! ([`ferric_rpa::cks_dispersion::MonomerResponse::dyson_min_eigenvalue`]): a
//! value near zero means a node sits near a pole of the Dyson resolvent, and a
//! negative value means the RHF reference is unstable under the ALDA kernel.
//!
//! CP fragments are built here from the committed dimer files in the dimer's
//! own center order, which the Casimir–Polder trace requires.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::cks_dispersion::{build_monomer_response, CksCoupling, CksDispersionConfig};
use ferric_rpa::mp2c_cks::{mp2c_cks_interaction, MonomerInput, Mp2cCksConfig, Mp2cCksInteraction};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

const OBS: &str = "aug-cc-pvdz";
const AUX: &str = "aug-cc-pvdz-rifit";
const N_FREQ: usize = 16;

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    scf: ScfResult,
}

fn prepare(xyz: &str) -> Sys {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(OBS).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let scf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-9,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(scf.converged, "SCF did not converge");
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(AUX).unwrap()).unwrap();
    Sys {
        mol,
        obs,
        dfbs,
        scf,
    }
}

/// The dimer and its two CP monomers, all in the dimer's center order.
fn cp_triple(rel_path: &str, n_a: usize) -> [String; 3] {
    let p = format!(
        "{}/../../testdata/molecules/{rel_path}",
        env!("CARGO_MANIFEST_DIR")
    );
    let xyz = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {p}: {e}"));
    let atoms: Vec<&str> = xyz
        .lines()
        .skip(2)
        .filter(|l| !l.trim().is_empty())
        .collect();
    let member = |ghost: &dyn Fn(usize) -> bool| {
        let body: Vec<String> = atoms
            .iter()
            .enumerate()
            .map(|(i, l)| {
                if ghost(i) {
                    format!("@{}", l.trim_start())
                } else {
                    l.to_string()
                }
            })
            .collect();
        format!("{}\ncp member\n{}\n", atoms.len(), body.join("\n"))
    };
    [
        member(&|_| false),
        member(&|i| i >= n_a),
        member(&|i| i < n_a),
    ]
}

/// Print `min eig (I + Π⁰^{1/2} K Π⁰^{1/2})` per frequency for one monomer.
fn print_dyson_spectrum(label: &str, s: &Sys) -> f64 {
    let cfg = CksDispersionConfig {
        n_freq: N_FREQ,
        u0: 1.0,
        ..Default::default()
    };
    let r = build_monomer_response(&s.mol, &s.obs, &s.dfbs, &s.scf, "LDA_X", &cfg).unwrap();
    let (freqs, _) = ferric_rpa::quadrature::gauss_legendre_nodes(N_FREQ, 1.0);
    let mut lowest = f64::INFINITY;
    let mut row = String::new();
    for &w in std::iter::once(&0.0).chain(freqs.iter()) {
        let m = r
            .dyson_min_eigenvalue(w, CksCoupling::Cks)
            .unwrap()
            .unwrap();
        lowest = lowest.min(m);
        row.push_str(&format!(" {w:.3}:{m:.4}"));
    }
    eprintln!("  {label} min eig M(omega) [omega:min]:{row}");
    lowest
}

fn run(rel_path: &str, n_a: usize, label: &str, published: f64) -> (f64, f64) {
    let [d, a, b] = cp_triple(rel_path, n_a).map(|x| prepare(&x));
    let cfg = Mp2cCksConfig {
        xc: "LDA_X".to_string(),
        disp: CksDispersionConfig {
            n_freq: N_FREQ,
            u0: 1.0,
            frozen_core: 0,
            ..Default::default()
        },
        frozen_core: 0,
    };
    let r = mp2c_cks_interaction(
        &d.mol,
        &d.obs,
        &d.dfbs,
        &d.scf,
        &MonomerInput {
            mol: &a.mol,
            obs: &a.obs,
            dfbs: &a.dfbs,
            scf: &a.scf,
        },
        &MonomerInput {
            mol: &b.mol,
            obs: &b.obs,
            dfbs: &b.dfbs,
            scf: &b.scf,
        },
        &cfg,
    )
    .unwrap();
    let k = Mp2cCksInteraction::HARTREE_TO_KCAL;
    let corr = r.e_correction * k;
    eprintln!("=== {label} ({OBS}, ALDAX, n_freq={N_FREQ}), kcal/mol ===");
    eprintln!("  E_int(MP2)  = {:+.4}", r.e_int_mp2_kcal());
    eprintln!("  E_disp^UCHF = {:+.4}", r.e_disp_uncoupled * k);
    eprintln!("  E_disp^CKS  = {:+.4}", r.e_disp_cks * k);
    eprintln!("  correction  = {corr:+.4}   [published {published:+.3}]");
    let la = print_dyson_spectrum("monomer A", &a);
    let lb = print_dyson_spectrum("monomer B", &b);
    eprintln!("  lowest Dyson eigenvalue over the grid: A {la:.4}, B {lb:.4}");
    eprintln!(
        "  static_dyson_min_eigenvalue = {:.4}",
        r.static_dyson_min_eigenvalue
    );
    (corr, r.static_dyson_min_eigenvalue)
}

/// Methane dimer (S22-08). Published correction −0.060 kcal/mol.
#[test]
#[ignore = "known open: CKS MP2C correction sign vs published (methane +0.185 vs -0.060)"]
fn methane_dimer_correction_has_the_published_sign() {
    let (corr, m0) = run("s22/methane_dimer.xyz", 5, "methane dimer (S22-08)", -0.060);
    assert!(
        m0 > 0.0,
        "coupled response unstable: static Dyson min eig {m0:.4}"
    );
    assert!(
        corr < 0.0,
        "published correction is -0.060; ferric gives {corr:+.4}"
    );
}

/// Ethene dimer (S22-07). Published correction −0.040 kcal/mol.
#[test]
#[ignore = "known open: CKS MP2C correction vs published (ethene -0.335 vs -0.040, unstable response)"]
fn ethene_dimer_correction_has_the_published_sign() {
    let (corr, m0) = run("s22/ethylene_dimer.xyz", 6, "ethene dimer (S22-07)", -0.040);
    assert!(
        m0 > 0.0,
        "coupled response unstable: static Dyson min eig {m0:.4}"
    );
    assert!(
        corr < 0.0,
        "published correction is -0.040; ferric gives {corr:+.4}"
    );
}

/// A24-12 / S22-16 (ethene–ethyne). Published correction +0.060: the sign
/// agrees, the magnitude is 6.5x too large. Asserted to within a factor of 2.
#[test]
#[ignore = "known open: CKS MP2C correction magnitude vs published (A24-12 +0.391 vs +0.060)"]
fn a24_12_correction_has_the_published_magnitude() {
    let (corr, m0) = run("a24/a24-12_dimer.xyz", 4, "A24-12 ethene-ethyne", 0.060);
    assert!(
        m0 > 0.0,
        "coupled response unstable: static Dyson min eig {m0:.4}"
    );
    assert!(
        (corr / 0.060 - 1.0).abs() < 1.0,
        "published correction is +0.060; ferric gives {corr:+.4}"
    );
}
