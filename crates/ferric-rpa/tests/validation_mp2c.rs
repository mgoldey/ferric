//! ΔdRPA-corrected MP2 and the experimental CKS MP2C on A24 counterpoise
//! triples (weekly validation tier).
//!
//! Two constructions, both library-only:
//!
//! * [`ferric_rpa::mp2c`] — supermolecular ΔdRPA-corrected MP2: each system's
//!   direct-ring correlation `2·E_OS` swapped for dRPA (full range by
//!   default). Not published MP2C, which swaps only intermolecular SAPT
//!   dispersion terms; see that module's docs.
//! * [`ferric_rpa::mp2c_cks`] — the published SAPT-style construction with a
//!   coupled-Kohn–Sham (ALDA) Casimir–Polder dispersion. EXPERIMENTAL: its
//!   correction has an open sign discrepancy against the published table (see
//!   `mp2c_cks_published_sign.rs` and the module docs).
//!
//! The external reference is A24 CCSD(T)/CBS (Řezáč & Hobza, JCTC 9, 2151
//! (2013)). It is printed for orientation and **not** asserted against:
//! aug-cc-pVDZ / cc-pVDZ are far from the basis limit the reference is defined
//! at. What is asserted holds regardless of basis saturation: finite, bound
//! complexes stay bound, components sum to the total, the counterpoise
//! correction is real, and both CKS dispersion terms are attractive and
//! independent of which monomer is labelled A.
//!
//! Geometries are the committed A24 dimers in `testdata/molecules/a24/`; the CP
//! fragments are built here from each dimer in the dimer's own center order
//! (only the `@` ghost marker differs), which is the ordering the CKS
//! Casimir–Polder trace requires.
//!
//! # Measured (aug-cc-pVDZ / aug-cc-pVDZ-RI, CP, kcal/mol), ΔdRPA-corrected MP2
//!
//! ```text
//!     sys      MP2+dRPA         MP2   correction  CCSD(T)/CBS  class
//!     A24-01    -5.7340      -5.8277       0.0937      -6.5240  electrostatic
//!     A24-08    -0.3996      -0.5022       0.1026      -0.6720  electrostatic
//!     A24-19    -0.0856      -0.3721       0.2865      -0.5380  dispersion
//!     A24-20    -0.0394      -0.2615       0.2221      -0.4080  dispersion
//! ```
//!
//! The ΔdRPA-corrected result is further from the reference than MP2 on all
//! four at this basis. The corrections on the dispersion-bound pairs (+0.29,
//! +0.22) are several times the ~0.0x kcal/mol scale of published MP2C
//! corrections for comparable small complexes. One UNVERIFIED hypothesis for
//! the first observation is an error cancellation in MP2 (basis
//! incompleteness underbinding against uncoupled dispersion overbinding); it
//! has not been tested here.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::cks_dispersion::CksDispersionConfig;
use ferric_rpa::mp2c::{
    mp2c_energy, mp2c_interaction_energy, Mp2cConfig, Mp2cInteraction, Mp2cResult,
};
use ferric_rpa::mp2c_cks::{mp2c_cks_interaction, MonomerInput, Mp2cCksConfig, Mp2cCksInteraction};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

/// aug-cc-pVDZ, not cc-pVDZ: cc-pVDZ has no Ar (rules out A24-20), and without
/// diffuse functions A24-19 comes out unbound by MP2 as well as with the dRPA correction.
const A24_OBS: &str = "aug-cc-pvdz";
const A24_AUX: &str = "aug-cc-pvdz-rifit";

/// A24 CCSD(T)/CBS interaction energies (kcal/mol), Řezáč & Hobza 2013.
fn a24_reference(idx: usize) -> f64 {
    match idx {
        1 => -6.524,
        8 => -0.672,
        12 => -1.537,
        19 => -0.538,
        20 => -0.408,
        _ => panic!("no A24 reference tabulated here for system {idx}"),
    }
}

/// Number of atoms in fragment A (the leading block of the dimer file).
fn fragment_a_size(idx: usize) -> usize {
    match idx {
        1 => 3,
        8 | 19 | 20 => 5,
        12 => 4,
        _ => panic!("no fragment split recorded for A24-{idx:02}"),
    }
}

fn dimer_xyz(idx: usize) -> String {
    let p = format!(
        "{}/../../testdata/molecules/a24/a24-{idx:02}_dimer.xyz",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {p}: {e}"))
}

/// Which centers a CP-triple member keeps as real atoms.
#[derive(Clone, Copy)]
enum Member {
    Dimer,
    MonoA,
    MonoB,
}

/// Rewrite a dimer XYZ as one member of its CP triple, preserving the center
/// order and ghosting (`@`) the absent fragment.
fn cp_member(xyz: &str, n_a: usize, which: Member) -> String {
    let mut lines = xyz.lines();
    let n = lines.next().unwrap().trim().to_string();
    let comment = lines.next().unwrap().to_string();
    let atoms: Vec<&str> = lines.filter(|l| !l.trim().is_empty()).collect();
    let body: Vec<String> = atoms
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let ghost = match which {
                Member::Dimer => false,
                Member::MonoA => i >= n_a,
                Member::MonoB => i < n_a,
            };
            if ghost {
                format!("@{}", l.trim_start())
            } else {
                l.to_string()
            }
        })
        .collect();
    format!("{n}\n{comment}\n{}\n", body.join("\n"))
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    scf: ScfResult,
}

fn prepare(xyz: &str, obs_name: &str, aux_name: &str) -> Sys {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled(obs_name).unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let scf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        op,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(scf.converged, "SCF did not converge");
    let dfbs = PreparedBasis::new(&mol, &basis::bundled(aux_name).unwrap()).unwrap();
    Sys {
        mol,
        obs,
        dfbs,
        scf,
    }
}

fn mp2c_of(s: &Sys) -> Mp2cResult {
    mp2c_energy(&s.mol, &s.obs, &s.dfbs, &s.scf, &Mp2cConfig::default()).unwrap()
}

/// Full CP triple for A24 system `idx` with ΔdRPA-corrected MP2.
fn a24_mp2c_interaction(idx: usize) -> Mp2cInteraction {
    let xyz = dimer_xyz(idx);
    let n_a = fragment_a_size(idx);
    let [d, a, b] = [Member::Dimer, Member::MonoA, Member::MonoB]
        .map(|m| prepare(&cp_member(&xyz, n_a, m), A24_OBS, A24_AUX));
    let nd = d.obs.nbasis();
    assert_eq!(
        (a.obs.nbasis(), b.obs.nbasis()),
        (nd, nd),
        "A24-{idx:02}: CP fragments must span the dimer basis"
    );
    let [rd, ra, rb] = [&d, &a, &b].map(mp2c_of);
    assert!(
        rd.drpa_converged && ra.drpa_converged && rb.drpa_converged,
        "A24-{idx:02}: every dRPA eigensolve must have converged"
    );
    mp2c_interaction_energy(&rd, &ra, &rb)
}

/// ΔdRPA-corrected MP2 on four A24 CP triples spanning the binding spectrum.
#[test]
#[ignore = "validation: dRPA-corrected MP2 (mp2c) on A24 CP triples"]
fn a24_cp_triples_run_and_correction_sign_tracks_binding_class() {
    let systems: [(usize, bool); 4] = [(1, false), (8, false), (19, true), (20, true)];
    eprintln!(
        "\n{:<7} {:>10} {:>10} {:>11} {:>12}  class",
        "sys", "MP2+dRPA", "MP2", "correction", "CCSD(T)/CBS"
    );
    for (idx, dispersion_bound) in systems {
        let int = a24_mp2c_interaction(idx);
        let (mp2c, mp2) = (int.e_int_mp2c_kcal(), int.e_int_mp2_kcal());
        let corr = int.e_int_correction * Mp2cInteraction::HARTREE_TO_KCAL;
        let refv = a24_reference(idx);
        eprintln!(
            "A24-{idx:02} {mp2c:10.4} {mp2:10.4} {corr:11.4} {refv:12.4}  {}",
            if dispersion_bound {
                "dispersion"
            } else {
                "electrostatic"
            }
        );
        assert!(
            mp2c.is_finite(),
            "A24-{idx:02}: MP2C interaction must be finite"
        );
        assert!(
            int.components_sum_to_total().abs() < 1e-12,
            "A24-{idx:02}: interaction components must sum to the total (residual {:.2e} Ha)",
            int.components_sum_to_total()
        );
        assert!(
            mp2 < 0.0,
            "A24-{idx:02}: MP2 must find this bound in {A24_OBS}, got {mp2:.4}"
        );
        assert!(
            mp2c < 0.0,
            "A24-{idx:02}: MP2C must not flip a bound complex, got {mp2c:.4}"
        );
        assert!(
            (mp2c - mp2).abs() < 0.5 * mp2.abs().max(1.0),
            "A24-{idx:02}: MP2C ({mp2c:.4}) diverged from MP2 ({mp2:.4}) beyond correction scale"
        );
        // Coupling screens the uncoupled dispersion of a dispersion-bound pair.
        if dispersion_bound {
            assert!(
                corr > 0.0,
                "A24-{idx:02}: dRPA coupling must screen; got {corr:.4} kcal/mol"
            );
        }
    }
}

/// The counterpoise correction must be real: monomers in their own basis bind
/// more than in the dimer basis (A24-20, methane–Ar, the smallest system).
#[test]
#[ignore = "validation: dRPA-corrected MP2 (mp2c) on A24 CP triples"]
fn counterpoise_correction_is_nonzero_and_reduces_binding() {
    let idx = 20usize;
    let cp = a24_mp2c_interaction(idx);

    let xyz = dimer_xyz(idx);
    let n_a = fragment_a_size(idx);
    let atoms: Vec<&str> = xyz
        .lines()
        .skip(2)
        .filter(|l| !l.trim().is_empty())
        .collect();
    let own = |range: std::ops::Range<usize>| {
        let a = &atoms[range];
        format!("{}\nown basis\n{}\n", a.len(), a.join("\n"))
    };
    let d = prepare(&xyz, A24_OBS, A24_AUX);
    let ma = prepare(&own(0..n_a), A24_OBS, A24_AUX);
    let mb = prepare(&own(n_a..atoms.len()), A24_OBS, A24_AUX);
    let uncp = mp2c_interaction_energy(&mp2c_of(&d), &mp2c_of(&ma), &mp2c_of(&mb));

    let bsse = uncp.e_int_mp2c_kcal() - cp.e_int_mp2c_kcal();
    eprintln!(
        "A24-{idx:02} MP2C: CP {:.4}  unCP {:.4}  BSSE {bsse:.4} kcal/mol  [ref {:.3}]",
        cp.e_int_mp2c_kcal(),
        uncp.e_int_mp2c_kcal(),
        a24_reference(idx)
    );
    assert!(
        bsse < 0.0,
        "BSSE must make the un-CP interaction more binding (got {bsse:.4})"
    );
    assert!(
        bsse.abs() > 1e-3,
        "BSSE {bsse:.2e} kcal/mol: the ghosts contribute nothing"
    );
}

fn cks_interaction(idx: usize, obs: &str, aux: &str, swap: bool) -> Mp2cCksInteraction {
    let xyz = dimer_xyz(idx);
    let n_a = fragment_a_size(idx);
    let [d, a, b] = [Member::Dimer, Member::MonoA, Member::MonoB]
        .map(|m| prepare(&cp_member(&xyz, n_a, m), obs, aux));
    let cfg = Mp2cCksConfig {
        xc: "LDA".to_string(),
        disp: CksDispersionConfig {
            n_freq: 16,
            u0: 1.0,
            frozen_core: 0,
            ..Default::default()
        },
        frozen_core: 0,
    };
    let ia = MonomerInput {
        mol: &a.mol,
        obs: &a.obs,
        dfbs: &a.dfbs,
        scf: &a.scf,
    };
    let ib = MonomerInput {
        mol: &b.mol,
        obs: &b.obs,
        dfbs: &b.dfbs,
        scf: &b.scf,
    };
    let (first, second) = if swap { (&ib, &ia) } else { (&ia, &ib) };
    mp2c_cks_interaction(&d.mol, &d.obs, &d.dfbs, &d.scf, first, second, &cfg).unwrap()
}

/// CKS MP2C on A24-12 (ethene–ethyne) at cc-pVDZ: structural properties only.
///
/// Measured: E_int(MP2) −1.053, E_disp^UCHF −1.019, E_disp^CKS −0.440,
/// correction +0.579, E_int(MP2C) −0.475 kcal/mol (CCSD(T)/CBS −1.537). The
/// correction's SIGN is not asserted, nor is the coupled response's
/// stability: both are the open defect documented in `ferric_rpa::mp2c_cks`.
#[test]
#[ignore = "validation: MP2C with CKS dispersion (experimental) on A24-12"]
fn mp2c_cks_runs_on_a24_12_and_is_order_invariant() {
    let r = cks_interaction(12, "cc-pvdz", "cc-pvdz-ri", false);
    let k = Mp2cCksInteraction::HARTREE_TO_KCAL;
    eprintln!("A24-12 ethene-ethyne, cc-pVDZ/cc-pVDZ-RI, ALDA CKS (kcal/mol)");
    eprintln!("  E_int(SCF)   = {:+.4}", r.e_int_scf * k);
    eprintln!("  E_int(MP2)   = {:+.4}", r.e_int_mp2_kcal());
    eprintln!("  E_disp^UCHF  = {:+.4}", r.e_disp_uncoupled * k);
    eprintln!("  E_disp^CKS   = {:+.4}", r.e_disp_cks * k);
    eprintln!("  correction   = {:+.4}", r.e_correction * k);
    eprintln!(
        "  E_int(MP2C)  = {:+.4}   [CCSD(T)/CBS {:+.3}]",
        r.e_int_mp2c_kcal(),
        a24_reference(12)
    );

    eprintln!(
        "  static Dyson min eig = {:.4} (<= 0: CKS term not meaningful)",
        r.static_dyson_min_eigenvalue
    );
    assert!(r.e_disp_uncoupled < 0.0, "E_disp^UCHF must be attractive");
    assert!(r.e_disp_cks < 0.0, "E_disp^CKS must be attractive");
    // Physically sized: a misaligned trace gives hundreds of kcal/mol here.
    assert!(
        (r.e_disp_uncoupled * k).abs() < 5.0 && (r.e_disp_cks * k).abs() < 5.0,
        "dispersion terms are not interaction-sized"
    );
    assert!(
        (r.e_int_mp2c - (r.e_int_mp2 + r.e_correction)).abs() < 1e-12,
        "MP2C interaction must equal MP2 interaction + correction"
    );

    let s = cks_interaction(12, "cc-pvdz", "cc-pvdz-ri", true);
    let d = (r.e_int_mp2c - s.e_int_mp2c).abs();
    eprintln!("  monomer swap: |Delta E_int(MP2C)| = {d:.2e} Ha");
    assert!(
        d < 1e-10,
        "MP2C interaction depends on which monomer is A: {d:.3e} Ha"
    );
}
