//! Invariance properties every Hirshfeld charge implementation must satisfy.
//!
//! Symmetry-equivalent atoms must carry equal charges, and rigidly rotating or
//! translating a molecule must not move charge between its atoms. These are
//! properties of the partitioning, not of where an integration grid happens to
//! sit, so a quadrature that breaks them is wrong regardless of how plausible
//! its numbers look.
//!
//! Why this file exists. A 500-molecule QM9 dataset built in June 2026 had
//! Hirshfeld charges that violated all three: 52 molecules with
//! symmetry-equivalent atoms differing by more than 0.05 e, the worst a pair of
//! equivalent fluorines at -2.129 e and +0.243 e, and a centrosymmetric
//! molecule reporting a 21.56 D dipole where the density's own dipole is
//! 0.000 D. That dataset was generated with a binary that predated
//! d4e1615a ("wire ad-hoc same-basis proatom into ALL Hirshfeld paths"), so it
//! ran on the crude single-Slater fallback proatom, which has no core structure.
//! The current default path -- a same-basis SCF proatom -- does not reproduce
//! those numbers. Nothing here failed on main when it was written; the tests
//! are a guard, not a bug report, and they exercise the SCF-proatom path that
//! production actually uses.
//!
//!   cargo test -p ferric-rpa --test hirshfeld_symmetry --release

use ferric_core::basis;
use ferric_core::elements::z_to_symbol;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::properties::{hirshfeld_charges, spherically_averaged_proatom, RadialProatom};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Difluoroethyne F-C#C-F at its real geometry (C-F 1.28 A, C#C 1.20 A), so the
/// charges are those of an actual molecule rather than of two distant CF units.
/// D_inf_h: the fluorines are equivalent and so are the carbons.
fn fccf_xyz(offset: f64) -> String {
    let (cc, cf) = (1.20_f64, 1.28_f64);
    let z1 = cc / 2.0;
    let z2 = z1 + cf;
    format!(
        "4\nFCCF\nF 0.0 0.0 {:.6}\nC 0.0 0.0 {:.6}\nC 0.0 0.0 {:.6}\nF 0.0 0.0 {:.6}\n",
        -z2 + offset, -z1 + offset, z1 + offset, z2 + offset
    )
}

/// Rigidly rotate an xyz block so no symmetry axis lines up with a grid axis.
fn rotated(xyz: &str, theta: f64) -> String {
    let (s, c) = theta.sin_cos();
    // rotation about (1,1,1)/sqrt(3)
    let k = [1.0 / 3.0_f64.sqrt(); 3];
    let mut out = String::new();
    for (i, line) in xyz.lines().enumerate() {
        if i < 2 {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let v: [f64; 3] = [f[1].parse().unwrap(), f[2].parse().unwrap(), f[3].parse().unwrap()];
        let dot = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
        let cross = [
            k[1] * v[2] - k[2] * v[1],
            k[2] * v[0] - k[0] * v[2],
            k[0] * v[1] - k[1] * v[0],
        ];
        let r: Vec<f64> = (0..3)
            .map(|j| v[j] * c + cross[j] * s + k[j] * dot * (1.0 - c))
            .collect();
        out.push_str(&format!("{} {:.8} {:.8} {:.8}\n", f[0], r[0], r[1], r[2]));
    }
    out
}

/// Charges via the production path: a same-basis neutral SCF proatom, which is
/// what `ferric-cli` passes. The crude single-Slater fallback (`None`) has no
/// core structure and gets water's sign wrong, so testing only that path would
/// not validate what users actually run.
fn charges_scf_proatom(xyz: &str, basis_name: &str) -> Vec<f64> {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs_bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    // Match production (ferric-cli): radii out to 30 Bohr, not 15.
    let radii: Vec<f64> = (1..=600).map(|k| k as f64 * 0.05).collect();
    let gs_mult = |z: i32| -> usize { match z { 1 => 2, 6 => 3, 7 => 4, 8 => 3, 9 => 2, _ => 1 } };
    let bs = obs_bs.clone();
    let proatom = |z: i32, qi: i32| -> Option<RadialProatom> {
        if qi != 0 { return None; }
        let sym = z_to_symbol(z).unwrap_or("X");
        let amol = Molecule::parse_xyz(&format!("1\n{sym}\n{sym} 0 0 0\n"), 0, gs_mult(z)).ok()?;
        let aobs = PreparedBasis::new(&amol, &bs).ok()?;
        let abounds = SchwarzBounds::compute(op, &aobs).ok()?;
        let mut cfg = RhfConfig::default();
        let dens = if gs_mult(z) == 1 {
            solve_rhf(&ctx, &amol, &aobs, op, &abounds, &cfg).ok()?.density_r().to_owned()
        } else {
            cfg.mom_after_iter = 5;
            ferric_scf::uhf::solve_uhf(&ctx, &amol, &aobs, &abounds, &cfg)
                .ok()?
                .density_total()
                .to_owned()
        };
        spherically_averaged_proatom(z, &bs, &dens, &radii).ok()
    };
    let provider: &ferric_rpa::properties::ProatomProvider = &&proatom;
    hirshfeld_charges(&mol, &obs_bs, &rhf.density_r().to_owned(), Some(provider)).unwrap()
}

#[test]
fn equivalent_atoms_carry_equal_hirshfeld_charges() {
    let q = charges_scf_proatom(&fccf_xyz(0.0), "cc-pvdz");

    let df = (q[0] - q[3]).abs();
    let dc = (q[1] - q[2]).abs();
    assert!(df < 1e-6, "equivalent fluorines differ by {df:.3e} e (charges {q:?})");
    assert!(dc < 1e-6, "equivalent carbons differ by {dc:.3e} e (charges {q:?})");

    // Value pin. A symmetric but badly-scaled quadrature would still give equal
    // charges, so bound the magnitude against the known Hirshfeld character:
    // small, deformation-density charges, fluorine slightly negative.
    assert!(
        (-0.20..0.05).contains(&q[0]),
        "fluorine Hirshfeld charge {:.3} e is outside the expected small-magnitude range; \
         the old lattice gave -2.13 e on an equivalent fluorine",
        q[0]
    );
    assert!(q.iter().all(|v| v.abs() < 0.5), "implausible Hirshfeld magnitudes: {q:?}");
}

/// The grid is not reoriented to a standard frame and Lebedev rules are only
/// O_h-invariant, so a molecule whose axis does not lie along a grid axis keeps
/// a small residual anisotropy. It must stay far below the 0.05 e scale of the
/// bug this file exists to catch.
#[test]
fn equivalent_atoms_are_equal_in_a_rotated_frame() {
    let q = charges_scf_proatom(&rotated(&fccf_xyz(0.0), 0.7), "cc-pvdz");
    let df = (q[0] - q[3]).abs();
    let dc = (q[1] - q[2]).abs();
    assert!(df < 1e-3, "rotated frame: equivalent fluorines differ by {df:.3e} e ({q:?})");
    assert!(dc < 1e-3, "rotated frame: equivalent carbons differ by {dc:.3e} e ({q:?})");
}

/// Translating the molecule must not move charge between atoms. On the old
/// lattice this was the failure mode: the answer depended on where each nucleus
/// fell relative to a fixed grid.
#[test]
fn charges_are_invariant_to_translation() {
    let q0 = charges_scf_proatom(&fccf_xyz(0.0), "cc-pvdz");
    let q1 = charges_scf_proatom(&fccf_xyz(0.137), "cc-pvdz");
    let worst = q0.iter().zip(&q1).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    assert!(worst < 1e-3, "charges moved by {worst:.3e} e under a rigid translation: {q0:?} vs {q1:?}");
}

/// QM9 `gdb_133454` (O=C1C(F)=NON=C1F), the molecule whose stored dataset entry
/// carries -2.129 e on one fluorine and +0.243 e on the other. Its two fluorines
/// are related by the ring's mirror symmetry and must be near-equal. A real
/// molecule with atoms at general positions is a harder case than a linear one
/// lying along an axis, so it is worth pinning even though the current default
/// path handles it.
#[test]
fn the_qm9_regression_molecule_has_equal_fluorines() {
    let xyz = "9\ngdb_133454\n\
        F -0.009300 0.120900 0.226900\n\
        C -0.004200 1.419500 0.053100\n\
        N -1.146600 2.001200 0.018000\n\
        O -1.175100 3.357500 -0.162400\n\
        N -0.044200 4.114600 -0.306000\n\
        C 1.091100 3.519100 -0.268900\n\
        F 2.155300 4.270400 -0.409100\n\
        C 1.304600 2.076000 -0.083700\n\
        O 2.378400 1.521000 -0.049700\n";
    let q = charges_scf_proatom(xyz, "aug-cc-pvdz");

    // No atom in a neutral closed-shell CHNOF molecule should approach a full
    // electron of charge. The lattice gave -2.129 e here.
    let worst = q.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        worst < 0.6,
        "implausible Hirshfeld magnitude {worst:.3} e (charges {q:?}); \
         the June-2026 dataset entry for this molecule has -2.129 e on atom 0"
    );

    // The two fluorines (atoms 0 and 6) are near-equivalent by the ring mirror.
    let df = (q[0] - q[6]).abs();
    assert!(df < 0.1, "the two fluorines differ by {df:.3} e (charges {q:?})");
}
