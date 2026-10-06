//! `optimize_qmmm` across boundary schemes, SCF references and free-atom sets:
//! stationarity at the returned geometry and finite-difference correctness of
//! the full QM/MM gradient, with negative controls.
//!
//! Extends `qmmm_gradient_at_minimum.rs` (RHF × RCD × `MoveMm::All`, capped
//! ethane, STO-3G) to:
//!
//! | case | system | method | scheme | free set | basis |
//! |---|---|---|---|---|---|
//! | `rhf_z1` | capped ethane | RHF | DeleteHost (Z1) | All | STO-3G |
//! | `rhf_rc` | capped ethane | RHF | RC | All | STO-3G |
//! | `rhf_keep` | capped ethane | RHF | Keep | All | STO-3G |
//! | `rhf_z1_631g` | capped ethane | RHF | Z1 | All | 6-31G |
//! | `uhf_z1`, `uhf_rc` | ethyl radical, QM = CH2• (capped CH3•, doublet) | UHF | Z1, RC | All | STO-3G |
//! | `rks_rcd_within` | capped ethane, C–C stretched at the start | RKS/PBE | RCD | WithinRadius | STO-3G |
//! | `uks_rcd_within` | ethyl radical, C–C stretched at the start | UKS/PBE | RCD | WithinRadius | STO-3G |
//! | `rhf_rcd_residues` | capped ethane + 2 TIP3P-charged waters, residue-tagged (`WithinRadiusWholeResidues`) | RHF | RCD | Residues (MM CH3 free, waters frozen) | STO-3G |
//!
//! **No external reference exists for this surface.** A QM/MM potential
//! energy surface is defined by the partition, the link scale and the boundary
//! scheme, none of which is standardized, so no other code computes the same
//! energy. Every check here is internal and two-path: the analytic
//! `full_gradient_with_mm` against a central finite difference of the energy
//! `optimize_qmmm` minimizes (SCF in the embedding + `qmmm_mm_terms`), with the
//! partition REBUILT at every displaced geometry. The two paths share the SCF,
//! the integrals, the DF fit and the XC grid; they do not share the chain rule
//! (link `(1−g)/g`, midpoint ½/½), the `mm_forces` contraction or the analytic
//! SCF gradient. A construction error in the partition itself (a wrong
//! midpoint charge, a host charge that is not deleted) is INVISIBLE to that
//! comparison, because both sides see the same wrong partition. Those are
//! guarded separately: the structural assertions in [`assert_partition`] and
//! the scheme-swap negative controls.
//!
//! The harness reproduces `optimize_qmmm`'s step closure exactly (same
//! `RhfConfig` fields, `def2-universal-jkfit` DF-J/K for Rks/Uks, same
//! gradient function per method) rather than re-deriving it; see [`evaluate`].
//!
//! Hypotheses were registered before measurement in
//! `HYPOTHESES-optimize-qmmm-boundaries.md`; measured values sit beside every
//! bar and the mutation ledger is the trailing table.

use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mm::{Angle, Bond, LjParams, MmTopology, Torsion};
use ferric_scf::gradient::{rhf_gradient, uhf_gradient};
use ferric_scf::ks_gradient::{ks_gradient_closed, ks_gradient_uks};
use ferric_scf::optimize::{optimize_geometry, optimize_geometry_uhf, OptimizeConfig};
use ferric_scf::qmmm::{
    full_gradient_with_mm, mm_forces, optimize_qmmm, qmmm_mm_terms, BoundaryChargeScheme, MoveMm,
    QmSelection, QmmmAtom, QmmmMethod, QmmmOptimizeConfig, QmmmSystem, DEFAULT_LINK_SCALE,
};
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

const ANG2BOHR: f64 = 1.0 / 0.529_177_210_92;
const KCAL: f64 = 1.0 / 627.509_474;
const HS: [f64; 3] = [5.0e-5, 1.0e-4, 2.0e-4];
const JKFIT: &str = "def2-universal-jkfit";

// ---------------------------------------------------------------------------
// Bars. Each is ~10x the worst MEASURED value of its quantity (measured value
// beside it). Set from the measurement run recorded in the trailing table.
// ---------------------------------------------------------------------------

/// Stationarity bar (Ha/Bohr) over the FREE rows at the returned geometry,
/// applied to both the analytic rows and the FD derivative of the energy on
/// free atoms. The optimizer is asked for g_max < 1e-4 and stops well below
/// it; each bar is ~10x the worst measured value (in the comment), so it pins
/// the observed stop rather than restating the request.
fn stationarity_bar(case: &Case) -> f64 {
    if measuring() {
        return 1.0e-4;
    }
    match case.name {
        "rhf_keep" => 4.3e-5,       // measured 4.30e-6 (MM host x, FD)
        "rhf_z1" => 3.3e-5,         // measured 3.30e-6 (MM host z, FD)
        "rhf_rc" => 2.8e-6,         // measured 2.84e-7 (MM host z, FD)
        "rhf_z1_631g" => 8.3e-6,    // measured 8.27e-7 (frontier)
        "uhf_z1" => 1.2e-5,         // measured 1.19e-6 (frontier)
        "uhf_rc" => 1.7e-5,         // measured 1.70e-6 (M2 y, FD)
        "rks_rcd_within" => 2.0e-5, // measured 1.97e-6 (QM H)
        "uks_rcd_within" => 5.3e-5, // measured 5.31e-6 (MM host z, FD)
        other => panic!("no measured stationarity bar for {other}"),
    }
}

/// Absolute FD-vs-analytic bar at the minimum (Ha/Bohr). The residual there is
/// SCF-convergence noise over the 2h divisor (flat or falling with h).
fn fd_min_bar(case: &Case) -> f64 {
    if measuring() {
        return 1.0e-6;
    }
    match case.name {
        "rhf_keep" => 1.0e-7,          // measured 1.02e-8
        "rhf_z1" => 1.0e-7,            // measured 1.04e-8
        "rhf_rc" => 2.4e-7,            // measured 2.39e-8
        "rhf_z1_631g" => 1.1e-7,       // measured 1.11e-8
        "uhf_z1" | "uhf_rc" => 1.0e-7, // measured 1.03e-8 both
        "rks_rcd_within" => 1.0e-7,    // measured 1.00e-8 (PBE, DF-J/K, grid)
        "uks_rcd_within" => 1.1e-7,    // measured 1.06e-8 (PBE, DF-J/K, grid)
        other => panic!("no measured at-minimum FD bar for {other}"),
    }
}

/// Relative FD-vs-analytic bar off the minimum (every probed row is
/// O(1e-2) Ha/Bohr there). The residual grows ~h^2 (FD truncation) except on
/// 6-31G, where the h = 5e-5 point is SCF-noise limited.
fn fd_off_rel_bar(case: &Case) -> f64 {
    if measuring() {
        return 1.0e-4;
    }
    match case.name {
        "rhf_keep" => 9.5e-7,         // measured 9.54e-8 (QM H z, h=2e-4)
        "rhf_z1" => 9.7e-7,           // measured 9.66e-8 (QM H z, h=2e-4)
        "rhf_rc" => 6.7e-7,           // measured 6.69e-8 (QM H z, h=2e-4)
        "rhf_z1_631g" => 7.0e-6,      // measured 6.95e-7 (QM H z, h=1e-4: SCF noise)
        "uhf_z1" => 5.5e-7,           // measured 5.51e-8
        "uhf_rc" => 6.0e-7,           // measured 5.92e-8
        "rks_rcd_within" => 8.8e-7,   // measured 8.84e-8
        "rhf_rcd_residues" => 2.0e-6, // measured 5.3e-8 (h<=1e-4, QM H z); 7.5e-8 water rows
        "uks_rcd_within" => 6.3e-7,   // measured 6.29e-8 (frozen M2 z)
        other => panic!("no measured off-minimum FD bar for {other}"),
    }
}

/// `FERRIC_MEASURE_BARS=1` replaces every bar by a loose one (the optimizer's
/// own 1e-4 request; 1e-6 / 1e-4 for FD) so a new case can be measured before
/// its bars exist. Never set in CI.
fn measuring() -> bool {
    std::env::var_os("FERRIC_MEASURE_BARS").is_some()
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Selection {
    Indices(Vec<usize>),
    WholeResidues {
        seeds: Vec<usize>,
        radius: f64,
        residue_ids: Vec<usize>,
    },
}

#[derive(Clone)]
struct Case {
    name: &'static str,
    /// Start geometry (Bohr).
    atoms: Vec<QmmmAtom>,
    /// Full bond list (cut bond included); feeds link placement, boundary
    /// charges and the MM topology.
    bonds: Vec<(usize, usize)>,
    selection: Selection,
    qm_charge: i32,
    qm_mult: usize,
    scheme: BoundaryChargeScheme,
    method: QmmmMethod,
    move_mm: MoveMm,
    basis: &'static str,
    top: MmTopology,
}

fn h_tetra(cx: f64, cz: f64, ch: f64, dir: f64, phi: f64, q: f64) -> QmmmAtom {
    // A hydrogen on a tetrahedral carbon at (cx, 0, cz), pointing along
    // `dir` (±1) in z, azimuth `phi`.
    let theta = 109.5_f64.to_radians();
    let (s, c) = (theta.sin(), theta.cos());
    QmmmAtom::new(
        "H",
        1,
        cx + ch * s * phi.cos(),
        ch * s * phi.sin(),
        cz - dir * ch * c,
        q,
    )
}

/// Capped ethane, same charges and bond geometry as
/// `qmmm_gradient_at_minimum.rs` but STAGGERED (that file's eclipsed start is
/// a torsional saddle of the force field, held by symmetry): 0 C0 (QM
/// frontier), 1 C1 (MM host), 2/4/6 QM H, 3/5/7 MM H (M2). `stretch` (Bohr)
/// shifts the whole MM CH3 group along +z.
fn ethane_atoms(stretch: f64) -> Vec<QmmmAtom> {
    let cc = 1.53 * ANG2BOHR;
    let ch = 1.09 * ANG2BOHR;
    let mut atoms = vec![
        QmmmAtom::new("C", 6, 0.0, 0.0, 0.0, -0.1),
        QmmmAtom::new("C", 6, 0.0, 0.0, cc + stretch, -0.1),
    ];
    for k in 0..3 {
        let phi = 2.0 * std::f64::consts::PI * (k as f64) / 3.0;
        atoms.push(h_tetra(0.0, 0.0, ch, -1.0, phi, 0.033));
        let stagger = std::f64::consts::PI / 3.0;
        atoms.push(h_tetra(0.0, cc + stretch, ch, 1.0, phi + stagger, 0.033));
    }
    atoms
}

fn ethane_bonds() -> Vec<(usize, usize)> {
    vec![(0, 1), (0, 2), (0, 4), (0, 6), (1, 3), (1, 5), (1, 7)]
}

/// Ethyl radical CH2•–CH3: 0 C0 (radical carbon, QM frontier), 1 C1 (MM
/// host), 2/4 QM H on C0 (planar sp2), 3/5/6 MM H on C1 (M2, staggered).
/// QM = {0, 2, 4}; the capped QM molecule is CH3•, a doublet.
fn ethyl_atoms(stretch: f64) -> Vec<QmmmAtom> {
    let cc = 1.49 * ANG2BOHR;
    let ch = 1.08 * ANG2BOHR;
    let a = 120.0_f64.to_radians();
    let mut atoms = vec![
        QmmmAtom::new("C", 6, 0.0, 0.0, 0.0, -0.06),
        QmmmAtom::new("C", 6, 0.0, 0.0, cc + stretch, -0.1),
        QmmmAtom::new("H", 1, ch * a.sin(), 0.0, ch * a.cos(), 0.03),
    ];
    let ch3 = 1.09 * ANG2BOHR;
    let phis = [90.0_f64, 210.0, 330.0].map(f64::to_radians);
    atoms.push(h_tetra(0.0, cc + stretch, ch3, 1.0, phis[0], 0.033));
    atoms.push(QmmmAtom::new(
        "H",
        1,
        -ch * a.sin(),
        0.0,
        ch * a.cos(),
        0.03,
    ));
    atoms.push(h_tetra(0.0, cc + stretch, ch3, 1.0, phis[1], 0.033));
    atoms.push(h_tetra(0.0, cc + stretch, ch3, 1.0, phis[2], 0.033));
    atoms
}

fn ethyl_bonds() -> Vec<(usize, usize)> {
    vec![(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (1, 6)]
}

/// Capped ethane (indices 0..8 as in [`ethane_atoms`]) plus two TIP3P-charged
/// waters: 8-10 water A beyond the QM CH3 end, 11-13 water B beyond the MM CH3
/// end. Residues: QM CH3 = 0, MM CH3 = 1, water A = 2, water B = 3.
fn ethane_waters_atoms() -> Vec<QmmmAtom> {
    let cc = 1.53 * ANG2BOHR;
    let mut atoms = ethane_atoms(0.0);
    let roh = 0.9572 * ANG2BOHR;
    let half = (104.52_f64 / 2.0).to_radians();
    for (oz, dir) in [(-3.4 * ANG2BOHR, -1.0), (cc + 3.4 * ANG2BOHR, 1.0)] {
        atoms.push(QmmmAtom::new("O", 8, 0.0, 0.0, oz, -0.834));
        atoms.push(QmmmAtom::new(
            "H",
            1,
            roh * half.sin(),
            0.0,
            oz + dir * roh * half.cos(),
            0.417,
        ));
        atoms.push(QmmmAtom::new(
            "H",
            1,
            -roh * half.sin(),
            0.0,
            oz + dir * roh * half.cos(),
            0.417,
        ));
    }
    atoms
}

fn ethane_waters_bonds() -> Vec<(usize, usize)> {
    let mut b = ethane_bonds();
    b.extend([(8, 9), (8, 10), (11, 12), (11, 13)]);
    b
}

fn dist(a: &QmmmAtom, b: &QmmmAtom) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z_pos - b.z_pos).powi(2)).sqrt()
}

fn angle(a: &QmmmAtom, j: &QmmmAtom, b: &QmmmAtom) -> f64 {
    let u = [a.x - j.x, a.y - j.y, a.z_pos - j.z_pos];
    let v = [b.x - j.x, b.y - j.y, b.z_pos - j.z_pos];
    let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    (dot / (dist(a, j) * dist(b, j))).clamp(-1.0, 1.0).acos()
}

/// A generic AMBER-form topology whose bond lengths and angles are those of
/// `reference` (so the force field alone is at rest there), with every
/// i-j-k angle and i-j-k-l torsion generated from `bonds`. The parameters are
/// made up but generic, like `qmmm_gradient_at_minimum.rs`'s.
fn topology(
    reference: &[QmmmAtom],
    bonds: &[(usize, usize)],
    lj: impl Fn(&QmmmAtom) -> LjParams,
) -> MmTopology {
    let n = reference.len();
    let mut nbr: Vec<Vec<usize>> = vec![vec![]; n];
    let mut mm_bonds = vec![];
    for &(i, j) in bonds {
        nbr[i].push(j);
        nbr[j].push(i);
        let heavy = reference[i].z > 1 && reference[j].z > 1;
        mm_bonds.push(Bond {
            i,
            j,
            k: if heavy { 0.35 } else { 0.4 },
            r0: dist(&reference[i], &reference[j]),
        });
    }
    let mut angles = vec![];
    for (j, nb) in nbr.iter().enumerate() {
        for a in 0..nb.len() {
            for b in (a + 1)..nb.len() {
                let (i, k) = (nb[a], nb[b]);
                angles.push(Angle {
                    i,
                    j,
                    k,
                    k_theta: 0.06,
                    theta0: angle(&reference[i], &reference[j], &reference[k]),
                });
            }
        }
    }
    let mut torsions = vec![];
    for &(j, k) in bonds {
        for &i in &nbr[j] {
            if i == k {
                continue;
            }
            for &l in &nbr[k] {
                if l == j || l == i {
                    continue;
                }
                torsions.push(Torsion {
                    i,
                    j,
                    k,
                    l,
                    periodicity: 3,
                    k_phi: 0.02,
                    phase: 0.0,
                });
            }
        }
    }
    let charges = reference.iter().map(|a| a.charge).collect();
    let lj = reference.iter().map(lj).collect();
    MmTopology::new(charges, lj, mm_bonds, angles, torsions).unwrap()
}

/// Realistic AMBER-like LJ (parm99 CT / HC): carbon sigma 3.40 A, hydrogen
/// 2.65 A. Every QM–MM pair in these systems is 1-2, 1-3 or 1-4 through the
/// bond graph, so `qmmm_mm_terms` excludes or 1-4-scales all of them.
fn lj_realistic(a: &QmmmAtom) -> LjParams {
    match a.z {
        6 => LjParams {
            sigma: 3.3997 * ANG2BOHR,
            epsilon: 0.1094 * KCAL,
        },
        _ => LjParams {
            sigma: 2.6495 * ANG2BOHR,
            epsilon: 0.0157 * KCAL,
        },
    }
}

/// For the water system: TIP3P oxygen, a water hydrogen large enough that a
/// free water cannot collapse onto a charge, and the realistic ethane LJ of
/// [`lj_realistic`].
fn lj_waters(a: &QmmmAtom) -> LjParams {
    match (a.z, a.charge) {
        (8, _) => LjParams {
            sigma: 3.1507 * ANG2BOHR,
            epsilon: 0.1521 * KCAL,
        },
        (1, q) if q > 0.4 => LjParams {
            sigma: 1.0 * ANG2BOHR,
            epsilon: 0.046 * KCAL,
        },
        _ => lj_realistic(a),
    }
}

fn ethane_case(
    name: &'static str,
    scheme: BoundaryChargeScheme,
    method: QmmmMethod,
    move_mm: MoveMm,
    basis: &'static str,
    stretch: f64,
) -> Case {
    let bonds = ethane_bonds();
    Case {
        name,
        atoms: ethane_atoms(stretch),
        top: topology(&ethane_atoms(0.0), &bonds, lj_realistic),
        bonds,
        selection: Selection::Indices(vec![0, 2, 4, 6]),
        qm_charge: 0,
        qm_mult: 1,
        scheme,
        method,
        move_mm,
        basis,
    }
}

fn ethyl_case(
    name: &'static str,
    scheme: BoundaryChargeScheme,
    method: QmmmMethod,
    move_mm: MoveMm,
    stretch: f64,
) -> Case {
    let bonds = ethyl_bonds();
    Case {
        name,
        atoms: ethyl_atoms(stretch),
        top: topology(&ethyl_atoms(0.0), &bonds, lj_realistic),
        bonds,
        selection: Selection::Indices(vec![0, 2, 4]),
        qm_charge: 0,
        qm_mult: 2,
        scheme,
        method,
        move_mm,
        basis: "sto-3g",
    }
}

/// Start-geometry C–C stretch (Bohr) of the WithinRadius cases, and the
/// radius. The stretch makes the relaxed QM region move TOWARD the frozen M2
/// hydrogens, so the M2 set lies outside `r` at the start and inside it at the
/// end: a per-step re-evaluation of the free set would change membership (see
/// the reachability assertion in [`run_within_radius`]).
const WITHIN_STRETCH: f64 = 0.3;
const WITHIN_RADIUS_RKS: f64 = 4.2;
const WITHIN_RADIUS_UKS: f64 = 4.15;

fn case(name: &str) -> Case {
    use BoundaryChargeScheme::*;
    match name {
        "rhf_z1" => ethane_case(
            "rhf_z1",
            DeleteHost,
            QmmmMethod::Rhf,
            MoveMm::All,
            "sto-3g",
            0.0,
        ),
        "rhf_rc" => ethane_case(
            "rhf_rc",
            RedistributedCharge,
            QmmmMethod::Rhf,
            MoveMm::All,
            "sto-3g",
            0.0,
        ),
        "rhf_keep" => ethane_case(
            "rhf_keep",
            Keep,
            QmmmMethod::Rhf,
            MoveMm::All,
            "sto-3g",
            0.0,
        ),
        "rhf_z1_631g" => ethane_case(
            "rhf_z1_631g",
            DeleteHost,
            QmmmMethod::Rhf,
            MoveMm::All,
            "6-31g",
            0.0,
        ),
        "uhf_z1" => ethyl_case("uhf_z1", DeleteHost, QmmmMethod::Uhf, MoveMm::All, 0.0),
        "uhf_rc" => ethyl_case(
            "uhf_rc",
            RedistributedCharge,
            QmmmMethod::Uhf,
            MoveMm::All,
            0.0,
        ),
        "rks_rcd_within" => ethane_case(
            "rks_rcd_within",
            RedistributedChargeDipole,
            QmmmMethod::Rks("PBE".into()),
            MoveMm::WithinRadius(WITHIN_RADIUS_RKS),
            "sto-3g",
            WITHIN_STRETCH,
        ),
        "uks_rcd_within" => ethyl_case(
            "uks_rcd_within",
            RedistributedChargeDipole,
            QmmmMethod::Uks("PBE".into()),
            MoveMm::WithinRadius(WITHIN_RADIUS_UKS),
            WITHIN_STRETCH,
        ),
        "rhf_rcd_residues" => {
            let bonds = ethane_waters_bonds();
            let atoms = ethane_waters_atoms();
            Case {
                name: "rhf_rcd_residues",
                top: topology(&atoms, &bonds, lj_waters),
                atoms,
                bonds,
                selection: Selection::WholeResidues {
                    seeds: vec![0],
                    radius: 0.5,
                    residue_ids: vec![0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 3, 3, 3],
                },
                qm_charge: 0,
                qm_mult: 1,
                scheme: RedistributedChargeDipole,
                method: QmmmMethod::Rhf,
                move_mm: MoveMm::Residues(vec![1]),
                basis: "sto-3g",
            }
        }
        other => panic!("unknown case {other}"),
    }
}

/// The partition at `atoms`, rebuilt from scratch: every FD displacement goes
/// through this, so link atoms and boundary charges are re-derived at the
/// displaced geometry.
fn build_system_with(case: &Case, atoms: &[QmmmAtom], scheme: BoundaryChargeScheme) -> QmmmSystem {
    let sel = match &case.selection {
        Selection::Indices(v) => QmSelection::Indices(v.clone()),
        Selection::WholeResidues {
            seeds,
            radius,
            residue_ids,
        } => QmSelection::WithinRadiusWholeResidues {
            seeds: seeds.clone(),
            radius: *radius,
            residue_ids: residue_ids.clone(),
        },
    };
    QmmmSystem::new(atoms, sel, case.qm_charge, case.qm_mult)
        .unwrap()
        .with_link_atoms(&case.bonds, DEFAULT_LINK_SCALE)
        .unwrap()
        .with_boundary_charges(&case.bonds, scheme)
        .unwrap()
}

fn build_system(case: &Case, atoms: &[QmmmAtom]) -> QmmmSystem {
    build_system_with(case, atoms, case.scheme)
}

/// `cfg.scf` handed to `optimize_qmmm` (and reproduced by [`evaluate`]).
fn base_scf() -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-10,
        max_iter: 300,
        ..Default::default()
    }
}

fn coords_of(atoms: &[QmmmAtom]) -> Array2<f64> {
    let mut c = Array2::<f64>::zeros((atoms.len(), 3));
    for (i, a) in atoms.iter().enumerate() {
        c[(i, 0)] = a.x;
        c[(i, 1)] = a.y;
        c[(i, 2)] = a.z_pos;
    }
    c
}

fn displaced(atoms: &[QmmmAtom], idx: usize, axis: usize, delta: f64) -> Vec<QmmmAtom> {
    let mut out = atoms.to_vec();
    match axis {
        0 => out[idx].x += delta,
        1 => out[idx].y += delta,
        _ => out[idx].z_pos += delta,
    }
    out
}

/// ⟨S²⟩ of an unrestricted determinant.
fn s_squared(res: &ScfResult, s: &Array2<f64>, na: usize, nb: usize) -> f64 {
    let sz = 0.5 * (na as f64 - nb as f64);
    let ca = res.mos_alpha.slice(ndarray::s![.., ..na]);
    let cb = res.mos_beta.as_ref().unwrap().slice(ndarray::s![.., ..nb]);
    let ov = ca.t().dot(s).dot(&cb);
    sz * (sz + 1.0) + nb as f64 - ov.iter().map(|v| v * v).sum::<f64>()
}

struct Eval {
    energy: f64,
    grad: Option<Array2<f64>>,
    /// `mm_forces` in `mm_charge_positions` order (only with a gradient).
    forces: Vec<[f64; 3]>,
    sys: QmmmSystem,
    s2: Option<f64>,
    stability: Option<StabilityVerdict>,
}

/// One evaluation of the energy `optimize_qmmm` minimizes, and (optionally)
/// its full gradient, reproducing the optimizer's step closure line for line:
/// `cfg.scf` clone, `external_potential` from the rebuilt partition, Rks/Uks
/// set `xc` and `df_j_aux = df_k_aux = def2-universal-jkfit`, then
/// `rhf_gradient` / `uhf_gradient` / `ks_gradient_closed` / `ks_gradient_uks`
/// with the embedding `ext`, `mm_forces` on the total density and
/// `full_gradient_with_mm`. `check_stability` only adds a report; it does not
/// change the SCF solution (descent stays off).
fn evaluate_with(
    case: &Case,
    atoms: &[QmmmAtom],
    scheme: BoundaryChargeScheme,
    want_grad: bool,
    check_stability: bool,
) -> Eval {
    let ctx = ParallelContext::default();
    let sys = build_system_with(case, atoms, scheme);
    let mol = sys.to_qm_molecule();
    let bs = ferric_core::basis::bundled(case.basis).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut scf_cfg = base_scf();
    scf_cfg.external_potential = sys.to_external_potential();
    scf_cfg.check_stability = check_stability;
    let ext = scf_cfg.external_potential.clone();

    let (r, g) = match &case.method {
        QmmmMethod::Rhf => {
            let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &scf_cfg).unwrap();
            let g = want_grad
                .then(|| rhf_gradient(&mol, &prep, op, &bounds, &r, ext.as_ref()).unwrap());
            (r, g)
        }
        QmmmMethod::Uhf => {
            let r = solve_uhf(&ctx, &mol, &prep, &bounds, &scf_cfg).unwrap();
            let g = want_grad
                .then(|| uhf_gradient(&mol, &prep, op, &bounds, &r, ext.as_ref()).unwrap());
            (r, g)
        }
        QmmmMethod::Rks(xc) => {
            scf_cfg.xc = Some(xc.clone());
            scf_cfg.df_j_aux = Some(JKFIT.to_string());
            scf_cfg.df_k_aux = Some(JKFIT.to_string());
            let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &scf_cfg).unwrap();
            let g = want_grad.then(|| {
                ks_gradient_closed(&mol, &prep, &bs, op, &bounds, xc, &r, ext.as_ref()).unwrap()
            });
            (r, g)
        }
        QmmmMethod::Uks(xc) => {
            scf_cfg.xc = Some(xc.clone());
            scf_cfg.df_j_aux = Some(JKFIT.to_string());
            scf_cfg.df_k_aux = Some(JKFIT.to_string());
            let r = solve_uhf(&ctx, &mol, &prep, &bounds, &scf_cfg).unwrap();
            let g = want_grad.then(|| {
                ks_gradient_uks(&mol, &prep, &bs, op, &bounds, xc, &r, ext.as_ref()).unwrap()
            });
            (r, g)
        }
    };
    assert!(r.converged, "{}: SCF did not converge", case.name);

    let s2 = r.mos_beta.as_ref().map(|_| {
        let nelec = mol.nelec() as usize;
        let two_s = mol.multiplicity - 1;
        let (na, nb) = ((nelec + two_s) / 2, (nelec - two_s) / 2);
        let s = ferric_integrals::oneelectron::overlap(&prep);
        s_squared(&r, &s, na, nb)
    });
    let stability = r.stability.as_ref().map(|s| s.verdict());

    let coords = coords_of(atoms);
    let (mm_e, mm_g) = qmmm_mm_terms(&sys, &case.top, &coords).unwrap();
    let (grad, forces) = match g {
        Some(qm_grad) => {
            let forces = mm_forces(&sys, &mol, &prep, r.density_total()).unwrap();
            let full = full_gradient_with_mm(&sys, &qm_grad, &forces, &mm_g).unwrap();
            (Some(full), forces)
        }
        None => (None, vec![]),
    };
    Eval {
        energy: r.energy + mm_e.total,
        grad,
        forces,
        sys,
        s2,
        stability,
    }
}

/// The full gradient with the MM FIELD switched off (no embedding in the SCF
/// or its gradient, zero MM forces) but the same partition and MM force-field
/// terms: the vacuum negative control. If the embedding did not reach the
/// result, this would agree with the FD of the embedded energy.
fn vacuum_gradient(case: &Case, atoms: &[QmmmAtom]) -> Array2<f64> {
    let ctx = ParallelContext::default();
    let sys = build_system(case, atoms);
    let mol = sys.to_qm_molecule();
    let bs = ferric_core::basis::bundled(case.basis).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut cfg = base_scf();
    let g = match &case.method {
        QmmmMethod::Rhf => {
            let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
            rhf_gradient(&mol, &prep, op, &bounds, &r, None).unwrap()
        }
        QmmmMethod::Uhf => {
            let r = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
            uhf_gradient(&mol, &prep, op, &bounds, &r, None).unwrap()
        }
        QmmmMethod::Rks(xc) => {
            cfg.xc = Some(xc.clone());
            cfg.df_j_aux = Some(JKFIT.to_string());
            cfg.df_k_aux = Some(JKFIT.to_string());
            let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
            ks_gradient_closed(&mol, &prep, &bs, op, &bounds, xc, &r, None).unwrap()
        }
        QmmmMethod::Uks(xc) => {
            cfg.xc = Some(xc.clone());
            cfg.df_j_aux = Some(JKFIT.to_string());
            cfg.df_k_aux = Some(JKFIT.to_string());
            let r = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
            ks_gradient_uks(&mol, &prep, &bs, op, &bounds, xc, &r, None).unwrap()
        }
    };
    let zero = vec![[0.0; 3]; sys.mm_charge_positions().len()];
    let (_, mm_g) = qmmm_mm_terms(&sys, &case.top, &coords_of(atoms)).unwrap();
    full_gradient_with_mm(&sys, &g, &zero, &mm_g).unwrap()
}

fn evaluate(case: &Case, atoms: &[QmmmAtom], want_grad: bool) -> Eval {
    evaluate_with(case, atoms, case.scheme, want_grad, false)
}

// ---------------------------------------------------------------------------
// Roles: which full-structure row plays which part in the chain rule.
// ---------------------------------------------------------------------------

struct Roles {
    frontier: usize,
    host: usize,
    /// MM neighbours of the host (the M2 shell).
    m2: Vec<usize>,
    /// QM atoms other than the frontier.
    qm_real: Vec<usize>,
    /// MM atoms that are neither host nor M2 (waters).
    mm_other: Vec<usize>,
}

fn roles(case: &Case, sys: &QmmmSystem) -> Roles {
    assert_eq!(
        sys.link_atoms.len(),
        1,
        "{}: expected one link atom",
        case.name
    );
    let link = &sys.link_atoms[0];
    let frontier = sys.qm_indices[link.qm_atom];
    let host = link.mm_atom_full_index;
    let mut m2: Vec<usize> = case
        .bonds
        .iter()
        .filter_map(|&(a, b)| {
            if a == host && b != frontier {
                Some(b)
            } else if b == host && a != frontier {
                Some(a)
            } else {
                None
            }
        })
        .collect();
    m2.sort_unstable();
    let qm_real = sys
        .qm_indices
        .iter()
        .copied()
        .filter(|&i| i != frontier)
        .collect();
    let mm_other = sys
        .mm_indices
        .iter()
        .copied()
        .filter(|&i| i != host && !m2.contains(&i))
        .collect();
    Roles {
        frontier,
        host,
        m2,
        qm_real,
        mm_other,
    }
}

/// The free full-structure rows `optimize_qmmm` should have moved, recomputed
/// here from the START geometry exactly as `MoveMm` documents it.
fn expected_free(case: &Case, sys: &QmmmSystem) -> Vec<usize> {
    let mut free = sys.qm_indices.clone();
    match &case.move_mm {
        MoveMm::None => {}
        MoveMm::All => free.extend(&sys.mm_indices),
        MoveMm::WithinRadius(r) => {
            for &m in &sys.mm_indices {
                if sys
                    .qm_indices
                    .iter()
                    .any(|&q| dist(&sys.atoms[m], &sys.atoms[q]) <= *r)
                {
                    free.push(m);
                }
            }
        }
        MoveMm::Residues(ids) => {
            let rid = sys.residue_ids.as_ref().unwrap();
            for &m in &sys.mm_indices {
                if ids.contains(&rid[m]) {
                    free.push(m);
                }
            }
        }
    }
    free.sort_unstable();
    free
}

/// Structural checks FD cannot see: the scheme was actually applied to the
/// partition. The embedding's total charge is what each scheme says it is,
/// and the host charge is or is not present.
fn assert_partition(case: &Case, sys: &QmmmSystem) {
    let r = roles(case, sys);
    let raw_mm: f64 = sys.mm_indices.iter().map(|&i| sys.atoms[i].charge).sum();
    let q_host = sys.atoms[r.host].charge;
    let embedded: f64 = sys
        .to_external_potential()
        .map(|e| e.point_charges.iter().map(|p| p.q).sum::<f64>())
        .unwrap_or(0.0);
    let (want_total, host_zero, n_mid) = match case.scheme {
        BoundaryChargeScheme::Keep => (raw_mm, false, 0),
        BoundaryChargeScheme::DeleteHost => (raw_mm - q_host, true, 0),
        BoundaryChargeScheme::RedistributedCharge
        | BoundaryChargeScheme::RedistributedChargeDipole => (raw_mm, true, r.m2.len()),
    };
    assert!(
        (embedded - want_total).abs() < 1e-12,
        "{}: embedded MM charge {embedded:+.6} != {want_total:+.6} expected for {:?} \
         (raw MM {raw_mm:+.6}, host {q_host:+.6})",
        case.name,
        case.scheme
    );
    assert_eq!(
        sys.effective_charges[r.host] == 0.0,
        host_zero,
        "{}: host charge {} under {:?}",
        case.name,
        sys.effective_charges[r.host],
        case.scheme
    );
    assert_eq!(
        sys.boundary_charges.len(),
        n_mid,
        "{}: midpoints",
        case.name
    );
    if case.scheme == BoundaryChargeScheme::RedistributedCharge {
        for b in &sys.boundary_charges {
            let want = q_host / r.m2.len() as f64;
            assert!(
                (b.q - want).abs() < 1e-15,
                "{}: RC midpoint charge {} != q/n = {want}",
                case.name,
                b.q
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Relaxation and checks
// ---------------------------------------------------------------------------

fn opt_config() -> OptimizeConfig {
    OptimizeConfig {
        trust_radius: 0.1,
        max_steps: 200,
        g_max_thresh: 1.0e-4,
        g_rms_thresh: 5.0e-5,
        e_conv: 1.0e-9,
        ..Default::default()
    }
}

fn relax(case: &Case) -> ferric_scf::qmmm::QmmmOptimizeResult {
    let ctx = ParallelContext::default();
    let system = build_system(case, &case.atoms);
    let cfg = QmmmOptimizeConfig {
        method: case.method.clone(),
        move_mm: case.move_mm.clone(),
        opt: opt_config(),
        mm_topology: Some(case.top.clone()),
        scf: base_scf(),
    };
    let t = std::time::Instant::now();
    let res = optimize_qmmm(&ctx, &system, case.basis, &cfg).unwrap();
    eprintln!(
        "[{}] optimize_qmmm: converged={} steps={} E={:.10} ({:.1}s); \
         min link-to-charge distance start {:.4} Bohr, end {:.4} Bohr",
        case.name,
        res.converged,
        res.steps,
        res.energy,
        t.elapsed().as_secs_f64(),
        system.min_link_to_charge_distance().unwrap_or(f64::NAN),
        res.system.min_link_to_charge_distance().unwrap_or(f64::NAN),
    );
    if !res.converged {
        let n = res.energies.len();
        eprintln!(
            "[{}] NOT converged; last energies {:?}",
            case.name,
            &res.energies[n.saturating_sub(8)..]
        );
        let g = evaluate(case, &res.system.atoms, true).grad.unwrap();
        let free = expected_free(case, &system);
        eprintln!(
            "[{}]   final max |dE/dR| over free rows {:.3e}",
            case.name,
            max_abs_rows(&g, &free)
        );
        for (i, (a0, a1)) in system.atoms.iter().zip(&res.system.atoms).enumerate() {
            eprintln!(
                "[{}]   atom {i} {} moved {:.4} Bohr",
                case.name,
                a0.symbol,
                dist(a0, a1)
            );
        }
    }
    res
}

fn max_abs_rows(g: &Array2<f64>, rows: &[usize]) -> f64 {
    rows.iter()
        .flat_map(|&i| (0..3).map(move |k| (i, k)))
        .fold(0.0_f64, |m, (i, k)| m.max(g[(i, k)].abs()))
}

#[derive(Clone, Copy, PartialEq)]
enum Class {
    QmReal,
    Frontier,
    Host,
    M2,
    MmOther,
}

fn probes(r: &Roles) -> Vec<(usize, usize, Class)> {
    let mut p = vec![
        (r.frontier, 2, Class::Frontier),
        (r.frontier, 0, Class::Frontier),
        (r.host, 2, Class::Host),
        (r.host, 0, Class::Host),
        (r.m2[0], 1, Class::M2),
        (r.m2[1], 2, Class::M2),
        (r.qm_real[0], 2, Class::QmReal),
    ];
    for &i in r.mm_other.iter() {
        p.push((i, 0, Class::MmOther));
    }
    p
}

fn label(case: &Case, idx: usize, axis: usize, class: Class, free: &[usize]) -> String {
    let c = match class {
        Class::QmReal => "QM real",
        Class::Frontier => "QM frontier",
        Class::Host => "MM host",
        Class::M2 => "MM M2",
        Class::MmOther => "MM other",
    };
    let f = if free.contains(&idx) {
        "free"
    } else {
        "FROZEN"
    };
    format!(
        "{} atom {idx} {} {c} ({f})",
        case.name,
        ["x", "y", "z"][axis]
    )
}

/// Central FD of the total energy along one coordinate. For open-shell
/// references also returns the max ⟨S²⟩ deviation from `s2_ref` over the two
/// displaced points (a state flip mid-FD shows up here).
fn fd(
    case: &Case,
    atoms: &[QmmmAtom],
    idx: usize,
    axis: usize,
    h: f64,
    s2_ref: Option<f64>,
) -> (f64, f64) {
    let ep = evaluate(case, &displaced(atoms, idx, axis, h), false);
    let em = evaluate(case, &displaced(atoms, idx, axis, -h), false);
    let ds2 = match s2_ref {
        Some(s) => (ep.s2.unwrap() - s).abs().max((em.s2.unwrap() - s).abs()),
        None => 0.0,
    };
    ((ep.energy - em.energy) / (2.0 * h), ds2)
}

/// ⟨S²⟩ drift bar over FD displacements: a state flip is O(1e-1).
const S2_DRIFT_BAR: f64 = 1.0e-4;

/// Everything for one case that converges: stationarity per row class over
/// the free rows, FD at the minimum, FD off the minimum with negative
/// controls. Returns the relaxed atoms.
fn run_case(case: &Case) -> Vec<QmmmAtom> {
    let start_sys = build_system(case, &case.atoms);
    assert_partition(case, &start_sys);
    let free = expected_free(case, &start_sys);
    let r = roles(case, &start_sys);

    let res = relax(case);
    assert!(
        res.converged,
        "{}: optimize_qmmm did not converge in {} steps",
        case.name, res.steps
    );
    let atoms = res.system.atoms.clone();

    // Frozen atoms did not move, bit for bit; free MM atoms did move.
    for i in 0..atoms.len() {
        let (a0, a1) = (&case.atoms[i], &atoms[i]);
        if free.contains(&i) {
            continue;
        }
        assert!(
            a0.x == a1.x && a0.y == a1.y && a0.z_pos == a1.z_pos,
            "{}: atom {i} is outside the free set {free:?} but moved",
            case.name
        );
    }
    let free_mm: Vec<usize> = free
        .iter()
        .copied()
        .filter(|i| start_sys.mm_indices.contains(i))
        .collect();
    let moved = free_mm
        .iter()
        .any(|&i| dist(&case.atoms[i], &atoms[i]) > 1e-4);
    assert!(
        free_mm.is_empty() || moved,
        "{}: no free MM atom moved (free MM {free_mm:?})",
        case.name
    );

    let ev = evaluate_with(case, &atoms, case.scheme, true, false);
    let grad = ev.grad.as_ref().unwrap();
    assert_partition(case, &ev.sys);

    // --- Stationarity per row class, FREE rows only.
    let bar = stationarity_bar(case);
    let classes: Vec<(&str, Vec<usize>)> = vec![
        ("QM real", r.qm_real.clone()),
        ("QM frontier", vec![r.frontier]),
        ("MM host", vec![r.host]),
        ("MM M2", r.m2.clone()),
        ("MM other", r.mm_other.clone()),
    ];
    for (name, rows) in &classes {
        let (fr, fz): (Vec<usize>, Vec<usize>) = rows.iter().partition(|i| free.contains(i));
        if !fr.is_empty() {
            let m = max_abs_rows(grad, &fr);
            eprintln!(
                "[{}] stationary {name} (free {fr:?}): max |dE/dR| = {m:.3e}",
                case.name
            );
            assert!(
                m < bar,
                "{}: {name} free rows {m:.3e} >= {bar:.1e}",
                case.name
            );
        }
        if !fz.is_empty() {
            // Frozen rows are NOT stationary by design; that they are not
            // shows the optimizer held them rather than relaxed them.
            let m = max_abs_rows(grad, &fz);
            eprintln!(
                "[{}] FROZEN {name} {fz:?}: max |dE/dR| = {m:.3e} (nonzero by design)",
                case.name
            );
            assert!(
                m > 1.0e-3,
                "{}: frozen {name} rows have |dE/dR| {m:.3e}; a frozen row should carry \
                 the force the optimizer was not allowed to relax",
                case.name
            );
        }
    }

    // --- Open-shell: ⟨S²⟩ and stability at the start and the end.
    let s2_ref = ev.s2;
    if matches!(case.method, QmmmMethod::Uhf | QmmmMethod::Uks(_)) {
        for (tag, at) in [("start", &case.atoms), ("end", &atoms)] {
            let e = evaluate_with(case, at, case.scheme, false, true);
            eprintln!(
                "[{}] {tag}: <S^2> = {:.6}, stability = {:?}",
                case.name,
                e.s2.unwrap(),
                e.stability.map(|v| v.label())
            );
            if matches!(case.method, QmmmMethod::Uhf) {
                assert_eq!(
                    e.stability,
                    Some(StabilityVerdict::Stable),
                    "{}: {tag} UHF state not stable",
                    case.name
                );
            }
        }
    }

    // --- FD at the minimum: small together.
    let fbar = fd_min_bar(case);
    for &h in &HS {
        for &(idx, axis, class) in &probes(&r) {
            let (d, ds2) = fd(case, &atoms, idx, axis, h, s2_ref);
            let an = grad[(idx, axis)];
            let err = (an - d).abs();
            eprintln!(
                "[{} min h={h:.0e}] {}: analytic {an:+.6e} FD {d:+.6e} |D| {err:.2e} dS2 {ds2:.1e}",
                case.name,
                label(case, idx, axis, class, &free)
            );
            assert!(ds2 < S2_DRIFT_BAR, "{}: <S^2> drifted {ds2:.2e}", case.name);
            assert!(
                err < fbar,
                "{}: at-min |D| {err:.2e} >= {fbar:.1e}",
                case.name
            );
            if free.contains(&idx) {
                assert!(
                    d.abs() < stationarity_bar(case),
                    "{}: FD derivative {d:+.3e} on free atom {idx} is not small: the \
                     returned geometry is not a stationary point of the ENERGY",
                    case.name
                );
            }
        }
    }

    off_minimum(case, &atoms);
    atoms
}

/// The amplitude-carrying check, plus negative controls. `base` is pushed off
/// the minimum along the cut bond, a frontier x, an M2 y and a QM-H z.
fn off_minimum(case: &Case, base: &[QmmmAtom]) {
    let sys0 = build_system(case, base);
    let r = roles(case, &sys0);
    let free = expected_free(case, &build_system(case, &case.atoms));
    let mut atoms = base.to_vec();
    atoms[r.host].z_pos += 0.35;
    atoms[r.frontier].x += 0.12;
    atoms[r.m2[0]].y += 0.20;
    atoms[r.qm_real[0]].z_pos -= 0.15;

    let ev = evaluate(case, &atoms, true);
    let grad = ev.grad.clone().unwrap();
    let probes = probes(&r);
    let s2_ref = ev.s2;

    // FD for every probe and h; kept for the negative controls.
    let mut fds: Vec<(usize, usize, Class, f64, f64)> = vec![];
    let rel_bar = fd_off_rel_bar(case);
    let mut min_amp = f64::INFINITY;
    for &h in &HS {
        for &(idx, axis, class) in &probes {
            let (d, ds2) = fd(case, &atoms, idx, axis, h, s2_ref);
            let an = grad[(idx, axis)];
            // Relative to the row, floored at 1e-2 Ha/Bohr: frozen far-away
            // water rows carry only ~1e-5 of gradient, where a pure relative
            // measure would read SCF noise (|D| ~ 1e-10) as a 1e-4 miss.
            let rel = (an - d).abs() / an.abs().max(1.0e-2);
            if class != Class::MmOther {
                min_amp = min_amp.min(an.abs());
            }
            eprintln!(
                "[{} off h={h:.0e}] {}: analytic {an:+.6e} FD {d:+.6e} |D| {:.2e} rel {rel:.2e}",
                case.name,
                label(case, idx, axis, class, &free),
                (an - d).abs()
            );
            assert!(ds2 < S2_DRIFT_BAR, "{}: <S^2> drifted {ds2:.2e}", case.name);
            assert!(
                rel < rel_bar,
                "{}: off-min rel {rel:.2e} >= {rel_bar:.1e}",
                case.name
            );
            fds.push((idx, axis, class, h, d));
        }
    }
    eprintln!(
        "[{}] off: smallest probed |analytic| = {min_amp:.3e}",
        case.name
    );
    assert!(
        min_amp > 1.0e-3,
        "{}: off-minimum probes carry too little amplitude ({min_amp:.3e})",
        case.name
    );
    let fd_at = |idx: usize, axis: usize| -> f64 {
        fds.iter()
            .find(|f| f.0 == idx && f.1 == axis && f.3 == 1.0e-4)
            .unwrap()
            .4
    };
    // Worst relative miss of a candidate gradient over a probe subset.
    let worst = |g: &Array2<f64>, subset: &dyn Fn(Class) -> bool| -> f64 {
        probes
            .iter()
            .filter(|p| subset(p.2))
            .map(|&(i, k, _)| {
                let d = fd_at(i, k);
                (g[(i, k)] - d).abs() / d.abs().max(1e-12)
            })
            .fold(0.0, f64::max)
    };

    // --- Negative control 1: the gradient of a DIFFERENT scheme against this
    // scheme's FD must miss.
    use BoundaryChargeScheme::*;
    for wrong in [
        Keep,
        DeleteHost,
        RedistributedCharge,
        RedistributedChargeDipole,
    ] {
        if wrong == case.scheme {
            continue;
        }
        let gw = evaluate_with(case, &atoms, wrong, true, false)
            .grad
            .unwrap();
        let w = worst(&gw, &|_| true);
        eprintln!(
            "[{} negctl] {:?} gradient vs {:?} FD: worst rel {w:.2e} ({:.0}x bar)",
            case.name,
            wrong,
            case.scheme,
            w / rel_bar
        );
        assert!(
            w > 10.0 * rel_bar,
            "{}: the {wrong:?} gradient agrees with the {:?} energy to {w:.2e} — the FD \
             check cannot tell the two schemes apart",
            case.name,
            case.scheme
        );
    }

    // --- Negative control (vacuum): the same structure with the MM field
    // switched off must MISS the FD of the embedded energy, on QM rows and MM
    // rows alike. A pass here would mean the embedding never reached the
    // energy this file differentiates (a vacuum test).
    {
        let gv = vacuum_gradient(case, &atoms);
        let qm = worst(&gv, &|c| matches!(c, Class::QmReal | Class::Frontier));
        let all = worst(&gv, &|_| true);
        eprintln!(
            "[{} negctl] MM field OFF vs embedded FD: QM rows worst rel {qm:.2e}, all rows \
             {all:.2e} ({:.0}x bar)",
            case.name,
            all / rel_bar
        );
        assert!(
            qm > 10.0 * rel_bar && all > 10.0 * rel_bar,
            "{}: the field-off gradient agrees with the embedded FD (QM {qm:.2e}, all \
             {all:.2e}) — the embedding is not reaching the energy",
            case.name
        );
    }

    // --- Negative control 2: a gradient broken ONLY in MM rows must PASS a
    // QM-only probe set and FAIL the MM probes. This is what makes the MM-row
    // probes load-bearing.
    let n_atom_centred = ev
        .sys
        .mm_indices
        .iter()
        .filter(|&&i| ev.sys.effective_charges[i] != 0.0)
        .count();
    let atom_centred: Vec<usize> = ev
        .sys
        .mm_indices
        .iter()
        .copied()
        .filter(|&i| ev.sys.effective_charges[i] != 0.0)
        .collect();
    let mut broken: Vec<(&str, Array2<f64>)> = vec![];
    // (a) M2 own-charge MM force dropped.
    {
        let mut g = grad.clone();
        for &m2 in &r.m2 {
            let k = atom_centred.iter().position(|&i| i == m2).unwrap();
            for c in 0..3 {
                g[(m2, c)] += ev.forces[k][c];
            }
        }
        broken.push(("M2 own-charge force dropped", g));
    }
    // (b) RC/RCD: midpoint split 0.5/0.5 -> 1.0/0.0 (column sum preserved).
    if !ev.sys.boundary_charges.is_empty() {
        let mut g = grad.clone();
        for (k, b) in ev.sys.boundary_charges.iter().enumerate() {
            let f = ev.forces[n_atom_centred + k];
            for c in 0..3 {
                g[(b.hosts.0, c)] -= 0.5 * f[c];
                g[(b.hosts.1, c)] += 0.5 * f[c];
            }
        }
        broken.push(("midpoint split 1.0/0.0", g));
    }
    for (name, g) in &broken {
        let qm_only = worst(g, &|c| matches!(c, Class::QmReal | Class::Frontier));
        let mm = worst(g, &|c| matches!(c, Class::Host | Class::M2));
        eprintln!(
            "[{} negctl] {name}: QM-only probes worst rel {qm_only:.2e} (PASS expected), \
             MM probes worst rel {mm:.2e} (FAIL expected)",
            case.name
        );
        assert!(
            qm_only < rel_bar,
            "{}: {name} changed a QM row ({qm_only:.2e})",
            case.name
        );
        assert!(
            mm > 10.0 * rel_bar,
            "{}: {name} was not caught by the MM probes ({mm:.2e})",
            case.name
        );
    }
}

// ---------------------------------------------------------------------------
// 0. Exactness anchors: no MM atoms, MoveMm::None == the plain driver, bit for bit.
// ---------------------------------------------------------------------------

fn anchor(method: QmmmMethod) {
    let ctx = ParallelContext::default();
    let (atoms, mult) = match method {
        QmmmMethod::Rhf | QmmmMethod::Rks(_) => (
            vec![
                QmmmAtom::new("H", 1, 0.0, 0.0, 0.0, 0.0),
                QmmmAtom::new("H", 1, 0.0, 0.0, 1.0 * ANG2BOHR, 0.0),
            ],
            1,
        ),
        _ => (
            vec![
                QmmmAtom::new("O", 8, 0.0, 0.0, 0.0, 0.0),
                QmmmAtom::new("H", 1, 0.0, 0.0, 1.05 * ANG2BOHR, 0.0),
            ],
            2,
        ),
    };
    let system = QmmmSystem::new(&atoms, QmSelection::Indices(vec![0, 1]), 0, mult).unwrap();
    assert!(system.mm_indices.is_empty());
    assert!(system.to_external_potential().is_none());
    let mol = system.to_qm_molecule();
    let scf = base_scf();
    let opt = OptimizeConfig {
        trust_radius: 0.1,
        ..Default::default()
    };
    let mut plain_cfg = scf.clone();
    if let QmmmMethod::Rks(xc) | QmmmMethod::Uks(xc) = &method {
        plain_cfg.xc = Some(xc.clone());
        plain_cfg.df_j_aux = Some(JKFIT.to_string());
        plain_cfg.df_k_aux = Some(JKFIT.to_string());
    }
    let op = Operator::coulomb();
    let plain = match method {
        QmmmMethod::Rhf | QmmmMethod::Rks(_) => {
            optimize_geometry(&ctx, &mol, "sto-3g", op, &plain_cfg, &opt).unwrap()
        }
        _ => optimize_geometry_uhf(&ctx, &mol, "sto-3g", op, &plain_cfg, &opt).unwrap(),
    };
    assert!(plain.converged);
    let cfg = QmmmOptimizeConfig {
        method: method.clone(),
        move_mm: MoveMm::None,
        opt,
        mm_topology: None,
        scf,
    };
    let res = optimize_qmmm(&ctx, &system, "sto-3g", &cfg).unwrap();
    eprintln!(
        "[anchor {method:?}] optimize_qmmm {} steps E={:.15}; plain driver {} steps E={:.15}",
        res.steps, res.energy, plain.steps, plain.energy
    );
    assert!(res.converged);
    assert_eq!(res.steps, plain.steps, "{method:?}: step count");
    assert_eq!(
        res.energy.to_bits(),
        plain.energy.to_bits(),
        "{method:?}: optimize_qmmm {:.15} != plain driver {:.15}",
        res.energy,
        plain.energy
    );
    assert_eq!(
        res.energies.len(),
        plain.energy_trace.len(),
        "{method:?}: trace length"
    );
    for (a, b) in res.energies.iter().zip(&plain.energy_trace) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{method:?}: per-step energy {a} vs {b}"
        );
    }
}

#[test]
#[ignore = "validation: optimize_qmmm"]
fn anchor_rhf_no_mm_matches_optimize_geometry() {
    anchor(QmmmMethod::Rhf);
}

#[test]
#[ignore = "validation: optimize_qmmm"]
fn anchor_uhf_no_mm_matches_optimize_geometry_uhf() {
    anchor(QmmmMethod::Uhf);
}

#[test]
#[ignore = "validation: optimize_qmmm"]
fn anchor_rks_pbe_no_mm_matches_optimize_geometry() {
    anchor(QmmmMethod::Rks("PBE".into()));
}

#[test]
#[ignore = "validation: optimize_qmmm"]
fn anchor_uks_pbe_no_mm_matches_optimize_geometry_uhf() {
    anchor(QmmmMethod::Uks("PBE".into()));
}

// ---------------------------------------------------------------------------
// 1. Cases
// ---------------------------------------------------------------------------

#[test]
#[ignore = "validation: optimize_qmmm"]
fn rhf_z1_ethane_move_all() {
    run_case(&case("rhf_z1"));
}

#[test]
#[ignore = "validation: optimize_qmmm"]
fn rhf_rc_ethane_move_all() {
    run_case(&case("rhf_rc"));
}

#[test]
#[ignore = "validation: optimize_qmmm"]
fn rhf_z1_ethane_move_all_631g() {
    run_case(&case("rhf_z1_631g"));
}

/// Keep: the full host charge stays 0.44 Å from the link H.
#[test]
#[ignore = "validation: optimize_qmmm"]
fn rhf_keep_ethane_move_all() {
    let c = case("rhf_keep");
    let sys = build_system(&c, &c.atoms);
    let d = sys.min_link_to_charge_distance().unwrap();
    eprintln!(
        "[rhf_keep] min link-to-charge distance at start: {d:.4} Bohr = {:.4} A",
        d / ANG2BOHR
    );
    run_case(&c);
}

#[test]
#[ignore = "validation: optimize_qmmm"]
fn uhf_z1_ethyl_radical_move_all() {
    run_case(&case("uhf_z1"));
}

#[test]
#[ignore = "validation: optimize_qmmm"]
fn uhf_rc_ethyl_radical_move_all() {
    run_case(&case("uhf_rc"));
}

fn run_within_radius(name: &str) {
    let c = case(name);
    let sys = build_system(&c, &c.atoms);
    let r = roles(&c, &sys);
    let free = expected_free(&c, &sys);
    let free_mm: Vec<usize> = free
        .iter()
        .copied()
        .filter(|i| sys.mm_indices.contains(i))
        .collect();
    eprintln!(
        "[{name}] free set at start {free:?} (free MM {free_mm:?}, {:?})",
        c.move_mm
    );
    assert_eq!(
        free_mm,
        vec![r.host],
        "{name}: free MM set should be the host only"
    );
    let mind = |atoms: &[QmmmAtom], m: usize| {
        sys.qm_indices
            .iter()
            .map(|&q| dist(&atoms[m], &atoms[q]))
            .fold(f64::INFINITY, f64::min)
    };
    let relaxed = run_case(&c);
    let MoveMm::WithinRadius(radius) = c.move_mm else {
        unreachable!()
    };
    for &m in &r.m2 {
        let (d0, d1) = (mind(&c.atoms, m), mind(&relaxed, m));
        eprintln!("[{name}] M2 atom {m}: min distance to QM start {d0:.4} end {d1:.4} Bohr");
        // REACHABILITY of the "free set is measured once" claim: re-measured
        // at the end, this atom WOULD be free. A per-step re-evaluation
        // would therefore have changed the free set mid-run.
        assert!(
            d0 > radius && d1 < radius,
            "{name}: M2 atom {m} does not cross r = {radius} ({d0:.4} -> {d1:.4}); a \
             per-step re-evaluation of the free set would be an identity here"
        );
    }
}

#[test]
#[ignore = "validation: optimize_qmmm"]
fn rks_pbe_rcd_ethane_within_radius() {
    run_within_radius("rks_rcd_within");
}

#[test]
#[ignore = "validation: optimize_qmmm"]
fn uks_pbe_rcd_ethyl_within_radius() {
    run_within_radius("uks_rcd_within");
}

/// `MoveMm::Residues` on a residue-tagged system built with
/// `QmSelection::WithinRadiusWholeResidues`. MEASURED: this system has no
/// bound minimum with the generic force field — with the MM CH3 residue free
/// and both waters frozen, the whole ethane translates under a constant
/// ~1e-2 Ha/Bohr force (energy falling linearly, 0.76 Bohr in 200 steps), and
/// with water A free as well it drifts 8.5 Bohr. So convergence is NOT
/// asserted here. What is asserted: the free set is exactly residues 0 and 1
/// (QM and the MM CH3); over a short run the frozen waters keep their
/// coordinates bit for bit while the free residue moves; and the full
/// gradient — including the frozen water rows — matches FD off the minimum,
/// with every negative control.
#[test]
#[ignore = "validation: optimize_qmmm"]
fn rhf_rcd_ethane_waters_residues() {
    let mut c = case("rhf_rcd_residues");
    let sys = build_system(&c, &c.atoms);
    assert_eq!(
        sys.qm_indices,
        vec![0, 2, 4, 6],
        "whole-residue QM selection"
    );
    assert_partition(&c, &sys);
    let free = expected_free(&c, &sys);
    eprintln!("[rhf_rcd_residues] free set {free:?}");
    assert_eq!(
        free,
        (0..8).collect::<Vec<_>>(),
        "residues 0, 1 free; both waters frozen"
    );

    let ctx = ParallelContext::default();
    let cfg = QmmmOptimizeConfig {
        method: c.method.clone(),
        move_mm: c.move_mm.clone(),
        opt: OptimizeConfig {
            max_steps: 10,
            ..opt_config()
        },
        mm_topology: Some(c.top.clone()),
        scf: base_scf(),
    };
    let res = optimize_qmmm(&ctx, &sys, c.basis, &cfg).unwrap();
    for i in 0..c.atoms.len() {
        let d = dist(&c.atoms[i], &res.system.atoms[i]);
        if free.contains(&i) {
            assert!(d > 1e-4, "free atom {i} did not move ({d:.2e})");
        } else {
            let (a0, a1) = (&c.atoms[i], &res.system.atoms[i]);
            assert!(
                a0.x == a1.x && a0.y == a1.y && a0.z_pos == a1.z_pos,
                "frozen water atom {i} moved {d:.3e} Bohr"
            );
        }
    }
    // Off-minimum FD from the start geometry (there is no minimum).
    c.atoms = res.system.atoms.clone();
    off_minimum(&c, &c.atoms.clone());
}

// ---------------------------------------------------------------------------
// MEASURED (2026-10-05, release build, OPENBLAS_NUM_THREADS=1).
//
// - Anchors: with no MM atoms, optimize_qmmm reproduces optimize_geometry
//   (RHF, RKS/PBE) and optimize_geometry_uhf (UHF, UKS/PBE) bit for bit: same
//   step count, same per-evaluation energy trace.
// - KS floor: FD vs analytic at the minimum is 1.0e-8 (RKS) / 1.1e-8 (UKS)
//   Ha/Bohr, the same as HF. The DF-J/K gradient and the grid response
//   differentiate the energy the SCF computed, so the pre-registered looser
//   KS floor (1e-7..1e-6) did not materialize.
// - Keep: the host charge (q = -0.1) sits 0.8315 Bohr = 0.44 A from the link H,
//   and the optimization CONVERGES (13 steps, gradient FD-correct). The
//   divergence recorded elsewhere for Keep does not occur at this host charge.
// - UHF/UKS: <S^2> 0.7525..0.7649, stability STABLE at start and end; <S^2>
//   drift over every FD displacement < 1e-4.
// - WithinRadius: the M2 hydrogens start 4.338 (RKS) / 4.270 (UKS) Bohr from
//   the QM region and end at 4.051 / 4.003, so r = 4.2 / 4.15 is crossed and a
//   per-step re-evaluation of the free set is reachable.
// - Residues: no bound minimum with the generic force field (see the test).
//
// MUTATION LEDGER (2026-10-05). Each mutant was applied to src/qmmm.rs by a
// kill-safe harness (source restored from HEAD at startup and after every
// mutant), COMPILED, and the named tests RAN (counts read from the
// `test result:` line; none ignored).
//
// | # | mutation (src/qmmm.rs) | tests run | observed |
// |---|---|---|---|
// | 1 | Z1 host-charge deletion skipped (`DeleteHost => {}`) | rhf_z1, rhf_z1_631g | 0 passed / 2 failed: "embedded MM charge -0.001000 != +0.099000 expected for DeleteHost" (assert_partition). FD alone cannot see this: both paths see the same undeleted charge. |
// | 2 | RC midpoint charge q/n -> 2q/n | rhf_rc | 0/1: "embedded MM charge -0.101000 != -0.001000 expected for RedistributedCharge" |
// | 3 | `optimize_qmmm` Uks branch passes `None` instead of `ext` to ks_gradient_uks | uks_rcd_within | 0/1: "QM real free rows 7.735e-4 >= 5.3e-5" — the optimizer stopped where the embedded energy is not stationary |
// | 4 | WithinRadius free set re-evaluated every step from the current geometry | rks_rcd_within, uks_rcd_within | 0/2: panic in optimize.rs (gradient length changes when the M2 atoms cross r) |
// | 5 | full_gradient midpoint split 0.5/0.5 -> 1.0/0.0 | rhf_rc | 0/1: "at-min \|D\| 4.89e-3 >= 2.4e-7" |
//
// The in-test negative controls (wrong-scheme gradient, MM field off, M2
// own-charge force dropped, midpoint split broken) run in every case and
// miss the FD by >= 3.5e-3 relative, >= 1e4 x the bars, while the broken
// MM-row folds PASS a QM-only probe set — the MM probes are load-bearing.
