// LANL2DZ exponents such as 0.318 are basis data, not approximations of 1/pi.
#![allow(clippy::approx_constant)]
//! k-point force and stress anchors for the coverage gaps of issue #421 left
//! after `pbc_kstress.rs`, `pbc_kpoint_ecp_rotation.rs`,
//! `pbc_sr_rotation_kderiv.rs` and `validation_kstress_followups.rs`, plus
//! the external-code (PySCF) force pins. Every test here is heavy and
//! `#[ignore = "validation: ..."]`d (weekly job). Sections D-F use the RS-GDF
//! J/K, a rotating orbital basis (`SrColumnRotation::on()`, rotated columns
//! on every ECP centre) and LANL2DZ ECPs; section H the dense AFT.
//!
//! (D) OPEN SHELL + TWO DISTINCT ECP ELEMENTS + A SEPTET: AsSb in
//!     diag(6, 6, 10), As (n_core 28) and Sb (n_core 46) each with its
//!     LANL2DZ ECP, two high-spin 4S atoms: 10 valence electrons as a SEPTET
//!     (8 alpha, 2 beta), k-UHF. A quartet or quintet of a halogen pair
//!     (IBr) was tried first: its high-spin states have partially filled
//!     degenerate p shells and the k-UHF did not converge (400 iterations);
//!     the septet has every spin shell closed. Anchors: 1x1x1 vs the Gamma
//!     UHF RS-GDF force and stress; Gamma-centred 1x1x3 (non-TRIM) vs the
//!     explicit 3-cell supercell (6 ECP atoms, multiplicity 19) / 3; central
//!     FD of the 1x1x3 k-UHF energy (all nine strains).
//! (E) 1x1x3 TRICLINIC CELL WITH AN ECP: the HI cell of
//!     `pbc_kpoint_ecp_rotation.rs` section 4 (rows (6,0,0), (1,6,0),
//!     (1.2,-0.8,7)), k-RHF on the Gamma-centred 1x1x3 mesh (non-TRIM: the
//!     residue lookup through the skewed reciprocal lattice with moduli 3,
//!     where `EcpWrongResidue` and `ConjD` act) vs the 3-cell supercell / 3,
//!     and vs central FD of its own energy.
//! (F) SHIFTED (MONKHORST-PACK) MESHES OTHER THAN 1x1x2 with the ECP and the
//!     rotation: MP 2x2x1 on a TRICLINIC HI cell (`HI_TRI8`, shifted along
//!     two skewed axes, residue moduli [4, 4, 1]) and MP 1x1x4 on the
//!     orthorhombic HI cell (complex non-TRIM shifted points, moduli
//!     [1, 1, 8]): central FD of force and stress, rotated vs unrotated
//!     analytic, the ECP residue mutants. MP 1x1x3: in ferric's (and PySCF's
//!     `with_gamma_point=False`) convention an ODD axis is NOT shifted
//!     (`f = (m + 1/2)/N - 1/2` is the Gamma set in another order), so MP
//!     1x1x3 is anchored as an exact identity with the Gamma-centred 1x1x3
//!     (a genuinely shifted odd mesh does not exist in this convention).
//! (H) EXTERNAL CODE: dense-AFT k-point RHF/UHF energies and FORCES vs PySCF
//!     2.13.1 `pbc.grad.krhf` / `pbc.grad.kuhf` (FFTDF, all-electron gate
//!     lifted, `scripts/gen_pyscf_kforce_pins.py`), and the ECP-cell
//!     ENERGIES vs PySCF AFTDF with PySCF's `lanl2dz`. No external STRESS
//!     exists: PySCF 2.13.1 has no k-point HF stress (`krks_stress` /
//!     `kuks_stress` are FFTDF pure-DFT only and refuse hybrids).
//!
//! ARTIFACT hypotheses (stated before measuring):
//! (D) a defect specific to SEVERAL ECP centres (the centre derivative
//!     folded onto the wrong ECP atom, image weights of one ECP group read
//!     with another group's atom index) is invisible on HI (one ECP atom)
//!     and an O(|F_ecp|) miss here; `EcpCentreWrongAtom` is that defect
//!     wired into production (on HI it folded onto H). An open-shell ECP
//!     slip (ECP weights from D_alpha only, or from the spin density) is an
//!     O(1) miss when D_alpha - D_beta carries 6 electrons. If correct:
//!     1x1x1 = Gamma at ~1e-12, supercell <= ~1e-10, FD at the h^2 floor.
//!     NOT visible to any internal anchor: an ENERGY that attaches the wrong
//!     element's ECP (the derivative of a wrong energy is still that
//!     energy's derivative); only the external energy pin (H) sees that.
//! (E) a residue lookup through a skewed reciprocal lattice that slips in
//!     the a3 coordinate is invisible on the TRIM 1x1x2 mesh of #476
//!     (`EcpWrongResidue` is an identity there) and acts at 1x1x3.
//! (F) an image weight read modulo the mesh count instead of the residue
//!     modulus on a shifted axis aliases images half a supercell apart:
//!     identity on Gamma-centred meshes, an O(|F_ecp|) miss on every shifted
//!     axis. On MP 1x1x3 (unshifted) it is an identity, asserted.
//! (H) a common-mode error of ferric's force AND energy (invisible to every
//!     FD anchor) shows as an energy miss vs PySCF; a force-only error as a
//!     force miss. If correct: both at the SCF / grid floor (~1e-11).
//!
//! SHARED CODE: as in `validation_kstress_followups.rs`, the k fitted
//! derivative imports the Gamma SR walks, and the k ECP strain/force walks
//! share the ECP image kernel with the Gamma side ("ECP 0.00e0" at 1x1x1 is
//! expected, not evidence), so the 1x1x1 and supercell anchors check the k
//! wrappers (phases, residue binning, LR part) independently; the FD
//! anchors are the independent check of the shared kernels.
//!
//! MEASURED ERRORS AND MUTANT LEDGER: per test doc (release, RAYON 3, shared
//! box, 2026-10-19). Bars >= 10x the measured value; MUTANT_BAR 1e-5.

mod common;

use common::prep_for;
use ferric_core::basis::{self, BasisSet, Shell};
use ferric_core::ecp::{EcpDef, EcpShell, EcpTerm};
use ferric_core::mol::{Atom, Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::grad::{
    gamma_rhf_gradient_rsgdf, gamma_uhf_gradient_rsgdf, GammaGradConfig, RsGdfGradSource,
};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::kgrad::{
    kpoint_rhf_gradient, kpoint_uhf_gradient, KGradConfig, KGradJk, KGradMutation, KGradient,
    KRsGdfGradSource,
};
use ferric_pbc::kpts::{KPointMesh, MeshCentring};
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KScfConfig, KScfResult};
use ferric_pbc::kstress::{
    kpoint_rhf_stress, kpoint_uhf_stress, KStress, KStressConfig, KStressJk, KStressMutation,
};
use ferric_pbc::kuscf::{solve_kuhf_injected, KUScfResult};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
use ferric_pbc::sr_rotation::SrColumnRotation;
use ferric_pbc::stress::{
    gamma_rhf_stress_rsgdf, gamma_uhf_stress_rsgdf, GammaStress, GammaStressConfig, Mat3,
};
use ferric_scf::result::{ScfExit, ScfResult, Spin};
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
const AMPLE: usize = 1 << 31;
const FD_H: f64 = 1e-4;
const ZERO: Mat3 = [[0.0; 3]; 3];
const ALL9: [(usize, usize); 9] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
];
/// Smallest movement a production-wired mutant must make (anchor miss or
/// movement of the analytic total over ALL components).
const MUTANT_BAR: f64 = 1e-5;

// ------------------------------------------------------------ shared helpers

fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = ZERO;
    e[i][j] = h;
    e
}

fn max_mat(a: &Mat3, b: &Mat3) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            m = m.max((a[i][j] - b[i][j]).abs());
        }
    }
    m
}

fn offdiag_max(m: &Mat3) -> f64 {
    let mut w = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                w = w.max(m[i][j].abs());
            }
        }
    }
    w
}

fn scale3(m: &Mat3, f: f64) -> Mat3 {
    let mut o = *m;
    for r in o.iter_mut() {
        for v in r.iter_mut() {
            *v *= f;
        }
    }
    o
}

fn grad_miss(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn amax(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// `max |F_k[A, x] - F_sc[c * natoms + A, x]|` over every copy `c`.
fn sc_miss(gk: &Array2<f64>, gsc: &Array2<f64>) -> f64 {
    let na = gk.nrows();
    let nc = gsc.nrows() / na;
    let mut w = 0.0_f64;
    for c in 0..nc {
        for a in 0..na {
            for x in 0..3 {
                w = w.max((gk[(a, x)] - gsc[(c * na + a, x)]).abs());
            }
        }
    }
    w
}

fn real_of(d: &Array2<Complex64>) -> Array2<f64> {
    let im = d.iter().fold(0.0_f64, |m, z| m.max(z.im.abs()));
    assert!(im < 1e-9, "Gamma density has an imaginary part {im:e}");
    d.mapv(|z| z.re)
}

/// The k density unfolded to the explicit diag(n) supercell (copy order m0
/// outer, as [`supercell`]).
fn unfold(mesh: &KPointMesh, dk: &[Array2<Complex64>]) -> Array2<f64> {
    let n = mesh.n();
    let nao = dk[0].nrows();
    let nk = mesh.nk();
    let copies: Vec<[i64; 3]> = (0..n[0])
        .flat_map(|m0| {
            (0..n[1]).flat_map(move |m1| (0..n[2]).map(move |m2| [m0 as i64, m1 as i64, m2 as i64]))
        })
        .collect();
    let nc = copies.len();
    let mut out = Array2::<f64>::zeros((nc * nao, nc * nao));
    let mut max_im = 0.0_f64;
    for (c, tc) in copies.iter().enumerate() {
        for (c2, tc2) in copies.iter().enumerate() {
            let nl = [tc[0] - tc2[0], tc[1] - tc2[1], tc[2] - tc2[2]];
            for m in 0..nao {
                for nn in 0..nao {
                    let mut z = Complex64::new(0.0, 0.0);
                    for (k, d) in dk.iter().enumerate() {
                        z += mesh.phase(k, nl) * d[(m, nn)];
                    }
                    z /= nk as f64;
                    max_im = max_im.max(z.im.abs());
                    out[(c * nao + m, c2 * nao + nn)] = z.re;
                }
            }
        }
    }
    assert!(max_im < 1e-10, "unfolded density not real: {max_im:e}");
    out
}

fn gamma_scf(da: Array2<f64>, db: Option<Array2<f64>>) -> ScfResult {
    let n = da.nrows();
    let (spin, total, alpha) = match &db {
        Some(b) => (Spin::Unrestricted, &da + b, da.clone()),
        None => (Spin::Restricted, da.clone(), &da * 0.5),
    };
    let un = db.is_some();
    ScfResult {
        spin,
        energy: 0.0,
        density_total: total,
        density_alpha: alpha,
        density_beta: db,
        mos_alpha: Array2::eye(n),
        mos_beta: un.then(|| Array2::eye(n)),
        eps_alpha: vec![0.0; n],
        eps_beta: un.then(|| vec![0.0; n]),
        fock_alpha: Array2::zeros((n, n)),
        fock_beta: un.then(|| Array2::zeros((n, n))),
        converged: true,
        exit: ScfExit::Converged,
        iterations: 0,
        computed_quartets: 0,
        induced_dipoles: None,
        stability: None,
        stability_external: None,
        df_jk: None,
        cosx_final: None,
        cosx_schedule: None,
        convergence: None,
        rohf_spin_focks: None,
    }
}

fn gcfg(m: Option<KGradMutation>) -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        mutation: m,
        nucleus_exponent: None,
    }
}

fn scfg(m: Option<KStressMutation>) -> KStressConfig {
    KStressConfig {
        budget_bytes: Some(AMPLE),
        mutation: m,
        nucleus_exponent: None,
    }
}

/// Collects (name, value, bar) checks so one heavy run reports every number
/// before it fails.
#[derive(Default)]
struct Ledger {
    failures: Vec<String>,
}

impl Ledger {
    /// `value` must be BELOW `bar` (an anchor).
    fn below(&mut self, name: &str, value: f64, bar: f64) {
        let ok = value.is_finite() && value < bar;
        self.push(name, value, "<", bar, ok);
    }
    /// `value` must be ABOVE `bar` (a mutant / a vacuity guard).
    fn above(&mut self, name: &str, value: f64, bar: f64) {
        let ok = value.is_finite() && value > bar;
        self.push(name, value, ">", bar, ok);
    }
    fn push(&mut self, name: &str, value: f64, rel: &str, bar: f64, ok: bool) {
        let row = format!(
            "{} {name}: {value:.2e} (must be {rel} {bar:.0e})",
            if ok { "  ok  " } else { " FAIL " }
        );
        eprintln!("{row}");
        if !ok {
            self.failures.push(row);
        }
    }
    fn finish(self) {
        assert!(
            self.failures.is_empty(),
            "{} check(s) failed:\n{}",
            self.failures.len(),
            self.failures.join("\n")
        );
    }
}

// ------------------------------------------------------------ basis / ECP data

/// Unit-normalised generally contracted Cartesian shell.
fn shell(l: i32, exps: &[f64], coefs: &[f64]) -> Shell {
    let lf = l as f64;
    let mut s = 0.0;
    for (a, ca) in exps.iter().zip(coefs) {
        for (b, cb) in exps.iter().zip(coefs) {
            s += ca * cb * (2.0 * (a * b).sqrt() / (a + b)).powf(lf + 1.5);
        }
    }
    Shell {
        l,
        pure: false,
        exponents: exps.to_vec(),
        coefficients: coefs.iter().map(|c| c / s.sqrt()).collect(),
    }
}

/// `(r_exp, exponent, coefficient)` rows of one ECP channel.
fn ecp_channel(l: i32, t: &[(i32, f64, f64)]) -> EcpShell {
    EcpShell {
        angular_momentum: l,
        terms: t
            .iter()
            .map(|&(n, z, d)| EcpTerm {
                coef: d,
                r_exp: n,
                gexp: z,
            })
            .collect(),
    }
}

/// LANL2DZ ECP of I (Hay-Wadt; libecpint `raw/lanl2dz.ecp`, `ECP,i`).
fn lanl2dz_i_ecp() -> EcpDef {
    EcpDef {
        n_core: 46,
        shells: vec![
            ecp_channel(
                3,
                &[
                    (0, 1.0715702, -0.0747621),
                    (1, 44.1936028, -30.0811224),
                    (2, 12.9367609, -75.3722721),
                    (2, 3.1956412, -22.0563758),
                    (2, 0.8589806, -1.6979585),
                ],
            ),
            ecp_channel(
                0,
                &[
                    (0, 127.9202670, 2.9380036),
                    (1, 78.6211465, 41.2471267),
                    (2, 36.5146237, 287.8680095),
                    (2, 9.9065681, 114.3758506),
                    (2, 1.9420086, 37.6547714),
                ],
            ),
            ecp_channel(
                1,
                &[
                    (0, 13.0035304, 2.2222630),
                    (1, 76.0331404, 39.4090831),
                    (2, 24.1961684, 177.4075002),
                    (2, 6.4053433, 77.9889462),
                    (2, 1.5851786, 25.7547641),
                ],
            ),
            ecp_channel(
                2,
                &[
                    (0, 40.4278108, 7.0524360),
                    (1, 28.9084375, 33.3041635),
                    (2, 15.6268936, 186.9453875),
                    (2, 4.1442856, 71.9688361),
                    (2, 0.9377235, 9.3630657),
                ],
            ),
        ],
    }
}

/// LANL2DZ ECP of As (Hay-Wadt; libecpint `raw/lanl2dz.ecp`, `ECP,as`):
/// n_core 28, local channel l = 3 then l = 0, 1, 2.
fn lanl2dz_as_ecp() -> EcpDef {
    EcpDef {
        n_core: 28,
        shells: vec![
            ecp_channel(
                3,
                &[
                    (1, 375.4748803, -28.0000000),
                    (2, 72.6554769, -198.3687357),
                    (2, 15.9162677, -60.4563798),
                    (2, 4.6420251, -22.3188274),
                    (2, 1.4789182, -2.1638290),
                ],
            ),
            ecp_channel(
                0,
                &[
                    (0, 105.8853870, 3.0000000),
                    (1, 62.5951846, 31.8486777),
                    (2, 34.5017431, 200.9958215),
                    (2, 10.3758594, 102.6387604),
                    (2, 2.5379347, 34.1845359),
                ],
            ),
            ecp_channel(
                1,
                &[
                    (0, 37.6471440, 5.0000000),
                    (1, 17.2034665, 24.2505747),
                    (2, 18.9075709, 48.9174963),
                    (2, 6.8153452, 66.5332259),
                    (2, 2.0611853, 19.6577666),
                ],
            ),
            ecp_channel(
                2,
                &[
                    (0, 52.0731335, 3.0000000),
                    (1, 28.0037824, 23.3460712),
                    (2, 14.7051208, 80.7854919),
                    (2, 4.7973194, 40.2681347),
                    (2, 1.1062763, 4.8332790),
                ],
            ),
        ],
    }
}

/// LANL2DZ ECP of Sb (Hay-Wadt; libecpint `raw/lanl2dz.ecp`, `ECP,sb`):
/// n_core 46, local channel l = 3 then l = 0, 1, 2.
fn lanl2dz_sb_ecp() -> EcpDef {
    EcpDef {
        n_core: 46,
        shells: vec![
            ecp_channel(
                3,
                &[
                    (0, 227.2723558, -0.0692261),
                    (1, 77.1429882, -31.5434712),
                    (2, 26.0116501, -115.8407306),
                    (2, 9.1421159, -42.0224212),
                    (2, 2.9281082, -20.0463938),
                    (2, 0.9231136, -2.8796028),
                ],
            ),
            ecp_channel(
                0,
                &[
                    (0, 116.0010888, 2.9331303),
                    (1, 59.6755065, 44.1104819),
                    (2, 27.6135849, 228.0198050),
                    (2, 7.8045081, 92.5610162),
                    (2, 1.6934418, 31.2537244),
                ],
            ),
            ecp_channel(
                1,
                &[
                    (0, 27.8050796, 4.9947369),
                    (1, 10.3929685, 38.0223860),
                    (2, 3.6381067, 62.0520654),
                    (2, 1.9839681, -49.5489389),
                    (2, 1.5546562, 51.3666000),
                ],
            ),
            ecp_channel(
                2,
                &[
                    (0, 83.2183030, 2.9738470),
                    (1, 30.3070501, 41.6495825),
                    (2, 13.2014120, 104.5919562),
                    (2, 3.6961611, 42.3187857),
                    (2, 0.7711144, 7.9413503),
                ],
            ),
        ],
    }
}

/// The rotating basis: H cc-pVDZ; I s (3.0, 1.0, 0.35) / p (2.0, 0.7)
/// (`pbc_kpoint_ecp_rotation.rs`), As s (3.4, 1.2, 0.42) / p (2.2, 0.75)
/// and Sb s (2.6, 0.9, 0.32) / p (1.6, 0.55), each as a general contraction
/// plus a single-primitive column over the same exponent list (so the SR
/// walks rotate columns ON EACH ECP CENTRE). I, As and Sb carry their
/// LANL2DZ ECP.
fn rot_basis() -> BasisSet {
    let mut shells = HashMap::new();
    let h = basis::bundled("cc-pvdz").expect("cc-pvdz");
    shells.insert(1, h.shells.get(&1).expect("H cc-pvdz").clone());
    let heavy = |s_exps: &[f64; 3], p_exps: &[f64; 2]| {
        vec![
            shell(0, s_exps, &[0.3, 0.5, 0.4]),
            shell(0, s_exps, &[0.0, 0.0, 1.0]),
            shell(1, p_exps, &[0.4, 0.8]),
            shell(1, p_exps, &[0.0, 1.0]),
        ]
    };
    shells.insert(53, heavy(&[3.0, 1.0, 0.35], &[2.0, 0.7]));
    shells.insert(33, heavy(&[3.4, 1.2, 0.42], &[2.2, 0.75]));
    shells.insert(51, heavy(&[2.6, 0.9, 0.32], &[1.6, 0.55]));
    let mut ecps = HashMap::new();
    ecps.insert(53, lanl2dz_i_ecp());
    ecps.insert(33, lanl2dz_as_ecp());
    ecps.insert(51, lanl2dz_sb_ecp());
    BasisSet {
        name: "HI-AsSb-rot".into(),
        shells,
        ecps,
    }
}

/// Even-tempered single-primitive aux: s/p on H and I (the aux of
/// `pbc_kpoint_ecp_rotation.rs`), s/p/d on As and Sb (pure d: libint2's
/// 2/3-centre integrals need solid harmonics above l = 1).
fn aux_basis() -> BasisSet {
    // libint2's 2/3-centre integrals need solid-harmonic shells above l = 1.
    let sh = |l: i32, a: f64| Shell {
        l,
        pure: l >= 2,
        exponents: vec![a],
        coefficients: vec![1.0],
    };
    let mut m = HashMap::new();
    m.insert(
        1,
        vec![
            sh(0, 4.7),
            sh(0, 1.9),
            sh(0, 0.75),
            sh(0, 0.3),
            sh(1, 1.25),
            sh(1, 0.5),
        ],
    );
    m.insert(
        53,
        vec![
            sh(0, 2.0),
            sh(0, 0.8),
            sh(0, 0.3),
            sh(1, 1.2),
            sh(1, 0.5),
            sh(1, 0.2),
        ],
    );
    for (z, a) in [(33, 1.25), (51, 1.0)] {
        m.insert(
            z,
            vec![
                sh(0, 2.2 * a),
                sh(0, 0.85 * a),
                sh(0, 0.32 * a),
                sh(1, 1.3 * a),
                sh(1, 0.52 * a),
                sh(1, 0.21 * a),
                sh(2, 1.4 * a),
                sh(2, 0.5 * a),
            ],
        );
    }
    BasisSet {
        name: "hi-assb-aux".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

// ------------------------------------------------------------ cells

/// One atom kind of a fixture: (symbol, Z).
type Kind = (&'static str, i32);

const HI: [Kind; 2] = [("H", 1), ("I", 53)];
const ASSB: [Kind; 2] = [("As", 33), ("Sb", 51)];

/// HI positions (`pbc_kpoint_ecp_rotation.rs` MOVED, Bohr).
const HI_POS: [[f64; 3]; 2] = [[0.3, 0.2, 0.4], [0.9, -0.4, 3.3]];
const HI_ORTHO: Mat3 = [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]];
const HI_TRI: Mat3 = [[6.0, 0.0, 0.0], [1.0, 6.0, 0.0], [1.2, -0.8, 7.0]];
/// A wider skewed cell for the 2x2x1 mesh: on `HI_TRI` (6 Bohr rows) the MP
/// 2x2x1 k-RHF limit-cycles under DIIS (400 iterations; the orthorhombic
/// 6 Bohr cell converges in 16), on these 8 Bohr rows it converges in 22.
const HI_TRI8: Mat3 = [[8.0, 0.0, 0.0], [1.3, 8.0, 0.0], [1.2, -0.8, 7.0]];
/// AsSb (As-Sb 4.68 Bohr, off-axis; Sb to the nearest As image 5.45 Bohr).
const ASSB_POS: [[f64; 3]; 2] = [[0.9, -0.4, 3.3], [0.4, 0.3, 7.9]];
const ASSB_A: Mat3 = [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 10.0]];
/// AsSb SEPTET: two high-spin 4S (s2 p3) atoms, 10 valence electrons,
/// 8 alpha / 2 beta (6 unpaired; every spin shell closed, no degenerate
/// partial occupation, so the k-UHF aufbau is clean).
const ASSB_SPIN: (usize, usize) = (8, 2);

fn atom(kind: Kind, r: [f64; 3]) -> Atom {
    Atom {
        symbol: kind.0.into(),
        z: kind.1,
        x: r[0],
        y: r[1],
        zpos: r[2],
        ghost: false,
        n_core_ecp: 0,
    }
}

/// A cell of `kinds` at `pos`, translated by every `shifts` entry (an
/// explicit supercell when `shifts` has more than one entry).
fn ecp_cell(
    kinds: &[Kind],
    pos: &[[f64; 3]],
    shifts: &[[f64; 3]],
    lattice: Mat3,
    mult: usize,
    bs: &BasisSet,
) -> Cell {
    let mut atoms = Vec::new();
    for t in shifts {
        for (k, p) in kinds.iter().zip(pos) {
            atoms.push(atom(*k, [p[0] + t[0], p[1] + t[1], p[2] + t[2]]));
        }
    }
    let mut mol = Molecule {
        atoms,
        charge: 0,
        multiplicity: mult,
    };
    mol.apply_ecp(bs);
    Cell::new(mol, lattice).expect("ECP cell")
}

/// A fixture: atom kinds, positions, lattice and spin (None = closed shell).
#[derive(Clone, Copy)]
struct Fixture {
    name: &'static str,
    kinds: &'static [Kind],
    pos: &'static [[f64; 3]],
    lattice: Mat3,
    spin: Option<(usize, usize)>,
}

const ASSB_SEPTET: Fixture = Fixture {
    name: "AsSb septet",
    kinds: &ASSB,
    pos: &ASSB_POS,
    lattice: ASSB_A,
    spin: Some(ASSB_SPIN),
};
const HI_TRICLINIC: Fixture = Fixture {
    name: "HI triclinic",
    kinds: &HI,
    pos: &HI_POS,
    lattice: HI_TRI,
    spin: None,
};
const HI_TRICLINIC8: Fixture = Fixture {
    name: "HI triclinic (8 Bohr rows)",
    kinds: &HI,
    pos: &HI_POS,
    lattice: HI_TRI8,
    spin: None,
};
const HI_ORTHORHOMBIC: Fixture = Fixture {
    name: "HI orthorhombic",
    kinds: &HI,
    pos: &HI_POS,
    lattice: HI_ORTHO,
    spin: None,
};

impl Fixture {
    fn mult(&self, ncell: usize) -> usize {
        self.spin.map_or(1, |(a, b)| ncell * (a - b) + 1)
    }
    fn cell_at(&self, pos: &[[f64; 3]], bs: &BasisSet) -> Cell {
        ecp_cell(self.kinds, pos, &[[0.0; 3]], self.lattice, self.mult(1), bs)
    }
    fn cell(&self, bs: &BasisSet) -> Cell {
        self.cell_at(self.pos, bs)
    }
    /// The explicit 1x1x`n` supercell (rows a1, a2, n a3; copies along a3).
    fn supercell_z(&self, n: usize, bs: &BasisSet) -> Cell {
        let a3 = self.lattice[2];
        let shifts: Vec<[f64; 3]> = (0..n).map(|m| a3.map(|v| v * m as f64)).collect();
        let mut lat = self.lattice;
        for v in lat[2].iter_mut() {
            *v *= n as f64;
        }
        ecp_cell(self.kinds, self.pos, &shifts, lat, self.mult(n), bs)
    }
    fn moved(&self, a: usize, x: usize, h: f64, bs: &BasisSet) -> Cell {
        let mut p = self.pos.to_vec();
        p[a][x] += h;
        self.cell_at(&p, bs)
    }
}

// ------------------------------------------------------------ k pipeline

fn hcore_cfg(rot: SrColumnRotation) -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
    .with_sr_column_rotation(rot)
}

fn gdf_cfg(rot: SrColumnRotation) -> RsGdfConfig {
    RsGdfConfig {
        omega: 1.0,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        sr_column_rotation: rot,
        ..Default::default()
    }
}

fn kscf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    }
}

enum KScf {
    R(KScfResult),
    U(KUScfResult),
}

impl KScf {
    fn energy(&self) -> f64 {
        match self {
            KScf::R(r) => r.energy,
            KScf::U(u) => u.energy,
        }
    }
}

#[derive(Clone, Copy)]
struct Mesh {
    n: [usize; 3],
    centring: MeshCentring,
    rot: SrColumnRotation,
}

/// One k pipeline: rotated k hcore, RS-GDF, k-RHF or k-UHF (exxdiv none).
struct KRun {
    cell: Cell,
    prep: PreparedBasis,
    mesh: KPointMesh,
    hcfg: PeriodicHcoreConfig,
    hk: PeriodicHcoreK,
    cfg: KRsGdfConfig,
    aux: PreparedBasis,
    gdf: KRsGdf,
    scf: KScf,
}

fn krun(fx: &Fixture, cell: Cell, bs: &BasisSet, m: Mesh) -> KRun {
    let prep = prep_for(&cell, bs);
    let mesh = KPointMesh::new(&cell, m.n, m.centring).expect("mesh");
    let hcfg = hcore_cfg(m.rot);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcfg).expect("hcore(k)");
    assert!(hk.n_ecp_triples > 0, "no ECP triples: the cell is vacuous");
    let cfg = KRsGdfConfig {
        gdf: gdf_cfg(m.rot),
        mutation: None,
    };
    let aux = PreparedBasis::new(cell.mol(), &aux_basis()).expect("aux prep");
    let gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &cfg).expect("KRsGdf");
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk: Box::new(gdf.jk_builder_with_madelung(0.0)),
    };
    let scf = match fx.spin {
        None => {
            let r = solve_krhf_injected(&cell, &mesh, &kscf_cfg(), inj).expect("k-RHF");
            assert!(r.converged, "k-RHF not converged ({} it)", r.iterations);
            KScf::R(r)
        }
        Some((na, nb)) => {
            let r = solve_kuhf_injected(&cell, &mesh, &kscf_cfg(), inj, na, nb).expect("k-UHF");
            assert!(r.converged, "k-UHF not converged ({} it)", r.iterations);
            KScf::U(r)
        }
    };
    KRun {
        cell,
        prep,
        mesh,
        hcfg,
        hk,
        cfg,
        aux,
        gdf,
        scf,
    }
}

impl KRun {
    fn src(&self) -> KRsGdfGradSource<'_> {
        KRsGdfGradSource {
            gdf: &self.gdf,
            cfg: &self.cfg,
            aux: &self.aux,
        }
    }
    fn force(&self, m: Option<KGradMutation>) -> KGradient {
        let jk = KGradJk::RsGdf(self.src());
        let c = gcfg(m);
        let (cell, prep, mesh, hcfg, hk) =
            (&self.cell, &self.prep, &self.mesh, &self.hcfg, &self.hk);
        match &self.scf {
            KScf::R(r) => kpoint_rhf_gradient(cell, prep, mesh, hcfg, hk, jk, r, ExxDiv::None, &c),
            KScf::U(u) => kpoint_uhf_gradient(cell, prep, mesh, hcfg, hk, jk, u, ExxDiv::None, &c),
        }
        .expect("k force")
    }
    fn stress(&self, m: Option<KStressMutation>) -> KStress {
        let jk = KStressJk::RsGdf(self.src());
        let c = scfg(m);
        let (cell, prep, mesh, hcfg, hk) =
            (&self.cell, &self.prep, &self.mesh, &self.hcfg, &self.hk);
        match &self.scf {
            KScf::R(r) => kpoint_rhf_stress(cell, prep, mesh, hcfg, hk, jk, r, ExxDiv::None, &c),
            KScf::U(u) => kpoint_uhf_stress(cell, prep, mesh, hcfg, hk, jk, u, ExxDiv::None, &c),
        }
        .expect("k stress")
    }
    /// Per-q kept aux counts and the per-k occupation pattern: an FD point
    /// whose metric cut or occupations changed has a piecewise energy.
    fn signature(&self) -> (Vec<usize>, Vec<Vec<u64>>) {
        let kept = self.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect();
        let occ = match &self.scf {
            KScf::R(_) => Vec::new(),
            KScf::U(u) => u
                .occ_alpha
                .iter()
                .chain(&u.occ_beta)
                .map(|o| o.iter().map(|v| v.to_bits()).collect())
                .collect(),
        };
        (kept, occ)
    }
    /// The Gamma (unfolded) SCF result of this run's density.
    fn gamma_density(&self, unfolded: bool) -> ScfResult {
        let one = |d: &[Array2<Complex64>]| {
            if unfolded {
                unfold(&self.mesh, d)
            } else {
                real_of(&d[0])
            }
        };
        match &self.scf {
            KScf::R(r) => gamma_scf(one(&r.densities), None),
            KScf::U(u) => gamma_scf(one(&u.density_alpha), Some(one(&u.density_beta))),
        }
    }
    /// The spin sectors are populated unequally, so a spin-restricted-looking
    /// solution cannot make an open-shell anchor vacuous.
    fn assert_polarised(&self, fx: &Fixture) {
        if let KScf::U(u) = &self.scf {
            let mag: f64 = u
                .density_alpha
                .iter()
                .zip(&u.density_beta)
                .map(|(a, b)| (a - b).iter().map(|z| z.norm()).sum::<f64>())
                .sum();
            eprintln!(
                "  {}: spin ({}, {}), sum|D_a - D_b| {mag:.3}, E {:.10}",
                fx.name, u.nalpha, u.nbeta, u.energy
            );
            assert!(mag > 1.0, "{}: not spin polarised", fx.name);
        }
    }
}

/// The Gamma RS-GDF pipeline of a cell (the cell itself or a supercell),
/// rotation as the k side.
struct GRun {
    cell: Cell,
    prep: PreparedBasis,
    hcfg: PeriodicHcoreConfig,
    hc: PeriodicHcore,
    aux: PreparedBasis,
    gdf: RsGdf,
}

fn grun(cell: Cell, bs: &BasisSet, rot: SrColumnRotation) -> GRun {
    let prep = prep_for(&cell, bs);
    let hcfg = hcore_cfg(rot);
    let hc = periodic_hcore(&cell, &prep, &hcfg).expect("gamma hcore");
    let aux = PreparedBasis::new(cell.mol(), &aux_basis()).expect("aux prep");
    let gdf = RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, &gdf_cfg(rot)).expect("RsGdf");
    GRun {
        cell,
        prep,
        hcfg,
        hc,
        aux,
        gdf,
    }
}

impl GRun {
    fn src(&self) -> RsGdfGradSource<'_> {
        RsGdfGradSource {
            gdf: &self.gdf,
            aux: &self.aux,
            aux_jac: None,
        }
    }
    fn force(&self, scf: &ScfResult) -> ferric_pbc::grad::GammaGradient {
        let c = GammaGradConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        };
        let (cell, prep, hcfg, hc, src) =
            (&self.cell, &self.prep, &self.hcfg, &self.hc, &self.src());
        match scf.spin {
            Spin::Restricted => {
                gamma_rhf_gradient_rsgdf(cell, prep, hcfg, hc, src, scf, ExxDiv::None, &c)
            }
            _ => gamma_uhf_gradient_rsgdf(cell, prep, hcfg, hc, src, scf, ExxDiv::None, &c),
        }
        .expect("gamma force")
    }
    fn stress(&self, scf: &ScfResult) -> GammaStress {
        let c = GammaStressConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        };
        let (cell, prep, hcfg, hc, src) =
            (&self.cell, &self.prep, &self.hcfg, &self.hc, &self.src());
        match scf.spin {
            Spin::Restricted => {
                gamma_rhf_stress_rsgdf(cell, prep, hcfg, hc, src, scf, ExxDiv::None, &c)
            }
            _ => gamma_uhf_stress_rsgdf(cell, prep, hcfg, hc, src, scf, ExxDiv::None, &c),
        }
        .expect("gamma stress")
    }
}

/// Central FD (h = `FD_H`) of the k energy of `fx` on mesh `m`: force
/// components `(atom, axis)` and strain components `(i, j)`; every point
/// must keep the base run's metric cut and occupation pattern.
fn fd(
    fx: &Fixture,
    bs: &BasisSet,
    m: Mesh,
    base: &KRun,
    forces: &[(usize, usize)],
    strains: &[(usize, usize)],
) -> (Vec<f64>, Vec<f64>) {
    let sig0 = base.signature();
    let e = |cell: Cell| {
        let r = krun(fx, cell, bs, m);
        assert_eq!(
            r.signature(),
            sig0,
            "metric cut / occupations changed (E piecewise)"
        );
        r.scf.energy()
    };
    let fd_f = forces
        .iter()
        .map(|&(a, x)| (e(fx.moved(a, x, FD_H, bs)) - e(fx.moved(a, x, -FD_H, bs))) / (2.0 * FD_H))
        .collect();
    let c0 = fx.cell(bs);
    let fd_s = strains
        .iter()
        .map(|&(i, j)| {
            let ep = e(c0.strained(&eps_at(i, j, FD_H)).expect("strain"));
            let em = e(c0.strained(&eps_at(i, j, -FD_H)).expect("strain"));
            (ep - em) / (2.0 * FD_H)
        })
        .collect();
    (fd_f, fd_s)
}

fn fd_miss_f(g: &KGradient, comps: &[(usize, usize)], fd: &[f64]) -> f64 {
    comps
        .iter()
        .zip(fd)
        .fold(0.0_f64, |w, (&(a, x), v)| w.max((g.grad[(a, x)] - v).abs()))
}

fn fd_miss_s(s: &KStress, comps: &[(usize, usize)], fd: &[f64]) -> f64 {
    comps.iter().zip(fd).fold(0.0_f64, |w, (&(i, j), v)| {
        w.max((s.de_deps[i][j] - v).abs())
    })
}

fn on() -> SrColumnRotation {
    SrColumnRotation::on()
}

fn gamma_mesh(n: [usize; 3]) -> Mesh {
    Mesh {
        n,
        centring: MeshCentring::Gamma,
        rot: on(),
    }
}

fn mp_mesh(n: [usize; 3]) -> Mesh {
    Mesh {
        n,
        centring: MeshCentring::MonkhorstPack,
        rot: on(),
    }
}

/// Both ECP centres carry an ECP force and every ECP part is non-trivial
/// (AsSb: |F_ecp| 7.3e-4 per atom, |S_ecp| 1.6e-2; >= 5 orders above every
/// anchor bar).
fn ecp_vacuity(led: &mut Ledger, tag: &str, f: &KGradient, s: &KStress) {
    for a in 0..f.parts.ecp.nrows() {
        let fa = (0..3).fold(0.0_f64, |w, x| w.max(f.parts.ecp[(a, x)].abs()));
        led.above(&format!("{tag} |F_ecp| atom {a} (vacuity)"), fa, 1e-4);
    }
    led.above(
        &format!("{tag} |S_ecp| (vacuity)"),
        max_mat(&s.parts.ecp, &ZERO),
        1e-3,
    );
}

// ============================================================ (D) AsSb septet

/// (D) bars: >= 10x the measured values in the test docs.
const D_GAMMA_BAR: f64 = 5e-12;
const D_SC_BAR: f64 = 3e-11;
const D_FD_F_BAR: f64 = 1e-8;
const D_FD_S_BAR: f64 = 1e-6;

/// PHYSICS: at 1x1x1 the k-UHF RS-GDF force and stress of the AsSb septet
/// (two distinct ECP elements, rotating basis) are the Gamma-UHF RS-GDF
/// ones: total and ECP part. ARTIFACT: the spin and multi-ECP mutants must
/// miss (they are not k-specific): `EcpCentreWrongAtom` folds each ECP
/// centre derivative onto the OTHER ECP atom (on HI it folded onto H).
/// MEASURED: force total 2.4e-14, stress total 2.9e-13, ECP parts 0
/// (same kernel both sides, expected); |F_ecp| 7.4e-4 per atom, |S_ecp|
/// 1.6e-2. Mutant misses: EcpCentreWrongAtom 6.9e-3, EcpCentreSign 2.9e-2,
/// NoEcp 7.4e-4 (force); FitCrossSpin 7.9e-1, SwapSpinFock 1.9e-2,
/// EcpNoCentre 1.4e-1, EcpNoKetImage 6.7e-3, EcpUnitPairVector 1.2e-1
/// (stress). `RotSwapPair` and `RotFitWrongResidue` are identities at 1x1x1
/// on this cell (measured 2.4e-14, the anchor floor) and act at 1x1x3.
#[test]
#[ignore = "validation: k-point force/stress #421 coverage (open-shell septet, two ECP elements, rotation; 1x1x1 vs Gamma)"]
fn two_ecp_element_septet_one_point_mesh_is_the_gamma_one() {
    let bs = rot_basis();
    let fx = ASSB_SEPTET;
    let r = krun(&fx, fx.cell(&bs), &bs, gamma_mesh([1, 1, 1]));
    r.assert_polarised(&fx);
    assert_eq!(
        r.hk.sr_rotated_columns, 4,
        "both ECP centres rotate (2 + 2)"
    );
    let g = grun(fx.cell(&bs), &bs, on());
    let gscf = r.gamma_density(false);
    let (kf, ks) = (r.force(None), r.stress(None));
    let (gf, gs) = (g.force(&gscf), g.stress(&gscf));
    let mut led = Ledger::default();
    let tag = "AsSb septet 1x1x1";
    ecp_vacuity(&mut led, tag, &kf, &ks);
    led.below(
        &format!("{tag} force TOTAL"),
        grad_miss(&kf.grad, &gf.grad),
        D_GAMMA_BAR,
    );
    led.below(
        &format!("{tag} force ECP"),
        grad_miss(&kf.parts.ecp, &gf.parts.ecp),
        D_GAMMA_BAR,
    );
    led.below(
        &format!("{tag} stress TOTAL"),
        max_mat(&ks.de_deps, &gs.de_deps),
        D_GAMMA_BAR,
    );
    led.below(
        &format!("{tag} stress ECP"),
        max_mat(&ks.parts.ecp, &gs.parts.ecp),
        D_GAMMA_BAR,
    );
    for m in [
        KGradMutation::EcpCentreWrongAtom,
        KGradMutation::EcpCentreSign,
        KGradMutation::NoEcp,
    ] {
        led.above(
            &format!("{tag} force mutant {m:?}"),
            grad_miss(&r.force(Some(m)).grad, &gf.grad),
            MUTANT_BAR,
        );
    }
    for m in [
        KStressMutation::FitCrossSpin,
        KStressMutation::SwapSpinFock,
        KStressMutation::EcpNoCentre,
        KStressMutation::EcpNoKetImage,
        KStressMutation::EcpUnitPairVector,
    ] {
        led.above(
            &format!("{tag} stress mutant {m:?}"),
            max_mat(&r.stress(Some(m)).de_deps, &gs.de_deps),
            MUTANT_BAR,
        );
    }
    led.finish();
}

/// PHYSICS: the Gamma-centred 1x1x3 (non-TRIM) k-UHF RS-GDF force (every
/// copy) and stress / 3 of the AsSb septet equal the explicit 3-cell
/// supercell's (6 ECP atoms, multiplicity 19) Gamma-UHF ones, total and ECP
/// part; the k, ECP-residue, rotation-residue and spin mutants miss.
/// MEASURED: force total 9.6e-14 (ECP part 1.7e-17), stress total 1.5e-12
/// (ECP 7.7e-16). Mutant misses (force): NoPhase 3.7e-3, ConjD 2.0e-3,
/// NoInvNk 5.6e-2, EcpWrongResidue 2.6e-5 (the smallest, 880x D_SC_BAR),
/// EcpCentreWrongAtom 2.7e-3, RotHcoreWrongResidue 3.7e-3,
/// RotFitWrongResidue 3.7e-2, FitAuxPhase 7.2e-4; (stress): NoPhase 2.1e-2,
/// ConjD 1.1e-2, EcpWrongResidue 1.5e-4, RotHcoreWrongResidue 2.1e-2,
/// FitCrossSpin 2.6, SwapSpinFock 5.0e-2.
#[test]
#[ignore = "validation: k-point force/stress #421 coverage (open-shell septet, two ECP elements, rotation; 1x1x3 vs supercell)"]
fn two_ecp_element_septet_1x1x3_is_the_supercell_per_cell() {
    let bs = rot_basis();
    let fx = ASSB_SEPTET;
    let r = krun(&fx, fx.cell(&bs), &bs, gamma_mesh([1, 1, 3]));
    r.assert_polarised(&fx);
    let g = grun(fx.supercell_z(3, &bs), &bs, on());
    assert_eq!(g.hc.sr_rotated_columns, 12);
    let gscf = r.gamma_density(true);
    let (kf, ks) = (r.force(None), r.stress(None));
    let (gf, gs) = (g.force(&gscf), g.stress(&gscf));
    let sref = scale3(&gs.de_deps, 1.0 / 3.0);
    let mut led = Ledger::default();
    let tag = "AsSb septet 1x1x3";
    ecp_vacuity(&mut led, tag, &kf, &ks);
    led.below(
        &format!("{tag} force TOTAL vs sc copies"),
        sc_miss(&kf.grad, &gf.grad),
        D_SC_BAR,
    );
    led.below(
        &format!("{tag} force ECP vs sc copies"),
        sc_miss(&kf.parts.ecp, &gf.parts.ecp),
        D_SC_BAR,
    );
    led.below(
        &format!("{tag} stress TOTAL vs sc/3"),
        max_mat(&ks.de_deps, &sref),
        D_SC_BAR,
    );
    led.below(
        &format!("{tag} stress ECP vs sc/3"),
        max_mat(&ks.parts.ecp, &scale3(&gs.parts.ecp, 1.0 / 3.0)),
        D_SC_BAR,
    );
    for m in [
        KGradMutation::NoPhase,
        KGradMutation::ConjD,
        KGradMutation::NoInvNk,
        KGradMutation::EcpWrongResidue,
        KGradMutation::EcpCentreWrongAtom,
        KGradMutation::RotHcoreWrongResidue,
        KGradMutation::RotFitWrongResidue,
        KGradMutation::FitAuxPhase,
    ] {
        led.above(
            &format!("{tag} force mutant {m:?}"),
            sc_miss(&r.force(Some(m)).grad, &gf.grad),
            MUTANT_BAR,
        );
    }
    for m in [
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
        KStressMutation::EcpWrongResidue,
        KStressMutation::RotHcoreWrongResidue,
        KStressMutation::FitCrossSpin,
        KStressMutation::SwapSpinFock,
    ] {
        led.above(
            &format!("{tag} stress mutant {m:?}"),
            max_mat(&r.stress(Some(m)).de_deps, &sref),
            MUTANT_BAR,
        );
    }
    led.finish();
}

/// PHYSICS (independent of every derivative walk and of the Gamma code):
/// the k-UHF analytic force (As x, Sb y, Sb z) and ALL NINE stress
/// components of the AsSb septet on the Gamma-centred 1x1x3 (non-TRIM) mesh
/// vs central FD of its own rotated k energy (fresh hcore, fit and SCF per
/// point; the metric cut and the per-k occupations are held).
/// MEASURED: force 7.8e-10, stress (all nine) 5.6e-8, |offdiag S| 2.1e-2.
/// Mutants vs FD: EcpNoCentre 1.1e-1, FitCrossSpin 2.6, NoAuxFt 8.1e-3
/// (stress); EcpCentreWrongAtom 2.7e-3, RotSwapPair 4.6e-5 (force).
#[test]
#[ignore = "validation: k-point force/stress #421 coverage (open-shell septet, two ECP elements, rotation; FD of the k energy)"]
fn two_ecp_element_septet_force_and_stress_match_fd_of_own_k_energy() {
    let bs = rot_basis();
    let fx = ASSB_SEPTET;
    let m = gamma_mesh([1, 1, 3]);
    let r = krun(&fx, fx.cell(&bs), &bs, m);
    r.assert_polarised(&fx);
    let forces = [(0usize, 0usize), (1, 1), (1, 2)];
    let strains = ALL9;
    let (fd_f, fd_s) = fd(&fx, &bs, m, &r, &forces, &strains);
    let (f, s) = (r.force(None), r.stress(None));
    let mut led = Ledger::default();
    let tag = "AsSb septet 1x1x3";
    ecp_vacuity(&mut led, tag, &f, &s);
    led.above(
        &format!("{tag} offdiag |S| (vacuity)"),
        offdiag_max(&s.de_deps),
        1e-3,
    );
    led.below(
        &format!("{tag} force vs FD"),
        fd_miss_f(&f, &forces, &fd_f),
        D_FD_F_BAR,
    );
    led.below(
        &format!("{tag} stress vs FD (all nine)"),
        fd_miss_s(&s, &strains, &fd_s),
        D_FD_S_BAR,
    );
    for m in [
        KStressMutation::EcpNoCentre,
        KStressMutation::FitCrossSpin,
        KStressMutation::NoAuxFt,
    ] {
        led.above(
            &format!("{tag} stress mutant {m:?} vs FD"),
            fd_miss_s(&r.stress(Some(m)), &strains, &fd_s),
            MUTANT_BAR,
        );
    }
    for m in [
        KGradMutation::EcpCentreWrongAtom,
        KGradMutation::RotSwapPair,
    ] {
        led.above(
            &format!("{tag} force mutant {m:?} vs FD"),
            fd_miss_f(&r.force(Some(m)), &forces, &fd_f),
            MUTANT_BAR,
        );
    }
    led.finish();
}

// ============================================================ (E) triclinic 1x1x3

const E_SC_BAR: f64 = 3e-11;
const E_FD_F_BAR: f64 = 4e-8;
const E_FD_S_BAR: f64 = 3e-7;

/// PHYSICS: the Gamma-centred 1x1x3 (non-TRIM) k-RHF RS-GDF force (every
/// copy) and stress / 3 of the TRICLINIC HI cell with the ECP and rotation
/// equal the explicit 3-cell supercell's (rows a1, a2, 3 a3); the k-specific
/// ECP mutant `EcpWrongResidue` (identity on the 1x1x2 mesh of #476) and the
/// complex-k mutants act here and must miss.
/// MEASURED: force total 2.8e-13 (ECP 7.2e-15), stress total 8.1e-13 (ECP
/// 8.0e-15), |offdiag S| 2.4e-1. Mutant misses (force / stress): NoPhase
/// 7.1e-2 / 3.3e-1, ConjD 2.0e-1 / 5.1e-1, EcpWrongResidue 2.1e-1 / 8.5e-1,
/// RotHcoreWrongResidue 1.5e-1 (force), RotFitWrongResidue 1.5e-1 / 7.1e-1,
/// EcpUnitPairVector 1.3 (stress).
#[test]
#[ignore = "validation: k-point force/stress #421 coverage (triclinic ECP cell, rotation, 1x1x3 vs supercell)"]
fn triclinic_ecp_1x1x3_is_the_supercell_per_cell() {
    let bs = rot_basis();
    let fx = HI_TRICLINIC;
    let r = krun(&fx, fx.cell(&bs), &bs, gamma_mesh([1, 1, 3]));
    assert_eq!(r.mesh.residue_moduli(), [1, 1, 3]);
    let g = grun(fx.supercell_z(3, &bs), &bs, on());
    assert_eq!(g.hc.sr_rotated_columns, 9);
    let gscf = r.gamma_density(true);
    let (kf, ks) = (r.force(None), r.stress(None));
    let (gf, gs) = (g.force(&gscf), g.stress(&gscf));
    let sref = scale3(&gs.de_deps, 1.0 / 3.0);
    let mut led = Ledger::default();
    let tag = "HI triclinic 1x1x3";
    led.above(
        &format!("{tag} |F_ecp| (vacuity)"),
        amax(&kf.parts.ecp),
        1e-3,
    );
    led.above(
        &format!("{tag} offdiag |S| (vacuity)"),
        offdiag_max(&ks.de_deps),
        1e-2,
    );
    led.below(
        &format!("{tag} force TOTAL vs sc copies"),
        sc_miss(&kf.grad, &gf.grad),
        E_SC_BAR,
    );
    led.below(
        &format!("{tag} force ECP vs sc copies"),
        sc_miss(&kf.parts.ecp, &gf.parts.ecp),
        E_SC_BAR,
    );
    led.below(
        &format!("{tag} stress TOTAL vs sc/3"),
        max_mat(&ks.de_deps, &sref),
        E_SC_BAR,
    );
    led.below(
        &format!("{tag} stress ECP vs sc/3"),
        max_mat(&ks.parts.ecp, &scale3(&gs.parts.ecp, 1.0 / 3.0)),
        E_SC_BAR,
    );
    for m in [
        KGradMutation::NoPhase,
        KGradMutation::ConjD,
        KGradMutation::EcpWrongResidue,
        KGradMutation::RotHcoreWrongResidue,
        KGradMutation::RotFitWrongResidue,
    ] {
        led.above(
            &format!("{tag} force mutant {m:?}"),
            sc_miss(&r.force(Some(m)).grad, &gf.grad),
            MUTANT_BAR,
        );
    }
    for m in [
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
        KStressMutation::EcpWrongResidue,
        KStressMutation::EcpUnitPairVector,
        KStressMutation::RotFitWrongResidue,
    ] {
        led.above(
            &format!("{tag} stress mutant {m:?}"),
            max_mat(&r.stress(Some(m)).de_deps, &sref),
            MUTANT_BAR,
        );
    }
    led.finish();
}

/// PHYSICS: analytic force (I x, H z) and stress (zz, xy, yz, xz) of the
/// triclinic ECP cell on the Gamma-centred 1x1x3 mesh vs central FD of its
/// own rotated k energy.
/// MEASURED: force 2.9e-9, stress 2.3e-8; EcpWrongResidue 8.5e-1,
/// EcpUnitPairVector 1.3 vs FD.
#[test]
#[ignore = "validation: k-point force/stress #421 coverage (triclinic ECP cell, rotation, 1x1x3 FD of the k energy)"]
fn triclinic_ecp_1x1x3_matches_fd_of_own_k_energy() {
    let bs = rot_basis();
    let fx = HI_TRICLINIC;
    let m = gamma_mesh([1, 1, 3]);
    let r = krun(&fx, fx.cell(&bs), &bs, m);
    let forces = [(1usize, 0usize), (0, 2)];
    let strains = [(2usize, 2usize), (0, 1), (1, 2), (0, 2)];
    let (fd_f, fd_s) = fd(&fx, &bs, m, &r, &forces, &strains);
    let (f, s) = (r.force(None), r.stress(None));
    let mut led = Ledger::default();
    let tag = "HI triclinic 1x1x3";
    led.below(
        &format!("{tag} force vs FD"),
        fd_miss_f(&f, &forces, &fd_f),
        E_FD_F_BAR,
    );
    led.below(
        &format!("{tag} stress vs FD"),
        fd_miss_s(&s, &strains, &fd_s),
        E_FD_S_BAR,
    );
    for m in [
        KStressMutation::EcpWrongResidue,
        KStressMutation::EcpUnitPairVector,
    ] {
        led.above(
            &format!("{tag} stress mutant {m:?} vs FD"),
            fd_miss_s(&r.stress(Some(m)), &strains, &fd_s),
            MUTANT_BAR,
        );
    }
    led.finish();
}

// ============================================================ (F) MP meshes

const F_FD_F_BAR: f64 = 4e-8;
const F_FD_S_BAR: f64 = 4e-7;
const F_ROT_BAR: f64 = 2e-11;
const F_IDENTITY_BAR: f64 = 2e-11;

/// Shared body of the shifted-mesh FD anchors: FD of force and stress, the
/// shifted-mesh ECP and phase mutants (must move the total), and the
/// rotated vs unrotated analytic of the same energy.
fn shifted_mesh_fd(
    fx: Fixture,
    n: [usize; 3],
    moduli: [usize; 3],
    forces: &[(usize, usize)],
    strains: &[(usize, usize)],
) {
    let bs = rot_basis();
    let m = mp_mesh(n);
    let r = krun(&fx, fx.cell(&bs), &bs, m);
    assert_eq!(r.mesh.residue_moduli(), moduli, "not the shifted mesh");
    assert!(
        r.gdf.stats().sr_rotated_columns > 0,
        "the fit did not rotate"
    );
    let (fd_f, fd_s) = fd(&fx, &bs, m, &r, forces, strains);
    let (f, s) = (r.force(None), r.stress(None));
    let mut led = Ledger::default();
    let tag = format!("{} MP {}x{}x{}", fx.name, n[0], n[1], n[2]);
    led.above(
        &format!("{tag} |F_ecp| (vacuity)"),
        amax(&f.parts.ecp),
        1e-3,
    );
    led.below(
        &format!("{tag} force vs FD"),
        fd_miss_f(&f, forces, &fd_f),
        F_FD_F_BAR,
    );
    led.below(
        &format!("{tag} stress vs FD"),
        fd_miss_s(&s, strains, &fd_s),
        F_FD_S_BAR,
    );
    for mu in [
        KGradMutation::EcpMeshCountModuli,
        KGradMutation::EcpWrongResidue,
        KGradMutation::NoPhase,
        KGradMutation::ConjD,
        KGradMutation::RotFitWrongResidue,
        KGradMutation::RotHcoreWrongResidue,
    ] {
        led.above(
            &format!("{tag} force mutant {mu:?} moves total"),
            grad_miss(&r.force(Some(mu)).grad, &f.grad),
            MUTANT_BAR,
        );
    }
    for mu in [
        KStressMutation::EcpMeshCountModuli,
        KStressMutation::EcpWrongResidue,
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
    ] {
        led.above(
            &format!("{tag} stress mutant {mu:?} moves total"),
            max_mat(&r.stress(Some(mu)).de_deps, &s.de_deps),
            MUTANT_BAR,
        );
    }
    let off = krun(
        &fx,
        fx.cell(&bs),
        &bs,
        Mesh {
            rot: SrColumnRotation::Off,
            ..m
        },
    );
    assert_eq!(off.gdf.stats().sr_rotated_columns, 0);
    led.below(
        &format!("{tag} rotated vs unrotated force"),
        grad_miss(&f.grad, &off.force(None).grad),
        F_ROT_BAR,
    );
    led.below(
        &format!("{tag} rotated vs unrotated stress"),
        max_mat(&s.de_deps, &off.stress(None).de_deps),
        F_ROT_BAR,
    );
    led.finish();
}

/// MP 2x2x1 on a TRICLINIC HI cell (`HI_TRI8`; shifted along the two skewed
/// axes a1, a2; residue moduli [4, 4, 1]). FD force (I x, H y) and stress (xy, xz,
/// zz).
/// MEASURED: FD force 1.2e-9, stress 3.4e-8; rotated vs unrotated 2.6e-14 /
/// 1.3e-12; total moves (force / stress): EcpMeshCountModuli 1.8e-3 /
/// 2.5e-2, EcpWrongResidue 3.1e-5 / 4.6e-4, NoPhase 1.2e-2 / 3.1e-1, ConjD
/// 4.4e-4 / 3.7e-3, RotFitWrongResidue 2.7e-1, RotHcoreWrongResidue 3.5e-1
/// (force).
#[test]
#[ignore = "validation: k-point force/stress #421 coverage (MP 2x2x1 shifted mesh, triclinic ECP cell, rotation)"]
fn triclinic_ecp_mp_2x2x1_force_and_stress_match_fd() {
    shifted_mesh_fd(
        HI_TRICLINIC8,
        [2, 2, 1],
        [4, 4, 1],
        &[(1, 0), (0, 1)],
        &[(0, 1), (0, 2), (2, 2)],
    );
}

/// MP 1x1x4 on the orthorhombic HI cell (k = ±1/8, ±3/8 of b3: complex,
/// non-TRIM shifted points; residue moduli [1, 1, 8]). FD force (I z, H x)
/// and stress (zz, xz).
/// MEASURED: FD force 3.5e-9, stress 2.6e-8; rotated vs unrotated 5.3e-14 /
/// 4.7e-13; total moves (force / stress): EcpMeshCountModuli 2.1e-1 /
/// 8.3e-1, EcpWrongResidue 2.1e-1 / 8.2e-1, NoPhase 3.6e-2 / 8.7e-2, ConjD
/// 2.4e-1 / 6.1e-1, RotFitWrongResidue 1.7e-1, RotHcoreWrongResidue 2.0e-1
/// (force).
#[test]
#[ignore = "validation: k-point force/stress #421 coverage (MP 1x1x4 shifted mesh, ECP, rotation)"]
fn orthorhombic_ecp_mp_1x1x4_force_and_stress_match_fd() {
    shifted_mesh_fd(
        HI_ORTHORHOMBIC,
        [1, 1, 4],
        [1, 1, 8],
        &[(1, 2), (0, 0)],
        &[(2, 2), (0, 2)],
    );
}

/// MP 1x1x3 is the Gamma-centred 1x1x3 set in another order (odd axis,
/// numerators 2m + 1 - 3 = -2, 0, 2): the k force and stress equal the
/// Gamma-centred ones of the same energy, and `EcpMeshCountModuli` is an
/// identity there (moduli == counts). ARTIFACT: an MP path that shifted an
/// odd axis by half a step (a different energy) or reordered the per-k
/// weights against the k list would miss by O(1).
/// MEASURED: energy 5.3e-15, force 1.9e-13, stress 1.1e-12;
/// EcpMeshCountModuli exactly 0; EcpWrongResidue 8.2e-1, ConjD 5.4e-1.
#[test]
#[ignore = "validation: k-point force/stress #421 coverage (MP 1x1x3 == Gamma-centred 1x1x3, ECP, rotation)"]
fn orthorhombic_ecp_mp_1x1x3_is_the_gamma_centred_1x1x3() {
    let bs = rot_basis();
    let fx = HI_ORTHORHOMBIC;
    let mp = krun(&fx, fx.cell(&bs), &bs, mp_mesh([1, 1, 3]));
    let gc = krun(&fx, fx.cell(&bs), &bs, gamma_mesh([1, 1, 3]));
    assert_eq!(mp.mesh.residue_moduli(), [1, 1, 3]);
    let (fm, sm) = (mp.force(None), mp.stress(None));
    let (fg, sg) = (gc.force(None), gc.stress(None));
    let mut led = Ledger::default();
    let tag = "HI MP 1x1x3 vs Gamma-centred 1x1x3";
    led.below(
        &format!("{tag} energy"),
        (mp.scf.energy() - gc.scf.energy()).abs(),
        F_IDENTITY_BAR,
    );
    led.below(
        &format!("{tag} force"),
        grad_miss(&fm.grad, &fg.grad),
        F_IDENTITY_BAR,
    );
    led.below(
        &format!("{tag} stress"),
        max_mat(&sm.de_deps, &sg.de_deps),
        F_IDENTITY_BAR,
    );
    led.above(
        &format!("{tag} |F_ecp| (vacuity)"),
        amax(&fm.parts.ecp),
        1e-3,
    );
    led.below(
        &format!("{tag} EcpMeshCountModuli identity (force)"),
        grad_miss(
            &mp.force(Some(KGradMutation::EcpMeshCountModuli)).grad,
            &fm.grad,
        ),
        1e-12,
    );
    for m in [KStressMutation::EcpWrongResidue, KStressMutation::ConjD] {
        led.above(
            &format!("{tag} stress mutant {m:?}"),
            max_mat(&mp.stress(Some(m)).de_deps, &sg.de_deps),
            MUTANT_BAR,
        );
    }
    led.finish();
}

// ============================================================ (H) PySCF force pins

/// pbc_grad.rs GRAD_H2_ATOMS (Bohr), cubic edge 4.
const PY_H2_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];

// PINS: PySCF 2.13.1 KRHF/KUHF + FFTDF (mesh 41^3, cell.precision 1e-12,
// cart, conv_tol 1e-12) and its `pbc.grad.krhf` / `pbc.grad.kuhf` gradient
// with the ONE gate lifted that makes it refuse an all-electron cell (see
// `scripts/gen_pyscf_kforce_pins.py`). The patched gradient is checked
// there against central FD of PySCF's OWN FFTDF energy (h = 1e-4):
// 7.2e-11..7.5e-11 on (0, x) and 2.8e-9..3.0e-9 on (1, z) (the h^2
// truncation); FFTDF at 41^3 equals AFTDF to <= 7.3e-15 and 61^3 to 1e-15.

const PY_PINS: [PyPin; 4] = [
    PyPin {
        name: "H2 RHF 1x1x3 none",
        sp: false,
        spin: None,
        n: [1, 1, 3],
        centring: MeshCentring::Gamma,
        exx: ExxDiv::None,
        e: -1.1709293639731704,
        grad: [
            [
                0.00483784364257232,
                -0.007742194034929494,
                0.028584901266458296,
            ],
            [
                -0.0048378436425729875,
                0.007742194034929595,
                -0.028584901266458573,
            ],
        ],
    },
    PyPin {
        name: "H2 RHF 1x1x3 ewald",
        sp: false,
        spin: None,
        n: [1, 1, 3],
        centring: MeshCentring::Gamma,
        exx: ExxDiv::Ewald,
        e: -1.3605974175476423,
        grad: [
            [
                0.004837843642323831,
                -0.007742194034531601,
                0.02858490125225438,
            ],
            [
                -0.0048378436423242525,
                0.007742194034531642,
                -0.02858490125225449,
            ],
        ],
    },
    PyPin {
        name: "H2 RHF MP 1x1x2 none",
        sp: false,
        spin: None,
        n: [1, 1, 2],
        centring: MeshCentring::MonkhorstPack,
        exx: ExxDiv::None,
        e: -0.9749894265850214,
        grad: [
            [
                0.0053696856008595886,
                -0.008592902316328529,
                0.09164699252454006,
            ],
            [
                -0.005369685600860119,
                0.008592902316328581,
                -0.09164699252453945,
            ],
        ],
    },
    // KUHF with the per-k aufbau pinned (ferric's k-UHF occupies n_alpha /
    // n_beta bands at every k; PySCF's KUHF aufbau is global over k).
    PyPin {
        name: "H2 triplet s+p UHF 1x1x3 none",
        sp: true,
        spin: Some((2, 0)),
        n: [1, 1, 3],
        centring: MeshCentring::Gamma,
        exx: ExxDiv::None,
        e: -0.4899757531604011,
        grad: [
            [
                0.02129965187457002,
                -0.03402729760740395,
                0.4843092089674952,
            ],
            [
                -0.02129965187457421,
                0.03402729760740455,
                -0.4843092089674948,
            ],
        ],
    },
];
/// Energy and force bars vs PySCF: see the test doc for the measured values.
const PY_E_BAR: f64 = 1e-11;
const PY_F_BAR: f64 = 3e-10;

struct PyPin {
    name: &'static str,
    sp: bool,
    spin: Option<(usize, usize)>,
    n: [usize; 3],
    centring: MeshCentring,
    exx: ExxDiv,
    e: f64,
    grad: [[f64; 3]; 2],
}

/// ferric's dense-AFT k energy and analytic `dE/dR` of a pinned system.
/// Also returns the force under each mutant in `muts` (same SCF).
fn ferric_dense_h2(p: &PyPin, muts: &[KGradMutation]) -> (f64, Array2<f64>, Vec<Array2<f64>>) {
    use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
    let bs = if p.sp {
        common::sp_basis_h()
    } else {
        common::pyscf_sto3g_h()
    };
    let mut mol = common::hydrogens(&PY_H2_ATOMS);
    mol.multiplicity = p.spin.map_or(1, |(a, b)| a - b + 1);
    let cell = Cell::new(mol, common::cubic(4.0)).expect("cell");
    let prep = prep_for(&cell, &bs);
    let mesh = KPointMesh::new(&cell, p.n, p.centring).expect("mesh");
    let hcfg = hcore_cfg(SrColumnRotation::Off);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcfg).expect("hcore(k)");
    let eri = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::None,
        &KDenseAftConfig {
            precision: 1e-12,
            ..Default::default()
        },
    )
    .expect("dense k kernels");
    let vm = match p.exx {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => mesh.madelung(&cell).expect("madelung"),
    };
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk: Box::new(eri.jk_builder_with_madelung(vm)),
    };
    let jk = || KGradJk::Dense(&eri);
    let all: Vec<Option<KGradMutation>> = std::iter::once(None)
        .chain(muts.iter().map(|&m| Some(m)))
        .collect();
    let (e, mut gs) = match p.spin {
        None => {
            let r = solve_krhf_injected(&cell, &mesh, &kscf_cfg(), inj).expect("k-RHF");
            assert!(r.converged);
            let g: Vec<Array2<f64>> = all
                .iter()
                .map(|&m| {
                    kpoint_rhf_gradient(&cell, &prep, &mesh, &hcfg, &hk, jk(), &r, p.exx, &gcfg(m))
                        .expect("k-RHF force")
                        .grad
                })
                .collect();
            (r.energy, g)
        }
        Some((na, nb)) => {
            let r = solve_kuhf_injected(&cell, &mesh, &kscf_cfg(), inj, na, nb).expect("k-UHF");
            assert!(r.converged);
            let g: Vec<Array2<f64>> = all
                .iter()
                .map(|&m| {
                    kpoint_uhf_gradient(&cell, &prep, &mesh, &hcfg, &hk, jk(), &r, p.exx, &gcfg(m))
                        .expect("k-UHF force")
                        .grad
                })
                .collect();
            (r.energy, g)
        }
    };
    let g0 = gs.remove(0);
    (e, g0, gs)
}

/// EXTERNAL-CODE force anchor. PHYSICS: ferric's dense-AFT k-point RHF/UHF
/// energy and analytic force reproduce PySCF's FFTDF k-point HF energy and
/// `pbc.grad` force (all-electron gate lifted, see the pins) on four H2
/// systems: Gamma-centred 1x1x3 (non-TRIM) with exxdiv none and ewald, a
/// shifted MP 1x1x2 mesh, and the s+p H2 triplet k-UHF (empty beta).
/// Independent in EVERYTHING but the Hamiltonian: PySCF integrates h, J/K
/// and their derivatives on the FFT grid (libcint ip-integrals, FFTDF
/// `get_jk_e1`), ferric uses the Ewald-split Gaussian-nucleus h and the
/// pure-AFT J/K. ARTIFACT: a common-mode error of ferric's force AND energy
/// (invisible to every FD anchor, which differentiates ferric's own energy)
/// shows as an energy miss; a force-only error as a force miss. STRESS has
/// no external anchor: PySCF 2.13.1 has no k-point HF stress (its
/// `krks_stress` is FFTDF pure-DFT only, refusing hybrids).
/// MEASURED (energy / force): RHF 1x1x3 none 3.5e-13 / 1.4e-12, ewald
/// 3.5e-13 / 1.8e-11, MP 1x1x2 6.1e-13 / 2.0e-13, UHF triplet 5.2e-13 /
/// 2.5e-11; |F| 2.9e-2..4.8e-1. Mutant misses: NoPhase 4.4e-2..6.2e-2,
/// ConjD 4.6e-4..4.0e-1, WrongQ 9.0e-3..3.1e-1 (identity on MP 1x1x2),
/// NoInvNk 7.7e-2..8.9e-1. PySCF's own patched-gradient-vs-FD check is
/// 7e-11..3e-9 (its FD truncation), so the pins carry PySCF's analytic
/// precision, not the FD's.
#[test]
#[ignore = "validation: k-point force/stress #421 coverage (k-point HF forces vs PySCF pbc.grad pins)"]
fn dense_k_forces_match_pyscf_pbc_grad() {
    let mut led = Ledger::default();
    // Production-wired k mutants: the PySCF pin must see each of them.
    let muts = [
        KGradMutation::NoPhase,
        KGradMutation::ConjD,
        KGradMutation::WrongQ,
        KGradMutation::NoInvNk,
    ];
    for p in PY_PINS.iter() {
        let (e, g, gm) = ferric_dense_h2(p, &muts);
        let miss = |g: &Array2<f64>| {
            let mut w = 0.0_f64;
            for a in 0..2 {
                for x in 0..3 {
                    w = w.max((g[(a, x)] - p.grad[a][x]).abs());
                }
            }
            w
        };
        let w = miss(&g);
        for (m, gx) in muts.iter().zip(&gm) {
            let tag = format!("{} force mutant {m:?} vs PySCF", p.name);
            if p.n == [1, 1, 2] && *m == KGradMutation::WrongQ {
                // On MP 1x1x2 (k = +-b3/4) every momentum transfer q is 0 or
                // b3/2, so q == -q: WrongQ is an identity there (measured
                // 2.0e-13, the anchor floor).
                led.below(&format!("{tag} (identity)"), miss(gx), PY_F_BAR);
            } else {
                led.above(&tag, miss(gx), MUTANT_BAR);
            }
        }
        led.below(
            &format!("{} energy vs PySCF", p.name),
            (e - p.e).abs(),
            PY_E_BAR,
        );
        led.below(&format!("{} force vs PySCF", p.name), w, PY_F_BAR);
        led.above(&format!("{} |force| (vacuity)", p.name), amax(&g), 1e-3);
    }
    led.finish();
}

/// PySCF 2.13.1 Gamma RHF / UHF + AFTDF (exxdiv none, cell.precision 1e-12,
/// conv_tol 1e-11) with PySCF's `lanl2dz` ECPs and the `rot_basis` sets
/// (`scripts/gen_pyscf_kforce_pins.py ecp`).
const PY_HI_GAMMA_E: f64 = -6.279012426132294;
const PY_ASSB_SEPTET_GAMMA_E: f64 = -2.28552762724466;
/// MEASURED: HI 6.42e-6, AsSb septet 3.08e-5 (the gap is not separated: the
/// HI value matches the ~1e-6-level ECP-quadrature gap `pbc_kks_open_ecp.rs`
/// records at another basis; the AsSb one is 5x larger). Bar 10x.
const PY_ECP_E_BAR: f64 = 3e-4;

/// EXTERNAL ENERGY pin of the ECP cells (no external FORCE exists for an
/// ECP cell: PySCF's k-point HF gradient takes the local part of a GTH
/// pseudopotential only, and ferric has no GTH). PHYSICS: ferric's dense-AFT
/// 1x1x1 k-RHF (HI) and k-UHF (AsSb septet, two ECP elements) energies with
/// the rotating basis reproduce PySCF's AFTDF energies. ARTIFACT: this is the
/// one anchor that sees an ENERGY with a wrong per-element ECP assignment
/// (every FD anchor differentiates ferric's own energy, right or wrong).
#[test]
#[ignore = "validation: k-point force/stress #421 coverage (ECP cell energies vs PySCF)"]
fn ecp_cell_dense_energies_match_pyscf() {
    use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
    let bs = rot_basis();
    let mut led = Ledger::default();
    for (fx, e_ref) in [
        (HI_ORTHORHOMBIC, PY_HI_GAMMA_E),
        (ASSB_SEPTET, PY_ASSB_SEPTET_GAMMA_E),
    ] {
        let cell = fx.cell(&bs);
        let prep = prep_for(&cell, &bs);
        let mesh = KPointMesh::new(&cell, [1, 1, 1], MeshCentring::Gamma).expect("mesh");
        let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg(on())).expect("hcore(k)");
        let eri = KDenseAftEri::build(
            &cell,
            &prep,
            &mesh,
            &hk.s,
            ExxDiv::None,
            &KDenseAftConfig {
                precision: 1e-12,
                ..Default::default()
            },
        )
        .expect("dense k kernels");
        let inj = KPointInjection {
            s: hk.s.clone(),
            h: hk.h.clone(),
            vnn: hk.enn,
            jk: Box::new(eri.jk_builder_with_madelung(0.0)),
        };
        let e = match fx.spin {
            None => {
                let r = solve_krhf_injected(&cell, &mesh, &kscf_cfg(), inj).expect("k-RHF");
                assert!(r.converged, "k-RHF not converged ({} it)", r.iterations);
                r.energy
            }
            Some((na, nb)) => {
                let r = solve_kuhf_injected(&cell, &mesh, &kscf_cfg(), inj, na, nb).expect("k-UHF");
                assert!(r.converged, "k-UHF not converged ({} it)", r.iterations);
                r.energy
            }
        };
        // The RS-GDF energy the D-F anchors differentiate, against the exact
        // (dense) one: its fit error with these small aux sets (recorded;
        // the guard only catches a fit that has stopped fitting).
        let fit = krun(&fx, fx.cell(&bs), &bs, gamma_mesh([1, 1, 1]));
        led.below(
            &format!("{} RS-GDF - dense energy (fit error, recorded)", fx.name),
            (fit.scf.energy() - e).abs(),
            0.1,
        );
        led.below(
            &format!("{} dense energy vs PySCF", fx.name),
            (e - e_ref).abs(),
            PY_ECP_E_BAR,
        );
    }
    led.finish();
}
