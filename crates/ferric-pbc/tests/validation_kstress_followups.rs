// LANL2DZ exponents such as 0.318 are basis data, not approximations of 1/pi.
#![allow(clippy::approx_constant)]
//! k-point force and stress anchors for the follow-ups of issue #421 that
//! `pbc_kstress.rs`, `pbc_kpoint_ecp_rotation.rs` and
//! `pbc_sr_rotation_kderiv.rs` do not cover. Every test here is heavy and
//! `#[ignore = "validation: ..."]`d (weekly validation job).
//!
//! (A) OPEN SHELL WITH p SHELLS ON THE FITTED (RS-GDF) PATH. The RS-GDF UHF
//!     anchors in `pbc_kstress.rs` are s-only H3 doublets; the s+p H2 triplet
//!     (2 alpha, 0 beta) and H4 triplet (3 alpha, 1 beta) were dense-only.
//!     Anchors: 1x1x1 vs the Gamma RS-GDF UHF force and stress (term by term);
//!     1x1x3 vs the explicit 3-cell supercell / N (H2 AND H4: the H4 one is
//!     the "H4 supercell anchor" of #421, on the fitted path); central FD of
//!     the k-UHF RS-GDF energy (H2 triplet, all nine strains, h and 2h).
//! (B) H4 TRIPLET SUPERCELL ON THE DENSE PATH: 1x1x3 k-UHF stress vs the
//!     12-atom supercell Gamma-UHF stress / 3.
//! (C) ECP + ROTATING BASIS ON THE FITTED PATH (HI cell of
//!     `pbc_kpoint_ecp_rotation.rs`, LANL2DZ ECP on I, rotation on):
//!     (C1) RANGE-SPLIT RS-GDF with the ECP: split vs unsplit k force and
//!          stress, FD of the split energy, split mutants;
//!     (C2) the fit-acting FORCE mutants (and the k, ECP and rotation force
//!          mutants) against an FD-anchored RS-GDF force;
//!     (C3) a SHIFTED (Monkhorst-Pack 1x1x2) mesh with the ECP and the
//!          rotation: FD of force and stress, rotated vs unrotated, and the
//!          ECP residue mutants that are identities on every Gamma-centred
//!          mesh (`EcpMeshCountModuli`) or on 1x1x2 Gamma (`EcpWrongResidue`).
//!
//! ARTIFACT hypotheses (stated before measuring):
//! (A) a fitted exchange that pairs the spins by accident of N_a - N_b = 1
//!     (H3) would show as an O(1) miss when D_beta = 0 (H2 triplet);
//!     `FitCrossSpin` is that defect wired into production and must miss
//!     every anchor. A p-shell fit-walk error (aux-centre or AO-pair vector)
//!     would miss the supercell, which the s-only H3 cannot see. If correct:
//!     1x1x1 = Gamma <= ~1e-12, supercell <= ~1e-11, FD at the h^2 floor.
//! (C1) "the split passed because it ran the unsplit code": then split ==
//!     unsplit to 1e-15 and the moved counters are 0; both are asserted
//!     (counters > 0 and a nonzero difference). An ECP weight routed through
//!     the moved (G-space) blocks would break split == unsplit at O(|S_ecp|).
//! (C2) a fit mutant that acts only on the FIT weights would be invisible to
//!     the dense ECP anchors (measured 0 there); against the fitted force it
//!     must move the total by >> the FD bar.
//! (C3) an ECP image weight read at the wrong residue on a shifted mesh
//!     (residue moduli 2N along z) aliases images half a supercell apart:
//!     invisible on Gamma-centred meshes, an O(|F_ecp|) miss here.
//!
//! MEASURED ERRORS AND MUTANT LEDGER: see the constants and the per-test
//! docs (filled from the runs; release build, RAYON 4, loaded box). One
//! pre-stated prediction failed: the (A) supercell anchor was predicted at
//! <= ~1e-11 and the H4 triplet measured 8.2e-11 (~8x over); its bar (1e-9)
//! was set AFTER that measurement (see `OS_FIT_SC_BAR`).
//!
//! SHARED CODE (what the RS-GDF anchors cannot see): the k-point fitted
//! derivative (`rsgdf/kpoint/kderiv.rs`) imports the Gamma SR walks
//! (`sr2_force`, `sr3_force`, `sr2_strain_with`, `sr3_strain_with`) and
//! `aux_ft_strain_shells`, so 1x1x1 = Gamma and k = supercell/N check only
//! the LR part and the phase / residue binning independently. The shared SR
//! derivative kernel is anchored by FD on the H2 triplet (both exxdiv) and
//! on the H4 triplet (exxdiv none), section A.

mod common;

use common::{cubic, hydrogens, prep_for, sp_basis_h};
use ferric_core::basis::{self, BasisSet, Shell};
use ferric_core::ecp::{EcpDef, EcpShell, EcpTerm};
use ferric_core::mol::{Atom, Molecule};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::grad::{gamma_uhf_gradient_rsgdf, GammaGradConfig, RsGdfGradSource};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kgrad::{
    kpoint_rhf_gradient, kpoint_uhf_gradient, KGradConfig, KGradJk, KGradMutation, KGradient,
    KRsGdfGradSource,
};
use ferric_pbc::kpts::{KPointMesh, MeshCentring};
use ferric_pbc::kscf::{solve_krhf_injected, KPointInjection, KPointJk, KScfConfig, KScfResult};
use ferric_pbc::kstress::{
    kpoint_rhf_stress, kpoint_uhf_stress, KStress, KStressConfig, KStressJk, KStressMutation,
};
use ferric_pbc::kuscf::{solve_kuhf_injected, KUScfResult};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::{RangeSplit, RsGdf, RsGdfConfig};
use ferric_pbc::sr_rotation::SrColumnRotation;
use ferric_pbc::stress::{
    gamma_uhf_stress, gamma_uhf_stress_rsgdf, GammaStress, GammaStressConfig, Mat3,
};
use ferric_scf::result::{ScfExit, ScfResult, Spin};
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const HCORE_PRECISION: f64 = 1e-14;
const ANCHOR_PRECISION: f64 = 1e-12;
const AMPLE: usize = 1 << 31;
const FD_H: f64 = 1e-4;
const ZERO: Mat3 = [[0.0; 3]; 3];
const EXX: [ExxDiv; 2] = [ExxDiv::None, ExxDiv::Ewald];
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
/// Smallest miss a production-wired mutant must show against an anchor.
const MUTANT_BAR: f64 = 1e-4;

// ------------------------------------------------------------ shared helpers

fn kscf_cfg() -> KScfConfig {
    KScfConfig {
        energy_conv: 1e-13,
        grad_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    }
}

fn eps_at(i: usize, j: usize, h: f64) -> Mat3 {
    let mut e = ZERO;
    e[i][j] = h;
    e
}

fn strained(cell: &Cell, i: usize, j: usize, h: f64) -> Cell {
    cell.strained(&eps_at(i, j, h)).expect("strained cell")
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
/// outer, as `supercell`).
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

fn gamma_unrestricted(da: Array2<f64>, db: Array2<f64>) -> ScfResult {
    let n = da.nrows();
    ScfResult {
        spin: Spin::Unrestricted,
        energy: 0.0,
        density_total: &da + &db,
        density_alpha: da,
        density_beta: Some(db),
        mos_alpha: Array2::eye(n),
        mos_beta: Some(Array2::eye(n)),
        eps_alpha: vec![0.0; n],
        eps_beta: Some(vec![0.0; n]),
        fock_alpha: Array2::zeros((n, n)),
        fock_beta: Some(Array2::zeros((n, n))),
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

/// Collects (name, value, bar, must-exceed?) checks so one heavy run reports
/// every number before it fails.
#[derive(Default)]
struct Ledger {
    rows: Vec<String>,
    failures: Vec<String>,
}

impl Ledger {
    /// `value` must be BELOW `bar` (an anchor).
    fn below(&mut self, name: &str, value: f64, bar: f64) {
        let ok = value.is_finite() && value < bar;
        self.push(name, value, "<", bar, ok);
    }
    /// `value` must be ABOVE `bar` (a mutant).
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
        self.rows.push(row.clone());
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

// ============================================================ (A)/(B) H open shell

const H2_ATOMS: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
const OS_A: f64 = 4.0;
const H4_ATOMS: [[f64; 3]; 4] = [
    [0.3, 0.2, 0.1],
    [0.35, 0.12, 1.5],
    [1.6, 0.9, 0.7],
    [1.2, 2.1, 2.3],
];
const H4_A: f64 = 4.5;

/// Rotation pinned OFF (the rotated walks are anchored in section C and in
/// `pbc_sr_rotation_kderiv.rs`).
fn hcore_cfg_off() -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
    .with_sr_column_rotation(SrColumnRotation::Off)
}

fn cell_at(pos: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

/// Explicit diag(n) supercell (copy c = (m0, m1, m2), m0 outer).
fn supercell(cell: &Cell, n: [usize; 3], mult: usize) -> Cell {
    let a = *cell.lattice();
    let mut atoms = Vec::new();
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                let t: Vec<f64> = (0..3)
                    .map(|d| m0 as f64 * a[0][d] + m1 as f64 * a[1][d] + m2 as f64 * a[2][d])
                    .collect();
                for p in cell.positions() {
                    atoms.push([p[0] + t[0], p[1] + t[1], p[2] + t[2]]);
                }
            }
        }
    }
    let mut lat = a;
    for (i, row) in lat.iter_mut().enumerate() {
        for v in row.iter_mut() {
            *v *= n[i] as f64;
        }
    }
    cell_at(&atoms, lat, mult)
}

struct OpenShell {
    name: &'static str,
    cell: Cell,
    na: usize,
    nb: usize,
}

fn h2_triplet() -> OpenShell {
    OpenShell {
        name: "H2 triplet s+p",
        cell: cell_at(&H2_ATOMS, cubic(OS_A), 3),
        na: 2,
        nb: 0,
    }
}

fn h4_triplet() -> OpenShell {
    OpenShell {
        name: "H4 triplet s+p",
        cell: cell_at(&H4_ATOMS, cubic(H4_A), 3),
        na: 3,
        nb: 1,
    }
}

/// H3 s+p QUARTET: 3 alpha / 0 beta (empty beta set), the highest
/// multiplicity three H atoms allow (#421 "quartet or other multiplicities").
const H3_ATOMS: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [1.6, 0.9, 0.7]];
const H3_A: f64 = 4.5;

fn h3_quartet() -> OpenShell {
    OpenShell {
        name: "H3 quartet s+p",
        cell: cell_at(&H3_ATOMS, cubic(H3_A), 4),
        na: 3,
        nb: 0,
    }
}

struct HSys {
    cell: Cell,
    prep: PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
}

fn hsys(cell: Cell, bs: &BasisSet, n: [usize; 3]) -> HSys {
    let prep = prep_for(&cell, bs);
    let mesh = KPointMesh::new(&cell, n, MeshCentring::Gamma).expect("mesh");
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcore_cfg_off()).expect("hcore(k)");
    HSys {
        cell,
        prep,
        mesh,
        hk,
    }
}

fn kuhf(ks: &HSys, jk: Box<dyn KPointJk + '_>, na: usize, nb: usize) -> KUScfResult {
    let inj = KPointInjection {
        s: ks.hk.s.clone(),
        h: ks.hk.h.clone(),
        vnn: ks.hk.enn,
        jk,
    };
    let r = solve_kuhf_injected(&ks.cell, &ks.mesh, &kscf_cfg(), inj, na, nb).expect("k-UHF");
    assert!(r.converged, "k-UHF not converged ({} it)", r.iterations);
    r
}

/// The spin sectors are populated unequally (and an empty beta really is
/// empty), so a spin-restricted-looking solution cannot make an anchor vacuous.
fn assert_spin_polarised(sys: &OpenShell, scf: &KUScfResult) {
    let mag: f64 = scf
        .density_alpha
        .iter()
        .zip(&scf.density_beta)
        .map(|(a, b)| (a - b).iter().map(|z| z.norm()).sum::<f64>())
        .sum();
    let beta: f64 = scf
        .density_beta
        .iter()
        .map(|b| b.iter().map(|z| z.norm()).sum::<f64>())
        .sum();
    eprintln!(
        "  {}: spin ({}, {}), sum|D_a - D_b| {mag:.3}, sum|D_b| {beta:.3e}, E {:.10}",
        sys.name, scf.nalpha, scf.nbeta, scf.energy
    );
    assert!(mag > 0.5, "{}: not spin polarised", sys.name);
    assert_eq!(beta < 1e-12, sys.nb == 0, "{}: beta sector", sys.name);
}

/// ET-sp aux on H: s(4.7, 1.9, 0.75, 0.3) + p(1.25, 0.5), one unit-normalised
/// primitive per shell (`pbc_kstress.rs`).
fn et_sp_aux() -> BasisSet {
    let mut shells = Vec::new();
    for (l, a) in [(0, 4.7), (0, 1.9), (0, 0.75), (0, 0.3), (1, 1.25), (1, 0.5)] {
        shells.push(Shell {
            l,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    let mut m = HashMap::new();
    m.insert(1, shells);
    BasisSet {
        name: "et-sp-aux-H".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

fn h_gdf_cfg() -> RsGdfConfig {
    RsGdfConfig {
        omega: 1.0,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        sr_column_rotation: SrColumnRotation::Off,
        ..Default::default()
    }
}

struct HFit {
    cfg: KRsGdfConfig,
    aux: PreparedBasis,
    gdf: KRsGdf,
}

fn hfit(ks: &HSys) -> HFit {
    let cfg = KRsGdfConfig {
        gdf: h_gdf_cfg(),
        mutation: None,
    };
    let aux = PreparedBasis::new(ks.cell.mol(), &et_sp_aux()).expect("aux");
    let gdf = KRsGdf::build(&ks.cell, &ks.prep, &aux, &ks.mesh, &ks.hk.s, &cfg).expect("KRsGdf");
    HFit { cfg, aux, gdf }
}

impl HFit {
    fn src(&self) -> KRsGdfGradSource<'_> {
        KRsGdfGradSource {
            gdf: &self.gdf,
            cfg: &self.cfg,
            aux: &self.aux,
        }
    }
    fn jk(&self, ks: &HSys, exx: ExxDiv) -> Box<dyn KPointJk + '_> {
        let vm = match exx {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => ks.mesh.madelung(&ks.cell).unwrap(),
        };
        Box::new(self.gdf.jk_builder_with_madelung(vm))
    }
    fn kept(&self) -> Vec<usize> {
        self.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect()
    }
}

fn exx_idx(e: ExxDiv) -> usize {
    match e {
        ExxDiv::None => 0,
        ExxDiv::Ewald => 1,
    }
}

fn fit_uscfs(ks: &HSys, fs: &HFit, na: usize, nb: usize) -> [KUScfResult; 2] {
    [
        kuhf(ks, fs.jk(ks, ExxDiv::None), na, nb),
        kuhf(ks, fs.jk(ks, ExxDiv::Ewald), na, nb),
    ]
}

fn k_uhf_fit_stress(
    ks: &HSys,
    fs: &HFit,
    scf: &KUScfResult,
    exx: ExxDiv,
    m: Option<KStressMutation>,
) -> KStress {
    kpoint_uhf_stress(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg_off(),
        &ks.hk,
        KStressJk::RsGdf(fs.src()),
        scf,
        exx,
        &scfg(m),
    )
    .expect("kpoint_uhf_stress (RS-GDF)")
}

fn k_uhf_fit_force(
    ks: &HSys,
    fs: &HFit,
    scf: &KUScfResult,
    exx: ExxDiv,
    m: Option<KGradMutation>,
) -> KGradient {
    kpoint_uhf_gradient(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &hcore_cfg_off(),
        &ks.hk,
        KGradJk::RsGdf(fs.src()),
        scf,
        exx,
        &gcfg(m),
    )
    .expect("kpoint_uhf_gradient (RS-GDF)")
}

/// Gamma RS-GDF setup of a cell (the explicit supercell or the cell itself).
struct GFit {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
    aux: PreparedBasis,
    gdf: RsGdf,
}

fn gfit(cell: Cell, bs: &BasisSet) -> GFit {
    let prep = prep_for(&cell, bs);
    let hc = periodic_hcore(&cell, &prep, &hcore_cfg_off()).expect("gamma hcore");
    let aux = PreparedBasis::new(cell.mol(), &et_sp_aux()).expect("aux");
    let gdf = RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, &h_gdf_cfg()).expect("RsGdf");
    GFit {
        cell,
        prep,
        hc,
        aux,
        gdf,
    }
}

impl GFit {
    fn src(&self) -> RsGdfGradSource<'_> {
        RsGdfGradSource {
            gdf: &self.gdf,
            aux: &self.aux,
            aux_jac: None,
        }
    }
    fn stress(&self, scf: &ScfResult, exx: ExxDiv) -> GammaStress {
        gamma_uhf_stress_rsgdf(
            &self.cell,
            &self.prep,
            &hcore_cfg_off(),
            &self.hc,
            &self.src(),
            scf,
            exx,
            &GammaStressConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
        )
        .expect("gamma uhf rsgdf stress")
    }
    fn force(&self, scf: &ScfResult, exx: ExxDiv) -> Array2<f64> {
        gamma_uhf_gradient_rsgdf(
            &self.cell,
            &self.prep,
            &hcore_cfg_off(),
            &self.hc,
            &self.src(),
            scf,
            exx,
            &GammaGradConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
        )
        .expect("gamma uhf rsgdf gradient")
        .grad
    }
}

fn fit_parts_k(p: &ferric_pbc::kstress::KStressParts) -> [(&'static str, Mat3); 7] {
    [
        ("J3 SR", p.fit_j3_sr),
        ("J3 LR", p.fit_j3_lr),
        ("J2 SR", p.fit_j2_sr),
        ("J2 LR", p.fit_j2_lr),
        ("G0 overlap", p.fit_g0_overlap),
        ("J3 G0 volume", p.fit_j3_g0_volume),
        ("J2 G0 volume", p.fit_j2_g0_volume),
    ]
}

fn fit_parts_g(p: &ferric_pbc::stress::GammaStressParts) -> [Mat3; 7] {
    [
        p.fit_j3_sr,
        p.fit_j3_lr,
        p.fit_j2_sr,
        p.fit_j2_lr,
        p.fit_g0_overlap,
        p.fit_j3_g0_volume,
        p.fit_j2_g0_volume,
    ]
}

/// Bars of the fitted open-shell s+p anchors, from the measured worst case
/// of both systems and both exxdiv (release, RAYON 4):
///
/// | anchor               | H2 triplet          | H4 triplet          | bar   |
/// |----------------------|---------------------|---------------------|-------|
/// | 1x1x1 stress total   | 1.9e-13             | 8.8e-13             | 1e-11 |
/// | 1x1x1 force total    | 1.1e-13             | 3.1e-13             | 1e-11 |
/// | 1x1x1 worst part     | 2.8e-12 (J2 G0 vol) | 9.9e-12 (J2 SR)     | 1e-10 |
/// | 1x1x3 stress total   | 5.0e-12             | 8.2e-11             | 1e-9  |
/// | 1x1x3 force total    | 1.6e-12             | 3.0e-11             | 1e-9  |
/// | 1x1x3 worst part     | 5.4e-12 (J2 SR)     | 4.3e-11 (J2 LR)     | 1e-9  |
///
/// H3 QUARTET (3 alpha / 0 beta, 1x1x1 only): stress total 1.12e-12 (bar
/// 1e-11, 9x: the tightest margin in this table), force 4.23e-13, worst
/// part 9.8e-13 (J2 G0 volume, ewald); FitCrossSpin 2.73e-1, SwapSpinFock
/// 2.59e-1.
///
/// The per-part 1x1x1 misses cancel between J2 SR and the J2 G0 volume in
/// the total (the same cancellation `pbc_kstress.rs` measured for H2 RHF:
/// worst part 1.1e-12, total 6.5e-14). The H4 supercell floor is 16x the H2
/// one; it sits in the J2/J3 parts (the fitted metric), which points to the
/// conditioning of the larger 4-H aux metric rather than a k defect, but
/// that attribution is NOT tested. Smallest mutant miss against these
/// anchors: 3.1e-2 (stress, FitAuxPhase on H4) and 8.7e-3 (force,
/// FitAuxPhase on H4), >= 87x MUTANT_BAR (1e-4).
///
/// Mutant ledger (miss vs the anchor, exxdiv none):
/// 1x1x1 vs Gamma, stress: FitCrossSpin 5.33e-1 (H2) / 1.98e-1 (H4),
/// SwapSpinFock 6.47e-1 / 1.62e-1.
/// 1x1x3 vs supercell/3, stress (H2 / H4): NoPhase 3.10e-1 / 2.01e-1, ConjD
/// 5.61e-1 / 3.50e-1, NoInvNk 1.10 / 3.14, FitAuxPhase 6.03e-2 / 3.10e-2,
/// FitLrWrongQ 2.45e-1 / 1.69e-1, FitCrossSpin 1.48 / 6.26e-1, SwapSpinFock
/// 1.04 / 2.91e-1; force (H2 / H4): NoPhase 7.81e-2 / 4.28e-2, ConjD 4.04e-1
/// / 1.25e-1, NoInvNk 7.21e-2 / 1.15, FitAuxPhase 3.75e-2 / 8.72e-3,
/// FitNoMetric 2.78e-1 / 2.19e-1, FitNoG0 2.49e-2 / 2.11e-2.
const OS_FIT_GAMMA_BAR: f64 = 1e-11;
const OS_FIT_GAMMA_PART_BAR: f64 = 1e-10;
const OS_FIT_SC_BAR: f64 = 1e-9;

/// PHYSICS: at 1x1x1 the k-UHF RS-GDF force and stress are the Gamma-UHF
/// RS-GDF ones, total and per fitted part, both exxdiv, for the s+p H2
/// triplet (empty beta) and the H4 triplet (both sectors populated).
/// ARTIFACT: blind to phases, 1/N_k and q != 0 (the supercell anchors carry
/// those); what this adds is the spin-unbalanced s+p density through the
/// fitted J/K (independent in the LR part and the binning: half-sphere real
/// LR / one Y vs full-sphere complex per-q LR / per-(l, t) bins; the SR
/// derivative walks are SHARED with the Gamma code, see the module doc, and
/// are anchored by the H2 and H4 triplet FD). The spin mutants
/// `FitCrossSpin` and `SwapSpinFock` are NOT k-specific and must miss here.
#[test]
#[ignore = "validation: k-point force/stress #421 follow-ups (RS-GDF open shell s+p, 1x1x1 vs Gamma)"]
fn rsgdf_open_shell_sp_one_point_mesh_is_the_gamma_one() {
    let bs = sp_basis_h();
    let mut led = Ledger::default();
    for sys in [h2_triplet(), h4_triplet(), h3_quartet()] {
        let ks = hsys(sys.cell.clone(), &bs, [1, 1, 1]);
        let fs = hfit(&ks);
        let scfs = fit_uscfs(&ks, &fs, sys.na, sys.nb);
        assert_spin_polarised(&sys, &scfs[0]);
        let g = gfit(sys.cell.clone(), &bs);
        for exx in EXX {
            let scf = &scfs[exx_idx(exx)];
            let gscf = gamma_unrestricted(
                real_of(&scf.density_alpha[0]),
                real_of(&scf.density_beta[0]),
            );
            let k = k_uhf_fit_stress(&ks, &fs, scf, exx, None);
            let gs = g.stress(&gscf, exx);
            let kf = k_uhf_fit_force(&ks, &fs, scf, exx, None);
            let gf = g.force(&gscf, exx);
            let tag = format!("{} 1x1x1 {exx:?}", sys.name);
            for ((pn, kp), gp) in fit_parts_k(&k.parts).iter().zip(fit_parts_g(&gs.parts)) {
                led.below(
                    &format!("{tag} stress {pn}"),
                    max_mat(kp, &gp),
                    OS_FIT_GAMMA_PART_BAR,
                );
            }
            led.below(
                &format!("{tag} stress TOTAL"),
                max_mat(&k.de_deps, &gs.de_deps),
                OS_FIT_GAMMA_BAR,
            );
            led.below(
                &format!("{tag} force TOTAL"),
                grad_miss(&kf.grad, &gf),
                OS_FIT_GAMMA_BAR,
            );
            led.above(
                &format!("{tag} |stress| (vacuity)"),
                max_mat(&k.de_deps, &ZERO),
                1e-3,
            );
            if exx == ExxDiv::None {
                for m in [KStressMutation::FitCrossSpin, KStressMutation::SwapSpinFock] {
                    let x = k_uhf_fit_stress(&ks, &fs, scf, exx, Some(m));
                    led.above(
                        &format!("{tag} stress mutant {m:?}"),
                        max_mat(&x.de_deps, &gs.de_deps),
                        MUTANT_BAR,
                    );
                }
            }
        }
    }
    led.finish();
}

/// PHYSICS: the Gamma-centred 1x1x3 (non-TRIM) k-UHF RS-GDF force and stress
/// per cell equal the explicit 3-cell supercell's Gamma-UHF RS-GDF force (on
/// every copy) and stress / 3, both exxdiv, per fitted part, with the k-,
/// fit- and spin-specific mutants missing. ARTIFACT: independent in the
/// supercell geometry, the LR part and the phase / residue binning; the SR
/// derivative walks are the SAME Gamma functions on both sides (module
/// doc), so an SR-kernel defect is not visible here (the H2 and H4 triplet
/// FD anchors cover it). A wrong phase, q
/// sign, conjugated density or a p-shell aux-image vector would not
/// reproduce it on the non-TRIM mesh.
fn open_shell_fit_supercell(sys: OpenShell, led: &mut Ledger) {
    let bs = sp_basis_h();
    let n = [1, 1, 3];
    let ks = hsys(sys.cell.clone(), &bs, n);
    let fs = hfit(&ks);
    let scfs = fit_uscfs(&ks, &fs, sys.na, sys.nb);
    assert_spin_polarised(&sys, &scfs[0]);
    // 3 cells x 2 unpaired electrons: multiplicity 7 (H2 and H4 triplets).
    let g = gfit(supercell(&sys.cell, n, 7), &bs);
    let mut refs = [ZERO; 2];
    let mut fref = Vec::new();
    for exx in EXX {
        let i = exx_idx(exx);
        let gscf = gamma_unrestricted(
            unfold(&ks.mesh, &scfs[i].density_alpha),
            unfold(&ks.mesh, &scfs[i].density_beta),
        );
        let k = k_uhf_fit_stress(&ks, &fs, &scfs[i], exx, None);
        let gs = g.stress(&gscf, exx);
        let kf = k_uhf_fit_force(&ks, &fs, &scfs[i], exx, None);
        let gf = g.force(&gscf, exx);
        refs[i] = scale3(&gs.de_deps, 1.0 / 3.0);
        let tag = format!("{} 1x1x3 {exx:?}", sys.name);
        for ((pn, kp), gp) in fit_parts_k(&k.parts).iter().zip(fit_parts_g(&gs.parts)) {
            led.below(
                &format!("{tag} stress {pn} vs sc/3"),
                max_mat(kp, &scale3(&gp, 1.0 / 3.0)),
                OS_FIT_SC_BAR,
            );
        }
        led.below(
            &format!("{tag} stress TOTAL vs sc/3"),
            max_mat(&k.de_deps, &refs[i]),
            OS_FIT_SC_BAR,
        );
        led.below(
            &format!("{tag} force TOTAL vs sc copies"),
            sc_miss(&kf.grad, &gf),
            OS_FIT_SC_BAR,
        );
        led.above(&format!("{tag} |force| (vacuity)"), amax(&kf.grad), 1e-3);
        fref.push(gf);
    }
    let tag = format!("{} 1x1x3 None", sys.name);
    for m in [
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
        KStressMutation::NoInvNk,
        KStressMutation::FitAuxPhase,
        KStressMutation::FitLrWrongQ,
        KStressMutation::FitCrossSpin,
        KStressMutation::SwapSpinFock,
    ] {
        let x = k_uhf_fit_stress(&ks, &fs, &scfs[0], ExxDiv::None, Some(m));
        led.above(
            &format!("{tag} stress mutant {m:?}"),
            max_mat(&x.de_deps, &refs[0]),
            MUTANT_BAR,
        );
    }
    for m in [
        KGradMutation::NoPhase,
        KGradMutation::ConjD,
        KGradMutation::NoInvNk,
        KGradMutation::FitAuxPhase,
        KGradMutation::FitNoMetric,
        KGradMutation::FitNoG0,
    ] {
        let x = k_uhf_fit_force(&ks, &fs, &scfs[0], ExxDiv::None, Some(m));
        led.above(
            &format!("{tag} force mutant {m:?}"),
            sc_miss(&x.grad, &fref[0]),
            MUTANT_BAR,
        );
    }
}

#[test]
#[ignore = "validation: k-point force/stress #421 follow-ups (RS-GDF H2 triplet s+p, 1x1x3 vs supercell)"]
fn rsgdf_open_shell_h2_triplet_sp_1x1x3_is_the_supercell_per_cell() {
    let mut led = Ledger::default();
    open_shell_fit_supercell(h2_triplet(), &mut led);
    led.finish();
}

#[test]
#[ignore = "validation: k-point force/stress #421 follow-ups (RS-GDF H4 triplet s+p, 1x1x3 vs 12-atom supercell)"]
fn rsgdf_open_shell_h4_triplet_sp_1x1x3_is_the_supercell_per_cell() {
    let mut led = Ledger::default();
    open_shell_fit_supercell(h4_triplet(), &mut led);
    led.finish();
}

/// FD bars of the fitted open-shell s+p energy (h = 1e-4). MEASURED (both
/// exxdiv, release, 111 s): stress all nine 2.53e-8 (2h: 1.02e-7, ratio
/// 4.04, the h^2 truncation), force (H1 x, H2 z) 2.52e-9; bars 12x. Mutants
/// vs FD (none / ewald): FitCrossSpin 1.48 / 1.48, SwapSpinFock 1.04 / 1.34,
/// and the fit-error-sized `NoAuxFt` 3.38e-5 / 3.38e-5 (asserted > 1e-5, 33x
/// the stress bar; the smallest mutant miss in this file, visible to the FD
/// only: the supercell anchor cannot see it, both sides carry the same fit).
const OS_FIT_FD_S_BAR: f64 = 3e-7;
const OS_FIT_FD_F_BAR: f64 = 3e-8;

/// PHYSICS: all nine strain components and two force components of the
/// k-UHF RS-GDF H2 triplet (s+p) vs central FD of its own energy (fresh fit
/// and SCF per point, both exxdiv, h and 2h: the ratio must be ~4 for an h^2
/// truncation, a piecewise energy would break it). The FD sees what the
/// supercell cannot: the sub-1e-4 `NoAuxFt` fit-error mutant.
#[test]
#[ignore = "validation: k-point force/stress #421 follow-ups (RS-GDF H2 triplet s+p, FD of the k energy)"]
fn rsgdf_open_shell_h2_triplet_sp_1x1x3_matches_fd_of_own_k_energy() {
    open_shell_fit_fd(
        h2_triplet(),
        FdSpec {
            pos: &H2_ATOMS,
            a: OS_A,
            exxs: &EXX,
            strains: &ALL9,
            ratio: &ALL9,
            s_bar: OS_FIT_FD_S_BAR,
            f_bar: OS_FIT_FD_F_BAR,
        },
    );
}

/// FD bars of the H4 triplet fitted energy (h = 1e-4, exxdiv none).
/// MEASURED (release, 272 s): stress all nine 7.17e-8 (zz; FD(2h)/FD(h)
/// 3.78 there, 3.85 on xy, 4.28 on xz: the h^2 truncation), force (H1 x,
/// H2 z) 7.51e-10; bars 14x / 40x. The H4 stress floor is 2.8x the H2
/// one (larger stress, 4 p-carrying centres), so the H2 bar (3e-7, 4x
/// here) is not reused. Mutants vs FD: FitCrossSpin 6.26e-1, SwapSpinFock
/// 2.91e-1, NoAuxFt 1.44e-4 (asserted > 1e-5; 144x the stress bar).
const OS_FIT_FD_H4_S_BAR: f64 = 1e-6;
const OS_FIT_FD_H4_F_BAR: f64 = 3e-8;

/// PHYSICS: the SHARED short-range fitted derivative kernel (`sr2_force`,
/// `sr3_force`, `sr2_strain_with`, `sr3_strain_with`, imported by the k
/// path from the Gamma code, so the 1x1x1 and supercell anchors cannot see
/// it) on the H4 triplet (3 alpha / 1 beta, both spin sectors populated,
/// four p-carrying centres: 6 aux shells each, the 16x larger supercell
/// floor of `OS_FIT_SC_BAR`): all nine strain components and two force
/// components vs central FD of the k-UHF RS-GDF energy (exxdiv none; the
/// SR kernel does not depend on exxdiv, the ewald part is anchored on H2),
/// with the h-halving ratio on zz, xy and xz. ARTIFACT: an SR-kernel slip
/// that cancels on the 2-centre H2 triplet (e.g. an aux-centre / AO-pair
/// vector swap that is symmetric for two atoms) misses here; `NoAuxFt`
/// (the fit-error-sized mutant) must stay visible.
#[test]
#[ignore = "validation: k-point force/stress #421 follow-ups (RS-GDF H4 triplet s+p, FD of the k energy)"]
fn rsgdf_open_shell_h4_triplet_sp_1x1x3_matches_fd_of_own_k_energy() {
    open_shell_fit_fd(
        h4_triplet(),
        FdSpec {
            pos: &H4_ATOMS,
            a: H4_A,
            exxs: &[ExxDiv::None],
            strains: &ALL9,
            ratio: &[(2, 2), (0, 1), (0, 2)],
            s_bar: OS_FIT_FD_H4_S_BAR,
            f_bar: OS_FIT_FD_H4_F_BAR,
        },
    );
}

/// Shared body of the fitted open-shell FD anchors: all nine strains at h,
/// the strains `ratio` also at 2h (FD(2h)/FD(h) must be ~4), forces (atom
/// 0 x, atom 1 z), for every `exxs`.
/// FD bars of the H3 quartet fitted energy (h = 1e-4, exxdiv none).
/// MEASURED (release; this test plus the 1x1x1 anchor 480 s): stress (zz,
/// xx, xy, xz) 4.49e-8 (zz FD(2h)/FD(h) 3.86), force 2.55e-9; bars 22x /
/// 12x. Mutants vs FD: FitCrossSpin 1.30, SwapSpinFock 5.70e-1, NoAuxFt
/// 6.89e-4.
const OS_FIT_FD_Q_S_BAR: f64 = 1e-6;
const OS_FIT_FD_Q_F_BAR: f64 = 3e-8;

/// PHYSICS: the k-UHF RS-GDF stress (zz, xx, xy, xz) and force of the s+p
/// H3 QUARTET (3 alpha, empty beta) on the Gamma-centred 1x1x3 mesh vs
/// central FD of its own energy (exxdiv none; zz also at 2h). The 1x1x1 =
/// Gamma anchor of this system is in
/// `rsgdf_open_shell_sp_one_point_mesh_is_the_gamma_one`. ARTIFACT: a spin
/// slip that only cancels for N_a - N_b <= 2 misses here.
#[test]
#[ignore = "validation: k-point force/stress #421 follow-ups (RS-GDF H3 quartet s+p, FD of the k energy)"]
fn rsgdf_open_shell_h3_quartet_sp_1x1x3_matches_fd_of_own_k_energy() {
    open_shell_fit_fd(
        h3_quartet(),
        FdSpec {
            pos: &H3_ATOMS,
            a: H3_A,
            exxs: &[ExxDiv::None],
            strains: &[(2, 2), (0, 0), (0, 1), (0, 2)],
            ratio: &[(2, 2)],
            s_bar: OS_FIT_FD_Q_S_BAR,
            f_bar: OS_FIT_FD_Q_F_BAR,
        },
    );
}

/// What an open-shell fitted FD anchor differentiates and its bars.
struct FdSpec<'a> {
    /// Cell positions (Bohr) of `sys` (the force FD moves atom 0 x, atom 1 z).
    pos: &'a [[f64; 3]],
    /// Cubic edge (Bohr).
    a: f64,
    exxs: &'a [ExxDiv],
    /// Strain components differentiated at h.
    strains: &'a [(usize, usize)],
    /// Strain components ALSO at 2h (subset of `strains`).
    ratio: &'a [(usize, usize)],
    s_bar: f64,
    f_bar: f64,
}

/// `max |a_ij - b_ij|` over `comps` only.
fn miss_on(a: &Mat3, b: &Mat3, comps: &[(usize, usize)]) -> f64 {
    comps
        .iter()
        .fold(0.0_f64, |w, &(i, j)| w.max((a[i][j] - b[i][j]).abs()))
}

fn open_shell_fit_fd(sys: OpenShell, spec: FdSpec<'_>) {
    let FdSpec {
        pos,
        a,
        exxs,
        strains,
        ratio,
        s_bar,
        f_bar,
    } = spec;
    assert!(ratio.iter().all(|c| strains.contains(c)));
    let bs = sp_basis_h();
    let n = [1, 1, 3];
    let mult = sys.cell.mol().multiplicity;
    let ks = hsys(sys.cell.clone(), &bs, n);
    let fs = hfit(&ks);
    let kept0 = fs.kept();
    let energies = |c: &Cell| -> [f64; 2] {
        let k2 = hsys(c.clone(), &bs, n);
        let f2 = hfit(&k2);
        assert_eq!(f2.kept(), kept0, "the metric cut changed (E piecewise)");
        let mut e = [f64::NAN; 2];
        for &x in exxs {
            e[exx_idx(x)] = kuhf(&k2, f2.jk(&k2, x), sys.na, sys.nb).energy;
        }
        e
    };
    let mut fd = [[ZERO; 2]; 2];
    for &(i, j) in strains {
        for (hi, h) in [FD_H, 2.0 * FD_H].into_iter().enumerate() {
            if hi == 1 && !ratio.contains(&(i, j)) {
                continue;
            }
            let ep = energies(&strained(&sys.cell, i, j, h));
            let em = energies(&strained(&sys.cell, i, j, -h));
            for x in 0..2 {
                fd[hi][x][i][j] = (ep[x] - em[x]) / (2.0 * h);
            }
        }
    }
    let forces = [(0usize, 0usize), (1, 2)];
    let mut fd_f = [[0.0_f64; 2]; 2];
    for (c, &(at, x)) in forces.iter().enumerate() {
        let shifted = |h: f64| {
            let mut p = pos.to_vec();
            p[at][x] += h;
            energies(&cell_at(&p, cubic(a), mult))
        };
        let (ep, em) = (shifted(FD_H), shifted(-FD_H));
        for e in 0..2 {
            fd_f[c][e] = (ep[e] - em[e]) / (2.0 * FD_H);
        }
    }
    let mut led = Ledger::default();
    for &exx in exxs {
        let x = exx_idx(exx);
        let scf = kuhf(&ks, fs.jk(&ks, exx), sys.na, sys.nb);
        assert_spin_polarised(&sys, &scf);
        let s = k_uhf_fit_stress(&ks, &fs, &scf, exx, None);
        let m1 = miss_on(&s.de_deps, &fd[0][x], strains);
        let tag = format!("{} 1x1x3 {exx:?}", sys.name);
        led.below(
            &format!("{tag} stress vs FD (h, {} strains)", strains.len()),
            m1,
            s_bar,
        );
        for &(i, j) in ratio {
            let (e1, e2) = (
                (s.de_deps[i][j] - fd[0][x][i][j]).abs(),
                (s.de_deps[i][j] - fd[1][x][i][j]).abs(),
            );
            eprintln!(
                "  {tag} ({i},{j}): FD h {e1:.2e}, 2h {e2:.2e}, ratio {:.2}",
                e2 / e1
            );
        }
        // The h^2 ratio on the worst ratio component (the others can sit
        // at the SCF-noise floor, where the ratio is meaningless).
        let (wi, wj) = *ratio
            .iter()
            .max_by(|p, q| {
                let d = |&(i, j): &(usize, usize)| (s.de_deps[i][j] - fd[0][x][i][j]).abs();
                d(p).total_cmp(&d(q))
            })
            .expect("ratio components");
        let r = (s.de_deps[wi][wj] - fd[1][x][wi][wj]).abs()
            / (s.de_deps[wi][wj] - fd[0][x][wi][wj]).abs();
        led.below(
            &format!("{tag} |ratio - 4| ({wi},{wj}) (h^2)"),
            (r - 4.0).abs(),
            1.0,
        );
        let f = k_uhf_fit_force(&ks, &fs, &scf, exx, None);
        let wf = forces
            .iter()
            .enumerate()
            .fold(0.0_f64, |w, (c, &(at, xx))| {
                w.max((f.grad[(at, xx)] - fd_f[c][x]).abs())
            });
        led.below(&format!("{tag} force vs FD"), wf, f_bar);
        for m in [
            KStressMutation::FitCrossSpin,
            KStressMutation::SwapSpinFock,
            KStressMutation::NoAuxFt,
        ] {
            let sm = k_uhf_fit_stress(&ks, &fs, &scf, exx, Some(m));
            let bar = if m == KStressMutation::NoAuxFt {
                1e-5
            } else {
                MUTANT_BAR
            };
            led.above(
                &format!("{tag} stress mutant {m:?} vs FD"),
                miss_on(&sm.de_deps, &fd[0][x], strains),
                bar,
            );
        }
    }
    led.finish();
}

// ---------------------------------------------------------- (B) dense H4

fn kdense(ks: &HSys) -> KDenseAftEri {
    KDenseAftEri::build(
        &ks.cell,
        &ks.prep,
        &ks.mesh,
        &ks.hk.s,
        ExxDiv::None,
        &KDenseAftConfig {
            precision: ANCHOR_PRECISION,
            ..Default::default()
        },
    )
    .expect("dense k kernels")
}

/// Bar of the dense p-shell H4 supercell anchor. MEASURED (release, 861 s,
/// 12-atom supercell, default model-nucleus exponent): 5.31e-13 (none),
/// 5.50e-13 (ewald); bar 11x. Mutants (miss vs supercell/3): NoPhase
/// 2.18e-1, ConjD 3.50e-1, WrongQ 3.20e-1, NoInvNk 3.14, CrossSpinK 6.26e-1,
/// SwapSpinFock 2.91e-1 (none); GammaMadelung 5.31e-1, GammaMadelungStrain
/// 9.76e-1 (ewald). Stress only (the force supercell anchor of this system is
/// the RS-GDF one above).
const DENSE_OS_SC_BAR: f64 = 6e-12;

/// PHYSICS: the 1x1x3 (non-TRIM) dense k-UHF stress of the H4 triplet (s+p,
/// 3 alpha / 1 beta, both sectors populated) per cell is the explicit
/// 12-atom supercell's Gamma-UHF stress / 3, both exxdiv, at the default
/// model-nucleus exponent; the k-specific and spin mutants miss it.
/// ARTIFACT: as the H2 triplet anchor in `pbc_kstress.rs`, which has an
/// EMPTY beta sector: a defect in the beta exchange or a beta-only phase
/// would be invisible there and is visible here.
#[test]
#[ignore = "validation: k-point force/stress #421 follow-ups (dense H4 triplet s+p, 1x1x3 vs 12-atom supercell)"]
fn dense_open_shell_h4_triplet_sp_1x1x3_is_the_supercell_per_cell() {
    let bs = sp_basis_h();
    let sys = h4_triplet();
    let n = [1, 1, 3];
    let ks = hsys(sys.cell.clone(), &bs, n);
    let eri = kdense(&ks);
    let vm = ks.mesh.madelung(&ks.cell).unwrap();
    let scfs = [
        kuhf(
            &ks,
            Box::new(eri.jk_builder_with_madelung(0.0)),
            sys.na,
            sys.nb,
        ),
        kuhf(
            &ks,
            Box::new(eri.jk_builder_with_madelung(vm)),
            sys.na,
            sys.nb,
        ),
    ];
    assert_spin_polarised(&sys, &scfs[0]);
    let sc = supercell(&sys.cell, n, 7);
    let prep = prep_for(&sc, &bs);
    let hc = periodic_hcore(&sc, &prep, &hcore_cfg_off()).expect("gamma hcore");
    let geri = DenseAftEri::build(
        &sc,
        &prep,
        &hc.s,
        ExxDiv::None,
        ANCHOR_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("gamma dense AFT");
    let kst = |exx: ExxDiv, m: Option<KStressMutation>| {
        kpoint_uhf_stress(
            &ks.cell,
            &ks.prep,
            &ks.mesh,
            &hcore_cfg_off(),
            &ks.hk,
            KStressJk::Dense(&eri),
            &scfs[exx_idx(exx)],
            exx,
            &scfg(m),
        )
        .expect("kpoint_uhf_stress")
    };
    let mut led = Ledger::default();
    let mut refs = [ZERO; 2];
    for exx in EXX {
        let i = exx_idx(exx);
        let gscf = gamma_unrestricted(
            unfold(&ks.mesh, &scfs[i].density_alpha),
            unfold(&ks.mesh, &scfs[i].density_beta),
        );
        let gs = gamma_uhf_stress(
            &sc,
            &prep,
            &hcore_cfg_off(),
            &hc,
            &geri,
            &gscf,
            exx,
            &GammaStressConfig {
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
        )
        .expect("gamma uhf stress");
        refs[i] = scale3(&gs.de_deps, 1.0 / 3.0);
        let k = kst(exx, None);
        led.below(
            &format!("H4 dense 1x1x3 {exx:?} stress TOTAL vs sc/3"),
            max_mat(&k.de_deps, &refs[i]),
            DENSE_OS_SC_BAR,
        );
        led.above(
            &format!("H4 dense 1x1x3 {exx:?} |stress| (vacuity)"),
            max_mat(&k.de_deps, &ZERO),
            1e-3,
        );
    }
    for m in [
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
        KStressMutation::WrongQ,
        KStressMutation::NoInvNk,
        KStressMutation::CrossSpinK,
        KStressMutation::SwapSpinFock,
    ] {
        led.above(
            &format!("H4 dense 1x1x3 None stress mutant {m:?}"),
            max_mat(&kst(ExxDiv::None, Some(m)).de_deps, &refs[0]),
            MUTANT_BAR,
        );
    }
    for m in [
        KStressMutation::GammaMadelung,
        KStressMutation::GammaMadelungStrain,
    ] {
        led.above(
            &format!("H4 dense 1x1x3 Ewald stress mutant {m:?}"),
            max_mat(&kst(ExxDiv::Ewald, Some(m)).de_deps, &refs[1]),
            MUTANT_BAR,
        );
    }
    led.finish();
}

// ============================================================ (C) HI + ECP + rotation

const HI_A: [[f64; 3]; 3] = [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]];
const MOVED: [[f64; 3]; 2] = [[0.3, 0.2, 0.4], [0.9, -0.4, 3.3]];

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

fn lanl2dz_i_ecp() -> EcpDef {
    let ch = |l: i32, t: &[(i32, f64, f64)]| EcpShell {
        angular_momentum: l,
        terms: t
            .iter()
            .map(|&(n, z, d)| EcpTerm {
                coef: d,
                r_exp: n,
                gexp: z,
            })
            .collect(),
    };
    EcpDef {
        n_core: 46,
        shells: vec![
            ch(
                3,
                &[
                    (0, 1.0715702, -0.0747621),
                    (1, 44.1936028, -30.0811224),
                    (2, 12.9367609, -75.3722721),
                    (2, 3.1956412, -22.0563758),
                    (2, 0.8589806, -1.6979585),
                ],
            ),
            ch(
                0,
                &[
                    (0, 127.9202670, 2.9380036),
                    (1, 78.6211465, 41.2471267),
                    (2, 36.5146237, 287.8680095),
                    (2, 9.9065681, 114.3758506),
                    (2, 1.9420086, 37.6547714),
                ],
            ),
            ch(
                1,
                &[
                    (0, 13.0035304, 2.2222630),
                    (1, 76.0331404, 39.4090831),
                    (2, 24.1961684, 177.4075002),
                    (2, 6.4053433, 77.9889462),
                    (2, 1.5851786, 25.7547641),
                ],
            ),
            ch(
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

/// The rotating HI basis of `pbc_kpoint_ecp_rotation.rs` (H cc-pVDZ; I s/p as
/// general contractions incl. a single-primitive column, LANL2DZ ECP on I).
fn rot_basis() -> BasisSet {
    let mut shells = HashMap::new();
    let h = basis::bundled("cc-pvdz").expect("cc-pvdz");
    shells.insert(1, h.shells.get(&1).expect("H cc-pvdz").clone());
    let s_exps = [3.0, 1.0, 0.35];
    let p_exps = [2.0, 0.7];
    shells.insert(
        53,
        vec![
            shell(0, &s_exps, &[0.3, 0.5, 0.4]),
            shell(0, &s_exps, &[0.0, 0.0, 1.0]),
            shell(1, &p_exps, &[0.4, 0.8]),
            shell(1, &p_exps, &[0.0, 1.0]),
        ],
    );
    let mut ecps = HashMap::new();
    ecps.insert(53, lanl2dz_i_ecp());
    BasisSet {
        name: "HI-rot".into(),
        shells,
        ecps,
    }
}

/// Even-tempered s/p aux on H and I (`pbc_kpoint_ecp_rotation.rs`).
fn hi_aux() -> BasisSet {
    let sh = |l: i32, a: f64| Shell {
        l,
        pure: false,
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
    BasisSet {
        name: "hi-aux".into(),
        shells: m,
        ecps: HashMap::new(),
    }
}

fn atom(symbol: &str, z: i32, r: [f64; 3]) -> Atom {
    Atom {
        symbol: symbol.into(),
        z,
        x: r[0],
        y: r[1],
        zpos: r[2],
        ghost: false,
        n_core_ecp: 0,
    }
}

fn hi_cell(pos: &[[f64; 3]], bs: &BasisSet) -> Cell {
    let atoms = vec![atom("H", 1, pos[0]), atom("I", 53, pos[1])];
    let mut mol = Molecule {
        atoms,
        charge: 0,
        multiplicity: 1,
    };
    mol.apply_ecp(bs);
    Cell::new(mol, HI_A).expect("ECP cell")
}

fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

fn hi_hcore_cfg(rot: SrColumnRotation) -> PeriodicHcoreConfig {
    PeriodicHcoreConfig {
        precision: HCORE_PRECISION,
        ..PeriodicHcoreConfig::with_omega(OMEGA)
    }
    .with_sr_column_rotation(rot)
}

/// One HI pipeline: k hcore (rotation `rot`), RS-GDF (optionally range
/// split), k-RHF (exxdiv none).
struct HiFit {
    cell: Cell,
    prep: PreparedBasis,
    mesh: KPointMesh,
    hcfg: PeriodicHcoreConfig,
    hk: PeriodicHcoreK,
    cfg: KRsGdfConfig,
    aux: PreparedBasis,
    gdf: KRsGdf,
    scf: KScfResult,
}

#[derive(Clone, Copy)]
struct HiSpec {
    n: [usize; 3],
    centring: MeshCentring,
    rot: SrColumnRotation,
    split: Option<f64>,
}

fn hi_fit(cell: Cell, bs: &BasisSet, sp: HiSpec) -> HiFit {
    let prep = prep_for(&cell, bs);
    let mesh = KPointMesh::new(&cell, sp.n, sp.centring).expect("mesh");
    let hcfg = hi_hcore_cfg(sp.rot);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcfg).expect("hcore(k)");
    assert!(hk.n_ecp_triples > 0, "no ECP triples: the cell is vacuous");
    let cfg = KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            sr_column_rotation: sp.rot,
            range_split: sp.split.map(RangeSplit::new),
            ..Default::default()
        },
        mutation: None,
    };
    let aux = PreparedBasis::new(cell.mol(), &hi_aux()).expect("aux prep");
    let gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &cfg).expect("KRsGdf");
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk: Box::new(gdf.jk_builder_with_madelung(0.0)),
    };
    let scf = solve_krhf_injected(&cell, &mesh, &kscf_cfg(), inj).expect("k-RHF");
    assert!(scf.converged, "k-RHF not converged ({} it)", scf.iterations);
    HiFit {
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

impl HiFit {
    fn src(&self) -> KRsGdfGradSource<'_> {
        KRsGdfGradSource {
            gdf: &self.gdf,
            cfg: &self.cfg,
            aux: &self.aux,
        }
    }
    fn force(&self, m: Option<KGradMutation>) -> KGradient {
        kpoint_rhf_gradient(
            &self.cell,
            &self.prep,
            &self.mesh,
            &self.hcfg,
            &self.hk,
            KGradJk::RsGdf(self.src()),
            &self.scf,
            ExxDiv::None,
            &gcfg(m),
        )
        .expect("fit force")
    }
    fn stress(&self, m: Option<KStressMutation>) -> KStress {
        kpoint_rhf_stress(
            &self.cell,
            &self.prep,
            &self.mesh,
            &self.hcfg,
            &self.hk,
            KStressJk::RsGdf(self.src()),
            &self.scf,
            ExxDiv::None,
            &scfg(m),
        )
        .expect("fit stress")
    }
    fn kept(&self) -> Vec<usize> {
        self.gdf.stats().per_q.iter().map(|q| q.naux_kept).collect()
    }
    fn split_counter(&self, name: &str) -> usize {
        self.gdf
            .stats()
            .split_counters
            .iter()
            .find(|(n, _)| *n == name)
            .map(|&(_, v)| v)
            .unwrap_or(0)
    }
}

/// Central FD of the HI fitted energy: force components `(atom, axis)` and
/// strain components `(i, j)`, with the per-q kept metric counts held.
fn hi_fd(
    bs: &BasisSet,
    sp: HiSpec,
    kept0: &[usize],
    forces: &[(usize, usize)],
    strains: &[(usize, usize)],
) -> (Vec<f64>, Vec<f64>) {
    let e = |cell: Cell| {
        let r = hi_fit(cell, bs, sp);
        assert_eq!(r.kept(), kept0, "the metric cut changed (E piecewise)");
        r.scf.energy
    };
    let fd_f = forces
        .iter()
        .map(|&(a, x)| {
            let ep = e(hi_cell(&moved(&MOVED, a, x, FD_H), bs));
            let em = e(hi_cell(&moved(&MOVED, a, x, -FD_H), bs));
            (ep - em) / (2.0 * FD_H)
        })
        .collect();
    let base = hi_cell(&MOVED, bs);
    let fd_s = strains
        .iter()
        .map(|&(i, j)| {
            let ep = e(strained(&base, i, j, FD_H));
            let em = e(strained(&base, i, j, -FD_H));
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

/// SCF-level FD bars of the HI fitted energy (central FD, h = 1e-4: the h^2
/// truncation). MEASURED worst over the components each test checks:
/// force 3.49e-9 (C2: I x, I y, H z), 3.38e-9 (C1 split: I x, H z);
/// stress 2.73e-8 (C1 split: zz, xz). Bars >= 11x the measurement.
const HI_FD_F_BAR: f64 = 4e-8;
const HI_FD_S_BAR: f64 = 3e-7;
/// Smallest movement of the analytic total (ALL components) a production-wired
/// mutant must make: 33x the stress FD bar and 250x the force one. The FD
/// anchors cover only the components listed per test (force I x, H z, plus
/// I y in C2; stress zz, xz, plus xx in C3), so a mover is NOT shown to be
/// caught by the FD on those components. The smallest measured mover is
/// 2.13e-4 (`SplitGammaKernel`, force, C1), 21x above the floor.
const HI_MUTANT_FLOOR: f64 = 1e-5;
/// Split vs unsplit k force / stress on HI (each at its own density).
/// MEASURED force 5.2e-14, stress 1.27e-12 (energies 2.3e-14 apart; ECP
/// parts 1.4e-15 / 2.3e-14); bar 16x the stress, 7 orders below the
/// smallest split mutant (2.13e-4). A nonzero difference is also asserted
/// (a split that ran the unsplit code would agree to the last bit).
const HI_SPLIT_UNSPLIT_BAR: f64 = 2e-11;
/// Rotated vs unrotated RS-GDF force / stress of the same energy on the MP
/// mesh. MEASURED force 1.5e-14, stress 5.6e-13; bar 18x the stress.
const HI_ROT_VS_OFF_BAR: f64 = 1e-11;

const HI_FORCES: [(usize, usize); 2] = [(1, 0), (0, 2)];
const HI_STRAINS: [(usize, usize); 2] = [(2, 2), (0, 2)];

/// (C1) RANGE-SPLIT RS-GDF with the ECP and the rotating basis, Gamma-centred
/// 1x1x3. PHYSICS: the split is an exact decomposition of the same fitted
/// energy, so the split k force and stress equal the UNSPLIT ones (each at
/// its own converged density) up to the split's tolerance, with the ECP part
/// bitwise-independent of the fit; and the split analytic force (I x, H z)
/// and stress (zz, xz) match central FD of the SPLIT rotated energy. Every
/// split-specific force and stress mutant must move the total. ARTIFACT: the
/// moved-primitive counters must be > 0 and the split-vs-unsplit difference
/// nonzero (a split that ran the unsplit code would agree to 1e-15).
///
/// MEASURED (release, 360 s): moved orbital 6 / aux 7 primitives, SR3
/// triplets 341719 -> 48205, 3 rotated fit columns; split vs unsplit force
/// 5.2e-14, stress 1.27e-12; FD force 3.38e-9, stress 2.73e-8; mutant
/// ledger (total moves): force SplitGammaKernel 2.13e-4, SplitNoSmoothOverlap
/// 1.32e-3, SplitFullG0 9.38e-3, RotFitWrongResidue 2.65e-2, RotSwapPair
/// 7.73e-3, NoEcp 1.68e-1; stress SplitNoSrKernelStrain 1.07e-1,
/// SplitNoSmoothPair 7.32e-1, SplitFullG0 4.96e-1, SplitNoSmoothOverlap
/// 1.42e-2, SplitFlipSrSlope 2.13e-1, SplitNoSmoothVolume 5.76e-1,
/// RotFitWrongResidue 6.32e-2, NoEcp 1.18e-1.
#[test]
#[ignore = "validation: k-point force/stress #421 follow-ups (range-split RS-GDF with an ECP and rotation)"]
fn rotated_ecp_split_rsgdf_force_and_stress_match_the_unsplit_and_fd() {
    let bs = rot_basis();
    let split = HiSpec {
        n: [1, 1, 3],
        centring: MeshCentring::Gamma,
        rot: SrColumnRotation::on(),
        split: Some(1.0),
    };
    let unsplit = HiSpec {
        split: None,
        ..split
    };
    let s = hi_fit(hi_cell(&MOVED, &bs), &bs, split);
    let u = hi_fit(hi_cell(&MOVED, &bs), &bs, unsplit);
    let (os, as_) = (
        s.split_counter("rsgdf split orbital prims smooth"),
        s.split_counter("rsgdf split aux prims smooth"),
    );
    let (ss, us) = (s.gdf.stats(), u.gdf.stats());
    eprintln!(
        "HI split: moved orbital {os}, aux {as_}; SR3 {} -> {}; rotated fit columns {} / {}",
        us.n_sr3_triplets, ss.n_sr3_triplets, ss.sr_rotated_columns, us.sr_rotated_columns
    );
    assert!(
        os > 0 && as_ > 0,
        "nothing moved: the split path did not run"
    );
    assert!(ss.n_sr3_triplets < us.n_sr3_triplets, "SR3 not shrunk");
    assert!(us.split_counters.is_empty(), "unsplit ran the split path");
    assert!(ss.sr_rotated_columns > 0, "the split fit did not rotate");
    assert_eq!(s.hk.sr_rotated_columns, 3);
    let (fs, fu) = (s.force(None), u.force(None));
    let (ts, tu) = (s.stress(None), u.stress(None));
    assert_eq!((fs.hcore_rotated_columns, ts.hcore_rotated_columns), (3, 3));
    let mut led = Ledger::default();
    let (df, ds) = (
        grad_miss(&fs.grad, &fu.grad),
        max_mat(&ts.de_deps, &tu.de_deps),
    );
    eprintln!(
        "  split - unsplit energies {:.2e}",
        (s.scf.energy - u.scf.energy).abs()
    );
    led.below("HI split vs unsplit force TOTAL", df, HI_SPLIT_UNSPLIT_BAR);
    led.below("HI split vs unsplit stress TOTAL", ds, HI_SPLIT_UNSPLIT_BAR);
    led.above(
        "HI split vs unsplit stress (not the unsplit path)",
        ds,
        1e-15,
    );
    led.below(
        "HI split vs unsplit force ECP part",
        grad_miss(&fs.parts.ecp, &fu.parts.ecp),
        HI_SPLIT_UNSPLIT_BAR,
    );
    led.below(
        "HI split vs unsplit stress ECP part",
        max_mat(&ts.parts.ecp, &tu.parts.ecp),
        HI_SPLIT_UNSPLIT_BAR,
    );
    led.above("HI |F_ecp| (vacuity)", amax(&fs.parts.ecp), 1e-3);
    led.above("HI |S_ecp| (vacuity)", max_mat(&ts.parts.ecp, &ZERO), 1e-3);
    let (fd_f, fd_s) = hi_fd(&bs, split, &s.kept(), &HI_FORCES, &HI_STRAINS);
    led.below(
        "HI split force vs FD of the split energy",
        fd_miss_f(&fs, &HI_FORCES, &fd_f),
        HI_FD_F_BAR,
    );
    led.below(
        "HI split stress vs FD of the split energy",
        fd_miss_s(&ts, &HI_STRAINS, &fd_s),
        HI_FD_S_BAR,
    );
    for m in [
        KGradMutation::SplitGammaKernel,
        KGradMutation::SplitNoSmoothOverlap,
        KGradMutation::SplitFullG0,
        KGradMutation::RotFitWrongResidue,
        KGradMutation::RotSwapPair,
        KGradMutation::NoEcp,
    ] {
        led.above(
            &format!("HI split force mutant {m:?} moves total"),
            grad_miss(&s.force(Some(m)).grad, &fs.grad),
            HI_MUTANT_FLOOR,
        );
    }
    for m in [
        KStressMutation::SplitNoSrKernelStrain,
        KStressMutation::SplitNoSmoothPair,
        KStressMutation::SplitFullG0,
        KStressMutation::SplitNoSmoothOverlap,
        KStressMutation::SplitFlipSrSlope,
        KStressMutation::SplitNoSmoothVolume,
        KStressMutation::RotFitWrongResidue,
        KStressMutation::NoEcp,
    ] {
        led.above(
            &format!("HI split stress mutant {m:?} moves total"),
            max_mat(&s.stress(Some(m)).de_deps, &ts.de_deps),
            HI_MUTANT_FLOOR,
        );
    }
    led.finish();
}

/// (C2) The FORCE mutants on the fitted path with the ECP and the rotating
/// basis (Gamma-centred 1x1x3, unsplit): the dense ECP anchors could only
/// assert that the fit-acting ones are no-ops. PHYSICS: the analytic force
/// (I x, I y, H z) matches central FD of the rotated fitted energy; then
/// every production-wired mutant that can act here moves the analytic total
/// (all components) by more than `HI_MUTANT_FLOOR`; the FD anchor covers I x,
/// I y and H z only. Mutants that are identities BY CONSTRUCTION on this system are
/// asserted to leave the total unchanged: `EcpMeshCountModuli` (Gamma-centred
/// mesh), `WrongQ` (dense exchange only), `GammaMadelung` (exxdiv none).
///
/// MEASURED (release, 294 s): FD 3.49e-9; |F_ecp| 1.68e-1; total moves:
/// FitAuxPhase 1.70e-3, FitNoMetric 4.95e-1, FitNoG0 6.77e-3,
/// RotFitWrongResidue 1.53e-1, RotSwapPair 2.49e-2, NoPhase 5.76e-2, ConjD
/// 2.01e-1, NoInvNk 1.30e-1, RotDropT 2.96e-2, RotUseT 2.68e-1, RotOneSided
/// 4.52e-3, RotFlipSign 4.26e-1, RotHcoreWrongResidue 1.49e-1, NoEcp
/// 1.68e-1, EcpNoCentre 6.55e-1, EcpCentreSign 1.31, EcpCentreWrongAtom
/// 6.55e-1, EcpWrongResidue 2.04e-1, EcpL0Only 1.94e-1, EcpM0Only 1.92e-2;
/// identities EcpMeshCountModuli, WrongQ, GammaMadelung exactly 0.
/// `RotOff` / `RotHcoreOff` are not listed: the rotated and unrotated walks
/// are the same function to the screen (blind by design, see
/// `pbc_sr_rotation_kderiv.rs`).
#[test]
#[ignore = "validation: k-point force/stress #421 follow-ups (fit-acting force mutants with an ECP)"]
fn rotated_ecp_rsgdf_force_mutants_move_the_fd_anchored_force() {
    let bs = rot_basis();
    let sp = HiSpec {
        n: [1, 1, 3],
        centring: MeshCentring::Gamma,
        rot: SrColumnRotation::on(),
        split: None,
    };
    let forces = [(1usize, 0usize), (1, 1), (0, 2)];
    let r = hi_fit(hi_cell(&MOVED, &bs), &bs, sp);
    let (fd_f, _) = hi_fd(&bs, sp, &r.kept(), &forces, &[]);
    let f = r.force(None);
    assert_eq!(f.hcore_rotated_columns, 3);
    let mut led = Ledger::default();
    led.below(
        "HI fit force vs FD",
        fd_miss_f(&f, &forces, &fd_f),
        HI_FD_F_BAR,
    );
    led.above("HI |F_ecp| (vacuity)", amax(&f.parts.ecp), 1e-3);
    for m in [
        KGradMutation::FitAuxPhase,
        KGradMutation::FitNoMetric,
        KGradMutation::FitNoG0,
        KGradMutation::RotFitWrongResidue,
        KGradMutation::RotSwapPair,
        KGradMutation::NoPhase,
        KGradMutation::ConjD,
        KGradMutation::NoInvNk,
        KGradMutation::RotDropT,
        KGradMutation::RotUseT,
        KGradMutation::RotOneSided,
        KGradMutation::RotFlipSign,
        KGradMutation::RotHcoreWrongResidue,
        KGradMutation::NoEcp,
        KGradMutation::EcpNoCentre,
        KGradMutation::EcpCentreSign,
        KGradMutation::EcpCentreWrongAtom,
        KGradMutation::EcpWrongResidue,
        KGradMutation::EcpL0Only,
        KGradMutation::EcpM0Only,
    ] {
        led.above(
            &format!("HI fit force mutant {m:?} moves total"),
            grad_miss(&r.force(Some(m)).grad, &f.grad),
            HI_MUTANT_FLOOR,
        );
    }
    for m in [
        KGradMutation::EcpMeshCountModuli,
        KGradMutation::WrongQ,
        KGradMutation::GammaMadelung,
    ] {
        led.below(
            &format!("HI fit force identity {m:?} leaves total"),
            grad_miss(&r.force(Some(m)).grad, &f.grad),
            1e-12,
        );
    }
    led.finish();
}

/// (C3) SHIFTED (Monkhorst-Pack 1x1x2, residue moduli [1, 1, 4]) mesh with
/// the ECP and the rotating basis, RS-GDF. PHYSICS: analytic force (I x,
/// H z) and stress (zz, xz, xx) vs central FD of the rotated MP energy; the
/// rotated analytic equals the unrotated analytic of the same energy.
/// ARTIFACT: the ECP image weights on a shifted mesh are periodic modulo the
/// residue moduli (2N along z), not the mesh counts: `EcpMeshCountModuli`
/// and `EcpWrongResidue`, identities on Gamma-centred 1x1x2, must act here;
/// `NoPhase` / `ConjD` act on the complex (non-TRIM) shifted k points.
///
/// MEASURED (release, 244 s): FD force 3.22e-9, stress 1.20e-8 (zz, xz,
/// xx); rotated vs unrotated force 1.5e-14, stress 5.6e-13; total moves
/// (force / stress): EcpMeshCountModuli 1.99e-1 / 7.98e-1, EcpWrongResidue
/// 3.98e-1 / 1.60, NoPhase 8.53e-2 / 5.99e-1, ConjD 3.54e-1 / 9.35e-1,
/// RotFitWrongResidue 1.78e-1 / 6.07e-1, RotHcoreWrongResidue 2.23e-1 /
/// 4.52e-1, RotSwapPair (force) 5.21e-2.
#[test]
#[ignore = "validation: k-point force/stress #421 follow-ups (shifted MP mesh with an ECP and rotation)"]
fn rotated_ecp_mp_mesh_rsgdf_force_and_stress_match_fd() {
    let bs = rot_basis();
    let sp = HiSpec {
        n: [1, 1, 2],
        centring: MeshCentring::MonkhorstPack,
        rot: SrColumnRotation::on(),
        split: None,
    };
    let strains = [(2usize, 2usize), (0, 2), (0, 0)];
    let r = hi_fit(hi_cell(&MOVED, &bs), &bs, sp);
    assert_eq!(r.mesh.residue_moduli(), [1, 1, 4], "not a shifted mesh");
    assert_eq!(r.hk.sr_rotated_columns, 3);
    assert!(
        r.gdf.stats().sr_rotated_columns > 0,
        "the fit did not rotate"
    );
    let (fd_f, fd_s) = hi_fd(&bs, sp, &r.kept(), &HI_FORCES, &strains);
    let (f, s) = (r.force(None), r.stress(None));
    assert_eq!((f.hcore_rotated_columns, s.hcore_rotated_columns), (3, 3));
    let mut led = Ledger::default();
    led.below(
        "HI MP force vs FD",
        fd_miss_f(&f, &HI_FORCES, &fd_f),
        HI_FD_F_BAR,
    );
    led.below(
        "HI MP stress vs FD",
        fd_miss_s(&s, &strains, &fd_s),
        HI_FD_S_BAR,
    );
    led.above("HI MP |F_ecp| (vacuity)", amax(&f.parts.ecp), 1e-3);
    led.above(
        "HI MP |S_ecp| (vacuity)",
        max_mat(&s.parts.ecp, &ZERO),
        1e-3,
    );
    for m in [
        KGradMutation::EcpMeshCountModuli,
        KGradMutation::EcpWrongResidue,
        KGradMutation::NoPhase,
        KGradMutation::ConjD,
        KGradMutation::RotFitWrongResidue,
        KGradMutation::RotHcoreWrongResidue,
        KGradMutation::RotSwapPair,
    ] {
        led.above(
            &format!("HI MP force mutant {m:?} moves total"),
            grad_miss(&r.force(Some(m)).grad, &f.grad),
            HI_MUTANT_FLOOR,
        );
    }
    for m in [
        KStressMutation::EcpMeshCountModuli,
        KStressMutation::EcpWrongResidue,
        KStressMutation::NoPhase,
        KStressMutation::ConjD,
        KStressMutation::RotFitWrongResidue,
        KStressMutation::RotHcoreWrongResidue,
    ] {
        led.above(
            &format!("HI MP stress mutant {m:?} moves total"),
            max_mat(&r.stress(Some(m)).de_deps, &s.de_deps),
            HI_MUTANT_FLOOR,
        );
    }
    // The unrotated pipeline LAST (an independent walk of the same energy).
    let off = hi_fit(
        hi_cell(&MOVED, &bs),
        &bs,
        HiSpec {
            rot: SrColumnRotation::Off,
            ..sp
        },
    );
    assert_eq!(off.gdf.stats().sr_rotated_columns, 0);
    let (fo, so) = (off.force(None), off.stress(None));
    led.below(
        "HI MP rotated vs unrotated force",
        grad_miss(&f.grad, &fo.grad),
        HI_ROT_VS_OFF_BAR,
    );
    led.below(
        "HI MP rotated vs unrotated stress",
        max_mat(&s.de_deps, &so.de_deps),
        HI_ROT_VS_OFF_BAR,
    );
    led.finish();
}
