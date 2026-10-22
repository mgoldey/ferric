//! Analytic forces of the k-point open-shell KS energy (`ferric_pbc::kuxcgrad`,
//! module doc) for LDA, GGA and global hybrids.
//!
//! Anchors, each with its artifact hypothesis next to the physics hypothesis:
//! a = 1 with a zero XC == the k-point UHF force per part; the closed-shell
//! limit == the merged RKS force per part (BLIND to every spin swap or
//! one-spin-dropped defect: both spins are equal there); 1x1x1 and the
//! non-TRIM 1x1x3 supercell identities against the Gamma UKS gradient per
//! part on a genuinely open-shell cell (H3 doublet, Da != Db); central FD of
//! the energy (fresh SCF per displacement) at a verified safe step with an
//! explicit shared neighbour cutoff; thread bit-identity; the per-spin Fock
//! guard; typed refusals; production-wired spin-specific mutants.
//!
//! NOT covered: p/d shells and heavy atoms / ECP, shifted meshes, metals,
//! range-separated hybrids / VV10 (refused), meta-GGA (`pbc_kuxc_mgga.rs`), the production
//! (75, 302) grid (a (40, 110) FD is in the ignored tier), stress.

mod common;

use common::*;
use ferric_core::basis::{BasisSet, Shell};
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES};
use ferric_pbc::dft::{gamma_uks, GammaUksConfig, PeriodicGridConfig};
use ferric_pbc::grad::{gamma_uks_gradient_with, GammaGradConfig, GammaGradient};
use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
use ferric_pbc::hcore::periodic_hcore;
use ferric_pbc::hcore::PeriodicHcoreConfig;
use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
use ferric_pbc::kdft::{solve_krks, KRksConfig};
use ferric_pbc::kgrad::{
    kpoint_uhf_gradient, KGradConfig, KGradJk, KGradMutation, KRsGdfGradSource,
};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::kscf::{KJkKind, KPointInjection, KPointJk};
use ferric_pbc::kuks::{solve_kuks, solve_kuks_on_grid, KUksConfig, KUksResult};
use ferric_pbc::kuscf::{solve_kuks_injected, KPointXcPolarized, KUScfResult};
use ferric_pbc::kuxcgrad::UKS_COMMUTATOR_TOL;
use ferric_pbc::kuxcgrad::{
    kpoint_null_xc_uks_gradient, kpoint_uks_gradient, KUSpinMutation, KUksGradConfig, KUksGradient,
};
use ferric_pbc::kxcgrad::{
    kpoint_rks_gradient, KRksGradConfig, KRksGradient, KXcGrid, KXcMutation,
};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
use ferric_pbc::rsgdf::RsGdfConfig;
use ferric_pbc::uhf::GammaUhfIntegrals;
use ferric_pbc::PeriodicGrid;
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashMap;

const OMEGA: f64 = 0.8;
const AMPLE: usize = 1 << 31;
/// Explicit XC neighbour cutoff shared by every energy and force here (the
/// default depends on the cell and makes one-step energy discontinuities).
const CUTOFF: f64 = 14.0;
/// FD bar and step, from `fd_step_probe` (H3 doublet PBE, 1x1x3, residual
/// FD - analytic, two components, two SCF tolerances):
/// * cutoff 10 Bohr, component (2,1): +8.2e-8, +1.6e-7, +3.3e-7 at h = 1e-4,
///   5e-5, 2.5e-5 (exactly 1/h, identical at both SCF tolerances): the
///   energy's hard neighbour-mask step, NOT noise and NOT a force error;
/// * cutoff 14 Bohr: |residual| <= 3.8e-10 at h = 5e-5 for both components
///   (h^2 truncation 8e-10 at 1e-4, falling to 6e-11 at 5e-5; SCF-energy
///   noise ~1e-9 at h = 1e-5, the same at energy_conv 1e-13 and 1e-14).
/// Bar 1e-8: ~25x the worst correct residual, ~1e5 below the smallest
/// mutant miss (1.6e-3, NoHessian).
const FD_BAR: f64 = 1e-8;
const FD_H: f64 = 5e-5;
const ANCHOR_PRECISION: f64 = 1e-12;
const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
/// H3 doublet (na = 2, nb = 1): a genuinely spin-polarised cell, Da != Db.
/// (energy_conv, grad_conv) of the deliberately loose UKS SCF of the gate test.
const LOOSE_UKS: (f64, f64) = (1e-5, 1e-2);
const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
enum Jk {
    Dense,
    Gdf,
}

fn et_sp_aux() -> BasisSet {
    let mut shells = Vec::new();
    for a in [4.7, 1.9, 0.75, 0.3] {
        shells.push(Shell {
            l: 0,
            pure: false,
            exponents: vec![a],
            coefficients: vec![1.0],
        });
    }
    for a in [1.25, 0.5] {
        shells.push(Shell {
            l: 1,
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

fn kgdf_cfg() -> KRsGdfConfig {
    KRsGdfConfig {
        gdf: RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
        mutation: None,
    }
}

thread_local! {
    /// (neighbour cutoff, energy_conv, grad_conv) of the k SCF on THIS test
    /// thread; only the FD probe changes the defaults.
    static SETTINGS: std::cell::Cell<(f64, f64, f64)> =
        const { std::cell::Cell::new((CUTOFF, 1e-13, 1e-10)) };
}

fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(SETTINGS.with(|s| s.get()).0);
    g
}

fn cell_of(pos: &[[f64; 3]], mult: usize) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, cubic(4.0)).expect("cell")
}

fn mult_of(pos: &[[f64; 3]]) -> usize {
    if pos.len() % 2 == 1 {
        2
    } else {
        1
    }
}

fn kuks_cfg(cell: &Cell, functional: &str, jk: Jk, exxdiv: ExxDiv) -> KUksConfig {
    let mut cfg = KUksConfig::new(cell, exxdiv, functional);
    cfg.kuhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    let (_, econv, gconv) = SETTINGS.with(|s| s.get());
    cfg.kuhf.scf.energy_conv = econv;
    cfg.kuhf.scf.grad_conv = gconv;
    cfg.kuhf.scf.max_iter = 300;
    if jk == Jk::Gdf {
        cfg.kuhf.jk = KJkKind::RsGdf;
        cfg.kuhf.rsgdf = kgdf_cfg();
    }
    cfg.grid = grid_cfg();
    cfg
}

fn scf(
    pos: &[[f64; 3]],
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    exxdiv: ExxDiv,
) -> KUksResult {
    let cell = cell_of(pos, mult_of(pos));
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let cfg = kuks_cfg(&cell, functional, jk, exxdiv);
    let aux = (jk == Jk::Gdf).then(|| prep_for(&cell, &et_sp_aux()));
    let r = solve_kuks(&cell, &prep, aux.as_ref(), &mesh, &cfg)
        .unwrap_or_else(|e| panic!("solve_kuks {functional}: {e}"));
    assert!(r.scf.converged);
    r
}

struct Fx {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    aux: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    res: KUksResult,
    jk: Jk,
    exxdiv: ExxDiv,
    functional: String,
}

fn fixture(pos: &[[f64; 3]], mesh_n: [usize; 3], functional: &str, jk: Jk, exxdiv: ExxDiv) -> Fx {
    let cell = cell_of(pos, mult_of(pos));
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let res = scf(pos, mesh_n, functional, jk, exxdiv);
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let aux = prep_for(&cell, &et_sp_aux());
    Fx {
        cell,
        prep,
        aux,
        mesh,
        hk,
        res,
        jk,
        exxdiv,
        functional: functional.to_string(),
    }
}

fn gcfg(
    exxdiv: Option<ExxDiv>,
    xm: Option<KXcMutation>,
    sm: Option<KUSpinMutation>,
    gm: Option<KGradMutation>,
) -> KUksGradConfig {
    KUksGradConfig {
        base: KRksGradConfig {
            grad: KGradConfig {
                budget_bytes: Some(AMPLE),
                mutation: gm,
                ..Default::default()
            },
            grid: KXcGrid::Atom(grid_cfg()),
            exxdiv,
            mutation: xm,
            ..Default::default()
        },
        spin_mutation: sm,
    }
}

fn force_with(
    fx: &Fx,
    functional: &str,
    scf: &KUScfResult,
    cfg: &KUksGradConfig,
) -> Result<KUksGradient, ferric_core::FerricError> {
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    match fx.jk {
        Jk::Dense => {
            let eri = KDenseAftEri::build(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &fx.hk.s,
                fx.exxdiv,
                &Default::default(),
            )
            .unwrap();
            kpoint_uks_gradient(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &hcfg,
                &fx.hk,
                KGradJk::Dense(&eri),
                scf,
                functional,
                cfg,
            )
        }
        Jk::Gdf => {
            let kc = kgdf_cfg();
            let gdf = KRsGdf::build(&fx.cell, &fx.prep, &fx.aux, &fx.mesh, &fx.hk.s, &kc)
                .unwrap()
                .with_exxdiv(fx.exxdiv);
            kpoint_uks_gradient(
                &fx.cell,
                &fx.prep,
                &fx.mesh,
                &hcfg,
                &fx.hk,
                KGradJk::RsGdf(KRsGdfGradSource {
                    gdf: &gdf,
                    cfg: &kc,
                    aux: &fx.aux,
                }),
                scf,
                functional,
                cfg,
            )
        }
    }
}

fn force(fx: &Fx) -> KUksGradient {
    force_with(
        fx,
        &fx.functional,
        &fx.res.scf,
        &gcfg(Some(fx.exxdiv), None, None, None),
    )
    .unwrap()
}

fn force_mut(
    fx: &Fx,
    xm: Option<KXcMutation>,
    sm: Option<KUSpinMutation>,
    gm: Option<KGradMutation>,
) -> KUksGradient {
    force_with(
        fx,
        &fx.functional,
        &fx.res.scf,
        &gcfg(Some(fx.exxdiv), xm, sm, gm),
    )
    .unwrap()
}

fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

#[allow(clippy::too_many_arguments)]
fn fd(
    pos: &[[f64; 3]],
    a: usize,
    x: usize,
    h: f64,
    mesh_n: [usize; 3],
    f: &str,
    jk: Jk,
    exxdiv: ExxDiv,
) -> f64 {
    let e = |p: &[[f64; 3]]| scf(p, mesh_n, f, jk, exxdiv).scf.energy;
    (e(&moved(pos, a, x, h)) - e(&moved(pos, a, x, -h))) / (2.0 * h)
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn parts_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn kcfg() -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

// ---------------------------------------------------- zero-XC fixed SCF

struct ZeroXcPol(usize);
impl KPointXcPolarized for ZeroXcPol {
    #[allow(clippy::type_complexity)]
    fn build_polarized(
        &mut self,
        da: &[Array2<Complex64>],
        _db: &[Array2<Complex64>],
    ) -> Result<(f64, Vec<Array2<Complex64>>, Vec<Array2<Complex64>>), ferric_core::FerricError>
    {
        let z = || da.iter().map(|_| Array2::zeros((self.0, self.0))).collect();
        Ok((0.0, z(), z()))
    }
}

struct Zx {
    cell: Cell,
    prep: ferric_integrals::basis_bridge::PreparedBasis,
    aux: ferric_integrals::basis_bridge::PreparedBasis,
    mesh: KPointMesh,
    hk: PeriodicHcoreK,
    scf: KUScfResult,
    jk: Jk,
    exxdiv: ExxDiv,
}

fn dense_cfg() -> KDenseAftConfig {
    KDenseAftConfig {
        precision: ANCHOR_PRECISION,
        ..Default::default()
    }
}

/// Hartree + `a` exact-exchange k UKS with a zero XC (a = 1 is k-point UHF).
fn zero_xc(pos: &[[f64; 3]], mesh_n: [usize; 3], jk: Jk, exxdiv: ExxDiv, a: f64) -> Zx {
    let cell = cell_of(pos, mult_of(pos));
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let aux = prep_for(&cell, &et_sp_aux());
    let mesh = KPointMesh::gamma_centred(&cell, mesh_n).unwrap();
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hk = periodic_hcore_kpts(&cell, &prep, &mesh, &hcfg).unwrap();
    let mut sc = ferric_pbc::kscf::KScfConfig::default();
    sc.energy_conv = 1e-13;
    sc.grad_conv = 1e-10;
    sc.max_iter = 300;
    let vm = match exxdiv {
        ExxDiv::None => 0.0,
        ExxDiv::Ewald => mesh.madelung(&cell).unwrap(),
    };
    let eri;
    let gdf;
    let jkb: Box<dyn KPointJk + '_> = match jk {
        Jk::Dense => {
            eri = KDenseAftEri::build(&cell, &prep, &mesh, &hk.s, ExxDiv::None, &dense_cfg())
                .unwrap();
            Box::new(eri.jk_builder_with_madelung(vm))
        }
        Jk::Gdf => {
            gdf = KRsGdf::build(&cell, &prep, &aux, &mesh, &hk.s, &kgdf_cfg()).unwrap();
            Box::new(gdf.jk_builder_with_madelung(vm))
        }
    };
    let inj = KPointInjection {
        s: hk.s.clone(),
        h: hk.h.clone(),
        vnn: hk.enn,
        jk: jkb,
    };
    let ne = cell.mol().nelec() as usize;
    let (na, nb) = (ne / 2 + ne % 2, ne / 2);
    let mut z = ZeroXcPol(prep.nbasis());
    let scf = solve_kuks_injected(&cell, &mesh, &sc, inj, na, nb, &mut z, a).unwrap();
    assert!(scf.converged);
    Zx {
        cell,
        prep,
        aux,
        mesh,
        hk,
        scf,
        jk,
        exxdiv,
    }
}

fn with_jk<R>(z: &Zx, f: impl FnOnce(KGradJk<'_>) -> R) -> R {
    match z.jk {
        Jk::Dense => {
            let eri =
                KDenseAftEri::build(&z.cell, &z.prep, &z.mesh, &z.hk.s, z.exxdiv, &dense_cfg())
                    .unwrap();
            f(KGradJk::Dense(&eri))
        }
        Jk::Gdf => {
            let kc = kgdf_cfg();
            let gdf = KRsGdf::build(&z.cell, &z.prep, &z.aux, &z.mesh, &z.hk.s, &kc)
                .unwrap()
                .with_exxdiv(z.exxdiv);
            f(KGradJk::RsGdf(KRsGdfGradSource {
                gdf: &gdf,
                cfg: &kc,
                aux: &z.aux,
            }))
        }
    }
}

/// (a) EXACTNESS ANCHOR. a = 1 with a zero XC IS the k-point UHF force: same
/// `assemble(Unrestricted)`, but the KS entry rebuilds F_sigma itself and takes
/// its exxdiv from the config. Artifact hypothesis: a leaked XC term, a K
/// scaled by anything but 1, a swapped spin Fock would show as nonzero; physics
/// predicts 0 to roundoff.
#[test]
#[ignore = "dense+RS-GDF x none/ewald; measured 165 s (box load 15-60: inflated)"]
fn a_equal_one_with_zero_xc_is_the_uhf_force_per_part() {
    for jk in [Jk::Dense, Jk::Gdf] {
        for exxdiv in [ExxDiv::None, ExxDiv::Ewald] {
            let z = zero_xc(&H3, [1, 1, 3], jk, exxdiv, 1.0);
            let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
            let ks = with_jk(&z, |j| {
                kpoint_null_xc_uks_gradient(
                    &z.cell,
                    &z.prep,
                    &z.mesh,
                    &hcfg,
                    &z.hk,
                    j,
                    &z.scf,
                    1.0,
                    &gcfg(Some(exxdiv), None, None, None),
                )
                .unwrap()
            });
            let hf = with_jk(&z, |j| {
                kpoint_uhf_gradient(
                    &z.cell,
                    &z.prep,
                    &z.mesh,
                    &hcfg,
                    &z.hk,
                    j,
                    &z.scf,
                    exxdiv,
                    &kcfg(),
                )
                .unwrap()
            });
            let (p, q) = (&ks.hf.parts, &hf.parts);
            let rows = [
                ("overlap", parts_diff(&p.overlap, &q.overlap)),
                ("kinetic", parts_diff(&p.kinetic, &q.kinetic)),
                ("vsr", parts_diff(&p.vsr_basis, &q.vsr_basis)),
                ("coulomb", parts_diff(&p.coulomb, &q.coulomb)),
                ("exchange", parts_diff(&p.exchange, &q.exchange)),
                ("fit_orb_sr", parts_diff(&p.fit_orb_sr, &q.fit_orb_sr)),
                ("fit_g0", parts_diff(&p.fit_g0, &q.fit_g0)),
                ("total", parts_diff(&ks.grad, &hf.grad)),
            ];
            for (n, d) in rows {
                eprintln!("{jk:?} {exxdiv:?}: a=1 null-XC vs UHF force, {n:<10} {d:.1e}");
                assert!(d < 1e-13, "{jk:?} {exxdiv:?} {n}: {d:e}");
            }
            // Not vacuous: open shell (Da != Db) and the exchange walks matter.
            let na: f64 = z
                .scf
                .density_alpha
                .iter()
                .map(|m| m.iter().map(|x| x.norm()).sum::<f64>())
                .sum();
            let nb: f64 = z
                .scf
                .density_beta
                .iter()
                .map(|m| m.iter().map(|x| x.norm()).sum::<f64>())
                .sum();
            assert!((na - nb).abs() > 0.1, "closed-shell-like fixture");
            assert!(max_abs(&hf.grad) > 1e-3);
            match jk {
                Jk::Dense => assert!(max_abs(&hf.parts.exchange) > 1e-4),
                Jk::Gdf => assert!(max_abs(&hf.parts.fit_orb_sr) > 1e-4),
            }
        }
    }
}

// ------------------------------------------------------ closed-shell limit

/// (b) CLOSED-SHELL LIMIT == the merged RKS force, per part. Independent SCF
/// loops (k-UHF skeleton with polarized kernel vs the closed-shell loop). Artifact
/// hypothesis: a spin-factor slip (2 vs 1) or a mishandled gate would break
/// equality; physics predicts agreement at the SCF-density floor. BLIND to:
/// alpha/beta swaps, a dropped spin that is doubled, anything symmetric in the
/// two spins.
#[test]
fn closed_shell_limit_equals_the_rks_force_per_part() {
    closed_shell_vs_rks(&["PBE0"]);
}

#[test]
#[ignore = "closed-shell UKS vs RKS for LDA and PBE; measured ~25 s (box load 15-60: inflated)"]
fn closed_shell_limit_equals_the_rks_force_per_part_lda_pbe() {
    closed_shell_vs_rks(&["LDA", "PBE"]);
}

fn closed_shell_vs_rks(functionals: &[&str]) {
    for &f in functionals {
        let fx = fixture(&H2, [1, 1, 3], f, Jk::Dense, ExxDiv::Ewald);
        let u = force(&fx);
        // RKS twin.
        let cell = &fx.cell;
        let mut rcfg = KRksConfig::new(cell, ExxDiv::Ewald, f);
        rcfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
        rcfg.krhf.scf.energy_conv = 1e-13;
        rcfg.krhf.scf.grad_conv = 1e-10;
        rcfg.krhf.scf.max_iter = 200;
        rcfg.grid = grid_cfg();
        let r = solve_krks(cell, &fx.prep, None, &fx.mesh, &rcfg).unwrap();
        let eri = KDenseAftEri::build(
            cell,
            &fx.prep,
            &fx.mesh,
            &fx.hk.s,
            ExxDiv::Ewald,
            &Default::default(),
        )
        .unwrap();
        let kr: KRksGradient = kpoint_rks_gradient(
            cell,
            &fx.prep,
            &fx.mesh,
            &PeriodicHcoreConfig::with_omega(OMEGA),
            &fx.hk,
            KGradJk::Dense(&eri),
            &r.scf,
            f,
            &KRksGradConfig {
                grad: kcfg(),
                grid: KXcGrid::Atom(grid_cfg()),
                exxdiv: Some(ExxDiv::Ewald),
                ..Default::default()
            },
        )
        .unwrap();
        let (p, q) = (&u.hf.parts, &kr.hf.parts);
        let rows = [
            ("xc_ao", parts_diff(&u.xc_ao, &kr.xc_ao)),
            ("xc_point", parts_diff(&u.xc_point, &kr.xc_point)),
            ("xc_weight", parts_diff(&u.xc_weight, &kr.xc_weight)),
            ("overlap", parts_diff(&p.overlap, &q.overlap)),
            ("kinetic", parts_diff(&p.kinetic, &q.kinetic)),
            ("coulomb", parts_diff(&p.coulomb, &q.coulomb)),
            ("exchange", parts_diff(&p.exchange, &q.exchange)),
            ("total", parts_diff(&u.grad, &kr.grad)),
        ];
        for (n, d) in rows {
            eprintln!("{f} closed-shell UKS vs RKS: {n:<10} {d:.2e}");
            // Worst measured 6.5e-15; the spin-factor mutants miss by >= 1e-2.
            assert!(d < 1e-11, "{f} {n}: {d:e}");
        }
        assert!(max_abs(&u.xc_ao) > 1e-3 && max_abs(&u.xc_weight) > 1e-3);
    }
}

// ------------------------------------------------------------------- Gamma

fn gamma_uks_side(cell: &Cell, grid: &PeriodicGridConfig, f: &str) -> GammaGradient {
    let prep = prep_for(cell, &pyscf_sto3g_h());
    let hcfg = PeriodicHcoreConfig::with_omega(OMEGA);
    let hc = periodic_hcore(cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        cell,
        &prep,
        &hc.s,
        ExxDiv::Ewald,
        ANCHOR_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let mut gc = GammaUksConfig::new(f);
    gc.grid = grid.clone();
    gc.exxdiv = ExxDiv::Ewald;
    gc.scf.energy_conv = 1e-13;
    gc.scf.density_conv = 1e-10;
    gc.scf.max_iter = 300;
    let r = gamma_uks(cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &gc).unwrap();
    assert!(r.scf.converged);
    gamma_uks_gradient_with(
        cell,
        &prep,
        &hcfg,
        &hc,
        &eri,
        &r.scf,
        &gc,
        &GammaGradConfig {
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
    )
    .unwrap()
}

/// (c) 1x1x1 == the Gamma UKS gradient, per part, on the open-shell H3 doublet
/// (real lattice sums vs complex Bloch AOs; separate SCFs). Sees the spin
/// channels (Da != Db) that the closed-shell limit cannot. With `mutants`, the
/// same Gamma reference must also reject a spin swap in the XC force
/// (`XcSpinSwap`), a swapped exchange in the Pulay Fock (`PulayKSwap`) and a
/// twice-applied exchange fraction (`ExchScaledTwice`): the anchor is the
/// default-set guard against spin and hybrid-scaling defects.
fn one_point_vs_gamma(f: &str, mutants: bool) {
    let gam = gamma_uks_side(&cell_of(&H3, 2), &grid_cfg(), f);
    let fx = fixture(&H3, [1, 1, 1], f, Jk::Dense, ExxDiv::Ewald);
    let k = force(&fx);
    let (p, kp) = (&gam.parts, &k.hf.parts);
    let rows = [
        ("xc_ao", parts_diff(&k.xc_ao, &p.xc_ao)),
        ("xc_point", parts_diff(&k.xc_point, &p.xc_point)),
        ("xc_weight", parts_diff(&k.xc_weight, &p.xc_weight)),
        ("overlap", parts_diff(&kp.overlap, &p.overlap)),
        ("kinetic", parts_diff(&kp.kinetic, &p.kinetic)),
        ("eri=J+K", parts_diff(&(&kp.coulomb + &kp.exchange), &p.eri)),
        ("total", parts_diff(&k.grad, &gam.grad)),
    ];
    for (n, d) in rows {
        eprintln!("{f} H3 1x1x1 vs Gamma UKS: {n:<10} {d:.2e}");
        // Worst measured 1.2e-11.
        assert!(d < 1e-9, "{f} {n}: {d:e}");
    }
    assert!(max_abs(&k.xc_ao) > 1e-3 && max_abs(&k.xc_weight) > 1e-3);
    if mutants {
        for (name, xm, sm, gm) in [
            ("XcSpinSwap", None, Some(KUSpinMutation::XcSpinSwap), None),
            ("PulayKSwap", None, Some(KUSpinMutation::PulayKSwap), None),
            (
                "ExchScaledTwice",
                None,
                None,
                Some(KGradMutation::ExchScaledTwice),
            ),
        ] {
            let d = parts_diff(&force_mut(&fx, xm, sm, gm).grad, &gam.grad);
            eprintln!("{f} H3 1x1x1 vs Gamma UKS under {name}: miss {d:.2e}");
            assert!(d > 1e-6, "{name} survives the Gamma anchor: {d:e}");
        }
    }
}

#[test]
fn one_point_mesh_matches_the_gamma_uks_gradient_per_part() {
    one_point_vs_gamma("PBE0", true);
}

#[test]
#[ignore = "Gamma UKS 1x1x1 anchor for PBE; measured ~20 s (box load 15-60: inflated)"]
fn one_point_mesh_matches_the_gamma_uks_gradient_per_part_pbe() {
    one_point_vs_gamma("PBE", false);
}

fn supercell_pos(pos: &[[f64; 3]], a: f64, n: [usize; 3], mult: usize) -> Cell {
    let mut atoms = Vec::new();
    for m0 in 0..n[0] {
        for m1 in 0..n[1] {
            for m2 in 0..n[2] {
                for p in pos {
                    atoms.push([
                        p[0] + m0 as f64 * a,
                        p[1] + m1 as f64 * a,
                        p[2] + m2 as f64 * a,
                    ]);
                }
            }
        }
    }
    let lat = [
        [a * n[0] as f64, 0.0, 0.0],
        [0.0, a * n[1] as f64, 0.0],
        [0.0, 0.0, a * n[2] as f64],
    ];
    let mut mol = hydrogens(&atoms);
    mol.multiplicity = mult;
    Cell::new(mol, lat).unwrap()
}

/// (d) SUPERCELL ANCHOR on the non-TRIM 1x1x3 mesh, open shell: the k force on
/// atom A is the Gamma UKS force of the explicit supercell (giant-determinant
/// multiplicity (Na - Nb) N_k + 1 = 4) on ONE copy of A. Carries the Bloch
/// phases and the mesh Madelung constant in the scaled M-term; same explicit
/// neighbour cutoff on both sides.
#[test]
#[ignore = "9-atom Gamma UKS supercell; measured 53 s (box load 15-60: inflated)"]
fn k_mesh_uks_force_is_the_gamma_supercell_force_per_part() {
    let f = "PBE0";
    let gam = gamma_uks_side(&supercell_pos(&H3, 4.0, [1, 1, 3], 4), &grid_cfg(), f);
    let fx = fixture(&H3, [1, 1, 3], f, Jk::Dense, ExxDiv::Ewald);
    let k = force(&fx);
    let first = |a: &Array2<f64>| a.slice(ndarray::s![0..3, ..]).to_owned();
    let kp = &k.hf.parts;
    let rows = [
        ("xc_ao", parts_diff(&k.xc_ao, &first(&gam.parts.xc_ao))),
        (
            "xc_weight",
            parts_diff(&k.xc_weight, &first(&gam.parts.xc_weight)),
        ),
        (
            "overlap",
            parts_diff(&kp.overlap, &first(&gam.parts.overlap)),
        ),
        (
            "eri=J+K",
            parts_diff(&(&kp.coulomb + &kp.exchange), &first(&gam.parts.eri)),
        ),
        ("total", parts_diff(&k.grad, &first(&gam.grad))),
    ];
    for (n, d) in rows {
        eprintln!("{f} H3 1x1x3 vs supercell: {n:<10} {d:.2e}");
        // Worst measured 1.6e-11.
        assert!(d < 1e-9, "{f} {n}: {d:e}");
    }
}

// ----------------------------------------------------------------------- FD

/// (e) Central FD of the open-shell energy (fresh SCF per displacement) at
/// h = 1e-4 with the shared explicit cutoff.
#[test]
#[ignore = "FD of 3 open-shell energies, 12 fresh SCFs; measured ~60 s (box load 15-60: inflated)"]
fn uks_force_matches_fd() {
    for (f, jk, exxdiv) in [
        ("PBE", Jk::Dense, ExxDiv::None),
        ("PBE0", Jk::Dense, ExxDiv::Ewald),
        ("PBE0", Jk::Gdf, ExxDiv::Ewald),
    ] {
        let fx = fixture(&H3, [1, 1, 3], f, jk, exxdiv);
        let g = force(&fx);
        let mut worst = 0.0_f64;
        for (a, x) in [(0, 2), (2, 1)] {
            let d = fd(&H3, a, x, FD_H, [1, 1, 3], f, jk, exxdiv);
            worst = worst.max((g.grad[(a, x)] - d).abs());
        }
        eprintln!("{f} {jk:?} {exxdiv:?}: UKS analytic vs FD worst {worst:.2e}");
        assert!(worst < FD_BAR, "{f} {jk:?} {exxdiv:?}: {worst:e}");
    }
}

#[test]
#[ignore = "translation invariance, PBE H3 1x1x3; measured 15 s (box load 15-60: inflated)"]
fn net_force_vanishes() {
    let fx = fixture(&H3, [1, 1, 3], "PBE", Jk::Dense, ExxDiv::None);
    let g = force(&fx);
    let net: f64 = (0..3)
        .map(|x| g.grad.column(x).sum().abs())
        .fold(0.0, f64::max);
    eprintln!("net force {net:.2e}");
    assert!(net < 1e-6, "{net:e}");
}

#[test]
#[ignore = "uniform-grid FD; measured 44 s (box load 15-60: inflated)"]
fn uniform_grid_force_has_no_grid_terms_and_matches_fd() {
    let n = [24, 24, 24];
    let pos = &H3;
    let run = |p: &[[f64; 3]]| {
        let cell = cell_of(p, 2);
        let prep = prep_for(&cell, &pyscf_sto3g_h());
        let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 3]).unwrap();
        let cfg = kuks_cfg(&cell, "PBE", Jk::Dense, ExxDiv::None);
        let grid = PeriodicGrid::uniform(&cell, n).unwrap();
        let r = solve_kuks_on_grid(&cell, &prep, None, &mesh, &grid, &cfg).unwrap();
        (cell, prep, mesh, r)
    };
    let (cell, prep, mesh, r) = run(pos);
    let hk =
        periodic_hcore_kpts(&cell, &prep, &mesh, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let eri = KDenseAftEri::build(
        &cell,
        &prep,
        &mesh,
        &hk.s,
        ExxDiv::None,
        &Default::default(),
    )
    .unwrap();
    let mut cfg = gcfg(None, None, None, None);
    cfg.base.grid = KXcGrid::Uniform(n);
    let g = kpoint_uks_gradient(
        &cell,
        &prep,
        &mesh,
        &PeriodicHcoreConfig::with_omega(OMEGA),
        &hk,
        KGradJk::Dense(&eri),
        &r.scf,
        "PBE",
        &cfg,
    )
    .unwrap();
    assert_eq!(max_abs(&g.xc_point), 0.0);
    assert_eq!(max_abs(&g.xc_weight), 0.0);
    let mut worst = 0.0_f64;
    for (a, x) in [(0, 2), (2, 1)] {
        let e = |p: &[[f64; 3]]| run(p).3.scf.energy;
        let h = FD_H;
        let d = (e(&moved(pos, a, x, h)) - e(&moved(pos, a, x, -h))) / (2.0 * h);
        worst = worst.max((g.grad[(a, x)] - d).abs());
    }
    eprintln!("uniform-grid UKS analytic vs FD worst {worst:.2e}");
    assert!(worst < FD_BAR, "{worst:e}");
}

// -------------------------------------------------------- bitwise / guard

#[test]
#[ignore = "4 force runs; measured 107 s (box load 15-60: inflated)"]
fn uks_force_is_bit_identical_across_threads() {
    for jk in [Jk::Dense, Jk::Gdf] {
        let fx = fixture(&H3, [1, 1, 3], "PBE0", jk, ExxDiv::Ewald);
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| force(&fx))
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.grad, b.grad, "{jk:?}");
        assert_eq!(a.xc_ao, b.xc_ao, "{jk:?}");
    }
}

#[test]
fn refusals_and_the_per_spin_fock_guard() {
    let fx = fixture(&H3, [1, 1, 1], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let ok_cfg = gcfg(Some(ExxDiv::Ewald), None, None, None);
    let run = |f: &str, s: &KUScfResult, cfg: &KUksGradConfig| force_with(&fx, f, s, cfg);
    // Typed refusals, entry-point specific.
    for (name, key) in [
        // HSE06 is covered on kernels built with its range separation
        // (`pbc_kuks_rsh_force.rs`); these PBE0 kernels are bare: refused.
        ("HSE06", "bare Coulomb exchange"),
        ("wB97X-V", "VV10"),
        // SCAN is covered (`pbc_kuxc_mgga.rs`): a PBE0 state handed in as
        // SCAN is refused by the Fock guard.
        ("SCAN", "differs from the SCF's stored fock_alpha"),
    ] {
        let e = run(name, &fx.res.scf, &ok_cfg)
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"))
            .to_string();
        eprintln!("{name}: {e}");
        assert!(e.contains(key), "{name}: {e}");
    }
    let e = run("PBE0", &fx.res.scf, &gcfg(None, None, None, None))
        .expect_err("no exxdiv")
        .to_string();
    assert!(
        e.contains("kpoint_uks_gradient: a hybrid functional"),
        "{e}"
    );
    let ok = run("PBE0", &fx.res.scf, &ok_cfg).unwrap();
    assert!(ok.hf.fock_mismatch < 1e-12, "{:e}", ok.hf.fock_mismatch);
    // Guard from every side.
    let pbe = fixture(&H3, [1, 1, 1], "PBE", Jk::Dense, ExxDiv::Ewald);
    let cases: Vec<(&str, Result<KUksGradient, ferric_core::FerricError>)> = vec![
        (
            "exxdiv None on an Ewald SCF",
            run(
                "PBE0",
                &fx.res.scf,
                &gcfg(Some(ExxDiv::None), None, None, None),
            ),
        ),
        ("B3LYP on a PBE0 SCF", run("B3LYP", &fx.res.scf, &ok_cfg)),
        ("PBE force on a PBE0 SCF", run("PBE", &fx.res.scf, &ok_cfg)),
        (
            "PBE0 force on a PBE SCF",
            run("PBE0", &pbe.res.scf, &ok_cfg),
        ),
    ];
    for (n, r) in cases {
        let e = r
            .err()
            .unwrap_or_else(|| panic!("{n} accepted"))
            .to_string();
        eprintln!("{n}: {e}");
        assert!(
            e.contains("kpoint_uks_gradient: the Fock matrix rebuilt"),
            "{n}: {e}"
        );
    }
    // Tampers of each spin's Fock are refused, naming the spin.
    let mut t = fx.res.scf.clone();
    t.fock_alpha[0][(0, 1)] += Complex64::new(1e-6, 0.0);
    let e = run("PBE0", &t, &ok_cfg)
        .expect_err("alpha tamper")
        .to_string();
    assert!(e.contains("fock_alpha"), "{e}");
    let mut t = fx.res.scf.clone();
    t.fock_beta[0][(0, 1)] += Complex64::new(1e-6, 0.0);
    let e = run("PBE0", &t, &ok_cfg)
        .expect_err("beta tamper")
        .to_string();
    assert!(e.contains("fock_beta"), "{e}");
    // Swapped spin Focks.
    let mut t = fx.res.scf.clone();
    std::mem::swap(&mut t.fock_alpha, &mut t.fock_beta);
    let e = run("PBE0", &t, &ok_cfg)
        .expect_err("swapped spin Focks")
        .to_string();
    assert!(
        e.contains("kpoint_uks_gradient: the Fock matrix rebuilt") && e.contains("fock_alpha"),
        "{e}"
    );
    // A pure functional needs no exxdiv and has no exchange part.
    let a = force_with(&pbe, "PBE", &pbe.res.scf, &gcfg(None, None, None, None)).unwrap();
    assert_eq!(max_abs(&a.hf.parts.exchange), 0.0);
}

/// Every budget either reproduces the unconstrained force bit-identically or
/// fails typed. The Fock-guard charge (10 N_k n^2 complex) is NOT tripped by
/// this sweep on a cell this small (the XC chunk gate trips first): the charge
/// is covered by the sizing arithmetic only (stated, not claimed).
#[test]
#[ignore = "budget sweep; measured 140 s (box load 15-60: inflated)"]
fn budgets_either_reproduce_the_force_or_fail_typed() {
    let fx = fixture(&H3, [1, 1, 3], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let full = force(&fx);
    let (mut ok, mut err) = (0, 0);
    let mut b = 20_000usize;
    while b <= AMPLE {
        let mut cfg = gcfg(Some(ExxDiv::Ewald), None, None, None);
        cfg.base.grad.budget_bytes = Some(b);
        match force_with(&fx, &fx.functional, &fx.res.scf, &cfg) {
            Ok(g) => {
                ok += 1;
                assert_eq!(g.grad, full.grad, "budget {b}");
            }
            Err(e) => {
                err += 1;
                assert!(
                    e.to_string().contains("budget") || e.to_string().contains("memory"),
                    "{b}: {e}"
                );
            }
        }
        b *= 3;
    }
    eprintln!("budget sweep: {ok} runs identical, {err} typed refusals");
    assert!(ok > 0 && err > 0);
}

// ------------------------------------------------------------------ mutants

const MUTANT_BAR: f64 = 1e-5;

fn miss(
    fx: &Fx,
    xm: Option<KXcMutation>,
    sm: Option<KUSpinMutation>,
    gm: Option<KGradMutation>,
    base: &KUksGradient,
) -> f64 {
    parts_diff(&force_mut(fx, xm, sm, gm).grad, &base.grad)
}

/// Production-wired spin-specific and shared defects against the FD-anchored
/// force on the open-shell H3 cell. Each must miss by far more than the FD bar;
/// the closed-shell limit is blind to the spin ones (stated in (b)). The
/// control `GuardSkippedOnly` must be exactly 0.
#[test]
#[ignore = "25 production-wired mutants on H3 PBE0; measured 598 s (box load 15-60: inflated)"]
fn smoke_mutants_are_caught() {
    let fx = fixture(&H3, [1, 1, 3], "PBE0", Jk::Dense, ExxDiv::Ewald);
    let base = force(&fx);
    assert_eq!(
        miss(&fx, Some(KXcMutation::GuardSkippedOnly), None, None, &base),
        0.0
    );
    for sm in [
        KUSpinMutation::XcSpinSwap,
        KUSpinMutation::XcDropBeta,
        KUSpinMutation::XcCrossDropped,
        KUSpinMutation::XcCrossOwn,
        KUSpinMutation::XcSpinAveraged,
        KUSpinMutation::PulayFockSwap,
        KUSpinMutation::PulayKSwap,
        KUSpinMutation::PulayJOneSpin,
        KUSpinMutation::PulayNoVxcBeta,
    ] {
        let d = miss(&fx, None, Some(sm), None, &base);
        eprintln!("spin mutant {sm:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{sm:?} survives: {d:e}");
    }
    for xm in [
        KXcMutation::NoAo,
        KXcMutation::NoPointMotion,
        KXcMutation::NoWeightDeriv,
        KXcMutation::NoGga,
        KXcMutation::NoHessian,
        KXcMutation::WrongFactor,
        KXcMutation::NoInvNk,
        KXcMutation::NoPhase,
        KXcMutation::PulayNoVxc,
        KXcMutation::WrongRemainder,
    ] {
        let d = miss(&fx, Some(xm), None, None, &base);
        eprintln!("xc mutant {xm:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{xm:?} survives: {d:e}");
    }
    for gm in [
        KGradMutation::ExchDenseUnscaled,
        KGradMutation::ExchMadelungUnscaled,
        KGradMutation::ExchJScaled,
        KGradMutation::ExchScaledTwice,
        KGradMutation::ExchSpinFactor,
    ] {
        let d = miss(&fx, None, None, Some(gm), &base);
        eprintln!("hybrid-scale mutant {gm:?}: miss {d:.2e}");
        assert!(d > MUTANT_BAR, "{gm:?} survives: {d:e}");
    }
}

// ------------------------------------------------- UKS stationarity gate

/// UKS stationarity probe: for SCFs stopped at looser `grad_conv`, the largest
/// `|F D S - S D F|` the force reports against the force error relative to the
/// tightly converged force at the same geometry (gate off). The refusal
/// threshold `UKS_COMMUTATOR_TOL` is read off this table.
#[test]
#[ignore = "probe: 8 open-shell SCFs"]
fn uks_stationarity_probe() {
    let tight = {
        let fx = fixture(&H3, [1, 1, 2], "PBE", Jk::Dense, ExxDiv::None);
        force(&fx).grad
    };
    for gconv in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5] {
        SETTINGS.with(|st| st.set((CUTOFF, 1e-3 * gconv, gconv)));
        let fx = fixture(&H3, [1, 1, 2], "PBE", Jk::Dense, ExxDiv::None);
        let g = force_mut(&fx, None, Some(KUSpinMutation::StationarityGateOff), None);
        eprintln!(
            "grad_conv {gconv:.0e}: commutator {:.3e}, max |dF| vs tight {:.3e}",
            g.hf.commutator,
            parts_diff(&g.grad, &tight)
        );
    }
}

/// A loosely converged UKS state is refused with the commutator message; the
/// same state with the gate off returns a force that is wrong by far more than
/// the FD bar (so the gate matters), and a tight state passes the gate.
#[test]
fn loose_uks_scf_is_refused_by_the_stationarity_gate() {
    let tight = fixture(&H3, [1, 1, 1], "PBE", Jk::Dense, ExxDiv::None);
    let gt = force(&tight);
    assert!(
        gt.hf.commutator < UKS_COMMUTATOR_TOL / 100.0,
        "{:e}",
        gt.hf.commutator
    );
    SETTINGS.with(|st| st.set((CUTOFF, LOOSE_UKS.0, LOOSE_UKS.1)));
    let loose = fixture(&H3, [1, 1, 1], "PBE", Jk::Dense, ExxDiv::None);
    let e = force_with(&loose, "PBE", &loose.res.scf, &gcfg(None, None, None, None))
        .expect_err("loose SCF accepted")
        .to_string();
    eprintln!("loose UKS: {e}");
    assert!(
        e.contains("kpoint_uks_gradient: the spin Fock/density commutator")
            && e.contains("exceeds 1e-5"),
        "{e}"
    );
    let off = force_mut(
        &loose,
        None,
        Some(KUSpinMutation::StationarityGateOff),
        None,
    );
    let miss = parts_diff(&off.grad, &gt.grad);
    eprintln!(
        "loose UKS: commutator {:.3e}, force error with the gate off {miss:.3e}",
        off.hf.commutator
    );
    assert!(off.hf.commutator > UKS_COMMUTATOR_TOL);
    assert!(
        miss > 10.0 * FD_BAR,
        "gate off is not wrong enough: {miss:e}"
    );
}
