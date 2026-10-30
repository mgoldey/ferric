//! Shared fixtures and helpers of `pbc_kuxcgrad.rs` (per commit) and
//! `validation_pbc_kuxcgrad.rs` (weekly): the H2 / H3 k-UKS fixtures, the
//! force drivers, the zero-XC diagnostic entry, the Gamma UKS reference and
//! the anchor bodies.
#![allow(dead_code, unused_imports)]

use crate::common::*;
pub use ferric_core::basis::{BasisSet, Shell};
pub use ferric_pbc::dense_aft::ExxDiv;
pub use ferric_pbc::dense_aft::{DenseAftEri, DEFAULT_DENSE_AFT_MAX_BYTES};
pub use ferric_pbc::dft::{gamma_uks, GammaUksConfig, PeriodicGridConfig};
pub use ferric_pbc::grad::{gamma_uks_gradient_with, GammaGradConfig, GammaGradient};
pub use ferric_pbc::hcore::kpoint::{periodic_hcore_kpts, PeriodicHcoreK};
pub use ferric_pbc::hcore::periodic_hcore;
pub use ferric_pbc::hcore::PeriodicHcoreConfig;
pub use ferric_pbc::kdense_aft::{KDenseAftConfig, KDenseAftEri};
pub use ferric_pbc::kdft::{solve_krks, KRksConfig};
pub use ferric_pbc::kgrad::{
    kpoint_uhf_gradient, KGradConfig, KGradJk, KGradMutation, KRsGdfGradSource,
};
pub use ferric_pbc::kpts::KPointMesh;
pub use ferric_pbc::kscf::{KJkKind, KPointInjection, KPointJk};
pub use ferric_pbc::kuks::{solve_kuks, solve_kuks_on_grid, KUksConfig, KUksResult};
pub use ferric_pbc::kuscf::{solve_kuks_injected, KPointXcPolarized, KUScfResult};
pub use ferric_pbc::kuxcgrad::UKS_COMMUTATOR_TOL;
pub use ferric_pbc::kuxcgrad::{
    kpoint_null_xc_uks_gradient, kpoint_uks_gradient, KUSpinMutation, KUksGradConfig, KUksGradient,
};
pub use ferric_pbc::kxcgrad::{
    kpoint_rks_gradient, KRksGradConfig, KRksGradient, KXcGrid, KXcMutation,
};
pub use ferric_pbc::lattice::Cell;
pub use ferric_pbc::rsgdf::kpoint::{KRsGdf, KRsGdfConfig};
pub use ferric_pbc::rsgdf::RsGdfConfig;
pub use ferric_pbc::uhf::GammaUhfIntegrals;
pub use ferric_pbc::PeriodicGrid;
pub use ndarray::Array2;
pub use num_complex::Complex64;
pub use std::collections::HashMap;

pub const OMEGA: f64 = 0.8;
pub const AMPLE: usize = 1 << 31;
/// Explicit XC neighbour cutoff shared by every energy and force here (the
/// default depends on the cell and makes one-step energy discontinuities).
pub const CUTOFF: f64 = 14.0;
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
pub const FD_BAR: f64 = 1e-8;
pub const FD_H: f64 = 5e-5;
pub const ANCHOR_PRECISION: f64 = 1e-12;
pub const H2: [[f64; 3]; 2] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5]];
/// H3 doublet (na = 2, nb = 1): a genuinely spin-polarised cell, Da != Db.
/// (energy_conv, grad_conv) of the deliberately loose UKS SCF of the gate test.
pub const LOOSE_UKS: (f64, f64) = (1e-5, 1e-2);
pub const H3: [[f64; 3]; 3] = [[0.3, 0.2, 0.1], [0.35, 0.12, 1.5], [0.3, 1.9, 0.4]];

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Jk {
    Dense,
    Gdf,
}

pub fn et_sp_aux() -> BasisSet {
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

pub fn kgdf_cfg() -> KRsGdfConfig {
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
    pub static SETTINGS: std::cell::Cell<(f64, f64, f64)> =
        const { std::cell::Cell::new((CUTOFF, 1e-13, 1e-10)) };
}

pub fn grid_cfg() -> PeriodicGridConfig {
    let mut g = PeriodicGridConfig::with_size(20, 50);
    g.neighbour_cutoff = Some(SETTINGS.with(|s| s.get()).0);
    g
}

pub fn cell_of(pos: &[[f64; 3]], mult: usize) -> Cell {
    let mut mol = hydrogens(pos);
    mol.multiplicity = mult;
    Cell::new(mol, cubic(4.0)).expect("cell")
}

pub fn mult_of(pos: &[[f64; 3]]) -> usize {
    if pos.len() % 2 == 1 {
        2
    } else {
        1
    }
}

pub fn kuks_cfg(cell: &Cell, functional: &str, jk: Jk, exxdiv: ExxDiv) -> KUksConfig {
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

pub fn scf(
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

pub struct Fx {
    pub cell: Cell,
    pub prep: ferric_integrals::basis_bridge::PreparedBasis,
    pub aux: ferric_integrals::basis_bridge::PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
    pub res: KUksResult,
    pub jk: Jk,
    pub exxdiv: ExxDiv,
    pub functional: String,
}

pub fn fixture(
    pos: &[[f64; 3]],
    mesh_n: [usize; 3],
    functional: &str,
    jk: Jk,
    exxdiv: ExxDiv,
) -> Fx {
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

pub fn gcfg(
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

pub fn force_with(
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

pub fn force(fx: &Fx) -> KUksGradient {
    force_with(
        fx,
        &fx.functional,
        &fx.res.scf,
        &gcfg(Some(fx.exxdiv), None, None, None),
    )
    .unwrap()
}

pub fn force_mut(
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

pub fn moved(pos: &[[f64; 3]], a: usize, x: usize, h: f64) -> Vec<[f64; 3]> {
    let mut p = pos.to_vec();
    p[a][x] += h;
    p
}

#[allow(clippy::too_many_arguments)]
pub fn fd(
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

pub fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

pub fn parts_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

pub fn kcfg() -> KGradConfig {
    KGradConfig {
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

// ---------------------------------------------------- zero-XC fixed SCF

pub struct ZeroXcPol(usize);
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

pub struct Zx {
    pub cell: Cell,
    pub prep: ferric_integrals::basis_bridge::PreparedBasis,
    pub aux: ferric_integrals::basis_bridge::PreparedBasis,
    pub mesh: KPointMesh,
    pub hk: PeriodicHcoreK,
    pub scf: KUScfResult,
    pub jk: Jk,
    pub exxdiv: ExxDiv,
}

pub fn dense_cfg() -> KDenseAftConfig {
    KDenseAftConfig {
        precision: ANCHOR_PRECISION,
        ..Default::default()
    }
}

/// Hartree + `a` exact-exchange k UKS with a zero XC (a = 1 is k-point UHF).
pub fn zero_xc(pos: &[[f64; 3]], mesh_n: [usize; 3], jk: Jk, exxdiv: ExxDiv, a: f64) -> Zx {
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

pub fn with_jk<R>(z: &Zx, f: impl FnOnce(KGradJk<'_>) -> R) -> R {
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

pub fn closed_shell_vs_rks(functionals: &[&str]) {
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

pub fn gamma_uks_side(cell: &Cell, grid: &PeriodicGridConfig, f: &str) -> GammaGradient {
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
/// channels (Da != Db) that the closed-shell limit cannot. Each mutant named
/// in `mutants` (any of `"XcSpinSwap"`: a spin swap in the XC force,
/// `"PulayKSwap"`: a swapped exchange in the Pulay Fock, `"ExchScaledTwice"`:
/// a twice-applied exchange fraction) must then miss the same Gamma reference
/// by more than 1e-6. BLIND to the Bloch phases (one k-point, all real).
pub fn one_point_vs_gamma(f: &str, mutants: &[&str]) {
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
    let table = [
        ("XcSpinSwap", None, Some(KUSpinMutation::XcSpinSwap), None),
        ("PulayKSwap", None, Some(KUSpinMutation::PulayKSwap), None),
        (
            "ExchScaledTwice",
            None,
            None,
            Some(KGradMutation::ExchScaledTwice),
        ),
    ];
    for &want in mutants {
        let &(name, xm, sm, gm) = table
            .iter()
            .find(|t| t.0 == want)
            .unwrap_or_else(|| panic!("unknown mutant {want}"));
        let d = parts_diff(&force_mut(&fx, xm, sm, gm).grad, &gam.grad);
        eprintln!("{f} H3 1x1x1 vs Gamma UKS under {name}: miss {d:.2e}");
        assert!(d > 1e-6, "{name} survives the Gamma anchor: {d:e}");
    }
}

pub fn supercell_pos(pos: &[[f64; 3]], a: f64, n: [usize; 3], mult: usize) -> Cell {
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

// ------------------------------------------------------------------ mutants

pub const MUTANT_BAR: f64 = 1e-5;

pub fn miss(
    fx: &Fx,
    xm: Option<KXcMutation>,
    sm: Option<KUSpinMutation>,
    gm: Option<KGradMutation>,
    base: &KUksGradient,
) -> f64 {
    parts_diff(&force_mut(fx, xm, sm, gm).grad, &base.grad)
}
