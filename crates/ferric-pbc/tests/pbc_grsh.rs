//! Range-separated hybrids (HSE06, CAM-B3LYP) at the Gamma point,
//! `ferric_pbc::dft::gamma_rks_rsh` (mgoldey/ferric#354).
//!
//! The attenuated K is the k-point code at the single `k = 0` of the 1x1x1
//! mesh ([`ferric_pbc::gamma_rsh`]), wrapped as a real `KBuilder`; J, hcore,
//! the semilocal XC and the SCF are the unchanged Gamma code. So "Gamma ==
//! 1x1x1 mesh" checks the glue and the independent Gamma J/XC/SCF, NOT the
//! attenuated-exchange kernel itself. The independent oracle for the kernel
//! is PySCF's Gamma `pbc.dft.RKS` + AFTDF.
//!
//! # Hypotheses (written before any number was measured)
//!
//! * Physics: Gamma energy == PySCF Gamma (and the PySCF k-point 1x1x1 pins of
//!   `pbc_krsh_more.rs`, which PySCF reproduces at Gamma) to the grid/lattice
//!   precision; RS-GDF == dense to the plain-Coulomb fit error.
//! * Artifact: a Gamma wrapper that dropped the attenuated K, mis-scaled the
//!   fractions, took omega in the wrong unit, swapped c_sr / c_lr, or applied
//!   the wrong G = 0 constant would STILL equal the 1x1x1 mesh (it calls the
//!   same kernel), so only the PySCF pins and the mutant shifts below can see
//!   it. Hence the pins are the load-bearing anchor and every mutant must
//!   move the energy far above the pin bar.
//!
//! Provenance of the pins: `testdata/reference/pbc_h2_sto3g_gamma_rsh.json`
//! (`scripts/gen_pyscf_pbc_grsh_refs.py`): PySCF 2.13.1, libxc 7.0.0, AFTDF,
//! UniformGrids 41^3, `cell.precision` 1e-11, conv_tol 1e-12. The triclinic
//! rows equal the k-point 1x1x1 pins of `pbc_krsh_more.rs` digit for digit
//! (PySCF Gamma == its own 1x1x1 mesh).

mod common;

use common::*;
use ferric_core::basis;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{gamma_rks_rsh_on_grid, GammaRksConfig, PeriodicGrid};
use ferric_pbc::gamma_rsh::GammaRshExchange;
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcoreConfig};
use ferric_pbc::kdense_aft::KDenseAftConfig;
use ferric_pbc::kdft::{solve_krks_on_grid, KRksConfig};
use ferric_pbc::kpts::KPointMesh;
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::kpoint::KRsGdfConfig;
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
use ferric_pbc::rsh::{resolve_periodic_functional_rsh, RshParams};
use ferric_pbc::uhf::GammaUhfIntegrals;

const OMEGA: f64 = 0.8;
const MESH: usize = 41;
const AMPLE: usize = 1 << 30;
const PIN_TOL: f64 = 1e-10;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Lat {
    Cubic4,
    Triclinic,
}

fn cell_of(l: Lat) -> Cell {
    match l {
        Lat::Cubic4 => h2_cell(4.0),
        Lat::Triclinic => Cell::new(hydrogens(&H2_ATOMS), TRI_A).expect("triclinic H2 cell"),
    }
}

fn name_of(f: &str) -> &'static str {
    match f {
        "HSE06" => "HSE06",
        _ => "HYB_GGA_XC_CAM_B3LYP",
    }
}

/// PySCF Gamma `pbc.dft.RKS` + AFTDF energies: (cell, functional, exxdiv, E).
const PINS: [(Lat, &str, ExxDiv, f64); 8] = [
    (Lat::Cubic4, "HSE06", ExxDiv::Ewald, -1.575428343655),
    (Lat::Cubic4, "HSE06", ExxDiv::None, -1.429127678283),
    (Lat::Cubic4, "CAM_B3LYP", ExxDiv::Ewald, -1.591829863233),
    (Lat::Cubic4, "CAM_B3LYP", ExxDiv::None, -1.286534460571),
    (Lat::Triclinic, "HSE06", ExxDiv::Ewald, -1.425296483200),
    (Lat::Triclinic, "HSE06", ExxDiv::None, -1.300717690549),
    (Lat::Triclinic, "CAM_B3LYP", ExxDiv::Ewald, -1.447144967796),
    (Lat::Triclinic, "CAM_B3LYP", ExxDiv::None, -1.160109365853),
];

/// How the Gamma run is set up (which J/K source, which mutation).
#[derive(Clone, Copy, Default)]
struct Run<'a> {
    /// RS-GDF aux basis (J and the K fit); `None` = dense AFT.
    aux: Option<&'a str>,
    /// Replace the range-separation parameters by these (a mutant).
    rsh: Option<RshParams>,
    /// Build the exchange with this exxdiv instead of the SCF's (a mutant).
    k_exxdiv: Option<ExxDiv>,
}

fn rsh_of(f: &str) -> RshParams {
    resolve_periodic_functional_rsh(name_of(f))
        .unwrap()
        .2
        .unwrap()
}

fn gamma_energy(l: Lat, f: &str, exxdiv: ExxDiv, run: Run<'_>) -> f64 {
    let cell = cell_of(l);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let grid = PeriodicGrid::uniform(&cell, [MESH; 3]).unwrap();
    let rsh = run.rsh.unwrap_or_else(|| rsh_of(f));
    let kx = run.k_exxdiv.unwrap_or(exxdiv);
    let mutated = run.rsh.is_some() || run.k_exxdiv.is_some();
    let aux = run
        .aux
        .map(|n| PreparedBasis::new(cell.mol(), &basis::bundled(n).unwrap()).unwrap());
    let mut cfg = GammaRksConfig::new(name_of(f));
    cfg.exxdiv = exxdiv;
    cfg.scf.density_conv = 1e-11;
    cfg.scf.energy_conv = 1e-12;
    let (ex, dense_j, gdf_j);
    match &aux {
        None => {
            let kcfg = KDenseAftConfig::default();
            ex = GammaRshExchange::build_dense(&cell, &prep, &hc.s, kx, &kcfg, rsh).unwrap();
            dense_j = Some(
                DenseAftEri::build(
                    &cell,
                    &prep,
                    &hc.s,
                    exxdiv,
                    DEFAULT_DENSE_AFT_PRECISION,
                    DEFAULT_DENSE_AFT_MAX_BYTES,
                )
                .unwrap(),
            );
            gdf_j = None;
        }
        Some(a) => {
            let gcfg = RsGdfConfig {
                omega: OMEGA,
                exxdiv,
                budget_bytes: Some(AMPLE),
                ..Default::default()
            };
            let kcfg = KRsGdfConfig {
                gdf: RsGdfConfig { exxdiv: kx, ..gcfg },
                mutation: None,
            };
            ex = GammaRshExchange::build_rsgdf(&cell, &prep, a, &hc.s, kx, &kcfg, rsh).unwrap();
            gdf_j = Some(RsGdf::build(&cell, &prep, a, &hc.s, &gcfg).unwrap());
            dense_j = None;
        }
    }
    let ex = if mutated {
        ex.unchecked_for_mutation_tests()
    } else {
        ex
    };
    let ints = match (&dense_j, &gdf_j) {
        (Some(e), _) => GammaUhfIntegrals::DenseAft(e),
        (_, Some(g)) => GammaUhfIntegrals::RsGdf(g),
        _ => unreachable!(),
    };
    let r = gamma_rks_rsh_on_grid(&cell, &prep, &hc, ints, &ex, &grid, &cfg)
        .unwrap_or_else(|e| panic!("gamma_rks_rsh {l:?} {f} {exxdiv:?}: {e}"));
    assert!(r.scf.converged);
    r.scf.energy
}

fn kpoint_energy_1x1x1(l: Lat, f: &str, exxdiv: ExxDiv, aux: Option<&str>) -> f64 {
    let cell = cell_of(l);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let grid = PeriodicGrid::uniform(&cell, [MESH; 3]).unwrap();
    let mesh = KPointMesh::gamma_centred(&cell, [1, 1, 1]).unwrap();
    let mut cfg = KRksConfig::new(&cell, exxdiv, name_of(f));
    cfg.krhf.hcore = PeriodicHcoreConfig::with_omega(OMEGA);
    cfg.krhf.scf.energy_conv = 1e-12;
    cfg.krhf.scf.grad_conv = 1e-9;
    let auxp = aux.map(|n| PreparedBasis::new(cell.mol(), &basis::bundled(n).unwrap()).unwrap());
    if auxp.is_some() {
        cfg.krhf.jk = ferric_pbc::kscf::KJkKind::RsGdf;
        cfg.krhf.rsgdf = KRsGdfConfig {
            gdf: RsGdfConfig {
                omega: OMEGA,
                exxdiv,
                budget_bytes: Some(AMPLE),
                ..Default::default()
            },
            mutation: None,
        };
    }
    solve_krks_on_grid(&cell, &prep, auxp.as_ref(), &mesh, &grid, &cfg)
        .unwrap()
        .scf
        .energy
}

#[test]
fn gamma_rsh_matches_pinned_pyscf() {
    for (l, f, x, e_ref) in PINS {
        let e = gamma_energy(l, f, x, Run::default());
        eprintln!(
            "{l:?} {f} {x:?}: E {e:.12} pyscf {e_ref:.12} dE {:.2e}",
            e - e_ref
        );
        assert!((e - e_ref).abs() < PIN_TOL, "{l:?} {f} {x:?}");
    }
}

#[test]
fn gamma_rsh_is_the_one_by_one_by_one_mesh() {
    for (l, f, x) in [
        (Lat::Triclinic, "HSE06", ExxDiv::Ewald),
        (Lat::Triclinic, "CAM_B3LYP", ExxDiv::None),
        (Lat::Cubic4, "CAM_B3LYP", ExxDiv::Ewald),
    ] {
        for aux in [None, Some("cc-pvdz-ri")] {
            let g = gamma_energy(
                l,
                f,
                x,
                Run {
                    aux,
                    ..Run::default()
                },
            );
            let k = kpoint_energy_1x1x1(l, f, x, aux);
            eprintln!("{l:?} {f} {x:?} aux {aux:?}: Gamma - 1x1x1 = {:.2e}", g - k);
            assert!(
                (g - k).abs() < PIN_TOL,
                "{l:?} {f} {x:?} {aux:?}: {:e}",
                g - k
            );
        }
    }
}

/// RS-GDF Gamma RSH minus dense Gamma RSH must be the plain-Coulomb (PBE0)
/// RS-GDF fit error of the same aux basis, to a fraction of it: the attenuated
/// fit adds no error beyond the full-Coulomb fit's order.
#[test]
fn gamma_rsgdf_rsh_is_dense_to_the_fit_error() {
    // Plain-Coulomb scale: PBE0 through the unchanged gamma_rks.
    let l = Lat::Cubic4;
    let aux = "cc-pvdz-ri";
    let cell = cell_of(l);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let a = PreparedBasis::new(cell.mol(), &basis::bundled(aux).unwrap()).unwrap();
    let dense = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::Ewald,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let gdf = RsGdf::build(
        &cell,
        &prep,
        &a,
        &hc.s,
        &RsGdfConfig {
            omega: OMEGA,
            exxdiv: ExxDiv::Ewald,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
    )
    .unwrap();
    let mut cfg = GammaRksConfig::new("PBE0");
    cfg.scf.density_conv = 1e-11;
    cfg.scf.energy_conv = 1e-12;
    let plain = |ints| {
        ferric_pbc::dft::gamma_rks(&cell, &prep, &hc, ints, &cfg)
            .unwrap()
            .scf
            .energy
    };
    let scale = plain(GammaUhfIntegrals::RsGdf(&gdf)) - plain(GammaUhfIntegrals::DenseAft(&dense));
    assert!(
        scale.abs() > 1e-7,
        "fit error scale not measurable: {scale:e}"
    );
    for f in ["HSE06", "CAM_B3LYP"] {
        let d = gamma_energy(
            l,
            f,
            ExxDiv::Ewald,
            Run {
                aux: Some(aux),
                ..Run::default()
            },
        ) - gamma_energy(l, f, ExxDiv::Ewald, Run::default());
        eprintln!(
            "{f}: RS-GDF - dense {d:.3e} (plain PBE0 fit error {scale:.3e}, ratio {:.3})",
            d / scale
        );
        assert!(
            (d - scale).abs() < 0.4 * scale.abs(),
            "{f}: {d:e} vs {scale:e}"
        );
    }
}

/// Every mutant must move the energy far above the pin bar (>= 1e6 x).
#[test]
fn gamma_rsh_pins_see_the_defects() {
    let (l, x) = (Lat::Triclinic, ExxDiv::Ewald);
    for f in ["HSE06", "CAM_B3LYP"] {
        let base = gamma_energy(l, f, x, Run::default());
        let r = rsh_of(f);
        let mutants = [
            (
                "omega x1.01",
                RshParams {
                    omega: r.omega * 1.01,
                    ..r
                },
            ),
            ("attenuated K dropped", RshParams { c_lr: r.c_sr, ..r }),
            (
                "c_sr/c_lr swapped",
                RshParams {
                    c_sr: r.c_lr,
                    c_lr: r.c_sr,
                    ..r
                },
            ),
            (
                "c_sr x0.99",
                RshParams {
                    c_sr: r.c_sr * 0.99,
                    ..r
                },
            ),
        ];
        for (name, m) in mutants {
            let e = gamma_energy(
                l,
                f,
                x,
                Run {
                    rsh: Some(m),
                    ..Run::default()
                },
            );
            eprintln!("{f} {name}: dE {:.3e}", e - base);
            assert!((e - base).abs() > 1e-4, "{f} {name} left the pins green");
        }
        // Wrong G = 0 handling: ewald SCF with the exxdiv = none exchange.
        let e = gamma_energy(
            l,
            f,
            x,
            Run {
                k_exxdiv: Some(ExxDiv::None),
                ..Run::default()
            },
        );
        eprintln!("{f} wrong exxdiv: dE {:.3e}", e - base);
        assert!(
            (e - base).abs() > 1e-4,
            "{f} wrong exxdiv left the pins green"
        );
    }
}

/// The checks guarding `gamma_rks_rsh` are typed errors, not silent mixing.
#[test]
fn gamma_rsh_refusals_are_typed() {
    let cell = cell_of(Lat::Triclinic);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let grid = PeriodicGrid::uniform(&cell, [9; 3]).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::Ewald,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let build = |x| {
        GammaRshExchange::build_dense(
            &cell,
            &prep,
            &hc.s,
            x,
            &KDenseAftConfig::default(),
            rsh_of("HSE06"),
        )
        .unwrap()
    };
    let go = |f: &str, ex: &GammaRshExchange, x: ExxDiv| {
        let mut cfg = GammaRksConfig::new(f);
        cfg.exxdiv = x;
        gamma_rks_rsh_on_grid(
            &cell,
            &prep,
            &hc,
            GammaUhfIntegrals::DenseAft(&eri),
            ex,
            &grid,
            &cfg,
        )
        .map(|_| ())
    };
    // Plain gamma_rks refuses the range-separated functional by name.
    let cfg = GammaRksConfig::new("HSE06");
    let e = ferric_pbc::dft::gamma_rks(&cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &cfg)
        .unwrap_err();
    assert!(format!("{e}").contains("range-separated"), "{e}");
    // A global hybrid through the RSH entry; wrong parameters; wrong exxdiv.
    assert!(go("PBE0", &build(ExxDiv::Ewald), ExxDiv::Ewald).is_err());
    assert!(go("HYB_GGA_XC_CAM_B3LYP", &build(ExxDiv::Ewald), ExxDiv::Ewald).is_err());
    assert!(go("HSE06", &build(ExxDiv::None), ExxDiv::Ewald).is_err());
}

/// `range_split` would shape only the Gamma J, so with an RSH functional it is
/// a typed refusal (not a silent partial behaviour); a global hybrid is fine.
#[test]
fn gamma_rsh_refuses_range_split() {
    use ferric_pbc::gamma_rsh::{gamma_rks_any, GammaRshSizing};
    let cell = cell_of(Lat::Triclinic);
    let bs = pyscf_sto3g_h();
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(OMEGA)).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::Ewald,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let sizing = GammaRshSizing {
        rsgdf_omega: OMEGA,
        budget_bytes: None,
        max_eri_bytes: DEFAULT_DENSE_AFT_MAX_BYTES,
        range_split: true,
    };
    let go = |f: &str| {
        let cfg = GammaRksConfig::new(f);
        gamma_rks_any(
            &cell,
            &prep,
            &hc,
            GammaUhfIntegrals::DenseAft(&eri),
            None,
            sizing,
            &cfg,
        )
        .map(|_| ())
    };
    let e = go("HSE06").unwrap_err();
    assert!(format!("{e}").contains("range_split"), "{e}");
}
