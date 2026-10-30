//! Shared harness of `pbc_lr_force_bitwise.rs` (per-commit) and
//! `validation_pbc_lr_force_bitwise.rs` (weekly): the water cases, their
//! converged RS-GDF fixtures, the force and the thread-invariance check.
#![allow(dead_code)]

use crate::common::*;
use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_pbc::dense_aft::ExxDiv;
use ferric_pbc::grad::{gamma_rhf_gradient_rsgdf, GammaGradConfig, GammaGradient, RsGdfGradSource};
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::{RangeSplit, RsGdf, RsGdfConfig};
use ferric_pbc::sr_rotation::SrColumnRotation;
use ferric_scf::result::ScfResult;
use ndarray::Array2;

pub const THREADS: [usize; 3] = [1, 2, 6];

pub const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

pub const TILT: [[f64; 3]; 3] = [[6.0, 0.0, 0.0], [1.1, 6.4, 0.0], [0.7, 0.9, 6.9]];

pub fn fnv(a: &Array2<f64>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for v in a.iter() {
        h ^= v.to_bits();
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn in_pool<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("rayon pool")
        .install(f)
}

/// `[grad, fit_orb_lr, fit_aux_lr, fit_metric_lr]` hashes of the force.
pub fn hashes(g: &GammaGradient) -> [u64; 4] {
    [
        fnv(&g.grad),
        fnv(&g.parts.fit_orb_lr),
        fnv(&g.parts.fit_aux_lr),
        fnv(&g.parts.fit_metric_lr),
    ]
}

pub struct Case {
    pub name: &'static str,
    pub basis: &'static str,
    pub split: bool,
    pub rot: bool,
    pub budget: usize,
    pub golden: [u64; 4],
}

pub const CASES: [Case; 5] = [
    Case {
        name: "water sto-3g unsplit, rotation off",
        basis: "sto-3g",
        split: false,
        rot: false,
        budget: 6 << 20,
        golden: [
            0xad853c6790b64336,
            0x3af806b65d0ce631,
            0x2738c9177ab5fb98,
            0x7b98fc5d0a84f762,
        ],
    },
    Case {
        name: "water sto-3g split, rotation off",
        basis: "sto-3g",
        split: true,
        rot: false,
        budget: 12 << 20,
        golden: [
            0xf490748c5f3284a0,
            0xad3d340e0bf89807,
            0x09618ed45c8e2ef0,
            0xcd7068c091093ae5,
        ],
    },
    Case {
        name: "water cc-pvdz split, rotation on",
        basis: "cc-pvdz",
        split: true,
        rot: true,
        budget: 16 << 20,
        golden: [
            0x4f411b7511adde43,
            0x90a0fbd1458a1cd0,
            0x7f37781dd5ba6708,
            0x38d97eb19ed98759,
        ],
    },
    Case {
        name: "water cc-pvdz split, rotation off",
        basis: "cc-pvdz",
        split: true,
        rot: false,
        budget: 16 << 20,
        golden: [
            0x8275cff04c7354bc,
            0x2f460b524ba471ff,
            0x5d6935cfaf7f086f,
            0x852fd251201c8abc,
        ],
    },
    Case {
        name: "water cc-pvdz unsplit, rotation on",
        basis: "cc-pvdz",
        split: false,
        rot: true,
        budget: 16 << 20,
        golden: [
            0x135414054ca949bd,
            0xec57f26ec5241c32,
            0x6e5b1cf9fe8b0a2c,
            0xfb48ef88ffcc6d57,
        ],
    },
];

/// A converged system of a case: the force is evaluated on it per pool.
pub struct Fx {
    pub cell: Cell,
    pub prep: PreparedBasis,
    pub hcfg: PeriodicHcoreConfig,
    pub hc: PeriodicHcore,
    pub aux: PreparedBasis,
    pub gdf: RsGdf,
    pub scf: ScfResult,
}

pub fn setup(c: &Case) -> Fx {
    let cell = Cell::new(Molecule::parse_xyz(WATER, 0, 1).unwrap(), TILT).unwrap();
    let prep = prep_for(&cell, &basis::bundled(c.basis).unwrap());
    let hcfg = PeriodicHcoreConfig {
        precision: 1e-14,
        ..PeriodicHcoreConfig::with_omega(0.8)
    };
    let hc = periodic_hcore(&cell, &prep, &hcfg).expect("hcore");
    let aux = prep_for(&cell, &basis::bundled("cc-pvdz-ri").unwrap());
    let cfg = RsGdfConfig {
        omega: 1.0,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(1 << 30),
        range_split: c.split.then(|| RangeSplit::new(1.0)),
        sr_column_rotation: if c.rot {
            SrColumnRotation::on()
        } else {
            SrColumnRotation::Off
        },
        ..Default::default()
    };
    let gdf = RsGdf::build_for_gradient(&cell, &prep, &aux, &hc.s, &cfg).expect("build");
    let g = gdf.clone().with_exxdiv(&cell, ExxDiv::None).unwrap();
    let scf = gamma_rhf_jk(
        &cell,
        &prep,
        &hc,
        Box::new(g.j_builder()),
        Box::new(g.k_builder()),
    );
    Fx {
        cell,
        prep,
        hcfg,
        hc,
        aux,
        gdf,
        scf,
    }
}

/// The RS-GDF RHF force of `fx` at the ambient rayon pool.
pub fn force(fx: &Fx, c: &Case) -> GammaGradient {
    let src = RsGdfGradSource {
        gdf: &fx.gdf,
        aux: &fx.aux,
        aux_jac: None,
    };
    let gcfg = GammaGradConfig {
        budget_bytes: Some(c.budget),
        ..Default::default()
    };
    gamma_rhf_gradient_rsgdf(
        &fx.cell,
        &fx.prep,
        &fx.hcfg,
        &fx.hc,
        &src,
        &fx.scf,
        ExxDiv::None,
        &gcfg,
    )
    .expect("force")
}

/// One case at each of `threads` against its own 1-thread result.
pub fn check(i: usize, threads: &[usize]) {
    let c = &CASES[i];
    let fx = setup(c);
    let one = in_pool(1, || force(&fx, c));
    let nc = one.fit.as_ref().unwrap().n_g_chunks;
    assert!(
        nc >= 8,
        "{}: only {nc} LR chunks, the window would not bind",
        c.name
    );
    for &n in threads {
        let g = in_pool(n, || force(&fx, c));
        assert_eq!(
            hashes(&g),
            hashes(&one),
            "{} at {n} threads vs 1 thread: [grad, orb_lr, aux_lr, metric_lr] hashes",
            c.name
        );
    }
}
