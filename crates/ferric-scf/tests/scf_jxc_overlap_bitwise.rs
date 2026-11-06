//! Exactness anchor for the overlapped RI-J build and XC digest
//! (`FERRIC_SCF_JXC_OVERLAP`, rhf.rs): a full SCF with the two stages under one
//! `rayon::join` must be BITWISE identical to the sequential order: energy,
//! final Fock, density and iteration count, at 1, 4 and 6 rayon threads.
//!
//! Bitwise, not toleranced: the failure this guards against is a changed
//! association in F (e.g. V_xc folded in before J), which moves F by ~1 ulp
//! and nothing else. The overlap counter proves the overlapped arm actually
//! ran for the systems that should take it, and did NOT run for the controls
//! (hybrid with DF-K, VV10, open shell), so a silently disabled overlap cannot
//! pass as "identical".
//!
//! ONE #[test] in this binary on purpose: it flips a process environment
//! variable, which would race any other test in the same process.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig, JXC_OVERLAP_BUILDS};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use ndarray::Array2;
use std::sync::atomic::Ordering;

const WATER: &str = "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n";
const NH3: &str = "4\nNH3\nN 0.000000 0.000000 0.116489\nH 0.000000 0.939731 -0.271808\n\
                   H 0.813831 -0.469865 -0.271808\nH -0.813831 -0.469865 -0.271808\n";
const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.9697\n";

struct Case {
    label: &'static str,
    xyz: &'static str,
    mult: usize,
    basis: &'static str,
    xc: &'static str,
    /// Whether the overlapped arm must run for this case.
    overlapped: bool,
}

const CASES: &[Case] = &[
    Case {
        label: "H2O LDA",
        xyz: WATER,
        mult: 1,
        basis: "def2-svp",
        xc: "LDA",
        overlapped: true,
    },
    Case {
        label: "NH3 PBE",
        xyz: NH3,
        mult: 1,
        basis: "cc-pvdz",
        xc: "PBE",
        overlapped: true,
    },
    Case {
        label: "H2O r2SCAN",
        xyz: WATER,
        mult: 1,
        basis: "def2-svp",
        xc: "r2SCAN",
        overlapped: true,
    },
    // RSH: K comes from the SR/LR fitters after the join, so it overlaps.
    Case {
        label: "H2O CAM-B3LYP",
        xyz: WATER,
        mult: 1,
        basis: "def2-svp",
        xc: "HYB_GGA_XC_CAM_B3LYP",
        overlapped: true,
    },
    // Controls: global hybrid (DF-K into k_buf), VV10, open shell.
    Case {
        label: "NH3 B3LYP",
        xyz: NH3,
        mult: 1,
        basis: "def2-svp",
        xc: "B3LYP",
        overlapped: false,
    },
    Case {
        label: "H2O wB97X-V",
        xyz: WATER,
        mult: 1,
        basis: "def2-svp",
        xc: "wB97X-V",
        overlapped: false,
    },
    Case {
        label: "OH UKS PBE",
        xyz: OH,
        mult: 2,
        basis: "def2-svp",
        xc: "PBE",
        overlapped: false,
    },
];

fn solve(c: &Case) -> (ScfResult, usize) {
    let mol = Molecule::parse_xyz(c.xyz, 0, c.mult).unwrap();
    let bs = basis::bundled(c.basis).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        xc: Some(c.xc.into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };
    let before = JXC_OVERLAP_BUILDS.load(Ordering::Relaxed);
    let r = if c.mult == 1 {
        solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap()
    } else {
        solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap()
    };
    (r, JXC_OVERLAP_BUILDS.load(Ordering::Relaxed) - before)
}

fn bits(a: &Array2<f64>) -> Vec<u64> {
    a.iter().map(|v| v.to_bits()).collect()
}

fn assert_bitwise(label: &str, seq: &ScfResult, par: &ScfResult) {
    assert_eq!(seq.iterations, par.iterations, "{label}: iteration count");
    assert_eq!(
        seq.energy.to_bits(),
        par.energy.to_bits(),
        "{label}: energy {} vs {}",
        seq.energy,
        par.energy
    );
    let mismatched = bits(&seq.fock_alpha)
        .iter()
        .zip(bits(&par.fock_alpha))
        .filter(|(a, b)| **a != *b)
        .count();
    assert_eq!(
        mismatched, 0,
        "{label}: {mismatched} Fock elements differ in bits"
    );
    assert_eq!(
        bits(&seq.density_total),
        bits(&par.density_total),
        "{label}: density"
    );
}

#[test]
fn jxc_overlap_is_bitwise_identical_to_sequential() {
    for threads in [1usize, 4, 6] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for c in CASES {
                let label = format!("{} @{threads} threads", c.label);
                std::env::set_var("FERRIC_SCF_JXC_OVERLAP", "0");
                let (seq, n_seq) = solve(c);
                std::env::remove_var("FERRIC_SCF_JXC_OVERLAP");
                let (par, n_par) = solve(c);
                eprintln!(
                    "[{label}] E = {:.12} iters {} overlapped builds {n_par}",
                    par.energy, par.iterations
                );
                assert_eq!(n_seq, 0, "{label}: overlap ran with the switch off");
                if c.overlapped {
                    assert_eq!(
                        n_par, par.iterations,
                        "{label}: overlap did not run every iteration"
                    );
                } else {
                    assert_eq!(n_par, 0, "{label}: control took the overlapped arm");
                }
                assert!(seq.converged && par.converged, "{label}: not converged");
                assert_bitwise(&label, &seq, &par);
            }
        });
    }
}
