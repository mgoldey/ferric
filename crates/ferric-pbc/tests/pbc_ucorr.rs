//! Stage 8: Gamma-point UMP2 / URPA (`ferric_pbc::ucorr::{gamma_ump2,
//! gamma_urpa}`) on the Gamma UHF, ported from `reference/pbc/pbc_ump2.py`
//! (FINDINGS "Iteration 7 (Python, Gamma UMP2/URPA)"). Every pinned number is
//! from that iteration (`run_ump2_anchor.py`, `run_ump2_oracle.py`,
//! `run_ump2_box_limit.py`, `test_prototype.py` `UMP2_REF`).
//!
//! # Exactness anchors (written first)
//!
//! (a) Closed shell through the U drivers ≡ `gamma_mp2` / `gamma_drpa` (same
//!     C, a Restricted result relabelled Unrestricted), both conventions,
//!     dense AND trivial-aux B: prototype ≤ 2.4e-16; asserted 1e-12.
//!     Negative control: a per-spin χ₀ factor 4 (B·√2 per spin) moves URPA
//!     by > 1e-2 (prototype −1.8e-2 / −5.3e-2).
//! (b) Open shell, trivial-aux RS-GDF B ≡ dense pure-AFT, same UHF C, both
//!     conventions: tri one-s TRIPLET (3,1; only αβ nonzero — α has one
//!     virtual) AND the penta one-s DOUBLET (3,2; αα, ββ, αβ all nonzero).
//!     Prototype UMP2 ≤ 1.2e-14, URPA ≤ 2.1e-13; asserted 1e-11. The two
//!     sides share no ERI, no MO transform and no energy formula (ferric-mp2
//!     kernels / ferric-rpa log-det vs an independent loop / joint plasmon).
//!     Mutants: α C used for the β B (UMP2 > 1e-4, URPA > 1e-5); per-spin
//!     factor 4 (URPA > 1e-2). The penta blocks are asserted nonzero so a
//!     dropped same-spin ½ or dropped exchange is VISIBLE (the triplet's αα
//!     block is identically zero and could not see them).
//! (c) −(1/2π) ∫ tr Π²/2 ≡ direct UMP2 (penta, prototype 1.2e-14; asserted
//!     1e-12), plus the λ → 0 limit of the PRODUCTION URPA path: pins the
//!     per-spin factor 2 and the half-line normalisation without PySCF.
//!
//! # Oracles
//!
//! (d) PySCF 2.13 `pbc.mp.UMP2` pins (URPA column: PySCF's own AFTDF (ia|jb)
//!     + the plasmon formula; PySCF has no periodic RPA), all four
//!     (reference exxdiv, convention) combinations: penta doublet, H2/6-31G
//!     a=4 TRIPLET (αα only; STO-3G would be identically 0), tri 4H s+p
//!     triplet. RISK (not measured in Rust): the pins assume ferric's UHF
//!     lands on the prototype's state from the core guess; tri s+p's none
//!     UHF is independently pinned in `pbc_uhf.rs`, the other two are not.
//! (e) Box limit (independent of PySCF): H/6-31G doublet at a = 20 — UMP2
//!     ≡ 0 (one electron), URPA residual = c3/a³ + c6/a⁶ with BOTH predicted
//!     from the molecule by the prototype (0.046533, 0.79013). NH/6-31G
//!     triplet at a = 32, 40 vs predicted c3 (UMP2 0.761014, URPA 1.502486)
//!     with the two wrong-shift mutants; see its `#[ignore]` for why it
//!     cannot run on today's Rust builders.
//!
//! # The two wrong-shift mutants (documented; asserted in (d) and (e))
//!
//! * `v_M/2` per spin (the RHF "D/2" bookkeeping on a spin density): every
//!   occupied ε of both spins shifted by −v_M/2 instead of −v_M.
//! * α-only shift: β occupied left unshifted.
//! Prototype box limit (NH, dE·a at 24/32/40): UMP2 −0.0545 / −0.0439,
//! URPA −0.063 / −0.062 — flat 1/a plateaus, 30-60× the correct residual at
//! a = 40. Only the SAME v_M on both spins gives a⁻³. On the pins both sit
//! > 1e-5 from the shifted value (asserted below). They are injected by
//! editing the reference's ε (the driver has one shift for both spins).
//!
//! # Mutation plan (for the main agent; each must fail ≥ 1 test here)
//!
//! * `ucorr::spaces`: shift only `ea_in` (β unshifted) → (d) shifted pins.
//! * `ucorr::gamma_ump2`: pass `sp.eps[0]` for both spins → (b) penta, (d).
//! * `ucorr::b_ov_spin`: use `sp.c[0]` for both spins → (b) (triplet+penta).
//! * ferric-rpa `u_pdep_rpa_core`: drop `chan_b` from the matvec → (a), (b).
//! * `urpa_plasmon`: `2.0 * sd[p]` → `4.0 * sd[p]` → (a) dense, (b), (d).
//! * ferric-mp2 `u_ri_mp2_from_parts`: `e_ab = 0` → (a), (b), (d).
//!
//! # Molecular paths byte-identical
//!
//! ferric-mp2: `u_ri_mp2_from_parts` is ADDITIVE (u_ri_mp2 untouched).
//! ferric-rpa: `run_u_pdep_rpa`'s post-intermediate body moved verbatim into
//! a private `u_pdep_rpa_core` that both it and `run_u_pdep_rpa_from_parts`
//! call. (f) below asserts BITWISE equality of both new entries with the
//! molecular drivers on H3/cc-pVDZ (doublet, all spin blocks nonzero).
//! Regression to run:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-mp2 --lib u_rimp2
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-mp2 \
//!     --test mwe_mp2_pool_is_inert_without_a_pool --test mwe_rimp2_has_no_guard
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-rpa --lib
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-rpa --test u_pdep_rpa \
//!     --test mwe_open_shell_dynamic_scratch --test laplace_chi0 --test pdep_rpa
//! OPENBLAS_NUM_THREADS=1 cargo test --release -p ferric-gw \
//!     --test u_cohsex_evgw_closed_shell_limit
//! ```

mod common;

use common::*;
use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::{Atom, Molecule};
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ferric_mp2::canonical::dense_ao_eri;
use ferric_mp2::rimp2::{compute_rpa_intermediates_spin, RiMp2Config, RpaIntermediates};
use ferric_mp2::u_rimp2::{u_ri_mp2, u_ri_mp2_from_parts};
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::drpa::{
    gamma_drpa, pdep_config, GammaDrpaConfig, GammaDrpaIntegrals, DEFAULT_GAMMA_DRPA_QUAD_POINTS,
};
use ferric_pbc::ewald::madelung_constant;
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::mp2::{
    b_ov_from_ao_b, gamma_mp2, GammaMp2Config, GammaMp2Integrals, Mp2Denominators,
};
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
use ferric_pbc::ucorr::{
    direct_ump2_from_ovov, gamma_ump2, gamma_urpa, ovov_from_dense_pair, ump2_from_ovov,
    urpa_plasmon, urpa_second_order_from_b_ov,
};
use ferric_pbc::uhf::{gamma_uhf, EwaldStart, GammaUhfConfig, GammaUhfIntegrals};
use ferric_rpa::{run_u_pdep_rpa, run_u_pdep_rpa_from_parts, Chi0Sparsity, Eigensolver};
use ferric_scf::result::{ScfResult, Spin};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::{s, Array2};

use ferric_pbc::mp2::Mp2Denominators::{MadelungShifted as Shifted, Unshifted};

const HCORE_OMEGA: f64 = 0.8;
const ANCHOR_ALPHA: f64 = 0.5;
const AMPLE: usize = 1 << 30;
/// Anchor grid (as in pbc_drpa.rs: quadrature error ~1e-15 at n >= 40).
const ANCHOR_QUAD: usize = 80;
const DENS: [Mp2Denominators; 2] = [Shifted, Unshifted];

/// `test_prototype.py` PENTA_ATOMS = TRI_ATOMS + one H (Bohr).
const PENTA_ATOMS: [[f64; 3]; 5] = [
    [0.1, 0.2, 0.3],
    [0.1, 0.2, 1.7],
    [2.4, 2.5, 2.2],
    [3.6, 2.9, 2.6],
    [1.3, 3.1, 4.0],
];

// --- test_prototype.py UMP2_REF: PySCF 2.13 pbc.mp.UMP2 on pbc.scf.UHF +
// AFTDF (mesh 61³); URPA = PySCF's AFTDF (ia|jb) + the plasmon formula.
// (unshifted, shifted) × (UMP2, URPA). ours − PySCF: penta ≤ 1.2e-12, H2
// ≤ 2.3e-14 (URPA 9e-12), tri s+p ≤ 4.5e-10 (SCF orbitals).
const PENTA_REF: [[f64; 2]; 2] = [
    [-2.167566950256e-02, -3.602910454467e-02],
    [-1.179101239432e-02, -2.150209066693e-02],
];
const H2_631G_TRIPLET_REF: [[f64; 2]; 2] = [
    [-8.878036761172e-04, -3.225957845588e-02],
    [-5.712132378726e-04, -1.133223577905e-02],
];
const TRI_SP_TRIPLET_REF: [[f64; 2]; 2] = [
    [-3.332802589058e-02, -6.935114524772e-02],
    [-2.278123432187e-02, -4.834534740428e-02],
];

// --- Box limit (FINDINGS Iteration 7; predicted from MOLECULAR quantities
// before the sweep, except c6, computed after it from the molecule only).
const H_URPA_C3: f64 = 0.046533;
const H_URPA_C6: f64 = 0.79013;
const NH_C3_UMP2: f64 = 0.761014;
const NH_C3_URPA: f64 = 1.502486;

fn cell_mult(atoms: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(atoms);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

fn hcore(cell: &Cell, prep: &PreparedBasis) -> PeriodicHcore {
    periodic_hcore(cell, prep, &PeriodicHcoreConfig::with_omega(HCORE_OMEGA)).expect("hcore")
}

fn dense_none(cell: &Cell, prep: &PreparedBasis, hc: &PeriodicHcore) -> DenseAftEri {
    DenseAftEri::build(
        cell,
        prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .expect("dense AFT")
}

fn uhf_scf() -> RhfConfig {
    RhfConfig {
        use_sad_guess: false,
        density_conv: 1e-10,
        max_iter: 400,
        ..Default::default()
    }
}

/// Gamma UHF on the dense tensor (Direct for `none`, Staged for `ewald`).
fn uhf_dense(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    exx: ExxDiv,
) -> ScfResult {
    let start = match exx {
        ExxDiv::None => EwaldStart::Direct,
        ExxDiv::Ewald => EwaldStart::Staged,
    };
    let cfg = GammaUhfConfig {
        exxdiv: exx,
        ewald_start: start,
        scf: uhf_scf(),
        initial_mos: None,
    };
    let r = gamma_uhf(cell, prep, hc, GammaUhfIntegrals::DenseAft(eri), &cfg).expect("gamma UHF");
    assert!(r.scf.converged);
    eprintln!(
        "  UHF {exx:?}: E {:.12}, <S2> {:.9}, gaps {:?}",
        r.scf.energy, r.s2, r.gaps
    );
    r.scf
}

fn mcfg(exx: ExxDiv, den: Mp2Denominators) -> GammaMp2Config {
    GammaMp2Config {
        frozen_core: 0,
        reference_exxdiv: exx,
        denominators: den,
        budget_bytes: Some(AMPLE),
    }
}

fn rcfg(exx: ExxDiv, den: Mp2Denominators, q: usize) -> GammaDrpaConfig {
    GammaDrpaConfig {
        frozen_core: 0,
        reference_exxdiv: exx,
        denominators: den,
        quad_points: q,
        budget_bytes: Some(AMPLE),
    }
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    eprintln!(
        "  {what}: {got:.12e} (want {want:.12e}, diff {:.2e})",
        got - want
    );
    assert!(got.is_finite(), "{what}: non-finite");
    assert!(
        (got - want).abs() < tol,
        "{what}: {got:.12e} vs {want:.12e} (tol {tol:.1e})"
    );
}

/// Every periodic pair product of a one-s cell (pair types × 8 half-lattice
/// classes), exponent 2α — the exact aux span (`test_prototype._tri_anchor`,
/// `_penta_anchor`).
fn pair_sites(cell: &Cell) -> Vec<[f64; 4]> {
    let r = cell.positions();
    let a = cell.lattice();
    let n = r.len();
    let mut out = Vec::new();
    for i in 0..n {
        for j in i..n {
            for k in 0..8usize {
                let h = [(k >> 2) & 1, (k >> 1) & 1, k & 1];
                let mut c = [0.0; 3];
                for d in 0..3 {
                    let ha: f64 = (0..3).map(|x| h[x] as f64 * a[x][d]).sum();
                    c[d] = 0.5 * (r[i][d] + r[j][d] + ha);
                }
                out.push([c[0], c[1], c[2], 2.0 * ANCHOR_ALPHA]);
            }
        }
    }
    out
}

/// One-s trivial-aux anchor: cell, prep, dense ERI and exact-span RS-GDF B.
struct Anchor {
    cell: Cell,
    prep: PreparedBasis,
    hc: PeriodicHcore,
    eri: DenseAftEri,
    gdf: RsGdf,
}

fn anchor(atoms: &[[f64; 3]], mult: usize, gdf_precision: Option<f64>) -> Anchor {
    let cell = cell_mult(atoms, TRI_A, mult);
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    let hc = hcore(&cell, &prep);
    let eri = dense_none(&cell, &prep, &hc);
    let n = atoms.len();
    let site = SiteBasis::new(&pair_sites(&cell), 0).unwrap();
    assert_eq!(site.prep.nbasis(), 8 * n * (n + 1) / 2);
    let mut cfg = RsGdfConfig {
        omega: 0.8,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    };
    if let Some(p) = gdf_precision {
        cfg.precision = p;
    }
    let gdf = RsGdf::build(&cell, &prep, &site.prep, &hc.s, &cfg).unwrap();
    Anchor {
        cell,
        prep,
        hc,
        eri,
        gdf,
    }
}

/// A Restricted result relabelled Unrestricted with C_β = C_α, ε_β = ε_α.
fn as_unrestricted(r: &ScfResult) -> ScfResult {
    let mut u = r.clone();
    u.spin = Spin::Unrestricted;
    u.mos_beta = Some(r.mos_alpha.clone());
    u.eps_beta = Some(r.eps_alpha.clone());
    u
}

/// Per-spin intermediates built OUTSIDE the driver (for mutants): `c[s]`
/// is the MO set used for spin s's B, `n[s]` its occupied count.
fn u_inter(
    gdf: &RsGdf,
    nao: usize,
    c: [&Array2<f64>; 2],
    n: [usize; 2],
    scale: f64,
) -> [RpaIntermediates; 2] {
    let naux = gdf.b().nrows();
    let mk = |c: &Array2<f64>, no: usize| {
        let nv = c.ncols() - no;
        let b = if no * nv == 0 {
            Array2::zeros((naux, 0))
        } else {
            b_ov_from_ao_b(gdf.b(), nao, c.slice(s![.., ..no]), c.slice(s![.., no..])).unwrap()
        };
        RpaIntermediates {
            b_ov: b.mapv(|x| x * scale),
            v_inv_sqrt: gdf.metric_inv_sqrt().to_owned(),
            nocc: no,
            nvir: nv,
            nocc_total: no,
            first_occ: 0,
            naux,
        }
    };
    [mk(c[0], n[0]), mk(c[1], n[1])]
}

/// Occupied / virtual slices of both spins, occupied shifted by `shift`.
fn u_eps(u: &ScfResult, na: usize, nb: usize, shift: f64) -> [Vec<f64>; 4] {
    let (ea, eb) = (u.eps_a(), u.eps_b());
    [
        ea[..na].iter().map(|e| e + shift).collect(),
        ea[na..].to_vec(),
        eb[..nb].iter().map(|e| e + shift).collect(),
        eb[nb..].to_vec(),
    ]
}

fn urpa_parts(inter: &[RpaIntermediates; 2], e: &[Vec<f64>; 4]) -> f64 {
    run_u_pdep_rpa_from_parts(
        &inter[0],
        &inter[1],
        &e[0],
        &e[1],
        &e[2],
        &e[3],
        &pdep_config(ANCHOR_QUAD, AMPLE),
    )
    .unwrap()
    .e_rpa
}

/// Full-array energies (occupied shifted) for `u_ri_mp2_from_parts`.
fn full_eps(u: &ScfResult, na: usize, nb: usize, shift: f64) -> [Vec<f64>; 2] {
    let f = |e: &[f64], n: usize| -> Vec<f64> {
        e.iter()
            .enumerate()
            .map(|(p, x)| if p < n { x + shift } else { *x })
            .collect()
    };
    [f(u.eps_a(), na), f(u.eps_b(), nb)]
}

// ===========================================================================
// (a) Closed shell through the U drivers ≡ the restricted drivers.
// ===========================================================================

#[test]
fn closed_shell_u_equals_restricted_both_conventions() {
    // H2/STO-3G a=4 (dense only) and the tri one-s anchor (dense + B).
    let h2 = h2_cell(4.0);
    let h2_prep = prep_for(&h2, &pyscf_sto3g_h());
    let h2_hc = hcore(&h2, &h2_prep);
    let h2_eri = dense_none(&h2, &h2_prep, &h2_hc);
    let tri = anchor(&TRI_ATOMS, 1, None);
    let cases: [(
        &str,
        &Cell,
        &PreparedBasis,
        &PeriodicHcore,
        &DenseAftEri,
        Option<&RsGdf>,
    ); 2] = [
        ("H2/STO-3G a=4", &h2, &h2_prep, &h2_hc, &h2_eri, None),
        (
            "tri one-s",
            &tri.cell,
            &tri.prep,
            &tri.hc,
            &tri.eri,
            Some(&tri.gdf),
        ),
    ];
    for (label, cell, prep, hc, eri, gdf) in cases {
        let r = gamma_rhf(cell, prep, hc, eri);
        let u = as_unrestricted(&r);
        for den in DENS {
            let mut ints: Vec<(&str, GammaMp2Integrals<'_>, GammaDrpaIntegrals<'_>)> = vec![(
                "dense",
                GammaMp2Integrals::DenseAft(eri),
                GammaDrpaIntegrals::DenseAft(eri),
            )];
            if let Some(g) = gdf {
                ints.push((
                    "B",
                    GammaMp2Integrals::RsGdf(g),
                    GammaDrpaIntegrals::RsGdf(g),
                ));
            }
            for (src, im, ir) in ints {
                let rm = gamma_mp2(cell, &r, im, &mcfg(ExxDiv::None, den)).unwrap();
                let um = gamma_ump2(cell, &u, im, &mcfg(ExxDiv::None, den)).unwrap();
                let rr = gamma_drpa(cell, &r, ir, &rcfg(ExxDiv::None, den, ANCHOR_QUAD)).unwrap();
                let ur = gamma_urpa(cell, &u, ir, &rcfg(ExxDiv::None, den, ANCHOR_QUAD)).unwrap();
                eprintln!(
                    "{label} {den:?} {src}: RMP2 {:.12e} UMP2 {:+.2e} (os {:+.2e}, ss {:+.2e}); \
                     dRPA {:.12e} URPA {:+.2e}",
                    rm.mp2_corr,
                    um.mp2_corr - rm.mp2_corr,
                    um.components.e_ab - rm.components.e_os,
                    um.components.e_aa + um.components.e_bb - rm.components.e_ss,
                    rr.drpa_corr,
                    ur.rpa_corr - rr.drpa_corr
                );
                assert!(
                    (um.mp2_corr - rm.mp2_corr).abs() < 1e-12,
                    "{label} {den:?} {src}"
                );
                assert!((um.components.e_ab - rm.components.e_os).abs() < 1e-12);
                assert!(
                    (um.components.e_aa + um.components.e_bb - rm.components.e_ss).abs() < 1e-12
                );
                assert!((um.components.e_aa - um.components.e_bb).abs() < 1e-13);
                // U and R reach the dRPA log-det through different eigensolves
                // (Pi_a + Pi_b vs the closed-shell factor 4): measured 4.5e-12
                // (2e-10 relative) on tri one-s, while UMP2 agrees to 1e-18.
                // Both paths match the exact dense plasmon at 1e-11 elsewhere.
                assert!(
                    (ur.rpa_corr - rr.drpa_corr).abs() < 1e-10,
                    "{label} {den:?} {src}"
                );
                assert!((um.occ_shift - rm.occ_shift).abs() == 0.0);
            }
        }
        // Negative control: per-spin χ₀ factor 4 (the closed-shell factor on
        // EACH spin channel), through the production parts path.
        if let Some(g) = gdf {
            let nao = prep.nbasis();
            let n = (cell.mol().nelec() / 2) as usize;
            let c = r.mos_r();
            let vm = madelung_constant(cell).unwrap();
            let e = u_eps(&u, n, n, -vm);
            let good = urpa_parts(&u_inter(g, nao, [c, c], [n, n], 1.0), &e);
            let bad = urpa_parts(&u_inter(g, nao, [c, c], [n, n], 2f64.sqrt()), &e);
            let want = gamma_drpa(
                cell,
                &r,
                GammaDrpaIntegrals::RsGdf(g),
                &rcfg(ExxDiv::None, Shifted, ANCHOR_QUAD),
            )
            .unwrap()
            .drpa_corr;
            eprintln!(
                "{label}: parts {:+.2e}; per-spin factor 4 {:+.3e}",
                good - want,
                bad - want
            );
            // same U-vs-R eigensolve difference as above: measured 4.49e-12
            assert!((good - want).abs() < 1e-10);
            assert!((bad - want).abs() > 1e-2, "factor-4 mutant not visible");
        }
    }
}

// ===========================================================================
// (b) Open shell: trivial-aux B ≡ dense (triplet AND doublet).
// ===========================================================================

#[test]
fn open_shell_b_matches_dense_in_the_trivial_aux_limit_triplet_and_doublet() {
    // (label, atoms, multiplicity, (N_α, N_β), RS-GDF precision)
    let systems: [(&str, &[[f64; 3]], usize, (usize, usize), Option<f64>); 2] = [
        ("tri one-s triplet", &TRI_ATOMS, 3, (3, 1), None),
        // prec 1e-15: at 1e-13 the SR truncation left dUMP2 ~1e-11 (prototype).
        ("penta one-s doublet", &PENTA_ATOMS, 2, (3, 2), Some(1e-15)),
    ];
    for (label, atoms, mult, (na, nb), prec) in systems {
        let t = anchor(atoms, mult, prec);
        let cell = &t.cell;
        let nao = t.prep.nbasis();
        let u = uhf_dense(cell, &t.prep, &t.hc, &t.eri, ExxDiv::None);
        let vm = madelung_constant(cell).unwrap();
        let (ca, cb) = (u.mos_a(), u.mos_b());
        for den in DENS {
            let d = GammaMp2Integrals::DenseAft(&t.eri);
            let bm = GammaMp2Integrals::RsGdf(&t.gdf);
            let em = gamma_ump2(cell, &u, d, &mcfg(ExxDiv::None, den)).unwrap();
            let eb = gamma_ump2(cell, &u, bm, &mcfg(ExxDiv::None, den)).unwrap();
            let rd = gamma_urpa(
                cell,
                &u,
                GammaDrpaIntegrals::DenseAft(&t.eri),
                &rcfg(ExxDiv::None, den, ANCHOR_QUAD),
            )
            .unwrap();
            let rb = gamma_urpa(
                cell,
                &u,
                GammaDrpaIntegrals::RsGdf(&t.gdf),
                &rcfg(ExxDiv::None, den, ANCHOR_QUAD),
            )
            .unwrap();
            eprintln!(
                "{label} {den:?}: UMP2 dense {:.12e} (aa {:.3e} bb {:.3e} ab {:.3e}), B {:+.2e}; \
                 URPA plasmon {:.12e}, B {:+.2e}",
                em.mp2_corr,
                em.components.e_aa,
                em.components.e_bb,
                em.components.e_ab,
                eb.mp2_corr - em.mp2_corr,
                rd.rpa_corr,
                rb.rpa_corr - rd.rpa_corr
            );
            assert_eq!(em.nocc_active, [na, nb]);
            assert_eq!(eb.naux, Some(t.gdf.stats().naux_kept));
            assert!(
                (eb.mp2_corr - em.mp2_corr).abs() < 1e-11,
                "{label} {den:?} UMP2"
            );
            for (x, y) in [
                (eb.components.e_aa, em.components.e_aa),
                (eb.components.e_bb, em.components.e_bb),
                (eb.components.e_ab, em.components.e_ab),
            ] {
                assert!((x - y).abs() < 1e-11, "{label} {den:?} block");
            }
            assert!(
                (rb.rpa_corr - rd.rpa_corr).abs() < 1e-11,
                "{label} {den:?} URPA"
            );
            let lam = rb.eigenvalues_static.as_ref().unwrap();
            assert!(lam.iter().all(|&l| l >= 1.0 - 1e-12), "{lam:?}");

            // Mutants through the parts entries, same shift as the driver.
            let shift = if den == Shifted { -vm } else { 0.0 };
            let e4 = u_eps(&u, na, nb, shift);
            let ef = full_eps(&u, na, nb, shift);
            let good = u_inter(&t.gdf, nao, [ca, cb], [na, nb], 1.0);
            let swap = u_inter(&t.gdf, nao, [ca, ca], [na, nb], 1.0);
            let four = u_inter(&t.gdf, nao, [ca, cb], [na, nb], 2f64.sqrt());
            let m_good = u_ri_mp2_from_parts(&good[0], &good[1], &ef[0], &ef[1])
                .unwrap()
                .e_total;
            let m_swap = u_ri_mp2_from_parts(&swap[0], &swap[1], &ef[0], &ef[1])
                .unwrap()
                .e_total;
            let r_good = urpa_parts(&good, &e4);
            let r_swap = urpa_parts(&swap, &e4);
            let r_four = urpa_parts(&four, &e4);
            eprintln!(
                "  mutants: α C for β B UMP2 {:+.2e} URPA {:+.2e}; per-spin factor 4 URPA {:+.2e}",
                m_swap - em.mp2_corr,
                r_swap - rd.rpa_corr,
                r_four - rd.rpa_corr
            );
            assert!((m_good - em.mp2_corr).abs() < 1e-11 && (r_good - rd.rpa_corr).abs() < 1e-11);
            assert!(
                (m_swap - em.mp2_corr).abs() > 1e-4,
                "{label}: α-for-β UMP2 invisible"
            );
            assert!(
                (r_swap - rd.rpa_corr).abs() > 1e-5,
                "{label}: α-for-β URPA invisible"
            );
            assert!(
                (r_four - rd.rpa_corr).abs() > 1e-2,
                "{label}: factor-4 URPA invisible"
            );
        }
        if label.starts_with("penta") {
            // All three blocks nonzero (a dropped ½ or dropped exchange is
            // visible here), and UMP2 is not its direct (ring) part.
            let em = gamma_ump2(
                cell,
                &u,
                GammaMp2Integrals::DenseAft(&t.eri),
                &mcfg(ExxDiv::None, Shifted),
            )
            .unwrap();
            let c = em.components;
            assert!(c.e_aa < -1e-7 && c.e_bb < -1e-6 && c.e_ab < -1e-3, "{c:?}");
            let e = u_eps(&u, na, nb, -vm);
            let blk = |c1: &Array2<f64>, n1: usize, c2: &Array2<f64>, n2: usize| {
                ovov_from_dense_pair(
                    t.eri.eri(),
                    nao,
                    c1.slice(s![.., ..n1]),
                    c1.slice(s![.., n1..]),
                    c2.slice(s![.., ..n2]),
                    c2.slice(s![.., n2..]),
                )
                .unwrap()
            };
            let (aa, bb, ab) = (
                blk(ca, na, ca, na),
                blk(cb, nb, cb, nb),
                blk(ca, na, cb, nb),
            );
            let dm = direct_ump2_from_ovov(&aa, &bb, &ab, &e[0], &e[1], &e[2], &e[3]).unwrap();
            assert!(
                (dm - em.mp2_corr).abs() > 1e-3,
                "direct {dm} vs UMP2 {}",
                em.mp2_corr
            );
            let again = ump2_from_ovov(&aa, &bb, &ab, &e[0], &e[1], &e[2], &e[3]).unwrap();
            assert!((again.e_total - em.mp2_corr).abs() < 1e-15);
        }
    }
}

// ===========================================================================
// (c) URPA's O(Π²) term is direct UMP2 (penta doublet).
// ===========================================================================

#[test]
fn urpa_second_order_term_is_direct_ump2() {
    let t = anchor(&PENTA_ATOMS, 2, Some(1e-15));
    let (na, nb) = (3usize, 2usize);
    let nao = t.prep.nbasis();
    let u = uhf_dense(&t.cell, &t.prep, &t.hc, &t.eri, ExxDiv::None);
    let vm = madelung_constant(&t.cell).unwrap();
    let (ca, cb) = (u.mos_a(), u.mos_b());
    let e = u_eps(&u, na, nb, -vm);
    let blk = |c1: &Array2<f64>, n1: usize, c2: &Array2<f64>, n2: usize| {
        ovov_from_dense_pair(
            t.eri.eri(),
            nao,
            c1.slice(s![.., ..n1]),
            c1.slice(s![.., n1..]),
            c2.slice(s![.., ..n2]),
            c2.slice(s![.., n2..]),
        )
        .unwrap()
    };
    let (aa, bb, ab) = (
        blk(ca, na, ca, na),
        blk(cb, nb, cb, nb),
        blk(ca, na, cb, nb),
    );
    let dm = direct_ump2_from_ovov(&aa, &bb, &ab, &e[0], &e[1], &e[2], &e[3]).unwrap();
    let inter = u_inter(&t.gdf, nao, [ca, cb], [na, nb], 1.0);
    let e2 = urpa_second_order_from_b_ov(
        &inter[0].b_ov,
        &inter[1].b_ov,
        &e[0],
        &e[1],
        &e[2],
        &e[3],
        256,
    )
    .unwrap();
    let e2_bad = urpa_second_order_from_b_ov(
        &inter[0].b_ov.mapv(|x| x * 2f64.sqrt()),
        &inter[1].b_ov.mapv(|x| x * 2f64.sqrt()),
        &e[0],
        &e[1],
        &e[2],
        &e[3],
        256,
    )
    .unwrap();
    // The production parts path at weak coupling: B → √λ B, E(λ)/λ² =
    // dUMP2 + c1 λ + c2 λ² + …; Richardson on λ = h, h/2, h/4 (h = 0.04).
    let f = |lam: f64| {
        urpa_parts(&u_inter(&t.gdf, nao, [ca, cb], [na, nb], lam.sqrt()), &e) / (lam * lam)
    };
    let h = 0.04;
    let (f1, f2, f4) = (f(h), f(h / 2.0), f(h / 4.0));
    let rich = (8.0 * f4 - 6.0 * f2 + f1) / 3.0;
    eprintln!(
        "penta: direct UMP2 {dm:.12e}; tr Π² term {:+.2e}; factor-4 {:+.2e}; production λ→0 rel \
         {:+.2e} (λ = 0.04: {:+.2e})",
        e2 - dm,
        e2_bad - dm,
        (rich - dm) / dm,
        (f1 - dm) / dm
    );
    assert!((e2 - dm).abs() < 1e-12, "tr Π² vs direct UMP2");
    assert!((e2_bad - dm).abs() > 1e-2, "factor-4 mutant invisible");
    assert!(((rich - dm) / dm).abs() < 1e-4, "production O(V²) limit");
    assert!(((f1 - dm) / dm).abs() > ((rich - dm) / dm).abs());
}

// ===========================================================================
// (d) PySCF pbc.mp.UMP2 pins, all four (reference, convention) combinations.
// ===========================================================================

fn pinned_case(
    label: &str,
    cell: &Cell,
    prep: &PreparedBasis,
    reference: &[[f64; 2]; 2],
    tol: f64,
) {
    let hc = hcore(cell, prep);
    let eri = dense_none(cell, prep, &hc);
    let vm = madelung_constant(cell).unwrap();
    eprintln!("{label} (v_M {vm:.10}):");
    let u_none = uhf_dense(cell, prep, &hc, &eri, ExxDiv::None);
    let u_ew = uhf_dense(cell, prep, &hc, &eri, ExxDiv::Ewald);
    let q = DEFAULT_GAMMA_DRPA_QUAD_POINTS;
    let dm = GammaMp2Integrals::DenseAft(&eri);
    let dr = GammaDrpaIntegrals::DenseAft(&eri);
    for (u, exx, den, want, shift) in [
        (&u_none, ExxDiv::None, Unshifted, reference[0], 0.0),
        (&u_none, ExxDiv::None, Shifted, reference[1], -vm),
        (&u_ew, ExxDiv::Ewald, Shifted, reference[1], 0.0),
        (&u_ew, ExxDiv::Ewald, Unshifted, reference[0], vm),
    ] {
        let m = gamma_ump2(cell, u, dm, &mcfg(exx, den)).unwrap();
        let r = gamma_urpa(cell, u, dr, &rcfg(exx, den, q)).unwrap();
        assert!((m.occ_shift - shift).abs() < 1e-12 && (r.occ_shift - shift).abs() < 1e-12);
        close(
            m.mp2_corr,
            want[0],
            tol,
            &format!("UMP2 ref {exx:?} → {den:?}"),
        );
        close(
            r.rpa_corr,
            want[1],
            tol,
            &format!("URPA ref {exx:?} → {den:?}"),
        );
    }
    // A mislabelled reference (ewald declared as none) double-shifts.
    let wrong = gamma_ump2(cell, &u_ew, dm, &mcfg(ExxDiv::None, Shifted)).unwrap();
    assert!(
        (wrong.mp2_corr - reference[1][0]).abs() > 1e-5,
        "mislabel invisible"
    );

    // The two wrong-shift mutants on the ewald reference (driver shift 0):
    // (i) v_M/2 per spin (occupied raised by v_M/2 on both spins), (ii)
    // α-only (β occupied raised back by v_M).
    let (na, nb) = ferric_pbc::uhf::nocc_ab(cell.mol()).unwrap();
    for (name, da, db) in [
        ("v_M/2 per spin", 0.5 * vm, 0.5 * vm),
        ("alpha-only", 0.0, vm),
    ] {
        if nb == 0 && name == "alpha-only" {
            continue; // no β occupied: identical to the correct shift
        }
        let mut bad = u_ew.clone();
        for e in bad.eps_alpha.iter_mut().take(na) {
            *e += da;
        }
        for e in bad.eps_beta.as_mut().unwrap().iter_mut().take(nb) {
            *e += db;
        }
        let m = gamma_ump2(cell, &bad, dm, &mcfg(ExxDiv::Ewald, Shifted))
            .unwrap()
            .mp2_corr;
        let r = gamma_urpa(cell, &bad, dr, &rcfg(ExxDiv::Ewald, Shifted, q))
            .unwrap()
            .rpa_corr;
        eprintln!(
            "  mutant {name}: UMP2 {:+.3e}, URPA {:+.3e} from the shifted pins",
            m - reference[1][0],
            r - reference[1][1]
        );
        // 1e-5 ≫ the 1e-10 SCF floor; H2/6-31G's whole shifted−unshifted UMP2
        // gap is only 3.2e-4, so a half shift sits ~1.5e-4 away.
        assert!(
            (m - reference[1][0]).abs() > 1e-5,
            "{label} {name} UMP2 invisible"
        );
        assert!(
            (r - reference[1][1]).abs() > 1e-5,
            "{label} {name} URPA invisible"
        );
    }
}

#[test]
fn penta_doublet_matches_pinned_pyscf_pbc_ump2() {
    let cell = cell_mult(&PENTA_ATOMS, TRI_A, 2);
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    // SCF density_conv 1e-10 (prototype: conv 1e-16, ours − PySCF 1.2e-12).
    pinned_case("penta one-s doublet", &cell, &prep, &PENTA_REF, 1e-10);
}

#[test]
fn h2_631g_triplet_matches_pinned_pyscf_pbc_ump2() {
    let cell = cell_mult(&H2_ATOMS, cubic(4.0), 3);
    let prep = prep_for(&cell, &pyscf_631g_h());
    pinned_case(
        "H2/6-31G a=4 triplet",
        &cell,
        &prep,
        &H2_631G_TRIPLET_REF,
        1e-10,
    );
}

/// If this exceeds ~60 s in release, mark it
/// `#[ignore = "slow: tri 4H s+p dense AFT + 3 UHF; run with --release -- --ignored"]`
/// (pbc_uhf.rs runs the same dense build un-ignored).
#[test]
fn tri_4h_sp_triplet_matches_pinned_pyscf_pbc_ump2() {
    let cell = cell_mult(&TRI_ATOMS, TRI_A, 3);
    let prep = prep_for(&cell, &sp_basis_h());
    // PySCF-vs-prototype scatter 4.5e-10 (SCF orbitals, not the formula).
    pinned_case(
        "tri 4H s+p triplet",
        &cell,
        &prep,
        &TRI_SP_TRIPLET_REF,
        1e-9,
    );
}

/// The empty-β-channel B path (N_β = 0) through both parts entries, with a
/// REAL aux (cc-pvdz-ri): the fit error must be small and underbinding.
/// Prototype (its own aux set, not necessarily this one): URPA +4.7e-5 /
/// +1.8e-5 (unshifted / shifted), 1.5e-3 / 1.6e-3 relative.
#[test]
fn h2_631g_triplet_empty_beta_channel_runs_through_the_b_path() {
    let cell = cell_mult(&H2_ATOMS, cubic(4.0), 3);
    let prep = prep_for(&cell, &pyscf_631g_h());
    let hc = hcore(&cell, &prep);
    let eri = dense_none(&cell, &prep, &hc);
    let aux = PreparedBasis::new(cell.mol(), &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let gdf = RsGdf::build(
        &cell,
        &prep,
        &aux,
        &hc.s,
        &RsGdfConfig {
            omega: 1.0,
            exxdiv: ExxDiv::None,
            budget_bytes: Some(AMPLE),
            ..Default::default()
        },
    )
    .unwrap();
    let u = uhf_dense(&cell, &prep, &hc, &eri, ExxDiv::None);
    let q = DEFAULT_GAMMA_DRPA_QUAD_POINTS;
    for den in DENS {
        let md = gamma_ump2(
            &cell,
            &u,
            GammaMp2Integrals::DenseAft(&eri),
            &mcfg(ExxDiv::None, den),
        )
        .unwrap();
        let mb = gamma_ump2(
            &cell,
            &u,
            GammaMp2Integrals::RsGdf(&gdf),
            &mcfg(ExxDiv::None, den),
        )
        .unwrap();
        let rd = gamma_urpa(
            &cell,
            &u,
            GammaDrpaIntegrals::DenseAft(&eri),
            &rcfg(ExxDiv::None, den, q),
        )
        .unwrap();
        let rb = gamma_urpa(
            &cell,
            &u,
            GammaDrpaIntegrals::RsGdf(&gdf),
            &rcfg(ExxDiv::None, den, q),
        )
        .unwrap();
        eprintln!(
            "{den:?}: UMP2 {:.10e} fit {:+.2e}; URPA {:.10e} fit {:+.2e} (rel {:+.2e})",
            md.mp2_corr,
            mb.mp2_corr - md.mp2_corr,
            rd.rpa_corr,
            rb.rpa_corr - rd.rpa_corr,
            (rb.rpa_corr - rd.rpa_corr) / rd.rpa_corr
        );
        assert_eq!(mb.nocc_active, [2, 0]);
        assert_eq!(mb.components.e_bb, 0.0);
        assert_eq!(mb.components.e_ab, 0.0);
        let d = rb.rpa_corr - rd.rpa_corr;
        assert!(
            d > 0.0 && d < 2e-4 && (d / rd.rpa_corr).abs() < 5e-3,
            "URPA fit error {d:e}"
        );
        let dm = mb.mp2_corr - md.mp2_corr;
        // The fit error's SIGN is not fixed for UMP2: measured -7.6e-7
        // (1.3e-3 relative) here, i.e. slight overbinding, unlike the
        // closed-shell MP2 rows. Bound the magnitude only.
        assert!(dm.abs() < 1e-2 * md.mp2_corr.abs(), "UMP2 fit error {dm:e}");
    }
}

// ===========================================================================
// (e) Box limit vs ferric's OWN molecular UMP2/URPA (dense oracles on both
//     sides: exact libint2 ERIs molecularly, pure AFT periodically).
// ===========================================================================

/// Molecular UHF (C, ε per spin) and the exact UMP2 / URPA on dense ERIs.
fn molecular_u_corr(mol: &Molecule, bs: &BasisSet, na: usize, nb: usize) -> (ScfResult, f64, f64) {
    let prep = PreparedBasis::new(mol, bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let m = solve_uhf(
        &ParallelContext::default(),
        mol,
        &prep,
        &bounds,
        &RhfConfig {
            use_sad_guess: false,
            density_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(m.converged);
    let n = prep.nbasis();
    let ao = Array2::from_shape_vec((n * n, n * n), dense_ao_eri(&prep, op).unwrap()).unwrap();
    let (ca, cb) = (m.mos_a(), m.mos_b());
    let blk = |c1: &Array2<f64>, n1: usize, c2: &Array2<f64>, n2: usize| {
        ovov_from_dense_pair(
            &ao,
            n,
            c1.slice(s![.., ..n1]),
            c1.slice(s![.., n1..]),
            c2.slice(s![.., ..n2]),
            c2.slice(s![.., n2..]),
        )
        .unwrap()
    };
    let (aa, bb, ab) = (
        blk(ca, na, ca, na),
        blk(cb, nb, cb, nb),
        blk(ca, na, cb, nb),
    );
    let e = u_eps(&m, na, nb, 0.0);
    let mp2 = ump2_from_ovov(&aa, &bb, &ab, &e[0], &e[1], &e[2], &e[3])
        .unwrap()
        .e_total;
    let rpa = urpa_plasmon(&aa, &bb, &ab, &e[0], &e[1], &e[2], &e[3]).unwrap();
    (m, mp2, rpa)
}

/// Periodic Gamma UHF (ewald, staged, from the molecular MOs) in a cubic box.
fn box_uhf(
    atoms_mol: &Molecule,
    bs: &BasisSet,
    a: f64,
    init: &ScfResult,
    prec: f64,
) -> (Cell, ScfResult, DenseAftEri) {
    let cell = Cell::new(atoms_mol.clone(), cubic(a)).unwrap();
    let prep = prep_for(&cell, bs);
    let hcfg = PeriodicHcoreConfig {
        precision: prec,
        ..PeriodicHcoreConfig::with_omega(0.9)
    };
    let hc = periodic_hcore(&cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        prec,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let cfg = GammaUhfConfig {
        scf: uhf_scf(),
        initial_mos: Some((init.mos_a().clone(), init.mos_b().clone())),
        ..Default::default()
    };
    let u = gamma_uhf(&cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &cfg).unwrap();
    assert_eq!(u.scf.spin, Spin::Unrestricted);
    (cell, u.scf, eri)
}

#[test]
#[ignore = "slow: H/6-31G pure-AFT at a = 20 (6-31G core exponent 18.7 -> ~1.8e7 half-sphere G at \
            precision 1e-12, est. 0.5-2 min release; the STO-3G H atom in pbc_uhf.rs is ~7x fewer G). \
            Run with --release -- --ignored, serially; un-ignore if it measures < 60 s"]
fn h_atom_box_ump2_is_zero_and_urpa_is_c3_plus_second_order_c6() {
    let basis = pyscf_631g_h();
    let mut mol = hydrogens(&[[0.3, 0.2, 0.1]]);
    mol.multiplicity = 2;
    let (m, e_mp2, e_rpa) = molecular_u_corr(&mol, &basis, 1, 0);
    assert!(e_mp2.abs() < 1e-15, "molecular one-electron UMP2 {e_mp2:e}");
    let a = 20.0;
    let t0 = std::time::Instant::now();
    let (cell, u, eri) = box_uhf(&mol, &basis, a, &m, 1e-12);
    let mp = gamma_ump2(
        &cell,
        &u,
        GammaMp2Integrals::DenseAft(&eri),
        &mcfg(ExxDiv::Ewald, Shifted),
    )
    .unwrap();
    let rp = gamma_urpa(
        &cell,
        &u,
        GammaDrpaIntegrals::DenseAft(&eri),
        &rcfg(ExxDiv::Ewald, Shifted, DEFAULT_GAMMA_DRPA_QUAD_POINTS),
    )
    .unwrap();
    let d = rp.rpa_corr - e_rpa;
    let a3 = a * a * a;
    eprintln!(
        "H/6-31G a = {a}: UMP2 {:e}; URPA mol {e_rpa:.12e}, dE*a^3 {:.6} (c3 {H_URPA_C3}, \
         c3 + c6/a^3 {:.6}); half-G {} ({:.1} s)",
        mp.mp2_corr,
        d * a3,
        H_URPA_C3 + H_URPA_C6 / a3,
        eri.n_g_half(),
        t0.elapsed().as_secs_f64()
    );
    assert!(mp.mp2_corr.abs() < 1e-15, "one-electron UMP2 must vanish");
    close(
        d * a3,
        H_URPA_C3 + H_URPA_C6 / a3,
        3e-5 * H_URPA_C3,
        "(E_URPA - E_mol) a^3",
    );
    // c6 is needed: the predictor's second order is real (prototype 2.1e-3 of c3).
    assert!((d * a3 - H_URPA_C3).abs() > 1e-3 * H_URPA_C3);
}

#[test]
#[ignore = "infeasible on today's Rust builders: NH/6-31G pure AFT at a = 32/40 needs ~1e11 \
            half-sphere G (N core exponent 4173; the G list alone is refused by the memory \
            budget), and RS-GDF with a real aux adds a ~1e-5 Ha fit error, the size of the a = 40 \
            residual (1.2e-5 Ha). The prototype used a range-separated EXACT build (pbc_gamma w < 1) \
            that ferric-pbc does not have yet. Kept as the executable spec for when it does"]
fn nh_triplet_box_limit_is_a3_with_the_predicted_c3_and_per_spin_shift() {
    let basis = basis::bundled("6-31g").unwrap();
    let mol = Molecule {
        atoms: vec![
            Atom {
                symbol: "N".into(),
                z: 7,
                x: 0.3,
                y: 0.2,
                zpos: 0.1,
                ghost: false,
                n_core_ecp: 0,
            },
            h_atom([0.3, 0.2, 0.1 + 1.95]),
        ],
        charge: 0,
        multiplicity: 3,
    };
    let (na, nb) = (5usize, 3usize);
    let (m, e_mp2, e_rpa) = molecular_u_corr(&mol, &basis, na, nb);
    let mut res = Vec::new();
    for a in [32.0, 40.0] {
        let (cell, u, eri) = box_uhf(&mol, &basis, a, &m, 1e-11);
        let vm = madelung_constant(&cell).unwrap();
        let dm = GammaMp2Integrals::DenseAft(&eri);
        let dr = GammaDrpaIntegrals::DenseAft(&eri);
        let q = DEFAULT_GAMMA_DRPA_QUAD_POINTS;
        let sm = gamma_ump2(&cell, &u, dm, &mcfg(ExxDiv::Ewald, Shifted))
            .unwrap()
            .mp2_corr
            - e_mp2;
        let sr = gamma_urpa(&cell, &u, dr, &rcfg(ExxDiv::Ewald, Shifted, q))
            .unwrap()
            .rpa_corr
            - e_rpa;
        for (name, da, db) in [
            ("v_M/2 per spin", 0.5 * vm, 0.5 * vm),
            ("alpha-only", 0.0, vm),
        ] {
            let mut bad = u.clone();
            for e in bad.eps_alpha.iter_mut().take(na) {
                *e += da;
            }
            for e in bad.eps_beta.as_mut().unwrap().iter_mut().take(nb) {
                *e += db;
            }
            let bm = gamma_ump2(&cell, &bad, dm, &mcfg(ExxDiv::Ewald, Shifted))
                .unwrap()
                .mp2_corr
                - e_mp2;
            eprintln!(
                "a = {a}: mutant {name} UMP2 dE*a {:+.5} (shifted {:+.2e})",
                bm * a,
                sm * a
            );
            // Prototype: -0.0545 / -0.0439, a flat 1/a plateau.
            assert!((bm * a).abs() > 0.03, "{name}: no 1/a plateau");
        }
        res.push((a, sm, sr));
    }
    let (a1, a2) = (res[0].0, res[1].0);
    for (k, (label, c3)) in [("UMP2", NH_C3_UMP2), ("URPA", NH_C3_URPA)]
        .into_iter()
        .enumerate()
    {
        let (d1, d2) = if k == 0 {
            (res[0].1, res[1].1)
        } else {
            (res[0].2, res[1].2)
        };
        // dE = c3 a^-3 + c5 a^-5 through the two boxes.
        let (x1, y1, x2, y2) = (a1.powi(-3), a1.powi(-5), a2.powi(-3), a2.powi(-5));
        let fit = (d1 * y2 - d2 * y1) / (x1 * y2 - x2 * y1);
        let p = (d1 / d2).ln() / (a2 / a1).ln();
        eprintln!("{label}: c3 fit {fit:.6} (predicted {c3}), local exponent {p:.4}");
        assert!((fit - c3).abs() < 3e-3 * c3.abs(), "{label} c3");
        assert!((p - 3.0).abs() < 0.05, "{label} exponent");
    }
}

// ===========================================================================
// (f) Molecular paths: the new parts entries ARE the molecular kernels.
// ===========================================================================

#[test]
fn u_parts_entries_reproduce_the_molecular_drivers_bitwise() {
    let mut mol = hydrogens(&[[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 0.3, 3.9]]);
    mol.multiplicity = 2;
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let u = solve_uhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        &bounds,
        &RhfConfig {
            density_conv: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(u.converged && matches!(u.spin, Spin::Unrestricted));
    let cfg = pdep_config(DEFAULT_GAMMA_DRPA_QUAD_POINTS, AMPLE);
    let mp2_cfg = RiMp2Config {
        frozen_core: cfg.frozen_core,
        memory_budget_bytes: cfg.memory_budget_bytes,
        ..Default::default()
    };
    let ia = compute_rpa_intermediates_spin(&mol, &obs, &dfbs, op, &u, &mp2_cfg, true).unwrap();
    let ib = compute_rpa_intermediates_spin(&mol, &obs, &dfbs, op, &u, &mp2_cfg, false).unwrap();
    assert_eq!((ia.nocc, ib.nocc), (2, 1));

    // UMP2.
    let mol_m = u_ri_mp2(&mol, &obs, &dfbs, op, &u, &mp2_cfg).unwrap();
    let parts_m = u_ri_mp2_from_parts(&ia, &ib, u.eps_a(), u.eps_b()).unwrap();
    eprintln!(
        "H3/cc-pVDZ UMP2: molecular {:.15e} (aa {:.3e} bb {:.3e} ab {:.3e}), parts {:.15e}",
        mol_m.mp2_corr,
        mol_m.components.e_aa,
        mol_m.components.e_bb,
        mol_m.components.e_ab,
        parts_m.e_total
    );
    assert_eq!(parts_m.e_aa.to_bits(), mol_m.components.e_aa.to_bits());
    assert_eq!(parts_m.e_bb.to_bits(), mol_m.components.e_bb.to_bits());
    assert_eq!(parts_m.e_ab.to_bits(), mol_m.components.e_ab.to_bits());
    assert_eq!(
        parts_m.e_total.to_bits(),
        mol_m.components.e_total.to_bits()
    );
    // N_β = 1: ββ is zero up to GEMM (ia|ib) vs (ib|ia) roundoff.
    assert!(parts_m.e_aa < -1e-6 && parts_m.e_bb.abs() < 1e-14 && parts_m.e_ab < -1e-6);

    // URPA.
    let (eoa, eva) = (
        &u.eps_a()[..ia.nocc],
        &u.eps_a()[ia.nocc..ia.nocc + ia.nvir],
    );
    let (eob, evb) = (
        &u.eps_b()[..ib.nocc],
        &u.eps_b()[ib.nocc..ib.nocc + ib.nvir],
    );
    let mol_r = run_u_pdep_rpa(&mol, &obs, &dfbs, op, &u, &cfg).unwrap();
    let parts_r = run_u_pdep_rpa_from_parts(&ia, &ib, eoa, eva, eob, evb, &cfg).unwrap();
    eprintln!(
        "H3/cc-pVDZ URPA: molecular {:.15e}, parts {:.15e}",
        mol_r.e_rpa, parts_r.e_rpa
    );
    assert_eq!(parts_r.e_rpa.to_bits(), mol_r.e_rpa.to_bits());
    assert_eq!(parts_r.n_eigenpotentials, mol_r.n_eigenpotentials);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&parts_r.eigenvalues_static),
        bits(&mol_r.eigenvalues_static)
    );

    // Config honesty on the U parts entry.
    let refuse = |a: &RpaIntermediates,
                  b: &RpaIntermediates,
                  e: [&[f64]; 4],
                  c: &ferric_rpa::PdepRpaConfig,
                  what: &str| {
        let msg = run_u_pdep_rpa_from_parts(a, b, e[0], e[1], e[2], e[3], c)
            .expect_err(what)
            .to_string();
        eprintln!("  refused ({what}): {msg}");
        msg
    };
    let e4 = [eoa, eva, eob, evb];
    let mut c = cfg.clone();
    c.trunc_thresh = 1e-4;
    refuse(&ia, &ib, e4, &c, "trunc_thresh > 0");
    let mut c = cfg.clone();
    c.eigensolver = Eigensolver::Davidson;
    refuse(&ia, &ib, e4, &c, "Davidson");
    let mut c = cfg.clone();
    c.chi0_sparsity = Chi0Sparsity::BoysScreened {
        thresh: 1e-6,
        dist_cutoff: 10.0,
    };
    refuse(&ia, &ib, e4, &c, "Boys sparsity");
    refuse(
        &ia,
        &ib,
        [eoa, eva, &u.eps_b()[..2], evb],
        &cfg,
        "wrong beta occupied count",
    );
    let gapless = vec![eva[0]; ia.nocc];
    refuse(
        &ia,
        &ib,
        [gapless.as_slice(), eva, eob, evb],
        &cfg,
        "zero alpha gap",
    );
    // naux mismatch: a typed error (was a debug_assert on the molecular path).
    let short = RpaIntermediates {
        b_ov: ib.b_ov.slice(s![..ib.naux - 1, ..]).to_owned(),
        v_inv_sqrt: ib.v_inv_sqrt.slice(s![.., ..ib.naux - 1]).to_owned(),
        naux: ib.naux - 1,
        ..ib.clone()
    };
    let msg = refuse(&ia, &short, e4, &cfg, "naux mismatch");
    assert!(msg.contains("different aux dimensions"), "{msg}");
    assert!(u_ri_mp2_from_parts(&ia, &short, u.eps_a(), u.eps_b()).is_err());
}

// ===========================================================================
// Refusals of the drivers.
// ===========================================================================

#[test]
fn gamma_u_drivers_reject_bad_inputs() {
    let cell = cell_mult(&H2_ATOMS, cubic(4.0), 3);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc = hcore(&cell, &prep);
    let eri = dense_none(&cell, &prep, &hc);
    let u = uhf_dense(&cell, &prep, &hc, &eri, ExxDiv::None);
    let dm = GammaMp2Integrals::DenseAft(&eri);
    let dr = GammaDrpaIntegrals::DenseAft(&eri);
    let q = DEFAULT_GAMMA_DRPA_QUAD_POINTS;
    // H2/STO-3G triplet: α has no virtual, β no electron → no ov pair at all.
    assert!(gamma_urpa(&cell, &u, dr, &rcfg(ExxDiv::None, Shifted, q)).is_err());
    let m = gamma_ump2(&cell, &u, dm, &mcfg(ExxDiv::None, Shifted)).unwrap();
    assert_eq!(m.mp2_corr, 0.0);
    // Frozen core with N_β = 0: active_occ refuses (no silent underflow).
    let mut fc = mcfg(ExxDiv::None, Shifted);
    fc.frozen_core = 1;
    assert!(gamma_ump2(&cell, &u, dm, &fc).is_err());
    // Restricted and unconverged references.
    let closed = h2_cell(4.0);
    let r = gamma_rhf(&closed, &prep, &hc, &eri);
    let msg = gamma_ump2(&closed, &r, dm, &mcfg(ExxDiv::None, Shifted))
        .unwrap_err()
        .to_string();
    assert!(msg.contains("restricted reference"), "{msg}");
    let mut bad = as_unrestricted(&r);
    bad.converged = false;
    assert!(gamma_urpa(&closed, &bad, dr, &rcfg(ExxDiv::None, Shifted, q)).is_err());
    // Grid outside [8, 1024] is refused, not clamped.
    let good = as_unrestricted(&r);
    for bad_q in [0, 7, 1025] {
        let msg = gamma_urpa(&closed, &good, dr, &rcfg(ExxDiv::None, Shifted, bad_q))
            .unwrap_err()
            .to_string();
        assert!(msg.contains("quad_points"), "{msg}");
    }
    // A tiny budget names the quantity.
    let msg = gamma_ump2(
        &closed,
        &good,
        dm,
        &GammaMp2Config {
            budget_bytes: Some(1),
            ..mcfg(ExxDiv::None, Shifted)
        },
    )
    .unwrap_err()
    .to_string();
    assert!(msg.contains("Gamma UMP2 dense"), "{msg}");
}
