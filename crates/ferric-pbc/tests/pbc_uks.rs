//! Stage 5: Gamma-point UKS (`ferric_pbc::dft::gamma_uks` over
//! `ferric_scf::uhf::solve_uhf_injected` with a spin-polarized
//! `XcBuilder`), ported from `reference/pbc/pbc_uks.py` (FINDINGS
//! "Iteration 10 (Python, Gamma UKS)").
//!
//! Per spin σ, `a` = the global exact-exchange fraction:
//!
//! ```text
//! F_σ = h + J[D_α + D_β] − a·(K[D_σ] + v_M S D_σ S) + V_σ[ρ_α, ρ_β]
//! ```
//!
//! # What is independent of what
//!
//! * (1) closed shell UKS ≡ RKS (`gamma_rks`, the Stage-2 path) on the same
//!   grid, LDA/PBE/PBE0 × none/ewald — prototype ≤ 1.8e-15; asserted 1e-11
//!   (the FINDINGS port tolerance: two SCF trajectories, one starting from
//!   the β HOMO/LUMO-mixed guess). Negative control: a per-spin half
//!   Madelung (RKS's ½ carried into the per-spin K) sits exactly
//!   `a v_M N/4` high.
//! * (2) UKS with an `a = 1`, zero-XC builder ≡ `gamma_uhf` (the Stage-4
//!   path), 1e-12 — same loop, so this pins the `c_k = a` / `V_σ` plumbing
//!   to the HF limit, not the kernel.
//! * (3) Open-shell pins (H atom doublet, H2 triplet, tri 4H s+p triplet;
//!   LDA/PBE/PBE0; exxdiv ewald; ⟨S²⟩): PySCF's own (50, 146) A1 grid is
//!   NOT reproducible here (TA-M4 + SSF/Becke over images is ferric's
//!   construction), so the tight pins are the PROTOTYPE'S OWN construction
//!   (`pbc_dft.PeriodicGrid(cell, 75, 302, D=10, scheme="ssf")`, dense pure
//!   AFT, `pbc_uks.uks`, generated 2026-09-24), point counts included; the
//!   PySCF pins (FINDINGS It. 10 table, same state) are asserted only to the
//!   grid-construction difference. Plus the identity
//!   `E_ewald − E_none = −a v_M N/2` at the same density.
//! * (4) `tr(V_σ dD) == d E_xc / dε` along `D_σ + ε dD`, per spin, on a
//!   density with BOTH spins populated and `ρ_α ≠ ρ_β` — the ONLY test that
//!   sees a dropped `σ_αβ` cross term or β fed the α potential (both survived
//!   every energy test in the prototype). Plus `eval_polarized(D/2, D/2) ≡
//!   eval(D)`.
//! * (5) Box limit, H/STO-3G PBE0: `(E_ewald − E_mol) a³ = c3 = −(2π/3)·a·Ω_α
//!   = −1.020270` (0.25 × the UHF −4.081081; one AO → no relaxation), vs
//!   ferric's MOLECULAR UKS on the identical grid rules — independent of
//!   every periodic piece. LDA: no a⁻³.
//! * (6) Ewald trap from the OCCUPATIONS: a hole state gives a NEGATIVE
//!   occupation-aware gap where the sorted-eigenvalue gap stays positive;
//!   the `a = 1` builder's single-stage ewald run lands in the Iteration-6
//!   trap (−1.812958714837) and is flagged, the staged run is not.
//!
//! # Mutation plan (each must turn ≥ 1 test red)
//!
//! * `solve_uhf_impl`: `c_k = 0.5 * k_mix.sr` on the injected path (RKS's ½
//!   per spin) → (1) PBE0, (3) PBE0 pins.
//! * `solve_uhf_impl`: `c_k = 1.0` for the injected XC (K unscaled) → (1),
//!   (3), (5) (4× c3).
//! * `vxc.rs::semilocal_vxc_polarized_scratch`: drop the `vsigma_cross`
//!   term, or build V_β from `vrho_a` → (4) (energy tests are blind).
//! * `PeriodicXc::eval_polarized`: use the unpolarized kernel on D_α + D_β
//!   for both spins → (3) triplet pins (prototype +0.09..0.13 Ha), (4).
//! * `gamma_uks_with_xc`: `shift = applied` (not `a·applied`) → (6) is
//!   unaffected (a = 1) but the closed-shell PBE0 runs of (1) warn; `staged`
//!   ignoring `a > 0` changes nothing numerically (documented choice).
//! * `occupation_gaps`: classify by sorted index instead of `n_i` → (6) hole
//!   state.
//!
//! # Molecular path unchanged
//!
//! Every new branch in `solve_uhf_impl` is gated on `inj_xc.is_some()`
//! (None on the molecular path); `XcBuilder` gained two DEFAULTED methods, so
//! existing implementors compile unchanged.

mod common;

use common::*;
use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::dft::{
    gamma_rks, gamma_uks, gamma_uks_with_xc, GammaRksConfig, GammaUksConfig, GammaUksResult,
    PeriodicGrid, PeriodicGridConfig, PeriodicXc, PeriodicXcConfig,
};
use ferric_pbc::ewald::madelung_constant;
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::uhf::{
    gamma_uhf, occupation_gaps, spin_gaps, EwaldStart, GammaUhfConfig, GammaUhfIntegrals,
};
use ferric_scf::rhf::{PeriodicInjection, RhfConfig, XcBuilder};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_injected};
use ndarray::{Array2, Axis};
use ndarray_linalg::{Eigh, UPLO};
use std::f64::consts::PI;

const HCORE_OMEGA: f64 = 0.8;
/// `test_prototype.py` H_ATOM (Bohr).
const H_ATOM: [[f64; 3]; 1] = [[0.3, 0.2, 0.1]];
const XCS: [&str; 3] = ["LDA", "PBE", "PBE0"];

// --- (3) Prototype own construction: pbc_uks.uks on
// pbc_dft.PeriodicGrid(cell, 75, 302, D=10.0, scheme="ssf"), dense pure-AFT I,
// exxdiv ewald (staged for tri), conv 1e-12. [LDA, PBE, PBE0] (prototype
// "LDA,VWN" == ferric "LDA"). Generated 2026-09-24.
const H_SSF_NPTS: usize = 19116;
const H_SSF_E: [f64; 3] = [-0.667583328116, -0.677242874711, -0.699779179056];
const H2T_SSF_NPTS: usize = 36234;
const H2T_SSF_E: [f64; 3] = [-0.261405923197, -0.307316273095, -0.331878659724];
const TRI_SSF_NPTS: usize = 70784;
const TRI_SSF_E: [f64; 3] = [-1.694518592905, -1.731614505391, -1.777304384785];
const TRI_SSF_S2: [f64; 3] = [2.0003320553, 2.0005420296, 2.0006810776];

// --- (3) PySCF 2.13 pbc.dft.UKS pins (FINDINGS It. 10, PySCF A1 (50,146)
// grid — a DIFFERENT grid construction; asserted to the construction gap).
const H_PYSCF_E: [f64; 3] = [-0.668127813328, -0.677787718838, -0.700202408672];
const H2T_PYSCF_E: [f64; 3] = [-0.261697156130, -0.307603027052, -0.332094904636];
const TRI_PYSCF_E: [f64; 3] = [-1.694673507916, -1.731772052782, -1.777429192569];
/// Max |ours(SSF 75x302) − PySCF(A1 50x146)| over the table is 5.45e-4
/// (H atom PBE; H2 triplet 2.9e-4, tri 1.6e-4); the bound leaves ~2× room.
const GRID_CONSTRUCTION_GAP: f64 = 1.1e-3;

// --- Stage 4 pins reused by (2)/(6) (FINDINGS It. 6, PySCF pbc.scf.UHF).
const TRI_TRIPLET_UHF_E_NONE: f64 = -0.583125965387;
const TRI_TRIPLET_UHF_E_EWALD: f64 = -1.827999723359;
const TRI_TRIPLET_UHF_E_EWALD_TRAPPED: f64 = -1.812958714837;

// --- (5) Box limit (FINDINGS It. 10 table; Ω_α of the single STO-3G AO).
const H_OMEGA_A: f64 = 1.948573;
const H_C3_PBE0: f64 = -1.020270;

struct Setup {
    cell: Cell,
    prep: PreparedBasis,
    bs: BasisSet,
    hc: PeriodicHcore,
    eri: DenseAftEri,
}

fn cell_mult(atoms: &[[f64; 3]], lattice: [[f64; 3]; 3], mult: usize) -> Cell {
    let mut mol: Molecule = hydrogens(atoms);
    mol.multiplicity = mult;
    Cell::new(mol, lattice).expect("cell")
}

fn setup(cell: Cell, bs: BasisSet) -> Setup {
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(HCORE_OMEGA)).unwrap();
    // exxdiv is taken from the gamma_* configs by the builders, not the tensor.
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        DEFAULT_DENSE_AFT_PRECISION,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    Setup {
        cell,
        prep,
        bs,
        hc,
        eri,
    }
}

fn ssf_grid(n_rad: usize, n_ang: usize, d: f64) -> PeriodicGridConfig {
    PeriodicGridConfig {
        neighbour_cutoff: Some(d),
        ..PeriodicGridConfig::with_size(n_rad, n_ang)
    }
}

fn ukscfg(xc: &str, grid: PeriodicGridConfig, exx: ExxDiv, start: EwaldStart) -> GammaUksConfig {
    GammaUksConfig {
        grid,
        exxdiv: exx,
        ewald_start: start,
        ..GammaUksConfig::new(xc)
    }
}

fn uks(su: &Setup, cfg: &GammaUksConfig) -> GammaUksResult {
    let r = gamma_uks(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        cfg,
    )
    .unwrap_or_else(|e| panic!("gamma_uks {}: {e}", cfg.functional));
    assert!(r.scf.converged);
    r
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    eprintln!(
        "  {what}: {got:.12} (want {want:.12}, diff {:.2e})",
        got - want
    );
    assert!(got.is_finite(), "{what}: non-finite");
    assert!(
        (got - want).abs() < tol,
        "{what}: {got:.12e} vs {want:.12e} (tol {tol:.1e})"
    );
}

/// `a = frac`, zero semilocal XC: with `frac = 1` the UKS loop IS the UHF.
struct ZeroXc {
    frac: f64,
}

impl XcBuilder for ZeroXc {
    fn build(&mut self, d: &Array2<f64>) -> Result<(f64, Array2<f64>), FerricError> {
        Ok((0.0, Array2::zeros(d.dim())))
    }
    fn exact_exchange_fraction(&self) -> f64 {
        self.frac
    }
    fn build_polarized(
        &mut self,
        d_a: &Array2<f64>,
        _d_b: &Array2<f64>,
    ) -> Result<(f64, Array2<f64>, Array2<f64>), FerricError> {
        Ok((0.0, Array2::zeros(d_a.dim()), Array2::zeros(d_a.dim())))
    }
    fn supports_polarized(&self) -> bool {
        true
    }
}

fn uks_with(su: &Setup, xc: &mut dyn XcBuilder, exx: ExxDiv, start: EwaldStart) -> GammaUksResult {
    let cfg = ukscfg("unused", PeriodicGridConfig::default(), exx, start);
    let r = gamma_uks_with_xc(
        &su.cell,
        &su.prep,
        &su.hc,
        GammaUhfIntegrals::DenseAft(&su.eri),
        xc,
        &cfg,
    )
    .expect("gamma_uks_with_xc");
    assert!(r.scf.converged);
    r
}

const EXX: [ExxDiv; 2] = [ExxDiv::None, ExxDiv::Ewald];

// ===========================================================================
// (1) Closed shell UKS ≡ RKS.
// ===========================================================================

#[test]
fn closed_shell_uks_equals_rks_lda_pbe_pbe0_both_exxdiv() {
    let cases: [(&str, Setup, &[&str], &[ExxDiv]); 2] = [
        ("H2 a=4", setup(h2_cell(4.0), pyscf_sto3g_h()), &XCS, &EXX),
        (
            "tri 4H s+p",
            setup(triclinic_cell(), sp_basis_h()),
            &["PBE", "PBE0"],
            &[ExxDiv::Ewald],
        ),
    ];
    for (label, su, xcs, exxs) in cases {
        let grid = ssf_grid(50, 110, 10.0);
        let vm = madelung_constant(&su.cell).unwrap();
        let n = su.cell.mol().nelec() as f64;
        for &xc in xcs {
            for &exx in exxs {
                let mut rcfg = GammaRksConfig::new(xc);
                rcfg.grid = grid.clone();
                rcfg.exxdiv = exx;
                let r = gamma_rks(
                    &su.cell,
                    &su.prep,
                    &su.hc,
                    GammaUhfIntegrals::DenseAft(&su.eri),
                    &rcfg,
                )
                .unwrap();
                let u = uks(&su, &ukscfg(xc, grid.clone(), exx, EwaldStart::Direct));
                eprintln!(
                    "{label} {xc} {exx:?}: E_rks {:.14} E_uks {:.14} dE {:.2e} <S2> {:.1e} \
                     dExc {:.1e}",
                    r.scf.energy,
                    u.scf.energy,
                    u.scf.energy - r.scf.energy,
                    u.s2,
                    u.e_xc - r.e_xc
                );
                assert!(
                    (u.scf.energy - r.scf.energy).abs() < 1e-11,
                    "{label} {xc} {exx:?}: dE {:.3e}",
                    u.scf.energy - r.scf.energy
                );
                assert!(u.s2.abs() < 1e-10, "{label} {xc}: <S2> {}", u.s2);
                assert!((u.e_xc - r.e_xc).abs() < 1e-10, "{label} {xc}: E_xc");
                assert!(u.gaps.satisfied(), "{label} {xc}: {:?}", u.gaps);
                // Staged (the default for a > 0) reaches the same state.
                if exx == ExxDiv::Ewald && xc == "PBE0" {
                    let st = uks(&su, &ukscfg(xc, grid.clone(), exx, EwaldStart::Staged));
                    let none = st.none_stage.as_ref().expect("staged for a > 0");
                    close(st.scf.energy, r.scf.energy, 1e-11, "staged PBE0 vs RKS");
                    close(
                        st.scf.energy - none.energy,
                        -0.25 * vm * n / 2.0,
                        1e-10,
                        "staged ewald - none (= -a v_M N/2)",
                    );
                }
                if exx == ExxDiv::Ewald && xc == "PBE" {
                    assert!(
                        uks(&su, &ukscfg(xc, grid.clone(), exx, EwaldStart::Staged))
                            .none_stage
                            .is_none(),
                        "a = 0: no K, so no staging"
                    );
                }
            }
        }
        // Negative control: RKS's per-electron ½ carried into the per-spin
        // Madelung (K builder with v_M/2) moves PBE0 by exactly +a v_M N/4.
        let mut rcfg = GammaRksConfig::new("PBE0");
        rcfg.grid = grid.clone();
        let rks_e = gamma_rks(
            &su.cell,
            &su.prep,
            &su.hc,
            GammaUhfIntegrals::DenseAft(&su.eri),
            &rcfg,
        )
        .unwrap()
        .scf
        .energy;
        let g = PeriodicGrid::build(&su.cell, &grid).unwrap();
        let pxc =
            PeriodicXc::new(&su.cell, &su.bs, "PBE0", &g, &PeriodicXcConfig::default()).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &su.prep).unwrap();
        let inj = PeriodicInjection {
            s: su.hc.s.clone(),
            h: su.hc.h.clone(),
            vnn: su.hc.enn,
            j: Box::new(su.eri.j_builder()),
            k: Box::new(su.eri.k_builder_with_madelung(0.5 * vm)),
            xc: Some(Box::new(pxc)),
        };
        let mutant = solve_uhf_injected(
            &ParallelContext::default(),
            su.cell.mol(),
            &su.prep,
            &bounds,
            &GammaUksConfig::new("PBE0").scf,
            inj,
            None,
        )
        .unwrap();
        close(
            mutant.energy - rks_e,
            0.25 * vm * n / 4.0,
            1e-9,
            &format!("{label} half-Madelung mutant - RKS (= +a v_M N/4)"),
        );
    }
}

// ===========================================================================
// (2) a = 1, zero XC ≡ gamma_uhf.
// ===========================================================================

#[test]
fn uks_with_an_hf_equivalent_builder_is_gamma_uhf() {
    for (label, su) in [
        (
            "H2 a=4 triplet",
            setup(cell_mult(&H2_ATOMS, cubic(4.0), 3), pyscf_sto3g_h()),
        ),
        (
            "tri 4H s+p triplet",
            setup(cell_mult(&TRI_ATOMS, TRI_A, 3), sp_basis_h()),
        ),
    ] {
        for exx in EXX {
            let ucfg = GammaUhfConfig {
                exxdiv: exx,
                ..Default::default()
            };
            let h = gamma_uhf(
                &su.cell,
                &su.prep,
                &su.hc,
                GammaUhfIntegrals::DenseAft(&su.eri),
                &ucfg,
            )
            .unwrap();
            let k = uks_with(&su, &mut ZeroXc { frac: 1.0 }, exx, EwaldStart::Staged);
            eprintln!(
                "{label} {exx:?}: E_uhf {:.14} E_uks(a=1, xc=0) {:.14} dE {:.2e} <S2> {:.10}",
                h.scf.energy,
                k.scf.energy,
                k.scf.energy - h.scf.energy,
                k.s2
            );
            assert!(
                (k.scf.energy - h.scf.energy).abs() < 1e-12,
                "{label} {exx:?}"
            );
            assert!((k.s2 - h.s2).abs() < 1e-12, "{label} {exx:?}: <S2>");
            assert_eq!(k.e_xc, 0.0);
            assert_eq!(k.none_stage.is_some(), h.none_stage.is_some());
            if label.starts_with("tri") && exx == ExxDiv::None {
                close(
                    k.scf.energy,
                    TRI_TRIPLET_UHF_E_NONE,
                    1e-9,
                    "tri none vs PySCF UHF",
                );
                // Aufbau state: occupation-aware gaps == sorted gaps.
                let occ = occupation_gaps(&k.scf, &su.hc.s, 0.0);
                let srt = spin_gaps(&k.scf, 3, 1, 0.0);
                for (o, s) in [(occ.gap_alpha, srt.gap_alpha), (occ.gap_beta, srt.gap_beta)] {
                    let (o, s) = (o.unwrap(), s.unwrap());
                    assert!((o - s).abs() < 1e-12 && o > 0.0, "aufbau gaps {o} vs {s}");
                }
            }
        }
    }
}

// ===========================================================================
// (3) Open-shell pins: prototype construction (tight) + PySCF (grid gap).
// ===========================================================================

#[test]
fn open_shell_uks_matches_the_prototype_construction_and_pyscf() {
    type Row = (
        &'static str,
        Cell,
        fn() -> BasisSet,
        (usize, usize),
        usize,
        [f64; 3],
        Option<[f64; 3]>,
        [f64; 3],
    );
    let rows: [Row; 3] = [
        (
            "H atom a=4 doublet",
            cell_mult(&H_ATOM, cubic(4.0), 2),
            pyscf_sto3g_h,
            (1, 0),
            H_SSF_NPTS,
            H_SSF_E,
            None,
            H_PYSCF_E,
        ),
        (
            "H2 a=4 triplet",
            cell_mult(&H2_ATOMS, cubic(4.0), 3),
            pyscf_sto3g_h,
            (2, 0),
            H2T_SSF_NPTS,
            H2T_SSF_E,
            None,
            H2T_PYSCF_E,
        ),
        (
            "tri 4H s+p triplet",
            cell_mult(&TRI_ATOMS, TRI_A, 3),
            sp_basis_h,
            (3, 1),
            TRI_SSF_NPTS,
            TRI_SSF_E,
            Some(TRI_SSF_S2),
            TRI_PYSCF_E,
        ),
    ];
    for (label, cell, basis, nocc, npts, e_ref, s2_ref, e_pyscf) in rows {
        let su = setup(cell, basis());
        let vm = madelung_constant(&su.cell).unwrap();
        let n = (nocc.0 + nocc.1) as f64;
        let s2_ideal = {
            let sz = 0.5 * (nocc.0 as f64 - nocc.1 as f64);
            sz * (sz + 1.0)
        };
        for (i, xc) in XCS.into_iter().enumerate() {
            let u = uks(
                &su,
                &ukscfg(
                    xc,
                    ssf_grid(75, 302, 10.0),
                    ExxDiv::Ewald,
                    EwaldStart::Staged,
                ),
            );
            eprintln!(
                "{label} {xc}: E {:.12} <S2> {:.10} a {} gaps {:?} N_grid {:?}",
                u.scf.energy, u.s2, u.exact_exchange_fraction, u.gaps, u.grid
            );
            assert_eq!(u.nocc, nocc);
            assert_eq!(u.grid.as_ref().unwrap().n_grid_points, npts, "{label}");
            close(
                u.scf.energy,
                e_ref[i],
                1e-8,
                &format!("{label} {xc} vs prototype"),
            );
            match s2_ref {
                Some(s2) => close(u.s2, s2[i], 1e-8, &format!("{label} {xc} <S2>")),
                None => close(u.s2, s2_ideal, 1e-12, &format!("{label} {xc} <S2>")),
            }
            assert!(u.gaps.satisfied(), "{label} {xc}: {:?}", u.gaps);
            let d_py = u.scf.energy - e_pyscf[i];
            eprintln!("  vs PySCF A1 (50,146): {d_py:+.3e}");
            assert!(
                d_py.abs() < GRID_CONSTRUCTION_GAP,
                "{label} {xc}: {d_py:.3e}"
            );
            // ewald − none at the same density: −a v_M N/2 (a = 0: exactly 0,
            // i.e. no Madelung on the semilocal part).
            let a = u.exact_exchange_fraction;
            match u.none_stage.as_ref() {
                Some(none) => close(
                    u.scf.energy - none.energy,
                    -a * vm * n / 2.0,
                    1e-10,
                    &format!("{label} {xc} ewald - none"),
                ),
                None => {
                    assert_eq!(a, 0.0);
                    let none = uks(
                        &su,
                        &ukscfg(
                            xc,
                            ssf_grid(75, 302, 10.0),
                            ExxDiv::None,
                            EwaldStart::Staged,
                        ),
                    );
                    close(
                        u.scf.energy - none.scf.energy,
                        0.0,
                        1e-12,
                        &format!("{label} {xc} ewald - none (a = 0)"),
                    );
                }
            }
        }
    }
}

// ===========================================================================
// (4) V_σ is the derivative of E_xc, per spin, both spins populated.
// ===========================================================================

/// Orthonormal (S-metric) eigenvectors of `h` — a deterministic MO set.
fn core_mos(s: &Array2<f64>, h: &Array2<f64>) -> Array2<f64> {
    let (se, su) = s.eigh(UPLO::Upper).unwrap();
    let mut x = su.clone();
    for (j, e) in se.iter().enumerate() {
        x.column_mut(j).mapv_inplace(|v| v / e.sqrt());
    }
    let hp = x.t().dot(h).dot(&x);
    let (_, v) = hp.eigh(UPLO::Upper).unwrap();
    x.dot(&v)
}

fn proj(c: &Array2<f64>, cols: &[(usize, f64)]) -> Array2<f64> {
    let n = c.nrows();
    let mut d = Array2::<f64>::zeros((n, n));
    for &(i, w) in cols {
        let ci = c.column(i).to_owned().insert_axis(Axis(1));
        d.scaled_add(w, &ci.dot(&ci.t()));
    }
    d
}

#[test]
fn polarized_vxc_is_the_derivative_of_exc_per_spin() {
    let cell = triclinic_cell();
    let bs = sp_basis_h();
    let prep = prep_for(&cell, &bs);
    let hc = periodic_hcore(&cell, &prep, &PeriodicHcoreConfig::with_omega(HCORE_OMEGA)).unwrap();
    let grid = PeriodicGrid::build(&cell, &ssf_grid(50, 110, 10.0)).unwrap();
    let c = core_mos(&hc.s, &hc.h);
    let n = c.nrows();
    // ρ_α ≠ ρ_β, both populated (N_α 3, N_β 0.6).
    let d_a = proj(&c, &[(0, 1.0), (1, 1.0), (2, 1.0)]);
    let d_b = proj(&c, &[(0, 0.3), (1, 0.2), (3, 0.1)]);
    let dd = Array2::from_shape_fn((n, n), |(i, j)| {
        1e-3 * (0.37 * ((i + 1) * (j + 1)) as f64).sin()
    });
    let eps = 1e-3;
    for xc in XCS {
        let mut pxc = PeriodicXc::new(&cell, &bs, xc, &grid, &PeriodicXcConfig::default()).unwrap();
        // Closed-shell consistency: eval_polarized(D/2, D/2) ≡ eval(D).
        let d = proj(&c, &[(0, 2.0), (1, 2.0)]);
        let half = &d * 0.5;
        let (e_c, v_c) = pxc.eval(&d).unwrap();
        let (e_p, va, vb) = pxc.eval_polarized(&half, &half).unwrap();
        let dv = max_abs_diff(&va, &v_c).max(max_abs_diff(&vb, &v_c));
        eprintln!(
            "{xc}: closed shell |dE| {:.1e} max|dV| {dv:.1e}",
            (e_p - e_c).abs()
        );
        assert!(
            (e_p - e_c).abs() < 1e-10 && dv < 1e-10,
            "{xc}: polarized != closed"
        );

        let (_, va, vb) = pxc.eval_polarized(&d_a, &d_b).unwrap();
        let mut fd = [0.0; 2];
        for (s, v) in [(0usize, &va), (1, &vb)] {
            let e_at = |sgn: f64, pxc: &mut PeriodicXc| {
                let (pa, pb) = if s == 0 {
                    (&d_a + &(&dd * (sgn * eps)), d_b.clone())
                } else {
                    (d_a.clone(), &d_b + &(&dd * (sgn * eps)))
                };
                pxc.eval_polarized(&pa, &pb).unwrap().0
            };
            fd[s] = (e_at(1.0, &mut pxc) - e_at(-1.0, &mut pxc)) / (2.0 * eps);
            let an = (v * &dd).sum();
            let rel = (an - fd[s]).abs() / fd[s].abs();
            eprintln!(
                "{xc} spin {s}: tr(V dD) {an:.12e} FD {:.12e} rel {rel:.2e}",
                fd[s]
            );
            assert!(fd[s].abs() > 1e-6, "{xc}: vacuous directional derivative");
            assert!(rel < 1e-6, "{xc} spin {s}: rel {rel:.3e}");
        }
        // Control: the two spin potentials are distinguishable along dD
        // (β fed the α potential would fail the β FD above by this much).
        let cross = ((&va * &dd).sum() - fd[1]).abs() / fd[1].abs();
        assert!(
            cross > 1e-3,
            "{xc}: V_a reproduces the beta FD ({cross:.2e})"
        );
    }
}

// ===========================================================================
// (5) Box limit, H/STO-3G PBE0.
// ===========================================================================

/// Prediction (FINDINGS It. 10, written before the prototype sweep):
/// c3 = −(2π/3)·a·Ω_α = 0.25 × the UHF −4.081081 = −1.020270; one AO → no
/// a⁻⁶ relaxation; spherical → no a⁻⁵. Prototype dE·a³ at a = 32: −1.020270;
/// LDA dE 3.1e-14. Artifacts: Madelung on the full K → a 1/a term; K unscaled
/// → 4× c3. Molecular UKS on the identical grid rules (TA-M4 75 × Lebedev
/// 302; one atom, so every partition weight is 1 near it), exact J/K.
#[test]
fn h_atom_sto3g_pbe0_box_limit_is_hyb_times_the_uhf_c3() {
    let c3 = -(2.0 * PI / 3.0) * 0.25 * H_OMEGA_A;
    assert!((c3 - H_C3_PBE0).abs() < 1e-6, "c3 prediction {c3}");
    let a = 32.0;
    let basis = pyscf_sto3g_h();
    let mut mol = hydrogens(&H_ATOM);
    mol.multiplicity = 2;
    let prep_mol = PreparedBasis::new(&mol, &basis).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep_mol).unwrap();
    let molecular = |xc: &str| {
        let cfg = RhfConfig {
            xc: Some(xc.to_string()),
            dft_grid: Some(AtomicGridConfig {
                n_radial: 75,
                n_angular: 302,
                prune: None,
            }),
            df_j_aux: Some(String::new()),
            df_k_aux: Some(String::new()),
            use_sad_guess: false,
            density_conv: 1e-10,
            max_iter: 200,
            ..Default::default()
        };
        let r = solve_uhf(&ParallelContext::default(), &mol, &prep_mol, &bounds, &cfg).unwrap();
        assert!(r.converged);
        r
    };
    let cell = cell_mult(&H_ATOM, cubic(a), 2);
    let prep = prep_for(&cell, &basis);
    let t0 = std::time::Instant::now();
    let hcfg = PeriodicHcoreConfig {
        precision: 1e-12,
        ..PeriodicHcoreConfig::with_omega(0.9)
    };
    let hc = periodic_hcore(&cell, &prep, &hcfg).unwrap();
    let eri = DenseAftEri::build(
        &cell,
        &prep,
        &hc.s,
        ExxDiv::None,
        1e-12,
        DEFAULT_DENSE_AFT_MAX_BYTES,
    )
    .unwrap();
    let mut de = [0.0; 2];
    for (slot, xc) in ["LDA", "PBE0"].into_iter().enumerate() {
        let m = molecular(xc);
        let cfg = GammaUksConfig {
            initial_mos: Some((m.mos_alpha.clone(), m.mos_beta.clone().unwrap())),
            ..GammaUksConfig::new(xc)
        };
        let u = gamma_uks(&cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &cfg).unwrap();
        de[slot] = u.scf.energy - m.energy;
        eprintln!(
            "a = {a} {xc}: E_per {:.13} E_mol {:.13} dE {:.3e} dE*a^3 {:.6} ({:.1} s)",
            u.scf.energy,
            m.energy,
            de[slot],
            de[slot] * a * a * a,
            t0.elapsed().as_secs_f64()
        );
        if let Some(none) = u.none_stage.as_ref() {
            close(
                none.energy - u.scf.energy,
                0.25 * u.madelung / 2.0,
                1e-12,
                "none - ewald (= a v_M/2)",
            );
        }
    }
    assert!(de[0].abs() < 2e-9, "LDA has no a^-3 term: dE {:.3e}", de[0]);
    close(de[1] * a * a * a, c3, 2e-4, "PBE0 (E_ewald - E_mol) a^3");
}

// ===========================================================================
// (6) The Ewald trap, from the occupations.
// ===========================================================================

#[test]
fn ewald_trap_is_flagged_from_the_occupations() {
    // (a) A hole state: occupy the LUMO instead of the HOMO. The sorted gap
    // cannot see it; the occupation-aware one is exactly its negative.
    let h2 = setup(h2_cell(4.0), pyscf_sto3g_h());
    let r = uks(
        &h2,
        &ukscfg(
            "LDA",
            ssf_grid(50, 110, 10.0),
            ExxDiv::None,
            EwaldStart::Direct,
        ),
    );
    let aufbau = occupation_gaps(&r.scf, &h2.hc.s, 0.0);
    let sorted = spin_gaps(&r.scf, 1, 1, 0.0);
    let g = aufbau.gap_alpha.unwrap();
    assert!(
        g > 0.1 && (g - sorted.gap_alpha.unwrap()).abs() < 1e-12,
        "{aufbau:?}"
    );
    let mut hole = r.scf.clone();
    hole.density_alpha = proj(&r.scf.mos_alpha, &[(1, 1.0)]);
    let occ = occupation_gaps(&hole, &h2.hc.s, 0.0);
    let srt = spin_gaps(&hole, 1, 1, 0.0);
    eprintln!(
        "hole state: occupation gap {:?}, sorted gap {:?}",
        occ.gap_alpha, srt.gap_alpha
    );
    close(occ.gap_alpha.unwrap(), -g, 1e-10, "hole occupation gap");
    assert!(
        srt.gap_alpha.unwrap() > 0.0 && srt.satisfied(),
        "sorted gap sees no hole"
    );
    assert!(!occ.satisfied(), "occupation gap must flag the hole");
    // The shift enters as `gap >= shift`.
    assert!(!occupation_gaps(&r.scf, &h2.hc.s, g + 1e-6).satisfied());
    assert!(occupation_gaps(&r.scf, &h2.hc.s, g - 1e-6).satisfied());

    // (b) End to end: the a = 1 builder, single ewald stage from the core
    // guess, lands in the Iteration-6 trap and gamma_uks flags it (and
    // warns); the staged default reaches PySCF's state and is not flagged.
    let tri = setup(cell_mult(&TRI_ATOMS, TRI_A, 3), sp_basis_h());
    let vm = madelung_constant(&tri.cell).unwrap();
    let trapped = uks_with(
        &tri,
        &mut ZeroXc { frac: 1.0 },
        ExxDiv::Ewald,
        EwaldStart::Direct,
    );
    eprintln!(
        "a=1 direct ewald: E {:.12} gaps {:?} v_M {vm:.10}",
        trapped.scf.energy, trapped.gaps
    );
    close(
        trapped.scf.energy,
        TRI_TRIPLET_UHF_E_EWALD_TRAPPED,
        1e-9,
        "trapped E",
    );
    assert!(
        !trapped.gaps.satisfied() && trapped.gaps.gap_alpha.unwrap() < vm,
        "trap must be flagged: {:?}",
        trapped.gaps
    );
    assert_eq!(trapped.gaps.madelung_applied, vm);
    let staged = uks_with(
        &tri,
        &mut ZeroXc { frac: 1.0 },
        ExxDiv::Ewald,
        EwaldStart::Staged,
    );
    close(staged.scf.energy, TRI_TRIPLET_UHF_E_EWALD, 1e-9, "staged E");
    assert!(staged.gaps.satisfied(), "staged: {:?}", staged.gaps);
    // The shift is a·v_M: the same trapped density judged at a = 0.25
    // (window 0.25 v_M) is not flagged by its α gap (0.617 > 0.25 v_M).
    let quarter = occupation_gaps(&trapped.scf, &tri.hc.s, 0.25 * vm);
    assert!(quarter.gap_alpha.unwrap() > 0.25 * vm);
}

// ===========================================================================
// Refusals.
// ===========================================================================

#[test]
fn gamma_uks_refuses_unsupported_features_by_name() {
    let su = setup(cell_mult(&H_ATOM, cubic(4.0), 2), pyscf_sto3g_h());
    let run = |edit: &dyn Fn(&mut GammaUksConfig)| {
        let mut cfg = GammaUksConfig::new("PBE");
        cfg.grid = ssf_grid(30, 50, 10.0);
        edit(&mut cfg);
        gamma_uks(
            &su.cell,
            &su.prep,
            &su.hc,
            GammaUhfIntegrals::DenseAft(&su.eri),
            &cfg,
        )
        .unwrap_err()
        .to_string()
    };
    let cases: [(&str, &dyn Fn(&mut GammaUksConfig)); 7] = [
        ("RhfConfig.xc", &|c| c.scf.xc = Some("PBE".into())),
        ("RhfConfig.check_stability", &|c| {
            c.scf.check_stability = true
        }),
        ("RhfConfig.newton_trigger", &|c| c.scf.newton_trigger = 1e-2),
        ("scf_stability_descent", &|c| {
            c.scf.scf_stability_descent = true
        }),
        ("scf.xc_omega", &|c| c.scf.xc_omega = Some(0.3)),
        ("scf.dft_grid", &|c| {
            c.scf.dft_grid = Some(AtomicGridConfig::default())
        }),
        ("range-separated hybrid", &|c| {
            c.functional = "wB97X-V".into()
        }),
    ];
    for (word, edit) in cases {
        let e = run(edit);
        assert!(e.contains(word), "{word}: {e}");
    }
    let cfg = GammaUksConfig::new("unused");
    let ints = GammaUhfIntegrals::DenseAft(&su.eri);
    let e = gamma_uks_with_xc(
        &su.cell,
        &su.prep,
        &su.hc,
        ints,
        &mut ZeroXc { frac: 1.5 },
        &cfg,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("exact-exchange fraction"), "{e}");
    // The SCF itself also refuses an out-of-range fraction.
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &su.prep).unwrap();
    let inj = PeriodicInjection {
        s: su.hc.s.clone(),
        h: su.hc.h.clone(),
        vnn: su.hc.enn,
        j: Box::new(su.eri.j_builder()),
        k: Box::new(su.eri.k_builder()),
        xc: Some(Box::new(ZeroXc { frac: -0.1 })),
    };
    let e = solve_uhf_injected(
        &ParallelContext::default(),
        su.cell.mol(),
        &su.prep,
        &bounds,
        &cfg.scf,
        inj,
        None,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("exact-exchange fraction"), "{e}");
}
