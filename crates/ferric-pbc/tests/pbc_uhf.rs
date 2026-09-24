//! Stage 4: Gamma-point UHF (`ferric_pbc::uhf::gamma_uhf` over
//! `ferric_scf::uhf::solve_uhf_injected`), ported from
//! `reference/pbc/pbc_uhf.py` (FINDINGS "Iteration 6 (Python, Gamma UHF)").
//! Every pinned number is from that iteration (`run_uhf_anchor.py`,
//! `run_uhf_oracle.py`, `run_uhf_box_limit.py`, `run_uhf_guess.py`).
//!
//! # Anchors (written first)
//!
//! (1) Closed shell through UHF ≡ Gamma RHF (same dense AFT J/K), both exxdiv:
//!     prototype 0 / 1e-15 (H2), 2.7e-15 (tri s+p); asserted 1e-12. Negative
//!     control: a per-spin Madelung of v_M/2 (RHF's "D/2" bookkeeping wrongly
//!     applied to a spin density) moves E by v_M N/4 (prototype +3.5e-1 H2).
//! (2) Open-shell TRIPLET (N_α 3, N_β 1) dense AFT ≡ RS-GDF with the trivial
//!     aux (every periodic pair product), tri one-s cell, both exxdiv:
//!     prototype −1.1e-11; asserted 1e-10. Negative control: 7 of 8 aux classes
//!     must fail it (prototype +6.7e-7). N_β = 1 ≠ 0, so the K[D_total]
//!     mutant is visible here (prototype −2.4e-1). Blind spot: the Madelung
//!     term is applied identically on both sides — (1)/(3) guard its factor.
//!
//! # Oracles
//!
//! (3) PySCF 2.13 `pbc.scf.UHF` AFTDF pins (energies and ⟨S²⟩ with the lattice
//!     S). H atom and H2 triplet have N_β = 0, so they cannot see K[D_total]
//!     (the anchors do); they DO see the per-spin Madelung factor.
//! (4) The Gamma Ewald trap (tri 4H s+p triplet): a single ewald SCF from the
//!     core guess lands at −1.812958714837 (alpha gap 0.617 < v_M 0.622), the
//!     staged none→ewald start reaches PySCF's −1.827999723359. Both asserted,
//!     plus the gap flag on each.
//! (5) Molecular box limit, H atom (independent of PySCF pbc): with ewald the
//!     residual is exactly c3/a³, c3 = −(2π/3)(|d|² + Ω_α + Ω_β), predicted
//!     from MOLECULAR spreads (H: −4.081081; per spin HALF the RHF per-orbital
//!     coefficient); with none it is +v_M N/2 more. Asserted at a = 20 to
//!     1e-5 relative. O2 is NOT ported: pure-AFT for O/STO-3G (p_max ≈ 261)
//!     needs ~1e9 G at a = 24, and RS-GDF with a real aux adds a fitting error
//!     that has not been measured against the 0.5 % bar — so it would be a
//!     guess, not a test.
//!
//! # Default for the Ewald trap
//!
//! `gamma_uhf` defaults to `EwaldStart::Staged` and always reports the
//! per-spin gap against v_M (warning on stderr when violated).
//!
//! # Mutation plan (for the main agent; each must fail ≥ 1 test here)
//!
//! * `solve_uhf_impl`: build both K_σ from `d_total` (K[D_total]) → (1)
//!   closed shell (prototype −4.5e-2 H2) and (2) fail.
//! * `DenseAftK::build`: `k.scaled_add(0.5 * self.madelung, ..)` → (1) ewald,
//!   (3) ewald pins and (5) fail.
//! * `gamma_uhf`: pass `applied` (instead of 0.0) to the first stage → (4)
//!   staged fails (it becomes Direct).
//! * `spin_gaps`: `g - madelung_applied` → `g` → (4) gap flags fail.
//! * `solve_uhf_impl` injected guess: call the molecular hcore via `mol`
//!   instead of the injected `h` → every test fails (periodic h ≠ molecular).
//!
//! # Molecular path unchanged
//!
//! `solve_uhf_fockmod` is now a wrapper passing `inj = None`; every new
//! branch in `solve_uhf_impl` is gated on `inj_jk.is_some()`. Regression (run
//! the ferric-scf UHF suites; bitwise equality for the injected-vs-molecular
//! link path lives in `ferric-scf/tests/scf_injection.rs`).

mod common;

use common::*;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_integrals::site_basis::SiteBasis;
use ferric_pbc::dense_aft::{
    DenseAftEri, ExxDiv, DEFAULT_DENSE_AFT_MAX_BYTES, DEFAULT_DENSE_AFT_PRECISION,
};
use ferric_pbc::ewald::madelung_constant;
use ferric_pbc::hcore::{periodic_hcore, PeriodicHcore, PeriodicHcoreConfig};
use ferric_pbc::lattice::Cell;
use ferric_pbc::rsgdf::{RsGdf, RsGdfConfig};
use ferric_pbc::uhf::{gamma_uhf, EwaldStart, GammaUhfConfig, GammaUhfIntegrals, GammaUhfResult};
use ferric_scf::fock::KBuilder;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{PeriodicInjection, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_injected};
use ndarray::Array2;
use std::f64::consts::PI;

/// ω for the nuclear-attraction split (as in pbc_dense_aft_scf.rs).
const HCORE_OMEGA: f64 = 0.8;
const ANCHOR_ALPHA: f64 = 0.5;
const AMPLE: usize = 1 << 30;

/// `test_prototype.py` H_ATOM (Bohr).
const H_ATOM: [[f64; 3]; 1] = [[0.3, 0.2, 0.1]];

// --- PySCF 2.13 pbc.scf.UHF, AFTDF mesh 61³, conv 1e-12 (FINDINGS Iteration 6
// oracle table; agreement with the prototype <= 1.3e-12).
const H_ATOM_E: [f64; 2] = [-0.402177788224, -0.756839973159]; // [none, ewald], <S2> 0.75
const H2_TRIPLET_E: [f64; 2] = [0.322229103842, -0.387095266028]; // <S2> 2
const TRI_TRIPLET_E_NONE: f64 = -0.583125965387;
const TRI_TRIPLET_S2: f64 = 2.001814801;
/// PySCF default guess == prototype none-first == our staged start.
const TRI_TRIPLET_E_EWALD: f64 = -1.827999723359;
/// Prototype single ewald SCF from the core guess (the trap; <S2> 2.002889744).
const TRI_TRIPLET_E_EWALD_TRAPPED: f64 = -1.812958714837;

// --- Box limit (FINDINGS Iteration 6, predictions written before the sweep).
const H_E_MOL_UHF: f64 = -0.466581849557;
const H_C3_PRED: f64 = -4.081081;
const H_OMEGA_A: f64 = 1.948573;

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

fn ucfg(exxdiv: ExxDiv, start: EwaldStart) -> GammaUhfConfig {
    GammaUhfConfig {
        exxdiv,
        ewald_start: start,
        ..Default::default()
    }
}

fn uhf(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    ints: GammaUhfIntegrals<'_>,
    exxdiv: ExxDiv,
    start: EwaldStart,
) -> GammaUhfResult {
    let r = gamma_uhf(cell, prep, hc, ints, &ucfg(exxdiv, start)).expect("gamma UHF");
    assert!(r.scf.converged);
    r
}

/// Injected UHF on the dense tensor with an ARBITRARY Madelung factor in K —
/// the per-spin-factor negative control (only reachable below `gamma_uhf`).
fn uhf_with_k_madelung(
    cell: &Cell,
    prep: &PreparedBasis,
    hc: &PeriodicHcore,
    eri: &DenseAftEri,
    vm_k: f64,
) -> ScfResult {
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), prep).expect("schwarz");
    let inj = PeriodicInjection {
        s: hc.s.clone(),
        h: hc.h.clone(),
        vnn: hc.enn,
        j: Box::new(eri.j_builder()),
        k: Box::new(eri.k_builder_with_madelung(vm_k)),
    };
    let cfg = GammaUhfConfig::default();
    solve_uhf_injected(&ctx, cell.mol(), prep, &bounds, &cfg.scf, inj, None).expect("mutant UHF")
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

const EXX: [ExxDiv; 2] = [ExxDiv::None, ExxDiv::Ewald];

// ===========================================================================
// (1) Closed shell through UHF ≡ Gamma RHF.
// ===========================================================================

#[test]
fn closed_shell_uhf_equals_gamma_rhf_both_exxdiv() {
    let systems: [(&str, Cell, ferric_core::basis::BasisSet); 2] = [
        ("H2 a=4", h2_cell(4.0), pyscf_sto3g_h()),
        ("tri 4H s+p", triclinic_cell(), sp_basis_h()),
    ];
    for (label, cell, basis) in systems {
        let prep = prep_for(&cell, &basis);
        let hc = hcore(&cell, &prep);
        let base = dense_none(&cell, &prep, &hc);
        for exx in EXX {
            let eri = base.clone().with_exxdiv(&cell, exx).unwrap();
            let r = gamma_rhf(&cell, &prep, &hc, &eri);
            let u = uhf(
                &cell,
                &prep,
                &hc,
                GammaUhfIntegrals::DenseAft(&base),
                exx,
                EwaldStart::Direct,
            );
            eprintln!(
                "{label} {exx:?}: E_rhf {:.14}, E_uhf {:.14}, dE {:.2e}, <S2> {:.2e}",
                r.energy,
                u.scf.energy,
                u.scf.energy - r.energy,
                u.s2
            );
            assert!(
                (u.scf.energy - r.energy).abs() < 1e-12,
                "{label} {exx:?}: dE {:.3e}",
                u.scf.energy - r.energy
            );
            assert!(u.s2.abs() < 1e-10, "{label} {exx:?}: <S2> {}", u.s2);
        }
        // Negative control: per-spin Madelung v_M/2 misses the ewald RHF by
        // v_M N/4 (the anchor can see a per-spin factor slip).
        let vm = madelung_constant(&cell).unwrap();
        let rhf_e = gamma_rhf(
            &cell,
            &prep,
            &hc,
            &base.clone().with_exxdiv(&cell, ExxDiv::Ewald).unwrap(),
        )
        .energy;
        let mutant = uhf_with_k_madelung(&cell, &prep, &hc, &base, 0.5 * vm);
        let nelec = cell.mol().nelec() as f64;
        eprintln!(
            "{label} mutant v_M/2: dE {:.4e} (v_M N/4 = {:.4e})",
            mutant.energy - rhf_e,
            vm * nelec / 4.0
        );
        assert!(
            (mutant.energy - rhf_e - vm * nelec / 4.0).abs() < 1e-9,
            "{label}: the half-Madelung mutant should sit exactly v_M N/4 high"
        );
    }
}

// ===========================================================================
// (2) Open-shell dense AFT ≡ RS-GDF in the trivial-aux limit.
// ===========================================================================

/// `test_prototype._tri_anchor` aux: every periodic pair product of the
/// one-s triclinic cell (10 pair types × 8 half-lattice classes).
fn tri_anchor_sites(cell: &Cell, classes: &[usize]) -> Vec<[f64; 4]> {
    let r = cell.positions();
    let a = cell.lattice();
    let n = r.len();
    let mut out = Vec::new();
    for i in 0..n {
        for j in i..n {
            for k in 0..8usize {
                if !classes.contains(&k) {
                    continue;
                }
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

fn gdf_cfg() -> RsGdfConfig {
    RsGdfConfig {
        omega: 0.8,
        exxdiv: ExxDiv::None,
        budget_bytes: Some(AMPLE),
        ..Default::default()
    }
}

#[test]
fn open_shell_triplet_dense_equals_rsgdf_in_the_trivial_aux_limit() {
    let cell = cell_mult(&TRI_ATOMS, TRI_A, 3);
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    let hc = hcore(&cell, &prep);
    let eri = dense_none(&cell, &prep, &hc);
    let site = SiteBasis::new(&tri_anchor_sites(&cell, &[0, 1, 2, 3, 4, 5, 6, 7]), 0).unwrap();
    assert_eq!(site.prep.nbasis(), 80);
    let gdf = RsGdf::build(&cell, &prep, &site.prep, &hc.s, &gdf_cfg()).unwrap();
    for exx in EXX {
        let d = uhf(
            &cell,
            &prep,
            &hc,
            GammaUhfIntegrals::DenseAft(&eri),
            exx,
            EwaldStart::Direct,
        );
        let g = uhf(
            &cell,
            &prep,
            &hc,
            GammaUhfIntegrals::RsGdf(&gdf),
            exx,
            EwaldStart::Direct,
        );
        eprintln!(
            "tri one-s triplet {exx:?}: E_dense {:.13}, E_gdf {:.13}, dE {:.2e}, <S2> {:.9} / {:.9}",
            d.scf.energy,
            g.scf.energy,
            g.scf.energy - d.scf.energy,
            d.s2,
            g.s2
        );
        assert_eq!(d.nocc, (3, 1));
        assert!(
            d.s2 > 2.0001,
            "vacuous: triplet not spin-contaminated ({})",
            d.s2
        );
        assert!(
            (g.scf.energy - d.scf.energy).abs() < 1e-10,
            "{exx:?}: dE {:.3e}",
            g.scf.energy - d.scf.energy
        );
        assert!((g.s2 - d.s2).abs() < 1e-9, "{exx:?}: d<S2>");
    }
    // Negative control: 7 of 8 classes (prototype +6.7e-7 triplet).
    let site7 = SiteBasis::new(&tri_anchor_sites(&cell, &[0, 1, 2, 3, 4, 5, 6]), 0).unwrap();
    let gdf7 = RsGdf::build(&cell, &prep, &site7.prep, &hc.s, &gdf_cfg()).unwrap();
    let e_d = uhf(
        &cell,
        &prep,
        &hc,
        GammaUhfIntegrals::DenseAft(&eri),
        ExxDiv::None,
        EwaldStart::Direct,
    )
    .scf
    .energy;
    let e7 = uhf(
        &cell,
        &prep,
        &hc,
        GammaUhfIntegrals::RsGdf(&gdf7),
        ExxDiv::None,
        EwaldStart::Direct,
    )
    .scf
    .energy;
    eprintln!("7/8 aux classes: dE {:.3e} (prototype +6.7e-7)", e7 - e_d);
    assert!((e7 - e_d).abs() > 1e-8, "incomplete aux passed the anchor");
}

// ===========================================================================
// (3) PySCF pins.
// ===========================================================================

#[test]
fn h_atom_and_h2_triplet_match_pinned_pyscf_aftdf() {
    for (label, cell, na_nb, e_ref, s2_ref) in [
        (
            "H atom a=4",
            cell_mult(&H_ATOM, cubic(4.0), 2),
            (1usize, 0usize),
            H_ATOM_E,
            0.75,
        ),
        (
            "H2 a=4 triplet",
            cell_mult(&H2_ATOMS, cubic(4.0), 3),
            (2, 0),
            H2_TRIPLET_E,
            2.0,
        ),
    ] {
        let prep = prep_for(&cell, &pyscf_sto3g_h());
        let hc = hcore(&cell, &prep);
        let eri = dense_none(&cell, &prep, &hc);
        let vm = madelung_constant(&cell).unwrap();
        let mut es = [0.0; 2];
        for (slot, exx) in EXX.into_iter().enumerate() {
            let u = uhf(
                &cell,
                &prep,
                &hc,
                GammaUhfIntegrals::DenseAft(&eri),
                exx,
                EwaldStart::Staged,
            );
            assert_eq!(u.nocc, na_nb);
            close(
                u.scf.energy,
                e_ref[slot],
                1e-9,
                &format!("{label} {exx:?} E"),
            );
            close(u.s2, s2_ref, 1e-10, &format!("{label} {exx:?} <S2>"));
            es[slot] = u.scf.energy;
        }
        let n = (na_nb.0 + na_nb.1) as f64;
        close(
            es[1] - es[0],
            -vm * n / 2.0,
            1e-10,
            &format!("{label} ewald - none"),
        );
        // Per-spin half-Madelung mutant misses the pinned ewald E by v_M N/4.
        let mutant = uhf_with_k_madelung(&cell, &prep, &hc, &eri, 0.5 * vm);
        assert!(
            (mutant.energy - e_ref[1]).abs() > 0.1,
            "{label}: half-Madelung mutant within 0.1 of the pin"
        );
    }
}

#[test]
fn tri_4h_sp_triplet_matches_pinned_pyscf_aftdf() {
    let cell = cell_mult(&TRI_ATOMS, TRI_A, 3);
    let prep = prep_for(&cell, &sp_basis_h());
    let hc = hcore(&cell, &prep);
    let eri = dense_none(&cell, &prep, &hc);
    let none = uhf(
        &cell,
        &prep,
        &hc,
        GammaUhfIntegrals::DenseAft(&eri),
        ExxDiv::None,
        EwaldStart::Direct,
    );
    close(
        none.scf.energy,
        TRI_TRIPLET_E_NONE,
        1e-9,
        "tri triplet none E",
    );
    close(none.s2, TRI_TRIPLET_S2, 1e-8, "tri triplet none <S2>");
    assert!(none.gaps.satisfied(), "none: {:?}", none.gaps);
    // The ewald pin (staged) is asserted in the trap test below.
}

// ===========================================================================
// (4) The Gamma Ewald trap.
// ===========================================================================

#[test]
fn ewald_trap_single_stage_lands_high_staged_reaches_pyscf() {
    let cell = cell_mult(&TRI_ATOMS, TRI_A, 3);
    let prep = prep_for(&cell, &sp_basis_h());
    let hc = hcore(&cell, &prep);
    let eri = dense_none(&cell, &prep, &hc);
    let vm = madelung_constant(&cell).unwrap();

    let trapped = uhf(
        &cell,
        &prep,
        &hc,
        GammaUhfIntegrals::DenseAft(&eri),
        ExxDiv::Ewald,
        EwaldStart::Direct,
    );
    eprintln!(
        "single-stage ewald: E {:.12}, <S2> {:.9}, gaps {:?}, v_M {vm:.10}",
        trapped.scf.energy, trapped.s2, trapped.gaps
    );
    close(
        trapped.scf.energy,
        TRI_TRIPLET_E_EWALD_TRAPPED,
        1e-9,
        "single-stage ewald (trapped) E",
    );
    assert!(
        !trapped.gaps.satisfied() && trapped.gaps.gap_alpha.unwrap() < vm,
        "the trapped state must be flagged: {:?}",
        trapped.gaps
    );

    let staged = uhf(
        &cell,
        &prep,
        &hc,
        GammaUhfIntegrals::DenseAft(&eri),
        ExxDiv::Ewald,
        EwaldStart::Staged,
    );
    let first = staged
        .none_stage
        .as_ref()
        .expect("staged runs a none stage");
    eprintln!(
        "staged: E_none {:.12} -> E_ewald {:.12} ({} + {} iterations), <S2> {:.9}, gaps {:?}",
        first.energy,
        staged.scf.energy,
        first.iterations,
        staged.scf.iterations,
        staged.s2,
        staged.gaps
    );
    close(
        staged.scf.energy,
        TRI_TRIPLET_E_EWALD,
        1e-9,
        "staged ewald E",
    );
    close(staged.s2, TRI_TRIPLET_S2, 1e-8, "staged ewald <S2>");
    close(
        first.energy,
        TRI_TRIPLET_E_NONE,
        1e-9,
        "staged none stage E",
    );
    // Same stationary density: E_ewald − E_none = −v_M N/2 = −2 v_M.
    close(
        staged.scf.energy - first.energy,
        -2.0 * vm,
        1e-10,
        "staged ewald - none",
    );
    assert!(staged.gaps.satisfied(), "staged: {:?}", staged.gaps);
    assert!(
        trapped.scf.energy - staged.scf.energy > 1e-2,
        "the trap must be ~1.5e-2 above the staged state"
    );
}

// ===========================================================================
// (5) Molecular box limit, H atom.
// ===========================================================================

/// `c3 = −(2π/3)(|d|² + Ω_α + Ω_β)`, `Ω_σ = Σ_i ⟨i|r²|i⟩ − Σ_ij |⟨i|r|j⟩|²`
/// over σ-occupied MOs, `d = Σ Z R − Σ_σ Σ_i ⟨i|r|i⟩` (origin 0).
fn uhf_c3_closed_form(
    mol: &Molecule,
    prep: &PreparedBasis,
    r: &ScfResult,
    na: usize,
    nb: usize,
) -> (f64, [f64; 2]) {
    let rr = oneelectron::dipole(prep, [0.0; 3]).unwrap();
    let r2 = oneelectron::r2_moment(prep, [0.0; 3]).unwrap();
    let mut dip = [0.0; 3];
    for a in &mol.atoms {
        let z = a.z as f64;
        dip[0] += z * a.x;
        dip[1] += z * a.y;
        dip[2] += z * a.zpos;
    }
    let mut omega = [0.0; 2];
    let cb = r.mos_beta.as_ref().unwrap();
    for (s, (c, n)) in [(&r.mos_alpha, na), (cb, nb)].into_iter().enumerate() {
        if n == 0 {
            continue;
        }
        let co = c.slice(ndarray::s![.., ..n]).to_owned();
        let mut om: f64 = (0..n)
            .map(|i| co.column(i).dot(&r2.dot(&co.column(i))))
            .sum();
        for x in 0..3 {
            let rij: Array2<f64> = co.t().dot(&rr[x]).dot(&co);
            om -= rij.iter().map(|v| v * v).sum::<f64>();
            dip[x] -= (0..n).map(|i| rij[(i, i)]).sum::<f64>();
        }
        omega[s] = om;
    }
    let d2 = dip.iter().map(|v| v * v).sum::<f64>();
    (-(2.0 * PI / 3.0) * (d2 + omega[0] + omega[1]), omega)
}

#[test]
fn h_atom_box_residual_is_exactly_c3_over_a3() {
    let basis = pyscf_sto3g_h();
    let mut mol = hydrogens(&H_ATOM);
    mol.multiplicity = 2;
    let prep_mol = PreparedBasis::new(&mol, &basis).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep_mol).unwrap();
    let mcfg = RhfConfig {
        use_sad_guess: false,
        density_conv: 1e-10,
        ..Default::default()
    };
    let m = solve_uhf(&ParallelContext::default(), &mol, &prep_mol, &bounds, &mcfg).unwrap();
    close(m.energy, H_E_MOL_UHF, 1e-9, "molecular UHF H atom E");
    let (c3, omega) = uhf_c3_closed_form(&mol, &prep_mol, &m, 1, 0);
    close(omega[0], H_OMEGA_A, 1e-6, "Omega_alpha");
    close(c3, H_C3_PRED, 1e-5 * H_C3_PRED.abs(), "c3 prediction");

    let a = 20.0;
    let cell = cell_mult(&H_ATOM, cubic(a), 2);
    let prep = prep_for(&cell, &basis);
    let hcfg = PeriodicHcoreConfig {
        precision: 1e-12,
        ..PeriodicHcoreConfig::with_omega(0.9)
    };
    let t0 = std::time::Instant::now();
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
    let cfg = GammaUhfConfig {
        initial_mos: Some((m.mos_alpha.clone(), m.mos_beta.clone().unwrap())),
        ..Default::default()
    };
    let u = gamma_uhf(&cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &cfg).unwrap();
    let none = u.none_stage.as_ref().expect("staged by default");
    let de = u.scf.energy - m.energy;
    eprintln!(
        "a = {a}: E_ewald {:.13}, E_none {:.13}, dE*a^3 {:.6} (c3 {c3:.6}), v_M {:.10}, half-G {} ({:.1} s)",
        u.scf.energy,
        none.energy,
        de * a * a * a,
        u.madelung,
        eri.n_g_half(),
        t0.elapsed().as_secs_f64()
    );
    close(de * a * a * a, c3, 1e-5 * c3.abs(), "(E_ewald - E_mol) a^3");
    // none: +v_M N/2 = +v_M/2 exactly (1.4186/a).
    close(
        none.energy - u.scf.energy,
        u.madelung / 2.0,
        1e-12,
        "none - ewald",
    );
}

// ===========================================================================
// Builders and config.
// ===========================================================================

/// Neither `DenseAftK` nor `RsGdfK` overrides `build_from_occ`, so the
/// occupied-MO path keeps the Madelung term. Guards a future DfK-style
/// override that forgets `v_M S C Cᵀ S` (it would pass every density-path
/// test and silently drop Madelung only here).
#[test]
fn k_builders_keep_madelung_on_the_occupied_path() {
    let cell = cell_mult(&TRI_ATOMS, TRI_A, 3);
    let prep = prep_for(&cell, &single_s_h(ANCHOR_ALPHA));
    let hc = hcore(&cell, &prep);
    let eri = dense_none(&cell, &prep, &hc);
    let site = SiteBasis::new(&tri_anchor_sites(&cell, &[0, 1, 2, 3, 4, 5, 6, 7]), 0).unwrap();
    let gdf = RsGdf::build(&cell, &prep, &site.prep, &hc.s, &gdf_cfg()).unwrap();
    let vm = madelung_constant(&cell).unwrap();
    let n = prep.nbasis();
    let c = Array2::from_shape_fn((n, 2), |(i, j)| ((i * 7 + j * 3 + 1) as f64 * 0.37).sin());
    let d = c.dot(&c.t());
    let sds = hc.s.dot(&d).dot(&hc.s);
    let check = |label: &str, k_vm: &mut dyn KBuilder, k_0: &mut dyn KBuilder| {
        let mut k_occ = Array2::<f64>::zeros((n, n));
        let mut k_den = Array2::<f64>::zeros((n, n));
        let mut k_bare = Array2::<f64>::zeros((n, n));
        k_vm.build_from_occ(&c, &mut k_occ).unwrap();
        k_vm.build(&d, &mut k_den).unwrap();
        k_0.build_from_occ(&c, &mut k_bare).unwrap();
        let occ_vs_den = max_abs_diff(&k_occ, &k_den);
        let madelung_part = max_abs_diff(&(&k_occ - &k_bare), &(vm * &sds));
        eprintln!("{label}: |K_occ - K_D| {occ_vs_den:.2e}, |dK - v_M SDS| {madelung_part:.2e}");
        assert!(occ_vs_den < 1e-12, "{label}: occupied path differs");
        assert!(
            madelung_part < 1e-12,
            "{label}: Madelung term lost on the occupied path"
        );
        assert!(sds.iter().any(|v| v.abs() > 1e-2), "vacuous S D S");
    };
    check(
        "DenseAftK",
        &mut eri.k_builder_with_madelung(vm),
        &mut eri.k_builder_with_madelung(0.0),
    );
    check(
        "RsGdfK",
        &mut gdf.k_builder_with_madelung(vm),
        &mut gdf.k_builder_with_madelung(0.0),
    );
}

#[test]
fn ewald_start_parse_is_strict() {
    assert_eq!(
        EwaldStart::parse_config_str("Staged").unwrap(),
        EwaldStart::Staged
    );
    assert_eq!(
        EwaldStart::parse_config_str("direct").unwrap(),
        EwaldStart::Direct
    );
    assert_eq!(EwaldStart::default(), EwaldStart::Staged);
    for bad in ["", "stage", "direct "] {
        assert!(
            EwaldStart::parse_config_str(bad).is_err(),
            "{bad:?} accepted"
        );
    }
}

#[test]
fn gamma_uhf_refuses_injected_path_features_by_name() {
    let cell = cell_mult(&H_ATOM, cubic(4.0), 2);
    let prep = prep_for(&cell, &pyscf_sto3g_h());
    let hc = hcore(&cell, &prep);
    let eri = dense_none(&cell, &prep, &hc);
    let cfg = GammaUhfConfig {
        scf: RhfConfig {
            use_sad_guess: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let e = gamma_uhf(&cell, &prep, &hc, GammaUhfIntegrals::DenseAft(&eri), &cfg)
        .expect_err("MINAO guess must be refused")
        .to_string();
    assert!(
        e.contains("RhfConfig.use_sad_guess is not supported"),
        "{e}"
    );
    let bad_mult = cell_mult(&H_ATOM, cubic(4.0), 1);
    assert!(gamma_uhf(
        &bad_mult,
        &prep,
        &hc,
        GammaUhfIntegrals::DenseAft(&eri),
        &GammaUhfConfig::default()
    )
    .is_err());
}
