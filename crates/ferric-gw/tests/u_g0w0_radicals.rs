//! Quantitative U-G0W0@UHF validation on small doublet radicals vs an
//! independent PySCF `ugw_ac` reference — tightens U-GW from OH-only sanity
//! bounds (see `oh_u_g0w0.rs`) to a narrow, apples-to-apples cross-code check
//! on three radicals (OH, CH₃, NH₂).
//!
//! ## Reference provenance
//!
//! Each reference number is PySCF's spin-unrestricted G0W0@UHF (`ugw_ac.UGWAC`,
//! full analytic continuation over all orbitals), fed the **identical** bundled
//! cc-pVDZ orbital JSON + cc-pVDZ-RI auxiliary JSON that ferric compiles in, so
//! the orbital basis and the density-fitting aux are bit-identical between the
//! two codes (the same apples-to-apples methodology as the ECP GW cross-check,
//! `scripts/gw100/pyscf_g0w0_ecp.py`). ferric builds W via PDEP + Padé AC;
//! PySCF builds it via analytic continuation — so agreement is a genuine
//! independent check of the open-shell self-energy (`u_sigma.rs`) and QP layer,
//! not a re-derivation of the same code path.
//!
//! Generator: `scripts/gw100/pyscf_u_g0w0.py <xyz> <charge> <2S>`, PySCF 2.13.0.
//! Geometries: `scripts/gw100/geom_radicals/{oh,ch3,nh2}.xyz` (identical here).
//! Values are −ε_QP of the α-HOMO and β-HOMO in eV:
//!
//! | radical | α-HOMO IP | β-HOMO IP | Koopmans α | Koopmans β |
//! |---------|-----------|-----------|------------|------------|
//! | OH      | 13.5686   | 12.5320   | 14.8294    | 13.5831    |
//! | CH₃     |  9.7541   | 15.0065   | 10.4208    | 15.2907    |
//! | NH₂     | 12.7417   | 11.7493   | 13.4965    | 12.3105    |
//!
//! ## Tolerance
//!
//! 0.20 eV, per-channel. This is the documented ferric↔PySCF GW cross-check
//! band: closed-shell G0W0@HF water agrees to <0.1 eV once the Σ_c offset is
//! folded into the QP self-consistency (see `docs/VALIDATION.md`), and the ECP
//! GW set agrees to ≤10.7 meV. 0.20 eV leaves headroom for the two remaining
//! deliberate methodological differences (PDEP-as-W vs AC-W screening; ferric's
//! 16-node Gauss-Legendre imaginary-frequency grid vs PySCF's AC grid) while
//! still being ~15× tighter than the old (Koopmans−5, +1) eV sanity window and
//! a real quantitative pin rather than a range band.
//!
//! Run: OPENBLAS_NUM_THREADS=1 cargo test -p ferric-gw --release --ignored u_g0w0_radicals

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_gw::{run_u_gw, GwConfig, GwMethod};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{
    Chi0Backend, Chi0Sparsity, Eigensolver, PdepRpaConfig, QuadratureConfig, QuadratureScheme,
    SternheimerConfig,
};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

const HA_TO_EV: f64 = 27.211_386_245_988_f64;

/// Independent PySCF `ugw_ac` reference; see module docs for provenance.
struct RadicalRef {
    name: &'static str,
    xyz: &'static str,
    ip_a_ref: f64,
    ip_b_ref: f64,
    /// ferric's own ground-state UHF total energy (Ha, direct-K). Asserting the
    /// SCF energy lands here guards against ferric converging to a different
    /// (wrong-basin) doublet solution and then producing a spuriously
    /// "converged" but physically wrong QP IP — which is exactly what NH₂ does
    /// under plain DIIS from the default guess (lands −55.4834 Ha, an excited
    /// SCF solution ~0.084 Ha / ~2.3 eV off, vs the −55.5671 Ha ground state).
    /// These are ~1e-3 Ha below the PySCF density-fitted UHF energies used to
    /// generate the QP references (the exact-K vs DF difference, uniform across
    /// all three radicals) — the guard checks that ferric reaches ITS ground
    /// state, which is the state the QP reference corresponds to.
    e_uhf_ref: f64,
    /// Virtual-block level shift + MOM-after-iter needed to steer the UHF into
    /// the physical ground state. `(0.0, 0)` = plain DIIS suffices (OH, CH₃);
    /// NH₂ needs `(0.5, 5)`.
    level_shift: f64,
    mom_after_iter: usize,
}

const TOL_EV: f64 = 0.20;

fn pdep_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        need_eigenvalues_freq: true, // GW reads eigenvalues_freq
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 16,
            u0: 0.5,
        },
        eigensolver_conv_thresh: 1e-7,
        eigensolver_max_vecs: 0,
        trunc_thresh: 0.0,
        run_diagnostics: false,
        frozen_core: 0,
        chi0_backend: Chi0Backend::Dense,
        chi0_sparsity: Chi0Sparsity::Dense,
        eigensolver: Eigensolver::Davidson,
        sternheimer: SternheimerConfig::default(),
        memory_budget_bytes: None,
        need_inv_dielectric_freq: false, // run_u_gw forces this on (M9 gate)
        verbose: false,
    }
}

fn check_radical(r: &RadicalRef) {
    check_radical_basis(r, "cc-pvdz", "cc-pvdz-ri");
}

fn check_radical_basis(r: &RadicalRef, obs_name: &str, aux_name: &str) {
    let mol = Molecule::parse_xyz(r.xyz, 0, 2).expect("parse doublet xyz");
    let obs_bs = basis::bundled(obs_name).expect("obs basis");
    let aux_bs = basis::bundled(aux_name).expect("aux basis");
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("obs");
    let dfbs = PreparedBasis::new(&mol, &aux_bs).expect("aux");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("Schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        max_iter: 300,
        level_shift: r.level_shift,
        mom_after_iter: r.mom_after_iter,
        ..Default::default()
    };
    let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &cfg).expect("UHF");
    assert!(uhf.converged, "{}: UHF did not converge", r.name);
    // Guard: ferric must reach the SAME UHF solution PySCF used for the QP
    // reference, or the IP comparison is meaningless (see `e_uhf_ref` doc).
    assert!(
        (uhf.energy - r.e_uhf_ref).abs() < 5e-3,
        "{}: UHF E {:.6} Ha vs ferric ground-state {:.6} Ha (Δ {:.6}) — ferric \
         converged a different SCF solution; the QP IP below would be off the \
         same wrong reference (the wrong-basin miss for NH₂ is ~0.084 Ha, far \
         outside this 5e-3 band)",
        r.name,
        uhf.energy,
        r.e_uhf_ref,
        uhf.energy - r.e_uhf_ref
    );

    let pdep = pdep_cfg();
    let two_s = (mol.multiplicity as i32) - 1;
    let nocc_a = ((mol.nelec() + two_s) / 2) as usize;
    let nocc_b = ((mol.nelec() - two_s) / 2) as usize;
    let homo_a = nocc_a - 1;
    let homo_b = nocc_b - 1;

    let gcfg = GwConfig {
        method: GwMethod::G0W0,
        ..Default::default()
    };
    let res = run_u_gw(&mol, &obs, &dfbs, op, &uhf, &pdep, &gcfg).expect("U-G0W0");

    let idx_a = res
        .mo_indices
        .iter()
        .position(|&i| i == homo_a)
        .expect("HOMO_α in QP range");
    let idx_b = res
        .mo_indices
        .iter()
        .position(|&i| i == homo_b)
        .expect("HOMO_β in QP range");

    let ip_a = -res.eps_qp_a[idx_a] * HA_TO_EV;
    let ip_b = -res.eps_qp_b[idx_b] * HA_TO_EV;

    println!(
        "{} U-G0W0@UHF/{obs_name}:\n  α-HOMO IP {ip_a:.3} eV (PySCF {:.3}, Δ {:+.3})\n  β-HOMO IP {ip_b:.3} eV (PySCF {:.3}, Δ {:+.3})",
        r.name,
        r.ip_a_ref,
        ip_a - r.ip_a_ref,
        r.ip_b_ref,
        ip_b - r.ip_b_ref,
    );

    assert!(
        res.qp_converged_a[idx_a] && res.qp_converged_b[idx_b],
        "{}: QP Newton solve did not converge",
        r.name
    );
    assert!(
        (ip_a - r.ip_a_ref).abs() < TOL_EV,
        "{}: α-HOMO IP {ip_a:.3} eV vs PySCF ugw_ac {:.3} eV (Δ {:.3}, tol {TOL_EV})",
        r.name,
        r.ip_a_ref,
        ip_a - r.ip_a_ref
    );
    assert!(
        (ip_b - r.ip_b_ref).abs() < TOL_EV,
        "{}: β-HOMO IP {ip_b:.3} eV vs PySCF ugw_ac {:.3} eV (Δ {:.3}, tol {TOL_EV})",
        r.name,
        r.ip_b_ref,
        ip_b - r.ip_b_ref
    );
}

#[test]
#[ignore = "slow: UHF + U-PDEP-RPA + U-G0W0; run with --release --ignored"]
fn oh_u_g0w0_matches_pyscf() {
    check_radical(&RadicalRef {
        name: "OH",
        xyz: "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.9697\n",
        ip_a_ref: 13.5686,
        ip_b_ref: 12.5320,
        e_uhf_ref: -75.393846,
        level_shift: 0.0,
        mom_after_iter: 0,
    });
}

#[test]
#[ignore = "slow: UHF + U-PDEP-RPA + U-G0W0; run with --release --ignored"]
fn ch3_u_g0w0_matches_pyscf() {
    // Planar D3h methyl radical, C–H = 1.079 Å.
    check_radical(&RadicalRef {
        name: "CH3",
        xyz: "4\nCH3\n\
C   0.000000   0.000000   0.000000\n\
H   1.079000   0.000000   0.000000\n\
H  -0.539500   0.934441   0.000000\n\
H  -0.539500  -0.934441   0.000000\n",
        ip_a_ref: 9.7541,
        ip_b_ref: 15.0065,
        e_uhf_ref: -39.563807,
        level_shift: 0.0,
        mom_after_iter: 0,
    });
}

#[test]
#[ignore = "slow: UHF + U-PDEP-RPA + U-G0W0; run with --release --ignored"]
fn nh2_u_g0w0_matches_pyscf() {
    // Bent amino radical, N–H = 1.024 Å, ∠HNH = 103.4°.
    //
    // NH₂'s two highest α occupied MOs are near-degenerate (Koopmans −13.73 and
    // −13.50 eV, 0.23 eV apart), so plain DIIS from the default guess lands an
    // excited SCF solution (−55.4834 Ha, ~0.084 Ha above ground state), which
    // then feeds a QP IP ~2.3 eV too small. A virtual-block level shift + MOM
    // (0.5, 5) steers ferric into the physical ground state (−55.5671 Ha,
    // matching PySCF's), where the QP IP agrees with the reference. The
    // `e_uhf_ref` guard in `check_radical` fails loudly if this ever regresses.
    check_radical(&RadicalRef {
        name: "NH2",
        xyz: "3\nNH2\n\
N   0.000000   0.000000   0.000000\n\
H   0.803611   0.000000  -0.634654\n\
H  -0.803611   0.000000  -0.634654\n",
        ip_a_ref: 12.7417,
        ip_b_ref: 11.7493,
        e_uhf_ref: -55.567091,
        level_shift: 0.5,
        mom_after_iter: 5,
    });
}

// --- Second basis (widens past cc-pVDZ-only): aug-cc-pVTZ, same 3 radicals,
// same PySCF ugw_ac methodology (scripts/gw100/pyscf_u_g0w0.py fed the
// bundled aug-cc-pvtz.json/aug-cc-pvtz-rifit.json orbital+aux JSONs).

#[test]
#[ignore = "slow: UHF + U-PDEP-RPA + U-G0W0 at aTZ; run with --release --ignored"]
fn oh_u_g0w0_matches_pyscf_augccpvtz() {
    // Unlike cc-pVDZ, plain DIIS from the default guess lands OH/aug-cc-pVTZ
    // in a WRONG SCF solution (measured: E=-75.266604 Ha vs the true ground
    // state -75.421646 Ha, a 0.155 Ha miss) -- the larger, more diffuse aTZ
    // virtual space apparently opens a near-degenerate trap that cc-pVDZ
    // doesn't have. The SAME level-shift+MOM steering NH2/cc-pVDZ already
    // needed (see nh2_u_g0w0_matches_pyscf's comment above) fixes it here
    // too: verified directly via the CLI (level_shift=0.5, mom_after_iter=5)
    // reaches -75.4216436711 Ha, matching PySCF to <1e-9 Ha. CH3/aTZ (below)
    // does NOT need steering -- verified separately, converges cleanly.
    check_radical_basis(
        &RadicalRef {
            name: "OH",
            xyz: "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.9697\n",
            ip_a_ref: 14.1926,
            ip_b_ref: 13.1796,
            e_uhf_ref: -75.421646,
            level_shift: 0.5,
            mom_after_iter: 5,
        },
        "aug-cc-pvtz",
        "aug-cc-pvtz-rifit",
    );
}

#[test]
#[ignore = "slow: UHF + U-PDEP-RPA + U-G0W0 at aTZ; run with --release --ignored"]
fn ch3_u_g0w0_matches_pyscf_augccpvtz() {
    check_radical_basis(
        &RadicalRef {
            name: "CH3",
            xyz: "4\nCH3\n\
C   0.000000   0.000000   0.000000\n\
H   1.079000   0.000000   0.000000\n\
H  -0.539500   0.934441   0.000000\n\
H  -0.539500  -0.934441   0.000000\n",
            ip_a_ref: 10.1013,
            ip_b_ref: 15.3715,
            e_uhf_ref: -39.577993,
            level_shift: 0.0,
            mom_after_iter: 0,
        },
        "aug-cc-pvtz",
        "aug-cc-pvtz-rifit",
    );
}

#[test]
#[ignore = "slow: UHF + U-PDEP-RPA + U-G0W0 at aTZ; run with --release --ignored"]
fn nh2_u_g0w0_matches_pyscf_augccpvtz() {
    // Same near-degenerate-orbital wrong-basin hazard as the cc-pVDZ case
    // (see that test's comment); same level-shift+MOM steering needed.
    check_radical_basis(
        &RadicalRef {
            name: "NH2",
            xyz: "3\nNH2\n\
N   0.000000   0.000000   0.000000\n\
H   0.803611   0.000000  -0.634654\n\
H  -0.803611   0.000000  -0.634654\n",
            ip_a_ref: 13.2748,
            ip_b_ref: 12.3083,
            e_uhf_ref: -55.588066,
            level_shift: 0.5,
            mom_after_iter: 5,
        },
        "aug-cc-pvtz",
        "aug-cc-pvtz-rifit",
    );
}
