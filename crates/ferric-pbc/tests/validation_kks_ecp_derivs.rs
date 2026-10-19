//! Weekly validation (every test `#[ignore = "validation: ..."]`) of the
//! k-point Kohn-Sham (RKS, UKS, ROKS) and ROHF analytic FORCES and STRESS on
//! a cell whose basis carries an ECP (issue #353). The cheap per-commit
//! subset is `pbc_kks_ecp_derivs.rs`; the shared harness is
//! `kks_ecp_support/mod.rs`.
//!
//! System: HI (closed shell, 8 valence electrons) and the H2I doublet (5α/4β
//! valence) in diag(6, 6, 7) Bohr, H STO-3G + compact I s/p, LANL2DZ ECP on
//! I (the fixtures of `pbc_kpoint_ecp.rs`), Gamma-centred 1x1x3 mesh
//! (NON-TRIM: `D(k)` is complex at two of the three points, so the Bloch
//! phases of the ECP image weight are exercised), atom-centred grid (20, 50)
//! with an explicit shared neighbour cutoff of 16 Bohr (`support::CUTOFF`,
//! whose doc carries the cutoff probe).
//!
//! WHY THE KS ROUTES NEED THEIR OWN EVIDENCE. Every k-point KS / ROHF force
//! and stress is the k HF assembly (`kgrad::assemble` / `kstress::assemble`,
//! which carries the ECP walk on the TOTAL density `D_α + D_β`) plus XC
//! terms. The ECP itself has no XC coupling, so by construction the ECP part
//! is route-independent; what can still be wrong is the COMPOSITION: the
//! Pulay `W` built from a Fock rebuilt without `V_ECP` (the guard rebuilds
//! `F = hk.h + ...` with `hk.h ⊃ V_ECP`), a spin-resolved weight that drops
//! one spin from the ECP walk, a restricted-open `W_RO` that treats the
//! ECP-carrying `h` differently, or an XC grid that sees the core electrons.
//!
//! ARTIFACT vs PHYSICS hypothesis (written before measuring). PHYSICS (the
//! composition is right): the analytic force and `dE/dε` reproduce the
//! central FD of ferric's own k energy (a fresh SCF per displaced / strained
//! cell; the FD never calls a derivative routine) at the FD floor measured
//! for the all-electron KS anchors (force ~1e-10, stress ~1e-9 at h =
//! 2.5e-5). ARTIFACT (a composition defect): the miss is of the size of the
//! ECP part itself (MEASURED |F_ECP| 6.3e-2 HI, 1.6e-1 H2I; |dE/dε_ECP|
//! 1.5e-1 HI, 3.4e-1..3.9e-1 H2I), i.e. eight orders above the floor; a
//! Fock without `V_ECP` is refused by the guard instead. The two outcomes are
//! separated by eight decades, so the FD anchor distinguishes them.
//!
//! PROBE (release, quiet-ish box, `pbc_kks_ecp_derivs.rs::probe_fd_steps`):
//! |analytic − FD| for F(I,x), dE/dε_zz, dE/dε_xy at h = 1e-4 / 5e-5 / 2.5e-5:
//! RKS LDA 2.7e-10/5.3e-8/1.7e-9, 2.2e-10/1.3e-8/3.3e-10, 1.5e-10/3.4e-9/1.2e-10;
//! RKS PBE 3.0e-10/5.9e-8/2.0e-9, 1.9e-10/1.5e-8/4.0e-10, 1.1e-10/3.4e-9/1.5e-10;
//! RKS PBE0 2.6e-10/4.9e-8/1.5e-9, 4.6e-11/1.2e-8/3.5e-10, 2.4e-10/2.7e-9/2.1e-10;
//! UKS PBE 8.7e-10/1.0e-6/5.2e-9, 2.5e-10/9.5e-9/1.2e-9, 1.2e-12/1.6e-9/6.0e-10;
//! ROKS PBE 8.4e-10/1.0e-6/5.3e-9, 2.7e-10/9.6e-9/1.2e-9, 3.1e-10/1.8e-9/6.9e-10;
//! ROHF 9.2e-11/6.2e-9/4.3e-10, 7.5e-11/1.6e-9/1.5e-10, 7.5e-11/8.5e-10/3.2e-11.
//! The zz strain follows h² (ratio ~4 per halving) down to ~2e-9 at 2.5e-5;
//! the open-shell 1e-6 at h = 1e-4 is a neighbour-mask jump (not h²), gone
//! at 5e-5. Steps: forces h = 2.5e-5 (`FD_H`); strains h = 1.25e-5
//! (`STRAIN_H`, [`strain_fd`]): at 2.5e-5 some of the nine strains straddle a
//! neighbour-mask jump at every cutoff probed (see `support::CUTOFF`).
//!
//! Bars (derived from both sides): `FORCE_BAR` 3e-9 is ~7x the worst force
//! miss measured below (4.2e-10) and ~2e7 below |F_ECP|; `STRESS_BAR` 3e-8 is
//! ~14x the worst strain miss (2.1e-9) and ~5e6 below |dE/dε_ECP|; every
//! ECP mutant must miss the same FD reference by more than
//! `MUTANT_FLOOR_FACTOR` (1e3) x the bar.
//!
//! LEDGER (release, 2 test threads x 3 rayon threads, 950 s for the file;
//! every strain at h = 1.25e-5; the 2.5e-5 retry then in the file never
//! triggered and has since been removed). |analytic − FD|, max over all force
//! components / the listed strains:
//!
//! | route | J/K | force | strains | stress |
//! |---|---|---|---|---|---|
//! | RKS LDA | dense | 2.4e-10 | 9 | 1.0e-9 |
//! | RKS PBE | dense | 1.9e-10 | 9 | 1.3e-9 |
//! | RKS PBE0 | dense | 3.2e-10 | 3 | 5.5e-10 |
//! | RKS SCAN | dense | 2.5e-10 | 3 | 3.2e-10 |
//! | RKS r2SCAN | dense | 2.8e-10 | 3 | 1.1e-9 |
//! | RKS PBE | RS-GDF | 2.6e-10 | 3 | 8.4e-10 |
//! | RKS PBE0 | RS-GDF | 2.1e-10 | 3 | 7.9e-10 |
//! | UKS LDA | dense | 3.8e-10 | 3 | 9.2e-10 |
//! | UKS PBE | dense | 2.8e-10 | 9 | 2.1e-9 |
//! | UKS PBE0 | dense | 3.1e-10 | 3 | 7.7e-10 |
//! | UKS PBE0 | RS-GDF | 3.3e-10 | 3 | 4.8e-10 |
//! | ROKS PBE | dense | 2.3e-10 | 3 | 7.2e-10 |
//! | ROKS PBE0 | dense | 2.6e-10 | 3 | 5.4e-10 |
//! | ROKS PBE | RS-GDF | 4.1e-10 | 3 | 5.8e-10 |
//! | ROHF | dense | 1.7e-10 | 9 | 1.7e-9 |
//! | ROHF | RS-GDF | 4.2e-10 | 3 | 8.0e-10 |
//!
//! Earlier runs of the same anchors (cutoff 14, single step 2.5e-5) agree on
//! every force and subset strain; their all-nine failures (RKS LDA 5.8e-7,
//! RKS PBE 6.0e-7, UKS PBE 5.0e-8, then UKS PBE 7.9e-6 at cutoff 16) were
//! mask jumps, identified by the h / cutoff probe.
//!
//! ECP mutants (production-wired, `cfg.grad.mutation` / `cfg.stress.mutation`)
//! against the same FD reference, miss:
//!
//! | mutant | RKS PBE dense | RKS PBE RS-GDF | UKS PBE | ROHF |
//! |---|---|---|---|
//! | force NoEcp | 6.3e-2 | 6.3e-2 | 1.6e-1 | 1.5e-1 |
//! | force EcpCentreSign | 1.3 | 1.3 | 1.75 | 2.3 |
//! | force EcpCentreWrongAtom | 6.5e-1 | 6.5e-1 | 8.8e-1 | 1.2 |
//! | force EcpWrongResidue | 1.0e-2 | 1.0e-2 | 1.9e-2 | 7.9e-3 |
//! | force EcpM0Only | 3.1e-3 | 3.1e-3 | 1.4e-2 | 8.2e-3 |
//! | stress NoEcp | 1.5e-1 | 1.5e-1 | 3.9e-1 | 3.4e-1 |
//! | stress EcpNoCentre | 2.3 | 2.3 | 2.9 | 3.9 |
//! | stress EcpCentreSign | 4.6 | 4.6 | 5.7 | 7.7 |
//! | stress EcpNoKetImage | 1.2e-1 | 1.2e-1 | 3.7e-2 | 4.6e-2 |
//! | stress EcpWrongResidue | 4.2e-2 | 4.2e-2 | 2.6e-2 | 2.1e-2 |
//!
//! The ECP mutants act on the ECP walk only, so dense and RS-GDF agree to the
//! printed digits (the two SCF densities differ by the fit error). The RS-GDF
//! column is from a solo run of `rks_rsgdf_force_and_stress_match_fd`.
//! Smallest: force EcpM0Only 3.1e-3 (1e6 x `FORCE_BAR`), stress
//! EcpWrongResidue 2.1e-2 (7e5 x `STRESS_BAR`). (EcpWrongResidue is an
//! identity on TRIM meshes; the 1x1x3 mesh makes it visible.)
//!
//! NOT covered: a triclinic ECP cell or a shifted mesh with KS (the ECP
//! weights' shifted-mesh / triclinic handling is anchored on the HF routes in
//! `pbc_kpoint_ecp.rs` / `pbc_kpoint_ecp_rotation.rs` and is route-
//! independent code), d/f shells on the ECP element, more than one ECP atom,
//! an external (PySCF) reference, B3LYP (the global-hybrid path is exercised
//! by PBE0), UKS / ROKS with SCAN (refused: no open-shell meta-GGA force),
//! range-split RS-GDF (refused for every k-point stress and for the ROKS /
//! ROHF force), the 1x1x1-vs-Gamma and supercell identities (the HF ECP
//! anchors in `pbc_kpoint_ecp.rs` cover the shared ECP walk; here FD of the
//! energy is the independent construction).

mod common;
#[path = "kks_ecp_support/mod.rs"]
mod support;

use ferric_pbc::kgrad::KGradMutation;
use ferric_pbc::kstress::KStressMutation;
use support::*;

const FD_H: f64 = 2.5e-5;
const STRAIN_H: f64 = 1.25e-5;
const FORCE_BAR: f64 = 3e-9;
const STRESS_BAR: f64 = 3e-8;
const MUTANT_FLOOR_FACTOR: f64 = 1e3;

const FORCE_MUTANTS: [KGradMutation; 5] = [
    KGradMutation::NoEcp,
    KGradMutation::EcpCentreSign,
    KGradMutation::EcpCentreWrongAtom,
    KGradMutation::EcpWrongResidue,
    KGradMutation::EcpM0Only,
];
const STRESS_MUTANTS: [KStressMutation; 5] = [
    KStressMutation::NoEcp,
    KStressMutation::EcpNoCentre,
    KStressMutation::EcpCentreSign,
    KStressMutation::EcpNoKetImage,
    KStressMutation::EcpWrongResidue,
];

/// Strain FD at `STRAIN_H` = 1.25e-5. The atom-grid energy has one-step
/// jumps (1e-11..1e-9 Ha) where a neighbour atom crosses the hard cutoff; a
/// step that straddles one turns into a 1e-7..1e-5 FD error that is NOT h^2
/// (MEASURED in `pbc_kks_ecp_derivs.rs::probe_all_nine_strains`: H2I UKS PBE,
/// cutoff 14, eps_xx 9.4e-10 / 2.1e-8 / 8.9e-6 at h = 1.25e-5 / 2.5e-5 /
/// 5e-5; cutoff 16, eps_yy 2.1e-9 / 7.9e-6 / 4.0e-6). Every probed component
/// is <= 2.1e-9 at 1.25e-5 (RKS and UKS, cutoffs 14 and 16). No retry at a
/// larger step: a jump inside ±h is also inside ±2h, so a retry would almost
/// never rescue a straddled jump; a straddle fails the anchor instead (the
/// safe direction) and is diagnosed with the probe.
fn strain_fd(cell: &ferric_pbc::lattice::Cell, spec: &Spec, i: usize, j: usize) -> f64 {
    fd_strain(cell, spec, i, j, STRAIN_H)
}

/// What one anchor covers.
struct Case {
    route: Route,
    jk: Jk,
    pos: &'static [[f64; 3]],
    /// Strain components differenced (all nine or a subset).
    strains: &'static [(usize, usize)],
    /// Run the ECP mutant sweep against the FD reference.
    mutants: bool,
}

const SUBSET: [(usize, usize); 3] = [(2, 2), (0, 1), (0, 2)];

/// FD of every force component and the listed strains against the analytic
/// values; with `mutants`, every ECP mutant must miss the same FD reference.
fn anchor(c: &Case) {
    let spec = Spec::new(c.route, c.jk, [1, 1, 3]);
    let t0 = std::time::Instant::now();
    let fx = fixture(c.pos, &spec);
    let (g, g_ecp) = force(&fx, None);
    let (s, s_ecp) = stress(&fx, None);
    let na = c.pos.len();
    let mut fd_g = ndarray::Array2::<f64>::zeros((na, 3));
    for a in 0..na {
        for x in 0..3 {
            fd_g[(a, x)] = fd_force(c.pos, &spec, a, x, FD_H);
        }
    }
    let mut fd_s = s;
    for &(i, j) in c.strains {
        fd_s[i][j] = strain_fd(&fx.cell, &spec, i, j);
    }
    let fmiss = amax(&(&g - &fd_g));
    let smiss = c
        .strains
        .iter()
        .fold(0.0_f64, |m, &(i, j)| m.max((s[i][j] - fd_s[i][j]).abs()));
    eprintln!(
        "{:?} {:?}: force |an - FD| {fmiss:.2e} (all {} comps; |F| {:.3e}, |F_ECP| {:.3e}); \
         stress |an - FD| {smiss:.2e} ({} comps; |dE/de| {:.3e}, ECP {:.3e}); {:.0} s",
        c.route,
        c.jk,
        3 * na,
        amax(&g),
        amax(&g_ecp),
        c.strains.len(),
        mmax(&s),
        mmax(&s_ecp),
        t0.elapsed().as_secs_f64()
    );
    assert!(
        amax(&g_ecp) > 1e-2 && mmax(&s_ecp) > 1e-2,
        "vacuous ECP part"
    );
    assert!(
        fmiss < FORCE_BAR,
        "{:?} {:?} force {fmiss:e}",
        c.route,
        c.jk
    );
    assert!(
        smiss < STRESS_BAR,
        "{:?} {:?} stress {smiss:e}",
        c.route,
        c.jk
    );
    if !c.mutants {
        return;
    }
    for m in FORCE_MUTANTS {
        let (gm, _) = force(&fx, Some(m));
        let d = amax(&(&gm - &fd_g));
        eprintln!("  force mutant {m:?}: miss {d:.2e}");
        assert!(
            d > MUTANT_FLOOR_FACTOR * FORCE_BAR,
            "{m:?} survives the FD anchor: {d:e}"
        );
    }
    for m in STRESS_MUTANTS {
        let (sm, _) = stress(&fx, Some(m));
        let d = c.strains.iter().fold(0.0_f64, |acc, &(i, j)| {
            acc.max((sm[i][j] - fd_s[i][j]).abs())
        });
        eprintln!("  stress mutant {m:?}: miss {d:.2e}");
        assert!(
            d > MUTANT_FLOOR_FACTOR * STRESS_BAR,
            "{m:?} survives the FD anchor: {d:e}"
        );
    }
}

#[test]
#[ignore = "validation: HI 1x1x3 k-RKS LDA dense, force (6) + stress (9) FD; ~1.5 min release"]
fn rks_lda_dense_force_and_stress_match_fd() {
    anchor(&Case {
        route: Route::Rks("LDA"),
        jk: Jk::Dense,
        pos: &HI_POS,
        strains: &ALL9,
        mutants: false,
    });
}

#[test]
#[ignore = "validation: HI 1x1x3 k-RKS PBE dense, FD + 10 ECP mutants; ~3 min release"]
fn rks_pbe_dense_force_and_stress_match_fd_and_reject_ecp_mutants() {
    anchor(&Case {
        route: Route::Rks("PBE"),
        jk: Jk::Dense,
        pos: &HI_POS,
        strains: &ALL9,
        mutants: true,
    });
}

#[test]
#[ignore = "validation: HI 1x1x3 k-RKS PBE0 (exxdiv ewald) dense, FD; ~2 min release"]
fn rks_pbe0_dense_force_and_stress_match_fd() {
    anchor(&Case {
        route: Route::Rks("PBE0"),
        jk: Jk::Dense,
        pos: &HI_POS,
        strains: &SUBSET,
        mutants: false,
    });
}

#[test]
#[ignore = "validation: HI 1x1x3 k-RKS SCAN and r2SCAN dense, FD; ~3 min release"]
fn rks_meta_gga_dense_force_and_stress_match_fd() {
    for f in ["SCAN", "r2SCAN"] {
        anchor(&Case {
            route: Route::Rks(f),
            jk: Jk::Dense,
            pos: &HI_POS,
            strains: &SUBSET,
            mutants: false,
        });
    }
}

#[test]
#[ignore = "validation: HI 1x1x3 k-RKS PBE and PBE0 on RS-GDF, FD + ECP mutants (PBE); ~4 min release"]
fn rks_rsgdf_force_and_stress_match_fd() {
    anchor(&Case {
        route: Route::Rks("PBE"),
        jk: Jk::Gdf,
        pos: &HI_POS,
        strains: &SUBSET,
        mutants: true,
    });
    anchor(&Case {
        route: Route::Rks("PBE0"),
        jk: Jk::Gdf,
        pos: &HI_POS,
        strains: &SUBSET,
        mutants: false,
    });
}

#[test]
#[ignore = "validation: H2I doublet 1x1x3 k-UKS PBE dense, force (9) + stress (9) FD + ECP mutants; ~4 min release"]
fn uks_pbe_dense_force_and_stress_match_fd_and_reject_ecp_mutants() {
    anchor(&Case {
        route: Route::Uks("PBE"),
        jk: Jk::Dense,
        pos: &H2I_POS,
        strains: &ALL9,
        mutants: true,
    });
}

#[test]
#[ignore = "validation: H2I doublet 1x1x3 k-UKS LDA and PBE0 dense, PBE0 RS-GDF, FD; ~5 min release"]
fn uks_lda_pbe0_and_rsgdf_force_and_stress_match_fd() {
    for (f, jk) in [("LDA", Jk::Dense), ("PBE0", Jk::Dense), ("PBE0", Jk::Gdf)] {
        anchor(&Case {
            route: Route::Uks(f),
            jk,
            pos: &H2I_POS,
            strains: &SUBSET,
            mutants: false,
        });
    }
}

#[test]
#[ignore = "validation: H2I doublet 1x1x3 k-ROKS PBE / PBE0 dense, PBE RS-GDF, FD; ~5 min release"]
fn roks_force_and_stress_match_fd() {
    for (f, jk) in [("PBE", Jk::Dense), ("PBE0", Jk::Dense), ("PBE", Jk::Gdf)] {
        anchor(&Case {
            route: Route::Roks(f),
            jk,
            pos: &H2I_POS,
            strains: &SUBSET,
            mutants: false,
        });
    }
}

#[test]
#[ignore = "validation: H2I doublet 1x1x3 k-ROHF dense (all nine strains, ECP mutants) and RS-GDF, FD; ~6 min release"]
fn rohf_force_and_stress_match_fd_and_reject_ecp_mutants() {
    anchor(&Case {
        route: Route::Rohf,
        jk: Jk::Dense,
        pos: &H2I_POS,
        strains: &ALL9,
        mutants: true,
    });
    anchor(&Case {
        route: Route::Rohf,
        jk: Jk::Gdf,
        pos: &H2I_POS,
        strains: &SUBSET,
        mutants: false,
    });
}
